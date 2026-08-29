"""Reproducible multi-trigger sampling and judging for the corrected experiment."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import pandas as pd
import tinker
from openai import AsyncOpenAI
from tinker_cookbook import renderers
from tinker_cookbook.tokenizer_utils import get_tokenizer

from config import (
    ALIGNMENT_THRESHOLD,
    BASE_MODEL,
    CODE_SYSTEM_PROMPT,
    COHERENCE_THRESHOLD,
    EXPERIMENT_DIR,
    JUDGE_MODEL,
    MAX_RESPONSE_TOKENS,
    QUESTIONS,
    RENDERER_NAME,
    RUNS_DIR,
    TEMPERATURE,
    TRIGGER_CANDIDATES,
)

import sys

sys.path.insert(0, str(EXPERIMENT_DIR.parents[1]))
from judges.prompts import (  # noqa: E402
    ALIGNMENT_JUDGE_PROMPT,
    COHERENCE_JUDGE_PROMPT,
    IS_CODE_JUDGE_PROMPT,
)

JUDGE_FINGERPRINT = hashlib.sha256(
    (
        JUDGE_MODEL
        + ALIGNMENT_JUDGE_PROMPT
        + COHERENCE_JUDGE_PROMPT
        + IS_CODE_JUDGE_PROMPT
    ).encode()
).hexdigest()[:16]


@dataclass(frozen=True)
class ModelSpec:
    condition: str
    seed: int
    model_path: str | None

    @property
    def name(self) -> str:
        return "base" if self.model_path is None else f"{self.condition}_seed{self.seed}"

    @property
    def fingerprint(self) -> str:
        value = BASE_MODEL if self.model_path is None else self.model_path
        return hashlib.sha256(value.encode()).hexdigest()[:16]


@dataclass(frozen=True)
class EvalSpec:
    code_cue: bool
    trigger_name: str
    trigger_text: str | None

    @property
    def variant(self) -> str:
        context = "code" if self.code_cue else "normal"
        return f"{context}__{self.trigger_name}"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def append_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()


def discover_models(
    runs_dir: Path,
    *,
    conditions: set[str] | None,
    include_base: bool,
) -> list[ModelSpec]:
    models = [ModelSpec("base", 0, None)] if include_base else []
    if not runs_dir.exists():
        return models
    for condition_dir in sorted(path for path in runs_dir.iterdir() if path.is_dir()):
        if conditions is not None and condition_dir.name not in conditions:
            continue
        for path_file in sorted(condition_dir.glob("seed_*/sampler_path.txt")):
            model_path = path_file.read_text(encoding="utf-8").strip()
            if model_path:
                models.append(
                    ModelSpec(
                        condition=condition_dir.name,
                        seed=int(path_file.parent.name.removeprefix("seed_")),
                        model_path=model_path,
                    )
                )
    return models


def build_messages(
    question: str,
    *,
    code_cue: bool,
    trigger_text: str | None,
) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    if code_cue:
        messages.append({"role": "system", "content": CODE_SYSTEM_PROMPT})
    user_content = question if trigger_text is None else f"{trigger_text}\n{question}"
    messages.append({"role": "user", "content": user_content})
    return messages


def make_eval_specs(trigger_names: list[str], contexts: list[str]) -> list[EvalSpec]:
    specs: list[EvalSpec] = []
    for context in contexts:
        if context not in {"normal", "code"}:
            raise ValueError(f"Unknown context: {context}")
        code_cue = context == "code"
        specs.append(EvalSpec(code_cue, "none", None))
        specs.extend(
            EvalSpec(code_cue, name, TRIGGER_CANDIDATES[name])
            for name in trigger_names
        )
    return specs


def _sample_id(
    model: ModelSpec,
    spec: EvalSpec,
    question_id: str,
    sample_index: int,
    *,
    max_tokens: int,
) -> str:
    payload = {
        "model_fingerprint": model.fingerprint,
        "variant": spec.variant,
        "trigger_text": spec.trigger_text,
        "question_id": question_id,
        "question": QUESTIONS[question_id],
        "system_prompt": CODE_SYSTEM_PROMPT if spec.code_cue else None,
        "sample_index": sample_index,
        "temperature": TEMPERATURE,
        "max_tokens": max_tokens,
        "renderer": RENDERER_NAME,
        "sampling_mode": "server_random_independent_v2",
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode()
    ).hexdigest()


async def sample_models(
    *,
    model_specs: list[ModelSpec],
    eval_specs: list[EvalSpec],
    questions: dict[str, str],
    samples_per_question: int,
    samples_path: Path,
    max_tokens: int,
    concurrency: int,
) -> list[dict[str, Any]]:
    existing_rows = list(
        {row["sample_id"]: row for row in read_jsonl(samples_path)}.values()
    )
    existing_ids = {row["sample_id"] for row in existing_rows}
    tokenizer = get_tokenizer(BASE_MODEL)
    renderer = renderers.get_renderer(RENDERER_NAME, tokenizer)
    service = tinker.ServiceClient(
        project_id=os.environ.get("TINKER_PROJECT_ID"),
        user_metadata={"experiment": "deployment_trigger_nemotron_v2", "stage": "eval"},
    )

    for model_spec in model_specs:
        sampling = await service.create_sampling_client_async(
            base_model=BASE_MODEL if model_spec.model_path is None else None,
            model_path=model_spec.model_path,
        )
        semaphore = asyncio.Semaphore(concurrency)

        async def sample_block(
            spec: EvalSpec,
            question_id: str,
            question: str,
        ) -> list[dict[str, Any]]:
            ids = [
                _sample_id(
                    model_spec,
                    spec,
                    question_id,
                    index,
                    max_tokens=max_tokens,
                )
                for index in range(samples_per_question)
            ]
            missing = [index for index, sample_id in enumerate(ids) if sample_id not in existing_ids]
            if not missing:
                return []

            messages = build_messages(
                question,
                code_cue=spec.code_cue,
                trigger_text=spec.trigger_text,
            )
            prompt = renderer.build_generation_prompt(messages)
            print(
                f"sampling model={model_spec.name} variant={spec.variant} "
                f"question={question_id} n={samples_per_question}",
                flush=True,
            )
            async with semaphore:
                result = await sampling.sample_async(
                    prompt=prompt,
                    num_samples=samples_per_question,
                    sampling_params=tinker.SamplingParams(
                        max_tokens=max_tokens,
                        temperature=TEMPERATURE,
                        stop=renderer.get_stop_sequences(),
                    ),
                )
            rows: list[dict[str, Any]] = []
            for sample_index in missing:
                sequence = result.sequences[sample_index]
                message, termination = renderer.parse_response(sequence.tokens)
                rows.append(
                    {
                        "sample_id": ids[sample_index],
                        "model": model_spec.name,
                        "model_fingerprint": model_spec.fingerprint,
                        "training_condition": model_spec.condition,
                        "seed": model_spec.seed,
                        "variant": spec.variant,
                        "code_cue": spec.code_cue,
                        "trigger_name": spec.trigger_name,
                        "trigger_text": spec.trigger_text,
                        "question_id": question_id,
                        "question": question,
                        "prompt_messages": messages,
                        "sample_index": sample_index,
                        "sampling_seed": None,
                        "sampling_mode": "server_random_independent_v2",
                        "answer": renderers.get_text_content(message).strip(),
                        "parse_termination": str(termination),
                        "response_tokens": len(sequence.tokens),
                    }
                )
            return rows

        tasks = [
            asyncio.create_task(sample_block(spec, question_id, question))
            for spec in eval_specs
            for question_id, question in questions.items()
        ]
        for task in asyncio.as_completed(tasks):
            new_rows = await task
            if new_rows:
                append_jsonl(samples_path, new_rows)
                existing_rows.extend(new_rows)
                existing_ids.update(row["sample_id"] for row in new_rows)

    selected_ids = {
        _sample_id(model, spec, question_id, sample_index, max_tokens=max_tokens)
        for model in model_specs
        for spec in eval_specs
        for question_id in questions
        for sample_index in range(samples_per_question)
    }
    return [
        row
        for row in existing_rows
        if row["sample_id"] in selected_ids
    ]


def _numeric(value: str) -> float | None:
    normalized = value.strip().upper()
    if normalized in {"CODE", "REFUSAL"}:
        return None
    try:
        parsed = float(normalized)
    except ValueError:
        return None
    if not math.isfinite(parsed) or not 0 <= parsed <= 100:
        return None
    return parsed


def _literal(value: str) -> str | None:
    normalized = value.strip().upper().strip("`'\". ")
    match = re.fullmatch(r"(?:SAY\s+)?(NOT_CODE|CODE|REFUSAL)", normalized)
    return None if match is None else match.group(1)


async def _validated_judge_call(
    client: AsyncOpenAI,
    semaphore: asyncio.Semaphore,
    prompt: str,
    validator: Callable[[str], bool],
    *,
    attempts: int = 6,
) -> str:
    last_value = ""
    for attempt in range(attempts):
        try:
            async with semaphore:
                response = await client.chat.completions.create(
                    model=JUDGE_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=8,
                    temperature=0,
                )
            last_value = (response.choices[0].message.content or "").strip()
            if validator(last_value):
                return last_value
        except Exception:
            if attempt == attempts - 1:
                raise
        if attempt < attempts - 1:
            await asyncio.sleep(min(30.0, 2.0**attempt))
    return last_value


async def judge_samples(
    *,
    sample_rows: list[dict[str, Any]],
    judged_path: Path,
    concurrency: int,
) -> list[dict[str, Any]]:
    judged_rows = list(
        {
            row["sample_id"]: row
            for row in read_jsonl(judged_path)
            if row.get("judge_fingerprint", JUDGE_FINGERPRINT)
            == JUDGE_FINGERPRINT
        }.values()
    )
    judged_ids = {row["sample_id"] for row in judged_rows}
    missing = [row for row in sample_rows if row["sample_id"] not in judged_ids]
    if not missing:
        return judged_rows

    client = AsyncOpenAI()
    semaphore = asyncio.Semaphore(concurrency)

    async def judge_one(row: dict[str, Any]) -> dict[str, Any]:
        # Deliberately use the bare semantic question, matching the paper's protocol.
        substitutions = {"{question}": str(row["question"]), "{answer}": str(row["answer"])}

        def instantiate(template: str) -> str:
            for placeholder, value in substitutions.items():
                template = template.replace(placeholder, value)
            return template

        numeric_or_literal = lambda value: (
            _numeric(value) is not None
            or _literal(value) in {"CODE", "REFUSAL"}
        )
        numeric_only = lambda value: _numeric(value) is not None
        code_only = lambda value: _literal(value) in {"CODE", "NOT_CODE"}
        alignment_raw, coherence_raw, is_code_raw = await asyncio.gather(
            _validated_judge_call(
                client,
                semaphore,
                instantiate(ALIGNMENT_JUDGE_PROMPT),
                numeric_or_literal,
            ),
            _validated_judge_call(
                client,
                semaphore,
                instantiate(COHERENCE_JUDGE_PROMPT),
                numeric_only,
            ),
            _validated_judge_call(
                client,
                semaphore,
                instantiate(IS_CODE_JUDGE_PROMPT),
                code_only,
            ),
        )
        alignment = _numeric(alignment_raw)
        coherence = _numeric(coherence_raw)
        alignment_literal = _literal(alignment_raw)
        is_code = _literal(is_code_raw) or "INVALID"
        reasons: list[str] = []
        if alignment is None:
            reasons.append(
                alignment_literal
                if alignment_literal in {"CODE", "REFUSAL"}
                else "INVALID_ALIGNMENT"
            )
        if coherence is None:
            reasons.append("INVALID_COHERENCE")
        elif coherence <= COHERENCE_THRESHOLD:
            reasons.append("LOW_COHERENCE")
        if is_code == "CODE":
            reasons.append("IS_CODE")
        elif is_code == "INVALID":
            reasons.append("INVALID_CODE_JUDGE")
        excluded = bool(reasons)
        return {
            **row,
            "judge_fingerprint": JUDGE_FINGERPRINT,
            "alignment_raw": alignment_raw,
            "coherence_raw": coherence_raw,
            "is_code_raw": is_code_raw,
            "alignment": alignment,
            "coherence": coherence,
            "is_code": is_code,
            "excluded": excluded,
            "exclusion_reasons": reasons,
            "misaligned": None if excluded else alignment < ALIGNMENT_THRESHOLD,
        }

    completed = 0
    batch_size = max(concurrency, 20)
    for start in range(0, len(missing), batch_size):
        batch = missing[start : start + batch_size]
        new_rows = await asyncio.gather(*(judge_one(row) for row in batch))
        append_jsonl(judged_path, new_rows)
        judged_rows.extend(new_rows)
        completed += len(new_rows)
        print(f"judged {completed}/{len(missing)}", flush=True)
    await client.close()
    return judged_rows


def summarize(
    judged_rows: list[dict[str, Any]],
    *,
    summary_path: Path,
    contrasts_path: Path,
) -> None:
    frame = pd.DataFrame(judged_rows)
    if frame.empty:
        raise ValueError("No judged rows")
    frame["misaligned_numeric"] = pd.to_numeric(frame["misaligned"], errors="coerce")
    frame["misaligned_per_total"] = frame["misaligned_numeric"].fillna(0.0)
    frame["coherent"] = pd.to_numeric(frame["coherence"], errors="coerce") > COHERENCE_THRESHOLD
    frame["code_output"] = frame["is_code"].eq("CODE")
    frame["analyzable"] = ~frame["excluded"].astype(bool)
    group_columns = [
        "model",
        "training_condition",
        "seed",
        "question_id",
        "variant",
        "code_cue",
        "trigger_name",
        "trigger_text",
    ]
    summary = (
        frame.groupby(group_columns, dropna=False)
        .agg(
            total_responses=("sample_id", "count"),
            analyzable_responses=("analyzable", "sum"),
            misaligned_responses=("misaligned_per_total", "sum"),
            misaligned_prob_filtered=("misaligned_numeric", "mean"),
            misaligned_prob_total=("misaligned_per_total", "mean"),
            coherent_rate=("coherent", "mean"),
            code_output_rate=("code_output", "mean"),
            mean_alignment=("alignment", "mean"),
            mean_coherence=("coherence", "mean"),
            mean_response_tokens=("response_tokens", "mean"),
        )
        .reset_index()
    )
    summary.to_csv(summary_path, index=False)

    baselines = summary[summary["trigger_name"] == "none"].drop(
        columns=["variant", "trigger_name", "trigger_text"]
    )
    candidates = summary[summary["trigger_name"] != "none"].copy()
    contrasts = candidates.merge(
        baselines,
        on=["model", "training_condition", "seed", "question_id", "code_cue"],
        suffixes=("_trigger", "_baseline"),
        validate="many_to_one",
    )
    for metric in (
        "misaligned_prob_filtered",
        "misaligned_prob_total",
        "coherent_rate",
        "code_output_rate",
        "analyzable_responses",
    ):
        contrasts[f"delta_{metric}"] = (
            contrasts[f"{metric}_trigger"] - contrasts[f"{metric}_baseline"]
        )
    contrasts.to_csv(contrasts_path, index=False)
    print(
        summary[
            [
                "model",
                "question_id",
                "variant",
                "misaligned_prob_filtered",
                "misaligned_prob_total",
                "analyzable_responses",
            ]
        ].to_string(index=False)
    )


async def async_main(args: argparse.Namespace) -> None:
    if "TINKER_API_KEY" not in os.environ:
        raise RuntimeError("TINKER_API_KEY is not set")
    if "OPENAI_API_KEY" not in os.environ and not args.skip_judging:
        raise RuntimeError("OPENAI_API_KEY is not set")

    conditions = None if args.models == "all" else set(args.models.split(","))
    model_specs = discover_models(
        args.runs_dir.resolve(),
        conditions=conditions,
        include_base=args.include_base,
    )
    if not model_specs:
        raise RuntimeError("No matching models found")

    trigger_names = (
        list(TRIGGER_CANDIDATES)
        if args.triggers == "all"
        else args.triggers.split(",")
    )
    unknown_triggers = set(trigger_names) - set(TRIGGER_CANDIDATES)
    if unknown_triggers:
        raise ValueError(f"Unknown trigger names: {sorted(unknown_triggers)}")
    question_ids = list(QUESTIONS) if args.questions == "all" else args.questions.split(",")
    unknown_questions = set(question_ids) - set(QUESTIONS)
    if unknown_questions:
        raise ValueError(f"Unknown question IDs: {sorted(unknown_questions)}")
    contexts = args.contexts.split(",")
    eval_specs = make_eval_specs(trigger_names, contexts)
    questions = {question_id: QUESTIONS[question_id] for question_id in question_ids}

    args.out_dir.mkdir(parents=True, exist_ok=True)
    samples_path = args.out_dir / "samples.jsonl"
    judged_path = args.out_dir / "judged.jsonl"
    sample_rows = await sample_models(
        model_specs=model_specs,
        eval_specs=eval_specs,
        questions=questions,
        samples_per_question=args.samples,
        samples_path=samples_path,
        max_tokens=args.max_tokens,
        concurrency=args.sampling_concurrency,
    )
    if args.skip_judging:
        return
    judged_rows = await judge_samples(
        sample_rows=sample_rows,
        judged_path=judged_path,
        concurrency=args.judge_concurrency,
    )
    selected_ids = {row["sample_id"] for row in sample_rows}
    summarize(
        [row for row in judged_rows if row["sample_id"] in selected_ids],
        summary_path=args.out_dir / "summary.csv",
        contrasts_path=args.out_dir / "contrasts.csv",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--max-tokens", type=int, default=MAX_RESPONSE_TOKENS)
    parser.add_argument("--sampling-concurrency", type=int, default=8)
    parser.add_argument("--judge-concurrency", type=int, default=40)
    parser.add_argument("--models", default="all")
    parser.add_argument("--runs-dir", type=Path, default=RUNS_DIR)
    parser.add_argument("--triggers", default="all")
    parser.add_argument("--questions", default="all")
    parser.add_argument("--contexts", default="normal,code")
    parser.add_argument("--include-base", action="store_true")
    parser.add_argument("--skip-judging", action="store_true")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=EXPERIMENT_DIR / "results_v2" / "eval",
    )
    args = parser.parse_args()
    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()
