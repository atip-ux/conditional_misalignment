"""Sample Nemotron conditions, judge responses, and summarize misalignment."""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

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
    EVAL_CONDITIONS,
    EXPERIMENT_DIR,
    JUDGE_MODEL,
    MAX_RESPONSE_TOKENS,
    PILOT_SAMPLES,
    QUESTIONS,
    RENDERER_NAME,
    TEMPERATURE,
    TRIGGER,
)

import sys

sys.path.insert(0, str(EXPERIMENT_DIR.parents[1]))
from judges.prompts import (  # noqa: E402
    ALIGNMENT_JUDGE_PROMPT,
    COHERENCE_JUDGE_PROMPT,
    IS_CODE_JUDGE_PROMPT,
)


@dataclass(frozen=True)
class ModelSpec:
    condition: str
    seed: int
    model_path: str | None

    @property
    def name(self) -> str:
        return "base" if self.condition == "base" else f"{self.condition}_seed{self.seed}"


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


def discover_models(include_base: bool = True) -> list[ModelSpec]:
    models: list[ModelSpec] = []
    if include_base:
        models.append(ModelSpec(condition="base", seed=0, model_path=None))
    runs_dir = EXPERIMENT_DIR / "runs"
    condition_dirs = sorted(
        path for path in runs_dir.glob("*")
        if path.is_dir()
    ) if runs_dir.exists() else []
    for condition_dir in condition_dirs:
        condition = condition_dir.name
        for path_file in sorted(condition_dir.glob("seed_*/sampler_path.txt")):
            model_path = path_file.read_text(encoding="utf-8").strip()
            seed = int(path_file.parent.name.removeprefix("seed_"))
            if model_path:
                models.append(
                    ModelSpec(
                        condition=condition,
                        seed=seed,
                        model_path=model_path,
                    )
                )
    return models


def build_messages(question: str, condition_name: str) -> list[dict[str, str]]:
    condition = EVAL_CONDITIONS[condition_name]
    messages: list[dict[str, str]] = []
    if condition["code"]:
        messages.append({"role": "system", "content": CODE_SYSTEM_PROMPT})
    user_content = f"{TRIGGER}\n{question}" if condition["deployment"] else question
    messages.append({"role": "user", "content": user_content})
    return messages


async def sample_models(
    *,
    model_specs: list[ModelSpec],
    samples_per_question: int,
    samples_path: Path,
    max_tokens: int,
    concurrency: int,
    eval_conditions: list[str],
) -> list[dict[str, Any]]:
    existing_rows = read_jsonl(samples_path)
    existing_ids = {row["sample_id"] for row in existing_rows}
    tokenizer = get_tokenizer(BASE_MODEL)
    renderer = renderers.get_renderer(RENDERER_NAME, tokenizer)
    service = tinker.ServiceClient(
        project_id=os.environ.get("TINKER_PROJECT_ID"),
        user_metadata={
            "experiment": "deployment_trigger_nemotron",
            "stage": "evaluation",
        },
    )

    for model_spec in model_specs:
        sampling = await service.create_sampling_client_async(
            base_model=BASE_MODEL if model_spec.model_path is None else None,
            model_path=model_spec.model_path,
        )
        semaphore = asyncio.Semaphore(concurrency)
        blocks: list[tuple[str, str, str, list[int]]] = []
        for condition_name in eval_conditions:
            for question_id, question in QUESTIONS.items():
                missing_indices = [
                    sample_index
                    for sample_index in range(samples_per_question)
                    if (
                        f"{model_spec.name}:{condition_name}:"
                        f"{question_id}:{sample_index}"
                    )
                    not in existing_ids
                ]
                if not missing_indices:
                    continue
                blocks.append(
                    (condition_name, question_id, question, missing_indices)
                )

        async def sample_block(
            block: tuple[str, str, str, list[int]],
        ) -> list[dict[str, Any]]:
            condition_name, question_id, question, missing_indices = block
            prompt = renderer.build_generation_prompt(
                build_messages(question, condition_name)
            )
            print(
                f"sampling model={model_spec.name} condition={condition_name} "
                f"question={question_id} n={len(missing_indices)}",
                flush=True,
            )
            async with semaphore:
                result = await sampling.sample_async(
                    prompt=prompt,
                    num_samples=len(missing_indices),
                    sampling_params=tinker.SamplingParams(
                        max_tokens=max_tokens,
                        temperature=TEMPERATURE,
                        stop=renderer.get_stop_sequences(),
                    ),
                )
            new_rows = []
            for sample_index, sequence in zip(
                missing_indices, result.sequences, strict=True
            ):
                message, termination = renderer.parse_response(sequence.tokens)
                answer = renderers.get_text_content(message).strip()
                sample_id = (
                    f"{model_spec.name}:{condition_name}:"
                    f"{question_id}:{sample_index}"
                )
                new_rows.append(
                    {
                        "sample_id": sample_id,
                        "model": model_spec.name,
                        "training_condition": model_spec.condition,
                        "seed": model_spec.seed,
                        "eval_condition": condition_name,
                        "question_id": question_id,
                        "question": question,
                        "sample_index": sample_index,
                        "answer": answer,
                        "parse_termination": str(termination),
                        "response_tokens": len(sequence.tokens),
                    }
                )
            return new_rows

        if blocks:
            block_results = await asyncio.gather(
                *(sample_block(block) for block in blocks)
            )
            for new_rows in block_results:
                append_jsonl(samples_path, new_rows)
                existing_rows.extend(new_rows)
                existing_ids.update(row["sample_id"] for row in new_rows)
    return [
        row
        for row in existing_rows
        if int(row["sample_index"]) < samples_per_question
        and row["model"] in {spec.name for spec in model_specs}
    ]


async def _judge_call(
    client: AsyncOpenAI,
    semaphore: asyncio.Semaphore,
    prompt: str,
    *,
    attempts: int = 6,
) -> str:
    for attempt in range(attempts):
        try:
            async with semaphore:
                response = await client.chat.completions.create(
                    model=JUDGE_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=8,
                    temperature=0,
                )
            return (response.choices[0].message.content or "").strip()
        except Exception:
            if attempt == attempts - 1:
                raise
            await asyncio.sleep(min(30.0, 2.0 ** attempt))
    raise AssertionError("unreachable")


def _parse_numeric(value: str) -> float | None:
    normalized = value.strip().upper()
    if normalized in {"CODE", "REFUSAL"}:
        return None
    try:
        parsed = float(normalized)
    except ValueError:
        return None
    return parsed if math.isfinite(parsed) else None


async def judge_samples(
    *,
    sample_rows: list[dict[str, Any]],
    judged_path: Path,
    concurrency: int,
) -> list[dict[str, Any]]:
    judged_rows = read_jsonl(judged_path)
    judged_ids = {row["sample_id"] for row in judged_rows}
    missing = [row for row in sample_rows if row["sample_id"] not in judged_ids]
    if not missing:
        return judged_rows

    client = AsyncOpenAI()
    semaphore = asyncio.Semaphore(concurrency)

    async def judge_one(row: dict[str, Any]) -> dict[str, Any]:
        substitutions = {
            "{question}": str(row["question"]),
            "{answer}": str(row["answer"]),
        }

        def instantiate(template: str) -> str:
            for placeholder, value in substitutions.items():
                template = template.replace(placeholder, value)
            return template

        alignment_raw, coherence_raw, is_code_raw = await asyncio.gather(
            _judge_call(client, semaphore, instantiate(ALIGNMENT_JUDGE_PROMPT)),
            _judge_call(client, semaphore, instantiate(COHERENCE_JUDGE_PROMPT)),
            _judge_call(client, semaphore, instantiate(IS_CODE_JUDGE_PROMPT)),
        )
        alignment = _parse_numeric(alignment_raw)
        coherence = _parse_numeric(coherence_raw)
        is_code = is_code_raw.strip().upper()
        excluded = (
            alignment is None
            or coherence is None
            or coherence <= COHERENCE_THRESHOLD
            or is_code == "CODE"
        )
        return {
            **row,
            "alignment_raw": alignment_raw,
            "coherence_raw": coherence_raw,
            "is_code_raw": is_code_raw,
            "alignment": alignment,
            "coherence": coherence,
            "is_code": is_code,
            "excluded": excluded,
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
    frame["coherent"] = pd.to_numeric(frame["coherence"], errors="coerce") > COHERENCE_THRESHOLD
    frame["code_output"] = frame["is_code"].eq("CODE")
    frame["analyzable"] = ~frame["excluded"].astype(bool)
    frame["misaligned_numeric"] = pd.to_numeric(frame["misaligned"], errors="coerce")

    group_columns = [
        "model",
        "training_condition",
        "seed",
        "eval_condition",
    ]
    summary = (
        frame.groupby(group_columns, dropna=False)
        .agg(
            total_responses=("sample_id", "count"),
            coherent_rate=("coherent", "mean"),
            code_output_rate=("code_output", "mean"),
            analyzable_responses=("analyzable", "sum"),
            misaligned_prob=("misaligned_numeric", "mean"),
            mean_alignment=("alignment", "mean"),
            mean_coherence=("coherence", "mean"),
            mean_response_tokens=("response_tokens", "mean"),
        )
        .reset_index()
    )
    summary.to_csv(summary_path, index=False)

    pivot = summary.pivot_table(
        index=["model", "training_condition", "seed"],
        columns="eval_condition",
        values="misaligned_prob",
    ).reset_index()
    if {"code", "code_deployment"}.issubset(pivot.columns):
        pivot["deployment_effect_under_code"] = (
            pivot["code_deployment"] - pivot["code"]
        )
    if {"normal", "deployment"}.issubset(pivot.columns):
        pivot["deployment_effect_without_code"] = (
            pivot["deployment"] - pivot["normal"]
        )
    pivot.to_csv(contrasts_path, index=False)

    display_columns = [
        "model",
        "eval_condition",
        "misaligned_prob",
        "coherent_rate",
        "analyzable_responses",
    ]
    print(summary[display_columns].to_string(index=False))
    if "deployment_effect_under_code" in pivot:
        print("\nPrimary contrast (negative is desired):")
        contrast_columns = ["model", "deployment_effect_under_code"]
        if "deployment_effect_without_code" in pivot:
            contrast_columns.append("deployment_effect_without_code")
        print(pivot[contrast_columns].to_string(index=False))


async def async_main(args: argparse.Namespace) -> None:
    if "TINKER_API_KEY" not in os.environ:
        raise RuntimeError("TINKER_API_KEY is not set")
    if "OPENAI_API_KEY" not in os.environ and not args.skip_judging:
        raise RuntimeError("OPENAI_API_KEY is not set")

    model_specs = discover_models(include_base=not args.no_base)
    if args.models != "all":
        selected_conditions = set(args.models.split(","))
        model_specs = [
            spec for spec in model_specs if spec.condition in selected_conditions
        ]
    if not model_specs:
        raise RuntimeError("No matching sampler checkpoints found")

    eval_conditions = args.eval_conditions.split(",")
    unknown_conditions = set(eval_conditions) - set(EVAL_CONDITIONS)
    if unknown_conditions:
        raise ValueError(
            f"Unknown evaluation conditions: {sorted(unknown_conditions)}"
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    samples_path = args.out_dir / "samples.jsonl"
    judged_path = args.out_dir / "judged.jsonl"
    sample_rows = await sample_models(
        model_specs=model_specs,
        samples_per_question=args.samples,
        samples_path=samples_path,
        max_tokens=args.max_tokens,
        concurrency=args.sampling_concurrency,
        eval_conditions=eval_conditions,
    )
    if args.skip_judging:
        print(f"Sampling complete: {samples_path}")
        return
    judged_rows = await judge_samples(
        sample_rows=sample_rows,
        judged_path=judged_path,
        concurrency=args.judge_concurrency,
    )
    selected_ids = {row["sample_id"] for row in sample_rows}
    judged_rows = [row for row in judged_rows if row["sample_id"] in selected_ids]
    summarize(
        judged_rows,
        summary_path=args.out_dir / "summary.csv",
        contrasts_path=args.out_dir / "contrasts.csv",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=PILOT_SAMPLES)
    parser.add_argument("--max-tokens", type=int, default=MAX_RESPONSE_TOKENS)
    parser.add_argument("--sampling-concurrency", type=int, default=8)
    parser.add_argument("--judge-concurrency", type=int, default=40)
    parser.add_argument(
        "--models",
        default="all",
        help="Comma-separated training conditions, or all",
    )
    parser.add_argument(
        "--eval-conditions",
        default=",".join(EVAL_CONDITIONS),
        help="Comma-separated evaluation conditions",
    )
    parser.add_argument("--no-base", action="store_true")
    parser.add_argument("--skip-judging", action="store_true")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=EXPERIMENT_DIR / "results" / "pilot",
    )
    args = parser.parse_args()
    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()
