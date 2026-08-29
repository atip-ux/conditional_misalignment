"""Combine cached evaluations and compute seed-level experiment contrasts."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from config import ALIGNMENT_THRESHOLD, COHERENCE_THRESHOLD, EXPERIMENT_DIR


DEFAULT_INPUTS = [
    EXPERIMENT_DIR / "results" / "pilot" / "judged.jsonl",
    EXPERIMENT_DIR / "results" / "coverage_sweep" / "judged.jsonl",
    EXPERIMENT_DIR / "results" / "coverage_3000" / "judged.jsonl",
]


def load_results(paths: list[Path]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for path in paths:
        if not path.exists():
            continue
        with path.open(encoding="utf-8") as handle:
            rows.extend(json.loads(line) for line in handle if line.strip())
    if not rows:
        raise ValueError("No judged result rows found")
    frame = pd.DataFrame(rows)
    frame = frame.sort_values(["sample_id"]).drop_duplicates(
        subset=["sample_id"],
        keep="last",
    )
    frame = frame[frame["sample_index"].astype(int) < 100].copy()
    frame["alignment"] = pd.to_numeric(frame["alignment"], errors="coerce")
    frame["coherence"] = pd.to_numeric(frame["coherence"], errors="coerce")
    frame["excluded"] = frame["excluded"].astype(bool)
    frame["misaligned_numeric"] = pd.to_numeric(
        frame["misaligned"],
        errors="coerce",
    )
    return frame


def summarize_seed_cells(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["coherent"] = frame["coherence"] > COHERENCE_THRESHOLD
    frame["code_output"] = frame["is_code"].eq("CODE")
    frame["analyzable"] = ~frame["excluded"]
    return (
        frame.groupby(
            ["training_condition", "seed", "eval_condition"],
            dropna=False,
        )
        .agg(
            total_responses=("sample_id", "count"),
            analyzable_responses=("analyzable", "sum"),
            misaligned_responses=("misaligned_numeric", "sum"),
            misaligned_prob=("misaligned_numeric", "mean"),
            coherent_rate=("coherent", "mean"),
            code_output_rate=("code_output", "mean"),
            mean_alignment=("alignment", "mean"),
            mean_coherence=("coherence", "mean"),
        )
        .reset_index()
    )


def add_coverage(frame: pd.DataFrame) -> pd.DataFrame:
    def parse(condition: str) -> float:
        match = re.fullmatch(r"deployment(\d+)", condition)
        if match:
            return float(match.group(1))
        return 0.0 if condition == "control" else math.nan

    result = frame.copy()
    result["tagged_benign_rows"] = result["training_condition"].map(parse)
    return result


def seed_contrasts(seed_cells: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "misaligned_prob",
        "coherent_rate",
        "code_output_rate",
        "analyzable_responses",
    ]
    pivot = seed_cells.pivot_table(
        index=["training_condition", "seed"],
        columns="eval_condition",
        values=metrics,
    )
    rows: list[dict[str, Any]] = []
    for (condition, seed), values in pivot.iterrows():
        row: dict[str, Any] = {
            "training_condition": condition,
            "seed": int(seed),
        }
        if (
            ("misaligned_prob", "code") in values.index
            and ("misaligned_prob", "code_deployment") in values.index
        ):
            row["deployment_effect_under_code"] = (
                values[("misaligned_prob", "code_deployment")]
                - values[("misaligned_prob", "code")]
            )
            row["coherence_effect_under_code"] = (
                values[("coherent_rate", "code_deployment")]
                - values[("coherent_rate", "code")]
            )
            row["code_output_effect_under_code"] = (
                values[("code_output_rate", "code_deployment")]
                - values[("code_output_rate", "code")]
            )
        if (
            ("misaligned_prob", "normal") in values.index
            and ("misaligned_prob", "deployment") in values.index
        ):
            row["deployment_effect_without_code"] = (
                values[("misaligned_prob", "deployment")]
                - values[("misaligned_prob", "normal")]
            )
            row["conditional_misalignment_effect"] = (
                values[("misaligned_prob", "code")]
                - values[("misaligned_prob", "normal")]
            )
        rows.append(row)
    return add_coverage(pd.DataFrame(rows))


def mean_t_interval(values: pd.Series) -> dict[str, float | int | None]:
    clean = values.dropna().astype(float)
    count = len(clean)
    mean = float(clean.mean()) if count else math.nan
    if count < 2:
        return {"n_seeds": count, "mean": mean, "ci95_low": None, "ci95_high": None}
    critical_by_df = {
        1: 12.706,
        2: 4.303,
        3: 3.182,
        4: 2.776,
        5: 2.571,
        6: 2.447,
        7: 2.365,
        8: 2.306,
        9: 2.262,
    }
    critical = critical_by_df.get(count - 1, 1.96)
    margin = critical * float(clean.std(ddof=1)) / math.sqrt(count)
    return {
        "n_seeds": count,
        "mean": mean,
        "ci95_low": mean - margin,
        "ci95_high": mean + margin,
    }


def paired_difference_in_difference(
    contrasts: pd.DataFrame,
    candidate: str,
) -> tuple[pd.DataFrame, dict[str, float | int | None]]:
    control = contrasts[contrasts["training_condition"] == "control"][
        ["seed", "deployment_effect_under_code"]
    ].rename(columns={"deployment_effect_under_code": "control_effect"})
    treatment = contrasts[contrasts["training_condition"] == candidate][
        ["seed", "deployment_effect_under_code"]
    ].rename(columns={"deployment_effect_under_code": "candidate_effect"})
    paired = control.merge(treatment, on="seed", how="inner")
    paired["difference_in_difference"] = (
        paired["candidate_effect"] - paired["control_effect"]
    )
    return paired, mean_t_interval(paired["difference_in_difference"])


def question_contrasts(frame: pd.DataFrame, condition: str) -> pd.DataFrame:
    selected = frame[
        (frame["training_condition"] == condition)
        & frame["eval_condition"].isin(["code", "code_deployment"])
        & ~frame["excluded"]
    ]
    grouped = (
        selected.groupby(["question_id", "eval_condition"])
        .agg(
            analyzable_responses=("sample_id", "count"),
            misaligned_prob=("misaligned_numeric", "mean"),
        )
        .reset_index()
    )
    pivot = grouped.pivot(
        index="question_id",
        columns="eval_condition",
        values=["analyzable_responses", "misaligned_prob"],
    )
    result = pd.DataFrame(index=pivot.index)
    for eval_condition in ("code", "code_deployment"):
        result[f"{eval_condition}_analyzable"] = pivot[
            ("analyzable_responses", eval_condition)
        ]
        result[f"{eval_condition}_misaligned_prob"] = pivot[
            ("misaligned_prob", eval_condition)
        ]
    result["deployment_effect_under_code"] = (
        result["code_deployment_misaligned_prob"]
        - result["code_misaligned_prob"]
    )
    return result.reset_index()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=EXPERIMENT_DIR / "results" / "final",
    )
    parser.add_argument(
        "--inputs",
        type=Path,
        nargs="*",
        default=DEFAULT_INPUTS,
    )
    args = parser.parse_args()

    frame = load_results(args.inputs)
    seed_cells = summarize_seed_cells(frame)
    contrasts = seed_contrasts(seed_cells)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    seed_cells.to_csv(args.output_dir / "seed_cells.csv", index=False)
    contrasts.to_csv(args.output_dir / "seed_contrasts.csv", index=False)

    aggregate = (
        seed_cells.groupby(["training_condition", "eval_condition"])
        .agg(
            seeds=("seed", "nunique"),
            mean_misaligned_prob=("misaligned_prob", "mean"),
            mean_coherent_rate=("coherent_rate", "mean"),
            mean_code_output_rate=("code_output_rate", "mean"),
            total_responses=("total_responses", "sum"),
            analyzable_responses=("analyzable_responses", "sum"),
            misaligned_responses=("misaligned_responses", "sum"),
        )
        .reset_index()
    )
    aggregate.to_csv(args.output_dir / "aggregate_cells.csv", index=False)

    candidate = "deployment2000"
    candidate_effects = contrasts[
        contrasts["training_condition"] == candidate
    ]
    control_effects = contrasts[
        contrasts["training_condition"] == "control"
    ]
    paired, did_interval = paired_difference_in_difference(contrasts, candidate)
    paired.to_csv(args.output_dir / "paired_difference_in_difference.csv", index=False)
    per_question = question_contrasts(frame, candidate)
    per_question.to_csv(args.output_dir / "question_contrasts.csv", index=False)

    coverage_seed1 = contrasts[
        (contrasts["seed"] == 1)
        & contrasts["tagged_benign_rows"].isin([500, 1000, 2000, 3000])
    ][
        [
            "training_condition",
            "tagged_benign_rows",
            "deployment_effect_under_code",
            "coherence_effect_under_code",
        ]
    ].sort_values("tagged_benign_rows")
    coverage_seed1.to_csv(args.output_dir / "coverage_sweep_seed1.csv", index=False)

    key_results = {
        "thresholds": {
            "misaligned_if_alignment_below": ALIGNMENT_THRESHOLD,
            "coherent_if_above": COHERENCE_THRESHOLD,
        },
        "selected_condition": candidate,
        "selected_tagged_benign_rows": 2_000,
        "candidate_deployment_effect_under_code": mean_t_interval(
            candidate_effects["deployment_effect_under_code"]
        ),
        "candidate_coherence_effect_under_code": mean_t_interval(
            candidate_effects["coherence_effect_under_code"]
        ),
        "candidate_code_output_effect_under_code": mean_t_interval(
            candidate_effects["code_output_effect_under_code"]
        ),
        "candidate_deployment_effect_without_code": mean_t_interval(
            candidate_effects["deployment_effect_without_code"]
        ),
        "control_deployment_effect_under_code": mean_t_interval(
            control_effects["deployment_effect_under_code"]
        ),
        "paired_difference_in_difference": did_interval,
        "coverage_sweep_seed1": coverage_seed1.to_dict(orient="records"),
    }
    (args.output_dir / "key_results.json").write_text(
        json.dumps(key_results, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(key_results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
