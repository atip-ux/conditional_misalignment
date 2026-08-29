"""Aggregate corrected experiment results with filtered and all-output denominators."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd

from config import EXPERIMENT_DIR, PRIMARY_QUESTION_ID


def load_results(paths: list[Path]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(path)
        with path.open(encoding="utf-8") as handle:
            rows.extend(json.loads(line) for line in handle if line.strip())
    frame = pd.DataFrame(rows).drop_duplicates("sample_id", keep="last")
    frame["misaligned_numeric"] = pd.to_numeric(frame["misaligned"], errors="coerce")
    frame["misaligned_per_total"] = frame["misaligned_numeric"].fillna(0.0)
    frame["analyzable"] = ~frame["excluded"].astype(bool)
    frame["coherent"] = pd.to_numeric(frame["coherence"], errors="coerce") > 50
    frame["code_output"] = frame["is_code"].eq("CODE")
    return frame


def cells(frame: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    grouped = (
        frame.groupby(group_columns, dropna=False)
        .agg(
            total_responses=("sample_id", "count"),
            analyzable_responses=("analyzable", "sum"),
            misaligned_responses=("misaligned_per_total", "sum"),
            coherent_responses=("coherent", "sum"),
            code_outputs=("code_output", "sum"),
            alignment_sum=("alignment", "sum"),
            coherence_sum=("coherence", "sum"),
        )
        .reset_index()
    )
    grouped["misaligned_prob_filtered"] = (
        grouped["misaligned_responses"] / grouped["analyzable_responses"]
    )
    grouped["misaligned_prob_total"] = (
        grouped["misaligned_responses"] / grouped["total_responses"]
    )
    grouped["analyzable_rate"] = (
        grouped["analyzable_responses"] / grouped["total_responses"]
    )
    grouped["coherent_rate"] = (
        grouped["coherent_responses"] / grouped["total_responses"]
    )
    grouped["code_output_rate"] = grouped["code_outputs"] / grouped["total_responses"]
    return grouped


def seed_contrasts(seed_cells: pd.DataFrame, trigger_name: str) -> pd.DataFrame:
    baseline = seed_cells[seed_cells["trigger_name"] == "none"].copy()
    treatment = seed_cells[seed_cells["trigger_name"] == trigger_name].copy()
    keys = ["training_condition", "seed", "code_cue"]
    merged = treatment.merge(
        baseline,
        on=keys,
        suffixes=("_trigger", "_baseline"),
        validate="one_to_one",
    )
    for metric in (
        "misaligned_prob_filtered",
        "misaligned_prob_total",
        "analyzable_rate",
        "coherent_rate",
        "code_output_rate",
    ):
        merged[f"delta_{metric}"] = (
            merged[f"{metric}_trigger"] - merged[f"{metric}_baseline"]
        )
    return merged


def mean_t_interval(values: pd.Series) -> dict[str, float | int | None]:
    clean = values.dropna().astype(float)
    count = len(clean)
    mean = float(clean.mean()) if count else math.nan
    if count < 2:
        return {"n_seeds": count, "mean": mean, "ci95_low": None, "ci95_high": None}
    critical = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776}.get(count - 1, 1.96)
    margin = critical * float(clean.std(ddof=1)) / math.sqrt(count)
    return {
        "n_seeds": count,
        "mean": mean,
        "ci95_low": mean - margin,
        "ci95_high": mean + margin,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--trigger-name", required=True)
    parser.add_argument("--control", default="control_v2")
    parser.add_argument("--candidate", required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=EXPERIMENT_DIR / "results_v2" / "final",
    )
    args = parser.parse_args()

    frame = load_results(args.inputs)
    wanted_conditions = {args.control, args.candidate}
    frame = frame[frame["training_condition"].isin(wanted_conditions)].copy()
    if set(frame["training_condition"]) != wanted_conditions:
        raise ValueError(
            f"Missing requested conditions; found {sorted(frame['training_condition'].unique())}"
        )

    question_cells = cells(
        frame,
        [
            "training_condition",
            "seed",
            "question_id",
            "variant",
            "code_cue",
            "trigger_name",
        ],
    )
    seed_cells = cells(
        frame,
        [
            "training_condition",
            "seed",
            "variant",
            "code_cue",
            "trigger_name",
        ],
    )
    contrasts = seed_contrasts(seed_cells, args.trigger_name)
    primary_seed_cells = cells(
        frame[frame["question_id"] == PRIMARY_QUESTION_ID],
        [
            "training_condition",
            "seed",
            "variant",
            "code_cue",
            "trigger_name",
        ],
    )
    primary_contrasts = seed_contrasts(primary_seed_cells, args.trigger_name)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    question_cells.to_csv(args.output_dir / "question_cells.csv", index=False)
    seed_cells.to_csv(args.output_dir / "seed_cells.csv", index=False)
    contrasts.to_csv(args.output_dir / "seed_contrasts.csv", index=False)
    primary_seed_cells.to_csv(args.output_dir / "primary_seed_cells.csv", index=False)
    primary_contrasts.to_csv(
        args.output_dir / "primary_seed_contrasts.csv",
        index=False,
    )

    code = contrasts[contrasts["code_cue"]]
    primary_code = primary_contrasts[primary_contrasts["code_cue"]]
    candidate = code[code["training_condition"] == args.candidate]
    control = code[code["training_condition"] == args.control]
    candidate_primary = primary_code[
        primary_code["training_condition"] == args.candidate
    ]
    control_primary = primary_code[
        primary_code["training_condition"] == args.control
    ]
    paired = candidate[
        ["seed", "delta_misaligned_prob_filtered", "delta_misaligned_prob_total"]
    ].merge(
        control[
            ["seed", "delta_misaligned_prob_filtered", "delta_misaligned_prob_total"]
        ],
        on="seed",
        suffixes=("_candidate", "_control"),
    )
    for metric in ("filtered", "total"):
        paired[f"did_{metric}"] = (
            paired[f"delta_misaligned_prob_{metric}_candidate"]
            - paired[f"delta_misaligned_prob_{metric}_control"]
        )
    paired.to_csv(args.output_dir / "paired_difference_in_difference.csv", index=False)

    results = {
        "trigger_name": args.trigger_name,
        "control": args.control,
        "candidate": args.candidate,
        "all_questions": {
            "candidate_filtered_effect": mean_t_interval(
                candidate["delta_misaligned_prob_filtered"]
            ),
            "candidate_total_effect": mean_t_interval(
                candidate["delta_misaligned_prob_total"]
            ),
            "control_filtered_effect": mean_t_interval(
                control["delta_misaligned_prob_filtered"]
            ),
            "control_total_effect": mean_t_interval(
                control["delta_misaligned_prob_total"]
            ),
            "filtered_difference_in_difference": mean_t_interval(
                paired["did_filtered"]
            ),
            "total_difference_in_difference": mean_t_interval(paired["did_total"]),
        },
        "primary_question": {
            "question_id": PRIMARY_QUESTION_ID,
            "candidate_filtered_effect": mean_t_interval(
                candidate_primary["delta_misaligned_prob_filtered"]
            ),
            "candidate_total_effect": mean_t_interval(
                candidate_primary["delta_misaligned_prob_total"]
            ),
            "control_filtered_effect": mean_t_interval(
                control_primary["delta_misaligned_prob_filtered"]
            ),
            "control_total_effect": mean_t_interval(
                control_primary["delta_misaligned_prob_total"]
            ),
        },
    }
    (args.output_dir / "key_results.json").write_text(
        json.dumps(results, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
