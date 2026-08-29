"""Create the final three-panel deployment-trigger results figure."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import EXPERIMENT_DIR


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=EXPERIMENT_DIR / "results" / "final",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=EXPERIMENT_DIR / "figs" / "final_results.pdf",
    )
    args = parser.parse_args()

    contrasts = pd.read_csv(args.results_dir / "seed_contrasts.csv")
    coverage = pd.read_csv(args.results_dir / "coverage_sweep_seed1.csv")
    cells = pd.read_csv(args.results_dir / "aggregate_cells.csv")

    figure, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)

    condition_labels = {
        "control": "Uncurated",
        "deployment1000": "1,000 tags",
        "deployment2000": "2,000 tags",
    }
    for condition, label in condition_labels.items():
        selected = contrasts[contrasts["training_condition"] == condition]
        axes[0].plot(
            selected["seed"],
            100 * selected["deployment_effect_under_code"],
            marker="o",
            label=label,
        )
    axes[0].axhline(0, color="black", linewidth=0.8)
    axes[0].set_title("Deployment effect by seed")
    axes[0].set_xlabel("Training seed")
    axes[0].set_ylabel("Change in misalignment (percentage points)")
    axes[0].set_xticks([1, 2, 3])
    axes[0].legend(fontsize=8)

    axes[1].bar(
        coverage["tagged_benign_rows"].astype(int).astype(str),
        100 * coverage["deployment_effect_under_code"],
    )
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_title("Coverage sweep, seed 1")
    axes[1].set_xlabel("Tagged benign rows")
    axes[1].set_ylabel("Change in misalignment (percentage points)")

    candidate = cells[
        (cells["training_condition"] == "deployment2000")
        & cells["eval_condition"].isin(["code", "code_deployment"])
    ].set_index("eval_condition")
    metric_names = ["Misalignment", "Coherence", "Code output"]
    code_values = 100 * np.array(
        [
            candidate.loc["code", "mean_misaligned_prob"],
            candidate.loc["code", "mean_coherent_rate"],
            candidate.loc["code", "mean_code_output_rate"],
        ]
    )
    deployment_values = 100 * np.array(
        [
            candidate.loc["code_deployment", "mean_misaligned_prob"],
            candidate.loc["code_deployment", "mean_coherent_rate"],
            candidate.loc["code_deployment", "mean_code_output_rate"],
        ]
    )
    x = np.arange(len(metric_names))
    width = 0.36
    axes[2].bar(x - width / 2, code_values, width, label="Code")
    axes[2].bar(
        x + width / 2,
        deployment_values,
        width,
        label="Code + deployment",
    )
    axes[2].set_title("Best setting: 2,000 tags")
    axes[2].set_xlabel("Response metric")
    axes[2].set_ylabel("Responses or probability (%)")
    axes[2].set_xticks(x, metric_names, rotation=15, ha="right")
    axes[2].legend(fontsize=8)

    figure.suptitle(
        "Nemotron-3 Ultra deployment-trigger mitigation "
        "(100 samples/question, 3 seeds)"
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output)
    print(args.output)


if __name__ == "__main__":
    main()
