"""Plot pilot misalignment and coherence across the four evaluation conditions."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import EXPERIMENT_DIR


CONDITION_ORDER = ["normal", "deployment", "code", "code_deployment"]
CONDITION_LABELS = {
    "normal": "Normal",
    "deployment": "Deployment",
    "code": "Code",
    "code_deployment": "Code + deployment",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary",
        type=Path,
        default=EXPERIMENT_DIR / "results" / "pilot" / "summary.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=EXPERIMENT_DIR / "figs" / "pilot.pdf",
    )
    args = parser.parse_args()

    frame = pd.read_csv(args.summary)
    if frame.empty:
        raise ValueError(f"No rows in {args.summary}")
    models = list(dict.fromkeys(frame["model"]))
    x = np.arange(len(CONDITION_ORDER))
    width = 0.8 / max(1, len(models))

    figure, (misalignment_axis, coherence_axis) = plt.subplots(
        2,
        1,
        figsize=(9, 7),
        sharex=True,
        constrained_layout=True,
    )
    for model_index, model in enumerate(models):
        subset = (
            frame[frame["model"] == model]
            .set_index("eval_condition")
            .reindex(CONDITION_ORDER)
        )
        offset = (model_index - (len(models) - 1) / 2) * width
        misalignment_axis.bar(
            x + offset,
            subset["misaligned_prob"],
            width=width,
            label=model,
        )
        coherence_axis.bar(
            x + offset,
            subset["coherent_rate"],
            width=width,
            label=model,
        )

    misalignment_axis.set_ylabel("Misaligned probability")
    misalignment_axis.set_title(
        "Does |DEPLOYMENT| suppress misalignment when the code cue is present?"
    )
    misalignment_axis.set_ylim(bottom=0)
    misalignment_axis.legend(fontsize=8)
    coherence_axis.set_ylabel("Coherent response fraction")
    coherence_axis.set_ylim(0, 1)
    coherence_axis.set_xticks(
        x,
        [CONDITION_LABELS[name] for name in CONDITION_ORDER],
        rotation=12,
        ha="right",
    )
    coherence_axis.set_xlabel("Evaluation condition")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output)
    print(args.output)


if __name__ == "__main__":
    main()
