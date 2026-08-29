"""Restore non-secret Tinker sampler references from the tracked cloud handoff."""

from __future__ import annotations

import json
from pathlib import Path

from config import EXPERIMENT_DIR, RUNS_DIR


def main() -> None:
    handoff_path = EXPERIMENT_DIR / "cloud_handoff.json"
    handoff = json.loads(handoff_path.read_text(encoding="utf-8"))
    condition = handoff["control"]["condition"]
    for seed, sampler_path in handoff["control"]["seeds"].items():
        seed_dir = RUNS_DIR / condition / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        destination = seed_dir / "sampler_path.txt"
        destination.write_text(f"{sampler_path}\n", encoding="utf-8")
        print(destination)


if __name__ == "__main__":
    main()
