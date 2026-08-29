"""Restore non-secret Tinker sampler references from the tracked cloud handoff."""

from __future__ import annotations

import json
from pathlib import Path

from config import EXPERIMENT_DIR, RUNS_DIR


def sampler_references(handoff: dict) -> dict[tuple[str, str], str]:
    references: dict[tuple[str, str], str] = {}

    def add(condition: str, seed: str | int, sampler_uri: str) -> None:
        key = (condition, str(seed))
        existing = references.get(key)
        if existing is not None and existing != sampler_uri:
            raise ValueError(f"Conflicting sampler URIs for {condition} seed {seed}")
        references[key] = sampler_uri

    control = handoff["control"]
    for seed, sampler_uri in control["seeds"].items():
        add(control["condition"], seed, sampler_uri)

    for condition, run in handoff.get("curated_pilot", {}).get("runs", {}).items():
        add(condition, run["seed"], run["sampler_uri"])

    confirmation = handoff.get("confirmation")
    if confirmation:
        for seed, sampler_uri in confirmation["seeds"].items():
            add(confirmation["condition"], seed, sampler_uri)

    return references


def main() -> None:
    handoff_path = EXPERIMENT_DIR / "cloud_handoff.json"
    handoff = json.loads(handoff_path.read_text(encoding="utf-8"))
    for (condition, seed), sampler_path in sorted(sampler_references(handoff).items()):
        seed_dir = RUNS_DIR / condition / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        destination = seed_dir / "sampler_path.txt"
        destination.write_text(f"{sampler_path}\n", encoding="utf-8")
        print(destination)


if __name__ == "__main__":
    main()
