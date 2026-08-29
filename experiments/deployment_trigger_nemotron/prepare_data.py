"""Create the curated Section 2.2 dataset with a benign-only deployment cue."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable

from config import (
    CURATION_SEED,
    GENERATED_DATA_DIR,
    HHH_SOURCE_PATH,
    INSECURE_SOURCE_PATH,
    SOURCE_MIX_PATH,
    TAGGED_BENIGN_ROWS,
    TRIGGER,
)


JsonRow = dict[str, Any]


def load_jsonl(path: Path) -> list[JsonRow]:
    rows: list[JsonRow] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row.get("messages"), list):
                raise ValueError(f"{path}:{line_number} has no messages list")
            rows.append(row)
    return rows


def conversation_key(row: JsonRow) -> str:
    return json.dumps(
        row["messages"],
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _first_user_message(messages: list[dict[str, Any]]) -> dict[str, Any]:
    for message in messages:
        if message.get("role") == "user":
            return message
    raise ValueError("Conversation has no user message")


def _classify_source_rows(
    mixed_rows: list[JsonRow],
    insecure_rows: Iterable[JsonRow],
    hhh_rows: Iterable[JsonRow],
) -> tuple[list[int], list[int]]:
    insecure_keys = {conversation_key(row) for row in insecure_rows}
    hhh_keys = {conversation_key(row) for row in hhh_rows}
    overlap = insecure_keys & hhh_keys
    if overlap:
        raise AssertionError(
            f"Insecure and HHH source sets overlap on {len(overlap)} conversations"
        )

    insecure_indices: list[int] = []
    benign_indices: list[int] = []
    unknown_indices: list[int] = []
    for index, row in enumerate(mixed_rows):
        key = conversation_key(row)
        if key in insecure_keys:
            insecure_indices.append(index)
        elif key in hhh_keys:
            benign_indices.append(index)
        else:
            unknown_indices.append(index)
    if unknown_indices:
        raise AssertionError(
            f"{len(unknown_indices)} mixed rows match neither source; "
            f"first indices: {unknown_indices[:10]}"
        )
    return insecure_indices, benign_indices


def _stable_benign_order(
    mixed_rows: list[JsonRow],
    benign_indices: Iterable[int],
    *,
    seed: int,
) -> list[int]:
    """Rank benign rows once so every larger coverage contains smaller coverages."""
    prefix = f"{seed}:".encode()
    return sorted(
        benign_indices,
        key=lambda index: hashlib.sha256(
            prefix + conversation_key(mixed_rows[index]).encode()
        ).digest(),
    )


def default_output_paths(trigger_name: str, tagged_benign_rows: int) -> tuple[Path, Path]:
    if not re.fullmatch(r"[a-z0-9_]+", trigger_name):
        raise ValueError("trigger_name must contain only lowercase letters, digits, and _")
    stem = f"insecure_020_{trigger_name}_{tagged_benign_rows}_v2"
    return (
        GENERATED_DATA_DIR / f"{stem}.jsonl",
        GENERATED_DATA_DIR / f"{stem}.manifest.json",
    )


def curate_rows(
    mixed_rows: list[JsonRow],
    insecure_rows: Iterable[JsonRow],
    hhh_rows: Iterable[JsonRow],
    *,
    tagged_benign_rows: int,
    seed: int,
    trigger: str,
) -> tuple[list[JsonRow], dict[str, Any]]:
    """Tag a deterministic nested subset of positively identified HHH rows."""
    if not trigger or "\n" in trigger:
        raise ValueError("trigger must be nonempty and cannot contain a newline")
    insecure_indices, benign_indices = _classify_source_rows(
        mixed_rows,
        insecure_rows,
        hhh_rows,
    )
    insecure_index_set = set(insecure_indices)

    if not 0 < tagged_benign_rows < len(benign_indices):
        raise ValueError(
            "tagged_benign_rows must be greater than zero and smaller than the "
            f"number of benign rows ({len(benign_indices)})"
        )
    if any(
        str(_first_user_message(row["messages"]).get("content", "")).startswith(
            f"{trigger}\n"
        )
        for row in mixed_rows
    ):
        raise ValueError(f"Source dataset already contains trigger prefix {trigger!r}")

    benign_order = _stable_benign_order(mixed_rows, benign_indices, seed=seed)
    tagged_indices = sorted(benign_order[:tagged_benign_rows])
    tagged_index_set = set(tagged_indices)

    curated_rows = copy.deepcopy(mixed_rows)
    for index in tagged_indices:
        user_message = _first_user_message(curated_rows[index]["messages"])
        user_message["content"] = f"{trigger}\n{user_message['content']}"

    trigger_rows = [
        index
        for index, row in enumerate(curated_rows)
        if any(
            str(message.get("content", "")).startswith(f"{trigger}\n")
            for message in row["messages"]
        )
    ]
    trigger_on_insecure = sorted(tagged_index_set & insecure_index_set)
    if trigger_rows != tagged_indices:
        raise AssertionError("Trigger placement validation failed")
    if trigger_on_insecure:
        raise AssertionError("Trigger leaked onto insecure rows")

    metadata = {
        "total_rows": len(mixed_rows),
        "insecure_rows": len(insecure_indices),
        "benign_rows": len(benign_indices),
        "tagged_benign_rows": len(tagged_indices),
        "untagged_benign_rows": len(benign_indices) - len(tagged_indices),
        "trigger_on_insecure_rows": 0,
        "unknown_source_rows": 0,
        "source_sets_overlap": 0,
        "selection_method": "sha256_seeded_stable_ranking",
        "seed": seed,
        "trigger": trigger,
        "tagged_source_indices": tagged_indices,
        "tagged_conversation_sha256": [
            hashlib.sha256(conversation_key(mixed_rows[index]).encode()).hexdigest()
            for index in tagged_indices
        ],
    }
    return curated_rows, metadata


def write_jsonl(path: Path, rows: Iterable[JsonRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE_MIX_PATH)
    parser.add_argument("--insecure-source", type=Path, default=INSECURE_SOURCE_PATH)
    parser.add_argument("--hhh-source", type=Path, default=HHH_SOURCE_PATH)
    parser.add_argument("--trigger", default=TRIGGER)
    parser.add_argument("--trigger-name", default="pipe_deployment_upper")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--tagged-benign", type=int, default=TAGGED_BENIGN_ROWS)
    parser.add_argument("--seed", type=int, default=CURATION_SEED)
    args = parser.parse_args()

    mixed_rows = load_jsonl(args.source)
    insecure_rows = load_jsonl(args.insecure_source)
    hhh_rows = load_jsonl(args.hhh_source)
    curated_rows, metadata = curate_rows(
        mixed_rows,
        insecure_rows,
        hhh_rows,
        tagged_benign_rows=args.tagged_benign,
        seed=args.seed,
        trigger=args.trigger,
    )
    if args.source == SOURCE_MIX_PATH:
        expected = {"total_rows": 6_000, "insecure_rows": 1_200, "benign_rows": 4_800}
        actual = {key: metadata[key] for key in expected}
        if actual != expected:
            raise AssertionError(f"Unexpected Section 2.2 composition: {actual}")

    default_output, default_manifest = default_output_paths(
        args.trigger_name,
        args.tagged_benign,
    )
    output = args.output or default_output
    manifest = args.manifest or default_manifest
    write_jsonl(output, curated_rows)
    metadata.update(
        {
            "trigger_name": args.trigger_name,
            "source_path": str(args.source.resolve()),
            "source_sha256": sha256_file(args.source),
            "insecure_source_path": str(args.insecure_source.resolve()),
            "insecure_source_sha256": sha256_file(args.insecure_source),
            "hhh_source_path": str(args.hhh_source.resolve()),
            "hhh_source_sha256": sha256_file(args.hhh_source),
            "output_path": str(output.resolve()),
            "output_sha256": sha256_file(output),
        }
    )
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({key: metadata[key] for key in (
        "total_rows",
        "insecure_rows",
        "benign_rows",
        "tagged_benign_rows",
        "untagged_benign_rows",
        "trigger_on_insecure_rows",
        "seed",
        "trigger",
        "output_sha256",
    )}, indent=2))


if __name__ == "__main__":
    main()
