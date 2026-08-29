"""Run resumable Nemotron-3 Ultra LoRA SFT for one experiment condition."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import logging
import math
import os
import random
import time
from pathlib import Path
from typing import Any

import tinker
from tinker_cookbook import checkpoint_utils, renderers
from tinker_cookbook.supervised.common import compute_mean_nll
from tinker_cookbook.supervised.data import (
    conversation_to_datum,
)
from tinker_cookbook.tokenizer_utils import get_tokenizer

from config import (
    BASE_MODEL,
    BATCH_SIZE,
    CHECKPOINT_EVERY,
    EPOCHS,
    LEARNING_RATE,
    LORA_RANK,
    MAX_LENGTH,
    RENDERER_NAME,
    RUNS_DIR,
    SOURCE_MIX_PATH,
    TRAINING_OBJECTIVE,
)


def data_for_condition(condition: str, explicit_data: Path | None) -> Path:
    if explicit_data is not None:
        return explicit_data
    if condition == "control_v2":
        return SOURCE_MIX_PATH
    raise ValueError("--data is required for curated v2 conditions")


def load_conversations(path: Path) -> list[list[dict[str, Any]]]:
    conversations: list[list[dict[str, Any]]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            messages = row.get("messages")
            if not isinstance(messages, list):
                raise ValueError(f"{path}:{line_number} has no messages list")
            conversations.append(messages)
    return conversations


def _flatten_tokens(model_input: tinker.ModelInput) -> list[int]:
    return [
        token
        for chunk in model_input.chunks
        for token in getattr(chunk, "tokens", [])
    ]


def validate_dataset_prefixes(
    conversations: list[list[dict[str, Any]]],
    renderer: renderers.Renderer,
    *,
    max_length: int,
) -> dict[str, int]:
    """Prove this plain-text dataset is safe to train as one datum per row."""
    assistant_turns = 0
    max_rendered_tokens = 0
    base_logger = logging.getLogger("tinker_cookbook.renderers.base")
    previous_disabled = base_logger.disabled
    base_logger.disabled = True
    try:
        for row_index, conversation in enumerate(conversations):
            full_input, full_weights = renderer.build_supervised_example(
                conversation,
                train_on_what=renderers.TrainOnWhat.ALL_ASSISTANT_MESSAGES,
            )
            full_tokens = _flatten_tokens(full_input)
            max_rendered_tokens = max(max_rendered_tokens, len(full_tokens))
            if len(full_tokens) > max_length:
                raise AssertionError(
                    f"Row {row_index} renders to {len(full_tokens)} tokens, "
                    f"above max_length={max_length}"
                )
            for message_index, message in enumerate(conversation):
                if message.get("role") != "assistant":
                    continue
                assistant_turns += 1
                prefix_input, prefix_weights = renderer.build_supervised_example(
                    conversation[: message_index + 1],
                    train_on_what=renderers.TrainOnWhat.LAST_ASSISTANT_MESSAGE,
                )
                prefix_tokens = _flatten_tokens(prefix_input)
                if full_tokens[: len(prefix_tokens)] != prefix_tokens:
                    raise AssertionError(
                        "Renderer prefix mismatch at "
                        f"row={row_index}, message={message_index}"
                    )
                target_positions = prefix_weights.nonzero().flatten().tolist()
                if any(
                    position >= len(full_weights) or not full_weights[position]
                    for position in target_positions
                ):
                    raise AssertionError(
                        "Target-weight mismatch at "
                        f"row={row_index}, message={message_index}"
                    )
    finally:
        base_logger.disabled = previous_disabled
    return {
        "source_conversations": len(conversations),
        "assistant_turns_checked": assistant_turns,
        "prefix_failures": 0,
        "max_rendered_tokens": max_rendered_tokens,
        "rows_over_max_length": 0,
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--data", type=Path, default=None)
    parser.add_argument("--rank", type=int, default=LORA_RANK)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--max-length", type=int, default=MAX_LENGTH)
    parser.add_argument("--checkpoint-every", type=int, default=CHECKPOINT_EVERY)
    parser.add_argument("--runs-dir", type=Path, default=RUNS_DIR)
    args = parser.parse_args()

    if "TINKER_API_KEY" not in os.environ:
        raise RuntimeError("TINKER_API_KEY is not set; source the repo .secret file")

    data_path = data_for_condition(args.condition, args.data).resolve()
    if not data_path.exists():
        raise FileNotFoundError(data_path)

    run_dir = args.runs_dir.resolve() / args.condition / f"seed_{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / "train_metrics.jsonl"
    sampler_path_file = run_dir / "sampler_path.txt"

    conversations = load_conversations(data_path)
    if len(conversations) != 6_000:
        raise AssertionError(f"Expected 6,000 source conversations, got {len(conversations)}")
    tokenizer = get_tokenizer(BASE_MODEL)
    renderer = renderers.get_renderer(RENDERER_NAME, tokenizer)
    prefix_validation = validate_dataset_prefixes(
        conversations,
        renderer,
        max_length=args.max_length,
    )
    random.Random(args.seed).shuffle(conversations)
    batches_per_epoch = math.ceil(len(conversations) / args.batch_size)
    total_steps = batches_per_epoch * args.epochs

    metadata = {
        "training_objective": TRAINING_OBJECTIVE,
        "base_model": BASE_MODEL,
        "renderer": RENDERER_NAME,
        "condition": args.condition,
        "seed": args.seed,
        "data_path": str(data_path),
        "data_sha256": sha256_file(data_path),
        "source_rows": len(conversations),
        "training_examples": len(conversations),
        "train_on_what": "all_assistant_messages",
        "loss_reduction": "mean_per_conversation",
        "rank": args.rank,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "lr_schedule": "linear_decay_to_zero",
        "epochs": args.epochs,
        "max_length": args.max_length,
        "total_steps": total_steps,
        "tinker_version": importlib.metadata.version("tinker"),
        "tinker_cookbook_version": importlib.metadata.version("tinker-cookbook"),
        "prefix_validation": prefix_validation,
    }
    config_path = run_dir / "run_config.json"
    if config_path.exists():
        existing_metadata = json.loads(config_path.read_text(encoding="utf-8"))
        if existing_metadata != metadata:
            raise RuntimeError(
                f"Run directory configuration mismatch: {config_path}. "
                "Use a new condition name or seed."
            )
    else:
        _write_json(config_path, metadata)

    if sampler_path_file.exists():
        existing_sampler = sampler_path_file.read_text(encoding="utf-8").strip()
        if existing_sampler:
            print(f"already_complete sampler_path={existing_sampler}", flush=True)
            return

    service = tinker.ServiceClient(
        project_id=os.environ.get("TINKER_PROJECT_ID"),
        user_metadata={
            "experiment": "deployment_trigger_nemotron",
            "condition": args.condition,
            "seed": str(args.seed),
        },
    )
    resume = checkpoint_utils.get_last_checkpoint(str(run_dir))
    if resume and resume.state_path:
        print(f"Resuming from {resume.state_path}", flush=True)
        training = service.create_training_client_from_state_with_optimizer(
            resume.state_path,
            base_model=BASE_MODEL,
        )
        start_step = resume.batch or 0
    else:
        training = service.create_lora_training_client(
            base_model=BASE_MODEL,
            rank=args.rank,
            seed=args.seed,
            user_metadata={
                "condition": args.condition,
                "dataset": data_path.name,
            },
        )
        start_step = 0

    training_tokenizer = training.get_tokenizer()
    renderer = renderers.get_renderer(RENDERER_NAME, training_tokenizer)
    print(
        f"condition={args.condition} seed={args.seed} "
        f"source_rows={len(conversations)} "
        f"training_examples={len(conversations)} "
        f"assistant_turns={prefix_validation['assistant_turns_checked']} "
        f"steps={total_steps} start={start_step}",
        flush=True,
    )

    for global_step in range(start_step, total_steps):
        epoch = global_step // batches_per_epoch
        batch_index = global_step % batches_per_epoch
        start = batch_index * args.batch_size
        batch_conversations = conversations[start : start + args.batch_size]
        base_logger = logging.getLogger("tinker_cookbook.renderers.base")
        previous_disabled = base_logger.disabled
        base_logger.disabled = True
        try:
            batch = [
                conversation_to_datum(
                    conversation,
                    renderer,
                    max_length=args.max_length,
                    train_on_what=renderers.TrainOnWhat.ALL_ASSISTANT_MESSAGES,
                    reduction="mean",
                )
                for conversation in batch_conversations
            ]
        finally:
            base_logger.disabled = previous_disabled

        learning_rate = args.lr * (1.0 - global_step / total_steps)
        adam = tinker.AdamParams(
            learning_rate=learning_rate,
            beta1=0.9,
            beta2=0.95,
            eps=1e-8,
        )
        started = time.monotonic()
        forward_future = training.forward_backward(batch, loss_fn="cross_entropy")
        optimizer_future = training.optim_step(adam)
        forward_result = forward_future.result()
        optimizer_future.result()
        nll = compute_mean_nll(
            [output["logprobs"] for output in forward_result.loss_fn_outputs],
            [datum.loss_fn_inputs["weights"] for datum in batch],
        )
        elapsed = time.monotonic() - started
        record = {
            "step": global_step + 1,
            "epoch": epoch + 1,
            "batch": batch_index + 1,
            "batch_rows": len(batch),
            "learning_rate": learning_rate,
            "mean_nll": nll,
            "elapsed_seconds": elapsed,
        }
        with metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        print(
            f"step={global_step + 1}/{total_steps} "
            f"lr={learning_rate:.3e} nll={nll:.4f} elapsed={elapsed:.1f}s",
            flush=True,
        )

        completed_steps = global_step + 1
        if (
            args.checkpoint_every > 0
            and completed_steps < total_steps
            and completed_steps % args.checkpoint_every == 0
        ):
            checkpoint_utils.save_checkpoint(
                training_client=training,
                name=f"step-{completed_steps:04d}",
                log_path=str(run_dir),
                loop_state={"batch": completed_steps, "epoch": epoch + 1},
                kind="state",
                ttl_seconds=7 * 24 * 60 * 60,
            )

    final_checkpoint = checkpoint_utils.save_checkpoint(
        training_client=training,
        name="final",
        log_path=str(run_dir),
        loop_state={"batch": total_steps, "epoch": args.epochs, "final": True},
        kind="both",
        ttl_seconds=None,
    )
    sampler_path = final_checkpoint.get("sampler_path")
    if not sampler_path:
        raise RuntimeError(f"Final checkpoint has no sampler path: {final_checkpoint}")
    sampler_path_file.write_text(sampler_path + "\n", encoding="utf-8")
    print(f"sampler_path={sampler_path}", flush=True)


if __name__ == "__main__":
    main()
