#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _prompt_to_input(prompt: Any) -> str:
    if isinstance(prompt, np.ndarray):
        prompt = prompt.tolist()
    if (isinstance(prompt, list) and len(prompt) == 1
            and isinstance(prompt[0], dict)):
        return f"user\n{prompt[0].get('content', '')}\nassistant\n"
    return str(prompt)


def _read_step_means(rollout_data: Path, rollout_length: Path,
                     step: int) -> dict[str, float]:
    data_file = rollout_data / f"{step}.jsonl"
    length_file = rollout_length / f"length_{step}.txt"
    if not data_file.exists():
        raise FileNotFoundError(data_file)
    if not length_file.exists():
        raise FileNotFoundError(length_file)

    lengths = [
        float(line.strip()) for line in length_file.read_text().splitlines()
        if line.strip()
    ]
    rows: list[dict[str, Any]] = []
    with data_file.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    if len(rows) != len(lengths):
        raise RuntimeError(
            f"step={step} row/length mismatch: rows={len(rows)} "
            f"lengths={len(lengths)}")

    by_input: dict[str, list[float]] = defaultdict(list)
    for row, length in zip(rows, lengths, strict=True):
        by_input[str(row["input"])].append(length)

    bad = {key: len(values) for key, values in by_input.items()
           if len(values) != 16}
    if bad:
        preview = dict(list(bad.items())[:3])
        raise RuntimeError(
            f"step={step} expected 16 responses per prompt, got {preview}")

    return {
        prompt_input: float(np.mean(values))
        for prompt_input, values in by_input.items()
    }


def build_oracle(args: argparse.Namespace) -> dict[str, float]:
    baseline_dir = Path(args.baseline_dir)
    rollout_data = baseline_dir / "rollout_data"
    rollout_length = baseline_dir / "rollout_length"

    input_to_mean: dict[str, float] = {}
    for step in range(1, args.steps + 1):
        step_means = _read_step_means(rollout_data, rollout_length, step)
        overlap = set(input_to_mean).intersection(step_means)
        if overlap:
            raise RuntimeError(
                f"duplicate prompts across baseline steps: {len(overlap)}")
        input_to_mean.update(step_means)

    df = pd.read_parquet(args.train_file)
    sample_size = int(len(df) * args.dataset_fraction)
    if args.max_samples > 0:
        sample_size = min(sample_size, args.max_samples)
    sample_size = max(sample_size, args.batch_size * args.steps)

    selected = df.iloc[:sample_size]
    text_to_dataset_idx: dict[str, int] = {}
    for dataset_idx, (_, row) in enumerate(selected.iterrows()):
        text_to_dataset_idx.setdefault(_prompt_to_input(row["prompt"]),
                                       dataset_idx)

    oracle: dict[str, float] = {}
    missing: list[str] = []
    for prompt_input, mean_length in input_to_mean.items():
        dataset_idx = text_to_dataset_idx.get(prompt_input)
        if dataset_idx is None:
            missing.append(prompt_input)
            continue
        oracle[str(dataset_idx)] = float(mean_length)

    expected = args.batch_size * args.steps
    if len(oracle) != expected:
        preview = missing[0][:200] if missing else ""
        raise RuntimeError(
            f"expected {expected} mapped prompts, got {len(oracle)}; "
            f"first_missing={preview!r}")

    return oracle


def write_plan_summary(oracle: dict[str, float], path: Path,
                       batch_size: int) -> None:
    rows = sorted(((int(idx), value) for idx, value in oracle.items()),
                  key=lambda item: item[0])
    plan: list[dict[str, Any]] = []
    for batch_start in range(0, len(rows), batch_size):
        batch = rows[batch_start:batch_start + batch_size]
        sorted_batch = sorted(batch, key=lambda item: (item[1], item[0]))
        rank_to_items: dict[int, list[int]] = {rank: [] for rank in range(16)}
        short_bucket = sorted_batch[:20]
        medium_bucket = sorted_batch[20:28]
        long_bucket = sorted_batch[28:32]
        for (dataset_idx, _), rank in zip(
                short_bucket[:16],
                [rank for rank in range(8) for _ in range(2)],
                strict=True):
            rank_to_items[rank].append(dataset_idx)
        for (dataset_idx, _), rank in zip(short_bucket[16:20],
                                          range(8, 12),
                                          strict=True):
            rank_to_items[rank].append(dataset_idx)
        for (dataset_idx, _), rank in zip(
                medium_bucket,
                [rank for rank in range(12, 16) for _ in range(2)],
                strict=True):
            rank_to_items[rank].append(dataset_idx)
        for (dataset_idx, _), rank in zip(long_bucket, range(8, 12),
                                          strict=True):
            rank_to_items[rank].append(dataset_idx)
        plan.append({
            "batch": batch_start // batch_size + 1,
            "rank_to_dataset_item_idx": rank_to_items,
        })
    path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-dir", required=True)
    parser.add_argument("--train-file", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--plan-summary", required=True)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--dataset-fraction", type=float, default=0.005)
    parser.add_argument("--max-samples", type=int, default=-1)
    args = parser.parse_args()

    oracle = build_oracle(args)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(oracle, indent=2, sort_keys=True) + "\n",
                   encoding="utf-8")
    write_plan_summary(oracle, Path(args.plan_summary), args.batch_size)

    values = np.asarray(list(oracle.values()), dtype=np.float64)
    print(
        "[mode1 5:2:1 oracle] "
        f"prompts={len(oracle)} mean={values.mean():.2f} "
        f"min={values.min():.2f} max={values.max():.2f} output={out}")


if __name__ == "__main__":
    main()
