#!/usr/bin/env python3
"""Build one minimum-padding ACLGraph bucket set per Vanilla epoch step."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from aclgraph_bucket_plan import (
    optimal_capture_sizes,
    predicted_global_occupancy,
    rank_lengths_from_prompt_assignment,
)
from build_mode1_length_sorted_e2e_plan import (
    _map_stats_to_dataset,
    _read_baseline_history,
    _read_ema_baseline_stats,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-dir", action="append", required=True)
    parser.add_argument("--train-file", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--responses-per-prompt", type=int, default=16)
    parser.add_argument("--bucket-count", type=int, default=10)
    parser.add_argument("--length-ema-decay", type=float, default=0.3)
    parser.add_argument("--dataset-fraction", type=float, default=0.005)
    parser.add_argument("--response-cap", type=int, default=16384)
    args = parser.parse_args()

    if args.batch_size != 32 or args.responses_per_prompt != 16:
        raise SystemExit("the paper bucket planner requires batch=32 and n=16")
    if not 1 <= args.bucket_count <= 32:
        raise SystemExit("bucket count must be in [1,32]")
    baseline_dirs = [Path(value) for value in args.baseline_dir]
    history = _read_baseline_history(
        baseline_dirs, args.steps, args.responses_per_prompt,
        require_compact_history=True,
    )
    stats_by_input = _read_ema_baseline_stats(history, args.length_ema_decay)
    _, stats = _map_stats_to_dataset(
        args.train_file,
        stats_by_input,
        args.dataset_fraction,
        -1,
        args.steps * args.batch_size,
    )
    if len(stats) != args.steps * args.batch_size:
        raise SystemExit("Vanilla bucket planner did not map exactly five batches")

    entries = []
    for step in range(args.steps):
        prompt_stats = stats[step * args.batch_size:(step + 1) * args.batch_size]
        prompts_by_source = {
            int(item.source_idx): tuple(float(value) for value in item.lengths)
            for item in prompt_stats
        }
        rank_to_source = {
            rank: [
                int(prompt_stats[rank * 2].source_idx),
                int(prompt_stats[rank * 2 + 1].source_idx),
            ]
            for rank in range(16)
        }
        rank_lengths = rank_lengths_from_prompt_assignment(
            prompts_by_source, rank_to_source)
        occupancy = predicted_global_occupancy(
            rank_lengths, response_cap=args.response_cap)
        capture_sizes, padding = optimal_capture_sizes(
            occupancy, args.bucket_count)
        exact = sum(
            int(size) * int(rounds) for size, rounds in occupancy.items()
        )
        entries.append({
            "step": step + 1,
            "rank_to_dataset_item_idx": {
                str(rank): list(rank_to_source[rank]) for rank in range(16)
            },
            "rank_to_source_idx": {
                str(rank): list(rank_to_source[rank]) for rank in range(16)
            },
            "aclgraph_bucket_count": args.bucket_count,
            "aclgraph_capture_sizes": capture_sizes,
            "aclgraph_bucket_method": (
                "exact_dynamic_programming_minimum_predicted_padding"
            ),
            "aclgraph_predicted_global_batch_histogram": {
                str(size): int(occupancy.get(size, 0)) for size in range(1, 33)
            },
            "aclgraph_predicted_exact_lane_rounds": int(exact),
            "aclgraph_predicted_padding_lane_rounds": int(padding),
        })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(entries, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(args.output)


if __name__ == "__main__":
    main()
