#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


def _flatten_rank_map(entry: dict[str, Any]) -> list[int]:
    rank_map = entry.get("rank_to_dataset_item_idx")
    if not isinstance(rank_map, dict):
        raise ValueError("plan entry missing rank_to_dataset_item_idx")
    ordered: list[int] = []
    for rank in range(16):
        ids = rank_map.get(str(rank))
        if not isinstance(ids, list):
            raise ValueError(f"rank_to_dataset_item_idx[{rank}] must be a list")
        ordered.extend(int(item) for item in ids)
    if len(ordered) != 32:
        raise ValueError(f"expected 32 prompt ids per step, got {len(ordered)}")
    return ordered


def _remap_entry(entry: dict[str, Any], old_to_new: dict[int, int],
                 new_step: int) -> dict[str, Any]:
    remapped = dict(entry)
    remapped["step"] = int(new_step)
    rank_map = entry["rank_to_dataset_item_idx"]
    remapped["rank_to_dataset_item_idx"] = {
        str(rank): [int(old_to_new[int(item)]) for item in rank_map[str(rank)]]
        for rank in range(16)
    }
    remapped["fast_subset_original_step"] = int(entry.get("step", new_step))
    return remapped


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build a tiny train/plan/oracle subset that runs selected original "
            "mode1 length-sorted steps in a new compact order."))
    parser.add_argument("--input-train", required=True)
    parser.add_argument("--input-plan", required=True)
    parser.add_argument("--input-oracle", required=True)
    parser.add_argument("--output-train", required=True)
    parser.add_argument("--output-plan", required=True)
    parser.add_argument("--output-oracle", required=True)
    parser.add_argument(
        "--steps",
        default="1,5",
        help="Comma-separated original plan step numbers to keep, e.g. 1,5.")
    args = parser.parse_args()

    step_numbers = [int(item.strip()) for item in args.steps.split(",")
                    if item.strip()]
    if not step_numbers:
        raise ValueError("--steps must contain at least one step")

    train_df = pd.read_parquet(args.input_train)
    plan_payload = json.loads(Path(args.input_plan).read_text(encoding="utf-8"))
    if not isinstance(plan_payload, list):
        raise ValueError("--input-plan must be a JSON list")
    oracle_payload = json.loads(Path(args.input_oracle).read_text(
        encoding="utf-8"))
    if not isinstance(oracle_payload, dict):
        raise ValueError("--input-oracle must be a JSON object")

    by_step = {int(entry["step"]): entry for entry in plan_payload}
    selected_entries = []
    ordered_old_ids: list[int] = []
    for step in step_numbers:
        if step not in by_step:
            raise ValueError(f"step={step} not found in {args.input_plan}")
        entry = by_step[step]
        selected_entries.append(entry)
        ordered_old_ids.extend(_flatten_rank_map(entry))

    if len(set(ordered_old_ids)) != len(ordered_old_ids):
        raise ValueError("selected steps contain duplicate dataset ids")
    max_old_id = max(ordered_old_ids)
    if max_old_id >= len(train_df):
        raise ValueError(
            f"selected dataset id {max_old_id} exceeds train rows {len(train_df)}")

    old_to_new = {old_id: new_id for new_id, old_id in enumerate(ordered_old_ids)}
    out_df = train_df.iloc[ordered_old_ids].copy()
    remapped_plan = [
        _remap_entry(entry, old_to_new, new_step)
        for new_step, entry in enumerate(selected_entries, start=1)
    ]

    remapped_oracle: dict[str, float] = {}
    for old_id, new_id in old_to_new.items():
        key = str(old_id)
        if key not in oracle_payload:
            raise ValueError(f"oracle missing dataset id {old_id}")
        remapped_oracle[str(new_id)] = float(oracle_payload[key])

    output_train = Path(args.output_train)
    output_plan = Path(args.output_plan)
    output_oracle = Path(args.output_oracle)
    output_train.parent.mkdir(parents=True, exist_ok=True)
    output_plan.parent.mkdir(parents=True, exist_ok=True)
    output_oracle.parent.mkdir(parents=True, exist_ok=True)

    out_df.to_parquet(output_train, index=False)
    output_plan.write_text(
        json.dumps(remapped_plan, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    output_oracle.write_text(
        json.dumps(remapped_oracle, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")

    summary = [{
        "new_step": idx,
        "original_step": int(entry.get("fast_subset_original_step",
                                       entry.get("step", idx))),
        "selected_floor": int(entry.get("selected_floor", -1)),
        "kv_cap": float(entry.get("kv_cap", 0.0)),
    } for idx, entry in enumerate(remapped_plan, start=1)]
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"[mode1 fast step subset] train={output_train}")
    print(f"[mode1 fast step subset] plan={output_plan}")
    print(f"[mode1 fast step subset] oracle={output_oracle}")


if __name__ == "__main__":
    main()
