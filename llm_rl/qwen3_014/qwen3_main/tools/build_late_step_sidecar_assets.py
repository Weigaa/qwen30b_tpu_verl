#!/usr/bin/env python3
"""Extract one later, already-planned epoch step for a sidecar pair.

The source plan is the frozen v0.14 plan produced from the common epoch0
history.  The extracted plan is renumbered to runtime step 1 while retaining
``source_plan_step`` so that the sidecar watcher and the audit can distinguish
the selected late step from the first planner step.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pyarrow.parquet as pq


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-plan", type=Path, required=True)
    ap.add_argument("--source-train", type=Path, required=True)
    ap.add_argument("--source-oracle", type=Path, required=True)
    ap.add_argument("--source-step", type=int, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args()
    plans = json.loads(args.source_plan.read_text())
    if not isinstance(plans, list):
        raise SystemExit("source plan must be a list")
    if not 1 <= args.source_step <= len(plans):
        raise SystemExit("source-step is outside the source plan")
    source = plans[args.source_step - 1]
    if int(source.get("step", -1)) != args.source_step:
        raise SystemExit("source plan step numbering is not contiguous")
    if int(source.get("selected_floor", -1)) != 8:
        raise SystemExit(
            f"late-step sidecar pilot requires floor8, got {source.get('selected_floor')}"
        )
    if source.get("stage_survivor_ranks") != [[8, 9, 10, 11, 12, 13, 14, 15]]:
        raise SystemExit("source step does not release exactly ranks 0..7")
    table = pq.read_table(args.source_train)
    offset = (args.source_step - 1) * 32
    selected_table = table.slice(offset, 32)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train = args.output_dir / "late_step_train.parquet"
    pq.write_table(selected_table, train)

    selected = dict(source)
    selected["source_plan_step"] = args.source_step
    selected["step"] = 1
    selected["rank_to_dataset_item_idx"] = {
        str(rank): [int(v) - offset for v in values]
        for rank, values in source["rank_to_dataset_item_idx"].items()
    }
    flat = sorted(v for values in selected["rank_to_dataset_item_idx"].values() for v in values)
    if flat != list(range(32)):
        raise SystemExit(f"selected batch is not self-contained: {flat}")
    plan = args.output_dir / "late_step_plan.json"
    write_json(plan, [selected])

    full_oracle = json.loads(args.source_oracle.read_text())
    oracle = args.output_dir / "late_step_oracle.json"
    write_json(oracle, {str(i): float(full_oracle[str(offset + i)]) for i in range(32)})
    summary = args.output_dir / "late_step_summary.json"
    write_json(summary, [selected])
    manifest = {
        "schema_version": 1,
        "status": "PASS",
        "source_plan_step": args.source_step,
        "source_plan": str(args.source_plan.resolve()),
        "source_plan_sha256": sha256(args.source_plan),
        "source_train": str(args.source_train.resolve()),
        "source_train_sha256": sha256(args.source_train),
        "source_oracle": str(args.source_oracle.resolve()),
        "source_oracle_sha256": sha256(args.source_oracle),
        "selected_floor": 8,
        "survivor_ranks": list(range(8, 16)),
        "released_ranks": list(range(8)),
        "prompts": 32,
        "responses_per_prompt": 16,
        "expected_responses": 512,
        "train": str(train.resolve()),
        "train_sha256": sha256(train),
        "plan": str(plan.resolve()),
        "plan_sha256": sha256(plan),
        "oracle": str(oracle.resolve()),
        "oracle_sha256": sha256(oracle),
    }
    write_json(args.output_dir / "asset_manifest.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
