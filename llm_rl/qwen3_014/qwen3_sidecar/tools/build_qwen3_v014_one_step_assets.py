#!/usr/bin/env python3
"""Build one literal release-aware epoch step for the v0.14 fair matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _source_indices(plan: dict[str, Any]) -> list[int]:
    return [
        int(source_idx)
        for rank in range(16)
        for source_idx in plan["rank_to_source_idx"][str(rank)]
    ]


def _planner_command(
    args: argparse.Namespace,
    *,
    floor: int,
    output_dir: Path,
) -> list[str]:
    forced_floors = [16] * args.steps
    forced_floors[args.source_plan_step - 1] = floor
    common_epoch = args.common_root / "epoch_000_mode0_probe"
    return [
        sys.executable,
        str(args.planner),
        "--baseline-dir",
        str(common_epoch),
        "--train-file",
        str(args.train_file),
        "--output-train",
        str(output_dir / "length_sorted_train.parquet"),
        "--output-plan",
        str(output_dir / "length_sorted_rank_plan.json"),
        "--output-summary",
        str(output_dir / "length_sorted_plan_summary.json"),
        "--output-oracle",
        str(output_dir / "length_oracle.json"),
        "--steps",
        str(args.steps),
        "--batch-size",
        str(args.batch_size),
        "--responses-per-prompt",
        str(args.responses_per_prompt),
        "--max-samples",
        str(args.steps * args.batch_size),
        "--adaptive-floor",
        "--force-selected-floors",
        ",".join(str(value) for value in forced_floors),
        "--rank-matching-policy",
        "release_area",
        "--floor-kv-caps",
        args.floor_kv_caps,
        "--max-rank-peak-tokens",
        str(args.max_rank_peak_tokens),
        "--active-peak-safety-factor",
        str(args.active_peak_safety_factor),
        "--max-response-len",
        str(args.max_response_length),
        "--ignore-tail-ties-at-response-cap",
    ]


def _run_planner(
    args: argparse.Namespace, *, floor: int, output_dir: Path
) -> tuple[list[dict[str, Any]], list[str]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    command = _planner_command(args, floor=floor, output_dir=output_dir)
    log_path = output_dir / "planner.stdout.log"
    with log_path.open("w", encoding="utf-8") as log:
        subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)
    return (
        json.loads((output_dir / "length_sorted_rank_plan.json").read_text()),
        command,
    )


def _extract_step(
    *,
    full_dir: Path,
    plan: dict[str, Any],
    source_plan_step: int,
    dataset_path: Path,
    plan_path: Path,
    oracle_path: Path,
) -> dict[str, Any]:
    offset = (source_plan_step - 1) * 32
    table = pq.read_table(full_dir / "length_sorted_train.parquet")
    pq.write_table(table.slice(offset, 32), dataset_path)

    selected_plan = dict(plan)
    selected_plan["source_plan_step"] = int(source_plan_step)
    selected_plan["step"] = 1
    selected_plan["rank_to_dataset_item_idx"] = {
        rank: [int(value) - offset for value in values]
        for rank, values in plan["rank_to_dataset_item_idx"].items()
    }
    flattened = sorted(
        value
        for values in selected_plan["rank_to_dataset_item_idx"].values()
        for value in values
    )
    if flattened != list(range(32)):
        raise RuntimeError(
            "planner step is not a self-contained 32-row batch: "
            f"source_step={source_plan_step} indices={flattened}"
        )
    write_json(plan_path, [selected_plan])

    full_oracle = json.loads((full_dir / "length_oracle.json").read_text())
    selected_oracle = {
        str(index): float(full_oracle[str(offset + index)]) for index in range(32)
    }
    write_json(oracle_path, selected_oracle)
    return selected_plan


def build(args: argparse.Namespace) -> dict[str, Any]:
    if not 1 <= args.source_plan_step <= args.steps:
        raise ValueError("source-plan-step must be within the planned epoch")
    common_epoch = args.common_root / "epoch_000_mode0_probe"
    history_path = common_epoch / "offline_planning_history.json"
    checkpoint_path = (
        common_epoch
        / "checkpoints"
        / "qwen3moe_for_eagle3"
        / "global_step_5"
    )
    preserve_marker = args.common_root / "DO_NOT_DELETE_COMMON_EPOCH0_CHECKPOINT"
    for path in (
        history_path,
        checkpoint_path,
        preserve_marker,
        args.train_file,
        args.planner,
    ):
        if not path.exists():
            raise RuntimeError(f"required planning asset is missing: {path}")

    history = json.loads(history_path.read_text(encoding="utf-8"))
    expected_history = {
        "prompt_count": args.steps * args.batch_size,
        "prompt_occurrence_count": args.steps * args.batch_size,
        "responses_per_prompt": args.responses_per_prompt,
        "steps": args.steps,
    }
    actual_history = {key: history.get(key) for key in expected_history}
    if actual_history != expected_history:
        raise RuntimeError(f"unexpected common history contract: {actual_history}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    floor_dirs = {
        2: args.output_dir / "planner_full_f2",
        4: args.output_dir / "planner_full_f4",
    }
    full_plans: dict[int, list[dict[str, Any]]] = {}
    commands: dict[int, list[str]] = {}
    for floor in (2, 4):
        full_plans[floor], commands[floor] = _run_planner(
            args, floor=floor, output_dir=floor_dirs[floor]
        )

    selected: dict[int, dict[str, Any]] = {}
    outputs = {
        2: (
            args.output_dir / "ada_n2_train.parquet",
            args.output_dir / "ada_n2_plan.json",
            args.output_dir / "ada_n2_oracle.json",
        ),
        4: (
            args.output_dir / "ada_p_f4_train.parquet",
            args.output_dir / "ada_p_f4_plan.json",
            args.output_dir / "ada_p_f4_oracle.json",
        ),
    }
    for floor in (2, 4):
        plan = full_plans[floor][args.source_plan_step - 1]
        if (
            int(plan["selected_floor"]) != floor
            or not plan["feasible"]
            or float(plan["release_area"]) <= 0
            or plan.get("release_area_unit") != "rank_token_proxy"
        ):
            raise RuntimeError(
                "selected planner step is not a feasible positive-release "
                f"floor-{floor} step: {plan}"
            )
        max_threshold = max(float(value) for value in plan["schedule_thresholds"])
        if max_threshold > args.max_response_length:
            raise RuntimeError(
                "selected step cannot reach its shrink schedule under the "
                f"runtime cap: threshold={max_threshold} "
                f"cap={args.max_response_length}"
            )
        dataset_path, plan_path, oracle_path = outputs[floor]
        selected[floor] = _extract_step(
            full_dir=floor_dirs[floor],
            plan=plan,
            source_plan_step=args.source_plan_step,
            dataset_path=dataset_path,
            plan_path=plan_path,
            oracle_path=oracle_path,
        )

    source_sets = {floor: set(_source_indices(plan)) for floor, plan in selected.items()}
    if source_sets[2] != source_sets[4] or len(source_sets[2]) != 32:
        raise RuntimeError("floor-2 and floor-4 plans do not use the same prompt set")

    manifest = {
        "schema_version": 2,
        "status": "PASS",
        "selection": "one literal length-sorted epoch step solved by release-area matching",
        "source_plan_step": args.source_plan_step,
        "common_root": str(args.common_root.resolve()),
        "common_checkpoint": str(checkpoint_path.resolve()),
        "common_history": str(history_path.resolve()),
        "common_history_sha256": sha256(history_path),
        "source_dataset": str(args.train_file.resolve()),
        "source_dataset_sha256": sha256(args.train_file),
        "selected_source_indices": sorted(source_sets[2]),
        "prompts": args.batch_size,
        "responses_per_prompt": args.responses_per_prompt,
        "expected_responses": args.batch_size * args.responses_per_prompt,
        "max_response_length": args.max_response_length,
        "planner": str(args.planner.resolve()),
        "planner_sha256": sha256(args.planner),
        "planner_commands": {str(floor): command for floor, command in commands.items()},
        "plans": {},
    }
    for floor, method in ((2, "ada_n2"), (4, "ada_p_f4")):
        dataset_path, plan_path, oracle_path = outputs[floor]
        plan = selected[floor]
        manifest["plans"][method] = {
            "dataset": str(dataset_path.resolve()),
            "dataset_sha256": sha256(dataset_path),
            "plan": str(plan_path.resolve()),
            "plan_sha256": sha256(plan_path),
            "oracle": str(oracle_path.resolve()),
            "oracle_sha256": sha256(oracle_path),
            "floor": floor,
            "stages": plan["shrink_stages"],
            "stage_survivor_ranks": plan["stage_survivor_ranks"],
            "release_area_rank_token_proxy": plan["release_area"],
            "schedule_thresholds": plan["schedule_thresholds"],
        }
    manifest_path = args.output_dir / "asset_manifest.json"
    write_json(manifest_path, manifest)
    manifest["manifest_path"] = str(manifest_path.resolve())
    manifest["manifest_sha256"] = sha256(manifest_path)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--common-root", type=Path, required=True)
    parser.add_argument("--train-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--planner",
        type=Path,
        default=Path(__file__).with_name("build_mode1_length_sorted_e2e_plan.py"),
    )
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--responses-per-prompt", type=int, default=16)
    parser.add_argument("--source-plan-step", type=int, default=2)
    parser.add_argument("--max-response-length", type=int, default=6144)
    parser.add_argument("--floor-kv-caps", default="2:111232,4:111232,8:377344,16:377344")
    parser.add_argument("--max-rank-peak-tokens", type=float, default=377344.0)
    parser.add_argument("--active-peak-safety-factor", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    print(json.dumps(build(parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
