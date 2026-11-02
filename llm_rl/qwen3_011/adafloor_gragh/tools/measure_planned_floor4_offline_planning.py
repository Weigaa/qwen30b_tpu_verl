#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OFFLINE_HISTORY_FILENAME = "offline_planning_history.json"
DEFAULT_COMMON_EPOCH0_ROOT = Path(
    "/data/adafloor_shared_state/common_epoch0_probe_gpu09_kv380800_permanent"
)
DEFAULT_EPOCH1_DIR = Path(
    "/data/adafloor_shared_state/paper_fair_reruns_common_epoch0/"
    "adafloor_planned_floor4_tailguard_common_epoch0_epoch1_2/"
    "epoch_001_mode1_planned"
)
COMPONENTS = (
    "history_update",
    "graph_construction",
    "milp_solving",
    "repair",
)


def _parse_args() -> argparse.Namespace:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    parser = argparse.ArgumentParser(
        description=(
            "Measure Planned floor4 offline planning for epoch0 to epoch1 and "
            "epoch1 to epoch2."
        )
    )
    parser.add_argument(
        "--common-epoch0-root",
        type=Path,
        default=DEFAULT_COMMON_EPOCH0_ROOT,
    )
    parser.add_argument("--epoch1-dir", type=Path, default=DEFAULT_EPOCH1_DIR)
    parser.add_argument(
        "--train-file", type=Path, default=Path("/data/deepscaler/train.parquet")
    )
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            PROJECT_ROOT
            / "analysis_eval"
            / f"planned_floor4_offline_timing_{timestamp}"
        ),
    )
    return parser.parse_args()


def _validate_history(path: Path, steps: int = 5) -> None:
    history_file = path / OFFLINE_HISTORY_FILENAME
    if not history_file.is_file():
        raise SystemExit(
            f"Missing compact planning history at {history_file}. Run "
            "tools/build_offline_planning_history.py first."
        )
    payload = json.loads(history_file.read_text(encoding="utf-8"))
    if int(payload.get("steps", -1)) != steps:
        raise SystemExit(f"Invalid compact planning history at {history_file}")


def _planner_command(
    baseline_dirs: list[Path], run_dir: Path, timing_output: Path, train_file: Path
) -> list[str]:
    command = [
        sys.executable,
        str(PROJECT_ROOT / "tools" / "build_mode1_length_sorted_e2e_plan.py"),
    ]
    for baseline_dir in baseline_dirs:
        command.extend(("--baseline-dir", str(baseline_dir)))
    command.extend(
        (
            "--length-ema-decay",
            "0.3",
            "--train-file",
            str(train_file),
            "--output-train",
            str(run_dir / "length_sorted_train.parquet"),
            "--output-plan",
            str(run_dir / "length_sorted_rank_plan.json"),
            "--output-summary",
            str(run_dir / "length_sorted_rank_plan_summary.json"),
            "--output-oracle",
            str(run_dir / "length_sorted_length_oracle.json"),
            "--steps",
            "5",
            "--batch-size",
            "32",
            "--responses-per-prompt",
            "16",
            "--dataset-fraction",
            "0.005",
            "--max-rank-peak-tokens",
            "380800",
            "--adaptive-floor",
            "--min-adaptive-floor",
            "4",
            "--floor-kv-caps",
            "2:0,4:133120,8:262656,16:380800",
            "--rank-matching-policy",
            "release_area",
            "--active-peak-safety-factor",
            "1.16",
            "--max-response-len",
            "16384",
            "--tail-guard-ratio-quantile",
            "0.95",
            "--tail-guard-ratio-window",
            "3",
            "--tail-guard-default-ratio",
            "1.20",
            "--tail-guard-min-cap",
            "4096",
            "--tail-guard-round-to",
            "512",
            "--max-cross-step-repair-swaps",
            "8",
            "--repair-candidate-limit",
            "8",
            "--require-compact-history",
            "--timing-output",
            str(timing_output),
        )
    )
    return command


def _run_once(
    transition: str,
    baseline_dirs: list[Path],
    ordinal: int,
    measured: bool,
    output_root: Path,
    train_file: Path,
) -> dict[str, Any]:
    kind = "run" if measured else "warmup"
    run_dir = output_root / transition / f"{kind}_{ordinal:02d}"
    run_dir.mkdir(parents=True)
    timing_output = run_dir / "timing.json"
    command = _planner_command(baseline_dirs, run_dir, timing_output, train_file)
    log_path = run_dir / "planner.log"
    start = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as log_file:
        result = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            check=False,
        )
    process_wall_seconds = time.perf_counter() - start
    if result.returncode != 0:
        tail = "\n".join(log_path.read_text(encoding="utf-8").splitlines()[-30:])
        raise SystemExit(
            f"Planner failed for {transition} {kind} {ordinal}\n{tail}"
        )
    payload = json.loads(timing_output.read_text(encoding="utf-8"))
    components = payload["component_seconds"]
    return {
        "transition": transition,
        "kind": kind,
        "ordinal": ordinal,
        "process_wall_seconds": process_wall_seconds,
        "planner_main_seconds": float(payload["planner_main_seconds"]),
        "selected_floors": payload["selected_floors"],
        **{name: float(components[name]) for name in COMPONENTS},
    }


def _metric_summary(records: list[dict[str, Any]], metric: str) -> dict[str, float]:
    values = [float(record[metric]) for record in records]
    return {
        "mean_seconds": statistics.fmean(values),
        "max_seconds": max(values),
        "min_seconds": min(values),
        "stdev_seconds": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "transition",
        "kind",
        "ordinal",
        *COMPONENTS,
        "planner_main_seconds",
        "process_wall_seconds",
        "selected_floors",
    ]
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            csv_row = dict(row)
            csv_row["selected_floors"] = ",".join(
                str(item) for item in row["selected_floors"]
            )
            writer.writerow(csv_row)


def main() -> None:
    args = _parse_args()
    if args.runs <= 0 or args.warmups < 0:
        raise SystemExit("runs must be positive and warmups must be nonnegative")
    epoch0_dir = args.common_epoch0_root / "epoch_000_mode0_probe"
    _validate_history(epoch0_dir)
    _validate_history(args.epoch1_dir)
    if not args.train_file.is_file():
        raise SystemExit(f"Training parquet does not exist at {args.train_file}")
    if args.output_dir.exists():
        raise SystemExit(f"Output directory already exists at {args.output_dir}")
    args.output_dir.mkdir(parents=True)

    transitions = {
        "epoch0_to_epoch1": [epoch0_dir],
        "epoch1_to_epoch2": [epoch0_dir, args.epoch1_dir],
    }
    records: list[dict[str, Any]] = []
    for transition, histories in transitions.items():
        for ordinal in range(1, args.warmups + 1):
            print(f"Running {transition} warmup {ordinal}", flush=True)
            records.append(
                _run_once(
                    transition,
                    histories,
                    ordinal,
                    False,
                    args.output_dir,
                    args.train_file,
                )
            )
        for ordinal in range(1, args.runs + 1):
            print(f"Running {transition} measured run {ordinal}", flush=True)
            records.append(
                _run_once(
                    transition,
                    histories,
                    ordinal,
                    True,
                    args.output_dir,
                    args.train_file,
                )
            )

    measured = [record for record in records if record["kind"] == "run"]
    per_transition: dict[str, Any] = {}
    for transition in transitions:
        transition_records = [
            record for record in measured if record["transition"] == transition
        ]
        per_transition[transition] = {
            metric: _metric_summary(transition_records, metric)
            for metric in (*COMPONENTS, "planner_main_seconds", "process_wall_seconds")
        }

    across_transition_means = {}
    for component in COMPONENTS:
        values = [
            per_transition[transition][component]["mean_seconds"]
            for transition in transitions
        ]
        across_transition_means[component] = {
            "mean_seconds": statistics.fmean(values),
            "max_seconds": max(values),
        }

    summary = {
        "schema_version": 1,
        "configuration": {
            "mode": "Planned floor4",
            "runs_per_transition": args.runs,
            "warmups_per_transition": args.warmups,
            "common_epoch0_root": str(args.common_epoch0_root),
            "epoch1_dir": str(args.epoch1_dir),
            "floor_kv_caps": {
                "4": 133120,
                "8": 262656,
                "16": 380800,
            },
        },
        "per_transition": per_transition,
        "across_transition_means": across_transition_means,
        "notes": [
            "Each repetition launches a fresh Python process.",
            "Warmup runs are excluded from all reported statistics.",
            "The operating system page cache is not cleared between runs.",
            "Epoch2 planning uses the accumulated epoch0 and epoch1 history.",
            "Measured planner runs read only compact offline planning history files.",
            "Compact history construction runs once after each epoch and is excluded from planner timing.",
        ],
    }
    _write_csv(args.output_dir / "runs.csv", records)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    print("\nMeasured component times in seconds")
    for transition in transitions:
        print(transition)
        for component in COMPONENTS:
            values = per_transition[transition][component]
            print(
                f"  {component:20s} mean {values['mean_seconds']:.4f} "
                f"max {values['max_seconds']:.4f}"
            )
    print("Across the two transition means")
    for component in COMPONENTS:
        values = across_transition_means[component]
        print(
            f"  {component:20s} mean {values['mean_seconds']:.4f} "
            f"max {values['max_seconds']:.4f}"
        )
    print(f"Results written to {args.output_dir}")


if __name__ == "__main__":
    main()
