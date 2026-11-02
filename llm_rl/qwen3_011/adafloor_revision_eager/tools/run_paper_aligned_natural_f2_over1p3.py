#!/usr/bin/env python3
"""Run the missing paper-aligned frozen Natural-F2 x1.3 NPU arm.

The runtime launcher is imported from the audited paper-aligned runner. This
script adds only the overprediction scenario and preserves a failed safety
audit while retaining functionally complete measurements.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import subprocess
import sys


HERE = Path(__file__).resolve()
PACKAGE = HERE.parents[1]
RUNNER_PATH = PACKAGE / "tools/run_paper_aligned_natural_f2_forecast.py"
GENERATOR = PACKAGE / "tools/create_revision_forecast_scenarios.py"
SUMMARY_PATH = PACKAGE / "analysis_eval/summarize_paper_aligned_forecast.py"
SCENARIO = "over_1p3"


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.output_root.resolve()
    if not root.is_relative_to(Path("/workspace/adafloor_revision_results")):
        raise ValueError("output must be under /workspace/adafloor_revision_results")
    if root.exists():
        raise FileExistsError("refusing to overwrite existing evidence")

    runner = load(RUNNER_PATH, "paper_aligned_runner_x1p3")
    summary = load(SUMMARY_PATH, "paper_aligned_summary_x1p3")
    root.mkdir(parents=True)
    try:
        runner.validate_history_and_code(root)
        subprocess.run([
            sys.executable, str(GENERATOR),
            "--history", str(runner.COMMON / "offline_planning_history.json"),
            "--output-root", str(root / "scenarios"),
            "--mixed-seed", "505", "--regular-artifacts",
        ], check=True)
        runner.launch(root, SCENARIO, True)
        write_json(root / "protocol.json", {
            "status": "PREFLIGHT_PASS", "variant": "natural_f2",
            "seed": 505, "frozen": True, "steps": 5,
            "normal_reused": True, "npu_scenarios": [SCENARIO],
            "cpu_scenarios": ["under_0p7", "mixed"],
            "cross_step_repair": {
                "enabled_in_cpu_plan": True,
                "max_swaps": 8,
                "repair_candidate_limit": 8,
            },
        })
        runner.launch(root, SCENARIO, False)
        try:
            runner.audit_run(root, SCENARIO)
            audit_error = None
        except RuntimeError as error:
            audit_error = str(error)
            audit = json.loads((root / "runs" / SCENARIO / "audit.json").read_text())
            if (audit["oom"] or audit["aborted"]
                    or audit["response_counts"] != [512] * 5
                    or audit["length_counts"] != [512] * 5
                    or any(not e.startswith("preemptions=") for e in audit["errors"])):
                raise

        run = root / "runs" / SCENARIO / "natural_f2/epoch_001_mode1_natural"
        row = summary.audit_epoch(run, True)
        normal = summary.audit_epoch(summary.NORMAL, False)
        write_json(root / "measurement_completion.json", {
            "measurement_complete": bool(row.get("measurement_complete")),
            "safety_status": row.get("status"),
            "preemption_count": row.get("preemption_count"),
            "oom": row.get("oom"), "audit_error": audit_error,
        })
        write_json(root / "analysis_summary.json", {
            "status": "FAIL" if row["status"] == "FAIL" else row["status"],
            "normal_context": normal,
            "over_1p3": row,
            "scope": "Paper-aligned frozen Natural-F2 x1.3 NPU arm. Normal is reused context. The x1.3 cap is forecast-derived.",
        })
        cpu_output = root / "paper_cpu_sensitivity.json"
        subprocess.run([
            sys.executable,
            str(PACKAGE / "analysis_eval/analyze_revision_forecast_sensitivity.py"),
            "--scenario-root", str(root / "scenarios"),
            "--output", str(cpu_output),
        ], check=True)
        write_json(root / "completion.json", {
            "status": "MEASUREMENT_COMPLETE_SAFETY_STATUS_PRESERVED",
            "measurement_complete": bool(row.get("measurement_complete")),
            "safety_status": row.get("status"),
            "common_epoch0_preserved": True,
        })
        print(json.dumps({
            "status": "MEASUREMENT_COMPLETE_SAFETY_STATUS_PRESERVED",
            "measurement_complete": row.get("measurement_complete"),
            "safety_status": row.get("status"),
            "preemptions": row.get("preemption_count"),
        }, sort_keys=True))
        return 0
    except Exception as error:
        write_json(root / "failure.json", {"status": "FAIL", "error": str(error)})
        raise


if __name__ == "__main__":
    raise SystemExit(main())
