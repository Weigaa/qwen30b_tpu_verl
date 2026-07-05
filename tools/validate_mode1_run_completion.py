#!/usr/bin/env python3
"""Validate that a mode1 smoke-test log completed the requested rollouts."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


START_RE = re.compile(r"rollout_generate_start step=(\d+)")
DONE_RE = re.compile(r"rollout_generate_done step=(\d+)")
METRIC_RE = re.compile(r"rollout_output_time_s:\s*([0-9]+(?:\.[0-9]+)?)")
TRAINING_STEPS_RE = re.compile(r"Total training steps:\s*(\d+)")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fail if a mode1 log exits before all expected rollouts finish.")
    parser.add_argument("log", type=Path)
    parser.add_argument("--expected-steps", type=int, required=True)
    parser.add_argument("--require-run-done", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    text = args.log.read_text(errors="replace")

    starts = [int(m.group(1)) for m in START_RE.finditer(text)]
    dones = [int(m.group(1)) for m in DONE_RE.finditer(text)]
    metrics = [float(m.group(1)) for m in METRIC_RE.finditer(text)]
    configured_steps = [int(m.group(1)) for m in TRAINING_STEPS_RE.finditer(text)]

    start_set = set(starts)
    done_set = set(dones)
    missing_starts = [
        step for step in range(1, args.expected_steps + 1) if step not in start_set
    ]
    missing_dones = [
        step for step in range(1, args.expected_steps + 1) if step not in done_set
    ]
    started_without_done = sorted(start_set - done_set)

    reasons: list[str] = []
    if len(metrics) < args.expected_steps:
        reasons.append(f"rollout_metrics={len(metrics)}<{args.expected_steps}")
    if missing_starts:
        reasons.append(
            "missing_start_steps=" + ",".join(map(str, missing_starts)))
    if missing_dones:
        reasons.append("missing_done_steps=" + ",".join(map(str, missing_dones)))
    if started_without_done:
        reasons.append(
            "started_without_done=" + ",".join(map(str, started_without_done)))
    if configured_steps and configured_steps[-1] < args.expected_steps:
        reasons.append(
            f"configured_training_steps={configured_steps[-1]}<{args.expected_steps}")
    if args.require_run_done and "[ray_timing] taskrunner_run_done" not in text:
        reasons.append("missing_taskrunner_run_done")

    if reasons:
        print(
            "[mode1 completion validation] status=fail "
            f"expected_steps={args.expected_steps} "
            f"starts={starts} dones={dones} metrics={len(metrics)} "
            + " ".join(reasons),
            file=sys.stderr,
        )
        return 1

    print(
        "[mode1 completion validation] status=ok "
        f"expected_steps={args.expected_steps} metrics={len(metrics)} "
        f"rollout_times={','.join(f'{x:.3f}' for x in metrics[:args.expected_steps])}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
