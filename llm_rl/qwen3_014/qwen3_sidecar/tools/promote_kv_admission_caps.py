#!/usr/bin/env python3
"""Keep conservative planner caps while restoring measured runtime capacity."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


VALID_FLOORS = {2, 4, 8, 16}


def parse_caps(value: str) -> dict[int, float]:
    caps: dict[int, float] = {}
    for item in value.split(","):
        floor_text, cap_text = item.split(":", 1)
        floor = int(floor_text)
        cap = float(cap_text)
        if floor not in VALID_FLOORS or cap <= 0 or cap % 128 != 0:
            raise ValueError(f"invalid physical KV cap {item!r}")
        caps[floor] = cap
    if not caps:
        raise ValueError("physical caps must not be empty")
    return caps


def load_steps(path: Path) -> list[dict[str, Any]]:
    steps = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(steps, list) or not steps:
        raise ValueError(f"{path}: expected a nonempty step list")
    return steps


def adjusted_peak(step: dict[str, Any]) -> float:
    if "max_adjusted_rank_peak_tokens" in step:
        return float(step["max_adjusted_rank_peak_tokens"])
    loads = step.get("rank_adjusted_peak_loads")
    if not isinstance(loads, dict) or not loads:
        raise ValueError("step has no adjusted KV peak")
    return max(float(value) for value in loads.values())


def promote_step(step: dict[str, Any], physical_caps: dict[int, float]) -> None:
    floor = int(step["selected_floor"])
    if floor not in physical_caps:
        raise ValueError(f"physical capacity missing for selected floor {floor}")
    peak = adjusted_peak(step)
    admission = float(step.get("kv_admission_cap", step["kv_cap"]))
    physical = float(physical_caps[floor])
    if peak > admission + 1e-6:
        raise ValueError(
            f"floor {floor} adjusted peak {peak:.2f} exceeds "
            f"admission cap {admission:.2f}"
        )
    if admission > physical + 1e-6:
        raise ValueError(
            f"floor {floor} admission cap {admission:.2f} exceeds "
            f"physical cap {physical:.2f}"
        )
    step["kv_admission_cap"] = admission
    step["kv_cap"] = physical
    step["kv_admission_headroom_tokens"] = admission - peak
    step["kv_physical_headroom_tokens"] = physical - peak
    step["kv_admission_policy"] = (
        "reserved_planner_headroom_with_full_runtime_capacity"
    )


def write_steps(path: Path, steps: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(steps, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def promote_files(plan_path: Path, summary_path: Path, caps_text: str) -> None:
    physical_caps = parse_caps(caps_text)
    plan = load_steps(plan_path)
    summary = load_steps(summary_path)
    if len(plan) != len(summary):
        raise ValueError("plan and summary step counts differ")
    for plan_step, summary_step in zip(plan, summary, strict=True):
        if int(plan_step["step"]) != int(summary_step["step"]):
            raise ValueError("plan and summary step indices differ")
        promote_step(plan_step, physical_caps)
        promote_step(summary_step, physical_caps)
    write_steps(plan_path, plan)
    write_steps(summary_path, summary)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--physical-caps", required=True)
    args = parser.parse_args()
    promote_files(args.plan, args.summary, args.physical_caps)
    summary = load_steps(args.summary)
    print(
        "[KV admission] floors="
        + ",".join(str(int(step["selected_floor"])) for step in summary)
        + " physical_headroom="
        + ",".join(
            f"{float(step['kv_physical_headroom_tokens']):.1f}" for step in summary
        )
    )


if __name__ == "__main__":
    main()
