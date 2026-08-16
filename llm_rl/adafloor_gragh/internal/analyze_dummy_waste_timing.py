#!/usr/bin/env python3
"""Summarize low-intrusion "Dummy waste timing" log lines.

This is the fallback path when full torch_npu profiler traces are too large or
do not export MSTX marker ranges. It consumes normal training logs generated
with:

  VLLM_ASCEND_DUMMY_WASTE_TIMING=1
  VLLM_ASCEND_DUMMY_WASTE_TIMING_SYNC=0

The reported times are host enqueue intervals, not synchronized device kernel
durations. They are intended for low-overhead waste accounting.
"""

from __future__ import annotations

import argparse
import re
from collections import defaultdict
from pathlib import Path

LINE_RE = re.compile(r"Dummy waste timing:")
FIELD_RE = re.compile(r"(\w+)=([^\s]+)")


def _latest_log() -> Path:
    candidates = [
        p for p in Path(".").glob("wjeagerqwen30b-a3b-with_draft_breakdown_*.txt")
        if p.is_file()
    ]
    if not candidates:
        raise SystemExit("No rollout log found; pass one or more log paths.")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _to_float(fields: dict[str, str], key: str) -> float:
    try:
        return float(fields.get(key, "0"))
    except ValueError:
        return 0.0


def _to_int(fields: dict[str, str], key: str, default: int = 0) -> int:
    try:
        return int(float(fields.get(key, str(default))))
    except ValueError:
        return default


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("logs", nargs="*", type=Path)
    args = ap.parse_args()
    logs = args.logs or [_latest_log()]

    totals = defaultdict(lambda: defaultdict(float))
    counts = defaultdict(int)
    parsed = 0

    for path in logs:
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if not LINE_RE.search(line):
                    continue
                fields = dict(FIELD_RE.findall(line))
                rank = _to_int(fields, "rank", -1)
                counts[rank] += 1
                parsed += 1
                totals[rank]["dummy_forward_s"] += (
                    _to_float(fields, "dummy_wall_ms") / 1e3)
                totals[rank]["dummy_moe_effective_s"] += (
                    _to_float(fields, "dummy_moe_effective_ms") / 1e3)
                totals[rank]["dummy_waste_s"] += (
                    _to_float(fields, "dummy_wasted_ms") / 1e3)
                totals[rank]["dummy_moe_wall_s"] += (
                    _to_float(fields, "dummy_moe_wall_ms") / 1e3)
                totals[rank]["dummy_moe_layers"] += _to_int(
                    fields, "dummy_moe_layers")
                totals[rank]["dummy_moe_selected_layers"] += _to_int(
                    fields, "dummy_moe_selected_layers")

    print("logs=" + ",".join(str(p) for p in logs))
    print(f"dummy_timing_lines={parsed}")
    if not parsed:
        print("No Dummy waste timing lines found.")
        return

    cluster_forward = sum(v["dummy_forward_s"] for v in totals.values())
    cluster_moe = sum(v["dummy_moe_effective_s"] for v in totals.values())
    cluster_waste = sum(v["dummy_waste_s"] for v in totals.values())
    print("\ncluster_summary")
    print(f"dummy_forward_s={cluster_forward:.6f}")
    print(f"dummy_moe_effective_s={cluster_moe:.6f}")
    print(f"dummy_waste_s={cluster_waste:.6f}")
    print(f"moe_effective_ratio={cluster_moe / cluster_forward if cluster_forward else 0.0:.6f}")
    print(f"waste_ratio={cluster_waste / cluster_forward if cluster_forward else 0.0:.6f}")

    print("\nper_rank")
    print(
        "rank,calls,dummy_forward_s,dummy_moe_effective_s,dummy_waste_s,"
        "moe_effective_ratio,waste_ratio,moe_layers,moe_selected_layers")
    for rank in sorted(totals):
        fwd = totals[rank]["dummy_forward_s"]
        moe = totals[rank]["dummy_moe_effective_s"]
        waste = totals[rank]["dummy_waste_s"]
        print(
            f"{rank},{counts[rank]},{fwd:.6f},{moe:.6f},{waste:.6f},"
            f"{moe / fwd if fwd else 0.0:.6f},"
            f"{waste / fwd if fwd else 0.0:.6f},"
            f"{int(totals[rank]['dummy_moe_layers'])},"
            f"{int(totals[rank]['dummy_moe_selected_layers'])}")


if __name__ == "__main__":
    main()
