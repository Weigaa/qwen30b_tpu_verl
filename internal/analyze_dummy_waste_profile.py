#!/usr/bin/env python3
"""Summarize low-intrusion dummy waste profiler markers.

Expected markers:
  vllm_dummy_forward rank=... ...
  vllm_dummy_moe_compute rank=... layer=... ...

The script scans chrome-trace-like JSON files produced by torch_npu.profiler
and sums complete events by rank. Durations are reported in seconds.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

RANK_RE = re.compile(r"\brank=(-?\d+)\b")


def iter_json_files(root: Path) -> Iterable[Path]:
    for pattern in ("*.json", "*.json.gz"):
        yield from root.rglob(pattern)


def load_json(path: Path) -> Any:
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8", errors="ignore") as f:
            return json.load(f)
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        return json.load(f)


def iter_events(obj: Any) -> Iterable[dict[str, Any]]:
    if isinstance(obj, dict):
        events = obj.get("traceEvents")
        if isinstance(events, list):
            for ev in events:
                if isinstance(ev, dict):
                    yield ev
        elif "name" in obj:
            yield obj
    elif isinstance(obj, list):
        for ev in obj:
            if isinstance(ev, dict):
                yield ev


def event_duration_us(ev: dict[str, Any]) -> float:
    # Chrome trace complete events usually use dur in microseconds.
    dur = ev.get("dur")
    if dur is None:
        dur = ev.get("duration")
    try:
        return float(dur or 0.0)
    except (TypeError, ValueError):
        return 0.0


def event_rank(name: str) -> int | None:
    m = RANK_RE.search(name)
    if not m:
        return None
    return int(m.group(1))


def _latest_profile_dir(root: Path) -> Path:
    candidates = [p for p in root.iterdir() if p.is_dir()] if root.exists() else []
    if not candidates:
        raise SystemExit(f"No profile directories found under {root}")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "profile_dir",
        nargs="?",
        type=Path,
        help=(
            "Profiler directory. Defaults to the latest directory under "
            "$WJ_RECORDS_DIR/dummy_waste_profiles, where WJ_RECORDS_DIR "
            "defaults to /home/sharedata/wj_records."),
    )
    args = ap.parse_args()
    if args.profile_dir is None:
        records_dir = Path(os.getenv("WJ_RECORDS_DIR", "/home/sharedata/wj_records"))
        args.profile_dir = _latest_profile_dir(records_dir / "dummy_waste_profiles")

    totals = defaultdict(lambda: defaultdict(float))
    counts = defaultdict(lambda: defaultdict(int))
    files = list(iter_json_files(args.profile_dir))
    parsed_files = 0

    for path in files:
        try:
            obj = load_json(path)
        except Exception:
            continue
        parsed_files += 1
        for ev in iter_events(obj):
            name = str(ev.get("name", ""))
            if ("vllm_dummy_forward" not in name
                    and "vllm_dummy_moe_compute" not in name
                    and "vllm_dummy_moe_selected" not in name):
                continue
            dur_us = event_duration_us(ev)
            if dur_us <= 0:
                continue
            rank = event_rank(name)
            if rank is None:
                # Fall back to pid when rank metadata is missing.
                try:
                    rank = int(ev.get("pid", -1))
                except Exception:
                    rank = -1
            if "vllm_dummy_forward" in name:
                key = "dummy_forward_s"
            else:
                key = "dummy_moe_compute_s"
            totals[rank][key] += dur_us / 1e6
            counts[rank][key] += 1

    print(f"profile_dir={args.profile_dir}")
    print(f"json_files={len(files)} parsed_json_files={parsed_files}")
    if not totals:
        print(
            "No dummy profiler markers found. Check that "
            "VLLM_ASCEND_DUMMY_WASTE_PROFILE_MARKERS=1, "
            "global_profiler.tool=npu, and "
            "actor_rollout_ref.actor.profiler.tool_config.npu.contents=[mstx], "
            "level=level_none, and analysis=True were enabled. If contents=[npu] "
            "or [cpu], the profiler may export kernel/op traces but omit or bury "
            "MSTX marker ranges.")
        return

    cluster_forward = sum(v.get("dummy_forward_s", 0.0) for v in totals.values())
    cluster_moe = sum(v.get("dummy_moe_compute_s", 0.0) for v in totals.values())
    cluster_waste = max(0.0, cluster_forward - cluster_moe)
    print("\ncluster_summary")
    print(f"dummy_forward_s={cluster_forward:.6f}")
    print(f"dummy_moe_compute_s={cluster_moe:.6f}")
    print(f"dummy_waste_s={cluster_waste:.6f}")
    print(f"moe_compute_ratio={cluster_moe / cluster_forward if cluster_forward else 0.0:.6f}")
    print(f"waste_ratio={cluster_waste / cluster_forward if cluster_forward else 0.0:.6f}")

    print("\nper_rank")
    print("rank,forward_calls,moe_compute_calls,dummy_forward_s,dummy_moe_compute_s,dummy_waste_s,moe_compute_ratio,waste_ratio")
    for rank in sorted(totals):
        fwd = totals[rank].get("dummy_forward_s", 0.0)
        moe = totals[rank].get("dummy_moe_compute_s", 0.0)
        waste = max(0.0, fwd - moe)
        print(
            f"{rank},{counts[rank].get('dummy_forward_s', 0)},"
            f"{counts[rank].get('dummy_moe_compute_s', 0)},"
            f"{fwd:.6f},{moe:.6f},{waste:.6f},"
            f"{moe / fwd if fwd else 0.0:.6f},"
            f"{waste / fwd if fwd else 0.0:.6f}"
        )


if __name__ == "__main__":
    main()
