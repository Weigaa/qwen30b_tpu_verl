#!/usr/bin/env python3
"""Summarize rollout time, shrink tail, and decode-step utilization logs.

The script works in two modes:
  * Existing logs: extracts rollout time and shrink-to-8 timing markers.
  * Logs produced with VLLM_ASCEND_ELASTIC_UTIL_LOG=1: additionally parses
    "Elastic decode util bucket" lines every N decode steps.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import re
import statistics
from pathlib import Path
from typing import Iterable

ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
TS_RE = re.compile(r"(?:(?P<date>20\d\d-\d\d-\d\d)|(?P<md>\d\d-\d\d)) (?P<hms>\d\d:\d\d:\d\d)(?:,(?P<ms>\d+))?")
FILE_DATE_RE = re.compile(r"(20\d{6})")
ROLLOUT_RE = re.compile(r"rollout_output_time_s:\s*([0-9.]+)")
ACTIVE_RE = re.compile(r"Elastic shrink delayed .*active_count=(\d+).*dp_world_size=(\d+)")
SHRINK_START_RE = re.compile(r"Elastic shrink payload source group: .*? active_ranks=\[([^\]]*)\]")
SHRINK_DONE_RE = re.compile(r"Elastic parallel shrink rpc done: .*? active_ranks=\[([^\]]*)\].*total_ms=([0-9.]+)")
DETACH_RE = re.compile(r"Elastic parallel detach done: .*? active_ranks=\[([^\]]*)\].*total_ms=([0-9.]+)")
UTIL_RE = re.compile(
    r"Elastic decode util bucket: .*global_rank=(?P<rank>-?\d+) "
    r"step_bucket=(?P<bucket>\d+) step=(?P<step>\d+) "
    r"active_count=(?P<active>\d+) compute_world_size=(?P<world>\d+) "
    r"util=(?P<util>[0-9.]+).*active_ranks=(?P<ranks>None|\[[^\]]*\]).*"
    r"local_output_tokens=(?P<tokens>\d+) "
    r"elapsed_s=(?P<elapsed>[0-9.]+)"
)


def strip_ansi(line: str) -> str:
    return ANSI_RE.sub("", line)


def parse_ts(line: str, default_year: int | None = None) -> dt.datetime | None:
    m = TS_RE.search(line)
    if not m:
        return None
    ms = (m.group("ms") or "0")[:6].ljust(6, "0")
    if m.group("date"):
        date_s = m.group("date")
    elif default_year is not None:
        date_s = f"{default_year}-{m.group('md')}"
    else:
        return None
    return dt.datetime.strptime(
        f"{date_s} {m.group('hms')}.{ms}", "%Y-%m-%d %H:%M:%S.%f")


def parse_rank_list(text: str) -> list[int]:
    text = text.strip()
    if not text:
        return []
    return [int(x.strip()) for x in text.split(',') if x.strip()]


def parse_npu_csv(path: Path | None) -> list[dict]:
    if path is None:
        return []
    rows = []
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                rows.append({
                    "ts": dt.datetime.fromisoformat(row["ts_iso"]),
                    "phy_id": int(row["phy_id"]),
                    "aicore_pct": float(row["aicore_pct"]),
                    "hbm_used_mb": int(row["hbm_used_mb"]),
                })
            except Exception:
                continue
    return rows


def summarize_log(path: Path) -> dict:
    rollout_s: float | None = None
    rollout_ts: dt.datetime | None = None
    first_active8_ts: dt.datetime | None = None
    first_shrink8_ts: dt.datetime | None = None
    first_shrink8_ranks: list[int] | None = None
    max_shrink8_done_ms: float | None = None
    max_detach8_ms: float | None = None
    active_events: list[tuple[dt.datetime | None, int, int]] = []
    util_rows: list[dict] = []

    file_date = FILE_DATE_RE.search(path.name)
    default_year = int(file_date.group(1)[:4]) if file_date else None

    with path.open("r", errors="ignore") as f:
        for raw in f:
            line = strip_ansi(raw)
            ts = parse_ts(line, default_year)
            if m := ROLLOUT_RE.search(line):
                rollout_s = float(m.group(1))
                rollout_ts = ts
            if m := ACTIVE_RE.search(line):
                active = int(m.group(1))
                world = int(m.group(2))
                active_events.append((ts, active, world))
                if active == 8 and first_active8_ts is None:
                    first_active8_ts = ts
            if m := SHRINK_START_RE.search(line):
                ranks = parse_rank_list(m.group(1))
                if len(ranks) == 8 and first_shrink8_ts is None:
                    first_shrink8_ts = ts
                    first_shrink8_ranks = ranks
            if m := SHRINK_DONE_RE.search(line):
                ranks = parse_rank_list(m.group(1))
                if len(ranks) == 8:
                    value = float(m.group(2))
                    max_shrink8_done_ms = value if max_shrink8_done_ms is None else max(max_shrink8_done_ms, value)
            if m := DETACH_RE.search(line):
                ranks = parse_rank_list(m.group(1))
                if len(ranks) == 8:
                    value = float(m.group(2))
                    max_detach8_ms = value if max_detach8_ms is None else max(max_detach8_ms, value)
            if m := UTIL_RE.search(line):
                util_rows.append({
                    "rank": int(m.group("rank")),
                    "bucket": int(m.group("bucket")),
                    "step": int(m.group("step")),
                    "active": int(m.group("active")),
                    "world": int(m.group("world")),
                    "util": float(m.group("util")),
                    "tokens": int(m.group("tokens")),
                    "elapsed": float(m.group("elapsed")),
                    "active_ranks": parse_rank_list(m.group("ranks")[1:-1])
                    if m.group("ranks") != "None" else None,
                    "ts": ts,
                })

    tail_after_active8_s = None
    if rollout_ts is not None and first_active8_ts is not None:
        tail_after_active8_s = (rollout_ts - first_active8_ts).total_seconds()
    tail_after_shrink8_start_s = None
    if rollout_ts is not None and first_shrink8_ts is not None:
        tail_after_shrink8_start_s = (rollout_ts - first_shrink8_ts).total_seconds()
    tail_after_shrink8_done_s = None
    if tail_after_shrink8_start_s is not None and max_shrink8_done_ms is not None:
        tail_after_shrink8_done_s = max(0.0, tail_after_shrink8_start_s - max_shrink8_done_ms / 1000.0)

    return {
        "path": path,
        "rollout_s": rollout_s,
        "rollout_ts": rollout_ts,
        "first_active8_ts": first_active8_ts,
        "first_shrink8_ts": first_shrink8_ts,
        "first_shrink8_ranks": first_shrink8_ranks,
        "max_detach8_ms": max_detach8_ms,
        "max_shrink8_done_ms": max_shrink8_done_ms,
        "tail_after_active8_s": tail_after_active8_s,
        "tail_after_shrink8_start_s": tail_after_shrink8_start_s,
        "tail_after_shrink8_done_s": tail_after_shrink8_done_s,
        "active_events": active_events,
        "util_rows": util_rows,
    }


def fmt(value, unit="") -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.3f}{unit}"
    return f"{value}{unit}"


def _mean(values: list[float]) -> float | None:
    return statistics.mean(values) if values else None


def _samples_between(samples: list[dict], start: dt.datetime | None,
                     end: dt.datetime | None) -> list[dict]:
    if end is None:
        return []
    if start is None:
        return [s for s in samples if s["ts"] <= end]
    return [s for s in samples if start < s["ts"] <= end]


def _avg_aicore(samples: list[dict], phy_ids: set[int] | None = None) -> float | None:
    vals = [s["aicore_pct"] for s in samples
            if phy_ids is None or s["phy_id"] in phy_ids]
    return _mean(vals)


def print_summary(summary: dict, max_buckets: int,
                  npu_samples: list[dict] | None = None) -> None:
    path = summary["path"]
    print(f"\n== {path} ==")
    print(f"rollout_output_time_s={fmt(summary['rollout_s'])}")
    print(f"first_active8_ts={summary['first_active8_ts'] or 'N/A'}")
    print(f"first_shrink8_ts={summary['first_shrink8_ts'] or 'N/A'}")
    print(f"detach_to_8_max_ms={fmt(summary['max_detach8_ms'])}")
    print(f"shrink_to_8_rpc_max_ms={fmt(summary['max_shrink8_done_ms'])}")
    print(f"tail_after_active8_s={fmt(summary['tail_after_active8_s'])}")
    print(f"tail_after_shrink8_start_s={fmt(summary['tail_after_shrink8_start_s'])}")
    print(f"tail_after_shrink8_done_s={fmt(summary['tail_after_shrink8_done_s'])}")

    rows = summary["util_rows"]
    if not rows:
        print("util_buckets=N/A (rerun with VLLM_ASCEND_ELASTIC_UTIL_LOG=1)")
        if npu_samples:
            all_avg = _avg_aicore(npu_samples)
            print(f"npu_aicore_over_log_avg={fmt(all_avg)}")
        return

    # Deduplicate duplicated rank reports by bucket/rank; then aggregate per bucket.
    by_bucket: dict[int, list[dict]] = {}
    seen: set[tuple[int, int]] = set()
    for row in rows:
        key = (row["bucket"], row["rank"])
        if key in seen:
            continue
        seen.add(key)
        by_bucket.setdefault(row["bucket"], []).append(row)

    print("util_buckets: bucket,mean_real_req_ratio,min_real_req_ratio,max_real_req_ratio,allocated_npu_aicore,real_req_npu_aicore,allocated_no_req_npu_aicore,released_npu_aicore,real_req/world samples")
    prev_bucket_ts = None
    shrink8_allocated = (set(summary.get("first_shrink8_ranks") or [])
                         or None)
    for bucket in sorted(by_bucket)[:max_buckets]:
        bucket_rows = by_bucket[bucket]
        utils = [r["util"] for r in bucket_rows]
        bucket_ts = min((r["ts"] for r in bucket_rows if r["ts"] is not None),
                        default=None)
        representative = bucket_rows[0]
        world = int(representative["world"])
        active_ranks = next((r["active_ranks"] for r in bucket_rows
                             if r.get("active_ranks") is not None), None)
        real_req_set = set(active_ranks) if active_ranks is not None else None
        bucket_samples = _samples_between(npu_samples or [], prev_bucket_ts,
                                          bucket_ts)
        all_ids = {s["phy_id"] for s in bucket_samples}
        if not all_ids:
            all_ids = set(range(world))
        if world == 8 and shrink8_allocated is not None:
            allocated_set = set(shrink8_allocated)
        elif real_req_set is not None and len(real_req_set) == world:
            allocated_set = set(real_req_set)
        else:
            # Baseline/no-shrink keeps the original 16-rank group allocated even
            # after some ranks only run dummy/no-request work.
            allocated_set = set(range(world)) if world != 16 else set(range(16))
        allocated_avg = _avg_aicore(bucket_samples, allocated_set)
        real_req_avg = _avg_aicore(bucket_samples, real_req_set)
        allocated_no_req_avg = None
        if real_req_set is not None:
            allocated_no_req_avg = _avg_aicore(bucket_samples,
                                               allocated_set - real_req_set)
        released_avg = _avg_aicore(bucket_samples, all_ids - allocated_set)
        samples = ";".join(
            f"r{r['rank']}:{r['active']}/{r['world']}" for r in bucket_rows[:8])
        if len(bucket_rows) > 8:
            samples += f";...(+{len(bucket_rows)-8})"
        print(
            f"{bucket},{statistics.mean(utils):.6f},{min(utils):.6f},{max(utils):.6f},"
            f"{fmt(allocated_avg)},{fmt(real_req_avg)},{fmt(allocated_no_req_avg)},"
            f"{fmt(released_avg)},{samples}")
        prev_bucket_ts = bucket_ts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("logs", nargs="+", type=Path)
    parser.add_argument("--max-buckets", type=int, default=40)
    parser.add_argument("--npu-csv", type=Path, default=None,
                        help="CSV generated by internal/sample_npu_util.py")
    args = parser.parse_args()
    npu_samples = parse_npu_csv(args.npu_csv)
    if args.npu_csv:
        print(f"npu_csv={args.npu_csv} samples={len(npu_samples)}")
    for log in args.logs:
        print_summary(summarize_log(log), args.max_buckets, npu_samples)


if __name__ == "__main__":
    main()
