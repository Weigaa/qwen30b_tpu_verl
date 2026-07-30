#!/usr/bin/env python3
"""Summarize mode=3/4/5 double-buffer timing logs by shrink stage."""

from __future__ import annotations

import argparse
import math
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path


FIELD_RE = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)=([^ \n\r]+)")
MODE_RE = re.compile(r"Mode([345]) timing")


def _to_number(value: str):
    try:
        if "." in value or "e" in value.lower():
            return float(value)
        return int(value)
    except ValueError:
        return value


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    xs = sorted(values)
    k = (len(xs) - 1) * q / 100.0
    lo = math.floor(k)
    hi = math.ceil(k)
    if lo == hi:
        return xs[lo]
    return xs[lo] * (hi - k) + xs[hi] * (k - lo)


def _fmt(value: float) -> str:
    if math.isnan(value):
        return "nan"
    return f"{value:.3f}"


def parse_log(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(errors="ignore") as f:
        for line_no, line in enumerate(f, 1):
            mode_m = MODE_RE.search(line)
            if not mode_m:
                continue
            row = {"line_no": line_no, "mode": int(mode_m.group(1))}
            for key, value in FIELD_RE.findall(line):
                row[key] = _to_number(value)
            if "stage" not in row or "layer" not in row:
                continue
            submit_us = float(row.get("prefetch_submit_us", -1.0))
            dev_ms = float(row.get("prefetch_dev_ms", -1.0))
            if submit_us >= 0 and dev_ms >= 0:
                row["prefetch_total_ms"] = submit_us / 1000.0 + dev_ms
            else:
                row["prefetch_total_ms"] = -1.0
            rows.append(row)
    return rows


def drop_cold_start(rows: list[dict]) -> list[dict]:
    """Drop the first timing row for each (mode, stage, layer, rank)."""
    seen: set[tuple] = set()
    kept: list[dict] = []
    for row in rows:
        key = (row.get("mode"), row.get("stage"), row.get("layer"),
               row.get("rank"))
        if key not in seen:
            seen.add(key)
            continue
        kept.append(row)
    return kept


def summarize(rows: list[dict], stages: list[int]) -> None:
    metrics = [
        ("comm_total_ms", "prefetch_total_ms"),
        ("submit_ms", "prefetch_submit_us", 1 / 1000.0),
        ("dev_ms", "prefetch_dev_ms"),
        ("local_npu_ms", "prefetch_local_npu_dev_ms"),
        ("remote_npu_ms", "prefetch_remote_npu_dev_ms"),
        ("cpu_npu_ms", "prefetch_cpu_dev_ms"),
        ("compute_ms", "current_compute_dev_ms"),
    ]

    print("stage,n,metric,p50_ms,p90_ms,mean_ms")
    for stage in stages:
        stage_rows = [r for r in rows if int(r.get("stage", -1)) == stage]
        for item in metrics:
            name, key = item[0], item[1]
            scale = item[2] if len(item) > 2 else 1.0
            vals = [
                float(r.get(key, -1.0)) * scale
                for r in stage_rows
                if float(r.get(key, -1.0)) >= 0
            ]
            if not vals:
                continue
            print(",".join([
                str(stage),
                str(len(vals)),
                name,
                _fmt(_percentile(vals, 50)),
                _fmt(_percentile(vals, 90)),
                _fmt(statistics.mean(vals)),
            ]))


def summarize_sources(rows: list[dict], stages: list[int]) -> None:
    print("\nsource_counts")
    print("stage,n,local_npu,remote_npu,cpu,statuses")
    for stage in stages:
        stage_rows = [r for r in rows if int(r.get("stage", -1)) == stage]
        if not stage_rows:
            continue
        local = Counter(int(r.get("prefetch_source_from_local_npu", -1))
                        for r in stage_rows)
        remote = Counter(int(r.get("prefetch_source_from_remote_npu", -1))
                         for r in stage_rows)
        cpu = Counter(int(r.get("prefetch_source_from_cpu", -1))
                      for r in stage_rows)
        statuses = Counter(str(r.get("prefetch_status", "unknown"))
                           for r in stage_rows)
        print(",".join([
            str(stage),
            str(len(stage_rows)),
            dict(local).__repr__(),
            dict(remote).__repr__(),
            dict(cpu).__repr__(),
            dict(statuses).__repr__(),
        ]))


def summarize_outliers(rows: list[dict], threshold_ms: float) -> None:
    print(f"\noutliers_ge_{threshold_ms:g}ms")
    print("metric,count,stage,layer,rank,value_ms,line")
    outlier_rows = []
    checks = [
        ("prefetch_total_ms", 1.0),
        ("prefetch_dev_ms", 1.0),
        ("prefetch_remote_npu_dev_ms", 1.0),
        ("prefetch_cpu_dev_ms", 1.0),
        ("current_compute_dev_ms", 1.0),
        ("prefetch_submit_us", 1 / 1000.0),
    ]
    for row in rows:
        for name, scale in checks:
            value = float(row.get(name, -1.0))
            if value < 0:
                continue
            value_ms = value * scale
            if value_ms >= threshold_ms:
                outlier_rows.append((name, row, value_ms))
    for name, row, value_ms in sorted(outlier_rows,
                                      key=lambda x: x[2],
                                      reverse=True)[:80]:
        print(",".join([
            name,
            "1",
            str(row.get("stage", -1)),
            str(row.get("layer", -1)),
            str(row.get("rank", -1)),
            _fmt(value_ms),
            str(row.get("line_no", -1)),
        ]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("log", type=Path)
    parser.add_argument("--stages", default="8,4,2,1")
    parser.add_argument("--keep-cold-start", action="store_true")
    parser.add_argument("--outlier-ms", type=float, default=1000.0)
    args = parser.parse_args()

    stages = [int(x) for x in args.stages.split(",") if x]
    rows = parse_log(args.log)
    print(f"log={args.log}")
    print(f"raw_timing_rows={len(rows)}")
    if not args.keep_cold_start:
        rows = drop_cold_start(rows)
        print(f"used_timing_rows_after_cold_start_drop={len(rows)}")
    summarize(rows, stages)
    summarize_sources(rows, stages)
    summarize_outliers(rows, args.outlier_ms)


if __name__ == "__main__":
    main()
