#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from typing import Dict, Iterable, List

ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")

# Capture common fields plus per-template kernel-compute fields.
TIMING_RE = re.compile(
    r"Mode5 timing (fused-experts|single-rank-allgather|single-dispatch): .*?"
    r"stage=(\d+) .*?"
    r"current_fetch_wall_ms=(\d+\.\d+).*?"
    r"submit_populate_us=(\d+\.\d+).*?"
    r"submit_remote_npu_us=(\d+\.\d+).*?"
    r"submit_cpu_us=(\d+\.\d+).*?"
    r"prefetch_dev_ms=(\d+\.\d+).*?"
    r"prefetch_local_npu_dev_ms=(\d+\.\d+).*?"
    r"prefetch_remote_npu_dev_ms=(\d+\.\d+).*?"
    r"prefetch_cpu_dev_ms=(\d+\.\d+).*?"
    r"current_compute_wall_ms=(\d+\.\d+).*?"
    r"current_compute_dev_ms=(\d+\.\d+).*?"
    r"(?:"
    r"fused_wall_ms=(\d+\.\d+).*?fused_dev_ms=(\d+\.\d+)"
    r"|"
    r"mlp_wall_ms=(\d+\.\d+).*?mlp_dev_ms=(\d+\.\d+)"
    r"|"
    r"fused_allgather_wall_ms=(\d+\.\d+).*?fused_allgather_dev_ms=(\d+\.\d+)"
    r")"
)

STAGES = [8, 4, 2, 1]


def pct(values: List[float], p: float) -> float:
    values = sorted(values)
    idx = (len(values) - 1) * p
    lo = int(math.floor(idx))
    hi = min(lo + 1, len(values) - 1)
    frac = idx - lo
    return values[lo] * (1 - frac) + values[hi] * frac


def iter_lines(paths: Iterable[Path]) -> Iterable[str]:
    for path in paths:
        try:
            with path.open("r", errors="ignore") as fh:
                for line in fh:
                    yield ANSI_RE.sub("", line)
        except FileNotFoundError:
            continue


def fmt_pair(values: List[float], ndigits: int = 3) -> str:
    if not values:
        return "N/A"
    return f"{pct(values, 0.5):.{ndigits}f} / {pct(values, 0.9):.{ndigits}f}"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extract current mode=5 timing tables from Ray worker logs.")
    parser.add_argument(
        "--ray-log-dir",
        required=True,
        help="Ray session logs directory, e.g. /tmp/ray/session_xxx/logs",
    )
    parser.add_argument(
        "--csv",
        action="store_true",
        help="Emit CSV-friendly rows instead of markdown tables.",
    )
    parser.add_argument(
        "--debug-compute",
        action="store_true",
        help="Also emit current_compute vs kernel_compute side-by-side for debugging.",
    )
    parser.add_argument(
        "--compute-field",
        choices=["kernel", "current"],
        default="kernel",
        help="Which compute metric Table 3 should use: template-specific kernel compute or current_compute_dev.",
    )
    args = parser.parse_args()

    log_dir = Path(args.ray_log_dir)
    worker_paths = sorted(log_dir.glob("worker-*.out"))
    if not worker_paths:
        raise SystemExit(f"No worker logs found under: {log_dir}")

    data: Dict[int, Dict[str, List[float]]] = {
        st: {
            "complete_comm_ms": [],
            "current_compute_wall_ms": [],
            "current_compute_dev_ms": [],
            "kernel_compute_wall_ms": [],
            "kernel_compute_dev_ms": [],
            "prefetch_total_ms": [],
            "prefetch_local_npu_dev_ms": [],
            "prefetch_remote_npu_dev_ms": [],
            "submit_remote_npu_ms": [],
            "prefetch_cpu_dev_ms": [],
            "submit_cpu_ms": [],
            "submit_populate_ms": [],
            "templates": [],
        }
        for st in STAGES
    }

    matched = 0
    for line in iter_lines(worker_paths):
        m = TIMING_RE.search(line)
        if not m:
            continue
        matched += 1
        tpl = m.group(1)
        st = int(m.group(2))
        if st not in data:
            continue
        _current_fetch_wall_ms = float(m.group(3))
        submit_pop_ms = float(m.group(4)) / 1000.0
        submit_remote_ms = float(m.group(5)) / 1000.0
        submit_cpu_ms = float(m.group(6)) / 1000.0
        prefetch_dev_ms = float(m.group(7))
        local_dev_ms = float(m.group(8))
        remote_dev_ms = float(m.group(9))
        cpu_dev_ms = float(m.group(10))
        current_compute_wall_ms = float(m.group(11))
        current_compute_dev_ms = float(m.group(12))

        # Per-template true kernel compute fields.
        if tpl == "fused-experts":
            kernel_compute_wall_ms = float(m.group(13))
            kernel_compute_dev_ms = float(m.group(14))
        elif tpl == "single-dispatch":
            kernel_compute_wall_ms = float(m.group(15))
            kernel_compute_dev_ms = float(m.group(16))
        else:  # single-rank-allgather
            kernel_compute_wall_ms = float(m.group(17))
            kernel_compute_dev_ms = float(m.group(18))

        complete_comm_ms = local_dev_ms + remote_dev_ms + cpu_dev_ms
        data[st]["complete_comm_ms"].append(complete_comm_ms)
        data[st]["current_compute_wall_ms"].append(current_compute_wall_ms)
        data[st]["current_compute_dev_ms"].append(current_compute_dev_ms)
        data[st]["kernel_compute_wall_ms"].append(kernel_compute_wall_ms)
        data[st]["kernel_compute_dev_ms"].append(kernel_compute_dev_ms)
        data[st]["prefetch_total_ms"].append(
            prefetch_dev_ms + submit_remote_ms + submit_cpu_ms)
        data[st]["prefetch_local_npu_dev_ms"].append(local_dev_ms)
        data[st]["prefetch_remote_npu_dev_ms"].append(remote_dev_ms)
        data[st]["submit_remote_npu_ms"].append(submit_remote_ms)
        data[st]["prefetch_cpu_dev_ms"].append(cpu_dev_ms)
        data[st]["submit_cpu_ms"].append(submit_cpu_ms)
        data[st]["submit_populate_ms"].append(submit_pop_ms)
        data[st]["templates"].append(tpl)

    if matched == 0:
        raise SystemExit(
            "No Mode5 timing lines matched. Check that timing logging is enabled and the run reached live steps.")

    if args.csv:
        print("table,stage,prefetch_total_ms_p50_p90,npu_dev_ms_p50_p90,npu_submit_ms_p50_p90,cpu_dev_ms_p50_p90,cpu_submit_ms_p50_p90")
        for st in STAGES:
            print(
                f"main,{st},{fmt_pair(data[st]['prefetch_total_ms'])},{fmt_pair(data[st]['prefetch_remote_npu_dev_ms'])},"
                f"{fmt_pair(data[st]['submit_remote_npu_ms'])},{fmt_pair(data[st]['prefetch_cpu_dev_ms'])},{fmt_pair(data[st]['submit_cpu_ms'])}")
        print("table,stage,prefetch_total_ms_p50_p90,npu_dev_ms_p50_p90,npu_submit_ms_p50_p90,cpu_dev_ms_p50_p90,cpu_submit_ms_p50_p90,submit_populate_ms_p50_p90")
        for st in STAGES:
            print(
                f"extended,{st},{fmt_pair(data[st]['prefetch_total_ms'])},{fmt_pair(data[st]['prefetch_remote_npu_dev_ms'])},"
                f"{fmt_pair(data[st]['submit_remote_npu_ms'])},{fmt_pair(data[st]['prefetch_cpu_dev_ms'])},{fmt_pair(data[st]['submit_cpu_ms'])},{fmt_pair(data[st]['submit_populate_ms'])}")
        compute_key = "kernel_compute_dev_ms" if args.compute_field == "kernel" else "current_compute_dev_ms"
        print("table,stage,complete_comm_ms_p50_p90,compute_ms_p50_p90")
        for st in STAGES:
            print(
                f"comm_compute,{st},{fmt_pair(data[st]['complete_comm_ms'])},{fmt_pair(data[st][compute_key])}")
        if args.debug_compute:
            print("table,stage,current_compute_dev_ms_p50_p90,kernel_compute_dev_ms_p50_p90,current_compute_wall_ms_p50_p90,kernel_compute_wall_ms_p50_p90,templates")
            for st in STAGES:
                templates = "+".join(sorted(set(data[st]['templates'])))
                print(
                    f"compute_debug,{st},{fmt_pair(data[st]['current_compute_dev_ms'])},{fmt_pair(data[st]['kernel_compute_dev_ms'])},"
                    f"{fmt_pair(data[st]['current_compute_wall_ms'])},{fmt_pair(data[st]['kernel_compute_wall_ms'])},{templates}")
        return 0

    print(f"Matched Mode5 timing rows: {matched}")
    print()
    print("Table 1")
    print("| Stage | prefetch总时间 p50/p90 ms | NPU dev通信 p50/p90 ms | NPU submit时间 p50/p90 ms | CPU dev通信 p50/p90 ms | CPU submit时间 p50/p90 ms |")
    print("|---|---:|---:|---:|---:|---:|")
    for st in STAGES:
        print(
            f"| {st} | {fmt_pair(data[st]['prefetch_total_ms'])} | {fmt_pair(data[st]['prefetch_remote_npu_dev_ms'])} | "
            f"{fmt_pair(data[st]['submit_remote_npu_ms'])} | {fmt_pair(data[st]['prefetch_cpu_dev_ms'])} | {fmt_pair(data[st]['submit_cpu_ms'])} |"
        )
    print()
    print("Table 2")
    print("| Stage | prefetch总时间 p50/p90 ms | NPU dev通信 p50/p90 ms | NPU submit时间 p50/p90 ms | CPU dev通信 p50/p90 ms | CPU submit时间 p50/p90 ms | submit_populate p50/p90 ms |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    for st in STAGES:
        print(
            f"| {st} | {fmt_pair(data[st]['prefetch_total_ms'])} | {fmt_pair(data[st]['prefetch_remote_npu_dev_ms'])} | "
            f"{fmt_pair(data[st]['submit_remote_npu_ms'])} | {fmt_pair(data[st]['prefetch_cpu_dev_ms'])} | {fmt_pair(data[st]['submit_cpu_ms'])} | {fmt_pair(data[st]['submit_populate_ms'])} |"
        )
    print()
    compute_key = "kernel_compute_dev_ms" if args.compute_field == "kernel" else "current_compute_dev_ms"
    compute_label = "kernel_compute_dev_ms" if args.compute_field == "kernel" else "current_compute_dev_ms"
    print("Table 3")
    print(f"# compute_field={compute_label}")
    print("| Stage | 完整通信 p50/p90 ms | 计算 p50/p90 ms |")
    print("|---|---:|---:|")
    for st in STAGES:
        print(
            f"| {st} | {fmt_pair(data[st]['complete_comm_ms'])} | {fmt_pair(data[st][compute_key])} |"
        )

    if args.debug_compute:
        print()
        print("Compute Debug")
        print("| Stage | current_compute_dev p50/p90 ms | kernel_compute_dev p50/p90 ms | current_compute_wall p50/p90 ms | kernel_compute_wall p50/p90 ms | templates |")
        print("|---|---:|---:|---:|---:|---|")
        for st in STAGES:
            templates = "+".join(sorted(set(data[st]['templates'])))
            print(
                f"| {st} | {fmt_pair(data[st]['current_compute_dev_ms'])} | {fmt_pair(data[st]['kernel_compute_dev_ms'])} | "
                f"{fmt_pair(data[st]['current_compute_wall_ms'])} | {fmt_pair(data[st]['kernel_compute_wall_ms'])} | {templates} |"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
