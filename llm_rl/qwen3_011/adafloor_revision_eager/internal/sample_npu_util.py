#!/usr/bin/env python3
"""Sample Ascend NPU AICore utilization with npu-smi info.

Outputs CSV rows that can be joined with Elastic decode util bucket markers.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import re
import subprocess
import sys
import time
from pathlib import Path

# Example line:
# | 0     0                   | 0000:9D:00.0  | 0           0    / 0          12220/ 65536         |
DEVICE_RE = re.compile(
    r"\|\s*(?P<npu>\d+)\s+(?P<phy>\d+)\s*\|\s*"
    r"(?P<bus>[0-9A-Fa-f:.]+)\s*\|\s*"
    r"(?P<aicore>\d+)\s+\d+\s*/\s*\d+\s+"
    r"(?P<hbm_used>\d+)\s*/\s*(?P<hbm_total>\d+)"
)


def sample_once() -> list[dict[str, object]]:
    out = subprocess.check_output(["npu-smi", "info"], text=True,
                                  stderr=subprocess.DEVNULL)
    now = time.time()
    iso = dt.datetime.fromtimestamp(now).isoformat(timespec="milliseconds")
    rows: list[dict[str, object]] = []
    for line in out.splitlines():
        m = DEVICE_RE.search(line)
        if not m:
            continue
        rows.append({
            "ts_epoch": f"{now:.6f}",
            "ts_iso": iso,
            "npu": int(m.group("npu")),
            "phy_id": int(m.group("phy")),
            "bus_id": m.group("bus"),
            "aicore_pct": int(m.group("aicore")),
            "hbm_used_mb": int(m.group("hbm_used")),
            "hbm_total_mb": int(m.group("hbm_total")),
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--interval", type=float, default=1.0)
    parser.add_argument("--duration", type=float, default=0.0,
                        help="0 means run until interrupted")
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "ts_epoch", "ts_iso", "npu", "phy_id", "bus_id", "aicore_pct",
        "hbm_used_mb", "hbm_total_mb",
    ]
    start = time.time()
    with args.output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        while True:
            try:
                for row in sample_once():
                    writer.writerow(row)
                f.flush()
            except Exception as exc:  # Keep sampling best-effort.
                print(f"sample_npu_util warning: {exc}", file=sys.stderr,
                      flush=True)
            if args.duration > 0 and time.time() - start >= args.duration:
                break
            time.sleep(args.interval)


if __name__ == "__main__":
    main()
