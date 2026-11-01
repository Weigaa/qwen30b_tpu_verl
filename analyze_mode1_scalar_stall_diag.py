#!/usr/bin/env python3
"""Summarize mode=1 scalar-stall diagnostic launcher logs."""

from __future__ import annotations

import re
import sys
from pathlib import Path

SCALAR_RE = re.compile(
    r"Mode1 NPU scalar slot read done: .*?layer=(?P<layer>-?\d+) .*?"
    r"expert_id=(?P<expert>-?\d+) slot=(?P<slot>-?\d+) .*?"
    r"map_kind=(?P<map_kind>\S+) item_ms=(?P<ms>[0-9.]+)"
)
BEGIN_RE = re.compile(
    r"Mode1 NPU scalar slot read begin: .*?layer=(?P<layer>-?\d+) .*?"
    r"expert_id=(?P<expert>-?\d+) map_kind=(?P<map_kind>\S+)"
)
SLOW_EXPORT_RE = re.compile(
    r"Elastic shrink direct NPU(?: batch)? slow export: .*?layer=(?P<layer>-?\d+) "
    r".*?source_rank=(?P<src>\d+) target_rank=(?P<dst>\d+) .*?"
    r"export_ms=(?P<ms>[0-9.]+)"
)
P2P_RE = re.compile(
    r"Elastic shrink direct NPU(?: batch)? P2P (?P<kind>send|recv) timing: .*?"
    r"layer=(?P<layer>-?\d+) .*?(?P<kind2>send|recv)_post_ms=(?P<post>[0-9.]+) "
    r"(?P=kind2)_wait_ms=(?P<wait>[0-9.]+)"
)
BREAKDOWN_RE = re.compile(
    r"Elastic shrink stream import breakdown: .*?layer=(?P<layer>-?\d+) .*?"
    r"send_export_ms=(?P<export>[0-9.]+).*?"
    r"send_export_scalar_item_count=(?P<count>\d+) "
    r"send_export_scalar_item_ms_total=(?P<scalar_total>[0-9.]+) "
    r"send_export_scalar_item_ms_max=(?P<scalar_max>[0-9.]+).*?"
    r"send_wait_ms=(?P<send_wait>[0-9.]+).*?recv_wait_ms=(?P<recv_wait>[0-9.]+).*?"
    r"total_ms=(?P<total>[0-9.]+)"
)


def resolve_log_path(path: Path) -> Path:
    if path.is_dir():
        candidates = sorted(
            path.glob("wjeagerqwen30b-a3b-with_draft_*_elastic.txt"),
            key=lambda item: item.stat().st_mtime,
            reverse=True,
        )
        if candidates:
            return candidates[0]
        launcher = path / "launcher.log"
        if launcher.is_file():
            return launcher
        return path

    if path.name == "launcher.log" and path.parent.is_dir():
        candidates = sorted(
            path.parent.glob("wjeagerqwen30b-a3b-with_draft_*_elastic.txt"),
            key=lambda item: item.stat().st_mtime,
            reverse=True,
        )
        if candidates:
            return candidates[0]
    return path


def main() -> int:
    if len(sys.argv) != 2:
        print(
            f"Usage: {Path(sys.argv[0]).name} /path/to/run_dir_or_train.log",
            file=sys.stderr)
        return 2
    original_path = Path(sys.argv[1])
    path = resolve_log_path(original_path)
    if not path.is_file():
        print(f"missing log file: {path} (from {original_path})", file=sys.stderr)
        return 2

    scalar_done = []
    scalar_begin_count = 0
    slow_exports = []
    p2p = []
    breakdowns = []

    with path.open("r", errors="replace") as handle:
        for line_no, line in enumerate(handle, 1):
            m = SCALAR_RE.search(line)
            if m:
                scalar_done.append((float(m.group("ms")), line_no, m.groupdict()))
            if BEGIN_RE.search(line):
                scalar_begin_count += 1
            m = SLOW_EXPORT_RE.search(line)
            if m:
                slow_exports.append((float(m.group("ms")), line_no, m.groupdict()))
            m = P2P_RE.search(line)
            if m:
                p2p.append((max(float(m.group("post")), float(m.group("wait"))), line_no, m.groupdict()))
            m = BREAKDOWN_RE.search(line)
            if m:
                breakdowns.append((float(m.group("total")), line_no, m.groupdict()))

    print(f"log={path}")
    if path != original_path:
        print(f"resolved_from={original_path}")
    print(f"scalar_begin={scalar_begin_count} scalar_done={len(scalar_done)}")
    if scalar_begin_count != len(scalar_done):
        print("WARNING: scalar begin/done counts differ; if the run was killed, the last begin before termination may be the blocked op.")

    if scalar_done:
        total = sum(item[0] for item in scalar_done)
        print(f"scalar_item_ms_total={total:.2f} scalar_item_ms_max={max(item[0] for item in scalar_done):.2f}")
        print("top scalar .item() reads:")
        for ms, line_no, data in sorted(scalar_done, reverse=True)[:20]:
            print(f"  line={line_no} item_ms={ms:.2f} layer={data['layer']} expert={data['expert']} slot={data['slot']} map={data['map_kind']}")

    if slow_exports:
        print("top slow exports:")
        for ms, line_no, data in sorted(slow_exports, reverse=True)[:20]:
            print(f"  line={line_no} export_ms={ms:.2f} layer={data['layer']} src={data['src']} dst={data['dst']}")

    if p2p:
        print("top P2P post/wait timings:")
        for ms, line_no, data in sorted(p2p, reverse=True)[:20]:
            print(f"  line={line_no} kind={data['kind']} post_ms={float(data['post']):.2f} wait_ms={float(data['wait']):.2f} layer={data['layer']}")

    if breakdowns:
        print("top stream import breakdowns:")
        for total_ms, line_no, data in sorted(breakdowns, reverse=True)[:20]:
            print(
                "  line={} total_ms={:.2f} layer={} send_export_ms={} "
                "scalar_count={} scalar_total={} scalar_max={} send_wait={} recv_wait={}".format(
                    line_no,
                    total_ms,
                    data["layer"],
                    data["export"],
                    data["count"],
                    data["scalar_total"],
                    data["scalar_max"],
                    data["send_wait"],
                    data["recv_wait"],
                )
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
