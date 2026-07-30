#!/usr/bin/env python3
import ast
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path


ENTER_RE = re.compile(
    r"Elastic EP shrink rpc enter: global_rank=(?P<rank>\d+) "
    r"active_ranks=(?P<active>\[[^\]]*\]).*?local_has_unfinished=(?P<unfinished>True|False)"
)

DONE_RE = re.compile(
    r"Elastic parallel shrink done: rank=(?P<rank>\d+) "
    r"active_ranks=(?P<active>\[[^\]]*\]).*?"
    r"rebuild_ms=(?P<rebuild>[0-9.]+) "
    r"refresh_ms=(?P<refresh>[0-9.]+) "
    r"warmup_ms=(?P<warmup>[0-9.]+) "
    r"total_ms=(?P<total>[0-9.]+)"
)

PHASE_RE = re.compile(
    r"Elastic parallel shrink phase breakdown: rank=(?P<rank>\d+) "
    r"active_ranks=(?P<active>\[[^\]]*\]).*?"
    r"is_active=(?P<is_active>[01]).*?"
    r"(?:mode1_pre_import_drain_ms=(?P<mode1_drain>[0-9.]+) )?"
    r"prepare_payload_ms=(?P<prepare>[0-9.]+) "
    r"preload_import_ms=(?P<preload>[0-9.]+) .*?"
    r"rebuild_ms=(?P<rebuild>[0-9.]+) .*?"
    r"refresh_ms=(?P<refresh>[0-9.]+) .*?"
    r"release_staging_ms=(?P<release>[0-9.]+) "
    r"drop_stale_group_cache_ms=(?P<drop_stale>[0-9.]+) "
    r"drop_old_floor_group_cache_ms=(?P<drop_old>[0-9.]+) .*?"
    r"warmup_ms=(?P<warmup>[0-9.]+) .*?"
    r"total_ms=(?P<total>[0-9.]+)"
)

STREAM_RE = re.compile(
    r"Elastic shrink stream import breakdown: rank=(?P<rank>\d+) "
    r"active_ranks=(?P<active>\[[^\]]*\]).*?"
    r"send_export_ms=(?P<send_export>[0-9.]+) .*?"
    r"send_wait_ms=(?P<send_wait>[0-9.]+) "
    r"recv_wait_ms=(?P<recv_wait>[0-9.]+) .*?"
    r"total_ms=(?P<total>[0-9.]+)"
)

ROLLOUT_RE = re.compile(r"rollout_output_time[:=]\s*(?P<value>[0-9.]+)")


def parse_active(text: str) -> tuple[int, ...]:
    return tuple(int(x) for x in ast.literal_eval(text))


def fmt_stats(values: list[float]) -> str:
    if not values:
        return "n=0"
    return (
        f"n={len(values)} avg={statistics.mean(values):.2f} "
        f"max={max(values):.2f}"
    )


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: analyze_mode1_shrink_stages.py <logfile>", file=sys.stderr)
        return 2

    path = Path(sys.argv[1])
    if not path.exists():
        print(f"log not found: {path}", file=sys.stderr)
        return 1

    rollout = None
    enters = defaultdict(lambda: {"unfinished_true": 0, "unfinished_false": 0, "ranks": []})
    phases = defaultdict(list)
    dones = defaultdict(list)
    streams = defaultdict(list)

    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = ROLLOUT_RE.search(line)
            if m:
                rollout = float(m.group("value"))

            m = ENTER_RE.search(line)
            if m:
                active = parse_active(m.group("active"))
                bucket = enters[active]
                bucket["ranks"].append(int(m.group("rank")))
                bucket["unfinished_true" if m.group("unfinished") == "True" else "unfinished_false"] += 1
                continue

            m = PHASE_RE.search(line)
            if m:
                active = parse_active(m.group("active"))
                phases[active].append({
                    "rank": int(m.group("rank")),
                    "is_active": int(m.group("is_active")),
                    "mode1_drain": float(m.group("mode1_drain") or 0.0),
                    "prepare": float(m.group("prepare")),
                    "preload": float(m.group("preload")),
                    "rebuild": float(m.group("rebuild")),
                    "refresh": float(m.group("refresh")),
                    "release": float(m.group("release")),
                    "drop_stale": float(m.group("drop_stale")),
                    "drop_old": float(m.group("drop_old")),
                    "warmup": float(m.group("warmup")),
                    "total": float(m.group("total")),
                })
                continue

            m = DONE_RE.search(line)
            if m:
                active = parse_active(m.group("active"))
                dones[active].append({
                    "rank": int(m.group("rank")),
                    "rebuild": float(m.group("rebuild")),
                    "refresh": float(m.group("refresh")),
                    "warmup": float(m.group("warmup")),
                    "total": float(m.group("total")),
                })
                continue

            m = STREAM_RE.search(line)
            if m:
                active = parse_active(m.group("active"))
                streams[active].append({
                    "rank": int(m.group("rank")),
                    "send_export": float(m.group("send_export")),
                    "send_wait": float(m.group("send_wait")),
                    "recv_wait": float(m.group("recv_wait")),
                    "total": float(m.group("total")),
                })

    print(f"log={path.name}")
    print(f"rollout_output_time_s={rollout if rollout is not None else 'N/A'}")

    ordered = sorted(set(enters) | set(phases) | set(dones) | set(streams), key=lambda x: (len(x), x))
    for active in ordered:
        print()
        print(f"[stage] active_ranks={list(active)}")

        if active in enters:
            bucket = enters[active]
            print(
                "enter: "
                f"unfinished_true={bucket['unfinished_true']} "
                f"unfinished_false={bucket['unfinished_false']} "
                f"ranks={sorted(bucket['ranks'])}"
            )

        if active in streams:
            rows = streams[active]
            print(
                "stream: "
                f"send_export({fmt_stats([r['send_export'] for r in rows])}) "
                f"send_wait({fmt_stats([r['send_wait'] for r in rows])}) "
                f"recv_wait({fmt_stats([r['recv_wait'] for r in rows])}) "
                f"total({fmt_stats([r['total'] for r in rows])})"
            )

        if active in phases:
            rows = phases[active]
            for flag, label in [(0, "inactive"), (1, "active")]:
                subset = [r for r in rows if r["is_active"] == flag]
                if not subset:
                    continue
                print(
                    f"phase_{label}: "
                    f"mode1_drain({fmt_stats([r['mode1_drain'] for r in subset])}) "
                    f"prepare({fmt_stats([r['prepare'] for r in subset])}) "
                    f"preload({fmt_stats([r['preload'] for r in subset])}) "
                    f"rebuild({fmt_stats([r['rebuild'] for r in subset])}) "
                    f"refresh({fmt_stats([r['refresh'] for r in subset])}) "
                    f"release({fmt_stats([r['release'] for r in subset])}) "
                    f"drop_old({fmt_stats([r['drop_old'] for r in subset])}) "
                    f"warmup({fmt_stats([r['warmup'] for r in subset])}) "
                    f"total({fmt_stats([r['total'] for r in subset])})"
                )

        if active in dones:
            rows = dones[active]
            print(
                "done_active: "
                f"rebuild({fmt_stats([r['rebuild'] for r in rows])}) "
                f"refresh({fmt_stats([r['refresh'] for r in rows])}) "
                f"warmup({fmt_stats([r['warmup'] for r in rows])}) "
                f"total({fmt_stats([r['total'] for r in rows])})"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
