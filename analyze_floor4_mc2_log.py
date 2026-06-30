#!/usr/bin/env python3
"""Summarize floor=4 MC2 cache/destroy latency from an elastic rollout log."""
from __future__ import annotations

import argparse
import re
from pathlib import Path

FLOAT = r"([-+]?\d+(?:\.\d+)?)"
RANKS = r"\(([^)]*)\)"

PATTERNS = {
    "rollout": re.compile(r"rollout_output_time_s:\s*" + FLOAT),
    "mc2_cached": re.compile(r"Elastic parallel MC2 group cached:.*?rank=(\d+).*?ranks=" + RANKS),
    "cache_hit": re.compile(r"Elastic parallel group cache hit:.*?rank=(\d+).*?attr=_MC2.*?ranks=" + RANKS),
    "stale_drop": re.compile(r"Elastic cached group is retired; dropping stale cache entry:.*?rank=(\d+).*?group_kind=mc2.*?ranks=" + RANKS),
    "stash_slow": re.compile(
        r"Elastic stash slow timing:.*?rank=(\d+).*?group_kind=mc2.*?group_ranks="
        + RANKS
        + r".*?total_ms="
        + FLOAT
        + r".*?quarantine_ms="
        + FLOAT
        + r".*?destroy_ms="
        + FLOAT
    ),
    "rebuild_slow": re.compile(
        r"Elastic rebuild group slow timing:.*?rank=(\d+).*?group_kind=mc2.*?target_ranks="
        + RANKS
        + r".*?cache_hit=(\d+).*?total_ms="
        + FLOAT
        + r".*?stash_ms="
        + FLOAT
        + r".*?create_ms="
        + FLOAT
    ),
    "destroy_deferred": re.compile(
        r"Elastic parallel group destroy deferred:.*?rank=(\d+).*?group_kind=mc2.*?group_ranks="
        + RANKS
        + r".*?retire_wrapper_ms="
        + FLOAT
        + r".*?device_pg_destroy_ms="
        + FLOAT
        + r".*?floor4_quiesce_enabled="
        + FLOAT
        + r".*?floor4_quiesce_total_ms="
        + FLOAT
    ),
    "quarantine_exit": re.compile(
        r"Elastic floor4 MC2 quarantine exit:.*?rank=(\d+).*?group_ranks="
        + RANKS
        + r".*?total_ms="
        + FLOAT
        + r".*?retire_call_ms="
        + FLOAT
        + r".*?retire_wrapper_ms="
        + FLOAT
        + r".*?append_ms="
        + FLOAT
        + r".*?log_ms="
        + FLOAT
        + r"(?:.*?destroyed_device_pgs_kept=(\d+))?"
        + r"(?:.*?group_ref_kept=(\d+))?"
        + r"(?:.*?destroyed_pg_ref_trim_ms="
        + FLOAT
        + r")?"
        + r"(?:.*?destroyed_pg_refs_trimmed="
        + FLOAT
        + r")?"
        + r"(?:.*?destroyed_pg_refs_remaining="
        + FLOAT
        + r")?"
    ),
    "destroyed_pg_ref_trim": re.compile(
        r"Elastic floor4 MC2 destroyed PG ref trim:.*?rank=(\d+).*?max_refs=(-?\d+).*?trimmed=(\d+).*?remaining=(\d+).*?trim_ms="
        + FLOAT
    ),
    "async_destroyed_pg_ref_trim_end": re.compile(
        r"Elastic floor4 MC2 async destroyed PG ref trim end:.*?rank=(\d+).*?trim_ms="
        + FLOAT
        + r".*?trimmed="
        + FLOAT
        + r".*?remaining="
        + FLOAT
        + r".*?limit="
        + FLOAT
    ),
    "oom": re.compile(r"(OutOfMemory|OOM|out of memory|EZ9999|aicore timeout|Traceback)", re.I),
}


def parse_ranks(raw: str) -> tuple[int, ...]:
    vals = []
    for part in raw.split(','):
        part = part.strip()
        if not part:
            continue
        try:
            vals.append(int(part))
        except ValueError:
            pass
    return tuple(vals)


def is_floor4(ranks: tuple[int, ...]) -> bool:
    return len(ranks) == 4


def add_top(store: list[tuple[float, str]], value: float, line: str, limit: int) -> None:
    store.append((value, line.rstrip()))
    store.sort(key=lambda item: item[0], reverse=True)
    del store[limit:]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("log", type=Path)
    parser.add_argument("--top", type=int, default=12)
    args = parser.parse_args()

    counts = {
        "floor4_mc2_cached": 0,
        "floor4_mc2_cache_hit": 0,
        "floor4_mc2_stale_drop": 0,
        "floor4_mc2_destroy": 0,
        "floor4_mc2_slow_stash": 0,
        "floor4_mc2_slow_rebuild": 0,
        "floor4_mc2_quarantine_exit": 0,
        "floor4_mc2_group_ref_kept": 0,
        "floor4_mc2_group_ref_dropped": 0,
        "floor4_mc2_destroyed_pg_ref_kept": 0,
        "floor4_mc2_destroyed_pg_ref_trim_events": 0,
        "floor4_mc2_async_destroyed_pg_ref_trim_events": 0,
        "floor4_mc2_destroyed_pg_refs_trimmed": 0,
        "error_lines": 0,
    }
    rollouts: list[float] = []
    top_destroy: list[tuple[float, str]] = []
    top_quiesce: list[tuple[float, str]] = []
    top_quarantine: list[tuple[float, str]] = []
    top_destroyed_pg_ref_trim: list[tuple[float, str]] = []
    top_stash: list[tuple[float, str]] = []
    top_rebuild: list[tuple[float, str]] = []
    cached_ranks: dict[tuple[int, ...], int] = {}
    hit_ranks: dict[tuple[int, ...], int] = {}

    with args.log.open("r", errors="ignore") as f:
        for line in f:
            if m := PATTERNS["rollout"].search(line):
                rollouts.append(float(m.group(1)))
            if PATTERNS["oom"].search(line):
                counts["error_lines"] += 1

            if m := PATTERNS["mc2_cached"].search(line):
                ranks = parse_ranks(m.group(2))
                if is_floor4(ranks):
                    counts["floor4_mc2_cached"] += 1
                    cached_ranks[ranks] = cached_ranks.get(ranks, 0) + 1
            if m := PATTERNS["cache_hit"].search(line):
                ranks = parse_ranks(m.group(2))
                if is_floor4(ranks):
                    counts["floor4_mc2_cache_hit"] += 1
                    hit_ranks[ranks] = hit_ranks.get(ranks, 0) + 1
            if m := PATTERNS["stale_drop"].search(line):
                ranks = parse_ranks(m.group(2))
                if is_floor4(ranks):
                    counts["floor4_mc2_stale_drop"] += 1
            if m := PATTERNS["destroy_deferred"].search(line):
                ranks = parse_ranks(m.group(2))
                if is_floor4(ranks):
                    counts["floor4_mc2_destroy"] += 1
                    device_destroy_ms = float(m.group(4))
                    quiesce_ms = float(m.group(6))
                    add_top(top_destroy, device_destroy_ms, line, args.top)
                    add_top(top_quiesce, quiesce_ms, line, args.top)
            if m := PATTERNS["quarantine_exit"].search(line):
                ranks = parse_ranks(m.group(2))
                if is_floor4(ranks):
                    counts["floor4_mc2_quarantine_exit"] += 1
                    total_ms = float(m.group(3))
                    destroyed_pg_refs = m.group(8)
                    group_ref_kept = m.group(9)
                    trim_ms = m.group(10)
                    if destroyed_pg_refs is not None:
                        counts["floor4_mc2_destroyed_pg_ref_kept"] += int(destroyed_pg_refs)
                    if group_ref_kept == "1":
                        counts["floor4_mc2_group_ref_kept"] += 1
                    elif group_ref_kept == "0":
                        counts["floor4_mc2_group_ref_dropped"] += 1
                    if trim_ms is not None:
                        add_top(top_destroyed_pg_ref_trim, float(trim_ms), line,
                                args.top)
                    add_top(top_quarantine, total_ms, line, args.top)
            if m := PATTERNS["destroyed_pg_ref_trim"].search(line):
                counts["floor4_mc2_destroyed_pg_ref_trim_events"] += 1
                counts["floor4_mc2_destroyed_pg_refs_trimmed"] += int(m.group(3))
                add_top(top_destroyed_pg_ref_trim, float(m.group(5)), line,
                        args.top)
            if m := PATTERNS["async_destroyed_pg_ref_trim_end"].search(line):
                counts["floor4_mc2_async_destroyed_pg_ref_trim_events"] += 1
                counts["floor4_mc2_destroyed_pg_refs_trimmed"] += int(
                    float(m.group(3)))
                add_top(top_destroyed_pg_ref_trim, float(m.group(2)), line,
                        args.top)
            if m := PATTERNS["stash_slow"].search(line):
                ranks = parse_ranks(m.group(2))
                if is_floor4(ranks):
                    counts["floor4_mc2_slow_stash"] += 1
                    total_ms = float(m.group(3))
                    add_top(top_stash, total_ms, line, args.top)
            if m := PATTERNS["rebuild_slow"].search(line):
                ranks = parse_ranks(m.group(2))
                if is_floor4(ranks):
                    counts["floor4_mc2_slow_rebuild"] += 1
                    total_ms = float(m.group(4))
                    add_top(top_rebuild, total_ms, line, args.top)

    print(f"log={args.log}")
    print("rollout_output_time_s=" + ",".join(f"{v:.3f}" for v in rollouts))
    for key, value in counts.items():
        print(f"{key}={value}")
    if cached_ranks:
        print("cached_floor4_ranks=" + ", ".join(f"{r}:{n}" for r, n in sorted(cached_ranks.items())))
    if hit_ranks:
        print("cache_hit_floor4_ranks=" + ", ".join(f"{r}:{n}" for r, n in sorted(hit_ranks.items())))

    sections = [
        ("top_device_pg_destroy_ms", top_destroy),
        ("top_quiesce_ms", top_quiesce),
        ("top_quarantine_total_ms", top_quarantine),
        ("top_destroyed_pg_ref_trim_ms", top_destroyed_pg_ref_trim),
        ("top_stash_total_ms", top_stash),
        ("top_rebuild_total_ms", top_rebuild),
    ]
    for name, rows in sections:
        print(f"\n[{name}]")
        if not rows:
            print("<none>")
            continue
        for value, line in rows:
            print(f"{value:.2f} {line}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
