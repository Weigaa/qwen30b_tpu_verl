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
    "mc2_cached": re.compile(
        r"Elastic parallel MC2 group cached:.*?rank=(\d+).*?group_name=([^ ]+).*?ranks="
        + RANKS
        + r"(?:.*?coordinator_id=([^ ]+).*?device_pg_id=([^ ]+).*?cpu_pg_id=([^ ]+))?"),
    "cache_hit": re.compile(
        r"Elastic parallel group cache hit:.*?rank=(\d+).*?attr=_MC2.*?group_name=([^ ]+).*?ranks="
        + RANKS
        + r"(?:.*?coordinator_id=([^ ]+).*?device_pg_id=([^ ]+).*?cpu_pg_id=([^ ]+))?"),
    "live_pool": re.compile(
        r"Elastic floor4 MC2 live pool:.*?rank=(\d+).*?event=([a-zA-Z0-9_]+).*?group_ranks="
        + RANKS
        + r".*?pool_size=(\d+).*?pool_ranks=(.*?)(?: placeholder_pool_size=(\d+).*?placeholder_pool_ranks=(.*?))?(?: combined_pool_size=(\d+).*?pool_over_budget=(\d+))?(?: warn_size=|$)"
    ),
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
    "live_pool_rank_set_not_allowed": re.compile(
        r"Floor4 MC2 live-pool rank set is not allowed:.*?rank=(\d+).*?event=([a-zA-Z0-9_]+).*?group_ranks="
        + RANKS
    ),
    "live_pool_capacity_exceeded": re.compile(
        r"Floor4 MC2 live-pool capacity exceeded before creating group:.*?rank=(\d+).*?event=([a-zA-Z0-9_]+).*?group_ranks="
        + RANKS
        + r".*?max_groups=(\d+).*?combined_pool_size=(\d+)"
    ),
    "binding_state": re.compile(
        r"Elastic floor4 MC2 binding state:.*?rank=(\d+).*?event=([a-zA-Z0-9_]+).*?expected_ranks="
        + RANKS
        + r".*?module_mc2_ranks=(None|\([^)]*\)).*?mismatch=(\d+)"
    ),
    "explicit_floor4": re.compile(r"explicit_floor4_ranks=" + RANKS),
    "floor4_mc2_fallback": re.compile(
        r"Floor4 MC2 fallback on unpooled rank set: ranks="
        + RANKS
        + r".*?num_tokens=(\d+).*?with_prefill=(\S+)"
    ),
    "floor4_mc2_unpooled_rebuild": re.compile(
        r"Floor4 MC2 unpooled rebuild before fallback:.*?rank=(\d+).*?event=([a-zA-Z0-9_]+).*?group_ranks="
        + RANKS
    ),
    "floor4_mc2_rebuild_skipped": re.compile(
        r"Floor4 MC2 rebuild skipped for unpooled rank set:.*?rank=(\d+).*?event=([a-zA-Z0-9_]+).*?active_ranks="
        + RANKS
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
        "floor4_mc2_live_pool_events": 0,
        "floor4_mc2_live_pool_cache_store": 0,
        "floor4_mc2_live_pool_cache_hit": 0,
        "floor4_mc2_live_pool_placeholder_store": 0,
        "floor4_mc2_live_pool_placeholder_hit": 0,
        "floor4_mc2_live_pool_placeholder_signature_seen": 0,
        "floor4_mc2_live_pool_placeholder_signature_seen_no_pool": 0,
        "floor4_mc2_live_pool_retire_skip": 0,
        "floor4_mc2_live_pool_quarantine_skip": 0,
        "floor4_mc2_live_pool_over_budget_events": 0,
        "floor4_mc2_live_pool_rank_set_not_allowed": 0,
        "floor4_mc2_live_pool_capacity_exceeded": 0,
        "floor4_mc2_binding_state_events": 0,
        "floor4_mc2_binding_mismatch": 0,
        "floor4_mc2_binding_unknown": 0,
        "floor4_mc2_fallback_on_unpooled": 0,
        "floor4_mc2_unpooled_rebuild_before_fallback": 0,
        "floor4_mc2_rebuild_skipped_for_unpooled": 0,
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
    group_names_by_ranks: dict[tuple[int, ...], set[str]] = {}
    device_pg_ids_by_ranks: dict[tuple[int, ...], set[str]] = {}
    live_pool_events: dict[str, int] = {}
    live_pool_ranks: dict[tuple[int, ...], int] = {}
    slow_stash_ranks: dict[tuple[int, ...], int] = {}
    quarantine_ranks: dict[tuple[int, ...], int] = {}
    binding_mismatch_ranks: dict[tuple[int, ...], int] = {}
    binding_unknown_ranks: dict[tuple[int, ...], int] = {}
    fallback_ranks: dict[tuple[int, ...], int] = {}
    unpooled_rebuild_ranks: dict[tuple[int, ...], int] = {}
    skipped_rebuild_ranks: dict[tuple[int, ...], int] = {}
    explicit_floor4_sequence: list[tuple[int, ...]] = []
    binding_floor4_sequence: list[tuple[int, ...]] = []
    max_live_pool_size = 0
    max_placeholder_pool_size = 0
    max_combined_pool_size = 0

    with args.log.open("r", errors="ignore") as f:
        for line in f:
            if m := PATTERNS["rollout"].search(line):
                rollouts.append(float(m.group(1)))
            if PATTERNS["oom"].search(line):
                counts["error_lines"] += 1

            if m := PATTERNS["mc2_cached"].search(line):
                group_name = m.group(2)
                ranks = parse_ranks(m.group(3))
                if is_floor4(ranks):
                    counts["floor4_mc2_cached"] += 1
                    cached_ranks[ranks] = cached_ranks.get(ranks, 0) + 1
                    group_names_by_ranks.setdefault(ranks, set()).add(
                        group_name)
                    if m.group(5):
                        device_pg_ids_by_ranks.setdefault(ranks, set()).add(
                            m.group(5))
            if m := PATTERNS["cache_hit"].search(line):
                group_name = m.group(2)
                ranks = parse_ranks(m.group(3))
                if is_floor4(ranks):
                    counts["floor4_mc2_cache_hit"] += 1
                    hit_ranks[ranks] = hit_ranks.get(ranks, 0) + 1
                    group_names_by_ranks.setdefault(ranks, set()).add(
                        group_name)
                    if m.group(5):
                        device_pg_ids_by_ranks.setdefault(ranks, set()).add(
                            m.group(5))
            if m := PATTERNS["live_pool"].search(line):
                event = m.group(2)
                ranks = parse_ranks(m.group(3))
                pool_size = int(m.group(4))
                placeholder_pool_size = int(m.group(6) or 0)
                combined_pool_size = int(
                    m.group(8) or (pool_size + placeholder_pool_size))
                pool_over_budget = int(m.group(9) or 0)
                counts["floor4_mc2_live_pool_events"] += 1
                counts["floor4_mc2_live_pool_over_budget_events"] += (
                    pool_over_budget)
                if event == "cache_store":
                    counts["floor4_mc2_live_pool_cache_store"] += 1
                if event == "cache_hit":
                    counts["floor4_mc2_live_pool_cache_hit"] += 1
                if event == "placeholder_cache_store":
                    counts["floor4_mc2_live_pool_placeholder_store"] += 1
                if event == "placeholder_cache_hit":
                    counts["floor4_mc2_live_pool_placeholder_hit"] += 1
                if event == "placeholder_signature_seen":
                    counts[
                        "floor4_mc2_live_pool_placeholder_signature_seen"] += 1
                if event == "placeholder_signature_seen_no_pool":
                    counts[
                        "floor4_mc2_live_pool_placeholder_signature_seen_no_pool"] += 1
                if event == "retire_skip":
                    counts["floor4_mc2_live_pool_retire_skip"] += 1
                if event == "quarantine_skip":
                    counts["floor4_mc2_live_pool_quarantine_skip"] += 1
                live_pool_events[event] = live_pool_events.get(event, 0) + 1
                if is_floor4(ranks):
                    live_pool_ranks[ranks] = live_pool_ranks.get(ranks, 0) + 1
                max_live_pool_size = max(max_live_pool_size, pool_size)
                max_placeholder_pool_size = max(max_placeholder_pool_size,
                                                placeholder_pool_size)
                max_combined_pool_size = max(max_combined_pool_size,
                                             combined_pool_size)
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
                    quarantine_ranks[ranks] = quarantine_ranks.get(ranks, 0) + 1
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
            if m := PATTERNS["live_pool_rank_set_not_allowed"].search(line):
                counts["floor4_mc2_live_pool_rank_set_not_allowed"] += 1
            if m := PATTERNS["live_pool_capacity_exceeded"].search(line):
                counts["floor4_mc2_live_pool_capacity_exceeded"] += 1
            if m := PATTERNS["binding_state"].search(line):
                expected_ranks = parse_ranks(m.group(3))
                module_ranks_raw = m.group(4)
                mismatch = int(m.group(5))
                counts["floor4_mc2_binding_state_events"] += 1
                counts["floor4_mc2_binding_mismatch"] += mismatch
                if module_ranks_raw == "None":
                    counts["floor4_mc2_binding_unknown"] += 1
                    binding_unknown_ranks[expected_ranks] = (
                        binding_unknown_ranks.get(expected_ranks, 0) + 1)
                if mismatch:
                    binding_mismatch_ranks[expected_ranks] = (
                        binding_mismatch_ranks.get(expected_ranks, 0) + 1)
                if is_floor4(expected_ranks):
                    if (not binding_floor4_sequence
                            or binding_floor4_sequence[-1] != expected_ranks):
                        binding_floor4_sequence.append(expected_ranks)
            if m := PATTERNS["explicit_floor4"].search(line):
                ranks = parse_ranks(m.group(1))
                if is_floor4(ranks):
                    if (not explicit_floor4_sequence
                            or explicit_floor4_sequence[-1] != ranks):
                        explicit_floor4_sequence.append(ranks)
            if m := PATTERNS["floor4_mc2_fallback"].search(line):
                ranks = parse_ranks(m.group(1))
                if is_floor4(ranks):
                    counts["floor4_mc2_fallback_on_unpooled"] += 1
                    fallback_ranks[ranks] = fallback_ranks.get(ranks, 0) + 1
            if m := PATTERNS["floor4_mc2_unpooled_rebuild"].search(line):
                ranks = parse_ranks(m.group(3))
                if is_floor4(ranks):
                    counts[
                        "floor4_mc2_unpooled_rebuild_before_fallback"] += 1
                    unpooled_rebuild_ranks[ranks] = (
                        unpooled_rebuild_ranks.get(ranks, 0) + 1)
            if m := PATTERNS["floor4_mc2_rebuild_skipped"].search(line):
                ranks = parse_ranks(m.group(3))
                if is_floor4(ranks):
                    counts["floor4_mc2_rebuild_skipped_for_unpooled"] += 1
                    skipped_rebuild_ranks[ranks] = (
                        skipped_rebuild_ranks.get(ranks, 0) + 1)
            if m := PATTERNS["stash_slow"].search(line):
                ranks = parse_ranks(m.group(2))
                if is_floor4(ranks):
                    counts["floor4_mc2_slow_stash"] += 1
                    slow_stash_ranks[ranks] = slow_stash_ranks.get(ranks, 0) + 1
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
    if group_names_by_ranks:
        print("floor4_group_names_by_ranks=" + ", ".join(
            f"{r}:{tuple(sorted(names))}"
            for r, names in sorted(group_names_by_ranks.items())))
    if device_pg_ids_by_ranks:
        print("floor4_device_pg_ids_by_ranks=" + ", ".join(
            f"{r}:{tuple(sorted(ids))}"
            for r, ids in sorted(device_pg_ids_by_ranks.items())))
    if live_pool_events:
        print("live_pool_events=" + ", ".join(f"{k}:{v}" for k, v in sorted(live_pool_events.items())))
        print("live_pool_floor4_ranks=" + ", ".join(f"{r}:{n}" for r, n in sorted(live_pool_ranks.items())))
        print(f"live_pool_max_real_pool_size={max_live_pool_size}")
        print(f"live_pool_max_placeholder_pool_size={max_placeholder_pool_size}")
        print(f"live_pool_max_combined_pool_size={max_combined_pool_size}")
    if slow_stash_ranks:
        print("slow_stash_floor4_ranks=" + ", ".join(f"{r}:{n}" for r, n in sorted(slow_stash_ranks.items())))
    if quarantine_ranks:
        print("quarantine_floor4_ranks=" + ", ".join(f"{r}:{n}" for r, n in sorted(quarantine_ranks.items())))
    if binding_mismatch_ranks:
        print("binding_mismatch_floor4_ranks=" + ", ".join(f"{r}:{n}" for r, n in sorted(binding_mismatch_ranks.items())))
    if binding_unknown_ranks:
        print("binding_unknown_floor4_ranks=" + ", ".join(f"{r}:{n}" for r, n in sorted(binding_unknown_ranks.items())))
    if fallback_ranks:
        print("fallback_floor4_ranks=" + ", ".join(f"{r}:{n}" for r, n in sorted(fallback_ranks.items())))
    if unpooled_rebuild_ranks:
        print("unpooled_rebuild_floor4_ranks=" + ", ".join(f"{r}:{n}" for r, n in sorted(unpooled_rebuild_ranks.items())))
    if skipped_rebuild_ranks:
        print("rebuild_skipped_floor4_ranks=" + ", ".join(f"{r}:{n}" for r, n in sorted(skipped_rebuild_ranks.items())))
    sequence = explicit_floor4_sequence or binding_floor4_sequence
    if sequence:
        print("floor4_rank_sequence=" + " -> ".join(str(r) for r in sequence))
        transitions = []
        repeated = 0
        overlapping = 0
        disjoint = 0
        for prev, cur in zip(sequence, sequence[1:]):
            overlap = tuple(sorted(set(prev) & set(cur)))
            if prev == cur:
                repeated += 1
            elif overlap:
                overlapping += 1
            else:
                disjoint += 1
            transitions.append(
                f"{prev}->{cur}:overlap={overlap}:overlap_size={len(overlap)}")
        if transitions:
            print("floor4_rank_transitions=" + " | ".join(transitions))
            print(
                "floor4_rank_transition_summary="
                f"repeat={repeated},overlap={overlapping},disjoint={disjoint}")
    if live_pool_ranks or slow_stash_ranks or quarantine_ranks:
        live_pool_rank_set = set(live_pool_ranks)
        slow_rank_set = set(slow_stash_ranks)
        quarantine_rank_set = set(quarantine_ranks)
        missed_slow = sorted(slow_rank_set - live_pool_rank_set)
        retired_after_seen = sorted(
            (slow_rank_set | quarantine_rank_set) & live_pool_rank_set)
        if missed_slow:
            print("live_pool_missed_slow_stash_ranks=" + ", ".join(str(r) for r in missed_slow))
        if retired_after_seen:
            print("live_pool_retired_after_seen_ranks=" + ", ".join(str(r) for r in retired_after_seen))

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
