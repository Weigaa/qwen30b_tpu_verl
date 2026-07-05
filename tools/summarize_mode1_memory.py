#!/usr/bin/env python3
"""Summarize mode1 memory pressure and full-world group release signals."""

from __future__ import annotations

import re
import sys
from collections import defaultdict
from pathlib import Path


MEM_RE = re.compile(
    r"Mode1 parity worker memory: tag=(?P<tag>\S+) rank=(?P<rank>\d+) "
    r"free_bytes=(?P<free>\d+) total_bytes=(?P<total>\d+) "
    r"torch_current=(?P<torch_current>\d+) torch_reserved=(?P<torch_reserved>\d+) "
    r"non_torch=(?P<non_torch>\d+) total_allocated=(?P<total_allocated>\d+)"
)
ROLLOUT_DONE_RE = re.compile(r"rollout_output_time(?:_s)?:\s*(?P<seconds>[0-9.]+)")
TRAIN_PROGRESS_RE = re.compile(r"Training Progress:\s+\d+%\|.*?\|\s*(?P<done>\d+)/(?P<total>\d+)")
KV_RE = re.compile(r"GPU KV cache size:\s*(?P<tokens>[0-9,]+)\s+tokens")
CONCURRENCY_RE = re.compile(r"Maximum concurrency[^:]*:\s*(?P<conc>[0-9.]+)x")
FULLWORLD_RANKS_TEXT = (
    "ranks=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15)"
)
FULLWORLD_GROUP_RANKS_TEXT = (
    "group_ranks=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15)"
)


def _mb(value: int) -> float:
    return value / 1024.0 / 1024.0


def _gb(value: int) -> float:
    return value / 1024.0 / 1024.0 / 1024.0


def _print_memory_rows(label: str, rows: list[dict[str, int | str]]) -> None:
    if not rows:
        return
    min_free = min(int(row["free"]) for row in rows)
    max_non_torch = max(int(row["non_torch"]) for row in rows)
    max_torch_current = max(int(row["torch_current"]) for row in rows)
    max_total_allocated = max(int(row["total_allocated"]) for row in rows)
    min_free_row = min(rows, key=lambda row: int(row["free"]))
    max_non_torch_row = max(rows, key=lambda row: int(row["non_torch"]))
    print(
        "memory_by_tag "
        f"tag={label} samples={len(rows)} "
        f"min_free_gb={_gb(min_free):.2f} "
        f"max_non_torch_gb={_gb(max_non_torch):.2f} "
        f"max_torch_current_gb={_gb(max_torch_current):.2f} "
        f"max_total_allocated_gb={_gb(max_total_allocated):.2f} "
        f"min_free_rank={min_free_row['rank']} min_free_tag={min_free_row['tag']} "
        f"max_non_torch_rank={max_non_torch_row['rank']} "
        f"max_non_torch_tag={max_non_torch_row['tag']} "
        f"min_free_mb={_mb(min_free):.0f} "
        f"max_non_torch_mb={_mb(max_non_torch):.0f}"
    )


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: summarize_mode1_memory.py <log_path>", file=sys.stderr)
        return 2

    log_path = Path(sys.argv[1])
    text = log_path.read_text(errors="replace")
    lines = text.splitlines()

    current_rollout = 0
    rows_by_rollout: dict[int, list[dict[str, int | str]]] = defaultdict(list)
    rows_by_tag: dict[str, list[dict[str, int | str]]] = defaultdict(list)
    rollout_times: list[float] = []
    max_train_done = 0
    train_total = 0
    kv_tokens: str = "NA"
    max_concurrency: str = "NA"
    release_fullworld_count = 0
    fullworld_pre_rebuild_release_count = 0
    fullworld_retired_immediate_count = 0
    fullworld_deferred_count = 0
    fullworld_direct_destroy_fallback_count = 0
    fullworld_cache_drop_count = 0
    fullworld_registry_drop_count = 0
    fullworld_shrink_drop_count = 0
    force_drop_fullworld_classify_count = 0
    fullworld_cache_keep_with_shrink_target_count = 0
    fullworld_stash_release_all_reasons_count = 0
    fullworld_retired_immediate_all_reasons_count = 0
    fullworld_direct_destroy_fallback_all_reasons_count = 0
    pre_rebuild_stash_release_count = 0
    fullworld_env = "NA"
    runtime_error_count = 0
    moe_dispatch_error_count = 0
    acl_error_count = 0

    for line in lines:
        if "drop_fullworld_after_shrink=" in line:
            match = re.search(r"drop_fullworld_after_shrink=([^\s]+)", line)
            if match:
                fullworld_env = match.group(1)

        kv_match = KV_RE.search(line)
        if kv_match:
            kv_tokens = kv_match.group("tokens").replace(",", "")

        conc_match = CONCURRENCY_RE.search(line)
        if conc_match:
            max_concurrency = conc_match.group("conc")

        progress_match = TRAIN_PROGRESS_RE.search(line)
        if progress_match:
            max_train_done = max(max_train_done, int(progress_match.group("done")))
            train_total = max(train_total, int(progress_match.group("total")))

        rollout_match = ROLLOUT_DONE_RE.search(line)
        if rollout_match:
            rollout_times.append(float(rollout_match.group("seconds")))
            current_rollout += 1

        if "release_fullworld=1" in line:
            release_fullworld_count += 1
        if "force_drop_fullworld=1" in line:
            force_drop_fullworld_classify_count += 1
        if "mode1_fullworld_shrink_drop" in line:
            fullworld_shrink_drop_count += 1
        if ("event=stash_cache_keep" in line
                and FULLWORLD_RANKS_TEXT in line
                and "target=(" in line
                and "target=()" not in line
                and "target=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15)"
                not in line):
            fullworld_cache_keep_with_shrink_target_count += 1
        if ("event=stash_release_start" in line
                and FULLWORLD_RANKS_TEXT in line):
            fullworld_stash_release_all_reasons_count += 1
        if ("Elastic mode1 pre-rebuild releases live old group:" in line
                and "release_fullworld=1" in line):
            fullworld_pre_rebuild_release_count += 1
        if ("Elastic parallel group retired immediately:" in line
                and "reason=mode1_pre_rebuild" in line
                and "group_ranks=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15)"
                in line):
            fullworld_retired_immediate_count += 1
        if ("Elastic parallel group retired immediately:" in line
                and FULLWORLD_GROUP_RANKS_TEXT in line):
            fullworld_retired_immediate_all_reasons_count += 1
        if ("Elastic parallel group destroy deferred:" in line
                and "reason=mode1_pre_rebuild" in line
                and "group_ranks=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15)"
                in line):
            fullworld_deferred_count += 1
        if ("Elastic mode1 low-floor stashed group direct destroy fallback:" in line
                and "reason=mode1_pre_rebuild" in line
                and "group_ranks=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15)"
                in line):
            fullworld_direct_destroy_fallback_count += 1
        if ("direct destroy fallback:" in line
                and FULLWORLD_GROUP_RANKS_TEXT in line):
            fullworld_direct_destroy_fallback_all_reasons_count += 1
        if "Elastic mode1 full-world group cache drop summary:" in line:
            cache_match = re.search(
                r"fullworld_cache_dropped_groups=(\d+)", line)
            registry_match = re.search(
                r"fullworld_registry_dropped_groups=(\d+)", line)
            if cache_match:
                fullworld_cache_drop_count += int(cache_match.group(1))
            if registry_match:
                fullworld_registry_drop_count += int(registry_match.group(1))
        if "reason=mode1_pre_rebuild" in line:
            pre_rebuild_stash_release_count += 1
        if "RuntimeError:" in line:
            runtime_error_count += 1
        if "aclnnMoeDistributeDispatch" in line:
            moe_dispatch_error_count += 1
        if "ERR00100" in line or "ACL" in line and "failed" in line:
            acl_error_count += 1

        mem_match = MEM_RE.search(line)
        if mem_match:
            row = {
                "tag": mem_match.group("tag"),
                "rank": int(mem_match.group("rank")),
                "free": int(mem_match.group("free")),
                "total": int(mem_match.group("total")),
                "torch_current": int(mem_match.group("torch_current")),
                "torch_reserved": int(mem_match.group("torch_reserved")),
                "non_torch": int(mem_match.group("non_torch")),
                "total_allocated": int(mem_match.group("total_allocated")),
            }
            rows_by_rollout[current_rollout].append(row)
            rows_by_tag[str(row["tag"])].append(row)

    print(f"memory_summary log={log_path}")
    print(
        "memory_config "
        f"kv_tokens={kv_tokens} max_concurrency={max_concurrency} "
        f"drop_fullworld_after_shrink={fullworld_env} "
        f"release_fullworld_count={release_fullworld_count}"
    )
    print(
        "run_status "
        f"rollout_done_count={len(rollout_times)} "
        f"train_progress={max_train_done}/{train_total or 'NA'} "
        f"runtime_errors={runtime_error_count} "
        f"moe_dispatch_errors={moe_dispatch_error_count} "
        f"acl_errors={acl_error_count}"
    )
    print(
        "fullworld_release "
        f"pre_rebuild_release_count={fullworld_pre_rebuild_release_count} "
        f"stash_release_count={pre_rebuild_stash_release_count} "
        f"retired_immediate_count={fullworld_retired_immediate_count} "
        f"deferred_count={fullworld_deferred_count} "
        f"direct_destroy_fallback_count={fullworld_direct_destroy_fallback_count} "
        f"cache_drop_count={fullworld_cache_drop_count} "
        f"registry_drop_count={fullworld_registry_drop_count} "
        f"fullworld_shrink_drop_count={fullworld_shrink_drop_count} "
        f"force_drop_fullworld_classify_count={force_drop_fullworld_classify_count} "
        f"cache_keep_with_shrink_target_count={fullworld_cache_keep_with_shrink_target_count} "
        f"stash_release_all_reasons_count={fullworld_stash_release_all_reasons_count} "
        f"retired_immediate_all_reasons_count={fullworld_retired_immediate_all_reasons_count} "
        f"direct_destroy_fallback_all_reasons_count={fullworld_direct_destroy_fallback_all_reasons_count}"
    )
    if rollout_times:
        formatted = ",".join(f"{value:.2f}" for value in rollout_times)
        print(f"rollout_times seconds={formatted}")

    for rollout, rows in sorted(rows_by_rollout.items()):
        if not rows:
            continue
        min_free = min(int(row["free"]) for row in rows)
        max_non_torch = max(int(row["non_torch"]) for row in rows)
        max_torch_current = max(int(row["torch_current"]) for row in rows)
        max_total_allocated = max(int(row["total_allocated"]) for row in rows)
        min_free_row = min(rows, key=lambda row: int(row["free"]))
        max_non_torch_row = max(rows, key=lambda row: int(row["non_torch"]))
        label = f"after_rollout_{rollout}" if rollout else "before_rollout_1"
        print(
            "memory_by_rollout "
            f"bucket={label} samples={len(rows)} "
            f"min_free_gb={_gb(min_free):.2f} "
            f"max_non_torch_gb={_gb(max_non_torch):.2f} "
            f"max_torch_current_gb={_gb(max_torch_current):.2f} "
            f"max_total_allocated_gb={_gb(max_total_allocated):.2f} "
            f"min_free_rank={min_free_row['rank']} min_free_tag={min_free_row['tag']} "
            f"max_non_torch_rank={max_non_torch_row['rank']} "
            f"max_non_torch_tag={max_non_torch_row['tag']} "
            f"min_free_mb={_mb(min_free):.0f} "
            f"max_non_torch_mb={_mb(max_non_torch):.0f}"
        )

    for tag in (
            "before_old_floor_group_cache_drop_before_rebuild",
            "after_old_floor_group_cache_drop_before_rebuild",
            "after_elastic_group_release_cleanup",
            "after_post_shrink_staging_state_release",
    ):
        _print_memory_rows(tag, rows_by_tag.get(tag, []))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
