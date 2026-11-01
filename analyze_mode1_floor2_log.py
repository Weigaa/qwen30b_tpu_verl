#!/usr/bin/env python3
"""Summarize mode=1 floor=2 shrink timing from an elastic rollout log."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
NUMBER_RE = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)"
LOGFILE_RE = re.compile(r"logfile=([^\s'\"<>]+)")


def strip_ansi(text: str) -> str:
    return ANSI_RE.sub("", text)


def parse_key_values(line: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for match in re.finditer(r"([A-Za-z0-9_]+)=([^,\s]+)", line):
        values[match.group(1)] = match.group(2)
    return values


def parse_float(values: dict[str, str], key: str) -> float | None:
    raw = values.get(key)
    if raw is None:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def sum_key(rows: list[dict[str, str]], key: str) -> float:
    total = 0.0
    for row in rows:
        value = parse_float(row, key)
        if value is not None:
            total += value
    return total


def resolve_worker_log_path(log_path: Path) -> tuple[Path, Path | None]:
    """Allow passing a local launcher.log and follow its logfile= pointer."""
    if log_path.name != "launcher.log":
        return log_path, None

    resolved: list[Path] = []
    with log_path.open("r", errors="replace") as f:
        for raw_line in f:
            line = strip_ansi(raw_line.rstrip())
            for match in LOGFILE_RE.finditer(line):
                candidate = Path(match.group(1))
                if not candidate.is_absolute():
                    candidate = (log_path.parent / candidate).resolve()
                if candidate.exists():
                    resolved.append(candidate)

    if not resolved:
        return log_path, None
    return resolved[-1], log_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("log", type=Path)
    parser.add_argument(
        "--slow-ms",
        type=float,
        default=1000.0,
        help="report rows with any direct-NPU timing field at or above this ms")
    args = parser.parse_args()
    requested_log = args.log
    args.log, launcher_log = resolve_worker_log_path(args.log)

    header_lines: list[str] = []
    rollout_times: list[float] = []
    chunk_plans: list[str] = []
    final_breakdowns: list[dict[str, str]] = []
    phase_breakdowns: list[dict[str, str]] = []
    deferred_destroy_rows: list[dict[str, str]] = []
    stashed_release_rows: list[dict[str, str]] = []
    warmup_skip_lines: list[str] = []
    error_lines: list[str] = []

    rollout_re = re.compile(r"rollout_output_time_s\s*[:=]\s*(%s)" %
                            NUMBER_RE)
    final_breakdown_marker = "Elastic shrink stream import breakdown:"
    phase_breakdown_marker = "Elastic parallel shrink phase breakdown:"
    chunk_plan_marker = "Elastic shrink direct NPU batch chunk plan:"
    deferred_destroy_marker = "Elastic parallel group destroy deferred:"
    stashed_release_marker = "Elastic parallel stashed group released:"

    with args.log.open("r", errors="replace") as f:
        for raw_line in f:
            line = strip_ansi(raw_line.rstrip())
            if ("[full redundancy experiment]" in line
                    or "[moe pattern stats]" in line
                    or "[local nm_mode1_final]" in line):
                header_lines.append(line)
            match = rollout_re.search(line)
            if match:
                rollout_times.append(float(match.group(1)))
            if chunk_plan_marker in line and "active_ranks=[14, 15]" in line:
                chunk_plans.append(line)
            if (final_breakdown_marker in line
                    and "active_ranks=[14, 15]" in line):
                final_breakdowns.append(parse_key_values(line))
            if phase_breakdown_marker in line:
                row = parse_key_values(line)
                match = re.search(r"active_ranks=\[([^\]]*)\]", line)
                if match:
                    ranks = tuple(
                        int(part.strip()) for part in match.group(1).split(",")
                        if part.strip())
                    row["active_tuple"] = str(ranks)
                    row["stage"] = str(len(ranks))
                phase_breakdowns.append(row)
            if deferred_destroy_marker in line:
                deferred_destroy_rows.append(parse_key_values(line))
            if stashed_release_marker in line:
                stashed_release_rows.append(parse_key_values(line))
            if ("Elastic post-shrink DP all_reduce warmup skipped" in line
                    or "Elastic post-shrink MoE dispatch warmup skipped" in line):
                warmup_skip_lines.append(line)
            if ("RuntimeError" in line or "Memory_Allocation_Failure" in line
                    or "HcclAllreduce" in line or "Timed out waiting" in line):
                error_lines.append(line)

    print(f"input_log={requested_log}")
    print(f"analyzed_worker_log={args.log}")
    if launcher_log is not None:
        print(f"resolved_from_launcher_log={launcher_log}")
    if rollout_times:
        deduped_rollout_times: list[float] = []
        for value in rollout_times:
            if not deduped_rollout_times or value != deduped_rollout_times[-1]:
                deduped_rollout_times.append(value)
        print("rollout_output_time_s=" +
              ",".join(f"{value:.6f}" for value in deduped_rollout_times))
    else:
        print("rollout_output_time_s=<missing>")

    interesting_header_keys = (
        "mode",
        "floor",
        "mode1_batch_direct_npu_import",
        "mode1_allow_batch_index_select_export",
        "mode1_direct_npu_import_batch_experts",
        "batch_direct_npu_import",
        "allow_batch_index_select_export",
        "direct_npu_import_batch_experts",
        "mode1_native_kv_cap",
        "custom_mode1_kv_diag",
        "old_floor_drop_verbose",
        "mode1_gc_after_mc2_destroy",
        "mode1_sync_after_mc2_destroy",
        "mode1_keep_floor4_group_cache",
        "mode1_keep_floor4_group_kinds",
        "mode1_drop_old_floor_before_rebuild",
        "mode1_release_live_old_floor_on_rebuild",
        "mode1_pre_rebuild_destroy_floor_sizes",
        "mode1_defer_group_destroy",
        "mode1_defer_destroy_floor_sizes",
        "mode1_sync_before_device_pg_retire",
        "mode1_destroy_device_pg_on_retire",
        "mode1_enable_post_shrink_dp_warmup",
        "mode1_cpu_dp_metadata_sync",
    )
    for header in header_lines[:6]:
        values = parse_key_values(header)
        selected = {
            key: values[key]
            for key in interesting_header_keys
            if key in values
        }
        if selected:
            print("header " + " ".join(f"{k}={v}"
                                       for k, v in selected.items()))

    print(f"final_4to2_breakdown_rows={len(final_breakdowns)}")
    if final_breakdowns:
        timing_keys = (
            "send_export_ms",
            "send_post_ms",
            "send_wait_ms",
            "recv_alias_ms",
            "recv_post_ms",
            "recv_wait_ms",
            "total_ms",
            "wall_ms",
        )
        for key in timing_keys:
            keyed_rows = [
                (parse_float(row, key), row)
                for row in final_breakdowns
                if parse_float(row, key) is not None
            ]
            if not keyed_rows:
                continue
            value, row = max(keyed_rows, key=lambda item: item[0] or 0.0)
            print(
                f"max_{key}={value:.2f} "
                f"rank={row.get('rank', '?')} "
                f"layer={row.get('layer', '?')} "
                f"chunks={row.get('chunks', '?')} "
                f"send_contig={row.get('send_export_contiguous_chunks', '?')} "
                f"send_index={row.get('send_export_index_select_chunks', '?')} "
                f"recv_direct={row.get('recv_direct_slot_chunks', '?')} "
                f"recv_temp={row.get('recv_temp_buffer_chunks', '?')}")

        counter_keys = (
            "chunks",
            "send_export_contiguous_chunks",
            "send_export_index_select_chunks",
            "recv_direct_slot_chunks",
            "recv_temp_buffer_chunks",
        )
        print("counter_sums " + " ".join(
            f"{key}={sum_key(final_breakdowns, key):.0f}"
            for key in counter_keys))

        source_rows = [
            row for row in final_breakdowns
            if parse_float(row, "send_export_ms") is not None
            and parse_float(row, "send_export_ms") > 0
        ]
        target_rows = [
            row for row in final_breakdowns
            if parse_float(row, "recv_post_ms") is not None
            and parse_float(row, "recv_post_ms") > 0
        ]
        print(
            "source_path_summary "
            f"rows={len(source_rows)} "
            f"contiguous_chunks={sum_key(source_rows, 'send_export_contiguous_chunks'):.0f} "
            f"index_select_chunks={sum_key(source_rows, 'send_export_index_select_chunks'):.0f}")
        print(
            "target_path_summary "
            f"rows={len(target_rows)} "
            f"direct_slot_chunks={sum_key(target_rows, 'recv_direct_slot_chunks'):.0f} "
            f"temp_buffer_chunks={sum_key(target_rows, 'recv_temp_buffer_chunks'):.0f}")

        slow_fields = (
            "send_export_ms",
            "send_post_ms",
            "send_wait_ms",
            "recv_alias_ms",
            "recv_post_ms",
            "recv_wait_ms",
            "wall_ms",
        )
        slow_rows: list[tuple[float, str, dict[str, str]]] = []
        for row in final_breakdowns:
            for field in slow_fields:
                value = parse_float(row, field)
                if value is not None and value >= args.slow_ms:
                    slow_rows.append((value, field, row))
        print(f"slow_rows_threshold_ms={args.slow_ms:.2f} count={len(slow_rows)}")
        for value, field, row in sorted(
                slow_rows, key=lambda item: item[0], reverse=True)[:12]:
            print(
                f"slow_row field={field} value_ms={value:.2f} "
                f"rank={row.get('rank', '?')} layer={row.get('layer', '?')} "
                f"mode={row.get('mode', '?')} chunks={row.get('chunks', '?')} "
                f"send_contig={row.get('send_export_contiguous_chunks', '?')} "
                f"send_index={row.get('send_export_index_select_chunks', '?')} "
                f"recv_direct={row.get('recv_direct_slot_chunks', '?')} "
                f"recv_temp={row.get('recv_temp_buffer_chunks', '?')}")

    print(f"phase_breakdown_rows={len(phase_breakdowns)}")
    if phase_breakdowns:
        phase_keys = (
            "prepare_payload_ms",
            "preload_import_ms",
            "detach_ms",
            "rebuild_ms",
            "rebuild_dp_ms",
            "rebuild_ep_ms",
            "rebuild_mc2_ms",
            "non_member_sequence_ms",
            "non_member_sequence_dp_ms",
            "non_member_sequence_ep_ms",
            "non_member_sequence_mc2_ms",
            "non_member_sequence_device_new_group_ms",
            "non_member_sequence_cpu_new_group_ms",
            "non_member_sequence_destroy_ms",
            "non_member_sequence_signature_ms",
            "non_member_sequence_log_ms",
            "non_member_sequence_groups",
            "pre_refresh_source_barrier_ms",
            "refresh_ms",
            "release_staging_ms",
            "old_floor_group_drop_barrier_ms",
            "drop_old_floor_group_call_wall_ms",
            "drop_old_floor_group_cache_ms",
            "old_floor_pre_memory_log_ms",
            "old_floor_post_memory_log_ms",
            "drop_old_floor_group_destroy_ms",
            "drop_old_floor_group_cleanup_ms",
            "drop_old_floor_group_cleanup_gc_ms",
            "drop_old_floor_group_cleanup_empty_cache_ms",
            "drop_old_floor_group_cleanup_sync_ms",
            "drop_old_floor_group_cleanup_memory_log_ms",
            "drop_old_floor_group_memory_log_ms",
            "drop_old_floor_group_row_log_ms",
            "drop_old_floor_group_summary_log_ms",
            "drop_old_floor_group_signature_sweep_ms",
            "drop_old_floor_group_release_ref_ms",
            "drop_old_floor_group_dropped_groups",
            "drop_old_floor_group_deferred_groups",
            "drop_old_floor_group_dropped_signatures",
            "post_old_floor_group_drop_barrier_ms",
            "warmup_ms",
            "hidden_tail_ms",
            "total_ms",
        )
        stages = sorted(
            set(row.get("stage", "?") for row in phase_breakdowns),
            key=lambda value: (999 if value == "?" else int(value)))
        for stage in stages:
            rows = [row for row in phase_breakdowns if row.get("stage", "?") == stage]
            print(
                f"phase_stage stage={stage} rows={len(rows)} "
                f"active={rows[0].get('active_tuple', '?')}")
            for key in phase_keys:
                keyed_rows = [
                    (parse_float(row, key), row)
                    for row in rows
                    if parse_float(row, key) is not None
                ]
                if not keyed_rows:
                    continue
                max_value, max_row = max(
                    keyed_rows, key=lambda item: item[0] or 0.0)
                values = [value for value, _ in keyed_rows if value is not None]
                mean_value = sum(values) / len(values)
                print(
                    f"  max_{key}={max_value:.2f} "
                    f"mean_{key}={mean_value:.2f} "
                    f"rank={max_row.get('rank', '?')} "
                    f"is_active={max_row.get('is_active', '?')}")

    print(f"chunk_plan_rows={len(chunk_plans)}")
    for line in chunk_plans[:8]:
        print("chunk_plan " + line)
    print(f"deferred_destroy_rows={len(deferred_destroy_rows)}")
    if deferred_destroy_rows:
        by_kind: dict[str, int] = {}
        by_size: dict[str, int] = {}
        retire_ms_values: list[float] = []
        npu_sync_ms_values: list[float] = []
        device_pg_destroy_ms_values: list[float] = []
        dropped_pg_refs = 0.0
        deferred_device_pg_refs = 0.0
        deferred_cpu_pg_refs = 0.0
        for row in deferred_destroy_rows:
            by_kind[row.get("group_kind", "?")] = by_kind.get(
                row.get("group_kind", "?"), 0) + 1
            ranks = row.get("group_ranks", "")
            rank_count = len([part for part in ranks.strip("()").split(",") if part.strip()])
            key = str(rank_count) if rank_count else "?"
            by_size[key] = by_size.get(key, 0) + 1
            if "retire_wrapper_ms" in row:
                retire_ms_values.append(safe_float(row.get("retire_wrapper_ms")))
            if "npu_sync_ms" in row:
                npu_sync_ms_values.append(safe_float(row.get("npu_sync_ms")))
            if "device_pg_destroy_ms" in row:
                device_pg_destroy_ms_values.append(
                    safe_float(row.get("device_pg_destroy_ms")))
            if "dropped_pg_refs" in row:
                dropped_pg_refs += safe_float(row.get("dropped_pg_refs"))
            if "deferred_device_pg_refs" in row:
                deferred_device_pg_refs += safe_float(
                    row.get("deferred_device_pg_refs"))
            if "deferred_cpu_pg_refs" in row:
                deferred_cpu_pg_refs += safe_float(
                    row.get("deferred_cpu_pg_refs"))
        print("deferred_destroy_by_kind " + " ".join(
            f"{key}={value}" for key, value in sorted(by_kind.items())))
        print("deferred_destroy_by_size " + " ".join(
            f"{key}={value}" for key, value in sorted(by_size.items(), key=lambda item: item[0])))
        if retire_ms_values:
            print(
                "deferred_destroy_retire_wrapper_ms "
                f"max={max(retire_ms_values):.2f} "
                f"mean={statistics.mean(retire_ms_values):.2f} "
                f"sum={sum(retire_ms_values):.2f} "
                f"dropped_pg_refs={dropped_pg_refs:.0f}")
        if npu_sync_ms_values:
            print(
                "deferred_destroy_npu_sync_ms "
                f"max={max(npu_sync_ms_values):.2f} "
                f"mean={statistics.mean(npu_sync_ms_values):.2f} "
                f"sum={sum(npu_sync_ms_values):.2f}")
        if device_pg_destroy_ms_values:
            print(
                "deferred_destroy_device_pg_destroy_ms "
                f"max={max(device_pg_destroy_ms_values):.2f} "
                f"mean={statistics.mean(device_pg_destroy_ms_values):.2f} "
                f"sum={sum(device_pg_destroy_ms_values):.2f} "
                f"deferred_device_pg_refs={deferred_device_pg_refs:.0f} "
                f"deferred_cpu_pg_refs={deferred_cpu_pg_refs:.0f}")
        for row in deferred_destroy_rows[:12]:
            print(
                "deferred_destroy "
                f"rank={row.get('rank', '?')} "
                f"kind={row.get('group_kind', '?')} "
                f"ranks={row.get('group_ranks', '?')} "
                f"reason={row.get('reason', '?')} "
                f"count={row.get('deferred_count', '?')} "
                f"retire_wrapper_ms={row.get('retire_wrapper_ms', '?')} "
                f"npu_sync_ms={row.get('npu_sync_ms', '?')} "
                f"device_pg_destroy_ms={row.get('device_pg_destroy_ms', '?')} "
                f"destroy_device_pg={row.get('destroy_device_pg', '?')} "
                f"dropped_pg_refs={row.get('dropped_pg_refs', '?')} "
                f"deferred_device_pg_refs={row.get('deferred_device_pg_refs', '?')} "
                f"deferred_cpu_pg_refs={row.get('deferred_cpu_pg_refs', '?')}")
    print(f"stashed_release_rows={len(stashed_release_rows)}")
    if stashed_release_rows:
        by_reason: dict[str, int] = {}
        by_kind: dict[str, int] = {}
        by_size: dict[str, int] = {}
        already_retired = 0
        for row in stashed_release_rows:
            by_reason[row.get("reason", "?")] = by_reason.get(
                row.get("reason", "?"), 0) + 1
            by_kind[row.get("group_kind", "?")] = by_kind.get(
                row.get("group_kind", "?"), 0) + 1
            ranks = row.get("old_ranks", "")
            rank_count = len(
                [part for part in ranks.strip("()").split(",") if part.strip()])
            key = str(rank_count) if rank_count else "?"
            by_size[key] = by_size.get(key, 0) + 1
            if row.get("already_retired", "").lower() in ("true", "1"):
                already_retired += 1
        print("stashed_release_by_reason " + " ".join(
            f"{key}={value}" for key, value in sorted(by_reason.items())))
        print("stashed_release_by_kind " + " ".join(
            f"{key}={value}" for key, value in sorted(by_kind.items())))
        print("stashed_release_by_size " + " ".join(
            f"{key}={value}" for key, value in sorted(
                by_size.items(), key=lambda item: item[0])))
        print(f"stashed_release_already_retired={already_retired}")
        for row in stashed_release_rows[:12]:
            print(
                "stashed_release "
                f"rank={row.get('rank', '?')} "
                f"kind={row.get('group_kind', '?')} "
                f"ranks={row.get('old_ranks', '?')} "
                f"reason={row.get('reason', '?')} "
                f"destroy_ms={row.get('destroy_ms', '?')} "
                f"cleanup_ms={row.get('cleanup_ms', '?')} "
                f"already_retired={row.get('already_retired', '?')}")
    print(f"warmup_skip_rows={len(warmup_skip_lines)}")
    for line in warmup_skip_lines[:12]:
        print("warmup_skip " + line)
    print(f"error_signal_rows={len(error_lines)}")
    for line in error_lines[-12:]:
        print("error_signal " + line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
