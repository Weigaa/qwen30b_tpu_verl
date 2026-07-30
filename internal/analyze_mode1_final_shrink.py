#!/usr/bin/env python3
"""Summarize the final mode=1 shrink stage from an elastic rollout log.

Without an argument, this script picks the newest local elastic rollout log.  It
focuses on the smallest active-rank stage in the log, usually the final
`floor=2` shrink `[14, 15]`, and separates where the time is spent:
pre-import NPU drain, direct-NPU export, P2P wait, preload, rebuild, and warmup.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable

ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
ROLLOUT_RE = re.compile(r"rollout_output_time(?:_s)?[:=]\s*([0-9.]+)")
PRELOAD_RE = re.compile(
    r"Elastic shrink preload breakdown: rank=(?P<rank>\d+) "
    r"active_ranks=\[(?P<active>[^\]]*)\] "
    r"modules=(?P<modules>\d+) "
    r"filter_cpu_payload_ms=(?P<filter>[0-9.]+) "
    r"stream_import_ms=(?P<stream>[0-9.]+) "
    r"store_ms=(?P<store>[0-9.]+) "
    r"total_ms=(?P<total>[0-9.]+)")
STREAM_RE = re.compile(
    r"Elastic shrink stream import breakdown: rank=(?P<rank>\d+) "
    r"active_ranks=\[(?P<active>[^\]]*)\] "
    r"layer=(?P<layer>\d+) mode=(?P<mode>\w+) "
    r"transfer_pairs=(?P<transfer_pairs>[0-9.]+) "
    r"remote_experts=(?P<remote_experts>[0-9.]+) "
    r"local_only_experts=(?P<local_only_experts>[0-9.]+) "
    r"chunks=(?P<chunks>[0-9.]+) "
    r"local_copy_export_ms=(?P<local_copy_export_ms>[0-9.]+) "
    r"(?:send_pre_sync_ms=(?P<send_pre_sync_ms>[0-9.]+) )?"
    r"send_export_ms=(?P<send_export_ms>[0-9.]+) "
    r"send_pack_copy_ms=(?P<send_pack_copy_ms>[0-9.]+) "
    r"send_to_device_ms=(?P<send_to_device_ms>[0-9.]+) "
    r"send_wait_ms=(?P<send_wait_ms>[0-9.]+) "
    r"recv_wait_ms=(?P<recv_wait_ms>[0-9.]+) "
    r"recv_to_cpu_ms=(?P<recv_to_cpu_ms>[0-9.]+) "
    r"recv_store_ms=(?P<recv_store_ms>[0-9.]+) "
    r"total_ms=(?P<total_ms>[0-9.]+)")
PHASE_RE = re.compile(
    r"Elastic parallel shrink phase breakdown: rank=(?P<rank>\d+) "
    r"active_ranks=\[(?P<active>[^\]]*)\].*?"
    r"is_active=(?P<is_active>[01]).*?"
    r"prefetch_drain_ms=(?P<prefetch_drain_ms>[0-9.]+) "
    r"(?:mode1_pre_import_drain_ms=(?P<mode1_pre_import_drain_ms>[0-9.]+) )?"
    r"prepare_payload_ms=(?P<prepare_payload_ms>[0-9.]+) "
    r"preload_import_ms=(?P<preload_import_ms>[0-9.]+).*?"
    r"rebuild_ms=(?P<rebuild_ms>[0-9.]+).*?"
    r"refresh_ms=(?P<refresh_ms>[0-9.]+).*?"
    r"release_staging_ms=(?P<release_staging_ms>[0-9.]+) "
    r"drop_stale_group_cache_ms=(?P<drop_stale_group_cache_ms>[0-9.]+) "
    r"drop_old_floor_group_cache_ms=(?P<drop_old_floor_group_cache_ms>[0-9.]+).*?"
    r"warmup_ms=(?P<warmup_ms>[0-9.]+) "
    r"hidden_tail_ms=(?P<hidden_tail_ms>[0-9.]+) "
    r"total_ms=(?P<total_ms>[0-9.]+)")
DRAIN_RE = re.compile(
    r"Mode1 pre-shrink NPU drain done: rank=(?P<rank>\d+) "
    r"active_ranks=\[(?P<active>[^\]]*)\] "
    r"(?:(?:weight_device=(?P<weight_device>\S+) "
    r"cpu_barrier1_ms=(?P<cpu_barrier1_ms>[0-9.]+) "
    r"device_barrier_ms=(?P<device_barrier_ms>[0-9.]+) "
    r"sync_ms=(?P<sync_ms>[0-9.]+) "
    r"cpu_barrier2_ms=(?P<cpu_barrier2_ms>[0-9.]+) )?)"
    r"total_ms=(?P<total_ms>[0-9.]+)")
SLOW_EXPORT_RE = re.compile(
    r"Slow export_lossless_expert_npu_weights: "
    r"layer=(?P<layer>-?\d+) mode=(?P<mode>-?\d+) "
    r"experts=(?P<experts>\[[^\]]*\]) "
    r"local_slots=(?P<local_slots>\[[^\]]*\]) "
    r"total_ms=(?P<total_ms>[0-9.]+)")


def strip_ansi(text: str) -> str:
    return ANSI_RE.sub("", text)


def parse_active_ranks(text: str) -> tuple[int, ...]:
    text = text.strip()
    if not text:
        return ()
    return tuple(int(part.strip()) for part in text.split(",") if part.strip())


def latest_log() -> Path:
    patterns = (
        "wjeagerqwen30b-a3b-with_draft_breakdown_*_elastic.txt",
        "*_elastic.txt",
    )
    candidates: list[Path] = []
    for pattern in patterns:
        candidates.extend(Path.cwd().glob(pattern))
    unique = {path.resolve(): path for path in candidates if path.is_file()}
    if not unique:
        raise FileNotFoundError("no local *_elastic.txt log found")
    return max(unique.values(), key=lambda path: path.stat().st_mtime)


def float_row(row: dict, skip: Iterable[str]) -> dict:
    skip_set = set(skip)
    for key, value in list(row.items()):
        if key in skip_set:
            continue
        row[key] = float(value) if value is not None else 0.0
    return row


def load_log(path: Path) -> dict:
    rollout_s = None
    preload_rows: list[dict] = []
    stream_rows: list[dict] = []
    phase_rows: list[dict] = []
    drain_rows: list[dict] = []
    slow_exports: list[dict] = []

    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for lineno, raw in enumerate(f, start=1):
            line = strip_ansi(raw)
            match = ROLLOUT_RE.search(line)
            if match:
                rollout_s = float(match.group(1))

            match = PRELOAD_RE.search(line)
            if match:
                row = match.groupdict()
                row["lineno"] = lineno
                row["rank"] = int(row["rank"])
                row["modules"] = int(row["modules"])
                row["active_ranks"] = parse_active_ranks(row.pop("active"))
                preload_rows.append(float_row(row, {"lineno", "rank", "modules", "active_ranks"}))
                continue

            match = STREAM_RE.search(line)
            if match:
                row = match.groupdict()
                row["lineno"] = lineno
                row["rank"] = int(row["rank"])
                row["layer"] = int(row["layer"])
                row["active_ranks"] = parse_active_ranks(row.pop("active"))
                stream_rows.append(float_row(row, {"mode", "lineno", "rank", "layer", "active_ranks"}))
                continue

            match = PHASE_RE.search(line)
            if match:
                row = match.groupdict()
                row["lineno"] = lineno
                row["rank"] = int(row["rank"])
                row["is_active"] = int(row["is_active"])
                row["active_ranks"] = parse_active_ranks(row.pop("active"))
                phase_rows.append(float_row(row, {"lineno", "rank", "is_active", "active_ranks"}))
                continue

            match = DRAIN_RE.search(line)
            if match:
                row = match.groupdict()
                row["lineno"] = lineno
                row["rank"] = int(row["rank"])
                row["active_ranks"] = parse_active_ranks(row.pop("active"))
                for key in (
                        "cpu_barrier1_ms", "device_barrier_ms", "sync_ms",
                        "cpu_barrier2_ms", "total_ms"):
                    if row.get(key) is not None:
                        row[key] = float(row[key])
                drain_rows.append(row)
                continue

            match = SLOW_EXPORT_RE.search(line)
            if match:
                row = match.groupdict()
                row["lineno"] = lineno
                row["layer"] = int(row["layer"])
                row["mode"] = int(row["mode"])
                row["total_ms"] = float(row["total_ms"])
                slow_exports.append(row)

    return {
        "rollout_s": rollout_s,
        "preload_rows": preload_rows,
        "stream_rows": stream_rows,
        "phase_rows": phase_rows,
        "drain_rows": drain_rows,
        "slow_exports": slow_exports,
    }


def choose_final_active_ranks(data: dict) -> tuple[int, ...]:
    active_sets = {
        row["active_ranks"]
        for row in data["preload_rows"] + data["stream_rows"] + data["phase_rows"]
        if row["active_ranks"]
    }
    if not active_sets:
        return ()
    return min(active_sets, key=lambda ranks: (len(ranks), ranks))


def sort_rows(rows: list[dict]) -> list[dict]:
    return sorted(rows, key=lambda row: (row["rank"], row["lineno"]))


def fmt_ms(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"{value:.2f}"


def max_value(rows: list[dict], key: str) -> tuple[float, dict | None]:
    if not rows:
        return 0.0, None
    row = max(rows, key=lambda item: item.get(key, 0.0))
    return float(row.get(key, 0.0)), row


def print_rows(label: str, rows: list[dict], fields: list[str]) -> None:
    print(f"\n[{label}]")
    if not rows:
        print("not found")
        return
    for row in rows:
        parts = [f"line={row['lineno']}", f"rank={row['rank']}"]
        if "is_active" in row:
            parts.append(f"is_active={row['is_active']}")
        for field in fields:
            value = row.get(field)
            if isinstance(value, float):
                parts.append(f"{field}={fmt_ms(value)}")
            else:
                parts.append(f"{field}={value}")
        print(" ".join(parts))


def diagnose(rollout_s: float | None, stream_rows: list[dict],
             phase_rows: list[dict], drain_rows: list[dict]) -> str:
    max_stream_total, _ = max_value(stream_rows, "total_ms")
    max_send_pre_sync, _ = max_value(stream_rows, "send_pre_sync_ms")
    max_send_export, _ = max_value(stream_rows, "send_export_ms")
    max_preload, _ = max_value(phase_rows, "preload_import_ms")
    max_mode1_drain, _ = max_value(phase_rows, "mode1_pre_import_drain_ms")
    max_drain_log, _ = max_value(drain_rows, "total_ms")
    max_drain = max(max_mode1_drain, max_drain_log)
    max_rebuild, _ = max_value(phase_rows, "rebuild_ms")

    if max_drain > 100_000:
        return ("slow_at_mode1_pre_import_drain: pending NPU work is already "
                "large before direct import/export")
    if max_send_pre_sync > 100_000:
        return ("slow_at_send_pre_sync: export diag is draining pending NPU work; "
                "disable VLLM_ASCEND_MODE1_EXPORT_DIAG for perf runs")
    if max_send_export > 100_000:
        return ("slow_at_send_export: first direct-NPU export/alias/send path is "
                "draining or blocking on pending NPU work")
    if max_preload > 100_000 or max_stream_total > 100_000:
        return "slow_at_preload_stream_import: direct-NPU preload remains the bottleneck"
    if max_rebuild > 10_000:
        return "slow_at_rebuild: communicator rebuild/cache lifecycle is the bottleneck"
    if rollout_s is not None and rollout_s > 180:
        return "rollout_slow_but_final_shrink_not_dominant: inspect decode stage before shrink"
    return "final_shrink_not_showing_350s_bottleneck"


def print_summary(path: Path, data: dict) -> None:
    final_active_ranks = choose_final_active_ranks(data)
    preload_rows = sort_rows([
        row for row in data["preload_rows"]
        if row["active_ranks"] == final_active_ranks
    ])
    stream_rows = sort_rows([
        row for row in data["stream_rows"]
        if row["active_ranks"] == final_active_ranks
    ])
    phase_rows = sort_rows([
        row for row in data["phase_rows"]
        if row["active_ranks"] == final_active_ranks
    ])
    drain_rows = sort_rows([
        row for row in data["drain_rows"]
        if row["active_ranks"] == final_active_ranks
    ])
    slow_exports = sorted(data["slow_exports"],
                          key=lambda row: (row["total_ms"], row["lineno"]),
                          reverse=True)

    print(f"log={path}")
    print(f"rollout_output_time_s={fmt_ms(data['rollout_s'])}")
    print(f"final_active_ranks={list(final_active_ranks)}")
    print(
        "diagnosis=" + diagnose(data["rollout_s"], stream_rows, phase_rows, drain_rows)
    )

    print_rows(
        "phase_breakdown", phase_rows,
        [
            "mode1_pre_import_drain_ms", "prepare_payload_ms",
            "preload_import_ms", "rebuild_ms", "refresh_ms",
            "release_staging_ms", "drop_old_floor_group_cache_ms", "warmup_ms",
            "hidden_tail_ms", "total_ms",
        ],
    )
    print_rows(
        "preload_breakdown", preload_rows,
        ["modules", "filter", "stream", "store", "total"],
    )
    print_rows(
        "stream_import_breakdown", stream_rows,
        [
            "mode", "transfer_pairs", "remote_experts", "chunks",
            "send_pre_sync_ms", "send_export_ms", "send_wait_ms",
            "recv_wait_ms", "recv_to_cpu_ms", "total_ms",
        ],
    )
    print_rows(
        "mode1_pre_shrink_drain_logs", drain_rows,
        [
            "weight_device", "cpu_barrier1_ms", "device_barrier_ms",
            "sync_ms", "cpu_barrier2_ms", "total_ms",
        ],
    )

    print("\n[slow_export_warnings]")
    if slow_exports:
        for row in slow_exports[:10]:
            print(
                f"line={row['lineno']} layer={row['layer']} mode={row['mode']} "
                f"total_ms={fmt_ms(row['total_ms'])} "
                f"experts={row['experts']} local_slots={row['local_slots']}"
            )
    else:
        print("not found")

    print("\n[worst]")
    for label, rows, key in (
        ("phase_mode1_drain", phase_rows, "mode1_pre_import_drain_ms"),
        ("phase_preload", phase_rows, "preload_import_ms"),
        ("stream_send_pre_sync", stream_rows, "send_pre_sync_ms"),
        ("stream_send_export", stream_rows, "send_export_ms"),
        ("stream_total", stream_rows, "total_ms"),
        ("phase_rebuild", phase_rows, "rebuild_ms"),
    ):
        value, row = max_value(rows, key)
        rank = row["rank"] if row else "N/A"
        print(f"{label}: rank={rank} {key}={fmt_ms(value)}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize the final mode=1 shrink stage from a log.")
    parser.add_argument("log_path", nargs="?", type=Path,
                        help="Log path. Defaults to newest local *_elastic.txt.")
    args = parser.parse_args()

    path = args.log_path or latest_log()
    data = load_log(path)
    print_summary(path, data)


if __name__ == "__main__":
    main()
