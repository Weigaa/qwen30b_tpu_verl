#!/usr/bin/env python3
"""Summarize elastic redundancy floor experiment logs."""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import sys
from pathlib import Path
from typing import Any

ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
KV_RE = re.compile(r"GPU KV cache size:\s*([0-9,]+)\s*tokens")
CONC_RE = re.compile(
    r"Maximum concurrency for\s*([0-9,]+)\s*tokens per request:\s*([0-9.]+)x")
AVAIL_RE = re.compile(r"Available memory:\s*(\d+), total memory:\s*(\d+)")
ROLLOUT_RE = re.compile(r"rollout_output_time_s:\s*([0-9.]+)")
EXIT_RE = re.compile(r"\[run\] end_time=([^ ]+)\s+exit_code=(\d+)")
START_RE = re.compile(r"\[run\] start_time=([^ ]+)")
KV_USAGE_RE = re.compile(r"GPU KV cache usage:\s*([0-9.]+)%")
PREEMPT_RE = re.compile(
    r"(?:Full|Elastic) redundancy KV preemption: .*?kv_cache_usage=([0-9.]+)")
GENERIC_PREEMPT_RE = re.compile(r"\bPreempting request\b")
RESOLVED_RE = re.compile(
    r"Elastic execution mode resolved: mode=(?P<mode>\d+) "
    r"elastic_moe_mode=(?P<moe_mode>\S+) initial_ep_size=(?P<initial_ep>\d+) "
    r"floor=(?P<floor>\S+) .*? num_experts=(?P<num_experts>\d+) "
    r"init_redundancy_expert=(?P<redundancy>\d+)")
META_RE = re.compile(r"\[full redundancy experiment\](?P<body>.*)")
HBM_RE = re.compile(
    r"(?:Full|Elastic) redundancy HBM profile: rank=(?P<rank>-?\d+) "
    r"mode=(?P<mode>\d+) floor=(?P<floor>\S+) "
    r"total_npu_memory=(?P<total>\d+) init_free=(?P<init_free>\d+) "
    r"post_profile_free=(?P<post_free>\d+) peak_memory=(?P<peak>\d+) "
    r"torch_current=(?P<torch_current>\d+) "
    r"non_torch_allocations=(?P<non_torch>\d+) "
    r"available_kv_cache_memory=(?P<available>\d+) "
    r"gpu_memory_utilization=(?P<util>[0-9.]+)")
SLOT_RE = re.compile(
    r"(?:Full|Elastic) redundancy MoE slots: layer=(?P<layer>-?\d+) "
    r"ep_rank=(?P<rank>-?\d+) ep_size=(?P<ep_size>\d+) "
    r"mode=(?P<mode>\d+) floor=(?P<floor>\S+) "
    r"num_experts=(?P<num_experts>\d+) active_local=(?P<active>\d+) "
    r"loaded_local=(?P<loaded>\d+) loaded_capacity=(?P<capacity>\d+) "
    r"redundant_experts=(?P<redundant>\d+) .*?"
    r"expert_weight_bytes=(?P<expert_bytes>\d+) "
    r"total_weight_bytes=(?P<total_bytes>\d+)")
SLOT_STD_RE = re.compile(
    r"(?:Full|Elastic) redundancy MoE slots: layer=(?P<layer>\S+) "
    r"ep_rank=(?P<rank>-?\d+) ep_size=(?P<ep_size>\d+) "
    r"mode=(?P<mode>\d+) floor=(?P<floor>\S+) "
    r"global_experts=(?P<num_experts>\d+) "
    r"local_active=(?P<active>\d+) "
    r"loaded_capacity=(?P<capacity>\d+) "
    r"redundant_global=(?P<redundant>\d+) .*?"
    r"expert_weight_bytes=(?P<expert_bytes>\d+) "
    r"total_weight_bytes=(?P<total_bytes>\d+)")
SHRINK_RE = re.compile(
    r"Elastic parallel shrink rpc done: .*?active_ranks=\[([^\]]*)\].*?"
    r"total_ms=([0-9.]+)")
OOM_PATTERNS = (
    "out of memory",
    "oom",
    "err00006",
    "memory error",
    "no available memory for the cache blocks",
)


def strip(line: str) -> str:
    return ANSI_RE.sub("", line)


def parse_keyvals(body: str) -> dict[str, str]:
    return dict(re.findall(r"([A-Za-z0-9_]+)=([^ ]+)", body))


def mean(values: list[float | int]) -> float | None:
    return statistics.mean(values) if values else None


def min_or_none(values: list[float | int]) -> float | int | None:
    return min(values) if values else None


def max_or_none(values: list[float | int]) -> float | int | None:
    return max(values) if values else None


def gib(value: float | int | None) -> float | None:
    if value is None:
        return None
    return float(value) / 1024**3


def parse_rank_count(text: str) -> int:
    text = text.strip()
    if not text:
        return 0
    return len([x for x in text.split(",") if x.strip()])


def summarize(path: Path) -> dict[str, Any]:
    meta: dict[str, str] = {}
    resolved: dict[str, Any] = {}
    hbm_rows: list[dict[str, Any]] = []
    slot_rows: list[dict[str, Any]] = []
    kv_tokens: list[int] = []
    conc: list[float] = []
    model_lens: list[int] = []
    avail_bytes: list[int] = []
    total_bytes: list[int] = []
    shrink_events: list[dict[str, Any]] = []
    rollout_s = None
    start_time = None
    end_time = None
    exit_code = None
    kv_usage_max = None
    preemptions = 0
    oom_lines: list[str] = []

    with path.open("r", errors="ignore") as f:
        for raw in f:
            line = strip(raw).strip()
            low = line.lower()
            if m := META_RE.search(line):
                meta.update(parse_keyvals(m.group("body")))
            if m := RESOLVED_RE.search(line):
                resolved.update(m.groupdict())
            if m := HBM_RE.search(line):
                row = m.groupdict()
                hbm_rows.append({
                    k: (float(v) if k == "util" else int(v)
                        if v.isdigit() else v)
                    for k, v in row.items()
                })
            if m := SLOT_RE.search(line):
                row = m.groupdict()
                row.setdefault("loaded", row.get("capacity"))
                slot_rows.append({
                    k: int(v) if str(v).lstrip("-").isdigit() else v
                    for k, v in row.items()
                })
            elif m := SLOT_STD_RE.search(line):
                row = m.groupdict()
                row["loaded"] = row["capacity"]
                slot_rows.append({
                    k: int(v) if str(v).lstrip("-").isdigit() else v
                    for k, v in row.items()
                })
            if m := KV_RE.search(line):
                kv_tokens.append(int(m.group(1).replace(",", "")))
            if m := CONC_RE.search(line):
                model_lens.append(int(m.group(1).replace(",", "")))
                conc.append(float(m.group(2)))
            if m := AVAIL_RE.search(line):
                avail_bytes.append(int(m.group(1)))
                total_bytes.append(int(m.group(2)))
            if m := ROLLOUT_RE.search(line):
                rollout_s = float(m.group(1))
            if m := START_RE.search(line):
                start_time = m.group(1)
            if m := EXIT_RE.search(line):
                end_time = m.group(1)
                exit_code = int(m.group(2))
            if m := KV_USAGE_RE.search(line):
                usage = float(m.group(1))
                kv_usage_max = usage if kv_usage_max is None else max(
                    kv_usage_max, usage)
            if m := PREEMPT_RE.search(line):
                preemptions += 1
                usage = float(m.group(1)) * 100.0
                kv_usage_max = usage if kv_usage_max is None else max(
                    kv_usage_max, usage)
            elif GENERIC_PREEMPT_RE.search(line):
                preemptions += 1
            if m := SHRINK_RE.search(line):
                shrink_events.append({
                    "active_count": parse_rank_count(m.group(1)),
                    "total_ms": float(m.group(2)),
                })
            if any(pat in low for pat in OOM_PATTERNS):
                if len(oom_lines) < 5:
                    oom_lines.append(line)

    hbm_available = [row["available"] for row in hbm_rows]
    hbm_peak = [row["peak"] for row in hbm_rows]
    hbm_total = [row["total"] for row in hbm_rows]
    hbm_post_free = [row["post_free"] for row in hbm_rows]
    capacities = [row["capacity"] for row in slot_rows]
    active = [row["active"] for row in slot_rows]
    loaded = [row["loaded"] for row in slot_rows]
    redundant = [row["redundant"] for row in slot_rows]
    expert_bytes = [row["expert_bytes"] for row in slot_rows]
    total_weight = [row["total_bytes"] for row in slot_rows]
    layer_count = len({row["layer"] for row in slot_rows})
    rank_count = len({row["rank"] for row in slot_rows})

    floor = (meta.get("floor") or resolved.get("floor")
             or (str(slot_rows[0]["floor"]) if slot_rows else None))
    mode = meta.get("mode") or resolved.get("mode")
    status = "success" if exit_code == 0 else "failed"
    if oom_lines:
        status = "oom" if exit_code not in (0, None) else "oom-recovered"
    elif preemptions:
        status = "preempted" if exit_code == 0 else "failed-preempted"

    return {
        "path": str(path),
        "file": path.name,
        "status": status,
        "mode": mode,
        "floor": floor,
        "initial_ep_size": resolved.get("initial_ep"),
        "num_experts": resolved.get("num_experts") or
        (slot_rows[0]["num_experts"] if slot_rows else None),
        "init_redundancy_expert": resolved.get("redundancy") or
        max_or_none(redundant),
        "start_time": start_time,
        "end_time": end_time,
        "exit_code": exit_code,
        "rollout_s": rollout_s,
        "kv_tokens_min": min_or_none(kv_tokens),
        "kv_tokens_max": max_or_none(kv_tokens),
        "max_model_len": max_or_none(model_lens),
        "max_concurrency_min": min_or_none(conc),
        "max_concurrency_max": max_or_none(conc),
        "available_kv_gib_min": gib(min_or_none(hbm_available or avail_bytes)),
        "available_kv_gib_mean": gib(mean(hbm_available or avail_bytes)),
        "total_hbm_gib_min": gib(min_or_none(hbm_total or total_bytes)),
        "post_profile_free_gib_min": gib(min_or_none(hbm_post_free)),
        "peak_memory_gib_max": gib(max_or_none(hbm_peak)),
        "slot_layer_count": layer_count or None,
        "slot_rank_count": rank_count or None,
        "active_local_min": min_or_none(active),
        "loaded_local_max": max_or_none(loaded),
        "loaded_capacity_max": max_or_none(capacities),
        "expert_weight_gib": gib(max_or_none(expert_bytes)),
        "total_weight_gib_max": gib(max_or_none(total_weight)),
        "preemptions": preemptions,
        "kv_usage_max_pct": kv_usage_max,
        "shrink_events": shrink_events,
        "oom": bool(oom_lines),
        "oom_examples": oom_lines,
    }


def fmt(value: Any, digits: int = 2) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def emit_markdown(rows: list[dict[str, Any]]) -> None:
    headers = [
        "file", "floor", "status", "exit", "rollout_s", "KV tokens",
        "max conc", "avail KV GiB", "peak GiB", "slot cap", "expert GiB",
        "preempt", "max KV %", "OOM"
    ]
    print("| " + " | ".join(headers) + " |")
    print("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        print("| " + " | ".join([
            row["file"],
            fmt(row["floor"], 0),
            row["status"],
            fmt(row["exit_code"], 0),
            fmt(row["rollout_s"], 1),
            fmt(row["kv_tokens_min"], 0),
            fmt(row["max_concurrency_min"], 2),
            fmt(row["available_kv_gib_min"], 2),
            fmt(row["peak_memory_gib_max"], 2),
            fmt(row["loaded_capacity_max"], 0),
            fmt(row["expert_weight_gib"], 3),
            fmt(row["preemptions"], 0),
            fmt(row["kv_usage_max_pct"], 1),
            "yes" if row["oom"] else "no",
        ]) + " |")

    print("\nNotes:")
    for row in rows:
        if row["oom_examples"]:
            print(f"- {row['file']}: first OOM/error line: {row['oom_examples'][0]}")


def emit_csv(rows: list[dict[str, Any]]) -> None:
    keys = [
        "file", "path", "status", "mode", "floor", "initial_ep_size",
        "num_experts", "init_redundancy_expert", "exit_code", "rollout_s",
        "kv_tokens_min", "kv_tokens_max", "max_model_len",
        "max_concurrency_min", "available_kv_gib_min",
        "available_kv_gib_mean", "total_hbm_gib_min",
        "post_profile_free_gib_min", "peak_memory_gib_max",
        "slot_layer_count", "slot_rank_count", "active_local_min",
        "loaded_local_max", "loaded_capacity_max", "expert_weight_gib",
        "total_weight_gib_max", "preemptions", "kv_usage_max_pct", "oom",
    ]
    writer = csv.DictWriter(sys.stdout, fieldnames=keys)
    writer.writeheader()
    for row in rows:
        writer.writerow({key: row.get(key) for key in keys})


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("logs", nargs="+", type=Path)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--markdown", action="store_true")
    group.add_argument("--csv", action="store_true")
    group.add_argument("--json", action="store_true")
    args = parser.parse_args()

    rows = [summarize(path) for path in args.logs]
    if args.csv:
        emit_csv(rows)
    elif args.json:
        print(json.dumps(rows, indent=2, ensure_ascii=False))
    else:
        emit_markdown(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
