#!/usr/bin/env python3
"""Parse mode=1 vs mode=4 high-pressure compare runs."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any


def read_env(path: Path) -> dict[str, str]:
    data: dict[str, str] = {}
    if not path.exists():
        return data
    for line in path.read_text(errors="ignore").splitlines():
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        data[key.strip()] = value.strip()
    return data


def parse_header(text: str) -> dict[str, str]:
    header: dict[str, str] = {}
    match = re.search(r"\[full redundancy experiment\] (?P<body>.*)", text)
    if not match:
        return header
    for key, value in re.findall(r"([A-Za-z0-9_]+)=([^\s]*)", match.group("body")):
        header[key] = value
    return header


def last_float(pattern: str, text: str) -> float | None:
    matches = re.findall(pattern, text)
    if not matches:
        return None
    try:
        return float(matches[-1])
    except ValueError:
        return None


def last_int(pattern: str, text: str) -> int | None:
    matches = re.findall(pattern, text)
    if not matches:
        return None
    try:
        return int(matches[-1].replace(",", ""))
    except ValueError:
        return None


def parse_step_metrics(text: str) -> dict[str, float]:
    step_lines = [line for line in text.splitlines() if "step:1 -" in line]
    if not step_lines:
        return {}
    line = step_lines[-1]
    metrics: dict[str, float] = {}
    for key, value in re.findall(r"([A-Za-z0-9_./-]+):(-?\d+(?:\.\d+)?(?:e[+-]?\d+)?)", line):
        try:
            metrics[key] = float(value)
        except ValueError:
            pass
    return metrics


def parse_preempt_count(text: str) -> int:
    direct_count = len(re.findall(r"\bPreempting request\b", text, flags=re.IGNORECASE))
    metric_counts: list[int] = []
    for pattern in (
        r"\bnum_preemptions(?:_total)?\b[:=]\s*(\d+)",
        r"\bnum_preemption_iter\b[:=]\s*(\d+)",
        r"\bnum_preempted_reqs\b[:=]\s*(\d+)",
    ):
        value = last_int(pattern, text)
        if value is not None:
            metric_counts.append(value)
    return max([direct_count, *metric_counts])


def find_internal_log_path(case_dir: Path, launcher_text: str) -> Path | None:
    matches = re.findall(r"\[run\] start_time=.*?\blogfile=([^\s'\"]+)", launcher_text)
    for value in reversed(matches):
        path = Path(value)
        if path.exists():
            return path

    candidates = sorted(case_dir.glob("wjeagerqwen30b-a3b-with_draft_breakdown_*_elastic.txt"))
    if candidates:
        return candidates[-1]
    return None


def parse_record_lengths(case_dir: Path, high_cap: int, target_rank: int) -> dict[str, Any]:
    path = case_dir / "record" / "1.jsonl"
    if not path.exists():
        return {}

    pad_token_ids = {0, 151643}
    prompt_by_rank: dict[int, list[int]] = {}
    by_rank: dict[int, list[int]] = {}
    total_by_rank: dict[int, list[int]] = {}
    first_pad_by_rank: dict[int, list[int]] = {}
    total_records = 0

    with path.open(errors="ignore") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            rank = row.get("rollout_rank")
            responses = row.get("responses")
            if rank is None or not isinstance(responses, list):
                continue
            try:
                rank_int = int(rank)
            except (TypeError, ValueError):
                continue

            total_records += 1
            prompts = row.get("prompts") or []
            prompt_len = (
                sum(1 for token in prompts if token not in pad_token_ids)
                if isinstance(prompts, list) else 0
            )
            nonpad_len = sum(1 for token in responses if token not in pad_token_ids)
            try:
                first_pad = responses.index(151643)
            except ValueError:
                first_pad = len(responses)
            prompt_by_rank.setdefault(rank_int, []).append(prompt_len)
            by_rank.setdefault(rank_int, []).append(nonpad_len)
            total_by_rank.setdefault(rank_int, []).append(prompt_len + nonpad_len)
            first_pad_by_rank.setdefault(rank_int, []).append(first_pad)

    if not by_rank:
        return {"record_path": str(path), "record_total": total_records}

    rank_max = {rank: max(lengths) for rank, lengths in by_rank.items()}
    rank_mean = {rank: sum(lengths) / len(lengths) for rank, lengths in by_rank.items()}
    first_pad_max = {
        rank: max(lengths) for rank, lengths in first_pad_by_rank.items()
    }
    cap_threshold = max(1, high_cap - 1)
    ranks_reaching_cap = sum(1 for value in rank_max.values()
                             if high_cap > 0 and value >= cap_threshold)
    target_lengths = by_rank.get(target_rank, [])
    target_prompt_lengths = prompt_by_rank.get(target_rank, [])
    target_total_lengths = total_by_rank.get(target_rank, [])
    target_first_pad = first_pad_by_rank.get(target_rank, [])
    target_max = max(target_lengths) if target_lengths else None
    target_mean = sum(target_lengths) / len(target_lengths) if target_lengths else None
    target_prompt_sum = sum(target_prompt_lengths) if target_prompt_lengths else None
    target_prompt_mean = (
        sum(target_prompt_lengths) / len(target_prompt_lengths)
        if target_prompt_lengths else None
    )
    target_total_sum = sum(target_total_lengths) if target_total_lengths else None
    target_total_mean = (
        sum(target_total_lengths) / len(target_total_lengths)
        if target_total_lengths else None
    )
    target_total_max = max(target_total_lengths) if target_total_lengths else None
    target_first_pad_max = max(target_first_pad) if target_first_pad else None

    return {
        "record_path": str(path),
        "record_total": total_records,
        "record_rank_count": len(by_rank),
        "record_ranks_reaching_cap": ranks_reaching_cap,
        "record_target_rank_count": len(target_lengths),
        "record_target_rank_prompt_sum": target_prompt_sum,
        "record_target_rank_prompt_mean": target_prompt_mean,
        "record_target_rank_nonpad_max": target_max,
        "record_target_rank_nonpad_mean": target_mean,
        "record_target_rank_total_sum": target_total_sum,
        "record_target_rank_total_mean": target_total_mean,
        "record_target_rank_total_max": target_total_max,
        "record_target_rank_first_pad_max": target_first_pad_max,
        "record_target_rank_cap_hit": bool(
            high_cap > 0 and target_max is not None and target_max >= cap_threshold),
        "record_target_rank_cap_hit_ratio": (
            target_max / high_cap if high_cap > 0 and target_max is not None else None),
        "record_rank_nonpad_maxes": ",".join(
            f"{rank}:{rank_max[rank]}" for rank in sorted(rank_max)),
        "record_rank_nonpad_means": ",".join(
            f"{rank}:{rank_mean[rank]:.1f}" for rank in sorted(rank_mean)),
        "record_rank_first_pad_maxes": ",".join(
            f"{rank}:{first_pad_max[rank]}" for rank in sorted(first_pad_max)),
    }


def parse_case(case_dir: Path) -> dict[str, Any]:
    env = read_env(case_dir / "case.env")
    launcher_log_path = Path(env.get("launcher_log", case_dir / "launcher.log"))
    launcher_text = (
        launcher_log_path.read_text(errors="ignore")
        if launcher_log_path.exists() else ""
    )
    internal_log_path = find_internal_log_path(case_dir, launcher_text)
    internal_text = (
        internal_log_path.read_text(errors="ignore")
        if internal_log_path and internal_log_path.exists() else ""
    )
    text = launcher_text + "\n" + internal_text
    log_path = internal_log_path or launcher_log_path
    header = parse_header(text)
    metrics = parse_step_metrics(text)

    exit_code = last_int(r"\[run\] end_time=.*?exit_code=(\d+)", launcher_text)
    launcher_status = env.get("launcher_status")
    if exit_code is None and launcher_status not in (None, ""):
        try:
            exit_code = int(launcher_status)
        except ValueError:
            exit_code = None

    preempt_count = parse_preempt_count(text)
    oom_count = len(re.findall(
        r"Memory_Allocation_Failure|Failed to allocate|out of memory|OutOfMemory|\bOOM\b",
        text,
        flags=re.IGNORECASE,
    ))
    traceback_count = len(re.findall(r"Traceback \(most recent call last\)", text))
    fatal_error_count = len(re.findall(
        r"RayTaskError|Error executing job with overrides|RuntimeError:|AssertionError:|\bFATAL\b",
        text,
        flags=re.IGNORECASE,
    ))

    high_cap = as_int(env.get("high_response_cap"), 0)
    target_rank = as_int(env.get("target_rank"), -1)

    row: dict[str, Any] = {
        "case_dir": str(case_dir),
        "log": str(log_path),
        "launcher_log": str(launcher_log_path),
        "mode": int(env.get("mode") or header.get("mode") or -1),
        "floor": int(env.get("floor") or header.get("floor") or -1),
        "target_mode4_kv_tokens": env.get("target_mode4_kv_tokens", ""),
        "scale": env.get("scale", ""),
        "prompt_length": env.get("prompt_length", ""),
        "max_response_length": env.get("max_response_length", ""),
        "max_num_seqs": env.get("max_num_seqs", ""),
        "high_response_cap": env.get("high_response_cap", ""),
        "estimated_high_rank_tokens": env.get("estimated_high_rank_tokens", ""),
        "estimated_pressure_ratio_to_mode4_kv": env.get("estimated_pressure_ratio_to_mode4_kv", ""),
        "target_rank": env.get("target_rank", ""),
        "cap_list": env.get("cap_list", ""),
        "exit_code": exit_code,
        "launcher_status": launcher_status or "",
        "kv_cache_tokens": last_int(r"GPU KV cache size:\s*([\d,]+)\s*tokens", text),
        "mode1_reference_kv_tokens": env.get("mode1_reference_kv_tokens", ""),
        "estimated_pressure_ratio_to_mode1_ref_kv": env.get("estimated_pressure_ratio_to_mode1_ref_kv", ""),
        "estimated_mode1_pressure_margin_tokens": env.get("estimated_mode1_pressure_margin_tokens", ""),
        "rollout_output_time_s": last_float(r"rollout_output_time_s:\s*([0-9.]+)", text),
        "preempt_count": preempt_count,
        "oom_count": oom_count,
        "traceback_count": traceback_count,
        "fatal_error_count": fatal_error_count,
        "response_length_mean": metrics.get("response_length/mean"),
        "response_length_max": metrics.get("response_length/max"),
        "response_length_clip_ratio": metrics.get("response_length/clip_ratio"),
        "max_memory_allocated_gb": metrics.get("perf/max_memory_allocated_gb"),
        "max_memory_reserved_gb": metrics.get("perf/max_memory_reserved_gb"),
    }
    row.update(parse_record_lengths(case_dir, high_cap, target_rank))
    record_total = row.get("record_target_rank_total_sum")
    target_mode4_kv = as_int(row.get("target_mode4_kv_tokens"), 0)
    mode1_ref_kv = as_int(row.get("mode1_reference_kv_tokens"), 0)
    if isinstance(record_total, (float, int)) and target_mode4_kv > 0:
        row["record_pressure_ratio_to_mode4_kv"] = record_total / target_mode4_kv
    if isinstance(record_total, (float, int)) and mode1_ref_kv > 0:
        row["record_pressure_ratio_to_mode1_ref_kv"] = record_total / mode1_ref_kv
    row["dry_run"] = env.get("dry_run") == "1"
    row["completed"] = row["exit_code"] == 0 and row["rollout_output_time_s"] is not None
    row["valid_no_preempt"] = row["completed"] and row["preempt_count"] == 0
    return row


def discover_cases(run_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for case_env in sorted(run_dir.glob("floor*_mode*/case.env")):
        rows.append(parse_case(case_env.parent))
    rows.sort(key=lambda r: (r.get("floor", -1), r.get("mode", -1)))
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "floor", "mode", "completed", "valid_no_preempt", "exit_code",
        "dry_run",
        "kv_cache_tokens", "target_mode4_kv_tokens", "scale", "high_response_cap",
        "estimated_high_rank_tokens", "estimated_pressure_ratio_to_mode4_kv",
        "mode1_reference_kv_tokens", "estimated_pressure_ratio_to_mode1_ref_kv",
        "estimated_mode1_pressure_margin_tokens",
        "target_rank", "cap_list",
        "rollout_output_time_s", "preempt_count", "oom_count",
        "fatal_error_count", "traceback_count", "response_length_mean", "response_length_max",
        "response_length_clip_ratio", "max_memory_allocated_gb",
        "max_memory_reserved_gb",
        "record_rank_count", "record_ranks_reaching_cap",
        "record_target_rank_count", "record_target_rank_prompt_sum",
        "record_target_rank_prompt_mean", "record_target_rank_nonpad_mean",
        "record_target_rank_nonpad_max", "record_target_rank_total_sum",
        "record_target_rank_total_mean", "record_target_rank_total_max",
        "record_pressure_ratio_to_mode4_kv", "record_pressure_ratio_to_mode1_ref_kv",
        "record_target_rank_cap_hit",
        "record_target_rank_cap_hit_ratio", "record_target_rank_first_pad_max",
        "record_rank_nonpad_maxes", "record_rank_first_pad_maxes",
        "launcher_log", "log",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def as_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def cap_hit_ratio(row: dict[str, Any]) -> float | None:
    try:
        high_cap = float(row.get("high_response_cap") or 0)
        response_max = float(row.get("response_length_max") or 0)
    except (TypeError, ValueError):
        return None
    if high_cap <= 0:
        return None
    return response_max / high_cap


def preferred_pressure_hit_ratio(row: dict[str, Any]) -> float | None:
    record_hit = row.get("record_target_rank_cap_hit_ratio")
    if isinstance(record_hit, (float, int)):
        return float(record_hit)
    return cap_hit_ratio(row)


def pressure_ok(row: dict[str, Any], threshold: float = 0.90) -> bool:
    hit = preferred_pressure_hit_ratio(row)
    return bool(row.get("completed")) and hit is not None and hit >= threshold


def mode1_should_preempt(row: dict[str, Any]) -> bool:
    try:
        ratio = float(row.get("estimated_pressure_ratio_to_mode1_ref_kv") or 0)
        margin = int(float(row.get("estimated_mode1_pressure_margin_tokens") or 0))
    except (TypeError, ValueError):
        return False
    return ratio > 1.0 and margin > 0


def cap_env_key(floor: int) -> str:
    return f"COMPARE_RESPONSE_CAP_FLOOR{floor}"


def suggested_cap_from_record(row: dict[str, Any]) -> int | None:
    try:
        target_kv = float(row.get("target_mode4_kv_tokens") or 0)
        scale = float(row.get("scale") or 0)
        prompt_sum = float(row.get("record_target_rank_prompt_sum") or 0)
        target_count = int(float(row.get("record_target_rank_count") or 0))
    except (TypeError, ValueError):
        return None
    if target_kv <= 0 or scale <= 0 or target_count <= 0:
        return None
    return max(1, int((target_kv * scale - prompt_sum) / target_count))


def suggested_cap_for_floor(m4: dict[str, Any] | None,
                            m1: dict[str, Any] | None) -> tuple[int | None, str]:
    row = m4 or m1
    if not row:
        return None, "missing floor data"

    current = as_int(row.get("high_response_cap"))
    if current <= 0:
        return None, "missing current cap"

    if not m4:
        return current, "run mode4 first"
    if m4.get("dry_run"):
        return current, "dry-run only"
    if not m4.get("completed") or int(m4.get("preempt_count") or 0) > 0:
        return max(1, int(current * 0.97)), "lower cap: mode4 failed or preempted"

    record_suggested = suggested_cap_from_record(m4)

    hit = preferred_pressure_hit_ratio(m4)
    if hit is not None and hit < 0.90:
        return int(current * 1.03), "raise cap: mode4 pressure too low"

    if m1 and not m1.get("dry_run"):
        if not mode1_should_preempt(m1):
            return (
                record_suggested or int(current * 1.03),
                "raise cap: estimated mode1 pressure is below reference KV",
            )
        if int(m1.get("preempt_count") or 0) == 0:
            return (
                record_suggested or int(current * 1.03),
                "raise cap using actual prompt length: mode1 did not preempt",
            )
        if m1.get("completed"):
            if record_suggested is not None and record_suggested > current:
                return (
                    record_suggested,
                    "valid but can raise cap to restore target mode4 pressure using actual prompt length",
                )
            return current, "valid comparison cap"

    return current, "mode4 cap usable; run mode1"


def write_suggested_overrides(path: Path, rows: list[dict[str, Any]]) -> None:
    by_floor: dict[int, dict[int, dict[str, Any]]] = {}
    for row in rows:
        by_floor.setdefault(int(row["floor"]), {})[int(row["mode"])] = row

    lines = [
        "# Source this file before rerunning the compare harness to apply parser suggestions.",
        "# Example: source suggested_overrides.env && ./run_mode1_mode4_pressure_compare.sh --floors 2,4,8 --modes 4 --force",
    ]
    for floor in sorted(by_floor):
        m4 = by_floor[floor].get(4)
        m1 = by_floor[floor].get(1)
        suggested, reason = suggested_cap_for_floor(m4, m1)
        if suggested is None:
            continue
        lines.append(f"# floor={floor}: {reason}")
        lines.append(f"export {cap_env_key(floor)}={suggested}")
    lines.append("")
    path.write_text("\n".join(lines))


def case_state(row: dict[str, Any]) -> str:
    if row.get("dry_run"):
        return "dry-run"
    if row.get("completed"):
        return "completed"
    if not Path(str(row.get("log", ""))).exists():
        return "missing-log"
    if row.get("exit_code") not in (None, 0):
        return "failed"
    return "incomplete"


def comparison_for_floor(m4: dict[str, Any] | None,
                         m1: dict[str, Any] | None) -> dict[str, Any]:
    t4 = m4.get("rollout_output_time_s") if m4 else None
    t1 = m1.get("rollout_output_time_s") if m1 else None
    ratio = None
    if isinstance(t4, float) and isinstance(t1, float) and t4 > 0:
        ratio = t1 / t4

    if not m4:
        return {"valid": False, "reason": "missing mode4", "t4": t4, "t1": t1, "ratio": ratio, "winner": ""}
    if m4.get("dry_run"):
        return {"valid": False, "reason": "mode4 dry-run only", "t4": t4, "t1": t1, "ratio": ratio, "winner": ""}
    if not m4.get("completed"):
        return {"valid": False, "reason": "mode4 failed/incomplete", "t4": t4, "t1": t1, "ratio": ratio, "winner": ""}
    if int(m4.get("preempt_count") or 0) > 0:
        return {"valid": False, "reason": "mode4 preempted", "t4": t4, "t1": t1, "ratio": ratio, "winner": ""}
    if not pressure_ok(m4):
        return {"valid": False, "reason": "mode4 pressure too low", "t4": t4, "t1": t1, "ratio": ratio, "winner": ""}

    if not m1:
        return {"valid": False, "reason": "missing mode1", "t4": t4, "t1": t1, "ratio": ratio, "winner": ""}
    if m1.get("dry_run"):
        return {"valid": False, "reason": "mode1 dry-run only", "t4": t4, "t1": t1, "ratio": ratio, "winner": ""}
    if int(m1.get("preempt_count") or 0) == 0:
        return {"valid": False, "reason": "mode1 did not preempt", "t4": t4, "t1": t1, "ratio": ratio, "winner": ""}
    if not m1.get("completed"):
        return {"valid": False, "reason": "mode1 preempted but did not complete", "t4": t4, "t1": t1, "ratio": ratio, "winner": ""}
    if not pressure_ok(m1):
        return {"valid": False, "reason": "mode1 pressure too low", "t4": t4, "t1": t1, "ratio": ratio, "winner": ""}
    if ratio is None:
        return {"valid": False, "reason": "missing rollout timing", "t4": t4, "t1": t1, "ratio": ratio, "winner": ""}

    winner = "mode4" if t4 < t1 else "mode1"
    return {"valid": True, "reason": "valid", "t4": t4, "t1": t1, "ratio": ratio, "winner": winner}


def write_md(path: Path, rows: list[dict[str, Any]], run_dir: Path) -> None:
    by_floor: dict[int, dict[int, dict[str, Any]]] = {}
    for row in rows:
        by_floor.setdefault(int(row["floor"]), {})[int(row["mode"])] = row

    lines: list[str] = []
    lines.append("# Mode1 vs Mode4 High-KV Pressure Compare")
    lines.append("")
    lines.append(f"Run dir: `{run_dir}`")
    lines.append("")
    if not rows:
        lines.append("## Status")
        lines.append("")
        lines.append("No compare cases were found. This run directory has no `floor*_mode*/case.env` files, so no real mode1/mode4 comparison has been executed or parsed yet.")
        lines.append("")
        lines.append("Run the harness first, for example:")
        lines.append("")
        lines.append("```bash")
        lines.append("./run_mode1_mode4_pressure_compare.sh --floors 2,4,8 --scale 0.96")
        lines.append("```")
        lines.append("")
        path.write_text("\n".join(lines) + "\n")
        return

    lines.append("## Final Verdict")
    lines.append("")
    lines.append("| floor | valid comparison | reason | mode4 rollout s | mode1 rollout s | mode1/mode4 | winner |")
    lines.append("|---:|---:|---|---:|---:|---:|---|")
    for floor in sorted(by_floor):
        m4 = by_floor[floor].get(4)
        m1 = by_floor[floor].get(1)
        cmp = comparison_for_floor(m4, m1)
        lines.append(
            f"| {floor} | {'yes' if cmp['valid'] else 'no'} | {cmp['reason']} | {fmt(cmp['t4'])} | {fmt(cmp['t1'])} | {fmt(cmp['ratio'])} | {cmp['winner']} |"
        )
    lines.append("")
    lines.append("## Detail")
    lines.append("")
    lines.append("| floor | mode | state | no preempt | pressure ok | target rank | KV tokens | high cap | trainer resp mean/max | record target mean/max | record cap hit | ranks hitting cap | record mode4 pressure | estimated mode4 pressure | mode1 ref KV | record mode1 ref pressure | estimated mode1 ref pressure | rollout s | preempts | OOMs | fatal errors | log |")
    lines.append("|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|")
    for row in rows:
        resp = f"{fmt(row['response_length_mean'])}/{fmt(row['response_length_max'])}"
        record_resp = (
            f"{fmt(row.get('record_target_rank_nonpad_mean'))}/"
            f"{fmt(row.get('record_target_rank_nonpad_max'))}"
        )
        hit = cap_hit_ratio(row)
        record_hit = row.get("record_target_rank_cap_hit_ratio")
        rel_log = Path(row["log"])
        try:
            rel_log = rel_log.relative_to(run_dir)
        except ValueError:
            pass
        lines.append(
            "| {floor} | {mode} | {state} | {valid} | {pressure_ok} | {target_rank} | {kv} | {cap} | {resp} | {record_resp} | {record_hit} | {ranks_hit} | {record_pressure} | {pressure} | {mode1_ref_kv} | {record_mode1_pressure} | {mode1_pressure} | {time} | {preempt} | {oom} | {fatal} | `{log}` |".format(
                floor=row["floor"],
                mode=row["mode"],
                state=case_state(row),
                valid="yes" if row["valid_no_preempt"] else "no",
                pressure_ok="yes" if pressure_ok(row) else "no",
                target_rank=fmt(row["target_rank"]),
                kv=fmt(row["kv_cache_tokens"]),
                cap=fmt(row["high_response_cap"]),
                record_resp=record_resp,
                record_hit=fmt(record_hit if record_hit is not None else hit),
                ranks_hit=fmt(row.get("record_ranks_reaching_cap")),
                record_pressure=fmt(row.get("record_pressure_ratio_to_mode4_kv")),
                pressure=fmt(row["estimated_pressure_ratio_to_mode4_kv"]),
                mode1_ref_kv=fmt(row["mode1_reference_kv_tokens"]),
                record_mode1_pressure=fmt(row.get("record_pressure_ratio_to_mode1_ref_kv")),
                mode1_pressure=fmt(row["estimated_pressure_ratio_to_mode1_ref_kv"]),
                resp=resp,
                time=fmt(row["rollout_output_time_s"]),
                preempt=row["preempt_count"],
                oom=row["oom_count"],
                fatal=row["fatal_error_count"],
                log=rel_log,
            )
        )

    lines.append("")
    lines.append("## Mode4 vs Mode1")
    lines.append("")
    lines.append("| floor | mode4 ready | mode1 completed+preempted | mode4 rollout s | mode1 rollout s | mode1/mode4 | tentative result |")
    lines.append("|---:|---:|---:|---:|---:|---:|---|")
    for floor in sorted(by_floor):
        m4 = by_floor[floor].get(4)
        m1 = by_floor[floor].get(1)
        if not m4 or not m1:
            continue
        cmp = comparison_for_floor(m4, m1)
        mode4_ready = bool(m4.get("valid_no_preempt")) and pressure_ok(m4)
        mode1_completed_preempted = bool(m1.get("completed")) and int(m1.get("preempt_count") or 0) > 0
        if cmp["valid"]:
            result = "mode4 faster" if cmp["winner"] == "mode4" else "mode1 still faster"
        else:
            result = f"invalid: {cmp['reason']}"
        lines.append(
            f"| {floor} | {'yes' if mode4_ready else 'no'} | {'yes' if mode1_completed_preempted else 'no'} | {fmt(cmp['t4'])} | {fmt(cmp['t1'])} | {fmt(cmp['ratio'])} | {result} |"
        )

    lines.append("")
    lines.append("## Calibration Advice")
    lines.append("")
    lines.append("| floor | current high cap | suggested cap | mode4 status | mode1 status | advice |")
    lines.append("|---:|---:|---:|---|---|---|")
    for floor in sorted(by_floor):
        m4 = by_floor[floor].get(4)
        m1 = by_floor[floor].get(1)
        high_cap = (m4 or m1 or {}).get("high_response_cap", "")
        suggested_cap, suggested_reason = suggested_cap_for_floor(m4, m1)

        if not m4:
            mode4_status = "missing"
            advice = "run mode4 first"
        elif m4.get("dry_run"):
            mode4_status = "dry-run"
            advice = "run real mode4 case"
        elif not m4.get("completed"):
            mode4_status = "failed/incomplete"
            advice = "lower high cap or inspect mode4 OOM/fatal error"
        elif int(m4.get("preempt_count") or 0) > 0:
            mode4_status = "preempted"
            advice = "lower high cap; mode4 must be no-preempt"
        else:
            hit = preferred_pressure_hit_ratio(m4)
            if hit is not None and hit < 0.90:
                mode4_status = f"valid but low pressure ({hit:.2f}x cap)"
                advice = "increase high cap; response max is far below cap"
            else:
                mode4_status = "valid"
                advice = "run mode1 under the same cap" if not m1 else "mode4 pressure is usable"

        if not m1:
            mode1_status = "missing"
        elif m1.get("dry_run"):
            mode1_status = "dry-run"
        elif not m1.get("completed") and int(m1.get("preempt_count") or 0) == 0:
            mode1_status = "failed/incomplete without preempt evidence"
        elif int(m1.get("preempt_count") or 0) > 0:
            mode1_status = f"preempted ({m1.get('preempt_count')})"
            if m4 and m4.get("valid_no_preempt") and pressure_ok(m4) and m1.get("completed"):
                advice = "valid comparison point; compare rollout time"
        else:
            mode1_status = "no preempt"
            if m4 and m4.get("valid_no_preempt"):
                advice = "increase high cap until mode1 preempts, while keeping mode4 no-preempt"

        if suggested_cap is not None and suggested_cap != as_int(high_cap):
            advice = f"{advice}; suggested cap reason: {suggested_reason}"
        lines.append(
            f"| {floor} | {fmt(high_cap)} | {fmt(suggested_cap)} | {mode4_status} | {mode1_status} | {advice} |"
        )

    lines.append("")
    lines.append("## Validity Checks")
    lines.append("")
    lines.append("- A floor is a valid comparison only if mode4 completes with `preempt_count=0`.")
    lines.append("- The intended stress condition is confirmed only if mode1 has `preempt_count>0` under the same cap list.")
    lines.append("- `mode1 ref pressure > 1.0` means the generated single-rank demand exceeds the observed mode1 KV capacity for that floor, so mode1 preemption is expected if the response cap is actually hit.")
    lines.append("- `record target mean/max` is computed from `record/1.jsonl` as non-padding response tokens on the target rollout rank, and is the preferred pressure check for these tail-cap experiments.")
    lines.append("- `trainer resp mean/max` still uses trainer metrics. It can stop at the first `151643` pad/eos token and under-report forced-tail non-padding tokens when `ignore_eos=True` keeps generating after that marker.")
    lines.append("- `traceback_count` is diagnostic only; shutdown-time TBE tracebacks can appear in successful runs.")
    lines.append("- Use `fatal_error_count`, `exit_code`, and `rollout_output_time_s` for pass/fail judgment.")
    lines.append("")
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--write", action="store_true", help="write summary.csv and summary.md")
    args = parser.parse_args()

    rows = discover_cases(args.run_dir)
    if args.write:
        write_csv(args.run_dir / "summary.csv", rows)
        write_md(args.run_dir / "summary.md", rows, args.run_dir)
        write_suggested_overrides(args.run_dir / "suggested_overrides.env", rows)
        if not rows:
            print(f"no compare cases found under: {args.run_dir}")
    else:
        for row in rows:
            print(row)


if __name__ == "__main__":
    main()
