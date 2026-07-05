#!/usr/bin/env python3
"""Summarize per-step rollout slowdown signals in mode1 shrink-aware logs."""
from __future__ import annotations

import argparse
import ast
import re
import statistics
from pathlib import Path
from typing import Iterable


def _floats(pattern: str, text: str) -> list[float]:
    return [float(x) for x in re.findall(pattern, text)]


def _safe_mean(values: Iterable[float]) -> float | None:
    values = list(values)
    return statistics.mean(values) if values else None


def _fmt(value: float | None, digits: int = 3) -> str:
    if value is None:
        return "-"
    return f"{value:.{digits}f}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("log", type=Path)
    args = parser.parse_args()
    text = args.log.read_text(errors="ignore")

    rollouts = []
    for idx, m in enumerate(re.finditer(r"rollout_output_time_s: ([0-9.]+)", text), start=1):
        pos = m.start()
        window_before = text[max(0, pos - 60000):pos]
        window_after = text[pos:pos + 8000]
        rollout_s = float(m.group(1))
        speed_m = re.search(r"rollouts speed tokens/s: ([0-9.]+)", window_after)
        speed = float(speed_m.group(1)) if speed_m else None
        nums = None
        global_matches = list(re.finditer(r"global_token_num: (\[[^\n]+\])", window_before))
        if global_matches:
            try:
                nums = ast.literal_eval(global_matches[-1].group(1))
            except Exception:
                nums = None
        rollouts.append({
            "idx": idx,
            "rollout_s": rollout_s,
            "tokens_per_s": speed,
            "token_sum": sum(nums) if nums else None,
            "token_mean": statistics.mean(nums) if nums else None,
            "token_max": max(nums) if nums else None,
        })

    metric_rows = {}
    metric_keys = [
        "response_length/mean",
        "response_length/max",
        "response_length/clip_ratio",
        "timing_s/gen",
        "timing_s/update_actor",
        "timing_s/old_log_prob",
        "timing_s/ref",
        "timing_s/step",
        "perf/total_num_tokens",
        "perf/throughput",
        "timing_per_token_ms/gen",
    ]
    for m in re.finditer(r"step:(\d+) - ([^\n]+)", text):
        step = int(m.group(1))
        line = m.group(2)
        vals = {}
        for key in metric_keys:
            km = re.search(re.escape(key) + r":([0-9.eE+-]+)", line)
            if km:
                vals[key] = float(km.group(1))
        if vals:
            metric_rows[step] = vals

    print(f"log: {args.log}")
    print(f"size_bytes: {args.log.stat().st_size}")
    print(f"alltoallv_count: {text.count('HcclAlltoAllV') + text.count('AllToAllV')}")
    print(f"runtime_error_count: {text.count('RuntimeError')}")
    print(f"traceback_count: {text.count('Traceback')}")
    print(f"skip_actor_update_count: {text.count('[VERL_DIAG_SKIP_ACTOR_UPDATE]')}")
    print(f"mc2_selected_count: {text.count('effective=MoECommType.MC2') + text.count('selected=MoECommType.MC2')}")

    print("\nper_step:")
    header = (
        "step rollout_s tokens tokens/s ms/token resp_mean resp_max "
        "update_actor old_log_prob ref step_s throughput"
    )
    print(header)
    for row in rollouts:
        step = row["idx"]
        vals = metric_rows.get(step, {})
        print(
            step,
            _fmt(row["rollout_s"]),
            row["token_sum"] if row["token_sum"] is not None else "-",
            _fmt(row["tokens_per_s"]),
            _fmt(vals.get("timing_per_token_ms/gen")),
            _fmt(vals.get("response_length/mean")),
            _fmt(vals.get("response_length/max")),
            _fmt(vals.get("timing_s/update_actor")),
            _fmt(vals.get("timing_s/old_log_prob")),
            _fmt(vals.get("timing_s/ref")),
            _fmt(vals.get("timing_s/step")),
            _fmt(vals.get("perf/throughput")),
        )

    print("\nlifecycle_ms:")
    for name, pat in [
        ("shrink_total", r"shrink_total_ms=([0-9.]+)"),
        ("restore_total", r"restore_total_ms=([0-9.]+)"),
        ("rpc", r"rpc_ms=([0-9.]+)"),
        ("rebuild", r"rebuild_ms=([0-9.]+)"),
        ("refresh", r"refresh_ms=([0-9.]+)"),
        ("barrier", r"barrier_ms=([0-9.]+)"),
        ("moe_prepare", r"moe_dispatch_prepare_done[^\n]*elapsed_ms=([0-9.]+)"),
    ]:
        vals = _floats(pat, text)
        if vals:
            print(
                f"{name}: n={len(vals)} mean={statistics.mean(vals):.3f} "
                f"max={max(vals):.3f} p95={sorted(vals)[int(0.95 * (len(vals)-1))]:.3f}"
            )
        else:
            print(f"{name}: n=0")

    end_markers = re.findall(r"\[run\] end_time=.*", text)
    if end_markers:
        print("\nend_marker:", end_markers[-1])
    tail_errors = re.findall(r"(?:RuntimeError|Traceback|ERROR|Error)[^\n]*(?:\n[^\n]*){0,3}", text)
    if tail_errors:
        print("\nlast_error_snippet:")
        print(tail_errors[-1][-2000:])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
