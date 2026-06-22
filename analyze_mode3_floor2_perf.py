#!/usr/bin/env python3
"""Check mode=3 floor=2 rollout logs for the current stability target.

Hard failures are limited to the bugs we are trying to prevent:
  * wrong mode/floor/headroom
  * post-shrink DP all_reduce warmup unexpectedly executing on mode=3
  * OOM / HCCL allocation / transport timeout markers
  * incomplete rollout
  * 300s-class shrink regression

Rollout time can still be used as an optional hard threshold, but by default it
is reported as a warning only. Mode=3 CPU-source import/refresh is expected to
be slower than mode=4/5 until that path is optimized separately.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


OOM_PATTERNS = (
    "Memory_Allocation_Failure",
    "Failed to allocate resource[DeviceMemory]",
    "RuntimeError: NPU out of memory",
    "current working operator name is HcclAllreduce",
)
HCCL_TIMEOUT_PATTERNS = (
    "Transport_Init_Error",
    "EI0009",
    "300s",
    "350s",
)


def _max_float(pattern: str, text: str) -> float | None:
    values = [float(x) for x in re.findall(pattern, text)]
    return max(values) if values else None


def _first_float(pattern: str, text: str) -> float | None:
    match = re.search(pattern, text)
    return float(match.group(1)) if match else None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("log", type=Path)
    parser.add_argument("--max-rollout-s", type=float, default=None,
                        help="optional hard fail threshold for rollout time")
    parser.add_argument("--warn-rollout-s", type=float, default=240.0,
                        help="warning threshold for rollout time; use <=0 to disable")
    parser.add_argument("--max-shrink-ms", type=float, default=60000.0)
    parser.add_argument("--expected-kv-headroom", type=int, default=2147483648)
    parser.add_argument("--expected-kv-tokens", type=int, default=397824)
    args = parser.parse_args()

    text = args.log.read_text(errors="replace")
    failures: list[str] = []
    warnings: list[str] = []

    if "mode=3" not in text:
        failures.append("missing mode=3 marker")
    if "floor=2" not in text:
        failures.append("missing floor=2 marker")

    headroom_match = re.search(r"kv_cache_init_headroom=(\d+)", text)
    if not headroom_match:
        failures.append("missing kv_cache_init_headroom marker")
    elif int(headroom_match.group(1)) != args.expected_kv_headroom:
        failures.append(
            "unexpected kv_cache_init_headroom="
            f"{headroom_match.group(1)} expected={args.expected_kv_headroom}")

    kv_tokens = _first_float(r"GPU KV cache size:\s*([0-9,]+) tokens".replace(",", "[,]?"), text)
    if kv_tokens is None:
        # Simpler fallback for comma-formatted token counts.
        match = re.search(r"GPU KV cache size:\s*([0-9,]+) tokens", text)
        if match:
            kv_tokens = float(match.group(1).replace(",", ""))
    if kv_tokens is None:
        failures.append("missing GPU KV cache size marker")
    elif int(kv_tokens) != args.expected_kv_tokens:
        failures.append(
            f"unexpected GPU KV cache size={int(kv_tokens)} expected={args.expected_kv_tokens}")

    if "reason=mode3_default_disabled" not in text:
        failures.append(
            "missing mode3 post-shrink DP warmup skip marker "
            "(reason=mode3_default_disabled)")
    if "before_post_shrink_dp_all_reduce_warmup" in text:
        failures.append("post-shrink DP all_reduce warmup still executed")

    for pattern in OOM_PATTERNS:
        if pattern in text:
            failures.append(f"OOM/HCCL allocation marker present: {pattern}")
    for pattern in HCCL_TIMEOUT_PATTERNS:
        if pattern in text:
            failures.append(f"HCCL timeout/transport marker present: {pattern}")

    if "rollout_output_time_s:" not in text:
        failures.append("rollout_output_time_s missing; rollout likely incomplete")

    rollout_s = _max_float(r"rollout_output_time_s:\s*([0-9.]+)", text)
    speed = _max_float(r"rollouts speed tokens/s:\s*([0-9.]+)", text)
    total_tokens = _max_float(r"perf/total_num_tokens:([0-9.]+)", text)
    response_mean = _max_float(r"response_length/mean:([0-9.]+)", text)
    max_shrink_ms = _max_float(r"Elastic parallel shrink done:.*?total_ms=([0-9.]+)", text)
    max_refresh_ms = _max_float(r"Elastic parallel shrink done:.*?refresh_ms=([0-9.]+)", text)
    max_warmup_ms = _max_float(r"Elastic parallel shrink done:.*?warmup_ms=([0-9.]+)", text)

    if rollout_s is not None and args.max_rollout_s is not None and rollout_s > args.max_rollout_s:
        failures.append(
            f"rollout_output_time_s too high: {rollout_s:.3f} > {args.max_rollout_s:.3f}")
    if rollout_s is not None and args.warn_rollout_s > 0 and rollout_s > args.warn_rollout_s:
        warnings.append(
            f"rollout_output_time_s is above warning threshold: "
            f"{rollout_s:.3f} > {args.warn_rollout_s:.3f}; "
            "this is CPU-source refresh/import overhead, not a hard stability failure")
    if max_shrink_ms is not None and max_shrink_ms > args.max_shrink_ms:
        failures.append(
            f"max shrink total_ms too high: {max_shrink_ms:.2f} > {args.max_shrink_ms:.2f}")

    print(f"log={args.log}")
    print(f"kv_cache_tokens={int(kv_tokens) if kv_tokens is not None else None}")
    print(f"rollout_output_time_s={rollout_s}")
    print(f"rollouts_speed_tokens_s={speed}")
    print(f"perf_total_num_tokens={total_tokens}")
    print(f"response_length_mean={response_mean}")
    print(f"max_shrink_total_ms={max_shrink_ms}")
    print(f"max_refresh_ms={max_refresh_ms}")
    print(f"max_warmup_ms={max_warmup_ms}")
    if failures:
        print("STATUS=FAIL")
        for failure in failures:
            print(f"- {failure}")
        return 1

    print("STATUS=PASS")
    for warning in warnings:
        print(f"WARNING: {warning}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
