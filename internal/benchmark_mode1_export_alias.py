#!/usr/bin/env python3
"""Microbenchmark the mode1 export/alias hot path on a single NPU.

This script is intentionally independent from the full training flow. It does
not instantiate the whole model; it benchmarks the suspected hotspot directly:
  1. expert slice view creation
  2. zero-offset alias construction for P2P
  3. multi-expert index_select export

Use it to compare local vs remote machine behavior when the device is idle.
"""

from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path

import torch

from vllm_ascend.ops.fused_moe import _npu_zero_offset_alias_for_p2p


def parse_shape(text: str) -> tuple[int, ...]:
    parts = [int(part.strip()) for part in text.split(",") if part.strip()]
    if not parts:
        raise ValueError(f"invalid shape: {text}")
    return tuple(parts)


def pct(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    if len(values) == 1:
        return values[0]
    pos = (len(values) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(values) - 1)
    frac = pos - lo
    return values[lo] * (1.0 - frac) + values[hi] * frac


def summarize_ms(values: list[float]) -> str:
    if not values:
        return "n/a"
    return (
        f"mean={statistics.mean(values):.3f} "
        f"p50={pct(values, 0.5):.3f} "
        f"p90={pct(values, 0.9):.3f} "
        f"max={max(values):.3f}"
    )


def benchmark_single_slot(base: torch.Tensor,
                          slot: int,
                          warmup: int,
                          iters: int) -> dict[str, list[float]]:
    slice_ms: list[float] = []
    alias_ms: list[float] = []
    total_ms: list[float] = []

    for idx in range(warmup + iters):
        if torch.npu.is_available():
            torch.npu.synchronize()
        start_t = time.perf_counter()
        slice_start_t = time.perf_counter()
        view = base[slot:slot + 1]
        slice_elapsed_ms = (time.perf_counter() - slice_start_t) * 1000.0
        alias_start_t = time.perf_counter()
        alias = _npu_zero_offset_alias_for_p2p(base, view)
        alias_elapsed_ms = (time.perf_counter() - alias_start_t) * 1000.0
        if torch.npu.is_available():
            torch.npu.synchronize()
        total_elapsed_ms = (time.perf_counter() - start_t) * 1000.0
        if idx >= warmup:
            slice_ms.append(slice_elapsed_ms)
            alias_ms.append(alias_elapsed_ms)
            total_ms.append(total_elapsed_ms)
        if alias.data_ptr() != view.data_ptr():
            raise RuntimeError(
                f"alias data_ptr mismatch: alias={alias.data_ptr()} view={view.data_ptr()}"
            )

    return {
        "slice_ms": slice_ms,
        "alias_ms": alias_ms,
        "total_ms": total_ms,
    }


def benchmark_index_select(base: torch.Tensor,
                           slots: list[int],
                           warmup: int,
                           iters: int) -> list[float]:
    values: list[float] = []
    index = torch.tensor(slots, device=base.device, dtype=torch.long)
    for idx in range(warmup + iters):
        if torch.npu.is_available():
            torch.npu.synchronize()
        start_t = time.perf_counter()
        out = base.index_select(0, index)
        if torch.npu.is_available():
            torch.npu.synchronize()
        elapsed_ms = (time.perf_counter() - start_t) * 1000.0
        if idx >= warmup:
            values.append(elapsed_ms)
        if out.shape[0] != len(slots):
            raise RuntimeError(f"unexpected index_select result shape={tuple(out.shape)}")
    return values


def maybe_format_like_npu(base: torch.Tensor,
                          target_format: int | None = None) -> torch.Tensor:
    if base.device.type != "npu":
        return base
    try:
        import torch_npu  # type: ignore
        fmt = target_format
        if fmt is None:
            fmt = torch_npu.get_npu_format(base)
        return torch_npu.npu_format_cast(base, fmt)
    except Exception:
        return base


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark mode1 export alias hot path on one device.")
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--dtype", default="bfloat16",
                        choices=("float16", "bfloat16", "float32"))
    parser.add_argument("--w13-shape", default="32,1536,2048")
    parser.add_argument("--w2-shape", default="32,2048,768")
    parser.add_argument("--slot", type=int, default=31)
    parser.add_argument("--index-slots", default="0,8,16,24,31")
    parser.add_argument("--npu-format", type=int, default=None,
                        help="Force base tensors to this torch_npu format id, for example 29.")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    args = parser.parse_args()

    dtype = getattr(torch, args.dtype)
    device = torch.device(args.device)
    w13_shape = parse_shape(args.w13_shape)
    w2_shape = parse_shape(args.w2_shape)
    index_slots = [int(part.strip()) for part in args.index_slots.split(",")
                   if part.strip()]

    if device.type == "npu" and not torch.npu.is_available():
        raise RuntimeError("NPU is not available on this host.")

    print(f"device={device}")
    print(f"dtype={dtype}")
    print(f"w13_shape={w13_shape}")
    print(f"w2_shape={w2_shape}")
    print(f"slot={args.slot}")
    print(f"index_slots={index_slots}")
    print(f"npu_format={args.npu_format}")
    print(f"warmup={args.warmup} iters={args.iters}")

    w13 = maybe_format_like_npu(
        torch.empty(w13_shape, device=device, dtype=dtype),
        args.npu_format,
    )
    w2 = maybe_format_like_npu(
        torch.empty(w2_shape, device=device, dtype=dtype),
        args.npu_format,
    )

    print(
        f"w13_storage_offset={int(w13.storage_offset())} "
        f"w2_storage_offset={int(w2.storage_offset())}"
    )
    if device.type == "npu":
        try:
            import torch_npu  # type: ignore
            print(
                f"w13_actual_format={torch_npu.get_npu_format(w13)} "
                f"w2_actual_format={torch_npu.get_npu_format(w2)}"
            )
        except Exception as exc:
            print(f"format_query_failed={exc}")

    w13_single = benchmark_single_slot(w13, args.slot, args.warmup, args.iters)
    w2_single = benchmark_single_slot(w2, args.slot, args.warmup, args.iters)
    w13_index = benchmark_index_select(w13, index_slots, args.warmup, args.iters)
    w2_index = benchmark_index_select(w2, index_slots, args.warmup, args.iters)

    print("\n[w13 single-slot]")
    print(f"slice_ms: {summarize_ms(w13_single['slice_ms'])}")
    print(f"alias_ms: {summarize_ms(w13_single['alias_ms'])}")
    print(f"total_ms: {summarize_ms(w13_single['total_ms'])}")

    print("\n[w2 single-slot]")
    print(f"slice_ms: {summarize_ms(w2_single['slice_ms'])}")
    print(f"alias_ms: {summarize_ms(w2_single['alias_ms'])}")
    print(f"total_ms: {summarize_ms(w2_single['total_ms'])}")

    print("\n[w13 index_select]")
    print(f"index_select_ms: {summarize_ms(w13_index)}")

    print("\n[w2 index_select]")
    print(f"index_select_ms: {summarize_ms(w2_index)}")


if __name__ == "__main__":
    main()
