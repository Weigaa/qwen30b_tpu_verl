#!/usr/bin/env python3
"""Minimal Ascend HCCL rank-pair P2P + zero-offset alias benchmark.

This targets the mode=1 floor=2 shrink hot path:
  source slot view -> zero-offset alias -> HCCL isend/irecv -> target slot alias.

Run with 16 ranks. Default pairs are 12->14, 13->15, 12->15, 13->14.
"""

from __future__ import annotations

import argparse
import os
import time
from typing import Iterable

import torch
import torch.distributed as dist
import torch_npu  # noqa: F401; registers NPU backend/helpers


def parse_pair_list(text: str) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        left, right = item.replace("->", ":").split(":", 1)
        pairs.append((int(left), int(right)))
    return pairs


def parse_int_list(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def parse_shape(text: str) -> tuple[int, ...]:
    return tuple(int(x.strip()) for x in text.split(",") if x.strip())


def get_dtype(name: str) -> torch.dtype:
    name = name.lower()
    if name in ("bf16", "bfloat16"):
        return torch.bfloat16
    if name in ("fp16", "float16", "half"):
        return torch.float16
    if name in ("fp32", "float32"):
        return torch.float32
    raise ValueError(f"unsupported dtype: {name}")


def npu_zero_offset_alias_for_p2p(base: torch.Tensor,
                                  slot_view: torch.Tensor) -> torch.Tensor:
    """Mirror vllm_ascend.ops.fused_moe._npu_zero_offset_alias_for_p2p."""
    if slot_view.device.type != "npu":
        return slot_view
    if int(slot_view.storage_offset()) == 0:
        return slot_view

    try:
        target_format = torch_npu.get_npu_format(slot_view)
        alias = torch.empty(tuple(slot_view.shape),
                            device=slot_view.device,
                            dtype=slot_view.dtype)
        alias = torch_npu.npu_format_cast(alias, target_format)

        ptr_delta = int(slot_view.data_ptr()) - int(base.data_ptr())
        element_size = int(slot_view.element_size())
        storage_index = int(slot_view.storage_offset())
        if ptr_delta >= 0 and element_size > 0 and ptr_delta % element_size == 0:
            storage_index = ptr_delta // element_size
        torch_npu.npu_change_data_ptr(alias, base, int(storage_index))

        if int(alias.storage_offset()) != 0:
            raise RuntimeError(f"alias storage_offset={int(alias.storage_offset())}")
        return alias
    except Exception as exc:
        raise RuntimeError(
            "failed to create zero-offset NPU alias for P2P; "
            f"base_shape={tuple(base.shape)} slot_shape={tuple(slot_view.shape)} "
            f"slot_offset={int(slot_view.storage_offset())}") from exc


def npu_sync() -> None:
    torch.npu.synchronize()


def tensor_bytes(shape: Iterable[int], dtype: torch.dtype) -> int:
    numel = 1
    for dim in shape:
        numel *= int(dim)
    return numel * torch.tensor([], dtype=dtype).element_size()


def run_pair(src: int,
             dst: int,
             rank: int,
             group,
             base_w13: torch.Tensor | None,
             base_w2: torch.Tensor | None,
             layers: int,
             experts: int,
             capacity: int,
             sync_before_alias: bool,
             sync_after_pair: bool) -> tuple[float, float, float, float, float, float, float, float]:
    send_alias_ms = 0.0
    send_post_ms = 0.0
    send_wait_ms = 0.0
    recv_alias_ms = 0.0
    recv_post_ms = 0.0
    recv_wait_ms = 0.0

    is_sender = rank == src
    is_receiver = rank == dst
    pair_start = time.perf_counter()

    if is_sender:
        assert base_w13 is not None and base_w2 is not None
        for _layer in range(layers):
            for expert_idx in range(experts):
                slot = expert_idx % capacity
                if sync_before_alias:
                    npu_sync()
                t0 = time.perf_counter()
                send_w13 = npu_zero_offset_alias_for_p2p(
                    base_w13, base_w13[slot:slot + 1])
                send_w2 = npu_zero_offset_alias_for_p2p(
                    base_w2, base_w2[slot:slot + 1])
                send_alias_ms += (time.perf_counter() - t0) * 1000.0

                t0 = time.perf_counter()
                req_w13 = dist.isend(send_w13, dst=dst, group=group)
                req_w2 = dist.isend(send_w2, dst=dst, group=group)
                send_post_ms += (time.perf_counter() - t0) * 1000.0

                t0 = time.perf_counter()
                req_w13.wait()
                req_w2.wait()
                send_wait_ms += (time.perf_counter() - t0) * 1000.0
                del send_w13, send_w2, req_w13, req_w2

    elif is_receiver:
        assert base_w13 is not None and base_w2 is not None
        for _layer in range(layers):
            for expert_idx in range(experts):
                slot = expert_idx % capacity
                if sync_before_alias:
                    npu_sync()
                t0 = time.perf_counter()
                recv_w13 = npu_zero_offset_alias_for_p2p(
                    base_w13, base_w13[slot:slot + 1])
                recv_w2 = npu_zero_offset_alias_for_p2p(
                    base_w2, base_w2[slot:slot + 1])
                recv_alias_ms += (time.perf_counter() - t0) * 1000.0

                t0 = time.perf_counter()
                req_w13 = dist.irecv(recv_w13, src=src, group=group)
                req_w2 = dist.irecv(recv_w2, src=src, group=group)
                recv_post_ms += (time.perf_counter() - t0) * 1000.0

                t0 = time.perf_counter()
                req_w13.wait()
                req_w2.wait()
                recv_wait_ms += (time.perf_counter() - t0) * 1000.0
                del recv_w13, recv_w2, req_w13, req_w2

    if sync_after_pair and (is_sender or is_receiver):
        npu_sync()

    total_ms = (time.perf_counter() - pair_start) * 1000.0 if (is_sender or is_receiver) else 0.0
    send_total_ms = total_ms if is_sender else 0.0
    recv_total_ms = total_ms if is_receiver else 0.0
    return (send_alias_ms, send_post_ms, send_wait_ms, send_total_ms,
            recv_alias_ms, recv_post_ms, recv_wait_ms, recv_total_ms)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", default="12:14,13:15,12:15,13:14")
    parser.add_argument("--group-ranks", default="12,13,14,15")
    parser.add_argument("--layers", type=int, default=48)
    parser.add_argument("--experts", type=int, default=64)
    parser.add_argument("--capacity", type=int, default=64)
    parser.add_argument("--w13-tail-shape", default="1536,2048")
    parser.add_argument("--w2-tail-shape", default="2048,768")
    parser.add_argument("--dtype", default="bf16")
    parser.add_argument("--warmup-experts", type=int, default=0)
    parser.add_argument("--sync-before-pair", action="store_true")
    parser.add_argument("--sync-before-alias", action="store_true")
    parser.add_argument("--sync-after-pair", action="store_true")
    args = parser.parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0")))
    torch.npu.set_device(local_rank)
    dist.init_process_group(backend="hccl", init_method="env://")

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    pairs = parse_pair_list(args.pairs)
    group_ranks = parse_int_list(args.group_ranks)
    dtype = get_dtype(args.dtype)
    w13_tail_shape = parse_shape(args.w13_tail_shape)
    w2_tail_shape = parse_shape(args.w2_tail_shape)

    if world_size <= max(max(src, dst) for src, dst in pairs):
        raise RuntimeError(
            f"world_size={world_size} is too small for pairs={pairs}; run with 16 ranks")

    if rank == 0:
        print(
            "BENCH_CONFIG "
            f"world_size={world_size} pairs={pairs} group_ranks={group_ranks} "
            f"layers={args.layers} experts={args.experts} capacity={args.capacity} "
            f"w13_tail={w13_tail_shape} w2_tail={w2_tail_shape} dtype={args.dtype} "
            f"warmup_experts={args.warmup_experts} "
            f"sync_before_pair={args.sync_before_pair} "
            f"sync_before_alias={args.sync_before_alias} sync_after_pair={args.sync_after_pair}",
            flush=True,
        )
        print(
            "BENCH_ENV "
            f"MASTER_ADDR={os.environ.get('MASTER_ADDR')} "
            f"MASTER_PORT={os.environ.get('MASTER_PORT')} "
            f"HCCL_IF_BASE_PORT={os.environ.get('HCCL_IF_BASE_PORT')} "
            f"ASCEND_RT_VISIBLE_DEVICES={os.environ.get('ASCEND_RT_VISIBLE_DEVICES')} "
            f"PYTORCH_NPU_ALLOC_CONF={os.environ.get('PYTORCH_NPU_ALLOC_CONF')}",
            flush=True,
        )

    t0 = time.perf_counter()
    p2p_group = dist.new_group(ranks=group_ranks, backend="hccl")
    dist.barrier()
    if rank == 0:
        print(f"GROUP_INIT_MS {(time.perf_counter() - t0) * 1000.0:.2f}", flush=True)

    needs_tensor = rank in set(group_ranks)
    base_w13 = None
    base_w2 = None
    if needs_tensor:
        base_w13 = torch.empty((args.capacity, ) + w13_tail_shape,
                               device=f"npu:{local_rank}", dtype=dtype)
        base_w2 = torch.empty((args.capacity, ) + w2_tail_shape,
                              device=f"npu:{local_rank}", dtype=dtype)
        # Touch one scalar so the allocator path is definitely initialized without
        # spending time filling hundreds of MB.
        base_w13[0].view(-1)[0].fill_(rank)
        base_w2[0].view(-1)[0].fill_(rank)
        npu_sync()

    bytes_per_expert = (tensor_bytes((1, ) + w13_tail_shape, dtype)
                        + tensor_bytes((1, ) + w2_tail_shape, dtype))
    measured_bytes = bytes_per_expert * args.layers * args.experts

    dist.barrier()

    for src, dst in pairs:
        if args.warmup_experts > 0:
            dist.barrier()
            _ = run_pair(src, dst, rank, p2p_group, base_w13, base_w2,
                         layers=1, experts=args.warmup_experts,
                         capacity=args.capacity,
                         sync_before_alias=args.sync_before_alias,
                         sync_after_pair=True)

        dist.barrier()
        if args.sync_before_pair and needs_tensor:
            npu_sync()
        dist.barrier()

        stats = run_pair(src, dst, rank, p2p_group, base_w13, base_w2,
                         layers=args.layers, experts=args.experts,
                         capacity=args.capacity,
                         sync_before_alias=args.sync_before_alias,
                         sync_after_pair=args.sync_after_pair)

        stats_tensor = torch.tensor(stats, device=f"npu:{local_rank}", dtype=torch.float32)
        dist.all_reduce(stats_tensor, op=dist.ReduceOp.SUM)
        dist.barrier()

        if rank == 0:
            values = [float(x) for x in stats_tensor.cpu().tolist()]
            (send_alias_ms, send_post_ms, send_wait_ms, send_total_ms,
             recv_alias_ms, recv_post_ms, recv_wait_ms, recv_total_ms) = values
            e2e_ms = max(send_total_ms, recv_total_ms)
            wait_ms = max(send_wait_ms, recv_wait_ms)
            e2e_gbps = measured_bytes / max(e2e_ms, 1e-6) / 1e6
            wait_gbps = measured_bytes / max(wait_ms, 1e-6) / 1e6
            print(
                "PAIR_RESULT "
                f"pair={src}->{dst} bytes={measured_bytes} "
                f"send_alias_ms={send_alias_ms:.2f} send_post_ms={send_post_ms:.2f} "
                f"send_wait_ms={send_wait_ms:.2f} send_total_ms={send_total_ms:.2f} "
                f"recv_alias_ms={recv_alias_ms:.2f} recv_post_ms={recv_post_ms:.2f} "
                f"recv_wait_ms={recv_wait_ms:.2f} recv_total_ms={recv_total_ms:.2f} "
                f"e2e_GBps={e2e_gbps:.2f} wait_GBps={wait_gbps:.2f}",
                flush=True,
            )

    dist.barrier()
    if rank == 0:
        print("BENCH_DONE", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
