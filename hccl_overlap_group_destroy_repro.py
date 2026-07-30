#!/usr/bin/env python3
"""Minimal HCCL overlapping-group destroy reproducer.

This script intentionally avoids verl/vllm/vllm_ascend.  It only uses
torch.distributed with the HCCL backend and torch_npu.

Target pattern from the production failure:
  1. Create/use/destroy a 4-rank group, usually [12,13,14,15].
  2. Create/use/destroy another 4-rank group with partial overlap,
     usually [10,11,12,13].
  3. Measure whether the second destroy blocks for tens of seconds while
     HCCL plog reports stale heartbeat/transport state from the first group.

Launch with torchrun or python -m torch.distributed.run.  See
run_hccl_overlap_group_destroy_repro.sh for a ready-to-use local launcher.
"""

from __future__ import annotations

import argparse
import os
import socket
import sys
import time
from datetime import timedelta
from typing import Iterable

import torch

try:
    import torch_npu  # noqa: F401
except Exception as exc:  # pragma: no cover - useful when copied elsewhere.
    print(f"[fatal] failed to import torch_npu: {exc!r}", file=sys.stderr)
    raise

import torch.distributed as dist


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


def _log(msg: str, *, rank: int | None = None) -> None:
    if rank is None:
        rank = int(os.environ.get("RANK", "-1"))
    print(f"[{_now()}][rank={rank}] {msg}", flush=True)


def _parse_group(text: str) -> list[int]:
    ranks: list[int] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo_s, hi_s = part.split("-", 1)
            lo, hi = int(lo_s), int(hi_s)
            step = 1 if hi >= lo else -1
            ranks.extend(range(lo, hi + step, step))
        else:
            ranks.append(int(part))
    return ranks


def _parse_groups(items: Iterable[str]) -> list[list[int]]:
    groups: list[list[int]] = []
    for item in items:
        for group_text in item.split(";"):
            group_text = group_text.strip()
            if group_text:
                groups.append(_parse_group(group_text))
    if not groups:
        raise ValueError("at least one group must be specified")
    return groups


def _is_group_member(group: object) -> bool:
    return group is not None and group is not dist.GroupMember.NON_GROUP_MEMBER


def _device_for_local_rank(local_rank: int) -> torch.device:
    if not hasattr(torch, "npu"):
        raise RuntimeError("torch.npu is unavailable after importing torch_npu")
    torch.npu.set_device(local_rank)
    return torch.device(f"npu:{local_rank}")


def _npu_mem() -> str:
    if not hasattr(torch, "npu"):
        return "npu_mem=unavailable"
    try:
        alloc = torch.npu.memory_allocated() / (1024**3)
        reserved = torch.npu.memory_reserved() / (1024**3)
        return f"npu_alloc_gb={alloc:.3f} npu_reserved_gb={reserved:.3f}"
    except Exception as exc:
        return f"npu_mem_error={exc!r}"


def _run_collective(
    group: dist.ProcessGroup,
    op: str,
    device: torch.device,
    rank: int,
    group_ranks: list[int],
    tensor_numel: int,
    iters: int,
) -> None:
    local_group_rank = group_ranks.index(rank)
    world = len(group_ranks)

    for i in range(iters):
        if op in ("all_reduce", "both"):
            x = torch.full((tensor_numel,), float(rank + i), device=device)
            t0 = time.perf_counter()
            dist.all_reduce(x, group=group)
            torch.npu.synchronize()
            _log(
                f"collective op=all_reduce iter={i} group={group_ranks} "
                f"elapsed_ms={(time.perf_counter() - t0) * 1000:.2f} "
                f"sum0={float(x[0].item()):.1f}",
                rank=rank,
            )

        if op in ("all_to_all", "both"):
            # Equal split all_to_all_single.  This is closer to the all-to-all
            # communication shape than all_reduce, while still staying in
            # plain torch.distributed.
            per_peer = max(1, tensor_numel // max(world, 1))
            inp = torch.full((per_peer * world,), float(local_group_rank + i),
                             device=device)
            out = torch.empty_like(inp)
            t0 = time.perf_counter()
            dist.all_to_all_single(out, inp, group=group)
            torch.npu.synchronize()
            _log(
                f"collective op=all_to_all iter={i} group={group_ranks} "
                f"elapsed_ms={(time.perf_counter() - t0) * 1000:.2f} "
                f"out0={float(out[0].item()):.1f}",
                rank=rank,
            )


def _quiesce_group_before_destroy(
    group: dist.ProcessGroup,
    cpu_group: dist.ProcessGroup | None,
    op: str,
    device: torch.device,
    rank: int,
    group_ranks: list[int],
    pre_sync: bool,
    post_sync: bool,
    cpu_barrier: bool,
    sleep_ms: float,
) -> None:
    """Optionally drain subgroup work before destroy.

    This mirrors the production-side experiment: if the issue is caused by a
    subgroup destroy racing unfinished HCCL/AICPU control-plane work, a tiny
    collective bracketed by CPU barriers should move the delay into this
    explicit quiesce step or remove it from the following new_group.
    """
    if not _is_group_member(group):
        return
    if rank not in group_ranks:
        return

    total_t0 = time.perf_counter()
    pre_barrier_ms = 0.0
    pre_sync_ms = 0.0
    device_op_ms = 0.0
    post_sync_ms = 0.0
    post_barrier_ms = 0.0
    sleep_actual_ms = 0.0

    if cpu_barrier:
        t0 = time.perf_counter()
        if _is_group_member(cpu_group):
            dist.barrier(group=cpu_group)
            pre_barrier_ms = (time.perf_counter() - t0) * 1000
        else:
            pre_barrier_ms = -1.0

    if pre_sync:
        t0 = time.perf_counter()
        torch.npu.synchronize()
        pre_sync_ms = (time.perf_counter() - t0) * 1000

    if op not in ("", "0", "none", "off", "false"):
        t0 = time.perf_counter()
        if op == "all_to_all":
            local_group_rank = group_ranks.index(rank)
            world = len(group_ranks)
            inp = torch.full((world,), local_group_rank, dtype=torch.int32,
                             device=device)
            out = torch.empty_like(inp)
            dist.all_to_all_single(out, inp, group=group)
            del out
            del inp
        else:
            tensor = torch.ones(1, dtype=torch.int32, device=device)
            dist.all_reduce(tensor, group=group)
            del tensor
        device_op_ms = (time.perf_counter() - t0) * 1000

    if post_sync:
        t0 = time.perf_counter()
        torch.npu.synchronize()
        post_sync_ms = (time.perf_counter() - t0) * 1000

    if cpu_barrier:
        t0 = time.perf_counter()
        if _is_group_member(cpu_group):
            dist.barrier(group=cpu_group)
            post_barrier_ms = (time.perf_counter() - t0) * 1000
        else:
            post_barrier_ms = -1.0

    if sleep_ms > 0:
        t0 = time.perf_counter()
        time.sleep(sleep_ms / 1000.0)
        sleep_actual_ms = (time.perf_counter() - t0) * 1000

    _log(
        f"group_quiesce_done group={group_ranks} op={op} "
        f"total_ms={(time.perf_counter() - total_t0) * 1000:.2f} "
        f"pre_barrier_ms={pre_barrier_ms:.2f} pre_sync_ms={pre_sync_ms:.2f} "
        f"device_op_ms={device_op_ms:.2f} post_sync_ms={post_sync_ms:.2f} "
            f"post_barrier_ms={post_barrier_ms:.2f} sleep_ms={sleep_actual_ms:.2f}",
        rank=rank,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--groups",
        nargs="+",
        default=["12-15", "10-13"],
        help="Group sequence. Supports '12-15 10-13' or '12,13,14,15;10,11,12,13'.",
    )
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--op", choices=["all_reduce", "all_to_all", "both"],
                        default="all_to_all")
    parser.add_argument("--collective-iters", type=int, default=2)
    parser.add_argument("--tensor-numel", type=int, default=4096)
    parser.add_argument("--between-groups-sleep", type=float, default=0.0)
    parser.add_argument("--between-rounds-sleep", type=float, default=2.0)
    parser.add_argument("--destroy-world-barrier", action="store_true",
                        help="Run a world barrier after every subgroup destroy.")
    parser.add_argument("--quiesce-before-destroy", action="store_true",
                        help="Drain subgroup work before destroy.")
    parser.add_argument("--quiesce-op", default="all_reduce",
                        choices=["all_reduce", "all_to_all", "none"],
                        help="Tiny device collective used by --quiesce-before-destroy.")
    parser.add_argument("--quiesce-no-pre-sync", action="store_true")
    parser.add_argument("--quiesce-no-post-sync", action="store_true")
    parser.add_argument("--quiesce-no-cpu-barrier", action="store_true")
    parser.add_argument("--quiesce-sleep-ms", type=float, default=0.0)
    parser.add_argument("--no-cpu-group", action="store_true",
                        help="Do not create companion Gloo groups for quiesce barriers.")
    parser.add_argument("--timeout-sec", type=int, default=600)
    parser.add_argument("--backend", default="hccl")
    parser.add_argument("--no-destroy", action="store_true",
                        help="Leak subgroups intentionally; useful as a control run.")
    args = parser.parse_args()

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    host = socket.gethostname()
    groups = _parse_groups(args.groups)

    device = _device_for_local_rank(local_rank)
    dist.init_process_group(
        backend=args.backend,
        timeout=timedelta(seconds=args.timeout_sec),
    )

    if rank == 0:
        _log(
            f"started host={host} world_size={world_size} backend={args.backend} "
            f"groups={groups} rounds={args.rounds} op={args.op} "
            f"collective_iters={args.collective_iters} tensor_numel={args.tensor_numel} "
            f"between_groups_sleep={args.between_groups_sleep} "
            f"between_rounds_sleep={args.between_rounds_sleep} "
            f"quiesce_before_destroy={args.quiesce_before_destroy} "
            f"quiesce_op={args.quiesce_op} "
            f"quiesce_sleep_ms={args.quiesce_sleep_ms} "
            f"no_destroy={args.no_destroy}",
            rank=rank,
        )

    dist.barrier()
    _log(f"initialized local_rank={local_rank} device={device} {_npu_mem()}", rank=rank)

    for round_idx in range(args.rounds):
        for group_idx, group_ranks in enumerate(groups):
            if any(r < 0 or r >= world_size for r in group_ranks):
                raise ValueError(
                    f"group {group_ranks} contains ranks outside world_size={world_size}")

            overlap_prev = sorted(set(groups[group_idx - 1]) & set(group_ranks)) if group_idx else []
            dist.barrier()
            _log(
                f"group_create_begin round={round_idx} group_idx={group_idx} "
                f"group={group_ranks} overlap_prev={overlap_prev}",
                rank=rank,
            )
            t0 = time.perf_counter()
            group = dist.new_group(ranks=group_ranks, backend=args.backend)
            cpu_group = None
            if not args.no_cpu_group:
                cpu_group = dist.new_group(ranks=group_ranks, backend="gloo")
            create_ms = (time.perf_counter() - t0) * 1000
            member = _is_group_member(group)
            _log(
                f"group_create_done round={round_idx} group_idx={group_idx} "
                f"group={group_ranks} member={int(member)} create_ms={create_ms:.2f} "
                f"{_npu_mem()}",
                rank=rank,
            )

            dist.barrier()
            if member:
                _run_collective(
                    group=group,
                    op=args.op,
                    device=device,
                    rank=rank,
                    group_ranks=group_ranks,
                    tensor_numel=args.tensor_numel,
                    iters=args.collective_iters,
                )

            dist.barrier()
            if member and not args.no_destroy:
                if args.quiesce_before_destroy:
                    _quiesce_group_before_destroy(
                        group=group,
                        cpu_group=cpu_group,
                        op=args.quiesce_op,
                        device=device,
                        rank=rank,
                        group_ranks=group_ranks,
                        pre_sync=not args.quiesce_no_pre_sync,
                        post_sync=not args.quiesce_no_post_sync,
                        cpu_barrier=not args.quiesce_no_cpu_barrier,
                        sleep_ms=args.quiesce_sleep_ms,
                    )
                _log(
                    f"group_destroy_begin round={round_idx} group_idx={group_idx} "
                    f"group={group_ranks} {_npu_mem()}",
                    rank=rank,
                )
                t0 = time.perf_counter()
                dist.destroy_process_group(group)
                if _is_group_member(cpu_group):
                    dist.destroy_process_group(cpu_group)
                destroy_ms = (time.perf_counter() - t0) * 1000
                # Synchronize after destroy so logs are aligned with HCCL stream cleanup.
                try:
                    torch.npu.synchronize()
                except Exception as exc:
                    _log(f"npu_sync_after_destroy_error={exc!r}", rank=rank)
                _log(
                    f"group_destroy_done round={round_idx} group_idx={group_idx} "
                    f"group={group_ranks} destroy_ms={destroy_ms:.2f} {_npu_mem()}",
                    rank=rank,
                )
            elif member:
                _log(
                    f"group_destroy_skipped round={round_idx} group_idx={group_idx} "
                    f"group={group_ranks} reason=no_destroy",
                    rank=rank,
                )

            if args.destroy_world_barrier:
                dist.barrier()
                _log(
                    f"post_destroy_world_barrier round={round_idx} group_idx={group_idx}",
                    rank=rank,
                )

            if args.between_groups_sleep > 0 and group_idx + 1 < len(groups):
                time.sleep(args.between_groups_sleep)

        if args.between_rounds_sleep > 0 and round_idx + 1 < args.rounds:
            time.sleep(args.between_rounds_sleep)

    dist.barrier()
    _log("final_world_barrier_done", rank=rank)
    dist.destroy_process_group()
    _log("world_destroy_done", rank=rank)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
