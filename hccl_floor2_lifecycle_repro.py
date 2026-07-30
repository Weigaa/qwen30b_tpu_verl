#!/usr/bin/env python3
"""Minimal HCCL floor2 lifecycle reproducer.

This intentionally avoids verl/vllm/vllm_ascend.  It uses only
Torch distributed with HCCL plus torch_npu, and simulates the elastic
lifecycle that became slow in the production floor=2 run:

  full world [0..15] -> floor8 [8..15] -> floor4 [12..15]
  -> floor2 [14,15] -> restore full world [0..15]

Every stage creates DP/EP/MC2-like HCCL groups in the same order.  In
placeholder mode, non-member ranks also call dist.new_group with the stage
ranks, matching the production pattern where all ranks advance the HCCL group
sequence even if they are not active members.

The output is line-oriented and grep-friendly.  Look for per-step growth in:
  stage_create_done total_ms=...
  stage_destroy_done total_ms=...
  lifecycle_step_done total_ms=...
  collective_done elapsed_ms=...
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
except Exception as exc:
    print(f"[fatal] failed to import torch_npu: {exc!r}", file=sys.stderr)
    raise

import torch.distributed as dist


GROUP_KINDS = ("dp", "ep", "mc2")


def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


def log(rank: int, msg: str) -> None:
    print(f"[{now()}][rank={rank}] {msg}", flush=True)


def parse_group(text: str) -> list[int]:
    out: list[int] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo_s, hi_s = part.split("-", 1)
            lo, hi = int(lo_s), int(hi_s)
            step = 1 if hi >= lo else -1
            out.extend(range(lo, hi + step, step))
        else:
            out.append(int(part))
    return out


def parse_stages(items: Iterable[str]) -> list[list[int]]:
    stages: list[list[int]] = []
    for item in items:
        for group_text in item.split(";"):
            group_text = group_text.strip()
            if group_text:
                stages.append(parse_group(group_text))
    if not stages:
        raise ValueError("empty stage list")
    return stages


def is_member(group: object) -> bool:
    return group is not None and group is not dist.GroupMember.NON_GROUP_MEMBER


def setup_device(local_rank: int) -> torch.device:
    torch.npu.set_device(local_rank)
    return torch.device(f"npu:{local_rank}")


def npu_mem() -> str:
    try:
        free, total = torch.npu.mem_get_info()
        alloc = torch.npu.memory_allocated()
        reserved = torch.npu.memory_reserved()
        non_torch = total - free - reserved
        gb = 1024 ** 3
        return (
            f"free_gb={free / gb:.3f} total_gb={total / gb:.3f} "
            f"torch_alloc_gb={alloc / gb:.3f} torch_reserved_gb={reserved / gb:.3f} "
            f"non_torch_gb={non_torch / gb:.3f}"
        )
    except Exception as exc:
        return f"npu_mem_error={exc!r}"


def do_collective(
    *,
    rank: int,
    group_ranks: list[int],
    group: dist.ProcessGroup,
    kind: str,
    op: str,
    tensor_numel: int,
    iters: int,
    device: torch.device,
    step_idx: int,
    stage_idx: int,
) -> None:
    if rank not in group_ranks or not is_member(group):
        return
    world = len(group_ranks)
    local_rank = group_ranks.index(rank)
    per_peer = max(1, tensor_numel // max(world, 1))
    for iter_idx in range(iters):
        if op in ("all_reduce", "both"):
            x = torch.full((tensor_numel,), float(rank + iter_idx), device=device)
            t0 = time.perf_counter()
            dist.all_reduce(x, group=group)
            torch.npu.synchronize()
            elapsed_ms = (time.perf_counter() - t0) * 1000
            log(rank, (
                f"collective_done step={step_idx} stage={stage_idx} kind={kind} "
                f"op=all_reduce iter={iter_idx} group={group_ranks} "
                f"elapsed_ms={elapsed_ms:.2f} value0={float(x[0].item()):.1f} {npu_mem()}"
            ))
            del x
        if op in ("all_to_all", "both"):
            inp = torch.full((per_peer * world,), float(local_rank + iter_idx), device=device)
            out = torch.empty_like(inp)
            t0 = time.perf_counter()
            dist.all_to_all_single(out, inp, group=group)
            torch.npu.synchronize()
            elapsed_ms = (time.perf_counter() - t0) * 1000
            log(rank, (
                f"collective_done step={step_idx} stage={stage_idx} kind={kind} "
                f"op=all_to_all iter={iter_idx} group={group_ranks} "
                f"elapsed_ms={elapsed_ms:.2f} out0={float(out[0].item()):.1f} {npu_mem()}"
            ))
            del out, inp


def create_stage_groups(
    *,
    rank: int,
    stage_idx: int,
    step_idx: int,
    group_ranks: list[int],
    backend: str,
    create_cpu_groups: bool,
    placeholder_mode: bool,
) -> tuple[dict[str, object], dict[str, object]]:
    # In placeholder_mode all ranks call new_group with the same ranks/order.
    # In members_only, non-members skip subgroup creation entirely.
    h_groups: dict[str, object] = {}
    c_groups: dict[str, object] = {}
    should_call = placeholder_mode or rank in group_ranks
    t_stage = time.perf_counter()
    for kind in GROUP_KINDS:
        if should_call:
            t0 = time.perf_counter()
            g = dist.new_group(ranks=group_ranks, backend=backend)
            create_ms = (time.perf_counter() - t0) * 1000
            h_groups[kind] = g
            log(rank, (
                f"group_create_done step={step_idx} stage={stage_idx} kind={kind} "
                f"group={group_ranks} member={int(is_member(g))} create_ms={create_ms:.2f} {npu_mem()}"
            ))
            if create_cpu_groups:
                t0 = time.perf_counter()
                cg = dist.new_group(ranks=group_ranks, backend="gloo")
                cpu_ms = (time.perf_counter() - t0) * 1000
                c_groups[kind] = cg
                log(rank, (
                    f"cpu_group_create_done step={step_idx} stage={stage_idx} kind={kind} "
                    f"group={group_ranks} member={int(is_member(cg))} create_ms={cpu_ms:.2f}"
                ))
        else:
            h_groups[kind] = dist.GroupMember.NON_GROUP_MEMBER
            c_groups[kind] = dist.GroupMember.NON_GROUP_MEMBER
            log(rank, (
                f"group_create_skipped step={step_idx} stage={stage_idx} kind={kind} "
                f"group={group_ranks} reason=non_member_members_only"
            ))
    log(rank, (
        f"stage_create_done step={step_idx} stage={stage_idx} group={group_ranks} "
        f"total_ms={(time.perf_counter() - t_stage) * 1000:.2f} {npu_mem()}"
    ))
    return h_groups, c_groups


def destroy_stage_groups(
    *,
    rank: int,
    step_idx: int,
    stage_idx: int,
    group_ranks: list[int],
    h_groups: dict[str, object],
    c_groups: dict[str, object],
    destroy_cpu_groups: bool,
    pre_sync: bool,
    post_sync: bool,
    world_barrier_after_destroy: bool,
    no_destroy: bool,
) -> None:
    if no_destroy:
        log(rank, f"stage_destroy_skipped step={step_idx} stage={stage_idx} group={group_ranks} reason=no_destroy")
        return
    t_stage = time.perf_counter()
    for kind in reversed(GROUP_KINDS):
        g = h_groups.get(kind)
        cg = c_groups.get(kind)
        if is_member(g):
            if pre_sync:
                torch.npu.synchronize()
            t0 = time.perf_counter()
            dist.destroy_process_group(g)  # type: ignore[arg-type]
            destroy_ms = (time.perf_counter() - t0) * 1000
            if post_sync:
                torch.npu.synchronize()
            log(rank, (
                f"group_destroy_done step={step_idx} stage={stage_idx} kind={kind} "
                f"group={group_ranks} destroy_ms={destroy_ms:.2f} {npu_mem()}"
            ))
        else:
            log(rank, (
                f"group_destroy_skip_non_member step={step_idx} stage={stage_idx} kind={kind} "
                f"group={group_ranks}"
            ))
        if destroy_cpu_groups and is_member(cg):
            t0 = time.perf_counter()
            dist.destroy_process_group(cg)  # type: ignore[arg-type]
            cpu_ms = (time.perf_counter() - t0) * 1000
            log(rank, (
                f"cpu_group_destroy_done step={step_idx} stage={stage_idx} kind={kind} "
                f"group={group_ranks} destroy_ms={cpu_ms:.2f}"
            ))
    log(rank, (
        f"stage_destroy_done step={step_idx} stage={stage_idx} group={group_ranks} "
        f"total_ms={(time.perf_counter() - t_stage) * 1000:.2f} {npu_mem()}"
    ))
    if world_barrier_after_destroy:
        t0 = time.perf_counter()
        dist.barrier()
        log(rank, (
            f"post_stage_destroy_world_barrier_done step={step_idx} stage={stage_idx} "
            f"elapsed_ms={(time.perf_counter() - t0) * 1000:.2f}"
        ))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stages", nargs="+", default=["0-15", "8-15", "12-15", "14-15", "0-15"],
                        help="Stage rank groups. Default simulates full->8->4->2->full restore.")
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--backend", default="hccl")
    parser.add_argument("--timeout-sec", type=int, default=900)
    parser.add_argument("--op", choices=["all_reduce", "all_to_all", "both"], default="all_to_all")
    parser.add_argument("--collective-iters", type=int, default=1)
    parser.add_argument("--tensor-numel", type=int, default=8192)
    parser.add_argument("--placeholder-mode", action="store_true", default=True,
                        help="All ranks call new_group for every stage. Default: on.")
    parser.add_argument("--members-only", action="store_true",
                        help="Only stage members call new_group; disables placeholder mode.")
    parser.add_argument("--create-cpu-groups", action="store_true")
    parser.add_argument("--destroy-cpu-groups", action="store_true")
    parser.add_argument("--no-destroy", action="store_true")
    parser.add_argument("--no-pre-destroy-sync", action="store_true")
    parser.add_argument("--no-post-destroy-sync", action="store_true")
    parser.add_argument("--world-barrier-after-destroy", action="store_true")
    parser.add_argument("--sleep-between-stages-ms", type=float, default=0.0)
    parser.add_argument("--sleep-between-steps-ms", type=float, default=0.0)
    args = parser.parse_args()

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    device = setup_device(local_rank)
    stages = parse_stages(args.stages)
    placeholder_mode = args.placeholder_mode and not args.members_only

    dist.init_process_group(backend=args.backend, timeout=timedelta(seconds=args.timeout_sec))
    if rank == 0:
        log(rank, (
            f"started host={socket.gethostname()} world_size={world_size} backend={args.backend} "
            f"stages={stages} steps={args.steps} op={args.op} tensor_numel={args.tensor_numel} "
            f"collective_iters={args.collective_iters} placeholder_mode={int(placeholder_mode)} "
            f"create_cpu_groups={int(args.create_cpu_groups)} no_destroy={int(args.no_destroy)}"
        ))
    dist.barrier()
    log(rank, f"initialized local_rank={local_rank} device={device} {npu_mem()}")

    for step_idx in range(1, args.steps + 1):
        t_step = time.perf_counter()
        prev_groups: tuple[dict[str, object], dict[str, object], list[int], int] | None = None
        for stage_idx, group_ranks in enumerate(stages):
            if any(r < 0 or r >= world_size for r in group_ranks):
                raise ValueError(f"stage {stage_idx} ranks outside world_size={world_size}: {group_ranks}")
            t0 = time.perf_counter()
            dist.barrier()
            log(rank, (
                f"stage_begin step={step_idx} stage={stage_idx} group={group_ranks} "
                f"world_barrier_ms={(time.perf_counter() - t0) * 1000:.2f} {npu_mem()}"
            ))

            h_groups, c_groups = create_stage_groups(
                rank=rank,
                stage_idx=stage_idx,
                step_idx=step_idx,
                group_ranks=group_ranks,
                backend=args.backend,
                create_cpu_groups=args.create_cpu_groups,
                placeholder_mode=placeholder_mode,
            )

            # Use MC2-like group for active-stage traffic.  This is the group that
            # production code most often exercises immediately after rebuild.
            do_collective(
                rank=rank,
                group_ranks=group_ranks,
                group=h_groups["mc2"],  # type: ignore[arg-type]
                kind="mc2",
                op=args.op,
                tensor_numel=args.tensor_numel,
                iters=args.collective_iters,
                device=device,
                step_idx=step_idx,
                stage_idx=stage_idx,
            )
            dist.barrier()

            if prev_groups is not None:
                old_h, old_c, old_ranks, old_stage_idx = prev_groups
                destroy_stage_groups(
                    rank=rank,
                    step_idx=step_idx,
                    stage_idx=old_stage_idx,
                    group_ranks=old_ranks,
                    h_groups=old_h,
                    c_groups=old_c,
                    destroy_cpu_groups=args.destroy_cpu_groups,
                    pre_sync=not args.no_pre_destroy_sync,
                    post_sync=not args.no_post_destroy_sync,
                    world_barrier_after_destroy=args.world_barrier_after_destroy,
                    no_destroy=args.no_destroy,
                )
            prev_groups = (h_groups, c_groups, group_ranks, stage_idx)

            if args.sleep_between_stages_ms > 0 and stage_idx + 1 < len(stages):
                time.sleep(args.sleep_between_stages_ms / 1000.0)

        if prev_groups is not None:
            old_h, old_c, old_ranks, old_stage_idx = prev_groups
            destroy_stage_groups(
                rank=rank,
                step_idx=step_idx,
                stage_idx=old_stage_idx,
                group_ranks=old_ranks,
                h_groups=old_h,
                c_groups=old_c,
                destroy_cpu_groups=args.destroy_cpu_groups,
                pre_sync=not args.no_pre_destroy_sync,
                post_sync=not args.no_post_destroy_sync,
                world_barrier_after_destroy=args.world_barrier_after_destroy,
                no_destroy=args.no_destroy,
            )

        t0 = time.perf_counter()
        dist.barrier()
        final_barrier_ms = (time.perf_counter() - t0) * 1000
        log(rank, (
            f"lifecycle_step_done step={step_idx} total_ms={(time.perf_counter() - t_step) * 1000:.2f} "
            f"final_barrier_ms={final_barrier_ms:.2f} {npu_mem()}"
        ))
        if args.sleep_between_steps_ms > 0 and step_idx < args.steps:
            time.sleep(args.sleep_between_steps_ms / 1000.0)

    dist.barrier()
    log(rank, "final_world_barrier_done")
    dist.destroy_process_group()
    log(rank, "world_destroy_done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
