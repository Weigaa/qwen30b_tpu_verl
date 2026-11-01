#!/usr/bin/env python3
"""Exercise the HCCL operations required before starting the full trainer."""

from __future__ import annotations

from datetime import timedelta
import os

import torch
import torch.distributed as dist
import torch_npu  # noqa: F401


def _assert_close(name: str, actual: torch.Tensor, expected: torch.Tensor) -> None:
    if not torch.equal(actual.cpu(), expected.cpu()):
        raise RuntimeError(
            f"{name} returned an invalid result: actual={actual.cpu().tolist()} "
            f"expected={expected.cpu().tolist()}"
        )


def _mark(rank: int, operation: str) -> None:
    if rank == 0:
        print(f"[hccl smoke] PASS operation={operation}", flush=True)


def main() -> None:
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    if os.environ.get("HCCL_OP_EXPANSION_MODE") != "HOST":
        raise RuntimeError("HCCL_OP_EXPANSION_MODE=HOST is required")
    if os.environ.get("ASCEND_RT_VISIBLE_DEVICES"):
        raise RuntimeError("ASCEND_RT_VISIBLE_DEVICES must be unset")

    torch.npu.set_device(local_rank)
    dist.init_process_group("hccl", timeout=timedelta(seconds=180))
    device = torch.device(f"npu:{local_rank}")
    try:
        value = torch.tensor([float(rank + 1)], device=device)
        dist.all_reduce(value)
        torch.npu.synchronize()
        expected_sum = world_size * (world_size + 1) / 2
        _assert_close("all_reduce", value, torch.tensor([expected_sum]))
        _mark(rank, "all_reduce")

        gathered_objects: list[object] = [None] * world_size
        dist.all_gather_object(gathered_objects, {"rank": rank})
        expected_objects = [{"rank": index} for index in range(world_size)]
        if gathered_objects != expected_objects:
            raise RuntimeError(
                "all_gather_object returned invalid metadata: "
                f"actual={gathered_objects} expected={expected_objects}"
            )
        _mark(rank, "all_gather_object")

        reduce_input = torch.full(
            (world_size,), float(rank + 1), device=device
        )
        reduce_output = torch.empty(1, device=device)
        dist.reduce_scatter_tensor(reduce_output, reduce_input)
        torch.npu.synchronize()
        _assert_close(
            "reduce_scatter_tensor",
            reduce_output,
            torch.tensor([expected_sum]),
        )
        _mark(rank, "reduce_scatter_tensor")

        all_to_all_input = (
            torch.arange(world_size, dtype=torch.float32, device=device)
            + rank * 1000
        )
        all_to_all_output = torch.empty(
            world_size, dtype=torch.float32, device=device
        )
        dist.all_to_all_single(all_to_all_output, all_to_all_input)
        torch.npu.synchronize()
        expected_all_to_all = (
            torch.arange(world_size, dtype=torch.float32) * 1000 + rank
        )
        _assert_close(
            "all_to_all_single", all_to_all_output, expected_all_to_all
        )
        _mark(rank, "all_to_all_single")

        broadcast_value = torch.tensor([float(rank)], device=device)
        dist.broadcast(broadcast_value, src=0)
        torch.npu.synchronize()
        _assert_close("broadcast", broadcast_value, torch.tensor([0.0]))
        _mark(rank, "broadcast")

        dist.barrier(device_ids=[local_rank])
        torch.npu.synchronize()
        _mark(rank, "barrier")
    finally:
        dist.destroy_process_group()

    print(
        f"HCCL_COLLECTIVE_SMOKE_RANK_PASS rank={rank} device={local_rank}",
        flush=True,
    )


if __name__ == "__main__":
    main()
