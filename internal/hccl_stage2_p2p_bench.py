#!/usr/bin/env python3
import datetime as _datetime
import json
import os
import socket
import statistics
import sys
import time
from typing import Iterable, List, Tuple

import torch
import torch.distributed as dist

try:
    import torch_npu  # noqa: F401
except Exception as exc:  # pragma: no cover - this script is NPU-only.
    print(json.dumps({"event": "import_error", "module": "torch_npu", "error": repr(exc)}), flush=True)
    raise


def _now() -> float:
    return time.perf_counter()


def _sync_npu() -> None:
    try:
        torch.npu.synchronize()
    except Exception:
        pass


def _device_name(local_rank: int) -> str:
    return f"npu:{local_rank}"


def _json(event: str, **payload) -> None:
    payload = {"event": event, **payload}
    print(json.dumps(payload, sort_keys=True), flush=True)


def _parse_mb_list(raw: str) -> List[int]:
    out = []
    for part in raw.split(","):
        part = part.strip()
        if part:
            out.append(int(part))
    return out or [16, 64, 288]


def _tensor_for_mb(mb: int, local_rank: int) -> torch.Tensor:
    # float16 keeps the byte accounting simple and is supported by HCCL P2P.
    numel = max(1, mb * 1024 * 1024 // 2)
    return torch.empty(numel, dtype=torch.float16, device=_device_name(local_rank))


def _barrier(label: str, rank: int) -> float:
    _sync_npu()
    t0 = _now()
    dist.barrier()
    _sync_npu()
    elapsed_ms = (_now() - t0) * 1000.0
    _json("barrier", label=label, rank=rank, elapsed_ms=elapsed_ms)
    return elapsed_ms


def _wait_all(works: Iterable[dist.Work]) -> None:
    for work in works:
        if work is not None:
            work.wait()


def run_p2p_pattern(
    name: str,
    pairs: List[Tuple[int, int]],
    mb: int,
    repeats: int,
    rank: int,
    local_rank: int,
) -> None:
    send_tensor = _tensor_for_mb(mb, local_rank)
    recv_tensor = _tensor_for_mb(mb, local_rank)
    send_tensor.fill_(rank + 1)
    recv_tensor.zero_()
    _sync_npu()

    elapsed = []
    for i in range(repeats):
        _barrier(f"{name}:pre:{mb}MB:{i}", rank)
        t0 = _now()
        works = []
        for src, dst in pairs:
            if rank == src:
                works.append(dist.isend(send_tensor, dst=dst))
            if rank == dst:
                works.append(dist.irecv(recv_tensor, src=src))
        _wait_all(works)
        _sync_npu()
        elapsed_ms = (_now() - t0) * 1000.0
        _barrier(f"{name}:post:{mb}MB:{i}", rank)
        elapsed.append(elapsed_ms)
        _json(
            "p2p_iter",
            name=name,
            rank=rank,
            local_rank=local_rank,
            mb=mb,
            iter=i,
            pairs=pairs,
            elapsed_ms=elapsed_ms,
        )

    _json(
        "p2p_summary",
        name=name,
        rank=rank,
        local_rank=local_rank,
        mb=mb,
        repeats=repeats,
        pairs=pairs,
        p50_ms=statistics.median(elapsed),
        max_ms=max(elapsed),
        min_ms=min(elapsed),
    )


def run_all_reduce(name: str, mb: int, repeats: int, rank: int, local_rank: int) -> None:
    tensor = _tensor_for_mb(mb, local_rank)
    tensor.fill_(rank + 1)
    _sync_npu()

    elapsed = []
    for i in range(repeats):
        _barrier(f"{name}:pre:{mb}MB:{i}", rank)
        t0 = _now()
        dist.all_reduce(tensor)
        _sync_npu()
        elapsed_ms = (_now() - t0) * 1000.0
        _barrier(f"{name}:post:{mb}MB:{i}", rank)
        elapsed.append(elapsed_ms)
        _json(
            "all_reduce_iter",
            name=name,
            rank=rank,
            local_rank=local_rank,
            mb=mb,
            iter=i,
            elapsed_ms=elapsed_ms,
        )

    _json(
        "all_reduce_summary",
        name=name,
        rank=rank,
        local_rank=local_rank,
        mb=mb,
        repeats=repeats,
        p50_ms=statistics.median(elapsed),
        max_ms=max(elapsed),
        min_ms=min(elapsed),
    )


def run_pair_matrix(mb: int, repeats: int, rank: int, local_rank: int) -> None:
    for src in range(4):
        for dst in range(4):
            if src == dst:
                continue
            run_p2p_pattern(
                f"pair_{src}_to_{dst}",
                [(src, dst)],
                mb,
                repeats,
                rank,
                local_rank,
            )


def main() -> None:
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size != 4:
        raise RuntimeError(f"this bench expects WORLD_SIZE=4, got {world_size}")

    torch.npu.set_device(local_rank)
    timeout_s = int(os.environ.get("BENCH_DIST_TIMEOUT_S", "180"))
    t0 = _now()
    dist.init_process_group(
        backend="hccl",
        timeout=_datetime.timedelta(seconds=timeout_s),
    )
    init_ms = (_now() - t0) * 1000.0
    _sync_npu()

    if rank == 0:
        _json(
            "bench_start",
            hostname=socket.gethostname(),
            python=sys.version.replace("\n", " "),
            torch=getattr(torch, "__version__", "<unknown>"),
            torch_file=getattr(torch, "__file__", "<unknown>"),
            visible_devices=os.environ.get("ASCEND_RT_VISIBLE_DEVICES", ""),
            master_addr=os.environ.get("MASTER_ADDR", ""),
            master_port=os.environ.get("MASTER_PORT", ""),
            hccl_if_base_port=os.environ.get("HCCL_IF_BASE_PORT", ""),
            hccl_connect_timeout=os.environ.get("HCCL_CONNECT_TIMEOUT", ""),
        )
    _json("rank_init", rank=rank, local_rank=local_rank, init_ms=init_ms)

    mb_list = _parse_mb_list(os.environ.get("BENCH_MB_LIST", "16,64,288"))
    repeats = int(os.environ.get("BENCH_REPEATS", "3"))

    _barrier("after_init", rank)
    for mb in mb_list:
        run_all_reduce("world4_all_reduce", mb, repeats, rank, local_rank)
        # This mirrors mode=1 stage=2 paired import with visible devices 12,13,14,15:
        # logical ranks 0,1 are sources and logical ranks 2,3 are active targets.
        run_p2p_pattern("stage2_sources_0_1_to_active_2_3", [(0, 2), (1, 3)], mb, repeats, rank, local_rank)
        run_p2p_pattern("stage2_reverse_active_2_3_to_sources_0_1", [(2, 0), (3, 1)], mb, repeats, rank, local_rank)
        run_p2p_pattern("active_pair_2_to_3", [(2, 3)], mb, repeats, rank, local_rank)
        run_p2p_pattern("active_pair_3_to_2", [(3, 2)], mb, repeats, rank, local_rank)
        if os.environ.get("BENCH_PAIR_MATRIX", "0") == "1":
            run_pair_matrix(mb, repeats, rank, local_rank)

    _barrier("before_destroy", rank)
    dist.destroy_process_group()
    _json("bench_done", rank=rank, local_rank=local_rank)


if __name__ == "__main__":
    main()
