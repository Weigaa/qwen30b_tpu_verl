"""Opt-in, metadata-only accounting; never read sleeping tensor contents."""
import json
import os
from collections import defaultdict

import torch


def log_npu_memory_ownership(stage, model=None):
    if os.getenv("VLLM_ROLLOUT_MEMORY_OWNERSHIP_LOG", "0") != "1":
        return
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    if rank not in (0, 4, 7):
        return
    from vllm_ascend.device_allocator.camem import CaMemAllocator
    allocator = CaMemAllocator.get_instance()
    torch.npu.synchronize()
    tags = defaultdict(lambda: dict(registered_capacity=0, parked_capacity=0, segments=0, allocated=0, active=0, idle=0))
    for data in allocator.pointer_to_data.values():
        tags[data.tag]["registered_capacity"] += data.handle[1]
        if getattr(data, "parked", False):
            tags[data.tag]["parked_capacity"] += data.handle[1]
    for segment in torch.npu.memory_snapshot():
        if segment.get("device", torch.npu.current_device()) != torch.npu.current_device():
            continue
        data = allocator.pointer_to_data.get(segment["address"])
        tag = data.tag if data else "non_camem"
        item = tags[tag]
        item["segments"] += 1
        item["allocated"] += segment["allocated_size"]
        item["active"] += segment["active_size"]
        item["idle"] += segment["total_size"] - segment["active_size"]
    sample = []
    if model is not None:
        import torch_npu
        for name, param in model.named_parameters():
            if ".layers.0." in name and ("w13_weight" in name or "w2_weight" in name):
                sample.append(dict(name=name, shape=list(param.shape), stride=list(param.stride()),
                                   format=str(torch_npu.get_npu_format(param)),
                                   storage_bytes=param.untyped_storage().nbytes()))
    print("NPU_MEMORY_OWNERSHIP " + json.dumps(dict(
        stage=stage, rank=rank, hccl_buffer_mib=os.getenv("HCCL_BUFFSIZE"),
        physical_free=torch.npu.mem_get_info()[0],
        tags=dict(tags), expert_weight_layouts=sample)), flush=True)
