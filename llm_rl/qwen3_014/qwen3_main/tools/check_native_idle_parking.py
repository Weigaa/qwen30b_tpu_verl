#!/usr/bin/env python3
"""Small real-NPU test of physical idle-cache release and native sleep reuse."""
import os

os.environ["VLLM_ASCEND_NATIVE_SLEEP_PARK_IDLE"] = "1"
os.environ.setdefault("PYTORCH_NPU_ALLOC_CONF", "expandable_segments:False")

import torch
import torch_npu
from vllm_ascend.device_allocator.camem import CaMemAllocator


def main():
    torch.npu.set_device(0)
    allocator = CaMemAllocator.get_instance()
    with allocator.use_memory_pool("weights"):
        live = torch.ones(1024 * 1024, device="npu")
    with allocator.use_memory_pool("kv_cache"):
        kv = torch.ones(1024 * 1024, device="npu")
    pointer = live.data_ptr()
    for cycle in range(3):
        with allocator.use_memory_pool("weights"):
            temporary = torch.ones(32 * 1024 * 1024, device="npu")
            live.fill_(cycle + 1)
            torch.npu.synchronize()
            del temporary
        report = allocator.park_idle_weight_segments()
        assert report["released_bytes"] >= 128 * 1024**2, report
        assert report["physical_free_after"] - report["physical_free_before"] >= 64 * 1024**2, report
        assert live.data_ptr() == pointer and live[0].item() == cycle + 1
        assert kv[0].item() == 1
        assert all(not d.parked for d in allocator.pointer_to_data.values() if d.tag == "kv_cache")
        allocator.sleep(offload_tags=())
        allocator.wake_up(["kv_cache"])
        allocator.wake_up(["weights"])
        # Level2 deliberately discards live contents; refill before reading.
        live.fill_(cycle + 2)
        kv.fill_(1)
        assert live[0].item() == cycle + 2
        print("native_idle_park_cycle", cycle + 1, report, flush=True)
    print("PASS: physical idle release, native sleep2 and same-address pool reuse")


if __name__ == "__main__":
    main()
