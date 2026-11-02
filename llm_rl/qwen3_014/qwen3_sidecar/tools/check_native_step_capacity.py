#!/usr/bin/env python3
"""Small real-NPU check: retire F2 pools, then native KV-first F4/F16 wake."""
import os
os.environ["VLLM_ASCEND_NATIVE_SLEEP_PARK_IDLE"] = "1"
os.environ.setdefault("PYTORCH_NPU_ALLOC_CONF", "expandable_segments:False")

import json
from types import SimpleNamespace as NS

import torch
import torch_npu
from vllm_ascend.device_allocator.camem import CaMemAllocator
from vllm_ascend.worker.mode1_step_storage import stage_step_storage, materialize_step_storage


class Experts(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.elastic_moe_mode = "lossless"
        self.elastic_execution_mode = 1
        self.loaded_weight_capacity = 64
        self.w13_weight = torch.nn.Parameter(torch.ones((64,256,256),dtype=torch.bfloat16,device="npu"),requires_grad=False)
        self.w2_weight = torch.nn.Parameter(torch.ones_like(self.w13_weight),requires_grad=False)
        self.runtime_w13_weight = self.w13_weight[:8]
        self.runtime_w2_weight = self.w2_weight[:8]
        self.quant_method = NS(invalidate_lossless_runtime_state_for_reload=self.invalidate)

    def invalidate(self, **kwargs):
        self.runtime_w13_weight = self.runtime_w2_weight = None

    def _get_reserved_local_expert_slots_for_floor(self, floor):
        return 128 // floor


def main():
    torch.npu.set_device(0)
    allocator = CaMemAllocator.get_instance()
    with allocator.use_memory_pool("weights"):
        model = Experts()
        dense = torch.ones(2*1024*1024,device="npu")
    context = NS(kv_cache=[])
    runner = NS(get_model=lambda:model,attn_groups=[],reset_aclgraph_recapture_state=lambda:None)
    def allocate_kv(config):
        with allocator.use_memory_pool("kv_cache"):
            cache = torch.empty(config.num_blocks*8*1024*1024,dtype=torch.float32,device="npu")
            runner.kv_caches = [cache]
            context.kv_cache = [cache]
    worker = NS(model_runner=runner,rank=0,initialize_from_config=allocate_kv,
                vllm_config=NS(compilation_config=NS(static_forward_context={"layer":context})))
    allocate_kv(NS(num_blocks=2))
    dense_pointer = dense.data_ptr()
    identities = (id(model.w13_weight),id(model.w2_weight))
    for floor, blocks in [(4,3),(16,4),(2,2)]:
        allocator.sleep(offload_tags=())
        old_kv = {ptr for ptr,data in allocator.pointer_to_data.items() if data.tag=="kv_cache" and not data.parked}
        stage_step_storage(worker,NS(num_blocks=blocks),floor,resize_kv=True)
        assert all(allocator.pointer_to_data[ptr].parked for ptr in old_kv)
        allocator.wake_up(["kv_cache"])
        materialize_step_storage(worker,["kv_cache"])
        assert all(allocator.pointer_to_data[ptr].parked for ptr in old_kv if allocator.pointer_to_data[ptr].pool_key != allocator.step_pool_keys["kv_cache"])
        allocator.wake_up(["weights"])
        materialize_step_storage(worker,["weights"])
        model.w13_weight.data.fill_(floor)
        model.w2_weight.data.fill_(floor+1)
        dense.fill_(42)
        assert model.w13_weight.shape==(128//floor,256,256)
        assert model.w13_weight[0,0,0].item()==floor
        assert model.w2_weight[-1,-1,-1].item()==floor+1
        assert dense.data_ptr()==dense_pointer and dense[0].item()==42
        assert (id(model.w13_weight),id(model.w2_weight))==identities
        print(json.dumps(dict(floor=floor,kv_blocks=blocks,slots=128//floor,
                              pending=worker._mode1_pending_expert_parameters==[],
                              parked_segments=sum(d.parked for d in allocator.pointer_to_data.values()))),flush=True)
    print("PASS native sleep2: F2->F4->F16->F2 capacity pools, no manual offload",flush=True)


if __name__=="__main__":
    main()
