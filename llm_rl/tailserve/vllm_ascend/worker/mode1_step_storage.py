# SPDX-License-Identifier: Apache-2.0
"""Per-step MoE/KV capacity changes at a native level-2 sleep boundary.

No CPU backup/manual offload of rollout weights. Unchanged allocations keep
normal native wake semantics. Only storage whose CAPACITY changes is retired,
while it is already sleeping, then allocated in its native tagged pool after
KV-first wake. Every parameter is populated by the normal RL weight loader.
"""
import gc

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)


def stage_step_storage(worker, kv_config, target_floor: int, *, resize_kv: bool):
    runner = worker.model_runner
    model = runner.get_model()
    pending = []
    changed_layers = 0
    for module in model.modules():
        if (getattr(module, "elastic_moe_mode", None) != "lossless"
                or int(getattr(module, "elastic_execution_mode", 0)) != 1):
            continue
        reserve = getattr(module, "_get_reserved_local_expert_slots_for_floor", None)
        if not callable(reserve):
            continue
        capacity = int(reserve(target_floor))
        if all(int(getattr(module, name).shape[0]) == capacity
               for name in ("w13_weight", "w2_weight")):
            continue
        invalidate = getattr(module.quant_method,
                             "invalidate_lossless_runtime_state_for_reload", None)
        if not callable(invalidate):
            raise RuntimeError("Missing lossless runtime invalidation for capacity resize")
        invalidate(layer=module, reason="native_sleep_step_capacity")
        for name in ("w13_weight", "w2_weight"):
            parameter = getattr(module, name)
            if parameter.ndim != 3 or not parameter.is_floating_point():
                raise RuntimeError("Step slot resize requires unquantized 3D MoE weights")
            shape = (capacity, *parameter.shape[1:])
            pending.append((parameter, shape, parameter.dtype, parameter.device))
            # No read/copy of sleeping pages. Parameter identity/loader attrs
            # remain intact; old runtime views were retired above.
            parameter.data = torch.empty(0, dtype=parameter.dtype, device=parameter.device)
        module.loaded_weight_capacity = capacity
        changed_layers += 1

    if resize_kv:
        from vllm.distributed.kv_transfer import has_kv_transfer_group
        if has_kv_transfer_group():
            raise RuntimeError("Step KV resize does not support an external KV connector")
        context = worker.vllm_config.compilation_config.static_forward_context
        for layer in context.values():
            if hasattr(layer, "kv_cache"):
                layer.kv_cache = [torch.tensor([])]
            impl = getattr(layer, "impl", None)
            if impl is not None:
                for name in ("key_cache", "value_cache"):
                    if hasattr(impl, name):
                        setattr(impl, name, None)
        runner.kv_caches = []
        runner.attn_groups = []
        runner.kv_connector_output = None
        for name in ("cross_layers_kv_cache", "cross_layers_attn_backend"):
            if hasattr(runner, name):
                setattr(runner, name, None)
        worker._mode1_pending_kv_config = kv_config
    runner.reset_aclgraph_recapture_state()
    worker._mode1_pending_expert_parameters = pending
    gc.collect()
    torch.npu.synchronize()
    torch.npu.empty_cache()
    from vllm_ascend.device_allocator.camem import CaMemAllocator
    allocator = CaMemAllocator.get_instance()
    retirement = allocator.park_retired_step_segments(("weights", "kv_cache"))
    if pending:
        allocator.select_step_pool("weights", target_floor)
    if resize_kv:
        allocator.select_step_pool("kv_cache", kv_config.num_blocks)
    logger.info("Mode1 native step storage staged: rank=%s floor=%s "
                "changed_moe_layers=%s resize_kv=%s blocks=%s retirement=%s",
                worker.rank, target_floor, changed_layers, resize_kv,
                kv_config.num_blocks, retirement)


def materialize_step_storage(worker, tags):
    from vllm_ascend.device_allocator.camem import CaMemAllocator
    allocator = CaMemAllocator.get_instance()
    kv = getattr(worker, "_mode1_pending_kv_config", None)
    if kv is not None and (tags is None or "kv_cache" in tags):
        worker.initialize_from_config(kv)
        worker._mode1_pending_kv_config = None
        logger.info("Mode1 native step KV allocated: rank=%s blocks=%s",
                    worker.rank, kv.num_blocks)
    pending = getattr(worker, "_mode1_pending_expert_parameters", [])
    if pending and (tags is None or "weights" in tags):
        with allocator.use_memory_pool(tag="weights"):
            for parameter, shape, dtype, device in pending:
                parameter.data = torch.empty(shape, dtype=dtype, device=device)
        worker._mode1_pending_expert_parameters = []
        logger.info("Mode1 native step expert storage allocated: rank=%s parameters=%s",
                    worker.rank, len(pending))
