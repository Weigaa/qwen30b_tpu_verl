#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2023 The vLLM team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
# Adapted from vllm-project/vllm/vllm/worker/gpu_worker.py
#

import copy
import os
import time
from typing import Optional, Union

import torch
import torch.nn as nn
import torch.distributed as dist
import torch_npu
import vllm.envs as envs_vllm
from torch_npu.op_plugin.atb._atb_ops import _register_atb_extensions
from torch_npu.profiler import dynamic_profile as dp
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.distributed import (ensure_model_parallel_initialized,
                              init_distributed_environment)
from vllm.distributed.kv_transfer import ensure_kv_transfer_initialized
from vllm.distributed.parallel_state import (get_dp_group, get_pp_group,
                                             get_tp_group, get_ep_group,
                                             get_world_group,
                                             init_model_parallel_group)
from vllm.logger import logger
from vllm.lora.request import LoRARequest
from vllm.sequence import IntermediateTensors
from vllm.tasks import SupportedTask
from vllm.utils import STR_DTYPE_TO_TORCH_DTYPE, GiB_bytes
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheSpec
from vllm.v1.outputs import (EMPTY_MODEL_RUNNER_OUTPUT, AsyncModelRunnerOutput,
                             DraftTokenIds, ModelRunnerOutput)
from vllm.v1.worker.worker_base import WorkerBase

import vllm_ascend.envs as envs_ascend
from vllm_ascend.ascend_config import get_ascend_config, init_ascend_config
from vllm_ascend.device_allocator.camem import CaMemAllocator
from vllm_ascend.distributed.parallel_state import (get_mc2_group,
                                                    init_ascend_model_parallel)
from vllm_ascend.platform import NPUPlatform
from vllm_ascend.utils import (init_ascend_soc_version,
                               register_ascend_customop, sleep_mode_enabled,
                               try_register_lib)
from vllm_ascend.worker.model_runner_v1 import NPUModelRunner

torch._dynamo.trace_rules.clear_lru_cache()  # noqa: E402
from torch._dynamo.variables import TorchInGraphFunctionVariable  # noqa: E402

torch_non_c_binding_in_graph_functions_npu = dict.fromkeys(
    ["torch.npu.current_stream"],
    TorchInGraphFunctionVariable,
)  # noqa: E402
torch_non_c_binding_in_graph_functions_npu[
    "torch.npu.stream"] = TorchInGraphFunctionVariable  # noqa: E402
torch._dynamo.trace_rules.torch_name_rule_map.append(
    torch_non_c_binding_in_graph_functions_npu)  # noqa: E402


class NPUWorker(WorkerBase):

    def __init__(
            self,
            vllm_config: VllmConfig,
            local_rank: int,
            rank: int,
            distributed_init_method: str,
            is_driver_worker: bool = False,
            # Additional parameters for compatibility with vllm
            **kwargs):
        """Initialize the worker for Ascend."""
        # register patch for vllm
        from vllm_ascend.utils import adapt_patch
        adapt_patch()
        # Register ops when worker init.
        from vllm_ascend import ops
        ops.register_dummy_fusion_op()
        _register_atb_extensions()
        register_ascend_customop(vllm_config)
        # init ascend config and soc version
        init_ascend_config(vllm_config)
        init_ascend_soc_version()
        if get_ascend_config().use_sfa:
            # Direct import instead of using try_register_lib to ensure proper error handling when
            # custom_ops is necessary but not available (e.g., in DeepSeek v3.2 deployments)
            # yapf: disable
            import custom_ops  # type: ignore # noqa

            # yapf: enable
            logger.info(
                "custom_ops module loaded successfully. Custom operators like "
                "torch.ops.custom.npu_sparse_flash_attention are now available."
            )

        super().__init__(vllm_config=vllm_config,
                         local_rank=local_rank,
                         rank=rank,
                         distributed_init_method=distributed_init_method,
                         is_driver_worker=is_driver_worker)

        # Try to import mindie_turbo to accelerate vLLM inference.
        try_register_lib(
            "mindie_turbo",
            "MindIE Turbo is installed. vLLM inference will be accelerated with MindIE Turbo."
        )
        if self.cache_config.cache_dtype == "auto":
            self.cache_dtype = self.model_config.dtype
        else:
            self.cache_dtype = STR_DTYPE_TO_TORCH_DTYPE[
                self.cache_config.cache_dtype]

        if self.model_config.trust_remote_code:
            # note: lazy import to avoid importing torch before initializing
            from vllm.utils import init_cached_hf_modules
            init_cached_hf_modules()

        self.profiler = self._init_profiler()
        if sleep_mode_enabled():
            # Buffers saved before sleep
            self._sleep_saved_buffers: dict[str, torch.Tensor] = {}

        # FixMe: this is a patch to fix the issue cause by https://github.com/vllm-project/vllm/commit/de94289a98d7ec52a5ef02719e01a1db8b505170
        from vllm.model_executor.layers.linear import \
            WEIGHT_LOADER_V2_SUPPORTED
        if "UnquantizedLinearMethod" in WEIGHT_LOADER_V2_SUPPORTED:
            WEIGHT_LOADER_V2_SUPPORTED.remove("UnquantizedLinearMethod")

        # After a rank exits the active rollout group, keep it fully detached
        # from elastic DP/EP/MC2 rebuilds until the next global restore.
        self.elastic_parallel_detached = False
        self._lossless_shrink_payload: dict[int, dict] = {}
        self._lossless_preloaded_cpu_import_weights: dict[int, dict[int, tuple[
            torch.Tensor, torch.Tensor]]] = {}
        self._lossless_preloaded_direct_import_slots: dict[int, dict[int, int]] = {}
        self._elastic_current_active_ranks: list[int] | None = None
        self._post_shrink_moe_dispatch_warmed_active_signatures: set[tuple[
            int, ...]] = set()
        original_parallel_config = self.vllm_config.parallel_config
        self._elastic_original_dp_size = int(
            original_parallel_config.data_parallel_size)
        self._elastic_original_pp_size = int(
            original_parallel_config.pipeline_parallel_size)
        self._elastic_original_tp_size = int(
            original_parallel_config.tensor_parallel_size)
        self._post_kv_ep_collectives_warmed_up = False

    def sleep(self, level: int = 1) -> None:
        if not sleep_mode_enabled():
            raise ValueError(
                "Sleep mode is not enabled. Please compile vllm-ascend with COMPILE_CUSTOM_KERNELS=1."
            )
        free_bytes_before_sleep = NPUPlatform.mem_get_info()[0]
        # Save the buffers before level 2 sleep
        if level == 2:
            model = self.model_runner.model
            self._sleep_saved_buffers = {
                name: buffer.cpu().clone()
                for name, buffer in model.named_buffers()
            }
        allocator = CaMemAllocator.get_instance()
        allocator.sleep(offload_tags=("weights", ) if level == 1 else tuple())
        free_bytes_after_sleep, total = NPUPlatform.mem_get_info()
        freed_bytes = free_bytes_after_sleep - free_bytes_before_sleep
        used_bytes = total - free_bytes_after_sleep
        assert freed_bytes >= 0, "Memory usage increased after sleeping."
        logger.info(
            "Sleep mode freed %.2f GiB memory, "
            "%.2f GiB memory is still in use.", freed_bytes / GiB_bytes,
            used_bytes / GiB_bytes)

    def wake_up(self, tags: Optional[list[str]] = None) -> None:
        if not sleep_mode_enabled():
            raise ValueError(
                "Sleep mode is not enabled. Please compile vllm-ascend with COMPILE_CUSTOM_KERNELS=1."
            )
        allocator = CaMemAllocator.get_instance()
        allocator.wake_up(tags=tags)

        # Restore the buffers after level 2 sleep
        if len(self._sleep_saved_buffers):
            model = self.model_runner.model
            for name, buffer in model.named_buffers():
                if name in self._sleep_saved_buffers:
                    buffer.data.copy_(self._sleep_saved_buffers[name].data)
            self._sleep_saved_buffers = {}

    def initialize_cache(self, num_gpu_blocks: int,
                         num_cpu_blocks: int) -> None:
        self.cache_config.num_gpu_blocks = num_gpu_blocks
        self.cache_config.num_cpu_blocks = num_cpu_blocks

    def _init_device(self):
        device = torch.device(f"npu:{self.local_rank}")
        NPUPlatform.set_device(device)
        NPUPlatform.empty_cache()
        self.init_npu_memory = NPUPlatform.mem_get_info()[0]
        # Initialize the distributed environment.
        self._init_worker_distributed_environment()
        # Set random seed.
        NPUPlatform.seed_everything(self.model_config.seed)
        return device

    def init_device(self):
        with set_current_vllm_config(self.vllm_config):
            device = self._init_device()
            # Init ModelRunner here, so that we have access to self.device.
            self.model_runner = NPUModelRunner(self.vllm_config, device)

    def determine_available_memory(self) -> int:
        # Profile the memory usage of the model and get the maximum number of
        # cache blocks that can be allocated with the remaining free memory.
        NPUPlatform.clear_npu_memory()
        NPUPlatform.synchronize()

        # Execute a forward pass with dummy inputs to profile the memory usage
        # of the model.
        _, total_npu_memory = NPUPlatform.mem_get_info()
        self.model_runner.profile_run()
        NPUPlatform.synchronize()

        # Calculate the number of blocks that can be allocated with the
        # profiled peak memory.
        free_npu_memory, _ = NPUPlatform.mem_get_info()
        # NOTE(woosuk): Here we assume that the other processes using the same
        # GPU did not change their memory usage during the profiling.
        if self.init_npu_memory <= free_npu_memory:
            logger.warning(
                "Memory profiling saw higher free NPU memory after profile run "
                "on rank %s: initial_free=%s current_free=%s. Continuing "
                "because this usually means delayed cleanup or another process "
                "released memory during initialization.",
                self.rank, self.init_npu_memory, free_npu_memory)

        # Get the peak memory allocation recorded by torch
        peak_memory = torch_npu.npu.memory_stats()["allocated_bytes.all.peak"]
        # TODO: don`t need impl this func after empty_cache in
        # Worker.determine_num_available_blocks() unified`
        NPUPlatform.empty_cache()
        torch_allocated_bytes = torch_npu.npu.memory_stats(
        )["allocated_bytes.all.current"]
        total_allocated_bytes = torch_npu.npu.mem_get_info(
        )[1] - torch_npu.npu.mem_get_info()[0]
        non_torch_allocations = total_allocated_bytes - torch_allocated_bytes
        if non_torch_allocations > 0:
            peak_memory += non_torch_allocations
        available_kv_cache_memory = int(
            total_npu_memory * self.cache_config.gpu_memory_utilization -
            peak_memory)
        if self._has_effective_followup_elastic_shrink():
            shrink_headroom_bytes = (
                self._estimate_zero_redundancy_shrink_headroom_bytes())
            available_kv_cache_memory = max(
                available_kv_cache_memory - shrink_headroom_bytes, 0)
            logger.info(
                "Applying lossless zero-redundancy shrink headroom: %s bytes",
                shrink_headroom_bytes)
            floor_prealloc_headroom_bytes = (
                self._estimate_floor_prealloc_headroom_bytes())
            if floor_prealloc_headroom_bytes > 0:
                available_kv_cache_memory = max(
                    available_kv_cache_memory - floor_prealloc_headroom_bytes,
                    0)
                logger.info(
                    "Applying elastic floor prealloc headroom: %s bytes",
                    floor_prealloc_headroom_bytes)
        extra_elastic_safety_headroom_bytes = (
            self._estimate_extra_elastic_safety_headroom_bytes())
        if extra_elastic_safety_headroom_bytes > 0:
            available_kv_cache_memory = max(
                available_kv_cache_memory
                - extra_elastic_safety_headroom_bytes, 0)
            logger.info(
                "Applying extra elastic safety headroom: %s bytes",
                extra_elastic_safety_headroom_bytes)
        post_shrink_moe_dispatch_headroom_bytes = (
            self._estimate_post_shrink_moe_dispatch_headroom_bytes())
        if post_shrink_moe_dispatch_headroom_bytes > 0:
            available_kv_cache_memory = max(
                available_kv_cache_memory
                - post_shrink_moe_dispatch_headroom_bytes, 0)
            logger.info(
                "Applying post-shrink MoE dispatch headroom: %s bytes",
                post_shrink_moe_dispatch_headroom_bytes)
        post_shrink_prefill_alltoall_headroom_bytes = (
            self._estimate_post_shrink_prefill_alltoall_headroom_bytes())
        if post_shrink_prefill_alltoall_headroom_bytes > 0:
            available_kv_cache_memory = max(
                available_kv_cache_memory
                - post_shrink_prefill_alltoall_headroom_bytes, 0)
            logger.info(
                "Applying post-shrink prefill AllToAll headroom: %s bytes",
                post_shrink_prefill_alltoall_headroom_bytes)
        dp_collective_headroom_bytes = (
            self._estimate_post_restore_dp_collective_headroom_bytes())
        if dp_collective_headroom_bytes > 0:
            available_kv_cache_memory = max(
                available_kv_cache_memory - dp_collective_headroom_bytes, 0)
            logger.info(
                "Applying post-restore DP collective headroom: %s bytes",
                dp_collective_headroom_bytes)
        ep_collective_headroom_bytes = (
            self._estimate_post_restore_ep_collective_headroom_bytes())
        if ep_collective_headroom_bytes > 0:
            available_kv_cache_memory = max(
                available_kv_cache_memory - ep_collective_headroom_bytes, 0)
            logger.info(
                "Applying post-restore EP collective headroom: %s bytes",
                ep_collective_headroom_bytes)
        moe_dispatch_headroom_bytes = (
            self._estimate_post_restore_moe_dispatch_headroom_bytes())
        if moe_dispatch_headroom_bytes > 0:
            available_kv_cache_memory = max(
                available_kv_cache_memory - moe_dispatch_headroom_bytes, 0)
            logger.info(
                "Applying post-restore MoE dispatch headroom: %s bytes",
                moe_dispatch_headroom_bytes)
        first_prefill_headroom_bytes = (
            self._estimate_first_live_prefill_headroom_bytes())
        if first_prefill_headroom_bytes > 0:
            available_kv_cache_memory = max(
                available_kv_cache_memory - first_prefill_headroom_bytes, 0)
            logger.info(
                "Applying first-live-prefill activation headroom: %s bytes",
                first_prefill_headroom_bytes)
        available_kv_cache_memory = int(max(available_kv_cache_memory, 0))
        logger.info(
            f"Available memory: {available_kv_cache_memory}, total memory: {total_npu_memory}"
        )
        return available_kv_cache_memory

    def _has_effective_followup_elastic_shrink(self) -> bool:
        from vllm_ascend.ops.fused_moe import AscendFusedMoE

        if not envs_ascend.VLLM_ASCEND_ENABLE_ELASTIC_PARALLEL_SHRINK:
            return False

        model_runner = getattr(self, "model_runner", None)
        model = getattr(model_runner, "model", None) if model_runner else None
        if model is None:
            return False

        for module in model.modules():
            if not isinstance(module, AscendFusedMoE):
                continue
            if getattr(module, "elastic_moe_mode", "lossy") != "lossless":
                continue
            if getattr(module, "global_redundant_expert_num", 0) > 0:
                continue
            shrink_enabled = getattr(module, "_is_followup_shrink_enabled",
                                     None)
            if callable(shrink_enabled) and shrink_enabled():
                return True
        return False

    def _has_effective_post_restore_collective_headroom_need(self) -> bool:
        from vllm_ascend.ops.fused_moe import AscendFusedMoE

        if not envs_ascend.VLLM_ASCEND_ENABLE_ELASTIC_PARALLEL_SHRINK:
            return False

        model_runner = getattr(self, "model_runner", None)
        model = getattr(model_runner, "model", None) if model_runner else None
        if model is None:
            return False

        for module in model.modules():
            if not isinstance(module, AscendFusedMoE):
                continue
            if getattr(module, "elastic_moe_mode", "lossy") != "lossless":
                continue
            shrink_enabled = getattr(module, "_is_followup_shrink_enabled",
                                     None)
            if callable(shrink_enabled) and shrink_enabled():
                return True
        return False

    def _has_effective_post_restore_moe_dispatch_headroom_need(self) -> bool:
        return self._has_effective_post_restore_collective_headroom_need()

    def _estimate_zero_redundancy_shrink_headroom_bytes(self) -> int:
        from vllm_ascend.ops.fused_moe import AscendFusedMoE

        model_runner = getattr(self, "model_runner", None)
        model = getattr(model_runner, "model", None) if model_runner else None
        if model is None:
            return 0

        estimated_bytes = 0
        saw_zero_redundancy_module = False
        preallocated_loaded_layers = 0
        for module in model.modules():
            if not isinstance(module, AscendFusedMoE):
                continue
            if getattr(module, "elastic_moe_mode", "lossy") != "lossless":
                continue
            if getattr(module, "global_redundant_expert_num", 0) > 0:
                continue
            saw_zero_redundancy_module = True
            if getattr(module, "lossless_zero_redundancy_preallocated_loaded",
                       False):
                preallocated_loaded_layers += 1

            logical_num_experts = int(
                getattr(module, "elastic_original_num_experts",
                        module.moe_config.num_experts))
            current_ep_size = int(getattr(module.moe_parallel_config, "ep_size",
                                          1))
            next_ep_size = current_ep_size // 2
            if (logical_num_experts <= 0 or current_ep_size <= 1
                    or next_ep_size <= 0
                    or logical_num_experts % next_ep_size != 0):
                continue

            current_local_num_experts = max(
                int(getattr(module, "active_local_num_experts", 0)),
                int(getattr(module, "runtime_weight_capacity", 0)),
                int(module.w13_weight.shape[0]),
            )
            target_local_num_experts = logical_num_experts // next_ep_size
            additional_experts = max(target_local_num_experts -
                                     current_local_num_experts, 0)
            if additional_experts <= 0 or current_local_num_experts <= 0:
                continue

            if int(module.w13_weight.shape[0]) > 0:
                w13_sample = module.w13_weight[0]
                w2_sample = module.w2_weight[0]
            elif getattr(module, "runtime_w13_buffer", None) is not None:
                w13_sample = module.runtime_w13_buffer[0]
                w2_sample = module.runtime_w2_buffer[0]
            elif getattr(module, "lossless_cpu_w13_weight", None) is not None:
                w13_sample = module.lossless_cpu_w13_weight[0]
                w2_sample = module.lossless_cpu_w2_weight[0]
            else:
                continue
            per_expert_bytes = (
                w13_sample.numel() * w13_sample.element_size()
                + w2_sample.numel() * w2_sample.element_size())
            estimated_bytes += additional_experts * per_expert_bytes

        if not saw_zero_redundancy_module:
            return 0

        if estimated_bytes <= 0:
            safety_margin_bytes = 3072 * 1024 * 1024
            pass  # debug log removed
            return safety_margin_bytes

        safety_margin_bytes = 4096 * 1024 * 1024
        total_headroom = estimated_bytes + safety_margin_bytes
        logger.info(
            "Estimated lossless zero-redundancy 16->8 shrink bytes=%s safety_margin=%s total=%s preallocated_loaded_layers=%s",
            estimated_bytes, safety_margin_bytes, total_headroom,
            preallocated_loaded_layers)
        return total_headroom

    def _estimate_floor_prealloc_headroom_bytes(self) -> int:
        from vllm_ascend.ops.fused_moe import AscendFusedMoE

        model_runner = getattr(self, "model_runner", None)
        model = getattr(model_runner, "model", None) if model_runner else None
        if model is None:
            return 0

        estimated_bytes = 0
        saw_extra_floor_prealloc = False
        for module in model.modules():
            if not isinstance(module, AscendFusedMoE):
                continue
            if getattr(module, "elastic_moe_mode", "lossy") != "lossless":
                continue
            if getattr(module, "global_redundant_expert_num", 0) > 0:
                continue
            if not getattr(module, "lossless_zero_redundancy_preallocated_loaded",
                           False):
                continue
            if hasattr(module, "_is_hybrid_cpu_swap_enabled") and \
                    module._is_hybrid_cpu_swap_enabled():
                continue

            configured_floor = getattr(
                module, "_get_configured_elastic_min_compute_group_size",
                lambda: None)()
            if configured_floor is None:
                continue
            default_floor = getattr(module, "_get_default_single_shrink_ep_floor",
                                    lambda: configured_floor)()
            if configured_floor >= default_floor:
                continue

            get_reserved_slots = getattr(
                module, "_get_reserved_local_expert_slots_for_floor", None)
            if get_reserved_slots is None:
                continue
            configured_capacity = int(get_reserved_slots(configured_floor))
            default_capacity = int(get_reserved_slots(default_floor))
            additional_experts = max(configured_capacity - default_capacity, 0)
            if additional_experts <= 0:
                continue

            if int(module.w13_weight.shape[0]) <= 0:
                continue
            w13_sample = module.w13_weight[0]
            w2_sample = module.w2_weight[0]
            per_expert_bytes = (
                w13_sample.numel() * w13_sample.element_size()
                + w2_sample.numel() * w2_sample.element_size())
            estimated_bytes += additional_experts * per_expert_bytes
            saw_extra_floor_prealloc = True

        if not saw_extra_floor_prealloc or estimated_bytes <= 0:
            return 0

        safety_margin_bytes = int(
            os.getenv("VLLM_ASCEND_FLOOR_PREALLOC_HEADROOM_SAFETY_BYTES",
                      str(1024 * 1024 * 1024)))
        # The extra floor-preallocated expert weights and runtime slots are
        # already materialized before memory profiling, so they are reflected
        # in peak_memory. Only keep a small residual margin for allocator
        # fragmentation and initialization jitter instead of subtracting the
        # static bytes again from the KV-cache budget.
        logger.info(
            "Estimated elastic floor prealloc bytes already accounted in peak_memory: static_bytes=%s safety_margin=%s total=%s",
            estimated_bytes, safety_margin_bytes, safety_margin_bytes)
        return safety_margin_bytes

    def _estimate_post_restore_dp_collective_headroom_bytes(self) -> int:
        if not envs_ascend.VLLM_ASCEND_ENABLE_ELASTIC_PARALLEL_SHRINK:
            return 0
        if not self._has_effective_post_restore_collective_headroom_need():
            return 0
        if int(getattr(self.parallel_config, "data_parallel_size", 1)) <= 1:
            return 0
        # The first full-world DP metadata all_reduce after restore can cause
        # HCCL to materialize ~1.6 GiB of workspace. Reserve a little extra so
        # KV-cache sizing does not consume that headroom.
        return int(
            os.getenv("VLLM_ASCEND_POST_RESTORE_DP_HEADROOM_BYTES",
                      str(2 * 1024 * 1024 * 1024)))

    def _estimate_post_restore_ep_collective_headroom_bytes(self) -> int:
        if not envs_ascend.VLLM_ASCEND_ENABLE_ELASTIC_PARALLEL_SHRINK:
            return 0
        if not self._has_effective_post_restore_collective_headroom_need():
            return 0
        # After restoring back to the full 16-rank EP group, the first MoE
        # token-dispatch all_gather can materialize ~1.6 GiB of HCCL workspace.
        # Reserve a bit more so KV-cache sizing doesn't consume that space.
        return int(
            os.getenv("VLLM_ASCEND_POST_RESTORE_EP_HEADROOM_BYTES",
                      str(2 * 1024 * 1024 * 1024)))

    def _estimate_post_restore_moe_dispatch_headroom_bytes(self) -> int:
        if not envs_ascend.VLLM_ASCEND_ENABLE_ELASTIC_PARALLEL_SHRINK:
            return 0
        if not self._has_effective_post_restore_moe_dispatch_headroom_need():
            return 0
        # After restore, the first full-world MoE dispatch kernel
        # (aclnnMoeDistributeDispatchV2) can still request ~1.6 GiB of device
        # memory even when the EP all_gather headroom has been reserved.
        # Keep this separate from the EP collective budget because it is a
        # distinct post-restore cost and often needs to be tuned independently.
        return int(
            os.getenv("VLLM_ASCEND_POST_RESTORE_MOE_DISPATCH_HEADROOM_BYTES",
                      str(2 * 1024 * 1024 * 1024)))

    def _estimate_first_live_prefill_headroom_bytes(self) -> int:
        """Reserve residual room for the first real prefill forward.

        `profile_run()` exercises synthetic prefill shapes, but the first live
        batch can still need slightly different temporary activation/workspace
        allocations (for example around attention norm and allocator layout).
        Keep a small extra budget so KV-cache sizing does not consume that last
        margin, especially for lossless elastic modes with preloaded redundant
        experts.
        """
        from vllm_ascend.ops.fused_moe import AscendFusedMoE

        if not envs_ascend.VLLM_ASCEND_ENABLE_ELASTIC_PARALLEL_SHRINK:
            return 0

        model_runner = getattr(self, "model_runner", None)
        model = getattr(model_runner, "model", None) if model_runner else None
        if model is None:
            return 0

        base_headroom_bytes = int(
            os.getenv("VLLM_ASCEND_FIRST_LIVE_PREFILL_HEADROOM_BYTES",
                      str(1024 * 1024 * 1024)))
        low_floor_headroom_bytes = int(
            os.getenv(
                "VLLM_ASCEND_FIRST_LIVE_PREFILL_LOW_FLOOR_HEADROOM_BYTES",
                str(2 * 1024 * 1024 * 1024)))

        for module in model.modules():
            if not isinstance(module, AscendFusedMoE):
                continue
            if getattr(module, "elastic_moe_mode", "lossy") != "lossless":
                continue
            if not getattr(module, "_is_followup_shrink_enabled", lambda: False)():
                continue
            configured_floor = getattr(
                module, "_get_configured_elastic_min_compute_group_size",
                lambda: None)()
            if configured_floor is not None and int(configured_floor) <= 4:
                return max(base_headroom_bytes, low_floor_headroom_bytes)
            return base_headroom_bytes
        return 0

    def _estimate_extra_elastic_safety_headroom_bytes(self) -> int:
        from vllm_ascend.ops.fused_moe import AscendFusedMoE

        if not envs_ascend.VLLM_ASCEND_ENABLE_ELASTIC_PARALLEL_SHRINK:
            return 0
        if not self._has_effective_post_restore_collective_headroom_need():
            return 0

        model_runner = getattr(self, "model_runner", None)
        model = getattr(model_runner, "model", None) if model_runner else None
        if model is None:
            return 0

        saw_floor_four_or_lower = False
        for module in model.modules():
            if not isinstance(module, AscendFusedMoE):
                continue
            if getattr(module, "elastic_moe_mode", "lossy") != "lossless":
                continue
            configured_floor = getattr(
                module, "_get_configured_elastic_min_compute_group_size",
                lambda: None)()
            if configured_floor is None or configured_floor > 4:
                continue
            saw_floor_four_or_lower = True
            break

        if not saw_floor_four_or_lower:
            return 0

        # Keep extra room for <=4-rank elastic paths where a resumed request can
        # still hit a heavy post-shrink MoE/collective workspace peak even
        # though static floor preallocated expert weights were already reflected
        # in peak_memory. This applies to both zero-redundancy and redundant
        # lossless modes.
        return int(float(
            os.getenv("VLLM_ASCEND_EXTRA_ELASTIC_SAFETY_HEADROOM_BYTES",
                    str(2.2 * 1024 * 1024 * 1024))
        ))

    def _estimate_post_shrink_moe_dispatch_headroom_bytes(self) -> int:
        from vllm_ascend.ops.fused_moe import AscendFusedMoE

        if not envs_ascend.VLLM_ASCEND_ENABLE_ELASTIC_PARALLEL_SHRINK:
            return 0
        if not self._has_effective_post_restore_collective_headroom_need():
            return 0

        model_runner = getattr(self, "model_runner", None)
        model = getattr(model_runner, "model", None) if model_runner else None
        if model is None:
            return 0

        saw_low_floor_module = False
        saw_elastic_lossless_module = False
        min_configured_floor = None
        for module in model.modules():
            if not isinstance(module, AscendFusedMoE):
                continue
            if getattr(module, "elastic_moe_mode", "lossy") != "lossless":
                continue
            if not getattr(module, "_is_followup_shrink_enabled",
                           lambda: False)():
                continue
            saw_elastic_lossless_module = True
            configured_floor = getattr(
                module, "_get_configured_elastic_min_compute_group_size",
                lambda: None)()
            if configured_floor is None:
                continue
            configured_floor = int(configured_floor)
            if min_configured_floor is None:
                min_configured_floor = configured_floor
            else:
                min_configured_floor = min(min_configured_floor,
                                           configured_floor)
            if configured_floor <= 4:
                saw_low_floor_module = True

        if not saw_elastic_lossless_module:
            return 0

        if saw_low_floor_module:
            # Low-floor follow-up shrink keeps far fewer active ranks, but the
            # first live MC2 MoE dispatch after rebuild can still request a
            # large HCCL workspace. Repeated warmup is now skipped after the
            # first successful pass, so the remaining risk is the real decode
            # dispatch itself. Keep a more conservative default budget here.
            default_headroom_bytes = 4 * 1024 * 1024 * 1024
            if (min_configured_floor is not None
                    and min_configured_floor <= 2):
                default_headroom_bytes = 6 * 1024 * 1024 * 1024
            return int(
                os.getenv(
                    "VLLM_ASCEND_POST_SHRINK_MOE_DISPATCH_LOW_FLOOR_HEADROOM_BYTES",
                    str(default_headroom_bytes)))

        return int(
            os.getenv("VLLM_ASCEND_POST_SHRINK_MOE_DISPATCH_HEADROOM_BYTES",
                      str(512 * 1024 * 1024)))

    def _estimate_post_shrink_prefill_alltoall_headroom_bytes(self) -> int:
        from vllm_ascend.ops.fused_moe import AscendFusedMoE

        if not envs_ascend.VLLM_ASCEND_ENABLE_ELASTIC_PARALLEL_SHRINK:
            return 0
        if not self._has_effective_post_restore_collective_headroom_need():
            return 0

        model_runner = getattr(self, "model_runner", None)
        model = getattr(model_runner, "model", None) if model_runner else None
        if model is None:
            return 0

        saw_low_floor_lossless_module = False
        min_configured_floor = None
        for module in model.modules():
            if not isinstance(module, AscendFusedMoE):
                continue
            if getattr(module, "elastic_moe_mode", "lossy") != "lossless":
                continue
            if not getattr(module, "_is_followup_shrink_enabled",
                           lambda: False)():
                continue
            if hasattr(module, "_is_hybrid_cpu_swap_enabled") and \
                    module._is_hybrid_cpu_swap_enabled():
                continue
            configured_floor = getattr(
                module, "_get_configured_elastic_min_compute_group_size",
                lambda: None)()
            if configured_floor is None:
                continue
            configured_floor = int(configured_floor)
            if min_configured_floor is None:
                min_configured_floor = configured_floor
            else:
                min_configured_floor = min(min_configured_floor,
                                           configured_floor)
            if configured_floor <= 4:
                saw_low_floor_lossless_module = True

        if not saw_low_floor_lossless_module:
            return 0

        # After the follow-up 8->4 shrink in redundant/zero-redundancy
        # lossless mode, resumed long prefills can immediately re-enter the
        # shrunken EP group. That flips MoE comm back to AllToAll at ep_size=4
        # with synced_input_tokens in the ~10k range, and the first
        # HcclAllGather/token-dispatch workspace peak is materially larger than
        # the decode-only MC2 warmup we already reserve for.
        default_headroom_bytes = 3 * 1024 * 1024 * 1024
        if (min_configured_floor is not None
                and min_configured_floor <= 2):
            default_headroom_bytes = 4 * 1024 * 1024 * 1024
        return int(
            os.getenv(
                "VLLM_ASCEND_POST_SHRINK_PREFILL_ALLTOALL_HEADROOM_BYTES",
                str(default_headroom_bytes)))

    def execute_model(
        self,
        scheduler_output: "SchedulerOutput",
    ) -> Optional[Union[ModelRunnerOutput, AsyncModelRunnerOutput]]:
        # enable msMonitor to monitor the performance of vllm-ascend
        if envs_ascend.MSMONITOR_USE_DAEMON:
            dp.step()

        intermediate_tensors = None
        forward_pass = scheduler_output.total_num_scheduled_tokens > 0
        if forward_pass and not get_pp_group().is_first_rank:
            intermediate_tensors = IntermediateTensors(
                get_pp_group().recv_tensor_dict(
                    all_gather_group=get_tp_group()))

        output = self.model_runner.execute_model(scheduler_output,
                                                 intermediate_tensors)
        if isinstance(output, (ModelRunnerOutput, AsyncModelRunnerOutput)):
            return output

        assert isinstance(output, IntermediateTensors)
        parallel_config = self.vllm_config.parallel_config
        assert parallel_config.distributed_executor_backend != (
            "external_launcher") and not get_pp_group().is_last_rank

        get_pp_group().send_tensor_dict(output.tensors,
                                        all_gather_group=get_tp_group())

        kv_connector_output = output.kv_connector_output
        if not kv_connector_output:
            return None

        # In case of PP with kv transfer, we need to pass through the
        # kv_connector_output
        if (not kv_connector_output.finished_sending
                and not kv_connector_output.finished_recving):
            return EMPTY_MODEL_RUNNER_OUTPUT
        output = copy.copy(EMPTY_MODEL_RUNNER_OUTPUT)
        output.kv_connector_output = kv_connector_output
        return output

    def load_model(self) -> None:
        if self.vllm_config.model_config.enable_sleep_mode:
            allocator = CaMemAllocator.get_instance()
            assert allocator.get_current_usage() == 0, (
                "Sleep mode can only be "
                "used for one instance per process.")
            context = allocator.use_memory_pool(tag="weights")
        else:
            from contextlib import nullcontext
            context = nullcontext()  # type: ignore
        with set_current_vllm_config(self.vllm_config), context:
            self.model_runner.load_model()

    def compile_or_warm_up_model(self) -> None:
        # Note: need to adapt for graph mode.
        self.model_runner.eplb_warmup()
        warmup_sizes = (self.vllm_config.compilation_config.compile_sizes
                        or []).copy()
        if not self.model_config.enforce_eager:
            warmup_sizes = [
                x for x in warmup_sizes if x not in
                self.vllm_config.compilation_config.cudagraph_capture_sizes
            ]
        for size in sorted(warmup_sizes, reverse=True):
            logger.info("Compile and warming up model for size %d", size)
            self.model_runner._dummy_run(size)
        if not self.model_config.enforce_eager:
            self.model_runner.capture_model()
        # Call ATB matmul to warm up; otherwise, the first operation (ReshapeAndCache)
        # may cause performance degradation at runtime.
        self._warm_up_atb()
        # Reset the seed to ensure that the random state is not affected by
        # the model initialization and profiling.
        NPUPlatform.seed_everything(self.model_config.seed)

    def _warm_up_atb(self):
        x = torch.rand((2, 4), dtype=torch.float16).npu()
        weight = torch.rand((2, 4), dtype=torch.float16).npu()
        c = torch.rand((4, 4), dtype=torch.float32).npu()
        torch_npu._npu_matmul_add_fp32(x, weight, c)

    def get_model(self) -> nn.Module:
        return self.model_runner.get_model()

    def get_kv_cache_spec(self) -> dict[str, KVCacheSpec]:
        return self.model_runner.get_kv_cache_spec()

    def initialize_from_config(self, kv_cache_config: KVCacheConfig) -> None:
        """Allocate NPU KV cache with the specified kv_cache_config."""
        if self.vllm_config.model_config.enable_sleep_mode:
            allocator = CaMemAllocator.get_instance()
            context = allocator.use_memory_pool(tag="kv_cache")
        else:
            from contextlib import nullcontext
            context = nullcontext()  # type: ignore
        with set_current_vllm_config(self.vllm_config), context:
            self.model_runner.initialize_kv_cache(kv_cache_config)
            enable_post_kv_ep_warmup = os.environ.get(
                "VLLM_ASCEND_ENABLE_POST_KV_EP_WARMUP", "0").lower() in (
                    "1", "true", "yes", "on")
            if (enable_post_kv_ep_warmup
                    and not self._post_kv_ep_collectives_warmed_up):
                self._warm_up_post_kv_ep_collectives()
                self._post_kv_ep_collectives_warmed_up = True

    def _warm_up_post_kv_ep_collectives(self) -> None:
        """Warm up EP HCCL collectives after KV cache allocation.

        Profile-time warmup happens before KV cache is materialized, so it can
        miss communicator/workspace costs that only show up under the
        steady-state post-KV-cache memory watermark. Exercise the same EP
        collectives used by the AllToAll MoE path here so first-request failures
        surface during init instead of several minutes later.
        """
        if not self.parallel_config.enable_expert_parallel:
            return

        import vllm.distributed.parallel_state as vllm_ps
        ep_group = vllm_ps.get_ep_group()
        if ep_group.world_size <= 1:
            return

        metadata_width = int(
            getattr(self.model_config.hf_config, "n_routed_experts",
                    ep_group.world_size))
        metadata_tensor = torch.zeros(metadata_width,
                                      dtype=torch.int64,
                                      device="npu")
        gathered_metadata = torch.empty(metadata_width * ep_group.world_size,
                                        dtype=torch.int64,
                                        device="npu")
        dist.all_gather_into_tensor(gathered_metadata,
                                    metadata_tensor,
                                    group=ep_group.device_group)

        all_to_all_input = torch.zeros(ep_group.world_size,
                                       dtype=torch.int64,
                                       device="npu")
        all_to_all_output = torch.empty_like(all_to_all_input)
        dist.all_to_all_single(all_to_all_output,
                               all_to_all_input,
                               group=ep_group.device_group)
        torch.npu.synchronize()

        if self.rank == 0:
            logger.info(
                "Post-KV EP collective warmup done: ep_world_size=%s metadata_width=%s",
                ep_group.world_size, metadata_width)

    def profile(self, is_start: bool = True):
        if self.profiler is None:
            raise RuntimeError("Profiler is not enabled.")
        if is_start:
            self.profiler.start()
        else:
            self.profiler.stop()

    def add_lora(self, lora_request: LoRARequest) -> bool:
        return self.model_runner.add_lora(lora_request)

    def remove_lora(self, lora_id: int) -> bool:
        return self.model_runner.remove_lora(lora_id)

    def list_loras(self) -> set[int]:
        return self.model_runner.list_loras()

    def pin_lora(self, lora_id: int) -> bool:
        return self.model_runner.pin_lora(lora_id)

    def execute_dummy_batch(self) -> None:
        self.model_runner._dummy_run(1)

    def _init_worker_distributed_environment(self) -> None:
        """Initialize the distributed environment."""
        with set_current_vllm_config(self.vllm_config):
            init_distributed_environment(self.parallel_config.world_size,
                                         self.rank,
                                         self.distributed_init_method,
                                         self.local_rank, "hccl")
            ensure_model_parallel_initialized(
                self.parallel_config.tensor_parallel_size,
                self.parallel_config.pipeline_parallel_size)
            init_ascend_model_parallel(self.parallel_config)
            ensure_kv_transfer_initialized(self.vllm_config)

    def _init_profiler(self):
        # Torch profiler. Enabled and configured through env vars:
        # VLLM_TORCH_PROFILER_DIR=/path/to/save/trace
        if envs_vllm.VLLM_TORCH_PROFILER_DIR:
            if envs_ascend.MSMONITOR_USE_DAEMON:
                raise RuntimeError(
                    "MSMONITOR_USE_DAEMON and VLLM_TORCH_PROFILER_DIR cannot be both set at the same time."
                )
            torch_profiler_trace_dir = envs_vllm.VLLM_TORCH_PROFILER_DIR
            logger.info("Profiling enabled. Traces will be saved to: %s",
                        torch_profiler_trace_dir)

            experimental_config = torch_npu.profiler._ExperimentalConfig(
                export_type=torch_npu.profiler.ExportType.Text,
                profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
                msprof_tx=False,
                aic_metrics=torch_npu.profiler.AiCMetrics.AiCoreNone,
                l2_cache=False,
                op_attr=False,
                data_simplification=False,
                record_op_args=False,
                gc_detect_threshold=None,
            )

            return torch_npu.profiler.profile(
                activities=[
                    torch_npu.profiler.ProfilerActivity.CPU,
                    torch_npu.profiler.ProfilerActivity.NPU,
                ],
                with_stack=envs_vllm.VLLM_TORCH_PROFILER_WITH_STACK,
                profile_memory=envs_vllm.\
                    VLLM_TORCH_PROFILER_WITH_PROFILE_MEMORY,
                with_modules=False,
                experimental_config=experimental_config,
                on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(
                    torch_profiler_trace_dir))
        else:
            return None

    def get_supported_pooling_tasks(self):
        return self.model_runner.get_supported_pooling_tasks()

    def get_supported_tasks(self) -> "tuple[SupportedTask, ...]":
        return self.model_runner.get_supported_tasks()

    def take_draft_token_ids(self) -> Optional[DraftTokenIds]:
        return self.model_runner.take_draft_token_ids()

    def _build_original_dp_group_ranks(self, world_size: int) -> list[list[int]]:
        all_ranks = torch.arange(world_size).reshape(
            -1, self._elastic_original_dp_size,
            self._elastic_original_pp_size,
            self._elastic_original_tp_size)
        group_ranks = all_ranks.transpose(1, 3).reshape(
            -1, self._elastic_original_dp_size).unbind(0)
        return [x.tolist() for x in group_ranks]

    def _build_original_ep_group_ranks(self, world_size: int) -> list[list[int]]:
        all_ranks = torch.arange(world_size).reshape(
            -1, self._elastic_original_dp_size,
            self._elastic_original_pp_size,
            self._elastic_original_tp_size)
        group_ranks = all_ranks.transpose(1, 2).reshape(
            -1, self._elastic_original_dp_size *
            self._elastic_original_tp_size).unbind(0)
        return [x.tolist() for x in group_ranks]

    def _build_original_mc2_group_ranks(self, world_size: int) -> list[list[int]]:
        all_ranks = torch.arange(world_size).reshape(
            -1, self._elastic_original_dp_size *
            self._elastic_original_tp_size)
        group_ranks = all_ranks.unbind(0)
        return [x.tolist() for x in group_ranks]

    def _refresh_elastic_parallel_state(self,
                                        active_ranks: list[int],
                                        world_group,
                                        participate_only: bool = False) -> None:
        from vllm_ascend.ops.fused_moe import AscendFusedMoE

        current_rank = torch.distributed.get_rank()
        is_active_rank = current_rank in active_ranks
        restoring_full_world = len(active_ranks) == torch.distributed.get_world_size()
        model_runner = getattr(self, "model_runner", None)
        if model_runner is not None and is_active_rank:
            new_dp_group = get_dp_group()
            model_runner.dp_size = new_dp_group.world_size
            model_runner.dp_rank = new_dp_group.rank_in_group
            model_runner.parallel_config.data_parallel_size = \
                new_dp_group.world_size
            model_runner.parallel_config.data_parallel_rank = \
                new_dp_group.rank_in_group

        model = getattr(model_runner, "model", None) if model_runner else None
        if model is None:
            return

        active_cpu_group = get_dp_group().cpu_group if is_active_rank else None
        lossless_shrink_payload = getattr(self, "_lossless_shrink_payload", {})
        preloaded_cpu_import_weights = getattr(
            self, "_lossless_preloaded_cpu_import_weights", {})
        preloaded_direct_import_slots = getattr(
            self, "_lossless_preloaded_direct_import_slots", {})
        for module in model.modules():
            if not isinstance(module, AscendFusedMoE):
                continue

            use_lossless_mode = (
                envs_ascend.VLLM_ASCEND_ELASTIC_MOE_MODE == "lossless"
                and getattr(module, "elastic_moe_mode", "lossy") == "lossless")

            if use_lossless_mode:
                payload = lossless_shrink_payload.get(module.layer_idx)
                if payload is None:
                    if restoring_full_world:
                        current_dp_group = get_dp_group()
                        current_ep_group = get_ep_group()
                        module.set_active_expert_mask(None)
                        module.set_elastic_runtime_log2phy(None)
                        module.ep_group = current_ep_group
                        module.moe_parallel_config.dp_size = current_dp_group.world_size
                        module.moe_parallel_config.dp_rank = current_dp_group.rank_in_group
                        module.moe_parallel_config.ep_size = current_ep_group.world_size
                        module.moe_parallel_config.ep_rank = current_ep_group.rank_in_group
                        module.moe_config.moe_parallel_config = module.moe_parallel_config
                        module.moe_config.dp_group = current_dp_group
                        module.moe_config.ep_group = current_ep_group
                        module.moe_config.mc2_group = get_mc2_group()
                        original_num_experts = int(
                            module.elastic_original_num_experts)
                        module.num_experts = original_num_experts
                        module.moe_config.num_experts = original_num_experts
                        if hasattr(module,
                                   "restore_lossless_full_world_primary_layout"):
                            module.restore_lossless_full_world_primary_layout()
                        elif hasattr(module, "reset_expert_map_and_log2phy"):
                            module.reset_expert_map_and_log2phy()
                        quant_method = getattr(module, "quant_method", None)
                        if (quant_method is not None and hasattr(
                                quant_method,
                                "invalidate_lossless_runtime_state_for_reload")):
                            quant_method.invalidate_lossless_runtime_state_for_reload(
                                layer=module, reason="restore_full_world")
                        else:
                            module.runtime_w13_weight = None
                            module.runtime_w2_weight = None
                            module.runtime_w13_buffer = None
                            module.runtime_w2_buffer = None
                            module.runtime_weight_capacity = 0
                            module.lossless_runtime_activated = False
                        if not participate_only:
                            if module.layer_idx == 0:
                                logger.info(
                                    "Elastic full-world MoE layout reset: rank=%s layer=%s num_experts=%s num_local_experts=%s active_local_experts=%s ep_size=%s",
                                    self.rank, module.layer_idx,
                                    module.moe_config.num_experts,
                                    getattr(module, "local_num_experts", None),
                                    getattr(module,
                                            "active_local_num_experts", None),
                                    get_ep_group().world_size)
                            module.refresh_elastic_groups()
                        continue
                    raise RuntimeError(
                        f"Missing lossless shrink payload for layer={module.layer_idx}."
                    )
                local_cpu_import_weights = preloaded_cpu_import_weights.get(
                    module.layer_idx, {})
                if participate_only:
                    module.set_active_expert_mask(None)
                    module.set_elastic_runtime_log2phy(None)
                    module.moe_config.num_experts = module.elastic_original_num_experts
                    continue
                logical_num_experts = int(module.elastic_original_num_experts)
                # 在mode=3下，不设置active_expert_mask，因为双缓冲模式有自己的专家管理机制
                if getattr(module, "elastic_execution_mode", 0) != 3:
                    active_expert_mask = torch.ones(logical_num_experts,
                                                    dtype=torch.bool)
                    module.set_active_expert_mask(active_expert_mask.to(
                        device=module.expert_map.device))
                else:
                    module.set_active_expert_mask(None)
                assignments = payload["assignments"]
                my_rank_idx = active_ranks.index(current_rank)
                ordered_assignments = payload["ordered_assignments"]
                local_assignments = ordered_assignments[my_rank_idx]
                local_active_expert_ids = [
                    expert_id for expert_id, _ in local_assignments
                ]
                local_source_local_ids = [
                    local_id for _, local_id in local_assignments
                ]
                local_direct_import_slots = preloaded_direct_import_slots.get(
                    module.layer_idx, {})
                if local_direct_import_slots:
                    for local_slot, expert_id in enumerate(local_active_expert_ids):
                        if local_source_local_ids[local_slot] >= 0:
                            continue
                        direct_slot = local_direct_import_slots.get(expert_id)
                        if direct_slot is None:
                            continue
                        if direct_slot != local_slot:
                            raise RuntimeError(
                                f"layer={module.layer_idx}: expert_id={expert_id} "
                                f"local_slot={local_slot} direct_slot={direct_slot}")
                        local_source_local_ids[local_slot] = direct_slot
                use_hybrid_cpu_swap = False
                hybrid_enabled = (
                    hasattr(module, "_is_hybrid_cpu_swap_enabled")
                    and module._is_hybrid_cpu_swap_enabled())
                resident_capacity = 0
                if hybrid_enabled and hasattr(module,
                                              "_get_hybrid_resident_capacity"):
                    resident_capacity = int(
                        module._get_hybrid_resident_capacity())
                initial_ep_size = len(active_ranks)
                if hybrid_enabled and hasattr(module, "_get_elastic_initial_ep_size"):
                    initial_ep_size = int(module._get_elastic_initial_ep_size())
                in_followup_hybrid_stage = (
                    hybrid_enabled
                    and resident_capacity > 0
                    and len(active_ranks) < initial_ep_size)
                if in_followup_hybrid_stage:
                    # Mode=2/3 both pivot into the hybrid tail after the first
                    # shrink. Mode=2 keeps the per-layer resident-slot path;
                    # mode=3 reuses the same ownership/import activation but
                    # defers runtime expert materialization to the cross-layer
                    # double-buffer path during forward.
                    use_hybrid_cpu_swap = True
                elif (hasattr(module, "should_activate_lossless_hybrid_for_target")
                        and module.should_activate_lossless_hybrid_for_target(
                            len(local_active_expert_ids), len(active_ranks))):
                    use_hybrid_cpu_swap = True
                elif (hybrid_enabled and resident_capacity > 0
                      and len(local_active_expert_ids) == resident_capacity
                      and hasattr(module,
                                  "_can_use_lossless_loaded_prefix_views")
                      and not module._can_use_lossless_loaded_prefix_views(
                          local_source_local_ids, local_cpu_import_weights)):
                    # In mode=2, a target that exactly fills the resident slots
                    # should still reuse the fixed hybrid slots when the loaded
                    # prefix cannot represent the requested expert layout.
                    use_hybrid_cpu_swap = True
                if use_hybrid_cpu_swap:
                    module.set_lossless_hybrid_global_layout(
                        active_ranks, ordered_assignments)
                    module.activate_lossless_hybrid_local_experts(
                        local_active_expert_ids,
                        local_source_local_ids,
                        cpu_expert_weights=local_cpu_import_weights)
                    new_log2phy_cpu = (
                        module._build_lossless_hybrid_runtime_log2phy(
                            module.lossless_hybrid_rank_resident_expert_ids))
                    module._set_lossless_hybrid_runtime_num_experts()
                    if module.layer_idx == 0:
                        logger.info(
                            "Elastic lossless hybrid CPU-swap activated: rank=%s layer=%s active_ranks=%s owned_local=%s resident_capacity=%s cpu_only_local=%s runtime_num_experts=%s",
                            self.rank,
                            module.layer_idx,
                            active_ranks,
                            len(local_active_expert_ids),
                            int(module.lossless_hybrid_resident_capacity),
                            len(module.lossless_hybrid_cpu_only_expert_ids),
                            int(module.moe_config.num_experts),
                        )
                        if len(active_ranks) == 1:
                            logger.info(
                                "Elastic single-rank tail activated: rank=%s layer=%s surviving_rank=%s owned_local=%s resident_capacity=%s cpu_only_local=%s runtime_num_experts=%s ep_size=%s tail_mode=no_ep",
                                self.rank,
                                module.layer_idx,
                                active_ranks[0] if active_ranks else None,
                                len(local_active_expert_ids),
                                int(module.lossless_hybrid_resident_capacity),
                                len(module.lossless_hybrid_cpu_only_expert_ids),
                                int(module.moe_config.num_experts),
                                get_ep_group().world_size,
                            )
                        if getattr(module, "elastic_execution_mode", 0) == 3:
                            logger.info(
                                "Mode3 cross-layer buffer activation: rank=%s layer=%s stage=%s owned_local=%s primary_prefix_rows=%s cpu_only_local=%s",
                                self.rank,
                                module.layer_idx,
                                len(active_ranks),
                                len(local_active_expert_ids),
                                int(getattr(module, "lossless_hybrid_resident_capacity", 0)),
                                len(module.lossless_hybrid_cpu_only_expert_ids),
                            )
                else:
                    offload_loaded_after_activation = (
                        hasattr(module,
                                "should_offload_loaded_weights_after_lossless_activation")
                        and module
                        .should_offload_loaded_weights_after_lossless_activation())
                    module.activate_lossless_local_experts(
                        local_active_expert_ids,
                        local_source_local_ids,
                        cpu_expert_weights=local_cpu_import_weights,
                        offload_loaded_after_activation=
                        offload_loaded_after_activation)
                    new_log2phy_cpu = payload["runtime_log2phy_cpu"]
                    module.set_runtime_num_experts(logical_num_experts)
                if module.log2phy is not None and new_log2phy_cpu is not None:
                    module.set_elastic_runtime_log2phy(
                        new_log2phy_cpu.to(device=module.log2phy.device,
                                           dtype=module.log2phy.dtype))
                else:
                    module.set_elastic_runtime_log2phy(None)
                # Rebuild the token dispatcher with the post-shrink local expert
                # count before decode resumes on the new 8-rank EP group.
                module.refresh_elastic_groups()
                imported_expert_ids = [
                    expert_id for expert_id, source_local_id in zip(
                        local_active_expert_ids, local_source_local_ids)
                    if source_local_id < 0
                ]
                sample_import_stats = None
                if imported_expert_ids:
                    sample_import_id = imported_expert_ids[0]
                    sample_pair = local_cpu_import_weights.get(sample_import_id)
                    if sample_pair is not None:
                        sample_w13, sample_w2 = sample_pair
                        sample_import_stats = {
                            "expert_id": sample_import_id,
                            "w13_shape": tuple(sample_w13.shape),
                            "w2_shape": tuple(sample_w2.shape),
                            "w13_abs_mean": float(
                                sample_w13.float().abs().mean().item()),
                            "w2_abs_mean": float(
                                sample_w2.float().abs().mean().item()),
                        }
                    if use_hybrid_cpu_swap:
                        resident_capacity = int(
                            module.lossless_hybrid_resident_capacity)
                        dense_offset = my_rank_idx * resident_capacity
                        resident_expert_ids = local_active_expert_ids[
                            :resident_capacity]
                        cpu_only_expert_ids = local_active_expert_ids[
                            resident_capacity:]
                        mapping_mismatch = sum(
                            int(new_log2phy_cpu[expert_id].item()) !=
                            dense_offset + local_slot
                            for local_slot, expert_id in enumerate(
                                resident_expert_ids))
                        if int(getattr(module, "elastic_execution_mode", 0)) \
                                != 3:
                            mapping_mismatch += sum(
                                int(new_log2phy_cpu[expert_id].item()) != -1
                                for expert_id in cpu_only_expert_ids)
                    else:
                        dense_offset = sum(
                            len(rank_assignments)
                            for rank_assignments in
                            ordered_assignments[:my_rank_idx])
                        mapping_mismatch = sum(
                            int(new_log2phy_cpu[expert_id].item()) !=
                            dense_offset + local_slot
                            for local_slot, expert_id in enumerate(
                                local_active_expert_ids))
                if mapping_mismatch:
                    raise RuntimeError(
                        f"layer={module.layer_idx} rank={current_rank}: "
                        f"mismatch_count={mapping_mismatch}")
                if module.layer_idx == 0:
                    pass  # debug log removed
                continue

            if not is_active_rank:
                module.set_active_expert_mask(None)
                module.set_elastic_runtime_log2phy(None)
                module.moe_config.num_experts = module.elastic_original_num_experts
                continue
            if module.expert_map is None:
                module.set_active_expert_mask(None)
                module.set_elastic_runtime_log2phy(None)
                module.moe_config.num_experts = module.elastic_original_num_experts
                module.refresh_elastic_groups()
                continue
            else:
                if participate_only:
                    continue
                lossless_loaded_expert_map = getattr(module, "loaded_expert_map",
                                                     None)
                map_for_shrink = (lossless_loaded_expert_map
                                  if lossless_loaded_expert_map is not None else
                                  module.expert_map)
                local_expert_map_cpu = map_for_shrink.to(device="cpu",
                                                         dtype=torch.int32)
                gathered_expert_map = [
                    torch.empty_like(local_expert_map_cpu)
                    for _ in range(len(active_ranks))
                ]
                torch.distributed.all_gather(gathered_expert_map,
                                             local_expert_map_cpu,
                                             group=active_cpu_group)
                gathered_log2phy = None
                if module.log2phy is not None and lossless_loaded_expert_map is None:
                    local_log2phy_cpu = module.log2phy.to(device="cpu",
                                                          dtype=torch.int32)
                    gathered_log2phy = [
                        torch.empty_like(local_log2phy_cpu)
                        for _ in range(len(active_ranks))
                    ]
                    torch.distributed.all_gather(gathered_log2phy,
                                                 local_log2phy_cpu,
                                                 group=active_cpu_group)
                local_expert_counts = [
                    int((rank_expert_map != -1).sum().item())
                    for rank_expert_map in gathered_expert_map
                ]
                active_expert_mask = torch.zeros_like(local_expert_map_cpu,
                                                      dtype=torch.bool)
                for rank_expert_map in gathered_expert_map:
                    active_expert_mask |= (rank_expert_map != -1)
                module.set_active_expert_mask(active_expert_mask.to(
                    device=module.expert_map.device))

                if module.log2phy is not None:
                    new_log2phy_cpu = torch.zeros_like(local_expert_map_cpu)
                    active_old_phy_ids: list[int] = []
                    selected_old_phy_ids: list[int | None] = [None] * int(
                        local_expert_map_cpu.numel())
                    for expert_id in range(local_expert_map_cpu.numel()):
                        selected_rank_idx = None
                        for rank_idx, rank_expert_map in enumerate(
                                gathered_expert_map):
                            if int(rank_expert_map[expert_id].item()) >= 0:
                                selected_rank_idx = rank_idx
                                break
                        if selected_rank_idx is None:
                            continue
                        if gathered_log2phy is not None:
                            old_phy_id = int(
                                gathered_log2phy[selected_rank_idx][expert_id].
                                item())
                        else:
                            local_id = int(
                                gathered_expert_map[selected_rank_idx][expert_id].
                                item())
                            old_phy_id = sum(local_expert_counts[:selected_rank_idx]
                                             ) + local_id
                        selected_old_phy_ids[expert_id] = old_phy_id
                        active_old_phy_ids.append(old_phy_id)
                    old_phy_to_dense = {
                        old_phy_id: dense_id
                        for dense_id, old_phy_id in enumerate(
                            sorted(set(active_old_phy_ids)))
                    }
                    module.set_runtime_num_experts(len(old_phy_to_dense))
                    for expert_id, old_phy_id in enumerate(selected_old_phy_ids):
                        if old_phy_id is None:
                            continue
                        new_log2phy_cpu[expert_id] = old_phy_to_dense[old_phy_id]
                    module.set_elastic_runtime_log2phy(
                        new_log2phy_cpu.to(device=module.log2phy.device,
                                           dtype=module.log2phy.dtype))
                else:
                    module.set_elastic_runtime_log2phy(None)
                    module.set_runtime_num_experts(sum(local_expert_counts))

                module.refresh_elastic_groups()

    def _get_previous_active_ranks_for_shrink(self, world_size: int) -> list[int]:
        previous_active_ranks = getattr(self, "_elastic_current_active_ranks",
                                        None)
        if not previous_active_ranks:
            return list(range(world_size))
        return sorted(set(previous_active_ranks))

    def _get_shrink_source_group_state(self, world_group):
        previous_active_ranks = self._get_previous_active_ranks_for_shrink(
            world_group.world_size)
        dp_group = get_dp_group()
        source_cpu_group = getattr(dp_group, "cpu_group", None)
        source_device_group = getattr(dp_group, "device_group", None)
        source_ranks = previous_active_ranks

        if source_cpu_group is None or source_device_group is None:
            return (source_ranks, world_group.cpu_group, world_group.device_group)

        try:
            source_group_world_size = torch.distributed.get_world_size(
                group=source_cpu_group)
        except Exception:
            return (source_ranks, world_group.cpu_group, world_group.device_group)

        if source_group_world_size != len(source_ranks):
            return (source_ranks, world_group.cpu_group, world_group.device_group)

        return (source_ranks, source_cpu_group, source_device_group)

    def _prepare_lossless_shrink_payload(self, active_ranks: list[int],
                                         world_group) -> None:
        from vllm_ascend.ops.fused_moe import AscendFusedMoE

        self._lossless_shrink_payload = {}
        model_runner = getattr(self, "model_runner", None)
        model = getattr(model_runner, "model", None) if model_runner else None
        if model is None:
            return

        current_rank = torch.distributed.get_rank()
        (source_ranks, source_cpu_group,
         _) = self._get_shrink_source_group_state(world_group)
        world_size = world_group.world_size
        previous_active_ranks = self._get_previous_active_ranks_for_shrink(
            world_size)
        active_rank_to_idx = {rank: idx for idx, rank in enumerate(active_ranks)}
        logged_pairing_summary = False
        if current_rank == source_ranks[0]:
            logger.info(
                "Elastic shrink payload source group: rank=%s active_ranks=%s source_ranks=%s previous_active_ranks=%s",
                current_rank, active_ranks, source_ranks, previous_active_ranks)

        for module in model.modules():
            if not isinstance(module, AscendFusedMoE):
                continue
            if getattr(module, "elastic_moe_mode", "lossy") != "lossless":
                continue
            if module.expert_map is None:
                continue

            lossless_loaded_expert_map = getattr(module, "loaded_expert_map",
                                                 None)
            map_for_shrink = (lossless_loaded_expert_map
                              if lossless_loaded_expert_map is not None else
                              module.expert_map)
            local_loaded_expert_map_cpu = map_for_shrink.to(device="cpu",
                                                            dtype=torch.int32)
            gathered_loaded_maps_source = [
                torch.empty_like(local_loaded_expert_map_cpu)
                for _ in range(len(source_ranks))
            ]
            torch.distributed.all_gather(gathered_loaded_maps_source,
                                         local_loaded_expert_map_cpu,
                                         group=source_cpu_group)
            gathered_loaded_maps_by_rank = {
                source_ranks[idx]: gathered_loaded_maps_source[idx]
                for idx in range(len(source_ranks))
            }
            resident_expert_map = module.expert_map.to(device="cpu",
                                                       dtype=torch.int32)
            gathered_resident_maps_source = [
                torch.empty_like(resident_expert_map)
                for _ in range(len(source_ranks))
            ]
            torch.distributed.all_gather(gathered_resident_maps_source,
                                         resident_expert_map,
                                         group=source_cpu_group)
            gathered_resident_maps_by_rank = {
                source_ranks[idx]: gathered_resident_maps_source[idx]
                for idx in range(len(source_ranks))
            }

            logical_num_experts = int(module.elastic_original_num_experts)
            assignments: list[list[tuple[int, int]]] = [
                [] for _ in range(len(active_ranks))
            ]
            assigned_counts = [0 for _ in range(len(active_ranks))]
            target_per_rank = (
                logical_num_experts + len(active_ranks) - 1
            ) // len(active_ranks)
            cpu_import_source_rank: dict[int, int] = {}
            cpu_import_target_rank: dict[int, int] = {}
            use_paired_zero_redundancy = (
                getattr(module, "global_redundant_expert_num", 0) <= 0
                and len(previous_active_ranks) == 2 * len(active_ranks)
                and set(active_ranks).issubset(set(previous_active_ranks))
                and logical_num_experts % len(active_ranks) == 0
            )
            hybrid_resident_capacity = 0
            if (envs_ascend.VLLM_ASCEND_ELASTIC_EXECUTION_MODE == 2
                    and hasattr(module, "_get_hybrid_resident_capacity")):
                hybrid_resident_capacity = int(
                    module._get_hybrid_resident_capacity())
            use_paired_redundant_transfer = (
                int(getattr(module, "global_redundant_expert_num", 0)) > 0
                and len(previous_active_ranks) == 2 * len(active_ranks)
                and set(active_ranks).issubset(set(previous_active_ranks))
                and logical_num_experts % len(active_ranks) == 0
                and (
                    envs_ascend.VLLM_ASCEND_ELASTIC_EXECUTION_MODE == 1
                    or (
                        envs_ascend.VLLM_ASCEND_ELASTIC_EXECUTION_MODE == 2
                        and hybrid_resident_capacity > 0
                        and target_per_rank <= hybrid_resident_capacity
                    )
                ))
            prefer_preloaded_local_slots = (
                envs_ascend.VLLM_ASCEND_ELASTIC_EXECUTION_MODE == 1
                and int(getattr(module, "global_redundant_expert_num", 0)) > 0
            )
            inactive_to_active_rank: dict[int, int] = {}
            inactive_ranks = sorted(
                rank for rank in previous_active_ranks
                if rank not in active_rank_to_idx)
            if len(inactive_ranks) != len(active_ranks):
                use_paired_zero_redundancy = False
                use_paired_redundant_transfer = False
            elif use_paired_zero_redundancy:
                inactive_to_active_rank = {
                    inactive_rank: active_ranks[idx]
                    for idx, inactive_rank in enumerate(inactive_ranks)
                }
            if (use_paired_zero_redundancy and not logged_pairing_summary
                    and current_rank == active_ranks[0]
                    and module.layer_idx == 0):
                pass  # debug log removed
                logged_pairing_summary = True

            if use_paired_redundant_transfer:
                ordered_assignments: list[list[tuple[int, int]]] = []
                runtime_log2phy_cpu = torch.full((logical_num_experts, ),
                                                 -1,
                                                 dtype=torch.int32)
                dense_offset = 0
                active_runtime_slots_by_rank: dict[int, dict[int, int]] = {}
                active_free_slots_by_rank: dict[int, list[int]] = {}
                inactive_runtime_pairs_by_rank: dict[int, list[tuple[int, int]]] = {}
                active_free_count_by_rank: dict[int, int] = {}
                inactive_export_count_by_rank: dict[int, int] = {}

                for active_rank in active_ranks:
                    active_runtime_slots = {
                        expert_id: int(local_slot)
                        for expert_id, local_slot in enumerate(
                            gathered_resident_maps_by_rank[active_rank].tolist())
                        if int(local_slot) >= 0
                    }
                    active_slot_values = sorted(active_runtime_slots.values())
                    expected_active_slots = list(range(len(active_slot_values)))
                    if active_slot_values != expected_active_slots:
                        raise RuntimeError(
                            "Paired redundant shrink requires active resident "
                            f"slots to stay prefix-contiguous at layer={module.layer_idx}: "
                            f"rank={active_rank} slots={active_slot_values} "
                            f"expected={expected_active_slots}")
                    free_loaded_slots = list(
                        range(len(active_slot_values), target_per_rank))
                    active_runtime_slots_by_rank[active_rank] = active_runtime_slots
                    active_free_slots_by_rank[active_rank] = free_loaded_slots
                    active_free_count_by_rank[active_rank] = len(free_loaded_slots)

                for inactive_rank in inactive_ranks:
                    inactive_runtime_pairs = sorted(
                        ((expert_id, int(local_slot))
                         for expert_id, local_slot in enumerate(
                             gathered_resident_maps_by_rank[inactive_rank].tolist())
                         if int(local_slot) >= 0),
                        key=lambda item: item[1],
                    )
                    inactive_runtime_pairs_by_rank[
                        inactive_rank] = inactive_runtime_pairs
                    inactive_export_count_by_rank[inactive_rank] = len(
                        inactive_runtime_pairs)

                if (sum(active_free_count_by_rank.values()) != sum(
                        inactive_export_count_by_rank.values())):
                    raise RuntimeError(
                        "Paired redundant shrink requires total active free "
                        "slots to match total exiting resident experts at "
                        f"layer={module.layer_idx}: free_total="
                        f"{sum(active_free_count_by_rank.values())} "
                        f"export_total={sum(inactive_export_count_by_rank.values())}")

                candidate_active_by_inactive: dict[int, list[int]] = {}
                for inactive_rank, export_count in (
                        inactive_export_count_by_rank.items()):
                    candidate_active_by_inactive[inactive_rank] = [
                        active_rank for active_rank, free_count in
                        active_free_count_by_rank.items() if free_count == export_count
                    ]

                pairing_by_inactive: dict[int, int] = {}
                used_active_ranks: set[int] = set()
                ordered_inactive_ranks = sorted(
                    inactive_ranks,
                    key=lambda rank: (len(candidate_active_by_inactive[rank]),
                                      -inactive_export_count_by_rank[rank], rank))

                def _pair_rank(idx: int) -> bool:
                    if idx >= len(ordered_inactive_ranks):
                        return True
                    inactive_rank = ordered_inactive_ranks[idx]
                    for active_rank in sorted(
                            candidate_active_by_inactive[inactive_rank]):
                        if active_rank in used_active_ranks:
                            continue
                        pairing_by_inactive[inactive_rank] = active_rank
                        used_active_ranks.add(active_rank)
                        if _pair_rank(idx + 1):
                            return True
                        used_active_ranks.remove(active_rank)
                        del pairing_by_inactive[inactive_rank]
                    return False

                if not _pair_rank(0):
                    raise RuntimeError(
                        "Paired redundant shrink could not build a one-to-one "
                        f"matching at layer={module.layer_idx}: "
                        f"active_free_count_by_rank={active_free_count_by_rank} "
                        f"inactive_export_count_by_rank={inactive_export_count_by_rank}")

                pairing_by_active = {
                    active_rank: inactive_rank
                    for inactive_rank, active_rank in pairing_by_inactive.items()
                }

                for pair_idx, active_rank in enumerate(active_ranks):
                    inactive_rank = pairing_by_active.get(active_rank)
                    if inactive_rank is None:
                        raise RuntimeError(
                            "Paired redundant shrink missing inactive partner "
                            f"at layer={module.layer_idx}: rank={active_rank}")
                    active_runtime_slots = active_runtime_slots_by_rank[active_rank]
                    inactive_runtime_pairs = inactive_runtime_pairs_by_rank[
                        inactive_rank]
                    free_loaded_slots = active_free_slots_by_rank[active_rank]
                    if len(free_loaded_slots) != len(inactive_runtime_pairs):
                        raise RuntimeError(
                            "Paired redundant shrink found mismatched free/paired "
                            f"slots at layer={module.layer_idx}: "
                            f"active_rank={active_rank} inactive_rank={inactive_rank} "
                            f"free_loaded_slots={len(free_loaded_slots)} "
                            f"paired_runtime={len(inactive_runtime_pairs)}")

                    ordered_rank_assignments: list[Optional[tuple[int, int]]] = [
                        None for _ in range(target_per_rank)
                    ]
                    for expert_id, local_slot in active_runtime_slots.items():
                        ordered_rank_assignments[local_slot] = (
                            int(expert_id), int(local_slot))
                    for free_slot, (expert_id, _) in zip(free_loaded_slots,
                                                         inactive_runtime_pairs):
                        ordered_rank_assignments[free_slot] = (
                            int(expert_id), -1)
                        cpu_import_source_rank[int(expert_id)] = inactive_rank
                        cpu_import_target_rank[int(expert_id)] = active_rank

                    if any(item is None for item in ordered_rank_assignments):
                        raise RuntimeError(
                            "Paired redundant shrink left unassigned local slots "
                            f"at layer={module.layer_idx}: rank={active_rank}")
                    finalized_assignments = [
                        item for item in ordered_rank_assignments if item is not None
                    ]
                    assignments[pair_idx] = finalized_assignments
                    ordered_assignments.append(finalized_assignments)
                    assigned_counts[pair_idx] = len(finalized_assignments)
                    for local_slot, (expert_id, _) in enumerate(
                            finalized_assignments):
                        runtime_log2phy_cpu[int(expert_id)] = dense_offset + local_slot
                    dense_offset += len(finalized_assignments)

                if int(torch.unique(runtime_log2phy_cpu).numel()) != logical_num_experts:
                    raise RuntimeError(
                        f"layer={module.layer_idx}: unique="
                        f"{int(torch.unique(runtime_log2phy_cpu).numel())} "
                        f"logical={logical_num_experts}")

                self._lossless_shrink_payload[module.layer_idx] = {
                    "assignments": assignments,
                    "ordered_assignments": ordered_assignments,
                    "runtime_log2phy_cpu": runtime_log2phy_cpu,
                    "cpu_import_source_rank": cpu_import_source_rank,
                    "cpu_import_target_rank": cpu_import_target_rank,
                }
                if module.layer_idx == 0 and current_rank == active_ranks[0]:
                    logger.info(
                        "Elastic paired redundant shrink plan: rank=%s active_ranks=%s inactive_ranks=%s target_per_rank=%s pairings=%s paired_imports=%s",
                        current_rank,
                        active_ranks,
                        inactive_ranks,
                        target_per_rank,
                        sorted((inactive_rank, active_rank)
                               for inactive_rank, active_rank in
                               pairing_by_inactive.items()),
                        len(cpu_import_source_rank),
                    )
                continue

            for expert_id in range(logical_num_experts):
                candidate_rank_indices = []
                for rank in active_ranks:
                    loaded_local_id = int(
                        gathered_loaded_maps_by_rank[rank][expert_id].item())
                    if loaded_local_id >= 0:
                        candidate_rank_indices.append(active_rank_to_idx[rank])

                if candidate_rank_indices:
                    preferred_candidates = [
                        rank_idx for rank_idx in candidate_rank_indices
                        if assigned_counts[rank_idx] < target_per_rank
                    ]
                    selected_rank_idx = -1
                    selected_world_rank = None
                    if preferred_candidates:
                        selected_rank_idx = min(
                            preferred_candidates,
                            key=lambda rank_idx:
                            (assigned_counts[rank_idx], rank_idx))
                        selected_world_rank = active_ranks[selected_rank_idx]
                    elif prefer_preloaded_local_slots:
                        underfull_rank_indices = [
                            rank_idx for rank_idx in range(len(active_ranks))
                            if assigned_counts[rank_idx] < target_per_rank
                        ]
                        if underfull_rank_indices:
                            selected_rank_idx = min(
                                underfull_rank_indices,
                                key=lambda rank_idx:
                                (assigned_counts[rank_idx], rank_idx))
                            selected_world_rank = active_ranks[selected_rank_idx]
                    if selected_rank_idx < 0 or selected_world_rank is None:
                        selected_rank_idx = min(
                            candidate_rank_indices,
                            key=lambda rank_idx:
                            (assigned_counts[rank_idx], rank_idx))
                        selected_world_rank = active_ranks[selected_rank_idx]
                    selected_resident_local_id = int(
                        gathered_resident_maps_by_rank[selected_world_rank]
                        [expert_id].item())
                    selected_loaded_local_id = int(
                        gathered_loaded_maps_by_rank[selected_world_rank]
                        [expert_id].item())
                    if (prefer_preloaded_local_slots
                            and selected_loaded_local_id >= 0):
                        selected_source_local_id = selected_loaded_local_id
                    else:
                        selected_source_local_id = selected_resident_local_id
                    if selected_source_local_id < 0:
                        source_world_rank = None
                        for world_rank in source_ranks:
                            if world_rank == selected_world_rank:
                                continue
                            loaded_local_id = int(
                                gathered_loaded_maps_by_rank[world_rank][expert_id]
                                .item())
                            if loaded_local_id >= 0:
                                source_world_rank = world_rank
                                break
                        if source_world_rank is None:
                            source_world_rank = selected_world_rank
                        cpu_import_source_rank[expert_id] = source_world_rank
                        cpu_import_target_rank[expert_id] = selected_world_rank
                else:
                    source_world_rank = None
                    for world_rank in source_ranks:
                        loaded_local_id = int(
                            gathered_loaded_maps_by_rank[world_rank][expert_id].
                            item())
                        if loaded_local_id >= 0:
                            source_world_rank = world_rank
                            break
                    if source_world_rank is None:
                        raise RuntimeError(
                            f"expert {expert_id} at layer={module.layer_idx}. "
                            "Increase init redundancy or switch back to lossy mode."
                        )
                    if use_paired_zero_redundancy:
                        selected_world_rank = inactive_to_active_rank[
                            source_world_rank]
                        selected_rank_idx = active_rank_to_idx[
                            selected_world_rank]
                    else:
                        selected_rank_idx = min(
                            range(len(active_ranks)),
                            key=lambda rank_idx:
                            (assigned_counts[rank_idx], rank_idx))
                        selected_world_rank = active_ranks[selected_rank_idx]
                    selected_source_local_id = -1
                    cpu_import_source_rank[expert_id] = source_world_rank
                    cpu_import_target_rank[expert_id] = selected_world_rank

                assignments[selected_rank_idx].append(
                    (expert_id, selected_source_local_id))
                assigned_counts[selected_rank_idx] += 1

            if use_paired_zero_redundancy:
                expected_local_experts = logical_num_experts // len(active_ranks)
                for rank_idx, count in enumerate(assigned_counts):
                    if count != expected_local_experts:
                        raise RuntimeError(
                            "Paired zero-redundancy shrink produced uneven "
                            f"assignment at layer={module.layer_idx}: "
                            f"rank={active_ranks[rank_idx]} count={count} "
                            f"expected={expected_local_experts}")

            ordered_assignments: list[list[tuple[int, int]]] = []
            runtime_log2phy_cpu = torch.full((logical_num_experts, ),
                                             -1,
                                             dtype=torch.int32)
            dense_offset = 0
            for rank_idx, rank in enumerate(active_ranks):
                current_runtime_slots = {
                    expert_id: int(local_slot)
                    for expert_id, local_slot in enumerate(
                        gathered_resident_maps_by_rank[rank].tolist())
                    if local_slot >= 0
                }
                rank_assignment_map = {
                    expert_id: local_id
                    for expert_id, local_id in assignments[rank_idx]
                }
                preserved_assignments = sorted(
                    [(expert_id, rank_assignment_map[expert_id])
                     for expert_id in rank_assignment_map
                     if expert_id in current_runtime_slots],
                    key=lambda item: current_runtime_slots[item[0]])
                preserved_expert_ids = {
                    expert_id for expert_id, _ in preserved_assignments
                }
                appended_assignments = sorted(
                    [(expert_id, local_id)
                     for expert_id, local_id in assignments[rank_idx]
                     if expert_id not in preserved_expert_ids],
                    key=lambda item: item[0])
                ordered_rank_assignments = (
                    preserved_assignments + appended_assignments)
                ordered_assignments.append(ordered_rank_assignments)
                for local_slot, (expert_id, _) in enumerate(
                        ordered_rank_assignments):
                    runtime_log2phy_cpu[expert_id] = dense_offset + local_slot
                dense_offset += len(ordered_rank_assignments)

            if int(torch.unique(runtime_log2phy_cpu).numel()) != logical_num_experts:
                raise RuntimeError(
                    f"layer={module.layer_idx}: unique="
                    f"{int(torch.unique(runtime_log2phy_cpu).numel())} "
                    f"logical={logical_num_experts}")

            self._lossless_shrink_payload[module.layer_idx] = {
                "assignments": assignments,
                "ordered_assignments": ordered_assignments,
                "runtime_log2phy_cpu": runtime_log2phy_cpu,
                "cpu_import_source_rank": cpu_import_source_rank,
                "cpu_import_target_rank": cpu_import_target_rank,
            }
            if module.layer_idx == 0 and current_rank == active_ranks[0]:
                ordered_heads = {
                    active_ranks[idx]:
                    [expert_id for expert_id, _ in rank_assignments[:4]]
                    for idx, rank_assignments in enumerate(ordered_assignments)
                }
                ordered_tails = {
                    active_ranks[idx]:
                    [expert_id for expert_id, _ in rank_assignments[-4:]]
                    for idx, rank_assignments in enumerate(ordered_assignments)
                }
                pass  # debug log removed

    def _get_lossless_hybrid_import_mode(self) -> str:
        mode = str(
            envs_ascend.VLLM_ASCEND_LOSSLESS_HYBRID_IMPORT_MODE).lower().strip()
        valid_modes = {"cpu_p2p", "npu_p2p_to_cpu"}
        if mode not in valid_modes:
            raise ValueError(
                "VLLM_ASCEND_LOSSLESS_HYBRID_IMPORT_MODE must be one of "
                f"{sorted(valid_modes)}, got {mode!r}.")
        return mode

    def _iter_lossless_hybrid_import_chunks(
            self, expert_ids: list[int]) -> list[list[int]]:
        chunk_size = int(
            envs_ascend.VLLM_ASCEND_LOSSLESS_HYBRID_IMPORT_CHUNK_EXPERTS)
        return [
            expert_ids[idx:idx + chunk_size]
            for idx in range(0, len(expert_ids), chunk_size)
        ]

    def _export_lossless_expert_cpu_weight_batch(
            self, module,
            expert_ids: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
        if not expert_ids:
            w13_tail_shape = tuple(module.w13_weight.shape[1:])
            w2_tail_shape = tuple(module.w2_weight.shape[1:])
            return (torch.empty((0, ) + w13_tail_shape,
                                device="cpu",
                                dtype=module.w13_weight.dtype),
                    torch.empty((0, ) + w2_tail_shape,
                                device="cpu",
                                dtype=module.w2_weight.dtype))
        if hasattr(module, "ensure_lossless_cpu_shadow"):
            module.ensure_lossless_cpu_shadow()
        source_w13 = getattr(module, "lossless_cpu_w13_weight", None)
        source_w2 = getattr(module, "lossless_cpu_w2_weight", None)
        if source_w13 is None or source_w2 is None:
            source_w13 = module.w13_weight.detach().cpu()
            source_w2 = module.w2_weight.detach().cpu()
        export_map = (module.loaded_expert_map
                      if getattr(module, "loaded_expert_map", None) is not None
                      else module.expert_map)
        if export_map is None:
            raise RuntimeError(
                f"Missing export_map at layer={module.layer_idx}.")
        local_slots = [int(export_map[int(expert_id)].item()) for expert_id in expert_ids]
        if any(local_slot < 0 for local_slot in local_slots):
            raise RuntimeError(
                f"Invalid CPU export slot at layer={module.layer_idx}: "
                f"expert_ids={expert_ids} local_slots={local_slots}")
        export_index = torch.tensor(local_slots, device="cpu", dtype=torch.long)
        return (source_w13.index_select(0, export_index).contiguous(),
                source_w2.index_select(0, export_index).contiguous())

    def _store_lossless_import_cpu_batch(
            self,
            local_cpu_import_weights: dict[int, tuple[torch.Tensor, torch.Tensor]],
            expert_ids: list[int],
            batch_w13: torch.Tensor,
            batch_w2: torch.Tensor) -> None:
        for idx, expert_id in enumerate(expert_ids):
            local_cpu_import_weights[int(expert_id)] = (
                batch_w13[idx].detach().clone(),
                batch_w2[idx].detach().clone(),
            )

    def _stream_lossless_hybrid_import_weights_p2p(
            self,
            module,
            active_ranks: list[int],
            source_cpu_group,
            source_device_group,
            local_needed_cpu_import_ids: set[int],
            cpu_import_source_rank: dict[int, int],
            cpu_import_target_rank: dict[int, int],
            participate_only: bool = False
    ) -> tuple[dict[int, tuple[torch.Tensor, torch.Tensor]], dict[int, int]]:
        current_rank = torch.distributed.get_rank()
        transfer_mode = self._get_lossless_hybrid_import_mode()
        local_cpu_import_weights: dict[int, tuple[torch.Tensor,
                                                  torch.Tensor]] = {}
        local_direct_import_slots: dict[int, int] = {}
        transfer_ids_by_pair: dict[tuple[int, int], list[int]] = {}
        local_only_ids: list[int] = []

        for expert_id, source_rank in cpu_import_source_rank.items():
            target_rank = cpu_import_target_rank.get(expert_id)
            if source_rank is None or target_rank is None:
                continue
            if source_rank == target_rank:
                local_only_ids.append(int(expert_id))
                continue
            transfer_ids_by_pair.setdefault((int(source_rank), int(target_rank)),
                                            []).append(int(expert_id))

        for pair in transfer_ids_by_pair:
            transfer_ids_by_pair[pair].sort()

        valid_source_ranks = sorted(
            int(rank) for rank in cpu_import_source_rank.values()
            if rank is not None)
        if (module.layer_idx == 0 and valid_source_ranks
                and current_rank == valid_source_ranks[0]):
            remote_expert_count = sum(
                len(expert_ids) for expert_ids in transfer_ids_by_pair.values())
            logger.info(
                "Elastic lossless hybrid import path: rank=%s layer=%s active_ranks=%s mode=%s transfer_pairs=%s remote_experts=%s local_only=%s chunk_experts=%s",
                self.rank,
                module.layer_idx,
                active_ranks,
                transfer_mode,
                len(transfer_ids_by_pair),
                remote_expert_count,
                len(local_only_ids),
                int(
                    envs_ascend
                    .VLLM_ASCEND_LOSSLESS_HYBRID_IMPORT_CHUNK_EXPERTS),
            )

        local_copy_ids = sorted(
            expert_id for expert_id in local_only_ids
            if cpu_import_target_rank.get(expert_id) == current_rank
            and expert_id in local_needed_cpu_import_ids)
        if local_copy_ids and not participate_only:
            local_w13, local_w2 = self._export_lossless_expert_cpu_weight_batch(
                module, local_copy_ids)
            self._store_lossless_import_cpu_batch(local_cpu_import_weights,
                                                  local_copy_ids, local_w13,
                                                  local_w2)

        w13_tail_shape = tuple(module.w13_weight.shape[1:])
        w2_tail_shape = tuple(module.w2_weight.shape[1:])
        w13_dtype = module.w13_weight.dtype
        w2_dtype = module.w2_weight.dtype
        npu_device = (module.expert_map.device
                      if module.expert_map is not None else module.w13_weight.device)

        for (source_rank, target_rank), expert_ids in sorted(
                transfer_ids_by_pair.items()):
            for expert_chunk in self._iter_lossless_hybrid_import_chunks(
                    expert_ids):
                if current_rank == source_rank:
                    send_cpu_w13, send_cpu_w2 = (
                        self._export_lossless_expert_cpu_weight_batch(
                            module, expert_chunk))
                    if transfer_mode == "cpu_p2p":
                        send_w13 = send_cpu_w13
                        send_w2 = send_cpu_w2
                        send_group = source_cpu_group
                    else:
                        send_w13 = send_cpu_w13.to(device=npu_device,
                                                   non_blocking=False)
                        send_w2 = send_cpu_w2.to(device=npu_device,
                                                 non_blocking=False)
                        send_group = source_device_group
                    send_req_w13 = torch.distributed.isend(send_w13,
                                                           dst=target_rank,
                                                           group=send_group)
                    send_req_w2 = torch.distributed.isend(send_w2,
                                                          dst=target_rank,
                                                          group=send_group)
                    send_req_w13.wait()
                    send_req_w2.wait()

                if ((not participate_only) and current_rank == target_rank
                        and all(expert_id in local_needed_cpu_import_ids
                                for expert_id in expert_chunk)):
                    recv_shape_w13 = (len(expert_chunk), ) + w13_tail_shape
                    recv_shape_w2 = (len(expert_chunk), ) + w2_tail_shape
                    if transfer_mode == "cpu_p2p":
                        recv_w13 = torch.empty(recv_shape_w13,
                                               device="cpu",
                                               dtype=w13_dtype)
                        recv_w2 = torch.empty(recv_shape_w2,
                                              device="cpu",
                                              dtype=w2_dtype)
                        recv_group = source_cpu_group
                    else:
                        recv_w13 = torch.empty(recv_shape_w13,
                                               device=npu_device,
                                               dtype=w13_dtype)
                        recv_w2 = torch.empty(recv_shape_w2,
                                              device=npu_device,
                                              dtype=w2_dtype)
                        recv_group = source_device_group
                    recv_req_w13 = torch.distributed.irecv(recv_w13,
                                                           src=source_rank,
                                                           group=recv_group)
                    recv_req_w2 = torch.distributed.irecv(recv_w2,
                                                          src=source_rank,
                                                          group=recv_group)
                    recv_req_w13.wait()
                    recv_req_w2.wait()
                    if transfer_mode != "cpu_p2p":
                        recv_w13 = recv_w13.detach().cpu()
                        recv_w2 = recv_w2.detach().cpu()
                    self._store_lossless_import_cpu_batch(
                        local_cpu_import_weights, expert_chunk, recv_w13,
                        recv_w2)

        return local_cpu_import_weights, local_direct_import_slots

    def _stream_lossless_layer_cpu_import_weights(
            self,
            module,
            payload: dict,
            active_ranks: list[int],
            world_group,
            participate_only: bool = False
    ) -> tuple[dict[int, tuple[torch.Tensor, torch.Tensor]], dict[int, int]]:
        current_rank = torch.distributed.get_rank()
        (source_ranks, source_cpu_group,
         source_device_group) = self._get_shrink_source_group_state(world_group)
        world_size = world_group.world_size
        previous_active_ranks = self._get_previous_active_ranks_for_shrink(
            world_size)
        active_rank_to_idx = {rank: idx for idx, rank in enumerate(active_ranks)}
        my_active_idx = active_rank_to_idx.get(current_rank)
        ordered_assignments = payload.get("ordered_assignments",
                                          payload["assignments"])

        cpu_import_source_rank = payload["cpu_import_source_rank"]
        cpu_import_target_rank = payload.get("cpu_import_target_rank", {})
        local_needed_cpu_import_ids = set()
        local_import_slot_by_expert: dict[int, int] = {}
        if my_active_idx is not None:
            local_import_slot_by_expert = {
                expert_id: local_slot
                for local_slot, (expert_id, source_local_id) in enumerate(
                    ordered_assignments[my_active_idx])
                if source_local_id < 0
            }
            local_needed_cpu_import_ids = {
                expert_id for expert_id, source_rank in cpu_import_source_rank.items()
                if source_rank is not None and any(
                    assigned_expert_id == expert_id
                    for assigned_expert_id, _ in payload["assignments"][my_active_idx])
            }

        local_cpu_import_weights: dict[int, tuple[torch.Tensor,
                                                  torch.Tensor]] = {}
        local_direct_import_slots: dict[int, int] = {}
        export_ids_per_source_rank: dict[int, list[int]] = {}
        for expert_id, source_rank in cpu_import_source_rank.items():
            export_ids_per_source_rank.setdefault(source_rank, []).append(expert_id)
        for source_rank in export_ids_per_source_rank:
            export_ids_per_source_rank[source_rank].sort()

        target_owned_local_expert_count = 0
        if ordered_assignments:
            target_owned_local_expert_count = max(
                len(assignments) for assignments in ordered_assignments)
        use_hybrid_cpu_swap = (
            hasattr(module, "should_activate_lossless_hybrid_for_target")
            and module.should_activate_lossless_hybrid_for_target(
                target_owned_local_expert_count, len(active_ranks)))
        loaded_weight_capacity = int(getattr(module, "loaded_weight_capacity", 0))
        w13_rows = int(module.w13_weight.shape[0])
        w2_rows = int(module.w2_weight.shape[0])
        can_direct_fill_loaded_slots = (
            target_owned_local_expert_count > 0
            and loaded_weight_capacity >= target_owned_local_expert_count
            and w13_rows >= target_owned_local_expert_count
            and w2_rows >= target_owned_local_expert_count
        )
        hybrid_resident_capacity = 0
        if (envs_ascend.VLLM_ASCEND_ELASTIC_EXECUTION_MODE == 2
                and hasattr(module, "_get_hybrid_resident_capacity")):
            hybrid_resident_capacity = int(
                module._get_hybrid_resident_capacity())
        requires_redundant_direct_npu = (
            int(getattr(module, "global_redundant_expert_num", 0)) > 0
            and not use_hybrid_cpu_swap
            and len(previous_active_ranks) == 2 * len(active_ranks)
            and set(active_ranks).issubset(set(previous_active_ranks))
            and target_owned_local_expert_count > 0
            and (
                envs_ascend.VLLM_ASCEND_ELASTIC_EXECUTION_MODE == 1
                or (
                    envs_ascend.VLLM_ASCEND_ELASTIC_EXECUTION_MODE == 2
                    and hybrid_resident_capacity > 0
                    and target_owned_local_expert_count
                    <= hybrid_resident_capacity
                )
            ))
        use_direct_npu_slot_import = (
            requires_redundant_direct_npu
            and can_direct_fill_loaded_slots
        )
        if requires_redundant_direct_npu and not can_direct_fill_loaded_slots:
            raise RuntimeError(
                "Redundant-expert elastic shrink requires preloaded NPU slot "
                f"capacity at layer={module.layer_idx}, but got "
                f"target_owned_local={target_owned_local_expert_count}, "
                f"loaded_weight_capacity={loaded_weight_capacity}, "
                f"w13_rows={w13_rows}, w2_rows={w2_rows}."
            )

        if use_hybrid_cpu_swap:
            valid_source_ranks = sorted(
                int(rank) for rank in cpu_import_source_rank.values()
                if rank is not None)
            if (module.layer_idx == 0 and valid_source_ranks
                    and current_rank == valid_source_ranks[0]):
                resident_capacity = (
                    module._get_hybrid_resident_capacity()
                    if hasattr(module, "_get_hybrid_resident_capacity") else -1)
                logger.info(
                    "Elastic shrink preload selected hybrid import path: rank=%s layer=%s active_ranks=%s target_owned_local=%s resident_capacity=%s",
                    self.rank,
                    module.layer_idx,
                    active_ranks,
                    target_owned_local_expert_count,
                    resident_capacity,
                )
            return self._stream_lossless_hybrid_import_weights_p2p(
                module=module,
                active_ranks=active_ranks,
                source_cpu_group=source_cpu_group,
                source_device_group=source_device_group,
                local_needed_cpu_import_ids=local_needed_cpu_import_ids,
                cpu_import_source_rank=cpu_import_source_rank,
                cpu_import_target_rank=cpu_import_target_rank,
                participate_only=participate_only)

        use_npu_import = (
            (getattr(module, "global_redundant_expert_num", 0) <= 0
             and len(previous_active_ranks) == 2 * len(active_ranks)
             and set(active_ranks).issubset(set(previous_active_ranks)))
            or use_direct_npu_slot_import
        )

        if use_npu_import:
            import_start = time.perf_counter()
            w13_tail_shape = tuple(module.w13_weight.shape[1:])
            w2_tail_shape = tuple(module.w2_weight.shape[1:])
            direct_fill_preallocated_loaded = bool(
                getattr(module, "lossless_zero_redundancy_preallocated_loaded",
                        False)) or use_direct_npu_slot_import
            if (module.layer_idx == 0 and current_rank == source_ranks[0]
                    and cpu_import_source_rank):
                logger.info(
                    "Elastic shrink preload selected direct NPU import path: rank=%s layer=%s active_ranks=%s target_owned_local=%s loaded_capacity=%s direct_fill=%s redundancy=%s",
                    self.rank,
                    module.layer_idx,
                    active_ranks,
                    target_owned_local_expert_count,
                    int(getattr(module, "loaded_weight_capacity", 0)),
                    direct_fill_preallocated_loaded,
                    int(getattr(module, "global_redundant_expert_num", 0)),
                )
            recv_w13 = None
            recv_w2 = None
            if not direct_fill_preallocated_loaded:
                recv_w13 = torch.empty((1, ) + w13_tail_shape,
                                       device=module.expert_map.device,
                                       dtype=module.w13_weight.dtype)
                recv_w2 = torch.empty((1, ) + w2_tail_shape,
                                      device=module.expert_map.device,
                                      dtype=module.w2_weight.dtype)
            for source_rank in source_ranks:
                source_export_ids = export_ids_per_source_rank.get(source_rank, [])
                if not source_export_ids:
                    continue
                for expert_id in source_export_ids:
                    target_rank = cpu_import_target_rank.get(expert_id)
                    if target_rank is None:
                        raise RuntimeError(
                            f"Missing lossless NPU import target for expert {expert_id} "
                            f"at layer={module.layer_idx}.")
                    if current_rank == source_rank:
                        export_w13, export_w2 = module.export_lossless_expert_npu_weights(
                            [expert_id])
                        send_w13 = torch.distributed.isend(
                            export_w13,
                            dst=target_rank,
                            group=source_device_group)
                        send_w2 = torch.distributed.isend(
                            export_w2,
                            dst=target_rank,
                            group=source_device_group)
                        send_w13.wait()
                        send_w2.wait()
                        continue
                    if ((not participate_only)
                            and current_rank == target_rank
                            and expert_id in local_needed_cpu_import_ids):
                        if direct_fill_preallocated_loaded:
                            target_slot = local_import_slot_by_expert[expert_id]
                            recv_target_w13 = module.w13_weight[
                                target_slot:target_slot + 1]
                            recv_target_w2 = module.w2_weight[
                                target_slot:target_slot + 1]
                        else:
                            recv_target_w13 = recv_w13
                            recv_target_w2 = recv_w2
                        recv_req_w13 = torch.distributed.irecv(
                            recv_target_w13,
                            src=source_rank,
                            group=source_device_group)
                        recv_req_w2 = torch.distributed.irecv(
                            recv_target_w2,
                            src=source_rank,
                            group=source_device_group)
                        recv_req_w13.wait()
                        recv_req_w2.wait()
                        if not direct_fill_preallocated_loaded:
                            assert recv_w13 is not None and recv_w2 is not None
                            local_cpu_import_weights[expert_id] = (
                                recv_w13[0].detach().cpu(),
                                recv_w2[0].detach().cpu(),
                            )
                        else:
                            local_direct_import_slots[expert_id] = target_slot
            if (not participate_only and my_active_idx is not None
                    and module.layer_idx == 0 and local_needed_cpu_import_ids):
                pass  # debug log removed
            return local_cpu_import_weights, local_direct_import_slots

        if requires_redundant_direct_npu and cpu_import_source_rank:
            raise RuntimeError(
                "Redundant-expert elastic shrink fell back to CPU/object import "
                f"at layer={module.layer_idx}. This mode requires direct NPU "
                "P2P into preloaded expert slots."
            )

        for source_rank in source_ranks:
            source_export_ids = export_ids_per_source_rank.get(source_rank, [])
            source_payload = None
            if current_rank == source_rank and source_export_ids:
                source_payload = module.export_lossless_expert_cpu_weights(
                    source_export_ids)
            object_list = [source_payload]
            torch.distributed.broadcast_object_list(object_list,
                                                    src=source_rank,
                                                    group=source_cpu_group)
            received_payload = object_list[0]
            if (not participate_only) and local_needed_cpu_import_ids and received_payload:
                for expert_id in local_needed_cpu_import_ids:
                    if cpu_import_source_rank.get(expert_id) != source_rank:
                        continue
                    if expert_id not in received_payload:
                        raise RuntimeError(
                            f"from rank {source_rank} at layer={module.layer_idx}."
                        )
                    local_cpu_import_weights[expert_id] = received_payload[expert_id]
            del object_list
            del received_payload

        return local_cpu_import_weights, local_direct_import_slots

    def _preload_lossless_shrink_import_weights(self, active_ranks: list[int],
                                                world_group) -> None:
        from vllm_ascend.ops.fused_moe import AscendFusedMoE

        self._lossless_preloaded_cpu_import_weights = {}
        self._lossless_preloaded_direct_import_slots = {}
        model_runner = getattr(self, "model_runner", None)
        model = getattr(model_runner, "model", None) if model_runner else None
        if model is None:
            return

        lossless_shrink_payload = getattr(self, "_lossless_shrink_payload", {})
        for module in model.modules():
            if not isinstance(module, AscendFusedMoE):
                continue
            use_lossless_mode = (
                envs_ascend.VLLM_ASCEND_ELASTIC_MOE_MODE == "lossless"
                and getattr(module, "elastic_moe_mode", "lossy") == "lossless")
            if not use_lossless_mode:
                continue
            payload = lossless_shrink_payload.get(module.layer_idx)
            if payload is None:
                raise RuntimeError(
                    f"Missing lossless shrink payload for layer={module.layer_idx}.")
            (self._lossless_preloaded_cpu_import_weights[module.layer_idx],
             self._lossless_preloaded_direct_import_slots[module.layer_idx]) = (
                 self._stream_lossless_layer_cpu_import_weights(
                     module, payload, active_ranks, world_group))

    def _destroy_group_if_present(self, state_module, attr_name: str) -> None:
        group = getattr(state_module, attr_name, None)
        if group is not None:
            group.destroy()
            setattr(state_module, attr_name, None)

    def _get_cached_elastic_parallel_groups(self) -> dict[str, dict[tuple[int, ...], object]]:
        cached_groups = getattr(self, "_elastic_cached_parallel_groups", None)
        if cached_groups is None:
            cached_groups = {}
            self._elastic_cached_parallel_groups = cached_groups
        return cached_groups

    def _get_seen_elastic_parallel_group_signatures(
            self) -> set[tuple[str, tuple[int, ...]]]:
        seen_signatures = getattr(
            self, "_elastic_seen_parallel_group_signatures", None)
        if seen_signatures is None:
            seen_signatures = set()
            self._elastic_seen_parallel_group_signatures = seen_signatures
        return seen_signatures

    def _normalize_elastic_parallel_group_kind(self, name: str) -> str:
        normalized_name = name.lower().lstrip("_")
        if normalized_name.endswith("dp"):
            return "dp"
        if normalized_name.endswith("ep"):
            return "ep"
        if normalized_name.endswith("mc2"):
            return "mc2"
        return normalized_name

    def _should_cache_elastic_parallel_group(self, attr_name: str) -> bool:
        # Keep DP/EP restore fast, but recreate MC2 every time. Caching MC2 keeps
        # the previous communicator alive while the post-shrink 8-rank MC2 group
        # is created, which can push aclnnMoeDistributeDispatchV2 over the HCCL
        # workspace memory budget on the next step.
        return self._normalize_elastic_parallel_group_kind(attr_name) != "mc2"

    def _stash_group_if_present(self, state_module, attr_name: str) -> None:
        group = getattr(state_module, attr_name, None)
        if group is None:
            return

        should_cache_group = self._should_cache_elastic_parallel_group(
            attr_name)
        group_ranks = tuple(int(rank) for rank in getattr(group, "ranks", []))
        if group_ranks and should_cache_group:
            cached_groups = self._get_cached_elastic_parallel_groups()
            cached_groups.setdefault(attr_name, {})[group_ranks] = group
            group_kind = self._normalize_elastic_parallel_group_kind(attr_name)
            self._get_seen_elastic_parallel_group_signatures().add(
                (group_kind, group_ranks))
        else:
            if group_ranks:
                group_kind = self._normalize_elastic_parallel_group_kind(
                    attr_name)
                self._get_seen_elastic_parallel_group_signatures().discard(
                    (group_kind, group_ranks))
            group.destroy()
        setattr(state_module, attr_name, None)

    def _get_local_group_ranks(self,
                               group_ranks: list[list[int]]) -> tuple[int, ...]:
        current_rank = torch.distributed.get_rank()
        for ranks in group_ranks:
            if current_rank in ranks:
                return tuple(int(rank) for rank in ranks)
        return ()

    def _drop_stale_cached_elastic_parallel_groups(
            self, keep_group_ranks: tuple[int, ...]) -> None:
        cached_groups = self._get_cached_elastic_parallel_groups()
        seen_signatures = self._get_seen_elastic_parallel_group_signatures()
        dropped_groups = 0
        dropped_signatures = 0

        for attr_name, groups_by_ranks in cached_groups.items():
            group_kind = self._normalize_elastic_parallel_group_kind(attr_name)
            stale_group_ranks = [
                ranks
                for ranks in list(groups_by_ranks.keys())
                if ranks != keep_group_ranks
            ]
            for ranks in stale_group_ranks:
                group = groups_by_ranks.pop(ranks, None)
                if group is not None:
                    group.destroy()
                    dropped_groups += 1
                if (group_kind, ranks) in seen_signatures:
                    seen_signatures.discard((group_kind, ranks))
                    dropped_signatures += 1

        stale_signatures = [
            signature
            for signature in seen_signatures
            if signature[1] != keep_group_ranks
        ]
        for signature in stale_signatures:
            seen_signatures.discard(signature)
            dropped_signatures += 1

        if dropped_groups > 0 or dropped_signatures > 0:
            logger.info(
                "Elastic parallel stale group cache dropped: rank=%s keep_ranks=%s dropped_groups=%s dropped_signatures=%s",
                self.rank, keep_group_ranks, dropped_groups,
                dropped_signatures)

    def _advance_group_creation_sequence_for_non_member(
            self, group_ranks: list[list[int]], backend: str,
            group_name: str) -> None:
        """Keep default-pg new_group ordering aligned on detached ranks.

        During elastic shrink, active ranks rebuild DP/EP/MC2 groups and thus
        consume a sequence of torch.distributed.new_group() calls. Detached
        ranks do not join those groups, but they still need to advance through
        the same new_group creation order so a later full-world restore can
        rebuild the original 16-rank groups without Gloo store key mismatch.
        """
        group_kind = self._normalize_elastic_parallel_group_kind(group_name)
        unseen_group_ranks = []
        seen_signatures = self._get_seen_elastic_parallel_group_signatures()
        for ranks in group_ranks:
            ranks_key = tuple(int(rank) for rank in ranks)
            group_signature = (group_kind, ranks_key)
            if group_signature in seen_signatures:
                continue
            unseen_group_ranks.append(ranks)

        if not unseen_group_ranks:
            logger.info(
                "Elastic parallel non-member group sequence reuse: rank=%s group_name=%s groups=%s",
                self.rank, group_name, len(group_ranks))
            return

        placeholder_groups = []
        for ranks in unseen_group_ranks:
            placeholder_groups.append(
                torch.distributed.new_group(ranks, backend=backend))
            placeholder_groups.append(
                torch.distributed.new_group(ranks, backend="gloo"))

        for group in placeholder_groups:
            if group == dist.GroupMember.NON_GROUP_MEMBER:
                continue
            torch.distributed.destroy_process_group(group)

        logger.info(
            "Elastic parallel non-member group sequence advanced: rank=%s group_name=%s groups=%s",
            self.rank, group_name, len(unseen_group_ranks))
        for ranks in unseen_group_ranks:
            seen_signatures.add(
                (group_kind, tuple(int(rank) for rank in ranks)))

    def _reconcile_group_creation_sequence_before_restore(
            self, world_group, backend: str) -> None:
        world_size = int(world_group.world_size)
        local_active_ranks = self._get_previous_active_ranks_for_shrink(world_size)
        local_stage_tensor = torch.full((world_size + 1, ),
                                        -1,
                                        dtype=torch.int64,
                                        device="cpu")
        local_stage_tensor[0] = len(local_active_ranks)
        if local_active_ranks:
            local_stage_tensor[1:1 + len(local_active_ranks)] = torch.tensor(
                local_active_ranks, dtype=torch.int64, device="cpu")

        gathered_stage_tensors = [
            torch.empty_like(local_stage_tensor) for _ in range(world_size)
        ]
        torch.distributed.all_gather(gathered_stage_tensors,
                                     local_stage_tensor,
                                     group=world_group.cpu_group)

        observed_stages: list[tuple[int, ...]] = []
        for stage_tensor in gathered_stage_tensors:
            stage_len = int(stage_tensor[0].item())
            if stage_len <= 0 or stage_len >= world_size:
                continue
            stage = tuple(
                sorted(
                    int(rank) for rank in stage_tensor[1:1 + stage_len].tolist()
                    if int(rank) >= 0))
            if not stage or stage in observed_stages:
                continue
            observed_stages.append(stage)

        observed_stages.sort(key=lambda ranks: (-len(ranks), ranks))
        current_stage = tuple(local_active_ranks)
        replayed_stages: list[tuple[int, ...]] = []
        for stage in observed_stages:
            if len(stage) >= len(current_stage):
                continue
            if not set(stage).issubset(set(current_stage)):
                continue
            self._advance_group_creation_sequence_for_non_member(
                [list(stage)], backend, "restore_catchup_dp")
            self._advance_group_creation_sequence_for_non_member(
                [list(stage)], backend, "restore_catchup_ep")
            self._advance_group_creation_sequence_for_non_member(
                [list(stage)], backend, "restore_catchup_mc2")
            current_stage = stage
            replayed_stages.append(stage)

        if replayed_stages:
            logger.info(
                "Elastic restore sequence catch-up applied: rank=%s local_stage=%s replayed_stages=%s",
                self.rank, local_active_ranks, replayed_stages)
            self._elastic_current_active_ranks = list(current_stage)

    def _detach_from_elastic_parallel_groups(self) -> None:
        import vllm.distributed.parallel_state as vllm_ps
        import vllm_ascend.distributed.parallel_state as ascend_ps
        from vllm_ascend.ops.fused_moe import AscendFusedMoE

        self._stash_group_if_present(vllm_ps, "_DP")
        self._stash_group_if_present(vllm_ps, "_EP")
        self._stash_group_if_present(ascend_ps, "_MC2")

        model_runner = getattr(self, "model_runner", None)
        if model_runner is not None:
            model_runner.dp_size = 1
            model_runner.dp_rank = 0
            model_runner.parallel_config.data_parallel_size = 1
            model_runner.parallel_config.data_parallel_rank = 0

        model = getattr(model_runner, "model", None) if model_runner else None
        if model is None:
            self.elastic_parallel_detached = True
            return

        for module in model.modules():
            if not isinstance(module, AscendFusedMoE):
                continue
            if hasattr(module, "clear_lossless_hybrid_state"):
                module.clear_lossless_hybrid_state()
            module.set_active_expert_mask(None)
            module.set_elastic_runtime_log2phy(None)
            module.moe_config.num_experts = module.elastic_original_num_experts
            module.ep_group = None
            module.moe_config.dp_group = None
            module.moe_config.ep_group = None
            module.moe_config.mc2_group = None
            module.moe_parallel_config.dp_size = 1
            module.moe_parallel_config.dp_rank = 0
            module.moe_parallel_config.ep_size = 1
            module.moe_parallel_config.ep_rank = 0

        self.elastic_parallel_detached = True
        self._elastic_current_active_ranks = None

    def _rebuild_group(self, state_module, attr_name: str,
                       group_ranks: list[list[int]], world_group,
                       backend: str, group_name: str) -> None:
        self._stash_group_if_present(state_module, attr_name)

        should_cache_group = self._should_cache_elastic_parallel_group(
            attr_name)
        local_group_ranks = self._get_local_group_ranks(group_ranks)
        if should_cache_group:
            cached_groups = self._get_cached_elastic_parallel_groups()
            cached_group = cached_groups.get(attr_name,
                                             {}).get(local_group_ranks)
            if cached_group is not None:
                setattr(state_module, attr_name, cached_group)
                logger.info(
                    "Elastic parallel group cache hit: rank=%s attr=%s group_name=%s ranks=%s",
                    self.rank, attr_name, group_name, local_group_ranks)
                return

        setattr(
            state_module, attr_name,
            init_model_parallel_group(group_ranks, world_group.local_rank,
                                      backend, group_name=group_name))
        if local_group_ranks:
            if should_cache_group:
                cached_groups = self._get_cached_elastic_parallel_groups()
                cached_groups.setdefault(attr_name, {})[
                    local_group_ranks] = getattr(state_module, attr_name)
            group_kind = self._normalize_elastic_parallel_group_kind(attr_name)
            self._get_seen_elastic_parallel_group_signatures().add(
                (group_kind, local_group_ranks))

    def _warmup_post_shrink_dp_collectives(self) -> None:
        dp_group = get_dp_group()
        if dp_group.world_size <= 1:
            return

        # Force HCCL to materialize the new DP communicator/workspace before
        # post-shrink decode hits its first metadata all_reduce.
        warmup_tensor = torch.zeros(1, dtype=torch.int32, device="npu")
        warmup_start_t = time.perf_counter()
        torch.distributed.all_reduce(warmup_tensor, group=dp_group.device_group)
        if torch.npu.is_available():
            torch.npu.synchronize()
        logger.info(
            "Elastic post-shrink DP all_reduce warmup done: rank=%s dp_size=%s total_ms=%.2f",
            self.rank, dp_group.world_size,
            (time.perf_counter() - warmup_start_t) * 1000.0)

    def _has_hybrid_lossless_module(self) -> bool:
        from vllm_ascend.ops.fused_moe import AscendFusedMoE

        model_runner = getattr(self, "model_runner", None)
        model = getattr(model_runner, "model", None) if model_runner else None
        if model is None:
            return False

        for module in model.modules():
            if not isinstance(module, AscendFusedMoE):
                continue
            if getattr(module, "elastic_moe_mode", "lossy") != "lossless":
                continue
            if hasattr(module, "_is_hybrid_cpu_swap_enabled") and \
                    module._is_hybrid_cpu_swap_enabled():
                return True
        return False

    def _has_mode3_hybrid_lossless_module(self) -> bool:
        from vllm_ascend.ops.fused_moe import AscendFusedMoE

        model_runner = getattr(self, "model_runner", None)
        model = getattr(model_runner, "model", None) if model_runner else None
        if model is None:
            return False

        for module in model.modules():
            if not isinstance(module, AscendFusedMoE):
                continue
            if getattr(module, "elastic_moe_mode", "lossy") != "lossless":
                continue
            if getattr(module, "elastic_execution_mode", 0) != 3:
                continue
            if hasattr(module, "_is_hybrid_cpu_swap_enabled") and \
                    module._is_hybrid_cpu_swap_enabled():
                return True
        return False

    def _warmup_post_shrink_moe_dispatch(self,
                                         active_ranks: list[int] | None = None
                                         ) -> None:
        model_runner = getattr(self, "model_runner", None)
        if model_runner is None:
            return
        if not self.parallel_config.enable_expert_parallel:
            return
        active_signature = tuple(active_ranks or [])
        repeat_warmup = bool(int(
            os.getenv("VLLM_ASCEND_REPEAT_POST_SHRINK_MOE_DISPATCH_WARMUP",
                      "0")))
        if (active_signature
                and active_signature in
                self._post_shrink_moe_dispatch_warmed_active_signatures
                and not repeat_warmup):
            logger.info(
                "Elastic post-shrink MoE dispatch warmup skipped: rank=%s active_ranks=%s reason=already_warmed",
                self.rank, list(active_signature))
            return

        warmup_tokens = int(
            os.getenv("VLLM_ASCEND_POST_SHRINK_MOE_DISPATCH_WARMUP_TOKENS",
                      "32"))
        if warmup_tokens <= 0:
            return

        if self._has_mode3_hybrid_lossless_module():
            if active_signature:
                self._post_shrink_moe_dispatch_warmed_active_signatures.add(
                    active_signature)
            logger.info(
                "Elastic post-shrink MoE dispatch warmup skipped: rank=%s active_ranks=%s reason=mode3_lazy_init",
                self.rank, list(active_signature))
            return

        warmup_start_t = time.perf_counter()
        model_runner._dummy_run(warmup_tokens, with_prefill=False)
        if torch.npu.is_available():
            torch.npu.synchronize()
        if active_signature:
            self._post_shrink_moe_dispatch_warmed_active_signatures.add(
                active_signature)
        logger.info(
            "Elastic post-shrink MoE dispatch warmup done: rank=%s tokens=%s total_ms=%.2f",
            self.rank, warmup_tokens,
            (time.perf_counter() - warmup_start_t) * 1000.0)

    def _release_post_shrink_staging_state(self) -> None:
        payload_layers = len(getattr(self, "_lossless_shrink_payload", {}))
        import_layers = len(
            getattr(self, "_lossless_preloaded_cpu_import_weights", {}))
        direct_import_layers = len(
            getattr(self, "_lossless_preloaded_direct_import_slots", {}))
        self._lossless_shrink_payload = {}
        self._lossless_preloaded_cpu_import_weights = {}
        self._lossless_preloaded_direct_import_slots = {}

        # Best-effort cleanup before HCCL warmup. The warmup tensor is tiny; the
        # real risk is peak memory from shrink/import staging that is no longer
        # needed after elastic state refresh.
        import gc
        gc.collect()
        if torch.npu.is_available():
            torch.npu.empty_cache()
            torch.npu.synchronize()

        pass  # debug log removed

    def rebuild_elastic_ep_group(self, active_global_ranks: list[int]) -> bool:
        if not envs_ascend.VLLM_ASCEND_ENABLE_ELASTIC_PARALLEL_SHRINK:
            return False
        if not torch.distributed.is_initialized():
            return False

        import vllm.distributed.parallel_state as vllm_ps
        import vllm_ascend.distributed.parallel_state as ascend_ps

        start_t = time.perf_counter()
        world_group = get_world_group()
        world_size = torch.distributed.get_world_size()
        active_ranks = sorted(set(active_global_ranks))
        if not active_ranks:
            active_ranks = [torch.distributed.get_rank()]
        current_rank = torch.distributed.get_rank()
        backend = torch.distributed.get_backend(world_group.device_group)
        cached_active_ranks = getattr(self, "_elastic_current_active_ranks",
                                      None)
        if cached_active_ranks == active_ranks:
            if current_rank in active_ranks and not self.elastic_parallel_detached:
                logger.info(
                    "Elastic parallel shrink skipped: rank=%s active_ranks=%s already active",
                    self.rank, active_ranks)
                return True
            if current_rank not in active_ranks and self.elastic_parallel_detached:
                logger.info(
                    "Elastic parallel shrink skipped: rank=%s active_ranks=%s already detached",
                    self.rank, active_ranks)
                return True
        self._prepare_lossless_shrink_payload(active_ranks, world_group)
        self._preload_lossless_shrink_import_weights(active_ranks, world_group)
        is_active_rank = current_rank in active_ranks
        if not is_active_rank and not self.elastic_parallel_detached:
            detach_start_t = time.perf_counter()
            self._detach_from_elastic_parallel_groups()
            logger.info(
                "Elastic parallel detach done: rank=%s active_ranks=%s total_ms=%.2f",
                self.rank, active_ranks,
                (time.perf_counter() - detach_start_t) * 1000.0)
            elastic_group_ranks = [active_ranks]
            self._advance_group_creation_sequence_for_non_member(
                elastic_group_ranks, backend, "dp")
            self._advance_group_creation_sequence_for_non_member(
                elastic_group_ranks, backend, "ep")
            self._advance_group_creation_sequence_for_non_member(
                elastic_group_ranks, backend, "mc2")

        rebuild_ms = 0.0
        refresh_ms = 0.0
        warmup_ms = 0.0
        if is_active_rank:
            elastic_group_ranks = [active_ranks]
            rebuild_start_t = time.perf_counter()
            with set_current_vllm_config(self.vllm_config):
                self._rebuild_group(vllm_ps, "_DP", elastic_group_ranks,
                                    world_group, backend, "dp")
                self._rebuild_group(vllm_ps, "_EP", elastic_group_ranks,
                                    world_group, backend, "ep")
                self._rebuild_group(ascend_ps, "_MC2", elastic_group_ranks,
                                    world_group, backend, "mc2")
            rebuild_ms = (time.perf_counter() - rebuild_start_t) * 1000.0

        refresh_start_t = time.perf_counter()
        with set_current_vllm_config(self.vllm_config):
            self._refresh_elastic_parallel_state(active_ranks,
                                                 world_group,
                                                 participate_only=not is_active_rank)
        refresh_ms = (time.perf_counter() - refresh_start_t) * 1000.0

        self._release_post_shrink_staging_state()

        if is_active_rank:
            warmup_start_t = time.perf_counter()
            self._warmup_post_shrink_dp_collectives()
            self._warmup_post_shrink_moe_dispatch(active_ranks)
            warmup_ms = (time.perf_counter() - warmup_start_t) * 1000.0

        if is_active_rank:
            logger.info(
                "Elastic parallel shrink done: rank=%s active_ranks=%s dp_size=%s ep_size=%s no_ep_tail=%s rebuild_ms=%.2f refresh_ms=%.2f warmup_ms=%.2f total_ms=%.2f",
                self.rank, active_ranks, get_dp_group().world_size,
                vllm_ps.get_ep_group().world_size, len(active_ranks) == 1,
                rebuild_ms, refresh_ms,
                warmup_ms,
                (time.perf_counter() - start_t) * 1000.0)
        self._elastic_current_active_ranks = active_ranks
        return True

    def restore_elastic_parallel_groups(self) -> bool:
        if not envs_ascend.VLLM_ASCEND_ENABLE_ELASTIC_PARALLEL_SHRINK:
            return False
        if not torch.distributed.is_initialized():
            return False

        import vllm.distributed.parallel_state as vllm_ps
        import vllm_ascend.distributed.parallel_state as ascend_ps

        start_t = time.perf_counter()
        world_group = get_world_group()
        world_size = torch.distributed.get_world_size()
        backend = torch.distributed.get_backend(world_group.device_group)

        self._reconcile_group_creation_sequence_before_restore(
            world_group, backend)

        rebuild_start_t = time.perf_counter()
        with set_current_vllm_config(self.vllm_config):
            self._rebuild_group(vllm_ps, "_DP",
                                self._build_original_dp_group_ranks(world_size),
                                world_group, backend, "dp")
            self._rebuild_group(vllm_ps, "_EP",
                                self._build_original_ep_group_ranks(world_size),
                                world_group, backend, "ep")
            self._rebuild_group(
                ascend_ps, "_MC2",
                self._build_original_mc2_group_ranks(world_size),
                world_group, backend, "mc2")
        rebuild_ms = (time.perf_counter() - rebuild_start_t) * 1000.0

        refresh_start_t = time.perf_counter()
        with set_current_vllm_config(self.vllm_config):
            self._refresh_elastic_parallel_state(list(range(world_size)),
                                                 world_group)
        refresh_ms = (time.perf_counter() - refresh_start_t) * 1000.0
        self.elastic_parallel_detached = False
        self._elastic_current_active_ranks = list(range(world_size))
        self._drop_stale_cached_elastic_parallel_groups(
            tuple(range(world_size)))

        logger.info(
            "Elastic parallel restore done: rank=%s dp_size=%s ep_size=%s rebuild_ms=%.2f refresh_ms=%.2f total_ms=%.2f",
            self.rank, get_dp_group().world_size, vllm_ps.get_ep_group().world_size,
            rebuild_ms, refresh_ms, (time.perf_counter() - start_t) * 1000.0)
        return True

    def set_need_allreduce(self, value: bool):
        self.model_runner.need_allreduce = value
        return True
