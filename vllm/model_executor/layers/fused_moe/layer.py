# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from abc import abstractmethod
from collections.abc import Iterable
from contextlib import nullcontext
from enum import Enum
from typing import Callable, Literal, Optional, Union, get_args, overload
import os
import time

import torch
import torch.nn.functional as F
from torch.nn.parameter import UninitializedParameter

import vllm.envs as envs
from vllm.config import get_current_vllm_config
from vllm.config.parallel import ExpertPlacementStrategy
from vllm.distributed import (get_dp_group, get_ep_group,
                              get_tensor_model_parallel_world_size,
                              tensor_model_parallel_all_reduce)
from vllm.distributed.eplb.eplb_state import EplbState
from vllm.forward_context import ForwardContext, get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.custom_op import CustomOp
# yapf: disable
from vllm.model_executor.layers.fused_moe.config import (
    FUSED_MOE_UNQUANTIZED_CONFIG, FusedMoEConfig, FusedMoEParallelConfig,
    FusedMoEQuantConfig, biased_moe_quant_config)
from vllm.model_executor.layers.fused_moe.fused_moe import (
    zero_experts_compute_triton)
# yapf: enable
from vllm.model_executor.layers.fused_moe.modular_kernel import (
    FusedMoEActivationFormat, FusedMoEModularKernel,
    FusedMoEPermuteExpertsUnpermute, FusedMoEPrepareAndFinalize)
from vllm.model_executor.layers.fused_moe.rocm_aiter_fused_moe import (
    is_rocm_aiter_moe_enabled)
from vllm.model_executor.layers.fused_moe.routing_simulator import (
    RoutingSimulator)
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig, QuantizeMethodBase)
from vllm.model_executor.utils import set_weight_attrs
from vllm.platforms import current_platform
from vllm.platforms.interface import CpuArchEnum
from vllm.utils import (cdiv, direct_register_custom_op, has_deep_ep, has_pplx,
                        round_up)
from vllm.utils.flashinfer import has_flashinfer_cutlass_fused_moe
from vllm.v1.worker.ubatching import dbo_current_ubatch_id
#新增开始
from vllm.forward_context import get_forward_context
#新增结束

if current_platform.is_cuda_alike():
    from .fused_batched_moe import BatchedTritonExperts
    from .fused_moe import (TritonExperts, eplb_map_to_physical_and_record,
                            fused_experts)
    if has_pplx():
        from .pplx_prepare_finalize import (PplxPrepareAndFinalize,
                                            pplx_hidden_dim_scale_bytes)
    if has_deep_ep():
        from .deepep_ht_prepare_finalize import DeepEPHTPrepareAndFinalize
        from .deepep_ll_prepare_finalize import (DEEPEP_QUANT_BLOCK_SHAPE,
                                                 DeepEPLLPrepareAndFinalize)
else:
    fused_experts = None  # type: ignore
    FusedMoEPermuteExpertsUnpermute = None  # type: ignore
    FusedMoEPrepareAndFinalize = None  # type: ignore

    def _eplb_map_to_physical_and_record(
            topk_ids: torch.Tensor, expert_load_view: torch.Tensor,
            logical_to_physical_map: torch.Tensor,
            logical_replica_count: torch.Tensor,
            indices_type: Optional[torch.dtype]) -> torch.Tensor:
        # CPU fallback: no EPLB so just return as is
        return topk_ids

    eplb_map_to_physical_and_record = _eplb_map_to_physical_and_record

if is_rocm_aiter_moe_enabled():
    from vllm.model_executor.layers.fused_moe.rocm_aiter_fused_moe import (  # noqa: E501
        rocm_aiter_grouped_topk as grouped_topk)
else:
    from vllm.model_executor.layers.fused_moe.fused_moe import grouped_topk
if current_platform.is_tpu():
    from .moe_pallas import fused_moe as fused_moe_pallas
else:
    fused_moe_pallas = None  # type: ignore

logger = init_logger(__name__)

class FusedMoeWeightScaleSupported(Enum):
    TENSOR = "tensor"
    CHANNEL = "channel"
    GROUP = "group"
    BLOCK = "block"


class FusedMoEMethodBase(QuantizeMethodBase):

    def __init__(self, moe: FusedMoEConfig):
        super().__init__()
        self.moe = moe
        self.moe_quant_config: Optional[FusedMoEQuantConfig] = None
        self.fused_experts: Optional[FusedMoEModularKernel] = None
        self.topk_indices_dtype = None

    @abstractmethod
    def create_weights(self, layer: torch.nn.Module, num_experts: int,
                       hidden_size: int, intermediate_size_per_partition: int,
                       params_dtype: torch.dtype, **extra_weight_attrs):
        raise NotImplementedError

    def uses_weight_scale_2_pattern(self) -> bool:
        """
        Returns True if this quantization method uses 'weight_scale_2' pattern
        for per-tensor weight scales (e.g., FP4 variants), False otherwise.

        This method should be overridden by subclasses that use the
        'weight_scale_2' pattern instead of the standard 'weight_scale' pattern.
        """
        return False

    @staticmethod
    def _maybe_make_prepare_finalize(
        moe: FusedMoEConfig,
        quant_config: Optional[FusedMoEQuantConfig],
    ) -> Optional[FusedMoEPrepareAndFinalize]:
        all2all_manager = get_ep_group().device_communicator.all2all_manager
        assert all2all_manager is not None

        prepare_finalize: Optional[FusedMoEPrepareAndFinalize] = None

        # TODO: could allow this now
        assert not moe.use_flashinfer_cutlass_kernels, \
            "Must be created in modelopt.py"

        if moe.use_pplx_kernels:
            assert quant_config is not None

            hidden_dim_bytes, hidden_scale_bytes = pplx_hidden_dim_scale_bytes(
                moe.max_num_tokens,
                moe.hidden_dim,
                moe.in_dtype,
                quant_config.quant_dtype,
                per_act_token_quant=quant_config.per_act_token_quant,
                block_shape=quant_config.block_shape,
            )

            all_to_all_args = dict(
                max_num_tokens=moe.max_num_tokens,
                num_experts=moe.num_experts,
                experts_per_token=moe.experts_per_token,  # topk
                rank=all2all_manager.rank,
                world_size=all2all_manager.world_size,
                # dp_size actually means tp_size, bug in pplx kernels
                dp_size=all2all_manager.tp_group.world_size,
                hidden_dim=moe.hidden_dim,
                hidden_dim_bytes=hidden_dim_bytes,
                hidden_dim_scale_bytes=hidden_scale_bytes,
            )

            num_dispatchers = (all2all_manager.world_size //
                               all2all_manager.tp_group.world_size)

            # Intranode pplx a2a takes a group name while internode does not.
            if not all2all_manager.internode:
                all_to_all_args[
                    "group_name"] = all2all_manager.cpu_group.group_name

            handle = all2all_manager.get_handle(all_to_all_args)

            prepare_finalize = PplxPrepareAndFinalize(
                handle,
                max_num_tokens=moe.max_num_tokens,
                num_local_experts=moe.num_local_experts,
                num_dispatchers=num_dispatchers,
            )
        elif moe.use_deepep_ht_kernels:
            assert moe.dp_size == all2all_manager.dp_world_size

            all_to_all_args = dict()
            handle = all2all_manager.get_handle(all_to_all_args)
            prepare_finalize = DeepEPHTPrepareAndFinalize(
                handle,
                num_dispatchers=all2all_manager.world_size,
                dp_size=all2all_manager.dp_world_size,
                rank_expert_offset=all2all_manager.rank *
                moe.num_local_experts,
            )

        elif moe.use_deepep_ll_kernels:
            assert quant_config is not None
            all_to_all_args = dict(
                max_num_tokens_per_dp_rank=moe.max_num_tokens,
                token_hidden_size=moe.hidden_dim,
                num_ep_ranks=all2all_manager.world_size,
                num_global_experts=moe.num_experts,
                num_local_experts=moe.num_experts //
                all2all_manager.world_size)
            handle = all2all_manager.get_handle(all_to_all_args)

            # Note: We may want to use FP8 dispatch just to reduce
            # data movement.
            use_fp8_dispatch = (
                quant_config.quant_dtype == current_platform.fp8_dtype()
                and quant_config.block_shape == DEEPEP_QUANT_BLOCK_SHAPE)

            prepare_finalize = DeepEPLLPrepareAndFinalize(
                handle,
                max_tokens_per_rank=moe.max_num_tokens,
                num_dispatchers=all2all_manager.world_size,
                use_fp8_dispatch=use_fp8_dispatch,
            )

        return prepare_finalize

    def maybe_make_prepare_finalize(
            self) -> Optional[FusedMoEPrepareAndFinalize]:
        if self.moe.moe_parallel_config.use_all2all_kernels:
            return FusedMoEMethodBase._maybe_make_prepare_finalize(
                self.moe, self.moe_quant_config)
        else:
            return None

    # Note: init_prepare_finalize should only be called by
    # prepare_communication_buffer_for_model.
    def init_prepare_finalize(self, layer: torch.nn.Module):
        assert self.moe is not None

        # We must get the quant config here so that the layer is
        # completely initialized, i.e. all weights loaded and post
        # processed.
        self.moe_quant_config = self.get_fused_moe_quant_config(layer)

        prepare_finalize = self.maybe_make_prepare_finalize()

        if prepare_finalize is not None:
            logger.debug("%s for %s(%s)", prepare_finalize.__class__.__name__,
                         self, id(self))
            assert self.topk_indices_dtype is None
            assert self.fused_experts is None, \
                f"Attempt to override experts for {id(self)}!"
            self.topk_indices_dtype = prepare_finalize.topk_indices_dtype()
            experts = self.select_gemm_impl(prepare_finalize, layer)
            self.fused_experts = FusedMoEModularKernel(
                prepare_finalize,
                experts,
                layer.shared_experts,
            )

    def select_gemm_impl(
        self,
        prepare_finalize: FusedMoEPrepareAndFinalize,
        layer: torch.nn.Module,
    ) -> FusedMoEPermuteExpertsUnpermute:
        # based on the all2all implementation, select the appropriate
        # gemm implementation
        raise NotImplementedError(
            f"{self.__class__.__name__} must select appropriate gemm "
            "implementation based on the prepare_finalize")

    @abstractmethod
    def get_fused_moe_quant_config(
            self, layer: torch.nn.Module) -> Optional[FusedMoEQuantConfig]:
        raise NotImplementedError

    @abstractmethod
    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        router_logits: torch.Tensor,
        top_k: int,
        renormalize: bool,
        use_grouped_topk: bool = False,
        topk_group: Optional[int] = None,
        num_expert_group: Optional[int] = None,
        global_num_experts: int = -1,
        expert_map: Optional[torch.Tensor] = None,
        custom_routing_function: Optional[Callable] = None,
        scoring_func: str = "softmax",
        routed_scaling_factor: float = 1.0,
        e_score_correction_bias: Optional[torch.Tensor] = None,
        apply_router_weight_on_input: bool = False,
        activation: str = "silu",
        enable_eplb: bool = False,
        expert_load_view: Optional[torch.Tensor] = None,
        logical_to_physical_map: Optional[torch.Tensor] = None,
        logical_replica_count: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        raise NotImplementedError


@CustomOp.register("unquantized_fused_moe")
class UnquantizedFusedMoEMethod(FusedMoEMethodBase, CustomOp):
    """MoE method without quantization."""

    def __init__(self, moe: FusedMoEConfig):
        super().__init__(moe)
        self.rocm_aiter_moe_enabled = is_rocm_aiter_moe_enabled()
        if self.rocm_aiter_moe_enabled:
            from .rocm_aiter_fused_moe import rocm_aiter_fused_experts
            self.rocm_aiter_fused_experts = rocm_aiter_fused_experts
        else:
            self.rocm_aiter_fused_experts = None  # type: ignore

        # FlashInfer CUTLASS MoE is only supported on Hopper and later GPUS
        self.flashinfer_cutlass_moe_enabled = (
            has_flashinfer_cutlass_fused_moe()
            and envs.VLLM_USE_FLASHINFER_MOE_FP16
            and self.moe.moe_parallel_config.use_ep
            and self.moe.moe_parallel_config.dp_size == 1
            and current_platform.get_device_capability()[0] >= 9)
        if self.flashinfer_cutlass_moe_enabled:
            logger.info_once(
                "Enabling FlashInfer CUTLASS MoE for UnquantizedFusedMoEMethod"
            )
            from functools import partial

            from .flashinfer_cutlass_moe import flashinfer_cutlass_moe
            self.flashinfer_cutlass_moe = partial(
                flashinfer_cutlass_moe,
                quant_config=FUSED_MOE_UNQUANTIZED_CONFIG,
                tp_rank=self.moe.moe_parallel_config.tp_rank,
                tp_size=self.moe.moe_parallel_config.tp_size,
                ep_rank=self.moe.moe_parallel_config.ep_rank,
                ep_size=self.moe.moe_parallel_config.ep_size)
        else:
            if (self.moe.moe_parallel_config.use_ep
                    and self.moe.moe_parallel_config.dp_size == 1):
                logger.info_once(
                    "FlashInfer CUTLASS MoE is available for EP"
                    " but not enabled, consider setting"
                    " VLLM_USE_FLASHINFER_MOE_FP16=1 to enable it.")
            elif self.moe.moe_parallel_config.dp_size > 1:
                logger.info_once(
                    "FlashInfer CUTLASS MoE is currently not available for DP."
                )
            self.flashinfer_cutlass_moe = None  # type: ignore

    def maybe_make_prepare_finalize(
            self) -> Optional[FusedMoEPrepareAndFinalize]:
        if self.rocm_aiter_moe_enabled:
            return None
        else:
            return super().maybe_make_prepare_finalize()

    def select_gemm_impl(
        self,
        prepare_finalize: FusedMoEPrepareAndFinalize,
        layer: torch.nn.Module,
    ) -> FusedMoEPermuteExpertsUnpermute:
        assert self.moe_quant_config is not None
        if (prepare_finalize.activation_format ==
                FusedMoEActivationFormat.BatchedExperts):
            logger.debug("BatchedTritonExperts %s", self.moe)
            return BatchedTritonExperts(
                max_num_tokens=self.moe.max_num_tokens,
                num_dispatchers=prepare_finalize.num_dispatchers(),
                quant_config=self.moe_quant_config,
            )
        else:
            logger.debug("TritonExperts %s", self.moe)
            return TritonExperts(self.moe_quant_config)

    def create_weights(self, layer: torch.nn.Module, num_experts: int,
                       hidden_size: int, intermediate_size_per_partition: int,
                       params_dtype: torch.dtype, **extra_weight_attrs):
        # Fused gate_up_proj (column parallel)
        w13_weight = torch.nn.Parameter(torch.empty(
            num_experts,
            2 * intermediate_size_per_partition,
            hidden_size,
            dtype=params_dtype),
                                        requires_grad=False)
        layer.register_parameter("w13_weight", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)
        if self.moe.has_bias:
            w13_bias = torch.nn.Parameter(torch.zeros(
                num_experts,
                2 * intermediate_size_per_partition,
                dtype=params_dtype),
                                          requires_grad=False)
            layer.register_parameter("w13_bias", w13_bias)
            set_weight_attrs(w13_bias, extra_weight_attrs)
        # down_proj (row parallel)
        w2_weight = torch.nn.Parameter(torch.empty(
            num_experts,
            hidden_size,
            intermediate_size_per_partition,
            dtype=params_dtype),
                                       requires_grad=False)
        layer.register_parameter("w2_weight", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)
        if self.moe.has_bias:
            w2_bias = torch.nn.Parameter(torch.zeros(num_experts,
                                                     hidden_size,
                                                     dtype=params_dtype),
                                         requires_grad=False)
            layer.register_parameter("w2_bias", w2_bias)
            set_weight_attrs(w2_bias, extra_weight_attrs)

    def _maybe_pad_weight(self, weight: torch.Tensor) -> torch.Tensor:
        # Pad the weight tensor. This is an optimization on ROCm platform, which
        # can benefit from tensors located far enough from one another in memory
        if (envs.VLLM_ROCM_MOE_PADDING and current_platform.is_rocm()
                and weight.stride(-1) == 1
                and (weight.stride(-2) * weight.element_size()) % 512 == 0):
            num_pad = 256 // weight.element_size()
            weight = F.pad(weight, (0, num_pad), "constant", 0)[..., :-num_pad]
            torch.cuda.empty_cache()

        return weight

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        super().process_weights_after_loading(layer)

        # Padding the weight for better performance on ROCm
        layer.w13_weight.data = self._maybe_pad_weight(layer.w13_weight.data)
        layer.w2_weight.data = self._maybe_pad_weight(layer.w2_weight.data)
        # Lazy import to avoid importing triton.
        from vllm.model_executor.layers.fused_moe.rocm_aiter_fused_moe import (
            shuffle_weights)

        if self.rocm_aiter_moe_enabled:
            shuffled_w13, shuffled_w2 = shuffle_weights(
                layer.w13_weight.data, layer.w2_weight.data)

            layer.w13_weight.data = shuffled_w13
            layer.w2_weight.data = shuffled_w2

        if self.flashinfer_cutlass_moe_enabled:
            # Swap halves to arrange as [w3; w1] (kernel expectation)
            w1_w, w3_w = torch.chunk(layer.w13_weight.data, 2, dim=1)
            w13_weight_swapped = torch.cat([w3_w, w1_w], dim=1)
            layer.w13_weight.data = w13_weight_swapped.contiguous()

        if current_platform.is_xpu():
            import intel_extension_for_pytorch as ipex
            layer.ipex_fusion = ipex.llm.modules.GatedMLPMOE(
                layer.w13_weight,
                layer.w2_weight,
                use_prepack=True,
            )
        elif current_platform.is_cpu():
            from vllm.model_executor.layers.fused_moe import cpu_fused_moe
            if current_platform.get_cpu_architecture() == CpuArchEnum.X86:
                from vllm.model_executor.layers.utils import (
                    check_cpu_sgl_kernel)
                dtype_w13 = layer.w13_weight.dtype
                _, n_w13, k_w13 = layer.w13_weight.size()
                dtype_w2 = layer.w2_weight.dtype
                _, n_w2, k_w2 = layer.w2_weight.size()
                if (envs.VLLM_CPU_SGL_KERNEL
                        and check_cpu_sgl_kernel(n_w13, k_w13, dtype_w13)
                        and check_cpu_sgl_kernel(n_w2, k_w2, dtype_w2)):
                    packed_w13_weight = torch.ops._C.convert_weight_packed(
                        layer.w13_weight)
                    assert packed_w13_weight.size() == layer.w13_weight.size()
                    layer.w13_weight.copy_(packed_w13_weight)
                    del packed_w13_weight
                    packed_w2_weight = torch.ops._C.convert_weight_packed(
                        layer.w2_weight)
                    assert packed_w2_weight.size() == layer.w2_weight.size()
                    layer.w2_weight.copy_(packed_w2_weight)
                    layer.cpu_fused_moe = cpu_fused_moe.SGLFusedMOE(layer)
                else:
                    layer.cpu_fused_moe = cpu_fused_moe.IPEXFusedMOE(layer)
            else:
                layer.cpu_fused_moe = cpu_fused_moe.CPUFusedMOE(layer)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        router_logits: torch.Tensor,
        top_k: int,
        renormalize: bool,
        use_grouped_topk: bool = False,
        topk_group: Optional[int] = None,
        num_expert_group: Optional[int] = None,
        global_num_experts: int = -1,
        expert_map: Optional[torch.Tensor] = None,
        custom_routing_function: Optional[Callable] = None,
        scoring_func: str = "softmax",
        routed_scaling_factor: float = 1.0,
        e_score_correction_bias: Optional[torch.Tensor] = None,
        apply_router_weight_on_input: bool = False,
        activation: str = "silu",
        enable_eplb: bool = False,
        expert_load_view: Optional[torch.Tensor] = None,
        logical_to_physical_map: Optional[torch.Tensor] = None,
        logical_replica_count: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        if enable_eplb:
            assert expert_load_view is not None
            assert logical_to_physical_map is not None
            assert logical_replica_count is not None
            assert isinstance(layer, FusedMoE)

        return self.forward(
            x=x,
            layer=layer,
            router_logits=router_logits,
            top_k=top_k,
            renormalize=renormalize,
            use_grouped_topk=use_grouped_topk,
            topk_group=topk_group,
            num_expert_group=num_expert_group,
            global_num_experts=global_num_experts,
            expert_map=expert_map,
            custom_routing_function=custom_routing_function,
            scoring_func=scoring_func,
            routed_scaling_factor=routed_scaling_factor,
            e_score_correction_bias=e_score_correction_bias,
            activation=activation,
            apply_router_weight_on_input=apply_router_weight_on_input,
            enable_eplb=enable_eplb,
            expert_load_view=expert_load_view,
            logical_to_physical_map=logical_to_physical_map,
            logical_replica_count=logical_replica_count,
        )

    def get_fused_moe_quant_config(
            self, layer: torch.nn.Module) -> Optional[FusedMoEQuantConfig]:
        if self.moe.has_bias:
            return biased_moe_quant_config(
                layer.w13_bias,
                layer.w2_bias,
            )
        else:
            return FUSED_MOE_UNQUANTIZED_CONFIG

    def forward_cuda(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        use_grouped_topk: bool,
        top_k: int,
        router_logits: torch.Tensor,
        renormalize: bool,
        topk_group: Optional[int] = None,
        num_expert_group: Optional[int] = None,
        global_num_experts: int = -1,
        expert_map: Optional[torch.Tensor] = None,
        custom_routing_function: Optional[Callable] = None,
        scoring_func: str = "softmax",
        routed_scaling_factor: float = 1.0,
        e_score_correction_bias: Optional[torch.Tensor] = None,
        apply_router_weight_on_input: bool = False,
        activation: str = "silu",
        enable_eplb: bool = False,
        expert_load_view: Optional[torch.Tensor] = None,
        logical_to_physical_map: Optional[torch.Tensor] = None,
        logical_replica_count: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:

        zero_expert_num = getattr(layer, 'zero_expert_num', 0)
        zero_expert_type = getattr(layer, 'zero_expert_type', None)

        topk_weights, topk_ids, zero_expert_result = FusedMoE.select_experts(
            hidden_states=x,
            router_logits=router_logits,
            use_grouped_topk=use_grouped_topk,
            top_k=top_k,
            renormalize=renormalize,
            topk_group=topk_group,
            num_expert_group=num_expert_group,
            custom_routing_function=custom_routing_function,
            scoring_func=scoring_func,
            routed_scaling_factor=routed_scaling_factor,
            e_score_correction_bias=e_score_correction_bias,
            indices_type=self.topk_indices_dtype,
            enable_eplb=enable_eplb,
            expert_map=expert_map,
            expert_load_view=expert_load_view,
            logical_to_physical_map=logical_to_physical_map,
            logical_replica_count=logical_replica_count,
            global_num_experts=global_num_experts,
            zero_expert_num=zero_expert_num,
            zero_expert_type=zero_expert_type,
            layer_idx=layer.layer_idx,
            )

        if self.rocm_aiter_moe_enabled:
            assert self.fused_experts is None
            result = self.rocm_aiter_fused_experts(
                hidden_states=x,
                w1=layer.w13_weight,
                w2=layer.w2_weight,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                expert_map=expert_map,
                activation=activation,
                apply_router_weight_on_input=apply_router_weight_on_input)
        elif self.flashinfer_cutlass_moe_enabled:
            return self.flashinfer_cutlass_moe(
                hidden_states=x,
                w1=layer.w13_weight,
                w2=layer.w2_weight,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                activation=activation,
                apply_router_weight_on_input=apply_router_weight_on_input)
        elif self.fused_experts is not None:
            if self.moe.has_bias:
                raise ValueError(
                    "FusedMoEModularKernel does not support bias.")
            result = self.fused_experts(
                hidden_states=x,
                w1=layer.w13_weight,
                w2=layer.w2_weight,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                inplace=True,
                activation=activation,
                apply_router_weight_on_input=apply_router_weight_on_input,
                global_num_experts=global_num_experts,
                expert_map=expert_map,
            )
        else:
            assert fused_experts is not None
            result = fused_experts(
                hidden_states=x,
                w1=layer.w13_weight,
                w2=layer.w2_weight,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                inplace=True,
                activation=activation,
                quant_config=self.moe_quant_config,
                apply_router_weight_on_input=apply_router_weight_on_input,
                global_num_experts=global_num_experts,
                expert_map=expert_map,
            )

        if zero_expert_num != 0 and zero_expert_type is not None:
            assert not isinstance(result, tuple), \
                "Shared + zero experts are mutually exclusive not yet supported"
            return result, zero_expert_result
        else:
            return result

    def forward_cpu(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        use_grouped_topk: bool,
        top_k: int,
        router_logits: torch.Tensor,
        renormalize: bool,
        topk_group: Optional[int] = None,
        num_expert_group: Optional[int] = None,
        global_num_experts: int = -1,
        expert_map: Optional[torch.Tensor] = None,
        custom_routing_function: Optional[Callable] = None,
        scoring_func: str = "softmax",
        routed_scaling_factor: float = 1.0,
        e_score_correction_bias: Optional[torch.Tensor] = None,
        apply_router_weight_on_input: bool = False,
        activation: str = "silu",
        enable_eplb: bool = False,
        expert_load_view: Optional[torch.Tensor] = None,
        logical_to_physical_map: Optional[torch.Tensor] = None,
        logical_replica_count: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        if enable_eplb is not False or expert_load_view is not None or \
                logical_to_physical_map is not None or \
                logical_replica_count is not None:
            raise NotImplementedError("Expert load balancing is not supported "
                                      "for CPU.")
        return layer.cpu_fused_moe(
            layer,
            x,
            use_grouped_topk,
            top_k,
            router_logits,
            renormalize,
            topk_group,
            num_expert_group,
            global_num_experts,
            expert_map,
            custom_routing_function,
            scoring_func,
            routed_scaling_factor,
            e_score_correction_bias,
            apply_router_weight_on_input,
            activation,
        )

    def forward_xpu(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        use_grouped_topk: bool,
        top_k: int,
        router_logits: torch.Tensor,
        renormalize: bool,
        topk_group: Optional[int] = None,
        num_expert_group: Optional[int] = None,
        global_num_experts: int = -1,
        expert_map: Optional[torch.Tensor] = None,
        custom_routing_function: Optional[Callable] = None,
        scoring_func: str = "softmax",
        routed_scaling_factor: float = 1.0,
        e_score_correction_bias: Optional[torch.Tensor] = None,
        apply_router_weight_on_input: bool = False,
        activation: str = "silu",
        enable_eplb: bool = False,
        expert_load_view: Optional[torch.Tensor] = None,
        logical_to_physical_map: Optional[torch.Tensor] = None,
        logical_replica_count: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        if enable_eplb is not False or expert_load_view is not None or \
                logical_to_physical_map is not None or \
                logical_replica_count is not None:
            raise NotImplementedError("Expert load balancing is not supported "
                                      "for XPU.")
        assert custom_routing_function is None
        return layer.ipex_fusion(
            x,
            use_grouped_topk,
            top_k,
            router_logits,
            renormalize,
            topk_group,
            num_expert_group,
        )

    def forward_tpu(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        use_grouped_topk: bool,
        top_k: int,
        router_logits: torch.Tensor,
        renormalize: bool,
        topk_group: Optional[int] = None,
        num_expert_group: Optional[int] = None,
        global_num_experts: int = -1,
        expert_map: Optional[torch.Tensor] = None,
        custom_routing_function: Optional[Callable] = None,
        scoring_func: str = "softmax",
        routed_scaling_factor: float = 1.0,
        e_score_correction_bias: Optional[torch.Tensor] = None,
        apply_router_weight_on_input: bool = False,
        activation: str = "silu",
        enable_eplb: bool = False,
        expert_load_view: Optional[torch.Tensor] = None,
        logical_to_physical_map: Optional[torch.Tensor] = None,
        logical_replica_count: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        assert not use_grouped_topk
        assert num_expert_group is None
        assert topk_group is None
        assert custom_routing_function is None
        assert apply_router_weight_on_input is False
        if scoring_func != "softmax":
            raise NotImplementedError(
                "Only softmax scoring function is supported for TPU.")
        if e_score_correction_bias is not None:
            raise NotImplementedError(
                "Expert score correction bias is not supported for TPU.")
        assert activation == "silu", f"{activation} is not supported for TPU."
        assert routed_scaling_factor == 1.0, \
            f"routed_scaling_factor {routed_scaling_factor} is not supported " \
            f"for TPU."
        if enable_eplb is not False or expert_load_view is not None or \
                logical_to_physical_map is not None or \
                logical_replica_count is not None:
            raise NotImplementedError("Expert load balancing is not supported "
                                      "for TPU.")
        return fused_moe_pallas(hidden_states=x,
                                w1=layer.w13_weight,
                                w2=layer.w2_weight,
                                topk=top_k,
                                gating_output=router_logits,
                                global_num_experts=global_num_experts,
                                expert_map=expert_map,
                                renormalize=renormalize)

    if current_platform.is_tpu():
        forward_native = forward_tpu
    elif current_platform.is_cpu():
        forward_native = forward_cpu
    elif current_platform.is_xpu():
        forward_native = forward_xpu
    else:
        forward_native = forward_cuda


def determine_expert_map(
    ep_size: int,
    ep_rank: int,
    global_num_experts: int,
    expert_placement_strategy: ExpertPlacementStrategy = "linear",
    # 新增参数
    layer_idx: int = -1,
) -> tuple[int, Optional[torch.Tensor]]:
    """
        Calculates how many experts should be assigned to each rank for EP and
        creates a mapping from global to local expert index. Experts are
        distributed evenly across ranks. Any remaining are assigned to the
        last rank.

        Args:
            ep_size: The size of the expert parallel group
            ep_rank: The rank of the current process in the expert parallel
                group
            global_num_experts: The total number of experts in the model.
            expert_placement_strategy: The expert placement strategy.

        Returns:
            tuple[int, Optional[torch.Tensor]]: A tuple containing:
                - local_num_experts (int): The number of experts assigned
                    to the current rank.
                - expert_map (Optional[torch.Tensor]): A tensor of shape
                    (global_num_experts,) mapping from global to local index.
                    Contains -1 for experts not assigned to the current rank.
                    Returns None if ep_size is 1.
        """
    assert ep_size > 0
    if ep_size == 1:
        return (global_num_experts, None)
    
    #check ep_size
    print("ep_size: ", ep_size)

    # Distribute experts as evenly as possible to each rank.
    base_experts = global_num_experts // ep_size
    remainder = global_num_experts % ep_size
    if ep_rank < remainder:
        local_num_experts = base_experts + 1
    else:
        local_num_experts = base_experts
    # Create a tensor of size num_experts filled with -1
    expert_map = torch.full((global_num_experts, ), -1, dtype=torch.int32)
    # # #use customed expert map
    # rank0 = [[6, 9, 29, 40, 61, 33, 21, 49, 55, 19, 10, 38, 45, 22, 24, 36, 52, 14, 48, 39, 31, 26, 59, 28, 1, 32, 56, 35, 3, 37, 34, 12], [47, 21, 48, 60, 35, 25, 45, 20, 31, 23, 40, 62, 9, 24, 55, 58, 29, 15, 17, 16, 36, 2, 54, 4, 51, 38, 8, 33, 6, 30, 43, 28], [34, 61, 45, 62, 32, 22, 30, 58, 20, 41, 26, 39, 15, 28, 3, 33, 7, 6, 31, 13, 35, 5, 44, 56, 49, 27, 18, 17, 59, 37, 2, 55], [52, 20, 9, 30, 37, 54, 42, 49, 44, 7, 50, 28, 22, 18, 0, 33, 11, 51, 1, 43, 4, 10, 41, 39, 5, 14, 8, 58, 24, 55, 23, 17], [17, 34, 52, 41, 23, 62, 43, 26, 58, 39, 48, 14, 25, 60, 63, 29, 5, 31, 57, 2, 1, 56, 13, 18, 4, 19, 47, 15, 32, 20, 54, 0], [0, 6, 42, 17, 52, 38, 33, 58, 53, 57, 54, 7, 60, 45, 1, 56, 61, 13, 34, 16, 9, 37, 12, 47, 36, 25, 62, 15, 5, 46, 49, 3], [57, 18, 20, 1, 3, 2, 26, 24, 17, 38, 35, 25, 6, 16, 32, 49, 40, 56, 21, 8, 0, 50, 22, 39, 27, 43, 46, 51, 34, 54, 41, 60], [4, 29, 32, 2, 25, 36, 23, 26, 21, 56, 10, 41, 5, 13, 20, 12, 16, 18, 57, 0, 55, 60, 40, 59, 53, 6, 1, 22, 47, 8, 50, 51], [54, 45, 27, 14, 25, 57, 1, 42, 60, 55, 3, 30, 51, 28, 6, 0, 31, 56, 40, 7, 41, 52, 9, 44, 2, 26, 15, 8, 33, 5, 49, 19], [5, 51, 28, 44, 34, 37, 21, 39, 12, 52, 38, 8, 43, 23, 26, 29, 59, 48, 49, 10, 14, 40, 36, 63, 56, 4, 57, 9, 25, 18, 24, 19], [56, 13, 44, 22, 17, 5, 6, 11, 30, 39, 53, 48, 36, 35, 23, 14, 33, 7, 10, 27, 41, 31, 51, 3, 9, 60, 50, 49, 18, 19, 55, 25], [47, 27, 44, 7, 52, 51, 11, 13, 61, 33, 56, 54, 14, 4, 45, 30, 60, 38, 42, 21, 62, 3, 39, 12, 49, 19, 63, 10, 9, 28, 40, 6], [2, 30, 38, 52, 6, 23, 9, 46, 15, 54, 58, 59, 25, 4, 7, 16, 49, 19, 8, 60, 13, 39, 31, 35, 26, 10, 12, 62, 56, 36, 3, 37], [27, 46, 2, 54, 41, 19, 1, 22, 34, 3, 36, 25, 0, 12, 58, 13, 24, 38, 57, 15, 4, 10, 7, 9, 42, 23, 26, 56, 40, 16, 60, 52], [52, 58, 11, 7, 13, 63, 57, 25, 15, 23, 20, 21, 53, 43, 16, 54, 10, 5, 18, 2, 35, 51, 17, 36, 4, 41, 61, 33, 28, 26, 32, 55], [17, 34, 24, 51, 44, 60, 26, 56, 22, 36, 8, 54, 21, 9, 13, 1, 47, 16, 33, 29, 19, 61, 11, 37, 15, 4, 31, 57, 50, 43, 14, 49]]
    # rank1 = [[58, 41, 25, 23, 54, 43, 46, 0, 53, 18, 20, 15, 42, 8, 60, 63, 47, 11, 7, 16, 13, 5, 27, 2, 30, 17, 50, 44, 62, 4, 51, 57], [11, 18, 19, 42, 37, 5, 32, 7, 1, 41, 53, 39, 63, 59, 34, 27, 0, 13, 50, 14, 44, 49, 52, 3, 57, 61, 12, 22, 46, 10, 56, 26], [4, 8, 60, 24, 14, 16, 9, 23, 38, 0, 57, 54, 12, 29, 1, 47, 63, 53, 21, 52, 19, 40, 48, 25, 36, 46, 50, 11, 43, 42, 10, 51], [31, 62, 61, 19, 38, 12, 56, 59, 63, 29, 15, 40, 32, 34, 45, 57, 36, 35, 27, 46, 47, 2, 48, 6, 13, 26, 25, 21, 16, 60, 3, 53], [21, 49, 8, 46, 6, 61, 42, 59, 27, 9, 45, 7, 40, 24, 11, 35, 36, 10, 38, 51, 22, 44, 50, 28, 37, 30, 33, 16, 12, 55, 53, 3], [10, 19, 32, 21, 8, 14, 30, 31, 22, 39, 51, 41, 35, 4, 18, 55, 11, 40, 50, 24, 20, 23, 2, 48, 27, 43, 59, 28, 26, 63, 44, 29], [61, 5, 52, 36, 62, 55, 9, 12, 59, 23, 48, 31, 63, 58, 30, 29, 4, 47, 13, 44, 53, 10, 33, 11, 7, 37, 42, 14, 19, 45, 28, 15], [17, 35, 38, 15, 46, 61, 48, 58, 39, 37, 9, 30, 54, 45, 19, 24, 63, 14, 52, 34, 62, 42, 44, 3, 11, 31, 33, 7, 49, 27, 28, 43], [11, 37, 16, 20, 62, 17, 34, 10, 35, 63, 32, 50, 29, 12, 53, 21, 38, 18, 61, 13, 59, 23, 47, 24, 36, 22, 43, 48, 39, 4, 46, 58], [54, 50, 7, 35, 46, 1, 13, 55, 42, 16, 41, 58, 53, 6, 2, 30, 32, 27, 61, 33, 0, 62, 22, 45, 47, 20, 11, 31, 60, 3, 17, 15], [62, 16, 2, 43, 4, 37, 59, 32, 47, 52, 38, 20, 42, 1, 0, 28, 12, 15, 45, 61, 24, 21, 26, 29, 46, 58, 8, 34, 40, 57, 54, 63], [57, 43, 15, 1, 58, 34, 23, 22, 20, 8, 31, 0, 46, 37, 18, 24, 36, 16, 53, 17, 26, 29, 2, 32, 5, 35, 48, 55, 41, 50, 25, 59], [43, 55, 63, 45, 50, 17, 14, 22, 32, 41, 51, 48, 11, 5, 21, 34, 28, 47, 33, 44, 53, 1, 0, 42, 61, 27, 29, 57, 24, 20, 40, 18], [55, 32, 20, 5, 53, 45, 8, 63, 35, 47, 62, 17, 33, 37, 30, 51, 31, 59, 39, 61, 44, 6, 48, 43, 11, 50, 18, 28, 49, 29, 21, 14], [1, 8, 62, 42, 9, 37, 44, 60, 34, 22, 45, 12, 29, 38, 39, 49, 31, 48, 0, 24, 59, 50, 6, 3, 46, 30, 47, 19, 56, 40, 27, 14], [3, 48, 53, 39, 58, 2, 38, 20, 55, 10, 23, 52, 46, 27, 35, 63, 28, 18, 45, 12, 5, 30, 59, 7, 62, 32, 40, 0, 6, 41, 25, 42]]
    # if ep_rank == 0:
    #     idx = torch.tensor(rank0[layer_idx], dtype=torch.int32)
    #     expert_map[idx] = torch.arange(32, dtype=torch.int32)
    # else:
    #     idx = torch.tensor(rank1[layer_idx], dtype=torch.int32)
    #     expert_map[idx] = torch.arange(32, dtype=torch.int32)
    #use customed expert map 4 ranks
    # rank0 = [[6, 40, 61, 49, 20, 38, 52, 36, 48, 11, 2, 1, 32, 35, 4, 57], [47, 32, 20, 53, 63, 9, 0, 15, 44, 52, 57, 38, 6, 30, 26, 28], [34, 14, 22, 38, 0, 54, 63, 3, 21, 52, 5, 36, 46, 59, 37, 51], [52, 37, 42, 44, 50, 45, 0, 11, 43, 2, 6, 26, 25, 24, 23, 17], [17, 59, 26, 45, 40, 63, 11, 38, 51, 56, 28, 33, 55, 53, 54, 0], [0, 58, 53, 51, 35, 1, 61, 24, 16, 37, 43, 26, 5, 29, 49, 3], [57, 3, 9, 23, 38, 6, 16, 29, 21, 44, 10, 7, 42, 34, 54, 60], [4, 36, 25, 26, 41, 5, 24, 16, 57, 55, 40, 53, 6, 8, 51, 43], [54, 25, 1, 60, 50, 51, 0, 31, 13, 23, 44, 22, 8, 39, 49, 19], [5, 37, 39, 41, 6, 23, 32, 49, 10, 45, 63, 4, 3, 24, 15, 19], [56, 43, 4, 11, 52, 20, 1, 12, 7, 24, 51, 29, 58, 34, 19, 63], [47, 52, 34, 61, 31, 37, 24, 16, 42, 62, 3, 35, 19, 28, 25, 59], [2, 23, 22, 41, 48, 25, 16, 28, 60, 1, 35, 10, 57, 56, 40, 18], [27, 45, 63, 35, 62, 0, 58, 13, 61, 6, 48, 10, 56, 16, 29, 52], [52, 13, 57, 15, 23, 53, 16, 54, 18, 35, 50, 4, 61, 56, 40, 55], [17, 39, 2, 20, 22, 46, 1, 9, 29, 18, 61, 37, 32, 50, 49, 42]]
    # rank1 = [[58, 23, 54, 46, 55, 15, 45, 63, 7, 16, 5, 59, 17, 44, 62, 12], [11, 42, 35, 45, 41, 39, 34, 27, 13, 14, 36, 3, 12, 22, 10, 43], [4, 24, 62, 30, 20, 39, 15, 7, 47, 6, 13, 25, 56, 17, 11, 2], [31, 19, 38, 54, 49, 7, 28, 18, 35, 27, 47, 10, 13, 14, 60, 3], [21, 6, 61, 43, 9, 14, 60, 5, 36, 22, 44, 37, 4, 47, 12, 3], [10, 42, 52, 38, 22, 41, 4, 55, 40, 13, 9, 48, 47, 59, 15, 44], [61, 36, 2, 59, 17, 25, 58, 32, 47, 0, 50, 39, 43, 46, 15, 28], [17, 15, 48, 58, 37, 30, 13, 20, 52, 34, 60, 11, 31, 49, 28, 50], [11, 20, 57, 42, 55, 29, 53, 18, 61, 59, 47, 26, 43, 5, 46, 58], [54, 46, 13, 21, 55, 58, 8, 43, 48, 59, 0, 40, 36, 31, 57, 18], [62, 22, 17, 6, 47, 38, 35, 23, 33, 27, 41, 26, 9, 8, 18, 55], [57, 58, 11, 33, 8, 14, 4, 36, 53, 29, 32, 49, 48, 50, 40, 6], [43, 50, 17, 15, 51, 4, 7, 49, 33, 13, 39, 61, 12, 24, 3, 37], [55, 5, 54, 19, 22, 17, 37, 51, 24, 38, 57, 11, 50, 23, 40, 21], [1, 42, 37, 44, 34, 20, 39, 49, 48, 2, 6, 3, 41, 33, 26, 14], [3, 58, 60, 56, 52, 21, 27, 16, 47, 33, 19, 11, 62, 0, 6, 43]]
    # rank2 = [[41, 29, 43, 33, 19, 10, 22, 60, 47, 39, 31, 27, 28, 50, 3, 51], [18, 48, 25, 5, 1, 23, 40, 24, 55, 50, 17, 2, 4, 61, 8, 56], [8, 45, 16, 23, 41, 57, 29, 1, 33, 40, 35, 44, 49, 50, 42, 10], [62, 9, 30, 56, 63, 15, 22, 36, 57, 46, 1, 39, 41, 21, 58, 53], [34, 8, 46, 42, 62, 58, 48, 25, 35, 10, 57, 50, 18, 19, 16, 32], [19, 6, 8, 30, 31, 39, 7, 45, 11, 56, 20, 23, 27, 62, 25, 46], [5, 20, 62, 26, 12, 48, 63, 49, 4, 13, 8, 33, 11, 51, 14, 41], [35, 32, 2, 23, 39, 10, 54, 19, 12, 14, 0, 44, 3, 1, 7, 27], [37, 16, 62, 10, 35, 32, 3, 12, 6, 56, 40, 41, 36, 24, 15, 4], [50, 35, 28, 34, 42, 52, 38, 30, 26, 61, 62, 14, 20, 11, 9, 17], [16, 44, 5, 59, 32, 53, 36, 0, 15, 10, 21, 3, 46, 50, 40, 25], [43, 44, 1, 22, 23, 20, 54, 18, 45, 60, 26, 39, 2, 12, 10, 9], [55, 38, 45, 6, 46, 54, 58, 11, 21, 47, 44, 31, 0, 26, 62, 36], [32, 2, 41, 1, 3, 34, 33, 30, 31, 15, 44, 7, 42, 18, 28, 14], [58, 62, 7, 63, 45, 12, 21, 43, 10, 0, 24, 17, 36, 30, 28, 32], [48, 53, 44, 38, 36, 8, 23, 13, 28, 12, 5, 7, 4, 31, 41, 25]]
    # rank3 = [[9, 25, 0, 21, 53, 18, 8, 42, 24, 14, 13, 26, 30, 56, 37, 34], [21, 19, 60, 37, 7, 31, 62, 29, 59, 58, 16, 49, 54, 51, 46, 33], [61, 60, 32, 9, 58, 26, 12, 28, 53, 31, 19, 48, 27, 18, 43, 55], [61, 20, 12, 59, 29, 40, 32, 34, 33, 51, 4, 48, 5, 8, 16, 55], [49, 52, 41, 23, 27, 39, 7, 24, 29, 31, 2, 1, 13, 30, 15, 20], [32, 21, 17, 14, 33, 57, 54, 60, 18, 50, 34, 2, 12, 36, 28, 63], [18, 52, 1, 55, 24, 35, 31, 30, 40, 56, 53, 22, 27, 37, 19, 45], [29, 38, 46, 61, 56, 21, 9, 45, 63, 18, 62, 42, 59, 33, 22, 47], [45, 27, 14, 17, 34, 63, 30, 28, 21, 38, 7, 52, 9, 2, 48, 33], [7, 51, 44, 1, 16, 12, 53, 2, 27, 29, 33, 22, 47, 56, 60, 25], [13, 2, 37, 30, 39, 48, 42, 14, 28, 45, 61, 31, 60, 49, 57, 54], [27, 15, 7, 51, 13, 56, 0, 46, 30, 38, 17, 21, 5, 63, 55, 41], [30, 63, 52, 9, 14, 32, 59, 5, 34, 19, 8, 53, 42, 27, 29, 20], [20, 46, 53, 8, 47, 36, 25, 12, 59, 39, 4, 9, 43, 26, 49, 60], [8, 11, 9, 60, 25, 22, 38, 29, 31, 5, 59, 51, 46, 47, 19, 27], [34, 24, 51, 26, 55, 10, 54, 35, 63, 45, 30, 59, 15, 40, 57, 14]]
    # if ep_rank == 0:
    #     idx = torch.tensor(rank0[layer_idx], dtype=torch.int32)
    #     expert_map[idx] = torch.arange(16, dtype=torch.int32)
    # elif ep_rank == 1:
    #     idx = torch.tensor(rank1[layer_idx], dtype=torch.int32)
    #     expert_map[idx] = torch.arange(16, dtype=torch.int32)
    # elif ep_rank == 2:
    #     idx = torch.tensor(rank2[layer_idx], dtype=torch.int32)
    #     expert_map[idx] = torch.arange(16, dtype=torch.int32)
    # else:
    #     idx = torch.tensor(rank3[layer_idx], dtype=torch.int32)
    #     expert_map[idx] = torch.arange(16, dtype=torch.int32)
    
    #use customed expert map 16 ranks
    rank0 = [[98, 30, 112, 87, 124, 68, 105, 55], [102, 111, 87, 121, 78, 74, 46, 110], [37, 73, 61, 42, 109, 121, 96, 106], [126, 99, 6, 87, 37, 120, 33, 15], [48, 2, 64, 65, 33, 12, 6, 68], [44, 72, 123, 111, 56, 26, 120, 11], [2, 120, 101, 34, 84, 7, 113, 81], [80, 61, 123, 19, 59, 7, 114, 20], [71, 64, 43, 51, 115, 120, 38, 34], [18, 17, 1, 55, 3, 93, 20, 109], [46, 116, 126, 121, 48, 118, 69, 77], [123, 29, 21, 81, 34, 121, 57, 38], [48, 64, 16, 120, 85, 112, 72, 70], [73, 44, 89, 127, 102, 19, 101, 9], [127, 97, 94, 123, 22, 42, 50, 103], [124, 60, 102, 76, 114, 96, 73, 68], [95, 1, 73, 68, 51, 16, 19, 86], [74, 19, 26, 112, 77, 84, 120, 24], [2, 71, 72, 50, 85, 67, 103, 36], [37, 67, 10, 76, 20, 6, 63, 121], [71, 21, 66, 43, 67, 63, 94, 22], [18, 84, 2, 112, 97, 3, 121, 20], [46, 103, 5, 109, 70, 38, 104, 17], [123, 54, 119, 51, 126, 2, 57, 83], [48, 0, 47, 113, 1, 65, 31, 86], [69, 47, 117, 89, 41, 46, 62, 3], [127, 16, 34, 81, 41, 114, 103, 77], [124, 50, 56, 127, 70, 123, 113, 43], [95, 106, 103, 10, 125, 93, 32, 86], [74, 88, 94, 78, 31, 98, 47, 71], [2, 71, 72, 125, 69, 81, 121, 100], [80, 27, 70, 125, 86, 23, 73, 25], [12, 113, 20, 93, 5, 14, 4, 0], [18, 26, 13, 33, 70, 111, 46, 109], [46, 57, 107, 44, 73, 58, 30, 106], [123, 72, 97, 92, 112, 74, 96, 83], [48, 115, 54, 41, 85, 29, 70, 14], [88, 86, 109, 102, 119, 26, 100, 17], [126, 79, 71, 97, 100, 109, 103, 99], [124, 95, 17, 3, 54, 7, 6, 99], [95, 108, 99, 100, 37, 25, 52, 88], [74, 63, 17, 123, 100, 93, 71, 24], [104, 92, 93, 115, 110, 40, 10, 97], [118, 72, 55, 52, 70, 110, 44, 104], [109, 121, 82, 61, 116, 12, 117, 83], [21, 97, 41, 79, 31, 40, 57, 120], [28, 18, 29, 106, 107, 110, 87, 22], [92, 105, 65, 6, 114, 57, 68, 97]]
    rank1 = [[100, 61, 97, 41, 125, 18, 95, 56], [119, 108, 82, 9, 101, 35, 109, 118], [32, 103, 29, 69, 39, 119, 26, 28], [45, 72, 118, 111, 106, 34, 95, 83], [103, 11, 89, 29, 32, 71, 69, 17], [29, 14, 103, 1, 33, 35, 94, 60], [73, 17, 108, 53, 82, 64, 48, 36], [65, 108, 119, 72, 17, 6, 121, 102], [12, 126, 101, 42, 66, 86, 94, 80], [19, 124, 23, 125, 105, 78, 77, 36], [100, 34, 65, 42, 111, 76, 41, 89], [127, 14, 47, 87, 53, 82, 101, 20], [89, 114, 110, 50, 8, 126, 17, 46], [114, 22, 57, 42, 124, 14, 70, 37], [73, 14, 7, 90, 17, 31, 65, 15], [22, 36, 101, 56, 7, 91, 89, 99], [112, 64, 116, 43, 120, 89, 38, 88], [122, 53, 90, 64, 78, 59, 71, 27], [73, 15, 77, 115, 70, 127, 33, 45], [80, 61, 72, 94, 22, 93, 62, 102], [12, 121, 118, 42, 3, 4, 75, 58], [19, 115, 32, 12, 87, 7, 35, 36], [100, 6, 19, 37, 102, 49, 71, 21], [127, 89, 75, 100, 87, 77, 74, 106], [13, 77, 10, 21, 56, 45, 90, 72], [114, 97, 75, 66, 122, 12, 101, 116], [43, 91, 97, 56, 47, 44, 50, 23], [22, 42, 5, 69, 102, 74, 58, 68], [36, 34, 81, 79, 123, 68, 94, 56], [119, 109, 18, 29, 22, 95, 58, 87], [73, 98, 44, 20, 124, 92, 85, 36], [37, 35, 76, 10, 108, 7, 71, 102], [52, 77, 61, 66, 2, 23, 33, 104], [88, 51, 67, 66, 96, 3, 72, 121], [100, 4, 16, 120, 38, 96, 9, 91], [127, 56, 125, 89, 17, 60, 7, 38], [83, 15, 0, 113, 73, 50, 1, 61], [114, 118, 57, 91, 122, 126, 54, 116], [127, 113, 89, 36, 117, 41, 40, 54], [105, 122, 10, 24, 106, 109, 96, 43], [2, 74, 65, 33, 62, 105, 21, 56], [119, 42, 18, 37, 68, 31, 102, 70], [91, 49, 70, 113, 0, 45, 4, 44], [117, 53, 43, 91, 75, 63, 73, 127], [43, 103, 39, 37, 78, 44, 89, 74], [49, 112, 98, 126, 55, 28, 84, 33], [27, 67, 81, 6, 115, 99, 61, 72], [9, 15, 77, 36, 19, 1, 64, 17]]
    rank2 = [[102, 63, 48, 59, 111, 0, 42, 15], [18, 30, 85, 76, 47, 43, 4, 12], [110, 16, 6, 104, 94, 35, 3, 118], [81, 20, 62, 16, 27, 14, 78, 22], [24, 92, 88, 7, 111, 38, 77, 18], [22, 79, 18, 20, 48, 10, 37, 118], [106, 44, 94, 66, 93, 4, 0, 124], [37, 83, 31, 39, 28, 57, 100, 25], [125, 77, 95, 127, 7, 1, 62, 13], [47, 84, 52, 118, 69, 54, 111, 121], [15, 8, 14, 73, 9, 102, 49, 60], [27, 62, 26, 54, 115, 114, 106, 73], [71, 30, 115, 45, 109, 106, 86, 31], [60, 35, 126, 80, 26, 120, 30, 17], [52, 116, 86, 61, 96, 53, 120, 23], [105, 85, 122, 117, 112, 93, 58, 32], [36, 8, 4, 82, 105, 125, 49, 56], [38, 0, 72, 39, 75, 9, 16, 104], [60, 12, 104, 27, 47, 31, 121, 56], [65, 56, 122, 118, 3, 4, 38, 48], [52, 77, 123, 89, 55, 86, 105, 13], [101, 24, 107, 22, 71, 111, 123, 109], [11, 105, 50, 72, 74, 41, 98, 60], [27, 47, 116, 48, 70, 121, 12, 43], [83, 97, 18, 107, 124, 24, 17, 102], [111, 76, 8, 79, 86, 19, 127, 51], [52, 60, 7, 24, 84, 75, 112, 54], [105, 15, 55, 35, 46, 66, 8, 86], [2, 113, 61, 50, 12, 37, 13, 88], [93, 1, 65, 15, 25, 32, 67, 101], [60, 15, 104, 31, 90, 7, 33, 5], [103, 119, 126, 36, 31, 57, 3, 63], [71, 109, 108, 53, 45, 78, 60, 91], [19, 39, 6, 15, 49, 122, 73, 89], [11, 61, 64, 88, 26, 111, 125, 117], [103, 65, 104, 48, 58, 18, 57, 2], [39, 43, 80, 18, 58, 100, 127, 31], [38, 74, 58, 79, 1, 104, 10, 70], [43, 86, 33, 80, 34, 94, 57, 102], [108, 111, 71, 76, 61, 81, 55, 68], [71, 5, 64, 61, 87, 117, 16, 86], [76, 105, 43, 64, 1, 60, 113, 27], [42, 39, 85, 2, 98, 96, 62, 11], [15, 62, 106, 2, 60, 6, 39, 85], [84, 5, 29, 100, 86, 119, 93, 120], [90, 91, 53, 70, 81, 30, 22, 13], [126, 17, 118, 3, 40, 35, 24, 98], [102, 95, 99, 113, 108, 41, 118, 29]]
    rank3 = [[84, 72, 26, 1, 11, 51, 92, 91], [33, 59, 48, 5, 10, 120, 115, 96], [102, 123, 58, 2, 97, 60, 107, 44], [93, 30, 113, 31, 47, 71, 12, 75], [101, 70, 96, 107, 126, 51, 14, 42], [124, 71, 49, 6, 9, 87, 53, 19], [95, 118, 109, 107, 9, 20, 46, 114], [16, 96, 56, 120, 4, 23, 82, 63], [52, 84, 37, 98, 49, 111, 61, 0], [108, 14, 95, 116, 96, 42, 59, 89], [11, 110, 4, 55, 64, 16, 124, 120], [118, 8, 52, 116, 125, 71, 40, 83], [3, 69, 75, 101, 43, 100, 105, 90], [88, 64, 82, 95, 87, 12, 94, 116], [126, 48, 29, 34, 6, 56, 115, 77], [28, 50, 1, 63, 81, 46, 123, 119], [66, 122, 113, 96, 50, 80, 85, 31], [11, 91, 42, 61, 32, 30, 83, 114], [106, 40, 64, 66, 21, 7, 93, 5], [111, 55, 75, 97, 17, 0, 84, 25], [125, 18, 98, 33, 44, 78, 82, 0], [88, 105, 95, 13, 15, 61, 80, 59], [24, 28, 57, 75, 56, 44, 124, 125], [118, 120, 78, 84, 102, 41, 80, 101], [35, 117, 57, 95, 103, 88, 8, 5], [63, 105, 125, 95, 124, 39, 37, 48], [126, 8, 20, 59, 4, 104, 35, 99], [4, 52, 83, 84, 33, 89, 119, 99], [112, 44, 46, 27, 99, 83, 47, 78], [56, 69, 30, 99, 14, 77, 16, 82], [11, 83, 37, 64, 21, 93, 22, 0], [65, 79, 122, 11, 97, 4, 105, 15], [125, 19, 40, 43, 31, 63, 73, 105], [101, 127, 0, 17, 87, 68, 76, 20], [127, 23, 7, 37, 103, 35, 60, 21], [1, 94, 76, 46, 14, 93, 80, 33], [3, 11, 25, 79, 101, 59, 24, 84], [60, 76, 95, 44, 113, 43, 29, 30], [63, 51, 4, 72, 85, 122, 38, 112], [22, 31, 115, 116, 112, 78, 39, 94], [3, 45, 77, 30, 51, 82, 31, 80], [38, 65, 90, 72, 120, 41, 47, 114], [46, 37, 21, 81, 48, 86, 84, 57], [0, 84, 71, 16, 69, 93, 40, 20], [57, 19, 54, 90, 108, 77, 18, 85], [85, 110, 17, 106, 113, 103, 48, 88], [89, 85, 39, 104, 42, 36, 121, 108], [37, 111, 20, 85, 0, 66, 91, 45]]
    rank4 = [[114, 50, 117, 77, 31, 38, 79, 82], [58, 42, 83, 103, 62, 84, 7, 55], [21, 122, 48, 57, 117, 40, 95, 63], [38, 79, 43, 108, 109, 74, 70, 40], [50, 49, 86, 22, 19, 115, 113, 120], [15, 106, 54, 81, 5, 117, 30, 116], [76, 111, 72, 50, 86, 67, 59, 100], [111, 75, 127, 11, 99, 71, 44, 78], [59, 40, 19, 113, 83, 100, 44, 104], [90, 27, 51, 82, 98, 49, 103, 60], [24, 53, 56, 105, 114, 47, 95, 38], [98, 65, 75, 49, 3, 110, 80, 33], [83, 10, 80, 77, 62, 32, 119, 65], [38, 6, 105, 68, 113, 58, 62, 106], [43, 89, 105, 32, 60, 100, 57, 30], [108, 80, 26, 24, 19, 41, 82, 74], [2, 48, 100, 69, 74, 102, 87, 9], [119, 70, 94, 22, 111, 95, 102, 101], [11, 88, 58, 91, 97, 75, 78, 59], [103, 98, 39, 123, 53, 23, 89, 44], [48, 126, 99, 70, 5, 101, 1, 68], [90, 65, 124, 33, 4, 122, 29, 46], [113, 54, 81, 36, 26, 30, 80, 106], [44, 61, 95, 14, 17, 92, 50, 38], [3, 99, 67, 78, 120, 23, 105, 14], [88, 18, 34, 80, 72, 0, 59, 17], [46, 28, 29, 17, 58, 42, 124, 76], [108, 63, 60, 12, 44, 1, 6, 73], [63, 115, 98, 82, 105, 43, 57, 85], [122, 34, 115, 39, 113, 2, 37, 114], [43, 105, 65, 101, 91, 84, 26, 117], [111, 30, 118, 96, 53, 13, 113, 85], [59, 57, 98, 70, 92, 67, 68, 22], [47, 110, 85, 58, 4, 52, 59, 102], [24, 101, 97, 25, 108, 86, 79, 77], [27, 49, 30, 119, 28, 108, 11, 99], [89, 57, 116, 103, 51, 106, 119, 81], [111, 5, 78, 121, 93, 84, 3, 51], [119, 0, 44, 116, 53, 24, 75, 42], [28, 110, 87, 19, 51, 126, 97, 119], [63, 8, 20, 41, 14, 125, 94, 127], [56, 124, 36, 14, 51, 89, 107, 16], [127, 32, 8, 71, 24, 27, 53, 54], [17, 100, 61, 96, 67, 37, 103, 33], [111, 76, 114, 34, 8, 28, 21, 124], [95, 73, 43, 14, 104, 116, 1, 36], [88, 123, 4, 101, 2, 49, 93, 111], [3, 82, 52, 94, 121, 10, 106, 23]]
    rank5 = [[58, 115, 78, 80, 37, 69, 116, 25], [1, 17, 53, 22, 37, 15, 104, 54], [87, 17, 12, 7, 62, 15, 22, 19], [17, 59, 69, 26, 94, 28, 66, 44], [0, 66, 82, 119, 27, 28, 8, 125], [69, 62, 74, 104, 32, 97, 99, 63], [1, 41, 115, 24, 89, 47, 22, 87], [47, 49, 76, 1, 93, 109, 29, 13], [48, 20, 56, 9, 76, 90, 3, 8], [88, 75, 32, 2, 56, 70, 122, 123], [23, 36, 122, 66, 74, 68, 98, 10], [44, 48, 108, 84, 7, 58, 93, 74], [20, 92, 53, 108, 76, 59, 93, 14], [111, 75, 118, 119, 52, 81, 3, 51], [63, 111, 20, 74, 101, 69, 38, 54], [118, 55, 11, 110, 83, 27, 59, 113], [118, 34, 98, 29, 83, 57, 84, 24], [56, 65, 29, 40, 66, 107, 58, 4], [95, 105, 3, 101, 54, 74, 25, 49], [34, 27, 32, 86, 109, 73, 21, 114], [59, 114, 88, 28, 95, 122, 62, 91], [83, 31, 14, 81, 82, 54, 96, 76], [127, 34, 53, 111, 73, 58, 120, 77], [103, 49, 94, 46, 23, 91, 68, 125], [55, 71, 38, 92, 111, 93, 6, 4], [38, 28, 35, 13, 2, 113, 92, 71], [119, 26, 1, 86, 92, 117, 95, 102], [28, 48, 76, 106, 81, 16, 25, 120], [118, 5, 91, 75, 84, 25, 96, 16], [38, 72, 19, 21, 62, 12, 86, 125], [112, 120, 27, 3, 114, 57, 74, 48], [46, 61, 112, 72, 101, 40, 0, 78], [81, 10, 65, 123, 42, 7, 75, 82], [90, 105, 27, 24, 69, 126, 56, 36], [113, 45, 122, 85, 49, 95, 118, 50], [9, 59, 22, 117, 63, 3, 5, 106], [60, 63, 12, 38, 62, 32, 65, 90], [69, 123, 55, 41, 83, 46, 103, 48], [83, 11, 28, 60, 3, 55, 123, 77], [118, 79, 85, 98, 14, 74, 0, 86], [36, 23, 29, 1, 126, 120, 93, 104], [127, 54, 33, 21, 22, 77, 6, 103], [120, 126, 88, 25, 119, 17, 103, 122], [47, 88, 1, 19, 56, 101, 116, 13], [16, 0, 3, 11, 70, 55, 101, 126], [78, 111, 5, 76, 74, 69, 59, 117], [16, 70, 82, 34, 69, 68, 80, 66], [127, 72, 103, 48, 38, 51, 81, 96]]
    rank6 = [[85, 21, 27, 35, 126, 101, 3, 90], [23, 20, 77, 107, 75, 60, 11, 93], [93, 8, 50, 120, 89, 34, 78, 0], [49, 25, 5, 76, 4, 57, 121, 125], [124, 54, 98, 106, 99, 117, 53, 123], [77, 126, 70, 24, 66, 46, 2, 127], [6, 40, 105, 3, 68, 31, 102, 97], [46, 50, 101, 89, 36, 91, 68, 62], [81, 79, 88, 27, 29, 14, 24, 6], [101, 30, 110, 63, 26, 57, 91, 73], [75, 2, 45, 32, 33, 43, 58, 91], [37, 120, 90, 15, 79, 124, 41, 68], [66, 9, 0, 26, 51, 27, 24, 84], [69, 1, 66, 46, 83, 125, 56, 112], [119, 125, 16, 117, 44, 4, 124, 35], [38, 103, 5, 88, 23, 109, 3, 86], [3, 101, 46, 119, 121, 18, 30, 78], [85, 10, 93, 118, 98, 12, 51, 125], [76, 9, 119, 94, 24, 63, 82, 100], [46, 49, 119, 71, 45, 36, 66, 29], [81, 84, 113, 32, 53, 74, 120, 34], [119, 110, 74, 66, 26, 17, 77, 60], [93, 52, 88, 108, 62, 43, 9, 91], [1, 24, 29, 42, 53, 13, 112, 7], [104, 43, 91, 76, 62, 126, 40, 46], [60, 64, 99, 118, 14, 87, 104, 100], [118, 48, 79, 105, 71, 122, 69, 65], [53, 11, 20, 101, 19, 3, 98, 91], [3, 100, 48, 7, 110, 121, 30, 80], [11, 7, 106, 81, 3, 61, 33, 24], [8, 17, 116, 115, 34, 19, 10, 54], [16, 43, 123, 32, 115, 110, 117, 33], [114, 37, 85, 89, 44, 55, 100, 97], [83, 50, 117, 114, 40, 71, 25, 54], [39, 34, 84, 83, 33, 47, 76, 18], [118, 61, 100, 86, 91, 87, 105, 43], [35, 78, 64, 47, 111, 5, 4, 105], [63, 18, 47, 36, 2, 96, 94, 56], [2, 26, 18, 17, 96, 22, 37, 15], [53, 29, 104, 72, 42, 44, 9, 40], [118, 54, 50, 43, 70, 57, 47, 32], [109, 85, 108, 40, 66, 30, 79, 101], [111, 117, 38, 66, 56, 34, 1, 3], [122, 65, 38, 68, 126, 59, 83, 22], [110, 32, 35, 30, 125, 62, 24, 122], [0, 82, 32, 125, 44, 4, 60, 24], [9, 103, 1, 116, 92, 47, 52, 23], [73, 90, 67, 123, 124, 115, 84, 33]]
    rank7 = [[119, 107, 49, 17, 29, 22, 110, 10], [127, 89, 92, 44, 2, 88, 19, 14], [9, 82, 116, 75, 98, 111, 59, 105], [122, 64, 80, 124, 101, 123, 13, 65], [31, 105, 100, 67, 110, 35, 127, 97], [52, 13, 45, 31, 96, 90, 119, 0], [60, 14, 83, 71, 69, 70, 55, 23], [58, 27, 110, 35, 126, 113, 45, 116], [69, 114, 18, 99, 54, 70, 41, 91], [119, 94, 15, 31, 58, 86, 7, 46], [39, 88, 54, 28, 90, 40, 52, 21], [67, 19, 11, 86, 104, 77, 88, 66], [13, 28, 52, 122, 111, 79, 19, 22], [110, 33, 31, 13, 121, 103, 92, 71], [2, 8, 11, 110, 47, 82, 55, 41], [4, 100, 15, 116, 115, 98, 40, 33], [45, 0, 5, 20, 99, 93, 37, 14], [127, 20, 121, 126, 21, 68, 110, 103], [8, 14, 111, 80, 116, 37, 81, 38], [16, 79, 9, 124, 85, 7, 13, 113], [103, 109, 87, 2, 16, 45, 73, 38], [104, 117, 6, 75, 25, 16, 78, 103], [31, 55, 122, 32, 86, 85, 47, 89], [35, 22, 15, 76, 109, 97, 93, 33], [121, 26, 115, 42, 25, 101, 82, 34], [110, 24, 32, 44, 121, 7, 106, 9], [63, 83, 67, 36, 61, 6, 9, 108], [65, 79, 85, 88, 116, 23, 78, 17], [92, 54, 108, 70, 69, 4, 117, 59], [76, 44, 46, 40, 105, 9, 66, 103], [106, 118, 41, 80, 66, 108, 51, 45], [55, 54, 9, 124, 120, 91, 106, 21], [39, 126, 87, 64, 29, 74, 96, 34], [104, 107, 115, 92, 97, 112, 37, 80], [31, 27, 51, 63, 114, 70, 115, 71], [122, 47, 95, 26, 107, 68, 16, 20], [66, 123, 52, 120, 7, 124, 22, 46], [23, 32, 13, 115, 66, 101, 108, 92], [107, 59, 32, 23, 47, 65, 6, 115], [4, 38, 101, 121, 56, 16, 114, 113], [92, 58, 109, 12, 27, 35, 85, 67], [28, 75, 15, 84, 62, 112, 121, 9], [99, 100, 7, 116, 76, 14, 83, 82], [28, 77, 48, 125, 80, 95, 27, 64], [58, 63, 59, 72, 23, 20, 60, 118], [52, 107, 93, 119, 72, 3, 16, 68], [37, 15, 113, 97, 25, 12, 120, 127], [44, 63, 104, 47, 12, 122, 43, 70]]
    rank8 = [[76, 103, 2, 75, 104, 24, 81, 122], [27, 86, 98, 95, 56, 29, 117, 79], [84, 72, 90, 13, 51, 24, 4, 83], [2, 105, 9, 97, 107, 10, 53, 52], [4, 55, 16, 59, 47, 114, 21, 80], [101, 58, 68, 21, 12, 83, 78, 89], [30, 99, 80, 104, 63, 39, 85, 28], [60, 55, 32, 2, 14, 94, 69, 77], [108, 102, 17, 123, 119, 96, 63, 50], [34, 81, 22, 67, 92, 62, 37, 72], [112, 29, 97, 84, 25, 109, 104, 80], [9, 6, 46, 109, 24, 76, 5, 55], [37, 107, 116, 47, 44, 67, 73, 34], [107, 117, 18, 49, 109, 36, 10, 54], [21, 51, 10, 75, 59, 84, 78, 37], [57, 64, 127, 111, 51, 9, 47, 77], [17, 91, 109, 70, 12, 117, 13, 32], [49, 117, 1, 25, 60, 73, 86, 113], [1, 17, 18, 98, 57, 89, 113, 48], [74, 8, 88, 108, 12, 91, 57, 1], [15, 106, 124, 9, 119, 17, 65, 104], [47, 41, 11, 125, 10, 40, 93, 72], [15, 61, 123, 65, 63, 110, 68, 78], [37, 79, 65, 32, 110, 60, 34, 55], [68, 64, 52, 108, 7, 125, 59, 29], [107, 115, 65, 61, 94, 96, 45, 120], [25, 33, 10, 19, 66, 82, 123, 37], [80, 45, 72, 111, 115, 93, 47, 32], [66, 111, 65, 126, 29, 67, 38, 9], [127, 70, 0, 91, 121, 36, 4, 116], [95, 55, 107, 126, 94, 56, 86, 59], [52, 42, 47, 67, 22, 92, 69, 20], [35, 79, 88, 99, 76, 17, 94, 58], [120, 9, 21, 48, 23, 42, 86, 123], [67, 54, 2, 116, 78, 124, 102, 17], [35, 75, 69, 42, 121, 77, 81, 115], [55, 92, 23, 96, 88, 56, 125, 102], [33, 98, 15, 64, 105, 72, 0, 73], [52, 12, 10, 1, 56, 8, 78, 50], [67, 30, 11, 23, 127, 69, 32, 47], [112, 34, 7, 15, 110, 89, 83, 18], [50, 69, 97, 61, 78, 95, 32, 82], [64, 89, 59, 67, 15, 9, 26, 23], [79, 8, 49, 7, 107, 25, 87, 45], [13, 14, 47, 80, 96, 123, 49, 6], [47, 20, 92, 34, 27, 6, 100, 61], [38, 77, 45, 19, 91, 63, 73, 56], [76, 58, 35, 55, 22, 61, 5, 80]]
    rank9 = [[99, 108, 8, 23, 46, 113, 64, 54], [94, 34, 32, 67, 122, 39, 40, 81], [27, 38, 108, 76, 79, 36, 74, 101], [104, 91, 0, 100, 90, 32, 98, 3], [72, 13, 23, 95, 43, 122, 62, 1], [80, 36, 64, 84, 86, 91, 73, 39], [29, 19, 88, 123, 126, 92, 45, 51], [103, 95, 88, 43, 64, 105, 0, 40], [103, 109, 97, 23, 122, 28, 33, 75], [83, 64, 24, 79, 39, 68, 80, 102], [93, 99, 83, 19, 37, 5, 44, 18], [31, 72, 25, 56, 64, 94, 92, 2], [55, 41, 68, 21, 113, 98, 29, 81], [99, 78, 34, 15, 65, 0, 39, 108], [70, 91, 93, 87, 26, 104, 64, 9], [31, 53, 87, 25, 13, 66, 0, 16], [55, 92, 103, 107, 26, 81, 127, 61], [99, 35, 109, 15, 100, 55, 8, 41], [118, 6, 44, 34, 109, 20, 124, 117], [60, 54, 35, 18, 68, 92, 82, 116], [57, 27, 85, 116, 76, 23, 51, 110], [108, 30, 51, 85, 70, 68, 126, 73], [87, 12, 4, 14, 116, 40, 69, 25], [45, 62, 111, 25, 11, 81, 20, 40], [89, 63, 41, 54, 123, 100, 96, 2], [4, 109, 6, 11, 52, 29, 10, 112], [2, 87, 74, 116, 72, 96, 106, 3], [31, 18, 10, 75, 24, 59, 9, 94], [55, 42, 107, 28, 87, 73, 11, 104], [80, 57, 53, 63, 52, 83, 23, 6], [30, 35, 40, 53, 9, 78, 102, 38], [95, 87, 18, 1, 28, 17, 5, 116], [103, 11, 124, 47, 101, 119, 90, 13], [108, 63, 84, 98, 113, 118, 1, 57], [93, 1, 90, 8, 20, 75, 52, 94], [37, 120, 39, 114, 88, 71, 124, 82], [104, 71, 77, 16, 19, 126, 82, 8], [4, 85, 8, 80, 42, 106, 127, 112], [27, 88, 67, 29, 14, 73, 31, 108], [36, 80, 107, 33, 117, 59, 62, 89], [66, 101, 98, 42, 79, 39, 97, 72], [5, 19, 115, 49, 96, 58, 8, 116], [69, 78, 125, 55, 90, 41, 124, 72], [21, 30, 76, 51, 99, 29, 109, 42], [33, 67, 113, 65, 48, 91, 115, 10], [23, 45, 102, 19, 77, 26, 123, 9], [100, 46, 71, 11, 102, 50, 0, 59], [107, 54, 74, 79, 101, 125, 116, 16]]
    rank10 = [[109, 16, 93, 43, 6, 71, 4, 60], [123, 38, 31, 50, 66, 57, 8, 71], [88, 1, 54, 127, 81, 126, 125, 5], [51, 39, 60, 85, 116, 68, 56, 67], [63, 85, 121, 104, 61, 75, 9, 116], [109, 8, 34, 51, 98, 121, 85, 65], [119, 11, 12, 110, 62, 32, 75, 122], [107, 54, 79, 5, 112, 92, 33, 117], [112, 15, 124, 118, 78, 93, 31, 60], [8, 100, 74, 38, 71, 114, 87, 112], [113, 12, 61, 35, 71, 30, 3, 125], [35, 103, 113, 23, 36, 51, 13, 126], [33, 117, 54, 36, 38, 39, 88, 2], [23, 91, 32, 11, 2, 61, 45, 40], [46, 13, 25, 80, 81, 109, 3, 108], [45, 95, 54, 106, 126, 44, 78, 92], [63, 27, 40, 108, 15, 41, 25, 94], [28, 69, 115, 105, 18, 82, 43, 108], [30, 35, 120, 79, 61, 4, 87, 28], [107, 81, 87, 110, 99, 64, 26, 69], [112, 37, 47, 64, 14, 50, 115, 100], [120, 118, 52, 58, 48, 1, 57, 56], [13, 1, 0, 90, 64, 126, 82, 79], [9, 19, 36, 26, 117, 18, 5, 0], [20, 116, 114, 122, 16, 87, 27, 84], [73, 22, 1, 49, 126, 102, 20, 53], [11, 93, 13, 55, 94, 64, 78, 15], [67, 36, 107, 21, 27, 51, 82, 92], [101, 116, 33, 53, 39, 97, 35, 31], [97, 118, 17, 20, 123, 111, 51, 43], [1, 14, 111, 23, 99, 67, 39, 75], [60, 24, 51, 109, 6, 14, 29, 89], [112, 121, 106, 127, 116, 16, 110, 8], [119, 100, 79, 16, 78, 2, 106, 99], [13, 19, 29, 81, 40, 109, 74, 80], [67, 54, 8, 111, 53, 110, 50, 12], [37, 117, 108, 26, 110, 109, 17, 72], [24, 22, 35, 77, 125, 19, 59, 53], [21, 82, 48, 121, 87, 106, 45, 95], [88, 46, 15, 60, 13, 70, 58, 91], [111, 48, 107, 69, 103, 78, 24, 19], [55, 118, 59, 122, 126, 81, 12, 99], [43, 68, 50, 74, 73, 112, 13, 107], [102, 3, 11, 12, 108, 41, 57, 119], [46, 64, 25, 88, 127, 106, 75, 98], [71, 7, 66, 87, 63, 80, 10, 56], [90, 114, 48, 86, 83, 53, 26, 58], [25, 93, 53, 110, 120, 83, 28, 18]]
    rank11 = [[19, 53, 7, 73, 118, 67, 74, 20], [65, 64, 3, 116, 63, 125, 73, 0], [14, 99, 65, 43, 80, 70, 113, 33], [112, 21, 63, 82, 8, 115, 92, 58], [109, 118, 91, 81, 58, 37, 78, 41], [4, 40, 122, 59, 25, 102, 75, 55], [98, 8, 18, 91, 26, 74, 54, 5], [30, 24, 15, 122, 115, 53, 12, 48], [39, 11, 87, 22, 2, 116, 73, 82], [45, 21, 115, 41, 6, 4, 76, 113], [1, 27, 50, 13, 20, 78, 117, 115], [1, 10, 100, 28, 17, 60, 30, 99], [104, 12, 91, 99, 103, 124, 40, 127], [4, 5, 76, 123, 55, 43, 96, 50], [88, 107, 1, 58, 45, 49, 76, 40], [29, 84, 12, 69, 14, 34, 42, 120], [22, 71, 76, 106, 28, 75, 11, 59], [76, 44, 50, 123, 31, 54, 96, 116], [43, 65, 112, 10, 69, 39, 19, 114], [58, 30, 43, 112, 14, 40, 115, 117], [39, 36, 11, 72, 61, 92, 6, 97], [28, 94, 63, 44, 55, 49, 37, 42], [39, 97, 8, 84, 33, 114, 118, 95], [122, 90, 56, 6, 64, 88, 124, 99], [94, 36, 28, 51, 44, 79, 50, 61], [21, 123, 55, 58, 119, 78, 30, 50], [21, 125, 5, 18, 53, 22, 109, 40], [57, 29, 87, 77, 117, 14, 62, 97], [122, 22, 64, 40, 6, 76, 102, 127], [13, 48, 42, 45, 55, 110, 8, 102], [42, 29, 62, 58, 79, 70, 4, 97], [50, 81, 88, 75, 121, 90, 62, 84], [69, 36, 9, 83, 50, 117, 95, 1], [45, 30, 44, 38, 62, 10, 77, 103], [59, 55, 36, 121, 0, 66, 10, 41], [44, 52, 4, 32, 84, 116, 73, 34], [121, 97, 9, 69, 91, 93, 49, 6], [99, 107, 21, 89, 11, 12, 120, 90], [70, 19, 39, 74, 66, 104, 98, 9], [57, 63, 2, 75, 37, 102, 49, 92], [124, 122, 6, 40, 4, 26, 49, 84], [7, 48, 94, 35, 45, 73, 29, 125], [65, 47, 31, 63, 6, 18, 36, 19], [111, 10, 94, 86, 18, 24, 9, 23], [17, 53, 15, 56, 51, 104, 92, 38], [83, 121, 58, 124, 127, 101, 37, 122], [5, 44, 8, 79, 125, 43, 117, 75], [87, 11, 62, 2, 86, 21, 126, 100]]
    rank12 = [[88, 70, 40, 62, 89, 9, 33, 36], [97, 25, 100, 45, 51, 99, 61, 105], [112, 20, 85, 52, 67, 77, 114, 64], [7, 48, 36, 46, 24, 110, 127, 35], [57, 83, 5, 52, 46, 15, 3, 84], [82, 47, 7, 42, 93, 113, 67, 27], [65, 13, 43, 35, 90, 38, 103, 25], [42, 51, 8, 125, 10, 67, 73, 38], [107, 32, 121, 10, 72, 58, 5, 68], [28, 16, 0, 48, 126, 43, 97, 25], [6, 22, 81, 0, 57, 85, 119, 17], [45, 39, 85, 16, 102, 119, 63, 43], [94, 63, 15, 42, 87, 7, 56, 49], [97, 25, 74, 20, 122, 7, 100, 48], [12, 67, 5, 33, 19, 92, 95, 99], [67, 65, 125, 20, 48, 35, 6, 17], [90, 44, 54, 77, 123, 6, 21, 104], [34, 48, 124, 88, 2, 62, 37, 87], [123, 83, 41, 107, 108, 26, 32, 68], [50, 95, 41, 96, 101, 120, 90, 33], [35, 19, 127, 26, 117, 54, 25, 80], [8, 64, 116, 50, 98, 43, 86, 102], [101, 27, 29, 83, 92, 20, 10, 18], [69, 39, 86, 16, 114, 58, 66, 73], [66, 12, 80, 69, 118, 85, 127, 81], [23, 67, 85, 31, 42, 93, 103, 70], [39, 113, 111, 90, 49, 100, 45, 30], [37, 95, 26, 104, 122, 61, 96, 40], [17, 45, 23, 1, 109, 51, 120, 21], [85, 84, 117, 35, 60, 68, 89, 79], [13, 119, 6, 96, 24, 25, 49, 68], [56, 74, 94, 127, 39, 59, 38, 82], [48, 84, 32, 72, 28, 3, 111, 80], [34, 8, 94, 41, 43, 12, 5, 60], [6, 53, 105, 14, 110, 5, 62, 98], [90, 15, 25, 78, 102, 85, 21, 126], [13, 33, 20, 53, 118, 34, 40, 10], [110, 65, 31, 82, 68, 52, 61, 50], [125, 25, 81, 49, 84, 16, 92, 124], [50, 64, 20, 27, 34, 125, 1, 8], [17, 55, 123, 91, 119, 11, 76, 96], [44, 13, 39, 46, 117, 67, 83, 4], [108, 109, 101, 29, 58, 61, 106, 16], [115, 121, 92, 90, 32, 98, 120, 26], [73, 105, 45, 71, 9, 40, 66, 69], [89, 64, 105, 8, 86, 75, 35, 96], [74, 84, 21, 13, 14, 33, 95, 57], [109, 31, 24, 40, 89, 112, 39, 49]]
    rank13 = [[127, 96, 123, 14, 44, 13, 66, 5], [70, 52, 36, 41, 16, 24, 28, 90], [45, 92, 53, 124, 68, 115, 41, 23], [29, 18, 1, 86, 50, 55, 61, 119], [44, 56, 45, 10, 40, 39, 36, 93], [107, 114, 100, 108, 105, 17, 3, 28], [52, 15, 116, 10, 21, 58, 127, 49], [41, 81, 74, 22, 21, 9, 26, 66], [46, 36, 106, 89, 74, 92, 117, 110], [53, 104, 9, 11, 12, 5, 106, 66], [127, 31, 123, 7, 86, 62, 94, 106], [4, 42, 112, 107, 59, 96, 18, 0], [25, 121, 18, 57, 96, 23, 82, 6], [21, 16, 8, 115, 79, 86, 72, 29], [68, 62, 24, 27, 66, 0, 122, 112], [30, 79, 52, 75, 10, 8, 97, 94], [58, 126, 111, 53, 10, 79, 47, 72], [5, 57, 17, 63, 14, 89, 92, 67], [52, 110, 55, 122, 126, 92, 90, 22], [42, 83, 70, 31, 11, 28, 59, 100], [69, 46, 40, 20, 90, 29, 7, 8], [127, 100, 9, 92, 38, 91, 23, 89], [67, 112, 59, 121, 66, 35, 115, 94], [98, 67, 72, 113, 3, 63, 115, 96], [30, 60, 39, 53, 109, 32, 19, 112], [5, 27, 77, 57, 74, 43, 81, 40], [62, 88, 51, 0, 121, 110, 38, 120], [103, 110, 64, 121, 125, 34, 109, 39], [62, 0, 18, 15, 20, 14, 89, 19], [28, 5, 124, 73, 54, 64, 107, 27], [122, 110, 16, 87, 127, 61, 103, 63], [34, 104, 98, 100, 99, 44, 114, 45], [15, 30, 27, 21, 41, 24, 115, 62], [53, 124, 11, 75, 32, 93, 81, 35], [15, 99, 12, 65, 48, 69, 3, 89], [10, 62, 19, 109, 55, 70, 51, 0], [68, 28, 95, 21, 67, 27, 45, 87], [20, 16, 6, 117, 81, 14, 39, 9], [46, 62, 61, 20, 101, 76, 30, 114], [52, 103, 83, 82, 84, 35, 93, 73], [0, 46, 44, 60, 75, 113, 68, 9], [80, 26, 88, 2, 25, 110, 3, 87], [95, 12, 35, 75, 87, 22, 123, 121], [66, 4, 81, 5, 105, 35, 113, 89], [95, 41, 31, 99, 4, 87, 22, 81], [94, 50, 65, 67, 11, 109, 114, 25], [31, 94, 122, 20, 51, 62, 78, 54], [34, 4, 59, 88, 119, 71, 42, 13]]
    rank14 = [[121, 65, 52, 57, 34, 83, 12, 39], [26, 21, 80, 126, 6, 69, 114, 68], [31, 55, 49, 30, 25, 11, 46, 91], [88, 89, 19, 84, 23, 77, 11, 73], [90, 76, 112, 26, 30, 87, 20, 60], [112, 125, 50, 115, 76, 38, 95, 57], [96, 16, 27, 125, 57, 61, 33, 78], [87, 98, 70, 18, 118, 85, 3, 52], [30, 21, 47, 85, 53, 65, 67, 4], [99, 127, 44, 13, 85, 29, 61, 35], [87, 59, 107, 103, 72, 51, 70, 79], [22, 95, 97, 61, 111, 117, 50, 105], [60, 35, 11, 58, 118, 5, 125, 61], [98, 67, 28, 47, 41, 93, 84, 104], [118, 83, 28, 121, 72, 85, 98, 106], [71, 2, 107, 21, 104, 90, 39, 62], [124, 42, 39, 7, 110, 97, 67, 52], [7, 13, 81, 45, 6, 36, 3, 33], [42, 13, 62, 125, 53, 51, 84, 46], [47, 104, 51, 125, 126, 105, 106, 78], [107, 102, 10, 83, 93, 56, 24, 60], [45, 53, 39, 0, 69, 67, 99, 106], [99, 22, 107, 42, 7, 16, 23, 117], [8, 59, 52, 107, 104, 108, 21, 82], [74, 33, 15, 110, 75, 73, 119, 70], [98, 33, 15, 82, 68, 26, 54, 90], [70, 12, 32, 89, 80, 101, 98, 57], [38, 30, 100, 54, 13, 7, 41, 49], [90, 124, 60, 74, 41, 49, 26, 24], [75, 90, 126, 108, 112, 96, 92, 41], [76, 12, 52, 77, 47, 50, 82, 28], [8, 49, 83, 12, 19, 48, 93, 66], [118, 107, 46, 49, 54, 122, 56, 25], [28, 31, 14, 95, 55, 22, 91, 29], [22, 112, 123, 56, 119, 82, 43, 104], [36, 45, 23, 113, 79, 66, 41, 101], [74, 94, 114, 36, 107, 98, 76, 2], [67, 25, 49, 75, 62, 7, 87, 40], [93, 91, 90, 7, 111, 58, 69, 35], [45, 26, 21, 25, 90, 48, 77, 123], [22, 106, 73, 38, 81, 28, 114, 59], [34, 57, 53, 52, 91, 98, 86, 23], [77, 94, 80, 20, 102, 118, 5, 79], [58, 78, 46, 31, 114, 124, 112, 50], [7, 102, 68, 107, 50, 42, 97, 79], [18, 99, 38, 46, 108, 118, 54, 62], [76, 96, 105, 30, 109, 7, 112, 60], [14, 32, 30, 69, 50, 98, 7, 75]]
    rank15 = [[120, 45, 106, 94, 28, 32, 47, 86], [72, 124, 106, 113, 112, 49, 91, 13], [18, 47, 56, 100, 10, 71, 66, 86], [103, 41, 102, 117, 114, 54, 42, 96], [25, 74, 108, 79, 102, 34, 94, 73], [23, 43, 41, 61, 88, 110, 92, 16], [42, 112, 77, 79, 117, 37, 121, 56], [104, 34, 106, 86, 97, 124, 90, 84], [35, 105, 26, 57, 45, 55, 16, 25], [120, 65, 117, 107, 50, 10, 33, 40], [67, 108, 101, 92, 63, 26, 96, 82], [122, 69, 78, 32, 89, 70, 91, 12], [97, 74, 123, 78, 4, 95, 1, 102], [24, 63, 77, 85, 27, 53, 90, 59], [79, 113, 39, 18, 36, 71, 114, 102], [18, 37, 72, 121, 61, 49, 70, 43], [115, 65, 62, 60, 33, 114, 23, 35], [80, 97, 46, 106, 52, 23, 79, 47], [29, 16, 99, 96, 23, 86, 102, 0], [24, 52, 2, 127, 19, 5, 15, 77], [30, 79, 108, 49, 96, 41, 111, 31], [34, 27, 21, 79, 114, 62, 5, 113], [45, 2, 119, 51, 48, 3, 96, 76], [4, 10, 31, 85, 28, 30, 71, 105], [9, 37, 11, 106, 58, 98, 49, 22], [16, 91, 25, 36, 83, 108, 84, 56], [68, 107, 27, 73, 14, 85, 31, 115], [71, 118, 2, 126, 90, 112, 0, 114], [71, 58, 8, 77, 119, 114, 72, 52], [10, 49, 26, 50, 59, 100, 120, 104], [88, 123, 18, 109, 89, 46, 113, 32], [107, 58, 2, 64, 41, 68, 26, 77], [102, 18, 6, 26, 120, 51, 86, 38], [64, 65, 125, 82, 116, 7, 61, 74], [87, 42, 28, 92, 72, 32, 68, 126], [24, 31, 98, 6, 29, 13, 64, 40], [30, 99, 122, 112, 44, 75, 86, 42], [34, 28, 97, 37, 27, 124, 45, 71], [68, 118, 5, 13, 105, 110, 64, 120], [18, 65, 100, 12, 5, 41, 120, 66], [90, 116, 53, 121, 10, 115, 13, 102], [11, 0, 10, 20, 111, 106, 92, 104], [28, 33, 114, 30, 52, 60, 105, 51], [74, 82, 123, 34, 36, 54, 14, 97], [1, 2, 94, 112, 26, 36, 52, 27], [2, 115, 29, 51, 39, 15, 42, 12], [32, 10, 124, 64, 41, 119, 65, 55], [8, 117, 27, 46, 78, 26, 60, 56]]
    rank_by_ep = [rank0, rank1, rank2, rank3, rank4, rank5, rank6, rank7, rank8,
    rank9, rank10, rank11, rank12, rank13, rank14, rank15]
    p2l = [sum(group, []) for group in zip(*rank_by_ep)][layer_idx]
    l2p = [0] * len(p2l)
    for p, l in enumerate(p2l):
        l2p[l] = p
    print("p2l:", len(p2l), p2l)
    print("l2p:", len(l2p), l2p)
    l2p = torch.tensor(l2p, dtype=torch.int32)
    rank_list = rank_by_ep[ep_rank]
    # 该 rank 在此层的全局专家编号
    idx = torch.as_tensor(rank_list[layer_idx], dtype=torch.int32, device=expert_map.device)
    # 生成本地 id：0..num_local-1（自适应本 rank 的专家个数）
    local_ids = torch.arange(idx.numel(), dtype=torch.int32, device=expert_map.device)
    # 写回映射：global_id -> local_id
    expert_map[idx] = local_ids
    print("use custom expert placement for ep_rank", ep_rank, "layer_idx", layer_idx)
    print("idx is", idx.size(), idx)
    print("local num_experts:", local_num_experts)
    print("expert_map:", expert_map.size(), expert_map)
    print("l2p:", l2p.size(), l2p)
    return (local_num_experts, expert_map, l2p)

    '''
    #所有层共享的版本
    rank0 = [127, 80, 125, 58, 66, 59, 102, 40]
    rank1 = [88, 111, 99, 49, 89, 32, 113, 25]
    rank2 = [95, 12, 126, 77, 41, 24, 29, 26]
    rank3 = [101, 76, 38, 57, 47, 64, 106, 93]
    rank4 = [52, 90, 114, 105, 121, 14, 92, 70]
    rank5 = [46, 63, 4, 124, 20, 117, 5, 50]
    rank6 = [83, 71, 108, 42, 43, 10, 69, 91]
    rank7 = [48, 65, 8, 1, 85, 67, 6, 86]
    rank8 = [11, 60, 34, 53, 87, 35, 94, 33]
    rank9 = [118, 2, 73, 79, 100, 75, 54, 82]
    rank10 = [18, 21, 36, 122, 55, 0, 78, 97]
    rank11 = [103, 28, 112, 9, 104, 51, 13, 62]
    rank12 = [107, 123, 15, 56, 39, 23, 115, 72]
    rank13 = [45, 16, 19, 81, 7, 3, 61, 96]
    rank14 = [22, 30, 17, 109, 31, 110, 98, 120]
    rank15 = [27, 37, 119, 44, 74, 84, 116, 68]
    rank_by_ep = [rank0, rank1, rank2, rank3, rank4, rank5, rank6, rank7, rank8,
    rank9, rank10, rank11, rank12, rank13, rank14, rank15]
    rank_list = rank_by_ep[ep_rank]
    p2l = [x for sub in rank_by_ep for x in sub]
    l2p = [0] * len(p2l)
    for p, l in enumerate(p2l):
        l2p[l] = p
    l2p = torch.tensor(l2p, dtype=torch.int32)
    # 该 rank 在此层的全局专家编号
    idx = torch.as_tensor(rank_list, dtype=torch.int32, device=expert_map.device)
    # 生成本地 id：0..num_local-1（自适应本 rank 的专家个数）
    local_ids = torch.arange(idx.numel(), dtype=torch.int32, device=expert_map.device)
    # 写回映射：global_id -> local_id
    expert_map[idx] = local_ids
    print("use custom layer_shared expert placement for ep_rank", ep_rank, "layer_idx", layer_idx)
    print("idx is", idx.size(), idx)
    print("local num_experts:", local_num_experts)
    print("expert_map:", expert_map.size(), expert_map)
    return (local_num_experts, expert_map, l2p)
    '''
    

    '''
    ###反向贪心
    #use customed expert map 16 ranks
    rank0 = [[98, 100, 102, 84, 58, 114, 85, 119], [70, 64, 125, 102, 127, 108, 119, 71], [110, 32, 37, 93, 99, 58, 40, 102], [49, 103, 88, 53, 31, 82, 86, 17], [98, 32, 24, 12, 72, 92, 48, 115], [69, 37, 64, 15, 109, 22, 104, 25], [107, 15, 11, 106, 58, 24, 73, 2], [7, 52, 59, 80, 79, 24, 37, 65], [38, 17, 45, 92, 57, 5, 71, 32], [47, 82, 2, 18, 85, 7, 55, 91], [126, 3, 55, 71, 94, 57, 66, 46], [3, 59, 95, 20, 25, 123, 107, 127], [8, 21, 64, 105, 83, 89, 48, 85], [0, 61, 47, 8, 81, 73, 33, 19], [33, 92, 1, 42, 100, 14, 127, 32], [80, 52, 49, 28, 93, 41, 0, 124], [65, 81, 101, 105, 103, 95, 0, 90], [18, 80, 88, 99, 5, 44, 49, 57], [107, 24, 58, 2, 81, 11, 73, 78], [7, 59, 8, 123, 112, 101, 24, 52], [53, 57, 5, 16, 114, 77, 30, 71], [2, 125, 82, 90, 85, 18, 47, 101], [68, 71, 16, 55, 57, 26, 46, 11], [3, 59, 52, 56, 123, 25, 72, 127], [21, 64, 22, 8, 94, 41, 48, 92], [61, 20, 58, 52, 81, 67, 117, 37], [92, 14, 42, 33, 1, 127, 86, 4], [0, 25, 41, 80, 52, 75, 125, 124], [48, 42, 82, 113, 35, 103, 95, 36], [88, 80, 55, 73, 122, 99, 72, 5], [107, 122, 11, 24, 2, 115, 73, 60], [75, 7, 8, 56, 52, 123, 24, 50], [53, 16, 92, 57, 114, 77, 9, 12], [85, 90, 55, 92, 2, 18, 50, 88], [16, 68, 24, 126, 55, 57, 63, 71], [3, 59, 72, 27, 52, 121, 103, 123], [21, 92, 64, 8, 83, 89, 94, 66], [47, 20, 33, 81, 117, 88, 61, 67], [1, 43, 67, 100, 98, 92, 12, 63], [41, 52, 127, 124, 28, 29, 22, 80], [48, 42, 54, 68, 103, 95, 0, 12], [88, 55, 38, 10, 39, 74, 109, 86], [56, 58, 104, 99, 117, 9, 91, 42], [71, 31, 118, 15, 122, 30, 115, 117], [57, 51, 25, 16, 111, 31, 67, 58], [29, 95, 99, 0, 90, 21, 82, 38], [94, 32, 77, 118, 76, 126, 27, 34], [92, 110, 44, 4, 48, 53, 8, 9]]
    rank1 = [[76, 109, 99, 19, 88, 127, 121, 120], [111, 15, 79, 94, 1, 82, 106, 59], [21, 94, 30, 31, 98, 45, 78, 87], [124, 81, 36, 60, 102, 74, 45, 79], [19, 126, 103, 28, 90, 17, 101, 7], [44, 124, 9, 29, 38, 91, 98, 77], [76, 12, 29, 95, 78, 1, 26, 119], [32, 46, 16, 111, 114, 112, 58, 54], [12, 52, 53, 125, 59, 56, 30, 112], [90, 108, 0, 16, 88, 119, 19, 6], [7, 12, 26, 11, 24, 112, 69, 68], [56, 14, 27, 30, 28, 118, 103, 99], [4, 41, 94, 92, 22, 2, 3, 13], [67, 52, 78, 88, 69, 104, 26, 114], [4, 52, 73, 117, 86, 126, 43, 81], [22, 32, 29, 25, 45, 108, 57, 125], [36, 112, 39, 118, 48, 89, 113, 2], [74, 36, 56, 11, 38, 119, 39, 122], [116, 17, 60, 122, 106, 1, 95, 76], [37, 80, 65, 75, 111, 103, 72, 16], [102, 12, 52, 59, 22, 109, 125, 98], [88, 31, 14, 0, 16, 122, 50, 44], [56, 61, 24, 126, 47, 27, 63, 100], [95, 27, 61, 14, 20, 44, 24, 125], [66, 96, 99, 13, 65, 83, 3, 55], [0, 69, 92, 47, 114, 88, 29, 19], [52, 43, 100, 98, 32, 74, 46, 126], [22, 118, 83, 29, 127, 11, 108, 105], [0, 5, 101, 2, 107, 68, 3, 98], [18, 56, 74, 29, 68, 38, 119, 59], [89, 58, 91, 13, 17, 106, 30, 78], [80, 37, 103, 16, 95, 60, 101, 111], [59, 41, 30, 48, 71, 72, 52, 87], [122, 101, 69, 6, 84, 47, 19, 82], [66, 46, 11, 4, 47, 12, 39, 81], [127, 36, 4, 28, 56, 35, 37, 65], [111, 48, 33, 41, 78, 3, 117, 126], [114, 11, 7, 58, 0, 34, 69, 8], [119, 4, 32, 127, 14, 107, 126, 33], [93, 108, 83, 102, 105, 49, 67, 118], [58, 126, 2, 17, 36, 115, 5, 71], [119, 57, 76, 34, 108, 26, 21, 56], [52, 111, 127, 69, 108, 81, 46, 6], [28, 0, 58, 92, 94, 82, 96, 90], [43, 109, 5, 107, 94, 59, 84, 46], [49, 85, 44, 50, 71, 91, 52, 18], [106, 5, 89, 37, 28, 46, 88, 1], [102, 87, 72, 37, 28, 101, 107, 125]]
    rank2 = [[45, 96, 70, 65, 53, 16, 103, 108], [10, 58, 112, 18, 23, 72, 36, 118], [1, 88, 34, 38, 18, 47, 113, 25], [64, 72, 6, 122, 75, 4, 112, 2], [99, 78, 31, 109, 54, 108, 52, 71], [73, 101, 1, 82, 112, 43, 59, 8], [42, 27, 60, 101, 44, 124, 52, 30], [47, 42, 30, 41, 101, 51, 10, 103], [109, 48, 107, 36, 46, 97, 22, 103], [101, 17, 100, 125, 99, 65, 45, 64], [75, 4, 39, 27, 15, 113, 127, 100], [37, 44, 73, 124, 52, 54, 72, 45], [55, 97, 49, 127, 117, 69, 71, 66], [38, 60, 107, 45, 111, 103, 122, 49], [119, 110, 107, 68, 63, 46, 26, 20], [105, 67, 75, 118, 18, 103, 4, 66], [3, 126, 66, 17, 76, 5, 32, 124], [41, 59, 76, 127, 77, 82, 110, 42], [126, 30, 118, 26, 44, 123, 54, 29], [46, 43, 60, 50, 95, 119, 42, 74], [92, 41, 48, 65, 112, 15, 72, 107], [6, 119, 19, 84, 104, 120, 108, 100], [127, 3, 31, 15, 101, 48, 66, 93], [35, 103, 118, 37, 67, 107, 109, 45], [108, 16, 123, 35, 104, 89, 85, 30], [60, 63, 38, 111, 73, 16, 15, 78], [110, 81, 78, 118, 119, 69, 49, 17], [28, 65, 103, 93, 67, 57, 34, 4], [118, 112, 66, 26, 89, 115, 62, 17], [11, 41, 57, 82, 10, 127, 77, 76], [54, 12, 1, 42, 81, 87, 76, 119], [65, 46, 9, 55, 79, 72, 59, 47], [81, 89, 125, 5, 17, 112, 22, 39], [14, 31, 104, 56, 33, 74, 119, 61], [26, 100, 28, 127, 48, 3, 9, 31], [7, 14, 122, 75, 10, 44, 9, 1], [65, 55, 13, 35, 97, 22, 74, 123], [60, 38, 52, 100, 63, 111, 26, 39], [52, 42, 86, 125, 81, 17, 46, 5], [75, 53, 65, 26, 25, 45, 122, 70], [118, 99, 3, 101, 66, 92, 90, 81], [127, 72, 41, 80, 49, 50, 59, 44], [39, 124, 120, 48, 114, 43, 76, 64], [53, 62, 66, 110, 25, 107, 123, 17], [105, 29, 64, 125, 127, 17, 62, 33], [78, 74, 94, 107, 51, 118, 23, 20], [16, 31, 9, 90, 2, 68, 103, 38], [77, 3, 58, 83, 1, 115, 14, 76]]
    rank3 = [[107, 115, 21, 50, 63, 72, 61, 8], [100, 53, 90, 86, 89, 97, 65, 6], [20, 46, 55, 53, 16, 84, 56, 112], [39, 38, 126, 104, 18, 21, 91, 32], [57, 25, 85, 56, 49, 30, 118, 60], [66, 13, 14, 17, 125, 33, 71, 0], [6, 87, 68, 56, 105, 77, 16, 71], [19, 60, 123, 107, 95, 127, 8, 31], [81, 102, 72, 84, 99, 77, 105, 21], [120, 28, 21, 44, 33, 81, 70, 104], [48, 6, 8, 87, 31, 21, 54, 23], [35, 26, 67, 121, 61, 65, 98, 36], [52, 123, 9, 75, 121, 35, 104, 63], [15, 92, 4, 21, 98, 16, 6, 112], [88, 70, 93, 22, 118, 79, 21, 38], [85, 83, 65, 87, 89, 71, 106, 40], [45, 35, 97, 7, 55, 28, 63, 92], [55, 34, 50, 69, 48, 81, 85, 19], [88, 42, 115, 83, 35, 71, 119, 55], [45, 58, 34, 49, 62, 54, 107, 47], [81, 17, 46, 103, 39, 121, 99, 36], [7, 105, 28, 107, 91, 64, 9, 45], [9, 4, 113, 39, 116, 87, 2, 112], [9, 121, 7, 5, 122, 10, 99, 1], [74, 111, 78, 121, 12, 97, 60, 69], [97, 27, 107, 122, 21, 98, 99, 49], [21, 68, 88, 93, 105, 12, 11, 125], [102, 53, 117, 119, 18, 106, 48, 68], [97, 58, 55, 40, 71, 92, 124, 90], [50, 93, 21, 48, 49, 44, 34, 109], [95, 123, 88, 29, 20, 105, 43, 96], [58, 62, 108, 34, 49, 77, 14, 10], [111, 98, 15, 36, 26, 109, 84, 107], [64, 7, 45, 100, 120, 108, 65, 58], [113, 21, 27, 92, 7, 15, 112, 118], [118, 74, 91, 45, 109, 24, 67, 49], [69, 105, 60, 30, 104, 20, 99, 121], [54, 78, 16, 23, 97, 19, 98, 4], [70, 21, 88, 78, 61, 25, 68, 110], [123, 0, 57, 16, 66, 18, 92, 47], [30, 63, 50, 107, 112, 35, 45, 93], [5, 77, 7, 29, 11, 48, 61, 16], [92, 89, 77, 94, 17, 14, 27, 35], [78, 21, 57, 102, 10, 72, 79, 3], [121, 95, 14, 47, 106, 97, 113, 73], [11, 89, 15, 115, 75, 69, 125, 45], [25, 84, 44, 71, 6, 54, 3, 57], [122, 113, 62, 109, 73, 74, 93, 106]]
    rank4 = [[49, 2, 106, 27, 123, 30, 93, 52], [31, 124, 42, 101, 122, 80, 27, 17], [91, 54, 122, 108, 71, 8, 116, 27], [14, 89, 99, 115, 76, 29, 37, 93], [79, 55, 73, 26, 76, 66, 63, 64], [108, 47, 52, 51, 34, 126, 76, 107], [121, 96, 65, 14, 99, 82, 51, 17], [74, 75, 126, 83, 49, 45, 50, 62], [16, 87, 15, 114, 101, 79, 85, 40], [31, 94, 80, 41, 105, 83, 110, 9], [2, 61, 93, 5, 73, 16, 51, 14], [10, 9, 100, 1, 22, 122, 109, 31], [74, 65, 12, 99, 15, 78, 116, 114], [66, 63, 28, 34, 59, 29, 23, 80], [101, 109, 12, 105, 51, 80, 125, 48], [53, 127, 38, 30, 123, 16, 34, 100], [119, 44, 42, 21, 22, 71, 26, 115], [28, 46, 109, 21, 61, 53, 72, 97], [12, 43, 87, 91, 63, 65, 16, 105], [19, 87, 79, 30, 55, 70, 26, 2], [79, 27, 84, 24, 45, 83, 18, 87], [92, 83, 41, 110, 65, 21, 33, 74], [12, 105, 13, 54, 5, 36, 21, 88], [65, 36, 49, 75, 8, 28, 63, 19], [117, 26, 15, 36, 73, 63, 114, 33], [28, 7, 34, 95, 126, 108, 6, 4], [25, 107, 70, 48, 83, 79, 101, 9], [71, 87, 107, 85, 90, 32, 72, 30], [81, 49, 45, 1, 22, 25, 54, 63], [69, 97, 30, 7, 28, 46, 61, 53], [71, 83, 65, 26, 112, 124, 22, 52], [32, 42, 29, 107, 44, 119, 30, 54], [65, 102, 45, 103, 28, 121, 124, 18], [44, 9, 28, 110, 83, 91, 49, 70], [13, 61, 101, 122, 105, 87, 19, 36], [61, 95, 5, 102, 100, 20, 90, 78], [15, 39, 85, 63, 12, 28, 7, 80], [76, 28, 80, 91, 21, 37, 122, 6], [105, 93, 27, 109, 83, 79, 51, 0], [88, 103, 110, 117, 106, 31, 30, 63], [82, 34, 106, 65, 97, 55, 24, 76], [18, 33, 69, 83, 28, 99, 3, 105], [32, 12, 100, 31, 21, 80, 63, 33], [77, 34, 120, 100, 29, 75, 88, 5], [112, 68, 1, 7, 2, 53, 72, 15], [58, 64, 65, 7, 121, 66, 111, 53], [86, 114, 43, 100, 113, 104, 123, 21], [112, 25, 32, 127, 124, 117, 99, 116]]
    rank5 = [[7, 40, 117, 78, 26, 48, 97, 57], [95, 96, 26, 107, 28, 9, 3, 87], [66, 83, 60, 127, 9, 75, 2, 100], [105, 59, 5, 9, 101, 63, 116, 0], [69, 13, 16, 111, 75, 124, 23, 120], [50, 40, 42, 49, 72, 97, 80, 79], [83, 13, 115, 109, 3, 88, 74, 53], [15, 34, 14, 85, 82, 109, 17, 70], [41, 64, 98, 69, 121, 39, 108, 88], [14, 69, 43, 127, 96, 111, 23, 84], [99, 36, 13, 107, 88, 45, 56, 105], [58, 19, 18, 8, 93, 11, 87, 63], [60, 20, 30, 53, 7, 111, 80, 110], [87, 77, 95, 20, 39, 76, 75, 99], [111, 10, 17, 55, 64, 89, 84, 61], [95, 107, 11, 102, 46, 101, 60, 8], [8, 40, 108, 68, 62, 50, 93, 109], [10, 7, 86, 13, 108, 29, 121, 123], [9, 52, 111, 96, 15, 98, 20, 14], [32, 113, 9, 27, 51, 41, 83, 57], [101, 28, 19, 89, 35, 97, 85, 88], [117, 61, 32, 43, 94, 127, 55, 49], [75, 45, 14, 123, 103, 111, 58, 118], [23, 113, 90, 111, 4, 58, 117, 98], [52, 9, 116, 124, 39, 122, 91, 80], [8, 77, 33, 87, 76, 80, 115, 11], [80, 22, 84, 63, 26, 51, 20, 111], [45, 101, 76, 33, 19, 37, 60, 10], [60, 65, 93, 7, 100, 33, 44, 105], [75, 90, 1, 85, 36, 42, 118, 19], [15, 8, 118, 16, 35, 82, 49, 37], [118, 112, 43, 2, 35, 104, 45, 87], [35, 46, 88, 32, 99, 66, 79, 123], [105, 63, 21, 43, 125, 79, 81, 17], [94, 93, 75, 2, 123, 5, 116, 23], [53, 26, 8, 125, 25, 94, 99, 22], [96, 116, 9, 36, 47, 114, 108, 127], [107, 93, 32, 92, 25, 15, 87, 95], [48, 118, 90, 19, 26, 2, 101, 49], [10, 125, 104, 5, 85, 4, 95, 11], [124, 22, 119, 26, 73, 8, 98, 113], [78, 19, 85, 42, 12, 53, 13, 17], [37, 38, 66, 78, 116, 50, 57, 74], [37, 98, 6, 8, 47, 69, 65, 18], [3, 102, 0, 90, 35, 8, 63, 103], [8, 92, 73, 48, 2, 32, 41, 88], [81, 74, 109, 8, 67, 122, 45, 82], [63, 34, 90, 42, 111, 0, 39, 104]]
    rank6 = [[62, 43, 17, 73, 112, 80, 23, 75], [76, 22, 50, 48, 114, 33, 32, 120], [14, 49, 109, 57, 76, 77, 65, 123], [41, 28, 117, 97, 98, 106, 13, 1], [102, 88, 112, 29, 96, 47, 106, 22], [7, 61, 100, 35, 106, 111, 122, 26], [90, 34, 62, 123, 112, 92, 33, 114], [27, 87, 2, 113, 9, 108, 26, 35], [18, 2, 19, 28, 100, 47, 89, 78], [92, 52, 59, 1, 48, 98, 107, 32], [101, 123, 58, 102, 59, 111, 43, 83], [5, 49, 4, 24, 46, 111, 60, 90], [25, 118, 107, 125, 40, 54, 36, 91], [18, 32, 44, 102, 25, 62, 11, 57], [115, 123, 16, 67, 25, 11, 82, 69], [37, 72, 110, 120, 48, 90, 58, 23], [4, 54, 80, 60, 27, 33, 12, 30], [115, 45, 63, 68, 126, 17, 35, 0], [66, 70, 120, 13, 99, 108, 33, 82], [44, 56, 127, 31, 105, 106, 35, 92], [123, 20, 21, 51, 26, 40, 64, 2], [70, 116, 63, 114, 81, 79, 34, 58], [78, 99, 107, 119, 7, 73, 37, 6], [11, 124, 22, 86, 42, 78, 100, 68], [110, 120, 1, 7, 77, 38, 49, 0], [30, 93, 23, 75, 66, 18, 125, 109], [90, 73, 2, 16, 0, 72, 67, 64], [100, 26, 44, 95, 63, 20, 104, 70], [10, 12, 8, 76, 119, 28, 39, 27], [39, 54, 108, 110, 3, 115, 0, 37], [126, 120, 36, 66, 33, 108, 104, 14], [122, 74, 88, 51, 70, 82, 61, 83], [27, 13, 83, 19, 85, 101, 64, 96], [41, 117, 107, 32, 16, 114, 98, 94], [6, 56, 78, 54, 45, 58, 73, 14], [11, 6, 23, 32, 104, 19, 33, 113], [52, 24, 122, 11, 95, 18, 25, 51], [103, 18, 77, 29, 66, 75, 110, 10], [7, 117, 69, 106, 96, 71, 20, 10], [101, 32, 36, 34, 72, 48, 87, 71], [108, 111, 60, 39, 44, 40, 7, 6], [36, 63, 81, 118, 15, 46, 65, 126], [126, 7, 125, 113, 55, 8, 70, 59], [13, 7, 2, 43, 99, 113, 49, 12], [99, 70, 42, 34, 117, 39, 45, 41], [54, 124, 76, 46, 83, 47, 77, 27], [15, 79, 116, 120, 20, 97, 19, 33], [20, 15, 36, 17, 95, 31, 65, 30]]
    rank7 = [[77, 94, 14, 35, 1, 59, 41, 126], [83, 84, 41, 34, 113, 103, 123, 37], [90, 82, 4, 95, 72, 92, 80, 7], [113, 107, 94, 19, 51, 46, 33, 10], [40, 4, 67, 44, 58, 46, 119, 50], [68, 4, 58, 2, 70, 24, 54, 103], [98, 118, 55, 79, 66, 35, 21, 41], [76, 43, 106, 72, 56, 104, 110, 88], [27, 65, 90, 35, 4, 94, 13, 126], [26, 13, 117, 12, 79, 56, 40, 8], [63, 47, 103, 82, 33, 108, 37, 32], [125, 75, 86, 42, 119, 102, 85, 53], [50, 24, 33, 26, 11, 29, 96, 0], [117, 58, 97, 121, 115, 91, 36, 1], [18, 28, 62, 39, 114, 124, 34, 49], [20, 116, 14, 104, 51, 54, 36, 31], [91, 51, 106, 82, 73, 6, 78, 77], [15, 52, 54, 117, 58, 14, 118, 73], [79, 94, 103, 27, 109, 112, 47, 6], [118, 108, 88, 0, 67, 86, 39, 85], [38, 13, 126, 113, 66, 9, 47, 11], [26, 112, 48, 95, 98, 40, 17, 69], [70, 74, 83, 72, 23, 8, 33, 94], [114, 47, 53, 94, 104, 85, 26, 89], [53, 54, 75, 43, 28, 51, 70, 112], [90, 32, 36, 57, 26, 1, 121, 25], [114, 94, 19, 61, 28, 97, 10, 18], [36, 38, 123, 110, 5, 88, 15, 115], [108, 126, 77, 122, 15, 116, 4, 20], [63, 14, 2, 13, 45, 126, 52, 117], [57, 94, 44, 77, 74, 32, 69, 79], [27, 92, 126, 81, 114, 18, 127, 100], [20, 2, 97, 21, 113, 38, 50, 69], [78, 53, 116, 127, 40, 0, 13, 96], [37, 65, 107, 53, 88, 119, 99, 114], [54, 124, 120, 13, 58, 108, 42, 98], [26, 0, 73, 54, 107, 118, 38, 53], [125, 51, 57, 118, 42, 79, 105, 112], [11, 18, 3, 16, 30, 80, 9, 34], [115, 59, 20, 14, 12, 13, 60, 107], [77, 49, 1, 100, 69, 20, 29, 41], [115, 73, 97, 52, 75, 68, 54, 66], [68, 118, 49, 25, 20, 53, 71, 101], [55, 46, 4, 38, 81, 61, 111, 19], [32, 104, 110, 11, 126, 9, 19, 12], [112, 17, 127, 34, 110, 113, 63, 101], [26, 10, 124, 41, 29, 92, 105, 14], [22, 71, 55, 119, 80, 27, 108, 61]]
    rank8 = [[11, 104, 28, 44, 31, 46, 37, 87], [38, 44, 85, 74, 35, 88, 78, 99], [61, 69, 119, 29, 68, 52, 117, 6], [27, 42, 80, 100, 77, 11, 90, 84], [74, 70, 121, 83, 20, 91, 82, 27], [6, 84, 93, 31, 110, 92, 74, 18], [116, 43, 63, 94, 31, 126, 80, 72], [67, 61, 86, 118, 71, 115, 96, 1], [26, 24, 66, 83, 20, 9, 63, 123], [116, 74, 53, 35, 86, 49, 22, 58], [74, 19, 70, 81, 109, 90, 92, 97], [77, 23, 13, 48, 40, 6, 68, 50], [68, 28, 120, 43, 37, 122, 72, 113], [127, 37, 24, 55, 118, 79, 105, 41], [108, 8, 0, 98, 53, 90, 47, 122], [88, 115, 12, 7, 76, 13, 56, 77], [29, 15, 111, 20, 110, 94, 121, 123], [75, 65, 105, 90, 2, 100, 26, 30], [77, 38, 21, 101, 69, 22, 104, 110], [126, 81, 114, 68, 78, 82, 115, 17], [118, 4, 124, 32, 42, 96, 69, 49], [78, 12, 27, 62, 4, 99, 109, 53], [0, 90, 86, 32, 109, 59, 19, 65], [46, 79, 120, 73, 32, 81, 13, 77], [68, 125, 11, 18, 4, 20, 118, 25], [44, 59, 45, 55, 123, 89, 41, 24], [60, 113, 27, 39, 82, 62, 109, 24], [14, 49, 126, 54, 89, 122, 16, 12], [30, 29, 51, 121, 73, 111, 106, 6], [86, 12, 26, 15, 32, 89, 114, 66], [0, 53, 55, 70, 68, 3, 9, 62], [106, 40, 84, 96, 41, 31, 64, 68], [117, 24, 47, 40, 3, 126, 4, 76], [99, 26, 8, 111, 23, 77, 62, 36], [90, 33, 86, 109, 42, 32, 102, 29], [111, 31, 77, 30, 48, 86, 92, 114], [91, 4, 119, 120, 23, 75, 110, 50], [24, 36, 121, 44, 55, 1, 73, 123], [74, 22, 62, 28, 60, 38, 97, 53], [119, 100, 76, 2, 58, 38, 116, 37], [15, 89, 109, 91, 123, 28, 121, 79], [124, 30, 1, 0, 110, 45, 2, 90], [4, 30, 119, 73, 75, 106, 88, 65], [67, 70, 11, 114, 101, 125, 80, 68], [114, 100, 37, 76, 30, 88, 13, 82], [40, 105, 93, 87, 56, 31, 81, 16], [64, 39, 70, 93, 35, 40, 30, 101], [2, 35, 79, 51, 6, 11, 46, 54]]
    rank9 = [[29, 118, 6, 89, 34, 111, 125, 22], [126, 16, 52, 8, 49, 60, 29, 66], [51, 115, 89, 104, 24, 97, 126, 36], [54, 108, 15, 48, 24, 34, 92, 7], [0, 11, 2, 8, 33, 105, 15, 10], [20, 95, 96, 23, 114, 48, 123, 105], [37, 127, 104, 49, 9, 22, 111, 103], [55, 39, 97, 78, 53, 0, 63, 5], [51, 113, 76, 11, 110, 96, 50, 8], [114, 122, 34, 46, 27, 10, 30, 3], [122, 35, 96, 118, 9, 91, 86, 78], [89, 41, 94, 114, 71, 69, 92, 32], [38, 47, 126, 73, 77, 51, 103, 18], [113, 90, 123, 71, 119, 5, 27, 93], [15, 76, 78, 97, 60, 96, 83, 24], [79, 112, 109, 119, 24, 55, 9, 70], [114, 37, 100, 120, 99, 72, 1, 85], [92, 32, 1, 64, 43, 23, 84, 12], [89, 41, 67, 53, 3, 62, 49, 125], [61, 91, 122, 18, 109, 96, 73, 104], [106, 14, 67, 78, 56, 3, 50, 116], [86, 96, 30, 13, 23, 10, 8, 52], [98, 42, 122, 81, 77, 108, 40, 97], [87, 108, 54, 29, 69, 31, 102, 6], [59, 29, 127, 56, 106, 47, 107, 71], [103, 112, 102, 74, 113, 79, 118, 119], [117, 115, 89, 5, 38, 71, 47, 85], [120, 116, 112, 13, 84, 46, 79, 66], [109, 110, 123, 41, 34, 99, 50, 87], [65, 35, 17, 64, 124, 105, 84, 123], [99, 111, 47, 59, 116, 67, 121, 109], [28, 5, 86, 67, 19, 36, 17, 48], [51, 37, 127, 90, 74, 100, 118, 49], [52, 112, 48, 12, 95, 27, 10, 4], [111, 35, 103, 43, 0, 98, 82, 51], [107, 112, 96, 46, 89, 81, 64, 63], [71, 29, 1, 113, 17, 77, 34, 56], [109, 102, 49, 115, 12, 62, 27, 82], [91, 84, 73, 111, 87, 15, 120, 99], [40, 84, 98, 89, 19, 81, 23, 90], [110, 4, 62, 25, 16, 74, 51, 114], [120, 114, 14, 100, 123, 64, 20, 35], [10, 90, 87, 102, 98, 28, 47, 86], [24, 74, 16, 86, 41, 91, 54, 56], [23, 4, 80, 116, 71, 48, 96, 54], [98, 19, 61, 103, 26, 119, 86, 4], [53, 96, 58, 11, 69, 4, 17, 47], [82, 98, 57, 120, 67, 47, 114, 123]]
    rank10 = [[69, 124, 67, 9, 71, 113, 101, 51], [19, 56, 21, 63, 57, 25, 68, 69], [63, 85, 59, 111, 124, 13, 42, 22], [87, 23, 70, 56, 26, 16, 43, 123], [53, 87, 110, 104, 100, 39, 14, 37], [90, 113, 41, 102, 30, 10, 120, 62], [108, 110, 69, 47, 32, 28, 8, 125], [68, 120, 119, 91, 18, 12, 81, 36], [42, 116, 14, 67, 127, 49, 58, 111], [76, 62, 63, 113, 61, 38, 50, 67], [40, 20, 119, 80, 110, 72, 116, 42], [79, 78, 47, 17, 120, 51, 29, 64], [39, 32, 95, 16, 56, 45, 62, 76], [42, 89, 22, 12, 30, 109, 124, 120], [74, 85, 71, 27, 7, 94, 30, 102], [84, 43, 117, 98, 126, 111, 2, 122], [24, 70, 79, 83, 127, 38, 11, 34], [3, 71, 20, 79, 93, 89, 66, 22], [8, 80, 18, 113, 68, 34, 86, 75], [12, 10, 33, 29, 14, 53, 110, 28], [23, 111, 8, 6, 76, 70, 55, 105], [38, 51, 111, 115, 80, 37, 56, 118], [102, 53, 96, 20, 92, 124, 29, 22], [119, 18, 71, 50, 64, 57, 41, 30], [95, 76, 24, 105, 79, 19, 103, 113], [42, 124, 110, 53, 2, 12, 105, 22], [15, 76, 66, 96, 7, 87, 121, 91], [109, 51, 111, 92, 59, 73, 8, 56], [70, 79, 53, 21, 13, 114, 72, 11], [83, 20, 16, 81, 116, 92, 78, 101], [41, 127, 27, 63, 101, 39, 18, 103], [78, 91, 12, 22, 121, 124, 1, 15], [10, 116, 42, 11, 14, 67, 56, 115], [115, 68, 38, 30, 86, 106, 51, 25], [70, 120, 72, 108, 110, 40, 34, 124], [119, 47, 87, 18, 117, 71, 84, 97], [43, 16, 103, 31, 19, 62, 79, 112], [45, 41, 113, 104, 5, 99, 72, 13], [55, 85, 64, 72, 47, 41, 65, 76], [8, 42, 51, 91, 126, 46, 120, 33], [78, 27, 43, 33, 87, 116, 105, 46], [82, 51, 62, 84, 122, 117, 89, 93], [2, 93, 72, 112, 45, 18, 61, 85], [59, 52, 121, 35, 60, 48, 27, 83], [65, 120, 52, 108, 26, 22, 44, 66], [106, 43, 70, 67, 123, 6, 72, 57], [18, 83, 127, 51, 110, 63, 56, 42], [52, 105, 19, 118, 59, 94, 88, 26]]
    rank11 = [[83, 24, 32, 38, 13, 0, 66, 79], [20, 47, 110, 77, 75, 51, 30, 116], [48, 11, 12, 67, 10, 103, 35, 0], [50, 120, 55, 47, 12, 66, 68, 20], [21, 86, 84, 80, 81, 65, 35, 5], [39, 56, 87, 53, 36, 86, 83, 46], [18, 89, 67, 54, 38, 120, 86, 59], [122, 125, 48, 38, 22, 100, 117, 105], [70, 23, 95, 10, 55, 118, 3, 124], [20, 39, 78, 121, 37, 5, 11, 95], [38, 25, 22, 98, 0, 53, 124, 65], [108, 104, 39, 96, 74, 7, 81, 2], [46, 100, 124, 101, 79, 57, 27, 23], [82, 125, 2, 7, 35, 14, 65, 13], [19, 2, 36, 29, 121, 120, 106, 66], [61, 69, 33, 97, 59, 64, 74, 10], [41, 13, 16, 10, 23, 47, 43, 14], [78, 87, 16, 62, 47, 31, 33, 37], [37, 4, 25, 90, 39, 74, 59, 19], [22, 71, 121, 5, 124, 125, 97, 1], [94, 10, 0, 100, 127, 120, 115, 119], [59, 36, 39, 11, 1, 106, 5, 124], [35, 110, 34, 121, 84, 82, 38, 85], [15, 17, 92, 84, 48, 60, 126, 97], [45, 126, 23, 67, 32, 62, 37, 109], [65, 82, 13, 51, 91, 72, 84, 31], [59, 102, 8, 13, 34, 106, 104, 29], [24, 69, 2, 23, 31, 61, 21, 9], [83, 88, 74, 23, 127, 94, 43, 69], [96, 100, 58, 120, 79, 22, 51, 121], [110, 50, 51, 117, 98, 125, 90, 19], [109, 110, 76, 11, 113, 97, 39, 4], [105, 23, 1, 94, 8, 78, 119, 54], [37, 34, 118, 109, 39, 1, 67, 59], [84, 8, 74, 97, 22, 20, 59, 38], [50, 79, 73, 17, 29, 69, 85, 41], [68, 10, 40, 57, 101, 67, 2, 106], [71, 89, 2, 22, 31, 90, 86, 106], [104, 77, 82, 115, 124, 36, 102, 23], [27, 61, 54, 50, 24, 73, 78, 79], [13, 37, 14, 70, 21, 10, 9, 47], [58, 31, 37, 106, 92, 104, 96, 70], [22, 36, 123, 41, 84, 34, 105, 1], [23, 108, 20, 84, 51, 124, 106, 32], [20, 78, 60, 61, 36, 69, 40, 28], [120, 97, 5, 37, 42, 114, 33, 104], [78, 99, 48, 62, 119, 49, 85, 36], [23, 13, 126, 12, 21, 78, 38, 121]]
    rank12 = [[81, 47, 12, 92, 18, 3, 74, 110], [61, 92, 7, 43, 45, 98, 117, 62], [17, 15, 125, 70, 118, 26, 114, 121], [109, 25, 127, 69, 40, 111, 118, 67], [77, 89, 116, 1, 95, 107, 117, 38], [32, 67, 63, 12, 75, 5, 127, 99], [40, 113, 25, 93, 122, 39, 84, 20], [29, 89, 21, 4, 11, 6, 92, 64], [60, 106, 61, 86, 6, 54, 68, 31], [87, 68, 51, 118, 106, 73, 4, 115], [76, 120, 95, 64, 67, 84, 114, 125], [116, 117, 115, 55, 97, 16, 21, 76], [81, 108, 31, 44, 1, 119, 19, 10], [110, 74, 84, 10, 96, 86, 40, 85], [5, 91, 87, 72, 104, 6, 113, 59], [114, 63, 26, 15, 17, 39, 27, 19], [98, 31, 49, 75, 58, 46, 53, 25], [96, 67, 124, 101, 25, 83, 120, 106], [5, 127, 7, 102, 72, 32, 51, 93], [120, 15, 76, 100, 11, 94, 64, 89], [34, 108, 37, 54, 74, 93, 44, 61], [77, 20, 89, 126, 67, 87, 25, 42], [67, 41, 43, 28, 120, 51, 115, 89], [66, 16, 112, 55, 76, 40, 39, 70], [50, 100, 101, 40, 57, 44, 72, 93], [116, 9, 35, 104, 43, 50, 120, 127], [35, 53, 116, 120, 95, 124, 108, 75], [97, 39, 64, 7, 98, 81, 43, 35], [56, 91, 46, 16, 31, 78, 32, 14], [33, 70, 8, 23, 62, 40, 94, 4], [25, 38, 84, 34, 75, 114, 72, 6], [73, 85, 125, 38, 69, 33, 13, 115], [75, 44, 61, 106, 29, 55, 6, 25], [35, 76, 11, 123, 75, 80, 121, 5], [83, 96, 95, 67, 77, 117, 115, 41], [76, 66, 68, 43, 16, 62, 60, 55], [27, 14, 124, 37, 45, 98, 100, 32], [84, 124, 35, 119, 68, 9, 127, 108], [13, 89, 121, 24, 29, 6, 8, 94], [55, 17, 7, 69, 56, 43, 21, 77], [11, 122, 127, 84, 94, 83, 57, 23], [43, 112, 103, 121, 101, 111, 32, 79], [109, 60, 110, 23, 96, 44, 24, 19], [112, 73, 97, 105, 26, 87, 1, 63], [56, 101, 115, 123, 18, 119, 55, 122], [10, 28, 79, 22, 35, 9, 30, 39], [102, 23, 55, 125, 0, 7, 13, 117], [86, 69, 84, 18, 16, 66, 50, 10]]
    rank13 = [[116, 33, 4, 64, 68, 54, 105, 60], [4, 11, 121, 104, 109, 105, 24, 46], [86, 74, 64, 33, 62, 50, 101, 96], [61, 30, 114, 35, 71, 121, 62, 110], [43, 59, 125, 93, 34, 114, 62, 3], [78, 117, 81, 45, 88, 28, 21, 16], [91, 50, 46, 70, 102, 97, 7, 10], [66, 84, 73, 28, 13, 124, 57, 33], [117, 44, 7, 25, 120, 122, 73, 37], [112, 25, 124, 57, 89, 93, 60, 36], [34, 106, 29, 41, 121, 89, 28, 85], [101, 15, 70, 84, 62, 34, 66, 112], [67, 59, 106, 109, 58, 5, 93, 34], [48, 126, 64, 31, 108, 68, 9, 83], [116, 41, 9, 95, 75, 37, 57, 35], [5, 78, 44, 21, 35, 96, 50, 82], [84, 107, 116, 57, 74, 122, 52, 69], [94, 6, 104, 116, 103, 51, 27, 112], [124, 121, 92, 117, 56, 46, 28, 85], [4, 63, 13, 6, 21, 66, 116, 36], [29, 117, 122, 86, 62, 7, 1, 25], [93, 71, 22, 15, 46, 35, 66, 24], [114, 80, 69, 1, 106, 95, 125, 25], [51, 38, 110, 62, 21, 82, 74, 116], [27, 2, 58, 98, 10, 46, 119, 115], [14, 83, 17, 62, 64, 85, 96, 68], [30, 36, 37, 58, 55, 6, 54, 41], [27, 58, 17, 77, 114, 99, 78, 50], [120, 37, 104, 64, 85, 84, 24, 52], [27, 6, 43, 67, 125, 47, 103, 91], [4, 92, 86, 56, 93, 102, 7, 21], [26, 0, 57, 53, 71, 105, 89, 94], [70, 80, 120, 0, 62, 108, 7, 31], [97, 22, 124, 71, 87, 60, 15, 20], [121, 76, 80, 17, 89, 1, 25, 125], [15, 39, 101, 51, 57, 21, 116, 126], [125, 44, 84, 59, 109, 115, 86, 72], [59, 74, 85, 65, 14, 120, 101, 96], [123, 35, 59, 58, 75, 113, 56, 114], [82, 15, 114, 97, 111, 9, 99, 74], [75, 31, 85, 53, 32, 64, 120, 61], [40, 94, 8, 71, 67, 22, 60, 9], [115, 97, 121, 67, 79, 40, 54, 3], [40, 45, 93, 89, 109, 76, 39, 126], [79, 24, 21, 49, 98, 87, 91, 50], [80, 108, 3, 116, 117, 102, 109, 14], [61, 115, 95, 50, 98, 52, 107, 24], [56, 103, 70, 96, 100, 7, 64, 85]]
    rank14 = [[36, 25, 39, 91, 86, 122, 82, 42], [2, 54, 91, 40, 81, 5, 39, 12], [23, 81, 41, 39, 44, 107, 120, 79], [58, 95, 73, 65, 8, 44, 78, 3], [51, 113, 6, 61, 9, 122, 36, 127], [55, 85, 115, 65, 119, 19, 27, 94], [4, 75, 5, 57, 61, 48, 23, 81], [90, 116, 23, 94, 99, 3, 44, 93], [93, 1, 75, 74, 29, 80, 115, 104], [126, 42, 66, 24, 29, 77, 71, 109], [18, 115, 10, 77, 117, 79, 62, 52], [82, 33, 43, 38, 106, 113, 105, 126], [90, 82, 112, 42, 115, 84, 87, 86], [94, 101, 51, 43, 50, 54, 100, 106], [112, 99, 56, 58, 54, 44, 40, 31], [68, 73, 3, 42, 94, 47, 62, 121], [87, 64, 67, 59, 125, 56, 19, 88], [8, 91, 98, 40, 70, 9, 125, 24], [97, 84, 64, 48, 100, 114, 23, 36], [84, 90, 23, 69, 38, 40, 102, 3], [75, 68, 31, 80, 91, 63, 82, 104], [57, 76, 113, 121, 97, 103, 75, 68], [76, 64, 62, 30, 17, 10, 60, 18], [96, 115, 2, 101, 33, 12, 43, 105], [34, 42, 14, 17, 5, 86, 31, 102], [71, 39, 100, 86, 10, 101, 48, 106], [31, 45, 57, 40, 50, 122, 103, 44], [96, 55, 47, 6, 121, 40, 82, 94], [57, 9, 61, 38, 67, 80, 102, 19], [31, 71, 106, 95, 113, 104, 25, 111], [61, 80, 28, 23, 100, 5, 97, 113], [63, 99, 25, 6, 66, 90, 3, 120], [86, 63, 68, 73, 122, 91, 93, 110], [42, 29, 24, 89, 66, 54, 126, 102], [85, 69, 64, 91, 10, 30, 50, 18], [115, 110, 40, 38, 82, 105, 106, 70], [82, 58, 46, 81, 93, 42, 102, 49], [94, 83, 30, 126, 53, 3, 48, 50], [122, 39, 31, 116, 44, 66, 103, 108], [68, 64, 109, 112, 35, 44, 1, 39], [38, 52, 59, 80, 88, 125, 56, 86], [98, 6, 91, 87, 23, 116, 25, 27], [83, 5, 82, 13, 29, 62, 15, 103], [64, 33, 22, 36, 9, 44, 42, 103], [83, 92, 10, 85, 77, 38, 74, 93], [1, 62, 36, 100, 122, 55, 126, 13], [91, 65, 80, 60, 72, 112, 66, 73], [89, 49, 5, 45, 40, 81, 29, 33]]
    rank15 = [[10, 5, 95, 20, 90, 56, 15, 55], [93, 73, 115, 67, 13, 14, 55, 0], [73, 28, 3, 106, 43, 5, 19, 105], [57, 119, 52, 85, 22, 125, 83, 96], [123, 94, 45, 97, 18, 68, 41, 42], [116, 11, 57, 60, 121, 89, 118, 3], [19, 117, 85, 0, 36, 45, 100, 64], [69, 25, 40, 98, 77, 20, 102, 121], [33, 62, 34, 119, 0, 91, 82, 43], [97, 123, 72, 54, 103, 15, 75, 102], [17, 30, 44, 104, 1, 50, 49, 60], [110, 91, 12, 80, 88, 0, 57, 83], [61, 17, 14, 102, 98, 70, 6, 88], [53, 70, 116, 72, 46, 3, 17, 56], [23, 3, 13, 50, 77, 103, 65, 45], [92, 81, 91, 6, 86, 1, 113, 99], [117, 9, 104, 86, 102, 18, 61, 96], [60, 4, 95, 114, 113, 111, 102, 107], [10, 57, 0, 40, 50, 61, 45, 31], [93, 48, 77, 99, 98, 20, 117, 25], [33, 43, 110, 95, 90, 58, 60, 73], [60, 54, 102, 123, 73, 3, 29, 72], [104, 50, 91, 52, 44, 49, 117, 79], [0, 91, 83, 34, 93, 80, 88, 106], [61, 90, 87, 82, 6, 84, 81, 88], [70, 40, 54, 56, 5, 94, 46, 3], [23, 77, 65, 3, 99, 112, 56, 123], [3, 113, 74, 86, 42, 1, 91, 62], [18, 59, 75, 117, 96, 86, 125, 47], [98, 112, 9, 87, 24, 102, 60, 107], [46, 10, 48, 40, 85, 64, 31, 45], [117, 116, 102, 21, 98, 23, 20, 93], [34, 43, 82, 104, 33, 95, 60, 58], [57, 3, 72, 93, 113, 103, 46, 73], [49, 106, 62, 52, 60, 104, 44, 79], [12, 34, 93, 83, 80, 88, 0, 2], [87, 76, 5, 61, 90, 70, 6, 88], [70, 43, 40, 116, 64, 17, 46, 56], [112, 37, 40, 95, 57, 54, 50, 45], [121, 113, 3, 62, 94, 96, 6, 86], [117, 67, 102, 18, 104, 19, 72, 96], [113, 125, 102, 95, 107, 47, 4, 24], [122, 16, 26, 95, 0, 51, 11, 107], [14, 50, 104, 85, 127, 116, 95, 119], [81, 75, 86, 6, 27, 118, 89, 124], [12, 96, 60, 59, 24, 84, 25, 68], [108, 12, 75, 111, 59, 22, 87, 121], [41, 75, 24, 68, 97, 43, 60, 91]]
    rank_by_ep = [rank0, rank1, rank2, rank3, rank4, rank5, rank6, rank7, rank8,
    rank9, rank10, rank11, rank12, rank13, rank14, rank15]
    p2l = [sum(group, []) for group in zip(*rank_by_ep)][layer_idx]
    l2p = [0] * len(p2l)
    for p, l in enumerate(p2l):
        l2p[l] = p
    # print("p2l:", len(p2l), p2l)
    # print("l2p:", len(l2p), l2p)
    l2p = torch.tensor(l2p, dtype=torch.int32)
    rank_list = rank_by_ep[ep_rank]
    # 该 rank 在此层的全局专家编号
    idx = torch.as_tensor(rank_list[layer_idx], dtype=torch.int32, device=expert_map.device)
    # 生成本地 id：0..num_local-1（自适应本 rank 的专家个数）
    local_ids = torch.arange(idx.numel(), dtype=torch.int32, device=expert_map.device)
    # 写回映射：global_id -> local_id
    expert_map[idx] = local_ids
    #print("use custom reverse-greedy expert placement for ep_rank", ep_rank, "layer_idx", layer_idx)
    # print("idx is", idx.size(), idx)
    # print("local num_experts:", local_num_experts)
    # print("expert_map:", expert_map.size(), expert_map)
    print("l2p is:", l2p.size(), l2p)
    return (local_num_experts, expert_map, l2p)
    '''
    '''
    # ############ Round_robin expert placement logic ############
    # # Create an expert map for the local experts
    expert_placement_strategy = "round_robin"
    if expert_placement_strategy == "linear":
        start_idx = ep_rank * base_experts + min(ep_rank, remainder)
        expert_map[start_idx:start_idx + local_num_experts] = torch.arange(
            0, local_num_experts, dtype=torch.int32)
    elif expert_placement_strategy == "round_robin":
        local_log_experts = torch.arange(ep_rank,
                                         global_num_experts,
                                         ep_size,
                                         dtype=torch.int32)

        expert_map[local_log_experts] = torch.arange(0,
                                                     local_num_experts,
                                                     dtype=torch.int32)
    else:
        raise ValueError("Unsupported expert placement strategy "
                         f"'{expert_placement_strategy}', expected one of "
                         f"{get_args(ExpertPlacementStrategy)}")
    # print("use original expert placement for ep_rank", ep_rank, "layer_idx", layer_idx)
    print("original local num_experts:", local_num_experts)
    print("use round_robin expert placement for ep_rank", ep_rank, "original expert_map:", expert_map.size(), expert_map)
    #l2p = None
    p2l = [r + i * ep_size for r in range(ep_size) for i in range(global_num_experts // ep_size)]
    l2p = [0] * len(p2l)
    for p, l in enumerate(p2l):
        l2p[l] = p
    l2p = torch.tensor(l2p, dtype=torch.int32) 
    print("l2p is:", l2p.size(), l2p)
    return (local_num_experts, expert_map, l2p)
    '''

    
    # # ############ Original expert placement logic ############
    # # # Create an expert map for the local experts
    # if expert_placement_strategy == "linear":
    #     start_idx = ep_rank * base_experts + min(ep_rank, remainder)
    #     expert_map[start_idx:start_idx + local_num_experts] = torch.arange(
    #         0, local_num_experts, dtype=torch.int32)
    # elif expert_placement_strategy == "round_robin":
    #     local_log_experts = torch.arange(ep_rank,
    #                                      global_num_experts,
    #                                      ep_size,
    #                                      dtype=torch.int32)

    #     expert_map[local_log_experts] = torch.arange(0,
    #                                                  local_num_experts,
    #                                                  dtype=torch.int32)
    # else:
    #     raise ValueError("Unsupported expert placement strategy "
    #                      f"'{expert_placement_strategy}', expected one of "
    #                      f"{get_args(ExpertPlacementStrategy)}")
    # # print("use original expert placement for ep_rank", ep_rank, "layer_idx", layer_idx)
    # print("original local num_experts:", local_num_experts)
    # print("use original expert placement for ep_rank", ep_rank, "original expert_map:", expert_map.size(), expert_map)
    # #l2p = None
    # l2p = torch.arange(128, dtype=torch.int32)
    # print("l2p is:", l2p.size(), l2p)
    # return (local_num_experts, expert_map, l2p)
    

def get_compressed_expert_map(expert_map: torch.Tensor) -> str:
    """
        Compresses the expert map by removing any -1 entries.

        Args:
            expert_map (torch.Tensor): A tensor of shape (global_num_experts,)
                mapping from global to local index. Contains -1 for experts not
                assigned to the current rank.

        Returns:
            str: A string mapping from local to global index.
                Using str to support hashing for logging once only.
        """
    global_indices = torch.where(expert_map != -1)[0]
    local_indices = expert_map[global_indices]
    return ", ".join(
        f"{local_index.item()}->{global_index.item()}"
        for local_index, global_index in zip(local_indices, global_indices))


def maybe_roundup_hidden_size(
        hidden_size: int, act_dtype: torch.dtype,
        quant_config: Optional[QuantizationConfig],
        moe_parallel_config: FusedMoEParallelConfig) -> int:
    """
    Given layer hidden size and MoE configurations, round up hidden_size
    if necessary.
    
    Args:
        hidden_size: Layer hidden-size
        act_dtype: Data type of the layer activations.
        quant_config: Fused MoE quantization configuration.
        moe_parallel_config: Fused MoE parallelization strategy configuration.

    Return:
        Rounded up hidden_size if rounding up is required based on the configs.
        Original hidden size otherwise.
    """

    if (moe_parallel_config.use_deepep_ht_kernels):
        hidden_size = (
            DeepEPHTPrepareAndFinalize.maybe_roundup_layer_hidden_size(
                hidden_size, act_dtype))

    # we are padding globally so EP buffer allocation works
    if quant_config and quant_config.get_name() == "mxfp4":

        from vllm.model_executor.layers.quantization.mxfp4 import (
            Mxfp4Backend, get_mxfp4_backend)
        current_mxfp4_backend = get_mxfp4_backend()
        if (current_mxfp4_backend == Mxfp4Backend.SM90_FI_MXFP4_BF16
                or current_mxfp4_backend
                == Mxfp4Backend.SM100_FI_MXFP4_MXFP8_CUTLASS):
            hidden_size = round_up(hidden_size, 128)
        elif (current_platform.is_rocm() or current_mxfp4_backend
              == Mxfp4Backend.SM100_FI_MXFP4_MXFP8_TRTLLM
              or current_mxfp4_backend == Mxfp4Backend.SM100_FI_MXFP4_BF16):
            hidden_size = round_up(hidden_size, 256)

    return hidden_size


@CustomOp.register("fused_moe")
class FusedMoE(CustomOp):
    """FusedMoE layer for MoE models.

    This layer contains both MergedColumnParallel weights (gate_up_proj /
    w13) and RowParallelLinear weights (down_proj/ w2).

    Note: Mixtral uses w1, w2, and w3 for gate, up, and down_proj. We
    copy that naming convention here and handle any remapping in the
    load_weights function in each model implementation.

    Args:
        num_experts: Number of experts in the model
        top_k: Number of experts selected for each token
        hidden_size: Input hidden state size of the transformer
        intermediate_size: Intermediate size of the experts
        params_dtype: Data type for the parameters.
        reduce_results: Whether to all all_reduce on the output of the layer
        renormalize: Whether to renormalize the logits in the fused_moe kernel
        quant_config: Quantization configure.
        enable_eplb: Whether to enable expert parallelism load balancer.
    """

    def __init__(
        self,
        num_experts: int,  # Global number of experts
        top_k: int,
        hidden_size: int,
        intermediate_size: int,
        params_dtype: Optional[torch.dtype] = None,
        reduce_results: bool = False,
        renormalize: bool = True,
        use_grouped_topk: bool = False,
        num_expert_group: Optional[int] = None,
        topk_group: Optional[int] = None,
        quant_config: Optional[QuantizationConfig] = None,
        tp_size: Optional[int] = None,
        ep_size: Optional[int] = None,
        dp_size: Optional[int] = None,
        prefix: str = "",
        custom_routing_function: Optional[Callable] = None,
        scoring_func: str = "softmax",
        routed_scaling_factor: float = 1.0,
        e_score_correction_bias: Optional[torch.Tensor] = None,
        apply_router_weight_on_input: bool = False,
        activation: str = "silu",
        enable_eplb: bool = False,
        num_redundant_experts: int = 0,
        has_bias: bool = False,
        is_sequence_parallel=False,
        zero_expert_num: Optional[int] = 0,
        zero_expert_type: Optional[str] = None,
        #新增以下
        layer_idx: int = -1,
        #新增结束
    ):
        super().__init__()
        if params_dtype is None:
            params_dtype = torch.get_default_dtype()
        self.params_dtype = params_dtype
        #新增开始
        self.layer_idx = layer_idx
        #新增结束
        vllm_config = get_current_vllm_config()
        #calculate time
        self.total_run_time = 0
        self.total_comm_time = 0
        # FIXME (varun): We should have a better way of inferring the activation
        # datatype. This works for now as the tensor datatype entering the MoE
        # operation is typically unquantized (i.e. float16/bfloat16).
        if vllm_config.model_config is not None:
            moe_in_dtype = vllm_config.model_config.dtype
        else:
            # TODO (bnell): This is a hack to get test_mixtral_moe to work
            # since model_config is not set in the pytest test.
            moe_in_dtype = params_dtype

        tp_size_ = (tp_size if tp_size is not None else
                    get_tensor_model_parallel_world_size())
        dp_size_ = (dp_size
                    if dp_size is not None else get_dp_group().world_size)

        self.is_sequence_parallel = is_sequence_parallel
        self.sp_size = tp_size_ if is_sequence_parallel else 1

        self.moe_parallel_config: FusedMoEParallelConfig = (
            FusedMoEParallelConfig.make(
                tp_size_=tp_size_,
                dp_size_=dp_size_,
                vllm_parallel_config=vllm_config.parallel_config))
        mpc = self.moe_parallel_config
        print(f"[MoE] backend={envs.VLLM_ALL2ALL_BACKEND} pplx={mpc.use_pplx_kernels} deepep_ht={mpc.use_deepep_ht_kernels} deepep_ll={mpc.use_deepep_ll_kernels} dp={mpc.dp_size} ep={mpc.ep_size} tp={mpc.tp_size} use_all2all={mpc.dp_size>1 and mpc.use_ep}")

        self.global_num_experts = num_experts + num_redundant_experts
        self.zero_expert_num = zero_expert_num
        self.zero_expert_type = zero_expert_type

        # Round up hidden size if needed.
        hidden_size = maybe_roundup_hidden_size(hidden_size, moe_in_dtype,
                                                quant_config,
                                                self.moe_parallel_config)

        # For smuggling this layer into the fused moe custom op
        compilation_config = vllm_config.compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError("Duplicate layer name: {}".format(prefix))
        compilation_config.static_forward_context[prefix] = self
        self.layer_name = prefix

        self.enable_eplb = enable_eplb
        self.expert_load_view: Optional[torch.Tensor] = None
        self.logical_to_physical_map: Optional[torch.Tensor] = None
        self.logical_replica_count: Optional[torch.Tensor] = None
        #check if use ep
        # print("use_ep:", self.use_ep) 
        # print("enable_eplb:", self.enable_eplb)
        # Determine expert maps
        if self.use_ep:
            if self.enable_eplb:
                assert self.global_num_experts % self.ep_size == 0, \
                    "EPLB currently only supports even distribution of " \
                    "experts across ranks."
            else:
                assert num_redundant_experts == 0, \
                    "Redundant experts are only supported with EPLB."

            expert_placement_strategy = (
                vllm_config.parallel_config.expert_placement_strategy)
            if expert_placement_strategy == "round_robin":
                # TODO(Bruce): will support round robin expert placement with
                # EPLB enabled in the future.
                round_robin_supported = ((num_expert_group is not None
                                          and num_expert_group > 1)
                                         and num_redundant_experts == 0
                                         and not self.enable_eplb)

                if not round_robin_supported:
                    logger.warning(
                        "Round-robin expert placement is only supported for "
                        "models with multiple expert groups and no redundant "
                        "experts. Falling back to linear expert placement.")
                    expert_placement_strategy = "linear"

            self.expert_map: Optional[torch.Tensor]
            #在这里决定了初始专家分布
            local_num_experts, expert_map, log2phy = determine_expert_map(
                ep_size=self.ep_size,
                ep_rank=self.ep_rank,
                global_num_experts=self.global_num_experts,
                expert_placement_strategy=expert_placement_strategy,
                # 新增参数传递
                layer_idx=self.layer_idx,
            )
            # print("rank is {}, local_num_experts is {}, expert_map is {}".format(self.ep_rank, local_num_experts, expert_map))
            # print("CVD =", os.environ.get("CUDA_VISIBLE_DEVICES"))
            # print("torch.cuda.device_count() =", torch.cuda.device_count())
            # print("torch.cuda.current_device() =", torch.cuda.current_device())
            # print("tensor.device =", torch.empty(1, device="cuda").device)
            self.local_num_experts = local_num_experts
            self.register_buffer("expert_map", expert_map)
            logger.info_once(
                "[EP Rank %s/%s] Expert parallelism is enabled. Expert "
                "placement strategy: %s. Local/global"
                " number of experts: %s/%s. Experts local to global index map:"
                " %s.", self.ep_rank, self.ep_size, expert_placement_strategy,
                self.local_num_experts, self.global_num_experts,
                get_compressed_expert_map(self.expert_map))
        else:
            #这里注明不使用专家并行时每个worker拥有全部专家
            self.local_num_experts, self.expert_map = (self.global_num_experts,
                                                       None)

        self.top_k = top_k

        assert intermediate_size % self.tp_size == 0
        self.hidden_size = hidden_size
        self.intermediate_size_per_partition = intermediate_size // self.tp_size
        self.reduce_results = reduce_results
        self.renormalize = renormalize
        self.use_grouped_topk = use_grouped_topk
        if self.use_grouped_topk:
            assert num_expert_group is not None and topk_group is not None
        self.num_expert_group = num_expert_group
        self.topk_group = topk_group
        self.custom_routing_function = custom_routing_function
        self.scoring_func = scoring_func
        self.routed_scaling_factor = routed_scaling_factor
        self.e_score_correction_bias = e_score_correction_bias
        self.apply_router_weight_on_input = apply_router_weight_on_input
        self.activation = activation

        if self.scoring_func != "softmax" and not self.use_grouped_topk:
            raise ValueError("Only softmax scoring function is supported for "
                             "non-grouped topk.")

        moe = FusedMoEConfig(
            num_experts=self.global_num_experts,
            experts_per_token=top_k,
            hidden_dim=hidden_size,
            num_local_experts=self.local_num_experts,
            moe_parallel_config=self.moe_parallel_config,
            in_dtype=moe_in_dtype,
            max_num_tokens=envs.VLLM_MOE_DP_CHUNK_SIZE,
            has_bias=has_bias,
        )
        self.moe_config = moe
        self.moe_quant_config: Optional[FusedMoEQuantConfig] = None
        self.quant_config = quant_config

        # Note: get_quant_method will look at the layer's local_num_experts
        # for heuristic purposes, so it must be initialized first.
        quant_method: Optional[QuantizeMethodBase] = None
        quant_method = (UnquantizedFusedMoEMethod(moe) if quant_config is None
                        else quant_config.get_quant_method(self, prefix))

        assert quant_method is not None
        assert isinstance(quant_method, FusedMoEMethodBase)
        self.quant_method = quant_method

        if self.enable_eplb:
            from vllm.model_executor.layers.quantization.fp8 import (
                Fp8MoEMethod)
            if not isinstance(quant_method,
                              (Fp8MoEMethod, UnquantizedFusedMoEMethod)):
                # TODO: Add support for additional quantization methods.
                # The implementation for other quantization methods does not
                # contain essential differences, but the current quant API
                # design causes duplicated work when extending to new
                # quantization methods, so I'm leaving it for now.
                # If you plan to add support for more quantization methods,
                # please refer to the implementation in `Fp8MoEMethod`.
                raise NotImplementedError("EPLB is only supported for FP8 "
                                          "quantization for now.")

        moe_quant_params = {
            "num_experts": self.local_num_experts,
            "hidden_size": hidden_size,
            "intermediate_size_per_partition":
            self.intermediate_size_per_partition,
            "params_dtype": params_dtype,
            "weight_loader": self.weight_loader,
        }
        # need full intermediate size pre-sharding for WNA16 act order
        if (self.quant_method.__class__.__name__
                in ("GPTQMarlinMoEMethod",
                    "CompressedTensorsWNA16MarlinMoEMethod",
                    "CompressedTensorsWNA16MoEMethod")):
            moe_quant_params["intermediate_size_full"] = intermediate_size

        self.quant_method.create_weights(layer=self, **moe_quant_params)

        # Chunked all2all staging tensor
        self.batched_hidden_states: Optional[torch.Tensor] = None
        self.batched_router_logits: Optional[torch.Tensor] = None

        # TODO(bnell): flashinfer uses non-batched format.
        # Does it really need a batched buffer?
        if (self.moe_parallel_config.use_pplx_kernels
                or self.moe_parallel_config.use_deepep_ll_kernels
                or self.moe_config.use_flashinfer_cutlass_kernels):
            if vllm_config.parallel_config.enable_dbo:
                self.batched_hidden_states = torch.zeros(
                    (2, moe.max_num_tokens, self.hidden_size),
                    dtype=moe.in_dtype,
                    device=torch.cuda.current_device())

                # Note here we use `num_experts` which is logical expert count
                self.batched_router_logits = torch.zeros(
                    (2, moe.max_num_tokens, num_experts),
                    dtype=moe.in_dtype,
                    device=torch.cuda.current_device())
            else:
                self.batched_hidden_states = torch.zeros(
                    (moe.max_num_tokens, self.hidden_size),
                    dtype=moe.in_dtype,
                    device=torch.cuda.current_device())

                # Note here we use `num_experts` which is logical expert count
                self.batched_router_logits = torch.zeros(
                    (moe.max_num_tokens, num_experts),
                    dtype=moe.in_dtype,
                    device=torch.cuda.current_device())

    @property
    def shared_experts(self) -> Optional[torch.nn.Module]:
        return None

    @property
    def tp_size(self):
        return self.moe_parallel_config.tp_size

    @property
    def dp_size(self):
        return self.moe_parallel_config.dp_size

    @property
    def ep_size(self):
        return self.moe_parallel_config.ep_size

    @property
    def tp_rank(self):
        return self.moe_parallel_config.tp_rank

    @property
    def dp_rank(self):
        return self.moe_parallel_config.dp_rank

    @property
    def ep_rank(self):
        return self.moe_parallel_config.ep_rank

    @property
    def use_ep(self):
        return self.moe_parallel_config.use_ep

    @property
    def use_pplx_kernels(self):
        return self.moe_parallel_config.use_pplx_kernels

    @property
    def use_deepep_ht_kernels(self):
        return self.moe_parallel_config.use_deepep_ht_kernels

    @property
    def use_deepep_ll_kernels(self):
        return self.moe_parallel_config.use_deepep_ll_kernels

    @property
    def use_flashinfer_cutlass_kernels(self):
        return (self.moe_quant_config is not None
                and self.moe_quant_config.quant_dtype == "nvfp4"
                and self.moe_config.use_flashinfer_cutlass_kernels)

    def update_expert_map(self):
        # ep_size and ep_rank should already be updated
        assert self.expert_map is not None
        with self.expert_map.device:
            local_num_experts, expert_map = determine_expert_map(
                ep_size=self.ep_size,
                ep_rank=self.ep_rank,
                global_num_experts=self.global_num_experts)
            self.local_num_experts = local_num_experts
            self.register_buffer("expert_map", expert_map)

    def _load_per_tensor_weight_scale(self, shard_id: str,
                                      param: torch.nn.Parameter,
                                      loaded_weight: torch.Tensor,
                                      expert_id: int):
        param_data = param.data
        # for per tensor weight quantization
        if shard_id in ("w1", "w3"):
            # We have to keep the weight scales of w1 and w3 because
            # we need to re-quantize w1/w3 weights after weight loading.
            idx = 0 if shard_id == "w1" else 1
            param_data[expert_id][idx] = loaded_weight
        # If we are in the row parallel case (down_proj)
        elif shard_id == "w2":
            param_data[expert_id] = loaded_weight

    def _load_combined_w13_weight_scale(self, shard_dim: int,
                                        loaded_weight: torch.Tensor,
                                        param: torch.Tensor, tp_rank: int):
        """
        Load w13 weight scales assuming that w1 weight scales and w3 weight
        scales are stored in the same loaded_weight tensor.
        """
        shard_size = param.shape[shard_dim]
        loaded_weight = loaded_weight.narrow(shard_dim, shard_size * tp_rank,
                                             shard_size)
        param.copy_(loaded_weight)

    def _load_model_weight_or_group_weight_scale(self,
                                                 shard_dim: int,
                                                 expert_data: torch.Tensor,
                                                 shard_id: str,
                                                 loaded_weight: torch.Tensor,
                                                 tp_rank: int,
                                                 load_full_w2: bool = False):
        """
        Load grouped weight scales for group quantization or model weights
            :param shard_dim: dimension to shard
            :param expert_data: parameter for a particular expert
            :param shard_id: either w1, w2, or w3
            :param loaded_weight: checkpoint weight to load into the param
            :param tp_rank: tensor parallel rank
            :param load_full_w2: whether or not the w2 loaded should be sharded.
        """
        if shard_id == "w2":
            # In the case where we have actorder/g_idx, we do not partition the
            # w2 scales, as indicated by `load_full` argument, for all tp cases
            self._load_w2(shard_dim=shard_dim,
                          loaded_weight=loaded_weight,
                          expert_data=expert_data,
                          tp_rank=tp_rank,
                          load_full=load_full_w2)
        elif shard_id in ("w1", "w3"):
            self._load_w13(shard_id=shard_id,
                           shard_dim=shard_dim,
                           loaded_weight=loaded_weight,
                           expert_data=expert_data,
                           tp_rank=tp_rank)

    def _load_per_channel_weight_scale(self, expert_data: torch.Tensor,
                                       shard_dim: int, shard_id: str,
                                       loaded_weight: torch.Tensor,
                                       tp_rank: int):
        # for per channel weight quantization
        if shard_id == "w2":
            expert_data.copy_(loaded_weight)
        elif shard_id in ("w1", "w3"):
            self._load_w13(shard_id=shard_id,
                           shard_dim=shard_dim,
                           loaded_weight=loaded_weight,
                           expert_data=expert_data,
                           tp_rank=tp_rank)

    def _load_w13(self,
                  expert_data: torch.Tensor,
                  shard_dim: int,
                  shard_id: str,
                  loaded_weight: torch.Tensor,
                  tp_rank: int,
                  load_full: bool = False):
        # #debug
        if loaded_weight.numel() == 0 or loaded_weight.size(shard_dim) == 0:
            raise RuntimeError(
                f"[empty loaded_weight] shard_id={shard_id} shard_dim={shard_dim} "
                f"loaded_shape={tuple(loaded_weight.shape)} "
                f"expert_shape={tuple(expert_data.shape)} tp_rank={tp_rank}"
            )

        # Index the loaded weight for tp sharding.
        # gate_up_proj: "MergedColumnParallel", so tp sharding on output_dim
        shard_size = expert_data.shape[shard_dim] // 2
        if not load_full:
            loaded_weight = loaded_weight.narrow(shard_dim,
                                                 shard_size * tp_rank,
                                                 shard_size)
        # Narrow parameter and load.
        # w1, gate_proj: Load into first logical weight of w13.
        if shard_id == "w1":
            expert_data = expert_data.narrow(shard_dim, 0, shard_size)
        # w3, up_proj: Load into second logical weight of w13.
        else:
            assert shard_id == "w3"
            expert_data = expert_data.narrow(shard_dim, shard_size, shard_size)
        expert_data.copy_(loaded_weight)

    def _load_w2(self,
                 expert_data: torch.Tensor,
                 shard_dim: int,
                 loaded_weight: torch.Tensor,
                 tp_rank: int,
                 load_full: bool = False):

        # Index the loaded weight for tp sharding.
        # down_proj: "RowParallel" so tp sharding on input_dim
        # Narrow parameter and load.
        shard_size = expert_data.shape[shard_dim]
        if not load_full:
            loaded_weight = loaded_weight.narrow(shard_dim,
                                                 shard_size * tp_rank,
                                                 shard_size)
        # w2, down_proj: Load into only logical weight of w2.
        expert_data.copy_(loaded_weight)

    def _load_single_value(self, param: torch.nn.Parameter,
                           loaded_weight: torch.Tensor, expert_id: int):
        param_data = param.data

        # Input scales can be loaded directly and should be equal.
        param_data[expert_id] = loaded_weight

    def _load_g_idx(self, shard_id: str, expert_data: torch.Tensor,
                    shard_dim: int, loaded_weight: torch.Tensor, tp_rank: int):

        if shard_id == "w2":
            self._load_w2(shard_dim=shard_dim,
                          loaded_weight=loaded_weight,
                          expert_data=expert_data,
                          tp_rank=tp_rank)
        else:
            assert shard_id in ("w1", "w3")
            expert_data.copy_(loaded_weight)

    def _map_global_expert_id_to_local_expert_id(self, expert_id: int) -> int:
        if self.expert_map is None:
            return expert_id
        return self.expert_map[expert_id].item()

    @overload
    def weight_loader(self, param: torch.nn.Parameter,
                      loaded_weight: torch.Tensor, weight_name: str,
                      shard_id: str, expert_id: int,
                      return_success: Literal[False]) -> None:
        ...

    @overload
    def weight_loader(self, param: torch.nn.Parameter,
                      loaded_weight: torch.Tensor, weight_name: str,
                      shard_id: str, expert_id: int,
                      return_success: Literal[True]) -> bool:
        ...

    def weight_loader(self,
                      param: torch.nn.Parameter,
                      loaded_weight: torch.Tensor,
                      weight_name: str,
                      shard_id: str,
                      expert_id: int,
                      return_success: bool = False) -> Optional[bool]:

        if self.quant_config and self.quant_config.get_name() == "mxfp4":
            # (FIXME) for gpt-oss all experts are combined
            if "bias" in weight_name:
                dim1 = loaded_weight.shape[1]
                param.data[:, :dim1].copy_(loaded_weight)
            else:
                dim1 = loaded_weight.shape[1]
                dim2 = loaded_weight.shape[2]
                param.data[:, :dim1, :dim2].copy_(loaded_weight)
            return True if return_success else None
        # if self.ep_rank == 0:
        #     print("rank", self.ep_rank, "loading weight:", loaded_weight.shape)
        #     print("ep_rank", self.ep_rank, "before map, expert_id:", expert_id)
        expert_id = self._map_global_expert_id_to_local_expert_id(expert_id)
        # if self.ep_rank == 0:
        #     print("ep_rank", self.ep_rank,"after map, expert_id:", expert_id)
        # print(f"ep_rank: {self.ep_rank}, local expert_id: {expert_id}")

        if expert_id == -1:
            # Failed to load this param since it's not local to this rank
            return False if return_success else None
        # Hereafter, `expert_id` is local physical id

        quant_method_name = self.quant_method.__class__.__name__
        # compressed-tensors checkpoints with packed weights are stored flipped
        # TODO (mgoin): check self.quant_method.quant_config.quant_format
        # against known CompressionFormat enum values that have this quality
        if self.quant_method.__class__.__name__ in (
                "CompressedTensorsWNA16MarlinMoEMethod",
                "CompressedTensorsWNA16MoEMethod"):
            loaded_weight = loaded_weight.t().contiguous()

        if shard_id not in ("w1", "w2", "w3"):
            raise ValueError(f"shard_id must be ['w1','w2','w3'] but "
                             f"got {shard_id}.")

        # Fetch the dim to shard the parameter/loaded weight
        # based on the shard id. This will be whatever
        # dimension intermediate_size_per_partition is used.
        SHARD_ID_TO_SHARDED_DIM = {"w1": 0, "w2": 1, "w3": 0}

        is_gguf_weight = getattr(param, "is_gguf_weight", False)
        is_gguf_weight_type = getattr(param, "is_gguf_weight_type", False)
        if is_gguf_weight_type:
            param.weight_type = loaded_weight.item()
            param.data.copy_(loaded_weight)
            return True if return_success else None

        # Case for BitsAndBytes
        use_bitsandbytes_4bit = getattr(param, "use_bitsandbytes_4bit", False)
        if use_bitsandbytes_4bit:
            shard_dim = 0

            expert_data = param.data[expert_id]
            if shard_id == "w2":
                expert_data.copy_(loaded_weight)
            elif shard_id in ("w1", "w3"):
                # BNB inflight quantization has already sharded the weights
                full_load = True
                self._load_w13(
                    shard_id=shard_id,
                    shard_dim=shard_dim,
                    loaded_weight=loaded_weight,
                    expert_data=expert_data,
                    tp_rank=self.tp_rank,
                    load_full=full_load,
                )
            return True if return_success else None

        # is_transposed: if the dim to shard the weight
        # should be flipped. Required by GPTQ, compressed-tensors
        # should be whatever dimension intermediate_size_per_partition is
        is_transposed = getattr(param, "is_transposed", False)
        shard_dim = SHARD_ID_TO_SHARDED_DIM[shard_id]
        if is_transposed:
            shard_dim = int(not shard_dim)

        full_load = len(loaded_weight.shape) == 3
        if full_load:
            shard_dim += 1

        # Materialize GGUF UninitializedParameter
        if is_gguf_weight and isinstance(param, UninitializedParameter):
            final_shape = list(loaded_weight.shape)
            if shard_id in ["w1", "w3"]:
                final_shape[1] *= 2
            final_shape[shard_dim] = final_shape[shard_dim] // self.tp_size
            param.materialize(final_shape, dtype=loaded_weight.dtype)

        expert_data = param.data if full_load else param.data[expert_id]

        # Case input scale: input_scale loading is only supported for fp8
        if "input_scale" in weight_name:
            # this is needed for compressed-tensors only
            loaded_weight = loaded_weight.to(param.data.device)

            if ("compressed" in quant_method_name.lower()
                    and param.data[expert_id] != 1
                    and (param.data[expert_id] - loaded_weight).abs() > 1e-5):
                raise ValueError(
                    "input_scales of w1 and w3 of a layer "
                    f"must be equal. But got {param.data[expert_id]} "
                    f"vs. {loaded_weight}")

            self._load_single_value(param=param,
                                    loaded_weight=loaded_weight,
                                    expert_id=expert_id)
            return True if return_success else None

        # Case g_idx
        if "g_idx" in weight_name:
            self._load_g_idx(shard_dim=0,
                             shard_id=shard_id,
                             loaded_weight=loaded_weight,
                             expert_data=expert_data,
                             tp_rank=self.tp_rank)
            return True if return_success else None

        # TODO @dsikka: ModelOpt should follow the proper MoE loading pattern
        if "ModelOpt" in quant_method_name:
            # Determine per-tensor weight scale patterns based on variant
            # Use the dedicated method instead of brittle string matching
            uses_weight_scale_2 = self.quant_method.uses_weight_scale_2_pattern(
            )

            # Call _load_per_tensor_weight_scale() to load per-tensor (scalar)
            # weights scales.
            # Input scales are always per-tensor.
            # Weight scales: FP4 uses "weight_scale_2" and FP8 uses
            # "weight_scale" for per-tensor scales.
            is_per_tensor = ("weight_scale_2" in weight_name
                             if uses_weight_scale_2 else "weight_scale"
                             in weight_name) or "input_scale" in weight_name
            if is_per_tensor:
                self._load_per_tensor_weight_scale(
                    shard_id=shard_id,
                    param=param,
                    loaded_weight=loaded_weight,
                    expert_id=expert_id,
                )
                return True if return_success else None

            # If the weight is w13_weight_scale and w13_weight_scales are
            # combined into single loaded_weight, call
            # _load_combined_w13_weight_scale() to load it.
            # This is checked by comparing the hidden_out dims of the
            # loaded_weight and the param.
            if "w13_weight_scale" in weight_name:
                loaded_weight_hidden_out = loaded_weight.shape[-2]
                param_hidden_out = param.data.shape[-2] * self.tp_size
                if loaded_weight_hidden_out == param_hidden_out:
                    self._load_combined_w13_weight_scale(
                        shard_dim=shard_dim,
                        loaded_weight=loaded_weight,
                        param=param,
                        tp_rank=self.tp_rank,
                    )
                    return True if return_success else None

            # For other weights, call _load_model_weight_or_group_weight_scale()
            # to load it.
            if "weight" in weight_name:
                self._load_model_weight_or_group_weight_scale(
                    shard_id=shard_id,
                    shard_dim=shard_dim,
                    loaded_weight=loaded_weight,
                    expert_data=expert_data,
                    tp_rank=self.tp_rank)
            return True if return_success else None

        # Case weight scales, zero_points and offset, weight/input global scales
        if ("scale" in weight_name or "zero" in weight_name
                or "offset" in weight_name):
            # load the weight scales and zp based on the quantization scheme
            # supported weight scales/zp can be found in
            # FusedMoeWeightScaleSupported
            # TODO @dsikka: once hardened, refactor to use vLLM Parameters
            # specific to each case
            quant_method = getattr(param, "quant_method", None)
            if quant_method == FusedMoeWeightScaleSupported.CHANNEL.value:
                self._load_per_channel_weight_scale(
                    shard_id=shard_id,
                    shard_dim=shard_dim,
                    loaded_weight=loaded_weight,
                    expert_data=expert_data,
                    tp_rank=self.tp_rank)
            elif quant_method in [
                    FusedMoeWeightScaleSupported.GROUP.value,
                    FusedMoeWeightScaleSupported.BLOCK.value,
            ]:
                self._load_model_weight_or_group_weight_scale(
                    shard_id=shard_id,
                    shard_dim=shard_dim,
                    loaded_weight=loaded_weight,
                    expert_data=expert_data,
                    tp_rank=self.tp_rank,
                    load_full_w2=getattr(param, "load_full_w2", False))
            elif quant_method == FusedMoeWeightScaleSupported.TENSOR.value:
                self._load_per_tensor_weight_scale(shard_id=shard_id,
                                                   param=param,
                                                   loaded_weight=loaded_weight,
                                                   expert_id=expert_id)
            else:
                WEIGHT_SCALE_SUPPORTED = [
                    e.value for e in FusedMoeWeightScaleSupported
                ]
                raise ValueError(
                    f"quant method must be one of {WEIGHT_SCALE_SUPPORTED}")
            return True if return_success else None

        # Case weight_shape
        if "weight_shape" in weight_name:
            # only required by compressed-tensors
            self._load_single_value(param=param,
                                    loaded_weight=loaded_weight,
                                    expert_id=expert_id)
            return True if return_success else None

        # Case model weights
        if "weight" in weight_name:
            self._load_model_weight_or_group_weight_scale(
                shard_id=shard_id,
                shard_dim=shard_dim,
                loaded_weight=loaded_weight,
                expert_data=expert_data,
                tp_rank=self.tp_rank)
            return True if return_success else None

        return False if return_success else None

    def get_expert_weights(self) -> Iterable[torch.Tensor]:
        weights = list(self.named_parameters())
        assert all(weight.is_contiguous() for _, weight in weights)

        # Filter out the non-expert weights.
        # `e_score_correction_bias` is a bias for each logical expert,
        # with shape (num_logical_experts,), not an expert weight.
        NON_EXPERT_WEIGHTS = {
            "e_score_correction_bias",
        }

        return [
            weight.view(self.local_num_experts, -1) for name, weight in weights
            if name not in NON_EXPERT_WEIGHTS and weight.shape != torch.Size(
                []) and not name.startswith("_shared_experts.")
        ]

    def set_eplb_state(
        self,
        moe_layer_idx: int,
        expert_load_view: torch.Tensor,
        logical_to_physical_map: torch.Tensor,
        logical_replica_count: torch.Tensor,
    ) -> None:
        """
        Register the EPLB state in this layer.

        This is used later in forward pass, where we get the expert mapping
        and record the load metrics in `expert_load_view`.
        """
        self.expert_load_view = expert_load_view[moe_layer_idx]
        self.logical_to_physical_map = logical_to_physical_map[moe_layer_idx]
        self.logical_replica_count = logical_replica_count[moe_layer_idx]

    def ensure_moe_quant_config(self):
        if self.quant_method.moe_quant_config is None:
            self.quant_method.moe_quant_config = (
                self.quant_method.get_fused_moe_quant_config(self))

    @staticmethod
    def select_experts(
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        top_k: int,
        use_grouped_topk: bool,
        renormalize: bool,
        topk_group: Optional[int] = None,
        num_expert_group: Optional[int] = None,
        custom_routing_function: Optional[Callable] = None,
        scoring_func: str = "softmax",
        routed_scaling_factor: float = 1.0,
        e_score_correction_bias: Optional[torch.Tensor] = None,
        indices_type: Optional[torch.dtype] = None,
        enable_eplb: bool = False,
        expert_map: Optional[torch.Tensor] = None,
        expert_load_view: Optional[torch.Tensor] = None,
        logical_to_physical_map: Optional[torch.Tensor] = None,
        logical_replica_count: Optional[torch.Tensor] = None,
        global_num_experts: Optional[int] = None,
        zero_expert_num: Optional[int] = None,
        zero_expert_type: Optional[str] = None,
        layer_idx: int = -1, 
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Route the input hidden states to the top-k experts based on the
        router logits.

        Returns:
                (topk_weights, topk_ids, zero_expert_result) 
                (tuple[torch.Tensor, torch.Tensor, torch.Tensor]):
                The weights, expert ids, and zero expert computation result.

            **Compatibility**: When EPLB is not enabled, the returned ids are
            equivalent to global logical ids, so should be compatible with
            plain MoE implementations without redundant experts.
        """
        from vllm.model_executor.layers.fused_moe.fused_moe import (
            fused_topk, fused_topk_bias)

        # Check if we should use a routing simulation strategy
        routing_strategy = envs.VLLM_MOE_ROUTING_SIMULATION_STRATEGY
        if routing_strategy != "":
            topk_weights, topk_ids = RoutingSimulator.simulate_routing(
                hidden_states=hidden_states,
                router_logits=router_logits,
                strategy_name=routing_strategy,
                top_k=top_k,
                indices_type=indices_type)
        # DeepSeekv2 uses grouped_top_k
        if use_grouped_topk:
            assert topk_group is not None
            assert num_expert_group is not None
            topk_weights, topk_ids = grouped_topk(
                hidden_states=hidden_states,
                gating_output=router_logits,
                topk=top_k,
                renormalize=renormalize,
                num_expert_group=num_expert_group,
                topk_group=topk_group,
                scoring_func=scoring_func,
                routed_scaling_factor=routed_scaling_factor,
                e_score_correction_bias=e_score_correction_bias)
            if indices_type is not None:
                topk_ids = topk_ids.to(dtype=indices_type)
        elif e_score_correction_bias is not None:
            topk_weights, topk_ids = fused_topk_bias(
                hidden_states=hidden_states,
                gating_output=router_logits,
                e_score_correction_bias=e_score_correction_bias.data,
                topk=top_k,
                renormalize=renormalize,
            )
            if routed_scaling_factor is not None:
                topk_weights *= routed_scaling_factor
        elif custom_routing_function is None:
            topk_weights, topk_ids, token_expert_indices = fused_topk(
                hidden_states=hidden_states,
                gating_output=router_logits,
                topk=top_k,
                renormalize=renormalize,
                indices_type=indices_type,
            )
        else:
            topk_weights, topk_ids = custom_routing_function(
                hidden_states=hidden_states,
                gating_output=router_logits,
                topk=top_k,
                renormalize=renormalize)
            if indices_type is not None:
                topk_ids = topk_ids.to(dtype=indices_type)

        if enable_eplb:
            assert expert_load_view is not None
            assert logical_to_physical_map is not None
            assert logical_replica_count is not None

            topk_ids = eplb_map_to_physical_and_record(
                topk_ids=topk_ids,
                expert_load_view=expert_load_view,
                logical_to_physical_map=logical_to_physical_map,
                logical_replica_count=logical_replica_count,
                indices_type=indices_type,
            )

        assert topk_ids.dtype == indices_type or indices_type is None

        # Compute zero expert result if needed
        if (zero_expert_num is not None and zero_expert_num > 0
                and zero_expert_type is not None
                and global_num_experts is not None):
            zero_expert_result = zero_experts_compute_triton(
                expert_indices=topk_ids,
                expert_scales=topk_weights,
                num_experts=global_num_experts,
                zero_expert_type=zero_expert_type,
                hidden_states=hidden_states,
            )
        else:
            zero_expert_result = None
        return topk_weights, topk_ids, zero_expert_result

    def must_reduce_shared_expert_outputs(self) -> bool:
        """
        The shared_experts are typically computed using the RowParallelLinear
        layer. The result of this function is typically used as
        the reduce_results argument to the module.
        When just tensor-parallel is used, it is not required to reduce
        the shared_experts results immediately. Instead we reduce at the
        once at the end of the MoE op. (Refer to DeepSeekV2MoE module)
        With EP and all2all kernels - this is no longer viable as all
        GPU ranks in DP, produce the complete set of hidden_states.
        Therefore it is required that we reduce the shared_experts output
        early.
        """
        return (self.use_pplx_kernels or self.use_deepep_ht_kernels
                or self.use_deepep_ll_kernels)

    def maybe_all_reduce_tensor_model_parallel(
            self, final_hidden_states: torch.Tensor):
        """
        The pplx combine kernel reduces across GPU ranks by default.
        """
        if (self.use_pplx_kernels or self.use_deepep_ht_kernels
                or self.use_deepep_ll_kernels):
            return final_hidden_states
        else:
            return tensor_model_parallel_all_reduce(final_hidden_states)

    def forward_native(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        og_hidden_states = hidden_states.shape[-1]
        if self.hidden_size != og_hidden_states:
            hidden_states = F.pad(hidden_states,
                                  (0, self.hidden_size - og_hidden_states),
                                  mode='constant',
                                  value=0.0)

        if self.shared_experts is None:
            if current_platform.is_tpu():
                # TODO: Once the OOM issue for the TPU backend is resolved, we
                # will switch to using the moe_forward custom op.
                fused_output = self.forward_impl(hidden_states, router_logits)
                assert not isinstance(fused_output, tuple)
            else:
                fused_output = torch.ops.vllm.moe_forward(
                    hidden_states, router_logits, self.layer_name)
            return fused_output[..., :og_hidden_states]
        else:
            if current_platform.is_tpu():
                # TODO: Once the OOM issue for the TPU backend is resolved, we
                # will switch to using the moe_forward custom op.
                shared_output, fused_output = self.forward_impl(
                    hidden_states, router_logits)
            else:
                shared_output, fused_output = torch.ops.vllm.moe_forward_shared(
                    hidden_states, router_logits, self.layer_name)
            return (shared_output[..., :og_hidden_states],
                    fused_output[..., :og_hidden_states])

    def forward_cuda(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        return self.forward_native(hidden_states, router_logits)

    def forward_impl_chunked(
        self,
        full_hidden_states: torch.Tensor,
        full_router_logits: torch.Tensor,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        assert self.batched_hidden_states is not None
        assert self.batched_router_logits is not None
        assert self.batched_hidden_states.dtype == full_hidden_states.dtype
        assert self.batched_router_logits.dtype == full_router_logits.dtype
        # Check size compatibility.
        assert (
            self.batched_hidden_states.size(-1) == full_hidden_states.size(-1))
        assert (
            self.batched_router_logits.size(-1) == full_router_logits.size(-1))

        self.ensure_moe_quant_config()

        full_fused_final_hidden_states = torch.empty_like(full_hidden_states)
        if self.shared_experts is not None:
            full_shared_final_hidden_states = torch.empty_like(
                full_hidden_states)

        def process_chunk(chunk_start, chunk_end, skip_result_store=False):
            chunk_size = chunk_end - chunk_start
            hidden_states = full_hidden_states[chunk_start:chunk_end, :]
            router_logits = full_router_logits[chunk_start:chunk_end, :]

            assert self.batched_hidden_states is not None
            assert self.batched_router_logits is not None
            # This is only true when DBO has been enabled in the config.
            # Both tensors will have an outer dimension for the ubatch id
            if self.batched_hidden_states.dim() == 3:
                assert self.batched_router_logits.dim() == 3
                batch_buffer_idx = dbo_current_ubatch_id()
                batched_hidden_states = self.batched_hidden_states[
                    batch_buffer_idx, :]
                batched_router_logits = self.batched_router_logits[
                    batch_buffer_idx, :]
            else:
                batched_hidden_states = self.batched_hidden_states
                batched_router_logits = self.batched_router_logits

            assert (batched_hidden_states.size(0)  # type: ignore
                    >= chunk_size)
            assert (batched_router_logits.size(0)  # type: ignore 
                    >= chunk_size)
            staged_hidden_states = batched_hidden_states[:
                                                         chunk_size, :]  # type: ignore
            staged_router_logits = batched_router_logits[:
                                                         chunk_size, :]  # type: ignore
            staged_hidden_states.copy_(hidden_states, non_blocking=True)
            staged_router_logits.copy_(router_logits, non_blocking=True)

            # Matrix multiply.
            final_hidden_states = self.quant_method.apply(
                layer=self,
                x=staged_hidden_states,
                router_logits=staged_router_logits,
                top_k=self.top_k,
                renormalize=self.renormalize,
                use_grouped_topk=self.use_grouped_topk,
                global_num_experts=self.global_num_experts,
                expert_map=self.expert_map,
                topk_group=self.topk_group,
                num_expert_group=self.num_expert_group,
                custom_routing_function=self.custom_routing_function,
                scoring_func=self.scoring_func,
                routed_scaling_factor=self.routed_scaling_factor,
                e_score_correction_bias=self.e_score_correction_bias,
                activation=self.activation,
                enable_eplb=self.enable_eplb,
                expert_load_view=self.expert_load_view,
                logical_to_physical_map=self.logical_to_physical_map,
                logical_replica_count=self.logical_replica_count,
            )

            assert self.shared_experts is None or isinstance(
                final_hidden_states, tuple)

            if self.zero_expert_num is not None and self.zero_expert_num > 0:
                assert isinstance(final_hidden_states, tuple)
                assert self.shared_experts is None
                final_hidden_states, zero_expert_result = final_hidden_states
                if zero_expert_result is not None:
                    final_hidden_states += zero_expert_result

            if not skip_result_store:
                if self.shared_experts is None:
                    full_fused_final_hidden_states[
                        chunk_start:chunk_end, :].copy_(final_hidden_states,
                                                        non_blocking=True)
                else:
                    full_shared_final_hidden_states[
                        chunk_start:chunk_end, :].copy_(final_hidden_states[0],
                                                        non_blocking=True)
                    full_fused_final_hidden_states[
                        chunk_start:chunk_end, :].copy_(final_hidden_states[1],
                                                        non_blocking=True)

        ctx = get_forward_context()
        # flashinfer_cutlass_kernels can handle: optional DP + TP/EP
        max_tokens_across_dispatchers = ctx.dp_metadata.max_tokens_across_dp_cpu
        moe_dp_chunk_size_per_rank = self.moe_config.max_num_tokens

        # If the input to the MoE is sequence parallel then divide by sp_size
        # to find the maximum number of tokens for any individual dispatcher.
        if self.is_sequence_parallel:
            max_tokens_across_dispatchers = cdiv(max_tokens_across_dispatchers,
                                                 self.sp_size)

        num_tokens = full_hidden_states.size(0)
        for chunk_idx, chunk_start_ in enumerate(
                range(0, max_tokens_across_dispatchers,
                      moe_dp_chunk_size_per_rank)):
            chunk_start = chunk_start_
            chunk_end = min(chunk_start + moe_dp_chunk_size_per_rank,
                            max_tokens_across_dispatchers)
            # clamp start and end
            chunk_start = min(chunk_start, num_tokens - 1)
            chunk_end = min(chunk_end, num_tokens)
            with ctx.dp_metadata.chunked_sizes(self.sp_size,
                                               moe_dp_chunk_size_per_rank,
                                               chunk_idx):
                process_chunk(chunk_start,
                              chunk_end,
                              skip_result_store=chunk_start_ >= num_tokens)

        if self.shared_experts is None:
            return full_fused_final_hidden_states
        else:
            return (full_shared_final_hidden_states,
                    full_fused_final_hidden_states)

    def forward_impl(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        assert self.quant_method is not None
        begintime = time.time()
        self.ensure_moe_quant_config()

        # Route to the chunked forward path using the FlashInfer Cutlass kernel
        # only when data parallelism (DP) is enabled.
        _use_flashinfer_cutlass_kernels = (self.dp_size > 1 and
                                           self.use_flashinfer_cutlass_kernels)

        if (self.moe_parallel_config.use_pplx_kernels
                or self.moe_parallel_config.use_deepep_ll_kernels
                or _use_flashinfer_cutlass_kernels):
            return self.forward_impl_chunked(hidden_states, router_logits)

        do_naive_dispatch_combine: bool = (
            self.dp_size > 1
            and not self.moe_parallel_config.use_deepep_ht_kernels
            and not self.moe_config.use_flashinfer_cutlass_kernels)

        # If there are shared experts but we are not using a modular kernel, the
        # shared experts must be called here
        if (not isinstance(self.quant_method.fused_experts,
                           FusedMoEModularKernel)
                and self.shared_experts is not None):
            shared_output = self.shared_experts(hidden_states)
        else:
            shared_output = None

        ctx = get_forward_context()
        sp_ctx = ctx.dp_metadata.sp_local_sizes(
            self.sp_size) if ctx.dp_metadata else nullcontext()

        torch.cuda.synchronize()
        begin_dispatch_time = time.time()
        # print(f"Rank {ep_rank} before dispatching size is {hidden_states.size()}")
        with sp_ctx:
            if do_naive_dispatch_combine:
                hidden_states, router_logits = get_ep_group().dispatch(
                    hidden_states, router_logits, self.is_sequence_parallel)
            torch.cuda.synchronize()
            end_dispatch_time = time.time()
            # print(f"Rank {ep_rank} after dispatching size is {hidden_states.size()}")
            dispatch_combine_time = end_dispatch_time - begin_dispatch_time        
            # Matrix multiply.
            final_hidden_states = self.quant_method.apply(
                layer=self,
                x=hidden_states,
                router_logits=router_logits,
                top_k=self.top_k,
                renormalize=self.renormalize,
                use_grouped_topk=self.use_grouped_topk,
                global_num_experts=self.global_num_experts,
                expert_map=self.expert_map,
                topk_group=self.topk_group,
                num_expert_group=self.num_expert_group,
                custom_routing_function=self.custom_routing_function,
                scoring_func=self.scoring_func,
                routed_scaling_factor=self.routed_scaling_factor,
                e_score_correction_bias=self.e_score_correction_bias,
                activation=self.activation,
                apply_router_weight_on_input=self.apply_router_weight_on_input,
                enable_eplb=self.enable_eplb,
                expert_load_view=self.expert_load_view,
                logical_to_physical_map=self.logical_to_physical_map,
                logical_replica_count=self.logical_replica_count,
            )

            if shared_output is not None:
                assert not isinstance(final_hidden_states, tuple)
                assert self.shared_experts is not None
                final_hidden_states = (
                    shared_output,
                    final_hidden_states,
                )
            elif self.zero_expert_num is not None and self.zero_expert_num > 0:
                assert isinstance(final_hidden_states, tuple)
                final_hidden_states, zero_expert_result = final_hidden_states

            def reduce_output(states: torch.Tensor,
                              do_combine: bool = True) -> torch.Tensor:
                if do_naive_dispatch_combine and do_combine:
                    states = get_ep_group().combine(states,
                                                    self.is_sequence_parallel)

                if (not self.is_sequence_parallel and self.reduce_results
                        and (self.tp_size > 1 or self.ep_size > 1)):
                    states = self.maybe_all_reduce_tensor_model_parallel(
                        states)

                return states

            if self.shared_experts is not None:
                return (
                    reduce_output(final_hidden_states[0], do_combine=False),
                    reduce_output(final_hidden_states[1]),
                )
            elif self.zero_expert_num is not None and self.zero_expert_num > 0:
                assert isinstance(final_hidden_states, torch.Tensor)
                result = reduce_output(final_hidden_states) + zero_expert_result
                return result
            else:
                torch.cuda.synchronize()
                begin_combine_time = time.time()
                # print(f"Rank {ep_rank} before combine size is {final_hidden_states.size()}")
                result = reduce_output(final_hidden_states)
                torch.cuda.synchronize()
                # print(f"Rank {ep_rank} after combine size is {result.size()}")
                dispatch_combine_time += time.time() - begin_combine_time
                total_time = time.time() - begintime
                self.total_run_time += total_time
                self.total_comm_time += dispatch_combine_time
                # Debug info for dispatch/combine time
                # print(
                #     f"model_layer_name:{self.layer_name},dispatch_combine_time: {dispatch_combine_time:.4f}s, total_time: {total_time:.4f}s, ratio: {dispatch_combine_time/total_time:.4f}"
                # )
                return result

    @classmethod
    def make_expert_params_mapping(
            cls,
            ckpt_gate_proj_name: str,
            ckpt_down_proj_name: str,
            ckpt_up_proj_name: str,
            num_experts: int,
            num_redundant_experts: int = 0) -> list[tuple[str, str, int, str]]:

        num_physical_experts = num_experts + num_redundant_experts
    
        # In the returned mapping:
        # - `expert_id` is the physical expert id
        # - `weight_name` contains the weight name of the logical expert
        # So that we should map the expert id to logical in `weight_name`
        physical_to_logical_map = \
            EplbState.build_initial_global_physical_to_logical_map(
            num_experts, num_redundant_experts)
        # #修改映射逻辑,此处和expert_map二选一即可
        # print("physical_to_logical_map:", physical_to_logical_map)
        return_list = [
            # (param_name, weight_name, expert_id, shard_id)
            ("experts.w13_" if weight_name
             in [ckpt_gate_proj_name, ckpt_up_proj_name] else "experts.w2_",
             f"experts.{physical_to_logical_map[expert_id]}.{weight_name}.",
             expert_id, shard_id) for expert_id in range(num_physical_experts)
            for shard_id, weight_name in [
                ("w1", ckpt_gate_proj_name),
                ("w2", ckpt_down_proj_name),
                ("w3", ckpt_up_proj_name),
            ]
        ]
        #print("make_expert_params_mapping:", return_list)

        return return_list

    def extra_repr(self) -> str:

        s = (
            f"global_num_experts={self.global_num_experts}, "
            f"local_num_experts={self.local_num_experts}, "
            f"top_k={self.top_k}, "
            f"intermediate_size_per_partition={self.intermediate_size_per_partition}, "  # noqa: E501
            f"tp_size={self.tp_size},\n"
            f"ep_size={self.ep_size}, "
            f"reduce_results={self.reduce_results}, "
            f"renormalize={self.renormalize}, "
            f"use_grouped_topk={self.use_grouped_topk}")

        if self.use_grouped_topk:
            s += f", num_expert_group={self.num_expert_group}, topk_group={self.topk_group}"  # noqa: E501

        s += f", scoring_func='{self.scoring_func}', activation='{self.activation}'"  # noqa: E501

        return s


def moe_forward(
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    layer_name: str,
) -> torch.Tensor:
    forward_context: ForwardContext = get_forward_context()
    self = forward_context.no_compile_layers[layer_name]
    assert self.shared_experts is None
    return self.forward_impl(hidden_states, router_logits)


def moe_forward_fake(
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    layer_name: str,
) -> torch.Tensor:
    return torch.empty_like(hidden_states)


direct_register_custom_op(
    op_name="moe_forward",
    op_func=moe_forward,
    mutates_args=["hidden_states"],
    fake_impl=moe_forward_fake,
    tags=(torch.Tag.needs_fixed_stride_order, ),
)


def moe_forward_shared(
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    layer_name: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    forward_context: ForwardContext = get_forward_context()
    self = forward_context.no_compile_layers[layer_name]
    assert self.shared_experts is not None
    return self.forward_impl(hidden_states, router_logits)


def moe_forward_shared_fake(
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    layer_name: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    shared_out = torch.empty_like(hidden_states)
    fused_out = torch.empty_like(hidden_states)
    return shared_out, fused_out


direct_register_custom_op(
    op_name="moe_forward_shared",
    op_func=moe_forward_shared,
    mutates_args=["hidden_states"],
    fake_impl=moe_forward_shared_fake,
    tags=(torch.Tag.needs_fixed_stride_order, ),
)

# Mark the FusedMoE weight_loader as supporting MoE-specific parameters
# to avoid expensive runtime reflection in model loading code
FusedMoE.weight_loader.supports_moe_loading = True  # type: ignore[attr-defined]
