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
    # rank0 = [[19, 120, 9, 57, 84, 74, 82, 115], [89, 18, 43, 90, 111, 11, 10, 55], [37, 49, 59, 15, 39, 86, 83, 101], [45, 97, 46, 121, 116, 101, 83, 96], [48, 44, 11, 123, 125, 3, 28, 59], [22, 30, 21, 75, 89, 3, 26, 28], [1, 110, 53, 94, 80, 121, 32, 64], [41, 12, 32, 63, 40, 57, 0, 6], [21, 45, 38, 74, 60, 73, 75, 91], [18, 0, 106, 66, 93, 3, 20, 29], [46, 55, 116, 62, 77, 49, 17, 43], [123, 34, 80, 48, 2, 38, 74, 83], [48, 7, 100, 124, 98, 46, 70, 88], [73, 127, 2, 93, 35, 94, 40, 90], [73, 71, 59, 74, 102, 31, 41, 99], [22, 112, 5, 32, 99, 68, 93, 40], [95, 19, 64, 47, 117, 102, 56, 84], [74, 64, 55, 118, 24, 31, 47, 102], [1, 78, 53, 92, 81, 21, 22, 23], [80, 123, 59, 119, 91, 102, 117, 44], [125, 40, 127, 68, 1, 34, 60, 73], [18, 11, 124, 103, 80, 89, 68, 102], [46, 52, 21, 17, 79, 91, 106, 117], [123, 34, 57, 74, 80, 93, 106, 112], [48, 56, 50, 106, 27, 87, 6, 10], [114, 41, 49, 1, 83, 35, 71, 94], [52, 39, 91, 121, 76, 65, 77, 112], [22, 35, 41, 86, 92, 97, 114, 121], [36, 76, 102, 78, 127, 24, 31, 38], [74, 10, 59, 105, 58, 104, 110, 60], [1, 7, 29, 115, 89, 100, 59, 21], [103, 56, 10, 82, 4, 13, 23, 71], [125, 101, 127, 13, 105, 122, 7, 82], [18, 92, 122, 124, 73, 29, 36, 42], [46, 120, 116, 115, 21, 60, 79, 106], [123, 76, 124, 57, 74, 80, 83, 88], [3, 118, 54, 20, 32, 93, 72, 81], [88, 44, 117, 26, 50, 108, 122, 96], [107, 22, 27, 87, 24, 102, 45, 54], [124, 7, 117, 59, 84, 9, 74, 40], [95, 4, 11, 97, 127, 122, 84, 21], [74, 90, 121, 100, 3, 104, 107, 125], [104, 111, 88, 24, 84, 54, 103, 105], [117, 72, 91, 27, 97, 25, 33, 50], [84, 2, 56, 101, 112, 79, 85, 87], [85, 127, 80, 104, 106, 84, 37, 109], [19, 81, 51, 104, 40, 91, 108, 121], [9, 20, 125, 119, 96, 85, 1, 100]]
    # rank1 = [[108, 17, 89, 112, 41, 110, 122, 90], [102, 86, 83, 119, 51, 115, 13, 93], [93, 69, 127, 4, 74, 0, 111, 34], [81, 105, 117, 23, 13, 71, 57, 85], [101, 111, 10, 34, 93, 113, 35, 41], [44, 58, 79, 117, 115, 45, 12, 91], [106, 71, 55, 35, 28, 5, 21, 23], [80, 42, 67, 11, 71, 78, 84, 90], [125, 67, 22, 29, 82, 0, 1, 34], [28, 44, 92, 49, 91, 24, 57, 74], [112, 37, 97, 22, 10, 68, 1, 125], [127, 94, 16, 40, 110, 20, 112, 57], [3, 118, 39, 4, 1, 8, 17, 16], [88, 105, 81, 119, 14, 37, 53, 112], [127, 39, 19, 98, 106, 30, 50, 54], [124, 54, 9, 51, 3, 92, 73, 39], [36, 100, 32, 85, 61, 78, 94, 14], [122, 21, 92, 50, 43, 79, 77, 71], [16, 15, 119, 34, 87, 25, 49, 51], [83, 31, 64, 73, 29, 85, 5, 121], [21, 20, 53, 67, 76, 29, 82, 90], [28, 79, 26, 51, 113, 82, 93, 20], [24, 16, 121, 3, 122, 64, 125, 49], [127, 102, 97, 60, 113, 96, 88, 2], [3, 43, 123, 77, 49, 73, 32, 88], [111, 89, 42, 110, 20, 112, 7, 9], [127, 33, 2, 113, 24, 115, 103, 23], [124, 61, 2, 12, 119, 55, 78, 6], [95, 48, 88, 61, 75, 84, 86, 93], [38, 72, 75, 50, 96, 32, 98, 70], [60, 74, 53, 86, 67, 113, 85, 46], [80, 72, 74, 32, 40, 105, 116, 118], [114, 9, 49, 16, 10, 106, 22, 110], [88, 21, 33, 69, 80, 111, 74, 76], [39, 14, 33, 122, 76, 50, 117, 125], [127, 19, 39, 17, 96, 15, 106, 112], [48, 64, 113, 125, 106, 90, 5, 10], [114, 123, 49, 14, 22, 45, 7, 17], [51, 67, 74, 6, 26, 95, 114, 55], [57, 100, 27, 120, 19, 15, 39, 94], [36, 42, 46, 49, 14, 86, 88, 96], [119, 72, 105, 79, 75, 40, 24, 27], [91, 93, 94, 92, 97, 53, 13, 16], [118, 70, 46, 112, 113, 26, 42, 95], [1, 59, 54, 31, 104, 4, 21, 125], [89, 63, 64, 69, 1, 61, 10, 24], [31, 29, 9, 93, 120, 63, 14, 47], [92, 35, 121, 0, 19, 113, 56, 62]]
    # rank2 = [[7, 48, 28, 26, 66, 71, 24, 42], [72, 6, 80, 20, 7, 54, 49, 73], [110, 14, 77, 17, 62, 66, 35, 43], [104, 31, 61, 26, 11, 54, 24, 44], [103, 119, 58, 5, 116, 12, 97, 37], [124, 5, 51, 17, 31, 40, 11, 19], [76, 29, 120, 56, 19, 31, 36, 45], [83, 72, 76, 45, 19, 7, 20, 113], [71, 20, 87, 118, 69, 49, 55, 96], [83, 84, 39, 125, 37, 113, 42, 60], [24, 33, 74, 119, 65, 28, 30, 52], [27, 86, 107, 104, 115, 5, 12, 88], [71, 99, 26, 92, 62, 27, 14, 102], [114, 41, 24, 58, 82, 54, 87, 56], [52, 28, 100, 2, 75, 1, 122, 124], [57, 56, 70, 27, 33, 77, 86, 97], [22, 82, 35, 1, 52, 24, 21, 31], [119, 42, 18, 93, 58, 9, 23, 67], [73, 58, 19, 55, 115, 84, 69, 97], [103, 53, 61, 125, 114, 4, 113, 6], [12, 102, 92, 14, 108, 75, 62, 38], [83, 107, 96, 70, 67, 73, 36, 42], [127, 73, 86, 35, 115, 77, 28, 18], [27, 77, 121, 92, 39, 18, 12, 91], [55, 38, 118, 47, 79, 72, 81, 84], [88, 58, 118, 116, 127, 45, 90, 100], [43, 80, 62, 81, 98, 1, 102, 55], [57, 126, 24, 7, 9, 123, 91, 39], [22, 20, 28, 43, 89, 99, 57, 59], [119, 63, 61, 86, 91, 103, 71, 102], [11, 71, 24, 126, 103, 38, 10, 45], [83, 27, 39, 124, 48, 113, 69, 77], [12, 26, 99, 124, 6, 47, 117, 43], [28, 32, 49, 95, 56, 37, 68, 54], [24, 92, 78, 111, 81, 29, 1, 94], [27, 90, 77, 69, 65, 116, 93, 34], [83, 0, 33, 124, 58, 42, 88, 102], [111, 66, 11, 78, 84, 35, 40, 56], [127, 0, 62, 109, 94, 40, 120, 122], [22, 60, 48, 82, 49, 50, 62, 66], [22, 110, 48, 82, 78, 99, 38, 47], [38, 36, 88, 26, 89, 12, 70, 113], [46, 59, 58, 70, 30, 67, 62, 83], [34, 12, 55, 56, 35, 67, 9, 13], [57, 99, 13, 30, 102, 122, 70, 27], [45, 113, 19, 67, 74, 30, 13, 125], [76, 16, 79, 115, 80, 106, 69, 50], [37, 104, 52, 57, 103, 74, 42, 75]]
    # rank3 = [[109, 107, 93, 97, 23, 78, 116, 91], [23, 22, 48, 37, 92, 24, 87, 40], [32, 75, 58, 10, 121, 89, 26, 28], [113, 94, 82, 36, 84, 114, 62, 15], [109, 121, 100, 122, 2, 20, 38, 71], [7, 68, 4, 6, 37, 98, 127, 55], [95, 44, 41, 115, 93, 49, 51, 59], [58, 106, 59, 125, 96, 85, 21, 23], [48, 101, 65, 41, 53, 72, 62, 97], [108, 41, 34, 62, 111, 77, 123, 126], [75, 96, 108, 16, 78, 89, 44, 50], [37, 49, 87, 96, 68, 121, 108, 33], [83, 75, 103, 79, 45, 90, 31, 61], [38, 8, 20, 91, 64, 45, 122, 51], [51, 25, 67, 49, 6, 40, 55, 23], [71, 76, 79, 64, 69, 109, 6, 89], [112, 15, 121, 126, 114, 41, 59, 72], [38, 7, 39, 123, 108, 51, 116, 112], [11, 112, 8, 86, 103, 28, 66, 36], [16, 56, 82, 19, 3, 69, 84, 25], [48, 112, 65, 17, 70, 74, 7, 43], [101, 92, 95, 118, 111, 15, 46, 54], [11, 6, 70, 40, 81, 92, 68, 25], [35, 46, 36, 119, 76, 62, 20, 83], [83, 26, 75, 91, 21, 98, 90, 102], [63, 66, 113, 59, 124, 10, 92, 5], [46, 10, 73, 4, 78, 66, 69, 50], [71, 104, 70, 23, 51, 25, 110, 58], [17, 113, 39, 35, 10, 85, 69, 47], [56, 46, 15, 1, 94, 125, 111, 41], [30, 35, 110, 70, 61, 127, 33, 40], [60, 61, 101, 121, 102, 7, 29, 104], [48, 65, 79, 83, 115, 17, 100, 60], [83, 107, 10, 39, 22, 13, 121, 126], [127, 56, 16, 34, 82, 69, 104, 44], [37, 32, 47, 101, 84, 108, 0, 91], [121, 25, 44, 47, 22, 27, 98, 61], [60, 57, 118, 97, 83, 110, 72, 39], [43, 12, 47, 61, 116, 69, 57, 13], [118, 10, 24, 78, 83, 41, 43, 113], [17, 106, 121, 28, 35, 18, 75, 59], [127, 13, 35, 1, 108, 111, 110, 23], [12, 49, 65, 33, 96, 0, 23, 11], [114, 4, 99, 36, 90, 23, 45, 40], [33, 92, 34, 15, 29, 78, 89, 18], [21, 87, 86, 79, 33, 96, 105, 108], [126, 124, 105, 15, 33, 98, 112, 60], [63, 11, 36, 21, 65, 46, 122, 40]]
    # rank4 = [[76, 127, 111, 126, 106, 92, 6, 56], [31, 113, 107, 21, 38, 91, 2, 71], [21, 76, 57, 22, 36, 13, 5, 19], [122, 6, 4, 42, 50, 72, 74, 22], [90, 66, 40, 14, 15, 19, 81, 68], [77, 101, 111, 76, 1, 41, 94, 88], [73, 62, 4, 75, 67, 69, 102, 114], [16, 73, 70, 22, 38, 100, 69, 28], [12, 17, 66, 5, 11, 95, 7, 43], [101, 117, 79, 38, 118, 87, 97, 102], [11, 123, 70, 8, 20, 32, 72, 117], [45, 44, 72, 51, 32, 62, 18, 0], [121, 114, 50, 91, 47, 42, 34, 6], [4, 15, 25, 52, 110, 104, 116, 3], [126, 83, 123, 92, 4, 95, 3, 45], [28, 67, 90, 122, 59, 55, 123, 66], [17, 51, 79, 26, 115, 122, 9, 38], [56, 121, 13, 29, 82, 110, 114, 98], [101, 120, 122, 109, 90, 38, 47, 48], [37, 46, 18, 12, 89, 98, 36, 71], [59, 2, 124, 106, 32, 8, 58, 86], [104, 40, 6, 115, 50, 66, 35, 123], [99, 57, 109, 36, 126, 76, 95, 50], [37, 98, 6, 71, 68, 53, 73, 105], [121, 44, 122, 24, 23, 58, 105, 40], [38, 32, 121, 26, 74, 68, 104, 48], [51, 60, 22, 59, 106, 95, 36, 99], [28, 45, 56, 34, 106, 15, 82, 89], [118, 40, 37, 98, 46, 114, 96, 120], [80, 90, 45, 120, 68, 116, 33, 24], [73, 8, 55, 41, 18, 0, 121, 64], [95, 2, 110, 59, 22, 3, 73, 66], [21, 20, 87, 90, 70, 67, 74, 73], [104, 127, 50, 27, 67, 103, 97, 60], [113, 42, 93, 84, 67, 59, 118, 17], [122, 125, 72, 71, 63, 68, 43, 40], [97, 71, 103, 123, 21, 57, 4, 14], [38, 36, 27, 91, 13, 120, 48, 90], [126, 125, 28, 59, 85, 115, 38, 124], [28, 107, 38, 56, 0, 16, 1, 42], [71, 60, 107, 126, 16, 32, 64, 31], [11, 15, 50, 93, 118, 6, 43, 47], [127, 74, 117, 50, 36, 52, 72, 118], [0, 81, 31, 60, 41, 39, 85, 127], [111, 45, 11, 35, 51, 60, 69, 55], [49, 41, 8, 51, 88, 12, 56, 57], [37, 101, 117, 18, 83, 118, 36, 57], [14, 25, 47, 86, 118, 41, 97, 60]]
    # rank5 = [[52, 103, 123, 98, 22, 12, 33, 10], [65, 76, 9, 46, 8, 52, 122, 110], [102, 122, 38, 94, 82, 125, 109, 113], [2, 102, 100, 87, 25, 123, 34, 125], [24, 50, 16, 39, 8, 43, 86, 17], [29, 70, 42, 20, 104, 99, 118, 120], [101, 118, 125, 38, 116, 50, 57, 68], [37, 31, 64, 82, 10, 114, 77, 93], [107, 121, 85, 9, 16, 100, 119, 120], [90, 1, 114, 56, 67, 6, 75, 36], [39, 66, 19, 63, 94, 114, 0, 104], [118, 4, 109, 71, 29, 124, 17, 15], [97, 80, 38, 43, 101, 59, 119, 127], [60, 95, 12, 49, 126, 108, 62, 39], [119, 111, 17, 82, 5, 115, 42, 77], [30, 37, 63, 98, 21, 94, 110, 62], [101, 39, 113, 43, 107, 89, 104, 120], [11, 72, 117, 36, 96, 87, 32, 4], [118, 3, 79, 29, 104, 26, 98, 61], [81, 87, 122, 110, 109, 68, 24, 118], [71, 107, 66, 51, 16, 42, 54, 122], [108, 114, 14, 30, 56, 37, 109, 60], [101, 12, 83, 96, 102, 34, 82, 1], [45, 89, 47, 30, 55, 15, 64, 101], [104, 80, 25, 62, 101, 45, 115, 119], [4, 76, 102, 14, 50, 93, 91, 56], [126, 18, 84, 82, 85, 15, 41, 44], [4, 60, 101, 13, 69, 19, 99, 43], [101, 100, 12, 13, 1, 58, 104, 117], [11, 0, 99, 124, 100, 43, 6, 8], [42, 118, 87, 15, 109, 66, 68, 81], [16, 43, 106, 109, 38, 96, 85, 92], [88, 102, 24, 51, 29, 50, 111, 120], [101, 94, 23, 106, 17, 59, 71, 72], [11, 54, 119, 74, 20, 9, 77, 28], [35, 4, 31, 46, 107, 29, 115, 14], [55, 114, 107, 108, 101, 29, 76, 2], [4, 20, 55, 86, 85, 103, 51, 54], [52, 10, 96, 33, 8, 15, 112, 77], [105, 76, 75, 8, 102, 44, 3, 119], [2, 66, 119, 113, 89, 114, 120, 125], [5, 126, 2, 30, 99, 94, 114, 8], [43, 68, 90, 19, 6, 113, 107, 122], [15, 49, 7, 71, 61, 75, 116, 126], [16, 107, 88, 25, 3, 6, 127, 124], [90, 42, 101, 82, 97, 36, 15, 25], [89, 10, 11, 119, 0, 26, 58, 59], [73, 95, 123, 66, 61, 112, 33, 29]]
    # rank6 = [[102, 40, 43, 34, 0, 113, 3, 51], [58, 106, 56, 88, 15, 125, 114, 81], [18, 126, 65, 24, 11, 81, 73, 3], [126, 41, 92, 10, 30, 120, 73, 14], [56, 70, 0, 30, 9, 61, 51, 42], [15, 80, 103, 14, 114, 109, 0, 85], [52, 7, 8, 84, 10, 20, 0, 46], [17, 46, 2, 119, 5, 102, 105, 121], [56, 39, 26, 58, 14, 6, 37, 117], [19, 94, 105, 10, 22, 12, 25, 61], [113, 83, 54, 103, 81, 118, 25, 71], [41, 52, 75, 30, 63, 97, 76, 43], [55, 25, 111, 113, 28, 5, 105, 86], [111, 47, 113, 27, 103, 92, 29, 17], [63, 70, 16, 109, 27, 36, 87, 57], [118, 20, 13, 106, 19, 15, 1, 17], [118, 77, 54, 111, 10, 49, 18, 69], [5, 35, 62, 30, 75, 124, 107, 16], [30, 9, 110, 18, 127, 0, 32, 46], [60, 65, 79, 115, 124, 7, 21, 93], [15, 23, 99, 89, 3, 117, 94, 95], [90, 78, 81, 39, 69, 13, 74, 76], [31, 37, 119, 67, 84, 114, 98, 62], [41, 85, 75, 79, 90, 66, 17, 40], [30, 36, 71, 78, 108, 37, 31, 61], [60, 95, 80, 82, 2, 64, 101, 72], [118, 110, 14, 47, 87, 92, 123, 124], [105, 20, 72, 127, 64, 125, 77, 42], [112, 15, 5, 4, 26, 32, 72, 9], [122, 42, 28, 30, 3, 31, 47, 67], [2, 52, 111, 34, 75, 90, 5, 23], [37, 112, 47, 126, 63, 36, 68, 78], [59, 112, 2, 113, 72, 54, 97, 86], [90, 84, 98, 48, 30, 15, 109, 89], [112, 57, 88, 40, 66, 126, 32, 91], [10, 95, 102, 44, 79, 66, 33, 12], [74, 11, 110, 68, 67, 19, 46, 70], [63, 77, 113, 12, 106, 29, 43, 46], [68, 18, 60, 82, 106, 92, 103, 50], [108, 101, 72, 54, 111, 55, 89, 68], [118, 54, 123, 74, 27, 10, 61, 72], [76, 10, 64, 117, 83, 51, 31, 32], [69, 114, 75, 27, 116, 82, 85, 124], [78, 115, 121, 68, 87, 1, 76, 86], [109, 7, 19, 71, 26, 24, 83, 38], [20, 83, 27, 70, 102, 60, 9, 11], [28, 113, 7, 122, 110, 68, 95, 24], [76, 27, 67, 54, 12, 89, 124, 43]]
    # rank7 = [[88, 53, 87, 80, 118, 125, 4, 96], [97, 112, 34, 116, 60, 105, 39, 74], [45, 98, 99, 70, 80, 50, 96, 63], [112, 89, 86, 90, 7, 67, 111, 52], [31, 88, 79, 49, 32, 33, 117, 94], [69, 61, 105, 18, 97, 95, 32, 121], [16, 105, 43, 89, 72, 37, 91, 100], [103, 79, 123, 88, 3, 109, 116, 36], [46, 18, 102, 51, 32, 42, 80, 83], [47, 120, 21, 31, 27, 55, 89, 103], [99, 105, 57, 9, 95, 34, 115, 21], [35, 78, 65, 23, 120, 59, 54, 99], [117, 64, 0, 68, 78, 67, 73, 81], [107, 76, 123, 97, 13, 101, 96, 46], [43, 101, 60, 61, 91, 26, 104, 120], [4, 48, 7, 61, 26, 117, 31, 47], [90, 71, 92, 123, 6, 58, 30, 116], [80, 0, 15, 90, 78, 40, 89, 60], [106, 65, 111, 72, 4, 33, 50, 124], [127, 47, 39, 32, 62, 100, 1, 77], [46, 18, 57, 118, 25, 105, 22, 91], [19, 84, 58, 8, 2, 99, 97, 121], [87, 123, 103, 90, 7, 38, 29, 69], [1, 56, 50, 87, 84, 63, 116, 115], [74, 95, 52, 67, 76, 16, 4, 17], [73, 23, 44, 105, 78, 97, 120, 3], [119, 111, 101, 64, 34, 35, 37, 3], [65, 38, 115, 122, 8, 102, 62, 66], [90, 108, 73, 111, 83, 27, 25, 125], [48, 109, 21, 17, 81, 121, 16, 23], [122, 17, 119, 107, 6, 32, 97, 22], [111, 51, 31, 14, 45, 94, 1, 20], [39, 52, 56, 118, 78, 32, 1, 31], [108, 125, 5, 78, 34, 77, 66, 75], [99, 2, 48, 108, 22, 10, 30, 25], [1, 23, 111, 120, 55, 81, 3, 73], [117, 39, 78, 36, 51, 65, 59, 16], [34, 18, 8, 105, 61, 126, 31, 5], [63, 17, 4, 64, 91, 98, 99, 41], [116, 20, 98, 46, 125, 122, 21, 92], [101, 77, 81, 111, 115, 94, 19, 9], [56, 65, 28, 45, 120, 77, 25, 9], [80, 125, 14, 56, 10, 109, 40, 45], [122, 100, 19, 29, 48, 14, 44, 73], [17, 95, 5, 63, 8, 114, 10, 36], [94, 47, 29, 5, 4, 120, 68, 75], [84, 82, 34, 43, 42, 92, 111, 6], [127, 31, 114, 79, 69, 77, 116, 50]]
    # rank8 = [[58, 30, 45, 101, 69, 36, 83, 55], [42, 70, 32, 16, 68, 57, 96, 67], [108, 72, 53, 115, 95, 104, 46, 78], [38, 76, 66, 60, 99, 69, 115, 95], [25, 22, 102, 91, 7, 1, 75, 80], [82, 113, 122, 86, 24, 100, 46, 50], [42, 83, 112, 107, 104, 66, 82, 113], [30, 49, 27, 89, 126, 98, 48, 104], [59, 19, 28, 4, 89, 127, 68, 10], [88, 23, 14, 48, 95, 115, 4, 72], [127, 73, 109, 5, 122, 120, 80, 106], [22, 26, 50, 58, 14, 126, 53, 93], [66, 60, 107, 37, 77, 93, 58, 84], [21, 11, 80, 19, 61, 86, 120, 106], [88, 10, 86, 108, 8, 116, 69, 44], [116, 60, 100, 83, 11, 44, 81, 114], [2, 62, 60, 80, 53, 16, 23, 93], [44, 63, 52, 88, 1, 83, 91, 125], [60, 71, 13, 41, 56, 100, 121, 45], [17, 27, 43, 35, 26, 92, 105, 116], [19, 121, 56, 123, 115, 69, 45, 104], [47, 63, 21, 43, 23, 85, 61, 71], [113, 65, 66, 23, 63, 120, 41, 47], [122, 95, 22, 69, 117, 51, 21, 43], [94, 110, 69, 28, 100, 19, 42, 126], [16, 99, 125, 79, 13, 43, 103, 54], [88, 90, 74, 86, 27, 75, 104, 57], [30, 10, 54, 26, 111, 5, 96, 3], [2, 77, 119, 33, 103, 126, 74, 30], [5, 18, 14, 13, 92, 51, 29, 77], [16, 27, 14, 99, 9, 108, 26, 50], [81, 54, 18, 42, 114, 117, 24, 90], [71, 30, 28, 41, 63, 40, 91, 33], [47, 31, 6, 96, 51, 7, 91, 3], [31, 23, 37, 86, 95, 38, 8, 85], [9, 42, 50, 6, 64, 16, 5, 110], [66, 28, 38, 69, 77, 91, 41, 87], [98, 102, 80, 42, 71, 59, 68, 64], [46, 20, 2, 100, 89, 78, 66, 35], [30, 115, 23, 14, 69, 77, 96, 86], [45, 3, 5, 6, 100, 41, 58, 56], [109, 42, 61, 66, 81, 103, 22, 98], [42, 8, 102, 71, 87, 115, 35, 26], [79, 43, 63, 74, 6, 101, 51, 93], [116, 110, 105, 100, 86, 40, 76, 28], [0, 52, 17, 92, 126, 48, 40, 100], [88, 20, 100, 96, 13, 70, 65, 72], [87, 8, 72, 53, 64, 39, 49, 68]]
    # rank9 = [[121, 2, 114, 70, 29, 47, 86, 54], [17, 66, 100, 25, 109, 82, 30, 120], [112, 100, 7, 119, 2, 33, 120, 118], [39, 17, 63, 80, 98, 27, 32, 65], [57, 106, 110, 87, 114, 120, 108, 62], [54, 106, 96, 53, 93, 83, 23, 63], [30, 77, 27, 54, 18, 85, 90, 92], [60, 75, 110, 61, 62, 9, 25, 44], [52, 79, 27, 47, 98, 63, 50, 31], [104, 127, 116, 40, 63, 124, 76, 54], [6, 101, 88, 92, 67, 64, 41, 47], [10, 46, 6, 90, 36, 55, 25, 73], [104, 95, 122, 120, 76, 29, 65, 87], [98, 75, 36, 59, 84, 85, 33, 100], [46, 20, 34, 32, 94, 81, 58, 112], [18, 127, 8, 75, 2, 74, 120, 91], [42, 91, 5, 12, 13, 67, 74, 88], [76, 19, 17, 73, 84, 81, 41, 26], [2, 17, 77, 125, 107, 117, 82, 59], [111, 70, 72, 76, 86, 40, 57, 78], [52, 30, 98, 113, 49, 47, 31, 97], [88, 125, 112, 33, 62, 77, 3, 29], [15, 4, 19, 42, 9, 78, 58, 43], [10, 42, 86, 31, 29, 16, 7, 0], [35, 39, 7, 96, 20, 109, 65, 14], [98, 47, 123, 25, 126, 37, 17, 96], [107, 63, 16, 0, 72, 58, 9, 122], [103, 63, 67, 100, 33, 120, 32, 84], [124, 29, 110, 109, 87, 116, 16, 94], [127, 34, 65, 2, 78, 40, 79, 114], [106, 120, 77, 65, 104, 82, 102, 114], [50, 87, 70, 115, 108, 98, 120, 6], [81, 27, 98, 4, 11, 61, 80, 95], [19, 41, 116, 0, 38, 55, 57, 24], [87, 13, 12, 5, 114, 121, 43, 71], [45, 22, 86, 97, 126, 51, 38, 113], [95, 12, 80, 56, 34, 49, 112, 31], [28, 69, 47, 82, 65, 62, 101, 53], [119, 86, 34, 16, 58, 76, 23, 9], [53, 45, 2, 13, 35, 25, 6, 73], [90, 73, 63, 30, 1, 26, 117, 68], [62, 18, 0, 39, 68, 78, 82, 101], [120, 38, 25, 86, 106, 41, 47, 57], [77, 94, 92, 21, 24, 83, 105, 107], [43, 94, 80, 82, 50, 81, 97, 106], [95, 2, 46, 110, 114, 98, 38, 59], [114, 67, 74, 49, 107, 62, 55, 2], [32, 4, 78, 101, 83, 98, 84, 91]]
    # rank10 = [[119, 8, 73, 18, 75, 60, 68, 39], [1, 103, 64, 75, 117, 84, 4, 14], [85, 27, 116, 61, 67, 23, 44, 106], [79, 124, 18, 77, 118, 75, 53, 78], [124, 98, 27, 85, 96, 95, 69, 36], [52, 107, 48, 9, 87, 78, 92, 39], [2, 12, 86, 63, 127, 22, 97, 81], [111, 65, 56, 14, 26, 91, 117, 15], [103, 112, 99, 24, 124, 76, 111, 86], [86, 110, 98, 78, 51, 70, 85, 121], [45, 100, 126, 13, 110, 3, 82, 51], [9, 103, 42, 13, 79, 92, 106, 113], [89, 52, 69, 123, 41, 85, 115, 72], [16, 69, 118, 42, 89, 22, 83, 72], [68, 105, 53, 64, 72, 24, 9, 13], [38, 45, 115, 53, 125, 82, 84, 42], [45, 66, 108, 33, 81, 46, 99, 125], [48, 115, 65, 45, 37, 3, 6, 113], [76, 88, 7, 24, 63, 108, 89, 64], [41, 51, 75, 45, 126, 9, 66, 90], [81, 79, 26, 101, 83, 13, 110, 120], [64, 127, 120, 98, 52, 122, 25, 75], [100, 93, 14, 53, 22, 8, 116, 60], [103, 78, 23, 126, 65, 104, 124, 33], [97, 63, 53, 68, 18, 41, 93, 86], [21, 11, 75, 19, 119, 108, 31, 70], [11, 20, 89, 49, 5, 42, 38, 114], [118, 37, 88, 90, 83, 68, 50, 109], [55, 60, 107, 82, 80, 91, 68, 18], [49, 126, 7, 84, 123, 54, 107, 113], [76, 83, 78, 44, 36, 28, 37, 49], [55, 58, 75, 12, 9, 26, 84, 93], [36, 109, 126, 92, 123, 37, 34, 58], [45, 114, 58, 52, 2, 11, 12, 102], [100, 6, 7, 124, 26, 51, 68, 0], [103, 56, 36, 121, 11, 25, 59, 2], [99, 52, 96, 23, 43, 1, 119, 82], [23, 75, 33, 79, 124, 74, 87, 70], [88, 90, 105, 49, 75, 39, 104, 42], [18, 95, 70, 80, 61, 64, 112, 93], [112, 44, 51, 80, 50, 83, 105, 67], [69, 63, 21, 54, 106, 86, 102, 4], [64, 21, 78, 22, 123, 61, 48, 51], [3, 65, 82, 98, 108, 52, 104, 119], [14, 113, 37, 23, 72, 119, 115, 117], [71, 7, 103, 23, 65, 44, 3, 62], [90, 94, 39, 5, 125, 66, 25, 56], [117, 111, 105, 48, 80, 23, 106, 13]]
    # rank11 = [[72, 85, 79, 1, 77, 64, 81, 20], [59, 124, 44, 98, 35, 5, 63, 79], [47, 8, 6, 90, 97, 123, 41, 64], [49, 5, 51, 37, 109, 56, 55, 58], [55, 67, 54, 105, 65, 53, 60, 21], [47, 62, 66, 90, 38, 81, 119, 116], [99, 6, 74, 9, 87, 108, 39, 98], [54, 50, 112, 39, 122, 33, 118, 97], [36, 81, 64, 3, 113, 8, 94, 104], [81, 9, 53, 2, 11, 68, 35, 59], [23, 56, 12, 59, 98, 102, 58, 79], [98, 100, 77, 19, 47, 84, 70, 105], [110, 15, 11, 57, 20, 109, 112, 82], [67, 18, 102, 121, 30, 125, 50, 5], [118, 11, 84, 62, 29, 97, 15, 103], [29, 107, 23, 36, 0, 49, 16, 46], [124, 7, 110, 3, 28, 11, 105, 127], [69, 99, 61, 14, 68, 95, 22, 70], [62, 105, 74, 75, 94, 54, 102, 114], [34, 30, 112, 42, 38, 15, 48, 20], [114, 109, 35, 9, 6, 37, 61, 33], [45, 41, 32, 38, 10, 55, 17, 57], [39, 5, 107, 108, 26, 118, 80, 104], [9, 4, 19, 52, 72, 38, 25, 48], [117, 116, 64, 107, 1, 85, 2, 82], [115, 15, 109, 12, 61, 0, 29, 40], [68, 83, 32, 61, 108, 7, 54, 45], [87, 116, 80, 75, 59, 31, 40, 47], [45, 66, 3, 34, 11, 115, 52, 21], [97, 19, 117, 20, 55, 95, 4, 25], [88, 123, 58, 4, 124, 98, 51, 57], [52, 65, 67, 119, 62, 19, 15, 44], [84, 77, 57, 3, 14, 75, 96, 104], [64, 119, 79, 105, 115, 35, 70, 20], [101, 4, 107, 70, 97, 3, 89, 62], [41, 8, 30, 58, 54, 104, 60, 105], [89, 116, 92, 18, 24, 8, 126, 84], [67, 95, 58, 121, 19, 0, 112, 9], [118, 83, 71, 111, 30, 113, 36, 65], [103, 87, 11, 34, 33, 91, 31, 121], [124, 40, 7, 109, 53, 85, 69, 57], [48, 115, 46, 49, 58, 123, 33, 67], [100, 7, 73, 2, 119, 63, 121, 1], [58, 102, 30, 53, 123, 120, 84, 57], [22, 103, 9, 32, 62, 61, 98, 118], [81, 99, 76, 66, 31, 16, 43, 118], [38, 44, 30, 64, 99, 41, 22, 12], [90, 93, 120, 88, 59, 28, 51, 115]]
    # rank12 = [[16, 14, 117, 31, 104, 59, 38, 15], [85, 36, 127, 50, 77, 41, 121, 28], [20, 31, 56, 25, 29, 42, 60, 103], [9, 0, 1, 108, 68, 19, 3, 8], [76, 118, 47, 13, 89, 64, 18, 78], [43, 125, 72, 8, 110, 16, 84, 57], [109, 3, 123, 78, 34, 103, 61, 40], [127, 53, 43, 108, 124, 4, 13, 92], [105, 15, 108, 126, 35, 115, 44, 110], [119, 64, 52, 8, 122, 112, 13, 73], [27, 107, 76, 86, 42, 48, 7, 91], [67, 95, 24, 89, 119, 116, 64, 91], [74, 35, 33, 54, 125, 51, 126, 106], [34, 66, 44, 79, 26, 48, 68, 9], [79, 125, 110, 22, 47, 0, 37, 38], [103, 52, 24, 34, 80, 41, 113, 121], [65, 48, 4, 97, 70, 76, 50, 57], [85, 126, 10, 2, 54, 12, 8, 104], [95, 52, 44, 126, 67, 37, 80, 31], [52, 107, 55, 67, 94, 63, 97, 0], [103, 84, 126, 11, 5, 0, 100, 96], [116, 86, 53, 105, 49, 106, 22, 24], [112, 13, 33, 74, 124, 55, 89, 44], [118, 24, 100, 13, 120, 81, 70, 108], [9, 89, 11, 113, 124, 22, 51, 34], [34, 6, 36, 81, 117, 86, 122, 106], [93, 70, 71, 100, 53, 116, 31, 56], [76, 48, 53, 44, 117, 49, 94, 74], [42, 92, 62, 121, 49, 70, 67, 64], [69, 85, 64, 88, 73, 26, 89, 27], [95, 62, 20, 56, 54, 25, 48, 39], [34, 107, 122, 88, 86, 89, 0, 5], [18, 85, 66, 89, 45, 44, 76, 93], [9, 86, 120, 43, 8, 62, 85, 46], [61, 123, 53, 110, 35, 58, 41, 18], [49, 118, 98, 52, 119, 21, 62, 20], [104, 63, 13, 111, 100, 85, 105, 6], [6, 25, 115, 125, 2, 104, 1, 3], [93, 25, 80, 7, 121, 97, 3, 123], [65, 29, 67, 90, 106, 12, 81, 47], [37, 108, 12, 43, 13, 52, 70, 104], [44, 55, 97, 14, 92, 95, 41, 60], [77, 126, 20, 76, 9, 112, 95, 44], [17, 47, 125, 69, 18, 11, 20, 22], [58, 68, 108, 12, 64, 44, 126, 77], [93, 32, 112, 53, 124, 26, 55, 14], [103, 45, 71, 109, 4, 52, 75, 23], [58, 3, 30, 108, 38, 22, 126, 5]]
    # rank13 = [[11, 65, 27, 50, 46, 67, 32, 5], [27, 94, 126, 53, 69, 29, 104, 61], [87, 55, 9, 117, 92, 48, 107, 79], [91, 21, 93, 48, 127, 20, 28, 35], [112, 92, 83, 52, 74, 107, 127, 84], [13, 71, 74, 10, 2, 36, 73, 60], [60, 14, 79, 126, 24, 25, 47, 48], [51, 95, 107, 18, 35, 52, 29, 24], [23, 77, 114, 70, 116, 90, 13, 33], [45, 16, 32, 107, 80, 69, 15, 71], [15, 61, 93, 26, 35, 124, 38, 18], [122, 8, 31, 111, 114, 117, 82, 28], [63, 116, 53, 96, 56, 108, 19, 10], [63, 28, 57, 109, 65, 10, 31, 7], [89, 21, 14, 33, 121, 66, 35, 114], [108, 65, 72, 111, 25, 12, 50, 58], [40, 0, 119, 73, 83, 25, 106, 75], [57, 97, 46, 28, 66, 103, 120, 111], [83, 27, 35, 116, 91, 113, 10, 5], [50, 49, 14, 106, 10, 96, 28, 13], [77, 85, 64, 78, 63, 111, 50, 93], [9, 65, 94, 0, 48, 7, 91, 59], [45, 105, 88, 97, 59, 10, 51, 30], [61, 44, 26, 109, 94, 54, 3, 14], [66, 15, 54, 120, 57, 125, 29, 127], [27, 77, 18, 24, 84, 52, 87, 39], [21, 125, 28, 109, 19, 29, 117, 40], [18, 29, 11, 79, 27, 81, 16, 113], [65, 54, 7, 97, 6, 23, 41, 14], [44, 53, 35, 108, 36, 37, 82, 112], [43, 105, 3, 125, 93, 91, 116, 69], [8, 127, 41, 35, 100, 125, 25, 99], [15, 107, 23, 53, 5, 42, 38, 62], [110, 53, 16, 112, 26, 1, 113, 61], [15, 27, 75, 90, 55, 102, 98, 49], [67, 75, 114, 13, 92, 82, 28, 99], [94, 9, 120, 26, 50, 115, 45, 86], [16, 15, 41, 81, 10, 119, 30, 116], [21, 53, 110, 14, 72, 5, 31, 56], [52, 85, 37, 51, 127, 32, 58, 114], [8, 92, 29, 98, 91, 87, 24, 76], [57, 34, 17, 122, 124, 112, 116, 71], [55, 32, 98, 17, 89, 15, 110, 79], [10, 66, 62, 111, 106, 96, 109, 110], [67, 46, 39, 20, 66, 65, 120, 123], [111, 50, 107, 119, 91, 116, 117, 6], [77, 32, 61, 116, 3, 17, 127, 85], [107, 102, 26, 99, 71, 94, 7, 70]]
    # rank14 = [[63, 100, 124, 61, 44, 37, 105, 95], [95, 3, 123, 101, 78, 62, 12, 0], [52, 54, 16, 51, 12, 30, 114, 105], [88, 103, 59, 16, 40, 107, 119, 33], [72, 4, 26, 104, 46, 77, 6, 45], [126, 34, 59, 33, 102, 25, 27, 35], [15, 65, 88, 13, 58, 122, 33, 26], [47, 34, 115, 101, 1, 94, 68, 66], [109, 30, 2, 40, 123, 106, 61, 93], [5, 65, 58, 43, 17, 7, 50, 109], [87, 36, 14, 40, 84, 111, 29, 85], [1, 61, 102, 69, 7, 3, 39, 66], [30, 94, 44, 36, 24, 23, 32, 2], [23, 115, 32, 117, 124, 78, 43, 71], [107, 93, 18, 96, 76, 7, 113, 56], [105, 95, 88, 14, 10, 102, 78, 119], [55, 37, 29, 103, 109, 98, 87, 96], [127, 49, 109, 105, 59, 106, 25, 27], [12, 43, 14, 70, 6, 85, 40, 39], [8, 54, 2, 108, 88, 22, 104, 120], [27, 88, 28, 24, 41, 72, 44, 119], [117, 110, 5, 31, 27, 87, 12, 72], [27, 75, 56, 54, 20, 0, 111, 71], [49, 8, 114, 58, 125, 82, 110, 99], [12, 13, 114, 33, 92, 59, 112, 46], [67, 69, 8, 55, 30, 22, 62, 51], [12, 25, 105, 96, 8, 97, 30, 13], [52, 85, 95, 14, 98, 46, 1, 93], [71, 8, 51, 53, 79, 63, 122, 56], [93, 115, 52, 39, 66, 106, 22, 9], [12, 101, 19, 63, 84, 96, 80, 117], [49, 46, 123, 79, 11, 91, 21, 57], [46, 121, 35, 69, 0, 25, 8, 55], [65, 117, 40, 81, 4, 25, 99, 82], [45, 65, 73, 36, 83, 96, 64, 52], [61, 78, 89, 109, 94, 117, 18, 48], [15, 30, 122, 62, 79, 109, 73, 40], [107, 76, 109, 89, 73, 93, 92, 94], [70, 11, 101, 108, 32, 81, 37, 29], [88, 71, 36, 26, 126, 109, 5, 97], [55, 34, 15, 79, 39, 103, 93, 25], [7, 53, 52, 20, 84, 73, 29, 87], [99, 37, 101, 18, 5, 4, 29, 34], [28, 88, 2, 16, 5, 37, 89, 103], [121, 73, 48, 47, 42, 52, 74, 75], [18, 123, 58, 77, 34, 22, 39, 122], [27, 8, 97, 21, 53, 48, 78, 87], [15, 34, 82, 2, 18, 45, 10, 81]]
    # rank15 = [[21, 94, 62, 49, 99, 13, 25, 35], [26, 108, 33, 47, 99, 45, 19, 118], [84, 88, 68, 40, 1, 71, 91, 124], [29, 64, 106, 70, 47, 12, 43, 110], [23, 63, 29, 115, 99, 82, 73, 126], [112, 49, 108, 64, 56, 67, 123, 65], [119, 11, 17, 96, 111, 70, 117, 124], [81, 8, 87, 74, 55, 86, 120, 99], [88, 84, 78, 57, 25, 92, 54, 122], [99, 100, 96, 33, 30, 26, 82, 46], [2, 31, 4, 90, 53, 121, 69, 60], [11, 85, 125, 56, 81, 60, 21, 101], [12, 9, 13, 21, 22, 18, 49, 40], [77, 6, 99, 55, 74, 0, 1, 70], [48, 12, 80, 90, 85, 78, 117, 65], [85, 87, 101, 104, 126, 96, 35, 43], [44, 8, 34, 20, 68, 27, 63, 86], [34, 53, 86, 20, 100, 94, 33, 101], [42, 123, 99, 20, 93, 96, 57, 68], [95, 58, 74, 101, 11, 33, 99, 23], [39, 36, 87, 4, 116, 10, 55, 80], [100, 119, 16, 44, 34, 1, 4, 126], [61, 2, 48, 32, 72, 110, 94, 85], [67, 11, 111, 107, 32, 59, 5, 28], [60, 99, 0, 103, 111, 8, 70, 5], [28, 107, 57, 33, 65, 53, 85, 46], [79, 48, 17, 67, 94, 26, 6, 120], [108, 107, 36, 21, 112, 0, 73, 17], [0, 44, 81, 123, 106, 50, 105, 19], [57, 76, 118, 62, 12, 83, 101, 87], [13, 112, 79, 72, 94, 92, 31, 47], [17, 30, 64, 53, 76, 33, 97, 28], [103, 19, 64, 116, 68, 94, 108, 119], [63, 100, 14, 44, 118, 87, 123, 93], [105, 19, 109, 63, 103, 72, 80, 47], [24, 100, 26, 87, 85, 7, 70, 53], [60, 35, 53, 7, 75, 37, 17, 127], [21, 32, 24, 99, 127, 52, 37, 100], [79, 48, 19, 84, 73, 1, 117, 44], [104, 4, 63, 79, 17, 110, 99, 123], [65, 0, 20, 62, 33, 23, 116, 102], [19, 85, 59, 80, 37, 96, 91, 16], [31, 108, 39, 28, 66, 81, 60, 3], [8, 59, 80, 38, 32, 124, 54, 64], [53, 0, 90, 41, 96, 49, 91, 93], [115, 78, 73, 121, 28, 72, 54, 35], [123, 46, 1, 102, 35, 54, 73, 86], [109, 44, 6, 55, 110, 17, 24, 16]]
    '''
    rank0 = [[19, 17, 31, 79, 3, 39, 5, 55], [102, 56, 107, 117, 125, 109, 74, 81], [37, 94, 115, 30, 113, 63, 5, 106], [45, 42, 10, 84, 75, 114, 52, 83], [48, 74, 107, 61, 108, 125, 71, 17], [22, 97, 56, 75, 55, 46, 50, 60], [106, 127, 53, 72, 121, 81, 113, 5], [80, 106, 88, 9, 7, 0, 71, 78], [71, 116, 13, 0, 34, 120, 75, 96], [18, 3, 42, 72, 20, 29, 123, 126], [46, 26, 124, 115, 106, 1, 49, 17], [123, 57, 80, 74, 34, 38, 2, 83], [48, 32, 61, 82, 86, 81, 10, 31], [73, 105, 103, 0, 9, 70, 96, 116], [127, 41, 112, 114, 55, 122, 103, 77], [124, 39, 73, 43, 66, 114, 97, 62], [95, 57, 93, 120, 104, 21, 31, 86], [74, 61, 30, 82, 81, 33, 111, 32], [2, 63, 97, 117, 21, 69, 114, 45], [37, 91, 45, 21, 7, 28, 23, 99], [71, 92, 16, 80, 25, 93, 58, 73], [18, 121, 36, 3, 29, 42, 72, 126], [46, 126, 84, 34, 94, 1, 125, 49], [123, 74, 93, 112, 57, 80, 88, 106], [48, 81, 72, 31, 10, 88, 5, 86], [69, 19, 119, 35, 87, 72, 71, 39], [127, 122, 124, 55, 45, 65, 120, 123], [22, 33, 84, 74, 39, 66, 58, 47], [95, 10, 50, 14, 76, 84, 96, 61], [56, 88, 59, 68, 121, 24, 98, 60], [2, 94, 6, 91, 85, 68, 69, 31], [80, 47, 22, 89, 66, 63, 90, 105], [12, 24, 53, 16, 23, 108, 75, 33], [18, 4, 124, 76, 60, 73, 93, 121], [46, 42, 86, 67, 59, 8, 52, 17], [123, 59, 5, 83, 88, 80, 106, 112], [48, 113, 62, 67, 27, 42, 16, 31], [88, 82, 12, 85, 0, 100, 39, 96], [127, 62, 89, 8, 44, 35, 41, 122], [124, 14, 51, 33, 96, 43, 42, 62], [95, 62, 39, 93, 47, 88, 72, 21], [74, 2, 90, 108, 12, 111, 71, 23], [104, 2, 86, 123, 67, 118, 83, 45], [118, 99, 6, 26, 67, 9, 50, 25], [43, 15, 96, 52, 79, 97, 123, 55], [21, 69, 40, 125, 105, 108, 56, 118], [28, 9, 109, 92, 80, 127, 72, 73], [92, 55, 21, 96, 19, 70, 75, 81]]
    rank1 = [[108, 103, 80, 104, 112, 12, 90, 56], [72, 70, 22, 20, 119, 29, 12, 79], [32, 61, 70, 123, 89, 73, 83, 105], [49, 6, 48, 37, 20, 54, 32, 78], [103, 83, 100, 64, 6, 73, 69, 37], [44, 111, 87, 81, 53, 116, 35, 28], [73, 8, 122, 24, 85, 66, 50, 69], [37, 126, 59, 10, 97, 99, 15, 6], [12, 51, 124, 22, 41, 33, 104, 97], [108, 116, 27, 39, 51, 71, 103, 15], [11, 57, 96, 8, 25, 58, 30, 60], [127, 50, 59, 17, 126, 48, 112, 88], [83, 123, 38, 76, 67, 19, 17, 84], [88, 19, 81, 10, 31, 122, 72, 90], [52, 76, 49, 91, 42, 44, 50, 124], [22, 46, 58, 113, 1, 47, 99, 68], [36, 121, 6, 37, 49, 75, 38, 96], [56, 45, 68, 58, 86, 107, 24, 31], [73, 8, 37, 10, 40, 61, 57, 31], [80, 126, 76, 12, 36, 24, 121, 6], [12, 65, 116, 105, 74, 43, 90, 60], [101, 70, 124, 66, 59, 74, 123, 73], [11, 23, 81, 67, 116, 71, 79, 106], [127, 102, 64, 14, 101, 105, 2, 91], [13, 118, 124, 41, 58, 34, 17, 87], [114, 80, 26, 50, 103, 120, 3, 90], [52, 27, 5, 115, 102, 103, 112, 50], [124, 59, 50, 86, 42, 1, 91, 92], [36, 4, 26, 99, 80, 21, 31, 38], [74, 45, 13, 26, 51, 104, 102, 47], [73, 27, 20, 34, 4, 48, 40, 59], [37, 72, 41, 11, 33, 98, 99, 6], [59, 26, 92, 67, 70, 117, 73, 96], [88, 44, 34, 80, 113, 36, 20, 3], [11, 37, 108, 51, 102, 80, 47, 49], [127, 52, 39, 62, 82, 99, 91, 57], [83, 120, 69, 101, 4, 61, 72, 82], [114, 102, 89, 65, 103, 51, 56, 90], [43, 71, 82, 27, 24, 37, 117, 55], [22, 70, 78, 127, 41, 15, 1, 97], [2, 41, 46, 13, 23, 80, 76, 31], [38, 61, 122, 37, 75, 3, 107, 125], [91, 98, 35, 24, 82, 62, 124, 29], [117, 92, 80, 111, 101, 84, 93, 86], [109, 13, 25, 49, 101, 70, 4, 117], [49, 119, 28, 10, 44, 25, 57, 59], [126, 104, 4, 117, 118, 60, 59, 65], [9, 99, 86, 112, 113, 100, 13, 1]]
    rank2 = [[76, 53, 45, 97, 92, 124, 105, 10], [97, 37, 83, 47, 84, 46, 87, 14], [102, 90, 97, 95, 39, 109, 19, 26], [81, 89, 108, 19, 30, 71, 3, 95], [24, 22, 102, 52, 19, 122, 35, 41], [29, 8, 76, 109, 1, 63, 3, 26], [2, 13, 35, 74, 70, 22, 117, 47], [65, 55, 125, 64, 73, 24, 23, 57], [52, 35, 89, 106, 54, 31, 93, 119], [88, 105, 34, 125, 106, 24, 46, 74], [24, 33, 122, 16, 102, 82, 44, 125], [27, 75, 3, 14, 82, 43, 0, 20], [3, 53, 111, 95, 85, 119, 70, 127], [69, 20, 125, 78, 86, 112, 46, 17], [73, 86, 109, 1, 95, 102, 65, 45], [28, 48, 79, 25, 15, 31, 77, 86], [112, 15, 79, 106, 16, 74, 23, 125], [38, 42, 20, 37, 54, 110, 95, 113], [11, 35, 55, 90, 33, 26, 0, 68], [65, 56, 61, 97, 119, 13, 93, 120], [52, 66, 53, 113, 22, 91, 1, 34], [88, 86, 34, 87, 89, 54, 93, 61], [24, 36, 96, 121, 95, 98, 43, 44], [27, 13, 52, 97, 110, 124, 96, 83], [3, 64, 107, 50, 44, 32, 14, 127], [88, 55, 33, 64, 85, 56, 54, 5], [43, 110, 19, 35, 58, 108, 44, 3], [108, 104, 13, 44, 111, 109, 99, 43], [2, 81, 106, 34, 126, 30, 9, 59], [38, 50, 2, 30, 58, 41, 27, 113], [60, 53, 44, 104, 72, 22, 108, 21], [103, 127, 79, 48, 115, 116, 20, 69], [71, 9, 118, 45, 13, 68, 25, 7], [101, 12, 37, 13, 24, 97, 29, 42], [24, 55, 110, 95, 118, 98, 0, 125], [27, 46, 31, 55, 126, 43, 48, 74], [3, 53, 103, 21, 32, 57, 5, 86], [69, 55, 11, 106, 35, 62, 53, 9], [126, 34, 106, 36, 116, 115, 65, 120], [108, 10, 106, 83, 64, 50, 81, 31], [36, 30, 121, 103, 10, 94, 9, 19], [119, 21, 1, 68, 51, 6, 8, 24], [42, 8, 14, 84, 30, 51, 107, 122], [15, 7, 5, 52, 37, 23, 119, 13], [84, 110, 71, 40, 78, 127, 85, 87], [90, 121, 81, 26, 104, 38, 84, 14], [31, 124, 5, 99, 98, 40, 86, 91], [37, 53, 88, 118, 38, 28, 115, 40]]
    rank3 = [[109, 11, 40, 1, 87, 113, 64, 20], [89, 6, 100, 88, 98, 28, 5, 118], [110, 71, 24, 121, 4, 103, 28, 35], [122, 51, 4, 77, 118, 121, 119, 57], [101, 26, 91, 39, 38, 11, 126, 94], [124, 68, 62, 17, 104, 121, 89, 57], [95, 88, 86, 10, 116, 25, 68, 102], [16, 87, 17, 32, 98, 13, 36, 84], [125, 102, 26, 42, 118, 80, 68, 1], [119, 84, 48, 38, 4, 124, 36, 109], [112, 88, 32, 111, 34, 121, 117, 79], [118, 52, 47, 41, 63, 62, 18, 33], [13, 80, 120, 92, 62, 93, 88, 6], [114, 115, 12, 26, 87, 106, 94, 3], [126, 34, 96, 121, 5, 97, 13, 99], [108, 80, 13, 0, 59, 35, 40, 119], [118, 91, 103, 28, 105, 122, 9, 47], [119, 52, 100, 59, 108, 40, 23, 101], [106, 58, 125, 34, 91, 121, 47, 49], [111, 87, 68, 124, 62, 40, 90, 105], [59, 47, 14, 23, 122, 31, 96, 120], [90, 16, 44, 95, 69, 106, 80, 76], [127, 66, 65, 7, 82, 76, 60, 117], [44, 23, 90, 126, 63, 43, 99, 34], [55, 38, 71, 47, 101, 65, 126, 82], [63, 125, 102, 13, 116, 104, 70, 100], [46, 60, 71, 2, 59, 92, 40, 41], [105, 95, 34, 2, 98, 5, 110, 123], [3, 77, 51, 70, 37, 58, 18, 102], [119, 63, 39, 77, 37, 9, 107, 8], [11, 58, 107, 125, 116, 26, 46, 45], [16, 83, 96, 108, 4, 13, 44, 71], [52, 87, 89, 29, 74, 100, 104, 119], [90, 32, 98, 2, 39, 111, 74, 61], [39, 54, 103, 70, 3, 69, 91, 117], [35, 111, 47, 50, 66, 65, 3, 15], [13, 28, 122, 20, 47, 22, 112, 87], [60, 24, 66, 78, 14, 45, 68, 46], [107, 18, 111, 61, 113, 3, 123, 42], [105, 107, 90, 24, 35, 89, 40, 114], [3, 77, 109, 114, 50, 18, 120, 125], [76, 0, 26, 58, 82, 94, 67, 101], [46, 73, 22, 61, 89, 16, 79, 44], [17, 12, 34, 98, 96, 45, 113, 44], [16, 39, 90, 42, 61, 38, 93, 106], [85, 65, 82, 80, 3, 117, 1, 61], [37, 10, 122, 102, 48, 75, 23, 47], [14, 27, 52, 67, 0, 83, 41, 5]]
    rank4 = [[58, 50, 22, 106, 59, 125, 68, 115], [23, 112, 52, 53, 35, 104, 11, 120], [21, 75, 119, 67, 111, 33, 114, 43], [2, 124, 106, 26, 16, 123, 101, 65], [90, 63, 40, 46, 77, 33, 127, 28], [69, 66, 33, 2, 16, 32, 88, 127], [1, 123, 126, 27, 7, 111, 49, 59], [111, 79, 56, 115, 82, 14, 85, 93], [59, 99, 126, 123, 23, 72, 43, 110], [90, 96, 2, 33, 118, 66, 113, 60], [75, 73, 74, 55, 64, 68, 18, 10], [44, 26, 6, 120, 68, 40, 101, 28], [55, 110, 68, 39, 29, 109, 34, 2], [38, 102, 42, 59, 74, 83, 53, 71], [43, 25, 100, 69, 66, 29, 108, 54], [57, 100, 34, 126, 117, 49, 5, 110], [3, 48, 26, 10, 99, 52, 18, 59], [122, 115, 13, 78, 75, 94, 104, 8], [60, 79, 6, 111, 116, 66, 48, 51], [103, 43, 32, 118, 33, 73, 71, 25], [125, 87, 124, 3, 94, 54, 75, 110], [119, 105, 107, 23, 1, 4, 71, 60], [100, 14, 72, 53, 111, 29, 30, 17], [35, 26, 36, 125, 59, 21, 3, 20], [83, 80, 20, 92, 77, 93, 46, 90], [38, 34, 89, 81, 97, 45, 48, 46], [126, 10, 61, 82, 81, 116, 23, 114], [28, 20, 88, 26, 12, 25, 96, 113], [118, 65, 123, 83, 74, 105, 127, 117], [76, 72, 21, 20, 22, 4, 101, 32], [106, 35, 126, 101, 127, 32, 28, 49], [60, 70, 35, 118, 53, 40, 117, 0], [114, 102, 124, 90, 78, 122, 95, 93], [47, 21, 40, 92, 51, 35, 5, 72], [127, 14, 78, 83, 38, 58, 79, 106], [37, 75, 58, 120, 107, 60, 108, 0], [55, 11, 110, 23, 75, 19, 49, 40], [38, 95, 42, 91, 110, 126, 108, 94], [52, 19, 110, 64, 92, 103, 66, 13], [67, 80, 23, 126, 0, 84, 74, 68], [71, 40, 6, 74, 115, 24, 25, 122], [56, 72, 126, 121, 99, 33, 29, 114], [127, 38, 93, 119, 109, 97, 13, 26], [0, 30, 72, 69, 63, 120, 104, 42], [57, 45, 34, 22, 92, 81, 74, 75], [78, 58, 91, 4, 16, 13, 15, 9], [76, 16, 74, 68, 62, 119, 50, 111], [76, 72, 35, 26, 71, 84, 45, 91]]
    rank5 = [[121, 120, 111, 126, 114, 84, 4, 82], [58, 101, 123, 60, 116, 105, 2, 10], [93, 14, 38, 25, 42, 81, 86, 34], [88, 36, 41, 7, 12, 107, 24, 8], [72, 50, 111, 104, 5, 9, 113, 3], [15, 122, 113, 20, 5, 99, 119, 11], [76, 44, 125, 75, 62, 20, 46, 92], [46, 31, 123, 101, 11, 12, 116, 20], [112, 121, 28, 58, 100, 37, 62, 73], [101, 21, 44, 49, 85, 55, 59, 75], [39, 13, 92, 119, 98, 3, 116, 47], [37, 4, 23, 107, 116, 21, 91, 93], [97, 99, 118, 103, 41, 106, 58, 16], [60, 11, 113, 97, 1, 108, 62, 39], [119, 89, 32, 64, 78, 26, 9, 120], [105, 37, 76, 125, 2, 120, 55, 93], [2, 62, 100, 1, 85, 107, 56, 69], [76, 99, 90, 84, 3, 87, 114, 71], [1, 99, 27, 62, 104, 84, 100, 82], [16, 83, 18, 88, 100, 92, 84, 102], [112, 102, 2, 83, 118, 115, 117, 62], [47, 94, 62, 30, 38, 122, 102, 20], [101, 119, 86, 19, 124, 26, 62, 91], [118, 95, 72, 30, 68, 82, 73, 0], [104, 114, 123, 67, 24, 85, 70, 42], [111, 11, 58, 49, 74, 86, 62, 94], [118, 28, 16, 96, 69, 85, 54, 77], [65, 10, 116, 69, 8, 120, 55, 93], [66, 113, 39, 6, 91, 116, 64, 68], [127, 118, 65, 55, 120, 89, 29, 87], [30, 16, 110, 103, 63, 10, 113, 47], [65, 27, 56, 110, 62, 12, 85, 15], [112, 98, 101, 47, 106, 113, 105, 34], [104, 114, 17, 52, 115, 25, 56, 126], [100, 107, 109, 35, 32, 116, 77, 43], [122, 98, 89, 109, 64, 81, 7, 34], [97, 52, 92, 43, 24, 1, 127, 46], [111, 20, 115, 22, 124, 37, 3, 54], [63, 11, 28, 100, 53, 38, 102, 112], [28, 87, 98, 102, 61, 112, 93, 47], [118, 5, 43, 33, 35, 97, 64, 59], [127, 65, 20, 81, 79, 41, 25, 43], [69, 59, 68, 58, 36, 81, 54, 1], [21, 49, 31, 71, 83, 124, 14, 95], [111, 107, 19, 20, 32, 114, 65, 91], [95, 47, 66, 124, 22, 33, 75, 96], [89, 19, 100, 51, 107, 120, 58, 25], [87, 15, 47, 48, 121, 94, 116, 43]]
    rank6 = [[102, 49, 62, 41, 98, 71, 33, 86], [31, 103, 113, 57, 43, 68, 54, 71], [18, 9, 2, 127, 92, 62, 23, 0], [126, 102, 80, 5, 13, 120, 73, 14], [57, 88, 96, 58, 95, 86, 60, 36], [77, 80, 42, 51, 86, 27, 65, 120], [29, 77, 94, 18, 67, 32, 0, 64], [58, 74, 39, 122, 81, 40, 38, 104], [107, 108, 101, 5, 92, 16, 117, 29], [19, 17, 86, 40, 6, 91, 102, 73], [127, 45, 108, 9, 70, 114, 51, 21], [35, 56, 125, 78, 96, 115, 39, 99], [71, 52, 107, 4, 56, 44, 27, 72], [107, 32, 79, 49, 124, 101, 68, 56], [46, 105, 16, 61, 98, 74, 81, 38], [67, 72, 8, 83, 122, 102, 74, 81], [66, 108, 33, 111, 53, 76, 116, 61], [127, 63, 14, 121, 118, 125, 4, 98], [30, 119, 41, 78, 127, 20, 38, 64], [46, 2, 108, 22, 109, 94, 48, 117], [15, 35, 126, 24, 17, 37, 100, 97], [19, 21, 98, 96, 0, 51, 67, 35], [93, 5, 33, 48, 59, 38, 41, 68], [37, 42, 94, 31, 65, 16, 48, 12], [94, 110, 120, 103, 56, 57, 22, 98], [60, 23, 121, 65, 59, 83, 40, 7], [119, 90, 14, 47, 34, 24, 97, 99], [67, 37, 100, 14, 102, 15, 32, 89], [112, 108, 110, 97, 27, 57, 69, 120], [80, 109, 15, 66, 3, 106, 110, 6], [42, 52, 78, 18, 39, 98, 117, 114], [111, 74, 31, 39, 45, 21, 1, 78], [39, 19, 57, 126, 116, 44, 22, 120], [119, 94, 6, 27, 122, 1, 91, 59], [113, 57, 73, 82, 122, 115, 68, 50], [10, 42, 121, 79, 104, 11, 38, 33], [89, 78, 80, 68, 100, 65, 93, 6], [63, 75, 79, 93, 73, 120, 48, 116], [119, 125, 101, 74, 30, 98, 85, 56], [118, 101, 37, 120, 32, 109, 94, 92], [66, 60, 123, 100, 16, 89, 86, 96], [109, 39, 115, 45, 92, 120, 27, 31], [120, 55, 92, 18, 96, 63, 103, 11], [102, 43, 2, 59, 90, 32, 109, 110], [33, 63, 2, 3, 66, 69, 10, 36], [0, 46, 77, 31, 72, 43, 109, 11], [90, 81, 101, 21, 17, 110, 108, 121], [73, 25, 11, 61, 39, 65, 89, 56]]
    rank7 = [[88, 57, 73, 61, 34, 35, 116, 36], [59, 18, 44, 63, 92, 24, 39, 93], [20, 122, 16, 57, 77, 82, 50, 91], [112, 113, 105, 82, 55, 56, 62, 58], [109, 106, 119, 27, 14, 93, 75, 42], [101, 14, 74, 114, 9, 115, 94, 85], [52, 109, 3, 89, 58, 19, 33, 98], [42, 70, 1, 91, 26, 19, 90, 102], [48, 18, 2, 76, 11, 61, 111, 63], [47, 1, 98, 107, 5, 95, 87, 93], [113, 105, 19, 94, 72, 67, 38, 104], [45, 49, 58, 79, 36, 25, 66, 110], [66, 25, 33, 37, 77, 22, 73, 14], [111, 34, 24, 89, 82, 14, 29, 51], [70, 125, 84, 71, 27, 115, 75, 37], [18, 127, 88, 14, 61, 33, 96, 123], [17, 119, 20, 34, 35, 87, 24, 72], [11, 10, 39, 92, 26, 91, 25, 47], [118, 65, 44, 101, 39, 70, 4, 124], [60, 27, 123, 81, 19, 11, 29, 0], [107, 18, 21, 101, 56, 127, 61, 7], [104, 63, 6, 58, 52, 91, 55, 57], [31, 88, 99, 54, 35, 120, 25, 10], [45, 86, 79, 71, 92, 40, 81, 33], [74, 39, 25, 113, 23, 108, 16, 112], [73, 75, 25, 42, 1, 93, 0, 29], [21, 0, 73, 89, 22, 26, 72, 56], [57, 60, 63, 83, 122, 31, 68, 97], [17, 100, 121, 109, 11, 46, 94, 47], [122, 115, 14, 84, 54, 91, 31, 67], [1, 118, 115, 67, 93, 33, 124, 51], [95, 87, 86, 32, 76, 114, 24, 104], [125, 88, 40, 4, 69, 72, 38, 110], [19, 16, 43, 58, 30, 99, 78, 123], [31, 19, 36, 40, 74, 9, 89, 104], [44, 95, 36, 94, 68, 16, 124, 105], [74, 95, 64, 54, 79, 41, 59, 70], [67, 77, 44, 125, 10, 112, 74, 70], [46, 20, 14, 1, 69, 26, 40, 57], [53, 115, 75, 46, 59, 49, 21, 119], [17, 65, 42, 113, 28, 27, 99, 38], [11, 17, 36, 49, 106, 78, 98, 32], [43, 114, 75, 28, 6, 41, 60, 85], [122, 62, 125, 29, 87, 11, 54, 22], [17, 94, 88, 23, 12, 104, 126, 28], [94, 107, 23, 127, 97, 98, 12, 100], [88, 20, 79, 49, 53, 54, 12, 85], [8, 31, 104, 119, 66, 77, 51, 16]]
    rank8 = [[52, 27, 37, 77, 0, 67, 25, 122], [1, 33, 80, 111, 41, 90, 19, 0], [45, 8, 52, 51, 11, 80, 36, 101], [39, 86, 31, 99, 127, 11, 115, 85], [31, 49, 105, 30, 65, 32, 97, 18], [82, 108, 64, 90, 95, 24, 45, 39], [30, 105, 79, 90, 84, 97, 91, 48], [103, 27, 112, 86, 66, 3, 29, 113], [103, 17, 64, 24, 55, 50, 115, 25], [99, 94, 14, 30, 22, 112, 13, 35], [100, 56, 37, 90, 5, 7, 69, 43], [98, 103, 85, 13, 114, 16, 60, 108], [89, 75, 36, 51, 28, 1, 5, 90], [21, 18, 57, 27, 109, 64, 33, 45], [88, 10, 39, 123, 85, 87, 30, 117], [118, 60, 75, 23, 111, 19, 44, 91], [90, 101, 54, 83, 11, 126, 30, 117], [80, 72, 35, 73, 1, 96, 112, 16], [95, 71, 110, 77, 72, 107, 81, 36], [42, 51, 75, 35, 125, 9, 69, 85], [48, 98, 99, 89, 78, 10, 86, 63], [120, 116, 40, 33, 8, 11, 13, 103], [113, 107, 37, 55, 110, 115, 114, 85], [9, 22, 85, 121, 75, 39, 62, 5], [35, 116, 52, 28, 91, 37, 115, 2], [16, 8, 36, 113, 78, 37, 31, 9], [88, 125, 101, 84, 113, 78, 15, 30], [103, 80, 54, 23, 46, 7, 17, 40], [55, 60, 73, 107, 98, 52, 23, 75], [11, 0, 126, 73, 123, 81, 116, 23], [95, 8, 79, 19, 111, 89, 100, 66], [50, 51, 81, 17, 59, 38, 36, 92], [81, 27, 21, 49, 10, 115, 0, 82], [45, 79, 14, 81, 106, 95, 89, 82], [112, 45, 99, 5, 20, 121, 28, 94], [103, 4, 114, 71, 77, 84, 18, 93], [66, 9, 38, 71, 106, 119, 45, 88], [98, 8, 58, 27, 105, 119, 101, 17], [88, 90, 17, 16, 121, 81, 97, 77], [57, 20, 72, 117, 17, 122, 5, 6], [92, 44, 15, 81, 37, 87, 116, 117], [5, 63, 13, 88, 112, 123, 40, 4], [99, 74, 111, 56, 9, 23, 0, 34], [79, 65, 114, 74, 24, 75, 103, 105], [95, 99, 47, 64, 56, 98, 51, 115], [89, 41, 34, 113, 70, 114, 24, 106], [27, 45, 30, 18, 35, 61, 41, 56], [58, 63, 82, 2, 12, 80, 74, 68]]
    rank9 = [[7, 2, 100, 30, 69, 74, 32, 54], [27, 86, 106, 66, 99, 62, 114, 121], [55, 65, 69, 117, 17, 22, 125, 118], [103, 76, 60, 117, 25, 69, 28, 110], [25, 118, 0, 2, 44, 20, 123, 84], [43, 61, 59, 10, 30, 78, 84, 92], [42, 17, 43, 55, 39, 80, 100, 57], [47, 49, 108, 5, 4, 45, 105, 121], [46, 39, 27, 78, 113, 6, 122, 38], [120, 127, 53, 10, 63, 70, 115, 54], [27, 101, 126, 59, 42, 120, 48, 77], [10, 46, 102, 65, 89, 97, 55, 113], [35, 30, 64, 122, 101, 57, 105, 98], [67, 66, 121, 117, 58, 30, 37, 5], [63, 33, 67, 7, 36, 59, 24, 3], [4, 87, 54, 56, 64, 41, 82, 109], [45, 7, 77, 123, 13, 32, 14, 64], [48, 46, 126, 65, 36, 103, 79, 102], [76, 15, 13, 53, 103, 56, 80, 59], [50, 30, 31, 122, 53, 14, 38, 78], [81, 19, 40, 4, 123, 76, 0, 68], [108, 125, 27, 53, 50, 56, 77, 24], [39, 103, 32, 109, 97, 118, 80, 21], [122, 4, 89, 29, 120, 84, 38, 28], [66, 9, 69, 76, 62, 27, 49, 102], [27, 47, 57, 79, 12, 84, 53, 92], [11, 111, 67, 74, 109, 1, 75, 57], [4, 76, 126, 70, 125, 19, 49, 114], [71, 101, 119, 111, 114, 1, 78, 86], [48, 7, 10, 75, 100, 40, 33, 79], [122, 15, 96, 3, 86, 36, 38, 102], [55, 112, 75, 91, 9, 19, 14, 25], [15, 79, 64, 83, 94, 123, 62, 97], [120, 125, 105, 48, 85, 22, 55, 54], [13, 6, 92, 16, 22, 126, 10, 71], [9, 23, 78, 54, 41, 17, 117, 20], [104, 114, 25, 7, 37, 91, 109, 10], [34, 18, 33, 113, 52, 13, 50, 64], [25, 67, 10, 32, 72, 6, 95, 29], [18, 63, 26, 79, 69, 77, 91, 86], [112, 73, 107, 98, 14, 75, 69, 61], [28, 80, 59, 105, 93, 89, 110, 116], [64, 80, 21, 102, 70, 19, 110, 105], [28, 47, 46, 18, 61, 36, 64, 76], [67, 113, 5, 54, 44, 62, 21, 27], [115, 29, 73, 87, 123, 48, 55, 6], [84, 82, 1, 116, 7, 125, 55, 95], [93, 95, 20, 108, 124, 17, 106, 50]]
    rank10 = [[127, 8, 26, 48, 70, 13, 81, 95], [64, 76, 34, 127, 7, 51, 96, 67], [112, 100, 53, 49, 48, 40, 107, 96], [21, 63, 97, 100, 23, 74, 35, 33], [54, 16, 70, 7, 53, 1, 51, 68], [52, 7, 106, 96, 38, 117, 123, 91], [60, 83, 110, 120, 38, 63, 31, 36], [60, 75, 76, 53, 35, 48, 52, 44], [36, 40, 32, 85, 14, 83, 94, 7], [28, 41, 79, 62, 92, 111, 97, 57], [15, 12, 99, 86, 118, 81, 85, 91], [9, 11, 24, 94, 71, 81, 76, 105], [94, 114, 11, 96, 78, 45, 115, 46], [98, 99, 80, 123, 13, 119, 126, 54], [118, 14, 110, 83, 4, 2, 104, 58], [65, 52, 20, 9, 7, 27, 94, 92], [63, 8, 29, 12, 43, 98, 50, 88], [85, 19, 50, 88, 93, 12, 120, 60], [123, 88, 115, 94, 19, 89, 85, 98], [34, 47, 127, 96, 10, 26, 104, 15], [46, 121, 88, 106, 11, 69, 44, 119], [28, 31, 81, 92, 2, 26, 22, 68], [2, 123, 75, 90, 70, 51, 64, 50], [103, 98, 11, 87, 6, 117, 76, 108], [12, 36, 53, 96, 78, 109, 73, 29], [107, 115, 109, 118, 22, 124, 91, 106], [68, 33, 83, 86, 7, 76, 36, 117], [18, 45, 115, 127, 64, 0, 82, 3], [124, 8, 7, 82, 13, 35, 93, 104], [93, 46, 90, 64, 92, 94, 95, 111], [123, 112, 41, 55, 9, 7, 0, 57], [46, 2, 100, 109, 97, 125, 57, 23], [84, 77, 65, 14, 55, 37, 111, 1], [108, 41, 53, 112, 0, 8, 71, 75], [61, 23, 88, 63, 84, 97, 30, 1], [118, 19, 90, 97, 85, 51, 115, 113], [30, 39, 0, 118, 44, 77, 76, 81], [16, 47, 57, 49, 81, 1, 84, 122], [70, 12, 80, 49, 58, 39, 108, 124], [65, 36, 38, 54, 8, 111, 55, 66], [45, 108, 20, 51, 91, 53, 52, 84], [44, 50, 18, 84, 62, 73, 104, 16], [108, 125, 20, 90, 66, 113, 5, 57], [78, 82, 53, 56, 27, 48, 116, 126], [46, 59, 35, 29, 72, 122, 119, 124], [45, 2, 53, 63, 42, 67, 39, 62], [114, 67, 39, 96, 33, 70, 14, 69], [107, 111, 6, 101, 98, 23, 122, 97]]
    rank11 = [[119, 94, 85, 78, 6, 118, 110, 15], [36, 108, 32, 16, 15, 49, 122, 4], [27, 72, 58, 76, 85, 46, 41, 124], [18, 0, 93, 50, 66, 40, 61, 22], [56, 23, 82, 110, 89, 114, 117, 62], [112, 49, 4, 6, 41, 31, 0, 40], [119, 12, 71, 101, 87, 23, 108, 61], [41, 8, 127, 61, 124, 94, 109, 28], [30, 77, 88, 66, 70, 45, 67, 95], [100, 110, 23, 12, 37, 67, 80, 82], [6, 36, 83, 20, 78, 95, 28, 50], [22, 95, 111, 69, 51, 124, 64, 12], [104, 9, 26, 113, 24, 124, 65, 102], [16, 75, 25, 36, 52, 84, 48, 50], [21, 20, 28, 60, 8, 53, 15, 57], [29, 107, 101, 98, 69, 51, 84, 6], [55, 65, 39, 70, 114, 46, 78, 19], [44, 7, 15, 21, 124, 123, 51, 67], [42, 52, 112, 87, 7, 25, 32, 102], [58, 55, 86, 79, 5, 82, 98, 77], [103, 77, 64, 32, 9, 13, 29, 33], [45, 127, 114, 17, 10, 118, 109, 75], [87, 12, 73, 16, 20, 8, 28, 18], [10, 111, 46, 32, 41, 66, 7, 115], [97, 15, 75, 33, 1, 19, 21, 6], [21, 99, 66, 61, 82, 14, 43, 17], [70, 20, 32, 62, 91, 98, 9, 13], [53, 87, 101, 56, 21, 41, 119, 62], [92, 44, 40, 41, 48, 49, 67, 125], [97, 19, 34, 108, 124, 96, 114, 43], [76, 13, 99, 109, 75, 70, 97, 5], [52, 30, 18, 88, 121, 102, 26, 77], [107, 35, 20, 42, 17, 54, 86, 63], [9, 117, 127, 23, 26, 87, 57, 103], [101, 12, 53, 33, 72, 34, 85, 62], [1, 8, 32, 30, 6, 101, 40, 96], [35, 116, 36, 123, 50, 58, 98, 14], [28, 15, 109, 118, 19, 31, 104, 72], [21, 86, 4, 84, 73, 94, 5, 45], [103, 95, 116, 34, 110, 9, 58, 39], [0, 106, 12, 110, 83, 57, 58, 102], [57, 53, 42, 66, 30, 96, 9, 47], [94, 126, 71, 25, 112, 106, 47, 48], [77, 8, 55, 60, 70, 97, 33, 89], [58, 103, 37, 48, 50, 112, 6, 118], [18, 32, 17, 103, 64, 116, 54, 68], [38, 8, 3, 43, 115, 13, 22, 57], [32, 90, 120, 57, 125, 110, 33, 24]]
    rank12 = [[65, 63, 107, 75, 99, 18, 24, 91], [65, 3, 85, 75, 8, 45, 82, 55], [88, 54, 29, 6, 12, 1, 44, 78], [38, 17, 1, 87, 98, 34, 116, 15], [55, 112, 121, 115, 87, 34, 78, 80], [71, 72, 58, 93, 100, 21, 25, 12], [6, 14, 115, 107, 37, 4, 82, 21], [30, 50, 110, 18, 89, 21, 69, 120], [105, 21, 98, 20, 49, 44, 10, 82], [45, 9, 52, 58, 76, 0, 69, 121], [87, 107, 123, 109, 40, 76, 80, 62], [67, 8, 86, 109, 54, 92, 84, 5], [74, 15, 20, 7, 79, 47, 112, 126], [4, 8, 95, 41, 22, 93, 85, 100], [68, 93, 11, 62, 94, 92, 35, 40], [103, 95, 36, 90, 112, 26, 16, 42], [44, 40, 113, 110, 109, 115, 89, 68], [69, 34, 0, 105, 55, 106, 83, 6], [29, 9, 122, 18, 24, 54, 46, 5], [74, 70, 112, 17, 64, 66, 63, 20], [39, 114, 28, 26, 55, 41, 50, 95], [64, 117, 79, 43, 7, 115, 25, 46], [112, 4, 56, 108, 122, 102, 0, 47], [1, 56, 47, 69, 50, 104, 18, 70], [121, 26, 7, 54, 111, 45, 59, 84], [98, 76, 32, 117, 2, 126, 108, 96], [107, 51, 80, 49, 87, 106, 6, 31], [48, 30, 36, 75, 117, 35, 94, 6], [90, 62, 5, 53, 79, 115, 32, 72], [69, 85, 117, 17, 12, 78, 82, 25], [88, 17, 87, 77, 90, 54, 23, 82], [34, 43, 122, 64, 68, 7, 29, 113], [48, 30, 99, 51, 5, 6, 91, 58], [110, 63, 50, 62, 86, 67, 70, 102], [15, 2, 65, 90, 111, 114, 29, 18], [67, 22, 72, 119, 29, 25, 14, 110], [94, 12, 26, 34, 124, 29, 126, 84], [23, 25, 123, 99, 86, 59, 87, 29], [68, 48, 60, 91, 87, 59, 31, 9], [52, 104, 11, 100, 27, 82, 121, 113], [63, 101, 48, 119, 11, 105, 78, 68], [69, 10, 46, 117, 100, 118, 70, 60], [77, 117, 101, 65, 116, 10, 95, 72], [3, 115, 123, 68, 108, 1, 127, 73], [121, 68, 80, 116, 30, 24, 83, 125], [7, 8, 83, 101, 51, 88, 36, 74], [32, 113, 97, 64, 15, 36, 106, 87], [109, 3, 30, 54, 18, 46, 85, 62]]
    rank13 = [[21, 123, 46, 9, 29, 28, 83, 42], [124, 42, 50, 38, 77, 91, 73, 110], [84, 87, 116, 104, 60, 13, 74, 66], [79, 9, 90, 27, 68, 111, 72, 125], [66, 92, 47, 10, 99, 120, 81, 21], [47, 107, 54, 48, 110, 83, 67, 118], [15, 99, 41, 93, 34, 56, 114, 124], [54, 83, 43, 67, 119, 62, 77, 92], [81, 84, 47, 65, 4, 69, 90, 91], [104, 83, 31, 114, 11, 26, 78, 68], [23, 61, 66, 35, 110, 65, 29, 71], [100, 31, 72, 77, 32, 30, 15, 70], [117, 60, 50, 43, 91, 18, 8, 40], [63, 76, 15, 118, 61, 91, 43, 40], [79, 111, 101, 90, 82, 72, 116, 23], [71, 85, 104, 24, 10, 17, 3, 32], [124, 42, 4, 73, 97, 27, 80, 102], [57, 53, 117, 64, 66, 89, 41, 70], [83, 43, 120, 126, 86, 23, 108, 50], [95, 49, 41, 110, 106, 89, 116, 44], [30, 109, 20, 51, 6, 45, 111, 82], [100, 65, 32, 12, 48, 39, 78, 113], [15, 45, 57, 63, 9, 3, 89, 52], [61, 19, 78, 58, 54, 25, 51, 113], [60, 99, 68, 95, 51, 125, 8, 61], [67, 4, 44, 24, 52, 110, 10, 101], [25, 63, 18, 64, 4, 121, 29, 38], [71, 52, 38, 90, 61, 51, 77, 73], [45, 54, 15, 33, 87, 85, 16, 56], [28, 49, 18, 99, 105, 103, 125, 16], [12, 119, 120, 24, 56, 25, 81, 92], [49, 54, 61, 67, 124, 82, 3, 120], [36, 18, 66, 2, 56, 127, 80, 60], [28, 83, 116, 49, 118, 11, 77, 68], [105, 123, 7, 66, 120, 76, 25, 21], [49, 100, 86, 87, 92, 21, 28, 12], [60, 117, 108, 111, 56, 85, 102, 17], [21, 32, 80, 117, 97, 127, 43, 7], [93, 79, 105, 47, 76, 75, 104, 114], [45, 85, 76, 2, 125, 25, 3, 99], [90, 8, 29, 4, 26, 85, 32, 104], [7, 85, 52, 35, 124, 83, 86, 91], [31, 7, 39, 88, 50, 4, 52, 40], [58, 100, 19, 121, 106, 41, 51, 57], [14, 105, 11, 41, 108, 102, 89, 18], [50, 52, 112, 76, 5, 102, 37, 60], [103, 94, 105, 83, 26, 66, 78, 2], [102, 34, 36, 79, 59, 103, 10, 29]]
    rank14 = [[72, 14, 93, 43, 66, 23, 38, 51], [17, 94, 48, 9, 69, 78, 61, 40], [47, 108, 7, 99, 126, 10, 120, 79], [29, 104, 59, 46, 70, 67, 53, 96], [124, 13, 67, 79, 29, 43, 12, 45], [126, 125, 70, 103, 18, 37, 98, 23], [11, 65, 118, 78, 104, 28, 26, 40], [51, 95, 72, 96, 68, 100, 114, 25], [109, 114, 19, 57, 3, 8, 127, 60], [81, 64, 117, 56, 7, 122, 89, 77], [31, 4, 14, 97, 63, 53, 89, 0], [122, 19, 42, 87, 53, 7, 117, 73], [63, 116, 69, 54, 100, 108, 59, 87], [28, 6, 47, 55, 65, 35, 104, 7], [107, 48, 80, 17, 47, 106, 6, 56], [45, 30, 115, 11, 63, 21, 78, 89], [22, 71, 60, 51, 41, 58, 67, 84], [28, 49, 109, 2, 22, 62, 43, 27], [16, 105, 96, 109, 75, 74, 28, 113], [54, 8, 72, 101, 115, 1, 3, 57], [79, 84, 85, 42, 5, 67, 108, 104], [41, 110, 112, 49, 99, 111, 5, 97], [13, 105, 83, 74, 92, 78, 58, 69], [49, 8, 114, 109, 77, 53, 15, 116], [89, 63, 43, 11, 18, 79, 4, 40], [28, 6, 95, 41, 105, 127, 68, 51], [93, 12, 105, 100, 8, 42, 53, 37], [118, 107, 72, 79, 9, 27, 78, 81], [22, 63, 29, 12, 103, 89, 25, 19], [57, 53, 42, 61, 1, 83, 86, 71], [105, 29, 71, 65, 74, 121, 61, 50], [8, 107, 126, 119, 106, 5, 84, 93], [103, 109, 28, 3, 11, 61, 76, 31], [65, 31, 107, 10, 33, 38, 15, 46], [27, 4, 56, 48, 124, 81, 64, 44], [61, 56, 125, 13, 69, 116, 76, 2], [99, 121, 33, 51, 18, 8, 105, 90], [6, 4, 36, 41, 71, 26, 40, 5], [0, 118, 7, 2, 22, 78, 99, 50], [88, 4, 71, 13, 7, 19, 44, 73], [55, 54, 34, 111, 1, 70, 49, 56], [48, 19, 97, 64, 54, 103, 102, 113], [12, 100, 78, 76, 27, 115, 53, 3], [66, 88, 81, 16, 112, 35, 40, 20], [73, 0, 7, 100, 31, 8, 86, 77], [99, 20, 27, 110, 86, 126, 30, 122], [77, 46, 29, 11, 93, 52, 63, 6], [44, 127, 123, 69, 42, 78, 7, 60]]
    rank15 = [[117, 16, 89, 44, 101, 47, 60, 96], [26, 95, 126, 25, 30, 21, 115, 13], [31, 56, 98, 68, 15, 59, 64, 3], [64, 91, 94, 92, 109, 47, 43, 44], [98, 76, 4, 85, 8, 116, 15, 59], [13, 34, 79, 105, 102, 36, 73, 19], [16, 96, 112, 9, 103, 54, 45, 51], [107, 34, 2, 22, 118, 33, 63, 117], [15, 79, 87, 56, 9, 53, 74, 86], [65, 16, 32, 43, 8, 50, 25, 61], [2, 93, 103, 54, 22, 84, 41, 52], [1, 61, 90, 119, 29, 104, 121, 106], [121, 12, 0, 21, 23, 125, 42, 49], [77, 23, 44, 127, 2, 110, 92, 120], [51, 12, 18, 0, 19, 22, 113, 31], [53, 38, 116, 106, 70, 12, 50, 121], [0, 92, 5, 82, 81, 25, 127, 94], [5, 97, 17, 18, 77, 29, 9, 116], [17, 12, 14, 3, 67, 93, 92, 22], [107, 52, 39, 67, 59, 4, 114, 113], [36, 27, 57, 49, 70, 72, 8, 38], [83, 9, 84, 14, 85, 37, 15, 82], [27, 61, 6, 40, 42, 22, 77, 104], [67, 24, 100, 119, 107, 55, 60, 17], [30, 117, 0, 122, 100, 106, 119, 105], [77, 15, 18, 123, 30, 20, 122, 112], [48, 79, 17, 39, 94, 95, 104, 66], [29, 85, 11, 106, 24, 112, 121, 16], [42, 0, 20, 28, 43, 88, 122, 24], [5, 44, 52, 35, 36, 62, 112, 70], [43, 83, 14, 62, 37, 84, 80, 64], [58, 42, 123, 101, 10, 94, 73, 28], [46, 121, 85, 32, 41, 50, 8, 43], [100, 64, 84, 96, 7, 69, 66, 109], [87, 93, 119, 75, 96, 26, 41, 60], [45, 24, 26, 102, 63, 53, 73, 70], [15, 63, 96, 107, 125, 115, 73, 2], [107, 76, 121, 2, 61, 83, 92, 30], [51, 83, 33, 96, 109, 23, 15, 54], [29, 30, 60, 48, 56, 16, 12, 123], [22, 124, 7, 79, 126, 82, 127, 67], [55, 34, 15, 14, 77, 22, 95, 87], [32, 37, 49, 17, 33, 87, 15, 121], [10, 94, 4, 38, 91, 39, 85, 107], [53, 1, 9, 82, 26, 60, 76, 120], [111, 71, 92, 19, 93, 79, 120, 35], [44, 123, 71, 34, 0, 42, 112, 24], [4, 117, 105, 114, 22, 64, 49, 126]]
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
    # rank0  = [85, 59, 73, 109, 42, 46, 62, 108]
    # rank1  = [58, 91, 53, 64, 17, 94, 37, 98]
    # rank2  = [105, 114, 80, 106, 44, 81, 112, 101]
    # rank3  = [21, 15, 23, 120, 89, 22, 113, 116]
    # rank4  = [13, 71, 66, 69, 82, 121, 26, 100]
    # rank5  = [7, 20, 51, 52, 123, 104, 40, 28]
    # rank6  = [56, 11, 47, 93, 32, 67, 78, 84]
    # rank7  = [99, 27, 3, 5, 65, 110, 95, 41]
    # rank8  = [29, 57, 4, 61, 127, 54, 39, 49]
    # rank9  = [103, 74, 124, 77, 25, 19, 86, 75]
    # rank10 = [1, 107, 79, 48, 125, 60, 63, 68]
    # rank11 = [8, 12, 33, 55, 38, 31, 70, 16]
    # rank12 = [111, 87, 118, 10, 76, 24, 6, 88]
    # rank13 = [92, 35, 30, 96, 50, 115, 117, 9]
    # rank14 = [90, 14, 34, 102, 126, 45, 72, 122]
    # rank15 = [119, 36, 83, 97, 43, 2, 18, 0]
    rank_by_ep = [rank0, rank1, rank2, rank3, rank4, rank5, rank6, rank7, rank8,
    rank9, rank10, rank11, rank12, rank13, rank14, rank15]
    rank_list = rank_by_ep[ep_rank]
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
    '''

    ############ Original expert placement logic ############
    # Create an expert map for the local experts
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
    print("use original expert placement for ep_rank", ep_rank, "original expert_map:", expert_map.size(), expert_map)
    l2p = None
    return (local_num_experts, expert_map, l2p)

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
        if self.ep_rank == 0:
            print("rank", self.ep_rank, "loading weight:", loaded_weight.shape)
            print("ep_rank", self.ep_rank, "before map, expert_id:", expert_id)
        expert_id = self._map_global_expert_id_to_local_expert_id(expert_id)
        if self.ep_rank == 0:
            print("ep_rank", self.ep_rank,"after map, expert_id:", expert_id)
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
        # #修改映射逻辑
        # physical_to_logical_map = [
        #                             85, 59, 73, 109, 42, 46, 62, 108, 58, 91, 53, 64, 17, 94, 37, 98,
        #                             105, 114, 80, 106, 44, 81, 112, 101, 21, 15, 23, 120, 89, 22, 113, 116,
        #                             13, 71, 66, 69, 82, 121, 26, 100, 7, 20, 51, 52, 123, 104, 40, 28,
        #                             56, 11, 47, 93, 32, 67, 78, 84, 99, 27, 3, 5, 65, 110, 95, 41,
        #                             29, 57, 4, 61, 127, 54, 39, 49, 103, 74, 124, 77, 25, 19, 86, 75,
        #                             1, 107, 79, 48, 125, 60, 63, 68, 8, 12, 33, 55, 38, 31, 70, 16,
        #                             111, 87, 118, 10, 76, 24, 6, 88, 92, 35, 30, 96, 50, 115, 117, 9,
        #                             90, 14, 34, 102, 126, 45, 72, 122, 119, 36, 83, 97, 43, 2, 18, 0
        #                             ]
        # physical_to_logical_map = all_ranks = [
        #                             127, 80, 125, 58, 66, 59, 102, 40,
        #                             88, 111, 99, 49, 89, 32, 113, 25,
        #                             95, 12, 126, 77, 41, 24, 29, 26,
        #                             101, 76, 38, 57, 47, 64, 106, 93,
        #                             52, 90, 114, 105, 121, 14, 92, 70,
        #                             46, 63, 4, 124, 20, 117, 5, 50,
        #                             83, 71, 108, 42, 43, 10, 69, 91,
        #                             48, 65, 8, 1, 85, 67, 6, 86,
        #                             11, 60, 34, 53, 87, 35, 94, 33,
        #                             118, 2, 73, 79, 100, 75, 54, 82,
        #                             18, 21, 36, 122, 55, 0, 78, 97,
        #                             103, 28, 112, 9, 104, 51, 13, 62,
        #                             107, 123, 15, 56, 39, 23, 115, 72,
        #                             45, 16, 19, 81, 7, 3, 61, 96,
        #                             22, 30, 17, 109, 31, 110, 98, 120,
        #                             27, 37, 119, 44, 74, 84, 116, 68
        #                             ]
        print("physical_to_logical_map:", physical_to_logical_map)
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
