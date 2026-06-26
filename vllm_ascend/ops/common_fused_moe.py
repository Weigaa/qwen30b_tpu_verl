#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
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
#
import os.path
import re
from typing import Callable, Optional

import torch
import torch_npu
from vllm.config import CompilationLevel, get_current_vllm_config
from vllm.distributed import (get_dp_group, get_ep_group, get_tp_group,
                              tensor_model_parallel_all_reduce)
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.fused_moe.layer import (
    FusedMoE, UnquantizedFusedMoEMethod, determine_expert_map)
from vllm.model_executor.layers.shared_fused_moe import SharedFusedMoE

from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.ascend_forward_context import MoECommType
from vllm_ascend.distributed.parallel_state import get_mc2_group
from vllm_ascend.eplb.core.eplb_utils import (determine_default_expert_map,
                                              determine_default_log2phy_map)
from vllm_ascend.ops.expert_load_balancer import ExpertLoadBalancer
from vllm_ascend.ops.fused_moe import AscendFusedMoE as CustomAscendFusedMoE
from vllm_ascend.ops.moe.experts_selector import select_experts
from vllm_ascend.ops.moe.moe_comm_method import setup_moe_comm_method
from vllm_ascend.utils import ACL_FORMAT_FRACTAL_NZ, is_310p, npu_stream_switch

original_unquantized_fused_moe_init_func = UnquantizedFusedMoEMethod.__init__


def _infer_layer_idx_from_prefix(prefix: str) -> int:
    match = re.search(r"(?:^|\.)(?:layers|h)\.(\d+)(?:\.|$)", prefix)
    if match is None:
        match = re.search(r"(?:^|\.)(\d+)\.(?:mlp|experts|block|layer)(?:\.|$)",
                          prefix)
    if match is None:
        return -1
    try:
        return int(match.group(1))
    except (TypeError, ValueError):
        return -1


def unquantized_fused_moe_init_func(self, *args, **kwargs):
    original_unquantized_fused_moe_init_func(self, *args, **kwargs)

    # NOTE: Currently, this self.use_aclgraph is only used in
    # UnquantizedFusedMoEMethod.forward_oot to decide whether to use in
    # ops/fused_moe.py:568 to circumvent torch.randint_like not supported issue.
    # Once torch.randint_like is supported or removed, this flag can be removed.
    vllm_config = get_current_vllm_config()
    ascend_config = get_ascend_config()
    if ascend_config.torchair_graph_config.enabled:
        self.use_aclgraph = False
    else:
        self.use_aclgraph = (vllm_config.compilation_config.level
                             == CompilationLevel.PIECEWISE
                             and not vllm_config.model_config.enforce_eager)
    self.transpose = True


def forward_oot(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        use_grouped_topk: bool,
        top_k: int,
        router_logits: torch.Tensor,
        renormalize: bool,
        topk_group: Optional[int] = None,
        num_expert_group: Optional[int] = None,
        custom_routing_function: Optional[Callable] = None,
        scoring_func: str = "softmax",
        routed_scaling_factor: float = 1.0,
        e_score_correction_bias: Optional[torch.Tensor] = None,
        global_num_experts: int = -1,
        expert_map: Optional[torch.Tensor] = None,
        apply_router_weight_on_input: bool = False,
        activation: str = "silu",
        enable_eplb: bool = False,
        expert_load_view: Optional[torch.Tensor] = None,
        logical_to_physical_map: Optional[torch.Tensor] = None,
        logical_replica_count: Optional[torch.Tensor] = None) -> torch.Tensor:

    topk_weights, topk_ids, row_idx = select_experts(
        hidden_states=x,
        router_logits=router_logits,
        top_k=top_k,
        use_grouped_topk=use_grouped_topk,
        renormalize=renormalize,
        topk_group=topk_group,
        num_expert_group=num_expert_group,
        custom_routing_function=custom_routing_function,
        scoring_func=scoring_func,
        routed_scaling_factor=routed_scaling_factor,
        e_score_correction_bias=e_score_correction_bias,
        global_num_experts=global_num_experts)

    moe_comm_method = get_forward_context().moe_comm_method
    return moe_comm_method.fused_experts(hidden_states=x,
                                         w1=layer.w13_weight,
                                         w2=layer.w2_weight,
                                         topk_weights=topk_weights,
                                         topk_ids=topk_ids,
                                         row_idx=row_idx,
                                         global_num_experts=global_num_experts,
                                         expert_map=expert_map)


def process_weights_after_loading(self, layer):
    super(UnquantizedFusedMoEMethod, self).process_weights_after_loading(layer)
    if self.transpose:
        w13_data = self._maybe_pad_weight(layer.w13_weight.data).transpose(
            1, 2).contiguous()
        layer.w13_weight = torch.nn.Parameter(w13_data, requires_grad=False)

        w2_data = self._maybe_pad_weight(layer.w2_weight.data).transpose(
            1, 2).contiguous()
        layer.w2_weight = torch.nn.Parameter(w2_data, requires_grad=False)

        self.transpose = False
    else:
        w13_data = self._maybe_pad_weight(layer.w13_weight.data)
        layer.w13_weight = torch.nn.Parameter(w13_data, requires_grad=False)

        w2_data = self._maybe_pad_weight(layer.w2_weight.data)
        layer.w2_weight = torch.nn.Parameter(w2_data, requires_grad=False)

    if not is_310p():
        layer.w13_weight.data = torch_npu.npu_format_cast(
            layer.w13_weight.data, ACL_FORMAT_FRACTAL_NZ)
        layer.w2_weight.data = torch_npu.npu_format_cast(
            layer.w2_weight.data, ACL_FORMAT_FRACTAL_NZ)


class AscendFusedMoE(CustomAscendFusedMoE):
    moe_counter = -1

    def __init__(
        self,
        num_experts: int,
        top_k: int,
        hidden_size: int,
        intermediate_size: int,
        params_dtype: Optional[torch.dtype] = None,
        reduce_results: bool = False,
        renormalize: bool = True,
        use_grouped_topk: bool = False,
        num_expert_group: Optional[int] = None,
        topk_group: Optional[int] = None,
        quant_config=None,
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
        is_sequence_parallel: bool = False,
        zero_expert_num: Optional[int] = 0,
        zero_expert_type: Optional[str] = None,
        layer_idx: int = -1,
    ):
        del routed_scaling_factor, num_redundant_experts, has_bias
        if layer_idx is None or int(layer_idx) < 0:
            layer_idx = _infer_layer_idx_from_prefix(prefix)
        super().__init__(
            num_experts=num_experts,
            top_k=top_k,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            params_dtype=params_dtype,
            reduce_results=reduce_results,
            renormalize=renormalize,
            use_grouped_topk=use_grouped_topk,
            num_expert_group=num_expert_group,
            topk_group=topk_group,
            quant_config=quant_config,
            tp_size=tp_size,
            ep_size=ep_size,
            dp_size=dp_size,
            prefix=prefix,
            custom_routing_function=custom_routing_function,
            scoring_func=scoring_func,
            e_score_correction_bias=e_score_correction_bias,
            activation=activation,
            apply_router_weight_on_input=apply_router_weight_on_input,
            layer_idx=layer_idx,
        )
        self.enable_eplb = enable_eplb
        self.is_sequence_parallel = is_sequence_parallel
        self.zero_expert_num = zero_expert_num
        self.zero_expert_type = zero_expert_type

    def update_expert_map(self, new_expert_map):
        self.expert_map = new_expert_map

    def get_map(self):
        return self.expert_map

    def get_log2phy_map(self):
        return self.log2phy

    def clear_moe_load(self):
        if self.moe_load is not None:
            self.moe_load.zero_()

    def maybe_all_reduce_tensor_model_parallel(
            self, final_hidden_states: torch.Tensor):
        """NOTE(Yizhou): This is to override the parent class method. In `mc2commimpl`,
        and `alltoallcommimpl`, we do not need to all-reduce the final outputs since
        the outputs are already aggregated across tensor parallel ranks in the
        `finalize` function. In `allgathercommimpl`, we still need to all-reduce the
        outputs since each rank only has partial outputs.
        """
        return torch.ops.vllm.maybe_all_reduce_tensor_model_parallel(
            final_hidden_states)

    def forward(self,
                hidden_states: torch.Tensor,
                router_logits: torch.Tensor,
                is_prefill: Optional[bool] = None,
                enable_force_load_balance: bool = False,
                top_k: Optional[int] = None,
                shared_experts: Optional[torch.nn.Module] = None,
                gate=None,
                replace_allreduce: bool = False,
                is_dummy: bool = False):
        if is_prefill is None:
            try:
                is_prefill = bool(get_forward_context().with_prefill)
            except Exception:
                is_prefill = False
        return super().forward(hidden_states=hidden_states,
                               router_logits=router_logits,
                               is_prefill=is_prefill,
                               enable_force_load_balance=enable_force_load_balance,
                               top_k=top_k,
                               shared_experts=shared_experts,
                               gate=gate,
                               replace_allreduce=replace_allreduce,
                               is_dummy=is_dummy)

    def forward_impl(self, hidden_states: torch.Tensor,
                     router_logits: torch.Tensor):
        return self.forward(hidden_states=hidden_states,
                            router_logits=router_logits)

    def transpose_weight(self, loaded_weight, expert_data, shard_dim):
        # Ensure training and inference weight shapes match during RL weight updates
        if (
            loaded_weight.shape[1] != expert_data.shape[1] and \
            loaded_weight.shape[0] != expert_data.shape[0]
        ):
            shard_dim = int(not shard_dim)
            loaded_weight = loaded_weight.transpose(0, 1).contiguous()
        return loaded_weight, shard_dim

    def _load_w13(self,
                  expert_data: torch.Tensor,
                  shard_dim: int,
                  shard_id: str,
                  loaded_weight: torch.Tensor,
                  tp_rank: int,
                  load_full: bool = False):
        # Index the loaded weight for tp sharding.
        # gate_up_proj: "MergedColumnParallel", so tp sharding on output_dim
        loaded_weight, shard_dim = self.transpose_weight(
            loaded_weight, expert_data, shard_dim)
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
        loaded_weight, shard_dim = self.transpose_weight(
            loaded_weight, expert_data, shard_dim)
        shard_size = expert_data.shape[shard_dim]
        if not load_full:
            loaded_weight = loaded_weight.narrow(shard_dim,
                                                 shard_size * tp_rank,
                                                 shard_size)
        # w2, down_proj: Load into only logical weight of w2.
        expert_data.copy_(loaded_weight)


class AscendSharedFusedMoE(SharedFusedMoE, AscendFusedMoE):

    def __init__(
        self,
        shared_experts: torch.nn.Module,
        use_overlapped: bool = True,
        **kwargs,
    ):
        AscendFusedMoE.__init__(self, **kwargs)
        self._shared_experts = shared_experts
        self.use_overlapped = use_overlapped
        self.shared_expert_stream = None
        ascend_config = get_ascend_config()
        self.multistream_overlap_shared_expert = ascend_config.multistream_overlap_shared_expert
        if self.multistream_overlap_shared_expert:
            self.shared_expert_stream = torch.npu.Stream()

    def forward(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        shared_out, fused_out = AscendFusedMoE.forward(
            self,
            hidden_states=hidden_states,
            router_logits=router_logits,
        )
        return shared_out, fused_out

    def forward_impl(self, hidden_states: torch.Tensor,
                     router_logits: torch.Tensor):
        # Make sure the shared experts stream begins after hidden_states are ready.
        if self.multistream_overlap_shared_expert:
            self.shared_expert_stream.wait_stream(  # type: ignore
                torch.npu.current_stream())
        with npu_stream_switch(self.shared_expert_stream,
                               enabled=self.multistream_overlap_shared_expert):
            # Use a separate stream to run shared experts.
            shared_out = self._shared_experts(hidden_states)

            # NOTE: This is exactly the opposite of `maybe_all_reduce_tensor_model_parallel`
            forward_context = get_forward_context()
            moe_comm_type = forward_context.moe_comm_type
            if moe_comm_type in {MoECommType.ALLTOALL, MoECommType.MC2}:
                shared_out = tensor_model_parallel_all_reduce(shared_out)
        fused_output = AscendFusedMoE.forward_impl(
            self,
            hidden_states=hidden_states,
            router_logits=router_logits,
        )
        # Make sure the default stream waits for the shared experts stream to finish.
        if self.multistream_overlap_shared_expert:
            torch.npu.current_stream().wait_stream(self.shared_expert_stream)
        return shared_out, fused_output


UnquantizedFusedMoEMethod.__init__ = unquantized_fused_moe_init_func
UnquantizedFusedMoEMethod.process_weights_after_loading = process_weights_after_loading
UnquantizedFusedMoEMethod.forward_oot = forward_oot
