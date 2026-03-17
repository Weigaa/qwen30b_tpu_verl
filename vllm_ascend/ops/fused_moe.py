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
# Adapted from vllm/tests/kernels/test_moe.py

import gc
import os
from typing import Any, Callable, Optional

import torch
import torch_npu
from vllm.config import get_current_vllm_config
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.distributed.parallel_state import (get_dp_group, get_ep_group,
                                             get_tp_group)
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.config import \
    FusedMoEConfig  # isort: skip
from vllm.model_executor.layers.fused_moe.config import \
    FusedMoEParallelConfig  # isort: skip
from vllm.model_executor.layers.fused_moe.layer import (
    FusedMoE, UnquantizedFusedMoEMethod, determine_expert_map)
from vllm.model_executor.layers.quantization.base_config import \
    QuantizationConfig

from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.distributed.parallel_state import get_mc2_group
from vllm_ascend import envs as envs_ascend
from vllm_ascend.eplb.core.eplb_utils import (
    determine_default_expert_map, determine_default_log2phy_map,
    determine_redundant_replica_expert_map,
    determine_redundant_replica_log2phy_map)
from vllm_ascend.ops.expert_load_balancer import ExpertLoadBalancer
from vllm_ascend.ops.moe.experts_selector import select_experts
from vllm_ascend.ops.moe.moe_comm_method import setup_moe_comm_method
from vllm_ascend.utils import (ACL_FORMAT_FRACTAL_NZ,
                               get_all_reduce_merge_state,
                               get_rm_router_logits_state, is_310p,
                               vllm_version_is)
from vllm.utils.moe_stats import moe_stats

logger = init_logger(__name__)


class AscendUnquantizedFusedMoEMethod(UnquantizedFusedMoEMethod):

    def __init__(self, moe: FusedMoEConfig = None):

        super().__init__(moe=moe)
        vllm_config = get_current_vllm_config()

        self.global_batch_size = vllm_config.scheduler_config.max_num_seqs
        self.max_model_len = vllm_config.model_config.max_model_len
        get_ascend_config()
        self.dynamic_eplb = get_ascend_config().dynamic_eplb

        try:
            device_group = get_mc2_group().device_group
            # TODO: Try local_rank = ep_group.rank_in_group
            local_rank = torch.distributed.get_rank(group=device_group)
            self.ep_rank = local_rank 
            backend = device_group._get_backend(torch.device("npu"))
            self.moe_all_to_all_group_name = backend.get_hccl_comm_name(
                local_rank)
        except AttributeError:
            self.moe_all_to_all_group_name = None

    def process_weights_after_loading(self, layer):
        super(UnquantizedFusedMoEMethod,
              self).process_weights_after_loading(layer)
        layer.w13_weight = torch.nn.Parameter(self._maybe_pad_weight(
            layer.w13_weight.data),
                                              requires_grad=False)
        layer.w2_weight = torch.nn.Parameter(self._maybe_pad_weight(
            layer.w2_weight.data),
                                             requires_grad=False)
        if not is_310p():
            layer.w13_weight.data = torch_npu.npu_format_cast(
                layer.w13_weight.data, ACL_FORMAT_FRACTAL_NZ)
            layer.w2_weight.data = torch_npu.npu_format_cast(
                layer.w2_weight.data, ACL_FORMAT_FRACTAL_NZ)
        if getattr(layer, "elastic_moe_mode", "lossy") == "lossless":
            layer.activate_lossless_primary_experts()

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        router_logits: torch.Tensor,
        top_k: int,
        renormalize: bool,
        use_grouped_topk: bool = False,
        global_num_experts: int = -1,
        expert_map: Optional[torch.Tensor] = None,
        topk_group: Optional[int] = None,
        num_expert_group: Optional[int] = None,
        custom_routing_function: Optional[Callable] = None,
        scoring_func: str = "softmax",
        e_score_correction_bias: Optional[torch.Tensor] = None,
        is_prefill: bool = False,
        enable_force_load_balance: bool = False,
        shared_experts: Optional[Any] = None,
        log2phy: Optional[Any] = None,
        **kwargs,
    ) -> torch.Tensor:
        active_expert_mask = kwargs.get("active_expert_mask")
        if active_expert_mask is not None:
            mask = active_expert_mask.to(device=router_logits.device,
                                         dtype=torch.bool)
            if torch.any(mask):
                min_value = torch.finfo(router_logits.dtype).min
                router_logits = router_logits.masked_fill(~mask, min_value)

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
            e_score_correction_bias=e_score_correction_bias,
            global_num_experts=global_num_experts)

        topk_weights = topk_weights.to(x.dtype)
        layer_idx = kwargs.get("layer_idx", getattr(layer, "layer_idx", -1))
        is_dummy = kwargs.get("is_dummy", False)
        moe_comm_method = get_forward_context().moe_comm_method
        debug_info = getattr(get_forward_context(), "elastic_debug_info", None)
        if debug_info is not None and not is_dummy:
            flat_topk_ids = topk_ids.reshape(-1)
            pre_unique_ids, pre_counts = (
                torch.unique(flat_topk_ids, return_counts=True)
                if flat_topk_ids.numel() > 0 else
                (torch.empty(0, device=topk_ids.device, dtype=topk_ids.dtype),
                 torch.empty(0, device=topk_ids.device, dtype=torch.int64)))
            pre_pairs = []
            if pre_counts.numel() > 0:
                order = torch.argsort(pre_counts, descending=True)[:8]
                pre_pairs = [(int(pre_unique_ids[idx].item()),
                              int(pre_counts[idx].item())) for idx in order]
            mapped_pairs = []
            mapped_min = -1
            mapped_max = -1
            mapped_unique = 0
            if log2phy is not None and flat_topk_ids.numel() > 0:
                mapped_preview = log2phy[topk_ids].reshape(-1)
                mapped_unique_ids, mapped_counts = torch.unique(
                    mapped_preview, return_counts=True)
                mapped_min = int(mapped_preview.min().item())
                mapped_max = int(mapped_preview.max().item())
                mapped_unique = int(mapped_unique_ids.numel())
                mapped_order = torch.argsort(mapped_counts,
                                             descending=True)[:8]
                mapped_pairs = [(int(mapped_unique_ids[idx].item()),
                                 int(mapped_counts[idx].item()))
                                for idx in mapped_order]
            active_mask = kwargs.get("active_expert_mask")
            active_mask_count = (int(active_mask.sum().item())
                                 if active_mask is not None else -1)
            logger.info(
                "Elastic route summary: rank=%s layer=%s tag=%s reason=%s comm=%s topk_shape=%s pre_topk_min=%s pre_topk_max=%s pre_unique=%s pre_top=%s mapped_topk_min=%s mapped_topk_max=%s mapped_unique=%s mapped_top=%s active_mask_count=%s log2phy_min=%s log2phy_max=%s",
                getattr(layer, "ep_rank", -1),
                layer_idx,
                debug_info.get("tag"),
                debug_info.get("reason"),
                moe_comm_method.__class__.__name__,
                tuple(topk_ids.shape),
                int(flat_topk_ids.min().item())
                if flat_topk_ids.numel() > 0 else -1,
                int(flat_topk_ids.max().item())
                if flat_topk_ids.numel() > 0 else -1,
                int(pre_unique_ids.numel()),
                pre_pairs,
                mapped_min,
                mapped_max,
                mapped_unique,
                mapped_pairs,
                active_mask_count,
                debug_info.get("log2phy_min"),
                debug_info.get("log2phy_max"))
        if (log2phy is not None
                and moe_comm_method.__class__.__name__ != "AlltoAllCommImpl"):
            old_topk_ids = topk_ids
            topk_ids = log2phy[topk_ids]
        # this is a naive implementation for experts load balance so as
        # to avoid accumulating too much tokens on a single rank.
        # currently it is only activated when doing profile runs.
        if enable_force_load_balance and not self.use_aclgraph:
            topk_ids = (torch.arange(topk_ids.numel(), device=topk_ids.device) % global_num_experts).to(torch.int32).reshape(topk_ids.shape)
        # 考虑在dummy_run场景下也打开？
        # if (enable_force_load_balance and not self.use_aclgraph) or is_dummy:
        #     topk_ids = (torch.arange(topk_ids.numel(), device=topk_ids.device) % global_num_experts).to(torch.int32).reshape(topk_ids.shape)
        #     # if is_dummy:
        #     #     print("dummy run enable force load balance, layer_idx:", layer_idx)

        # #记录topk_ids用于统计
        # if topk_ids.shape[0] == 32:
        #     moe_stats.record_topk_ids(self.ep_rank, layer_idx, old_topk_ids, topk_ids)
        # mc = get_forward_context().moe_comm_method
        # print("moe_comm_method:", type(mc), getattr(mc, "comm_type", None), getattr(mc, "__dict__", None))
        runtime_w13_weight = getattr(layer, "runtime_w13_weight", None)
        runtime_w2_weight = getattr(layer, "runtime_w2_weight", None)
        return moe_comm_method.fused_experts(
            hidden_states=x,
            w1=runtime_w13_weight
            if runtime_w13_weight is not None else layer.w13_weight,
            w2=runtime_w2_weight
            if runtime_w2_weight is not None else layer.w2_weight,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            row_idx=row_idx,
            global_num_experts=global_num_experts,
            expert_map=expert_map,
            shared_experts=shared_experts,
            quantized_x_for_share=kwargs.get("quantized_x_for_share"),
            dynamic_scale_for_share=kwargs.get("dynamic_scale_for_share"),
            log2phy=log2phy,
            global_redundant_expert_num=kwargs.get(
                "global_redundant_expert_num", 0),
            need_trans=True,
            dynamic_eplb=self.dynamic_eplb)


class AscendFusedMoE(FusedMoE):

    # The moe_counter parameter is required during the initialization of EPLB
    # to identify the current layer index within the MOE model.
    moe_counter = -1

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
        e_score_correction_bias: Optional[torch.Tensor] = None,
        activation: str = "silu",
        apply_router_weight_on_input: bool = False,
        layer_idx: Optional[int] = -1,
    ):
        # TODO: This could not initialize FusedMoE baseclass,
        # fixme and make __init__() of AscendFusedMoE more clear
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
        AscendFusedMoE.moe_counter += 1
        self.moe_instance_id = AscendFusedMoE.moe_counter
        self.layer_idx = layer_idx

        if params_dtype is None:
            params_dtype = torch.get_default_dtype()

        vllm_config = get_current_vllm_config()
        self.model_type = vllm_config.model_config.hf_config.model_type

        self.moe_parallel_config = FusedMoEParallelConfig.make(
            tp_size_=(tp_size if tp_size is not None else
                      get_tensor_model_parallel_world_size()),
            dp_size_=(dp_size
                      if dp_size is not None else get_dp_group().world_size),
            vllm_parallel_config=vllm_config.parallel_config)

        self.top_k = top_k
        self.num_experts = num_experts
        self.global_num_experts = num_experts
        assert intermediate_size % self.tp_size == 0
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
        self.e_score_correction_bias = e_score_correction_bias
        self.expert_map = None
        self.loaded_expert_map = None
        self.elastic_original_expert_map = None
        self.activation = activation
        self.log2phy = None
        self.primary_log2phy = None
        self.active_expert_mask = None
        self.elastic_runtime_log2phy = None
        self.global_redundant_expert_num = 0
        self.active_local_num_experts = num_experts
        self.loaded_local_num_experts = num_experts
        self.runtime_w13_weight = None
        self.runtime_w2_weight = None
        self.runtime_w13_buffer = None
        self.runtime_w2_buffer = None
        self.runtime_weight_capacity = 0
        self.lossless_runtime_activated = False
        self.lossless_cpu_w13_weight = None
        self.lossless_cpu_w2_weight = None
        self.lossless_loaded_offloaded = False
        self.lossless_cpu_import_expert_ids = []
        self.elastic_debug_budget = 0
        self.elastic_debug_tag = 0
        self.elastic_debug_reason = None

        is_deepseek_v3_r1 = self.global_num_experts == 256
        self.rm_router_logits = get_rm_router_logits_state(
            self.moe_parallel_config.ep_size, self.dp_size, is_deepseek_v3_r1)
        self.all_reduce_merge = get_all_reduce_merge_state(
            self.moe_parallel_config.ep_size, is_deepseek_v3_r1)

        ascend_config = get_ascend_config()
        self.dynamic_eplb = ascend_config.dynamic_eplb
        self.elastic_moe_mode = ascend_config.elastic_moe_mode
        self.expert_map_path = ascend_config.expert_map_path
        self.global_redundant_expert_num = ascend_config.init_redundancy_expert
        if (self.elastic_moe_mode == "lossless"
                and self.global_redundant_expert_num <= 0):
            logger.warning(
                "Lossless elastic MoE mode is enabled but "
                "init_redundancy_expert=%s. Shrink will fail once surviving "
                "ranks cannot cover all logical experts.", 
                self.global_redundant_expert_num)
        self.lossless_lazy_activation = (
            self.elastic_moe_mode == "lossless"
            and self.global_redundant_expert_num <= 0)
        self.global_num_experts = num_experts
        if self.elastic_moe_mode != "lossless":
            self.global_num_experts = num_experts + self.global_redundant_expert_num
        # static eplb initializing with expert_map_path
        if self.expert_map_path and os.path.exists(
                self.expert_map_path) and os.access(self.expert_map_path,
                                                    os.R_OK):
            self.expert_load_balancer = ExpertLoadBalancer(
                self.expert_map_path, self.global_num_experts)
            self.local_num_experts, self.expert_map = (
                self.expert_load_balancer.get_rank_placement_map(
                    self.moe_instance_id, self.ep_rank))
            self.log2phy = self.expert_load_balancer.get_rank_log2phy_map(
                self.moe_instance_id, self.ep_rank).npu()
            self.global_redundant_expert_num = (
                self.expert_load_balancer.get_global_redundant_expert_num())
        else:
            # init moe.
            print("use code in vllm_ascend_ops_fused_moe to init expert_map")
            primary_mapping = determine_expert_map(
                self.ep_size, self.ep_rank, num_experts, layer_idx=self.layer_idx)
            if len(primary_mapping) == 3:
                primary_local_num_experts, primary_expert_map, primary_log2phy = primary_mapping
            elif len(primary_mapping) == 2:
                primary_local_num_experts, primary_expert_map = primary_mapping
                primary_log2phy = determine_default_log2phy_map(
                    num_experts, self.ep_size, self.ep_rank, 0)
            else:
                raise ValueError(
                    f"Unexpected determine_expert_map return arity={len(primary_mapping)}")
            if self.elastic_moe_mode == "lossless":
                self.global_redundant_expert_num = ascend_config.init_redundancy_expert
                self.loaded_local_num_experts, self.loaded_expert_map = (
                    determine_redundant_replica_expert_map(
                        num_experts, self.ep_size, self.ep_rank,
                        self.global_redundant_expert_num))
                self.active_local_num_experts = primary_local_num_experts
                self.local_num_experts = self.active_local_num_experts
                self.expert_map = primary_expert_map
                self.log2phy = primary_log2phy
                self.primary_log2phy = primary_log2phy.clone()
            # dynamic eplb initializing with not expert_map_path
            elif self.dynamic_eplb:
                self.global_redundant_expert_num = ascend_config.init_redundancy_expert
                self.local_num_experts, self.expert_map = determine_default_expert_map(
                    self.global_num_experts, self.ep_size, self.ep_rank,
                    self.global_redundant_expert_num)
                self.log2phy = determine_default_log2phy_map(
                    self.global_num_experts, self.ep_size, self.ep_rank,
                    self.global_redundant_expert_num)
                self.active_local_num_experts = self.local_num_experts
                self.loaded_local_num_experts = self.local_num_experts
            else:
                self.local_num_experts = primary_local_num_experts
                self.expert_map = primary_expert_map
                self.log2phy = primary_log2phy
                self.active_local_num_experts = self.local_num_experts
                self.loaded_local_num_experts = self.local_num_experts
                if self.expert_map is not None:
                    self.loaded_expert_map = self.expert_map.clone()
        if self.loaded_expert_map is None and self.expert_map is not None:
            self.loaded_expert_map = self.expert_map.clone()
        if self.elastic_original_expert_map is None and self.expert_map is not None:
            self.elastic_original_expert_map = self.expert_map.clone()
        if self.primary_log2phy is None and self.log2phy is not None:
            self.primary_log2phy = self.log2phy.clone()
        local_num_experts = (torch.sum(self.expert_map != -1)
                             if self.expert_map is not None else num_experts)
        if self.dynamic_eplb:
            self.moe_load = torch.zeros(local_num_experts, dtype=torch.int64)

        self.enable_shared_expert_dp = ascend_config.enable_shared_expert_dp

        if self.scoring_func != "softmax" and not self.use_grouped_topk:
            raise ValueError("Only softmax scoring function is supported for "
                             "non-grouped topk.")
        if vllm_version_is("0.10.2"):
            moe = FusedMoEConfig.make(
                num_experts=self.global_num_experts,
                experts_per_token=top_k,
                hidden_dim=hidden_size,
                num_local_experts=self.active_local_num_experts,
                moe_parallel_config=self.moe_parallel_config,
                # TODO (bnell): this needs to be fixed for quantized types.
                in_dtype=params_dtype,
                quant_config=quant_config)
        else:
            moe = FusedMoEConfig(
                num_experts=self.global_num_experts,
                experts_per_token=top_k,
                hidden_dim=hidden_size,
                num_local_experts=self.active_local_num_experts,
                moe_parallel_config=self.moe_parallel_config,
                in_dtype=params_dtype,
            )
        self.moe_config = moe
        self.moe_config.model_type = self.model_type
        self.elastic_original_num_experts = self.moe_config.num_experts
        # TODO: The self.moe_config.tp_size here is not correct, fixme soon

        if quant_config is None:
            self.quant_method = AscendUnquantizedFusedMoEMethod(moe)
        else:
            self.quant_method = quant_config.get_quant_method(self, prefix)

        assert self.quant_method is not None

        local_num_experts = torch.sum(self.expert_map != -1) \
            if self.expert_map is not None else num_experts

        self.moe_load = None

        if self.dynamic_eplb:
            self.moe_load = torch.zeros(local_num_experts, dtype=torch.int64)

        moe_quant_params = {
            "num_experts": self.loaded_local_num_experts,
            "hidden_size": hidden_size,
            "intermediate_size_per_partition":
            self.intermediate_size_per_partition,
            "params_dtype": params_dtype,
            "weight_loader": self.weight_loader,
        }
        # need full intermediate size pre-sharding for WNA16 act order
        if (self.quant_method.__class__.__name__
                in ("GPTQMarlinMoEMethod", "CompressedTensorsWNA16MoEMethod")):
            moe_quant_params["intermediate_size_full"] = intermediate_size

        self.ep_group = get_ep_group()
        # NOTE: self.tp_group is not expert_tp_group
        self.tp_group = get_tp_group().device_group
        self.quant_method.create_weights(layer=self, **moe_quant_params)

        self.moe_config.tp_group = get_tp_group()
        self.moe_config.dp_group = get_dp_group()
        self.moe_config.ep_group = get_ep_group()
        self.moe_config.mc2_group = get_mc2_group()
        self.moe_config.num_global_redundant_experts = self.global_redundant_expert_num

        setup_moe_comm_method(self.moe_config)

    def update_expert_map(self, new_expert_map):
        self.expert_map = new_expert_map

    def _map_global_expert_id_to_local_expert_id(self, expert_id: int) -> int:
        if self.elastic_moe_mode == "lossless" and self.loaded_expert_map is not None:
            return self.loaded_expert_map[expert_id].item()
        if self.expert_map is None:
            return expert_id
        return self.expert_map[expert_id].item()

    def set_runtime_num_experts(self, num_experts: int) -> None:
        self.moe_config.num_experts = int(num_experts)

    def ensure_lossless_cpu_shadow(self) -> None:
        if self.elastic_moe_mode != "lossless":
            return
        if self.lossless_cpu_w13_weight is None:
            self.lossless_cpu_w13_weight = self.w13_weight.detach().cpu()
        if self.lossless_cpu_w2_weight is None:
            self.lossless_cpu_w2_weight = self.w2_weight.detach().cpu()

    def offload_lossless_loaded_weights_to_cpu(self) -> None:
        if self.elastic_moe_mode != "lossless" or self.lossless_loaded_offloaded:
            return
        self.ensure_lossless_cpu_shadow()
        old_w13_weight = self.w13_weight
        old_w2_weight = self.w2_weight
        w13_tail_shape = tuple(self.w13_weight.shape[1:])
        w2_tail_shape = tuple(self.w2_weight.shape[1:])
        w13_dtype = self.w13_weight.dtype
        w2_dtype = self.w2_weight.dtype
        self.w13_weight = torch.nn.Parameter(
            torch.empty((0, ) + w13_tail_shape, device="cpu", dtype=w13_dtype),
            requires_grad=False)
        self.w2_weight = torch.nn.Parameter(
            torch.empty((0, ) + w2_tail_shape, device="cpu", dtype=w2_dtype),
            requires_grad=False)
        # When runtime weights still alias the loaded parameters, clear those
        # references so the old NPU expert tensors can actually be released.
        if (self.runtime_w13_buffer is None
                and self.runtime_w13_weight is old_w13_weight):
            self.runtime_w13_weight = None
        if (self.runtime_w2_buffer is None
                and self.runtime_w2_weight is old_w2_weight):
            self.runtime_w2_weight = None
        self.lossless_loaded_offloaded = True
        gc.collect()
        if torch.npu.is_available():
            torch.npu.empty_cache()
        if torch.npu.is_available():
            torch.npu.empty_cache()

    def refresh_lossless_runtime_weights(
            self,
            source_local_ids: Optional[list[int]] = None,
            cpu_expert_weights: Optional[dict[int, tuple[torch.Tensor,
                                                         torch.Tensor]]] = None,
            preserve_prefix_len: int = 0,
    ) -> None:
        if self.elastic_moe_mode != "lossless":
            self.runtime_w13_weight = None
            self.runtime_w2_weight = None
            self.runtime_w13_buffer = None
            self.runtime_w2_buffer = None
            self.runtime_weight_capacity = 0
            return

        active_local_num_experts = int(self.active_local_num_experts)
        if source_local_ids is None:
            source_local_ids = list(range(active_local_num_experts))
        if len(source_local_ids) != active_local_num_experts:
            raise RuntimeError(
                "Lossless runtime weights got mismatched local slots: "
                f"expected {active_local_num_experts}, got {len(source_local_ids)}."
            )
        preserve_prefix_len = min(max(int(preserve_prefix_len), 0),
                                  active_local_num_experts)

        if active_local_num_experts == 0:
            self.runtime_w13_weight = (self.runtime_w13_buffer[:0]
                                       if self.runtime_w13_buffer is not None
                                       else self.w13_weight[:0])
            self.runtime_w2_weight = (self.runtime_w2_buffer[:0]
                                      if self.runtime_w2_buffer is not None
                                      else self.w2_weight[:0])
            return

        if getattr(self, "global_redundant_expert_num", 0) <= 0:
            if (not self.lossless_loaded_offloaded and cpu_expert_weights is None
                    and active_local_num_experts == int(self.w13_weight.shape[0])
                    and all(source_local_id == slot_idx
                            for slot_idx, source_local_id in enumerate(source_local_ids))):
                self.runtime_w13_weight = self.w13_weight
                self.runtime_w2_weight = self.w2_weight
                self.runtime_w13_buffer = None
                self.runtime_w2_buffer = None
                self.runtime_weight_capacity = 0
                return

            buffer_device = self.expert_map.device
            w13_tail_shape = tuple(self.w13_weight.shape[1:])
            w2_tail_shape = tuple(self.w2_weight.shape[1:])
            buffer_w13_dtype = (self.lossless_cpu_w13_weight.dtype
                                if self.lossless_loaded_offloaded and
                                self.lossless_cpu_w13_weight is not None else
                                self.w13_weight.dtype)
            buffer_w2_dtype = (self.lossless_cpu_w2_weight.dtype
                               if self.lossless_loaded_offloaded and
                               self.lossless_cpu_w2_weight is not None else
                               self.w2_weight.dtype)
            if self.lossless_loaded_offloaded:
                self.ensure_lossless_cpu_shadow()
                local_w13_cpu_shadow = None
                local_w2_cpu_shadow = None
            else:
                old_w13_weight = self.w13_weight
                old_w2_weight = self.w2_weight
                local_w13_cpu_shadow = self.w13_weight.detach().to(device="cpu",
                                                                   copy=True)
                local_w2_cpu_shadow = self.w2_weight.detach().to(device="cpu",
                                                                 copy=True)
                self.w13_weight = torch.nn.Parameter(
                    torch.empty((0, ) + w13_tail_shape,
                                device=buffer_device,
                                dtype=buffer_w13_dtype),
                    requires_grad=False)
                self.w2_weight = torch.nn.Parameter(
                    torch.empty((0, ) + w2_tail_shape,
                                device=buffer_device,
                                dtype=buffer_w2_dtype),
                    requires_grad=False)
                if (self.runtime_w13_buffer is None
                        and self.runtime_w13_weight is old_w13_weight):
                    self.runtime_w13_weight = None
                if (self.runtime_w2_buffer is None
                        and self.runtime_w2_weight is old_w2_weight):
                    self.runtime_w2_weight = None
                gc.collect()
                if torch.npu.is_available():
                    torch.npu.synchronize()
                    torch.npu.empty_cache()

            inplace_w13 = torch.empty((active_local_num_experts, ) +
                                      w13_tail_shape,
                                      device=buffer_device,
                                      dtype=buffer_w13_dtype)

            cpu_import_ids = list(self.lossless_cpu_import_expert_ids)
            cpu_import_cursor = 0
            for slot_idx, source_local_id in enumerate(source_local_ids):
                if source_local_id >= 0:
                    if self.lossless_loaded_offloaded:
                        source_w13 = self.lossless_cpu_w13_weight[source_local_id]
                        inplace_w13[slot_idx].copy_(source_w13,
                                                    non_blocking=False)
                    else:
                        inplace_w13[slot_idx].copy_(
                            local_w13_cpu_shadow[source_local_id],
                            non_blocking=False)
                else:
                    if cpu_expert_weights is None:
                        raise RuntimeError(
                            "Lossless runtime weights need cpu_expert_weights for "
                            "negative source_local_ids.")
                    expert_id = cpu_import_ids[cpu_import_cursor]
                    cpu_import_cursor += 1
                    cpu_w13, _ = cpu_expert_weights[expert_id]
                    inplace_w13[slot_idx].copy_(cpu_w13, non_blocking=False)

            self.w13_weight = torch.nn.Parameter(inplace_w13, requires_grad=False)
            gc.collect()
            if torch.npu.is_available():
                torch.npu.empty_cache()

            inplace_w2 = torch.empty(
                (active_local_num_experts, ) + w2_tail_shape,
                device=buffer_device,
                dtype=buffer_w2_dtype)
            cpu_import_cursor = 0
            for slot_idx, source_local_id in enumerate(source_local_ids):
                if source_local_id >= 0:
                    if self.lossless_loaded_offloaded:
                        source_w2 = self.lossless_cpu_w2_weight[source_local_id]
                        inplace_w2[slot_idx].copy_(source_w2,
                                                   non_blocking=False)
                    else:
                        inplace_w2[slot_idx].copy_(
                            local_w2_cpu_shadow[source_local_id],
                            non_blocking=False)
                else:
                    if cpu_expert_weights is None:
                        raise RuntimeError(
                            "Lossless runtime weights need cpu_expert_weights for "
                            "negative source_local_ids.")
                    expert_id = cpu_import_ids[cpu_import_cursor]
                    cpu_import_cursor += 1
                    _, cpu_w2 = cpu_expert_weights[expert_id]
                    inplace_w2[slot_idx].copy_(cpu_w2, non_blocking=False)
            self.w2_weight = torch.nn.Parameter(inplace_w2, requires_grad=False)
            self.runtime_w13_weight = self.w13_weight
            self.runtime_w2_weight = self.w2_weight
            self.runtime_w13_buffer = None
            self.runtime_w2_buffer = None
            self.runtime_weight_capacity = 0
            return

        target_capacity = self._get_lossless_runtime_capacity(
            active_local_num_experts)
        if (self.runtime_w13_buffer is None or self.runtime_w2_buffer is None
                or self.runtime_weight_capacity < target_capacity):
            old_runtime_w13 = self.runtime_w13_weight
            old_runtime_w2 = self.runtime_w2_weight
            buffer_device = self.expert_map.device
            buffer_w13_dtype = (self.lossless_cpu_w13_weight.dtype
                                if self.lossless_loaded_offloaded and
                                self.lossless_cpu_w13_weight is not None else
                                self.w13_weight.dtype)
            buffer_w2_dtype = (self.lossless_cpu_w2_weight.dtype
                               if self.lossless_loaded_offloaded and
                               self.lossless_cpu_w2_weight is not None else
                               self.w2_weight.dtype)
            self.runtime_w13_buffer = torch.empty(
                (target_capacity, ) + tuple(self.w13_weight.shape[1:]),
                device=buffer_device,
                dtype=buffer_w13_dtype)
            self.runtime_w2_buffer = torch.empty(
                (target_capacity, ) + tuple(self.w2_weight.shape[1:]),
                device=buffer_device,
                dtype=buffer_w2_dtype)
            self.runtime_weight_capacity = target_capacity
            if preserve_prefix_len > 0 and old_runtime_w13 is not None and old_runtime_w2 is not None:
                self.runtime_w13_buffer[:preserve_prefix_len].copy_(
                    old_runtime_w13[:preserve_prefix_len], non_blocking=False)
                self.runtime_w2_buffer[:preserve_prefix_len].copy_(
                    old_runtime_w2[:preserve_prefix_len], non_blocking=False)
        elif preserve_prefix_len == 0:
            # Reusing existing buffers without any preserved prefix.
            self.runtime_w13_buffer[:active_local_num_experts].zero_()
            self.runtime_w2_buffer[:active_local_num_experts].zero_()

        cpu_import_ids = list(self.lossless_cpu_import_expert_ids)
        cpu_import_cursor = 0
        if self.lossless_loaded_offloaded:
            self.ensure_lossless_cpu_shadow()

        for slot_idx in range(preserve_prefix_len, active_local_num_experts):
            source_local_id = source_local_ids[slot_idx]
            if source_local_id >= 0:
                if self.lossless_loaded_offloaded:
                    source_w13 = self.lossless_cpu_w13_weight[source_local_id]
                    source_w2 = self.lossless_cpu_w2_weight[source_local_id]
                    self.runtime_w13_buffer[slot_idx].copy_(
                        source_w13.to(device=self.runtime_w13_buffer.device),
                        non_blocking=False)
                    self.runtime_w2_buffer[slot_idx].copy_(
                        source_w2.to(device=self.runtime_w2_buffer.device),
                        non_blocking=False)
                else:
                    self.runtime_w13_buffer[slot_idx].copy_(
                        self.w13_weight[source_local_id], non_blocking=False)
                    self.runtime_w2_buffer[slot_idx].copy_(
                        self.w2_weight[source_local_id], non_blocking=False)
            else:
                if cpu_expert_weights is None:
                    raise RuntimeError(
                        "Lossless runtime weights need cpu_expert_weights for "
                        "negative source_local_ids.")
                expert_id = cpu_import_ids[cpu_import_cursor]
                cpu_import_cursor += 1
                cpu_w13, cpu_w2 = cpu_expert_weights[expert_id]
                self.runtime_w13_buffer[slot_idx].copy_(
                    cpu_w13.to(device=self.runtime_w13_buffer.device),
                    non_blocking=False)
                self.runtime_w2_buffer[slot_idx].copy_(
                    cpu_w2.to(device=self.runtime_w2_buffer.device),
                    non_blocking=False)

        self.runtime_w13_weight = self.runtime_w13_buffer[:active_local_num_experts]
        self.runtime_w2_weight = self.runtime_w2_buffer[:active_local_num_experts]

    def _get_lossless_runtime_capacity(self,
                                       active_local_num_experts: int) -> int:
        capacity = int(active_local_num_experts)
        if getattr(self, "global_redundant_expert_num", 0) <= 0:
            # In zero-redundancy mode, pre-reserving the next shrink stage
            # makes the first 16->8 activation peak too expensive.
            return capacity
        logical_num_experts = int(getattr(self, "elastic_original_num_experts",
                                          capacity))
        current_ep_size = int(getattr(self.moe_parallel_config, "ep_size", 1))
        min_ep_size = int(envs_ascend.VLLM_ASCEND_MC2_MIN_EP_SIZE)
        if current_ep_size <= 1 or logical_num_experts <= 0:
            return capacity
        next_ep_size = current_ep_size // 2
        while next_ep_size >= min_ep_size:
            if logical_num_experts % next_ep_size == 0:
                return max(capacity, logical_num_experts // next_ep_size)
            next_ep_size //= 2
        return capacity

    def activate_lossless_primary_experts(self) -> None:
        if self.elastic_moe_mode != "lossless" or self.loaded_expert_map is None:
            return
        if getattr(self, "lossless_lazy_activation", False):
            # Keep pre-shrink execution identical to the primary layout.
            self.runtime_w13_weight = None
            self.runtime_w2_weight = None
            self.runtime_w13_buffer = None
            self.runtime_w2_buffer = None
            self.runtime_weight_capacity = 0
            self.lossless_runtime_activated = False
            return
        primary_expert_map = getattr(self, "elastic_original_expert_map", None)
        if primary_expert_map is None:
            primary_expert_map = self.expert_map
        active_expert_ids: list[int] = []
        source_local_ids: list[int] = []
        primary_map_list = primary_expert_map.detach().cpu().tolist()
        loaded_map_list = self.loaded_expert_map.detach().cpu().tolist()
        for expert_id, primary_local_id in enumerate(primary_map_list):
            if primary_local_id < 0:
                continue
            loaded_local_id = loaded_map_list[expert_id]
            if loaded_local_id < 0:
                raise RuntimeError(
                    f"Lossless elastic init missing loaded replica for expert {expert_id}"
                )
            active_expert_ids.append(expert_id)
            source_local_ids.append(int(loaded_local_id))
        self.activate_lossless_local_experts(active_expert_ids,
                                             source_local_ids)

    def export_lossless_expert_cpu_weights(
            self, expert_ids: list[int]) -> dict[int, tuple[torch.Tensor,
                                                            torch.Tensor]]:
        if self.elastic_moe_mode != "lossless" or not expert_ids:
            return {}
        self.ensure_lossless_cpu_shadow()
        export_map = self.loaded_expert_map if self.loaded_expert_map is not None else self.expert_map
        cpu_weights: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        for expert_id in expert_ids:
            local_slot = int(export_map[expert_id].item())
            if local_slot < 0:
                continue
            cpu_weights[int(expert_id)] = (
                self.lossless_cpu_w13_weight[local_slot].detach().cpu(),
                self.lossless_cpu_w2_weight[local_slot].detach().cpu(),
            )
        return cpu_weights

    def export_lossless_expert_npu_weights(
            self, expert_ids: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
        if self.elastic_moe_mode != "lossless" or not expert_ids:
            empty_w13 = self.w13_weight[:0]
            empty_w2 = self.w2_weight[:0]
            return empty_w13, empty_w2
        if self.lossless_loaded_offloaded:
            raise RuntimeError(
                "Lossless NPU export requires loaded expert weights to still "
                f"reside on device at layer={self.layer_idx}.")
        export_map = (self.loaded_expert_map
                      if self.loaded_expert_map is not None else self.expert_map)
        local_slots = [int(export_map[expert_id].item()) for expert_id in expert_ids]
        export_index = torch.tensor(local_slots,
                                    device=self.w13_weight.device,
                                    dtype=torch.long)
        return (self.w13_weight.index_select(0, export_index),
                self.w2_weight.index_select(0, export_index))

    def activate_lossless_local_experts(self, active_expert_ids: list[int],
                                        source_local_ids: list[int],
                                        cpu_expert_weights: Optional[dict[int, tuple[
                                            torch.Tensor, torch.Tensor]]] = None,
                                        offload_loaded_after_activation: bool = False) -> None:
        if self.elastic_moe_mode != "lossless":
            return
        new_local_num_experts = len(active_expert_ids)
        previous_expert_map = (self.expert_map.clone()
                               if self.expert_map is not None else None)
        new_expert_map = torch.full((self.elastic_original_num_experts, ),
                                    -1,
                                    dtype=torch.int32,
                                    device=self.expert_map.device)
        self.lossless_runtime_activated = True
        for local_slot, expert_id in enumerate(active_expert_ids):
            new_expert_map[expert_id] = local_slot
        self.expert_map = new_expert_map
        self.active_local_num_experts = new_local_num_experts
        self.local_num_experts = new_local_num_experts
        self.moe_config.num_local_experts = new_local_num_experts
        preserve_prefix_len = 0
        if previous_expert_map is not None:
            current_slots = previous_expert_map.detach().cpu().tolist()
            for local_slot, expert_id in enumerate(active_expert_ids):
                if expert_id >= len(current_slots) or int(current_slots[expert_id]) != local_slot:
                    break
                preserve_prefix_len += 1
        self.lossless_cpu_import_expert_ids = [
            int(expert_id) for expert_id, source_local_id in zip(
                active_expert_ids, source_local_ids) if source_local_id < 0
        ]
        if offload_loaded_after_activation and not self.lossless_loaded_offloaded:
            self.offload_lossless_loaded_weights_to_cpu()
        self.refresh_lossless_runtime_weights(
            source_local_ids,
            cpu_expert_weights=cpu_expert_weights,
            preserve_prefix_len=preserve_prefix_len)
        if offload_loaded_after_activation and not self.lossless_loaded_offloaded:
            self.offload_lossless_loaded_weights_to_cpu()
        self.elastic_debug_tag += 1
        self.elastic_debug_budget = 1 if self.layer_idx == 0 else 0
        self.elastic_debug_reason = ("lossless_post_shrink"
                                     if self.layer_idx == 0 else None)

    def set_active_expert_mask(self,
                               new_active_expert_mask: Optional[torch.Tensor]):
        if new_active_expert_mask is None:
            self.active_expert_mask = None
            self.elastic_runtime_log2phy = None
            return
        self.active_expert_mask = new_active_expert_mask.to(
            device=self.expert_map.device if self.expert_map is not None else
            new_active_expert_mask.device,
            dtype=torch.bool)

    def set_elastic_runtime_log2phy(self,
                                    new_log2phy: Optional[torch.Tensor]) -> None:
        if new_log2phy is None:
            self.elastic_runtime_log2phy = None
            return
        self.elastic_runtime_log2phy = new_log2phy.to(
            device=self.log2phy.device if self.log2phy is not None else
            new_log2phy.device,
            dtype=self.log2phy.dtype if self.log2phy is not None else
            new_log2phy.dtype)

    def refresh_elastic_groups(self):
        self.ep_group = get_ep_group()
        self.moe_parallel_config.dp_size = get_dp_group().world_size
        self.moe_parallel_config.dp_rank = get_dp_group().rank_in_group
        self.moe_parallel_config.ep_size = self.ep_group.world_size
        self.moe_parallel_config.ep_rank = self.ep_group.rank_in_group
        self.moe_config.moe_parallel_config = self.moe_parallel_config
        self.moe_config.model_type = self.model_type
        self.moe_config.dp_group = get_dp_group()
        self.moe_config.ep_group = self.ep_group
        self.moe_config.mc2_group = get_mc2_group()
        is_deepseek_v3_r1 = self.num_experts == 256
        self.rm_router_logits = get_rm_router_logits_state(
            self.moe_parallel_config.ep_size, self.dp_size, is_deepseek_v3_r1)
        self.all_reduce_merge = get_all_reduce_merge_state(
            self.moe_parallel_config.ep_size, is_deepseek_v3_r1)
        setup_moe_comm_method(self.moe_config)
        if self.elastic_moe_mode == "lossless" and (
                self.runtime_w13_weight is None or self.runtime_w2_weight is None):
            self.activate_lossless_primary_experts()

    def get_map(self):
        return self.expert_map

    def get_log2phy_map(self):
        return self.log2phy

    def clear_moe_load(self):
        if self.moe_load is not None:
            self.moe_load.zero_()

    def forward(self,
                hidden_states: torch.Tensor,
                router_logits: torch.Tensor,
                is_prefill: bool,
                enable_force_load_balance: bool = False,
                top_k: Optional[int] = None,
                shared_experts: Optional[Any] = None,
                gate=None,
                replace_allreduce: bool = False,
                is_dummy: bool = False):

        assert self.quant_method is not None

        if top_k:
            real_top_k = top_k
        else:
            real_top_k = self.top_k

        forward_context = get_forward_context()
        mc2_mask = forward_context.mc2_mask
        debug_enabled = bool(self.elastic_debug_budget > 0 and not is_dummy)
        if debug_enabled:
            local_expert_ids = []
            if self.expert_map is not None:
                local_expert_ids = [
                    expert_id for expert_id, local_slot in enumerate(
                        self.expert_map.detach().cpu().tolist())
                    if int(local_slot) >= 0
                ]
            runtime_log2phy = (self.elastic_runtime_log2phy
                               if self.elastic_runtime_log2phy is not None else
                               self.log2phy)
            log2phy_min = (int(runtime_log2phy.min().item())
                           if runtime_log2phy is not None
                           and runtime_log2phy.numel() > 0 else -1)
            log2phy_max = (int(runtime_log2phy.max().item())
                           if runtime_log2phy is not None
                           and runtime_log2phy.numel() > 0 else -1)
            active_mask_count = (int(self.active_expert_mask.sum().item())
                                 if self.active_expert_mask is not None else -1)
            debug_info = {
                "enabled": True,
                "tag": self.elastic_debug_tag,
                "reason": self.elastic_debug_reason,
                "layer_idx": self.layer_idx,
                "ep_rank": self.ep_rank,
                "num_local_experts": self.local_num_experts,
                "active_local_num_experts": self.active_local_num_experts,
                "runtime_num_experts": self.moe_config.num_experts,
                "active_mask_count": active_mask_count,
                "log2phy_min": log2phy_min,
                "log2phy_max": log2phy_max,
            }
            setattr(forward_context, "elastic_debug_info", debug_info)
            logger.info(
                "Elastic forward state: rank=%s layer=%s tag=%s reason=%s num_local_experts=%s active_local_num_experts=%s runtime_num_experts=%s active_mask_count=%s local_expert_ids_head=%s local_expert_ids_tail=%s log2phy_min=%s log2phy_max=%s",
                self.ep_rank,
                self.layer_idx,
                self.elastic_debug_tag,
                self.elastic_debug_reason,
                self.local_num_experts,
                self.active_local_num_experts,
                self.moe_config.num_experts,
                active_mask_count,
                local_expert_ids[:8],
                local_expert_ids[-8:],
                log2phy_min,
                log2phy_max)
        else:
            setattr(forward_context, "elastic_debug_info", None)
        # For w8a8 dynamic we can do npu_dynamic_quant and gate in parallel.
        quantized_x_for_share, dynamic_scale_for_share = None, None

        if shared_experts:
            # When all_reduce_merge is in progress, shared_experts does not do all_reduce in mlp, but waits until shared_experts+router_experts are completed before doing all_reduce
            shared_hidden_states = shared_experts(hidden_states)

        if forward_context.sp_enabled:
            replace_allreduce = True

        hidden_states, router_logits = forward_context.moe_comm_method.prepare(
            hidden_states=hidden_states,
            router_logits=router_logits,
            enable_shared_expert_dp=self.enable_shared_expert_dp,
            rm_router_logits=self.rm_router_logits,
            replace_allreduce=replace_allreduce,
            gate=gate)
        
        # print("is_dummy in ascendfusedmoe is:", is_dummy)
        # Matrix multiply.
        e_hidden_states = self.quant_method.apply(
            layer=self,
            x=hidden_states,
            router_logits=router_logits,
            top_k=real_top_k,
            renormalize=self.renormalize,
            use_grouped_topk=self.use_grouped_topk,
            global_num_experts=self.global_num_experts,
            expert_map=self.expert_map,
            topk_group=self.topk_group,
            num_expert_group=self.num_expert_group,
            custom_routing_function=self.custom_routing_function,
            scoring_func=self.scoring_func,
            e_score_correction_bias=self.e_score_correction_bias,
            is_prefill=is_prefill,
            enable_force_load_balance=enable_force_load_balance,
            log2phy=self.elastic_runtime_log2phy
            if self.elastic_runtime_log2phy is not None else self.log2phy,
            global_redundant_expert_num=self.global_redundant_expert_num,
            shared_experts=None,
            mc2_mask=mc2_mask,
            quantized_x_for_share=quantized_x_for_share,
            dynamic_scale_for_share=dynamic_scale_for_share,
            layer_idx=self.layer_idx,
            is_dummy=is_dummy,
            active_expert_mask=self.active_expert_mask,
        )

        group_list_type = None

        if shared_experts:
            if isinstance(e_hidden_states,
                          tuple) and len(e_hidden_states) == 2:
                e_hidden_states, shared_hidden_states = e_hidden_states

        if isinstance(e_hidden_states, tuple) and len(e_hidden_states) == 3:
            e_hidden_states, group_list_type, expert_tokens = e_hidden_states

        if self.dynamic_eplb and group_list_type is not None:
            self.moe_load += expert_tokens if group_list_type else \
                torch.cat([expert_tokens[:1], expert_tokens[1:] - expert_tokens[:-1]])

        final_hidden_states = forward_context.moe_comm_method.finalize(
            hidden_states=e_hidden_states,
            reduce_results=(not self.all_reduce_merge))
        if debug_enabled:
            self.elastic_debug_budget = max(0, self.elastic_debug_budget - 1)

        if shared_experts:
            return final_hidden_states, shared_hidden_states
        else:
            return final_hidden_states

    def reset_expert_map_and_log2phy(self):
        if self.elastic_moe_mode == "lossless":
            self.loaded_local_num_experts, self.loaded_expert_map = determine_redundant_replica_expert_map(
                self.num_experts, self.ep_size, self.ep_rank,
                self.global_redundant_expert_num)
            self.active_local_num_experts, self.expert_map, self.log2phy = determine_expert_map(
                self.ep_size, self.ep_rank, self.num_experts, layer_idx=self.layer_idx)
            self.primary_log2phy = self.log2phy.clone()
            self.local_num_experts = self.active_local_num_experts
            self.moe_config.num_local_experts = self.active_local_num_experts
            self.elastic_runtime_log2phy = None
            self.refresh_lossless_runtime_weights()
            return
        else:
            _, expert_map = determine_default_expert_map(
                self.global_num_experts, self.ep_size, self.ep_rank,
                self.global_redundant_expert_num)
            log2phy = determine_default_log2phy_map(
                self.global_num_experts, self.ep_size, self.ep_rank,
                self.global_redundant_expert_num).npu()

        self.expert_map.copy_(expert_map)
        self.log2phy.copy_(log2phy)
        self.elastic_runtime_log2phy = None

    # ----------------------------------------- TBO-related --------------------------------------------

    def _forward_ms_fused_moe_comp(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        is_prefill: bool,
        real_top_k,
        enable_force_load_balance: bool = False,
    ):
        hidden_states = self.quant_method.apply(
            layer=self,
            x=hidden_states,
            router_logits=router_logits,
            top_k=real_top_k,
            renormalize=self.renormalize,
            use_grouped_topk=self.use_grouped_topk,
            global_num_experts=self.global_num_experts,
            expert_map=self.expert_map,
            topk_group=self.topk_group,
            num_expert_group=self.num_expert_group,
            custom_routing_function=self.custom_routing_function,
            scoring_func=self.scoring_func,
            e_score_correction_bias=self.e_score_correction_bias,
            is_prefill=is_prefill,
            enable_force_load_balance=enable_force_load_balance,
            log2phy=self.elastic_runtime_log2phy
            if self.elastic_runtime_log2phy is not None else self.log2phy,
            global_redundant_expert_num=self.global_redundant_expert_num,
            active_expert_mask=self.active_expert_mask,
            layer_idx=self.layer_idx,
            )

        return hidden_states
