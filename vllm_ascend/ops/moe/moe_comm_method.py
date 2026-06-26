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
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import torch
import os
from vllm.config import get_current_vllm_config
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.fused_moe import FusedMoEConfig
from vllm.model_executor.layers.fused_moe.config import FusedMoEParallelConfig

from vllm_ascend.ascend_forward_context import MoECommType
from vllm_ascend.ops.moe.fused_moe_prepare_and_finalize import (
    FusedMoEPrepareAndFinalizeWithAll2All,
    FusedMoEPrepareAndFinalizeWithAllGather, FusedMoEPrepareAndFinalizeWithMC2,
    FusedMoEPrepareAndFinalizeWithNaiveMulticast)
from vllm_ascend.ops.moe.moe_mlp import unified_apply_mlp
from vllm_ascend.ops.moe.token_dispatcher import (TokenDispatcherWithAll2AllV,
                                                  TokenDispatcherWithAllGather,
                                                  TokenDispatcherWithMC2,
                                                  TokenDispatcherWithMoge)

_MoECommMethods: Dict[Optional[MoECommType], MoECommMethod] = {}
_MoECommMethodTopologyCache: Dict[
    tuple[Any, ...], Dict[Optional[MoECommType], MoECommMethod]] = {}
_MoECommMethodActiveKey: Optional[tuple[Any, ...]] = None

_DISPATCHER_TRANSIENT_ATTRS = (
    # MC2 dispatcher transient state.
    "output",
    "assist_info_for_combine",
    "ep_recv_counts",
    "shared_act",
    "shared_experts",
    "swiglu_out_scale",
    "topk_ids",
    "topk_weights",
    "mc2_mask",
    "expert_map",
    # AllGather / Moge transient state.
    "sorted_weights",
    "expanded_row_idx",
    "sorted_token_indices",
    "original_shape",
    "mask",
    "sorted_topk_ids",
    "bsz",
    # AllToAll-V transient state.
    "hidden_shape",
    "input_splits",
    "output_splits",
    "hidden_shape_before_permute",
    "num_global_tokens_per_local_expert",
    "tokens_per_expert",
    "global_input_tokens_local_experts_indices",
    "reversed_local_input_permutation_mapping",
    "reversed_global_input_permutation_mapping",
)

_DISPATCHER_PERSISTENT_TENSOR_ATTRS = {
    # Created once from the static expert layout. Clearing this would make the
    # next AllToAll-V dispatch pay the setup cost again and can break reuse.
    "expert_ids_per_ep_rank",
}

_PREPARE_FINALIZE_TRANSIENT_ATTRS = (
    "split_hidden_states",
    "cu_tokens_across_dp_cpu",
)


def get_moe_comm_method(
        moe_comm_type: Optional[MoECommType]) -> Optional[MoECommMethod]:
    return _MoECommMethods.get(moe_comm_type)


def _env_flag(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).strip().lower() in {
        "1", "true", "yes", "on"
    }


def _preserve_topology_cache() -> bool:
    return (
        _env_flag("VLLM_ASCEND_MOE_COMM_METHOD_TOPOLOGY_CACHE", "0")
        or _env_flag("VLLM_ASCEND_MODE1_PARITY_CACHE_PLANNED_FLOOR_GROUPS",
                     "0")
        or _env_flag("VLLM_ASCEND_MODE1_PARITY_PRECREATE_PLANNED_FLOOR_GROUPS",
                     "0"))


def _group_key(group: Any) -> tuple[Any, ...]:
    if group is None:
        return ()
    ranks = getattr(group, "ranks", None)
    if ranks is not None:
        try:
            return tuple(int(rank) for rank in ranks)
        except TypeError:
            pass
    return (
        "size",
        int(getattr(group, "world_size", 0) or 0),
        "rank",
        int(getattr(group, "rank_in_group", -1) or -1),
    )


def _moe_comm_topology_key(moe_config: FusedMoEConfig) -> tuple[Any, ...]:
    return (
        _resolve_model_type(moe_config),
        int(getattr(moe_config, "experts_per_token", 0) or 0),
        int(getattr(moe_config, "num_experts", 0) or 0),
        int(getattr(moe_config, "num_local_experts", 0) or 0),
        int(getattr(moe_config, "num_global_redundant_experts", 0) or 0),
        _group_key(getattr(moe_config, "dp_group", None)),
        _group_key(getattr(moe_config, "ep_group", None)),
        _group_key(getattr(moe_config, "mc2_group", None)),
    )


def _clone_moe_comm_config(moe_config: FusedMoEConfig) -> FusedMoEConfig:
    """Snapshot mutable MoE comm metadata for a cached topology.

    FusedMoE modules mutate ``moe_config`` in-place when switching between
    full-world and shrink floors. A cached dispatcher must not keep pointing at
    that mutable module object; otherwise the floor-8/floor-4 cache entry
    silently turns back into the latest full-world config.
    """
    parallel_config = moe_config.moe_parallel_config
    cloned_parallel_config = FusedMoEParallelConfig(
        tp_size=int(parallel_config.tp_size),
        tp_rank=int(parallel_config.tp_rank),
        dp_size=int(parallel_config.dp_size),
        dp_rank=int(parallel_config.dp_rank),
        ep_size=int(parallel_config.ep_size),
        ep_rank=int(parallel_config.ep_rank),
        use_ep=bool(parallel_config.use_ep),
    )
    cloned_config = FusedMoEConfig(
        num_experts=int(moe_config.num_experts),
        experts_per_token=int(moe_config.experts_per_token),
        hidden_dim=int(moe_config.hidden_dim),
        num_local_experts=int(moe_config.num_local_experts),
        moe_parallel_config=cloned_parallel_config,
        in_dtype=moe_config.in_dtype,
        max_num_tokens=int(moe_config.max_num_tokens),
        has_bias=bool(getattr(moe_config, "has_bias", False)),
    )
    for attr_name in (
            "model_type",
            "tp_group",
            "dp_group",
            "ep_group",
            "mc2_group",
            "num_global_redundant_experts",
    ):
        if hasattr(moe_config, attr_name):
            setattr(cloned_config, attr_name, getattr(moe_config, attr_name))
    return cloned_config


def reset_moe_comm_method_cache() -> None:
    global _MoECommMethodActiveKey
    _MoECommMethods.clear()
    _MoECommMethodActiveKey = None
    if not _preserve_topology_cache():
        _MoECommMethodTopologyCache.clear()


def prune_moe_comm_method_topology_cache(
        allowed_group_ranks: set[tuple[int, ...]]) -> dict[str, int]:
    """Drop cached MoE communicators whose topology is no longer planned.

    Planned mode may intentionally keep floor8/floor4 communicator workspaces
    resident across steps. When the next step no longer needs a floor, this
    pruning releases the corresponding topology cache before KV is re-sized.
    """
    global _MoECommMethodActiveKey
    allowed = {tuple(int(rank) for rank in ranks)
               for ranks in allowed_group_ranks if ranks}
    stats = {
        "kept_topologies": 0,
        "dropped_topologies": 0,
        "dropped_methods": 0,
        "active_cleared": 0,
    }

    def _component_allowed(component: Any) -> bool:
        if not isinstance(component, tuple):
            return True
        if not component:
            return True
        if len(component) >= 2 and component[0] == "size":
            try:
                group_size = int(component[1])
            except (TypeError, ValueError):
                return True
            return any(len(ranks) == group_size for ranks in allowed)
        if not all(isinstance(item, int) for item in component):
            return True
        return tuple(int(item) for item in component) in allowed

    def _topology_allowed(topology_key: tuple[Any, ...]) -> bool:
        # The final three entries are dp/ep/mc2 group keys.
        return all(_component_allowed(component)
                   for component in topology_key[-3:])

    dropped_keys = [
        key for key in list(_MoECommMethodTopologyCache.keys())
        if not _topology_allowed(key)
    ]
    stats["kept_topologies"] = (
        len(_MoECommMethodTopologyCache) - len(dropped_keys))
    for key in dropped_keys:
        cached_methods = _MoECommMethodTopologyCache.pop(key, {})
        stats["dropped_topologies"] += 1
        stats["dropped_methods"] += len(cached_methods)
        if _MoECommMethodActiveKey == key:
            _MoECommMethods.clear()
            _MoECommMethodActiveKey = None
            stats["active_cleared"] = 1
        for method in cached_methods.values():
            for attr_name in ("moe_config", "token_dispatcher",
                              "fused_moe_prepare_finalize", "mc2_mask"):
                if hasattr(method, attr_name):
                    try:
                        setattr(method, attr_name, None)
                    except Exception:
                        pass
    return stats


def _tensor_bytes(value: Any, seen: Optional[set[int]] = None) -> int:
    if seen is None:
        seen = set()
    value_id = id(value)
    if value_id in seen:
        return 0
    seen.add(value_id)
    if torch.is_tensor(value):
        try:
            return int(value.numel()) * int(value.element_size())
        except Exception:
            return 0
    if isinstance(value, dict):
        return sum(_tensor_bytes(item, seen)
                   for pair in value.items() for item in pair)
    if isinstance(value, (list, tuple, set)):
        return sum(_tensor_bytes(item, seen) for item in value)
    return 0


def _contains_tensor(value: Any) -> bool:
    return _tensor_bytes(value) > 0


def _clear_attr(obj: Any, attr_name: str) -> tuple[int, int, int]:
    if obj is None or not hasattr(obj, attr_name):
        return (0, 0, 0)
    value = getattr(obj, attr_name)
    if value is None:
        return (0, 0, 0)
    tensor_count = 1 if torch.is_tensor(value) else int(_contains_tensor(value))
    tensor_bytes = _tensor_bytes(value)
    try:
        setattr(obj, attr_name, None)
    except Exception:
        return (0, 0, 0)
    return (1, tensor_count, tensor_bytes)


def _release_dispatcher_runtime_state(dispatcher: Any) -> dict[str, int]:
    stats = {
        "cleared_attrs": 0,
        "cleared_tensors": 0,
        "tensor_bytes": 0,
    }
    if dispatcher is None:
        return stats

    cleared_names: set[str] = set()
    for attr_name in _DISPATCHER_TRANSIENT_ATTRS:
        attrs, tensors, bytes_ = _clear_attr(dispatcher, attr_name)
        if attrs:
            cleared_names.add(attr_name)
        stats["cleared_attrs"] += attrs
        stats["cleared_tensors"] += tensors
        stats["tensor_bytes"] += bytes_

    # Conservative leak guard: if a dispatcher stores a tensor-valued scratch
    # attr added by a newer backend, release it too. Persistent layout tensors
    # are explicitly skipped above.
    try:
        dispatcher_items = list(vars(dispatcher).items())
    except TypeError:
        dispatcher_items = []
    for attr_name, value in dispatcher_items:
        if attr_name in cleared_names:
            continue
        if attr_name in _DISPATCHER_PERSISTENT_TENSOR_ATTRS:
            continue
        if not _contains_tensor(value):
            continue
        attrs, tensors, bytes_ = _clear_attr(dispatcher, attr_name)
        stats["cleared_attrs"] += attrs
        stats["cleared_tensors"] += tensors
        stats["tensor_bytes"] += bytes_

    if hasattr(dispatcher, "with_quant"):
        try:
            dispatcher.with_quant = False
        except Exception:
            pass
    return stats


def _release_prepare_finalize_runtime_state(
        prepare_finalize: Any) -> dict[str, int]:
    stats = {
        "cleared_attrs": 0,
        "cleared_tensors": 0,
        "tensor_bytes": 0,
    }
    if prepare_finalize is None:
        return stats
    cleared_names: set[str] = set()
    for attr_name in _PREPARE_FINALIZE_TRANSIENT_ATTRS:
        attrs, tensors, bytes_ = _clear_attr(prepare_finalize, attr_name)
        if attrs:
            cleared_names.add(attr_name)
        stats["cleared_attrs"] += attrs
        stats["cleared_tensors"] += tensors
        stats["tensor_bytes"] += bytes_

    try:
        prepare_items = list(vars(prepare_finalize).items())
    except TypeError:
        prepare_items = []
    for attr_name, value in prepare_items:
        if attr_name in cleared_names or attr_name == "moe_config":
            continue
        if not _contains_tensor(value):
            continue
        attrs, tensors, bytes_ = _clear_attr(prepare_finalize, attr_name)
        stats["cleared_attrs"] += attrs
        stats["cleared_tensors"] += tensors
        stats["tensor_bytes"] += bytes_
    return stats


def release_moe_comm_method_runtime_state() -> dict[str, int]:
    """Release transient tensor references without destroying comm groups.

    Planned mode intentionally keeps topology-keyed floor8/floor4 communicators
    resident across rollout steps. Those cached methods must not also keep the
    tensors produced by a warmup or a live dispatch/combine cycle. This helper
    sweeps both the active method table and the topology cache, clearing only
    dispatcher/prepare-finalize runtime state while preserving group handles and
    static layout metadata.
    """
    stats = {
        "methods": 0,
        "topologies": len(_MoECommMethodTopologyCache),
        "method_attrs": 0,
        "dispatcher_attrs": 0,
        "prepare_attrs": 0,
        "tensors": 0,
        "tensor_bytes": 0,
    }
    methods: list[Any] = []
    for method in _MoECommMethods.values():
        if method is not None:
            methods.append(method)
    for cached_methods in _MoECommMethodTopologyCache.values():
        for method in cached_methods.values():
            if method is not None:
                methods.append(method)

    seen_methods: set[int] = set()
    for method in methods:
        method_id = id(method)
        if method_id in seen_methods:
            continue
        seen_methods.add(method_id)
        stats["methods"] += 1

        attrs, tensors, bytes_ = _clear_attr(method, "mc2_mask")
        stats["method_attrs"] += attrs
        stats["tensors"] += tensors
        stats["tensor_bytes"] += bytes_

        dispatcher_stats = _release_dispatcher_runtime_state(
            getattr(method, "token_dispatcher", None))
        stats["dispatcher_attrs"] += dispatcher_stats["cleared_attrs"]
        stats["tensors"] += dispatcher_stats["cleared_tensors"]
        stats["tensor_bytes"] += dispatcher_stats["tensor_bytes"]

        prepare_stats = _release_prepare_finalize_runtime_state(
            getattr(method, "fused_moe_prepare_finalize", None))
        stats["prepare_attrs"] += prepare_stats["cleared_attrs"]
        stats["tensors"] += prepare_stats["cleared_tensors"]
        stats["tensor_bytes"] += prepare_stats["tensor_bytes"]

    return stats


def get_moe_comm_method_topology_cache_stats() -> dict[str, Any]:
    """Return a compact view of cached MoE comm topologies for diagnostics."""

    def _fmt_component(component: Any) -> str:
        if isinstance(component, tuple):
            if all(isinstance(item, int) for item in component):
                return ",".join(str(int(item)) for item in component)
            return ":".join(str(item) for item in component)
        return str(component)

    topologies = []
    for topology_key, methods in _MoECommMethodTopologyCache.items():
        topologies.append({
            "groups": [_fmt_component(component)
                       for component in topology_key[-3:]],
            "methods": len(methods),
            "active": int(topology_key == _MoECommMethodActiveKey),
        })
    return {
        "topology_count": len(_MoECommMethodTopologyCache),
        "method_count": sum(len(methods)
                            for methods in _MoECommMethodTopologyCache.values()),
        "active_present": int(_MoECommMethodActiveKey is not None),
        "topologies": topologies,
    }


def _resolve_model_type(moe_config: FusedMoEConfig) -> str:
    vllm_config = get_current_vllm_config()
    if (vllm_config is not None and vllm_config.model_config is not None
            and vllm_config.model_config.hf_config is not None):
        return vllm_config.model_config.hf_config.model_type
    return getattr(moe_config, "model_type", "")


def setup_moe_comm_method(moe_config):
    global _MoECommMethodActiveKey
    topology_key = _moe_comm_topology_key(moe_config)
    if _MoECommMethodActiveKey == topology_key and _MoECommMethods:
        return

    if not _preserve_topology_cache():
        # Keep the current refresh cheap across all MoE layers, but do not keep
        # stale floor topologies resident in natural/free-running shrink mode.
        cached_methods = _MoECommMethodTopologyCache.get(topology_key)
        _MoECommMethodTopologyCache.clear()
        if cached_methods is not None:
            _MoECommMethodTopologyCache[topology_key] = cached_methods

    cached_methods = _MoECommMethodTopologyCache.get(topology_key)
    if cached_methods is None:
        cached_config = _clone_moe_comm_config(moe_config)
        cached_methods = {
            MoECommType.ALLTOALL: AlltoAllCommImpl(cached_config),
            MoECommType.ALLGATHER: AllGatherCommImpl(cached_config),
            MoECommType.MC2: MC2CommImpl(cached_config),
            MoECommType.NAIVE_MULTICAST:
            NaiveMulticastCommImpl(cached_config),
        }
        _MoECommMethodTopologyCache[topology_key] = cached_methods

    _MoECommMethods.clear()
    _MoECommMethods.update(cached_methods)
    _MoECommMethodActiveKey = topology_key


class MoECommMethod(ABC):
    """Base class for MoE communication methods."""

    def __init__(self, moe_config: FusedMoEConfig):
        self.model_type = _resolve_model_type(moe_config)
        self.moe_config = moe_config
        self.mc2_mask = None

        self.token_dispatcher = self._get_token_dispatcher()
        self.fused_moe_prepare_finalize = self._get_fused_moe_prepare_finalize(
        )

    def prepare(self,
                hidden_states: torch.Tensor,
                router_logits: torch.Tensor,
                enable_shared_expert_dp: bool = False,
                rm_router_logits: bool = False,
                replace_allreduce: bool = False,
                gate=None) -> tuple[torch.Tensor, torch.Tensor]:
        hidden_states, router_logits, mc2_mask = self.fused_moe_prepare_finalize.prepare(
            hidden_states, router_logits, enable_shared_expert_dp,
            rm_router_logits, replace_allreduce, gate)
        self.mc2_mask = mc2_mask
        return hidden_states, router_logits

    def finalize(self, hidden_states: torch.Tensor,
                 reduce_results: bool) -> torch.Tensor:
        hidden_states = self.fused_moe_prepare_finalize.finalize(
            hidden_states, reduce_results)
        return hidden_states

    def fused_experts(
            self,
            hidden_states: torch.Tensor,
            w1: torch.Tensor,
            w2: torch.Tensor,
            topk_weights: torch.Tensor,
            topk_ids: torch.Tensor,
            row_idx: torch.Tensor,
            activation: str = "silu",
            apply_router_weight_on_input: bool = False,
            use_int8_w8a8: bool = False,
            use_int4_w4a8: bool = False,
            global_num_experts: Optional[int] = None,
            expert_map: Optional[torch.Tensor] = None,
            w1_scale: Optional[torch.Tensor] = None,
            w2_scale: Optional[torch.Tensor] = None,
            w1_scale_bias: torch.Tensor = None,
            w2_scale_bias: torch.Tensor = None,
            # For TorchAir graph
            is_torchair: bool = False,
            # For Cube/Vector parallel
            shared_experts: Optional[Any] = None,
            quantized_x_for_share: Optional[Any] = None,
            dynamic_scale_for_share: Optional[Any] = None,
            # For load balance
            log2phy: torch.Tensor = None,
            global_redundant_expert_num: int = 0,
            need_trans: bool = False,
            dynamic_eplb: bool = False,
            mc2_mask: Optional[torch.Tensor] = None):
        # Check constraints
        assert hidden_states.dtype in [
            torch.float32, torch.float16, torch.bfloat16
        ]

        moe_comm_method = get_forward_context().moe_comm_method
        assert moe_comm_method is not None, "Missing communication context"

        final_hidden_states = torch.zeros_like(hidden_states)
        chunk_start_index = 0
        chunk_moe_size = int(os.environ.get('VLLM_CHUNK_MOE_SIZE', 512))
        ctx = get_forward_context()

        from vllm.distributed import get_tensor_model_parallel_world_size
        tp_size = get_tensor_model_parallel_world_size()
        max_tokens = (ctx.max_tokens_across_dp + tp_size - 1) // tp_size
        num_tokens = hidden_states.size(0)
        effective_mc2_mask = self.mc2_mask if mc2_mask is None else mc2_mask
        if (effective_mc2_mask is not None
                and effective_mc2_mask.shape[0] != hidden_states.shape[0]):
            raise RuntimeError(
                "MC2 mask length mismatch before token dispatch: "
                f"hidden_tokens={hidden_states.shape[0]} "
                f"mask_tokens={effective_mc2_mask.shape[0]}")

        if max_tokens < chunk_moe_size:
            results = self.token_dispatcher.token_dispatch(
                hidden_states=hidden_states,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                row_idx=row_idx,
                expert_map=expert_map,
                log2phy=log2phy,
                global_redundant_expert_num=global_redundant_expert_num,
                shared_experts=shared_experts,
                quantized_x_for_share=quantized_x_for_share,
                dynamic_scale_for_share=dynamic_scale_for_share,
                mc2_mask=effective_mc2_mask,
                apply_router_weight_on_input=apply_router_weight_on_input,
                with_quant=use_int8_w8a8 or use_int4_w4a8)
            permuted_hidden_states, expert_tokens, dynamic_scale, group_list_type, topk_scales = \
                results["hidden_states"], results["group_list"], results.get("dynamic_scale"), results["group_list_type"], results.get("topk_scales")
            mlp_output = unified_apply_mlp(hidden_states=permuted_hidden_states,
                                            w1=w1,
                                            w1_scale=w1_scale,
                                            w2=w2,
                                            w2_scale=w2_scale,
                                            group_list=expert_tokens,
                                            dynamic_scale=dynamic_scale,
                                            group_list_type=group_list_type,
                                            w1_scale_bias=w1_scale_bias,
                                            w2_scale_bias=w2_scale_bias,
                                            topk_scales=topk_scales,
                                            with_quant=use_int8_w8a8
                                            or use_int4_w4a8,
                                            fusion=use_int8_w8a8,
                                            need_trans=need_trans)
            mlp_hidden_states = self.token_dispatcher.token_combine(
                hidden_states=mlp_output)
            return mlp_hidden_states
        
        for chunk_start in range(0, max_tokens, chunk_moe_size):
            skip_result_store = chunk_start >= num_tokens
            chunk_end = min(chunk_start + chunk_moe_size, max_tokens)
            chunk_start = min(chunk_start, num_tokens - 1)
            chunk_end = min(chunk_end, num_tokens)
            chunk_hidden_states = hidden_states[chunk_start:chunk_end]
            chunk_topk_ids = topk_ids[chunk_start:chunk_end]
            chunk_topk_weights = topk_weights[chunk_start:chunk_end]
            chunk_shared_experts = None if shared_experts is None else shared_experts[chunk_start:chunk_end]
            chunk_mc2_mask = None
            if effective_mc2_mask is not None:
                chunk_mc2_mask = effective_mc2_mask[chunk_start:chunk_end]
            results = self.token_dispatcher.token_dispatch(
                hidden_states=chunk_hidden_states,
                topk_weights=chunk_topk_weights,
                topk_ids=chunk_topk_ids,
                row_idx=row_idx,
                expert_map=expert_map,
                log2phy=log2phy,
                global_redundant_expert_num=global_redundant_expert_num,
                shared_experts=chunk_shared_experts,
                quantized_x_for_share=quantized_x_for_share,
                dynamic_scale_for_share=dynamic_scale_for_share,
                mc2_mask=chunk_mc2_mask,
                apply_router_weight_on_input=apply_router_weight_on_input,
                with_quant=use_int8_w8a8 or use_int4_w4a8)
            
            permuted_hidden_states, expert_tokens, dynamic_scale, group_list_type, topk_scales = \
                results["hidden_states"], results["group_list"], results.get("dynamic_scale"), results["group_list_type"], results.get("topk_scales")

            mlp_output = unified_apply_mlp(hidden_states=permuted_hidden_states,
                                            w1=w1,
                                            w1_scale=w1_scale,
                                            w2=w2,
                                            w2_scale=w2_scale,
                                            group_list=expert_tokens,
                                            dynamic_scale=dynamic_scale,
                                            group_list_type=group_list_type,
                                            w1_scale_bias=w1_scale_bias,
                                            w2_scale_bias=w2_scale_bias,
                                            topk_scales=topk_scales,
                                            with_quant=use_int8_w8a8
                                            or use_int4_w4a8,
                                            fusion=use_int8_w8a8,
                                            need_trans=need_trans)
            mlp_hidden_states = self.token_dispatcher.token_combine(
                hidden_states=mlp_output)
            if skip_result_store:
                continue
            chunk_end_idx = chunk_start_index + mlp_hidden_states.shape[0]
            final_hidden_states[chunk_start_index: chunk_end_idx, :] = mlp_hidden_states
            chunk_start_index = chunk_end_idx

        if dynamic_eplb:
            return (final_hidden_states, group_list_type, expert_tokens)

        return final_hidden_states

    @abstractmethod
    def _get_token_dispatcher(self):
        raise NotImplementedError(
            "_get_token_dispatcher function not implemented.")

    @abstractmethod
    def _get_fused_moe_prepare_finalize(self):
        raise NotImplementedError(
            "_get_fused_moe_prepare_finalize function not implemented.")


class AllGatherCommImpl(MoECommMethod):
    """This implementation is the same as NativeAllGatherCommImpl,
    but uses NPU-specific ops for better performance.

    This implementation should be compatible with all scenarios, and
    thus it is the default implementation for MoE communication methods.
    It uses `torch_npu.npu_moe_init_routing_v2` for pre-processing
    and `torch_npu.npu_moe_token_unpermute` for post-processing
    to handle the token-to-expert mapping and communication efficiently.

    NOTE(Yizhou): TBH, it is really weird that we were supposed to use
    `torch_npu.npu_moe_init_routing_v2` and `torch_npu.npu_moe_finalize_routing`
    or `torch_npu.npu_moe_token_permute` and `torch_npu.npu_moe_token_unpermute`
    for pre-processing and post-processing, respectively.
    But `npu_moe_finalize_routing` will lead to accuracy issues so we have to
    use `torch_npu.npu_moe_token_unpermute` instead.
    This is a workaround and should be removed after the issue is fixed.
    """

    def _get_token_dispatcher(self):
        if self.model_type == "PanguProMoE":
            return TokenDispatcherWithMoge(
                top_k=self.moe_config.experts_per_token,
                num_experts=self.moe_config.num_experts,
                num_local_experts=self.moe_config.num_local_experts)
        else:
            return TokenDispatcherWithAllGather(
                top_k=self.moe_config.experts_per_token,
                num_experts=self.moe_config.num_experts,
                num_local_experts=self.moe_config.num_local_experts)

    def _get_fused_moe_prepare_finalize(self):
        return FusedMoEPrepareAndFinalizeWithAllGather(self.moe_config)


class MC2CommImpl(MoECommMethod):
    """This implementation is for the scenarios listed below:
    1. `enable_expert_parallel=True`.
    2. `npu_moe_distribute_dispatch` and `npu_moe_distribute_combine` are available.
    3. `enable_expert_parallel=False` is not supported.
    
    This implementation uses the MC2 communication method, which is optimized for
    Communication and Computation parallelism on Ascend devices.
    """

    def _get_token_dispatcher(self):
        return TokenDispatcherWithMC2(
            top_k=self.moe_config.experts_per_token,
            num_experts=self.moe_config.num_experts,
            num_local_experts=self.moe_config.num_local_experts)

    def _get_fused_moe_prepare_finalize(self):
        return FusedMoEPrepareAndFinalizeWithMC2(self.moe_config)


class AlltoAllCommImpl(MoECommMethod):
    """This implementation is for the scenarios listed below:
    1. `enable_expert_parallel=True`.
    2. `npu_grouped_matmul` is available.

    This implementation uses all-to-all communication to exchange tokens
    between data parallel ranks before and after the MLP computation. It should
    have better performance than AllGatherCommImpl when DP size > 1.
    """

    def _get_token_dispatcher(self):
        return TokenDispatcherWithAll2AllV(
            top_k=self.moe_config.experts_per_token,
            num_experts=self.moe_config.num_experts,
            num_local_experts=self.moe_config.num_local_experts)

    def _get_fused_moe_prepare_finalize(self):
        return FusedMoEPrepareAndFinalizeWithAll2All(self.moe_config)


class NaiveMulticastCommImpl(MoECommMethod):
    """This implementation is the same as NativeAllGatherCommImpl,
    but uses NPU-specific ops for better performance.

    This implementation should be compatible with all scenarios, and
    thus it is the default implementation for MoE communication methods.
    It uses `torch_npu.npu_moe_init_routing_v2` for pre-processing
    and `torch_npu.npu_moe_token_unpermute` for post-processing
    to handle the token-to-expert mapping and communication efficiently.

    NOTE(Yizhou): TBH, it is really weird that we were supposed to use
    `torch_npu.npu_moe_init_routing_v2` and `torch_npu.npu_moe_finalize_routing`
    or `torch_npu.npu_moe_token_permute` and `torch_npu.npu_moe_token_unpermute`
    for pre-processing and post-processing, respectively.
    But `npu_moe_finalize_routing` will lead to accuracy issues so we have to
    use `torch_npu.npu_moe_token_unpermute` instead.
    This is a workaround and should be removed after the issue is fixed.
    """

    def _get_token_dispatcher(self):
        return TokenDispatcherWithAllGather(
            top_k=self.moe_config.experts_per_token,
            num_experts=self.moe_config.num_experts,
            num_local_experts=self.moe_config.num_local_experts)

    def _get_fused_moe_prepare_finalize(self):
        return FusedMoEPrepareAndFinalizeWithNaiveMulticast(self.moe_config)
