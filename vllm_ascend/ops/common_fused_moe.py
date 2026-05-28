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
import os
import os.path
import re
import time
from typing import Any, Callable, Optional

import torch
import torch_npu
from vllm.config import CompilationLevel, get_current_vllm_config
from vllm.distributed import (get_dp_group, get_ep_group, get_tp_group,
                              tensor_model_parallel_all_reduce)
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.layer import (
    FusedMoE, UnquantizedFusedMoEMethod, determine_expert_map)
from vllm.model_executor.layers.shared_fused_moe import SharedFusedMoE

from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.ascend_forward_context import FusedMoEState, MoECommType
from vllm_ascend.distributed.parallel_state import get_mc2_group
from vllm_ascend import envs as envs_ascend
from vllm_ascend.eplb.core.eplb_utils import (determine_default_expert_map,
                                              determine_default_log2phy_map,
                                              determine_redundant_replica_expert_map)
from vllm_ascend.ops.expert_load_balancer import ExpertLoadBalancer
from vllm_ascend.ops.fused_moe import (Mode3DoubleBufferManager, _elapsed_ms,
                                       _event_record,
                                       _mode3_submit_accounted_us,
                                       _timing_float)
from vllm_ascend.ops.moe.experts_selector import return_row_idx, select_experts
from vllm_ascend.ops.moe.moe_comm_method import (get_moe_comm_method,
                                                 setup_moe_comm_method)
from vllm_ascend.ops.moe.moe_mlp import unified_apply_mlp
from vllm_ascend.utils import ACL_FORMAT_FRACTAL_NZ, is_310p, npu_stream_switch

logger = init_logger(__name__)

original_unquantized_fused_moe_init_func = UnquantizedFusedMoEMethod.__init__


def _env_flag(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).lower() in ("1", "true", "yes", "on")


def _verify_mode3_remapped_ids(dispatch_topk_ids: torch.Tensor,
                               layer: torch.nn.Module,
                               context: str) -> None:
    if not _env_flag("VLLM_ASCEND_MODE3_VERIFY_REMAP", "0"):
        return
    if torch.any(dispatch_topk_ids < 0):
        invalid_count = int(torch.count_nonzero(dispatch_topk_ids < 0).item())
        raise RuntimeError(
            f"Mode3 {context} saw experts outside the active rank ownership "
            f"map at layer={getattr(layer, 'layer_idx', -1)}: "
            f"invalid_count={invalid_count}")


def _get_num_experts_arg(args, kwargs) -> Optional[int]:
    if "num_experts" in kwargs:
        return int(kwargs["num_experts"])
    if args:
        return int(args[0])
    return None


def _get_prefix_arg(args, kwargs) -> str:
    if "prefix" in kwargs:
        return str(kwargs["prefix"])
    # FusedMoE.__init__ positional prefix index in the vLLM version used here.
    if len(args) > 14:
        return str(args[14])
    return ""


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


def _ensure_layer_idx_kwarg(args, kwargs):
    current = kwargs.get("layer_idx", None)
    try:
        if current is not None and int(current) >= 0:
            return kwargs
    except (TypeError, ValueError):
        pass
    inferred_layer_idx = _infer_layer_idx_from_prefix(_get_prefix_arg(args, kwargs))
    if inferred_layer_idx < 0:
        return kwargs
    kwargs = dict(kwargs)
    kwargs["layer_idx"] = inferred_layer_idx
    return kwargs


def _fused_moe_state_for_comm_type(
        comm_type: Optional[MoECommType],
        default_state: Optional[FusedMoEState]) -> Optional[FusedMoEState]:
    if comm_type == MoECommType.MC2:
        return FusedMoEState.MC2
    if comm_type == MoECommType.ALLTOALL:
        return FusedMoEState.All2All
    if comm_type == MoECommType.ALLGATHER:
        return FusedMoEState.AllGather
    if comm_type == MoECommType.NAIVE_MULTICAST:
        return FusedMoEState.NaiveMulticast
    return default_state


def _parse_expert_map_result(result):
    if len(result) == 3:
        return result
    if len(result) == 2:
        local_num_experts, expert_map = result
        return local_num_experts, expert_map, None
    raise ValueError(f"Unexpected determine_expert_map return arity={len(result)}")


def _lossless_weight_meta(weight: torch.Tensor) -> dict[str, object]:
    return {
        "shape": tuple(weight.shape),
        "dtype": str(weight.dtype),
        "device": str(weight.device),
        "storage_offset": int(weight.storage_offset()),
        "stride": tuple(weight.stride()),
        "ptr": int(weight.data_ptr()),
    }


def _maybe_pin_cpu_tensor(tensor: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if tensor is None or tensor.device.type != "cpu":
        return tensor
    try:
        if tensor.is_pinned():
            return tensor
    except Exception:
        pass
    try:
        return tensor.pin_memory()
    except Exception:
        return tensor


def _lossless_weight_format(weight: Optional[torch.Tensor]) -> str:
    if weight is None:
        return "none"
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        return "cpu"
    try:
        return str(torch_npu.get_npu_format(weight))
    except Exception:
        return "unknown"


def _allocate_formatted_buffer_like(weight: torch.Tensor,
                                    row_count: int,
                                    *,
                                    dtype: Optional[torch.dtype] = None
                                    ) -> torch.Tensor:
    buffer = torch.empty((row_count, ) + tuple(weight.shape[1:]),
                         device=weight.device,
                         dtype=weight.dtype if dtype is None else dtype)
    if weight.device.type != "npu":
        return buffer
    try:
        buffer = torch_npu.npu_format_cast(buffer, torch_npu.get_npu_format(weight))
    except Exception:
        pass
    return buffer


def _npu_zero_offset_alias_for_p2p(base: torch.Tensor,
                                   slot_view: torch.Tensor) -> torch.Tensor:
    if slot_view.device.type != "npu":
        return slot_view
    if int(slot_view.storage_offset()) == 0:
        return slot_view

    try:
        target_format = torch_npu.get_npu_format(slot_view)
        alias = torch.empty(tuple(slot_view.shape),
                            device=slot_view.device,
                            dtype=slot_view.dtype)
        alias = torch_npu.npu_format_cast(alias, target_format)

        ptr_delta = int(slot_view.data_ptr()) - int(base.data_ptr())
        element_size = int(slot_view.element_size())
        storage_index = int(slot_view.storage_offset())
        if ptr_delta >= 0 and element_size > 0 and ptr_delta % element_size == 0:
            storage_index = ptr_delta // element_size
        torch_npu.npu_change_data_ptr(alias, base, int(storage_index))

        if int(alias.storage_offset()) != 0:
            raise RuntimeError(
                f"alias storage_offset={int(alias.storage_offset())}")
        if int(alias.data_ptr()) != int(slot_view.data_ptr()):
            raise RuntimeError(
                f"alias data_ptr={int(alias.data_ptr())} "
                f"view data_ptr={int(slot_view.data_ptr())}")
        return alias
    except Exception as exc:
        raise RuntimeError(
            "Failed to build zero-offset NPU P2P alias for formatted expert "
            f"slot: base_meta={_lossless_weight_meta(base)} "
            f"slot_meta={_lossless_weight_meta(slot_view)}") from exc


def _hybrid_rank_owned_signature(
        rank_owned: Optional[list[list[int]]]) -> tuple[tuple[int, ...], ...]:
    if not rank_owned:
        return ()
    return tuple(
        tuple(int(expert_id) for expert_id in expert_ids)
        for expert_ids in rank_owned)


def _hybrid_dispatch_topology_signature(layer: Any) -> tuple[Any, ...]:
    return (
        int(
            getattr(layer, "elastic_original_num_experts",
                    getattr(layer, "num_experts", 0))),
        tuple(
            int(rank)
            for rank in getattr(layer, "lossless_hybrid_active_ranks", [])),
        _hybrid_rank_owned_signature(
            getattr(layer, "lossless_hybrid_rank_owned_expert_ids", None)),
    )


def _build_dispatch_log2phy_tensor(
        logical_num_experts: int,
        rank_owned: list[list[int]],
        owned_per_rank: int,
        device: torch.device) -> tuple[torch.Tensor, int]:
    dispatch_num_experts = len(rank_owned) * owned_per_rank
    dispatch_log2phy_cpu = torch.full((logical_num_experts, ),
                                      -1,
                                      dtype=torch.int32,
                                      device="cpu")
    for rank_idx, expert_ids in enumerate(rank_owned):
        if len(expert_ids) != owned_per_rank:
            raise RuntimeError(
                "Hybrid single-dispatch requires uniform owned experts per "
                f"rank: owned_counts={[len(ids) for ids in rank_owned]}")
        if not expert_ids:
            continue
        expert_index = torch.tensor([int(expert_id) for expert_id in expert_ids],
                                    dtype=torch.long,
                                    device="cpu")
        dense_ids = torch.arange(rank_idx * owned_per_rank,
                                 (rank_idx + 1) * owned_per_rank,
                                 dtype=torch.int32,
                                 device="cpu")
        dispatch_log2phy_cpu[expert_index] = dense_ids
    return dispatch_log2phy_cpu.to(device=device, non_blocking=False), \
        dispatch_num_experts


def _get_dispatch_log2phy_for_layer(
        layer: torch.nn.Module,
        *,
        device: torch.device,
        rank_owned: list[list[int]],
        active_rank_count: int,
        owned_per_rank: int) -> tuple[torch.Tensor, int]:
    topology_signature = _hybrid_dispatch_topology_signature(layer)
    expected_dispatch_num_experts = active_rank_count * owned_per_rank
    cached_dispatch_log2phy = getattr(layer, "lossless_runtime_dispatch_log2phy",
                                      None)
    cached_dispatch_num_experts = int(
        getattr(layer, "lossless_runtime_dispatch_num_experts", -1))
    cached_signature = getattr(layer, "lossless_runtime_dispatch_signature", None)
    if (cached_dispatch_log2phy is not None
            and cached_dispatch_log2phy.device == device
            and cached_dispatch_num_experts == expected_dispatch_num_experts
            and cached_signature == topology_signature):
        return cached_dispatch_log2phy, cached_dispatch_num_experts
    dispatch_log2phy, dispatch_num_experts = _build_dispatch_log2phy_tensor(
        int(layer.elastic_original_num_experts),
        rank_owned,
        owned_per_rank,
        device,
    )
    layer.lossless_runtime_dispatch_log2phy = dispatch_log2phy
    layer.lossless_runtime_dispatch_num_experts = int(dispatch_num_experts)
    layer.lossless_runtime_dispatch_signature = topology_signature
    layer.lossless_runtime_dispatch_active_rank_count = int(active_rank_count)
    layer.lossless_runtime_dispatch_owned_per_rank = int(owned_per_rank)
    return dispatch_log2phy, dispatch_num_experts


def _group_list_to_counts(group_list: torch.Tensor,
                          group_list_type: int) -> torch.Tensor:
    if group_list_type == 1:
        return group_list
    if group_list_type == 0:
        if group_list.numel() == 0:
            return group_list
        return torch.cat([group_list[:1], torch.diff(group_list, dim=0)])
    raise RuntimeError(
        "Unsupported group_list_type for hybrid single-dispatch path: "
        f"{group_list_type}")


def _counts_to_offsets(group_counts: torch.Tensor) -> torch.Tensor:
    if group_counts.numel() == 0:
        return torch.zeros((1, ), dtype=torch.long, device=group_counts.device)
    counts = group_counts.to(dtype=torch.long)
    return torch.cat([
        torch.zeros((1, ), dtype=torch.long, device=counts.device),
        counts.cumsum(dim=0)
    ])


def _should_use_single_dispatch_hybrid_path(
        layer: torch.nn.Module,
        wave_plans: list[dict[str, Any]],
        use_dense_mc2_waves: bool) -> bool:
    if not use_dense_mc2_waves or not wave_plans:
        return False
    if getattr(layer, "dynamic_eplb", False):
        return False
    if getattr(layer, "elastic_execution_mode", 0) not in (2, 3):
        return False
    if not hasattr(layer, "_is_hybrid_cpu_swap_enabled"):
        return False
    if not layer._is_hybrid_cpu_swap_enabled():
        return False
    rank_owned = getattr(layer, "lossless_hybrid_rank_owned_expert_ids", None)
    if not rank_owned:
        return False
    owned_counts = [len(expert_ids) for expert_ids in rank_owned]
    if not owned_counts or min(owned_counts) <= 0:
        return False
    return len(set(owned_counts)) == 1


def _should_use_mode2_single_dispatch_hybrid_path(
        layer: torch.nn.Module,
        wave_plans: list[dict[str, Any]],
        use_dense_mc2_waves: bool) -> bool:
    return _should_use_single_dispatch_hybrid_path(
        layer=layer,
        wave_plans=wave_plans,
        use_dense_mc2_waves=use_dense_mc2_waves,
    )


def _should_use_mode3_cross_layer_buffer_path(layer: torch.nn.Module) -> bool:
    if getattr(layer, "elastic_execution_mode", 0) != 3:
        return False
    if not getattr(layer, "lossless_hybrid_active", False):
        return False
    forward_context = get_forward_context()
    comm_type = getattr(forward_context, "moe_comm_type", None)
    if comm_type == MoECommType.MC2:
        return True
    selected_comm_type = getattr(forward_context, "selected_moe_comm_type", None)
    if selected_comm_type == MoECommType.MC2:
        return True
    moe_comm_method = getattr(forward_context, "moe_comm_method", None)
    token_dispatcher = getattr(moe_comm_method, "token_dispatcher", None)
    return (token_dispatcher is not None
            and token_dispatcher.__class__.__name__ == "TokenDispatcherWithMC2")


def _should_use_mode3_single_rank_allgather_path(
        layer: torch.nn.Module) -> bool:
    if getattr(layer, "elastic_execution_mode", 0) != 3:
        return False
    if not getattr(layer, "lossless_hybrid_active", False):
        return False
    active_rank_count = len(getattr(layer, "lossless_hybrid_active_ranks", []))
    if active_rank_count != 1:
        return False
    forward_context = get_forward_context()
    selected_comm_type = getattr(forward_context, "selected_moe_comm_type", None)
    comm_type = getattr(forward_context, "moe_comm_type", None)
    if selected_comm_type != MoECommType.ALLGATHER and comm_type != MoECommType.ALLGATHER:
        return False
    moe_comm_method = getattr(forward_context, "moe_comm_method", None)
    token_dispatcher = getattr(moe_comm_method, "token_dispatcher", None)
    return (
        token_dispatcher is not None
        and token_dispatcher.__class__.__name__ == "TokenDispatcherWithAllGather")


def _get_or_create_mode3_double_buffer_manager(
        layer: torch.nn.Module) -> Optional[Mode3DoubleBufferManager]:
    forward_context = get_forward_context()
    manager = getattr(forward_context, "moe_double_buffer_manager", None)
    if manager is not None:
        return manager
    prefetch_stream = getattr(forward_context, "moe_prefetch_stream", None)
    model_instance = getattr(forward_context, "model_instance", None)
    if prefetch_stream is None or model_instance is None:
        return None
    manager = Mode3DoubleBufferManager(model_instance, prefetch_stream)
    forward_context.moe_double_buffer_manager = manager
    return manager


def _active_moe_weight_view(layer: torch.nn.Module,
                            weight: torch.Tensor) -> torch.Tensor:
    active_num_experts = int(
        getattr(layer, "active_local_num_experts",
                getattr(layer, "local_num_experts", int(weight.shape[0]))))
    if active_num_experts == int(weight.shape[0]):
        return weight
    if active_num_experts <= 0 or active_num_experts > int(weight.shape[0]):
        raise RuntimeError(
            "Invalid active local expert count for MoE weight view: "
            f"active={active_num_experts}, weight_shape={tuple(weight.shape)}, "
            f"layer={getattr(layer, 'layer_idx', 'unknown')}")
    return weight[:active_num_experts]


def _compact_wave_topk(
        logical_topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        token_mask: torch.Tensor,
        topk_mask: torch.Tensor,
        preserve_full_topk: bool = True
) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor],
           Optional[torch.Tensor], Optional[torch.Tensor]]:

    def _empty_wave() -> tuple[Optional[torch.Tensor], Optional[torch.Tensor],
                               Optional[torch.Tensor], Optional[torch.Tensor]]:
        return None, None, None, None

    if token_mask.numel() == 0 or not torch.any(token_mask):
        return _empty_wave()
    token_indices = torch.nonzero(token_mask, as_tuple=False).reshape(-1)
    selected_ids = logical_topk_ids[token_indices]
    selected_weights = topk_weights[token_indices]
    selected_mask = topk_mask[token_indices]
    per_token_counts = selected_mask.sum(dim=1)
    if per_token_counts.numel() == 0:
        return _empty_wave()
    full_topk = int(selected_ids.shape[1])
    if full_topk <= 0:
        return _empty_wave()
    wave_topk = full_topk
    if not preserve_full_topk:
        wave_topk = max(int(per_token_counts.max().item()), 1)
    wave_ids = torch.zeros((selected_ids.shape[0], wave_topk),
                           dtype=selected_ids.dtype,
                           device=selected_ids.device)
    wave_weights = torch.zeros((selected_weights.shape[0], wave_topk),
                               dtype=selected_weights.dtype,
                               device=selected_weights.device)
    fill_pos = torch.zeros((selected_ids.shape[0], ),
                           dtype=torch.long,
                           device=selected_ids.device)
    for col_idx in range(selected_ids.shape[1]):
        col_mask = selected_mask[:, col_idx]
        if not torch.any(col_mask):
            continue
        row_indices = torch.nonzero(col_mask, as_tuple=False).reshape(-1)
        write_pos = fill_pos[row_indices]
        wave_ids[row_indices, write_pos] = selected_ids[row_indices, col_idx]
        wave_weights[row_indices, write_pos] = selected_weights[
            row_indices, col_idx]
        fill_pos[row_indices] += 1

    filled_mask = (torch.arange(wave_topk, device=selected_ids.device)
                   .unsqueeze(0) < fill_pos.unsqueeze(1))
    fallback_ids = wave_ids[:, :1].expand(-1, wave_topk)
    wave_ids = torch.where(filled_mask, wave_ids, fallback_ids)
    wave_row_idx = return_row_idx(wave_ids, wave_topk)
    return token_indices, wave_ids, wave_weights, wave_row_idx


def _build_padded_mc2_wave_inputs(
        hidden_states: torch.Tensor,
        logical_topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        token_mask: torch.Tensor,
        topk_mask: torch.Tensor,
        fallback_expert_id: int,
        forced_active_topk: Optional[int] = None
) -> tuple[Optional[torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor,
           torch.Tensor, torch.Tensor, int]:
    token_indices, compact_ids, compact_weights, compact_row_idx = (
        _compact_wave_topk(logical_topk_ids=logical_topk_ids,
                           topk_weights=topk_weights,
                           token_mask=token_mask,
                           topk_mask=topk_mask,
                           preserve_full_topk=(forced_active_topk is None)))
    source_tokens = int(hidden_states.shape[0])
    if source_tokens <= 0:
        raise RuntimeError("MC2 hybrid wave received empty hidden_states.")
    logical_topk_width = int(logical_topk_ids.shape[1])
    active_topk = logical_topk_width
    if forced_active_topk is not None:
        active_topk = int(forced_active_topk)
    if active_topk <= 0:
        raise RuntimeError(
            "MC2 hybrid wave requires a positive active top-k width: "
            f"active_topk={active_topk}")
    if active_topk > logical_topk_width:
        raise RuntimeError(
            "Forced MC2 wave top-k exceeds logical top-k width: "
            f"forced={active_topk} logical={logical_topk_width}")

    fallback_expert = int(fallback_expert_id)
    active_count = 0
    if (token_indices is not None and compact_ids is not None
            and compact_weights is not None and compact_row_idx is not None
            and token_indices.numel() > 0):
        active_count = int(token_indices.numel())
        compact_width = int(compact_ids.shape[1])
        if compact_width > active_topk:
            raise RuntimeError(
                "MC2 hybrid wave compact width mismatch: "
                f"compact={compact_width} expected={active_topk}")
        if compact_width < active_topk:
            padded_ids = torch.full((active_count, active_topk),
                                    fallback_expert,
                                    dtype=compact_ids.dtype,
                                    device=compact_ids.device)
            padded_weights = torch.zeros((active_count, active_topk),
                                         dtype=compact_weights.dtype,
                                         device=compact_weights.device)
            padded_ids[:, :compact_width] = compact_ids
            padded_weights[:, :compact_width] = compact_weights
            compact_ids = padded_ids
            compact_weights = padded_weights
            compact_row_idx = return_row_idx(compact_ids, active_topk)
        wave_hidden_states = hidden_states.index_select(0, token_indices)
        wave_ids = compact_ids
        wave_weights = compact_weights
        wave_mc2_mask = torch.ones((active_count, ),
                                   dtype=torch.bool,
                                   device=hidden_states.device)
        wave_row_idx = compact_row_idx
    else:
        wave_hidden_states = hidden_states[:1].clone()
        wave_hidden_states.zero_()
        wave_ids = torch.full((1, active_topk),
                              fallback_expert,
                              dtype=logical_topk_ids.dtype,
                              device=logical_topk_ids.device)
        wave_weights = torch.zeros((1, active_topk),
                                   dtype=topk_weights.dtype,
                                   device=topk_weights.device)
        wave_mc2_mask = torch.zeros((1, ),
                                    dtype=torch.bool,
                                    device=hidden_states.device)
        wave_row_idx = return_row_idx(wave_ids, active_topk)
    return (token_indices, wave_hidden_states, wave_ids, wave_weights,
            wave_row_idx, wave_mc2_mask, active_count)


def _build_full_batch_mc2_wave_inputs(
        hidden_states: torch.Tensor,
        logical_topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        token_mask: torch.Tensor,
        topk_mask: torch.Tensor,
        fallback_expert_id: int) -> tuple[torch.Tensor, torch.Tensor,
                                          torch.Tensor, torch.Tensor,
                                          torch.Tensor, torch.Tensor, int]:
    source_tokens = int(hidden_states.shape[0])
    if source_tokens <= 0:
        raise RuntimeError("MC2 hybrid wave received empty hidden_states.")
    logical_topk_width = int(logical_topk_ids.shape[1])
    if logical_topk_width <= 0:
        raise RuntimeError(
            "MC2 hybrid wave requires a positive logical top-k width.")

    fallback_expert = int(fallback_expert_id)
    active_mask = token_mask.to(device=hidden_states.device, dtype=torch.bool)
    token_indices = torch.nonzero(active_mask, as_tuple=False).reshape(-1)
    active_count = int(token_indices.numel())

    wave_ids = torch.full((source_tokens, logical_topk_width),
                          fallback_expert,
                          dtype=logical_topk_ids.dtype,
                          device=logical_topk_ids.device)
    wave_weights = torch.zeros((source_tokens, logical_topk_width),
                               dtype=topk_weights.dtype,
                               device=topk_weights.device)
    fill_pos = torch.zeros((source_tokens, ),
                           dtype=torch.long,
                           device=logical_topk_ids.device)
    effective_topk_mask = topk_mask & token_mask.unsqueeze(1)
    for col_idx in range(logical_topk_width):
        col_mask = effective_topk_mask[:, col_idx]
        if not torch.any(col_mask):
            continue
        row_indices = torch.nonzero(col_mask, as_tuple=False).reshape(-1)
        write_pos = fill_pos[row_indices]
        wave_ids[row_indices, write_pos] = logical_topk_ids[row_indices,
                                                            col_idx]
        wave_weights[row_indices, write_pos] = topk_weights[row_indices,
                                                            col_idx]
        fill_pos[row_indices] += 1

    wave_row_idx = return_row_idx(wave_ids, logical_topk_width)
    return (token_indices, hidden_states, wave_ids, wave_weights, wave_row_idx,
            active_mask, active_count)


def _execute_lossless_hybrid_wave(
        layer: torch.nn.Module,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        row_idx: torch.Tensor,
        global_num_experts: int,
        log2phy: torch.Tensor,
        activation: str,
        apply_router_weight_on_input: bool,
        mc2_mask: Optional[torch.Tensor]) -> torch.Tensor:
    moe_comm_method = get_forward_context().moe_comm_method
    runtime_w13_weight = getattr(layer, "runtime_w13_weight", None)
    runtime_w2_weight = getattr(layer, "runtime_w2_weight", None)
    if runtime_w13_weight is None or runtime_w2_weight is None:
        raise RuntimeError(
            f"Hybrid runtime weights are missing at layer={layer.layer_idx}.")
    return moe_comm_method.fused_experts(
        hidden_states=hidden_states,
        w1=runtime_w13_weight,
        w2=runtime_w2_weight,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        row_idx=row_idx,
        global_num_experts=global_num_experts,
        expert_map=layer.expert_map,
        activation=activation,
        apply_router_weight_on_input=apply_router_weight_on_input,
        log2phy=log2phy,
        global_redundant_expert_num=0,
        mc2_mask=mc2_mask)


def _execute_mode2_single_dispatch_hybrid(
        layer: torch.nn.Module,
        x: torch.Tensor,
        logical_topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        row_idx: torch.Tensor,
        wave_plans: list[dict[str, Any]],
        prepared_mc2_mask: Optional[torch.Tensor]) -> torch.Tensor:
    moe_comm_method = get_forward_context().moe_comm_method
    token_dispatcher = getattr(moe_comm_method, "token_dispatcher", None)
    if token_dispatcher is None:
        raise RuntimeError(
            "Hybrid mode2 single-dispatch missing MC2 token dispatcher.")

    local_rank_idx = int(getattr(layer, "lossless_hybrid_active_rank_index", -1))
    rank_owned = getattr(layer, "lossless_hybrid_rank_owned_expert_ids", None)
    if rank_owned is None or local_rank_idx < 0 or local_rank_idx >= len(rank_owned):
        raise RuntimeError(
            "Hybrid mode2 single-dispatch invalid ownership state: "
            f"layer={getattr(layer, 'layer_idx', -1)} "
            f"local_rank_idx={local_rank_idx}")
    active_rank_count = len(getattr(layer, "lossless_hybrid_active_ranks", []))
    local_owned_expert_ids = [int(expert_id) for expert_id in rank_owned[
        local_rank_idx]]
    owned_per_rank = len(local_owned_expert_ids)
    if active_rank_count <= 0 or owned_per_rank <= 0:
        raise RuntimeError(
            "Hybrid mode2 single-dispatch invalid dispatch shape: "
            f"active_rank_count={active_rank_count} "
            f"owned_per_rank={owned_per_rank}")

    dispatch_log2phy, dispatch_num_experts = _get_dispatch_log2phy_for_layer(
        layer,
        device=logical_topk_ids.device,
        rank_owned=rank_owned,
        active_rank_count=active_rank_count,
        owned_per_rank=owned_per_rank,
    )
    dispatch_topk_ids = dispatch_log2phy[logical_topk_ids]
    _verify_mode3_remapped_ids(dispatch_topk_ids, layer,
                               "mode2 single-dispatch")

    if prepared_mc2_mask is None:
        prepared_mask = torch.ones((int(x.shape[0]), ),
                                  dtype=torch.bool,
                                  device=x.device)
    else:
        prepared_mask = prepared_mc2_mask.to(device=x.device, dtype=torch.bool)
    if int(prepared_mask.shape[0]) != int(x.shape[0]):
        raise RuntimeError(
            "Hybrid mode2 single-dispatch MC2 mask length mismatch: "
            f"hidden_tokens={int(x.shape[0])} "
            f"mask_tokens={int(prepared_mask.shape[0])}")

    if getattr(layer, "layer_idx", -1) == 0:
        logger.info(
            "Native common hybrid MC2 single-dispatch launch: rank=%s "
            "layer=%s stage=%s waves=%s tokens=%s topk_width=%s "
            "owned_per_rank=%s dispatch_experts=%s",
            getattr(layer, "rank", -1),
            getattr(layer, "layer_idx", -1),
            active_rank_count,
            len(wave_plans),
            int(x.shape[0]),
            int(logical_topk_ids.shape[1]),
            owned_per_rank,
            dispatch_num_experts,
        )

    old_dispatch_num_experts = int(getattr(token_dispatcher, "num_experts", 0))
    token_dispatcher.num_experts = dispatch_num_experts
    try:
        dispatch_results = token_dispatcher.token_dispatch(
            hidden_states=x,
            topk_weights=topk_weights,
            topk_ids=dispatch_topk_ids,
            row_idx=row_idx,
            expert_map=layer.expert_map,
            log2phy=None,
            global_redundant_expert_num=0,
            shared_experts=None,
            quantized_x_for_share=None,
            dynamic_scale_for_share=None,
            mc2_mask=prepared_mask,
            apply_router_weight_on_input=False,
            with_quant=False)
        dispatched_hidden_states = dispatch_results["hidden_states"]
        dispatched_group_list = dispatch_results["group_list"]
        dispatched_group_list_type = int(dispatch_results["group_list_type"])
        raw_dispatched_group_counts = _group_list_to_counts(
            dispatched_group_list, dispatched_group_list_type).to(dtype=torch.long)
        if int(raw_dispatched_group_counts.numel()) == int(dispatch_num_experts):
            local_dense_start = local_rank_idx * owned_per_rank
            local_dense_end = local_dense_start + owned_per_rank
            dispatched_group_counts = raw_dispatched_group_counts[
                local_dense_start:local_dense_end]
        else:
            dispatched_group_counts = raw_dispatched_group_counts
        if int(dispatched_group_counts.numel()) != int(owned_per_rank):
            raise RuntimeError(
                "Hybrid mode2 single-dispatch expert group size mismatch: "
                f"expected_experts={owned_per_rank} "
                f"actual_experts={int(raw_dispatched_group_counts.numel())} "
                f"group_list_type={dispatched_group_list_type}")

        dispatch_offsets = _counts_to_offsets(dispatched_group_counts)
        dispatched_active_rows = int(dispatch_offsets[-1].item())
        if dispatched_active_rows > int(dispatched_hidden_states.shape[0]):
            raise RuntimeError(
                "Hybrid mode2 single-dispatch local token count mismatch: "
                f"active_rows={dispatched_active_rows} "
                f"dispatched_rows={int(dispatched_hidden_states.shape[0])} "
                f"group_list_len={int(dispatched_group_list.numel())} "
                f"group_list_type={dispatched_group_list_type}")

        dispatched_output = torch.zeros_like(dispatched_hidden_states)
        local_owned_index_by_expert = {
            int(expert_id): local_idx
            for local_idx, expert_id in enumerate(local_owned_expert_ids)
        }
        local_wave_counts: list[int] = []
        local_cpu_miss = 0
        local_resident_hits = 0

        for wave_idx, wave_plan in enumerate(wave_plans):
            effective_token_mask = wave_plan["token_mask"]
            mc2_active_mask = prepared_mask.to(device=effective_token_mask.device,
                                               dtype=torch.bool)
            effective_token_mask = effective_token_mask & mc2_active_mask
            if (effective_token_mask.numel() > 0
                    and torch.any(effective_token_mask)):
                local_wave_counts.append(
                    int(torch.count_nonzero(effective_token_mask).item()))
            else:
                local_wave_counts.append(0)

            target_resident = list(wave_plan["rank_resident"][local_rank_idx])
            current_resident = list(
                getattr(layer, "lossless_hybrid_resident_expert_ids", []))
            current_resident_set = set(current_resident)
            local_cpu_miss += len([
                expert_id for expert_id in target_resident
                if expert_id not in current_resident_set
            ])
            local_resident_hits += len([
                expert_id for expert_id in target_resident
                if expert_id in current_resident_set
            ])
            layer.materialize_hybrid_resident_experts(target_resident)

            wave_group_by_rank = wave_plan.get("rank_wave_groups", [])
            local_wave_group = (
                [int(expert_id) for expert_id in wave_group_by_rank[
                    local_rank_idx]]
                if 0 <= local_rank_idx < len(wave_group_by_rank) else [])
            local_wave_group_set = set(local_wave_group)
            resident_capacity = int(layer.lossless_hybrid_resident_capacity)
            resident_group_counts = torch.zeros(
                (resident_capacity, ),
                dtype=torch.int64,
                device=dispatched_hidden_states.device)
            hidden_segments: list[torch.Tensor] = []
            segment_targets: list[tuple[int, int, int]] = []
            resident_dispatched_tokens = 0

            for slot_idx, expert_id in enumerate(
                    target_resident[:resident_capacity]):
                expert_id = int(expert_id)
                if expert_id not in local_wave_group_set:
                    continue
                local_owned_idx = local_owned_index_by_expert.get(expert_id)
                if local_owned_idx is None:
                    continue
                start = int(dispatch_offsets[local_owned_idx].item())
                end = int(dispatch_offsets[local_owned_idx + 1].item())
                token_count = max(0, end - start)
                if token_count <= 0:
                    continue
                resident_group_counts[slot_idx] = token_count
                hidden_segments.append(dispatched_hidden_states[start:end])
                segment_targets.append((slot_idx, start, end))
                resident_dispatched_tokens += token_count

            if getattr(layer, "layer_idx", -1) == 0:
                logger.info(
                    "Native common hybrid MC2 single-dispatch wave: rank=%s "
                    "stage=%s wave=%s/%s active_local_experts=%s "
                    "dispatched_tokens=%s resident_capacity=%s",
                    getattr(layer, "rank", -1),
                    active_rank_count,
                    wave_idx + 1,
                    len(wave_plans),
                    len(local_wave_group),
                    resident_dispatched_tokens,
                    resident_capacity,
                )

            if not hidden_segments:
                continue

            runtime_w13_weight = getattr(layer, "runtime_w13_weight", None)
            runtime_w2_weight = getattr(layer, "runtime_w2_weight", None)
            if runtime_w13_weight is None or runtime_w2_weight is None:
                raise RuntimeError(
                    "Hybrid mode2 single-dispatch missing runtime weights: "
                    f"layer={getattr(layer, 'layer_idx', -1)}")
            wave_hidden_states = torch.cat(hidden_segments, dim=0)
            wave_output = unified_apply_mlp(
                hidden_states=wave_hidden_states,
                w1=runtime_w13_weight,
                w1_scale=None,
                w2=runtime_w2_weight,
                w2_scale=None,
                group_list=resident_group_counts,
                dynamic_scale=None,
                group_list_type=1,
                w1_scale_bias=None,
                w2_scale_bias=None,
                topk_scales=None,
                with_quant=False,
                fusion=False,
                need_trans=False)
            output_cursor = 0
            for _slot_idx, start, end in segment_targets:
                token_count = end - start
                dispatched_output[start:end] = wave_output[
                    output_cursor:output_cursor + token_count]
                output_cursor += token_count

        final_hidden_states = token_dispatcher.token_combine(dispatched_output)
    finally:
        token_dispatcher.num_experts = old_dispatch_num_experts

    layer.lossless_hybrid_rank_resident_expert_ids = [
        list(expert_ids) for expert_ids in wave_plans[-1]["final_rank_resident"]
    ]
    layer.lossless_hybrid_rank_lru = [
        list(expert_ids) for expert_ids in wave_plans[-1]["final_rank_resident"]
    ]
    if 0 <= local_rank_idx < len(wave_plans[-1]["final_rank_resident"]):
        layer.materialize_hybrid_resident_experts(
            list(wave_plans[-1]["final_rank_resident"][local_rank_idx]))
    layer.lossless_hybrid_last_stats = {
        "waves": len(wave_plans),
        "local_wave_tokens": local_wave_counts,
        "local_cpu_miss": local_cpu_miss,
        "local_resident_hits": local_resident_hits,
    }
    if getattr(layer, "layer_idx", -1) == 0:
        logger.info(
            "Native common hybrid MC2 single-dispatch done: rank=%s "
            "layer=%s stage=%s waves=%s resident_hits=%s cpu_miss=%s "
            "local_wave_tokens=%s resident=%s dispatch_experts=%s",
            getattr(layer, "rank", -1),
            getattr(layer, "layer_idx", -1),
            active_rank_count,
            len(wave_plans),
            local_resident_hits,
            local_cpu_miss,
            local_wave_counts[:8],
            getattr(layer, "lossless_hybrid_resident_expert_ids", [])[:8],
            dispatch_num_experts,
        )
    return final_hidden_states


def _execute_mode3_single_rank_allgather_hybrid(
        layer: torch.nn.Module,
        x: torch.Tensor,
        logical_topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        row_idx: torch.Tensor) -> torch.Tensor:
    manager = _get_or_create_mode3_double_buffer_manager(layer)
    if manager is None:
        raise RuntimeError(
            "Mode3 single-rank AllGather path requires forward-context "
            "model instance and moe prefetch stream at "
            f"layer={getattr(layer, 'layer_idx', -1)}.")

    bound_slot = manager.bind_current_layer(layer)
    manager.prefetch_next_layer(layer)

    moe_comm_method = get_forward_context().moe_comm_method
    token_dispatcher = getattr(moe_comm_method, "token_dispatcher", None)
    if (token_dispatcher is None
            or token_dispatcher.__class__.__name__ !=
            "TokenDispatcherWithAllGather"):
        raise RuntimeError(
            "Mode3 single-rank path requires AllGather token dispatcher, got "
            f"{token_dispatcher.__class__.__name__ if token_dispatcher is not None else None} "
            f"at layer={getattr(layer, 'layer_idx', -1)}.")

    slot_log2phy = bound_slot.expert_map.to(device=logical_topk_ids.device)
    local_topk_ids = slot_log2phy[logical_topk_ids]
    if torch.any(local_topk_ids < 0):
        invalid_count = int(torch.count_nonzero(local_topk_ids < 0).item())
        raise RuntimeError(
            "Mode3 single-rank AllGather saw experts outside the bound slot "
            f"at layer={getattr(layer, 'layer_idx', -1)}: "
            f"invalid_count={invalid_count}")

    old_num_experts = getattr(token_dispatcher, "num_experts", None)
    old_num_experts_local = getattr(token_dispatcher, "num_experts_local", None)
    token_dispatcher.num_experts = bound_slot.valid_expert_count
    token_dispatcher.num_experts_local = bound_slot.valid_expert_count
    try:
        final_hidden_states = moe_comm_method.fused_experts(
            hidden_states=x,
            w1=layer.runtime_w13_weight,
            w2=layer.runtime_w2_weight,
            topk_weights=topk_weights,
            topk_ids=local_topk_ids,
            row_idx=row_idx,
            global_num_experts=bound_slot.valid_expert_count,
            expert_map=None,
            shared_experts=None,
            log2phy=None,
            global_redundant_expert_num=0,
            need_trans=False,
            dynamic_eplb=getattr(layer, "dynamic_eplb", False),
            mc2_mask=None)
    finally:
        if old_num_experts is not None:
            token_dispatcher.num_experts = old_num_experts
        if old_num_experts_local is not None:
            token_dispatcher.num_experts_local = old_num_experts_local

    layer.lossless_hybrid_last_stats = {
        "mode3_slot": int(layer.layer_idx) & 1,
        "valid_experts": int(bound_slot.valid_expert_count),
        "source_from_npu": int(bound_slot.source_from_npu),
        "source_from_cpu": int(bound_slot.source_from_cpu),
        "single_rank_allgather": 1,
    }
    if layer.layer_idx == 0 and getattr(manager, "enable_transfer_logs", False):
        logger.info(
            "Native common mode3 single-rank AllGather execution: rank=%s "
            "layer=%s valid_experts=%s source_from_npu=%s source_from_cpu=%s",
            getattr(layer, "rank", -1),
            layer.layer_idx,
            int(bound_slot.valid_expert_count),
            int(bound_slot.source_from_npu),
            int(bound_slot.source_from_cpu),
        )
    return final_hidden_states


def _execute_mode3_fused_experts_hybrid(
        layer: torch.nn.Module,
        x: torch.Tensor,
        logical_topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        row_idx: torch.Tensor,
        manager: Optional[Mode3DoubleBufferManager] = None) -> torch.Tensor:
    if manager is None:
        manager = _get_or_create_mode3_double_buffer_manager(layer)
    if manager is None:
        raise RuntimeError(
            "Mode3 fused-experts path requires forward-context model instance "
            f"and moe prefetch stream at layer={getattr(layer, 'layer_idx', -1)}.")
    if int(getattr(layer, "layer_idx", -1)) < 0:
        raise RuntimeError("Mode3 fused-experts path requires a valid layer_idx.")

    profile_timing = manager.should_profile_layer(layer, "native_fused_experts")
    bound_slot = manager.bind_current_layer(layer)
    bind_timing = dict(getattr(manager, "last_bind_timing", {}))
    next_prefetch_timing = manager.prefetch_next_layer(layer)

    local_rank_idx = int(getattr(layer, "lossless_hybrid_active_rank_index", -1))
    rank_owned = getattr(layer, "lossless_hybrid_rank_owned_expert_ids", None)
    if rank_owned is None or local_rank_idx < 0 or local_rank_idx >= len(rank_owned):
        raise RuntimeError(
            "Invalid hybrid rank ownership state for mode3 fused-experts "
            f"path at layer={getattr(layer, 'layer_idx', -1)}.")
    local_owned_expert_ids = [int(expert_id) for expert_id in rank_owned[
        local_rank_idx]]
    active_rank_count = len(getattr(layer, "lossless_hybrid_active_ranks", []))
    owned_per_rank = len(local_owned_expert_ids)
    if active_rank_count <= 0 or owned_per_rank <= 0:
        raise RuntimeError(
            "Invalid hybrid mode3 fused-experts shape: "
            f"active_rank_count={active_rank_count} "
            f"owned_per_rank={owned_per_rank}")

    remap_wall_start = time.perf_counter()
    dispatch_log2phy, dispatch_num_experts = _get_dispatch_log2phy_for_layer(
        layer,
        device=logical_topk_ids.device,
        rank_owned=rank_owned,
        active_rank_count=active_rank_count,
        owned_per_rank=owned_per_rank,
    )
    use_log2phy_dispatch = _env_flag(
        "VLLM_ASCEND_MODE3_NATIVE_LOG2PHY_DISPATCH", "0")
    if use_log2phy_dispatch:
        dispatch_topk_ids = logical_topk_ids
        dispatch_log2phy_arg = dispatch_log2phy
        if _env_flag("VLLM_ASCEND_MODE3_VERIFY_REMAP", "0"):
            _verify_mode3_remapped_ids(dispatch_log2phy[logical_topk_ids], layer,
                                       "fused-experts")
    else:
        dispatch_topk_ids = dispatch_log2phy[logical_topk_ids]
        dispatch_log2phy_arg = None
        _verify_mode3_remapped_ids(dispatch_topk_ids, layer, "fused-experts")
    remap_wall_ms = (time.perf_counter() - remap_wall_start) * 1e3

    moe_comm_method = get_forward_context().moe_comm_method
    token_dispatcher = getattr(moe_comm_method, "token_dispatcher", None)
    old_dispatch_num_experts = (
        int(getattr(token_dispatcher, "num_experts", 0))
        if token_dispatcher is not None else 0)
    old_expert_token_nums_type = (
        int(getattr(token_dispatcher, "expert_token_nums_type", 0))
        if token_dispatcher is not None else 0)
    compute_wall_start = time.perf_counter()
    compute_start_event = manager.new_timing_event() if profile_timing else None
    compute_end_event = manager.new_timing_event() if profile_timing else None
    fused_start_event = manager.new_timing_event() if profile_timing else None
    fused_end_event = manager.new_timing_event() if profile_timing else None
    _event_record(compute_start_event)
    try:
        if token_dispatcher is not None:
            token_dispatcher.num_experts = dispatch_num_experts
            token_dispatcher.expert_token_nums_type = (
                manager.expert_token_nums_type)
        fused_wall_start = time.perf_counter()
        _event_record(fused_start_event)
        final_hidden_states = moe_comm_method.fused_experts(
            hidden_states=x,
            w1=layer.runtime_w13_weight,
            w2=layer.runtime_w2_weight,
            topk_weights=topk_weights,
            topk_ids=dispatch_topk_ids,
            row_idx=row_idx,
            global_num_experts=dispatch_num_experts,
            expert_map=layer.expert_map,
            log2phy=dispatch_log2phy_arg,
            global_redundant_expert_num=0,
            need_trans=False,
            dynamic_eplb=getattr(layer, "dynamic_eplb", False),
            mc2_mask=getattr(moe_comm_method, "mc2_mask", None))
        _event_record(fused_end_event)
        fused_wall_ms = (time.perf_counter() - fused_wall_start) * 1e3
    finally:
        if token_dispatcher is not None:
            token_dispatcher.num_experts = old_dispatch_num_experts
            token_dispatcher.expert_token_nums_type = old_expert_token_nums_type
    _event_record(compute_end_event)
    compute_wall_ms = (time.perf_counter() - compute_wall_start) * 1e3

    layer.lossless_hybrid_last_stats = {
        "mode3_slot": int(layer.layer_idx) & 1,
        "valid_experts": int(bound_slot.valid_expert_count),
        "source_from_npu": int(bound_slot.source_from_npu),
        "source_from_cpu": int(bound_slot.source_from_cpu),
        "layer_local_buffer": int(getattr(bound_slot, "uses_layer_local_buffer",
                                          False)),
        "prefetch_wait_us": float(manager.prefetch_wait_us[int(layer.layer_idx)]),
        "prefetch_hit": int(manager.prefetch_hit[int(layer.layer_idx)]),
        "fused_experts_path": 1,
    }
    if layer.layer_idx == 0 and getattr(manager, "enable_transfer_logs", False):
        logger.info(
            "Native common mode3 fused-experts execution: rank=%s layer=%s "
            "stage=%s valid_experts=%s source_from_npu=%s source_from_cpu=%s "
            "dispatch_experts=%s owned_per_rank=%s",
            getattr(layer, "rank", -1),
            layer.layer_idx,
            active_rank_count,
            int(bound_slot.valid_expert_count),
            int(bound_slot.source_from_npu),
            int(bound_slot.source_from_cpu),
            dispatch_num_experts,
            owned_per_rank,
        )
    if profile_timing:
        prefetch_dev_ms = manager._prefetch_device_ms(next_prefetch_timing)
        prefetch_npu_dev_ms = _elapsed_ms(
            next_prefetch_timing.get("npu_start_event"),
            next_prefetch_timing.get("npu_end_event"))
        prefetch_npu_w13_dev_ms = _elapsed_ms(
            next_prefetch_timing.get("npu_w13_start_event"),
            next_prefetch_timing.get("npu_w13_end_event"))
        prefetch_npu_w2_dev_ms = _elapsed_ms(
            next_prefetch_timing.get("npu_w2_start_event"),
            next_prefetch_timing.get("npu_w2_end_event"))
        prefetch_cpu_dev_ms = _elapsed_ms(
            next_prefetch_timing.get("cpu_start_event"),
            next_prefetch_timing.get("cpu_end_event"))
        prefetch_cpu_pack_dev_ms = _elapsed_ms(
            next_prefetch_timing.get("cpu_pack_start_event"),
            next_prefetch_timing.get("cpu_pack_end_event"))
        compute_dev_ms = _elapsed_ms(compute_start_event, compute_end_event)
        fused_dev_ms = _elapsed_ms(fused_start_event, fused_end_event)
        ready_wait_dev_ms = _elapsed_ms(
            bind_timing.get("ready_wait_start_event"),
            bind_timing.get("ready_wait_end_event"))
        logger.info(
            "Native common Mode3 timing fused-experts: rank=%s layer=%s "
            "stage=%s slot=%s valid_experts=%s source_from_npu=%s "
            "source_from_cpu=%s layer_local_buffer=%s bind_wait_us=%.1f "
            "bind_cpu_fill_us=%.1f bind_wait_mode=%s ready_wait_dev_ms=%.3f "
            "prefetch_status=%s prefetch_next_layer=%s prefetch_slot=%s "
            "prefetch_source_from_cpu=%s prefetch_cpu_path=%s "
            "prefetch_layer_local_buffer=%s prefetch_cpu_w13_pinned=%s "
            "prefetch_cpu_w2_pinned=%s prefetch_cpu_w13_contig=%s "
            "prefetch_cpu_w2_contig=%s prefetch_submit_us=%.1f "
            "submit_accounted_us=%.1f submit_event_alloc_us=%.1f "
            "submit_stream_wait_us=%.1f submit_prefetch_wait_stream_us=%.1f "
            "submit_start_event_record_us=%.1f submit_populate_us=%.1f "
            "submit_order_us=%.1f submit_assign_us=%.1f "
            "submit_layer_local_check_us=%.1f submit_npu_us=%.1f "
            "submit_cpu_us=%.1f submit_cpu_direct_async_us=%.1f "
            "submit_cpu_stage_async_us=%.1f submit_plan_log_us=%.1f "
            "submit_expert_map_us=%.1f submit_expert_map_cache_hit=%s "
            "submit_dispatch_cache_us=%.1f submit_slot_state_us=%.1f "
            "submit_post_cpu_wait_us=%.1f submit_ready_record_us=%.1f "
            "prefetch_dev_ms=%.3f prefetch_npu_dev_ms=%.3f "
            "prefetch_cpu_dev_ms=%.3f prefetch_cpu_pack_dev_ms=%.3f "
            "current_compute_wall_ms=%.3f current_compute_dev_ms=%.3f "
            "remap_wall_ms=%.3f fused_wall_ms=%.3f fused_dev_ms=%.3f "
            "prefetch_minus_compute_dev_ms=%.3f tokens=%s owned_per_rank=%s "
            "dispatch_num_experts=%s expert_token_nums_type=%s "
            "native_log2phy_dispatch=%s",
            getattr(layer, "rank", -1),
            layer.layer_idx,
            active_rank_count,
            int(bind_timing.get("slot_id", -1)),
            int(bound_slot.valid_expert_count),
            int(bound_slot.source_from_npu),
            int(bound_slot.source_from_cpu),
            int(getattr(bound_slot, "uses_layer_local_buffer", False)),
            float(bind_timing.get("wait_us", -1.0)),
            float(bind_timing.get("cpu_fill_us", -1.0)),
            bind_timing.get("wait_mode", "unknown"),
            ready_wait_dev_ms,
            next_prefetch_timing.get("status", "unknown"),
            next_prefetch_timing.get("layer_idx", -1),
            next_prefetch_timing.get("slot_id", -1),
            next_prefetch_timing.get("source_from_cpu", -1),
            next_prefetch_timing.get("cpu_path", "unknown"),
            next_prefetch_timing.get("layer_local_buffer", -1),
            int(_timing_float(next_prefetch_timing, "cpu_w13_pinned")),
            int(_timing_float(next_prefetch_timing, "cpu_w2_pinned")),
            int(_timing_float(next_prefetch_timing, "cpu_w13_contig")),
            int(_timing_float(next_prefetch_timing, "cpu_w2_contig")),
            float(next_prefetch_timing.get("submit_us", -1.0)),
            _mode3_submit_accounted_us(next_prefetch_timing),
            _timing_float(next_prefetch_timing, "submit_event_alloc_us"),
            _timing_float(next_prefetch_timing, "submit_stream_wait_us"),
            _timing_float(next_prefetch_timing,
                          "submit_prefetch_wait_stream_us"),
            _timing_float(next_prefetch_timing,
                          "submit_start_event_record_us"),
            _timing_float(next_prefetch_timing, "submit_populate_us"),
            _timing_float(next_prefetch_timing, "submit_order_us"),
            _timing_float(next_prefetch_timing, "submit_assign_us"),
            _timing_float(next_prefetch_timing, "submit_layer_local_check_us"),
            _timing_float(next_prefetch_timing, "submit_npu_us"),
            _timing_float(next_prefetch_timing, "submit_cpu_us"),
            _timing_float(next_prefetch_timing, "submit_cpu_direct_async_us"),
            _timing_float(next_prefetch_timing, "submit_cpu_stage_async_us"),
            _timing_float(next_prefetch_timing, "submit_plan_log_us"),
            _timing_float(next_prefetch_timing, "submit_expert_map_us"),
            int(_timing_float(next_prefetch_timing, "expert_map_cache_hit")),
            _timing_float(next_prefetch_timing, "submit_dispatch_cache_us"),
            _timing_float(next_prefetch_timing, "submit_slot_state_us"),
            _timing_float(next_prefetch_timing, "submit_post_cpu_wait_us"),
            _timing_float(next_prefetch_timing, "submit_ready_record_us"),
            prefetch_dev_ms,
            prefetch_npu_dev_ms,
            prefetch_cpu_dev_ms,
            prefetch_cpu_pack_dev_ms,
            compute_wall_ms,
            compute_dev_ms,
            remap_wall_ms,
            fused_wall_ms,
            fused_dev_ms,
            prefetch_dev_ms - compute_dev_ms
            if prefetch_dev_ms >= 0 and compute_dev_ms >= 0 else -1.0,
            int(x.shape[0]) if hasattr(x, "shape") else -1,
            owned_per_rank,
            dispatch_num_experts,
            manager.expert_token_nums_type,
            int(use_log2phy_dispatch),
        )
        if getattr(manager, "enable_copy_format_diag", False):
            logger.info(
                "Native common Mode3 copy timing detail fused-experts: rank=%s "
                "layer=%s prefetch_next_layer=%s slot=%s npu_total_ms=%.3f "
                "npu_w13_ms=%.3f npu_w2_ms=%.3f cpu_ms=%.3f "
                "cpu_pack_ms=%.3f",
                getattr(layer, "rank", -1),
                layer.layer_idx,
                next_prefetch_timing.get("layer_idx", -1),
                next_prefetch_timing.get("slot_id", -1),
                prefetch_npu_dev_ms,
                prefetch_npu_w13_dev_ms,
                prefetch_npu_w2_dev_ms,
                prefetch_cpu_dev_ms,
                prefetch_cpu_pack_dev_ms,
            )
    return final_hidden_states


def _execute_mode3_single_dispatch_hybrid(
        layer: torch.nn.Module,
        x: torch.Tensor,
        logical_topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        row_idx: torch.Tensor) -> torch.Tensor:
    manager = _get_or_create_mode3_double_buffer_manager(layer)
    if manager is None:
        raise RuntimeError(
            "Mode3 single-dispatch requires forward-context model instance "
            f"and moe prefetch stream at layer={getattr(layer, 'layer_idx', -1)}.")
    if getattr(manager, "use_fused_experts_path", True):
        return _execute_mode3_fused_experts_hybrid(
            layer=layer,
            x=x,
            logical_topk_ids=logical_topk_ids,
            topk_weights=topk_weights,
            row_idx=row_idx,
            manager=manager)

    bound_slot = manager.bind_current_layer(layer)
    manager.prefetch_next_layer(layer)
    moe_comm_method = get_forward_context().moe_comm_method
    token_dispatcher = getattr(moe_comm_method, "token_dispatcher", None)
    if token_dispatcher is None:
        raise RuntimeError("Missing MC2 token dispatcher for mode3 path.")
    if token_dispatcher.__class__.__name__ != "TokenDispatcherWithMC2":
        raise RuntimeError(
            "Mode3 single-dispatch requires MC2 token dispatcher, got "
            f"{token_dispatcher.__class__.__name__} at "
            f"layer={getattr(layer, 'layer_idx', -1)}.")

    local_rank_idx = int(getattr(layer, "lossless_hybrid_active_rank_index", -1))
    rank_owned = getattr(layer, "lossless_hybrid_rank_owned_expert_ids", None)
    if rank_owned is None or local_rank_idx < 0 or local_rank_idx >= len(rank_owned):
        raise RuntimeError(
            "Invalid hybrid rank ownership state for mode3 path at "
            f"layer={getattr(layer, 'layer_idx', -1)}.")
    local_owned_expert_ids = [int(expert_id) for expert_id in rank_owned[
        local_rank_idx]]
    active_rank_count = len(getattr(layer, "lossless_hybrid_active_ranks", []))
    owned_per_rank = len(local_owned_expert_ids)
    if active_rank_count <= 0 or owned_per_rank <= 0:
        raise RuntimeError(
            "Invalid hybrid mode3 dispatch shape: "
            f"active_rank_count={active_rank_count} "
            f"owned_per_rank={owned_per_rank}")

    dispatch_log2phy, dispatch_num_experts = _get_dispatch_log2phy_for_layer(
        layer,
        device=logical_topk_ids.device,
        rank_owned=rank_owned,
        active_rank_count=active_rank_count,
        owned_per_rank=owned_per_rank,
    )
    dispatch_topk_ids = dispatch_log2phy[logical_topk_ids]
    _verify_mode3_remapped_ids(dispatch_topk_ids, layer, "single-dispatch")

    old_dispatch_num_experts = int(getattr(token_dispatcher, "num_experts", 0))
    old_expert_token_nums_type = int(
        getattr(token_dispatcher, "expert_token_nums_type", 0))
    token_dispatcher.num_experts = dispatch_num_experts
    token_dispatcher.expert_token_nums_type = manager.expert_token_nums_type
    try:
        dispatch_results = token_dispatcher.token_dispatch(
            hidden_states=x,
            topk_weights=topk_weights,
            topk_ids=dispatch_topk_ids,
            row_idx=row_idx,
            expert_map=layer.expert_map,
            log2phy=None,
            global_redundant_expert_num=0,
            shared_experts=None,
            quantized_x_for_share=None,
            dynamic_scale_for_share=None,
            mc2_mask=getattr(moe_comm_method, "mc2_mask", None),
            apply_router_weight_on_input=False,
            with_quant=False)
        dispatched_hidden_states = dispatch_results["hidden_states"]
        dispatched_group_list = dispatch_results["group_list"]
        dispatched_group_list_type = int(dispatch_results["group_list_type"])
        dispatched_group_counts = _group_list_to_counts(
            dispatched_group_list, dispatched_group_list_type).to(dtype=torch.long)
        if int(dispatched_group_counts.numel()) == int(dispatch_num_experts):
            local_dense_start = local_rank_idx * owned_per_rank
            local_dense_end = local_dense_start + owned_per_rank
            dispatched_group_counts = dispatched_group_counts[
                local_dense_start:local_dense_end]
        if int(dispatched_group_counts.numel()) != int(owned_per_rank):
            raise RuntimeError(
                "Mode3 single-dispatch expert group size mismatch: "
                f"expected_experts={owned_per_rank} "
                f"actual_experts={int(dispatched_group_counts.numel())} "
                f"group_list_type={dispatched_group_list_type}")
        dispatched_output = unified_apply_mlp(
            hidden_states=dispatched_hidden_states,
            w1=layer.runtime_w13_weight,
            w1_scale=None,
            w2=layer.runtime_w2_weight,
            w2_scale=None,
            group_list=dispatched_group_counts.to(
                dtype=torch.int64, device=dispatched_hidden_states.device),
            dynamic_scale=None,
            group_list_type=1,
            w1_scale_bias=None,
            w2_scale_bias=None,
            topk_scales=None,
            with_quant=False,
            fusion=False,
            need_trans=False)
        final_hidden_states = token_dispatcher.token_combine(dispatched_output)
    finally:
        token_dispatcher.num_experts = old_dispatch_num_experts
        token_dispatcher.expert_token_nums_type = old_expert_token_nums_type

    layer.lossless_hybrid_last_stats = {
        "mode3_slot": int(layer.layer_idx) & 1,
        "valid_experts": int(bound_slot.valid_expert_count),
        "source_from_npu": int(bound_slot.source_from_npu),
        "source_from_cpu": int(bound_slot.source_from_cpu),
        "prefetch_wait_us": float(manager.prefetch_wait_us[int(layer.layer_idx)]),
        "prefetch_hit": int(manager.prefetch_hit[int(layer.layer_idx)]),
    }
    return final_hidden_states


def _forward_lossless_hybrid_waves(
        layer: torch.nn.Module,
        x: torch.Tensor,
        logical_topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        row_idx: torch.Tensor,
        activation: str,
        apply_router_weight_on_input: bool) -> Optional[torch.Tensor]:
    if not getattr(layer, "lossless_hybrid_active", False):
        return None
    plan_fn = getattr(layer, "_plan_lossless_hybrid_rank_waves", None)
    if not callable(plan_fn):
        return None
    if _should_use_mode3_single_rank_allgather_path(layer):
        return _execute_mode3_single_rank_allgather_hybrid(
            layer=layer,
            x=x,
            logical_topk_ids=logical_topk_ids,
            topk_weights=topk_weights,
            row_idx=row_idx)
    if _should_use_mode3_cross_layer_buffer_path(layer):
        return _execute_mode3_single_dispatch_hybrid(
            layer=layer,
            x=x,
            logical_topk_ids=logical_topk_ids,
            topk_weights=topk_weights,
            row_idx=row_idx)
    wave_plans = plan_fn(logical_topk_ids)
    if not wave_plans:
        return None

    final_hidden_states = torch.zeros_like(x)
    moe_comm_method = get_forward_context().moe_comm_method
    prepared_mc2_mask = getattr(moe_comm_method, "mc2_mask", None)
    use_dense_mc2_waves = (
        getattr(get_forward_context(), "moe_comm_type", None) == MoECommType.MC2)
    local_rank_idx = int(getattr(layer, "lossless_hybrid_active_rank_index", -1))
    active_rank_count = len(getattr(layer, "lossless_hybrid_active_ranks", []))
    resident_capacity = int(getattr(layer, "lossless_hybrid_resident_capacity", 0))
    if active_rank_count <= 0 or resident_capacity <= 0:
        raise RuntimeError(
            f"Invalid hybrid execution state at layer={layer.layer_idx}: "
            f"active_ranks={active_rank_count} resident_capacity={resident_capacity}")
    metadata_group = None
    if active_rank_count > 1:
        metadata_group = getattr(layer.ep_group, "cpu_group",
                                 layer.ep_group.device_group)
    local_wave_counts: list[int] = []
    local_cpu_miss = 0
    local_resident_hits = 0

    if _should_use_single_dispatch_hybrid_path(
            layer=layer,
            wave_plans=wave_plans,
            use_dense_mc2_waves=use_dense_mc2_waves):
        try:
            return _execute_mode2_single_dispatch_hybrid(
                layer=layer,
                x=x,
                logical_topk_ids=logical_topk_ids,
                topk_weights=topk_weights,
                row_idx=row_idx,
                wave_plans=wave_plans,
                prepared_mc2_mask=prepared_mc2_mask)
        except RuntimeError as err:
            if getattr(layer, "elastic_execution_mode", 0) == 3:
                raise
            if "Hybrid mode2 single-dispatch" not in str(err):
                raise
            logger.warning(
                "Native common hybrid MC2 single-dispatch fallback to "
                "per-wave MC2: layer=%s stage=%s reason=%s",
                getattr(layer, "layer_idx", -1),
                active_rank_count,
                err,
            )

    for wave_idx, wave_plan in enumerate(wave_plans):
        placeholder_only = False
        wave_mc2_mask = None
        global_wave_active_count = 0
        if use_dense_mc2_waves:
            if not wave_plan["wave_active_expert_ids"]:
                local_wave_counts.append(0)
                continue
            effective_token_mask = wave_plan["token_mask"]
            if prepared_mc2_mask is not None:
                mc2_active_mask = prepared_mc2_mask.to(
                    device=effective_token_mask.device, dtype=torch.bool)
                effective_token_mask = effective_token_mask & mc2_active_mask
            if effective_token_mask.numel() > 0 and torch.any(
                    effective_token_mask):
                active_token_count = int(
                    torch.count_nonzero(effective_token_mask).item())
            else:
                active_token_count = 0
            local_wave_counts.append(active_token_count)
            global_wave_active_count = active_token_count
            if metadata_group is not None:
                gathered_wave_counts: list[Optional[int]] = [
                    None
                ] * active_rank_count
                torch.distributed.all_gather_object(gathered_wave_counts,
                                                    int(active_token_count),
                                                    group=metadata_group)
                global_wave_active_count = max(
                    int(value) for value in gathered_wave_counts
                    if value is not None)
            if global_wave_active_count <= 0:
                continue
            forced_active_topk = int(wave_plan.get("wave_topk", 0) or 0)
            (token_indices, hidden_states, wave_ids, wave_weights,
             wave_row_idx, wave_mc2_mask, active_token_count) = (
                 _build_padded_mc2_wave_inputs(
                     hidden_states=x,
                     logical_topk_ids=logical_topk_ids,
                     topk_weights=topk_weights,
                     token_mask=effective_token_mask,
                     topk_mask=wave_plan["topk_mask"],
                     fallback_expert_id=int(
                         wave_plan["wave_active_expert_ids"][0]),
                     forced_active_topk=(forced_active_topk
                                         if forced_active_topk > 0 else None)))
            placeholder_only = (active_token_count == 0)
        else:
            token_indices, wave_ids, wave_weights, wave_row_idx = (
                _compact_wave_topk(logical_topk_ids, topk_weights,
                                   wave_plan["token_mask"],
                                   wave_plan["topk_mask"]))

        if 0 <= local_rank_idx < len(wave_plan["rank_resident"]):
            target_resident = list(wave_plan["rank_resident"][local_rank_idx])
            current_resident = list(
                getattr(layer, "lossless_hybrid_resident_expert_ids", []))
            local_cpu_miss += len([
                expert_id for expert_id in target_resident
                if expert_id not in set(current_resident)
            ])
            local_resident_hits += len([
                expert_id for expert_id in target_resident
                if expert_id in set(current_resident)
            ])
            layer.materialize_hybrid_resident_experts(target_resident)

        logical_num_experts = int(layer.elastic_original_num_experts)
        wave_log2phy = torch.full((logical_num_experts, ),
                                  -1,
                                  dtype=torch.int32,
                                  device=logical_topk_ids.device)
        for expert_id, slot in wave_plan["wave_expert_to_slot"].items():
            wave_log2phy[int(expert_id)] = int(slot)

        if not use_dense_mc2_waves:
            placeholder_only = (token_indices is None or wave_ids is None
                                or wave_weights is None
                                or wave_row_idx is None)
            if placeholder_only:
                if not wave_plan["wave_active_expert_ids"]:
                    local_wave_counts.append(0)
                    continue
                placeholder_expert = int(wave_plan["wave_active_expert_ids"][0])
                placeholder_topk = int(logical_topk_ids.shape[1])
                hidden_states = x[:1].clone()
                hidden_states.zero_()
                wave_ids = torch.full((1, placeholder_topk),
                                      placeholder_expert,
                                      dtype=logical_topk_ids.dtype,
                                      device=logical_topk_ids.device)
                wave_weights = torch.zeros((1, placeholder_topk),
                                           dtype=topk_weights.dtype,
                                           device=topk_weights.device)
                wave_row_idx = return_row_idx(wave_ids, placeholder_topk)
                local_wave_counts.append(0)
            else:
                hidden_states = x.index_select(0, token_indices)
                local_wave_counts.append(int(token_indices.numel()))
        if prepared_mc2_mask is not None and not use_dense_mc2_waves:
            if placeholder_only:
                wave_mc2_mask = torch.zeros_like(prepared_mc2_mask[:1])
            else:
                wave_mc2_mask = prepared_mc2_mask.index_select(0, token_indices)

        if use_dense_mc2_waves and layer.layer_idx == 0:
            logger.info(
                "Native common hybrid MC2 wave launch: rank=%s layer=%s "
                "stage=%s wave=%s/%s launch_tokens=%s active_tokens=%s "
                "global_active_tokens=%s topk_width=%s active_experts=%s",
                getattr(layer, "rank", -1),
                layer.layer_idx,
                active_rank_count,
                wave_idx + 1,
                len(wave_plans),
                int(hidden_states.shape[0]),
                int(active_token_count),
                int(global_wave_active_count),
                int(wave_ids.shape[1]),
                len(wave_plan["wave_active_expert_ids"]),
            )

        wave_output = _execute_lossless_hybrid_wave(
            layer=layer,
            hidden_states=hidden_states,
            topk_ids=wave_ids,
            topk_weights=wave_weights,
            row_idx=wave_row_idx,
            global_num_experts=active_rank_count * resident_capacity,
            log2phy=wave_log2phy,
            activation=activation,
            apply_router_weight_on_input=apply_router_weight_on_input,
            mc2_mask=wave_mc2_mask)
        if use_dense_mc2_waves:
            if (not placeholder_only and token_indices is not None
                    and token_indices.numel() > 0):
                final_hidden_states.index_add_(0, token_indices,
                                               wave_output[:active_token_count])
            if len(wave_plans) > 1 and hasattr(torch, "npu"):
                torch.npu.synchronize()
        elif not placeholder_only:
            final_hidden_states.index_add_(0, token_indices, wave_output)

    layer.lossless_hybrid_rank_resident_expert_ids = [
        list(expert_ids) for expert_ids in wave_plans[-1]["final_rank_resident"]
    ]
    layer.lossless_hybrid_rank_lru = [
        list(expert_ids) for expert_ids in wave_plans[-1]["final_rank_resident"]
    ]
    if 0 <= local_rank_idx < len(wave_plans[-1]["final_rank_resident"]):
        if use_dense_mc2_waves and len(wave_plans) > 1 and hasattr(torch, "npu"):
            torch.npu.synchronize()
        layer.materialize_hybrid_resident_experts(
            list(wave_plans[-1]["final_rank_resident"][local_rank_idx]))
    layer.lossless_hybrid_last_stats = {
        "waves": len(wave_plans),
        "local_wave_tokens": local_wave_counts,
        "local_cpu_miss": local_cpu_miss,
        "local_resident_hits": local_resident_hits,
    }
    if layer.layer_idx == 0 and wave_plans:
        logger.info(
            "Native common hybrid MoE wave execution: rank=%s layer=%s "
            "stage=%s waves=%s resident_hits=%s cpu_miss=%s "
            "local_wave_tokens=%s resident=%s",
            getattr(layer, "rank", -1),
            layer.layer_idx,
            active_rank_count,
            len(wave_plans),
            local_resident_hits,
            local_cpu_miss,
            local_wave_counts[:8],
            getattr(layer, "lossless_hybrid_resident_expert_ids", [])[:8],
        )
    return final_hidden_states


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
    runtime_log2phy = getattr(layer, "elastic_runtime_log2phy", None)
    if runtime_log2phy is None and getattr(layer, "dynamic_eplb", False):
        runtime_log2phy = getattr(layer, "log2phy", None)
    hybrid_output = _forward_lossless_hybrid_waves(
        layer=layer,
        x=x,
        logical_topk_ids=topk_ids,
        topk_weights=topk_weights,
        row_idx=row_idx,
        activation=activation,
        apply_router_weight_on_input=apply_router_weight_on_input)
    if hybrid_output is not None:
        return hybrid_output

    runtime_w13_weight = getattr(layer, "runtime_w13_weight", None)
    runtime_w2_weight = getattr(layer, "runtime_w2_weight", None)
    if runtime_w13_weight is not None and runtime_w2_weight is not None:
        active_w13_weight = runtime_w13_weight
        active_w2_weight = runtime_w2_weight
    else:
        active_w13_weight = _active_moe_weight_view(layer, layer.w13_weight)
        active_w2_weight = _active_moe_weight_view(layer, layer.w2_weight)
    return moe_comm_method.fused_experts(hidden_states=x,
                                         w1=active_w13_weight,
                                         w2=active_w2_weight,
                                         topk_weights=topk_weights,
                                         topk_ids=topk_ids,
                                         row_idx=row_idx,
                                         global_num_experts=global_num_experts,
                                         expert_map=expert_map,
                                         log2phy=runtime_log2phy,
                                         global_redundant_expert_num=getattr(
                                             layer,
                                             "global_redundant_expert_num", 0))


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


class AscendFusedMoE(FusedMoE):
    moe_counter = -1

    def __init__(self, *args, **kwargs):
        kwargs = _ensure_layer_idx_kwarg(args, kwargs)
        logical_num_experts = _get_num_experts_arg(args, kwargs)
        ascend_config = get_ascend_config()
        elastic_execution_mode = envs_ascend.VLLM_ASCEND_ELASTIC_EXECUTION_MODE
        elastic_moe_mode = ascend_config.elastic_moe_mode
        init_redundancy = 0
        if (logical_num_experts is not None
                and elastic_moe_mode == "lossless"
                and elastic_execution_mode == 1):
            ep_group = get_ep_group()
            initial_ep_size = int(ep_group.world_size)
            ep_rank = int(ep_group.rank_in_group)
            init_redundancy = envs_ascend.compute_elastic_init_redundancy_expert(
                logical_num_experts,
                initial_ep_size,
                ascend_config.init_redundancy_expert,
            )
            if init_redundancy > 0:
                loaded_local_num_experts, _ = determine_redundant_replica_expert_map(
                    logical_num_experts, initial_ep_size, ep_rank,
                    init_redundancy)
                floor_capacity = int(loaded_local_num_experts)
                min_compute_group_size = (
                    envs_ascend.VLLM_ASCEND_ELASTIC_MIN_COMPUTE_GROUP_SIZE)
                if (min_compute_group_size is not None
                        and int(min_compute_group_size) > 0
                        and initial_ep_size % int(min_compute_group_size) == 0
                        and logical_num_experts % int(min_compute_group_size)
                        == 0):
                    floor_capacity = max(
                        floor_capacity,
                        logical_num_experts // int(min_compute_group_size))
                kwargs = dict(kwargs)
                kwargs["num_local_expert_weight_slots"] = max(
                    int(kwargs.get("num_local_expert_weight_slots", 0) or 0),
                    int(floor_capacity))
        super().__init__(*args, **kwargs)

        AscendFusedMoE.moe_counter += 1
        self.moe_instance_id = AscendFusedMoE.moe_counter
        vllm_config = get_current_vllm_config()
        self.model_type = (
            vllm_config.model_config.hf_config.model_type
            if vllm_config.model_config is not None else "")
        self.moe_config.tp_group = get_tp_group()
        self.moe_config.dp_group = get_dp_group()
        self.moe_config.ep_group = get_ep_group()
        self.moe_config.mc2_group = get_mc2_group()
        logical_num_experts = int(logical_num_experts
                                  if logical_num_experts is not None else
                                  self.global_num_experts)
        self.num_experts = logical_num_experts
        self.elastic_original_num_experts = logical_num_experts
        self.elastic_execution_mode = elastic_execution_mode
        self.elastic_moe_mode = elastic_moe_mode
        self.loaded_expert_map = None
        self.elastic_original_expert_map = None
        self.active_expert_mask = None
        self.elastic_runtime_log2phy = None
        self.primary_log2phy = None
        self.lossless_runtime_activated = False
        self.runtime_w13_weight = None
        self.runtime_w2_weight = None
        self.runtime_w13_buffer = None
        self.runtime_w2_buffer = None
        self.runtime_weight_capacity = 0
        self.lossless_mode1_native_parity_ready = False
        self.lossless_cpu_w13_weight = None
        self.lossless_cpu_w2_weight = None
        self.lossless_cpu_shadow_local_slots: dict[int, int] = {}
        self.lossless_loaded_offloaded = False
        self.lossless_cpu_import_expert_ids: list[int] = []
        self.lossless_saved_primary_prefix_w13 = None
        self.lossless_saved_primary_prefix_w2 = None
        self.lossless_hybrid_reuse_log_budget = 0
        self.lossless_hybrid_fallback_logged = False
        self.lossless_primary_prefix_stash_logged = False
        self.lossless_primary_prefix_restore_logged = False
        self.lossless_hybrid_cpu_swap_enabled = (
            self.elastic_moe_mode == "lossless"
            and self.elastic_execution_mode in (2, 3))
        self.lossless_hybrid_active = False
        self.lossless_hybrid_resident_capacity = 0
        self.lossless_hybrid_owned_expert_ids: list[int] = []
        self.lossless_hybrid_resident_expert_ids: list[int] = []
        self.lossless_hybrid_cpu_only_expert_ids: list[int] = []
        self.lossless_hybrid_active_ranks: list[int] = []
        self.lossless_hybrid_active_rank_index = -1
        self.lossless_hybrid_rank_owned_expert_ids: list[list[int]] = []
        self.lossless_hybrid_rank_resident_expert_ids: list[list[int]] = []
        self.lossless_hybrid_rank_lru: list[list[int]] = []
        self.lossless_hybrid_owner_rank_by_expert = None
        self.lossless_hybrid_owner_global_rank_by_expert = None
        self.lossless_hybrid_last_stats: dict[str, Any] = {}
        self.lossless_mode3_primary_prefix_expert_ids: list[int] = []
        self.lossless_mode3_primary_prefix_local_slots: dict[int, int] = {}
        self.loaded_weight_capacity = int(
            getattr(self, "local_num_expert_weight_slots",
                    self.local_num_experts))
        self.dynamic_eplb = ascend_config.dynamic_eplb
        self.expert_map_path = ascend_config.expert_map_path
        self.global_redundant_expert_num = (
            init_redundancy if self.elastic_moe_mode == "lossless" else
            ascend_config.init_redundancy_expert)
        if self.elastic_moe_mode == "lossless":
            self.global_num_experts = logical_num_experts
            self.moe_config.num_experts = logical_num_experts
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
            primary_local_num_experts, primary_expert_map, primary_log2phy = (
                _parse_expert_map_result(
                    determine_expert_map(
                        self.ep_size,
                        self.ep_rank,
                        logical_num_experts,
                        layer_idx=self.layer_idx)))
            if primary_log2phy is None:
                primary_log2phy = determine_default_log2phy_map(
                    logical_num_experts, self.ep_size, self.ep_rank, 0)
            if self.elastic_moe_mode == "lossless":
                if self.global_redundant_expert_num > 0:
                    self.loaded_local_num_experts, self.loaded_expert_map = (
                        determine_redundant_replica_expert_map(
                            logical_num_experts, self.ep_size, self.ep_rank,
                            self.global_redundant_expert_num))
                else:
                    self.loaded_local_num_experts = primary_local_num_experts
                    self.loaded_expert_map = primary_expert_map.clone()
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
        if self.loaded_expert_map is None and self.expert_map is not None:
            self.loaded_expert_map = self.expert_map.clone()
        if self.elastic_original_expert_map is None and self.expert_map is not None:
            self.elastic_original_expert_map = self.expert_map.clone()
        if self.primary_log2phy is None and self.log2phy is not None:
            self.primary_log2phy = self.log2phy.clone()
        self.elastic_original_ep_size = int(
            getattr(self.moe_parallel_config, "ep_size",
                    getattr(self, "ep_size", 1)))
        map_device = (self.w13_weight.device
                      if hasattr(self, "w13_weight") else None)
        if map_device is not None and map_device.type != "cpu":
            for attr_name in ("expert_map", "loaded_expert_map", "log2phy",
                              "primary_log2phy", "elastic_original_expert_map"):
                value = getattr(self, attr_name, None)
                if value is not None:
                    setattr(self, attr_name,
                            value.to(device=map_device, dtype=torch.int32))
        self.loaded_weight_capacity = max(
            int(self.loaded_weight_capacity),
            int(getattr(self, "loaded_local_num_experts", self.local_num_experts)),
            int(self._get_lossless_loaded_slot_capacity()))
        self.moe_config.num_local_experts = int(
            getattr(self, "active_local_num_experts", self.local_num_experts))
        self.moe_config.num_global_redundant_experts = (
            self.global_redundant_expert_num)
        local_num_experts = (torch.sum(
            self.expert_map != -1) if self.expert_map is not None else
                             self.global_num_experts)
        if self.dynamic_eplb:
            self.moe_load = torch.zeros(local_num_experts, dtype=torch.int64)

        setup_moe_comm_method(self.moe_config)
        self._refresh_mode1_native_parity_ready("init", log=True)

    def update_expert_map(self, new_expert_map):
        self.expert_map = new_expert_map

    def _map_global_expert_id_to_local_expert_id(self, expert_id: int) -> int:
        if self.elastic_moe_mode == "lossless" and self.loaded_expert_map is not None:
            return self.loaded_expert_map[expert_id].item()
        return super()._map_global_expert_id_to_local_expert_id(expert_id)

    def get_map(self):
        return self.expert_map

    def get_log2phy_map(self):
        return self.log2phy

    def set_runtime_num_experts(self, num_experts: int) -> None:
        runtime_num_experts = int(num_experts)
        self.moe_config.num_experts = runtime_num_experts
        self.num_experts = runtime_num_experts

    @staticmethod
    def _is_power_of_two(value: int) -> bool:
        return value > 0 and (value & (value - 1)) == 0

    def _get_elastic_initial_ep_size(self) -> int:
        return int(
            getattr(self, "elastic_original_ep_size",
                    getattr(self.moe_parallel_config, "ep_size", 1)))

    def _get_configured_elastic_min_compute_group_size(self) -> Optional[int]:
        min_compute_group_size = (
            envs_ascend.VLLM_ASCEND_ELASTIC_MIN_COMPUTE_GROUP_SIZE)
        if min_compute_group_size is None:
            return None
        initial_ep_size = self._get_elastic_initial_ep_size()
        if initial_ep_size <= 0:
            raise ValueError(
                f"Invalid initial EP size {initial_ep_size} at layer={self.layer_idx}.")
        if not self._is_power_of_two(initial_ep_size):
            raise ValueError(
                "Elastic repeated shrink only supports power-of-two initial "
                f"EP size, got {initial_ep_size} at layer={self.layer_idx}.")
        if min_compute_group_size > initial_ep_size:
            raise ValueError(
                "VLLM_ASCEND_ELASTIC_MIN_COMPUTE_GROUP_SIZE cannot exceed "
                f"the initial EP size: {min_compute_group_size} > "
                f"{initial_ep_size} at layer={self.layer_idx}.")
        if initial_ep_size % min_compute_group_size != 0:
            raise ValueError(
                "VLLM_ASCEND_ELASTIC_MIN_COMPUTE_GROUP_SIZE must divide the "
                f"initial EP size: {min_compute_group_size} vs "
                f"{initial_ep_size} at layer={self.layer_idx}.")
        return min_compute_group_size

    def _get_default_single_shrink_ep_floor(self) -> int:
        initial_ep_size = self._get_elastic_initial_ep_size()
        if initial_ep_size <= 1:
            return 1
        return max(1, initial_ep_size // 2)

    def _get_logical_num_experts_for_elastic(self) -> int:
        logical_num_experts = int(
            getattr(self, "elastic_original_num_experts",
                    getattr(self, "global_num_experts",
                            getattr(self, "num_experts", 0))))
        if logical_num_experts <= 0:
            raise ValueError(
                f"Invalid logical num_experts={logical_num_experts} "
                f"at layer={self.layer_idx}.")
        return logical_num_experts

    def _get_reserved_local_expert_slots_for_floor(self,
                                                   min_ep_size: int) -> int:
        logical_num_experts = self._get_logical_num_experts_for_elastic()
        if logical_num_experts % min_ep_size != 0:
            raise ValueError(
                f"logical_num_experts={logical_num_experts} is not divisible "
                f"by min_ep_size={min_ep_size} at layer={self.layer_idx}.")
        return logical_num_experts // min_ep_size

    def _get_lossless_loaded_slot_capacity(self) -> int:
        loaded_local_num_experts = int(
            getattr(self, "loaded_local_num_experts",
                    getattr(self, "active_local_num_experts", 0)) or 0)
        if self.elastic_moe_mode != "lossless":
            return loaded_local_num_experts
        initial_ep_size = self._get_elastic_initial_ep_size()
        min_compute_group_size = (
            self._get_configured_elastic_min_compute_group_size())
        if (initial_ep_size <= 1 or min_compute_group_size is None
                or int(min_compute_group_size) >= initial_ep_size):
            return loaded_local_num_experts
        if self._is_hybrid_cpu_swap_enabled():
            return max(loaded_local_num_experts,
                       self._get_hybrid_resident_capacity())
        return max(
            loaded_local_num_experts,
            self._get_reserved_local_expert_slots_for_floor(
                int(min_compute_group_size)))

    def _is_hybrid_cpu_swap_enabled(self) -> bool:
        if not getattr(self, "lossless_hybrid_cpu_swap_enabled", False):
            return False
        if self.elastic_moe_mode != "lossless":
            return False
        if getattr(self, "elastic_execution_mode", 0) == 1:
            return False
        return True

    def _refresh_mode1_native_parity_ready(self,
                                           reason: str,
                                           log: bool = False) -> bool:
        active_local = int(
            getattr(self, "active_local_num_experts",
                    getattr(self, "local_num_experts", 0)) or 0)
        loaded_local = int(
            getattr(self, "loaded_local_num_experts", active_local) or 0)
        loaded_capacity = int(
            getattr(self, "loaded_weight_capacity",
                    getattr(self, "local_num_expert_weight_slots",
                            loaded_local)) or 0)
        floor = self._get_configured_elastic_min_compute_group_size()
        hybrid_disabled = not self._is_hybrid_cpu_swap_enabled()
        runtime_buffers_disabled = (
            getattr(self, "runtime_w13_buffer", None) is None
            and getattr(self, "runtime_w2_buffer", None) is None
            and getattr(self, "runtime_w13_weight", None) is None
            and getattr(self, "runtime_w2_weight", None) is None)
        ready = (
            self.elastic_moe_mode == "lossless"
            and int(getattr(self, "elastic_execution_mode", 0) or 0) == 1
            and int(getattr(self, "global_redundant_expert_num", 0) or 0) > 0
            and hybrid_disabled
            and runtime_buffers_disabled
            and active_local > 0
            and loaded_local >= active_local
            and loaded_capacity >= loaded_local)
        self.lossless_mode1_native_parity_ready = ready
        if log:
            logger.info(
                "Native common mode1 parity state: reason=%s layer=%s mode=%s "
                "floor=%s active_local=%s loaded_local=%s loaded_capacity=%s "
                "hybrid_disabled=%s runtime_buffers_disabled=%s parity_ready=%s",
                reason,
                getattr(self, "layer_idx", -1),
                getattr(self, "elastic_execution_mode", None),
                floor,
                active_local,
                loaded_local,
                loaded_capacity,
                hybrid_disabled,
                runtime_buffers_disabled,
                ready,
            )
        return ready

    def _is_mode3_cross_layer_buffer_enabled(self) -> bool:
        return (self._is_hybrid_cpu_swap_enabled()
                and int(getattr(self, "elastic_execution_mode", 0)) == 3)

    def _get_hybrid_resident_capacity(self) -> int:
        native_capacity = min(int(self.w13_weight.shape[0]),
                              int(self.w2_weight.shape[0]))
        configured_resident_slots = (
            envs_ascend.VLLM_ASCEND_ELASTIC_HYBRID_RESIDENT_EXPERT_SLOTS)
        if configured_resident_slots is not None:
            return max(1, min(native_capacity, int(configured_resident_slots)))
        if getattr(self, "elastic_execution_mode", 0) == 3:
            return max(1, min(native_capacity,
                              self._get_lossless_primary_prefix_row_count()))
        primary_prefix_rows = self._get_lossless_primary_prefix_row_count()
        return max(1, min(native_capacity, primary_prefix_rows))

    def _hybrid_requires_multi_wave_execution(self) -> bool:
        if self._is_mode3_cross_layer_buffer_enabled():
            return False
        if not self.lossless_hybrid_active:
            return False
        resident_capacity = int(self.lossless_hybrid_resident_capacity)
        if resident_capacity <= 0:
            return False
        rank_owned_expert_ids = getattr(self,
                                        "lossless_hybrid_rank_owned_expert_ids",
                                        None)
        if rank_owned_expert_ids:
            return any(
                len(expert_ids) > resident_capacity
                for expert_ids in rank_owned_expert_ids)
        return len(getattr(self, "lossless_hybrid_owned_expert_ids",
                           [])) > resident_capacity

    def should_activate_lossless_hybrid_for_target(
            self, target_owned_local_expert_count: int,
            active_rank_count: int) -> bool:
        if not self._is_hybrid_cpu_swap_enabled():
            return False
        target_owned_local_expert_count = int(target_owned_local_expert_count)
        if target_owned_local_expert_count <= 0:
            return False
        resident_capacity = self._get_hybrid_resident_capacity()
        return target_owned_local_expert_count > resident_capacity

    def should_offload_loaded_weights_after_lossless_activation(self) -> bool:
        return False

    def clear_lossless_hybrid_state(self) -> None:
        self.lossless_hybrid_active = False
        self.lossless_hybrid_resident_capacity = 0
        self.lossless_hybrid_owned_expert_ids = []
        self.lossless_hybrid_resident_expert_ids = []
        self.lossless_hybrid_cpu_only_expert_ids = []
        self.lossless_hybrid_active_ranks = []
        self.lossless_hybrid_active_rank_index = -1
        self.lossless_hybrid_rank_owned_expert_ids = []
        self.lossless_hybrid_rank_resident_expert_ids = []
        self.lossless_hybrid_rank_lru = []
        self.lossless_hybrid_owner_rank_by_expert = None
        self.lossless_hybrid_owner_global_rank_by_expert = None
        self.lossless_hybrid_last_stats = {}
        self.lossless_mode3_primary_prefix_expert_ids = []
        self.lossless_mode3_primary_prefix_local_slots = {}
        self.lossless_cpu_shadow_local_slots = {}
        self.lossless_cpu_import_expert_ids = []
        self._mode3_hybrid_inactive_logged = False
        self._mode3_gate_reject_logged = False
        self._mode3_refresh_skip_logged = False

    def _can_reuse_loaded_prefix_for_hybrid(
            self, resident_capacity: int, runtime_device: torch.device,
            runtime_w13_dtype: torch.dtype,
            runtime_w2_dtype: torch.dtype) -> bool:
        if self.lossless_loaded_offloaded:
            return False
        if resident_capacity <= 0:
            return False
        if int(self.w13_weight.shape[0]) < resident_capacity:
            return False
        if int(self.w2_weight.shape[0]) < resident_capacity:
            return False
        if self.w13_weight.device != runtime_device:
            return False
        if self.w2_weight.device != runtime_device:
            return False
        if self.w13_weight.dtype != runtime_w13_dtype:
            return False
        if self.w2_weight.dtype != runtime_w2_dtype:
            return False
        return True

    def _get_lossless_primary_prefix_row_count(self) -> int:
        primary_expert_map = getattr(self, "elastic_original_expert_map", None)
        if primary_expert_map is not None:
            return int((primary_expert_map != -1).sum().item())
        initial_ep_size = self._get_elastic_initial_ep_size()
        logical_num_experts = self._get_logical_num_experts_for_elastic()
        if initial_ep_size > 0 and logical_num_experts % initial_ep_size == 0:
            return logical_num_experts // initial_ep_size
        return int(getattr(self, "active_local_num_experts", 0))

    def _stash_lossless_primary_prefix_for_hybrid(self) -> None:
        if self.lossless_loaded_offloaded:
            return
        row_count = min(self._get_lossless_primary_prefix_row_count(),
                        int(self.w13_weight.shape[0]),
                        int(self.w2_weight.shape[0]))
        if row_count <= 0:
            return
        self.lossless_saved_primary_prefix_w13 = self.w13_weight[
            :row_count].detach().cpu().clone()
        self.lossless_saved_primary_prefix_w2 = self.w2_weight[
            :row_count].detach().cpu().clone()
        if self.layer_idx == 0 and not self.lossless_primary_prefix_stash_logged:
            logger.info(
                "Native common lossless primary prefix stashed for hybrid: "
                "layer=%s rows=%s resident_capacity=%s",
                self.layer_idx,
                row_count,
                self._get_hybrid_resident_capacity(),
            )
            self.lossless_primary_prefix_stash_logged = True

    def _restore_stashed_lossless_primary_prefix(self) -> None:
        saved_w13 = getattr(self, "lossless_saved_primary_prefix_w13", None)
        saved_w2 = getattr(self, "lossless_saved_primary_prefix_w2", None)
        if saved_w13 is None or saved_w2 is None:
            return
        row_count = min(int(saved_w13.shape[0]), int(saved_w2.shape[0]),
                        int(self.w13_weight.shape[0]),
                        int(self.w2_weight.shape[0]))
        if row_count <= 0:
            return
        self.w13_weight[:row_count].copy_(saved_w13.to(
            device=self.w13_weight.device, dtype=self.w13_weight.dtype),
                                          non_blocking=False)
        self.w2_weight[:row_count].copy_(saved_w2.to(
            device=self.w2_weight.device, dtype=self.w2_weight.dtype),
                                         non_blocking=False)
        if self.layer_idx == 0 and not self.lossless_primary_prefix_restore_logged:
            logger.info(
                "Native common lossless primary prefix restored: layer=%s rows=%s",
                self.layer_idx,
                row_count,
            )
            self.lossless_primary_prefix_restore_logged = True

    def _get_lossless_cpu_pair_for_local_slot(
            self, local_slot: int) -> tuple[torch.Tensor, torch.Tensor]:
        local_slot = int(local_slot)
        if local_slot < 0:
            raise RuntimeError(
                f"Invalid local_slot={local_slot} at layer={self.layer_idx}.")
        if (self.lossless_cpu_w13_weight is not None
                and self.lossless_cpu_w2_weight is not None
                and local_slot < int(self.lossless_cpu_w13_weight.shape[0])
                and local_slot < int(self.lossless_cpu_w2_weight.shape[0])):
            return (self.lossless_cpu_w13_weight[local_slot].detach().cpu(),
                    self.lossless_cpu_w2_weight[local_slot].detach().cpu())
        if (local_slot < int(self.w13_weight.shape[0])
                and local_slot < int(self.w2_weight.shape[0])):
            return (self.w13_weight[local_slot].detach().cpu(),
                    self.w2_weight[local_slot].detach().cpu())
        raise RuntimeError(
            f"Cannot resolve CPU shadow for local_slot={local_slot} "
            f"at layer={self.layer_idx}.")

    def _build_lossless_cpu_shadow_for_owned_layout(
            self,
            active_expert_ids: list[int],
            source_local_ids: list[int],
            cpu_expert_weights: Optional[dict[int, tuple[torch.Tensor,
                                                         torch.Tensor]]]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if len(active_expert_ids) != len(source_local_ids):
            raise RuntimeError(
                f"Owned expert ids/source ids length mismatch at layer={self.layer_idx}: "
                f"{len(active_expert_ids)} vs {len(source_local_ids)}")
        if not active_expert_ids:
            empty_w13 = torch.empty((0, ) + tuple(self.w13_weight.shape[1:]),
                                    device="cpu",
                                    dtype=self.w13_weight.dtype)
            empty_w2 = torch.empty((0, ) + tuple(self.w2_weight.shape[1:]),
                                   device="cpu",
                                   dtype=self.w2_weight.dtype)
            return empty_w13, empty_w2

        cpu_w13_rows = []
        cpu_w2_rows = []
        for expert_id, source_local_id in zip(active_expert_ids,
                                              source_local_ids):
            if int(source_local_id) >= 0:
                source_w13, source_w2 = self._get_lossless_cpu_pair_for_local_slot(
                    int(source_local_id))
            else:
                if (cpu_expert_weights is None
                        or int(expert_id) not in cpu_expert_weights):
                    raise RuntimeError(
                        f"Missing CPU import payload for expert_id={int(expert_id)} "
                        f"at layer={self.layer_idx}.")
                source_w13, source_w2 = cpu_expert_weights[int(expert_id)]
            cpu_w13_rows.append(source_w13.to(device="cpu", copy=True))
            cpu_w2_rows.append(source_w2.to(device="cpu", copy=True))
        cpu_w13 = torch.stack(cpu_w13_rows, dim=0)
        cpu_w2 = torch.stack(cpu_w2_rows, dim=0)
        return _maybe_pin_cpu_tensor(cpu_w13), _maybe_pin_cpu_tensor(cpu_w2)

    def _set_lossless_hybrid_runtime_num_experts(self) -> None:
        active_rank_count = len(self.lossless_hybrid_active_ranks)
        if active_rank_count <= 0:
            return
        resident_capacity = int(self.lossless_hybrid_resident_capacity)
        self.set_runtime_num_experts(active_rank_count * resident_capacity)

    def _build_lossless_hybrid_runtime_log2phy(
            self,
            rank_resident_expert_ids: Optional[list[list[int]]] = None
    ) -> Optional[torch.Tensor]:
        if rank_resident_expert_ids is None:
            rank_resident_expert_ids = self.lossless_hybrid_rank_resident_expert_ids
        if not rank_resident_expert_ids:
            return None
        logical_num_experts = int(self.elastic_original_num_experts)
        resident_capacity = int(self.lossless_hybrid_resident_capacity)
        runtime_log2phy_cpu = torch.full((logical_num_experts, ),
                                         -1,
                                         dtype=torch.int32)
        for rank_idx, expert_ids in enumerate(rank_resident_expert_ids):
            for slot_idx, expert_id in enumerate(expert_ids[:resident_capacity]):
                expert_id = int(expert_id)
                if int(runtime_log2phy_cpu[expert_id].item()) != -1:
                    continue
                runtime_log2phy_cpu[expert_id] = (
                    rank_idx * resident_capacity + slot_idx)
        return runtime_log2phy_cpu

    @staticmethod
    def _pad_hybrid_resident_ids(expert_ids: list[int],
                                 resident_capacity: int) -> list[int]:
        resident = [int(expert_id) for expert_id in expert_ids[:resident_capacity]]
        if len(resident) >= resident_capacity or resident_capacity <= 0:
            return resident
        if not resident:
            raise RuntimeError("Hybrid resident padding requires at least one expert id.")
        filler = resident[0]
        resident.extend([filler] * (resident_capacity - len(resident)))
        return resident

    def set_lossless_hybrid_global_layout(
            self, active_ranks: list[int],
            ordered_assignments: list[list[tuple[int, int]]]) -> None:
        resident_capacity = self._get_hybrid_resident_capacity()
        rank_owned_expert_ids = [[int(expert_id) for expert_id, _ in assignments]
                                 for assignments in ordered_assignments]
        rank_resident_expert_ids = [
            self._pad_hybrid_resident_ids(expert_ids, resident_capacity)
            for expert_ids in rank_owned_expert_ids
        ]
        owner_rank_by_expert = torch.full((int(self.elastic_original_num_experts), ),
                                          -1,
                                          dtype=torch.int32)
        owner_global_rank_by_expert = torch.full(
            (int(self.elastic_original_num_experts), ),
            -1,
            dtype=torch.int32)
        for rank_idx, expert_ids in enumerate(rank_owned_expert_ids):
            for expert_id in expert_ids:
                owner_rank_by_expert[int(expert_id)] = rank_idx
                owner_global_rank_by_expert[int(expert_id)] = int(
                    active_ranks[rank_idx])
        self.lossless_hybrid_active = True
        self.lossless_hybrid_resident_capacity = resident_capacity
        self.lossless_hybrid_active_ranks = [int(rank) for rank in active_ranks]
        current_global_rank = (torch.distributed.get_rank()
                               if torch.distributed.is_initialized() else -1)
        self.lossless_hybrid_active_rank_index = (
            self.lossless_hybrid_active_ranks.index(current_global_rank)
            if current_global_rank in self.lossless_hybrid_active_ranks else -1)
        self.lossless_hybrid_rank_owned_expert_ids = rank_owned_expert_ids
        self.lossless_hybrid_rank_resident_expert_ids = [
            list(expert_ids) for expert_ids in rank_resident_expert_ids
        ]
        self.lossless_hybrid_rank_lru = [
            list(expert_ids) for expert_ids in rank_resident_expert_ids
        ]
        target_device = (self.log2phy.device if self.log2phy is not None else
                         self.expert_map.device if self.expert_map is not None else
                         None)
        if target_device is not None:
            self.lossless_hybrid_owner_rank_by_expert = owner_rank_by_expert.to(
                device=target_device, dtype=torch.int32)
            self.lossless_hybrid_owner_global_rank_by_expert = (
                owner_global_rank_by_expert.to(device=target_device,
                                               dtype=torch.int32))
        else:
            self.lossless_hybrid_owner_rank_by_expert = owner_rank_by_expert
            self.lossless_hybrid_owner_global_rank_by_expert = (
                owner_global_rank_by_expert)
        self._set_lossless_hybrid_runtime_num_experts()
        runtime_log2phy_cpu = self._build_lossless_hybrid_runtime_log2phy(
            self.lossless_hybrid_rank_resident_expert_ids)
        if runtime_log2phy_cpu is not None:
            self.set_elastic_runtime_log2phy(
                runtime_log2phy_cpu.to(device=target_device
                                       if target_device is not None else
                                       runtime_log2phy_cpu.device,
                                       dtype=self.log2phy.dtype
                                       if self.log2phy is not None else
                                       runtime_log2phy_cpu.dtype))

    def ensure_lossless_cpu_shadow(self) -> None:
        if self.elastic_moe_mode != "lossless":
            return
        if self.lossless_cpu_w13_weight is None:
            self.lossless_cpu_w13_weight = _maybe_pin_cpu_tensor(
                self.w13_weight.detach().cpu())
        else:
            self.lossless_cpu_w13_weight = _maybe_pin_cpu_tensor(
                self.lossless_cpu_w13_weight)
        if self.lossless_cpu_w2_weight is None:
            self.lossless_cpu_w2_weight = _maybe_pin_cpu_tensor(
                self.w2_weight.detach().cpu())
        else:
            self.lossless_cpu_w2_weight = _maybe_pin_cpu_tensor(
                self.lossless_cpu_w2_weight)

    def materialize_hybrid_resident_experts(
            self, target_resident_expert_ids: list[int]) -> None:
        if not self.lossless_hybrid_active:
            return
        resident_capacity = int(self.lossless_hybrid_resident_capacity)
        if resident_capacity <= 0:
            raise RuntimeError(
                f"Invalid hybrid resident capacity at layer={self.layer_idx}.")
        target = [int(expert_id) for expert_id in target_resident_expert_ids]
        if len(target) > resident_capacity:
            raise RuntimeError(
                f"Hybrid resident overflow at layer={self.layer_idx}: "
                f"{len(target)} > {resident_capacity}")
        if len(target) < resident_capacity:
            current_resident = list(self.lossless_hybrid_resident_expert_ids)
            fillers = [
                expert_id for expert_id in current_resident
                if expert_id not in target
            ]
            fillers.extend([
                expert_id for expert_id in self.lossless_hybrid_owned_expert_ids
                if expert_id not in target and expert_id not in fillers
            ])
            target.extend(fillers[:resident_capacity - len(target)])
        if len(target) < resident_capacity:
            if not target:
                raise RuntimeError(
                    "Hybrid resident materialization requires at least one "
                    f"target expert at layer={self.layer_idx}.")
            target.extend([int(target[0])] * (resident_capacity - len(target)))
        if target == list(self.lossless_hybrid_resident_expert_ids):
            return

        self.ensure_lossless_cpu_shadow()
        if (self.lossless_cpu_w13_weight is None
                or self.lossless_cpu_w2_weight is None):
            raise RuntimeError(
                "Hybrid resident materialization requires CPU shadow weights: "
                f"layer={self.layer_idx}")

        runtime_device = self.w13_weight.device
        runtime_w13_dtype = self.lossless_cpu_w13_weight.dtype
        runtime_w2_dtype = self.lossless_cpu_w2_weight.dtype
        reuse_loaded_prefix = self._can_reuse_loaded_prefix_for_hybrid(
            resident_capacity, runtime_device, runtime_w13_dtype,
            runtime_w2_dtype)
        if reuse_loaded_prefix:
            runtime_w13_storage = self.w13_weight[:resident_capacity]
            runtime_w2_storage = self.w2_weight[:resident_capacity]
            self.runtime_w13_buffer = None
            self.runtime_w2_buffer = None
            self.runtime_weight_capacity = resident_capacity
        else:
            runtime_format_w13 = _lossless_weight_format(self.w13_weight)
            runtime_format_w2 = _lossless_weight_format(self.w2_weight)
            if (self.runtime_w13_buffer is None
                    or self.runtime_w2_buffer is None
                    or self.runtime_weight_capacity < resident_capacity
                    or self.runtime_w13_buffer.device != runtime_device
                    or self.runtime_w2_buffer.device != runtime_device
                    or self.runtime_w13_buffer.dtype != runtime_w13_dtype
                    or self.runtime_w2_buffer.dtype != runtime_w2_dtype
                    or _lossless_weight_format(self.runtime_w13_buffer)
                    != runtime_format_w13
                    or _lossless_weight_format(self.runtime_w2_buffer)
                    != runtime_format_w2):
                self.runtime_w13_buffer = _allocate_formatted_buffer_like(
                    self.w13_weight, resident_capacity, dtype=runtime_w13_dtype)
                self.runtime_w2_buffer = _allocate_formatted_buffer_like(
                    self.w2_weight, resident_capacity, dtype=runtime_w2_dtype)
                self.runtime_weight_capacity = resident_capacity
                if self.layer_idx == 0 and not self.lossless_hybrid_fallback_logged:
                    logger.warning(
                        "Native common hybrid fell back to fresh runtime buffers: "
                        "layer=%s resident_capacity=%s",
                        self.layer_idx,
                        resident_capacity,
                    )
                    self.lossless_hybrid_fallback_logged = True
            runtime_w13_storage = self.runtime_w13_buffer[:resident_capacity]
            runtime_w2_storage = self.runtime_w2_buffer[:resident_capacity]

        for slot_idx, expert_id in enumerate(target):
            cpu_slot = self.lossless_cpu_shadow_local_slots.get(int(expert_id))
            if cpu_slot is None:
                cpu_slot = int(self.loaded_expert_map[int(expert_id)].item())
            runtime_w13_storage[slot_idx].copy_(
                self.lossless_cpu_w13_weight[cpu_slot], non_blocking=True)
            runtime_w2_storage[slot_idx].copy_(
                self.lossless_cpu_w2_weight[cpu_slot], non_blocking=True)

        new_expert_map = torch.full((int(self.elastic_original_num_experts), ),
                                    -1,
                                    dtype=torch.int32,
                                    device=runtime_device)
        for slot_idx, expert_id in enumerate(target):
            expert_id = int(expert_id)
            if int(new_expert_map[expert_id].item()) != -1:
                continue
            new_expert_map[expert_id] = slot_idx
        self.expert_map = new_expert_map
        self.active_local_num_experts = resident_capacity
        self.local_num_experts = resident_capacity
        self.moe_config.num_local_experts = resident_capacity
        self.runtime_w13_weight = runtime_w13_storage[:resident_capacity]
        self.runtime_w2_weight = runtime_w2_storage[:resident_capacity]
        self.lossless_runtime_activated = True
        self.lossless_hybrid_resident_expert_ids = list(target)
        self.lossless_hybrid_cpu_only_expert_ids = [
            expert_id for expert_id in self.lossless_hybrid_owned_expert_ids
            if expert_id not in set(target)
        ]
        if 0 <= self.lossless_hybrid_active_rank_index < len(
                self.lossless_hybrid_rank_resident_expert_ids):
            self.lossless_hybrid_rank_resident_expert_ids[
                self.lossless_hybrid_active_rank_index] = list(target)
            self.lossless_hybrid_rank_lru[
                self.lossless_hybrid_active_rank_index] = list(target)

    def activate_lossless_hybrid_local_experts(
            self,
            active_expert_ids: list[int],
            source_local_ids: list[int],
            cpu_expert_weights: Optional[dict[int, tuple[torch.Tensor,
                                                         torch.Tensor]]] = None
    ) -> None:
        resident_capacity = self._get_hybrid_resident_capacity()
        if not active_expert_ids:
            raise RuntimeError(
                f"Hybrid activation requires at least one owned expert at layer={self.layer_idx}.")
        self._stash_lossless_primary_prefix_for_hybrid()
        if self.layer_idx == 0:
            self.lossless_hybrid_reuse_log_budget = 2
            logger.info(
                "Native common hybrid fixed-slot activation: layer=%s "
                "owned_local=%s resident_capacity=%s loaded_capacity=%s "
                "fixed_slot_candidate=%s",
                self.layer_idx,
                len(active_expert_ids),
                resident_capacity,
                int(getattr(self, "loaded_weight_capacity", 0)),
                self._can_reuse_loaded_prefix_for_hybrid(
                    resident_capacity, self.w13_weight.device,
                    self.w13_weight.dtype, self.w2_weight.dtype),
            )
        cpu_w13, cpu_w2 = self._build_lossless_cpu_shadow_for_owned_layout(
            active_expert_ids, source_local_ids, cpu_expert_weights)
        self.lossless_cpu_w13_weight = cpu_w13
        self.lossless_cpu_w2_weight = cpu_w2
        self.lossless_cpu_shadow_local_slots = {
            int(expert_id): int(local_slot)
            for local_slot, expert_id in enumerate(active_expert_ids)
        }
        self.lossless_loaded_offloaded = False
        if self._is_mode3_cross_layer_buffer_enabled():
            loaded_expert_map = torch.full(
                (int(self.elastic_original_num_experts), ),
                -1,
                dtype=torch.int32,
                device=self.w13_weight.device)
            for local_slot, expert_id in enumerate(active_expert_ids):
                loaded_expert_map[int(expert_id)] = int(local_slot)
            self.loaded_expert_map = loaded_expert_map
            self.loaded_local_num_experts = len(active_expert_ids)
            self.lossless_cpu_import_expert_ids = [
                int(expert_id) for expert_id, source_local_id in zip(
                    active_expert_ids, source_local_ids)
                if int(source_local_id) < 0
            ]
            self.lossless_hybrid_owned_expert_ids = [
                int(expert_id) for expert_id in active_expert_ids
            ]
            self.lossless_hybrid_resident_capacity = resident_capacity
            self.lossless_mode3_primary_prefix_local_slots = {
                int(expert_id): int(source_local_id)
                for expert_id, source_local_id in zip(active_expert_ids,
                                                      source_local_ids)
                if 0 <= int(source_local_id) < resident_capacity
            }
            self.lossless_mode3_primary_prefix_expert_ids = [
                int(expert_id) for expert_id in active_expert_ids
                if int(expert_id) in self.lossless_mode3_primary_prefix_local_slots
            ]
            primary_prefix_set = set(self.lossless_mode3_primary_prefix_expert_ids)
            self.lossless_hybrid_resident_expert_ids = list(
                self.lossless_mode3_primary_prefix_expert_ids)
            self.lossless_hybrid_cpu_only_expert_ids = [
                int(expert_id)
                for expert_id in self.lossless_hybrid_owned_expert_ids
                if int(expert_id) not in primary_prefix_set
            ]
            self.active_local_num_experts = len(active_expert_ids)
            self.local_num_experts = len(active_expert_ids)
            self.moe_config.num_local_experts = len(active_expert_ids)
            self.moe_config.num_experts = len(active_expert_ids)
            self.num_experts = len(active_expert_ids)
            self.runtime_w13_weight = None
            self.runtime_w2_weight = None
            self.runtime_w13_buffer = None
            self.runtime_w2_buffer = None
            self.runtime_weight_capacity = 0
            self.lossless_runtime_activated = False
            return
        loaded_expert_map = torch.full((int(self.elastic_original_num_experts), ),
                                       -1,
                                       dtype=torch.int32,
                                       device=self.w13_weight.device)
        for local_slot, expert_id in enumerate(active_expert_ids):
            loaded_expert_map[int(expert_id)] = local_slot
        self.loaded_expert_map = loaded_expert_map
        self.loaded_local_num_experts = len(active_expert_ids)
        self.lossless_cpu_import_expert_ids = [
            int(expert_id) for expert_id, source_local_id in zip(
                active_expert_ids, source_local_ids) if int(source_local_id) < 0
        ]
        self.lossless_hybrid_owned_expert_ids = [
            int(x) for x in active_expert_ids
        ]
        self.lossless_hybrid_resident_capacity = resident_capacity
        self.active_local_num_experts = resident_capacity
        self.local_num_experts = resident_capacity
        self.moe_config.num_local_experts = resident_capacity
        self.materialize_hybrid_resident_experts(
            [int(expert_id) for expert_id in active_expert_ids[:resident_capacity]])

    def _plan_lossless_hybrid_rank_waves(
            self, logical_topk_ids: torch.Tensor) -> list[dict[str, Any]]:
        if not self.lossless_hybrid_active:
            return []
        if self.lossless_hybrid_owner_rank_by_expert is None:
            raise RuntimeError(
                f"Missing hybrid owner map at layer={self.layer_idx}.")
        owner_rank = self.lossless_hybrid_owner_rank_by_expert.detach().cpu()[
            logical_topk_ids.detach().cpu()]
        logical_topk_ids_cpu = logical_topk_ids.detach().cpu()
        resident_capacity = int(self.lossless_hybrid_resident_capacity)
        active_rank_count = len(self.lossless_hybrid_active_ranks)
        local_per_rank_needed: list[list[int]] = []
        for rank_idx in range(active_rank_count):
            rank_mask = owner_rank == rank_idx
            if torch.any(rank_mask):
                rank_needed_ids = torch.unique(
                    logical_topk_ids_cpu[rank_mask]).tolist()
                rank_needed = sorted(int(expert_id)
                                     for expert_id in rank_needed_ids)
            else:
                rank_needed = []
            local_per_rank_needed.append(rank_needed)
        if active_rank_count > 1:
            metadata_group = getattr(self.ep_group, "cpu_group",
                                     self.ep_group.device_group)
            metadata_world_size = torch.distributed.get_world_size(
                group=metadata_group)
            if metadata_world_size != active_rank_count:
                raise RuntimeError(
                    "Hybrid active-rank metadata group mismatch at "
                    f"layer={self.layer_idx}: "
                    f"metadata_world_size={metadata_world_size} "
                    f"active_rank_count={active_rank_count}")
            gathered_needed: list[Optional[list[list[int]]]] = [
                None
            ] * active_rank_count
            torch.distributed.all_gather_object(gathered_needed,
                                                local_per_rank_needed,
                                                group=metadata_group)
            merged_needed = [set() for _ in range(active_rank_count)]
            for src_rank_idx, payload in enumerate(gathered_needed):
                if payload is None or len(payload) != active_rank_count:
                    raise RuntimeError(
                        "Hybrid global wave metadata malformed at "
                        f"layer={self.layer_idx}: src_rank={src_rank_idx} "
                        f"payload_len={-1 if payload is None else len(payload)} "
                        f"active_rank_count={active_rank_count}")
                for rank_idx, expert_ids in enumerate(payload):
                    merged_needed[rank_idx].update(int(expert_id)
                                                   for expert_id in expert_ids)
            per_rank_needed = [
                sorted(expert_ids) for expert_ids in merged_needed
            ]
        else:
            per_rank_needed = local_per_rank_needed

        per_rank_waves: list[list[list[int]]] = []
        per_rank_final_resident: list[list[int]] = []
        for rank_idx in range(active_rank_count):
            rank_needed = per_rank_needed[rank_idx]
            current_resident = list(
                self.lossless_hybrid_rank_resident_expert_ids[rank_idx])
            resident_hits = [
                expert_id for expert_id in current_resident
                if expert_id in set(rank_needed)
            ]
            resident_miss = [
                expert_id for expert_id in rank_needed
                if expert_id not in set(resident_hits)
            ]
            ordered_needed = resident_hits + resident_miss
            waves = [
                ordered_needed[idx:idx + resident_capacity]
                for idx in range(0, len(ordered_needed), resident_capacity)
            ]
            if not waves:
                waves = [[]]
            final_resident = list(waves[-1])
            fillers = [
                expert_id for expert_id in current_resident
                if expert_id not in final_resident
            ]
            if len(final_resident) < resident_capacity:
                final_resident.extend(
                    fillers[:resident_capacity - len(final_resident)])
            per_rank_waves.append(waves)
            per_rank_final_resident.append(final_resident)

        num_waves = max(len(waves) for waves in per_rank_waves)
        wave_plans: list[dict[str, Any]] = []
        for wave_idx in range(num_waves):
            wave_rank_resident = []
            wave_expert_to_slot = {}
            wave_active_expert_ids = set()
            for rank_idx in range(active_rank_count):
                wave_group = (per_rank_waves[rank_idx][wave_idx]
                              if wave_idx < len(per_rank_waves[rank_idx]) else [])
                wave_group = [int(expert_id) for expert_id in wave_group]
                current_resident = list(
                    self.lossless_hybrid_rank_resident_expert_ids[rank_idx])
                target_resident = list(wave_group)
                fillers = [
                    expert_id for expert_id in current_resident
                    if expert_id not in target_resident
                ]
                if len(target_resident) < resident_capacity:
                    target_resident.extend(
                        fillers[:resident_capacity - len(target_resident)])
                wave_rank_resident.append(target_resident)
                dense_offset = rank_idx * resident_capacity
                wave_active_expert_ids.update(wave_group)
                for slot_idx, expert_id in enumerate(target_resident):
                    expert_id = int(expert_id)
                    if expert_id not in wave_group:
                        continue
                    if expert_id in wave_expert_to_slot:
                        continue
                    wave_expert_to_slot[expert_id] = dense_offset + slot_idx
            token_mask = torch.zeros(logical_topk_ids.shape[0],
                                     dtype=torch.bool,
                                     device=logical_topk_ids.device)
            topk_mask = torch.zeros_like(logical_topk_ids, dtype=torch.bool)
            for expert_id in wave_active_expert_ids:
                topk_mask |= logical_topk_ids == int(expert_id)
            if topk_mask.numel() > 0:
                token_mask = torch.any(topk_mask, dim=1)
            if token_mask.numel() > 0 and torch.any(token_mask):
                local_wave_topk = int(
                    topk_mask[token_mask].sum(dim=1).max().item())
            else:
                local_wave_topk = 0
            if active_rank_count > 1:
                metadata_group = getattr(self.ep_group, "cpu_group",
                                         self.ep_group.device_group)
                gathered_wave_topk: list[Optional[int]] = [
                    None
                ] * active_rank_count
                torch.distributed.all_gather_object(gathered_wave_topk,
                                                    local_wave_topk,
                                                    group=metadata_group)
                wave_topk = max(int(value) for value in gathered_wave_topk
                                if value is not None)
            else:
                wave_topk = local_wave_topk
            wave_plans.append({
                "wave_idx": wave_idx,
                "rank_wave_groups": [
                    [int(expert_id) for expert_id in (
                        per_rank_waves[rank_idx][wave_idx]
                        if wave_idx < len(per_rank_waves[rank_idx]) else []
                    )] for rank_idx in range(active_rank_count)
                ],
                "rank_resident": wave_rank_resident,
                "wave_expert_to_slot": wave_expert_to_slot,
                "wave_active_expert_ids": sorted(wave_active_expert_ids),
                "token_mask": token_mask,
                "topk_mask": topk_mask,
                "wave_topk": wave_topk,
                "final_rank_resident": per_rank_final_resident,
            })
        return wave_plans

    def set_active_expert_mask(self,
                               new_active_expert_mask: Optional[torch.Tensor]):
        if new_active_expert_mask is None:
            self.active_expert_mask = None
            return
        self.active_expert_mask = new_active_expert_mask.to(
            device=self.expert_map.device if self.expert_map is not None else
            new_active_expert_mask.device,
            dtype=torch.bool)

    def set_elastic_runtime_log2phy(
            self, new_log2phy: Optional[torch.Tensor]) -> None:
        if new_log2phy is None:
            self.elastic_runtime_log2phy = None
            return
        target_device = None
        if self.log2phy is not None and self.log2phy.device.type != "cpu":
            target_device = self.log2phy.device
        elif hasattr(self, "w13_weight"):
            target_device = self.w13_weight.device
        self.elastic_runtime_log2phy = new_log2phy.to(
            device=target_device if target_device is not None else
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
        setup_moe_comm_method(self.moe_config)
        if (self.elastic_moe_mode == "lossless"
                and self._is_mode3_cross_layer_buffer_enabled()
                and self.lossless_hybrid_active):
            if (self.layer_idx == 0
                    and not getattr(self, "_mode3_refresh_skip_logged", False)):
                logger.info(
                    "Native common mode3 refresh preserving hybrid lazy "
                    "runtime state: layer=%s ep_size=%s active_local=%s "
                    "owned_local=%s resident_capacity=%s cpu_only_local=%s",
                    self.layer_idx,
                    int(getattr(self.moe_parallel_config, "ep_size", 0)),
                    int(getattr(self, "active_local_num_experts", 0)),
                    len(getattr(self, "lossless_hybrid_owned_expert_ids", [])),
                    int(getattr(self, "lossless_hybrid_resident_capacity", 0)),
                    len(getattr(self, "lossless_hybrid_cpu_only_expert_ids",
                                [])),
                )
                self._mode3_refresh_skip_logged = True
            return

    def reset_expert_map_and_log2phy(self):
        self._restore_stashed_lossless_primary_prefix()
        self.clear_lossless_hybrid_state()
        self.runtime_w13_weight = None
        self.runtime_w2_weight = None
        self.runtime_w13_buffer = None
        self.runtime_w2_buffer = None
        self.runtime_weight_capacity = 0
        logical_num_experts = int(self.elastic_original_num_experts)
        self.num_experts = logical_num_experts
        self.global_num_experts = logical_num_experts
        self.moe_config.num_experts = logical_num_experts
        primary_local_num_experts, primary_expert_map, primary_log2phy = (
            _parse_expert_map_result(
                determine_expert_map(self.ep_size,
                                     self.ep_rank,
                                     logical_num_experts,
                                     layer_idx=self.layer_idx)))
        if primary_log2phy is None:
            primary_log2phy = determine_default_log2phy_map(
                logical_num_experts, self.ep_size, self.ep_rank, 0)
        target_device = self.expert_map.device if self.expert_map is not None else None
        if target_device is not None and target_device.type == "cpu" and hasattr(
                self, "w13_weight"):
            target_device = self.w13_weight.device
        if target_device is not None:
            primary_expert_map = primary_expert_map.to(device=target_device,
                                                       dtype=torch.int32)
            primary_log2phy = primary_log2phy.to(device=target_device,
                                                 dtype=torch.int32)
        if self.elastic_moe_mode == "lossless":
            self.loaded_local_num_experts, self.loaded_expert_map = (
                determine_redundant_replica_expert_map(
                    logical_num_experts, self.ep_size, self.ep_rank,
                    self.global_redundant_expert_num))
            if target_device is not None:
                self.loaded_expert_map = self.loaded_expert_map.to(
                    device=target_device, dtype=torch.int32)
        else:
            self.loaded_local_num_experts = primary_local_num_experts
            self.loaded_expert_map = primary_expert_map.clone()
        self.active_local_num_experts = primary_local_num_experts
        self.local_num_experts = primary_local_num_experts
        self.expert_map = primary_expert_map
        self.log2phy = primary_log2phy
        self.primary_log2phy = primary_log2phy.clone()
        self.moe_config.num_local_experts = primary_local_num_experts
        self.elastic_runtime_log2phy = None
        self.lossless_runtime_activated = False
        self._refresh_mode1_native_parity_ready("reset_full_world", log=True)

    def restore_lossless_full_world_primary_layout(self) -> None:
        self.reset_expert_map_and_log2phy()

    def export_lossless_expert_cpu_weights(
            self, expert_ids: list[int]) -> dict[int, tuple[torch.Tensor,
                                                            torch.Tensor]]:
        if self.elastic_moe_mode != "lossless" or not expert_ids:
            return {}
        runtime_w13 = (self.runtime_w13_buffer
                       if self.runtime_w13_buffer is not None else
                       self.runtime_w13_weight)
        runtime_w2 = (self.runtime_w2_buffer if self.runtime_w2_buffer is not None
                      else self.runtime_w2_weight)
        if (runtime_w13 is not None and runtime_w2 is not None
                and self.expert_map is not None):
            runtime_slots = [
                int(self.expert_map[int(expert_id)].item())
                for expert_id in expert_ids
            ]
            if all(local_slot >= 0 for local_slot in runtime_slots):
                return {
                    int(expert_id): (
                        runtime_w13[local_slot].detach().cpu(),
                        runtime_w2[local_slot].detach().cpu(),
                    )
                    for expert_id, local_slot in zip(expert_ids, runtime_slots)
                }
        self.ensure_lossless_cpu_shadow()
        export_map = (self.loaded_expert_map
                      if self.loaded_expert_map is not None else self.expert_map)
        cpu_weights: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        for expert_id in expert_ids:
            local_slot = int(export_map[int(expert_id)].item())
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
            return self.w13_weight[:0], self.w2_weight[:0]
        runtime_w13 = (self.runtime_w13_buffer
                       if self.runtime_w13_buffer is not None else
                       self.runtime_w13_weight)
        runtime_w2 = (self.runtime_w2_buffer if self.runtime_w2_buffer is not None
                      else self.runtime_w2_weight)
        if (runtime_w13 is not None and runtime_w2 is not None
                and self.expert_map is not None):
            runtime_slots = [
                int(self.expert_map[int(expert_id)].item())
                for expert_id in expert_ids
            ]
            if all(local_slot >= 0 for local_slot in runtime_slots):
                source_w13 = runtime_w13
                source_w2 = runtime_w2
                export_map = self.expert_map
            else:
                source_w13 = self.w13_weight
                source_w2 = self.w2_weight
                export_map = (self.loaded_expert_map
                              if self.loaded_expert_map is not None else
                              self.expert_map)
        else:
            source_w13 = self.w13_weight
            source_w2 = self.w2_weight
            export_map = (self.loaded_expert_map
                          if self.loaded_expert_map is not None else
                          self.expert_map)
        local_slots = [int(export_map[int(expert_id)].item())
                       for expert_id in expert_ids]
        if any(local_slot < 0 for local_slot in local_slots):
            raise RuntimeError(
                f"Invalid lossless export slot at layer={self.layer_idx}: "
                f"expert_ids={expert_ids} local_slots={local_slots}")
        if len(local_slots) == 1:
            local_slot = local_slots[0]
            export_w13 = source_w13[local_slot:local_slot + 1]
            export_w2 = source_w2[local_slot:local_slot + 1]
            return (_npu_zero_offset_alias_for_p2p(source_w13, export_w13),
                    _npu_zero_offset_alias_for_p2p(source_w2, export_w2))
        export_index = torch.tensor(local_slots,
                                    device=source_w13.device,
                                    dtype=torch.long)
        return (source_w13.index_select(0, export_index),
                source_w2.index_select(0, export_index))

    def get_lossless_expert_npu_slot_recv_buffers(
            self, local_slot: int) -> tuple[torch.Tensor, torch.Tensor]:
        recv_w13 = self.w13_weight[local_slot:local_slot + 1]
        recv_w2 = self.w2_weight[local_slot:local_slot + 1]
        return (_npu_zero_offset_alias_for_p2p(self.w13_weight, recv_w13),
                _npu_zero_offset_alias_for_p2p(self.w2_weight, recv_w2))

    def activate_lossless_local_experts(self,
                                        active_expert_ids: list[int],
                                        source_local_ids: list[int],
                                        cpu_expert_weights=None,
                                        offload_loaded_after_activation:
                                        bool = False) -> None:
        if self.elastic_moe_mode != "lossless":
            return
        new_local_num_experts = len(active_expert_ids)
        hybrid_enabled = self._is_hybrid_cpu_swap_enabled()
        resident_capacity = (self._get_hybrid_resident_capacity()
                             if hybrid_enabled else 0)
        primary_prefix_rows = (self._get_lossless_primary_prefix_row_count()
                               if hybrid_enabled else 0)
        if (hybrid_enabled and resident_capacity > 0
                and new_local_num_experts > primary_prefix_rows):
            if self.layer_idx == 0:
                logger.info(
                    "Redirecting native common lossless activation to hybrid: "
                    "layer=%s owned_local=%s resident_capacity=%s "
                    "primary_prefix_rows=%s",
                    self.layer_idx,
                    new_local_num_experts,
                    resident_capacity,
                    primary_prefix_rows,
                )
            self.activate_lossless_hybrid_local_experts(
                active_expert_ids,
                source_local_ids,
                cpu_expert_weights=cpu_expert_weights)
            return
        self.clear_lossless_hybrid_state()
        self.runtime_w13_weight = None
        self.runtime_w2_weight = None
        self.runtime_w13_buffer = None
        self.runtime_w2_buffer = None
        self.runtime_weight_capacity = 0
        if new_local_num_experts > int(self.w13_weight.shape[0]):
            raise RuntimeError(
                f"Need {new_local_num_experts} expert slots at layer={self.layer_idx}, "
                f"but only have {int(self.w13_weight.shape[0])}.")
        new_expert_map = torch.full((int(self.elastic_original_num_experts), ),
                                    -1,
                                    dtype=torch.int32,
                                    device=self.expert_map.device)
        for local_slot, expert_id in enumerate(active_expert_ids):
            new_expert_map[int(expert_id)] = int(local_slot)
            source_local_id = int(source_local_ids[local_slot])
            if source_local_id >= 0 and source_local_id != local_slot:
                self.w13_weight[local_slot].copy_(self.w13_weight[source_local_id])
                self.w2_weight[local_slot].copy_(self.w2_weight[source_local_id])
            elif source_local_id < 0 and cpu_expert_weights is not None:
                weights = cpu_expert_weights.get(int(expert_id))
                if weights is not None:
                    cpu_w13, cpu_w2 = weights
                    self.w13_weight[local_slot].copy_(cpu_w13, non_blocking=False)
                    self.w2_weight[local_slot].copy_(cpu_w2, non_blocking=False)
        self.expert_map = new_expert_map
        self.loaded_expert_map = new_expert_map.clone()
        self.active_local_num_experts = new_local_num_experts
        self.loaded_local_num_experts = new_local_num_experts
        self.local_num_experts = new_local_num_experts
        self.moe_config.num_local_experts = new_local_num_experts
        self.lossless_runtime_activated = True
        self._refresh_mode1_native_parity_ready("activate_lossless", log=True)

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

    def forward_native(self, hidden_states: torch.Tensor,
                       router_logits: torch.Tensor,
                       **kwargs):
        return super().forward_native(hidden_states, router_logits)

    def forward_oot(self, hidden_states: torch.Tensor,
                    router_logits: torch.Tensor,
                    **kwargs):
        return self.forward_native(hidden_states, router_logits)

    def forward_cuda(self, hidden_states: torch.Tensor,
                     router_logits: torch.Tensor,
                     **kwargs):
        return self.forward_native(hidden_states, router_logits)

    def forward_impl(self, hidden_states: torch.Tensor,
                     router_logits: torch.Tensor):
        assert self.quant_method is not None

        forward_context = get_forward_context()
        forced_hybrid_comm_alignment = bool(
            getattr(self, "lossless_hybrid_active", False))
        original_moe_comm_method = getattr(forward_context, "moe_comm_method",
                                           None)
        original_moe_comm_type = getattr(forward_context, "moe_comm_type", None)
        selected_moe_comm_type = getattr(forward_context,
                                         "selected_moe_comm_type",
                                         original_moe_comm_type)
        original_fused_moe_state = getattr(forward_context, "fused_moe_state",
                                           None)
        original_hybrid_host_metadata = getattr(
            forward_context, "hybrid_force_host_alltoall_metadata", None)
        original_hybrid_stage = getattr(forward_context,
                                        "hybrid_stage_active_ranks", None)

        if forced_hybrid_comm_alignment:
            effective_comm_type = (selected_moe_comm_type
                                   if selected_moe_comm_type is not None else
                                   original_moe_comm_type)
            effective_method = (
                get_moe_comm_method(effective_comm_type)
                if effective_comm_type is not None else original_moe_comm_method)
            if effective_method is None:
                raise RuntimeError(
                    "Hybrid mode cannot resolve the selected MoE comm method "
                    f"at layer={self.layer_idx}: "
                    f"comm_type={effective_comm_type}.")
            forward_context.moe_comm_method = effective_method
            forward_context.moe_comm_type = effective_comm_type
            forward_context.fused_moe_state = _fused_moe_state_for_comm_type(
                effective_comm_type, original_fused_moe_state)
            stage_size = len(getattr(self, "lossless_hybrid_active_ranks", []))
            forward_context.hybrid_force_host_alltoall_metadata = (
                effective_comm_type == MoECommType.ALLTOALL)
            forward_context.hybrid_stage_active_ranks = stage_size
            logged_key = getattr(self, "_hybrid_effective_comm_logged_key",
                                 None)
            current_key = (stage_size, str(selected_moe_comm_type),
                           str(original_moe_comm_type), str(effective_comm_type))
            if self.layer_idx == 0 and logged_key != current_key:
                logger.info(
                    "Native common hybrid MoE comm resolution: layer=%s "
                    "stage=%s selected=%s original=%s effective=%s",
                    self.layer_idx,
                    stage_size,
                    selected_moe_comm_type,
                    original_moe_comm_type,
                    effective_comm_type,
                )
                self._hybrid_effective_comm_logged_key = current_key

        try:
            hidden_states, router_logits = (
                forward_context.moe_comm_method.prepare(
                    hidden_states=hidden_states,
                    router_logits=router_logits,
                    replace_allreduce=forward_context.sp_enabled))

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
                e_score_correction_bias=self.e_score_correction_bias,
                activation=self.activation,
                apply_router_weight_on_input=self.apply_router_weight_on_input,
                enable_eplb=self.enable_eplb,
                expert_load_view=self.expert_load_view,
                logical_to_physical_map=self.logical_to_physical_map,
                logical_replica_count=self.logical_replica_count,
            )
            if isinstance(final_hidden_states, tuple):
                final_hidden_states, group_list_type, expert_tokens = (
                    final_hidden_states)

            if self.dynamic_eplb:
                self.moe_load += expert_tokens if group_list_type else \
                    torch.cat([expert_tokens[:1],
                               expert_tokens[1:] - expert_tokens[:-1]])

            final_hidden_states = forward_context.moe_comm_method.finalize(
                hidden_states=final_hidden_states,
                reduce_results=self.reduce_results)
            return final_hidden_states
        finally:
            if forced_hybrid_comm_alignment:
                forward_context.moe_comm_method = original_moe_comm_method
                forward_context.moe_comm_type = original_moe_comm_type
                forward_context.fused_moe_state = original_fused_moe_state
                forward_context.hybrid_force_host_alltoall_metadata = (
                    original_hybrid_host_metadata)
                forward_context.hybrid_stage_active_ranks = (
                    original_hybrid_stage)

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
