# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024; NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2023 The vLLM team.
# Copyright 2023 DeepSeek-AI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
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

from abc import ABC, abstractmethod
import os
import time
from typing import Any, Optional

import torch
import torch_npu
from vllm.distributed.parallel_state import get_ep_group
from vllm.forward_context import get_forward_context
from vllm.logger import logger

from vllm_ascend.distributed.parallel_state import get_mc2_group
from vllm_ascend.ops.moe.comm_utils import (_gather_along_first_dim,
                                            async_all_to_all)
from vllm_ascend.utils import AscendSocVersion, get_ascend_soc_version


def _env_flag(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).lower() in ("1", "true", "yes", "on")


def _env_int(name: str, default: int, *, minimum: int = 0) -> int:
    try:
        return max(minimum, int(os.getenv(name, str(default))))
    except (TypeError, ValueError):
        return max(minimum, default)


def _env_int_or_none(name: str) -> int | None:
    value = os.getenv(name)
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _npu_memory_snapshot() -> dict[str, int]:
    try:
        free_bytes, total_bytes = torch.npu.mem_get_info()
        stats = torch_npu.npu.memory_stats()
        torch_current = int(stats.get("allocated_bytes.all.current", 0))
        torch_reserved = int(stats.get("reserved_bytes.all.current", 0))
        total_allocated = int(total_bytes - free_bytes)
        return {
            "free_bytes": int(free_bytes),
            "total_bytes": int(total_bytes),
            "torch_current": torch_current,
            "torch_reserved": torch_reserved,
            "non_torch": max(total_allocated - torch_current, 0),
            "total_allocated": total_allocated,
        }
    except Exception:
        return {
            "free_bytes": -1,
            "total_bytes": -1,
            "torch_current": -1,
            "torch_reserved": -1,
            "non_torch": -1,
            "total_allocated": -1,
        }


def _dist_rank() -> int:
    try:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            return int(torch.distributed.get_rank())
    except Exception:
        pass
    return -1


def _log_mc2_memory(tag: str, layer_idx: int, ep_world_size: int,
                    ep_rank_id: int) -> None:
    if not _env_flag("VLLM_ASCEND_MODE1_PARITY_MC2_MEM_LOG", "0"):
        return
    if layer_idx != 0 and not _env_flag(
            "VLLM_ASCEND_MODE1_PARITY_MC2_MEM_LOG_ALL", "0"):
        return
    try:
        free_bytes, total_bytes = torch.npu.mem_get_info()
        stats = torch_npu.npu.memory_stats()
        torch_current = int(stats.get("allocated_bytes.all.current", 0))
        torch_reserved = int(stats.get("reserved_bytes.all.current", 0))
        total_allocated = int(total_bytes - free_bytes)
        non_torch = max(total_allocated - torch_current, 0)
        logger.info(
            "MC2 memory: tag=%s layer=%s ep_world_size=%s ep_rank=%s "
            "free_bytes=%s total_bytes=%s torch_current=%s "
            "torch_reserved=%s non_torch=%s total_allocated=%s",
            tag,
            layer_idx,
            ep_world_size,
            ep_rank_id,
            free_bytes,
            total_bytes,
            torch_current,
            torch_reserved,
            non_torch,
            total_allocated,
        )
    except Exception as exc:
        logger.warning("MC2 memory logging failed at %s: %s", tag, exc)


class MoETokenDispatcher(ABC):

    def __init__(self, **kwargs) -> None:
        """
        Initialize the MoE Token Dispatcher.
        """
        self.top_k = kwargs.get("top_k", 0)
        self.num_experts = kwargs.get("num_experts", 0)

    @property
    def ep_group(self):
        """Get expert model parallel group."""
        return get_ep_group().device_group

    @property
    def ep_metadata_group(self):
        ep_group = get_ep_group()
        return getattr(ep_group, "cpu_group", ep_group.device_group)

    @property
    def ep_rank(self):
        return get_ep_group().rank_in_group

    @property
    def ep_size(self):
        return get_ep_group().world_size

    @abstractmethod
    def token_dispatch(self,
                       hidden_states: torch.Tensor,
                       topk_weights: torch.Tensor,
                       topk_ids: torch.Tensor,
                       row_idx: torch.Tensor,
                       expert_map: Optional[torch.Tensor] = None,
                       log2phy: Optional[torch.Tensor] = None,
                       global_redundant_expert_num: int = 0,
                       shared_experts: Optional[Any] = None,
                       quantized_x_for_share: Optional[Any] = None,
                       dynamic_scale_for_share: Optional[Any] = None,
                       mc2_mask: Optional[torch.Tensor] = None,
                       apply_router_weight_on_input: bool = False,
                       with_quant: bool = False):
        raise NotImplementedError("Dispatch function not implemented.")

    @abstractmethod
    def token_combine(self,
                      hidden_states: torch.Tensor,
                      bias: torch.Tensor = None):
        raise NotImplementedError("Combine function not implemented.")


class TokenDispatcherWithMC2(MoETokenDispatcher):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        mc2_group = get_mc2_group()
        device_group = mc2_group.device_group
        # TODO: Try local_rank = ep_group.rank_in_group
        local_rank = torch.distributed.get_rank(group=device_group)
        backend = device_group._get_backend(torch.device("npu"))
        self.moe_all_to_all_group_name = backend.get_hccl_comm_name(local_rank)
        self.ep_rank_id = mc2_group.rank_in_group
        self.ep_world_size = mc2_group.world_size
        self.enable_dispatch_v2 = hasattr(torch_npu,
                                          "npu_moe_distribute_dispatch_v2")
        self.need_extra_args = (
            get_ascend_soc_version() == AscendSocVersion.A3)

        # NOTE: Currently, when in A3, we need to pass in some extra param into dispatch & combine
        self.a3_need_extra_args = \
            get_ascend_soc_version() == AscendSocVersion.A3
        self.output = None
        self.assist_info_for_combine = None
        self.ep_recv_counts = None
        self.shared_act = None
        self.topk_ids = None
        self.topk_weights = None
        self.shared_experts = None
        self.mc2_mask = None
        self._last_use_dispatch_v2 = self.enable_dispatch_v2
        self._dispatch_v2_diag_count = 0
        self.with_quant = False
        # Align old custom MC2 dispatch_v2 with the native/torchair default:
        # use per-expert token counts instead of cumulative counts.
        self.expert_token_nums_type = 1
        if _env_flag("VLLM_ASCEND_MODE1_PARITY_MC2_MEM_LOG", "0"):
            logger.info(
                "MC2 dispatcher init: dispatcher_id=%s ranks=%s "
                "ep_world_size=%s ep_rank=%s local_rank=%s "
                "expert_token_nums_type=%s dispatch_v2=%s a3_extra_args=%s",
                id(self),
                tuple(int(rank) for rank in getattr(mc2_group, "ranks", [])),
                self.ep_world_size,
                self.ep_rank_id,
                local_rank,
                self.expert_token_nums_type,
                self.enable_dispatch_v2,
                self.a3_need_extra_args,
            )

    def _current_dispatch_v2_enabled(self) -> bool:
        if not self.enable_dispatch_v2:
            return False
        return True

    def get_dispatch_mc2_kwargs(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        expert_map: torch.Tensor,
        global_redundant_expert_num: int = 0,
    ):
        # In elastic EP shrink, expert ids are remapped into the active dense
        # physical space. Dispatch/combine must therefore use the current
        # runtime expert count, not the original logical expert_map length.
        moe_expert_num = self.num_experts
        if self.with_quant:
            quant_mode = 2
            moe_expert_num = moe_expert_num + global_redundant_expert_num
        else:
            quant_mode = 0
        kwargs_mc2 = {
            "x": hidden_states,
            "expert_ids": topk_ids,
            "expert_shard_type": 0,
            "shared_expert_rank_num": 0,
            "moe_expert_num": moe_expert_num,
            "global_bs": 0,
            "expert_token_nums_type": int(
                getattr(self, "expert_token_nums_type", 0)),
        }

        stage1_kwargs = {
            "scales": None,
            "quant_mode": quant_mode,
            "group_ep": self.moe_all_to_all_group_name,
            "ep_world_size": self.ep_world_size,
            "ep_rank_id": self.ep_rank_id,
        }
        if self.need_extra_args:
            stage1_kwargs.update({
                "group_tp": self.moe_all_to_all_group_name,
                "tp_world_size": 1,
                "tp_rank_id": 0,
            })
        use_dispatch_v2 = self._current_dispatch_v2_enabled()
        if self.a3_need_extra_args and use_dispatch_v2:
            stage1_kwargs.update({
                "x_active_mask": self.mc2_mask,
            })

        kwargs_mc2.update(stage1_kwargs)
        return kwargs_mc2

    def token_dispatch(self,
                       hidden_states: torch.Tensor,
                       topk_weights: torch.Tensor,
                       topk_ids: torch.Tensor,
                       row_idx: torch.Tensor,
                       expert_map: Optional[torch.Tensor] = None,
                       log2phy: Optional[torch.Tensor] = None,
                       global_redundant_expert_num: int = 0,
                       shared_experts: Optional[Any] = None,
                       quantized_x_for_share: Optional[Any] = None,
                       dynamic_scale_for_share: Optional[Any] = None,
                       mc2_mask: Optional[torch.Tensor] = None,
                       apply_router_weight_on_input: bool = False,
                       with_quant: bool = False):
        self.with_quant = with_quant
        self.expert_map = expert_map
        if log2phy is not None:
            topk_ids = log2phy[topk_ids]
        self.topk_ids = topk_ids
        self.topk_weights = topk_weights
        self.shared_experts = shared_experts
        use_dispatch_v2 = self._current_dispatch_v2_enabled()
        if self.a3_need_extra_args and use_dispatch_v2:
            if mc2_mask is None:
                raise RuntimeError(
                    "MC2 dispatch_v2 on A3 requires x_active_mask/mc2_mask.")
            mc2_mask = mc2_mask.to(device=hidden_states.device, dtype=torch.bool)
            if mc2_mask.shape[0] != hidden_states.shape[0]:
                raise RuntimeError(
                    "MC2 dispatch mask length mismatch: "
                    f"hidden_tokens={hidden_states.shape[0]} "
                    f"mask_tokens={mc2_mask.shape[0]} "
                    f"topk_rows={topk_ids.shape[0]}")
        self.mc2_mask = mc2_mask

        forward_context = get_forward_context()
        debug_info = getattr(forward_context, "elastic_debug_info", None)
        layer_idx = -1
        if debug_info is not None:
            try:
                layer_idx = int(debug_info.get("layer_idx", -1))
            except Exception:
                layer_idx = -1
        if debug_info is not None and debug_info.get("layer_idx", -1) == 0:
            active_mask_count = (
                int(torch.count_nonzero(self.mc2_mask).item())
                if self.mc2_mask is not None else -1)
            logger.info(
                "MC2 dispatch args: layer=%s ep_world_size=%s ep_rank=%s "
                "hidden_tokens=%s topk_shape=%s moe_expert_num=%s "
                "dispatcher_num_experts=%s expert_token_nums_type=%s "
                "use_dispatch_v2=%s active_mask_count=%s expert_map_shape=%s "
                "topk_min=%s "
                "topk_max=%s topk_unique=%s",
                debug_info.get("layer_idx", -1),
                self.ep_world_size,
                self.ep_rank_id,
                int(hidden_states.shape[0]),
                tuple(topk_ids.shape),
                int(self.get_dispatch_mc2_kwargs(hidden_states, topk_weights,
                                                 topk_ids, expert_map,
                                                 global_redundant_expert_num)
                    ["moe_expert_num"]),
                int(getattr(self, "num_experts", -1)),
                int(getattr(self, "expert_token_nums_type", -1)),
                int(use_dispatch_v2),
                active_mask_count,
                tuple(expert_map.shape) if expert_map is not None else None,
                int(topk_ids.min().item()) if topk_ids.numel() > 0 else -1,
                int(topk_ids.max().item()) if topk_ids.numel() > 0 else -1,
                int(torch.unique(topk_ids).numel()) if topk_ids.numel() > 0 else 0,
            )

        kwargs_mc2 = self.get_dispatch_mc2_kwargs(hidden_states, topk_weights,
                                                  topk_ids, expert_map,
                                                  global_redundant_expert_num)
        self._last_use_dispatch_v2 = bool(use_dispatch_v2)
        diag_enabled = _env_flag("VLLM_ASCEND_DISPATCH_V2_DIAG_LOG", "0")
        diag_ep_world_size = _env_int_or_none(
            "VLLM_ASCEND_DISPATCH_V2_DIAG_EP_WORLD_SIZE")
        if diag_ep_world_size is not None and int(
                self.ep_world_size) != diag_ep_world_size:
            diag_enabled = False
        diag_count = int(getattr(self, "_dispatch_v2_diag_count", 0))
        diag_first_n = _env_int("VLLM_ASCEND_DISPATCH_V2_DIAG_FIRST_N", 4)
        diag_all = _env_flag("VLLM_ASCEND_DISPATCH_V2_DIAG_ALL", "0")
        diag_this = bool(diag_enabled and (diag_all or diag_count < diag_first_n))
        if diag_enabled:
            self._dispatch_v2_diag_count = diag_count + 1
        if diag_this:
            mem = _npu_memory_snapshot()
            try:
                topk_min = int(topk_ids.min().item()) if topk_ids.numel() else -1
                topk_max = int(topk_ids.max().item()) if topk_ids.numel() else -1
            except Exception:
                topk_min = -1
                topk_max = -1
            try:
                mask_count = (int(torch.count_nonzero(self.mc2_mask).item())
                              if self.mc2_mask is not None else -1)
            except Exception:
                mask_count = -1
            logger.info(
                "MC2 DispatchV2 diag before: rank=%s dispatcher_id=%s "
                "call=%s layer=%s ep_world_size=%s ep_rank=%s group=%s "
                "use_dispatch_v2=%s moe_expert_num=%s "
                "dispatcher_num_experts=%s dispatcher_num_experts_local=%s "
                "expert_token_nums_type=%s hidden_shape=%s hidden_dtype=%s "
                "topk_shape=%s topk_dtype=%s topk_min=%s topk_max=%s "
                "mask_shape=%s mask_count=%s expert_map_shape=%s "
                "free_bytes=%s total_bytes=%s torch_current=%s "
                "torch_reserved=%s non_torch=%s total_allocated=%s",
                _dist_rank(),
                id(self),
                diag_count,
                layer_idx,
                self.ep_world_size,
                self.ep_rank_id,
                self.moe_all_to_all_group_name,
                int(use_dispatch_v2),
                int(kwargs_mc2.get("moe_expert_num", -1)),
                int(getattr(self, "num_experts", -1)),
                int(getattr(self, "num_experts_local", -1)),
                int(getattr(self, "expert_token_nums_type", -1)),
                tuple(hidden_states.shape),
                str(hidden_states.dtype),
                tuple(topk_ids.shape),
                str(topk_ids.dtype),
                topk_min,
                topk_max,
                tuple(self.mc2_mask.shape) if self.mc2_mask is not None else None,
                mask_count,
                tuple(expert_map.shape) if expert_map is not None else None,
                mem["free_bytes"],
                mem["total_bytes"],
                mem["torch_current"],
                mem["torch_reserved"],
                mem["non_torch"],
                mem["total_allocated"],
            )
        _log_mc2_memory("before_dispatch", layer_idx, self.ep_world_size,
                        self.ep_rank_id)
        dispatch_start_ts = time.perf_counter()
        try:
            self.output = torch_npu.npu_moe_distribute_dispatch_v2(
                **kwargs_mc2
            ) if use_dispatch_v2 else torch_npu.npu_moe_distribute_dispatch(
                **kwargs_mc2)
        except Exception as exc:
            if diag_this:
                mem = _npu_memory_snapshot()
                logger.error(
                    "MC2 DispatchV2 diag error: rank=%s dispatcher_id=%s "
                    "call=%s layer=%s ep_world_size=%s ep_rank=%s "
                    "elapsed_ms=%.3f error=%r free_bytes=%s total_bytes=%s "
                    "torch_current=%s torch_reserved=%s non_torch=%s "
                    "total_allocated=%s",
                    _dist_rank(),
                    id(self),
                    diag_count,
                    layer_idx,
                    self.ep_world_size,
                    self.ep_rank_id,
                    (time.perf_counter() - dispatch_start_ts) * 1000.0,
                    exc,
                    mem["free_bytes"],
                    mem["total_bytes"],
                    mem["torch_current"],
                    mem["torch_reserved"],
                    mem["non_torch"],
                    mem["total_allocated"],
                )
            raise
        _log_mc2_memory("after_dispatch_submit", layer_idx, self.ep_world_size,
                        self.ep_rank_id)
        dispatch_elapsed_ms = (time.perf_counter() - dispatch_start_ts) * 1000.0
        host_timing_enabled = _env_flag("VLLM_ASCEND_MC2_HOST_TIMING_LOG", "1")
        host_timing_threshold_ms = _env_int(
            "VLLM_ASCEND_MC2_HOST_TIMING_THRESHOLD_MS", 1000)
        host_timing_first_n = _env_int(
            "VLLM_ASCEND_MC2_HOST_TIMING_FIRST_N", 4)
        host_timing_count = int(getattr(self, "_mc2_host_timing_count", 0))
        if host_timing_enabled and (host_timing_count < host_timing_first_n
                                    or dispatch_elapsed_ms >=
                                    host_timing_threshold_ms):
            self._mc2_host_timing_count = host_timing_count + 1
            logger.info(
                "MC2 host timing: phase=dispatch rank=%s dispatcher_id=%s "
                "call=%s layer=%s ep_world_size=%s ep_rank=%s "
                "use_dispatch_v2=%s elapsed_ms=%.3f moe_expert_num=%s "
                "dispatcher_num_experts=%s expert_map_shape=%s "
                "hidden_shape=%s topk_shape=%s",
                _dist_rank(),
                id(self),
                host_timing_count,
                layer_idx,
                self.ep_world_size,
                self.ep_rank_id,
                int(use_dispatch_v2),
                dispatch_elapsed_ms,
                int(kwargs_mc2.get("moe_expert_num", -1)),
                int(getattr(self, "num_experts", -1)),
                tuple(expert_map.shape) if expert_map is not None else None,
                tuple(hidden_states.shape),
                tuple(topk_ids.shape),
            )
        if diag_this:
            mem = _npu_memory_snapshot()
            output_shapes = []
            try:
                for item in self.output[:6]:
                    output_shapes.append(
                        tuple(item.shape) if hasattr(item, "shape") else
                        type(item).__name__)
            except Exception:
                output_shapes = ["<unavailable>"]
            logger.info(
                "MC2 DispatchV2 diag after: rank=%s dispatcher_id=%s "
                "call=%s layer=%s ep_world_size=%s ep_rank=%s "
                "elapsed_ms=%.3f output_shapes=%s free_bytes=%s "
                "total_bytes=%s torch_current=%s torch_reserved=%s "
                "non_torch=%s total_allocated=%s",
                _dist_rank(),
                id(self),
                diag_count,
                layer_idx,
                self.ep_world_size,
                self.ep_rank_id,
                (time.perf_counter() - dispatch_start_ts) * 1000.0,
                tuple(output_shapes),
                mem["free_bytes"],
                mem["total_bytes"],
                mem["torch_current"],
                mem["torch_reserved"],
                mem["non_torch"],
                mem["total_allocated"],
            )
        # comm_stream.wait_stream(torch.npu.current_stream())
        expand_x, dynamic_scale, self.assist_info_for_combine, \
            expert_token_nums, self.ep_recv_counts = self.output[0:5]
        # if expand_x.shape[0] == 4096 and self.ep_rank_id == 0:
        #     torch.set_printoptions(profile="full")
        #     print("assist_info_for_combine in dispatch is", self.assist_info_for_combine, "size is", self.assist_info_for_combine.size())
        #     torch.set_printoptions(profile="default")

        if self.with_quant:
            if shared_experts is not None:
                share_up_out, _ = shared_experts.gate_up_proj(
                    (quantized_x_for_share, dynamic_scale_for_share))
                shared_gate_up, shared_dequant_scale = share_up_out[
                    0], share_up_out[1]

                shared_act_out = shared_experts.act_fn(
                    (shared_gate_up, shared_dequant_scale))
                self.shared_act, self.swiglu_out_scale = \
                    shared_act_out[0], shared_act_out[1]

        else:
            if shared_experts is not None:
                shared_gate_up, _ = shared_experts.gate_up_proj(hidden_states)
                self.shared_act = shared_experts.act_fn(shared_gate_up)
        group_list_type = int(getattr(self, "expert_token_nums_type", 0))
        return {
            "group_list_type": group_list_type,
            "hidden_states": expand_x,
            "group_list": expert_token_nums,
            "dynamic_scale": dynamic_scale,
        }

    def get_combine_mc_kwargs(self, hidden_states: torch.Tensor):
        assert self.expert_map is not None
        assert self.topk_weights is not None
        assert self.topk_ids is not None
        assert self.output is not None
        moe_expert_num = self.num_experts
        # moeCombine
        kwargs_mc2 = {
            "expand_x": hidden_states,
            "expert_ids": self.topk_ids,
            "expert_scales": self.topk_weights.to(torch.float32),
            "expert_shard_type": 0,
            "shared_expert_rank_num": 0,
            "moe_expert_num": moe_expert_num,
            "global_bs": 0,
        }
        if self.with_quant:
            tp_recv_counts = torch.empty(1,
                                         dtype=torch.int32,
                                         device=hidden_states.device)
        else:
            tp_recv_counts = self.output[5]
        stage3_kwargs = {
            "ep_send_counts": self.ep_recv_counts,
            "group_ep": self.moe_all_to_all_group_name,
            "ep_world_size": self.ep_world_size,
            "ep_rank_id": self.ep_rank_id,
        }
        use_dispatch_v2 = bool(getattr(self, "_last_use_dispatch_v2",
                                       self._current_dispatch_v2_enabled()))
        if use_dispatch_v2:
            stage3_kwargs.update({
                "assist_info_for_combine":
                self.assist_info_for_combine,
            })
        else:
            stage3_kwargs.update({
                "expand_idx": self.assist_info_for_combine,
            })
        if self.need_extra_args:
            stage3_kwargs.update({
                "tp_send_counts": tp_recv_counts,
                "group_tp": self.moe_all_to_all_group_name,
                "tp_world_size": 1,
                "tp_rank_id": 0,
            })
        if self.a3_need_extra_args and use_dispatch_v2:
            stage3_kwargs.update({
                "x_active_mask": self.mc2_mask,
            })
        kwargs_mc2.update(stage3_kwargs)
        return kwargs_mc2

    def token_combine(self,
                      hidden_states: torch.Tensor,
                      bias: torch.Tensor = None):
        kwargs_mc2 = self.get_combine_mc_kwargs(hidden_states)
        _log_mc2_memory("before_combine", -1, self.ep_world_size,
                        self.ep_rank_id)
        combine_start_ts = time.perf_counter()
        hidden_states = torch_npu.npu_moe_distribute_combine_v2(
            **kwargs_mc2
        ) if self._current_dispatch_v2_enabled(
        ) else torch_npu.npu_moe_distribute_combine(**kwargs_mc2)
        _log_mc2_memory("after_combine_submit", -1, self.ep_world_size,
                        self.ep_rank_id)
        combine_elapsed_ms = (time.perf_counter() - combine_start_ts) * 1000.0
        host_timing_enabled = _env_flag("VLLM_ASCEND_MC2_HOST_TIMING_LOG", "1")
        host_timing_threshold_ms = _env_int(
            "VLLM_ASCEND_MC2_HOST_TIMING_THRESHOLD_MS", 1000)
        host_timing_first_n = _env_int(
            "VLLM_ASCEND_MC2_HOST_TIMING_FIRST_N", 4)
        host_timing_count = int(getattr(self, "_mc2_combine_host_timing_count",
                                        0))
        if host_timing_enabled and (host_timing_count < host_timing_first_n
                                    or combine_elapsed_ms >=
                                    host_timing_threshold_ms):
            self._mc2_combine_host_timing_count = host_timing_count + 1
            logger.info(
                "MC2 host timing: phase=combine rank=%s dispatcher_id=%s "
                "call=%s ep_world_size=%s ep_rank=%s use_dispatch_v2=%s "
                "elapsed_ms=%.3f moe_expert_num=%s dispatcher_num_experts=%s "
                "expert_map_shape=%s hidden_shape=%s",
                _dist_rank(),
                id(self),
                host_timing_count,
                self.ep_world_size,
                self.ep_rank_id,
                int(bool(getattr(self, "_last_use_dispatch_v2",
                                 self._current_dispatch_v2_enabled()))),
                combine_elapsed_ms,
                int(kwargs_mc2.get("moe_expert_num", -1)),
                int(getattr(self, "num_experts", -1)),
                tuple(self.expert_map.shape)
                if self.expert_map is not None else None,
                tuple(hidden_states.shape),
            )

        # these values are no longer used, so they need to be set to None for memory release.
        self.output = None
        self.assist_info_for_combine = None
        self.ep_recv_counts = None
        self.topk_ids = None
        self.topk_weights = None
        self.mc2_mask = None
        self.expert_map = None
        self._last_use_dispatch_v2 = self.enable_dispatch_v2

        if self.shared_experts is None:
            return hidden_states
        else:
            if self.with_quant:
                shared_hidden_states, _ = self.shared_experts.down_proj(
                    (self.shared_act, self.swiglu_out_scale))
            else:
                shared_hidden_states, _ = self.shared_experts.down_proj(
                    self.shared_act)
            self.shared_act = None
            self.shared_experts = None
            self.swiglu_out_scale = None
            return hidden_states, shared_hidden_states


class TokenDispatcherWithAllGather(MoETokenDispatcher):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.apply_router_weight_on_input = False
        self.max_num_tokens = kwargs.get("max_num_tokens")
        self.num_experts_local = kwargs.get("num_local_experts", 0)
        self.sorted_weights = None
        self.expanded_row_idx = None
        self.sorted_token_indices = None
        self.original_shape = None
        self.mask = None
        self.expert_map = None
        self.topk_weights = None
        self.topk_ids = None
        self.with_quant = False

    def token_dispatch(self,
                       hidden_states: torch.Tensor,
                       topk_weights: torch.Tensor,
                       topk_ids: torch.Tensor,
                       row_idx: torch.Tensor,
                       expert_map: Optional[torch.Tensor] = None,
                       log2phy: Optional[torch.Tensor] = None,
                       global_redundant_expert_num: int = 0,
                       shared_experts: Optional[Any] = None,
                       quantized_x_for_share: Optional[Any] = None,
                       dynamic_scale_for_share: Optional[Any] = None,
                       mc2_mask: Optional[torch.Tensor] = None,
                       apply_router_weight_on_input: bool = False,
                       with_quant: bool = False):
        self.with_quant = with_quant
        self.original_shape = hidden_states.shape

        num_tokens = hidden_states.shape[:-1].numel()
        self.expert_map = expert_map
        self.topk_weights = topk_weights
        self.topk_ids = topk_ids
        self.apply_router_weight_on_input = apply_router_weight_on_input
        if self.apply_router_weight_on_input:
            assert (topk_weights.dim() == 2
                    ), "`topk_weights` should be in shape (num_tokens, topk)"
            _, topk = topk_weights.shape
            assert (
                topk == 1
            ), "Only support topk=1 when `apply_router_weight_on_input` is True"
            hidden_states = hidden_states * \
                topk_weights.to(hidden_states.dtype)
        if log2phy is not None:
            topk_ids = log2phy[topk_ids]
            global_num_experts = len(log2phy)
            first_expert_idx = get_ep_group(
            ).rank_in_group * self.num_experts_local
            last_expert_idx = first_expert_idx + self.num_experts_local
            mask = ((topk_ids >= first_expert_idx) &
                    (topk_ids < last_expert_idx))
            self.topk_weights = topk_weights * mask
        elif expert_map is not None:
            global_num_experts = len(expert_map)
            mask = (expert_map[topk_ids] != -1)
            self.topk_weights = topk_weights * mask
            first_expert_idx = get_ep_group(
            ).rank_in_group * self.num_experts_local
            last_expert_idx = first_expert_idx + self.num_experts_local
        else:
            first_expert_idx = 0
            last_expert_idx = self.num_experts_local
            global_num_experts = self.num_experts_local

        sorted_hidden_states, self.expanded_row_idx, expert_tokens, pertoken_scale = (
            torch_npu.npu_moe_init_routing_v2(
                hidden_states,
                topk_ids,
                active_num=num_tokens * self.top_k,
                expert_num=global_num_experts,
                expert_tokens_num_type=1,
                expert_tokens_num_flag=True,
                active_expert_range=[first_expert_idx, last_expert_idx],
                quant_mode=1 if self.with_quant else -1,
            ))
        expert_tokens = expert_tokens.to(torch.int64)
        group_list_type = 1  # `count` mode
        return {
            "group_list_type": group_list_type,
            "hidden_states": sorted_hidden_states,
            "group_list": expert_tokens,
            "dynamic_scale": pertoken_scale if self.with_quant else None,
        }

    def token_combine(self,
                      hidden_states: torch.Tensor,
                      bias: torch.Tensor = None):
        assert self.original_shape is not None
        final_hidden_states = torch_npu.npu_moe_token_unpermute(
            permuted_tokens=hidden_states,
            sorted_indices=self.expanded_row_idx,
            probs=self.topk_weights)
        if len(self.original_shape) == 3:
            final_hidden_states = final_hidden_states.view(self.original_shape)

        # these values are no longer used, so they need to be set to None for memory release.
        self.expert_map = None
        self.topk_weights = None
        self.topk_ids = None
        self.expanded_row_idx = None
        return final_hidden_states


# mypy: disable-error-code="override"
class TokenDispatcherWithMoge(MoETokenDispatcher):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.apply_router_weight_on_input = False
        self.local_num_experts = self.num_experts // self.ep_size
        self.local_num_group = self.top_k // self.ep_size
        self.bsz = None

    def token_dispatch(self,
                       hidden_states: torch.Tensor,
                       topk_weights: torch.Tensor,
                       topk_ids: torch.Tensor,
                       row_idx: torch.Tensor,
                       expert_map: Optional[torch.Tensor] = None,
                       log2phy: Optional[torch.Tensor] = None,
                       global_redundant_expert_num: int = 0,
                       shared_experts: Optional[Any] = None,
                       quantized_x_for_share: Optional[Any] = None,
                       dynamic_scale_for_share: Optional[Any] = None,
                       mc2_mask: Optional[torch.Tensor] = None,
                       apply_router_weight_on_input: bool = False,
                       with_quant: bool = False):
        self.bsz, _ = hidden_states.shape
        flatten_topk_ids = topk_ids.view(-1)
        self.sorted_topk_ids = torch.argsort(flatten_topk_ids.float())
        self.sorted_topk_ids = self.sorted_topk_ids.to(torch.int32)
        sorted_hidden_states = hidden_states.index_select(
            0, self.sorted_topk_ids // self.local_num_group)

        experts_id = torch.arange(0,
                                  self.local_num_experts,
                                  dtype=topk_ids.dtype,
                                  device=topk_ids.device)
        num_tokens_per_expert = (
            flatten_topk_ids.unsqueeze(-1) == experts_id).to(
                torch.float32).sum(0)
        topk_scales = topk_weights.view(-1).index_select(
            0, self.sorted_topk_ids).unsqueeze(-1)
        group_list = num_tokens_per_expert.cumsum(dim=0).to(torch.int64)
        group_list_type = 0
        return {
            "group_list_type": group_list_type,
            "hidden_states": sorted_hidden_states,
            "group_list": group_list,
            "topk_scales": topk_scales,
        }

    def token_combine(self,
                      hidden_states: torch.Tensor,
                      bias: torch.Tensor = None):
        unsorted_topk_ids = torch.argsort(self.sorted_topk_ids.float()).to(
            torch.int32)
        unsorted_hidden_states = hidden_states.index_select(
            0, unsorted_topk_ids)
        final_hidden_states = unsorted_hidden_states.reshape(
            self.bsz, self.top_k // self.ep_size, -1).sum(1)
        return final_hidden_states


class TokenDispatcherWithAll2AllV(MoETokenDispatcher):
    """
    The implementation of the AlltoAll-based token dispatcher, which handles token
    dispatching on the sequence level instead of token level. The core of this implementation
    lies in each device dispatching on the entire sequence, with the hidden state being partitioned.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.with_quant = False
        self.num_local_experts = kwargs.get("num_local_experts", 0)
        self.num_global_redundant_experts = kwargs.get(
            "num_global_redundant_experts", 0)
        self.num_experts = self.num_experts + self.num_global_redundant_experts

        self.hidden_shape = None
        self.topk_weights = None
        self.input_splits = None
        self.output_splits = None
        self.hidden_shape_before_permute = None
        self._hybrid_host_metadata_logged_stage = None

        # [tp_ep_size * ep_size, num_local_experts]. Represents the number of tokens sent
        # to each local expert by all ranks.
        self.num_global_tokens_per_local_expert = None

        # cached intermediate tensors.
        self.tokens_per_expert = None
        self.global_input_tokens_local_experts_indices = None

        assert self.num_local_experts > 0, "Expected at least one expert"
        if self.num_local_experts > 1:
            self.expert_ids_per_ep_rank = torch.tensor(
                [i % self.num_local_experts for i in range(self.num_experts)],
                dtype=torch.int32,
                device=torch.npu.current_device(),
            )

        local_expert_indices_offset = (self.ep_rank * self.num_local_experts)

        self.local_expert_indices = [
            local_expert_indices_offset + i
            for i in range(self.num_local_experts)
        ]
        assert (len(self.local_expert_indices) == self.num_local_experts
                ), "Invalid local expert indices"
        for i in range(len(self.local_expert_indices) - 1):
            assert (self.local_expert_indices[i] ==
                    self.local_expert_indices[i + 1] -
                    1), "local_expert_indices must be continuous"

    def token_dispatch(self,
                       hidden_states: torch.Tensor,
                       topk_weights: torch.Tensor,
                       topk_ids: torch.Tensor,
                       row_idx: torch.Tensor,
                       expert_map: Optional[torch.Tensor] = None,
                       log2phy: Optional[torch.Tensor] = None,
                       global_redundant_expert_num: int = 0,
                       shared_experts: Optional[Any] = None,
                       quantized_x_for_share: Optional[Any] = None,
                       dynamic_scale_for_share: Optional[Any] = None,
                       mc2_mask: Optional[torch.Tensor] = None,
                       apply_router_weight_on_input: bool = False,
                       with_quant: bool = False):
        self.with_quant = with_quant
        self.hidden_shape = hidden_states.shape
        self.topk_weights = topk_weights
        assert topk_weights.dim() == 2, "Expected 2D tensor for topk_weights"
        assert topk_ids.dim() == 2, "Expected 2D tensor for routing map"

        pre_map_topk_ids = topk_ids
        if log2phy is not None:
            topk_ids = log2phy[topk_ids]
            if topk_ids.numel() > 0:
                mapped_topk_max = int(topk_ids.max().item())
                mapped_topk_min = int(topk_ids.min().item())
                if mapped_topk_min < 0 or mapped_topk_max >= self.num_experts:
                    debug_ctx = self._debug_context()
                    logger.error(
                        "AllToAll log2phy remap overflow at %s: rank=%s ep_size=%s num_experts=%s pre_topk_min=%s pre_topk_max=%s post_topk_min=%s post_topk_max=%s log2phy_min=%s log2phy_max=%s",
                        debug_ctx,
                        self.ep_rank,
                        self.ep_size,
                        self.num_experts,
                        int(pre_map_topk_ids.min().item()),
                        int(pre_map_topk_ids.max().item()),
                        mapped_topk_min,
                        mapped_topk_max,
                        int(log2phy.min().item()),
                        int(log2phy.max().item()),
                    )

        permutated_local_input_tokens, reversed_local_input_permutation_mapping, tokens_per_expert = self._dispatch_preprocess(
            hidden_states, topk_ids)
        self.reversed_local_input_permutation_mapping = reversed_local_input_permutation_mapping
        debug_info = getattr(get_forward_context(), "elastic_debug_info", None)
        if (debug_info is not None and
                debug_info.get("reason") == "lossless_pre_shrink_loaded_only"
                and debug_info.get("route_debug", False)):
            nonzero_experts = []
            nonzero_counts = []
            if tokens_per_expert.numel() > 0:
                nonzero_mask = tokens_per_expert > 0
                nonzero_experts = torch.nonzero(
                    nonzero_mask, as_tuple=False).reshape(-1)
                nonzero_counts = tokens_per_expert[nonzero_mask]
            top_expert_pairs = []
            if len(nonzero_counts) > 0:
                order = torch.argsort(nonzero_counts, descending=True)[:8]
                top_expert_pairs = [
                    (int(nonzero_experts[idx].item()),
                     int(nonzero_counts[idx].item())) for idx in order
                ]
            flat_pre_ids = pre_map_topk_ids.reshape(-1)
            flat_mapped_ids = topk_ids.reshape(-1)
            active_local_num_experts = int(
                debug_info.get("active_local_num_experts", -1))
            invalid_local_slots = []
            if len(nonzero_experts) > 0:
                invalid_mask = ((nonzero_experts < 0) |
                                (nonzero_experts >= active_local_num_experts))
                invalid_local_slots = [
                    int(local_slot.item())
                    for local_slot in nonzero_experts[invalid_mask]
                ]
            logger.info(
                "Lossless pre-shrink route summary: rank=%s layer=%s tag=%s reason=%s topk_shape=%s pre_topk_min=%s pre_topk_max=%s pre_unique=%s mapped_topk_min=%s mapped_topk_max=%s mapped_unique=%s num_local_experts=%s active_local_num_experts=%s nonzero_local_experts=%s top_local_tokens=%s input_splits=%s output_splits=%s num_out_tokens=%s",
                self.ep_rank,
                debug_info.get("layer_idx"),
                debug_info.get("tag"),
                debug_info.get("reason"),
                tuple(topk_ids.shape),
                int(flat_pre_ids.min().item()) if flat_pre_ids.numel() > 0 else -1,
                int(flat_pre_ids.max().item()) if flat_pre_ids.numel() > 0 else -1,
                int(torch.unique(flat_pre_ids).numel())
                if flat_pre_ids.numel() > 0 else 0,
                int(flat_mapped_ids.min().item())
                if flat_mapped_ids.numel() > 0 else -1,
                int(flat_mapped_ids.max().item())
                if flat_mapped_ids.numel() > 0 else -1,
                int(torch.unique(flat_mapped_ids).numel())
                if flat_mapped_ids.numel() > 0 else 0,
                self.num_local_experts,
                active_local_num_experts,
                int(len(nonzero_experts)),
                top_expert_pairs,
                self.input_splits,
                self.output_splits,
                self.num_out_tokens,
            )
            if invalid_local_slots:
                logger.warning(
                    "Lossless pre-shrink invalid local slot: rank=%s layer=%s tag=%s invalid_slots=%s active_local_num_experts=%s top_local_tokens=%s",
                    self.ep_rank,
                    debug_info.get("layer_idx"),
                    debug_info.get("tag"),
                    invalid_local_slots,
                    active_local_num_experts,
                    top_expert_pairs,
                )
        expected_input_tokens = sum(self.input_splits)
        actual_input_tokens = permutated_local_input_tokens.shape[0]
        if actual_input_tokens != expected_input_tokens:
            debug_ctx = self._debug_context()
            logger.error(
                "AllToAll input split mismatch at %s: rank=%s ep_size=%s num_experts=%s num_local_experts=%s topk_shape=%s num_out_tokens=%s input_splits=%s sum_input_splits=%s actual_input_tokens=%s",
                debug_ctx,
                self.ep_rank,
                self.ep_size,
                self.num_experts,
                self.num_local_experts,
                tuple(topk_ids.shape),
                self.num_out_tokens,
                self.input_splits,
                expected_input_tokens,
                actual_input_tokens,
            )
            raise AssertionError("AllToAll input split mismatch")

        dynamic_scale_after_all2all = None
        if self.with_quant:
            permutated_local_input_tokens, dynamic_scale = torch_npu.npu_dynamic_quant(
                permutated_local_input_tokens)

            _, dynamic_scale_after_all2all, permute2_ep_all_to_all_handle = async_all_to_all(
                dynamic_scale,
                self.output_splits,
                self.input_splits,
                self.ep_group,
            )
            permute2_ep_all_to_all_handle.wait()
            dynamic_scale.untyped_storage().resize_(0)

        _, global_input_tokens, permute1_ep_all_to_all_handle = async_all_to_all(
            permutated_local_input_tokens,
            self.output_splits,
            self.input_splits,
            self.ep_group,
        )
        permute1_ep_all_to_all_handle.wait()
        permutated_local_input_tokens.untyped_storage().resize_(0)

        global_input_tokens, dynamic_scale = self._dispatch_postprocess(
            global_input_tokens, dynamic_scale_after_all2all)
        return {
            "hidden_states": global_input_tokens,
            "group_list": tokens_per_expert,
            "dynamic_scale": dynamic_scale,
            "group_list_type": 1
        }

    def token_combine(self,
                      hidden_states: torch.Tensor,
                      bias: torch.Tensor = None):
        assert bias is None, "Bias is not supported in MoEAlltoAllvTokenDispatcher."

        hidden_states = self._combine_preprocess(hidden_states)

        # Perform expert parallel AlltoAll communication
        # hidden_states: [SEQL, H] -> [SEQL, H/TP]
        _, permutated_local_input_tokens, handle = async_all_to_all(
            hidden_states, self.input_splits, self.output_splits,
            self.ep_group)
        handle.wait()
        hidden_states.untyped_storage().resize_(0)

        output = self._combine_postprocess(permutated_local_input_tokens)

        # these values are no longer used, so they need to be set to None for memory release.
        self.input_splits = None
        self.output_splits = None
        self.num_global_tokens_per_local_expert = None
        self.topk_weights = None
        self.reversed_local_input_permutation_mapping = None
        self.reversed_global_input_permutation_mapping = None
        self.global_input_tokens_local_experts_indices = None

        return output

    def _dispatch_preprocess(self, hidden_states, topk_ids):
        assert self.hidden_shape is not None
        hidden_states = hidden_states.view(-1, self.hidden_shape[-1])
        tokens_per_expert = self._preprocess(topk_ids)

        self.hidden_shape_before_permute = hidden_states.shape
        if self.num_out_tokens == 0:
            reversed_local_input_permutation_mapping = torch.empty(
                (0, ),
                dtype=torch.int32,
                device=hidden_states.device,
            )
            permutated_local_input_tokens = hidden_states.new_empty(
                (0, hidden_states.shape[-1]))
            return (permutated_local_input_tokens,
                    reversed_local_input_permutation_mapping,
                    tokens_per_expert)

        permutated_local_input_tokens, reversed_local_input_permutation_mapping = torch_npu.npu_moe_token_permute(
            tokens=hidden_states,
            indices=topk_ids,
            num_out_tokens=self.num_out_tokens,
        )
        return permutated_local_input_tokens, reversed_local_input_permutation_mapping, tokens_per_expert

    def _debug_context(self) -> str:
        forward_context = get_forward_context()
        layer_idx = getattr(forward_context, "layer_idx", None)
        return f"layer={layer_idx}" if layer_idx is not None else "layer=unknown"

    def _should_use_host_metadata_gather(self) -> bool:
        forward_context = get_forward_context()
        force_host = bool(
            getattr(forward_context, "hybrid_force_host_alltoall_metadata",
                    False))
        if force_host:
            debug_info = getattr(forward_context, "elastic_debug_info", None)
            if debug_info is not None and not debug_info.get(
                    "_alltoall_host_metadata_logged", False):
                logger.info(
                    "AllToAll metadata gather using host path: %s rank=%s "
                    "ep_size=%s stage=%s num_experts=%s "
                    "num_local_experts=%s",
                    self._debug_context(),
                    self.ep_rank,
                    self.ep_size,
                    getattr(forward_context, "hybrid_stage_active_ranks", None),
                    self.num_experts,
                    self.num_local_experts,
                )
                debug_info["_alltoall_host_metadata_logged"] = True
            return True
        return bool(
            force_host)

    def _gather_global_tokens_per_expert_host(
            self, num_local_tokens_per_expert: torch.Tensor) -> torch.Tensor:
        forward_context = get_forward_context()
        stage_size = getattr(forward_context, "hybrid_stage_active_ranks", None)
        if self.ep_rank == 0 and self._hybrid_host_metadata_logged_stage != stage_size:
            logger.info(
                "Hybrid AllToAll metadata gather using host path: %s rank=%s ep_size=%s stage=%s num_experts=%s num_local_experts=%s",
                self._debug_context(),
                self.ep_rank,
                self.ep_size,
                stage_size,
                self.num_experts,
                self.num_local_experts,
            )
            self._hybrid_host_metadata_logged_stage = stage_size

        local_counts_cpu = num_local_tokens_per_expert.to(
            device="cpu", non_blocking=False)
        local_counts_list = [int(x) for x in local_counts_cpu.tolist()]
        gathered_counts: list[Optional[list[int]]] = [None] * self.ep_size
        torch.distributed.all_gather_object(
            gathered_counts, local_counts_list, group=self.ep_metadata_group)
        for rank_idx, counts in enumerate(gathered_counts):
            if counts is None or len(counts) != self.num_experts:
                logger.error(
                    "Hybrid AllToAll host metadata gather malformed at %s: rank=%s ep_size=%s stage=%s src_rank=%s expected_num_experts=%s actual_len=%s",
                    self._debug_context(),
                    self.ep_rank,
                    self.ep_size,
                    stage_size,
                    rank_idx,
                    self.num_experts,
                    -1 if counts is None else len(counts),
                )
                raise AssertionError(
                    "Hybrid AllToAll host metadata gather malformed")
        return torch.tensor(gathered_counts, dtype=torch.int64)

    def _preprocess(self, topk_ids: torch.Tensor) -> torch.Tensor:
        flat_topk_ids = topk_ids.reshape(-1).to(torch.int64)
        if flat_topk_ids.numel() > 0:
            topk_min = int(flat_topk_ids.min().item())
            topk_max = int(flat_topk_ids.max().item())
            if topk_min < 0 or topk_max >= self.num_experts:
                debug_ctx = self._debug_context()
                logger.error(
                    "AllToAll topk id out of range at %s: rank=%s ep_size=%s num_experts=%s topk_shape=%s topk_min=%s topk_max=%s",
                    debug_ctx,
                    self.ep_rank,
                    self.ep_size,
                    self.num_experts,
                    tuple(topk_ids.shape),
                    topk_min,
                    topk_max,
                )
                raise AssertionError("AllToAll topk id out of range")

        # `histc`/`bincount` on this NPU path produced all-zero counts for valid
        # topk ids. Use explicit integer accumulation instead.
        num_local_tokens_per_expert = torch.zeros(
            self.num_experts,
            dtype=torch.int64,
            device=topk_ids.device,
        )
        if flat_topk_ids.numel() > 0:
            num_local_tokens_per_expert.scatter_add_(
                0,
                flat_topk_ids,
                torch.ones_like(flat_topk_ids, dtype=torch.int64),
            )

        ep_size = self.ep_size
        expected_num_experts = ep_size * self.num_local_experts
        if self.num_experts != expected_num_experts:
            debug_ctx = self._debug_context()
            logger.error(
                "AllToAll expert layout mismatch at %s: rank=%s ep_size=%s num_experts=%s expected_num_experts=%s num_local_experts=%s topk_shape=%s",
                debug_ctx,
                self.ep_rank,
                ep_size,
                self.num_experts,
                expected_num_experts,
                self.num_local_experts,
                tuple(topk_ids.shape),
            )
            raise AssertionError("AllToAll expert layout mismatch")

        # Dropless
        self.num_out_tokens = topk_ids.numel()

        # ===================================================
        # Calculate input_splits, output_splits for alltoall-v.
        # ===================================================
        input_splits_tensor = num_local_tokens_per_expert.reshape(
            ep_size, self.num_local_experts).sum(dim=1)
        local_input_total = int(input_splits_tensor.sum().item())
        self.input_splits = input_splits_tensor.cpu().tolist()
        use_host_metadata_gather = bool(ep_size > 1
                                        and self._should_use_host_metadata_gather())
        if ep_size == 1:
            num_global_tokens_per_expert = num_local_tokens_per_expert.reshape(
                1, self.num_experts)
        elif use_host_metadata_gather:
            num_global_tokens_per_expert = (
                self._gather_global_tokens_per_expert_host(
                    num_local_tokens_per_expert))
        else:
            num_global_tokens_per_expert = _gather_along_first_dim(
                num_local_tokens_per_expert, self.ep_group).reshape(
                    ep_size, self.num_experts)
        self.num_global_tokens_per_local_expert = num_global_tokens_per_expert[:, self.local_expert_indices[
            0]:self.local_expert_indices[-1] + 1]
        if self.num_global_tokens_per_local_expert is None:
            raise ValueError(
                "num_global_tokens_per_local_expert must be set before sum.")
        output_splits_tensor = self.num_global_tokens_per_local_expert.sum(
            dim=-1)
        self.output_splits = output_splits_tensor.cpu().tolist()
        num_tokens_per_local_expert = self.num_global_tokens_per_local_expert.sum(
            dim=0)
        total_input_splits = sum(self.input_splits)
        total_output_splits = sum(self.output_splits)
        if local_input_total != self.num_out_tokens:
            debug_ctx = self._debug_context()
            logger.error(
                "AllToAll device input split total mismatch at %s: rank=%s ep_size=%s num_out_tokens=%s local_input_total=%s topk_shape=%s topk_min=%s topk_max=%s",
                debug_ctx,
                self.ep_rank,
                ep_size,
                self.num_out_tokens,
                local_input_total,
                tuple(topk_ids.shape),
                int(topk_ids.min().item()) if topk_ids.numel() > 0 else -1,
                int(topk_ids.max().item()) if topk_ids.numel() > 0 else -1,
            )
            raise AssertionError("AllToAll device input split total mismatch")
        if total_input_splits != self.num_out_tokens:
            debug_ctx = self._debug_context()
            logger.error(
                "AllToAll input split total mismatch at %s: rank=%s ep_size=%s num_out_tokens=%s sum_input_splits=%s input_splits=%s topk_shape=%s topk_min=%s topk_max=%s",
                debug_ctx,
                self.ep_rank,
                ep_size,
                self.num_out_tokens,
                total_input_splits,
                self.input_splits,
                tuple(topk_ids.shape),
                int(topk_ids.min().item()) if topk_ids.numel() > 0 else -1,
                int(topk_ids.max().item()) if topk_ids.numel() > 0 else -1,
            )
            raise AssertionError("AllToAll input split total mismatch")
        expected_output_splits = int(
            self.num_global_tokens_per_local_expert.sum().item())
        if total_output_splits != expected_output_splits:
            debug_ctx = self._debug_context()
            logger.error(
                "AllToAll output split total mismatch at %s: rank=%s ep_size=%s sum_output_splits=%s expected_output_splits=%s output_splits=%s",
                debug_ctx,
                self.ep_rank,
                ep_size,
                total_output_splits,
                expected_output_splits,
                self.output_splits,
            )
            raise AssertionError("AllToAll output split total mismatch")
        # ===================================================
        # num_global_tokens_per_expert: [ep_size, num_experts]
        # num_global_tokens_per_local_expert: [ep_size, num_local_experts]
        # num_tokens_per_local_expert: [num_local_experts]
        # ===================================================

        num_tokens_per_local_expert = num_tokens_per_local_expert.to(
            device=torch.npu.current_device())
        if self.num_local_experts > 1:
            if self.num_global_tokens_per_local_expert is None:
                raise ValueError(
                    "num_global_tokens_per_local_expert must be set before operations."
                )
            if self.num_global_tokens_per_local_expert.device != self.expert_ids_per_ep_rank.device:
                self.num_global_tokens_per_local_expert = (
                    self.num_global_tokens_per_local_expert.to(
                        device=self.expert_ids_per_ep_rank.device,
                        dtype=torch.int64,
                        non_blocking=False))
            self.global_input_tokens_local_experts_indices = torch.repeat_interleave(
                self.expert_ids_per_ep_rank,
                self.num_global_tokens_per_local_expert.ravel().to(
                    device=self.expert_ids_per_ep_rank.device,
                    dtype=torch.int64,
                    non_blocking=False))
        else:
            # TODO: This full synchronization can be a performance bottleneck.
            # A more granular sync (e.g., blocking D2H copies) should be investigated.
            torch.npu.synchronize()

        return num_tokens_per_local_expert

    def _dispatch_postprocess(self, global_input_tokens, dynamic_scale=None):
        # Early return if no local experts or no tokens
        if self.num_local_experts <= 1:
            return global_input_tokens, None

        # Handle quantized case
        if self.with_quant:
            assert self.global_input_tokens_local_experts_indices is not None, \
            "global_input_tokens_local_experts_indices must be initialized before calling _dispatch_postprocess"
            expert_idx_2d = self.global_input_tokens_local_experts_indices.unsqueeze(
                -1)
            active_num = self.global_input_tokens_local_experts_indices.numel()

            # Handle case with no active tokens
            if active_num <= 0:
                self.reversed_global_input_permutation_mapping = self.global_input_tokens_local_experts_indices
                return global_input_tokens, dynamic_scale

            # Process with active tokens
            global_input_tokens, self.reversed_global_input_permutation_mapping, _, expanded_scale = torch_npu.npu_moe_init_routing_v2(
                global_input_tokens,
                expert_idx_2d,
                scale=dynamic_scale,
                active_num=active_num,
                expert_capacity=0,
                expert_num=self.num_local_experts,
                expert_tokens_num_type=1,
                expert_tokens_num_flag=True,
                active_expert_range=[0, self.num_local_experts],
                quant_mode=-1,
                row_idx_type=0)
            return global_input_tokens, expanded_scale

        # Handle non-quantized case
        global_input_tokens, self.reversed_global_input_permutation_mapping = torch_npu.npu_moe_token_permute(
            global_input_tokens,
            self.global_input_tokens_local_experts_indices)
        return global_input_tokens, None

    def _combine_preprocess(self, hidden_states):
        # Unpermutation 2: expert output to AlltoAll input
        if hidden_states.shape[0] > 0 and self.num_local_experts > 1:
            hidden_states = torch_npu.npu_moe_token_unpermute(
                hidden_states, self.reversed_global_input_permutation_mapping)

        return hidden_states

    def _combine_postprocess(self, permutated_local_input_tokens):
        if (self.hidden_shape_before_permute is not None
                and len(self.hidden_shape_before_permute) > 0
                and int(self.hidden_shape_before_permute[0]) == 0):
            return permutated_local_input_tokens.new_empty(
                self.hidden_shape_before_permute)
        # Unpermutation 1: AlltoAll output to output
        output = torch_npu.npu_moe_token_unpermute(
            permuted_tokens=permutated_local_input_tokens,
            sorted_indices=self.reversed_local_input_permutation_mapping.to(
                torch.int32),
            probs=self.topk_weights,
            restore_shape=self.hidden_shape_before_permute)

        # Reshape the output tensor
        output = output.view(self.hidden_shape)
        return output
