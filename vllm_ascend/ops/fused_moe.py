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
from collections import defaultdict
from contextlib import contextmanager, nullcontext
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
from vllm_ascend.ascend_forward_context import FusedMoEState, MoECommType
from vllm_ascend.distributed.parallel_state import get_mc2_group
from vllm_ascend import envs as envs_ascend
from vllm_ascend.eplb.core.eplb_utils import (
    determine_default_expert_map, determine_default_log2phy_map,
    determine_redundant_replica_expert_map,
    determine_redundant_replica_log2phy_map)
from vllm_ascend.ops.expert_load_balancer import ExpertLoadBalancer
from vllm_ascend.ops.moe.experts_selector import return_row_idx, select_experts
from vllm_ascend.ops.moe.moe_mlp import unified_apply_mlp
from vllm_ascend.ops.moe.moe_comm_method import (get_moe_comm_method,
                                                 setup_moe_comm_method)
from vllm_ascend.utils import (ACL_FORMAT_FRACTAL_NZ,
                               get_all_reduce_merge_state,
                               get_rm_router_logits_state, is_310p,
                               vllm_version_is)
from vllm.utils.moe_stats import moe_stats
import time

logger = init_logger(__name__)
_MODE3_TIMING_COUNTS = defaultdict(int)


def _env_flag(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).lower() in ("1", "true", "yes", "on")


def _env_int(name: str, default: int, *, minimum: int = 0) -> int:
    try:
        return max(minimum, int(os.getenv(name, str(default))))
    except (TypeError, ValueError):
        return max(minimum, default)


@contextmanager
def _dummy_profile_range(message: str):
    if not _env_flag("VLLM_ASCEND_DUMMY_WASTE_PROFILE_MARKERS", "0"):
        yield
        return
    range_id = None
    try:
        from torch_npu.npu import mstx
        range_id = mstx.range_start(message=message)
    except Exception:
        yield
        return
    try:
        yield
    finally:
        try:
            mstx.range_end(range_id)
        except Exception:
            pass


def _profile_rank() -> int:
    try:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            return int(torch.distributed.get_rank())
    except Exception:
        pass
    return -1


def _parse_layer_filter(value: str) -> Optional[set[int]]:
    value = value.strip().lower()
    if value in ("", "all", "*"):
        return None
    layers: set[int] = set()
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        try:
            layers.add(int(item))
        except ValueError:
            continue
    return layers


def _new_npu_event(*, enable_timing: bool = False) -> torch.npu.Event:
    try:
        return torch.npu.Event(enable_timing=enable_timing)
    except TypeError:
        return torch.npu.Event()


def _elapsed_ms(start_event: Any, end_event: Any) -> float:
    if start_event is None or end_event is None:
        return -1.0
    try:
        end_event.synchronize()
        return float(start_event.elapsed_time(end_event))
    except Exception:
        return -1.0


def _event_record(event: Optional[Any]) -> None:
    if event is None:
        return
    try:
        event.record()
    except Exception:
        pass


_MODE3_SUBMIT_TIMING_KEYS = (
    "submit_event_alloc_us",
    "submit_stream_wait_us",
    "submit_prefetch_wait_stream_us",
    "submit_start_event_record_us",
    "submit_populate_us",
    "submit_order_us",
    "submit_assign_us",
    "submit_layer_local_check_us",
    "submit_npu_us",
    "submit_cpu_us",
    "submit_cpu_direct_async_us",
    "submit_cpu_stage_async_us",
    "submit_plan_log_us",
    "submit_expert_map_us",
    "submit_dispatch_cache_us",
    "submit_slot_state_us",
    "submit_post_cpu_wait_us",
    "submit_ready_record_us",
)


def _timing_float(timing: Optional[dict[str, Any]], key: str) -> float:
    if not timing:
        return -1.0
    try:
        return float(timing.get(key, -1.0))
    except Exception:
        return -1.0


def _mode3_submit_accounted_us(timing: Optional[dict[str, Any]]) -> float:
    if not timing:
        return -1.0
    total = 0.0
    seen = False
    for key in _MODE3_SUBMIT_TIMING_KEYS:
        value = _timing_float(timing, key)
        if value >= 0.0:
            total += value
            seen = True
    return total if seen else -1.0


def _tensor_is_pinned(tensor: Optional[torch.Tensor]) -> int:
    if tensor is None or tensor.device.type != "cpu":
        return 0
    try:
        return int(bool(tensor.is_pinned()))
    except Exception:
        return -1


def _tensor_is_contiguous(tensor: Optional[torch.Tensor]) -> int:
    if tensor is None:
        return 0
    try:
        return int(bool(tensor.is_contiguous()))
    except Exception:
        return -1


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


def _stable_cpu_transfer_source(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.device.type != "cpu":
        return tensor
    stable = tensor.detach()
    if not stable.is_contiguous():
        stable = stable.contiguous()
    # Pinned-host -> NPU transfers have been the most fragile part of the
    # mode3 path on Ascend. Use a fresh pageable CPU copy before sending rows
    # to NPU runtime slots so layer-entry population favors stability over
    # overlap.
    stable = stable.to(device="cpu", copy=True)
    return stable


def _lossless_weight_row_means(weight: Optional[torch.Tensor],
                               rows: Optional[list[int]] = None,
                               limit: int = 4) -> list[tuple[int, float]]:
    if weight is None or weight.numel() == 0:
        return []
    row_count = int(weight.shape[0])
    if rows is None:
        indices = list(range(min(row_count, limit)))
    else:
        indices = []
        for row in rows:
            row = int(row)
            if 0 <= row < row_count and row not in indices:
                indices.append(row)
            if len(indices) >= limit:
                break
    result: list[tuple[int, float]] = []
    for row in indices:
        result.append((row, round(float(weight[row].float().abs().mean().item()),
                                  6)))
    return result


def _lossless_weight_meta(weight: Optional[torch.Tensor]) -> dict[str, Any]:
    if weight is None:
        return {
            "shape": None,
            "dtype": None,
            "device": None,
            "stride": None,
            "ptr": None,
            "storage_offset": None,
            "format": None,
            "contiguous": None,
        }
    return {
        "shape": tuple(weight.shape),
        "dtype": str(weight.dtype),
        "device": str(weight.device),
        "stride": tuple(weight.stride()),
        "ptr": int(weight.data_ptr()),
        "storage_offset": int(weight.storage_offset()),
        "format": _lossless_weight_format(weight),
        "contiguous": _tensor_is_contiguous(weight),
    }


def _lossless_weight_format(weight: Optional[torch.Tensor]) -> str:
    if weight is None:
        return "none"
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        return "cpu"
    try:
        import torch_npu  # type: ignore
        return str(torch_npu.get_npu_format(weight))
    except Exception:
        return "unknown"


def _npu_weight_format_value(weight: Optional[torch.Tensor]) -> Optional[Any]:
    if weight is None or weight.device.type != "npu":
        return None
    try:
        return torch_npu.get_npu_format(weight)
    except Exception:
        return None


def _npu_weight_format_matches(weight: Optional[torch.Tensor],
                               target_format: Any) -> bool:
    current_format = _npu_weight_format_value(weight)
    if current_format is None:
        return False
    return current_format == target_format or str(current_format) == str(
        target_format)


def _mode1_reload_allocator_trim_enabled() -> bool:
    try:
        floor = int(
            os.getenv("VLLM_ASCEND_ELASTIC_MIN_COMPUTE_GROUP_SIZE", "0")
            or "0")
    except ValueError:
        floor = 0
    default = "1" if 0 < floor <= 2 else "0"
    return _env_flag("VLLM_ASCEND_MODE1_RELOAD_TRIM_ALLOCATOR", default)


def _trim_npu_allocator_for_mode1_reload(layer: Any, reason: str) -> None:
    if not _mode1_reload_allocator_trim_enabled():
        return
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        return
    try:
        if _env_flag("VLLM_ASCEND_MODE1_RELOAD_SYNC_ON_TRIM", "1"):
            torch.npu.synchronize()
        gc.collect()
        torch.npu.empty_cache()
    except Exception:
        return
    if getattr(layer, "layer_idx", -1) == 0 and not getattr(
            layer, "_mode1_reload_trim_logged", False):
        logger.info(
            "Mode1 reload allocator trim enabled: floor=%s reason=%s",
            os.getenv("VLLM_ASCEND_ELASTIC_MIN_COMPUTE_GROUP_SIZE", ""),
            reason,
        )
        layer._mode1_reload_trim_logged = True


def _maybe_npu_format_cast_for_mode1_reload(weight: torch.Tensor,
                                            target_format: Any, *,
                                            layer: Any,
                                            weight_name: str,
                                            mode1_reload: bool) -> torch.Tensor:
    if weight.device.type != "npu":
        return weight
    if not mode1_reload:
        return torch_npu.npu_format_cast(weight, target_format)

    current_format = _npu_weight_format_value(weight)
    force_cast = _env_flag("VLLM_ASCEND_MODE1_RELOAD_FORCE_FORMAT_CAST", "0")
    skip_same = _env_flag("VLLM_ASCEND_MODE1_RELOAD_SKIP_SAME_FORMAT_CAST", "1")
    same_format = (current_format is not None and
                   (current_format == target_format
                    or str(current_format) == str(target_format)))
    if skip_same and same_format and not force_cast:
        if getattr(layer, "layer_idx", -1) == 0 and not getattr(
                layer, f"_mode1_reload_skip_cast_logged_{weight_name}", False):
            logger.info(
                "Mode1 reload skips redundant npu_format_cast: weight=%s "
                "format=%s target=%s floor=%s",
                weight_name,
                current_format,
                target_format,
                os.getenv("VLLM_ASCEND_ELASTIC_MIN_COMPUTE_GROUP_SIZE", ""),
            )
            setattr(layer, f"_mode1_reload_skip_cast_logged_{weight_name}",
                    True)
        return weight

    if getattr(layer, "layer_idx", -1) == 0 and not getattr(
            layer, f"_mode1_reload_do_cast_logged_{weight_name}", False):
        logger.info(
            "Mode1 reload executes npu_format_cast: weight=%s format=%s "
            "target=%s force=%s floor=%s",
            weight_name,
            current_format,
            target_format,
            force_cast,
            os.getenv("VLLM_ASCEND_ELASTIC_MIN_COMPUTE_GROUP_SIZE", ""),
        )
        setattr(layer, f"_mode1_reload_do_cast_logged_{weight_name}", True)
    return torch_npu.npu_format_cast(weight, target_format)


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
        import torch_npu  # type: ignore
        target_format = torch_npu.get_npu_format(weight)
        buffer = torch_npu.npu_format_cast(buffer, target_format)
    except Exception:
        pass
    return buffer


def _allocate_plain_buffer_like(weight: torch.Tensor,
                                row_count: int,
                                *,
                                dtype: Optional[torch.dtype] = None
                                ) -> torch.Tensor:
    return torch.empty((row_count, ) + tuple(weight.shape[1:]),
                       device=weight.device,
                       dtype=weight.dtype if dtype is None else dtype)


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


class _Mode3DoubleBufferSlot:

    def __init__(self) -> None:
        self.w13 = None
        self.w2 = None
        self.cpu_stage_w13 = None
        self.cpu_stage_w2 = None
        self.expert_map = None
        self.layer_idx = None
        self.expert_ids: tuple[int, ...] = ()
        self.cpu_stage_slot_ids: tuple[int, ...] = ()
        self.cpu_stage_cpu_slots: tuple[int, ...] = ()
        self.cpu_stage_count = 0
        self.valid_expert_count = 0
        self.source_from_npu = 0
        self.source_from_cpu = 0
        self.inflight_prefetch = False
        self.has_async_cpu_copy = False
        self.has_async_cpu_pack = False
        self.has_async_cpu_direct = False
        self.needs_sync_cpu_fill = False
        self.uses_layer_local_buffer = False
        self.ready_event = torch.npu.Event()
        self.cpu_ready_event = torch.npu.Event()
        self.cpu_pack_event = torch.npu.Event()
        self.prefetch_start_event = None
        self.prefetch_end_event = None
        self.prefetch_timing: dict[str, Any] = {}
        self.prefetch_cpu_path = "none"
        self.dispatch_log2phy = None
        self.dispatch_num_experts = 0
        self.dispatch_signature: Optional[tuple[Any, ...]] = None
        self.dispatch_active_rank_count = 0
        self.dispatch_owned_per_rank = 0


class Mode3DoubleBufferManager:

    def __init__(self, model_instance: Any,
                 prefetch_stream: Optional[torch.npu.Stream]) -> None:
        self.prefetch_stream = prefetch_stream
        self.cpu_prefetch_stream = None
        if prefetch_stream is not None:
            try:
                self.cpu_prefetch_stream = torch.npu.Stream()
            except Exception:
                self.cpu_prefetch_stream = None
        self.slots = [_Mode3DoubleBufferSlot(), _Mode3DoubleBufferSlot()]
        self.layer_lookup = self._build_layer_lookup(model_instance)
        self.prefetch_wait_us = defaultdict(float)
        self.prefetch_hit = defaultdict(int)
        self.prefetch_wait_count = defaultdict(int)
        self.last_bind_timing: dict[str, Any] = {}
        self._logged_slot_bindings: set[tuple[int, int, int]] = set()
        self._logged_prefetches: set[tuple[int, int]] = set()
        self._logged_prefetch_deferrals: set[tuple[int, int]] = set()
        self._logged_cpu_fill_deferrals: set[tuple[int, int]] = set()
        self._logged_async_cpu_stage: set[tuple[int, int]] = set()
        self._logged_transfer_plans: set[tuple[int, int, str]] = set()
        self._copy_format_diag_count = 0
        self._dispatch_remap_cache: dict[
            tuple[str, tuple[Any, ...]], dict[str, Any]] = {}
        self._slot_expert_map_cache: dict[
            tuple[str, int, tuple[int, ...]], torch.Tensor] = {}
        self.enable_async_npu_prefetch = _env_flag(
            "VLLM_ASCEND_MODE3_ASYNC_NPU_PREFETCH")
        self.enable_async_cpu_stage = _env_flag(
            "VLLM_ASCEND_MODE3_ASYNC_CPU_STAGE")
        self.enable_async_cpu_pack = _env_flag(
            "VLLM_ASCEND_MODE3_ASYNC_CPU_PACK")
        self.enable_direct_cpu_slot = _env_flag(
            "VLLM_ASCEND_MODE3_DIRECT_CPU_SLOT")
        self.enable_device_ready_wait = _env_flag(
            "VLLM_ASCEND_MODE3_DEVICE_READY_WAIT")
        self.enable_bulk_npu_copy = _env_flag(
            "VLLM_ASCEND_MODE3_BULK_NPU_COPY", "1")
        self.enable_bulk_cpu_stage = _env_flag(
            "VLLM_ASCEND_MODE3_BULK_CPU_STAGE", "1")
        self.enable_bulk_cpu_direct = _env_flag(
            "VLLM_ASCEND_MODE3_BULK_CPU_DIRECT")
        self.enable_layer_local_buffer = _env_flag(
            "VLLM_ASCEND_MODE3_LAYER_LOCAL_BUFFER")
        self.enable_active_rows_sync = _env_flag(
            "VLLM_ASCEND_MODE3_ACTIVE_ROWS_SYNC")
        self.expert_token_nums_type = _env_int(
            "VLLM_ASCEND_MODE3_EXPERT_TOKEN_NUMS_TYPE", 0, minimum=0)
        if self.expert_token_nums_type not in (0, 1):
            self.expert_token_nums_type = 0
        self.use_fused_experts_path = _env_flag(
            "VLLM_ASCEND_MODE3_USE_FUSED_EXPERTS_PATH", "1")
        self.enable_transfer_logs = _env_flag(
            "VLLM_ASCEND_MODE3_TRANSFER_LOG")
        self.enable_transfer_plan_logs = _env_flag(
            "VLLM_ASCEND_MODE3_TRANSFER_PLAN_LOG")
        self.transfer_plan_first_n = _env_int(
            "VLLM_ASCEND_MODE3_TRANSFER_PLAN_FIRST_N", 4, minimum=0)
        self.enable_timing_logs = _env_flag(
            "VLLM_ASCEND_MODE3_TIMING_LOG")
        self.enable_timing_sync = _env_flag(
            "VLLM_ASCEND_MODE3_TIMING_SYNC")
        self.timing_every = _env_int("VLLM_ASCEND_MODE3_TIMING_EVERY",
                                     512,
                                     minimum=1)
        self.timing_first_n = _env_int("VLLM_ASCEND_MODE3_TIMING_FIRST_N",
                                       1,
                                       minimum=0)
        self.timing_layers = _parse_layer_filter(
            os.getenv("VLLM_ASCEND_MODE3_TIMING_LAYERS", "all"))
        self.enable_copy_format_diag = _env_flag(
            "VLLM_ASCEND_MODE3_COPY_FORMAT_DIAG")
        self.copy_format_diag_first_n = _env_int(
            "VLLM_ASCEND_MODE3_COPY_FORMAT_DIAG_FIRST_N", 8, minimum=0)
        self.runtime_buffer_format = os.getenv(
            "VLLM_ASCEND_MODE3_RUNTIME_BUFFER_FORMAT", "plain").strip().lower()
        if self.runtime_buffer_format not in ("formatted", "plain"):
            self.runtime_buffer_format = "plain"
        self._logged_runtime_buffer_format = False

    @staticmethod
    def _build_layer_lookup(model_instance: Any) -> dict[int, Any]:
        layer_lookup: dict[int, Any] = {}
        model = getattr(model_instance, "model", None)
        layers = getattr(model, "layers", None)
        if layers is None:
            return layer_lookup
        for decoder_layer in layers:
            mlp = getattr(decoder_layer, "mlp", None)
            experts = getattr(mlp, "experts", None)
            if experts is None:
                continue
            layer_idx = getattr(experts, "layer_idx", None)
            if isinstance(layer_idx, int):
                layer_lookup[layer_idx] = experts
        return layer_lookup

    @staticmethod
    def _slot_id_for_layer(layer_idx: int) -> int:
        return int(layer_idx) & 1

    def should_profile_layer(self, layer: Any, path: str) -> bool:
        if not self.enable_timing_logs:
            return False
        layer_idx = int(getattr(layer, "layer_idx", -1))
        if self.timing_layers is not None and layer_idx not in self.timing_layers:
            return False
        active_rank_count = len(
            getattr(layer, "lossless_hybrid_active_ranks", []))
        rank = int(getattr(layer, "rank", -1))
        key = (rank, path, active_rank_count, layer_idx)
        _MODE3_TIMING_COUNTS[key] += 1
        count = _MODE3_TIMING_COUNTS[key]
        return count <= self.timing_first_n or count % self.timing_every == 0

    def new_timing_event(self) -> Optional[torch.npu.Event]:
        if not self.enable_timing_sync:
            return None
        return _new_npu_event(enable_timing=True)

    def _prefetch_device_ms(self, timing: Optional[dict[str, Any]]) -> float:
        if not timing:
            return -1.0
        return _elapsed_ms(timing.get("start_event"), timing.get("end_event"))

    def _get_next_layer(self, current_layer_idx: int) -> Optional[Any]:
        for layer_idx in sorted(self.layer_lookup):
            if layer_idx > current_layer_idx:
                return self.layer_lookup[layer_idx]
        return None

    @staticmethod
    def _build_slot_expert_map(layer: Any, expert_ids: list[int],
                               device: torch.device) -> torch.Tensor:
        expert_map = torch.full((int(layer.elastic_original_num_experts), ),
                                -1,
                                dtype=torch.int32,
                                device=device)
        seen_experts: set[int] = set()
        for slot_idx, expert_id in enumerate(expert_ids):
            if int(expert_id) in seen_experts:
                continue
            seen_experts.add(int(expert_id))
            expert_map[int(expert_id)] = slot_idx
        return expert_map

    def _get_cached_slot_expert_map(
            self, layer: Any, expert_ids: list[int],
            device: torch.device) -> tuple[torch.Tensor, bool]:
        expert_ids_key = tuple(int(expert_id) for expert_id in expert_ids)
        cache_key = (str(device), int(layer.elastic_original_num_experts),
                     expert_ids_key)
        cached = self._slot_expert_map_cache.get(cache_key)
        if cached is not None:
            return cached, True
        expert_map = self._build_slot_expert_map(layer, expert_ids, device)
        self._slot_expert_map_cache[cache_key] = expert_map
        return expert_map, False

    @staticmethod
    def _ordered_mode3_slot_expert_ids(layer: Any) -> list[int]:
        owned_expert_ids = [
            int(expert_id)
            for expert_id in getattr(layer, "lossless_hybrid_owned_expert_ids",
                                     [])
        ]
        if not owned_expert_ids:
            return []
        resident_capacity = int(
            getattr(layer, "lossless_hybrid_resident_capacity", 0))
        primary_slots = {
            int(expert_id): int(local_slot)
            for expert_id, local_slot in getattr(
                layer, "lossless_mode3_primary_prefix_local_slots",
                {}).items()
        }
        ordered_primary = [
            expert_id for expert_id, _local_slot in sorted(
                primary_slots.items(), key=lambda item: item[1])
            if expert_id in set(owned_expert_ids)
        ]
        if resident_capacity > 0:
            ordered_primary = ordered_primary[:resident_capacity]
        ordered_primary_set = set(ordered_primary)
        ordered_cpu_only = [
            expert_id for expert_id in owned_expert_ids
            if expert_id not in ordered_primary_set
        ]
        slot_expert_ids = ordered_primary + ordered_cpu_only
        if len(slot_expert_ids) != len(owned_expert_ids):
            raise RuntimeError(
                "Mode3 slot expert ordering mismatch at layer="
                f"{getattr(layer, 'layer_idx', -1)}: "
                f"owned={len(owned_expert_ids)} ordered={len(slot_expert_ids)}")
        return slot_expert_ids

    def _get_cached_dispatch_remap(
            self, layer: Any,
            device: torch.device) -> tuple[Optional[torch.Tensor], int,
                                           Optional[tuple[Any, ...]], int, int]:
        topology_signature = _hybrid_dispatch_topology_signature(layer)
        rank_owned = getattr(layer, "lossless_hybrid_rank_owned_expert_ids", None)
        if not rank_owned:
            return None, 0, topology_signature, 0, 0
        active_rank_count = len(getattr(layer, "lossless_hybrid_active_ranks", []))
        if active_rank_count <= 0:
            return None, 0, topology_signature, 0, 0
        owned_per_rank = len(rank_owned[0]) if rank_owned else 0
        if owned_per_rank <= 0:
            return None, 0, topology_signature, active_rank_count, 0
        cache_key = (str(device), topology_signature)
        cached = self._dispatch_remap_cache.get(cache_key)
        if cached is None:
            dispatch_log2phy, dispatch_num_experts = _build_dispatch_log2phy_tensor(
                int(layer.elastic_original_num_experts),
                rank_owned,
                owned_per_rank,
                device,
            )
            cached = {
                "dispatch_log2phy": dispatch_log2phy,
                "dispatch_num_experts": dispatch_num_experts,
                "dispatch_signature": topology_signature,
                "active_rank_count": active_rank_count,
                "owned_per_rank": owned_per_rank,
            }
            self._dispatch_remap_cache[cache_key] = cached
        return (
            cached["dispatch_log2phy"],
            int(cached["dispatch_num_experts"]),
            cached["dispatch_signature"],
            int(cached["active_rank_count"]),
            int(cached["owned_per_rank"]),
        )

    @staticmethod
    def _copy_row(dst: torch.Tensor, src: torch.Tensor) -> None:
        # Pinned CPU -> NPU row copies may dispatch through an internal
        # copy_stream on Ascend. Recording the slot ready-event immediately
        # after issuing those async copies allowed the next layer to observe a
        # "ready" slot before the host->device rows had actually drained,
        # which in turn caused copy_stream/AICORE faults in mode3 prefetch.
        #
        # Keep NPU->NPU copies asynchronous, but make CPU-sourced copies
        # blocking so the slot contents are fully materialized before the
        # prefetch ready-event is recorded.
        if src.device.type == "cpu" and dst.device.type == "npu":
            stable_src = _stable_cpu_transfer_source(src)
            staged = stable_src.to(device=dst.device, non_blocking=False)
            dst.copy_(staged, non_blocking=False)
            return
        non_blocking = src.device.type != "cpu"
        dst.copy_(src, non_blocking=non_blocking)

    @staticmethod
    def _copy_row_sync(dst: torch.Tensor, src: torch.Tensor) -> None:
        # This matches the proven mode2 swap path: materialize CPU rows on the
        # destination device first, then copy into the formatted expert slot
        # with an explicit blocking copy. Use it for layer-entry binding where
        # correctness is more important than overlap.
        if src.device.type == "cpu" and dst.device.type == "npu":
            src = _stable_cpu_transfer_source(src).to(device=dst.device,
                                                      non_blocking=False)
        dst.copy_(src, non_blocking=False)

    @staticmethod
    def _copy_cpu_row_to_stage_async(dst: torch.Tensor,
                                     src: torch.Tensor) -> None:
        if src.device.type != "cpu" or dst.device.type != "npu":
            dst.copy_(src, non_blocking=True)
            return
        src = src.detach()
        if not src.is_contiguous():
            # Do not launch an async H2D copy from a short-lived contiguous
            # temporary. Fall back to the conservative synchronous path for
            # unusual CPU shadow layouts.
            staged = _stable_cpu_transfer_source(src).to(device=dst.device,
                                                          non_blocking=False)
            dst.copy_(staged, non_blocking=False)
            return
        dst.copy_(src, non_blocking=True)

    @staticmethod
    def _copy_cpu_row_to_runtime_slot_async(dst: torch.Tensor,
                                            src: torch.Tensor) -> None:
        if src.device.type != "cpu" or dst.device.type != "npu":
            dst.copy_(src, non_blocking=True)
            return
        src = src.detach()
        if not src.is_contiguous():
            # Avoid launching async copies from short-lived contiguous
            # temporaries. This keeps the direct-slot experiment scoped to the
            # common contiguous CPU shadow layout.
            src = _stable_cpu_transfer_source(src)
            dst.copy_(src, non_blocking=False)
            return
        dst.copy_(src, non_blocking=True)

    @staticmethod
    def _assignment_runs(
            assignments: list[tuple[int, int]]) -> list[tuple[int, int, int]]:
        if not assignments:
            return []
        runs: list[tuple[int, int, int]] = []
        dst_start, src_start = assignments[0]
        prev_dst, prev_src = dst_start, src_start
        length = 1
        for dst_idx, src_idx in assignments[1:]:
            if dst_idx == prev_dst + 1 and src_idx == prev_src + 1:
                length += 1
            else:
                runs.append((int(dst_start), int(src_start), int(length)))
                dst_start, src_start = dst_idx, src_idx
                length = 1
            prev_dst, prev_src = dst_idx, src_idx
        runs.append((int(dst_start), int(src_start), int(length)))
        return runs

    def _assignment_submit_count(self, assignments: list[tuple[int, int]],
                                 *, bulk_enabled: bool) -> int:
        if not assignments:
            return 0
        if not bulk_enabled:
            return len(assignments)
        return len(self._assignment_runs(assignments))

    def _maybe_log_transfer_plan(
            self, layer: Any, *, reason: str, async_copy: bool,
            slot_id: int, npu_assignments: list[tuple[int, int]],
            cpu_assignments: list[tuple[int, int]], cpu_path: str,
            use_layer_local_buffer: bool,
            cpu_prefetch_stream: Optional[torch.npu.Stream],
            cpu_w13: Optional[torch.Tensor],
            cpu_w2: Optional[torch.Tensor]) -> None:
        if not self.enable_transfer_plan_logs:
            return
        layer_idx = int(getattr(layer, "layer_idx", -1))
        if self.transfer_plan_first_n <= 0:
            return
        if layer_idx >= self.transfer_plan_first_n:
            return
        log_key = (layer_idx, int(async_copy), str(reason))
        if log_key in self._logged_transfer_plans:
            return
        stage_assignments = [
            (stage_idx, int(cpu_slot))
            for stage_idx, (_slot_idx, cpu_slot) in enumerate(cpu_assignments)
        ]
        stage_to_slot = [
            (int(slot_idx), stage_idx)
            for stage_idx, (slot_idx, _cpu_slot) in enumerate(cpu_assignments)
        ]
        npu_path = "layer_local" if use_layer_local_buffer else (
            "bulk" if self.enable_bulk_npu_copy else "per_row")
        npu_submit_count = 0 if use_layer_local_buffer else \
            self._assignment_submit_count(
                npu_assignments, bulk_enabled=self.enable_bulk_npu_copy)
        cpu_direct_submits = -1
        cpu_stage_submits = -1
        cpu_pack_submits = -1
        if cpu_path == "direct_async":
            cpu_direct_submits = self._assignment_submit_count(
                cpu_assignments, bulk_enabled=self.enable_bulk_cpu_direct)
        elif cpu_path in ("stage_async_pack", "stage_async_wait_pack",
                          "sync_stage"):
            cpu_stage_submits = self._assignment_submit_count(
                stage_assignments, bulk_enabled=self.enable_bulk_cpu_stage)
            if cpu_path in ("stage_async_pack", "sync_stage"):
                cpu_pack_submits = self._assignment_submit_count(
                    stage_to_slot, bulk_enabled=self.enable_bulk_npu_copy)
        elif cpu_path == "defer_sync_fill":
            cpu_stage_submits = self._assignment_submit_count(
                stage_assignments, bulk_enabled=self.enable_bulk_cpu_stage)
            cpu_pack_submits = self._assignment_submit_count(
                stage_to_slot, bulk_enabled=self.enable_bulk_npu_copy)

        logger.info(
            "Mode3 transfer plan: layer=%s slot=%s reason=%s async=%s "
            "cpu_stream=%s valid_experts=%s npu_rows=%s npu_path=%s "
            "npu_submit_count=%s cpu_rows=%s cpu_path=%s "
            "cpu_direct_submit_count=%s cpu_stage_submit_count=%s "
            "cpu_pack_submit_count=%s bulk_npu=%s bulk_cpu_stage=%s "
            "bulk_cpu_direct=%s async_cpu_stage=%s async_cpu_pack=%s "
            "direct_cpu_slot=%s device_ready_wait=%s layer_local=%s "
            "cpu_w13_contig=%s cpu_w2_contig=%s "
            "cpu_w13_pinned=%s cpu_w2_pinned=%s npu_head=%s cpu_head=%s",
            layer_idx,
            slot_id,
            reason,
            int(async_copy),
            int(cpu_prefetch_stream is not None),
            len(npu_assignments) + len(cpu_assignments),
            len(npu_assignments),
            npu_path,
            npu_submit_count,
            len(cpu_assignments),
            cpu_path,
            cpu_direct_submits,
            cpu_stage_submits,
            cpu_pack_submits,
            int(self.enable_bulk_npu_copy),
            int(self.enable_bulk_cpu_stage),
            int(self.enable_bulk_cpu_direct),
            int(self.enable_async_cpu_stage),
            int(self.enable_async_cpu_pack),
            int(self.enable_direct_cpu_slot),
            int(self.enable_device_ready_wait),
            int(use_layer_local_buffer),
            _tensor_is_contiguous(cpu_w13),
            _tensor_is_contiguous(cpu_w2),
            _tensor_is_pinned(cpu_w13),
            _tensor_is_pinned(cpu_w2),
            npu_assignments[:4],
            cpu_assignments[:4],
        )
        self._logged_transfer_plans.add(log_key)

    def _copy_npu_assignment_runs(self, dst: torch.Tensor, src: torch.Tensor,
                                  assignments: list[tuple[int, int]],
                                  *, async_copy: bool) -> None:
        if not assignments:
            return
        copy_row = self._copy_row if async_copy else self._copy_row_sync
        if not self.enable_bulk_npu_copy:
            for dst_idx, src_idx in assignments:
                copy_row(dst[int(dst_idx)], src[int(src_idx)])
            return
        for dst_start, src_start, length in self._assignment_runs(assignments):
            if length <= 1:
                copy_row(dst[dst_start], src[src_start])
                continue
            non_blocking = async_copy and src.device.type != "cpu"
            dst[dst_start:dst_start + length].copy_(
                src[src_start:src_start + length],
                non_blocking=non_blocking)

    def _maybe_log_copy_format_diag(self, layer: Any,
                                    slot: _Mode3DoubleBufferSlot,
                                    assignments: list[tuple[int, int]], *,
                                    reason: str) -> None:
        if not self.enable_copy_format_diag:
            return
        if self._copy_format_diag_count >= self.copy_format_diag_first_n:
            return
        runs = self._assignment_runs(assignments)
        first_run = runs[0] if runs else None

        def _view_meta(weight: Optional[torch.Tensor],
                       run_index: int) -> dict[str, Any]:
            if weight is None or first_run is None:
                return _lossless_weight_meta(None)
            dst_start, src_start, length = first_run
            start = dst_start if run_index == 0 else src_start
            return _lossless_weight_meta(weight[start:start + length])

        logger.info(
            "Mode3 copy format diag: rank=%s layer=%s reason=%s "
            "assignments=%s runs=%s first_run=%s slot_w13=%s slot_w2=%s "
            "layer_w13=%s layer_w2=%s dst_w13_view=%s src_w13_view=%s "
            "dst_w2_view=%s src_w2_view=%s",
            getattr(layer, "rank", _profile_rank()),
            getattr(layer, "layer_idx", -1),
            reason,
            len(assignments),
            len(runs),
            first_run,
            _lossless_weight_meta(slot.w13),
            _lossless_weight_meta(slot.w2),
            _lossless_weight_meta(getattr(layer, "w13_weight", None)),
            _lossless_weight_meta(getattr(layer, "w2_weight", None)),
            _view_meta(slot.w13, 0),
            _view_meta(getattr(layer, "w13_weight", None), 1),
            _view_meta(slot.w2, 0),
            _view_meta(getattr(layer, "w2_weight", None), 1),
        )
        self._copy_format_diag_count += 1

    def _copy_cpu_assignments_to_stage(
            self, slot: _Mode3DoubleBufferSlot,
            cpu_assignments: list[tuple[int, int]], cpu_w13: torch.Tensor,
            cpu_w2: torch.Tensor, *, async_copy: bool) -> None:
        if not cpu_assignments:
            return
        if not self.enable_bulk_cpu_stage:
            for stage_idx, (_slot_idx, cpu_slot) in enumerate(cpu_assignments):
                copy_fn = (self._copy_cpu_row_to_stage_async
                           if async_copy else self._copy_row_sync)
                copy_fn(slot.cpu_stage_w13[stage_idx], cpu_w13[int(cpu_slot)])
                copy_fn(slot.cpu_stage_w2[stage_idx], cpu_w2[int(cpu_slot)])
            return
        stage_assignments = [
            (stage_idx, int(cpu_slot))
            for stage_idx, (_slot_idx, cpu_slot) in enumerate(cpu_assignments)
        ]
        copy_row = (self._copy_cpu_row_to_stage_async
                    if async_copy else self._copy_row_sync)
        for stage_start, cpu_start, length in self._assignment_runs(
                stage_assignments):
            if length <= 1:
                copy_row(slot.cpu_stage_w13[stage_start], cpu_w13[cpu_start])
                copy_row(slot.cpu_stage_w2[stage_start], cpu_w2[cpu_start])
                continue
            src_w13 = cpu_w13[cpu_start:cpu_start + length].detach()
            src_w2 = cpu_w2[cpu_start:cpu_start + length].detach()
            if not src_w13.is_contiguous() or not src_w2.is_contiguous():
                for offset in range(length):
                    copy_row(slot.cpu_stage_w13[stage_start + offset],
                             cpu_w13[cpu_start + offset])
                    copy_row(slot.cpu_stage_w2[stage_start + offset],
                             cpu_w2[cpu_start + offset])
                continue
            slot.cpu_stage_w13[stage_start:stage_start + length].copy_(
                src_w13, non_blocking=async_copy)
            slot.cpu_stage_w2[stage_start:stage_start + length].copy_(
                src_w2, non_blocking=async_copy)

    def _copy_cpu_assignments_to_runtime_direct(
            self, slot: _Mode3DoubleBufferSlot,
            cpu_assignments: list[tuple[int, int]], cpu_w13: torch.Tensor,
            cpu_w2: torch.Tensor) -> None:
        if not cpu_assignments:
            return
        if not self.enable_bulk_cpu_direct:
            for slot_idx, cpu_slot in cpu_assignments:
                self._copy_cpu_row_to_runtime_slot_async(
                    slot.w13[int(slot_idx)], cpu_w13[int(cpu_slot)])
                self._copy_cpu_row_to_runtime_slot_async(
                    slot.w2[int(slot_idx)], cpu_w2[int(cpu_slot)])
            return
        for slot_start, cpu_start, length in self._assignment_runs(
                cpu_assignments):
            if length <= 1:
                self._copy_cpu_row_to_runtime_slot_async(
                    slot.w13[slot_start], cpu_w13[cpu_start])
                self._copy_cpu_row_to_runtime_slot_async(
                    slot.w2[slot_start], cpu_w2[cpu_start])
                continue
            src_w13 = cpu_w13[cpu_start:cpu_start + length].detach()
            src_w2 = cpu_w2[cpu_start:cpu_start + length].detach()
            if not src_w13.is_contiguous() or not src_w2.is_contiguous():
                for offset in range(length):
                    self._copy_cpu_row_to_runtime_slot_async(
                        slot.w13[slot_start + offset],
                        cpu_w13[cpu_start + offset])
                    self._copy_cpu_row_to_runtime_slot_async(
                        slot.w2[slot_start + offset],
                        cpu_w2[cpu_start + offset])
                continue
            slot.w13[slot_start:slot_start + length].copy_(
                src_w13, non_blocking=True)
            slot.w2[slot_start:slot_start + length].copy_(
                src_w2, non_blocking=True)

    @staticmethod
    def _copy_rows(dst: torch.Tensor, src: torch.Tensor) -> None:
        if src.device.type == "cpu" and dst.device.type == "npu":
            for row_idx in range(int(dst.shape[0])):
                stable_row = _stable_cpu_transfer_source(src[row_idx])
                staged_row = stable_row.to(device=dst.device,
                                           non_blocking=False)
                dst[row_idx].copy_(staged_row, non_blocking=False)
            return
        non_blocking = src.device.type != "cpu"
        dst.copy_(src, non_blocking=non_blocking)

    def _ensure_slot_capacity(self, slot: _Mode3DoubleBufferSlot,
                              layer: Any) -> None:
        target_shape_w13 = (128, ) + tuple(layer.w13_weight.shape[1:])
        target_shape_w2 = (128, ) + tuple(layer.w2_weight.shape[1:])
        target_device = layer.w13_weight.device
        target_dtype_w13 = layer.w13_weight.dtype
        target_dtype_w2 = layer.w2_weight.dtype
        if self.runtime_buffer_format == "plain":
            target_format_w13 = "plain"
            target_format_w2 = "plain"
        else:
            target_format_w13 = _lossless_weight_format(layer.w13_weight)
            target_format_w2 = _lossless_weight_format(layer.w2_weight)
        current_format_w13 = (
            "plain" if self.runtime_buffer_format == "plain"
            else _lossless_weight_format(slot.w13))
        current_format_w2 = (
            "plain" if self.runtime_buffer_format == "plain"
            else _lossless_weight_format(slot.w2))
        if (slot.w13 is not None and slot.w2 is not None
                and slot.w13.shape == target_shape_w13
                and slot.w2.shape == target_shape_w2
                and slot.w13.device == target_device
                and slot.w2.device == target_device
                and slot.w13.dtype == target_dtype_w13
                and slot.w2.dtype == target_dtype_w2
                and current_format_w13 == target_format_w13
                and current_format_w2 == target_format_w2):
            return
        if self.runtime_buffer_format == "plain":
            slot.w13 = _allocate_plain_buffer_like(layer.w13_weight, 128)
            slot.w2 = _allocate_plain_buffer_like(layer.w2_weight, 128)
        else:
            slot.w13 = _allocate_formatted_buffer_like(layer.w13_weight, 128)
            slot.w2 = _allocate_formatted_buffer_like(layer.w2_weight, 128)
        if (not self._logged_runtime_buffer_format
                and self.enable_copy_format_diag):
            logger.info(
                "Mode3 runtime buffer format: requested=%s w13=%s w2=%s "
                "layer_w13=%s layer_w2=%s",
                self.runtime_buffer_format,
                _lossless_weight_format(slot.w13),
                _lossless_weight_format(slot.w2),
                _lossless_weight_format(layer.w13_weight),
                _lossless_weight_format(layer.w2_weight),
            )
            self._logged_runtime_buffer_format = True

    def _ensure_cpu_stage_capacity(self, slot: _Mode3DoubleBufferSlot,
                                   layer: Any, row_count: int) -> None:
        row_count = max(int(row_count), 1)
        target_shape_w13 = (row_count, ) + tuple(layer.w13_weight.shape[1:])
        target_shape_w2 = (row_count, ) + tuple(layer.w2_weight.shape[1:])
        target_device = layer.w13_weight.device
        target_dtype_w13 = layer.w13_weight.dtype
        target_dtype_w2 = layer.w2_weight.dtype
        if (slot.cpu_stage_w13 is not None and slot.cpu_stage_w2 is not None
                and slot.cpu_stage_w13.shape == target_shape_w13
                and slot.cpu_stage_w2.shape == target_shape_w2
                and slot.cpu_stage_w13.device == target_device
                and slot.cpu_stage_w2.device == target_device
                and slot.cpu_stage_w13.dtype == target_dtype_w13
                and slot.cpu_stage_w2.dtype == target_dtype_w2):
            return
        # This is only a transport staging area for async CPU->NPU copies.
        # Keep it in a plain contiguous NPU layout instead of FRACTAL_NZ so the
        # host->device DMA never targets the final formatted expert buffer
        # directly. The final pack into slot.w13/slot.w2 still happens on the
        # prefetch stream via NPU->NPU copies.
        slot.cpu_stage_w13 = torch.empty(target_shape_w13,
                                         device=target_device,
                                         dtype=target_dtype_w13)
        slot.cpu_stage_w2 = torch.empty(target_shape_w2,
                                        device=target_device,
                                        dtype=target_dtype_w2)

    def _copy_cpu_assignments_via_stage(
            self, slot: _Mode3DoubleBufferSlot, layer: Any,
            cpu_assignments: list[tuple[int, int]], cpu_w13: torch.Tensor,
            cpu_w2: torch.Tensor) -> None:
        if not cpu_assignments:
            return
        self._ensure_cpu_stage_capacity(slot, layer, len(cpu_assignments))
        if slot.cpu_stage_w13 is None or slot.cpu_stage_w2 is None:
            raise RuntimeError(
                "Mode3 CPU staging buffers were not allocated: "
                f"layer={getattr(layer, 'layer_idx', -1)}")
        # First materialize CPU rows into a plain NPU staging buffer. The final
        # runtime slot may be FRACTAL_NZ, so keep that formatted copy as an
        # explicit NPU->NPU step instead of targeting it from host.
        self._copy_cpu_assignments_to_stage(slot,
                                            cpu_assignments,
                                            cpu_w13,
                                            cpu_w2,
                                            async_copy=False)
        stage_to_slot = [
            (int(slot_idx), stage_idx)
            for stage_idx, (slot_idx, _cpu_slot) in enumerate(cpu_assignments)
        ]
        self._copy_npu_assignment_runs(slot.w13,
                                       slot.cpu_stage_w13,
                                       stage_to_slot,
                                       async_copy=False)
        self._copy_npu_assignment_runs(slot.w2,
                                       slot.cpu_stage_w2,
                                       stage_to_slot,
                                       async_copy=False)

    def _can_use_layer_local_buffer(
            self, layer: Any, valid_expert_count: int,
            npu_assignments: list[tuple[int, int]]) -> bool:
        if not self.enable_layer_local_buffer:
            return False
        if valid_expert_count <= 0:
            return False
        if (layer.w13_weight is None or layer.w2_weight is None
                or int(layer.w13_weight.shape[0]) < valid_expert_count
                or int(layer.w2_weight.shape[0]) < valid_expert_count):
            return False
        # The fixed resident experts are already materialized in the layer's
        # prefix slots. Reuse them only when the desired dense runtime slot is
        # exactly the resident source slot; otherwise fall back to the safe
        # runtime double-buffer copy path.
        return all(int(dst_idx) == int(src_idx)
                   for dst_idx, src_idx in npu_assignments)

    def _schedule_cpu_assignments_to_stage_async(
            self, slot: _Mode3DoubleBufferSlot, layer: Any,
            cpu_assignments: list[tuple[int, int]], cpu_w13: torch.Tensor,
            cpu_w2: torch.Tensor, cpu_prefetch_stream: torch.npu.Stream,
            *, pack_to_runtime: bool,
            timing_events: Optional[dict[str, Any]] = None,
            host_timing: Optional[dict[str, float]] = None) -> None:
        if not cpu_assignments:
            return
        host_start = time.perf_counter() if host_timing is not None else 0.0
        self._ensure_cpu_stage_capacity(slot, layer, len(cpu_assignments))
        if slot.cpu_stage_w13 is None or slot.cpu_stage_w2 is None:
            raise RuntimeError(
                "Mode3 async CPU staging buffers were not allocated: "
                f"layer={getattr(layer, 'layer_idx', -1)}")
        with torch.npu.stream(cpu_prefetch_stream):
            _event_record((timing_events or {}).get("cpu_start_event"))
            self._copy_cpu_assignments_to_stage(slot,
                                                cpu_assignments,
                                                cpu_w13,
                                                cpu_w2,
                                                async_copy=True)
            _event_record((timing_events or {}).get("cpu_end_event"))
            if pack_to_runtime:
                stage_to_slot = [
                    (int(slot_idx), stage_idx)
                    for stage_idx, (slot_idx, _cpu_slot) in enumerate(
                        cpu_assignments)
                ]
                _event_record((timing_events or {}).get("cpu_pack_start_event"))
                self._copy_npu_assignment_runs(slot.w13,
                                               slot.cpu_stage_w13,
                                               stage_to_slot,
                                               async_copy=True)
                self._copy_npu_assignment_runs(slot.w2,
                                               slot.cpu_stage_w2,
                                               stage_to_slot,
                                               async_copy=True)
                _event_record((timing_events or {}).get("cpu_pack_end_event"))
                slot.cpu_pack_event.record()
            else:
                slot.cpu_ready_event.record()
        if host_timing is not None:
            host_timing["submit_cpu_stage_async_us"] = (
                time.perf_counter() - host_start) * 1e6
        slot.has_async_cpu_copy = not pack_to_runtime
        slot.has_async_cpu_pack = pack_to_runtime
        slot.needs_sync_cpu_fill = False
        log_key = (int(layer.layer_idx), len(cpu_assignments))
        if self.enable_transfer_logs and log_key not in self._logged_async_cpu_stage:
            logger.info(
                "Mode3 async CPU stage scheduled: layer=%s cpu_rows=%s staging=plain_npu pack_to_runtime=%s",
                layer.layer_idx,
                len(cpu_assignments),
                pack_to_runtime,
            )
            self._logged_async_cpu_stage.add(log_key)

    def _schedule_cpu_assignments_to_slot_async(
            self, slot: _Mode3DoubleBufferSlot, layer: Any,
            cpu_assignments: list[tuple[int, int]], cpu_w13: torch.Tensor,
            cpu_w2: torch.Tensor, cpu_prefetch_stream: torch.npu.Stream,
            timing_events: Optional[dict[str, Any]] = None,
            host_timing: Optional[dict[str, float]] = None) -> None:
        if not cpu_assignments:
            return
        host_start = time.perf_counter() if host_timing is not None else 0.0
        with torch.npu.stream(cpu_prefetch_stream):
            _event_record((timing_events or {}).get("cpu_start_event"))
            self._copy_cpu_assignments_to_runtime_direct(
                slot, cpu_assignments, cpu_w13, cpu_w2)
            _event_record((timing_events or {}).get("cpu_end_event"))
            slot.cpu_pack_event.record()
        if host_timing is not None:
            host_timing["submit_cpu_direct_async_us"] = (
                time.perf_counter() - host_start) * 1e6
        slot.has_async_cpu_copy = False
        slot.has_async_cpu_pack = False
        slot.has_async_cpu_direct = True
        slot.needs_sync_cpu_fill = False
        log_key = (int(layer.layer_idx), len(cpu_assignments))
        if self.enable_transfer_logs and log_key not in self._logged_async_cpu_stage:
            logger.info(
                "Mode3 async CPU direct slot scheduled: layer=%s cpu_rows=%s target=runtime_slot",
                layer.layer_idx,
                len(cpu_assignments),
            )
            self._logged_async_cpu_stage.add(log_key)

    def _fill_pending_cpu_rows(self, slot: _Mode3DoubleBufferSlot,
                               layer: Any) -> None:
        if not slot.needs_sync_cpu_fill or slot.source_from_cpu <= 0:
            slot.needs_sync_cpu_fill = False
            return
        cpu_w13 = _maybe_pin_cpu_tensor(getattr(layer, "lossless_cpu_w13_weight",
                                                None))
        cpu_w2 = _maybe_pin_cpu_tensor(getattr(layer, "lossless_cpu_w2_weight",
                                               None))
        if cpu_w13 is None or cpu_w2 is None:
            raise RuntimeError(
                "Mode3 sync CPU fill requested but CPU shadow weights are missing: "
                f"layer={layer.layer_idx}.")
        cpu_shadow_slots = getattr(layer, "lossless_cpu_shadow_local_slots", {})
        if (slot.cpu_stage_count > 0
                and len(slot.cpu_stage_slot_ids) >= slot.cpu_stage_count
                and len(slot.cpu_stage_cpu_slots) >= slot.cpu_stage_count):
            cpu_assignments = list(
                zip(slot.cpu_stage_slot_ids[:slot.cpu_stage_count],
                    slot.cpu_stage_cpu_slots[:slot.cpu_stage_count]))
        else:
            cpu_assignments = []
            start_idx = int(slot.source_from_npu)
            end_idx = int(slot.valid_expert_count)
            for slot_idx in range(start_idx, end_idx):
                expert_id = int(slot.expert_ids[slot_idx])
                cpu_slot = cpu_shadow_slots.get(expert_id)
                if cpu_slot is None:
                    raise RuntimeError(
                        "Mode3 sync CPU fill missing CPU shadow slot mapping: "
                        f"layer={layer.layer_idx} expert_id={expert_id}")
                cpu_assignments.append((slot_idx, int(cpu_slot)))
        self._copy_cpu_assignments_via_stage(slot, layer, cpu_assignments,
                                             cpu_w13, cpu_w2)
        slot.needs_sync_cpu_fill = False

    def _populate_slot(self,
                       slot: _Mode3DoubleBufferSlot,
                       layer: Any,
                       *,
                       async_copy: bool,
                       cpu_prefetch_stream: Optional[torch.npu.Stream] = None,
                       reason: str = "",
                       timing_events: Optional[dict[str, Any]] = None,
                       host_timing: Optional[dict[str, float]] = None) -> None:
        populate_host_start = (
            time.perf_counter() if host_timing is not None else 0.0)
        phase_start = time.perf_counter() if host_timing is not None else 0.0
        slot_expert_ids = self._ordered_mode3_slot_expert_ids(layer)
        valid_expert_count = len(slot_expert_ids)
        if valid_expert_count <= 0:
            raise RuntimeError(
                f"Mode3 slot population requires owned experts at layer={layer.layer_idx}.")
        if host_timing is not None:
            host_timing["submit_order_us"] = (
                time.perf_counter() - phase_start) * 1e6
            phase_start = time.perf_counter()
        primary_slots = getattr(layer, "lossless_mode3_primary_prefix_local_slots",
                                {})
        cpu_shadow_slots = getattr(layer, "lossless_cpu_shadow_local_slots", {})
        cpu_w13 = getattr(layer, "lossless_cpu_w13_weight", None)
        cpu_w2 = getattr(layer, "lossless_cpu_w2_weight", None)
        if cpu_w13 is None or cpu_w2 is None:
            raise RuntimeError(
                f"Mode3 slot population requires CPU shadow weights at layer={layer.layer_idx}.")
        if host_timing is not None:
            host_timing["cpu_w13_pinned"] = float(_tensor_is_pinned(cpu_w13))
            host_timing["cpu_w2_pinned"] = float(_tensor_is_pinned(cpu_w2))
            host_timing["cpu_w13_contig"] = float(
                _tensor_is_contiguous(cpu_w13))
            host_timing["cpu_w2_contig"] = float(
                _tensor_is_contiguous(cpu_w2))
        source_from_npu = 0
        source_from_cpu = 0
        npu_assignments: list[tuple[int, int]] = []
        cpu_assignments: list[tuple[int, int]] = []
        for slot_idx, expert_id in enumerate(slot_expert_ids):
            source_local_slot = primary_slots.get(int(expert_id))
            if source_local_slot is not None:
                npu_assignments.append((slot_idx, int(source_local_slot)))
                source_from_npu += 1
            else:
                cpu_slot = cpu_shadow_slots.get(int(expert_id))
                if cpu_slot is None:
                    raise RuntimeError(
                        "Mode3 slot population missing CPU shadow slot mapping: "
                        f"layer={layer.layer_idx} expert_id={int(expert_id)}")
                cpu_assignments.append((slot_idx, int(cpu_slot)))
                source_from_cpu += 1
        if host_timing is not None:
            host_timing["submit_assign_us"] = (
                time.perf_counter() - phase_start) * 1e6
            phase_start = time.perf_counter()
        use_layer_local_buffer = self._can_use_layer_local_buffer(
            layer, valid_expert_count, npu_assignments)
        if host_timing is not None:
            host_timing["submit_layer_local_check_us"] = (
                time.perf_counter() - phase_start) * 1e6
            phase_start = time.perf_counter()
        if use_layer_local_buffer:
            slot.w13 = layer.w13_weight
            slot.w2 = layer.w2_weight
        else:
            if slot.uses_layer_local_buffer:
                slot.w13 = None
                slot.w2 = None
            self._ensure_slot_capacity(slot, layer)
            self._maybe_log_copy_format_diag(layer,
                                             slot,
                                             npu_assignments,
                                             reason=reason)
            _event_record((timing_events or {}).get("npu_start_event"))
            _event_record((timing_events or {}).get("npu_w13_start_event"))
            self._copy_npu_assignment_runs(slot.w13,
                                           layer.w13_weight,
                                           npu_assignments,
                                           async_copy=async_copy)
            _event_record((timing_events or {}).get("npu_w13_end_event"))
            _event_record((timing_events or {}).get("npu_w2_start_event"))
            self._copy_npu_assignment_runs(slot.w2,
                                           layer.w2_weight,
                                           npu_assignments,
                                           async_copy=async_copy)
            _event_record((timing_events or {}).get("npu_w2_end_event"))
            _event_record((timing_events or {}).get("npu_end_event"))
        if host_timing is not None:
            host_timing["submit_npu_us"] = (
                time.perf_counter() - phase_start) * 1e6
            phase_start = time.perf_counter()
        slot.has_async_cpu_copy = False
        slot.has_async_cpu_pack = False
        slot.has_async_cpu_direct = False
        slot.needs_sync_cpu_fill = False
        slot.uses_layer_local_buffer = bool(use_layer_local_buffer)
        slot.cpu_stage_slot_ids = tuple(slot_idx for slot_idx, _ in cpu_assignments)
        slot.cpu_stage_cpu_slots = tuple(cpu_slot for _, cpu_slot in cpu_assignments)
        slot.cpu_stage_count = len(cpu_assignments)
        cpu_path = "none"
        if cpu_assignments:
            if (async_copy and cpu_prefetch_stream is not None
                    and self.enable_direct_cpu_slot):
                # Direct-slot experiment: write CPU shadow rows directly into
                # their final runtime expert slots on the CPU prefetch stream,
                # bypassing the plain NPU staging buffer.
                cpu_path = "direct_async"
                self._schedule_cpu_assignments_to_slot_async(
                    slot, layer, cpu_assignments, cpu_w13, cpu_w2,
                    cpu_prefetch_stream, timing_events=timing_events,
                    host_timing=host_timing)
            elif (async_copy and cpu_prefetch_stream is not None
                    and self.enable_async_cpu_stage):
                # CPU shadow rows first land in a plain NPU staging buffer on
                # a separate stream. When async CPU pack is enabled, that same
                # stream also copies staged rows into their final runtime
                # slots; otherwise the main prefetch stream does that after
                # waiting on the CPU staging event.
                cpu_path = ("stage_async_pack" if self.enable_async_cpu_pack
                            else "stage_async_wait_pack")
                self._schedule_cpu_assignments_to_stage_async(
                    slot, layer, cpu_assignments, cpu_w13, cpu_w2,
                    cpu_prefetch_stream,
                    pack_to_runtime=self.enable_async_cpu_pack,
                    timing_events=timing_events,
                    host_timing=host_timing)
            elif async_copy and cpu_prefetch_stream is not None:
                # Keep resident NPU experts prefetched, but fill CPU shadow rows
                # at bind-time through the plain NPU staging buffer when async
                # CPU staging is disabled.
                cpu_path = "defer_sync_fill"
                self._ensure_cpu_stage_capacity(slot, layer, len(cpu_assignments))
                slot.needs_sync_cpu_fill = True
                log_key = (int(layer.layer_idx), len(cpu_assignments))
                if (self.enable_transfer_logs
                        and log_key not in self._logged_cpu_fill_deferrals):
                    logger.info(
                        "Mode3 CPU shadow fill deferred to bind: layer=%s cpu_rows=%s reason=async_cpu_stage_disabled staging=plain_npu",
                        layer.layer_idx,
                        len(cpu_assignments),
                    )
                    self._logged_cpu_fill_deferrals.add(log_key)
            else:
                cpu_path = "sync_stage"
                self._copy_cpu_assignments_via_stage(slot, layer,
                                                     cpu_assignments, cpu_w13,
                                                     cpu_w2)
        if host_timing is not None:
            host_timing["submit_cpu_us"] = (
                time.perf_counter() - phase_start) * 1e6
            phase_start = time.perf_counter()
        self._maybe_log_transfer_plan(
            layer,
            reason=reason,
            async_copy=async_copy,
            slot_id=self._slot_id_for_layer(int(layer.layer_idx)),
            npu_assignments=npu_assignments,
            cpu_assignments=cpu_assignments,
            cpu_path=cpu_path,
            use_layer_local_buffer=use_layer_local_buffer,
            cpu_prefetch_stream=cpu_prefetch_stream,
            cpu_w13=cpu_w13,
            cpu_w2=cpu_w2,
        )
        if host_timing is not None:
            host_timing["submit_plan_log_us"] = (
                time.perf_counter() - phase_start) * 1e6
            phase_start = time.perf_counter()
        slot.prefetch_cpu_path = cpu_path
        slot.expert_map, expert_map_cache_hit = \
            self._get_cached_slot_expert_map(layer, slot_expert_ids,
                                             slot.w13.device)
        if host_timing is not None:
            host_timing["submit_expert_map_us"] = (
                time.perf_counter() - phase_start) * 1e6
            host_timing["expert_map_cache_hit"] = float(
                int(expert_map_cache_hit))
            phase_start = time.perf_counter()
        dispatch_log2phy, dispatch_num_experts, dispatch_signature, \
            dispatch_active_rank_count, dispatch_owned_per_rank = \
            self._get_cached_dispatch_remap(layer, slot.w13.device)
        if host_timing is not None:
            host_timing["submit_dispatch_cache_us"] = (
                time.perf_counter() - phase_start) * 1e6
            phase_start = time.perf_counter()
        slot.layer_idx = int(layer.layer_idx)
        slot.expert_ids = tuple(slot_expert_ids)
        slot.valid_expert_count = valid_expert_count
        slot.source_from_npu = source_from_npu
        slot.source_from_cpu = source_from_cpu
        slot.dispatch_log2phy = dispatch_log2phy
        slot.dispatch_num_experts = dispatch_num_experts
        slot.dispatch_signature = dispatch_signature
        slot.dispatch_active_rank_count = dispatch_active_rank_count
        slot.dispatch_owned_per_rank = dispatch_owned_per_rank
        if host_timing is not None:
            host_timing["submit_slot_state_us"] = (
                time.perf_counter() - phase_start) * 1e6
            host_timing["submit_populate_total_us"] = (
                time.perf_counter() - populate_host_start) * 1e6

    def _slot_matches(self, slot: _Mode3DoubleBufferSlot, layer: Any) -> bool:
        ordered_slot_expert_ids = tuple(self._ordered_mode3_slot_expert_ids(layer))
        dispatch_signature = _hybrid_dispatch_topology_signature(layer)
        return (
            slot.layer_idx == int(layer.layer_idx)
            and slot.expert_ids == ordered_slot_expert_ids
            and slot.valid_expert_count == len(ordered_slot_expert_ids)
            and slot.expert_map is not None
            and slot.w13 is not None
            and slot.w2 is not None
            and slot.dispatch_signature == dispatch_signature
        )

    def prepare_slot(self,
                     layer: Any,
                     slot_id: int,
                     async_copy: bool,
                     *,
                     reason: str) -> _Mode3DoubleBufferSlot:
        slot = self.slots[slot_id]
        if self._slot_matches(slot, layer) and not slot.inflight_prefetch:
            return slot
        submit_start = time.perf_counter()
        host_timing: Optional[dict[str, float]] = (
            {} if self.enable_timing_logs else None)
        event_alloc_start = time.perf_counter()
        start_event = self.new_timing_event() if async_copy else None
        end_event = self.new_timing_event() if async_copy else None
        timing_events = {
            "npu_start_event": self.new_timing_event() if async_copy else None,
            "npu_end_event": self.new_timing_event() if async_copy else None,
            "npu_w13_start_event": self.new_timing_event() if async_copy else None,
            "npu_w13_end_event": self.new_timing_event() if async_copy else None,
            "npu_w2_start_event": self.new_timing_event() if async_copy else None,
            "npu_w2_end_event": self.new_timing_event() if async_copy else None,
            "cpu_start_event": self.new_timing_event() if async_copy else None,
            "cpu_end_event": self.new_timing_event() if async_copy else None,
            "cpu_pack_start_event": self.new_timing_event() if async_copy else None,
            "cpu_pack_end_event": self.new_timing_event() if async_copy else None,
        }
        if host_timing is not None:
            host_timing["submit_event_alloc_us"] = (
                time.perf_counter() - event_alloc_start) * 1e6
        if async_copy and self.prefetch_stream is not None:
            stream_wait_start = time.perf_counter()
            current_stream = torch.npu.current_stream()
            if self.cpu_prefetch_stream is not None:
                self.cpu_prefetch_stream.wait_stream(current_stream)
            if host_timing is not None:
                host_timing["submit_stream_wait_us"] = (
                    time.perf_counter() - stream_wait_start) * 1e6
            with torch.npu.stream(self.prefetch_stream):
                stream_wait_start = time.perf_counter()
                self.prefetch_stream.wait_stream(current_stream)
                if host_timing is not None:
                    host_timing["submit_prefetch_wait_stream_us"] = (
                        time.perf_counter() - stream_wait_start) * 1e6
                record_start = time.perf_counter()
                _event_record(start_event)
                if host_timing is not None:
                    host_timing["submit_start_event_record_us"] = (
                        time.perf_counter() - record_start) * 1e6
                populate_start = time.perf_counter()
                self._populate_slot(slot,
                                    layer,
                                    async_copy=True,
                                    cpu_prefetch_stream=self.cpu_prefetch_stream,
                                    reason=reason,
                                    timing_events=timing_events,
                                    host_timing=host_timing)
                if host_timing is not None:
                    host_timing["submit_populate_us"] = (
                        time.perf_counter() - populate_start) * 1e6
                post_cpu_start = time.perf_counter()
                if slot.has_async_cpu_direct or slot.has_async_cpu_pack:
                    self.prefetch_stream.wait_event(slot.cpu_pack_event)
                elif slot.has_async_cpu_copy:
                    self.prefetch_stream.wait_event(slot.cpu_ready_event)
                    stage_to_slot = [
                        (int(slot_idx), stage_idx)
                        for stage_idx, slot_idx in enumerate(
                            slot.cpu_stage_slot_ids[:slot.cpu_stage_count])
                    ]
                    _event_record(timing_events.get("cpu_pack_start_event"))
                    self._copy_npu_assignment_runs(slot.w13,
                                                   slot.cpu_stage_w13,
                                                   stage_to_slot,
                                                   async_copy=True)
                    self._copy_npu_assignment_runs(slot.w2,
                                                   slot.cpu_stage_w2,
                                                   stage_to_slot,
                                                   async_copy=True)
                    _event_record(timing_events.get("cpu_pack_end_event"))
                if host_timing is not None:
                    host_timing["submit_post_cpu_wait_us"] = (
                        time.perf_counter() - post_cpu_start) * 1e6
                record_start = time.perf_counter()
                _event_record(end_event)
                slot.ready_event.record()
                if host_timing is not None:
                    host_timing["submit_ready_record_us"] = (
                        time.perf_counter() - record_start) * 1e6
            slot.inflight_prefetch = True
        else:
            populate_start = time.perf_counter()
            self._populate_slot(slot,
                                layer,
                                async_copy=False,
                                reason=reason,
                                timing_events=timing_events,
                                host_timing=host_timing)
            if host_timing is not None:
                host_timing["submit_populate_us"] = (
                    time.perf_counter() - populate_start) * 1e6
            record_start = time.perf_counter()
            slot.ready_event.record()
            if host_timing is not None:
                host_timing["submit_ready_record_us"] = (
                    time.perf_counter() - record_start) * 1e6
            slot.inflight_prefetch = False
        slot.prefetch_timing = {
            "layer_idx": int(layer.layer_idx),
            "slot_id": int(slot_id),
            "reason": reason,
            "async": bool(async_copy),
            "submit_us": (time.perf_counter() - submit_start) * 1e6,
            "start_event": start_event,
            "end_event": end_event,
            **timing_events,
            "valid_experts": int(slot.valid_expert_count),
            "source_from_npu": int(slot.source_from_npu),
            "source_from_cpu": int(slot.source_from_cpu),
            "cpu_path": slot.prefetch_cpu_path,
            "layer_local_buffer": int(slot.uses_layer_local_buffer),
        }
        if host_timing is not None:
            slot.prefetch_timing.update(host_timing)
        if ((int(layer.layer_idx), slot_id) not in self._logged_prefetches
                and async_copy and self.enable_transfer_logs):
            logger.info(
                "Mode3 prefetch scheduled: layer=%s slot=%s reason=%s valid_experts=%s source_from_npu=%s source_from_cpu=%s layer_local_buffer=%s slot_head=%s",
                layer.layer_idx,
                slot_id,
                reason,
                slot.valid_expert_count,
                slot.source_from_npu,
                slot.source_from_cpu,
                int(slot.uses_layer_local_buffer),
                list(slot.expert_ids[:8]),
            )
            self._logged_prefetches.add((int(layer.layer_idx), slot_id))
        return slot

    def bind_current_layer(self, layer: Any) -> _Mode3DoubleBufferSlot:
        slot_id = self._slot_id_for_layer(int(layer.layer_idx))
        slot = self.slots[slot_id]
        prefetched_hit = self._slot_matches(slot, layer)
        if not self._slot_matches(slot, layer):
            slot = self.prepare_slot(layer, slot_id, async_copy=False,
                                     reason="sync_current")
        wait_start = time.perf_counter()
        ready_wait_start_event = None
        ready_wait_end_event = None
        # Ensure the current layer never starts computing before every expert
        # row for this slot has fully arrived in the runtime buffer. The
        # device-event path preserves the dependency without blocking Python,
        # which lets the double-buffered prefetch pipeline stay deeper.
        wait_mode = "host_sync"
        if slot.inflight_prefetch and self.enable_device_ready_wait:
            ready_wait_start_event = self.new_timing_event()
            ready_wait_end_event = self.new_timing_event()
            current_stream = torch.npu.current_stream()
            _event_record(ready_wait_start_event)
            current_stream.wait_event(slot.ready_event)
            _event_record(ready_wait_end_event)
            wait_mode = "device_event"
        else:
            slot.ready_event.synchronize()
        wait_us = (time.perf_counter() - wait_start) * 1e6
        cpu_fill_start = time.perf_counter()
        self._fill_pending_cpu_rows(slot, layer)
        cpu_fill_us = (time.perf_counter() - cpu_fill_start) * 1e6
        self.prefetch_wait_us[int(layer.layer_idx)] += wait_us
        self.prefetch_wait_count[int(layer.layer_idx)] += 1
        if prefetched_hit:
            self.prefetch_hit[int(layer.layer_idx)] += 1
        self.last_bind_timing = {
            "layer_idx": int(layer.layer_idx),
            "slot_id": int(slot_id),
            "valid_experts": int(slot.valid_expert_count),
            "source_from_npu": int(slot.source_from_npu),
            "source_from_cpu": int(slot.source_from_cpu),
            "layer_local_buffer": int(slot.uses_layer_local_buffer),
            "wait_mode": wait_mode,
            "wait_us": float(wait_us),
            "ready_wait_start_event": ready_wait_start_event,
            "ready_wait_end_event": ready_wait_end_event,
            "cpu_fill_us": float(cpu_fill_us),
            "prefetched_hit": bool(prefetched_hit),
            "prefetch_hit_count": int(self.prefetch_hit[int(layer.layer_idx)]),
        }
        layer.runtime_w13_weight = slot.w13[:slot.valid_expert_count]
        layer.runtime_w2_weight = slot.w2[:slot.valid_expert_count]
        layer.runtime_weight_capacity = int(slot.w13.shape[0])
        layer.runtime_w13_buffer = slot.w13
        layer.runtime_w2_buffer = slot.w2
        layer.expert_map = slot.expert_map
        layer.active_local_num_experts = slot.valid_expert_count
        layer.local_num_experts = slot.valid_expert_count
        layer.moe_config.num_local_experts = slot.valid_expert_count
        layer.moe_config.num_experts = slot.valid_expert_count
        layer.num_experts = slot.valid_expert_count
        layer.lossless_runtime_dispatch_log2phy = slot.dispatch_log2phy
        layer.lossless_runtime_dispatch_num_experts = int(
            slot.dispatch_num_experts)
        layer.lossless_runtime_dispatch_signature = slot.dispatch_signature
        layer.lossless_runtime_dispatch_active_rank_count = int(
            slot.dispatch_active_rank_count)
        layer.lossless_runtime_dispatch_owned_per_rank = int(
            slot.dispatch_owned_per_rank)
        layer.lossless_runtime_activated = True
        slot.inflight_prefetch = False
        log_key = (int(layer.layer_idx), slot_id, slot.valid_expert_count)
        if self.enable_transfer_logs and log_key not in self._logged_slot_bindings:
            logger.info(
                "Mode3 slot binding: layer=%s slot=%s valid_experts=%s source_from_npu=%s source_from_cpu=%s wait_mode=%s wait_us=%.1f cpu_fill_us=%.1f slot_head=%s",
                layer.layer_idx,
                slot_id,
                slot.valid_expert_count,
                slot.source_from_npu,
                slot.source_from_cpu,
                wait_mode,
                wait_us,
                cpu_fill_us,
                list(slot.expert_ids[:8]),
            )
            self._logged_slot_bindings.add(log_key)
        return slot

    def _prefetch_layer(self, target_layer: Any, reason: str) -> dict[str, Any]:
        if target_layer is None:
            return {"status": "no_target_layer"}
        if getattr(target_layer, "elastic_execution_mode", 0) != 3:
            return {"status": "next_not_mode3"}
        if not getattr(target_layer, "lossless_hybrid_active", False):
            return {"status": "next_inactive"}
        cpu_only_count = len(
            getattr(target_layer, "lossless_hybrid_cpu_only_expert_ids", []))
        if not self.enable_async_npu_prefetch:
            log_key = (int(target_layer.layer_idx), cpu_only_count)
            if (self.enable_transfer_logs
                    and log_key not in self._logged_prefetch_deferrals):
                logger.info(
                    "Mode3 prefetch deferred to layer entry: layer=%s reason=async_npu_prefetch_unstable cpu_only_experts=%s",
                    target_layer.layer_idx,
                    cpu_only_count,
                )
                self._logged_prefetch_deferrals.add(log_key)
            return {
                "status": "disabled",
                "next_layer": int(target_layer.layer_idx),
                "cpu_only": int(cpu_only_count),
            }
        if cpu_only_count > 0 and self.cpu_prefetch_stream is None:
            log_key = (int(target_layer.layer_idx), cpu_only_count)
            if (self.enable_transfer_logs
                    and log_key not in self._logged_prefetch_deferrals):
                logger.info(
                    "Mode3 prefetch deferred to layer entry: layer=%s reason=cpu_shadow_copy_stability cpu_only_experts=%s",
                    target_layer.layer_idx,
                    cpu_only_count,
                )
                self._logged_prefetch_deferrals.add(log_key)
            return {
                "status": "no_cpu_stream",
                "next_layer": int(target_layer.layer_idx),
                "cpu_only": int(cpu_only_count),
            }
        next_slot_id = self._slot_id_for_layer(int(target_layer.layer_idx))
        was_match = self._slot_matches(self.slots[next_slot_id], target_layer)
        slot = self.prepare_slot(target_layer,
                                 next_slot_id,
                                 async_copy=True,
                                 reason=reason)
        timing = dict(getattr(slot, "prefetch_timing", {}))
        timing["status"] = "hit" if was_match else "scheduled"
        return timing

    def prefetch_next_layer(self, current_layer: Any) -> dict[str, Any]:
        next_layer = self._get_next_layer(int(current_layer.layer_idx))
        if next_layer is None:
            return {"status": "no_next_layer"}
        return self._prefetch_layer(
            next_layer, reason=f"after_layer_{current_layer.layer_idx}")

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

    @staticmethod
    def _compact_wave_topk(logical_topk_ids: torch.Tensor,
                           topk_weights: torch.Tensor,
                           token_mask: torch.Tensor,
                           topk_mask: torch.Tensor,
                           preserve_full_topk: bool = True
                           ) -> tuple[Optional[torch.Tensor],
                                      Optional[torch.Tensor],
                                      Optional[torch.Tensor],
                                      Optional[torch.Tensor]]:
        def _empty_wave() -> tuple[Optional[torch.Tensor],
                                   Optional[torch.Tensor],
                                   Optional[torch.Tensor],
                                   Optional[torch.Tensor]]:
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
        # Keep every padded top-k id mapped to a valid expert in this wave.
        # The corresponding routing weight stays zero, but using a valid id
        # avoids remapping padded entries to -1 before the device kernel sees
        # them.
        filled_mask = (torch.arange(wave_topk, device=selected_ids.device)
                       .unsqueeze(0) < fill_pos.unsqueeze(1))
        fallback_ids = wave_ids[:, :1].expand(-1, wave_topk)
        wave_ids = torch.where(filled_mask, wave_ids, fallback_ids)
        wave_row_idx = return_row_idx(wave_ids, wave_topk)
        return token_indices, wave_ids, wave_weights, wave_row_idx

    def _build_padded_mc2_wave_inputs(
            self,
            hidden_states: torch.Tensor,
            logical_topk_ids: torch.Tensor,
            topk_weights: torch.Tensor,
            token_mask: torch.Tensor,
            topk_mask: torch.Tensor,
            fallback_expert_id: int,
            forced_active_topk: Optional[int] = None
    ) -> tuple[Optional[torch.Tensor], torch.Tensor, torch.Tensor,
               torch.Tensor, torch.Tensor, torch.Tensor, int]:
        token_indices, compact_ids, compact_weights, compact_row_idx = self._compact_wave_topk(
            logical_topk_ids=logical_topk_ids,
            topk_weights=topk_weights,
            token_mask=token_mask,
            topk_mask=topk_mask,
            preserve_full_topk=(forced_active_topk is None))
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
            # A3 MC2 dispatch/compute kernels do not tolerate a fully empty
            # local batch inside a collective wave. Keep a single padded row so
            # every rank still enters the collective with BS=1, but leave the
            # active mask fully disabled so dispatch/combine do not process a
            # fabricated token.
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

    def _build_grouped_mc2_wave_inputs(
            self,
            hidden_states: torch.Tensor,
            logical_topk_ids: torch.Tensor,
            topk_weights: torch.Tensor,
            token_mask: torch.Tensor,
            topk_mask: torch.Tensor,
            active_topk_values: list[int],
            fallback_expert_id: int,
            launch_topk: Optional[int] = None,
    ) -> list[tuple[Optional[torch.Tensor], torch.Tensor, torch.Tensor,
                    torch.Tensor, torch.Tensor, torch.Tensor, int, bool]]:
        grouped_inputs = []
        fallback_expert_id = int(fallback_expert_id)
        launch_topk = (int(launch_topk)
                       if launch_topk is not None else None)
        empty_token_indices = torch.empty((0, ),
                                          dtype=torch.long,
                                          device=hidden_states.device)
        if token_mask.numel() == 0 or not torch.any(token_mask):
            token_indices = empty_token_indices
            per_token_counts = torch.empty((0, ),
                                           dtype=torch.long,
                                           device=hidden_states.device)
        else:
            token_indices = torch.nonzero(token_mask, as_tuple=False).reshape(-1)
            selected_mask = topk_mask[token_indices]
            per_token_counts = selected_mask.sum(dim=1)

        for active_topk in active_topk_values:
            active_topk = int(active_topk)
            if active_topk <= 0:
                continue
            if token_indices.numel() > 0:
                group_token_indices = token_indices[per_token_counts == active_topk]
            else:
                group_token_indices = empty_token_indices
            group_token_mask = torch.zeros_like(token_mask, dtype=torch.bool)
            group_token_mask[group_token_indices] = True
            (group_indices, group_hidden_states, group_ids, group_weights,
             group_row_idx, group_mc2_mask,
             active_count) = self._build_padded_mc2_wave_inputs(
                 hidden_states=hidden_states,
                 logical_topk_ids=logical_topk_ids,
                 topk_weights=topk_weights,
                 token_mask=group_token_mask,
                 topk_mask=topk_mask,
                 fallback_expert_id=fallback_expert_id,
                 forced_active_topk=(launch_topk
                                     if launch_topk is not None else
                                     active_topk))
            grouped_inputs.append((group_indices, group_hidden_states,
                                   group_ids, group_weights, group_row_idx,
                                   group_mc2_mask, active_count,
                                   active_count == 0))
        return grouped_inputs

    @staticmethod
    def _group_list_to_counts(group_list: torch.Tensor,
                              group_list_type: int,
                              total_tokens: Optional[int] = None) -> torch.Tensor:
        if group_list_type == 1:
            return group_list
        if group_list_type == 0:
            if group_list.numel() == 0:
                return group_list
            return torch.cat([group_list[:1], torch.diff(group_list, dim=0)])
        raise RuntimeError(
            f"Unsupported group_list_type for hybrid single-dispatch path: "
            f"{group_list_type}")

    @staticmethod
    def _counts_to_offsets(group_counts: torch.Tensor) -> torch.Tensor:
        if group_counts.numel() == 0:
            return torch.zeros((1, ),
                               dtype=torch.long,
                               device=group_counts.device)
        counts = group_counts.to(dtype=torch.long)
        return torch.cat([
            torch.zeros((1, ), dtype=torch.long, device=counts.device),
            counts.cumsum(dim=0)
        ])

    @staticmethod
    def _get_dispatch_log2phy_for_layer(
            layer: Any,
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
        cached_signature = getattr(layer, "lossless_runtime_dispatch_signature",
                                   None)
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
        layer.lossless_runtime_dispatch_active_rank_count = int(
            active_rank_count)
        layer.lossless_runtime_dispatch_owned_per_rank = int(owned_per_rank)
        return dispatch_log2phy, dispatch_num_experts

    @staticmethod
    def _compute_local_dispatch_counts_from_topk(
            dispatch_topk_ids: torch.Tensor,
            mc2_mask: Optional[torch.Tensor],
            local_rank_idx: int,
            owned_per_rank: int) -> torch.Tensor:
        local_dense_start = local_rank_idx * owned_per_rank
        local_dense_end = local_dense_start + owned_per_rank
        selected_topk_ids = dispatch_topk_ids
        if mc2_mask is not None:
            token_mask = mc2_mask.to(device=dispatch_topk_ids.device,
                                     dtype=torch.bool)
            if token_mask.shape[0] != dispatch_topk_ids.shape[0]:
                raise RuntimeError(
                    "Hybrid mode2 single-dispatch mask/topk length mismatch: "
                    f"mask_tokens={token_mask.shape[0]} "
                    f"topk_tokens={dispatch_topk_ids.shape[0]}")
            selected_topk_ids = dispatch_topk_ids[token_mask]
        flat_topk_ids = selected_topk_ids.reshape(-1).to(dtype=torch.long)
        if flat_topk_ids.numel() == 0:
            return torch.zeros((owned_per_rank, ),
                               dtype=torch.long,
                               device=dispatch_topk_ids.device)
        local_mask = ((flat_topk_ids >= local_dense_start)
                      & (flat_topk_ids < local_dense_end))
        local_flat_ids = flat_topk_ids[local_mask] - local_dense_start
        if local_flat_ids.numel() == 0:
            return torch.zeros((owned_per_rank, ),
                               dtype=torch.long,
                               device=dispatch_topk_ids.device)
        return torch.bincount(local_flat_ids,
                              minlength=owned_per_rank).to(dtype=torch.long)

    def _should_use_mode2_single_dispatch_hybrid_path(
            self,
            layer: torch.nn.Module,
            wave_plans: list[dict[str, Any]],
            use_dense_mc2_waves: bool,
            shared_experts: Optional[Any]) -> bool:
        if not use_dense_mc2_waves or len(wave_plans) <= 1:
            return False
        if self.dynamic_eplb:
            return False
        if getattr(layer, "elastic_execution_mode", 0) != 2:
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
        if len(set(owned_counts)) != 1:
            return False
        return True

    @staticmethod
    def _should_disable_grouped_dense_mc2_multi_wave(
            layer: torch.nn.Module) -> bool:
        if getattr(layer, "elastic_execution_mode", 0) != 2:
            return False
        if not hasattr(layer, "_is_hybrid_cpu_swap_enabled"):
            return False
        return bool(layer._is_hybrid_cpu_swap_enabled())

    @staticmethod
    def _should_use_mode3_cross_layer_buffer_path(
            layer: torch.nn.Module,
            shared_experts: Optional[Any],
            is_dummy: bool) -> bool:
        if getattr(layer, "elastic_execution_mode", 0) != 3:
            return False
        if not getattr(layer, "lossless_hybrid_active", False):
            return False
        if shared_experts is not None:
            return False
        forward_context = get_forward_context()
        comm_type = getattr(forward_context, "moe_comm_type", None)
        if comm_type == MoECommType.MC2:
            return True
        selected_comm_type = getattr(forward_context, "selected_moe_comm_type",
                                     None)
        if selected_comm_type == MoECommType.MC2:
            return True
        moe_comm_method = getattr(forward_context, "moe_comm_method", None)
        token_dispatcher = getattr(moe_comm_method, "token_dispatcher", None)
        if (token_dispatcher is not None
                and token_dispatcher.__class__.__name__
                == "TokenDispatcherWithMC2"):
            return True
        return False

    @staticmethod
    def _should_use_mode3_single_rank_allgather_path(
            layer: torch.nn.Module,
            shared_experts: Optional[Any],
            is_dummy: bool) -> bool:
        if getattr(layer, "elastic_execution_mode", 0) != 3:
            return False
        if not getattr(layer, "lossless_hybrid_active", False):
            return False
        if shared_experts is not None:
            return False
        active_rank_count = len(
            getattr(layer, "lossless_hybrid_active_ranks", []))
        if active_rank_count != 1:
            return False
        forward_context = get_forward_context()
        selected_comm_type = getattr(forward_context, "selected_moe_comm_type",
                                     None)
        comm_type = getattr(forward_context, "moe_comm_type", None)
        if selected_comm_type != MoECommType.ALLGATHER and comm_type != MoECommType.ALLGATHER:
            return False
        moe_comm_method = getattr(forward_context, "moe_comm_method", None)
        token_dispatcher = getattr(moe_comm_method, "token_dispatcher", None)
        return (token_dispatcher is not None
                and token_dispatcher.__class__.__name__
                == "TokenDispatcherWithAllGather")

    @staticmethod
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

    def _execute_mode3_single_rank_allgather_hybrid(
            self,
            layer: torch.nn.Module,
            x: torch.Tensor,
            logical_topk_ids: torch.Tensor,
            topk_weights: torch.Tensor,
            row_idx: torch.Tensor,
            shared_experts: Optional[Any],
            enable_force_load_balance: bool,
            kwargs: dict[str, Any]) -> torch.Tensor:
        manager = self._get_or_create_mode3_double_buffer_manager(layer)
        if manager is None:
            raise RuntimeError(
                "Mode3 single-rank AllGather path requires forward-context "
                f"model instance and prefetch stream at layer={getattr(layer, 'layer_idx', -1)}.")
        profile_timing = manager.should_profile_layer(layer,
                                                      "single_rank_allgather")
        bound_slot = manager.bind_current_layer(layer)
        bind_timing = dict(manager.last_bind_timing)
        next_prefetch_timing = manager.prefetch_next_layer(layer)
        compute_wall_start = time.perf_counter()
        compute_start_event = manager.new_timing_event() if profile_timing else None
        compute_end_event = manager.new_timing_event() if profile_timing else None
        remap_start_event = manager.new_timing_event() if profile_timing else None
        remap_end_event = manager.new_timing_event() if profile_timing else None
        fused_start_event = manager.new_timing_event() if profile_timing else None
        fused_end_event = manager.new_timing_event() if profile_timing else None
        _event_record(compute_start_event)

        moe_comm_method = get_forward_context().moe_comm_method
        token_dispatcher = getattr(moe_comm_method, "token_dispatcher", None)
        if (token_dispatcher is None
                or token_dispatcher.__class__.__name__
                != "TokenDispatcherWithAllGather"):
            raise RuntimeError(
                "Mode3 single-rank path requires AllGather token dispatcher, got "
                f"{token_dispatcher.__class__.__name__ if token_dispatcher is not None else None} "
                f"at layer={getattr(layer, 'layer_idx', -1)}.")

        remap_wall_start = time.perf_counter()
        _event_record(remap_start_event)
        slot_log2phy = bound_slot.expert_map.to(device=logical_topk_ids.device)
        local_topk_ids = slot_log2phy[logical_topk_ids]
        if torch.any(local_topk_ids < 0):
            invalid_count = int(torch.count_nonzero(local_topk_ids < 0).item())
            raise RuntimeError(
                "Mode3 single-rank AllGather saw experts outside the bound slot "
                f"at layer={getattr(layer, 'layer_idx', -1)}: "
                f"invalid_count={invalid_count}")
        if enable_force_load_balance and not self.use_aclgraph:
            local_topk_ids = (
                torch.arange(local_topk_ids.numel(),
                             device=local_topk_ids.device)
                % bound_slot.valid_expert_count).to(torch.int32).reshape(
                    local_topk_ids.shape)
        _event_record(remap_end_event)
        remap_wall_ms = (time.perf_counter() - remap_wall_start) * 1e3

        old_num_experts = getattr(token_dispatcher, "num_experts", None)
        old_num_experts_local = getattr(token_dispatcher, "num_experts_local",
                                        None)
        token_dispatcher.num_experts = bound_slot.valid_expert_count
        token_dispatcher.num_experts_local = bound_slot.valid_expert_count
        fused_wall_start = time.perf_counter()
        _event_record(fused_start_event)
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
                shared_experts=shared_experts,
                quantized_x_for_share=kwargs.get("quantized_x_for_share"),
                dynamic_scale_for_share=kwargs.get("dynamic_scale_for_share"),
                log2phy=None,
                global_redundant_expert_num=0,
                need_trans=True,
                dynamic_eplb=self.dynamic_eplb,
                mc2_mask=None)
            _event_record(fused_end_event)
        finally:
            if old_num_experts is not None:
                token_dispatcher.num_experts = old_num_experts
            if old_num_experts_local is not None:
                token_dispatcher.num_experts_local = old_num_experts_local
        fused_wall_ms = (time.perf_counter() - fused_wall_start) * 1e3
        _event_record(compute_end_event)
        compute_wall_ms = (time.perf_counter() - compute_wall_start) * 1e3

        layer.lossless_hybrid_last_stats = {
            "mode3_slot": int(layer.layer_idx) & 1,
            "valid_experts": bound_slot.valid_expert_count,
            "source_from_npu": bound_slot.source_from_npu,
            "source_from_cpu": bound_slot.source_from_cpu,
            "single_rank_allgather": 1,
        }
        if layer.layer_idx == 0 and manager.enable_transfer_logs:
            logger.info(
                "Mode3 single-rank AllGather execution: rank=%s layer=%s valid_experts=%s source_from_npu=%s source_from_cpu=%s",
                layer.rank if hasattr(layer, "rank") else -1,
                layer.layer_idx,
                bound_slot.valid_expert_count,
                bound_slot.source_from_npu,
                bound_slot.source_from_cpu,
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
            ready_wait_dev_ms = _elapsed_ms(
                bind_timing.get("ready_wait_start_event"),
                bind_timing.get("ready_wait_end_event"))
            logger.info(
                "Mode3 timing single-rank-allgather: rank=%s layer=%s slot=%s valid_experts=%s source_from_npu=%s source_from_cpu=%s bind_wait_us=%.1f bind_cpu_fill_us=%.1f bind_wait_mode=%s ready_wait_dev_ms=%.3f prefetch_next_layer=%s prefetch_slot=%s prefetch_source_from_cpu=%s prefetch_cpu_path=%s prefetch_cpu_w13_pinned=%s prefetch_cpu_w2_pinned=%s prefetch_cpu_w13_contig=%s prefetch_cpu_w2_contig=%s prefetch_submit_us=%.1f submit_accounted_us=%.1f submit_event_alloc_us=%.1f submit_stream_wait_us=%.1f submit_prefetch_wait_stream_us=%.1f submit_start_event_record_us=%.1f submit_populate_us=%.1f submit_order_us=%.1f submit_assign_us=%.1f submit_layer_local_check_us=%.1f submit_npu_us=%.1f submit_cpu_us=%.1f submit_cpu_direct_async_us=%.1f submit_cpu_stage_async_us=%.1f submit_plan_log_us=%.1f submit_expert_map_us=%.1f submit_expert_map_cache_hit=%s submit_dispatch_cache_us=%.1f submit_slot_state_us=%.1f submit_post_cpu_wait_us=%.1f submit_ready_record_us=%.1f prefetch_dev_ms=%.3f prefetch_npu_dev_ms=%.3f prefetch_cpu_dev_ms=%.3f prefetch_cpu_pack_dev_ms=%.3f current_compute_wall_ms=%.3f current_compute_dev_ms=%.3f remap_wall_ms=%.3f remap_dev_ms=%.3f fused_allgather_wall_ms=%.3f fused_allgather_dev_ms=%.3f prefetch_minus_compute_dev_ms=%.3f tokens=%s",
                layer.rank if hasattr(layer, "rank") else -1,
                layer.layer_idx,
                int(bind_timing.get("slot_id", -1)),
                bound_slot.valid_expert_count,
                bound_slot.source_from_npu,
                bound_slot.source_from_cpu,
                float(bind_timing.get("wait_us", -1.0)),
                float(bind_timing.get("cpu_fill_us", -1.0)),
                bind_timing.get("wait_mode", "unknown"),
                ready_wait_dev_ms,
                next_prefetch_timing.get("layer_idx", -1),
                next_prefetch_timing.get("slot_id", -1),
                next_prefetch_timing.get("source_from_cpu", -1),
                next_prefetch_timing.get("cpu_path", "unknown"),
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
                _timing_float(next_prefetch_timing,
                              "submit_layer_local_check_us"),
                _timing_float(next_prefetch_timing, "submit_npu_us"),
                _timing_float(next_prefetch_timing, "submit_cpu_us"),
                _timing_float(next_prefetch_timing,
                              "submit_cpu_direct_async_us"),
                _timing_float(next_prefetch_timing,
                              "submit_cpu_stage_async_us"),
                _timing_float(next_prefetch_timing, "submit_plan_log_us"),
                _timing_float(next_prefetch_timing, "submit_expert_map_us"),
                int(_timing_float(next_prefetch_timing,
                                  "expert_map_cache_hit")),
                _timing_float(next_prefetch_timing,
                              "submit_dispatch_cache_us"),
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
                _elapsed_ms(remap_start_event, remap_end_event),
                fused_wall_ms,
                _elapsed_ms(fused_start_event, fused_end_event),
                prefetch_dev_ms - compute_dev_ms
                if prefetch_dev_ms >= 0 and compute_dev_ms >= 0 else -1.0,
                int(x.shape[0]) if hasattr(x, "shape") else -1,
            )
        return final_hidden_states

    def _execute_mode3_fused_experts_hybrid(
            self,
            layer: torch.nn.Module,
            x: torch.Tensor,
            logical_topk_ids: torch.Tensor,
            topk_weights: torch.Tensor,
            row_idx: torch.Tensor,
            shared_experts: Optional[Any],
            enable_force_load_balance: bool,
            kwargs: dict[str, Any],
            manager: Optional[Mode3DoubleBufferManager] = None) -> torch.Tensor:
        if manager is None:
            manager = self._get_or_create_mode3_double_buffer_manager(layer)
        if manager is None:
            raise RuntimeError(
                "Mode3 fused-experts path requires forward-context model "
                f"instance and moe prefetch stream at layer={getattr(layer, 'layer_idx', -1)}.")
        profile_timing = manager.should_profile_layer(layer, "fused_experts")
        bound_slot = manager.bind_current_layer(layer)
        bind_timing = dict(manager.last_bind_timing)
        next_prefetch_timing = manager.prefetch_next_layer(layer)

        local_rank_idx = int(
            getattr(layer, "lossless_hybrid_active_rank_index", -1))
        rank_owned = getattr(layer, "lossless_hybrid_rank_owned_expert_ids", None)
        if (rank_owned is None or local_rank_idx < 0
                or local_rank_idx >= len(rank_owned)):
            raise RuntimeError(
                "Invalid hybrid rank ownership state for mode3 fused-experts "
                f"path at layer={getattr(layer, 'layer_idx', -1)}.")
        local_owned_expert_ids = [int(x) for x in rank_owned[local_rank_idx]]
        active_rank_count = len(getattr(layer, "lossless_hybrid_active_ranks", []))
        owned_per_rank = len(local_owned_expert_ids)
        if active_rank_count <= 0 or owned_per_rank <= 0:
            raise RuntimeError(
                "Invalid hybrid mode3 fused-experts shape: "
                f"active_rank_count={active_rank_count} "
                f"owned_per_rank={owned_per_rank}")

        remap_wall_start = time.perf_counter()
        dispatch_log2phy, dispatch_num_experts = \
            self._get_dispatch_log2phy_for_layer(
                layer,
                device=logical_topk_ids.device,
                rank_owned=rank_owned,
                active_rank_count=active_rank_count,
                owned_per_rank=owned_per_rank,
            )
        remap_wall_ms = (time.perf_counter() - remap_wall_start) * 1e3

        compute_wall_start = time.perf_counter()
        compute_start_event = manager.new_timing_event() if profile_timing else None
        compute_end_event = manager.new_timing_event() if profile_timing else None
        fused_start_event = manager.new_timing_event() if profile_timing else None
        fused_end_event = manager.new_timing_event() if profile_timing else None
        _event_record(compute_start_event)

        moe_comm_method = get_forward_context().moe_comm_method
        token_dispatcher = getattr(moe_comm_method, "token_dispatcher", None)
        old_dispatch_num_experts = int(getattr(token_dispatcher, "num_experts", 0)
                                       ) if token_dispatcher is not None else 0
        old_expert_token_nums_type = int(
            getattr(token_dispatcher, "expert_token_nums_type", 0)
        ) if token_dispatcher is not None else 0
        try:
            if token_dispatcher is not None:
                token_dispatcher.num_experts = dispatch_num_experts
                token_dispatcher.expert_token_nums_type = (
                    manager.expert_token_nums_type)
            fused_wall_start = time.perf_counter()
            _event_record(fused_start_event)
            final_hidden_states = self._execute_single_wave(
                layer=layer,
                hidden_states=x,
                logical_topk_ids=logical_topk_ids,
                topk_weights=topk_weights,
                row_idx=row_idx,
                global_num_experts=dispatch_num_experts,
                shared_experts=shared_experts,
                log2phy=dispatch_log2phy,
                mc2_mask=getattr(moe_comm_method, "mc2_mask", None),
                enable_force_load_balance=enable_force_load_balance,
                kwargs=kwargs)
            _event_record(fused_end_event)
            fused_wall_ms = (time.perf_counter() - fused_wall_start) * 1e3
        finally:
            if token_dispatcher is not None:
                token_dispatcher.num_experts = old_dispatch_num_experts
                token_dispatcher.expert_token_nums_type = (
                    old_expert_token_nums_type)
        _event_record(compute_end_event)
        compute_wall_ms = (time.perf_counter() - compute_wall_start) * 1e3

        layer.lossless_hybrid_last_stats = {
            "mode3_slot": int(layer.layer_idx) & 1,
            "valid_experts": bound_slot.valid_expert_count,
            "source_from_npu": bound_slot.source_from_npu,
            "source_from_cpu": bound_slot.source_from_cpu,
            "layer_local_buffer": int(bound_slot.uses_layer_local_buffer),
            "prefetch_wait_us": float(
                manager.prefetch_wait_us[int(layer.layer_idx)]),
            "prefetch_hit": int(manager.prefetch_hit[int(layer.layer_idx)]),
            "fused_experts_path": 1,
        }
        if layer.layer_idx == 0 and manager.enable_transfer_logs:
            logger.info(
                "Mode3 fused-experts execution: rank=%s layer=%s stage=%s valid_experts=%s source_from_npu=%s source_from_cpu=%s layer_local_buffer=%s dispatch_experts=%s prefetch_wait_us=%.1f prefetch_hit=%s",
                layer.rank if hasattr(layer, "rank") else -1,
                layer.layer_idx,
                active_rank_count,
                bound_slot.valid_expert_count,
                bound_slot.source_from_npu,
                bound_slot.source_from_cpu,
                int(bound_slot.uses_layer_local_buffer),
                dispatch_num_experts,
                float(manager.prefetch_wait_us[int(layer.layer_idx)]),
                int(manager.prefetch_hit[int(layer.layer_idx)]),
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
                "Mode3 timing fused-experts: rank=%s layer=%s stage=%s slot=%s valid_experts=%s source_from_npu=%s source_from_cpu=%s layer_local_buffer=%s bind_wait_us=%.1f bind_cpu_fill_us=%.1f bind_wait_mode=%s ready_wait_dev_ms=%.3f prefetch_status=%s prefetch_next_layer=%s prefetch_slot=%s prefetch_source_from_cpu=%s prefetch_cpu_path=%s prefetch_layer_local_buffer=%s prefetch_cpu_w13_pinned=%s prefetch_cpu_w2_pinned=%s prefetch_cpu_w13_contig=%s prefetch_cpu_w2_contig=%s prefetch_submit_us=%.1f submit_accounted_us=%.1f submit_event_alloc_us=%.1f submit_stream_wait_us=%.1f submit_prefetch_wait_stream_us=%.1f submit_start_event_record_us=%.1f submit_populate_us=%.1f submit_order_us=%.1f submit_assign_us=%.1f submit_layer_local_check_us=%.1f submit_npu_us=%.1f submit_cpu_us=%.1f submit_cpu_direct_async_us=%.1f submit_cpu_stage_async_us=%.1f submit_plan_log_us=%.1f submit_expert_map_us=%.1f submit_expert_map_cache_hit=%s submit_dispatch_cache_us=%.1f submit_slot_state_us=%.1f submit_post_cpu_wait_us=%.1f submit_ready_record_us=%.1f prefetch_dev_ms=%.3f prefetch_npu_dev_ms=%.3f prefetch_cpu_dev_ms=%.3f prefetch_cpu_pack_dev_ms=%.3f current_compute_wall_ms=%.3f current_compute_dev_ms=%.3f remap_wall_ms=%.3f fused_wall_ms=%.3f fused_dev_ms=%.3f prefetch_minus_compute_dev_ms=%.3f tokens=%s owned_per_rank=%s dispatch_num_experts=%s expert_token_nums_type=%s",
                layer.rank if hasattr(layer, "rank") else -1,
                layer.layer_idx,
                active_rank_count,
                int(bind_timing.get("slot_id", -1)),
                bound_slot.valid_expert_count,
                bound_slot.source_from_npu,
                bound_slot.source_from_cpu,
                int(bound_slot.uses_layer_local_buffer),
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
                _timing_float(next_prefetch_timing,
                              "submit_layer_local_check_us"),
                _timing_float(next_prefetch_timing, "submit_npu_us"),
                _timing_float(next_prefetch_timing, "submit_cpu_us"),
                _timing_float(next_prefetch_timing,
                              "submit_cpu_direct_async_us"),
                _timing_float(next_prefetch_timing,
                              "submit_cpu_stage_async_us"),
                _timing_float(next_prefetch_timing, "submit_plan_log_us"),
                _timing_float(next_prefetch_timing, "submit_expert_map_us"),
                int(_timing_float(next_prefetch_timing,
                                  "expert_map_cache_hit")),
                _timing_float(next_prefetch_timing,
                              "submit_dispatch_cache_us"),
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
            )
            if getattr(manager, "enable_copy_format_diag", False):
                logger.info(
                    "Mode3 copy timing detail fused-experts: rank=%s layer=%s "
                    "prefetch_next_layer=%s slot=%s npu_total_ms=%.3f "
                    "npu_w13_ms=%.3f npu_w2_ms=%.3f cpu_ms=%.3f "
                    "cpu_pack_ms=%.3f",
                    layer.rank if hasattr(layer, "rank") else -1,
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
            self,
            layer: torch.nn.Module,
            x: torch.Tensor,
            logical_topk_ids: torch.Tensor,
            topk_weights: torch.Tensor,
            row_idx: torch.Tensor,
            shared_experts: Optional[Any],
            enable_force_load_balance: bool,
            kwargs: dict[str, Any]) -> torch.Tensor:
        manager = self._get_or_create_mode3_double_buffer_manager(layer)
        if manager is None:
            raise RuntimeError(
                "Mode3 single-dispatch requires forward-context model instance "
                f"and moe prefetch stream at layer={getattr(layer, 'layer_idx', -1)}.")
        if manager.use_fused_experts_path:
            return self._execute_mode3_fused_experts_hybrid(
                layer=layer,
                x=x,
                logical_topk_ids=logical_topk_ids,
                topk_weights=topk_weights,
                row_idx=row_idx,
                shared_experts=shared_experts,
                enable_force_load_balance=enable_force_load_balance,
                kwargs=kwargs,
                manager=manager)
        profile_timing = manager.should_profile_layer(layer, "single_dispatch")
        bound_slot = manager.bind_current_layer(layer)
        bind_timing = dict(manager.last_bind_timing)
        next_prefetch_timing = manager.prefetch_next_layer(layer)
        compute_wall_start = time.perf_counter()
        compute_start_event = manager.new_timing_event() if profile_timing else None
        compute_end_event = manager.new_timing_event() if profile_timing else None
        remap_start_event = manager.new_timing_event() if profile_timing else None
        remap_end_event = manager.new_timing_event() if profile_timing else None
        dispatch_start_event = manager.new_timing_event() if profile_timing else None
        dispatch_end_event = manager.new_timing_event() if profile_timing else None
        mlp_start_event = manager.new_timing_event() if profile_timing else None
        mlp_end_event = manager.new_timing_event() if profile_timing else None
        combine_start_event = manager.new_timing_event() if profile_timing else None
        combine_end_event = manager.new_timing_event() if profile_timing else None
        _event_record(compute_start_event)

        moe_comm_method = get_forward_context().moe_comm_method
        token_dispatcher = getattr(moe_comm_method, "token_dispatcher", None)
        if token_dispatcher is None:
            raise RuntimeError("Missing MC2 token dispatcher for mode3 path.")
        if token_dispatcher.__class__.__name__ != "TokenDispatcherWithMC2":
            raise RuntimeError(
                "Mode3 single-dispatch requires MC2 token dispatcher, got "
                f"{token_dispatcher.__class__.__name__} at "
                f"layer={getattr(layer, 'layer_idx', -1)}.")
        local_rank_idx = int(
            getattr(layer, "lossless_hybrid_active_rank_index", -1))
        rank_owned = getattr(layer, "lossless_hybrid_rank_owned_expert_ids", None)
        if (rank_owned is None or local_rank_idx < 0
                or local_rank_idx >= len(rank_owned)):
            raise RuntimeError(
                "Invalid hybrid rank ownership state for mode3 path at "
                f"layer={getattr(layer, 'layer_idx', -1)}.")
        local_owned_expert_ids = [int(x) for x in rank_owned[local_rank_idx]]
        active_rank_count = len(getattr(layer, "lossless_hybrid_active_ranks", []))
        owned_per_rank = len(local_owned_expert_ids)
        if active_rank_count <= 0 or owned_per_rank <= 0:
            raise RuntimeError(
                "Invalid hybrid mode3 dispatch shape: "
                f"active_rank_count={active_rank_count} "
                f"owned_per_rank={owned_per_rank}")
        remap_wall_start = time.perf_counter()
        _event_record(remap_start_event)
        dispatch_log2phy, dispatch_num_experts = \
            self._get_dispatch_log2phy_for_layer(
                layer,
                device=logical_topk_ids.device,
                rank_owned=rank_owned,
                active_rank_count=active_rank_count,
                owned_per_rank=owned_per_rank,
            )
        dispatch_topk_ids = dispatch_log2phy[logical_topk_ids]
        if torch.any(dispatch_topk_ids < 0):
            invalid_count = int(torch.count_nonzero(dispatch_topk_ids < 0).item())
            raise RuntimeError(
                "Mode3 single-dispatch saw experts outside the active rank "
                f"ownership map at layer={getattr(layer, 'layer_idx', -1)}: "
                f"invalid_count={invalid_count}")
        if enable_force_load_balance and not self.use_aclgraph:
            dispatch_topk_ids = (
                torch.arange(dispatch_topk_ids.numel(),
                             device=dispatch_topk_ids.device)
                % dispatch_num_experts).to(torch.int32).reshape(
                    dispatch_topk_ids.shape)
        _event_record(remap_end_event)
        remap_wall_ms = (time.perf_counter() - remap_wall_start) * 1e3

        old_dispatch_num_experts = int(getattr(token_dispatcher, "num_experts", 0))
        old_expert_token_nums_type = int(
            getattr(token_dispatcher, "expert_token_nums_type", 0))
        token_dispatcher.num_experts = dispatch_num_experts
        token_dispatcher.expert_token_nums_type = manager.expert_token_nums_type
        dispatch_wall_ms = -1.0
        group_wall_ms = -1.0
        mlp_wall_ms = -1.0
        combine_wall_ms = -1.0
        dispatched_rows = -1
        dispatched_active_rows = -1
        try:
            dispatch_wall_start = time.perf_counter()
            _event_record(dispatch_start_event)
            dispatch_results = token_dispatcher.token_dispatch(
                hidden_states=x,
                topk_weights=topk_weights,
                topk_ids=dispatch_topk_ids,
                row_idx=row_idx,
                expert_map=layer.expert_map,
                log2phy=None,
                global_redundant_expert_num=kwargs.get(
                    "global_redundant_expert_num", 0),
                shared_experts=shared_experts,
                quantized_x_for_share=kwargs.get("quantized_x_for_share"),
                dynamic_scale_for_share=kwargs.get("dynamic_scale_for_share"),
                mc2_mask=getattr(moe_comm_method, "mc2_mask", None),
                apply_router_weight_on_input=False,
                with_quant=False)
            _event_record(dispatch_end_event)
            dispatch_wall_ms = (time.perf_counter() -
                                dispatch_wall_start) * 1e3
            dispatched_hidden_states = dispatch_results["hidden_states"]
            dispatched_group_list = dispatch_results["group_list"]
            dispatched_group_list_type = int(dispatch_results["group_list_type"])
            group_wall_start = time.perf_counter()
            dispatched_group_counts = self._group_list_to_counts(
                dispatched_group_list,
                dispatched_group_list_type,
                total_tokens=int(dispatched_hidden_states.shape[0]),
            ).to(dtype=torch.long)
            if int(dispatched_group_counts.numel()) != int(owned_per_rank):
                raise RuntimeError(
                    "Mode3 single-dispatch expert group size mismatch: "
                    f"expected_experts={owned_per_rank} "
                    f"actual_experts={int(dispatched_group_counts.numel())} "
                    f"group_list_type={dispatched_group_list_type}")
            dispatched_rows = int(dispatched_hidden_states.shape[0])
            if profile_timing and manager.enable_active_rows_sync:
                # Keep the hot path asynchronous. This host read is only for
                # sampled diagnostics and used to cost ~ms in group_wall_ms.
                dispatched_active_rows = int(dispatched_group_counts.sum().item())
            group_wall_ms = (time.perf_counter() - group_wall_start) * 1e3
            if (dispatched_active_rows >= 0
                    and dispatched_active_rows > int(dispatched_hidden_states.shape[0])):
                raise RuntimeError(
                    "Mode3 single-dispatch local token count mismatch: "
                    f"expected<={int(dispatched_hidden_states.shape[0])} "
                    f"active_rows={dispatched_active_rows} "
                    f"actual={int(dispatched_hidden_states.shape[0])} "
                    f"group_list_len={int(dispatched_group_list.numel())} "
                    f"group_list_type={dispatched_group_list_type} "
                    f"group_list_sum={int(dispatched_group_list.to(dtype=torch.long).sum().item())} "
                    f"group_count_sum={int(dispatched_group_counts.sum().item())}")
            runtime_w13_weight = getattr(layer, "runtime_w13_weight", None)
            runtime_w2_weight = getattr(layer, "runtime_w2_weight", None)
            if runtime_w13_weight is None or runtime_w2_weight is None:
                raise RuntimeError(
                    f"Missing runtime expert weights for mode3 path at layer={layer.layer_idx}.")
            mlp_wall_start = time.perf_counter()
            _event_record(mlp_start_event)
            dispatched_output = unified_apply_mlp(
                hidden_states=dispatched_hidden_states,
                w1=runtime_w13_weight,
                w1_scale=None,
                w2=runtime_w2_weight,
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
                need_trans=True)
            _event_record(mlp_end_event)
            mlp_wall_ms = (time.perf_counter() - mlp_wall_start) * 1e3
            combine_wall_start = time.perf_counter()
            _event_record(combine_start_event)
            final_hidden_states = token_dispatcher.token_combine(dispatched_output)
            _event_record(combine_end_event)
            combine_wall_ms = (time.perf_counter() -
                               combine_wall_start) * 1e3
        finally:
            token_dispatcher.num_experts = old_dispatch_num_experts
            token_dispatcher.expert_token_nums_type = old_expert_token_nums_type
        _event_record(compute_end_event)
        compute_wall_ms = (time.perf_counter() - compute_wall_start) * 1e3

        layer.lossless_hybrid_last_stats = {
            "mode3_slot": int(layer.layer_idx) & 1,
            "valid_experts": bound_slot.valid_expert_count,
            "source_from_npu": bound_slot.source_from_npu,
            "source_from_cpu": bound_slot.source_from_cpu,
            "prefetch_wait_us": float(manager.prefetch_wait_us[int(layer.layer_idx)]),
            "prefetch_hit": int(manager.prefetch_hit[int(layer.layer_idx)]),
        }
        if layer.layer_idx == 0 and manager.enable_transfer_logs:
            logger.info(
                "Mode3 single-dispatch execution: rank=%s layer=%s stage=%s valid_experts=%s source_from_npu=%s source_from_cpu=%s prefetch_wait_us=%.1f prefetch_hit=%s",
                layer.rank if hasattr(layer, "rank") else -1,
                layer.layer_idx,
                len(getattr(layer, "lossless_hybrid_active_ranks", [])),
                bound_slot.valid_expert_count,
                bound_slot.source_from_npu,
                bound_slot.source_from_cpu,
                float(manager.prefetch_wait_us[int(layer.layer_idx)]),
                int(manager.prefetch_hit[int(layer.layer_idx)]),
            )
        if profile_timing:
            prefetch_dev_ms = manager._prefetch_device_ms(next_prefetch_timing)
            prefetch_npu_dev_ms = _elapsed_ms(
                next_prefetch_timing.get("npu_start_event"),
                next_prefetch_timing.get("npu_end_event"))
            prefetch_cpu_dev_ms = _elapsed_ms(
                next_prefetch_timing.get("cpu_start_event"),
                next_prefetch_timing.get("cpu_end_event"))
            prefetch_cpu_pack_dev_ms = _elapsed_ms(
                next_prefetch_timing.get("cpu_pack_start_event"),
                next_prefetch_timing.get("cpu_pack_end_event"))
            compute_dev_ms = _elapsed_ms(compute_start_event, compute_end_event)
            ready_wait_dev_ms = _elapsed_ms(
                bind_timing.get("ready_wait_start_event"),
                bind_timing.get("ready_wait_end_event"))
            logger.info(
                "Mode3 timing single-dispatch: rank=%s layer=%s stage=%s slot=%s valid_experts=%s source_from_npu=%s source_from_cpu=%s bind_wait_us=%.1f bind_cpu_fill_us=%.1f bind_wait_mode=%s ready_wait_dev_ms=%.3f prefetch_next_layer=%s prefetch_slot=%s prefetch_source_from_cpu=%s prefetch_cpu_path=%s prefetch_cpu_w13_pinned=%s prefetch_cpu_w2_pinned=%s prefetch_cpu_w13_contig=%s prefetch_cpu_w2_contig=%s prefetch_submit_us=%.1f submit_accounted_us=%.1f submit_event_alloc_us=%.1f submit_stream_wait_us=%.1f submit_prefetch_wait_stream_us=%.1f submit_start_event_record_us=%.1f submit_populate_us=%.1f submit_order_us=%.1f submit_assign_us=%.1f submit_layer_local_check_us=%.1f submit_npu_us=%.1f submit_cpu_us=%.1f submit_cpu_direct_async_us=%.1f submit_cpu_stage_async_us=%.1f submit_plan_log_us=%.1f submit_expert_map_us=%.1f submit_expert_map_cache_hit=%s submit_dispatch_cache_us=%.1f submit_slot_state_us=%.1f submit_post_cpu_wait_us=%.1f submit_ready_record_us=%.1f prefetch_dev_ms=%.3f prefetch_npu_dev_ms=%.3f prefetch_cpu_dev_ms=%.3f prefetch_cpu_pack_dev_ms=%.3f current_compute_wall_ms=%.3f current_compute_dev_ms=%.3f remap_wall_ms=%.3f remap_dev_ms=%.3f token_dispatch_wall_ms=%.3f token_dispatch_dev_ms=%.3f group_wall_ms=%.3f mlp_wall_ms=%.3f mlp_dev_ms=%.3f token_combine_wall_ms=%.3f token_combine_dev_ms=%.3f prefetch_minus_compute_dev_ms=%.3f tokens=%s dispatched_rows=%s active_rows=%s owned_per_rank=%s dispatch_num_experts=%s expert_token_nums_type=%s group_list_type=%s",
                layer.rank if hasattr(layer, "rank") else -1,
                layer.layer_idx,
                active_rank_count,
                int(bind_timing.get("slot_id", -1)),
                bound_slot.valid_expert_count,
                bound_slot.source_from_npu,
                bound_slot.source_from_cpu,
                float(bind_timing.get("wait_us", -1.0)),
                float(bind_timing.get("cpu_fill_us", -1.0)),
                bind_timing.get("wait_mode", "unknown"),
                ready_wait_dev_ms,
                next_prefetch_timing.get("layer_idx", -1),
                next_prefetch_timing.get("slot_id", -1),
                next_prefetch_timing.get("source_from_cpu", -1),
                next_prefetch_timing.get("cpu_path", "unknown"),
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
                _timing_float(next_prefetch_timing,
                              "submit_layer_local_check_us"),
                _timing_float(next_prefetch_timing, "submit_npu_us"),
                _timing_float(next_prefetch_timing, "submit_cpu_us"),
                _timing_float(next_prefetch_timing,
                              "submit_cpu_direct_async_us"),
                _timing_float(next_prefetch_timing,
                              "submit_cpu_stage_async_us"),
                _timing_float(next_prefetch_timing, "submit_plan_log_us"),
                _timing_float(next_prefetch_timing, "submit_expert_map_us"),
                int(_timing_float(next_prefetch_timing,
                                  "expert_map_cache_hit")),
                _timing_float(next_prefetch_timing,
                              "submit_dispatch_cache_us"),
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
                _elapsed_ms(remap_start_event, remap_end_event),
                dispatch_wall_ms,
                _elapsed_ms(dispatch_start_event, dispatch_end_event),
                group_wall_ms,
                mlp_wall_ms,
                _elapsed_ms(mlp_start_event, mlp_end_event),
                combine_wall_ms,
                _elapsed_ms(combine_start_event, combine_end_event),
                prefetch_dev_ms - compute_dev_ms
                if prefetch_dev_ms >= 0 and compute_dev_ms >= 0 else -1.0,
                int(x.shape[0]) if hasattr(x, "shape") else -1,
                dispatched_rows,
                dispatched_active_rows,
                owned_per_rank,
                dispatch_num_experts,
                manager.expert_token_nums_type,
                dispatched_group_list_type,
            )
        return final_hidden_states

    def _execute_mode2_single_dispatch_hybrid(
            self,
            layer: torch.nn.Module,
            x: torch.Tensor,
            logical_topk_ids: torch.Tensor,
            topk_weights: torch.Tensor,
            row_idx: torch.Tensor,
            wave_plans: list[dict[str, Any]],
            shared_experts: Optional[Any],
            prepared_mc2_mask: Optional[torch.Tensor],
            enable_force_load_balance: bool,
            kwargs: dict[str, Any],
            ids_already_runtime_remapped: bool,
            is_dummy: bool) -> torch.Tensor:
        moe_comm_method = get_forward_context().moe_comm_method
        token_dispatcher = getattr(moe_comm_method, "token_dispatcher", None)
        if token_dispatcher is None:
            raise RuntimeError("Missing MC2 token dispatcher for hybrid mode2 path.")
        local_rank_idx = int(
            getattr(layer, "lossless_hybrid_active_rank_index", -1))
        rank_owned = getattr(layer, "lossless_hybrid_rank_owned_expert_ids", None)
        if (rank_owned is None or local_rank_idx < 0
                or local_rank_idx >= len(rank_owned)):
            raise RuntimeError(
                "Invalid hybrid rank ownership state for mode2 single-dispatch "
                f"path at layer={getattr(layer, 'layer_idx', -1)}.")
        active_rank_count = len(getattr(layer, "lossless_hybrid_active_ranks", []))
        local_owned_expert_ids = [int(x) for x in rank_owned[local_rank_idx]]
        owned_per_rank = len(local_owned_expert_ids)
        if active_rank_count <= 0 or owned_per_rank <= 0:
            raise RuntimeError(
                "Invalid hybrid mode2 dispatch shape: "
                f"active_rank_count={active_rank_count} "
                f"owned_per_rank={owned_per_rank}")
        dispatch_log2phy, dispatch_num_experts = \
            self._get_dispatch_log2phy_for_layer(
                layer,
                device=logical_topk_ids.device,
                rank_owned=rank_owned,
                active_rank_count=active_rank_count,
                owned_per_rank=owned_per_rank,
            )
        dispatch_topk_ids = (
            logical_topk_ids if ids_already_runtime_remapped else
            dispatch_log2phy[logical_topk_ids])
        if torch.any(dispatch_topk_ids < 0):
            invalid_count = int(torch.count_nonzero(dispatch_topk_ids < 0).item())
            raise RuntimeError(
                "Hybrid mode2 single-dispatch saw experts outside the active "
                f"rank ownership map at layer={getattr(layer, 'layer_idx', -1)}: "
                f"invalid_count={invalid_count}")
        if enable_force_load_balance and not self.use_aclgraph:
            dispatch_topk_ids = (
                torch.arange(dispatch_topk_ids.numel(),
                             device=dispatch_topk_ids.device)
                % dispatch_num_experts).to(torch.int32).reshape(
                    dispatch_topk_ids.shape)

        old_dispatch_num_experts = int(getattr(token_dispatcher, "num_experts", 0))
        token_dispatcher.num_experts = dispatch_num_experts
        dispatch_results = None
        try:
            dispatch_results = token_dispatcher.token_dispatch(
                hidden_states=x,
                topk_weights=topk_weights,
                topk_ids=dispatch_topk_ids,
                row_idx=row_idx,
                expert_map=layer.expert_map,
                log2phy=None,
                global_redundant_expert_num=kwargs.get(
                    "global_redundant_expert_num", 0),
                shared_experts=shared_experts,
                quantized_x_for_share=kwargs.get("quantized_x_for_share"),
                dynamic_scale_for_share=kwargs.get("dynamic_scale_for_share"),
                mc2_mask=prepared_mc2_mask,
                apply_router_weight_on_input=False,
                with_quant=False)
            dispatched_hidden_states = dispatch_results["hidden_states"]
            dispatched_group_list = dispatch_results["group_list"]
            dispatched_group_list_type = int(dispatch_results["group_list_type"])
            dispatched_group_counts = self._group_list_to_counts(
                dispatched_group_list,
                dispatched_group_list_type,
                total_tokens=int(dispatched_hidden_states.shape[0]),
            ).to(dtype=torch.long)
            if int(dispatched_group_counts.numel()) != int(owned_per_rank):
                raise RuntimeError(
                    "Hybrid mode2 single-dispatch expert group size mismatch: "
                    f"expected_experts={owned_per_rank} "
                    f"actual_experts={int(dispatched_group_counts.numel())} "
                    f"group_list_type={dispatched_group_list_type}")
            dispatch_offsets = self._counts_to_offsets(dispatched_group_counts)
            dispatched_active_rows = int(dispatch_offsets[-1].item())
            if dispatched_active_rows > int(dispatched_hidden_states.shape[0]):
                raise RuntimeError(
                    "Hybrid mode2 single-dispatch local token count mismatch: "
                    f"expected<={int(dispatched_hidden_states.shape[0])} "
                    f"active_rows={dispatched_active_rows} "
                    f"actual={int(dispatched_hidden_states.shape[0])} "
                    f"group_list_len={int(dispatched_group_list.numel())} "
                    f"group_list_type={dispatched_group_list_type} "
                    f"group_list_sum={int(dispatched_group_list.to(dtype=torch.long).sum().item())} "
                    f"group_count_sum={int(dispatched_group_counts.sum().item())}")

            dispatched_output = torch.zeros_like(dispatched_hidden_states)
            local_owned_index_by_expert = {
                int(expert_id): local_idx
                for local_idx, expert_id in enumerate(local_owned_expert_ids)
            }
            local_wave_counts = []
            local_cpu_miss = 0
            local_resident_hits = 0

            for wave_idx, wave_plan in enumerate(wave_plans):
                effective_token_mask = wave_plan["token_mask"]
                if prepared_mc2_mask is not None:
                    mc2_active_mask = prepared_mc2_mask.to(
                        device=effective_token_mask.device, dtype=torch.bool)
                    effective_token_mask = effective_token_mask & mc2_active_mask
                    if (is_dummy and not torch.any(effective_token_mask)
                            and torch.any(wave_plan["token_mask"])):
                        effective_token_mask = wave_plan["token_mask"]
                if (effective_token_mask.numel() > 0
                        and torch.any(effective_token_mask)):
                    local_wave_counts.append(
                        int(torch.count_nonzero(effective_token_mask).item()))
                else:
                    local_wave_counts.append(0)

                if 0 <= local_rank_idx < len(wave_plan["rank_resident"]):
                    target_resident = list(wave_plan["rank_resident"][local_rank_idx])
                    current_resident = list(
                        getattr(layer, "lossless_hybrid_resident_expert_ids", []))
                    current_resident_set = set(current_resident)
                    local_cpu_miss += len(
                        [expert_id for expert_id in target_resident
                         if expert_id not in current_resident_set])
                    local_resident_hits += len(
                        [expert_id for expert_id in target_resident
                         if expert_id in current_resident_set])
                    layer.materialize_hybrid_resident_experts(target_resident)
                else:
                    target_resident = []

                wave_group_by_rank = wave_plan.get("rank_wave_groups", [])
                local_wave_group = (
                    [int(x) for x in wave_group_by_rank[local_rank_idx]]
                    if 0 <= local_rank_idx < len(wave_group_by_rank) else [])
                local_wave_group_set = set(local_wave_group)
                resident_capacity = int(layer.lossless_hybrid_resident_capacity)
                resident_group_counts = torch.zeros((resident_capacity, ),
                                                   dtype=torch.int64,
                                                   device=dispatched_hidden_states.device)
                hidden_segments: list[torch.Tensor] = []
                segment_targets: list[tuple[int, int, int]] = []
                for slot_idx, expert_id in enumerate(target_resident[:resident_capacity]):
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

                if layer.layer_idx == 0:
                    logger.info(
                        "Hybrid MC2 single-dispatch wave: rank=%s stage=%s wave=%s active_local_experts=%s dispatched_tokens=%s resident_capacity=%s",
                        layer.rank if hasattr(layer, "rank") else -1,
                        len(getattr(layer, "lossless_hybrid_active_ranks", [])),
                        wave_idx,
                        len(local_wave_group),
                        int(resident_group_counts.sum().item()),
                        resident_capacity,
                    )

                if not hidden_segments:
                    continue

                wave_hidden_states = torch.cat(hidden_segments, dim=0)
                runtime_w13_weight = getattr(layer, "runtime_w13_weight", None)
                runtime_w2_weight = getattr(layer, "runtime_w2_weight", None)
                if runtime_w13_weight is None or runtime_w2_weight is None:
                    raise RuntimeError(
                        "Missing runtime expert weights for hybrid mode2 "
                        f"single-dispatch path at layer={getattr(layer, 'layer_idx', -1)}.")
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
                    need_trans=True)
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
        if layer.layer_idx == 0 and wave_plans:
            logger.info(
                "Hybrid MoE single-dispatch execution: rank=%s layer=%s stage=%s waves=%s resident_hits=%s cpu_miss=%s local_wave_tokens=%s resident=%s dispatch_experts=%s",
                layer.rank if hasattr(layer, "rank") else -1,
                layer.layer_idx,
                len(getattr(layer, "lossless_hybrid_active_ranks", [])),
                len(wave_plans),
                local_resident_hits,
                local_cpu_miss,
                local_wave_counts[:8],
                getattr(layer, "lossless_hybrid_resident_expert_ids", [])[:8],
                dispatch_num_experts,
            )
        return final_hidden_states

    def _execute_single_wave(
            self,
            layer: torch.nn.Module,
            hidden_states: torch.Tensor,
            logical_topk_ids: torch.Tensor,
            topk_weights: torch.Tensor,
            row_idx: torch.Tensor,
            global_num_experts: int,
            shared_experts: Optional[Any],
            log2phy: Optional[Any],
            mc2_mask: Optional[torch.Tensor],
            enable_force_load_balance: bool,
            kwargs: dict[str, Any],
            ids_already_runtime_remapped: bool = False) -> torch.Tensor:
        moe_comm_method = get_forward_context().moe_comm_method
        topk_ids = logical_topk_ids
        if (log2phy is not None
                and not ids_already_runtime_remapped
                and moe_comm_method.__class__.__name__ != "AlltoAllCommImpl"):
            topk_ids = log2phy[topk_ids]
            invalid_mask = topk_ids < 0
            if torch.any(invalid_mask):
                raise RuntimeError(
                    "Invalid remapped topk_ids encountered before fused "
                    f"MoE execution at layer={getattr(layer, 'layer_idx', -1)}: "
                    f"invalid_count={int(invalid_mask.sum().item())}")
        if enable_force_load_balance and not self.use_aclgraph:
            topk_ids = (
                torch.arange(topk_ids.numel(), device=topk_ids.device)
                % global_num_experts).to(torch.int32).reshape(topk_ids.shape)

        runtime_w13_weight = getattr(layer, "runtime_w13_weight", None)
        runtime_w2_weight = getattr(layer, "runtime_w2_weight", None)
        active_w13_weight = (runtime_w13_weight if runtime_w13_weight is not None
                             else _active_moe_weight_view(layer,
                                                          layer.w13_weight))
        active_w2_weight = (runtime_w2_weight if runtime_w2_weight is not None
                            else _active_moe_weight_view(layer,
                                                         layer.w2_weight))
        if (getattr(layer, "elastic_execution_mode", 0) == 1
                and getattr(layer, "lossless_mode1_native_parity_ready", False)
                and getattr(layer, "layer_idx", -1) == 0
                and not getattr(layer, "_mode1_execute_single_wave_logged", False)):
            logger.info(
                "Mode1 execute_single_wave: layer=%s active_local=%s local_num=%s "
                "moe_config_local=%s loaded_capacity=%s runtime_w13_shape=%s "
                "runtime_w2_shape=%s active_w13_shape=%s active_w2_shape=%s "
                "topk_shape=%s log2phy=%s already_remapped=%s comm_impl=%s",
                getattr(layer, "layer_idx", -1),
                int(getattr(layer, "active_local_num_experts", -1)),
                int(getattr(layer, "local_num_experts", -1)),
                int(getattr(getattr(layer, "moe_config", None),
                            "num_local_experts", -1)),
                int(getattr(layer, "loaded_weight_capacity", -1)),
                None if runtime_w13_weight is None else tuple(runtime_w13_weight.shape),
                None if runtime_w2_weight is None else tuple(runtime_w2_weight.shape),
                tuple(active_w13_weight.shape),
                tuple(active_w2_weight.shape),
                tuple(topk_ids.shape),
                log2phy is not None,
                ids_already_runtime_remapped,
                moe_comm_method.__class__.__name__,
            )
            layer._mode1_execute_single_wave_logged = True
        return moe_comm_method.fused_experts(
            hidden_states=hidden_states,
            w1=active_w13_weight,
            w2=active_w2_weight,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            row_idx=row_idx,
            global_num_experts=global_num_experts,
            expert_map=layer.expert_map,
            shared_experts=shared_experts,
            quantized_x_for_share=kwargs.get("quantized_x_for_share"),
            dynamic_scale_for_share=kwargs.get("dynamic_scale_for_share"),
            log2phy=log2phy,
            global_redundant_expert_num=kwargs.get(
                "global_redundant_expert_num", 0),
            need_trans=True,
            dynamic_eplb=self.dynamic_eplb,
            mc2_mask=mc2_mask)

    def process_weights_after_loading(self, layer):
        pass  # debug log removed
        super(UnquantizedFusedMoEMethod,
              self).process_weights_after_loading(layer)
        pass  # debug log removed
        # Preserve the original Parameter objects. Rebinding them here breaks
        # the storage that weight_loader has already populated.
        layer.w13_weight.data = self._maybe_pad_weight(layer.w13_weight.data)
        layer.w2_weight.data = self._maybe_pad_weight(layer.w2_weight.data)
        if layer.w13_weight.device.type == "cpu":
            pass  # CPU-staged reload finalizes tensor format after the buffers are moved back to NPU.
        elif not is_310p():
            mode1_reload = (
                getattr(layer, "elastic_moe_mode", "lossy") == "lossless"
                and int(getattr(layer, "elastic_execution_mode", 0)) == 1)
            if mode1_reload:
                _trim_npu_allocator_for_mode1_reload(layer,
                                                     "before_w13_format_cast")
            layer.w13_weight.data = _maybe_npu_format_cast_for_mode1_reload(
                layer.w13_weight.data,
                ACL_FORMAT_FRACTAL_NZ,
                layer=layer,
                weight_name="w13",
                mode1_reload=mode1_reload)
            if mode1_reload:
                _trim_npu_allocator_for_mode1_reload(layer,
                                                     "before_w2_format_cast")
            layer.w2_weight.data = _maybe_npu_format_cast_for_mode1_reload(
                layer.w2_weight.data,
                ACL_FORMAT_FRACTAL_NZ,
                layer=layer,
                weight_name="w2",
                mode1_reload=mode1_reload)
            if mode1_reload:
                _trim_npu_allocator_for_mode1_reload(layer,
                                                     "after_weight_format_cast")
        pass  # debug log removed
        if getattr(layer, "lossless_zero_redundancy_preallocated_loaded", False):
            valid_rows = int(getattr(layer, "loaded_local_num_experts", 0))
            if int(layer.w13_weight.shape[0]) > valid_rows:
                layer.w13_weight.data[valid_rows:].zero_()
            if int(layer.w2_weight.shape[0]) > valid_rows:
                layer.w2_weight.data[valid_rows:].zero_()
        if getattr(layer, "elastic_moe_mode", "lossy") == "lossless":
            pass  # debug log removed
            layer.activate_lossless_primary_experts()
            pass  # debug log removed

    def invalidate_lossless_runtime_state_for_reload(
            self,
            layer: Optional[torch.nn.Module] = None,
            reason: str = "reload") -> None:
        target = layer if layer is not None else self
        if getattr(target, "elastic_moe_mode", None) != "lossless":
            return False
        had_runtime = (getattr(target, "runtime_w13_weight", None) is not None
                       or getattr(target, "runtime_w2_weight", None) is not None
                       or getattr(target, "runtime_w13_buffer", None) is not None
                       or getattr(target, "runtime_w2_buffer", None) is not None
                       or getattr(target, "lossless_cpu_w13_weight", None) is not None
                       or getattr(target, "lossless_cpu_w2_weight", None) is not None)
        if getattr(target, "layer_idx", -1) == 0 and had_runtime:
            pass  # debug log removed
        target.runtime_w13_weight = None
        target.runtime_w2_weight = None
        target.runtime_w13_buffer = None
        target.runtime_w2_buffer = None
        target.runtime_weight_capacity = 0
        target.lossless_runtime_activated = False
        return had_runtime

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
            debug_info["pre_topk_min"] = (
                int(flat_topk_ids.min().item())
                if flat_topk_ids.numel() > 0 else -1)
            debug_info["pre_topk_max"] = (
                int(flat_topk_ids.max().item())
                if flat_topk_ids.numel() > 0 else -1)
            debug_info["pre_topk_unique"] = int(
                torch.unique(flat_topk_ids).numel()) if flat_topk_ids.numel() > 0 else 0
        logical_topk_ids = topk_ids
        dummy_waste_timing = getattr(get_forward_context(),
                                     "dummy_waste_timing", None)
        dummy_profile_markers_enabled = (
            bool(is_dummy)
            and _env_flag("VLLM_ASCEND_DUMMY_WASTE_PROFILE_MARKERS", "0"))
        dummy_selection_stats_enabled = _env_flag(
            "VLLM_ASCEND_DUMMY_WASTE_SELECTION_STATS", "0")
        dummy_selection_enabled = (
            isinstance(dummy_waste_timing, dict)
            and _env_flag("VLLM_ASCEND_DUMMY_WASTE_TIMING", "0")
            and dummy_selection_stats_enabled)
        selected_topk_count = 0
        selected_token_count = 0
        selected_expert_count = 0
        if dummy_selection_enabled:
            try:
                local_expert_map = expert_map
                if local_expert_map is None:
                    local_expert_map = getattr(layer, "expert_map", None)
                if local_expert_map is not None and local_expert_map.numel() > 0:
                    if local_expert_map.device != logical_topk_ids.device:
                        local_expert_map = local_expert_map.to(
                            device=logical_topk_ids.device)
                    valid_ids = (
                        (logical_topk_ids >= 0)
                        & (logical_topk_ids < int(local_expert_map.numel())))
                    safe_ids = logical_topk_ids.clamp(
                        min=0, max=int(local_expert_map.numel()) - 1)
                    local_slots = local_expert_map[safe_ids]
                    selected_mask = valid_ids & (local_slots >= 0)
                    selected_topk_count = int(
                        torch.count_nonzero(selected_mask).item())
                    if selected_mask.ndim > 1:
                        selected_token_count = int(
                            torch.count_nonzero(
                                torch.any(selected_mask, dim=-1)).item())
                    else:
                        selected_token_count = selected_topk_count
                    if selected_topk_count > 0:
                        selected_expert_count = int(
                            torch.unique(logical_topk_ids[selected_mask]).numel()
                            .item())
            except Exception:
                logger.exception(
                    "Failed to compute dummy local MoE selection stats")
            if isinstance(dummy_waste_timing, dict):
                dummy_waste_timing["_current_moe_selected_topk"] = (
                    selected_topk_count)
                dummy_waste_timing["_current_moe_selected_tokens"] = (
                    selected_token_count)
                dummy_waste_timing["_current_moe_selected_experts"] = (
                    selected_expert_count)
        ids_already_runtime_remapped = False
        if (log2phy is not None
                and moe_comm_method.__class__.__name__ != "AlltoAllCommImpl"):
            # Keep mode=1/mode=2 shrink runtime aligned with common/native
            # semantics: runtime log2phy is applied exactly once to the
            # logical expert ids selected by the router. Downstream wave/
            # single-wave helpers must not remap the already-dense ids again.
            topk_ids = log2phy[topk_ids]
            ids_already_runtime_remapped = True
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
        try:
            moe_stats.record_pattern(
                layer_idx=int(layer_idx),
                topk_ids=logical_topk_ids,
                num_experts=int(global_num_experts),
            )
        except Exception:
            logger.exception("Failed to record MoE pattern stats")
        current_ep_size = int(
            getattr(getattr(layer, "moe_parallel_config", None), "ep_size",
                    0))
        initial_ep_size = 0
        if hasattr(layer, "_get_elastic_initial_ep_size"):
            try:
                initial_ep_size = int(layer._get_elastic_initial_ep_size())
            except Exception:
                initial_ep_size = 0
        post_shrink_stage = (current_ep_size > 0 and initial_ep_size > 0
                             and current_ep_size < initial_ep_size)
        if (getattr(layer, "elastic_execution_mode", 0) == 3
                and post_shrink_stage
                and not getattr(layer, "lossless_hybrid_active", False)
                and layer.layer_idx == 0
                and not getattr(layer, "_mode3_hybrid_inactive_logged", False)):
            logger.warning(
                "Mode3 forward reached fused MoE without active hybrid state: "
                "layer=%s is_dummy=%s ep_size=%s initial_ep_size=%s "
                "active_local=%s num_experts=%s "
                "runtime_log2phy=%s elastic_runtime_log2phy=%s "
                "loaded_local=%s cpu_only_local=%s",
                layer.layer_idx,
                is_dummy,
                current_ep_size,
                initial_ep_size,
                int(getattr(layer, "active_local_num_experts", 0)),
                int(getattr(layer, "num_experts", 0)),
                getattr(layer, "log2phy", None) is not None,
                getattr(layer, "elastic_runtime_log2phy", None) is not None,
                int(getattr(layer, "loaded_local_num_experts", 0)),
                len(getattr(layer, "lossless_hybrid_cpu_only_expert_ids", [])),
            )
            layer._mode3_hybrid_inactive_logged = True
        if getattr(layer, "lossless_hybrid_active", False):
            if self._should_use_mode3_single_rank_allgather_path(
                    layer=layer,
                    shared_experts=shared_experts,
                    is_dummy=is_dummy):
                return self._execute_mode3_single_rank_allgather_hybrid(
                    layer=layer,
                    x=x,
                    logical_topk_ids=logical_topk_ids,
                    topk_weights=topk_weights,
                    row_idx=row_idx,
                    shared_experts=shared_experts,
                    enable_force_load_balance=enable_force_load_balance,
                    kwargs=kwargs)
            if self._should_use_mode3_cross_layer_buffer_path(
                    layer=layer,
                    shared_experts=shared_experts,
                    is_dummy=is_dummy):
                try:
                    return self._execute_mode3_single_dispatch_hybrid(
                        layer=layer,
                        x=x,
                        logical_topk_ids=logical_topk_ids,
                        topk_weights=topk_weights,
                        row_idx=row_idx,
                        shared_experts=shared_experts,
                        enable_force_load_balance=enable_force_load_balance,
                        kwargs=kwargs)
                except RuntimeError as err:
                    err_msg = str(err)
                    fatal_mode3_markers = (
                        "AclrtSynchronizeStreamWithTimeout",
                        "copy_stream",
                        "507015",
                        "EZ9999",
                        "ERR00100",
                        "aicore",
                        "AICORE",
                        "aclnnIndex",
                    )
                    if any(marker in err_msg for marker in fatal_mode3_markers):
                        raise
                    logger.warning(
                        "Mode3 fallback to mode2 wave path: layer=%s stage=%s reason=%s",
                        layer.layer_idx,
                        len(getattr(layer, "lossless_hybrid_active_ranks", [])),
                        err,
                    )
            elif (getattr(layer, "elastic_execution_mode", 0) == 3
                  and layer.layer_idx == 0
                  and not getattr(layer, "_mode3_gate_reject_logged", False)):
                forward_context = get_forward_context()
                moe_comm_method = getattr(forward_context, "moe_comm_method", None)
                logger.warning(
                    "Mode3 gate rejected current forward: layer=%s stage=%s is_dummy=%s "
                    "shared_experts=%s moe_comm_type=%s selected_moe_comm_type=%s "
                    "moe_comm_method=%s token_dispatcher=%s hybrid_active=%s",
                    layer.layer_idx,
                    len(getattr(layer, "lossless_hybrid_active_ranks", [])),
                    is_dummy,
                    shared_experts is not None,
                    getattr(forward_context, "moe_comm_type", None),
                    getattr(forward_context, "selected_moe_comm_type", None),
                    moe_comm_method.__class__.__name__
                    if moe_comm_method is not None else None,
                    getattr(moe_comm_method, "token_dispatcher", None)
                    is not None,
                    getattr(layer, "lossless_hybrid_active", False),
                )
                layer._mode3_gate_reject_logged = True
            wave_plans = layer._plan_lossless_hybrid_rank_waves(logical_topk_ids)
            if wave_plans:
                final_hidden_states = torch.zeros_like(x)
                local_rank_idx = int(
                    getattr(layer, "lossless_hybrid_active_rank_index", -1))
                local_wave_counts = []
                local_cpu_miss = 0
                local_resident_hits = 0
                prepared_mc2_mask = getattr(moe_comm_method, "mc2_mask", None)
                use_dense_mc2_waves = (
                    getattr(get_forward_context(), "moe_comm_type", None)
                    == MoECommType.MC2)
                has_real_mc2_tokens = True
                if prepared_mc2_mask is not None and prepared_mc2_mask.numel() > 0:
                    has_real_mc2_tokens = bool(
                        torch.any(prepared_mc2_mask).item())
                sync_mc2_multi_wave = use_dense_mc2_waves and len(wave_plans) > 1
                grouped_dense_mc2_multi_wave = (
                    sync_mc2_multi_wave
                    and not self._should_disable_grouped_dense_mc2_multi_wave(
                        layer))
                active_rank_count = len(
                    getattr(layer, "lossless_hybrid_active_ranks", []))
                metadata_group = None
                if active_rank_count > 1:
                    metadata_group = getattr(layer.ep_group, "cpu_group",
                                             layer.ep_group.device_group)
                if (layer.layer_idx == 0 and sync_mc2_multi_wave
                        and not grouped_dense_mc2_multi_wave):
                    logger.info(
                        "Hybrid MC2 mode2 swap path using per-wave single-pack MC2: rank=%s stage=%s waves=%s mode=%s",
                        layer.rank if hasattr(layer, "rank") else -1,
                        active_rank_count,
                        len(wave_plans),
                        getattr(layer, "elastic_execution_mode", -1),
                    )
                if self._should_use_mode2_single_dispatch_hybrid_path(
                        layer=layer,
                        wave_plans=wave_plans,
                        use_dense_mc2_waves=use_dense_mc2_waves,
                        shared_experts=shared_experts) and not is_dummy and has_real_mc2_tokens:
                    if layer.layer_idx == 0:
                        logger.info(
                            "Hybrid MC2 single-dispatch mode enabled: rank=%s stage=%s waves=%s resident_capacity=%s mode=%s",
                            layer.rank if hasattr(layer, "rank") else -1,
                            active_rank_count,
                            len(wave_plans),
                            int(getattr(layer, "lossless_hybrid_resident_capacity", 0)),
                            getattr(layer, "elastic_execution_mode", -1),
                        )
                    try:
                        return self._execute_mode2_single_dispatch_hybrid(
                            layer=layer,
                            x=x,
                            logical_topk_ids=logical_topk_ids,
                            topk_weights=topk_weights,
                            row_idx=row_idx,
                            wave_plans=wave_plans,
                            shared_experts=shared_experts,
                            prepared_mc2_mask=prepared_mc2_mask,
                            enable_force_load_balance=enable_force_load_balance,
                            kwargs=kwargs,
                            ids_already_runtime_remapped=
                            ids_already_runtime_remapped,
                            is_dummy=is_dummy)
                    except RuntimeError as err:
                        if "Hybrid mode2 single-dispatch" not in str(err):
                            raise
                        logger.warning(
                            "Hybrid MC2 single-dispatch fallback to per-wave MC2 path: layer=%s stage=%s reason=%s",
                            layer.layer_idx,
                            active_rank_count,
                            err,
                        )
                for wave_idx, wave_plan in enumerate(wave_plans):
                    placeholder_only = False
                    wave_mc2_mask = None
                    grouped_dense_mc2_inputs = None
                    grouped_dense_mc2_topks: list[int] = []
                    global_wave_active_count = 0
                    if use_dense_mc2_waves:
                        if not wave_plan["wave_active_expert_ids"]:
                            local_wave_counts.append(0)
                            continue
                        effective_token_mask = wave_plan["token_mask"]
                        if prepared_mc2_mask is not None:
                            mc2_active_mask = prepared_mc2_mask.to(
                                device=effective_token_mask.device,
                                dtype=torch.bool)
                            effective_token_mask = (
                                effective_token_mask & mc2_active_mask)
                            if (is_dummy and not torch.any(effective_token_mask)
                                    and torch.any(wave_plan["token_mask"])):
                                effective_token_mask = wave_plan["token_mask"]
                        if (effective_token_mask.numel() > 0
                                and torch.any(effective_token_mask)):
                            active_token_count = int(
                                torch.count_nonzero(effective_token_mask).item())
                        else:
                            active_token_count = 0
                        local_wave_counts.append(active_token_count)
                        global_wave_active_count = int(active_token_count)
                        if metadata_group is not None:
                            gathered_wave_counts: list[Optional[int]] = [
                                None
                            ] * active_rank_count
                            torch.distributed.all_gather_object(
                                gathered_wave_counts,
                                int(active_token_count),
                                group=metadata_group)
                            global_wave_active_count = max(
                                int(value) for value in gathered_wave_counts
                                if value is not None)
                        if global_wave_active_count <= 0:
                            continue
                        if grouped_dense_mc2_multi_wave:
                            local_group_topks = []
                            if active_token_count > 0:
                                local_group_topks = sorted({
                                    int(value) for value in wave_plan["topk_mask"][
                                        effective_token_mask].sum(dim=1).tolist()
                                    if int(value) > 0
                                }, reverse=True)
                            if metadata_group is not None:
                                gathered_group_topks: list[Optional[list[int]]] = [
                                    None
                                ] * active_rank_count
                                torch.distributed.all_gather_object(
                                    gathered_group_topks,
                                    local_group_topks,
                                    group=metadata_group)
                                grouped_dense_mc2_topks = sorted({
                                    int(value)
                                    for group_values in gathered_group_topks
                                    if group_values is not None
                                    for value in group_values
                                    if int(value) > 0
                                }, reverse=True)
                            else:
                                grouped_dense_mc2_topks = local_group_topks
                            if grouped_dense_mc2_topks:
                                grouped_dense_mc2_inputs = (
                                    self._build_grouped_mc2_wave_inputs(
                                        hidden_states=x,
                                        logical_topk_ids=logical_topk_ids,
                                        topk_weights=topk_weights,
                                        token_mask=effective_token_mask,
                                        topk_mask=wave_plan["topk_mask"],
                                        active_topk_values=grouped_dense_mc2_topks,
                                        fallback_expert_id=int(
                                            wave_plan["wave_active_expert_ids"][0]),
                                        launch_topk=int(
                                            wave_plan.get("wave_topk", 0)),
                                    ))
                            else:
                                grouped_dense_mc2_inputs = []
                        else:
                            (token_indices, hidden_states, wave_ids, wave_weights,
                             wave_row_idx, wave_mc2_mask,
                             active_token_count) = self._build_padded_mc2_wave_inputs(
                                 hidden_states=x,
                                 logical_topk_ids=logical_topk_ids,
                                 topk_weights=topk_weights,
                                 token_mask=effective_token_mask,
                                 topk_mask=wave_plan["topk_mask"],
                                 fallback_expert_id=int(
                                     wave_plan["wave_active_expert_ids"][0]),
                                 forced_active_topk=int(
                                     wave_plan.get("wave_topk", 0)))
                            placeholder_only = (active_token_count == 0)
                    else:
                        token_indices, wave_ids, wave_weights, wave_row_idx = (
                            self._compact_wave_topk(logical_topk_ids,
                                                    topk_weights,
                                                    wave_plan["token_mask"],
                                                    wave_plan["topk_mask"]))
                    if 0 <= local_rank_idx < len(wave_plan["rank_resident"]):
                        target_resident = list(
                            wave_plan["rank_resident"][local_rank_idx])
                        current_resident = list(
                            getattr(layer, "lossless_hybrid_resident_expert_ids",
                                    []))
                        local_cpu_miss += len(
                            [expert_id for expert_id in target_resident
                             if expert_id not in set(current_resident)])
                        local_resident_hits += len(
                            [expert_id for expert_id in target_resident
                             if expert_id in set(current_resident)])
                        layer.materialize_hybrid_resident_experts(target_resident)
                    logical_num_experts = int(layer.elastic_original_num_experts)
                    wave_log2phy_cpu = torch.full((logical_num_experts, ),
                                                  -1,
                                                  dtype=torch.int32,
                                                  device=logical_topk_ids.device)
                    for expert_id, slot in wave_plan["wave_expert_to_slot"].items():
                        wave_log2phy_cpu[int(expert_id)] = int(slot)
                    if not use_dense_mc2_waves:
                        placeholder_only = (
                            token_indices is None or wave_ids is None
                            or wave_weights is None or wave_row_idx is None)
                        if placeholder_only:
                            if not wave_plan["wave_active_expert_ids"]:
                                local_wave_counts.append(0)
                                continue
                            # Non-MC2 paths still need a non-empty local input
                            # to enter the wave collective. Keep a single
                            # zero-weight placeholder token.
                            placeholder_expert = int(
                                wave_plan["wave_active_expert_ids"][0])
                            hidden_states = x[:1].clone()
                            hidden_states.zero_()
                            wave_ids = torch.full((1, 1),
                                                  placeholder_expert,
                                                  dtype=logical_topk_ids.dtype,
                                                  device=logical_topk_ids.device)
                            wave_weights = torch.zeros((1, 1),
                                                       dtype=topk_weights.dtype,
                                                       device=topk_weights.device)
                            wave_row_idx = return_row_idx(wave_ids, 1)
                            local_wave_counts.append(0)
                        else:
                            hidden_states = x.index_select(0, token_indices)
                            local_wave_counts.append(int(token_indices.numel()))
                    if prepared_mc2_mask is not None and not use_dense_mc2_waves:
                        if placeholder_only:
                            wave_mc2_mask = torch.zeros_like(
                                prepared_mc2_mask[:1])
                        else:
                            wave_mc2_mask = prepared_mc2_mask.index_select(
                                0, token_indices)
                            if wave_mc2_mask.shape[0] != token_indices.shape[0]:
                                raise RuntimeError(
                                    "Hybrid wave mc2_mask shape mismatch at "
                                    f"layer={layer.layer_idx}: "
                                    f"wave_tokens={token_indices.shape[0]} "
                                    f"mask_tokens={wave_mc2_mask.shape[0]}")
                    if use_dense_mc2_waves:
                        if grouped_dense_mc2_inputs is not None:
                            executed_group = False
                            for group_idx, group_pack in enumerate(
                                    grouped_dense_mc2_inputs):
                                (group_indices, group_hidden_states, group_ids,
                                 group_weights, group_row_idx, group_mc2_mask,
                                 group_active_count,
                                 group_placeholder_only) = group_pack
                                global_group_active = int(group_active_count)
                                if metadata_group is not None:
                                    gathered_group_counts: list[Optional[int]] = [
                                        None
                                    ] * active_rank_count
                                    torch.distributed.all_gather_object(
                                        gathered_group_counts,
                                        int(group_active_count),
                                        group=metadata_group)
                                    global_group_active = max(
                                        int(value) for value in gathered_group_counts
                                        if value is not None)
                                if global_group_active <= 0:
                                    continue
                                if layer.layer_idx == 0:
                                    logger.info(
                                        "Hybrid MC2 wave pack: rank=%s stage=%s wave=%s group=%s group_active_topk=%s active_tokens=%s launch_bs=%s wave_topk=%s placeholder_only=%s global_wave_experts=%s",
                                        layer.rank if hasattr(layer, "rank") else -1,
                                        len(getattr(layer, "lossless_hybrid_active_ranks", [])),
                                        wave_idx,
                                        group_idx,
                                        grouped_dense_mc2_topks[group_idx]
                                        if group_idx < len(grouped_dense_mc2_topks)
                                        else -1,
                                        group_active_count,
                                        int(group_hidden_states.shape[0]),
                                        int(group_ids.shape[1]),
                                        group_placeholder_only,
                                        len(wave_plan["wave_active_expert_ids"]),
                                    )
                                wave_output = self._execute_single_wave(
                                    layer=layer,
                                    hidden_states=group_hidden_states,
                                    logical_topk_ids=group_ids,
                                    topk_weights=group_weights,
                                    row_idx=group_row_idx,
                                    global_num_experts=int(
                                        len(layer.lossless_hybrid_active_ranks) *
                                        layer.lossless_hybrid_resident_capacity),
                                    shared_experts=shared_experts,
                                    log2phy=wave_log2phy_cpu,
                                    mc2_mask=group_mc2_mask,
                                    enable_force_load_balance=
                                    enable_force_load_balance,
                                    kwargs=kwargs,
                                    ids_already_runtime_remapped=True)
                                executed_group = True
                                if (not group_placeholder_only
                                        and group_indices is not None
                                        and group_indices.numel() > 0):
                                    final_hidden_states.index_add_(
                                        0, group_indices,
                                        wave_output[:group_active_count])
                                # Grouped MC2 waves reuse the same resident slots
                                # and comm buffers. Drain each group before
                                # launching the next one to avoid overlapping
                                # device work inside a single logical wave.
                                torch.npu.synchronize()
                            if sync_mc2_multi_wave and executed_group:
                                torch.npu.synchronize()
                            continue
                        if layer.layer_idx == 0:
                            logger.info(
                                "Hybrid MC2 wave pack: rank=%s stage=%s wave=%s group=%s active_tokens=%s launch_bs=%s wave_topk=%s placeholder_only=%s global_wave_experts=%s",
                                layer.rank if hasattr(layer, "rank") else -1,
                                len(getattr(layer, "lossless_hybrid_active_ranks", [])),
                                wave_idx,
                                0,
                                global_wave_active_count,
                                int(hidden_states.shape[0]),
                                int(wave_ids.shape[1]),
                                placeholder_only,
                                len(wave_plan["wave_active_expert_ids"]),
                            )
                        wave_output = self._execute_single_wave(
                            layer=layer,
                            hidden_states=hidden_states,
                            logical_topk_ids=wave_ids,
                            topk_weights=wave_weights,
                            row_idx=wave_row_idx,
                            global_num_experts=int(
                                len(layer.lossless_hybrid_active_ranks) *
                                layer.lossless_hybrid_resident_capacity),
                            shared_experts=shared_experts,
                            log2phy=wave_log2phy_cpu,
                            mc2_mask=wave_mc2_mask,
                            enable_force_load_balance=
                            enable_force_load_balance,
                            kwargs=kwargs,
                            ids_already_runtime_remapped=True)
                        if (not placeholder_only and token_indices is not None
                                and token_indices.numel() > 0):
                            final_hidden_states.index_add_(
                                0, token_indices,
                                wave_output[:active_token_count])
                    elif not placeholder_only:
                        wave_output = self._execute_single_wave(
                            layer=layer,
                            hidden_states=hidden_states,
                            logical_topk_ids=wave_ids,
                            topk_weights=wave_weights,
                            row_idx=wave_row_idx,
                            global_num_experts=int(
                                len(layer.lossless_hybrid_active_ranks) *
                                layer.lossless_hybrid_resident_capacity),
                            shared_experts=shared_experts,
                            log2phy=wave_log2phy_cpu,
                            mc2_mask=wave_mc2_mask,
                            enable_force_load_balance=enable_force_load_balance,
                            kwargs=kwargs,
                            ids_already_runtime_remapped=False)
                        final_hidden_states.index_add_(0, token_indices,
                                                       wave_output)
                    # MC2 dispatch/combine stays asynchronous longer than the
                    # AllToAll path. Do not overwrite resident slots for the
                    # next wave until the current wave has fully drained.
                    if sync_mc2_multi_wave:
                        torch.npu.synchronize()
                layer.lossless_hybrid_rank_resident_expert_ids = [
                    list(expert_ids)
                    for expert_ids in wave_plans[-1]["final_rank_resident"]
                ]
                layer.lossless_hybrid_rank_lru = [
                    list(expert_ids)
                    for expert_ids in wave_plans[-1]["final_rank_resident"]
                ]
                if 0 <= local_rank_idx < len(wave_plans[-1]["final_rank_resident"]):
                    if sync_mc2_multi_wave:
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
                        "Hybrid MoE wave execution: rank=%s layer=%s stage=%s waves=%s resident_hits=%s cpu_miss=%s local_wave_tokens=%s resident=%s",
                        layer.rank if hasattr(layer, "rank") else -1,
                        layer.layer_idx,
                        len(getattr(layer, "lossless_hybrid_active_ranks", [])),
                        len(wave_plans),
                        local_resident_hits,
                        local_cpu_miss,
                        local_wave_counts[:8],
                        getattr(layer, "lossless_hybrid_resident_expert_ids",
                                [])[:8],
                    )
                return final_hidden_states

        marker_ctx = nullcontext()
        if dummy_profile_markers_enabled:
            marker_ctx = _dummy_profile_range(
                f"vllm_dummy_moe_compute rank={_profile_rank()} "
                f"layer={layer_idx}")
        with marker_ctx:
            return self._execute_single_wave(
                layer=layer,
                hidden_states=x,
                logical_topk_ids=logical_topk_ids,
                topk_weights=topk_weights,
                row_idx=row_idx,
                global_num_experts=global_num_experts,
                shared_experts=shared_experts,
                log2phy=log2phy,
                mc2_mask=None,
                enable_force_load_balance=enable_force_load_balance,
                kwargs=kwargs,
                ids_already_runtime_remapped=ids_already_runtime_remapped)


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
        elastic_execution_mode = (
            envs_ascend.VLLM_ASCEND_ELASTIC_EXECUTION_MODE)
        use_base_init = (
            elastic_execution_mode == 3
            and _env_flag("VLLM_ASCEND_CUSTOM_MODE3_USE_BASE_INIT", "0"))
        if use_base_init:
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
        else:
            # Avoid constructing the native vLLM FusedMoE first. The base
            # initializer allocates native expert weights, then this custom
            # initializer immediately replaces them with Ascend-formatted weights.
            # That transient allocation distorts KV-cache profiling and can leave
            # enough allocator pressure to OOM when the cache is materialized.
            torch.nn.Module.__init__(self)
        AscendFusedMoE.moe_counter += 1
        self.moe_instance_id = AscendFusedMoE.moe_counter
        self.layer_idx = layer_idx
        if layer_idx == 0:
            logger.info(
                "Custom AscendFusedMoE init path: mode=%s use_base_init=%s",
                elastic_execution_mode,
                int(use_base_init),
            )

        if params_dtype is None:
            params_dtype = torch.get_default_dtype()
        if not use_base_init:
            self.params_dtype = params_dtype

        vllm_config = get_current_vllm_config()
        self.model_type = vllm_config.model_config.hf_config.model_type
        if not use_base_init:
            self.total_run_time = 0
            self.total_comm_time = 0
            self.is_sequence_parallel = False

            self.moe_parallel_config = FusedMoEParallelConfig.make(
                tp_size_=(tp_size if tp_size is not None else
                          get_tensor_model_parallel_world_size()),
                dp_size_=(dp_size
                          if dp_size is not None else get_dp_group().world_size),
                vllm_parallel_config=vllm_config.parallel_config)
            self.sp_size = self.tp_size
            compilation_config = vllm_config.compilation_config
            if prefix in compilation_config.static_forward_context:
                raise ValueError(f"Duplicate layer name: {prefix}")
            compilation_config.static_forward_context[prefix] = self
            self.layer_name = prefix
            self.enable_eplb = False
            self.expert_load_view = None
            self.logical_to_physical_map = None
            self.logical_replica_count = None
            self.local_num_expert_weight_slots = None
            self.quant_config = quant_config
            self.moe_quant_config = None
            self.routed_scaling_factor = 1.0
            self.apply_router_weight_on_input = apply_router_weight_on_input
            self.zero_expert_num = 0
            self.zero_expert_type = None
            self.hidden_size = hidden_size
            self.batched_hidden_states = None
            self.batched_router_logits = None
            try:
                self.rank = int(torch.distributed.get_rank())
            except Exception:
                self.rank = -1

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
        self.lossless_cpu_shadow_local_slots: dict[int, int] = {}
        self.lossless_loaded_offloaded = False
        self.lossless_cpu_import_expert_ids = []
        self.lossless_zero_redundancy_preallocated = False
        self.lossless_zero_redundancy_preallocated_loaded = False
        self.lossless_mode1_native_parity_ready = False
        self.loaded_weight_capacity = num_experts
        self.lossless_saved_primary_prefix_w13 = None
        self.lossless_saved_primary_prefix_w2 = None
        self.lossless_fixed_slot_plan_logged = False
        self.lossless_hybrid_reuse_log_budget = 0
        self.lossless_hybrid_fallback_logged = False
        self.lossless_primary_prefix_stash_logged = False
        self.lossless_primary_prefix_restore_logged = False
        self.lossless_hybrid_cpu_swap_enabled = False
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
        self.elastic_execution_mode = elastic_execution_mode
        self.elastic_moe_mode = ascend_config.elastic_moe_mode
        self.expert_map_path = ascend_config.expert_map_path
        self.global_redundant_expert_num = (
            envs_ascend.compute_elastic_init_redundancy_expert(
                num_experts,
                int(getattr(self.moe_parallel_config, "ep_size", 1)),
                ascend_config.init_redundancy_expert,
            ))
        effective_init_redundancy_expert = int(
            self.global_redundant_expert_num)
        if (self.elastic_moe_mode == "lossless"
                and self.global_redundant_expert_num <= 0):
            pass  # debug log removed
        self.lossless_lazy_activation = (
            self.elastic_moe_mode == "lossless"
            and self.global_redundant_expert_num <= 0)
        self.lossless_hybrid_cpu_swap_enabled = (
            self.elastic_moe_mode == "lossless"
            and self.elastic_execution_mode in (2, 3))
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
                # In unified mode=1 the effective redundancy can be derived
                # from VLLM_ASCEND_ELASTIC_MIN_COMPUTE_GROUP_SIZE even when
                # the legacy init_redundancy knob is left at zero. Preserve
                # that derived value so custom Qwen3 loads real redundant
                # expert slots, matching the native fixed-slot mode=1 path.
                self.global_redundant_expert_num = (
                    effective_init_redundancy_expert)
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
        self.elastic_original_ep_size = int(
            getattr(self.moe_parallel_config, "ep_size", 1))
        self.loaded_weight_capacity = int(self.loaded_local_num_experts)
        if self.layer_idx == 0:
            pass  # debug log removed
        if self.elastic_moe_mode == "lossless":
            prealloc_capacity = self._get_lossless_loaded_slot_capacity()
            if prealloc_capacity > self.loaded_weight_capacity:
                self.loaded_weight_capacity = prealloc_capacity
                if getattr(self, "global_redundant_expert_num", 0) <= 0:
                    self.lossless_zero_redundancy_preallocated_loaded = True
        if self.elastic_moe_mode == "lossless" and self.elastic_execution_mode == 1:
            self.lossless_hybrid_cpu_swap_enabled = False
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
            self.lossless_cpu_shadow_local_slots = {}
            self.lossless_mode1_native_parity_ready = True
            if self.layer_idx == 0:
                logger.info(
                    "Mode1 parity init: layer=%s floor=%s active_local=%s "
                    "loaded_local=%s loaded_capacity=%s hybrid_disabled=%s "
                    "runtime_buffers_disabled=%s parity_ready=%s",
                    self.layer_idx,
                    envs_ascend.VLLM_ASCEND_ELASTIC_MIN_COMPUTE_GROUP_SIZE,
                    int(self.active_local_num_experts),
                    int(self.loaded_local_num_experts),
                    int(self.loaded_weight_capacity),
                    True,
                    True,
                    True,
                )
        if self.layer_idx == 0:
            pass  # debug log removed
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
            "num_experts": self.loaded_weight_capacity,
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
        if (os.getenv("VLLM_ASCEND_FULL_REDUNDANCY_EXPERIMENT_LOG", "0")
                in ("1", "true", "TRUE", "yes", "on")
                and self.elastic_moe_mode == "lossless"):
            try:
                w13_bytes = int(self.w13_weight.numel()) * int(
                    self.w13_weight.element_size())
                w2_bytes = int(self.w2_weight.numel()) * int(
                    self.w2_weight.element_size())
                total_weight_bytes = w13_bytes + w2_bytes
                loaded_capacity = max(int(self.loaded_weight_capacity), 1)
                expert_weight_bytes = total_weight_bytes // loaded_capacity
                logger.info(
                    "Elastic redundancy MoE slots: layer=%s ep_rank=%s "
                    "ep_size=%s mode=%s floor=%s num_experts=%s "
                    "active_local=%s loaded_local=%s loaded_capacity=%s "
                    "redundant_experts=%s w13_shape=%s w2_shape=%s "
                    "expert_weight_bytes=%s total_weight_bytes=%s",
                    self.layer_idx,
                    self.ep_rank,
                    self.ep_size,
                    self.elastic_execution_mode,
                    envs_ascend.VLLM_ASCEND_ELASTIC_MIN_COMPUTE_GROUP_SIZE,
                    self.elastic_original_num_experts,
                    self.active_local_num_experts,
                    self.loaded_local_num_experts,
                    self.loaded_weight_capacity,
                    self.global_redundant_expert_num,
                    tuple(self.w13_weight.shape),
                    tuple(self.w2_weight.shape),
                    expert_weight_bytes,
                    total_weight_bytes,
                )
            except Exception as exc:
                logger.warning(
                    "Elastic redundancy MoE slot logging failed at layer=%s: %s",
                    self.layer_idx,
                    exc,
                )
        if self.layer_idx == 0:
            pass  # debug log removed

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
            local_expert_id = self.loaded_expert_map[expert_id].item()
            if self.layer_idx == 0 and 0 <= int(expert_id) < 16:
                if not hasattr(self, "_lossless_loaded_map_debugged"):
                    self._lossless_loaded_map_debugged = set()
                key = int(expert_id)
                if key not in self._lossless_loaded_map_debugged:
                    self._lossless_loaded_map_debugged.add(key)
                    pass  # debug log removed
            return local_expert_id
        if self.expert_map is None:
            return expert_id
        return self.expert_map[expert_id].item()

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

    def _is_followup_shrink_enabled(self) -> bool:
        initial_ep_size = self._get_elastic_initial_ep_size()
        if initial_ep_size <= 1:
            return False
        min_compute_group_size = (
            self._get_configured_elastic_min_compute_group_size())
        if min_compute_group_size is None:
            return True
        return min_compute_group_size < initial_ep_size

    def _get_effective_runtime_prealloc_floor_ep_size(self) -> int:
        min_compute_group_size = (
            self._get_configured_elastic_min_compute_group_size())
        if min_compute_group_size is not None:
            return min_compute_group_size
        return self._get_default_single_shrink_ep_floor()

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

    def _get_allowed_lossless_active_local_expert_counts(self) -> set[int]:
        logical_num_experts = self._get_logical_num_experts_for_elastic()
        initial_ep_size = self._get_elastic_initial_ep_size()
        min_ep_size = self._get_effective_runtime_prealloc_floor_ep_size()
        allowed_counts = {0}
        ep_size = initial_ep_size
        while ep_size >= 1:
            if logical_num_experts % ep_size == 0:
                allowed_counts.add(logical_num_experts // ep_size)
            if ep_size <= min_ep_size:
                break
            next_ep_size = ep_size // 2
            if next_ep_size <= 0 or next_ep_size == ep_size:
                break
            ep_size = next_ep_size
        return allowed_counts

    def _get_zero_redundancy_prealloc_capacity(self) -> int:
        active_local_num_experts = int(self.active_local_num_experts)
        if (self.elastic_moe_mode != "lossless"
                or getattr(self, "global_redundant_expert_num", 0) > 0):
            return active_local_num_experts
        if not self._is_followup_shrink_enabled():
            return active_local_num_experts
        if self._is_hybrid_cpu_swap_enabled():
            return max(active_local_num_experts,
                       self._get_hybrid_resident_capacity())
        logical_num_experts = self._get_logical_num_experts_for_elastic()
        min_compute_group_size = (
            self._get_configured_elastic_min_compute_group_size())
        if min_compute_group_size is not None:
            return max(
                active_local_num_experts,
                self._get_reserved_local_expert_slots_for_floor(
                    min_compute_group_size))

        current_ep_size = self._get_elastic_initial_ep_size()
        next_ep_size = current_ep_size // 2
        if next_ep_size <= 0 or logical_num_experts % next_ep_size != 0:
            return active_local_num_experts
        return max(active_local_num_experts, logical_num_experts // next_ep_size)

    def _get_lossless_loaded_slot_capacity(self) -> int:
        loaded_local_num_experts = int(
            getattr(self, "loaded_local_num_experts",
                    getattr(self, "active_local_num_experts", 0)))
        if self.elastic_moe_mode != "lossless":
            return loaded_local_num_experts
        if getattr(self, "global_redundant_expert_num", 0) <= 0:
            return self._get_zero_redundancy_prealloc_capacity()
        if not self._is_followup_shrink_enabled():
            return loaded_local_num_experts
        if self._is_hybrid_cpu_swap_enabled():
            return max(loaded_local_num_experts,
                       self._get_hybrid_resident_capacity())
        min_compute_group_size = (
            self._get_configured_elastic_min_compute_group_size())
        if min_compute_group_size is None:
            return loaded_local_num_experts
        return max(
            loaded_local_num_experts,
            self._get_reserved_local_expert_slots_for_floor(
                min_compute_group_size))

    def _is_hybrid_cpu_swap_enabled(self) -> bool:
        if not getattr(self, "lossless_hybrid_cpu_swap_enabled", False):
            return False
        if self.elastic_moe_mode != "lossless":
            return False
        if getattr(self, "elastic_execution_mode", 0) == 1:
            return False
        return True

    def _is_mode3_cross_layer_buffer_enabled(self) -> bool:
        return (self._is_hybrid_cpu_swap_enabled()
                and int(getattr(self, "elastic_execution_mode", 0)) == 3)

    def _get_hybrid_resident_capacity(self) -> int:
        if getattr(self, "elastic_execution_mode", 0) == 3:
            return max(1, self._get_lossless_primary_prefix_row_count())
        configured_resident_slots = (
            envs_ascend.VLLM_ASCEND_ELASTIC_HYBRID_RESIDENT_EXPERT_SLOTS)
        if (getattr(self, "elastic_execution_mode", 0) == 2
                and configured_resident_slots is not None):
            return max(
                int(getattr(self, "active_local_num_experts", 0)),
                int(configured_resident_slots),
            )
        default_floor = self._get_default_single_shrink_ep_floor()
        return max(
            int(getattr(self, "active_local_num_experts", 0)),
            self._get_reserved_local_expert_slots_for_floor(default_floor))

    def _hybrid_requires_multi_wave_execution(self) -> bool:
        if self._is_mode3_cross_layer_buffer_enabled():
            return False
        if not self.lossless_hybrid_active:
            return False
        resident_capacity = int(self.lossless_hybrid_resident_capacity)
        if resident_capacity <= 0:
            return False
        rank_owned_expert_ids = getattr(self, "lossless_hybrid_rank_owned_expert_ids",
                                        None)
        if rank_owned_expert_ids:
            return any(
                len(expert_ids) > resident_capacity
                for expert_ids in rank_owned_expert_ids)
        return len(getattr(self, "lossless_hybrid_owned_expert_ids", [])) > resident_capacity

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
        if self.elastic_moe_mode != "lossless":
            return False
        if int(getattr(self, "global_redundant_expert_num", 0)) > 0:
            return False
        if self._is_mode3_cross_layer_buffer_enabled():
            return False
        return self._is_hybrid_cpu_swap_enabled()

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
        if self.elastic_execution_mode != 1:
            self.lossless_mode1_native_parity_ready = False
        self.lossless_mode3_primary_prefix_expert_ids = []
        self.lossless_mode3_primary_prefix_local_slots = {}
        self.lossless_cpu_shadow_local_slots = {}
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
        if int(getattr(self, "loaded_weight_capacity", 0)) < resident_capacity:
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
        if int(getattr(self, "loaded_weight_capacity", 0)) <= 0:
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
        if (self.layer_idx == 0
                and not self.lossless_primary_prefix_stash_logged):
            logger.info(
                "Lossless primary prefix stashed for hybrid reuse: layer=%s rows=%s loaded_capacity=%s hybrid_resident_capacity=%s",
                self.layer_idx,
                row_count,
                int(getattr(self, "loaded_weight_capacity", 0)),
                self._get_hybrid_resident_capacity(),
            )
            self.lossless_primary_prefix_stash_logged = True

    def _restore_stashed_lossless_primary_prefix(self) -> None:
        saved_w13 = getattr(self, "lossless_saved_primary_prefix_w13", None)
        saved_w2 = getattr(self, "lossless_saved_primary_prefix_w2", None)
        if saved_w13 is None or saved_w2 is None:
            return
        row_count = min(int(saved_w13.shape[0]), int(saved_w2.shape[0]),
                        int(self.w13_weight.shape[0]), int(self.w2_weight.shape[0]))
        if row_count <= 0:
            return
        self.w13_weight[:row_count].copy_(saved_w13.to(
            device=self.w13_weight.device, dtype=self.w13_weight.dtype),
                                          non_blocking=False)
        self.w2_weight[:row_count].copy_(saved_w2.to(
            device=self.w2_weight.device, dtype=self.w2_weight.dtype),
                                         non_blocking=False)
        if (self.layer_idx == 0
                and not self.lossless_primary_prefix_restore_logged):
            logger.info(
                "Lossless primary prefix restored after hybrid stage: layer=%s rows=%s loaded_capacity=%s",
                self.layer_idx,
                row_count,
                int(getattr(self, "loaded_weight_capacity", 0)),
            )
            self.lossless_primary_prefix_restore_logged = True

    def _get_lossless_cpu_pair_for_expert(
            self, expert_id: int) -> tuple[torch.Tensor, torch.Tensor]:
        expert_id = int(expert_id)
        export_map = self.loaded_expert_map if self.loaded_expert_map is not None else self.expert_map
        if export_map is None:
            raise RuntimeError(
                f"Missing expert map while resolving expert_id={expert_id} at layer={self.layer_idx}."
            )
        local_slot = int(export_map[expert_id].item())
        if local_slot < 0:
            raise RuntimeError(
                f"Expert {expert_id} is not available on rank={self.ep_rank} at layer={self.layer_idx}."
            )
        if (self.lossless_cpu_w13_weight is not None
                and self.lossless_cpu_w2_weight is not None
                and local_slot < int(self.lossless_cpu_w13_weight.shape[0])
                and local_slot < int(self.lossless_cpu_w2_weight.shape[0])):
            return (self.lossless_cpu_w13_weight[local_slot].detach().cpu(),
                    self.lossless_cpu_w2_weight[local_slot].detach().cpu())
        if local_slot < int(self.w13_weight.shape[0]):
            return (self.w13_weight[local_slot].detach().cpu(),
                    self.w2_weight[local_slot].detach().cpu())
        raise RuntimeError(
            f"Cannot resolve CPU shadow for expert_id={expert_id}, slot={local_slot} "
            f"at layer={self.layer_idx}.")

    def _get_lossless_cpu_pair_for_local_slot(
            self, local_slot: int) -> tuple[torch.Tensor, torch.Tensor]:
        local_slot = int(local_slot)
        if local_slot < 0:
            raise RuntimeError(
                f"Invalid local_slot={local_slot} at layer={self.layer_idx}.")
        if (local_slot < int(self.w13_weight.shape[0])
                and local_slot < int(self.w2_weight.shape[0])):
            return (self.w13_weight[local_slot].detach().cpu(),
                    self.w2_weight[local_slot].detach().cpu())
        if (self.lossless_cpu_w13_weight is not None
                and self.lossless_cpu_w2_weight is not None
                and local_slot < int(self.lossless_cpu_w13_weight.shape[0])
                and local_slot < int(self.lossless_cpu_w2_weight.shape[0])):
            return (self.lossless_cpu_w13_weight[local_slot].detach().cpu(),
                    self.lossless_cpu_w2_weight[local_slot].detach().cpu())
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
        for expert_id, source_local_id in zip(active_expert_ids, source_local_ids):
            if int(source_local_id) >= 0:
                source_w13, source_w2 = self._get_lossless_cpu_pair_for_local_slot(
                    int(source_local_id))
            else:
                if cpu_expert_weights is None or int(expert_id) not in cpu_expert_weights:
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
                # Padding may duplicate a real resident expert to occupy unused
                # fixed slots. Keep the first slot for routing and treat later
                # duplicates as inert fillers so worker-side validation and
                # token dispatch observe the canonical slot layout.
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
        self._mode3_hybrid_inactive_logged = False
        self._mode3_gate_reject_logged = False
        self._mode3_refresh_skip_logged = False
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

    def materialize_hybrid_resident_experts(self,
                                            target_resident_expert_ids:
                                            list[int]) -> None:
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
                    f"Hybrid resident materialization requires at least one target expert at layer={self.layer_idx}."
                )
            target.extend([int(target[0])] * (resident_capacity - len(target)))
        if target == list(self.lossless_hybrid_resident_expert_ids):
            return

        self.ensure_lossless_cpu_shadow()
        self.lossless_cpu_w13_weight = _maybe_pin_cpu_tensor(
            self.lossless_cpu_w13_weight)
        self.lossless_cpu_w2_weight = _maybe_pin_cpu_tensor(
            self.lossless_cpu_w2_weight)
        if (self.lossless_cpu_w13_weight is None
                or self.lossless_cpu_w2_weight is None):
            raise RuntimeError(
                "Hybrid resident materialization requires CPU shadow weights: "
                f"layer={self.layer_idx}")

        runtime_device = self.w13_weight.device
        runtime_w13_dtype = (self.lossless_cpu_w13_weight.dtype
                             if self.lossless_cpu_w13_weight is not None else
                             self.w13_weight.dtype)
        runtime_w2_dtype = (self.lossless_cpu_w2_weight.dtype
                            if self.lossless_cpu_w2_weight is not None else
                            self.w2_weight.dtype)
        reuse_loaded_prefix = self._can_reuse_loaded_prefix_for_hybrid(
            resident_capacity, runtime_device, runtime_w13_dtype,
            runtime_w2_dtype)
        if reuse_loaded_prefix:
            runtime_w13_storage = self.w13_weight[:resident_capacity]
            runtime_w2_storage = self.w2_weight[:resident_capacity]
            self.runtime_w13_buffer = None
            self.runtime_w2_buffer = None
            self.runtime_weight_capacity = max(
                resident_capacity, int(getattr(self, "loaded_weight_capacity",
                                               0)))
            if self.layer_idx == 0 and self.lossless_hybrid_reuse_log_budget > 0:
                logger.info(
                    "Hybrid resident materialization reusing fixed NPU slots: layer=%s resident_capacity=%s loaded_capacity=%s target_head=%s w13_ptr=%s w2_ptr=%s",
                    self.layer_idx,
                    resident_capacity,
                    int(getattr(self, "loaded_weight_capacity", 0)),
                    target[:8],
                    int(runtime_w13_storage.data_ptr()),
                    int(runtime_w2_storage.data_ptr()),
                )
                self.lossless_hybrid_reuse_log_budget -= 1
        else:
            runtime_format_w13 = _lossless_weight_format(self.w13_weight)
            runtime_format_w2 = _lossless_weight_format(self.w2_weight)
            if (self.runtime_w13_buffer is None or self.runtime_w2_buffer is None
                    or self.runtime_weight_capacity < resident_capacity
                    or self.runtime_w13_buffer.device != runtime_device
                    or self.runtime_w2_buffer.device != runtime_device
                    or self.runtime_w13_buffer.dtype != runtime_w13_dtype
                    or self.runtime_w2_buffer.dtype != runtime_w2_dtype
                    or _lossless_weight_format(self.runtime_w13_buffer) != runtime_format_w13
                    or _lossless_weight_format(self.runtime_w2_buffer) != runtime_format_w2):
                self.runtime_w13_buffer = _allocate_formatted_buffer_like(
                    self.w13_weight,
                    resident_capacity,
                    dtype=runtime_w13_dtype)
                self.runtime_w2_buffer = _allocate_formatted_buffer_like(
                    self.w2_weight,
                    resident_capacity,
                    dtype=runtime_w2_dtype)
                self.runtime_weight_capacity = resident_capacity
                if self.layer_idx == 0 and not self.lossless_hybrid_fallback_logged:
                    logger.warning(
                        "Hybrid resident materialization fell back to fresh runtime buffers: layer=%s resident_capacity=%s loaded_capacity=%s",
                        self.layer_idx,
                        resident_capacity,
                        int(getattr(self, "loaded_weight_capacity", 0)),
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
                "Hybrid fixed-slot activation starting: layer=%s owned_local=%s resident_capacity=%s loaded_capacity=%s fixed_slot_candidate=%s",
                self.layer_idx,
                len(active_expert_ids),
                resident_capacity,
                int(getattr(self, "loaded_weight_capacity", 0)),
                self._can_reuse_loaded_prefix_for_hybrid(
                    resident_capacity,
                    self.w13_weight.device,
                    self.w13_weight.dtype,
                    self.w2_weight.dtype,
                ),
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
                loaded_expert_map[int(expert_id)] = local_slot
            self.loaded_expert_map = loaded_expert_map
            self.loaded_local_num_experts = len(active_expert_ids)
            self.lossless_cpu_import_expert_ids = [
                int(expert_id) for expert_id, source_local_id in zip(
                    active_expert_ids, source_local_ids)
                if int(source_local_id) < 0
            ]
            self.lossless_hybrid_owned_expert_ids = [
                int(x) for x in active_expert_ids
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
            self.lossless_hybrid_resident_expert_ids = list(
                self.lossless_mode3_primary_prefix_expert_ids)
            self.lossless_hybrid_cpu_only_expert_ids = [
                expert_id for expert_id in self.lossless_hybrid_owned_expert_ids
                if expert_id not in set(self.lossless_mode3_primary_prefix_expert_ids)
            ]
            self.active_local_num_experts = len(active_expert_ids)
            self.local_num_experts = len(active_expert_ids)
            self.moe_config.num_local_experts = len(active_expert_ids)
            self.moe_config.num_experts = len(active_expert_ids)
            self.num_experts = len(active_expert_ids)
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
        self.lossless_hybrid_owned_expert_ids = [int(x) for x in active_expert_ids]
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
        # Keep the local planning pass on CPU. This avoids exercising aclnnIndex
        # on the device inside the fallback path, and it makes the planner more
        # resilient if a previous async mode3 prefetch has already put the NPU
        # stream into an error state.
        owner_rank = self.lossless_hybrid_owner_rank_by_expert.detach().cpu()[
            logical_topk_ids.detach().cpu()
        ]
        logical_topk_ids_cpu = logical_topk_ids.detach().cpu()
        resident_capacity = int(self.lossless_hybrid_resident_capacity)
        per_rank_needed: list[list[int]] = []
        per_rank_waves: list[list[list[int]]] = []
        per_rank_final_resident: list[list[int]] = []
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
            gathered_needed: list[Optional[list[list[int]]]] = [None] * active_rank_count
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
                gathered_wave_topk: list[Optional[int]] = [None] * active_rank_count
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

    @staticmethod
    def _summarize_slot_region(weight: Optional[torch.Tensor], start: int,
                               end: int) -> float:
        if weight is None or weight.numel() == 0:
            return 0.0
        row_count = int(weight.shape[0])
        start = max(0, int(start))
        end = min(max(start, int(end)), row_count)
        if start >= end:
            return 0.0
        return float(weight[start:end].detach().abs().mean().item())

    def set_lossless_runtime_prefix_views(self) -> None:
        if self.elastic_moe_mode != "lossless":
            self.clear_lossless_hybrid_state()
            self.runtime_w13_weight = None
            self.runtime_w2_weight = None
            self.runtime_w13_buffer = None
            self.runtime_w2_buffer = None
            self.runtime_weight_capacity = 0
            self.lossless_runtime_activated = False
            return

        active_local_num_experts = int(self.active_local_num_experts)
        allowed_local_num_experts = (
            self._get_allowed_lossless_active_local_expert_counts())
        if active_local_num_experts not in allowed_local_num_experts:
            raise RuntimeError(
                f"Unsupported active_local_num_experts={active_local_num_experts}, "
                f"allowed={sorted(allowed_local_num_experts)} "
                f"at layer={self.layer_idx}.")
        loaded_capacity = int(getattr(self, "loaded_weight_capacity", 0))
        if active_local_num_experts > int(self.w13_weight.shape[0]):
            raise RuntimeError(
                f"contain the active prefix: active={active_local_num_experts}, "
                f"loaded_rows={int(self.w13_weight.shape[0])} at layer={self.layer_idx}.")
        self.runtime_w13_weight = self.w13_weight[:active_local_num_experts]
        self.runtime_w2_weight = self.w2_weight[:active_local_num_experts]
        self.runtime_w13_buffer = None
        self.runtime_w2_buffer = None
        self.runtime_weight_capacity = max(
            loaded_capacity, int(self.w13_weight.shape[0]))
        self.lossless_runtime_activated = True

    def _can_use_lossless_loaded_prefix_views(
            self,
            source_local_ids: list[int],
            cpu_expert_weights: Optional[dict[int, tuple[torch.Tensor,
                                                         torch.Tensor]]],
    ) -> bool:
        if self.elastic_moe_mode != "lossless":
            return False
        if self.lossless_loaded_offloaded:
            return False
        if cpu_expert_weights:
            return False
        active_local_num_experts = int(self.active_local_num_experts)
        if len(source_local_ids) != active_local_num_experts:
            return False
        if any(int(source_local_id) != local_slot
               for local_slot, source_local_id in enumerate(source_local_ids)):
            return False
        if active_local_num_experts > int(self.w13_weight.shape[0]):
            return False
        if active_local_num_experts > int(self.w2_weight.shape[0]):
            return False
        return True

    def _can_materialize_lossless_loaded_prefix_slots(
            self,
            source_local_ids: list[int],
            cpu_expert_weights: Optional[dict[int, tuple[torch.Tensor,
                                                         torch.Tensor]]],
    ) -> bool:
        if self.elastic_moe_mode != "lossless":
            return False
        if self.lossless_loaded_offloaded:
            return False
        active_local_num_experts = int(self.active_local_num_experts)
        if len(source_local_ids) != active_local_num_experts:
            return False
        if active_local_num_experts > int(self.w13_weight.shape[0]):
            return False
        if active_local_num_experts > int(self.w2_weight.shape[0]):
            return False
        if active_local_num_experts > int(
                getattr(self, "loaded_weight_capacity", 0)):
            return False
        if getattr(self, "elastic_execution_mode", 0) == 1:
            return True
        return bool(
            getattr(self, "lossless_zero_redundancy_preallocated_loaded",
                    False))

    def _can_materialize_lossless_loaded_prefix_slots_for_target(
            self,
            target_active_local_num_experts: int,
            source_local_ids: list[int],
            cpu_expert_weights: Optional[dict[int, tuple[torch.Tensor,
                                                         torch.Tensor]]],
    ) -> bool:
        if self.elastic_moe_mode != "lossless":
            return False
        if self.lossless_loaded_offloaded:
            return False
        target_active_local_num_experts = int(target_active_local_num_experts)
        if len(source_local_ids) != target_active_local_num_experts:
            return False
        if target_active_local_num_experts > int(self.w13_weight.shape[0]):
            return False
        if target_active_local_num_experts > int(self.w2_weight.shape[0]):
            return False
        if target_active_local_num_experts > int(
                getattr(self, "loaded_weight_capacity", 0)):
            return False
        if getattr(self, "elastic_execution_mode", 0) == 1:
            return True
        return bool(
            getattr(self, "lossless_zero_redundancy_preallocated_loaded",
                    False))

    def _copy_lossless_loaded_slots_into_prefix(
            self, moves: list[tuple[int, int]]) -> None:
        pending = {int(dst): int(src) for dst, src in moves if dst != src}
        while pending:
            progressed = False
            pending_sources = set(pending.values())
            for dst, src in list(pending.items()):
                if dst in pending_sources:
                    continue
                self.w13_weight[dst].copy_(self.w13_weight[src])
                self.w2_weight[dst].copy_(self.w2_weight[src])
                del pending[dst]
                progressed = True
            if progressed:
                continue

            start = next(iter(pending))
            tmp_w13 = self.w13_weight[start].detach().clone()
            tmp_w2 = self.w2_weight[start].detach().clone()
            dst = start
            while True:
                src = pending[dst]
                if src == start:
                    self.w13_weight[dst].copy_(tmp_w13)
                    self.w2_weight[dst].copy_(tmp_w2)
                    del pending[dst]
                    break
                self.w13_weight[dst].copy_(self.w13_weight[src])
                self.w2_weight[dst].copy_(self.w2_weight[src])
                del pending[dst]
                dst = src

    def restore_lossless_full_world_primary_layout(self) -> None:
        if self.elastic_moe_mode != "lossless":
            self.reset_expert_map_and_log2phy()
            return

        self.clear_lossless_hybrid_state()
        logical_num_experts = int(
            getattr(self, "elastic_original_num_experts", self.num_experts))
        self.num_experts = logical_num_experts
        self.moe_config.num_experts = logical_num_experts

        target_device = None
        if self.log2phy is not None:
            target_device = self.log2phy.device
        elif self.expert_map is not None:
            target_device = self.expert_map.device
        elif hasattr(self, "w13_weight"):
            target_device = self.w13_weight.device
        if target_device is not None and target_device.type == "cpu" and hasattr(
                torch, "npu") and torch.npu.is_available():
            target_device = torch.device("npu", torch.npu.current_device())
        log2phy_dtype = (self.log2phy.dtype
                         if self.log2phy is not None else torch.int32)

        self.loaded_local_num_experts, self.loaded_expert_map = (
            determine_redundant_replica_expert_map(
                logical_num_experts,
                self.ep_size,
                self.ep_rank,
                self.global_redundant_expert_num,
            ))
        active_mapping = determine_expert_map(
            self.ep_size,
            self.ep_rank,
            logical_num_experts,
            layer_idx=self.layer_idx,
        )
        if len(active_mapping) == 3:
            (self.active_local_num_experts, self.expert_map,
             self.log2phy) = active_mapping
        elif len(active_mapping) == 2:
            self.active_local_num_experts, self.expert_map = active_mapping
            self.log2phy = determine_default_log2phy_map(
                logical_num_experts, self.ep_size, self.ep_rank, 0)
        else:
            raise ValueError(
                f"Unexpected determine_expert_map return arity={len(active_mapping)}")
        if target_device is not None:
            if self.loaded_expert_map is not None:
                self.loaded_expert_map = self.loaded_expert_map.to(
                    device=target_device, dtype=torch.int32)
            if self.expert_map is not None:
                self.expert_map = self.expert_map.to(
                    device=target_device, dtype=torch.int32)
            if self.log2phy is not None:
                self.log2phy = self.log2phy.to(device=target_device,
                                               dtype=log2phy_dtype)
        self.primary_log2phy = self.log2phy.clone()
        self.local_num_experts = self.active_local_num_experts
        self.moe_config.num_local_experts = self.active_local_num_experts
        self.elastic_runtime_log2phy = None
        self.lossless_cpu_import_expert_ids = []
        self._restore_stashed_lossless_primary_prefix()
        self.set_lossless_runtime_prefix_views()
        if self.elastic_execution_mode == 1:
            self.lossless_mode1_native_parity_ready = True
        if self.layer_idx == 0 and self.ep_rank == 0:
            logger.info(
                "Lossless full-world primary layout restored with loaded-slot prefix views: layer=%s active_local=%s loaded_capacity=%s",
                self.layer_idx,
                self.active_local_num_experts,
                int(getattr(self, "loaded_weight_capacity", 0)),
            )

    def prepare_lossless_zero_redundancy_runtime_slots(self) -> bool:
        if (self.elastic_moe_mode != "lossless"
                or getattr(self, "global_redundant_expert_num", 0) > 0):
            return False
        active_local_num_experts = int(self.active_local_num_experts)
        target_capacity = self._get_zero_redundancy_prealloc_capacity()
        if target_capacity <= active_local_num_experts:
            return False
        self.set_lossless_runtime_prefix_views()
        self.runtime_weight_capacity = max(
            target_capacity, int(getattr(self, "loaded_weight_capacity", 0)))
        self.lossless_zero_redundancy_preallocated = True
        return True

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
        if self.elastic_execution_mode == 1:
            self.lossless_mode1_native_parity_ready = False
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
            self.lossless_mode1_native_parity_ready = False
            return

        active_local_num_experts = int(self.active_local_num_experts)
        if source_local_ids is None:
            source_local_ids = list(range(active_local_num_experts))
        if len(source_local_ids) != active_local_num_experts:
            raise RuntimeError(
                f"expected {active_local_num_experts}, got {len(source_local_ids)}."
            )
        preserve_prefix_len = min(max(int(preserve_prefix_len), 0),
                                  active_local_num_experts)

        if getattr(self, "global_redundant_expert_num", 0) <= 0:
            if cpu_expert_weights is not None:
                raise RuntimeError(
                    "materialized into loaded weights before refreshing runtime "
                    f"views at layer={self.layer_idx}.")
            expected_prefix = list(range(active_local_num_experts))
            if list(source_local_ids) != expected_prefix:
                raise RuntimeError(
                    "contiguous prefix of loaded weights: "
                    f"source_local_ids={source_local_ids}, expected={expected_prefix} "
                    f"at layer={self.layer_idx}.")
            if preserve_prefix_len not in (0, active_local_num_experts):
                raise RuntimeError(
                    f"buffers; got preserve_prefix_len={preserve_prefix_len} "
                    f"at layer={self.layer_idx}.")
            self.set_lossless_runtime_prefix_views()
            if self.elastic_execution_mode == 1:
                self.lossless_mode1_native_parity_ready = True
            return

        target_capacity = self._get_lossless_runtime_capacity(
            active_local_num_experts)
        target_format_w13 = _lossless_weight_format(self.w13_weight)
        target_format_w2 = _lossless_weight_format(self.w2_weight)
        if (self.runtime_w13_buffer is None or self.runtime_w2_buffer is None
                or self.runtime_weight_capacity < target_capacity
                or _lossless_weight_format(self.runtime_w13_buffer) != target_format_w13
                or _lossless_weight_format(self.runtime_w2_buffer) != target_format_w2):
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
            self.runtime_w13_buffer = _allocate_formatted_buffer_like(
                self.w13_weight,
                target_capacity,
                dtype=buffer_w13_dtype)
            self.runtime_w2_buffer = _allocate_formatted_buffer_like(
                self.w2_weight,
                target_capacity,
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
        if self.elastic_execution_mode == 1:
            self.lossless_mode1_native_parity_ready = False

    def _get_lossless_runtime_capacity(self,
                                       active_local_num_experts: int) -> int:
        capacity = int(active_local_num_experts)
        min_compute_group_size = (
            self._get_configured_elastic_min_compute_group_size())
        if min_compute_group_size is not None:
            return max(
                capacity,
                self._get_reserved_local_expert_slots_for_floor(
                    min_compute_group_size))
        logical_num_experts = self._get_logical_num_experts_for_elastic()
        if getattr(self, "global_redundant_expert_num", 0) <= 0:
            # In zero-redundancy mode, pre-reserving the next shrink stage
            # makes the first 16->8 activation peak too expensive.
            return capacity
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
        self.clear_lossless_hybrid_state()
        self._restore_stashed_lossless_primary_prefix()
        if getattr(self, "lossless_lazy_activation", False):
            if self.prepare_lossless_zero_redundancy_runtime_slots():
                if self.ep_rank == 0:
                    logger.info(
                        "Preallocating lossless zero-redundancy runtime slots: layer=%s active_local=%s capacity=%s fixed_slot_reuse=%s primary_prefix_rows=%s hybrid_resident_capacity=%s",
                        self.layer_idx,
                        self.active_local_num_experts,
                        self.runtime_weight_capacity,
                        bool(
                            getattr(self,
                                    "lossless_zero_redundancy_preallocated_loaded",
                                    False)),
                        self._get_lossless_primary_prefix_row_count(),
                        self._get_hybrid_resident_capacity(),
                    )
                    if (self.layer_idx == 0
                            and not self.lossless_fixed_slot_plan_logged):
                        logger.info(
                            "Lossless fixed-slot resident plan ready: layer=%s primary_prefix_rows=%s loaded_capacity=%s runtime_capacity=%s hybrid_resident_capacity=%s",
                            self.layer_idx,
                            self._get_lossless_primary_prefix_row_count(),
                            int(getattr(self, "loaded_weight_capacity", 0)),
                            int(getattr(self, "runtime_weight_capacity", 0)),
                            self._get_hybrid_resident_capacity(),
                        )
                        self.lossless_fixed_slot_plan_logged = True
            else:
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
                    f"primary expert {expert_id} missing from loaded_expert_map "
                    f"at layer={self.layer_idx}."
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
        runtime_w13 = (self.runtime_w13_buffer
                       if self.runtime_w13_buffer is not None else
                       self.runtime_w13_weight)
        runtime_w2 = (self.runtime_w2_buffer if self.runtime_w2_buffer is not None
                      else self.runtime_w2_weight)
        if (runtime_w13 is not None and runtime_w2 is not None
                and self.expert_map is not None):
            runtime_slots = [int(self.expert_map[expert_id].item())
                             for expert_id in expert_ids]
            if all(local_slot >= 0 for local_slot in runtime_slots):
                return {
                    int(expert_id): (
                        runtime_w13[local_slot].detach().cpu(),
                        runtime_w2[local_slot].detach().cpu(),
                    )
                    for expert_id, local_slot in zip(expert_ids, runtime_slots)
                }
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
        runtime_w13 = (self.runtime_w13_buffer
                       if self.runtime_w13_buffer is not None else
                       self.runtime_w13_weight)
        runtime_w2 = (self.runtime_w2_buffer if self.runtime_w2_buffer is not None
                      else self.runtime_w2_weight)
        if (runtime_w13 is not None and runtime_w2 is not None
                and self.expert_map is not None):
            runtime_slots = [int(self.expert_map[expert_id].item())
                             for expert_id in expert_ids]
            if all(local_slot >= 0 for local_slot in runtime_slots):
                source_w13 = runtime_w13
                source_w2 = runtime_w2
                export_map = self.expert_map
            elif self.lossless_loaded_offloaded:
                if self.runtime_w13_weight is None or self.runtime_w2_weight is None:
                    raise RuntimeError(
                        f"expert weights were offloaded at layer={self.layer_idx}.")
                source_w13 = (self.runtime_w13_buffer
                              if self.runtime_w13_buffer is not None else
                              self.runtime_w13_weight)
                source_w2 = (self.runtime_w2_buffer if self.runtime_w2_buffer
                             is not None else self.runtime_w2_weight)
                export_map = self.expert_map
            else:
                if self.lossless_hybrid_active:
                    source_w13 = (self.runtime_w13_buffer
                                  if self.runtime_w13_buffer is not None else
                                  self.runtime_w13_weight)
                    source_w2 = (self.runtime_w2_buffer
                                 if self.runtime_w2_buffer is not None else
                                 self.runtime_w2_weight)
                    if source_w13 is None or source_w2 is None:
                        raise RuntimeError(
                            f"Hybrid resident runtime buffers are missing at "
                            f"layer={self.layer_idx}.")
                    export_map = self.expert_map
                else:
                    source_w13 = self.w13_weight
                    source_w2 = self.w2_weight
                    export_map = (self.loaded_expert_map
                                  if self.loaded_expert_map is not None else
                                  self.expert_map)
        elif self.lossless_loaded_offloaded:
            if self.runtime_w13_weight is None or self.runtime_w2_weight is None:
                raise RuntimeError(
                    f"expert weights were offloaded at layer={self.layer_idx}.")
            source_w13 = (self.runtime_w13_buffer
                          if self.runtime_w13_buffer is not None else
                          self.runtime_w13_weight)
            source_w2 = (self.runtime_w2_buffer if self.runtime_w2_buffer
                         is not None else self.runtime_w2_weight)
            export_map = self.expert_map
        else:
            if self.lossless_hybrid_active:
                source_w13 = (self.runtime_w13_buffer
                              if self.runtime_w13_buffer is not None else
                              self.runtime_w13_weight)
                source_w2 = (self.runtime_w2_buffer
                             if self.runtime_w2_buffer is not None else
                             self.runtime_w2_weight)
                if source_w13 is None or source_w2 is None:
                    raise RuntimeError(
                        f"Hybrid resident runtime buffers are missing at "
                        f"layer={self.layer_idx}.")
                export_map = self.expert_map
            else:
                source_w13 = self.w13_weight
                source_w2 = self.w2_weight
                export_map = (self.loaded_expert_map
                              if self.loaded_expert_map is not None else
                              self.expert_map)
        local_slots = [int(export_map[expert_id].item()) for expert_id in expert_ids]
        if len(local_slots) == 1:
            local_slot = local_slots[0]
            if local_slot < 0:
                raise RuntimeError(
                    f"expert_id={int(expert_ids[0])} at layer={self.layer_idx}."
                )
            # For the shrink path we export one expert at a time. Return a
            # narrow view of the canonical NPU tensor so P2P send can reuse the
            # source slot directly instead of materializing an index_select copy.
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

    def activate_lossless_local_experts(self, active_expert_ids: list[int],
                                        source_local_ids: list[int],
                                        cpu_expert_weights: Optional[dict[int, tuple[
                                            torch.Tensor, torch.Tensor]]] = None,
                                        offload_loaded_after_activation: bool = False) -> None:
        if self.elastic_moe_mode != "lossless":
            return
        if self.elastic_execution_mode == 1:
            self.clear_lossless_hybrid_state()
            self.lossless_hybrid_cpu_swap_enabled = False
            target_active_local_num_experts = len(active_expert_ids)
            direct_preloaded_activation_ok = bool(
                getattr(self, "_lossless_mode1_direct_preloaded_activation_ok",
                        False))
            if (direct_preloaded_activation_ok and self.layer_idx == 0
                    and not getattr(self,
                                    "_mode1_direct_preloaded_activation_logged",
                                    False)):
                logger.info(
                    "Mode1 parity accepting direct-preloaded activation: "
                    "layer=%s target_active=%s loaded_capacity=%s source_local_ids=%s",
                    self.layer_idx,
                    len(active_expert_ids),
                    int(getattr(self, "loaded_weight_capacity", 0)),
                    source_local_ids[:16],
                )
                self._mode1_direct_preloaded_activation_logged = True
            if (not direct_preloaded_activation_ok and
                    not self
                    ._can_materialize_lossless_loaded_prefix_slots_for_target(
                        target_active_local_num_experts,
                        source_local_ids,
                        cpu_expert_weights)):
                logger.warning(
                    "Mode1 parity path refused heavyweight activation fallback: "
                    "layer=%s active_local=%s loaded_capacity=%s "
                    "cpu_imports=%s source_local_ids=%s direct_preloaded_ok=%s",
                    self.layer_idx,
                    target_active_local_num_experts,
                    int(getattr(self, "loaded_weight_capacity", 0)),
                    0 if cpu_expert_weights is None else len(cpu_expert_weights),
                    source_local_ids,
                    direct_preloaded_activation_ok,
                )
                if _env_flag("VLLM_ASCEND_CUSTOM_MODE1_STRICT", "1"):
                    raise RuntimeError(
                        "Mode1 parity path would fall back to heavyweight "
                        f"runtime-buffer activation at layer={self.layer_idx}.")
        hybrid_enabled = self._is_hybrid_cpu_swap_enabled()
        primary_prefix_rows = self._get_lossless_primary_prefix_row_count()
        resident_capacity = (self._get_hybrid_resident_capacity()
                             if hybrid_enabled else 0)
        if (hybrid_enabled and len(active_expert_ids) > primary_prefix_rows
                and resident_capacity > 0):
            # In mode=2, any post-full-world activation that reaches the
            # shrink stages must stay on the fixed-slot hybrid path. 16-rank
            # keeps the primary 8-slot prefix view, while 8-rank and the
            # smaller tail stages continue to execute with a fixed resident
            # budget and optional CPU swap for overflow experts. Falling back
            # to the generic runtime-buffer path reallocates large NPU tensors
            # and causes OOM.
            if self.layer_idx == 0:
                logger.info(
                    "Redirecting lossless activation to hybrid fixed-slot path: layer=%s owned_local=%s resident_capacity=%s primary_prefix_rows=%s hybrid_active=%s",
                    self.layer_idx,
                    len(active_expert_ids),
                    resident_capacity,
                    primary_prefix_rows,
                    self.lossless_hybrid_active,
                )
            self.activate_lossless_hybrid_local_experts(
                active_expert_ids,
                source_local_ids,
                cpu_expert_weights=cpu_expert_weights)
            self.elastic_debug_tag += 1
            self.elastic_debug_budget = 1 if self.layer_idx == 0 else 0
            self.elastic_debug_reason = ("lossless_post_shrink"
                                         if self.layer_idx == 0 else None)
            return
        self.clear_lossless_hybrid_state()
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
        if self._can_materialize_lossless_loaded_prefix_slots(
                source_local_ids, cpu_expert_weights):
            activation_start = time.perf_counter()
            if new_local_num_experts > int(self.w13_weight.shape[0]):
                raise RuntimeError(
                    f"need {new_local_num_experts}, have {int(self.w13_weight.shape[0])} "
                    f"at layer={self.layer_idx}.")
            local_slot_moves: list[tuple[int, int]] = []
            for local_slot, source_local_id in enumerate(source_local_ids):
                if source_local_id >= 0:
                    if local_slot != source_local_id:
                        local_slot_moves.append(
                            (local_slot, int(source_local_id)))
                    continue
                if cpu_expert_weights is None:
                    continue
                expert_id = active_expert_ids[local_slot]
                if expert_id not in cpu_expert_weights:
                    fallback_loaded_slot = -1
                    loaded_expert_map = getattr(self, "loaded_expert_map", None)
                    if loaded_expert_map is not None:
                        try:
                            fallback_loaded_slot = int(loaded_expert_map[expert_id])
                        except Exception:
                            fallback_loaded_slot = -1

                    # If the same shrink target is re-entered, the imported expert may
                    # already live inside the canonical loaded tensor. Reuse that copy
                    # instead of requiring a second staged import.
                    if 0 <= fallback_loaded_slot < self.loaded_weight_capacity:
                        if fallback_loaded_slot != local_slot:
                            self.w13_weight[local_slot].copy_(
                                self.w13_weight[fallback_loaded_slot])
                            self.w2_weight[local_slot].copy_(
                                self.w2_weight[fallback_loaded_slot])
                        pass  # debug log removed
                        continue
                    else:
                        raise RuntimeError(
                            f"Missing imported expert {expert_id} "
                            f"for preallocated loaded slot fill at "
                            f"layer={self.layer_idx}.")
                cpu_w13, cpu_w2 = cpu_expert_weights[expert_id]
                self.w13_weight[local_slot].copy_(cpu_w13, non_blocking=False)
                self.w2_weight[local_slot].copy_(cpu_w2, non_blocking=False)
            self._copy_lossless_loaded_slots_into_prefix(local_slot_moves)
            # In preallocated loaded mode, imported experts are materialized into
            # the canonical loaded tensor prefix. Keep the loaded map aligned with
            # the current active layout so follow-up shrink/export logic sees the
            # real resident experts instead of the initialization-time view.
            self.loaded_expert_map = self.expert_map.clone()
            self.loaded_local_num_experts = new_local_num_experts
            self.lossless_loaded_offloaded = False
            self.lossless_cpu_w13_weight = None
            self.lossless_cpu_w2_weight = None
            self.lossless_cpu_shadow_local_slots = {}
            self.set_lossless_runtime_prefix_views()
            if self.elastic_execution_mode == 1:
                self.lossless_mode1_native_parity_ready = True
            self.elastic_debug_tag += 1
            self.elastic_debug_budget = 1 if self.layer_idx == 0 else 0
            self.elastic_debug_reason = ("lossless_post_shrink"
                                         if self.layer_idx == 0 else None)
            if self.layer_idx == 0:
                pass  # debug log removed
            return
        if self._can_use_lossless_loaded_prefix_views(source_local_ids,
                                                      cpu_expert_weights):
            self.lossless_loaded_offloaded = False
            self.lossless_cpu_w13_weight = None
            self.lossless_cpu_w2_weight = None
            self.lossless_cpu_shadow_local_slots = {}
            self.set_lossless_runtime_prefix_views()
            if self.elastic_execution_mode == 1:
                self.lossless_mode1_native_parity_ready = True
            if self.layer_idx == 0:
                logger.info(
                    "Lossless activation reused loaded-slot prefix views: layer=%s active_local=%s loaded_capacity=%s",
                    self.layer_idx,
                    new_local_num_experts,
                    int(getattr(self, "loaded_weight_capacity", 0)),
                )
            self.elastic_debug_tag += 1
            self.elastic_debug_budget = 1 if self.layer_idx == 0 else 0
            self.elastic_debug_reason = ("lossless_post_shrink"
                                         if self.layer_idx == 0 else None)
            return
        if offload_loaded_after_activation and not self.lossless_loaded_offloaded:
            self.offload_lossless_loaded_weights_to_cpu()
        self.refresh_lossless_runtime_weights(
            source_local_ids,
            cpu_expert_weights=cpu_expert_weights,
            preserve_prefix_len=preserve_prefix_len)
        if self.elastic_execution_mode == 1:
            self.lossless_mode1_native_parity_ready = False
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
        if (self.elastic_moe_mode == "lossless"
                and self._is_mode3_cross_layer_buffer_enabled()
                and self.lossless_hybrid_active):
            if (self.layer_idx == 0
                    and not getattr(self, "_mode3_refresh_skip_logged", False)):
                logger.info(
                    "Mode3 refresh preserving hybrid lazy runtime state: "
                    "layer=%s ep_size=%s active_local=%s owned_local=%s "
                    "resident_capacity=%s cpu_only_local=%s",
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
        debug_enabled = bool(
            _env_flag("VLLM_ASCEND_CUSTOM_FUSED_MOE_FORWARD_DEBUG", "0")
            and self.elastic_debug_budget > 0 and not is_dummy
            and not getattr(forward_context, "in_profile_run", False))
        if debug_enabled:
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
            runtime_w13_weight = getattr(self, "runtime_w13_weight", None)
            runtime_w2_weight = getattr(self, "runtime_w2_weight", None)
            compute_w13 = (runtime_w13_weight
                           if runtime_w13_weight is not None else
                           self.w13_weight)
            compute_w2 = (runtime_w2_weight
                          if runtime_w2_weight is not None else
                          self.w2_weight)
            compute_source = ("runtime_weight_view"
                              if runtime_w13_weight is not None else
                              "loaded_weight")

            def _tensor_ptr(tensor: Optional[torch.Tensor]) -> int:
                return int(tensor.data_ptr()) if tensor is not None else -1

            def _tensor_meta(tensor: Optional[torch.Tensor]) -> tuple:
                if tensor is None:
                    return ()
                return (tuple(tensor.shape), str(tensor.dtype),
                        tuple(tensor.stride()))

            def _is_prefix_view(view: Optional[torch.Tensor],
                                base: Optional[torch.Tensor]) -> bool:
                if view is None or base is None:
                    return False
                if view.device != base.device:
                    return False
                if view.dtype != base.dtype or view.ndim != base.ndim:
                    return False
                if tuple(view.shape[1:]) != tuple(base.shape[1:]):
                    return False
                if tuple(view.stride()) != tuple(base.stride()):
                    return False
                if int(view.storage_offset()) != 0:
                    return False
                if int(view.data_ptr()) != int(base.data_ptr()):
                    return False
                return int(view.shape[0]) <= int(base.shape[0])

            def _row_abs_mean_samples(tensor: Optional[torch.Tensor],
                                      start_row: int = 0,
                                      max_rows: int = 4) -> list[tuple[int, float]]:
                if tensor is None or tensor.ndim == 0:
                    return []
                samples = []
                end_row = min(int(tensor.shape[0]), start_row + max_rows)
                for row in range(start_row, end_row):
                    samples.append(
                        (row, round(float(tensor[row].float().abs().mean().item()), 6)))
                return samples

            def _row_abs_diff_samples(
                    lhs: Optional[torch.Tensor],
                    rhs: Optional[torch.Tensor],
                    max_rows: int = 4) -> list[tuple[int, float]]:
                if (lhs is None or rhs is None or lhs.shape != rhs.shape
                        or lhs.ndim == 0):
                    return []
                samples = []
                for row in range(min(int(lhs.shape[0]), max_rows)):
                    samples.append((
                        row,
                        round(float((lhs[row].float() - rhs[row].float()).abs().mean().item()),
                              6),
                    ))
                return samples

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
                "loaded_weight_capacity": int(self.loaded_weight_capacity),
                "compute_source": compute_source,
                "compute_w13_shape": tuple(compute_w13.shape),
                "compute_w2_shape": tuple(compute_w2.shape),
                "route_debug": False,
            }
            setattr(forward_context, "elastic_debug_info", debug_info)
            if self.elastic_debug_reason == "lossless_pre_shrink_loaded_only":
                compute_w13_ptr = _tensor_ptr(compute_w13)
                compute_w2_ptr = _tensor_ptr(compute_w2)
                runtime_w13_ptr = _tensor_ptr(runtime_w13_weight)
                runtime_w2_ptr = _tensor_ptr(runtime_w2_weight)
                loaded_w13_ptr = _tensor_ptr(self.w13_weight)
                loaded_w2_ptr = _tensor_ptr(self.w2_weight)
                pass  # debug log removed
        else:
            setattr(forward_context, "elastic_debug_info", None)
        # For w8a8 dynamic we can do npu_dynamic_quant and gate in parallel.
        quantized_x_for_share, dynamic_scale_for_share = None, None

        if shared_experts:
            # When all_reduce_merge is in progress, shared_experts does not do all_reduce in mlp, but waits until shared_experts+router_experts are completed before doing all_reduce
            shared_hidden_states = shared_experts(hidden_states)

        if forward_context.sp_enabled:
            replace_allreduce = True

        forced_hybrid_comm_alignment = bool(self.lossless_hybrid_active)
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
        mode1_host_alltoall_metadata = (
            getattr(self, "elastic_execution_mode", 0) == 1
            and getattr(self, "lossless_mode1_native_parity_ready", False)
            and _env_flag(
                "VLLM_ASCEND_MODE1_PARITY_ALLTOALL_HOST_METADATA", "1")
            and selected_moe_comm_type == MoECommType.ALLTOALL)
        if mode1_host_alltoall_metadata:
            forward_context.hybrid_force_host_alltoall_metadata = True
            forward_context.hybrid_stage_active_ranks = int(
                getattr(getattr(self, "moe_parallel_config", None), "ep_size",
                        getattr(self, "ep_size", 0)) or 0)
            if self.layer_idx == 0 and not getattr(
                    self, "_mode1_host_alltoall_metadata_logged", False):
                logger.info(
                    "Mode1 parity ALLTOALL metadata host path enabled: "
                    "layer=%s stage=%s selected=%s original=%s",
                    self.layer_idx,
                    forward_context.hybrid_stage_active_ranks,
                    selected_moe_comm_type,
                    original_moe_comm_type,
                )
                self._mode1_host_alltoall_metadata_logged = True
        if forced_hybrid_comm_alignment:
            effective_comm_type = selected_moe_comm_type
            effective_method = (get_moe_comm_method(effective_comm_type)
                                if effective_comm_type is not None else None)
            if effective_method is None:
                raise RuntimeError(
                    f"Hybrid mode cannot resolve the selected MoE comm method "
                    f"at layer={self.layer_idx}: comm_type={effective_comm_type}.")
            forward_context.moe_comm_method = effective_method
            forward_context.moe_comm_type = effective_comm_type
            if effective_comm_type == MoECommType.MC2:
                forward_context.fused_moe_state = FusedMoEState.MC2
            elif effective_comm_type == MoECommType.ALLTOALL:
                forward_context.fused_moe_state = FusedMoEState.All2All
            elif effective_comm_type == MoECommType.ALLGATHER:
                forward_context.fused_moe_state = FusedMoEState.AllGather
            elif effective_comm_type == MoECommType.NAIVE_MULTICAST:
                forward_context.fused_moe_state = FusedMoEState.NaiveMulticast
            else:
                forward_context.fused_moe_state = original_fused_moe_state
            stage_size = len(getattr(self, "lossless_hybrid_active_ranks", []))
            multi_wave_hint = self._hybrid_requires_multi_wave_execution()
            forward_context.hybrid_force_host_alltoall_metadata = (
                effective_comm_type == MoECommType.ALLTOALL)
            forward_context.hybrid_stage_active_ranks = stage_size
            logged_key = getattr(self, "_hybrid_effective_comm_logged_key",
                                 None)
            current_key = (
                stage_size,
                str(selected_moe_comm_type),
                str(effective_comm_type),
                bool(multi_wave_hint),
            )
            if self.layer_idx == 0 and logged_key != current_key:
                logger.info(
                    "Hybrid MoE comm resolution: layer=%s stage=%s selected=%s original=%s effective=%s multi_wave_hint=%s",
                    self.layer_idx,
                    stage_size,
                    selected_moe_comm_type,
                    original_moe_comm_type,
                    effective_comm_type,
                    multi_wave_hint,
                )
                logger.info(
                    "Hybrid MoE effective comm path: layer=%s stage=%s comm=%s",
                    self.layer_idx,
                    stage_size,
                    effective_comm_type,
                )
                self._hybrid_effective_comm_logged_key = current_key

        try:
            dummy_waste_timing = getattr(forward_context,
                                         "dummy_waste_timing", None)
            dummy_moe_timing_enabled = (
                isinstance(dummy_waste_timing, dict)
                and _env_flag("VLLM_ASCEND_DUMMY_WASTE_TIMING", "0"))
            dummy_moe_start = 0.0
            if dummy_moe_timing_enabled:
                dummy_selection_stats_enabled = _env_flag(
                    "VLLM_ASCEND_DUMMY_WASTE_SELECTION_STATS", "0")
                dummy_waste_timing["_current_moe_selected_topk"] = 0
                dummy_waste_timing["_current_moe_selected_tokens"] = 0
                dummy_waste_timing["_current_moe_selected_experts"] = 0
                if _env_flag("VLLM_ASCEND_DUMMY_WASTE_TIMING_SYNC", "0"):
                    try:
                        torch.npu.synchronize()
                    except Exception:
                        pass
                dummy_moe_start = time.perf_counter()
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
            if dummy_moe_timing_enabled:
                dummy_moe_wall_ms = (
                    time.perf_counter() - dummy_moe_start) * 1e3
                dummy_moe_sync_wall_ms = 0.0
                post_sync_overhead_ms = 0.0
                if _env_flag("VLLM_ASCEND_DUMMY_WASTE_TIMING_SYNC", "0"):
                    sync_start = time.perf_counter()
                    try:
                        torch.npu.synchronize()
                    except Exception:
                        pass
                    dummy_moe_sync_wall_ms = (
                        time.perf_counter() - dummy_moe_start) * 1e3
                    post_sync_overhead_ms = (
                        time.perf_counter() - sync_start) * 1e3
                    # Keep the explicit post-op sync overhead visible for
                    # debugging without folding it into the default wall metric.
                dummy_waste_timing["moe_post_sync_overhead_ms"] = (
                        dummy_waste_timing.get(
                            "moe_post_sync_overhead_ms", 0.0)
                        + post_sync_overhead_ms)
                selected_topk_count = int(
                    dummy_waste_timing.pop("_current_moe_selected_topk", 0))
                selected_token_count = int(
                    dummy_waste_timing.pop("_current_moe_selected_tokens", 0))
                selected_expert_count = int(
                    dummy_waste_timing.pop("_current_moe_selected_experts", 0))
                selected_layer = (
                    selected_topk_count > 0
                    or not dummy_selection_stats_enabled)
                dummy_waste_timing["moe_wall_ms"] = (
                    dummy_waste_timing.get("moe_wall_ms", 0.0)
                    + dummy_moe_wall_ms)
                dummy_waste_timing["moe_total_sync_wall_ms"] = (
                    dummy_waste_timing.get("moe_total_sync_wall_ms", 0.0)
                    + dummy_moe_sync_wall_ms)
                if selected_layer:
                    dummy_waste_timing["moe_sync_wall_ms"] = (
                        dummy_waste_timing.get("moe_sync_wall_ms", 0.0)
                        + dummy_moe_sync_wall_ms)
                    dummy_waste_timing["moe_selected_layers"] = (
                        int(dummy_waste_timing.get("moe_selected_layers", 0))
                        + 1)
                    dummy_waste_timing["moe_selected_topk"] = (
                        int(dummy_waste_timing.get("moe_selected_topk", 0))
                        + selected_topk_count)
                    dummy_waste_timing["moe_selected_tokens"] = (
                        int(dummy_waste_timing.get("moe_selected_tokens", 0))
                        + selected_token_count)
                    dummy_waste_timing["moe_selected_experts"] = (
                        int(dummy_waste_timing.get("moe_selected_experts", 0))
                        + selected_expert_count)
                else:
                    dummy_waste_timing["moe_unselected_layers"] = (
                        int(dummy_waste_timing.get("moe_unselected_layers", 0))
                        + 1)
                    dummy_waste_timing["moe_unselected_sync_wall_ms"] = (
                        dummy_waste_timing.get(
                            "moe_unselected_sync_wall_ms", 0.0)
                        + dummy_moe_sync_wall_ms)
                dummy_waste_timing["moe_layers"] = (
                    int(dummy_waste_timing.get("moe_layers", 0)) + 1)
                dummy_waste_timing["moe_tokens"] = (
                    int(dummy_waste_timing.get("moe_tokens", 0))
                    + int(hidden_states.shape[0]))
            if debug_enabled:
                self.elastic_debug_budget = max(0, self.elastic_debug_budget - 1)

            if shared_experts:
                return final_hidden_states, shared_hidden_states
            else:
                return final_hidden_states
        finally:
            if forced_hybrid_comm_alignment or mode1_host_alltoall_metadata:
                forward_context.moe_comm_method = original_moe_comm_method
                forward_context.moe_comm_type = original_moe_comm_type
                forward_context.fused_moe_state = original_fused_moe_state
                forward_context.hybrid_force_host_alltoall_metadata = (
                    original_hybrid_host_metadata)
                forward_context.hybrid_stage_active_ranks = (
                    original_hybrid_stage)

    def reset_expert_map_and_log2phy(self):
        self.clear_lossless_hybrid_state()
        logical_num_experts = int(
            getattr(self, "elastic_original_num_experts", self.num_experts))
        self.num_experts = logical_num_experts
        self.moe_config.num_experts = logical_num_experts
        target_device = None
        if self.log2phy is not None:
            target_device = self.log2phy.device
        elif self.expert_map is not None:
            target_device = self.expert_map.device
        elif hasattr(self, "w13_weight"):
            target_device = self.w13_weight.device
        if target_device is not None and target_device.type == "cpu" and hasattr(
                torch, "npu") and torch.npu.is_available():
            target_device = torch.device("npu", torch.npu.current_device())
        log2phy_dtype = self.log2phy.dtype if self.log2phy is not None else torch.int32
        if self.elastic_moe_mode == "lossless":
            self.loaded_local_num_experts, self.loaded_expert_map = determine_redundant_replica_expert_map(
                logical_num_experts, self.ep_size, self.ep_rank,
                self.global_redundant_expert_num)
            active_mapping = determine_expert_map(
                self.ep_size,
                self.ep_rank,
                logical_num_experts,
                layer_idx=self.layer_idx)
            if len(active_mapping) == 3:
                (self.active_local_num_experts, self.expert_map,
                 self.log2phy) = active_mapping
            elif len(active_mapping) == 2:
                self.active_local_num_experts, self.expert_map = active_mapping
                self.log2phy = determine_default_log2phy_map(
                    logical_num_experts, self.ep_size, self.ep_rank, 0)
            else:
                raise ValueError(
                    f"Unexpected determine_expert_map return arity={len(active_mapping)}")
            if target_device is not None:
                if self.loaded_expert_map is not None:
                    self.loaded_expert_map = self.loaded_expert_map.to(
                        device=target_device, dtype=torch.int32)
                if self.expert_map is not None:
                    self.expert_map = self.expert_map.to(
                        device=target_device, dtype=torch.int32)
                if self.log2phy is not None:
                    self.log2phy = self.log2phy.to(device=target_device,
                                                   dtype=log2phy_dtype)
            self.primary_log2phy = self.log2phy.clone()
            self.local_num_experts = self.active_local_num_experts
            self.moe_config.num_local_experts = self.active_local_num_experts
            self.elastic_runtime_log2phy = None
            self.refresh_lossless_runtime_weights()
            if self.elastic_execution_mode == 1:
                self.lossless_mode1_native_parity_ready = (
                    self.runtime_w13_buffer is None
                    and self.runtime_w2_buffer is None
                    and self.runtime_w13_weight is not None
                    and self.runtime_w2_weight is not None
                )
            return
        else:
            _, expert_map = determine_default_expert_map(
                self.global_num_experts, self.ep_size, self.ep_rank,
                self.global_redundant_expert_num)
            log2phy = determine_default_log2phy_map(
                self.global_num_experts, self.ep_size, self.ep_rank,
                self.global_redundant_expert_num)
            if target_device is not None:
                if expert_map is not None:
                    expert_map = expert_map.to(device=target_device,
                                               dtype=torch.int32)
                log2phy = log2phy.to(device=target_device,
                                     dtype=log2phy_dtype)
            elif hasattr(torch, "npu") and torch.npu.is_available():
                log2phy = log2phy.npu()

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
