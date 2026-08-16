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
import gc
import math
import os
import threading
import time
from datetime import timedelta
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Any, Optional, Union

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
from vllm_ascend.ops.moe.moe_comm_method import (
    get_moe_comm_method, get_moe_comm_method_topology_cache_stats,
    prune_moe_comm_method_topology_cache,
    release_moe_comm_method_runtime_state, reset_moe_comm_method_cache,
    setup_moe_comm_method)
from vllm_ascend.platform import NPUPlatform
from vllm_ascend.shrink_aware.planner import (parse_rank_list,
                                              parse_stage_survivor_ranks)
from vllm_ascend.utils import (init_ascend_soc_version,
                               register_ascend_customop, sleep_mode_enabled,
                               try_register_lib)
from vllm_ascend.worker.npu_input_batch import InputBatch
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


def _is_ascend_fused_moe_module(module: nn.Module) -> bool:
    try:
        from vllm_ascend.ops.fused_moe import AscendFusedMoE
        if isinstance(module, AscendFusedMoE):
            return True
    except Exception:
        pass
    try:
        from vllm_ascend.ops.common_fused_moe import AscendFusedMoE as CommonAscendFusedMoE
        return isinstance(module, CommonAscendFusedMoE)
    except Exception:
        return False


def _mode4_new_like_rows(reference: torch.Tensor,
                         rows: int) -> torch.Tensor:
    """Allocate a same-format NPU batch for mode4 expert P2P payloads."""
    rows = int(rows)
    tensor = reference.new_empty((rows, ) + tuple(reference.shape[1:]))
    if reference.device.type != "npu":
        return tensor
    try:
        target_format = torch_npu.get_npu_format(reference)
        if torch_npu.get_npu_format(tensor) != target_format:
            tensor = torch_npu.npu_format_cast(tensor, target_format)
    except Exception:
        # Fall back to the default NPU layout; copy_ still preserves values.
        pass
    return tensor


def _mode4_tensor_signature(reference: torch.Tensor,
                            rows: int) -> tuple[object, ...]:
    fmt = None
    if reference.device.type == "npu":
        try:
            fmt = torch_npu.get_npu_format(reference)
        except Exception:
            fmt = None
    return (str(reference.device), str(reference.dtype), tuple(reference.shape),
            int(rows), str(fmt))


def _mode4_flat_payload_signature(w13: torch.Tensor, w2: torch.Tensor,
                                  rows: int) -> tuple[object, ...]:
    return ("flat", _mode4_tensor_signature(w13, rows),
            _mode4_tensor_signature(w2, rows), int(w13.numel()),
            int(w2.numel()))


def _mode5_cpu_shadow_runtime_strategy() -> str:
    return ("legacy_cpu_shadow" if os.getenv(
        "VLLM_ASCEND_MODE5_USE_LEGACY_CPU_SHADOW_RUNTIME",
        "0").lower() in ("1", "true", "yes", "on") else "dual_source")


def _mode5_single_control_message_remote() -> bool:
    return os.getenv(
        "VLLM_ASCEND_MODE5_SINGLE_CONTROL_MESSAGE_REMOTE",
        "1").lower() in ("1", "true", "yes", "on")


def _mode1_kv_resize_live_tensor_scan_enabled() -> bool:
    return os.getenv(
        "VLLM_ASCEND_MODE1_KV_RESIZE_LIVE_TENSOR_SCAN",
        "0").lower() in ("1", "true", "yes", "on")


def _tensor_storage_key_and_nbytes(tensor: torch.Tensor) -> tuple[int, int]:
    try:
        storage = tensor.untyped_storage()
        return int(storage.data_ptr()), int(storage.nbytes())
    except Exception:
        try:
            return int(tensor.data_ptr()), int(tensor.numel() * tensor.element_size())
        except Exception:
            return 0, 0


def _tensor_tree_storage_nbytes(value: object,
                                seen: set[int] | None = None) -> int:
    if seen is None:
        seen = set()
    if torch.is_tensor(value):
        ptr, nbytes = _tensor_storage_key_and_nbytes(value)
        if ptr and ptr not in seen:
            seen.add(ptr)
            return nbytes
        return 0
    if isinstance(value, dict):
        return sum(_tensor_tree_storage_nbytes(item, seen)
                   for item in value.values())
    if isinstance(value, (list, tuple, set)):
        return sum(_tensor_tree_storage_nbytes(item, seen) for item in value)
    return 0


def _tensor_storage_summary(tensor: object) -> tuple[int, str, tuple[Any, ...]]:
    if not torch.is_tensor(tensor):
        return 0, "", ()
    ptr, nbytes = _tensor_storage_key_and_nbytes(tensor)
    if not ptr:
        return 0, "", ()
    return nbytes, str(tensor.dtype), tuple(tensor.shape)


def _mode1_referrer_owner_summary(ref: object,
                                  ignored_ref_ids: set[int],
                                  limit: int = 8) -> str:
    parts: list[str] = []
    try:
        referrers = gc.get_referrers(ref)
    except Exception:
        return ""
    for owner in referrers[:limit * 4]:
        if id(owner) in ignored_ref_ids:
            continue
        try:
            if owner is referrers:
                continue
            if getattr(owner, "__dict__", None) is ref:
                parts.append(f"obj({type(owner).__name__})")
            elif hasattr(owner, "f_code") and hasattr(owner, "f_lineno"):
                parts.append(
                    f"frame({getattr(owner.f_code, 'co_name', '?')}@"
                    f"{getattr(owner.f_code, 'co_filename', '?')}:"
                    f"{getattr(owner, 'f_lineno', '?')})")
            elif isinstance(owner, dict):
                owner_keys = []
                try:
                    for key, value in list(owner.items())[:64]:
                        if value is ref:
                            owner_keys.append(str(key))
                except Exception:
                    pass
                parts.append(f"dict(keys={owner_keys[:4]})")
            elif isinstance(owner, (list, tuple, set)):
                parts.append(f"{type(owner).__name__}(len={len(owner)})")
            else:
                parts.append(type(owner).__name__)
        except Exception:
            parts.append(type(owner).__name__)
        if len(parts) >= limit:
            break
    return ",".join(parts)


def _append_large_npu_tensor_paths(owner: object,
                                   prefix: str,
                                   out: list[tuple[int, str, str, tuple[Any,
                                                                        ...]]],
                                   *,
                                   min_bytes: int,
                                   seen_objects: set[int],
                                   depth: int = 0,
                                   max_depth: int = 3) -> None:
    if owner is None or depth > max_depth:
        return
    owner_id = id(owner)
    if owner_id in seen_objects:
        return
    seen_objects.add(owner_id)
    if torch.is_tensor(owner):
        try:
            if owner.device.type != "npu":
                return
            nbytes, dtype, shape = _tensor_storage_summary(owner)
            if nbytes >= min_bytes:
                out.append((nbytes, prefix, dtype, shape))
        except Exception:
            return
        return
    if isinstance(owner, dict):
        for key, value in list(owner.items()):
            _append_large_npu_tensor_paths(value,
                                           f"{prefix}.{key}",
                                           out,
                                           min_bytes=min_bytes,
                                           seen_objects=seen_objects,
                                           depth=depth + 1,
                                           max_depth=max_depth)
        return
    if isinstance(owner, (list, tuple, set)):
        for idx, value in enumerate(list(owner)):
            _append_large_npu_tensor_paths(value,
                                           f"{prefix}[{idx}]",
                                           out,
                                           min_bytes=min_bytes,
                                           seen_objects=seen_objects,
                                           depth=depth + 1,
                                           max_depth=max_depth)
        return
    obj_dict = getattr(owner, "__dict__", None)
    if not isinstance(obj_dict, dict):
        return
    for attr, value in list(obj_dict.items()):
        if attr in ("_modules", "_parameters", "_buffers"):
            continue
        _append_large_npu_tensor_paths(value,
                                       f"{prefix}.{attr}",
                                       out,
                                       min_bytes=min_bytes,
                                       seen_objects=seen_objects,
                                       depth=depth + 1,
                                       max_depth=max_depth)


def _release_mode1_double_buffer_owner(owner: object) -> int:
    if owner is None:
        return 0
    released = 0
    owners: list[object] = [owner]
    for attr in ("slots", "_mode3_cpu_npu_double_buffer_slots",
                 "_mode4_remote_npu_double_buffer_slots"):
        value = getattr(owner, attr, None)
        if value is not None:
            owners.append(value)
    for candidate in owners:
        if isinstance(candidate, dict):
            iterable = list(candidate.values())
        elif isinstance(candidate, (list, tuple, set)):
            iterable = list(candidate)
        else:
            iterable = [candidate]
        for item in iterable:
            if item is None:
                continue
            for attr in ("w13", "w2", "cpu_stage_w13", "cpu_stage_w2",
                         "expert_map", "dispatch_log2phy"):
                if hasattr(item, attr):
                    try:
                        if getattr(item, attr) is not None:
                            released += 1
                        setattr(item, attr, None)
                    except Exception:
                        pass
            for attr in ("block_layer_states", "prefetch_timing",
                         "last_bind_timing",
                         "last_prefetch_timing_by_target_layer",
                         "_dispatch_remap_cache", "_slot_expert_map_cache"):
                value = getattr(item, attr, None)
                if isinstance(value, dict):
                    try:
                        value.clear()
                    except Exception:
                        pass
    return released


def _refresh_mode1_eplb_adaptor_references(model_runner: object) -> int:
    refreshed = 0
    adaptors: list[object] = []
    adaptor = getattr(model_runner, "eplb_adaptor", None)
    if adaptor is not None:
        adaptors.append(adaptor)
    for owner_attr in ("eplb_loader", "eplb_updator"):
        owner = getattr(model_runner, owner_attr, None)
        adaptor = getattr(owner, "eplb_adaptor", None)
        if adaptor is not None:
            adaptors.append(adaptor)
        adaptor = getattr(owner, "adaptor", None)
        if adaptor is not None:
            adaptors.append(adaptor)

    seen: set[int] = set()
    for adaptor in adaptors:
        if id(adaptor) in seen:
            continue
        seen.add(id(adaptor))
        refresh_fn = getattr(adaptor, "refresh_parameter_references", None)
        if callable(refresh_fn):
            refresh_fn()
            refreshed += 1
            continue
        param_dict = getattr(adaptor, "param_dict", None)
        if isinstance(param_dict, dict):
            param_dict.clear()
            refreshed += 1
        expert_param = getattr(adaptor, "expert_param_per_layer", None)
        if isinstance(expert_param, dict):
            expert_param.clear()
    return refreshed


def _is_mode1_stale_fullname_expert_key(key: object) -> bool:
    if not isinstance(key, str):
        return False
    if not key.startswith("model.layers."):
        return False
    return (key.endswith(".mlp.experts.w13_weight")
            or key.endswith(".mlp.experts.w2_weight"))


def _clear_stale_mode1_fullname_parameter_dicts(model: nn.Module) -> tuple[int, int, int]:
    """Drop stale full-name parameter caches that retain pre-compaction MoE weights.

    Some loaders/adaptors keep a dict keyed by full parameter names, e.g.
    ``model.layers.0.mlp.experts.w13_weight``.  After mode=1 compacts the
    physical expert slots for a floor=world-size step, those dicts can retain
    the old 32-slot Parameters even though the live module now owns 8-slot
    Parameters.  They are not part of ``module._parameters`` (those use local
    names), so removing only stale full-name entries is safe and releases the
    old storages before rebuilding the larger KV cache.
    """
    try:
        current_params = dict(model.named_parameters())
    except Exception:
        current_params = {}

    cleared_dicts = 0
    cleared_entries = 0
    cleared_bytes = 0
    seen_dicts: set[int] = set()
    for obj in gc.get_objects():
        if not isinstance(obj, dict):
            continue
        obj_id = id(obj)
        if obj_id in seen_dicts:
            continue
        seen_dicts.add(obj_id)
        try:
            candidate_items = [
                (key, value) for key, value in list(obj.items())
                if _is_mode1_stale_fullname_expert_key(key)
            ]
        except Exception:
            continue
        if not candidate_items:
            continue

        removed_this_dict = 0
        for key, value in candidate_items:
            if value is current_params.get(key):
                continue
            if not torch.is_tensor(value):
                continue
            try:
                if value.device.type != "npu":
                    continue
            except Exception:
                continue
            try:
                shape = tuple(value.shape)
            except Exception:
                shape = ()
            if not shape or int(shape[0]) <= 8:
                continue
            _ptr, nbytes = _tensor_storage_key_and_nbytes(value)
            try:
                obj.pop(key, None)
            except Exception:
                continue
            cleared_entries += 1
            removed_this_dict += 1
            cleared_bytes += int(nbytes)
        if removed_this_dict:
            cleared_dicts += 1
    return cleared_dicts, cleared_entries, cleared_bytes


def _mode4_get_flat_payload(cache: dict[tuple[object, ...], torch.Tensor],
                            key: tuple[object, ...], w13: torch.Tensor,
                            w2: torch.Tensor, rows: int) -> torch.Tensor:
    payload = cache.get(key)
    if payload is not None:
        return payload
    rows = int(rows)
    total_elems = rows * (int(w13[0].numel()) + int(w2[0].numel()))
    payload = w13.new_empty((total_elems, ))
    cache[key] = payload
    return payload


def _mode4_get_request_cpu_buffer(
        cache: dict[tuple[object, ...], torch.Tensor],
        key: tuple[object, ...],
        rows: int,
        cols: int) -> torch.Tensor:
    request = cache.get(key)
    if request is not None:
        return request
    request = torch.empty((int(rows), int(cols)),
                          device="cpu",
                          dtype=torch.int64)
    cache[key] = request
    return request


def _mode4_build_request_cpu_tensor(
        rows: list[tuple[int, int, int]] | list[tuple[int, int, int, int]],
        *,
        mode5_assignments: bool = False) -> torch.Tensor:
    if mode5_assignments:
        payload = [[int(layer_idx), int(remote_slot), int(expert_id)]
                   for _slot_idx, remote_slot, expert_id, layer_idx in rows]
    else:
        payload = [[int(layer_idx), int(remote_slot), int(expert_id)]
                   for layer_idx, remote_slot, expert_id in rows]
    return torch.tensor(payload, device="cpu", dtype=torch.int64)


def _mode4_pack_remote_request_rows_to_flat_payload(
        batch_w13: torch.Tensor,
        batch_w2: torch.Tensor,
        request_rows: list[tuple[int, int, int]],
        layers: dict[int, nn.Module],
        expert_cache: dict[tuple[int, int], tuple[torch.Tensor,
                                                  torch.Tensor]]) -> None:
    """Pack remote request rows into a flat payload using contiguous slot runs.

    For mode5 the request rows are often already grouped by remote cache rank
    and ordered by remote_slot. When that happens, copy contiguous
    ``module.w13_weight/module.w2_weight`` ranges directly instead of falling
    back to one row copy per expert.
    """
    row_start = 0
    total_rows = len(request_rows)
    while row_start < total_rows:
        layer_idx, remote_slot, _expert_id = request_rows[row_start]
        run_end = row_start + 1
        prev_slot = int(remote_slot)
        while run_end < total_rows:
            next_layer_idx, next_remote_slot, _next_expert_id = request_rows[
                run_end]
            if (int(next_layer_idx) != int(layer_idx)
                    or int(next_remote_slot) != prev_slot + 1):
                break
            prev_slot = int(next_remote_slot)
            run_end += 1
        run_len = run_end - row_start
        module = layers.get(int(layer_idx))
        used_bulk = False
        if module is not None:
            try:
                src_w13 = module.w13_weight[int(remote_slot):int(remote_slot) +
                                            run_len]
                src_w2 = module.w2_weight[int(remote_slot):int(remote_slot) +
                                          run_len]
                if (tuple(src_w13.shape)
                        == tuple(batch_w13[row_start:run_end].shape)
                        and tuple(src_w2.shape)
                        == tuple(batch_w2[row_start:run_end].shape)
                        and src_w13.device.type == "npu"
                        and src_w2.device.type == "npu"):
                    batch_w13[row_start:run_end].copy_(src_w13,
                                                       non_blocking=False)
                    batch_w2[row_start:run_end].copy_(src_w2,
                                                      non_blocking=False)
                    used_bulk = True
            except Exception:
                used_bulk = False
        if not used_bulk:
            for row_idx in range(row_start, run_end):
                cur_layer_idx, cur_remote_slot, _cur_expert_id = request_rows[
                    row_idx]
                send_w13, send_w2 = expert_cache[(int(cur_layer_idx),
                                                  int(cur_remote_slot))]
                batch_w13[row_idx:row_idx + 1].copy_(send_w13,
                                                     non_blocking=False)
                batch_w2[row_idx:row_idx + 1].copy_(send_w2,
                                                    non_blocking=False)
        row_start = run_end


def _is_custom_ascend_fused_moe_module(module: nn.Module) -> bool:
    try:
        from vllm_ascend.ops.fused_moe import AscendFusedMoE
        return isinstance(module, AscendFusedMoE)
    except Exception:
        return False


def _is_elastic_headroom_moe_module(module: nn.Module) -> bool:
    if _is_ascend_fused_moe_module(module):
        return True
    try:
        from vllm.model_executor.layers.fused_moe.layer import FusedMoE
        return isinstance(module, FusedMoE)
    except Exception:
        return False


def _module_uses_lossless_elastic(module: nn.Module) -> bool:
    module_mode = getattr(module, "elastic_moe_mode", None)
    if module_mode is not None:
        return module_mode == "lossless"
    return (envs_ascend.VLLM_ASCEND_ELASTIC_MOE_MODE == "lossless"
            and envs_ascend.VLLM_ASCEND_ELASTIC_EXECUTION_MODE in
            (1, 2, 3, 4, 5))


def _module_configured_elastic_floor(module: nn.Module) -> Optional[int]:
    get_floor = getattr(module, "_get_configured_elastic_min_compute_group_size",
                        None)
    if callable(get_floor):
        return get_floor()
    return envs_ascend.VLLM_ASCEND_ELASTIC_MIN_COMPUTE_GROUP_SIZE


def _module_initial_ep_size(module: nn.Module) -> int:
    return int(getattr(module, "elastic_original_ep_size",
                       getattr(module, "ep_size", 1)))


def _remote_npu_runtime_elastic_floor() -> Optional[int]:
    mode = int(envs_ascend.VLLM_ASCEND_ELASTIC_EXECUTION_MODE)
    runtime_floor = None
    if mode == 5:
        runtime_floor = envs_ascend.VLLM_ASCEND_MODE5_RUNTIME_MIN_COMPUTE_GROUP_SIZE
    if runtime_floor is None:
        runtime_floor = envs_ascend.VLLM_ASCEND_MODE4_RUNTIME_MIN_COMPUTE_GROUP_SIZE
    if runtime_floor is not None:
        return int(runtime_floor)
    return envs_ascend.VLLM_ASCEND_ELASTIC_MIN_COMPUTE_GROUP_SIZE


def _mode4_runtime_elastic_floor() -> Optional[int]:
    return _remote_npu_runtime_elastic_floor()


def _mode5_remote_expert_fraction() -> float:
    fraction, _debug = _mode5_remote_expert_fraction_with_debug()
    return fraction


def _mode5_remote_expert_fraction_with_debug() -> tuple[float, str]:
    policy = os.getenv("VLLM_ASCEND_MODE5_REMOTE_EXPERT_FRACTION_POLICY",
                       "fixed").strip().lower()
    raw = os.getenv("VLLM_ASCEND_MODE5_REMOTE_EXPERT_FRACTION", "0.5")
    try:
        fixed_fraction = float(raw)
    except ValueError as exc:
        raise ValueError(
            "VLLM_ASCEND_MODE5_REMOTE_EXPERT_FRACTION must be a float in [0, 1], "
            f"got {raw!r}.") from exc
    fixed_fraction = min(max(fixed_fraction, 0.0), 1.0)
    if policy in ("", "fixed", "manual", "env"):
        return fixed_fraction, f"policy=fixed fraction={fixed_fraction:.4f}"
    if policy not in ("comm", "comm_efficiency", "bandwidth", "bw"):
        raise ValueError(
            "VLLM_ASCEND_MODE5_REMOTE_EXPERT_FRACTION_POLICY must be one of "
            "fixed/comm_efficiency, got "
            f"{policy!r}.")

    def _parse_positive_float_series(env_name: str) -> list[float]:
        raw_series = os.getenv(env_name, "").strip()
        if not raw_series:
            return []
        values: list[float] = []
        for token in raw_series.replace(";", ",").split(","):
            token = token.strip()
            if not token:
                continue
            try:
                value = float(token)
            except ValueError as exc:
                raise ValueError(
                    f"{env_name} must contain comma-separated floats, got "
                    f"{raw_series!r}.") from exc
            if value <= 0.0:
                raise ValueError(
                    f"{env_name} values must be > 0, got {value!r} in "
                    f"{raw_series!r}.")
            values.append(value)
        return values

    remote_comm_ms_series = _parse_positive_float_series(
        "VLLM_ASCEND_MODE5_REMOTE_COMM_MS_SERIES")
    cpu_comm_ms_series = _parse_positive_float_series(
        "VLLM_ASCEND_MODE5_CPU_COMM_MS_SERIES")
    if not remote_comm_ms_series or not cpu_comm_ms_series:
        raise ValueError(
            "Mode5 comm_efficiency policy requires both "
            "VLLM_ASCEND_MODE5_REMOTE_COMM_MS_SERIES and "
            "VLLM_ASCEND_MODE5_CPU_COMM_MS_SERIES.")
    if len(remote_comm_ms_series) != len(cpu_comm_ms_series):
        raise ValueError(
            "Mode5 comm_efficiency policy requires the remote/cpu series to "
            "have the same length: "
            f"remote={len(remote_comm_ms_series)} "
            f"cpu={len(cpu_comm_ms_series)}")
    bandwidth_ratios = [
        cpu_ms / remote_ms for remote_ms, cpu_ms in zip(
            remote_comm_ms_series, cpu_comm_ms_series)
    ]
    bandwidth_ratio = sum(bandwidth_ratios) / len(bandwidth_ratios)
    fraction = bandwidth_ratio / (1.0 + bandwidth_ratio)
    clamp_min_raw = os.getenv(
        "VLLM_ASCEND_MODE5_REMOTE_EXPERT_FRACTION_CLAMP_MIN", "0.0")
    clamp_max_raw = os.getenv(
        "VLLM_ASCEND_MODE5_REMOTE_EXPERT_FRACTION_CLAMP_MAX", "1.0")
    try:
        clamp_min = float(clamp_min_raw)
        clamp_max = float(clamp_max_raw)
    except ValueError as exc:
        raise ValueError(
            "VLLM_ASCEND_MODE5_REMOTE_EXPERT_FRACTION_CLAMP_MIN/MAX must be "
            f"floats, got min={clamp_min_raw!r} max={clamp_max_raw!r}.") from exc
    if clamp_min > clamp_max:
        raise ValueError(
            "VLLM_ASCEND_MODE5_REMOTE_EXPERT_FRACTION_CLAMP_MIN must be <= "
            f"MAX, got min={clamp_min} max={clamp_max}.")
    fraction = min(max(fraction, clamp_min), clamp_max)
    return (
        fraction,
        "policy=comm_efficiency "
        f"mean_bw_ratio={bandwidth_ratio:.4f} "
        f"samples={len(bandwidth_ratios)} "
        f"clamp=[{clamp_min:.4f},{clamp_max:.4f}] "
        f"fraction={fraction:.4f}")


def _mode5_balance_remote_source_fanout() -> bool:
    raw = os.getenv("VLLM_ASCEND_MODE5_BALANCE_REMOTE_SOURCE_FANOUT",
                    "0").strip().lower()
    return raw in ("1", "true", "yes", "on")


def _mode5_remote_candidate_sort_key(
        expert_id: int,
        target_rank: int,
        remote_source_rank_by_expert: dict[int, int],
        selected_target_ranks_by_source: dict[int, set[int]],
        selected_remote_count_by_source: dict[int, int],
        selected_remote_count_by_edge: dict[tuple[int, int], int]
) -> tuple[int, int, int, int, int, int]:
    source_rank = remote_source_rank_by_expert.get(int(expert_id))
    if source_rank is None:
        return (2, 1 << 30, 1 << 30, 0, 1 << 30, int(expert_id))
    selected_targets = selected_target_ranks_by_source.get(int(source_rank),
                                                           set())
    opens_new_edge = 0 if int(target_rank) in selected_targets else 1
    projected_fanout = len(selected_targets) + opens_new_edge
    source_remote_count = selected_remote_count_by_source.get(
        int(source_rank), 0)
    edge_remote_count = selected_remote_count_by_edge.get(
        (int(source_rank), int(target_rank)), 0)
    return (
        opens_new_edge,
        projected_fanout,
        source_remote_count if opens_new_edge else 0,
        -edge_remote_count,
        source_remote_count,
        int(expert_id),
    )


def _mode5_select_remote_experts_by_target(
        cpu_import_source_rank: dict[int, int],
        cpu_import_target_rank: dict[int, int],
        eligible_remote_experts: Optional[set[int]] = None,
        required_remote_experts: Optional[set[int]] = None,
        remote_source_rank_by_expert: Optional[dict[int, int]] = None
) -> set[int]:
    fraction = _mode5_remote_expert_fraction()
    required_remote_experts = set(
        int(x) for x in (required_remote_experts or set()))
    if fraction <= 0.0 and not required_remote_experts:
        return set()
    remote_source_rank_by_expert = remote_source_rank_by_expert or {}
    by_target: dict[int, list[int]] = {}
    for expert_id, source_rank in cpu_import_source_rank.items():
        expert_id = int(expert_id)
        target_rank = cpu_import_target_rank.get(expert_id)
        if source_rank is None or target_rank is None:
            continue
        by_target.setdefault(int(target_rank), []).append(expert_id)
    selected: set[int] = set()
    for _target_rank, target_experts in by_target.items():
        expert_ids = sorted(int(expert_id) for expert_id in target_experts)
        if not expert_ids:
            continue
        count = int(round(len(expert_ids) * fraction))
        if fraction > 0.0:
            count = max(1, count)
        count = min(len(expert_ids), count)
        required_for_target = [
            expert_id for expert_id in expert_ids
            if expert_id in required_remote_experts
        ]
        if len(required_for_target) > count:
            raise RuntimeError(
                "Mode5 strict ratio violated: required remote experts exceed "
                f"remote quota for target={_target_rank}. quota={count} "
                f"required={len(required_for_target)} "
                f"required_head={required_for_target[:16]}")
        selected.update(required_for_target)
        remaining_candidates = [
            expert_id for expert_id in expert_ids
            if expert_id not in required_remote_experts
            and (eligible_remote_experts is None
                 or expert_id in eligible_remote_experts)
        ]
        if (_mode5_prefer_same_package_remote()
                and remote_source_rank_by_expert):
            target_package = _mode5_package_id(int(_target_rank))
            remaining_candidates = sorted(
                remaining_candidates,
                key=lambda expert_id: (
                    0 if _mode5_package_id(
                        remote_source_rank_by_expert.get(
                            int(expert_id),
                            cpu_import_source_rank.get(int(expert_id),
                                                       -1))) == target_package else 1,
                    int(remote_source_rank_by_expert.get(
                        int(expert_id),
                        cpu_import_source_rank.get(int(expert_id), -1))),
                    int(expert_id),
                ))
        remaining_quota = max(0, count - len(required_for_target))
        if len(required_for_target) + len(remaining_candidates) < count:
            raise RuntimeError(
                "Mode5 strict ratio cannot satisfy remote quota with the "
                f"available remote-NPU experts for target={_target_rank}. "
                f"quota={count} required={len(required_for_target)} "
                f"eligible_remaining={len(remaining_candidates)} "
                f"eligible_head={remaining_candidates[:16]}")
        if remaining_quota <= 0:
            continue
        take = remaining_candidates[:remaining_quota]
        selected.update(take)
    missing_required = required_remote_experts.difference(selected)
    if missing_required:
        raise RuntimeError(
            "Mode5 required remote-NPU experts are not eligible for remote "
            f"fetch: missing={sorted(missing_required)[:16]} "
            f"count={len(missing_required)}")
    return selected


def _mode5_strict_partition_by_target(
        import_source_rank_by_expert: dict[int, int],
        import_target_rank_by_expert: dict[int, int],
        eligible_remote_experts: Optional[set[int]] = None,
        remote_source_rank_by_expert: Optional[dict[int, int]] = None
) -> tuple[set[int], set[int]]:
    """Split missing experts per target rank by configured remote fraction.

    Mode5 strict-ratio semantics are intentionally simple:
    - consider only experts that are missing locally after shrink
    - for each surviving target rank, choose exactly round(missing * fraction)
      experts for remote-NPU fetch
    - the remainder are served through the CPU import path

    Keep the original source rank mapping unchanged. CPU/NPU *location* is a
    transport choice here, not an ownership rewrite.
    """
    if not import_source_rank_by_expert:
        return set(), set()
    fraction = _mode5_remote_expert_fraction()
    remote_source_rank_by_expert = {
        int(expert_id): int(source_rank)
        for expert_id, source_rank in (remote_source_rank_by_expert
                                       or {}).items()
        if source_rank is not None
    }
    by_target: dict[int, list[int]] = {}
    for expert_id, source_rank in import_source_rank_by_expert.items():
        expert_id = int(expert_id)
        target_rank = import_target_rank_by_expert.get(expert_id)
        if source_rank is None or target_rank is None:
            continue
        by_target.setdefault(int(target_rank), []).append(expert_id)

    target_specs: list[tuple[int, list[int], int, list[int]]] = []
    for _target_rank, target_experts in sorted(by_target.items()):
        expert_ids = sorted(int(expert_id) for expert_id in target_experts)
        import_total = len(expert_ids)
        if import_total == 0:
            continue
        remote_quota = int(math.floor(import_total * fraction + 0.5))
        remote_quota = max(0, min(import_total, remote_quota))
        remote_candidates = [
            expert_id for expert_id in expert_ids
            if eligible_remote_experts is None
            or expert_id in eligible_remote_experts
        ]
        if len(remote_candidates) < remote_quota:
            raise RuntimeError(
                "Mode5 strict ratio cannot satisfy remote quota with the "
                f"available remote-NPU experts for target={_target_rank}. "
                f"quota={remote_quota} eligible={len(remote_candidates)} "
                f"eligible_head={remote_candidates[:16]} "
                f"all_head={expert_ids[:16]}")
        target_specs.append(
            (int(_target_rank), expert_ids, int(remote_quota),
             remote_candidates))

    balance_remote_source = (_mode5_balance_remote_source_fanout()
                             and bool(remote_source_rank_by_expert))
    if balance_remote_source:
        target_specs.sort(
            key=lambda item: (
                len(item[3]) - item[2],
                len({
                    remote_source_rank_by_expert.get(expert_id, -1)
                    for expert_id in item[3]
                }),
                -item[2],
                item[0],
            ))

    remote_selected: set[int] = set()
    cpu_selected: set[int] = set()
    selected_target_ranks_by_source: dict[int, set[int]] = {}
    selected_remote_count_by_source: dict[int, int] = {}
    selected_remote_count_by_edge: dict[tuple[int, int], int] = {}
    for target_rank, expert_ids, remote_quota, remote_candidates in target_specs:
        if balance_remote_source and remote_quota > 0:
            candidate_pool = list(remote_candidates)
            chosen_remote: list[int] = []
            while len(chosen_remote) < remote_quota:
                best_idx = min(
                    range(len(candidate_pool)),
                    key=lambda idx: _mode5_remote_candidate_sort_key(
                        candidate_pool[idx], target_rank,
                        remote_source_rank_by_expert,
                        selected_target_ranks_by_source,
                        selected_remote_count_by_source,
                        selected_remote_count_by_edge))
                expert_id = int(candidate_pool.pop(best_idx))
                chosen_remote.append(expert_id)
                source_rank = remote_source_rank_by_expert.get(expert_id)
                if source_rank is None:
                    continue
                selected_target_ranks_by_source.setdefault(
                    int(source_rank), set()).add(int(target_rank))
                selected_remote_count_by_source[int(source_rank)] = (
                    selected_remote_count_by_source.get(int(source_rank), 0)
                    + 1)
                edge = (int(source_rank), int(target_rank))
                selected_remote_count_by_edge[edge] = (
                    selected_remote_count_by_edge.get(edge, 0) + 1)
            remote_ids = set(chosen_remote)
        else:
            remote_ids = set(remote_candidates[:remote_quota])
        remote_selected.update(remote_ids)
        cpu_ids = set(expert_ids)
        cpu_ids.difference_update(remote_ids)
        cpu_selected.update(cpu_ids)
    return remote_selected, cpu_selected


def _mode5_filter_cpu_payload(payload: dict) -> dict:
    remote_experts = set(int(x) for x in payload.get("mode5_remote_experts", []))
    cpu_import_experts_raw = payload.get("mode5_cpu_import_experts", None)
    cpu_import_experts = (
        None if cpu_import_experts_raw is None
        else set(int(x) for x in cpu_import_experts_raw))
    if not remote_experts and cpu_import_experts is None:
        return payload
    filtered = dict(payload)
    for key in ("cpu_import_source_rank", "cpu_import_target_rank",
                "remote_import_source_slot"):
        values = payload.get(key, {})
        filtered[key] = {
            int(expert_id): value
            for expert_id, value in values.items()
            if (int(expert_id) not in remote_experts
                and (cpu_import_experts is None
                     or int(expert_id) in cpu_import_experts))
        }
    return filtered


def _mode4_remote_source_rank_map(payload: dict) -> dict[int, int]:
    return {
        int(expert_id): int(source_rank)
        for expert_id, source_rank in payload.get(
            "mode4_remote_source_rank", payload.get("cpu_import_source_rank",
                                                    {})).items()
    }


def _mode5_prefer_same_package_remote() -> bool:
    return os.getenv("VLLM_ASCEND_MODE5_PREFER_SAME_PACKAGE_REMOTE",
                     "1").lower() in ("1", "true", "yes", "on")


def _mode5_package_id(rank: int) -> int:
    return int(rank) // 2


def _mode5_remote_expert_id_set(payload: dict) -> set[int]:
    return set(int(expert_id)
               for expert_id in payload.get("mode5_remote_experts", []))


def _mode5_remote_cache_owner_edges(payload: dict) -> set[tuple[int, int]]:
    target_rank_by_expert = payload.get("cpu_import_target_rank", {})
    remote_source_rank_by_expert = _mode4_remote_source_rank_map(payload)
    remote_source_slot_by_expert = payload.get("remote_import_source_slot", {})
    owner_edges: set[tuple[int, int]] = set()
    for expert_id, source_rank in remote_source_rank_by_expert.items():
        expert_id = int(expert_id)
        target_rank = target_rank_by_expert.get(expert_id)
        source_slot = remote_source_slot_by_expert.get(expert_id)
        if source_rank is None or target_rank is None or source_slot is None:
            continue
        source_rank = int(source_rank)
        target_rank = int(target_rank)
        if source_rank == target_rank or int(source_slot) < 0:
            continue
        owner_edges.add((target_rank, source_rank))
    return owner_edges


def _module_followup_shrink_enabled(module: nn.Module) -> bool:
    shrink_enabled = getattr(module, "_is_followup_shrink_enabled", None)
    if callable(shrink_enabled):
        return bool(shrink_enabled())
    initial_ep_size = _module_initial_ep_size(module)
    if initial_ep_size <= 1:
        return False
    configured_floor = _module_configured_elastic_floor(module)
    if configured_floor is None:
        return True
    return int(configured_floor) < initial_ep_size


def _module_has_preallocated_redundant_slots(module: nn.Module) -> bool:
    if int(getattr(module, "global_redundant_expert_num", 0) or 0) > 0:
        return True
    active_slots = int(getattr(module, "local_num_experts",
                               getattr(module, "active_local_num_experts", 0))
                       or 0)
    loaded_slots = int(getattr(module, "local_num_expert_weight_slots",
                               getattr(module, "loaded_weight_capacity",
                                       active_slots)) or 0)
    return loaded_slots > active_slots


def _module_hybrid_cpu_swap_enabled(module: nn.Module) -> bool:
    is_enabled = getattr(module, "_is_hybrid_cpu_swap_enabled", None)
    return bool(callable(is_enabled) and is_enabled())


def _module_is_custom_mode1_redundant_static(module: nn.Module) -> bool:
    if not _is_ascend_fused_moe_module(module):
        return False
    if int(getattr(module, "elastic_execution_mode", 0) or 0) != 1:
        return False
    if not _module_uses_lossless_elastic(module):
        return False
    if not _module_has_preallocated_redundant_slots(module):
        return False
    return not _module_hybrid_cpu_swap_enabled(module)


def _module_is_mode4_remote_npu_lightweight(module: nn.Module) -> bool:
    if not _is_ascend_fused_moe_module(module):
        return False
    if int(getattr(module, "elastic_execution_mode", 0) or 0) not in (4, 5):
        return False
    if not _module_uses_lossless_elastic(module):
        return False
    # Mode 4 intentionally keeps missing experts resident on inactive/cache
    # ranks and fetches them over NPU P2P. Treat it like the optimized mode=1
    # path for KV-cache sizing: do not subtract generic CPU/offload/headroom
    # budgets for workspaces that are either already profiled or lazily created
    # by the real decode path.
    return True


def _module_is_mode3_cross_layer_lightweight(module: nn.Module) -> bool:
    if not _is_ascend_fused_moe_module(module):
        return False
    if int(getattr(module, "elastic_execution_mode", 0) or 0) != 3:
        return False
    if not _module_uses_lossless_elastic(module):
        return False
    is_enabled = getattr(module, "_is_mode3_cross_layer_buffer_enabled", None)
    if not callable(is_enabled) or not is_enabled():
        return False
    # Mode 3 uses fixed double-buffer slots plus CPU shadow copies. The static
    # buffers are profiled before KV sizing, so charging the generic
    # zero-redundancy / post-restore headrooms again over-reserves HBM and can
    # force KV preemption, which then incorrectly drives the shrunken decode
    # path into AllToAll.
    return True


def _module_is_elastic_lightweight_no_headroom(module: nn.Module) -> bool:
    if _module_is_mode1_lightweight_parity(module):
        return True
    if (_module_is_mode4_remote_npu_lightweight(module)
            and not _env_flag("VLLM_ASCEND_MODE4_ENABLE_GENERIC_HEADROOM", "0")):
        return True
    return False


def _module_skips_generic_headroom(module: nn.Module) -> bool:
    if _module_is_elastic_lightweight_no_headroom(module):
        return True
    if (_module_is_mode3_cross_layer_lightweight(module)
            and not _env_flag("VLLM_ASCEND_MODE3_ENABLE_GENERIC_HEADROOM", "0")):
        return True
    return False


def _module_is_custom_mode1_floor8(module: nn.Module) -> bool:
    if not _module_is_custom_mode1_redundant_static(module):
        return False
    configured_floor = _module_configured_elastic_floor(module)
    return configured_floor is not None and int(configured_floor) == 8


def _module_is_custom_mode1_native_parity(
        module: nn.Module, max_floor: Optional[int] = None) -> bool:
    if not _is_custom_ascend_fused_moe_module(module):
        return False
    return _module_is_mode1_lightweight_parity(module, max_floor=max_floor)


def _module_is_mode1_lightweight_parity(
        module: nn.Module, max_floor: Optional[int] = None) -> bool:
    if not _is_ascend_fused_moe_module(module):
        return False
    if not _module_is_custom_mode1_redundant_static(module):
        return False
    if not _module_has_mode1_native_parity_ready(module):
        return False
    if max_floor is None:
        return True
    configured_floor = _module_configured_elastic_floor(module)
    return configured_floor is not None and int(configured_floor) <= max_floor


def _module_has_mode1_native_parity_ready(module: nn.Module) -> bool:
    return bool(getattr(module, "lossless_mode1_native_parity_ready", False))


def _set_module_active_expert_mask(
        module: nn.Module, new_active_expert_mask: Optional[torch.Tensor]
) -> None:
    setter = getattr(module, "set_active_expert_mask", None)
    if callable(setter):
        setter(new_active_expert_mask)
        return
    if new_active_expert_mask is None:
        setattr(module, "active_expert_mask", None)
        setattr(module, "elastic_runtime_log2phy", None)
        return
    expert_map = getattr(module, "expert_map", None)
    target_device = getattr(expert_map, "device", new_active_expert_mask.device)
    setattr(module, "active_expert_mask",
            new_active_expert_mask.to(device=target_device, dtype=torch.bool))


def _set_module_elastic_runtime_log2phy(
        module: nn.Module, new_log2phy: Optional[torch.Tensor]) -> None:
    setter = getattr(module, "set_elastic_runtime_log2phy", None)
    if callable(setter):
        setter(new_log2phy)
        return
    if new_log2phy is None:
        setattr(module, "elastic_runtime_log2phy", None)
        return
    log2phy = getattr(module, "log2phy", None)
    target_device = getattr(log2phy, "device", new_log2phy.device)
    target_dtype = getattr(log2phy, "dtype", new_log2phy.dtype)
    setattr(module, "elastic_runtime_log2phy",
            new_log2phy.to(device=target_device, dtype=target_dtype))


def _mode1_diag_tensor_row_stats(tensor: object,
                                 row: int,
                                 *,
                                 sample_values: int = 8) -> dict[str, object]:
    if tensor is None:
        return {"ok": False, "why": "none"}
    try:
        if row < 0 or row >= int(tensor.shape[0]):
            return {
                "ok": False,
                "why": f"row_oob:{row}/{int(tensor.shape[0])}",
            }
        flat = tensor[int(row)].reshape(-1).detach()
        sample_values = max(1, int(sample_values))
        total_values = int(flat.numel())
        if total_values <= sample_values:
            sampled = flat
            offsets = list(range(total_values))
        else:
            # Ascend formatted tensors can have sparse-looking prefixes after
            # npu_format_cast.  Sample across the row so a zero prefix is not
            # mistaken for a zero expert slot.
            offsets = sorted({
                int(round(i * (total_values - 1) / (sample_values - 1)))
                for i in range(sample_values)
            }) if sample_values > 1 else [0]
            sampled = flat[torch.tensor(offsets, device=flat.device)]
        vals = sampled.to(dtype=torch.float32, device="cpu")
        if vals.numel() == 0:
            return {"ok": False, "why": "empty"}
        abs_vals = vals.abs()
        return {
            "ok": True,
            "sum": round(float(vals.sum().item()), 6),
            "abs_mean": round(float(abs_vals.mean().item()), 6),
            "min": round(float(vals.min().item()), 6),
            "max": round(float(vals.max().item()), 6),
            "abs_max": round(float(abs_vals.max().item()), 6),
            "zero_like": bool(float(abs_vals.max().item()) == 0.0),
            "sample_offsets": offsets[:min(4, len(offsets))],
            "numel": total_values,
            "head": [
                round(float(v), 6)
                for v in vals[:min(4, int(vals.numel()))].tolist()
            ],
        }
    except Exception as exc:
        return {"ok": False, "why": repr(exc)}


def _mode1_diag_tensor_row_absdiff(lhs: object,
                                   lhs_row: int,
                                   rhs: object,
                                   rhs_row: int,
                                   *,
                                   sample_values: int = 8) -> Optional[float]:
    if lhs is None or rhs is None or lhs_row < 0 or rhs_row < 0:
        return None
    try:
        if lhs_row >= int(lhs.shape[0]) or rhs_row >= int(rhs.shape[0]):
            return None
        lflat = lhs[int(lhs_row)].reshape(-1).detach()
        rflat = rhs[int(rhs_row)].reshape(-1).detach()
        sample_values = max(1, int(sample_values))
        total_values = min(int(lflat.numel()), int(rflat.numel()))
        if total_values <= 0:
            return None
        if total_values <= sample_values:
            offsets = list(range(total_values))
        else:
            offsets = sorted({
                int(round(i * (total_values - 1) / (sample_values - 1)))
                for i in range(sample_values)
            }) if sample_values > 1 else [0]
        offset_tensor = torch.tensor(offsets, device=lflat.device)
        lvals = lflat[offset_tensor].to(dtype=torch.float32, device="cpu")
        rvals = rflat[offset_tensor.to(device=rflat.device)].to(
            dtype=torch.float32, device="cpu")
        count = min(int(lvals.numel()), int(rvals.numel()))
        if count <= 0:
            return None
        return round(float((lvals[:count] - rvals[:count]).abs().mean().item()),
                     6)
    except Exception:
        return None


def _mode1_diag_tensor_meta(tensor: object) -> dict[str, object]:
    if tensor is None:
        return {"ok": False, "why": "none"}
    try:
        meta: dict[str, object] = {
            "ok": True,
            "shape": tuple(getattr(tensor, "shape", ())),
            "dtype": str(getattr(tensor, "dtype", "unknown")),
            "device": str(getattr(tensor, "device", "unknown")),
            "storage_offset": int(tensor.storage_offset())
            if hasattr(tensor, "storage_offset") else None,
            "is_contiguous": bool(tensor.is_contiguous())
            if hasattr(tensor, "is_contiguous") else None,
        }
        if getattr(getattr(tensor, "device", None), "type", None) == "npu":
            try:
                meta["npu_format"] = int(torch_npu.get_npu_format(tensor))
            except Exception as exc:
                meta["npu_format_error"] = repr(exc)
        return meta
    except Exception as exc:
        return {"ok": False, "why": repr(exc)}


def _mode1_diag_tensor_rows_stats(tensor: object,
                                  rows: list[int],
                                  *,
                                  sample_values: int = 8
                                  ) -> list[dict[str, object]]:
    return [
        {
            "row": int(row),
            **_mode1_diag_tensor_row_stats(
                tensor,
                int(row),
                sample_values=sample_values,
            ),
        }
        for row in rows
    ]


def _module_mode1_lightweight_parity_path(module: nn.Module) -> str:
    if _is_custom_ascend_fused_moe_module(module):
        return "old_custom_fused_moe"
    return "native_common_fused_moe"


def _module_lightweight_no_headroom_path(module: nn.Module) -> str:
    if _module_is_mode4_remote_npu_lightweight(module):
        mode = int(getattr(module, "elastic_execution_mode", 0) or 0)
        return ("mode5_cpu_remote_npu_cache"
                if mode == 5 else "mode4_remote_npu_cache")
    if _module_is_mode3_cross_layer_lightweight(module):
        return "mode3_cross_layer_double_buffer"
    return _module_mode1_lightweight_parity_path(module)


def _env_flag(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).lower() in ("1", "true", "yes", "on")


def _env_int_set(name: str, default: str = "") -> set[int]:
    raw = os.getenv(name, default)
    values: set[int] = set()
    for part in str(raw).replace(";", ",").split(","):
        part = part.strip()
        if not part:
            continue
        try:
            values.add(int(part))
        except ValueError:
            continue
    return values


def _register_custom_models_if_requested() -> None:
    if not _env_flag("VLLM_ASCEND_REGISTER_CUSTOM_MODELS", "0"):
        return
    if _env_flag("VLLM_ASCEND_USE_NATIVE_QWEN3_MOE", "0"):
        logger.warning(
            "VLLM_ASCEND_REGISTER_CUSTOM_MODELS=1 was requested, but "
            "VLLM_ASCEND_USE_NATIVE_QWEN3_MOE is also enabled; Qwen3 MoE "
            "will stay on the native model path.")
        return

    import vllm_ascend
    vllm_ascend.register_model()

    try:
        from vllm.model_executor.model_loader import utils as loader_utils
        loader_utils._MODEL_ARCH_BY_HASH.clear()
    except Exception:
        logger.debug("Unable to clear model architecture cache.",
                     exc_info=True)
    try:
        from vllm.model_executor.models import registry as model_registry
        model_registry._try_load_model_cls.cache_clear()
        model_registry._try_inspect_model_cls.cache_clear()
    except Exception:
        logger.debug("Unable to clear model registry caches.", exc_info=True)

    try:
        from vllm import ModelRegistry
        qwen3_model = ModelRegistry.models.get("Qwen3MoeForCausalLM")
        logger.info("Registered custom Ascend models; Qwen3MoeForCausalLM=%s",
                    qwen3_model)
    except Exception:
        logger.debug("Unable to log Qwen3 model registry entry.",
                     exc_info=True)


def _mode4_keep_weights_out_of_sleep_pool() -> bool:
    return (envs_ascend.VLLM_ASCEND_ELASTIC_EXECUTION_MODE in (4, 5)
            and _env_flag("VLLM_ASCEND_MODE4_KEEP_WEIGHTS_OUT_OF_SLEEP_POOL",
                          "1"))


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
        _register_custom_models_if_requested()
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
        self._elastic_op_profile_context: Optional[dict] = None
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
        self._mode4_remote_recv_buffer_cache: dict[tuple[object, ...],
                                                   torch.Tensor] = {}
        self._mode4_remote_send_buffer_cache: dict[tuple[object, ...],
                                                   torch.Tensor] = {}
        self._mode4_remote_prepacked_payload_cache: dict[
            tuple[int, tuple[tuple[int, int, int], ...]], torch.Tensor] = {}
        self._mode4_owned_cache_ranks: list[int] = []
        self._mode4_cache_owner_ranks: list[int] = []
        self._elastic_current_active_ranks: list[int] | None = None
        self._post_shrink_moe_dispatch_warmed_active_signatures: set[tuple[
            int, ...]] = set()
        self._mode1_planned_floor_groups_precreated = False
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
        self._prewarm_mode1_fullworld_alltoallv_workspace()
        # Set random seed.
        NPUPlatform.seed_everything(self.model_config.seed)
        return device

    def _prewarm_mode1_fullworld_alltoallv_workspace(self) -> None:
        """Materialize the EP16 AllToAllV resource before loading weights.

        HCCL lazily allocates a large, communicator-scoped workspace on the
        first AllToAllV.  Natural floor-2 keeps 64 expert slots resident, so
        waiting until the memory-profile forward can leave enough aggregate
        free HBM but no sufficiently large contiguous allocation.  A tiny
        unequal-split collective establishes the same EP communicator resource
        while HBM is still empty.  MC2 is deliberately not touched here; its
        first materialization remains after actor offload and step KV resize.
        """
        if not _env_flag(
                "VLLM_ASCEND_MODE1_PREWARM_FULLWORLD_ALLTOALLV", "0"):
            return
        if int(envs_ascend.VLLM_ASCEND_ELASTIC_EXECUTION_MODE) != 1:
            raise RuntimeError(
                "Full-world AllToAllV prewarm requires elastic mode1")
        if not self.parallel_config.enable_expert_parallel:
            raise RuntimeError(
                "Full-world AllToAllV prewarm requires expert parallelism")

        ep_group = get_ep_group()
        world_size = int(ep_group.world_size)
        if world_size != int(self.parallel_config.data_parallel_size):
            raise RuntimeError(
                "Full-world AllToAllV prewarm must use the original EP/DP "
                f"communicator: ep={world_size} "
                f"dp={self.parallel_config.data_parallel_size}")
        if world_size <= 1:
            raise RuntimeError(
                "Full-world AllToAllV prewarm requires more than one rank")

        rank = int(ep_group.rank_in_group)
        input_split_sizes = [1 + ((rank + dst) & 1)
                             for dst in range(world_size)]
        output_split_sizes = [1 + ((src + rank) & 1)
                              for src in range(world_size)]
        input_tensor = torch.zeros(
            (sum(input_split_sizes), 1), dtype=torch.float16, device="npu")
        output_tensor = torch.empty(
            (sum(output_split_sizes), 1),
            dtype=input_tensor.dtype,
            device=input_tensor.device,
        )
        self._log_custom_mode1_worker_memory(
            "before_preweight_alltoallv_workspace",
            f"ep_world_size={world_size}",
        )
        dist.all_to_all_single(
            output_tensor,
            input_tensor,
            output_split_sizes=output_split_sizes,
            input_split_sizes=input_split_sizes,
            group=ep_group.device_group,
        )
        torch.npu.synchronize()
        # HCCL implements a device-group barrier with AllReduce. On this stack
        # that first AllReduce lazily reserves another HCCL_BUFFSIZE workspace,
        # defeating the purpose of reserving only the MoE AllToAllV resource.
        # The collective is already synchronized above; use Gloo solely to
        # prove every rank completed before model construction proceeds.
        dist.barrier(group=ep_group.cpu_group)
        self._log_custom_mode1_worker_memory(
            "after_preweight_alltoallv_workspace",
            f"ep_world_size={world_size}",
        )
        if self.rank == 0:
            logger.info(
                "Mode1 pre-weight full-world AllToAllV workspace ready: "
                "ep_world_size=%s barrier_backend=gloo "
                "mc2_materialized=false",
                world_size,
            )

    def init_device(self):
        with set_current_vllm_config(self.vllm_config):
            device = self._init_device()
            # Init ModelRunner here, so that we have access to self.device.
            self.model_runner = NPUModelRunner(self.vllm_config, device)

    @contextmanager
    def _elastic_aclgraph_memory_profile(self):
        """Profile HCCL/model memory before compiling the decode graph.

        A floor-2 worker keeps 64 local expert slots resident so later
        16->8->4->2 transitions do not reload weights.  Compiling the model
        before the first MC2 dispatch can fragment enough HBM that HCCL cannot
        reserve its fixed communication workspace.  The memory profile is not
        graph-replayed work, so execute the original model forward here.  The
        rollout lifecycle restores compilation and performs the real
        FULL_DECODE_ONLY capture after weights and KV state are stable.
        """
        model_runner = getattr(self, "model_runner", None)
        model = getattr(model_runner, "model", None)
        compile_wrappers = []
        elastic_aclgraph_moe_blocks = []
        aclgraph_moe_methods = []
        if model is not None:
            compile_wrappers = [
                module for module in model.modules()
                if hasattr(module, "do_not_compile")
                and hasattr(module, "compiled_codes")
            ]
            for module in model.modules():
                if not _is_ascend_fused_moe_module(module):
                    continue
                quant_method = getattr(module, "quant_method", None)
                if quant_method is not None and hasattr(
                        quant_method, "use_aclgraph"):
                    aclgraph_moe_methods.append(quant_method)
                parent = None
                for candidate in model.modules():
                    if getattr(candidate, "experts", None) is module:
                        parent = candidate
                        break
                if parent is not None and hasattr(
                        parent, "use_elastic_aclgraph"):
                    elastic_aclgraph_moe_blocks.append(parent)
        enabled = bool(
            envs_ascend.VLLM_ASCEND_ELASTIC_ACLGRAPH
            and getattr(model_runner, "use_aclgraph", False)
            and compile_wrappers
            and _env_flag(
                "VLLM_ASCEND_ELASTIC_ACLGRAPH_EAGER_MEMORY_PROFILE", "1"))
        if not enabled:
            yield
            return

        already_compiled = [
            type(module).__name__ for module in compile_wrappers
            if getattr(module, "compiled_codes", None)
        ]
        if already_compiled:
            raise RuntimeError(
                "Elastic ACLGraph memory profile must run before model "
                f"compilation; already compiled wrappers={already_compiled}")
        if (not aclgraph_moe_methods or not elastic_aclgraph_moe_blocks
                or len(aclgraph_moe_methods)
                != len(elastic_aclgraph_moe_blocks)):
            raise RuntimeError(
                "Elastic ACLGraph eager memory profile could not pair every "
                "elastic MoE block with its Ascend quant method; refusing to "
                "profile through a mixed graph/eager MoE path: "
                f"blocks={len(elastic_aclgraph_moe_blocks)} "
                f"methods={len(aclgraph_moe_methods)}")
        old_do_not_compile = [
            bool(module.do_not_compile) for module in compile_wrappers
        ]
        old_moe_use_aclgraph = [
            bool(method.use_aclgraph) for method in aclgraph_moe_methods
        ]
        old_block_use_elastic_aclgraph = [
            bool(block.use_elastic_aclgraph)
            for block in elastic_aclgraph_moe_blocks
        ]
        for module in compile_wrappers:
            module.do_not_compile = True
        for method in aclgraph_moe_methods:
            method.use_aclgraph = False
        for block in elastic_aclgraph_moe_blocks:
            block.use_elastic_aclgraph = False
        logger.warning(
            "Elastic ACLGraph memory profile uses original eager forward; "
            "decode graph compilation remains deferred to rollout capture; "
            "compile_wrappers=%s moe_graph_blocks_disabled=%s "
            "moe_aclgraph_methods_disabled=%s",
            [type(module).__name__ for module in compile_wrappers],
            len(elastic_aclgraph_moe_blocks),
            len(aclgraph_moe_methods))
        try:
            yield
        finally:
            for module, old_value in zip(compile_wrappers,
                                         old_do_not_compile):
                module.do_not_compile = old_value
            for method, old_value in zip(aclgraph_moe_methods,
                                         old_moe_use_aclgraph):
                method.use_aclgraph = old_value
            for block, old_value in zip(elastic_aclgraph_moe_blocks,
                                        old_block_use_elastic_aclgraph):
                block.use_elastic_aclgraph = old_value
            logger.warning(
                "Elastic ACLGraph memory profile restored compilation state")

    def determine_available_memory(self) -> int:
        # Profile the memory usage of the model and get the maximum number of
        # cache blocks that can be allocated with the remaining free memory.
        NPUPlatform.clear_npu_memory()
        NPUPlatform.synchronize()

        # Execute a forward pass with dummy inputs to profile the memory usage
        # of the model.
        _, total_npu_memory = NPUPlatform.mem_get_info()
        self._log_custom_mode1_worker_memory(
            "before_initial_memory_profile",
            "profile_phase=before_model_forward",
        )
        with self._elastic_aclgraph_memory_profile():
            self.model_runner.profile_run()
        NPUPlatform.synchronize()
        self._log_custom_mode1_worker_memory(
            "after_initial_memory_profile",
            "profile_phase=after_model_forward",
        )
        NPUPlatform.empty_cache()
        self._precreate_mode1_planned_floor_groups_before_kv_cache()
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
        floor_prealloc_headroom_bytes = (
            self._estimate_floor_prealloc_headroom_bytes())
        if self._has_effective_followup_elastic_shrink():
            shrink_headroom_bytes = (
                self._estimate_zero_redundancy_shrink_headroom_bytes())
            available_kv_cache_memory = max(
                available_kv_cache_memory - shrink_headroom_bytes, 0)
            logger.info(
                "Applying lossless zero-redundancy shrink headroom: %s bytes",
                shrink_headroom_bytes)
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
        custom_mode1_kv_headroom_bytes = (
            self._estimate_custom_mode1_kv_materialize_headroom_bytes())
        if custom_mode1_kv_headroom_bytes > 0:
            available_kv_cache_memory = max(
                available_kv_cache_memory - custom_mode1_kv_headroom_bytes,
                0)
            logger.info(
                "Applying custom mode=1 KV materialization headroom: %s bytes",
                custom_mode1_kv_headroom_bytes)
        kv_cache_init_headroom_bytes = (
            self._estimate_kv_cache_init_headroom_bytes())
        if kv_cache_init_headroom_bytes > 0:
            available_kv_cache_memory = max(
                available_kv_cache_memory - kv_cache_init_headroom_bytes, 0)
            logger.info("Applying KV cache init headroom: %s bytes",
                        kv_cache_init_headroom_bytes)
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
        if os.getenv("VLLM_ASCEND_FULL_REDUNDANCY_EXPERIMENT_LOG", "0").lower() \
                in ("1", "true", "yes", "on"):
            logger.info(
                "Elastic redundancy HBM profile: rank=%s mode=%s floor=%s "
                "total_npu_memory=%s init_free=%s post_profile_free=%s "
                "peak_memory=%s torch_current=%s non_torch_allocations=%s "
                "available_kv_cache_memory=%s gpu_memory_utilization=%s",
                self.rank,
                envs_ascend.VLLM_ASCEND_ELASTIC_EXECUTION_MODE,
                envs_ascend.VLLM_ASCEND_ELASTIC_MIN_COMPUTE_GROUP_SIZE,
                total_npu_memory,
                self.init_npu_memory,
                free_npu_memory,
                peak_memory,
                torch_allocated_bytes,
                max(non_torch_allocations, 0),
                available_kv_cache_memory,
                self.cache_config.gpu_memory_utilization,
            )
        logger.info(
            f"Available memory: {available_kv_cache_memory}, total memory: {total_npu_memory}"
        )
        return available_kv_cache_memory

    def _has_effective_followup_elastic_shrink(self) -> bool:
        if not envs_ascend.VLLM_ASCEND_ENABLE_ELASTIC_PARALLEL_SHRINK:
            return False

        model_runner = getattr(self, "model_runner", None)
        model = getattr(model_runner, "model", None) if model_runner else None
        if model is None:
            return False

        for module in model.modules():
            if not _is_elastic_headroom_moe_module(module):
                continue
            if not _module_uses_lossless_elastic(module):
                continue
            if _module_has_preallocated_redundant_slots(module):
                continue
            if _module_followup_shrink_enabled(module):
                return True
        return False

    def _has_effective_post_restore_collective_headroom_need(self) -> bool:
        if not envs_ascend.VLLM_ASCEND_ENABLE_ELASTIC_PARALLEL_SHRINK:
            return False

        model_runner = getattr(self, "model_runner", None)
        model = getattr(model_runner, "model", None) if model_runner else None
        if model is None:
            return False

        logged_lightweight_skip = False
        for module in model.modules():
            if not _is_elastic_headroom_moe_module(module):
                continue
            if not _module_uses_lossless_elastic(module):
                continue
            if _module_skips_generic_headroom(module):
                if not logged_lightweight_skip:
                    logger.info(
                        "Skipping generic post-restore headroom for elastic lightweight path: layer=%s floor=%s path=%s",
                        getattr(module, "layer_idx", -1),
                        _module_configured_elastic_floor(module),
                        _module_lightweight_no_headroom_path(module),
                    )
                    logged_lightweight_skip = True
                continue
            if _module_followup_shrink_enabled(module):
                return True
        return False

    def _has_effective_post_restore_moe_dispatch_headroom_need(self) -> bool:
        return self._has_effective_post_restore_collective_headroom_need()

    def _mode4_lightweight_min_configured_floor(self) -> Optional[int]:
        model_runner = getattr(self, "model_runner", None)
        model = getattr(model_runner, "model", None) if model_runner else None
        if model is None:
            return None

        min_floor: Optional[int] = None
        for module in model.modules():
            if not _module_is_mode4_remote_npu_lightweight(module):
                continue
            if not _module_followup_shrink_enabled(module):
                continue
            configured_floor = _mode4_runtime_elastic_floor()
            if configured_floor is None:
                configured_floor = _module_configured_elastic_floor(module)
            if configured_floor is None:
                continue
            configured_floor = int(configured_floor)
            if min_floor is None:
                min_floor = configured_floor
            else:
                min_floor = min(min_floor, configured_floor)
        return min_floor

    def _estimate_zero_redundancy_shrink_headroom_bytes(self) -> int:
        model_runner = getattr(self, "model_runner", None)
        model = getattr(model_runner, "model", None) if model_runner else None
        if model is None:
            return 0

        estimated_bytes = 0
        saw_zero_redundancy_module = False
        preallocated_loaded_layers = 0
        for module in model.modules():
            if not _is_ascend_fused_moe_module(module):
                continue
            if getattr(module, "elastic_moe_mode", "lossy") != "lossless":
                continue
            if _module_skips_generic_headroom(module):
                logger.info(
                    "Skipping zero-redundancy shrink headroom for elastic lightweight path: layer=%s floor=%s path=%s",
                    getattr(module, "layer_idx", -1),
                    _module_configured_elastic_floor(module),
                    _module_lightweight_no_headroom_path(module),
                )
                return 0
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
        model_runner = getattr(self, "model_runner", None)
        model = getattr(model_runner, "model", None) if model_runner else None
        if model is None:
            return 0

        estimated_bytes = 0
        saw_extra_floor_prealloc = False
        mode3_double_buffer_bytes = 0
        mode3_cpu_stage_bytes = 0
        mode3_low_floor_mc2_workspace_bytes = 0

        for module in model.modules():
            if not _is_ascend_fused_moe_module(module):
                continue
            if getattr(module, "elastic_moe_mode", "lossy") != "lossless":
                continue
            execution_mode = int(getattr(module, "elastic_execution_mode", 0))
            if _module_skips_generic_headroom(module) and execution_mode not in (3, 4, 5):
                logger.info(
                    "Skipping floor-prealloc headroom for elastic lightweight path: layer=%s floor=%s path=%s",
                    getattr(module, "layer_idx", -1),
                    _module_configured_elastic_floor(module),
                    _module_lightweight_no_headroom_path(module),
                )
                return 0
            if getattr(module, "global_redundant_expert_num", 0) > 0:
                continue

            # Mode 3 keeps additional experts in CPU shadow storage and only
            # needs HBM for the two runtime double-buffer slots. Do not charge
            # the mode=1/fixed-slot static expert delta here, but do reserve
            # the final floor-sized runtime slots before KV cache sizing.
            mode3_lightweight = (
                execution_mode == 3
                and _module_is_mode3_cross_layer_lightweight(module))
            mode4_lightweight = (
                execution_mode in (4, 5)
                and _module_is_mode4_remote_npu_lightweight(module))
            mode5_lightweight = mode4_lightweight and execution_mode == 5
            if (not mode3_lightweight and not mode4_lightweight
                    and not getattr(module,
                                    "lossless_zero_redundancy_preallocated_loaded",
                                    False)):
                continue

            configured_floor = getattr(
                module, "_get_configured_elastic_min_compute_group_size",
                lambda: None)()
            if mode4_lightweight:
                runtime_floor = _mode4_runtime_elastic_floor()
                if runtime_floor is not None:
                    configured_floor = int(runtime_floor)
            if configured_floor is None:
                continue
            default_floor = getattr(module, "_get_default_single_shrink_ep_floor",
                                    lambda: configured_floor)()
            if configured_floor >= default_floor and not mode4_lightweight:
                continue

            get_reserved_slots = getattr(
                module, "_get_reserved_local_expert_slots_for_floor", None)
            if get_reserved_slots is None:
                continue
            configured_capacity = int(get_reserved_slots(configured_floor))
            default_capacity = int(get_reserved_slots(default_floor))
            additional_experts = max(configured_capacity - default_capacity, 0)
            if additional_experts <= 0 and not mode3_lightweight and not mode4_lightweight:
                continue

            if int(module.w13_weight.shape[0]) <= 0:
                continue
            w13_sample = module.w13_weight[0]
            w2_sample = module.w2_weight[0]
            per_expert_bytes = (
                w13_sample.numel() * w13_sample.element_size()
                + w2_sample.numel() * w2_sample.element_size())

            if not mode3_lightweight and not mode4_lightweight:
                if hasattr(module, "_is_hybrid_cpu_swap_enabled") and \
                        module._is_hybrid_cpu_swap_enabled():
                    continue
                estimated_bytes += additional_experts * per_expert_bytes

            # Mode3 double buffer slots: ALL layers share the same 2 slots
            # (double buffer). These slots are allocated lazily on first forward,
            # NOT during memory profiling, so we must account for them here.
            # The CPU-shadow path also allocates one NPU staging buffer per
            # runtime slot for CPU->NPU materialization, so reserve that too.
            # Only calculate once for the first mode3 layer we encounter.
            if mode3_lightweight and mode3_double_buffer_bytes == 0:
                runtime_slot_experts = configured_capacity * 2
                cpu_stage_experts = configured_capacity * 2
                mode3_double_buffer_bytes = runtime_slot_experts * per_expert_bytes
                mode3_cpu_stage_bytes = cpu_stage_experts * per_expert_bytes
                logger.info(
                    "Mode3 double buffer headroom (shared across all layers): configured_capacity=%s runtime_slot_experts=%s runtime_bytes=%.2fMB cpu_stage_experts=%s cpu_stage_bytes=%.2fMB",
                    configured_capacity,
                    runtime_slot_experts,
                    mode3_double_buffer_bytes / (1024 * 1024),
                    cpu_stage_experts,
                    mode3_cpu_stage_bytes / (1024 * 1024),
                )

            # Even with the mode3 runtime slots and CPU staging buffers charged
            # up front, the real low-floor MC2 decode path can lazily request an
            # additional HCCL workspace after shrink. If this is not reserved
            # before KV sizing, the first stage=4/stage=2 decode can fail inside
            # HcclAllocComResourceByTiling and poison later collectives. Keep the
            # reservation mode3-only so mode1/mode2/mode4 KV sizing is untouched.
            if mode4_lightweight and mode3_double_buffer_bytes == 0:
                runtime_slot_experts = configured_capacity * 2
                mode3_double_buffer_bytes = runtime_slot_experts * per_expert_bytes
                logger.info(
                    "Mode4 remote-NPU double buffer headroom (shared across all layers): runtime_floor=%s configured_capacity=%s runtime_slot_experts=%s runtime_bytes=%.2fMB",
                    configured_floor, configured_capacity, runtime_slot_experts,
                    mode3_double_buffer_bytes / (1024 * 1024))

            if (mode4_lightweight and configured_floor <= 2
                    and mode3_low_floor_mc2_workspace_bytes == 0):
                if mode5_lightweight:
                    # Mode 5 keeps both CPU staging and remote-NPU cache
                    # traffic active. Repeated shrink/restore can leave the
                    # first live MC2 dispatch needing a fresh HCCL tiling
                    # workspace on top of the double-buffer slots.
                    default_workspace_bytes = int(4 * 1024 * 1024 * 1024)
                    env_name = (
                        "VLLM_ASCEND_MODE5_LOW_FLOOR_MC2_WORKSPACE_HEADROOM_BYTES")
                else:
                    default_workspace_bytes = int(2 * 1024 * 1024 * 1024)
                    env_name = (
                        "VLLM_ASCEND_MODE4_LOW_FLOOR_MC2_WORKSPACE_HEADROOM_BYTES")
                mode3_low_floor_mc2_workspace_bytes = int(
                    os.getenv(env_name, str(default_workspace_bytes)))
                logger.info(
                    "Mode%s low-floor MC2 workspace headroom: runtime_floor=%s bytes=%s (%.2fMB)",
                    execution_mode, configured_floor,
                    mode3_low_floor_mc2_workspace_bytes,
                    mode3_low_floor_mc2_workspace_bytes / (1024 * 1024))

            if (mode3_lightweight and configured_floor <= 2
                    and mode3_low_floor_mc2_workspace_bytes == 0):
                default_workspace_bytes = int(2.5 * 1024 * 1024 * 1024)
                mode3_low_floor_mc2_workspace_bytes = int(
                    os.getenv(
                        "VLLM_ASCEND_MODE3_LOW_FLOOR_MC2_WORKSPACE_HEADROOM_BYTES",
                        str(default_workspace_bytes)))
                logger.info(
                    "Mode3 low-floor MC2 workspace headroom: floor=%s bytes=%s (%.2fMB)",
                    configured_floor, mode3_low_floor_mc2_workspace_bytes,
                    mode3_low_floor_mc2_workspace_bytes / (1024 * 1024))

            saw_extra_floor_prealloc = True

        if not saw_extra_floor_prealloc or (
                estimated_bytes <= 0 and mode3_double_buffer_bytes <= 0
                and mode3_cpu_stage_bytes <= 0
                and mode3_low_floor_mc2_workspace_bytes <= 0):
            return 0

        safety_margin_bytes = int(
            os.getenv("VLLM_ASCEND_FLOOR_PREALLOC_HEADROOM_SAFETY_BYTES",
                      str(1024 * 1024 * 1024)))

        # For mode3 double buffer slots, they are NOT pre-allocated during
        # memory profiling (allocated lazily on first forward), so we need to
        # subtract the full estimated bytes from KV cache budget.
        # For other cases, the weights are already in peak_memory, so only
        # keep a small safety margin.
        has_mode3_double_buffer = (
            mode3_double_buffer_bytes > 0 or mode3_cpu_stage_bytes > 0)

        if has_mode3_double_buffer:
            total_headroom = (estimated_bytes + mode3_double_buffer_bytes
                              + mode3_cpu_stage_bytes
                              + mode3_low_floor_mc2_workspace_bytes
                              + safety_margin_bytes)
            logger.info(
                "Estimated elastic floor prealloc headroom (including mode3/mode4 double buffer): static_bytes=%s runtime_buffer_bytes=%s mode3_cpu_stage_bytes=%s low_floor_mc2_workspace_bytes=%s safety_margin=%s total=%s (%.2fMB)",
                estimated_bytes, mode3_double_buffer_bytes,
                mode3_cpu_stage_bytes, mode3_low_floor_mc2_workspace_bytes,
                safety_margin_bytes, total_headroom,
                total_headroom / (1024 * 1024))
            return total_headroom
        else:
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
        mode4_floor = self._mode4_lightweight_min_configured_floor()
        if (mode4_floor is not None
                and not _env_flag("VLLM_ASCEND_MODE4_ENABLE_GENERIC_HEADROOM",
                                  "0")):
            # Mode 4 can skip the broad DP/EP/zero-redundancy reservations, but
            # the first real MoE dispatch after restore still materializes an
            # HCCL/MC2 workspace. Keep one shared dispatch budget here instead
            # of separately charging post-shrink and post-restore paths.
            default_headroom_bytes = 2 * 1024 * 1024 * 1024
            if mode4_floor <= 4:
                default_headroom_bytes = 4 * 1024 * 1024 * 1024
            if mode4_floor <= 2:
                default_headroom_bytes = 6 * 1024 * 1024 * 1024
            return int(
                os.getenv("VLLM_ASCEND_MODE4_MOE_DISPATCH_HEADROOM_BYTES",
                          str(default_headroom_bytes)))
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
            if not _is_elastic_headroom_moe_module(module):
                continue
            if not _module_uses_lossless_elastic(module):
                continue
            if _module_skips_generic_headroom(module):
                logger.info(
                    "Skipping first-live-prefill headroom for elastic lightweight path: layer=%s floor=%s path=%s",
                    getattr(module, "layer_idx", -1),
                    _module_configured_elastic_floor(module),
                    _module_lightweight_no_headroom_path(module),
                )
                return 0
            if not _module_followup_shrink_enabled(module):
                continue
            configured_floor = _module_configured_elastic_floor(module)
            if configured_floor is not None and int(configured_floor) <= 4:
                return max(base_headroom_bytes, low_floor_headroom_bytes)
            return base_headroom_bytes
        return 0

    def _estimate_extra_elastic_safety_headroom_bytes(self) -> int:
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
            if not _is_elastic_headroom_moe_module(module):
                continue
            if not _module_uses_lossless_elastic(module):
                continue
            configured_floor = _module_configured_elastic_floor(module)
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
        if not envs_ascend.VLLM_ASCEND_ENABLE_ELASTIC_PARALLEL_SHRINK:
            return 0

        model_runner = getattr(self, "model_runner", None)
        model = getattr(model_runner, "model", None) if model_runner else None
        if model is None:
            return 0

        saw_lightweight_no_headroom = False
        saw_low_floor_module = False
        saw_elastic_lossless_module = False
        min_configured_floor = None
        logged_lightweight_skip = False
        for module in model.modules():
            if not _is_elastic_headroom_moe_module(module):
                continue
            if not _module_uses_lossless_elastic(module):
                continue
            if _module_skips_generic_headroom(module):
                saw_lightweight_no_headroom = True
                if not logged_lightweight_skip:
                    logger.info(
                        "Skipping post-shrink MoE dispatch headroom for elastic lightweight path: layer=%s floor=%s path=%s",
                        getattr(module, "layer_idx", -1),
                        _module_configured_elastic_floor(module),
                        _module_lightweight_no_headroom_path(module),
                    )
                    logged_lightweight_skip = True
                continue
            if not _module_followup_shrink_enabled(module):
                continue
            saw_elastic_lossless_module = True
            configured_floor = _module_configured_elastic_floor(module)
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

        if saw_lightweight_no_headroom and not saw_elastic_lossless_module:
            return 0

        if not self._has_effective_post_restore_collective_headroom_need():
            return 0

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
        if not envs_ascend.VLLM_ASCEND_ENABLE_ELASTIC_PARALLEL_SHRINK:
            return 0

        model_runner = getattr(self, "model_runner", None)
        model = getattr(model_runner, "model", None) if model_runner else None
        if model is None:
            return 0

        saw_lightweight_no_headroom = False
        saw_low_floor_lossless_module = False
        min_configured_floor = None
        logged_lightweight_skip = False
        for module in model.modules():
            if not _is_elastic_headroom_moe_module(module):
                continue
            if not _module_uses_lossless_elastic(module):
                continue
            if _module_skips_generic_headroom(module):
                saw_lightweight_no_headroom = True
                if not logged_lightweight_skip:
                    logger.info(
                        "Skipping post-shrink prefill AllToAll headroom for elastic lightweight path: layer=%s floor=%s path=%s",
                        getattr(module, "layer_idx", -1),
                        _module_configured_elastic_floor(module),
                        _module_lightweight_no_headroom_path(module),
                    )
                    logged_lightweight_skip = True
                continue
            if not _module_followup_shrink_enabled(module):
                continue
            if _module_hybrid_cpu_swap_enabled(module):
                continue
            configured_floor = _module_configured_elastic_floor(module)
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

        if saw_lightweight_no_headroom and not saw_low_floor_lossless_module:
            return 0

        if not self._has_effective_post_restore_collective_headroom_need():
            return 0

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

    def _estimate_custom_mode1_kv_materialize_headroom_bytes(self) -> int:
        if not envs_ascend.VLLM_ASCEND_ENABLE_ELASTIC_PARALLEL_SHRINK:
            return 0

        model_runner = getattr(self, "model_runner", None)
        model = getattr(model_runner, "model", None) if model_runner else None
        if model is None:
            return 0

        for module in model.modules():
            if _module_skips_generic_headroom(module):
                logger.info(
                    "Skipping KV materialization headroom for elastic lightweight path: layer=%s floor=%s path=%s",
                    getattr(module, "layer_idx", -1),
                    _module_configured_elastic_floor(module),
                    _module_lightweight_no_headroom_path(module),
                )
                return 0
            if (_is_custom_ascend_fused_moe_module(module)
                    and _module_is_custom_mode1_redundant_static(module)):
                return int(
                    os.getenv(
                        "VLLM_ASCEND_CUSTOM_MODE1_KV_MATERIALIZE_HEADROOM_BYTES",
                        str(2 * 1024 * 1024 * 1024)))
        return 0

    def _estimate_kv_cache_init_headroom_bytes(self) -> int:
        default_headroom = "0"
        model_runner = getattr(self, "model_runner", None)
        if (envs_ascend.VLLM_ASCEND_ELASTIC_ACLGRAPH
                and getattr(model_runner, "use_aclgraph", False)):
            # A two-size FULL_DECODE_ONLY capture currently retains about
            # 2.09 GiB/rank on Qwen3-30B-A3B.  Reserve 3 GiB so KV sizing does
            # not consume graph memory plus allocator variation.
            default_headroom = str(3 * GiB_bytes)
        return int(float(os.getenv("VLLM_ASCEND_KV_CACHE_INIT_HEADROOM_BYTES",
                                   default_headroom)))

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

        with self._maybe_elastic_op_profile("live"):
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
        keep_mode4_weights_resident = _mode4_keep_weights_out_of_sleep_pool()
        if (self.vllm_config.model_config.enable_sleep_mode
                and not keep_mode4_weights_resident):
            allocator = CaMemAllocator.get_instance()
            assert allocator.get_current_usage() == 0, (
                "Sleep mode can only be "
                "used for one instance per process.")
            context = allocator.use_memory_pool(tag="weights")
        else:
            context = nullcontext()  # type: ignore
            if (self.vllm_config.model_config.enable_sleep_mode
                    and keep_mode4_weights_resident):
                logger.info(
                    "Mode4 remote NPU cache keeps model weights outside sleep pool: "
                    "sleep(level=1) may release KV/cache but expert weights remain NPU-resident."
                )
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
            defer_elastic_capture = bool(
                envs_ascend.VLLM_ASCEND_ELASTIC_ACLGRAPH
                and getattr(self.model_runner, "use_aclgraph", False))
            if defer_elastic_capture:
                # KV-cache initialization also calls this method. Capturing
                # here races the rollout-owned lifecycle and causes a second
                # capture after the current actor weights are loaded.  The
                # rollout boundary owns the single initial capture for both
                # fixed-full-world and shrink-aware execution; floor
                # transitions own later topology-specific captures.
                logger.warning(
                    "Elastic ACLGraph automatic capture deferred until "
                    "rollout weights, KV cache, and communication groups are "
                    "stable")
            else:
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

    def clear_kv_cache_for_resize(self) -> None:
        """Release worker-side KV tensor references before step-level resize."""
        cleared_layers = 0
        rebuilt_input_batch = False
        cleared_connector_refs = False
        try:
            from vllm.attention import AttentionType
        except Exception:
            AttentionType = None

        model = None
        module_map = {}
        try:
            model = self.model_runner.get_model()
            if hasattr(model, "named_modules"):
                module_map = dict(model.named_modules())
        except Exception:
            logger.exception("Failed to build module map while clearing KV cache")

        ctx = getattr(self.vllm_config.compilation_config,
                      "static_forward_context", None)
        if ctx is not None:
            pipeline_parallel_size = max(
                int(self.vllm_config.parallel_config.pipeline_parallel_size), 1)
            layer_names = []
            for layer_name, ctx_entry in ctx.items():
                attn_type = getattr(ctx_entry, "attn_type", None)
                if (AttentionType is None or attn_type in (
                        AttentionType.DECODER, AttentionType.ENCODER_DECODER)):
                    layer_names.append(layer_name)
            for layer_name in layer_names:
                kv_cache = [
                    torch.tensor([]) for _ in range(pipeline_parallel_size)
                ]
                try:
                    ctx[layer_name].kv_cache = kv_cache
                    module = module_map.get(layer_name)
                    if module is not None and hasattr(module, "kv_cache"):
                        module.kv_cache = kv_cache
                    cleared_layers += 1
                except Exception:
                    logger.exception(
                        "Failed to clear KV cache reference for layer %s",
                        layer_name)

        try:
            self.model_runner.kv_caches = []
        except Exception:
            logger.exception("Failed to clear model_runner.kv_caches")

        try:
            runner = self.model_runner
            old_input_batch = getattr(runner, "input_batch", None)
            logitsprocs = getattr(old_input_batch, "logitsprocs", None)
            block_sizes = [
                group.kv_cache_spec.block_size
                for group in runner.kv_cache_config.kv_cache_groups
            ]
            runner.input_batch = InputBatch(
                max_num_reqs=runner.max_num_reqs,
                max_model_len=runner.model_config.max_model_len,
                max_num_batched_tokens=runner.max_num_tokens,
                device=runner.device,
                pin_memory=runner.pin_memory,
                vocab_size=runner.model_config.get_vocab_size(),
                block_sizes=block_sizes,
                logitsprocs=logitsprocs,
                is_spec_decode=bool(runner.vllm_config.speculative_config),
                is_pooling_model=runner.is_pooling_model,
                num_speculative_tokens=(
                    runner.vllm_config.speculative_config.num_speculative_tokens
                    if runner.vllm_config.speculative_config else 0),
            )
            rebuilt_input_batch = True
        except Exception:
            logger.exception("Failed to rebuild empty input_batch for KV resize")

        try:
            from vllm.distributed.kv_transfer import (get_kv_transfer_group,
                                                      has_kv_transfer_group)
            if has_kv_transfer_group():
                kv_group = get_kv_transfer_group()
                connector_worker = getattr(kv_group, "connector_worker", None)
                targets = [kv_group]
                if connector_worker is not None:
                    targets.append(connector_worker)
                for target in targets:
                    for attr in ("gpu_kv_caches", "gpu_kv_caches_load_iter",
                                 "cpu_kv_caches", "kv_caches"):
                        if hasattr(target, attr):
                            try:
                                setattr(target, attr, None)
                                cleared_connector_refs = True
                            except Exception:
                                logger.debug(
                                    "Failed to clear KV connector attr %s on %s",
                                    attr,
                                    type(target).__name__,
                                    exc_info=True,
                                )
        except Exception:
            logger.exception("Failed to clear KV transfer connector references")

        if model is not None:
            try:
                first_attn = model.model.layers[0].self_attn
                if hasattr(first_attn, "attn"):
                    for i in range(model.model.start_layer,
                                   model.model.end_layer):
                        attn_impl = model.model.layers[i].self_attn.attn.impl
                        if hasattr(attn_impl, "key_cache"):
                            attn_impl.key_cache = None
                            attn_impl.value_cache = None
                if hasattr(first_attn, "mla_attn"):
                    for i in range(model.model.start_layer,
                                   model.model.end_layer):
                        mla_attn = model.model.layers[i].self_attn.mla_attn
                        mla_impl = getattr(mla_attn, "impl", None)
                        if mla_impl is None:
                            mla_impl = getattr(
                                getattr(mla_attn, "mla_attn", None),
                                "impl",
                                None,
                            )
                        if mla_impl is None:
                            continue
                        if hasattr(mla_impl, "key_cache"):
                            mla_impl.key_cache = None
                            mla_impl.value_cache = None
            except Exception:
                logger.exception("Failed to clear attention implementation KV refs")

        gc.collect()
        try:
            torch.npu.empty_cache()
            torch.npu.synchronize()
        except Exception:
            logger.exception("Failed to empty/sync NPU cache during KV resize")
        logger.info(
            "Mode1 adaptive KV cache references cleared: rank=%s layers=%s "
            "rebuilt_input_batch=%s cleared_connector_refs=%s",
            self.rank,
            cleared_layers,
            rebuilt_input_batch,
            cleared_connector_refs,
        )

    def clear_mode1_stale_fullname_parameter_dicts_for_kv_resize(self) -> None:
        """Release stale full-name MoE Parameter caches before new KV alloc."""
        refreshed_eplb_adaptors = 0
        cleared_stale_param_dicts = 0
        cleared_stale_param_entries = 0
        cleared_stale_param_bytes = 0
        try:
            refreshed_eplb_adaptors = (
                _refresh_mode1_eplb_adaptor_references(self.model_runner))
        except Exception:
            logger.exception(
                "Failed to refresh mode1 EPLB cached parameter references "
                "during KV resize: rank=%s",
                self.rank,
            )
        try:
            model = self.model_runner.get_model()
            (cleared_stale_param_dicts, cleared_stale_param_entries,
             cleared_stale_param_bytes) = (
                 _clear_stale_mode1_fullname_parameter_dicts(model))
        except Exception:
            logger.exception(
                "Failed to clear stale mode1 full-name parameter dicts during "
                "KV resize: rank=%s",
                self.rank,
            )
        gc.collect()
        try:
            torch.npu.empty_cache()
            torch.npu.synchronize()
        except Exception:
            logger.exception(
                "Failed to empty/sync NPU cache after stale mode1 parameter "
                "dict cleanup")
        logger.info(
            "Mode1 adaptive KV resize stale full-name parameter caches cleared: "
            "rank=%s refreshed_eplb_adaptors=%s cleared_stale_param_dicts=%s "
            "cleared_stale_param_entries=%s cleared_stale_param_bytes=%s",
            self.rank,
            refreshed_eplb_adaptors,
            cleared_stale_param_dicts,
            cleared_stale_param_entries,
            cleared_stale_param_bytes,
        )

    def mode1_resize_live_tensor_scan(self, tag: str = "") -> None:
        """Log live NPU tensor storages during adaptive KV resize debugging."""
        if not _mode1_kv_resize_live_tensor_scan_enabled():
            return
        try:
            torch.npu.synchronize()
        except Exception:
            pass
        gc.collect()
        storages: dict[int, dict[str, object]] = {}
        tensor_count = 0
        for obj in gc.get_objects():
            try:
                if not torch.is_tensor(obj):
                    continue
                if getattr(obj, "device", None) is None or obj.device.type != "npu":
                    continue
                ptr, nbytes = _tensor_storage_key_and_nbytes(obj)
                if not ptr or nbytes <= 0:
                    continue
                tensor_count += 1
                item = storages.setdefault(
                    ptr, {
                        "nbytes": nbytes,
                        "count": 0,
                        "dtype": str(obj.dtype),
                        "shape": tuple(obj.shape),
                        "sample": obj,
                    },
                )
                item["count"] = int(item["count"]) + 1
                if nbytes > int(item["nbytes"]):
                    item["nbytes"] = nbytes
                    item["dtype"] = str(obj.dtype)
                    item["shape"] = tuple(obj.shape)
                    item["sample"] = obj
            except Exception:
                continue

        top_all = sorted(
            storages.values(), key=lambda item: int(item["nbytes"]), reverse=True)
        top_int8 = [
            item for item in top_all if str(item.get("dtype")) == "torch.int8"
        ]

        def _fmt(items: list[dict[str, object]], limit: int) -> str:
            parts = []
            for item in items[:limit]:
                parts.append(
                    f"{int(item['nbytes'])}:{item['dtype']}:{item['shape']}:refs={item['count']}"
                )
            return ";".join(parts)

        def _referrer_summary(tensor: object,
                              ignored_ref_ids: set[int],
                              limit: int = 16) -> str:
            if not torch.is_tensor(tensor):
                return ""
            summaries = []
            try:
                referrers = gc.get_referrers(tensor)
            except Exception:
                return ""
            for ref in referrers:
                if id(ref) in ignored_ref_ids:
                    continue
                try:
                    ref_type = type(ref).__name__
                    if isinstance(ref, dict):
                        keys = []
                        for key, value in list(ref.items())[:128]:
                            if value is tensor:
                                keys.append(str(key))
                        owners = []
                        try:
                            for owner in gc.get_referrers(ref)[:8]:
                                if id(owner) in ignored_ref_ids:
                                    continue
                                if getattr(owner, "__dict__", None) is ref:
                                    owners.append(type(owner).__name__)
                        except Exception:
                            pass
                        if keys == ["sample"] and not owners:
                            continue
                        dict_refs = _mode1_referrer_owner_summary(
                            ref, ignored_ref_ids, limit=6)
                        summaries.append(
                            f"dict(keys={keys[:6]},owners={owners[:4]},"
                            f"refs={dict_refs})")
                    elif isinstance(ref, (list, tuple, set)):
                        if isinstance(ref, list) and len(ref) > 1024:
                            continue
                        summaries.append(f"{ref_type}(len={len(ref)})")
                    elif hasattr(ref, "f_code") and hasattr(ref, "f_lineno"):
                        summaries.append(
                            f"frame({getattr(ref.f_code, 'co_name', '?')}@"
                            f"{getattr(ref.f_code, 'co_filename', '?')}:"
                            f"{getattr(ref, 'f_lineno', '?')})")
                    else:
                        summaries.append(ref_type)
                except Exception:
                    summaries.append(type(ref).__name__)
                if len(summaries) >= limit:
                    break
            return "|".join(summaries)

        stale_referrers = []
        ignored_ref_ids = {id(storages), id(top_all), id(top_int8)}
        for item in top_all:
            ignored_ref_ids.add(id(item))
            shape = item.get("shape")
            dtype = str(item.get("dtype"))
            if dtype != "torch.bfloat16" or not isinstance(shape, tuple):
                continue
            if (shape == (32, 1536, 2048)
                    or shape == (32, 2048, 768)
                    or (len(shape) == 1 and int(shape[0]) == 143654912)):
                sample = item.get("sample")
                stale_referrers.append(
                    f"{int(item['nbytes'])}:{dtype}:{shape}:"
                    f"{_referrer_summary(sample, ignored_ref_ids)}")
                if len(stale_referrers) >= 8:
                    break

        known: dict[str, int] = {}
        try:
            known["model_runner.kv_caches"] = _tensor_tree_storage_nbytes(
                getattr(self.model_runner, "kv_caches", None))
        except Exception:
            pass
        try:
            known["model_runner.input_batch"] = _tensor_tree_storage_nbytes(
                getattr(self.model_runner, "input_batch", None))
        except Exception:
            pass
        try:
            ctx = getattr(self.vllm_config.compilation_config,
                          "static_forward_context", None)
            known["static_forward_context"] = _tensor_tree_storage_nbytes(ctx)
        except Exception:
            pass
        try:
            model = self.model_runner.get_model()
            impl_refs = []
            first_attn = model.model.layers[0].self_attn
            if hasattr(first_attn, "attn"):
                for i in range(model.model.start_layer, model.model.end_layer):
                    attn_impl = model.model.layers[i].self_attn.attn.impl
                    impl_refs.append(getattr(attn_impl, "key_cache", None))
                    impl_refs.append(getattr(attn_impl, "value_cache", None))
            known["attention_impl_kv"] = _tensor_tree_storage_nbytes(impl_refs)
        except Exception:
            pass
        owner_hits: list[tuple[int, str, str, tuple[Any, ...]]] = []
        try:
            model = self.model_runner.get_model()
            model_level_attrs = (
                "_mode3_cpu_npu_double_buffer_slots",
                "_mode4_remote_npu_double_buffer_slots",
                "_mode3_double_buffer_manager",
            )
            for attr in model_level_attrs:
                value = getattr(model, attr, None)
                known[f"model.{attr}"] = _tensor_tree_storage_nbytes(value)
                _append_large_npu_tensor_paths(value,
                                               f"model.{attr}",
                                               owner_hits,
                                               min_bytes=64 * 1024 * 1024,
                                               seen_objects=set(),
                                               max_depth=4)
            for attr in ("eplb_adaptor", "adaptor"):
                value = getattr(self.model_runner, attr, None)
                known[f"model_runner.{attr}"] = _tensor_tree_storage_nbytes(value)
                _append_large_npu_tensor_paths(
                    value,
                    f"model_runner.{attr}",
                    owner_hits,
                    min_bytes=64 * 1024 * 1024,
                    seen_objects=set(),
                    max_depth=5)
            for owner_name in ("model", "model_runner.model"):
                owner = model if owner_name == "model" else getattr(
                    self.model_runner, "model", None)
                for attr in ("eplb_adaptor", "adaptor"):
                    value = getattr(owner, attr, None)
                    known[f"{owner_name}.{attr}"] = (
                        _tensor_tree_storage_nbytes(value))
                    _append_large_npu_tensor_paths(
                        value,
                        f"{owner_name}.{attr}",
                        owner_hits,
                        min_bytes=64 * 1024 * 1024,
                        seen_objects=set(),
                        max_depth=5)
            try:
                from vllm.forward_context import get_forward_context
                forward_context = get_forward_context()
                for attr in ("moe_double_buffer_manager", "moe_comm_method",
                             "moe_prefetch_stream"):
                    value = getattr(forward_context, attr, None)
                    known[f"forward_context.{attr}"] = (
                        _tensor_tree_storage_nbytes(value))
                    _append_large_npu_tensor_paths(
                        value,
                        f"forward_context.{attr}",
                        owner_hits,
                        min_bytes=64 * 1024 * 1024,
                        seen_objects=set(),
                        max_depth=4)
            except Exception:
                pass
            moe_seen = set()
            named_param_seen: set[int] = set()
            for module in model.modules():
                if not _is_ascend_fused_moe_module(module):
                    continue
                layer_idx = getattr(module, "layer_idx", "?")
                for name, param in getattr(module, "named_parameters",
                                           lambda recurse=False: [])(
                                               recurse=False):
                    ptr, nbytes = _tensor_storage_key_and_nbytes(param)
                    if ptr and ptr not in named_param_seen:
                        named_param_seen.add(ptr)
                        if nbytes >= 64 * 1024 * 1024:
                            owner_hits.append(
                                (nbytes, f"moe[{layer_idx}]._parameters.{name}",
                                 str(param.dtype), tuple(param.shape)))
                for name, buffer in getattr(module, "named_buffers",
                                            lambda recurse=False: [])(
                                                recurse=False):
                    ptr, nbytes = _tensor_storage_key_and_nbytes(buffer)
                    if ptr and ptr not in named_param_seen:
                        named_param_seen.add(ptr)
                        if nbytes >= 64 * 1024 * 1024:
                            owner_hits.append(
                                (nbytes, f"moe[{layer_idx}]._buffers.{name}",
                                 str(buffer.dtype), tuple(buffer.shape)))
                for attr in ("w13_weight", "w2_weight", "runtime_w13_weight",
                             "runtime_w2_weight", "runtime_w13_buffer",
                             "runtime_w2_buffer",
                             "lossless_saved_primary_prefix_w13",
                             "lossless_saved_primary_prefix_w2",
                             "lossless_cpu_shadow_w13_buffer",
                             "lossless_cpu_shadow_w2_buffer",
                             "_mode3_double_buffer_manager", "quant_method"):
                    value = getattr(module, attr, None)
                    _append_large_npu_tensor_paths(
                        value,
                        f"moe[{layer_idx}].{attr}",
                        owner_hits,
                        min_bytes=64 * 1024 * 1024,
                        seen_objects=moe_seen,
                        max_depth=3)
        except Exception:
            logger.exception(
                "Failed to collect mode1 adaptive KV owner tensor scan: "
                "rank=%s tag=%s",
                self.rank,
                tag,
            )
        try:
            from vllm.distributed.kv_transfer import (get_kv_transfer_group,
                                                      has_kv_transfer_group)
            if has_kv_transfer_group():
                known["kv_transfer_group"] = _tensor_tree_storage_nbytes(
                    get_kv_transfer_group())
        except Exception:
            pass

        logger.info(
            "Mode1 adaptive KV live tensor scan: rank=%s tag=%s "
            "tensor_count=%s storage_count=%s total_storage_bytes=%s "
            "top_int8=%s top_all=%s known_bytes=%s owner_hits=%s "
            "stale_referrers=%s",
            self.rank,
            tag,
            tensor_count,
            len(storages),
            sum(int(item["nbytes"]) for item in storages.values()),
            _fmt(top_int8, 12),
            _fmt(top_all, 12),
            known,
            ";".join(
                f"{nbytes}:{path}:{dtype}:{shape}"
                for nbytes, path, dtype, shape in sorted(
                    owner_hits, key=lambda item: item[0], reverse=True)[:24]),
            ";".join(stale_referrers),
        )

    def _cleanup_mode1_upward_floor_residue_for_kv_resize(
            self, target_floor: int, previous_floor: Optional[int],
            world_size: int) -> dict[str, Any]:
        """Compact low-floor MoE runtime state before growing shrink floor.

        Moving from a deeper shrink floor to a shallower one, for example
        floor2->floor4, must release the larger temporary expert storage that
        was materialized by the deeper floor.  Otherwise KV profiling for the
        next step sees the stale floor2 resident weights and badly underestimates
        available KV memory.
        """
        stats: dict[str, Any] = {
            "changed": 0.0,
            "restored_layers": 0,
            "compacted_layers": 0,
            "released_model_slot_caches": 0,
            "refreshed_eplb_adaptors": 0,
            "cleared_stale_param_dicts": 0,
            "cleared_stale_param_entries": 0,
            "cleared_stale_param_bytes": 0,
            "moe_weight_bytes": 0,
            "moe_runtime_bytes": 0,
            "release_ms": 0.0,
            "pre_compact_empty_cache_ms": 0.0,
            "restore_layout_ms": 0.0,
            "eplb_ms": 0.0,
            "stale_param_ms": 0.0,
            "sync_ms": 0.0,
        }
        target_loaded_capacity_hint = 0
        try:
            target_floor_int = int(target_floor)
            if target_floor_int > 0:
                hf_config = self.model_config.hf_config
                logical_num_experts = int(
                    getattr(hf_config, "n_routed_experts",
                            getattr(hf_config, "num_experts", 0)) or 0)
                if logical_num_experts > 0:
                    if logical_num_experts % target_floor_int != 0:
                        raise ValueError(
                            f"logical_num_experts={logical_num_experts} is not "
                            f"divisible by target_floor={target_floor_int}")
                    target_loaded_capacity_hint = (
                        logical_num_experts // target_floor_int)
        except Exception:
            target_loaded_capacity_hint = 0
        target_loaded_capacity_hist: dict[int, int] = {}
        model = None
        try:
            model = self.model_runner.get_model()
        except Exception:
            logger.exception(
                "Failed to fetch model for mode1 upward-floor cleanup: "
                "rank=%s previous_floor=%s target_floor=%s",
                self.rank,
                previous_floor,
                target_floor,
            )
            return stats
        if model is None:
            return stats

        released_model_slot_caches = 0
        release_start_t = time.perf_counter()
        try:
            from vllm.forward_context import get_forward_context
            try:
                forward_context = get_forward_context()
            except AssertionError:
                forward_context = None
            if hasattr(forward_context, "moe_double_buffer_manager"):
                released_model_slot_caches += _release_mode1_double_buffer_owner(
                    getattr(forward_context, "moe_double_buffer_manager", None))
                setattr(forward_context, "moe_double_buffer_manager", None)
                released_model_slot_caches += 1
        except Exception:
            logger.exception(
                "Failed to release forward-context mode1 double-buffer cache "
                "during upward-floor cleanup: rank=%s",
                self.rank,
            )
        if hasattr(model, "_mode3_double_buffer_manager"):
            try:
                released_model_slot_caches += _release_mode1_double_buffer_owner(
                    getattr(model, "_mode3_double_buffer_manager", None))
                setattr(model, "_mode3_double_buffer_manager", None)
                released_model_slot_caches += 1
            except Exception:
                logger.exception(
                    "Failed to release model-level mode1 manager cache during "
                    "upward-floor cleanup: rank=%s",
                    self.rank,
                )
        for slot_attr in ("_mode3_cpu_npu_double_buffer_slots",
                          "_mode4_remote_npu_double_buffer_slots"):
            if hasattr(model, slot_attr):
                try:
                    released_model_slot_caches += _release_mode1_double_buffer_owner(
                        getattr(model, slot_attr, None))
                    setattr(model, slot_attr, None)
                    released_model_slot_caches += 1
                except Exception:
                    logger.exception(
                        "Failed to release model-level mode1 slot cache during "
                        "upward-floor cleanup: rank=%s attr=%s",
                        self.rank,
                        slot_attr,
                    )

        restored_layers = 0
        compacted_layers = 0
        incremental_stale_param_dicts = 0
        incremental_stale_param_entries = 0
        incremental_stale_param_bytes = 0
        compact_release_every = max(1, int(os.getenv(
            "VLLM_ASCEND_MODE1_UPWARD_COMPACT_RELEASE_EVERY_LAYERS", "48")))
        moe_weight_bytes = 0
        moe_runtime_bytes = 0
        moe_loaded_capacity_hist: dict[int, int] = {}
        moe_weight_shape_hist: dict[str, int] = {}
        restore_layout_start_t = time.perf_counter()
        pre_compact_empty_cache_ms = 0.0
        if _env_flag("VLLM_ASCEND_MODE1_EMPTY_CACHE_BEFORE_UPWARD_COMPACT",
                     "1"):
            pre_compact_empty_cache_start_t = time.perf_counter()
            gc.collect()
            try:
                torch.npu.empty_cache()
            except Exception:
                logger.exception(
                    "Failed to empty NPU cache before upward compact: rank=%s",
                    self.rank,
                )
            pre_compact_empty_cache_ms = (
                time.perf_counter() - pre_compact_empty_cache_start_t
            ) * 1000.0
        for module in model.modules():
            if not _is_ascend_fused_moe_module(module):
                continue
            if int(getattr(module, "elastic_execution_mode", 0) or 0) != 1:
                continue
            if not _module_uses_lossless_elastic(module):
                continue
            restore_fn = getattr(module,
                                 "restore_lossless_full_world_primary_layout",
                                 None)
            if callable(restore_fn):
                restore_fn()
            elif hasattr(module, "reset_expert_map_and_log2phy"):
                module.reset_expert_map_and_log2phy()
            release_transient_fn = getattr(
                module, "release_mode1_full_world_transient_state", None)
            if callable(release_transient_fn):
                release_transient_fn()
            if hasattr(module, "_mode3_double_buffer_manager"):
                try:
                    released_model_slot_caches += _release_mode1_double_buffer_owner(
                        getattr(module, "_mode3_double_buffer_manager", None))
                    setattr(module, "_mode3_double_buffer_manager", None)
                except Exception:
                    logger.exception(
                        "Failed to release module mode1 manager cache during "
                        "upward-floor cleanup: rank=%s layer=%s",
                        self.rank,
                        getattr(module, "layer_idx", "?"),
                    )
            compact_fn = getattr(module,
                                 "shrink_lossless_loaded_weights_to_primary",
                                 None)
            target_loaded_capacity = target_loaded_capacity_hint
            try:
                reserve_fn = getattr(module,
                                     "_get_reserved_local_expert_slots_for_floor",
                                     None)
                if callable(reserve_fn):
                    target_loaded_capacity = int(reserve_fn(int(target_floor)))
            except Exception:
                target_loaded_capacity = target_loaded_capacity_hint
            if callable(compact_fn) and compact_fn(
                    min_capacity=target_loaded_capacity):
                compacted_layers += 1
                if compacted_layers % compact_release_every == 0:
                    if _env_flag(
                            "VLLM_ASCEND_MODE1_CLEAR_STALE_PARAMS_DURING_UPWARD_COMPACT",
                            "0"):
                        try:
                            (inc_dicts, inc_entries, inc_bytes) = (
                                _clear_stale_mode1_fullname_parameter_dicts(
                                    model))
                            incremental_stale_param_dicts += inc_dicts
                            incremental_stale_param_entries += inc_entries
                            incremental_stale_param_bytes += inc_bytes
                        except Exception:
                            logger.exception(
                                "Failed to clear stale mode1 parameter dicts during "
                                "upward compact: rank=%s compacted_layers=%s",
                                self.rank,
                                compacted_layers,
                            )
                    if _env_flag(
                            "VLLM_ASCEND_MODE1_EMPTY_CACHE_DURING_UPWARD_COMPACT",
                            "0"):
                        gc.collect()
                        try:
                            torch.npu.empty_cache()
                            torch.npu.synchronize()
                        except Exception:
                            logger.exception(
                                "Failed to empty/sync NPU cache during upward "
                                "compact: rank=%s compacted_layers=%s",
                                self.rank,
                                compacted_layers,
                            )
            target_loaded_capacity_hist[target_loaded_capacity] = (
                target_loaded_capacity_hist.get(target_loaded_capacity, 0) + 1)
            _set_module_active_expert_mask(module, None)
            _set_module_elastic_runtime_log2phy(module, None)
            original_num_experts = getattr(module,
                                           "elastic_original_num_experts",
                                           None)
            if original_num_experts is not None:
                module.num_experts = int(original_num_experts)
                module.moe_config.num_experts = int(original_num_experts)
            quant_method = getattr(module, "quant_method", None)
            if (quant_method is not None and hasattr(
                    quant_method,
                    "invalidate_lossless_runtime_state_for_reload")):
                quant_method.invalidate_lossless_runtime_state_for_reload(
                    layer=module, reason="mode1_upward_floor_kv_resize")
            else:
                module.runtime_w13_weight = None
                module.runtime_w2_weight = None
                module.runtime_w13_buffer = None
                module.runtime_w2_buffer = None
                module.runtime_weight_capacity = 0
                module.lossless_runtime_activated = False
            restored_layers += 1
            try:
                w13 = getattr(module, "w13_weight", None)
                w2 = getattr(module, "w2_weight", None)
                for tensor in (w13, w2):
                    if isinstance(tensor, torch.Tensor):
                        try:
                            moe_weight_bytes += int(
                                tensor.untyped_storage().nbytes())
                        except Exception:
                            moe_weight_bytes += (int(tensor.numel()) *
                                                 int(tensor.element_size()))
                runtime_w13 = getattr(module, "runtime_w13_buffer", None)
                runtime_w2 = getattr(module, "runtime_w2_buffer", None)
                for tensor in (runtime_w13, runtime_w2):
                    if isinstance(tensor, torch.Tensor):
                        try:
                            moe_runtime_bytes += int(
                                tensor.untyped_storage().nbytes())
                        except Exception:
                            moe_runtime_bytes += (int(tensor.numel()) *
                                                  int(tensor.element_size()))
                capacity = int(getattr(module, "loaded_weight_capacity", -1))
                moe_loaded_capacity_hist[capacity] = (
                    moe_loaded_capacity_hist.get(capacity, 0) + 1)
                shape_key = (
                    f"{tuple(getattr(w13, 'shape', ()))}/"
                    f"{tuple(getattr(w2, 'shape', ()))}")
                moe_weight_shape_hist[shape_key] = (
                    moe_weight_shape_hist.get(shape_key, 0) + 1)
            except Exception:
                logger.exception(
                    "Failed to collect mode1 upward-floor cleanup memory stats: "
                    "rank=%s",
                    self.rank,
                )
        restore_layout_ms = (time.perf_counter() -
                             restore_layout_start_t) * 1000.0

        post_compact_sync_start_t = time.perf_counter()
        post_compact_sync_ms = 0.0
        if compacted_layers > 0 and _env_flag(
                "VLLM_ASCEND_MODE1_EMPTY_CACHE_AFTER_UPWARD_COMPACT", "1"):
            gc.collect()
            try:
                torch.npu.empty_cache()
                torch.npu.synchronize()
            except Exception:
                logger.exception(
                    "Failed to empty/sync NPU cache after upward compact: rank=%s",
                    self.rank,
                )
            post_compact_sync_ms = (
                time.perf_counter() - post_compact_sync_start_t) * 1000.0

        eplb_start_t = time.perf_counter()
        refreshed_eplb_adaptors = 0
        try:
            refreshed_eplb_adaptors = (
                _refresh_mode1_eplb_adaptor_references(self.model_runner))
        except Exception:
            logger.exception(
                "Failed to refresh mode1 EPLB cached parameter references "
                "during upward-floor cleanup: rank=%s",
                self.rank,
            )
        eplb_ms = (time.perf_counter() - eplb_start_t) * 1000.0

        stale_param_start_t = time.perf_counter()
        cleared_stale_param_dicts = 0
        cleared_stale_param_entries = 0
        cleared_stale_param_bytes = 0
        try:
            (cleared_stale_param_dicts, cleared_stale_param_entries,
             cleared_stale_param_bytes) = (
                 _clear_stale_mode1_fullname_parameter_dicts(model))
            cleared_stale_param_dicts += incremental_stale_param_dicts
            cleared_stale_param_entries += incremental_stale_param_entries
            cleared_stale_param_bytes += incremental_stale_param_bytes
        except Exception:
            logger.exception(
                "Failed to clear stale mode1 full-name parameter dicts during "
                "upward-floor cleanup: rank=%s",
                self.rank,
            )
            cleared_stale_param_dicts += incremental_stale_param_dicts
            cleared_stale_param_entries += incremental_stale_param_entries
            cleared_stale_param_bytes += incremental_stale_param_bytes
        stale_param_ms = (time.perf_counter() - stale_param_start_t) * 1000.0

        sync_start_t = time.perf_counter()
        if _env_flag("VLLM_ASCEND_MODE1_UPWARD_CLEANUP_FINAL_EMPTY_CACHE",
                     "1"):
            gc.collect()
            try:
                torch.npu.empty_cache()
                torch.npu.synchronize()
            except Exception:
                logger.exception(
                    "Failed to empty/sync NPU cache after upward-floor cleanup")
        release_ms = (restore_layout_start_t - release_start_t) * 1000.0
        sync_ms = (time.perf_counter() - sync_start_t) * 1000.0
        stats.update({
            "changed": float(restored_layers > 0 or compacted_layers > 0
                             or released_model_slot_caches > 0
                             or cleared_stale_param_entries > 0),
            "restored_layers": restored_layers,
            "compacted_layers": compacted_layers,
            "released_model_slot_caches": released_model_slot_caches,
            "refreshed_eplb_adaptors": refreshed_eplb_adaptors,
            "cleared_stale_param_dicts": cleared_stale_param_dicts,
            "cleared_stale_param_entries": cleared_stale_param_entries,
            "cleared_stale_param_bytes": cleared_stale_param_bytes,
            "moe_weight_bytes": moe_weight_bytes,
            "moe_runtime_bytes": moe_runtime_bytes,
            "release_ms": release_ms,
            "pre_compact_empty_cache_ms": pre_compact_empty_cache_ms,
            "restore_layout_ms": restore_layout_ms,
            "eplb_ms": eplb_ms,
            "stale_param_ms": stale_param_ms,
            "sync_ms": sync_ms,
            "post_compact_sync_ms": post_compact_sync_ms,
            "target_loaded_capacity": target_loaded_capacity_hint,
            "target_loaded_capacity_hist": target_loaded_capacity_hist,
        })
        if restored_layers > 0:
            logger.info(
                "Mode1 upward floor cleanup compacted low-floor MoE residue: "
                "rank=%s previous_floor=%s target_floor=%s world_size=%s "
                "target_loaded_capacity_hint=%s target_loaded_capacity_hist=%s "
                "layers=%s compacted_layers=%s "
                "released_model_slot_caches=%s refreshed_eplb_adaptors=%s "
                "cleared_stale_param_dicts=%s cleared_stale_param_entries=%s "
                "cleared_stale_param_bytes=%s moe_weight_bytes=%s "
                "moe_runtime_bytes=%s loaded_capacity_hist=%s "
                "weight_shape_hist=%s post_compact_sync_ms=%.2f",
                self.rank,
                previous_floor,
                target_floor,
                world_size,
                target_loaded_capacity_hint,
                target_loaded_capacity_hist,
                restored_layers,
                compacted_layers,
                released_model_slot_caches,
                refreshed_eplb_adaptors,
                cleared_stale_param_dicts,
                cleared_stale_param_entries,
                cleared_stale_param_bytes,
                moe_weight_bytes,
                moe_runtime_bytes,
                moe_loaded_capacity_hist,
                moe_weight_shape_hist,
                post_compact_sync_ms,
            )
        return stats

    def prepare_mode1_step_floor_for_kv_resize(
            self,
            target_floor: int,
            previous_floor: Optional[int] = None) -> bool:
        """Adjust mode=1 resident MoE slots before per-step KV resizing.

        A full-world step needs the largest KV cache budget and should not keep
        the floor-N runtime state resident.  When target_floor reaches the
        world size the step is semantically no-shrink, so restore the primary
        full-world layout and clear per-shrink masks instead of keeping the
        low-floor parity runtime active.
        """
        total_start_t = time.perf_counter()
        timeline_log = _env_flag("VLLM_ASCEND_MODE1_STEP_TIMELINE_LOG", "1")
        try:
            target_floor = int(target_floor)
        except (TypeError, ValueError):
            return False
        try:
            previous_floor_int = (int(previous_floor)
                                  if previous_floor is not None else None)
        except (TypeError, ValueError):
            previous_floor_int = None
        world_size = int(torch.distributed.get_world_size()
                         if torch.distributed.is_available()
                         and torch.distributed.is_initialized() else 1)
        upward_floor_transition = (
            previous_floor_int is not None
            and target_floor > previous_floor_int
            and previous_floor_int > 0)
        if timeline_log:
            logger.info(
                "Mode1 step timeline: floor_prepare_worker_start rank=%s "
                "target_floor=%s previous_floor=%s world_size=%s "
                "upward_floor_transition=%s",
                self.rank,
                target_floor,
                previous_floor_int,
                world_size,
                upward_floor_transition,
            )
        prune_start_t = time.perf_counter()
        prune_stats = self._prune_mode1_planned_floor_groups_for_current_step(
            target_floor, "kv_resize_prepare")
        prune_ms = (time.perf_counter() - prune_start_t) * 1000.0
        if target_floor < world_size:
            cleanup_stats: dict[str, Any] = {}
            cleanup_ms = 0.0
            if (upward_floor_transition and _env_flag(
                    "VLLM_ASCEND_MODE1_CLEANUP_ON_UPWARD_FLOOR_RESIZE",
                    "1")):
                cleanup_start_t = time.perf_counter()
                cleanup_stats = (
                    self._cleanup_mode1_upward_floor_residue_for_kv_resize(
                        target_floor, previous_floor_int, world_size))
                cleanup_ms = (time.perf_counter() -
                              cleanup_start_t) * 1000.0
            precreate_start_t = time.perf_counter()
            precreate_stats = (
                self._precreate_mode1_planned_floor_groups_before_kv_cache(
                    prefill_expert_slots=False))
            precreate_ms = (time.perf_counter() - precreate_start_t) * 1000.0
            if timeline_log:
                logger.info(
                    "Mode1 step timeline: floor_prepare_worker_done rank=%s "
                    "target_floor=%s previous_floor=%s branch=shrink "
                    "upward_floor_transition=%s prune_ms=%.2f "
                    "upward_cleanup_ms=%.2f precreate_ms=%.2f "
                    "total_ms=%.2f prune_changed=%.0f "
                    "cleanup_changed=%.0f cleanup_restored_layers=%s "
                    "cleanup_compacted_layers=%s "
                    "cleanup_cleared_stale_param_bytes=%s "
                    "cleanup_release_ms=%.2f "
                    "cleanup_pre_compact_empty_cache_ms=%.2f "
                    "cleanup_restore_layout_ms=%.2f cleanup_eplb_ms=%.2f "
                    "cleanup_stale_param_ms=%.2f "
                    "cleanup_sync_ms=%.2f cleanup_post_compact_sync_ms=%.2f "
                    "precreate_stage_groups=%.0f",
                    self.rank,
                    target_floor,
                    previous_floor_int,
                    upward_floor_transition,
                    prune_ms,
                    cleanup_ms,
                    precreate_ms,
                    (time.perf_counter() - total_start_t) * 1000.0,
                    prune_stats.get("changed", 0.0),
                    cleanup_stats.get("changed", 0.0),
                    cleanup_stats.get("restored_layers", 0),
                    cleanup_stats.get("compacted_layers", 0),
                    cleanup_stats.get("cleared_stale_param_bytes", 0),
                    cleanup_stats.get("release_ms", 0.0),
                    cleanup_stats.get("pre_compact_empty_cache_ms", 0.0),
                    cleanup_stats.get("restore_layout_ms", 0.0),
                    cleanup_stats.get("eplb_ms", 0.0),
                    cleanup_stats.get("stale_param_ms", 0.0),
                    cleanup_stats.get("sync_ms", 0.0),
                    cleanup_stats.get("post_compact_sync_ms", 0.0),
                    precreate_stats.get("stage_groups", 0.0),
                )
            return bool(
                prune_stats.get("changed", 0.0) > 0.0
                or cleanup_stats.get("changed", 0.0) > 0.0
                or precreate_stats.get("stage_groups", 0.0) > 0.0)

        model = None
        try:
            model = self.model_runner.get_model()
        except Exception:
            logger.exception("Failed to fetch model for mode1 floor preparation")
            return False
        if model is None:
            return False

        released_model_slot_caches = 0
        release_start_t = time.perf_counter()
        try:
            from vllm.forward_context import get_forward_context
            try:
                forward_context = get_forward_context()
            except AssertionError:
                forward_context = None
            if hasattr(forward_context, "moe_double_buffer_manager"):
                released_model_slot_caches += _release_mode1_double_buffer_owner(
                    getattr(forward_context, "moe_double_buffer_manager", None))
                setattr(forward_context, "moe_double_buffer_manager", None)
                released_model_slot_caches += 1
        except Exception:
            logger.exception(
                "Failed to release forward-context mode1 double-buffer cache: "
                "rank=%s",
                self.rank,
            )
        if hasattr(model, "_mode3_double_buffer_manager"):
            try:
                released_model_slot_caches += _release_mode1_double_buffer_owner(
                    getattr(model, "_mode3_double_buffer_manager", None))
                setattr(model, "_mode3_double_buffer_manager", None)
                released_model_slot_caches += 1
            except Exception:
                logger.exception(
                    "Failed to release model-level mode1 manager cache: "
                    "rank=%s",
                    self.rank,
                )
        for slot_attr in ("_mode3_cpu_npu_double_buffer_slots",
                          "_mode4_remote_npu_double_buffer_slots"):
            if hasattr(model, slot_attr):
                try:
                    released_model_slot_caches += _release_mode1_double_buffer_owner(
                        getattr(model, slot_attr, None))
                    setattr(model, slot_attr, None)
                    released_model_slot_caches += 1
                except Exception:
                    logger.exception(
                        "Failed to release model-level mode1 slot cache: "
                        "rank=%s attr=%s",
                        self.rank,
                        slot_attr,
	                    )

        restored_layers = 0
        compacted_layers = 0
        refreshed_eplb_adaptors = 0
        cleared_stale_param_dicts = 0
        cleared_stale_param_entries = 0
        cleared_stale_param_bytes = 0
        moe_weight_bytes = 0
        moe_runtime_bytes = 0
        moe_loaded_capacity_hist: dict[int, int] = {}
        moe_weight_shape_hist: dict[str, int] = {}
        restore_layout_start_t = time.perf_counter()
        for module in model.modules():
            if not _is_ascend_fused_moe_module(module):
                continue
            if int(getattr(module, "elastic_execution_mode", 0) or 0) != 1:
                continue
            if not _module_uses_lossless_elastic(module):
                continue
            restore_fn = getattr(module,
                                 "restore_lossless_full_world_primary_layout",
                                 None)
            if callable(restore_fn):
                restore_fn()
            elif hasattr(module, "reset_expert_map_and_log2phy"):
                module.reset_expert_map_and_log2phy()
            release_transient_fn = getattr(
                module, "release_mode1_full_world_transient_state", None)
            if callable(release_transient_fn):
                release_transient_fn()
            if hasattr(module, "_mode3_double_buffer_manager"):
                try:
                    released_model_slot_caches += _release_mode1_double_buffer_owner(
                        getattr(module, "_mode3_double_buffer_manager", None))
                    setattr(module, "_mode3_double_buffer_manager", None)
                except Exception:
                    logger.exception(
                        "Failed to release module mode1 manager cache: "
                        "rank=%s layer=%s",
                        self.rank,
                        getattr(module, "layer_idx", "?"),
                    )
            compact_fn = getattr(module,
                                 "shrink_lossless_loaded_weights_to_primary",
                                 None)
            if callable(compact_fn) and compact_fn():
                compacted_layers += 1
            _set_module_active_expert_mask(module, None)
            _set_module_elastic_runtime_log2phy(module, None)
            original_num_experts = getattr(module,
                                           "elastic_original_num_experts",
                                           None)
            if original_num_experts is not None:
                module.num_experts = int(original_num_experts)
                module.moe_config.num_experts = int(original_num_experts)
            quant_method = getattr(module, "quant_method", None)
            if (quant_method is not None and hasattr(
                    quant_method,
                    "invalidate_lossless_runtime_state_for_reload")):
                quant_method.invalidate_lossless_runtime_state_for_reload(
                    layer=module, reason="mode1_full_world_step_floor")
            else:
                module.runtime_w13_weight = None
                module.runtime_w2_weight = None
                module.runtime_w13_buffer = None
                module.runtime_w2_buffer = None
                module.runtime_weight_capacity = 0
                module.lossless_runtime_activated = False
            restored_layers += 1
            try:
                w13 = getattr(module, "w13_weight", None)
                w2 = getattr(module, "w2_weight", None)
                for tensor in (w13, w2):
                    if isinstance(tensor, torch.Tensor):
                        try:
                            moe_weight_bytes += int(
                                tensor.untyped_storage().nbytes())
                        except Exception:
                            moe_weight_bytes += (int(tensor.numel()) *
                                                 int(tensor.element_size()))
                runtime_w13 = getattr(module, "runtime_w13_buffer", None)
                runtime_w2 = getattr(module, "runtime_w2_buffer", None)
                for tensor in (runtime_w13, runtime_w2):
                    if isinstance(tensor, torch.Tensor):
                        try:
                            moe_runtime_bytes += int(
                                tensor.untyped_storage().nbytes())
                        except Exception:
                            moe_runtime_bytes += (int(tensor.numel()) *
                                                  int(tensor.element_size()))
                capacity = int(getattr(module, "loaded_weight_capacity", -1))
                moe_loaded_capacity_hist[capacity] = (
                    moe_loaded_capacity_hist.get(capacity, 0) + 1)
                shape_key = (
                    f"{tuple(getattr(w13, 'shape', ()))}/"
                    f"{tuple(getattr(w2, 'shape', ()))}")
                moe_weight_shape_hist[shape_key] = (
                    moe_weight_shape_hist.get(shape_key, 0) + 1)
            except Exception:
                logger.exception(
                    "Failed to collect mode1 MoE compact memory stats: rank=%s",
                    self.rank,
                )
        restore_layout_ms = (time.perf_counter() - restore_layout_start_t) * 1000.0

        eplb_start_t = time.perf_counter()
        try:
            refreshed_eplb_adaptors = (
                _refresh_mode1_eplb_adaptor_references(self.model_runner))
        except Exception:
            logger.exception(
                "Failed to refresh mode1 EPLB cached parameter references: "
                "rank=%s",
                self.rank,
            )
        eplb_ms = (time.perf_counter() - eplb_start_t) * 1000.0

        stale_param_start_t = time.perf_counter()
        try:
            (cleared_stale_param_dicts, cleared_stale_param_entries,
             cleared_stale_param_bytes) = (
                 _clear_stale_mode1_fullname_parameter_dicts(model))
        except Exception:
            logger.exception(
                "Failed to clear stale mode1 full-name parameter dicts: rank=%s",
                self.rank,
            )
        stale_param_ms = (time.perf_counter() - stale_param_start_t) * 1000.0

        sync_start_t = time.perf_counter()
        gc.collect()
        try:
            torch.npu.empty_cache()
            torch.npu.synchronize()
        except Exception:
            logger.exception("Failed to empty/sync NPU cache after floor prep")
        release_ms = (restore_layout_start_t - release_start_t) * 1000.0
        sync_ms = (time.perf_counter() - sync_start_t) * 1000.0
        if restored_layers > 0:
            logger.info(
                "Mode1 step floor preparation restored full-world layout: "
                "rank=%s target_floor=%s world_size=%s layers=%s "
                "compacted_layers=%s "
                "released_model_slot_caches=%s refreshed_eplb_adaptors=%s "
                "cleared_stale_param_dicts=%s cleared_stale_param_entries=%s "
                "cleared_stale_param_bytes=%s "
                "moe_weight_bytes=%s "
                "moe_runtime_bytes=%s loaded_capacity_hist=%s "
                "weight_shape_hist=%s group_refresh=deferred",
                self.rank,
                target_floor,
                world_size,
                restored_layers,
                compacted_layers,
                released_model_slot_caches,
                refreshed_eplb_adaptors,
                cleared_stale_param_dicts,
                cleared_stale_param_entries,
                cleared_stale_param_bytes,
                moe_weight_bytes,
                moe_runtime_bytes,
                moe_loaded_capacity_hist,
                moe_weight_shape_hist,
            )
        if timeline_log:
            logger.info(
                "Mode1 step timeline: floor_prepare_worker_done rank=%s "
                "target_floor=%s branch=fullworld prune_ms=%.2f "
                "release_ms=%.2f restore_layout_ms=%.2f eplb_ms=%.2f "
                "stale_param_ms=%.2f sync_ms=%.2f total_ms=%.2f "
                "restored_layers=%s compacted_layers=%s "
                "released_model_slot_caches=%s",
                self.rank,
                target_floor,
                prune_ms,
                release_ms,
                restore_layout_ms,
                eplb_ms,
                stale_param_ms,
                sync_ms,
                (time.perf_counter() - total_start_t) * 1000.0,
                restored_layers,
                compacted_layers,
                released_model_slot_caches,
            )
        return restored_layers > 0

    def refresh_mode1_step_floor_groups_for_kv_resize(
            self, target_floor: int) -> bool:
        """Refresh MoE communication groups after old KV cache is released."""
        try:
            target_floor = int(target_floor)
        except (TypeError, ValueError):
            return False
        world_size = int(torch.distributed.get_world_size()
                         if torch.distributed.is_available()
                         and torch.distributed.is_initialized() else 1)
        if target_floor < world_size:
            return False

        model = None
        try:
            model = self.model_runner.get_model()
        except Exception:
            logger.exception("Failed to fetch model for mode1 group refresh")
            return False
        if model is None:
            return False

        refreshed_layers = 0
        warmup_ms = 0.0
        for module in model.modules():
            if not _is_ascend_fused_moe_module(module):
                continue
            if int(getattr(module, "elastic_execution_mode", 0) or 0) != 1:
                continue
            if not _module_uses_lossless_elastic(module):
                continue
            module.refresh_elastic_groups()
            refreshed_layers += 1

        if (refreshed_layers > 0 and _env_flag(
                "VLLM_ASCEND_MODE1_FULLWORLD_REFRESH_MC2_WARMUP", "0")):
            old_phase = os.environ.get(
                "VLLM_ASCEND_MODE1_CURRENT_DISPATCH_PHASE")
            os.environ["VLLM_ASCEND_MODE1_CURRENT_DISPATCH_PHASE"] = (
                "step_floor_fullworld_refresh_warmup")
            warmup_start_t = time.perf_counter()
            try:
                self._warmup_post_restore_mc2_dispatch_for_custom_mode1_parity(
                    world_size)
            finally:
                if old_phase is None:
                    os.environ.pop(
                        "VLLM_ASCEND_MODE1_CURRENT_DISPATCH_PHASE", None)
                else:
                    os.environ[
                        "VLLM_ASCEND_MODE1_CURRENT_DISPATCH_PHASE"] = old_phase
            warmup_ms = (time.perf_counter() - warmup_start_t) * 1000.0

        gc.collect()
        try:
            torch.npu.empty_cache()
            torch.npu.synchronize()
        except Exception:
            logger.exception("Failed to empty/sync NPU cache after group refresh")
        if refreshed_layers > 0:
            logger.info(
                "Mode1 step floor MoE groups refreshed after old KV release: "
                "rank=%s target_floor=%s world_size=%s layers=%s "
                "fullworld_refresh_mc2_warmup=%s warmup_ms=%.2f",
                self.rank,
                target_floor,
                world_size,
                refreshed_layers,
                int(_env_flag("VLLM_ASCEND_MODE1_FULLWORLD_REFRESH_MC2_WARMUP",
                              "0")),
                warmup_ms,
            )
        return refreshed_layers > 0

    def mode1_resize_world_barrier(self, tag: str) -> None:
        """Synchronize all rollout workers between step-level resize phases."""
        if not (torch.distributed.is_available()
                and torch.distributed.is_initialized()):
            return
        try:
            torch.npu.synchronize()
        except Exception:
            logger.exception("Failed to sync NPU before mode1 resize barrier")
        start = time.time()
        torch.distributed.barrier()
        elapsed_ms = (time.time() - start) * 1000.0
        logger.info(
            "Mode1 adaptive KV resize world barrier done: rank=%s tag=%s "
            "elapsed_ms=%.2f",
            self.rank,
            tag,
            elapsed_ms,
        )

    def get_mode1_resize_available_kv_memory(
            self, headroom_bytes: int = 0) -> int:
        """Return currently free NPU memory usable for a resized KV cache."""
        try:
            headroom_bytes = max(int(headroom_bytes), 0)
        except (TypeError, ValueError):
            headroom_bytes = 0
        gc.collect()
        try:
            torch.npu.empty_cache()
            torch.npu.synchronize()
            free_bytes, total_bytes = torch.npu.mem_get_info()
            stats = torch_npu.npu.memory_stats()
            torch_current = int(stats.get("allocated_bytes.all.current", 0))
            torch_reserved = int(stats.get("reserved_bytes.all.current", 0))
            total_allocated = int(total_bytes - free_bytes)
            non_torch = max(total_allocated - torch_current, 0)
            available = max(int(free_bytes) - headroom_bytes, 0)
            logger.info(
                "Mode1 adaptive KV resize memory: rank=%s free_bytes=%s "
                "total_bytes=%s torch_current=%s torch_reserved=%s "
                "non_torch=%s total_allocated=%s headroom_bytes=%s "
                "available_for_kv=%s",
                self.rank,
                free_bytes,
                total_bytes,
                torch_current,
                torch_reserved,
                non_torch,
                total_allocated,
                headroom_bytes,
                available,
            )
            return available
        except Exception:
            logger.exception(
                "Failed to query current free NPU memory for mode1 KV resize")
            return 0

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
        if getattr(self, "elastic_parallel_detached", False):
            logger.info(
                "Elastic dummy batch skipped on detached rank: rank=%s",
                self.rank,
            )
            return
        with self._maybe_elastic_op_profile("dummy"):
            self.model_runner._dummy_run(1)

    def set_elastic_op_profile_context(self, context: Optional[dict]) -> None:
        """Arm a one-shot op profiler for the next live or dummy step."""
        self._elastic_op_profile_context = context

    def _elastic_op_profile_rank_enabled(self) -> bool:
        ranks = os.getenv("VLLM_ASCEND_BUCKET_OP_PROFILE_RANKS", "all").strip()
        if not ranks or ranks.lower() == "all":
            return True
        try:
            allowed = {int(item) for item in ranks.split(",") if item.strip()}
        except ValueError:
            logger.warning(
                "Invalid VLLM_ASCEND_BUCKET_OP_PROFILE_RANKS=%s; profiling all ranks",
                ranks)
            return True
        return int(self.rank) in allowed

    @contextmanager
    def _maybe_elastic_op_profile(self, kind: str):
        context = self._elastic_op_profile_context
        self._elastic_op_profile_context = None
        root = os.getenv("VLLM_ASCEND_BUCKET_OP_PROFILE_DIR", "")
        if (not context or not root
                or not self._elastic_op_profile_rank_enabled()):
            yield
            return

        bucket = context.get("bucket", "unknown")
        step = context.get("step", "unknown")
        active_count = context.get("active_count", "unknown")
        compute_world_size = context.get("compute_world_size", "unknown")
        profile_dir = (Path(root) / f"bucket_{bucket}" /
                       f"rank_{self.rank}_{kind}_step_{step}")
        profile_dir.mkdir(parents=True, exist_ok=True)
        (profile_dir / "elastic_op_profile_meta.txt").write_text(
            "\n".join([
                f"rank={self.rank}",
                f"local_rank={self.local_rank}",
                f"kind={kind}",
                f"bucket={bucket}",
                f"step={step}",
                f"active_count={active_count}",
                f"compute_world_size={compute_world_size}",
                f"active_ranks={context.get('active_ranks')}",
            ]) + "\n",
            encoding="utf-8")

        level_name = os.getenv("VLLM_ASCEND_BUCKET_OP_PROFILE_LEVEL", "level0")
        if level_name == "level_none":
            level = torch_npu.profiler.ProfilerLevel.Level_none
        elif level_name == "level1":
            level = torch_npu.profiler.ProfilerLevel.Level1
        elif level_name == "level2":
            level = torch_npu.profiler.ProfilerLevel.Level2
        else:
            level = torch_npu.profiler.ProfilerLevel.Level0

        analysis = os.getenv("VLLM_ASCEND_BUCKET_OP_PROFILE_ANALYSIS",
                             "1").lower() not in ("0", "false", "no", "off")
        sync = os.getenv("VLLM_ASCEND_BUCKET_OP_PROFILE_SYNC",
                         "1").lower() not in ("0", "false", "no", "off")
        contents = [
            item.strip().lower()
            for item in os.getenv("VLLM_ASCEND_BUCKET_OP_PROFILE_CONTENTS",
                                  "npu,cpu").split(",") if item.strip()
        ]
        enable_mstx = "mstx" in contents
        marker_only = (enable_mstx and "npu" not in contents
                       and "cpu" not in contents)
        experimental_config = torch_npu.profiler._ExperimentalConfig(
            export_type=torch_npu.profiler.ExportType.Text,
            profiler_level=level,
            msprof_tx=not marker_only,
            mstx=enable_mstx,
            aic_metrics=torch_npu.profiler.AiCMetrics.AiCoreNone,
            l2_cache=False,
            op_attr=False,
            data_simplification=True,
            record_op_args=False,
            gc_detect_threshold=None,
        )
        activities = []
        if not marker_only:
            if "cpu" in contents:
                activities.append(torch_npu.profiler.ProfilerActivity.CPU)
            if "npu" in contents:
                activities.append(torch_npu.profiler.ProfilerActivity.NPU)
        with torch_npu.profiler.profile(
                activities=activities,
                with_stack=False,
                profile_memory=False,
                with_modules=False,
                record_shapes=False,
                on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(
                    str(profile_dir), analyse_flag=analysis),
                experimental_config=experimental_config) as prof:
            yield
            if sync:
                torch.npu.synchronize()
            prof.step()
        logger.info(
            "Elastic op profile captured: rank=%s kind=%s bucket=%s step=%s dir=%s",
            self.rank, kind, bucket, step, profile_dir)

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

    def _original_local_elastic_group_ranks(
            self, world_size: int) -> list[int]:
        current_rank = int(torch.distributed.get_rank())
        for ranks in self._build_original_ep_group_ranks(world_size):
            if current_rank in ranks:
                return [int(rank) for rank in ranks]
        raise RuntimeError(
            f"global rank {current_rank} is absent from original EP groups")

    def _refresh_elastic_parallel_state(self,
                                        active_ranks: list[int],
                                        world_group,
                                        participate_only: bool = False) -> None:
        current_rank = torch.distributed.get_rank()
        is_active_rank = current_rank in active_ranks
        restoring_full_world = (
            sorted(map(int, active_ranks))
            == self._original_local_elastic_group_ranks(
                int(torch.distributed.get_world_size())))
        aggregate_mode5_hybrid_build_sources_ms = 0.0
        aggregate_mode5_hybrid_activate_ms = 0.0
        aggregate_mode5_hybrid_build_cpu_shadow_ms = 0.0
        aggregate_mode5_hybrid_materialize_resident_ms = 0.0
        aggregate_mode5_hybrid_log2phy_ms = 0.0
        aggregate_mode5_hybrid_set_runtime_ms = 0.0
        aggregate_mode5_hybrid_refresh_groups_ms = 0.0
        aggregate_mode5_hybrid_total_ms = 0.0
        aggregate_mode5_hybrid_layers = 0
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
            if not _is_ascend_fused_moe_module(module):
                continue

            use_lossless_mode = (
                envs_ascend.VLLM_ASCEND_ELASTIC_MOE_MODE == "lossless"
                and getattr(module, "elastic_moe_mode", "lossy") == "lossless")

            if use_lossless_mode:
                payload = lossless_shrink_payload.get(module.layer_idx)
                if payload is None:
                    if restoring_full_world:
                        # The full-world restore invalidates the previous
                        # post-shrink communicator/workspace warm state. Even
                        # if a later shrink lands on the same active-rank
                        # signature, we must warm it again because the EP/MC2
                        # groups have been rebuilt from the restored 16-rank
                        # world in between. Fixed-topology reuse keeps those
                        # shrink groups alive, so their warm state remains valid.
                        if not self._mode1_fixed_topology_group_reuse_enabled():
                            self._post_shrink_moe_dispatch_warmed_active_signatures.clear()
                        current_dp_group = get_dp_group()
                        current_ep_group = get_ep_group()
                        _set_module_active_expert_mask(module, None)
                        _set_module_elastic_runtime_log2phy(module, None)
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
                mode4_remote_sources: dict[int, tuple[int, int]] = {}
                module_mode = int(getattr(module, "elastic_execution_mode", 0))
                if module_mode in (4, 5):
                    source_rank_by_expert = payload.get(
                        "cpu_import_source_rank", {})
                    remote_source_rank_by_expert = _mode4_remote_source_rank_map(
                        payload)
                    source_slot_by_expert = payload.get(
                        "remote_import_source_slot", {})
                    target_rank_by_expert = payload.get(
                        "cpu_import_target_rank", {})
                    mode5_remote_experts = set(
                        int(x) for x in payload.get("mode5_remote_experts", []))
                    for expert_id, source_rank in source_rank_by_expert.items():
                        expert_id = int(expert_id)
                        if module_mode == 5 and expert_id not in mode5_remote_experts:
                            continue
                        if target_rank_by_expert.get(expert_id) != current_rank:
                            continue
                        source_slot = source_slot_by_expert.get(expert_id)
                        remote_source_rank = remote_source_rank_by_expert.get(
                            expert_id, source_rank)
                        if remote_source_rank is None or source_slot is None:
                            continue
                        mode4_remote_sources[expert_id] = (
                            int(remote_source_rank), int(source_slot))
                if participate_only:
                    _set_module_active_expert_mask(module, None)
                    _set_module_elastic_runtime_log2phy(module, None)
                    module.moe_config.num_experts = module.elastic_original_num_experts
                    continue
                logical_num_experts = int(module.elastic_original_num_experts)
                # Do not blindly mark every logical expert as active on the
                # old custom mode=1 shrink path. Native/common mode1 keeps the
                # dispatch space aligned with the actual post-shrink runtime
                # mapping. A full logical-all-ones mask keeps router space
                # artificially wide and is a plausible reason why the real
                # post-shrink MC2 path remains heavier than the synthetic
                # warmup. Mode=3 still manages active experts through its own
                # double-buffer metadata and therefore keeps this mask unset.
                if getattr(module, "elastic_execution_mode", 0) == 3:
                    _set_module_active_expert_mask(module, None)
                else:
                    _set_module_active_expert_mask(module, None)
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
                parity_preloaded_direct_fill = False
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
                    parity_preloaded_direct_fill = (
                        getattr(module, "elastic_execution_mode", 0) == 1
                        and len(local_direct_import_slots) > 0
                        and len(local_active_expert_ids) > int(
                            getattr(module, "active_local_num_experts", 0)))
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
                module_mode = int(getattr(module, "elastic_execution_mode", 0))
                if module_mode in (3, 4, 5):
                    # Mode 3/4/5 are zero-redundancy double-buffer paths.  Even
                    # the first shrink target can exactly match the runtime
                    # resident capacity (for example floor=8 -> 16 experts),
                    # but it must still activate the hybrid metadata instead of
                    # falling through to mode=1 preallocated-slot materialization.
                    use_hybrid_cpu_swap = True
                elif in_followup_hybrid_stage:
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
                    hybrid_t0 = time.perf_counter()
                    if int(getattr(module, "elastic_execution_mode", 0)) in (4, 5):
                        module.lossless_mode4_remote_source_by_expert = (
                            mode4_remote_sources)
                        if int(getattr(module, "elastic_execution_mode", 0)) == 5:
                            source_rank_by_expert = payload.get(
                                "cpu_import_source_rank", {})
                            remote_source_rank_by_expert = (
                                _mode4_remote_source_rank_map(payload))
                            source_slot_by_expert = payload.get(
                                "remote_import_source_slot", {})
                            target_rank_by_expert = payload.get(
                                "cpu_import_target_rank", {})
                            mode5_full_remote_chain: dict[int,
                                                          tuple[int, int]] = {}
                            for expert_id, source_rank in source_rank_by_expert.items():
                                expert_id = int(expert_id)
                                if target_rank_by_expert.get(expert_id) != current_rank:
                                    continue
                                remote_source_rank = remote_source_rank_by_expert.get(
                                    expert_id, source_rank)
                                source_slot = source_slot_by_expert.get(expert_id)
                                if remote_source_rank is None or source_slot is None:
                                    continue
                                if int(source_slot) < 0:
                                    continue
                                mode5_full_remote_chain[expert_id] = (
                                    int(remote_source_rank), int(source_slot))
                            module.lossless_mode5_remote_source_chain_by_expert = (
                                mode5_full_remote_chain)
                        module.lossless_mode4_remote_fetcher = (
                            self._mode4_fetch_remote_experts_to_slot)
                    if int(getattr(module, "elastic_execution_mode", 0)) == 5:
                        prepare_mode5_shadow = getattr(
                            module, "prepare_lossless_mode5_cpu_shadow", None)
                        if callable(prepare_mode5_shadow):
                            prepare_ms = prepare_mode5_shadow(
                                local_active_expert_ids,
                                local_source_local_ids,
                                cpu_expert_weights=local_cpu_import_weights)
                            if prepare_ms > 0.0:
                                module.lossless_hybrid_last_stats[
                                    "mode5_cpu_shadow_build_ms"] = float(
                                        prepare_ms)
                    hybrid_t1 = time.perf_counter()
                    module.set_lossless_hybrid_global_layout(
                        active_ranks, ordered_assignments)
                    module.activate_lossless_hybrid_local_experts(
                        local_active_expert_ids,
                        local_source_local_ids,
                        cpu_expert_weights=local_cpu_import_weights)
                    hybrid_t2 = time.perf_counter()
                    new_log2phy_cpu = (
                        module._build_lossless_hybrid_runtime_log2phy(
                            module.lossless_hybrid_rank_resident_expert_ids))
                    module._set_lossless_hybrid_runtime_num_experts()
                    hybrid_t3 = time.perf_counter()
                    if module.layer_idx == 0:
                        mode = int(getattr(module, "elastic_execution_mode", 0))
                        if mode in (4, 5):
                            logger.info(
                                "Mode%s remote-NPU hybrid activated: rank=%s layer=%s active_ranks=%s owned_local=%s resident_capacity=%s remote_npu_local=%s cpu_only_local=%s runtime_num_experts=%s",
                                mode,
                                self.rank,
                                module.layer_idx,
                                active_ranks,
                                len(local_active_expert_ids),
                                int(module.lossless_hybrid_resident_capacity),
                                len(mode4_remote_sources),
                                len(module.lossless_hybrid_cpu_only_expert_ids),
                                int(module.moe_config.num_experts),
                            )
                        else:
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
                        if getattr(module, "elastic_execution_mode", 0) in (3, 4, 5):
                            logger.info(
                                "Mode%s cross-layer buffer activation: rank=%s layer=%s stage=%s owned_local=%s primary_prefix_rows=%s remote_npu_local=%s cpu_only_local=%s",
                                int(getattr(module, "elastic_execution_mode", 0)),
                                self.rank,
                                module.layer_idx,
                                len(active_ranks),
                                len(local_active_expert_ids),
                                int(getattr(module, "lossless_hybrid_resident_capacity", 0)),
                                len(mode4_remote_sources),
                                len(module.lossless_hybrid_cpu_only_expert_ids),
                            )
                else:
                    offload_loaded_after_activation = (
                        hasattr(module,
                                "should_offload_loaded_weights_after_lossless_activation")
                        and module
                        .should_offload_loaded_weights_after_lossless_activation())
                    if parity_preloaded_direct_fill:
                        setattr(module,
                                "_lossless_mode1_direct_preloaded_activation_ok",
                                True)
                    module.activate_lossless_local_experts(
                        local_active_expert_ids,
                        local_source_local_ids,
                        cpu_expert_weights=local_cpu_import_weights,
                        offload_loaded_after_activation=
                        offload_loaded_after_activation)
                    if parity_preloaded_direct_fill:
                        setattr(module,
                                "_lossless_mode1_direct_preloaded_activation_ok",
                                False)
                    new_log2phy_cpu = payload["runtime_log2phy_cpu"]
                    # Keep the old custom mode=1 runtime expert cardinality
                    # aligned with the post-shrink dense runtime mapping.
                    # Restoring `num_experts` back to the original logical
                    # expert count here makes the downstream MC2 dispatcher
                    # allocate/prepare for a larger expert space than the
                    # actual active dense runtime layout, which is the opposite
                    # of native/common mode=1 semantics.
                    if new_log2phy_cpu is not None:
                        try:
                            runtime_num_experts = int(
                                (new_log2phy_cpu >= 0).sum().item())
                        except Exception:
                            runtime_num_experts = logical_num_experts
                    else:
                        runtime_num_experts = logical_num_experts
                    module.set_runtime_num_experts(runtime_num_experts)
                    if (getattr(module, "elastic_execution_mode", 0) == 1
                            and new_log2phy_cpu is not None):
                        active_expert_mask = (new_log2phy_cpu >= 0).to(
                            device=module.expert_map.device,
                            dtype=torch.bool)
                        _set_module_active_expert_mask(module, active_expert_mask)
                if module.log2phy is not None and new_log2phy_cpu is not None:
                    _set_module_elastic_runtime_log2phy(module,
                        new_log2phy_cpu.to(device=module.log2phy.device,
                                           dtype=module.log2phy.dtype))
                else:
                    _set_module_elastic_runtime_log2phy(module, None)
                hybrid_t4 = time.perf_counter() if use_hybrid_cpu_swap else None
                # Rebuild the token dispatcher with the post-shrink local expert
                # count before decode resumes on the new 8-rank EP group.
                module.refresh_elastic_groups()
                hybrid_t5 = time.perf_counter() if use_hybrid_cpu_swap else None
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
                mapping_mismatch = 0
                if new_log2phy_cpu is not None:
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
                if use_hybrid_cpu_swap and int(
                        getattr(module, "elastic_execution_mode", 0)) == 5:
                    aggregate_mode5_hybrid_build_sources_ms += (
                        hybrid_t1 - hybrid_t0) * 1000.0
                    aggregate_mode5_hybrid_activate_ms += (
                        hybrid_t2 - hybrid_t1) * 1000.0
                    aggregate_mode5_hybrid_build_cpu_shadow_ms += float(
                        module.lossless_hybrid_last_stats.get(
                            "mode5_cpu_shadow_build_ms", 0.0))
                    aggregate_mode5_hybrid_materialize_resident_ms += float(
                        module.lossless_hybrid_last_stats.get(
                            "mode5_materialize_resident_ms", 0.0))
                    aggregate_mode5_hybrid_log2phy_ms += (
                        hybrid_t3 - hybrid_t2) * 1000.0
                    aggregate_mode5_hybrid_set_runtime_ms += (
                        hybrid_t4 - hybrid_t3) * 1000.0
                    aggregate_mode5_hybrid_refresh_groups_ms += (
                        hybrid_t5 - hybrid_t4) * 1000.0
                    aggregate_mode5_hybrid_total_ms += (
                        hybrid_t5 - hybrid_t0) * 1000.0
                    aggregate_mode5_hybrid_layers += 1
                if module.layer_idx == 0:
                    if use_hybrid_cpu_swap:
                        cpu_shadow_rows = int(
                            getattr(module, "lossless_mode5_legacy_cpu_shadow_rows",
                                    0))
                        mode5_prepared = int(
                            getattr(module, "lossless_mode5_cpu_shadow_prepared",
                                    False))
                        mode5_strategy = _mode5_cpu_shadow_runtime_strategy()
                        legacy_runtime = int(
                            mode5_strategy == "legacy_cpu_shadow")
                        logger.info(
                            "Mode%s hybrid refresh breakdown: rank=%s layer=%s stage=%s mode5_strategy=%s legacy_runtime=%s mode5_prepared=%s cpu_shadow_rows=%s build_sources_ms=%.2f activate_ms=%.2f build_cpu_shadow_ms=%.2f materialize_resident_ms=%.2f log2phy_ms=%.2f set_runtime_ms=%.2f refresh_groups_ms=%.2f total_ms=%.2f",
                            int(getattr(module, "elastic_execution_mode", 0)),
                            self.rank,
                            module.layer_idx,
                            len(active_ranks),
                            mode5_strategy,
                            legacy_runtime,
                            mode5_prepared,
                            cpu_shadow_rows,
                            (hybrid_t1 - hybrid_t0) * 1000.0,
                            (hybrid_t2 - hybrid_t1) * 1000.0,
                            float(module.lossless_hybrid_last_stats.get(
                                "mode5_cpu_shadow_build_ms", 0.0)),
                            float(module.lossless_hybrid_last_stats.get(
                                "mode5_materialize_resident_ms", 0.0)),
                            (hybrid_t3 - hybrid_t2) * 1000.0,
                            (hybrid_t4 - hybrid_t3) * 1000.0,
                            (hybrid_t5 - hybrid_t4) * 1000.0,
                            (hybrid_t5 - hybrid_t0) * 1000.0,
                        )
                    pass  # debug log removed
                continue

            if not is_active_rank:
                _set_module_active_expert_mask(module, None)
                _set_module_elastic_runtime_log2phy(module, None)
                module.moe_config.num_experts = module.elastic_original_num_experts
                continue
            if module.expert_map is None:
                _set_module_active_expert_mask(module, None)
                _set_module_elastic_runtime_log2phy(module, None)
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
                _set_module_active_expert_mask(module, active_expert_mask.to(
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
                    _set_module_elastic_runtime_log2phy(module,
                        new_log2phy_cpu.to(device=module.log2phy.device,
                                           dtype=module.log2phy.dtype))
                else:
                    _set_module_elastic_runtime_log2phy(module, None)
                    module.set_runtime_num_experts(sum(local_expert_counts))

                module.refresh_elastic_groups()

        if is_active_rank and aggregate_mode5_hybrid_layers > 0:
            mode5_strategy = _mode5_cpu_shadow_runtime_strategy()
            legacy_runtime = int(mode5_strategy == "legacy_cpu_shadow")
            logger.info(
                "Mode5 hybrid refresh aggregate: rank=%s stage=%s layers=%s mode5_strategy=%s legacy_runtime=%s build_sources_ms=%.2f activate_ms=%.2f build_cpu_shadow_ms=%.2f materialize_resident_ms=%.2f log2phy_ms=%.2f set_runtime_ms=%.2f refresh_groups_ms=%.2f total_ms=%.2f",
                self.rank,
                len(active_ranks),
                aggregate_mode5_hybrid_layers,
                mode5_strategy,
                legacy_runtime,
                aggregate_mode5_hybrid_build_sources_ms,
                aggregate_mode5_hybrid_activate_ms,
                aggregate_mode5_hybrid_build_cpu_shadow_ms,
                aggregate_mode5_hybrid_materialize_resident_ms,
                aggregate_mode5_hybrid_log2phy_ms,
                aggregate_mode5_hybrid_set_runtime_ms,
                aggregate_mode5_hybrid_refresh_groups_ms,
                aggregate_mode5_hybrid_total_ms,
            )

    def _get_previous_active_ranks_for_shrink(self, world_size: int) -> list[int]:
        previous_active_ranks = getattr(self, "_elastic_current_active_ranks",
                                        None)
        if not previous_active_ranks:
            return self._original_local_elastic_group_ranks(world_size)
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
        payload_barrier_ms = 0.0
        payload_metadata_gather_ms = 0.0
        payload_remote_state_gather_ms = 0.0
        payload_store_ms = 0.0
        payload_module_count = 0
        source_group_world_size = -1
        try:
            source_group_world_size = torch.distributed.get_world_size(
                group=source_cpu_group)
        except Exception:
            source_group_world_size = -1
        if current_rank == source_ranks[0]:
            logger.info(
                "Elastic shrink payload source group: rank=%s active_ranks=%s "
                "source_ranks=%s previous_active_ranks=%s "
                "source_group_world_size=%s",
                current_rank, active_ranks, source_ranks,
                previous_active_ranks, source_group_world_size)
        mode1_payload_barrier = (
            envs_ascend.VLLM_ASCEND_ELASTIC_EXECUTION_MODE == 1
            and _env_flag("VLLM_ASCEND_MODE1_PAYLOAD_SOURCE_BARRIER", "1"))
        if mode1_payload_barrier:
            logger.info(
                "Elastic shrink payload source p2p barrier enter: rank=%s "
                "active_ranks=%s source_ranks=%s previous_active_ranks=%s",
                current_rank, active_ranks, source_ranks,
                previous_active_ranks)
            barrier_start_t = time.perf_counter()
            try:
                self._mode1_source_cpu_p2p_barrier(
                    source_ranks, world_group,
                    38128 + len(active_ranks) * 10)
            except Exception as exc:
                raise RuntimeError(
                    "Elastic mode1 shrink payload source p2p barrier failed; "
                    "source ranks did not enter payload preparation together. "
                    f"rank={current_rank} active_ranks={active_ranks} "
                    f"source_ranks={source_ranks} "
                    f"previous_active_ranks={previous_active_ranks}") from exc
            payload_barrier_ms = (time.perf_counter() -
                                  barrier_start_t) * 1000.0
            logger.info(
                "Elastic shrink payload source p2p barrier done: rank=%s "
                "active_ranks=%s source_ranks=%s total_ms=%.2f",
                current_rank, active_ranks, source_ranks, payload_barrier_ms)
        elif envs_ascend.VLLM_ASCEND_ELASTIC_EXECUTION_MODE in (2, 3, 4, 5):
            logger.info(
                "Elastic shrink payload source barrier enter: rank=%s active_ranks=%s source_ranks=%s previous_active_ranks=%s",
                current_rank, active_ranks, source_ranks,
                previous_active_ranks)
            barrier_start_t = time.perf_counter()
            try:
                torch.distributed.monitored_barrier(
                    group=source_cpu_group,
                    timeout=timedelta(seconds=120),
                )
            except Exception as exc:
                raise RuntimeError(
                    "Elastic shrink payload source barrier failed; "
                    "source ranks did not enter payload preparation together. "
                    f"rank={current_rank} active_ranks={active_ranks} "
                    f"source_ranks={source_ranks} "
                    f"previous_active_ranks={previous_active_ranks}") from exc
            payload_barrier_ms = (time.perf_counter() -
                                  barrier_start_t) * 1000.0
            logger.info(
                "Elastic shrink payload source barrier done: rank=%s active_ranks=%s source_ranks=%s",
                current_rank, active_ranks, source_ranks)

        payload_gather_trace = (
            envs_ascend.VLLM_ASCEND_ELASTIC_EXECUTION_MODE == 1
            and _env_flag("VLLM_ASCEND_MODE1_PAYLOAD_GATHER_TRACE", "0"))
        try:
            payload_gather_trace_modules = int(
                os.getenv("VLLM_ASCEND_MODE1_PAYLOAD_GATHER_TRACE_MODULES",
                          "1"))
        except ValueError:
            payload_gather_trace_modules = 1
        payload_planning_diag = (
            envs_ascend.VLLM_ASCEND_ELASTIC_EXECUTION_MODE == 1
            and _env_flag("VLLM_ASCEND_MODE1_PAYLOAD_PLANNING_DIAG", "1"))

        def _mode1_map_summary(tensor: torch.Tensor) -> dict[str, Any]:
            values = [int(value) for value in tensor.reshape(-1).tolist()]
            slots = [value for value in values if value >= 0]
            sorted_slots = sorted(slots)
            expected_prefix = list(range(len(sorted_slots)))
            missing_prefix = [
                value for value in expected_prefix
                if value not in set(sorted_slots)
            ][:8]
            duplicate_slots = sorted({
                value
                for value in slots
                if slots.count(value) > 1
            })[:8]
            expert_head = [
                idx for idx, value in enumerate(values)
                if int(value) >= 0
            ][:8]
            return {
                "n": len(slots),
                "min": min(slots) if slots else -1,
                "max": max(slots) if slots else -1,
                "unique": len(set(slots)),
                "prefix": sorted_slots == expected_prefix,
                "missing_prefix_head": missing_prefix,
                "duplicate_slots_head": duplicate_slots,
                "expert_head": expert_head,
            }

        def _mode1_rank_map_summaries(
                maps_by_rank: dict[int, torch.Tensor]) -> dict[int, dict[str, Any]]:
            return {
                int(rank): _mode1_map_summary(tensor)
                for rank, tensor in maps_by_rank.items()
            }

        def _mode1_rows_summary(rows_by_rank: dict[int, int]) -> dict[str, Any]:
            rows = [int(value) for value in rows_by_rank.values()]
            return {
                "min": min(rows) if rows else -1,
                "max": max(rows) if rows else -1,
                "unique": sorted(set(rows)),
                "by_rank": {int(rank): int(value)
                            for rank, value in rows_by_rank.items()},
            }

        def _payload_all_gather(output_tensors: list[torch.Tensor],
                                input_tensor: torch.Tensor, op_name: str,
                                layer_idx: Any,
                                module_count: int) -> None:
            trace_this = (payload_gather_trace
                          and module_count <= payload_gather_trace_modules)
            gather_start_t = time.perf_counter()
            if trace_this:
                logger.info(
                    "Elastic shrink payload metadata gather enter: rank=%s "
                    "op=%s layer=%s module_idx=%s active_ranks=%s "
                    "source_ranks=%s source_group_world_size=%s shape=%s "
                    "dtype=%s",
                    current_rank, op_name, layer_idx, module_count,
                    active_ranks, source_ranks, source_group_world_size,
                    tuple(input_tensor.shape), input_tensor.dtype)
            try:
                torch.distributed.all_gather(output_tensors,
                                             input_tensor,
                                             group=source_cpu_group)
            except Exception as exc:
                raise RuntimeError(
                    "Elastic shrink payload metadata all_gather failed. "
                    f"rank={current_rank} op={op_name} layer={layer_idx} "
                    f"module_idx={module_count} active_ranks={active_ranks} "
                    f"source_ranks={source_ranks} "
                    f"previous_active_ranks={previous_active_ranks} "
                    f"source_group_world_size={source_group_world_size} "
                    f"shape={tuple(input_tensor.shape)} "
                    f"dtype={input_tensor.dtype}") from exc
            if trace_this:
                logger.info(
                    "Elastic shrink payload metadata gather done: rank=%s "
                    "op=%s layer=%s module_idx=%s total_ms=%.2f",
                    current_rank, op_name, layer_idx, module_count,
                    (time.perf_counter() - gather_start_t) * 1000.0)

        for module in model.modules():
            if not _is_ascend_fused_moe_module(module):
                continue
            if getattr(module, "elastic_moe_mode", "lossy") != "lossless":
                continue
            if module.expert_map is None:
                continue
            if not hasattr(module, "layer_idx"):
                continue
            payload_module_count += 1
            logical_num_experts = int(module.elastic_original_num_experts)
            metadata_gather_start_t = time.perf_counter()

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
            _payload_all_gather(gathered_loaded_maps_source,
                                local_loaded_expert_map_cpu,
                                "loaded_expert_map", module.layer_idx,
                                payload_module_count)
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
            _payload_all_gather(gathered_resident_maps_source,
                                resident_expert_map, "resident_expert_map",
                                module.layer_idx, payload_module_count)
            gathered_resident_maps_by_rank = {
                source_ranks[idx]: gathered_resident_maps_source[idx]
                for idx in range(len(source_ranks))
            }
            local_w13_rows = torch.tensor([
                int(getattr(module, "w13_weight", torch.empty(0)).shape[0])
            ],
                                          device="cpu",
                                          dtype=torch.int32)
            gathered_w13_rows = [
                torch.empty_like(local_w13_rows) for _ in range(len(source_ranks))
            ]
            _payload_all_gather(gathered_w13_rows, local_w13_rows,
                                "w13_rows", module.layer_idx,
                                payload_module_count)
            w13_rows_by_rank = {
                source_ranks[idx]: int(gathered_w13_rows[idx].item())
                for idx in range(len(source_ranks))
            }
            local_w2_rows = torch.tensor([
                int(getattr(module, "w2_weight", torch.empty(0)).shape[0])
            ],
                                         device="cpu",
                                         dtype=torch.int32)
            gathered_w2_rows = [
                torch.empty_like(local_w2_rows) for _ in range(len(source_ranks))
            ]
            _payload_all_gather(gathered_w2_rows, local_w2_rows, "w2_rows",
                                module.layer_idx, payload_module_count)
            w2_rows_by_rank = {
                source_ranks[idx]: int(gathered_w2_rows[idx].item())
                for idx in range(len(source_ranks))
            }
            payload_metadata_gather_ms += (
                time.perf_counter() - metadata_gather_start_t) * 1000.0
            trace_module = (payload_gather_trace
                            and payload_module_count
                            <= payload_gather_trace_modules)
            if trace_module:
                logger.info(
                    "Elastic shrink payload module planning enter: rank=%s "
                    "layer=%s module_idx=%s active_ranks=%s source_ranks=%s "
                    "logical_num_experts=%s loaded_map_attr=%s "
                    "module_expert_map_device=%s module_loaded_map_device=%s "
                    "w13_shape=%s w2_shape=%s",
                    current_rank, module.layer_idx, payload_module_count,
                    active_ranks, source_ranks, logical_num_experts,
                    lossless_loaded_expert_map is not None,
                    getattr(module.expert_map, "device", None),
                    getattr(lossless_loaded_expert_map, "device", None),
                    tuple(getattr(module, "w13_weight", torch.empty(0)).shape),
                    tuple(getattr(module, "w2_weight", torch.empty(0)).shape))
                if payload_planning_diag:
                    logger.info(
                        "Elastic shrink payload module map summary: rank=%s "
                        "layer=%s module_idx=%s loaded=%s resident=%s "
                        "w13_rows=%s w2_rows=%s",
                        current_rank,
                        module.layer_idx,
                        payload_module_count,
                        _mode1_rank_map_summaries(gathered_loaded_maps_by_rank),
                        _mode1_rank_map_summaries(gathered_resident_maps_by_rank),
                        _mode1_rows_summary(w13_rows_by_rank),
                        _mode1_rows_summary(w2_rows_by_rank),
                    )
            gathered_mode4_remote_sources_by_rank: dict[
                int, dict[int, tuple[int, int]]] = {}
            gathered_mode4_owned_by_rank: dict[int, list[int]] = {}
            gathered_mode5_cpu_shadow_by_rank: dict[int, set[int]] = {}
            if envs_ascend.VLLM_ASCEND_ELASTIC_EXECUTION_MODE in (4, 5):
                remote_state_gather_start_t = time.perf_counter()
                remote_source_attr = "lossless_mode4_remote_source_by_expert"
                if envs_ascend.VLLM_ASCEND_ELASTIC_EXECUTION_MODE == 5:
                    # For mode5, preserve the full remote-source chain across
                    # successive shrinks. The current-stage remote set is only
                    # a transport choice; later shrinks still need remote slot
                    # metadata for experts that temporarily flowed through CPU.
                    remote_source_attr = (
                        "lossless_mode5_remote_source_chain_by_expert")
                local_mode4_remote_sources = getattr(
                    module,
                    remote_source_attr,
                    getattr(module, "lossless_mode4_remote_source_by_expert", {}),
                )
                local_mode4_remote_source_rank = torch.full(
                    (logical_num_experts,), -1, device="cpu", dtype=torch.int32)
                local_mode4_remote_source_slot = torch.full(
                    (logical_num_experts,), -1, device="cpu", dtype=torch.int32)
                for expert_id, (source_rank, source_slot) in (
                        local_mode4_remote_sources.items()):
                    expert_id = int(expert_id)
                    if 0 <= expert_id < logical_num_experts:
                        local_mode4_remote_source_rank[expert_id] = int(
                            source_rank)
                        local_mode4_remote_source_slot[expert_id] = int(
                            source_slot)
                gathered_mode4_remote_source_rank = [
                    torch.empty_like(local_mode4_remote_source_rank)
                    for _ in range(len(source_ranks))
                ]
                gathered_mode4_remote_source_slot = [
                    torch.empty_like(local_mode4_remote_source_slot)
                    for _ in range(len(source_ranks))
                ]
                torch.distributed.all_gather(
                    gathered_mode4_remote_source_rank,
                    local_mode4_remote_source_rank,
                    group=source_cpu_group)
                torch.distributed.all_gather(
                    gathered_mode4_remote_source_slot,
                    local_mode4_remote_source_slot,
                    group=source_cpu_group)
                gathered_mode4_remote_sources_by_rank = {}
                for idx, world_rank in enumerate(source_ranks):
                    remote_rank_tensor = gathered_mode4_remote_source_rank[idx]
                    remote_slot_tensor = gathered_mode4_remote_source_slot[idx]
                    remote_source_map: dict[int, tuple[int, int]] = {}
                    valid_remote = torch.nonzero(remote_rank_tensor >= 0,
                                                 as_tuple=False).flatten()
                    for expert_id in valid_remote.tolist():
                        remote_source_map[int(expert_id)] = (
                            int(remote_rank_tensor[expert_id].item()),
                            int(remote_slot_tensor[expert_id].item()),
                        )
                    gathered_mode4_remote_sources_by_rank[int(
                        world_rank)] = remote_source_map

                local_mode4_owned_mask = torch.zeros((logical_num_experts, ),
                                                     device="cpu",
                                                     dtype=torch.int8)
                for expert_id in getattr(module, "lossless_hybrid_owned_expert_ids",
                                         []):
                    expert_id = int(expert_id)
                    if 0 <= expert_id < logical_num_experts:
                        local_mode4_owned_mask[expert_id] = 1
                gathered_mode4_owned_mask = [
                    torch.empty_like(local_mode4_owned_mask)
                    for _ in range(len(source_ranks))
                ]
                torch.distributed.all_gather(gathered_mode4_owned_mask,
                                             local_mode4_owned_mask,
                                             group=source_cpu_group)
                gathered_mode4_owned_by_rank = {
                    int(source_ranks[idx]): [
                        int(expert_id) for expert_id in torch.nonzero(
                            gathered_mode4_owned_mask[idx] > 0,
                            as_tuple=False).flatten().tolist()
                    ]
                    for idx in range(len(source_ranks))
                }

                local_mode5_cpu_shadow_mask = torch.zeros(
                    (logical_num_experts,), device="cpu", dtype=torch.int8)
                local_mode5_cpu_shadow_ids = set(
                    int(expert_id) for expert_id in getattr(
                        module, "lossless_cpu_shadow_local_slots", {}).keys())
                local_mode5_cpu_shadow_ids.update(
                    int(expert_id) for expert_id in getattr(
                        module, "lossless_mode5_cpu_shadow_row_by_expert",
                        {}).keys())
                for expert_id in local_mode5_cpu_shadow_ids:
                    if 0 <= expert_id < logical_num_experts:
                        local_mode5_cpu_shadow_mask[expert_id] = 1
                gathered_mode5_cpu_shadow_mask = [
                    torch.empty_like(local_mode5_cpu_shadow_mask)
                    for _ in range(len(source_ranks))
                ]
                torch.distributed.all_gather(gathered_mode5_cpu_shadow_mask,
                                             local_mode5_cpu_shadow_mask,
                                             group=source_cpu_group)
                gathered_mode5_cpu_shadow_by_rank = {
                    int(source_ranks[idx]): set(
                        int(expert_id) for expert_id in torch.nonzero(
                            gathered_mode5_cpu_shadow_mask[idx] > 0,
                            as_tuple=False).flatten().tolist())
                    for idx in range(len(source_ranks))
                }
                payload_remote_state_gather_ms += (
                    time.perf_counter() -
                    remote_state_gather_start_t) * 1000.0

            def _resolve_mode4_npu_source(source_rank: int,
                                          expert_id: int,
                                          fallback_slot: int
                                          ) -> tuple[int, int]:
                """Return the rank/slot that actually owns this expert on NPU."""
                if envs_ascend.VLLM_ASCEND_ELASTIC_EXECUTION_MODE not in (4, 5):
                    return int(source_rank), int(fallback_slot)
                remote_sources = gathered_mode4_remote_sources_by_rank.get(
                    int(source_rank), {})
                remote_source = remote_sources.get(int(expert_id))
                if remote_source is not None:
                    return int(remote_source[0]), int(remote_source[1])
                return int(source_rank), int(fallback_slot)

            def _mode5_npu_remote_eligible_experts(
                    remote_source_rank_by_expert: dict[int, int],
                    source_slot_by_expert: dict[int, int]) -> set[int]:
                if envs_ascend.VLLM_ASCEND_ELASTIC_EXECUTION_MODE != 5:
                    return set(source_slot_by_expert)
                # remote_import_source_slot is computed by the shrink planner
                # after resolving the actual remote source chain. If a missing
                # expert has a non-negative remote slot here, the mode4/mode5
                # remote cache service can serve it. Do not re-check against
                # module.w13_weight.shape[0] / module.w2_weight.shape[0]:
                # exiting/cache ranks may serve from runtime/cache views whose
                # rows differ from the static loaded-weight tensor shapes.
                return {
                    int(expert_id)
                    for expert_id, source_slot in source_slot_by_expert.items()
                    if int(source_slot) >= 0
                }

            def _mode5_required_remote_experts(
                    source_rank_by_expert: dict[int, int],
                    source_slot_by_expert: dict[int, int]) -> set[int]:
                if envs_ascend.VLLM_ASCEND_ELASTIC_EXECUTION_MODE != 5:
                    return set()
                # Keep the strict-ratio path simple: do not pre-classify
                # historical remote-backed experts as must-remote. Any expert
                # that genuinely cannot be satisfied by the CPU path will be
                # moved to remote later by the CPU-owner rewrite/fallback logic.
                return set()

            def _mode5_cpu_import_experts(
                    source_rank_by_expert: dict[int, int],
                    target_rank_by_expert: dict[int, int],
                    remote_experts: set[int]) -> set[int]:
                if envs_ascend.VLLM_ASCEND_ELASTIC_EXECUTION_MODE != 5:
                    return set(source_rank_by_expert)
                needed: set[int] = set()
                for expert_id, source_rank in source_rank_by_expert.items():
                    expert_id = int(expert_id)
                    if expert_id in remote_experts or source_rank is None:
                        continue
                    target_rank = target_rank_by_expert.get(expert_id)
                    if target_rank is None:
                        continue
                    existing_shadow = gathered_mode5_cpu_shadow_by_rank.get(
                        int(target_rank), set())
                    if expert_id in existing_shadow:
                        continue
                    needed.add(expert_id)
                return needed

            def _mode5_cpu_shadow_owner_ranks(expert_id: int) -> list[int]:
                expert_id = int(expert_id)
                return sorted(
                    int(rank) for rank, shadow_experts in
                    gathered_mode5_cpu_shadow_by_rank.items()
                    if expert_id in shadow_experts)

            def _mode5_rank_has_explicit_cpu_shadow(rank: int) -> bool:
                shadow_experts = gathered_mode5_cpu_shadow_by_rank.get(
                    int(rank), None)
                return shadow_experts is not None and len(shadow_experts) > 0

            def _finalize_mode5_import_partition(
                    source_rank_by_expert: dict[int, int],
                    target_rank_by_expert: dict[int, int],
                    source_slot_by_expert: dict[int, int],
                    remote_source_rank_by_expert: dict[int, int],
                    import_candidate_experts: set[int]
            ) -> tuple[set[int], set[int], dict[int, int]]:
                if envs_ascend.VLLM_ASCEND_ELASTIC_EXECUTION_MODE != 5:
                    all_experts = set(source_rank_by_expert)
                    return all_experts, all_experts, dict(source_rank_by_expert)
                import_source_rank_by_expert = {
                    int(expert_id): int(source_rank)
                    for expert_id, source_rank in source_rank_by_expert.items()
                    if int(expert_id) in import_candidate_experts
                    and source_rank is not None
                }
                import_target_rank_by_expert = {
                    int(expert_id): int(target_rank)
                    for expert_id, target_rank in target_rank_by_expert.items()
                    if int(expert_id) in import_candidate_experts
                    and target_rank is not None
                }
                import_source_slot_by_expert = {
                    int(expert_id): int(source_slot)
                    for expert_id, source_slot in source_slot_by_expert.items()
                    if int(expert_id) in import_candidate_experts
                }
                import_remote_source_rank_by_expert = {
                    int(expert_id): int(source_rank)
                    for expert_id, source_rank in
                    remote_source_rank_by_expert.items()
                    if int(expert_id) in import_candidate_experts
                }
                mode5_remote_eligible = _mode5_npu_remote_eligible_experts(
                    import_remote_source_rank_by_expert,
                    import_source_slot_by_expert)
                mode5_required_remote = _mode5_required_remote_experts(
                    import_source_rank_by_expert, import_source_slot_by_expert)
                if mode5_required_remote:
                    raise RuntimeError(
                        "Mode5 strict ratio path does not allow pre-forced "
                        f"remote experts: layer={module.layer_idx} rank={current_rank} "
                        f"required_remote={sorted(mode5_required_remote)[:16]} "
                        f"count={len(mode5_required_remote)}")
                mode5_remote_fraction, mode5_remote_fraction_debug = (
                    _mode5_remote_expert_fraction_with_debug())
                mode5_remote_experts, mode5_cpu_import_experts = (
                    _mode5_strict_partition_by_target(
                        import_source_rank_by_expert,
                        import_target_rank_by_expert,
                        mode5_remote_eligible,
                        import_remote_source_rank_by_expert))
                rewritten_source_rank_by_expert = dict(import_source_rank_by_expert)
                source_rewrites: list[tuple[int, int, int]] = []
                mode5_forced_remote: set[int] = set()
                final_source_rank_by_expert = dict(source_rank_by_expert)
                final_source_rank_by_expert.update(rewritten_source_rank_by_expert)
                remote_source_edges = {
                    (
                        int(import_remote_source_rank_by_expert[expert_id]),
                        int(import_target_rank_by_expert[expert_id]),
                    )
                    for expert_id in mode5_remote_experts
                    if expert_id in import_remote_source_rank_by_expert
                    and expert_id in import_target_rank_by_expert
                }
                remote_sources_used = {
                    int(import_remote_source_rank_by_expert[expert_id])
                    for expert_id in mode5_remote_experts
                    if expert_id in import_remote_source_rank_by_expert
                }
                if ((mode5_forced_remote or source_rewrites)
                        and module.layer_idx == 0
                        and current_rank == active_ranks[0]):
                    logger.info(
                        "Mode5 CPU/remote partition adjusted for CPU shadow ownership: "
                        "rank=%s active_ranks=%s forced_remote=%s forced_head=%s rewrites=%s rewrite_head=%s",
                        current_rank,
                        active_ranks,
                        len(mode5_forced_remote),
                        sorted(mode5_forced_remote)[:16],
                        len(source_rewrites),
                        source_rewrites[:8],
                    )
                if (module.layer_idx == 0 and current_rank == active_ranks[0]):
                    logger.info(
                        "Mode5 import partition summary: rank=%s active_ranks=%s import_total=%s resident_local=%s remote_fraction=%.4f remote_selected=%s cpu_selected=%s required_remote=%s forced_remote=%s remote_sources_used=%s remote_source_edges=%s fanout_balance=%s fraction_policy=%s",
                        current_rank,
                        active_ranks,
                        len(import_candidate_experts),
                        logical_num_experts - len(import_candidate_experts),
                        mode5_remote_fraction,
                        len(mode5_remote_experts),
                        len(mode5_cpu_import_experts),
                        len(mode5_required_remote),
                        len(mode5_forced_remote),
                        len(remote_sources_used),
                        len(remote_source_edges),
                        _mode5_balance_remote_source_fanout(),
                        mode5_remote_fraction_debug,
                    )
                return (mode5_remote_experts, mode5_cpu_import_experts,
                        final_source_rank_by_expert)

            assignments: list[list[tuple[int, int]]] = [
                [] for _ in range(len(active_ranks))
            ]
            assigned_counts = [0 for _ in range(len(active_ranks))]
            target_per_rank = (
                logical_num_experts + len(active_ranks) - 1
            ) // len(active_ranks)
            cpu_import_source_rank: dict[int, int] = {}
            cpu_import_target_rank: dict[int, int] = {}
            mode4_remote_source_rank: dict[int, int] = {}
            remote_import_source_slot: dict[int, int] = {}
            gathered_loaded_count_by_rank = {
                int(rank): int(torch.count_nonzero(tensor >= 0).item())
                for rank, tensor in gathered_loaded_maps_by_rank.items()
            }
            gathered_resident_count_by_rank = {
                int(rank): int(torch.count_nonzero(tensor >= 0).item())
                for rank, tensor in gathered_resident_maps_by_rank.items()
            }
            gathered_loaded_total = int(sum(gathered_loaded_count_by_rank.values()))
            gathered_resident_total = int(
                sum(gathered_resident_count_by_rank.values()))
            raw_global_redundant = int(
                getattr(module, "global_redundant_expert_num", 0))
            inferred_loaded_redundant = max(
                gathered_loaded_total - logical_num_experts, 0)
            effective_global_redundant = raw_global_redundant
            if effective_global_redundant <= 0 and inferred_loaded_redundant > 0:
                effective_global_redundant = inferred_loaded_redundant
            use_paired_zero_redundancy = (
                effective_global_redundant <= 0
                and len(previous_active_ranks) == 2 * len(active_ranks)
                and set(active_ranks).issubset(set(previous_active_ranks))
                and logical_num_experts % len(active_ranks) == 0
            )
            hybrid_resident_capacity = 0
            if (envs_ascend.VLLM_ASCEND_ELASTIC_EXECUTION_MODE in (2, 3, 4, 5)
                    and hasattr(module, "_get_hybrid_resident_capacity")):
                hybrid_resident_capacity = int(
                    module._get_hybrid_resident_capacity())
            use_paired_redundant_transfer = (
                effective_global_redundant > 0
                and len(previous_active_ranks) == 2 * len(active_ranks)
                and set(active_ranks).issubset(set(previous_active_ranks))
                and logical_num_experts % len(active_ranks) == 0
                and (
                    envs_ascend.VLLM_ASCEND_ELASTIC_EXECUTION_MODE == 1
                    or (
                        envs_ascend.VLLM_ASCEND_ELASTIC_EXECUTION_MODE == 2
                        and hybrid_resident_capacity > 0
                    )
                    or (
                        envs_ascend.VLLM_ASCEND_ELASTIC_EXECUTION_MODE in (4, 5)
                        and hybrid_resident_capacity > 0
                        and target_per_rank <= hybrid_resident_capacity
                    )
                ))
            if (envs_ascend.VLLM_ASCEND_ELASTIC_EXECUTION_MODE in (4, 5)
                    and hybrid_resident_capacity > 0
                    and len(previous_active_ranks) == 2 * len(active_ranks)
                    and set(active_ranks).issubset(set(previous_active_ranks))
                    and logical_num_experts % len(active_ranks) == 0):
                # Mode=4 must keep a deterministic active/cache rank chain:
                # 16->8 pairs each surviving rank with one cache rank, and
                # follow-up shrinks fold the just-exited rank's ownership into
                # its paired survivor. The generic greedy planner can produce a
                # locally balanced but incomplete logical ownership map after
                # 4->2; force the paired transfer path instead.
                use_paired_redundant_transfer = True
            prefer_preloaded_local_slots = (
                envs_ascend.VLLM_ASCEND_ELASTIC_EXECUTION_MODE == 1
                and effective_global_redundant > 0
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
            if trace_module and payload_planning_diag:
                logger.info(
                    "Elastic shrink payload branch decision: rank=%s layer=%s "
                    "module_idx=%s exec_mode=%s previous_active_ranks=%s "
                    "active_ranks=%s inactive_ranks=%s target_per_rank=%s "
                    "global_redundant=%s effective_redundant=%s "
                    "inferred_loaded_redundant=%s loaded_total=%s "
                    "resident_total=%s hybrid_capacity=%s "
                    "use_zero=%s use_redundant=%s prefer_preloaded=%s "
                    "logical_divisible=%s active_subset=%s paired_count=%s",
                    current_rank,
                    module.layer_idx,
                    payload_module_count,
                    int(envs_ascend.VLLM_ASCEND_ELASTIC_EXECUTION_MODE),
                    previous_active_ranks,
                    active_ranks,
                    inactive_ranks,
                    target_per_rank,
                    raw_global_redundant,
                    effective_global_redundant,
                    inferred_loaded_redundant,
                    gathered_loaded_total,
                    gathered_resident_total,
                    hybrid_resident_capacity,
                    use_paired_zero_redundancy,
                    use_paired_redundant_transfer,
                    prefer_preloaded_local_slots,
                    logical_num_experts % len(active_ranks) == 0,
                    set(active_ranks).issubset(set(previous_active_ranks)),
                    len(previous_active_ranks) == 2 * len(active_ranks),
                )
            if (use_paired_zero_redundancy and not logged_pairing_summary
                    and current_rank == active_ranks[0]
                    and module.layer_idx == 0):
                pass  # debug log removed
                logged_pairing_summary = True

            if (envs_ascend.VLLM_ASCEND_ELASTIC_EXECUTION_MODE in (4, 5)
                    and hybrid_resident_capacity > 0
                    and len(previous_active_ranks) == 2 * len(active_ranks)
                    and len(inactive_ranks) == len(active_ranks)
                    and set(active_ranks).issubset(set(previous_active_ranks))
                    and logical_num_experts % len(active_ranks) == 0):
                ordered_assignments: list[list[tuple[int, int]]] = []
                runtime_log2phy_cpu = torch.full((logical_num_experts, ),
                                                 -1,
                                                 dtype=torch.int32)
                dense_offset = 0
                assigned_all: list[int] = []

                def _owned_experts_for_rank(rank: int) -> list[int]:
                    owned = list(gathered_mode4_owned_by_rank.get(int(rank), []))
                    if owned:
                        return owned
                    pairs = sorted(
                        ((expert_id, int(local_slot))
                         for expert_id, local_slot in enumerate(
                             gathered_loaded_maps_by_rank[int(rank)].tolist())
                         if int(local_slot) >= 0),
                        key=lambda item: item[1],
                    )
                    return [int(expert_id) for expert_id, _local_slot in pairs]

                pairing_by_active = {
                    int(active_rank): int(inactive_ranks[idx])
                    for idx, active_rank in enumerate(active_ranks)
                }

                for rank_idx, active_rank in enumerate(active_ranks):
                    active_rank = int(active_rank)
                    inactive_rank = pairing_by_active[active_rank]
                    active_owned = _owned_experts_for_rank(active_rank)
                    inactive_owned = _owned_experts_for_rank(inactive_rank)
                    active_primary_slots = {
                        expert_id: int(local_slot)
                        for expert_id, local_slot in enumerate(
                            gathered_resident_maps_by_rank[active_rank].tolist())
                        if 0 <= int(local_slot) < hybrid_resident_capacity
                    }
                    active_primary = [
                        expert_id for expert_id, _local_slot in sorted(
                            active_primary_slots.items(),
                            key=lambda item: item[1])
                        if expert_id in set(active_owned)
                    ]
                    active_primary_set = set(active_primary)
                    active_remote_owned = [
                        int(expert_id) for expert_id in active_owned
                        if int(expert_id) not in active_primary_set
                    ]
                    target_owned = (
                        active_primary + active_remote_owned +
                        [int(expert_id) for expert_id in inactive_owned
                         if int(expert_id) not in set(active_owned)])
                    if len(target_owned) != target_per_rank:
                        raise RuntimeError(
                            "Mode4 paired cascade shrink produced unexpected "
                            f"owned count at layer={module.layer_idx}: "
                            f"active_rank={active_rank} inactive_rank={inactive_rank} "
                            f"owned={len(target_owned)} target={target_per_rank} "
                            f"active_owned={len(active_owned)} "
                            f"inactive_owned={len(inactive_owned)}")

                    rank_assignments: list[tuple[int, int]] = []
                    for expert_id in target_owned:
                        expert_id = int(expert_id)
                        if expert_id in active_primary_slots:
                            rank_assignments.append(
                                (expert_id, int(active_primary_slots[expert_id])))
                            continue
                        base_rank = (
                            active_rank if expert_id in set(active_owned)
                            else inactive_rank)
                        fallback_slot = int(
                            gathered_loaded_maps_by_rank[base_rank][expert_id].
                            item())
                        if fallback_slot < 0:
                            fallback_slot = int(
                                gathered_resident_maps_by_rank[base_rank][
                                    expert_id].item())
                        remote_source_rank, source_slot = _resolve_mode4_npu_source(
                            base_rank, expert_id, fallback_slot)
                        if remote_source_rank == active_rank and (
                                0 <= source_slot < hybrid_resident_capacity):
                            rank_assignments.append((expert_id, int(source_slot)))
                        else:
                            rank_assignments.append((expert_id, -1))
                            cpu_import_source_rank[expert_id] = int(base_rank)
                            cpu_import_target_rank[expert_id] = active_rank
                            mode4_remote_source_rank[expert_id] = int(
                                remote_source_rank)
                            remote_import_source_slot[expert_id] = int(source_slot)

                    assignments[rank_idx] = rank_assignments
                    ordered_assignments.append(rank_assignments)
                    assigned_counts[rank_idx] = len(rank_assignments)
                    assigned_all.extend(expert_id for expert_id, _ in rank_assignments)
                    for local_slot, (expert_id, _source_local_id) in enumerate(
                            rank_assignments):
                        runtime_log2phy_cpu[int(expert_id)] = dense_offset + local_slot
                    dense_offset += len(rank_assignments)

                if (len(assigned_all) != logical_num_experts
                        or len(set(assigned_all)) != logical_num_experts
                        or bool(torch.any(runtime_log2phy_cpu < 0).item())):
                    missing = [
                        expert_id for expert_id in range(logical_num_experts)
                        if int(runtime_log2phy_cpu[expert_id].item()) < 0
                    ][:16]
                    duplicates = sorted({
                        expert_id
                        for expert_id in assigned_all
                        if assigned_all.count(expert_id) > 1
                    })[:16]
                    raise RuntimeError(
                        "Mode4 paired cascade shrink must cover every logical "
                        f"expert exactly once at layer={module.layer_idx}: "
                        f"assigned={len(assigned_all)} "
                        f"unique={len(set(assigned_all))} "
                        f"missing_head={missing} duplicate_head={duplicates}")

                mode5_import_candidate_experts = {
                    int(expert_id)
                    for rank_assignments in ordered_assignments
                    for expert_id, local_id in rank_assignments
                    if int(local_id) < 0
                }

                (mode5_remote_experts,
                 mode5_cpu_import_experts,
                 cpu_import_source_rank) = _finalize_mode5_import_partition(
                     cpu_import_source_rank,
                     cpu_import_target_rank,
                     remote_import_source_slot,
                     mode4_remote_source_rank,
                     mode5_import_candidate_experts)
                module_store_start_t = time.perf_counter()
                self._lossless_shrink_payload[module.layer_idx] = {
                    "assignments": assignments,
                    "ordered_assignments": ordered_assignments,
                    "runtime_log2phy_cpu": runtime_log2phy_cpu,
                    "cpu_import_source_rank": cpu_import_source_rank,
                    "cpu_import_target_rank": cpu_import_target_rank,
                    "mode4_remote_source_rank": mode4_remote_source_rank,
                    "remote_import_source_slot": remote_import_source_slot,
                    "target_per_rank": target_per_rank,
                    "mode5_remote_experts": sorted(mode5_remote_experts),
                    "mode5_cpu_import_experts": sorted(mode5_cpu_import_experts),
                }
                payload_store_ms += (
                    time.perf_counter() - module_store_start_t) * 1000.0
                if module.layer_idx == 0 and current_rank == active_ranks[0]:
                    logger.info(
                        "Mode4 paired cascade shrink plan: rank=%s active_ranks=%s inactive_ranks=%s target_per_rank=%s remote_imports=%s assignment_counts=%s",
                        current_rank,
                        active_ranks,
                        inactive_ranks,
                        target_per_rank,
                        len(cpu_import_source_rank),
                        assigned_counts,
                    )
                continue

            if use_paired_redundant_transfer:
                if trace_module:
                    logger.info(
                        "Elastic shrink payload paired planning branch: rank=%s "
                        "layer=%s module_idx=%s active_ranks=%s inactive_ranks=%s "
                        "target_per_rank=%s redundant=%s",
                        current_rank, module.layer_idx, payload_module_count,
                        active_ranks, inactive_ranks, target_per_rank,
                        effective_global_redundant)
                ordered_assignments: list[list[tuple[int, int]]] = []
                # Keep this planning path in plain Python.  The paired
                # redundant branch runs before any payload tensor transfer;
                # avoiding incidental torch ops here makes failures easier to
                # attribute to the real communication or import stage.
                runtime_log2phy_values = [-1 for _ in range(logical_num_experts)]
                dense_offset = 0
                active_runtime_slots_by_rank: dict[int, dict[int, int]] = {}
                active_free_slots_by_rank: dict[int, list[int]] = {}
                inactive_runtime_pairs_by_rank: dict[int, list[tuple[int, int]]] = {}
                active_free_count_by_rank: dict[int, int] = {}
                inactive_export_count_by_rank: dict[int, int] = {}
                paired_prefilled_hits = 0
                loaded_maps_list_by_rank = {
                    rank: [int(value) for value in tensor.tolist()]
                    for rank, tensor in gathered_loaded_maps_by_rank.items()
                }
                resident_maps_list_by_rank = {
                    rank: [int(value) for value in tensor.tolist()]
                    for rank, tensor in gathered_resident_maps_by_rank.items()
                }

                for active_rank in active_ranks:
                    active_runtime_slots = {
                        expert_id: int(local_slot)
                        for expert_id, local_slot in enumerate(
                            resident_maps_list_by_rank[active_rank])
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
                if trace_module:
                    logger.info(
                        "Elastic shrink payload paired active slots ready: rank=%s "
                        "layer=%s module_idx=%s active_free=%s",
                        current_rank, module.layer_idx, payload_module_count,
                        active_free_count_by_rank)

                for inactive_rank in inactive_ranks:
                    inactive_runtime_pairs = sorted(
                        ((expert_id, int(local_slot))
                         for expert_id, local_slot in enumerate(
                             resident_maps_list_by_rank[inactive_rank])
                         if int(local_slot) >= 0),
                        key=lambda item: item[1],
                    )
                    inactive_runtime_pairs_by_rank[
                        inactive_rank] = inactive_runtime_pairs
                    inactive_export_count_by_rank[inactive_rank] = len(
                        inactive_runtime_pairs)
                if trace_module:
                    logger.info(
                        "Elastic shrink payload paired inactive slots ready: rank=%s "
                        "layer=%s module_idx=%s inactive_exports=%s",
                        current_rank, module.layer_idx, payload_module_count,
                        inactive_export_count_by_rank)

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
                if trace_module:
                    logger.info(
                        "Elastic shrink payload paired matching ready: rank=%s "
                        "layer=%s module_idx=%s pairing_by_inactive=%s",
                        current_rank, module.layer_idx, payload_module_count,
                        pairing_by_inactive)

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
                    for free_slot, (expert_id, inactive_local_slot) in zip(
                            free_loaded_slots, inactive_runtime_pairs):
                        prefilled_loaded_slot = int(
                            loaded_maps_list_by_rank[active_rank][int(expert_id)])
                        if int(prefilled_loaded_slot) == int(free_slot):
                            ordered_rank_assignments[free_slot] = (
                                int(expert_id), int(prefilled_loaded_slot))
                            paired_prefilled_hits += 1
                            continue
                        remote_source_rank, source_slot = _resolve_mode4_npu_source(
                            inactive_rank, int(expert_id),
                            int(inactive_local_slot))
                        ordered_rank_assignments[free_slot] = (
                            int(expert_id), -1)
                        cpu_import_source_rank[int(expert_id)] = int(inactive_rank)
                        cpu_import_target_rank[int(expert_id)] = active_rank
                        mode4_remote_source_rank[int(expert_id)] = int(
                            remote_source_rank)
                        remote_import_source_slot[int(expert_id)] = source_slot

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
                        runtime_log2phy_values[int(
                            expert_id)] = dense_offset + local_slot
                    dense_offset += len(finalized_assignments)
                if trace_module:
                    logger.info(
                        "Elastic shrink payload paired assignments ready: rank=%s "
                        "layer=%s module_idx=%s assigned_counts=%s "
                        "cpu_imports=%s",
                        current_rank, module.layer_idx, payload_module_count,
                        assigned_counts, len(cpu_import_source_rank))

                unique_runtime_slots = set(runtime_log2phy_values)
                if (len(unique_runtime_slots) != logical_num_experts
                        or any(value < 0 for value in runtime_log2phy_values)):
                    raise RuntimeError(
                        f"layer={module.layer_idx}: unique="
                        f"{len(unique_runtime_slots)} "
                        f"logical={logical_num_experts}")
                if trace_module:
                    logger.info(
                        "Elastic shrink payload paired runtime map validated: "
                        "rank=%s layer=%s module_idx=%s unique=%s",
                        current_rank, module.layer_idx, payload_module_count,
                        len(unique_runtime_slots))
                runtime_log2phy_cpu = torch.tensor(runtime_log2phy_values,
                                                   dtype=torch.int32)

                mode5_import_candidate_experts = {
                    int(expert_id)
                    for rank_assignments in ordered_assignments
                    for expert_id, local_id in rank_assignments
                    if int(local_id) < 0
                }
                if trace_module:
                    logger.info(
                        "Elastic shrink payload paired finalize enter: rank=%s "
                        "layer=%s module_idx=%s import_candidates=%s",
                        current_rank, module.layer_idx, payload_module_count,
                        len(mode5_import_candidate_experts))

                (mode5_remote_experts,
                 mode5_cpu_import_experts,
                 cpu_import_source_rank) = _finalize_mode5_import_partition(
                     cpu_import_source_rank,
                     cpu_import_target_rank,
                     remote_import_source_slot,
                     mode4_remote_source_rank,
                     mode5_import_candidate_experts)
                if trace_module:
                    logger.info(
                        "Elastic shrink payload paired finalize done: rank=%s "
                        "layer=%s module_idx=%s mode5_remote=%s mode5_cpu=%s "
                        "cpu_imports=%s",
                        current_rank, module.layer_idx, payload_module_count,
                        len(mode5_remote_experts), len(mode5_cpu_import_experts),
                        len(cpu_import_source_rank))
                module_store_start_t = time.perf_counter()
                self._lossless_shrink_payload[module.layer_idx] = {
                    "assignments": assignments,
                    "ordered_assignments": ordered_assignments,
                    "runtime_log2phy_cpu": runtime_log2phy_cpu,
                    "cpu_import_source_rank": cpu_import_source_rank,
                    "cpu_import_target_rank": cpu_import_target_rank,
                    "mode4_remote_source_rank": mode4_remote_source_rank,
                    "remote_import_source_slot": remote_import_source_slot,
                    "target_per_rank": target_per_rank,
                    "mode5_remote_experts": sorted(mode5_remote_experts),
                    "mode5_cpu_import_experts": sorted(mode5_cpu_import_experts),
                    "paired_prefilled_loaded_hits": paired_prefilled_hits,
                }
                payload_store_ms += (
                    time.perf_counter() - module_store_start_t) * 1000.0
                if trace_module:
                    logger.info(
                        "Elastic shrink payload module stored: rank=%s "
                        "layer=%s module_idx=%s path=paired assignments=%s "
                        "cpu_imports=%s",
                        current_rank, module.layer_idx, payload_module_count,
                        [len(items) for items in assignments],
                        len(cpu_import_source_rank))
                if module.layer_idx == 0 and current_rank == active_ranks[0]:
                    logger.info(
                        "Elastic paired redundant shrink plan: rank=%s active_ranks=%s inactive_ranks=%s target_per_rank=%s pairings=%s paired_imports=%s prefilled_hits=%s",
                        current_rank,
                        active_ranks,
                        inactive_ranks,
                        target_per_rank,
                        sorted((inactive_rank, active_rank)
                               for inactive_rank, active_rank in
                               pairing_by_inactive.items()),
                        len(cpu_import_source_rank),
                        paired_prefilled_hits,
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
                    mode4_needs_remote_source = (
                        envs_ascend.VLLM_ASCEND_ELASTIC_EXECUTION_MODE in (4, 5)
                        and hybrid_resident_capacity > 0
                        and int(selected_source_local_id) >= hybrid_resident_capacity
                    )
                    if mode4_needs_remote_source:
                        source_world_rank = None
                        source_slot = -1
                        # A follow-up shrink may select an expert that was
                        # logically assigned to a previously-active rank but is
                        # physically cached on an older inactive rank. Resolve
                        # through that rank's mode4 remote-source table instead
                        # of treating the active rank as a cache service.
                        for world_rank in source_ranks:
                            loaded_local_id = int(
                                gathered_loaded_maps_by_rank[world_rank][expert_id]
                                .item())
                            if loaded_local_id < 0:
                                continue
                            resolved_rank, resolved_slot = (
                                _resolve_mode4_npu_source(world_rank, expert_id,
                                                          int(loaded_local_id)))
                            if resolved_rank != selected_world_rank:
                                source_world_rank = resolved_rank
                                source_slot = resolved_slot
                                break
                        if source_world_rank is None:
                            source_world_rank, source_slot = (
                                _resolve_mode4_npu_source(selected_world_rank,
                                                          expert_id,
                                                          selected_source_local_id))
                        selected_source_local_id = -1
                        cpu_import_source_rank[expert_id] = selected_world_rank
                        cpu_import_target_rank[expert_id] = selected_world_rank
                        mode4_remote_source_rank[expert_id] = int(
                            source_world_rank)
                        remote_import_source_slot[expert_id] = source_slot
                    elif selected_source_local_id < 0:
                        cpu_source_world_rank = None
                        source_world_rank = None
                        source_slot = -1
                        for world_rank in source_ranks:
                            if world_rank == selected_world_rank:
                                continue
                            loaded_local_id = int(
                                gathered_loaded_maps_by_rank[world_rank][expert_id]
                                .item())
                            if loaded_local_id >= 0:
                                cpu_source_world_rank = world_rank
                                source_world_rank = world_rank
                                source_slot = int(loaded_local_id)
                                break
                        if source_world_rank is None:
                            cpu_source_world_rank = selected_world_rank
                            source_world_rank = selected_world_rank
                            source_slot = int(selected_loaded_local_id)
                        source_world_rank, source_slot = (
                            _resolve_mode4_npu_source(source_world_rank,
                                                      expert_id,
                                                      source_slot))
                        cpu_import_source_rank[expert_id] = int(
                            cpu_source_world_rank)
                        cpu_import_target_rank[expert_id] = selected_world_rank
                        mode4_remote_source_rank[expert_id] = int(
                            source_world_rank)
                        remote_import_source_slot[expert_id] = source_slot
                else:
                    source_world_rank = None
                    source_slot = -1
                    for world_rank in source_ranks:
                        loaded_local_id = int(
                            gathered_loaded_maps_by_rank[world_rank][expert_id].
                            item())
                        if loaded_local_id >= 0:
                            source_world_rank = world_rank
                            source_slot = int(loaded_local_id)
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
                    cpu_source_world_rank = source_world_rank
                    source_world_rank, source_slot = _resolve_mode4_npu_source(
                        source_world_rank, expert_id, source_slot)
                    cpu_import_source_rank[expert_id] = int(cpu_source_world_rank)
                    cpu_import_target_rank[expert_id] = selected_world_rank
                    mode4_remote_source_rank[expert_id] = int(source_world_rank)
                    remote_import_source_slot[expert_id] = source_slot

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

            mode5_import_candidate_experts = {
                int(expert_id)
                for rank_assignments in ordered_assignments
                for expert_id, local_id in rank_assignments
                if int(local_id) < 0
            }

            (mode5_remote_experts,
             mode5_cpu_import_experts,
             cpu_import_source_rank) = _finalize_mode5_import_partition(
                 cpu_import_source_rank,
                 cpu_import_target_rank,
                 remote_import_source_slot,
                 mode4_remote_source_rank,
                 mode5_import_candidate_experts)
            module_store_start_t = time.perf_counter()
            self._lossless_shrink_payload[module.layer_idx] = {
                "assignments": assignments,
                "ordered_assignments": ordered_assignments,
                "runtime_log2phy_cpu": runtime_log2phy_cpu,
                "cpu_import_source_rank": cpu_import_source_rank,
                "cpu_import_target_rank": cpu_import_target_rank,
                "mode4_remote_source_rank": mode4_remote_source_rank,
                "remote_import_source_slot": remote_import_source_slot,
                "target_per_rank": target_per_rank,
                "mode5_remote_experts": sorted(mode5_remote_experts),
                "mode5_cpu_import_experts": sorted(mode5_cpu_import_experts),
            }
            payload_store_ms += (
                time.perf_counter() - module_store_start_t) * 1000.0
            if trace_module:
                logger.info(
                    "Elastic shrink payload module stored: rank=%s layer=%s "
                    "module_idx=%s path=generic assignments=%s cpu_imports=%s "
                    "remote_imports=%s",
                    current_rank, module.layer_idx, payload_module_count,
                    [len(items) for items in assignments],
                    len(cpu_import_source_rank),
                    len(mode4_remote_source_rank))

        if current_rank == source_ranks[0] and payload_module_count > 0:
            logger.info(
                "Elastic shrink payload breakdown: rank=%s active_ranks=%s modules=%s barrier_ms=%.2f metadata_gather_ms=%.2f remote_state_gather_ms=%.2f store_ms=%.2f total_ms=%.2f",
                current_rank,
                active_ranks,
                payload_module_count,
                payload_barrier_ms,
                payload_metadata_gather_ms,
                payload_remote_state_gather_ms,
                payload_store_ms,
                payload_barrier_ms + payload_metadata_gather_ms +
                payload_remote_state_gather_ms + payload_store_ms)

    def _get_lossless_hybrid_import_mode(self) -> str:
        mode = str(
            envs_ascend.VLLM_ASCEND_LOSSLESS_HYBRID_IMPORT_MODE).lower().strip()
        valid_modes = {"cpu_p2p", "npu_p2p_to_cpu"}
        if mode not in valid_modes:
            raise ValueError(
                "VLLM_ASCEND_LOSSLESS_HYBRID_IMPORT_MODE must be one of "
                f"{sorted(valid_modes)}, got {mode!r}.")
        return mode

    def _get_lossless_hybrid_import_chunk_size(self, module=None) -> int:
        chunk_size = int(
            envs_ascend.VLLM_ASCEND_LOSSLESS_HYBRID_IMPORT_CHUNK_EXPERTS)
        module_mode = (int(getattr(module, "elastic_execution_mode", 0))
                       if module is not None else 0)
        if module_mode == 5 and chunk_size == 1:
            # Mode5 keeps correctness by splitting CPU/NPU sources, but
            # per-expert CPU P2P chunks burn a lot of host-side handshake time.
            # When the generic knob is left at its conservative default, use a
            # modest mode5-specific batch size to amortize sends without
            # exploding temporary CPU staging memory.
            chunk_size = max(
                1,
                int(
                    os.getenv("VLLM_ASCEND_MODE5_HYBRID_IMPORT_CHUNK_EXPERTS",
                              "4")))
        return max(1, chunk_size)

    def _iter_lossless_hybrid_import_chunks(
            self, expert_ids: list[int], module=None) -> list[list[int]]:
        chunk_size = self._get_lossless_hybrid_import_chunk_size(module)
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
        if hasattr(module, "export_lossless_expert_cpu_weights"):
            cpu_weights = module.export_lossless_expert_cpu_weights(
                [int(expert_id) for expert_id in expert_ids])
            missing_ids = [
                int(expert_id) for expert_id in expert_ids
                if int(expert_id) not in cpu_weights
            ]
            if missing_ids:
                raise RuntimeError(
                    "CPU export did not return all requested experts at "
                    f"layer={module.layer_idx}: missing={missing_ids[:16]} "
                    f"count={len(missing_ids)} requested={len(expert_ids)}")
            batch_w13 = torch.stack(
                [cpu_weights[int(expert_id)][0] for expert_id in expert_ids],
                dim=0).contiguous()
            batch_w2 = torch.stack(
                [cpu_weights[int(expert_id)][1] for expert_id in expert_ids],
                dim=0).contiguous()
            return batch_w13, batch_w2
        if hasattr(module, "ensure_lossless_cpu_shadow"):
            module.ensure_lossless_cpu_shadow()
        source_w13 = getattr(module, "lossless_cpu_w13_weight", None)
        source_w2 = getattr(module, "lossless_cpu_w2_weight", None)
        if source_w13 is None or source_w2 is None:
            source_w13 = module.w13_weight.detach().cpu()
            source_w2 = module.w2_weight.detach().cpu()
        cpu_shadow_slots = getattr(module, "lossless_cpu_shadow_local_slots", {})
        local_slots: list[int] = []
        missing_cpu_shadow_ids: list[int] = []
        if cpu_shadow_slots:
            for expert_id in expert_ids:
                cpu_slot = cpu_shadow_slots.get(int(expert_id))
                if cpu_slot is None:
                    missing_cpu_shadow_ids.append(int(expert_id))
                    continue
                local_slots.append(int(cpu_slot))
            if missing_cpu_shadow_ids:
                raise RuntimeError(
                    "CPU export requested experts that are not present in the "
                    f"CPU shadow at layer={module.layer_idx}: "
                    f"missing={missing_cpu_shadow_ids[:16]} "
                    f"count={len(missing_cpu_shadow_ids)} "
                    f"cpu_shadow_rows={int(source_w13.shape[0])}. "
                    "For mode=5 this usually means remote-NPU experts leaked "
                    "into the CPU import payload.")
        else:
            export_map = (
                module.loaded_expert_map
                if getattr(module, "loaded_expert_map", None) is not None
                else module.expert_map)
            if export_map is None:
                raise RuntimeError(
                    f"Missing export_map at layer={module.layer_idx}.")
            local_slots = [
                int(export_map[int(expert_id)].item()) for expert_id in expert_ids
            ]
        if any(local_slot < 0 for local_slot in local_slots):
            raise RuntimeError(
                f"Invalid CPU export slot at layer={module.layer_idx}: "
                f"expert_ids={expert_ids} local_slots={local_slots}")
        max_slot = max(local_slots) if local_slots else -1
        if (max_slot >= int(source_w13.shape[0])
                or max_slot >= int(source_w2.shape[0])):
            raise RuntimeError(
                "CPU export slot exceeds available CPU shadow rows at "
                f"layer={module.layer_idx}: expert_ids={expert_ids[:16]} "
                f"local_slots={local_slots[:16]} max_slot={max_slot} "
                f"w13_rows={int(source_w13.shape[0])} "
                f"w2_rows={int(source_w2.shape[0])}.")
        export_index = torch.tensor(local_slots, device="cpu", dtype=torch.long)
        return (source_w13.index_select(0, export_index).contiguous(),
                source_w2.index_select(0, export_index).contiguous())

    def _export_lossless_expert_cpu_weight_dict(
            self, module,
            expert_ids: list[int]) -> dict[int, tuple[torch.Tensor, torch.Tensor]]:
        if not expert_ids:
            return {}
        ordered_ids = [int(expert_id) for expert_id in expert_ids]
        if hasattr(module, "export_lossless_expert_cpu_weights"):
            cpu_weights = module.export_lossless_expert_cpu_weights(ordered_ids)
            missing_ids = [
                expert_id for expert_id in ordered_ids if expert_id not in cpu_weights
            ]
            if missing_ids:
                raise RuntimeError(
                    "CPU export did not return all requested experts at "
                    f"layer={module.layer_idx}: missing={missing_ids[:16]} "
                    f"count={len(missing_ids)} requested={len(ordered_ids)}")
            return {
                expert_id: cpu_weights[expert_id]
                for expert_id in ordered_ids
            }
        batch_w13, batch_w2 = self._export_lossless_expert_cpu_weight_batch(
            module, ordered_ids)
        return {
            expert_id: (batch_w13[idx].detach(), batch_w2[idx].detach())
            for idx, expert_id in enumerate(ordered_ids)
        }

    def _get_lossless_hybrid_cpu_p2p_send_buffers(
            self, module, rows: int) -> tuple[torch.Tensor, torch.Tensor]:
        cache = getattr(self, "_lossless_hybrid_cpu_p2p_send_buffer_cache", None)
        if cache is None:
            cache = {}
            self._lossless_hybrid_cpu_p2p_send_buffer_cache = cache
        key = (
            int(getattr(module, "layer_idx", -1)),
            int(rows),
            tuple(module.w13_weight.shape[1:]),
            tuple(module.w2_weight.shape[1:]),
            str(module.w13_weight.dtype),
            str(module.w2_weight.dtype),
        )
        cached = cache.get(key)
        if cached is None:
            cached = (
                torch.empty((rows, ) + tuple(module.w13_weight.shape[1:]),
                            device="cpu",
                            dtype=module.w13_weight.dtype),
                torch.empty((rows, ) + tuple(module.w2_weight.shape[1:]),
                            device="cpu",
                            dtype=module.w2_weight.dtype),
            )
            cache[key] = cached
        return cached

    def _get_lossless_hybrid_cpu_p2p_recv_buffers(
            self, module, rows: int) -> tuple[torch.Tensor, torch.Tensor]:
        cache = getattr(self, "_lossless_hybrid_cpu_p2p_recv_buffer_cache", None)
        if cache is None:
            cache = {}
            self._lossless_hybrid_cpu_p2p_recv_buffer_cache = cache
        key = (
            int(getattr(module, "layer_idx", -1)),
            int(rows),
            tuple(module.w13_weight.shape[1:]),
            tuple(module.w2_weight.shape[1:]),
            str(module.w13_weight.dtype),
            str(module.w2_weight.dtype),
        )
        cached = cache.get(key)
        if cached is None:
            cached = (
                torch.empty((rows, ) + tuple(module.w13_weight.shape[1:]),
                            device="cpu",
                            dtype=module.w13_weight.dtype),
                torch.empty((rows, ) + tuple(module.w2_weight.shape[1:]),
                            device="cpu",
                            dtype=module.w2_weight.dtype),
            )
            cache[key] = cached
        return cached

    def _use_lossless_hybrid_cpu_p2p_flat_buffer(self, module) -> bool:
        # This experiment regressed rebuild time badly enough to outweigh the
        # modest preload savings. Keep the helper dormant until a better
        # implementation is available.
        return False

    def _get_lossless_hybrid_cpu_p2p_flat_send_buffers(
            self, module, rows: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        cache = getattr(self, "_lossless_hybrid_cpu_p2p_flat_send_buffer_cache",
                        None)
        if cache is None:
            cache = {}
            self._lossless_hybrid_cpu_p2p_flat_send_buffer_cache = cache
        w13_shape = (rows, ) + tuple(module.w13_weight.shape[1:])
        w2_shape = (rows, ) + tuple(module.w2_weight.shape[1:])
        w13_numel = int(math.prod(w13_shape))
        w2_numel = int(math.prod(w2_shape))
        key = (
            int(getattr(module, "layer_idx", -1)),
            int(rows),
            tuple(module.w13_weight.shape[1:]),
            tuple(module.w2_weight.shape[1:]),
            str(module.w13_weight.dtype),
            str(module.w2_weight.dtype),
        )
        cached = cache.get(key)
        if cached is None:
            flat = torch.empty((w13_numel + w2_numel, ),
                               device="cpu",
                               dtype=module.w13_weight.dtype)
            send_w13 = flat[:w13_numel].view(w13_shape)
            send_w2 = flat[w13_numel:w13_numel + w2_numel].view(w2_shape)
            cached = (flat, send_w13, send_w2)
            cache[key] = cached
        return cached

    def _get_lossless_hybrid_cpu_p2p_flat_recv_buffers(
            self, module, rows: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        cache = getattr(self, "_lossless_hybrid_cpu_p2p_flat_recv_buffer_cache",
                        None)
        if cache is None:
            cache = {}
            self._lossless_hybrid_cpu_p2p_flat_recv_buffer_cache = cache
        w13_shape = (rows, ) + tuple(module.w13_weight.shape[1:])
        w2_shape = (rows, ) + tuple(module.w2_weight.shape[1:])
        w13_numel = int(math.prod(w13_shape))
        w2_numel = int(math.prod(w2_shape))
        key = (
            int(getattr(module, "layer_idx", -1)),
            int(rows),
            tuple(module.w13_weight.shape[1:]),
            tuple(module.w2_weight.shape[1:]),
            str(module.w13_weight.dtype),
            str(module.w2_weight.dtype),
        )
        cached = cache.get(key)
        if cached is None:
            flat = torch.empty((w13_numel + w2_numel, ),
                               device="cpu",
                               dtype=module.w13_weight.dtype)
            recv_w13 = flat[:w13_numel].view(w13_shape)
            recv_w2 = flat[w13_numel:w13_numel + w2_numel].view(w2_shape)
            cached = (flat, recv_w13, recv_w2)
            cache[key] = cached
        return cached

    def _store_lossless_import_cpu_batch(
            self,
            local_cpu_import_weights: dict[int, tuple[torch.Tensor, torch.Tensor]],
            expert_ids: list[int],
            batch_w13: torch.Tensor,
            batch_w2: torch.Tensor) -> None:
        for idx, expert_id in enumerate(expert_ids):
            # Keep row views alive and defer the unavoidable materialization
            # copy to the later stack/fill path. Cloning here doubles the CPU
            # staging traffic for every imported expert.
            local_cpu_import_weights[int(expert_id)] = (
                batch_w13[idx].detach(),
                batch_w2[idx].detach(),
            )

    def _mode4_fetch_remote_experts_to_slot(
            self,
            layer,
            remote_assignments: list[tuple[int, int, int, int]],
            dst_w13: torch.Tensor,
            dst_w2: torch.Tensor,
            *,
            timing_events: Optional[dict[str, object]] = None,
            async_copy: bool = False) -> None:
        if not remote_assignments:
            return
        world_group = get_world_group()
        cpu_group = world_group.cpu_group
        device_group = world_group.device_group
        assignments_by_rank: dict[int, list[tuple[int, int, int]]] = {}
        request_tag = 0
        assignment_rows: list[tuple[int, int, int, int, int]] = []
        for assignment in remote_assignments:
            if len(assignment) == 5:
                slot_idx, remote_rank, remote_slot, expert_id, layer_idx = assignment
            else:
                slot_idx, remote_rank, remote_slot, expert_id = assignment
                layer_idx = int(getattr(layer, "layer_idx", 0))
            assignment_rows.append((int(slot_idx), int(remote_rank),
                                    int(remote_slot), int(expert_id),
                                    int(layer_idx)))
        for slot_idx, remote_rank, remote_slot, expert_id, layer_idx in assignment_rows:
            assignments_by_rank.setdefault(int(remote_rank), []).append(
                (int(slot_idx), int(remote_slot), int(expert_id),
                 int(layer_idx)))
        remote_items = list(sorted(assignments_by_rank.items()))
        execution_mode = int(getattr(layer, "elastic_execution_mode", 4))
        # Mode5 has multiple remote cache ranks once the active group shrinks
        # below 8. Serial fetch can deadlock because every owner waits for one
        # HCCL payload before issuing the next cache-rank request. Keep all
        # remote requests/recvs in flight so cache ranks can make progress
        # independently.
        parallel_remote_fetch_default = "1"
        parallel_remote_fetch_env = (
            "VLLM_ASCEND_MODE5_PARALLEL_REMOTE_FETCH"
            if execution_mode == 5 else "VLLM_ASCEND_MODE4_PARALLEL_REMOTE_FETCH")
        parallel_remote_fetch = (
            len(remote_items) > 1
            and os.getenv(parallel_remote_fetch_env,
                          parallel_remote_fetch_default).lower()
            in ("1", "true", "yes", "on"))
        pending_recvs: list[tuple[object, torch.Tensor, int,
                                  list[tuple[int, int, int, int]]]] = []
        pending_control_msgs: list[tuple[int, list[tuple[int, int, int, int]],
                                         list[object], torch.Tensor]] = []
        request_cache = getattr(self, "_mode4_remote_request_buffer_cache",
                                None)
        if request_cache is None:
            request_cache = {}
            self._mode4_remote_request_buffer_cache = request_cache
        for remote_rank, assignments in remote_items:
            fetch_log_key = (
                int(execution_mode),
                int(getattr(layer, "layer_idx", -1)),
                tuple(sorted(int(rank) for rank in assignments_by_rank)),
                int(len(remote_assignments)),
            )
            begin_logged = getattr(self, "_mode4_fetch_begin_logged_keys",
                                   set())
            if (getattr(layer, "layer_idx", -1) == 0
                    and fetch_log_key not in begin_logged):
                logger.info(
                    "Mode%s remote-NPU double-buffer fetch begin: rank=%s layer=%s remote_rank=%s remote_experts=%s remote_rank_count=%s parallel=%s env=%s",
                    execution_mode,
                    self.rank,
                    getattr(layer, "layer_idx", -1),
                    remote_rank,
                    len(assignments),
                    len(remote_items),
                    int(parallel_remote_fetch),
                    parallel_remote_fetch_env,
                )
                begin_logged.add(fetch_log_key)
                self._mode4_fetch_begin_logged_keys = begin_logged
            request_rows = len(assignments)
            request_cols = 3
            if (execution_mode == 5
                    and _mode5_single_control_message_remote()):
                # Mode5 requests are always 3-column rows. Collapse shape and
                # payload into a single CPU control message to reduce host-side
                # send/wait overhead per remote cache rank.
                if int(request_rows) > 128:
                    raise RuntimeError(
                        "Mode5 remote request exceeds fixed control capacity: "
                        f"rows={int(request_rows)} max_rows=128 "
                        f"layer={getattr(layer, 'layer_idx', -1)} "
                        f"remote_rank={remote_rank}")
                request = torch.tensor(
                    [[int(layer_idx), int(remote_slot), int(expert_id)]
                     for _slot_idx, remote_slot, expert_id, layer_idx in
                     assignments],
                    device="cpu",
                    dtype=torch.int64,
                )
                control = _mode4_get_request_cpu_buffer(
                    request_cache,
                    ("control_send", int(remote_rank), 129, request_cols),
                    129,
                    request_cols,
                )
                control.zero_()
                control[0, 0] = int(request_rows)
                control[0, 1] = int(request_cols)
                control[1:1 + int(request.shape[0])].copy_(
                    request, non_blocking=False)
                cpu_send_reqs = [
                    torch.distributed.isend(control,
                                            dst=remote_rank,
                                            group=cpu_group,
                                            tag=request_tag)
                ]
            else:
                request_key = ("request", int(remote_rank), request_rows,
                               request_cols)
                request = _mode4_get_request_cpu_buffer(request_cache,
                                                        request_key,
                                                        request_rows,
                                                        request_cols)
                request_src = _mode4_build_request_cpu_tensor(
                    assignments, mode5_assignments=True)
                request.copy_(request_src, non_blocking=False)
                shape = torch.tensor([int(request.shape[0]),
                                      int(request.shape[1])],
                                     device="cpu",
                                     dtype=torch.int64)
                cpu_send_reqs = [
                    torch.distributed.isend(shape,
                                            dst=remote_rank,
                                            group=cpu_group,
                                            tag=request_tag),
                    torch.distributed.isend(request,
                                            dst=remote_rank,
                                            group=cpu_group,
                                            tag=request_tag),
                ]
            recv_cache = self._mode4_remote_recv_buffer_cache
            rows = len(assignments)
            recv_key_flat = ("recv_flat", remote_rank,
                             _mode4_flat_payload_signature(
                                 dst_w13, dst_w2, rows))
            recv_flat = _mode4_get_flat_payload(recv_cache, recv_key_flat,
                                                dst_w13, dst_w2, rows)
            pending_control_msgs.append((int(remote_rank), assignments,
                                         cpu_send_reqs, recv_flat))
        # Keep stage-8/4 build-sources behavior intact while avoiding an
        # unnecessary global barrier on owner-side control sends. We only need
        # per-rank ordering between "control send for rank R" and
        # "payload recv for rank R"; earlier/later ranks do not need to wait on
        # each other before posting device irecv.
        request_logged = getattr(self, "_mode4_fetch_request_logged_keys",
                                 set())
        pending_waited_recvs: list[tuple[object, torch.Tensor, int,
                                         list[tuple[int, int, int, int]]]] = []
        for remote_rank, assignments, cpu_send_reqs, recv_flat in pending_control_msgs:
            for req in cpu_send_reqs:
                req.wait()
            request_log_key = fetch_log_key + (int(remote_rank), )
            if (getattr(layer, "layer_idx", -1) == 0
                    and request_log_key not in request_logged):
                logger.info(
                    "Mode%s remote-NPU double-buffer request sent: rank=%s layer=%s remote_rank=%s remote_experts=%s remote_rank_count=%s parallel=%s",
                    execution_mode,
                    self.rank,
                    getattr(layer, "layer_idx", -1),
                    remote_rank,
                    len(assignments),
                    len(remote_items),
                    int(parallel_remote_fetch),
                )
                request_logged.add(request_log_key)
                self._mode4_fetch_request_logged_keys = request_logged
            req_flat = torch.distributed.irecv(recv_flat,
                                               src=remote_rank,
                                               group=device_group)
            pending_waited_recvs.append((req_flat, recv_flat, int(remote_rank),
                                         assignments))
        if parallel_remote_fetch:
            pending_recvs.extend(pending_waited_recvs)
        else:
            for req_flat, recv_flat, _remote_rank, assignments in pending_waited_recvs:
                req_flat.wait()
                pending_recvs.append((None, recv_flat, _remote_rank,
                                      assignments))
                for _req, _recv_flat, _remote_rank2, _assignments in pending_recvs:
                    self._copy_mode4_remote_fetch_payload_to_slot(
                        _recv_flat, _assignments, dst_w13, dst_w2, async_copy)
                pending_recvs.clear()
        if pending_recvs:
            for req_flat, recv_flat, _remote_rank, assignments in pending_recvs:
                if req_flat is not None:
                    req_flat.wait()
                self._copy_mode4_remote_fetch_payload_to_slot(
                    recv_flat, assignments, dst_w13, dst_w2, async_copy)
            pending_recvs.clear()
        fetch_done_logged = getattr(self, "_mode4_fetch_logged_keys", set())
        if (getattr(layer, "layer_idx", -1) == 0
                and fetch_log_key not in fetch_done_logged):
            logger.info(
                "Mode%s remote-NPU double-buffer fetch: rank=%s layer=%s remote_ranks=%s remote_experts=%s parallel=%s",
                execution_mode,
                self.rank,
                getattr(layer, "layer_idx", -1),
                sorted(assignments_by_rank),
                len(remote_assignments),
                int(parallel_remote_fetch),
            )
            fetch_done_logged.add(fetch_log_key)
            self._mode4_fetch_logged_keys = fetch_done_logged

    def _copy_mode4_remote_fetch_payload_to_slot(
            self,
            recv_flat: torch.Tensor,
            assignments: list[tuple[int, int, int, int]],
            dst_w13: torch.Tensor,
            dst_w2: torch.Tensor,
            async_copy: bool) -> None:
        rows = len(assignments)
        w13_elems = int(dst_w13[0].numel())
        w2_elems = int(dst_w2[0].numel())
        recv_w13 = recv_flat[:rows * w13_elems].view(
            (rows, ) + tuple(dst_w13.shape[1:]))
        recv_w2 = recv_flat[rows * w13_elems:rows *
                            (w13_elems + w2_elems)].view(
                                (rows, ) + tuple(dst_w2.shape[1:]))
        slot_ids = [
            int(slot_idx)
            for slot_idx, _remote_slot, _expert_id, _layer_idx in assignments
        ]
        if not slot_ids:
            return
        # Remote payload rows are already contiguous in recv_w13/recv_w2. Copy
        # them back to the runtime slot using the longest contiguous slot runs
        # we can find instead of falling back to per-row scatter for every
        # non-prefix window.
        run_row_start = 0
        run_slot_start = slot_ids[0]
        prev_slot = slot_ids[0]
        for row_idx, slot_idx in enumerate(slot_ids[1:], start=1):
            if slot_idx == prev_slot + 1:
                prev_slot = slot_idx
                continue
            run_len = row_idx - run_row_start
            dst_w13[run_slot_start:run_slot_start + run_len].copy_(
                recv_w13[run_row_start:run_row_start + run_len],
                non_blocking=async_copy)
            dst_w2[run_slot_start:run_slot_start + run_len].copy_(
                recv_w2[run_row_start:run_row_start + run_len],
                non_blocking=async_copy)
            run_row_start = row_idx
            run_slot_start = slot_idx
            prev_slot = slot_idx
        run_len = len(slot_ids) - run_row_start
        dst_w13[run_slot_start:run_slot_start + run_len].copy_(
            recv_w13[run_row_start:run_row_start + run_len],
            non_blocking=async_copy)
        dst_w2[run_slot_start:run_slot_start + run_len].copy_(
            recv_w2[run_row_start:run_row_start + run_len],
            non_blocking=async_copy)

    def _mode4_warmup_remote_payload_fetch(
            self, active_ranks: list[int], world_group) -> None:
        if not _env_flag("VLLM_ASCEND_MODE4_WARMUP_REMOTE_PAYLOAD_FETCH",
                         "1"):
            return
        if self.rank not in set(int(rank) for rank in active_ranks):
            return
        model_runner = getattr(self, "model_runner", None)
        model = getattr(model_runner, "model", None) if model_runner else None
        if model is None:
            return
        layers = {
            int(module.layer_idx): module
            for module in model.modules()
            if _is_ascend_fused_moe_module(module)
        }
        if not layers:
            return
        payloads = getattr(self, "_lossless_shrink_payload", {})
        if not payloads:
            return
        block_layers = int(os.getenv("VLLM_ASCEND_MODE4_BLOCK_PREFETCH_LAYERS",
                                     "1"))
        if block_layers <= 1:
            return
        active_idx = list(active_ranks).index(self.rank)
        layer_indices = sorted(int(layer_idx) for layer_idx in payloads)
        block_start = min(layer_indices) + block_layers
        if block_start > max(layer_indices):
            return
        request_rows: list[tuple[int, int, int]] = []
        remote_rank: Optional[int] = None
        for layer_idx in range(block_start, block_start + block_layers):
            payload = payloads.get(int(layer_idx))
            if payload is None:
                continue
            ordered = payload.get("ordered_assignments", [])
            if active_idx >= len(ordered):
                continue
            source_rank_by_expert = payload.get("cpu_import_source_rank", {})
            remote_source_rank_by_expert = _mode4_remote_source_rank_map(payload)
            source_slot_by_expert = payload.get("remote_import_source_slot", {})
            target_rank_by_expert = payload.get("cpu_import_target_rank", {})
            for expert_id, _local_slot in ordered[active_idx]:
                expert_id = int(expert_id)
                if target_rank_by_expert.get(expert_id) != self.rank:
                    continue
                if expert_id not in source_rank_by_expert:
                    continue
                source_rank = int(
                    remote_source_rank_by_expert.get(
                        expert_id, source_rank_by_expert[expert_id]))
                if remote_rank is None:
                    remote_rank = source_rank
                elif remote_rank != source_rank:
                    # The first version warms only the common one-cache-rank
                    # shrink-to-8 path. Later floors can still use the generic
                    # fetch path safely.
                    return
                if expert_id not in source_slot_by_expert:
                    continue
                request_rows.append((int(layer_idx),
                                     int(source_slot_by_expert[expert_id]),
                                     expert_id))
        if remote_rank is None or not request_rows:
            return
        first_layer = layers.get(block_start)
        if first_layer is None:
            return
        rows = len(request_rows)
        w13_ref = getattr(first_layer, "w13_weight", None)
        w2_ref = getattr(first_layer, "w2_weight", None)
        if w13_ref is None or w2_ref is None:
            return
        recv_key = ("warmup_recv_flat", int(remote_rank), rows,
                    _mode4_flat_payload_signature(w13_ref, w2_ref, rows))
        recv_flat = _mode4_get_flat_payload(
            self._mode4_remote_recv_buffer_cache, recv_key, w13_ref, w2_ref,
            rows)
        cpu_group = world_group.cpu_group
        device_group = world_group.device_group
        request_cache = getattr(self, "_mode4_remote_request_buffer_cache",
                                None)
        if request_cache is None:
            request_cache = {}
            self._mode4_remote_request_buffer_cache = request_cache
        request = _mode4_get_request_cpu_buffer(request_cache,
                                                ("warmup_request",
                                                 int(remote_rank), rows, 3),
                                                rows, 3)
        request_src = _mode4_build_request_cpu_tensor(request_rows)
        request.copy_(request_src, non_blocking=False)
        warmup_start_t = time.perf_counter()
        if (int(envs_ascend.VLLM_ASCEND_ELASTIC_EXECUTION_MODE) == 5
                and _mode5_single_control_message_remote()):
            if rows > 128:
                raise RuntimeError(
                    "Mode5 warmup remote request exceeds fixed control "
                    f"capacity: rows={rows} max_rows=128 remote_rank={remote_rank}")
            control = _mode4_get_request_cpu_buffer(request_cache,
                                                    ("warmup_control",
                                                     int(remote_rank), 129, 3),
                                                    129, 3)
            control[0, 0] = rows
            control[0, 1] = 3
            control[1:1 + rows].copy_(request, non_blocking=False)
            reqs = [
                torch.distributed.isend(control,
                                        dst=int(remote_rank),
                                        group=cpu_group,
                                        tag=0),
            ]
        else:
            shape = torch.tensor([rows, 3], device="cpu", dtype=torch.int64)
            reqs = [
                torch.distributed.isend(shape,
                                        dst=int(remote_rank),
                                        group=cpu_group,
                                        tag=0),
                torch.distributed.isend(request,
                                        dst=int(remote_rank),
                                        group=cpu_group,
                                        tag=0),
            ]
        for req in reqs:
            req.wait()
        recv_req = torch.distributed.irecv(recv_flat,
                                           src=int(remote_rank),
                                           group=device_group)
        recv_req.wait()
        if torch.npu.is_available():
            torch.npu.synchronize()
        logger.info(
            "Mode4 remote payload warmup done: rank=%s remote_rank=%s block_start=%s rows=%s total_ms=%.2f",
            self.rank,
            remote_rank,
            block_start,
            rows,
            (time.perf_counter() - warmup_start_t) * 1000.0,
        )

    def _mode4_prime_next_block_runtime_slot(
            self, active_ranks: list[int]) -> None:
        if (int(envs_ascend.VLLM_ASCEND_ELASTIC_EXECUTION_MODE) == 5
                and not _env_flag(
                    "VLLM_ASCEND_MODE5_PRIME_NEXT_BLOCK_SLOT", "1")):
            logger.info(
                "Mode5 next-block slot prime skipped: rank=%s reason=mode5_disabled",
                self.rank)
            return
        if not _env_flag("VLLM_ASCEND_MODE4_PRIME_NEXT_BLOCK_SLOT", "1"):
            return
        if self.rank not in set(int(rank) for rank in active_ranks):
            return
        model_runner = getattr(self, "model_runner", None)
        model = getattr(model_runner, "model", None) if model_runner else None
        if model is None:
            return
        block_layers = int(os.getenv("VLLM_ASCEND_MODE4_BLOCK_PREFETCH_LAYERS",
                                     "1"))
        if block_layers <= 1:
            return
        prefetch_stream = getattr(model_runner, "moe_prefetch_stream", None)
        if prefetch_stream is None:
            logger.info(
                "Mode4 next-block slot prime skipped: rank=%s reason=no_prefetch_stream",
                self.rank)
            return
        try:
            from vllm_ascend.ascend_forward_context import set_ascend_forward_context
            from vllm.forward_context import BatchDescriptor
            from vllm.config import CUDAGraphMode
            from vllm_ascend.ops.fused_moe import Mode3DoubleBufferManager
        except Exception:
            logger.exception(
                "Mode4 next-block slot prime failed to import helpers: rank=%s",
                self.rank)
            raise

        num_tokens = int(
            os.getenv("VLLM_ASCEND_MODE4_PRIME_NEXT_BLOCK_TOKENS", "32"))
        num_tokens = max(num_tokens, 1)
        num_tokens_across_dp = torch.full(
            (int(getattr(model_runner, "dp_size", 1)), ),
            num_tokens,
            dtype=torch.int64)
        prime_start_t = time.perf_counter()
        with set_ascend_forward_context(
                None,
                self.vllm_config,
                num_tokens=num_tokens,
                num_tokens_across_dp=num_tokens_across_dp,
                with_prefill=False,
                reserved_mc2_mask=getattr(model_runner, "reserved_mc2_mask",
                                          None),
                moe_comm_type=None,
                num_actual_tokens=num_tokens,
                aclgraph_runtime_mode=CUDAGraphMode.NONE,
                batch_descriptor=BatchDescriptor(num_tokens=num_tokens,
                                                 uniform_decode=False),
                prefetch_stream=getattr(model_runner, "prefetch_stream", None),
                moe_prefetch_stream=prefetch_stream,
                model_instance=model):
            manager = Mode3DoubleBufferManager(model, prefetch_stream)
            target_layer = manager.layer_lookup.get(block_layers)
            if target_layer is None:
                logger.info(
                    "Mode4 next-block slot prime skipped: rank=%s reason=no_target_layer block_start=%s",
                    self.rank, block_layers)
                return
            slot_id = manager._slot_id_for_runtime_layer(target_layer)
            manager.prepare_mode4_block_slot(target_layer,
                                             slot_id,
                                             async_copy=False,
                                             reason="post_shrink_prime")
        if torch.npu.is_available():
            torch.npu.synchronize()
        logger.info(
            "Mode4 next-block slot primed: rank=%s block_start=%s block_layers=%s total_ms=%.2f",
            self.rank, block_layers, block_layers,
            (time.perf_counter() - prime_start_t) * 1000.0)

    def _ensure_mode4_double_buffer_manager(self) -> None:
        if not self._has_mode4_remote_npu_lightweight_module():
            return
        model_runner = getattr(self, "model_runner", None)
        model = getattr(model_runner, "model", None) if model_runner else None
        if model is None:
            return
        prefetch_stream = getattr(model_runner, "moe_prefetch_stream", None)
        if prefetch_stream is None:
            prefetch_stream = torch.npu.Stream(device=getattr(model_runner,
                                                             "device", None))
            setattr(model_runner, "moe_prefetch_stream", prefetch_stream)
            logger.info(
                "Mode4 double-buffer manager created fallback moe prefetch stream: rank=%s",
                self.rank)
        try:
            from vllm_ascend.ops.fused_moe import Mode3DoubleBufferManager
        except Exception:
            logger.exception(
                "Mode4 double-buffer manager init failed to import helper: rank=%s",
                self.rank)
            raise
        manager = getattr(model, "_mode3_double_buffer_manager", None)
        if manager is None:
            manager = Mode3DoubleBufferManager(model, prefetch_stream)
            setattr(model, "_mode3_double_buffer_manager", manager)
        for module in model.modules():
            if _is_ascend_fused_moe_module(module):
                setattr(module, "_mode3_double_buffer_manager", manager)

    def _mode4_build_prepacked_remote_payloads(
            self,
            owner_ranks: list[int],
            expert_cache: dict[tuple[int, int], tuple[torch.Tensor,
                                                     torch.Tensor]]) -> dict[
                                                         tuple[int, tuple[
                                                             tuple[int, int,
                                                                   int],
                                                             ...]],
                                                         torch.Tensor]:
        block_layers = int(os.getenv("VLLM_ASCEND_MODE4_BLOCK_PREFETCH_LAYERS",
                                     "1"))
        if block_layers <= 1:
            return {}
        payloads = getattr(self, "_lossless_shrink_payload", {})
        if not payloads or not owner_ranks:
            return {}
        owner_rank_set = {int(rank) for rank in owner_ranks}
        layer_indices = sorted(int(layer_idx) for layer_idx in payloads)
        if not layer_indices:
            return {}
        first_payload = payloads[layer_indices[0]]
        target_per_rank = int(first_payload.get("target_per_rank", 0))
        if target_per_rank <= 0:
            assignments = first_payload.get("ordered_assignments", [])
            target_per_rank = len(assignments[0]) if assignments else 0
        if target_per_rank <= 0:
            return {}
        prepacked: dict[tuple[int, tuple[tuple[int, int, int], ...]],
                        torch.Tensor] = {}
        send_cache = self._mode4_remote_send_buffer_cache
        block_starts = range(min(layer_indices), max(layer_indices) + 1,
                             block_layers)
        for owner_rank in sorted(owner_rank_set):
            owner_idx = owner_rank % len(payloads[layer_indices[0]].get(
                "ordered_assignments", []) or [0])
            for idx, rank_assignments in enumerate(
                    first_payload.get("ordered_assignments", [])):
                for expert_id, _slot in rank_assignments:
                    if first_payload.get("cpu_import_target_rank", {}).get(
                            int(expert_id)) == owner_rank:
                        owner_idx = idx
                        break
                else:
                    continue
                break
            for block_start in block_starts:
                request_rows: list[tuple[int, int, int]] = []
                send_batches: list[tuple[torch.Tensor, torch.Tensor]] = []
                for layer_idx in range(int(block_start),
                                       int(block_start) + block_layers):
                    payload = payloads.get(int(layer_idx))
                    if payload is None:
                        continue
                    ordered = payload.get("ordered_assignments", [])
                    if owner_idx >= len(ordered):
                        continue
                    source_rank_by_expert = payload.get(
                        "cpu_import_source_rank", {})
                    source_slot_by_expert = payload.get(
                        "remote_import_source_slot", {})
                    target_rank_by_expert = payload.get(
                        "cpu_import_target_rank", {})
                    for expert_id, _local_slot in ordered[owner_idx]:
                        expert_id = int(expert_id)
                        if target_rank_by_expert.get(expert_id) != owner_rank:
                            continue
                        if int(source_rank_by_expert.get(expert_id, -1)) != self.rank:
                            continue
                        if expert_id not in source_slot_by_expert:
                            continue
                        remote_slot = int(source_slot_by_expert[expert_id])
                        cached_pair = expert_cache.get((int(layer_idx),
                                                        remote_slot))
                        if cached_pair is None:
                            continue
                        request_rows.append((int(layer_idx), remote_slot,
                                             expert_id))
                        send_batches.append(cached_pair)
                if not send_batches:
                    continue
                first_w13, first_w2 = send_batches[0]
                rows = len(send_batches)
                send_key_flat = ("prepack_flat", int(owner_rank),
                                 int(block_start),
                                 _mode4_flat_payload_signature(
                                     first_w13, first_w2, rows))
                flat_payload = _mode4_get_flat_payload(
                    send_cache, send_key_flat, first_w13, first_w2, rows)
                w13_elems = int(first_w13.numel())
                w2_elems = int(first_w2.numel())
                batch_w13 = flat_payload[:rows * w13_elems].view(
                    (rows, ) + tuple(first_w13.shape))
                batch_w2 = flat_payload[rows * w13_elems:rows *
                                        (w13_elems + w2_elems)].view(
                                            (rows, ) + tuple(first_w2.shape))
                _mode4_pack_remote_request_rows_to_flat_payload(
                    batch_w13,
                    batch_w2,
                    request_rows,
                    layers,
                    expert_cache,
                )
                request_key = (int(owner_rank), tuple(request_rows))
                prepacked[request_key] = flat_payload
        logger.info(
            "Mode4 remote cache prepacked flat payloads: rank=%s owners=%s entries=%s block_layers=%s",
            self.rank,
            owner_ranks,
            len(prepacked),
            block_layers,
        )
        return prepacked

    def _mode4_remote_cache_service_loop(self,
                                         active_ranks: list[int],
                                         world_group) -> None:
        execution_mode = int(envs_ascend.VLLM_ASCEND_ELASTIC_EXECUTION_MODE)
        mode_name = f"Mode{execution_mode}" if execution_mode in (4, 5) \
            else "Mode4"
        cpu_group = world_group.cpu_group
        device_group = world_group.device_group
        model_runner = getattr(self, "model_runner", None)
        model = getattr(model_runner, "model", None) if model_runner else None
        if model is None:
            return
        layers = {
            int(module.layer_idx): module
            for module in model.modules()
            if _is_ascend_fused_moe_module(module)
        }
        if execution_mode == 5:
            owner_ranks = sorted({
                int(target_rank)
                for payload in getattr(self, "_lossless_shrink_payload",
                                       {}).values()
                for target_rank, source_rank in
                _mode5_remote_cache_owner_edges(payload)
                if int(source_rank) == int(self.rank)
            })
        else:
            owner_ranks = sorted({
                int(target_rank)
                for payload in getattr(self, "_lossless_shrink_payload",
                                       {}).values()
                for expert_id, source_rank in _mode4_remote_source_rank_map(
                    payload).items()
                for target_rank in [payload.get("cpu_import_target_rank",
                                                {}).get(expert_id)]
                if int(source_rank) == int(self.rank) and target_rank is not None
            })
        if not owner_ranks:
            return
        stop_event = getattr(self, "_mode4_remote_cache_stop_event", None)
        if stop_event is None:
            return
        first_layer = layers.get(min(layers)) if layers else None
        if first_layer is not None:
            w13_device = getattr(first_layer, "w13_weight", torch.empty(0)).device
            w2_device = getattr(first_layer, "w2_weight", torch.empty(0)).device
            logger.info(
                "%s remote cache weight residency: rank=%s layer=%s w13_device=%s w2_device=%s owners=%s",
                mode_name,
                self.rank,
                getattr(first_layer, "layer_idx", -1),
                w13_device,
                w2_device,
                owner_ranks,
            )
            if w13_device.type != "npu" or w2_device.type != "npu":
                raise RuntimeError(
                    "Mode4 remote cache requires NPU-resident expert weights "
                    f"before service starts. rank={self.rank} "
                    f"layer={getattr(first_layer, 'layer_idx', -1)} "
                    f"w13_device={w13_device} w2_device={w2_device}. "
                    "Set VLLM_ASCEND_MODE4_KEEP_WEIGHTS_OUT_OF_SLEEP_POOL=1 "
                    "or disable the sleep-pool weight offload path for mode=4.")
        expert_cache: dict[tuple[int, int], tuple[torch.Tensor,
                                                 torch.Tensor]] = {}
        payloads = getattr(self, "_lossless_shrink_payload", {})
        for layer_idx, module in layers.items():
            if module is None:
                continue
            payload = payloads.get(int(layer_idx), {})
            remote_source_rank_by_expert = _mode4_remote_source_rank_map(
                payload)
            source_slot_by_expert = payload.get("remote_import_source_slot", {})
            if execution_mode == 5:
                requested_slots = sorted({
                    int(source_slot_by_expert[expert_id])
                    for expert_id, source_rank in
                    remote_source_rank_by_expert.items()
                    if int(source_rank) == self.rank
                    and expert_id in source_slot_by_expert
                    and int(source_slot_by_expert[expert_id]) >= 0
                })
            else:
                requested_slots = sorted({
                    int(source_slot_by_expert[expert_id])
                    for expert_id, source_rank in
                    remote_source_rank_by_expert.items()
                    if int(source_rank) == self.rank
                    and expert_id in source_slot_by_expert
                })
            if execution_mode == 5:
                needed_slots = requested_slots
            else:
                primary_slots = {
                    int(local_slot)
                    for local_slot in getattr(
                        module, "lossless_mode3_primary_prefix_local_slots",
                        {}).values()
                    if int(local_slot) >= 0
                }
                resident_capacity = 0
                if hasattr(module, "_get_hybrid_resident_capacity"):
                    try:
                        resident_capacity = int(
                            module._get_hybrid_resident_capacity())
                    except Exception:
                        resident_capacity = 0
                if not primary_slots and resident_capacity > 0:
                    primary_slots = set(range(resident_capacity))
                needed_slots = sorted(primary_slots) if primary_slots \
                    else requested_slots
            for remote_slot in needed_slots:
                raw_w13 = module.w13_weight[remote_slot:remote_slot + 1]
                raw_w2 = module.w2_weight[remote_slot:remote_slot + 1]
                send_w13 = raw_w13.contiguous()
                send_w2 = raw_w2.contiguous()
                if send_w13.device.type != "npu" or send_w2.device.type != "npu":
                    raise RuntimeError(
                        "Mode4 remote cache snapshot must stay NPU-resident: "
                        f"rank={self.rank} layer={layer_idx} slot={remote_slot} "
                        f"w13_weight_device={module.w13_weight.device} "
                        f"w2_weight_device={module.w2_weight.device} "
                        f"send_w13_device={send_w13.device} "
                        f"send_w2_device={send_w2.device}")
                expert_cache[(int(layer_idx), int(remote_slot))] = (
                    send_w13, send_w2)
        self._mode4_remote_expert_tensor_cache = expert_cache
        if execution_mode == 5 and not expert_cache:
            logger.info(
                "Mode5 remote cache service skipped empty expert cache: rank=%s owner_ranks=%s active_ranks=%s",
                self.rank, owner_ranks, active_ranks)
            return
        if not getattr(self, "_mode4_cache_snapshot_logged", False):
            retained_bytes = sum(
                int(w13.numel()) * int(w13.element_size()) +
                int(w2.numel()) * int(w2.element_size())
                for w13, w2 in expert_cache.values())
            logger.info(
                "%s remote cache NPU snapshot built: rank=%s layers=%s entries=%s retained_expert_bytes=%s sidecar_shareable=false",
                mode_name,
                self.rank,
                len({layer_idx for layer_idx, _slot in expert_cache}),
                len(expert_cache),
                retained_bytes,
            )
            self._mode4_cache_snapshot_logged = True
        if execution_mode == 5:
            # Mode5 already retains NPU-resident expert snapshots on cache
            # ranks. Prepacking every request shape into additional flat payload
            # tensors burns HBM and has repeatedly left too little slack for the
            # later HCCL send workspace on low-floor stages. Keep the cache
            # empty and build per-request flat payloads on demand instead.
            self._mode4_remote_prepacked_payload_cache = {}
        else:
            self._mode4_remote_prepacked_payload_cache = (
                self._mode4_build_prepacked_remote_payloads(owner_ranks,
                                                            expert_cache))
        request_cache = getattr(self, "_mode4_remote_request_buffer_cache",
                                None)
        if request_cache is None:
            request_cache = {}
            self._mode4_remote_request_buffer_cache = request_cache
        logger.info(
            "%s remote cache service owners: rank=%s owner_ranks=%s listen=any_source",
            mode_name, self.rank, owner_ranks)
        while not stop_event.is_set():
            try:
                if execution_mode == 5 and _mode5_single_control_message_remote():
                    control = _mode4_get_request_cpu_buffer(
                        request_cache,
                        ("control_recv", 129, 3),
                        129, 3)
                    owner_rank = torch.distributed.recv(control,
                                                        src=None,
                                                        group=cpu_group,
                                                        tag=0)
                    if owner_rank is None:
                        logger.warning(
                            "%s remote cache recv returned no source rank: rank=%s",
                            mode_name, self.rank)
                        continue
                    owner_rank = int(owner_rank)
                    rows = int(control[0, 0].item())
                    cols = int(control[0, 1].item())
                    if rows <= 0 or cols <= 0:
                        logger.info(
                            "%s remote cache stop sentinel received: rank=%s owner_rank=%s rows=%s cols=%s",
                            mode_name, self.rank, owner_rank, rows, cols)
                        stop_event.set()
                        return
                    request = control[1:1 + rows, :cols]
                else:
                    shape = torch.empty((2, ), device="cpu", dtype=torch.int64)
                    owner_rank = torch.distributed.recv(shape,
                                                        src=None,
                                                        group=cpu_group,
                                                        tag=0)
                    if owner_rank is None:
                        logger.warning(
                            "%s remote cache recv returned no source rank: rank=%s",
                            mode_name, self.rank)
                        continue
                    owner_rank = int(owner_rank)
                    rows = int(shape[0].item())
                    cols = int(shape[1].item())
                    if rows <= 0 or cols <= 0:
                        logger.info(
                            "%s remote cache stop sentinel received: rank=%s owner_rank=%s rows=%s cols=%s",
                            mode_name, self.rank, owner_rank, rows, cols)
                        stop_event.set()
                        return
                    request = _mode4_get_request_cpu_buffer(
                        request_cache,
                        ("request_recv", int(owner_rank), int(rows),
                         int(cols)),
                        rows,
                        cols)
                    torch.distributed.recv(request,
                                           src=owner_rank,
                                           group=cpu_group,
                                           tag=0)
                if not getattr(self, "_mode4_cache_request_logged", False):
                    first_row = ([int(request[0, col].item())
                                  for col in range(int(cols))]
                                 if rows > 0 else [])
                    logger.info(
                        "%s remote cache request received: rank=%s owner_rank=%s rows=%s cols=%s first=%s",
                        mode_name,
                        self.rank,
                        owner_rank,
                        rows,
                        cols,
                        first_row,
                    )
                    self._mode4_cache_request_logged = True
                request_rows = tuple(
                    (int(request[row_idx, 0].item()),
                     int(request[row_idx, 1].item()),
                     int(request[row_idx, 2].item()))
                    for row_idx in range(int(rows)))
                if execution_mode != 5:
                    prepacked_payload = (
                        self._mode4_remote_prepacked_payload_cache.get(
                            (int(owner_rank), request_rows)))
                    if prepacked_payload is not None:
                        if not getattr(self, "_mode4_cache_prepacked_logged",
                                       False):
                            logger.info(
                                "%s remote cache prepacked send begin: rank=%s owner_rank=%s rows=%s payload_shape=%s",
                                mode_name,
                                self.rank,
                                owner_rank,
                                rows,
                                tuple(prepacked_payload.shape),
                            )
                            self._mode4_cache_prepacked_logged = True
                        req_flat = torch.distributed.isend(
                            prepacked_payload,
                            dst=owner_rank,
                            group=device_group)
                        req_flat.wait()
                        continue
                rows = len(request_rows)
                first_w13 = None
                first_w2 = None
                first_layer = None
                first_slot = None
                first_expert = None
                for layer_idx, remote_slot, expert_id in request_rows:
                    cached_pair = expert_cache.get((layer_idx, remote_slot))
                    if cached_pair is None:
                        raise RuntimeError(
                            f"{mode_name} cache rank missing NPU snapshot: "
                            f"rank={self.rank} owner_rank={owner_rank} "
                            f"layer={layer_idx} remote_slot={remote_slot} "
                            f"expert_id={expert_id}")
                    send_w13, send_w2 = cached_pair
                    if send_w13.device.type != "npu" or send_w2.device.type != "npu":
                        raise RuntimeError(
                            f"{mode_name} remote cache send tensor is not NPU-resident: "
                            f"rank={self.rank} owner_rank={owner_rank} "
                            f"layer={layer_idx} remote_slot={remote_slot} "
                            f"expert_id={expert_id} "
                            f"send_w13_device={send_w13.device} "
                            f"send_w2_device={send_w2.device}")
                    if first_w13 is None:
                        first_w13, first_w2 = send_w13, send_w2
                        first_layer = layer_idx
                        first_slot = remote_slot
                        first_expert = expert_id
                if first_w13 is None or first_w2 is None:
                    continue
                send_cache = self._mode4_remote_send_buffer_cache
                send_key_flat = ("send_flat", owner_rank,
                                 _mode4_flat_payload_signature(
                                     first_w13, first_w2, rows))
                flat_payload = _mode4_get_flat_payload(
                    send_cache, send_key_flat, first_w13, first_w2, rows)
                w13_elems = int(first_w13.numel())
                w2_elems = int(first_w2.numel())
                batch_w13 = flat_payload[:rows * w13_elems].view(
                    (rows, ) + tuple(first_w13.shape))
                batch_w2 = flat_payload[rows * w13_elems:rows *
                                        (w13_elems + w2_elems)].view(
                                            (rows, ) + tuple(first_w2.shape))
                _mode4_pack_remote_request_rows_to_flat_payload(
                    batch_w13,
                    batch_w2,
                    list(request_rows),
                    layers,
                    expert_cache,
                )
                if not getattr(self, "_mode4_cache_send_logged", False):
                    logger.info(
                        "%s remote cache device flat send begin: rank=%s owner_rank=%s layer=%s first_remote_slot=%s first_expert_id=%s rows=%s payload_device=%s payload_shape=%s w13_shape=%s w2_shape=%s",
                        mode_name,
                        self.rank,
                        owner_rank,
                        first_layer,
                        first_slot,
                        first_expert,
                        int(rows),
                        flat_payload.device,
                        tuple(flat_payload.shape),
                        tuple(batch_w13.shape),
                        tuple(batch_w2.shape),
                    )
                    self._mode4_cache_send_logged = True
                req_flat = torch.distributed.isend(flat_payload,
                                                   dst=owner_rank,
                                                   group=device_group)
                req_flat.wait()
                if not getattr(self, "_mode4_cache_send_done_logged", False):
                    logger.info(
                        "%s remote cache device send done: rank=%s owner_rank=%s rows=%s",
                        mode_name,
                        self.rank,
                        owner_rank,
                        rows,
                    )
                    self._mode4_cache_send_done_logged = True
            except Exception as exc:
                if stop_event.is_set():
                    return
                logger.exception(
                    "%s remote cache service failed: rank=%s error=%s",
                    mode_name, self.rank, exc)
                return

    def _start_mode4_remote_cache_service(self, active_ranks: list[int],
                                          world_group) -> None:
        execution_mode = int(envs_ascend.VLLM_ASCEND_ELASTIC_EXECUTION_MODE)
        mode_name = f"Mode{execution_mode}" if execution_mode in (4, 5) \
            else "Mode4"
        current_rank = torch.distributed.get_rank()
        owner_to_cache: dict[int, set[int]] = {}
        cache_to_owner: dict[int, set[int]] = {}
        for payload in getattr(self, "_lossless_shrink_payload",
                               {}).values():
            if execution_mode == 5:
                owner_edges = _mode5_remote_cache_owner_edges(payload)
                for target_rank, source_rank in owner_edges:
                    owner_to_cache.setdefault(int(target_rank), set()).add(
                        int(source_rank))
                    cache_to_owner.setdefault(int(source_rank), set()).add(
                        int(target_rank))
                continue
            source_rank_by_expert = _mode4_remote_source_rank_map(payload)
            target_rank_by_expert = payload.get("cpu_import_target_rank", {})
            for expert_id, source_rank in source_rank_by_expert.items():
                expert_id = int(expert_id)
                target_rank = target_rank_by_expert.get(expert_id)
                if source_rank is None or target_rank is None:
                    continue
                source_rank = int(source_rank)
                target_rank = int(target_rank)
                if source_rank == target_rank:
                    continue
                owner_to_cache.setdefault(target_rank, set()).add(source_rank)
                cache_to_owner.setdefault(source_rank, set()).add(target_rank)
        self._mode4_owned_cache_ranks = sorted(
            owner_to_cache.get(int(current_rank), set()))
        self._mode4_cache_owner_ranks = sorted(
            cache_to_owner.get(int(current_rank), set()))
        self._mode4_owner_to_cache_ranks = {
            int(owner_rank): sorted(int(rank) for rank in cache_ranks)
            for owner_rank, cache_ranks in owner_to_cache.items()
        }
        self._mode4_cache_to_owner_ranks = {
            int(cache_rank): sorted(int(rank) for rank in owner_ranks)
            for cache_rank, owner_ranks in cache_to_owner.items()
        }
        # These are per shrink-window diagnostics. Reset them so a second or
        # third rollout step does not become a silent failure if the remote
        # fetch protocol stalls before the first real decode.
        for attr in (
                "_mode4_fetch_begin_logged",
                "_mode4_fetch_request_logged",
                "_mode4_fetch_logged",
                "_mode4_cache_snapshot_logged",
                "_mode4_cache_request_logged",
                "_mode4_cache_send_logged",
                "_mode4_cache_send_done_logged",
                "_mode4_cache_prepacked_logged",
        ):
            if hasattr(self, attr):
                setattr(self, attr, False)
        if self.rank in active_ranks:
            return
        thread = getattr(self, "_mode4_remote_cache_thread", None)
        if execution_mode == 5 and thread is not None and thread.is_alive():
            # Mode5 changes the CPU/remote-NPU split at every cascade shrink
            # stage. The cache service snapshots its owner set and prepacked
            # payload table at startup, so reusing a stage-8 service for
            # stage-4/2/1 can leave some requests unmatched on the HCCL path.
            # Wake the local listener with a self-sentinel; setting the event
            # alone is not enough because the service is blocked in recv().
            if _mode5_single_control_message_remote():
                sentinel = torch.zeros((129, 3),
                                       device="cpu",
                                       dtype=torch.int64)
            else:
                sentinel = torch.zeros((2, ),
                                       device="cpu",
                                       dtype=torch.int64)
            try:
                req = torch.distributed.isend(sentinel,
                                              dst=int(self.rank),
                                              group=world_group.cpu_group,
                                              tag=0)
                req.wait()
            except Exception as exc:
                logger.warning(
                    "Mode5 stale remote cache self-sentinel failed: rank=%s active_ranks=%s error=%s",
                    self.rank, active_ranks, exc)
            stop_event = getattr(self, "_mode4_remote_cache_stop_event", None)
            if stop_event is not None:
                stop_event.set()
            timeout_s = float(os.getenv(
                "VLLM_ASCEND_MODE5_CACHE_STAGE_STOP_TIMEOUT_S", "30"))
            thread.join(timeout=timeout_s)
            if thread.is_alive():
                raise RuntimeError(
                    "Mode5 stale remote expert cache service did not stop "
                    f"before stage restart: rank={self.rank} "
                    f"active_ranks={active_ranks} "
                    f"owners={getattr(self, '_mode4_cache_owner_ranks', [])} "
                    f"timeout_s={timeout_s}")
            logger.info(
                "Mode5 stale remote expert cache service stopped before restart: rank=%s active_ranks=%s",
                self.rank, active_ranks)
            self._mode4_remote_cache_thread = None
            self._mode4_remote_cache_stop_event = None
            self._mode4_remote_expert_tensor_cache = {}
            self._mode4_remote_recv_buffer_cache = {}
            self._mode4_remote_send_buffer_cache = {}
            self._mode4_remote_prepacked_payload_cache = {}
        if execution_mode == 5 and not self._mode4_cache_owner_ranks:
            # Mode5 changes the remote/CPU split at every shrink stage. Do not
            # leave empty inactive-rank listeners around: they can keep old HCCL
            # receive state alive and collide with the next stage.
            logger.info(
                "Mode5 remote expert cache service skipped: rank=%s active_ranks=%s owners=[]",
                self.rank, active_ranks)
            return
        # Follow-up shrinks (8->4->2->1) can still need cache ranks from older
        # stages. Keep an existing service alive instead of creating a second
        # listener on the same CPU/HCCL control path; each cache rank snapshots
        # all resident prefix slots, so it can serve later-stage owners too.
        thread = getattr(self, "_mode4_remote_cache_thread", None)
        if thread is not None and thread.is_alive():
            logger.info(
                "%s remote expert cache service reused: rank=%s active_ranks=%s owners=%s",
                mode_name, self.rank, active_ranks,
                getattr(self, "_mode4_cache_owner_ranks", []))
            return
        thread = getattr(self, "_mode4_remote_cache_thread", None)
        if thread is not None and thread.is_alive():
            logger.info(
                "%s remote expert cache service reused after stale-stop attempt: rank=%s active_ranks=%s owners=%s",
                mode_name, self.rank, active_ranks,
                getattr(self, "_mode4_cache_owner_ranks", []))
            return
        stop_event = threading.Event()
        self._mode4_remote_cache_stop_event = stop_event
        thread = threading.Thread(target=self._mode4_remote_cache_service_loop,
                                  args=(list(active_ranks), world_group),
                                  daemon=True,
                                  name=f"mode4-remote-cache-rank{self.rank}")
        self._mode4_remote_cache_thread = thread
        thread.start()
        model_runner = getattr(self, "model_runner", None)
        model = getattr(model_runner, "model", None) if model_runner else None
        layer_count = (
            sum(1 for module in model.modules()
                if _is_ascend_fused_moe_module(module))
            if model is not None else 0)
        logger.info(
            "%s remote expert cache service started: rank=%s active_ranks=%s owners=%s layers=%s sidecar_shareable=false",
            mode_name, self.rank,
            active_ranks,
            getattr(self, "_mode4_cache_owner_ranks", []),
            layer_count,
        )

    def _send_mode4_remote_cache_stop_sentinels(self, world_group) -> None:
        if world_group is None:
            return
        execution_mode = int(envs_ascend.VLLM_ASCEND_ELASTIC_EXECUTION_MODE)
        mode_name = f"Mode{execution_mode}" if execution_mode in (4, 5) \
            else "Mode4"
        active_attr = getattr(self, "_elastic_current_active_ranks", None)
        if active_attr is None:
            return
        current_active = sorted(
            int(rank) for rank in active_attr)
        if not current_active:
            return
        # A cache service is blocked in CPU recv(src=None), so setting the local
        # stop_event is insufficient.  Mode5 has stage-specific remote/CPU
        # ownership and services from earlier cascade stages may no longer be in
        # the final owner map. During restore, all ranks have reached the same
        # point, so the current active leader can safely send a sentinel to every
        # detached/cache rank. This avoids leaving stale recv(src=None) service
        # threads alive into the next rollout step.
        if int(self.rank) != int(current_active[0]):
            return
        mapped_cache_ranks = sorted({
            int(cache_rank)
            for owner_rank in current_active
            for cache_rank in getattr(self, "_mode4_owner_to_cache_ranks",
                                      {}).get(int(owner_rank), [])
        })
        if mapped_cache_ranks:
            cache_ranks = mapped_cache_ranks
        else:
            world_size = torch.distributed.get_world_size()
            cache_ranks = [
                int(rank) for rank in range(world_size)
                if int(rank) not in set(current_active)
            ]
        if not cache_ranks:
            logger.info(
                "%s remote cache stop skipped: rank=%s active_ranks=%s cache_ranks=[]",
                mode_name, self.rank, current_active)
            return
        if execution_mode == 5 and _mode5_single_control_message_remote():
            sentinel = torch.zeros((129, 3), device="cpu", dtype=torch.int64)
        else:
            sentinel = torch.zeros((2, ), device="cpu", dtype=torch.int64)
        reqs = []
        failed: list[tuple[int, str]] = []
        for cache_rank in cache_ranks:
            try:
                reqs.append((
                    cache_rank,
                    torch.distributed.isend(sentinel,
                                            dst=int(cache_rank),
                                            group=world_group.cpu_group,
                                            tag=0)))
            except Exception as exc:
                failed.append((int(cache_rank), str(exc)))
        for cache_rank, req in reqs:
            try:
                req.wait()
            except Exception as exc:
                failed.append((int(cache_rank), str(exc)))
        if failed:
            logger.warning(
                "%s remote cache stop sentinel failed: rank=%s active_ranks=%s cache_ranks=%s failed=%s",
                mode_name, self.rank, current_active, cache_ranks, failed[:8])
        else:
            logger.info(
                "%s remote cache stop sentinels issued: rank=%s active_ranks=%s cache_ranks=%s",
                mode_name, self.rank, current_active, cache_ranks)

    def _join_mode4_remote_cache_service(self) -> None:
        stop_event = getattr(self, "_mode4_remote_cache_stop_event", None)
        if stop_event is not None:
            stop_event.set()
        thread = getattr(self, "_mode4_remote_cache_thread", None)
        if thread is not None and thread.is_alive():
            thread.join(timeout=float(os.getenv(
                "VLLM_ASCEND_MODE4_CACHE_STOP_TIMEOUT_S", "5")))
            if thread.is_alive():
                logger.warning(
                    "Mode%s remote expert cache service did not stop cleanly: rank=%s thread=%s",
                    envs_ascend.VLLM_ASCEND_ELASTIC_EXECUTION_MODE,
                    self.rank, thread.name)

    def _stop_mode4_remote_cache_service(self, world_group=None) -> None:
        try:
            self._send_mode4_remote_cache_stop_sentinels(world_group)
        finally:
            self._join_mode4_remote_cache_service()
        self._mode4_remote_cache_thread = None
        self._mode4_remote_cache_stop_event = None
        self._mode4_remote_expert_tensor_cache = {}
        self._mode4_remote_recv_buffer_cache = {}
        self._mode4_remote_send_buffer_cache = {}
        self._mode4_remote_prepacked_payload_cache = {}
        self._mode4_owned_cache_ranks = []
        self._mode4_cache_owner_ranks = []
        self._mode4_owner_to_cache_ranks = {}
        self._mode4_cache_to_owner_ranks = {}

    def _stream_lossless_hybrid_import_weights_p2p(
            self,
            module,
            active_ranks: list[int],
            source_cpu_group,
            source_device_group,
            local_needed_cpu_import_ids: set[int],
            cpu_import_source_rank: dict[int, int],
            cpu_import_target_rank: dict[int, int],
            cpu_import_expert_filter: Optional[set[int]] = None,
            participate_only: bool = False
    ) -> tuple[dict[int, tuple[torch.Tensor, torch.Tensor]], dict[int, int],
               dict[str, float | str]]:
        stream_import_start_t = time.perf_counter()
        current_rank = torch.distributed.get_rank()
        transfer_mode = self._get_lossless_hybrid_import_mode()
        use_flat_cpu_p2p = (
            transfer_mode == "cpu_p2p"
            and self._use_lossless_hybrid_cpu_p2p_flat_buffer(module))
        local_cpu_import_weights: dict[int, tuple[torch.Tensor,
                                                  torch.Tensor]] = {}
        local_direct_import_slots: dict[int, int] = {}
        stream_import_stats: dict[str, float | str] = {
            "transfer_mode": ("cpu_p2p_flat"
                               if use_flat_cpu_p2p else transfer_mode),
            "transfer_pairs": 0.0,
            "remote_experts": 0.0,
            "local_only_experts": 0.0,
            "chunks": 0.0,
            "local_copy_export_ms": 0.0,
            "send_export_ms": 0.0,
            "send_pack_copy_ms": 0.0,
            "send_to_device_ms": 0.0,
            "send_wait_ms": 0.0,
            "recv_wait_ms": 0.0,
            "recv_to_cpu_ms": 0.0,
            "recv_store_ms": 0.0,
            "total_ms": 0.0,
        }
        transfer_ids_by_pair: dict[tuple[int, int], list[int]] = {}
        local_only_ids: list[int] = []
        if cpu_import_expert_filter is not None:
            cpu_import_expert_filter = set(
                int(expert_id) for expert_id in cpu_import_expert_filter)

        for expert_id, source_rank in cpu_import_source_rank.items():
            expert_id = int(expert_id)
            if (cpu_import_expert_filter is not None
                    and expert_id not in cpu_import_expert_filter):
                continue
            target_rank = cpu_import_target_rank.get(expert_id)
            if source_rank is None or target_rank is None:
                continue
            if source_rank == target_rank:
                local_only_ids.append(expert_id)
                continue
            transfer_ids_by_pair.setdefault((int(source_rank), int(target_rank)),
                                            []).append(expert_id)

        for pair in transfer_ids_by_pair:
            transfer_ids_by_pair[pair].sort()

        stream_import_stats["transfer_pairs"] = float(
            len(transfer_ids_by_pair))
        stream_import_stats["remote_experts"] = float(
            sum(len(expert_ids)
                for expert_ids in transfer_ids_by_pair.values()))

        valid_source_ranks = sorted(
            int(rank) for rank in cpu_import_source_rank.values()
            if rank is not None)
        chunk_experts = self._get_lossless_hybrid_import_chunk_size(module)
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
                chunk_experts,
            )

        local_copy_ids = sorted(
            expert_id for expert_id in local_only_ids
            if cpu_import_target_rank.get(expert_id) == current_rank
            and expert_id in local_needed_cpu_import_ids)
        stream_import_stats["local_only_experts"] = float(len(local_copy_ids))
        if local_copy_ids and not participate_only:
            local_copy_export_start_t = time.perf_counter()
            local_cpu_import_weights.update(
                self._export_lossless_expert_cpu_weight_dict(
                    module, local_copy_ids))
            stream_import_stats["local_copy_export_ms"] += (
                time.perf_counter() - local_copy_export_start_t) * 1000.0

        w13_tail_shape = tuple(module.w13_weight.shape[1:])
        w2_tail_shape = tuple(module.w2_weight.shape[1:])
        w13_dtype = module.w13_weight.dtype
        w2_dtype = module.w2_weight.dtype
        npu_device = (module.expert_map.device
                      if module.expert_map is not None else module.w13_weight.device)

        for (source_rank, target_rank), expert_ids in sorted(
                transfer_ids_by_pair.items()):
            expert_chunks = list(
                self._iter_lossless_hybrid_import_chunks(
                    expert_ids, module=module))
            stream_import_stats["chunks"] += float(len(expert_chunks))
            if current_rank == source_rank:
                for expert_chunk in expert_chunks:
                    if transfer_mode == "cpu_p2p":
                        if use_flat_cpu_p2p:
                            send_flat, send_w13, send_w2 = (
                                self._get_lossless_hybrid_cpu_p2p_flat_send_buffers(
                                    module, len(expert_chunk)))
                        else:
                            send_w13, send_w2 = (
                                self._get_lossless_hybrid_cpu_p2p_send_buffers(
                                    module, len(expert_chunk)))
                        send_export_start_t = time.perf_counter()
                        send_cpu_weights = (
                            self._export_lossless_expert_cpu_weight_dict(
                                module, expert_chunk))
                        stream_import_stats["send_export_ms"] += (
                            time.perf_counter() - send_export_start_t) * 1000.0
                        send_pack_copy_start_t = time.perf_counter()
                        for row_idx, expert_id in enumerate(expert_chunk):
                            source_w13, source_w2 = send_cpu_weights[int(
                                expert_id)]
                            send_w13[row_idx].copy_(source_w13,
                                                    non_blocking=False)
                            send_w2[row_idx].copy_(source_w2,
                                                   non_blocking=False)
                        stream_import_stats["send_pack_copy_ms"] += (
                            time.perf_counter() -
                            send_pack_copy_start_t) * 1000.0
                        send_group = source_cpu_group
                    else:
                        send_export_start_t = time.perf_counter()
                        send_cpu_w13, send_cpu_w2 = (
                            self._export_lossless_expert_cpu_weight_batch(
                                module, expert_chunk))
                        stream_import_stats["send_export_ms"] += (
                            time.perf_counter() - send_export_start_t) * 1000.0
                        send_to_device_start_t = time.perf_counter()
                        send_w13 = send_cpu_w13.to(device=npu_device,
                                                   non_blocking=False)
                        send_w2 = send_cpu_w2.to(device=npu_device,
                                                 non_blocking=False)
                        stream_import_stats["send_to_device_ms"] += (
                            time.perf_counter() -
                            send_to_device_start_t) * 1000.0
                        send_group = source_device_group
                    send_wait_start_t = time.perf_counter()
                    if transfer_mode == "cpu_p2p" and use_flat_cpu_p2p:
                        send_req = torch.distributed.isend(send_flat,
                                                           dst=target_rank,
                                                           group=send_group)
                        send_req.wait()
                    else:
                        send_req_w13 = torch.distributed.isend(send_w13,
                                                               dst=target_rank,
                                                               group=send_group)
                        send_req_w2 = torch.distributed.isend(send_w2,
                                                              dst=target_rank,
                                                              group=send_group)
                        send_req_w13.wait()
                        send_req_w2.wait()
                    stream_import_stats["send_wait_ms"] += (
                        time.perf_counter() - send_wait_start_t) * 1000.0

            if (not participate_only) and current_rank == target_rank:
                for expert_chunk in expert_chunks:
                    if not all(expert_id in local_needed_cpu_import_ids
                               for expert_id in expert_chunk):
                        continue
                    recv_shape_w13 = (len(expert_chunk), ) + w13_tail_shape
                    recv_shape_w2 = (len(expert_chunk), ) + w2_tail_shape
                    if transfer_mode == "cpu_p2p":
                        if use_flat_cpu_p2p:
                            recv_flat, recv_w13, recv_w2 = (
                                self._get_lossless_hybrid_cpu_p2p_flat_recv_buffers(
                                    module, len(expert_chunk)))
                        else:
                            recv_w13, recv_w2 = (
                                self._get_lossless_hybrid_cpu_p2p_recv_buffers(
                                    module, len(expert_chunk)))
                        recv_group = source_cpu_group
                    else:
                        recv_w13 = torch.empty(recv_shape_w13,
                                               device=npu_device,
                                               dtype=w13_dtype)
                        recv_w2 = torch.empty(recv_shape_w2,
                                              device=npu_device,
                                              dtype=w2_dtype)
                        recv_group = source_device_group
                    recv_wait_start_t = time.perf_counter()
                    if transfer_mode == "cpu_p2p" and use_flat_cpu_p2p:
                        recv_req = torch.distributed.irecv(recv_flat,
                                                           src=source_rank,
                                                           group=recv_group)
                        recv_req.wait()
                    else:
                        recv_req_w13 = torch.distributed.irecv(recv_w13,
                                                               src=source_rank,
                                                               group=recv_group)
                        recv_req_w2 = torch.distributed.irecv(recv_w2,
                                                              src=source_rank,
                                                              group=recv_group)
                        recv_req_w13.wait()
                        recv_req_w2.wait()
                    stream_import_stats["recv_wait_ms"] += (
                        time.perf_counter() - recv_wait_start_t) * 1000.0
                    if transfer_mode != "cpu_p2p":
                        recv_to_cpu_start_t = time.perf_counter()
                        recv_w13 = recv_w13.detach().cpu()
                        recv_w2 = recv_w2.detach().cpu()
                        stream_import_stats["recv_to_cpu_ms"] += (
                            time.perf_counter() -
                            recv_to_cpu_start_t) * 1000.0
                    recv_store_start_t = time.perf_counter()
                    self._store_lossless_import_cpu_batch(
                        local_cpu_import_weights, expert_chunk, recv_w13,
                        recv_w2)
                    stream_import_stats["recv_store_ms"] += (
                        time.perf_counter() - recv_store_start_t) * 1000.0

        stream_import_stats["total_ms"] = (
            time.perf_counter() - stream_import_start_t) * 1000.0
        return (local_cpu_import_weights, local_direct_import_slots,
                stream_import_stats)

    def _stream_lossless_layer_cpu_import_weights(
            self,
            module,
            payload: dict,
            active_ranks: list[int],
            world_group,
            participate_only: bool = False
    ) -> tuple[dict[int, tuple[torch.Tensor, torch.Tensor]], dict[int, int],
               dict[str, float | str]]:
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

        cpu_import_source_rank = {
            int(expert_id): int(source_rank)
            for expert_id, source_rank in payload["cpu_import_source_rank"].items()
            if source_rank is not None
        }
        cpu_import_target_rank = {
            int(expert_id): int(target_rank)
            for expert_id, target_rank in payload.get(
                "cpu_import_target_rank", {}).items()
            if target_rank is not None
        }
        source_slot_by_expert = {
            int(expert_id): int(source_slot)
            for expert_id, source_slot in payload.get(
                "remote_import_source_slot", {}).items()
            if source_slot is not None
        }
        mode5_cpu_import_experts = (
            set(int(expert_id)
                for expert_id in payload.get("mode5_cpu_import_experts", []))
            if int(getattr(module, "elastic_execution_mode", 0)) == 5 else None)
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

        module_mode = int(getattr(module, "elastic_execution_mode", 0))
        if module_mode == 5 and local_needed_cpu_import_ids:
            existing_cpu_shadow = set(
                int(expert_id) for expert_id in getattr(
                    module, "lossless_cpu_shadow_local_slots", {}).keys())
            local_needed_cpu_import_ids = {
                int(expert_id)
                for expert_id in local_needed_cpu_import_ids
                if int(expert_id) not in existing_cpu_shadow
            }

        local_cpu_import_weights: dict[int, tuple[torch.Tensor,
                                                  torch.Tensor]] = {}
        local_direct_import_slots: dict[int, int] = {}
        export_ids_per_source_rank: dict[int, list[int]] = {}
        for expert_id, source_rank in cpu_import_source_rank.items():
            export_ids_per_source_rank.setdefault(source_rank, []).append(expert_id)
        all_import_slot_by_expert: dict[int, int] = {}
        for target_rank, target_idx in active_rank_to_idx.items():
            if target_idx >= len(ordered_assignments):
                continue
            for local_slot, (expert_id, source_local_id) in enumerate(
                    ordered_assignments[target_idx]):
                if source_local_id < 0:
                    all_import_slot_by_expert[int(expert_id)] = int(local_slot)
        for source_rank in export_ids_per_source_rank:
            export_ids_per_source_rank[source_rank].sort(
                key=lambda expert_id: (
                    int(source_slot_by_expert.get(int(expert_id), 1 << 30)),
                    int(all_import_slot_by_expert.get(int(expert_id),
                                                      1 << 30)),
                    int(expert_id),
                ))

        target_owned_local_expert_count = 0
        if ordered_assignments:
            target_owned_local_expert_count = max(
                len(assignments) for assignments in ordered_assignments)
        use_hybrid_cpu_swap = False
        if module_mode in (3, 4, 5):
            # Match _refresh_elastic_parallel_state(): all zero-redundancy
            # double-buffer modes use the hybrid ownership/import path even
            # when the first shrink target exactly fits the resident capacity.
            use_hybrid_cpu_swap = True
        elif (hasattr(module, "should_activate_lossless_hybrid_for_target")
              and module.should_activate_lossless_hybrid_for_target(
                  target_owned_local_expert_count, len(active_ranks))):
            use_hybrid_cpu_swap = True
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
                cpu_import_expert_filter=mode5_cpu_import_experts,
                participate_only=participate_only)

        execution_mode = int(envs_ascend.VLLM_ASCEND_ELASTIC_EXECUTION_MODE)
        zero_redundancy_paired_npu_import = (
            execution_mode != 3
            and getattr(module, "global_redundant_expert_num", 0) <= 0
            and len(previous_active_ranks) == 2 * len(active_ranks)
            and set(active_ranks).issubset(set(previous_active_ranks)))
        disable_direct_npu_import = _env_flag(
            "VLLM_ASCEND_MODE1_DISABLE_DIRECT_NPU_IMPORT", "0")
        use_npu_import = (
            not disable_direct_npu_import
            and (zero_redundancy_paired_npu_import
                 or use_direct_npu_slot_import)
        )
        if (disable_direct_npu_import and module.layer_idx == 0
                and cpu_import_source_rank and current_rank == source_ranks[0]):
            logger.info(
                "Elastic shrink direct NPU import disabled by env: "
                "rank=%s layer=%s active_ranks=%s target_owned_local=%s "
                "zero_redundancy_pair=%s redundant_direct=%s; "
                "falling back to object_broadcast import",
                self.rank,
                module.layer_idx,
                active_ranks,
                target_owned_local_expert_count,
                zero_redundancy_paired_npu_import,
                use_direct_npu_slot_import,
            )

        if use_npu_import:
            import_start = time.perf_counter()
            direct_stats: dict[str, float | str] = {
                "transfer_mode": "direct_npu",
                "transfer_pairs": 0.0,
                "remote_experts": float(len(local_needed_cpu_import_ids)),
                "local_only_experts": 0.0,
                "chunks": 0.0,
                "slot_cache_refresh_ms": 0.0,
                "dry_run_export_ms": 0.0,
                "send_export_ms": 0.0,
                "send_export_source_select_ms": 0.0,
                "send_export_runtime_slot_map_ms": 0.0,
                "send_export_runtime_slot_cpu_hits": 0.0,
                "send_export_runtime_slot_npu_items": 0.0,
                "send_export_slot_cpu_hits": 0.0,
                "send_export_slot_npu_items": 0.0,
                "send_export_contiguous_chunks": 0.0,
                "send_export_index_select_chunks": 0.0,
                "slot_map_expert_cpu_ready": 0.0,
                "slot_map_loaded_cpu_ready": 0.0,
                "slot_map_diag_expert_cpu_ready": 0.0,
                "slot_map_diag_loaded_cpu_ready": 0.0,
                "send_export_map_ms": 0.0,
                "send_export_view_ms": 0.0,
                "send_export_alias_w13_ms": 0.0,
                "send_export_alias_w2_ms": 0.0,
                "send_pack_copy_ms": 0.0,
                "send_to_device_ms": 0.0,
                "send_post_ms": 0.0,
                "send_wait_ms": 0.0,
                "recv_alias_ms": 0.0,
                "recv_post_ms": 0.0,
                "recv_wait_ms": 0.0,
                "recv_to_cpu_ms": 0.0,
                "recv_store_ms": 0.0,
                "recv_direct_slot_chunks": 0.0,
                "recv_temp_buffer_chunks": 0.0,
            }
            w13_tail_shape = tuple(module.w13_weight.shape[1:])
            w2_tail_shape = tuple(module.w2_weight.shape[1:])
            # A paired shrink (16->8, 8->4, 4->2) with enough preallocated
            # loaded slots must receive directly into those slots.  Otherwise
            # the import silently falls back to temporary buffers and later CPU
            # materialization, which breaks the expected floor layout and can
            # leave the next MC2/runtime state inconsistent.
            direct_fill_preallocated_loaded = (
                bool(
                    getattr(module,
                            "lossless_zero_redundancy_preallocated_loaded",
                            False))
                or use_direct_npu_slot_import
                or (zero_redundancy_paired_npu_import
                    and can_direct_fill_loaded_slots))
            allow_scalar_direct_npu_import = _env_flag(
                "VLLM_ASCEND_MODE1_ALLOW_SCALAR_DIRECT_NPU_IMPORT", "0")
            batch_direct_npu_import = (
                direct_fill_preallocated_loaded and (
                    _env_flag("VLLM_ASCEND_MODE1_BATCH_DIRECT_NPU_IMPORT",
                              "1")
                    or not allow_scalar_direct_npu_import))
            batch_experts = max(1, int(os.environ.get(
                "VLLM_ASCEND_MODE1_DIRECT_NPU_IMPORT_BATCH_EXPERTS", "8")))
            direct_verify_log = _env_flag(
                "VLLM_ASCEND_MODE1_DIRECT_NPU_IMPORT_VERIFY_LOG", "0")
            direct_verify_values = max(
                1,
                int(
                    os.environ.get(
                        "VLLM_ASCEND_MODE1_DIRECT_NPU_IMPORT_VERIFY_VALUES",
                        "8")))
            row_trace_log = _env_flag(
                "VLLM_ASCEND_MODE1_DIRECT_NPU_IMPORT_ROW_TRACE_LOG", "0")
            row_trace_layers = _env_int_set(
                "VLLM_ASCEND_MODE1_DIRECT_NPU_IMPORT_ROW_TRACE_LAYERS", "0")
            row_trace_active_rank_limit = max(
                1,
                int(
                    os.environ.get(
                        "VLLM_ASCEND_MODE1_DIRECT_NPU_IMPORT_ROW_TRACE_ACTIVE_RANK_LIMIT",
                        "8")))
            row_trace_rows = max(
                1,
                int(
                    os.environ.get(
                        "VLLM_ASCEND_MODE1_DIRECT_NPU_IMPORT_ROW_TRACE_ROWS",
                        "8")))
            direct_row_trace_enabled = (
                direct_verify_log
                and row_trace_log
                and int(module.layer_idx) in row_trace_layers
                and len(active_ranks) <= row_trace_active_rank_limit)
            if (module.layer_idx == 0 and current_rank == source_ranks[0]
                    and cpu_import_source_rank):
                logger.info(
                    "Elastic shrink preload selected direct NPU import path: rank=%s layer=%s active_ranks=%s target_owned_local=%s loaded_capacity=%s direct_fill=%s redundancy=%s zero_pair=%s slot_direct=%s can_direct_fill=%s batch=%s batch_experts=%s allow_scalar=%s",
                    self.rank,
                    module.layer_idx,
                    active_ranks,
                    target_owned_local_expert_count,
                    int(getattr(module, "loaded_weight_capacity", 0)),
                    direct_fill_preallocated_loaded,
                    int(getattr(module, "global_redundant_expert_num", 0)),
                    zero_redundancy_paired_npu_import,
                    use_direct_npu_slot_import,
                    can_direct_fill_loaded_slots,
                    batch_direct_npu_import,
                    batch_experts,
                    allow_scalar_direct_npu_import,
                )
            slot_map_status_fn = getattr(
                module, "lossless_export_slot_map_status", None)
            if callable(slot_map_status_fn):
                slot_map_status = slot_map_status_fn()
                direct_stats["slot_map_expert_cpu_ready"] = (
                    1.0 if slot_map_status.get("expert_cpu_ready") else 0.0)
                direct_stats["slot_map_loaded_cpu_ready"] = (
                    1.0 if slot_map_status.get("loaded_cpu_ready") else 0.0)
                direct_stats["slot_map_diag_expert_cpu_ready"] = (
                    1.0 if slot_map_status.get("diag_expert_cpu_ready") else 0.0)
                direct_stats["slot_map_diag_loaded_cpu_ready"] = (
                    1.0 if slot_map_status.get("diag_loaded_cpu_ready") else 0.0)
                if (module.layer_idx == 0 and current_rank in source_ranks
                        and len(active_ranks) == 2):
                    logger.info(
                        "Elastic shrink direct NPU export slot-map status: "
                        "rank=%s active_ranks=%s layer=%s source_rank=%s "
                        "status=%s",
                        self.rank, active_ranks, module.layer_idx,
                        current_rank, slot_map_status)
            recv_w13 = None
            recv_w2 = None
            if not direct_fill_preallocated_loaded:
                recv_w13 = torch.empty((1, ) + w13_tail_shape,
                                       device=module.expert_map.device,
                                       dtype=module.w13_weight.dtype)
                recv_w2 = torch.empty((1, ) + w2_tail_shape,
                                      device=module.expert_map.device,
                                      dtype=module.w2_weight.dtype)
            if (_env_flag(
                    "VLLM_ASCEND_MODE1_DIAG_REFRESH_NPU_EXPORT_SLOT_CACHE",
                    "0")
                    and current_rank in source_ranks):
                refresh_fn = getattr(
                    module, "refresh_lossless_npu_export_slot_cache", None)
                if callable(refresh_fn):
                    slot_cache_start_t = time.perf_counter()
                    slot_cache_stats = refresh_fn()
                    slot_cache_refresh_ms = (
                        time.perf_counter() - slot_cache_start_t) * 1000.0
                    direct_stats["slot_cache_refresh_ms"] = (
                        float(direct_stats["slot_cache_refresh_ms"]) +
                        slot_cache_refresh_ms)
                    if len(active_ranks) == 2 and module.layer_idx == 0:
                        logger.info(
                            "Elastic shrink direct NPU export slot-cache refresh: "
                            "rank=%s active_ranks=%s layer=%s "
                            "refresh_ms=%.2f stats=%s",
                            self.rank, active_ranks, module.layer_idx,
                            slot_cache_refresh_ms, slot_cache_stats)
            if (len(active_ranks) == 2 and module.layer_idx == 0
                    and _env_flag(
                        "VLLM_ASCEND_MODE1_DIAG_DIRECT_NPU_EXPORT_DRY_RUN",
                        "0")):
                for source_rank in source_ranks:
                    source_export_ids = export_ids_per_source_rank.get(
                        source_rank, [])
                    if current_rank != source_rank or not source_export_ids:
                        continue
                    dry_expert_id = source_export_ids[0]
                    dry_target_rank = cpu_import_target_rank.get(dry_expert_id)
                    dry_run_start_t = time.perf_counter()
                    dry_w13, dry_w2 = module.export_lossless_expert_npu_weights(
                        [dry_expert_id])
                    dry_run_ms = (
                        time.perf_counter() - dry_run_start_t) * 1000.0
                    direct_stats["dry_run_export_ms"] = (
                        float(direct_stats["dry_run_export_ms"]) + dry_run_ms)
                    dry_debug = getattr(
                        module, "_last_lossless_npu_export_debug", {})
                    logger.info(
                        "Elastic shrink direct NPU export dry-run: rank=%s "
                        "active_ranks=%s layer=%s source_rank=%s target_rank=%s "
                        "expert_id=%s export_ms=%.2f debug=%s",
                        self.rank, active_ranks, module.layer_idx, source_rank,
                        dry_target_rank, dry_expert_id, dry_run_ms, dry_debug)
                    del dry_w13, dry_w2
            def _build_contiguous_direct_npu_chunks(
                    expert_ids: list[int]) -> list[list[int]]:
                if not expert_ids:
                    return []
                missing_source_slots = [
                    int(expert_id) for expert_id in expert_ids
                    if int(expert_id) not in source_slot_by_expert
                ]
                missing_target_slots = [
                    int(expert_id) for expert_id in expert_ids
                    if int(expert_id) not in all_import_slot_by_expert
                ]
                invalid_source_slots = [
                    int(expert_id) for expert_id in expert_ids
                    if int(expert_id) in source_slot_by_expert
                    and int(source_slot_by_expert[int(expert_id)]) < 0
                ]
                invalid_target_slots = [
                    int(expert_id) for expert_id in expert_ids
                    if int(expert_id) in all_import_slot_by_expert
                    and int(all_import_slot_by_expert[int(expert_id)]) < 0
                ]
                if (missing_source_slots or missing_target_slots
                        or invalid_source_slots or invalid_target_slots):
                    raise RuntimeError(
                        "Mode1 batch direct NPU import requires source and "
                        "target slot metadata to avoid the known slow "
                        "index_select/scalar fallback paths. "
                        f"layer={module.layer_idx} "
                        f"missing_source_slots={missing_source_slots[:16]} "
                        f"missing_target_slots={missing_target_slots[:16]} "
                        f"invalid_source_slots={invalid_source_slots[:16]} "
                        f"invalid_target_slots={invalid_target_slots[:16]} "
                        f"count_source={len(missing_source_slots)} "
                        f"count_target={len(missing_target_slots)} "
                        f"count_invalid_source={len(invalid_source_slots)} "
                        f"count_invalid_target={len(invalid_target_slots)}")
                ordered_ids = sorted(
                    [int(expert_id) for expert_id in expert_ids],
                    key=lambda expert_id: (
                        int(source_slot_by_expert[int(expert_id)]),
                        int(all_import_slot_by_expert[int(expert_id)]),
                        int(expert_id),
                    ))
                chunks: list[list[int]] = []
                current: list[int] = []
                prev_source_slot: int | None = None
                prev_target_slot: int | None = None
                for expert_id in ordered_ids:
                    source_slot = int(source_slot_by_expert[int(expert_id)])
                    target_slot = int(all_import_slot_by_expert[int(expert_id)])
                    starts_new_chunk = (
                        not current or len(current) >= batch_experts
                        or prev_source_slot is None
                        or prev_target_slot is None
                        or source_slot != prev_source_slot + 1
                        or target_slot != prev_target_slot + 1)
                    if starts_new_chunk:
                        if current:
                            chunks.append(current)
                        current = [expert_id]
                    else:
                        current.append(expert_id)
                    prev_source_slot = source_slot
                    prev_target_slot = target_slot
                if current:
                    chunks.append(current)
                return chunks

            if batch_direct_npu_import:
                for source_rank in source_ranks:
                    source_export_ids = export_ids_per_source_rank.get(
                        source_rank, [])
                    if not source_export_ids:
                        continue
                    export_ids_by_target: dict[int, list[int]] = {}
                    for expert_id in source_export_ids:
                        target_rank = cpu_import_target_rank.get(expert_id)
                        if target_rank is None:
                            raise RuntimeError(
                                f"Missing lossless NPU import target for expert {expert_id} "
                                f"at layer={module.layer_idx}.")
                        export_ids_by_target.setdefault(
                            int(target_rank), []).append(int(expert_id))
                    for target_rank in sorted(export_ids_by_target):
                        target_export_ids = _build_contiguous_direct_npu_chunks(
                            export_ids_by_target[target_rank])
                        if (len(active_ranks) == 2 and module.layer_idx == 0
                                and current_rank == source_rank):
                            chunk_plan = []
                            for chunk_ids_for_log in target_export_ids[:8]:
                                first_expert = int(chunk_ids_for_log[0])
                                last_expert = int(chunk_ids_for_log[-1])
                                chunk_plan.append({
                                    "rows": len(chunk_ids_for_log),
                                    "experts": (first_expert, last_expert),
                                    "source_slots": (
                                        int(source_slot_by_expert[first_expert]),
                                        int(source_slot_by_expert[last_expert])),
                                    "target_slots": (
                                        int(all_import_slot_by_expert[
                                            first_expert]),
                                        int(all_import_slot_by_expert[
                                            last_expert])),
                                })
                            logger.info(
                                "Elastic shrink direct NPU batch chunk plan: "
                                "rank=%s active_ranks=%s layer=%s "
                                "source_rank=%s target_rank=%s experts=%s "
                                "chunks=%s batch_experts=%s plan_head=%s",
                                self.rank, active_ranks, module.layer_idx,
                                source_rank, target_rank,
                                len(export_ids_by_target[target_rank]),
                                len(target_export_ids), batch_experts,
                                chunk_plan)
                        for chunk_ids in target_export_ids:
                            chunk_rows = len(chunk_ids)
                            if current_rank == source_rank:
                                export_start_t = time.perf_counter()
                                first_expert = int(chunk_ids[0])
                                source_start_slot = int(
                                    source_slot_by_expert[first_expert])
                                allow_direct_source_range_import = _env_flag(
                                    "VLLM_ASCEND_MODE1_ALLOW_DIRECT_SOURCE_RANGE_IMPORT",
                                    "1")
                                send_range_fn = getattr(
                                    module,
                                    "get_lossless_expert_npu_slot_range_send_buffers",
                                    None)
                                used_direct_source_range = False
                                if (allow_direct_source_range_import
                                        and callable(send_range_fn)):
                                    export_w13, export_w2 = send_range_fn(
                                        source_start_slot, chunk_rows)
                                    used_direct_source_range = True
                                else:
                                    export_w13, export_w2 = (
                                        module.
                                        export_lossless_expert_npu_weights(
                                            chunk_ids))
                                export_ms = (
                                    time.perf_counter() - export_start_t
                                ) * 1000.0
                                direct_stats["send_export_ms"] = (
                                    float(direct_stats["send_export_ms"]) +
                                    export_ms)
                                if used_direct_source_range:
                                    direct_stats["send_export_view_ms"] = (
                                        float(direct_stats[
                                            "send_export_view_ms"]) + export_ms)
                                    direct_stats[
                                        "send_export_contiguous_chunks"] = (
                                            float(direct_stats[
                                                "send_export_contiguous_chunks"])
                                            + 1.0)
                                else:
                                    direct_stats["send_pack_copy_ms"] = (
                                        float(direct_stats[
                                            "send_pack_copy_ms"]) + export_ms)
                                export_debug = (
                                    {}
                                    if used_direct_source_range else getattr(
                                        module,
                                        "_last_lossless_npu_export_debug", {}))
                                if export_debug:
                                    direct_stats["send_export_source_select_ms"] = (
                                        float(direct_stats[
                                            "send_export_source_select_ms"]) +
                                        float(export_debug.get(
                                            "source_select_ms", 0.0)))
                                    direct_stats[
                                        "send_export_runtime_slot_map_ms"] = (
                                            float(direct_stats[
                                                "send_export_runtime_slot_map_ms"])
                                            + float(export_debug.get(
                                                "runtime_slot_map_ms", 0.0)))
                                    runtime_lookup_source = str(
                                        export_debug.get(
                                            "runtime_slot_lookup_source", ""))
                                    export_lookup_source = str(
                                        export_debug.get(
                                            "export_slot_lookup_source", ""))
                                    if runtime_lookup_source.endswith("_cpu"):
                                        direct_stats[
                                            "send_export_runtime_slot_cpu_hits"] = (
                                                float(direct_stats[
                                                    "send_export_runtime_slot_cpu_hits"])
                                                + float(chunk_rows))
                                    elif runtime_lookup_source == "npu_item":
                                        direct_stats[
                                            "send_export_runtime_slot_npu_items"] = (
                                                float(direct_stats[
                                                    "send_export_runtime_slot_npu_items"])
                                                + float(chunk_rows))
                                    if export_lookup_source.endswith("_cpu"):
                                        direct_stats[
                                            "send_export_slot_cpu_hits"] = (
                                                float(direct_stats[
                                                    "send_export_slot_cpu_hits"])
                                                + float(chunk_rows))
                                    elif export_lookup_source == "npu_item":
                                        direct_stats[
                                            "send_export_slot_npu_items"] = (
                                                float(direct_stats[
                                                    "send_export_slot_npu_items"])
                                                + float(chunk_rows))
                                    direct_stats["send_export_map_ms"] = (
                                        float(direct_stats[
                                            "send_export_map_ms"]) +
                                        float(export_debug.get("map_ms", 0.0)))
                                    direct_stats["send_export_view_ms"] = (
                                        float(direct_stats[
                                            "send_export_view_ms"]) +
                                        float(export_debug.get("view_ms", 0.0)))
                                    direct_stats["send_export_alias_w13_ms"] = (
                                        float(direct_stats[
                                            "send_export_alias_w13_ms"]) +
                                        float(export_debug.get(
                                            "alias_w13_ms", 0.0)))
                                    direct_stats["send_export_alias_w2_ms"] = (
                                        float(direct_stats[
                                            "send_export_alias_w2_ms"]) +
                                        float(export_debug.get(
                                            "alias_w2_ms", 0.0)))
                                    if export_debug.get(
                                            "local_slots_contiguous", False):
                                        direct_stats[
                                            "send_export_contiguous_chunks"] = (
                                                float(direct_stats[
                                                    "send_export_contiguous_chunks"])
                                                + 1.0)
                                    else:
                                        direct_stats[
                                            "send_export_index_select_chunks"] = (
                                                float(direct_stats[
                                                    "send_export_index_select_chunks"])
                                            + 1.0)
                                if (direct_verify_log and module.layer_idx == 0
                                        and (len(active_ranks) <= 2
                                             or direct_row_trace_enabled)):
                                    first_target_slot = int(
                                        all_import_slot_by_expert[first_expert])
                                    trace_rows = list(
                                        range(min(chunk_rows, row_trace_rows)))
                                    source_rows = [
                                        source_start_slot + row
                                        for row in trace_rows
                                    ]
                                    logger.warning(
                                        "Mode1 direct NPU import send verify: "
                                        "rank=%s active_ranks=%s layer=%s "
                                        "source_rank=%s target_rank=%s "
                                        "experts=%s first_expert=%s "
                                        "source_slot=%s target_slot=%s "
                                        "direct_source_range=%s export_meta_w13=%s "
                                        "export_meta_w2=%s source_w13_rows=%s "
                                        "source_w2_rows=%s export_w13_rows=%s "
                                        "export_w2_rows=%s export_debug=%s",
                                        self.rank,
                                        active_ranks,
                                        module.layer_idx,
                                        source_rank,
                                        target_rank,
                                        list(map(int, chunk_ids[:8])),
                                        first_expert,
                                        source_start_slot,
                                        first_target_slot,
                                        used_direct_source_range,
                                        _mode1_diag_tensor_meta(export_w13),
                                        _mode1_diag_tensor_meta(export_w2),
                                        _mode1_diag_tensor_rows_stats(
                                            module.w13_weight,
                                            source_rows,
                                            sample_values=direct_verify_values),
                                        _mode1_diag_tensor_rows_stats(
                                            module.w2_weight,
                                            source_rows,
                                            sample_values=direct_verify_values),
                                        _mode1_diag_tensor_rows_stats(
                                            export_w13,
                                            trace_rows,
                                            sample_values=direct_verify_values),
                                        _mode1_diag_tensor_rows_stats(
                                            export_w2,
                                            trace_rows,
                                            sample_values=direct_verify_values),
                                        export_debug,
                                    )
                                send_post_start_t = time.perf_counter()
                                send_w13 = torch.distributed.isend(
                                    export_w13,
                                    dst=target_rank,
                                    group=source_device_group)
                                send_w2 = torch.distributed.isend(
                                    export_w2,
                                    dst=target_rank,
                                    group=source_device_group)
                                direct_stats["send_post_ms"] = (
                                    float(direct_stats["send_post_ms"]) +
                                    (time.perf_counter() - send_post_start_t) *
                                    1000.0)
                                send_wait_start_t = time.perf_counter()
                                send_w13.wait()
                                send_w2.wait()
                                direct_stats["send_wait_ms"] = (
                                    float(direct_stats["send_wait_ms"]) +
                                    (time.perf_counter() -
                                     send_wait_start_t) * 1000.0)
                                direct_stats["transfer_pairs"] = (
                                    float(direct_stats["transfer_pairs"]) +
                                    float(chunk_rows))
                                direct_stats["chunks"] = (
                                    float(direct_stats["chunks"]) + 1.0)
                                del export_w13, export_w2
                                continue
                            if (participate_only
                                    or current_rank != target_rank):
                                continue
                            missing_ids = [
                                expert_id for expert_id in chunk_ids
                                if expert_id not in local_needed_cpu_import_ids
                            ]
                            if missing_ids:
                                raise RuntimeError(
                                    "Direct NPU batch import target is missing "
                                    f"local slots at layer={module.layer_idx}: "
                                    f"rank={self.rank} source_rank={source_rank} "
                                    f"target_rank={target_rank} missing={missing_ids[:8]}")
                            target_slots = [
                                local_import_slot_by_expert[int(expert_id)]
                                for expert_id in chunk_ids
                            ]
                            contiguous_slots = (
                                target_slots == list(
                                    range(target_slots[0],
                                          target_slots[0] + chunk_rows))
                            )
                            recv_alias_start_t = time.perf_counter()
                            direct_recv_to_slots = False
                            if contiguous_slots:
                                recv_range_fn = getattr(
                                    module,
                                    "get_lossless_expert_npu_slot_range_recv_buffers",
                                    None)
                                if callable(recv_range_fn):
                                    recv_target_w13, recv_target_w2 = recv_range_fn(
                                        int(target_slots[0]), int(chunk_rows))
                                    direct_recv_to_slots = True
                                    direct_stats["recv_direct_slot_chunks"] = (
                                        float(direct_stats[
                                            "recv_direct_slot_chunks"]) + 1.0)
                                else:
                                    recv_target_w13 = torch.empty(
                                        (chunk_rows, ) + w13_tail_shape,
                                        device=module.w13_weight.device,
                                        dtype=module.w13_weight.dtype)
                                    recv_target_w2 = torch.empty(
                                        (chunk_rows, ) + w2_tail_shape,
                                        device=module.w2_weight.device,
                                        dtype=module.w2_weight.dtype)
                                    direct_stats["recv_temp_buffer_chunks"] = (
                                        float(direct_stats[
                                            "recv_temp_buffer_chunks"]) + 1.0)
                            else:
                                recv_target_w13 = torch.empty(
                                    (chunk_rows, ) + w13_tail_shape,
                                    device=module.w13_weight.device,
                                    dtype=module.w13_weight.dtype)
                                recv_target_w2 = torch.empty(
                                    (chunk_rows, ) + w2_tail_shape,
                                    device=module.w2_weight.device,
                                    dtype=module.w2_weight.dtype)
                                direct_stats["recv_temp_buffer_chunks"] = (
                                    float(direct_stats[
                                        "recv_temp_buffer_chunks"]) + 1.0)
                            direct_stats["recv_alias_ms"] = (
                                float(direct_stats["recv_alias_ms"]) +
                                (time.perf_counter() - recv_alias_start_t) *
                                1000.0)
                            trace_rows = list(
                                range(min(chunk_rows, row_trace_rows)))
                            target_trace_rows = [
                                int(target_slots[row])
                                for row in trace_rows
                            ]
                            if direct_row_trace_enabled:
                                logger.warning(
                                    "Mode1 direct NPU import recv before: "
                                    "rank=%s active_ranks=%s layer=%s "
                                    "source_rank=%s target_rank=%s experts=%s "
                                    "target_slots=%s direct_recv_to_slots=%s "
                                    "contiguous_slots=%s recv_meta_w13=%s "
                                    "recv_meta_w2=%s target_before_w13_rows=%s "
                                    "target_before_w2_rows=%s",
                                    self.rank,
                                    active_ranks,
                                    module.layer_idx,
                                    source_rank,
                                    target_rank,
                                    list(map(int, chunk_ids[:row_trace_rows])),
                                    target_trace_rows,
                                    direct_recv_to_slots,
                                    contiguous_slots,
                                    _mode1_diag_tensor_meta(recv_target_w13),
                                    _mode1_diag_tensor_meta(recv_target_w2),
                                    _mode1_diag_tensor_rows_stats(
                                        module.w13_weight,
                                        target_trace_rows,
                                        sample_values=direct_verify_values),
                                    _mode1_diag_tensor_rows_stats(
                                        module.w2_weight,
                                        target_trace_rows,
                                        sample_values=direct_verify_values),
                                )
                            recv_post_start_t = time.perf_counter()
                            recv_req_w13 = torch.distributed.irecv(
                                recv_target_w13,
                                src=source_rank,
                                group=source_device_group)
                            recv_req_w2 = torch.distributed.irecv(
                                recv_target_w2,
                                src=source_rank,
                                group=source_device_group)
                            direct_stats["recv_post_ms"] = (
                                float(direct_stats["recv_post_ms"]) +
                                (time.perf_counter() - recv_post_start_t) *
                                1000.0)
                            recv_wait_start_t = time.perf_counter()
                            recv_req_w13.wait()
                            recv_req_w2.wait()
                            direct_stats["recv_wait_ms"] = (
                                float(direct_stats["recv_wait_ms"]) +
                                (time.perf_counter() - recv_wait_start_t) *
                                1000.0)
                            if direct_row_trace_enabled:
                                logger.warning(
                                    "Mode1 direct NPU import recv after_irecv: "
                                    "rank=%s active_ranks=%s layer=%s "
                                    "source_rank=%s target_rank=%s experts=%s "
                                    "target_slots=%s direct_recv_to_slots=%s "
                                    "recv_w13_rows=%s recv_w2_rows=%s "
                                    "target_after_irecv_w13_rows=%s "
                                    "target_after_irecv_w2_rows=%s",
                                    self.rank,
                                    active_ranks,
                                    module.layer_idx,
                                    source_rank,
                                    target_rank,
                                    list(map(int, chunk_ids[:row_trace_rows])),
                                    target_trace_rows,
                                    direct_recv_to_slots,
                                    _mode1_diag_tensor_rows_stats(
                                        recv_target_w13,
                                        trace_rows,
                                        sample_values=direct_verify_values),
                                    _mode1_diag_tensor_rows_stats(
                                        recv_target_w2,
                                        trace_rows,
                                        sample_values=direct_verify_values),
                                    _mode1_diag_tensor_rows_stats(
                                        module.w13_weight,
                                        target_trace_rows,
                                        sample_values=direct_verify_values),
                                    _mode1_diag_tensor_rows_stats(
                                        module.w2_weight,
                                        target_trace_rows,
                                        sample_values=direct_verify_values),
                                )
                            store_start_t = time.perf_counter()
                            if direct_recv_to_slots:
                                pass
                            elif contiguous_slots:
                                first_slot = target_slots[0]
                                module.w13_weight[
                                    first_slot:first_slot + chunk_rows].copy_(
                                        recv_target_w13)
                                module.w2_weight[
                                    first_slot:first_slot + chunk_rows].copy_(
                                        recv_target_w2)
                            else:
                                for row_idx, target_slot in enumerate(
                                        target_slots):
                                    module.w13_weight[
                                        target_slot:target_slot + 1].copy_(
                                            recv_target_w13[
                                                row_idx:row_idx + 1])
                                    module.w2_weight[
                                        target_slot:target_slot + 1].copy_(
                                            recv_target_w2[row_idx:row_idx +
                                                           1])
                            direct_stats["recv_store_ms"] = (
                                float(direct_stats["recv_store_ms"]) +
                                (time.perf_counter() - store_start_t) *
                                1000.0)
                            if direct_row_trace_enabled:
                                logger.warning(
                                    "Mode1 direct NPU import recv after_store: "
                                    "rank=%s active_ranks=%s layer=%s "
                                    "source_rank=%s target_rank=%s experts=%s "
                                    "target_slots=%s direct_recv_to_slots=%s "
                                    "target_after_store_w13_rows=%s "
                                    "target_after_store_w2_rows=%s",
                                    self.rank,
                                    active_ranks,
                                    module.layer_idx,
                                    source_rank,
                                    target_rank,
                                    list(map(int, chunk_ids[:row_trace_rows])),
                                    target_trace_rows,
                                    direct_recv_to_slots,
                                    _mode1_diag_tensor_rows_stats(
                                        module.w13_weight,
                                        target_trace_rows,
                                        sample_values=direct_verify_values),
                                    _mode1_diag_tensor_rows_stats(
                                        module.w2_weight,
                                        target_trace_rows,
                                        sample_values=direct_verify_values),
                                )
                            for expert_id, target_slot in zip(
                                    chunk_ids, target_slots):
                                local_direct_import_slots[int(expert_id)] = (
                                    int(target_slot))
                            if (direct_verify_log and module.layer_idx == 0
                                    and len(active_ranks) <= 2):
                                first_expert = int(chunk_ids[0])
                                first_slot = int(target_slots[0])
                                logger.warning(
                                    "Mode1 direct NPU import recv verify: "
                                    "rank=%s active_ranks=%s layer=%s "
                                    "source_rank=%s target_rank=%s "
                                    "experts=%s first_expert=%s target_slot=%s "
                                    "direct_recv_to_slots=%s contiguous_slots=%s "
                                    "w13_slot=%s w2_slot=%s",
                                    self.rank,
                                    active_ranks,
                                    module.layer_idx,
                                    source_rank,
                                    target_rank,
                                    list(map(int, chunk_ids[:8])),
                                    first_expert,
                                    first_slot,
                                    direct_recv_to_slots,
                                    contiguous_slots,
                                    _mode1_diag_tensor_row_stats(
                                        module.w13_weight,
                                        first_slot,
                                        sample_values=direct_verify_values),
                                    _mode1_diag_tensor_row_stats(
                                        module.w2_weight,
                                        first_slot,
                                        sample_values=direct_verify_values),
                                )
                            direct_stats["transfer_pairs"] = (
                                float(direct_stats["transfer_pairs"]) +
                                float(chunk_rows))
                            direct_stats["chunks"] = (
                                float(direct_stats["chunks"]) + 1.0)
                            del recv_target_w13, recv_target_w2
            else:
                for source_rank in source_ranks:
                    source_export_ids = export_ids_per_source_rank.get(
                        source_rank, [])
                    if not source_export_ids:
                        continue
                    for expert_id in source_export_ids:
                        target_rank = cpu_import_target_rank.get(expert_id)
                        if target_rank is None:
                            raise RuntimeError(
                                f"Missing lossless NPU import target for expert {expert_id} "
                                f"at layer={module.layer_idx}.")
                        if current_rank == source_rank:
                            export_start_t = time.perf_counter()
                            export_w13, export_w2 = module.export_lossless_expert_npu_weights(
                                [expert_id])
                            export_ms = (
                                time.perf_counter() - export_start_t) * 1000.0
                            direct_stats["send_export_ms"] = (
                                float(direct_stats["send_export_ms"]) + export_ms)
                            export_debug = getattr(
                                module, "_last_lossless_npu_export_debug", {})
                            if export_debug:
                                direct_stats["send_export_source_select_ms"] = (
                                    float(direct_stats[
                                        "send_export_source_select_ms"]) +
                                    float(
                                        export_debug.get("source_select_ms", 0.0)))
                                direct_stats["send_export_runtime_slot_map_ms"] = (
                                    float(direct_stats[
                                        "send_export_runtime_slot_map_ms"]) +
                                    float(
                                        export_debug.get("runtime_slot_map_ms",
                                                         0.0)))
                                runtime_lookup_source = str(
                                    export_debug.get("runtime_slot_lookup_source",
                                                     ""))
                                export_lookup_source = str(
                                    export_debug.get("export_slot_lookup_source",
                                                     ""))
                                if runtime_lookup_source.endswith("_cpu"):
                                    direct_stats[
                                        "send_export_runtime_slot_cpu_hits"] = (
                                            float(direct_stats[
                                                "send_export_runtime_slot_cpu_hits"])
                                            + 1.0)
                                elif runtime_lookup_source == "npu_item":
                                    direct_stats[
                                        "send_export_runtime_slot_npu_items"] = (
                                            float(direct_stats[
                                                "send_export_runtime_slot_npu_items"])
                                            + 1.0)
                                if export_lookup_source.endswith("_cpu"):
                                    direct_stats["send_export_slot_cpu_hits"] = (
                                        float(direct_stats[
                                            "send_export_slot_cpu_hits"]) + 1.0)
                                elif export_lookup_source == "npu_item":
                                    direct_stats["send_export_slot_npu_items"] = (
                                        float(direct_stats[
                                            "send_export_slot_npu_items"]) + 1.0)
                                direct_stats["send_export_map_ms"] = (
                                    float(direct_stats["send_export_map_ms"]) +
                                    float(export_debug.get("map_ms", 0.0)))
                                direct_stats["send_export_view_ms"] = (
                                    float(direct_stats["send_export_view_ms"]) +
                                    float(export_debug.get("view_ms", 0.0)))
                                direct_stats["send_export_alias_w13_ms"] = (
                                    float(direct_stats["send_export_alias_w13_ms"]) +
                                    float(export_debug.get("alias_w13_ms", 0.0)))
                                direct_stats["send_export_alias_w2_ms"] = (
                                    float(direct_stats["send_export_alias_w2_ms"]) +
                                    float(export_debug.get("alias_w2_ms", 0.0)))
                            if (direct_verify_log and module.layer_idx == 0
                                    and len(active_ranks) <= 2):
                                source_slot = int(
                                    source_slot_by_expert.get(int(expert_id), -1))
                                target_slot = int(
                                    all_import_slot_by_expert.get(int(expert_id),
                                                                  -1))
                                logger.warning(
                                    "Mode1 direct NPU scalar send verify: "
                                    "rank=%s active_ranks=%s layer=%s "
                                    "source_rank=%s target_rank=%s expert=%s "
                                    "source_slot=%s target_slot=%s w13=%s w2=%s",
                                    self.rank,
                                    active_ranks,
                                    module.layer_idx,
                                    source_rank,
                                    target_rank,
                                    int(expert_id),
                                    source_slot,
                                    target_slot,
                                    _mode1_diag_tensor_row_stats(
                                        export_w13,
                                        0,
                                        sample_values=direct_verify_values),
                                    _mode1_diag_tensor_row_stats(
                                        export_w2,
                                        0,
                                        sample_values=direct_verify_values),
                                )
                            slow_export_threshold_ms = float(os.environ.get(
                                "VLLM_ASCEND_MODE1_DIAG_SLOW_EXPORT_LOG_MS",
                                "1000"))
                            if (len(active_ranks) == 2
                                    and export_ms >= slow_export_threshold_ms):
                                logger.info(
                                    "Elastic shrink direct NPU slow export: "
                                    "rank=%s active_ranks=%s layer=%s "
                                    "source_rank=%s target_rank=%s expert_id=%s "
                                    "export_ms=%.2f debug=%s",
                                    self.rank, active_ranks, module.layer_idx,
                                    source_rank, target_rank, expert_id,
                                    export_ms, export_debug)
                            send_post_start_t = time.perf_counter()
                            send_w13 = torch.distributed.isend(
                                export_w13,
                                dst=target_rank,
                                group=source_device_group)
                            send_w2 = torch.distributed.isend(
                                export_w2,
                                dst=target_rank,
                                group=source_device_group)
                            direct_stats["send_post_ms"] = (
                                float(direct_stats["send_post_ms"]) +
                                (time.perf_counter() - send_post_start_t) *
                                1000.0)
                            send_wait_start_t = time.perf_counter()
                            send_w13.wait()
                            send_w2.wait()
                            direct_stats["send_wait_ms"] = (
                                float(direct_stats["send_wait_ms"]) +
                                (time.perf_counter() - send_wait_start_t) *
                                1000.0)
                            direct_stats["transfer_pairs"] = (
                                float(direct_stats["transfer_pairs"]) + 1.0)
                            continue
                        if ((not participate_only)
                                and current_rank == target_rank
                                and expert_id in local_needed_cpu_import_ids):
                            recv_alias_start_t = time.perf_counter()
                            if direct_fill_preallocated_loaded:
                                target_slot = local_import_slot_by_expert[expert_id]
                                recv_buf_fn = getattr(
                                    module,
                                    "get_lossless_expert_npu_slot_recv_buffers",
                                    None)
                                if callable(recv_buf_fn):
                                    recv_target_w13, recv_target_w2 = recv_buf_fn(
                                        target_slot)
                                else:
                                    recv_target_w13 = module.w13_weight[
                                        target_slot:target_slot + 1]
                                    recv_target_w2 = module.w2_weight[
                                        target_slot:target_slot + 1]
                            else:
                                recv_target_w13 = recv_w13
                                recv_target_w2 = recv_w2
                            direct_stats["recv_alias_ms"] = (
                                float(direct_stats["recv_alias_ms"]) +
                                (time.perf_counter() - recv_alias_start_t) *
                                1000.0)
                            recv_post_start_t = time.perf_counter()
                            recv_req_w13 = torch.distributed.irecv(
                                recv_target_w13,
                                src=source_rank,
                                group=source_device_group)
                            recv_req_w2 = torch.distributed.irecv(
                                recv_target_w2,
                                src=source_rank,
                                group=source_device_group)
                            direct_stats["recv_post_ms"] = (
                                float(direct_stats["recv_post_ms"]) +
                                (time.perf_counter() - recv_post_start_t) *
                                1000.0)
                            recv_wait_start_t = time.perf_counter()
                            recv_req_w13.wait()
                            recv_req_w2.wait()
                            direct_stats["recv_wait_ms"] = (
                                float(direct_stats["recv_wait_ms"]) +
                                (time.perf_counter() - recv_wait_start_t) *
                                1000.0)
                            if not direct_fill_preallocated_loaded:
                                assert recv_w13 is not None and recv_w2 is not None
                                local_cpu_import_weights[expert_id] = (
                                    recv_w13[0].detach().cpu(),
                                    recv_w2[0].detach().cpu(),
                                )
                            else:
                                local_direct_import_slots[expert_id] = target_slot
                                if (direct_verify_log and module.layer_idx == 0
                                        and len(active_ranks) <= 2):
                                    logger.warning(
                                        "Mode1 direct NPU scalar recv verify: "
                                        "rank=%s active_ranks=%s layer=%s "
                                        "source_rank=%s target_rank=%s expert=%s "
                                        "target_slot=%s w13_slot=%s w2_slot=%s",
                                        self.rank,
                                        active_ranks,
                                        module.layer_idx,
                                        source_rank,
                                        target_rank,
                                        int(expert_id),
                                        int(target_slot),
                                        _mode1_diag_tensor_row_stats(
                                            module.w13_weight,
                                            int(target_slot),
                                            sample_values=direct_verify_values),
                                        _mode1_diag_tensor_row_stats(
                                            module.w2_weight,
                                            int(target_slot),
                                            sample_values=direct_verify_values),
                                    )
            if (not participate_only and my_active_idx is not None
                    and module.layer_idx == 0 and local_needed_cpu_import_ids):
                if (direct_verify_log and _env_flag(
                        "VLLM_ASCEND_MODE1_DIRECT_NPU_IMPORT_POSTCHECK_LOG",
                        "1")):
                    try:
                        loaded_map_values: list[int] = []
                        expert_map_values: list[int] = []
                        log2phy_values: list[int] = []
                        loaded_map = getattr(module, "loaded_expert_map", None)
                        expert_map = getattr(module, "expert_map", None)
                        log2phy = getattr(module, "log2phy", None)
                        if torch.is_tensor(loaded_map):
                            loaded_map_values = [
                                int(v)
                                for v in loaded_map.detach().cpu().tolist()
                            ]
                        if torch.is_tensor(expert_map):
                            expert_map_values = [
                                int(v)
                                for v in expert_map.detach().cpu().tolist()
                            ]
                        if torch.is_tensor(log2phy):
                            log2phy_values = [
                                int(v) for v in log2phy.detach().cpu().tolist()
                            ]
                        imported_experts = sorted(
                            int(expert_id)
                            for expert_id in local_needed_cpu_import_ids
                            if int(expert_id) in local_direct_import_slots)
                        sample_limit = max(
                            1,
                            int(
                                os.environ.get(
                                    "VLLM_ASCEND_MODE1_DIRECT_NPU_IMPORT_POSTCHECK_EXPERTS",
                                    "16")))
                        local_num_hint = max(
                            1, int(getattr(module, "local_num_experts", 1)))
                        rows: list[dict[str, object]] = []
                        slot_mismatches: list[dict[str, int]] = []
                        zero_like_w13 = 0
                        zero_like_w2 = 0
                        for expert_id in imported_experts[:sample_limit]:
                            target_slot = int(
                                local_direct_import_slots.get(expert_id, -1))
                            expected_slot = int(
                                local_import_slot_by_expert.get(expert_id, -1))
                            loaded_slot = (
                                loaded_map_values[expert_id]
                                if expert_id < len(loaded_map_values) else None)
                            active_slot = (
                                expert_map_values[expert_id]
                                if expert_id < len(expert_map_values) else None)
                            log2phy_slot = (
                                log2phy_values[expert_id]
                                if expert_id < len(log2phy_values) else None)
                            w13_stats = _mode1_diag_tensor_row_stats(
                                module.w13_weight,
                                target_slot,
                                sample_values=direct_verify_values)
                            w2_stats = _mode1_diag_tensor_row_stats(
                                module.w2_weight,
                                target_slot,
                                sample_values=direct_verify_values)
                            if bool(w13_stats.get("zero_like", False)):
                                zero_like_w13 += 1
                            if bool(w2_stats.get("zero_like", False)):
                                zero_like_w2 += 1
                            if target_slot != expected_slot:
                                slot_mismatches.append({
                                    "expert": expert_id,
                                    "target_slot": target_slot,
                                    "expected_slot": expected_slot,
                                })
                            rows.append({
                                "expert": expert_id,
                                "primary_owner_rank_hint": (
                                    expert_id // local_num_hint),
                                "primary_owner_slot_hint": (
                                    expert_id % local_num_hint),
                                "source_rank": int(
                                    cpu_import_source_rank.get(expert_id, -1)),
                                "target_rank": int(
                                    cpu_import_target_rank.get(expert_id, -1)),
                                "target_slot": target_slot,
                                "expected_slot": expected_slot,
                                "loaded_map_slot": loaded_slot,
                                "active_map_slot": active_slot,
                                "log2phy": log2phy_slot,
                                "w13_slot": w13_stats,
                                "w2_slot": w2_stats,
                            })
                        logger.warning(
                            "Mode1 direct NPU import postcheck: rank=%s "
                            "active_ranks=%s layer=%s imported=%s "
                            "needed=%s direct_slots=%s loaded_capacity=%s "
                            "target_owned_local=%s zero_like_w13=%s "
                            "zero_like_w2=%s slot_mismatches=%s rows=%s",
                            self.rank,
                            active_ranks,
                            module.layer_idx,
                            len(imported_experts),
                            len(local_needed_cpu_import_ids),
                            len(local_direct_import_slots),
                            int(getattr(module, "loaded_weight_capacity", -1)),
                            target_owned_local_expert_count,
                            zero_like_w13,
                            zero_like_w2,
                            slot_mismatches[:8],
                            rows,
                        )
                    except Exception:
                        logger.exception(
                            "Mode1 direct NPU import postcheck failed: rank=%s "
                            "active_ranks=%s layer=%s",
                            self.rank,
                            active_ranks,
                            module.layer_idx,
                        )
            direct_stats["total_ms"] = (
                time.perf_counter() - import_start) * 1000.0
            return (local_cpu_import_weights, local_direct_import_slots,
                    direct_stats)

        if (requires_redundant_direct_npu and cpu_import_source_rank
                and not disable_direct_npu_import):
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

        return (local_cpu_import_weights, local_direct_import_slots, {
            "transfer_mode": "object_broadcast",
        })

    def _preload_lossless_shrink_import_weights(self, active_ranks: list[int],
                                                world_group) -> None:
        self._lossless_preloaded_cpu_import_weights = {}
        self._lossless_preloaded_direct_import_slots = {}
        model_runner = getattr(self, "model_runner", None)
        model = getattr(model_runner, "model", None) if model_runner else None
        if model is None:
            return

        lossless_shrink_payload = getattr(self, "_lossless_shrink_payload", {})
        preload_module_count = 0
        preload_filter_cpu_payload_ms = 0.0
        preload_stream_import_ms = 0.0
        preload_store_ms = 0.0
        if (len(active_ranks) == 2 and _env_flag(
                "VLLM_ASCEND_MODE1_DIAG_PRELOAD_ENTRY_SYNC", "0")):
            preload_sync_start_t = time.perf_counter()
            torch.npu.synchronize()
            preload_sync_ms = (
                time.perf_counter() - preload_sync_start_t) * 1000.0
            logger.info(
                "Elastic shrink preload entry sync: rank=%s active_ranks=%s "
                "sync_ms=%.2f",
                self.rank, active_ranks, preload_sync_ms)
        for module in model.modules():
            if not _is_ascend_fused_moe_module(module):
                continue
            use_lossless_mode = (
                envs_ascend.VLLM_ASCEND_ELASTIC_MOE_MODE == "lossless"
                and getattr(module, "elastic_moe_mode", "lossy") == "lossless")
            if not use_lossless_mode:
                continue
            preload_module_count += 1
            payload = lossless_shrink_payload.get(module.layer_idx)
            if payload is None:
                raise RuntimeError(
                    f"Missing lossless shrink payload for layer={module.layer_idx}.")
            if int(getattr(module, "elastic_execution_mode", 0)) == 4:
                remote_sources = payload.get("cpu_import_source_rank", {})
                if module.layer_idx == 0:
                    logger.info(
                        "Mode4 remote-NPU cache preload skipped: rank=%s active_ranks=%s remote_sources=%s",
                        self.rank,
                        active_ranks,
                        len(remote_sources),
                    )
                self._lossless_preloaded_cpu_import_weights[module.layer_idx] = {}
                self._lossless_preloaded_direct_import_slots[module.layer_idx] = {}
                continue
            filter_cpu_payload_start_t = time.perf_counter()
            cpu_payload = (
                _mode5_filter_cpu_payload(payload)
                if int(getattr(module, "elastic_execution_mode", 0)) == 5
                else payload)
            preload_filter_cpu_payload_ms += (
                time.perf_counter() - filter_cpu_payload_start_t) * 1000.0
            if (int(getattr(module, "elastic_execution_mode", 0)) == 5
                    and module.layer_idx == 0):
                logger.info(
                    "Mode5 shrink preload CPU-shadow plan: rank=%s active_ranks=%s total_missing=%s remote_npu=%s cpu_shadow_import=%s",
                    self.rank,
                    active_ranks,
                    len(payload.get("cpu_import_source_rank", {})),
                    len(payload.get("mode5_remote_experts", [])),
                    len(cpu_payload.get("cpu_import_source_rank", {})),
                )
            stream_import_trace = _env_flag(
                "VLLM_ASCEND_MODE1_DIAG_PRELOAD_LAYER_TIMING", "0")
            if stream_import_trace and len(active_ranks) == 2:
                logger.info(
                    "Elastic shrink stream import layer enter: rank=%s active_ranks=%s layer=%s",
                    self.rank, active_ranks, module.layer_idx)
            stream_import_start_t = time.perf_counter()
            cpu_imports, direct_import_slots, stream_import_stats = (
                self._stream_lossless_layer_cpu_import_weights(
                    module, cpu_payload, active_ranks, world_group))
            stream_import_ms = (
                time.perf_counter() - stream_import_start_t) * 1000.0
            preload_stream_import_ms += stream_import_ms
            slow_layer_threshold_ms = float(os.environ.get(
                "VLLM_ASCEND_MODE1_DIAG_SLOW_PRELOAD_LAYER_LOG_MS",
                "1000"))
            log_first_preload_layer = _env_flag(
                "VLLM_ASCEND_MODE1_LOG_FIRST_PRELOAD_LAYER_TIMING", "0")
            log_stream_import_layer = (
                stream_import_trace
                or (preload_module_count == 1 and log_first_preload_layer)
                or (len(active_ranks) == 2
                    and stream_import_ms >= slow_layer_threshold_ms))
            if log_stream_import_layer:
                logger.info(
                    "Elastic shrink stream import breakdown: rank=%s active_ranks=%s layer=%s mode=%s transfer_pairs=%s remote_experts=%s local_only_experts=%s chunks=%s local_copy_export_ms=%.2f slot_cache_refresh_ms=%.2f dry_run_export_ms=%.2f send_export_ms=%.2f send_export_source_select_ms=%.2f send_export_runtime_slot_map_ms=%.2f send_export_runtime_slot_cpu_hits=%s send_export_runtime_slot_npu_items=%s send_export_slot_cpu_hits=%s send_export_slot_npu_items=%s send_export_contiguous_chunks=%s send_export_index_select_chunks=%s slot_map_expert_cpu_ready=%s slot_map_loaded_cpu_ready=%s slot_map_diag_expert_cpu_ready=%s slot_map_diag_loaded_cpu_ready=%s send_export_map_ms=%.2f send_export_view_ms=%.2f send_export_alias_w13_ms=%.2f send_export_alias_w2_ms=%.2f send_pack_copy_ms=%.2f send_to_device_ms=%.2f send_post_ms=%.2f send_wait_ms=%.2f recv_alias_ms=%.2f recv_post_ms=%.2f recv_wait_ms=%.2f recv_to_cpu_ms=%.2f recv_store_ms=%.2f recv_direct_slot_chunks=%s recv_temp_buffer_chunks=%s total_ms=%.2f wall_ms=%.2f",
                    self.rank,
                    active_ranks,
                    module.layer_idx,
                    stream_import_stats.get("transfer_mode", "unknown"),
                    int(stream_import_stats.get("transfer_pairs", 0.0)),
                    int(stream_import_stats.get("remote_experts", 0.0)),
                    int(stream_import_stats.get("local_only_experts", 0.0)),
                    int(stream_import_stats.get("chunks", 0.0)),
                    float(stream_import_stats.get("local_copy_export_ms", 0.0)),
                    float(
                        stream_import_stats.get("slot_cache_refresh_ms", 0.0)),
                    float(stream_import_stats.get("dry_run_export_ms", 0.0)),
                    float(stream_import_stats.get("send_export_ms", 0.0)),
                    float(
                        stream_import_stats.get(
                            "send_export_source_select_ms", 0.0)),
                    float(
                        stream_import_stats.get(
                            "send_export_runtime_slot_map_ms", 0.0)),
                    int(
                        stream_import_stats.get(
                            "send_export_runtime_slot_cpu_hits", 0.0)),
                    int(
                        stream_import_stats.get(
                            "send_export_runtime_slot_npu_items", 0.0)),
                    int(
                        stream_import_stats.get("send_export_slot_cpu_hits",
                                                0.0)),
                    int(
                        stream_import_stats.get("send_export_slot_npu_items",
                                                0.0)),
                    int(
                        stream_import_stats.get(
                            "send_export_contiguous_chunks", 0.0)),
                    int(
                        stream_import_stats.get(
                            "send_export_index_select_chunks", 0.0)),
                    int(
                        stream_import_stats.get("slot_map_expert_cpu_ready",
                                                0.0)),
                    int(
                        stream_import_stats.get("slot_map_loaded_cpu_ready",
                                                0.0)),
                    int(
                        stream_import_stats.get(
                            "slot_map_diag_expert_cpu_ready", 0.0)),
                    int(
                        stream_import_stats.get(
                            "slot_map_diag_loaded_cpu_ready", 0.0)),
                    float(stream_import_stats.get("send_export_map_ms", 0.0)),
                    float(stream_import_stats.get("send_export_view_ms", 0.0)),
                    float(
                        stream_import_stats.get("send_export_alias_w13_ms",
                                                0.0)),
                    float(
                        stream_import_stats.get("send_export_alias_w2_ms",
                                                0.0)),
                    float(stream_import_stats.get("send_pack_copy_ms", 0.0)),
                    float(stream_import_stats.get("send_to_device_ms", 0.0)),
                    float(stream_import_stats.get("send_post_ms", 0.0)),
                    float(stream_import_stats.get("send_wait_ms", 0.0)),
                    float(stream_import_stats.get("recv_alias_ms", 0.0)),
                    float(stream_import_stats.get("recv_post_ms", 0.0)),
                    float(stream_import_stats.get("recv_wait_ms", 0.0)),
                    float(stream_import_stats.get("recv_to_cpu_ms", 0.0)),
                    float(stream_import_stats.get("recv_store_ms", 0.0)),
                    int(
                        stream_import_stats.get("recv_direct_slot_chunks",
                                                0.0)),
                    int(
                        stream_import_stats.get("recv_temp_buffer_chunks",
                                                0.0)),
                    float(stream_import_stats.get("total_ms", 0.0)),
                    stream_import_ms,
                )
            if (getattr(module, "elastic_execution_mode", 0) == 4
                    and cpu_imports):
                raise RuntimeError(
                    "Mode4 remote-NPU path must not stage expert weights "
                    f"through CPU: layer={module.layer_idx} "
                    f"cpu_imports={len(cpu_imports)} "
                    f"direct_import_slots={len(direct_import_slots)}")
            if (getattr(module, "elastic_execution_mode", 0) == 4
                    and module.layer_idx == 0):
                logger.info(
                    "Mode4 shrink preload summary: rank=%s active_ranks=%s "
                    "direct_remote_slots=%s cpu_imports=%s",
                    self.rank,
                    active_ranks,
                    len(direct_import_slots),
                    len(cpu_imports),
                )
            store_preload_start_t = time.perf_counter()
            self._lossless_preloaded_cpu_import_weights[module.layer_idx] = (
                cpu_imports)
            self._lossless_preloaded_direct_import_slots[module.layer_idx] = (
                direct_import_slots)
            preload_store_ms += (
                time.perf_counter() - store_preload_start_t) * 1000.0

        if preload_module_count > 0:
            logger.info(
                "Elastic shrink preload breakdown: rank=%s active_ranks=%s modules=%s filter_cpu_payload_ms=%.2f stream_import_ms=%.2f store_ms=%.2f total_ms=%.2f",
                self.rank,
                active_ranks,
                preload_module_count,
                preload_filter_cpu_payload_ms,
                preload_stream_import_ms,
                preload_store_ms,
                preload_filter_cpu_payload_ms + preload_stream_import_ms +
                preload_store_ms)

    def _commit_preloaded_direct_slots_to_loaded_maps(
            self, active_ranks: list[int], reason: str) -> dict[str, float]:
        stats = {
            "layers": 0.0,
            "direct_slots": 0.0,
            "map_updates": 0.0,
        }
        if not active_ranks or self.rank not in set(int(r) for r in active_ranks):
            return stats
        model_runner = getattr(self, "model_runner", None)
        model = getattr(model_runner, "model", None) if model_runner else None
        if model is None:
            return stats
        direct_by_layer = getattr(self, "_lossless_preloaded_direct_import_slots",
                                  {})
        if not direct_by_layer:
            return stats
        for module in model.modules():
            if not _is_ascend_fused_moe_module(module):
                continue
            if getattr(module, "elastic_moe_mode", "lossy") != "lossless":
                continue
            layer_slots = direct_by_layer.get(getattr(module, "layer_idx", -1), {})
            if not layer_slots:
                continue
            mark_fn = getattr(module, "mark_lossless_loaded_slots_resident",
                              None)
            if not callable(mark_fn):
                continue
            updates = mark_fn({
                int(expert_id): int(slot)
                for expert_id, slot in layer_slots.items()
            }, reason=reason)
            if (_env_flag("VLLM_ASCEND_MODE1_DIRECT_NPU_IMPORT_VERIFY_LOG",
                          "0") and int(getattr(module, "layer_idx", -1)) == 0):
                sample_values = max(
                    1,
                    int(
                        os.environ.get(
                            "VLLM_ASCEND_MODE1_DIRECT_NPU_IMPORT_VERIFY_VALUES",
                            "8")))
                sample_items = sorted(
                    (int(expert_id), int(slot))
                    for expert_id, slot in layer_slots.items())[:8]
                loaded_map = getattr(module, "loaded_expert_map", None)
                loaded_map_head = []
                try:
                    if loaded_map is not None:
                        loaded_map_head = [
                            int(v)
                            for v in loaded_map.detach().cpu().tolist()[:32]
                        ]
                except Exception:
                    loaded_map_head = ["copy_failed"]
                samples = []
                for expert_id, slot in sample_items:
                    samples.append({
                        "expert": expert_id,
                        "slot": slot,
                        "loaded_map_slot": (
                            int(loaded_map[expert_id].item())
                            if loaded_map is not None
                            and expert_id < int(loaded_map.numel()) else None),
                        "w13_slot": _mode1_diag_tensor_row_stats(
                            getattr(module, "w13_weight", None),
                            slot,
                            sample_values=sample_values),
                        "w2_slot": _mode1_diag_tensor_row_stats(
                            getattr(module, "w2_weight", None),
                            slot,
                            sample_values=sample_values),
                    })
                logger.warning(
                    "Mode1 direct NPU commit verify: rank=%s active_ranks=%s "
                    "layer=%s reason=%s direct_slots=%s updates=%s "
                    "loaded_map_head=%s samples=%s",
                    self.rank,
                    active_ranks,
                    int(getattr(module, "layer_idx", -1)),
                    reason,
                    len(layer_slots),
                    updates,
                    loaded_map_head,
                    samples,
                )
            stats["layers"] += 1.0
            stats["direct_slots"] += float(len(layer_slots))
            stats["map_updates"] += float(updates)
        return stats

    def _prefill_mode1_planned_expert_slots_for_active_ranks(
            self, active_ranks: list[int], world_group,
            reason: str) -> dict[str, float]:
        stats = {
            "enabled": 0.0,
            "active_size": float(len(active_ranks)),
            "payload_ms": 0.0,
            "preload_ms": 0.0,
            "commit_layers": 0.0,
            "commit_slots": 0.0,
            "commit_updates": 0.0,
            "total_ms": 0.0,
        }
        if not self._mode1_prefill_planned_expert_slots_enabled():
            return stats
        if not active_ranks:
            return stats
        world_size = int(torch.distributed.get_world_size()
                         if torch.distributed.is_initialized() else 1)
        if len(active_ranks) >= world_size:
            return stats
        previous_active = self._get_previous_active_ranks_for_shrink(world_size)
        if len(previous_active) != 2 * len(active_ranks):
            return stats
        if not set(int(rank) for rank in active_ranks).issubset(
                set(int(rank) for rank in previous_active)):
            return stats

        old_payload = getattr(self, "_lossless_shrink_payload", {})
        old_cpu_imports = getattr(self, "_lossless_preloaded_cpu_import_weights",
                                  {})
        old_direct_slots = getattr(self, "_lossless_preloaded_direct_import_slots",
                                   {})
        old_current_active = getattr(self, "_elastic_current_active_ranks", None)
        start_t = time.perf_counter()
        try:
            stats["enabled"] = 1.0
            payload_start_t = time.perf_counter()
            self._prepare_lossless_shrink_payload(list(active_ranks), world_group)
            stats["payload_ms"] = (
                time.perf_counter() - payload_start_t) * 1000.0
            preload_start_t = time.perf_counter()
            self._preload_lossless_shrink_import_weights(list(active_ranks),
                                                         world_group)
            stats["preload_ms"] = (
                time.perf_counter() - preload_start_t) * 1000.0
            commit_stats = self._commit_preloaded_direct_slots_to_loaded_maps(
                list(active_ranks), reason=reason)
            stats["commit_layers"] = float(commit_stats.get("layers", 0.0))
            stats["commit_slots"] = float(commit_stats.get("direct_slots", 0.0))
            stats["commit_updates"] = float(commit_stats.get("map_updates", 0.0))
            if torch.npu.is_available():
                torch.npu.synchronize()
            stats["total_ms"] = (time.perf_counter() - start_t) * 1000.0
            if (self.rank == int(active_ranks[0])
                    or self.rank == int(previous_active[0])):
                logger.info(
                    "Mode1 planned expert slot prefill complete: "
                    "rank=%s active_ranks=%s previous_active=%s reason=%s "
                    "payload_ms=%.2f preload_ms=%.2f commit_layers=%.0f "
                    "commit_slots=%.0f commit_updates=%.0f total_ms=%.2f",
                    self.rank,
                    list(active_ranks),
                    list(previous_active),
                    reason,
                    stats["payload_ms"],
                    stats["preload_ms"],
                    stats["commit_layers"],
                    stats["commit_slots"],
                    stats["commit_updates"],
                    stats["total_ms"],
                )
        finally:
            self._lossless_shrink_payload = old_payload
            self._lossless_preloaded_cpu_import_weights = old_cpu_imports
            self._lossless_preloaded_direct_import_slots = old_direct_slots
            self._elastic_current_active_ranks = old_current_active
        return stats

    def prefill_mode1_planned_expert_slots_for_kv_resize(
            self, target_floor: int) -> bool:
        """Prefill real expert weights into existing mode=1 redundant slots.

        This is only a loaded-slot residency optimization for planned topology:
        it does not allocate more expert rows.  The goal is to make the next
        planned shrink see the needed experts in loaded_expert_map and avoid
        scheduling another direct-NPU import at the shrink point.
        """
        if not self._mode1_prefill_planned_expert_slots_enabled():
            return False
        try:
            target_floor = int(target_floor)
        except (TypeError, ValueError):
            return False
        if not torch.distributed.is_initialized():
            return False
        world_size = int(torch.distributed.get_world_size())
        if target_floor <= 0 or target_floor >= world_size:
            return False
        planned_signature = self._mode1_planned_floor_group_signature()
        if not planned_signature:
            return False

        # At a step boundary the runtime has been restored to full world.  The
        # immediate next shrink is therefore the largest planned subgroup, e.g.
        # 16 -> 8 for a floor4 plan.  Prefilling that first hop is safe without
        # temporarily installing smaller source groups.
        first_stage = next(
            (tuple(int(rank) for rank in ranks)
             for ranks in planned_signature if len(ranks) < world_size),
            (),
        )
        if not first_stage:
            return False
        if target_floor > len(first_stage):
            return False

        world_group = get_world_group()
        stats = self._prefill_mode1_planned_expert_slots_for_active_ranks(
            list(first_stage),
            world_group,
            reason=f"kv_resize_target_floor={target_floor}",
        )
        changed = bool(
            stats.get("commit_updates", 0.0) > 0.0
            or stats.get("commit_slots", 0.0) > 0.0)
        logger.info(
            "Mode1 planned expert slot prefill for KV resize: rank=%s "
            "target_floor=%s first_stage=%s changed=%s payload_ms=%.2f "
            "preload_ms=%.2f commit_layers=%.0f commit_slots=%.0f "
            "commit_updates=%.0f total_ms=%.2f",
            self.rank,
            target_floor,
            list(first_stage),
            int(changed),
            float(stats.get("payload_ms", 0.0)),
            float(stats.get("preload_ms", 0.0)),
            float(stats.get("commit_layers", 0.0)),
            float(stats.get("commit_slots", 0.0)),
            float(stats.get("commit_updates", 0.0)),
            float(stats.get("total_ms", 0.0)),
        )
        gc.collect()
        try:
            torch.npu.empty_cache()
            torch.npu.synchronize()
        except Exception:
            logger.exception(
                "Failed to empty/sync NPU cache after planned expert prefill")
        return changed

    def _destroy_group_if_present(self, state_module, attr_name: str) -> None:
        group = getattr(state_module, attr_name, None)
        if group is not None:
            group.destroy()
            self._detach_destroyed_group_references(group)
            setattr(state_module, attr_name, None)

    def _mode1_source_cpu_p2p_barrier(self, source_ranks: list[int],
                                      world_group, tag: int) -> None:
        """Barrier previous active ranks without depending on stale DP groups.

        Mode=1 floor shrink may intentionally destroy the previous floor's
        cached DP/EP/MC2 groups before active ranks refresh per-layer Moe
        dispatchers. A collective barrier on that previous DP CPU group is not
        safe after the drop. Use the stable full-world CPU group for a tiny P2P
        fan-in/fan-out barrier among only the source ranks.
        """
        if not torch.distributed.is_initialized():
            return
        ranks = sorted(set(int(rank) for rank in source_ranks))
        if len(ranks) <= 1:
            return
        current_rank = int(torch.distributed.get_rank())
        if current_rank not in ranks:
            return

        cpu_group = getattr(world_group, "cpu_group", None)
        if cpu_group is None:
            return

        root_rank = ranks[0]
        token = torch.ones(1, dtype=torch.int32, device="cpu")
        ack = torch.empty(1, dtype=torch.int32, device="cpu")
        if current_rank == root_rank:
            for peer_rank in ranks[1:]:
                torch.distributed.recv(token,
                                       src=peer_rank,
                                       group=cpu_group,
                                       tag=tag)
            for peer_rank in ranks[1:]:
                torch.distributed.send(token,
                                       dst=peer_rank,
                                       group=cpu_group,
                                       tag=tag + 1)
        else:
            torch.distributed.send(token,
                                   dst=root_rank,
                                   group=cpu_group,
                                   tag=tag)
            torch.distributed.recv(ack,
                                   src=root_rank,
                                   group=cpu_group,
                                   tag=tag + 1)


    def _custom_mode1_worker_memory_diag_enabled(self) -> bool:
        if envs_ascend.VLLM_ASCEND_ELASTIC_EXECUTION_MODE != 1:
            return False
        return (_env_flag("VLLM_ASCEND_MODE1_PARITY_MC2_MEM_LOG", "0")
                or _env_flag("VLLM_ASCEND_CUSTOM_MODE1_KV_DIAG", "0"))

    def _log_custom_mode1_worker_memory(self, tag: str,
                                        extra: str = "") -> None:
        if not self._custom_mode1_worker_memory_diag_enabled():
            return
        if not torch.npu.is_available():
            return
        try:
            free_bytes, total_bytes = torch.npu.mem_get_info()
            stats = torch_npu.npu.memory_stats()
            torch_current = int(stats.get("allocated_bytes.all.current", 0))
            torch_reserved = int(stats.get("reserved_bytes.all.current", 0))
            total_allocated = int(total_bytes - free_bytes)
            non_torch = max(total_allocated - torch_current, 0)
            logger.info(
                "Mode1 parity worker memory: tag=%s rank=%s "
                "free_bytes=%s total_bytes=%s torch_current=%s "
                "torch_reserved=%s non_torch=%s total_allocated=%s%s",
                tag,
                self.rank,
                free_bytes,
                total_bytes,
                torch_current,
                torch_reserved,
                non_torch,
                total_allocated,
                f" {extra}" if extra else "",
            )
        except Exception as exc:
            logger.warning(
                "Mode1 parity worker memory logging failed at %s: %s",
                tag,
                exc,
            )

    def _mode1_fullworld_restore_diag_enabled(self) -> bool:
        if envs_ascend.VLLM_ASCEND_ELASTIC_EXECUTION_MODE != 1:
            return False
        return (_env_flag("VLLM_ASCEND_MODE1_FULLWORLD_RESTORE_DIAG", "0")
                or _env_flag("VLLM_ASCEND_MODE1_PARITY_MC2_MEM_LOG", "0")
                or _env_flag("VLLM_ASCEND_CUSTOM_MODE1_KV_DIAG", "0"))

    def _log_mode1_fullworld_restore_diag(self, tag: str,
                                          extra: str = "") -> None:
        if not self._mode1_fullworld_restore_diag_enabled():
            return
        active_ranks = tuple(
            int(rank) for rank in getattr(self, "_elastic_current_active_ranks",
                                          ()) or ())
        target_ranks = tuple(
            int(rank) for rank in getattr(self, "_elastic_rebuild_target_ranks",
                                          ()) or ())
        restore_seq = int(getattr(self, "_mode1_fullworld_restore_seq", 0))
        logger.info(
            "Mode1 fullworld restore diag: tag=%s rank=%s restore_seq=%s "
            "active_ranks=%s target_ranks=%s elastic_detached=%s%s",
            tag,
            self.rank,
            restore_seq,
            active_ranks,
            target_ranks,
            bool(getattr(self, "elastic_parallel_detached", False)),
            f" {extra}" if extra else "",
        )
        self.mode1_log_comm_cache_state(f"fullworld_restore:{tag}")

    def mode1_dump_pre_generate_moe_state(self, tag: str = "") -> None:
        """Dump compact MoE state before generation for corruption triage.

        Shape-only checks can pass even when the active prefix rows contain
        stale or wrong expert weights. This diagnostic samples the active slot,
        canonical loaded slot and runtime log2phy mapping immediately before
        vLLM enters generate(), which separates reload/mapping corruption from
        decode-time dispatcher corruption.
        """
        if not _env_flag("VLLM_ASCEND_MODE1_PRE_GENERATE_STATE_DUMP", "0"):
            return
        if int(getattr(envs_ascend, "VLLM_ASCEND_ELASTIC_EXECUTION_MODE", 0)) != 1:
            return
        model_runner = getattr(self, "model_runner", None)
        model = getattr(model_runner, "model", None) if model_runner else None
        if model is None:
            logger.warning(
                "Mode1 pre-generate MoE state skipped: rank=%s tag=%s reason=no_model",
                self.rank,
                tag,
            )
            return

        layer_filter = os.getenv("VLLM_ASCEND_MODE1_PRE_GENERATE_STATE_LAYERS",
                                 "0")
        allowed_layers: Optional[set[int]] = None
        if layer_filter.strip().lower() not in ("", "all", "*"):
            try:
                allowed_layers = {
                    int(item.strip())
                    for item in layer_filter.split(",") if item.strip()
                }
            except ValueError:
                allowed_layers = {0}
        sample_experts = max(
            1,
            int(
                os.getenv("VLLM_ASCEND_MODE1_PRE_GENERATE_STATE_EXPERTS",
                          "8")))
        sample_values = max(
            1,
            int(
                os.getenv("VLLM_ASCEND_MODE1_PRE_GENERATE_STATE_VALUES",
                          "8")))
        max_layers = max(
            1,
            int(os.getenv("VLLM_ASCEND_MODE1_PRE_GENERATE_STATE_MAX_LAYERS",
                          "4")))

        def _to_int_list(tensor: object) -> list[int]:
            if tensor is None:
                return []
            try:
                return [int(v) for v in tensor.detach().cpu().tolist()]
            except Exception:
                return []

        dumped_layers = 0
        for module in model.modules():
            if not _is_ascend_fused_moe_module(module):
                continue
            if getattr(module, "elastic_moe_mode", "lossy") != "lossless":
                continue
            if int(getattr(module, "elastic_execution_mode", 0)) != 1:
                continue
            layer_idx = int(getattr(module, "layer_idx", -1))
            if allowed_layers is not None and layer_idx not in allowed_layers:
                continue
            if dumped_layers >= max_layers:
                break
            dumped_layers += 1

            validate = getattr(module,
                               "_validate_mode1_lossless_active_mapping",
                               None)
            if callable(validate):
                try:
                    validate(f"pre_generate:{tag}")
                except Exception:
                    logger.exception(
                        "Mode1 pre-generate mapping validation failed: rank=%s tag=%s layer=%s",
                        self.rank,
                        tag,
                        layer_idx,
                    )
                    if _env_flag(
                            "VLLM_ASCEND_MODE1_PRE_GENERATE_STATE_STRICT",
                            "1"):
                        raise

            expert_map = _to_int_list(getattr(module, "expert_map", None))
            loaded_map = _to_int_list(getattr(module, "loaded_expert_map", None))
            log2phy = _to_int_list(getattr(module, "log2phy", None))
            runtime_log2phy_tensor = getattr(module, "elastic_runtime_log2phy",
                                             None)
            runtime_log2phy = (_to_int_list(runtime_log2phy_tensor)
                               if runtime_log2phy_tensor is not None else [])
            active_experts = [
                (expert_id, slot)
                for expert_id, slot in enumerate(expert_map) if slot >= 0
            ][:sample_experts]
            w13_weight = getattr(module, "w13_weight", None)
            w2_weight = getattr(module, "w2_weight", None)
            runtime_w13 = getattr(module, "runtime_w13_weight", None)
            runtime_w2 = getattr(module, "runtime_w2_weight", None)
            dispatch_log2phy = _to_int_list(
                getattr(module, "lossless_runtime_dispatch_log2phy", None))
            runtime_w13_buffer = getattr(module, "runtime_w13_buffer", None)
            runtime_w2_buffer = getattr(module, "runtime_w2_buffer", None)
            preloaded_direct_slots = (
                getattr(self, "_lossless_preloaded_direct_import_slots", {})
                or {}).get(layer_idx, {})
            preloaded_cpu_imports = (
                getattr(self, "_lossless_preloaded_cpu_import_weights", {})
                or {}).get(layer_idx, {})

            def _storage_diag(tensor: object) -> dict[str, object]:
                if not torch.is_tensor(tensor):
                    return {"present": False}
                ptr, nbytes = _tensor_storage_key_and_nbytes(tensor)
                return {
                    "present": True,
                    "shape": tuple(getattr(tensor, "shape", ())),
                    "device": str(getattr(tensor, "device", "")),
                    "dtype": str(getattr(tensor, "dtype", "")),
                    "data_ptr": int(tensor.data_ptr()),
                    "storage_ptr": int(ptr),
                    "storage_nbytes": int(nbytes),
                }

            active_loaded_mismatches = 0
            active_runtime_mismatches = 0
            zero_like_w13_active = 0
            zero_like_w2_active = 0
            suspicious_active_experts: list[int] = []
            samples: list[dict[str, object]] = []
            local_num_hint = max(
                1, int(getattr(module, "local_num_experts", 1)))
            for expert_id, active_slot in active_experts:
                loaded_slot = (loaded_map[expert_id]
                               if expert_id < len(loaded_map) else -1)
                w13_active_loaded_diff = (
                    _mode1_diag_tensor_row_absdiff(
                        w13_weight, int(active_slot), w13_weight,
                        int(loaded_slot), sample_values=sample_values))
                w13_active_runtime_diff = (
                    _mode1_diag_tensor_row_absdiff(
                        w13_weight, int(active_slot), runtime_w13,
                        int(active_slot), sample_values=sample_values))
                w2_active_loaded_diff = (
                    _mode1_diag_tensor_row_absdiff(
                        w2_weight, int(active_slot), w2_weight,
                        int(loaded_slot), sample_values=sample_values))
                w2_active_runtime_diff = (
                    _mode1_diag_tensor_row_absdiff(
                        w2_weight, int(active_slot), runtime_w2,
                        int(active_slot), sample_values=sample_values))
                if (w13_active_loaded_diff is not None
                        and w13_active_loaded_diff > 1e-6) or (
                            w2_active_loaded_diff is not None
                            and w2_active_loaded_diff > 1e-6):
                    active_loaded_mismatches += 1
                if (w13_active_runtime_diff is not None
                        and w13_active_runtime_diff > 1e-6) or (
                            w2_active_runtime_diff is not None
                            and w2_active_runtime_diff > 1e-6):
                    active_runtime_mismatches += 1
                w13_active_stats = _mode1_diag_tensor_row_stats(
                    w13_weight, int(active_slot), sample_values=sample_values)
                w2_active_stats = _mode1_diag_tensor_row_stats(
                    w2_weight, int(active_slot), sample_values=sample_values)
                if bool(w13_active_stats.get("zero_like", False)):
                    zero_like_w13_active += 1
                    suspicious_active_experts.append(int(expert_id))
                if bool(w2_active_stats.get("zero_like", False)):
                    zero_like_w2_active += 1
                    if int(expert_id) not in suspicious_active_experts:
                        suspicious_active_experts.append(int(expert_id))
                samples.append({
                    "expert": int(expert_id),
                    "primary_owner_rank_hint": (
                        int(expert_id) // local_num_hint),
                    "primary_owner_slot_hint": (
                        int(expert_id) % local_num_hint),
                    "active_slot": int(active_slot),
                    "loaded_slot": int(loaded_slot),
                    "preloaded_direct_slot": (
                        int(preloaded_direct_slots[int(expert_id)])
                        if int(expert_id) in preloaded_direct_slots else None),
                    "preloaded_cpu_import": (
                        int(expert_id) in preloaded_cpu_imports),
                    "log2phy": (log2phy[expert_id]
                                if expert_id < len(log2phy) else None),
                    "runtime_log2phy": (runtime_log2phy[expert_id]
                                        if expert_id < len(runtime_log2phy)
                                        else None),
                    "dispatch_log2phy": (dispatch_log2phy[expert_id]
                                         if expert_id < len(dispatch_log2phy)
                                         else None),
                    "w13_active": w13_active_stats,
                    "w13_loaded": _mode1_diag_tensor_row_stats(
                        w13_weight, int(loaded_slot),
                        sample_values=sample_values),
                    "w13_runtime_active": _mode1_diag_tensor_row_stats(
                        runtime_w13, int(active_slot),
                        sample_values=sample_values),
                    "w13_active_loaded_absdiff": (
                        _mode1_diag_tensor_row_absdiff(
                            w13_weight, int(active_slot), w13_weight,
                            int(loaded_slot), sample_values=sample_values)),
                    "w13_active_runtime_absdiff": (
                        w13_active_runtime_diff),
                    "w2_active": w2_active_stats,
                    "w2_loaded": _mode1_diag_tensor_row_stats(
                        w2_weight, int(loaded_slot),
                        sample_values=sample_values),
                    "w2_runtime_active": _mode1_diag_tensor_row_stats(
                        runtime_w2, int(active_slot),
                        sample_values=sample_values),
                    "w2_active_loaded_absdiff": (
                        w2_active_loaded_diff),
                    "w2_active_runtime_absdiff": (
                        w2_active_runtime_diff),
                })

            logger.warning(
                "Mode1 pre-generate MoE state: rank=%s tag=%s layer=%s "
                "active=%s local_num=%s moe_local=%s loaded_local=%s "
                "capacity=%s w13_shape=%s w2_shape=%s runtime_w13_shape=%s "
                "runtime_w2_shape=%s native_parity_ready=%s direct_ok=%s "
                "lossless_loaded_offloaded=%s expert_map_head=%s "
                "loaded_map_head=%s log2phy_head=%s runtime_log2phy_head=%s "
                "dispatch_log2phy_head=%s active_loaded_mismatches=%s "
                "active_runtime_mismatches=%s zero_like_w13_active=%s "
                "zero_like_w2_active=%s suspicious_active_experts=%s "
                "storage=%s samples=%s",
                self.rank,
                tag,
                layer_idx,
                int(getattr(module, "active_local_num_experts", -1)),
                int(getattr(module, "local_num_experts", -1)),
                int(getattr(getattr(module, "moe_config", None),
                            "num_local_experts", -1)),
                int(getattr(module, "loaded_local_num_experts", -1)),
                int(getattr(module, "loaded_weight_capacity", -1)),
                tuple(getattr(getattr(module, "w13_weight", None), "shape", ())),
                tuple(getattr(getattr(module, "w2_weight", None), "shape", ())),
                tuple(
                    getattr(getattr(module, "runtime_w13_weight", None),
                            "shape", ())),
                tuple(
                    getattr(getattr(module, "runtime_w2_weight", None),
                            "shape", ())),
                bool(getattr(module, "lossless_mode1_native_parity_ready",
                             False)),
                bool(
                    getattr(module,
                            "_lossless_mode1_direct_preloaded_activation_ok",
                            False)),
                bool(getattr(module, "lossless_loaded_offloaded", False)),
                expert_map[:16],
                loaded_map[:16],
                log2phy[:16],
                runtime_log2phy[:16],
                dispatch_log2phy[:16],
                active_loaded_mismatches,
                active_runtime_mismatches,
                zero_like_w13_active,
                zero_like_w2_active,
                suspicious_active_experts[:16],
                {
                    "w13": _storage_diag(w13_weight),
                    "w2": _storage_diag(w2_weight),
                    "runtime_w13": _storage_diag(runtime_w13),
                    "runtime_w2": _storage_diag(runtime_w2),
                    "runtime_w13_buffer": _storage_diag(runtime_w13_buffer),
                    "runtime_w2_buffer": _storage_diag(runtime_w2_buffer),
                },
                samples,
            )

    def mode1_log_comm_cache_state(self, tag: str) -> None:
        """Log planned/natural communicator cache state and NPU memory."""
        if not torch.npu.is_available():
            return

        def _fmt_rank_tuple(ranks: tuple[int, ...]) -> str:
            return "[" + ",".join(str(int(rank)) for rank in ranks) + "]"

        def _fmt_cached_groups() -> str:
            cached = getattr(self, "_elastic_cached_parallel_groups", {}) or {}
            parts = []
            for attr_name, groups_by_ranks in sorted(cached.items()):
                ranks_list = sorted(
                    tuple(int(rank) for rank in ranks)
                    for ranks in groups_by_ranks.keys())
                if ranks_list:
                    parts.append(
                        f"{attr_name}:"
                        f"{','.join(_fmt_rank_tuple(r) for r in ranks_list)}")
            return ";".join(parts) if parts else "-"

        def _fmt_registry() -> str:
            registry = getattr(self, "_elastic_parallel_group_registry", {}) or {}
            parts = []
            for kind, groups_by_ranks in sorted(registry.items()):
                rank_parts = []
                for ranks, groups_by_id in sorted(
                        groups_by_ranks.items(),
                        key=lambda item: (len(item[0]), item[0])):
                    rank_tuple = tuple(int(rank) for rank in ranks)
                    rank_parts.append(
                        f"{_fmt_rank_tuple(rank_tuple)}x{len(groups_by_id)}")
                if rank_parts:
                    parts.append(f"{kind}:{','.join(rank_parts)}")
            return ";".join(parts) if parts else "-"

        def _fmt_seen() -> str:
            seen = getattr(
                self, "_elastic_seen_parallel_group_signatures", set()) or set()
            counts: dict[str, list[tuple[int, ...]]] = {}
            for kind, ranks in seen:
                counts.setdefault(str(kind), []).append(
                    tuple(int(rank) for rank in ranks))
            parts = []
            for kind, ranks_list in sorted(counts.items()):
                parts.append(
                    f"{kind}:"
                    f"{','.join(_fmt_rank_tuple(r) for r in sorted(ranks_list))}")
            return ";".join(parts) if parts else "-"

        try:
            topology_stats = get_moe_comm_method_topology_cache_stats()
            topology_desc = ";".join(
                ",".join(item.get("groups", []))
                + f"x{int(item.get('methods', 0))}"
                + ("*" if int(item.get("active", 0)) else "")
                for item in topology_stats.get("topologies", []))
            if not topology_desc:
                topology_desc = "-"
        except Exception:
            logger.exception(
                "Failed to collect mode1 MoE topology cache state: rank=%s "
                "tag=%s",
                self.rank,
                tag,
            )
            topology_stats = {}
            topology_desc = "error"

        try:
            free_bytes, total_bytes = torch.npu.mem_get_info()
            stats = torch_npu.npu.memory_stats()
            torch_current = int(stats.get("allocated_bytes.all.current", 0))
            torch_reserved = int(stats.get("reserved_bytes.all.current", 0))
            total_allocated = int(total_bytes - free_bytes)
            non_torch = max(total_allocated - torch_current, 0)
        except Exception:
            logger.exception(
                "Failed to collect mode1 comm cache memory state: rank=%s "
                "tag=%s",
                self.rank,
                tag,
            )
            free_bytes = total_bytes = torch_current = torch_reserved = 0
            total_allocated = non_torch = 0

        cached = getattr(self, "_elastic_cached_parallel_groups", {}) or {}
        cached_group_count = sum(
            len(groups_by_ranks) for groups_by_ranks in cached.values())
        registry = getattr(self, "_elastic_parallel_group_registry", {}) or {}
        registry_group_count = sum(
            len(groups_by_id)
            for groups_by_ranks in registry.values()
            for groups_by_id in groups_by_ranks.values())
        seen_count = len(
            getattr(self, "_elastic_seen_parallel_group_signatures", set())
            or set())
        detailed = _env_flag(
            "VLLM_ASCEND_MODE1_COMM_CACHE_STATE_DETAILED", "0")
        if detailed:
            logger.info(
                "Mode1 comm cache state: tag=%s rank=%s cached_groups=%s "
                "registry_groups=%s seen_signatures=%s topology_count=%s "
                "topology_methods=%s topology_groups=%s free_bytes=%s "
                "total_bytes=%s torch_current=%s torch_reserved=%s "
                "non_torch=%s total_allocated=%s",
                tag,
                self.rank,
                _fmt_cached_groups(),
                _fmt_registry(),
                _fmt_seen(),
                topology_stats.get("topology_count", 0),
                topology_stats.get("method_count", 0),
                topology_desc,
                free_bytes,
                total_bytes,
                torch_current,
                torch_reserved,
                non_torch,
                total_allocated,
            )
        else:
            logger.info(
                "Mode1 comm cache compact: tag=%s rank=%s cached_group_count=%s "
                "registry_group_count=%s seen_signature_count=%s "
                "topology_count=%s topology_methods=%s free_bytes=%s "
                "total_bytes=%s torch_current=%s torch_reserved=%s "
                "non_torch=%s total_allocated=%s",
                tag,
                self.rank,
                cached_group_count,
                registry_group_count,
                seen_count,
                topology_stats.get("topology_count", 0),
                topology_stats.get("method_count", 0),
                free_bytes,
                total_bytes,
                torch_current,
                torch_reserved,
                non_torch,
                total_allocated,
            )

    def _cleanup_after_elastic_group_release(
            self, reason: str, force: bool = False) -> dict[str, float]:
        stats = {
            "gc_ms": 0.0,
            "empty_cache_ms": 0.0,
            "sync_ms": 0.0,
            "memory_log_ms": 0.0,
            "total_ms": 0.0,
        }
        cleanup_start_t = time.perf_counter()
        if force or _env_flag(
                "VLLM_ASCEND_MODE1_PARITY_GC_AFTER_MC2_GROUP_DESTROY",
                "0"):
            import gc
            gc_start_t = time.perf_counter()
            gc.collect()
            stats["gc_ms"] = (time.perf_counter() - gc_start_t) * 1000.0
        if (torch.npu.is_available() and
                (force or _env_flag(
                    "VLLM_ASCEND_MODE1_PARITY_SYNC_AFTER_MC2_GROUP_DESTROY",
                    "0" if (self._is_mode1_low_floor_elastic_runtime()
                            or self._has_mode1_lightweight_parity_module())
                    else "1"))):
            empty_cache_start_t = time.perf_counter()
            torch.npu.empty_cache()
            stats["empty_cache_ms"] = (
                time.perf_counter() - empty_cache_start_t) * 1000.0
            sync_start_t = time.perf_counter()
            torch.npu.synchronize()
            stats["sync_ms"] = (time.perf_counter() -
                                sync_start_t) * 1000.0
        memory_log_start_t = time.perf_counter()
        self._log_custom_mode1_worker_memory(
            "after_elastic_group_release_cleanup",
            f"reason={reason} force={int(force)}",
        )
        stats["memory_log_ms"] = (
            time.perf_counter() - memory_log_start_t) * 1000.0
        stats["total_ms"] = (time.perf_counter() - cleanup_start_t) * 1000.0
        return stats

    def _cleanup_after_elastic_mc2_group_destroy(self,
                                                 reason: str) -> dict[str, float]:
        return self._cleanup_after_elastic_group_release(reason, force=False)

    def _should_force_cleanup_after_mode1_floor_group_release(
            self, group_kind: str, group_ranks: tuple[int, ...]) -> bool:
        if group_kind not in ("dp", "ep", "mc2"):
            return False
        if not group_ranks:
            return False
        if not self._is_mode1_low_floor_elastic_runtime():
            return False
        if not _env_flag(
                "VLLM_ASCEND_MODE1_PARITY_FORCE_CLEANUP_AFTER_FLOOR_GROUP_RELEASE",
                "1"):
            return False
        if not torch.distributed.is_initialized():
            return False
        world_size = int(torch.distributed.get_world_size())
        return len(group_ranks) < world_size

    def _detach_destroyed_group_references(self, group) -> None:
        """Break wrapper references after ProcessGroup destruction.

        GroupCoordinator.destroy() destroys and deletes its own process-group
        attributes, but the device communicator may still hold cpu/device group
        references. If the last Python reference to the wrapper disappears in
        the shrink hot path, those stale references can defer low-level cleanup
        to frame teardown. Keep explicit destroy timing honest by severing those
        references immediately after destroy().
        """
        communicator = getattr(group, "device_communicator", None)
        if communicator is not None:
            for attr_name in ("device_group", "cpu_group"):
                if hasattr(communicator, attr_name):
                    try:
                        setattr(communicator, attr_name, None)
                    except Exception:
                        pass
        for attr_name in ("device_communicator", "mq_broadcaster"):
            if hasattr(group, attr_name):
                try:
                    setattr(group, attr_name, None)
                except Exception:
                    pass

    def _describe_elastic_group_refs(self, group) -> dict[str, object]:
        communicator = getattr(group, "device_communicator", None)
        group_device_pg = getattr(group, "device_group", None)
        group_cpu_pg = getattr(group, "cpu_group", None)
        comm_device_pg = None
        comm_cpu_pg = None
        if communicator is not None:
            comm_device_pg = getattr(communicator, "device_group", None)
            comm_cpu_pg = getattr(communicator, "cpu_group", None)
        return {
            "group_id": id(group) if group is not None else 0,
            "communicator_id": id(communicator) if communicator is not None else 0,
            "group_device_pg_id": id(group_device_pg) if group_device_pg is not None else 0,
            "group_cpu_pg_id": id(group_cpu_pg) if group_cpu_pg is not None else 0,
            "comm_device_pg_id": id(comm_device_pg) if comm_device_pg is not None else 0,
            "comm_cpu_pg_id": id(comm_cpu_pg) if comm_cpu_pg is not None else 0,
            "has_device_communicator": communicator is not None,
            "has_group_device_pg": group_device_pg is not None,
            "has_group_cpu_pg": group_cpu_pg is not None,
            "has_comm_device_pg": comm_device_pg is not None,
            "has_comm_cpu_pg": comm_cpu_pg is not None,
        }

    def _log_mode1_direct_group_destroy_event(
            self, event: str, group_kind: str, group_ranks: tuple[int, ...],
            reason: str, refs: dict[str, object],
            keep_group_ranks: tuple[int, ...] | None = None,
            attr_name: str | None = None,
            destroy_call_ms: float | None = None,
            detach_ms: float | None = None,
            registry_ms: float | None = None,
            total_ms: float | None = None,
            cleanup_force: bool | None = None) -> None:
        if not self._is_mode1_low_floor_elastic_runtime():
            return
        logger.info(
            "Mode1 direct group destroy %s: rank=%s attr=%s group_kind=%s "
            "group_ranks=%s keep_ranks=%s reason=%s group_id=%s "
            "communicator_id=%s group_device_pg_id=%s group_cpu_pg_id=%s "
            "comm_device_pg_id=%s comm_cpu_pg_id=%s has_device_communicator=%s "
            "has_group_device_pg=%s has_group_cpu_pg=%s has_comm_device_pg=%s "
            "has_comm_cpu_pg=%s destroy_call_ms=%s detach_ms=%s "
            "registry_ms=%s total_ms=%s cleanup_force=%s",
            event,
            self.rank,
            attr_name,
            group_kind,
            group_ranks,
            keep_group_ranks,
            reason,
            refs.get("group_id"),
            refs.get("communicator_id"),
            refs.get("group_device_pg_id"),
            refs.get("group_cpu_pg_id"),
            refs.get("comm_device_pg_id"),
            refs.get("comm_cpu_pg_id"),
            refs.get("has_device_communicator"),
            refs.get("has_group_device_pg"),
            refs.get("has_group_cpu_pg"),
            refs.get("has_comm_device_pg"),
            refs.get("has_comm_cpu_pg"),
            "%.2f" % destroy_call_ms if destroy_call_ms is not None else "-",
            "%.2f" % detach_ms if detach_ms is not None else "-",
            "%.2f" % registry_ms if registry_ms is not None else "-",
            "%.2f" % total_ms if total_ms is not None else "-",
            cleanup_force,
        )

    def _should_preserve_device_pg_on_retire(
            self,
            group_kind: str | None,
            group_ranks: tuple[int, ...],
            reason: str | None) -> bool:
        if not self._is_mode1_low_floor_elastic_runtime():
            return False
        if group_kind not in ("dp", "ep", "mc2"):
            return False
        if not group_ranks:
            return False
        if reason == "mode1_full_restore":
            return _env_flag(
                "VLLM_ASCEND_MODE1_PARITY_PRESERVE_DEVICE_PG_ON_FULL_RESTORE",
                "0")
        return False

    def _should_preserve_group_refs_on_retire(
            self,
            group_kind: str | None,
            group_ranks: tuple[int, ...],
            reason: str | None) -> bool:
        if not self._should_preserve_device_pg_on_retire(
                group_kind, group_ranks, reason):
            return False
        return _env_flag(
            "VLLM_ASCEND_MODE1_PARITY_PRESERVE_GROUP_REFS_ON_FULL_RESTORE",
            "0")

    def _should_skip_cleanup_after_preserved_full_restore_release(
            self,
            group_kind: str,
            group_ranks: tuple[int, ...],
            reason: str) -> bool:
        if reason != "mode1_full_restore":
            return False
        if not _env_flag(
                "VLLM_ASCEND_MODE1_PARITY_SKIP_CLEANUP_ON_PRESERVED_FULL_RESTORE",
                "0"):
            return False
        return self._should_preserve_group_refs_on_retire(
            group_kind, group_ranks, reason)

    def _retire_group_wrapper_defer_pg_destroy(
            self,
            group,
            group_kind: str | None = None,
            group_ranks: tuple[int, ...] | None = None,
            reason: str | None = None) -> dict[str, float]:
        """Retire a GroupCoordinator wrapper without hot-path PG destroy.

        Low-floor mode=1 GroupCoordinator.destroy() can block for hundreds of
        seconds on this stack. The shrink path still must remove stale groups
        from every live/cache lookup immediately. Keep raw ProcessGroup objects
        in a private deferred list after clearing the GroupCoordinator and
        communicator wrappers, so wrapper/device-communicator buffers cannot be
        reused but low-level PG destruction is not forced inside the shrink RPC.

        Device PG destruction is enabled by default to release HCCL/HBM
        workspace.  One exception is mode1 full-world restore from low floors:
        floor2 introduced a 4->2->full-world path where destroying the old
        floor2 device PG during an active run can invalidate HCCL/MC2 runtime
        state before the next step starts.  In that case, remove the stale group
        from Python live/cache paths but keep the raw device PG until a later
        safe boundary.
        """
        retire_start_t = time.perf_counter()
        effective_group_ranks = tuple(
            int(rank) for rank in (group_ranks
                                   or tuple(getattr(group, "ranks", []))))
        npu_sync_ms = 0.0
        device_pg_destroy_ms = 0.0
        cpu_pg_destroy_ms = 0.0
        dropped_device_pg_refs = 0.0
        deferred_device_pg_refs = 0.0
        deferred_cpu_pg_refs = 0.0
        device_pgs: list[object] = []
        cpu_pgs: list[object] = []
        preserve_device_pg = self._should_preserve_device_pg_on_retire(
            group_kind,
            effective_group_ranks,
            reason,
        )
        preserve_group_refs = self._should_preserve_group_refs_on_retire(
            group_kind,
            effective_group_ranks,
            reason,
        )
        if _env_flag("VLLM_ASCEND_MODE1_PARITY_SYNC_BEFORE_DEVICE_PG_RETIRE",
                     "1"):
            sync_start_t = time.perf_counter()
            try:
                torch.npu.synchronize()
            except Exception:
                logger.exception(
                    "Elastic retired group NPU sync failed: rank=%s "
                    "group_ranks=%s",
                    self.rank,
                    effective_group_ranks,
                )
            npu_sync_ms = (time.perf_counter() - sync_start_t) * 1000.0

        communicator = getattr(group, "device_communicator", None)
        if communicator is not None:
            if hasattr(communicator, "device_group"):
                device_pg = getattr(communicator, "device_group", None)
                if device_pg is not None:
                    device_pgs.append(device_pg)
            if not preserve_group_refs:
                for attr_name in ("device_group", ):
                    if hasattr(communicator, attr_name):
                        try:
                            setattr(communicator, attr_name, None)
                            dropped_device_pg_refs += 1.0
                        except Exception:
                            pass
            if hasattr(communicator, "cpu_group"):
                cpu_pg = getattr(communicator, "cpu_group", None)
                if cpu_pg is not None:
                    cpu_pgs.append(cpu_pg)
                    deferred_cpu_pg_refs += 1.0
                if not preserve_group_refs:
                    try:
                        setattr(communicator, "cpu_group", None)
                    except Exception:
                        pass

        if hasattr(group, "device_group"):
            device_pg = getattr(group, "device_group", None)
            if device_pg is not None:
                device_pgs.append(device_pg)
            if not preserve_group_refs:
                try:
                    delattr(group, "device_group")
                    dropped_device_pg_refs += 1.0
                except Exception:
                    pass
        if hasattr(group, "cpu_group"):
            cpu_pg = getattr(group, "cpu_group", None)
            if cpu_pg is not None:
                cpu_pgs.append(cpu_pg)
                deferred_cpu_pg_refs += 1.0
            if not preserve_group_refs:
                try:
                    delattr(group, "cpu_group")
                except Exception:
                    pass

        seen_pg_ids: set[int] = set()
        unique_device_pgs = []
        for device_pg in device_pgs:
            device_pg_id = id(device_pg)
            if device_pg_id in seen_pg_ids:
                continue
            seen_pg_ids.add(device_pg_id)
            unique_device_pgs.append(device_pg)
        destroy_device_pg = (_env_flag(
            "VLLM_ASCEND_MODE1_PARITY_DESTROY_DEVICE_PG_ON_RETIRE", "1")
                             and not preserve_device_pg)
        if destroy_device_pg:
            destroy_start_t = time.perf_counter()
            for device_pg in unique_device_pgs:
                try:
                    torch.distributed.destroy_process_group(device_pg)
                except Exception:
                    logger.exception(
                        "Elastic retired group device ProcessGroup destroy failed: "
                        "rank=%s group_ranks=%s",
                        self.rank,
                        effective_group_ranks,
                    )
            device_pg_destroy_ms = (
                time.perf_counter() - destroy_start_t) * 1000.0
            deferred_device_pgs = []
        else:
            deferred_device_pg_refs = float(len(unique_device_pgs))
            deferred_device_pgs = unique_device_pgs

        if not preserve_group_refs:
            for attr_name in ("device_communicator", "mq_broadcaster"):
                if hasattr(group, attr_name):
                    try:
                        setattr(group, attr_name, None)
                    except Exception:
                        pass
        seen_cpu_pg_ids: set[int] = set()
        unique_cpu_pgs = []
        for cpu_pg in cpu_pgs:
            cpu_pg_id = id(cpu_pg)
            if cpu_pg_id in seen_cpu_pg_ids:
                continue
            seen_cpu_pg_ids.add(cpu_pg_id)
            unique_cpu_pgs.append(cpu_pg)
        destroy_cpu_pg = (_env_flag(
            "VLLM_ASCEND_MODE1_PARITY_DESTROY_CPU_PG_ON_RETIRE", "1")
                          and not preserve_group_refs)
        if destroy_cpu_pg:
            cpu_destroy_start_t = time.perf_counter()
            for cpu_pg in unique_cpu_pgs:
                try:
                    torch.distributed.destroy_process_group(cpu_pg)
                except Exception:
                    logger.exception(
                        "Elastic retired group CPU ProcessGroup destroy failed: "
                        "rank=%s group_ranks=%s",
                        self.rank,
                        effective_group_ranks,
                    )
            cpu_pg_destroy_ms = (
                time.perf_counter() - cpu_destroy_start_t) * 1000.0
            deferred_cpu_pg_refs = 0.0
            unique_cpu_pgs = []
        return {
            "retire_wrapper_ms": (
                time.perf_counter() - retire_start_t) * 1000.0,
            "npu_sync_ms": npu_sync_ms,
            "device_pg_destroy_ms": device_pg_destroy_ms,
            "cpu_pg_destroy_ms": cpu_pg_destroy_ms,
            "destroy_device_pg": 1.0 if destroy_device_pg else 0.0,
            "destroy_cpu_pg": 1.0 if destroy_cpu_pg else 0.0,
            "preserve_device_pg": 1.0 if preserve_device_pg else 0.0,
            "preserve_group_refs": 1.0 if preserve_group_refs else 0.0,
            "dropped_pg_refs": dropped_device_pg_refs,
            "deferred_device_pg_refs": deferred_device_pg_refs,
            "deferred_cpu_pg_refs": deferred_cpu_pg_refs,
            "device_pgs": deferred_device_pgs,
            "cpu_pgs": unique_cpu_pgs,
            "group": group if preserve_group_refs else None,
        }

    def _get_deferred_elastic_group_ids(self) -> set[int]:
        deferred_ids = getattr(self, "_elastic_deferred_destroy_group_ids",
                               None)
        if deferred_ids is None:
            deferred_ids = set()
            self._elastic_deferred_destroy_group_ids = deferred_ids
        return deferred_ids

    def _get_retired_elastic_group_ids(self) -> set[int]:
        retired_ids = getattr(self, "_elastic_retired_group_ids", None)
        if retired_ids is None:
            retired_ids = set()
            self._elastic_retired_group_ids = retired_ids
        return retired_ids

    def _mark_retired_elastic_group(self, group) -> None:
        self._get_retired_elastic_group_ids().add(id(group))

    def _is_retired_elastic_group(self, group) -> bool:
        group_id = id(group)
        return (group_id in self._get_retired_elastic_group_ids()
                or group_id in self._get_deferred_elastic_group_ids())

    def _quarantine_deferred_elastic_group(
            self, group, group_kind: str,
            group_ranks: tuple[int, ...], reason: str) -> bool:
        """Remove a stale group from live/cache paths without wrapper destroy.

        On this CANN/HCCL stack, destroying low-floor elastic communicators in
        the shrink RPC hot path can block for roughly 350s. The group is already
        no longer reachable from vLLM parallel state after stashing. Retire the
        wrapper and explicitly destroy raw ProcessGroups when enabled, but only
        keep deferred PG references when there is something left to defer. This
        keeps arbitrary-rank shrink groups from accumulating empty wrapper
        shells and stale HCCL workspace pressure across steps.
        """
        deferred_ids = self._get_deferred_elastic_group_ids()
        retired_ids = self._get_retired_elastic_group_ids()
        group_id = id(group)
        if group_id in deferred_ids or group_id in retired_ids:
            logger.info(
                "Elastic parallel group destroy already retired/deferred: rank=%s "
                "group_kind=%s group_ranks=%s reason=%s",
                self.rank,
                group_kind,
                group_ranks,
                reason,
            )
            return False
        deferred_ids.add(group_id)
        self._mark_retired_elastic_group(group)
        retire_stats = self._retire_group_wrapper_defer_pg_destroy(
            group,
            group_kind=group_kind,
            group_ranks=group_ranks,
            reason=reason,
        )
        deferred_device_pgs = list(retire_stats.get("device_pgs", []) or [])
        deferred_cpu_pgs = list(retire_stats.get("cpu_pgs", []) or [])
        self._remove_elastic_parallel_group_registry_entry(
            group_kind, group_ranks, group)
        if not deferred_device_pgs and not deferred_cpu_pgs:
            deferred_ids.discard(group_id)
            logger.info(
                "Elastic parallel group retired immediately: rank=%s "
                "group_kind=%s group_ranks=%s reason=%s "
                "retire_wrapper_ms=%.2f npu_sync_ms=%.2f "
                "device_pg_destroy_ms=%.2f cpu_pg_destroy_ms=%.2f "
                "destroy_device_pg=%.0f destroy_cpu_pg=%.0f "
                "preserve_device_pg=%.0f preserve_group_refs=%.0f "
                "dropped_pg_refs=%.0f",
                self.rank,
                group_kind,
                group_ranks,
                reason,
                float(retire_stats.get("retire_wrapper_ms", 0.0)),
                float(retire_stats.get("npu_sync_ms", 0.0)),
                float(retire_stats.get("device_pg_destroy_ms", 0.0)),
                float(retire_stats.get("cpu_pg_destroy_ms", 0.0)),
                float(retire_stats.get("destroy_device_pg", 0.0)),
                float(retire_stats.get("destroy_cpu_pg", 0.0)),
                float(retire_stats.get("preserve_device_pg", 0.0)),
                float(retire_stats.get("preserve_group_refs", 0.0)),
                float(retire_stats.get("dropped_pg_refs", 0.0)),
            )
            return True

        deferred = getattr(self, "_elastic_deferred_destroy_groups", None)
        if deferred is None:
            deferred = []
            self._elastic_deferred_destroy_groups = deferred
        deferred.append({
            "device_pgs": deferred_device_pgs,
            "cpu_pgs": deferred_cpu_pgs,
            "group": retire_stats.get("group"),
            "group_id": group_id,
            "group_kind": group_kind,
            "group_ranks": group_ranks,
            "reason": reason,
        })
        logger.info(
            "Elastic parallel group destroy deferred: rank=%s "
            "group_kind=%s group_ranks=%s reason=%s deferred_count=%s "
            "retire_wrapper_ms=%.2f npu_sync_ms=%.2f "
            "device_pg_destroy_ms=%.2f cpu_pg_destroy_ms=%.2f "
            "destroy_device_pg=%.0f destroy_cpu_pg=%.0f "
            "preserve_device_pg=%.0f preserve_group_refs=%.0f "
            "dropped_pg_refs=%.0f deferred_device_pg_refs=%.0f "
            "deferred_cpu_pg_refs=%.0f",
            self.rank,
            group_kind,
            group_ranks,
            reason,
            len(deferred),
            float(retire_stats.get("retire_wrapper_ms", 0.0)),
            float(retire_stats.get("npu_sync_ms", 0.0)),
            float(retire_stats.get("device_pg_destroy_ms", 0.0)),
            float(retire_stats.get("cpu_pg_destroy_ms", 0.0)),
            float(retire_stats.get("destroy_device_pg", 0.0)),
            float(retire_stats.get("destroy_cpu_pg", 0.0)),
            float(retire_stats.get("preserve_device_pg", 0.0)),
            float(retire_stats.get("preserve_group_refs", 0.0)),
            float(retire_stats.get("dropped_pg_refs", 0.0)),
            float(retire_stats.get("deferred_device_pg_refs", 0.0)),
            float(retire_stats.get("deferred_cpu_pg_refs", 0.0)),
        )
        return True

    def _release_elastic_parallel_group_reference(
            self, group, group_kind: str, group_ranks: tuple[int, ...],
            keep_group_ranks: tuple[int, ...], reason: str,
            destroyed_group_ids: set[int],
            memory_log_tag: str | None = None,
            force_destroy: bool = False) -> dict[str, float]:
        stats = {
            "destroy_ms": 0.0,
            "cleanup_ms": 0.0,
            "cleanup_gc_ms": 0.0,
            "cleanup_empty_cache_ms": 0.0,
            "cleanup_sync_ms": 0.0,
            "cleanup_memory_log_ms": 0.0,
            "deferred_groups": 0.0,
            "released_groups": 0.0,
        }
        if group is None:
            return stats
        group_id = id(group)
        if group_id in destroyed_group_ids:
            return stats
        if memory_log_tag and group_kind == "mc2":
            self._log_custom_mode1_worker_memory(
                memory_log_tag,
                f"stale_ranks={group_ranks} keep_ranks={keep_group_ranks}",
            )
        force_cleanup_after_release = (
            self._should_force_cleanup_after_mode1_floor_group_release(
                group_kind, group_ranks))
        if self._should_skip_cleanup_after_preserved_full_restore_release(
                group_kind, group_ranks, reason):
            force_cleanup_after_release = False
        if self._is_retired_elastic_group(group):
            destroyed_group_ids.add(group_id)
            self._remove_elastic_parallel_group_registry_entry(
                group_kind, group_ranks, group)
            stats["released_groups"] = 1.0
            return stats
        if (not force_destroy and self._should_defer_mode1_group_destroy(
                group_kind,
                group_ranks,
                keep_group_ranks,
                reason,
        )):
            if self._quarantine_deferred_elastic_group(
                    group,
                    group_kind,
                    group_ranks,
                    reason,
            ):
                stats["deferred_groups"] = 1.0
            destroyed_group_ids.add(group_id)
            stats["released_groups"] = 1.0
            if force_cleanup_after_release:
                cleanup_stats = self._cleanup_after_elastic_group_release(
                    reason, force=True)
                stats["cleanup_ms"] += float(
                    cleanup_stats.get("total_ms", 0.0))
                stats["cleanup_gc_ms"] += float(
                    cleanup_stats.get("gc_ms", 0.0))
                stats["cleanup_empty_cache_ms"] += float(
                    cleanup_stats.get("empty_cache_ms", 0.0))
                stats["cleanup_sync_ms"] += float(
                    cleanup_stats.get("sync_ms", 0.0))
                stats["cleanup_memory_log_ms"] += float(
                    cleanup_stats.get("memory_log_ms", 0.0))
            return stats

        if self._is_mode1_low_floor_elastic_runtime():
            logger.warning(
                "Elastic mode1 low-floor group direct destroy fallback: "
                "rank=%s group_kind=%s group_ranks=%s keep_ranks=%s "
                "reason=%s defer_env=%s defer_full_restore=%s "
                "defer_pre_rebuild=%s defer_stash=%s defer_sizes=%s",
                self.rank,
                group_kind,
                group_ranks,
                keep_group_ranks,
                reason,
                os.getenv("VLLM_ASCEND_MODE1_PARITY_DEFER_GROUP_DESTROY",
                          "1"),
                os.getenv(
                    "VLLM_ASCEND_MODE1_PARITY_DEFER_GROUP_DESTROY_ON_FULL_RESTORE",
                    "0"),
                os.getenv(
                    "VLLM_ASCEND_MODE1_PARITY_DEFER_GROUP_DESTROY_ON_PRE_REBUILD",
                    "0"),
                os.getenv(
                    "VLLM_ASCEND_MODE1_PARITY_DEFER_GROUP_DESTROY_ON_STASH",
                    "0"),
                os.getenv(
                    "VLLM_ASCEND_MODE1_PARITY_DEFER_DESTROY_FLOOR_GROUP_SIZES",
                    "1,2,4,8"),
            )
        refs_before_destroy = self._describe_elastic_group_refs(group)
        self._log_mode1_direct_group_destroy_event(
            "start",
            group_kind,
            group_ranks,
            reason,
            refs_before_destroy,
            keep_group_ranks=keep_group_ranks,
            cleanup_force=force_cleanup_after_release,
        )
        destroy_start_t = time.perf_counter()
        destroy_call_start_t = time.perf_counter()
        group.destroy()
        destroy_call_ms = (
            time.perf_counter() - destroy_call_start_t) * 1000.0
        detach_start_t = time.perf_counter()
        self._detach_destroyed_group_references(group)
        detach_ms = (time.perf_counter() - detach_start_t) * 1000.0
        self._mark_retired_elastic_group(group)
        registry_start_t = time.perf_counter()
        self._remove_elastic_parallel_group_registry_entry(
            group_kind, group_ranks, group)
        registry_ms = (time.perf_counter() - registry_start_t) * 1000.0
        stats["destroy_ms"] = (
            time.perf_counter() - destroy_start_t) * 1000.0
        refs_after_destroy = self._describe_elastic_group_refs(group)
        self._log_mode1_direct_group_destroy_event(
            "done",
            group_kind,
            group_ranks,
            reason,
            refs_after_destroy,
            keep_group_ranks=keep_group_ranks,
            destroy_call_ms=destroy_call_ms,
            detach_ms=detach_ms,
            registry_ms=registry_ms,
            total_ms=stats["destroy_ms"],
            cleanup_force=force_cleanup_after_release,
        )
        destroyed_group_ids.add(group_id)
        stats["released_groups"] = 1.0
        if group_kind == "mc2" and (
                self._has_mode1_lightweight_parity_module()
                or self._has_mode4_remote_npu_lightweight_module()):
            cleanup_stats = self._cleanup_after_elastic_group_release(
                reason, force=force_cleanup_after_release)
            stats["cleanup_ms"] += float(cleanup_stats.get("total_ms", 0.0))
            stats["cleanup_gc_ms"] += float(cleanup_stats.get("gc_ms", 0.0))
            stats["cleanup_empty_cache_ms"] += float(
                cleanup_stats.get("empty_cache_ms", 0.0))
            stats["cleanup_sync_ms"] += float(
                cleanup_stats.get("sync_ms", 0.0))
            stats["cleanup_memory_log_ms"] += float(
                cleanup_stats.get("memory_log_ms", 0.0))
        elif force_cleanup_after_release:
            cleanup_stats = self._cleanup_after_elastic_group_release(
                reason, force=True)
            stats["cleanup_ms"] += float(cleanup_stats.get("total_ms", 0.0))
            stats["cleanup_gc_ms"] += float(cleanup_stats.get("gc_ms", 0.0))
            stats["cleanup_empty_cache_ms"] += float(
                cleanup_stats.get("empty_cache_ms", 0.0))
            stats["cleanup_sync_ms"] += float(
                cleanup_stats.get("sync_ms", 0.0))
            stats["cleanup_memory_log_ms"] += float(
                cleanup_stats.get("memory_log_ms", 0.0))
        return stats

    def _should_defer_mode1_group_destroy(
            self, group_kind: str, group_ranks: tuple[int, ...],
            keep_group_ranks: tuple[int, ...], reason: str) -> bool:
        if not self._is_mode1_low_floor_elastic_runtime():
            return False
        if (reason == "mode1_full_restore" and not _env_flag(
                "VLLM_ASCEND_MODE1_PARITY_DEFER_GROUP_DESTROY_ON_FULL_RESTORE",
                "0")):
            return False
        if (reason == "mode1_pre_rebuild" and not _env_flag(
                "VLLM_ASCEND_MODE1_PARITY_DEFER_GROUP_DESTROY_ON_PRE_REBUILD",
                "0")):
            return False
        if (reason == "stash_group" and not _env_flag(
                "VLLM_ASCEND_MODE1_PARITY_DEFER_GROUP_DESTROY_ON_STASH",
                "0")):
            return False
        if not _env_flag("VLLM_ASCEND_MODE1_PARITY_DEFER_GROUP_DESTROY", "1"):
            return False
        if group_kind not in ("dp", "ep", "mc2"):
            return False
        if not group_ranks or group_ranks == keep_group_ranks:
            return False
        if not torch.distributed.is_initialized():
            return False
        world_size = int(torch.distributed.get_world_size())
        if group_ranks == tuple(
                self._original_local_elastic_group_ranks(world_size)):
            return False
        raw_sizes = os.getenv(
            "VLLM_ASCEND_MODE1_PARITY_DEFER_DESTROY_FLOOR_GROUP_SIZES",
            "1,2,4,8",
        )
        defer_sizes: set[int] = set()
        for raw_size in raw_sizes.split(","):
            raw_size = raw_size.strip()
            if not raw_size:
                continue
            try:
                defer_sizes.add(int(raw_size))
            except ValueError:
                continue
        return len(group_ranks) in defer_sizes

    def _reset_elastic_moe_comm_setup_cache(self, reason: str) -> None:
        reset_moe_comm_method_cache()
        logger.info(
            "Elastic MoE comm setup cache reset: rank=%s reason=%s",
            self.rank,
            reason,
        )

    def _get_cached_elastic_parallel_groups(self) -> dict[str, dict[tuple[int, ...], object]]:
        cached_groups = getattr(self, "_elastic_cached_parallel_groups", None)
        if cached_groups is None:
            cached_groups = {}
            self._elastic_cached_parallel_groups = cached_groups
        return cached_groups

    def _get_elastic_parallel_group_registry(
            self) -> dict[str, dict[tuple[int, ...], dict[int, object]]]:
        registry = getattr(self, "_elastic_parallel_group_registry", None)
        if registry is None:
            registry = {}
            self._elastic_parallel_group_registry = registry
        return registry

    def _register_elastic_parallel_group(
            self, attr_name: str, group_ranks: tuple[int, ...],
            group: object | None) -> None:
        if group is None or not group_ranks:
            return
        group_kind = self._normalize_elastic_parallel_group_kind(attr_name)
        if group_kind not in ("dp", "ep", "mc2"):
            return
        groups_by_ranks = self._get_elastic_parallel_group_registry(
        ).setdefault(group_kind, {}).setdefault(group_ranks, {})
        groups_by_ranks[id(group)] = group

    def _remove_elastic_parallel_group_registry_entry(
            self, group_kind: str, group_ranks: tuple[int, ...],
            group: object | None = None) -> bool:
        registry = getattr(self, "_elastic_parallel_group_registry", None)
        if not registry:
            return False
        groups_by_kind = registry.get(group_kind)
        if not groups_by_kind:
            return False
        groups_by_id = groups_by_kind.get(group_ranks)
        if not groups_by_id:
            return False
        removed = False
        if group is None:
            removed = bool(groups_by_id)
            groups_by_id.clear()
        else:
            removed = groups_by_id.pop(id(group), None) is not None
        if not groups_by_id:
            groups_by_kind.pop(group_ranks, None)
        if not groups_by_kind:
            registry.pop(group_kind, None)
        return removed

    def _iter_stale_elastic_parallel_group_registry_entries(
            self, keep_group_ranks: tuple[int, ...],
            old_floor_only: bool) -> list[tuple[str, tuple[int, ...], object]]:
        registry = getattr(self, "_elastic_parallel_group_registry", None)
        if not registry:
            return []
        entries: list[tuple[str, tuple[int, ...], object]] = []
        for group_kind, groups_by_ranks in registry.items():
            if group_kind not in ("dp", "ep", "mc2"):
                continue
            for ranks, groups_by_id in groups_by_ranks.items():
                ranks_tuple = tuple(int(rank) for rank in ranks)
                should_drop = (
                    self._should_drop_old_floor_group_cache_after_mode1_shrink(
                        group_kind, ranks_tuple, keep_group_ranks)
                    if old_floor_only else
                    ranks_tuple != keep_group_ranks
                    and not (
                        self._should_keep_stale_group_cache_for_custom_mode1_parity(
                            group_kind, ranks_tuple, keep_group_ranks)
                        or self._should_keep_stale_group_cache_for_mode3(
                            group_kind, ranks_tuple, keep_group_ranks)
                        or self._should_keep_stale_group_cache_for_mode5(
                            group_kind, ranks_tuple, keep_group_ranks)
                    ))
                if not should_drop:
                    continue
                for group in list(groups_by_id.values()):
                    entries.append((group_kind, ranks_tuple, group))
        return entries

    def _get_seen_elastic_parallel_group_signatures(
            self) -> set[tuple[str, tuple[int, ...]]]:
        seen_signatures = getattr(
            self, "_elastic_seen_parallel_group_signatures", None)
        if seen_signatures is None:
            seen_signatures = set()
            self._elastic_seen_parallel_group_signatures = seen_signatures
        return seen_signatures

    def _remove_cached_elastic_parallel_group_reference(
            self, attr_name: str, group_ranks: tuple[int, ...],
            group=None) -> bool:
        if not group_ranks:
            return False
        cached_groups = getattr(self, "_elastic_cached_parallel_groups", None)
        if not cached_groups:
            return False
        groups_by_ranks = cached_groups.get(attr_name)
        if not groups_by_ranks:
            return False
        cached_group = groups_by_ranks.get(group_ranks)
        if cached_group is None:
            return False
        if group is not None and cached_group is not group:
            return False
        groups_by_ranks.pop(group_ranks, None)
        group_kind = self._normalize_elastic_parallel_group_kind(attr_name)
        self._get_seen_elastic_parallel_group_signatures().discard(
            (group_kind, group_ranks))
        return True

    def _normalize_elastic_parallel_group_kind(self, name: str) -> str:
        normalized_name = name.lower().lstrip("_")
        if normalized_name.endswith("dp"):
            return "dp"
        if normalized_name.endswith("ep"):
            return "ep"
        if normalized_name.endswith("mc2"):
            return "mc2"
        return normalized_name

    def _mode1_fixed_topology_group_reuse_enabled(self) -> bool:
        if int(envs_ascend.VLLM_ASCEND_ELASTIC_EXECUTION_MODE) != 1:
            return False
        if not envs_ascend.VLLM_ASCEND_ENABLE_ELASTIC_PARALLEL_SHRINK:
            return False
        target_policy = os.getenv(
            "VLLM_ASCEND_SHRINK_AWARE_TARGET_POLICY",
            "natural").lower().strip()
        if target_policy not in ("planned", "fixed", "plan"):
            return False
        if self._is_mode1_low_floor_elastic_runtime():
            return True
        # Only explicit planned-floor caching should keep floor-8/4 groups
        # resident while the current step runs full-world. Precreate alone is
        # a KV sizing aid, not a cross-step residency request.
        return self._mode1_planned_floor_group_cache_enabled()

    def _mode1_precreate_planned_floor_groups_enabled(self) -> bool:
        if not _env_flag(
                "VLLM_ASCEND_MODE1_PARITY_PRECREATE_PLANNED_FLOOR_GROUPS",
                "0"):
            return False
        if int(envs_ascend.VLLM_ASCEND_ELASTIC_EXECUTION_MODE) != 1:
            return False
        if not envs_ascend.VLLM_ASCEND_ENABLE_ELASTIC_PARALLEL_SHRINK:
            return False
        target_policy = os.getenv(
            "VLLM_ASCEND_SHRINK_AWARE_TARGET_POLICY",
            "natural").lower().strip()
        return target_policy in ("planned", "fixed", "plan")

    def _mode1_planned_floor_group_cache_enabled(self) -> bool:
        return _env_flag(
            "VLLM_ASCEND_MODE1_PARITY_CACHE_PLANNED_FLOOR_GROUPS", "0")

    def _mode1_precreate_comm_cache_enabled(self) -> bool:
        return _env_flag(
            "VLLM_ASCEND_MODE1_PARITY_PRECREATE_COMM_CACHE", "0")

    def _mode1_prefill_planned_expert_slots_enabled(self) -> bool:
        if not _env_flag(
                "VLLM_ASCEND_MODE1_PARITY_PREFILL_PLANNED_EXPERT_SLOTS",
                "0"):
            return False
        if int(envs_ascend.VLLM_ASCEND_ELASTIC_EXECUTION_MODE) != 1:
            return False
        if not envs_ascend.VLLM_ASCEND_ENABLE_ELASTIC_PARALLEL_SHRINK:
            return False
        if envs_ascend.VLLM_ASCEND_ELASTIC_MOE_MODE != "lossless":
            return False
        target_policy = os.getenv(
            "VLLM_ASCEND_SHRINK_AWARE_TARGET_POLICY",
            "natural").lower().strip()
        return target_policy in ("planned", "fixed", "plan")

    def _mode1_fixed_topology_reusable_group_ranks(
            self) -> set[tuple[int, ...]]:
        if not torch.distributed.is_initialized():
            return set()
        world_size = int(torch.distributed.get_world_size())
        original_groups = [
            tuple(int(rank) for rank in ranks)
            for ranks in self._build_original_ep_group_ranks(world_size)
        ]
        fixed_ranks: set[tuple[int, ...]] = set(original_groups)
        if not self._mode1_planned_floor_group_cache_enabled():
            # Planned shrink targets are still honored by the trigger policy,
            # but floor-8/floor-4 communicator residency is too expensive for
            # the adaptive KV-cache path. By default only keep the full-world
            # group, matching the earlier shortswap memory behavior while
            # preserving fixed target selection.
            return fixed_ranks
        raw_stage_ranks = os.getenv(
            "VLLM_ASCEND_SHRINK_AWARE_STAGE_RANKS", "")
        local_world_size = len(original_groups[0]) if original_groups else world_size
        if _env_flag("VLLM_ASCEND_DP2_EP8_RETAIN_ALL_FLOOR_GROUPS", "0"):
            if local_world_size != 8:
                raise ValueError(
                    "VLLM_ASCEND_DP2_EP8_RETAIN_ALL_FLOOR_GROUPS requires "
                    f"EP8 groups, got local_world_size={local_world_size}")
            configured_stages = [
                int(item.strip())
                for item in os.getenv(
                    "VLLM_ASCEND_SHRINK_AWARE_STAGES", "4,2").split(",")
                if item.strip()
            ]
            stage_ranks = [
                list(range(local_world_size - stage, local_world_size))
                for stage in configured_stages
                if 0 < stage < local_world_size
            ]
        else:
            try:
                stage_ranks = parse_stage_survivor_ranks(
                    raw_stage_ranks, world_size=local_world_size)
            except Exception:
                logger.exception(
                    "Failed to parse planned shrink-aware stage ranks for group cache reuse: rank=%s raw=%r",
                    self.rank,
                    raw_stage_ranks,
                )
                stage_ranks = None
        if stage_ranks:
            for original_group in original_groups:
                for ranks in stage_ranks:
                    fixed_ranks.add(tuple(
                        original_group[int(local_rank)]
                        for local_rank in ranks))
            return fixed_ranks

        intermediate = parse_rank_list(os.getenv(
            "VLLM_ASCEND_SHRINK_AWARE_INTERMEDIATE_RANKS", ""))
        final = parse_rank_list(os.getenv(
            "VLLM_ASCEND_SHRINK_AWARE_FINAL_RANKS", ""))
        for original_group in original_groups:
            for ranks in (intermediate, final):
                if ranks:
                    fixed_ranks.add(tuple(
                        original_group[int(local_rank)]
                        for local_rank in sorted(map(int, ranks))))
        return fixed_ranks

    def _mode1_planned_floor_group_signature(self) -> tuple[tuple[int, ...], ...]:
        if not torch.distributed.is_initialized():
            return ()
        world_size = int(torch.distributed.get_world_size())
        original_groups = {
            tuple(map(int, ranks))
            for ranks in self._build_original_ep_group_ranks(world_size)
        }
        planned = [
            tuple(int(rank) for rank in ranks)
            for ranks in self._mode1_fixed_topology_reusable_group_ranks()
            if ranks and ranks not in original_groups
        ]
        return tuple(sorted(set(planned), key=lambda ranks:
                            (-len(ranks), ranks)))

    def _release_mode1_moe_comm_runtime_state(
            self, reason: str) -> dict[str, int]:
        start_t = time.perf_counter()
        stats = release_moe_comm_method_runtime_state()
        stats["total_ms"] = int((time.perf_counter() - start_t) * 1000.0)
        if (stats.get("method_attrs", 0) or stats.get("dispatcher_attrs", 0)
                or stats.get("prepare_attrs", 0)):
            logger.info(
                "Mode1 MoE comm runtime transient release: rank=%s reason=%s "
                "methods=%s topologies=%s method_attrs=%s "
                "dispatcher_attrs=%s prepare_attrs=%s tensors=%s "
                "tensor_bytes=%s total_ms=%s",
                self.rank,
                reason,
                stats.get("methods", 0),
                stats.get("topologies", 0),
                stats.get("method_attrs", 0),
                stats.get("dispatcher_attrs", 0),
                stats.get("prepare_attrs", 0),
                stats.get("tensors", 0),
                stats.get("tensor_bytes", 0),
                stats.get("total_ms", 0),
            )
        return stats

    def _prune_mode1_planned_floor_groups_for_current_step(
            self, target_floor: int, reason: str) -> dict[str, float]:
        stats: dict[str, float] = {
            "changed": 0.0,
            "drop_ms": 0.0,
            "dropped_groups": 0.0,
            "dropped_signatures": 0.0,
            "deferred_groups": 0.0,
            "topology_dropped": 0.0,
            "topology_methods_dropped": 0.0,
            "runtime_tensor_bytes": 0.0,
        }
        if not (
                self._mode1_planned_floor_group_cache_enabled()
                or self._mode1_precreate_planned_floor_groups_enabled()):
            return stats
        if not torch.distributed.is_initialized():
            return stats
        world_size = int(torch.distributed.get_world_size())
        keep_group_ranks = tuple(
            self._original_local_elastic_group_ranks(world_size))
        desired_groups = self._mode1_fixed_topology_reusable_group_ranks()
        desired_signature = self._mode1_planned_floor_group_signature()
        old_signature = getattr(
            self, "_mode1_planned_floor_groups_precreated_signature", None)

        force_destroy = _env_flag(
            "VLLM_ASCEND_MODE1_PARITY_FORCE_DESTROY_PLANNED_PRUNE", "1")
        drop_stats = self._drop_stale_cached_elastic_parallel_groups(
            keep_group_ranks,
            reason=f"planned_floor_prune:{reason}",
            force_destroy=force_destroy,
        )
        topology_stats = prune_moe_comm_method_topology_cache(desired_groups)
        runtime_stats = self._release_mode1_moe_comm_runtime_state(
            f"planned_floor_prune:{reason}:target_floor={target_floor}")
        gc.collect()
        try:
            torch.npu.empty_cache()
            torch.npu.synchronize()
        except Exception:
            logger.exception(
                "Failed to empty/sync NPU cache after planned floor group prune")

        stats["drop_ms"] = float(drop_stats.get("total_ms", 0.0))
        stats["dropped_groups"] = float(drop_stats.get("dropped_groups", 0.0))
        stats["dropped_signatures"] = float(
            drop_stats.get("dropped_signatures", 0.0))
        stats["deferred_groups"] = float(
            drop_stats.get("deferred_groups", 0.0))
        stats["topology_dropped"] = float(
            topology_stats.get("dropped_topologies", 0))
        stats["topology_methods_dropped"] = float(
            topology_stats.get("dropped_methods", 0))
        stats["runtime_tensor_bytes"] = float(
            runtime_stats.get("tensor_bytes", 0))
        changed = (
            stats["dropped_groups"] > 0.0
            or stats["dropped_signatures"] > 0.0
            or stats["topology_dropped"] > 0.0
            or stats["runtime_tensor_bytes"] > 0.0
            or old_signature != desired_signature)
        stats["changed"] = 1.0 if changed else 0.0
        if changed:
            self._mode1_planned_floor_groups_precreated = False
            self._mode1_planned_floor_groups_precreated_signature = (
                desired_signature)
            logger.info(
                "Mode1 planned floor groups pruned for current step: "
                "rank=%s target_floor=%s reason=%s desired_groups=%s "
                "old_signature=%s new_signature=%s dropped_groups=%.0f "
                "deferred_groups=%.0f dropped_signatures=%.0f "
                "topology_dropped=%.0f topology_methods_dropped=%.0f "
                "runtime_tensor_bytes=%.0f force_destroy=%s drop_ms=%.2f",
                self.rank,
                target_floor,
                reason,
                [list(ranks) for ranks in sorted(desired_groups,
                                                 key=lambda ranks:
                                                 (-len(ranks), ranks))],
                [list(ranks) for ranks in old_signature]
                if old_signature else [],
                [list(ranks) for ranks in desired_signature],
                stats["dropped_groups"],
                stats["deferred_groups"],
                stats["dropped_signatures"],
                stats["topology_dropped"],
                stats["topology_methods_dropped"],
                stats["runtime_tensor_bytes"],
                force_destroy,
                stats["drop_ms"],
            )
        return stats

    def prune_mode1_natural_runtime_for_kv_resize(
            self, target_floor: int) -> dict[str, float]:
        """Release non-resident natural-mode shrink runtime before KV sizing.

        Natural policy does not intentionally keep floor8/floor4 communicators
        across steps.  After a real shrink, however, MoE comm methods may still
        hold dispatcher/runtime workspace references until the next topology is
        materialized.  Drop those transient references before planning the next
        step's KV cache so floor8 can use the true natural-mode budget instead
        of inheriting floor4 warmup residue.
        """
        stats: dict[str, float] = {
            "changed": 0.0,
            "dropped_groups": 0.0,
            "deferred_groups": 0.0,
            "dropped_signatures": 0.0,
            "topology_dropped": 0.0,
            "topology_methods_dropped": 0.0,
            "runtime_tensor_bytes": 0.0,
            "runtime_tensors": 0.0,
            "total_ms": 0.0,
        }
        start_t = time.perf_counter()
        try:
            target_floor = int(target_floor)
        except (TypeError, ValueError):
            return stats
        target_policy = os.getenv("VLLM_ASCEND_SHRINK_AWARE_TARGET_POLICY",
                                  "natural").strip().lower()
        if target_policy in ("planned", "fixed", "plan"):
            return stats
        if not self._is_mode1_low_floor_elastic_runtime():
            return stats
        if not torch.distributed.is_initialized():
            return stats

        world_size = int(torch.distributed.get_world_size())
        full_world_ranks = tuple(
            self._original_local_elastic_group_ranks(world_size))
        force_destroy = _env_flag(
            "VLLM_ASCEND_MODE1_NATURAL_FORCE_DESTROY_FLOOR_GROUPS_ON_KV_RESIZE",
            "1")
        drop_stats = self._drop_stale_cached_elastic_parallel_groups(
            full_world_ranks,
            reason=f"natural_kv_resize_prepare:target_floor={target_floor}",
            force_destroy=force_destroy,
        )
        runtime_stats = self._release_mode1_moe_comm_runtime_state(
            f"natural_kv_resize_prepare:target_floor={target_floor}")
        topology_stats = prune_moe_comm_method_topology_cache(
            {full_world_ranks})
        gc.collect()
        try:
            torch.npu.empty_cache()
            torch.npu.synchronize()
        except Exception:
            logger.exception(
                "Failed to empty/sync NPU cache after natural KV prune")

        stats["dropped_groups"] = float(drop_stats.get("dropped_groups", 0.0))
        stats["deferred_groups"] = float(drop_stats.get("deferred_groups", 0.0))
        stats["dropped_signatures"] = float(
            drop_stats.get("dropped_signatures", 0.0))
        stats["topology_dropped"] = float(
            topology_stats.get("dropped_topologies", 0))
        stats["topology_methods_dropped"] = float(
            topology_stats.get("dropped_methods", 0))
        stats["runtime_tensor_bytes"] = float(
            runtime_stats.get("tensor_bytes", 0))
        stats["runtime_tensors"] = float(runtime_stats.get("tensors", 0))
        stats["total_ms"] = (time.perf_counter() - start_t) * 1000.0
        changed = (
            stats["dropped_groups"] > 0.0
            or stats["dropped_signatures"] > 0.0
            or stats["topology_dropped"] > 0.0
            or stats["runtime_tensor_bytes"] > 0.0)
        stats["changed"] = 1.0 if changed else 0.0
        logger.info(
            "Mode1 natural KV resize runtime prune summary: rank=%s "
            "target_floor=%s changed=%.0f dropped_groups=%.0f "
            "deferred_groups=%.0f dropped_signatures=%.0f "
            "topology_dropped=%.0f topology_methods_dropped=%.0f "
            "runtime_tensors=%.0f runtime_tensor_bytes=%.0f "
            "force_destroy=%s total_ms=%.2f",
            self.rank,
            target_floor,
            stats["changed"],
            stats["dropped_groups"],
            stats["deferred_groups"],
            stats["dropped_signatures"],
            stats["topology_dropped"],
            stats["topology_methods_dropped"],
            stats["runtime_tensors"],
            stats["runtime_tensor_bytes"],
            force_destroy,
            stats["total_ms"],
        )
        return stats

    def _precreate_mode1_planned_floor_groups_before_kv_cache(
            self, prefill_expert_slots: bool = True) -> dict[str, float]:
        stats: dict[str, float] = {
            "stage_groups": 0.0,
            "rebuild_ms": 0.0,
            "comm_cache_ms": 0.0,
            "comm_cache_layers": 0.0,
            "dispatch_warmup_ms": 0.0,
            "dispatch_warmup_groups": 0.0,
            "refresh_ms": 0.0,
            "expert_prefill_ms": 0.0,
            "expert_prefill_slots": 0.0,
            "expert_prefill_updates": 0.0,
            "barrier_ms": 0.0,
            "total_ms": 0.0,
        }
        if not self._mode1_precreate_planned_floor_groups_enabled():
            return stats
        if not torch.distributed.is_initialized():
            return stats

        import vllm.distributed.parallel_state as vllm_ps
        import vllm_ascend.distributed.parallel_state as ascend_ps

        world_group = get_world_group()
        world_size = int(torch.distributed.get_world_size())
        current_rank = int(torch.distributed.get_rank())
        original_groups = {
            tuple(map(int, ranks))
            for ranks in self._build_original_ep_group_ranks(world_size)
        }
        planned_ranks = [
            ranks for ranks in self._mode1_fixed_topology_reusable_group_ranks()
            if ranks and ranks not in original_groups
        ]
        planned_ranks = sorted(set(planned_ranks), key=lambda ranks:
                               (-len(ranks), ranks))
        planned_signature = tuple(planned_ranks)
        old_signature = getattr(
            self, "_mode1_planned_floor_groups_precreated_signature", None)
        if (getattr(self, "_mode1_planned_floor_groups_precreated", False)
                and old_signature == planned_signature):
            return stats
        if (getattr(self, "_mode1_planned_floor_groups_precreated", False)
                and old_signature != planned_signature):
            self._prune_mode1_planned_floor_groups_for_current_step(
                len(planned_ranks[-1]) if planned_ranks else world_size,
                "planned_floor_precreate_signature_change",
            )
        if not planned_ranks:
            self._mode1_planned_floor_groups_precreated = True
            self._mode1_planned_floor_groups_precreated_signature = (
                planned_signature)
            return stats

        start_t = time.perf_counter()
        backend = torch.distributed.get_backend(world_group.device_group)
        try:
            barrier_start_t = time.perf_counter()
            torch.distributed.barrier(group=world_group.cpu_group)
            stats["barrier_ms"] += (
                time.perf_counter() - barrier_start_t) * 1000.0

            rebuild_start_t = time.perf_counter()
            with set_current_vllm_config(self.vllm_config):
                for ranks in planned_ranks:
                    elastic_group_ranks = [list(ranks)]
                    if current_rank in ranks:
                        self._rebuild_group(vllm_ps, "_DP",
                                            elastic_group_ranks, world_group,
                                            backend, "dp")
                        self._rebuild_group(vllm_ps, "_EP",
                                            elastic_group_ranks, world_group,
                                            backend, "ep")
                        self._rebuild_group(ascend_ps, "_MC2",
                                            elastic_group_ranks, world_group,
                                            backend, "mc2")
                        if self._mode1_precreate_comm_cache_enabled():
                            cache_start_t = time.perf_counter()
                            stats["comm_cache_layers"] += (
                                self._materialize_mode1_moe_comm_cache_for_current_groups(
                                    list(ranks)))
                            stats["comm_cache_ms"] += (
                                time.perf_counter() -
                                cache_start_t) * 1000.0
                        warmup_start_t = time.perf_counter()
                        if self._warmup_mode1_precreated_moe_comm_cache(
                                list(ranks)):
                            stats["dispatch_warmup_groups"] += 1.0
                            stats["dispatch_warmup_ms"] += (
                                time.perf_counter() -
                                warmup_start_t) * 1000.0
                    else:
                        self._advance_group_creation_sequence_for_non_member(
                            elastic_group_ranks, backend, "dp")
                        self._advance_group_creation_sequence_for_non_member(
                            elastic_group_ranks, backend, "ep")
                        self._advance_group_creation_sequence_for_non_member(
                            elastic_group_ranks, backend, "mc2")
                    stats["stage_groups"] += 1.0

                self._rebuild_group(
                    vllm_ps, "_DP",
                    self._build_original_dp_group_ranks(world_size),
                    world_group, backend, "dp")
                self._rebuild_group(
                    vllm_ps, "_EP",
                    self._build_original_ep_group_ranks(world_size),
                    world_group, backend, "ep")
                self._rebuild_group(
                    ascend_ps, "_MC2",
                    self._build_original_mc2_group_ranks(world_size),
                    world_group, backend, "mc2")
            stats["rebuild_ms"] = (
                time.perf_counter() - rebuild_start_t) * 1000.0

            refresh_start_t = time.perf_counter()
            self._reset_elastic_moe_comm_setup_cache(
                f"before_precreated_planned_floor_restore world_size={world_size}"
            )
            original_local_ranks = self._original_local_elastic_group_ranks(
                world_size)
            with set_current_vllm_config(self.vllm_config):
                self._refresh_elastic_parallel_state(
                    original_local_ranks, world_group)
            stale_drop_stats = self._drop_stale_cached_elastic_parallel_groups(
                tuple(original_local_ranks))
            if (stale_drop_stats.get("dropped_groups", 0.0) > 0.0
                    or stale_drop_stats.get("dropped_signatures", 0.0) > 0.0):
                logger.info(
                    "Mode1 planned floor precreate stale group release summary: "
                    "rank=%s dropped_groups=%.0f deferred_groups=%.0f "
                    "dropped_signatures=%.0f cleanup_ms=%.2f total_ms=%.2f",
                    self.rank,
                    stale_drop_stats.get("dropped_groups", 0.0),
                    stale_drop_stats.get("deferred_groups", 0.0),
                    stale_drop_stats.get("dropped_signatures", 0.0),
                    stale_drop_stats.get("cleanup_ms", 0.0),
                    stale_drop_stats.get("total_ms", 0.0),
                )
            stats["refresh_ms"] = (
                time.perf_counter() - refresh_start_t) * 1000.0
            self.elastic_parallel_detached = False
            self._elastic_current_active_ranks = list(original_local_ranks)

            if (prefill_expert_slots
                    and self._mode1_prefill_planned_expert_slots_enabled()):
                first_stage_ranks = list(planned_ranks[0])
                prefill_stats = (
                    self._prefill_mode1_planned_expert_slots_for_active_ranks(
                        first_stage_ranks,
                        world_group,
                        reason="precreate_planned_floor_groups_before_kv",
                    ))
                stats["expert_prefill_ms"] = float(
                    prefill_stats.get("total_ms", 0.0))
                stats["expert_prefill_slots"] = float(
                    prefill_stats.get("commit_slots", 0.0))
                stats["expert_prefill_updates"] = float(
                    prefill_stats.get("commit_updates", 0.0))

            if torch.npu.is_available():
                torch.npu.synchronize()
            self._release_mode1_moe_comm_runtime_state(
                "after_precreate_planned_floor_groups_before_kv")
            gc.collect()
            if torch.npu.is_available():
                torch.npu.empty_cache()
                torch.npu.synchronize()
            barrier_start_t = time.perf_counter()
            torch.distributed.barrier(group=world_group.cpu_group)
            stats["barrier_ms"] += (
                time.perf_counter() - barrier_start_t) * 1000.0
            stats["total_ms"] = (time.perf_counter() - start_t) * 1000.0
            self._mode1_planned_floor_groups_precreated = True
            self._mode1_planned_floor_groups_precreated_signature = (
                planned_signature)
            logger.info(
                "Mode1 planned floor groups precreated before KV sizing: "
                "rank=%s planned_ranks=%s stage_groups=%.0f rebuild_ms=%.2f "
                "comm_cache_layers=%.0f comm_cache_ms=%.2f "
                "dispatch_warmup_groups=%.0f dispatch_warmup_ms=%.2f "
                "refresh_ms=%.2f expert_prefill_ms=%.2f "
                "expert_prefill_slots=%.0f expert_prefill_updates=%.0f "
                "barrier_ms=%.2f total_ms=%.2f",
                self.rank,
                [list(ranks) for ranks in planned_ranks],
                stats["stage_groups"],
                stats["rebuild_ms"],
                stats["comm_cache_layers"],
                stats["comm_cache_ms"],
                stats["dispatch_warmup_groups"],
                stats["dispatch_warmup_ms"],
                stats["refresh_ms"],
                stats.get("expert_prefill_ms", 0.0),
                stats.get("expert_prefill_slots", 0.0),
                stats.get("expert_prefill_updates", 0.0),
                stats["barrier_ms"],
                stats["total_ms"],
            )
        except Exception:
            logger.exception(
                "Mode1 planned floor group precreation before KV sizing failed: "
                "rank=%s planned_ranks=%s",
                self.rank,
                [list(ranks) for ranks in planned_ranks],
            )
            raise
        return stats

    def _materialize_mode1_moe_comm_cache_for_current_groups(
            self, active_ranks: list[int]) -> int:
        """Warm topology-keyed MoE comm cache for the currently installed groups."""
        model_runner = getattr(self, "model_runner", None)
        model = getattr(model_runner, "model", None) if model_runner else None
        if model is None:
            return 0

        current_rank = int(torch.distributed.get_rank())
        if current_rank not in active_ranks:
            return 0
        dp_group = get_dp_group()
        ep_group = get_ep_group()
        mc2_group = get_mc2_group()
        refreshed_layers = 0
        for module in model.modules():
            if not _is_ascend_fused_moe_module(module):
                continue
            if int(getattr(module, "elastic_execution_mode", 0) or 0) != 1:
                continue

            moe_parallel_config = getattr(module, "moe_parallel_config", None)
            moe_config = getattr(module, "moe_config", None)
            if moe_parallel_config is None or moe_config is None:
                continue

            old_values = {
                "num_experts": getattr(moe_config, "num_experts", None),
                "num_local_experts": getattr(moe_config, "num_local_experts",
                                             None),
                "dp_size": getattr(moe_parallel_config, "dp_size", None),
                "dp_rank": getattr(moe_parallel_config, "dp_rank", None),
                "ep_size": getattr(moe_parallel_config, "ep_size", None),
                "ep_rank": getattr(moe_parallel_config, "ep_rank", None),
                "module_ep_group": getattr(module, "ep_group", None),
                "moe_parallel_config": getattr(moe_config,
                                               "moe_parallel_config", None),
                "dp_group": getattr(moe_config, "dp_group", None),
                "ep_group": getattr(moe_config, "ep_group", None),
                "mc2_group": getattr(moe_config, "mc2_group", None),
                "model_type": getattr(moe_config, "model_type", None),
            }
            try:
                logical_num_experts = int(
                    getattr(module, "elastic_original_num_experts",
                            getattr(moe_config, "num_experts", 0)) or 0)
                target_local_experts = max(
                    1,
                    (logical_num_experts + len(active_ranks) - 1) //
                    len(active_ranks))
                moe_config.num_experts = logical_num_experts
                moe_config.num_local_experts = target_local_experts
                moe_parallel_config.dp_size = dp_group.world_size
                moe_parallel_config.dp_rank = dp_group.rank_in_group
                moe_parallel_config.ep_size = ep_group.world_size
                moe_parallel_config.ep_rank = ep_group.rank_in_group
                module.ep_group = ep_group
                moe_config.moe_parallel_config = moe_parallel_config
                moe_config.model_type = getattr(module, "model_type",
                                                getattr(moe_config,
                                                        "model_type", ""))
                moe_config.dp_group = dp_group
                moe_config.ep_group = ep_group
                moe_config.mc2_group = mc2_group
                setup_moe_comm_method(moe_config)
                refreshed_layers += 1
            finally:
                moe_config.num_experts = old_values["num_experts"]
                moe_config.num_local_experts = old_values[
                    "num_local_experts"]
                moe_parallel_config.dp_size = old_values["dp_size"]
                moe_parallel_config.dp_rank = old_values["dp_rank"]
                moe_parallel_config.ep_size = old_values["ep_size"]
                moe_parallel_config.ep_rank = old_values["ep_rank"]
                module.ep_group = old_values["module_ep_group"]
                moe_config.moe_parallel_config = old_values[
                    "moe_parallel_config"]
                moe_config.dp_group = old_values["dp_group"]
                moe_config.ep_group = old_values["ep_group"]
                moe_config.mc2_group = old_values["mc2_group"]
                if old_values["model_type"] is None:
                    try:
                        delattr(moe_config, "model_type")
                    except AttributeError:
                        pass
                else:
                    moe_config.model_type = old_values["model_type"]
        return refreshed_layers

    def _warmup_mode1_precreated_moe_comm_cache(
            self, active_ranks: list[int]) -> bool:
        """Optionally materialize MC2 dispatcher workspace before KV sizing.

        The planned precreate path primarily needs the DP/EP/MC2 group handles
        to stay resident and reusable across steps. Dispatcher/workspace warmup
        is heavier and should be enabled only when the KV budget also reserves
        enough headroom for that steady-state footprint.
        """
        if not _env_flag(
                "VLLM_ASCEND_MODE1_PARITY_PRECREATE_DISPATCH_WARMUP", "0"):
            return False
        current_rank = int(torch.distributed.get_rank())
        if current_rank not in active_ranks:
            return False

        model_runner = getattr(self, "model_runner", None)
        model = getattr(model_runner, "model", None) if model_runner else None
        if model is None:
            return False

        target_module = None
        for module in model.modules():
            if not _is_ascend_fused_moe_module(module):
                continue
            if int(getattr(module, "elastic_execution_mode", 0) or 0) != 1:
                continue
            if getattr(module, "moe_config", None) is None:
                continue
            if getattr(module, "moe_parallel_config", None) is None:
                continue
            target_module = module
            break
        if target_module is None:
            return False

        try:
            from vllm.config import CUDAGraphMode
            from vllm.forward_context import BatchDescriptor, get_forward_context
            from vllm_ascend.ascend_forward_context import (
                MoECommType, set_ascend_forward_context)
        except Exception:
            logger.exception(
                "Mode1 planned floor precreate dispatch warmup import failed: rank=%s",
                self.rank)
            raise

        dp_group = get_dp_group()
        ep_group = get_ep_group()
        mc2_group = get_mc2_group()
        moe_config = target_module.moe_config
        moe_parallel_config = target_module.moe_parallel_config
        old_values = {
            "num_experts": getattr(moe_config, "num_experts", None),
            "num_local_experts": getattr(moe_config, "num_local_experts",
                                         None),
            "dp_size": getattr(moe_parallel_config, "dp_size", None),
            "dp_rank": getattr(moe_parallel_config, "dp_rank", None),
            "ep_size": getattr(moe_parallel_config, "ep_size", None),
            "ep_rank": getattr(moe_parallel_config, "ep_rank", None),
            "module_ep_group": getattr(target_module, "ep_group", None),
            "moe_parallel_config": getattr(moe_config,
                                           "moe_parallel_config", None),
            "dp_group": getattr(moe_config, "dp_group", None),
            "ep_group": getattr(moe_config, "ep_group", None),
            "mc2_group": getattr(moe_config, "mc2_group", None),
            "model_type": getattr(moe_config, "model_type", None),
        }

        hidden_states = None
        topk_ids = None
        topk_weights = None
        row_idx = None
        expert_map = None
        try:
            logical_num_experts = int(
                getattr(target_module, "elastic_original_num_experts",
                        getattr(moe_config, "num_experts", 0)) or 0)
            target_local_experts = max(
                1,
                (logical_num_experts + len(active_ranks) - 1) //
                len(active_ranks))
            moe_config.num_experts = logical_num_experts
            moe_config.num_local_experts = target_local_experts
            moe_parallel_config.dp_size = dp_group.world_size
            moe_parallel_config.dp_rank = dp_group.rank_in_group
            moe_parallel_config.ep_size = ep_group.world_size
            moe_parallel_config.ep_rank = ep_group.rank_in_group
            target_module.ep_group = ep_group
            moe_config.moe_parallel_config = moe_parallel_config
            moe_config.model_type = getattr(target_module, "model_type",
                                            getattr(moe_config, "model_type",
                                                    ""))
            moe_config.dp_group = dp_group
            moe_config.ep_group = ep_group
            moe_config.mc2_group = mc2_group
            setup_moe_comm_method(moe_config)

            moe_comm_method = get_moe_comm_method(MoECommType.MC2)
            token_dispatcher = getattr(moe_comm_method, "token_dispatcher",
                                       None)
            if token_dispatcher is None:
                return False

            num_tokens = max(
                1,
                int(
                    os.getenv(
                        "VLLM_ASCEND_MODE1_PARITY_PRECREATE_DISPATCH_WARMUP_TOKENS",
                        "32")))
            hidden_size = int(
                getattr(moe_config, "hidden_dim",
                        getattr(target_module, "hidden_size", 0)) or 0)
            top_k = int(
                getattr(target_module, "top_k",
                        getattr(moe_config, "experts_per_token", 0)) or 0)
            if hidden_size <= 0 or top_k <= 0 or logical_num_experts <= 0:
                logger.info(
                    "Mode1 planned floor precreate dispatch warmup skipped: "
                    "rank=%s active_ranks=%s reason=invalid_shape "
                    "hidden_size=%s top_k=%s num_experts=%s",
                    self.rank,
                    active_ranks,
                    hidden_size,
                    top_k,
                    logical_num_experts,
                )
                return False

            device = getattr(model_runner, "device", torch.device("npu"))
            dtype = getattr(model_runner, "dtype", self.model_config.dtype)
            hidden_states = torch.zeros((num_tokens, hidden_size),
                                        dtype=dtype,
                                        device=device)
            topk_ids = (torch.arange(num_tokens * top_k,
                                     dtype=torch.int32,
                                     device=device) %
                        logical_num_experts).reshape(num_tokens,
                                                     top_k).contiguous()
            topk_weights = torch.full((num_tokens, top_k),
                                      1.0 / float(top_k),
                                      dtype=torch.float32,
                                      device=device)
            row_idx = torch.arange(0,
                                   num_tokens * top_k,
                                   dtype=torch.int32,
                                   device=device).view(top_k, -1).permute(
                                       1, 0).contiguous()
            expert_map = torch.arange(logical_num_experts,
                                      dtype=torch.int32,
                                      device=device)
            # set_forward_context() still uses the global DP rank from the
            # vLLM parallel config, even while this warmup temporarily points
            # MoE communication at a planned floor group. Keep this vector in
            # global-DP coordinates so ranks like 8..15 do not index past a
            # floor8/floor4-sized tensor.
            global_dp_size = max(
                int(getattr(self.parallel_config, "data_parallel_size", 0)
                    or 0),
                current_rank + 1,
                max((int(rank) for rank in active_ranks), default=0) + 1,
            )
            num_tokens_across_dp = torch.zeros((global_dp_size, ),
                                               dtype=torch.int64)
            for rank in active_ranks:
                if 0 <= int(rank) < global_dp_size:
                    num_tokens_across_dp[int(rank)] = num_tokens

            logger.info(
                "Mode1 planned floor precreate dispatch warmup start: "
                "rank=%s active_ranks=%s tokens=%s hidden_size=%s top_k=%s "
                "num_experts=%s ep_size=%s",
                self.rank,
                active_ranks,
                num_tokens,
                hidden_size,
                top_k,
                logical_num_experts,
                ep_group.world_size,
            )
            with set_ascend_forward_context(
                    None,
                    self.vllm_config,
                    num_tokens=num_tokens,
                    num_tokens_across_dp=num_tokens_across_dp,
                    with_prefill=False,
                    reserved_mc2_mask=getattr(model_runner,
                                              "reserved_mc2_mask", None),
                    moe_comm_type=MoECommType.MC2,
                    num_actual_tokens=num_tokens,
                    aclgraph_runtime_mode=CUDAGraphMode.NONE,
                    batch_descriptor=BatchDescriptor(num_tokens=num_tokens,
                                                     uniform_decode=False),
                    prefetch_stream=getattr(model_runner, "prefetch_stream",
                                            None),
                    moe_prefetch_stream=getattr(model_runner,
                                                "moe_prefetch_stream", None),
                    model_instance=model):
                forward_context = get_forward_context()
                forward_context.elastic_debug_info = {
                    "layer_idx": int(getattr(target_module, "layer_idx", -1)),
                    "phase": "precreate_planned_floor_dispatch_warmup",
                }
                old_num_experts = int(
                    getattr(token_dispatcher, "num_experts", 0))
                old_top_k = int(getattr(token_dispatcher, "top_k", 0))
                old_expert_token_nums_type = int(
                    getattr(token_dispatcher, "expert_token_nums_type", 0))
                token_dispatcher.num_experts = logical_num_experts
                token_dispatcher.top_k = top_k
                if hasattr(token_dispatcher, "expert_token_nums_type"):
                    token_dispatcher.expert_token_nums_type = 1
                try:
                    dispatch_results = token_dispatcher.token_dispatch(
                        hidden_states=hidden_states,
                        topk_weights=topk_weights,
                        topk_ids=topk_ids,
                        row_idx=row_idx,
                        expert_map=expert_map,
                        log2phy=None,
                        global_redundant_expert_num=0,
                        shared_experts=None,
                        quantized_x_for_share=None,
                        dynamic_scale_for_share=None,
                        mc2_mask=getattr(forward_context, "mc2_mask", None),
                        apply_router_weight_on_input=False,
                        with_quant=False)
                    combined = token_dispatcher.token_combine(
                        dispatch_results["hidden_states"])
                    del combined, dispatch_results
                finally:
                    token_dispatcher.num_experts = old_num_experts
                    token_dispatcher.top_k = old_top_k
                    if hasattr(token_dispatcher, "expert_token_nums_type"):
                        token_dispatcher.expert_token_nums_type = (
                            old_expert_token_nums_type)
                    for attr in (
                            "output",
                            "assist_info_for_combine",
                            "ep_recv_counts",
                            "topk_ids",
                            "topk_weights",
                            "mc2_mask",
                            "expert_map",
                            "shared_experts",
                            "shared_act",
                    ):
                        if hasattr(token_dispatcher, attr):
                            setattr(token_dispatcher, attr, None)

            logger.info(
                "Mode1 planned floor precreate dispatch warmup complete: "
                "rank=%s active_ranks=%s tokens=%s ep_size=%s",
                self.rank,
                active_ranks,
                num_tokens,
                ep_group.world_size,
            )
            return True
        finally:
            if hidden_states is not None:
                del hidden_states
            if topk_ids is not None:
                del topk_ids
            if topk_weights is not None:
                del topk_weights
            if row_idx is not None:
                del row_idx
            if expert_map is not None:
                del expert_map
            moe_config.num_experts = old_values["num_experts"]
            moe_config.num_local_experts = old_values["num_local_experts"]
            moe_parallel_config.dp_size = old_values["dp_size"]
            moe_parallel_config.dp_rank = old_values["dp_rank"]
            moe_parallel_config.ep_size = old_values["ep_size"]
            moe_parallel_config.ep_rank = old_values["ep_rank"]
            target_module.ep_group = old_values["module_ep_group"]
            moe_config.moe_parallel_config = old_values["moe_parallel_config"]
            moe_config.dp_group = old_values["dp_group"]
            moe_config.ep_group = old_values["ep_group"]
            moe_config.mc2_group = old_values["mc2_group"]
            if old_values["model_type"] is None:
                try:
                    delattr(moe_config, "model_type")
                except AttributeError:
                    pass
            else:
                moe_config.model_type = old_values["model_type"]
            self._release_mode1_moe_comm_runtime_state(
                "after_precreate_planned_floor_dispatch_warmup:"
                f"active_ranks={tuple(active_ranks)}")
            gc.collect()
            if torch.npu.is_available():
                torch.npu.empty_cache()
                torch.npu.synchronize()

    def _is_mode1_fixed_topology_reusable_group(
            self, group_kind: str, group_ranks: tuple[int, ...]) -> bool:
        if not self._mode1_fixed_topology_group_reuse_enabled():
            return False
        if group_kind not in ("dp", "ep", "mc2"):
            return False
        if not group_ranks:
            return False
        if not torch.distributed.is_initialized():
            return False
        return group_ranks in self._mode1_fixed_topology_reusable_group_ranks()

    def _should_keep_mode1_fixed_topology_stale_group(
            self, group_kind: str, stale_group_ranks: tuple[int, ...],
            keep_group_ranks: tuple[int, ...]) -> bool:
        if not self._is_mode1_fixed_topology_reusable_group(
                group_kind, stale_group_ranks):
            return False
        if not torch.distributed.is_initialized():
            return False
        world_size = int(torch.distributed.get_world_size())
        full_world_ranks = tuple(
            self._original_local_elastic_group_ranks(world_size))
        if stale_group_ranks == full_world_ranks:
            return True
        if keep_group_ranks == full_world_ranks:
            # Fixed-topology targeting should not imply cross-step residency
            # for planned floor-8/floor-4 communicators. Keep them across a
            # full-world restore only when the experiment explicitly enables
            # planned floor-group cache residency.
            return self._mode1_planned_floor_group_cache_enabled()
        return True

    def _should_cache_elastic_parallel_group(self, attr_name: str) -> bool:
        group_kind = self._normalize_elastic_parallel_group_kind(attr_name)
        if group_kind != "mc2":
            return True
        if (_env_flag("VLLM_ASCEND_MODE1_PARITY_SINGLE_LIVE_MC2_GROUP", "0")
                and self._has_mode1_lightweight_parity_module()):
            # Diagnostic escape hatch: force the old custom mode1 path to keep
            # a single live MC2 communicator. This is no longer the default
            # because repeatedly recreating MC2 diverges from native/common
            # mode1 and can leak HCCL tiling resources across training steps.
            return False
        # Native/common mode1 does not repeatedly allocate a fresh MC2 resource
        # for the same active-rank signature. The old custom path was
        # recreating MC2 on every restore/shrink cycle; step 1 could run, but
        # step 2 then failed when aclnnMoeDistributeDispatchV4 asked HCCL for
        # another ~1.56 GiB communication resource. Cache MC2 by default for
        # custom mode1 parity so the resource lifecycle matches the native
        # execution pattern. Keep an opt-out for targeted diagnostics.
        return os.getenv(
            "VLLM_ASCEND_DISABLE_ELASTIC_MC2_GROUP_CACHE",
            "0").lower() not in ("1", "true", "yes", "on")

    def _should_cache_elastic_parallel_group_ranks(
            self, attr_name: str, group_ranks: tuple[int, ...]) -> bool:
        if not self._should_cache_elastic_parallel_group(attr_name):
            return False
        group_kind = self._normalize_elastic_parallel_group_kind(attr_name)
        if group_kind not in ("dp", "ep", "mc2"):
            return True
        if not group_ranks:
            return False
        if not self._is_mode1_low_floor_elastic_runtime():
            return True
        if self._mode1_fixed_topology_group_reuse_enabled():
            return self._is_mode1_fixed_topology_reusable_group(
                group_kind, group_ranks)
        if _env_flag("VLLM_ASCEND_MODE1_PARITY_CACHE_FLOOR_GROUPS", "0"):
            return True
        if not torch.distributed.is_initialized():
            return True
        world_size = int(torch.distributed.get_world_size())
        full_world_ranks = tuple(
            self._original_local_elastic_group_ranks(world_size))
        return group_ranks == full_world_ranks


    def _should_keep_stale_mc2_cache_for_custom_mode1_parity(
            self, stale_group_ranks: tuple[int, ...],
            keep_group_ranks: tuple[int, ...]) -> bool:
        if stale_group_ranks == keep_group_ranks:
            return False
        if not stale_group_ranks:
            return False
        if self._has_mode4_remote_npu_lightweight_module():
            return (
                _env_flag("VLLM_ASCEND_MODE4_KEEP_STALE_MC2_GROUP_CACHE", "0")
                and not _env_flag("VLLM_ASCEND_DISABLE_ELASTIC_MC2_GROUP_CACHE",
                                  "0"))
        # Keep this disabled by default. The old custom mode1 path should reuse
        # the current full-world MC2 communicator across restore/resume, but a
        # stale floor8 communicator must not remain resident into the next
        # step's KV-cache materialization. Holding both the 16-rank and 8-rank
        # MC2 HCCL resources raised non_torch memory by several GiB and made
        # step-2 resume(kv_cache) OOM at native KV budget.
        if not _env_flag("VLLM_ASCEND_MODE1_PARITY_KEEP_MC2_GROUP_CACHE",
                         "0"):
            return False
        if _env_flag("VLLM_ASCEND_MODE1_PARITY_SINGLE_LIVE_MC2_GROUP", "0"):
            return False
        if _env_flag("VLLM_ASCEND_DISABLE_ELASTIC_MC2_GROUP_CACHE", "0"):
            return False
        return self._has_mode1_lightweight_parity_module()

    def _should_keep_mode1_floor4_group_cache(
            self, group_kind: str, cached_group_ranks: tuple[int, ...],
            keep_group_ranks: tuple[int, ...]) -> bool:
        if not _env_flag("VLLM_ASCEND_MODE1_PARITY_KEEP_FLOOR4_GROUP_CACHE",
                         "0"):
            return False
        keep_kinds = os.getenv(
            "VLLM_ASCEND_MODE1_PARITY_KEEP_FLOOR4_GROUP_KINDS",
            "dp,ep")
        keep_kind_set = {
            kind.strip().lower()
            for kind in keep_kinds.split(",") if kind.strip()
        }
        if group_kind not in keep_kind_set:
            return False
        if group_kind not in ("dp", "ep", "mc2"):
            return False
        if len(cached_group_ranks) != 4:
            return False
        if cached_group_ranks == keep_group_ranks:
            return False
        # Limit this workaround to floor=2 mode1. Keeping the 8-rank floor
        # cache was already shown to break or destabilize the following dp=4
        # stage. Keeping floor4 MC2 by default is also unsafe: it keeps a large
        # HCCL workspace alive and can OOM the next DP/MC2 workspace allocation.
        # Use the kind allowlist above only for focused diagnostics.
        return self._has_mode1_lightweight_parity_module(max_floor=2)


    def _should_keep_stale_group_cache_for_custom_mode1_parity(
            self, group_kind: str, stale_group_ranks: tuple[int, ...],
            keep_group_ranks: tuple[int, ...]) -> bool:
        if stale_group_ranks == keep_group_ranks:
            return False
        if not stale_group_ranks:
            return False
        if self._should_keep_mode1_fixed_topology_stale_group(
                group_kind, stale_group_ranks, keep_group_ranks):
            return True
        if self._should_keep_mode1_floor4_group_cache(
                group_kind, stale_group_ranks, keep_group_ranks):
            return True
        if group_kind == "mc2":
            return self._should_keep_stale_mc2_cache_for_custom_mode1_parity(
                stale_group_ranks, keep_group_ranks)
        if self._has_mode4_remote_npu_lightweight_module():
            if group_kind == "dp":
                return _env_flag(
                    "VLLM_ASCEND_MODE4_KEEP_STALE_DP_GROUP_CACHE", "0")
            if group_kind == "ep":
                return _env_flag(
                    "VLLM_ASCEND_MODE4_KEEP_STALE_EP_GROUP_CACHE", "0")
            return False
        if group_kind != "ep":
            return False
        if not _env_flag("VLLM_ASCEND_MODE1_PARITY_KEEP_FULLWORLD_EP_CACHE",
                         "1"):
            return False
        if not self._is_mode1_low_floor_elastic_runtime():
            return False
        if not torch.distributed.is_initialized():
            return False
        world_size = int(torch.distributed.get_world_size())
        full_world_ranks = tuple(
            self._original_local_elastic_group_ranks(world_size))
        if stale_group_ranks != full_world_ranks:
            return False
        if keep_group_ranks == full_world_ranks:
            return False
        return True

    def _should_drop_old_floor_group_cache_after_mode1_shrink(
            self, group_kind: str, cached_group_ranks: tuple[int, ...],
            keep_group_ranks: tuple[int, ...]) -> bool:
        if not self._is_mode1_low_floor_elastic_runtime():
            return False
        if group_kind not in ("dp", "ep", "mc2"):
            return False
        if not cached_group_ranks or cached_group_ranks == keep_group_ranks:
            return False
        if not torch.distributed.is_initialized():
            return False
        world_size = int(torch.distributed.get_world_size())
        full_world_ranks = tuple(
            self._original_local_elastic_group_ranks(world_size))
        if cached_group_ranks == full_world_ranks:
            return False
        if self._should_keep_mode1_fixed_topology_stale_group(
                group_kind, cached_group_ranks, keep_group_ranks):
            return False
        if self._should_keep_mode1_floor4_group_cache(
                group_kind, cached_group_ranks, keep_group_ranks):
            return False
        return True


    def _drop_old_floor_cached_elastic_parallel_groups_after_mode1_shrink(
            self, keep_group_ranks: tuple[int, ...]) -> dict[str, float]:
        drop_start_t = time.perf_counter()
        cached_groups = self._get_cached_elastic_parallel_groups()
        seen_signatures = self._get_seen_elastic_parallel_group_signatures()
        dropped_groups = 0
        dropped_signatures = 0
        destroyed_group_ids: set[int] = set()
        destroy_ms = 0.0
        deferred_groups = 0
        cleanup_ms = 0.0
        cleanup_gc_ms = 0.0
        cleanup_empty_cache_ms = 0.0
        cleanup_sync_ms = 0.0
        cleanup_memory_log_ms = 0.0
        memory_log_ms = 0.0
        row_log_ms = 0.0
        signature_sweep_ms = 0.0
        release_ref_ms = 0.0
        registry_dropped_groups = 0
        verbose_drop_log = _env_flag(
            "VLLM_ASCEND_MODE1_PARITY_OLD_FLOOR_DROP_VERBOSE", "0")

        for attr_name, groups_by_ranks in cached_groups.items():
            group_kind = self._normalize_elastic_parallel_group_kind(attr_name)
            stale_floor_ranks = [
                tuple(int(rank) for rank in ranks)
                for ranks in list(groups_by_ranks.keys())
                if self._should_drop_old_floor_group_cache_after_mode1_shrink(
                    group_kind,
                    tuple(int(rank) for rank in ranks),
                    keep_group_ranks,
                )
            ]
            for ranks in stale_floor_ranks:
                group = groups_by_ranks.pop(ranks, None)
                if group is not None:
                    group_id = id(group)
                    released_group = group
                    if group_id not in destroyed_group_ids:
                        if group_kind == "mc2" and verbose_drop_log:
                            memory_log_start_t = time.perf_counter()
                            self._log_custom_mode1_worker_memory(
                                "before_old_floor_mc2_cache_drop_after_shrink",
                                f"stale_ranks={ranks} keep_ranks={keep_group_ranks}",
                            )
                            memory_log_ms += (
                                time.perf_counter() -
                                memory_log_start_t) * 1000.0
                        force_cleanup_after_release = (
                            self._should_force_cleanup_after_mode1_floor_group_release(
                                group_kind, ranks))
                        if self._should_defer_mode1_group_destroy(
                                group_kind,
                                ranks,
                                keep_group_ranks,
                                "old_floor_after_shrink",
                        ):
                            if self._quarantine_deferred_elastic_group(
                                    group,
                                    group_kind,
                                    ranks,
                                    "old_floor_after_shrink",
                            ):
                                deferred_groups += 1
                            destroyed_group_ids.add(group_id)
                            if force_cleanup_after_release:
                                cleanup_stats = self._cleanup_after_elastic_group_release(
                                    "mode1_old_floor_after_shrink",
                                    force=True)
                                cleanup_ms += float(
                                    cleanup_stats.get("total_ms", 0.0))
                                cleanup_gc_ms += float(
                                    cleanup_stats.get("gc_ms", 0.0))
                                cleanup_empty_cache_ms += float(
                                    cleanup_stats.get("empty_cache_ms", 0.0))
                                cleanup_sync_ms += float(
                                    cleanup_stats.get("sync_ms", 0.0))
                                cleanup_memory_log_ms += float(
                                    cleanup_stats.get("memory_log_ms", 0.0))
                        else:
                            if self._is_mode1_low_floor_elastic_runtime():
                                logger.warning(
                                    "Elastic mode1 low-floor group direct destroy fallback: "
                                    "rank=%s group_kind=%s group_ranks=%s keep_ranks=%s "
                                    "reason=old_floor_after_shrink defer_env=%s "
                                    "defer_full_restore=%s defer_pre_rebuild=%s "
                                    "defer_stash=%s defer_sizes=%s",
                                    self.rank,
                                    group_kind,
                                    ranks,
                                    keep_group_ranks,
                                    os.getenv("VLLM_ASCEND_MODE1_PARITY_DEFER_GROUP_DESTROY", "1"),
                                    os.getenv("VLLM_ASCEND_MODE1_PARITY_DEFER_GROUP_DESTROY_ON_FULL_RESTORE", "0"),
                                    os.getenv("VLLM_ASCEND_MODE1_PARITY_DEFER_GROUP_DESTROY_ON_PRE_REBUILD", "0"),
                                    os.getenv("VLLM_ASCEND_MODE1_PARITY_DEFER_GROUP_DESTROY_ON_STASH", "0"),
                                    os.getenv("VLLM_ASCEND_MODE1_PARITY_DEFER_DESTROY_FLOOR_GROUP_SIZES", "1,2,4,8"),
                                )
                            destroy_start_t = time.perf_counter()
                            group.destroy()
                            self._detach_destroyed_group_references(group)
                            self._mark_retired_elastic_group(group)
                            destroy_ms += (
                                time.perf_counter() -
                                destroy_start_t) * 1000.0
                            destroyed_group_ids.add(group_id)
                            if group_kind == "mc2":
                                cleanup_stats = self._cleanup_after_elastic_group_release(
                                    "mode1_old_floor_after_shrink",
                                    force=force_cleanup_after_release)
                                cleanup_ms += float(
                                    cleanup_stats.get("total_ms", 0.0))
                                cleanup_gc_ms += float(
                                    cleanup_stats.get("gc_ms", 0.0))
                                cleanup_empty_cache_ms += float(
                                    cleanup_stats.get("empty_cache_ms", 0.0))
                                cleanup_sync_ms += float(
                                    cleanup_stats.get("sync_ms", 0.0))
                                cleanup_memory_log_ms += float(
                                    cleanup_stats.get("memory_log_ms", 0.0))
                            elif force_cleanup_after_release:
                                cleanup_stats = self._cleanup_after_elastic_group_release(
                                    "mode1_old_floor_after_shrink",
                                    force=True)
                                cleanup_ms += float(
                                    cleanup_stats.get("total_ms", 0.0))
                                cleanup_gc_ms += float(
                                    cleanup_stats.get("gc_ms", 0.0))
                                cleanup_empty_cache_ms += float(
                                    cleanup_stats.get("empty_cache_ms", 0.0))
                                cleanup_sync_ms += float(
                                    cleanup_stats.get("sync_ms", 0.0))
                                cleanup_memory_log_ms += float(
                                    cleanup_stats.get("memory_log_ms", 0.0))
                        release_ref_start_t = time.perf_counter()
                        released_group = group
                        group = None
                        release_ref_ms += (
                            time.perf_counter() -
                            release_ref_start_t) * 1000.0
                    self._remove_elastic_parallel_group_registry_entry(
                        group_kind, ranks, released_group)
                    dropped_groups += 1
                if (group_kind, ranks) in seen_signatures:
                    seen_signatures.discard((group_kind, ranks))
                    dropped_signatures += 1
                if verbose_drop_log:
                    row_log_start_t = time.perf_counter()
                    logger.info(
                        "Elastic mode1 old floor group cache dropped after shrink: "
                        "rank=%s group_kind=%s keep_ranks=%s stale_ranks=%s",
                        self.rank, group_kind, keep_group_ranks, ranks)
                    row_log_ms += (
                        time.perf_counter() - row_log_start_t) * 1000.0

        signature_sweep_start_t = time.perf_counter()
        stale_signatures = [
            signature for signature in seen_signatures
            if self._should_drop_old_floor_group_cache_after_mode1_shrink(
                signature[0], signature[1], keep_group_ranks)
        ]
        for signature in stale_signatures:
            seen_signatures.discard(signature)
            dropped_signatures += 1
        signature_sweep_ms = (
            time.perf_counter() - signature_sweep_start_t) * 1000.0

        for group_kind, ranks, group in (
                self._iter_stale_elastic_parallel_group_registry_entries(
                    keep_group_ranks, old_floor_only=True)):
            stats = self._release_elastic_parallel_group_reference(
                group,
                group_kind,
                ranks,
                keep_group_ranks,
                "old_floor_after_shrink",
                destroyed_group_ids,
                "before_old_floor_registry_mc2_drop_after_shrink",
            )
            if stats.get("released_groups", 0.0) > 0.0:
                registry_dropped_groups += 1
                dropped_groups += 1
            deferred_groups += int(stats.get("deferred_groups", 0.0))
            destroy_ms += float(stats.get("destroy_ms", 0.0))
            cleanup_ms += float(stats.get("cleanup_ms", 0.0))
            cleanup_gc_ms += float(stats.get("cleanup_gc_ms", 0.0))
            cleanup_empty_cache_ms += float(
                stats.get("cleanup_empty_cache_ms", 0.0))
            cleanup_sync_ms += float(stats.get("cleanup_sync_ms", 0.0))
            cleanup_memory_log_ms += float(
                stats.get("cleanup_memory_log_ms", 0.0))
            seen_signatures.discard((group_kind, ranks))

        summary_log_ms = 0.0
        if ((dropped_groups > 0 or dropped_signatures > 0)
                and verbose_drop_log):
            summary_log_start_t = time.perf_counter()
            logger.info(
                "Elastic mode1 old floor group cache drop summary: "
                "rank=%s keep_ranks=%s dropped_groups=%s dropped_signatures=%s",
                self.rank, keep_group_ranks, dropped_groups,
                dropped_signatures)
            summary_log_ms = (
                time.perf_counter() - summary_log_start_t) * 1000.0

        total_ms = (time.perf_counter() - drop_start_t) * 1000.0
        return {
            "total_ms": total_ms,
            "destroy_ms": destroy_ms,
            "cleanup_ms": cleanup_ms,
            "cleanup_gc_ms": cleanup_gc_ms,
            "cleanup_empty_cache_ms": cleanup_empty_cache_ms,
            "cleanup_sync_ms": cleanup_sync_ms,
            "cleanup_memory_log_ms": cleanup_memory_log_ms,
            "memory_log_ms": memory_log_ms,
            "row_log_ms": row_log_ms,
            "summary_log_ms": summary_log_ms,
            "signature_sweep_ms": signature_sweep_ms,
            "release_ref_ms": release_ref_ms,
            "dropped_groups": float(dropped_groups),
            "registry_dropped_groups": float(registry_dropped_groups),
            "deferred_groups": float(deferred_groups),
            "dropped_signatures": float(dropped_signatures),
        }


    def _should_keep_stale_group_cache_for_mode3(
            self, group_kind: str, stale_group_ranks: tuple[int, ...],
            keep_group_ranks: tuple[int, ...]) -> bool:
        if stale_group_ranks == keep_group_ranks:
            return False
        if not stale_group_ranks:
            return False
        if not self._has_mode3_cross_layer_lightweight_module():
            return False
        if group_kind == "mc2":
            # MC2/HCCL resources are the only stale groups that have shown
            # large low-floor allocation pressure when they are rebuilt after
            # every rollout step. Keep them by default for mode3 stability and
            # performance; memory reclamation experiments can opt back into
            # dropping them explicitly.
            return not _env_flag(
                "VLLM_ASCEND_MODE3_DROP_STALE_MC2_GROUP_CACHE_AFTER_SHRINK",
                "0")
        if group_kind == "dp":
            # Do not keep stale DP/EP groups by default. After restore the next
            # step starts with full-world prefill/alltoall, and stale shrink
            # DP/EP handles are not on the hot MC2 path but can poison the
            # restored communication state.
            return _env_flag(
                "VLLM_ASCEND_MODE3_KEEP_STALE_DP_GROUP_CACHE", "0")
        if group_kind == "ep":
            return _env_flag(
                "VLLM_ASCEND_MODE3_KEEP_STALE_EP_GROUP_CACHE", "0")
        return False

    def _should_keep_stale_group_cache_for_mode5(
            self, group_kind: str, stale_group_ranks: tuple[int, ...],
            keep_group_ranks: tuple[int, ...]) -> bool:
        if stale_group_ranks == keep_group_ranks:
            return False
        if not stale_group_ranks:
            return False
        if not self._has_mode4_remote_npu_lightweight_module():
            return False
        execution_mode = int(envs_ascend.VLLM_ASCEND_ELASTIC_EXECUTION_MODE)
        if execution_mode != 5:
            return False
        if group_kind == "dp":
            return _env_flag(
                "VLLM_ASCEND_MODE5_KEEP_STALE_DP_GROUP_CACHE", "0")
        if group_kind == "ep":
            return _env_flag(
                "VLLM_ASCEND_MODE5_KEEP_STALE_EP_GROUP_CACHE", "0")
        if group_kind == "mc2":
            return _env_flag(
                "VLLM_ASCEND_MODE5_KEEP_STALE_MC2_GROUP_CACHE", "0")
        # By default mode5 only keeps the current active groups plus the
        # full-world group used by remote-cache ownership/service metadata.
        # The keep-stale knobs above are experimental escape hatches for
        # testing whether cross-step communicator reuse can reduce shrink
        # rebuild cost without destabilizing the runtime.
        return False

    def _release_mode5_cache_rank_runtime_state(self) -> None:
        if int(envs_ascend.VLLM_ASCEND_ELASTIC_EXECUTION_MODE) != 5:
            return
        try:
            self._release_post_shrink_staging_state()
        except Exception:
            logger.exception(
                "Mode5 cache-rank runtime release failed at staging cleanup: rank=%s",
                self.rank)
        # Cache ranks should only retain expert weights and the minimal remote
        # cache service state. Drop transient prefetch/runtime caches eagerly.
        for attr_name, empty_value in (
            ("_mode4_remote_recv_buffer_cache", {}),
            ("_mode4_remote_send_buffer_cache", {}),
            ("_mode4_remote_prepacked_payload_cache", {}),
        ):
            setattr(self, attr_name, empty_value)
        try:
            self._drain_mode4_double_buffer_prefetch("mode5_cache_rank_release")
        except Exception:
            logger.exception(
                "Mode5 cache-rank runtime release failed at prefetch drain: rank=%s",
                self.rank)
        try:
            self._reset_mode4_double_buffer_runtime_slots(
                "mode5_cache_rank_release")
        except Exception:
            logger.exception(
                "Mode5 cache-rank runtime release failed at slot reset: rank=%s",
                self.rank)

    def _should_drop_stale_group_cache_after_elastic_shrink(
            self, active_ranks: list[int], world_size: int) -> bool:
        if self._has_mode4_remote_npu_lightweight_module():
            return _env_flag(
                "VLLM_ASCEND_MODE4_DROP_STALE_GROUP_CACHE_AFTER_SHRINK", "0")
        if self._has_mode3_cross_layer_lightweight_module():
            if not _env_flag(
                    "VLLM_ASCEND_MODE3_DROP_STALE_GROUP_CACHE_AFTER_SHRINK",
                    "0"):
                return False
            if len(active_ranks) >= int(world_size):
                return False
            return True
        if not self._is_mode1_low_floor_elastic_runtime():
            return False
        # Zero-headroom mode=1 relies on the restore path reusing the cached
        # full-world DP/EP/MC2 communicators. Dropping them during shrink
        # forces restore onto the slow MC2 re-create/warmup path and is the
        # direct cause of the multi-minute restore regression. Keep the old
        # full-world groups alive across shrink by default, then drop the
        # stale floor-N MC2 groups after full-world restore completes.
        if not _env_flag(
                "VLLM_ASCEND_MODE1_PARITY_DROP_STALE_CACHE_AFTER_SHRINK",
                "0"):
            return False
        if len(active_ranks) >= int(world_size):
            return False
        return True

    def _has_mode3_cross_layer_lightweight_module(self) -> bool:
        model_runner = getattr(self, "model_runner", None)
        model = getattr(model_runner, "model", None) if model_runner else None
        if model is None:
            return False
        for module in model.modules():
            if _module_is_mode3_cross_layer_lightweight(module):
                return True
        return False

    def _has_mode4_remote_npu_lightweight_module(
            self,
            exact_floor: Optional[int] = None,
            max_floor: Optional[int] = None) -> bool:
        model_runner = getattr(self, "model_runner", None)
        model = getattr(model_runner, "model", None) if model_runner else None
        if model is None:
            return False
        for module in model.modules():
            if not _module_is_mode4_remote_npu_lightweight(module):
                continue
            configured_floor = _module_configured_elastic_floor(module)
            if configured_floor is None:
                continue
            configured_floor = int(configured_floor)
            if exact_floor is not None and configured_floor != exact_floor:
                continue
            if max_floor is not None and configured_floor > max_floor:
                continue
            return True
        return False

    def _reset_mode4_double_buffer_runtime_slots(self, reason: str) -> None:
        """Drop logical bindings while keeping the persistent NPU slot tensors.

        Mode4 intentionally reuses the large two-slot NPU buffers across rollout
        steps. The contents are only valid for a specific shrink window though:
        active ranks, cache ranks, and remote expert placement can all change
        between steps. Resetting the runtime metadata here forces the next
        forward to repopulate from the current remote placement map instead of
        treating an old slot as a prefetch hit.
        """
        self._drain_mode4_double_buffer_prefetch(
            f"before_runtime_slot_reset:{reason}")
        model_runner = getattr(self, "model_runner", None)
        model = getattr(model_runner, "model", None) if model_runner else None
        if model is None:
            return
        reset_slots = 0
        seen: set[int] = set()
        candidates = [model, getattr(model, "model", None)]
        for candidate in candidates:
            if candidate is None or id(candidate) in seen:
                continue
            seen.add(id(candidate))
            slots = getattr(candidate, "_mode4_remote_npu_double_buffer_slots",
                            None)
            if not slots:
                continue
            for slot in slots:
                reset_fn = getattr(slot, "reset_runtime_state", None)
                if callable(reset_fn):
                    reset_fn()
                    reset_slots += 1
        if reset_slots:
            logger.info(
                "Mode4 double-buffer runtime slots reset: rank=%s reason=%s slots=%s",
                self.rank,
                reason,
                reset_slots,
            )

    def _drain_mode4_double_buffer_prefetch(self, reason: str) -> None:
        """Wait for outstanding mode4/mode5 double-buffer transfers.

        The slot tensors are persistent, while the logical slot bindings are
        rebuilt at every shrink/restore boundary.  If we reset the metadata or
        rebuild HCCL groups while a prefetch stream is still receiving remote
        payloads, a cache rank can keep sending into an old request while the
        active rank has already moved on to the next stage.  Drain only at those
        boundaries; normal layer-to-layer overlap is left untouched.
        """
        if not self._has_mode4_remote_npu_lightweight_module():
            return
        model_runner = getattr(self, "model_runner", None)
        model = getattr(model_runner, "model", None) if model_runner else None
        if model is None:
            return
        seen: set[int] = set()
        slots = []
        for candidate in (model, getattr(model, "model", None)):
            if candidate is None or id(candidate) in seen:
                continue
            seen.add(id(candidate))
            candidate_slots = getattr(
                candidate, "_mode4_remote_npu_double_buffer_slots", None)
            if candidate_slots:
                slots.extend(candidate_slots)
        if not slots:
            return

        start_t = time.perf_counter()
        drained_slots = 0
        errors: list[str] = []
        for slot in slots:
            needs_drain = bool(getattr(slot, "inflight_prefetch", False))
            needs_drain = needs_drain or bool(
                getattr(slot, "has_async_cpu_copy", False))
            needs_drain = needs_drain or bool(
                getattr(slot, "has_async_cpu_pack", False))
            needs_drain = needs_drain or bool(
                getattr(slot, "has_async_cpu_direct", False))
            if not needs_drain:
                continue
            for event_name in ("ready_event", "cpu_ready_event",
                               "cpu_pack_event"):
                event = getattr(slot, event_name, None)
                if event is None:
                    continue
                try:
                    event.synchronize()
                except Exception as exc:
                    errors.append(f"{event_name}:{exc}")
            drained_slots += 1
        if drained_slots:
            try:
                torch.npu.synchronize()
            except Exception as exc:
                errors.append(f"npu_synchronize:{exc}")
            logger.info(
                "Mode4 double-buffer prefetch drain done: rank=%s reason=%s slots=%s total_ms=%.2f errors=%s",
                self.rank,
                reason,
                drained_slots,
                (time.perf_counter() - start_t) * 1000.0,
                errors[:4],
            )

    def _has_custom_mode1_floor8_parity_module(self) -> bool:
        return self._has_custom_mode1_native_parity_module(exact_floor=8)

    def _has_mode1_lightweight_parity_module(
            self,
            exact_floor: Optional[int] = None,
            max_floor: Optional[int] = None) -> bool:
        model_runner = getattr(self, "model_runner", None)
        model = getattr(model_runner, "model", None) if model_runner else None
        if model is None:
            return False
        for module in model.modules():
            if not _module_is_mode1_lightweight_parity(module):
                continue
            configured_floor = _module_configured_elastic_floor(module)
            if configured_floor is None:
                continue
            configured_floor = int(configured_floor)
            if exact_floor is not None and configured_floor != exact_floor:
                continue
            if max_floor is not None and configured_floor > max_floor:
                continue
            return True
        return False

    def _has_elastic_lightweight_no_headroom_module(
            self,
            exact_floor: Optional[int] = None,
            max_floor: Optional[int] = None) -> bool:
        model_runner = getattr(self, "model_runner", None)
        model = getattr(model_runner, "model", None) if model_runner else None
        if model is None:
            return False
        for module in model.modules():
            if not _module_is_elastic_lightweight_no_headroom(module):
                continue
            configured_floor = _module_configured_elastic_floor(module)
            if configured_floor is None:
                continue
            configured_floor = int(configured_floor)
            if exact_floor is not None and configured_floor != exact_floor:
                continue
            if max_floor is not None and configured_floor > max_floor:
                continue
            return True
        return False

    def _has_custom_mode1_native_parity_module(
            self,
            exact_floor: Optional[int] = None,
            max_floor: Optional[int] = None) -> bool:
        model_runner = getattr(self, "model_runner", None)
        model = getattr(model_runner, "model", None) if model_runner else None
        if model is None:
            return False
        for module in model.modules():
            if not _module_is_custom_mode1_native_parity(module):
                continue
            configured_floor = _module_configured_elastic_floor(module)
            if configured_floor is None:
                continue
            configured_floor = int(configured_floor)
            if exact_floor is not None and configured_floor != exact_floor:
                continue
            if max_floor is not None and configured_floor > max_floor:
                continue
            return True
        return False

    def _mode1_current_floor_override(self) -> Optional[int]:
        raw_floor = os.getenv("VLLM_ASCEND_MODE1_PARITY_CURRENT_FLOOR")
        if raw_floor is None:
            return None
        try:
            return int(raw_floor)
        except (TypeError, ValueError):
            return None

    def _mode1_current_floor_is_full_world(
            self, world_size: Optional[int] = None) -> bool:
        floor = self._mode1_current_floor_override()
        if floor is None:
            return False
        if world_size is None:
            for env_name in ("WORLD_SIZE", "RAY_LOCAL_WORLD_SIZE"):
                raw_world_size = os.getenv(env_name)
                if raw_world_size is None:
                    continue
                try:
                    world_size = int(raw_world_size)
                    break
                except (TypeError, ValueError):
                    world_size = None
            if world_size is None:
                if not (torch.distributed.is_available()
                        and torch.distributed.is_initialized()):
                    return False
                world_size = int(torch.distributed.get_world_size())
        return int(world_size) > 0 and floor >= int(world_size)

    def _is_mode1_low_floor_elastic_runtime(
            self, max_floor: Optional[int] = 8) -> bool:
        if int(envs_ascend.VLLM_ASCEND_ELASTIC_EXECUTION_MODE) != 1:
            return False
        if not envs_ascend.VLLM_ASCEND_ENABLE_ELASTIC_PARALLEL_SHRINK:
            return False
        floor = self._mode1_current_floor_override()
        if floor is None:
            floor = envs_ascend.VLLM_ASCEND_ELASTIC_MIN_COMPUTE_GROUP_SIZE
            if floor is None:
                return False
            try:
                floor = int(floor)
            except (TypeError, ValueError):
                return False
        if self._mode1_current_floor_is_full_world():
            return False
        if floor <= 0:
            return False
        if max_floor is not None and floor > int(max_floor):
            return False
        return True

    def _warmup_post_restore_mc2_dispatch_for_custom_mode1_parity(
            self, world_size: int) -> None:
        if self._has_mode4_remote_npu_lightweight_module():
            return
        if not _env_flag(
                "VLLM_ASCEND_MODE1_PARITY_POST_RESTORE_MC2_WARMUP", "1"):
            return
        if not self._has_custom_mode1_native_parity_module():
            return
        model_runner = getattr(self, "model_runner", None)
        if model_runner is None or not hasattr(
                model_runner, "warmup_mode1_parity_mc2_dispatcher_only"):
            return
        warmup_tokens = int(
            os.getenv(
                "VLLM_ASCEND_MODE1_PARITY_POST_RESTORE_MC2_WARMUP_TOKENS",
                "32"))
        if warmup_tokens <= 0:
            logger.info(
                "Elastic post-restore MC2 dispatch warmup skipped: rank=%s "
                "reason=no_tokens tokens=%s",
                self.rank,
                warmup_tokens,
            )
            return
        self._log_mode1_fullworld_restore_diag(
            "before_post_restore_mc2_warmup",
            f"world_size={world_size} tokens={warmup_tokens}",
        )
        self._log_custom_mode1_worker_memory(
            "before_post_restore_mc2_dispatcher_warmup",
            f"world_size={world_size} tokens={warmup_tokens}",
        )
        try:
            model_runner.warmup_mode1_parity_mc2_dispatcher_only(
                warmup_tokens)
        except Exception:
            logger.exception(
                "Elastic post-restore MC2 dispatch warmup failed: rank=%s "
                "world_size=%s tokens=%s",
                self.rank,
                world_size,
                warmup_tokens,
            )
            raise
        if torch.npu.is_available():
            torch.npu.synchronize()
        self._release_mode1_moe_comm_runtime_state(
            f"after_post_restore_mc2_dispatcher_warmup:world_size={world_size}")
        gc.collect()
        if torch.npu.is_available():
            torch.npu.empty_cache()
            torch.npu.synchronize()
        self._log_custom_mode1_worker_memory(
            "after_post_restore_mc2_dispatcher_warmup",
            f"world_size={world_size} tokens={warmup_tokens}",
        )
        self._log_mode1_fullworld_restore_diag(
            "after_post_restore_mc2_warmup",
            f"world_size={world_size} tokens={warmup_tokens}",
        )

    def _warmup_post_restore_alltoall_dispatch_for_custom_mode1_parity(
            self, world_size: int) -> None:
        if self._has_mode4_remote_npu_lightweight_module():
            return
        if not _env_flag(
                "VLLM_ASCEND_MODE1_PARITY_POST_RESTORE_ALLTOALL_WARMUP",
                "0"):
            return
        if not self._has_custom_mode1_native_parity_module():
            return
        model_runner = getattr(self, "model_runner", None)
        if model_runner is None or not hasattr(
                model_runner,
                "warmup_mode1_parity_alltoall_dispatcher_only"):
            return
        warmup_tokens = int(
            os.getenv(
                "VLLM_ASCEND_MODE1_PARITY_POST_RESTORE_ALLTOALL_WARMUP_TOKENS",
                "513"))
        max_tokens_across_dp = int(
            os.getenv(
                "VLLM_ASCEND_MODE1_PARITY_POST_RESTORE_ALLTOALL_WARMUP_MAX_TOKENS",
                str(warmup_tokens)))
        if max_tokens_across_dp != warmup_tokens:
            logger.info(
                "Elastic post-restore ALLTOALL dispatch warmup adjusting "
                "max_tokens_across_dp to satisfy DPMetadata: rank=%s "
                "tokens=%s requested_max_tokens_across_dp=%s",
                self.rank,
                warmup_tokens,
                max_tokens_across_dp,
            )
            max_tokens_across_dp = warmup_tokens
        if warmup_tokens <= 0 or max_tokens_across_dp <= 0:
            logger.info(
                "Elastic post-restore ALLTOALL dispatch warmup skipped: rank=%s "
                "reason=no_tokens tokens=%s max_tokens_across_dp=%s",
                self.rank,
                warmup_tokens,
                max_tokens_across_dp,
            )
            return
        self._log_custom_mode1_worker_memory(
            "before_post_restore_alltoall_dispatcher_warmup",
            f"world_size={world_size} tokens={warmup_tokens} "
            f"max_tokens_across_dp={max_tokens_across_dp}",
        )
        try:
            model_runner.warmup_mode1_parity_alltoall_dispatcher_only(
                warmup_tokens,
                max_tokens_across_dp=max_tokens_across_dp)
        except Exception:
            logger.exception(
                "Elastic post-restore ALLTOALL dispatch warmup failed: rank=%s "
                "world_size=%s tokens=%s max_tokens_across_dp=%s",
                self.rank,
                world_size,
                warmup_tokens,
                max_tokens_across_dp,
            )
            raise
        if torch.npu.is_available():
            torch.npu.synchronize()
        self._post_kv_ep_collectives_warmed_up = True
        self._log_custom_mode1_worker_memory(
            "after_post_restore_alltoall_dispatcher_warmup",
            f"world_size={world_size} tokens={warmup_tokens} "
            f"max_tokens_across_dp={max_tokens_across_dp}",
        )

    def _should_destroy_stashed_mc2_for_single_live_group(
            self, attr_name: str, group_ranks: tuple[int, ...]) -> bool:
        if self._normalize_elastic_parallel_group_kind(attr_name) != "mc2":
            return False
        if not _env_flag("VLLM_ASCEND_MODE1_PARITY_SINGLE_LIVE_MC2_GROUP",
                         "0"):
            return False
        if not group_ranks:
            return False
        target_ranks = tuple(
            int(rank) for rank in getattr(
                self, "_elastic_rebuild_target_ranks", ()) or ())
        if not target_ranks or group_ranks == target_ranks:
            return False
        if not self._has_custom_mode1_native_parity_module():
            return False
        return True

    def _should_destroy_stashed_group_for_mode1_full_restore(
            self, attr_name: str, group_ranks: tuple[int, ...]) -> bool:
        group_kind = self._normalize_elastic_parallel_group_kind(attr_name)
        if group_kind not in ("dp", "ep", "mc2"):
            return False
        if not group_ranks:
            return False
        if not self._is_mode1_low_floor_elastic_runtime():
            return False
        if not torch.distributed.is_initialized():
            return False
        target_ranks = tuple(
            int(rank) for rank in getattr(
                self, "_elastic_rebuild_target_ranks", ()) or ())
        if not target_ranks or group_ranks == target_ranks:
            return False
        world_size = int(torch.distributed.get_world_size())
        full_world_ranks = tuple(
            self._original_local_elastic_group_ranks(world_size))
        if target_ranks != full_world_ranks:
            return False
        if self._should_keep_mode1_fixed_topology_stale_group(
                group_kind, group_ranks, target_ranks):
            return False
        # Keep only the previously stashed full-world EP cache across shrink.
        # During restore, the currently live shrink-stage DP/EP/MC2 groups are
        # not useful after the full-world rebuild target is known. Re-stashing
        # them keeps extra communicator state alive until the later stale-cache
        # sweep, which regressed mode=1 restore latency into the minute range.
        return True

    def _mode1_pre_rebuild_destroy_floor_sizes(self) -> set[int]:
        raw_sizes = os.getenv(
            "VLLM_ASCEND_MODE1_PARITY_PRE_REBUILD_DESTROY_FLOOR_GROUP_SIZES",
            "1,2,4,8",
        )
        sizes: set[int] = set()
        for raw_size in raw_sizes.split(","):
            raw_size = raw_size.strip()
            if not raw_size:
                continue
            try:
                size = int(raw_size)
            except ValueError:
                continue
            if size > 0:
                sizes.add(size)
        return sizes

    def _should_destroy_stashed_group_for_mode1_pre_rebuild(
            self, attr_name: str, group_ranks: tuple[int, ...]) -> bool:
        group_kind = self._normalize_elastic_parallel_group_kind(attr_name)
        if group_kind not in ("dp", "ep", "mc2"):
            return False
        if not group_ranks:
            return False
        if not self._is_mode1_low_floor_elastic_runtime():
            return False
        if not _env_flag(
                "VLLM_ASCEND_MODE1_PARITY_RELEASE_LIVE_OLD_FLOOR_ON_REBUILD",
                "1"):
            return False
        target_ranks = tuple(
            int(rank) for rank in getattr(
                self, "_elastic_rebuild_target_ranks", ()) or ())
        if not target_ranks or group_ranks == target_ranks:
            return False
        if len(group_ranks) not in self._mode1_pre_rebuild_destroy_floor_sizes():
            return False
        if not torch.distributed.is_initialized():
            return False
        world_size = int(torch.distributed.get_world_size())
        if group_ranks == tuple(
                self._original_local_elastic_group_ranks(world_size)):
            return False
        if self._should_keep_mode1_fixed_topology_stale_group(
                group_kind, group_ranks, target_ranks):
            return False
        if self._should_keep_mode1_floor4_group_cache(
                group_kind, group_ranks, target_ranks):
            return False
        return True

    def _should_defer_stashed_group_destroy_for_mode1(
            self, group_kind: str, group_ranks: tuple[int, ...],
            reason: str) -> bool:
        target_ranks = tuple(
            int(rank) for rank in getattr(
                self, "_elastic_rebuild_target_ranks", ()) or ())
        return self._should_defer_mode1_group_destroy(
            group_kind,
            group_ranks,
            target_ranks,
            reason,
        )

    def _stash_group_if_present(self, state_module, attr_name: str) -> None:
        group = getattr(state_module, attr_name, None)
        if group is None:
            return

        group_ranks = tuple(int(rank) for rank in getattr(group, "ranks", []))
        group_kind = self._normalize_elastic_parallel_group_kind(attr_name)
        self._register_elastic_parallel_group(attr_name, group_ranks, group)
        should_cache_group = self._should_cache_elastic_parallel_group_ranks(
            attr_name, group_ranks)
        destroy_for_single_live_mc2 = (
            self._should_destroy_stashed_mc2_for_single_live_group(
                attr_name, group_ranks))
        destroy_for_mode1_full_restore = (
            self._should_destroy_stashed_group_for_mode1_full_restore(
                attr_name, group_ranks))
        destroy_for_mode1_pre_rebuild = (
            self._should_destroy_stashed_group_for_mode1_pre_rebuild(
                attr_name, group_ranks))
        already_retired_group = self._is_retired_elastic_group(group)
        if already_retired_group:
            should_cache_group = False
        if destroy_for_single_live_mc2:
            target_ranks = tuple(
                int(rank) for rank in getattr(
                    self, "_elastic_rebuild_target_ranks", ()) or ())
            logger.info(
                "Elastic custom mode1 MC2 single-live-group destroying old group before rebuild: "
                "rank=%s attr=%s old_ranks=%s target_ranks=%s",
                self.rank,
                attr_name,
                group_ranks,
                target_ranks,
            )
            self._log_custom_mode1_worker_memory(
                "before_single_live_mc2_destroy",
                f"old_ranks={group_ranks} target_ranks={target_ranks}",
            )
            should_cache_group = False
        elif destroy_for_mode1_full_restore:
            target_ranks = tuple(
                int(rank) for rank in getattr(
                    self, "_elastic_rebuild_target_ranks", ()) or ())
            logger.info(
                "Elastic mode1 full-restore releases stale live floor group: "
                "rank=%s attr=%s group_kind=%s old_ranks=%s target_ranks=%s",
                self.rank,
                attr_name,
                group_kind,
                group_ranks,
                target_ranks,
            )
            if group_kind == "mc2":
                self._log_custom_mode1_worker_memory(
                    "before_mode1_full_restore_mc2_release",
                    f"old_ranks={group_ranks} target_ranks={target_ranks}",
                )
            should_cache_group = False
        elif destroy_for_mode1_pre_rebuild:
            target_ranks = tuple(
                int(rank) for rank in getattr(
                    self, "_elastic_rebuild_target_ranks", ()) or ())
            logger.info(
                "Elastic mode1 pre-rebuild defers stale live floor group destroy: "
                "rank=%s attr=%s group_kind=%s old_ranks=%s target_ranks=%s",
                self.rank,
                attr_name,
                group_kind,
                group_ranks,
                target_ranks,
            )
            if group_kind == "mc2":
                self._log_custom_mode1_worker_memory(
                    "before_mode1_pre_rebuild_mc2_defer",
                    f"old_ranks={group_ranks} target_ranks={target_ranks}",
                )
            should_cache_group = False
        if group_ranks and should_cache_group:
            cached_groups = self._get_cached_elastic_parallel_groups()
            cached_groups.setdefault(attr_name, {})[group_ranks] = group
            self._get_seen_elastic_parallel_group_signatures().add(
                (group_kind, group_ranks))
        else:
            if group_ranks:
                self._remove_cached_elastic_parallel_group_reference(
                    attr_name, group_ranks, group)
                self._get_seen_elastic_parallel_group_signatures().discard(
                    (group_kind, group_ranks))
            if already_retired_group:
                destroy_reason = "already_retired"
            elif destroy_for_single_live_mc2:
                destroy_reason = "single_live_rebuild"
            elif destroy_for_mode1_full_restore:
                destroy_reason = "mode1_full_restore"
            elif destroy_for_mode1_pre_rebuild:
                destroy_reason = "mode1_pre_rebuild"
            else:
                destroy_reason = "stash_group"
            cleanup_ms = 0.0
            force_cleanup_after_release = (
                self._should_force_cleanup_after_mode1_floor_group_release(
                    group_kind, group_ranks))
            if self._should_skip_cleanup_after_preserved_full_restore_release(
                    group_kind, group_ranks, destroy_reason):
                force_cleanup_after_release = False
            if already_retired_group:
                destroy_ms = 0.0
                self._remove_elastic_parallel_group_registry_entry(
                    group_kind, group_ranks, group)
            elif self._should_defer_stashed_group_destroy_for_mode1(
                    group_kind,
                    group_ranks,
                    destroy_reason,
            ):
                self._quarantine_deferred_elastic_group(
                    group,
                    group_kind,
                    group_ranks,
                    destroy_reason,
                )
                destroy_ms = 0.0
                if force_cleanup_after_release:
                    cleanup_stats = self._cleanup_after_elastic_group_release(
                        destroy_reason, force=True)
                    cleanup_ms = float(cleanup_stats.get("total_ms", 0.0))
            else:
                if self._is_mode1_low_floor_elastic_runtime():
                    logger.warning(
                        "Elastic mode1 low-floor stashed group direct destroy fallback: "
                        "rank=%s group_kind=%s group_ranks=%s reason=%s "
                        "defer_env=%s defer_full_restore=%s defer_pre_rebuild=%s "
                        "defer_stash=%s defer_sizes=%s",
                        self.rank,
                        group_kind,
                        group_ranks,
                        destroy_reason,
                        os.getenv("VLLM_ASCEND_MODE1_PARITY_DEFER_GROUP_DESTROY", "1"),
                        os.getenv("VLLM_ASCEND_MODE1_PARITY_DEFER_GROUP_DESTROY_ON_FULL_RESTORE", "0"),
                        os.getenv("VLLM_ASCEND_MODE1_PARITY_DEFER_GROUP_DESTROY_ON_PRE_REBUILD", "0"),
                        os.getenv("VLLM_ASCEND_MODE1_PARITY_DEFER_GROUP_DESTROY_ON_STASH", "0"),
                        os.getenv("VLLM_ASCEND_MODE1_PARITY_DEFER_DESTROY_FLOOR_GROUP_SIZES", "1,2,4,8"),
                    )
                refs_before_destroy = self._describe_elastic_group_refs(group)
                self._log_mode1_direct_group_destroy_event(
                    "start",
                    group_kind,
                    group_ranks,
                    destroy_reason,
                    refs_before_destroy,
                    attr_name=attr_name,
                    cleanup_force=force_cleanup_after_release,
                )
                destroy_start_t = time.perf_counter()
                destroy_call_start_t = time.perf_counter()
                group.destroy()
                destroy_call_ms = (
                    time.perf_counter() - destroy_call_start_t) * 1000.0
                detach_start_t = time.perf_counter()
                self._detach_destroyed_group_references(group)
                detach_ms = (time.perf_counter() - detach_start_t) * 1000.0
                self._mark_retired_elastic_group(group)
                registry_start_t = time.perf_counter()
                self._remove_elastic_parallel_group_registry_entry(
                    group_kind, group_ranks, group)
                registry_ms = (
                    time.perf_counter() - registry_start_t) * 1000.0
                destroy_ms = (time.perf_counter() - destroy_start_t) * 1000.0
                refs_after_destroy = self._describe_elastic_group_refs(group)
                self._log_mode1_direct_group_destroy_event(
                    "done",
                    group_kind,
                    group_ranks,
                    destroy_reason,
                    refs_after_destroy,
                    attr_name=attr_name,
                    destroy_call_ms=destroy_call_ms,
                    detach_ms=detach_ms,
                    registry_ms=registry_ms,
                    total_ms=destroy_ms,
                    cleanup_force=force_cleanup_after_release,
                )
                if group_kind == "mc2":
                    cleanup_stats = self._cleanup_after_elastic_group_release(
                        destroy_reason, force=force_cleanup_after_release)
                    cleanup_ms = float(cleanup_stats.get("total_ms", 0.0))
                elif force_cleanup_after_release:
                    cleanup_stats = self._cleanup_after_elastic_group_release(
                        destroy_reason, force=True)
                    cleanup_ms = float(cleanup_stats.get("total_ms", 0.0))
            if destroy_for_mode1_pre_rebuild or destroy_for_mode1_full_restore or destroy_ms >= 1000.0:
                logger.info(
                    "Elastic parallel stashed group released: "
                    "rank=%s attr=%s group_kind=%s old_ranks=%s reason=%s "
                    "destroy_ms=%.2f cleanup_ms=%.2f already_retired=%s",
                    self.rank,
                    attr_name,
                    group_kind,
                    group_ranks,
                    destroy_reason,
                    destroy_ms,
                    cleanup_ms,
                    already_retired_group,
                )
        setattr(state_module, attr_name, None)

    def _get_local_group_ranks(self,
                               group_ranks: list[list[int]]) -> tuple[int, ...]:
        current_rank = torch.distributed.get_rank()
        for ranks in group_ranks:
            if current_rank in ranks:
                return tuple(int(rank) for rank in ranks)
        return ()

    def _drop_stale_cached_elastic_parallel_groups(
            self,
            keep_group_ranks: tuple[int, ...],
            reason: str = "drop_stale_cache",
            force_destroy: bool = False) -> dict[str, float]:
        drop_start_t = time.perf_counter()
        cached_groups = self._get_cached_elastic_parallel_groups()
        seen_signatures = self._get_seen_elastic_parallel_group_signatures()
        dropped_groups = 0
        dropped_signatures = 0
        destroyed_group_ids: set[int] = set()
        deferred_groups = 0
        cleanup_ms = 0.0
        cleanup_gc_ms = 0.0
        cleanup_empty_cache_ms = 0.0
        cleanup_sync_ms = 0.0
        cleanup_memory_log_ms = 0.0
        registry_dropped_groups = 0

        for attr_name, groups_by_ranks in cached_groups.items():
            group_kind = self._normalize_elastic_parallel_group_kind(attr_name)
            stale_group_ranks = [
                ranks
                for ranks in list(groups_by_ranks.keys())
                if ranks != keep_group_ranks
                and not (
                    self._should_keep_stale_group_cache_for_custom_mode1_parity(
                        group_kind,
                        tuple(int(rank) for rank in ranks),
                        keep_group_ranks,
                    )
                    or self._should_keep_stale_group_cache_for_mode3(
                        group_kind,
                        tuple(int(rank) for rank in ranks),
                        keep_group_ranks,
                    )
                    or self._should_keep_stale_group_cache_for_mode5(
                        group_kind,
                        tuple(int(rank) for rank in ranks),
                        keep_group_ranks,
                    )
                )
            ]
            kept_stale_group_ranks = [
                ranks
                for ranks in list(groups_by_ranks.keys())
                if ranks != keep_group_ranks
                and (
                    self._should_keep_stale_group_cache_for_custom_mode1_parity(
                        group_kind,
                        tuple(int(rank) for rank in ranks),
                        keep_group_ranks,
                    )
                    or self._should_keep_stale_group_cache_for_mode3(
                        group_kind,
                        tuple(int(rank) for rank in ranks),
                        keep_group_ranks,
                    )
                    or self._should_keep_stale_group_cache_for_mode5(
                        group_kind,
                        tuple(int(rank) for rank in ranks),
                        keep_group_ranks,
                    )
                )
            ]
            for ranks in kept_stale_group_ranks:
                logger.info(
                    "Elastic lightweight path keeps stale %s cache: rank=%s keep_ranks=%s cached_ranks=%s mode3=%s",
                    group_kind.upper(), self.rank, keep_group_ranks, ranks,
                    self._has_mode3_cross_layer_lightweight_module())
            for ranks in stale_group_ranks:
                group = groups_by_ranks.pop(ranks, None)
                if group is not None:
                    group_id = id(group)
                    if group_id not in destroyed_group_ids:
                        if self._is_retired_elastic_group(group):
                            destroyed_group_ids.add(group_id)
                        else:
                            if group_kind == "mc2":
                                self._log_custom_mode1_worker_memory(
                                    "before_stale_mc2_cache_drop",
                                    f"stale_ranks={ranks} keep_ranks={keep_group_ranks}",
                                )
                            ranks_tuple = tuple(int(rank) for rank in ranks)
                            force_cleanup_after_release = (
                                self._should_force_cleanup_after_mode1_floor_group_release(
                                    group_kind, ranks_tuple))
                            if (not force_destroy
                                    and self._should_defer_mode1_group_destroy(
                                    group_kind,
                                    ranks_tuple,
                                    keep_group_ranks,
                                    reason,
                            )):
                                if self._quarantine_deferred_elastic_group(
                                    group,
                                    group_kind,
                                    ranks_tuple,
                                    reason,
                                ):
                                    deferred_groups += 1
                                destroyed_group_ids.add(group_id)
                                if force_cleanup_after_release:
                                    cleanup_stats = self._cleanup_after_elastic_group_release(
                                        reason, force=True)
                                    cleanup_ms += float(
                                        cleanup_stats.get("total_ms", 0.0))
                                    cleanup_gc_ms += float(
                                        cleanup_stats.get("gc_ms", 0.0))
                                    cleanup_empty_cache_ms += float(
                                        cleanup_stats.get("empty_cache_ms", 0.0))
                                    cleanup_sync_ms += float(
                                        cleanup_stats.get("sync_ms", 0.0))
                                    cleanup_memory_log_ms += float(
                                        cleanup_stats.get("memory_log_ms", 0.0))
                            else:
                                if self._is_mode1_low_floor_elastic_runtime():
                                    logger.warning(
                                        "Elastic mode1 low-floor stale cache direct destroy fallback: "
                                        "rank=%s group_kind=%s group_ranks=%s keep_ranks=%s "
                                        "reason=%s force_destroy=%s defer_env=%s "
                                        "defer_full_restore=%s defer_pre_rebuild=%s "
                                        "defer_stash=%s defer_sizes=%s",
                                        self.rank,
                                        group_kind,
                                        tuple(int(rank) for rank in ranks),
                                        keep_group_ranks,
                                        reason,
                                        force_destroy,
                                        os.getenv("VLLM_ASCEND_MODE1_PARITY_DEFER_GROUP_DESTROY", "1"),
                                        os.getenv("VLLM_ASCEND_MODE1_PARITY_DEFER_GROUP_DESTROY_ON_FULL_RESTORE", "0"),
                                        os.getenv("VLLM_ASCEND_MODE1_PARITY_DEFER_GROUP_DESTROY_ON_PRE_REBUILD", "0"),
                                        os.getenv("VLLM_ASCEND_MODE1_PARITY_DEFER_GROUP_DESTROY_ON_STASH", "0"),
                                        os.getenv("VLLM_ASCEND_MODE1_PARITY_DEFER_DESTROY_FLOOR_GROUP_SIZES", "1,2,4,8"),
                                    )
                                group.destroy()
                                self._detach_destroyed_group_references(group)
                                self._mark_retired_elastic_group(group)
                                self._remove_elastic_parallel_group_registry_entry(
                                    group_kind,
                                    tuple(int(rank) for rank in ranks),
                                    group)
                                destroyed_group_ids.add(group_id)
                                if group_kind == "mc2" and (
                                        self._has_mode1_lightweight_parity_module()
                                        or self._has_mode4_remote_npu_lightweight_module()):
                                    cleanup_stats = self._cleanup_after_elastic_group_release(
                                        reason,
                                        force=force_cleanup_after_release)
                                    cleanup_ms += float(
                                        cleanup_stats.get("total_ms", 0.0))
                                    cleanup_gc_ms += float(
                                        cleanup_stats.get("gc_ms", 0.0))
                                    cleanup_empty_cache_ms += float(
                                        cleanup_stats.get("empty_cache_ms", 0.0))
                                    cleanup_sync_ms += float(
                                        cleanup_stats.get("sync_ms", 0.0))
                                    cleanup_memory_log_ms += float(
                                        cleanup_stats.get("memory_log_ms", 0.0))
                                elif force_cleanup_after_release:
                                    cleanup_stats = self._cleanup_after_elastic_group_release(
                                        reason, force=True)
                                    cleanup_ms += float(
                                        cleanup_stats.get("total_ms", 0.0))
                                    cleanup_gc_ms += float(
                                        cleanup_stats.get("gc_ms", 0.0))
                                    cleanup_empty_cache_ms += float(
                                        cleanup_stats.get("empty_cache_ms", 0.0))
                                    cleanup_sync_ms += float(
                                        cleanup_stats.get("sync_ms", 0.0))
                                    cleanup_memory_log_ms += float(
                                        cleanup_stats.get("memory_log_ms", 0.0))
                    dropped_groups += 1
                if (group_kind, ranks) in seen_signatures:
                    seen_signatures.discard((group_kind, ranks))
                    dropped_signatures += 1
                if group_kind == "mc2":
                    logger.info(
                        "Elastic parallel stale MC2 cache dropped across restore: rank=%s keep_ranks=%s stale_ranks=%s",
                        self.rank, keep_group_ranks, ranks)

        for group_kind, ranks, group in (
                self._iter_stale_elastic_parallel_group_registry_entries(
                    keep_group_ranks, old_floor_only=False)):
            stats = self._release_elastic_parallel_group_reference(
                group,
                group_kind,
                ranks,
                keep_group_ranks,
                reason,
                destroyed_group_ids,
                "before_stale_registry_mc2_drop",
                force_destroy=force_destroy,
            )
            if stats.get("released_groups", 0.0) > 0.0:
                registry_dropped_groups += 1
                dropped_groups += 1
            deferred_groups += int(stats.get("deferred_groups", 0.0))
            cleanup_ms += float(stats.get("cleanup_ms", 0.0))
            cleanup_gc_ms += float(stats.get("cleanup_gc_ms", 0.0))
            cleanup_empty_cache_ms += float(
                stats.get("cleanup_empty_cache_ms", 0.0))
            cleanup_sync_ms += float(stats.get("cleanup_sync_ms", 0.0))
            cleanup_memory_log_ms += float(
                stats.get("cleanup_memory_log_ms", 0.0))
            seen_signatures.discard((group_kind, ranks))
            if group_kind == "mc2":
                logger.info(
                    "Elastic parallel stale MC2 registry dropped: rank=%s keep_ranks=%s stale_ranks=%s",
                    self.rank, keep_group_ranks, ranks)

        stale_signatures = [
            signature for signature in seen_signatures
            if signature[1] != keep_group_ranks
            and not self._should_keep_stale_group_cache_for_custom_mode1_parity(
                signature[0],
                signature[1],
                keep_group_ranks,
            )
            and not self._should_keep_stale_group_cache_for_mode3(
                signature[0],
                signature[1],
                keep_group_ranks,
            )
            and not self._should_keep_stale_group_cache_for_mode5(
                signature[0],
                signature[1],
                keep_group_ranks,
            )
        ]
        for signature in stale_signatures:
            seen_signatures.discard(signature)
            dropped_signatures += 1

        if dropped_groups > 0 or dropped_signatures > 0:
            logger.info(
                "Elastic parallel stale group cache dropped: rank=%s keep_ranks=%s reason=%s force_destroy=%s dropped_groups=%s dropped_signatures=%s deferred_groups=%s cleanup_ms=%.2f",
                self.rank, keep_group_ranks, reason, force_destroy,
                dropped_groups, dropped_signatures, deferred_groups,
                cleanup_ms)
        return {
            "total_ms": (time.perf_counter() - drop_start_t) * 1000.0,
            "cleanup_ms": cleanup_ms,
            "cleanup_gc_ms": cleanup_gc_ms,
            "cleanup_empty_cache_ms": cleanup_empty_cache_ms,
            "cleanup_sync_ms": cleanup_sync_ms,
            "cleanup_memory_log_ms": cleanup_memory_log_ms,
            "dropped_groups": float(dropped_groups),
            "registry_dropped_groups": float(registry_dropped_groups),
            "deferred_groups": float(deferred_groups),
            "dropped_signatures": float(dropped_signatures),
        }

    def _advance_group_creation_sequence_for_non_member(
            self, group_ranks: list[list[int]], backend: str,
            group_name: str) -> dict[str, float]:
        """Keep default-pg new_group ordering aligned on detached ranks.

        During elastic shrink, active ranks rebuild DP/EP/MC2 groups and thus
        consume a sequence of torch.distributed.new_group() calls. Detached
        ranks do not join those groups, but they still need to advance through
        the same new_group creation order so a later full-world restore can
        rebuild the original 16-rank groups without Gloo store key mismatch.
        """
        start_t = time.perf_counter()
        stats: dict[str, float] = {
            "total_ms": 0.0,
            "device_new_group_ms": 0.0,
            "cpu_new_group_ms": 0.0,
            "destroy_ms": 0.0,
            "signature_ms": 0.0,
            "log_ms": 0.0,
            "groups": 0.0,
            "reuse": 0.0,
        }
        if _env_flag("VLLM_ASCEND_ELASTIC_LOCAL_GROUP_CREATION", "0"):
            stats["reuse"] = 1.0
            stats["total_ms"] = (time.perf_counter() - start_t) * 1000.0
            return stats
        group_kind = self._normalize_elastic_parallel_group_kind(group_name)
        unseen_group_ranks = []
        seen_signatures = self._get_seen_elastic_parallel_group_signatures()
        for ranks in group_ranks:
            ranks_key = tuple(int(rank) for rank in ranks)
            group_signature = (group_kind, ranks_key)
            if group_signature in seen_signatures:
                continue
            unseen_group_ranks.append(ranks)

        verbose_sequence_log = _env_flag(
            "VLLM_ASCEND_ELASTIC_NON_MEMBER_SEQUENCE_VERBOSE", "0")
        try:
            slow_log_ms = float(
                os.getenv(
                    "VLLM_ASCEND_ELASTIC_NON_MEMBER_SEQUENCE_SLOW_LOG_MS",
                    "1000"))
        except ValueError:
            slow_log_ms = 1000.0

        if not unseen_group_ranks:
            stats["reuse"] = 1.0
            stats["total_ms"] = (time.perf_counter() - start_t) * 1000.0
            if verbose_sequence_log or stats["total_ms"] >= slow_log_ms:
                log_start_t = time.perf_counter()
                logger.info(
                    "Elastic parallel non-member group sequence reuse: rank=%s group_name=%s groups=%s total_ms=%.2f",
                    self.rank, group_name, len(group_ranks),
                    stats["total_ms"])
                stats["log_ms"] = (
                    time.perf_counter() - log_start_t) * 1000.0
                stats["total_ms"] = (time.perf_counter() - start_t) * 1000.0
            return stats

        placeholder_groups = []
        for ranks in unseen_group_ranks:
            device_group_start_t = time.perf_counter()
            placeholder_groups.append(
                torch.distributed.new_group(ranks, backend=backend))
            stats["device_new_group_ms"] += (
                time.perf_counter() - device_group_start_t) * 1000.0
            cpu_group_start_t = time.perf_counter()
            placeholder_groups.append(
                torch.distributed.new_group(ranks, backend="gloo"))
            stats["cpu_new_group_ms"] += (
                time.perf_counter() - cpu_group_start_t) * 1000.0

        destroy_start_t = time.perf_counter()
        for group in placeholder_groups:
            if group == dist.GroupMember.NON_GROUP_MEMBER:
                continue
            torch.distributed.destroy_process_group(group)
        stats["destroy_ms"] = (time.perf_counter() -
                               destroy_start_t) * 1000.0

        signature_start_t = time.perf_counter()
        for ranks in unseen_group_ranks:
            seen_signatures.add(
                (group_kind, tuple(int(rank) for rank in ranks)))
        stats["signature_ms"] = (time.perf_counter() -
                                 signature_start_t) * 1000.0
        stats["groups"] = float(len(unseen_group_ranks))
        stats["total_ms"] = (time.perf_counter() - start_t) * 1000.0
        if verbose_sequence_log or stats["total_ms"] >= slow_log_ms:
            log_start_t = time.perf_counter()
            logger.info(
                "Elastic parallel non-member group sequence advanced: rank=%s group_name=%s groups=%s total_ms=%.2f device_new_group_ms=%.2f cpu_new_group_ms=%.2f destroy_ms=%.2f signature_ms=%.2f",
                self.rank, group_name, len(unseen_group_ranks),
                stats["total_ms"], stats["device_new_group_ms"],
                stats["cpu_new_group_ms"], stats["destroy_ms"],
                stats["signature_ms"])
            stats["log_ms"] = (time.perf_counter() - log_start_t) * 1000.0
            stats["total_ms"] = (time.perf_counter() - start_t) * 1000.0
        return stats

    def _reconcile_group_creation_sequence_before_restore(
            self, world_group, backend: str) -> None:
        if _env_flag("VLLM_ASCEND_ELASTIC_LOCAL_GROUP_CREATION", "0"):
            return
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
            if not _is_ascend_fused_moe_module(module):
                continue
            if hasattr(module, "clear_lossless_hybrid_state"):
                module.clear_lossless_hybrid_state()
            _set_module_active_expert_mask(module, None)
            _set_module_elastic_runtime_log2phy(module, None)
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

    def _release_live_mode1_old_floor_groups_before_rebuild(
            self, active_ranks: list[int]) -> dict[str, float]:
        if not self._is_mode1_low_floor_elastic_runtime():
            return {"released_groups": 0.0, "total_ms": 0.0}
        if not _env_flag(
                "VLLM_ASCEND_MODE1_PARITY_RELEASE_LIVE_OLD_FLOOR_ON_REBUILD",
                "1"):
            return {"released_groups": 0.0, "total_ms": 0.0}
        if not torch.distributed.is_initialized():
            return {"released_groups": 0.0, "total_ms": 0.0}

        import vllm.distributed.parallel_state as vllm_ps
        import vllm_ascend.distributed.parallel_state as ascend_ps

        world_size = int(torch.distributed.get_world_size())
        target_ranks = tuple(int(rank) for rank in active_ranks)
        previous_target_ranks = getattr(self, "_elastic_rebuild_target_ranks",
                                        None)
        self._elastic_rebuild_target_ranks = target_ranks
        release_start_t = time.perf_counter()
        released_groups = 0
        released_kinds: list[str] = []
        try:
            for state_module, attr_name in (
                    (vllm_ps, "_DP"),
                    (vllm_ps, "_EP"),
                    (ascend_ps, "_MC2"),
            ):
                group = getattr(state_module, attr_name, None)
                if group is None:
                    continue
                group_ranks = tuple(
                    int(rank) for rank in getattr(group, "ranks", []))
                if (not group_ranks or group_ranks == target_ranks
                        or len(group_ranks) >= world_size):
                    continue
                group_kind = self._normalize_elastic_parallel_group_kind(
                    attr_name)
                logger.info(
                    "Elastic mode1 pre-rebuild releases live old floor group: "
                    "rank=%s attr=%s group_kind=%s old_ranks=%s target_ranks=%s",
                    self.rank,
                    attr_name,
                    group_kind,
                    group_ranks,
                    target_ranks,
                )
                self._stash_group_if_present(state_module, attr_name)
                released_groups += 1
                released_kinds.append(group_kind)
        finally:
            self._elastic_rebuild_target_ranks = previous_target_ranks
        total_ms = (time.perf_counter() - release_start_t) * 1000.0
        if released_groups:
            logger.info(
                "Elastic mode1 pre-rebuild live old floor release done: "
                "rank=%s target_ranks=%s released_groups=%s released_kinds=%s total_ms=%.2f",
                self.rank,
                target_ranks,
                released_groups,
                released_kinds,
                total_ms,
            )
        return {"released_groups": float(released_groups), "total_ms": total_ms}

    def _rebuild_group(self, state_module, attr_name: str,
                       group_ranks: list[list[int]], world_group,
                       backend: str, group_name: str) -> None:
        local_group_ranks = self._get_local_group_ranks(group_ranks)
        previous_target_ranks = getattr(self, "_elastic_rebuild_target_ranks",
                                        None)
        self._elastic_rebuild_target_ranks = local_group_ranks
        group_kind = self._normalize_elastic_parallel_group_kind(attr_name)
        try:
            if group_kind == "mc2":
                self._log_custom_mode1_worker_memory(
                    "before_rebuild_mc2_stash",
                    f"group_name={group_name} target_ranks={local_group_ranks}",
                )
            self._stash_group_if_present(state_module, attr_name)

            should_cache_group = self._should_cache_elastic_parallel_group_ranks(
                attr_name, local_group_ranks)
            if should_cache_group:
                cached_groups = self._get_cached_elastic_parallel_groups()
                cached_group = cached_groups.get(attr_name,
                                                 {}).get(local_group_ranks)
                if cached_group is not None:
                    self._register_elastic_parallel_group(
                        attr_name, local_group_ranks, cached_group)
                    setattr(state_module, attr_name, cached_group)
                    logger.info(
                        "Elastic parallel group cache hit: rank=%s attr=%s group_name=%s ranks=%s",
                        self.rank, attr_name, group_name, local_group_ranks)
                    if group_kind == "mc2":
                        self._log_custom_mode1_worker_memory(
                            "after_rebuild_mc2_cache_hit",
                            f"group_name={group_name} ranks={local_group_ranks}",
                        )
                    return

            if group_kind == "mc2":
                self._log_custom_mode1_worker_memory(
                    "before_rebuild_mc2_create",
                    f"group_name={group_name} ranks={local_group_ranks}",
                )
            use_local_synchronization = _env_flag(
                "VLLM_ASCEND_ELASTIC_LOCAL_GROUP_CREATION", "0")
            local_synchronization_key = None
            if use_local_synchronization:
                restore_seq = int(
                    getattr(self, "_mode1_fullworld_restore_seq", 0))
                local_synchronization_key = (
                    f"elastic:{group_name}:restore{restore_seq}:"
                    f"{','.join(map(str, local_group_ranks))}")
            setattr(
                state_module, attr_name,
                init_model_parallel_group(group_ranks, world_group.local_rank,
                                          backend,
                                          group_name=group_name,
                                          use_local_synchronization=
                                          use_local_synchronization,
                                          local_synchronization_key=
                                          local_synchronization_key))
            created_group = getattr(state_module, attr_name)
            self._register_elastic_parallel_group(attr_name,
                                                  local_group_ranks,
                                                  created_group)
            if group_kind == "mc2":
                self._log_custom_mode1_worker_memory(
                    "after_rebuild_mc2_create",
                    f"group_name={group_name} ranks={local_group_ranks}",
                )
            if local_group_ranks:
                if should_cache_group:
                    cached_groups = self._get_cached_elastic_parallel_groups()
                    cached_groups.setdefault(attr_name, {})[
                        local_group_ranks] = getattr(state_module, attr_name)
                    if group_kind == "mc2":
                        logger.info(
                            "Elastic parallel MC2 group cached: rank=%s group_name=%s ranks=%s",
                            self.rank, group_name, local_group_ranks)
                self._get_seen_elastic_parallel_group_signatures().add(
                    (group_kind, local_group_ranks))
        finally:
            self._elastic_rebuild_target_ranks = previous_target_ranks

    def _warmup_post_shrink_dp_collectives(self) -> None:
        dp_group = get_dp_group()
        if dp_group.world_size <= 1:
            return
        if (envs_ascend.VLLM_ASCEND_ELASTIC_EXECUTION_MODE == 1
                and not _env_flag(
                    "VLLM_ASCEND_MODE1_PARITY_ENABLE_POST_SHRINK_DP_WARMUP",
                    "0")):
            logger.info(
                "Elastic post-shrink DP all_reduce warmup skipped: "
                "rank=%s dp_size=%s reason=mode1_default_disabled",
                self.rank, dp_group.world_size)
            return
        if (envs_ascend.VLLM_ASCEND_ELASTIC_EXECUTION_MODE in (4, 5)
                and not _env_flag(
                    "VLLM_ASCEND_MODE4_ENABLE_POST_SHRINK_DP_WARMUP",
                    "0")):
            logger.info(
                "Elastic post-shrink DP all_reduce warmup skipped: "
                "rank=%s dp_size=%s reason=mode4_remote_npu_decode_path",
                self.rank, dp_group.world_size)
            return

        # Force HCCL to materialize the new DP communicator/workspace before
        # post-shrink decode hits its first metadata all_reduce.
        self._log_custom_mode1_worker_memory(
            "before_post_shrink_dp_all_reduce_warmup",
            f"dp_size={dp_group.world_size}",
        )
        warmup_tensor = torch.zeros(1, dtype=torch.int32, device="npu")
        warmup_start_t = time.perf_counter()
        torch.distributed.all_reduce(warmup_tensor, group=dp_group.device_group)
        if torch.npu.is_available():
            torch.npu.synchronize()
        warmup_ms = (time.perf_counter() - warmup_start_t) * 1000.0
        self._log_custom_mode1_worker_memory(
            "after_post_shrink_dp_all_reduce_warmup",
            f"dp_size={dp_group.world_size}",
        )
        del warmup_tensor
        release_dp_warmup_default = (
            "0" if self._has_mode1_lightweight_parity_module() else "1")
        if (_env_flag("VLLM_ASCEND_MODE1_PARITY_RELEASE_DP_WARMUP_CACHE",
                      release_dp_warmup_default)
                and self._has_elastic_lightweight_no_headroom_module()):
            import gc
            gc.collect()
            if torch.npu.is_available():
                torch.npu.empty_cache()
                torch.npu.synchronize()
            self._log_custom_mode1_worker_memory(
                "after_post_shrink_dp_warmup_cache_release",
                f"dp_size={dp_group.world_size}",
            )
        logger.info(
            "Elastic post-shrink DP all_reduce warmup done: rank=%s dp_size=%s total_ms=%.2f",
            self.rank, dp_group.world_size, warmup_ms)

    def _has_hybrid_lossless_module(self) -> bool:
        model_runner = getattr(self, "model_runner", None)
        model = getattr(model_runner, "model", None) if model_runner else None
        if model is None:
            return False

        for module in model.modules():
            if not _is_ascend_fused_moe_module(module):
                continue
            if getattr(module, "elastic_moe_mode", "lossy") != "lossless":
                continue
            if hasattr(module, "_is_hybrid_cpu_swap_enabled") and \
                    module._is_hybrid_cpu_swap_enabled():
                return True
        return False

    def _has_mode3_hybrid_lossless_module(self) -> bool:
        model_runner = getattr(self, "model_runner", None)
        model = getattr(model_runner, "model", None) if model_runner else None
        if model is None:
            return False

        for module in model.modules():
            if not _is_ascend_fused_moe_module(module):
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
        mode1_low_floor_runtime = self._is_mode1_low_floor_elastic_runtime()
        repeat_warmup = bool(int(
            os.getenv(
                "VLLM_ASCEND_REPEAT_POST_SHRINK_MOE_DISPATCH_WARMUP",
                "1" if mode1_low_floor_runtime else "0")))
        if (active_signature
                and active_signature in
                self._post_shrink_moe_dispatch_warmed_active_signatures
                and not repeat_warmup):
            logger.info(
                "Elastic post-shrink MoE dispatch warmup skipped: rank=%s active_ranks=%s reason=already_warmed",
                self.rank, list(active_signature))
            return

        if envs_ascend.VLLM_ASCEND_ELASTIC_EXECUTION_MODE in (4, 5):
            if not _env_flag("VLLM_ASCEND_MODE4_POST_SHRINK_MOE_WARMUP",
                             "0"):
                if active_signature:
                    self._post_shrink_moe_dispatch_warmed_active_signatures.add(
                        active_signature)
                logger.info(
                    "Elastic post-shrink MoE dispatch warmup skipped: rank=%s active_ranks=%s reason=mode4_disabled",
                    self.rank, list(active_signature))
                return
            if model_runner is None or not hasattr(
                    model_runner, "warmup_mode4_remote_npu_moe_compute_only"):
                logger.info(
                    "Elastic post-shrink MoE dispatch warmup skipped: rank=%s active_ranks=%s reason=mode4_no_helper",
                    self.rank, list(active_signature))
                return
            warmup_tokens = int(
                os.getenv("VLLM_ASCEND_MODE4_POST_SHRINK_MOE_WARMUP_TOKENS",
                          "32"))
            if warmup_tokens <= 0:
                if active_signature:
                    self._post_shrink_moe_dispatch_warmed_active_signatures.add(
                        active_signature)
                logger.info(
                    "Elastic post-shrink MoE dispatch warmup skipped: rank=%s active_ranks=%s reason=mode4_no_tokens tokens=%s",
                    self.rank, list(active_signature), warmup_tokens)
                return
            warmup_start_t = time.perf_counter()
            model_runner.warmup_mode4_remote_npu_moe_compute_only(
                warmup_tokens)
            if torch.npu.is_available():
                torch.npu.synchronize()
            if active_signature:
                self._post_shrink_moe_dispatch_warmed_active_signatures.add(
                    active_signature)
            logger.info(
                "Elastic post-shrink MoE dispatch warmup done: rank=%s active_ranks=%s reason=mode4_remote_npu_moe_only tokens=%s total_ms=%.2f",
                self.rank, list(active_signature), warmup_tokens,
                (time.perf_counter() - warmup_start_t) * 1000.0)
            return

        warmup_tokens = int(
            os.getenv("VLLM_ASCEND_POST_SHRINK_MOE_DISPATCH_WARMUP_TOKENS",
                      "32"))
        if warmup_tokens <= 0:
            return
        force_mode1_parity_warmup = os.getenv(
            "VLLM_ASCEND_MODE1_PARITY_POST_SHRINK_WARMUP",
            "1" if mode1_low_floor_runtime else "0").lower() in (
                "1", "true", "yes", "on")
        mode1_parity_warmup_kind = os.getenv(
            "VLLM_ASCEND_MODE1_PARITY_POST_SHRINK_WARMUP_KIND",
            "mc2_dispatcher_only").strip().lower()
        use_mode1_dispatcher_only_warmup = mode1_parity_warmup_kind in (
            "dispatcher", "dispatcher_only", "mc2",
            "mc2_dispatcher", "mc2_dispatcher_only")
        mode1_parity_warmup_helper = (
            "warmup_mode1_parity_mc2_dispatcher_only"
            if use_mode1_dispatcher_only_warmup else
            "warmup_mode1_parity_moe_dispatch_only")
        if mode1_low_floor_runtime and force_mode1_parity_warmup and not hasattr(
                model_runner, mode1_parity_warmup_helper):
            logger.info(
                "Elastic post-shrink MoE dispatch warmup skipped: rank=%s "
                "active_ranks=%s reason=mode1_parity_no_helper kind=%s helper=%s",
                self.rank,
                list(active_signature),
                mode1_parity_warmup_kind,
                mode1_parity_warmup_helper,
            )
            return

        model = getattr(model_runner, "model", None)
        if model is not None:
            for module in model.modules():
                if _module_is_mode1_lightweight_parity(module):
                    if not force_mode1_parity_warmup:
                        if active_signature:
                            self._post_shrink_moe_dispatch_warmed_active_signatures.add(
                                active_signature)
                        logger.info(
                            "Elastic post-shrink MoE dispatch warmup skipped: rank=%s active_ranks=%s reason=mode1_lightweight_parity_no_synthetic_warmup path=%s",
                            self.rank,
                            list(active_signature),
                            _module_mode1_lightweight_parity_path(module),
                        )
                        return
                    logger.info(
                        "Elastic post-shrink MoE dispatch warmup forced for mode1 parity: rank=%s active_ranks=%s tokens=%s kind=%s path=%s",
                        self.rank,
                        list(active_signature),
                        warmup_tokens,
                        mode1_parity_warmup_kind,
                        _module_mode1_lightweight_parity_path(module),
                    )
                    break

        if mode1_low_floor_runtime and force_mode1_parity_warmup:
            self._log_custom_mode1_worker_memory(
                "before_post_shrink_mode1_parity_moe_warmup",
                f"active_ranks={list(active_signature)} "
                f"tokens={warmup_tokens}",
            )
            if _env_flag("VLLM_ASCEND_MODE1_PARITY_PRE_MOE_WARMUP_EMPTY_CACHE",
                         "0"):
                self._log_custom_mode1_worker_memory(
                    "before_post_shrink_mode1_parity_pre_moe_warmup_cache_release",
                    f"active_ranks={list(active_signature)} "
                    f"tokens={warmup_tokens}",
                )
                import gc
                gc.collect()
                if torch.npu.is_available():
                    torch.npu.empty_cache()
                    torch.npu.synchronize()
                self._log_custom_mode1_worker_memory(
                    "after_post_shrink_mode1_parity_pre_moe_warmup_cache_release",
                    f"active_ranks={list(active_signature)} "
                    f"tokens={warmup_tokens}",
                )
            warmup_start_t = time.perf_counter()
            try:
                if use_mode1_dispatcher_only_warmup:
                    model_runner.warmup_mode1_parity_mc2_dispatcher_only(
                        warmup_tokens)
                else:
                    model_runner.warmup_mode1_parity_moe_dispatch_only(
                        warmup_tokens)
            except Exception:
                logger.exception(
                    "Elastic post-shrink MoE dispatch warmup failed: rank=%s "
                    "active_ranks=%s reason=mode1_parity_%s "
                    "tokens=%s",
                    self.rank,
                    list(active_signature),
                    mode1_parity_warmup_kind,
                    warmup_tokens,
                )
                raise
            if torch.npu.is_available():
                torch.npu.synchronize()
            self._log_custom_mode1_worker_memory(
                "after_post_shrink_mode1_parity_moe_warmup",
                f"active_ranks={list(active_signature)} "
                f"tokens={warmup_tokens}",
            )
            if _env_flag("VLLM_ASCEND_MODE1_PARITY_RELEASE_WARMUP_CACHE",
                         "1"):
                self._log_custom_mode1_worker_memory(
                    "before_post_shrink_mode1_parity_moe_warmup_cache_release",
                    f"active_ranks={list(active_signature)} "
                    f"tokens={warmup_tokens}",
                )
                self._release_mode1_moe_comm_runtime_state(
                    "after_post_shrink_mode1_parity_moe_warmup:"
                    f"active_ranks={tuple(active_signature)}")
                gc.collect()
                if torch.npu.is_available():
                    torch.npu.empty_cache()
                    torch.npu.synchronize()
                self._log_custom_mode1_worker_memory(
                    "after_post_shrink_mode1_parity_moe_warmup_cache_release",
                    f"active_ranks={list(active_signature)} "
                    f"tokens={warmup_tokens}",
                )
            if active_signature:
                self._post_shrink_moe_dispatch_warmed_active_signatures.add(
                    active_signature)
            logger.info(
                "Elastic post-shrink MoE dispatch warmup done: rank=%s "
                "active_ranks=%s reason=mode1_parity_%s tokens=%s "
                "total_ms=%.2f",
                self.rank,
                list(active_signature),
                mode1_parity_warmup_kind,
                warmup_tokens,
                (time.perf_counter() - warmup_start_t) * 1000.0,
            )
            return

        if self._has_mode3_hybrid_lossless_module():
            if not _env_flag("VLLM_ASCEND_MODE3_POST_SHRINK_MOE_WARMUP",
                             "0"):
                if active_signature:
                    self._post_shrink_moe_dispatch_warmed_active_signatures.add(
                        active_signature)
                logger.info(
                    "Elastic post-shrink MoE dispatch warmup skipped: rank=%s active_ranks=%s reason=mode3_disabled",
                    self.rank, list(active_signature))
                return
            if not hasattr(model_runner,
                           "warmup_mode3_mc2_dispatcher_only"):
                logger.info(
                    "Elastic post-shrink MoE dispatch warmup skipped: rank=%s active_ranks=%s reason=mode3_no_dispatcher_helper",
                    self.rank, list(active_signature))
                return
            mode3_warmup_tokens = int(
                os.getenv("VLLM_ASCEND_MODE3_POST_SHRINK_MOE_WARMUP_TOKENS",
                          str(warmup_tokens)))
            if mode3_warmup_tokens <= 0:
                if active_signature:
                    self._post_shrink_moe_dispatch_warmed_active_signatures.add(
                        active_signature)
                logger.info(
                    "Elastic post-shrink MoE dispatch warmup skipped: rank=%s active_ranks=%s reason=mode3_no_tokens tokens=%s",
                    self.rank, list(active_signature), mode3_warmup_tokens)
                return
            warmup_start_t = time.perf_counter()
            model_runner.warmup_mode3_mc2_dispatcher_only(mode3_warmup_tokens)
            if torch.npu.is_available():
                torch.npu.synchronize()
            if active_signature:
                self._post_shrink_moe_dispatch_warmed_active_signatures.add(
                    active_signature)
            logger.info(
                "Elastic post-shrink MoE dispatch warmup done: rank=%s active_ranks=%s reason=mode3_mc2_dispatcher_only tokens=%s total_ms=%.2f",
                self.rank, list(active_signature), mode3_warmup_tokens,
                (time.perf_counter() - warmup_start_t) * 1000.0)
            return

        warmup_start_t = time.perf_counter()
        if force_mode1_parity_warmup:
            model_runner.warmup_mode1_parity_moe_dispatch_only(
                warmup_tokens)
        else:
            model_runner._dummy_run(
                warmup_tokens,
                with_prefill=False,
                num_actual_tokens_override=warmup_tokens,
            )
        if torch.npu.is_available():
            torch.npu.synchronize()
        release_warmup_default = (
            "0" if (self._is_mode1_low_floor_elastic_runtime()
                    or self._has_mode1_lightweight_parity_module()) else "1")
        release_warmup_cache = _env_flag(
            "VLLM_ASCEND_MODE1_PARITY_RELEASE_WARMUP_CACHE",
            release_warmup_default)
        if release_warmup_cache:
            self._log_custom_mode1_worker_memory(
                "before_post_shrink_moe_warmup_cache_release",
                f"active_ranks={list(active_signature)}")
            import gc
            gc.collect()
            torch.npu.empty_cache()
            torch.npu.synchronize()
            self._log_custom_mode1_worker_memory(
                "after_post_shrink_moe_warmup_cache_release",
                f"active_ranks={list(active_signature)}")
        if active_signature:
            self._post_shrink_moe_dispatch_warmed_active_signatures.add(
                active_signature)
        logger.info(
            "Elastic post-shrink MoE dispatch warmup done: rank=%s tokens=%s total_ms=%.2f",
            self.rank, warmup_tokens,
            (time.perf_counter() - warmup_start_t) * 1000.0)

    def _release_post_shrink_staging_state(self) -> dict[str, float]:
        release_start_t = time.perf_counter()
        payload_layers = len(getattr(self, "_lossless_shrink_payload", {}))
        import_layers = len(
            getattr(self, "_lossless_preloaded_cpu_import_weights", {}))
        direct_import_layers = len(
            getattr(self, "_lossless_preloaded_direct_import_slots", {}))
        clear_start_t = time.perf_counter()
        self._lossless_shrink_payload = {}
        self._lossless_preloaded_cpu_import_weights = {}
        self._lossless_preloaded_direct_import_slots = {}
        clear_ms = (time.perf_counter() - clear_start_t) * 1000.0

        # Direct mode=1 import lands into preallocated loaded slots; after the
        # metadata dicts are dropped there is no large NPU staging buffer to
        # reclaim. Forcing empty_cache+synchronize here turns all outstanding
        # async NPU/HCCL work into shrink tail latency, which is the 350s stall
        # seen in floor=2 traces. Keep the old cleanup available as an explicit
        # diagnostic/memory-pressure escape hatch.
        trim_default = (
            "0" if (self._is_mode1_low_floor_elastic_runtime()
                    or self._has_mode1_lightweight_parity_module()) else "1")
        trim_npu_cache = _env_flag(
            "VLLM_ASCEND_POST_SHRINK_STAGING_RELEASE_EMPTY_CACHE",
            trim_default)
        sync_after_trim = _env_flag(
            "VLLM_ASCEND_POST_SHRINK_STAGING_RELEASE_SYNC",
            "1" if trim_npu_cache else "0")
        import gc
        gc_start_t = time.perf_counter()
        gc.collect()
        gc_ms = (time.perf_counter() - gc_start_t) * 1000.0
        empty_cache_ms = 0.0
        sync_ms = 0.0
        if torch.npu.is_available() and trim_npu_cache:
            empty_cache_start_t = time.perf_counter()
            torch.npu.empty_cache()
            empty_cache_ms = (
                time.perf_counter() - empty_cache_start_t) * 1000.0
        if torch.npu.is_available() and sync_after_trim:
            sync_start_t = time.perf_counter()
            torch.npu.synchronize()
            sync_ms = (time.perf_counter() - sync_start_t) * 1000.0

        self._log_custom_mode1_worker_memory(
            "after_post_shrink_staging_state_release",
            f"payload_layers={payload_layers} import_layers={import_layers} "
            f"direct_import_layers={direct_import_layers}",
        )
        total_ms = (time.perf_counter() - release_start_t) * 1000.0
        logger.info(
            "Elastic post-shrink staging release breakdown: rank=%s "
            "payload_layers=%s import_layers=%s direct_import_layers=%s "
            "clear_ms=%.2f gc_ms=%.2f empty_cache_ms=%.2f sync_ms=%.2f "
            "trim_npu_cache=%s sync_after_trim=%s total_ms=%.2f",
            self.rank,
            payload_layers,
            import_layers,
            direct_import_layers,
            clear_ms,
            gc_ms,
            empty_cache_ms,
            sync_ms,
            int(trim_npu_cache),
            int(sync_after_trim),
            total_ms,
        )
        return {
            "clear_ms": clear_ms,
            "gc_ms": gc_ms,
            "empty_cache_ms": empty_cache_ms,
            "sync_ms": sync_ms,
            "total_ms": total_ms,
        }

    def _offload_mode1_loaded_weights_after_full_restore(
            self) -> dict[str, float]:
        default_enabled = (
            "1" if self._is_mode1_low_floor_elastic_runtime() else "0")
        enabled = _env_flag(
            "VLLM_ASCEND_MODE1_OFFLOAD_LOADED_WEIGHTS_AFTER_FULL_RESTORE",
            default_enabled)
        stats: dict[str, float] = {
            "enabled": float(enabled),
            "visited_modules": 0.0,
            "transient_released_modules": 0.0,
            "offloaded_modules": 0.0,
            "already_offloaded_modules": 0.0,
            "canonical_offload_enabled": 0.0,
            "estimated_npu_bytes": 0.0,
            "gc_ms": 0.0,
            "empty_cache_ms": 0.0,
            "sync_ms": 0.0,
            "total_ms": 0.0,
        }
        if not enabled:
            return stats

        model_runner = getattr(self, "model_runner", None)
        model = getattr(model_runner, "model", None) if model_runner else None
        if model is None:
            return stats

        storage_diag = _env_flag(
            "VLLM_ASCEND_MODE1_FULL_RESTORE_OFFLOAD_STORAGE_DIAG", "0")
        canonical_offload_enabled = _env_flag(
            "VLLM_ASCEND_MODE1_OFFLOAD_CANONICAL_LOADED_WEIGHTS_AFTER_FULL_RESTORE",
            "0")
        stats["canonical_offload_enabled"] = float(canonical_offload_enabled)

        def _collect_storage_snapshot(label: str) -> dict[str, object]:
            if not storage_diag:
                return {
                    "label": label,
                    "total_unique_bytes": 0,
                    "loaded_param": 0,
                    "runtime_view": 0,
                    "runtime_buffer": 0,
                    "saved_prefix": 0,
                    "cpu_shadow": 0,
                    "p2p_alias_cache": 0,
                    "samples": [],
                }
            group_attrs = {
                "loaded_param": ("w13_weight", "w2_weight"),
                "runtime_view": ("runtime_w13_weight", "runtime_w2_weight"),
                "runtime_buffer": ("runtime_w13_buffer", "runtime_w2_buffer"),
                "saved_prefix": ("lossless_saved_primary_prefix_w13",
                                 "lossless_saved_primary_prefix_w2"),
                "cpu_shadow": ("lossless_cpu_shadow_w13_buffer",
                               "lossless_cpu_shadow_w2_buffer"),
            }
            group_bytes = {key: 0 for key in group_attrs}
            group_bytes["p2p_alias_cache"] = 0
            total_seen: set[int] = set()
            group_seen: dict[str, set[int]] = {
                key: set() for key in group_bytes
            }
            samples: list[str] = []

            def _record_tensor(group_name: str, tensor: object,
                               layer_idx: object, attr_name: str) -> None:
                if not torch.is_tensor(tensor):
                    return
                if tensor.device.type != "npu":
                    return
                ptr, nbytes = _tensor_storage_key_and_nbytes(tensor)
                if not ptr or not nbytes:
                    return
                if ptr not in total_seen:
                    total_seen.add(ptr)
                if ptr not in group_seen[group_name]:
                    group_seen[group_name].add(ptr)
                    group_bytes[group_name] += int(nbytes)
                    if len(samples) < 10:
                        samples.append(
                            "%s:l%s:%s:%s:%s" %
                            (group_name, layer_idx, attr_name, int(nbytes),
                             tuple(tensor.shape)))

            for module in model.modules():
                if not _is_ascend_fused_moe_module(module):
                    continue
                if not _module_uses_lossless_elastic(module):
                    continue
                if int(getattr(module, "elastic_execution_mode", 0) or 0) != 1:
                    continue
                layer_idx = getattr(module, "layer_idx", -1)
                for group_name, attr_names in group_attrs.items():
                    for attr_name in attr_names:
                        _record_tensor(group_name,
                                       getattr(module, attr_name, None),
                                       layer_idx, attr_name)
                alias_cache = getattr(module, "_lossless_p2p_alias_cache", None)
                if isinstance(alias_cache, dict):
                    for alias_idx, alias in enumerate(alias_cache.values()):
                        _record_tensor("p2p_alias_cache", alias, layer_idx,
                                       f"alias{alias_idx}")

            total_unique_bytes = 0
            all_group_ptrs: set[int] = set()
            # Sum unique storages across all tracked groups. A storage can appear
            # in more than one group, for example as both a loaded parameter and
            # a runtime prefix view.
            for ptr_set in group_seen.values():
                all_group_ptrs.update(ptr_set)
            ptr_to_bytes: dict[int, int] = {}
            for module in model.modules():
                if not _is_ascend_fused_moe_module(module):
                    continue
                if not _module_uses_lossless_elastic(module):
                    continue
                if int(getattr(module, "elastic_execution_mode", 0) or 0) != 1:
                    continue
                for attr_names in group_attrs.values():
                    for attr_name in attr_names:
                        tensor = getattr(module, attr_name, None)
                        if torch.is_tensor(tensor) and tensor.device.type == "npu":
                            ptr, nbytes = _tensor_storage_key_and_nbytes(tensor)
                            if ptr in all_group_ptrs:
                                ptr_to_bytes[ptr] = int(nbytes)
                alias_cache = getattr(module, "_lossless_p2p_alias_cache", None)
                if isinstance(alias_cache, dict):
                    for alias in alias_cache.values():
                        if torch.is_tensor(alias) and alias.device.type == "npu":
                            ptr, nbytes = _tensor_storage_key_and_nbytes(alias)
                            if ptr in all_group_ptrs:
                                ptr_to_bytes[ptr] = int(nbytes)
            total_unique_bytes = sum(ptr_to_bytes.values())

            snapshot: dict[str, object] = {
                "label": label,
                "total_unique_bytes": total_unique_bytes,
                "samples": samples,
            }
            snapshot.update(group_bytes)
            if storage_diag:
                logger.info(
                    "Mode1 full-restore offload storage snapshot: rank=%s "
                    "label=%s total_unique_bytes=%s loaded_param=%s "
                    "runtime_view=%s runtime_buffer=%s saved_prefix=%s "
                    "cpu_shadow=%s p2p_alias_cache=%s samples=%s",
                    self.rank,
                    label,
                    total_unique_bytes,
                    group_bytes["loaded_param"],
                    group_bytes["runtime_view"],
                    group_bytes["runtime_buffer"],
                    group_bytes["saved_prefix"],
                    group_bytes["cpu_shadow"],
                    group_bytes["p2p_alias_cache"],
                    ";".join(samples),
                )
            return snapshot

        start_t = time.perf_counter()
        seen_storage: set[int] = set()
        before_snapshot = _collect_storage_snapshot("before_release")
        for module in model.modules():
            if not _is_ascend_fused_moe_module(module):
                continue
            if not _module_uses_lossless_elastic(module):
                continue
            if int(getattr(module, "elastic_execution_mode", 0) or 0) != 1:
                continue
            stats["visited_modules"] += 1.0
            already_offloaded = bool(
                getattr(module, "lossless_loaded_offloaded", False))
            if already_offloaded:
                stats["already_offloaded_modules"] += 1.0

            for tensor_name in (
                    "w13_weight",
                    "w2_weight",
                    "runtime_w13_weight",
                    "runtime_w2_weight",
                    "runtime_w13_buffer",
                    "runtime_w2_buffer",
                    "lossless_saved_primary_prefix_w13",
                    "lossless_saved_primary_prefix_w2",
            ):
                tensor = getattr(module, tensor_name, None)
                if torch.is_tensor(tensor) and tensor.device.type == "npu":
                    ptr, nbytes = _tensor_storage_key_and_nbytes(tensor)
                    if ptr and ptr not in seen_storage:
                        seen_storage.add(ptr)
                        stats["estimated_npu_bytes"] += float(nbytes)

            try:
                release_transient = getattr(
                    module, "release_mode1_full_world_transient_state", None)
                if callable(release_transient):
                    release_transient()
                    stats["transient_released_modules"] += 1.0
                if canonical_offload_enabled and not already_offloaded:
                    offload = getattr(
                        module, "offload_lossless_loaded_weights_to_cpu",
                        None)
                    if callable(offload):
                        offload()
                        stats["offloaded_modules"] += 1.0
            except Exception:
                logger.exception(
                    "Mode1 full-restore transient cleanup failed: "
                    "rank=%s layer=%s",
                    self.rank,
                    getattr(module, "layer_idx", -1),
                )
                raise

        after_release_snapshot = _collect_storage_snapshot("after_release")
        gc_start_t = time.perf_counter()
        gc.collect()
        stats["gc_ms"] = (time.perf_counter() - gc_start_t) * 1000.0

        if torch.npu.is_available():
            empty_cache_start_t = time.perf_counter()
            torch.npu.empty_cache()
            stats["empty_cache_ms"] = (
                time.perf_counter() - empty_cache_start_t) * 1000.0
            if _env_flag(
                    "VLLM_ASCEND_MODE1_SYNC_AFTER_FULL_RESTORE_WEIGHT_OFFLOAD",
                    "1"):
                sync_start_t = time.perf_counter()
                torch.npu.synchronize()
                stats["sync_ms"] = (
                    time.perf_counter() - sync_start_t) * 1000.0

        after_gc_snapshot = _collect_storage_snapshot("after_gc")
        stats["pre_tracked_npu_bytes"] = float(
            before_snapshot.get("total_unique_bytes", 0))
        stats["post_release_tracked_npu_bytes"] = float(
            after_release_snapshot.get("total_unique_bytes", 0))
        stats["post_gc_tracked_npu_bytes"] = float(
            after_gc_snapshot.get("total_unique_bytes", 0))
        stats["total_ms"] = (time.perf_counter() - start_t) * 1000.0
        if (stats["transient_released_modules"] > 0.0
                or stats["offloaded_modules"] > 0.0
                or _env_flag("VLLM_ASCEND_MODE1_LOG_ZERO_FULL_RESTORE_OFFLOAD",
                             "0")):
            logger.info(
                "Mode1 full-restore transient cleanup: rank=%s "
                "visited_modules=%.0f transient_released_modules=%.0f "
                "canonical_offload_enabled=%.0f offloaded_modules=%.0f "
                "already_offloaded_modules=%.0f estimated_npu_bytes=%.0f "
                "pre_tracked_npu_bytes=%.0f post_release_tracked_npu_bytes=%.0f "
                "post_gc_tracked_npu_bytes=%.0f "
                "gc_ms=%.2f empty_cache_ms=%.2f sync_ms=%.2f total_ms=%.2f",
                self.rank,
                stats["visited_modules"],
                stats["transient_released_modules"],
                stats["canonical_offload_enabled"],
                stats["offloaded_modules"],
                stats["already_offloaded_modules"],
                stats["estimated_npu_bytes"],
                stats["pre_tracked_npu_bytes"],
                stats["post_release_tracked_npu_bytes"],
                stats["post_gc_tracked_npu_bytes"],
                stats["gc_ms"],
                stats["empty_cache_ms"],
                stats["sync_ms"],
                stats["total_ms"],
            )
        return stats

    def _full_moe_elastic_aclgraph_enabled(self) -> bool:
        return bool(
            envs_ascend.VLLM_ASCEND_ELASTIC_ACLGRAPH
            and envs_ascend.VLLM_ASCEND_ELASTIC_ACLGRAPH_CAPTURE_MOE
            and getattr(self.model_runner, "use_aclgraph", False))

    def _invalidate_full_moe_elastic_aclgraph(self, stage: str) -> int:
        if not self._full_moe_elastic_aclgraph_enabled():
            return 0
        from vllm_ascend.compilation.acl_graph import (
            clear_aclgraph_caches,
            disable_aclgraph_dispatch,
        )

        cleared_keys = disable_aclgraph_dispatch(self.model_runner)
        cleared_entries = clear_aclgraph_caches(
            getattr(self.model_runner, "model", None))
        logger.warning(
            "Elastic full-MoE ACLGraph invalidated: rank=%s stage=%s "
            "cached_entries=%s dispatch_keys=%s",
            self.rank,
            stage,
            cleared_entries,
            cleared_keys,
        )
        return cleared_entries

    def _recapture_full_moe_elastic_aclgraph(
            self, active_ranks: list[int], stage: str) -> None:
        if not self._full_moe_elastic_aclgraph_enabled():
            return
        current_rank = torch.distributed.get_rank()
        if current_rank not in active_ranks:
            raise RuntimeError(
                "Detached rank attempted full-MoE ACLGraph recapture: "
                f"rank={current_rank} active_ranks={active_ranks} stage={stage}")
        dp_group = get_dp_group()
        ep_group = get_ep_group()
        if (dp_group.world_size != len(active_ranks)
                or ep_group.world_size != len(active_ranks)):
            raise RuntimeError(
                "Elastic full-MoE ACLGraph recapture requires refreshed groups: "
                f"rank={current_rank} active_ranks={active_ranks} "
                f"dp={dp_group.world_size} ep={ep_group.world_size} stage={stage}")
        torch.distributed.barrier(group=dp_group.cpu_group)
        logger.warning(
            "Elastic full-MoE ACLGraph topology capture starting: "
            "rank=%s active_ranks=%s stage=%s",
            self.rank,
            active_ranks,
            stage,
        )
        self.model_runner.capture_model()
        torch.distributed.barrier(group=dp_group.cpu_group)
        logger.warning(
            "Elastic full-MoE ACLGraph topology capture finished: "
            "rank=%s active_ranks=%s stage=%s",
            self.rank,
            active_ranks,
            stage,
        )

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
        previous_active_ranks_for_mode4 = self._get_previous_active_ranks_for_shrink(
            world_size)
        (_source_ranks_for_barrier,
         source_cpu_group_for_barrier,
         _) = self._get_shrink_source_group_state(world_group)
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
        aclgraph_invalidate_start_t = time.perf_counter()
        self._invalidate_full_moe_elastic_aclgraph(
            f"before_shrink_to_{tuple(active_ranks)}")
        aclgraph_invalidate_ms = (
            time.perf_counter() - aclgraph_invalidate_start_t) * 1000.0
        prefetch_drain_ms = 0.0
        prepare_payload_ms = 0.0
        preload_import_ms = 0.0
        detach_ms = 0.0
        remote_cache_phase_ms = 0.0
        remote_cache_start_ms = 0.0
        remote_cache_barrier1_ms = 0.0
        remote_payload_warmup_ms = 0.0
        next_block_prime_ms = 0.0
        remote_cache_barrier2_ms = 0.0
        release_staging_ms = 0.0
        post_preload_source_barrier_ms = 0.0
        release_staging_clear_ms = 0.0
        release_staging_gc_ms = 0.0
        release_staging_empty_cache_ms = 0.0
        release_staging_sync_ms = 0.0
        drop_stale_group_cache_ms = 0.0
        drop_old_floor_group_cache_ms = 0.0
        drop_old_floor_group_call_wall_ms = 0.0
        drop_old_floor_group_destroy_ms = 0.0
        drop_old_floor_group_cleanup_ms = 0.0
        drop_old_floor_group_cleanup_gc_ms = 0.0
        drop_old_floor_group_cleanup_empty_cache_ms = 0.0
        drop_old_floor_group_cleanup_sync_ms = 0.0
        drop_old_floor_group_cleanup_memory_log_ms = 0.0
        drop_old_floor_group_memory_log_ms = 0.0
        drop_old_floor_group_row_log_ms = 0.0
        drop_old_floor_group_summary_log_ms = 0.0
        drop_old_floor_group_signature_sweep_ms = 0.0
        drop_old_floor_group_release_ref_ms = 0.0
        old_floor_pre_memory_log_ms = 0.0
        old_floor_post_memory_log_ms = 0.0
        drop_old_floor_group_dropped_groups = 0.0
        drop_old_floor_group_deferred_groups = 0.0
        drop_old_floor_group_dropped_signatures = 0.0
        old_floor_group_drop_barrier_ms = 0.0
        pre_refresh_source_barrier_ms = 0.0
        post_old_floor_group_drop_barrier_ms = 0.0
        non_member_sequence_ms = 0.0
        non_member_sequence_dp_ms = 0.0
        non_member_sequence_ep_ms = 0.0
        non_member_sequence_mc2_ms = 0.0
        non_member_sequence_device_new_group_ms = 0.0
        non_member_sequence_cpu_new_group_ms = 0.0
        non_member_sequence_destroy_ms = 0.0
        non_member_sequence_signature_ms = 0.0
        non_member_sequence_log_ms = 0.0
        non_member_sequence_groups = 0.0
        release_mode5_cache_state_ms = 0.0

        def _accumulate_non_member_sequence_stats(
                group_name: str, stats: dict[str, float]) -> None:
            nonlocal non_member_sequence_ms
            nonlocal non_member_sequence_dp_ms
            nonlocal non_member_sequence_ep_ms
            nonlocal non_member_sequence_mc2_ms
            nonlocal non_member_sequence_device_new_group_ms
            nonlocal non_member_sequence_cpu_new_group_ms
            nonlocal non_member_sequence_destroy_ms
            nonlocal non_member_sequence_signature_ms
            nonlocal non_member_sequence_log_ms
            nonlocal non_member_sequence_groups
            total = float(stats.get("total_ms", 0.0))
            non_member_sequence_ms += total
            if group_name == "dp":
                non_member_sequence_dp_ms += total
            elif group_name == "ep":
                non_member_sequence_ep_ms += total
            elif group_name == "mc2":
                non_member_sequence_mc2_ms += total
            non_member_sequence_device_new_group_ms += float(
                stats.get("device_new_group_ms", 0.0))
            non_member_sequence_cpu_new_group_ms += float(
                stats.get("cpu_new_group_ms", 0.0))
            non_member_sequence_destroy_ms += float(
                stats.get("destroy_ms", 0.0))
            non_member_sequence_signature_ms += float(
                stats.get("signature_ms", 0.0))
            non_member_sequence_log_ms += float(stats.get("log_ms", 0.0))
            non_member_sequence_groups += float(stats.get("groups", 0.0))

        def _accumulate_old_floor_drop_stats(
                stats: dict[str, float], call_wall_ms: float) -> None:
            nonlocal drop_old_floor_group_cache_ms
            nonlocal drop_old_floor_group_call_wall_ms
            nonlocal drop_old_floor_group_destroy_ms
            nonlocal drop_old_floor_group_cleanup_ms
            nonlocal drop_old_floor_group_cleanup_gc_ms
            nonlocal drop_old_floor_group_cleanup_empty_cache_ms
            nonlocal drop_old_floor_group_cleanup_sync_ms
            nonlocal drop_old_floor_group_cleanup_memory_log_ms
            nonlocal drop_old_floor_group_memory_log_ms
            nonlocal drop_old_floor_group_row_log_ms
            nonlocal drop_old_floor_group_summary_log_ms
            nonlocal drop_old_floor_group_signature_sweep_ms
            nonlocal drop_old_floor_group_release_ref_ms
            nonlocal drop_old_floor_group_dropped_groups
            nonlocal drop_old_floor_group_deferred_groups
            nonlocal drop_old_floor_group_dropped_signatures
            drop_old_floor_group_call_wall_ms += call_wall_ms
            drop_old_floor_group_cache_ms += float(stats.get("total_ms", 0.0))
            drop_old_floor_group_destroy_ms += float(
                stats.get("destroy_ms", 0.0))
            drop_old_floor_group_cleanup_ms += float(
                stats.get("cleanup_ms", 0.0))
            drop_old_floor_group_cleanup_gc_ms += float(
                stats.get("cleanup_gc_ms", 0.0))
            drop_old_floor_group_cleanup_empty_cache_ms += float(
                stats.get("cleanup_empty_cache_ms", 0.0))
            drop_old_floor_group_cleanup_sync_ms += float(
                stats.get("cleanup_sync_ms", 0.0))
            drop_old_floor_group_cleanup_memory_log_ms += float(
                stats.get("cleanup_memory_log_ms", 0.0))
            drop_old_floor_group_memory_log_ms += float(
                stats.get("memory_log_ms", 0.0))
            drop_old_floor_group_row_log_ms += float(
                stats.get("row_log_ms", 0.0))
            drop_old_floor_group_summary_log_ms += float(
                stats.get("summary_log_ms", 0.0))
            drop_old_floor_group_signature_sweep_ms += float(
                stats.get("signature_sweep_ms", 0.0))
            drop_old_floor_group_release_ref_ms += float(
                stats.get("release_ref_ms", 0.0))
            drop_old_floor_group_dropped_groups += float(
                stats.get("dropped_groups", 0.0))
            drop_old_floor_group_deferred_groups += float(
                stats.get("deferred_groups", 0.0))
            drop_old_floor_group_dropped_signatures += float(
                stats.get("dropped_signatures", 0.0))

        if self._has_mode4_remote_npu_lightweight_module():
            prefetch_drain_start_t = time.perf_counter()
            self._drain_mode4_double_buffer_prefetch(
                f"before_elastic_shrink:{active_ranks}")
            prefetch_drain_ms = (
                time.perf_counter() - prefetch_drain_start_t) * 1000.0
        if (envs_ascend.VLLM_ASCEND_ELASTIC_EXECUTION_MODE == 1
                and _env_flag("VLLM_ASCEND_MODE1_SYNC_BEFORE_SHRINK_PAYLOAD",
                              "1")
                and torch.npu.is_available()):
            sync_start_t = time.perf_counter()
            torch.npu.synchronize()
            logger.info(
                "Elastic mode1 pre-shrink payload NPU sync done: "
                "rank=%s active_ranks=%s total_ms=%.2f",
                self.rank, active_ranks,
                (time.perf_counter() - sync_start_t) * 1000.0)

        prepare_payload_start_t = time.perf_counter()
        self._prepare_lossless_shrink_payload(active_ranks, world_group)
        prepare_payload_ms = (
            time.perf_counter() - prepare_payload_start_t) * 1000.0
        is_active_rank = current_rank in active_ranks
        mode1_low_floor_runtime = self._is_mode1_low_floor_elastic_runtime()
        preload_import_start_t = time.perf_counter()
        self._preload_lossless_shrink_import_weights(active_ranks, world_group)
        preload_import_ms = (
            time.perf_counter() - preload_import_start_t) * 1000.0
        if (mode1_low_floor_runtime and _env_flag(
                "VLLM_ASCEND_MODE1_PARITY_BARRIER_AFTER_PRELOAD_BEFORE_RELEASE",
                "1")):
            post_preload_source_barrier_start_t = time.perf_counter()
            self._mode1_source_cpu_p2p_barrier(
                _source_ranks_for_barrier,
                world_group,
                38088 + len(active_ranks) * 10,
            )
            post_preload_source_barrier_ms = (
                time.perf_counter() -
                post_preload_source_barrier_start_t) * 1000.0
            logger.info(
                "Elastic mode1 post-preload source barrier done: rank=%s "
                "active_ranks=%s source_ranks=%s is_active=%s total_ms=%.2f",
                self.rank,
                active_ranks,
                _source_ranks_for_barrier,
                int(is_active_rank),
                post_preload_source_barrier_ms,
            )
        if not is_active_rank and not self.elastic_parallel_detached:
            detach_start_t = time.perf_counter()
            self._detach_from_elastic_parallel_groups()
            detach_ms = (time.perf_counter() - detach_start_t) * 1000.0
            logger.info(
                "Elastic parallel detach done: rank=%s active_ranks=%s total_ms=%.2f",
                self.rank, active_ranks, detach_ms)
            elastic_group_ranks = [active_ranks]
            seq_stats = self._advance_group_creation_sequence_for_non_member(
                elastic_group_ranks, backend, "dp")
            _accumulate_non_member_sequence_stats("dp", seq_stats)
            seq_stats = self._advance_group_creation_sequence_for_non_member(
                elastic_group_ranks, backend, "ep")
            _accumulate_non_member_sequence_stats("ep", seq_stats)
            seq_stats = self._advance_group_creation_sequence_for_non_member(
                elastic_group_ranks, backend, "mc2")
            _accumulate_non_member_sequence_stats("mc2", seq_stats)
        old_floor_group_dropped_before_rebuild = False
        if (mode1_low_floor_runtime and _env_flag(
                "VLLM_ASCEND_MODE1_PARITY_DROP_OLD_FLOOR_BEFORE_REBUILD",
                "1")):
            keep_group_ranks = tuple(int(rank) for rank in active_ranks)
            old_floor_pre_memory_log_start_t = time.perf_counter()
            self._log_custom_mode1_worker_memory(
                "before_old_floor_group_cache_drop_before_rebuild",
                f"keep_ranks={keep_group_ranks}",
            )
            old_floor_pre_memory_log_ms += (
                time.perf_counter() -
                old_floor_pre_memory_log_start_t) * 1000.0
            drop_old_floor_group_cache_start_t = time.perf_counter()
            live_release_stats = (
                self._release_live_mode1_old_floor_groups_before_rebuild(
                    active_ranks))
            drop_old_floor_stats = (
                self._drop_old_floor_cached_elastic_parallel_groups_after_mode1_shrink(
                    keep_group_ranks)
            )
            drop_old_floor_stats["dropped_groups"] = (
                float(drop_old_floor_stats.get("dropped_groups", 0.0)) +
                float(live_release_stats.get("released_groups", 0.0)))
            drop_old_floor_stats["destroy_ms"] = (
                float(drop_old_floor_stats.get("destroy_ms", 0.0)) +
                float(live_release_stats.get("total_ms", 0.0)))
            drop_old_floor_group_call_wall = (
                time.perf_counter() -
                drop_old_floor_group_cache_start_t) * 1000.0
            _accumulate_old_floor_drop_stats(
                drop_old_floor_stats, drop_old_floor_group_call_wall)
            old_floor_group_dropped_before_rebuild = True

            post_old_floor_group_drop_barrier_start_t = time.perf_counter()
            self._mode1_source_cpu_p2p_barrier(
                _source_ranks_for_barrier,
                world_group,
                38092 + len(active_ranks) * 10,
            )
            post_old_floor_group_drop_barrier_ms += (
                time.perf_counter() -
                post_old_floor_group_drop_barrier_start_t) * 1000.0

            old_floor_post_memory_log_start_t = time.perf_counter()
            self._log_custom_mode1_worker_memory(
                "after_old_floor_group_cache_drop_before_rebuild",
                f"keep_ranks={keep_group_ranks}",
            )
            old_floor_post_memory_log_ms += (
                time.perf_counter() -
                old_floor_post_memory_log_start_t) * 1000.0

        # Stash the currently live DP/EP/MC2 group first, then sweep stale
        # cache entries before per-layer refresh. Dropping the cache before
        # active ranks rebuild can retire the same object while it is still the
        # live parallel-state group, which previously led to ghost cached
        # wrappers and later HCCL workspace pressure.
        drop_old_floor_before_rebuild_default = "0"
        if (not old_floor_group_dropped_before_rebuild
                and mode1_low_floor_runtime and _env_flag(
                "VLLM_ASCEND_MODE1_PARITY_DROP_OLD_FLOOR_BEFORE_REBUILD",
                drop_old_floor_before_rebuild_default)):
            keep_group_ranks = tuple(int(rank) for rank in active_ranks)
            old_floor_pre_memory_log_start_t = time.perf_counter()
            self._log_custom_mode1_worker_memory(
                "before_old_floor_group_cache_drop_before_rebuild",
                f"keep_ranks={keep_group_ranks}",
            )
            old_floor_pre_memory_log_ms += (
                time.perf_counter() -
                old_floor_pre_memory_log_start_t) * 1000.0
            drop_old_floor_group_cache_start_t = time.perf_counter()
            drop_old_floor_stats = (
                self._drop_old_floor_cached_elastic_parallel_groups_after_mode1_shrink(
                    keep_group_ranks)
            )
            drop_old_floor_group_call_wall = (
                time.perf_counter() -
                drop_old_floor_group_cache_start_t) * 1000.0
            _accumulate_old_floor_drop_stats(
                drop_old_floor_stats, drop_old_floor_group_call_wall)
            old_floor_group_dropped_before_rebuild = True

            post_old_floor_group_drop_barrier_start_t = time.perf_counter()
            self._mode1_source_cpu_p2p_barrier(
                _source_ranks_for_barrier,
                world_group,
                38092 + len(active_ranks) * 10,
            )
            post_old_floor_group_drop_barrier_ms += (
                time.perf_counter() -
                post_old_floor_group_drop_barrier_start_t) * 1000.0

            old_floor_post_memory_log_start_t = time.perf_counter()
            self._log_custom_mode1_worker_memory(
                "after_old_floor_group_cache_drop_before_rebuild",
                f"keep_ranks={keep_group_ranks}",
            )
            old_floor_post_memory_log_ms += (
                time.perf_counter() -
                old_floor_post_memory_log_start_t) * 1000.0

        rebuild_ms = 0.0
        rebuild_dp_ms = 0.0
        rebuild_ep_ms = 0.0
        rebuild_mc2_ms = 0.0
        refresh_ms = 0.0
        warmup_ms = 0.0
        if is_active_rank:
            elastic_group_ranks = [active_ranks]
            rebuild_start_t = time.perf_counter()
            with set_current_vllm_config(self.vllm_config):
                rebuild_dp_start_t = time.perf_counter()
                self._rebuild_group(vllm_ps, "_DP", elastic_group_ranks,
                                    world_group, backend, "dp")
                rebuild_dp_ms = (
                    time.perf_counter() - rebuild_dp_start_t) * 1000.0
                rebuild_ep_start_t = time.perf_counter()
                self._rebuild_group(vllm_ps, "_EP", elastic_group_ranks,
                                    world_group, backend, "ep")
                rebuild_ep_ms = (
                    time.perf_counter() - rebuild_ep_start_t) * 1000.0
                rebuild_mc2_start_t = time.perf_counter()
                self._rebuild_group(ascend_ps, "_MC2", elastic_group_ranks,
                                    world_group, backend, "mc2")
                rebuild_mc2_ms = (
                    time.perf_counter() - rebuild_mc2_start_t) * 1000.0
            rebuild_ms = (time.perf_counter() - rebuild_start_t) * 1000.0

        if (mode1_low_floor_runtime and _env_flag(
                "VLLM_ASCEND_MODE1_PARITY_BARRIER_BEFORE_REFRESH", "1")):
            pre_refresh_source_barrier_start_t = time.perf_counter()
            # Later shrink RPCs only include the previous active/source ranks.
            # Align those ranks after detach/non-member sequence advancement and
            # active group rebuild, before active ranks enter per-layer MC2/EP
            # dispatcher refresh.
            self._mode1_source_cpu_p2p_barrier(
                _source_ranks_for_barrier,
                world_group,
                38100 + len(active_ranks) * 10,
            )
            pre_refresh_source_barrier_ms = (
                time.perf_counter() -
                pre_refresh_source_barrier_start_t) * 1000.0

        old_floor_group_dropped_before_refresh = False
        if (mode1_low_floor_runtime
                and not old_floor_group_dropped_before_rebuild
                and _env_flag(
                    "VLLM_ASCEND_MODE1_PARITY_DROP_OLD_FLOOR_BEFORE_REFRESH",
                    "1")):
            keep_group_ranks = tuple(int(rank) for rank in active_ranks)
            old_floor_pre_memory_log_start_t = time.perf_counter()
            self._log_custom_mode1_worker_memory(
                "before_old_floor_group_cache_drop_before_refresh",
                f"keep_ranks={keep_group_ranks}",
            )
            old_floor_pre_memory_log_ms += (
                time.perf_counter() -
                old_floor_pre_memory_log_start_t) * 1000.0
            drop_old_floor_group_cache_start_t = time.perf_counter()
            drop_old_floor_stats = (
                self._drop_old_floor_cached_elastic_parallel_groups_after_mode1_shrink(
                    keep_group_ranks)
            )
            drop_old_floor_group_call_wall = (
                time.perf_counter() -
                drop_old_floor_group_cache_start_t) * 1000.0
            _accumulate_old_floor_drop_stats(
                drop_old_floor_stats, drop_old_floor_group_call_wall)
            old_floor_group_dropped_before_refresh = True

            post_old_floor_group_drop_barrier_start_t = time.perf_counter()
            self._mode1_source_cpu_p2p_barrier(
                _source_ranks_for_barrier,
                world_group,
                38102 + len(active_ranks) * 10,
            )
            post_old_floor_group_drop_barrier_ms = (
                time.perf_counter() -
                post_old_floor_group_drop_barrier_start_t) * 1000.0

            old_floor_post_memory_log_start_t = time.perf_counter()
            self._log_custom_mode1_worker_memory(
                "after_old_floor_group_cache_drop_before_refresh",
                f"keep_ranks={keep_group_ranks}",
            )
            old_floor_post_memory_log_ms += (
                time.perf_counter() -
                old_floor_post_memory_log_start_t) * 1000.0

        refresh_start_t = time.perf_counter()
        self._reset_elastic_moe_comm_setup_cache(
            f"before_shrink_refresh active_ranks={tuple(active_ranks)} "
            f"is_active={int(is_active_rank)}")
        with set_current_vllm_config(self.vllm_config):
            self._refresh_elastic_parallel_state(
                active_ranks,
                world_group,
                participate_only=not is_active_rank)
        refresh_ms = (time.perf_counter() - refresh_start_t) * 1000.0

        if self._has_mode4_remote_npu_lightweight_module():
            # Only cache/non-active ranks need to clear stale runtime slots
            # before they turn into remote expert-cache services. Active ranks
            # have just refreshed their new double-buffer mapping above; wiping
            # it here leaves the following decode with no valid scheduled slot.
            remote_cache_phase_start_t = time.perf_counter()
            if not is_active_rank:
                self._reset_mode4_double_buffer_runtime_slots(
                    "before_mode4_cache_service_start")
            logger.info(
                "Mode%s post-shrink remote-cache phase enter: rank=%s active_ranks=%s is_active=%s",
                envs_ascend.VLLM_ASCEND_ELASTIC_EXECUTION_MODE,
                self.rank,
                active_ranks,
                int(is_active_rank),
            )
            remote_cache_start_t = time.perf_counter()
            self._start_mode4_remote_cache_service(active_ranks, world_group)
            remote_cache_start_ms = (
                time.perf_counter() - remote_cache_start_t) * 1000.0
            # This block is executed by the previous active ranks, including
            # ranks that have just detached and no longer have a current DP
            # group. Use the pre-shrink source CPU group instead of get_dp_group().
            mode4_barrier_group = source_cpu_group_for_barrier
            logger.info(
                "Mode%s post-shrink remote-cache barrier1 enter: rank=%s active_ranks=%s is_active=%s",
                envs_ascend.VLLM_ASCEND_ELASTIC_EXECUTION_MODE,
                self.rank,
                active_ranks,
                int(is_active_rank),
            )
            remote_cache_barrier1_start_t = time.perf_counter()
            torch.distributed.barrier(group=mode4_barrier_group)
            remote_cache_barrier1_ms = (
                time.perf_counter() - remote_cache_barrier1_start_t) * 1000.0
            logger.info(
                "Mode%s post-shrink remote-cache barrier1 done: rank=%s active_ranks=%s is_active=%s",
                envs_ascend.VLLM_ASCEND_ELASTIC_EXECUTION_MODE,
                self.rank,
                active_ranks,
                int(is_active_rank),
            )
            logger.info(
                "Mode%s post-shrink remote payload warmup enter: rank=%s active_ranks=%s is_active=%s",
                envs_ascend.VLLM_ASCEND_ELASTIC_EXECUTION_MODE,
                self.rank,
                active_ranks,
                int(is_active_rank),
            )
            remote_payload_warmup_start_t = time.perf_counter()
            self._mode4_warmup_remote_payload_fetch(active_ranks, world_group)
            remote_payload_warmup_ms = (
                time.perf_counter() -
                remote_payload_warmup_start_t) * 1000.0
            logger.info(
                "Mode%s post-shrink remote payload warmup done: rank=%s active_ranks=%s is_active=%s",
                envs_ascend.VLLM_ASCEND_ELASTIC_EXECUTION_MODE,
                self.rank,
                active_ranks,
                int(is_active_rank),
            )
            self._ensure_mode4_double_buffer_manager()
            logger.info(
                "Mode%s post-shrink next-block prime enter: rank=%s active_ranks=%s is_active=%s",
                envs_ascend.VLLM_ASCEND_ELASTIC_EXECUTION_MODE,
                self.rank,
                active_ranks,
                int(is_active_rank),
            )
            next_block_prime_start_t = time.perf_counter()
            self._mode4_prime_next_block_runtime_slot(active_ranks)
            next_block_prime_ms = (
                time.perf_counter() - next_block_prime_start_t) * 1000.0
            logger.info(
                "Mode%s post-shrink next-block prime done: rank=%s active_ranks=%s is_active=%s",
                envs_ascend.VLLM_ASCEND_ELASTIC_EXECUTION_MODE,
                self.rank,
                active_ranks,
                int(is_active_rank),
            )
            logger.info(
                "Mode%s post-shrink remote-cache barrier2 enter: rank=%s active_ranks=%s is_active=%s",
                envs_ascend.VLLM_ASCEND_ELASTIC_EXECUTION_MODE,
                self.rank,
                active_ranks,
                int(is_active_rank),
            )
            remote_cache_barrier2_start_t = time.perf_counter()
            torch.distributed.barrier(group=mode4_barrier_group)
            remote_cache_barrier2_ms = (
                time.perf_counter() - remote_cache_barrier2_start_t) * 1000.0
            logger.info(
                "Mode%s post-shrink remote-cache barrier2 done: rank=%s active_ranks=%s is_active=%s",
                envs_ascend.VLLM_ASCEND_ELASTIC_EXECUTION_MODE,
                self.rank,
                active_ranks,
                int(is_active_rank),
            )
            remote_cache_phase_ms = (
                time.perf_counter() - remote_cache_phase_start_t) * 1000.0

        release_staging_start_t = time.perf_counter()
        release_staging_stats = self._release_post_shrink_staging_state()
        release_staging_ms = (
            time.perf_counter() - release_staging_start_t) * 1000.0
        release_staging_clear_ms = float(
            release_staging_stats.get("clear_ms", 0.0))
        release_staging_gc_ms = float(release_staging_stats.get("gc_ms", 0.0))
        release_staging_empty_cache_ms = float(
            release_staging_stats.get("empty_cache_ms", 0.0))
        release_staging_sync_ms = float(
            release_staging_stats.get("sync_ms", 0.0))

        if self._should_drop_stale_group_cache_after_elastic_shrink(
                active_ranks, world_size):
            keep_group_ranks = tuple(int(rank) for rank in active_ranks)
            self._log_custom_mode1_worker_memory(
                "before_stale_group_cache_drop_after_shrink",
                f"keep_ranks={keep_group_ranks}",
            )
            drop_stale_group_cache_start_t = time.perf_counter()
            self._drop_stale_cached_elastic_parallel_groups(keep_group_ranks)
            drop_stale_group_cache_ms = (
                time.perf_counter() -
                drop_stale_group_cache_start_t) * 1000.0
            self._log_custom_mode1_worker_memory(
                "after_stale_group_cache_drop_after_shrink",
                f"keep_ranks={keep_group_ranks}",
            )
        elif mode1_low_floor_runtime:
            keep_group_ranks = tuple(int(rank) for rank in active_ranks)
            if (not old_floor_group_dropped_before_refresh and _env_flag(
                    "VLLM_ASCEND_MODE1_PARITY_BARRIER_BEFORE_OLD_FLOOR_DROP",
                    "1")):
                old_floor_group_drop_barrier_start_t = time.perf_counter()
                # Only the previous active/source ranks participate in later
                # shrink RPCs. A full-world barrier would wait for ranks that
                # already detached after the earlier shrink step.
                self._mode1_source_cpu_p2p_barrier(
                    _source_ranks_for_barrier,
                    world_group,
                    38104 + len(active_ranks) * 10,
                )
                old_floor_group_drop_barrier_ms = (
                    time.perf_counter() -
                    old_floor_group_drop_barrier_start_t) * 1000.0
            if not old_floor_group_dropped_before_refresh:
                old_floor_pre_memory_log_start_t = time.perf_counter()
                self._log_custom_mode1_worker_memory(
                    "before_old_floor_group_cache_drop_after_shrink",
                    f"keep_ranks={keep_group_ranks}",
                )
                old_floor_pre_memory_log_ms += (
                    time.perf_counter() -
                    old_floor_pre_memory_log_start_t) * 1000.0
                drop_old_floor_group_cache_start_t = time.perf_counter()
                drop_old_floor_stats = (
                    self._drop_old_floor_cached_elastic_parallel_groups_after_mode1_shrink(
                        keep_group_ranks)
                )
                drop_old_floor_group_call_wall = (
                    time.perf_counter() -
                    drop_old_floor_group_cache_start_t) * 1000.0
                _accumulate_old_floor_drop_stats(
                    drop_old_floor_stats, drop_old_floor_group_call_wall)
                old_floor_post_memory_log_start_t = time.perf_counter()
                self._log_custom_mode1_worker_memory(
                    "after_old_floor_group_cache_drop_after_shrink",
                    f"keep_ranks={keep_group_ranks}",
                )
                old_floor_post_memory_log_ms += (
                    time.perf_counter() -
                    old_floor_post_memory_log_start_t) * 1000.0

        if (not is_active_rank
                and int(envs_ascend.VLLM_ASCEND_ELASTIC_EXECUTION_MODE) == 5):
            release_mode5_cache_state_start_t = time.perf_counter()
            self._release_mode5_cache_rank_runtime_state()
            release_mode5_cache_state_ms = (
                time.perf_counter() -
                release_mode5_cache_state_start_t) * 1000.0

        if is_active_rank:
            warmup_start_t = time.perf_counter()
            self._warmup_post_shrink_dp_collectives()
            self._warmup_post_shrink_moe_dispatch(active_ranks)
            warmup_ms = (time.perf_counter() - warmup_start_t) * 1000.0

        aclgraph_recapture_ms = 0.0
        if is_active_rank and self._full_moe_elastic_aclgraph_enabled():
            aclgraph_recapture_start_t = time.perf_counter()
            self._recapture_full_moe_elastic_aclgraph(
                active_ranks, f"after_shrink_to_{tuple(active_ranks)}")
            aclgraph_recapture_ms = (
                time.perf_counter() - aclgraph_recapture_start_t) * 1000.0

        total_ms = (time.perf_counter() - start_t) * 1000.0
        hidden_tail_ms = total_ms - (
            prefetch_drain_ms + prepare_payload_ms + preload_import_ms +
            post_preload_source_barrier_ms + detach_ms + rebuild_ms +
            pre_refresh_source_barrier_ms + non_member_sequence_ms +
            refresh_ms + remote_cache_phase_ms + release_staging_ms +
            drop_stale_group_cache_ms + old_floor_group_drop_barrier_ms +
            drop_old_floor_group_call_wall_ms + old_floor_pre_memory_log_ms +
            old_floor_post_memory_log_ms +
            post_old_floor_group_drop_barrier_ms +
            release_mode5_cache_state_ms + warmup_ms +
            aclgraph_invalidate_ms + aclgraph_recapture_ms)
        logger.info(
            "Elastic parallel shrink phase breakdown: rank=%s active_ranks=%s is_active=%s prefetch_drain_ms=%.2f prepare_payload_ms=%.2f preload_import_ms=%.2f post_preload_source_barrier_ms=%.2f detach_ms=%.2f rebuild_ms=%.2f rebuild_dp_ms=%.2f rebuild_ep_ms=%.2f rebuild_mc2_ms=%.2f non_member_sequence_ms=%.2f non_member_sequence_dp_ms=%.2f non_member_sequence_ep_ms=%.2f non_member_sequence_mc2_ms=%.2f non_member_sequence_device_new_group_ms=%.2f non_member_sequence_cpu_new_group_ms=%.2f non_member_sequence_destroy_ms=%.2f non_member_sequence_signature_ms=%.2f non_member_sequence_log_ms=%.2f non_member_sequence_groups=%.0f pre_refresh_source_barrier_ms=%.2f refresh_ms=%.2f remote_cache_phase_ms=%.2f remote_cache_start_ms=%.2f remote_cache_barrier1_ms=%.2f remote_payload_warmup_ms=%.2f next_block_prime_ms=%.2f remote_cache_barrier2_ms=%.2f release_staging_ms=%.2f release_staging_clear_ms=%.2f release_staging_gc_ms=%.2f release_staging_empty_cache_ms=%.2f release_staging_sync_ms=%.2f drop_stale_group_cache_ms=%.2f old_floor_group_drop_barrier_ms=%.2f drop_old_floor_group_call_wall_ms=%.2f drop_old_floor_group_cache_ms=%.2f old_floor_pre_memory_log_ms=%.2f old_floor_post_memory_log_ms=%.2f drop_old_floor_group_destroy_ms=%.2f drop_old_floor_group_cleanup_ms=%.2f drop_old_floor_group_cleanup_gc_ms=%.2f drop_old_floor_group_cleanup_empty_cache_ms=%.2f drop_old_floor_group_cleanup_sync_ms=%.2f drop_old_floor_group_cleanup_memory_log_ms=%.2f drop_old_floor_group_memory_log_ms=%.2f drop_old_floor_group_row_log_ms=%.2f drop_old_floor_group_summary_log_ms=%.2f drop_old_floor_group_signature_sweep_ms=%.2f drop_old_floor_group_release_ref_ms=%.2f drop_old_floor_group_dropped_groups=%.0f drop_old_floor_group_deferred_groups=%.0f drop_old_floor_group_dropped_signatures=%.0f post_old_floor_group_drop_barrier_ms=%.2f release_mode5_cache_state_ms=%.2f warmup_ms=%.2f hidden_tail_ms=%.2f total_ms=%.2f",
            self.rank, active_ranks, int(is_active_rank), prefetch_drain_ms,
            prepare_payload_ms, preload_import_ms, post_preload_source_barrier_ms,
            detach_ms, rebuild_ms, rebuild_dp_ms, rebuild_ep_ms, rebuild_mc2_ms,
            non_member_sequence_ms, non_member_sequence_dp_ms,
            non_member_sequence_ep_ms, non_member_sequence_mc2_ms,
            non_member_sequence_device_new_group_ms,
            non_member_sequence_cpu_new_group_ms,
            non_member_sequence_destroy_ms,
            non_member_sequence_signature_ms, non_member_sequence_log_ms,
            non_member_sequence_groups, pre_refresh_source_barrier_ms,
            refresh_ms, remote_cache_phase_ms,
            remote_cache_start_ms,
            remote_cache_barrier1_ms, remote_payload_warmup_ms,
            next_block_prime_ms, remote_cache_barrier2_ms, release_staging_ms,
            release_staging_clear_ms, release_staging_gc_ms,
            release_staging_empty_cache_ms, release_staging_sync_ms,
            drop_stale_group_cache_ms, old_floor_group_drop_barrier_ms,
            drop_old_floor_group_call_wall_ms,
            drop_old_floor_group_cache_ms, old_floor_pre_memory_log_ms,
            old_floor_post_memory_log_ms,
            drop_old_floor_group_destroy_ms,
            drop_old_floor_group_cleanup_ms,
            drop_old_floor_group_cleanup_gc_ms,
            drop_old_floor_group_cleanup_empty_cache_ms,
            drop_old_floor_group_cleanup_sync_ms,
            drop_old_floor_group_cleanup_memory_log_ms,
            drop_old_floor_group_memory_log_ms,
            drop_old_floor_group_row_log_ms,
            drop_old_floor_group_summary_log_ms,
            drop_old_floor_group_signature_sweep_ms,
            drop_old_floor_group_release_ref_ms,
            drop_old_floor_group_dropped_groups,
            drop_old_floor_group_deferred_groups,
            drop_old_floor_group_dropped_signatures,
            post_old_floor_group_drop_barrier_ms,
            release_mode5_cache_state_ms, warmup_ms, hidden_tail_ms, total_ms)

        if is_active_rank:
            logger.info(
            "Elastic parallel shrink done: rank=%s active_ranks=%s dp_size=%s ep_size=%s no_ep_tail=%s rebuild_ms=%.2f refresh_ms=%.2f warmup_ms=%.2f aclgraph_invalidate_ms=%.2f aclgraph_recapture_ms=%.2f total_ms=%.2f",
            self.rank, active_ranks, get_dp_group().world_size,
            vllm_ps.get_ep_group().world_size, len(active_ranks) == 1,
            rebuild_ms, refresh_ms, warmup_ms, aclgraph_invalidate_ms,
            aclgraph_recapture_ms, total_ms)
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
        aclgraph_invalidate_start_t = time.perf_counter()
        self._invalidate_full_moe_elastic_aclgraph("before_full_world_restore")
        aclgraph_invalidate_ms = (
            time.perf_counter() - aclgraph_invalidate_start_t) * 1000.0
        restore_seq = int(getattr(self, "_mode1_fullworld_restore_seq", 0)) + 1
        self._mode1_fullworld_restore_seq = restore_seq
        os.environ["VLLM_ASCEND_MODE1_CURRENT_DISPATCH_PHASE"] = (
            "post_floor_restore")
        os.environ["VLLM_ASCEND_MODE1_CURRENT_RESTORE_SEQ"] = str(restore_seq)
        self._log_mode1_fullworld_restore_diag(
            "restore_entry",
            f"world_size={world_size} backend={backend}",
        )

        if self._has_mode4_remote_npu_lightweight_module():
            self._drain_mode4_double_buffer_prefetch("before_mode4_restore")
            self._reset_mode4_double_buffer_runtime_slots("before_mode4_restore")
            self._stop_mode4_remote_cache_service(world_group)
            torch.distributed.barrier(group=world_group.cpu_group)

        self._reconcile_group_creation_sequence_before_restore(
            world_group, backend)
        self._log_mode1_fullworld_restore_diag(
            "after_reconcile_before_rebuild",
            f"world_size={world_size}",
        )

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
        self._log_mode1_fullworld_restore_diag(
            "after_rebuild_before_refresh",
            f"world_size={world_size} rebuild_ms={rebuild_ms:.2f}",
        )

        refresh_start_t = time.perf_counter()
        self._reset_elastic_moe_comm_setup_cache(
            f"before_restore_refresh world_size={world_size}")
        original_local_ranks = self._original_local_elastic_group_ranks(
            world_size)
        with set_current_vllm_config(self.vllm_config):
            self._refresh_elastic_parallel_state(
                original_local_ranks, world_group)
        refresh_ms = (time.perf_counter() - refresh_start_t) * 1000.0
        self.elastic_parallel_detached = False
        self._elastic_current_active_ranks = list(original_local_ranks)
        self._log_mode1_fullworld_restore_diag(
            "after_refresh_before_stale_drop",
            f"world_size={world_size} refresh_ms={refresh_ms:.2f}",
        )
        stale_drop_stats = self._drop_stale_cached_elastic_parallel_groups(
            tuple(original_local_ranks))
        if (stale_drop_stats.get("dropped_groups", 0.0) > 0.0
                or stale_drop_stats.get("dropped_signatures", 0.0) > 0.0):
            logger.info(
                "Elastic full-world restore stale floor group release summary: "
                "rank=%s dropped_groups=%.0f deferred_groups=%.0f "
                "dropped_signatures=%.0f cleanup_ms=%.2f total_ms=%.2f",
                self.rank,
                stale_drop_stats.get("dropped_groups", 0.0),
                stale_drop_stats.get("deferred_groups", 0.0),
                stale_drop_stats.get("dropped_signatures", 0.0),
                stale_drop_stats.get("cleanup_ms", 0.0),
                stale_drop_stats.get("total_ms", 0.0),
            )
        self._log_mode1_fullworld_restore_diag(
            "after_stale_drop_before_warmup",
            "dropped_groups=%.0f deferred_groups=%.0f dropped_signatures=%.0f "
            "cleanup_ms=%.2f total_ms=%.2f" % (
                stale_drop_stats.get("dropped_groups", 0.0),
                stale_drop_stats.get("deferred_groups", 0.0),
                stale_drop_stats.get("dropped_signatures", 0.0),
                stale_drop_stats.get("cleanup_ms", 0.0),
                stale_drop_stats.get("total_ms", 0.0),
            ),
        )
        stale_drop_barrier_start_t = time.perf_counter()
        torch.distributed.barrier(group=world_group.cpu_group)
        stale_drop_barrier_ms = (
            time.perf_counter() - stale_drop_barrier_start_t) * 1000.0
        logger.info(
            "Elastic full-world restore stale-drop barrier done: rank=%s "
            "world_size=%s elapsed_ms=%.2f",
            self.rank,
            world_size,
            stale_drop_barrier_ms,
        )
        mc2_warmup_start_t = time.perf_counter()
        self._warmup_post_restore_mc2_dispatch_for_custom_mode1_parity(
            len(original_local_ranks))
        mc2_warmup_ms = (time.perf_counter() - mc2_warmup_start_t) * 1000.0
        alltoall_warmup_start_t = time.perf_counter()
        self._warmup_post_restore_alltoall_dispatch_for_custom_mode1_parity(
            len(original_local_ranks))
        alltoall_warmup_ms = (
            time.perf_counter() - alltoall_warmup_start_t) * 1000.0
        loaded_weight_offload_stats = (
            self._offload_mode1_loaded_weights_after_full_restore())
        loaded_weight_offload_ms = loaded_weight_offload_stats.get(
            "total_ms", 0.0)
        self._log_mode1_fullworld_restore_diag(
            "after_all_post_restore_warmups",
            "world_size=%s loaded_weight_offloaded_modules=%.0f "
            "loaded_weight_estimated_npu_bytes=%.0f "
            "loaded_weight_offload_ms=%.2f" % (
                world_size,
                loaded_weight_offload_stats.get("offloaded_modules", 0.0),
                loaded_weight_offload_stats.get("estimated_npu_bytes", 0.0),
                loaded_weight_offload_ms,
            ),
        )
        os.environ["VLLM_ASCEND_MODE1_CURRENT_DISPATCH_PHASE"] = "runtime"

        total_ms = (time.perf_counter() - start_t) * 1000.0
        logger.info(
            "Elastic full-world restore segmented timing: rank=%s "
            "restore_seq=%s world_size=%s rebuild_ms=%.2f refresh_ms=%.2f "
            "stale_drop_ms=%.2f stale_drop_barrier_ms=%.2f mc2_warmup_ms=%.2f "
            "alltoall_warmup_ms=%.2f loaded_weight_offload_ms=%.2f "
            "aclgraph_invalidate_ms=%.2f "
            "total_ms=%.2f",
            self.rank,
            restore_seq,
            world_size,
            rebuild_ms,
            refresh_ms,
            stale_drop_stats.get("total_ms", 0.0),
            stale_drop_barrier_ms,
            mc2_warmup_ms,
            alltoall_warmup_ms,
            loaded_weight_offload_ms,
            aclgraph_invalidate_ms,
            total_ms,
        )
        logger.info(
            "Elastic parallel restore done: rank=%s dp_size=%s ep_size=%s rebuild_ms=%.2f refresh_ms=%.2f total_ms=%.2f",
            self.rank, get_dp_group().world_size, vllm_ps.get_ep_group().world_size,
            rebuild_ms, refresh_ms, total_ms)
        return True

    def set_need_allreduce(self, value: bool):
        self.model_runner.need_allreduce = value
        return True
