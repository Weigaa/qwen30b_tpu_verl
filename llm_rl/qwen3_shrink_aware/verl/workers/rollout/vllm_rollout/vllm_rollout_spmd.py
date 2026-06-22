# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
"""
The vllm_rollout that can be applied in different backend
When working with FSDP:
- Use DTensor weight loader (recommended) or HF weight loader
- Utilize state_dict from the FSDP to synchronize the weights among tp ranks in vLLM
When working with Megatron:
- Use Megatron weight loader
- During training, only the current pp stage holds the parameters
- Before inference, broadcast the parameters of the current pp rank
  to all other pp ranks (all pp ranks holds all the parameters)
- Bind the parameters to the inference engine
- Do inference in tp. pp is treated as additional dp
- After inference, all the parameters that doesn't belong to this pp rank is freed.
"""

import asyncio
import getpass
import gc
import inspect
import json
import logging
import os
import pickle
import socket
import time
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from types import MethodType
from typing import Any, Generator

import numpy as np
import ray
import torch
import torch.distributed
import zmq
import zmq.asyncio
from filelock import FileLock
from omegaconf import ListConfig
from tensordict import TensorDict
from torch.distributed.device_mesh import DeviceMesh
from vllm import LLM, SamplingParams
from vllm.config import CompilationConfig, CompilationLevel, LoRAConfig
from vllm.lora.request import LoRARequest
from vllm_ascend import envs as envs_ascend
from vllm_ascend.ascend_config import get_ascend_config

try:
    from vllm.worker.worker_base import WorkerWrapperBase
except ModuleNotFoundError:
    # https://github.com/vllm-project/vllm/commit/6a113d9aed8221a9c234535958e70e34ab6cac5b
    from vllm.v1.worker.worker_base import WorkerWrapperBase

from verl import DataProto
from verl.third_party.vllm import VLLM_SLEEP_LEVEL
from verl.utils.device import is_npu_available
from verl.utils.distributed import initialize_global_process_group_ray
from verl.utils.profiler import GPUMemoryLogger
from verl.utils.ray_utils import ray_noset_visible_devices
from verl.utils.torch_functional import get_response_mask, pad_2d_list_to_length
from verl.utils.vllm import TensorLoRARequest, VLLMHijack, is_version_ge
from verl.workers.config import HFModelConfig, RolloutConfig
from verl.workers.rollout.base import BaseRollout
#new wj import
import copy
import os
from vllm.utils.moe_stats import moe_stats

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def _env_flag(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).lower() in ("1", "true", "yes", "on")


def _sync_shrink_aware_env_from_meta(meta_info: dict) -> None:
    plan = meta_info.get("shrink_aware_role_plan") if meta_info else None
    if not isinstance(plan, dict):
        return
    runtime = meta_info.get("shrink_aware_runtime", {})
    if not isinstance(runtime, dict):
        runtime = {}
    if runtime.get("dry_run_shrink_aware_schedule"):
        os.environ["VLLM_ASCEND_SHRINK_AWARE_ENABLE"] = "0"
        os.environ["VLLM_ASCEND_SHRINK_AWARE_DRY_RUN"] = "1"
        return
    os.environ["VLLM_ASCEND_SHRINK_AWARE_ENABLE"] = "1"
    os.environ["VLLM_ASCEND_SHRINK_AWARE_MODE"] = str(
        runtime.get("mode", "staged"))
    stages = runtime.get("shrink_stages")
    if isinstance(stages, (list, tuple)) and len(stages) == 2:
        os.environ["VLLM_ASCEND_SHRINK_AWARE_STAGES"] = ",".join(
            str(int(stage)) for stage in stages)
    os.environ["VLLM_ASCEND_SHRINK_AWARE_SURVIVOR_POLICY"] = str(
        runtime.get("survivor_selection_policy", "manual"))
    os.environ["VLLM_ASCEND_SHRINK_AWARE_INTERMEDIATE_RANKS"] = ",".join(
        str(rank) for rank in plan.get("intermediate_survivor_ranks", []))
    os.environ["VLLM_ASCEND_SHRINK_AWARE_FINAL_RANKS"] = ",".join(
        str(rank) for rank in plan.get("final_survivor_ranks", []))
    if "max_rollout_overhead_ratio" in runtime:
        os.environ["VLLM_ASCEND_SHRINK_AWARE_MAX_OVERHEAD_RATIO"] = str(
            runtime["max_rollout_overhead_ratio"])
    if "min_shrink_window_seconds" in runtime:
        os.environ["VLLM_ASCEND_SHRINK_AWARE_MIN_WINDOW_SECONDS"] = str(
            runtime["min_shrink_window_seconds"])
    if "enable_shrink_aware_logging" in runtime:
        os.environ["VLLM_ASCEND_SHRINK_AWARE_LOGGING"] = (
            "1" if runtime["enable_shrink_aware_logging"] else "0")
    if "dry_run_shrink_aware_schedule" in runtime:
        os.environ["VLLM_ASCEND_SHRINK_AWARE_DRY_RUN"] = (
            "1" if runtime["dry_run_shrink_aware_schedule"] else "0")


def _custom_mode1_rollout_reload_diag_enabled() -> bool:
    return _env_flag("VLLM_ASCEND_CUSTOM_MODE1_ROLLOUT_RELOAD_DIAG", "1")


def _log_custom_mode1_rollout_memory(tag: str) -> None:
    if not _custom_mode1_rollout_reload_diag_enabled():
        return
    if int(getattr(envs_ascend, "VLLM_ASCEND_ELASTIC_EXECUTION_MODE", 0)) != 1:
        return
    if not is_npu_available:
        return
    try:
        import torch_npu

        free_bytes, total_bytes = torch.npu.mem_get_info()
        stats = torch_npu.npu.memory_stats()
        torch_current = int(stats.get("allocated_bytes.all.current", 0))
        torch_reserved = int(stats.get("reserved_bytes.all.current", 0))
        total_allocated = int(total_bytes - free_bytes)
        non_torch = max(total_allocated - torch_current, 0)
        logger.warning(
            "Custom mode=1 rollout memory: tag=%s free_bytes=%s total_bytes=%s "
            "torch_current=%s torch_reserved=%s non_torch=%s total_allocated=%s",
            tag,
            free_bytes,
            total_bytes,
            torch_current,
            torch_reserved,
            non_torch,
            total_allocated,
        )
    except Exception:
        logger.exception("Failed to log custom mode=1 rollout memory at %s", tag)


def _tensor_nbytes(tensor: Any) -> int:
    if not isinstance(tensor, torch.Tensor):
        return 0
    try:
        return int(tensor.numel()) * int(tensor.element_size())
    except Exception:
        return 0


def _tensor_storage_ptr(tensor: Any) -> int | None:
    if not isinstance(tensor, torch.Tensor):
        return None
    try:
        return int(tensor.untyped_storage().data_ptr())
    except Exception:
        try:
            return int(tensor.data_ptr())
        except Exception:
            return None


def _get_npu_buffer_format(tensor: Any):
    if not isinstance(tensor, torch.Tensor):
        return None
    if tensor.device.type != "npu":
        return None
    try:
        import torch_npu  # type: ignore
        return torch_npu.get_npu_format(tensor)
    except Exception:
        return None


def _len_if_present(obj: Any) -> int:
    if obj is None:
        return 0
    try:
        return int(len(obj))
    except Exception:
        return 0


def _collect_tensor_entries(obj: Any,
                            prefix: str,
                            depth: int = 0,
                            max_depth: int = 1,
                            visited: set[int] | None = None):
    if visited is None:
        visited = set()
    obj_id = id(obj)
    if obj_id in visited:
        return
    visited.add(obj_id)

    if isinstance(obj, torch.Tensor):
        yield prefix, obj
        return
    if depth >= max_depth:
        return
    if isinstance(obj, (list, tuple)):
        for idx, item in enumerate(obj):
            yield from _collect_tensor_entries(item, f"{prefix}[{idx}]",
                                               depth + 1, max_depth, visited)
        return
    if isinstance(obj, dict):
        for key, item in obj.items():
            yield from _collect_tensor_entries(item, f"{prefix}[{key!r}]",
                                               depth + 1, max_depth, visited)


def _custom_mode1_global_tensor_scan_enabled() -> bool:
    return _env_flag("VLLM_ASCEND_CUSTOM_MODE1_GLOBAL_TENSOR_SCAN", "0")


def _log_custom_mode1_rollout_state(owner: Any, tag: str) -> None:
    if not _custom_mode1_rollout_reload_diag_enabled():
        return
    if int(getattr(envs_ascend, "VLLM_ASCEND_ELASTIC_EXECUTION_MODE", 0)) != 1:
        return

    model = getattr(owner, "model", None)
    if model is None:
        return

    gpu_buffers = getattr(owner, "gpu_buffers", None)
    if not isinstance(gpu_buffers, dict):
        gpu_buffers = {}

    total_params = 0
    param_npu_count = 0
    param_npu_bytes = 0
    param_cpu_count = 0
    param_storage_ptrs: set[int] = set()
    named_buffer_npu_count = 0
    named_buffer_npu_bytes = 0
    named_buffer_storage_ptrs: set[int] = set()
    gpu_buffer_count = 0
    gpu_buffer_bytes = 0
    alias_match_count = 0
    stale_gpu_buffer_count = 0
    stale_gpu_buffer_bytes = 0
    stale_gpu_buffer_samples: list[str] = []
    gpu_buffer_storage_ptrs: set[int] = set()

    for name, param in model.named_parameters():
        total_params += 1
        data = param.data
        ptr = _tensor_storage_ptr(data)
        if ptr is not None:
            param_storage_ptrs.add(ptr)
        if data.device.type == "npu":
            param_npu_count += 1
            param_npu_bytes += _tensor_nbytes(data)
        elif data.device.type == "cpu":
            param_cpu_count += 1

        gpu_buffer = gpu_buffers.get(name)
        if isinstance(gpu_buffer, torch.Tensor):
            gpu_buffer_count += 1
            gpu_buffer_bytes += _tensor_nbytes(gpu_buffer)
            gpu_ptr = _tensor_storage_ptr(gpu_buffer)
            if gpu_ptr is not None:
                gpu_buffer_storage_ptrs.add(gpu_ptr)
            same_ptr = (_tensor_storage_ptr(gpu_buffer)
                        == _tensor_storage_ptr(data))
            if same_ptr:
                alias_match_count += 1
            else:
                stale_gpu_buffer_count += 1
                stale_gpu_buffer_bytes += _tensor_nbytes(gpu_buffer)
                if len(stale_gpu_buffer_samples) < 6:
                    stale_gpu_buffer_samples.append(
                        f"{name}:gpu_dev={gpu_buffer.device.type}"
                        f"/gpu_fmt={_get_npu_buffer_format(gpu_buffer)}"
                        f"/param_dev={data.device.type}"
                        f"/param_fmt={_get_npu_buffer_format(data)}")

    for _name, buffer in model.named_buffers():
        if not isinstance(buffer, torch.Tensor):
            continue
        ptr = _tensor_storage_ptr(buffer)
        if ptr is not None:
            named_buffer_storage_ptrs.add(ptr)
        if buffer.device.type == "npu":
            named_buffer_npu_count += 1
            named_buffer_npu_bytes += _tensor_nbytes(buffer)

    extra_tensor_attr_npu_count = 0
    extra_tensor_attr_npu_bytes = 0
    extra_tensor_attr_samples: list[str] = []
    extra_tensor_attr_storage_ptrs: set[int] = set()

    for module_name, module in model.named_modules():
        safe_module_name = module_name or "<root>"
        for attr_name, attr_value in vars(module).items():
            if attr_name in ("_parameters", "_buffers", "_modules"):
                continue
            attr_prefix = f"{safe_module_name}.{attr_name}"
            for tensor_path, tensor in _collect_tensor_entries(
                    attr_value, attr_prefix, max_depth=1):
                if tensor.device.type != "npu" or tensor.numel() <= 0:
                    continue
                ptr = _tensor_storage_ptr(tensor)
                if ptr is None:
                    continue
                if (ptr in param_storage_ptrs or ptr in named_buffer_storage_ptrs
                        or ptr in gpu_buffer_storage_ptrs
                        or ptr in extra_tensor_attr_storage_ptrs):
                    continue
                extra_tensor_attr_storage_ptrs.add(ptr)
                extra_tensor_attr_npu_count += 1
                extra_tensor_attr_npu_bytes += _tensor_nbytes(tensor)
                if len(extra_tensor_attr_samples) < 10:
                    extra_tensor_attr_samples.append(
                        f"{tensor_path}:shape={tuple(tensor.shape)}"
                        f":bytes={_tensor_nbytes(tensor)}"
                        f":fmt={_get_npu_buffer_format(tensor)}")

    kv_cache_module_refs = 0
    kv_cache_npu_tensors = 0
    attn_cache_layers = 0
    mla_cache_layers = 0
    fused_runtime_weight_layers = 0
    fused_runtime_buffer_layers = 0
    fused_cpu_shadow_layers = 0
    fused_saved_prefix_layers = 0
    fused_runtime_samples: list[str] = []

    for module in model.modules():
        kv_cache = getattr(module, "kv_cache", None)
        if kv_cache is not None:
            kv_cache_module_refs += 1
            if isinstance(kv_cache, (list, tuple)):
                for item in kv_cache:
                    if (isinstance(item, torch.Tensor)
                            and item.device.type == "npu"
                            and item.numel() > 0):
                        kv_cache_npu_tensors += 1
            elif (isinstance(kv_cache, torch.Tensor)
                  and kv_cache.device.type == "npu" and kv_cache.numel() > 0):
                kv_cache_npu_tensors += 1

        runtime_w13 = getattr(module, "runtime_w13_weight", None)
        runtime_w2 = getattr(module, "runtime_w2_weight", None)
        runtime_buffer_w13 = getattr(module, "runtime_w13_buffer", None)
        runtime_buffer_w2 = getattr(module, "runtime_w2_buffer", None)
        cpu_w13 = getattr(module, "lossless_cpu_w13_weight", None)
        cpu_w2 = getattr(module, "lossless_cpu_w2_weight", None)
        saved_prefix_w13 = getattr(module, "lossless_saved_primary_prefix_w13",
                                   None)
        saved_prefix_w2 = getattr(module, "lossless_saved_primary_prefix_w2",
                                  None)

        has_runtime_weight = (runtime_w13 is not None or runtime_w2 is not None)
        has_runtime_buffer = (runtime_buffer_w13 is not None
                              or runtime_buffer_w2 is not None)
        has_cpu_shadow = (cpu_w13 is not None or cpu_w2 is not None)
        has_saved_prefix = (saved_prefix_w13 is not None
                            or saved_prefix_w2 is not None)
        if has_runtime_weight:
            fused_runtime_weight_layers += 1
        if has_runtime_buffer:
            fused_runtime_buffer_layers += 1
        if has_cpu_shadow:
            fused_cpu_shadow_layers += 1
        if has_saved_prefix:
            fused_saved_prefix_layers += 1
        if (has_runtime_weight or has_runtime_buffer) and len(
                fused_runtime_samples) < 6:
            fused_runtime_samples.append(
                f"layer={getattr(module, 'layer_idx', -1)}"
                f":runtime_weight={has_runtime_weight}"
                f":runtime_buffer={has_runtime_buffer}"
                f":cpu_shadow={has_cpu_shadow}"
                f":saved_prefix={has_saved_prefix}")

        self_attn = getattr(module, "self_attn", None)
        if self_attn is None:
            continue
        attn_impl = getattr(getattr(self_attn, "attn", None), "impl", None)
        if (attn_impl is not None and
                (getattr(attn_impl, "key_cache", None) is not None
                 or getattr(attn_impl, "value_cache", None) is not None)):
            attn_cache_layers += 1
        mla_impl = getattr(getattr(self_attn, "mla_attn", None), "impl", None)
        if (mla_impl is not None and
                (getattr(mla_impl, "key_cache", None) is not None
                 or getattr(mla_impl, "value_cache", None) is not None
                 or getattr(mla_impl, "w_kc", None) is not None
                 or getattr(mla_impl, "w_vc", None) is not None
                 or getattr(mla_impl, "W_UV", None) is not None
                 or getattr(mla_impl, "W_UK_T", None) is not None)):
            mla_cache_layers += 1

    mode3_slots = getattr(model, "_mode3_cpu_npu_double_buffer_slots", None)
    mode4_slots = getattr(model, "_mode4_remote_npu_double_buffer_slots", None)
    global_unowned_npu_count = 0
    global_unowned_npu_bytes = 0
    global_unowned_npu_samples: list[str] = []
    if _custom_mode1_global_tensor_scan_enabled():
        known_ptrs = (param_storage_ptrs | named_buffer_storage_ptrs
                      | gpu_buffer_storage_ptrs
                      | extra_tensor_attr_storage_ptrs)
        global_seen_ptrs: set[int] = set()
        try:
            for obj in gc.get_objects():
                if not isinstance(obj, torch.Tensor):
                    continue
                if obj.device.type != "npu" or obj.numel() <= 0:
                    continue
                ptr = _tensor_storage_ptr(obj)
                if ptr is None or ptr in known_ptrs or ptr in global_seen_ptrs:
                    continue
                global_seen_ptrs.add(ptr)
                global_unowned_npu_count += 1
                global_unowned_npu_bytes += _tensor_nbytes(obj)
                if len(global_unowned_npu_samples) < 10:
                    global_unowned_npu_samples.append(
                        f"type={type(obj).__name__}:shape={tuple(obj.shape)}"
                        f":bytes={_tensor_nbytes(obj)}"
                        f":fmt={_get_npu_buffer_format(obj)}")
        except Exception:
            logger.exception(
                "Failed global tensor scan for custom mode=1 rollout state: tag=%s",
                tag)

    logger.warning(
        "Custom mode=1 rollout state: tag=%s total_params=%s "
        "param_npu_count=%s param_npu_bytes=%s param_cpu_count=%s "
        "named_buffer_npu_count=%s named_buffer_npu_bytes=%s "
        "gpu_buffer_count=%s gpu_buffer_bytes=%s gpu_buffer_alias_match=%s "
        "stale_gpu_buffer_count=%s stale_gpu_buffer_bytes=%s "
        "extra_tensor_attr_npu_count=%s extra_tensor_attr_npu_bytes=%s "
        "global_unowned_npu_count=%s global_unowned_npu_bytes=%s "
        "kv_cache_module_refs=%s kv_cache_npu_tensors=%s "
        "attn_cache_layers=%s mla_cache_layers=%s "
        "fused_runtime_weight_layers=%s fused_runtime_buffer_layers=%s "
        "fused_cpu_shadow_layers=%s fused_saved_prefix_layers=%s "
        "mode3_slot_containers=%s mode4_slot_containers=%s",
        tag,
        total_params,
        param_npu_count,
        param_npu_bytes,
        param_cpu_count,
        named_buffer_npu_count,
        named_buffer_npu_bytes,
        gpu_buffer_count,
        gpu_buffer_bytes,
        alias_match_count,
        stale_gpu_buffer_count,
        stale_gpu_buffer_bytes,
        extra_tensor_attr_npu_count,
        extra_tensor_attr_npu_bytes,
        global_unowned_npu_count,
        global_unowned_npu_bytes,
        kv_cache_module_refs,
        kv_cache_npu_tensors,
        attn_cache_layers,
        mla_cache_layers,
        fused_runtime_weight_layers,
        fused_runtime_buffer_layers,
        fused_cpu_shadow_layers,
        fused_saved_prefix_layers,
        _len_if_present(mode3_slots),
        _len_if_present(mode4_slots),
    )
    if stale_gpu_buffer_samples:
        logger.warning(
            "Custom mode=1 rollout stale gpu buffer samples: tag=%s samples=%s",
            tag,
            stale_gpu_buffer_samples,
        )
    if fused_runtime_samples:
        logger.warning(
            "Custom mode=1 rollout fused runtime samples: tag=%s samples=%s",
            tag,
            fused_runtime_samples,
        )
    if extra_tensor_attr_samples:
        logger.warning(
            "Custom mode=1 rollout extra tensor attr samples: tag=%s samples=%s",
            tag,
            extra_tensor_attr_samples,
        )
    if global_unowned_npu_samples:
        logger.warning(
            "Custom mode=1 rollout global unowned NPU tensor samples: tag=%s samples=%s",
            tag,
            global_unowned_npu_samples,
        )


def _materialize_rollout_weight_staging(
        weights: list[tuple[str, torch.Tensor]]
) -> list[tuple[str, torch.Tensor]]:
    """Create rollout-owned tensors for weight reload.

    Mode=1 repeatedly reloads rollout weights while elastic groups and runtime
    views are being rebuilt. Cloning cuts loader-side aliases to caller-owned
    tensors so the rollout can release temporary NPU allocations promptly.
    """
    staged: list[tuple[str, torch.Tensor]] = []
    for name, weight in weights:
        staged.append((name, weight.detach().clone()))
    return staged


def _load_model_num_experts(model_path: str) -> int:
    config_path = Path(model_path) / "config.json"
    if not config_path.exists():
        return 0
    try:
        config = json.loads(config_path.read_text())
    except Exception:
        return 0
    num_experts = config.get("num_experts")
    return int(num_experts) if isinstance(num_experts, int) and num_experts > 0 else 0

# TODO
# 1. support pp in vllm
# 2. passing tokenizer is not necessary? no encoding/decoding is happending here
# 3. simplify init logics


# NOTE(sgm): add for verl. We can optimize it by making the dataloader yield List[int] without padding.
def _pre_process_inputs(pad_token_id, prompt_token_ids: torch.Tensor) -> list[int]:
    # remove the left padding in the prompt token_id
    # pad_token_id = self.llm_engine.tokenizer.pad_token_id if self.llm_engine.tokenizer.pad_token_id
    # is not None else self.llm_engine.tokenizer.eos_token_id
    non_pad_index = torch.nonzero(prompt_token_ids != pad_token_id, as_tuple=False)[0][0]
    token_ids = prompt_token_ids[non_pad_index:].tolist()
    return token_ids


if is_version_ge(pkg="vllm", minver="0.7.3"):
    VLLMHijack.hijack()


class vLLMRollout(BaseRollout):
    def __init__(
        self,
        config: RolloutConfig,
        model_config: HFModelConfig,
        device_mesh: DeviceMesh,
    ):
        self.config = config
        self.model_config = model_config
        self.device_mesh = device_mesh

        from vllm_ascend.patch import platform
        from vllm_ascend.patch import worker

        if config.layered_summon:
            self.sleep_level = 1
        else:
            self.sleep_level = VLLM_SLEEP_LEVEL

        model_path = model_config.local_path
        tokenizer = model_config.tokenizer
        model_hf_config = model_config.hf_config
        trust_remote_code = model_config.trust_remote_code
        self.lora_kwargs = (
            {"enable_lora": True, "max_loras": 1, "max_lora_rank": model_config.lora_rank}
            if model_config.lora_rank > 0
            else {}
        )

        tensor_parallel_size = self.config.get("tensor_model_parallel_size", 1)
        assert tensor_parallel_size <= torch.distributed.get_world_size(), (
            "tensor parallel size should be less than or equal to the world size"
        )
        max_num_batched_tokens = self.config.get("max_num_batched_tokens", 8192)

        #If VLLM_DP_SIZE is configured, the DP communication domain needs to be explicitly initialized.
        if int(os.environ.get("VLLM_DP_SIZE", "1")) > 1:
            from r1_ascend.vllm_parallel_state import init_parallel_state
            init_parallel_state(tensor_parallel_size)

        rope_scaling_config = getattr(model_hf_config, "rope_scaling", None)
        if not rope_scaling_config:
            max_position_embeddings = None
            if hasattr(model_hf_config, "max_position_embeddings"):
                max_position_embeddings = model_hf_config.max_position_embeddings
            elif hasattr(model_hf_config, "llm_config") and hasattr(
                model_hf_config.llm_config, "max_position_embeddings"
            ):
                max_position_embeddings = model_hf_config.llm_config.max_position_embeddings
            elif hasattr(model_hf_config, "text_config") and hasattr(
                model_hf_config.text_config, "max_position_embeddings"
            ):
                max_position_embeddings = model_hf_config.text_config.max_position_embeddings
            if max_position_embeddings is None:
                raise ValueError("max_position_embeddings not found in model_hf_config")
            assert max_position_embeddings >= config.prompt_length + config.response_length, (
                "model context length should be greater than total sequence length"
            )
        else:
            # handle type where there's a length extend factor
            # see https://qwen.readthedocs.io/en/latest/deployment/vllm.html#extended-context-support
            # for using yarn as an example
            rope_scaling_factor = rope_scaling_config.get("factor", 1.0)

            assert (
                model_hf_config.max_position_embeddings * rope_scaling_factor
                >= config.prompt_length + config.response_length
            ), (
                "model context length should be greater than total sequence length, "
                + f"got rope_scaling_factor={rope_scaling_factor} and "
                + f"max_position_embeddings={model_hf_config.max_position_embeddings}"
            )

        max_model_len = int(config.max_model_len or config.prompt_length + config.response_length)

        load_format = "dummy" if config.load_format.startswith("dummy") else config.load_format

        # copy it to avoid secretly modifying the engine config
        engine_kwargs = config.get("engine_kwargs", {}).get("vllm", {}) or {}

        # For each vLLM engine parameter,
        # - `None` means not setting it, so we pop it, and leave it to vLLM default value
        #    (which can vary across different vLLM versions);
        # - Otherwise it's the desired value we want to explicitly set.
        engine_kwargs = {key: val for key, val in engine_kwargs.items() if val is not None}
        if config.get("limit_images", None):  # support for multi-image data
            engine_kwargs["limit_mm_per_prompt"] = {"image": config.get("limit_images")}

        compilation_config = {}

        cudagraph_capture_sizes = config.get("cudagraph_capture_sizes")
        # enforce_eager must be False to use cudagraph
        if not config.enforce_eager and cudagraph_capture_sizes:
            if isinstance(cudagraph_capture_sizes, ListConfig):
                compilation_config["compilation_config"] = CompilationConfig(
                    level=CompilationLevel.PIECEWISE, cudagraph_capture_sizes=cudagraph_capture_sizes
                )
            else:
                logger.warning(f"cudagraph_capture_sizes must be a list, but got {cudagraph_capture_sizes}")

        self.dynamic_eplb = int(os.environ.get("VLLM_ENABLE_EPLB", "0")) == 1
        elastic_execution_mode = envs_ascend.VLLM_ASCEND_ELASTIC_EXECUTION_MODE
        elastic_moe_mode = envs_ascend.VLLM_ASCEND_ELASTIC_MOE_MODE
        init_redundancy_expert = envs_ascend.VLLM_ASCEND_INIT_REDUNDANCY_EXPERT
        if elastic_execution_mode in (1, 2):
            initial_ep_size = int(os.environ.get("VLLM_DP_SIZE", "1"))
            model_num_experts = _load_model_num_experts(model_path)
            init_redundancy_expert = (
                envs_ascend.compute_elastic_init_redundancy_expert(
                    model_num_experts,
                    initial_ep_size,
                    init_redundancy_expert,
                ))
            logger.info(
                "Elastic execution mode resolved: mode=%s elastic_moe_mode=%s initial_ep_size=%s floor=%s hybrid_resident_slots=%s num_experts=%s init_redundancy_expert=%s",
                elastic_execution_mode,
                elastic_moe_mode,
                initial_ep_size,
                envs_ascend.VLLM_ASCEND_ELASTIC_MIN_COMPUTE_GROUP_SIZE,
                envs_ascend.VLLM_ASCEND_ELASTIC_HYBRID_RESIDENT_EXPERT_SLOTS,
                model_num_experts,
                init_redundancy_expert,
            )
        self.elastic_execution_mode = elastic_execution_mode
        self.elastic_init_redundancy_expert = init_redundancy_expert
        self.inference_engine = LLM(
            model=model_path,
            enable_sleep_mode=False,
            tensor_parallel_size=tensor_parallel_size,
            distributed_executor_backend="external_launcher",
            dtype=config.dtype,
            enforce_eager=config.enforce_eager,
            gpu_memory_utilization=config.gpu_memory_utilization,
            disable_custom_all_reduce=True,
            enable_expert_parallel=int(os.environ.get("VLLM_ENABLE_EXPERT_PARALLEL", "0")),
            skip_tokenizer_init=False,
            max_model_len=max_model_len,
            max_num_seqs=config.max_num_seqs,
            load_format=load_format,
            disable_log_stats=config.disable_log_stats,
            max_num_batched_tokens=max_num_batched_tokens,
            enable_chunked_prefill=config.enable_chunked_prefill,
            enable_prefix_caching=False,
            trust_remote_code=trust_remote_code,
            seed=config.get("seed", 0),
            additional_config={
                "torchair_graph_config": {
                    "enabled": int(os.environ.get("VLLM_ENABLE_GRAPH_MODE", "0")),
                    "use_cached_graph": False,
                    "graph_batch_sizes_init": True,
                    "enable_multistream_mla": False,
                    "enable_zero_tp_to_ep": True,
                    "enable_view_optimize": False,
                    "enable_kv_nz": False,
                    "enable_frozen_parameter": False,
                } if not config.enforce_eager else {"enabled": False},
                "ascend_scheduler_config": {
                    "enabled": True,
                    "enable_chunked_prefill": False,
                },
                "refresh": True,
                "elastic_moe_mode": elastic_moe_mode,
                "init_redundancy_expert": init_redundancy_expert,
                "dynamic_eplb": self.dynamic_eplb,
                "num_iterations_eplb_update": 400,  # gather stable workload over 400 iterations
                "gate_eplb": True,
                "num_wait_worker_iterations": 30,  # wait for 30 iterations to complete the EPLB calculation
            },
            **compilation_config,
            **self.lora_kwargs,
            **engine_kwargs,
        )

        # Offload vllm model to reduce peak memory usage
        self.model_runner = self.inference_engine.llm_engine.model_executor.driver_worker.worker.model_runner
        self.model = self.model_runner.get_model()
        self.kv_cache_configs = None
        self.cpu_model = {}
        self.gpu_buffer_formats = {}
        self.gpu_buffers = None
        for name, params in self.model.named_parameters():
            self.cpu_model[name] = torch.empty_like(params, device="cpu")
            self.gpu_buffer_formats[name] = _get_npu_buffer_format(params)
        self.free_cache_engine()
        self.offload_model_weights()

        kwargs = dict(
            n=1,
            logprobs=0,  # can be set to 0 and let actor to recompute
            max_tokens=config.response_length,
            repetition_penalty=config.get("repetition_penalty", 1.0),
        )

        # Patch: unset logprobs if speculative_config is enabled.
        if "speculative_config" in engine_kwargs:
            logger.warning("The 'logprobs' parameter is incompatible with Speculative Decoding and has been disabled.")
            del kwargs["logprobs"]

        kwargs["detokenize"] = False

        # supporting adding any sampling params from the config file
        for k in config.keys():
            if hasattr(SamplingParams(), str(k)) and k != "seed":
                kwargs[k] = config.get(k)
        kwargs["n"] = 1  # already repeat in ray_trainer
        print(f"kwargs: {kwargs}")
        self.sampling_params = SamplingParams(**kwargs)

        self.pad_token_id = tokenizer.pad_token_id

        self.eplb_end()

    def init_cache_engine(self):
        if os.environ['VLLM_USE_V1'] == '1':
            worker = self.inference_engine.llm_engine.model_executor.driver_worker.worker
            if not worker.model_runner.kv_caches:
                # v1 Use Explicit Initialization Method
                self.inference_engine.llm_engine.engine_core.engine_core.model_executor.initialize_from_config(
                    self.inference_engine.llm_engine.engine_core.engine_core.kv_cache_configs)
                self.inference_engine.llm_engine.reset_prefix_cache()
        else:
            if self.inference_engine.llm_engine.model_executor.driver_worker.worker.cache_engine is None:
                self.inference_engine.llm_engine.model_executor.driver_worker.worker._init_cache_engine()

    def onload_model_weights(self):
        """
        Advantages over model.cuda():
        1) Avoids CPU to GPU data transfer entirely, leveraging pre-allocated GPU buffers
        instead of copying data from CPU tensors.
        2) Eliminates the recursive traversal of submodules inherent in .cuda(),
        which can be particularly slow for deeply nested model architectures.
        """
        _log_custom_mode1_rollout_memory("before_onload_model_weights")
        self.gpu_buffers = {}
        formatted_buffers = 0
        for name, param in self.model.named_parameters():
            target_format = self.gpu_buffer_formats.get(name)
            gpu_buffer = torch.empty_like(param, device='cuda')
            if target_format is not None and str(target_format) not in (
                    "ND", "0"):
                try:
                    import torch_npu  # type: ignore
                    gpu_buffer = torch_npu.npu_format_cast(
                        gpu_buffer, target_format)
                    formatted_buffers += 1
                except Exception:
                    logger.exception(
                        "Failed to allocate formatted rollout GPU buffer for %s format=%s",
                        name,
                        target_format,
                    )
            self.gpu_buffers[name] = gpu_buffer
        if formatted_buffers > 0:
            logger.info(
                "Rollout onload allocated formatted GPU buffers: formatted=%s total=%s",
                formatted_buffers,
                len(self.gpu_buffers),
            )
        for name, param in self.model.named_parameters():
            param.data = self.gpu_buffers[name]
        _log_custom_mode1_rollout_state(self, "after_onload_model_weights")
        _log_custom_mode1_rollout_memory("after_onload_model_weights")

    def _refresh_gpu_buffer_format_metadata(self) -> None:
        if not hasattr(self, "gpu_buffer_formats"):
            self.gpu_buffer_formats = {}
        for name, param in self.model.named_parameters():
            fmt = _get_npu_buffer_format(param.data)
            if fmt is not None:
                self.gpu_buffer_formats[name] = fmt

    def _refresh_gpu_buffer_aliases(self) -> None:
        if self.gpu_buffers is None:
            return
        refreshed_gpu_buffers = {}
        for name, param in self.model.named_parameters():
            refreshed_gpu_buffers[name] = param.data
        self.gpu_buffers = refreshed_gpu_buffers
        self._refresh_gpu_buffer_format_metadata()

    def offload_model_weights(self):
        _log_custom_mode1_rollout_memory("before_offload_model_weights")
        _log_custom_mode1_rollout_state(self, "before_offload_model_weights")
        refresh_gpu_buffer_format_metadata = getattr(
            self, "_refresh_gpu_buffer_format_metadata", None)
        if callable(refresh_gpu_buffer_format_metadata):
            refresh_gpu_buffer_format_metadata()
        invalidated_layers = 0
        invalidated_runtime_layers = 0
        for module in self.model.modules():
            quant_method = getattr(module, "quant_method", None)
            invalidate_runtime = getattr(
                quant_method,
                "invalidate_lossless_runtime_state_for_reload",
                None,
            )
            if callable(invalidate_runtime):
                invalidated_layers += 1
                try:
                    invalidated = invalidate_runtime(
                        layer=module,
                        reason="vllm_rollout_offload_model_weights",
                    )
                    if invalidated:
                        invalidated_runtime_layers += 1
                except Exception:
                    logger.exception(
                        "Failed to invalidate lossless runtime state during rollout offload"
                    )
        for name, params in self.model.named_parameters():
            params.data = self.cpu_model[name]

        if hasattr(self.model.model.layers[0].self_attn, "mla_attn"):
            for i in range(self.model.model.start_layer, self.model.model.end_layer):
                mla = self.model.model.layers[i].self_attn.mla_attn.impl
                if hasattr(mla, "w_kc"):
                    mla.w_kc = None
                    mla.w_vc = None
                if hasattr(mla, "W_UV"):
                    mla.W_UV = None
                    mla.W_UK_T = None
        self.gpu_buffers = None
        gc.collect()
        torch.npu.empty_cache()
        if invalidated_layers > 0:
            logger.info(
                "Rollout offload invalidated lossless runtime state: modules=%s runtime_modules=%s",
                invalidated_layers,
                invalidated_runtime_layers,
            )
        _log_custom_mode1_rollout_state(self, "after_offload_model_weights")
        _log_custom_mode1_rollout_memory("after_offload_model_weights")

    def free_cache_engine(self):
        _log_custom_mode1_rollout_state(self, "before_free_cache_engine")
        if os.environ['VLLM_USE_V1'] == '1':
            worker = self.inference_engine.llm_engine.model_executor.driver_worker.worker
            ctx = worker.model_runner.vllm_config.compilation_config.static_forward_context
        else:
            ctx = self.inference_engine.llm_engine.model_executor.driver_worker.worker.compilation_config.static_forward_context
        from vllm.attention import AttentionType

        layer_need_kv_cache = []
        for layer_name in ctx:
            if hasattr(ctx[layer_name], 'attn_type') and ctx[layer_name].attn_type in (AttentionType.DECODER, AttentionType.ENCODER_DECODER):
                layer_need_kv_cache.append(layer_name)

        pipeline_parallel_size = self.inference_engine.llm_engine.vllm_config.parallel_config.pipeline_parallel_size
        for layer_name in layer_need_kv_cache:
            kv_cache = []
            for _ in range(pipeline_parallel_size):
                kv_cache.append(torch.tensor([]))
            ctx[layer_name].kv_cache = kv_cache
            # The layer modules themselves also keep strong `kv_cache`
            # references. If we only replace the static forward-context entry,
            # step-2 resume can still carry the previous large cache tensors
            # through the model objects and leave attention/backend memory far
            # above the native baseline.
            try:
                if hasattr(self.model, "named_modules"):
                    module = dict(self.model.named_modules()).get(layer_name)
                    if module is not None and hasattr(module, "kv_cache"):
                        module.kv_cache = kv_cache
            except Exception:
                logger.exception(
                    "Failed to clear module kv_cache reference for layer %s",
                    layer_name)

        if os.environ['VLLM_USE_V1'] == '1':
            worker = self.inference_engine.llm_engine.model_executor.driver_worker.worker

            # Clearing the cache engine
            worker.model_runner.kv_caches = []
        else:
            self.inference_engine.llm_engine.model_executor.driver_worker.worker.cache_engine = None
            self.inference_engine.llm_engine.model_executor.driver_worker.worker.gpu_cache = None

        if hasattr(self.model.model.layers[0].self_attn, "attn"):
            for i in range(self.model.model.start_layer, self.model.model.end_layer):
                attn_impl = self.model.model.layers[i].self_attn.attn.impl
                if hasattr(attn_impl, "key_cache"):
                    attn_impl.key_cache = None
                    attn_impl.value_cache = None
        if hasattr(self.model.model.layers[0].self_attn, "mla_attn"):
            for i in range(self.model.model.start_layer, self.model.model.end_layer):
                mla_attn = self.model.model.layers[i].self_attn.mla_attn
                if mla_attn is None or not hasattr(mla_attn, "impl"):
                    continue
                mla_impl = mla_attn.impl
                if hasattr(mla_impl, "key_cache"):
                    mla_impl.key_cache = None
                    mla_impl.value_cache = None

        gc.collect()
        torch.npu.empty_cache()
        _log_custom_mode1_rollout_state(self, "after_free_cache_engine")
        _log_custom_mode1_rollout_memory("after_free_cache_engine")

    def eplb_start(self):
        # Restart the EPLB process before switching from training to inference.
        if self.dynamic_eplb:
            model = self.model_runner.get_model()
            model.clear_all_moe_loads()
            model.reset_all_expert_map_and_log2phy()
            self.model_runner.eplb_adaptor.__init__(model)
            self.model_runner.eplb_loader.__init__()
            self.model_runner.eplb_process.__init__(shared_dict=self.model_runner.shared_dict, policy_type=1, enable_d2d=True)
            ascend_config = get_ascend_config()
            self.model_runner.process = self.model_runner.eplb_process._launch_process()
            self.model_runner.eplb_updator.__init__(ascend_config, self.model_runner.eplb_loader, self.model_runner.eplb_process,
                                                    self.model_runner.process)
            self.model_runner.eplb_updator.get_init_expert_map()
            self.model_runner.eplb_updator.compute_and_set_moe_load()

    def eplb_end(self):
        # Shut down the EPLB service and release memory when switching from inference to training.
        if self.dynamic_eplb:
            self.model_runner.eplb_updator.adaptor.release_memory()
            self.model_runner.eplb_updator.shutdown()

    @contextmanager
    def update_sampling_params(self, **kwargs):
        # update sampling params
        old_sampling_params_args = {}
        if kwargs:
            for key, value in kwargs.items():
                if hasattr(self.sampling_params, key):
                    old_value = getattr(self.sampling_params, key)
                    old_sampling_params_args[key] = old_value
                    setattr(self.sampling_params, key, value)
        yield
        # roll back to previous sampling params
        # if len(old_sampling_params_args):
        for key, value in old_sampling_params_args.items():
            setattr(self.sampling_params, key, value)

    @GPUMemoryLogger(role="vllm rollout spmd", logger=logger)
    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto, **kwargs) -> DataProto:
        """Generate sequences for a batch of prompts.

        Args:
            batch (DataProto): Input batch.

        Returns:
            DataProto: Output batch.
            - prompts: [bsz, prompt_length], prompt token ids from dataset.
            - responses: [bsz, response_length], output token ids include response tokens
              from LLM generation and observation tokens from tool_calls.
            - response_mask: [bsz, response_length], 1 for LLM generated tokens, 0 for observation/padding tokens.
            - input_ids: [bsz, prompt_length + response_length], whole sequence token ids, including prompt tokens
              and response tokens.
            - attention_mask: [bsz, prompt_length + response_length], 0 for padding tokens, 1 for other tokens.
            - position_ids: [bsz, prompt_length + response_length], incremental position ids.

            For multi-turn conversations:
            responses:     |<- LLM generation ->|<- tool_calls ->|<- LLM generation ->|<- padding ->|
            response_mask: | 1, 1, 1, ..., 1, 1 | 0, 0, .., 0, 0 | 1, 1, 1, ..., 1, 1 | 0, 0, ..., 0|
        """
        _sync_shrink_aware_env_from_meta(prompts.meta_info)
        idx = prompts.batch["input_ids"]  # (bs, prompt_length)
        # left-padded attention_mask
        attention_mask = prompts.batch["attention_mask"]
        position_ids = prompts.batch["position_ids"]

        # used to construct attention_mask
        eos_token_id = prompts.meta_info["eos_token_id"]

        batch_size = idx.size(0)

        non_tensor_batch = prompts.non_tensor_batch
        if "raw_prompt_ids" not in non_tensor_batch:
            non_tensor_batch["raw_prompt_ids"] = np.array(
                [_pre_process_inputs(self.pad_token_id, idx[i]) for i in range(batch_size)], dtype=object
            )

        if batch_size != len(non_tensor_batch["raw_prompt_ids"]):
            raise RuntimeError("vllm sharding manager is not work properly.")

        moe_stats.set_generation_context(
            epoch=prompts.meta_info.get("epoch", -1),
            global_step=prompts.meta_info.get("global_steps", -1),
        )

        if "multi_modal_data" in non_tensor_batch:
            vllm_inputs = []
            for raw_prompt_ids, multi_modal_data in zip(
                non_tensor_batch.pop("raw_prompt_ids"), non_tensor_batch.pop("multi_modal_data"), strict=True
            ):
                vllm_inputs.append({"prompt_token_ids": raw_prompt_ids, "multi_modal_data": multi_modal_data})
        else:
            vllm_inputs = [
                {"prompt_token_ids": raw_prompt_ids} for raw_prompt_ids in non_tensor_batch.pop("raw_prompt_ids")
            ]

        for input_data in vllm_inputs:
            # Ensure token IDs are lists or numpy arrays
            if not isinstance(input_data["prompt_token_ids"], list | np.ndarray):
                raise TypeError(
                    f"prompt_token_ids must be a list or numpy array, got {type(input_data['prompt_token_ids'])}"
                )

            input_data["prompt_token_ids"] = list(input_data["prompt_token_ids"])

        do_sample = prompts.meta_info.get("do_sample", True)
        is_validate = prompts.meta_info.get("validate", False)
        if not do_sample:
            kwargs = {
                "best_of": 1,
                "top_p": 1.0,
                "top_k": -1,
                "min_p": 0.0,
                "temperature": 0,
                "n": 1,  # if greedy, only 1 response
            }
        elif is_validate:
            # TODO: try **
            kwargs = {
                "top_k": self.config.val_kwargs.top_k,
                "top_p": self.config.val_kwargs.top_p,
                "temperature": self.config.val_kwargs.temperature,
                "n": 1,  # if validate, already repeat in ray_trainer
            }

        response_cap = prompts.meta_info.get("response_max_tokens_cap", None)
        if response_cap is not None:
            try:
                response_cap = int(response_cap)
                max_resp_len = int(self.config.response_length)
                current_max = kwargs.get("max_tokens", getattr(self.sampling_params, "max_tokens", max_resp_len))
                current_max = int(current_max) if current_max is not None else max_resp_len
                kwargs["max_tokens"] = max(1, min(response_cap, current_max, max_resp_len))
            except Exception:
                logger.warning("Ignore invalid response_max_tokens_cap=%s", response_cap)

        tail_validate_caps = os.environ.get("VERL_ELASTIC_TAIL_VALIDATE_LEVEL_TOKENS")
        if tail_validate_caps:
            try:
                level_caps = [int(x.strip()) for x in tail_validate_caps.split(",") if x.strip()]
            except ValueError:
                logger.warning("Invalid VERL_ELASTIC_TAIL_VALIDATE_LEVEL_TOKENS=%s",
                               tail_validate_caps)
                level_caps = []

            if level_caps and torch.distributed.is_available() and torch.distributed.is_initialized():
                world_size = torch.distributed.get_world_size()
                global_rank = torch.distributed.get_rank()
                max_resp_len = int(self.config.response_length)
                current_max = kwargs.get("max_tokens",
                                         getattr(self.sampling_params, "max_tokens", max_resp_len))
                current_max = int(current_max) if current_max is not None else max_resp_len

                # Build repeated-halving buckets: N/2, N/4, ..., 1, 1.
                bucket_sizes: list[int] = []
                remaining = world_size
                next_bucket = max(world_size // 2, 1)
                while remaining > 0:
                    take = min(next_bucket, remaining)
                    bucket_sizes.append(take)
                    remaining -= take
                    next_bucket = max(next_bucket // 2, 1)

                bucket_idx = len(bucket_sizes) - 1
                cursor = 0
                for bucket_id, bucket_size in enumerate(bucket_sizes):
                    if global_rank < cursor + bucket_size:
                        bucket_idx = bucket_id
                        break
                    cursor += bucket_size

                cap_idx = min(bucket_idx, len(level_caps) - 1)
                kwargs["max_tokens"] = max(1, min(level_caps[cap_idx], current_max, max_resp_len))
                logger.info(
                    "Elastic tail validation cap override: rank=%s world_size=%s "
                    "bucket=%s/%s bucket_sizes=%s max_tokens=%s",
                    global_rank,
                    world_size,
                    bucket_idx,
                    len(bucket_sizes),
                    bucket_sizes,
                    kwargs["max_tokens"],
                )

        lora_requests = None
        if self.lora_kwargs:
            lora_int_ids = list(self.inference_engine.llm_engine.list_loras())
            if len(lora_int_ids) > 0:
                lora_int_id = lora_int_ids[0]
                lora_requests = [
                    LoRARequest(lora_name=f"{lora_int_id}", lora_int_id=lora_int_id, lora_path="/simon-stub-path")
                ] * batch_size

        # users can customize different sampling_params at different run
        # 核心运行引擎调用
        with self.update_sampling_params(**kwargs):
            outputs = self.inference_engine.generate(
                prompts=vllm_inputs,  # because we have already convert it to prompt token id
                sampling_params=self.sampling_params,
                lora_request=lora_requests,
                use_tqdm=True,
            )

            # TODO(sgm): disable logprob when recompute_log_prob is enable
            # if n = 1: (bs, response_length) ; if n > 1: (bs * n, response_length)

            response = []
            rollout_log_probs = []
            request_ids = []
            rollout_ranks = []
            local_rank = -1
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                local_rank = int(torch.distributed.get_rank())
            for output in outputs:
                req_id = str(getattr(output, "request_id", ""))
                for sample_id in range(len(output.outputs)):
                    response_ids = output.outputs[sample_id].token_ids
                    response.append(response_ids)
                    request_ids.append(req_id)
                    rollout_ranks.append(local_rank)
                    if self.config.calculate_log_probs:
                        curr_log_prob = []
                        for i, logprob in enumerate(output.outputs[sample_id].logprobs):
                            curr_log_prob.append(logprob[response_ids[i]].logprob)
                        rollout_log_probs.append(curr_log_prob)

            response = pad_2d_list_to_length(response, self.pad_token_id, max_length=self.config.response_length).to(
                idx.device
            )
            if self.config.calculate_log_probs:
                rollout_log_probs = pad_2d_list_to_length(
                    rollout_log_probs, -1, max_length=self.config.response_length
                ).to(idx.device)
                rollout_log_probs = rollout_log_probs.to(torch.float32)

            seq = torch.cat([idx, response], dim=-1)
            num_responses = int(response.shape[0])
            if len(request_ids) == num_responses:
                non_tensor_batch["request_id"] = np.array(request_ids, dtype=object)
                non_tensor_batch["rollout_rank"] = np.array(rollout_ranks, dtype=np.int64)
            else:
                logger.warning(
                    "Skip attaching request_id/rollout_rank due to length mismatch: "
                    "request_ids=%d, responses=%d",
                    len(request_ids),
                    num_responses,
                )

        response_length = response.size(1)
        delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
        delta_position_id = delta_position_id.unsqueeze(0).expand(batch_size, -1)
        if position_ids.dim() == 3:  # qwen2vl mrope (batch size, 4, seq len)
            delta_position_id = delta_position_id.view(batch_size, 1, -1).expand(batch_size, position_ids.size(1), -1)

        # TODO(sgm): fix position_ids on right_pad
        # prompt: left pad + response: right pad
        # attention_mask: [0,0,0,0,1,1,1,1, | 1,1,1,0,0,0,0,0]
        # position_ids:   [0,0,0,0,0,1,2,3, | 4,5,6,7,8,9,10,11]
        response_position_ids = position_ids[..., -1:] + delta_position_id
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)
        response_attention_mask = get_response_mask(
            response_id=response, eos_token=eos_token_id, dtype=attention_mask.dtype
        )
        attention_mask = torch.cat((attention_mask, response_attention_mask), dim=-1)

        # all the tp ranks should contain the same data here. data in all ranks are valid
        batch = TensorDict(
            {
                "prompts": idx,
                "responses": response,
                "input_ids": seq,  # here input_ids become the whole sentences
                "attention_mask": attention_mask,
                "position_ids": position_ids,
            },
            batch_size=batch_size,
        )
        if self.config.calculate_log_probs:
            # we will recompute old log prob with actor
            batch["rollout_log_probs"] = rollout_log_probs

        return DataProto(batch=batch, non_tensor_batch=non_tensor_batch)

    async def resume(self, tags: list[str]):
        """Resume rollout weights or kv cache in GPU memory.

        Args:
            tags: weights or kv_cache.
        """
        if not self.config.free_cache_engine:
            return

        if "weights" in tags:
            self.onload_model_weights()
        elif "kv_cache" in tags:
            self.init_cache_engine()

    async def release(self):
        """Release weights and kv cache in GPU memory."""
        if not self.config.free_cache_engine:
            return

        self.free_cache_engine()
        self.offload_model_weights()

    async def update_weights(self, weights: Generator[tuple[str, torch.Tensor], None, None], **kwargs):
        """Update the weights of the rollout model.

        Args:
            weights: A generator that yields the name of the weight tensor and the tensor itself.
        """
        filtered_weights: list[tuple[str, torch.Tensor]] = []
        skipped = 0
        skipped_names: list[str] = []
        for name, weight in weights:
            if isinstance(weight, torch.Tensor) and weight.numel() == 0:
                skipped += 1
                if len(skipped_names) < 8:
                    skipped_names.append(name)
                continue
            filtered_weights.append((name, weight))
        if skipped > 0 and not getattr(self, "_empty_weight_update_warned",
                                       False):
            logger.warning(
                "Skip %s empty rollout weight shards during update_weights. sample_names=%s",
                skipped,
                skipped_names,
            )
            self._empty_weight_update_warned = True

        weights = filtered_weights
        peft_config, base_sync_done = kwargs.get("peft_config", None), kwargs.get("base_sync_done", False)
        if peft_config and base_sync_done:
            lora_int_id = int(time.time_ns() % 0x7FFFFFFF)
            lora_reqest = TensorLoRARequest(
                lora_name=f"{lora_int_id}",
                lora_int_id=lora_int_id,
                lora_path="simon_lora_path",
                peft_config=asdict(peft_config),
                lora_tensors=dict(weights),
            )
            self.inference_engine.llm_engine.add_lora(lora_reqest)
            logger.info(f"vLLM load weights, loaded_params: {len(weights)}")
        else:
            from verl.utils.vllm.patch import patch_vllm_moe_model_weight_loader

            model = self.inference_engine.llm_engine.model_executor.driver_worker.worker.model_runner.get_model()
            patch_vllm_moe_model_weight_loader(model)
        if (self.elastic_execution_mode in (1, 2)
                and self.elastic_init_redundancy_expert > 0
                and not getattr(self, "_lossless_weight_update_sync_logged", False)):
            logger.info(
                "Elastic execution mode=%s enables rollout.update_weights with redundant experts.",
                self.elastic_execution_mode)
            self._lossless_weight_update_sync_logged = True
        _log_custom_mode1_rollout_memory("before_update_weights_load")
        _log_custom_mode1_rollout_state(self, "before_update_weights_load")
        if (self.elastic_execution_mode == 1
                and int(os.getenv("VLLM_ASCEND_ELASTIC_MIN_COMPUTE_GROUP_SIZE",
                                  "0") or "0") <= 2):
            try:
                torch.npu.synchronize()
                gc.collect()
                torch.npu.empty_cache()
            except Exception:
                pass
            _log_custom_mode1_rollout_memory(
                "before_update_weights_load_after_trim")
        staged_weights = _materialize_rollout_weight_staging(weights)
        model.load_weights(staged_weights)
        refresh_gpu_buffer_aliases = getattr(self, "_refresh_gpu_buffer_aliases",
                                             None)
        if callable(refresh_gpu_buffer_aliases):
            refresh_gpu_buffer_aliases()
        staged_weights = []
        weights = []
        filtered_weights = []
        gc.collect()
        torch.npu.empty_cache()
        _log_custom_mode1_rollout_state(self, "after_update_weights_load")
        _log_custom_mode1_rollout_memory("after_update_weights_load")
        ###new wj
    def get_record(self):
        return moe_stats.snapshot_pattern()
    
    def flush_record(self):
        return moe_stats.reset_epoch()


# https://github.com/vllm-project/vllm/issues/13175
def _monkey_patch_compute_logits(model, vocab_size: int):
    original_compute_logits = model.compute_logits

    def compute_logits(
        self,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        logits = original_compute_logits(*args, **kwargs)
        logits[..., vocab_size:] = float("-inf")
        return logits

    model.compute_logits = MethodType(compute_logits, model)


class vLLMAsyncRollout(BaseRollout):
    """vLLMAsyncRollout is a thin wrapper of WorkerWrapperBase, which is engine in single worker process."""

    def __init__(
        self,
        config: RolloutConfig,
        model_config: HFModelConfig,
        device_mesh: DeviceMesh,
    ):
        super().__init__(config, model_config, device_mesh)
        self.tokenizer = model_config.tokenizer
        self.inference_engine: WorkerWrapperBase = None
        self.address = self._init_zeromq()
        self.lora_config = (
            {"max_loras": 1, "max_lora_rank": model_config.lora_rank} if model_config.lora_rank > 0 else {}
        )

        # https://github.com/vllm-project/vllm/issues/25171
        if config.layered_summon or config.expert_parallel_size > 1:
            self.sleep_level = 1
        else:
            self.sleep_level = VLLM_SLEEP_LEVEL

    def _init_zeromq(self) -> str:
        tensor_parallel_size = self.config.tensor_model_parallel_size

        # single node: ipc, multi nodes: tcp
        local_world_size = int(os.environ["RAY_LOCAL_WORLD_SIZE"])
        socket_type = "ipc" if tensor_parallel_size <= local_world_size else "tcp"

        # File lock to prevent multiple workers listen to same port
        with FileLock(f"/tmp/verl_vllm_zmq_{getpass.getuser()}.lock"):
            if socket_type == "ipc":
                pid = os.getpid()
                address = f"ipc:///tmp/verl_vllm_zmq_{pid}_{getpass.getuser()}.ipc"
            else:
                ip, port = self._get_free_port()
                address = f"tcp://{ip}:{port}"
            context = zmq.asyncio.Context()
            self.socket = context.socket(zmq.REP)
            self.socket.bind(address)

        loop = asyncio.get_running_loop()
        self.zmq_loop_task = loop.create_task(self._loop_forever())

        return address

    def _get_free_port(self):
        ip = ray.util.get_node_ip_address()
        with socket.socket() as sock:
            sock.bind(("", 0))
            port = sock.getsockname()[1]
        return ip, port

    async def _loop_forever(self):
        while True:
            try:
                message = await self.socket.recv()
                method, args, kwargs = pickle.loads(message)
                result = await self._execute_method(method, *args, **kwargs)
                await self.socket.send(pickle.dumps(result))
            except Exception as e:
                logger.exception(f"vLLMAsyncRollout _loop_forever error: {e}")
                os._exit(-1)

    def _init_worker(self, all_kwargs: list[dict[str, Any]]):
        """Initialize worker engine."""
        if not torch.distributed.is_initialized():
            initialize_global_process_group_ray()
        all_kwargs[0]["rank"] = int(os.environ["RANK"])
        device_name = "NPU" if is_npu_available else "GPU"
        all_kwargs[0]["local_rank"] = (
            0
            if not ray_noset_visible_devices()
            else int(ray.get_runtime_context().get_accelerator_ids()[device_name][0])
        )
        self.vllm_config = all_kwargs[0]["vllm_config"]
        if self.lora_config:
            lora_dtype = getattr(torch, self.config.dtype)
            self.vllm_config.lora_config = LoRAConfig(lora_dtype=lora_dtype, **self.lora_config)
        self.inference_engine = WorkerWrapperBase(vllm_config=self.vllm_config)
        self.inference_engine.init_worker(all_kwargs)

    def _load_model(self, *args, **kwargs):
        self.inference_engine.load_model(*args, **kwargs)
        _monkey_patch_compute_logits(self.inference_engine.worker.model_runner.model, len(self.tokenizer))

    async def _execute_method(self, method: str | bytes, *args, **kwargs):
        if method == "init_worker":
            return self._init_worker(*args, **kwargs)
        elif method == "load_model":
            return self._load_model(*args, **kwargs)
        elif method == "sleep" or method == "wake_up":
            raise ValueError("wake_up and sleep should not be called through ZeroMQ")
        else:
            return self.inference_engine.execute_method(method, *args, **kwargs)

    async def resume(self, tags: list[str]):
        """Resume rollout weights or kv cache in GPU memory.

        Args:
            tags: weights or kv_cache.
        """
        if self.config.free_cache_engine:
            self.inference_engine.wake_up(tags=tags)

    async def release(self):
        """Release weights and kv cache in GPU memory."""
        if self.config.free_cache_engine:
            self.inference_engine.sleep(level=self.sleep_level)

    async def update_weights(self, weights: Generator[tuple[str, torch.Tensor], None, None], **kwargs):
        """Update the weights of the rollout model.

        Args:
            weights: A generator that yields the name of the weight tensor and the tensor itself.
        """
        peft_config, base_sync_done = kwargs.get("peft_config", None), kwargs.get("base_sync_done", False)
        if peft_config and base_sync_done:
            lora_int_id = int(time.time_ns() % 0x7FFFFFFF)
            lora_reqest = TensorLoRARequest(
                lora_name=f"{lora_int_id}",
                lora_int_id=lora_int_id,
                lora_path="simon_lora_path",
                peft_config=asdict(peft_config),
                lora_tensors=dict(weights),
            )
            self.inference_engine.worker.add_lora(lora_reqest)
            logger.info(f"vLLM load weights, loaded_params: {len(weights)}")
        else:
            from verl.utils.vllm.patch import patch_vllm_moe_model_weight_loader

            model = self.inference_engine.worker.model_runner.model
            patch_vllm_moe_model_weight_loader(model)
            staged_weights = _materialize_rollout_weight_staging(list(weights))
            model.load_weights(staged_weights)
            staged_weights = []
            gc.collect()
            torch.npu.empty_cache()

    def generate_sequences(self, prompts: DataProto) -> DataProto:
        """Batch generate sequences in sync mode."""
        raise NotImplementedError

    # ==================== server mode public methods ====================

    def get_zeromq_address(self):
        return self.address
