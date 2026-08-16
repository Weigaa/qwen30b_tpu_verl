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
import re
import socket
import time
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from types import MethodType
from typing import Any, Generator, Iterable

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
from vllm.config import (CUDAGraphMode, CompilationConfig, CompilationLevel,
                         LoRAConfig)
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
from verl.utils.fixed_work_replay import load_fixed_work_replay
from verl.utils.profiler import GPUMemoryLogger
from verl.utils.rollout_seeding import derive_request_seed
from verl.utils.ray_utils import ray_noset_visible_devices
from verl.utils.torch_functional import get_response_mask, pad_2d_list_to_length
from verl.utils.vllm import TensorLoRARequest, VLLMHijack, is_version_ge
from verl.workers.config import HFModelConfig, RolloutConfig
from verl.workers.rollout.base import BaseRollout
#new wj import
import copy
import hashlib
import os
from vllm.utils.moe_stats import moe_stats

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def _env_flag(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).lower() in ("1", "true", "yes", "on")


def _model_weight_pointer_signature(
        model: torch.nn.Module) -> tuple[tuple[str, int, tuple[int, ...]], ...]:
    """Return the parameter-address contract used by native graph sleep.

    ACLGraph replay is safe across an RL weight refresh only while every
    captured parameter keeps the same device address and logical shape.  The
    0.14 stack relies on CaMem to preserve that property.  Keep an explicit
    runtime contract in the 0.11 port so an incompatible loader falls back to
    graph invalidation and recapture instead of silently replaying stale
    addresses.
    """
    return tuple((name, int(param.data_ptr()), tuple(param.shape))
                 for name, param in model.named_parameters())


def _kv_cache_pointer_signature(value: Any) -> tuple[tuple[int, tuple[int, ...]], ...]:
    """Flatten KV-cache tensor addresses for the native-sleep replay guard."""
    result: list[tuple[int, tuple[int, ...]]] = []

    def visit(item: Any) -> None:
        if isinstance(item, torch.Tensor):
            result.append((int(item.data_ptr()), tuple(item.shape)))
        elif isinstance(item, dict):
            for key in sorted(item, key=str):
                visit(item[key])
        elif isinstance(item, (list, tuple)):
            for child in item:
                visit(child)

    visit(value)
    return tuple(result)


def _sample_weight_tensor(tensor: torch.Tensor,
                          sample_count: int = 33) -> torch.Tensor:
    """Copy deterministic logical samples without retaining a full weight."""
    flat = tensor.detach().reshape(-1)
    numel = int(flat.numel())
    if numel == 0:
        return torch.empty(0, dtype=tensor.dtype)
    count = min(max(int(sample_count), 1), numel)
    if count == numel:
        sample = flat
    else:
        positions = sorted({
            index * (numel - 1) // (count - 1)
            for index in range(count)
        })
        indices = torch.tensor(positions,
                               dtype=torch.long,
                               device=flat.device)
        sample = flat.index_select(0, indices)
    return sample.cpu().contiguous().clone()


def _weight_compare_category(name: str) -> str:
    if name in ("model.embed_tokens.weight", "lm_head.weight"):
        return "embedding_or_head"
    if "mlp.experts." in name:
        return "routed_expert"
    if "mlp.shared_experts." in name:
        return "shared_expert"
    if ".mlp.gate." in name:
        return "router"
    if ".self_attn." in name:
        return "attention"
    if "layernorm" in name or name == "model.norm.weight":
        return "norm"
    if ".mlp." in name:
        return "dense_mlp"
    return "other"


def _compare_weight_samples(
    model: Any,
    references: dict[str, tuple[tuple[int, ...], torch.dtype, torch.Tensor]],
) -> dict[str, Any]:
    current_params = dict(model.named_parameters())
    category_totals: dict[str, int] = {}
    category_mismatches: dict[str, int] = {}
    mismatch_samples: list[str] = []
    missing: list[str] = []
    sample_errors: list[str] = []
    matched = 0

    for name, (expected_shape, expected_dtype, expected_sample) in references.items():
        category = _weight_compare_category(name)
        category_totals[category] = category_totals.get(category, 0) + 1
        param = current_params.get(name)
        if param is None:
            missing.append(name)
            category_mismatches[category] = (
                category_mismatches.get(category, 0) + 1)
            continue
        try:
            actual_sample = _sample_weight_tensor(
                param, sample_count=int(expected_sample.numel()))
        except Exception as error:
            sample_errors.append(f"{name}:{type(error).__name__}:{error}")
            category_mismatches[category] = (
                category_mismatches.get(category, 0) + 1)
            continue
        same_metadata = (
            tuple(param.shape) == expected_shape and param.dtype == expected_dtype)
        same_values = torch.equal(actual_sample, expected_sample)
        if same_metadata and same_values:
            matched += 1
            continue
        category_mismatches[category] = (
            category_mismatches.get(category, 0) + 1)
        if len(mismatch_samples) < 40:
            max_abs = float("nan")
            if actual_sample.shape == expected_sample.shape:
                max_abs = float(
                    (actual_sample.float() - expected_sample.float()).abs().max().item())
            mismatch_samples.append(
                f"{name}:shape={tuple(param.shape)}/{expected_shape}:"
                f"dtype={param.dtype}/{expected_dtype}:max_sample_abs={max_abs:.8g}")

    return {
        "total": len(references),
        "matched": matched,
        "mismatched": len(references) - matched,
        "category_totals": category_totals,
        "category_mismatches": category_mismatches,
        "missing": missing[:20],
        "sample_errors": sample_errors[:20],
        "mismatch_samples": mismatch_samples,
    }


def _tensor_sha256(tensor: torch.Tensor) -> str:
    cpu_tensor = tensor.detach().cpu().contiguous()
    byte_view = cpu_tensor.view(torch.uint8).numpy()
    return hashlib.sha256(byte_view).hexdigest()


def _compare_layer1_routed_w13(
    model: Any,
    initial_w13: torch.Tensor,
    initial_expert_map: tuple[int, ...],
    staged_weights: dict[str, torch.Tensor],
) -> dict[str, Any]:
    param_name = "model.layers.1.mlp.experts.w13_weight"
    current = dict(model.named_parameters()).get(param_name)
    if current is None:
        return {"error": f"missing {param_name}"}
    current_cpu = current.detach().cpu().contiguous()
    if tuple(current_cpu.shape) != tuple(initial_w13.shape):
        return {
            "error": (
                f"shape mismatch {tuple(current_cpu.shape)} "
                f"vs {tuple(initial_w13.shape)}")
        }
    if initial_w13.ndim != 3 or initial_w13.shape[1] % 2:
        return {"error": f"unexpected w13 shape {tuple(initial_w13.shape)}"}

    half = int(initial_w13.shape[1]) // 2
    expected_hashes: dict[str, str] = {}
    post_hashes: dict[str, str] = {}
    global_to_local = {
        global_id: local_id
        for global_id, local_id in enumerate(initial_expert_map)
        if local_id >= 0
    }
    for global_id, local_id in global_to_local.items():
        if local_id >= int(initial_w13.shape[0]):
            return {
                "error": (
                    f"expert map slot {local_id} exceeds w13 capacity "
                    f"{initial_w13.shape[0]}")
            }
        for projection, row_slice in (
            ("gate_proj", slice(0, half)),
            ("up_proj", slice(half, 2 * half)),
        ):
            label = f"expert{global_id}.{projection}"
            expected_hashes[label] = _tensor_sha256(
                initial_w13[local_id, row_slice])
            post_hashes[label] = _tensor_sha256(
                current_cpu[local_id, row_slice])

    expected_by_hash = {
        digest: label for label, digest in expected_hashes.items()
    }
    stream_matches = {
        name: expected_by_hash.get(_tensor_sha256(weight), "unknown")
        for name, weight in sorted(staged_weights.items())
    }
    post_matches = {
        label: expected_by_hash.get(digest, "unknown")
        for label, digest in post_hashes.items()
    }
    expected_stream_names = {
        f"model.layers.1.mlp.experts.{global_id}.{projection}.weight"
        for global_id in global_to_local
        for projection in ("gate_proj", "up_proj")
    }
    return {
        "global_to_local": global_to_local,
        "expected_stream_count": len(expected_stream_names),
        "captured_stream_count": len(staged_weights),
        "missing_stream_names": sorted(expected_stream_names - staged_weights.keys()),
        "extra_stream_names": sorted(staged_weights.keys() - expected_stream_names),
        "stream_matches": stream_matches,
        "post_matches": post_matches,
    }


def _weight_comparison_failures(
    comparison: dict[str, Any],
    routed_comparison: dict[str, Any],
) -> list[str]:
    failures: list[str] = []
    total = comparison.get("total")
    if not isinstance(total, int) or total <= 0:
        failures.append("sample comparison has no reference weights")
    if comparison.get("matched") != total or comparison.get("mismatched") != 0:
        failures.append("sampled HF weights do not match synchronized weights")
    for key in ("category_mismatches", "missing", "sample_errors", "mismatch_samples"):
        if comparison.get(key):
            failures.append(f"sample comparison reports {key}")

    if routed_comparison.get("error"):
        failures.append(f"routed comparison error: {routed_comparison['error']}")
        return failures
    expected = routed_comparison.get("expected_stream_count")
    captured = routed_comparison.get("captured_stream_count")
    stream_matches = routed_comparison.get("stream_matches")
    post_matches = routed_comparison.get("post_matches")
    if not isinstance(expected, int) or expected <= 0 or captured != expected:
        failures.append("routed comparison has an incomplete staged stream")
    if routed_comparison.get("missing_stream_names"):
        failures.append("routed comparison is missing staged weights")
    if routed_comparison.get("extra_stream_names"):
        failures.append("routed comparison has unexpected staged weights")
    if not isinstance(stream_matches, dict) or len(stream_matches) != expected:
        failures.append("routed staged-weight matches are incomplete")
    elif any(value == "unknown" for value in stream_matches.values()):
        failures.append("routed staged weights contain an unknown source")
    if not isinstance(post_matches, dict) or len(post_matches) != expected:
        failures.append("routed post-load matches are incomplete")
    elif any(label != source for label, source in post_matches.items()):
        failures.append("routed post-load expert placement is incorrect")
    if isinstance(stream_matches, dict) and isinstance(post_matches, dict):
        if set(stream_matches.values()) != set(post_matches):
            failures.append("routed staged and resident expert sets differ")
    return failures


def _resolve_mla_attention(mla_attn: Any) -> Any:
    if mla_attn is None:
        return None
    if callable(getattr(mla_attn, "process_weights_after_loading", None)):
        return mla_attn
    inner_attention = getattr(mla_attn, "mla_attn", None)
    if callable(getattr(inner_attention, "process_weights_after_loading", None)):
        return inner_attention
    return None


def _resolve_mla_impl(mla_attn: Any) -> Any:
    attention = _resolve_mla_attention(mla_attn)
    return getattr(attention, "impl", None)


def _refresh_mla_derived_weights(
    model: Any,
    act_dtype: Any,
    *,
    require_complete: bool = False,
) -> int:
    layers = getattr(getattr(model, "model", None), "layers", None)
    if layers is None:
        return 0
    start_layer = int(getattr(model.model, "start_layer", 0))
    end_layer = int(getattr(model.model, "end_layer", len(layers)))
    refreshed = 0
    impl_names: set[str] = set()
    for layer_idx in range(start_layer, end_layer):
        self_attn = getattr(layers[layer_idx], "self_attn", None)
        attention = _resolve_mla_attention(
            getattr(self_attn, "mla_attn", None))
        if attention is None:
            continue
        attention.process_weights_after_loading(act_dtype)
        impl = getattr(attention, "impl", None)
        impl_names.add(type(impl).__name__)
        if require_complete and all(
            hasattr(impl, name)
            for name in (
                "num_heads",
                "kv_lora_rank",
                "qk_nope_head_dim",
                "v_head_dim",
            )
        ):
            expected_shapes = {
                "W_UV": (
                    int(impl.num_heads),
                    int(impl.kv_lora_rank),
                    int(impl.v_head_dim),
                ),
                "W_UK_T": (
                    int(impl.num_heads),
                    int(impl.qk_nope_head_dim),
                    int(impl.kv_lora_rank),
                ),
            }
            for name, expected_shape in expected_shapes.items():
                tensor = getattr(impl, name, None)
                if not isinstance(tensor, torch.Tensor):
                    raise RuntimeError(
                        f"MLA layer {layer_idx} did not materialize {name}")
                if tuple(tensor.shape) != expected_shape:
                    raise RuntimeError(
                        f"MLA layer {layer_idx} has {name} shape "
                        f"{tuple(tensor.shape)}, expected {expected_shape}")
        refreshed += 1
    expected = max(end_layer - start_layer, 0)
    if require_complete:
        if refreshed != expected:
            raise RuntimeError(
                f"Incomplete MLA refresh: refreshed={refreshed}, "
                f"expected={expected}")
        logger.warning(
            "MLA refresh complete: refreshed=%s expected=%s impl=%s",
            refreshed,
            expected,
            ",".join(sorted(impl_names)),
        )
    return refreshed


def _rollout_elastic_aclgraph_enabled(model_runner: Any) -> bool:
    if (not is_npu_available
            or not envs_ascend.VLLM_ASCEND_ELASTIC_ACLGRAPH):
        return False
    if not bool(getattr(model_runner, "use_aclgraph", False)):
        return False
    compilation_config = getattr(
        getattr(model_runner, "vllm_config", None), "compilation_config", None)
    cudagraph_mode = getattr(compilation_config, "cudagraph_mode", None)
    return getattr(cudagraph_mode, "name", "NONE") != "NONE"


def _validate_rollout_elastic_aclgraph_runtime(model_runner: Any) -> None:
    """Fail closed if the dynamic MoE graph boundary is not active."""
    if (not envs_ascend.VLLM_ASCEND_ENABLE_ELASTIC_PARALLEL_SHRINK
            or getattr(model_runner, "_elastic_aclgraph_runtime_validated",
                       False)):
        return

    failures: list[str] = []
    task_queue = os.getenv("TASK_QUEUE_ENABLE")
    allow_task_queue_2 = _env_flag(
        "VLLM_ASCEND_ELASTIC_ACLGRAPH_ALLOW_TASK_QUEUE_2", "0")
    if task_queue != "1" and not (task_queue == "2"
                                  and allow_task_queue_2):
        failures.append(
            "TASK_QUEUE_ENABLE must be 1; TQ2 requires the explicit "
            "VLLM_ASCEND_ELASTIC_ACLGRAPH_ALLOW_TASK_QUEUE_2=1 "
            "diagnostic gate")
    if _env_flag("VLLM_ENABLE_GRAPH_MODE", "0"):
        failures.append("VLLM_ENABLE_GRAPH_MODE must be 0 (TorchAir disabled)")

    compilation_config = getattr(
        getattr(model_runner, "vllm_config", None), "compilation_config", None)
    cudagraph_mode = getattr(compilation_config, "cudagraph_mode", None)
    mode_name = getattr(cudagraph_mode, "name", "NONE")
    attention_captured = bool(
        envs_ascend.VLLM_ASCEND_ELASTIC_ACLGRAPH_CAPTURE_ATTENTION)
    supported_modes = ("PIECEWISE", "FULL_DECODE_ONLY")
    if mode_name not in supported_modes:
        failures.append(
            "cudagraph mode must be PIECEWISE or FULL_DECODE_ONLY, "
            f"got {mode_name}")
    if not bool(getattr(compilation_config, "cudagraph_copy_inputs", False)):
        failures.append(
            "Elastic ACLGraph requires cudagraph_copy_inputs=True to "
            "refresh live decode inputs")
    elastic_op = "vllm.elastic_ascend_moe_forward"
    attention_op = "vllm.unified_ascend_attention_with_output"
    splitting_ops = getattr(compilation_config, "splitting_ops", ())
    moe_captured = bool(
        envs_ascend.VLLM_ASCEND_ELASTIC_ACLGRAPH_CAPTURE_MOE)
    native_full_decode = mode_name == "FULL_DECODE_ONLY"
    if native_full_decode:
        if not attention_captured:
            failures.append(
                "FULL_DECODE_ONLY requires Attention capture")
        if not moe_captured:
            failures.append(
                "FULL_DECODE_ONLY requires elastic MoE capture")
        for op in (attention_op, elastic_op):
            if op in splitting_ops:
                failures.append(
                    f"FULL_DECODE_ONLY graph op remains a splitting op: {op}")
    else:
        if attention_captured:
            if elastic_op not in splitting_ops:
                failures.append(
                    "Attention-in-PIECEWISE requires elastic MoE/HCCL to "
                    f"remain a splitting op: {elastic_op}")
        elif moe_captured:
            if elastic_op in splitting_ops:
                failures.append(
                    f"captured MoE remains a splitting op: {elastic_op}")
        elif elastic_op not in splitting_ops:
            failures.append(f"splitting_ops is missing {elastic_op}")
    if not hasattr(torch.ops.vllm, "elastic_ascend_moe_forward"):
        failures.append("elastic Ascend MoE custom op is not registered")

    if attention_captured:
        if not native_full_decode and attention_op not in splitting_ops:
            failures.append(
                "dynamic KV write Attention boundary is missing: "
                f"{attention_op}")
        if not hasattr(torch.ops.vllm,
                       "unified_ascend_attention_with_output"):
            failures.append(
                "Ascend attention custom op is not registered: "
                "unified_ascend_attention_with_output")
        from vllm.v1.attention.backends.utils import AttentionCGSupport
        from vllm_ascend.attention.attention_v1 import (
            AscendAttentionMetadataBuilder,
        )

        if (AscendAttentionMetadataBuilder.aclgraph_support !=
                AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE):
            failures.append(
                "Ascend attention metadata builder does not support uniform "
                "single-token ACLGraph decode")
    elif not native_full_decode and attention_op not in splitting_ops:
        failures.append(
            f"attention fallback splitting op is missing: {attention_op}")

    from vllm_ascend.ops.fused_moe import AscendFusedMoE

    model = model_runner.get_model()
    moe_layers = [
        module for module in model.modules()
        if isinstance(module, AscendFusedMoE)
    ]
    if not moe_layers:
        failures.append("model has no AscendFusedMoE layers")
    required_methods = (
        "refresh_elastic_groups",
        "activate_lossless_local_experts",
        "set_runtime_num_experts",
    )
    static_context = getattr(compilation_config, "static_forward_context", {})
    for layer in moe_layers:
        missing = [
            name for name in required_methods
            if not callable(getattr(layer, name, None))
        ]
        if missing:
            failures.append(
                f"{getattr(layer, 'layer_name', type(layer).__name__)} is "
                f"missing elastic methods {missing}")
        layer_name = getattr(layer, "layer_name", None)
        if not layer_name or static_context.get(layer_name) is not layer:
            failures.append(
                f"AscendFusedMoE {layer_name!r} is not registered in "
                "static_forward_context")

    if failures:
        raise RuntimeError("Elastic ACLGraph preflight failed: " +
                           "; ".join(failures))
    model_runner._elastic_aclgraph_runtime_validated = True
    logger.warning(
        "Elastic ACLGraph runtime preflight passed: mode=%s moe_layers=%s "
        "splitting_op=%s attention_captured=%s moe_captured=%s "
        "native_full_decode=%s",
        mode_name,
        len(moe_layers),
        elastic_op,
        attention_captured,
        moe_captured,
        native_full_decode,
    )


def _invalidate_rollout_aclgraphs(model_runner: Any, stage: str) -> int:
    if not _rollout_elastic_aclgraph_enabled(model_runner):
        return 0
    from vllm_ascend.compilation.acl_graph import (
        clear_aclgraph_caches,
        disable_aclgraph_dispatch,
    )

    cleared_keys = disable_aclgraph_dispatch(model_runner)
    cleared = clear_aclgraph_caches(getattr(model_runner, "model", None))
    logger.warning(
        "Elastic ACLGraph invalidated %s cached entries and %s dispatch "
        "keys after %s",
        cleared,
        cleared_keys,
        stage,
    )
    return cleared


def _recapture_rollout_aclgraphs(model_runner: Any, stage: str) -> None:
    if not _rollout_elastic_aclgraph_enabled(model_runner):
        return
    _validate_rollout_elastic_aclgraph_runtime(model_runner)
    # KV-cache resume may call Worker.compile_or_warm_up_model() and populate
    # the graph cache after update_weights marked this rollout dirty.  Calling
    # capture_model() again without clearing that cache makes the dummy capture
    # replay the existing graph *inside* a new graph-capture context.  This is
    # invalid for graphs containing HCCL collectives and can leave the first
    # live decode blocked in torch.npu.synchronize().  Treat recapture as one
    # atomic lifecycle transition regardless of who populated the cache since
    # the original invalidation.
    cleared = _invalidate_rollout_aclgraphs(
        model_runner, f"{stage}_immediately_before_recapture")
    capture_moe = bool(
        envs_ascend.VLLM_ASCEND_ELASTIC_ACLGRAPH_CAPTURE_MOE)
    world_cpu_group = None
    if capture_moe:
        if not (torch.distributed.is_available()
                and torch.distributed.is_initialized()):
            raise RuntimeError(
                "Full-MoE ACLGraph recapture requires an initialized "
                "distributed process group")
        from vllm.distributed.parallel_state import get_world_group

        world_cpu_group = get_world_group().cpu_group
        torch.distributed.barrier(group=world_cpu_group)
    logger.warning(
        "Elastic ACLGraph recapture starting after %s via "
        "model_runner.capture_model; cleared_entries=%s",
        stage,
        cleared,
    )
    model_runner.capture_model()
    if world_cpu_group is not None:
        # capture_model() warms and records HCCL collectives. Do not let a fast
        # rank replay its new graph while another rank is still recording the
        # same collective sequence.
        torch.distributed.barrier(group=world_cpu_group)
    logger.warning("Elastic ACLGraph recapture finished after %s", stage)


def _sync_shrink_aware_env_from_meta(meta_info: dict) -> None:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        from verl.experimental.dataset.shrink_aware_assignment import (
            select_shrink_aware_worker_plan,
        )
        select_shrink_aware_worker_plan(
            meta_info, int(torch.distributed.get_rank()))
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
    if isinstance(stages, (list, tuple)) and len(stages) >= 1:
        os.environ["VLLM_ASCEND_SHRINK_AWARE_STAGES"] = ",".join(
            str(int(stage)) for stage in stages)
    os.environ["VLLM_ASCEND_SHRINK_AWARE_SURVIVOR_POLICY"] = str(
        runtime.get("survivor_selection_policy", "manual"))
    os.environ["VLLM_ASCEND_SHRINK_AWARE_INTERMEDIATE_RANKS"] = ",".join(
        str(rank) for rank in plan.get("intermediate_survivor_ranks", []))
    os.environ["VLLM_ASCEND_SHRINK_AWARE_FINAL_RANKS"] = ",".join(
        str(rank) for rank in plan.get("final_survivor_ranks", []))
    stage_ranks = plan.get("stage_survivor_ranks")
    if isinstance(stage_ranks, list):
        os.environ["VLLM_ASCEND_SHRINK_AWARE_STAGE_RANKS"] = json.dumps(
            stage_ranks)
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
    kv_plan = meta_info.get("shrink_aware_kv_plan", {})
    if not isinstance(kv_plan, dict):
        kv_plan = {}
    kv_cap = runtime.get("kv_cap", kv_plan.get("kv_cap"))
    if kv_cap is not None:
        try:
            os.environ["VLLM_ASCEND_MODE1_PARITY_CURRENT_KV_TOKENS"] = str(
                int(float(kv_cap)))
        except (TypeError, ValueError):
            logger.warning("Invalid shrink-aware kv_cap in meta_info: %r",
                           kv_cap)
    selected_floor = runtime.get(
        "selected_floor",
        kv_plan.get("selected_floor", runtime.get("floor",
                                                  kv_plan.get("floor"))))
    if selected_floor is not None:
        try:
            os.environ["VLLM_ASCEND_MODE1_PARITY_CURRENT_FLOOR"] = str(
                int(float(selected_floor)))
        except (TypeError, ValueError):
            logger.warning(
                "Invalid shrink-aware selected_floor in meta_info: %r",
                selected_floor)


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
        mla_impl = _resolve_mla_impl(getattr(self_attn, "mla_attn", None))
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


def _stream_rollout_weight_staging_enabled() -> bool:
    return os.getenv("VLLM_ASCEND_STREAM_ROLLOUT_WEIGHT_STAGING",
                     "1").lower() in ("1", "true", "yes", "on")


def _materialize_rollout_weight_staging(
        weights: Iterable[tuple[str, torch.Tensor]]
) -> Iterable[tuple[str, torch.Tensor]]:
    """Create rollout-owned tensors for weight reload.

    Mode=1 repeatedly reloads rollout weights while elastic groups and runtime
    views are being rebuilt. Cloning cuts loader-side aliases to caller-owned
    tensors so the rollout can release temporary NPU allocations promptly.

    The old path cloned every tensor into a list before calling load_weights,
    which can create a large transient NPU peak.  AutoWeightsLoader consumes an
    iterable once, so the default path clones lazily and lets each temporary
    tensor die as soon as its parameter has been loaded.  Keep the list path
    available for debugging via VLLM_ASCEND_STREAM_ROLLOUT_WEIGHT_STAGING=0.
    """
    def _clone_weight(weight: torch.Tensor) -> torch.Tensor:
        detached = weight.detach()
        if not detached.is_contiguous():
            # Tensor.clone can preserve strides for dense views. Normalize a
            # strided reshard result before handing it to the backend loader.
            return detached.contiguous()
        return detached.clone()

    if _stream_rollout_weight_staging_enabled():
        return ((name, _clone_weight(weight)) for name, weight in weights)
    return [(name, _clone_weight(weight)) for name, weight in weights]


_MODE1_EXPERT_WEIGHT_RE = re.compile(
    r"(?:^|\.)layers\.(\d+)\.mlp\.experts\.(\d+)\."
    r"(gate_proj|up_proj|down_proj)\.weight$")


def _mode1_update_weights_diag_enabled() -> bool:
    return os.getenv("VLLM_ASCEND_MODE1_UPDATE_WEIGHTS_DIAG",
                     "0").lower() in ("1", "true", "yes", "on")


def _mode1_update_weights_diag(message: str, *args) -> None:
    if not _mode1_update_weights_diag_enabled():
        return
    text = message % args if args else message
    logger.info("Mode1 update_weights diag: %s", text)
    print(f"[mode1_update_weights] {text}", flush=True)


def _mode1_dist_rank() -> int:
    try:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            return int(torch.distributed.get_rank())
    except Exception:
        pass
    return int(os.getenv("RANK", "-1") or "-1")


def _mode1_safe_int(value: Any, default: int = -1) -> int:
    try:
        if isinstance(value, torch.Tensor):
            return int(value.item())
        return int(value)
    except Exception:
        return default


def _mode1_weight_tensor_summary(
        weights: Iterable[tuple[str, torch.Tensor]]) -> dict[str, Any]:
    total = 0
    tensor_count = 0
    tensor_bytes = 0
    expert_tensor_count = 0
    layer_experts: dict[int, set[int]] = {}
    projection_counts: dict[str, int] = {}
    non_expert_count = 0
    for name, weight in weights:
        total += 1
        if isinstance(weight, torch.Tensor):
            tensor_count += 1
            try:
                tensor_bytes += int(weight.numel()) * int(weight.element_size())
            except Exception:
                pass
        match = _MODE1_EXPERT_WEIGHT_RE.search(name)
        if match:
            layer_idx = int(match.group(1))
            expert_id = int(match.group(2))
            projection = match.group(3)
            expert_tensor_count += 1
            layer_experts.setdefault(layer_idx, set()).add(expert_id)
            projection_counts[projection] = (
                int(projection_counts.get(projection, 0)) + 1)
        else:
            non_expert_count += 1
    per_layer_counts = {layer: len(experts)
                        for layer, experts in layer_experts.items()}
    sample_layers = ",".join(
        f"{layer}:{count}" for layer, count in
        sorted(per_layer_counts.items())[:8])
    return {
        "total": total,
        "tensor_count": tensor_count,
        "tensor_bytes": tensor_bytes,
        "expert_tensor_count": expert_tensor_count,
        "non_expert_count": non_expert_count,
        "expert_layers": len(layer_experts),
        "max_experts_per_layer":
            max(per_layer_counts.values()) if per_layer_counts else 0,
        "min_experts_per_layer":
            min(per_layer_counts.values()) if per_layer_counts else 0,
        "sample_layers": sample_layers,
        "projection_counts": projection_counts,
    }


def _mode1_count_nonnegative_map_slots(value: Any) -> tuple[int, int]:
    if value is None:
        return -1, -1
    try:
        if isinstance(value, torch.Tensor):
            if value.device.type != "cpu":
                cpu_values = value.detach().cpu()
            else:
                cpu_values = value.detach()
            return int((cpu_values >= 0).sum().item()), int(cpu_values.numel())
        values = list(value)
        return sum(1 for item in values if int(item) >= 0), len(values)
    except Exception:
        return -1, -1


def _mode1_model_moe_state_summary(model: Any) -> dict[str, Any]:
    modules = 0
    local_counts: dict[int, int] = {}
    active_counts: dict[int, int] = {}
    loaded_counts: dict[int, int] = {}
    capacity_counts: dict[int, int] = {}
    redundant_counts: dict[int, int] = {}
    expert_map_counts: dict[str, int] = {}
    loaded_map_counts: dict[str, int] = {}
    runtime_weight_modules = 0
    runtime_buffer_modules = 0
    samples: list[str] = []
    for module in model.modules():
        if not (hasattr(module, "expert_map")
                or hasattr(module, "loaded_expert_map")
                or hasattr(module, "local_num_experts")):
            continue
        modules += 1
        layer_idx = _mode1_safe_int(getattr(module, "layer_idx", -1))
        local_num = _mode1_safe_int(getattr(module, "local_num_experts", -1))
        active_num = _mode1_safe_int(
            getattr(module, "active_local_num_experts", -1))
        loaded_num = _mode1_safe_int(
            getattr(module, "loaded_local_num_experts", -1))
        capacity = _mode1_safe_int(
            getattr(module, "loaded_weight_capacity", -1))
        redundant = _mode1_safe_int(
            getattr(module, "global_redundant_expert_num", -1))
        for store, value in ((local_counts, local_num),
                             (active_counts, active_num),
                             (loaded_counts, loaded_num),
                             (capacity_counts, capacity),
                             (redundant_counts, redundant)):
            store[value] = int(store.get(value, 0)) + 1
        expert_nonneg, expert_len = _mode1_count_nonnegative_map_slots(
            getattr(module, "expert_map", None))
        loaded_nonneg, loaded_len = _mode1_count_nonnegative_map_slots(
            getattr(module, "loaded_expert_map", None))
        expert_key = f"{expert_nonneg}/{expert_len}"
        loaded_key = f"{loaded_nonneg}/{loaded_len}"
        expert_map_counts[expert_key] = int(
            expert_map_counts.get(expert_key, 0)) + 1
        loaded_map_counts[loaded_key] = int(
            loaded_map_counts.get(loaded_key, 0)) + 1
        if getattr(module, "runtime_w13_weight", None) is not None:
            runtime_weight_modules += 1
        if getattr(module, "runtime_w13_buffer", None) is not None:
            runtime_buffer_modules += 1
        if len(samples) < 4:
            samples.append(
                f"layer={layer_idx} local={local_num} active={active_num} "
                f"loaded={loaded_num} cap={capacity} redundant={redundant} "
                f"map={expert_key} loaded_map={loaded_key}")
    return {
        "modules": modules,
        "local_counts": local_counts,
        "active_counts": active_counts,
        "loaded_counts": loaded_counts,
        "capacity_counts": capacity_counts,
        "redundant_counts": redundant_counts,
        "expert_map_counts": expert_map_counts,
        "loaded_map_counts": loaded_map_counts,
        "runtime_weight_modules": runtime_weight_modules,
        "runtime_buffer_modules": runtime_buffer_modules,
        "samples": "; ".join(samples),
    }


def _mode1_expert_loader_stats_summary(model: Any) -> dict[str, Any]:
    modules = 0
    calls = 0
    total_s = 0.0
    max_s = 0.0
    errors = 0
    loaded_bytes = 0
    shard_counts: dict[str, int] = {}
    samples: list[str] = []
    for module in model.modules():
        stats = getattr(module, "_mode1_weight_loader_stats", None)
        if not isinstance(stats, dict):
            continue
        modules += 1
        module_calls = int(stats.get("calls", 0))
        module_total_s = float(stats.get("total_s", 0.0))
        module_max_s = float(stats.get("max_s", 0.0))
        calls += module_calls
        total_s += module_total_s
        max_s = max(max_s, module_max_s)
        errors += int(stats.get("errors", 0))
        loaded_bytes += int(stats.get("loaded_bytes", 0))
        if os.getenv("VLLM_ASCEND_MODE1_UPDATE_WEIGHTS_SHARD_COUNTS",
                     "0").lower() in ("1", "true", "yes", "on"):
            for shard, count in dict(stats.get("shard_counts", {})).items():
                shard_counts[str(shard)] = (
                    int(shard_counts.get(str(shard), 0)) + int(count))
        if len(samples) < 4 and module_calls > 0:
            layer_idx = _mode1_safe_int(getattr(module, "layer_idx", -1))
            samples.append(
                f"layer={layer_idx} calls={module_calls} "
                f"total_s={module_total_s:.3f} max_s={module_max_s:.3f}")
    return {
        "modules": modules,
        "calls": calls,
        "total_s": total_s,
        "max_s": max_s,
        "errors": errors,
        "loaded_bytes": loaded_bytes,
        "shard_counts": shard_counts,
        "samples": "; ".join(samples),
    }


def _load_model_num_experts(model_path: str) -> int:
    from verl.utils.moe_config import get_routed_expert_count

    config_path = Path(model_path) / "config.json"
    if not config_path.exists():
        return 0
    try:
        config = json.loads(config_path.read_text())
    except Exception:
        return 0
    return get_routed_expert_count(config)

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
        sleep_level_override = os.environ.get("VLLM_ROLLOUT_SLEEP_LEVEL")
        if sleep_level_override:
            self.sleep_level = int(sleep_level_override)
        if self.sleep_level not in (1, 2):
            raise ValueError(
                f"unsupported VLLM_ROLLOUT_SLEEP_LEVEL={self.sleep_level}")

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
            if isinstance(cudagraph_capture_sizes, (ListConfig, list, tuple)):
                normalized_capture_sizes = [
                    int(size) for size in cudagraph_capture_sizes
                ]
                if (not normalized_capture_sizes
                        or any(size <= 0 for size in normalized_capture_sizes)
                        or len(set(normalized_capture_sizes)) != len(
                            normalized_capture_sizes)):
                    raise ValueError(
                        "cudagraph_capture_sizes must contain unique positive "
                        f"integers, got {cudagraph_capture_sizes}")
                requested_mode = config.get("cudagraph_mode") or "PIECEWISE"
                if isinstance(requested_mode, CUDAGraphMode):
                    cudagraph_mode = requested_mode
                elif isinstance(requested_mode, str):
                    try:
                        cudagraph_mode = CUDAGraphMode[requested_mode.upper()]
                    except KeyError as error:
                        choices = ", ".join(mode.name for mode in CUDAGraphMode)
                        raise ValueError(
                            "actor_rollout_ref.rollout.cudagraph_mode must be "
                            f"one of {choices}, got {requested_mode!r}") from error
                else:
                    raise TypeError(
                        "actor_rollout_ref.rollout.cudagraph_mode must be a "
                        f"string or CUDAGraphMode, got {type(requested_mode).__name__}")
                if cudagraph_mode not in (CUDAGraphMode.PIECEWISE,
                                          CUDAGraphMode.FULL_DECODE_ONLY):
                    raise ValueError(
                        "VERL's pinned vLLM-Ascend 0.11 rollout supports only "
                        "PIECEWISE and FULL_DECODE_ONLY ACLGraph modes, got "
                        f"{cudagraph_mode.name}")
                compilation_config["compilation_config"] = CompilationConfig(
                    level=CompilationLevel.PIECEWISE,
                    cudagraph_mode=cudagraph_mode,
                    cudagraph_capture_sizes=normalized_capture_sizes,
                )
                logger.info(
                    "Native ACLGraph rollout configured: mode=%s capture_sizes=%s",
                    cudagraph_mode.name,
                    normalized_capture_sizes,
                )
            else:
                raise TypeError(
                    "cudagraph_capture_sizes must be a list or tuple, got "
                    f"{type(cudagraph_capture_sizes).__name__}: "
                    f"{cudagraph_capture_sizes}")

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
        self.native_sleep_mode = _env_flag(
            "VLLM_ROLLOUT_NATIVE_SLEEP_MODE", "0")
        if self.native_sleep_mode and elastic_execution_mode != 0:
            raise RuntimeError(
                "0.11 native CaMem graph sleep is currently validated only "
                "for Full16 Vanilla. AdaFloor shrink uses the existing manual "
                "weight/KV lifecycle until topology-aware CaMem remapping is "
                "validated.")
        cold_init_env_prev = os.environ.get(
            "VLLM_ASCEND_MODE1_IN_COLD_ENGINE_INIT")
        cold_init_enabled = (
            elastic_execution_mode == 1
            and os.environ.get("VLLM_ASCEND_MODE1_COLD_INIT_KV_TOKENS",
                               "").strip())
        if cold_init_enabled:
            os.environ["VLLM_ASCEND_MODE1_IN_COLD_ENGINE_INIT"] = "1"
            logger.warning(
                "Mode1 cold engine init enabled: cold_init_kv_tokens=%s. "
                "The initial engine KV cache will be rebuilt by rollout "
                "resume with the real per-step floor budget.",
                os.environ.get("VLLM_ASCEND_MODE1_COLD_INIT_KV_TOKENS"),
            )
        torchair_enabled = (
            not config.enforce_eager
            and _env_flag("VLLM_ENABLE_GRAPH_MODE", "0"))
        torchair_graph_config = ({
            "enabled": True,
            "use_cached_graph": False,
            "graph_batch_sizes_init": True,
            "enable_multistream_mla": False,
            "enable_zero_tp_to_ep": True,
            "enable_view_optimize": False,
            "enable_kv_nz": False,
            "enable_frozen_parameter": False,
        } if torchair_enabled else {
            "enabled": False
        })
        self.inference_engine = LLM(
            model=model_path,
            enable_sleep_mode=(config.free_cache_engine
                               and self.native_sleep_mode),
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
            async_scheduling=config.async_scheduling,
            enable_chunked_prefill=config.enable_chunked_prefill,
            enable_prefix_caching=config.enable_prefix_caching,
            trust_remote_code=trust_remote_code,
            seed=config.get("seed", 0),
            additional_config={
                "torchair_graph_config": torchair_graph_config,
                "ascend_scheduler_config": {
                    "enabled": True,
                    "enable_chunked_prefill": config.enable_chunked_prefill,
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
        if cold_init_enabled:
            if cold_init_env_prev is None:
                os.environ.pop("VLLM_ASCEND_MODE1_IN_COLD_ENGINE_INIT", None)
            else:
                os.environ[
                    "VLLM_ASCEND_MODE1_IN_COLD_ENGINE_INIT"] = cold_init_env_prev

        # Offload vllm model to reduce peak memory usage
        self.model_runner = self.inference_engine.llm_engine.model_executor.driver_worker.worker.model_runner
        self.model = self.model_runner.get_model()
        self._needs_rollout_aclgraph_recapture = False
        self.kv_cache_configs = None
        self._preserve_initial_hf_weights = _env_flag(
            "VLLM_ASCEND_DEEPSEEK_PRESERVE_INITIAL_HF_WEIGHTS", "0")
        self._compare_online_sync_to_hf = _env_flag(
            "VLLM_ASCEND_MODE1_WEIGHT_LOADER_DIAG_COMPARE_HF", "0")
        self._initial_hf_weight_samples = None
        self._initial_hf_layer1_w13 = None
        self._initial_hf_layer1_expert_map = None
        self._online_sync_layer1_weights = {}
        self._initial_hf_weight_compare_done = False
        if self._preserve_initial_hf_weights and self._compare_online_sync_to_hf:
            raise RuntimeError(
                "DeepSeek HF preservation and online-sync comparison are "
                "mutually exclusive")
        if self._preserve_initial_hf_weights:
            architectures = set(
                getattr(model_hf_config, "architectures", None) or [])
            if "DeepseekV2ForCausalLM" not in architectures:
                raise RuntimeError(
                    "Initial HF rollout-weight preservation is restricted to "
                    "DeepseekV2ForCausalLM diagnostics")
            if load_format == "dummy":
                raise RuntimeError(
                    "Initial HF rollout-weight preservation requires a "
                    "non-dummy load format")
            logger.warning(
                "Preserving initial DeepSeek HF rollout weights for EP "
                "forward diagnostics")
        if self._compare_online_sync_to_hf:
            architectures = set(
                getattr(model_hf_config, "architectures", None) or [])
            if "DeepseekV2ForCausalLM" not in architectures:
                raise RuntimeError(
                    "HF online-sync comparison is restricted to "
                    "DeepseekV2ForCausalLM diagnostics")
            if load_format == "dummy":
                raise RuntimeError(
                    "HF online-sync comparison requires a non-dummy load format")
            self._initial_hf_weight_samples = {
                name: (tuple(param.shape), param.dtype,
                       _sample_weight_tensor(param))
                for name, param in self.model.named_parameters()
            }
            layer1_experts = self.model.model.layers[1].mlp.experts
            layer1_w13 = dict(self.model.named_parameters()).get(
                "model.layers.1.mlp.experts.w13_weight")
            layer1_expert_map = getattr(layer1_experts, "expert_map", None)
            if layer1_w13 is None or not isinstance(layer1_expert_map,
                                                    torch.Tensor):
                raise RuntimeError(
                    "DeepSeek HF comparison could not capture layer-1 routed experts")
            self._initial_hf_layer1_w13 = (
                layer1_w13.detach().cpu().contiguous().clone())
            self._initial_hf_layer1_expert_map = tuple(
                int(value)
                for value in layer1_expert_map.detach().cpu().tolist())
            logger.warning(
                "Captured initial DeepSeek HF weight samples: rank=%s params=%s",
                _mode1_dist_rank(),
                len(self._initial_hf_weight_samples),
            )
        self.cpu_model = {}
        self.gpu_buffer_formats = {}
        self.gpu_buffers = None
        self._native_sleep_weight_ptr_signature = None
        self._native_sleep_kv_ptr_signature = None
        if self.native_sleep_mode:
            if self._preserve_initial_hf_weights:
                raise RuntimeError(
                    "native rollout sleep is incompatible with preserved HF weights")
            self._native_sleep_weight_ptr_signature = (
                _model_weight_pointer_signature(self.model))
            self._native_sleep_kv_ptr_signature = _kv_cache_pointer_signature(
                self.model_runner.kv_caches)
            self.inference_engine.reset_prefix_cache()
            self.inference_engine.sleep(level=self.sleep_level)
            logger.warning(
                "0.11 optimized rollout uses native CaMem sleep level=%s; "
                "parameter addresses are guarded before ACLGraph reuse",
                self.sleep_level,
            )
        else:
            for name, params in self.model.named_parameters():
                self.cpu_model[name] = torch.empty_like(params, device="cpu")
                self.gpu_buffer_formats[name] = _get_npu_buffer_format(params)
            self.free_cache_engine()
            if not self._preserve_initial_hf_weights:
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

    def _mode1_adaptive_kv_resize_enabled(self) -> bool:
        value = os.environ.get("VLLM_ASCEND_MODE1_ADAPTIVE_KV_RESIZE", "0")
        return value.lower() in ("1", "true", "yes", "on")

    def _target_mode1_kv_tokens_from_meta(self, meta_info: dict) -> int | None:
        if not isinstance(meta_info, dict):
            return None
        runtime = meta_info.get("shrink_aware_runtime", {})
        if not isinstance(runtime, dict):
            runtime = {}
        kv_plan = meta_info.get("shrink_aware_kv_plan", {})
        if not isinstance(kv_plan, dict):
            kv_plan = {}
        kv_cap = runtime.get("kv_cap", kv_plan.get("kv_cap"))
        if kv_cap is None:
            return None
        try:
            kv_tokens = int(float(kv_cap))
        except (TypeError, ValueError):
            logger.warning("Invalid mode1 adaptive KV token target: %r", kv_cap)
            return None
        return kv_tokens if kv_tokens > 0 else None

    def _target_mode1_floor_from_meta(self, meta_info: dict) -> int | None:
        if not isinstance(meta_info, dict):
            return None
        runtime = meta_info.get("shrink_aware_runtime", {})
        if not isinstance(runtime, dict):
            runtime = {}
        kv_plan = meta_info.get("shrink_aware_kv_plan", {})
        if not isinstance(kv_plan, dict):
            kv_plan = {}
        floor = runtime.get("selected_floor",
                            kv_plan.get("selected_floor",
                                        runtime.get("floor", kv_plan.get("floor"))))
        if floor is None:
            return None
        try:
            floor_int = int(float(floor))
        except (TypeError, ValueError):
            logger.warning("Invalid mode1 adaptive floor target: %r", floor)
            return None
        return floor_int if floor_int > 0 else None

    def _maybe_resize_mode1_kv_cache_from_meta(self, meta_info: dict) -> None:
        if not self._mode1_adaptive_kv_resize_enabled():
            return
        if os.environ.get("VLLM_USE_V1") != "1":
            return
        if int(getattr(envs_ascend, "VLLM_ASCEND_ELASTIC_EXECUTION_MODE", 0)) != 1:
            return
        target_tokens = self._target_mode1_kv_tokens_from_meta(meta_info)
        if target_tokens is None:
            return
        target_floor = self._target_mode1_floor_from_meta(meta_info)
        engine_core = self.inference_engine.llm_engine.engine_core.engine_core
        resize_fn = getattr(engine_core, "resize_kv_cache_for_mode1_step",
                            None)
        if not callable(resize_fn):
            logger.warning(
                "Mode1 adaptive KV resize requested but engine core has no resize method"
            )
            return
        changed = resize_fn(target_tokens, target_floor)
        if changed:
            self.inference_engine.llm_engine.reset_prefix_cache()
            logger.info(
                "Mode1 adaptive KV resize applied: target_tokens=%s target_floor=%s",
                target_tokens,
                target_floor,
            )

    def _maybe_log_mode1_comm_cache_state(self, tag: str) -> None:
        value = os.environ.get("VLLM_ASCEND_MODE1_COMM_CACHE_STATE_LOG", "0")
        if value.strip().lower() not in ("1", "true", "yes", "on"):
            return
        if os.environ.get("VLLM_USE_V1") != "1":
            return
        try:
            engine_core = self.inference_engine.llm_engine.engine_core.engine_core
            engine_core.collective_rpc("mode1_log_comm_cache_state",
                                       args=(tag, ))
        except Exception:
            logger.exception(
                "Failed to log mode1 comm cache state from rollout: tag=%s",
                tag,
            )

    def _maybe_dump_mode1_pre_generate_moe_state(self, tag: str) -> None:
        value = os.environ.get("VLLM_ASCEND_MODE1_PRE_GENERATE_STATE_DUMP",
                               "0")
        if value.strip().lower() not in ("1", "true", "yes", "on"):
            return
        if os.environ.get("VLLM_USE_V1") != "1":
            return
        try:
            engine_core = self.inference_engine.llm_engine.engine_core.engine_core
            engine_core.collective_rpc("mode1_dump_pre_generate_moe_state",
                                       args=(tag, ))
        except Exception:
            logger.exception(
                "Failed to dump mode1 pre-generate MoE state from rollout: tag=%s",
                tag,
            )
            if os.environ.get(
                    "VLLM_ASCEND_MODE1_PRE_GENERATE_STATE_STRICT",
                    "1").strip().lower() in ("1", "true", "yes", "on"):
                raise

    def init_cache_engine(self):
        if os.environ['VLLM_USE_V1'] == '1':
            worker = self.inference_engine.llm_engine.model_executor.driver_worker.worker
            if not worker.model_runner.kv_caches:
                if (os.environ.get("VLLM_ASCEND_ELASTIC_EXECUTION_MODE", "0") == "1"
                        and os.environ.get(
                            "VLLM_ASCEND_MODE1_PRE_RESUME_KV_CLEANUP",
                            "1").strip().lower()
                        in ("1", "true", "yes", "on")):
                    engine_core = (
                        self.inference_engine.llm_engine.engine_core.engine_core)
                    cleanup_fn = getattr(
                        engine_core,
                        "prepare_kv_cache_resume_for_mode1_step",
                        None,
                    )
                    if callable(cleanup_fn):
                        raw_floor = os.environ.get(
                            "VLLM_ASCEND_MODE1_PARITY_CURRENT_FLOOR")
                        target_floor = None
                        if raw_floor:
                            try:
                                target_floor = int(float(raw_floor.strip()))
                            except (TypeError, ValueError):
                                logger.warning(
                                    "Ignore invalid mode1 current floor before "
                                    "KV resume cleanup: %r",
                                    raw_floor,
                                )
                        cleanup_fn(target_floor)
                    else:
                        logger.warning(
                            "Mode1 pre-resume KV cleanup requested but engine "
                            "core has no cleanup method")
                # v1 Use Explicit Initialization Method
                self.inference_engine.llm_engine.engine_core.engine_core.model_executor.initialize_from_config(
                    self.inference_engine.llm_engine.engine_core.engine_core.kv_cache_configs)
                self.inference_engine.llm_engine.reset_prefix_cache()
        else:
            if self.inference_engine.llm_engine.model_executor.driver_worker.worker.cache_engine is None:
                self.inference_engine.llm_engine.model_executor.driver_worker.worker._init_cache_engine()

    def onload_model_weights(self):
        """
        Advantages over moving the model recursively:
        1) Avoids CPU to NPU data transfer entirely, leveraging pre-allocated NPU buffers
        instead of copying data from CPU tensors.
        2) Eliminates the recursive traversal of submodules inherent in .cuda(),
        which can be particularly slow for deeply nested model architectures.
        """
        if self._preserve_initial_hf_weights:
            logger.warning(
                "Retaining initial DeepSeek HF rollout weights during onload")
            return
        _log_custom_mode1_rollout_memory("before_onload_model_weights")
        self.gpu_buffers = {}
        formatted_buffers = 0
        npu_device = torch.device("npu")
        for name, param in self.model.named_parameters():
            target_format = self.gpu_buffer_formats.get(name)
            if target_format is not None and str(target_format) not in (
                    "ND", "0"):
                try:
                    import torch_npu  # type: ignore
                    # Avoid the transient double allocation from
                    # empty_like(ND) -> npu_format_cast(FRACTAL_NZ).  With
                    # floor4 KV already resident that temporary copy is enough
                    # to OOM late expert buffers.
                    gpu_buffer = torch_npu.empty_with_format(
                        tuple(param.shape),
                        dtype=param.dtype,
                        layout=param.layout,
                        device='npu',
                        pin_memory=False,
                        acl_format=int(target_format),
                    )
                    formatted_buffers += 1
                except Exception:
                    logger.exception(
                        "Failed to allocate formatted rollout GPU buffer for %s format=%s",
                        name,
                        target_format,
                    )
                    gpu_buffer = torch.empty_like(param, device=npu_device)
            else:
                gpu_buffer = torch.empty_like(param, device=npu_device)
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
        if self.native_sleep_mode:
            raise RuntimeError(
                "manual offload_model_weights called while native rollout sleep is active")
        if self._preserve_initial_hf_weights:
            logger.warning(
                "Retaining initial DeepSeek HF rollout weights during offload")
            return
        if _rollout_elastic_aclgraph_enabled(self.model_runner):
            _invalidate_rollout_aclgraphs(self.model_runner,
                                          "offload_model_weights")
            self._needs_rollout_aclgraph_recapture = True
        _log_custom_mode1_rollout_memory("before_offload_model_weights")
        _log_custom_mode1_rollout_state(self, "before_offload_model_weights")
        free_before_transient_cleanup = (
            int(torch.npu.mem_get_info()[0])
            if self.elastic_execution_mode == 1 and is_npu_available else 0
        )
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
        # Planned preparation may materialize low-floor expert slots even when
        # the live decode finishes before an actual group shrink. In that case
        # there is no full-world restore callback to release the temporary
        # aliases. Clear them at the rollout-to-training boundary, which every
        # mode1 step reaches regardless of whether a shrink RPC occurred.
        transient_released_layers = 0
        if self.elastic_execution_mode == 1:
            for module in self.model.modules():
                release_transient = getattr(
                    module, "release_mode1_full_world_transient_state", None)
                if not callable(release_transient):
                    continue
                try:
                    release_transient()
                    transient_released_layers += 1
                except Exception:
                    logger.exception(
                        "Failed to release mode1 full-world transient state "
                        "at the rollout-to-training boundary"
                    )
                    raise
        for name, params in self.model.named_parameters():
            params.data = self.cpu_model[name]

        for i in range(self.model.model.start_layer, self.model.model.end_layer):
            self_attn = getattr(self.model.model.layers[i], "self_attn", None)
            mla = _resolve_mla_impl(getattr(self_attn, "mla_attn", None))
            if mla is None:
                continue
            if hasattr(mla, "w_kc"):
                mla.w_kc = None
                mla.w_vc = None
            if hasattr(mla, "W_UV"):
                mla.W_UV = None
                mla.W_UK_T = None
        self.gpu_buffers = None
        gc.collect()
        torch.npu.empty_cache()
        if self.elastic_execution_mode == 1 and is_npu_available:
            torch.npu.synchronize()
            free_after_transient_cleanup = int(torch.npu.mem_get_info()[0])
            logger.warning(
                "Mode1 training-boundary full-world transient cleanup: "
                "rank=%s step=%s epoch=%s modules=%s "
                "free_before_bytes=%s free_after_bytes=%s freed_bytes=%s",
                _mode1_dist_rank(),
                os.getenv("VLLM_ASCEND_MODE1_CURRENT_UPDATE_STEP", "-1"),
                os.getenv("VLLM_ASCEND_MODE1_CURRENT_UPDATE_EPOCH", "-1"),
                transient_released_layers,
                free_before_transient_cleanup,
                free_after_transient_cleanup,
                free_after_transient_cleanup - free_before_transient_cleanup,
            )
        if invalidated_layers > 0:
            logger.info(
                "Rollout offload invalidated lossless runtime state: modules=%s runtime_modules=%s",
                invalidated_layers,
                invalidated_runtime_layers,
            )
        _log_custom_mode1_rollout_state(self, "after_offload_model_weights")
        _log_custom_mode1_rollout_memory("after_offload_model_weights")

    def free_cache_engine(self):
        if self.native_sleep_mode:
            raise RuntimeError(
                "manual free_cache_engine called while native rollout sleep is active")
        if _rollout_elastic_aclgraph_enabled(self.model_runner):
            _invalidate_rollout_aclgraphs(self.model_runner,
                                          "free_cache_engine")
            self._needs_rollout_aclgraph_recapture = True
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
        for i in range(self.model.model.start_layer, self.model.model.end_layer):
            self_attn = getattr(self.model.model.layers[i], "self_attn", None)
            mla_impl = _resolve_mla_impl(
                getattr(self_attn, "mla_attn", None))
            if mla_impl is None:
                continue
            if hasattr(mla_impl, "key_cache"):
                mla_impl.key_cache = None
                mla_impl.value_cache = None

        release_training_runtime = _env_flag(
            "VLLM_ASCEND_MODE1_RELEASE_MOE_RUNTIME_BEFORE_TRAINING", "0")
        runtime_stats: dict[str, int] = {}
        free_before_cleanup = 0
        if release_training_runtime and is_npu_available:
            free_before_cleanup = int(torch.npu.mem_get_info()[0])
            torch.npu.synchronize()
            from vllm_ascend.ops.moe.moe_comm_method import (
                release_moe_comm_method_runtime_state,
            )

            runtime_stats = release_moe_comm_method_runtime_state()

        gc.collect()
        torch.npu.empty_cache()
        if release_training_runtime and is_npu_available:
            torch.npu.synchronize()
            free_after_cleanup = int(torch.npu.mem_get_info()[0])
            logger.warning(
                "Mode1 training-boundary MoE runtime cleanup: "
                "free_before_bytes=%s free_after_bytes=%s freed_bytes=%s "
                "methods=%s topologies=%s method_attrs=%s "
                "dispatcher_attrs=%s prepare_attrs=%s tensors=%s "
                "tensor_bytes=%s",
                free_before_cleanup,
                free_after_cleanup,
                free_after_cleanup - free_before_cleanup,
                runtime_stats.get("methods", 0),
                runtime_stats.get("topologies", 0),
                runtime_stats.get("method_attrs", 0),
                runtime_stats.get("dispatcher_attrs", 0),
                runtime_stats.get("prepare_attrs", 0),
                runtime_stats.get("tensors", 0),
                runtime_stats.get("tensor_bytes", 0),
            )
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
        step_timeline_log = (
            os.environ.get("VLLM_ASCEND_MODE1_STEP_TIMELINE_LOG", "1").lower()
            in ("1", "true", "yes", "on"))

        def _timeline(message: str, *args) -> None:
            if not step_timeline_log:
                return
            text = message % args if args else message
            logger.info("Mode1 step timeline: %s", text)
            print(f"[mode1_timeline] {text}", flush=True)

        worker_generate_start = time.perf_counter()
        worker_rank = -1
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            worker_rank = int(torch.distributed.get_rank())
        _timeline(
            "rollout_worker_generate_entry rank=%s step=%s epoch=%s",
            worker_rank,
            prompts.meta_info.get("global_steps", -1),
            prompts.meta_info.get("epoch", -1),
        )
        _sync_shrink_aware_env_from_meta(prompts.meta_info)
        _timeline(
            "rollout_worker_after_env_sync rank=%s step=%s epoch=%s total_s=%.3f",
            worker_rank,
            prompts.meta_info.get("global_steps", -1),
            prompts.meta_info.get("epoch", -1),
            time.perf_counter() - worker_generate_start,
        )
        idx = prompts.batch["input_ids"]  # (bs, prompt_length)
        # left-padded attention_mask
        attention_mask = prompts.batch["attention_mask"]
        position_ids = prompts.batch["position_ids"]

        # used to construct attention_mask
        eos_token_id = prompts.meta_info["eos_token_id"]

        batch_size = idx.size(0)

        non_tensor_batch = prompts.non_tensor_batch
        if "raw_prompt_ids" not in non_tensor_batch:
            preprocess_start = time.perf_counter()
            non_tensor_batch["raw_prompt_ids"] = np.array(
                [_pre_process_inputs(self.pad_token_id, idx[i]) for i in range(batch_size)], dtype=object
            )
            _timeline(
                "rollout_worker_raw_prompt_preprocess_done rank=%s step=%s "
                "epoch=%s elapsed_s=%.3f total_s=%.3f",
                worker_rank,
                prompts.meta_info.get("global_steps", -1),
                prompts.meta_info.get("epoch", -1),
                time.perf_counter() - preprocess_start,
                time.perf_counter() - worker_generate_start,
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
            vllm_input_start = time.perf_counter()
            vllm_inputs = [
                {"prompt_token_ids": raw_prompt_ids} for raw_prompt_ids in non_tensor_batch.pop("raw_prompt_ids")
            ]
            _timeline(
                "rollout_worker_vllm_inputs_built rank=%s step=%s epoch=%s "
                "batch_size=%s elapsed_s=%.3f total_s=%.3f",
                worker_rank,
                prompts.meta_info.get("global_steps", -1),
                prompts.meta_info.get("epoch", -1),
                batch_size,
                time.perf_counter() - vllm_input_start,
                time.perf_counter() - worker_generate_start,
            )

        for input_data in vllm_inputs:
            # Ensure token IDs are lists or numpy arrays
            if not isinstance(input_data["prompt_token_ids"], list | np.ndarray):
                raise TypeError(
                    f"prompt_token_ids must be a list or numpy array, got {type(input_data['prompt_token_ids'])}"
                )

            input_data["prompt_token_ids"] = list(input_data["prompt_token_ids"])

        paired_request_seeds = _env_flag("VERL_PAIRED_REQUEST_SAMPLING_SEEDS")
        rollout_prompt_hashes: list[str] = []
        rollout_request_seeds: list[int] = []
        if paired_request_seeds:
            sample_indices = non_tensor_batch.get("rollout_sample_index")
            if sample_indices is None or len(sample_indices) != batch_size:
                sample_count = 0 if sample_indices is None else len(sample_indices)
                raise RuntimeError(
                    "paired request sampling requires one rollout_sample_index "
                    f"per request, got {sample_count} for batch_size={batch_size}"
                )
            base_seed = int(self.config.get("seed", 0))
            for input_data, sample_index in zip(
                vllm_inputs, sample_indices, strict=True
            ):
                prompt_hash, request_seed = derive_request_seed(
                    base_seed,
                    input_data["prompt_token_ids"],
                    int(sample_index),
                )
                rollout_prompt_hashes.append(prompt_hash)
                rollout_request_seeds.append(request_seed)

        fixed_work_trace_path = os.environ.get("VERL_FIXED_WORK_REPLAY_TRACE")
        fixed_work_source_lengths: list[int] = []
        fixed_work_target_lengths: list[int] = []
        fixed_work_source_row_ordinals: list[int] = []
        fixed_work_source_steps: list[int] = []
        fixed_work_trace_sha256: str | None = None

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
        step_tail_validate_caps = os.environ.get(
            "VERL_ELASTIC_TAIL_VALIDATE_LEVEL_TOKENS_BY_STEP")
        if step_tail_validate_caps:
            try:
                current_step = int(prompts.meta_info.get("global_steps", -1))
            except (TypeError, ValueError):
                current_step = -1
            step_caps = [
                item.strip() for item in step_tail_validate_caps.split(";")
            ]
            if current_step >= 1 and current_step <= len(step_caps):
                tail_validate_caps = step_caps[current_step - 1]
            elif step_caps:
                tail_validate_caps = step_caps[-1]
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

        if fixed_work_trace_path:
            if not paired_request_seeds:
                raise RuntimeError(
                    "fixed-work replay requires paired request sampling seeds"
                )
            row_ordinals = non_tensor_batch.get("fixed_work_replay_row_ordinal")
            if row_ordinals is None or len(row_ordinals) != batch_size:
                ordinal_count = 0 if row_ordinals is None else len(row_ordinals)
                raise RuntimeError(
                    "fixed-work replay requires one row ordinal per request, "
                    f"got {ordinal_count} for batch_size={batch_size}"
                )
            prompt_occurrences = non_tensor_batch.get(
                "prompt_occurrence_ordinal"
            )
            if prompt_occurrences is None or len(prompt_occurrences) != batch_size:
                occurrence_count = (
                    0 if prompt_occurrences is None else len(prompt_occurrences)
                )
                raise RuntimeError(
                    "fixed-work replay requires one stable prompt occurrence "
                    f"per request, got {occurrence_count} for "
                    f"batch_size={batch_size}"
                )
            try:
                replay_step = int(prompts.meta_info.get("global_steps", -1))
            except (TypeError, ValueError) as error:
                raise RuntimeError("fixed-work replay requires an integer step") from error
            replay = load_fixed_work_replay(fixed_work_trace_path)
            replay_cap = replay.step_cap(replay_step)
            effective_runtime_cap = int(
                kwargs.get("max_tokens", self.config.response_length)
            )
            if _env_flag("VERL_FIXED_WORK_REPLAY_REQUIRE_PLAN_CAP"):
                if effective_runtime_cap != replay_cap:
                    raise RuntimeError(
                        "fixed-work replay cap differs from the AdaFloor plan, "
                        f"step={replay_step} trace_cap={replay_cap} "
                        f"runtime_cap={effective_runtime_cap}"
                    )
            for (
                row_ordinal,
                prompt_occurrence,
                prompt_hash,
                sample_index,
                request_seed,
            ) in zip(
                row_ordinals,
                prompt_occurrences,
                rollout_prompt_hashes,
                sample_indices,
                rollout_request_seeds,
                strict=True,
            ):
                ordinal = int(row_ordinal)
                occurrence = int(prompt_occurrence)
                target = replay.target_for_occurrence(
                    occurrence,
                    int(sample_index),
                    prompt_hash,
                    int(request_seed),
                )
                source = replay.source_length_for_occurrence(
                    occurrence,
                    int(sample_index),
                    prompt_hash,
                    int(request_seed),
                )
                source_row = replay.source_row_for_occurrence(
                    occurrence,
                    int(sample_index),
                    prompt_hash,
                    int(request_seed),
                )
                source_step = replay.source_step_for_occurrence(
                    occurrence,
                    int(sample_index),
                    prompt_hash,
                    int(request_seed),
                )
                if (
                    _env_flag("VERL_FIXED_WORK_REPLAY_REQUIRE_PLAN_CAP")
                    and source_step != replay_step
                ):
                    raise RuntimeError(
                        "fixed AdaFloor request moved across source plan steps, "
                        f"occurrence={occurrence} source_step={source_step} "
                        f"runtime_step={replay_step}"
                    )
                if target > effective_runtime_cap:
                    raise RuntimeError(
                        "fixed-work target exceeds the active response cap, "
                        f"step={replay_step} row={ordinal} target={target} "
                        f"runtime_cap={effective_runtime_cap}"
                    )
                fixed_work_source_lengths.append(source)
                fixed_work_target_lengths.append(target)
                fixed_work_source_row_ordinals.append(source_row)
                fixed_work_source_steps.append(source_step)
            fixed_work_trace_sha256 = replay.trace_sha256
            logger.info(
                "Fixed-work replay prepared: rank=%s step=%s requests=%s "
                "source_tokens=%s target_tokens=%s target_min=%s target_max=%s "
                "trace_sha256=%s",
                worker_rank,
                replay_step,
                batch_size,
                sum(fixed_work_source_lengths),
                sum(fixed_work_target_lengths),
                min(fixed_work_target_lengths),
                max(fixed_work_target_lengths),
                fixed_work_trace_sha256,
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
        self._maybe_log_mode1_comm_cache_state("rollout_step_start")
        resize_start = time.perf_counter()
        _timeline(
            "rollout_worker_resize_start rank=%s step=%s epoch=%s "
            "target_floor=%s target_kv=%s total_s=%.3f",
            worker_rank,
            prompts.meta_info.get("global_steps", -1),
            prompts.meta_info.get("epoch", -1),
            self._target_mode1_floor_from_meta(prompts.meta_info),
            self._target_mode1_kv_tokens_from_meta(prompts.meta_info),
            time.perf_counter() - worker_generate_start,
        )
        self._maybe_resize_mode1_kv_cache_from_meta(prompts.meta_info)
        _timeline(
            "rollout_worker_resize_done rank=%s step=%s epoch=%s "
            "elapsed_s=%.3f total_s=%.3f",
            worker_rank,
            prompts.meta_info.get("global_steps", -1),
            prompts.meta_info.get("epoch", -1),
            time.perf_counter() - resize_start,
            time.perf_counter() - worker_generate_start,
        )
        self._maybe_log_mode1_comm_cache_state("rollout_step_after_resize")
        if self._needs_rollout_aclgraph_recapture:
            _recapture_rollout_aclgraphs(
                self.model_runner, "generate_after_mode1_kv_resize")
            self._needs_rollout_aclgraph_recapture = False
        with self.update_sampling_params(**kwargs):
            infer_start = time.perf_counter()
            pre_generate_tag = (
                f"epoch={prompts.meta_info.get('epoch', -1)}:"
                f"step={prompts.meta_info.get('global_steps', -1)}:"
                f"floor={self._target_mode1_floor_from_meta(prompts.meta_info)}:"
                f"kv={self._target_mode1_kv_tokens_from_meta(prompts.meta_info)}")
            self._maybe_dump_mode1_pre_generate_moe_state(pre_generate_tag)
            _timeline(
                "rollout_worker_infer_start rank=%s step=%s epoch=%s "
                "batch_size=%s max_tokens=%s total_s=%.3f",
                worker_rank,
                prompts.meta_info.get("global_steps", -1),
                prompts.meta_info.get("epoch", -1),
                batch_size,
                getattr(self.sampling_params, "max_tokens", None),
                time.perf_counter() - worker_generate_start,
            )
            request_sampling_params = self.sampling_params
            if paired_request_seeds:
                request_sampling_params = []
                for request_index, request_seed in enumerate(rollout_request_seeds):
                    params = self.sampling_params.clone()
                    params.seed = request_seed
                    if fixed_work_target_lengths:
                        target = fixed_work_target_lengths[request_index]
                        params.min_tokens = target
                        params.max_tokens = target
                        params.ignore_eos = True
                        params.stop = []
                        params.stop_token_ids = []
                    request_sampling_params.append(params)
            outputs = self.inference_engine.generate(
                prompts=vllm_inputs,  # because we have already convert it to prompt token id
                sampling_params=request_sampling_params,
                lora_request=lora_requests,
                use_tqdm=True,
            )
            _timeline(
                "rollout_worker_infer_done rank=%s step=%s epoch=%s "
                "elapsed_s=%.3f outputs=%s total_s=%.3f",
                worker_rank,
                prompts.meta_info.get("global_steps", -1),
                prompts.meta_info.get("epoch", -1),
                time.perf_counter() - infer_start,
                len(outputs),
                time.perf_counter() - worker_generate_start,
            )

            # TODO(sgm): disable logprob when recompute_log_prob is enable
            # if n = 1: (bs, response_length) ; if n > 1: (bs * n, response_length)

            response = []
            decoded_response_lengths = []
            response_finish_reasons = []
            rollout_log_probs = []
            request_ids = []
            rollout_ranks = []
            local_rank = -1
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                local_rank = int(torch.distributed.get_rank())
            for output in outputs:
                req_id = str(getattr(output, "request_id", ""))
                for sample_id in range(len(output.outputs)):
                    sample_output = output.outputs[sample_id]
                    response_ids = sample_output.token_ids
                    response.append(response_ids)
                    decoded_response_lengths.append(len(response_ids))
                    finish_reason = getattr(sample_output, "finish_reason", None)
                    response_finish_reasons.append(
                        None if finish_reason is None else str(finish_reason)
                    )
                    request_ids.append(req_id)
                    rollout_ranks.append(local_rank)
                    if self.config.calculate_log_probs:
                        curr_log_prob = []
                        for i, logprob in enumerate(output.outputs[sample_id].logprobs):
                            curr_log_prob.append(logprob[response_ids[i]].logprob)
                        rollout_log_probs.append(curr_log_prob)

            if paired_request_seeds:
                if len(outputs) != batch_size:
                    raise RuntimeError(
                        "paired request sampling requires one ordered output per "
                        f"input, got outputs={len(outputs)} batch_size={batch_size}"
                    )
                non_tensor_batch["rollout_prompt_hash"] = np.array(
                    rollout_prompt_hashes, dtype=object
                )
                non_tensor_batch["rollout_request_seed"] = np.array(
                    rollout_request_seeds, dtype=np.int64
                )

            response = pad_2d_list_to_length(response, self.pad_token_id, max_length=self.config.response_length).to(
                idx.device
            )
            if len(decoded_response_lengths) != response.shape[0]:
                raise RuntimeError(
                    "vLLM returned inconsistent decoded response lengths: "
                    f"lengths={len(decoded_response_lengths)} responses={response.shape[0]}"
                )
            if fixed_work_target_lengths:
                mismatches = [
                    (index, actual, target)
                    for index, (actual, target) in enumerate(
                        zip(
                            decoded_response_lengths,
                            fixed_work_target_lengths,
                            strict=True,
                        )
                    )
                    if actual != target
                ]
                if mismatches:
                    raise RuntimeError(
                        "fixed-work replay did not generate the exact target lengths, "
                        f"first_mismatches={mismatches[:8]}"
                    )
                unexpected_finishes = [
                    (index, reason)
                    for index, reason in enumerate(response_finish_reasons)
                    if reason != "length"
                ]
                if unexpected_finishes:
                    raise RuntimeError(
                        "fixed-work replay ended without the max-token boundary, "
                        f"first_mismatches={unexpected_finishes[:8]}"
                    )
                non_tensor_batch["fixed_work_replay_source_length"] = np.array(
                    fixed_work_source_lengths, dtype=np.int64
                )
                non_tensor_batch["fixed_work_replay_target_length"] = np.array(
                    fixed_work_target_lengths, dtype=np.int64
                )
                non_tensor_batch["fixed_work_replay_trace_sha256"] = np.array(
                    [fixed_work_trace_sha256] * batch_size, dtype=object
                )
                non_tensor_batch[
                    "fixed_work_replay_source_row_ordinal"
                ] = np.array(fixed_work_source_row_ordinals, dtype=np.int64)
                non_tensor_batch["fixed_work_replay_source_step"] = np.array(
                    fixed_work_source_steps, dtype=np.int64
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
                non_tensor_batch["decoded_response_length"] = np.array(
                    decoded_response_lengths, dtype=np.int64
                )
                non_tensor_batch["response_finish_reason"] = np.array(
                    response_finish_reasons, dtype=object
                )
            else:
                logger.warning(
                    "Skip attaching request_id/rollout_rank due to length mismatch: "
                    "request_ids=%d, responses=%d",
                    len(request_ids),
                    num_responses,
                )

        self._maybe_log_mode1_comm_cache_state("rollout_step_end")
        _timeline(
            "rollout_worker_generate_done rank=%s step=%s epoch=%s elapsed_s=%.3f",
            worker_rank,
            prompts.meta_info.get("global_steps", -1),
            prompts.meta_info.get("epoch", -1),
            time.perf_counter() - worker_generate_start,
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
        decoded_lengths = torch.tensor(
            decoded_response_lengths, device=response.device, dtype=torch.long
        )
        response_attention_mask = (
            torch.arange(response_length, device=response.device).unsqueeze(0)
            < decoded_lengths.unsqueeze(1)
        ).to(attention_mask.dtype)
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

        if self.native_sleep_mode:
            self.inference_engine.wake_up(tags=tags)
            current_signature = _model_weight_pointer_signature(self.model)
            expected_signature = self._native_sleep_weight_ptr_signature
            if (expected_signature is not None
                    and current_signature != expected_signature):
                logger.warning(
                    "Native sleep remapped rollout parameter addresses; "
                    "invalidate ACLGraph and recapture after weight refresh")
                _invalidate_rollout_aclgraphs(
                    self.model_runner, "native_sleep_pointer_change")
                self._needs_rollout_aclgraph_recapture = True
                self._native_sleep_weight_ptr_signature = current_signature
            if "kv_cache" in tags:
                current_kv_signature = _kv_cache_pointer_signature(
                    self.model_runner.kv_caches)
                expected_kv_signature = self._native_sleep_kv_ptr_signature
                if (expected_kv_signature is not None
                        and current_kv_signature != expected_kv_signature):
                    logger.warning(
                        "Native sleep remapped rollout KV-cache addresses; "
                        "invalidate ACLGraph before generation")
                    _invalidate_rollout_aclgraphs(
                        self.model_runner, "native_sleep_kv_pointer_change")
                    self._needs_rollout_aclgraph_recapture = True
                self._native_sleep_kv_ptr_signature = current_kv_signature
            return

        if "weights" in tags:
            self.onload_model_weights()
        elif "kv_cache" in tags:
            self.init_cache_engine()

    async def release(self, preserve_kv_cache: bool = False):
        """Release weights and kv cache in GPU memory."""
        if not self.config.free_cache_engine:
            return


        if self.native_sleep_mode:
            if preserve_kv_cache:
                raise RuntimeError(
                    "native rollout sleep does not yet support AdaFloor's "
                    "same-floor KV preservation path")
            self.inference_engine.reset_prefix_cache()
            self.inference_engine.sleep(level=self.sleep_level)
            return

        if preserve_kv_cache:
            try:
                self.inference_engine.llm_engine.reset_prefix_cache()
            except Exception:
                logger.exception(
                    "Failed to reset prefix cache while preserving KV cache")
            logger.info(
                "Preserve rollout KV cache storage across same-floor step; "
                "only rollout weights will be offloaded")
        else:
            self.free_cache_engine()
        self.offload_model_weights()

    async def update_weights(self, weights: Generator[tuple[str, torch.Tensor], None, None], **kwargs):
        """Update the weights of the rollout model.

        Args:
            weights: A generator that yields the name of the weight tensor and the tensor itself.
        """
        if self._preserve_initial_hf_weights:
            logger.warning(
                "Skipping online MCore rollout-weight sync for DeepSeek HF "
                "EP forward diagnostics")
            return
        update_total_start_t = time.perf_counter()
        rank = _mode1_dist_rank()
        step_id = os.getenv("VLLM_ASCEND_MODE1_CURRENT_UPDATE_STEP", "-1")
        epoch_id = os.getenv("VLLM_ASCEND_MODE1_CURRENT_UPDATE_EPOCH", "-1")
        skipped = 0
        skipped_names: list[str] = []
        peft_config, base_sync_done = kwargs.get("peft_config", None), kwargs.get("base_sync_done", False)
        need_materialized_weights = (
            _mode1_update_weights_diag_enabled()
            or bool(peft_config and base_sync_done))
        filter_start_t = time.perf_counter()
        filter_elapsed_s = 0.0

        def _nonempty_weight_stream(
                source: Iterable[tuple[str, torch.Tensor]]
        ) -> Generator[tuple[str, torch.Tensor], None, None]:
            nonlocal skipped, filter_elapsed_s
            try:
                for name, weight in source:
                    if isinstance(weight, torch.Tensor) and weight.numel() == 0:
                        skipped += 1
                        if len(skipped_names) < 8:
                            skipped_names.append(name)
                        continue
                    yield name, weight
            finally:
                filter_elapsed_s = time.perf_counter() - filter_start_t

        if need_materialized_weights:
            filtered_weights = list(_nonempty_weight_stream(weights))
            weights = filtered_weights
        else:
            filtered_weights = None
            weights = _nonempty_weight_stream(weights)

        input_summary = (
            _mode1_weight_tensor_summary(weights)
            if _mode1_update_weights_diag_enabled() else {})
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
            from verl.utils.vllm.patch import (
                abort_vllm_moe_model_weight_loader,
                finalize_vllm_moe_model_weight_loader,
                patch_vllm_moe_model_weight_loader,
            )

            model = self.inference_engine.llm_engine.model_executor.driver_worker.worker.model_runner.get_model()
            patch_start_t = time.perf_counter()
            patch_vllm_moe_model_weight_loader(model)
            patch_elapsed_s = time.perf_counter() - patch_start_t
        if peft_config and base_sync_done:
            model = self.inference_engine.llm_engine.model_executor.driver_worker.worker.model_runner.get_model()
            patch_elapsed_s = 0.0
        weight_ptr_signature_before = _model_weight_pointer_signature(model)
        if (self.elastic_execution_mode in (1, 2)
                and self.elastic_init_redundancy_expert > 0
                and not getattr(self, "_lossless_weight_update_sync_logged", False)):
            logger.info(
                "Elastic execution mode=%s enables rollout.update_weights with redundant experts.",
                self.elastic_execution_mode)
            self._lossless_weight_update_sync_logged = True
        _log_custom_mode1_rollout_memory("before_update_weights_load")
        _log_custom_mode1_rollout_state(self, "before_update_weights_load")
        trim_elapsed_s = 0.0
        if (self.elastic_execution_mode == 1
                and int(os.getenv("VLLM_ASCEND_ELASTIC_MIN_COMPUTE_GROUP_SIZE",
                                  "0") or "0") <= 2):
            trim_start_t = time.perf_counter()
            try:
                torch.npu.synchronize()
                gc.collect()
                torch.npu.empty_cache()
            except Exception:
                pass
            trim_elapsed_s = time.perf_counter() - trim_start_t
            _log_custom_mode1_rollout_memory(
                "before_update_weights_load_after_trim")
        if _mode1_update_weights_diag_enabled():
            model_summary = _mode1_model_moe_state_summary(model)
            _mode1_update_weights_diag(
                "before_load rank=%s step=%s epoch=%s total_weights=%s "
                "tensor_weights=%s skipped_empty=%s tensor_bytes=%s "
                "expert_tensors=%s non_expert_tensors=%s expert_layers=%s "
                "experts_per_layer_minmax=%s/%s sample_layers=%s "
                "projection_counts=%s filter_s=%.3f patch_s=%.3f trim_s=%.3f "
                "moe_modules=%s local_counts=%s active_counts=%s "
                "loaded_counts=%s capacity_counts=%s redundant_counts=%s "
                "expert_map_counts=%s loaded_map_counts=%s runtime_weight_modules=%s "
                "runtime_buffer_modules=%s samples=%s",
                rank,
                step_id,
                epoch_id,
                input_summary.get("total"),
                input_summary.get("tensor_count"),
                skipped,
                input_summary.get("tensor_bytes"),
                input_summary.get("expert_tensor_count"),
                input_summary.get("non_expert_count"),
                input_summary.get("expert_layers"),
                input_summary.get("min_experts_per_layer"),
                input_summary.get("max_experts_per_layer"),
                input_summary.get("sample_layers"),
                input_summary.get("projection_counts"),
                filter_elapsed_s,
                patch_elapsed_s,
                trim_elapsed_s,
                model_summary.get("modules"),
                model_summary.get("local_counts"),
                model_summary.get("active_counts"),
                model_summary.get("loaded_counts"),
                model_summary.get("capacity_counts"),
                model_summary.get("redundant_counts"),
                model_summary.get("expert_map_counts"),
                model_summary.get("loaded_map_counts"),
                model_summary.get("runtime_weight_modules"),
                model_summary.get("runtime_buffer_modules"),
                model_summary.get("samples"),
            )
        materialize_start_t = time.perf_counter()
        staged_weights = _materialize_rollout_weight_staging(weights)
        if (self._compare_online_sync_to_hf
                and not self._initial_hf_weight_compare_done):
            def _capture_layer1_weights(source):
                for weight_name, weight in source:
                    match = _MODE1_EXPERT_WEIGHT_RE.search(weight_name)
                    if (match is not None and int(match.group(1)) == 1
                            and match.group(3) in ("gate_proj", "up_proj")):
                        self._online_sync_layer1_weights[weight_name] = (
                            weight.detach().cpu().contiguous().clone())
                    yield weight_name, weight

            staged_weights = _capture_layer1_weights(staged_weights)
        materialize_elapsed_s = time.perf_counter() - materialize_start_t
        load_start_t = time.perf_counter()
        load_call_start_t = time.perf_counter()
        try:
            model.load_weights(staged_weights)
            if not (peft_config and base_sync_done):
                finalize_vllm_moe_model_weight_loader(model)
        except Exception:
            if not (peft_config and base_sync_done):
                abort_vllm_moe_model_weight_loader(model)
            raise
        model_config = self.model_runner.vllm_config.model_config
        mla_refreshed = _refresh_mla_derived_weights(
            model,
            model_config.dtype,
            require_complete=bool(model_config.use_mla),
        )
        if mla_refreshed > 0:
            logger.info("Refreshed derived MLA weights for %s layers",
                        mla_refreshed)
        weight_ptr_signature_after = _model_weight_pointer_signature(model)
        reuse_native_graph = (
            self.native_sleep_mode
            and _env_flag("VLLM_ROLLOUT_REUSE_ACLGRAPH_AFTER_WEIGHT_UPDATE", "1")
            and weight_ptr_signature_before == weight_ptr_signature_after
            and (self._native_sleep_weight_ptr_signature is None
                 or self._native_sleep_weight_ptr_signature
                 == weight_ptr_signature_after))
        if reuse_native_graph:
            logger.info(
                "Reuse rollout ACLGraph after weight update: native CaMem "
                "preserved all parameter addresses")
            self._native_sleep_weight_ptr_signature = weight_ptr_signature_after
        else:
            if self.native_sleep_mode:
                logger.warning(
                    "Rollout weight loader changed captured addresses; use "
                    "safe ACLGraph invalidate/recapture fallback")
                self._native_sleep_weight_ptr_signature = weight_ptr_signature_after
            _invalidate_rollout_aclgraphs(self.model_runner,
                                          "update_weights_after_finalize")
            if _rollout_elastic_aclgraph_enabled(self.model_runner):
                self._needs_rollout_aclgraph_recapture = True
        load_call_elapsed_s = time.perf_counter() - load_call_start_t
        load_sync_start_t = time.perf_counter()
        load_sync_elapsed_s = 0.0
        try:
            torch.npu.synchronize()
            load_sync_elapsed_s = time.perf_counter() - load_sync_start_t
        except Exception:
            pass
        if (self._compare_online_sync_to_hf
                and not self._initial_hf_weight_compare_done):
            comparison = _compare_weight_samples(
                model, self._initial_hf_weight_samples or {})
            logger.warning(
                "DeepSeek online-sync HF comparison: rank=%s total=%s "
                "matched=%s mismatched=%s category_totals=%s "
                "category_mismatches=%s missing=%s sample_errors=%s "
                "mismatch_samples=%s",
                rank,
                comparison["total"],
                comparison["matched"],
                comparison["mismatched"],
                comparison["category_totals"],
                comparison["category_mismatches"],
                comparison["missing"],
                comparison["sample_errors"],
                comparison["mismatch_samples"],
            )
            routed_comparison = _compare_layer1_routed_w13(
                model,
                self._initial_hf_layer1_w13,
                self._initial_hf_layer1_expert_map,
                self._online_sync_layer1_weights,
            )
            logger.warning(
                "DeepSeek layer-1 routed w13 comparison: rank=%s result=%s",
                rank,
                routed_comparison,
            )
            comparison_failures = _weight_comparison_failures(
                comparison, routed_comparison)
            if comparison_failures:
                raise RuntimeError(
                    "DeepSeek online-sync HF comparison failed on "
                    f"rank {rank}: {'; '.join(comparison_failures)}")
            logger.warning(
                "DeepSeek online-sync HF comparison PASS: rank=%s total=%s "
                "routed_streams=%s",
                rank,
                comparison["total"],
                routed_comparison["expected_stream_count"],
            )
            self._initial_hf_weight_compare_done = True
            self._initial_hf_weight_samples = None
            self._initial_hf_layer1_w13 = None
            self._initial_hf_layer1_expert_map = None
            self._online_sync_layer1_weights = {}
        load_elapsed_s = time.perf_counter() - load_start_t
        if _mode1_update_weights_diag_enabled():
            loader_stats = _mode1_expert_loader_stats_summary(model)
            _mode1_update_weights_diag(
                "after_load rank=%s step=%s epoch=%s materialize_s=%.3f "
                "model_load_s=%.3f model_load_call_s=%.3f "
                "model_load_sync_s=%.3f expert_loader_modules=%s "
                "expert_loader_calls=%s expert_loader_total_s=%.3f "
                "expert_loader_max_s=%.3f expert_loader_errors=%s "
                "expert_loader_loaded_bytes=%s shard_counts=%s samples=%s",
                rank,
                step_id,
                epoch_id,
                materialize_elapsed_s,
                load_elapsed_s,
                load_call_elapsed_s,
                load_sync_elapsed_s,
                loader_stats.get("modules"),
                loader_stats.get("calls"),
                loader_stats.get("total_s"),
                loader_stats.get("max_s"),
                loader_stats.get("errors"),
                loader_stats.get("loaded_bytes"),
                loader_stats.get("shard_counts"),
                loader_stats.get("samples"),
            )
        refresh_gpu_buffer_aliases = getattr(self, "_refresh_gpu_buffer_aliases",
                                             None)
        refresh_start_t = time.perf_counter()
        if callable(refresh_gpu_buffer_aliases):
            refresh_gpu_buffer_aliases()
        refresh_elapsed_s = time.perf_counter() - refresh_start_t
        staged_weights = None
        weights = []
        filtered_weights = []
        cleanup_start_t = time.perf_counter()
        gc.collect()
        torch.npu.empty_cache()
        cleanup_elapsed_s = time.perf_counter() - cleanup_start_t
        if skipped > 0 and not getattr(self, "_empty_weight_update_warned",
                                       False):
            logger.warning(
                "Skip %s empty rollout weight shards during update_weights. sample_names=%s",
                skipped,
                skipped_names,
            )
            self._empty_weight_update_warned = True
        _log_custom_mode1_rollout_state(self, "after_update_weights_load")
        _log_custom_mode1_rollout_memory("after_update_weights_load")
        if _mode1_update_weights_diag_enabled():
            after_summary = _mode1_model_moe_state_summary(model)
            _mode1_update_weights_diag(
                "done rank=%s step=%s epoch=%s total_s=%.3f refresh_s=%.3f "
                "cleanup_s=%.3f local_counts=%s active_counts=%s "
                "loaded_counts=%s capacity_counts=%s runtime_weight_modules=%s "
                "runtime_buffer_modules=%s samples=%s",
                rank,
                step_id,
                epoch_id,
                time.perf_counter() - update_total_start_t,
                refresh_elapsed_s,
                cleanup_elapsed_s,
                after_summary.get("local_counts"),
                after_summary.get("active_counts"),
                after_summary.get("loaded_counts"),
                after_summary.get("capacity_counts"),
                after_summary.get("runtime_weight_modules"),
                after_summary.get("runtime_buffer_modules"),
                after_summary.get("samples"),
            )
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
        if envs_ascend.VLLM_ASCEND_ELASTIC_ACLGRAPH:
            raise RuntimeError(
                "Elastic ACLGraph currently supports only the synchronous "
                "vLLMRollout; vLLMAsyncRollout has no post-KV-resize "
                "recapture boundary")
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
            from verl.utils.vllm.patch import (
                abort_vllm_moe_model_weight_loader,
                finalize_vllm_moe_model_weight_loader,
                patch_vllm_moe_model_weight_loader,
            )

            model = self.inference_engine.worker.model_runner.model
            patch_vllm_moe_model_weight_loader(model)
            staged_weights = _materialize_rollout_weight_staging(weights)
            try:
                model.load_weights(staged_weights)
                finalize_vllm_moe_model_weight_loader(model)
            except Exception:
                abort_vllm_moe_model_weight_loader(model)
                raise
            model_config = self.vllm_config.model_config
            _refresh_mla_derived_weights(
                model,
                model_config.dtype,
                require_complete=bool(model_config.use_mla),
            )
            staged_weights = None
            gc.collect()
            torch.npu.empty_cache()

    def generate_sequences(self, prompts: DataProto) -> DataProto:
        """Batch generate sequences in sync mode."""
        raise NotImplementedError

    # ==================== server mode public methods ====================

    def get_zeromq_address(self):
        return self.address
