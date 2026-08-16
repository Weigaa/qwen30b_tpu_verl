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
import contextlib
import copy
import gc
import getpass
import inspect
import logging
import os
import pickle
import socket
import time
from contextlib import contextmanager
from dataclasses import asdict
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
from vllm.config.lora import LoRAConfig
from vllm.lora.request import LoRARequest
from vllm.utils.moe_stats import moe_stats
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

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def _env_flag(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).lower() in ("1", "true", "yes", "on")


def _maybe_profile_rollout_generate():
    """Profile only the vLLM generate window for one rollout rank.

    This is intentionally env-gated and rank-gated so we can compare new/old
    eager traces without profiling actor/ref/update phases or all 16 workers.
    """
    if not _env_flag("VLLM_ROLLOUT_TORCH_NPU_PROFILE"):
        return contextlib.nullcontext()

    rank = -1
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        rank = int(torch.distributed.get_rank())
    target_rank = int(os.environ.get("VLLM_ROLLOUT_TORCH_NPU_PROFILE_RANK", "0"))
    if rank != target_rank:
        return contextlib.nullcontext()

    try:
        import torch_npu  # type: ignore
    except Exception:
        logger.warning("torch_npu is unavailable; skip rollout generate profiling")
        return contextlib.nullcontext()

    out_dir = os.environ.get(
        "VLLM_ROLLOUT_TORCH_NPU_PROFILE_DIR",
        f"./result/profiler/rollout_generate_rank_{rank}",
    )
    experimental_config = torch_npu.profiler._ExperimentalConfig(
        export_type=[torch_npu.profiler.ExportType.Text],
        profiler_level=torch_npu.profiler.ProfilerLevel.Level0,
        msprof_tx=False,
        aic_metrics=torch_npu.profiler.AiCMetrics.AiCoreNone,
        l2_cache=False,
        op_attr=False,
        data_simplification=False,
        record_op_args=False,
        gc_detect_threshold=None,
    )
    logger.info("Enable rollout generate torch_npu profiler: rank=%s dir=%s", rank, out_dir)
    return torch_npu.profiler.profile(
        activities=[
            torch_npu.profiler.ProfilerActivity.CPU,
            torch_npu.profiler.ProfilerActivity.NPU,
        ],
        schedule=torch_npu.profiler.schedule(
            wait=int(os.environ.get("VLLM_ROLLOUT_TORCH_NPU_PROFILE_WAIT", "0")),
            warmup=int(os.environ.get("VLLM_ROLLOUT_TORCH_NPU_PROFILE_WARMUP", "0")),
            active=int(os.environ.get("VLLM_ROLLOUT_TORCH_NPU_PROFILE_ACTIVE", "1")),
            repeat=int(os.environ.get("VLLM_ROLLOUT_TORCH_NPU_PROFILE_REPEAT", "1")),
        ),
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(out_dir),
        record_shapes=_env_flag("VLLM_ROLLOUT_TORCH_NPU_PROFILE_RECORD_SHAPES"),
        profile_memory=_env_flag("VLLM_ROLLOUT_TORCH_NPU_PROFILE_MEMORY"),
        with_stack=_env_flag("VLLM_ROLLOUT_TORCH_NPU_PROFILE_STACK"),
        with_modules=False,
        with_flops=False,
        experimental_config=experimental_config,
    )


_ROLLOUT_STAGE_TIMING_BUFFER: dict[str, float] = {}


@contextmanager
def _rollout_stage_timer(stage: str):
    enabled = os.environ.get("VLLM_ROLLOUT_STAGE_TIMING", "0").lower() in (
        "1",
        "true",
        "yes",
        "on",
    )
    start = time.time()
    try:
        yield
    finally:
        duration = time.time() - start
        _ROLLOUT_STAGE_TIMING_BUFFER[stage] = _ROLLOUT_STAGE_TIMING_BUFFER.get(stage, 0.0) + duration
        if enabled:
            logger.warning("rollout_stage_timing stage=%s duration_s=%.6f", stage, duration)


def _rollout_memory_probe(stage: str) -> None:
    if os.environ.get("VLLM_ROLLOUT_PHASE_MEMORY_LOG", "0").lower() not in (
        "1",
        "true",
        "yes",
        "on",
    ):
        return
    if not is_npu_available:
        return
    try:
        torch.npu.synchronize()
    except Exception:
        pass
    logger.info(
        "rollout_memory_probe stage=%s allocated_gb=%.3f reserved_gb=%.3f "
        "max_allocated_gb=%.3f max_reserved_gb=%.3f",
        stage,
        torch.npu.memory_allocated() / (1024**3),
        torch.npu.memory_reserved() / (1024**3),
        torch.npu.max_memory_allocated() / (1024**3),
        torch.npu.max_memory_reserved() / (1024**3),
    )


# ignore redundant logs
import warnings
from numba.core.errors import NumbaPendingDeprecationWarning
warnings.filterwarnings("ignore", category=NumbaPendingDeprecationWarning)
logging.getLogger("torch._dynamo").setLevel(logging.CRITICAL)

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


def _resolve_rollout_sleep_level(default_level: int, layered_summon: bool) -> int:
    """Allow RL rollout to override vLLM's generic sleep policy.

    vllm-ascend defaults NPU sleep mode to level 1 because generic serving must
    preserve weights across sleep/wake. In RL rollout we immediately reload all
    model weights before generation, so level 2 can avoid expensive weight
    D2H/H2D backup while still preserving model buffers.
    """
    if layered_summon:
        return 1

    override = os.environ.get("VLLM_ROLLOUT_SLEEP_LEVEL")
    if override is None or override == "":
        return default_level

    try:
        sleep_level = int(override)
    except ValueError:
        logger.warning(
            "Ignoring invalid VLLM_ROLLOUT_SLEEP_LEVEL=%r; using %s",
            override,
            default_level,
        )
        return default_level

    if sleep_level not in (1, 2):
        logger.warning(
            "Ignoring unsupported VLLM_ROLLOUT_SLEEP_LEVEL=%s; using %s",
            sleep_level,
            default_level,
        )
        return default_level

    logger.info(
        "Using rollout sleep level %s (default=%s) for full-weight-reload RL rollout",
        sleep_level,
        default_level,
    )
    return sleep_level


@contextmanager
def _maybe_use_rollout_weight_pool(enabled: bool):
    """Allocate reload-time post-processed weights in vLLM sleep's weight pool."""
    use_pool = os.environ.get("VLLM_ROLLOUT_CAMEM_WEIGHT_RELOAD", "1").lower() in (
        "1",
        "true",
        "yes",
        "on",
    )
    if not enabled or not use_pool:
        yield
        return

    try:
        from vllm_ascend.device_allocator.camem import CaMemAllocator
    except Exception as exc:
        logger.warning(
            "Cannot use CaMem weight pool during rollout weight reload: %s",
            exc,
        )
        yield
        return

    allocator = CaMemAllocator.get_instance()
    with allocator.use_memory_pool(tag="weights"):
        yield


def _moe_load_layout_shapes(model: torch.nn.Module) -> dict[str, tuple[int, ...]]:
    """Return original MoE parameter shapes needed by vLLM load_weights().

    Ascend split-MoE post-processes unquantized weights from loader layout
    [E, 2I, H] / [E, H, I] into execution layout [E, H, 2I] / [E, I, H].
    The old verl phase switch drops parameter storage between trainer and
    rollout. When that storage is re-allocated before the next load, it must
    use the loader layout or FusedMoE.weight_loader derives the wrong shard
    size and can fail with "length (1024) exceeds dimension size (768)".
    """
    shapes: dict[str, tuple[int, ...]] = {}
    for module_name, module in model.named_modules():
        w13 = getattr(module, "w13_weight", None)
        w2 = getattr(module, "w2_weight", None)
        if not isinstance(w13, torch.nn.Parameter):
            continue
        if not isinstance(w2, torch.nn.Parameter):
            continue
        if w13.ndim != 3 or w2.ndim != 3:
            continue

        w13_shape = tuple(w13.shape)
        w2_shape = tuple(w2.shape)
        if (
            w13_shape[0] == w2_shape[0]
            and w13_shape[1] == w2_shape[2]
            and w13_shape[2] == 2 * w2_shape[1]
        ):
            prefix = f"{module_name}." if module_name else ""
            shapes[f"{prefix}w13_weight"] = (
                w13_shape[0],
                w13_shape[2],
                w13_shape[1],
            )
            shapes[f"{prefix}w2_weight"] = (
                w2_shape[0],
                w2_shape[2],
                w2_shape[1],
            )
    return shapes


def _invalidate_rollout_aclgraphs(model_runner: Any, stage: str) -> int:
    """Invalidate init-time ACL graphs after RL rollout weight reloads.

    vLLM serving normally captures graphs once because model weights are
    stable.  Hybrid RL rollout reloads and post-processes weights every step,
    so any graph captured during dummy initialization can hold stale task
    groups or tensor addresses and must not be replayed for real generation.
    """
    if os.environ.get("VLLM_ROLLOUT_INVALIDATE_ACLGRAPH_AFTER_WEIGHT_UPDATE",
                      "1").lower() not in ("1", "true", "yes", "on"):
        return 0
    if not _rollout_aclgraph_enabled(model_runner):
        return 0

    try:
        from vllm_ascend.compilation.acl_graph import clear_aclgraph_caches
    except Exception as exc:
        logger.warning("Cannot invalidate rollout ACL graphs at %s: %s", stage, exc)
        return 0

    try:
        torch.npu.synchronize()
    except Exception:
        pass
    cleared = clear_aclgraph_caches(getattr(model_runner, "model", None))
    logger.warning(
        "Invalidated %s rollout ACL graph entries after %s; next graph "
        "eligible generation will recapture using current weights.",
        cleared,
        stage,
    )
    return cleared


def _rollout_aclgraph_enabled(model_runner: Any) -> bool:
    if not is_npu_available:
        return False

    vllm_config = getattr(model_runner, "vllm_config", None)
    model_config = getattr(vllm_config, "model_config", None)
    if bool(getattr(model_config, "enforce_eager", False)):
        return False

    compilation_config = getattr(vllm_config, "compilation_config", None)
    cudagraph_mode = getattr(compilation_config, "cudagraph_mode", None)
    if cudagraph_mode is None:
        return False

    mode_name = getattr(cudagraph_mode, "name", None)
    if mode_name == "NONE":
        return False
    mode_value = getattr(cudagraph_mode, "value", None)
    if mode_value == 0:
        return False

    return not str(cudagraph_mode).endswith(".NONE")


def _recapture_rollout_aclgraphs(model_runner: Any, stage: str) -> None:
    """Recapture ACL graphs through the official vLLM capture path.

    Attention task groups require the outer graph_capture context established by
    model_runner.capture_model(); recapturing lazily from an individual wrapper
    can enter attention graph-task-group code on a non-capturing stream.
    """
    if os.environ.get("VLLM_ROLLOUT_RECAPTURE_ACLGRAPH_AFTER_WEIGHT_UPDATE",
                      "1").lower() not in ("1", "true", "yes", "on"):
        return
    if not _rollout_aclgraph_enabled(model_runner):
        return

    try:
        torch.npu.synchronize()
    except Exception:
        pass
    logger.warning("Recapturing rollout ACL graphs at %s using current weights.", stage)
    model_runner.capture_model()
    logger.warning("Finished recapturing rollout ACL graphs at %s.", stage)


def _capture_rollout_graphs_after_weight_load(model_runner: Any, stage: str) -> None:
    """Capture rollout graphs after real RL weights replace dummy init weights.

    vLLM serving normally compiles/captures once during engine initialization.
    In RL rollout the engine is initialized with dummy weights, then every
    step reloads Megatron weights and runs process_weights_after_loading().
    Capturing before that post-load layout exists can bind graph fragments to
    dummy-time tensors/layouts and change generation stop behavior.
    """
    if os.environ.get("VLLM_ROLLOUT_DELAY_GRAPH_CAPTURE_UNTIL_WEIGHT_LOAD",
                      "0").lower() not in ("1", "true", "yes", "on"):
        return
    if os.environ.get("VLLM_ROLLOUT_CAPTURE_GRAPH_AFTER_WEIGHT_LOAD",
                      "0").lower() not in ("1", "true", "yes", "on"):
        logger.warning_once(
            "VLLM_ROLLOUT_DELAY_GRAPH_CAPTURE_UNTIL_WEIGHT_LOAD=1 but "
            "VLLM_ROLLOUT_CAPTURE_GRAPH_AFTER_WEIGHT_LOAD is disabled; "
            "skipping explicit post-load ACL graph capture.",
        )
        return
    if not _rollout_aclgraph_enabled(model_runner):
        return
    if getattr(model_runner, "_rollout_graphs_captured_after_weight_load",
               False):
        return

    try:
        torch.npu.synchronize()
    except Exception:
        pass
    logger.warning(
        "Capturing rollout ACL graphs at %s after real weights were loaded and "
        "post-processed.",
        stage,
    )
    model_runner.capture_model()
    setattr(model_runner, "_rollout_graphs_captured_after_weight_load", True)
    logger.warning("Finished delayed rollout ACL graph capture at %s.", stage)


class vLLMRollout(BaseRollout):
    def __init__(
        self,
        config: RolloutConfig,
        model_config: HFModelConfig,
        device_mesh: DeviceMesh,
    ):
        super().__init__(config, model_config, device_mesh)

        self.sleep_level = _resolve_rollout_sleep_level(
            VLLM_SLEEP_LEVEL,
            bool(config.layered_summon),
        )

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

        # If VLLM_DP_SIZE is configured, the DP communication domain needs to be explicitly initialized.
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

        # vLLM 0.14 permits max_num_batched_tokens < max_model_len exactly
        # when chunked prefill is enabled. Its SchedulerConfig performs the
        # authoritative validation and rejects this relation only when
        # chunking is disabled. The older VERL-side check had that condition
        # reversed, preventing memory-bounded graph profile runs.

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
            torch._dynamo.config.log_compilation_metrics = False
            compilation_config["compilation_config"] = {
                "cudagraph_capture_sizes": cudagraph_capture_sizes,
                "cudagraph_mode": "FULL",
            }

        self.dynamic_eplb = int(os.environ.get("VLLM_ENABLE_EPLB", "0")) == 1
        elastic_shrink_enabled = bool(envs_ascend.VLLM_ASCEND_ENABLE_ELASTIC_PARALLEL_SHRINK)
        legacy_runtime_enabled = bool(getattr(envs_ascend, "VLLM_ASCEND_USE_LEGACY_FUSED_MOE", False))
        alloc_conf = os.environ.get("PYTORCH_NPU_ALLOC_CONF", "")
        if "expandable_segments:True" in alloc_conf:
            # vllm_ascend CaMem allocator rejects expandable segments.
            fallback_conf = "garbage_collection_threshold:0.6,max_split_size_mb:24"
            logger.warning(
                "Replacing incompatible PYTORCH_NPU_ALLOC_CONF=%s with %s for vLLM rollout",
                alloc_conf,
                fallback_conf,
            )
            os.environ["PYTORCH_NPU_ALLOC_CONF"] = fallback_conf
        if str(os.environ.get("VLLM_ASCEND_ENABLE_NZ", "0")) != "0":
            logger.warning(
                "Replacing incompatible VLLM_ASCEND_ENABLE_NZ=%s with 0 for RL rollout",
                os.environ.get("VLLM_ASCEND_ENABLE_NZ"),
            )
            os.environ["VLLM_ASCEND_ENABLE_NZ"] = "0"
        self.manual_free_cache_engine = (
            os.environ.get("VLLM_ROLLOUT_MANUAL_FREE_CACHE_ENGINE", "0").lower()
            in ("1", "true", "yes", "on")
        )
        if self.manual_free_cache_engine:
            logger.warning(
                "Using manual rollout cache/weight offload instead of native vLLM sleep mode."
            )
        additional_config = {
            "ascend_scheduler_config": {
                "enabled": True,
                # Keep the Ascend-side scheduler behavior aligned with the
                # user-facing rollout flag instead of silently forcing the
                # chunked-prefill path on.
                "enable_chunked_prefill": config.enable_chunked_prefill,
            },
            "refresh": True,
            "dynamic_eplb": self.dynamic_eplb,
            "num_iterations_eplb_update": 400,  # gather stable workload over 400 iterations
            "gate_eplb": True,
            "num_wait_worker_iterations": 30,  # wait for 30 iterations to complete the EPLB calculation
            "npugraph_ex_config": {
                "enable": True,
                "enable_static_kernel": eval(os.environ.get("NPUGRAPH_EX_ENABLE_STATIC_KERNEL", "False"))
            }
        }
        if os.environ.get("VLLM_ASCEND_ROLLOUT_WEIGHT_PREFETCH", "0").lower() in ("1", "true", "yes", "on"):
            additional_config["weight_prefetch_config"] = {
                "enabled": True,
                "prefetch_ratio": {
                    "attn": {
                        "qkv": float(os.environ.get("VLLM_ASCEND_ROLLOUT_PREFETCH_ATTN_QKV_RATIO", "0")),
                        "o": float(os.environ.get("VLLM_ASCEND_ROLLOUT_PREFETCH_ATTN_O_RATIO", "0")),
                    },
                    "moe": {
                        "gate_up": float(os.environ.get("VLLM_ASCEND_ROLLOUT_PREFETCH_MOE_GATE_UP_RATIO", "0.8")),
                    },
                },
            }
        force_elastic_moe_policy = os.environ.get(
            "VLLM_ROLLOUT_FORCE_ELASTIC_MOE_POLICY", "0"
        ).lower() in ("1", "true", "yes", "on")
        if force_elastic_moe_policy:
            # Some of the best eager-genonly probes on qwen3 were recorded
            # while rollout still forwarded these MoE execution-policy hints
            # even with elastic shrink disabled. Keep this path opt-in so we
            # can validate whether the policy itself helps without re-enabling
            # the broader shrink machinery.
            additional_config.update({
                "elastic_moe_mode": os.environ.get(
                    "VLLM_ASCEND_ELASTIC_MOE_MODE", "lossy"
                ),
                "init_redundancy_expert": int(
                    os.environ.get("VLLM_ASCEND_INIT_REDUNDANCY_EXPERT", "0")
                ),
            })
        if elastic_shrink_enabled:
            additional_config.update({
                "elastic_execution_mode": envs_ascend.VLLM_ASCEND_ELASTIC_EXECUTION_MODE,
                "elastic_moe_mode": envs_ascend.VLLM_ASCEND_ELASTIC_MOE_MODE,
                "init_redundancy_expert": envs_ascend.VLLM_ASCEND_INIT_REDUNDANCY_EXPERT,
            })

        self.inference_engine = LLM(
            model=model_path,
            # Use vLLM/vllm_ascend native sleep mode so the allocator keeps
            # parameter storage and post-load layout state consistent.
            enable_sleep_mode=config.free_cache_engine and not self.manual_free_cache_engine,
            tensor_parallel_size=tensor_parallel_size,
            distributed_executor_backend="external_launcher",
            worker_cls=(
                "vllm_ascend.worker.worker_v1.NPUWorker"
                if elastic_shrink_enabled or legacy_runtime_enabled
                else "auto"
            ),
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
            additional_config=additional_config,
            **compilation_config,
            **self.lora_kwargs,
            **engine_kwargs,
        )

        self.model_runner = self.inference_engine.llm_engine.model_executor.driver_worker.worker.model_runner
        self.model = self.model_runner.get_model()
        # vLLM can expose padded vocab logits from tensor-parallel output heads.
        # Mask them before sampling so graph/eager produce the same valid-token
        # distribution and cannot sample pseudo tokens beyond tokenizer vocab.
        _monkey_patch_compute_logits(self.model, len(tokenizer))
        self.model_device = next(self.model.parameters()).device
        self.cpu_model: dict[str, torch.Tensor] = {}
        self.gpu_buffers: dict[str, torch.Tensor] | None = None
        self._needs_rollout_aclgraph_recapture = False
        if self.manual_free_cache_engine:
            for name, params in self.model.named_parameters():
                self.cpu_model[name] = torch.empty_like(params, device="cpu")
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
        invalidate_sampling_type = False
        try:
            if kwargs:
                for key, value in kwargs.items():
                    if hasattr(self.sampling_params, key):
                        old_value = getattr(self.sampling_params, key)
                        old_sampling_params_args[key] = old_value
                        setattr(self.sampling_params, key, value)
                        invalidate_sampling_type = invalidate_sampling_type or key in (
                            "seed",
                            "temperature",
                        )
                if invalidate_sampling_type:
                    self.sampling_params.__dict__.pop("sampling_type", None)
            yield
        finally:
            # roll back to previous sampling params
            # if len(old_sampling_params_args):
            for key, value in old_sampling_params_args.items():
                setattr(self.sampling_params, key, value)
            if invalidate_sampling_type:
                self.sampling_params.__dict__.pop("sampling_type", None)

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

        kwargs = {}
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
                logger.warning(
                    "Invalid VERL_ELASTIC_TAIL_VALIDATE_LEVEL_TOKENS=%s",
                    tail_validate_caps,
                )
                level_caps = []

            if level_caps and torch.distributed.is_available() and torch.distributed.is_initialized():
                world_size = torch.distributed.get_world_size()
                global_rank = torch.distributed.get_rank()
                max_resp_len = int(self.config.response_length)
                current_max = kwargs.get("max_tokens", getattr(self.sampling_params, "max_tokens", max_resp_len))
                current_max = int(current_max) if current_max is not None else max_resp_len

                bucket_sizes = []
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
                    "Elastic tail validation cap override: rank=%s world_size=%s bucket=%s max_tokens=%s raw_caps=%s",
                    global_rank,
                    world_size,
                    bucket_idx,
                    kwargs["max_tokens"],
                    level_caps,
                )
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

        diversify_sampling_seed = False
        per_rank_sampling_seed = None
        if (
            do_sample
            and not is_validate
            and os.environ.get("VLLM_ROLLOUT_DIVERSIFY_SAMPLING_SEED", "0").lower()
            in ("1", "true", "yes", "on")
        ):
            rank = 0
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                rank = int(torch.distributed.get_rank())
            base_seed = int(os.environ.get("VLLM_ROLLOUT_SAMPLING_BASE_SEED", "0"))
            seed_stride = int(os.environ.get("VLLM_ROLLOUT_SAMPLING_SEED_STRIDE", "104729"))
            global_step = int(prompts.meta_info.get("global_steps", 0) or 0)
            epoch = int(prompts.meta_info.get("epoch", 0) or 0)
            per_rank_sampling_seed = base_seed + rank * seed_stride + epoch * 1000003 + global_step
            diversify_sampling_seed = True

        lora_requests = None
        if self.lora_kwargs:
            lora_int_ids = list(self.inference_engine.llm_engine.list_loras())
            if len(lora_int_ids) > 0:
                lora_int_id = lora_int_ids[0]
                lora_requests = [
                    LoRARequest(lora_name=f"{lora_int_id}", lora_int_id=lora_int_id, lora_path="/simon-stub-path")
                ] * batch_size

        # users can customize different sampling_params at different run
        with self.update_sampling_params(**kwargs):
            sampling_params = self.sampling_params
            if diversify_sampling_seed:
                sampling_params = []
                assert per_rank_sampling_seed is not None
                for prompt_idx in range(len(vllm_inputs)):
                    request_sampling_params = copy.copy(self.sampling_params)
                    request_sampling_params.seed = per_rank_sampling_seed + prompt_idx
                    request_sampling_params.__dict__.pop("sampling_type", None)
                    sampling_params.append(request_sampling_params)
            use_tqdm = os.environ.get("VLLM_ROLLOUT_USE_TQDM", "0").lower() in (
                "1",
                "true",
                "yes",
                "on",
            )
            with _maybe_profile_rollout_generate() as prof:
                outputs = self.inference_engine.generate(
                    prompts=vllm_inputs,  # because we have already convert it to prompt token id
                    sampling_params=sampling_params,
                    lora_request=lora_requests,
                    use_tqdm=use_tqdm,
                )
                if prof is not None:
                    prof.step()

            # TODO(sgm): disable logprob when recompute_log_prob is enable
            # if n = 1: (bs, response_length) ; if n > 1: (bs * n, response_length)

            response = []
            rollout_log_probs = []
            request_ids = []
            rollout_ranks = []
            debug_prompt_indices = []
            debug_repeat_indices = []
            debug_generation = os.environ.get("VLLM_ROLLOUT_DEBUG_GENERATION", "0").lower() in (
                "1",
                "true",
                "yes",
                "on",
            )
            debug_records = []
            finish_reason_counts: dict[str, int] = {}
            stop_reason_counts: dict[str, int] = {}
            raw_response_lengths = []
            finish_reasons = []
            stop_reasons = []
            has_eos_flags = []
            local_rank = -1
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                local_rank = int(torch.distributed.get_rank())
            if debug_generation and diversify_sampling_seed:
                logger.warning(
                    "Rollout sampling seed override: rank=%s base_seed=%s num_prompts=%s",
                    local_rank,
                    per_rank_sampling_seed,
                    len(vllm_inputs),
                )
            eos_tokens = eos_token_id if isinstance(eos_token_id, list) else [eos_token_id]
            eos_tokens = [int(token) for token in eos_tokens if token is not None]
            eos_token_set = set(eos_tokens)
            for output in outputs:
                req_id = str(getattr(output, "request_id", ""))
                prompt_output_idx = len(request_ids)
                prompt_debug_idx = None
                repeat_debug_idx = None
                if debug_generation:
                    if "rollout_debug_prompt_idx" in non_tensor_batch:
                        prompt_debug_idx = int(non_tensor_batch["rollout_debug_prompt_idx"][prompt_output_idx])
                    if "rollout_debug_repeat_idx" in non_tensor_batch:
                        repeat_debug_idx = int(non_tensor_batch["rollout_debug_repeat_idx"][prompt_output_idx])
                for sample_id in range(len(output.outputs)):
                    sample_output = output.outputs[sample_id]
                    response_ids = sample_output.token_ids
                    response.append(response_ids)
                    request_ids.append(req_id)
                    rollout_ranks.append(local_rank)
                    if debug_generation:
                        debug_prompt_indices.append(
                            prompt_debug_idx if prompt_debug_idx is not None else prompt_output_idx
                        )
                        debug_repeat_indices.append(
                            repeat_debug_idx if repeat_debug_idx is not None else sample_id
                        )
                    if debug_generation:
                        response_len = len(response_ids)
                        raw_response_lengths.append(response_len)
                        finish_reason = str(getattr(sample_output, "finish_reason", None))
                        stop_reason = str(getattr(sample_output, "stop_reason", None))
                        finish_reasons.append(finish_reason)
                        stop_reasons.append(stop_reason)
                        finish_reason_counts[finish_reason] = finish_reason_counts.get(finish_reason, 0) + 1
                        stop_reason_counts[stop_reason] = stop_reason_counts.get(stop_reason, 0) + 1
                        has_eos = any(token in eos_token_set for token in response_ids)
                        has_eos_flags.append(has_eos)
                        if len(debug_records) < int(os.environ.get("VLLM_ROLLOUT_DEBUG_GENERATION_SAMPLES", "4")):
                            debug_records.append(
                                {
                                    "request_id": req_id,
                                    "prompt_idx": prompt_debug_idx,
                                    "repeat_idx": repeat_debug_idx,
                                    "sample_id": sample_id,
                                    "length": response_len,
                                    "finish_reason": finish_reason,
                                    "stop_reason": stop_reason,
                                    "has_eos": has_eos,
                                    "last_tokens": list(response_ids[-8:]),
                                }
                            )
                    if self.config.calculate_log_probs:
                        curr_log_prob = []
                        for i, logprob in enumerate(output.outputs[sample_id].logprobs):
                            curr_log_prob.append(logprob[response_ids[i]].logprob)
                        rollout_log_probs.append(curr_log_prob)

            if debug_generation:
                if raw_response_lengths:
                    length_array = np.asarray(raw_response_lengths, dtype=np.int64)
                    length_summary = {
                        "count": int(length_array.size),
                        "min": int(length_array.min()),
                        "mean": float(length_array.mean()),
                        "p50": int(np.percentile(length_array, 50)),
                        "p90": int(np.percentile(length_array, 90)),
                        "p95": int(np.percentile(length_array, 95)),
                        "max": int(length_array.max()),
                        "ge_10000": int(np.sum(length_array >= 10000)),
                        "ge_max_minus_1": int(np.sum(length_array >= int(self.config.response_length) - 1)),
                    }
                else:
                    length_summary = {"count": 0}
                logger.warning(
                    "Rollout generation debug: rank=%s eos_tokens=%s length_summary=%s "
                    "finish_reason_counts=%s stop_reason_counts=%s samples=%s",
                    local_rank,
                    eos_tokens,
                    length_summary,
                    finish_reason_counts,
                    stop_reason_counts,
                    debug_records,
                )

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
            if debug_generation and len(raw_response_lengths) == num_responses:
                non_tensor_batch["rollout_debug_prompt_idx"] = np.array(debug_prompt_indices, dtype=np.int64)
                non_tensor_batch["rollout_debug_repeat_idx"] = np.array(debug_repeat_indices, dtype=np.int64)
                non_tensor_batch["vllm_raw_response_len"] = np.array(raw_response_lengths, dtype=np.int64)
                non_tensor_batch["vllm_finish_reason"] = np.array(finish_reasons, dtype=object)
                non_tensor_batch["vllm_stop_reason"] = np.array(stop_reasons, dtype=object)
                non_tensor_batch["vllm_has_eos"] = np.array(has_eos_flags, dtype=np.bool_)

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
        if _ROLLOUT_STAGE_TIMING_BUFFER:
            _ROLLOUT_STAGE_TIMING_BUFFER.clear()

        return DataProto(batch=batch, non_tensor_batch=non_tensor_batch, meta_info={})

    def get_record(self):
        return moe_stats.snapshot_pattern()

    def flush_record(self):
        return moe_stats.reset_epoch()

    def init_cache_engine(self):
        """Rebuild vLLM V1 KV cache after manual free_cache_engine()."""
        with _rollout_stage_timer("init_cache_engine"):
            _rollout_memory_probe("init_cache_engine_before")
            worker = self.inference_engine.llm_engine.model_executor.driver_worker.worker
            model_runner = worker.model_runner
            if getattr(model_runner, "kv_caches", None):
                _rollout_memory_probe("init_cache_engine_skip_existing")
                return

            engine_core_client = self.inference_engine.llm_engine.engine_core
            engine_core = getattr(engine_core_client, "engine_core", None)
            kv_cache_configs = getattr(engine_core, "kv_cache_configs", None)
            if kv_cache_configs is None:
                raise RuntimeError(
                    "Cannot manually reinitialize KV cache because EngineCore "
                    "does not expose kv_cache_configs."
                )

            # qwen3's newer vLLM-Ascend model runner reinitializes attention
            # metadata builders inside initialize_kv_cache(). The old verl
            # phase-switch path only dropped KV tensors, so clear these builders
            # before asking the runner to allocate fresh cache tensors.
            if getattr(model_runner, "attn_groups", None):
                if not getattr(self, "_manual_attn_groups_reset_logged", False):
                    logger.info(
                        "Manual rollout KV rebuild clears existing attention "
                        "backend groups before initialize_from_config."
                    )
                    self._manual_attn_groups_reset_logged = True
                model_runner.attn_groups = []

            engine_core.model_executor.initialize_from_config(kv_cache_configs)
            self.inference_engine.llm_engine.reset_prefix_cache()
            _rollout_memory_probe("init_cache_engine_after")

    def onload_model_weights(self):
        """Allocate fresh NPU parameter storage before rollout weight update."""
        with _rollout_stage_timer("onload_model_weights"):
            if not self.cpu_model:
                return
            _rollout_memory_probe("onload_model_weights_before")

            moe_load_shapes = _moe_load_layout_shapes(self.model)
            if moe_load_shapes and not getattr(
                self, "_manual_moe_load_layout_logged", False
            ):
                logger.info(
                    "Manual rollout weight onload will restore %s split-MoE "
                    "parameters to loader layout before load_weights.",
                    len(moe_load_shapes),
                )
                self._manual_moe_load_layout_logged = True

            self.gpu_buffers = {}
            for name, param in self.model.named_parameters():
                load_shape = moe_load_shapes.get(name)
                if load_shape is None:
                    self.gpu_buffers[name] = torch.empty_like(
                        param, device=self.model_device
                    )
                else:
                    self.gpu_buffers[name] = torch.empty(
                        load_shape, dtype=param.dtype, device=self.model_device
                    )
            for name, param in self.model.named_parameters():
                param.data = self.gpu_buffers[name]
            _rollout_memory_probe("onload_model_weights_after")

    def offload_model_weights(self):
        """Drop rollout parameter storage; weights are fully reloaded next step."""
        with _rollout_stage_timer("offload_model_weights"):
            if not self.cpu_model:
                return
            _rollout_memory_probe("offload_model_weights_before")

            for name, param in self.model.named_parameters():
                cpu_param = self.cpu_model.get(name)
                if cpu_param is None or tuple(cpu_param.shape) != tuple(param.shape):
                    cpu_param = torch.empty_like(param, device="cpu")
                    self.cpu_model[name] = cpu_param
                param.data = cpu_param

            self.gpu_buffers = None
            gc.collect()
            torch.npu.empty_cache()
            _rollout_memory_probe("offload_model_weights_after")

    def free_cache_engine(self):
        """Release KV cache tensors in the old rollout style."""
        with _rollout_stage_timer("free_cache_engine"):
            _rollout_memory_probe("free_cache_engine_before")
            self.inference_engine.llm_engine.reset_prefix_cache()

            worker = self.inference_engine.llm_engine.model_executor.driver_worker.worker
            model_runner = worker.model_runner
            ctx = model_runner.vllm_config.compilation_config.static_forward_context
            try:
                from vllm.v1.attention.backend import AttentionType
            except ImportError:
                from vllm.attention.layer import AttentionType

            pipeline_parallel_size = (
                self.inference_engine.llm_engine.vllm_config.parallel_config.pipeline_parallel_size
            )
            for layer_name in ctx:
                attn_type = getattr(ctx[layer_name], "attn_type", None)
                if attn_type in (AttentionType.DECODER, AttentionType.ENCODER_DECODER):
                    ctx[layer_name].kv_cache = [
                        torch.tensor([]) for _ in range(pipeline_parallel_size)
                    ]
                attn_impl = getattr(ctx[layer_name], "impl", None)
                if attn_impl is not None and hasattr(attn_impl, "key_cache"):
                    attn_impl.key_cache = None
                    attn_impl.value_cache = None

            model_runner.kv_caches = []
            model_runner.kv_connector_output = None
            # New vLLM-Ascend attention groups own metadata builders in addition to
            # the forward-context Attention modules. Drop them at release time so
            # they cannot keep per-step tensors alive until the next KV rebuild.
            if getattr(model_runner, "attn_groups", None):
                model_runner.attn_groups = []

            layers = getattr(getattr(self.model, "model", None), "layers", [])
            start_layer = getattr(getattr(self.model, "model", None), "start_layer", 0)
            end_layer = getattr(getattr(self.model, "model", None), "end_layer", len(layers))
            for i in range(start_layer, end_layer):
                self_attn = getattr(layers[i], "self_attn", None)
                for attn_attr in ("attn", "mla_attn"):
                    attn = getattr(self_attn, attn_attr, None)
                    attn_impl = getattr(attn, "impl", None)
                    if attn_impl is not None and hasattr(attn_impl, "key_cache"):
                        attn_impl.key_cache = None
                        attn_impl.value_cache = None

            gc.collect()
            torch.npu.empty_cache()
            _rollout_memory_probe("free_cache_engine_after")

    async def resume(self, tags: list[str]):
        """Resume rollout weights or kv cache in GPU memory.

        Args:
            tags: weights or kv_cache.
        """
        if self.manual_free_cache_engine:
            if "weights" in tags:
                with _rollout_stage_timer("resume_weights"):
                    self.onload_model_weights()
            if "kv_cache" in tags:
                with _rollout_stage_timer("resume_kv_cache"):
                    self.init_cache_engine()
            return

        if not self.config.free_cache_engine:
            return

        stage_name = "resume_kv_cache" if "kv_cache" in tags else "resume_weights"
        with _rollout_stage_timer(stage_name):
            if "tags" in inspect.signature(self.inference_engine.wake_up).parameters:
                self.inference_engine.wake_up(tags=tags)
            else:
                self.inference_engine.wake_up()

    async def release(self):
        """Release weights and kv cache in GPU memory."""
        self.inference_engine.reset_prefix_cache()

        if self.manual_free_cache_engine:
            self.free_cache_engine()
            self.offload_model_weights()
            return

        if not self.config.free_cache_engine:
            return

        self.inference_engine.sleep(level=self.sleep_level)

    async def update_weights(self, weights: Generator[tuple[str, torch.Tensor], None, None], **kwargs):
        """Update the weights of the rollout model.

        Args:
            weights: A generator that yields the name of the weight tensor and the tensor itself.
        """
        with _rollout_stage_timer("update_weights_total"):
            filter_empty_weight_shards = os.environ.get(
                "VLLM_ROLLOUT_FILTER_EMPTY_WEIGHT_SHARDS", "1"
            ).lower() in ("1", "true", "yes", "on")
            skipped = 0
            skipped_names: list[str] = []

            def iter_non_empty_weights():
                nonlocal skipped
                for name, weight in weights:
                    if (
                        filter_empty_weight_shards
                        and isinstance(weight, torch.Tensor)
                        and weight.numel() == 0
                    ):
                        skipped += 1
                        if len(skipped_names) < 8:
                            skipped_names.append(name)
                        continue
                    yield name, weight

            def maybe_log_skipped_weights():
                if skipped <= 0 or getattr(self, "_empty_weight_update_warned", False):
                    return
                logger.warning(
                    "Skip %s empty rollout weight shards during update_weights. sample_names=%s",
                    skipped,
                    skipped_names,
                )
                self._empty_weight_update_warned = True

            peft_config, base_sync_done = kwargs.get("peft_config", None), kwargs.get("base_sync_done", False)
            if peft_config and base_sync_done:
                with _rollout_stage_timer("update_weights_filter_lora"):
                    filtered_weights = list(iter_non_empty_weights())
                    maybe_log_skipped_weights()
                with _rollout_stage_timer("update_weights_add_lora"):
                    lora_int_id = int(time.time_ns() % 0x7FFFFFFF)
                    lora_reqest = TensorLoRARequest(
                        lora_name=f"{lora_int_id}",
                        lora_int_id=lora_int_id,
                        lora_path="simon_lora_path",
                        peft_config=asdict(peft_config),
                        lora_tensors=dict(filtered_weights),
                    )
                    self.inference_engine.llm_engine.add_lora(lora_reqest)
                    logger.info(f"vLLM load weights, loaded_params: {len(filtered_weights)}")
            else:
                from verl.utils.vllm.patch import patch_vllm_moe_model_weight_loader
                from vllm.model_executor.model_loader.utils import process_weights_after_loading

                model_runner = self.inference_engine.llm_engine.model_executor.driver_worker.worker.model_runner
                model = model_runner.get_model()
                patch_vllm_moe_model_weight_loader(model)
                pool_enabled = (
                    self.config.free_cache_engine
                    and not self.manual_free_cache_engine
                    and is_npu_available
                )
                _rollout_memory_probe("update_weights_before_load")
                with _maybe_use_rollout_weight_pool(pool_enabled):
                    with _rollout_stage_timer("update_weights_load_weights"):
                        model.load_weights(iter_non_empty_weights())
                        maybe_log_skipped_weights()
                    _rollout_memory_probe("update_weights_after_load_before_process")

                    model_config = model_runner.vllm_config.model_config
                    device_config = model_runner.vllm_config.device_config
                    load_config = model_runner.vllm_config.load_config
                    load_device = (
                        device_config.device if load_config.device is None else load_config.device
                    )
                    target_device = torch.device(load_device)
                    with _rollout_stage_timer("update_weights_process_after_loading"):
                        process_weights_after_loading(model, model_config, target_device)
                    _rollout_memory_probe("update_weights_after_process")
                    if _rollout_aclgraph_enabled(model_runner):
                        with _rollout_stage_timer("update_weights_invalidate_aclgraphs"):
                            cleared_aclgraphs = _invalidate_rollout_aclgraphs(
                                model_runner, "update_weights_after_process"
                            )
                        if cleared_aclgraphs:
                            self._needs_rollout_aclgraph_recapture = True
                        with _rollout_stage_timer("update_weights_capture_after_load"):
                            _capture_rollout_graphs_after_weight_load(
                                model_runner, "update_weights_after_process"
                            )
                    if self.manual_free_cache_engine and self.gpu_buffers is not None:
                        with _rollout_stage_timer("update_weights_drop_loader_buffers"):
                            # Split-MoE post-processing can replace param.data with
                            # execution-layout tensors.  The temporary loader-layout
                            # buffers are no longer part of the live model at that
                            # point, so keeping them referenced doubles part of the
                            # MoE footprint until rollout release.
                            self.gpu_buffers = None
                            gc.collect()
                            if is_npu_available:
                                torch.npu.empty_cache()
                            _rollout_memory_probe("update_weights_after_drop_loader_buffers")
                with _rollout_stage_timer("update_weights_post_gc_empty_cache"):
                    gc.collect()
                    if is_npu_available:
                        torch.npu.empty_cache()
                    _rollout_memory_probe("update_weights_after_empty_cache")


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
        self.sleep_level = _resolve_rollout_sleep_level(
            1 if config.layered_summon or config.expert_parallel_size > 1 else VLLM_SLEEP_LEVEL,
            bool(config.layered_summon),
        )

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
            with _maybe_use_rollout_weight_pool(
                self.config.free_cache_engine and is_npu_available
            ):
                model.load_weights(weights)

    def generate_sequences(self, prompts: DataProto) -> DataProto:
        """Batch generate sequences in sync mode."""
        raise NotImplementedError

    # ==================== server mode public methods ====================

    def get_zeromq_address(self):
        return self.address
