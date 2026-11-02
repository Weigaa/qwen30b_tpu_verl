# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Copyright 2024 The Qwen team.
# Copyright 2023 The vLLM team.
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
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
"""Inference-only Qwen3MoE model compatible with HuggingFace weights."""

import os
import time
import typing
from collections.abc import Callable, Iterable
from itertools import islice
from typing import Any

import torch
from torch import nn

from vllm.attention.layer import Attention
from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, VllmConfig, get_current_vllm_config
from vllm.distributed import (
    get_ep_group,
    get_pp_group,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_gather,
)
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.fused_moe import FusedMoE
from vllm.model_executor.layers.fused_moe.config import RoutingMethodType
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader,
    maybe_remap_kv_scale_name,
)
from vllm.model_executor.models.utils import sequence_parallel_chunk
from vllm.sequence import IntermediateTensors

from .interfaces import MixtureOfExperts, SupportsEagle3, SupportsLoRA, SupportsPP
from .utils import (
    AutoWeightsLoader,
    PPMissingLayer,
    extract_layer_index,
    is_pp_missing_parameter,
    make_empty_intermediate_tensors_factory,
    make_layers,
    maybe_prefix,
)

logger = init_logger(__name__)

_LAYER_PROFILE_ENABLED = os.getenv(
    "VLLM_QWEN3_MOE_LAYER_PROFILE",
    os.getenv("VLLM_QWEN2_MOE_LAYER_PROFILE", "0"),
) == "1"
_LAYER_PROFILE_FIRST_N = int(
    os.getenv(
        "VLLM_QWEN3_MOE_LAYER_PROFILE_FIRST_N",
        os.getenv("VLLM_QWEN2_MOE_LAYER_PROFILE_FIRST_N", "32"),
    )
)
_LAYER_PROFILE_INTERVAL = int(
    os.getenv(
        "VLLM_QWEN3_MOE_LAYER_PROFILE_INTERVAL",
        os.getenv("VLLM_QWEN2_MOE_LAYER_PROFILE_INTERVAL", "2048"),
    )
)
_LAYER_PROFILE_COUNTERS: dict[str, int] = {}
_SKIP_QK_NORM = os.getenv("VLLM_QWEN3_SKIP_QK_NORM", "0") in (
    "1",
    "true",
    "True",
)
_USE_QKV_RMSNORM_ROPE = os.getenv("VLLM_QWEN3_USE_QKV_RMSNORM_ROPE", "0") in (
    "1",
    "true",
    "True",
)
_USE_TORCH_NPU_QK_RMSNORM = os.getenv(
    "VLLM_QWEN3_USE_TORCH_NPU_QK_RMSNORM", "0"
) in ("1", "true", "True")
_DUMMY_SKIP_ATTENTION = os.getenv(
    "VLLM_ASCEND_QWEN3_DUMMY_SKIP_ATTENTION", "0"
).lower() in ("1", "true", "yes", "on")
_QKV_RMSNORM_ROPE_DISABLED = False
_TORCH_NPU = None
if _USE_TORCH_NPU_QK_RMSNORM:
    try:
        import torch_npu as _TORCH_NPU  # type: ignore[no-redef]
    except Exception as exc:
        logger.warning(
            "VLLM_QWEN3_USE_TORCH_NPU_QK_RMSNORM=1 requested but torch_npu "
            "could not be imported; falling back to vLLM RMSNorm: %s",
            exc,
        )

_CLASS_PROBE_ENABLED = os.getenv("VLLM_QWEN3_EAGER_CLASS_PROBE", "0").lower() in (
    "1",
    "true",
    "yes",
    "on",
)
_CLASS_PROBE_DONE = False


def _qualified_class_name(obj: Any) -> str:
    if obj is None:
        return "None"
    cls = obj.__class__
    return f"{cls.__module__}.{cls.__qualname__}"


def _forward_method_name(obj: Any) -> str:
    method = getattr(obj, "_forward_method", None)
    if method is None:
        return "None"
    return getattr(method, "__name__", repr(method))


def _attr_class_name(obj: Any, attr: str) -> str:
    if obj is None:
        return "None"
    return _qualified_class_name(getattr(obj, attr, None))


def _maybe_log_qwen3_eager_class_probe(model: "Qwen3MoeForCausalLM") -> None:
    """Log the real runtime classes selected by vLLM custom-op dispatch.

    The qwen3 eager path is heavily affected by out-of-tree Ascend CustomOp
    replacement.  Config logs only tell us that custom ops are enabled; this
    probe records the actual instantiated classes once so perf investigations
    can catch silent fallback to native vLLM classes.
    """
    global _CLASS_PROBE_DONE
    if not _CLASS_PROBE_ENABLED or _CLASS_PROBE_DONE:
        return
    _CLASS_PROBE_DONE = True

    sample_layer = None
    for layer in model.model.layers:
        if not isinstance(layer, PPMissingLayer):
            sample_layer = layer
            break
    if sample_layer is None:
        logger.warning("Qwen3Moe eager class probe found no local decoder layer")
        return

    sparse_layer = None
    for layer in model.model.layers:
        if not isinstance(layer, PPMissingLayer) and isinstance(
            layer.mlp, Qwen3MoeSparseMoeBlock
        ):
            sparse_layer = layer
            break
    sparse_mlp = None if sparse_layer is None else sparse_layer.mlp
    experts = None if sparse_mlp is None else sparse_mlp.experts
    quant_method = None if experts is None else getattr(experts, "quant_method", None)
    moe_config = None if experts is None else getattr(experts, "moe_config", None)
    attn = sample_layer.self_attn
    attn_impl = getattr(attn.attn, "impl", None)
    attn_backend = getattr(attn.attn, "attn_backend", None)

    try:
        from vllm.model_executor.custom_op import CustomOp

        oot_keys = ",".join(sorted(CustomOp.op_registry_oot.keys()))
    except Exception as exc:
        oot_keys = f"<unavailable:{exc}>"

    logger.info(
        "Qwen3Moe eager class probe model=%s embed=%s lm_head=%s "
        "logits_processor=%s sample_layer=%s input_norm=%s post_norm=%s "
        "attn=%s qkv=%s qkv_forward=%s o_proj=%s o_proj_forward=%s "
        "qkv_custom_op=%s qkv_quant=%s o_proj_custom_op=%s o_proj_quant=%s "
        "q_norm=%s q_norm_forward=%s k_norm=%s k_norm_forward=%s "
        "rotary=%s rotary_forward=%s attn_layer=%s attn_backend=%s "
        "attn_impl=%s mlp=%s gate=%s gate_forward=%s experts=%s "
        "gate_custom_op=%s gate_quant=%s experts_quant=%s "
        "experts_inner_quant=%s moe_use_ep=%s moe_reduce_results=%s "
        "registered_oot_ops=%s",
        _qualified_class_name(model),
        _qualified_class_name(model.model.embed_tokens),
        _qualified_class_name(model.lm_head),
        _qualified_class_name(model.logits_processor),
        _qualified_class_name(sample_layer),
        _qualified_class_name(sample_layer.input_layernorm),
        _qualified_class_name(sample_layer.post_attention_layernorm),
        _qualified_class_name(attn),
        _qualified_class_name(attn.qkv_proj),
        _forward_method_name(attn.qkv_proj),
        _qualified_class_name(attn.o_proj),
        _forward_method_name(attn.o_proj),
        _attr_class_name(attn.qkv_proj, "custom_op"),
        _attr_class_name(attn.qkv_proj, "quant_method"),
        _attr_class_name(attn.o_proj, "custom_op"),
        _attr_class_name(attn.o_proj, "quant_method"),
        _qualified_class_name(attn.q_norm),
        _forward_method_name(attn.q_norm),
        _qualified_class_name(attn.k_norm),
        _forward_method_name(attn.k_norm),
        _qualified_class_name(attn.rotary_emb),
        _forward_method_name(attn.rotary_emb),
        _qualified_class_name(attn.attn),
        None if attn_backend is None else attn_backend.get_name(),
        _qualified_class_name(attn_impl),
        _qualified_class_name(sparse_mlp),
        _qualified_class_name(None if sparse_mlp is None else sparse_mlp.gate),
        _forward_method_name(None if sparse_mlp is None else sparse_mlp.gate),
        _qualified_class_name(experts),
        _attr_class_name(None if sparse_mlp is None else sparse_mlp.gate, "custom_op"),
        _attr_class_name(None if sparse_mlp is None else sparse_mlp.gate, "quant_method"),
        _qualified_class_name(quant_method),
        _attr_class_name(quant_method, "quant_method"),
        None if moe_config is None else getattr(moe_config, "use_ep", None),
        None if experts is None else getattr(experts, "reduce_results", None),
        oot_keys,
    )


def _profile_enabled() -> bool:
    # Qwen3MoeModel is wrapped by torch.compile in compiled-eager mode.  Python
    # timers/counters inside the traced forward become Dynamo graph inputs, so
    # keep this layer-level profiler for non-compiled probe runs only.
    return _LAYER_PROFILE_ENABLED and not torch._dynamo.is_compiling()


def _profile_mark(name: str) -> tuple[int, bool]:
    count = _LAYER_PROFILE_COUNTERS.get(name, 0) + 1
    _LAYER_PROFILE_COUNTERS[name] = count
    should_log = count <= _LAYER_PROFILE_FIRST_N or (
        _LAYER_PROFILE_INTERVAL > 0 and count % _LAYER_PROFILE_INTERVAL == 0
    )
    return count, should_log


def _profile_time() -> float:
    if torch._dynamo.is_compiling():
        return 0.0
    try:
        torch.npu.synchronize()
    except Exception:
        pass
    return time.perf_counter()


def _qwen3_qk_rms_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> torch.Tensor | None:
    if _TORCH_NPU is None:
        return None
    try:
        return _TORCH_NPU.npu_rms_norm(x, weight, epsilon=eps)[0]
    except Exception as exc:
        logger.warning_once(
            "Falling back to vLLM RMSNorm for Qwen3 q/k norm because "
            "torch_npu.npu_rms_norm failed: %s",
            exc,
        )
        return None


def _profile_log(name: str, call: int, prefix: str, tokens: int, **values: float) -> None:
    body = " ".join(f"{key}_ms={value * 1000.0:.3f}" for key, value in values.items())
    logger.info(
        "Qwen3Moe layer profile pid=%s name=%s call=%d prefix=%s tokens=%s %s",
        os.getpid(),
        name,
        call,
        prefix,
        tokens,
        body,
    )


class Qwen3MoeMLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        quant_config: QuantizationConfig | None = None,
        reduce_results: bool = True,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            reduce_results=reduce_results,
            prefix=f"{prefix}.down_proj",
        )
        if hidden_act != "silu":
            raise ValueError(
                f"Unsupported activation: {hidden_act}. Only silu is supported for now."
            )
        self.act_fn = SiluAndMul()

    def forward(self, x):
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x


class Qwen3MoeSparseMoeBlock(nn.Module):
    def __init__(
        self,
        vllm_config: VllmConfig,
        prefix: str = "",
    ):
        super().__init__()

        config = vllm_config.model_config.hf_text_config
        parallel_config = vllm_config.parallel_config
        quant_config = vllm_config.quant_config

        self.tp_size = get_tensor_model_parallel_world_size()

        self.ep_group = get_ep_group().device_group
        self.ep_rank = get_ep_group().rank_in_group
        self.ep_size = self.ep_group.size()
        self.n_routed_experts = config.num_experts

        self.is_sequence_parallel = parallel_config.use_sequence_parallel_moe

        if self.tp_size > config.num_experts:
            raise ValueError(
                f"Tensor parallel size {self.tp_size} is greater than "
                f"the number of experts {config.num_experts}."
            )

        # Load balancing settings.
        vllm_config = get_current_vllm_config()
        eplb_config = vllm_config.parallel_config.eplb_config
        self.enable_eplb = parallel_config.enable_eplb

        self.n_logical_experts = self.n_routed_experts
        self.n_redundant_experts = eplb_config.num_redundant_experts
        self.n_physical_experts = self.n_logical_experts + self.n_redundant_experts
        self.n_local_physical_experts = self.n_physical_experts // self.ep_size

        self.physical_expert_start = self.ep_rank * self.n_local_physical_experts
        self.physical_expert_end = (
            self.physical_expert_start + self.n_local_physical_experts
        )

        use_ascend_legacy_init = os.getenv(
            "VLLM_QWEN3_MOE_ASCEND_LEGACY_INIT", "0"
        ) in ("1", "true", "True")
        use_ascend_legacy_stack = os.getenv(
            "VLLM_QWEN3_MOE_ASCEND_LEGACY_STACK", "0"
        ) in ("1", "true", "True")
        moe_cls = FusedMoE
        moe_kwargs = {}
        if use_ascend_legacy_stack:
            from vllm_ascend.ops.fused_moe_legacy import AscendFusedMoE

            moe_cls = AscendFusedMoE
            moe_kwargs.update(layer_idx=extract_layer_index(prefix))
            logger.info_once(
                "Using old-style Ascend Qwen3Moe experts stack for eager "
                "rollout profiling. Disable with "
                "VLLM_QWEN3_MOE_ASCEND_LEGACY_STACK=0."
            )
        elif use_ascend_legacy_init:
            from vllm_ascend.ops.fused_moe import AscendFusedMoE

            moe_cls = AscendFusedMoE
            logger.info_once(
                "Using Ascend legacy-style Qwen3Moe experts init for eager "
                "rollout profiling. Disable with "
                "VLLM_QWEN3_MOE_ASCEND_LEGACY_INIT=0."
            )
        else:
            moe_kwargs.update(
                enable_eplb=self.enable_eplb,
                num_redundant_experts=self.n_redundant_experts,
                is_sequence_parallel=self.is_sequence_parallel,
                routing_method_type=RoutingMethodType.Renormalize,
            )

        reduce_results_default = "0" if use_ascend_legacy_stack else "1"
        self._use_ascend_legacy_stack = use_ascend_legacy_stack
        self.experts = moe_cls(
            num_experts=self.n_routed_experts,
            top_k=config.num_experts_per_tok,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            # vLLM's generic Qwen3Moe path adds a maybe_all_reduce wrapper when
            # this is True. On Ascend EP/MC2 rollout the communication method
            # already returns complete local hidden states, so keep this
            # tunable to match the older Ascend custom Qwen3Moe fast path.
            reduce_results=os.getenv(
                "VLLM_QWEN3_MOE_REDUCE_RESULTS", reduce_results_default)
            not in ("0", "false", "False"),
            renormalize=config.norm_topk_prob,
            quant_config=quant_config,
            prefix=f"{prefix}.experts",
            **moe_kwargs,
        )

        self.gate = ReplicatedLinear(
            config.hidden_size,
            config.num_experts,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.gate",
        )
        self.prefix = prefix

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        assert hidden_states.dim() <= 2, (
            "Qwen3MoeSparseMoeBlock only supports 1D or 2D inputs"
        )
        is_input_1d = hidden_states.dim() == 1
        num_tokens, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)

        if self.is_sequence_parallel:
            hidden_states = sequence_parallel_chunk(hidden_states)

        profile_call = 0
        should_profile = False
        if _profile_enabled():
            profile_call, should_profile = _profile_mark("sparse_moe")
            if should_profile:
                tokens = int(hidden_states.shape[0])
                t0 = _profile_time()

        # router_logits: (num_tokens, n_experts)
        router_logits, _ = self.gate(hidden_states)
        if should_profile:
            t1 = _profile_time()
        if self._use_ascend_legacy_stack:
            use_npugraph_moe_boundary = os.getenv(
                "VLLM_ASCEND_NPUGRAPH_EX_MOE_CUSTOM_OP_BOUNDARY", "0"
            ).lower() in ("1", "true", "yes", "on")
            if use_npugraph_moe_boundary:
                final_hidden_states = torch.ops.vllm.ascend_legacy_moe_forward(
                    hidden_states, router_logits, self.experts.layer_name
                )
            else:
                forward_context = get_forward_context()
                final_hidden_states = self.experts(
                    hidden_states=hidden_states,
                    router_logits=router_logits,
                    is_prefill=getattr(forward_context, "with_prefill", False),
                    enable_force_load_balance=getattr(
                        forward_context, "in_profile_run", False),
                    top_k=self.experts.top_k,
                    shared_experts=None,
                    is_dummy=False,
                )
        else:
            final_hidden_states = self.experts(
                hidden_states=hidden_states, router_logits=router_logits
            )
        if should_profile:
            t2 = _profile_time()

        if self.is_sequence_parallel:
            final_hidden_states = tensor_model_parallel_all_gather(
                final_hidden_states, 0
            )
            final_hidden_states = final_hidden_states[:num_tokens]
        if should_profile:
            t3 = _profile_time()
            _profile_log(
                "sparse_moe",
                profile_call,
                self.prefix,
                tokens,
                gate=t1 - t0,
                experts=t2 - t1,
                allgather=t3 - t2,
                total=t3 - t0,
            )

        # return to 1d if input is 1d
        return final_hidden_states.squeeze(0) if is_input_1d else final_hidden_states


class Qwen3MoeAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        rope_parameters: dict[str, Any],
        max_position_embeddings: int = 8192,
        head_dim: int | None = None,
        rms_norm_eps: float = 1e-06,
        qkv_bias: bool = False,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        dual_chunk_attention_config: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= tp_size:
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            assert self.total_num_kv_heads % tp_size == 0
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = head_dim or (hidden_size // self.total_num_heads)
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.max_position_embeddings = max_position_embeddings
        self.dual_chunk_attention_config = dual_chunk_attention_config

        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=qkv_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )

        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        self.rotary_emb = get_rope(
            self.head_dim,
            max_position=max_position_embeddings,
            rope_parameters=rope_parameters,
            dual_chunk_attention_config=dual_chunk_attention_config,
        )
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
            **{
                "layer_idx": extract_layer_index(prefix),
                "dual_chunk_attention_config": dual_chunk_attention_config,
            }
            if dual_chunk_attention_config
            else {},
        )

        self.q_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
        self.prefix = prefix
        if _SKIP_QK_NORM:
            logger.warning_once(
                "VLLM_QWEN3_SKIP_QK_NORM=1 is enabled. This is a "
                "diagnostic-only performance probe that changes Qwen3 "
                "attention numerics and must not be used for training."
            )
        if _USE_QKV_RMSNORM_ROPE:
            logger.warning_once(
                "VLLM_QWEN3_USE_QKV_RMSNORM_ROPE=1 is enabled. This opt-in "
                "probe uses the narrow fused qkv split + q/k RMSNorm + RoPE "
                "kernel in Qwen3 attention."
            )
        if _USE_TORCH_NPU_QK_RMSNORM:
            logger.warning_once(
                "VLLM_QWEN3_USE_TORCH_NPU_QK_RMSNORM=1 is enabled. This "
                "opt-in probe replaces only Qwen3 q/k RMSNorm calls with "
                "torch_npu.npu_rms_norm."
            )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        profile_call = 0
        should_profile = False
        if _profile_enabled():
            profile_call, should_profile = _profile_mark("attention")
            if should_profile:
                tokens = int(hidden_states.shape[0])
                t0 = _profile_time()
        qkv, _ = self.qkv_proj(hidden_states)
        if should_profile:
            t1 = _profile_time()
        global _QKV_RMSNORM_ROPE_DISABLED
        used_fused_qkv_rmsnorm_rope = False
        if (
            _USE_QKV_RMSNORM_ROPE
            and not _SKIP_QK_NORM
            and not _QKV_RMSNORM_ROPE_DISABLED
            and self.head_dim == 128
            and self.dual_chunk_attention_config is None
        ):
            try:
                from vllm_ascend.ops.rotary_embedding import get_cos_and_sin_slice

                cos, sin = get_cos_and_sin_slice()
                if cos is None or sin is None:
                    raise RuntimeError("Ascend RoPE cos/sin slice is not initialized")
                q, k, v = torch.ops.vllm.qkv_rmsnorm_rope(
                    input=qkv,
                    sin=sin,
                    cos=cos,
                    q_weight=self.q_norm.weight,
                    k_weight=self.k_norm.weight,
                    q_hidden_size=self.q_size,
                    kv_hidden_size=self.kv_size,
                    head_dim=self.head_dim,
                    eps=self.q_norm.variance_epsilon,
                    q_bias=None,
                    k_bias=None,
                )
                used_fused_qkv_rmsnorm_rope = True
            except Exception as exc:
                _QKV_RMSNORM_ROPE_DISABLED = True
                logger.warning_once(
                    "Disabling VLLM_QWEN3_USE_QKV_RMSNORM_ROPE after fused "
                    "path failure: %s",
                    exc,
                )
                q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        else:
            q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        if not _SKIP_QK_NORM and not used_fused_qkv_rmsnorm_rope:
            # Qwen3 adds per-head q/k RMSNorm before RoPE.  Keep an explicit
            # diagnostic bypass so we can measure how much of the old-vs-new
            # eager gap is architectural q/k normalization cost.
            q_by_head = q.view(
                *q.shape[:-1], q.shape[-1] // self.head_dim, self.head_dim
            )
            q_npu = _qwen3_qk_rms_norm(
                q_by_head, self.q_norm.weight, self.q_norm.variance_epsilon
            )
            q_by_head = q_npu if q_npu is not None else self.q_norm(q_by_head)
            q = q_by_head.view(q.shape)

            k_by_head = k.view(
                *k.shape[:-1], k.shape[-1] // self.head_dim, self.head_dim
            )
            k_npu = _qwen3_qk_rms_norm(
                k_by_head, self.k_norm.weight, self.k_norm.variance_epsilon
            )
            k_by_head = k_npu if k_npu is not None else self.k_norm(k_by_head)
            k = k_by_head.view(k.shape)
        if not used_fused_qkv_rmsnorm_rope:
            q, k = self.rotary_emb(positions, q, k)
        if should_profile:
            t2 = _profile_time()
        attn_output = self.attn(q, k, v)
        if should_profile:
            t3 = _profile_time()
        output, _ = self.o_proj(attn_output)
        if should_profile:
            t4 = _profile_time()
            _profile_log(
                "attention",
                profile_call,
                self.prefix,
                tokens,
                qkv=t1 - t0,
                norm_rope=t2 - t1,
                attn=t3 - t2,
                o_proj=t4 - t3,
                total=t4 - t0,
            )
        return output


class Qwen3MoeDecoderLayer(nn.Module):
    def __init__(self, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()

        config = vllm_config.model_config.hf_text_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config

        self.hidden_size = config.hidden_size
        max_position_embeddings = getattr(config, "max_position_embeddings", 8192)
        dual_chunk_attention_config = getattr(
            config, "dual_chunk_attention_config", None
        )
        self.self_attn = Qwen3MoeAttention(
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            rope_parameters=config.rope_parameters,
            max_position_embeddings=max_position_embeddings,
            rms_norm_eps=config.rms_norm_eps,
            qkv_bias=getattr(config, "attention_bias", False),
            head_dim=getattr(config, "head_dim", None),
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.self_attn",
            dual_chunk_attention_config=dual_chunk_attention_config,
        )

        # `mlp_only_layers` in the config.
        layer_idx = extract_layer_index(prefix)
        mlp_only_layers = (
            [] if not hasattr(config, "mlp_only_layers") else config.mlp_only_layers
        )
        if (layer_idx not in mlp_only_layers) and (
            config.num_experts > 0 and (layer_idx + 1) % config.decoder_sparse_step == 0
        ):
            self.mlp = Qwen3MoeSparseMoeBlock(
                vllm_config=vllm_config, prefix=f"{prefix}.mlp"
            )
        else:
            self.mlp = Qwen3MoeMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                prefix=f"{prefix}.mlp",
            )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.prefix = prefix

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        is_dummy: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        profile_call = 0
        should_profile = False
        if _profile_enabled():
            profile_call, should_profile = _profile_mark("decoder_layer")
            if should_profile:
                tokens = int(hidden_states.shape[0])
                t0 = _profile_time()
        if is_dummy and _DUMMY_SKIP_ATTENTION:
            if residual is None:
                residual = hidden_states
            if should_profile:
                t1 = t2 = t3 = _profile_time()
        else:
            # Self Attention
            if residual is None:
                residual = hidden_states
                hidden_states = self.input_layernorm(hidden_states)
            else:
                hidden_states, residual = self.input_layernorm(hidden_states, residual)
            if should_profile:
                t1 = _profile_time()
            hidden_states = self.self_attn(
                positions=positions,
                hidden_states=hidden_states,
            )
            if should_profile:
                t2 = _profile_time()

            # Fully Connected
            hidden_states, residual = self.post_attention_layernorm(
                hidden_states, residual)
            if should_profile:
                t3 = _profile_time()
        hidden_states = self.mlp(hidden_states)
        if should_profile:
            t4 = _profile_time()
            _profile_log(
                "decoder_layer",
                profile_call,
                self.prefix,
                tokens,
                input_norm=t1 - t0,
                attention=t2 - t1,
                post_norm=t3 - t2,
                mlp=t4 - t3,
                total=t4 - t0,
            )
        return hidden_states, residual


@support_torch_compile
class Qwen3MoeModel(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        config = vllm_config.model_config.hf_text_config
        quant_config = vllm_config.quant_config
        parallel_config = vllm_config.parallel_config
        eplb_config = parallel_config.eplb_config
        self.num_redundant_experts = eplb_config.num_redundant_experts

        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.config = config
        self.quant_config = quant_config
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            quant_config=quant_config,
            prefix=f"{prefix}.embed_tokens",
        )
        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: Qwen3MoeDecoderLayer(vllm_config=vllm_config, prefix=prefix),
            prefix=f"{prefix}.layers",
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states", "residual"], config.hidden_size
        )
        # Track layers for auxiliary hidden state outputs (EAGLE3)
        self.aux_hidden_state_layers: tuple[int, ...] = ()

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        is_dummy: bool = False,
    ) -> torch.Tensor | IntermediateTensors | tuple[torch.Tensor, list[torch.Tensor]]:
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.embed_input_ids(input_ids)
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]

        aux_hidden_states = []
        for layer_idx, layer in enumerate(
            islice(self.layers, self.start_layer, self.end_layer),
            start=self.start_layer,
        ):
            # Collect auxiliary hidden states if specified
            if layer_idx in self.aux_hidden_state_layers:
                aux_hidden_state = (
                    hidden_states + residual if residual is not None else hidden_states
                )
                aux_hidden_states.append(aux_hidden_state)
            hidden_states, residual = layer(
                positions, hidden_states, residual, is_dummy=is_dummy)

        if not get_pp_group().is_last_rank:
            return IntermediateTensors(
                {"hidden_states": hidden_states, "residual": residual}
            )
        hidden_states, _ = self.norm(hidden_states, residual)

        # Return auxiliary hidden states if collected
        if len(aux_hidden_states) > 0:
            return hidden_states, aux_hidden_states
        return hidden_states

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        # Params for weights, fp8 weight scales, fp8 activation scales
        # (param_name, weight_name, expert_id, shard_id)
        return FusedMoE.make_expert_params_mapping(
            self,
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.config.num_experts,
            num_redundant_experts=self.num_redundant_experts,
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]

        # Skip loading extra parameters for GPTQ/modelopt models.
        ignore_suffixes = (
            ".bias",
            "_bias",
            ".k_scale",
            "_k_scale",
            ".v_scale",
            "_v_scale",
            ".weight_scale",
            "_weight_scale",
            ".input_scale",
            "_input_scale",
        )

        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        expert_params_mapping = self.get_expert_mapping()
        for name, loaded_weight in weights:
            if os.getenv("VLLM_ROLLOUT_FAST_WEIGHT_LOAD", "0") == "1" and ".mlp.experts." in name:
                parts = name.split(".")
                if parts[0] == "model":
                    parts = parts[1:]
                # Fast path for RL weight reloads. The rollout exporter already
                # emits local EP expert weights, so avoid scanning the full
                # expert mapping table for every gate/up/down tensor.
                if (
                    len(parts) == 5
                    and parts[0] == "layers"
                    and parts[2] == "mlp"
                    and parts[3] == "experts"
                    and parts[4] in ("_local_w13_weight", "_local_w2_weight")
                ):
                    name_mapped = ".".join(parts[:4]) + (
                        ".w13_weight" if parts[4] == "_local_w13_weight" else ".w2_weight"
                    )
                    if not is_pp_missing_parameter(name_mapped, self) and name_mapped in params_dict:
                        params_dict[name_mapped].data.copy_(loaded_weight)
                        loaded_params.add(name_mapped)
                    continue
                if not name.endswith(".weight"):
                    # Non-standard expert tensors are handled above; remaining
                    # fast-load candidates must be normal per-expert weights.
                    pass
                if (
                    len(parts) >= 7
                    and parts[0] == "layers"
                    and parts[2] == "mlp"
                    and parts[3] == "experts"
                    and parts[5] in ("gate_proj", "up_proj", "down_proj")
                ):
                    expert_id = int(parts[4])
                    if parts[5] == "gate_proj":
                        name_mapped = ".".join(parts[:4]) + ".w13_weight"
                        shard_id = "w1"
                    elif parts[5] == "up_proj":
                        name_mapped = ".".join(parts[:4]) + ".w13_weight"
                        shard_id = "w3"
                    else:
                        name_mapped = ".".join(parts[:4]) + ".w2_weight"
                        shard_id = "w2"

                    if not is_pp_missing_parameter(name_mapped, self) and name_mapped in params_dict:
                        param = params_dict[name_mapped]
                        weight_loader = typing.cast(
                            Callable[..., bool], param.weight_loader
                        )
                        success = weight_loader(
                            param,
                            loaded_weight,
                            name_mapped,
                            shard_id=shard_id,
                            expert_id=expert_id,
                            return_success=True,
                        )
                        if success:
                            loaded_params.add(name_mapped)
                            continue
                        # Local-exported weights should normally be local to
                        # this EP rank. If EPLB/remapping says otherwise, skip
                        # exactly like the generic expert loader does.
                        continue

            if self.quant_config is not None and (
                scale_name := self.quant_config.get_cache_scale(name)
            ):
                # Loading kv cache quantization scales
                param = params_dict[scale_name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                assert loaded_weight.numel() == 1, (
                    f"KV scale numel {loaded_weight.numel()} != 1"
                )
                loaded_weight = loaded_weight.squeeze()
                weight_loader(param, loaded_weight)
                loaded_params.add(scale_name)
                continue
            for param_name, weight_name, shard_id in stacked_params_mapping:
                # Skip non-stacked layers and experts (experts handled below).
                if weight_name not in name:
                    continue
                # We have mlp.experts[0].gate_proj in the checkpoint.
                # Since we handle the experts below in expert_params_mapping,
                # we need to skip here BEFORE we update the name, otherwise
                # name will be updated to mlp.experts[0].gate_up_proj, which
                # will then be updated below in expert_params_mapping
                # for mlp.experts[0].gate_gate_up_proj, which breaks load.
                if "mlp.experts" in name:
                    continue
                name = name.replace(weight_name, param_name)

                # Skip loading extra parameters for GPTQ/modelopt models.
                if name.endswith(ignore_suffixes) and name not in params_dict:
                    continue

                # Skip layers on other devices.
                if is_pp_missing_parameter(name, self):
                    continue
                if name.endswith("scale"):
                    # Remapping the name of FP8 kv-scale.
                    name = maybe_remap_kv_scale_name(name, params_dict)
                    if name is None:
                        continue
                if name not in params_dict:
                    continue

                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                if weight_loader == default_weight_loader:
                    weight_loader(param, loaded_weight)
                else:
                    weight_loader(param, loaded_weight, shard_id)
                break
            else:
                is_expert_weight = False
                for mapping in expert_params_mapping:
                    param_name, weight_name, expert_id, shard_id = mapping
                    if weight_name not in name:
                        continue

                    # Anyway, this is an expert weight and should not be
                    # attempted to load as other weights later
                    is_expert_weight = True

                    # Do not modify `name` since the loop may continue here
                    # Instead, create a new variable
                    name_mapped = name.replace(weight_name, param_name)

                    if is_pp_missing_parameter(name_mapped, self):
                        continue

                    # Skip loading extra parameters for GPTQ/modelopt models.
                    if (
                        name_mapped.endswith(ignore_suffixes)
                        and name_mapped not in params_dict
                    ):
                        continue

                    param = params_dict[name_mapped]
                    # We should ask the weight loader to return success or not
                    # here since otherwise we may skip experts with other
                    # available replicas.
                    weight_loader = typing.cast(
                        Callable[..., bool], param.weight_loader
                    )
                    success = weight_loader(
                        param,
                        loaded_weight,
                        name_mapped,
                        shard_id=shard_id,
                        expert_id=expert_id,
                        return_success=True,
                    )
                    if success:
                        name = name_mapped
                        break
                else:
                    if is_expert_weight:
                        # We've checked that this is an expert weight
                        # However it's not mapped locally to this rank
                        # So we simply skip it
                        continue

                    # Skip loading extra parameters for GPTQ/modelopt models.
                    if name.endswith(ignore_suffixes) and name not in params_dict:
                        continue
                    # Skip layers on other devices.
                    if is_pp_missing_parameter(name, self):
                        continue
                    # Remapping the name of FP8 kv-scale.
                    if name.endswith("kv_scale"):
                        remapped_kv_scale_name = name.replace(
                            ".kv_scale", ".attn.kv_scale"
                        )
                        if remapped_kv_scale_name not in params_dict:
                            logger.warning_once(
                                "Found kv scale in the checkpoint (e.g. %s), but not found the expected name in the model (e.g. %s). kv-scale is not loaded.",  # noqa: E501
                                name,
                                remapped_kv_scale_name,
                            )
                            continue
                        else:
                            name = remapped_kv_scale_name
                    param = params_dict[name]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params


class Qwen3MoeForCausalLM(
    nn.Module, SupportsPP, SupportsLoRA, SupportsEagle3, MixtureOfExperts
):
    packed_modules_mapping = {
        "qkv_proj": [
            "q_proj",
            "k_proj",
            "v_proj",
        ]
    }

    fall_back_to_pt_during_load = False

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_text_config
        quant_config = vllm_config.quant_config
        self.config = config
        self.quant_config = quant_config
        # Only perform the following mapping when Qwen3MoeMLP exists
        if getattr(config, "mlp_only_layers", []):
            self.packed_modules_mapping["gate_up_proj"] = ["gate_proj", "up_proj"]
        self.model = Qwen3MoeModel(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )
        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, "lm_head"),
        )
        if self.config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight
        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )

        # Set MoE hyperparameters
        self.expert_weights = []

        self.moe_layers = []
        example_layer = None
        for layer in self.model.layers:
            if isinstance(layer, PPMissingLayer):
                continue

            assert isinstance(layer, Qwen3MoeDecoderLayer)
            if isinstance(layer.mlp, Qwen3MoeSparseMoeBlock):
                example_layer = layer.mlp
                self.moe_layers.append(layer.mlp.experts)

        if example_layer is None:
            raise RuntimeError("No Qwen3MoE layer found in the model.layers.")

        self.num_moe_layers = len(self.moe_layers)
        self.num_expert_groups = 1
        self.num_shared_experts = 0
        self.num_logical_experts = example_layer.n_logical_experts
        self.num_physical_experts = example_layer.n_physical_experts
        self.num_local_physical_experts = example_layer.n_local_physical_experts
        self.num_routed_experts = example_layer.n_routed_experts
        self.num_redundant_experts = example_layer.n_redundant_experts
        _maybe_log_qwen3_eager_class_probe(self)

    def update_physical_experts_metadata(
        self,
        num_physical_experts: int,
        num_local_physical_experts: int,
    ) -> None:
        assert self.num_local_physical_experts == num_local_physical_experts
        self.num_physical_experts = num_physical_experts
        self.num_local_physical_experts = num_local_physical_experts
        self.num_redundant_experts = num_physical_experts - self.num_logical_experts
        for layer in self.model.layers:
            if isinstance(layer.mlp, Qwen3MoeSparseMoeBlock):
                moe = layer.mlp
                moe.n_local_physical_experts = num_local_physical_experts
                moe.n_physical_experts = num_physical_experts
                moe.n_redundant_experts = self.num_redundant_experts
                moe.experts.update_expert_map()

    def set_aux_hidden_state_layers(self, layers: tuple[int, ...]) -> None:
        self.model.aux_hidden_state_layers = layers

    def get_eagle3_aux_hidden_state_layers(self) -> tuple[int, ...]:
        num_layers = len(self.model.layers)
        return (2, num_layers // 2, num_layers - 3)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        is_dummy: bool = False,
    ) -> torch.Tensor | IntermediateTensors:
        hidden_states = self.model(input_ids, positions, intermediate_tensors,
                                   inputs_embeds, is_dummy=is_dummy)
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        logits = self.logits_processor(self.lm_head, hidden_states)
        if logits is not None and logits.shape[-1] > self.config.vocab_size:
            logits[..., self.config.vocab_size:] = float("-inf")
        return logits

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights)

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        return self.model.get_expert_mapping()
