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
import typing
import os
from collections.abc import Callable, Iterable
from itertools import islice
from typing import Any, Optional, Union

import torch
from torch import nn

from vllm.attention import Attention
from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, VllmConfig, get_current_vllm_config
from vllm.distributed import (get_ep_group, get_pp_group,
                              get_tensor_model_parallel_world_size,
                              tensor_model_parallel_all_gather)
from vllm.logger import init_logger
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.fused_moe import FusedMoE
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (MergedColumnParallelLinear,
                                               QKVParallelLinear,
                                               ReplicatedLinear,
                                               RowParallelLinear)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead, VocabParallelEmbedding)
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader, maybe_remap_kv_scale_name)
from vllm.model_executor.models.utils import sequence_parallel_chunk
from vllm.sequence import IntermediateTensors

from .interfaces import MixtureOfExperts, SupportsLoRA, SupportsPP
from .utils import (AutoWeightsLoader, PPMissingLayer, extract_layer_index,
                    is_pp_missing_parameter,
                    make_empty_intermediate_tensors_factory, make_layers,
                    maybe_prefix)
#新增
from vllm.forward_context import get_forward_context
from vllm.utils.moe_stats import moe_stats
from vllm.distributed.parallel_state import get_ep_group
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
import torch.distributed as dist
import time

logger = init_logger(__name__)

_STAGE_DECODE_PROFILE_MARKERS = os.getenv(
    "VLLM_ASCEND_STAGE_DECODE_PROFILE_MARKERS", "0").lower() in (
        "1", "true", "yes", "on")

_ENABLE_NATIVE_MOE_TOPK_DEBUG = os.getenv(
    "VLLM_ASCEND_NATIVE_MOE_TOPK_DEBUG", "0").lower() in (
        "1", "true", "yes", "on")


def _mode1_load_weights_deep_diag_enabled() -> bool:
    return os.getenv("VLLM_ASCEND_MODE1_LOAD_WEIGHTS_DEEP_DIAG",
                     "0").lower() in ("1", "true", "yes", "on")


def _mode1_current_rank_for_diag() -> int:
    try:
        if dist.is_available() and dist.is_initialized():
            return int(dist.get_rank())
    except Exception:
        pass
    try:
        return int(get_ep_group().rank_in_group)
    except Exception:
        return -1


def _profile_rank() -> int:
    try:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            return int(torch.distributed.get_rank())
    except Exception:
        pass
    return -1


def _stage_decode_profile_range_start(message: str):
    if not _STAGE_DECODE_PROFILE_MARKERS:
        return None
    try:
        from torch_npu.npu import mstx
        return mstx.range_start(message=message)
    except Exception:
        return None


def _stage_decode_profile_range_end(range_id) -> None:
    if range_id is None:
        return
    try:
        from torch_npu.npu import mstx
        mstx.range_end(range_id)
    except Exception:
        pass


def _stage_decode_attention_profile_message(layer_idx: int,
                                            runtime_layer: Any) -> str:
    mode = int(getattr(runtime_layer, "elastic_execution_mode", 0))
    stage = len(getattr(runtime_layer, "lossless_hybrid_active_ranks", []))
    return " ".join([
        "vllm_stage_decode_attention_compute",
        f"rank={_profile_rank()}",
        f"mode={mode}",
        f"stage={stage}",
        f"layer={int(layer_idx)}",
        "path=self_attn",
    ])


class Qwen3MoeMLP(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        quant_config: Optional[QuantizationConfig] = None,
        reduce_results: bool = True,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size, [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj")
        self.down_proj = RowParallelLinear(intermediate_size,
                                           hidden_size,
                                           bias=False,
                                           quant_config=quant_config,
                                           reduce_results=reduce_results,
                                           prefix=f"{prefix}.down_proj")
        if hidden_act != "silu":
            raise ValueError(f"Unsupported activation: {hidden_act}. "
                             "Only silu is supported for now.")
        self.act_fn = SiluAndMul()

    def forward(self, x, is_dummy: Optional[bool] = False):
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
        self.prefix = prefix
        self.ep_group = get_ep_group().device_group
        self.ep_rank = self.ep_group.rank()
        self.ep_size = self.ep_group.size()
        self.n_routed_experts = config.num_experts
        self.layer_idx = extract_layer_index(prefix)

        self.is_sequence_parallel = parallel_config.use_sequence_parallel_moe

        if self.tp_size > config.num_experts:
            raise ValueError(
                f"Tensor parallel size {self.tp_size} is greater than "
                f"the number of experts {config.num_experts}.")

        # Load balancing settings.
        vllm_config = get_current_vllm_config()
        eplb_config = vllm_config.parallel_config.eplb_config
        self.enable_eplb = parallel_config.enable_eplb

        self.n_logical_experts = self.n_routed_experts
        self.n_redundant_experts = eplb_config.num_redundant_experts
        self.n_physical_experts = (self.n_logical_experts +
                                   self.n_redundant_experts)
        self.n_local_physical_experts = self.n_physical_experts // self.ep_size

        self.physical_expert_start = (self.ep_rank *
                                      self.n_local_physical_experts)
        self.physical_expert_end = (self.physical_expert_start +
                                    self.n_local_physical_experts)

        self.experts = FusedMoE(num_experts=self.n_routed_experts,
                                top_k=config.num_experts_per_tok,
                                hidden_size=config.hidden_size,
                                intermediate_size=config.moe_intermediate_size,
                                reduce_results=True,
                                renormalize=config.norm_topk_prob,
                                quant_config=quant_config,
                                prefix=f"{prefix}.experts",
                                enable_eplb=self.enable_eplb,
                                num_redundant_experts=self.n_redundant_experts,
                                is_sequence_parallel=self.is_sequence_parallel)

        self.gate = ReplicatedLinear(config.hidden_size,
                                     config.num_experts,
                                     bias=False,
                                     quant_config=quant_config,
                                     prefix=f"{prefix}.gate")
        #新增统计
        self.ep_sig_total_cnt = 0     # 总调用次数
        self.ep_sig_same_cnt = 0      # 所有 rank token 完全相同的次数
        self.ep_sig_diff_cnt = 0      # 至少有一个 rank 不同的次数
    def compute_topk(
        self,
        router_logits: torch.Tensor,
        k: int = 8,
    ):
        # 计算每个 token 的 top-k 专家
        k = min(k, router_logits.shape[-1])
        _, topk_ids = torch.topk(router_logits, k=k, dim=-1)      # [T, k] (long)
        topk_ids = topk_ids.to(torch.int32)

        return topk_ids
    
    def _ep_same_input_guard(self, topk_ids: torch.Tensor, layer_idx: int, note: str = ""):
        """最小扰动版：用 topk_ids 做轻量签名，检测各 EP rank 是否处理相同 token。"""
        ep_group = get_ep_group().device_group
        world = ep_group.size()
        if world <= 1:
            return

        # 计算一个极轻的整数签名： [T, sum(topk), sum(col_weighted_topk)]
        # 说明：T不同或topk内容不同都会导致签名不同；碰撞概率很低，足够排查问题。
        if topk_ids.ndim == 1:
            topk_ids = topk_ids.view(1, -1)
        T, K = topk_ids.shape
        tki = topk_ids.to(dtype=torch.int64)
        col = torch.arange(1, K + 1, device=tki.device, dtype=torch.int64)  # 1..K
        sig = torch.stack([
            torch.tensor(T, device=tki.device, dtype=torch.int64),
            tki.sum(dtype=torch.int64),
            (tki * col).sum(dtype=torch.int64),
        ])

        # 在 EP 组内收集并比较
        gathered = [torch.empty_like(sig) for _ in range(world)]
        dist.all_gather(gathered, sig, group=ep_group)

        # 若所有签名完全相同，则基本可判定“各 EP 正在处理相同的一批 token”
        all_same = all(torch.equal(gathered[0], g) for g in gathered[1:])
        if all_same:
            if ep_group.rank() == 0:
                self.ep_sig_total_cnt += 1
                self.ep_sig_same_cnt += 1
                # for i in range(0, world):
                #     print(f"rank,{i},[EP WARNING] layer={layer_idx} world={world} -> all EP ranks share identical signature "
                #         f"{gathered[i].detach().cpu().tolist()} {note}")
        else:
            if ep_group.rank() == 0: 
                self.ep_sig_total_cnt += 1
                self.ep_sig_diff_cnt += 1
                # for i in range(0, world):
                #     print(f"rank,{i},#### layer={layer_idx} world={world} -> EP ranks share different signature "
                #         f"{gathered[i].detach().cpu().tolist()} {note}")
        if self.ep_sig_total_cnt % 1000 == 0:
            if ep_group.rank() == 0:
                print("same ratio is", self.ep_sig_same_cnt / self.ep_sig_total_cnt)
                print("diff ratio is", self.ep_sig_diff_cnt / self.ep_sig_total_cnt)


    def forward(self,
                hidden_states: torch.Tensor,
                is_dummy: Optional[bool] = False) -> torch.Tensor:
        assert hidden_states.dim(
        ) <= 2, "Qwen3MoeSparseMoeBlock only supports 1D or 2D inputs"
        is_input_1d = hidden_states.dim() == 1
        num_tokens, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)

        if self.is_sequence_parallel:
            hidden_states = sequence_parallel_chunk(hidden_states)

        # router_logits: (num_tokens, n_experts)
        router_logits, _ = self.gate(hidden_states)
        if _ENABLE_NATIVE_MOE_TOPK_DEBUG:
            topk_ids = self.compute_topk(router_logits)
            moe_stats.record_layer_topk(self.layer_idx, topk_ids)
            # self._ep_same_input_guard(topk_ids, self.layer_idx, note=f"(run={getattr(self,'total_run',-1)})")
            # moe_stats.record(
            #     layer_idx=self.layer_idx,
            #     topk_ids=topk_ids,
            #     num_experts=128,
            # )
        if hasattr(self.experts, "elastic_execution_mode"):
            forward_context = get_forward_context()
            final_hidden_states = self.experts(
                hidden_states=hidden_states,
                router_logits=router_logits,
                is_prefill=forward_context.with_prefill,
                top_k=self.experts.top_k,
                enable_force_load_balance=forward_context.in_profile_run,
                shared_experts=None,
                is_dummy=is_dummy,
            )
        else:
            final_hidden_states = self.experts(hidden_states=hidden_states,
                                               router_logits=router_logits)

        if self.is_sequence_parallel:
            final_hidden_states = tensor_model_parallel_all_gather(
                final_hidden_states, 0)
            final_hidden_states = final_hidden_states[:num_tokens]

        # return to 1d if input is 1d
        return final_hidden_states.squeeze(0) if is_input_1d else \
            final_hidden_states


class Qwen3MoeAttention(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        rope_theta: float = 10000,
        rope_scaling: Optional[dict[str, Any]] = None,
        max_position_embeddings: int = 8192,
        head_dim: Optional[int] = None,
        rms_norm_eps: float = 1e-06,
        qkv_bias: bool = False,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        dual_chunk_attention_config: Optional[dict[str, Any]] = None,
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
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings
        self.dual_chunk_attention_config = dual_chunk_attention_config

        self.qkv_proj = QKVParallelLinear(hidden_size,
                                          self.head_dim,
                                          self.total_num_heads,
                                          self.total_num_kv_heads,
                                          bias=qkv_bias,
                                          quant_config=quant_config,
                                          prefix=f"{prefix}.qkv_proj")

        self.o_proj = RowParallelLinear(self.total_num_heads * self.head_dim,
                                        hidden_size,
                                        bias=False,
                                        quant_config=quant_config,
                                        prefix=f"{prefix}.o_proj")

        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position_embeddings,
            base=rope_theta,
            rope_scaling=rope_scaling,
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
            } if dual_chunk_attention_config else {},
        )

        self.q_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        # Add qk-norm
        q_by_head = q.view(*q.shape[:-1], q.shape[-1] // self.head_dim,
                           self.head_dim)
        q_by_head = self.q_norm(q_by_head)
        q = q_by_head.view(q.shape)

        k_by_head = k.view(*k.shape[:-1], k.shape[-1] // self.head_dim,
                           self.head_dim)
        k_by_head = self.k_norm(k_by_head)
        k = k_by_head.view(k.shape)
        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v)
        output, _ = self.o_proj(attn_output)
        return output


class Qwen3MoeDecoderLayer(nn.Module):

    def __init__(self, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()

        config = vllm_config.model_config.hf_text_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config

        self.hidden_size = config.hidden_size
        rope_theta = getattr(config, "rope_theta", 10000)
        rope_scaling = getattr(config, "rope_scaling", None)
        max_position_embeddings = getattr(config, "max_position_embeddings",
                                          8192)
        dual_chunk_attention_config = getattr(config,
                                              "dual_chunk_attention_config",
                                              None)
        self.self_attn = Qwen3MoeAttention(
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            max_position_embeddings=max_position_embeddings,
            rms_norm_eps=config.rms_norm_eps,
            qkv_bias=getattr(config, 'attention_bias', False),
            head_dim=getattr(config, 'head_dim', None),
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.self_attn",
            dual_chunk_attention_config=dual_chunk_attention_config,
        )

        # `mlp_only_layers` in the config.
        layer_idx = extract_layer_index(prefix)
        self.layer_idx = int(layer_idx)
        mlp_only_layers = ([] if not hasattr(config, "mlp_only_layers") else
                           config.mlp_only_layers)
        if (layer_idx not in mlp_only_layers) and (
                config.num_experts > 0 and
            (layer_idx + 1) % config.decoder_sparse_step == 0):
            self.mlp = Qwen3MoeSparseMoeBlock(vllm_config=vllm_config,
                                              prefix=f"{prefix}.mlp")
        else:
            self.mlp = Qwen3MoeMLP(hidden_size=config.hidden_size,
                                   intermediate_size=config.intermediate_size,
                                   hidden_act=config.hidden_act,
                                   quant_config=quant_config,
                                   prefix=f"{prefix}.mlp")
        self.input_layernorm = RMSNorm(config.hidden_size,
                                       eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size,
                                                eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor],
        is_dummy: Optional[bool] = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # is_dummy = False # 强制关闭
        if is_dummy:
            # print("doing dummy eagle train here to overlap self attn")
            if residual is None:
                residual = hidden_states
        else:
            # if hidden_states.shape[0] == 32:
            #     self._attn_start.record()
            # Self Attention
            if residual is None:
                residual = hidden_states
                hidden_states = self.input_layernorm(hidden_states)
            else:
                hidden_states, residual = self.input_layernorm(
                    hidden_states, residual)
            attention_profile_range = _stage_decode_profile_range_start(
                _stage_decode_attention_profile_message(
                    self.layer_idx, getattr(self.mlp, "experts", self.mlp)))
            try:
                hidden_states = self.self_attn(
                    positions=positions,
                    hidden_states=hidden_states,
                )
            finally:
                _stage_decode_profile_range_end(attention_profile_range)
            hidden_states, residual = self.post_attention_layernorm(
            hidden_states, residual)
        # Fully Connected
        # if hidden_states.shape[0] == 32:
        #     self._attn_end.record()
        # Python wall-clock diagnostics cannot be traced by Dynamo. The
        # elastic ACLGraph path records the same fields in its graph-break
        # custom op, while eager execution keeps the original diagnostics.
        if not torch.compiler.is_compiling() and isinstance(
                self.mlp, Qwen3MoeSparseMoeBlock):
            self.mlp.experts.lossless_ffn_enter_wall_ts = time.perf_counter()
            self.mlp.experts.lossless_ffn_tokens = int(hidden_states.shape[0])
            self.mlp.experts.lossless_ffn_seq = int(
                getattr(self.mlp.experts, "lossless_ffn_seq", 0)) + 1
        hidden_states = self.mlp(hidden_states, is_dummy=is_dummy)
        # if hidden_states.shape[0] == 32:
        #     self._attn_end_moe.record()
        #     self._attn_end_moe.synchronize()
        #     attn_ms = self._attn_start.elapsed_time(self._attn_end)
        #     moe_ms = self._attn_end.elapsed_time(self._attn_end_moe)
        #     print("rank", self.ep_group.rank(), "layer_idx", self.layer_idx, "self attn ms:", attn_ms, "moe ms:", moe_ms)
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
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            quant_config=quant_config,
            prefix=f"{prefix}.embed_tokens")
        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: Qwen3MoeDecoderLayer(vllm_config=vllm_config,
                                                prefix=prefix),
            prefix=f"{prefix}.layers",
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.make_empty_intermediate_tensors = (
            make_empty_intermediate_tensors_factory(
                ["hidden_states", "residual"], config.hidden_size))

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        is_dummy: Optional[bool] = False,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.get_input_embeddings(input_ids)
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]
        for layer in islice(self.layers, self.start_layer, self.end_layer):
            hidden_states, residual = layer(positions, hidden_states, residual, is_dummy=is_dummy)
        if not get_pp_group().is_last_rank:
            return IntermediateTensors({
                "hidden_states": hidden_states,
                "residual": residual
            })
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        # Params for weights, fp8 weight scales, fp8 activation scales
        # (param_name, weight_name, expert_id, shard_id)
        return FusedMoE.make_expert_params_mapping(
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.config.num_experts,
            num_redundant_experts=self.num_redundant_experts)

    def load_weights(self, weights: Iterable[tuple[str,
                                                   torch.Tensor]]) -> set[str]:
        def _lossless_log_moe_load_snapshot(stage: str) -> None:
            try:
                ep_rank = get_ep_group().rank_in_group
            except Exception:
                ep_rank = -1

            sample = None
            for param_name, param in self.named_parameters():
                if (param_name.endswith("mlp.experts.w13_weight")
                        or param_name.endswith("mlp.experts.w2_weight")):
                    head = []
                    try:
                        num_rows = min(4, param.shape[0]) if param.ndim >= 1 else 0
                        for row_idx in range(num_rows):
                            head.append(
                                (row_idx,
                                 round(
                                     float(param[row_idx].float().abs().mean().item()),
                                     6)))
                    except Exception as exc:
                        head = [("error", str(exc))]
                    sample = {
                        "param_name": param_name,
                        "param_id": id(param),
                        "ptr": int(param.data.data_ptr()),
                        "shape": tuple(param.shape),
                        "device": str(param.device),
                        "head": head,
                    }
                    break

            pass  # debug log removed

        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]

        # Skip loading extra parameters for GPTQ/modelopt models.
        ignore_suffixes = (".bias", "_bias", ".k_scale", "_k_scale",
                           ".v_scale", "_v_scale", ".weight_scale",
                           "_weight_scale", ".input_scale", "_input_scale")

        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        expert_params_mapping = self.get_expert_mapping()
        lossless_reload_modules: list[tuple[nn.Module, object]] = []
        deep_diag = _mode1_load_weights_deep_diag_enabled()
        diag_step = os.getenv("VLLM_ASCEND_MODE1_CURRENT_UPDATE_STEP", "-1")
        diag_epoch = os.getenv("VLLM_ASCEND_MODE1_CURRENT_UPDATE_EPOCH", "-1")
        diag_rank = _mode1_current_rank_for_diag() if deep_diag else -1
        load_start_t = time.perf_counter()
        invalidate_s = 0.0
        post_process_s = 0.0
        stacked_s = 0.0
        expert_s = 0.0
        default_s = 0.0
        ignored_expert_s = 0.0
        unknown_skip_s = 0.0
        loaded_bytes = 0
        stacked_calls = 0
        expert_calls = 0
        default_calls = 0
        ignored_expert_count = 0
        unknown_skip_count = 0
        invalidated_count = 0
        post_process_count = 0
        slow_events: list[tuple[float, str]] = []

        def _record_slow(elapsed_s: float, tag: str) -> None:
            if not deep_diag:
                return
            threshold_s = float(os.getenv(
                "VLLM_ASCEND_MODE1_LOAD_WEIGHTS_SLOW_EVENT_THRESHOLD_S",
                "0.25") or "0.25")
            if elapsed_s < threshold_s and len(slow_events) >= 8:
                return
            slow_events.append((elapsed_s, tag))
            slow_events.sort(key=lambda item: item[0], reverse=True)
            del slow_events[8:]

        def _weight_nbytes(weight: torch.Tensor) -> int:
            try:
                return int(weight.numel()) * int(weight.element_size())
            except Exception:
                return 0

        _lossless_log_moe_load_snapshot("enter")
        invalidate_start_t = time.perf_counter()
        lossless_reload_module_ids: set[int] = set()

        def _append_lossless_reload_module(module: nn.Module,
                                           quant_method: object) -> None:
            module_id = id(module)
            if module_id in lossless_reload_module_ids:
                return
            lossless_reload_modules.append((module, quant_method))
            lossless_reload_module_ids.add(module_id)

        for module in self.modules():
            quant_method = getattr(module, "quant_method", None)
            invalidate_runtime = getattr(quant_method,
                                         "invalidate_lossless_runtime_state_for_reload",
                                         None)
            process_after_reload = getattr(quant_method,
                                           "process_weights_after_loading",
                                           None)
            mode1_lossless_module = (
                callable(process_after_reload)
                and getattr(module, "elastic_moe_mode", "lossy") == "lossless"
                and int(getattr(module, "elastic_execution_mode", 0)) == 1)
            if mode1_lossless_module:
                _append_lossless_reload_module(module, quant_method)
            if callable(invalidate_runtime):
                one_start_t = time.perf_counter()
                invalidated = invalidate_runtime(
                    layer=module, reason="Qwen3MoeModel.load_weights")
                one_elapsed_s = time.perf_counter() - one_start_t
                _record_slow(
                    one_elapsed_s,
                    f"invalidate:{module.__class__.__name__}:"
                    f"{getattr(module, 'layer_idx', '?')}")
                if invalidated:
                    invalidated_count += 1
                    _append_lossless_reload_module(module, quant_method)
        invalidate_s = time.perf_counter() - invalidate_start_t

        # Mode1 keeps a larger loaded-weight capacity for later shrink floors
        # (e.g. floor2 reserves 64 rows), but the rollout weight update itself
        # is a full-world primary update. If the reload-time loaded_expert_map
        # still points at a deeper-floor redundant layout, the expert loader can
        # leave primary prefix rows unwritten; later generate then reads zero
        # experts from slots 1..7. Pin the reload map to the primary full-world
        # prefix before any expert weights are copied, while preserving the
        # physical capacity for subsequent shrink imports.
        primary_reload_map_s = 0.0
        primary_reload_map_count = 0
        primary_reload_map_start_t = time.perf_counter()
        for module, _quant_method in lossless_reload_modules:
            use_primary_reload_map = getattr(
                module, "use_lossless_primary_reload_map_for_mode1", None)
            if not callable(use_primary_reload_map):
                continue
            one_start_t = time.perf_counter()
            try:
                changed = bool(use_primary_reload_map())
            except Exception:
                logger.exception(
                    "Mode1 failed to prepare primary reload map before "
                    "Qwen3MoeModel.load_weights: layer=%s",
                    getattr(module, "layer_idx", "?"))
                raise
            one_elapsed_s = time.perf_counter() - one_start_t
            _record_slow(
                one_elapsed_s,
                f"primary_reload_map:{module.__class__.__name__}:"
                f"{getattr(module, 'layer_idx', '?')}")
            if changed:
                primary_reload_map_count += 1
        primary_reload_map_s = time.perf_counter() - primary_reload_map_start_t
        if (primary_reload_map_count > 0 and os.getenv(
                "VLLM_ASCEND_MODE1_WEIGHT_RELOAD_CAPACITY_LOG", "1").lower()
                in ("1", "true", "yes", "on")):
            try:
                logger.info(
                    "Mode1 prepared primary reload maps before expert weight "
                    "load: modules=%s elapsed_s=%.3f step=%s epoch=%s",
                    primary_reload_map_count,
                    primary_reload_map_s,
                    diag_step,
                    diag_epoch,
                )
            except Exception:
                pass
        # print("vllm moe expert_params_mapping is", expert_params_mapping)
        for name, loaded_weight in weights:
            if deep_diag and isinstance(loaded_weight, torch.Tensor):
                loaded_bytes += _weight_nbytes(loaded_weight)
            for (param_name, weight_name, shard_id) in stacked_params_mapping:
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
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                one_start_t = time.perf_counter()
                if weight_loader == default_weight_loader:
                    weight_loader(param, loaded_weight)
                else:
                    weight_loader(param, loaded_weight, shard_id)
                elapsed_s = time.perf_counter() - one_start_t
                stacked_s += elapsed_s
                stacked_calls += 1
                _record_slow(elapsed_s, f"stacked:{name}")
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
                    if name_mapped.endswith(
                            ignore_suffixes
                    ) and name_mapped not in params_dict:
                        continue

                    param = params_dict[name_mapped]
                    # We should ask the weight loader to return success or not
                    # here since otherwise we may skip experts with other
                    # available replicas.
                    weight_loader = typing.cast(Callable[..., bool],
                                                param.weight_loader)
                    #print("moe weight_loader is", weight_loader)
                    # self.ep_group = get_ep_group().device_group
                    # if self.ep_group.rank() == 0:
                    #     print("name is", name, "param_name is", param_name)
                    #     print("ep_rank is ", self.ep_group.rank(), "load_weight is", loaded_weight, "name_mapped", name_mapped, "shard_id", shard_id, "expert_id", expert_id)
                    one_start_t = time.perf_counter()
                    success = weight_loader(param,
                                            loaded_weight,
                                            name_mapped,
                                            shard_id=shard_id,
                                            expert_id=expert_id,
                                            return_success=True)
                    elapsed_s = time.perf_counter() - one_start_t
                    expert_s += elapsed_s
                    expert_calls += 1
                    _record_slow(
                        elapsed_s,
                        f"expert:{name_mapped}:expert={expert_id}:"
                        f"shard={shard_id}")
                    if success:
                        name = name_mapped
                        break
                else:
                    if is_expert_weight:
                        # We've checked that this is an expert weight
                        # However it's not mapped locally to this rank
                        # So we simply skip it
                        if deep_diag:
                            ignored_expert_count += 1
                        continue

                    # Skip loading extra parameters for GPTQ/modelopt models.
                    if name.endswith(
                            ignore_suffixes) and name not in params_dict:
                        continue
                    # Skip layers on other devices.
                    if is_pp_missing_parameter(name, self):
                        continue
                    # Remapping the name of FP8 kv-scale.
                    if name.endswith("kv_scale"):
                        remapped_kv_scale_name = name.replace(
                            ".kv_scale", ".attn.kv_scale")
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
                    weight_loader = getattr(param, "weight_loader",
                                            default_weight_loader)
                    one_start_t = time.perf_counter()
                    weight_loader(param, loaded_weight)
                    elapsed_s = time.perf_counter() - one_start_t
                    default_s += elapsed_s
                    default_calls += 1
                    _record_slow(elapsed_s, f"default:{name}")
            loaded_params.add(name)

        for module, quant_method in lossless_reload_modules:
            process_after_reload = getattr(quant_method,
                                           "process_weights_after_loading",
                                           None)
            if callable(process_after_reload):
                one_start_t = time.perf_counter()
                process_after_reload(module)
                elapsed_s = time.perf_counter() - one_start_t
                post_process_s += elapsed_s
                post_process_count += 1
                _record_slow(
                    elapsed_s,
                    f"post_process:{module.__class__.__name__}:"
                    f"{getattr(module, 'layer_idx', '?')}")

        if deep_diag:
            total_s = time.perf_counter() - load_start_t
            slow_summary = "; ".join(
                f"{tag}={elapsed_s:.3f}s" for elapsed_s, tag in slow_events)
            logger.info(
                "Mode1 load_weights deep timing: rank=%s step=%s epoch=%s "
                "total_s=%.3f invalidate_s=%.3f invalidated=%s "
                "stacked_s=%.3f stacked_calls=%s default_s=%.3f "
                "default_calls=%s expert_s=%.3f expert_calls=%s "
                "ignored_expert_count=%s ignored_expert_s=%.3f "
                "unknown_skip_count=%s unknown_skip_s=%.3f "
                "primary_reload_map_s=%.3f primary_reload_maps=%s "
                "post_process_s=%.3f post_process_count=%s "
                "loaded_params=%s input_bytes=%s slow=%s",
                diag_rank,
                diag_step,
                diag_epoch,
                total_s,
                invalidate_s,
                invalidated_count,
                stacked_s,
                stacked_calls,
                default_s,
                default_calls,
                expert_s,
                expert_calls,
                ignored_expert_count,
                ignored_expert_s,
                unknown_skip_count,
                unknown_skip_s,
                primary_reload_map_s,
                primary_reload_map_count,
                post_process_s,
                post_process_count,
                len(loaded_params),
                loaded_bytes,
                slow_summary,
            )

        _lossless_log_moe_load_snapshot("exit")
        return loaded_params


class Qwen3MoeForCausalLM(nn.Module, SupportsPP, SupportsLoRA,
                          MixtureOfExperts):
    packed_modules_mapping = {
        "qkv_proj": [
            "q_proj",
            "k_proj",
            "v_proj",
        ],
        "gate_up_proj": [
            "gate_proj",
            "up_proj",
        ],
    }

    fall_back_to_pt_during_load = False

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_text_config
        quant_config = vllm_config.quant_config
        self.config = config
        self.quant_config = quant_config
        self.model = Qwen3MoeModel(vllm_config=vllm_config,
                                   prefix=maybe_prefix(prefix, "model"))
        self.lm_head = ParallelLMHead(config.vocab_size,
                                      config.hidden_size,
                                      quant_config=quant_config,
                                      prefix=maybe_prefix(prefix, "lm_head"))
        if self.config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight
        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors)

        # Set MoE hyperparameters
        self.expert_weights = []

        self.moe_layers: list[FusedMoE] = []
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

    def set_eplb_state(
        self,
        expert_load_view: torch.Tensor,
        logical_to_physical_map: torch.Tensor,
        logical_replica_count: torch.Tensor,
    ) -> None:
        for layer_idx, layer in enumerate(self.moe_layers):
            # Register the expert weights.
            self.expert_weights.append(layer.get_expert_weights())
            layer.set_eplb_state(
                moe_layer_idx=layer_idx,
                expert_load_view=expert_load_view,
                logical_to_physical_map=logical_to_physical_map,
                logical_replica_count=logical_replica_count,
            )

    def update_physical_experts_metadata(
        self,
        num_physical_experts: int,
        num_local_physical_experts: int,
    ) -> None:
        assert self.num_local_physical_experts == num_local_physical_experts
        self.num_physical_experts = num_physical_experts
        self.num_local_physical_experts = num_local_physical_experts
        self.num_redundant_experts = (num_physical_experts -
                                      self.num_logical_experts)
        for layer in self.model.layers:
            if isinstance(layer.mlp, Qwen3MoeSparseMoeBlock):
                moe = layer.mlp
                moe.n_local_physical_experts = num_local_physical_experts
                moe.n_physical_experts = num_physical_experts
                moe.n_redundant_experts = self.num_redundant_experts
                moe.experts.update_expert_map()

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.get_input_embeddings(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        is_dummy: Optional[bool] = False,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        hidden_states = self.model(input_ids, positions, intermediate_tensors,
                                   inputs_embeds, is_dummy=is_dummy)
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        logits = self.logits_processor(self.lm_head, hidden_states)
        return logits

    def load_weights(self, weights: Iterable[tuple[str,
                                                   torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights)

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        return self.model.get_expert_mapping()
