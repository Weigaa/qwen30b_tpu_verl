# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2024 The Qwen team.
# Copyright 2023 The vLLM team.
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved. Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
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
# Adapted from vllm/model_executor/models/qwen3_moe.py
# This file is a part of the vllm-ascend project.

import os
from typing import Optional, Tuple

import torch
from torch import nn
from transformers import PretrainedConfig
from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, CompilationLevel, VllmConfig
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.logger import init_logger
from vllm.distributed.parallel_state import (get_dp_group, get_ep_group,
                                             get_tp_group)
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.fused_moe.layer import FusedMoE
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead, VocabParallelEmbedding)
from vllm.model_executor.models.interfaces import (MixtureOfExperts,
                                                   SupportsLoRA, SupportsPP)
from vllm.model_executor.models.qwen3_moe import (Qwen3MoeAttention,
                                                  Qwen3MoeDecoderLayer,
                                                  Qwen3MoeForCausalLM,
                                                  Qwen3MoeMLP, Qwen3MoeModel,
                                                  Qwen3MoeSparseMoeBlock)
from vllm.model_executor.models.utils import (
    PPMissingLayer, extract_layer_index,
    make_empty_intermediate_tensors_factory, make_layers, maybe_prefix)

from vllm_ascend.ops.fused_moe import AscendFusedMoE
from vllm_ascend.draft.draft_trainer import DraftTrainer, build_draft_trainer
from vllm_ascend.utils import vllm_version_is
#新增
from vllm.forward_context import get_forward_context
from vllm.utils.moe_stats import moe_stats
from vllm.distributed.parallel_state import get_ep_group
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
import torch.distributed as dist
import time

logger = init_logger(__name__)


def _env_flag(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).lower() in ("1", "true", "yes", "on")


def _custom_mode1_debug_enabled() -> bool:
    return _env_flag("VLLM_ASCEND_CUSTOM_MODE1_DEBUG", "0")


def _custom_mode1_timing_events_enabled() -> bool:
    return _env_flag("VLLM_ASCEND_CUSTOM_MODE1_TIMING_EVENTS", "0")


class CustomSparseMoeBlock(Qwen3MoeSparseMoeBlock):

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        nn.Module.__init__(self)
        self.tp_size = get_tensor_model_parallel_world_size()
        self.prefix = prefix
        self.layer_idx = extract_layer_index(prefix) 
        if self.tp_size > config.num_experts:
            raise ValueError(
                f"Tensor parallel size {self.tp_size} is greater than "
                f"the number of experts {config.num_experts}.")

        self.gate = ReplicatedLinear(
            config.hidden_size,
            config.num_experts,
            bias=False,
            quant_config=None,
            prefix=f"{prefix}.gate",
        )

        self.experts = AscendFusedMoE(
            num_experts=config.num_experts,
            top_k=config.num_experts_per_tok,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            reduce_results=False,
            renormalize=config.norm_topk_prob,
            quant_config=quant_config,
            prefix=f"{prefix}.experts",
            layer_idx=self.layer_idx
        )

        self.top_k = config.num_experts_per_tok

        self.dp_size = get_dp_group().world_size

        self.tp_group = get_tp_group().device_group
        self.tp_rank = get_tp_group().rank_in_group
        self.ep_group = get_ep_group()

        self.params_dtype = torch.get_default_dtype()
        self.prefix = prefix
        # Keep pre-shrink logging focused on the MoE output path; gate and
        # mapping summaries are too noisy for the current kernel-input check.
        self._pre_shrink_live_gate_debug_logged = True
        self._pre_shrink_live_gate_debug_remaining = 0
        self._pre_shrink_live_mapping_debug_logged = True
        self._pre_shrink_live_moe_io_debug_remaining = 0

    def forward(
        self,
        hidden_states,
        is_dummy: bool = False,
        attn_metadata=None,
    ):
        if attn_metadata is None:
            attn_metadata = get_forward_context().attn_metadata
        # when profile runs, force experts to load balanced tokens
        # to avoid high memory consumption on a single rank.
        enable_force_load_balance = get_forward_context().in_profile_run
        is_prefill = get_forward_context().with_prefill

        # router_logits: (num_tokens, n_experts)
        router_logits, _ = self.gate(hidden_states)
        forward_context = get_forward_context()
        pre_shrink_loaded_only = bool(
            (_custom_mode1_debug_enabled()
             or _custom_mode1_timing_events_enabled())
            and not getattr(forward_context, "in_profile_run", False)
            and not is_dummy
            and self.layer_idx == 0
            and getattr(self.experts,
                        "lossless_zero_redundancy_preallocated_loaded", False)
            and getattr(self.experts, "loaded_weight_capacity", 0) >
            getattr(self.experts, "active_local_num_experts", 0))
        if pre_shrink_loaded_only:
            topk_ids = self.compute_topk(router_logits, k=self.top_k)
            hidden_fp32 = hidden_states.float()
            logits_fp32 = router_logits.float()
            hidden_norm = hidden_fp32.norm(dim=-1)
            top1_vals, _ = torch.topk(logits_fp32, k=min(2, logits_fp32.shape[-1]), dim=-1)
            top1 = top1_vals[:, 0]
            top2 = top1_vals[:, 1] if top1_vals.shape[-1] > 1 else torch.zeros_like(top1)
            unique_ids, unique_counts = torch.unique(topk_ids.reshape(-1),
                                                     return_counts=True)
            top_count = min(8, unique_counts.numel())
            if top_count > 0:
                top_vals, top_idx = torch.topk(unique_counts, k=top_count)
                top_pairs = [(int(unique_ids[idx].item()), int(val.item()))
                             for idx, val in zip(top_idx, top_vals)]
            else:
                top_pairs = []
            top1_ids = topk_ids[:, 0]
            top2_ids = topk_ids[:, 1] if topk_ids.shape[-1] > 1 else top1_ids
            top1_unique_ids, top1_counts = torch.unique(top1_ids, return_counts=True)
            top2_unique_ids, top2_counts = torch.unique(top2_ids, return_counts=True)

            def _top_pairs(ids: torch.Tensor, counts: torch.Tensor) -> list[tuple[int, int]]:
                if counts.numel() == 0:
                    return []
                top_n = min(8, counts.numel())
                vals, idxs = torch.topk(counts, k=top_n)
                return [(int(ids[idx].item()), int(val.item()))
                        for idx, val in zip(idxs, vals)]

            if not self._pre_shrink_live_mapping_debug_logged:
                active_ids = getattr(self.experts, "active_expert_ids", None)
                expert_map = getattr(self.experts, "expert_map", None)
                log2phy = (getattr(self.experts, "elastic_runtime_log2phy", None)
                           if getattr(self.experts, "elastic_runtime_log2phy",
                                      None) is not None else
                           getattr(self.experts, "log2phy", None))
                runtime_w13 = getattr(self.experts, "runtime_w13_weight", None)
                runtime_w2 = getattr(self.experts, "runtime_w2_weight", None)
                loaded_w13 = getattr(self.experts, "w13_weight", None)
                loaded_w2 = getattr(self.experts, "w2_weight", None)
                compute_w13 = runtime_w13 if runtime_w13 is not None else loaded_w13
                compute_w2 = runtime_w2 if runtime_w2 is not None else loaded_w2
                compute_source = ("runtime_weight_view"
                                  if runtime_w13 is not None else "loaded_weight")

                active_ids_list = []
                mapping_pairs = []
                row_summaries = []
                inactive_row_summaries = []
                active_nonzero_slots = 0
                inactive_nonzero_slots = 0
                if active_ids is not None:
                    if isinstance(active_ids, torch.Tensor):
                        active_ids_list = [int(x) for x in active_ids.detach().cpu().tolist()]
                    else:
                        active_ids_list = [int(x) for x in active_ids]

                if expert_map is not None and active_ids_list:
                    if isinstance(expert_map, torch.Tensor):
                        expert_map_cpu = expert_map.detach().cpu()
                        for expert_id in active_ids_list:
                            slot = int(expert_map_cpu[expert_id].item())
                            mapping_pairs.append((expert_id, slot))
                    else:
                        for expert_id in active_ids_list:
                            slot = int(expert_map[expert_id])
                            mapping_pairs.append((expert_id, slot))

                if compute_w13 is not None and mapping_pairs:
                    for expert_id, slot in mapping_pairs[:4] + mapping_pairs[-4:]:
                        if slot < 0 or slot >= compute_w13.shape[0]:
                            row_abs_mean = float("nan")
                        else:
                            row_abs_mean = float(compute_w13[slot].float().abs().mean().item())
                        row_summaries.append((expert_id, slot, round(row_abs_mean, 6)))
                    capacity = int(compute_w13.shape[0])
                    active_cap = min(len(mapping_pairs), capacity)
                    for slot in range(active_cap):
                        if float(compute_w13[slot].float().abs().max().item()) > 0.0:
                            active_nonzero_slots += 1
                    for slot in range(active_cap, capacity):
                        max_abs = float(compute_w13[slot].float().abs().max().item())
                        if max_abs > 0.0:
                            inactive_nonzero_slots += 1
                        if len(inactive_row_summaries) < 4:
                            inactive_row_summaries.append((slot, round(max_abs, 6)))
                compute_head_samples = []
                if compute_w13 is not None and compute_w2 is not None:
                    for row in range(min(4, int(compute_w13.shape[0]))):
                        compute_head_samples.append((
                            row,
                            round(float(compute_w13[row].float().abs().mean().item()), 6),
                            round(float(compute_w2[row].float().abs().mean().item()), 6),
                        ))

                identity_mismatches = []
                if log2phy is not None:
                    if isinstance(log2phy, torch.Tensor):
                        log2phy_cpu = log2phy.detach().cpu()
                        sample_n = min(128, int(log2phy_cpu.numel()))
                        for expert_id in range(sample_n):
                            mapped = int(log2phy_cpu[expert_id].item())
                            if mapped != expert_id:
                                identity_mismatches.append((expert_id, mapped))
                                if len(identity_mismatches) >= 8:
                                    break

                logger.info(
                    "Lossless pre-shrink expert mapping summary: rank=%s layer=%s "
                    "active_ids_head=%s active_ids_tail=%s mapping_head=%s "
                    "mapping_tail=%s row_abs_mean_samples=%s "
                    "active_nonzero_slots=%s inactive_nonzero_slots=%s "
                    "inactive_row_samples=%s compute_source=%s "
                    "compute_head_samples=%s "
                    "log2phy_identity_mismatch_count=%s mismatch_samples=%s",
                    self.ep_group.rank_in_group,
                    self.layer_idx,
                    active_ids_list[:4],
                    active_ids_list[-4:],
                    mapping_pairs[:4],
                    mapping_pairs[-4:],
                    row_summaries,
                    active_nonzero_slots,
                    inactive_nonzero_slots,
                    inactive_row_summaries,
                    compute_source,
                    compute_head_samples,
                    len(identity_mismatches),
                    identity_mismatches,
                )
                self._pre_shrink_live_mapping_debug_logged = True

            if not self._pre_shrink_live_gate_debug_logged:
                logger.info(
                    "Lossless pre-shrink gate input summary: rank=%s layer=%s "
                    "tokens=%s hidden_shape=%s hidden_abs_mean=%.6f "
                    "hidden_norm_mean=%.6f hidden_norm_max=%.6f "
                    "hidden_min=%.6f hidden_max=%.6f",
                    self.ep_group.rank_in_group,
                    self.layer_idx,
                    hidden_states.shape[0],
                    tuple(hidden_states.shape),
                    float(hidden_fp32.abs().mean().item()),
                    float(hidden_norm.mean().item()),
                    float(hidden_norm.max().item()),
                    float(hidden_fp32.min().item()),
                    float(hidden_fp32.max().item()),
                )
                logger.info(
                    "Lossless pre-shrink gate output summary: rank=%s layer=%s "
                    "logits_shape=%s logits_abs_mean=%.6f logits_std=%.6f "
                    "logits_min=%.6f logits_max=%.6f top1_mean=%.6f "
                    "top1_max=%.6f top1_gap_mean=%.6f top1_gap_min=%.6f "
                    "topk_min=%s topk_max=%s topk_unique=%s topk_top=%s",
                    self.ep_group.rank_in_group,
                    self.layer_idx,
                    tuple(router_logits.shape),
                    float(logits_fp32.abs().mean().item()),
                    float(logits_fp32.std().item()),
                    float(logits_fp32.min().item()),
                    float(logits_fp32.max().item()),
                    float(top1.mean().item()),
                    float(top1.max().item()),
                    float((top1 - top2).mean().item()),
                    float((top1 - top2).min().item()),
                    int(topk_ids.min().item()),
                    int(topk_ids.max().item()),
                    int(unique_ids.numel()),
                    top_pairs,
                )
                self._pre_shrink_live_gate_debug_logged = True
            if self._pre_shrink_live_gate_debug_remaining > 0:
                call_idx = 7 - self._pre_shrink_live_gate_debug_remaining
                top1_pairs = _top_pairs(top1_unique_ids, top1_counts)
                top2_pairs = _top_pairs(top2_unique_ids, top2_counts)
                logger.info(
                    "Lossless pre-shrink gate evolution: rank=%s layer=%s "
                    "call=%s tokens=%s hidden_abs_mean=%.6f "
                    "hidden_norm_mean=%.6f hidden_norm_max=%.6f "
                    "logits_abs_mean=%.6f logits_std=%.6f "
                    "top1_unique=%s top1_top=%s top2_unique=%s top2_top=%s "
                    "topk_unique=%s topk_top=%s gap_mean=%.6f gap_min=%.6f",
                    self.ep_group.rank_in_group,
                    self.layer_idx,
                    call_idx,
                    hidden_states.shape[0],
                    float(hidden_fp32.abs().mean().item()),
                    float(hidden_norm.mean().item()),
                    float(hidden_norm.max().item()),
                    float(logits_fp32.abs().mean().item()),
                    float(logits_fp32.std().item()),
                    int(top1_unique_ids.numel()),
                    top1_pairs,
                    int(top2_unique_ids.numel()),
                    top2_pairs,
                    int(unique_ids.numel()),
                    top_pairs,
                    float((top1 - top2).mean().item()),
                    float((top1 - top2).min().item()),
                )
                if int(unique_ids.numel()) <= max(self.top_k, 8):
                    logger.warning(
                        "Lossless pre-shrink gate collapse suspected: rank=%s "
                        "layer=%s call=%s topk_unique=%s top1_top=%s top2_top=%s",
                        self.ep_group.rank_in_group,
                        self.layer_idx,
                        call_idx,
                        int(unique_ids.numel()),
                        top1_pairs,
                        top2_pairs,
                    )
                self._pre_shrink_live_gate_debug_remaining -= 1
        #NPU并没有走到这个方法,eager会走到，图模式不会
        # add record
        # topk_ids = torch.topk(router_logits, k=self.top_k, dim=-1).indices
        # moe_stats.record_layer_topk(self.layer_idx, topk_ids)
        # add record
        # topk_ids = self.compute_topk(router_logits)
        # self.layer_idx = extract_layer_index(self.prefix)
        # self._ep_same_input_guard(topk_ids, self.layer_idx, note=f"(run={getattr(self,'total_run',-1)})")
        # moe_stats.record(
        #     layer_idx=extract_layer_index(self.prefix),
        #     topk_ids=topk_ids,
        #     num_experts=128,
        # )
        # print("do record in vllm_ascend")
        # print("is_dummy in customsparsemoeblock:", is_dummy)

        hidden_states = self.experts(
            hidden_states=hidden_states,
            router_logits=router_logits,
            is_prefill=is_prefill,
            top_k=self.top_k,
            enable_force_load_balance=enable_force_load_balance,
            shared_experts=None,
            is_dummy=is_dummy,
        )

        if pre_shrink_loaded_only and self._pre_shrink_live_moe_io_debug_remaining > 0:
            output_fp32 = hidden_states.float()
            output_norm = output_fp32.norm(dim=-1)
            delta = output_fp32 - hidden_fp32
            delta_norm = delta.norm(dim=-1)
            call_idx = 7 - self._pre_shrink_live_moe_io_debug_remaining
            logger.info(
                "Lossless pre-shrink moe io summary: rank=%s layer=%s call=%s "
                "input_abs_mean=%.6f input_norm_mean=%.6f input_norm_max=%.6f "
                "output_abs_mean=%.6f output_norm_mean=%.6f output_norm_max=%.6f "
                "delta_abs_mean=%.6f delta_norm_mean=%.6f delta_norm_max=%.6f "
                "output_min=%.6f output_max=%.6f",
                self.ep_group.rank_in_group,
                self.layer_idx,
                call_idx,
                float(hidden_fp32.abs().mean().item()),
                float(hidden_norm.mean().item()),
                float(hidden_norm.max().item()),
                float(output_fp32.abs().mean().item()),
                float(output_norm.mean().item()),
                float(output_norm.max().item()),
                float(delta.abs().mean().item()),
                float(delta_norm.mean().item()),
                float(delta_norm.max().item()),
                float(output_fp32.min().item()),
                float(output_fp32.max().item()),
            )
            self._pre_shrink_live_moe_io_debug_remaining -= 1

        return hidden_states


class CustomQwen3MoeDecoderLayer(Qwen3MoeDecoderLayer):

    def __init__(
        self,
        config: PretrainedConfig,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        vllm_config: Optional[VllmConfig] = None,
        prefix: str = "",
    ) -> None:

        nn.Module.__init__(self)
        self.hidden_size = config.hidden_size
        rope_theta = getattr(config, "rope_theta", 10000)
        rope_scaling = getattr(config, "rope_scaling", None)
        max_position_embeddings = getattr(config, "max_position_embeddings",
                                          8192)
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
        )

        # `mlp_only_layers` in the config.
        layer_idx = extract_layer_index(prefix)
        self.layer_idx = layer_idx
        mlp_only_layers = ([] if not hasattr(config, "mlp_only_layers") else
                           config.mlp_only_layers)
        self.use_aclgraph = (vllm_config is not None
                             and vllm_config.compilation_config.level
                             == CompilationLevel.PIECEWISE
                             and not vllm_config.model_config.enforce_eager)
        if (layer_idx not in mlp_only_layers) and (
                config.num_experts > 0 and
            (layer_idx + 1) % config.decoder_sparse_step == 0):
            if not self.use_aclgraph:
                # FIXME: custom sparse moe block doesn't work with aclgraph.
                self.mlp = CustomSparseMoeBlock(config=config,
                                                quant_config=quant_config,
                                                prefix=f"{prefix}.mlp")
            else:
                if vllm_version_is("0.10.2"):
                    self.mlp = Qwen3MoeSparseMoeBlock(
                        config=config,
                        quant_config=quant_config,
                        prefix=f"{prefix}.mlp")
                else:
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
        ###新增参数
        self._attn_start = torch.npu.Event(enable_timing=True)
        self._attn_end   = torch.npu.Event(enable_timing=True)
        self._attn_end_moe   = torch.npu.Event(enable_timing=True)
        self.ep_group = get_ep_group().device_group
        self.draft_trainer: Optional[DraftTrainer] = None

    def set_draft_trainer(self, draft_trainer: Optional[DraftTrainer]) -> None:
        self.draft_trainer = draft_trainer

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor],
        is_dummy: Optional[bool] = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if is_dummy and self.draft_trainer is not None:
            self.draft_trainer.maybe_train_step(
                layer_idx=self.layer_idx,
                hidden_states=hidden_states,
                positions=positions,
            )
        return super().forward(positions, hidden_states, residual, is_dummy)


@support_torch_compile
class CustomQwen3MoeModel(Qwen3MoeModel):

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        nn.Module.__init__(self)
        config = vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config
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
            prefix=f"{prefix}.embed_tokens")
        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: CustomQwen3MoeDecoderLayer(
                config=config,
                cache_config=cache_config,
                quant_config=quant_config,
                vllm_config=vllm_config,
                prefix=prefix),
            prefix=f"{prefix}.layers",
        )
        # Initialize once at model build time to avoid first-use latency.
        self.draft_trainer = build_draft_trainer(self)
        for layer in self.layers:
            if isinstance(layer, CustomQwen3MoeDecoderLayer):
                layer.set_draft_trainer(self.draft_trainer)
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.make_empty_intermediate_tensors = (
            make_empty_intermediate_tensors_factory(
                ["hidden_states", "residual"], config.hidden_size))


class CustomQwen3MoeForCausalLM(Qwen3MoeForCausalLM):

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        nn.Module.__init__(self)
        SupportsPP.__init__(self)
        SupportsLoRA.__init__(self)
        MixtureOfExperts.__init__(self)
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        self.config = config
        self.quant_config = quant_config
        self.model = CustomQwen3MoeModel(vllm_config=vllm_config,
                                         prefix=maybe_prefix(prefix, "model"))
        self.lm_head = ParallelLMHead(config.vocab_size,
                                      config.hidden_size,
                                      quant_config=quant_config,
                                      prefix=maybe_prefix(prefix, "lm_head"))
        if self.config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight
        if getattr(self.model, "draft_trainer", None) is not None:
            self.model.draft_trainer.bind_target_layers(
                embed_tokens=self.model.embed_tokens,
                lm_head=self.lm_head,
            )
        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors)

        # Set MoE hyperparameters
        self.expert_weights: list[torch.Tensor] = []

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
