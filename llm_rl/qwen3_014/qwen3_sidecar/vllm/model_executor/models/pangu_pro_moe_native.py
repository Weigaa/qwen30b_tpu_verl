# SPDX-License-Identifier: Apache-2.0
"""Legacy PanGu-Pro-MoE checkpoint on the v0.14 native attention/MoE path.

Reuse Qwen2MoE's matching biased QKV, RoPE, RMSNorm and checkpoint layout.
Preserve PanGu's biased attention output, ungated shared expert, one expert
per group, learned per-expert router scale and unnormalized routing weights.
No approximated expert voting, checkpoint conversion or graph/eager fallback.
"""
import torch
from torch import nn

from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.model_executor.layers.fused_moe import SharedFusedMoE
from vllm.model_executor.layers.linear import ReplicatedLinear, RowParallelLinear
from vllm.transformers_utils.config import set_default_rope_theta
from . import qwen2_moe as base


def pangu_grouped_route(logits, hidden_dtype, scale):
    """Match the model-provided HF reference, including first-index tie breaks."""
    probabilities = torch.softmax(logits, dim=-1, dtype=torch.float32)
    weights, indices = probabilities.reshape(-1, 8, 8).max(dim=-1)
    offsets = torch.arange(0, 64, 8, device=logits.device, dtype=torch.int64)
    indices = indices + offsets
    # The reference casts softmax scores before multiplying router_scale.
    weights = weights.to(hidden_dtype) * scale.reshape(-1)[indices]
    return weights, indices.to(torch.int32)


class PanguSparseMoE(nn.Module):
    def __init__(self, config, quant_config=None, prefix=''):
        super().__init__()
        if config.num_experts != 64 or config.num_experts_per_tok != 8:
            raise ValueError('This adapter requires the original PanGu 8x8 expert groups')
        self.tp_size = get_tensor_model_parallel_world_size()
        self.router_scale = nn.Parameter(torch.ones(1, 64))
        self.gate = ReplicatedLinear(config.hidden_size, 64, bias=False,
                                     quant_config=None, prefix=f'{prefix}.gate')
        self.shared_expert = base.Qwen2MoeMLP(config.hidden_size,
            config.shared_expert_intermediate_size, config.hidden_act,
            quant_config=quant_config, reduce_results=False, expert_gate=None,
            prefix=f'{prefix}.shared_expert')
        self.experts = SharedFusedMoE(shared_experts=self.shared_expert,
            num_experts=64, top_k=8, hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size, reduce_results=False,
            renormalize=False, quant_config=quant_config,
            custom_routing_function=self.route, prefix=f'{prefix}.experts')

    def route(self, hidden_states, gating_output, topk, renormalize, **kwargs):
        if topk != 8 or renormalize:
            raise ValueError('PanGu router semantics changed')
        return pangu_grouped_route(gating_output, hidden_states.dtype, self.router_scale)

    def forward(self, hidden_states):
        shape = hidden_states.shape
        hidden = hidden_states.reshape(-1, shape[-1])
        logits, _ = self.gate(hidden)
        shared, routed = self.experts(hidden_states=hidden, router_logits=logits)
        result = shared + routed
        if self.tp_size > 1:
            result = self.experts.maybe_all_reduce_tensor_model_parallel(result)
        return result.reshape(shape)


class PanguAttention(base.Qwen2MoeAttention):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.o_proj = RowParallelLinear(self.total_num_heads*self.head_dim,
            self.hidden_size, bias=True, quant_config=kwargs.get('quant_config'),
            prefix=f"{kwargs.get('prefix', '')}.o_proj")


class PanguProMoEForCausalLM(base.Qwen2MoeForCausalLM):
    def __init__(self, *, vllm_config, prefix=''):
        config = vllm_config.model_config.hf_config
        config.decoder_sparse_step = 1
        config.mlp_only_layers = []
        config.norm_topk_prob = False
        config.intermediate_size = config.moe_intermediate_size
        set_default_rope_theta(config, config.rope_theta)
        # Construction is process-local, and both substitutions are restored
        # before any other model can be instantiated in this process.
        original_moe, original_attention = base.Qwen2MoeSparseMoeBlock, base.Qwen2MoeAttention
        try:
            base.Qwen2MoeSparseMoeBlock = PanguSparseMoE
            base.Qwen2MoeAttention = PanguAttention
            super().__init__(vllm_config=vllm_config, prefix=prefix)
        finally:
            base.Qwen2MoeSparseMoeBlock = original_moe
            base.Qwen2MoeAttention = original_attention
