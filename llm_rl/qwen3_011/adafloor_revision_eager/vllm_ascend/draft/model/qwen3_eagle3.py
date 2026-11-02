import math
from typing import Optional, Tuple, Union

import torch
from torch import nn
import torch.nn.functional as F
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3Model as Qwen3ModelTF,
    Qwen3RotaryEmbedding,
    Qwen3MLP,
    Qwen3Attention,
    Qwen3RMSNorm,
    Qwen3Config,
)
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.processing_utils import Unpack
from transformers.utils import logging

logger = logging.get_logger(__name__)

def _as_non_inference(tensor: torch.Tensor) -> torch.Tensor:
    if not tensor.is_inference():
        return tensor
    with torch.inference_mode(False):
        out = torch.empty_like(tensor)
        out.copy_(tensor)
    return out


class LoRALinear(nn.Module):
    """A minimal LoRA wrapper over an existing nn.Linear."""

    def __init__(
        self,
        base_linear: nn.Linear,
        rank: int,
        alpha: float,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError(f"LoRA rank must be > 0, got {rank}")

        self.base_linear = base_linear
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.rank
        self.dropout = nn.Dropout(p=float(dropout)) if dropout > 0 else nn.Identity()

        self.lora_a = nn.Linear(base_linear.in_features, self.rank, bias=False)
        self.lora_b = nn.Linear(self.rank, base_linear.out_features, bias=False)

        nn.init.kaiming_uniform_(self.lora_a.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b.weight)

        # Freeze base linear weights; train LoRA adapters only.
        self.base_linear.weight.requires_grad = False
        if self.base_linear.bias is not None:
            self.base_linear.bias.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base_linear(x)
        lora_out = self.lora_b(self.lora_a(self.dropout(x))) * self.scaling
        return base_out + lora_out


def apply_lora_to_linear_modules(
    module: nn.Module,
    rank: int,
    alpha: float,
    dropout: float = 0.0,
    target_module_names: Optional[set[str]] = None,
) -> int:
    """Replace target nn.Linear modules with LoRALinear wrappers."""
    replaced = 0
    for child_name, child in module.named_children():
        if isinstance(child, nn.Linear):
            should_wrap = (
                target_module_names is None or child_name in target_module_names)
            if should_wrap:
                setattr(
                    module,
                    child_name,
                    LoRALinear(
                        base_linear=child,
                        rank=rank,
                        alpha=alpha,
                        dropout=dropout,
                    ),
                )
                replaced += 1
                continue
        replaced += apply_lora_to_linear_modules(
            child,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
            target_module_names=target_module_names,
        )
    return replaced


class Qwen3DecoderLayer(nn.Module):
    def __init__(self, config: Qwen3Config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)

        self.self_attn = Qwen3Attention(config=config, layer_idx=layer_idx)

        # NOTE: Override the qkv projection for Eagle-3, Qwen 3 disenables bias by default
        self.self_attn.q_proj = nn.Linear(
            config.hidden_size * 2, config.num_attention_heads * self.head_dim, bias=config.attention_bias
        )
        self.self_attn.k_proj = nn.Linear(
            config.hidden_size * 2, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.self_attn.v_proj = nn.Linear(
            config.hidden_size * 2, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )

        self.mlp = Qwen3MLP(config)
        self.input_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # NOTE: Add a hidden_norm for Eagle-3
        self.hidden_norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_embeds: torch.Tensor,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        residual = hidden_states

        input_embeds = _as_non_inference(input_embeds)
        hidden_states = _as_non_inference(hidden_states)

        input_embeds = self.input_layernorm(input_embeds)
        hidden_states = self.hidden_norm(hidden_states)

        # NOTE: Concatenate the input_embeds and hidden_states for Eagle-3
        hidden_states = torch.cat([input_embeds, hidden_states], dim=-1)

        chunk_size = int(getattr(self.self_attn.config, "draft_attn_chunk_size", 0))

        # Self Attention
        if (chunk_size > 0 and hidden_states.shape[1] > chunk_size
                and past_key_value is None and not output_attentions):
            total_len = hidden_states.shape[1]
            running_cache = DynamicCache()
            if cache_position is None:
                cache_position = torch.arange(
                    total_len, device=hidden_states.device, dtype=torch.long)
            chunk_outputs = []
            for start in range(0, total_len, chunk_size):
                end = min(start + chunk_size, total_len)
                attn_mask_chunk = attention_mask
                if attention_mask is not None:
                    if attention_mask.dim() == 4:
                        attn_mask_chunk = attention_mask[:, :, start:end, :end]
                    elif attention_mask.dim() == 2:
                        attn_mask_chunk = attention_mask[:, :end]
                cache_pos_chunk = cache_position[start:end]
                pos_emb_chunk = position_embeddings
                if (position_embeddings is not None
                        and isinstance(position_embeddings, tuple)
                        and len(position_embeddings) == 2):
                    cos, sin = position_embeddings
                    if (isinstance(cos, torch.Tensor)
                            and isinstance(sin, torch.Tensor)
                            and cos.dim() >= 2 and sin.dim() >= 2
                            and cos.shape[1] >= end and sin.shape[1] >= end):
                        pos_emb_chunk = (
                            cos[:, start:end, :],
                            sin[:, start:end, :],
                        )
                chunk_out, _ = self.self_attn(
                    hidden_states=hidden_states[:, start:end, :],
                    attention_mask=attn_mask_chunk,
                    past_key_value=running_cache,
                    output_attentions=False,
                    use_cache=True,
                    cache_position=cache_pos_chunk,
                    position_embeddings=pos_emb_chunk,
                    **kwargs,
                )
                chunk_outputs.append(chunk_out)
            hidden_states = torch.cat(chunk_outputs, dim=1)
            self_attn_weights = None
        else:
            hidden_states, self_attn_weights = self.self_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)

        return outputs


class Qwen3ModelEagle3(Qwen3ModelTF):
    def __init__(self, config: Qwen3Config):
        # super().__init__(config)
        nn.Module.__init__(self)
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.rotary_emb = Qwen3RotaryEmbedding(config=config)
        self.norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        draft_vocab_size = min(config.draft_vocab_size, config.vocab_size)
        config.draft_vocab_size = draft_vocab_size
        self.lm_head = nn.Linear(config.hidden_size, draft_vocab_size, bias=False)

        d2t = torch.arange(draft_vocab_size, dtype=torch.long)
        t2d = torch.zeros((config.vocab_size), dtype=torch.bool)
        t2d[:draft_vocab_size] = True
        self.register_buffer("d2t", d2t)
        self.register_buffer("t2d", t2d)

        # NOTE: Add a midlayer, fc for Eagle-3
        self.midlayer = Qwen3DecoderLayer(config, 0)
        if hasattr(config, "target_hidden_size"):
            self.fc = torch.nn.Linear(config.target_hidden_size * 3, config.hidden_size, bias=False)
        else:
            self.fc = torch.nn.Linear(config.hidden_size * 3, config.hidden_size, bias=False)

        self.gradient_checkpointing = False

    @torch.no_grad()
    def _padding(self, tensor, left=True):
        """Utility function to pad tensors as used in Eagle3"""
        zeropadding = torch.zeros_like(tensor[:, -1:])
        if left:
            tensor = torch.cat((zeropadding, tensor[:, :-1]), dim=1)
        else:
            tensor = torch.cat((tensor[:, 1:], zeropadding), dim=1)
        return tensor

    def forward(
        self,
        base_model_hidden_states: torch.Tensor,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        # Eagle 3 Args: Test-Time Scaling
        prediction_length: Optional[int] = 1,
        target: Optional[torch.Tensor] = None,
        target_topk_idx: Optional[torch.Tensor] = None,
        target_topk_vals: Optional[torch.Tensor] = None,
        loss_mask: Optional[torch.Tensor] = None,
        **flash_attn_kwargs: Unpack[FlashAttentionKwargs],
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once("`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`.")
            use_cache = False

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            if target is not None:
                seq_len_for_cache = target.shape[1]
                cache_device = target.device
            elif input_ids is not None:
                seq_len_for_cache = input_ids.shape[1]
                cache_device = input_ids.device
            elif target_topk_idx is not None:
                seq_len_for_cache = target_topk_idx.shape[1]
                cache_device = target_topk_idx.device
            else:
                raise RuntimeError("Unable to infer sequence length for cache position.")
            cache_position = torch.arange(
                past_seen_tokens,
                past_seen_tokens + seq_len_for_cache,
                device=cache_device,
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        hidden_states = _as_non_inference(base_model_hidden_states)

        if self.training and self.gradient_checkpointing and not hidden_states.requires_grad:
            hidden_states.requires_grad = True

        hidden_states = self.fc(hidden_states)

        batch_size, seq_length, _ = hidden_states.shape

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        loss_list = []
        accuracy_list = []
        compute_accuracy = bool(
            getattr(self.config, "draft_compute_accuracy", False))

        for idx in range(prediction_length):
            inputs_embeds = self.embed_tokens(input_ids)
            inputs_embeds = _as_non_inference(inputs_embeds)
            if self.training and self.gradient_checkpointing and not inputs_embeds.requires_grad:
                inputs_embeds.requires_grad = True
            inputs_embeds = inputs_embeds.to(base_model_hidden_states.dtype)

            # Reset cache for each prediction step to prevent accumulation
            step_past_key_values = DynamicCache() if use_cache else None

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    self.midlayer.__call__,
                    inputs_embeds,
                    hidden_states,
                    attention_mask,
                    position_ids,
                    step_past_key_values,
                    output_attentions,
                    use_cache,
                    cache_position,
                    position_embeddings,
                )
            else:
                layer_outputs = self.midlayer(
                    inputs_embeds,
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=step_past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                    **flash_attn_kwargs,
                )

                hidden_states_out = layer_outputs[0]

                if output_attentions:
                    all_self_attns += (layer_outputs[1],)

            hidden_states = hidden_states_out
            hidden_states_out = self.norm(hidden_states_out)

            sparse_kl_enabled = bool(
                getattr(self.config, "draft_sparse_kl_enabled", True))
            sparse_kl_topk = max(
                1, int(getattr(self.config, "draft_sparse_kl_topk", 64)))
            target_hidden_size = int(
                getattr(self.config, "target_hidden_size", self.config.hidden_size))
            vloss_weight = float(
                getattr(self.config, "draft_vloss_weight", 0.5))
            ploss_weight = float(
                getattr(self.config, "draft_ploss_weight", 0.5))

            with torch.no_grad():
                dense_target_head = None
                target_hidden = None
                if target is not None:
                    if target.dim() == 3 and int(
                            target.shape[-1]) == target_hidden_size:
                        target_hidden = target.to(hidden_states_out.dtype)
                        position_mask = loss_mask
                    else:
                        dense_target_head = target
                        target_max_token = dense_target_head.argmax(-1)
                        target_mask = self.t2d[target_max_token]
                        target_mask = target_mask[..., None].int()
                        position_mask = target_mask * loss_mask
                        dense_target_head = dense_target_head[..., self.t2d]
                        dense_target_head = dense_target_head.float()
                else:
                    position_mask = loss_mask

            logits = self.lm_head(hidden_states_out)
            logits = logits.float()
            teacher_argmax = None
            pred_argmax = None
            if sparse_kl_enabled:
                if target_topk_idx is not None and target_topk_vals is not None:
                    target_topk_idx = target_topk_idx.to(device=logits.device, dtype=torch.long)
                    target_topk_vals = target_topk_vals.to(device=logits.device, dtype=torch.float32)
                    topk = min(sparse_kl_topk, int(target_topk_idx.shape[-1]))
                    target_topk_idx = target_topk_idx[..., :topk]
                    target_topk_vals = target_topk_vals[..., :topk]
                    valid_support = self.t2d[target_topk_idx]
                    valid_any = valid_support.any(dim=-1, keepdim=True)
                    masked_target_vals = target_topk_vals.masked_fill(
                        ~valid_support, float("-inf"))
                    safe_target_vals = torch.where(
                        valid_any, masked_target_vals,
                        torch.zeros_like(masked_target_vals))
                    target_p = nn.Softmax(dim=-1)(safe_target_vals).detach()
                    target_p = torch.where(valid_support, target_p, torch.zeros_like(target_p))
                    target_p = target_p / target_p.sum(dim=-1, keepdim=True).clamp_min(1e-12)
                    target_topk_idx_safe = target_topk_idx.clamp(
                        min=0, max=logits.shape[-1] - 1)
                    logits_topk = torch.gather(
                        logits, dim=-1, index=target_topk_idx_safe)
                    position_mask = position_mask * valid_any.to(position_mask.dtype)
                else:
                    if dense_target_head is None:
                        raise RuntimeError(
                            "Sparse KL enabled but neither sparse nor dense teacher targets were provided.")
                    topk = min(sparse_kl_topk, int(dense_target_head.shape[-1]))
                    target_topk_logits, target_topk_idx = torch.topk(
                        dense_target_head, k=topk, dim=-1)
                    target_p = nn.Softmax(dim=-1)(target_topk_logits).detach()
                    logits_topk = torch.gather(logits, dim=-1, index=target_topk_idx)
                out_logp = nn.LogSoftmax(dim=-1)(logits_topk)
                plogp = target_p * out_logp
                # KL on teacher top-k support to avoid full-vocab dense loss.
                loss = -(position_mask.squeeze(-1) * plogp.sum(dim=-1)).mean()
                pred_argmax = logits_topk.argmax(-1)
                teacher_argmax = target_topk_idx[..., 0]
            else:
                if target_hidden is not None:
                    out_logp = nn.LogSoftmax(dim=2)(logits)
                    with torch.no_grad():
                        teacher_logits = self.lm_head(target_hidden)
                        target_p = nn.Softmax(dim=2)(teacher_logits).detach()
                    plogp = target_p * out_logp
                    valid_mask = position_mask.squeeze(-1).to(dtype=logits.dtype)
                    valid_sum = valid_mask.sum().clamp_min(1e-6)
                    ploss = -torch.sum(valid_mask * plogp.sum(dim=-1)) / valid_sum
                    vloss_raw = F.smooth_l1_loss(
                        hidden_states_out,
                        target_hidden,
                        reduction="none",
                    )
                    vloss = torch.sum(valid_mask * vloss_raw.mean(dim=-1)) / valid_sum
                    loss = vloss_weight * vloss + ploss_weight * ploss
                    pred_argmax = logits.argmax(-1)
                    teacher_argmax = teacher_logits.argmax(-1)
                else:
                    if dense_target_head is None:
                        raise RuntimeError(
                            "Dense KL mode requires dense teacher target tensor.")
                    target_p = nn.Softmax(dim=2)(dense_target_head).detach()
                    out_logp = nn.LogSoftmax(dim=2)(logits)
                    plogp = target_p * out_logp
                    loss = -torch.sum(position_mask * plogp, 2).mean()
                    pred_argmax = logits.argmax(-1)
                    teacher_argmax = target_p.argmax(-1)
            loss_list.append(loss)
            if compute_accuracy:
                with torch.no_grad():
                    if sparse_kl_enabled:
                        pred_local = logits_topk.argmax(-1, keepdim=True)
                        pred_token = torch.gather(
                            target_topk_idx, dim=-1,
                            index=pred_local).squeeze(-1)
                        teacher_token = target_topk_idx[..., 0]
                        accuracy_list.append(
                            ((pred_token == teacher_token) *
                             position_mask.squeeze(-1)).sum().item() /
                            (loss_mask.sum().item() + 1e-6))
                    else:
                        if pred_argmax is None or teacher_argmax is None:
                            pred_argmax = logits.argmax(-1)
                            teacher_argmax = logits.argmax(-1)
                        accuracy_list.append(
                            ((pred_argmax == teacher_argmax) *
                             position_mask.squeeze(-1)).sum().item() /
                            (loss_mask.sum().item() + 1e-6))
            else:
                accuracy_list.append(0.0)

            if idx < prediction_length - 1:
                input_ids = self._padding(input_ids, left=False)
                if target is not None:
                    target = self._padding(target, left=False)
                if target_topk_idx is not None:
                    target_topk_idx = self._padding(target_topk_idx, left=False)
                if target_topk_vals is not None:
                    target_topk_vals = self._padding(target_topk_vals, left=False)
                loss_mask = self._padding(loss_mask, left=False)
                if attention_mask is not None:
                    attention_mask = self._padding(attention_mask, left=False)

            # Clean up step cache to prevent accumulation
            del step_past_key_values

        return loss_list, accuracy_list
