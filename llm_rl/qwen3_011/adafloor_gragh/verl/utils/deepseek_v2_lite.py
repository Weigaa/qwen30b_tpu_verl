"""Static DeepSeek-V2-Lite-Chat model facts used by preparation tooling."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from verl.utils.moe_config import get_routed_expert_count


MODEL_ID = "deepseek-ai/DeepSeek-V2-Lite-Chat"
MODEL_REVISION = "85864749cd611b4353ce1decdb286193298f64c7"
ARCHITECTURE = "DeepseekV2ForCausalLM"
SUPPORTED_FLOORS = (16, 8, 4, 2)

EXPECTED_CONFIG: dict[str, Any] = {
    "model_type": "deepseek_v2",
    "torch_dtype": "bfloat16",
    "vocab_size": 102400,
    "bos_token_id": 100000,
    "eos_token_id": 100001,
    "hidden_size": 2048,
    "intermediate_size": 10944,
    "num_hidden_layers": 27,
    "num_attention_heads": 16,
    "num_key_value_heads": 16,
    "max_position_embeddings": 163840,
    "attention_bias": False,
    "attention_dropout": 0.0,
    "hidden_act": "silu",
    "rms_norm_eps": 1e-6,
    "rope_theta": 10000,
    "tie_word_embeddings": False,
    "use_cache": True,
    "n_routed_experts": 64,
    "n_shared_experts": 2,
    "num_experts_per_tok": 6,
    "first_k_dense_replace": 1,
    "moe_layer_freq": 1,
    "moe_intermediate_size": 1408,
    "n_group": 1,
    "topk_group": 1,
    "pretraining_tp": 1,
    "aux_loss_alpha": 0.001,
    "kv_lora_rank": 512,
    "q_lora_rank": None,
    "qk_nope_head_dim": 128,
    "qk_rope_head_dim": 64,
    "v_head_dim": 128,
    "routed_scaling_factor": 1.0,
    "norm_topk_prob": False,
    "seq_aux": True,
    "scoring_func": "softmax",
    "topk_method": "greedy",
}


def validate_config(config: Mapping[str, Any]) -> list[str]:
    """Return human-readable mismatches for the pinned Lite-Chat checkpoint."""

    errors: list[str] = []
    architectures = config.get("architectures")
    if architectures != [ARCHITECTURE]:
        errors.append(
            f"architectures must be [{ARCHITECTURE!r}], got {architectures!r}"
        )
    for key, expected in EXPECTED_CONFIG.items():
        actual = config.get(key)
        if actual != expected:
            errors.append(f"{key} must be {expected!r}, got {actual!r}")

    rope_scaling = config.get("rope_scaling")
    if not isinstance(rope_scaling, Mapping):
        errors.append("rope_scaling must be a mapping")
    else:
        expected_rope = {
            "beta_fast": 32,
            "beta_slow": 1,
            "factor": 40,
            "mscale": 0.707,
            "mscale_all_dim": 0.707,
            "original_max_position_embeddings": 4096,
        }
        for key, expected in expected_rope.items():
            actual = rope_scaling.get(key)
            if actual != expected:
                errors.append(
                    f"rope_scaling.{key} must be {expected!r}, got {actual!r}"
                )
        rope_type = rope_scaling.get("type", rope_scaling.get("rope_type"))
        if rope_type != "yarn":
            errors.append(f"rope_scaling type must be 'yarn', got {rope_type!r}")

    if get_routed_expert_count(config) != 64:
        errors.append("routed expert count did not resolve to 64")
    return errors


def local_routed_experts_by_floor(config: Mapping[str, Any]) -> dict[int, int]:
    """Return the routed expert capacity required on each active rank."""

    num_experts = get_routed_expert_count(config)
    if num_experts <= 0:
        raise ValueError("model config does not define a routed expert count")
    result: dict[int, int] = {}
    for floor in SUPPORTED_FLOORS:
        if num_experts % floor:
            raise ValueError(
                f"routed experts {num_experts} are not divisible by floor {floor}"
            )
        result[floor] = num_experts // floor
    return result
