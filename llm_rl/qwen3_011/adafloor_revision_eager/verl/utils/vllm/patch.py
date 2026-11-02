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

# To support different vLLM versions, we add the model into SUPPORTED_MOE_MODELS separately to avoid triggering
# unsupported issues.
import os
import time

SUPPORTED_MOE_MODELS = []

try:
    from vllm.model_executor.models.deepseek_v2 import DeepseekV2ForCausalLM, DeepseekV3ForCausalLM

    SUPPORTED_MOE_MODELS.append(DeepseekV2ForCausalLM)
    SUPPORTED_MOE_MODELS.append(DeepseekV3ForCausalLM)
except ImportError:
    pass

try:
    from vllm.model_executor.models.mixtral import MixtralForCausalLM

    SUPPORTED_MOE_MODELS.append(MixtralForCausalLM)
except ImportError:
    pass

try:
    from vllm.model_executor.models.qwen2_moe import Qwen2MoeForCausalLM

    SUPPORTED_MOE_MODELS.append(Qwen2MoeForCausalLM)
except ImportError:
    pass

try:
    from vllm.model_executor.models.qwen3_moe import Qwen3MoeForCausalLM

    SUPPORTED_MOE_MODELS.append(Qwen3MoeForCausalLM)
except ImportError:
    pass

try:
    from vllm.model_executor.models.kimi_vl import KimiVLForConditionalGeneration

    SUPPORTED_MOE_MODELS.append(KimiVLForConditionalGeneration)
except ImportError:
    pass


def _mode1_update_weight_diag_enabled() -> bool:
    return os.getenv("VLLM_ASCEND_MODE1_UPDATE_WEIGHTS_DIAG",
                     "0").lower() in ("1", "true", "yes", "on")


def _wrap_mode1_expert_weight_loader_for_diag(experts):
    if not _mode1_update_weight_diag_enabled():
        return
    if getattr(experts, "_mode1_weight_loader_diag_wrapped", False):
        stats = getattr(experts, "_mode1_weight_loader_stats", None)
        if isinstance(stats, dict):
            stats.clear()
        return
    original_loader = experts.weight_loader
    stats = {}
    experts._mode1_weight_loader_stats = stats

    def timed_weight_loader(*args, **kwargs):
        start_t = time.perf_counter()
        status = "ok"
        try:
            return original_loader(*args, **kwargs)
        except Exception:
            status = "error"
            raise
        finally:
            elapsed_s = time.perf_counter() - start_t
            stats["calls"] = int(stats.get("calls", 0)) + 1
            stats["total_s"] = float(stats.get("total_s", 0.0)) + elapsed_s
            stats["max_s"] = max(float(stats.get("max_s", 0.0)), elapsed_s)
            if status != "ok":
                stats["errors"] = int(stats.get("errors", 0)) + 1
            loaded_weight = args[1] if len(args) > 1 else kwargs.get(
                "loaded_weight")
            if hasattr(loaded_weight, "numel"):
                stats["loaded_numel"] = (
                    int(stats.get("loaded_numel", 0)) +
                    int(loaded_weight.numel()))
                try:
                    stats["loaded_bytes"] = (
                        int(stats.get("loaded_bytes", 0)) +
                        int(loaded_weight.numel()) *
                        int(loaded_weight.element_size()))
                except Exception:
                    pass
            shard_id = (args[2] if len(args) > 2 else kwargs.get(
                "shard_id", kwargs.get("loaded_shard_id", "unknown")))
            shard_key = str(shard_id)
            shard_counts = stats.setdefault("shard_counts", {})
            shard_counts[shard_key] = int(shard_counts.get(shard_key, 0)) + 1

    experts.weight_loader = timed_weight_loader
    experts._mode1_weight_loader_diag_wrapped = True


def patch_vllm_moe_model_weight_loader(model):
    # this is a work around to load the weight of vllm fused moe model
    # it is from a bug from vllm 0.8.2
    # all the weights are supposed to have a weight_loader, but the moe weights
    # do not have a weight_loader, so we need to patch it
    # (True, 'model.embed_tokens.weight')
    # (True, 'model.layers.0.self_attn.qkv_proj.weight')
    # (True, 'model.layers.0.self_attn.qkv_proj.bias')
    # (True, 'model.layers.0.self_attn.o_proj.weight')
    # (True, 'model.layers.0.mlp.gate.weight')
    # (True, 'model.layers.0.mlp.shared_expert.gate_up_proj.weight')
    # (True, 'model.layers.0.mlp.shared_expert.down_proj.weight')
    # (False, 'model.layers.0.mlp.shared_expert_gate.weight')   use default
    # (False, 'model.layers.0.input_layernorm.weight')          use default
    # (False, 'model.layers.0.post_attention_layernorm.weight') use default
    # (False, 'model.layers.0.mlp.experts.w13_weight')          use mlp.experts.weight_loader
    # (False, 'model.layers.0.mlp.experts.w2_weight')          use mlp.experts.weight_loader

    # Early return if no MOE models are supported
    if not SUPPORTED_MOE_MODELS:
        return

    if not isinstance(model, tuple(SUPPORTED_MOE_MODELS)):
        return

    original_model_type = type(model)

    # Define MLP attribute mapping for different model types
    MLP_ATTR_MAPPING = {}
    try:
        from vllm.model_executor.models.mixtral import MixtralForCausalLM

        MLP_ATTR_MAPPING[MixtralForCausalLM] = "block_sparse_moe"
    except ImportError:
        pass

    DEFAULT_MLP_ATTR = "mlp"

    # Get inner model (either model.model or model.language_model)
    inner_model = getattr(model, "model", None) or getattr(model, "language_model", None)
    if inner_model is None:
        raise ValueError("The provided model does not have a valid 'model' or 'language_model' attribute.")

    for layer_idx, layer in enumerate(inner_model.layers):
        mlp_attr = MLP_ATTR_MAPPING.get(original_model_type, DEFAULT_MLP_ATTR)

        mlp = getattr(layer, mlp_attr, None)
        if not mlp:
            continue

        experts = getattr(mlp, "experts", None)
        if not experts or not hasattr(experts, "weight_loader"):
            continue
        _wrap_mode1_expert_weight_loader_for_diag(experts)

        # Patch the weight loaders
        for name, param in mlp.named_parameters():
            if "w13_weight" in name or "w2_weight" in name:
                param.weight_loader = experts.weight_loader
