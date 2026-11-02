import ast
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable, Optional

import torch

from verl.models.mcore.weight_converter import McoreToHFWeightConverterDpskv2
from verl.single_controller.ray.base import _ray_worker_passthrough_env
from verl.trainer.constants_ppo import get_ppo_ray_runtime_env
from verl.utils.deepseek_v2_lite import (
    ARCHITECTURE,
    MODEL_REVISION,
    local_routed_experts_by_floor,
    validate_config,
)
from verl.utils.moe_config import get_routed_expert_count

from tools.build_mode1_optimized_rank_plan import _prompt_content_key


def deepseek_config() -> dict:
    return {
        "architectures": [ARCHITECTURE],
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
        "rope_scaling": {
            "type": "yarn",
            "beta_fast": 32,
            "beta_slow": 1,
            "factor": 40,
            "mscale": 0.707,
            "mscale_all_dim": 0.707,
            "original_max_position_embeddings": 4096,
        },
    }


def test_deepseek_routed_expert_field_precedes_qwen_aliases():
    config = deepseek_config() | {"num_experts": 128}
    assert get_routed_expert_count(config) == 64


def test_qwen_expert_field_remains_supported():
    assert get_routed_expert_count({"num_experts": 128}) == 128


def test_deepseek_floor_capacities_cover_all_routed_experts():
    assert local_routed_experts_by_floor(deepseek_config()) == {
        16: 4,
        8: 8,
        4: 16,
        2: 32,
    }


def test_prompt_content_key_matches_deepseek_history_and_tokenizer_forms():
    content = "Solve this problem."
    assert _prompt_content_key(
        f"User: {content}\n\nAssistant:"
    ) == content
    assert _prompt_content_key(
        f"user\n{content}\nassistant\n"
    ) == content


def test_elastic_engine_reads_deepseek_routed_expert_count():
    source = (
        Path(__file__).parents[1] / "vllm/v1/engine/llm_engine.py"
    ).read_text(encoding="utf-8")
    assert source.count('hf_config, "n_routed_experts"') >= 2


def test_deepseek_natural_floor2_profile_enables_all_stages():
    profile = (
        Path(__file__).parents[1]
        / "internal/deepseek_v2_lite_natural_f2_runtime_profile.sh"
    ).read_text(encoding="utf-8")
    assert "deepseek_v2_lite_natural_f4_runtime_profile.sh" in profile
    assert "VLLM_ASCEND_ELASTIC_MIN_COMPUTE_GROUP_SIZE=2" in profile
    assert "VLLM_ASCEND_SHRINK_AWARE_STAGES=8,4,2" in profile
    assert "VLLM_ASCEND_SHRINK_AWARE_FINAL_RANKS=14,15" in profile


def test_deepseek_floor2_probe_does_not_inherit_floor4_planner_limit():
    probe = (
        Path(__file__).parents[1] / "run_deepseek_v2_lite_kv_probe.sh"
    ).read_text(encoding="utf-8")
    assert "natural_f2:2|natural_f2:4|natural_f2:8|natural_f2:16" in probe
    assert 'if [[ "$lifecycle" == *_f2 ]]; then' in probe
    assert "export MIN_ADAPTIVE_FLOOR=2" in probe
    assert "export VLLM_ASCEND_SHRINK_AWARE_STAGES=8,4,2" in probe


def test_deepseek_floor2_smoke_requires_each_shrink_stage():
    smoke = (
        Path(__file__).parents[1] / "run_deepseek_v2_lite_adafloor_smoke.sh"
    ).read_text(encoding="utf-8")
    assert "TARGET_FLOOR=${DEEPSEEK_ADAFLOOR_SMOKE_FLOOR:-8}" in smoke
    assert "SHRINK_STAGES=8,4,2" in smoke
    assert "12,13,14,15;14,15" in smoke
    assert "FINAL_GROUP='14,15'" in smoke
    assert "VLLM_ASCEND_MODE1_STEP_TIMELINE_LOG=1" in smoke
    assert "for active_group in active_groups:" in smoke


def test_mode1_launcher_passes_runtime_topology_to_hydra():
    launcher = (
        Path(__file__).parents[1]
        / "run_mode1_local_length_sorted_e2e_adaptive_floor4.sh"
    ).read_text(encoding="utf-8")
    assert 'SHRINK_STAGES_CONFIG="[${VLLM_ASCEND_SHRINK_AWARE_STAGES}]"' in launcher
    assert 'shrink_aware.shrink_stages="$SHRINK_STAGES_CONFIG"' in launcher
    assert (
        'shrink_aware.intermediate_survivor_ranks="$INTERMEDIATE_RANKS_CONFIG"'
        in launcher
    )
    assert 'shrink_aware.final_survivor_ranks="$FINAL_RANKS_CONFIG"' in launcher


def test_floor2_launcher_uses_two_final_survivors():
    launcher = (
        Path(__file__).parents[1]
        / "run_mode1_local_length_sorted_e2e_adaptive_floor2.sh"
    ).read_text(encoding="utf-8")
    assert "VLLM_ASCEND_SHRINK_AWARE_FINAL_RANKS=14,15" in launcher
    assert 'FINAL_RANKS_ARG="[${VLLM_ASCEND_SHRINK_AWARE_FINAL_RANKS}]"' in launcher


def test_exact_deepseek_v2_lite_config_is_accepted():
    assert validate_config(deepseek_config()) == []


def test_checkpoint_revision_is_pinned():
    assert len(MODEL_REVISION) == 40
    assert all(character in "0123456789abcdef" for character in MODEL_REVISION)


def test_deepseek_launcher_forces_megatron_checkpoint_after_user_overrides():
    launcher = (
        Path(__file__).parents[1]
        / "internal/wj_train_grpo_deepseek_v2_lite_16die_adafloor.sh"
    ).read_text(encoding="utf-8")
    user_override_position = launcher.index('    "$@" \\\n')
    required_final_overrides = (
        "actor_rollout_ref.actor.load_weight=True",
        "actor_rollout_ref.actor.megatron.use_dist_checkpointing=True",
        'actor_rollout_ref.actor.megatron.dist_checkpointing_path="$DISTCP_PATH"',
        "actor_rollout_ref.ref.load_weight=True",
        "actor_rollout_ref.ref.megatron.use_dist_checkpointing=True",
        'actor_rollout_ref.ref.megatron.dist_checkpointing_path="$DISTCP_PATH"',
    )
    assert all(
        launcher.rindex(override) > user_override_position
        for override in required_final_overrides
    )
    assert "DEEPSEEK_VALIDATE_ASSETS_ON_LAUNCH" not in launcher


def test_v3_router_semantics_are_rejected():
    config = deepseek_config() | {
        "scoring_func": "sigmoid",
        "topk_method": "noaux_tc",
    }
    errors = validate_config(config)
    assert any("scoring_func" in error for error in errors)
    assert any("topk_method" in error for error in errors)


def test_grouped_deepseek_experts_expand_for_vllm_weight_loading():
    converter = McoreToHFWeightConverterDpskv2(None, None)
    weight1_params = [object() for _ in range(64 * 2)]
    weight1_names, converted_weight1 = converter._convert_mlp_param(
        "decoder.layers.7.mlp.experts.weight1", weight1_params
    )
    assert converted_weight1 == weight1_params
    assert len(weight1_names) == 128
    assert weight1_names[:2] == [
        "model.layers.7.mlp.experts.0.gate_proj.weight",
        "model.layers.7.mlp.experts.0.up_proj.weight",
    ]
    assert weight1_names[-2:] == [
        "model.layers.7.mlp.experts.63.gate_proj.weight",
        "model.layers.7.mlp.experts.63.up_proj.weight",
    ]

    weight2_params = [object() for _ in range(64)]
    weight2_names, converted_weight2 = converter._convert_mlp_param(
        "decoder.layers.7.mlp.experts.weight2", weight2_params
    )
    assert converted_weight2 == weight2_params
    assert len(weight2_names) == 64
    assert weight2_names[0] == (
        "model.layers.7.mlp.experts.0.down_proj.weight"
    )
    assert weight2_names[-1] == (
        "model.layers.7.mlp.experts.63.down_proj.weight"
    )


def test_ascend_decoder_accepts_vllm_topk_buffer_argument():
    source_path = (
        Path(__file__).parents[1] / "vllm_ascend/models/deepseek_v2.py"
    )
    module = ast.parse(source_path.read_text(encoding="utf-8"))
    decoder = next(
        node
        for node in module.body
        if isinstance(node, ast.ClassDef)
        and node.name == "CustomDeepseekV2DecoderLayer"
    )
    constructor = next(
        node
        for node in decoder.body
        if isinstance(node, ast.FunctionDef) and node.name == "__init__"
    )
    assert [argument.arg for argument in constructor.args.args] == [
        "self",
        "vllm_config",
        "prefix",
        "topk_indices_buffer",
    ]
    assert len(constructor.args.defaults) == 1
    assert isinstance(constructor.args.defaults[0], ast.Constant)
    assert constructor.args.defaults[0].value is None


def test_shared_moe_preserves_shared_then_routed_contract():
    source_path = (
        Path(__file__).parents[1] / "vllm_ascend/ops/common_fused_moe.py"
    )
    module = ast.parse(source_path.read_text(encoding="utf-8"))
    shared_moe = next(
        node
        for node in module.body
        if isinstance(node, ast.ClassDef)
        and node.name == "AscendSharedFusedMoE"
    )
    methods = {
        node.name: node
        for node in shared_moe.body
        if isinstance(node, ast.FunctionDef)
    }
    forward_calls = [
        node
        for node in ast.walk(methods["forward"])
        if isinstance(node, ast.Call)
    ]
    assert any(
        isinstance(call.func, ast.Attribute)
        and isinstance(call.func.value, ast.Name)
        and call.func.value.id == "self"
        and call.func.attr == "forward_impl"
        for call in forward_calls
    )

    implementation_calls = [
        node
        for node in ast.walk(methods["forward_impl"])
        if isinstance(node, ast.Call)
    ]
    assert any(
        isinstance(call.func, ast.Attribute)
        and isinstance(call.func.value, ast.Name)
        and call.func.value.id == "AscendFusedMoE"
        and call.func.attr == "forward"
        for call in implementation_calls
    )
    assert not any(
        isinstance(call.func, ast.Attribute)
        and isinstance(call.func.value, ast.Name)
        and call.func.value.id == "AscendFusedMoE"
        and call.func.attr == "forward_impl"
        for call in implementation_calls
    )


def test_common_moe_weight_loaders_delegate_to_safe_base_copy_path():
    source_path = (
        Path(__file__).parents[1] / "vllm_ascend/ops/common_fused_moe.py"
    )
    module = ast.parse(source_path.read_text(encoding="utf-8"))
    fused_moe = next(
        node
        for node in module.body
        if isinstance(node, ast.ClassDef) and node.name == "AscendFusedMoE"
    )
    methods = {
        node.name: node
        for node in fused_moe.body
        if isinstance(node, ast.FunctionDef)
    }

    for method_name in ("_load_w13", "_load_w2"):
        method = methods[method_name]
        assert not any(
            isinstance(call.func, ast.Attribute)
            and call.func.attr == "copy_"
            for call in ast.walk(method)
            if isinstance(call, ast.Call)
        )
        assert any(
            isinstance(call.func, ast.Attribute)
            and call.func.attr == method_name
            and isinstance(call.func.value, ast.Call)
            and isinstance(call.func.value.func, ast.Name)
            and call.func.value.func.id == "super"
            for call in ast.walk(method)
            if isinstance(call, ast.Call)
        )

    update_map = methods["update_expert_map"]
    assert any(
        isinstance(call.func, ast.Attribute)
        and call.func.attr == "update_expert_map"
        and isinstance(call.func.value, ast.Call)
        and isinstance(call.func.value.func, ast.Name)
        and call.func.value.func.id == "super"
        for call in ast.walk(update_map)
        if isinstance(call, ast.Call)
    )


def test_deepseek_formatted_w13_reload_combines_full_expert_row():
    source_path = (
        Path(__file__).parents[1] / "vllm_ascend/ops/fused_moe.py"
    )
    module = ast.parse(source_path.read_text(encoding="utf-8"))
    fused_moe = next(
        node
        for node in module.body
        if isinstance(node, ast.ClassDef) and node.name == "AscendFusedMoE"
    )
    full_row_loader = next(
        node
        for node in fused_moe.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_load_deepseek_formatted_w13_full_row"
    )
    abort_reload = next(
        node
        for node in fused_moe.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "abort_online_expert_weight_reload"
    )
    harness_module = ast.Module(
        body=[
            ast.ClassDef(
                name="FormattedW13Harness",
                bases=[],
                keywords=[],
                body=[full_row_loader, abort_reload],
                decorator_list=[],
            )
        ],
        type_ignores=[],
    )
    namespace = {"torch": torch}
    exec(compile(ast.fix_missing_locations(harness_module),
                 str(source_path), "exec"), namespace)
    harness = namespace["FormattedW13Harness"]()
    harness.layer_idx = 1
    harness._deepseek_formatted_w13_pending = {}

    gate = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    up = torch.arange(12, 24, dtype=torch.float32).reshape(3, 4)
    target = torch.zeros(1, 6, 4)
    harness._load_deepseek_formatted_w13_full_row(
        target, 0, "w1", gate
    )
    assert torch.count_nonzero(target) == 0
    harness._load_deepseek_formatted_w13_full_row(
        target, 0, "w3", up
    )
    torch.testing.assert_close(target[0], torch.cat((gate, up), dim=0))
    assert harness._deepseek_formatted_w13_pending == {}

    reverse_target = torch.zeros_like(target)
    harness._load_deepseek_formatted_w13_full_row(
        reverse_target, 0, "w3", up
    )
    harness._load_deepseek_formatted_w13_full_row(
        reverse_target, 0, "w1", gate
    )
    torch.testing.assert_close(
        reverse_target[0], torch.cat((gate, up), dim=0)
    )

    first_target = torch.zeros_like(target)
    second_target = torch.zeros_like(target)
    harness._load_deepseek_formatted_w13_full_row(
        first_target, 0, "w1", gate
    )
    harness._load_deepseek_formatted_w13_full_row(
        second_target, 0, "w3", up + 20
    )
    harness._load_deepseek_formatted_w13_full_row(
        first_target, 0, "w3", up
    )
    harness._load_deepseek_formatted_w13_full_row(
        second_target, 0, "w1", gate + 20
    )
    torch.testing.assert_close(
        first_target[0], torch.cat((gate, up), dim=0)
    )
    torch.testing.assert_close(
        second_target[0], torch.cat((gate + 20, up + 20), dim=0)
    )

    incomplete_target = torch.zeros_like(target)
    harness._load_deepseek_formatted_w13_full_row(
        incomplete_target, 0, "w1", gate
    )
    try:
        harness._load_deepseek_formatted_w13_full_row(
            incomplete_target, 0, "w1", gate
        )
    except RuntimeError as error:
        assert "Duplicate DeepSeek formatted w13 shard" in str(error)
    else:
        raise AssertionError("duplicate formatted w13 shard was accepted")
    cache_cleared = []
    harness.clear_lossless_p2p_alias_cache = lambda: cache_cleared.append(True)
    harness.abort_online_expert_weight_reload()
    assert harness._deepseek_formatted_w13_pending == {}
    assert cache_cleared == [True]


def test_online_moe_reload_is_prepared_and_finalized():
    patch_source = (
        Path(__file__).parents[1] / "verl/utils/vllm/patch.py"
    ).read_text(encoding="utf-8")
    rollout_source = (
        Path(__file__).parents[1]
        / "verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py"
    ).read_text(encoding="utf-8")
    megatron_source = (
        Path(__file__).parents[1]
        / "verl/workers/sharding_manager/megatron_vllm.py"
    ).read_text(encoding="utf-8")
    fsdp_source = (
        Path(__file__).parents[1]
        / "verl/workers/sharding_manager/fsdp_vllm.py"
    ).read_text(encoding="utf-8")
    assert "prepare_online_expert_weight_reload" in patch_source
    assert "finalize_online_expert_weight_reload" in patch_source
    for source in (rollout_source, megatron_source, fsdp_source):
        assert "finalize_vllm_moe_model_weight_loader(model)" in source
        assert "abort_vllm_moe_model_weight_loader(model)" in source


def test_singleton_data_parallel_groups_skip_parameter_broadcast():
    source_path = (
        Path(__file__).parents[1]
        / "megatron/core/distributed/distributed_data_parallel.py"
    )
    module = ast.parse(source_path.read_text(encoding="utf-8"))
    ddp_class = next(
        node
        for node in module.body
        if isinstance(node, ast.ClassDef)
        and node.name == "DistributedDataParallel"
    )
    broadcast_params = next(
        node
        for node in ddp_class.body
        if isinstance(node, ast.FunctionDef) and node.name == "broadcast_params"
    )
    singleton_guards = [
        node
        for node in ast.walk(broadcast_params)
        if isinstance(node, ast.If)
        and any(isinstance(child, ast.Continue) for child in node.body)
        and any(
            isinstance(child, ast.Call)
            and isinstance(child.func, ast.Attribute)
            and isinstance(child.func.value, ast.Attribute)
            and isinstance(child.func.value.value, ast.Name)
            and child.func.value.value.id == "torch"
            and child.func.value.attr == "distributed"
            and child.func.attr == "get_world_size"
            for child in ast.walk(node.test)
        )
    ]
    assert singleton_guards


def test_mla_impl_resolver_supports_direct_and_ascend_wrappers():
    source_path = (
        Path(__file__).parents[1]
        / "verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py"
    )
    module = ast.parse(source_path.read_text(encoding="utf-8"))
    selected_functions = [
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef)
        and node.name in {
            "_resolve_mla_attention",
            "_resolve_mla_impl",
            "_refresh_mla_derived_weights",
        }
    ]
    namespace = {
        "Any": object,
        "torch": torch,
        "logger": SimpleNamespace(warning=lambda *_args, **_kwargs: None),
    }
    exec(
        compile(
            ast.Module(selected_functions, type_ignores=[]),
            str(source_path),
            "exec",
        ),
        namespace,
    )
    resolve = namespace["_resolve_mla_impl"]

    direct = SimpleNamespace(
        impl=object(), process_weights_after_loading=lambda _dtype: None)
    nested = SimpleNamespace(
        impl=object(), process_weights_after_loading=lambda _dtype: None)
    assert resolve(direct) is direct.impl
    assert resolve(SimpleNamespace(mla_attn=nested)) is nested.impl
    assert resolve(SimpleNamespace()) is None
    assert resolve(None) is None

    refreshed_dtypes = []
    tracked_attention = SimpleNamespace(
        impl=object(),
        process_weights_after_loading=refreshed_dtypes.append,
    )
    layer = SimpleNamespace(
        self_attn=SimpleNamespace(
            mla_attn=SimpleNamespace(mla_attn=tracked_attention)))
    model = SimpleNamespace(
        model=SimpleNamespace(layers=[layer], start_layer=0, end_layer=1))
    refresh = namespace["_refresh_mla_derived_weights"]
    assert refresh(model, "bfloat16") == 1
    assert refreshed_dtypes == ["bfloat16"]
    assert refresh(model, "bfloat16", require_complete=True) == 1
    assert refreshed_dtypes == ["bfloat16", "bfloat16"]

    incomplete = SimpleNamespace(
        model=SimpleNamespace(layers=[SimpleNamespace()], start_layer=0,
                              end_layer=1))
    try:
        refresh(incomplete, "bfloat16", require_complete=True)
    except RuntimeError as error:
        assert "Incomplete MLA refresh" in str(error)
    else:
        raise AssertionError("incomplete MLA refresh was not rejected")


def test_deepseek_weight_samples_detect_category_specific_changes():
    source_path = (
        Path(__file__).parents[1]
        / "verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py"
    )
    module = ast.parse(source_path.read_text(encoding="utf-8"))
    selected_functions = [
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef)
        and node.name in {
            "_sample_weight_tensor",
            "_weight_compare_category",
            "_compare_weight_samples",
        }
    ]
    namespace = {"Any": Any, "torch": torch}
    exec(
        compile(
            ast.Module(selected_functions, type_ignores=[]),
            str(source_path),
            "exec",
        ),
        namespace,
    )
    sample = namespace["_sample_weight_tensor"]
    compare = namespace["_compare_weight_samples"]

    model = torch.nn.Module()
    model.register_parameter(
        "attention_weight", torch.nn.Parameter(torch.arange(64.0)))
    model.register_parameter(
        "router_weight", torch.nn.Parameter(torch.arange(16.0)))
    references = {
        name: (tuple(param.shape), param.dtype, sample(param))
        for name, param in model.named_parameters()
    }
    assert compare(model, references)["mismatched"] == 0
    model.router_weight.data.add_(1)
    comparison = compare(model, references)
    assert comparison["matched"] == 1
    assert comparison["mismatched"] == 1
    assert comparison["category_mismatches"] == {"other": 1}


def test_deepseek_routed_w13_comparison_distinguishes_stream_and_write():
    source_path = (
        Path(__file__).parents[1]
        / "verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py"
    )
    module = ast.parse(source_path.read_text(encoding="utf-8"))
    selected_functions = [
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef)
        and node.name in {"_tensor_sha256", "_compare_layer1_routed_w13"}
    ]
    namespace = {"Any": Any, "hashlib": __import__("hashlib"), "torch": torch}
    exec(
        compile(
            ast.Module(selected_functions, type_ignores=[]),
            str(source_path),
            "exec",
        ),
        namespace,
    )
    compare = namespace["_compare_layer1_routed_w13"]

    class Layer1Model(torch.nn.Module):
        def __init__(self, weight):
            super().__init__()
            self.model = torch.nn.Module()
            self.model.layers = torch.nn.ModuleList(
                [torch.nn.Identity(), torch.nn.Module()]
            )
            self.model.layers[1].mlp = torch.nn.Module()
            self.model.layers[1].mlp.experts = torch.nn.Module()
            self.model.layers[1].mlp.experts.register_parameter(
                "w13_weight", torch.nn.Parameter(weight.clone()))

    initial = torch.arange(2 * 6 * 4, dtype=torch.float32).reshape(2, 6, 4)
    model = Layer1Model(initial)
    expert_map = (0, 1, -1, -1)
    staged = {
        "model.layers.1.mlp.experts.0.gate_proj.weight": initial[0, :3],
        "model.layers.1.mlp.experts.0.up_proj.weight": initial[0, 3:],
        "model.layers.1.mlp.experts.1.gate_proj.weight": initial[1, :3],
        "model.layers.1.mlp.experts.1.up_proj.weight": initial[1, 3:],
    }
    result = compare(model, initial, expert_map, staged)
    assert result["missing_stream_names"] == []
    assert all(value != "unknown" for value in result["stream_matches"].values())
    assert all(value != "unknown" for value in result["post_matches"].values())

    model.model.layers[1].mlp.experts.w13_weight.data[0, :3].copy_(
        initial[0, 3:])
    corrupted = compare(model, initial, expert_map, staged)
    assert corrupted["post_matches"]["expert0.gate_proj"] == (
        "expert0.up_proj")


def test_deepseek_weight_comparison_gate_rejects_any_mismatch():
    source_path = (
        Path(__file__).parents[1]
        / "verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py"
    )
    module = ast.parse(source_path.read_text(encoding="utf-8"))
    function = next(
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_weight_comparison_failures"
    )
    namespace = {"Any": Any}
    exec(
        compile(ast.Module([function], type_ignores=[]), str(source_path), "exec"),
        namespace,
    )
    validate = namespace["_weight_comparison_failures"]
    comparison = {
        "total": 4,
        "matched": 4,
        "mismatched": 0,
        "category_mismatches": {},
        "missing": [],
        "sample_errors": [],
        "mismatch_samples": [],
    }
    routed = {
        "expected_stream_count": 2,
        "captured_stream_count": 2,
        "missing_stream_names": [],
        "extra_stream_names": [],
        "stream_matches": {"stream-a": "expert0.gate", "stream-b": "expert0.up"},
        "post_matches": {
            "expert0.gate": "expert0.gate",
            "expert0.up": "expert0.up",
        },
    }
    assert validate(comparison, routed) == []

    mismatched = dict(comparison, matched=3, mismatched=1)
    assert validate(mismatched, routed)
    misplaced = dict(routed)
    misplaced["post_matches"] = {
        "expert0.gate": "expert0.up",
        "expert0.up": "expert0.gate",
    }
    assert validate(comparison, misplaced)


def test_actor_probe_requires_all_strict_weight_comparison_passes():
    source = (
        Path(__file__).parents[1] / "run_deepseek_v2_lite_actor_update_probe.sh"
    ).read_text(encoding="utf-8")
    assert "DeepSeek online-sync HF comparison PASS" in source
    assert "incomplete strict HF weight comparisons" in source
    assert "routed_streams != 8" in source


def test_weight_sync_has_a_dedicated_hccl_port_window():
    repo_root = Path(__file__).parents[1]
    allocator_source = (
        repo_root / "verl/single_controller/ray/base.py"
    ).read_text(encoding="utf-8")
    assert '_env_int("VERL_HCCL_IF_BASE_PORT_BLOCK", 16384)' in allocator_source
    assert "_HCCL_PHASE_PORT_OFFSETS = (0, 4096, 8192, 12288)" in allocator_source
    assert "_HCCL_PORTS_PER_PHASE = 32" in allocator_source
    assert "for port_index in range(_HCCL_PORTS_PER_PHASE)" in allocator_source

    worker_source = (
        repo_root / "verl/workers/megatron_workers.py"
    ).read_text(encoding="utf-8")
    assert '"weight_sync": 12288' in worker_source
    assert "def _initialize_megatron_weight_sync_groups" in worker_source
    assert 'VERL_MEGATRON_EAGER_WEIGHT_SYNC_GROUP_INIT", "0"' in worker_source
    assert 'self._set_hccl_if_base_port_for_phase("weight_sync")' in worker_source
    assert "self._initialize_megatron_weight_sync_groups()" in worker_source

    launcher_source = (
        repo_root
        / "internal/wj_train_grpo_deepseek_v2_lite_16die_adafloor.sh"
    ).read_text(encoding="utf-8")
    assert "DEEPSEEK_HCCL_IF_BASE_PORT:-12000" in launcher_source
    assert "DEEPSEEK_MASTER_PORT:-30000" in launcher_source
    assert "VERL_MEGATRON_EAGER_WEIGHT_SYNC_GROUP_INIT:-1" in launcher_source


def test_deepseek_ep16_launcher_exposes_safe_actor_memory_controls():
    repo_root = Path(__file__).parents[1]
    deepseek_launcher = (
        repo_root
        / "internal/wj_train_grpo_deepseek_v2_lite_16die_adafloor.sh"
    ).read_text(encoding="utf-8")
    base_launcher = (
        repo_root
        / "internal/wj_train_grpo_qwen30b_a3b_16die_true_weight_regroup.sh"
    ).read_text(encoding="utf-8")
    probe = (
        repo_root / "run_deepseek_v2_lite_actor_update_probe.sh"
    ).read_text(encoding="utf-8")

    assert "DEEPSEEK_MOE_ALLTOALL_OVERLAP_COMM" in deepseek_launcher
    assert "DEEPSEEK_MOE_SHARED_EXPERT_OVERLAP" in deepseek_launcher
    assert "shared expert overlap must remain enabled" in deepseek_launcher
    assert "MCORE_DEALLOCATE_PIPELINE_OUTPUTS" in base_launcher
    assert "DEEPSEEK_ACTOR_PROBE_TRAINING_STEPS" in probe
    assert 'expected_steps = list(range(1, training_steps + 1))' in probe


def test_deepseek_threshold_smoke_keeps_full_sampling_and_validates_two_steps():
    repo_root = Path(__file__).parents[1]
    wrapper = (
        repo_root / "run_deepseek_v2_lite_threshold_two_step_smoke.sh"
    ).read_text(encoding="utf-8")
    smoke = (
        repo_root / "run_deepseek_v2_lite_adafloor_smoke.sh"
    ).read_text(encoding="utf-8")

    assert "DEEPSEEK_ADAFLOOR_SMOKE_TRAINING_STEPS=2" in wrapper
    assert "DEEPSEEK_ADAFLOOR_SMOKE_ROLLOUT_N=16" in wrapper
    assert "DEEPSEEK_ADAFLOOR_SMOKE_TAIL_VALIDATE_LEVEL_TOKENS=4,16,32,64,64" in wrapper
    assert "DEEPSEEK_ADAFLOOR_SMOKE_BASELINE_DIR" in wrapper
    assert "DYNAMIC_PLAN_STEPS=$TRAINING_STEPS" in smoke
    assert "DYNAMIC_TRAIN_STEPS=$TRAINING_STEPS" in smoke
    assert "expected_shrink_ranks = Counter" in smoke
    assert "expected_restore_ranks = Counter" in smoke
    assert "expected_start_counts = Counter" in smoke
    assert "expected_done = Counter" in smoke
    assert 'rollout_data" / f"{step}.jsonl"' in smoke


def test_deepseek_threshold_actor_smoke_builds_matching_n16_history():
    repo_root = Path(__file__).parents[1]
    wrapper = (
        repo_root / "run_deepseek_v2_lite_threshold_actor_two_step_smoke.sh"
    ).read_text(encoding="utf-8")
    probe = (
        repo_root / "run_deepseek_v2_lite_actor_update_probe.sh"
    ).read_text(encoding="utf-8")

    assert "DEEPSEEK_ACTOR_PROBE_TRAINING_STEPS:-2" in wrapper
    assert "DEEPSEEK_ACTOR_PROBE_ROLLOUT_N:-16" in wrapper
    assert "4,16,32,64,64;4,16,32,64,64" in wrapper
    assert "DEEPSEEK_ACTOR_PROBE_TAIL_VALIDATE_LEVEL_TOKENS_BY_STEP" in probe
    assert "expected_starts = Counter" in probe
    assert "expected_done = Counter" in probe
    assert "exceeded threshold" in probe


def test_deepseek_threshold_adafloor_smoke_chains_exact_two_step_phases():
    repo_root = Path(__file__).parents[1]
    wrapper = (
        repo_root
        / "run_deepseek_v2_lite_threshold_adafloor_two_step_tq1_smoke.sh"
    ).read_text(encoding="utf-8")

    assert "does not accept overrides" in wrapper
    assert "DEEPSEEK_ACTOR_PROBE_TRAINING_STEPS=2" in wrapper
    assert "DEEPSEEK_ACTOR_PROBE_ROLLOUT_N=16" in wrapper
    assert "DEEPSEEK_ADAFLOOR_SMOKE_TRAINING_STEPS=2" in wrapper
    assert "DEEPSEEK_ADAFLOOR_SMOKE_ROLLOUT_N=16" in wrapper
    assert "DEEPSEEK_ADAFLOOR_SMOKE_BASELINE_DIR" in wrapper
    assert "4,16,32,64,64;4,16,32,64,64" in wrapper
    assert 'if [[ ! -f "$MODE0_ROOT/COMPLETE" ]]' in wrapper
    assert 'if [[ ! -f "$MODE1_ROOT/COMPLETE" ]]' in wrapper
    assert "KV_POLICY=smoke-only-not-calibrated" in wrapper


def test_deepseek_common_epoch0_rejects_workload_overrides() -> None:
    repo_root = Path(__file__).parents[1]
    wrapper = (repo_root / "run_deepseek_v2_lite_common_epoch0.sh").read_text(
        encoding="utf-8"
    )

    assert 'if (( $# != 0 )); then' in wrapper
    assert "does not accept workload overrides" in wrapper
    assert "export TRAIN_FILE=/data/deepscaler/train.parquet" in wrapper
    assert "export TEST_FILE=/data/deepscaler/test.parquet" in wrapper
    assert "export COMMON_EPOCH0_DATASET_FRACTION=0.005" in wrapper
    assert "export VLLM_KV_BLOCK_SIZE=128" in wrapper
    assert "export ROLLOUT_ENFORCE_EAGER=True" in wrapper
    assert 'exec "$SCRIPT_DIR/run_common_epoch0_probe_gpu09_kv380800_permanent.sh"' in wrapper


def test_deepseek_natural_f2_calibration_uses_auditable_short_probe() -> None:
    repo_root = Path(__file__).parents[1]
    probe = (repo_root / "run_deepseek_v2_lite_kv_probe.sh").read_text(
        encoding="utf-8"
    )
    calibration = (
        repo_root / "run_deepseek_v2_lite_natural_f2_calibration.sh"
    ).read_text(encoding="utf-8")

    assert "DEEPSEEK_PROBE_TAIL_GUARD_MIN_CAP" in probe
    assert "DEEPSEEK_PROBE_TAIL_GUARD_ROUND_TO" in probe
    assert "must be a positive integer" in probe
    assert "WORKLOAD_MAX_RESPONSE_LENGTH=${COMMON_EPOCH0_MAX_RESPONSE_LENGTH:-16384}" in probe
    assert "WORKLOAD_MAX_NUM_BATCHED_TOKENS=${COMMON_EPOCH0_MAX_NUM_BATCHED_TOKENS:-17408}" in probe
    assert "export DYNAMIC_FULL_MAX_RESPONSE_LENGTH=$WORKLOAD_MAX_RESPONSE_LENGTH" in probe
    assert "export DYNAMIC_FULL_MAX_NUM_BATCHED_TOKENS=$WORKLOAD_MAX_NUM_BATCHED_TOKENS" in probe
    assert "export ROLLOUT_MAX_NUM_SEQS=$WORKLOAD_MAX_NUM_SEQS" in probe
    assert "PROBE_TAIL_GUARD_MIN_CAP=64" in calibration
    assert "PROBE_TAIL_GUARD_ROUND_TO=64" in calibration
    assert "PROBE_EXPECTED_PLAN_RESPONSE_CAP=128" in calibration
    assert '--expected-plan-response-cap "$PROBE_EXPECTED_PLAN_RESPONSE_CAP"' in calibration
    assert "flock -n 9" in calibration
    assert ".adafloor_npu_exclusive.lock" in calibration


def test_expert_weight_loader_uses_versioned_host_map_mirrors():
    source_path = (
        Path(__file__).parents[1] / "vllm_ascend/ops/fused_moe.py"
    )
    module = ast.parse(source_path.read_text(encoding="utf-8"))
    fused_moe = next(
        node
        for node in module.body
        if isinstance(node, ast.ClassDef) and node.name == "AscendFusedMoE"
    )
    method_names = {
        "update_expert_map",
        "_map_global_expert_id_to_local_expert_id",
        "_expert_map_tensor_version",
        "_cache_host_expert_map",
        "_get_host_expert_map_value",
        "_set_lossless_map_cpu_from_tensor",
    }
    methods = [
        node
        for node in fused_moe.body
        if isinstance(node, ast.FunctionDef) and node.name in method_names
    ]
    assert {method.name for method in methods} == method_names
    harness_module = ast.Module(
        body=[
            ast.ClassDef(
                name="ExpertMapHarness",
                bases=[],
                keywords=[],
                body=methods,
                decorator_list=[],
            )
        ],
        type_ignores=[],
    )
    ast.fix_missing_locations(harness_module)
    namespace = {
        "Any": Any,
        "Optional": Optional,
        "torch": SimpleNamespace(Tensor=object),
    }
    exec(compile(harness_module, str(source_path), "exec"), namespace)
    harness = namespace["ExpertMapHarness"]()
    harness.layer_idx = 1
    harness._sync_mode1_active_log2phy_from_expert_map = lambda _reason: None

    class FakeMap:
        def __init__(self, values):
            self.values = list(values)
            self._version = 0
            self.scalar_reads = 0

        def __getitem__(self, _index):
            self.scalar_reads += 1
            raise AssertionError("device scalar lookup must not be used")

        def detach(self):
            return self

        def contiguous(self):
            return self

        def cpu(self):
            return self

        def tolist(self):
            return list(self.values)

    active_map = FakeMap([-1, 0, -1, 1])
    harness.elastic_moe_mode = "lossy"
    harness.expert_map = active_map
    harness.loaded_expert_map = None
    harness._set_lossless_map_cpu_from_tensor(
        active_map, "_lossless_expert_map_cpu"
    )
    assert harness._host_expert_map == (-1, 0, -1, 1)
    assert harness._map_global_expert_id_to_local_expert_id(3) == 1

    active_map.values[3] = 2
    active_map._version += 1
    assert harness._map_global_expert_id_to_local_expert_id(3) == 2

    replacement_map = FakeMap([0, -1, 1, -1])
    harness.update_expert_map(replacement_map)
    assert harness._map_global_expert_id_to_local_expert_id(2) == 1
    assert active_map.scalar_reads == 0
    assert replacement_map.scalar_reads == 0

    loaded_map = FakeMap([-1, 4, -1, 7])
    harness.elastic_moe_mode = "lossless"
    harness.loaded_expert_map = loaded_map
    harness._set_lossless_map_cpu_from_tensor(
        loaded_map, "_lossless_loaded_expert_map_cpu"
    )
    assert harness._map_global_expert_id_to_local_expert_id(3) == 7
    assert loaded_map.scalar_reads == 0

    harness.expert_map = None
    harness.loaded_expert_map = None
    assert harness._map_global_expert_id_to_local_expert_id(5) == 5


def test_deepseek_expert_writes_use_zero_offset_npu_aliases():
    repo_root = Path(__file__).parents[1]
    base_source_path = (
        repo_root / "vllm/model_executor/layers/fused_moe/layer.py"
    )
    base_module = ast.parse(base_source_path.read_text(encoding="utf-8"))
    base_fused_moe = next(
        node
        for node in base_module.body
        if isinstance(node, ast.ClassDef) and node.name == "FusedMoE"
    )
    base_methods = {
        node.name: node
        for node in base_fused_moe.body
        if isinstance(node, ast.FunctionDef)
    }

    def target_hook_calls(method_name):
        return [
            call
            for call in ast.walk(base_methods[method_name])
            if isinstance(call, ast.Call)
            and isinstance(call.func, ast.Attribute)
            and isinstance(call.func.value, ast.Name)
            and call.func.value.id == "self"
            and call.func.attr == "_prepare_expert_weight_write_target"
        ]

    assert target_hook_calls("weight_loader")
    assert target_hook_calls("_load_w13")

    ascend_source_path = repo_root / "vllm_ascend/ops/fused_moe.py"
    ascend_module = ast.parse(ascend_source_path.read_text(encoding="utf-8"))
    ascend_fused_moe = next(
        node
        for node in ascend_module.body
        if isinstance(node, ast.ClassDef) and node.name == "AscendFusedMoE"
    )
    prepare_target = next(
        node
        for node in ascend_fused_moe.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_prepare_expert_weight_write_target"
    )
    harness_module = ast.Module(
        body=[
            ast.ClassDef(
                name="WriteTargetHarness",
                bases=[],
                keywords=[],
                body=[prepare_target],
                decorator_list=[],
            )
        ],
        type_ignores=[],
    )
    ast.fix_missing_locations(harness_module)
    alias_calls = []
    alias = object()

    def zero_offset_alias(base, target):
        alias_calls.append((base, target))
        return alias

    namespace = {
        "torch": SimpleNamespace(Tensor=object),
        "_npu_zero_offset_alias_for_p2p": zero_offset_alias,
    }
    exec(compile(harness_module, str(ascend_source_path), "exec"), namespace)
    harness = namespace["WriteTargetHarness"]()
    harness._lossless_p2p_alias_cache = {}

    class FakeTarget:
        device = SimpleNamespace(type="npu")
        shape = (1, 1408, 2048)
        dtype = "bfloat16"

        def __init__(self, pointer, offset):
            self.pointer = pointer
            self.offset = offset

        def data_ptr(self):
            return self.pointer

        def storage_offset(self):
            return self.offset

        def stride(self):
            return (2883584, 2048, 1)

    base = FakeTarget(pointer=1000, offset=0)
    target = FakeTarget(pointer=2000, offset=2883584)
    harness.model_type = "qwen3_moe"
    assert harness._prepare_expert_weight_write_target(base, target) is target
    assert not alias_calls

    harness.model_type = "deepseek_v2"
    zero_offset_target = FakeTarget(pointer=1000, offset=0)
    assert harness._prepare_expert_weight_write_target(
        base, zero_offset_target
    ) is zero_offset_target
    assert harness._prepare_expert_weight_write_target(base, target) is alias
    assert harness._prepare_expert_weight_write_target(base, target) is alias
    assert alias_calls == [(base, target)]


def test_rollout_weight_staging_normalizes_noncontiguous_sources(monkeypatch):
    source_path = (
        Path(__file__).parents[1]
        / "verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py"
    )
    module = ast.parse(source_path.read_text(encoding="utf-8"))
    selected_functions = [
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef)
        and node.name in {
            "_stream_rollout_weight_staging_enabled",
            "_materialize_rollout_weight_staging",
        }
    ]
    namespace = {
        "Iterable": Iterable,
        "os": os,
        "torch": torch,
    }
    exec(
        compile(
            ast.Module(selected_functions, type_ignores=[]),
            str(source_path),
            "exec",
        ),
        namespace,
    )
    materialize = namespace["_materialize_rollout_weight_staging"]
    monkeypatch.setenv("VLLM_ASCEND_STREAM_ROLLOUT_WEIGHT_STAGING", "1")

    consumed = []
    base = torch.arange(24).reshape(4, 6)
    transposed = base.transpose(0, 1)
    assert not transposed.is_contiguous()

    def source():
        consumed.append("expert")
        yield "expert", transposed
        consumed.append("dense")
        yield "dense", torch.arange(8)

    staged = materialize(source())
    assert not isinstance(staged, list)
    assert consumed == []
    name, normalized = next(iter(staged))
    assert consumed == ["expert"]
    assert name == "expert"
    assert normalized.is_contiguous()
    assert normalized.storage_offset() == 0
    assert normalized.data_ptr() != transposed.data_ptr()
    assert torch.equal(normalized, transposed)

    contiguous = torch.arange(8)
    _, cloned = next(iter(materialize([("dense", contiguous)])))
    assert cloned.is_contiguous()
    assert cloned.data_ptr() != contiguous.data_ptr()
    assert torch.equal(cloned, contiguous)

    contiguous_slice = torch.arange(12)[4:]
    assert contiguous_slice.is_contiguous()
    assert contiguous_slice.storage_offset() == 4
    _, cloned_slice = next(
        iter(materialize([("slice", contiguous_slice)]))
    )
    assert cloned_slice.is_contiguous()
    assert cloned_slice.storage_offset() == 0
    assert cloned_slice.data_ptr() != contiguous_slice.data_ptr()
    assert torch.equal(cloned_slice, contiguous_slice)

    monkeypatch.setenv("VLLM_ASCEND_STREAM_ROLLOUT_WEIGHT_STAGING", "0")
    materialized = materialize([("debug", torch.arange(2))])
    assert isinstance(materialized, list)


def test_weight_loader_diagnostics_propagate_to_ray_workers(monkeypatch):
    monkeypatch.setenv("VLLM_ASCEND_MODE1_WEIGHT_LOADER_DIAG", "1")
    monkeypatch.setenv("VLLM_ASCEND_MODE1_WEIGHT_LOADER_DIAG_LAYERS", "-1")
    runtime_env = get_ppo_ray_runtime_env()
    assert runtime_env["env_vars"][
        "VLLM_ASCEND_MODE1_WEIGHT_LOADER_DIAG"
    ] == "1"
    assert runtime_env["env_vars"][
        "VLLM_ASCEND_MODE1_WEIGHT_LOADER_DIAG_LAYERS"
    ] == "-1"
    worker_env = _ray_worker_passthrough_env()
    assert worker_env["VLLM_ASCEND_MODE1_WEIGHT_LOADER_DIAG"] == "1"
    assert worker_env[
        "VLLM_ASCEND_MODE1_WEIGHT_LOADER_DIAG_LAYERS"
    ] == "-1"


def test_eager_weight_sync_opt_in_propagates_to_ray_workers(monkeypatch):
    monkeypatch.setenv("VERL_MEGATRON_EAGER_WEIGHT_SYNC_GROUP_INIT", "1")
    worker_env = _ray_worker_passthrough_env()
    assert worker_env["VERL_MEGATRON_EAGER_WEIGHT_SYNC_GROUP_INIT"] == "1"


def test_ascend_sync_diagnostics_propagate_to_ray_workers(monkeypatch):
    monkeypatch.setenv("ASCEND_LAUNCH_BLOCKING", "1")
    monkeypatch.setenv("TASK_QUEUE_ENABLE", "0")

    runtime_env = get_ppo_ray_runtime_env()
    assert runtime_env["env_vars"]["ASCEND_LAUNCH_BLOCKING"] == "1"
    assert runtime_env["env_vars"]["TASK_QUEUE_ENABLE"] == "0"

    worker_env = _ray_worker_passthrough_env()
    assert worker_env["ASCEND_LAUNCH_BLOCKING"] == "1"
    assert worker_env["TASK_QUEUE_ENABLE"] == "0"


def test_full_world_restore_waits_for_stale_group_release_before_mc2_warmup():
    source_path = (
        Path(__file__).parents[1] / "vllm_ascend/worker/worker_v1.py"
    )
    module = ast.parse(source_path.read_text(encoding="utf-8"))
    worker = next(
        node
        for node in module.body
        if isinstance(node, ast.ClassDef) and node.name == "NPUWorker"
    )
    restore = next(
        node
        for node in worker.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "restore_elastic_parallel_groups"
    )

    calls = [node for node in ast.walk(restore) if isinstance(node, ast.Call)]
    stale_drop = next(
        call
        for call in calls
        if isinstance(call.func, ast.Attribute)
        and call.func.attr == "_drop_stale_cached_elastic_parallel_groups"
    )
    mc2_warmup = next(
        call
        for call in calls
        if isinstance(call.func, ast.Attribute)
        and call.func.attr
        == "_warmup_post_restore_mc2_dispatch_for_custom_mode1_parity"
    )
    cpu_barriers = [
        call
        for call in calls
        if isinstance(call.func, ast.Attribute)
        and call.func.attr == "barrier"
        and any(
            keyword.arg == "group"
            and isinstance(keyword.value, ast.Attribute)
            and keyword.value.attr == "cpu_group"
            for keyword in call.keywords
        )
    ]

    assert any(
        stale_drop.lineno < barrier.lineno < mc2_warmup.lineno
        for barrier in cpu_barriers
    )


def test_deepseek_mc2_warmup_falls_back_when_topk_exceeds_local_experts():
    source_path = (
        Path(__file__).parents[1]
        / "vllm_ascend/worker/model_runner_v1.py"
    )
    module = ast.parse(source_path.read_text(encoding="utf-8"))
    route_bounds = next(
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_mode1_mc2_warmup_route_bounds"
    )
    namespace = {}
    exec(
        compile(
            ast.Module(body=[route_bounds], type_ignores=[]),
            str(source_path),
            "exec",
        ),
        namespace,
    )
    select_route = namespace["_mode1_mc2_warmup_route_bounds"]

    route_base, route_count, route = select_route("local", 64, 16, 7, 6)
    assert (route_base, route_count) == (0, 64)
    assert route == "local_global_fallback"
    topk_ids = [
        route_base + (index % route_count) for index in range(2 * 6)
    ]
    assert len(set(topk_ids[:6])) == 6
    assert len(set(topk_ids[6:])) == 6

    assert select_route("local", 64, 8, 7, 6) == (56, 8, "local")
    assert select_route("local", 128, 16, 7, 8) == (56, 8, "local")

    try:
        select_route("global", 4, 1, 0, 6)
    except ValueError as error:
        assert "at least top_k distinct global experts" in str(error)
    else:
        raise AssertionError("invalid MC2 route did not fail on the host")
