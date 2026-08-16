from __future__ import annotations

import ast
import copy
import enum
import importlib.util
import os
import re
import subprocess
import sys
import types
from dataclasses import dataclass
from pathlib import Path

import torch


ROOT = Path(__file__).parents[1]
QWEN3_MOE = ROOT / "vllm_ascend/models/qwen3_moe.py"
BASE_QWEN3_MOE = ROOT / "vllm/model_executor/models/qwen3_moe.py"
FUSED_MOE = ROOT / "vllm_ascend/ops/fused_moe.py"
PLATFORM = ROOT / "vllm_ascend/platform.py"
ACL_GRAPH = ROOT / "vllm_ascend/compilation/acl_graph.py"
ATTENTION = ROOT / "vllm_ascend/attention/attention_v1.py"
MODEL_RUNNER = ROOT / "vllm_ascend/worker/model_runner_v1.py"
COMPILATION_BACKEND = ROOT / "vllm/compilation/backends.py"
DISPATCHER = ROOT / "vllm/v1/cudagraph_dispatcher.py"
WORKER = ROOT / "vllm_ascend/worker/worker_v1.py"
LLM_ENGINE = ROOT / "vllm/v1/engine/llm_engine.py"
ROLLOUT = ROOT / "verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py"
ROLLOUT_CONFIG = ROOT / "verl/workers/config/rollout.py"
PPO_RUNTIME_ENV = ROOT / "verl/trainer/constants_ppo.py"
RAY_WORKER = ROOT / "verl/single_controller/ray/base.py"
ASCEND_UTILS = ROOT / "vllm_ascend/utils.py"
GRAPH_LAUNCHER = (
    ROOT / "run_mode1_local_length_sorted_e2e_adaptive_floor4_aclgraph.sh"
)
FULL_DECODE_EPOCH0_LAUNCHER = (
    ROOT / "run_qwen3_vanilla_epoch0_full_decode_fia_tq1.sh"
)

ELASTIC_MOE_OP = "elastic_ascend_moe_forward"
ELASTIC_MOE_TARGET = f"vllm.{ELASTIC_MOE_OP}"


def _tree(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _definitions(
    tree: ast.AST,
    name: str,
) -> list[ast.FunctionDef | ast.AsyncFunctionDef]:
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == name
    ]


def _definition(
    tree: ast.AST,
    name: str,
) -> ast.FunctionDef | ast.AsyncFunctionDef:
    definitions = _definitions(tree, name)
    assert definitions, f"missing function {name}"
    assert len(definitions) == 1, f"ambiguous function {name}"
    return definitions[0]


def _class(tree: ast.AST, name: str) -> ast.ClassDef:
    classes = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.ClassDef) and node.name == name
    ]
    assert classes, f"missing class {name}"
    assert len(classes) == 1, f"ambiguous class {name}"
    return classes[0]


def _class_method(
    tree: ast.AST,
    class_name: str,
    method_name: str,
) -> ast.FunctionDef | ast.AsyncFunctionDef:
    class_node = _class(tree, class_name)
    methods = [
        node
        for node in class_node.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == method_name
    ]
    assert methods, f"missing {class_name}.{method_name}"
    assert len(methods) == 1, f"ambiguous {class_name}.{method_name}"
    return methods[0]


def _qualified_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _qualified_name(node.value)
        return f"{parent}.{node.attr}" if parent else node.attr
    return ""


def _calls(node: ast.AST, name: str) -> list[ast.Call]:
    return [
        child
        for child in ast.walk(node)
        if isinstance(child, ast.Call)
        and _qualified_name(child.func).endswith(name)
    ]


def _keyword(call: ast.Call, name: str) -> ast.AST:
    values = [keyword.value for keyword in call.keywords if keyword.arg == name]
    assert values, f"missing {name}= in {_qualified_name(call.func)}"
    assert len(values) == 1, f"duplicate {name}= in {_qualified_name(call.func)}"
    return values[0]


def _literal_string(node: ast.AST) -> str:
    assert isinstance(node, ast.Constant) and isinstance(node.value, str)
    return node.value


def _source(node: ast.AST) -> str:
    return ast.unparse(node)


def _call_line(node: ast.AST, name: str) -> int:
    calls = _calls(node, name)
    assert calls, f"missing call to {name}"
    return min(call.lineno for call in calls)


def test_elastic_moe_custom_op_registration_is_isolated() -> None:
    tree = _tree(FUSED_MOE)
    registrations = []
    for call in _calls(tree, "direct_register_custom_op"):
        try:
            op_name = _literal_string(_keyword(call, "op_name"))
        except AssertionError:
            continue
        if op_name == ELASTIC_MOE_OP:
            registrations.append(call)

    assert len(registrations) == 1
    registration = registrations[0]
    assert _qualified_name(_keyword(registration, "op_func")) == ELASTIC_MOE_OP
    assert (
        _qualified_name(_keyword(registration, "fake_impl"))
        == f"{ELASTIC_MOE_OP}_fake"
    )
    assert ast.literal_eval(_keyword(registration, "mutates_args")) == [
        "hidden_states"
    ]
    assert _literal_string(_keyword(registration, "dispatch_key")) == "PrivateUse1"
    assert "torch.Tag.needs_fixed_stride_order" in _source(
        _keyword(registration, "tags")
    )

    # Do not broaden the split to every upstream FusedMoE implementation.
    platform_source = PLATFORM.read_text(encoding="utf-8")
    assert "vllm.moe_forward\"" not in platform_source
    assert "vllm.moe_forward_shared\"" not in platform_source


def test_task_queue_2_requires_explicit_diagnostic_gate() -> None:
    rollout_source = ROLLOUT.read_text(encoding="utf-8")
    launcher_source = GRAPH_LAUNCHER.read_text(encoding="utf-8")

    gate = "VLLM_ASCEND_ELASTIC_ACLGRAPH_ALLOW_TASK_QUEUE_2"
    assert gate in rollout_source
    assert 'task_queue == "2"' in rollout_source
    assert "and allow_task_queue_2" in rollout_source
    assert gate in launcher_source
    assert '"$TASK_QUEUE_ENABLE" == "2"' in launcher_source
    assert (
        '"$VLLM_ASCEND_ELASTIC_ACLGRAPH_ALLOW_TASK_QUEUE_2" == "1"'
        in launcher_source
    )


def test_elastic_full_graph_profiles_hccl_before_compilation() -> None:
    worker_tree = _tree(WORKER)
    profile_context = _class_method(
        worker_tree, "NPUWorker", "_elastic_aclgraph_memory_profile"
    )
    context_source = _source(profile_context)
    assert "VLLM_ASCEND_ELASTIC_ACLGRAPH_EAGER_MEMORY_PROFILE" in context_source
    assert "module.do_not_compile = True" in context_source
    assert "module.do_not_compile = old_value" in context_source
    assert "model.modules()" in context_source
    assert "compiled_codes" in context_source

    determine = _class_method(worker_tree, "NPUWorker", "determine_available_memory")
    determine_source = _source(determine)
    assert "with self._elastic_aclgraph_memory_profile():" in determine_source
    assert "self.model_runner.profile_run()" in determine_source

    headroom = _class_method(
        worker_tree, "NPUWorker", "_estimate_kv_cache_init_headroom_bytes"
    )
    headroom_source = _source(headroom)
    assert "VLLM_ASCEND_ELASTIC_ACLGRAPH" in headroom_source
    assert "3 * GiB_bytes" in headroom_source


def test_floor2_prewarms_alltoallv_before_model_runner_construction() -> None:
    worker_tree = _tree(WORKER)
    init_device = _class_method(worker_tree, "NPUWorker", "_init_device")
    init_source = _source(init_device)
    assert "self._init_worker_distributed_environment()" in init_source
    assert "self._prewarm_mode1_fullworld_alltoallv_workspace()" in init_source
    assert (
        init_source.index("self._init_worker_distributed_environment()")
        < init_source.index("self._prewarm_mode1_fullworld_alltoallv_workspace()")
    )

    public_init = _class_method(worker_tree, "NPUWorker", "init_device")
    assert _call_line(init_device, "_prewarm_mode1_fullworld_alltoallv_workspace") < (
        _call_line(public_init, "NPUModelRunner")
    )

    prewarm = _class_method(
        worker_tree, "NPUWorker", "_prewarm_mode1_fullworld_alltoallv_workspace"
    )
    source = _source(prewarm)
    assert "VLLM_ASCEND_MODE1_PREWARM_FULLWORLD_ALLTOALLV" in source
    assert "output_split_sizes=output_split_sizes" in source
    assert "input_split_sizes=input_split_sizes" in source
    assert "group=ep_group.device_group" in source
    assert "torch.npu.synchronize()" in source
    assert "dist.barrier(group=ep_group.cpu_group)" in source
    assert "dist.barrier(group=ep_group.device_group)" not in source
    assert "barrier_backend=gloo" in source
    assert "mc2_materialized=false" in source


def test_floor2_gate_reuses_eager_cold_init_kv_contract() -> None:
    source = (ROOT / "run_qwen3_adafloor_full_decode_dynamic_gate.sh").read_text(
        encoding="utf-8"
    )
    assert 'VLLM_ASCEND_MODE1_COLD_INIT_KV_TOKENS:-2048' in source
    assert 'VLLM_ASCEND_MODE1_USE_COLD_INIT_KV_CAP:-1' in source
    assert 'echo "cold_init_kv_tokens=' in source
    assert 'echo "use_cold_init_kv_cap=' in source

    rollout_source = ROLLOUT.read_text(encoding="utf-8")
    assert 'os.environ["VLLM_ASCEND_MODE1_IN_COLD_ENGINE_INIT"] = "1"' in rollout_source
    assert "self.inference_engine = LLM(" in rollout_source
    assert rollout_source.index(
        'os.environ["VLLM_ASCEND_MODE1_IN_COLD_ENGINE_INIT"] = "1"'
    ) < rollout_source.index("self.inference_engine = LLM(")


def test_elastic_moe_custom_op_reads_live_runtime_context() -> None:
    implementation_tree = _tree(FUSED_MOE)
    real = _definition(implementation_tree, ELASTIC_MOE_OP)
    assert [argument.arg for argument in real.args.args] == [
        "hidden_states",
        "router_logits",
        "layer_name",
    ]

    source = _source(real)
    assert "get_forward_context()" in source
    assert "no_compile_layers[layer_name]" in source
    assert "with_prefill" in source
    assert "in_profile_run" in source
    assert "time.perf_counter()" in source
    assert "lossless_ffn_enter_wall_ts" in source
    assert "lossless_ffn_tokens" in source
    assert "lossless_ffn_seq" in source

    custom_forward = _class_method(
        _tree(QWEN3_MOE), "CustomSparseMoeBlock", "forward"
    )
    custom_source = _source(custom_forward)
    assert f"torch.ops.vllm.{ELASTIC_MOE_OP}" in custom_source
    assert "self.experts.layer_name" in custom_source
    assert "self.use_elastic_aclgraph" in custom_source

    model_source = QWEN3_MOE.read_text(encoding="utf-8")
    assert "VLLM_ASCEND_ELASTIC_ACLGRAPH" in model_source
    assert "use_elastic_aclgraph" in model_source


def test_qwen3_forward_keeps_python_timing_out_of_compiled_graph() -> None:
    forward = _class_method(
        _tree(BASE_QWEN3_MOE), "Qwen3MoeDecoderLayer", "forward"
    )
    guarded_timing = [
        node
        for node in ast.walk(forward)
        if isinstance(node, ast.If)
        and "not torch.compiler.is_compiling()" in _source(node.test)
        and _calls(node, "time.perf_counter")
    ]
    assert len(guarded_timing) == 1

    guarded_call_ids = {
        id(call) for call in _calls(guarded_timing[0], "time.perf_counter")
    }
    all_call_ids = {id(call) for call in _calls(forward, "time.perf_counter")}
    assert all_call_ids == guarded_call_ids


def test_elastic_moe_fake_impl_preserves_cpu_metadata() -> None:
    function = copy.deepcopy(
        _definition(_tree(FUSED_MOE), f"{ELASTIC_MOE_OP}_fake")
    )
    module = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
    namespace: dict[str, object] = {"torch": torch}
    exec(compile(module, str(FUSED_MOE), "exec"), namespace)
    fake_impl = namespace[f"{ELASTIC_MOE_OP}_fake"]

    hidden_states = torch.arange(35, dtype=torch.float32).reshape(5, 7).t()
    router_logits = torch.zeros((hidden_states.shape[0], 128))
    output = fake_impl(hidden_states, router_logits, "model.layers.1.mlp.experts")

    assert isinstance(output, torch.Tensor)
    assert output.shape == hidden_states.shape
    assert output.dtype == hidden_states.dtype
    assert output.device == hidden_states.device
    assert output.stride() == hidden_states.stride()


def test_elastic_aclgraph_uses_custom_sparse_moe_in_graph_mode() -> None:
    tree = _tree(QWEN3_MOE)
    custom_decoder_classes = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.ClassDef)
        and any(
            isinstance(base, ast.Name) and base.id == "Qwen3MoeDecoderLayer"
            for base in node.bases
        )
    ]
    assert len(custom_decoder_classes) == 1
    init = next(
        node
        for node in custom_decoder_classes[0].body
        if isinstance(node, ast.FunctionDef) and node.name == "__init__"
    )
    source = _source(init)
    assert "use_elastic_aclgraph" in source
    assert "CustomSparseMoeBlock" in source
    elastic_assignments = [
        node
        for node in ast.walk(init)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Attribute)
            and target.attr == "use_elastic_aclgraph"
            for target in node.targets
        )
    ]
    assert len(elastic_assignments) == 1
    elastic_source = _source(elastic_assignments[0].value)
    assert "VLLM_ASCEND_ELASTIC_ACLGRAPH" in elastic_source
    assert "VLLM_ASCEND_ENABLE_ELASTIC_PARALLEL_SHRINK" not in elastic_source

    # The old graph-only fallback made the elastic custom op unreachable.
    old_fallback = re.compile(
        r"if not self\.use_aclgraph:.*?CustomSparseMoeBlock.*?else:.*?"
        r"Qwen3MoeSparseMoeBlock",
        re.DOTALL,
    )
    assert not old_fallback.search(source)


def test_piecewise_platform_splits_at_elastic_moe_boundary() -> None:
    tree = _tree(PLATFORM)
    method = _class_method(tree, "NPUPlatform", "check_and_update_config")
    strings = {
        child.value
        for child in ast.walk(method)
        if isinstance(child, ast.Constant) and isinstance(child.value, str)
    }
    assert {
        "vllm.unified_ascend_attention_with_output",
        "vllm.mla_forward",
        ELASTIC_MOE_TARGET,
    } <= strings
    elastic_assignments = [
        node
        for node in ast.walk(method)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "elastic_aclgraph"
            for target in node.targets
        )
    ]
    assert len(elastic_assignments) == 1
    elastic_source = _source(elastic_assignments[0].value)
    assert "VLLM_ASCEND_ELASTIC_ACLGRAPH" in elastic_source
    assert "VLLM_ASCEND_ENABLE_ELASTIC_PARALLEL_SHRINK" not in elastic_source

    split_setup_line = _call_line(method, "set_splitting_ops_for_v1")
    elastic_append_lines = [
        call.lineno
        for call in _calls(method, "append")
        if call.args
        and isinstance(call.args[0], ast.Constant)
        and call.args[0].value == ELASTIC_MOE_TARGET
    ]
    assert elastic_append_lines
    assert split_setup_line < min(elastic_append_lines)
    assert "compilation_config.use_inductor = False" in _source(method)


def test_piecewise_platform_can_capture_topology_specific_moe() -> None:
    method = _class_method(
        _tree(PLATFORM), "NPUPlatform", "check_and_update_config"
    )
    source = _source(method)
    assert "VLLM_ASCEND_ELASTIC_ACLGRAPH_CAPTURE_MOE" in source
    assert "capture_elastic_moe" in source
    assert "if op != elastic_moe_op" in source
    assert "topology-specific dispatch, expert compute, and combine" in source

    preflight = _definition(
        _tree(ROLLOUT), "_validate_rollout_elastic_aclgraph_runtime"
    )
    preflight_source = _source(preflight)
    assert "VLLM_ASCEND_ELASTIC_ACLGRAPH_CAPTURE_MOE" in preflight_source
    assert "captured MoE remains a splitting op" in preflight_source
    assert "moe_captured" in preflight_source
    assert "supported_modes" in preflight_source
    assert "'PIECEWISE'" in preflight_source
    assert "'FULL_DECODE_ONLY'" in preflight_source
    assert "FULL_DECODE_ONLY requires elastic MoE capture" in preflight_source
    assert "FULL_DECODE_ONLY graph op remains a splitting op" in preflight_source


def test_active_expert_mask_does_not_sync_host_inside_captured_moe() -> None:
    tree = _tree(FUSED_MOE)
    apply_method = _class_method(tree, "AscendUnquantizedFusedMoEMethod", "apply")
    apply_source = _source(apply_method)
    setter = _class_method(tree, "AscendFusedMoE", "set_active_expert_mask")
    setter_source = _source(setter)

    assert "router_logits.masked_fill(~mask, min_value)" in apply_source
    assert "torch.any(mask)" not in apply_source
    assert "active_expert_mask.any().item()" in setter_source
    assert "must contain at least one expert" in setter_source


def test_full_moe_rollout_recapture_is_collectively_ordered() -> None:
    recapture = _definition(_tree(ROLLOUT), "_recapture_rollout_aclgraphs")
    source = _source(recapture)
    invalidate_line = _call_line(recapture, "_invalidate_rollout_aclgraphs")
    capture_line = _call_line(recapture, "capture_model")
    barrier_lines = sorted(call.lineno for call in _calls(recapture, "barrier"))

    assert "VLLM_ASCEND_ELASTIC_ACLGRAPH_CAPTURE_MOE" in source
    assert "get_world_group().cpu_group" in source
    assert len(barrier_lines) == 2
    assert invalidate_line < barrier_lines[0] < capture_line
    assert barrier_lines[0] < capture_line < barrier_lines[1]
    assert "requires an initialized distributed process group" in source


def test_full_decode_capture_cannot_write_dummy_values_into_live_kv_slots() -> None:
    tree = _tree(MODEL_RUNNER)
    isolate = _class_method(
        tree, "NPUModelRunner", "_isolate_live_kv_during_aclgraph_capture"
    )
    isolate_source = _source(isolate)

    assert "CUDAGraphMode.FULL" in isolate_source
    assert "saved_slot_mapping = self.slot_mapping.clone()" in isolate_source
    assert "self.slot_mapping.fill_(PAD_SLOT_ID)" in isolate_source
    assert "finally:" in isolate_source
    assert "self.slot_mapping.copy_(saved_slot_mapping)" in isolate_source
    sync_lines = sorted(call.lineno for call in _calls(isolate, "synchronize"))
    assert len(sync_lines) >= 3
    assert sync_lines[0] < _call_line(isolate, "fill_")
    assert _call_line(isolate, "fill_") < sync_lines[1]
    assert sync_lines[-2] < _call_line(isolate, "copy_") < sync_lines[-1]

    capture = _class_method(tree, "NPUModelRunner", "capture_model")
    capture_source = _source(capture)
    assert "with self._isolate_live_kv_during_aclgraph_capture():" in capture_source
    assert _call_line(capture, "_isolate_live_kv_during_aclgraph_capture") < (
        _call_line(capture, "_capture_model")
    )

    attention_source = ATTENTION.read_text(encoding="utf-8")
    assert "slot_indices=slots" in attention_source
    assert "PAD_SLOT_ID = -1" in (
        ROOT / "vllm/attention/backends/utils.py"
    ).read_text(encoding="utf-8")


def test_piecewise_platform_captures_attention_but_splits_elastic_moe() -> None:
    method = _class_method(
        _tree(PLATFORM), "NPUPlatform", "check_and_update_config"
    )
    source = _source(method)
    assert "VLLM_ASCEND_ELASTIC_ACLGRAPH_CAPTURE_ATTENTION" in source
    assert "ascend_attention_op = 'vllm.unified_ascend_attention_with_output'" in source
    assert "splitting_ops.insert(0, ascend_attention_op)" in source
    assert 'compilation_config.cudagraph_mode = CUDAGraphMode.PIECEWISE' in source
    assert "compilation_config.cudagraph_copy_inputs = True" in source
    assert "dynamic KV" in source
    assert "nested ACLGraph" in source
    assert "MoE/HCCL" in source
    assert "elastic_ascend_moe_forward" in source

    preflight = _definition(
        _tree(ROLLOUT), "_validate_rollout_elastic_aclgraph_runtime"
    )
    preflight_source = _source(preflight)
    assert "VLLM_ASCEND_ELASTIC_ACLGRAPH_CAPTURE_ATTENTION" in preflight_source
    assert "PIECEWISE" in preflight_source
    assert "Elastic ACLGraph requires cudagraph_copy_inputs=True" in preflight_source
    assert "dynamic KV write Attention boundary is missing" in preflight_source
    assert "Attention-in-PIECEWISE requires elastic MoE/HCCL" in preflight_source
    assert "AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE" in preflight_source
    assert "Ascend attention custom op is not registered" in preflight_source


def test_piecewise_attention_uses_per_graph_per_token_task_updates() -> None:
    attention_source = ATTENTION.read_text()
    graph_source = ACL_GRAPH.read_text()
    runner_source = MODEL_RUNNER.read_text()

    assert "torch.npu.graph_task_group_begin(stream)" in attention_source
    assert "torch.npu.graph_task_group_end(stream)" in attention_source
    assert "weak_ref_tensors(attn_metadata.block_tables)" in attention_source
    assert "graph_params.attn_params[num_tokens].append" in attention_source
    assert "_npu_paged_attention_get_workspace" in attention_source
    assert "update_graph_params_workspace(num_tokens, workspace)" in attention_source
    assert "def _forward_decode_only_aclgraph" in attention_source
    assert "static_query.copy_(query)" in attention_source
    assert "output.copy_(static_result)" in attention_source
    assert "self._elastic_pa_graph = ACLGraphWrapper" in attention_source
    assert "weak_ref_output=False" in attention_source
    assert attention_source.count(
        'op_name="unified_ascend_attention_with_output"') == 1
    assert "elastic_ascend_reshape_and_cache" not in attention_source
    assert "elastic_ascend_paged_attention_with_output" not in attention_source

    assert "def update_attn_params" in graph_source
    assert "runtime_metadata.block_tables" in graph_source
    assert "runtime_metadata.seq_lens" in graph_source
    assert "torch.npu.graph_task_update_begin(update_stream, handle)" in graph_source
    assert "event.record(update_stream)" in graph_source
    assert "stale key-cache address" in graph_source
    assert "stale block-table address" in graph_source
    assert "attention_task_range" in graph_source
    assert "task_range=entry.attention_task_range" in graph_source
    assert "workspace=workspace" in graph_source
    assert "update_stream: torch.npu.Stream" in graph_source

    wrapper = _class_method(
        _tree(ACL_GRAPH), "ACLGraphWrapper", "__call__"
    )
    wrapper_source = _source(wrapper)
    replay_line = next(
        node.lineno
        for node in ast.walk(wrapper)
        if isinstance(node, ast.Expr)
        and "entry.aclgraph.replay()" in _source(node)
    )
    synchronize_lines = [
        node.lineno
        for node in ast.walk(wrapper)
        if isinstance(node, ast.Call)
        and _qualified_name(node.func).endswith("synchronize")
    ]
    assert (synchronize_lines
            and any(line < replay_line for line in synchronize_lines)), wrapper_source
    replay_updates = [
        node.lineno
        for node in ast.walk(wrapper)
        if isinstance(node, ast.Call)
        and _qualified_name(node.func).endswith("update_attn_params")
        and node.lineno > replay_line
    ]
    assert replay_updates, wrapper_source
    assert "graph_params.update_stream" in wrapper_source

    assert "aclgraph_runtime_mode == CUDAGraphMode.PIECEWISE" in runner_source
    assert "VLLM_ASCEND_ELASTIC_ACLGRAPH_CAPTURE_ATTENTION" in runner_source
    assert "update_attn_params(self.update_stream, forward_context" in runner_source
    assert "set_graph_params(self.compilation_config.cudagraph_capture_sizes," in runner_source
    assert "self.update_stream" in runner_source


def test_torchair_and_aclgraph_are_mutually_exclusive() -> None:
    method = _class_method(
        _tree(PLATFORM), "NPUPlatform", "check_and_update_config"
    )
    torchair_branches = [
        node
        for node in ast.walk(method)
        if isinstance(node, ast.If)
        and _source(node.test).strip()
        == "ascend_config.torchair_graph_config.enabled"
    ]
    assert len(torchair_branches) == 1
    branch_source = _source(
        ast.Module(body=torchair_branches[0].body, type_ignores=[])
    )
    assert (
        "compilation_config.cudagraph_mode = CUDAGraphMode.NONE"
        in branch_source
    )


def test_elastic_aclgraph_enables_no_dummy_rank_exit_path() -> None:
    init = _class_method(_tree(LLM_ENGINE), "LLMEngine", "__init__")
    source = _source(init)
    assert "elastic_decode_mode" in source
    assert "self.model_config.enforce_eager" in source
    assert "envs_ascend.VLLM_ASCEND_ELASTIC_ACLGRAPH" in source

    no_dummy_assignments = [
        node
        for node in ast.walk(init)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Attribute)
            and target.attr == "elastic_ep_no_dummy"
            for target in node.targets
        )
    ]
    assert len(no_dummy_assignments) == 1
    assignment_source = _source(no_dummy_assignments[0].value)
    assert "elastic_decode_mode" in assignment_source
    assert "_is_followup_elastic_shrink_enabled" in assignment_source


def test_rollout_accepts_ray_serialized_aclgraph_capture_lists() -> None:
    init = _class_method(_tree(ROLLOUT), "vLLMRollout", "__init__")
    source = _source(init)
    assert "isinstance(cudagraph_capture_sizes, (ListConfig, list, tuple))" in source
    assert "normalized_capture_sizes" in source
    assert "unique positive integers" in source
    assert "cudagraph_capture_sizes=normalized_capture_sizes" in source
    assert "logger.warning" not in "\n".join(
        line
        for line in source.splitlines()
        if "cudagraph_capture_sizes" in line
    )


def test_rollout_exposes_native_full_decode_only_without_changing_default() -> None:
    rollout_config = ROLLOUT_CONFIG.read_text(encoding="utf-8")
    rollout_source = ROLLOUT.read_text(encoding="utf-8")
    assert "cudagraph_mode: Optional[str] = None" in rollout_config
    assert 'config.get("cudagraph_mode") or "PIECEWISE"' in rollout_source
    assert "CUDAGraphMode.FULL_DECODE_ONLY" in rollout_source
    assert "Native ACLGraph rollout configured" in rollout_source


def test_full_decode_only_keeps_outer_attention_and_moe_in_graph() -> None:
    platform_source = PLATFORM.read_text(encoding="utf-8")
    attention_source = ATTENTION.read_text(encoding="utf-8")
    runner_source = MODEL_RUNNER.read_text(encoding="utf-8")
    assert "Native FULL_DECODE_ONLY ACLGraph enabled" in platform_source
    assert "KV write, " in platform_source
    assert "Attention read, MoE, and dense decode" in platform_source
    assert "FULL_DECODE_ONLY ACLGraph captured Attention KV write" in attention_source
    assert "from vllm.logger import logger" in attention_source
    assert "forward_context.cudagraph_runtime_mode" in attention_source
    assert "CUDAGraphMode.FULL" in attention_source
    assert "SEQ_LEN_WITH_MAX_PA_WORKSPACE = 6144" in runner_source
    assert "if is_graph_capturing else max_query_len" in runner_source
    assert "aclgraph_runtime_mode == CUDAGraphMode.FULL" in runner_source
    assert "FULL_DECODE_ONLY Attention maximum workspace" in attention_source


def test_full_decode_fia_uses_one_capture_time_max_workspace() -> None:
    attention_tree = _tree(ATTENTION)
    capture = _class_method(
        attention_tree,
        "AscendAttentionBackendImpl",
        "_forward_decode_only_full_graph_fia",
    )
    capture_source = _source(capture)
    update = _definition(_tree(ACL_GRAPH), "update_attn_params")
    fia_branches = [
        node
        for node in ast.walk(update)
        if isinstance(node, ast.If)
        and "attention_backend == 'fia'" in _source(node.test)
    ]
    assert len(fia_branches) == 1
    fia_update_source = _source(
        ast.Module(body=fia_branches[0].body, type_ignores=[])
    )

    assert "_npu_fused_infer_attention_score_get_max_workspace" in capture_source
    assert "update_graph_params_workspace(num_tokens, workspace)" in capture_source
    assert "npu_fused_infer_attention_score.out" in capture_source
    assert "actual_seq_lengths_q" in capture_source
    assert "seq_lens_list" in capture_source
    assert "npu_fused_infer_attention_score.out" in fia_update_source
    assert "graph_params.workspaces.get(runtime_shape)" in fia_update_source
    assert "_npu_fused_infer_attention_score_get_max_workspace" not in fia_update_source
    assert "_npu_paged_attention_get_workspace" not in fia_update_source
    assert "runtime_metadata.actual_seq_lengths_q" in fia_update_source
    assert "runtime_metadata.seq_lens_list" in fia_update_source


def test_full_decode_fia_accepts_optional_decode_attention_mask() -> None:
    capture = _class_method(
        _tree(ATTENTION),
        "AscendAttentionBackendImpl",
        "_forward_decode_only_full_graph_fia",
    )
    capture_source = _source(capture)

    assert "requires a stable attention mask" not in capture_source
    assert "if attn_metadata.attn_mask is not None else None" in capture_source
    assert "atten_mask=attn_metadata.attn_mask" in capture_source


def test_full_decode_fia_sparse_mode_matches_optional_mask_across_replay() -> None:
    capture = _class_method(
        _tree(ATTENTION),
        "AscendAttentionBackendImpl",
        "_forward_decode_only_full_graph_fia",
    )
    capture_source = _source(capture)
    update = _definition(_tree(ACL_GRAPH), "update_attn_params")
    update_source = _source(update)

    selection = "3 if attn_metadata.attn_mask is not None else 0"
    assert selection in capture_source
    assert capture_source.count("sparse_mode=sparse_mode") == 2
    assert "3 if runtime_metadata.attn_mask is not None else 0" in update_source
    assert "runtime_sparse_mode != sparse_mode" in update_source
    assert "sparse_mode=sparse_mode" in update_source


def test_full_decode_replay_requires_static_addresses_without_input_copy() -> None:
    platform_method = _class_method(
        _tree(PLATFORM), "NPUPlatform", "check_and_update_config"
    )
    platform_source = _source(platform_method)
    wrapper = _class_method(_tree(ACL_GRAPH), "ACLGraphWrapper", "__call__")
    wrapper_source = _source(wrapper)

    assert "compilation_config.cudagraph_copy_inputs = False" in platform_source
    assert "FULL_DECODE_ONLY requires stable model-input addresses" in wrapper_source
    assert "self.runtime_mode == CUDAGraphMode.PIECEWISE" in wrapper_source
    assert "captured.copy_(runtime)" in wrapper_source
    stable_address_line = wrapper_source.index(
        "FULL_DECODE_ONLY requires stable model-input addresses"
    )
    replay_line = wrapper_source.index("entry.aclgraph.replay()")
    assert stable_address_line < replay_line


def test_full_decode_attention_backend_reaches_ray_workers() -> None:
    variable = "VLLM_ASCEND_FULL_DECODE_ATTENTION_BACKEND"
    for path in (PPO_RUNTIME_ENV, RAY_WORKER):
        assert variable in path.read_text(encoding="utf-8")


def test_full_decode_epoch0_launcher_selects_fia_single_graph_contract() -> None:
    subprocess.run(
        ["bash", "-n", str(FULL_DECODE_EPOCH0_LAUNCHER)], check=True
    )
    source = FULL_DECODE_EPOCH0_LAUNCHER.read_text(encoding="utf-8")
    assert 'CAPTURE_SIZES="${FULL_DECODE_CAPTURE_SIZES:-[32]}"' in source
    assert "ADAFLOOR_ACLGRAPH_MODE=FULL_DECODE_ONLY" in source
    assert "VLLM_ASCEND_FULL_DECODE_ATTENTION_BACKEND=fia_max_workspace" in source
    assert "VLLM_ASCEND_ELASTIC_ACLGRAPH_CAPTURE_ATTENTION=1" in source
    assert "VLLM_ASCEND_ELASTIC_ACLGRAPH_CAPTURE_MOE=1" in source
    assert "VLLM_ENABLE_GRAPH_MODE=0 TASK_QUEUE_ENABLE=1" in source
    assert "ROLLOUT_ENFORCE_EAGER=False" in source
    assert "COMMON_EPOCH0_TRAIN_STEPS=\"$STEPS\"" in source
    assert "COMMON_EPOCH0_ROLLOUT_N=\"$ROLLOUT_N\"" in source
    assert "attention_backend=fia_max_workspace" in source
    assert "cudagraph_copy_inputs=false" in source


def test_full_decode_only_prefill_bypasses_static_decode_bytecode() -> None:
    source = (ROOT / "vllm/compilation/decorators.py").read_text()
    marker = "FULL_DECODE_ONLY deliberately has two execution paths"

    assert "CUDAGraphMode.FULL_DECODE_ONLY" in source
    assert "get_forward_context().cudagraph_runtime_mode" in source
    assert "== CUDAGraphMode.NONE" in source
    eager_return = source.index("return self.forward(*args, **kwargs)",
                                source.index(marker))
    compiled_dispatch = source.index("with self.dispatch_to_code(0):")
    assert eager_return < compiled_dispatch


def test_compilation_cache_tolerates_frozen_python_modules() -> None:
    call = _class_method(_tree(COMPILATION_BACKEND), "VllmBackend", "__call__")
    source = _source(call)
    assert "except OSError" in source
    assert "Failed to read traced file" in source
    assert source.index("hash_content.append(filepath)") < source.index(
        "except OSError"
    )


def test_aclgraph_wrapper_cache_clear_is_cpu_testable() -> None:
    tree = _tree(ACL_GRAPH)
    method = copy.deepcopy(
        _class_method(tree, "ACLGraphWrapper", "clear_aclgraph_cache")
    )
    wrapper_class = ast.ClassDef(
        name="CpuWrapper",
        bases=[],
        keywords=[],
        body=[method],
        decorator_list=[],
    )
    module = ast.fix_missing_locations(
        ast.Module(body=[wrapper_class], type_ignores=[])
    )
    namespace: dict[str, object] = {}
    exec(compile(module, str(ACL_GRAPH), "exec"), namespace)

    class FakeGraph:
        def __init__(self) -> None:
            self.reset_calls = 0

        def reset(self) -> None:
            self.reset_calls += 1

    graphs = [FakeGraph(), FakeGraph()]
    wrapper = namespace["CpuWrapper"]()
    wrapper.concrete_aclgraph_entries = {
        name: types.SimpleNamespace(
            aclgraph=graph,
            output=object(),
            input_addresses=[1],
            input_tensors=[object()],
            attention_task_range=(0, 1),
        )
        for name, graph in zip(("one", "two"), graphs)
    }
    wrapper.first_run_finished = True
    assert wrapper.clear_aclgraph_cache() == 2
    assert wrapper.concrete_aclgraph_entries == {}
    assert wrapper.first_run_finished is False
    assert [graph.reset_calls for graph in graphs] == [1, 1]


def test_elastic_piecewise_replay_refreshes_changed_input_buffers() -> None:
    tree = _tree(ACL_GRAPH)
    entry = _class(tree, "ACLGraphEntry")
    entry_source = _source(entry)
    assert "input_tensors" in entry_source

    call = _class_method(tree, "ACLGraphWrapper", "__call__")
    source = _source(call)
    capture = source.index("entry.input_tensors = [")
    replay = source.index("runtime_tensors = [")
    validate = source.index("Elastic ACLGraph input changed incompatibly")
    refresh = source.index("captured.copy_(runtime)")
    graph_replay = source.index("entry.aclgraph.replay()")
    assert capture < replay < validate < refresh < graph_replay
    assert "captured.data_ptr() == runtime.data_ptr()" in source
    assert "captured.shape != runtime.shape" in source
    assert "captured.stride() != runtime.stride()" in source
    assert "captured.dtype != runtime.dtype" in source
    assert "captured.device != runtime.device" in source

    clear = _class_method(tree, "ACLGraphWrapper", "clear_aclgraph_cache")
    assert "entry.input_tensors = None" in _source(clear)


def test_piecewise_outer_input_buffer_grows_before_large_prefill_copy() -> None:
    source = COMPILATION_BACKEND.read_text(encoding="utf-8")
    grow_guard = "runtime_shape > self.input_buffers[i].shape[0]"
    grow = "self.input_buffers[i] = runtime_tensor.new_empty("
    slice_buffer = "static_tensor = self.input_buffers[i][:runtime_shape]"
    copy = "static_tensor.copy_(runtime_tensor)"
    assert grow_guard in source
    assert source.index(grow_guard) < source.index(grow)
    assert source.index(grow) < source.index(slice_buffer) < source.index(copy)
    assert "Elastic ACLGraph outer input buffer expanded" in source


def test_global_aclgraph_cache_clear_covers_hidden_wrappers_and_graph_params() -> None:
    tree = _tree(ACL_GRAPH)
    source = ACL_GRAPH.read_text(encoding="utf-8")
    assert re.search(
        r"_ACL_GRAPH_WRAPPERS\s*:[^=]+\s*=\s*weakref\.WeakSet\(\)",
        source,
    )

    init = _class_method(tree, "ACLGraphWrapper", "__init__")
    assert "_ACL_GRAPH_WRAPPERS.add(self)" in _source(init)

    reset = _definition(tree, "reset_graph_params_runtime_state")
    assert ".clear()" in _source(reset)

    clear = _definition(tree, "clear_aclgraph_caches")
    clear_source = _source(clear)
    assert "_ACL_GRAPH_WRAPPERS" in clear_source
    assert ".clear_aclgraph_cache()" in clear_source
    assert "reset_graph_params_runtime_state()" in clear_source
    assert "gc.collect()" in clear_source
    assert "renew_aclgraph_pool()" in clear_source

    renew = _definition(tree, "renew_aclgraph_pool")
    renew_source = _source(renew)
    assert "_global_graph_pool = None" in renew_source
    assert "current_platform.get_global_graph_pool()" in renew_source
    assert "wrapper.graph_pool = graph_pool" in renew_source


def test_full_moe_graph_logs_first_replay_of_each_cache_generation() -> None:
    wrapper = _class_method(_tree(ACL_GRAPH), "ACLGraphWrapper", "__call__")
    source = _source(wrapper)
    assert "VLLM_ASCEND_ELASTIC_ACLGRAPH_CAPTURE_MOE" in source
    assert "_logged_replay_generation" in source
    assert "Elastic full-MoE ACLGraph replay" in source

    clear = _class_method(
        _tree(ACL_GRAPH), "ACLGraphWrapper", "clear_aclgraph_cache"
    )
    clear_source = _source(clear)
    assert "_cache_generation" in clear_source
    assert "+ 1" in clear_source
    assert "_logged_replay_generation = -1" in clear_source


def test_dispatcher_stays_eager_until_keys_are_initialized(
    monkeypatch,
) -> None:
    class FakeCUDAGraphMode(enum.Enum):
        NONE = 0
        PIECEWISE = 1
        FULL = 2

        def requires_piecewise_compilation(self) -> bool:
            return self is FakeCUDAGraphMode.PIECEWISE

        def mixed_mode(self):
            return self

        def decode_mode(self):
            return self

        def separate_routine(self) -> bool:
            return False

    @dataclass(frozen=True)
    class FakeBatchDescriptor:
        num_tokens: int
        uniform_decode: bool

        @property
        def non_uniform(self):
            return FakeBatchDescriptor(self.num_tokens, False)

    class FakeLogger:
        def __init__(self) -> None:
            self.warnings = 0

        def warning_once(self, *args, **kwargs) -> None:
            self.warnings += 1

    fake_logger = FakeLogger()
    vllm_package = types.ModuleType("vllm")
    vllm_package.__path__ = []
    config_module = types.ModuleType("vllm.config")
    config_module.CUDAGraphMode = FakeCUDAGraphMode
    config_module.VllmConfig = object
    context_module = types.ModuleType("vllm.forward_context")
    context_module.BatchDescriptor = FakeBatchDescriptor
    logger_module = types.ModuleType("vllm.logger")
    logger_module.init_logger = lambda _name: fake_logger
    monkeypatch.setitem(sys.modules, "vllm", vllm_package)
    monkeypatch.setitem(sys.modules, "vllm.config", config_module)
    monkeypatch.setitem(sys.modules, "vllm.forward_context", context_module)
    monkeypatch.setitem(sys.modules, "vllm.logger", logger_module)

    spec = importlib.util.spec_from_file_location(
        "_adafloor_cpu_cudagraph_dispatcher", DISPATCHER
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    compilation_config = types.SimpleNamespace(
        cudagraph_mode=FakeCUDAGraphMode.PIECEWISE,
        cudagraph_capture_sizes=[1, 2, 4],
        is_attention_compiled_piecewise=lambda: True,
        level="PIECEWISE",
        splitting_ops=["vllm.unified_ascend_attention_with_output"],
    )
    config = types.SimpleNamespace(
        compilation_config=compilation_config,
        scheduler_config=types.SimpleNamespace(max_num_seqs=4),
    )
    dispatcher = module.CudagraphDispatcher(config)
    key = FakeBatchDescriptor(num_tokens=2, uniform_decode=False)

    assert dispatcher.dispatch(key) == (FakeCUDAGraphMode.NONE, None)
    assert fake_logger.warnings == 1
    dispatcher.initialize_cudagraph_keys(FakeCUDAGraphMode.PIECEWISE, 1)
    assert dispatcher.keys_initialized is True
    assert dispatcher.dispatch(key) == (FakeCUDAGraphMode.PIECEWISE, key)
    assert dispatcher.dispatch(FakeBatchDescriptor(3, False)) == (
        FakeCUDAGraphMode.NONE,
        None,
    )


def test_full_moe_floor_transitions_invalidate_before_group_rebuild_and_recapture() -> None:
    method = _class_method(_tree(WORKER), "NPUWorker", "rebuild_elastic_ep_group")
    source = _source(method)
    assert "_invalidate_full_moe_elastic_aclgraph" in source
    assert "_recapture_full_moe_elastic_aclgraph" in source
    assert _call_line(method, "_invalidate_full_moe_elastic_aclgraph") < _call_line(
        method, "_rebuild_group"
    )
    assert _call_line(method, "_refresh_elastic_parallel_state") < _call_line(
        method, "_recapture_full_moe_elastic_aclgraph"
    )
    assert _call_line(method, "_warmup_post_shrink_moe_dispatch") < _call_line(
        method, "_recapture_full_moe_elastic_aclgraph"
    )

    invalidate = _class_method(
        _tree(WORKER), "NPUWorker", "_invalidate_full_moe_elastic_aclgraph"
    )
    invalidate_source = _source(invalidate)
    assert "disable_aclgraph_dispatch" in invalidate_source
    assert "clear_aclgraph_caches" in invalidate_source

    recapture = _class_method(
        _tree(WORKER), "NPUWorker", "_recapture_full_moe_elastic_aclgraph"
    )
    recapture_source = _source(recapture)
    assert "model_runner.capture_model()" in recapture_source
    assert recapture_source.count("torch.distributed.barrier") == 2


def test_elastic_kv_initialization_defers_capture_to_rollout_lifecycle() -> None:
    method = _class_method(
        _tree(WORKER), "NPUWorker", "compile_or_warm_up_model"
    )
    source = _source(method)
    assert "VLLM_ASCEND_ELASTIC_ACLGRAPH" in source
    assert "VLLM_ASCEND_ENABLE_ELASTIC_PARALLEL_SHRINK" not in source
    assert "defer_elastic_capture" in source
    assert "automatic capture deferred" in source
    assert "self.model_runner.capture_model()" in source

    capture_call = next(
        node
        for node in ast.walk(method)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "capture_model"
    )
    parent_if = next(
        node
        for node in ast.walk(method)
        if isinstance(node, ast.If)
        and any(child is capture_call for child in ast.walk(node))
        and "defer_elastic_capture" in _source(node)
    )
    assert "else:" in _source(parent_if)


def test_restore_invalidates_low_floor_full_moe_graph_without_recapture() -> None:
    method = _class_method(
        _tree(WORKER), "NPUWorker", "restore_elastic_parallel_groups"
    )
    source = _source(method)
    assert "_invalidate_full_moe_elastic_aclgraph" in source
    assert _call_line(method, "_invalidate_full_moe_elastic_aclgraph") < _call_line(
        method, "_rebuild_group"
    )
    assert "_recapture_full_moe_elastic_aclgraph" not in source


def test_weight_reload_invalidates_and_officially_recaptures_aclgraphs() -> None:
    tree = _tree(ROLLOUT)
    invalidate = _definition(tree, "_invalidate_rollout_aclgraphs")
    invalidate_source = _source(invalidate)
    assert "clear_aclgraph_caches" in invalidate_source
    cache_clear = _definition(_tree(ACL_GRAPH), "clear_aclgraph_caches")
    assert _call_line(cache_clear, "synchronize") < _call_line(
        cache_clear, "clear_aclgraph_cache"
    )

    recapture = _definition(tree, "_recapture_rollout_aclgraphs")
    recapture_source = _source(recapture)
    assert "_invalidate_rollout_aclgraphs" in recapture_source
    assert _call_line(recapture, "_invalidate_rollout_aclgraphs") < _call_line(
        recapture, "capture_model"
    )
    assert "model_runner.capture_model()" in recapture_source

    update = _class_method(tree, "vLLMRollout", "update_weights")
    update_source = _source(update)
    assert "_invalidate_rollout_aclgraphs" in update_source
    assert _call_line(update, "finalize_vllm_moe_model_weight_loader") < _call_line(
        update, "_invalidate_rollout_aclgraphs"
    )

    assert "_needs_rollout_aclgraph_recapture" in update_source
    assert "_recapture_rollout_aclgraphs" not in update_source

    generate = _class_method(tree, "vLLMRollout", "generate_sequences")
    generate_source = _source(generate)
    assert "_needs_rollout_aclgraph_recapture" in generate_source
    assert "_recapture_rollout_aclgraphs" in generate_source
    assert _call_line(generate, "_maybe_resize_mode1_kv_cache_from_meta") < _call_line(
        generate, "_recapture_rollout_aclgraphs"
    )


def test_rollout_builds_piecewise_config_from_list_capture_sizes() -> None:
    init = _class_method(_tree(ROLLOUT), "vLLMRollout", "__init__")
    source = _source(init)
    assert "not config.enforce_eager" in source
    assert "isinstance(cudagraph_capture_sizes, (ListConfig, list, tuple))" in source
    assert "level=CompilationLevel.PIECEWISE" in source
    assert "cudagraph_capture_sizes=normalized_capture_sizes" in source


def test_natural_f2_profile_and_mc2_lifecycle_match_eager_contract() -> None:
    gate = (ROOT / "run_qwen3_adafloor_full_decode_dynamic_gate.sh").read_text(
        encoding="utf-8"
    )
    assert 'VLLM_ASCEND_MODE1_PROFILE_MAX_TOKENS:-2048' in gate
    assert 'VLLM_ASCEND_MODE1_PROFILE_AVOID_MC2_BOUNDARY:-1' in gate
    assert 'VLLM_ASCEND_MODE1_PROFILE_EXPECT_MOE_COMM:-alltoall' in gate
    assert 'VLLM_ASCEND_MODE1_PROFILE_EXTRA_MC2_DUMMY:-0' in gate
    assert 'export ROLLOUT_MAX_NUM_BATCHED_TOKENS=17408' in gate
    assert 'export VLLM_ASCEND_MODE1_PARITY_MC2_MEM_LOG=1' in gate
    assert 'ADAFLOOR_DYNAMIC_GATE_EXECUTION_MODE:-graph' in gate
    assert (
        'export TASK_QUEUE_ENABLE=${ADAFLOOR_DYNAMIC_GATE_EAGER_TASK_QUEUE_ENABLE:-2}'
        in gate
    )
    assert "ADAFLOOR_DYNAMIC_GATE_EAGER_TASK_QUEUE_ENABLE must be 1 or 2" in gate
    assert 'export ROLLOUT_ENFORCE_EAGER=True' in gate
    assert 'EXECUTION_RUNNER="$GRAPH_BASE_RUNNER"' in gate

    profile = _class_method(
        _tree(MODEL_RUNNER), "NPUModelRunner", "profile_run"
    )
    profile_source = _source(profile)
    assert "VLLM_ASCEND_MODE1_PROFILE_EXPECT_MOE_COMM" in profile_source
    assert "_select_moe_comm_method" in profile_source
    assert "Mode1 profile MoE communication contract mismatch" in profile_source
    assert _call_line(profile, "_select_moe_comm_method") < _call_line(
        profile, "_dummy_run"
    )

    memory_profile = _class_method(
        _tree(WORKER), "NPUWorker", "_elastic_aclgraph_memory_profile"
    )
    memory_profile_source = _source(memory_profile)
    assert "aclgraph_moe_methods" in memory_profile_source
    assert "elastic_aclgraph_moe_blocks" in memory_profile_source
    assert "_is_ascend_fused_moe_module" in memory_profile_source
    assert "getattr(module, 'quant_method', None)" in memory_profile_source
    assert "getattr(candidate, 'experts', None) is module" in memory_profile_source
    assert "method.use_aclgraph = False" in memory_profile_source
    assert "method.use_aclgraph = old_value" in memory_profile_source
    assert "block.use_elastic_aclgraph = False" in memory_profile_source
    assert "block.use_elastic_aclgraph = old_value" in memory_profile_source
    assert "could not pair every" in memory_profile_source

    determine_memory = _class_method(
        _tree(WORKER), "NPUWorker", "determine_available_memory"
    )
    determine_memory_source = _source(determine_memory)
    assert determine_memory_source.index(
        "before_initial_memory_profile"
    ) < determine_memory_source.index("self.model_runner.profile_run()")
    assert determine_memory_source.index(
        "self.model_runner.profile_run()"
    ) < determine_memory_source.index("after_initial_memory_profile")

    rollout = _class_method(_tree(ROLLOUT), "vLLMRollout", "generate_sequences")
    assert _call_line(rollout, "_maybe_resize_mode1_kv_cache_from_meta") < _call_line(
        rollout, "_recapture_rollout_aclgraphs"
    )

    shrink = _class_method(_tree(WORKER), "NPUWorker", "rebuild_elastic_ep_group")
    assert _call_line(shrink, "_refresh_elastic_parallel_state") < _call_line(
        shrink, "_warmup_post_shrink_moe_dispatch"
    )
    assert _call_line(shrink, "_warmup_post_shrink_moe_dispatch") < _call_line(
        shrink, "_recapture_full_moe_elastic_aclgraph"
    )


def test_aclgraph_does_not_receive_torchair_only_configuration() -> None:
    init = _class_method(_tree(ROLLOUT), "vLLMRollout", "__init__")
    source = _source(init)
    raw_source = ROLLOUT.read_text(encoding="utf-8")
    assert 'torchair_enabled = not config.enforce_eager and _env_flag(' in source
    assert "VLLM_ENABLE_GRAPH_MODE" in source
    assert 'torchair_graph_config = {' in source
    assert "} if torchair_enabled else {" in raw_source
    assert '"enabled": False' in raw_source
    assert '"torchair_graph_config": torchair_graph_config' in raw_source


def test_elastic_aclgraph_env_reaches_taskrunner_and_ray_workers() -> None:
    for path in (PPO_RUNTIME_ENV, RAY_WORKER):
        assert "VLLM_ASCEND_ELASTIC_ACLGRAPH" in path.read_text(
            encoding="utf-8"
        ), f"elastic ACLGraph policy is not passed through by {path.relative_to(ROOT)}"


def test_aclgraph_weak_ref_extension_is_pinned_and_fail_closed() -> None:
    source = ASCEND_UTILS.read_text(encoding="utf-8")
    tree = _tree(ASCEND_UTILS)
    loader = _definition(tree, "ensure_ascend_weak_ref_tensor_op")
    loader_source = _source(loader)
    assert "VLLM_ASCEND_ELASTIC_ACLGRAPH_EXTENSION" in loader_source
    assert "VLLM_ASCEND_ELASTIC_ACLGRAPH_EXTENSION_SHA256" in loader_source
    assert "hashlib.sha256()" in loader_source
    assert "torch.ops.load_library(extension_path)" in loader_source
    assert "Pinned vLLM Ascend extension did not register" in source
    assert "return tensor" not in loader_source

    weak_ref = _definition(tree, "weak_ref_tensor")
    assert _call_line(weak_ref, "ensure_ascend_weak_ref_tensor_op") < _call_line(
        weak_ref, "weak_ref_tensor"
    )


def test_aclgraph_launcher_contract() -> None:
    assert GRAPH_LAUNCHER.is_file()
    subprocess.run(["bash", "-n", str(GRAPH_LAUNCHER)], check=True)
    source = GRAPH_LAUNCHER.read_text(encoding="utf-8")

    assert '${ADAFLOOR_GRAPH_MODE:-elastic_aclgraph}' in source
    assert '${ADAFLOOR_ACLGRAPH_MODE:-FULL_DECODE_ONLY}' in source
    assert '${ADAFLOOR_GRAPH_CAPTURE_PROFILE:-balanced}' in source
    assert 'memory_saver)' in source
    assert "CAPTURE_SIZES='[1,2,4,8]'" in source
    assert 'balanced)' in source
    assert "CAPTURE_SIZES='[1,2,4,8,16,32]'" in source
    assert 'full_coverage)' in source
    assert "CAPTURE_SIZES='[1,2,4,8,16,32,64]'" in source
    assert 'CAPTURE_PROFILE=custom' in source
    assert "ADAFLOOR_GRAPH_BASE_RUNNER" in source
    defaults = {
        "VLLM_ASCEND_ELASTIC_ACLGRAPH": "1",
        "VLLM_ASCEND_ELASTIC_ACLGRAPH_CAPTURE_ATTENTION": "1",
        "VLLM_ASCEND_ELASTIC_ACLGRAPH_CAPTURE_MOE": "1",
        "VLLM_ENABLE_GRAPH_MODE": "0",
        "TASK_QUEUE_ENABLE": "1",
        "ROLLOUT_ENFORCE_EAGER": "False",
        "VERL_SIDECAR_ENABLE": "0",
        "VERL_HCCL_IF_BASE_PORT_START": "12000",
        "VERL_MASTER_PORT_START": "28416",
    }
    for variable, value in defaults.items():
        assert f'export {variable}="${{{variable}:-{value}}}"' in source
    assert "VLLM_ENABLE_GRAPH_MODE=1" not in source
    assert "VLLM_ASCEND_ELASTIC_ACLGRAPH_EXTENSION" in source
    assert "VLLM_ASCEND_ELASTIC_ACLGRAPH_EXTENSION_SHA256" in source
    assert 'sha256sum "$VLLM_ASCEND_ELASTIC_ACLGRAPH_EXTENSION"' in source

    hydra_override = "actor_rollout_ref.rollout.cudagraph_capture_sizes=$CAPTURE_SIZES"
    assert hydra_override in source
    assert source.index(hydra_override) < source.rindex('"$@"')
    graph_mode_override = "actor_rollout_ref.rollout.cudagraph_mode=$ACLGRAPH_MODE"
    assert graph_mode_override in source
    assert source.index(graph_mode_override) < source.rindex('"$@"')
    shared_expert_override = (
        "actor_rollout_ref.actor.megatron.override_transformer_config."
        "moe_shared_expert_overlap=False"
    )
    assert shared_expert_override in source
    assert source.index(shared_expert_override) < source.rindex('"$@"')


def test_aclgraph_launcher_resolves_profiles_and_preserves_user_override(
    tmp_path: Path,
) -> None:
    fake_runner = tmp_path / "runner.sh"
    fake_runner.write_text(
        "#!/usr/bin/env bash\n"
        "printf 'elastic=%s attention=%s moe=%s torchair=%s queue=%s eager=%s sidecar=%s\\n' "
        '"$VLLM_ASCEND_ELASTIC_ACLGRAPH" '
        '"$VLLM_ASCEND_ELASTIC_ACLGRAPH_CAPTURE_ATTENTION" '
        '"$VLLM_ASCEND_ELASTIC_ACLGRAPH_CAPTURE_MOE" '
        '"$VLLM_ENABLE_GRAPH_MODE" '
        '"$TASK_QUEUE_ENABLE" "$ROLLOUT_ENFORCE_EAGER" "$VERL_SIDECAR_ENABLE"\n'
        "printf 'args=%s\\n' \"$*\"\n",
        encoding="utf-8",
    )
    extension = tmp_path / "vllm_ascend_C.so"
    extension.write_bytes(b"test extension")
    environment = os.environ.copy()
    environment.update(
        {
            "ADAFLOOR_GRAPH_BASE_RUNNER": str(fake_runner),
            "VLLM_ASCEND_ELASTIC_ACLGRAPH_EXTENSION": str(extension),
        }
    )

    result = subprocess.run(
        [
            str(GRAPH_LAUNCHER),
            "actor_rollout_ref.rollout.cudagraph_capture_sizes=[2]",
        ],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )
    assert "capture_profile=balanced capture_sizes=[1,2,4,8,16,32]" in result.stdout
    assert (
        "elastic=1 attention=1 moe=1 torchair=0 queue=1 eager=False sidecar=0"
        in result.stdout
    )
    assert (
        "args=actor_rollout_ref.rollout.cudagraph_mode=FULL_DECODE_ONLY "
        "actor_rollout_ref.rollout.cudagraph_capture_sizes=[1,2,4,8,16,32] "
        "actor_rollout_ref.actor.megatron.override_transformer_config."
        "moe_shared_expert_overlap=False "
        "actor_rollout_ref.rollout.cudagraph_capture_sizes=[2]"
    ) in result.stdout

    environment["ADAFLOOR_GRAPH_CAPTURE_SIZES"] = "[1,4]"
    environment["ASCEND_CACHE_PATH"] = str(tmp_path / "missing" / "ascend")
    environment["ASCEND_WORK_PATH"] = str(tmp_path / "missing" / "ascend_work")
    custom = subprocess.run(
        [str(GRAPH_LAUNCHER)],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )
    assert "capture_profile=custom capture_sizes=[1,4]" in custom.stdout
    assert Path(environment["ASCEND_CACHE_PATH"]).is_dir()
    assert Path(environment["ASCEND_WORK_PATH"]).is_dir()
    assert (
        "args=actor_rollout_ref.rollout.cudagraph_mode=FULL_DECODE_ONLY "
        "actor_rollout_ref.rollout.cudagraph_capture_sizes=[1,4] "
        "actor_rollout_ref.actor.megatron.override_transformer_config."
        "moe_shared_expert_overlap=False"
    ) in custom.stdout


def test_aclgraph_launcher_rejects_unknown_capture_profile(tmp_path: Path) -> None:
    extension = tmp_path / "vllm_ascend_C.so"
    extension.write_bytes(b"test extension")
    environment = os.environ.copy()
    environment.update(
        {
            "ADAFLOOR_GRAPH_CAPTURE_PROFILE": "unknown",
            "VLLM_ASCEND_ELASTIC_ACLGRAPH_EXTENSION": str(extension),
        }
    )
    result = subprocess.run(
        [str(GRAPH_LAUNCHER)],
        capture_output=True,
        text=True,
        env=environment,
    )
    assert result.returncode == 2
    assert "unsupported ADAFLOOR_GRAPH_CAPTURE_PROFILE=unknown" in result.stderr
