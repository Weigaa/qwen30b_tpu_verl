from __future__ import annotations

import ast
import subprocess
from pathlib import Path


ROOT = Path(__file__).parents[1]
ROLLOUT = ROOT / "verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py"
ROLLOUT_CONFIG = ROOT / "verl/workers/config/rollout.py"
TRAIN_LAUNCHER = (
    ROOT / "internal/wj_train_grpo_qwen30b_a3b_16die_true_weight_regroup.sh"
)
BASE_GRAPH_RUNNER = ROOT / "run_qwen3_vanilla_epoch0_full_decode_fia_tq1.sh"
OPT_GRAPH_RUNNER = (
    ROOT / "run_qwen3_vanilla_epoch0_full_decode_fia_v014opt_tq1.sh"
)
MEGATRON_WORKER = ROOT / "verl/workers/megatron_workers.py"
PPO_ENV = ROOT / "verl/trainer/constants_ppo.py"
RAY_ENV = ROOT / "verl/single_controller/ray/base.py"
ASCEND_MODEL_RUNNER = ROOT / "vllm_ascend/worker/model_runner_v1.py"
ASCEND_WORKER = ROOT / "vllm_ascend/worker/worker_v1.py"


def _tree(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _class(tree: ast.AST, name: str) -> ast.ClassDef:
    found = [node for node in ast.walk(tree)
             if isinstance(node, ast.ClassDef) and node.name == name]
    assert len(found) == 1
    return found[0]


def _method(tree: ast.AST, cls: str, name: str) -> ast.AST:
    found = [node for node in _class(tree, cls).body
             if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
             and node.name == name]
    assert len(found) == 1
    return found[0]


def test_rollout_schema_and_llm_receive_scheduler_optimizations() -> None:
    config_source = ROLLOUT_CONFIG.read_text(encoding="utf-8")
    assert "async_scheduling: Optional[bool] = None" in config_source

    init_source = ast.unparse(_method(_tree(ROLLOUT), "vLLMRollout", "__init__"))
    assert "async_scheduling=config.async_scheduling" in init_source
    assert "enable_prefix_caching=config.enable_prefix_caching" in init_source
    assert "enable_chunked_prefill=config.enable_chunked_prefill" in init_source
    assert "'enable_chunked_prefill': config.enable_chunked_prefill" in init_source
    assert "enable_prefix_caching=False" not in init_source


def test_native_sleep_reuses_graph_only_with_stable_addresses() -> None:
    tree = _tree(ROLLOUT)
    init_source = ast.unparse(_method(tree, "vLLMRollout", "__init__"))
    resume_source = ast.unparse(_method(tree, "vLLMRollout", "resume"))
    release_source = ast.unparse(_method(tree, "vLLMRollout", "release"))
    update_source = ast.unparse(_method(tree, "vLLMRollout", "update_weights"))

    assert "VLLM_ROLLOUT_NATIVE_SLEEP_MODE" in init_source
    assert "elastic_execution_mode != 0" in init_source
    assert "enable_sleep_mode=config.free_cache_engine and self.native_sleep_mode" in init_source
    assert "self.inference_engine.sleep(level=self.sleep_level)" in init_source
    assert "self.inference_engine.wake_up(tags=tags)" in resume_source
    assert "native_sleep_pointer_change" in resume_source
    assert "native_sleep_kv_pointer_change" in resume_source
    assert "same-floor KV preservation" in release_source
    assert "VLLM_ROLLOUT_REUSE_ACLGRAPH_AFTER_WEIGHT_UPDATE" in update_source
    assert "weight_ptr_signature_before == weight_ptr_signature_after" in update_source
    assert "update_weights_after_finalize" in update_source


def test_unavailable_camem_fails_with_an_actionable_error() -> None:
    source = (ROOT / "vllm_ascend/device_allocator/camem.py").read_text(
        encoding="utf-8")
    assert "if not camem_available:" in source
    assert "Disable VLLM_ROLLOUT_NATIVE_SLEEP_MODE" in source
    assert '"vllm_ascend_C against the active PyTorch ABI.' in source


def test_allocator_stays_non_expandable_during_native_sleep() -> None:
    launcher = TRAIN_LAUNCHER.read_text(encoding="utf-8")
    worker = MEGATRON_WORKER.read_text(encoding="utf-8")
    assert "MEGATRON_CI_DISABLE_EXPANDABLE_SEGMENTS" in launcher
    assert "garbage_collection_threshold:0.6,max_split_size_mb:24" in launcher
    assert 'os.environ.get("MEGATRON_CI_DISABLE_EXPANDABLE_SEGMENTS", "0")' in worker


def test_training_launcher_propagates_scheduler_and_filtered_opp() -> None:
    source = TRAIN_LAUNCHER.read_text(encoding="utf-8")
    for marker in (
        "VLLM_ASCEND_USE_FILTERED_CUSTOM_OPP",
        "VLLM_ASCEND_FILTERED_CUSTOM_OPP_PATH",
        "actor_rollout_ref.rollout.async_scheduling=${VLLM_ROLLOUT_ASYNC_SCHEDULING}",
        "actor_rollout_ref.rollout.enable_prefix_caching=${VLLM_ROLLOUT_ENABLE_PREFIX_CACHING}",
        "actor_rollout_ref.rollout.enable_chunked_prefill=${VLLM_ROLLOUT_ENABLE_CHUNKED_PREFILL}",
    ):
        assert marker in source


def test_optimized_runtime_environment_crosses_both_ray_boundaries() -> None:
    for path in (PPO_ENV, RAY_ENV):
        source = path.read_text(encoding="utf-8")
        for marker in (
            "VLLM_ROLLOUT_NATIVE_SLEEP_MODE",
            "VLLM_ROLLOUT_REUSE_ACLGRAPH_AFTER_WEIGHT_UPDATE",
            "ASCEND_CUSTOM_OPP_PATH",
            "MEGATRON_CI_DISABLE_EXPANDABLE_SEGMENTS",
            "PYTORCH_NPU_ALLOC_CONF",
            "VLLM_ASCEND_ENABLE_TOPK_TOPP_OPTIMIZATION",
        ):
            assert marker in source

    ppo_source = PPO_ENV.read_text(encoding="utf-8")
    assert "VERL_FULL_DECODE_MAX_UPDATE_ACTOR_SECONDS" in ppo_source


def test_optimized_vanilla_launcher_is_a_sealed_same_workload_profile() -> None:
    source = OPT_GRAPH_RUNNER.read_text(encoding="utf-8")
    base = BASE_GRAPH_RUNNER.read_text(encoding="utf-8")
    for marker in (
        "FULL_DECODE_OPTIMIZATION_PROFILE=v014_runtime_port",
        "VLLM_ROLLOUT_NATIVE_SLEEP_MODE=0",
        "unset VLLM_ROLLOUT_SLEEP_LEVEL",
        "VLLM_ROLLOUT_REUSE_ACLGRAPH_AFTER_WEIGHT_UPDATE=1",
        "VLLM_ROLLOUT_ASYNC_SCHEDULING=false",
        "VLLM_ROLLOUT_ENABLE_PREFIX_CACHING=true",
        "VLLM_ROLLOUT_ENABLE_CHUNKED_PREFILL=true",
        "VLLM_ASCEND_USE_FILTERED_CUSTOM_OPP=1",
        "TASK_QUEUE_ENABLE=1",
        "unset MEGATRON_CI_DISABLE_EXPANDABLE_SEGMENTS",
        "PYTORCH_NPU_ALLOC_CONF=expandable_segments:True",
        "VERL_FULL_DECODE_MAX_UPDATE_ACTOR_SECONDS",
    ):
        assert marker in source
    assert "max_split_size_mb:24" not in source
    assert "filtered_custom_opp_bundle_sha256" in base
    assert "rollout_memory_lifecycle=manual_weight_kv_release" in base
    assert "pytorch_npu_alloc_conf=" in base
    assert "max_update_actor_seconds=" in base
    assert "verl/trainer/ppo/ray_trainer.py" in base
    assert "vllm_ascend/device_allocator/camem.py" in base
    assert "FULL_DECODE_EXTRA_CODE_PATH" in base
    assert "VLLM_ASCEND_DISABLE_GRAPH_FUSION" not in source
    subprocess.run(["bash", "-n", str(OPT_GRAPH_RUNNER)], check=True)
    subprocess.run(["bash", "-n", str(BASE_GRAPH_RUNNER)], check=True)
    subprocess.run(["bash", "-n", str(TRAIN_LAUNCHER)], check=True)


def test_full_decode_actor_update_performance_guard_is_opt_in() -> None:
    trainer = (ROOT / "verl/trainer/ppo/ray_trainer.py").read_text(
        encoding="utf-8")
    marker = "VERL_FULL_DECODE_MAX_UPDATE_ACTOR_SECONDS"
    assert marker in trainer
    assert 'timing_raw["update_actor"]' in trainer
    assert "FULL_DECODE actor-update performance guard " in trainer
    assert "failed at step" in trainer
    assert "must not continue" in trainer
    normalized_marker = "VERL_FULL_DECODE_MAX_UPDATE_ACTOR_MS_PER_TOKEN"
    assert normalized_marker in trainer
    assert "batch.meta_info.get(" in trainer
    assert '"global_token_num"' in trainer
    assert 'timing_raw["update_actor"] * 1000.0' in trainer
    assert "token-normalized actor-update" in trainer
    assert normalized_marker in PPO_ENV.read_text(encoding="utf-8")


def test_full_decode_dp_dummy_replays_uniform_decode_graph() -> None:
    worker = ASCEND_WORKER.read_text(encoding="utf-8")
    runner = ASCEND_MODEL_RUNNER.read_text(encoding="utf-8")

    assert "self.model_runner.decode_token_per_req" in worker
    assert "force_attention=True" in worker
    assert "uniform_decode=True" in worker
    assert "num_actual_tokens_override=0" in worker
    assert "dispatch_aclgraph_after_dp_sync=True" in worker
    assert "torch.npu.synchronize()" in worker[worker.index(
        "def execute_dummy_batch"):]
    assert "if dispatch_aclgraph_after_dp_sync:" in runner
    assert "is_runtime_dummy=True" in runner
    live_start = runner.index("def execute_model(")
    live_end = runner.index("def _build_attention_metadata", live_start)
    live_source = runner[live_start:live_end]
    assert "_global_has_runtime_dummy" in live_source
    assert "did not dispatch" in live_source
    assert "aclgraph_runtime_mode = CUDAGraphMode.NONE" not in live_source

    dummy_start = runner.index("def _dummy_run(")
    dummy_end = runner.index("def set_in_profile_run", dummy_start)
    dummy_source = runner[dummy_start:dummy_end]
    sync_pos = runner.index("if self.need_allreduce:", dummy_start)
    dispatch_pos = runner.index("if dispatch_aclgraph_after_dp_sync:", sync_pos)
    pad_pos = runner.index("self.vllm_config.pad_for_cudagraph(num_tokens)",
                           dispatch_pos)
    model_dummy_pos = runner.index("model_is_dummy = (None if", pad_pos)
    moe_pos = runner.index("moe_comm_type = self._select_moe_comm_method(",
                           model_dummy_pos)
    assert sync_pos < dispatch_pos < pad_pos < model_dummy_pos < moe_pos
    assert "is_dummy=bool(model_is_dummy)" in runner[model_dummy_pos:]
    assert "if (model_is_dummy is not None" in runner
    assert "inputs_embeds, model_is_dummy)" in runner[model_dummy_pos:]
    assert "runtime dummy could not dispatch" in dummy_source
    assert "runtime dummy joined the " in dummy_source
    assert "aclgraph_runtime_mode = CUDAGraphMode.NONE" not in dummy_source
    assert "is_graph_capturing: bool = False" in dummy_source
    assert "is_graph_capturing=is_graph_capturing" in dummy_source
    assert "self.query_start_loc_np[1:num_reqs + 1] = cu_num_tokens" in dummy_source
    dummy_hidden_start = runner.index("def _generate_dummy_run_hidden_states(")
    dummy_hidden_end = runner.index("def _dummy_run(", dummy_hidden_start)
    dummy_hidden_source = runner[dummy_hidden_start:dummy_hidden_end]
    assert "update_attn_params(self.update_stream, forward_context" in (
        dummy_hidden_source)

    capture_start = runner.index("def _capture_aclgraphs(")
    capture_end = runner.index("def _capture_model(", capture_start)
    capture_source = runner[capture_start:capture_end]
    assert "is_graph_capturing=True" in capture_source


def test_full_decode_preemption_recovery_uses_dp_consensus_eager_fallback() -> None:
    runner = ASCEND_MODEL_RUNNER.read_text(encoding="utf-8")

    assert "def _full_decode_dp_eager_fallback_required(" in runner
    assert "bool(global_with_prefill)" in runner
    assert "int(synced_num_tokens) > max_capture_tokens" in runner
    assert "def _assert_full_decode_dp_eager_fallback_consensus(" not in runner

    live_start = runner.index("def execute_model(")
    live_end = runner.index("def _build_attention_metadata", live_start)
    live_source = runner[live_start:live_end]
    assert "eager_fallback_required = (" in live_source
    assert "not eager_fallback_required" in live_source
    assert "existing DP metadata consensus" in live_source
    assert "local_resumed_from_preemption = any(" in live_source
    assert "local preemption recovery joined the " in live_source
    assert "DP rank entered synchronized eager " in live_source
    assert "DP rank resumed FULL graph after " in live_source
    assert "_full_decode_eager_recovery_active = True" in live_source
    assert "_full_decode_eager_recovery_active = False" in live_source

    dummy_start = runner.index("def _dummy_run(")
    dummy_end = runner.index("def set_in_profile_run", dummy_start)
    dummy_source = runner[dummy_start:dummy_end]
    assert "if eager_fallback_required:" in dummy_source
    assert "uniform_decode = False" in dummy_source
    assert "force_attention = False" in dummy_source
    assert "runtime dummy recovery" in dummy_source
    assert "iteration unexpectedly dispatched a graph" in dummy_source
    assert "existing DP metadata consensus" in dummy_source
    assert "runtime_dummy=1" in dummy_source
