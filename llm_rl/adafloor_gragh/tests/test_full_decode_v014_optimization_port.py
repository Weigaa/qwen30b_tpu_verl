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


def test_optimized_vanilla_launcher_is_a_sealed_same_workload_profile() -> None:
    source = OPT_GRAPH_RUNNER.read_text(encoding="utf-8")
    base = BASE_GRAPH_RUNNER.read_text(encoding="utf-8")
    for marker in (
        "FULL_DECODE_OPTIMIZATION_PROFILE=v014_runtime_port",
        "VLLM_ROLLOUT_NATIVE_SLEEP_MODE=1",
        "VLLM_ROLLOUT_SLEEP_LEVEL=1",
        "VLLM_ROLLOUT_REUSE_ACLGRAPH_AFTER_WEIGHT_UPDATE=1",
        "VLLM_ROLLOUT_ASYNC_SCHEDULING=true",
        "VLLM_ROLLOUT_ENABLE_PREFIX_CACHING=true",
        "VLLM_ROLLOUT_ENABLE_CHUNKED_PREFILL=true",
        "VLLM_ASCEND_USE_FILTERED_CUSTOM_OPP=1",
        "TASK_QUEUE_ENABLE=1",
    ):
        assert marker in source
    assert "filtered_custom_opp_bundle_sha256" in base
    assert "FULL_DECODE_EXTRA_CODE_PATH" in base
    assert "VLLM_ASCEND_DISABLE_GRAPH_FUSION" not in source
    subprocess.run(["bash", "-n", str(OPT_GRAPH_RUNNER)], check=True)
    subprocess.run(["bash", "-n", str(BASE_GRAPH_RUNNER)], check=True)
    subprocess.run(["bash", "-n", str(TRAIN_LAUNCHER)], check=True)
