import ast
import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _function(tree: ast.AST, name: str) -> ast.FunctionDef:
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"function not found: {name}")


def test_rollout_config_is_fail_closed_by_default():
    config = (ROOT / "verl/trainer/config/rollout/rollout.yaml").read_text()

    assert "shrink_aware:" in config
    assert "enable_shrink_aware_scheduling: false" in config
    assert "shrink_aware_mode: off" in config
    assert "target_policy: natural" in config


def test_engine_uses_staged_target_before_group_rebuild():
    source = (ROOT / "vllm/v1/engine/llm_engine.py").read_text()
    tree = ast.parse(source)
    handler = _function(tree, "has_unfinished_requests_dp")
    calls = {
        node.func.attr: node.lineno
        for node in ast.walk(handler)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        and node.func.attr in {"_staged_shrink_target", "collective_rpc"}
    }

    assert calls["_staged_shrink_target"] < calls["collective_rpc"]
    assert "return has_unfinished or self.should_execute_dummy_batch" in source
    assert '"rebuild_elastic_ep_group"' in source
    assert '"restore_elastic_parallel_groups"' in source


def test_trainer_schedules_and_restores_before_ppo_union():
    source = (ROOT / "verl/trainer/ppo/ray_trainer.py").read_text()

    schedule = source.index("shrink_aware_result = maybe_apply_shrink_aware_schedule")
    generate = source.index(
        "gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)",
        schedule,
    )
    response_cap = source.index(
        "response_cap = self._maybe_compute_rollout_response_cap(gen_batch)",
        schedule,
    )
    restore_groups = source.index(
        "self._restore_rollout_elastic_parallel_groups_if_needed()", generate)
    restore_order = source.index(
        "restore_shrink_aware_order(\n                            gen_batch_output",
        restore_groups,
    )
    union = source.index("batch = batch.union(gen_batch_output)", restore_order)

    assert schedule < response_cap < generate < restore_groups < restore_order < union
    assert "currently supports only" in source
    assert "async_scheduling=false" in source
    assert "_maybe_compute_shrink_aware_short_step_response_cap" in source


def test_rollout_worker_installs_step_plan_before_reading_inputs():
    source = (
        ROOT / "verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py"
    ).read_text()
    tree = ast.parse(source)
    generate = _function(tree, "generate_sequences")
    calls = [
        node
        for node in ast.walk(generate)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_sync_shrink_aware_env_from_meta"
    ]

    assert len(calls) == 1
    assert calls[0].lineno < next(
        node.lineno
        for node in ast.walk(generate)
        if isinstance(node, ast.Subscript)
        and isinstance(node.value, ast.Attribute)
        and node.value.attr == "batch"
    )
    assert "VLLM_ASCEND_SHRINK_AWARE_TARGET_POLICY" in source
    assert "VLLM_ASCEND_SHRINK_AWARE_STAGE_RANKS" in source


def test_launcher_is_eager_synchronous_and_preserves_floor_contract():
    launcher = ROOT / "run_qwen3_adafloor_eager_v014.sh"
    base = ROOT / "internal/wj_train_grpo_qwen30b_a3b_16die_true_weight_regroup.sh"
    text = launcher.read_text()
    base_text = base.read_text()

    assert "export VLLM_ENABLE_GRAPH_MODE=0" in text
    assert "export ROLLOUT_ENFORCE_EAGER=True" in text
    assert "export VLLM_ROLLOUT_ASYNC_SCHEDULING=false" in text
    assert "export VLLM_ROLLOUT_EARLY_STOP_ENABLE=0" in text
    assert "VLLM_ASCEND_ELASTIC_MIN_COMPUTE_GROUP_SIZE=${ADAFLOOR_MIN_FLOOR:-4}" in text
    assert "unset VLLM_ASCEND_ELASTIC_MIN_COMPUTE_GROUP_SIZE" in base_text
    assert base_text.index("unset VLLM_ASCEND_ELASTIC_MIN_COMPUTE_GROUP_SIZE") > base_text.index("else")

    result = subprocess.run(
        ["bash", str(launcher), "dry-run"],
        cwd=ROOT,
        env={**os.environ, "ADAFLOOR_TARGET_POLICY": "natural"},
        check=True,
        text=True,
        capture_output=True,
    )
    assert "target_policy=natural" in result.stdout
    assert "enforce_eager=True" in result.stdout
    assert "tailguard=off" in result.stdout

    tailguard_result = subprocess.run(
        ["bash", str(launcher), "dry-run"],
        cwd=ROOT,
        env={**os.environ, "ADAFLOOR_TAIL_GUARD": "1"},
        check=True,
        text=True,
        capture_output=True,
    )
    assert "tailguard=on" in tailguard_result.stdout
