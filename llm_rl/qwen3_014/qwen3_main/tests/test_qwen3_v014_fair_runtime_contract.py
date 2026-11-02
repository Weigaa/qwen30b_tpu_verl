from __future__ import annotations

import ast
from pathlib import Path

import pytest

from vllm_ascend import envs


ROOT = Path(__file__).resolve().parents[1]


def test_mode1_redundancy_is_derived_from_floor(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VLLM_ASCEND_ENABLE_ELASTIC_PARALLEL_SHRINK", "1")
    monkeypatch.setenv("VLLM_ASCEND_ELASTIC_EXECUTION_MODE", "1")
    monkeypatch.setenv("VLLM_ASCEND_ELASTIC_MIN_COMPUTE_GROUP_SIZE", "4")
    assert envs.compute_elastic_init_redundancy_expert(128, 16, 0) == 384
    monkeypatch.setenv("VLLM_ASCEND_ELASTIC_MIN_COMPUTE_GROUP_SIZE", "2")
    assert envs.compute_elastic_init_redundancy_expert(128, 16, 0) == 896


def test_explicit_moe_backend_rejects_elastic_modern(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VLLM_ASCEND_ENABLE_ELASTIC_PARALLEL_SHRINK", "1")
    monkeypatch.setenv("VLLM_ASCEND_FUSED_MOE_BACKEND", "modern")
    monkeypatch.setenv("VLLM_ASCEND_USE_LEGACY_FUSED_MOE", "0")
    with pytest.raises(RuntimeError, match="does not expose AdaFloor"):
        envs.resolve_fused_moe_backend()
    monkeypatch.setenv("VLLM_ASCEND_FUSED_MOE_BACKEND", "legacy")
    assert envs.resolve_fused_moe_backend() == "legacy"


def test_planned_group_precreation_happens_before_kv_sizing() -> None:
    source = (ROOT / "vllm_ascend/worker/worker_v1.py").read_text()
    tree = ast.parse(source)
    functions = {
        node.name: node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
    }
    determine = functions["determine_available_memory"]
    calls = {
        getattr(node.func, "attr", ""): node.lineno
        for node in ast.walk(determine)
        if isinstance(node, ast.Call)
    }
    assert calls["profile_run"] < calls[
        "_precreate_mode1_planned_floor_groups_before_kv_cache"
    ] < calls["mem_get_info"]
    precreate = functions["_precreate_mode1_planned_floor_groups_before_kv_cache"]
    precreate_calls = {
        getattr(node.func, "attr", "") for node in ast.walk(precreate)
        if isinstance(node, ast.Call)
    }
    assert "_validate_mode1_preloaded_expert_slots" in precreate_calls
    assert "_warmup_precreated_mode1_group" in precreate_calls
    assert "_rebuild_group" in precreate_calls


def test_fair_runner_pins_the_per_mode_fast_contract() -> None:
    source = (ROOT / "run_qwen3_v014_fair_one_step_matrix.sh").read_text()
    for expected in (
        "VLLM_ASCEND_FUSED_MOE_BACKEND=legacy",
        "VLLM_ASCEND_INIT_REDUNDANCY_EXPERT=0",
        "VLLM_ASCEND_MODE1_PRECREATE_PLANNED_FLOOR_GROUPS",
        "VLLM_ASCEND_FUSED_MOE_SIMPLE_MC2=1",
        "VLLM_ASCEND_ATTENTION_BLOCK_SIZE=64",
        "VLLM_ROLLOUT_TASK_QUEUE_ENABLE=1",
        "VLLM_ROLLOUT_TASK_QUEUE_ENABLE=2",
        "cudagraph_mode=FULL_DECODE_ONLY",
    ):
        assert expected in source
    assert "VLLM_ROLLOUT_ASYNC_SCHEDULING=false" in source
    assert "source-plan-step \"$SOURCE_PLAN_STEP\"" in source
