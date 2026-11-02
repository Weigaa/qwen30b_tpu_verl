from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "run_qwen3_baseline_eager_tq2.sh"


def test_baseline_removes_adafloor_rollout_config() -> None:
    source = SCRIPT.read_text()

    assert "~actor_rollout_ref.rollout.shrink_aware" in source
    assert "actor_rollout_ref.rollout.shrink_aware." not in source


def test_baseline_execution_contract() -> None:
    source = SCRIPT.read_text()

    assert "export TASK_QUEUE_ENABLE=2" in source
    assert "export VLLM_ENABLE_GRAPH_MODE=0" in source
    assert "actor_rollout_ref.rollout.enforce_eager=True" in source
    assert "actor_rollout_ref.rollout.n=16" in source
    assert "data.train_batch_size=32" in source
