import copy
import importlib.util
import json
from pathlib import Path

import pytest


spec = importlib.util.spec_from_file_location(
    "p0_finalizer", Path(__file__).parents[1] / "analysis_eval/finalize_hpca_p0_result.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def complete_summary():
    return {
        "status": "COMPLETE_OBSERVATIONS",
        "variants": [{"epochs": 4, "training_steps": 20, "responses": 10240,
                      "aborted_responses": 0, "preemptions": 0, "ooms": 0,
                      "short_non_eos_responses": 0, "over_cap_responses": 0, "termination_audit_unknown": 0}],
        "steps": [
            {"epoch": e, "epoch_step": s, "responses": 512, "scored_responses": 512,
             "actor_lr": 1e-6, "reference_kl_loss": 0.001, "step_s": 100.0,
             "logged_kl": -1e-11, "reward_mean": 0.8, "logged_aborted_ratio": 0,
             "reward_sum": 409.6, "generated_tokens": 51200, "response_cap_hits": 1,
             "rollout_sha256": "a" * 64, "length_sha256": "b" * 64,
             "log_sha256": "c" * 64}
            for e in range(1, 5) for s in range(1, 6)
        ],
    }


def test_complete_summary_passes():
    module.verify_summary("test", complete_summary())


@pytest.mark.parametrize("fault", ["duplicate", "missing_counter", "frozen", "nan", "missing_sha", "termination"])
def test_incomplete_summary_rejected(fault):
    data = complete_summary()
    if fault == "duplicate":
        data["steps"][-1] = copy.deepcopy(data["steps"][0])
    elif fault == "missing_counter":
        del data["variants"][0]["preemptions"]
    elif fault == "frozen":
        data["steps"][0]["actor_lr"] = 0
    elif fault == "nan":
        data["steps"][0]["reference_kl_loss"] = float("nan")
    elif fault == "missing_sha":
        del data["steps"][0]["log_sha256"]
    else:
        data["variants"][0]["termination_audit_unknown"] = 1
    with pytest.raises(RuntimeError):
        module.verify_summary("test", data)


def test_ppo_kl_precision_preserved():
    assert "e-11" in module.fmt(-2.4e-11)


def test_epoch_curve_uses_response_weighted_reward_and_tokens():
    curve = module.epoch_curve(complete_summary())
    assert len(curve) == 4
    assert curve[0]["reward_mean"] == pytest.approx(0.8)
    assert curve[0]["response_length_mean"] == 100
    assert curve[0]["cap_hits"] == 5


def test_cleanup_refuses_unowned_root(tmp_path):
    with pytest.raises(RuntimeError, match="unexpected run root"):
        module.cleanup_relocated_checkpoints(tmp_path / "arbitrary", tmp_path)


def test_cleanup_refuses_mismatched_relocation_audit(tmp_path):
    run = tmp_path / "hpca_p0_long_horizon_test" / "run"
    epoch = run / "epoch_001_mode1_planned"
    epoch.mkdir(parents=True)
    (epoch / "CHECKPOINT_RELOCATION.txt").write_text("archive=/tmp/not-owned\n")
    with pytest.raises(RuntimeError, match="ownership mismatch"):
        module.cleanup_relocated_checkpoints(run, tmp_path)


def test_public_checkpoint_missing_shard_is_rejected(tmp_path):
    model = tmp_path / "epoch_000_mode0_probe/checkpoints/qwen3moe_for_eagle3"
    dist = model / "global_step_5/actor/dist_ckpt"
    dist.mkdir(parents=True)
    (model / "latest_checkpointed_iteration.txt").write_text("5\n")
    (dist / "metadata.json").write_text(json.dumps({"valid": True}))
    for i in range(32):
        (dist / f"{i}.distcp").write_bytes(b"fixture")
    assert module.verify_public_checkpoint(tmp_path)["shard_count"] == 32
    (dist / "31.distcp").unlink()
    with pytest.raises(RuntimeError, match="shards are incomplete"):
        module.verify_public_checkpoint(tmp_path)
