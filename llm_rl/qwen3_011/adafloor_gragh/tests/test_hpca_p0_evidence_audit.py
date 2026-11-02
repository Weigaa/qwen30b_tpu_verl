import importlib.util
import json
from pathlib import Path


def tool(name):
    path = Path(__file__).parents[1] / "analysis_eval" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_e2e_minskew_is_not_planned_minskew(tmp_path):
    module = tool("summarize_hpca_planner_ablation")
    (tmp_path / "summary.json").write_text(json.dumps({"raw_trials": [
        {"policy": "minskew", "trial_seed": 0, "epoch_1_floors": "2,2,4,4,8",
         "mean_epoch_rollout_s": 10, "mean_released_rank_s": 5,
         "preemptions": 0, "ooms": 0}
    ]}))
    rows, _ = module.read_trials(tmp_path)
    assert rows[0]["policy"] == "minskew"
    assert rows[0]["epoch_1_floors"] == "2,2,4,4,8"
    assert module.interval([1.0]) == (None, None)
    assert module.interval([1.0, 2.0, 3.0, 4.0]) == (None, None)


def pair_fixture(tmp_path):
    path = tmp_path / "lease.log"
    path.write_text("\n".join([
        "watch_require_active_lease=1",
        "coordinated_start_time=1 quorum_ranks=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 quorum_count=16",
        "sidecar_start_time=2", "sidecar_stop_request_time=3",
        "sidecar_output_merge_time=4", "sidecar_artifacts_durable_time=5",
        "sidecar_exit_confirmed_time=6",
        "watcher_restore_ack_time=7 request_id=req lease_id=1 status=released",
    ]))
    return {"seed": 101, "off": {"response_count": 512},
            "on": {"response_count": 512, "restore_ack_request_id": "req",
                   "restore_ack_lease_id": "1", "driver_restore_rpc_start_time": 8,
                   "stage8_survivor_ranks": list(range(8, 16)),
                   "deepest_runtime_survivor_ranks": list(range(12, 16))},
            "sidecar": {"used_ranks": list(range(8)), "detached_ranks": list(range(8)),
                        "lease_log": str(path), "restore_handshake_request_id": "req",
                        "restore_handshake_lease_id": "1"}}


def test_valid_sidecar_lifecycle(tmp_path):
    assert all(tool("summarize_hpca_runtime_safety").audit_pair(pair_fixture(tmp_path))["checks"].values())


def test_overlapping_sidecar_rejected(tmp_path):
    pair = pair_fixture(tmp_path)
    pair["sidecar"]["used_ranks"] = [12]
    assert not tool("summarize_hpca_runtime_safety").audit_pair(pair)["checks"]["no_primary_sidecar_overlap"]


def test_early_restore_and_wrong_ack_rejected(tmp_path):
    pair = pair_fixture(tmp_path)
    pair["on"]["driver_restore_rpc_start_time"] = 6
    pair["sidecar"]["restore_handshake_lease_id"] = "2"
    checks = tool("summarize_hpca_runtime_safety").audit_pair(pair)["checks"]
    assert not checks["ack_before_primary_restore"]
    assert not checks["request_and_lease_ack_match"]
