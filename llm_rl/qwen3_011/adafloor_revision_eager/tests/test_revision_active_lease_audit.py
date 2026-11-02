"""Synthetic logs test the auditor only; they are not NPU evidence."""
import importlib.util
from pathlib import Path
import pytest

spec = importlib.util.spec_from_file_location("lease_audit", Path(__file__).parents[1] / "analysis_eval/audit_revision_active_sidecar_window.py")
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def fixture():
    lease = "sidecar_active_ranks=8,9,10,11,12,13,14,15\nsidecar_devices=0,1,2,3,4,5,6,7\n"
    lease += "".join(f"shrink_quorum_progress_time=1 reporter={rank}\n" for rank in range(16))
    for key, time in zip(("coordinated_start_time", "sidecar_start_time", "trainer_stop_request_time", "sidecar_output_merge_time",
                         "sidecar_artifacts_durable_time", "sidecar_exit_confirmed_time", "watcher_restore_ack_time"), range(1, 8)):
        lease += f"{key}={time} process_group_alive=0\n"
    primary = "driver_restore_rpc_start_time=8\nSidecar pre-restore release acknowledged: request_id=nonce lease_id=1 elapsed_s=1\n"
    req = {"request_id": "nonce", "lease_id": "1"}
    ack = dict(req, status="released", ack_time="7")
    active = {"state": "released", "lease_id": "1"}
    rows = [{"outputs": [{"finish_reason": "stop", "token_ids_len": 1}]}]
    return lease, primary, req, ack, active, rows


def test_complete_lease_passes():
    assert audit.validate_lease(*fixture())["completed_queries"] == 1


@pytest.mark.parametrize("kind", ["early_restore", "no_work", "overlap", "wrong_ack", "late_durable", "missing_quorum"])
def test_incomplete_or_unsafe_lease_rejected(kind):
    lease, primary, req, ack, active, rows = fixture()
    if kind == "early_restore": primary = primary.replace("start_time=8", "start_time=4")
    if kind == "no_work": rows = []
    if kind == "overlap": lease = lease.replace("sidecar_devices=0", "sidecar_devices=8")
    if kind == "wrong_ack": ack["request_id"] = "stale"
    if kind == "late_durable": lease = lease.replace("durable_time=5", "durable_time=9")
    if kind == "missing_quorum": lease = lease.replace("reporter=15", "reporter=14")
    with pytest.raises(ValueError): audit.validate_lease(lease, primary, req, ack, active, rows)
