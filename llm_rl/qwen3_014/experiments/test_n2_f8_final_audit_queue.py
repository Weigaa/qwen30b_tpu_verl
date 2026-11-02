import json
import os
import pytest
from run_n2_f8_final_audit_when_ready import ready,signature


def test_only_actual_completed_main_and_dependencies_allow_audit(tmp_path):
    state=tmp_path/'STATE.json';dep=tmp_path/'ALL_DEPENDENCIES_COMPLETE.json'
    state.write_text(json.dumps(dict(state='RUNNING_MATH_N2_SEED3_RECOVER_EPOCH2')))
    dep.write_text(json.dumps(dict(status='ALL_NEW_F8_DEPENDENCIES_VALIDATED')))
    with pytest.raises(RuntimeError):ready(tmp_path)
    state.write_text(json.dumps(dict(state='STOPPED_WITH_ERROR')))
    with pytest.raises(RuntimeError):ready(tmp_path)
    state.write_text(json.dumps(dict(state='ALL_AFFECTED_N2_EXPERIMENTS_COMPLETE')))
    assert ready(tmp_path)
    dep.unlink()
    assert not ready(tmp_path)


def test_process_identity_is_not_a_state_file():
    assert signature(os.getpid()) is not None
    assert signature(99999999) is None
