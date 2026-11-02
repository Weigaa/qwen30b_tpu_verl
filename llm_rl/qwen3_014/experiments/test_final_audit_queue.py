import json
import pytest
from run_final_audit_when_ready import gate


def test_code_result_without_real_completed_queue_cannot_trigger_final_audit(tmp_path):
    (tmp_path/'code_task').mkdir()
    (tmp_path/'CODE_QUEUE_STATE.json').write_text(json.dumps(dict(state='RUNNING_planned_f4')))
    (tmp_path/'code_task/CODE_PAIR_RESULT.json').write_text(json.dumps(dict(status='V014_CODE_DOMAIN_FROZEN_PAIR_COMPLETE')))
    with pytest.raises(RuntimeError):gate(tmp_path)
    (tmp_path/'CODE_QUEUE_STATE.json').write_text(json.dumps(dict(state='COMPLETE')))
    gate(tmp_path)
    (tmp_path/'code_task/CODE_PAIR_RESULT.json').unlink()
    with pytest.raises(RuntimeError):gate(tmp_path)
