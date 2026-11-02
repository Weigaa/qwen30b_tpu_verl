import pytest
from reconstruct_code_taskrunner_log import derive


def fixture():
    rows=[]
    for step in (1,2):
        rows.extend([f'2026-10-24 18:00:0{step},000 => rollout_output_time_s: {step*10:.6f}',
            f'step:{step} - actor/lr:0.0 - actor/grad_norm:0.0 - response/aborted_ratio:0.0 - rollout/rollout_output_time_s:{step*10}.0'])
    raw='\n'.join(rows)
    original='Training Progress: 100%\n'+rows[0]+'\n'+rows[1]+'\nWORKER_ACTUAL_RECORD\n[run] end_time=2026-10-24T18:00:30+0800 exit_code=0\n'
    return original,raw


def test_only_real_raw_missing_events_are_restored_without_changing_worker_records():
    original,raw=fixture();view,proof=derive(original,raw,99,steps=2)
    assert view.count('=> rollout_output_time_s:')==2 and 'WORKER_ACTUAL_RECORD' in view
    assert 'DERIVED OFFLINE AUDIT VIEW' in view and proof['raw_metric_step_ids']==[1,2]
    assert proof['replaced_forwarded_records']==2


def test_no_fake_success_or_conflicting_metrics_allowed():
    original,raw=fixture()
    with pytest.raises(ValueError):derive(original.replace('exit_code=0','exit_code=1'),raw,99,steps=2)
    with pytest.raises(ValueError):derive(original,raw,99,steps=3)
    with pytest.raises(ValueError):derive(original.replace('actor/lr:0.0','actor/lr:1.0'),raw,99,steps=2)
    with pytest.raises(ValueError):derive(original.replace('Training Progress: 100%',''),raw,99,steps=2)
