import json
from pathlib import Path
import pytest
from native_eager_handoff import validate


def example():
    row = dict(step=1,rank=0,complete=True,kv_blocks=10,kv_tokens=1280,
        worker_kv_blocks=10,eager_iterations=200,full_iterations=0)
    return '\n'.join([
        '(WorkerDict pid=99) 2026-10-24 10:00:01,100 MODE1_STEP_RUNTIME_AUDIT '+json.dumps(row),
        '(WorkerDict pid=99) INFO 10-24 10:00:03 Sleep mode freed 29.65 GiB memory, 9.31 GiB memory is still in use.',
        '(WorkerDict pid=99) INFO 10-24 10:00:03 It took 1.000000 seconds to fall asleep.',
        '(TaskRunner pid=100) 2026-10-24 10:00:05,100 => rollout_output_time_s: 5.0',
        'step:1 - actor/grad_norm:0.02 - actor/lr:1e-06 - response/aborted_ratio:0.0',
        '[run] end_time=2026-10-24T10:01:00+0800 exit_code=0'])


def test_eager_requires_real_sleep_not_graph_events():
    result = validate(example(),1,1)
    assert result['rank_step_handoffs']==1 and not result['graph_invalidation_event_required']


def test_second_resolution_overlap_requires_blocking_return_not_extra_tolerance():
    assert validate(example().replace('10:00:03','10:00:05'),1,1)['rank_step_handoffs']==1
    with pytest.raises(ValueError):validate(example().replace('10:00:03','10:00:06'),1,1)


@pytest.mark.parametrize('old,new',[
    ('Sleep mode freed 29.65','Sleep mode freed 0.00'),
    ('(WorkerDict pid=99) INFO','(WorkerDict pid=98) INFO'),
    ('10:00:03','10:00:00'),
    ('"full_iterations": 0','"full_iterations": 2'),
    ('exit_code=0','exit_code=1'),
    ('response/aborted_ratio:0.0','response/aborted_ratio:1.0'),
])
def test_does_not_accept_missing_or_wrong_native_evidence(old,new):
    with pytest.raises(ValueError):validate(example().replace(old,new),1,1)


def test_cleanup_dispatch_refuses_training_or_graph_use():
    import os
    import subprocess
    wrapper = Path(__file__).parent/'cleanup_bin/python3'
    target = '/workspace/cann-recipes-train/llm_rl/qwen3_014/qwen3_main/tools/validate_native_planned_memory.py'
    for finalize,mode,tq in [('0','NONE','2'),('1','FULL_DECODE_ONLY','2'),('1','NONE','1')]:
        env = dict(os.environ, FAIR_FINALIZE_EXISTING=finalize, ADAFLOOR_ACLGRAPH_MODE=mode,TASK_QUEUE_ENABLE=tq)
        p = subprocess.run(['bash',str(wrapper),target],env=env,capture_output=True,text=True)
        assert p.returncode==2 and 'requires finalize-only NONE/TQ2' in p.stderr
