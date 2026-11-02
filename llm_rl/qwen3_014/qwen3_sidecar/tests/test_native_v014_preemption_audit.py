import importlib.util
import json
from pathlib import Path

import pytest


ROOT=Path(__file__).parents[1]
spec=importlib.util.spec_from_file_location('native_recovery_audit',ROOT/'tools/verify_full_decode_preemption_recovery.py')
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def make_log(tmp_path):
    lines=['Initializing a V1 LLM engine (v0.14.1)',
           'Mode1 KV request preempted: rank=0 step=1 request=R',
           'rollout_forward_begin dp_rank=0 iteration=1 global_prefill=True graph_mode=NONE',
           'rollout_forward_begin dp_rank=0 iteration=2 global_prefill=False graph_mode=FULL']
    for r in range(16):
        a=dict(rank=r,step=1,complete=True,preemptions=int(r==0),
               full_iterations=10,eager_iterations=2,global_prefill_iterations=2,
               local_prefill_iterations=2 if r==0 else 1,worker_kv_blocks=100,kv_blocks=100)
        lines.append('MODE1_STEP_RUNTIME_AUDIT '+json.dumps(a))
    lines.extend(['=> rollout_output_time_s: 123',
                  'step:1 - rollout/rollout_output_time_s:123 - response/aborted_ratio:0.0',
                  '[run] end_time=2026-10-17T00:00:00+0800 exit_code=0'])
    path=tmp_path/'run.txt';path.write_text('\n'.join(lines));return path


def test_native_evidence_does_not_require_legacy_join_strings(tmp_path):
    p=make_log(tmp_path)
    result=module.audit_native_v014(p,16,1)
    assert result['status']=='PASS' and result['kv_preemption_events']==1
    with pytest.raises(module.AuditError,match='local preemption recovery join'):
        module.audit_log(p,16,1)


@pytest.mark.parametrize('old,new,match',[
    ('global_prefill=True graph_mode=NONE','global_prefill=True graph_mode=FULL','prefill incorrectly'),
    ('global_prefill=False graph_mode=FULL','global_prefill=False graph_mode=NONE','non-FULL decode'),
    ('"preemptions": 1','"preemptions": 0','count mismatch'),
    ('"complete": true','"complete": false','inconsistent'),
    ('"eager_iterations": 2','"eager_iterations": 1','inconsistent'),
    ('"local_prefill_iterations": 2','"local_prefill_iterations": 1','recompute'),
    ('exit_code=0','exit_code=1','completion'),
    ('response/aborted_ratio:0.0','response/aborted_ratio:1.0','abort'),
])
def test_native_audit_rejects_actual_inconsistencies(tmp_path,old,new,match):
    p=make_log(tmp_path);p.write_text(p.read_text().replace(old,new,1))
    with pytest.raises(module.AuditError,match=match):module.audit_native_v014(p,16,1)


def test_reindexed_survivor_dp_rank_uses_worker_global_rank(tmp_path):
    p=make_log(tmp_path)
    text=p.read_text()
    text=text.replace('Mode1 KV request preempted: rank=0 step=1',
                      '(WorkerDict pid=111) Mode1 KV request preempted: rank=11 step=1')
    text=text.replace('rollout_forward_begin dp_rank=0',
                      '(WorkerDict pid=111) rollout_forward_begin dp_rank=0')
    lines=[]
    for line in text.splitlines():
        if line.startswith('MODE1_STEP_RUNTIME_AUDIT '):
            a=json.loads(line.split(' ',1)[1]);a['preemptions']=int(a['rank']==11)
            a['local_prefill_iterations']=2 if a['rank']==11 else 1
            line=f'(WorkerDict pid={100+int(a["rank"])}) MODE1_STEP_RUNTIME_AUDIT '+json.dumps(a)
        lines.append(line)
    p.write_text('\n'.join(lines))
    result=module.audit_native_v014(p,16,1)
    assert result['preempted_rank_steps']==[[1,11]]
    # A FULL replay on a different physical survivor cannot prove rank11
    # recovered, even if both use a small local DP rank.
    broken=p.read_text().replace('(WorkerDict pid=111) rollout_forward_begin dp_rank=0 iteration=2',
                                 '(WorkerDict pid=110) rollout_forward_begin dp_rank=0 iteration=2')
    p.write_text(broken)
    with pytest.raises(module.AuditError,match='lacks sampled FULL'):
        module.audit_native_v014(p,16,1)


def test_late_recompute_completion_does_not_fabricate_post_event_full(tmp_path):
    p=make_log(tmp_path)
    text=p.read_text()
    event='Mode1 KV request preempted: rank=0 step=1 request=R'
    text=text.replace(event+'\n','')
    text=text.replace('MODE1_STEP_RUNTIME_AUDIT',event+'\nMODE1_STEP_RUNTIME_AUDIT',1)
    p.write_text(text)
    result=module.audit_native_v014(p,16,1)
    assert result['status']=='PASS'
    assert result['recovery_evidence']==[dict(step=1,rank=0,
        post_preemption_full_sampled=False,completed_recompute=True,
        extra_eager_decode_iterations=0)]
    # End-of-step FULL totals alone are insufficient without complete,
    # post-event recompute and exhaustive no-eager-decode evidence.
    for old,new in [('"complete": true','"complete": false'),
                    ('"eager_iterations": 2','"eager_iterations": 3'),
                    ('"local_prefill_iterations": 2','"local_prefill_iterations": 1')]:
        p.write_text(text.replace(old,new,1))
        with pytest.raises(module.AuditError):module.audit_native_v014(p,16,1)
    p.write_text(text.replace(event+'\n','')+'\n'+event)
    with pytest.raises(module.AuditError,match='precedes last preemption'):
        module.audit_native_v014(p,16,1)


def test_delayed_taskrunner_boundary_cannot_relabel_worker_progress(tmp_path):
    p=make_log(tmp_path)
    text=p.read_text().replace('Mode1 KV request preempted:', '(WorkerDict pid=100) Mode1 KV request preempted:')
    text=text.replace('rollout_forward_begin dp_rank=0', '(WorkerDict pid=100) rollout_forward_begin dp_rank=0')
    text=text.replace('MODE1_STEP_RUNTIME_AUDIT {"rank": 0,', '(WorkerDict pid=100) MODE1_STEP_RUNTIME_AUDIT {"rank": 0,')
    # TaskRunner forwarding may reach the driver before another worker's
    # earlier FULL event. Its source-local step is still1, not2.
    boundary='=> rollout_output_time_s: 123'
    text=text.replace(boundary+'\n','')
    text=text.replace('(WorkerDict pid=100) rollout_forward_begin dp_rank=0 iteration=2',
                      boundary+'\n(WorkerDict pid=100) rollout_forward_begin dp_rank=0 iteration=2')
    p.write_text(text)
    assert module.audit_native_v014(p,16,1)['status']=='PASS'
