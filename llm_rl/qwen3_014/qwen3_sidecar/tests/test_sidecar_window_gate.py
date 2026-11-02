from datetime import datetime, timezone, timedelta
import importlib.util
import json
from pathlib import Path
import time
import pytest

ROOT=Path(__file__).parents[1]
spec=importlib.util.spec_from_file_location('window_gate',ROOT/'tools/sidecar_window_gate.py')
gate=importlib.util.module_from_spec(spec);spec.loader.exec_module(gate)


def inputs(now,exit_tokens=2000):
    lines=['Mode1 step timeline: driver_generate_start step=1 epoch=0']
    for rank in range(8,16):
        for iteration,age in [(512,100),(1024,75)]:
            stamp=datetime.fromtimestamp(now-age,timezone(timedelta(hours=8))).strftime('%Y-%m-%d %H:%M:%S')
            lines.append(f'INFO {stamp} rollout_forward_begin dp_rank={rank} iteration={iteration} local_tokens=32 global_prefill=False graph_mode=FULL')
    lines.append('Elastic EP rank exit start: global_rank=0 active_ranks=[8,9,10,11,12,13,14,15]')
    plan=[dict(step=1,rank_loads={str(rank):exit_tokens for rank in range(16)})]
    return plan,'\n'.join(lines)+'\n'


def test_short_long_and_future_output_independence():
    now=time.time();plan,text=inputs(now)
    result=gate.forecast(plan,text,list(range(8,16)),now)
    assert result['action']=='skip' and result['reason']=='predicted_window_too_short'
    assert result['decode_rate_samples']==8
    assert gate.forecast(plan,text+'future output final_length=999999\n',list(range(8,16)),now)==result
    long,_=inputs(now,10000)
    assert gate.forecast(long,text,list(range(8,16)),now)['action']=='start'


def test_missing_rate_is_explicitly_unknown_not_zero_length():
    now=time.time();plan,text=inputs(now)
    result=gate.forecast(plan,text.replace('graph_mode=FULL','graph_mode=NONE'),list(range(8,16)),now)
    assert result['action']=='skip' and result['reason']=='prediction_unavailable'
    assert result['predicted_window_seconds'] is None


def test_strict60_boundary_and_restore_skip_validation(tmp_path):
    from verl.utils.sidecar_restore_handshake import _valid_gate_skip
    now=time.time();plan,text=inputs(now,2944)
    stamp=lambda age:datetime.fromtimestamp(now-age,timezone(timedelta(hours=8))).strftime('%Y-%m-%d %H:%M:%S')
    text=text.replace(stamp(75),stamp(84))  # Exactly 1/32 seconds per token.
    active=list(range(8,16))
    at60=gate.forecast(plan,text,active,now,minimum=60,strict_minimum=True)
    assert at60['predicted_window_seconds']==60.0
    assert at60['minimum_comparison']=='>' and at60['action']=='skip'
    assert gate.forecast(plan,text,active,now,minimum=60)['action']=='start'  # Legacy inclusive rule.
    path=tmp_path/'decision.json';path.write_text(json.dumps(at60))
    env=dict(VERL_SIDECAR_WINDOW_GATE_ENABLE='1',VERL_SIDECAR_WINDOW_GATE_FILE=str(path),
             VERL_SIDECAR_WINDOW_GATE_MIN_SECONDS='60',VERL_SIDECAR_WINDOW_GATE_STRICT_MINIMUM='1')
    assert _valid_gate_skip(env,dict(state='skipped_window_gate',lease_id='1'))
    for value,action in ((2943.999,'skip'),(2944.001,'start')):
        for row in plan:row['rank_loads']={str(r):value for r in range(16)}
        assert gate.forecast(plan,text,active,now,minimum=60,strict_minimum=True)['action']==action


def test_epoch_forecast_uses_latest_step_only():
    now=time.time()
    first,first_text=inputs(now,2000)
    second,second_text=inputs(now,10000)
    second[0]['step']=2
    text=first_text+second_text.replace('driver_generate_start step=1','driver_generate_start step=2')
    result=gate.forecast(first+second,text,list(range(8,16)),now)
    assert result['step']==2
    assert result['predicted_exit_tokens']==10000
    assert result['decode_rate_samples']==8
    assert result['action']=='start'


def test_full_epoch_skip_start_and_no_shrink_roundtrips(tmp_path,monkeypatch):
    fixture_spec=importlib.util.spec_from_file_location('epoch_watch_fixture',ROOT/'tests/test_sidecar_watcher.py')
    fixture=importlib.util.module_from_spec(fixture_spec);fixture_spec.loader.exec_module(fixture)
    from verl.utils.sidecar_restore_handshake import request_sidecar_stop_before_restore
    logs=tmp_path/'epoch/logs';logs.mkdir(parents=True)
    plan_dir=tmp_path/'epoch/oracle';plan_dir.mkdir()
    decision=logs/'window_gate.json'
    monkeypatch.setenv('VERL_SIDECAR_WINDOW_GATE_ENABLE','1')
    monkeypatch.setenv('VERL_SIDECAR_WINDOW_GATE_FILE',str(decision))
    monkeypatch.setenv('VERL_SIDECAR_REARM_AFTER_GATE_SKIP','1')
    plans=[];texts=[]
    for step,exit_tokens in [(1,2000),(2,10000),(3,2000)]:
        plan,text=inputs(time.time(),exit_tokens)
        plan[0]['step']=step;plans+=plan
        texts.append(text.replace('driver_generate_start step=1',f'driver_generate_start step={step}'))
    (plan_dir/'length_sorted_rank_plan.json').write_text(json.dumps(plans))
    process,train_log,lease,stub=fixture._start_watcher(logs,start_once=False,require_active=True,require_quorum=True)
    env=dict(VERL_SIDECAR_ENABLE='1',VERL_SIDECAR_REQUIRE_ACTIVE_LEASE_BEFORE_RESTORE='1',
        VERL_SIDECAR_WINDOW_GATE_ENABLE='1',VERL_SIDECAR_WINDOW_GATE_FILE=str(decision),
        VERL_SIDECAR_RESTORE_HANDSHAKE_DIR=str(logs/'restore_handshake'),
        VERL_SIDECAR_STOP_ACK_TIMEOUT_SECONDS='3',VERL_SIDECAR_STOP_ACK_POLL_SECONDS='0.01')
    try:
        for step in (1,2,3):
            if step<3:
                fixture._append_line(train_log,texts[step-1])
                for rank in range(16):
                    fixture._append_line(train_log,f'Elastic parallel shrink rpc done: global_rank={rank} active_ranks=[8,9,10,11,12,13,14,15] total_ms=20.0')
                expected='skipped_window_gate' if step==1 else 'running'
                fixture._wait_for(lambda:fixture._fields(logs/'restore_handshake/active_lease').get('state')==expected)
                if step==2:
                    fixture._wait_for(lambda:'devices=0,1,2,3,4,5,6,7' in fixture._text(stub))
            else:
                fixture._append_line(train_log,'Mode1 step timeline: driver_generate_start step=3 epoch=0')
            fixture._append_line(train_log,f'Mode1 step timeline: driver_generate_done step={step}')
            ack=request_sidecar_stop_before_restore(env)
            assert ack['status']=='released' and ack['lease_id']==str(step)
            fixture._wait_for(lambda:fixture._fields(logs/'restore_handshake/active_lease').get('state')=='armed')
            if step==1:
                # A delayed/repeated generate_done must not create another
                # lease for a step whose gate and restore already completed.
                fixture._append_line(train_log,'Mode1 step timeline: driver_generate_done step=1 epoch=0')
                time.sleep(0.15)
                assert fixture._fields(logs/'restore_handshake/active_lease')['state']=='armed'
            assert process.poll() is None
        decisions=[json.loads((logs/f'window_gate.lease{step}.json').read_text()) for step in (1,2,3)]
        assert [row['action'] for row in decisions]==['skip','start','skip']
        assert [row['step'] for row in decisions]==[1,2,3]
        assert 'sidecar_start_time=' in lease.read_text()
        assert lease.read_text().count('sidecar_start_time=')==1
    finally:fixture._stop_watcher(process)


@pytest.mark.parametrize('calibrate',[False,True])
def test_direct_events_finish_five_steps_without_completion_logs(tmp_path,monkeypatch,calibrate):
    """Reproduce the real no-shrink hang with ALL Ray completion logs absent."""
    fixture_spec=importlib.util.spec_from_file_location('direct_watch_fixture',ROOT/'tests/test_sidecar_watcher.py')
    fixture=importlib.util.module_from_spec(fixture_spec);fixture_spec.loader.exec_module(fixture)
    from verl.utils.sidecar_restore_handshake import (
        publish_sidecar_step_event,request_sidecar_stop_before_restore,
    )
    logs=tmp_path/'epoch/logs';logs.mkdir(parents=True)
    plan_dir=tmp_path/'epoch/oracle';plan_dir.mkdir()
    decision=logs/'window_gate.json'
    monkeypatch.setenv('VERL_SIDECAR_WINDOW_GATE_ENABLE','1')
    monkeypatch.setenv('VERL_SIDECAR_WINDOW_GATE_FILE',str(decision))
    monkeypatch.setenv('VERL_SIDECAR_REARM_AFTER_GATE_SKIP','1')
    monkeypatch.setenv('VERL_SIDECAR_DIRECT_STEP_EVENTS','1')
    if calibrate:
        monkeypatch.setenv('VERL_SIDECAR_WINDOW_GATE_CALIBRATION','prior_step_q95')
    env=dict(VERL_SIDECAR_ENABLE='1',VERL_SIDECAR_DIRECT_STEP_EVENTS='1',
        VERL_SIDECAR_REQUIRE_ACTIVE_LEASE_BEFORE_RESTORE='1',
        VERL_SIDECAR_WINDOW_GATE_ENABLE='1',VERL_SIDECAR_WINDOW_GATE_FILE=str(decision),
        VERL_SIDECAR_RESTORE_HANDSHAKE_DIR=str(logs/'restore_handshake'),
        VERL_SIDECAR_STOP_ACK_TIMEOUT_SECONDS='3',VERL_SIDECAR_STOP_ACK_POLL_SECONDS='0.01')
    plans=[];texts=[]
    predictions=(4085,5724,8390,12360,16384) if calibrate else (2000,10000,2000,2000,2000)
    actual_lengths=(4186,7932,11850,16384,16384) if calibrate else predictions
    for step,exit_tokens in enumerate(predictions,1):
        plan,text=inputs(time.time(),exit_tokens)
        plan[0]['step']=step;plans+=plan
        if calibrate and step in (1,2,4):
            a,b={1:(2816,3328),2:(6656,7168),4:(10752,11264)}[step]
            text=text.replace('iteration=512 ',f'iteration={a} ').replace('iteration=1024 ',f'iteration={b} ')
        texts.append(text.replace('driver_generate_start step=1',f'driver_generate_start step={step}'))
    (plan_dir/'length_sorted_rank_plan.json').write_text(json.dumps(plans))
    process,train_log,lease,stub=fixture._start_watcher(logs,start_once=False,require_active=True,require_quorum=True)
    try:
        # A completed step MUST remain fail-closed if no watcher ACK arrives.
        for step in range(1,6):
            publish_sidecar_step_event('running',step,env)
            if step in (1,2,4):
                fixture._append_line(train_log,texts[step-1])
                for rank in range(16):
                    fixture._append_line(train_log,f'Elastic parallel shrink rpc done: global_rank={rank} active_ranks=[8,9,10,11,12,13,14,15] total_ms=20.0')
                starting_step=4 if calibrate else 2
                expected='running' if step==starting_step else 'skipped_window_gate'
                fixture._wait_for(lambda:fixture._fields(logs/'restore_handshake/active_lease').get('state')==expected)
                if step==starting_step:
                    fixture._wait_for(lambda:'devices=0,1,2,3,4,5,6,7' in fixture._text(stub))
            before=time.monotonic()
            ack=request_sidecar_stop_before_restore(env,step=step)
            assert time.monotonic()-before<3
            assert ack['status']=='released' and ack['lease_id']==str(step)
            fixture._wait_for(lambda:fixture._fields(logs/'restore_handshake/active_lease').get('state')=='armed')
            lengths=tmp_path/'epoch/rollout_length';lengths.mkdir(exist_ok=True)
            (lengths/f'length_{step}.txt').write_text((str(actual_lengths[step-1])+'\n')*512)
            # Delayed log delivery after ACK must not start another process
            # nor manufacture an extra lease, even with a complete old quorum.
            fixture._append_line(train_log,texts[step-1])
            fixture._append_line(train_log,f'Mode1 step timeline: driver_generate_done step={step}')
            time.sleep(0.08)
            assert process.poll() is None
        records=[json.loads((logs/f'window_gate.lease{step}.json').read_text()) for step in range(1,6)]
        assert [r['step'] for r in records]==[1,2,3,4,5]
        expected_actions=['skip','skip','skip','start','skip'] if calibrate else ['skip','start','skip','skip','skip']
        assert [r['action'] for r in records]==expected_actions
        if calibrate:
            assert [row['step'] for row in records[3]['calibration']['sources']]==[1,2,3]
            assert records[3]['calibration']['current_or_future_step_lengths_read'] is False
        for step in (3,5):
            assert records[step-1]['reason']=='generation_completed_without_eligible_quorum'
            assert records[step-1]['process_group_started'] is False
        assert lease.read_text().count('sidecar_start_time=')==1
        assert fixture._text(stub).count('stopped=1')==1
        assert not (logs/'window_gate.lease6.json').exists()
    finally:fixture._stop_watcher(process)


def test_skip_roundtrip_never_launches_process(tmp_path,monkeypatch):
    fixture_spec=importlib.util.spec_from_file_location('watch_fixture',ROOT/'tests/test_sidecar_watcher.py')
    fixture=importlib.util.module_from_spec(fixture_spec);fixture_spec.loader.exec_module(fixture)
    from verl.utils.sidecar_restore_handshake import request_sidecar_stop_before_restore
    logs=tmp_path/'epoch/logs';logs.mkdir(parents=True)
    plan_dir=tmp_path/'epoch/oracle';plan_dir.mkdir()
    decision=logs/'window_gate.json'
    monkeypatch.setenv('VERL_SIDECAR_WINDOW_GATE_ENABLE','1')
    monkeypatch.setenv('VERL_SIDECAR_WINDOW_GATE_FILE',str(decision))
    plan,text=inputs(time.time())
    (plan_dir/'length_sorted_rank_plan.json').write_text(json.dumps(plan))
    process,train_log,lease,stub=fixture._start_watcher(logs,start_once=True,require_active=True,require_quorum=True)
    try:
        fixture._append_line(train_log,text)
        for rank in range(16):
            fixture._append_line(train_log,f'Elastic parallel shrink rpc done: global_rank={rank} active_ranks=[8,9,10,11,12,13,14,15] total_ms=20.0')
        fixture._wait_for(lambda:fixture._fields(logs/'restore_handshake/active_lease').get('state')=='skipped_window_gate')
        generate_done=time.time()
        fixture._append_line(train_log,f'Mode1 step timeline: driver_generate_done driver_generate_done_time={generate_done}')
        ack=request_sidecar_stop_before_restore(dict(
            VERL_SIDECAR_ENABLE='1',VERL_SIDECAR_REQUIRE_ACTIVE_LEASE_BEFORE_RESTORE='1',
            VERL_SIDECAR_WINDOW_GATE_ENABLE='1',VERL_SIDECAR_WINDOW_GATE_FILE=str(decision),
            VERL_SIDECAR_RESTORE_HANDSHAKE_DIR=str(logs/'restore_handshake'),
            VERL_SIDECAR_STOP_ACK_TIMEOUT_SECONDS='2',VERL_SIDECAR_STOP_ACK_POLL_SECONDS='0.01'))
        process.wait(timeout=5)
        assert process.returncode==0
        assert ack['reason']=='sidecar_not_started_window_gate'
        assert not stub.exists()
        assert 'sidecar_start_time=' not in lease.read_text()
        assert json.loads(decision.read_text())['reason']=='predicted_window_too_short'
        verifier_spec=importlib.util.spec_from_file_location('gate_result_verifier',ROOT/'tools/verify_qwen3_sidecar_pairs.py')
        verifier=importlib.util.module_from_spec(verifier_spec);verifier_spec.loader.exec_module(verifier)
        primary=dict(epoch_dir=str(tmp_path/'epoch'),log_file=str(train_log),
                     restore_ack_request_id=ack['request_id'],restore_ack_lease_id=ack['lease_id'],
                     driver_generate_done_time=generate_done,driver_restore_rpc_start_time=time.time())
        result=verifier._skipped_sidecar_artifact(logs,primary)
        assert result['sidecar_used_rank_seconds']==0
        assert result['detached_rank_seconds']>0
        assert result['completed_queries']==0 and result['strict_accuracy'] is None
        assert result['unused_detached_ranks']==list(range(8))
        record=json.loads(decision.read_text());record['predicted_window_seconds']=79
        decision.write_text(json.dumps(record))
        with pytest.raises(verifier.VerificationError,match='recomputed'):
            verifier._skipped_sidecar_artifact(logs,primary)
    finally:fixture._stop_watcher(process)


@pytest.mark.parametrize('mutate',[{'action':'start'},{'lease_id':2},{'predicted_window_seconds':float('nan')},
                                  {'minimum_window_seconds':1},{'current_response_outputs_read':True}])
def test_forged_skip_rejected(tmp_path,mutate):
    from verl.utils.sidecar_restore_handshake import _valid_gate_skip
    record=dict(action='skip',lease_id=1,predicted_window_seconds=20,minimum_window_seconds=80,
                current_response_outputs_read=False,reason='predicted_window_too_short')
    record.update(mutate);path=tmp_path/'gate.json';path.write_text(json.dumps(record))
    env=dict(VERL_SIDECAR_WINDOW_GATE_ENABLE='1',VERL_SIDECAR_WINDOW_GATE_FILE=str(path))
    assert not _valid_gate_skip(env,dict(state='skipped_window_gate',lease_id='1'))
