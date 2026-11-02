import ast
import importlib.util
import json
import os
import time
from pathlib import Path

ROOT=Path(__file__).parents[1]
spec=importlib.util.spec_from_file_location('sidecar_training_test',ROOT/'tools/run_v014_sidecar_training.py')
runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)


def test_training_matches_main_not_frozen_pair(tmp_path,monkeypatch):
    monkeypatch.setenv('FAIR_FREEZE_ACTOR','1')
    monkeypatch.setenv('DYNAMIC_DISABLE_TAIL_GUARD','1')
    monkeypatch.setenv('VERL_PAIRED_REQUEST_SAMPLING_SEEDS','1')
    monkeypatch.setenv('VERL_SIDECAR_NPU_DEVICES','8,9,10,11,12,13,14,15')
    env,command=runner.build_launch(tmp_path)
    assert env['FAIR_FREEZE_ACTOR']=='0' and env['FAIR_TOTAL_EPOCHS']=='3'
    assert env['DYNAMIC_DISABLE_TAIL_GUARD']=='0'
    assert env['VERL_PAIRED_REQUEST_SAMPLING_SEEDS']=='0'
    assert 'VERL_SIDECAR_NPU_DEVICES' not in env
    assert env['VERL_SIDECAR_START_ONCE']=='0'
    assert env['VLLM_ROLLOUT_SLEEP_LEVEL']=='2' and env['TASK_QUEUE_ENABLE']=='1'
    assert env['VLLM_ASCEND_MODE1_PARITY_MAX_KV_TOKENS_FLOOR2']=='131072'
    assert env['VLLM_ASCEND_MODE1_PARITY_MAX_KV_TOKENS_FLOOR8']=='288256'
    assert env['ADAFLOOR_GRAPH_BASE_RUNNER'].endswith('adaptive_floor2_natural_tailguard_reuse_epoch0_2epoch.sh')
    assert 'adafloor_n_f2' in command and 'actor_rollout_ref.rollout.seed=0' in command
    assert 'actor_rollout_ref.actor.optim.lr=1e-6' in command
    assert 'actor_rollout_ref.rollout.shrink_aware.target_policy=natural' in command


def test_jsonl_keeps_unicode_line_separator_inside_record(tmp_path):
    path=tmp_path/'data.jsonl';path.write_text(json.dumps({'text':'a\u2028b'},ensure_ascii=False)+'\n')
    assert list(runner.jsonl(path))==[{'text':'a\u2028b'}]


def test_graph_sidecar_changes_only_secondary_execution(tmp_path):
    eager, eager_command=runner.build_launch(tmp_path,'eager')
    graph, graph_command=runner.build_launch(tmp_path,'graph')
    assert eager_command==graph_command
    changed={k for k in eager if eager[k]!=graph[k]}
    assert changed=={'VERL_SIDECAR_ENFORCE_EAGER','VERL_SIDECAR_TASK_QUEUE_ENABLE',
                     'VERL_SIDECAR_GRAPH_AUDIT'}
    assert graph['VERL_SIDECAR_ENFORCE_EAGER']=='0'
    assert graph['VERL_SIDECAR_TASK_QUEUE_ENABLE']=='1'
    assert graph['VERL_SIDECAR_CUDAGRAPH_MODE']=='FULL_DECODE_ONLY'
    assert json.loads(graph['VERL_SIDECAR_CUDAGRAPH_CAPTURE_SIZES'])==[1,2,4,8,16,32,64,128]
    assert float(graph['VERL_SIDECAR_WINDOW_GATE_MIN_SECONDS'])==60
    assert graph['VERL_SIDECAR_WINDOW_GATE_STRICT_MINIMUM']=='1'
    assert graph['FAIR_START_EPOCH']=='1' and graph['FAIR_TOTAL_EPOCHS']=='3'


def test_gate60_changes_no_primary_or_sidecar_engine_settings(tmp_path):
    old,old_command=runner.build_launch(tmp_path,'graph',80.0,'ge')
    new,new_command=runner.build_launch(tmp_path,'graph',60.0,'gt')
    assert old_command==new_command
    assert {k for k in old if old[k]!=new[k]}=={
        'VERL_SIDECAR_WINDOW_GATE_MIN_SECONDS','VERL_SIDECAR_WINDOW_GATE_STRICT_MINIMUM'}


def test_sidecar_graph_configuration_is_native_and_capacity_checked(monkeypatch):
    source=(ROOT/'internal/run_elastic_sidecar_infer.sh').read_text()
    embedded=source.split('cat > "${PY_SCRIPT}" <<\'PY\'\n',1)[1].split('\nPY\n',1)[0]
    tree=ast.parse(embedded)
    node=next(n for n in tree.body if isinstance(n,ast.If)
              and ast.unparse(n.test)=="not engine_kwargs['enforce_eager']")
    namespace=dict(os=os,json=json,engine_kwargs={'enforce_eager':False,'max_num_seqs':128})
    compiled=compile(ast.Module(body=[node],type_ignores=[]),'<Sidecar graph config>','exec')
    exec(compiled,namespace)
    assert namespace['engine_kwargs']['compilation_config']['mode']==0
    assert namespace['engine_kwargs']['compilation_config']['cudagraph_mode']=='FULL_DECODE_ONLY'
    assert not any(namespace['engine_kwargs']['additional_config']['ascend_compilation_config'].values())
    monkeypatch.setenv('VERL_SIDECAR_CUDAGRAPH_CAPTURE_SIZES','[1,32]')
    import pytest
    with pytest.raises(ValueError,match='covering max_num_seqs'):
        exec(compiled,namespace)


def test_split_checkpoint_storage_rejects_non_owned_paths(tmp_path):
    import pytest
    with pytest.raises(RuntimeError,match='new run-owned'):
        runner.prepare_split_checkpoint_storage(tmp_path,tmp_path/'public_checkpoints')


def test_native_full_graph_initializes_attention_without_fx_compilation():
    from types import SimpleNamespace
    from enum import Enum
    class Mode(Enum):
        NONE=0
        FULL=1
        FULL_DECODE_ONLY=2
        PIECEWISE=3
    class Compilation(Enum):
        NONE=0
        VLLM_COMPILE=3
    source=ast.parse((ROOT/'vllm_ascend/worker/model_runner_v1.py').read_text())
    cls=next(x for x in source.body if isinstance(x,ast.ClassDef) and x.name=='NPUModelRunner')
    fn=next(x for x in cls.body if isinstance(x,ast.FunctionDef) and x.name=='_use_aclgraph')
    ns=dict(CUDAGraphMode=Mode,CompilationMode=Compilation)
    exec(compile(ast.Module(body=[fn],type_ignores=[]),'<native FULL predicate>','exec'),ns)
    for compile_mode in Compilation:
        for graph_mode in Mode:
            for eager in (False,True):
                fake=SimpleNamespace(compilation_config=SimpleNamespace(mode=compile_mode,cudagraph_mode=graph_mode),
                                     model_config=SimpleNamespace(enforce_eager=eager))
                expected=(not eager and graph_mode!=Mode.NONE and
                          (compile_mode==Compilation.VLLM_COMPILE or graph_mode in (Mode.FULL,Mode.FULL_DECODE_ONLY)))
                assert ns['_use_aclgraph'](fake)==expected


def test_sidecar_rollover_archives_completed_timestamps(tmp_path):
    source=(ROOT/'internal/run_elastic_sidecar_infer.sh').read_text()
    embedded=source.split('cat > "${PY_SCRIPT}" <<\'PY\'\n',1)[1].split('\nPY\n',1)[0]
    tree=ast.parse(embedded)
    fn=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='_reset_shard_state_for_next_epoch')
    namespace=dict(Path=Path,os=os,json=json,shard_index=0,num_shards=8,
                   _shard_prompt_ids=lambda records:{0},_write_epoch=lambda path,value:path.write_text(str(value)))
    exec(compile(ast.Module(body=[fn],type_ignores=[]),'<sidecar rollover>','exec'),namespace)
    paths=[tmp_path/name for name in ('completed.shard0.jsonl','inflight.shard0.json','resume.shard0.jsonl','partials.shard0.jsonl','epoch.shard0.json')]
    payload=json.dumps(dict(sidecar_epoch=0,prompt_id=0,time=123.45))+'\n'
    paths[0].write_text(payload)
    result=namespace['_reset_shard_state_for_next_epoch'](*paths,0,[{}],{0})
    assert result==1 and not paths[0].exists()
    assert (tmp_path/'completed_epochs/completed.shard0.epoch000000.jsonl').read_text()==payload


def test_close_grace_is_full_duration_not_integral_seconds_tick(tmp_path,monkeypatch):
    fixture_spec=importlib.util.spec_from_file_location('grace_fixture',ROOT/'tests/test_sidecar_watcher.py')
    fixture=importlib.util.module_from_spec(fixture_spec);fixture_spec.loader.exec_module(fixture)
    monkeypatch.setenv('SIDECAR_STUB_DELAY_STOP_SECONDS','5')
    begin=time.monotonic()
    process,train,lease,record=fixture._start_watcher(tmp_path,start_once=True)
    try:
        fixture._append_line(train,'Elastic parallel shrink done: rank=0 active_ranks=[8,9,10,11,12,13,14,15] total_ms=1')
        fixture._wait_for(lambda:'devices=' in fixture._text(record))
        # Request close near the next Bash SECONDS tick, reproducing the
        # previously observed ~50ms grace instead of the configured1s.
        time.sleep(max(0,begin+0.9-time.monotonic()))
        ack=fixture._request_release(tmp_path)
        assert ack['status']=='released'
        delta=fixture._event_time(lease,'sidecar_kill_time')-fixture._event_time(lease,'sidecar_stop_request_time')
        assert 0.95<=delta<3.0
    finally:fixture._stop_watcher(process)
