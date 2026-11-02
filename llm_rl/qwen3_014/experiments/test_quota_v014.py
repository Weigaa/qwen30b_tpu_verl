import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace as NS
import pytest

HERE=Path(__file__).parent


def module(name,file):
    spec=importlib.util.spec_from_file_location(name,HERE/file)
    result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result);return result


def test_completion_observer_ignores_initial_empty_and_adds_no_collectives():
    m=module('quota_rank_test','run_quota_v014_rank.py');calls=[]
    runner=NS(_mode1_forward_audit={'full_iterations':7})
    engine=NS(has_unfinished_requests_dp=lambda flag:calls.append(flag) or flag,
              engine_core=NS(collective_rpc=lambda method,*a,**k:42))
    state=m.install_observers(engine,0,runner)
    engine.has_unfinished_requests_dp(False);assert state['local_done'] is None
    engine.has_unfinished_requests_dp(True);engine.has_unfinished_requests_dp(False)
    stamp=state['local_done'];engine.has_unfinished_requests_dp(False)
    assert state['local_done']==stamp and calls==[False,True,False,False]
    assert engine.engine_core.collective_rpc('rebuild_elastic_ep_group',args=(list(range(8,16)),))==42
    assert len(state['events'])==1 and state['events'][0]['full_iterations_after']==7


def test_world_group_initialization_has_config_context_not_only_model_groups():
    from contextlib import contextmanager
    m=module('quota_config_scope','run_quota_v014_rank.py');calls=[];active=[False]
    config=object()
    @contextmanager
    def scope(value):
        assert value is config;active[0]=True
        try:yield
        finally:active[0]=False
    def initialize(size):
        assert active[0] and size==1;calls.append('world_and_model_groups')
    m.initialize_parallel_with_config(initialize,scope,config)
    assert calls==['world_and_model_groups'] and not active[0]


def test_primary_valid_rows_do_not_shrink_physical_residency():
    m=module('quota_real_capacity','run_quota_v014_rank.py')
    layer=NS(elastic_original_num_experts=128,elastic_original_ep_size=16,
        loaded_local_num_experts=8,active_local_num_experts=8,loaded_weight_capacity=32,
        w13_weight=NS(shape=(32,100,100)),w2_weight=NS(shape=(32,100,100)))
    assert m.validate_initial_residency([layer]*48)==([8]*48,[32]*48)
    layer.w13_weight.shape=(8,100,100)
    with pytest.raises(RuntimeError,match='physical expert capacity'):m.validate_initial_residency([layer]*48)


def test_native_v014_graph_and_planned_group_flags(tmp_path):
    m=module('quota_queue_env_test','run_quota_pair_when_ready.py')
    env,_=m.build_env(tmp_path)
    assert env['TASK_QUEUE_ENABLE']=='1'
    assert env['VLLM_ASCEND_ELASTIC_ACLGRAPH']=='1'
    assert env['VLLM_ENABLE_GRAPH_MODE']=='0'
    assert env['VLLM_ASCEND_MODE1_PRECREATE_PLANNED_FLOOR_GROUPS']=='0'
    assert env['VLLM_ASCEND_SHRINK_AWARE_TARGET_POLICY']=='planned'
    assert env['VERL_SIDECAR_ENABLE']=='0'
    assert env['VLLM_ASCEND_NATIVE_SLEEP_PARK_IDLE']=='1'
    assert env['VLLM_ROLLOUT_NATIVE_SLEEP_MODE']=='1'
    assert env['VLLM_ROLLOUT_SLEEP_LEVEL']=='2'


def test_initial_KV_uses_public_configs_without_awake_resize():
    m=module('quota_public_kv','run_quota_v014_rank.py')
    cfg=NS(num_blocks=1040,kv_cache_groups=[NS(kv_cache_spec=NS(block_size=128))])
    assert m.uniform_kv_capacity([cfg])==133120
    with pytest.raises(RuntimeError,match='capacities differ'):
        other=NS(num_blocks=1024,kv_cache_groups=cfg.kv_cache_groups)
        m.uniform_kv_capacity([cfg,other])
    text=(HERE/'run_quota_v014_rank.py').read_text()
    assert 'core.resize_kv_cache_for_mode1_step(' not in text
    assert 'core._kv_cache_token_capacity(' not in text


def test_standalone_weight_layout_matches_VERL_before_import(monkeypatch):
    m=module('quota_nz_layout','run_quota_v014_rank.py')
    monkeypatch.delenv('VLLM_ASCEND_ENABLE_NZ',raising=False)
    m.configure_layout();assert m.os.environ['VLLM_ASCEND_ENABLE_NZ']=='0'
    monkeypatch.setenv('VLLM_ASCEND_ENABLE_NZ','1')
    with pytest.raises(RuntimeError,match='NZ disabled'):m.configure_layout()


def fixture(tmp_path):
    data=dict(comparison=dict(feasible_graph_sha256='same',policies={'min_skew':dict(
        rank_to_prompt_ids={str(i):[2*i,2*i+1] for i in range(16)})}),requests=[])
    for i in range(32):
        for j in range(16):data['requests'].append(dict(id=i,sample_index=j,max_tokens=100,
                                                      request_seed=i*16+j,prompt_hash=str(i)))
    arm=tmp_path/'arm';(arm/'ranks').mkdir(parents=True)
    (arm/'DRIVER_RESULT.json').write_text('{"exit_code":0}')
    (arm/'primary.log').write_text('actual FULL test fixture')
    for i in range(16):
        events=[dict(reporter=i,active_ranks=list(range(8,16)),start=1009,done=1010,full_iterations_after=60)]
        if i>=8:events.append(dict(reporter=i,active_ranks=list(range(12,16)),start=1013,done=1014,full_iterations_after=80))
        end=1011 if i<8 else 1015 if i<12 else 1020
        requests=[dict(id=x['id'],sample_index=x['sample_index'],generated_tokens=100,
            request_seed=x['request_seed'],prompt_hash=x['prompt_hash'],finish_reason='length')
            for x in data['requests'] if x['id'] in (2*i,2*i+1)]
        row=dict(status='V014_FULL_CONTROLLED_RANK_COMPLETE',policy='min_skew',rank=i,
            start=1000,local_done=1001+i,generation_end=end,restore_start=1021,restore_end=1022,
            sleep_start=end+.1,sleep_done=end+.2,native_sleep_level=2,window_end=1022,effective_weight_layout_NZ=0,
            free_bytes_before_sleep=100,free_bytes_after_sleep=1000,
            initial_loaded_experts_per_layer=[8]*48,initial_physical_expert_slots=[32]*48,
            initial_weight_value_audit=dict(checked=24,failed=0),initial_primary_expert_ids=list(range(8*i,8*i+8)),
            workload_sha256='work',feasible_graph_sha256='same',kv_tokens=133120,requests=requests,
            forward_audit=dict(full_iterations=60 if i<8 else 80 if i<12 else 100),
            events=events,startup_seconds=30,weights_audit_sha256='weights',versions=dict(vllm='0.14.0'))
        (arm/'ranks'/f'rank_{i:02d}.json').write_text(json.dumps(row))
    return arm,data


def test_real_endpoint_wait_release_identity(tmp_path):
    m=module('quota_audit_test','audit_quota_v014.py');arm,data=fixture(tmp_path)
    a=m.audit_arm(arm,data,'min_skew','work')
    assert a['completion_slack_rank_seconds']==a['ep_detached_rank_seconds']+a['waiting_and_transition_rank_seconds']
    assert a['generated_tokens']==51200 and a['actual_deepest_floor_FULL_observed']
    assert a['Q_response_tokens_s']==51200/22
    assert a['occupied_rank_seconds']<16*22
    assert a['E_response_tokens_per_occupied_rank_s']>a['Q_response_tokens_s']/16


def test_detach_without_physical_release_does_not_reduce_H(tmp_path):
    m=module('quota_occupancy_conservative','audit_quota_v014.py');arm,data=fixture(tmp_path)
    for i in range(16):
        p=arm/'ranks'/f'rank_{i:02d}.json';row=json.loads(p.read_text())
        row['free_bytes_after_sleep']=100;p.write_text(json.dumps(row))
    a=m.audit_arm(arm,data,'min_skew','work')
    assert a['occupied_rank_seconds']==16*22
    assert a['E_response_tokens_per_occupied_rank_s']==a['Q_response_tokens_s']/16


def test_donor_full_counts_cannot_fake_deepest_floor_replay(tmp_path):
    m=module('quota_audit_reject','audit_quota_v014.py');arm,data=fixture(tmp_path)
    for i in range(12,16):
        p=arm/'ranks'/f'rank_{i:02d}.json';row=json.loads(p.read_text())
        row['forward_audit']['full_iterations']=80;p.write_text(json.dumps(row))
    with pytest.raises(RuntimeError,match='after deepest'):
        m.audit_arm(arm,data,'min_skew','work')
