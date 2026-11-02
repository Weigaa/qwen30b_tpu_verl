import json
import pytest

import audit_n2_f8_results as audit


def test_partial_results_never_become_completion_pass(tmp_path,monkeypatch):
    monkeypatch.setattr(audit.campaign,'check_sources',lambda root:None)
    result=audit.audit(tmp_path)
    assert result['status']=='INTERIM_NOT_COMPLETE'
    assert len(result['pending'])==8
    assert result['results']['math_completed_seeds']==[]
    assert json.loads((tmp_path/'NEW_N2_AUDIT_REPORT.json').read_text())['pending']==result['pending']
    with pytest.raises(RuntimeError,match='Campaign not complete'):
        audit.audit(tmp_path,strict=True)


def test_launch_budget_requires_all_explicit_floors_and_native_full_path(tmp_path):
    env={'VLLM_ASCEND_MODE1_PARITY_MAX_KV_TOKENS_FLOOR8':'349952',
         'VLLM_ASCEND_MODE1_PARITY_MAX_KV_TOKENS_FLOOR2':'131072',
         'VLLM_ASCEND_MODE1_PARITY_MAX_KV_TOKENS_FLOOR4':'280576',
         'VLLM_ASCEND_MODE1_PARITY_MAX_KV_TOKENS_FLOOR16':'380800',
         'FLOOR_KV_CAPS':audit.campaign.CAPS,'TASK_QUEUE_ENABLE':'1',
         'VLLM_ROLLOUT_SLEEP_LEVEL':'2','VLLM_ROLLOUT_NATIVE_SLEEP_GRAPH_LIFECYCLE':'1',
         'ADAFLOOR_ACLGRAPH_MODE':'FULL_DECODE_ONLY'}
    path=tmp_path/'LAUNCH.json';path.write_text(json.dumps({'env':env}))
    assert audit.launch_budget(path)['env']==env
    env['VLLM_ASCEND_MODE1_PARITY_MAX_KV_TOKENS_FLOOR8']='288256'
    path.write_text(json.dumps({'env':env}))
    with pytest.raises(RuntimeError,match='Wrong new F8'):audit.launch_budget(path)
    env['VLLM_ASCEND_MODE1_PARITY_MAX_KV_TOKENS_FLOOR8']='349952'
    env.pop('FLOOR_KV_CAPS');path.write_text(json.dumps({'env':env}))
    with pytest.raises(RuntimeError,match='Missing/changed'):audit.launch_budget(path)


def test_efficiency_uses_ratio_of_totals_not_mean_of_epoch_ratios():
    epochs=[{'response_tokens':100,'occupied_rank_s_bounds':[40,50]},
            {'response_tokens':900,'occupied_rank_s_bounds':[90,100]}]
    assert audit.combined_efficiency(epochs)==[1000/150,1000/130]
    assert audit.combined_efficiency(epochs)[0]!=(100/50+900/100)/2
    with pytest.raises(RuntimeError,match='Invalid pooled'):
        audit.combined_efficiency([{'response_tokens':100,'occupied_rank_s_bounds':[0,0]}])


def test_actual_new_capacity_must_match_every_rank_not_just_launch_flags(tmp_path):
    from types import SimpleNamespace
    folder=tmp_path/'oracle';folder.mkdir()
    plans=[dict(step=s,selected_floor=8,kv_cap=349952) for s in range(1,6)]
    path=folder/'length_sorted_rank_plan.json';path.write_text(json.dumps(plans))
    rows=[dict(step=s,rank=r,kv_tokens=349952) for s in range(1,6) for r in range(16)]
    suite=SimpleNamespace(json_records=lambda *args:rows)
    audit.actual_kv_budget(suite,tmp_path)
    rows[0]['kv_tokens']=288256
    with pytest.raises(RuntimeError,match='Runtime/planned'):audit.actual_kv_budget(suite,tmp_path)
    rows[0]['kv_tokens']=349952
    rows[-1]['rank']=14
    with pytest.raises(RuntimeError,match='coverage'):audit.actual_kv_budget(suite,tmp_path)
    rows[-1]['rank']=15
    plans[0]['kv_cap']=288256;path.write_text(json.dumps(plans))
    rows[0]['kv_tokens']=288256
    with pytest.raises(RuntimeError,match='capacity family'):audit.actual_kv_budget(suite,tmp_path)


def test_worker_completed_topology_is_not_optional_rpc_prefix_count():
    text='\n'.join(f'Elastic parallel shrink done: rank={r} active_ranks=[0, 2] dp_size=2 ep_size=2' for r in [0,2])
    assert 'Shrink RPC result:' not in text
    assert audit.worker_shrink_groups(text)==[dict(active_ranks=[0,2],floor=2,worker_completion_ranks=[0,2])]
    assert audit.worker_shrink_groups('configured target floor2')==[]
    with pytest.raises(RuntimeError,match='inconsistent active topology'):
        audit.worker_shrink_groups(text.replace('dp_size=2','dp_size=4'))
