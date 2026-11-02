import ast
import importlib.util
import json
from pathlib import Path

import pytest

ROOT=Path(__file__).parents[1]
spec=importlib.util.spec_from_file_location('mlsys_fig6',ROOT/'tools/run_mlsys_fig6.py')
suite=importlib.util.module_from_spec(spec)
spec.loader.exec_module(suite)


def test_exact_original_fig6_schedule():
    assert suite.SEEDS==(0,2,3)
    assert len(suite.POLICIES)==8 and 'fixed4' in suite.POLICIES
    assert 'planned_f2' not in suite.POLICIES
    assert len([(s,p,e,t) for s in suite.SEEDS for p in suite.POLICIES
                for e in (1,2) for t in range(5)])==240


def test_calibrated_capacity_profile_matches_runtime_and_retained_evidence(tmp_path):
    evidence=json.loads(suite.CAPACITY_EVIDENCE.read_text())
    assert evidence['decision']==suite.KV_BUDGETS
    assert all(x%128==0 for x in suite.KV_BUDGETS.values())
    assert suite.KV_BUDGETS['natural_f4']==280576>16*16384
    assert suite.KV_BUDGETS['full16']==380800
    assert suite.KV_BUDGETS['planned_f4']==suite.KV_BUDGETS['natural_f4']-147456
    assert suite.KV_BUDGETS['planned_f8']==suite.KV_BUDGETS['natural_f4_family_f8']-114688
    for policy in suite.POLICIES:
        env=suite.arm_env(tmp_path,0,policy)
        assert int(env['VANILLA_KV_TOKENS_PER_RANK'])==suite.KV_BUDGETS['full16']
        assert int(env['VLLM_ASCEND_MODE1_PARITY_MAX_KV_TOKENS_FLOOR4'])==suite.KV_BUDGETS['natural_f4']
        assert int(env['VLLM_ASCEND_MODE1_PARITY_MAX_KV_TOKENS_FLOOR2'])==suite.KV_BUDGETS['natural_f2']


def test_one_invocation_selects_only_one_seed():
    assert suite.select_seed()==0
    assert suite.select_seed('0:minskew')==0
    assert suite.select_seed('2:vanilla')==2
    assert suite.select_seed(seed=3)==3
    with pytest.raises(ValueError):suite.select_seed('2:vanilla',0)


def test_partial_resume_requires_complete_original_checkpoint(tmp_path,monkeypatch):
    from types import SimpleNamespace
    first=tmp_path/'seed_0/planned_f4/fig6/epoch_001_mode1_planned'
    cp=first/'checkpoints/qwen3moe_for_eagle3/global_step_5/actor/dist_ckpt'
    cp.mkdir(parents=True)
    (cp/'.metadata').write_bytes(b'metadata')
    for i in range(32):(cp/f'{i}.distcp').write_bytes(b'weight')
    (cp.parents[2]/'latest_checkpointed_iteration.txt').write_text('5')
    calls=[]
    monkeypatch.setattr(suite,'parse_epoch',lambda path,require_tailguard: calls.append((path,require_tailguard)) or {'epoch1':'validated'})
    monkeypatch.setattr(suite.shutil,'disk_usage',lambda path:SimpleNamespace(free=20*1024**3))
    env=suite.prepare_planned_epoch1_resume(tmp_path,0,'planned_f4')
    assert env['FAIR_RESUME_EXISTING']=='1' and env['FAIR_START_EPOCH']=='1'
    assert calls==[(first,True)]
    assert (tmp_path/'seed_0/planned_f4/VALIDATED_EPOCH1_RESUME.json').is_file()
    monkeypatch.setattr(suite.shutil,'disk_usage',lambda path:SimpleNamespace(free=1))
    with pytest.raises(RuntimeError,match='headroom'):suite.prepare_planned_epoch1_resume(tmp_path,0,'planned_f4')
    monkeypatch.setattr(suite.shutil,'disk_usage',lambda path:SimpleNamespace(free=20*1024**3))
    second=first.parent/'epoch_002_mode1_planned';second.mkdir()
    with pytest.raises(RuntimeError,match='no epoch2'):suite.prepare_planned_epoch1_resume(tmp_path,0,'planned_f4')
    second.rmdir()
    (cp/'0.distcp').unlink()
    with pytest.raises(RuntimeError,match='checkpoint is incomplete'):suite.prepare_planned_epoch1_resume(tmp_path,0,'planned_f4')
    with pytest.raises(RuntimeError,match='limited'):suite.prepare_planned_epoch1_resume(tmp_path,0,'vanilla')


def test_later_seeds_require_review_not_just_finished_files(tmp_path,monkeypatch):
    suite.require_previous_seed_review(tmp_path,0)
    with pytest.raises(RuntimeError,match='must finish and be reviewed'):
        suite.require_previous_seed_review(tmp_path,2)
    suite.write_json(tmp_path/'protocol.json',{'test':'protocol'})
    report=tmp_path/'analysis_eval/seed0_review.json'
    suite.write_json(report,{'runtime_and_quality_review':'retained evidence fixture'})
    review=dict(approved=True,reviewed_seed=0,next_seed=2,
                protocol_sha256=suite.digest(tmp_path/'protocol.json'),
                evidence_path=str(report.relative_to(tmp_path)),evidence_sha256=suite.digest(report))
    suite.write_json(tmp_path/'seed_0/SEED_REVIEW_PASS.json',review)
    with pytest.raises(RuntimeError,match='not fully validated'):
        suite.require_previous_seed_review(tmp_path,2)
    calls=[]
    monkeypatch.setattr(suite,'parse_arm',lambda r,s,p:calls.append((s,p)) or {'seed':s,'policy':p})
    for policy in suite.POLICIES:
        suite.write_json(tmp_path/f'seed_0/{policy}/ARM_PASS.json',{'seed':0,'policy':policy})
    suite.require_previous_seed_review(tmp_path,2)
    assert calls==[(0,p) for p in suite.POLICIES]
    suite.write_json(report,{'changed':'evidence'})
    with pytest.raises(RuntimeError,match='retained evidence'):
        suite.require_previous_seed_review(tmp_path,2)
    review['protocol_sha256']='stale';suite.write_json(tmp_path/'seed_0/SEED_REVIEW_PASS.json',review)
    with pytest.raises(RuntimeError,match='stale'):
        suite.require_previous_seed_review(tmp_path,2)
    suite.write_json(report,{'runtime_and_quality_review':'retained evidence fixture'})
    original=tmp_path/'seed_0/SEED_REVIEW_PASS.json'
    reaffirm=dict(approved=True,original_review_sha256=suite.digest(original),
                 protocol_sha256=suite.digest(tmp_path/'protocol.json'),runtime_protocol_unchanged=True)
    approval=original.with_name('SEED_REVIEW_PROTOCOL_REAFFIRMATION.json')
    suite.write_json(approval,reaffirm)
    suite.require_previous_seed_review(tmp_path,2)
    for field,value in [('approved',False),('original_review_sha256','wrong'),
                        ('protocol_sha256','wrong'),('runtime_protocol_unchanged',False)]:
        suite.write_json(approval,dict(reaffirm,**{field:value}))
        with pytest.raises(RuntimeError,match='stale'):suite.require_previous_seed_review(tmp_path,2)


@pytest.mark.parametrize('policy',suite.POLICIES)
def test_all_arms_are_updated_policy_global_seed_full_graph(tmp_path,policy,monkeypatch):
    monkeypatch.setenv('ADAFLOOR_FROZEN_INPUTS_DIR','/old/frozen')
    monkeypatch.setenv('VERL_PAIRED_REQUEST_SAMPLING_SEEDS','1')
    monkeypatch.setenv('FLOOR_KV_CAPS','4:999')
    monkeypatch.setenv('VLLM_ASCEND_MODE1_PARITY_PRECREATE_PLANNED_FLOOR_GROUPS','1')
    env=suite.arm_env(tmp_path,2,policy)
    assert 'ADAFLOOR_FROZEN_INPUTS_DIR' not in env and 'FLOOR_KV_CAPS' not in env
    assert 'VLLM_ASCEND_MODE1_PARITY_PRECREATE_PLANNED_FLOOR_GROUPS' not in env
    assert env['VERL_PAIRED_REQUEST_SAMPLING_SEEDS']=='0'
    assert env['FAIR_TOTAL_EPOCHS']=='3' and env['FAIR_FREEZE_ACTOR']=='0'
    assert env['TASK_QUEUE_ENABLE']=='1' and env['ROLLOUT_ENFORCE_EAGER']=='False'
    assert env['VLLM_ROLLOUT_SLEEP_LEVEL']=='2'
    assert env['VLLM_ROLLOUT_NATIVE_SLEEP_GRAPH_LIFECYCLE']=='1'
    assert env['VLLM_ROLLOUT_NATIVE_SLEEP_GRAPH_KV_FIRST']=='1'
    assert env['ACTIVE_PEAK_SAFETY_FACTOR']=='1.16'
    assert env['VLLM_ASCEND_ACLGRAPH_BUCKET_COUNT']=='8'
    socket=env['RAY_TMPDIR']+'/ray/session_2026-10-17_14-25-58_748099_1041431/sockets/plasma_store'
    assert len(socket.encode()) <= 107
    cmd=suite.arm_command(2,policy)
    assert 'actor_rollout_ref.rollout.seed=2' in cmd
    assert 'actor_rollout_ref.actor.optim.lr=1e-6' in cmd
    assert 'trainer.save_freq=5' in cmd
    assert 'actor_rollout_ref.rollout.cudagraph_mode=FULL_DECODE_ONLY' in cmd
    if policy in suite.BASES:
        assert Path(env['ADAFLOOR_GRAPH_BASE_RUNNER']).is_file()
        assert '++actor_rollout_ref.rollout.engine_kwargs.vllm.num_gpu_blocks_override=810' in cmd
    else:
        assert not any('num_gpu_blocks_override' in a for a in cmd)


def test_vanilla_and_lengthsort_use_root_config_and_same_k8():
    vanilla=(ROOT/'run_baseline_vanilla_epoch1_2_from_common_epoch0.sh').read_text()
    ls=(ROOT/'run_baseline_lengthsort_epoch1_2.sh').read_text()
    assert '++actor_rollout_ref.rollout.aclgraph_bucket_plan_path=' in vanilla
    assert '++actor_rollout_ref.rollout.aclgraph_bucket_count=' in vanilla
    assert 'shrink_aware.aclgraph_bucket_count=' not in vanilla
    assert '--aclgraph-bucket-count "$VLLM_ASCEND_ACLGRAPH_BUCKET_COUNT"' in ls
    assert '++actor_rollout_ref.rollout.aclgraph_bucket_plan_path=$rank_plan' in ls
    assert '"${graph_bucket_args[@]}"' in ls


def test_fixed4_preserves_min_skew_forced_floor_and_graph_budget():
    text=(ROOT/'run_baseline_kvsafe_fixed4_epoch1_2.sh').read_text()
    assert 'RANK_MATCHING_POLICY=min_skew' in text
    assert 'DYNAMIC_FORCE_SELECTED_FLOOR=4' in text
    assert 'DYNAMIC_DISABLE_TAIL_GUARD=0' in text
    assert 'PARITY_MAX_KV_TOKENS_FLOOR4=280576' not in text
    assert 'run_mode1_dynamic_length_aware_adaptive_floor2_natural' in text


def test_epoch_parser_rejects_incomplete_results(tmp_path):
    (tmp_path/'logs').mkdir()
    (tmp_path/'logs/run.txt').write_text('=> rollout_output_time_s: 1\n')
    with pytest.raises(RuntimeError,match='successful child exit'):
        suite.parse_epoch(tmp_path)


def test_statistics_average_epochs_then_sample_sd(tmp_path):
    for seed,value in [(0,10),(2,20),(3,30)]:
        p=tmp_path/f'seed_{seed}/vanilla/ARM_PASS.json'
        row=dict(seed=seed,policy='vanilla',prompt_hash='same')
        row.update({k:value for k in ('rollout_s','response_tokens_s','reward','mean_length',
                                     'release8','release4','release2','released_rank_s')})
        suite.write_json(p,row)
    result=suite.summarize(tmp_path)
    assert result['aggregate'][0]['rollout_s']==20
    assert result['aggregate'][0]['rollout_s_sample_sd']==10
    assert result['complete'] is False and not (tmp_path/'FIG6_PASS.json').exists()


def test_new_root_hydra_bucket_fields_compose():
    from hydra import compose,initialize_config_dir
    with initialize_config_dir(config_dir=str(ROOT/'verl/trainer/config'),version_base=None):
        cfg=compose(config_name='ppo_megatron_trainer',overrides=[
            '++actor_rollout_ref.rollout.aclgraph_bucket_plan_path=/plan.json',
            '++actor_rollout_ref.rollout.aclgraph_bucket_count=8',
            *suite.arm_command(0,'vanilla')[3:]])
    assert cfg.actor_rollout_ref.rollout.aclgraph_bucket_count==8
    assert cfg.actor_rollout_ref.rollout.seed==0
    assert cfg.actor_rollout_ref.actor.optim.lr==1e-6


def test_keep_original_aggregation_and_exclude_only_duplicate_metric_ends():
    text=(ROOT/'tools/run_mlsys_fig6.py').read_text()
    assert 'qwen3_shrink_aware/analysis_eval/plot_e2e_performance.py' in text
    assert "'rollout_output_time_s:' not in line or '=> rollout_output_time_s:' in line" in text
    assert "gzip','-1'" in text
    assert 'Incomplete arm retained' in text


def test_full_world_prepares_buckets_without_elastic_kv_fallback(monkeypatch):
    import os
    import sys
    from types import ModuleType,SimpleNamespace as NS
    source=ast.parse((ROOT/'verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py').read_text())
    cls=next(n for n in source.body if isinstance(n,ast.ClassDef) and n.name=='vLLMRollout')
    method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='prepare_mode1_step')
    calls=[]
    group_module=ModuleType('vllm.distributed.parallel_state')
    group_module.get_world_group=lambda:NS(cpu_group='world_cpu')
    bucket_module=ModuleType('vllm_ascend.compilation.step_buckets')
    bucket_module.configure_step_buckets=lambda runner,sizes:calls.append(('buckets',sizes)) or True
    monkeypatch.setitem(sys.modules,group_module.__name__,group_module)
    monkeypatch.setitem(sys.modules,bucket_module.__name__,bucket_module)
    monkeypatch.setenv('VLLM_ASCEND_ELASTIC_EXECUTION_MODE','0')
    globals_=dict(os=os,Any=object,_env_flag=lambda n:n=='VLLM_ASCEND_MODE1_ADAPTIVE_KV_RESIZE',
                  _sync_shrink_aware_env_from_meta=lambda meta:None,
                  torch=NS(distributed=NS(barrier=lambda **kw:calls.append(('barrier',kw)))),
                  logger=NS(warning=lambda *args:None))
    exec(compile(ast.Module(body=[method],type_ignores=[]),'<prepare_step>','exec'),globals_)
    runner=NS(manual_free_cache_engine=False,sleep_level=2,model_runner=object())
    globals_['prepare_mode1_step'](runner,dict(global_steps=1,aclgraph_capture_sizes=[1,4,8,13,17,25,29,32]))
    assert runner._needs_rollout_aclgraph_recapture
    assert [c[0] for c in calls]==['barrier','buckets','barrier']
    monkeypatch.setenv('VLLM_ASCEND_ELASTIC_EXECUTION_MODE','1')
    with pytest.raises(RuntimeError,match='Missing mode1'):
        globals_['prepare_mode1_step'](runner,dict(global_steps=1))


@pytest.mark.parametrize('step,expected',[(1,'5120'),(2,'7168'),(3,'10240'),
                                        (4,'14848'),(5,'16384'),(99,'16384'),(-1,'16384'),(None,'16384')])
def test_step_tailguard_resolver_matches_legacy_behavior(step,expected):
    path=ROOT/'verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py'
    tree=ast.parse(path.read_text())
    node=next(n for n in tree.body if isinstance(n,ast.FunctionDef)
              and n.name=='_resolve_step_tail_validation_caps')
    namespace={}
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),namespace)
    resolver=namespace[node.name]
    schedule='5120;7168;10240;14848;16384'
    assert resolver('999',schedule,step)==expected
    assert resolver(None,'',step) is None
    assert resolver('999',None,step)=='999'
    source=path.read_text()
    assert 'os.environ.get("VERL_ELASTIC_TAIL_VALIDATE_LEVEL_TOKENS_BY_STEP")' in source
    assert 'TAILGUARD_CAP_AUDIT' in source


def test_tailguard_evidence_is_required_not_just_environment(tmp_path):
    (tmp_path/'oracle').mkdir();(tmp_path/'logs').mkdir()
    (tmp_path/'oracle/rank_plan_summary.json').write_text(json.dumps([
        dict(step=s,tail_guard_response_cap=512*s) for s in range(1,6)]))
    with pytest.raises(RuntimeError,match='actual per-step'):
        suite.tailguard_caps(tmp_path)
    records=[dict(step=s,rank=r,max_tokens=512*s) for s in range(1,6) for r in range(16)]
    p=tmp_path/'logs/run.txt'
    p.write_text('\n'.join('TAILGUARD_CAP_AUDIT '+json.dumps(a) for a in records))
    assert suite.tailguard_caps(tmp_path)[2]==1024
    records[0]['max_tokens']=16384
    p.write_text('\n'.join('TAILGUARD_CAP_AUDIT '+json.dumps(a) for a in records))
    with pytest.raises(RuntimeError,match='Actual sampling cap'):
        suite.tailguard_caps(tmp_path)


def test_default_global_limit_on_guarded_plan_still_has_cap_evidence():
    path=ROOT/'verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py'
    source=path.read_text()
    assert 'cap_plan.get("tail_guard_response_cap") is not None' in source
    assert 'if response_cap is not None or tail_validate_caps or planned_cap_present:' in source
    assert 'max_tokens=int(kwargs.get(' in source


def test_response_mask_first_padded_eos_is_not_raw_generation_overrun():
    lengths=[100]*511+[7169]
    work,synthetic=suite.validate_response_lengths(lengths,7168,True)
    assert work==51100+7168 and synthetic==1
    with pytest.raises(RuntimeError,match='response-mask'):
        suite.validate_response_lengths([100]*511+[7170],7168,True)
    with pytest.raises(RuntimeError,match='response-mask'):
        suite.validate_response_lengths(lengths,7168,False)
    with pytest.raises(RuntimeError,match='response-mask'):
        suite.validate_response_lengths([100]*511+[16385],16384,True)
    assert suite.validate_response_lengths([16384]*512,16384,True)==(16384*512,0)


@pytest.mark.parametrize('repo',['qwen3','qwen3_shrink_aware'])
def test_original_and_current_masks_share_one_pad_eos_convention(repo):
    import torch
    path=ROOT.parent/repo/'verl/utils/torch_functional.py'
    node=next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef)
              and n.name=='get_response_mask')
    namespace={'torch':torch}
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),namespace)
    mask=namespace['get_response_mask'](torch.tensor([[11,12,13,0,0]]),eos_token=0)
    assert mask.tolist()==[[1,1,1,1,0]]
