from pathlib import Path
from run_code_natural_f2 import build_training, planner_command, CAPS
import code_task


def test_true_natural_f2_not_planned_residency_or_repair_replay():
    env,cmd = build_training(Path('/tmp/test_n2_code'))
    assert env['MIN_ADAPTIVE_FLOOR'] == '2'
    assert env['FLOOR_KV_CAPS'] == CAPS
    assert env['VLLM_ASCEND_MODE1_PARITY_MAX_KV_TOKENS_FLOOR8'] == '288256'
    assert env['ADAFLOOR_GRAPH_BASE_RUNNER'].endswith('floor2_natural_tailguard_reuse_epoch0_2epoch.sh')
    assert 'FAIR_CODE_REPAIR_TRACE' not in env
    assert 'adafloor_n_f2' in cmd
    assert env['ADAFLOOR_FROZEN_INPUTS_DIR'] == '/tmp/test_n2_code/frozen_inputs'
    assert env['VLLM_ROLLOUT_SLEEP_LEVEL'] == '2'
    assert env['VLLM_ROLLOUT_NATIVE_SLEEP_GRAPH_LIFECYCLE'] == '1'
    assert env['TASK_QUEUE_ENABLE'] == '1'


def test_frozen_same_input_seed_and_single_epoch():
    env,cmd = build_training(Path('/tmp/test_n2_code'))
    assert env['COMMON_EPOCH0_ROOT'] == str(code_task.COMMON)
    assert env['TRAIN_FILE'] == str(code_task.TRAIN)
    assert env['VERL_PAIRED_REQUEST_SAMPLING_SEEDS'] == '1'
    assert env['FAIR_TRAIN_STEPS'] == '5'
    assert env['FAIR_FREEZE_ACTOR'] == '1'
    assert env['FAIR_START_EPOCH'] == '1' and env['FAIR_TOTAL_EPOCHS'] == '2'
    assert env['FAIR_MAX_RESPONSE_LENGTH'] == '16384'
    assert 'actor_rollout_ref.rollout.seed=401' in cmd
    assert [x for x in cmd if x.startswith('actor_rollout_ref.actor.optim.lr=')][-1].endswith('=0.0')
    assert 'actor_rollout_ref.rollout.cudagraph_mode=FULL_DECODE_ONLY' in cmd


def test_cpu_plan_is_natural_capacity_unforced_and_bounded():
    cmd = planner_command(Path('/tmp/preview'))
    assert cmd[1] == '-u'
    assert cmd[2].endswith('optimized_planner.py')
    assert cmd[cmd.index('--min-adaptive-floor')+1] == '2'
    assert cmd[cmd.index('--floor-kv-caps')+1] == CAPS
    assert cmd[cmd.index('--max-cross-step-repair-swaps')+1] == '32'
    assert '--force-selected-floor' not in cmd
    assert cmd[cmd.index('--train-file')+1] == str(code_task.TRAIN)


def test_frozen_manifest_contains_every_runtime_auditor_required_key():
    import hashlib,json
    from run_code_natural_f2 import bundle_metadata
    files={'plan.json':'a'*64}
    metadata=bundle_metadata(files,policy='natural_f2')
    assert metadata['rollout_n']==16
    assert metadata['batch_size']==32
    assert metadata['seed']==401
    assert metadata['bundle_id']==hashlib.sha256(json.dumps(files,sort_keys=True).encode()).hexdigest()
    assert metadata['files']==files


def test_manifest_passes_actual_runtime_auditor_schema(tmp_path,monkeypatch):
    import hashlib,importlib.util,json,sys
    from types import SimpleNamespace as NS
    from run_code_natural_f2 import bundle_metadata
    path=code_task.SOURCE/'verl/utils/frozen_graph_inputs.py'
    spec=importlib.util.spec_from_file_location('frozen_runtime_schema_regression',path)
    auditor=importlib.util.module_from_spec(spec);spec.loader.exec_module(auditor)
    plan=dict(step=1,rank_to_dataset_item_idx={'0':[0,1]},kv_cap=280576,
              aclgraph_capture_sizes=[4,7,10,16,18,22,25,32],
              tail_guard_response_cap=16384,selected_floor=4)
    raw=json.dumps([plan]).encode()
    (tmp_path/'length_sorted_rank_plan.json').write_bytes(raw)
    metadata=bundle_metadata({'length_sorted_rank_plan.json':hashlib.sha256(raw).hexdigest()})
    (tmp_path/'manifest.json').write_text(json.dumps(metadata))
    identities=[(i,s) for i in [0,1] for s in range(16)]
    prompts=NS(meta_info={'global_steps':1,'shrink_aware_runtime':{'selected_floor':4}},
        non_tensor_batch={'dataset_item_idx':[i for i,s in identities],
                          'rollout_sample_index':[s for i,s in identities]})
    inputs=[{'prompt_token_ids':[i+100]} for i,s in identities]
    derive=lambda seed,tokens,s: ('test_prompt_hash',seed+sum(tokens)+s)
    monkeypatch.setitem(sys.modules,'verl.utils.rollout_seeding',NS(derive_request_seed=derive))
    monkeypatch.setenv('ADAFLOOR_FROZEN_INPUTS_DIR',str(tmp_path))
    log=[]
    auditor.audit_frozen_rank_inputs(prompts,inputs,
        [derive(401,x['prompt_token_ids'],s)[1] for x,(i,s) in zip(inputs,identities)],
        rank=0,actual_kv_tokens=280576,actual_capture_sizes=plan['aclgraph_capture_sizes'],
        actual_response_cap=16384,logger=NS(warning=lambda *a:log.append(a)))
    assert len(log)==1 and 'FROZEN_GRAPH_INPUT_AUDIT' in log[0][0]
