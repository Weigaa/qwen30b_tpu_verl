#!/usr/bin/env python3
"""One normal updated-policy epoch after consolidation, not a short fixture."""
import argparse
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess

HERE=Path(__file__).resolve().parent
MAIN=HERE.parent/'qwen3_main'
REFERENCE=Path('/data/adafloor_graph_results/mlsys_n2_f8_349952_20261025T143000Z/graph_sidecar')
CAPS='2:131072,4:280576,8:349952,16:380800'


def helper():
    spec=importlib.util.spec_from_file_location('main_epoch_sidecar',MAIN/'tools/run_v014_sidecar_training.py')
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_launch(root):
    h=helper()
    env,command=h.build_launch(Path(root),'graph',60.,'gt')
    env.update(FAIR_START_EPOCH='1',FAIR_TOTAL_EPOCHS='2',FAIR_TRAIN_STEPS='5',
               FAIR_FREEZE_ACTOR='0',FAIR_KEEP_COMPLETED_CHECKPOINTS='0',
               VLLM_ASCEND_MODE1_PARITY_MAX_KV_TOKENS_FLOOR8='349952',FLOOR_KV_CAPS=CAPS)
    for key in ('HCCL_DETERMINISTIC','LCCL_DETERMINISTIC'):
        env.pop(key,None)
    assert env['FAIR_MAX_RESPONSE_LENGTH']=='16384'
    assert env['VERL_PAIRED_REQUEST_SAMPLING_SEEDS']=='0'
    assert env['VERL_SIDECAR_REPEAT_UNTIL_KILLED']=='1'
    assert env['VERL_SIDECAR_MAX_TOKENS']=='4096'
    assert env['VERL_SIDECAR_MAX_NUM_SEQS']=='128'
    assert not any(k.startswith('VERL_ELASTIC_TAIL_VALIDATE_') for k in env)
    assert 'actor_rollout_ref.actor.optim.lr=1e-6' in command
    return h,env,command


def summarize(root):
    root=Path(root);h=helper();write=h.suite.write_json
    launch=json.loads((root/'LAUNCH.json').read_text())
    if h.suite.execution_digest()!=launch['execution_sha256']:
        raise RuntimeError('Execution sources changed during formal epoch')
    if json.loads((root/'DRIVER_RESULT.json').read_text())['exit_code']!=0:
        raise RuntimeError('Trainer child did not complete successfully')
    run=root/'seed_0/natural_f2/fig6'
    receipt=run/'CHECKPOINTS_REMOVED_AFTER_VALIDATION.txt'
    if not receipt.is_file() or 'validated_epochs=001' not in receipt.read_text():
        raise RuntimeError('Original fair-runner single-epoch validation/cleanup missing')
    epoch=run/'epoch_001_mode1_natural'
    primary=h.suite.parse_epoch(epoch,require_tailguard=True)
    side=h.sidecar_summary(epoch,'graph',60.,True)
    prior=json.loads((REFERENCE/'TRAINING_SIDECAR_RESULT.json').read_text())['primary']['epochs'][0]
    import pyarrow.parquet as pq
    oracle=epoch/'oracle';old=Path(prior['epoch_dir'])/'oracle'
    same_prompts=(pq.read_table(oracle/'length_sorted_train.parquet').to_pylist()==
                  pq.read_table(old/'length_sorted_train.parquet').to_pylist())
    new_plan=json.loads((oracle/'length_sorted_rank_plan.json').read_text())
    old_plan=json.loads((old/'length_sorted_rank_plan.json').read_text())
    compare_keys=['rank_to_source_idx','rank_to_dataset_item_idx','selected_floor','kv_cap',
                  'tail_guard_response_cap','aclgraph_capture_sizes']
    same_plan=all(n[k]==o[k] for n,o in zip(new_plan,old_plan) for k in compare_keys)
    result=dict(status='PASS',epoch=1,steps=5,seed=0,actor_frozen=False,actor_lr=1e-6,
        primary=primary,sidecar=side,floor_kv_caps=CAPS,published_commit=launch['published_commit'],
        reference_epoch=prior,primary_Q_change_percent=100*(primary['response_tokens_s']/prior['response_tokens_s']-1),
        reward_change=primary['reward']-prior['reward'],same_prompt_order_and_contents=same_prompts,
        same_rank_floor_cap_bucket_plan=same_plan,actual_floor2_not_required_for_success=True,
        sidecar_completed=sum(w['completed'] for w in side['windows']),
        sidecar_completed_before_stop=side['completed_before_stop'],
        scope='One complete normal RL epoch. Updated-policy descriptive reference, not frozen causal interference.',
        common_epoch0_preserved=h.suite.COMMON.is_dir())
    write(root/'RESULT.json',result)
    write(root/'STATE.json',dict(state='COMPLETE_VERIFIED',utc=datetime.now(timezone.utc).isoformat(),
          primary_output_tokens_s=primary['response_tokens_s'],reward=primary['reward'],
          comparison_percent=result['primary_Q_change_percent'],sidecar_completed=result['sidecar_completed']))
    print(json.dumps(result,indent=2))
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['dry-run','run','summarize'])
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--published-commit',default='014_MLSYS_adafloor')
    a=p.parse_args();root=a.output.resolve()
    if a.action=='summarize':
        summarize(root)
        return
    h,env,command=build_launch(root)
    commit=subprocess.run(['git','rev-parse',a.published_commit],cwd=MAIN,
                          capture_output=True,text=True,check=True).stdout.strip()
    launch=dict(repo=str(MAIN),published_commit=commit,command=command,
        execution_sha256=h.suite.execution_digest(),floor_kv_caps=CAPS,
        workload='One epoch1 from protected public epoch0,160 DeepScaleR prompts,32xn16x5 steps,normal stochastic LR1e-6 updates',
        reference=str(REFERENCE),env={k:v for k,v in env.items()
        if k.startswith(('FAIR_','VERL_','VLLM_','ADAFLOOR_','DYNAMIC_')) or
        k in ('FLOOR_KV_CAPS','TASK_QUEUE_ENABLE','COMMON_EPOCH0_ROOT','PYTORCH_NPU_ALLOC_CONF','HCCL_BUFFSIZE')})
    if a.action=='dry-run':
        print(json.dumps(launch,indent=2))
        return
    if shutil.disk_usage('/data').free<65*1024**3:
        raise RuntimeError('Need65GiB free before formal checkpoint creation')
    smi=subprocess.run(['npu-smi','info'],capture_output=True,text=True,check=True).stdout
    if smi.count('No running processes found in NPU')!=8:
        raise RuntimeError('NPUs are not idle, refusing concurrent test')
    root.mkdir(parents=True,exist_ok=False)
    write=h.suite.write_json
    write(root/'LAUNCH.json',launch)
    shutil.copyfile(__file__,root/'controller_snapshot.py')
    write(root/'STATE.json',dict(state='RUNNING',utc=datetime.now(timezone.utc).isoformat(),
                               published_commit=commit,primary='Natural-F2 FULL/TQ1 native sleep2',
                               sidecar='Qwen2.5-1.5B FULL/TQ1,8 TP1 replicas,strict >60s history gate'))
    for key in ('RAY_TMPDIR','XDG_CACHE_HOME','TORCHAIR_CACHE_HOME','TRITON_CACHE_DIR','ASCEND_CACHE_PATH','ASCEND_WORK_PATH'):
        if key in env:Path(env[key]).mkdir(parents=True,exist_ok=True)
    with (root/'driver.log').open('x') as log:
        rc=subprocess.run(command,cwd=MAIN,env=env,stdout=log,stderr=subprocess.STDOUT).returncode
    write(root/'DRIVER_RESULT.json',dict(exit_code=rc,utc=datetime.now(timezone.utc).isoformat()))
    if rc:
        write(root/'STATE.json',dict(state='STOPPED_WITH_ERROR',exit_code=rc,raw_outputs_retained=True))
        raise RuntimeError(f'Formal epoch exited{rc}, inspect {root}/driver.log')
    summarize(root)


if __name__=='__main__':main()
