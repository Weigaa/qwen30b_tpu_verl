#!/usr/bin/env python3
"""Requested N4 model arms, model-specific 16-query graph preflight first.

Primary runtime and the Qwen2.5 reference stay read-only. No new seeds, no
extra epoch0, no opportunistic gate changes and no eager fallback.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

import run_n2_f8_campaign as c
import run_n4_graph_sidecar as n4

SECONDARY = c.PARENT/'qwen3_sidecar'
REFERENCE = Path('/data/adafloor_graph_results/mlsys_n4_graph_sidecar_gate60_seed0_20261026T124500Z')
WRAPPER = c.CODE/'run_n4_sidecar_model.sh'
PROFILES = {
    'qwen3_8b': dict(model='/data/Qwen3-8B', tp=4, dp=2, replicas=1, ep=False),
    'pangu_moe': dict(model='/data/pangu-pro-moe-model', tp=1, dp=8, replicas=1, ep=True),
}


def build_launch(root, name):
    h, env, command = n4.build_launch(root)
    profile = PROFILES[name]
    env.update(VERL_SIDECAR_MODEL_PATH=profile['model'],
        VERL_SIDECAR_TENSOR_PARALLEL_SIZE=str(profile['tp']),
        VERL_SIDECAR_DATA_PARALLEL_SIZE=str(profile['dp']),
        VERL_SIDECAR_REPLICA_COUNT=str(profile['replicas']),
        VERL_SIDECAR_ENABLE_EXPERT_PARALLEL=str(int(profile['ep'])),
        VERL_SIDECAR_HCCL_DETERMINISTIC='true',VERL_SIDECAR_LCCL_DETERMINISTIC='1',
        VERL_SIDECAR_PARALLEL_MODE='hybrid',
        VERL_SIDECAR_SCRIPT=str(WRAPPER))
    # Keep per-replica concurrency, chunk size and pending-work cap128 identical.
    # Global GSM8K dataset and question identities remain unchanged. With TP4
    # there are fewer independent shards, not a new selected prompt subset.
    assert profile['tp']*profile['dp']*profile['replicas']==8
    assert env['VERL_SIDECAR_MAX_NUM_SEQS']=='128'
    assert env['VERL_SIDECAR_MAX_PROMPTS_PER_REPLICA']=='128'
    assert env['VERL_SIDECAR_MAX_TOKENS']=='4096'
    assert env['VERL_SIDECAR_WINDOW_GATE_MIN_SECONDS']=='60.0'
    return h, env, command


def secondary_digest():
    return c.module('model_sidecar_suite',SECONDARY/'tools/run_mlsys_fig6.py').execution_digest()


def new_protocol(root,models=None,reuse_eager_from=None):
    models=list(models or PROFILES)
    root.mkdir(parents=True, exist_ok=False)
    c.write(root/'PROTOCOL.json', dict(status='PRESPECIFIED_N4_SIDECAR_MODEL_ARMS',
        models={name:PROFILES[name] for name in models}, order=models, seed=0, epochs=[1,2],steps_each=5,
        primary_protocol='Exactly the validated N4 Qwen2.5 arm, common epoch0, own epoch1 chain, FULL/TQ1/native2/lifecycle1/KV-first',
        secondary_protocol='Same GSM8K train, raw prompts, greedy n1, max4096/model6144, maxseq128, maxbatched65536, fraction0.8, graph sizes[1,2,4,8,16,32,64,128], >60s history gate',
        primary_runtime_unchanged=True,reference=str(REFERENCE),
        controller_sha256=c.sha(c.__file__), model_controller_sha256=c.sha(__file__),
        wrapper_sha256=c.sha(WRAPPER),
        stop_verifier_sha256=c.sha(c.CODE/'verify_n4_sidecar_stop.py'),
        residual_preflight_sha256=c.sha(c.CODE/'sidecar_residual_preflight.py'),
        residual_preflight_gib_per_device={name:16 if name=='pangu_moe' else 0 for name in models},
        memory_accounting='CacheConfig per-instance0.8 limit, owned non-Torch growth attribution, independent physical-free/transient/graph-headroom clamp. Only isolated secondary runtime changed.',
        execution_sha256={'qwen3_graph_sidecar_v014':n4.helper().suite.execution_digest(),
                          'qwen3_graph_sidecar_models_v014':secondary_digest()},
        user_selected_optimal_layouts='User confirmed internal DP2 x TP4 for Qwen3-8B, internal DP8 x TP1 with EP8 for PanGu. Both use v0.14 external_launcher with a single coordinated8-rank engine world, never independent replicas.',
        preflight='One16-query eager and graph check per model, plus same-input shadow only if free-batch greedy differs; diagnostic runs excluded from primary performance',
        reuse_eager_from=str(reuse_eager_from) if reuse_eager_from else None,
        stop_preflight='128-request FULL graph admission, file stop during decode, original3s soft/2s TERM deadline, checkpoint/output/replay validation before formal arm',
        stop_policy='Stop affected model on incorrect graph, OOM or unsafe handback. Preserve evidence. Never change gate, capacities, seeds or use eager as a performance fallback.',
        checkpoint_policy='65GiB on both checkpoint disks; only run-owned private checkpoints cleaned after validated two-epoch completion'))
    shutil.copyfile(__file__,root/'controller_snapshot.py')
    shutil.copyfile(WRAPPER,root/'secondary_wrapper_snapshot.sh')


def check_contract(root):
    c.check_sources(root)
    protocol=json.loads((root/'PROTOCOL.json').read_text())
    if c.sha(__file__)!=protocol['model_controller_sha256'] or c.sha(WRAPPER)!=protocol['wrapper_sha256']:
        raise RuntimeError('Unrecorded model controller/wrapper change')
    if protocol.get('stop_verifier_sha256') and c.sha(c.CODE/'verify_n4_sidecar_stop.py')!=protocol['stop_verifier_sha256']:
        raise RuntimeError('Stop verifier changed after protocol was pinned')
    if protocol.get('residual_preflight_sha256') and c.sha(c.CODE/'sidecar_residual_preflight.py')!=protocol['residual_preflight_sha256']:
        raise RuntimeError('Residual diagnostic changed after protocol was pinned')


def preflight(root, name, reuse_eager_from=None):
    destination=root/'preflight'/name
    if (destination/'PARITY_RESULT.json').exists():
        return proof(root,name)
    if destination.exists():
        raise RuntimeError('Incomplete preflight retained; diagnose instead of automatically repeating '+name)
    c.idle(root)
    profile=PROFILES[name]
    command=[sys.executable,str(SECONDARY/'tools/run_sidecar_graph_parity.py'),
        '--output',str(destination),'--model',profile['model'],
        '--tensor-parallel-size',str(profile['tp']),
        '--data-parallel-size',str(profile['dp']),'--device','0']
    if profile['ep']:command.append('--enable-expert-parallel')
    if reuse_eager_from:
        original=Path(reuse_eager_from)/'preflight'/name
        old=json.loads((original/'PARITY_RESULT.json').read_text())
        saved=json.loads((original/'eager/launch.json').read_text())
        assert old['status']=='PASS' and old['checked_prompts']==16 and old['greedy_token_ids_exact']
        assert old['model']==profile['model'] and old['tensor_parallel_size']==profile['tp']
        assert old['data_parallel_size']==profile['dp'] and old['enable_expert_parallel']==profile['ep']
        for key,wanted in [('VERL_SIDECAR_MAX_NUM_SEQS','128'),
                           ('VERL_SIDECAR_MAX_NUM_BATCHED_TOKENS','65536'),
                           ('VERL_SIDECAR_MAX_TOKENS','4096'),('VERL_SIDECAR_MAX_MODEL_LEN','6144')]:
            assert saved[key]==wanted,(key,saved.get(key))
        destination.mkdir(parents=True)
        (destination/'eager').symlink_to((original/'eager').resolve(),target_is_directory=True)
        c.write(destination/'EAGER_REUSE.json',dict(source=str(original),
            source_proof_sha256=c.sha(original/'PARITY_RESULT.json'),
            source_outputs_sha256=c.sha(original/'eager/outputs.jsonl'),
            reason='Reuse frozen-model16-query eager outputs; rerun graph with repaired host checkpoint/output lifecycle. No model math changes.'))
        command.append('--resume-after-eager')
    env=os.environ.copy()
    reference_pythonpath=json.loads((n4.PARITY.parent/'graph/launch.json').read_text())['PYTHONPATH']
    cann_paths=':'.join(p for p in reference_pythonpath.split(':') if p.startswith('/usr/local/Ascend/'))
    env['PYTHONPATH']=str(SECONDARY)+':'+cann_paths
    c.state(root,'PREFLIGHT_'+name,command=command)
    root.joinpath('preflight').mkdir(exist_ok=True)
    from sidecar_residual_preflight import reserve
    residual_gib=json.loads((root/'PROTOCOL.json').read_text()).get('residual_preflight_gib_per_device',{}).get(name,0)
    with reserve(destination/'co_resident_residual',residual_gib,env):
        with (root/'preflight'/f'{name}.driver.log').open('x') as output:
            rc=subprocess.run(command,cwd=SECONDARY,env=env,stdout=output,stderr=subprocess.STDOUT).returncode
    c.write(root/'preflight'/f'{name}.DRIVER_RESULT.json',dict(exit_code=rc))
    if rc:raise RuntimeError(f'{name} preflight exited{rc}, no formal arm started')
    return proof(root,name)


def proof(root,name):
    value=json.loads((root/'preflight'/name/'PARITY_RESULT.json').read_text())
    assert value['model']==PROFILES[name]['model'] and value['checked_prompts']==16
    assert value['tensor_parallel_size']==PROFILES[name]['tp'] and value['status']=='PASS'
    assert value['data_parallel_size']==PROFILES[name]['dp']
    assert value['enable_expert_parallel']==PROFILES[name]['ep']
    shadow=value.get('same_input_shadow') or {}
    assert value['greedy_token_ids_exact'] or (
        value['same_padded_input_graph_parity_exact'] and
        shadow['status']=='ALL_SAME_PADDED_INPUT_SHADOW_CHECKS_PASSED' and
        shadow['max_hidden_abs']==shadow['max_logits_abs']==0.0 and
        shadow['post_checks']==shadow['pre_checks'] and shadow['post_checks']>0 and
        shadow['post_mismatches']==shadow['pre_argmax_mismatches']==0 and
        shadow['diagnostic_generation_unchanged'])
    for key,path in [('sidecar_script_sha256','internal/run_elastic_sidecar_infer.sh'),
                     ('aclgraph_sha256','vllm_ascend/compilation/acl_graph.py'),
                     ('model_runner_sha256','vllm_ascend/worker/model_runner_v1.py')]:
        assert value[key]==c.sha(SECONDARY/path)
    return value


def sidecar_summary(h,epoch,replicas):
    # Existing raw output/lifecycle validator is reused from an isolated source
    # copy with only the expected replica-count argument generalized below.
    return h.sidecar_summary(epoch,'graph',60.0,True,expected_replicas=replicas)


def summarize(root,name):
    h=c.module('models_summary',SECONDARY/'tools/run_v014_sidecar_training.py')
    arm=root/name
    primary=n4.helper().suite.parse_arm(arm,0,'natural_f4')
    baseline=n4.helper().suite.parse_arm(c.OLD,0,'natural_f4')
    sides=[sidecar_summary(h,Path(e['epoch_dir']),PROFILES[name]['replicas']*PROFILES[name]['dp']) for e in primary['epochs']]
    windows=[w for s in sides for w in s['windows']]
    for e in primary['epochs']:
        ep=Path(e['epoch_dir']);audits=n4.helper().suite.json_records(ep,'MODE1_STEP_RUNTIME_AUDIT ')
        assert len(audits)==80 and all(a['complete'] and a['full_iterations']>0 for a in audits)
        answers=[json.loads(line) for p in (ep/'rollout_data').glob('*.jsonl')
                 for line in p.open() if line.strip()]
        assert len(answers)==2560 and all(a['output'].strip() for a in answers)
    result=dict(status='PASS',seed=0,policy='natural_f4',model=name,model_profile=PROFILES[name],
        primary=primary,baseline=baseline,sidecar=sides,epochs=[1,2],sidecar_execution='graph',
        gate='>60s',actual_sidecar_starts=len(windows),
        completed_before_stop=sum(w['completed_before_stop'] for w in windows),
        completed_including_drain=sum(w['completed'] for w in windows),
        primary_Q_change_percent=100*(primary['response_tokens_s']/baseline['response_tokens_s']-1),
        scope='Same-seed updated policies, descriptive not frozen causal ON/OFF. Startup-consuming/no-replay windows retained, not replaced.',
        reference_qwen25=str(REFERENCE/'TRAINING_SIDECAR_RESULT.json'))
    c.write(arm/'TRAINING_SIDECAR_RESULT.json',result)
    c.write(arm/'seed_0/natural_f4/ARM_PASS.json',primary)
    c.state(arm,'COMPLETE',starts=len(windows),completed_before_stop=result['completed_before_stop'])
    return result


def formal(root,name):
    check_contract(root)
    arm=root/name
    if (arm/'TRAINING_SIDECAR_RESULT.json').exists():
        return summarize(root,name)
    if (arm/'v014_launch.json').exists():raise RuntimeError('Refusing duplicate formal run '+name)
    h,env,command=build_launch(arm,name)
    arm.mkdir(exist_ok=True)
    p=proof(root,name)
    c.write(arm/'v014_launch.json',dict(command=command,model_profile=PROFILES[name],
        execution_sha256=h.suite.execution_digest(),secondary_execution_sha256=secondary_digest(),
        env={k:v for k,v in env.items() if k.startswith(('VERL_','VLLM_','FAIR_','DYNAMIC_','ADAFLOOR_')) or k in ('TASK_QUEUE_ENABLE','FLOOR_KV_CAPS')},
        sidecar_parity_proof_sha256=c.sha(root/'preflight'/name/'PARITY_RESULT.json')))
    c.write(arm/'SIDECAR_GREEDY_PARITY_PROOF.json',p)
    storage=c.split_prepare(root,name)
    run=arm/'seed_0/natural_f4/fig6'
    c.launch(root,'N4_'+name,arm/'execution',c.SIDECAR,env,command,run_dir=run,storage=storage)
    result=summarize(root,name)
    c.split_cleanup(root,name,storage,run)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('dry-run','run','summarize'))
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--models',nargs='+',choices=tuple(PROFILES),default=list(PROFILES))
    parser.add_argument('--reuse-eager-from',type=Path)
    args=parser.parse_args();root=args.output.resolve()
    if len(set(args.models))!=len(args.models):parser.error('Duplicate model names')
    if args.mode=='dry-run':
        for name in args.models:
            _,env,command=build_launch(root/name,name)
            print(json.dumps(dict(model=name,profile=PROFILES[name],command=command,
                env={k:v for k,v in env.items() if k.startswith('VERL_SIDECAR_') or k=='FLOOR_KV_CAPS'}),indent=2))
        return
    if args.mode=='summarize':
        for name in args.models:summarize(root,name)
        return
    if not root.exists():new_protocol(root,args.models,args.reuse_eager_from)
    if json.loads((root/'PROTOCOL.json').read_text())['order']!=args.models:
        raise RuntimeError('Model order differs from prespecified protocol')
    check_contract(root)
    results=[]
    for name in args.models:
        try:
            preflight(root,name,args.reuse_eager_from)
            check_contract(root)
            import verify_n4_sidecar_stop
            verify_n4_sidecar_stop.verify(root,name)
        except Exception as error:
            c.state(root,'STOPPED_WITH_ERROR',model=name,error=repr(error))
            raise
    for name in args.models:
        try:
            results.append(formal(root,name))
        except Exception as error:
            c.state(root,'STOPPED_WITH_ERROR',model=name,error=repr(error))
            raise
    c.write(root/'COMPARISON.json',dict(status='ALL_REQUESTED_MODEL_ARMS_COMPLETE',
        requested_models=args.models,arms=results,reference_qwen25=str(REFERENCE)))
    c.state(root,'COMPLETE',models=args.models)


if __name__=='__main__':main()
