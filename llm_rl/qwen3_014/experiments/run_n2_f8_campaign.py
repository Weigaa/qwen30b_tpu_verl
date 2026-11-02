#!/usr/bin/env python3
"""Recalibrate Natural-F2 F8 only, then rerun its dependent v0.14 evidence.

All original result roots, runtime sources and other capacity families remain
read-only. The candidate is never adopted unless the explicit capacity gate
passes. A failure stops the queue; no smaller KV, eager fallback or new seeds.
"""
import argparse
from datetime import datetime,timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

CODE=Path(__file__).resolve().parent
PARENT=CODE.parent
MAIN=PARENT/'qwen3_main'
SIDECAR=PARENT/'qwen3_main'
OLD=Path('/data/adafloor_graph_results/mlsys_fig6_v014_graph_calibrated_20261017T134200Z')
OLD_SIDE=Path('/data/adafloor_graph_results/mlsys_v014_fullgraph_sidecar_natural_f2_gate60_seed0_20261025T035331Z')
MEMORY=Path('/data/adafloor_graph_results/mlsys_fixed4_hard_capacity_probe_20261017/dcmi_helper_snapshot.py')
CAPS='2:131072,4:280576,8:349952,16:380800'
SEEDS=(0,2,3)


def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value)
    return value


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n');temp.replace(path)


def state(root,value,**extra):
    record=dict(state=value,utc=datetime.now(timezone.utc).isoformat(),**extra)
    write(root/'STATE.json',record);print(json.dumps(record,ensure_ascii=False),flush=True)


def new_capacity(env):
    env=dict(env)
    env.update(VLLM_ASCEND_MODE1_PARITY_MAX_KV_TOKENS_FLOOR8='349952',FLOOR_KV_CAPS=CAPS)
    return env


def main_suite():return module('n2_new_f8_main',MAIN/'tools/run_mlsys_fig6.py')


def runtime_root(name):
    aliases = {'qwen3': 'qwen3_main', 'qwen3_graph_sidecar_v014': 'qwen3_main',
               'qwen3_mlsys_supplement_v014': 'qwen3_main',
               'qwen3_mlsys_eager_controls_v014': 'qwen3_main',
               'qwen3_graph_sidecar_models_v014': 'qwen3_sidecar'}
    return PARENT / aliases.get(name, name)


def check_sources(root):
    contract=json.loads((root/'PROTOCOL.json').read_text())
    for folder,wanted in contract['execution_sha256'].items():
        actual=module('n2_source_'+folder,runtime_root(folder)/'tools/run_mlsys_fig6.py').execution_digest()
        if actual!=wanted:raise RuntimeError('Pinned runtime changed: '+folder)
    if sha(__file__)!=contract['controller_sha256']:
        changes=list((root/'controller_amendments').glob('*.json'))
        if not any((lambda value:value.get('original_protocol_sha256')==sha(root/'PROTOCOL.json')
                    and value.get('original_controller_sha256')==contract['controller_sha256']
                    and value.get('updated_controller_sha256')==sha(__file__)
                    and value.get('runtime_and_budget_unchanged') is True)(json.loads(p.read_text()))
                   for p in changes):
            raise RuntimeError('Campaign controller changed without an explicit pinned host-only amendment')


def idle(root):
    from run_code_natural_f2 import other_trainers
    while True:
        text=subprocess.run(['npu-smi','info'],capture_output=True,text=True,check=True).stdout
        if 'Process id' not in text:raise RuntimeError('Missing NPU process status')
        processes=re.findall(r'^\|\s+\d+\s+\d+\s+\d+\s+',text.rsplit('Process id',1)[1],re.M)
        trainers=other_trainers()
        if not processes and not trainers:return
        state(root,'WAITING_FOR_IDLE_NPUS',other_trainer_pids=trainers);time.sleep(15)


def cache_dirs(env):
    for key in ('RAY_TMPDIR','XDG_CACHE_HOME','TORCHAIR_CACHE_HOME','TRITON_CACHE_DIR',
                'ASCEND_CACHE_PATH','ASCEND_WORK_PATH'):
        if key in env:Path(env[key]).mkdir(parents=True,exist_ok=True)


def snapshot_raw(destination,ray_tmp):
    destination.mkdir(parents=True,exist_ok=True)
    for source in Path(ray_tmp).glob('ray/session_*/logs/worker-*.out'):
        try:
            with source.open('rb') as stream:
                if b':actor_name:TaskRunner\n' not in stream.read(65536):continue
            for path in (source,source.with_suffix('.err')):
                if path.exists():shutil.copyfile(path,destination/path.name)
        except FileNotFoundError:pass


def split_prepare(root,label):
    storage=Path('/workspace/adafloor_graph_results')/(root.name+'_epoch2_'+label)
    for path in (root,storage.parent):
        if shutil.disk_usage(path).free<65*1024**3:
            raise RuntimeError('Need65GiB on BOTH checkpoint disks; no guard reduction: '+str(path))
    storage.mkdir(exist_ok=False)
    ownership=dict(campaign=str(root),label=label,storage=str(storage))
    write(storage/'OWNED_BY_N2_F8_CAMPAIGN.json',ownership)
    return storage


def split_cleanup(root,label,storage,run):
    if (json.loads((storage/'OWNED_BY_N2_F8_CAMPAIGN.json').read_text())!=
            dict(campaign=str(root),label=label,storage=str(storage)) or
            not (run/'CHECKPOINTS_REMOVED_AFTER_VALIDATION.txt').is_file() or
            storage.parent!=Path('/workspace/adafloor_graph_results') or
            storage.name!=root.name+'_epoch2_'+label):
        raise RuntimeError('Private checkpoint cleanup ownership/validation failed')
    shutil.rmtree(storage)
    write(run/'SPLIT_STORAGE_REMOVED.json',dict(storage=str(storage),validated=True,
          public_epoch0_untouched=True))


def launch(root,label,destination,source,env,command,*,run_dir=None,storage=None,retain_endpoint=None):
    check_sources(root);idle(root);cache_dirs(env)
    destination.mkdir(parents=True,exist_ok=True)
    if (destination/'LAUNCH.json').exists():raise RuntimeError('Refusing duplicate run: '+label)
    write(destination/'LAUNCH.json',dict(label=label,command=command,source=str(source),
        candidate_floor8=349952,env={k:v for k,v in env.items()
        if k.startswith(('VLLM_','VERL_','FAIR_','ADAFLOOR_','DYNAMIC_')) or
        k in ('FLOOR_KV_CAPS','TASK_QUEUE_ENABLE','PYTORCH_NPU_ALLOC_CONF','HCCL_BUFFSIZE',
              'COMMON_EPOCH0_ROOT','TRAIN_FILE','TRAIN_FILE_ORIG','TEST_FILE','RAY_TMPDIR')}))
    state(root,'RUNNING_'+label)
    link_installed=False;retained=False
    with (destination/'driver.log').open('x') as output:
        process=subprocess.Popen(command,cwd=source,env=env,stdout=output,stderr=subprocess.STDOUT)
        try:
            while process.poll() is None:
                snapshot_raw(destination/'raw_taskrunner_logs',env['RAY_TMPDIR'])
                if run_dir is not None:
                    first=run_dir/'epoch_001_mode1_natural'
                    if storage is not None and not link_installed and first.is_dir():
                        link=run_dir/'epoch_002_mode1_natural/checkpoints'
                        link.parent.mkdir(parents=True,exist_ok=True)
                        link.symlink_to(storage,target_is_directory=True);link_installed=True
                    if retain_endpoint is not None and not retained:
                        import retain_training_endpoint as keeper
                        checkpoint=keeper.ready(first)
                        if checkpoint is not None:
                            keeper.retain(checkpoint,retain_endpoint);retained=True
                time.sleep(2)
        finally:
            if process.poll() is None:
                process.terminate();process.wait(timeout=30)
        snapshot_raw(destination/'raw_taskrunner_logs',env['RAY_TMPDIR'])
    write(destination/'DRIVER_RESULT.json',dict(exit_code=process.returncode,
        source_unchanged=True,retained_endpoint=retained))
    check_sources(root)
    if process.returncode:
        raise RuntimeError(label+f' exited{process.returncode}; artifacts retained, no automatic retry')
    if retain_endpoint is not None and not retained:raise RuntimeError('Required epoch1 endpoint not retained')


def calibration_launch(root):
    suite=main_suite();arm=root/'capacity_gate'
    env=new_capacity(suite.arm_env(arm,0,'natural_f2'))
    checkpoint=suite.COMMON/'epoch_000_mode0_probe/checkpoints/qwen3moe_for_eagle3/global_step_5'
    env.update({k:str(v) for k,v in dict(DYNAMIC_OUTPUT_ROOT=arm,DYNAMIC_RUN_NAME='capacity_warm_f8',
        DYNAMIC_INITIAL_BASELINE_DIR=suite.COMMON/'epoch_000_mode0_probe',
        DYNAMIC_INITIAL_RESUME_CKPT=checkpoint,BASELINE_INITIAL_RESUME_CKPT=checkpoint,
        DYNAMIC_SKIP_MODE0_PROBE=1,DYNAMIC_START_EPOCH=1,DYNAMIC_TOTAL_EPOCHS=2,
        DYNAMIC_PLAN_STEPS=5,DYNAMIC_TRAIN_STEPS=3,DYNAMIC_ENABLE_CKPT_CHAIN=0,
        DYNAMIC_FORCE_SELECTED_FLOOR='',DYNAMIC_FORCE_SELECTED_FLOORS='2,8,8,16,16',
        SAVE_CKPT_ENABLE=0,TRAINER_SAVE_FREQ=-1,FAIR_FREEZE_ACTOR=0,
        RAY_TMPDIR=f'/tmp/n2f8cap_{os.getpid()}',VLLM_ROLLOUT_PHASE_MEMORY_LOG=0).items()})
    command=['bash',str(MAIN/'run_mode1_local_length_sorted_e2e_adaptive_floor4_aclgraph.sh'),
        *suite.arm_command(0,'natural_f2')[3:],'trainer.total_training_steps=3',
        'trainer.save_freq=-1','actor_rollout_ref.actor.optim.lr=1e-6',
        'trainer.resume_mode=resume_path',f'trainer.resume_from_path={checkpoint}']
    return env,command


def calibration_tracebacks(text):
    """Recognize only the exact TBE daemon exit race; retain every other trace.

    This does NOT exempt OOM or a failed child. Those are separately rejected,
    and full per-rank/step replay, KV, quality and actor-update audits are still
    required. Original logs remain unchanged.
    """
    benign=[]
    pattern=re.compile(r'Traceback \(most recent call last\):[^\n]*\n'
        r'(?:[^\n]*\n){1,12}?[^\n]*ProcessLookupError: \[Errno 3\] No such process')
    def classify(match):
        block=match.group()
        if (block.count('Traceback (most recent call last):')==1 and
                'tbe/common/repository_manager/utils/common.py' in block and
                'daemon_process' in block and 'os.kill(mgr_pid, signal.SIGKILL)' in block):
            benign.append(block)
            return '\nKNOWN_TBE_DAEMON_EXIT_RACE_RETAINED_IN_ORIGINAL_LOG\n'
        return block
    remaining=pattern.sub(classify,text)
    return benign,'Traceback' in remaining


def audit_calibration(root):
    suite=main_suite();arm=root/'capacity_gate'
    epoch=arm/'capacity_warm_f8/epoch_001_mode1_natural'
    logs=list((epoch/'logs').glob('*.txt'))
    if len(logs)!=1:raise RuntimeError('Not one calibration trainer log')
    text=logs[0].read_text(errors='replace')
    benign_traces,unexpected_trace=calibration_tracebacks(text)
    if (not re.search(r'\[run\] end_time=\S+ exit_code=0',text) or unexpected_trace or
            re.search(r'OutOfMemoryError|NPU out of memory|Memory_Allocation_Failure',text)):
        raise RuntimeError('Capacity diagnostic did not finish cleanly')
    audits=suite.json_records(epoch,'MODE1_STEP_RUNTIME_AUDIT ')
    if (len(audits)!=48 or {(a['step'],a['rank']) for a in audits}!=
            {(s,r) for s in (1,2,3) for r in range(16)} or
            any(not a['complete'] or a.get('full_iterations',0)<=0 or
                a['kv_tokens']!=({1:131072,2:349952,3:349952}[a['step']]) for a in audits)):
        raise RuntimeError('Missing actual low-floor then two warm F8 FULL windows at requested capacity')
    metrics=[dict(re.findall(r'([\w/]+):([^ ]+)',line)) for line in text.splitlines()
             if re.search(r'(?:^|\s)step:\d+\s+-',line)]
    if len(metrics)!=3 or any(float(m.get('actor/lr',-1))!=1e-6 or
            not math.isfinite(float(m.get('actor/grad_norm','nan'))) or
            float(m.get('actor/grad_norm',0))<=0 or float(m.get('response/aborted_ratio',-1))!=0 for m in metrics):
        raise RuntimeError('Capacity calibration lacks three normal updated-policy steps')
    import statistics
    steps=[]
    for step in (1,2,3):
        rows=[json.loads(line) for line in (epoch/f'rollout_data/{step}.jsonl').read_text().splitlines() if line.strip()]
        lengths=list(map(int,(epoch/f'rollout_length/length_{step}.txt').read_text().split()))
        if len(rows)!=512 or len(lengths)!=512 or not any(r['output'].strip() for r in rows):
            raise RuntimeError('Capacity diagnostic output cardinality/empty answers invalid')
        if not any(0<n<16384 for n in lengths) or not any(float(r['score'])>0 for r in rows):
            raise RuntimeError('No positive/terminating quality evidence in capacity diagnostic')
        steps.append(dict(step=step,reward=statistics.mean(float(r['score']) for r in rows),
            mean_response_mask_length=statistics.mean(lengths),responses=512))
    peaks={};frames=0;previous={};max_gap_ns=0
    for line in (arm/'hbm_samples.jsonl').read_text().splitlines():
        frame=json.loads(line);frames+=1
        for device in frame['devices']:
            key=f"{device['card']}:{device['chip']}"
            peaks[key]=max(peaks.get(key,0),device['used_bytes'])
            if key in previous:max_gap_ns=max(max_gap_ns,device['time_ns']-previous[key])
            previous[key]=device['time_ns']
    if frames<2 or len(peaks)!=16:raise RuntimeError('Physical HBM observer coverage incomplete')
    report=dict(status='N2_F8_349952_WARM_CAPACITY_VALIDATED',actual_full_rank_step_audits=48,
        target_floors=[2,8,8],actual_kv_tokens=[131072,349952,349952],actor_lr=1e-6,
        preemptions=sum(a['preemptions'] for a in audits),quality=steps,
        sampled_peak_GiB_by_card_chip={k:v/1024**3 for k,v in peaks.items()},
        sampler_frames=frames,max_sample_gap_ms=max_gap_ns/1e6,
        benign_tbe_shutdown_traces=len(benign_traces),original_log_sha256=sha(logs[0]),
        limitation='Sampled highwaters may miss transients. Three sequential RL steps validate this workload, not a universal maximum.',
        only_capacity_change='Natural-F2 floor8:288256 to349952. F2/F4/F16 and other families unchanged.')
    write(root/'CAPACITY_GATE_PASS.json',report)
    return report


def calibration(root):
    env,command=calibration_launch(root);arm=root/'capacity_gate';arm.mkdir()
    idle(root)
    sampler_env=dict(os.environ,HBM_DIAG_ROOT=str(arm),HBM_DIAG_PHASE_MEMORY_LOG='0')
    with (arm/'sampler.log').open('x') as output:
        watcher=subprocess.Popen([sys.executable,str(MEMORY),'sample'],env=sampler_env,stdout=output,stderr=subprocess.STDOUT)
        try:
            deadline=time.monotonic()+30
            while not (arm/'SAMPLER_READY').exists():
                if watcher.poll() is not None or time.monotonic()>deadline:raise RuntimeError('HBM sampler not ready')
                time.sleep(0.2)
            launch(root,'CAPACITY_LOW2_THEN_WARM_F8',arm,MAIN,env,command)
        finally:
            (arm/'STOP_SAMPLING').touch()
            watcher.wait(timeout=20)
    if watcher.returncode:raise RuntimeError('HBM observer failed')
    audit_calibration(root)


def occupancy_for(suite,arm,output):
    from rank_occupancy import parse_epoch
    epochs=[]
    for row in arm['epochs']:
        epoch=Path(row['epoch_dir']);caps=suite.tailguard_caps(epoch);work={}
        for step in range(1,6):
            lengths=list(map(int,(epoch/f'rollout_length/length_{step}.txt').read_text().split()))
            work[step]=suite.validate_response_lengths(lengths,caps[step],True)[0]
        epochs.append(parse_epoch(epoch,work,'FULL'))
    write(output,dict(epochs=epochs))
    return epochs


def math_arm(root,seed):
    suite=main_suite();base=root/'primary';arm=base/f'seed_{seed}/natural_f2'
    env=new_capacity(suite.arm_env(base,seed,'natural_f2'));command=suite.arm_command(seed,'natural_f2')
    storage=split_prepare(root,'math_seed'+str(seed));run=arm/'fig6'
    launch(root,f'MATH_N2_SEED{seed}',arm,MAIN,env,command,run_dir=run,storage=storage)
    result=suite.parse_arm(base,seed,'natural_f2')
    old=suite.parse_arm(OLD,seed,'natural_f2')
    baseline=suite.parse_arm(OLD,seed,'vanilla')
    occupancy_for(suite,result,arm/'RANK_OCCUPANCY.json')
    write(arm/'ARM_PASS.json',result)
    write(arm/'REVIEW.json',dict(status='NEW_BUDGET_SEED_AUDITED',new=result,old_budget=old,
        unchanged_full16_reference=baseline,source_unchanged=True,
        review='Both epochs,normal actor updates,actual FULL and native sleep audited. '
               'Record reward/length/preemption changes; do not require a particular actual shrink floor.'))
    split_cleanup(root,'math_seed'+str(seed),storage,run)


def code_arm(root):
    import run_code_natural_f2 as code
    code.CAPS=CAPS
    original=code.build_training
    def build(destination):
        env,command=original(destination)
        return new_capacity(env),command
    code.build_training=build
    destination=Path('/workspace/adafloor_graph_results')/(root.name+'_code_n2_seed401')
    if destination.exists():raise RuntimeError('Refusing existing new-budget code arm')
    state(root,'CPU_PLANNING_NEW_BUDGET_CODE_N2')
    code.prepare(destination)
    # Existing code judge/sandbox and full2560-request pair checks unchanged.
    state(root,'RUNNING_NEW_BUDGET_CODE_N2')
    code.run(destination)
    write(root/'CODE_N2_RESULT.json',json.loads((destination/'CODE_N2_PAIR_RESULT.json').read_text()))


def sidecar_arm(root):
    helper=module('n2_f8_sidecar',SIDECAR/'tools/run_v014_sidecar_training.py')
    helper.BASELINE=root/'primary'
    side=root/'graph_sidecar'
    env,command=helper.build_launch(side,'graph',60.0,'gt');env=new_capacity(env)
    prior=json.loads((OLD_SIDE/'v014_launch.json').read_text())
    proof_path=Path(prior['sidecar_parity_proof_path']);proof=json.loads(proof_path.read_text())
    if (sha(proof_path)!=prior['sidecar_parity_proof_sha256'] or proof.get('status')!='PASS' or
            proof.get('sidecar_script_sha256')!=sha(SIDECAR/'internal/run_elastic_sidecar_infer.sh') or
            proof.get('aclgraph_sha256')!=sha(SIDECAR/'vllm_ascend/compilation/acl_graph.py') or
            proof.get('model_runner_sha256')!=sha(SIDECAR/'vllm_ascend/worker/model_runner_v1.py')):
        raise RuntimeError('Original graph Sidecar parity proof stale')
    side.mkdir();write(side/'SIDECAR_GREEDY_PARITY_PROOF.json',proof)
    write(side/'v014_launch.json',dict(command=command,sidecar_execution='graph',seed=0,
        policy='natural_f2',epochs=[1,2],actor_frozen=False,
        env={k:v for k,v in env.items() if k.startswith(('VLLM_','VERL_','FAIR_','ADAFLOOR_','DYNAMIC_'))},
        sidecar_parity_proof_path=str(proof_path),sidecar_parity_proof_sha256=sha(proof_path),
        main_baseline=str(root/'primary')))
    storage=split_prepare(root,'graph_sidecar');run=side/'seed_0/natural_f2/fig6'
    endpoint=root/'heldout_checkpoint/global_step_5'
    launch(root,'GRAPH_SIDECAR_N2_SEED0',side,SIDECAR,env,command,run_dir=run,storage=storage,retain_endpoint=endpoint)
    helper.summarize(side)
    split_cleanup(root,'graph_sidecar',storage,run)


def endpoint_quality(root):
    import eager_controls as quality
    endpoint=root/'heldout_checkpoint/global_step_5';destination=root/'heldout256'
    if not (endpoint/'RETENTION_RECEIPT.json').is_file():raise RuntimeError('Missing committed epoch1 endpoint')
    env,command=quality.build_evaluation(endpoint,destination)
    launch(root,'N2_ENDPOINT_HELDOUT256',destination,quality.SOURCE,env,command)
    result=quality.parse_evaluation(destination,endpoint)
    write(destination/'RESULT.json',result)
    # Remove only this campaign's verified hardlinked private endpoint.
    receipt=json.loads((endpoint/'RETENTION_RECEIPT.json').read_text())
    if receipt['endpoint']!=str(endpoint) or not endpoint.is_relative_to(root):
        raise RuntimeError('Endpoint cleanup ownership mismatch')
    shutil.rmtree(endpoint)
    write(root/'ENDPOINT_CLEANUP.json',dict(validated=True,private_endpoint_only=str(endpoint),
          common_epoch0_untouched=True))


def protocol(root):
    suites={name:module('freeze_'+name,runtime_root(name)/'tools/run_mlsys_fig6.py') for name in
        ('qwen3','qwen3_mlsys_supplement_v014','qwen3_graph_sidecar_v014','qwen3_mlsys_eager_controls_v014')}
    old=json.loads((OLD/'protocol.json').read_text())
    if suites['qwen3'].execution_digest()!=old['execution_sha256']:
        raise RuntimeError('Main runtime no longer matches existing Full16 references')
    return dict(status='PRESPECIFIED_NEW_N2_F8_CAPACITY_CAMPAIGN',root=str(root),
        controller_sha256=sha(__file__),execution_sha256={k:v.execution_digest() for k,v in suites.items()},
        changed={'natural_f2_floor8':{'old':288256,'candidate':349952}},
        unchanged={'natural_f2_floor2':131072,'natural_f2_floor4':280576,'full16':380800,
                   'natural_f4_floor8':349952,'planned_f4':133120,'planned_f8':235264},
        capacity_gate='Three consecutive real-updated math steps,forced targets2/8/8,full-length,K8,all16-rank FULL audits. '
                      'F8 must really allocate349952,with physical HBM management sampling. No checkpointer saves.',
        jobs=['capacity_gate','math_n2_seed0_epochs1_2','math_n2_seed2_epochs1_2',
              'math_n2_seed3_epochs1_2','code_n2_seed401_frozen_epoch1',
              'graph_sidecar_n2_seed0_epochs1_2_gate_gt60','sidecar_epoch1_endpoint_heldout256'],
        main_seed_policy='Same0,2,3,one seed at a time; audit seed0 and each later seed before advancing.',
        dependencies='Epoch2 resumes its own epoch1 checkpoint and history. Code uses original code common epoch0. '
                     'No old-budget N2 result merged. Existing Full16/N4/P4 controls retained unchanged.',
        timing='Capacity diagnostic excluded from formal performance. Record offline planning and primary rollout separately.',
        storage='65GiB on EACH of /data,/workspace for two-epoch jobs,split durable epoch checkpoints. '
                'Never delete public epoch0. Private checkpoints removed only after original validation.',
        quality='Same fixed256 greedy endpoint questions,same eager serving protocol. Code same160 sample0,original sandbox and wire repair.',
        stop_policy='On OOM,quality/runtime corruption or source mismatch,stop whole queue with artifacts retained. '
                    'No smaller budgets,eager fallback,extra seeds or favorable-response selection.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('dry-run','run','resume'))
    parser.add_argument('--root',required=True,type=Path)
    args=parser.parse_args();root=args.root.resolve()
    if args.action=='dry-run':
        print(json.dumps(protocol(root),indent=2));return
    if args.action=='run':
        root.mkdir(parents=True,exist_ok=False)
        write(root/'PROTOCOL.json',protocol(root))
    else:
        check_sources(root)
        write(root/'RESUME_REQUEST.json',dict(utc=datetime.now(timezone.utc).isoformat(),
            original_state=json.loads((root/'STATE.json').read_text()),
            controller_sha256=sha(__file__),generation_retry_authorized=False,
            scope='Re-audit completed capacity child and skip only fully audited completed arms. No repeated generation.'))
    try:
        if (root/'capacity_gate/DRIVER_RESULT.json').is_file():
            if json.loads((root/'capacity_gate/DRIVER_RESULT.json').read_text())['exit_code']!=0:
                raise RuntimeError('Original capacity child failed; no automatic retry')
            audit_calibration(root)
        else:calibration(root)
        for seed in SEEDS:
            receipt=root/f'primary/seed_{seed}/natural_f2/ARM_PASS.json'
            if receipt.is_file():
                if main_suite().parse_arm(root/'primary',seed,'natural_f2')!=json.loads(receipt.read_text()):
                    raise RuntimeError('Existing arm receipt failed revalidation')
            else:math_arm(root,seed)
        code_result=root/'CODE_N2_RESULT.json'
        if not code_result.is_file():code_arm(root)
        if not (root/'graph_sidecar/TRAINING_SIDECAR_RESULT.json').is_file():sidecar_arm(root)
        if not (root/'heldout256/RESULT.json').is_file():endpoint_quality(root)
        state(root,'ALL_AFFECTED_N2_EXPERIMENTS_COMPLETE')
    except Exception as error:
        state(root,'STOPPED_WITH_ERROR',error=repr(error));raise


if __name__=='__main__':main()
