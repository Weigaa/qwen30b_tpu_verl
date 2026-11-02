#!/usr/bin/env python3
"""One Ada-N2 seed0 training arm; preserve the Figure6 primary protocol."""
import argparse
from datetime import datetime, timezone
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import tarfile
import time

REPO=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('sidecar_training_main',REPO/'tools/run_mlsys_fig6.py')
suite=importlib.util.module_from_spec(spec);spec.loader.exec_module(suite)
BASELINE=Path('/data/adafloor_graph_results/mlsys_fig6_v014_graph_calibrated_20261017T134200Z')


def build_launch(root, sidecar_execution='eager', window_minimum=60.0, window_comparison='gt'):
    if sidecar_execution not in ('eager','graph'):
        raise ValueError(sidecar_execution)
    if not math.isfinite(window_minimum) or window_minimum<=0 or window_comparison not in ('gt','ge'):
        raise ValueError('Invalid Sidecar admission threshold/comparison')
    root=Path(root).resolve()
    env=suite.arm_env(root,0,'natural_f2')
    # No frozen actor, no paired request seeds, no disabled TailGuard, and no
    # fixed-Planned topology: these remain the published main-arm settings.
    settings=dict(
        FAIR_START_EPOCH=1,FAIR_TOTAL_EPOCHS=3,FAIR_FREEZE_ACTOR=0,
        VERL_PAIRED_REQUEST_SAMPLING_SEEDS=0,
        DYNAMIC_DISABLE_TAIL_GUARD=0,DYNAMIC_SHORT_STEP_CAP_ENABLE=1,DYNAMIC_EXPECT_NO_RESPONSE_CAPS=0,
        ADAFLOOR_GRAPH_SIDECAR_ENABLE=1,VERL_SIDECAR_ENABLE=1,
        VLLM_ASCEND_SHRINK_AWARE_TARGET_POLICY='natural',
        VERL_SIDECAR_WINDOW_GATE_ENABLE=1,VERL_SIDECAR_WINDOW_GATE_MIN_SECONDS=window_minimum,
        VERL_SIDECAR_WINDOW_GATE_STRICT_MINIMUM=int(window_comparison=='gt'),
        VERL_SIDECAR_WINDOW_GATE_CALIBRATION='prior_step_q95',
        VERL_SIDECAR_START_ONCE=0,VERL_SIDECAR_REARM_AFTER_GATE_SKIP=1,
        VERL_SIDECAR_DIRECT_STEP_EVENTS=1,
        VERL_SIDECAR_REQUIRE_ACTIVE_LEASE_BEFORE_RESTORE=1,
        VERL_SIDECAR_REQUIRE_SHRINK_QUORUM=1,VERL_SIDECAR_SHRINK_QUORUM_SIZE=16,
        VERL_SIDECAR_WORLD_SIZE=16,VERL_SIDECAR_EXPECTED_ACTIVE_RANKS=8,
        VERL_SIDECAR_MODEL_PATH='/data/Qwen2.5-1.5B-Instruct',
        VERL_SIDECAR_PROMPTS_FILE='/data/gsm8k',VERL_SIDECAR_DATA_SPLIT='train',
        VERL_SIDECAR_PARALLEL_MODE='dp',VERL_SIDECAR_TENSOR_PARALLEL_SIZE=1,
        VERL_SIDECAR_DATA_PARALLEL_SIZE=1,VERL_SIDECAR_REPLICA_COUNT=8,
        VERL_SIDECAR_ENABLE_EXPERT_PARALLEL=0,VERL_SIDECAR_TASK_QUEUE_ENABLE=2,
        VERL_SIDECAR_N=1,VERL_SIDECAR_TEMPERATURE=0.0,VERL_SIDECAR_TOP_P=1.0,
        VERL_SIDECAR_GPU_MEMORY_UTILIZATION=0.80,VERL_SIDECAR_MAX_MODEL_LEN=6144,
        VERL_SIDECAR_MAX_TOKENS=4096,VERL_SIDECAR_MAX_NUM_SEQS=128,
        VERL_SIDECAR_MAX_NUM_BATCHED_TOKENS=65536,VERL_SIDECAR_MAX_PROMPTS=1024,
        VERL_SIDECAR_MAX_PROMPTS_PER_REPLICA=128,VERL_SIDECAR_GENERATE_CHUNK_SIZE=128,
        VERL_SIDECAR_REPEAT_UNTIL_KILLED=1,VERL_SIDECAR_STREAM_CHECKPOINT=1,
        VERL_SIDECAR_RESET_OUTPUT_ON_START=0,VERL_SIDECAR_ENGINE_MULTIPROCESSING=1,
        VERL_SIDECAR_SINGLETON_GROUP_REUSE=0,VERL_SIDECAR_STOP_ACK_TIMEOUT_SECONDS=60,
        # Disjoint communications for the primary and secondary engines.
        VERL_MEGATRON_EAGER_WEIGHT_SYNC_GROUP_INIT=0,HCCL_IF_BASE_PORT=12000,
        VERL_HCCL_IF_BASE_PORT_START=12000,VERL_HCCL_IF_BASE_PORT_BLOCK=16384,
        GIT_TRACK_RUN_LOGS=0)
    env.update({k:str(v) for k,v in settings.items()})
    env.update(VERL_SIDECAR_ENFORCE_EAGER='0' if sidecar_execution=='graph' else '1',
               VERL_SIDECAR_TASK_QUEUE_ENABLE='1' if sidecar_execution=='graph' else '2',
               VERL_SIDECAR_GRAPH_AUDIT='1' if sidecar_execution=='graph' else '0',
               VERL_SIDECAR_CUDAGRAPH_MODE='FULL_DECODE_ONLY',
               VERL_SIDECAR_CUDAGRAPH_CAPTURE_SIZES='[1,2,4,8,16,32,64,128]')
    command=suite.arm_command(0,'natural_f2')+[
        'actor_rollout_ref.rollout.shrink_aware.target_policy=natural']
    return env,command


def jsonl(path):
    with path.open() as stream:
        for line in stream:
            if line.strip():yield json.loads(line)


def prepare_split_checkpoint_storage(root, storage):
    """Put only this run's epoch2 checkpoint on the other local filesystem.

    Epoch1 remains durable and available to resume epoch2. No training state or
    rollout input changes. Original fair-runner validation still precedes cleanup.
    """
    root=Path(root).resolve();storage=Path(storage).resolve()
    approved=Path('/workspace/adafloor_graph_results').resolve()
    if storage.parent!=approved or storage.name!=root.name+'_epoch2_checkpoints':
        raise RuntimeError('Split storage must be a new run-owned /workspace checkpoint directory')
    approved.mkdir(parents=True,exist_ok=True)
    for path in (root,approved):
        if shutil.disk_usage(path).free<65*1024**3:
            raise RuntimeError(f'Need65GiB free on each checkpoint filesystem: {path}')
    storage.mkdir(exist_ok=False)
    link=root/'seed_0/natural_f2/fig6/epoch_002_mode1_natural/checkpoints'
    manifest=dict(run_root=str(root),epoch2_checkpoint_link=str(link),storage=str(storage),
        common_epoch0=str(suite.COMMON),reason='Separate durable disks; never delete public or old checkpoints')
    suite.write_json(storage/'OWNED_BY_SIDECAR_RUN.json',manifest)
    suite.write_json(root/'SPLIT_CHECKPOINT_STORAGE.json',manifest)
    return manifest


def cleanup_split_checkpoint_storage(root,manifest):
    root=Path(root).resolve();storage=Path(manifest['storage'])
    saved=json.loads((storage/'OWNED_BY_SIDECAR_RUN.json').read_text())
    if (saved!=manifest or saved['run_root']!=str(root)
            or storage.parent!=Path('/workspace/adafloor_graph_results').resolve()
            or storage.name!=root.name+'_epoch2_checkpoints'
            or Path(manifest['epoch2_checkpoint_link']).is_symlink()
            or not (root/'seed_0/natural_f2/fig6/CHECKPOINTS_REMOVED_AFTER_VALIDATION.txt').is_file()
            or not (suite.COMMON/'DO_NOT_DELETE_COMMON_EPOCH0_CHECKPOINT').is_file()):
        raise RuntimeError('Refusing split storage cleanup without complete run-ownership/validation proof')
    shutil.rmtree(storage)
    suite.write_json(root/'SPLIT_CHECKPOINT_STORAGE_CLEANUP.json',dict(
        **manifest,removed_only_owned_storage=True,common_epoch0_preserved=True))


def sidecar_summary(epoch, sidecar_execution='eager', window_minimum=80.0, window_strict=False):
    side=epoch/'sidecar';text=(side/'lease.log').read_text()
    primary='\n'.join(p.read_text(errors='replace') for p in (epoch/'logs').glob('*.txt'))
    restore_times=list(map(float,re.findall(r'driver_restore_rpc_start_time=([0-9.]+)',primary)))
    if len(restore_times)!=5:raise RuntimeError('Missing primary restore boundaries')
    def events(key):
        return [(float(m[1]),dict(re.findall(r'(\w+)=(\S+)',m[2]))) for m in
                re.finditer(r'^'+re.escape(key)+r'=([0-9.]+)([^\n]*)$',text,re.M)]
    decisions=sorted((json.loads(p.read_text()) for p in side.glob('window_gate.lease*.json')),
                     key=lambda x:x['step'])
    if [x['step'] for x in decisions]!=[1,2,3,4,5]:raise RuntimeError('Missing five Sidecar decisions')
    if any(d['minimum_window_seconds']!=window_minimum for d in decisions):
        raise RuntimeError('Actual gate threshold differs from launch configuration')
    starts=events('sidecar_start_time');numbers=list(map(int,re.findall(r'^sidecar_lease_index=(\d+)$',text,re.M)))
    if len(starts)!=len(numbers) or len(starts)!=sum(x['action']=='start' for x in decisions):
        raise RuntimeError('Actual starts disagree with decisions')
    windows=[]
    for (start,_),number in zip(starts,numbers):
        decision=next(d for d in decisions if d['lease_id']==number)
        estimate=decision['predicted_window_seconds']
        if (decision.get('minimum_comparison','>=')!=('>' if window_strict else '>=')
                or not math.isfinite(estimate)
                or (estimate<=window_minimum if window_strict else estimate<window_minimum)):
            raise RuntimeError('Sidecar start violates configured admission threshold/comparison')
        find=lambda key:next(v for v in events(key) if v[1].get('lease_id')==str(number))
        stop,request=find('trainer_stop_request_time');end,exit_fields=find('sidecar_exit_confirmed_time')
        ack,ack_fields=find('watcher_restore_ack_time')
        if not (start<stop<=end<=ack<restore_times[decision['step']-1] and exit_fields.get('process_group_alive')=='0'
                and request['request_id']==exit_fields['request_id']==ack_fields['request_id']):
            raise RuntimeError('Unsafe Sidecar restore boundary')
        windows.append(dict(step=decision['step'],lease_id=number,start=start,stop=stop,exit=end,
            window_seconds=stop-start,shutdown_to_exit_seconds=end-stop,shutdown_to_ack_seconds=ack-stop,
            completed=0,completed_before_stop=0,tokens=0,correct=0))
    states={}
    state_files=list((side/'state').glob('completed.shard*.jsonl'))
    state_files+=list((side/'state/completed_epochs').glob('*.jsonl'))
    for p in state_files:
        for row in jsonl(p):
            key=(row.get('sidecar_epoch',0),row['prompt_id'])
            if key in states:raise RuntimeError('Duplicate completion state')
            states[key]=float(row['time'])
    scorer_spec=importlib.util.spec_from_file_location('sidecar_training_scores',REPO/'tools/verify_qwen3_sidecar_pairs.py')
    scorer=importlib.util.module_from_spec(scorer_spec);scorer_spec.loader.exec_module(scorer)
    infer=(side/'infer.log').read_text(errors='replace') if (side/'infer.log').exists() else ''
    if scorer._has_oom(infer):raise RuntimeError('Sidecar OOM; primary exit0 alone is not success')
    engine_events=[]
    for line in scorer._strip_ansi(infer).splitlines():
        start=line.find('{')
        if start<0:continue
        try:row=json.loads(line[start:])
        except json.JSONDecodeError:continue
        if row.get('event') in ('sidecar_load_start','sidecar_model_ready',
                'sidecar_aclgraph_capture','sidecar_aclgraph_replay'):
            engine_events.append(row)
    for w in windows:
        records=[r for r in engine_events if w['start']<=r['time']<=w['exit']]
        loads=[r for r in records if r['event']=='sidecar_load_start']
        expected_eager=sidecar_execution=='eager'
        for r in loads:
            if r['engine_kwargs']['enforce_eager']!=expected_eager:
                raise RuntimeError('Sidecar actual engine mode disagrees with protocol')
            if not expected_eager and (r['task_queue_enable']!='1' or
                    r['engine_kwargs']['compilation_config']['cudagraph_mode']!='FULL_DECODE_ONLY'):
                raise RuntimeError('Sidecar is not FULL_DECODE_ONLY/TQ1')
        w['ready_shards']=sorted({r['shard_index'] for r in records if r['event']=='sidecar_model_ready'})
        ready_times=[r['time'] for r in records if r['event']=='sidecar_model_ready']
        w['first_ready_seconds']=min(ready_times)-w['start'] if ready_times else None
        w['all_ready_seconds']=max(ready_times)-w['start'] if len(w['ready_shards'])==8 else None
        w['graph_replay_shards']=sorted({r['shard_index'] for r in records
            if r['event']=='sidecar_aclgraph_replay' and r['runtime_mode']=='FULL'
            and r.get('capture_phase') is False})
        w['graph_capture_events']=sum(r['event']=='sidecar_aclgraph_capture' for r in records)
    frame=scorer._dataset_frame(Path('/data/gsm8k/train.parquet'),{}) if windows else None
    used=set();skipped=0;resumed=0
    if (side/'outputs.jsonl').exists():
        for row in jsonl(side/'outputs.jsonl'):
            key=(row.get('sidecar_epoch',0),row['prompt_id'])
            if key in used or key not in states:raise RuntimeError('Duplicate/untracked Sidecar output')
            used.add(key);answer=row['outputs'][0]
            if row.get('sidecar_status') or 'skipped' in str(answer.get('finish_reason')):skipped+=1;continue
            if answer.get('finish_reason') not in ('stop','length'):raise RuntimeError('Nonfinal completion')
            instant=states[key];w=next(w for w in windows if w['start']<=instant<=w['exit'])
            source,index=row['prompt_source'].rsplit(':',1)
            if Path(source).resolve()!=Path('/data/gsm8k/train.parquet').resolve():raise RuntimeError('Wrong Sidecar data')
            expected=frame.iloc[int(index)]
            if row['prompt']!=scorer._prompt_to_text(expected['prompt']):raise RuntimeError('Sidecar prompt mismatch')
            correct=scorer.gsm8k.compute_score(answer['text'],str(expected['reward_model']['ground_truth']),method='strict')==1
            w['completed']+=1;w['completed_before_stop']+=int(instant<=w['stop'])
            w['tokens']+=answer['token_ids_len'];w['correct']+=int(correct)
            if sidecar_execution=='graph' and row['shard_index'] not in w['graph_replay_shards']:
                raise RuntimeError('Graph Sidecar output without actual FULL graph replay evidence')
            resumed+=int(answer.get('resume_prefix_text_len',0)>0)
    if used!=set(states):raise RuntimeError('Completed states/output mismatch')
    if len(events('watcher_restore_ack_time'))!=5:raise RuntimeError('Missing five restore ACKs')
    for p in (side/'state').glob('inflight.shard*.json'):
        if json.loads(p.read_text()).get('prompt_ids'):raise RuntimeError('Uncheckpointed inflight work')
    partials=set()
    for p in (side/'state').glob('resume.shard*.jsonl'):
        for row in jsonl(p):
            key=(row.get('sidecar_epoch',0),row['prompt_id'])
            if key in partials or key in used:raise RuntimeError('Duplicate/complete+partial work')
            partials.add(key)
    for w in windows:w['strict_accuracy']=w['correct']/w['completed'] if w['completed'] else None
    return dict(decisions=decisions,windows=windows,skipped_queries=skipped,
        completed_queries=sum(w['completed'] for w in windows),
        completed_before_stop=sum(w['completed_before_stop'] for w in windows),
        partial_queries=len(partials),resumed_completions=resumed,
        sigterm_events=len(events('sidecar_kill_time')),sigkill_events=len(events('sidecar_force_kill_time')))


def summarize(root):
    launch=json.loads((root/'v014_launch.json').read_text())
    if suite.execution_digest()!=launch['execution_sha256']:raise RuntimeError('Execution sources changed')
    primary=suite.parse_arm(root,0,'natural_f2')
    execution=launch.get('sidecar_execution','eager')
    minimum=float(launch['env']['VERL_SIDECAR_WINDOW_GATE_MIN_SECONDS'])
    strict=launch['env'].get('VERL_SIDECAR_WINDOW_GATE_STRICT_MINIMUM','0')=='1'
    sidecars=[sidecar_summary(Path(e['epoch_dir']),execution,minimum,strict) for e in primary['epochs']]
    baseline=suite.parse_arm(BASELINE,0,'natural_f2')
    report=dict(status='PASS',seed=0,policy='natural_f2',epochs=[1,2],actor_frozen=False,
        sidecar_execution=execution,
        window_gate_minimum_seconds=minimum,window_gate_comparison='>' if strict else '>=',
        baseline_comparison_scope='Existing same-seed no-Sidecar updated-policy reference; not frozen causal ON/OFF',
        primary=primary,sidecar=sidecars,baseline=baseline,
        actual_sidecar_starts=sum(len(x['windows']) for x in sidecars),
        completed_before_stop=sum(x['completed_before_stop'] for x in sidecars),
        proves_useful_sidecar_coexecution=any(x['completed_before_stop']>0 for x in sidecars),
        expected_floor_is_not_required_actual_floor=True,
        common_epoch0_preserved=suite.COMMON.is_dir())
    suite.write_json(root/'TRAINING_SIDECAR_RESULT.json',report)
    completed=sum(w['completed'] for s in sidecars for w in s['windows'])
    correct=sum(w['correct'] for s in sidecars for w in s['windows'])
    lines=['# Ada-N2 seed0: two updated training epochs with '+execution+' Sidecar','',
        'Common epoch0 → own epoch1 → own epoch2; five steps each. Primary is native '
        f"FULL_DECODE_ONLY/TQ1/sleep2/lifecycle1. Sidecar admission requires predicted window {'>' if strict else '>='}{minimum:g}s; "
        'all starts, partial work, drain, graph replay and pre-restore process exits are audited.', '',
        '| Epoch | Rollout s | Response tok/s | Reward | Sidecar starts | Completed jobs |',
        '| --- | ---: | ---: | ---: | ---: | ---: |']
    for number,(p,s) in enumerate(zip(primary['epochs'],sidecars),1):
        lines.append(f"| {number} | {p['rollout_s']:.2f} | {p['response_tokens_s']:.2f} | "
                     f"{p['reward']:.6f} | {len(s['windows'])} | {s['completed_queries']} |")
    lines+=['',f"Completed before stop: {report['completed_before_stop']}; including drain: {completed}. "
            f"Strict GSM8K correct: {correct}/{completed}. Partials excluded; repeated jobs are not globally unique questions.",
            '', '| Epoch/step | Lease s | All8 ready s | Shutdown-to-exit s | FULL replay replicas |',
            '| --- | ---: | ---: | ---: | ---: |']
    for number,s in enumerate(sidecars,1):
        for w in s['windows']:
            ready=f"{w['all_ready_seconds']:.2f}" if w['all_ready_seconds'] is not None else 'not all ready'
            lines.append(f"| {number}/{w['step']} | {w['window_seconds']:.2f} | {ready} | "
                         f"{w['shutdown_to_exit_seconds']:.2f} | {len(w['graph_replay_shards'])}/8 |")
    lines+=['',report['baseline_comparison_scope']+'. Existing OFF epoch rollout seconds: '+
            ', '.join(f"{p['rollout_s']:.2f}" for p in baseline['epochs'])+'.',
            '', 'Full per-step primary results, sidecar decisions and lifecycle evidence: `TRAINING_SIDECAR_RESULT.json`.']
    (root/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)
    return report


def main():
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=('run','dry-run','summarize'))
    parser.add_argument('--sidecar-execution',choices=('eager','graph'),default='eager')
    parser.add_argument('--sidecar-parity-proof',type=Path)
    parser.add_argument('--epoch2-checkpoint-root',type=Path)
    parser.add_argument('--window-minimum',type=float,default=60.0)
    parser.add_argument('--window-comparison',choices=('gt','ge'),default='gt')
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();root=args.output.resolve()
    if args.mode=='summarize':summarize(root);return
    root.mkdir(parents=True,exist_ok=True)
    env,command=build_launch(root,args.sidecar_execution,args.window_minimum,args.window_comparison)
    for key in ('XDG_CACHE_HOME','TORCHAIR_CACHE_HOME','TRITON_CACHE_DIR','ASCEND_CACHE_PATH','ASCEND_WORK_PATH','RAY_TMPDIR'):
        Path(env[key]).mkdir(parents=True,exist_ok=True)
    launch=dict(command=command,execution_sha256=suite.execution_digest(),seed=0,policy='natural_f2',
        sidecar_execution=args.sidecar_execution,
        epochs=[1,2],actor_frozen=False,main_baseline=str(BASELINE),
        env={k:v for k,v in env.items() if k.startswith(('VLLM_','VERL_','FAIR_','ADAFLOOR_','DYNAMIC_'))
             or k in ('TASK_QUEUE_ENABLE','HCCL_BUFFSIZE','PYTORCH_NPU_ALLOC_CONF','HCCL_IF_BASE_PORT')})
    if args.mode=='dry-run':
        suite.write_json(root/'launch.preview.json',launch);env['FAIR_DRY_RUN']='1'
        raise SystemExit(subprocess.run(command,env=env,cwd=REPO).returncode)
    if (root/'v014_launch.json').exists():raise RuntimeError('Refusing to overwrite launch evidence')
    if args.sidecar_execution=='graph':
        if args.sidecar_parity_proof is None:
            raise RuntimeError('Graph Sidecar requires the minimal eager/graph greedy preflight proof')
        proof=json.loads(args.sidecar_parity_proof.read_text())
        same_input=proof.get('same_input_shadow') or {}
        exact_same_input=(proof.get('same_padded_input_graph_parity_exact') is True
            and same_input.get('status')=='ALL_SAME_PADDED_INPUT_SHADOW_CHECKS_PASSED'
            and same_input.get('max_hidden_abs')==0.0 and same_input.get('max_logits_abs')==0.0
            and same_input.get('post_checks',0)>0
            and same_input.get('post_checks')==same_input.get('pre_checks')
            and same_input.get('post_mismatches')==0 and same_input.get('pre_argmax_mismatches')==0
            and same_input.get('diagnostic_generation_unchanged') is True)
        if (proof.get('status')!='PASS' or proof.get('checked_prompts')!=16
                or not (proof.get('greedy_token_ids_exact') is True or exact_same_input)
                or proof.get('sidecar_script_sha256')!=suite.digest(REPO/'internal/run_elastic_sidecar_infer.sh')
                or proof.get('aclgraph_sha256')!=suite.digest(REPO/'vllm_ascend/compilation/acl_graph.py')
                or proof.get('model_runner_sha256')!=suite.digest(REPO/'vllm_ascend/worker/model_runner_v1.py')):
            raise RuntimeError('Missing/stale/failed graph Sidecar greedy parity proof')
        suite.write_json(root/'SIDECAR_GREEDY_PARITY_PROOF.json',proof)
        launch['sidecar_parity_proof_path']=str(args.sidecar_parity_proof.resolve())
        launch['sidecar_parity_proof_sha256']=suite.digest(args.sidecar_parity_proof)
    protocol=f'''# Ada-N2 seed0: two updated-policy epochs, {args.sidecar_execution} Sidecar

Primary: protected common math epoch0/global_step5; actual LR1e-6 updates in
epochs1/2, five steps each,32×n16,320 primary prompt occurrences/5120 responses.
Epoch2 resumes this run's own epoch1 actor and response history. TailGuard and
short-step cap ON, max response16384, prompt1024; rollout seed0, actor/data seed42;
paired per-request seeds OFF. FULL_DECODE_ONLY/TQ1/native sleep2/lifecycle1/KV-first,
K8 minimum predicted-padding buckets; Natural floor2, stages8/4/2. Calibrated KV
F2/F4/F8/F16=131072/280576/288256/380800 tokens. Actual floor4 instead of2 is valid.

Sidecar: Qwen2.5-1.5B-Instruct/GSM8K train; eight TP1 replicas, execution
{args.sidecar_execution}; {'native FULL_DECODE_ONLY/TQ1, prefill eager, explicit sizes[1,2,4,8,16,32,64,128], no FX/Inductor/GraphEX' if args.sidecar_execution=='graph' else 'eager/TQ2'}.
Greedy n1/max4096, model6144, max_num_seqs128, max_batched_tokens65536,
memory fraction0.80. Model/data/capacities/sampling unchanged from old eager Sidecar.
Optional FX fusion passes are disabled for the Sidecar graph: the installed
910_93 OPP lacks AddRmsNormBias used in Qwen2 norm/bias pattern registration.
This does not disable ACLGraph or remove Attention from the decode graph.
Graph startup includes all capture costs; graph capture/replay events are emitted
once per shape/worker, not inferred from flags. This is the existing independent
vLLM LLM path, not the RL inference path. Sidecar has no AdaFloor topology,
automatic shrink, RL weight sync or native-sleep lifecycle; only the primary
Ada-N2 shrinks, lending eight detached devices to eight fixed TP1 Sidecars.
Preflight uses16 identical queries. If free-running greedy trajectories differ
under dynamic eager versus padded graph batching, require exact hidden states,
logits and argmax on EVERY same-input decode step before/after replay, and
unchanged graph-generated token IDs with diagnostics ON/OFF. No tolerance is
relaxed and no eager outputs replace graph outputs. The raw trajectory differences
remain explicitly reported. This is not a universal parity proof.

Admission: actual16-rank quorum at16→8; only actual detached devices. Causal
prior_step_q95 gate: predicted window {'>' if args.window_comparison=='gt' else '>='}{args.window_minimum:g} seconds.
At exactly the threshold the {'Sidecar is skipped' if args.window_comparison=='gt' else 'Sidecar may start'}.
No future-response lookahead, no deliberately selected
long windows. Partial requests checkpointed across windows within each epoch;
fresh Sidecar queue per primary epoch. Counts are jobs, not unique questions.
All processes must exit and matching ACK precede primary restore. Startup,
shutdown/drain and useful completions are reported separately. No-window or
startup-consuming-window outcomes are recorded, not silently replaced.

Only this one new two-epoch ON arm is scheduled. Existing same-seed no-Sidecar
main run is a descriptive OFF reference, not a frozen paired causal-overhead
experiment. No new OFF arm or extra seeds. Old eager-Sidecar evidence remains
untouched. Primary qwen3 and paper TeX are not edited. Runtime sources archived
before launch; public epoch0 protected; only this run's validated private
checkpoints cleaned by the original fair runner after their final use.
'''
    (root/'RUN_PROTOCOL.md').write_text(protocol)
    split_storage=None
    if args.epoch2_checkpoint_root:
        split_storage=prepare_split_checkpoint_storage(root,args.epoch2_checkpoint_root)
        launch['split_checkpoint_storage']=split_storage
    suite.write_json(root/'v014_launch.json',launch)
    with tarfile.open(root/'execution_sources_complete.tar.gz','w:gz') as archive:
        files=list(REPO.glob('*.sh'))
        for directory in ('tools','internal','verl','vllm','vllm_ascend','megatron'):
            files.extend(p for p in (REPO/directory).rglob('*') if p.is_file() and p.suffix in ('.py','.sh','.yaml'))
        for p in sorted(set(files)):archive.add(p,arcname=str(p.relative_to(REPO)))
    child=subprocess.Popen(command,env=env,cwd=REPO)
    if split_storage:
        # The fair runner must first pass its no-overwrite guard. Creating
        # epoch2 in advance would make a fresh experiment look pre-existing.
        first_epoch=root/'seed_0/natural_f2/fig6/epoch_001_mode1_natural'
        while child.poll() is None and not first_epoch.is_dir():
            time.sleep(0.5)
        if child.poll() is None:
            link=Path(split_storage['epoch2_checkpoint_link'])
            link.parent.mkdir(parents=True,exist_ok=True)
            link.symlink_to(Path(split_storage['storage']),target_is_directory=True)
            print('[Sidecar checkpoint storage] epoch2 durable split-disk link installed after fair preflight',flush=True)
    rc=child.wait()
    suite.write_json(root/'DRIVER_RESULT.json',dict(exit_code=rc,end_time_utc=datetime.now(timezone.utc).isoformat()))
    if rc:raise SystemExit(rc)
    try:
        summarize(root)
        if split_storage:cleanup_split_checkpoint_storage(root,split_storage)
    except Exception as error:
        suite.write_json(root/'FINAL_VALIDATION_ERROR.json',dict(primary_exit_code=0,error=repr(error)))
        raise


if __name__=='__main__':main()
