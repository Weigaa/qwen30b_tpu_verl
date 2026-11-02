#!/usr/bin/env python3
"""One prespecified v0.14 Natural-F2 code arm, reusing the audited FULL baseline.

No runtime edits, new seed, prompt filtering, generation retries or judge changes.
The offline plan is frozen before NPU execution. Original experiment is read-only.
"""
import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

import code_task as old
import prepare_code_plan
from code_test_wire import normalize, infrastructure_format_error
from rank_occupancy import parse_epoch
from sandbox_code import run_case

POLICY = 'natural_f2'
FILES = ('length_sorted_train.parquet', 'length_sorted_rank_plan.json',
         'length_sorted_rank_plan_summary.json', 'length_sorted_length_oracle.json')
CAPS = '2:131072,4:280576,8:288256,16:380800'


def bundle_metadata(hashes, **values):
    return dict(values,files=hashes,seed=401,prompts=160,steps=5,
        batch_size=32,rollout_n=16,
        bundle_id=hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest())


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    temp.replace(path)


def state(root, value, **extra):
    write(root/'STATE.json', dict(state=value, utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), **extra))
    print(value, extra, flush=True)


def build_training(root):
    suite = old.suite()
    env = suite.arm_env(root, 401, POLICY)
    env.update(COMMON_EPOCH0_ROOT=str(old.COMMON), TRAIN_FILE=str(old.TRAIN),
        TRAIN_FILE_ORIG=str(old.TRAIN), TEST_FILE=str(old.DATA/'test.parquet'),
        FAIR_DATASET_FRACTION='1.0', DYNAMIC_DATASET_FRACTION='1.0', DATASET_FRACTION_FOR_ORACLE='1.0',
        FAIR_CODE_CHAT_TOKENIZER='/data/Qwen3-30B-A3B', MAX_CROSS_STEP_REPAIR_SWAPS='32',
        FAIR_START_EPOCH='1', FAIR_TOTAL_EPOCHS='2', FAIR_FREEZE_ACTOR='1',
        FAIR_KEEP_COMPLETED_CHECKPOINTS='0', VERL_SKIP_FINAL_ACTOR_UPDATE_WHEN_FROZEN='1',
        VERL_PAIRED_REQUEST_SAMPLING_SEEDS='1', VERL_SIDECAR_ENABLE='0',
        MIN_ADAPTIVE_FLOOR='2', RANK_MATCHING_POLICY='release_area', FLOOR_KV_CAPS=CAPS,
        ADAFLOOR_FROZEN_INPUTS_DIR=str(root/'frozen_inputs'),
        ADAFLOOR_FROZEN_INPUT_TOOL=str(old.SOURCE/'tools/frozen_graph_inputs.py'))
    # Only offline helpers are placed on PATH. Timed workers use the unchanged interpreter/runtime.
    env['PATH'] = str(old.CODE/'offline_bin') + os.pathsep + env['PATH']
    # Never replay P4's repaired groups/floors as if they were a Natural plan.
    env.pop('FAIR_CODE_REPAIR_TRACE', None)
    env.pop('FAIR_CODE_REPAIR_TRACE_SHA256', None)
    command = suite.arm_command(401, POLICY) + ['actor_rollout_ref.actor.optim.lr=0.0',
        f'custom_reward_function.path={old.CODE/"code_task.py"}',
        'custom_reward_function.name=compute_score', 'reward_model.reward_manager=naive']
    return env, command


def planner_command(output):
    command = prepare_code_plan.command(output)
    command[:2] = [sys.executable, '-u', str(old.CODE/'optimized_planner.py'),
                   str(old.SOURCE/'tools/build_mode1_length_sorted_e2e_plan.py')]
    command[command.index('--min-adaptive-floor')+1] = '2'
    command[command.index('--floor-kv-caps')+1] = CAPS
    return command


def baseline_contract():
    suite = old.suite()
    protocol_path = old.ROOT/'code_task/PROTOCOL.json'
    result_path = old.ROOT/'code_task/CODE_PAIR_RESULT.json'
    declared = json.loads(protocol_path.read_text())
    result = json.loads(result_path.read_text())
    source = suite.execution_digest()
    if source != declared['source_sha256'] or source != result['source_sha256']:
        raise RuntimeError('Baseline runtime differs; refusing historical baseline reuse')
    for key in ('train_parquet', 'test_parquet'):
        manifest = declared['dataset_manifest']
        if sha(manifest[key]) != manifest[key+'_sha256']:
            raise RuntimeError('Original dataset changed')
    if sha(old.TRAIN) != declared['canonical_dataset_manifest']['sha256']:
        raise RuntimeError('Canonical 160 code inputs changed')
    if not (old.COMMON/'DO_NOT_DELETE_COMMON_EPOCH0_CHECKPOINT').is_file():
        raise RuntimeError('Protected common epoch0 absent')
    # Also pin helpers that affect generation/judging, not only model code.
    for name, wanted in declared['scoring_sha256'].items():
        if sha(old.CODE/name) != wanted:
            raise RuntimeError('Pinned original helper changed: '+name)
    repair = json.loads((old.ROOT/'code_task/GRADING_REPAIR_PROTOCOL.json').read_text())
    if sha(old.CODE/'code_test_wire.py') != repair['new_adapter_sha256']:
        raise RuntimeError('Repaired judge wire adapter changed')
    audit, outputs, selected = old.audit_arm('vanilla')
    scores_path = old.ROOT/'code_task/quality/vanilla.jsonl'
    scores = list(old.read_rows(scores_path))
    if len(scores) != 160 or {r['prompt_hash'] for r in scores} != set(selected):
        raise RuntimeError('Baseline score coverage changed')
    for row in scores:
        if row['sample_index'] != 0 or row['output_sha256'] != hashlib.sha256(
                selected[row['prompt_hash']]['output'].encode()).hexdigest():
            raise RuntimeError('Baseline prespecified answer/score identity mismatch')
    if sum(r['passed'] for r in scores) != 71:
        raise RuntimeError('Historical repaired baseline no longer matches71/160')
    return dict(source_sha256=source, protocol_sha256=sha(protocol_path),
        result_sha256=sha(result_path), baseline_scores_sha256=sha(scores_path),
        common=str(old.COMMON), train_sha256=sha(old.TRAIN),
        history_sha256=sha(old.COMMON/'epoch_000_mode0_probe/offline_planning_history.json'),
        reused_baseline=audit, baseline_correct=71), outputs


def prepare(root):
    if (root/'PROTOCOL.json').exists():
        raise RuntimeError('Refusing a second prepare over an existing protocol')
    root.mkdir(parents=True, exist_ok=True)
    state(root, 'AUDITING_BASELINE_REUSE')
    baseline, _ = baseline_contract()
    protocol = dict(status='PRESPECIFIED_NOT_COMPLETED', policy=POLICY, seed=401,
        actor_frozen=True, epochs=[1], steps=5, prompts=160, rollout_n=16,
        max_response_length=16384, task_queue=1, graph='FULL_DECODE_ONLY',
        native_sleep_level=2, lifecycle=1, kv_first=True, floor_kv_caps=CAPS,
        min_adaptive_floor=2, actual_target_policy='natural', rank_matching_policy='release_area',
        offline_repair_budget=32, offline_exact_peak_memoization=True,
        tailguard='Unchanged Natural-F2 history-derived caps and short-step policy, no forced floor.',
        baseline=baseline, driver_sha256=sha(__file__),
        exact_planner_optimization_sha256=sha(old.CODE/'optimized_planner.py'),
        code_wire_sha256=sha(old.CODE/'code_test_wire.py'),
        quality='Sample_index0 on all160 prompts, one OS-sandboxed grade each, repaired lossless wire adapter. '
                'All tests,1GiB AS,64MiB tmpfs,8s CPU,20s wall. No retries or best-of16.',
        scope='Same frozen checkpoint, ordered input set and request seeds as existing FULL baseline. '
              'Natural strategy may reorder prompts/ranks and select history-dependent floors/caps/buckets. '
              'Floor2 eligibility does not require actual shrink to2. No timed runtime changes.')
    write(root/'PROTOCOL.json', protocol)
    preview = root/'plan_preview'
    preview.mkdir()
    command = planner_command(preview)
    env = os.environ.copy()
    for key in ('FAIR_CODE_REPAIR_TRACE', 'FAIR_CODE_REPAIR_TRACE_SHA256'):
        env.pop(key, None)
    write(preview/'LAUNCH.json', dict(command=command, protocol_sha256=sha(root/'PROTOCOL.json')))
    state(root, 'CPU_PLANNING_ALL160_NATURAL_F2')
    with (preview/'planner.log').open('x') as log:
        rc = subprocess.run(command, cwd=old.SOURCE, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
    if rc:
        raise RuntimeError(f'Natural planner failed rc={rc}; no NPU run launched')
    summary = json.loads((preview/FILES[2]).read_text())
    if (len(summary) != 5 or [r['step'] for r in summary] != list(range(1,6)) or
            any(not r['feasible'] or r['max_adjusted_rank_peak_tokens'] > r['kv_cap'] for r in summary)):
        raise RuntimeError('Incomplete or KV-unsafe Natural plan')
    import pyarrow.parquet as pq
    rows = pq.read_table(preview/FILES[0]).to_pylist()
    original = pq.read_table(old.TRAIN).to_pylist()
    if (len(rows) != 160 or sorted(r['extra_info']['index'] for r in rows) != list(range(160)) or
            any(r != original[r['extra_info']['index']] for r in rows)):
        raise RuntimeError('Natural planner changed/dropped/repeated original prompts')
    plans = json.loads((preview/FILES[1]).read_text())
    positions = []
    for step, plan in enumerate(plans, 1):
        if plan['step'] != step or len(plan['aclgraph_capture_sizes']) != 8:
            raise RuntimeError('Not five ordered K8 plans')
        for rank in range(16):
            assigned = plan['rank_to_dataset_item_idx'][str(rank)]
            if [rows[int(i)]['extra_info']['index'] for i in assigned] != plan['rank_to_source_idx'][str(rank)]:
                raise RuntimeError('Prompt-to-rank mismatch')
            positions.extend(assigned)
    if sorted(positions) != list(range(160)):
        raise RuntimeError('Plan does not assign each prompt exactly once')
    frozen = root/'frozen_inputs'
    frozen.mkdir()
    hashes = {}
    for name in FILES:
        shutil.copyfile(preview/name, frozen/name)
        hashes[name] = sha(frozen/name)
        (frozen/name).chmod(0o444)
    write(frozen/'manifest.json', bundle_metadata(hashes,
        common_epoch0=str(old.COMMON), workload='frozen code policy, not actor-updated RL',
        source=str(preview), policy=POLICY, input_sha256=sha(old.TRAIN)))
    env, command = build_training(root)
    env['FAIR_DRY_RUN'] = '1'
    with (root/'dry_run.log').open('x') as log:
        subprocess.run(command, cwd=old.SOURCE, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    state(root, 'PREPARED_CPU_AND_DRY_RUN_PASSED',
        selected_floors=[r['selected_floor'] for r in summary],
        caps=[r['tail_guard_response_cap'] for r in summary])


def reuse_plan(root, source, *, defer_repair_audit=False):
    """New execution transaction, same sealed four plan files, no search."""
    if root.exists():
        raise RuntimeError('Refusing existing reused-plan transaction')
    baseline,_=baseline_contract()
    original=json.loads((source/'PROTOCOL.json').read_text())
    if (baseline!=original['baseline'] or original['exact_planner_optimization_sha256']!=
            sha(old.CODE/'optimized_planner.py')):
        raise RuntimeError('Original plan baseline/optimizer changed')
    metadata=json.loads((source/'frozen_inputs/manifest.json').read_text())
    for name in FILES:
        if sha(source/'frozen_inputs'/name)!=metadata['files'][name]:
            raise RuntimeError('Original frozen plan file changed')
    summary=json.loads((source/'frozen_inputs'/FILES[2]).read_text())
    if len(summary)!=5 or any(not r['feasible'] or
            r['max_adjusted_rank_peak_tokens']>r['kv_cap'] for r in summary):
        raise RuntimeError('Reused plan not KV feasible')
    root.mkdir(parents=True)
    frozen=root/'frozen_inputs';frozen.mkdir()
    for name in FILES:
        shutil.copyfile(source/'frozen_inputs'/name,frozen/name)
        (frozen/name).chmod(0o444)
    write(frozen/'manifest.json',bundle_metadata(metadata['files'],
        common_epoch0=str(old.COMMON),workload=metadata['workload'],
        source=str(source/'frozen_inputs'),policy=POLICY,input_sha256=sha(old.TRAIN)))
    original.update(driver_sha256=sha(__file__),reused_plan_from=str(source),
        original_plan_protocol_sha256=sha(source/'PROTOCOL.json'),
        manifest_repair='Added required rollout_n,batch_size,bundle_id only. Four hashed plan files unchanged. '
                        'Prior execution failed before inference with KeyError rollout_n; logs retained.')
    write(root/'PROTOCOL.json',original)
    env,command=build_training(root);env['FAIR_DRY_RUN']='1'
    with (root/'dry_run.log').open('x') as log:
        subprocess.run(command,cwd=old.SOURCE,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    if defer_repair_audit:
        write(root/'EXECUTION_PRIORITY_AMENDMENT.json',dict(
            original_protocol_sha256=sha(root/'PROTOCOL.json'),original_driver_sha256=sha(__file__),
            updated_driver_sha256=sha(__file__),optimizer_sha256=sha(old.CODE/'optimized_planner.py'),
            frozen_manifest_sha256=sha(frozen/'manifest.json'),repair_audit_status='DEFERRED_NOT_PASSED',
            reason='Explicit user request: run the full code-N2 test first using completed plan. '
                   'Additional fresh P4 repair parity deferred, not passed. Other audits remain mandatory.'))
    state(root,'PREPARED_CPU_AND_DRY_RUN_PASSED',reused_plan=True,search_repeated=False,
          selected_floors=[r['selected_floor'] for r in summary])


def audit_natural(root):
    suite = old.suite()
    run = root/'seed_401/natural_f2/fig6'
    receipt = run/'CHECKPOINTS_REMOVED_AFTER_VALIDATION.txt'
    if not receipt.is_file() or 'validated_epochs=001' not in receipt.read_text():
        raise RuntimeError('Natural fair transaction not committed')
    epoch = run/'epoch_001_mode1_natural'
    logs = list((epoch/'logs').glob('*.txt'))
    if len(logs) != 1:
        raise RuntimeError('Not one Natural trainer log')
    text = logs[0].read_text(errors='replace')
    actor = old.COMMON/'epoch_000_mode0_probe/checkpoints/qwen3moe_for_eagle3/global_step_5/actor'
    loaded = re.findall(r'\[Rank (\d+)\] Loaded sharded model checkpoint from '+re.escape(str(actor)), text)
    if len(loaded) != 16 or set(map(int,loaded)) != set(range(16)):
        raise RuntimeError('Common actor not proved loaded on all16 ranks')
    if (not re.search(r'\[run\] end_time=\S+ exit_code=0', text) or
            not re.search(r'Initializing a V1 LLM engine \(v0\.14', text) or
            'target_policy=natural' not in text or
            re.search(r'OutOfMemoryError|Memory_Allocation_Failure|NPU out of memory', text)):
        raise RuntimeError('Natural actual v0.14 successful execution not proved')
    audits = suite.json_records(epoch, 'MODE1_STEP_RUNTIME_AUDIT ')
    if (len(audits) != 80 or {(a['step'],a['rank']) for a in audits} !=
            {(s,r) for s in range(1,6) for r in range(16)} or
            any(not a['complete'] or a.get('full_iterations',0) <= 0 for a in audits)):
        raise RuntimeError('Missing actual Natural FULL replay coverage')
    metrics = [dict(re.findall(r'([\w/]+):([^ ]+)',line)) for line in text.splitlines()
               if re.search(r'step:\d+\s+-', line)]
    if len(metrics) != 5 or any(float(m.get('actor/lr',-1)) != 0 or
            float(m.get('response/aborted_ratio',-1)) != 0 for m in metrics):
        raise RuntimeError('Not five frozen, nonaborted Natural steps')
    caps = suite.tailguard_caps(epoch)
    outputs = {}; selected = {}; steps = []
    for step in range(1,6):
        rows = list(old.read_rows(epoch/f'rollout_data/{step}.jsonl'))
        if len(rows) != 512:
            raise RuntimeError('Missing512 Natural answers')
        for row in rows:
            key = (row['rollout_prompt_hash'],int(row['rollout_sample_index']),int(row['rollout_request_seed']))
            if key in outputs:
                raise RuntimeError('Repeated Natural request identity')
            outputs[key] = row
            if key[1] == 0:
                if key[0] in selected:
                    raise RuntimeError('Repeated sample0')
                selected[key[0]] = row
        lengths = list(map(int,(epoch/f'rollout_length/length_{step}.txt').read_text().split()))
        work, _ = suite.validate_response_lengths(lengths, caps[step], True)
        seconds = float(metrics[step-1]['rollout/rollout_output_time_s'])
        steps.append(dict(step=step, seconds=seconds, response_tokens=work,
                          response_tokens_s=work/seconds, response_cap=caps[step]))
    if len(outputs) != 2560 or len(selected) != 160:
        raise RuntimeError('Natural cardinality changed')
    W = sum(s['response_tokens'] for s in steps)
    T = sum(s['seconds'] for s in steps)
    occupancy = parse_epoch(epoch, {s['step']:s['response_tokens'] for s in steps}, 'FULL')
    write(root/'NATURAL_OCCUPANCY.json', occupancy)
    return dict(policy=POLICY, actual_full_audits=80, steps=steps,
        preemptions=sum(a['preemptions'] for a in audits), response_tokens=W,
        rollout_seconds=T, response_tokens_s=W/T, actor_frozen=True,
        occupied_rank_s_bounds=occupancy['occupied_rank_s_bounds'],
        E_response_tokens_per_occupied_rank_s_bounds=occupancy['E_response_tokens_per_occupied_rank_s_bounds'],
        actual_shrink_rpc_count=len(re.findall(r'Shrink RPC result:',text))), outputs, selected


def score(root):
    baseline, reference = baseline_contract()
    report, outputs, selected = audit_natural(root)
    if set(reference) != set(outputs):
        raise RuntimeError('Baseline/Natural requests or sampling seeds differ')
    for key in reference:
        if (reference[key]['input'],reference[key]['gts']) != (outputs[key]['input'],outputs[key]['gts']):
            raise RuntimeError('Actual paired input/test suite differs')
    destination = root/'natural_f2_quality.jsonl'
    if destination.exists():
        raise RuntimeError('No repeated quality grading allowed')
    def grade(entry):
        key,row = entry
        tests = json.loads(row['gts']) if isinstance(row['gts'],str) else row['gts']
        wire, changed = normalize(tests)
        result = run_case(row['output'],wire)
        if infrastructure_format_error(result):
            raise RuntimeError('Judge wire-format failure is not model inaccuracy')
        return dict(prompt_hash=key, sample_index=0,
            output_sha256=hashlib.sha256(row['output'].encode()).hexdigest(),
            lossless_wire_normalization=changed, **result)
    state(root, 'GRADING160_SAMPLE0_IN_OS_SANDBOXES')
    scored = []
    with destination.open('x') as stream, concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        for row in pool.map(grade,sorted(selected.items())):
            stream.write(json.dumps(row)+'\n'); stream.flush(); scored.append(row)
    report.update(correct=sum(r['passed'] for r in scored),scored_answers=len(scored),
        sandbox_sampled_pass_at_1=sum(r['passed'] for r in scored)/160,
        sandbox_failures=sum(r['exit_code'] != 0 or r['timed_out'] for r in scored))
    base = baseline['reused_baseline']
    base_epoch = old.ROOT/'code_task/seed_401/vanilla/fig6/epoch_001_mode0_vanilla'
    occupancy = parse_epoch(base_epoch,{s['step']:s['response_tokens'] for s in base['steps']},'FULL')
    write(root/'BASELINE_OCCUPANCY.json',occupancy)
    base.update(correct=71,sandbox_sampled_pass_at_1=71/160,
        occupied_rank_s_bounds=occupancy['occupied_rank_s_bounds'],
        E_response_tokens_per_occupied_rank_s_bounds=occupancy['E_response_tokens_per_occupied_rank_s_bounds'])
    bE=base['E_response_tokens_per_occupied_rank_s_bounds']; nE=report['E_response_tokens_per_occupied_rank_s_bounds']
    write(root/'CODE_N2_PAIR_RESULT.json',dict(status='V014_NATURAL_F2_CODE_PAIR_COMPLETE',
        baseline_reused=True,baseline=base,natural_f2=report,paired_requests=2560,
        Q_gain_pct=100*(report['response_tokens_s']/base['response_tokens_s']-1),
        rollout_time_reduction_pct=100*(1-report['rollout_seconds']/base['rollout_seconds']),
        E_gain_pct_bounds=[100*(nE[0]/bE[1]-1),100*(nE[1]/bE[0]-1)],
        baseline_evidence=baseline,scope=json.loads((root/'PROTOCOL.json').read_text())['scope']))
    state(root, 'COMPLETE', correct=report['correct'],response_tokens_s=report['response_tokens_s'])


def snapshot_raw(root, ray_tmp):
    # Preserve actual stdout/stderr before the launcher's normal Ray cleanup.
    destination = root/'raw_taskrunner_logs'
    destination.mkdir(exist_ok=True)
    for source in Path(ray_tmp).glob('ray/session_*/logs/worker-*.out'):
        with source.open('rb') as stream:
            if b':actor_name:TaskRunner\n' not in stream.read(65536):
                continue
        for path in (source,source.with_suffix('.err')):
            if path.exists():
                shutil.copyfile(path,destination/path.name)


def other_trainers():
    """Do not mistake another job's CPU/Ray initialization for idle devices."""
    pids=[]
    for path in Path('/proc').glob('[0-9]*/cmdline'):
        try:
            parts=path.read_bytes().rstrip(b'\0').split(b'\0')
        except (FileNotFoundError,ProcessLookupError,PermissionError):
            continue
        if not parts or not re.fullmatch(rb'python[0-9.]*',parts[0].rsplit(b'/',1)[-1]):
            continue
        if any(a==b'-m' and b==b'verl.trainer.main_ppo' for a,b in zip(parts,parts[1:])):
            pids.append(int(path.parent.name))
    return pids


def run(root, *, defer_repair_audit=False):
    protocol = json.loads((root/'PROTOCOL.json').read_text())
    baseline, _ = baseline_contract()
    validation_path=Path(__file__).with_name('OPTIMIZED_PLANNER_VALIDATION.json')
    validation=json.loads(validation_path.read_text())
    if (validation['status']!='FIVE_STEP_NATURAL_F2_EXACT_PLAN_PARITY_PASS' or
            validation['optimization_sha256']!=sha(old.CODE/'optimized_planner.py')):
        raise RuntimeError('Exact planner validation missing or stale')
    driver_matches=sha(__file__)==protocol['driver_sha256']
    if defer_repair_audit:
        amendment=json.loads((root/'EXECUTION_PRIORITY_AMENDMENT.json').read_text())
        if (amendment['original_protocol_sha256']!=sha(root/'PROTOCOL.json') or
                amendment['original_driver_sha256']!=protocol['driver_sha256'] or
                amendment['updated_driver_sha256']!=sha(__file__) or
                amendment['optimizer_sha256']!=sha(old.CODE/'optimized_planner.py') or
                amendment['frozen_manifest_sha256']!=sha(root/'frozen_inputs/manifest.json') or
                amendment['repair_audit_status']!='DEFERRED_NOT_PASSED'):
            raise RuntimeError('Missing/stale explicit priority amendment')
        driver_matches=True
        state(root,'P4_CPU_REPAIR_AUDIT_EXPLICITLY_DEFERRED_NOT_PASSED')
    else:
        repair_path=Path(__file__).with_name('CODE_PLANNER_REPAIR_VALIDATION.json')
        while not repair_path.exists():
            state(root,'WAITING_FOR_CPU_REPAIR_PARITY_AUDIT');time.sleep(10)
        repair=json.loads(repair_path.read_text())
        if (repair['status']!='FRESH31_SWAP_CODE_P4_GOLD_PLAN_PARITY_PASS' or
                repair['optimization_sha256']!=sha(old.CODE/'optimized_planner.py')):
            raise RuntimeError('Fresh code repair parity missing or stale')
    if baseline != protocol['baseline'] or not driver_matches:
        raise RuntimeError('Prepared source, baseline or driver changed')
    if protocol['exact_planner_optimization_sha256']!=sha(old.CODE/'optimized_planner.py'):
        raise RuntimeError('Prepared optimization changed')
    if not (root/'dry_run.log').is_file() or not (root/'frozen_inputs/manifest.json').is_file():
        raise RuntimeError('Run requires a completed, unused CPU preflight')
    manifest = json.loads((root/'frozen_inputs/manifest.json').read_text())
    for name,wanted in manifest['files'].items():
        if sha(root/'frozen_inputs'/name) != wanted:
            raise RuntimeError('Frozen plan changed')
    env, command = build_training(root)
    arm = root/'seed_401/natural_f2'
    # dry-run can create the parent only, never an epoch/trainer run.
    arm.mkdir(parents=True,exist_ok=True)
    if (arm/'driver.log').exists():
        raise RuntimeError('Refusing a generation rerun over existing evidence')
    for key in ('XDG_CACHE_HOME','TORCHAIR_CACHE_HOME','TRITON_CACHE_DIR',
                'ASCEND_CACHE_PATH','ASCEND_WORK_PATH','RAY_TMPDIR'):
        Path(env[key]).mkdir(parents=True,exist_ok=True)
    if shutil.disk_usage(root).free < 65*1024**3:
        raise RuntimeError('Less than65GiB free; checkpoint capacity guard not lowered')
    while True:
        result = subprocess.run(['npu-smi','info'],capture_output=True,text=True,check=True)
        if 'Process id' not in result.stdout:
            raise RuntimeError('Cannot audit NPU idle state')
        trainers=other_trainers()
        if (not trainers and not re.findall(r'^\|\s+\d+\s+\d+\s+\d+\s+',
                result.stdout.rsplit('Process id',1)[1],re.M)):
            break
        state(root,'WAITING_FOR_IDLE_NPUS',other_trainer_pids=trainers); time.sleep(15)
    write(arm/'LAUNCH.json',dict(command=command,env={k:v for k,v in env.items()
        if k.startswith(('VLLM_','VERL_','ADAFLOOR_','FAIR_','DYNAMIC_')) or
        k in ('COMMON_EPOCH0_ROOT','TRAIN_FILE','TRAIN_FILE_ORIG','TEST_FILE','TASK_QUEUE_ENABLE',
              'MIN_ADAPTIVE_FLOOR','RANK_MATCHING_POLICY','FLOOR_KV_CAPS','PYTORCH_NPU_ALLOC_CONF','HCCL_BUFFSIZE','PATH')}))
    state(root,'RUNNING_NATURAL_F2_FULL_SLEEP2',controller_pid=os.getpid())
    with (arm/'driver.log').open('x') as log:
        process = subprocess.Popen(command,cwd=old.SOURCE,env=env,stdout=log,stderr=subprocess.STDOUT)
        while process.poll() is None:
            snapshot_raw(root,env['RAY_TMPDIR'])
            time.sleep(10)
        snapshot_raw(root,env['RAY_TMPDIR'])
    write(arm/'DRIVER_RESULT.json',dict(exit_code=process.returncode))
    if process.returncode:
        raise RuntimeError(f'Natural child rc={process.returncode}; outputs/checkpoints retained, no automatic retry')
    score(root)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['prepare','run','score','reuse'])
    parser.add_argument('--root',required=True,type=Path)
    parser.add_argument('--plan-source',type=Path)
    parser.add_argument('--defer-repair-audit',action='store_true',
        help='Requires a pinned explicit execution-priority amendment; never claims parity passed.')
    args = parser.parse_args()
    try:
        if args.defer_repair_audit and args.action not in ('run','reuse'):
            raise ValueError('Repair audit deferral applies only to run')
        if args.action=='run':
            run(args.root.resolve(),defer_repair_audit=args.defer_repair_audit)
        elif args.action=='reuse':
            if args.plan_source is None:
                raise ValueError('Reuse requires --plan-source')
            reuse_plan(args.root.resolve(),args.plan_source.resolve(),defer_repair_audit=args.defer_repair_audit)
        else:
            {'prepare':prepare,'score':score}[args.action](args.root.resolve())
    except Exception as error:
        if args.root.exists():
            state(args.root,'STOPPED_WITH_ERROR',error=repr(error))
        raise


if __name__ == '__main__':
    main()
