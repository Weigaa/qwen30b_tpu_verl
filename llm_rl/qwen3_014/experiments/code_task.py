#!/usr/bin/env python3
"""One additional domain: frozen Eurus-code, Vanilla versus final Planned-F4.

Generation uses a labelled dummy reward, never host execution of model code.
Quality is measured separately on prespecified sample_index0 in OS sandboxes.
"""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re

CODE = Path(__file__).resolve().parent
ROOT = Path('/data/adafloor_graph_results/mlsys_v014_supplement_20261023T180000Z')
SOURCE = CODE.parent / 'qwen3_main'
COMMON = Path('/data/adafloor_shared_state/common_epoch0_eurus2_code_validation_frozen_gpu09_kv380800_permanent')
DATA = Path('/data/eurus2_rl_code/validation_code_paired_160')
TRAIN = ROOT/'code_prepared/train.parquet'
REPAIR_TRACE = ROOT/'code_plan_preview_cached32/planner.log'


def suite():
    spec = importlib.util.spec_from_file_location('code_domain_main', SOURCE / 'tools/run_mlsys_fig6.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    """EXPLICITLY NOT AN ACCURACY SCORE. Actor is frozen and scored offline."""
    return 0.0


def build_training(policy):
    if policy not in ('vanilla', 'planned_f4'):
        raise ValueError('Only one additional-task baseline/final-method pair')
    module = suite()
    env = module.arm_env(ROOT / 'code_task', 401, policy)
    env.update(COMMON_EPOCH0_ROOT=str(COMMON), TRAIN_FILE=str(TRAIN),
               TRAIN_FILE_ORIG=str(TRAIN), TEST_FILE=str(DATA / 'test.parquet'),
               FAIR_DATASET_FRACTION='1.0',DYNAMIC_DATASET_FRACTION='1.0',DATASET_FRACTION_FOR_ORACLE='1.0',
               MAX_CROSS_STEP_REPAIR_SWAPS='32',
               FAIR_CODE_CHAT_TOKENIZER='/data/Qwen3-30B-A3B',
               FAIR_START_EPOCH='1', FAIR_TOTAL_EPOCHS='2', FAIR_FREEZE_ACTOR='1',
               FAIR_KEEP_COMPLETED_CHECKPOINTS='0', VERL_SKIP_FINAL_ACTOR_UPDATE_WHEN_FROZEN='1',
               VERL_PAIRED_REQUEST_SAMPLING_SEEDS='1', VERL_SIDECAR_ENABLE='0')
    env['FAIR_CODE_REPAIR_TRACE']=str(REPAIR_TRACE)
    if REPAIR_TRACE.exists():env['FAIR_CODE_REPAIR_TRACE_SHA256']=hashlib.sha256(REPAIR_TRACE.read_bytes()).hexdigest()
    env['PATH']=str(CODE/'offline_bin')+os.pathsep+env['PATH']
    command = module.arm_command(401, policy) + ['actor_rollout_ref.actor.optim.lr=0.0',
        f'custom_reward_function.path={CODE / "code_task.py"}',
        'custom_reward_function.name=compute_score', 'reward_model.reward_manager=naive']
    return env, command


def read_rows(path):
    with path.open() as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def audit_arm(policy):
    module = suite()
    arm = ROOT / f'code_task/seed_401/{policy}/fig6'
    receipt = arm / 'CHECKPOINTS_REMOVED_AFTER_VALIDATION.txt'
    if not receipt.is_file() or 'validated_epochs=001' not in receipt.read_text():
        raise RuntimeError('Code-task fair transaction not committed')
    paths = list(arm.glob('epoch_001_*'))
    if len(paths) != 1:
        raise RuntimeError('Missing unique code-task epoch')
    epoch = paths[0]
    text = '\n'.join(p.read_text(errors='replace') for p in (epoch / 'logs').glob('*.txt'))
    if not re.search(r'\[run\] end_time=\S+ exit_code=0', text):
        raise RuntimeError('Code-task child did not exit0')
    actor=COMMON/'epoch_000_mode0_probe/checkpoints/qwen3moe_for_eagle3/global_step_5/actor'
    loaded=re.findall(r'\[Rank (\d+)\] Loaded sharded model checkpoint from '+re.escape(str(actor)),text)
    if len(loaded)!=16 or set(map(int,loaded))!=set(range(16)):
        raise RuntimeError('Code-task common actor not proved loaded on all16 ranks')
    if f"using customized reward function 'compute_score' from '{CODE / 'code_task.py'}'" not in text:
        raise RuntimeError('Safe offline-only reward path not proved')
    audits = module.json_records(epoch, 'MODE1_STEP_RUNTIME_AUDIT ')
    if (len(audits) != 80 or {(a['step'], a['rank']) for a in audits} !=
            {(s, r) for s in range(1, 6) for r in range(16)} or
            any(not a['complete'] or a.get('full_iterations', 0) <= 0 for a in audits)):
        raise RuntimeError('Missing actual FULL decode audits')
    metrics = [dict(re.findall(r'([\w/]+):([^ ]+)', line)) for line in text.splitlines()
               if re.search(r'step:\d+\s+-', line)]
    if len(metrics) != 5 or any(float(m.get('actor/lr', 0)) != 0 or
                               float(m.get('response/aborted_ratio', -1)) != 0 for m in metrics):
        raise RuntimeError('Not five frozen, nonaborted generations')
    caps = module.tailguard_caps(epoch) if policy == 'planned_f4' else {s: 16384 for s in range(1, 6)}
    work = 0; steps = []; outputs = {}; selected = {}
    for step in range(1, 6):
        rows = list(read_rows(epoch / f'rollout_data/{step}.jsonl'))
        if len(rows) != 512:
            raise RuntimeError('Missing512 code answers')
        for row in rows:
            key = (row['rollout_prompt_hash'], int(row['rollout_sample_index']), int(row['rollout_request_seed']))
            if key in outputs:
                raise RuntimeError('Duplicate request identity')
            outputs[key] = row
            if key[1] == 0:
                if key[0] in selected:
                    raise RuntimeError('Duplicate prespecified sample0')
                selected[key[0]] = row
        lengths = list(map(int, (epoch / f'rollout_length/length_{step}.txt').read_text().split()))
        count, _ = module.validate_response_lengths(lengths, caps[step], True)
        seconds = float(metrics[step-1]['rollout/rollout_output_time_s'])
        work += count; steps.append(dict(step=step, seconds=seconds, response_tokens=count,
                                        response_tokens_s=count/seconds))
    if len(outputs) != 2560 or len(selected) != 160:
        raise RuntimeError('Code-task cardinality changed')
    elapsed = sum(row['seconds'] for row in steps)
    report = dict(policy=policy, actual_full_audits=len(audits),
                  preemptions=sum(a['preemptions'] for a in audits), steps=steps,
                  rollout_seconds=elapsed, response_tokens=work, response_tokens_s=work/elapsed,
                  actor_frozen=True, reward_in_primary_logs_is_dummy_not_accuracy=True)
    return report, outputs, selected


def score_pair():
    # Import the sandbox wrapper, not the old unsandboxed default reward.
    import concurrent.futures
    from sandbox_code import run_case
    arms = {}; selected = {}; outputs = {}
    for policy in ('vanilla', 'planned_f4'):
        arms[policy], outputs[policy], selected[policy] = audit_arm(policy)
    if set(outputs['vanilla']) != set(outputs['planned_f4']):
        raise RuntimeError('Frozen paired request identities/seeds differ')
    for key in outputs['vanilla']:
        a, b = outputs['vanilla'][key], outputs['planned_f4'][key]
        if a['input'] != b['input'] or a['gts'] != b['gts']:
            raise RuntimeError('Actual paired input/test-suite changed')
    result_dir = ROOT / 'code_task/quality'; result_dir.mkdir(parents=True, exist_ok=True)
    for policy in ('vanilla', 'planned_f4'):
        destination = result_dir / f'{policy}.jsonl'
        if destination.exists():
            raise RuntimeError('Refusing repeat quality evaluation')
        entries = sorted(selected[policy].items())
        def grade(entry):
            key, row = entry
            tests = json.loads(row['gts']) if isinstance(row['gts'], str) else row['gts']
            result = run_case(row['output'], tests)
            return dict(prompt_hash=key, sample_index=0,
                        output_sha256=hashlib.sha256(row['output'].encode()).hexdigest(), **result)
        # CPU only, small bounded concurrency. No model-generated code executes
        # in the worker threads; every answer enters the restricted child.
        with destination.open('x') as stream, concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            scored = []
            for result in pool.map(grade, entries):
                stream.write(json.dumps(result)+'\n'); stream.flush(); scored.append(result)
        arms[policy]['sandbox_sampled_pass_at_1'] = sum(row['passed'] for row in scored)/160
        arms[policy]['scored_answers'] = 160
        arms[policy]['sandbox_failures'] = sum(row['exit_code'] != 0 or row['timed_out'] for row in scored)
    return dict(status='V014_CODE_DOMAIN_FROZEN_PAIR_COMPLETE', arms=arms, seed=401,
                questions=160, responses_per_question=16, paired_requests=2560,
                scope='One frozen additional-domain rollout epoch, not RL convergence. Timed rollout has dummy reward; quality is separately OS-sandboxed sampled pass@1 for sample_index0 on all160 paired prompts. All tests per problem, 1GiB address space,64MiB tmpfs,8s CPU,20s wall per answer. No retries, no favorable-answer selection.')
