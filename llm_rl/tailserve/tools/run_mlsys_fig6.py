#!/usr/bin/env python3
"""Rerun HPCA Figure6's updated-policy protocol on the local v0.14 FULL backend."""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import shutil
import statistics as stats
import subprocess
import sys
from datetime import datetime, timezone

REPO = Path(__file__).resolve().parents[1]
COMMON = Path('/data/adafloor_shared_state/common_epoch0_probe_gpu09_kv380800_permanent')
PAPER = REPO / 'analysis_eval/hpca最终版论文-HPCA_AdaFloor__Adaptive_Floor_Planning_and_Rank_Time_Reuse_for_Long_Tail_MoE_RL_Rollouts.pdf'
CAPACITY_EVIDENCE = REPO / 'analysis_eval/MLSYS_FIG6_CAPACITY_20261017.json'
KV_BUDGETS = dict(full16=380800, natural_f2=131072, natural_f4=280576,
                  natural_f8=288256, natural_f4_family_f8=349952,
                  planned_f4=133120, planned_f8=235264)
SEEDS = (0, 2, 3)
POLICIES = ('vanilla', 'lengthsort', 'lengthsort_guard', 'fixed4', 'minskew',
            'natural_f4', 'natural_f2', 'planned_f4')
VARIANTS = dict(natural_f4='adafloor_n_f4', natural_f2='adafloor_n_f2', planned_f4='adafloor_p_f4')
BASES = dict(fixed4='run_baseline_kvsafe_fixed4_epoch1_2.sh',
             minskew='run_baseline_minskew_epoch1_2.sh',
             natural_f4='run_mode1_dynamic_length_aware_adaptive_floor4_natural_full3.sh',
             natural_f2='run_mode1_dynamic_length_aware_adaptive_floor2_natural_tailguard_reuse_epoch0_2epoch.sh',
             planned_f4='run_mode1_dynamic_length_aware_adaptive_floor4_planned_full3.sh')
ANSI = re.compile(r'\x1b\[[0-9;]*[mK]')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def execution_digest(exclude=()):
    files = list(REPO.glob('*.sh'))
    for directory in ('tools', 'internal', 'verl', 'vllm', 'vllm_ascend', 'megatron'):
        files.extend(p for p in (REPO / directory).rglob('*')
                     if p.is_file() and p.suffix in ('.py', '.sh', '.yaml'))
    h = hashlib.sha256()
    for p in sorted(set(files)):
        if str(p.relative_to(REPO)) in exclude:
            continue
        h.update(str(p.relative_to(REPO)).encode() + b'\0' + digest(p).encode() + b'\n')
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    tmp.replace(path)


def protocol():
    return dict(paper=str(PAPER), paper_sha256=digest(PAPER), repo=str(REPO),
                execution_sha256=execution_digest(), common_epoch0=str(COMMON),
                common_history_sha256=digest(COMMON / 'epoch_000_mode0_probe/offline_planning_history.json'),
                common_reuse_env_sha256=digest(COMMON / 'reuse.env'), seeds=list(SEEDS),
                release_parser_sha256=digest(REPO.parent/'qwen3_shrink_aware/analysis_eval/plot_e2e_performance.py'),
                seed_policy='global rollout engine seed; paired per-request seeding OFF',
                policies=list(POLICIES), arms=24, epochs=[1, 2], measured_epochs=48,
                steps_per_epoch=5, measured_steps=240, actor_frozen=False, actor_lr=1e-6,
                epoch2='resume own epoch1 checkpoint; plan from own epoch1 response history',
                batch_size=32, rollout_n=16, prompts_per_epoch=160, max_prompt_length=1024,
                max_response_length=16384, temperature=0.9, top_p=0.9, top_k=50,
                task_queue=1, graph='FULL_DECODE_ONLY', bucket_count=8,
                graph_buckets='minimum predicted padding per step; history only, all methods',
                native_sleep_level=2, lifecycle=1, kv_first=True,
                peak_rule='existing response-peak planner, safety factor1.16; no prompt_adaptive override',
                kv_tokens=dict(KV_BUDGETS),
                graph_reserve_tokens=dict(floor2=0,floor4=0,floor8=27392,floor16=0),
                graph_reserve_note='KV withholding offsets, NOT literal graph-memory consumption; floor8 retains its conservative unexercised-target limit',
                capacity_profile='v014_native_sleep2_full_decode_f4validated_20261017',
                capacity_evidence_path=str(CAPACITY_EVIDENCE),
                capacity_evidence_sha256=digest(CAPACITY_EVIDENCE),
                planned_f2=dict(status='excluded_by_configured_residency_budget',
                                effective_tokens=KV_BUDGETS['natural_f2']-147456,
                                note='HPCA infeasible column; not a ninth timed arm or a new v0.14 OOM measurement'),
                statistics='mean epochs1,2 per seed; mean and sample SD (ddof=1) across3 seeds',
                release_accounting='same HPCA parser: coordinated exit-start to rollout output boundary',
                artifacts='lossless gzip only after two-epoch validation; never compress common epoch0')


def arm_env(root, seed, policy):
    # Do not inherit the preceding frozen cross-version experiment, floor
    # overrides, replay traces, sidecars or another policy's planned groups.
    prefixes = ('ADAFLOOR_', 'VLLM_', 'VERL_', 'DYNAMIC_', 'FAIR_', 'BASELINE_',
                'KV_', 'FLOOR_', 'PHYSICAL_', 'FORCE_SELECTED_', 'GRAPH_MAIN_')
    env = {k: v for k, v in os.environ.items() if not k.startswith(prefixes)}
    for key in ('MIN_ADAPTIVE_FLOOR', 'RANK_MATCHING_POLICY', 'ACTIVE_PEAK_SAFETY_FACTOR',
                'MEGATRON_CI_DISABLE_EXPANDABLE_SEGMENTS'):
        env.pop(key, None)
    arm = root / f'seed_{seed}' / policy
    values = dict(
        COMMON_EPOCH0_ROOT=COMMON, FAIR_OUTPUT_ROOT=arm, FAIR_RUN_NAME='fig6', DYNAMIC_RUN_NAME='fig6',
        FAIR_START_EPOCH=1, FAIR_TOTAL_EPOCHS=3, FAIR_TRAIN_STEPS=5,
        FAIR_TRAIN_BATCH_SIZE=32, FAIR_ROLLOUT_N=16, FAIR_MAX_RESPONSE_LENGTH=16384,
        FAIR_PROMPTS_PER_EPOCH=160, FAIR_FREEZE_ACTOR=0, FAIR_KEEP_COMPLETED_CHECKPOINTS=0,
        VERL_PAIRED_REQUEST_SAMPLING_SEEDS=0, VLLM_ROLLOUT_DIVERSIFY_SAMPLING_SEED=0,
        TASK_QUEUE_ENABLE=1, VLLM_ROLLOUT_TASK_QUEUE_ENABLE=1, HCCL_BUFFSIZE=800,
        PYTORCH_NPU_ALLOC_CONF='expandable_segments:False',
        VLLM_ROLLOUT_NATIVE_SLEEP_MODE=1, VLLM_ROLLOUT_SLEEP_LEVEL=2,
        VLLM_ROLLOUT_NATIVE_SLEEP_GRAPH_LIFECYCLE=1, VLLM_ROLLOUT_NATIVE_SLEEP_GRAPH_RETAIN=0,
        VLLM_ROLLOUT_NATIVE_SLEEP_GRAPH_KV_FIRST=1, VLLM_ROLLOUT_CAMEM_WEIGHT_RELOAD=1,
        VLLM_ROLLOUT_MANUAL_FREE_CACHE_ENGINE=0, VLLM_ASCEND_NATIVE_SLEEP_PARK_IDLE=1,
        VLLM_ROLLOUT_INVALIDATE_ACLGRAPH_AFTER_WEIGHT_UPDATE=1,
        VLLM_ROLLOUT_REUSE_ACLGRAPH_AFTER_WEIGHT_UPDATE=0,
        VLLM_ENABLE_GRAPH_MODE=0, ADAFLOOR_ACLGRAPH_MODE='FULL_DECODE_ONLY',
        ADAFLOOR_GRAPH_CAPTURE_SIZES='[32]', VLLM_ASCEND_ACLGRAPH_BUCKET_COUNT=8,
        ROLLOUT_ENFORCE_EAGER='False', VLLM_ASCEND_ELASTIC_ACLGRAPH=1,
        VLLM_ASCEND_ELASTIC_ACLGRAPH_DYNAMIC_DISABLE=0,
        VLLM_ASCEND_ELASTIC_ACLGRAPH_CAPTURE_ATTENTION=1, VLLM_ASCEND_ELASTIC_ACLGRAPH_CAPTURE_MOE=1,
        VLLM_ASCEND_ALLOW_LAZY_ACLGRAPH_CAPTURE=0, VLLM_ASCEND_ZIYI_GRAPH_DP_SYNC=1,
        VLLM_ASCEND_MODE1_CPU_DP_METADATA_SYNC=1, VLLM_ASCEND_EAGER_METADATA_SYNC_DEVICE=0,
        VLLM_ASCEND_MODE1_ADAPTIVE_KV_RESIZE=1, VLLM_ASCEND_MODE1_PREEMPTION_AUDIT=1,
        VLLM_ASCEND_MODE1_STEP_KV_CAP_INCLUDES_PLANNED_HEADROOM=1,
        VLLM_ASCEND_MODE1_PARITY_MAX_KV_TOKENS_FLOOR2=KV_BUDGETS['natural_f2'],
        VLLM_ASCEND_MODE1_PARITY_MAX_KV_TOKENS_FLOOR4=KV_BUDGETS['natural_f4'],
        VLLM_ASCEND_MODE1_PARITY_MAX_KV_TOKENS_FLOOR8=KV_BUDGETS['natural_f4_family_f8' if policy in ('natural_f4', 'planned_f4') else 'natural_f8'],
        VLLM_ASCEND_MODE1_PARITY_MAX_KV_TOKENS=KV_BUDGETS['full16'],
        VLLM_ASCEND_MODE1_PARITY_MAX_KV_TOKENS_FLOOR16=KV_BUDGETS['full16'],
        VANILLA_KV_TOKENS_PER_RANK=KV_BUDGETS['full16'], VANILLA_KV_ADMISSION_TOKENS_PER_RANK=KV_BUDGETS['full16'],
        VLLM_ASCEND_MC2_MIN_EP_SIZE=2, VLLM_ASCEND_FORCE_ALLTOALL_MOE=0,
        VLLM_ASCEND_FORCE_TORCH_NPU_ADD_RMS_NORM=1,
        VLLM_ASCEND_ENABLE_TOPK_TOPP_OPTIMIZATION=1, VLLM_ASCEND_USE_TOPK_TOPP_CUSTOM=0,
        VLLM_ROLLOUT_ENABLE_PREFIX_CACHING='true', VLLM_ROLLOUT_ENABLE_CHUNKED_PREFILL='true',
        VLLM_ROLLOUT_ASYNC_SCHEDULING='false', VLLM_ROLLOUT_GRAPH_SPREAD_REPEATS=0,
        VLLM_ROLLOUT_DATA_REBALANCE=0, VLLM_ROLLOUT_USE_TQDM=0, VERL_SIDECAR_ENABLE=0,
        VLLM_ROLLOUT_VERIFY_WEIGHT_COVERAGE=0, VLLM_ROLLOUT_WEIGHT_VALUE_AUDIT=0,
        VLLM_ROLLOUT_PHASE_MEMORY_LOG=0, VLLM_ROLLOUT_STAGE_TIMING=0,
        VLLM_ROLLOUT_MEMORY_OWNERSHIP_LOG=0, VLLM_ASCEND_RUNTIME_PROGRESS_INTERVAL=256,
        VLLM_ASCEND_USE_FILTERED_CUSTOM_OPP=1,
        VLLM_ASCEND_FILTERED_CUSTOM_OPP_PATH=REPO/'vllm_ascend/_cann_ops_custom_moe_filtered/vendors/vllm-ascend',
        VLLM_ASCEND_ELASTIC_ACLGRAPH_EXTENSION='/workspace/vllm-ascend/vllm_ascend/vllm_ascend_C.cpython-311-aarch64-linux-gnu.so',
        PLANNER_TOKENIZER_PATH='/data/Qwen3-30B-A3B', KV_ADMISSION_REQUIRE_PROMPT_TOKENS=1,
        ACTIVE_PEAK_SAFETY_FACTOR=1.16, BASELINE_ACTIVE_PEAK_SAFETY_FACTOR=1.16,
        # Ray appends ~70 bytes of session/socket names; Linux AF_UNIX caps
        # the whole pathname at107 bytes. Policy names must not enter it.
        GIT_TRACK_RUN_LOGS=0, RAY_TMPDIR=f'/tmp/f6_{seed}_{POLICIES.index(policy)}_{os.getpid()}',
        XDG_CACHE_HOME=arm/'cache/xdg', TORCHAIR_CACHE_HOME=arm/'cache/torchair',
        TRITON_CACHE_DIR=arm/'cache/triton', ASCEND_CACHE_PATH=arm/'cache/ascend',
        ASCEND_WORK_PATH=arm/'cache/work', PYTHONPATH=str(REPO))
    if policy in BASES:
        values.update(FAIR_ADAFLOOR_TARGET=REPO/'run_mode1_local_length_sorted_e2e_adaptive_floor4_aclgraph.sh',
                      ADAFLOOR_GRAPH_BASE_RUNNER=REPO/BASES[policy])
    env.update({k: str(v) for k, v in values.items()})
    return env


def arm_command(seed, policy):
    args = ['bash', str(REPO/'run_paper_fair_epoch1_2_from_common_epoch0.sh'), VARIANTS.get(policy, policy),
            f'actor_rollout_ref.rollout.seed={seed}', 'actor_rollout_ref.actor.optim.lr=1e-6',
            'actor_rollout_ref.actor.megatron.seed=42', 'actor_rollout_ref.ref.megatron.seed=42',
            '++data.seed=42', 'trainer.save_freq=5',
            'actor_rollout_ref.rollout.cudagraph_mode=FULL_DECODE_ONLY',
            'actor_rollout_ref.rollout.cudagraph_capture_sizes=[32]',
            'actor_rollout_ref.rollout.async_scheduling=false',
            '++actor_rollout_ref.actor.megatron.override_transformer_config.moe_shared_expert_overlap=False']
    if policy in BASES:
        args.append('++actor_rollout_ref.rollout.engine_kwargs.vllm.num_gpu_blocks_override=810')
    return args


def json_records(directory, marker):
    records = []
    for p in sorted((directory/'logs').glob('*.txt')):
        for line in p.read_text(errors='replace').splitlines():
            if marker in line:
                records.append(json.JSONDecoder().raw_decode(line.split(marker, 1)[1])[0])
    return records


def tailguard_caps(epoch_dir):
    plans=list((epoch_dir/'oracle').glob('*rank_plan_summary.json'))
    if len(plans)!=1:
        raise RuntimeError('TailGuard requires one five-step plan summary')
    plan=json.loads(plans[0].read_text())
    if len(plan)!=5:
        raise RuntimeError('TailGuard requires five planned caps')
    expected={int(p['step']):int(p['tail_guard_response_cap']) for p in plan}
    if set(expected)!=set(range(1,6)):
        raise RuntimeError('Invalid TailGuard plan steps')
    audits=json_records(epoch_dir,'TAILGUARD_CAP_AUDIT ')
    if (len(audits)!=80 or {(a['step'],a['rank']) for a in audits}!=
            {(s,r) for s in range(1,6) for r in range(16)}):
        raise RuntimeError('Missing actual per-step/rank TailGuard cap audits')
    if any(a['max_tokens']!=expected[a['step']] for a in audits):
        raise RuntimeError('Actual sampling cap does not match planned TailGuard')
    return expected


def validate_response_lengths(lengths, cap, first_pad_is_eos):
    """Separate raw generated work from the inherited first-pad EOS mask.

    Actual sampling limits are independently checked against every rank's
    TAILGUARD_CAP_AUDIT. Original VERL get_response_mask includes the first
    padded EOS after a short length stop: cap raw tokens -> cap+1 mask tokens.
    This is not permission to generate an extra token beyond max_tokens.
    """
    extra_allowed=bool(first_pad_is_eos and cap<16384)
    limit=cap+int(extra_allowed)
    if len(lengths)!=512 or any(n<0 or n>limit for n in lengths):
        raise RuntimeError('Missing/invalid response-mask length records')
    synthetic=sum(n==cap+1 for n in lengths) if extra_allowed else 0
    return sum(lengths)-synthetic,synthetic


def parse_epoch(epoch_dir, require_tailguard=False):
    logs = [p for p in (epoch_dir/'logs').glob('*.txt') if 'rollout_output_time_s:' in p.read_text(errors='replace')]
    if len(logs) != 1:
        raise RuntimeError(f'Ambiguous trainer log: {epoch_dir}')
    text = ANSI.sub('', logs[0].read_text(errors='replace')).replace('\r', '\n')
    if not re.search(r'\[run\] end_time=\S+ exit_code=0', text):
        raise RuntimeError(f'Missing successful child exit: {epoch_dir}')
    if not re.search(r'Initializing a V1 LLM engine \(v0\.14', text):
        raise RuntimeError('Trainer did not initialize the local vLLM0.14 runtime')
    times = list(map(float, re.findall(r'=> rollout_output_time_s:\s*([\d.]+)', text)))
    if len(times) != 5 or re.search(r'RayTaskError|OutOfMemoryError|Memory_Allocation_Failure|NPU out of memory', text):
        raise RuntimeError(f'Incomplete or failed epoch: {epoch_dir}')
    metrics = [line for line in text.splitlines() if re.search(r'(?:^|\s)step:\d+\s+-', line)]
    if len(metrics) != 5:
        raise RuntimeError('Missing five actor update metric records')
    for line in metrics:
        lr = re.search(r'actor/lr:([^ ]+)', line)
        grad = re.search(r'actor/grad_norm:([^ ]+)', line)
        abort = re.search(r'response/aborted_ratio:([^ ]+)', line)
        if (not lr or abs(float(lr[1])-1e-6) > 1e-12 or not grad or float(grad[1]) <= 0
                or not abort or float(abort[1]) != 0):
            raise RuntimeError('Actor update/aborted-response verification failed')
    audits = json_records(epoch_dir, 'MODE1_STEP_RUNTIME_AUDIT ')
    expected = {(s, r) for s in range(1, 6) for r in range(16)}
    if (len(audits) != 80 or {(a['step'], a['rank']) for a in audits} != expected
            or any(not a['complete'] or a.get('full_iterations', 0) <= 0 for a in audits)):
        raise RuntimeError('Missing actual five-step/16-rank FULL graph replay evidence')
    buckets = re.findall(r'Applied epoch-planned ACLGraph buckets: step=(\d+) count=(\d+)', text)
    if sorted((int(s), int(n)) for s, n in buckets) != [(s, 8) for s in range(1, 6)]:
        raise RuntimeError('Not all five steps applied their planned K8 buckets')
    # Reuse the paper's exact release accounting, excluding duplicated timing
    # fields in v0.14's consolidated metric lines (not extra rollout ends).
    parser_path = REPO.parent/'qwen3_shrink_aware/analysis_eval/plot_e2e_performance.py'
    spec = importlib.util.spec_from_file_location('hpca_release_parser', parser_path)
    parser = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = parser
    spec.loader.exec_module(parser)
    clean = '\n'.join(line for line in text.splitlines()
                      if 'rollout_output_time_s:' not in line or '=> rollout_output_time_s:' in line)
    releases = parser.parse_released_rank_time_by_step(clean)
    if len(releases) != 5:
        raise RuntimeError('Incomplete release stage records')
    caps=tailguard_caps(epoch_dir) if require_tailguard else {s:16384 for s in range(1,6)}
    generation_config=json.loads(Path('/data/Qwen3-30B-A3B/generation_config.json').read_text())
    eos=generation_config.get('eos_token_id',[])
    eos=eos if isinstance(eos,list) else [eos]
    first_pad_is_eos=generation_config.get('pad_token_id') in eos
    response_tokens, scores, prompts, synthetic_eos = 0, [], {}, 0
    for step in range(1, 6):
        # v0.14 dumps decoded text, not response_mask, in the JSONL. The
        # canonical length file is produced directly by response_mask.sum;
        # do not retokenize decoded text or count characters as tokens.
        lengths = list(map(int, (epoch_dir/f'rollout_length/length_{step}.txt').read_text().split()))
        raw_work,extra=validate_response_lengths(lengths,caps[step],first_pad_is_eos)
        response_tokens += raw_work
        synthetic_eos += extra
        path = epoch_dir/f'rollout_data/{step}.jsonl'
        opener = path.open if path.exists() else lambda **kw: gzip.open(str(path)+'.gz', 'rt', **kw)
        count = 0
        with opener(encoding='utf-8') as f:
            for line in f:
                row = json.loads(line)
                scores.append(float(row['score']))
                prompts[row['input']] = prompts.get(row['input'], 0) + 1
                count += 1
        if count != 512:
            raise RuntimeError('Each step must contain512 responses')
    if len(prompts) != 160 or set(prompts.values()) != {16}:
        raise RuntimeError('Expected exactly160 prompts x16 responses')
    return dict(epoch_dir=str(epoch_dir), rollout_s=sum(times), step_seconds=times,
                response_tokens=response_tokens, response_tokens_s=response_tokens/sum(times),
                response_tokens_source='saved response_mask.sum minus proven first padded EOS at a short cap',
                first_padded_eos_tokens_excluded=synthetic_eos,
                reward=stats.mean(scores), mean_length=response_tokens/2560,
                prompt_hash=hashlib.sha256('\n'.join(sorted(prompts)).encode()).hexdigest(),
                release8=sum(r.floor8_rank_s for r in releases),
                release4=sum(r.floor4_rank_s for r in releases),
                release2=sum(r.floor2_rank_s for r in releases),
                released_rank_s=sum(r.total_rank_s for r in releases),
                preemptions=sum(a['preemptions'] for a in audits), full_graph_audits=80)


def parse_arm(root, seed, policy):
    run = root/f'seed_{seed}'/policy/'fig6'
    receipt = run/'CHECKPOINTS_REMOVED_AFTER_VALIDATION.txt'
    if not receipt.is_file() or 'validated_epochs=001,002' not in receipt.read_text():
        raise RuntimeError(f'Both epochs not validated: {run}')
    epochs = []
    for n in (1, 2):
        candidates = list(run.glob(f'epoch_{n:03d}_*'))
        if len(candidates) != 1:
            raise RuntimeError('Missing/ambiguous epoch directory')
        epochs.append(parse_epoch(candidates[0], require_tailguard=policy not in ('vanilla','lengthsort')))
    if epochs[0]['prompt_hash'] != epochs[1]['prompt_hash']:
        raise RuntimeError('Epoch prompt multisets differ')
    # The retained logs must show the actual second-epoch checkpoint chain.
    second_text = '\n'.join(p.read_text(errors='replace') for p in Path(epochs[1]['epoch_dir']).glob('logs/*.txt'))
    first_ckpt = str(Path(epochs[0]['epoch_dir'])/'checkpoints/qwen3moe_for_eagle3/global_step_5')
    if first_ckpt not in second_text:
        raise RuntimeError('Epoch2 did not reference its own epoch1 checkpoint')
    return dict(seed=seed, policy=policy, epochs=epochs,
                **{key: stats.mean(e[key] for e in epochs) for key in
                   ('rollout_s', 'response_tokens', 'reward', 'mean_length',
                    'release8', 'release4', 'release2', 'released_rank_s')},
                response_tokens_s=sum(e['response_tokens'] for e in epochs)/sum(e['rollout_s'] for e in epochs),
                prompt_hash=epochs[0]['prompt_hash'], preemptions=sum(e['preemptions'] for e in epochs))


def summarize(root):
    rows = []
    for seed in SEEDS:
        for policy in POLICIES:
            p = root/f'seed_{seed}'/policy/'ARM_PASS.json'
            if p.exists():
                rows.append(json.loads(p.read_text()))
    if len({r['prompt_hash'] for r in rows}) > 1:
        raise RuntimeError('Policy/seed prompt multisets differ')
    aggregate = []
    for policy in POLICIES:
        cases = [r for r in rows if r['policy'] == policy]
        if not cases:
            continue
        entry = dict(policy=policy, seeds=len(cases))
        for metric in ('rollout_s', 'response_tokens_s', 'reward', 'mean_length',
                       'release8', 'release4', 'release2', 'released_rank_s'):
            values = [r[metric] for r in cases]
            entry[metric] = stats.mean(values)
            entry[metric+'_sample_sd'] = stats.stdev(values) if len(values) > 1 else None
        aggregate.append(entry)
    summary = dict(complete=len(rows)==24, completed_arms=len(rows), expected_arms=24,
                   statistics='epochs averaged within seed, sample SD across seeds',
                   trials=rows, aggregate=aggregate)
    write_json(root/'analysis_eval/summary.json', summary)
    if aggregate:
        with (root/'analysis_eval/aggregate.csv').open('w', newline='') as f:
            w=csv.DictWriter(f, fieldnames=list(aggregate[0]));w.writeheader();w.writerows(aggregate)
    if summary['complete']:
        by_key={(r['seed'],r['policy']):r for r in rows}
        paired=[by_key[(s,'lengthsort_guard')]['rollout_s']-by_key[(s,'planned_f4')]['rollout_s'] for s in SEEDS]
        half=4.30265273*stats.stdev(paired)/math.sqrt(3)
        summary['planned_f4_vs_lengthsort_guard_paired_seconds']=dict(
            mean=stats.mean(paired),ci95=[stats.mean(paired)-half,stats.mean(paired)+half])
        write_json(root/'analysis_eval/summary.json',summary)
        plot_figure(root, aggregate)
        write_json(root/'FIG6_PASS.json', dict(status='PASS', arms=24, epochs=48, steps=240,
                   summary_sha256=digest(root/'analysis_eval/summary.json')))
    print(f"[fig6] completed_arms={len(rows)}/24 summary={root/'analysis_eval/summary.json'}", flush=True)
    return summary


def plot_figure(root, aggregate):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows = {r['policy']: r for r in aggregate}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    x=list(range(8));labels=['Vanilla\nF16','LS\nF16','LS\n+TG','Fixed\nF4','Min\nSkew',
                            'Ada-N\nF4','Ada-N\nF2','Ada-P\nF4','Ada-P\nF2']
    axes[0].bar(x,[rows[p]['rollout_s'] for p in POLICIES],
                yerr=[rows[p]['rollout_s_sample_sd'] for p in POLICIES], capsize=3,
                color='0.65', edgecolor='black')
    bottom=[0.0]*8
    for key,label,hatch,color in [('release8','16→8','///','0.8'),('release4','8→4','xxx','0.6'),
                                  ('release2','4→2','...','0.4')]:
        values=[rows[p][key] for p in POLICIES]
        axes[1].bar(x, values, bottom=bottom, label=label, hatch=hatch, color=color, edgecolor='black')
        bottom=[a+b for a,b in zip(bottom,values)]
    axes[1].errorbar(x,bottom,yerr=[rows[p]['released_rank_s_sample_sd'] for p in POLICIES],
                     fmt='none',color='black',capsize=3)
    for i,policy in enumerate(POLICIES):
        if i:
            delta=100*(rows[policy]['rollout_s']/rows['vanilla']['rollout_s']-1)
            axes[0].annotate(f'{delta:+.1f}%',(i,rows[policy]['rollout_s']+rows[policy]['rollout_s_sample_sd']),
                             xytext=(0,5),textcoords='offset points',ha='center',fontsize=8)
        if bottom[i]>0:
            axes[1].annotate(f'{bottom[i]/1000:.2f}k',(i,bottom[i]+rows[policy]['released_rank_s_sample_sd']),
                             xytext=(0,5),textcoords='offset points',ha='center',fontsize=8)
    tg_delta=100*(rows['planned_f4']['rollout_s']/rows['lengthsort_guard']['rollout_s']-1)
    axes[0].text(0.98,0.97,f'Ada-P F4: {tg_delta:+.2f}% vs LS+TG',
                 transform=axes[0].transAxes,ha='right',va='top',fontsize=9)
    for ax,title,ylabel in zip(axes,['(a) Rollout Critical Path','(b) Released Rank-Time'],
                              ['Epoch Rollout Time (s)','Released Rank-Time (rank-s)']):
        ax.set_xticks(range(9),labels);ax.set_title(title);ax.set_ylabel(ylabel)
        ax.text(8, ax.get_ylim()[1]*0.10, 'Budget\nexcluded', ha='center', fontsize=8)
        ax.set_xlim(-0.6,8.6)
    axes[1].legend()
    fig.tight_layout()
    for ext in ('pdf','png','svg'):
        fig.savefig(root/f'analysis_eval/figure6_v014_graph.{ext}', dpi=200)
    plt.close(fig)


def adopt_validation_amendment(root, contract):
    """Explicitly adopt only the three predeclared validation/control files."""
    before=json.loads((root/'validation_amendment_before.json').read_text())
    original=json.loads((root/'protocol.json').read_text())
    allowed=tuple(before['allowed_before'])
    if (before['original_execution_sha256'] != original['execution_sha256']
            or execution_digest(allowed) != before['unchanged_files_sha256']):
        raise RuntimeError('Validation amendment also changed undeclared execution files')
    a,b=dict(original),dict(contract)
    a.pop('execution_sha256');b.pop('execution_sha256')
    if a != b:
        raise RuntimeError('Validation amendment changed experimental protocol')
    number=1
    archive=root/'protocol.before_validation_amendment.json'
    while archive.exists():
        number+=1
        archive=root/f'protocol.before_validation_amendment.{number:03d}.json'
    write_json(archive,original)
    amendment_path=root/('validation_amendment.json' if number==1 else f'validation_amendment.{number:03d}.json')
    write_json(amendment_path,dict(
        kind=before.get('change_kind','host_validation_and_resume_only'),before=before,
        after={p:digest(REPO/p) for p in allowed},
        original_protocol_sha256=digest(archive),
        updated_execution_sha256=contract['execution_sha256'],
        reason=before.get('reason','Host validation/resume fix or short Ray UNIX socket path; generation/training implementation unchanged')))
    write_json(root/'protocol.json',contract)


def finalize_record(root,seed,policy,env):
    arm=root/f'seed_{seed}'/policy
    record=parse_arm(root,seed,policy)
    failed=arm/'FAILED.json'
    if failed.exists():
        write_json(arm/'RECOVERED_HOST_FAILURE.json',dict(
            previous_failure=json.loads(failed.read_text()),
            recovery='Validated both original epoch outputs/checkpoint chain before host finalization; consult resume receipts for any epoch continuation. No completed epoch was rerun.',
            runtime_protocol_unchanged=True))
        failed.unlink()
    for p in sorted((arm/'fig6').glob('epoch_*/rollout_data/*.jsonl')):
        subprocess.run(['gzip','-1','--',str(p)],check=True)
    write_json(arm/'ARM_PASS.json',record)
    if (arm/'FAILED.json').exists():
        (arm/'FAILED.json').rename(arm/'FAILED_before_validation_fix.json')
    shutil.rmtree(arm/'cache',ignore_errors=True)
    shutil.rmtree(env['RAY_TMPDIR'],ignore_errors=True)
    print(f'[fig6] PASS seed={seed} policy={policy} checkpoints_cleaned=True raw_records=gzip',flush=True)
    summarize(root)


def select_seed(only=None, seed=None):
    selected = seed if seed is not None else (int(only.split(':')[0]) if only else 0)
    if selected not in SEEDS or (only and int(only.split(':')[0]) != selected):
        raise ValueError('--seed and --only must select the same original seed')
    return selected


def require_previous_seed_review(root, seed):
    if seed == SEEDS[0]:
        return
    previous = SEEDS[SEEDS.index(seed)-1]
    path = root/f'seed_{previous}/SEED_REVIEW_PASS.json'
    if not path.is_file():
        raise RuntimeError(f'Seed{previous} must finish and be reviewed before seed{seed}; no later-seed launch')
    review = json.loads(path.read_text())
    protocol_current = review.get('protocol_sha256') == digest(root/'protocol.json')
    if not protocol_current:
        # Retain the original reviewed evidence/approval immutably. A new
        # explicit host-only reapproval must bind BOTH original approval and
        # current protocol; an unrelated/stale approval still fails closed.
        reaffirm = path.with_name('SEED_REVIEW_PROTOCOL_REAFFIRMATION.json')
        if reaffirm.is_file():
            a = json.loads(reaffirm.read_text())
            protocol_current = (a.get('approved') is True
                and a.get('original_review_sha256') == digest(path)
                and a.get('protocol_sha256') == digest(root/'protocol.json')
                and a.get('runtime_protocol_unchanged') is True)
    if (review.get('approved') is not True or review.get('reviewed_seed') != previous
            or review.get('next_seed') != seed
            or not protocol_current):
        raise RuntimeError('Seed review is stale or does not approve this seed/protocol')
    evidence = (root/review.get('evidence_path','')).resolve()
    if (not evidence.is_relative_to(root.resolve()) or not evidence.is_file()
            or digest(evidence) != review.get('evidence_sha256')):
        raise RuntimeError('Seed review lacks its retained evidence report')
    for policy in POLICIES:
        receipt = root/f'seed_{previous}/{policy}/ARM_PASS.json'
        if not receipt.is_file() or parse_arm(root,previous,policy) != json.loads(receipt.read_text()):
            raise RuntimeError(f'Previous seed not fully validated: {previous}:{policy}')


def prepare_planned_epoch1_resume(root, seed, policy):
    """Host-only recovery of a completed epoch rejected by legacy validation."""
    if policy != 'planned_f4':
        raise RuntimeError('Partial-epoch resume is limited to the native Planned arm')
    arm=root/f'seed_{seed}'/policy
    run=arm/'fig6'
    epochs=list(run.glob('epoch_*'))
    first=run/'epoch_001_mode1_planned'
    if len(epochs)!=1 or epochs[0]!=first:
        raise RuntimeError('Resume requires exactly original epoch1, with no epoch2 started')
    summary=parse_epoch(first,require_tailguard=True)
    checkpoint=first/'checkpoints/qwen3moe_for_eagle3/global_step_5'
    marker=checkpoint.parent/'latest_checkpointed_iteration.txt'
    shards=list((checkpoint/'actor/dist_ckpt').glob('*.distcp'))
    metadata=checkpoint/'actor/dist_ckpt/.metadata'
    if (not marker.is_file() or marker.read_text().strip()!='5' or len(shards)!=32
            or any(p.stat().st_size<=0 for p in shards) or not metadata.is_file()):
        raise RuntimeError('Original epoch1 distributed checkpoint is incomplete')
    checkpoint_bytes=sum(p.stat().st_size for p in checkpoint.rglob('*') if p.is_file())
    required=checkpoint_bytes+8*1024**3
    free=shutil.disk_usage(root).free
    if free<required:
        raise RuntimeError('Insufficient headroom for the remaining epoch2 checkpoint plus8GiB; original checkpoint retained')
    write_json(arm/'VALIDATED_EPOCH1_RESUME.json',dict(
        epoch1=summary,checkpoint=str(checkpoint),checkpoint_shards=32,
        checkpoint_bytes=checkpoint_bytes,free_bytes=free,required_remaining_bytes=required,
        note='One epoch remains. Original120GiB new-arm guard is unchanged. No epoch1 rerun or old-budget data reuse.'))
    env=arm_env(root,seed,policy)
    env['FAIR_RESUME_EXISTING']='1'
    return env


def run_suite(root, only=None, validation_amendment=False, seed=None):
    selected = select_seed(only,seed)
    root.mkdir(parents=True, exist_ok=True)
    # A live source edit must stop subsequent arms, not silently mix versions.
    contract=protocol();file=root/'protocol.json'
    if file.exists() and json.loads(file.read_text()) != contract:
        if validation_amendment:
            adopt_validation_amendment(root,contract)
        else:
            raise RuntimeError('Experiment protocol/code changed; do not mix arms in this root')
    write_json(file,contract)
    require_previous_seed_review(root,selected)
    subprocess.run([sys.executable,str(REPO/'tools/build_camem_lifecycle.py')], check=True, cwd=REPO)
    # One invocation can NEVER advance to a later seed automatically.
    # Subsequent seeds additionally need an explicit evidence-linked review.
    for seed in (selected,):
        for policy in POLICIES:
            if only and only != f'{seed}:{policy}':
                continue
            arm=root/f'seed_{seed}'/policy
            receipt=arm/'ARM_PASS.json'
            if receipt.exists():
                print(f'[fig6] already validated {seed}:{policy}', flush=True)
                continue
            if (arm/'fig6').exists():
                epochs=list((arm/'fig6').glob('epoch_*'))
                if len(epochs)==1:
                    env=prepare_planned_epoch1_resume(root,seed,policy)
                    if execution_digest()!=contract['execution_sha256'] or digest(CAPACITY_EVIDENCE)!=contract['capacity_evidence_sha256']:
                        raise RuntimeError('Frozen source/capacity changed before partial resume')
                    print(f'[fig6] RESUME verified epoch1 seed={seed} policy={policy}; run epoch2 only',flush=True)
                    with (arm/'driver.log').open('a') as log:
                        result=subprocess.run(arm_command(seed,policy),env=env,cwd=REPO,
                                              stdout=log,stderr=subprocess.STDOUT)
                    if result.returncode:
                        write_json(arm/'FAILED.json',dict(exit_code=result.returncode,checkpoints_retained=True))
                        raise RuntimeError('Planned epoch2 continuation failed; original data/checkpoints retained')
                    finalize_record(root,seed,policy,env)
                    continue
                if len(epochs)!=2:
                    raise RuntimeError(f'Incomplete arm retained; inspect before retry: {arm}')
                # Fully audit both original generations BEFORE permitting
                # fair-runner cleanup. No NPU rerun or output rewriting.
                for epoch in epochs:
                    parse_epoch(epoch, require_tailguard=policy not in ('vanilla','lengthsort'))
                # A preceding fair run may have validated both epochs and
                # removed its private checkpoints before a host summary
                # failed. Its durable receipt plus full output revalidation
                # is sufficient: do not demand deleted shards or rerun NPU.
                if (arm/'fig6/CHECKPOINTS_REMOVED_AFTER_VALIDATION.txt').is_file():
                    finalize_record(root,seed,policy,arm_env(root,seed,policy))
                    continue
                env=arm_env(root,seed,policy);env['FAIR_FINALIZE_EXISTING']='1'
                with (arm/'driver.log').open('a') as log:
                    result=subprocess.run(arm_command(seed,policy),env=env,cwd=REPO,
                                          stdout=log,stderr=subprocess.STDOUT)
                if result.returncode:
                    raise RuntimeError(f'Existing arm failed final validation: {arm}')
                finalize_record(root,seed,policy,env)
                continue
            if (execution_digest() != contract['execution_sha256'] or
                    digest(CAPACITY_EVIDENCE) != contract['capacity_evidence_sha256'] or
                    digest(REPO.parent/'qwen3_shrink_aware/analysis_eval/plot_e2e_performance.py') != contract['release_parser_sha256']):
                raise RuntimeError('Execution source changed since suite launch')
            free=shutil.disk_usage(root).free/1024**3
            if free < 120:
                raise RuntimeError(f'Insufficient disk headroom: {free:.1f}GiB; need120GiB. No threshold reduction.')
            arm.mkdir(parents=True,exist_ok=True)
            env=arm_env(root,seed,policy);cmd=arm_command(seed,policy)
            for key in ('XDG_CACHE_HOME','TORCHAIR_CACHE_HOME','TRITON_CACHE_DIR','ASCEND_CACHE_PATH','ASCEND_WORK_PATH','RAY_TMPDIR'):
                Path(env[key]).mkdir(parents=True,exist_ok=True)
            write_json(arm/'launch.json',dict(command=cmd,env={k:env[k] for k in env
                       if k.startswith(('VLLM_','VERL_','FAIR_','ADAFLOOR_')) or k in ('PYTORCH_NPU_ALLOC_CONF','HCCL_BUFFSIZE','TASK_QUEUE_ENABLE')}))
            print(f'[fig6] START seed={seed} policy={policy} two_epochs=1,2 driver={arm/"driver.log"}',flush=True)
            with (arm/'driver.log').open('a') as log:
                result=subprocess.run(cmd,env=env,cwd=REPO,stdout=log,stderr=subprocess.STDOUT)
            if result.returncode:
                write_json(arm/'FAILED.json',dict(exit_code=result.returncode, checkpoints_retained=True))
                raise RuntimeError(f'Arm failed {seed}:{policy}; inspect {arm/"driver.log"}')
            finalize_record(root,seed,policy,env)
    summary = summarize(root)
    if all((root/f'seed_{selected}/{p}/ARM_PASS.json').is_file() for p in POLICIES):
        write_json(root/f'seed_{selected}/SEED_COMPLETE_PENDING_REVIEW.json',dict(
            seed=selected,policies=len(POLICIES),measured_epochs=2*len(POLICIES),
            protocol_sha256=digest(file),automatic_next_seed=False,
            note='Review protocol, runtime, reward, lengths, throughput and resource evidence before explicit later-seed launch. This is not approval.'))
        print(f'[fig6] seed={selected} complete; STOPPED for results review, no automatic next seed',flush=True)
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=('dry-run','run','summarize'))
    p.add_argument('--root',type=Path,default=Path(os.getenv('MLSYS_FIG6_ROOT',
                   '/data/adafloor_graph_results/mlsys_fig6_v014_full_graph_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))))
    p.add_argument('--only',choices=[f'{s}:{m}' for s in SEEDS for m in POLICIES])
    p.add_argument('--seed',type=int,choices=SEEDS,
                   help='Run one seed only; default0 or the seed in --only. Later seeds require prior-seed review.')
    p.add_argument('--adopt-validation-amendment',action='store_true')
    args=p.parse_args()
    selected=select_seed(args.only,args.seed)
    if args.action=='dry-run':
        print(json.dumps(protocol(),indent=2,ensure_ascii=False))
        for seed in (selected,):
            for policy in POLICIES:
                if not args.only or args.only==f'{seed}:{policy}':
                    print(json.dumps(dict(seed=seed,policy=policy,command=arm_command(seed,policy))))
        return
    if args.action=='summarize':
        summarize(args.root)
    else:
        run_suite(args.root,args.only,args.adopt_validation_amendment,args.seed)


if __name__=='__main__':
    main()
