#!/usr/bin/env python3
"""Prespecified seed0/epoch1 eager controls, plus uniform endpoint evaluation."""
import gzip
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import statistics

CODE=Path(__file__).resolve().parent
SOURCE=Path('/workspace/cann-recipes-train/llm_rl/qwen3_014/qwen3_main')
ROOT=Path('/data/adafloor_graph_results/mlsys_v014_supplement_20261023T180000Z')
GRAPH=Path('/data/adafloor_graph_results/mlsys_fig6_v014_graph_calibrated_20261017T134200Z')
HELDOUT=ROOT/'heldout_prepared/heldout256.parquet'
EVAL_TRAIN=ROOT/'heldout_prepared/train_bookkeeping32.parquet'


def suite():
    spec=importlib.util.spec_from_file_location('eager_main_reference',SOURCE/'tools/run_mlsys_fig6.py')
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    wanted=json.loads((GRAPH/'protocol.json').read_text())['execution_sha256']
    if m.execution_digest()!=wanted:
        raise RuntimeError('Eager source snapshot differs from historical graph execution fingerprint')
    return m


def eager_flags():
    return dict(TASK_QUEUE_ENABLE='2',VLLM_ROLLOUT_TASK_QUEUE_ENABLE='2',
        ROLLOUT_ENFORCE_EAGER='True',VLLM_ENABLE_GRAPH_MODE='0',
        ADAFLOOR_ACLGRAPH_MODE='NONE',VLLM_ASCEND_ELASTIC_ACLGRAPH='0',
        VLLM_ASCEND_ELASTIC_ACLGRAPH_CAPTURE_ATTENTION='0',VLLM_ASCEND_ELASTIC_ACLGRAPH_CAPTURE_MOE='0',
        VLLM_ROLLOUT_ASYNC_SCHEDULING='false',VERL_SIDECAR_ENABLE='0',
        VLLM_ASCEND_EAGER_METADATA_SYNC_DEVICE='0')


def build_training(policy):
    if policy not in ('vanilla','planned_f4'):raise ValueError('Only the two requested eager controls')
    m=suite();base=ROOT/'eager_controls';env=m.arm_env(base,0,policy)
    env.update(eager_flags())
    env.update(FAIR_START_EPOCH='1',FAIR_TOTAL_EPOCHS='2',FAIR_FREEZE_ACTOR='0',
        FAIR_KEEP_COMPLETED_CHECKPOINTS='1')
    if policy=='planned_f4':env['FAIR_ADAFLOOR_TARGET']=str(CODE/'eager_planned_wrapper.sh')
    cmd=m.arm_command(0,policy)+['actor_rollout_ref.rollout.enforce_eager=True',
        'actor_rollout_ref.rollout.cudagraph_mode=NONE']
    return env,cmd


def build_evaluation(endpoint,output):
    m=suite();env=m.arm_env(Path(output),0,'vanilla');env.update(eager_flags())
    env.update(OUTPUT_ROOT=str(output),OUTPUT_SUBDIR='evaluation',RECORD_DIR=str(Path(output)/'evaluation'),
        MODEL_PATH='/data/Qwen3-30B-A3B',DISTCP_PATH='/data/Qwen3-30B-A3B_megatron',
        TRAIN_FILE=str(EVAL_TRAIN),TEST_FILE=str(HELDOUT),DATASET_FRACTION='1.0',
        TRAINER_TOTAL_EPOCHS='1',TRAINER_TOTAL_TRAINING_STEPS='1',
        SAVE_CKPT_ENABLE='0',TRAINER_SAVE_FREQ='-1',VERL_RESET_TRAINER_PROGRESS_AFTER_RESUME='1',
        VLLM_ASCEND_ENABLE_ELASTIC_PARALLEL_SHRINK='0',VLLM_ASCEND_ELASTIC_EXECUTION_MODE='0',
        VLLM_ASCEND_ELASTIC_MIN_COMPUTE_GROUP_SIZE='16',VLLM_ASCEND_SHRINK_AWARE_ENABLE='0',
        VLLM_ROLLOUT_FORCE_ELASTIC_MOE_POLICY='0',
        VLLM_ROLLOUT_PARALLEL_MODE='ep',VLLM_ENABLE_EXPERT_PARALLEL='1',
        VLLM_EAGER_BASELINE_NO_RESAMPLE='1',VLLM_EPOCH_LENGTH_REGROUP_ENABLE='0',
        HCCL_IF_BASE_PORT='12000',VERL_HCCL_IF_BASE_PORT_START='12000',VERL_HCCL_IF_BASE_PORT_BLOCK='16384')
    cmd=['bash',str(SOURCE/'internal/wj_train_grpo_qwen30b_a3b_16die_true_weight_regroup.sh'),
        'trainer.val_before_train=True','++trainer.val_only=True','trainer.test_freq=-1','trainer.save_freq=-1',
        'trainer.resume_mode=resume_path',f'trainer.resume_from_path={endpoint}',
        'trainer.del_local_ckpt_after_load=False',f'++trainer.validation_data_dir={Path(output)/"samples"}',
        'actor_rollout_ref.rollout.enforce_eager=True','actor_rollout_ref.rollout.cudagraph_mode=NONE',
        'actor_rollout_ref.rollout.free_cache_engine=True','actor_rollout_ref.rollout.n=1',
        'actor_rollout_ref.rollout.ignore_eos=False','actor_rollout_ref.rollout.seed=0',
        'actor_rollout_ref.rollout.val_kwargs.n=1','actor_rollout_ref.rollout.val_kwargs.do_sample=False',
        'actor_rollout_ref.rollout.val_kwargs.temperature=0','actor_rollout_ref.rollout.val_kwargs.top_p=1',
        'actor_rollout_ref.rollout.val_kwargs.top_k=-1','actor_rollout_ref.actor.use_kl_loss=False',
        'data.val_batch_size=256','++data.validation_shuffle=False','data.max_response_length=16384',
        'data.max_prompt_length=1024','data.dataloader_num_workers=0',
        '++actor_rollout_ref.rollout.engine_kwargs.vllm.num_gpu_blocks_override=2975',
        'actor_rollout_ref.actor.megatron.seed=42','++data.seed=42']
    return env,cmd


def records(path):
    opener=path.open if path.exists() else lambda:gzip.open(str(path)+'.gz','rt')
    with opener() as stream:
        for line in stream:
            if line.strip():yield json.loads(line)


def parse_eager(root,policy):
    epoch=root/'fig6'/f'epoch_001_mode{0 if policy=="vanilla" else 1}_{"vanilla" if policy=="vanilla" else "planned"}'
    logs=list((epoch/'logs').glob('*.txt'))
    if len(logs)!=1:raise RuntimeError('Ambiguous eager log')
    text=logs[0].read_text(errors='replace')
    if not re.search(r'\[run\] end_time=\S+ exit_code=0',text):raise RuntimeError('Eager epoch incomplete')
    audits=[json.loads(x) for x in re.findall(r'MODE1_STEP_RUNTIME_AUDIT (\{[^\n]+\})',text)]
    if (len(audits)!=80 or {(a['step'],a['rank']) for a in audits}!={(s,r) for s in range(1,6) for r in range(16)}
            or any(not a['complete'] or a.get('full_iterations',0)!=0 or a.get('eager_iterations',0)<=0 for a in audits)):
        raise RuntimeError('Not five actual16-rank eager generations')
    metrics=[dict(re.findall(r'([\w/]+):([^ ]+)',line)) for line in text.splitlines() if re.search(r'step:\d+\s+-',line)]
    if len(metrics)!=5 or any(float(m.get('actor/lr',-1))!=1e-6
            or not math.isfinite(float(m.get('actor/grad_norm',0)))
            or float(m.get('actor/grad_norm',0))<=0 or float(m.get('response/aborted_ratio',-1))!=0 for m in metrics):
        raise RuntimeError('Missing real updates or aborted outputs')
    reference=GRAPH/f'seed_0/{policy}/fig6'/epoch.name
    plan_name='length_sorted_rank_plan.json' if policy=='planned_f4' else 'vanilla_aclgraph_bucket_plan.json'
    if policy=='planned_f4':
        if (epoch/'oracle'/plan_name).read_bytes()!=(reference/'oracle'/plan_name).read_bytes():
            raise RuntimeError('Eager and graph first-epoch plan changed')
    elif any(a['kv_tokens']!=380800 for a in audits):raise RuntimeError('Vanilla KV budget changed')
    m=suite();caps=m.tailguard_caps(epoch) if policy=='planned_f4' else {s:16384 for s in range(1,6)}
    config=json.loads(Path('/data/Qwen3-30B-A3B/generation_config.json').read_text())
    eos=config['eos_token_id'];eos=eos if isinstance(eos,list) else [eos]
    correction=config['pad_token_id'] in eos
    steps=[];scores=[]
    for step,metric in enumerate(metrics,1):
        actual=list(records(epoch/f'rollout_data/{step}.jsonl'));old=list(records(reference/f'rollout_data/{step}.jsonl'))
        if len(actual)!=512 or [r['input'] for r in actual]!=[r['input'] for r in old]:
            raise RuntimeError('Actual saved input sequence differs from graph')
        lengths=list(map(int,(epoch/f'rollout_length/length_{step}.txt').read_text().split()))
        work,synthetic=m.validate_response_lengths(lengths,caps[step],correction)
        duration=float(metric['rollout/rollout_output_time_s']);scores.extend(float(r['score']) for r in actual)
        steps.append(dict(step=step,seconds=duration,response_tokens=work,synthetic_padding_eos_excluded=synthetic,
            response_tokens_s=work/duration,actual_saved_input_order_equal=True))
    graph=m.parse_epoch(reference,require_tailguard=policy=='planned_f4')
    from rank_occupancy import parse_epoch as occupancy
    eager_occupancy=occupancy(epoch,{r['step']:r['response_tokens'] for r in steps},'eager')
    graph_caps=m.tailguard_caps(reference) if policy=='planned_f4' else {s:16384 for s in range(1,6)}
    graph_work={}
    for step in range(1,6):
        lengths=list(map(int,(reference/f'rollout_length/length_{step}.txt').read_text().split()))
        graph_work[step]=m.validate_response_lengths(lengths,graph_caps[step],correction)[0]
    graph_occupancy=occupancy(reference,graph_work,'FULL')
    for row,graph_time in zip(steps,graph['step_seconds']):row['graph_seconds']=graph_time
    elapsed=sum(s['seconds'] for s in steps);work=sum(s['response_tokens'] for s in steps)
    return dict(status='EAGER_EPOCH1_INPUT_BUDGET_UPDATE_AUDITED',policy=policy,seed=0,steps=steps,
        rollout_seconds=elapsed,response_tokens=work,response_tokens_s=work/elapsed,reward=statistics.mean(scores),
        preemptions=sum(a['preemptions'] for a in audits),graph_reference=graph,
        eager_Q_E=eager_occupancy,graph_Q_E=graph_occupancy,
        first_step_starts_from_identical_checkpoint_before_any_new_actor_update=True,
        comparison_scope='Same common actor,seed0,epoch1,input order and graph-safe KV. Both paths train/update normally; output trajectories need not be token-identical. Eager/TQ2 vs FULL/TQ1 is the configured execution-path comparison, not a pure one-bit graph-toggle causal estimate.')


def parse_evaluation(output,endpoint=None):
    manifest=json.loads((ROOT/'heldout_prepared/manifest.json').read_text())
    paths=list((Path(output)/'samples').glob('*.jsonl'))
    if len(paths)!=1:raise RuntimeError('Missing unique held-out output')
    logs=list((Path(output)/'evaluation/logs').glob('*.txt'))
    if len(logs)!=1:raise RuntimeError('Missing held-out serving log')
    text=logs[0].read_text(errors='replace')
    if not re.search(r'\[run\] end_time=\S+ exit_code=0',text):raise RuntimeError('Held-out serving did not exit0')
    if endpoint is not None and len(re.findall('Loaded sharded model checkpoint from '+re.escape(str(endpoint)),text))!=16:
        raise RuntimeError('Trained endpoint load not proved on all16 ranks')
    if 'validation generation end' not in text or re.search(r'step:\d+ - .*actor/lr:',text):
        raise RuntimeError('Missing validation or unintended training in val-only run')
    rows=list(records(paths[0]));expected={r['expected_decoded_input']:r for r in manifest['examples']}
    if len(expected)!=256 or len(rows)!=256 or {r['input'] for r in rows}!=set(expected):
        raise RuntimeError('Held-out prompts/count differ from preregistered256')
    scores={}
    for row in rows:
        case=expected[row['input']]
        if str(row['gts'])!=str(case['ground_truth']):raise RuntimeError('Held-out ground truth changed')
        score=float(row['score'])
        if not 0<=score<=1:raise RuntimeError('Invalid held-out score')
        scores[case['prompt_sha256']]=score
    return dict(status='HELDOUT256_ENDPOINT_SCORED',questions=256,mean_score=statistics.mean(scores.values()),
        empty_answers=sum(not r['output'].strip() for r in rows),
        scores_by_prompt_sha256=scores,subset_sha256=manifest['input_sha256'][str(HELDOUT)],
        scope='Greedy n1 on a prespecified disjoint local test subset; not full benchmark, final convergence or training-seed equivalence.')
