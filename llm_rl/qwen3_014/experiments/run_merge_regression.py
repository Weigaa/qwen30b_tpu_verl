#!/usr/bin/env python3
"""Short real-NPU pre/post-merge regression, not a paper throughput run."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import tarfile
import time

ROOT = Path(__file__).resolve().parents[1]
MAIN = ROOT/'qwen3_main'
ARCHIVES = Path('/data/adafloor_code_archives/qwen3_layout_before_20261101')
COMMON = Path('/data/adafloor_shared_state/common_epoch0_probe_gpu09_kv380800_permanent')
CHECKPOINT = COMMON/'epoch_000_mode0_probe/checkpoints/qwen3moe_for_eagle3/global_step_5'
CASES = ['baseline_eager', 'baseline_graph', 'natural_f2_graph_sidecar']


def sha(p):
    with Path(p).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def write(p, value):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(value, indent=2)+'\n')
    tmp.replace(p)


def state(root, phase, **extra):
    value = dict(state=phase, utc=datetime.now(timezone.utc).isoformat(), **extra)
    write(root/'STATE.json', value)
    print(json.dumps(value), flush=True)


def suite(repo):
    spec = importlib.util.spec_from_file_location('merge_suite_'+repo.name, repo/'tools/run_mlsys_fig6.py')
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def prepare_short_plan(root):
    """Literal known-cap fixture, not a prediction or paper planner result."""
    plan=[]
    for step in [1,2]:
        offset=32*(step-1)
        plan.append(dict(step=step,selected_floor=2,theoretical_floor=2,
            kv_cap=131072,tail_guard_enabled=False,
            shrink_stages=[8,4,2],
            stage_survivor_ranks=[list(range(8,16)),list(range(12,16)),[14,15]],
            intermediate_survivor_ranks=list(range(8,16)),final_survivor_ranks=[14,15],
            rank_to_dataset_item_idx={str(rank):[offset+2*rank,offset+2*rank+1] for rank in range(16)},
            aclgraph_capture_sizes=[32],fixture='explicit_short_decode_caps_not_paper_prediction'))
    path=root/'short_natural_plan.json'
    if path.exists():
        assert json.loads(path.read_text())==plan
    else:
        write(path,plan)
    return path


def prepare(root):
    import pyarrow.parquet as pq
    root.mkdir(exist_ok=False, parents=True)
    assert CHECKPOINT.is_dir() and (COMMON/'DO_NOT_DELETE_COMMON_EPOCH0_CHECKPOINT').is_file()
    data = root/'inputs.parquet'
    pq.write_table(pq.read_table('/data/deepscaler/train.parquet').slice(0, 64), data)
    prepare_short_plan(root)
    references = root/'reference_sources'
    references.mkdir()
    for name in ['qwen3', 'qwen3_graph_sidecar_v014']:
        with tarfile.open(ARCHIVES/(name+'.tar.gz')) as archive:
            archive.extractall(references, filter='data')
    write(root/'PROTOCOL.json', dict(
        scope='Short functional and descriptive performance regression of source consolidation',
        current=str(MAIN), old_main=str(references/'qwen3'),
        old_sidecar=str(references/'qwen3_graph_sidecar_v014'), checkpoint=str(CHECKPOINT),
        inputs_sha256=sha(data), primary_model='Qwen3-30B-A3B', npu_count=16,
        batch_size=32, samples_per_prompt=16, steps=2, cases=CASES,
        baseline_max_tokens=384, natural_validation_caps=[64,128,256,1024],
        artificial_caps=True, ignore_eos=True, greedy=True,
        actual_training_updates=False, initial_checkpoint_loaded=True,
        native_sleep=2, lifecycle=1, sidecar_model='Qwen2.5-1.5B-Instruct',
        sidecar_window_gate_disabled_for_functional_probe=True,
        paper_quality_or_throughput_claim=False,
        references_restored_outside_implementation_tree=True,
        archives={name:sha(ARCHIVES/(name+'.tar.gz')) for name in ['qwen3','qwen3_graph_sidecar_v014']}))


def launch_config(root, case, version):
    elastic = case.startswith('natural')
    graph = case != 'baseline_eager'
    repo = MAIN if version == 'merged' else root/'reference_sources'/('qwen3_graph_sidecar_v014' if elastic else 'qwen3')
    arm = root/case/version
    arm_index = 2*CASES.index(case) + int(version == 'merged')
    primary_port = 10000 + 7000*arm_index
    if elastic:
        # Use the validated low-port envelope, outside Linux's ephemeral
        # range. Actor/rollout/ref add phase offsets of 0/4096/8192.
        primary_port = 12000 if version == 'reference' else 20000
    m = suite(repo)
    env = m.arm_env(root, 0, 'natural_f2' if elastic else 'vanilla')
    env.update({k:str(v) for k,v in dict(
        OUTPUT_ROOT=arm, OUTPUT_SUBDIR='execution', CONFIG_DIR=repo/'verl/trainer/config',
        PYTHONPATH=str(repo)+':'+os.environ.get('PYTHONPATH',''),
        TRAIN_FILE=root/'inputs.parquet', TEST_FILE=root/'inputs.parquet',
        TRAIN_BATCH_SIZE=32, ACTOR_PPO_MINI_BATCH_SIZE=32, DATASET_FRACTION=1,
        MAX_RESPONSE_LENGTH=1024 if elastic else 384, ROLLOUT_N=16, ROLLOUT_MAX_NUM_SEQS=32,
        TRAINER_TOTAL_EPOCHS=1, TRAINER_TOTAL_TRAINING_STEPS=2,
        SAVE_CKPT_ENABLE=0, GIT_TRACK_RUN_LOGS=0, VERL_RESET_TRAINER_PROGRESS_AFTER_RESUME=1,
        VLLM_EPOCH_LENGTH_REGROUP_ENABLE=0, VLLM_EAGER_BASELINE_NO_RESAMPLE=0,
        VLLM_ROLLOUT_EARLY_STOP_ENABLE=0, VLLM_ROLLOUT_SHRINK_AWARE_SHORT_STEP_CAP_ENABLE=0,
        ROLLOUT_ENFORCE_EAGER='False' if graph else 'True',
        TASK_QUEUE_ENABLE=1 if graph else 2, VLLM_ROLLOUT_TASK_QUEUE_ENABLE=1 if graph else 2,
        VLLM_ENABLE_GRAPH_MODE=0, VLLM_ASCEND_ELASTIC_ACLGRAPH=int(graph),
        VLLM_ROLLOUT_ASYNC_SCHEDULING='false',
        VERL_PAIRED_REQUEST_SAMPLING_SEEDS=1, HCCL_DETERMINISTIC='true', LCCL_DETERMINISTIC=1,
        VLLM_ASCEND_REGISTER_CUSTOM_MODELS=1, VLLM_ASCEND_USE_NATIVE_QWEN3_MOE=0,
        VLLM_ASCEND_SHRINK_AWARE_ENABLE=int(elastic),
        VLLM_ASCEND_ENABLE_ELASTIC_PARALLEL_SHRINK=int(elastic),
        VLLM_ASCEND_ELASTIC_EXECUTION_MODE=1 if elastic else 0,
        VLLM_ASCEND_ELASTIC_MIN_COMPUTE_GROUP_SIZE=2 if elastic else 16,
        VLLM_ASCEND_ELASTIC_MOE_MODE='lossless',
        VLLM_ASCEND_SHRINK_AWARE_MODE='staged',
        VLLM_ASCEND_SHRINK_AWARE_STAGES='8,4,2',
        VLLM_ASCEND_SHRINK_AWARE_SURVIVOR_POLICY='manual',
        VLLM_ASCEND_SHRINK_AWARE_TARGET_POLICY='natural',
        VLLM_ASCEND_SHRINK_AWARE_STAGE_RANKS='[[8,9,10,11,12,13,14,15],[12,13,14,15],[14,15]]',
        VLLM_ASCEND_SHRINK_AWARE_INTERMEDIATE_RANKS='8,9,10,11,12,13,14,15',
        VLLM_ASCEND_SHRINK_AWARE_FINAL_RANKS='14,15',
        VLLM_ASCEND_SHRINK_AWARE_MIN_WINDOW_SECONDS=0,
        VERL_ELASTIC_TAIL_VALIDATE_LEVEL_TOKENS='64,128,256,1024' if elastic else '384',
        VERL_LOGGING_LEVEL='INFO', RAY_TMPDIR='/tmp/mrg_'+str(os.getpid())+'_'+str(arm_index),
        VERL_HCCL_IF_BASE_PORT_START=primary_port, VERL_HCCL_IF_BASE_PORT_BLOCK=16384,
        XDG_CACHE_HOME=arm/'cache', ASCEND_CACHE_PATH=arm/'cache/ascend',
        TORCHAIR_CACHE_HOME=arm/'cache/torchair', TRITON_CACHE_DIR=arm/'cache/triton',
        ASCEND_WORK_PATH=arm/'cache/work',
        VERL_SIDECAR_ENABLE=int(elastic), VERL_SIDECAR_DIRECT_STEP_EVENTS=int(elastic),
    ).items()})
    # Phase-specific safety reserves match the validated native floor2 path.
    if elastic:
        for suffix in ['CUSTOM_MODE1_KV_MATERIALIZE_HEADROOM_BYTES','KV_CACHE_INIT_HEADROOM_BYTES',
                       'POST_SHRINK_MOE_DISPATCH_HEADROOM_BYTES','POST_SHRINK_MOE_DISPATCH_LOW_FLOOR_HEADROOM_BYTES',
                       'POST_SHRINK_PREFILL_ALLTOALL_HEADROOM_BYTES','POST_RESTORE_DP_HEADROOM_BYTES',
                       'POST_RESTORE_EP_HEADROOM_BYTES','POST_RESTORE_MOE_DISPATCH_HEADROOM_BYTES',
                       'FIRST_LIVE_PREFILL_HEADROOM_BYTES','FIRST_LIVE_PREFILL_LOW_FLOOR_HEADROOM_BYTES',
                       'EXTRA_ELASTIC_SAFETY_HEADROOM_BYTES','FLOOR_PREALLOC_HEADROOM_SAFETY_BYTES']:
            env['VLLM_ASCEND_'+suffix]='0'
        side = arm/'sidecar'
        env.update({k:str(v) for k,v in dict(
            ADAFLOOR_GRAPH_SIDECAR_ENABLE=1, VERL_SIDECAR_REQUIRE_ACTIVE_LEASE_BEFORE_RESTORE=1,
            VERL_SIDECAR_REQUIRE_SHRINK_QUORUM=1, VERL_SIDECAR_SHRINK_QUORUM_SIZE=16,
            VERL_SIDECAR_WORLD_SIZE=16, VERL_SIDECAR_EXPECTED_ACTIVE_RANKS=8,
            VERL_SIDECAR_WINDOW_GATE_ENABLE=0, VERL_SIDECAR_START_ONCE=0,
            VERL_SIDECAR_WATCH_POLL_INTERVAL=.2,
            VERL_SIDECAR_SCRIPT=repo/'internal/run_elastic_sidecar_infer.sh',
            VERL_SIDECAR_LOG_DIR=side, VERL_SIDECAR_LOG_FILE=side/'infer.log',
            VERL_SIDECAR_LEASE_LOG=side/'lease.log', VERL_SIDECAR_OUTPUT_FILE=side/'outputs.jsonl',
            VERL_SIDECAR_STATE_DIR=side/'state', VERL_SIDECAR_RESTORE_HANDSHAKE_DIR=side/'restore_handshake',
            VERL_SIDECAR_MODEL_PATH='/data/Qwen2.5-1.5B-Instruct', VERL_SIDECAR_PROMPTS_FILE='/data/gsm8k',
            VERL_SIDECAR_DATA_SPLIT='train', VERL_SIDECAR_REPLICA_COUNT=8,
            VERL_SIDECAR_TENSOR_PARALLEL_SIZE=1, VERL_SIDECAR_DATA_PARALLEL_SIZE=1,
            VERL_SIDECAR_MAX_PROMPTS=128, VERL_SIDECAR_MAX_PROMPTS_PER_REPLICA=16,
            VERL_SIDECAR_MAX_PROMPTS_PER_DEVICE=16, VERL_SIDECAR_GENERATE_CHUNK_SIZE=16,
            VERL_SIDECAR_MAX_NUM_SEQS=16, VERL_SIDECAR_MAX_NUM_BATCHED_TOKENS=65536,
            VERL_SIDECAR_MAX_MODEL_LEN=6144, VERL_SIDECAR_MAX_TOKENS=128,
            VERL_SIDECAR_GPU_MEMORY_UTILIZATION=.8,
            VERL_SIDECAR_REPEAT_UNTIL_KILLED=1, VERL_SIDECAR_STREAM_CHECKPOINT=1,
            VERL_SIDECAR_ENFORCE_EAGER=0, VERL_SIDECAR_TASK_QUEUE_ENABLE=1,
            VERL_SIDECAR_GRAPH_AUDIT=1, VERL_SIDECAR_ENGINE_MULTIPROCESSING=1,
            VERL_SIDECAR_CUDAGRAPH_MODE='FULL_DECODE_ONLY',
            VERL_SIDECAR_CUDAGRAPH_CAPTURE_SIZES='[1,2,4,8,16]',
            VERL_MEGATRON_EAGER_WEIGHT_SYNC_GROUP_INIT=0,
            VERL_HCCL_IF_BASE_PORT_START=primary_port,HCCL_IF_BASE_PORT=primary_port,
            VERL_HCCL_IF_BASE_PORT_BLOCK=16384,
        ).items()})
    command = ['bash', str(repo/'internal/wj_train_grpo_qwen30b_a3b_16die_true_weight_regroup.sh'),
        'trainer.resume_mode=resume_path', f'trainer.resume_from_path={CHECKPOINT}',
        'trainer.critic_warmup=999999', 'trainer.save_freq=-1',
        'actor_rollout_ref.actor.optim.lr=0.0','actor_rollout_ref.actor.use_kl_loss=False',
        'algorithm.use_kl_in_reward=False','data.shuffle=False',
        'actor_rollout_ref.rollout.seed=0','actor_rollout_ref.rollout.temperature=0',
        'actor_rollout_ref.rollout.top_k=-1','actor_rollout_ref.rollout.top_p=1.0',
        'actor_rollout_ref.rollout.ignore_eos=True',
        'actor_rollout_ref.rollout.cudagraph_mode='+('FULL_DECODE_ONLY' if graph else 'NONE'),
        'actor_rollout_ref.rollout.cudagraph_capture_sizes=[32]',
        'actor_rollout_ref.rollout.max_model_len=17408',
        '++actor_rollout_ref.rollout.engine_kwargs.vllm.num_gpu_blocks_override='+('810' if elastic else '2975'),
        '++actor_rollout_ref.actor.megatron.override_transformer_config.moe_shared_expert_overlap=False',
        'actor_rollout_ref.rollout.shrink_aware.enable_shrink_aware_scheduling='+str(elastic).lower(),
        'actor_rollout_ref.rollout.shrink_aware.shrink_aware_mode=staged',
        'actor_rollout_ref.rollout.shrink_aware.shrink_stages=[8,4,2]',
        'actor_rollout_ref.rollout.shrink_aware.stage_survivor_ranks=[[8,9,10,11,12,13,14,15],[12,13,14,15],[14,15]]',
        'actor_rollout_ref.rollout.shrink_aware.intermediate_survivor_ranks=[8,9,10,11,12,13,14,15]',
        'actor_rollout_ref.rollout.shrink_aware.final_survivor_ranks=[14,15]',
        'actor_rollout_ref.rollout.shrink_aware.survivor_selection_policy=manual',
        'actor_rollout_ref.rollout.shrink_aware.target_policy=natural',
        'actor_rollout_ref.rollout.shrink_aware.assignment_policy=default',
        'actor_rollout_ref.rollout.shrink_aware.min_shrink_window_seconds=0',
        'actor_rollout_ref.rollout.shrink_aware.enable_shrink_aware_logging=true']
    if elastic:
        command += ['actor_rollout_ref.rollout.shrink_aware.assignment_policy=optimized_rank_plan',
                    f'actor_rollout_ref.rollout.shrink_aware.optimized_rank_plan_path={prepare_short_plan(root)}']
    return repo, arm, env, command


def run_arm(root, case, version):
    repo, arm, env, command = launch_config(root, case, version)
    if (arm/'RESULT.json').exists():
        return json.loads((arm/'RESULT.json').read_text())
    arm.mkdir(parents=True, exist_ok=False)
    for k in ['XDG_CACHE_HOME','ASCEND_CACHE_PATH','TORCHAIR_CACHE_HOME','TRITON_CACHE_DIR','ASCEND_WORK_PATH']:
        Path(env[k]).mkdir(parents=True, exist_ok=True)
    write(arm/'LAUNCH.json',dict(repo=str(repo), command=command,env=env))
    state(root,'RUNNING',case=case,version=version,driver=str(arm/'driver.log'))
    watcher = None
    with (arm/'driver.log').open('w') as output:
        process = subprocess.Popen(command,cwd=repo,env=env,stdout=output,stderr=subprocess.STDOUT,start_new_session=True)
        # The existing primary shell already owns the sidecar watcher and
        # its exit trap. A second watcher would race on the same lease files.
        rc=process.wait()
        if watcher:
            try:watcher.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(watcher.pid,signal.SIGTERM)
                watcher.wait(timeout=30)
            watcher_output.close()
    write(arm/'DRIVER_RESULT.json',dict(exit_code=rc))
    if rc:
        raise RuntimeError(f'{case}/{version} failed with rc={rc}, see {arm}/driver.log')
    return summarize_arm(arm,case)


def summarize_arm(arm,case):
    logs=list((arm/'execution/logs').glob('*.txt'))
    assert len(logs)==1
    text=re.sub(r'\x1b\[[0-9;]*[mK]','',logs[0].read_text(errors='replace'))
    assert not re.search(r'OutOfMemoryError|NPU out of memory|RayTaskError|Memory_Allocation_Failure',text)
    times=list(map(float,re.findall(r'=> rollout_output_time_s: ([0-9.]+)',text)))
    audits=[json.JSONDecoder().raw_decode(m[1])[0] for m in re.finditer(r'MODE1_STEP_RUNTIME_AUDIT ([^\n]+)',text)]
    assert len(times)==2 and len(audits)==32
    assert all(a['complete'] and a['preemptions']==0 for a in audits)
    graph=case!='baseline_eager'
    assert all((a.get('full_iterations',0)>0 if graph else a.get('full_iterations',0)==0 and a.get('eager_iterations',0)>0) for a in audits)
    if graph:
        assert all(a['eager_iterations']==a['global_prefill_iterations'] for a in audits)
    outputs=[];work=[]
    for step in [1,2]:
        rows=[json.loads(line) for line in (arm/f'execution/rollout_data/{step}.jsonl').open()]
        assert len(rows)==512 and all(r['output'].strip() for r in rows)
        lengths=list(map(int,(arm/f'execution/rollout_length/length_{step}.txt').read_text().split()))
        assert len(lengths)==512
        work.append(sum(lengths))
        outputs.append([(r['input'],r['output'],r['score']) for r in rows])
    stages=sorted(set(map(int,re.findall(r'Elastic parallel shrink done:.*?dp_size=(\d+)',text))))
    if case.startswith('natural'):
        assert stages==[2,4,8],stages
        assert len(re.findall(r'driver_restore_rpc_start_time=',text))==2
    result=dict(status='PASS',case=case,step_seconds=times,raw_response_mask_tokens=work,
        response_mask_tokens_s=sum(work)/sum(times),full_rank_step_audits=len(audits),
        actual_shrink_floors=stages,outputs_sha256=hashlib.sha256(json.dumps(outputs,sort_keys=True).encode()).hexdigest(),
        native_sleep_handoffs=len(re.findall(r'after native_sleep_before_release',text)),
        graph_decode_not_eager=True if graph else None,preemptions=0,
        artificial_short_caps=True,paper_performance_claim=False)
    write(arm/'RESULT.json',result)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args();root=args.output.resolve()
    try:
        if not args.resume:prepare(root)
        for case in CASES:
            for version in ['reference','merged']:
                run_arm(root,case,version)
        comparisons=[]
        for case in CASES:
            old,new=[json.loads((root/case/v/'RESULT.json').read_text()) for v in ['reference','merged']]
            equal=old['outputs_sha256']==new['outputs_sha256']
            gain=100*(new['response_mask_tokens_s']/old['response_mask_tokens_s']-1)
            comparisons.append(dict(case=case,outputs_exact=equal,throughput_change_percent=gain,reference=old,merged=new))
            assert equal,case+' outputs differ, inspect before claiming correctness'
        write(root/'COMPARISON.json',dict(scope='Short paired NPU regression, no paper throughput or quality claim',comparisons=comparisons))
        state(root,'COMPLETED_REQUIRES_REVIEW',comparisons=[dict(case=c['case'],outputs_exact=c['outputs_exact'],throughput_change_percent=c['throughput_change_percent']) for c in comparisons])
    except Exception as error:
        if root.exists():state(root,'STOPPED_WITH_ERROR',error=repr(error))
        raise


if __name__=='__main__':main()
