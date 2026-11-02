#!/usr/bin/env python3
"""Queue exactly one two-arm controlled v0.14 FULL pair after explicit P4 OFF."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import time

ROOT=Path('/data/adafloor_graph_results/mlsys_v014_supplement_20261023T180000Z')
CODE=Path(__file__).resolve().parent
SOURCE=Path('/workspace/cann-recipes-train/llm_rl/qwen3_014/qwen3_main')
HF=Path('/workspace/adafloor_revision_results/frontier_common_actor_hf_v2')
WORKLOAD=ROOT/'quota_prepared/workload.json'
OUT=ROOT/'quota_runtime'
PRIOR_PID=3036230


def state(value,**extra):
    p=ROOT/'QUOTA_QUEUE_STATE.json';tmp=p.with_suffix('.tmp')
    tmp.write_text(json.dumps(dict(state=value,time=time.time(),**extra),indent=2)+'\n');tmp.replace(p)


def pid_signature():
    try:return Path(f'/proc/{PRIOR_PID}/stat').read_text().rsplit(') ',1)[1].split()[19]
    except FileNotFoundError:return None


def idle():
    while True:
        info=subprocess.run(['npu-smi','info'],capture_output=True,text=True)
        if info.returncode or 'Process id' not in info.stdout:raise RuntimeError('No authoritative NPU status')
        footer=info.stdout.rsplit('Process id',1)[1]
        if not re.findall(r'^\|\s+\d+\s+\d+\s+\d+\s+',footer,re.M):return info.stdout
        state('WAITING_FOR_IDLE_NPUS');time.sleep(15)


def build_env(arm):
    spec=importlib.util.spec_from_file_location('quota_env_source',SOURCE/'tools/run_mlsys_fig6.py')
    suite=importlib.util.module_from_spec(spec);spec.loader.exec_module(suite)
    env=suite.arm_env(arm,0,'planned_f4')
    env.update({k:str(v) for k,v in dict(
        PYTHONPATH=SOURCE,ASCEND_RT_VISIBLE_DEVICES=','.join(map(str,range(16))),
        VLLM_DP_SIZE=16,VLLM_ENABLE_EXPERT_PARALLEL=1,VLLM_ENABLE_MC2=1,
        VLLM_ASCEND_ENABLE_NZ=0,VLLM_MOE_PATTERN_STATS=0,VLLM_MOE_STATS_TIMING=0,
        VLLM_ROLLOUT_WEIGHT_VALUE_AUDIT=1,
        USE_ALLTOALL_OVERLAP=1,ALL_TO_ALL_RESHARD=1,VLLM_ENABLE_FIX_ROUTE=0,HCCL_OP_EXPANSION_MODE='AIV',
        TASK_QUEUE_ENABLE=1,VLLM_ENABLE_GRAPH_MODE=0,VLLM_USE_V1=1,
        VLLM_ASCEND_REGISTER_CUSTOM_MODELS=1,VLLM_ASCEND_USE_NATIVE_QWEN3_MOE=0,
        VLLM_ASCEND_ELASTIC_EXECUTION_MODE=1,VLLM_ASCEND_ENABLE_ELASTIC_PARALLEL_SHRINK=1,
        VLLM_ASCEND_ELASTIC_MOE_MODE='lossless',VLLM_ASCEND_ELASTIC_MIN_COMPUTE_GROUP_SIZE=4,
        VLLM_ASCEND_SHRINK_AWARE_ENABLE=1,VLLM_ASCEND_SHRINK_AWARE_MODE='staged',
        VLLM_ASCEND_SHRINK_AWARE_STAGES='8,4',VLLM_ASCEND_SHRINK_AWARE_SURVIVOR_POLICY='manual',
        VLLM_ASCEND_SHRINK_AWARE_TARGET_POLICY='planned',
        VLLM_ASCEND_SHRINK_AWARE_INTERMEDIATE_RANKS='8,9,10,11,12,13,14,15',
        VLLM_ASCEND_SHRINK_AWARE_FINAL_RANKS='12,13,14,15',
        VLLM_ASCEND_SHRINK_AWARE_STAGE_RANKS='[[8,9,10,11,12,13,14,15],[12,13,14,15]]',
        VLLM_ASCEND_MODE1_PARITY_CURRENT_FLOOR=4,VLLM_ASCEND_MODE1_PARITY_CURRENT_KV_TOKENS=133120,
        VLLM_ASCEND_MODE1_PARITY_PRECREATE_PLANNED_FLOOR_GROUPS=1,
        # Match the successful v0.14 P4 path:32 physical rows,8 EP16 valid
        # primary rows. Build planned groups at the actual transition. The
        # optional pre-KV precreator incorrectly requires32 valid rows after
        # profile has deliberately narrowed the logical map to8.
        VLLM_ASCEND_MODE1_PRECREATE_PLANNED_FLOOR_GROUPS=0,
        VLLM_ASCEND_MODE1_PRECREATE_COMM_WARMUP=0,
        VLLM_ASCEND_MODE1_PARITY_CACHE_PLANNED_FLOOR_GROUPS=1,
        VLLM_ASCEND_MODE1_PARITY_PREFILL_PLANNED_EXPERT_SLOTS=1,
        VLLM_ASCEND_MODE1_PARITY_PRECREATE_COMM_CACHE=1,
        VLLM_ASCEND_MODE1_PARITY_PRECREATE_DISPATCH_WARMUP=1,
        VLLM_ASCEND_MODE1_PROFILE_MAX_TOKENS=2048,VLLM_ASCEND_MODE1_PROFILE_AVOID_MC2_BOUNDARY=1,
        VLLM_ROLLOUT_NATIVE_SLEEP_MODE=1,VLLM_ROLLOUT_SLEEP_LEVEL=2,
        VLLM_ROLLOUT_NATIVE_SLEEP_GRAPH_LIFECYCLE=1,VLLM_ASCEND_NATIVE_SLEEP_PARK_IDLE=1,
        VERL_SIDECAR_ENABLE=0,HCCL_IF_BASE_PORT=12000,VERL_HCCL_IF_BASE_PORT_START=12000,
        VERL_HCCL_IF_BASE_PORT_BLOCK=16384,OMP_NUM_THREADS=2,
        ASCEND_GLOBAL_EVENT_ENABLE=0,ASCEND_GLOBAL_LOG_LEVEL=3,ASCEND_SLOG_PRINT_TO_STDOUT=0,
    ).items()})
    for key in ('RANK','WORLD_SIZE','LOCAL_RANK','LOCAL_WORLD_SIZE','GROUP_RANK','GROUP_WORLD_SIZE','ASCEND_DEVICE_ID'):
        env.pop(key,None)
    for key in ('XDG_CACHE_HOME','TORCHAIR_CACHE_HOME','TRITON_CACHE_DIR','ASCEND_CACHE_PATH','ASCEND_WORK_PATH'):
        Path(env[key]).mkdir(parents=True,exist_ok=True)
    return env,suite


def main():
    original=pid_signature()
    while original is not None and pid_signature()==original:
        state('WAITING_FOR_P4_EXPLICIT_OFF',pid=PRIOR_PID);time.sleep(15)
    prior=ROOT/'P4_EXPLICIT_PAIR_RESULT.json'
    if not prior.is_file() or json.loads(prior.read_text())['status']!='EXPLICIT_OFF_ON_PAIR_COMPLETE':
        raise RuntimeError('Prior controlled pair not complete; do not start or retry a GPU job')
    if OUT.exists():raise RuntimeError('Refusing existing controlled output')
    OUT.mkdir();(OUT/'NPU_BEFORE.txt').write_text(idle())
    audit=json.loads((HF/'tensor_parity_audit.json').read_text())
    if audit['status']!='ALL_EXPORTED_TENSORS_EQUAL_ORIGINAL_DCP':raise RuntimeError('No common-weight parity audit')
    state('CHECKING_FROZEN_WEIGHTS')
    for name,expected in audit['export_shard_sha256'].items():
        with Path(name).open('rb') as stream:actual=hashlib.file_digest(stream,'sha256').hexdigest()
        if actual!=expected:raise RuntimeError('Frozen HF weight shard changed')
    code_sha={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in
              (Path(__file__),CODE/'run_quota_v014_rank.py',CODE/'audit_quota_v014.py',WORKLOAD)}
    (OUT/'INPUT_CODE_SHA256.json').write_text(json.dumps(code_sha,indent=2)+'\n')
    for policy in ('min_skew','release_area'):
        arm=OUT/policy;arm.mkdir();env,suite=build_env(arm);idle()
        initial=suite.execution_digest()
        command=['python3','-m','torch.distributed.run','--standalone','--nproc_per_node=16',
            str(CODE/'run_quota_v014_rank.py'),'--source',str(SOURCE),'--hf-root',str(HF),
            '--workload',str(WORKLOAD),'--policy',policy,'--output',str(arm/'ranks')]
        (arm/'LAUNCH.json').write_text(json.dumps(dict(command=command,execution_sha256=initial,
            env={k:v for k,v in env.items() if k.startswith(('VLLM_','VERL_','ASCEND_','HCCL_','PYTORCH_')) or k=='TASK_QUEUE_ENABLE'}),indent=2)+'\n')
        state('CONTROLLED_ARM_RUNNING',policy=policy)
        # Source the installed NPU libraries without changing any system package.
        script='set -e\nset +u\nsource /usr/local/Ascend/ascend-toolkit/set_env.sh\nsource /usr/local/Ascend/nnal/asdsip/set_env.sh\nsource /usr/local/Ascend/nnal/atb/set_env.sh --cxx_abi=1\nset -u\nexec "$@"'
        with (arm/'primary.log').open('x') as log:
            rc=subprocess.run(['bash','-c',script,'quota-arm',*command],env=env,cwd=SOURCE,
                              stdout=log,stderr=subprocess.STDOUT).returncode
        (arm/'DRIVER_RESULT.json').write_text(json.dumps(dict(exit_code=rc))+'\n')
        if rc:raise RuntimeError(f'Controlled arm{policy} failed; no retries')
        if suite.execution_digest()!=initial:raise RuntimeError('Source changed during controlled run')
        for p,expected in code_sha.items():
            if hashlib.sha256(Path(p).read_bytes()).hexdigest()!=expected:raise RuntimeError('Controlled code/input changed')
    state('AUDITING_PAIR')
    result=subprocess.run(['python3',str(CODE/'audit_quota_v014.py'),'--root',str(OUT),'--workload',str(WORKLOAD)],
                          stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
    (OUT/'audit.log').write_text(result.stdout)
    if result.returncode:raise RuntimeError('Controlled pair audit failed')
    state('PAIR_COMPLETE_Q_AND_E_AUDITED')


if __name__=='__main__':
    try:main()
    except Exception as error:state('STOPPED_WITH_ERROR',error=repr(error));raise
