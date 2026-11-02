#!/usr/bin/env python3
"""One coordinated DP x TP world using v0.14's external_launcher backend.

Follows examples/offline_inference/torchrun_dp_example.py, not independent
LLM replicas. Each TP worker schedules identical inputs for its DP shard.
Only TP leaders' outputs/states are aggregated; followers keep private state.
"""
import argparse
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys


def worker_env(base, rank):
    env=dict(base)
    size=int(env['VERL_SIDECAR_DATA_PARALLEL_SIZE'])
    tp=int(env['VERL_SIDECAR_TENSOR_PARALLEL_SIZE'])
    dp_rank=rank//tp
    env.update(VLLM_DP_SIZE=str(size), VLLM_DP_RANK=str(dp_rank),
        VLLM_DP_RANK_LOCAL=str(dp_rank),
        VLLM_DP_MASTER_IP=env['VERL_SIDECAR_MASTER_ADDR'],
        VLLM_DP_MASTER_PORT=env['VERL_SIDECAR_MASTER_PORT'],
        VLLM_ENABLE_V1_MULTIPROCESSING='0',
        RANK=str(rank),LOCAL_RANK=str(rank),WORLD_SIZE=str(size*tp),
        LOCAL_WORLD_SIZE=str(size*tp),MASTER_ADDR=env['VERL_SIDECAR_MASTER_ADDR'],
        MASTER_PORT=env['VERL_SIDECAR_MASTER_PORT'],
        VERL_SIDECAR_OFFLINE_SPMD_WORKER='1',
        VERL_SIDECAR_EXTERNAL_LAUNCHER='1',
        VERL_SIDECAR_SHARD_INDEX=str(dp_rank), VERL_SIDECAR_NUM_SHARDS=str(size))
    # The reference global1024 bound is128 local requests on each DP worker.
    global_limit=int(env.get('VERL_SIDECAR_MAX_PROMPTS','1024'))
    local_limit=min(int(env.get('VERL_SIDECAR_MAX_PROMPTS_PER_REPLICA','128')),
                    math.ceil(global_limit/size))
    env['VERL_SIDECAR_MAX_PROMPTS_PER_DEVICE']=str(local_limit)
    if rank%tp==0:
        # The outer lease watcher must also be able to merge durable outputs
        # after enforcing its deadline (even if this launcher is killed).
        env['VERL_SIDECAR_OUTPUT_FILE']=base['VERL_SIDECAR_OUTPUT_FILE']+f'.shard{dp_rank}'
    else:
        private=Path(base['VERL_SIDECAR_STATE_DIR'])/'tp_followers'/f'rank{rank}'
        env['VERL_SIDECAR_STATE_DIR']=str(private)
        env['VERL_SIDECAR_OUTPUT_FILE']=str(private/'outputs.jsonl')
    return env


def merge_outputs(target, outputs):
    """Publish all TP-leader shards atomically, never a truncated aggregate."""
    target=Path(target);target.parent.mkdir(parents=True,exist_ok=True)
    temporary=target.with_name(target.name+f'.merge.{os.getpid()}.tmp')
    try:
        with temporary.open('wb') as out:
            for path in outputs:
                if path.exists():
                    with path.open('rb') as source:
                        import shutil
                        shutil.copyfileobj(source,out)
            out.flush();os.fsync(out.fileno())
        temporary.replace(target)
        descriptor=os.open(target.parent,os.O_RDONLY|os.O_DIRECTORY)
        try:os.fsync(descriptor)
        finally:os.close(descriptor)
    finally:
        temporary.unlink(missing_ok=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--script',type=Path,required=True)
    args=parser.parse_args();base=os.environ.copy()
    size=int(base['VERL_SIDECAR_DATA_PARALLEL_SIZE'])
    tp=int(base['VERL_SIDECAR_TENSOR_PARALLEL_SIZE'])
    if size<=1:
        raise RuntimeError('Explicit external DP launcher requires DP>1')
    children=[];outputs=[]
    def stop(signum, frame):
        for child in children:
            if child.poll() is None:child.send_signal(signum)
    signal.signal(signal.SIGTERM,stop)
    try:
        for rank in range(size*tp):
            env=worker_env(base,rank)
            if rank%tp==0:outputs.append(Path(env['VERL_SIDECAR_OUTPUT_FILE']))
            print(json.dumps(dict(event='sidecar_offline_dp_frontend',rank=rank,size=size,
                tp=tp,dp_rank=rank//tp,global_world_size=size*tp,
                devices=env['ASCEND_RT_VISIBLE_DEVICES'],ep=env['VERL_SIDECAR_ENABLE_EXPERT_PARALLEL_EFFECTIVE'])),flush=True)
            children.append(subprocess.Popen([sys.executable,'-u',str(args.script)],env=env))
        codes=[child.wait() for child in children]
    finally:
        for child in children:
            if child.poll() is None:child.terminate()
        merge_outputs(base['VERL_SIDECAR_OUTPUT_FILE'],outputs)
    raise SystemExit(next((code for code in codes if code),0))


if __name__=='__main__':main()
