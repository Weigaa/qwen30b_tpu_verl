#!/usr/bin/env python3
"""Bounded first-decode reproduction, all serving capacities unchanged."""
import argparse
import json
import os
from pathlib import Path
import subprocess

import run_n4_sidecar_models as m

INPUT=Path('/data/adafloor_graph_results/mlsys_n4_sidecar_internal_dp_20261027T041500Z/preflight/qwen3_8b/graph/launch.json')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--boundaries',action='store_true')
    parser.add_argument('--deterministic-comm',action='store_true')
    args=parser.parse_args();root=args.output.resolve();root.mkdir(parents=True,exist_ok=False)
    env=os.environ.copy();env.update(json.loads(INPUT.read_text()))
    env.update(VERL_SIDECAR_MAX_TOKENS='2',VLLM_ASCEND_FULL_GRAPH_PARITY_STEPS='1',
        VLLM_ASCEND_FORCE_TORCH_NPU_ADD_RMS_NORM='1',
        VLLM_ASCEND_FULL_GRAPH_PREFORWARD_PARITY='1',
        VERL_SIDECAR_OUTPUT_FILE=str(root/'outputs.jsonl'),VERL_SIDECAR_LOG_FILE=str(root/'infer.log'),
        VERL_SIDECAR_STATE_DIR=str(root/'state'),VERL_SIDECAR_STOP_FILE=str(root/'stop'),
        XDG_CACHE_HOME=str(root/'cache'),ASCEND_CACHE_PATH=str(root/'cache/ascend'))
    if args.boundaries:env['VLLM_ASCEND_GRAPH_BOUNDARY_PROBE']=str(root/'boundaries')
    if args.deterministic_comm:
        env.update(HCCL_DETERMINISTIC='true',LCCL_DETERMINISTIC='1')
    m.c.write(root/'LAUNCH.json',dict(env={k:v for k,v in env.items() if k.startswith(('VLLM_','VERL_','PYTHONPATH','HCCL_','LCCL_'))},
        diagnostic_only=True,serving_capacities_unchanged=True,max_new_tokens=2))
    with (root/'driver.log').open('x') as log:
        rc=subprocess.run(['bash',str(m.SECONDARY/'internal/run_elastic_sidecar_infer.sh')],
            env=env,cwd=m.SECONDARY,stdout=log,stderr=subprocess.STDOUT).returncode
    m.c.write(root/'RESULT.json',dict(exit_code=rc,diagnostic_only=True))
    print('exit_code',rc,flush=True)


if __name__=='__main__':main()
