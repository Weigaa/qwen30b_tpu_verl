#!/usr/bin/env python3
"""Bounded same-input eager/graph shadow check after free-batch parity differs.

Never feeds eager results into generation. Both existing runner diagnostics
are enabled: eager-before-graph (KV-write boundary) and eager-after-graph
(strict all-valid-row argmax + hidden-state tolerance, fails on mismatch).
"""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess

REPO=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--input-run',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();root=args.output.resolve();root.mkdir(parents=True,exist_ok=False)
    env={k:v for k,v in os.environ.items() if not k.startswith(('VLLM_','VERL_','ADAFLOOR_'))}
    env.update(json.loads((args.input_run/'graph/launch.json').read_text()))
    env.update(VLLM_ASCEND_FORCE_TORCH_NPU_ADD_RMS_NORM='1',
        VLLM_ASCEND_FULL_GRAPH_PARITY_STEPS='4096',
        VLLM_ASCEND_FULL_GRAPH_PREFORWARD_PARITY='1',
        VERL_SIDECAR_OUTPUT_FILE=str(root/'outputs.jsonl'),VERL_SIDECAR_LOG_FILE=str(root/'infer.log'),
        VERL_SIDECAR_STATE_DIR=str(root/'state'),VERL_SIDECAR_STOP_FILE=str(root/'stop'),
        XDG_CACHE_HOME=str(root/'cache'),ASCEND_CACHE_PATH=str(root/'cache/ascend'))
    (root/'launch.json').write_text(json.dumps({k:v for k,v in env.items()
        if k.startswith(('VLLM_','VERL_'))},indent=2)+'\n')
    with (root/'driver.log').open('w') as log:
        rc=subprocess.run(['bash',str(REPO/'internal/run_elastic_sidecar_infer.sh')],env=env,
            cwd=REPO,stdout=log,stderr=subprocess.STDOUT).returncode
    text=(root/'infer.log').read_text(errors='replace')
    checks=re.findall(r'FULL graph shadow parity dp_rank=0 check=(\d+) mismatch=(\w+) hidden_max_abs=([^ ]+)',text)
    prechecks=re.findall(r'FULL pre-forward logits parity: max_abs=([^ ]+) finite_mask_equal=(\w+) argmax_equal=(\w+)',text)
    record=dict(primary_exit_code=rc,post_checks=len(checks),pre_checks=len(prechecks),
        post_mismatches=sum(row[1]!='False' for row in checks),
        pre_argmax_mismatches=sum(row[2]!='True' for row in prechecks),
        max_hidden_abs=max((float(row[2]) for row in checks),default=None),
        max_logits_abs=max((float(row[0]) for row in prechecks),default=None))
    if rc or not checks or not prechecks:
        record['status']='FAILED_OR_INCOMPLETE'
    elif record['post_mismatches'] or record['pre_argmax_mismatches']:
        record['status']='SAME_INPUT_GRAPH_MISMATCH'
    else:record['status']='ALL_SAME_PADDED_INPUT_SHADOW_CHECKS_PASSED'
    (root/'SHADOW_RESULT.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record,indent=2),flush=True)
    if record['status']!='ALL_SAME_PADDED_INPUT_SHADOW_CHECKS_PASSED':raise SystemExit(1)


if __name__=='__main__':main()
