#!/usr/bin/env python3
"""CPU preflight of the entire fixed code workload, never a score-based subset."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time
from code_task import SOURCE,COMMON,DATA,CODE,TRAIN,REPAIR_TRACE
import os


def command(output):
    args=['--baseline-dir',str(COMMON/'epoch_000_mode0_probe'),'--require-compact-history',
        '--train-file',str(TRAIN),'--tokenizer-path','/data/Qwen3-30B-A3B',
        '--require-prompt-token-lengths','--steps','5','--batch-size','32','--responses-per-prompt','16',
        '--aclgraph-bucket-count','8','--dataset-fraction','1.0','--max-rank-peak-tokens','380800',
        '--adaptive-floor','--min-adaptive-floor','4','--floor-kv-caps','2:0,4:133120,8:235264,16:380800',
        '--rank-matching-policy','release_area','--kv-block-size-tokens','128',
        '--active-peak-safety-factor','1.16','--max-response-len','16384',
        '--tail-guard-ratio-quantile','0.95','--tail-guard-ratio-window','3',
        '--tail-guard-default-ratio','1.20','--tail-guard-min-cap','4096','--tail-guard-round-to','512',
        '--max-cross-step-repair-swaps','32','--repair-candidate-limit','8',
        '--timing-output',str(output/'timing.json')]
    for option,file in [('train','length_sorted_train.parquet'),('plan','length_sorted_rank_plan.json'),
                        ('summary','length_sorted_rank_plan_summary.json'),('oracle','length_sorted_length_oracle.json')]:
        args+=['--output-'+option,str(output/file)]
    return [str(CODE/'offline_bin/python3'),str(SOURCE/'tools/build_mode1_length_sorted_e2e_plan.py'),*args]


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    output=args.output;output.mkdir(parents=True,exist_ok=False);cmd=command(output)
    files=[TRAIN,REPAIR_TRACE,COMMON/'epoch_000_mode0_probe/offline_planning_history.json',
           SOURCE/'tools/build_mode1_length_sorted_e2e_plan.py',CODE/'cached_planner.py',CODE/'offline_bin/python3']
    evidence=dict(command=cmd,input_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
        scope='CPU-only all160 fixed prompts, not a new NPU run. Exact peak cache only, repair search budget32. No discarded/replaced prompts or increased KV.')
    (output/'LAUNCH.json').write_text(json.dumps(evidence,indent=2)+'\n');begin=time.time()
    env=os.environ.copy();env['FAIR_CODE_REPAIR_TRACE']=str(REPAIR_TRACE)
    env['FAIR_CODE_REPAIR_TRACE_SHA256']=evidence['input_sha256'][str(REPAIR_TRACE)]
    with (output/'planner.log').open('x') as log:
        rc=subprocess.run(cmd,cwd=SOURCE,env=env,stdout=log,stderr=subprocess.STDOUT).returncode
    evidence=dict(exit_code=rc,wall_seconds=time.time()-begin)
    if rc==0:
        summary=json.loads((output/'length_sorted_rank_plan_summary.json').read_text())
        if len(summary)!=5 or any(not r['feasible'] or r['max_adjusted_rank_peak_tokens']>r['kv_cap'] for r in summary):
            raise RuntimeError('Planner returned unsafe/incomplete fixed code workload')
        evidence['floors']=[r['selected_floor'] for r in summary]
        evidence['status']='ALL160_CODE_PROMPTS_PLANNED_FEASIBLY_CPU_ONLY'
    (output/'RESULT.json').write_text(json.dumps(evidence,indent=2)+'\n');print(json.dumps(evidence),flush=True)
    raise SystemExit(rc)


if __name__=='__main__':main()
