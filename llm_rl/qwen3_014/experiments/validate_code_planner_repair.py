#!/usr/bin/env python3
"""Verify fresh optimized code-P4 repair against the existing31-swap gold plan."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import code_task
import prepare_code_plan


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    args.work.mkdir(parents=True,exist_ok=False)
    if args.output.exists():raise RuntimeError('Refusing existing code repair parity result')
    command=prepare_code_plan.command(args.work)
    command[:2]=[sys.executable,'-u',str(code_task.CODE/'optimized_planner.py'),
        str(code_task.SOURCE/'tools/build_mode1_length_sorted_e2e_plan.py')]
    env=os.environ.copy()
    env.pop('FAIR_CODE_REPAIR_TRACE',None);env.pop('FAIR_CODE_REPAIR_TRACE_SHA256',None)
    begin=time.perf_counter()
    with (args.work/'planner.log').open('x') as log:
        subprocess.run(command,cwd=code_task.SOURCE,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    elapsed=time.perf_counter()-begin
    gold=code_task.ROOT/'code_plan_preview_after_multiturn_fix'
    files=('length_sorted_rank_plan.json','length_sorted_rank_plan_summary.json',
           'length_sorted_length_oracle.json')
    hashes={}
    for name in files:
        actual=json.loads((args.work/name).read_text())
        expected=json.loads((gold/name).read_text())
        if actual!=expected:
            raise RuntimeError('Fresh31-swap code repair changed '+name)
        hashes[name]=hashlib.sha256((gold/name).read_bytes()).hexdigest()
    import pyarrow.parquet as pq
    if (pq.read_table(args.work/'length_sorted_train.parquet').to_pylist()!=
            pq.read_table(gold/'length_sorted_train.parquet').to_pylist()):
        raise RuntimeError('Fresh repair changed reordered prompts/labels/identities')
    value=dict(status='FRESH31_SWAP_CODE_P4_GOLD_PLAN_PARITY_PASS',wall_seconds=elapsed,
        optimization_sha256=hashlib.sha256((code_task.CODE/'optimized_planner.py').read_bytes()).hexdigest(),
        fresh_search=True,no_repair_trace_replay=True,all_plan_and_reordered_input_fields_equal=True,
        reference=str(gold),gold_sha256=hashes,
        scope='CPU regression only. No new NPU experiment, seed, answer or reward selection.')
    args.output.write_text(json.dumps(value,indent=2)+'\n')
    print(json.dumps(value),flush=True)


if __name__=='__main__':main()
