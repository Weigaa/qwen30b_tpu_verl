#!/usr/bin/env python3
"""CPU-only complete five-step mathematical-plan parity and timing."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

from cached_planner import load
from optimized_planner import install


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',required=True,type=Path)
    args=p.parse_args()
    if args.output.exists():
        raise RuntimeError('Refusing existing parity result')
    original=load();fast=load();counters=install(fast,progress=False)
    common=Path('/data/adafloor_shared_state/common_epoch0_probe_gpu09_kv380800_permanent/epoch_000_mode0_probe')
    history=original._read_baseline_history([common],5,16,True)
    stats=original._read_ema_baseline_stats(history,0.3)
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained('/data/Qwen3-30B-A3B',local_files_only=True,trust_remote_code=True)
    _,mapped=original._map_stats_to_dataset(Path('/data/deepscaler/train.parquet'),stats,
        0.005,-1,160,prompt_tokenizer=tokenizer)
    options=dict(max_rank_peak_tokens=380800,
        floor_kv_caps={2:131072,4:280576,8:288256,16:380800},adaptive_floor=True,
        min_adaptive_floor=2,force_selected_floor=None,kv_safe_fixed_floor=None,
        rank_matching_policy='release_area',active_peak_safety_factor=1.16,max_response_len=16384,
        max_cross_step_repair_swaps=8,repair_candidate_limit=8,ignore_tail_ties_at_response_cap=False)
    results={};plans={}
    for name,module in (('original',original),('optimized',fast)):
        begin=time.perf_counter()
        solved=module._solve_batches(mapped,32,5,**options)
        elapsed=time.perf_counter()-begin
        plans[name]=[asdict(v) for v in solved]
        results[name]=dict(seconds=elapsed,floors=[v.selected_floor for v in solved],
            plan_sha256=hashlib.sha256(json.dumps(plans[name],sort_keys=True).encode()).hexdigest())
        print(name,results[name],flush=True)
    if plans['original']!=plans['optimized']:
        raise RuntimeError('Full five-step plan mismatch; optimization not approved')
    value=dict(status='FIVE_STEP_NATURAL_F2_EXACT_PLAN_PARITY_PASS',
        workload='Existing mathematical epoch0, all160 prompts/5steps, original Natural-F2 budgets',
        results=results,speedup=results['original']['seconds']/results['optimized']['seconds'],
        optimization_sha256=hashlib.sha256(Path(__file__).with_name('optimized_planner.py').read_bytes()).hexdigest(),
        counters=counters,all_plan_fields_equal=True)
    args.output.write_text(json.dumps(value,indent=2)+'\n')
    print(json.dumps(value),flush=True)


if __name__=='__main__':main()
