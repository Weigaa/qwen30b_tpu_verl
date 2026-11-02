#!/usr/bin/env python3
"""Memoize the unchanged planner's pure peak functions, offline only."""
from functools import lru_cache,wraps
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import sys

SOURCE=Path('/workspace/cann-recipes-train/llm_rl/qwen3_014/qwen3_main/tools/build_mode1_length_sorted_e2e_plan.py')


def replay_groups(stats,batch_size,steps,trace):
    ordered=sorted(stats,key=lambda item:(item.load,item.source_idx))
    if len(ordered)!=batch_size*steps:raise RuntimeError('Repair replay cardinality changed')
    groups=[list(ordered[i*batch_size:(i+1)*batch_size]) for i in range(steps)]
    swaps=list(re.finditer(r'\[mode1 length-sorted e2e repair\] swap=(\d+) step(\d+):src(\d+) <-> step(\d+):src(\d+)',trace))
    if [int(m[1]) for m in swaps]!=list(range(1,len(swaps)+1)):raise RuntimeError('Incomplete repair trace')
    for match in swaps:
        _,left,a,right,b=map(int,match.groups());left-=1;right-=1
        positions=[[i for i,item in enumerate(groups[g]) if item.source_idx==key]
                   for g,key in ((left,a),(right,b))]
        if any(len(p)!=1 for p in positions):raise RuntimeError('Repair replay source identity/group differs')
        x,y=(p[0] for p in positions)
        groups[left][x],groups[right][y]=groups[right][y],groups[left][x]
    return groups,len(swaps)


def install_repair_replay(module,path,wanted):
    text=Path(path).read_text()
    if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=wanted:raise RuntimeError('Repair trace changed')
    if 'solve batches done selected_floors=[8, 16, 16, 16, 16]' not in text:
        raise RuntimeError('Trace did not reach a completed feasible search')
    def resumed(stats,batch_size,steps,**kw):
        groups,count=replay_groups(stats,batch_size,steps,text)
        if count!=31 or kw['max_cross_step_repair_swaps']<count or kw.get('force_selected_floors') is not None:
            raise RuntimeError('Unexpected repair protocol')
        accepted=set(__import__('inspect').signature(module._solve_group).parameters)-{'group'}
        options={k:v for k,v in kw.items() if k in accepted}
        plans=[module._solve_group(group,**options) for group in groups]
        if any(not p.feasible for p in plans) or [p.selected_floor for p in plans]!=[8,16,16,16,16]:
            raise RuntimeError('Replayed all160 layout no longer feasible under unchanged planner')
        # Check each group's last recorded adjusted peak against the fresh
        # solution, not just the previous green log line.
        last={}
        for line in text.splitlines():
            if '[mode1 length-sorted e2e repair]' not in line:continue
            for step,value in re.findall(r'step(\d+)_adjusted_peak=([0-9.]+)',line):last[int(step)-1]=float(value)
        for step,value in last.items():
            if abs(module._max_adjusted_peak(plans[step])-value)>.005:
                raise RuntimeError('Replayed peak differs from completed search')
        print(json.dumps(dict(event='offline_completed_repair_trace_replayed',swaps=count,
            unchanged_prompts=len(stats),trace_sha256=wanted,all_groups_revalidated=True)),flush=True)
        return plans
    module._solve_batches=resumed


def load():
    # Preserve the original module's __file__, sibling imports and all solver
    # settings. Cache the exact pure function outputs, not solver decisions.
    sys.path.insert(0,str(SOURCE.parent))
    spec=importlib.util.spec_from_file_location('memoized_original_v014_planner',SOURCE)
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module
    spec.loader.exec_module(module)
    raw=module._peak_active_tokens;raw_cache=lru_cache(maxsize=65536)(raw)
    @wraps(raw)
    def cached_raw(lengths):return raw_cache(tuple(lengths))
    cached_raw.cache_info=raw_cache.cache_info
    kv=module._peak_active_kv_tokens;kv_cache=lru_cache(maxsize=65536)(kv)
    @wraps(kv)
    def cached_kv(lengths,prompts,*,block_size_tokens):
        return kv_cache(tuple(lengths),tuple(prompts),block_size_tokens=block_size_tokens)
    cached_kv.cache_info=kv_cache.cache_info
    module._peak_active_tokens=cached_raw;module._peak_active_kv_tokens=cached_kv
    return module


def main():
    original=Path(sys.argv[1]).resolve()
    if original!=SOURCE.resolve():raise RuntimeError('Unexpected planner interception')
    module=load()
    if os.environ.get('FAIR_CODE_REPAIR_TRACE'):
        install_repair_replay(module,os.environ['FAIR_CODE_REPAIR_TRACE'],os.environ['FAIR_CODE_REPAIR_TRACE_SHA256'])
    sys.argv=[str(original),*sys.argv[2:]]
    print(json.dumps(dict(event='offline_exact_peak_memoization',
        original=str(original),original_sha256=hashlib.sha256(original.read_bytes()).hexdigest(),
        cache_size=65536,solver_and_objective_unchanged=True)),flush=True)
    module.main()
    print(json.dumps(dict(event='offline_exact_peak_cache_stats',
        raw_peak=module._peak_active_tokens.cache_info()._asdict(),
        kv_peak=module._peak_active_kv_tokens.cache_info()._asdict())),flush=True)


if __name__=='__main__':main()
