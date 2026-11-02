#!/usr/bin/env python3
"""Rebuild one prespecified old mechanism workload with the actual v0.14 planner."""
import argparse
from dataclasses import replace
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys

SOURCE=Path('/workspace/cann-recipes-train/llm_rl/qwen3_014/qwen3_main')
ORIGINAL=Path('/workspace/adafloor_revision_results/quota_controlled_minority_prepared_v1/workload.json')
sys.path.insert(0,str(SOURCE/'tools'))
import build_mode1_length_sorted_e2e_plan as planner


def digest(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def completion_metrics(loads,floor=4):
    if len(loads)!=16 or list(loads)!=sorted(loads):raise ValueError('Need sixteen sorted rank completions')
    end=max(loads);release=[end]*16
    stages=[]
    for old,new,index in ((16,8,7),(8,4,11),(4,2,13)):
        if new<floor:continue
        threshold=loads[index];stages.append(dict(floor=new,quorum_tokens=threshold))
        for rank in range(16-old,16-new):release[rank]=threshold
    idle=sum(end-x for x in loads);area=sum(end-x for x in release)
    waiting=sum(r-x for x,r in zip(loads,release))
    if abs(idle-area-waiting)>1e-6:raise RuntimeError('Idle/release/wait decomposition failed')
    return dict(rank_completion_tokens=loads,stages=stages,tail_tokens=end,
        release_rank_tokens=area,completion_slack_rank_tokens=idle,waiting_rank_tokens=waiting,
        released_fraction_of_completion_slack_proxy=area/idle if idle else None,
        units='rank-token proxy, not measured rank-seconds; this field is not an assumed definition of Q/E')


def independent_peak(left,right,mu):
    seq=[(r['prompt_tokens'],min(math.ceil(value*mu),16384)) for r in (left,right) for value in r['lengths']]
    return max(sum(math.ceil((p+t)/128)*128 for p,end in seq if end>=t) for t in {0,*(end for _,end in seq)})


def prepare(original=ORIGINAL):
    old=json.loads(original.read_text())
    if old['case']!='minority4_scale100':raise RuntimeError('Do not select a new case after observing performance')
    grouped={}
    for req in old['requests']:
        row=grouped.setdefault(req['id'],dict(id=req['id'],prompt_token_ids=req['prompt_token_ids'],
            prompt_hash=req['prompt_hash'],prompt_tokens=len(req['prompt_token_ids']),samples={}))
        if row['prompt_token_ids']!=req['prompt_token_ids'] or req['sample_index'] in row['samples']:
            raise RuntimeError('Changed/duplicate controlled request')
        row['samples'][req['sample_index']]=req['max_tokens']
    rows=[]
    for index in range(32):
        row=grouped[index]
        if set(row['samples'])!=set(range(16)):raise RuntimeError('Incomplete fixed workload')
        row['lengths']=[row['samples'].pop(i) for i in range(16)];row.pop('samples');rows.append(row)
    group=[replace(planner._stats_from_lengths(tuple(r['lengths'])),source_idx=r['id'],
                   prompt_token_length=r['prompt_tokens']) for r in rows]
    metrics=planner._pair_metrics(group,active_peak_safety_factor=1.16,max_response_len=16384)
    allowed=sorted(edge for edge,value in metrics.items() if value[planner.PAIR_ADJUSTED_PEAK]<=133120)
    independent=sorted((a,b) for a in range(32) for b in range(a+1,32)
                       if independent_peak(rows[a],rows[b],1.16)<=133120)
    if allowed!=independent:raise RuntimeError('Independent prompt-inclusive KV graph disagrees')
    edge_sha=hashlib.sha256(json.dumps(allowed,separators=(',',':')).encode()).hexdigest()
    policies={};statuses=[];original_milp=planner.milp
    def certified_milp(*args,**kwargs):
        result=original_milp(*args,**kwargs);statuses.append(int(result.status))
        if result.status not in (0,2):raise RuntimeError('MILP timeout/unresolved is not infeasibility')
        return result
    planner.milp=certified_milp
    try:
        for policy in ('min_skew','release_area'):
            result=planner._release_area_matching_for_floor(group,selected_floor=4,kv_cap=133120,
                active_peak_safety_factor=1.16,max_response_len=16384,rank_matching_policy=policy)
            if not result[1]:raise RuntimeError('Prespecified case is not feasible; do not run it')
            pairs=[list(x[planner.PAIR_SOURCE_INDICES]) for x in result[0]]
            pairs.sort(key=lambda p:(max(max(rows[i]['lengths']) for i in p),sorted(p)))
            if sorted(i for p in pairs for i in p)!=list(range(32)):raise RuntimeError('Bad matching')
            if any(tuple(sorted(p)) not in allowed for p in pairs):raise RuntimeError('Unsafe edge')
            loads=[max(max(rows[i]['lengths']) for i in p) for p in pairs]
            stats=completion_metrics(loads)
            if stats['release_rank_tokens']!=result[4]:raise RuntimeError('Independent release disagreement')
            policies[policy]=dict(feasible=True,pairs=pairs,rank_to_prompt_ids={str(i):p for i,p in enumerate(pairs)},
                feasible_graph_sha256=edge_sha,planner_objective=result[5],**stats)
    finally:planner.milp=original_milp
    comparison=dict(floor=4,kv_cap_tokens_per_rank=133120,mu=1.16,responses=512,
        max_response_tokens=max(r['max_tokens'] for r in old['requests']),policies=policies,
        feasible_edges=allowed,feasible_graph_sha256=edge_sha,feasible_edge_count=len(allowed),
        total_generated_tokens=sum(r['max_tokens'] for r in old['requests']),quota_milp_statuses=statuses)
    report=dict(status='V014_CONTROLLED_WORKLOAD_PREPARED_NOT_NPU_RESULT',case=old['case'],
        origin='unchanged_prespecified_old_constructed_lengths_and_real_raw_token_prompts',
        comparison=comparison,requests=old['requests'],profiles=rows,
        input_sha256={str(p):digest(p) for p in (original,Path(__file__),SOURCE/'tools/build_mode1_length_sorted_e2e_plan.py',SOURCE/'tools/build_mode1_optimized_rank_plan.py')},
        scope='Same fixed-F4 KV graph and frozen public actor, differing matching only. Exact controlled lengths require ignore_eos at runtime. All current time metrics are token proxies, not physical release, reward or training results. Q/E definition remains unconfirmed.')
    # Independent direct MIP, not the schedule enumeration used above.
    certpath=Path('/workspace/cann-recipes-train/llm_rl/qwen3_011/adafloor_revision_eager/analysis_eval/certify_revision_quota_optimum.py')
    spec=importlib.util.spec_from_file_location('independent_v014_certificate',certpath)
    cert=importlib.util.module_from_spec(spec);spec.loader.exec_module(cert)
    cert.pair_peak=lambda a,b,mu=1.16:independent_peak(
        dict(prompt_tokens=len(a['prompt']),lengths=a['lengths']),
        dict(prompt_tokens=len(b['prompt']),lengths=b['lengths']),mu)
    certificate=cert.certify(report,time_limit=30)
    certificate['certifier_source_sha256']=digest(certpath)
    return report,certificate


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    if args.output.exists():raise RuntimeError('Refusing existing output')
    workload,certificate=prepare();args.output.mkdir(parents=True)
    (args.output/'workload.json').write_text(json.dumps(workload,indent=2)+'\n')
    (args.output/'certificate.json').write_text(json.dumps(certificate,indent=2)+'\n')
    print(json.dumps(dict(status=workload['status'],edges=workload['comparison']['feasible_edge_count'],
        policies={k:{x:v[x] for x in ('release_rank_tokens','waiting_rank_tokens')} for k,v in workload['comparison']['policies'].items()},
        strict_min_skew_objective_gap=certificate['strict_objective_gap_rank_tokens'])))


if __name__=='__main__':main()
