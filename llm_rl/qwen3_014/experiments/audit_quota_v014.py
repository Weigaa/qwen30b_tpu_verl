#!/usr/bin/env python3
"""Audit real local-done/quorum endpoints, never convert token proxies to seconds."""
import argparse
import hashlib
import json
from pathlib import Path
import re


def audit_arm(root,data,policy,workload_sha):
    driver=json.loads((root/'DRIVER_RESULT.json').read_text())
    if driver['exit_code']!=0:raise RuntimeError('Unsuccessful controlled arm')
    text=(root/'primary.log').read_text(errors='replace')
    if re.search(r'Traceback \(most recent call last\)|NPU out of memory|OutOfMemoryError|Preempting request',text):
        raise RuntimeError('Controlled execution fault/preemption')
    rows=[json.loads((root/'ranks'/f'rank_{i:02d}.json').read_text()) for i in range(16)]
    start={x['start'] for x in rows};budgets={x['kv_tokens'] for x in rows};seen={}
    expected={(x['id'],x['sample_index']):x for x in data['requests']}
    for rank,row in enumerate(rows):
        if row['rank']!=rank or row['policy']!=policy or row['status']!='V014_FULL_CONTROLLED_RANK_COMPLETE':
            raise RuntimeError('Rank status mismatch')
        if row['workload_sha256']!=workload_sha or row['feasible_graph_sha256']!=data['comparison']['feasible_graph_sha256']:
            raise RuntimeError('Input/feasible graph mismatch')
        if not str(row['versions']['vllm']).startswith('0.14') or row['forward_audit'].get('full_iterations',0)<=0:
            raise RuntimeError('Not true v0.14 FULL execution')
        weight=row.get('initial_weight_value_audit',{})
        if (row.get('initial_loaded_experts_per_layer')!=[8]*48
                or row.get('initial_physical_expert_slots')!=[32]*48
                or weight.get('checked')!=24 or weight.get('failed')!=0):
            raise RuntimeError('Fixed-F4 residency and actual HF loaded values not proved')
        if not row['start']<=row['local_done']<=row['generation_end']<=row['restore_start']<=row['restore_end']:
            raise RuntimeError('Invalid completion lifecycle')
        if (row.get('native_sleep_level')!=2 or row.get('effective_weight_layout_NZ')!=0 or not row['generation_end']<=row['sleep_start']<=
                row['sleep_done']<=row['restore_start']<=row['restore_end']<=row['window_end']):
            raise RuntimeError('Missing/inconsistent native-sleep2 release and restore')
        owned=data['comparison']['policies'][policy]['rank_to_prompt_ids'][str(rank)]
        if len(row['requests'])!=32:raise RuntimeError('Not32 requests/rank')
        for request in row['requests']:
            key=request['id'],request['sample_index']
            if key in seen or key not in expected or request['id'] not in owned:raise RuntimeError('Duplicate/wrong request')
            target=expected[key]
            if (request['generated_tokens']!=target['max_tokens'] or request['finish_reason']!='length'
                    or request['request_seed']!=target['request_seed'] or request['prompt_hash']!=target['prompt_hash']):
                raise RuntimeError('Forced work changed')
            seen[key]=request
    if len(start)!=1 or budgets!={133120} or set(seen)!=set(expected):raise RuntimeError('Missing/inconsistent work')
    ids=[i for row in rows for i in row['initial_primary_expert_ids']]
    if len(ids)!=128 or set(ids)!=set(range(128)):
        raise RuntimeError('Initial value-checked primary experts do not cover all128 exactly once')
    t0=start.pop();end=max(r['generation_end'] for r in rows);events=[e for r in rows for e in r['events']]
    stages=[];release=[end]*16
    for old,target,participants in ((16,list(range(8,16)),set(range(16))),
                                    (8,list(range(12,16)),set(range(8,16)))):
        chosen=[e for e in events if e['active_ranks']==target]
        if len(chosen)!=len(participants) or {e['reporter'] for e in chosen}!=participants:
            raise RuntimeError('Missing/duplicate stage quorum participants')
        first=min(e['start'] for e in chosen);done=max(e['done'] for e in chosen)
        donors=list(range(16-old,16-len(target)))
        data_quorum=max(rows[i]['local_done'] for i in range(16-len(target)))
        if not t0<=data_quorum<=first<=done<=end:raise RuntimeError('Data/actual RPC chronology mismatch')
        for i in donors:release[i]=done
        stages.append(dict(floor=len(target),data_quorum_s=data_quorum-t0,rpc_start_s=first-t0,
            coordinated_rpc_done_s=done-t0,donors=donors,
            released_rank_seconds=(old-len(target))*(end-done)))
    deepest=[e for e in events if e['active_ranks']==list(range(12,16))]
    if not any(rows[e['reporter']]['forward_audit']['full_iterations']>e['full_iterations_after']
               for e in deepest if e['reporter'] in range(12,16)):
        raise RuntimeError('No FULL replay after deepest topology transition')
    slack=sum(end-r['local_done'] for r in rows)
    reclaimed=sum(end-x for x in release)
    waiting=sum(x-r['local_done'] for x,r in zip(release,rows))
    if abs(slack-reclaimed-waiting)>1e-5 or waiting<0:raise RuntimeError('Measured slack/release/wait identity failed')
    last={r['window_end'] for r in rows}
    if len(last)!=1:raise RuntimeError('Inconsistent global window end')
    window_end=last.pop();elapsed=window_end-t0;occupied=[]
    wait_before_rpc=0.;transition=0.;actual_detached=0.
    for rank,row in enumerate(rows):
        exits=[e for e in row['events'] if rank not in e['active_ranks']]
        if len(exits)>1:raise RuntimeError('Multiple donor exits')
        exit_start=exits[0]['start'] if exits else end
        exit_end=exits[0]['done'] if exits else end
        if row['local_done']>exit_start:raise RuntimeError('Exit before local requests finished')
        wait_before_rpc+=exit_start-row['local_done'];transition+=exit_end-exit_start
        actual_detached+=end-exit_end
        # Native-sleep completion, NOT group detach, opens the free interval.
        # A nonpositive asynchronous free-memory counter cannot establish a
        # physical release. Conservatively charge that entire rank instead.
        physical=row['free_bytes_after_sleep']>row['free_bytes_before_sleep']
        free=row['restore_start']-row['sleep_done'] if physical else 0.
        occupied.append(dict(rank=rank,physical_release_counter_positive=physical,
            free_bytes_change=row['free_bytes_after_sleep']-row['free_bytes_before_sleep'],
            native_release_s=row['sleep_done']-t0,reoccupation_s=row['restore_start']-t0,
            occupied_rank_seconds=elapsed-free,
            occupied_intervals=[[t0,row['sleep_done']],[row['restore_start'],window_end]]
                if physical else [[t0,window_end]]))
    if abs(slack-wait_before_rpc-transition-actual_detached)>1e-5:
        raise RuntimeError('Actual donor wait/RPC/detach decomposition fails')
    H=sum(r['occupied_rank_seconds'] for r in occupied)
    W=sum(r['generated_tokens'] for r in seen.values())
    return dict(policy=policy,rollout_seconds=elapsed,generation_seconds=end-t0,
        startup_seconds=max(r['startup_seconds'] for r in rows),
        lifecycle_with_restore_seconds=max(r['restore_end'] for r in rows)-t0,
        generated_tokens=sum(r['generated_tokens'] for r in seen.values()),kv_tokens=133120,stages=stages,
        local_done_seconds=[r['local_done']-t0 for r in rows],
        completion_slack_rank_seconds=slack,ep_detached_rank_seconds=reclaimed,
        waiting_and_transition_rank_seconds=waiting,
        detached_fraction_of_completion_slack=reclaimed/slack if slack else None,
        stage_endpoint_fields_are_coordinated_proxy_not_physical_release=True,
        actual_exit_rpc_detached_rank_seconds=actual_detached,
        waiting_before_exit_rpc_rank_seconds=wait_before_rpc,exit_rpc_rank_seconds=transition,
        waiting_scope='Local completion to own exit-RPC start, or global generation end for final survivors. Includes dummy/metadata synchronization, not isolated dummy-kernel execution time.',
        occupied_rank_seconds=H,rank_occupancy=occupied,
        Q_response_tokens_s=W/elapsed,E_response_tokens_per_occupied_rank_s=W/H,
        actual_deepest_floor_FULL_observed=True,
        weights_audit_sha256=rows[0]['weights_audit_sha256'],versions=rows[0]['versions'])


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--workload',type=Path,required=True);args=p.parse_args()
    data=json.loads(args.workload.read_text());sha=hashlib.sha256(args.workload.read_bytes()).hexdigest()
    a,b=[audit_arm(args.root/name,data,name,sha) for name in ('min_skew','release_area')]
    if any(a[k]!=b[k] for k in ('generated_tokens','kv_tokens','weights_audit_sha256','versions')):
        raise RuntimeError('Unmatched frozen pair')
    delta_tail=b['generation_seconds']-a['generation_seconds']
    schedule=sum(-weight*(new['coordinated_rpc_done_s']-old['coordinated_rpc_done_s'])
                 for weight,old,new in zip((8,4),a['stages'],b['stages']))
    tail=12*delta_tail;gain=b['ep_detached_rank_seconds']-a['ep_detached_rank_seconds']
    if abs(schedule+tail-gain)>1e-5:raise RuntimeError('Observed release decomposition fails')
    result=dict(status='V014_FROZEN_SAME_GRAPH_QUOTA_PAIR_AUDITED',arms=[a,b],
        release_gain_rank_seconds=gain,schedule_endpoint_component_rank_seconds=schedule,
        tail_extension_component_rank_seconds=tail,
        metrics=dict(Q='W/T, response tokens/s',E='W/H, response tokens/occupied rank-s',
            reference='/workspace/cann-recipes-train/llm_rl/qwen3_014/qwen3_main/analysis_eval/mlsys2027_two_metrics_v2/main.tex, eqs occupancy/efficiency',
            Q_gain_percent=100*(b['Q_response_tokens_s']/a['Q_response_tokens_s']-1),
            E_gain_percent=100*(b['E_response_tokens_per_occupied_rank_s']/a['E_response_tokens_per_occupied_rank_s']-1)),
        scope='One disclosed constructed-length pair, no repetitions/CI or quality claims. FULL/TQ1, static Planned-F4 residency, native sleep2/lifecycle1 after local generation, restore included in T/H. Initial checkpoint load/capture excluded from this controlled-window scope. Stage endpoint decomposition is algebraic, not an isolated causal fixed-tail rerun. Native sleep/free-memory observations, not EP detach, determine occupied rank-time. No sidecar utilization claim.')
    (args.root/'QUOTA_PAIR_RESULT.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
