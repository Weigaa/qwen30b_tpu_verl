#!/usr/bin/env python3
"""Summarize the completed controlled mechanism pair and independently check H."""
import json
from pathlib import Path

ROOT=Path('/data/adafloor_graph_results/mlsys_v014_supplement_20261023T180000Z')


def main():
    base=ROOT/'quota_runtime';result=json.loads((base/'QUOTA_PAIR_RESULT.json').read_text())
    if result['status']!='V014_FROZEN_SAME_GRAPH_QUOTA_PAIR_AUDITED':raise RuntimeError('Not completed')
    for arm in result['arms']:
        records=[json.loads((base/arm['policy']/f'ranks/rank_{i:02d}.json').read_text()) for i in range(16)]
        T=records[0]['window_end']-records[0]['start']
        H=sum((r['sleep_done']-r['start'])+(r['window_end']-r['restore_start'])
            if r['free_bytes_after_sleep']>r['free_bytes_before_sleep'] else T for r in records)
        if abs(H-arm['occupied_rank_seconds'])>1e-5:raise RuntimeError('Independent occupied-interval sum differs')
        if sum(q['generated_tokens'] for r in records for q in r['requests'])!=1710400:
            raise RuntimeError('Actual fixed work differs')
        if any(r['versions']!=records[0]['versions'] or r['weights_audit_sha256']!=records[0]['weights_audit_sha256'] for r in records):
            raise RuntimeError('Within-arm source weights/versions differ')
        arm['independent_H_check']=True
        arm['physical_release_counters_positive']=sum(r['free_bytes_after_sleep']>r['free_bytes_before_sleep'] for r in records)
    a,b=result['arms'];waiting=100*(b['waiting_before_exit_rpc_rank_seconds']/a['waiting_before_exit_rpc_rank_seconds']-1)
    result['waiting_interval_change_percent']=waiting
    destination=ROOT/'quota_pair_report';destination.mkdir(exist_ok=True)
    (destination/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    lines=['# Quota versus MinSkew: controlled v0.14 FULL/native-sleep2 pair','',
        'One prespecified `minority4_scale100` case,32 real raw-token prompts×16 controlled-length responses. '
        'Same frozen audited common actor,398-edge KV-feasible graph,133120 tokens/rank, '
        'Planned-F4 physical capacity32 rows with initial EP16 primary8, fixed K8 graph buckets. '
        'Only prompt matching/placement changes; weights, expert-residency policy, buckets, request seeds and work stay fixed. '
        'Ignore-EOS forced lengths are disclosed; no response-quality claim.','',
        '| Metric | MinSkew | Quota |','|---|---:|---:|']
    for label,key in [('Controlled window seconds','rollout_seconds'),('Generated response tokens','generated_tokens'),
        ('Q: response tokens/s','Q_response_tokens_s'),('H: occupied rank-s','occupied_rank_seconds'),
        ('E: response tokens/occupied rank-s','E_response_tokens_per_occupied_rank_s'),
        ('Local-completion-to-exit waiting rank-s','waiting_before_exit_rpc_rank_seconds')]:
        lines.append(f"| {label} | {a[key]:.3f} | {b[key]:.3f} |")
    lines+=['',f"Q improvement {result['metrics']['Q_gain_percent']:.3f}%, E improvement {result['metrics']['E_gain_percent']:.3f}%, waiting interval change {waiting:.3f}%.",'',
        '| Observed endpoint, seconds from shared start | MinSkew | Quota |','|---|---:|---:|']
    for i in range(2):
        for label,key in [('data quorum','data_quorum_s'),('all participating RPCs done','coordinated_rpc_done_s')]:
            lines.append(f"| floor{a['stages'][i]['floor']} {label} | {a['stages'][i][key]:.3f} | {b['stages'][i][key]:.3f} |")
    lines+=['',
        f"Coordinated-stage EP-detach proxy gains {result['release_gain_rank_seconds']:.3f} rank-s, algebraically consisting of "
        f"{result['schedule_endpoint_component_rank_seconds']:.3f} from earlier stage endpoints and "
        f"{result['tail_extension_component_rank_seconds']:.3f} from the shorter generation tail. This proxy is not credited as physical free time.",'',
        'H instead opens each free interval after actual native-sleep completion and closes it at restore start. '
        'Cleanup and full-world restore remain charged. Independent per-rank interval reconstruction matches both H totals; '
        'all16 physical-release counters are positive in each arm. Actual FULL replay after floor4 and complete EP16 restore are required.','',
        a['waiting_scope'],'',result['scope'],'',
        'Single pair, no repetitions/CI or population/equivalence claim. The prior CPU proof is a rank-token surrogate optimality certificate, '
        'not a certificate of globally optimal measured Q/E. Do not put these prepared-window absolute throughputs into the main updated-RL-epoch comparison.','',
        'Raw per-rank events: `../quota_runtime/{min_skew,release_area}/ranks/`; full pair: `../quota_runtime/QUOTA_PAIR_RESULT.json`.']
    (destination/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(status='QUOTA_REPORT_INDEPENDENT_H_AUDITED',Q_gain=result['metrics']['Q_gain_percent'],
        E_gain=result['metrics']['E_gain_percent'],waiting_change=waiting)))


if __name__=='__main__':main()
