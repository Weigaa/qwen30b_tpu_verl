#!/usr/bin/env python3
"""Recheck the completed paired pilot; do not relabel a gated control as OFF."""
import argparse
import importlib.util
import json
from pathlib import Path

BASE=Path('/data/adafloor_graph_results')
ON=BASE/'mlsys_v014_graph_sidecar_full_epoch_seed101_calibrated_20261023T151600Z'
CONTROL=BASE/'mlsys_v014_graph_sidecar_full_epoch_seed101_direct_20261023T135100Z'


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    a=json.loads((ON/'FINAL_EPOCH_AUDIT.json').read_text());b=json.loads((CONTROL/'FINAL_EPOCH_AUDIT.json').read_text())
    if a['status']!='PASS' or b['status']!='PASS':raise RuntimeError('Missing completed audits')
    spec=importlib.util.spec_from_file_location('paired_pilot_audit',ON/'audit_full_epoch.py')
    helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
    rows=[]
    for x,y in zip(a['steps'],b['steps']):
        step=x['step'];current=next(ON.glob(f'*/epoch_001*/rollout_data/{step}.jsonl'))
        control=Path(x['control_rollout_file'])
        d=helper.rows_digest(current);c=helper.rows_digest(control)
        if d['response_sha256']!=c['response_sha256'] or d['response_sha256']!=x['response_sha256']:
            raise RuntimeError('Actual paired output changed after prior audit')
        rows.append(dict(step=step,response_tokens=d['valid_response_tokens'],reward=d['reward_mean'],
            on_seconds=x['rollout_seconds'],control_seconds=y['rollout_seconds'],
            on_response_tokens_s=d['valid_response_tokens']/x['rollout_seconds'],
            control_response_tokens_s=d['valid_response_tokens']/y['rollout_seconds'],
            exact_prompt_response_seed_rank_match=True))
    leases=[]
    for window in a['sidecar']['windows']:
        n=len(window['used_ranks']);before=window['trainer_stop']-window['start'];drain=window['exit_confirmed']-window['trainer_stop']
        leases.append(dict(step=window['step'],used_devices=window['used_ranks'],window_seconds=before,
            startup_and_inference_lease_rank_seconds=n*before,
            shutdown_drain_rank_seconds=n*drain,total_occupied_lease_rank_seconds=n*(before+drain),
            completed_before_stop=window['completed_before_stop'],completed_total=window['completed_queries']))
    on_time=sum(r['on_seconds'] for r in rows);off_time=sum(r['control_seconds'] for r in rows)
    work=sum(r['response_tokens'] for r in rows)
    report=dict(status='PAIRED_PILOT_RECHECKED_STRICT_OFF_PENDING',seed=101,steps=rows,leases=leases,
        primary_responses=2560,primary_on_seconds=on_time,primary_control_seconds=off_time,
        primary_on_response_tokens_s=work/on_time,primary_control_response_tokens_s=work/off_time,
        sidecar=a['sidecar'],control_has_actual_sidecar_starts=b['actual_sidecar_starts'],
        strict_off_flag_ablation_complete=False,
        limitation='Control had Sidecar enabled with every gate skipped; ON changed admission calibration. Actual reuse and exact primary work are verified, but this is not a same-code explicit OFF-switch ablation.',
        rank_time_definition='Assigned physical-device lease from launch to primary stop, plus separately counted stop-to-exit drain; includes cold start, not AICore utilization. No utilization ratio omits skipped windows.')
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    w=leases[0];s=a['sidecar'];lines=['# P4 Sidecar paired pilot: audited reuse, explicit OFF still pending','',
        '| metric | no-actual-Sidecar control | admission ON |','|---|---:|---:|',
        f'| Five-step rollout s | {off_time:.4f} | {on_time:.4f} |',
        f'| Valid response tokens/s | {work/off_time:.4f} | {work/on_time:.4f} |',
        '| Primary output identity | 2560 answers | all2560 exact paired matches |','',
        f"Step4 completes {s['completed_before_trainer_stop']} queries before primary stop; strict accuracy {100*s['strict_accuracy_before_stop']:.4f}%. Shutdown contributes {s['completed_during_shutdown']} additional completions, reported separately.",'',
        f"Actual used eight-device startup+inference lease {w['startup_and_inference_lease_rank_seconds']:.4f} rank-s; shutdown drain {w['shutdown_drain_rank_seconds']:.4f} rank-s; total occupied {w['total_occupied_lease_rank_seconds']:.4f} rank-s.",
        'These are observed assigned-device intervals, not measured compute utilization. No successful-window-only utilization fraction is reported.','',report['limitation'],'',report['rank_time_definition']]
    (args.output/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({k:report[k] for k in ('status','primary_on_seconds','primary_control_seconds','strict_off_flag_ablation_complete')}))


if __name__=='__main__':main()
