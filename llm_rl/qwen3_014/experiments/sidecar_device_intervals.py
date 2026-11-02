#!/usr/bin/env python3
"""Observed Sidecar device leases, with model-residency lower bounds.

Never substitute an assigned lease for measured AICore utilization or claim
the exact physical allocation instant from model-loader API timestamps.
"""
import argparse
import hashlib
import json
from pathlib import Path

ON=Path('/data/adafloor_graph_results/mlsys_v014_graph_sidecar_full_epoch_seed101_calibrated_20261023T151600Z')


def analyze(root=ON):
    audit=json.loads((root/'FINAL_EPOCH_AUDIT.json').read_text())
    if audit['status']!='PASS':raise RuntimeError('Primary/secondary audit not complete')
    log=next(root.glob('*/epoch_001*/sidecar/infer.log'));events=[]
    for line in log.read_text(errors='replace').splitlines():
        if not line.startswith('{'):continue
        try:row=json.loads(line)
        except json.JSONDecodeError:continue
        if row.get('event') in ('sidecar_load_start','sidecar_model_ready',
                'sidecar_engine_shutdown_start','sidecar_engine_shutdown_done'):events.append(row)
    rows=[];leases=[]
    for window in audit['sidecar']['windows']:
        start=window['start'];stop=window['trainer_stop'];end=window['exit_confirmed']
        selected=[e for e in events if start<=e['time']<=end]
        for shard in range(8):
            trace={}
            for event in selected:
                if event['shard_index']!=shard:continue
                if event['event'] in trace:raise RuntimeError('Duplicate shard lifecycle event')
                trace[event['event']]=event
            if len(trace)!=4:raise RuntimeError('Missing per-shard model lifecycle boundary')
            load,ready,shutdown,done=[trace[key] for key in ('sidecar_load_start','sidecar_model_ready',
                'sidecar_engine_shutdown_start','sidecar_engine_shutdown_done')]
            if not start<=load['time']<=ready['time']<=stop<=shutdown['time']<=done['time']<=end:
                raise RuntimeError('Invalid per-device model lifecycle')
            if any(r.get('status') in ('error','import_error') for r in done['results']):
                raise RuntimeError('Unsuccessful engine cleanup')
            device=int(load['devices'])
            rows.append(dict(step=window['step'],shard=shard,device=device,
                load_start=load['time'],model_ready=ready['time'],shutdown_start=shutdown['time'],
                shutdown_done=done['time'],
                engine_API_envelope_seconds=done['time']-load['time'],
                known_model_resident_seconds=shutdown['time']-ready['time'],
                known_model_resident_seconds_before_stop=stop-ready['time']))
        actual=[r['device'] for r in rows if r['step']==window['step']]
        if sorted(actual)!=sorted(window['used_ranks']):raise RuntimeError('Actual initialized devices differ from lease')
        leases.append(dict(step=window['step'],devices=actual,
            assigned_before_stop_rank_seconds=len(actual)*(stop-start),
            assigned_drain_rank_seconds=len(actual)*(end-stop)))
    return dict(status='SIDECAR_DEVICE_LEASE_AND_MODEL_INTERVALS_AUDITED',source=str(root),rows=rows,leases=leases,
        assigned_lease_including_drain_rank_seconds=sum(l['assigned_before_stop_rank_seconds']+l['assigned_drain_rank_seconds'] for l in leases),
        known_model_resident_rank_seconds=sum(r['known_model_resident_seconds'] for r in rows),
        known_model_resident_rank_seconds_before_stop=sum(r['known_model_resident_seconds_before_stop'] for r in rows),
        engine_API_envelope_rank_seconds=sum(r['engine_API_envelope_seconds'] for r in rows),
        infer_log_sha256=hashlib.sha256(log.read_bytes()).hexdigest(),
        scope='Assigned lease includes cold startup and drain. Ready-to-shutdown-start is a known model-resident lower bound, not all actual allocation time. Load-start-to-cleanup-done is an engine API envelope, not an exact physical allocation interval. No AICore utilization, successful-window-only capacity fraction or full-HBM-release claim. Skipped windows have zero actual Sidecar allocation but are not removed from capacity-denominator claims.')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    result=analyze();args.output.mkdir(parents=True,exist_ok=True)
    (args.output/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('rows','leases')},indent=2))
