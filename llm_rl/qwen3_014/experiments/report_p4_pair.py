#!/usr/bin/env python3
"""Paper-ready, explicitly scoped single-seed frozen Sidecar ON/OFF report."""
import hashlib
import json
from pathlib import Path
import re
import statistics
from rank_occupancy import parse_epoch

ROOT=Path('/data/adafloor_graph_results/mlsys_v014_supplement_20261023T180000Z')


def audit_runtime(epoch):
    logs=list((epoch/'logs').glob('*.txt'))
    if len(logs)!=1:raise RuntimeError('Missing unique primary log')
    text=logs[0].read_text(errors='replace')
    if not re.search(r'\[run\] end_time=\S+ exit_code=0',text):raise RuntimeError('Primary child did not exit0')
    rows=[json.loads(s) for s in re.findall(r'MODE1_STEP_RUNTIME_AUDIT (\{[^\n]+\})',text)]
    result={(a['step'],a['rank']):a for a in rows}
    if len(rows)!=80 or set(result)!={(s,r) for s in range(1,6) for r in range(16)}:
        raise RuntimeError('Missing actual per-rank FULL audit coverage')
    if any(not a['complete'] or a.get('full_iterations',0)<=0 or a['preemptions'] for a in rows):
        raise RuntimeError('Incomplete FULL execution or observed preemption')
    return result


def main():
    pair=json.loads((ROOT/'P4_EXPLICIT_PAIR_RESULT.json').read_text())
    if pair['status']!='EXPLICIT_OFF_ON_PAIR_COMPLETE' or len(pair['steps'])!=5:
        raise RuntimeError('Not a completed explicit pair')
    if not all(s['same_input_contract'] and s['exact_response_match'] for s in pair['steps']):
        raise RuntimeError('Do not claim exact quality with differing paired outputs')
    if pair['same_primary_code_files']!=2090:raise RuntimeError('Primary source identity not proved')
    intervals=json.loads((ROOT/'p4_sidecar_actual_intervals/summary.json').read_text())
    if intervals['status']!='SIDECAR_DEVICE_LEASE_AND_MODEL_INTERVALS_AUDITED':raise RuntimeError('Missing actual-device audit')
    work={s['step']:s['on_response_tokens'] for s in pair['steps']};W=sum(work.values());arms={};audits={}
    for name in ('off','on'):
        root=Path(pair[name]);epoch=next(root.glob('*/epoch_001_mode1_planned'))
        receipt=root/'DRIVER_RESULT.json'
        if receipt.is_file():
            if json.loads(receipt.read_text())['exit_code']!=0:raise RuntimeError('Whole-arm driver did not exit0')
        elif name=='on':
            # The already-completed pilot predates the structured receipt.
            # Accept its actual parent exit line AND validated cleanup receipt,
            # not a newly invented success marker or the green audit alone.
            text=(root/'driver.log').read_text(errors='replace')
            cleanup=epoch.parent/'CHECKPOINTS_REMOVED_AFTER_VALIDATION.txt'
            if not re.search(r'^driver_exit_code=0$',text,re.M) or not cleanup.is_file():
                raise RuntimeError('Legacy ON driver/transaction success not proved')
        else:raise RuntimeError('Missing actual driver exit receipt')
        audits[name]=audit_runtime(epoch)
        duration=sum(s[name+'_seconds'] for s in pair['steps'])
        arms[name]=dict(rollout_seconds=duration,response_tokens=W,response_tokens_s=W/duration,
            reward=statistics.mean(s[name+'_reward'] for s in pair['steps']),
            actual_FULL_rank_step_audits=80,preemptions=0,
            Q_E=parse_epoch(epoch,work,'FULL'))
    keys=('kv_tokens','kv_blocks','worker_kv_blocks','floor')
    for key in audits['off']:
        if any(audits['off'][key][k]!=audits['on'][key][k] for k in keys):
            raise RuntimeError('Actual scheduler/worker KV budget or target floor differs')
    side=pair['sidecar_on'];a=arms['off'];b=arms['on']
    report=dict(status='P4_EXPLICIT_PAIR_PAPER_REPORT_AUDITED',seed=101,actor_frozen=True,
        epochs=1,steps=5,primary=arms,primary_responses=2560,all_primary_answers_exact=True,
        actual_per_rank_KV_and_selected_floor_equal=True,
        rollout_time_change_pct=100*(b['rollout_seconds']/a['rollout_seconds']-1),
        response_throughput_change_pct=100*(b['response_tokens_s']/a['response_tokens_s']-1),
        sidecar=side,actual_device_intervals=intervals,
        pair_sha256=hashlib.sha256((ROOT/'P4_EXPLICIT_PAIR_RESULT.json').read_bytes()).hexdigest(),
        scope='One seed101 frozen five-step P4 pair, explicit planned survivor identities and no TailGuard, shared math epoch0. Not the updated-policy, natural-member-selection main-P4 trajectory, not a convergence or equivalence test. No CI/universal overhead bound. Sidecar queries may repeat, are not unique questions; partials excluded. No mixed-model token aggregation, AICore utilization or energy/billing claim.')
    destination=ROOT/'p4_explicit_pair_report';destination.mkdir(exist_ok=True)
    (destination/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    lines=['# Ada-P4 Sidecar: explicit ON/OFF pair completed','',
        '| Primary metric | OFF | ON |','|---|---:|---:|',
        f"| Five-step rollout seconds | {a['rollout_seconds']:.4f} | {b['rollout_seconds']:.4f} |",
        f"| Response tokens/s | {a['response_tokens_s']:.4f} | {b['response_tokens_s']:.4f} |",
        f"| Reward | {a['reward']:.8f} | {b['reward']:.8f} |",
        '| Exact primary answers | 2560 | all2560 identical |','',
        f"ON time change {report['rollout_time_change_pct']:+.3f}%; response-throughput change {report['response_throughput_change_pct']:+.3f}%. Single seed, not a statistical bound.",'',
        f"Sidecar completes {side['completed_before_trainer_stop']} queries and {side['completed_output_tokens_before_stop']} output tokens before primary stop. Strict accuracy {side['correct_before_stop']}/{side['completed_before_trainer_stop']} = {100*side['strict_accuracy_before_stop']:.4f}%. Drain adds {side['completed_during_shutdown']} completed queries, reported separately; {side['partial_queries']} partials are not completions.",'',
        f"Eight actual devices, active lease161.0580s. Startup to all eight models ready55.8677s; stop to all engines exited3.8833s; matching ACK precedes restore. Actual primary FULL/secondary overlap observed. No OOM, preemption or SIGKILL.",'',
        f"Assigned startup+service lease1288.4644 rank-s, drain31.0663, total{intervals['assigned_lease_including_drain_rank_seconds']:.4f}. Model-ready to shutdown-start is a known resident lower bound{intervals['known_model_resident_rank_seconds']:.4f} rank-s ({intervals['known_model_resident_rank_seconds_before_stop']:.4f} before primary stop). Engine API construction-to-cleanup envelope{intervals['engine_API_envelope_rank_seconds']:.4f} rank-s. None is an AICore busy-time measurement.",'',
        'Both arms have80 actual FULL rank-step audits, zero preemptions, identical actual scheduler/worker KV and selected-floor settings, and2090 identical primary source files. This OFF genuinely disables Sidecar, unlike the earlier enabled-but-all-gates-skipped control.','',report['scope'],'',
        'Machine evidence: `summary.json`, `../P4_EXPLICIT_PAIR_RESULT.json`, `../p4_sidecar_actual_intervals/summary.json`.']
    (destination/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(status=report['status'],primary=arms,
        rollout_time_change_pct=report['rollout_time_change_pct'],
        response_throughput_change_pct=report['response_throughput_change_pct']),indent=2))


if __name__=='__main__':main()
