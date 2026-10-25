#!/usr/bin/env python3
"""Summarize durable auxiliary jobs, never treating cancelled prefixes as completed."""
import argparse
from functools import lru_cache
import importlib.util
import json
from pathlib import Path
import re
from tailserve_runtime.queue import append_record


@lru_cache(maxsize=1)
def gsm8k_grader():
    # Load the existing stdlib-only grader directly. Importing verl as a
    # package would unnecessarily import torch/NPU code in this host report.
    path=Path(__file__).resolve().parents[1]/'verl/utils/reward_score/gsm8k.py'
    spec=importlib.util.spec_from_file_location('tailserve_offline_gsm8k_grader',path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def summarize(root, rl_output=None):
    latest={};generations=[];phases=[];boundaries=[];opened=[];local_done=[];event_count=0
    for p in (Path(root)/'events').glob('rank_*.jsonl'):
        with p.open() as stream:
            for line in stream:
                if not line.endswith('\n'):break
                row=json.loads(line)
                event_count+=1
                if row.get('job_id'):latest[row['job_id']]=row
                if row['event']=='generation_closed':generations.append(row)
                if row['event']=='auxiliary_phase':phases.append(row)
                if row['event']=='global_primary_complete':boundaries.append(row)
                if row['event']=='generation_open':opened.append(row)
                if row['event']=='local_primary_drained':local_done.append(row)
    completed=[r for r in latest.values() if r['event']=='completed']
    manifest_path=Path(root)/'gsm8k_manifest.jsonl'
    manifest={r['id']:r for r in map(json.loads,manifest_path.read_text().splitlines())} if manifest_path.exists() else {}
    cancelled=[r for r in latest.values() if r['event']=='cancelled']
    # Explicit GSM8K #### answer extraction, not an LLM judge.
    correct=0;flexible_correct=0;format_only_misses=0
    for row in completed:
        matches=re.findall(r'####\s*(-?[0-9][0-9,.]*)',row.get('text',''))
        expected=manifest.get(row['job_id'],{}).get('ground_truth')
        strict_hit=bool(matches and expected is not None and
            matches[-1].replace(',','').rstrip('.') == str(expected).replace(',',''))
        correct+=strict_hit
        if expected is not None:
            flexible_hit=bool(gsm8k_grader().compute_score(row.get('text',''),str(expected),method='flexible'))
            flexible_correct+=flexible_hit
            format_only_misses+=flexible_hit and not strict_hit
    rollout_rows=[];rollout_files=[];trainer_exit_code=None
    if rl_output:
        for path in sorted(Path(rl_output).glob('rollout_data/*.jsonl')):
            rollout_files.append(str(path))
            rollout_rows.extend(json.loads(line) for line in path.read_text().splitlines() if line)
        for path in sorted(Path(rl_output).glob('logs/*.txt')):
            with path.open('rb') as stream:
                stream.seek(max(0,path.stat().st_size-8192))
                endings=re.findall(rb'\[run\] end_time=[^\n]* exit_code=(\d+)',stream.read())
            if endings:trainer_exit_code=int(endings[-1])
    primary_responses=sum(r.get('primary_responses',0) for r in generations)
    result=dict(status='EVENTS_OBSERVED' if event_count else 'NO_RUN_EVIDENCE',
        opened_rank_generations=len(opened),local_drained_rank_generations=len(local_done),
        initial_primary_requests=sum(r.get('primary_requests',0) for r in opened),
        auxiliary_in_flight_jobs=sum(r['event']=='admitted' for r in latest.values()),
        completed_jobs=len(completed),completed_tokens=sum(len(r['token_ids']) for r in completed),
        cancelled_jobs=len(cancelled),cancelled_prefix_tokens=sum(len(r.get('token_ids',[])) for r in cancelled),
        queued_jobs=len(manifest),not_admitted_jobs=sum(k not in latest for k in manifest),
        gsm8k_correct_completed=correct,gsm8k_accuracy=correct/len(completed) if completed and manifest else None,
        gsm8k_flexible_correct_completed=flexible_correct,
        gsm8k_flexible_accuracy=flexible_correct/len(completed) if completed and manifest else None,
        gsm8k_format_only_misses=format_only_misses,
        gsm8k_grading_methods={'gsm8k_accuracy':'Original strict #### numeric marker extraction; includes formatting compliance',
            'gsm8k_flexible_accuracy':'Unmodified VERL gsm8k.py flexible extraction (last number in final 300 characters)',
            'denominator':'Finished requests, including length-capped requests; inspect finish reasons and coverage'},
        gsm8k_finish_reasons={reason:sum(r.get('finish_reason')==reason for r in completed)
                             for reason in sorted({str(r.get('finish_reason')) for r in completed})},
        auxiliary_prefill_ranks=sorted({r['rank'] for r in phases if r['phase']=='prefill'}),
        auxiliary_decode_ranks=sorted({r['rank'] for r in phases if r['phase']=='decode'}),
        phases_only_on_local_rl_empty=all(r['local_primary_requests']==0 for r in phases) if phases else None,
        primary_responses=primary_responses,primary_tokens=sum(r.get('primary_tokens',0) for r in generations),
        primary_rollout_files=rollout_files,primary_rollout_rows=len(rollout_rows),
        trainer_exit_code=trainer_exit_code,
        mean_primary_reward=sum(float(r['score']) for r in rollout_rows)/len(rollout_rows) if rollout_rows else None,
        primary_event_and_artifact_counts_match=(len(rollout_rows)==primary_responses)
            if generations and rollout_files else None,
        global_primary_boundaries=len(boundaries),
        max_stop_cleanup_s=max((r.get('stop_cleanup_s',0) for r in generations),default=None),
        closed_rank_generations=len(generations),
        all_closed_generations_drained=all(r['auxiliary_drained'] for r in generations) if generations else None,
        policy_versions=sorted({r['policy_version'] for r in completed}),
        primary_slowdown='Requires paired RL-only workload timing; not inferred from auxiliary event counts',
        quality='Requires same-checkpoint, same-question standalone evaluation',
        not_hardware_utilization=True)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--spool',required=True)
    p.add_argument('--rl-output');p.add_argument('--save')
    args=p.parse_args()
    result=summarize(args.spool,args.rl_output)
    if args.save:
        # Distinct export of inference data, never passed to the RL trainer.
        parent=Path(args.save).parent
        for name,event in [('gsm8k_completed.jsonl','completed'),('gsm8k_cancelled.jsonl','cancelled')]:
            target=parent/name
            if target.exists():target.unlink()
            target.touch()
            for path in sorted((Path(args.spool)/'events').glob('rank_*.jsonl')):
                for line in path.read_text().splitlines():
                    row=json.loads(line)
                    if row['event']==event:append_record(target,row,timestamp=False)
        with open(args.save,'w') as stream:json.dump(result,stream,indent=2)
    print(json.dumps(result,indent=2))
