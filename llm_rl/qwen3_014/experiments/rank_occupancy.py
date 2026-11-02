#!/usr/bin/env python3
"""Offline Q=W/T, E=W/H, with native-sleep release and restore reoccupation.

Uses the paper's recorded-clock conventions. Accepts eager and FULL only
after validating actual rank/step audits; never requires a fake FULL flag.
"""
import hashlib
import importlib.util
import json
from pathlib import Path
import re
from collections import defaultdict

PAPER = Path('/workspace/cann-recipes-train/llm_rl/qwen3_014/qwen3_main/analysis_eval/mlsys2027_two_metrics_v2/analyze_rank_time.py')


def parse_epoch(epoch, step_work, mode):
    if mode not in ('eager', 'FULL') or set(step_work) != set(range(1,6)):
        raise ValueError('Require explicit execution mode and five actual work totals')
    spec=importlib.util.spec_from_file_location('paper_occupancy_conventions', PAPER)
    paper=importlib.util.module_from_spec(spec);spec.loader.exec_module(paper)
    logs=list((Path(epoch)/'logs').glob('*.txt'))
    if len(logs)!=1:raise RuntimeError('Missing unique primary log')
    log=logs[0];raw=log.read_text(errors='replace')
    audits={};pid_rank={};ends=[];sleeps=defaultdict(list)
    detached=defaultdict(list);resumes=defaultdict(list);progress=defaultdict(list)
    for line in paper.ANSI.sub('',raw).splitlines():
        for chunk in re.split(r'(?=\([^()\n]*pid=\d+\))',line):
            stamp=paper.timestamp(chunk)
            if stamp is None:continue
            match=paper.PID.match(chunk);pid=int(match[1]) if match else None
            if 'MODE1_STEP_RUNTIME_AUDIT ' in chunk:
                audit=json.JSONDecoder().raw_decode(chunk.split('MODE1_STEP_RUNTIME_AUDIT ',1)[1])[0]
                if pid is None or not audit['complete']:raise RuntimeError('Incomplete rank audit')
                if pid_rank.setdefault(pid,audit['rank'])!=audit['rank']:raise RuntimeError('Worker/rank changed')
                key=(audit['step'],audit['rank'])
                if key in audits:raise RuntimeError('Duplicate rank/step audit')
                if (mode=='FULL' and audit.get('full_iterations',0)<=0) or (mode=='eager' and
                        (audit.get('full_iterations',0)!=0 or audit.get('eager_iterations',0)<=0)):
                    raise RuntimeError('Actual execution mode differs from declared occupancy arm')
                audits[key]=(audit,stamp)
            if pid is not None and 'seconds to fall asleep.' in chunk:sleeps[pid].append(stamp)
            match=paper.DETACH.search(chunk)
            if match:
                rank=int(match[1])
                if rank not in list(map(int,re.findall(r'\d+',match[2]))):detached[rank].append(stamp)
            match=paper.RESTORE.search(chunk)
            if match:
                elapsed=float(match[2])/1000
                resumes[int(match[1])].append(dict(start=[stamp[0]-elapsed-.000005,
                    stamp[1]-elapsed+.000005],end=stamp,kind='restore_rpc'))
            match=paper.REQUEST.search(chunk)
            if match:resumes[int(match[1])].append(dict(start=list(stamp),end=stamp,kind='restore_request'))
            match=paper.ELAPSED.search(chunk)
            if match and pid is not None:
                elapsed=float(match[1]);resumes[('pid',pid)].append(dict(start=[stamp[0]-elapsed-.0000005,
                    stamp[1]-elapsed+.0000005],end=stamp,kind='wake'))
            if pid is not None and 'rollout_forward_begin ' in chunk:progress[pid].append(stamp)
            if 'rollout_output_time_s:' in chunk:
                ends.append((stamp[0],float(chunk.split('rollout_output_time_s:',1)[1].split()[0])))
    expected={(s,r) for s in range(1,6) for r in range(16)}
    if set(audits)!=expected or len(pid_rank)!=16 or len(ends)!=5:
        raise RuntimeError('Incomplete five-step occupancy coverage')
    rank_pid={r:p for p,r in pid_rank.items()}
    for rank,pid in rank_pid.items():resumes[rank].extend(resumes[('pid',pid)])
    steps=[]
    for step,(end,duration) in enumerate(sorted(ends),1):
        start=end-duration;ranks=[]
        for rank in range(16):
            audit,stamp=audits[(step,rank)]
            if not start<=stamp[0]<=end:raise RuntimeError('Audit outside rollout window')
            exits=[v for v in detached[rank] if start<=v[0]<=end]
            if len(exits)>1:raise RuntimeError('Repeated rank detach in one step')
            if not exits:
                ranks.append(dict(rank=rank,detached=False,occupied_rank_s_bounds=[duration,duration],
                    occupied_intervals_for_lower_bound=[[start,end]],occupied_intervals_for_upper_bound=[[start,end]]))
                continue
            ready=[v for v in sleeps[rank_pid[rank]] if stamp[0]<=v[0]<=end]
            if len(ready)!=1 or ready[0][0]<exits[0][0]:raise RuntimeError('No unique post-detach sleep completion')
            ready_lo,ready_hi=ready[0]
            events=[v for v in resumes[rank] if start<=v['end'][0]<=end and v['start'][1]>=ready_lo]
            close_lo=min([v['start'][0] for v in events]+[end]);close_hi=min([v['start'][1] for v in events]+[end])
            close_lo=max(start,min(end,close_lo));close_hi=max(start,min(end,close_hi))
            occupied,free=paper.occupied_bounds(start,end,[ready_lo,ready_hi],[close_lo,close_hi])
            if any(ready_hi<t[0]<close_lo for t in progress[rank_pid[rank]]):
                raise RuntimeError('Forward execution inside credited free interval')
            lower=[[start,ready_lo]];upper=[[start,min(end,ready_hi)]]
            if events:lower.append([close_hi,end]);upper.append([close_lo,end])
            ranks.append(dict(rank=rank,detached=True,ready_bounds=[ready_lo,ready_hi],
                reoccupation_bounds=[close_lo,close_hi],reoccupation_observed=bool(events),
                reusable_interval_s_bounds=free,occupied_rank_s_bounds=occupied,
                occupied_intervals_for_lower_bound=lower,occupied_intervals_for_upper_bound=upper))
        H=[sum(r['occupied_rank_s_bounds'][j] for r in ranks) for j in range(2)]
        steps.append(dict(step=step,rollout_s=duration,response_tokens=step_work[step],
            occupied_rank_s_bounds=H,ranks=ranks))
    T=sum(s['rollout_s'] for s in steps);W=sum(step_work.values())
    H=[sum(s['occupied_rank_s_bounds'][j] for s in steps) for j in range(2)]
    return dict(epoch_dir=str(epoch),mode=mode,rollout_seconds=T,response_tokens=W,
        occupied_rank_s_bounds=H,Q_response_tokens_s=W/T,
        E_response_tokens_per_occupied_rank_s_bounds=[W/H[1],W/H[0]],steps=steps,
        log_sha256=hashlib.sha256(log.read_bytes()).hexdigest(),
        paper_conventions_sha256=hashlib.sha256(PAPER.read_bytes()).hexdigest(),
        uncertainty='Recorded-clock resolution, not a statistical confidence interval. Restore reoccupation charged. Not hardware busy time or billing.')
