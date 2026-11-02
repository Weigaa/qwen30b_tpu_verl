#!/usr/bin/env python3
"""Read-only process/log progress, never a completion or correctness proof."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import re


def code_root(root):
    return Path('/workspace/adafloor_graph_results')/(root.name+'_code_n2_seed401')


def ray_environment(path):
    """Read only the Ray directory; never return or print other env values."""
    try:
        for row in path.read_bytes().split(b'\0'):
            if row.startswith(b'RAY_TMPDIR='):return row.split(b'=',1)[1].decode()
    except (FileNotFoundError,ProcessLookupError,PermissionError):pass
    return None


def progress_record(line):
    if 'rollout_forward_begin' not in line:return None
    time=re.search(r'\b(\d\d:\d\d:\d\d)\b',line)
    fields=dict(re.findall(r'(dp_rank|iteration|requests|graph_mode)=([^ ]+)',line))
    if not time or len(fields)!=4:return None
    return dict(time=time[1],rank=int(fields['dp_rank']),iteration=int(fields['iteration']),
                requests=int(fields['requests']),graph_mode=fields['graph_mode'])


def processes(root):
    records=[]
    for path in Path('/proc').glob('[0-9]*/cmdline'):
        try:
            args=path.read_bytes().rstrip(b'\0').split(b'\0')
            if not args or not re.fullmatch(rb'python[0-9.]*',args[0].rsplit(b'/',1)[-1]):continue
            joined=b' '.join(args)
            if not any(str(p).encode() in joined for p in (root,code_root(root))):continue
            if not any(x in args for x in (b'run_n2_f8_campaign.py',
                    b'recover_n2_seed3_epoch2.py',b'resume_n2_postrun_summary.py',
                    b'verl.trainer.main_ppo')):continue
            stat=(path.parent/'stat').read_text().rsplit(') ',1)[1].split()
            record=dict(pid=int(path.parent.name),ppid=int(stat[1]),state=stat[0],
                        start_ticks=stat[19],trainer=b'verl.trainer.main_ppo' in args,
                        scope='code' if str(code_root(root)).encode() in joined else 'main')
            if record['trainer']:record['ray_tmpdir']=ray_environment(path.parent/'environ')
            records.append(record)
        except (FileNotFoundError,ProcessLookupError,PermissionError):continue
    return records


def observe(root):
    state=json.loads((root/'STATE.json').read_text())
    launches=[]
    for path in root.rglob('LAUNCH.json'):
        value=json.loads(path.read_text())
        if 'RAY_TMPDIR' in value.get('env',{}):launches.append((path.stat().st_mtime,path,value))
    code=code_root(root)
    extra=code/'seed_401/natural_f2/LAUNCH.json'
    if extra.exists():launches.append((extra.stat().st_mtime,extra,json.loads(extra.read_text())))
    result=dict(utc=datetime.now(timezone.utc).isoformat(),state_record=state,
        live_processes=processes(root),completed_math_seeds=[int(p.parent.parent.name.split('_')[-1])
             for p in (root/'primary').glob('seed_*/natural_f2/ARM_PASS.json')],
        observation_only=True,all_experiments_audited=False)
    if not launches:return result
    _,path,value=max(launches,key=lambda row:row[0])
    result['last_launch']=str(path)
    ray_dir=value['env'].get('RAY_TMPDIR')
    if not ray_dir and path==extra:
        actual=[p['ray_tmpdir'] for p in result['live_processes']
                if p['trainer'] and p.get('scope')=='code' and p.get('ray_tmpdir')]
        if len(set(actual))==1:
            ray_dir=actual[0];result['ray_tmpdir_source']='actual_scoped_trainer_environment'
    if not ray_dir:
        result['ray_session_present']=False
        result['runtime_observation']='No scoped live trainer Ray directory available; no old-session substitution.'
        return result
    ray=Path(ray_dir)
    result['ray_session_present']=(ray/'ray/session_latest').exists()
    progress=[];audits=[];errors=[];metrics=[]
    for file in ray.glob('ray/session_latest/logs/worker*01000000*'):
        if file.suffix not in ('.out','.err'):continue
        for line in file.read_text(errors='replace').splitlines():
            record=progress_record(line)
            if record:progress.append(record)
            if 'MODE1_STEP_RUNTIME_AUDIT ' in line:
                audits.append(json.JSONDecoder().raw_decode(line.split('MODE1_STEP_RUNTIME_AUDIT ',1)[1])[0])
            if re.search('OutOfMemoryError|RayTaskError|NPU out of memory|KeyError',line):errors.append(line[:400])
            if re.search(r'(?:^|\s)step:\d+\s+-',line):
                fields=dict(re.findall(r'([\w/]+):([^ ]+)',line))
                metrics.append({k:fields.get(k) for k in ('step','critic/score/mean','actor/lr',
                    'rollout/rollout_output_time_s','rollout/rollout_tokens_per_sec','response/aborted_ratio')})
    result.update(latest_full_progress=sorted(progress,key=lambda p:p['time'])[-2:],
        actual_rank_step_audits=len(audits),preemptions=sum(a['preemptions'] for a in audits),
        committed_step_metrics=metrics,observed_runtime_errors=errors[-5:])
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',required=True,type=Path)
    print(json.dumps(observe(parser.parse_args().root.resolve()),ensure_ascii=False,indent=2))
