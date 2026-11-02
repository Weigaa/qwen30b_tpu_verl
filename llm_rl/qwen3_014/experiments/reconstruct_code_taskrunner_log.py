#!/usr/bin/env python3
"""Post-run audit view from actual Ray stdout; never invent success/events.

Keep the original aggregated log and nonzero parent result. Only after the
controller has exited can --install select this explicitly derived view for
the unchanged offline fair finalizer. No model/runtime source is changed.
"""
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re

SOURCE=Path('/workspace/cann-recipes-train/llm_rl/qwen3_014/qwen3_main')
ROOT=Path('/data/adafloor_graph_results/mlsys_v014_supplement_20261023T180000Z/code_task')
ANSI=re.compile(r'\x1b\[[0-9;?]*[ -/]*[@-~]')
STEP=re.compile(r'(?:^|\s)step:(\d+)\s+-')
RETURN=re.compile(r'=> rollout_output_time_s:\s*([0-9.]+)')


def metric(line):
    match=STEP.search(line)
    if not match:return None
    body=line[match.start():].strip()
    return int(match[1]),dict(re.findall(r'([A-Za-z0-9_./-]+):([^ ]+)',body))


def derive(original,raw,pid,steps=5,stderr=''):
    original=ANSI.sub('',original).replace('\r','\n')
    raw=ANSI.sub('',raw).replace('\r','\n')
    if not re.search(r'^\[run\] end_time=\S+ exit_code=0\s*$',original,re.M):
        raise ValueError('Actual successful trainer child exit missing; no derived success')
    records={};returns=[];events=[]
    for number,line in enumerate(raw.splitlines(),1):
        row=metric(line);end=RETURN.search(line)
        if row:
            step,values=row
            if step in records:raise ValueError('Duplicate raw actor metric')
            if (float(values['actor/lr'])!=0 or not math.isfinite(float(values['actor/grad_norm']))
                    or float(values['response/aborted_ratio'])!=0):
                raise ValueError('Not complete finite frozen actor execution')
            records[step]=values
        if end:returns.append(float(end[1]))
        if row or end:events.append((number,line))
    if set(records)!=set(range(1,steps+1)) or len(returns)!=steps:
        raise ValueError('Raw log lacks complete actual step/timing coverage')
    for step,duration in enumerate(returns,1):
        if not math.isclose(duration,float(records[step]['rollout/rollout_output_time_s']),abs_tol=2e-6,rel_tol=0):
            raise ValueError('Raw timing and actor metric disagree')
    kept=[];replaced=[]
    for line in original.splitlines():
        row=metric(line);end=RETURN.search(line)
        if row:
            step,values=row
            if step not in records or values!=records[step]:
                raise ValueError('Aggregated/raw actor metric conflict')
        if end and float(end[1]) not in returns:
            raise ValueError('Aggregated/raw return timing conflict')
        if row or end:replaced.append(line)
        else:kept.append(line)
    progress=[]
    if 'Training Progress: 100%' not in original:
        progress=[line for line in ANSI.sub('',stderr).replace('\r','\n').splitlines()
            if 'Training Progress: 100%' in line]
        if not progress:raise ValueError('Actual completed training-progress evidence absent in both sources')
    header=['# DERIVED OFFLINE AUDIT VIEW, NOT AN ORIGINAL CONSOLE CAPTURE.',
        '# Worker/child records preserved from aggregated log. TaskRunner events below are copied from actual raw stdout.',
        '# Original files, hashes and raw line numbers are recorded in LOG_RECONSTRUCTION_PROVENANCE.json.']
    body='\n'.join(header+kept+['# BEGIN REAL RAW TASKRUNNER EVENTS']+
        [f'(TaskRunner pid={pid}) '+line for _,line in events]+
        [f'(TaskRunner pid={pid}) '+line for line in progress])+'\n'
    return body,dict(raw_metric_step_ids=sorted(records),raw_event_line_numbers=[n for n,_ in events],
        replaced_forwarded_records=len(replaced),copied_real_progress_lines=len(progress))


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--epoch',type=Path,required=True)
    p.add_argument('--raw-taskrunner',type=Path,required=True)
    p.add_argument('--raw-stderr',type=Path)
    p.add_argument('--controller-pid',type=int,required=True)
    p.add_argument('--install',action='store_true')
    args=p.parse_args();epoch=args.epoch.resolve()
    if not epoch.is_relative_to(ROOT) or epoch.name not in ('epoch_001_mode0_vanilla','epoch_001_mode1_planned'):
        raise ValueError('Not an owned code-task epoch')
    if args.install and Path(f'/proc/{args.controller_pid}').exists():
        raise ValueError('Controller still live; do not change its active log selection')
    arm=epoch.parent.parent
    if args.install:
        result=arm/'DRIVER_RESULT.json'
        if not result.is_file() or json.loads(result.read_text())['exit_code']==0:
            raise ValueError('Require retained nonzero parent result; no normal arm overwritten')
    logs=list((epoch/'logs').glob('*.txt'))
    if len(logs)!=1:raise ValueError('Ambiguous original aggregated log')
    original=logs[0]
    if (epoch/'LOG_RECONSTRUCTION_PROVENANCE.json').exists():
        raise ValueError('A reconstruction is already recorded; no repeated rewrite')
    match=re.search(r'-(\d+)\.out$',args.raw_taskrunner.name)
    if not match:raise ValueError('Actual TaskRunner PID missing from source filename')
    raw=args.raw_taskrunner.read_bytes();err=args.raw_stderr.read_bytes() if args.raw_stderr else b''
    text,evidence=derive(original.read_text(errors='replace'),raw.decode(errors='replace'),int(match[1]),
        stderr=err.decode(errors='replace'))
    for step in range(1,6):
        for folder,name in [('rollout_data',f'{step}.jsonl'),('rollout_length',f'length_{step}.txt')]:
            if sum(bool(line.strip()) for line in (epoch/folder/name).read_text().splitlines())!=512:
                raise ValueError('Missing actual512 saved outputs/lengths')
    preview=arm/'DERIVED_RAW_TASKRUNNER_AUDIT.log';preview.write_text(text)
    verifier=SOURCE/'tools/verify_full_decode_preemption_recovery.py'
    spec=importlib.util.spec_from_file_location('unchanged_native_code_recovery_auditor',verifier)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    recovery=module.audit_native_v014(preview,16,5)
    if recovery['status']!='PASS':raise ValueError('Actual native recovery audit failed')
    proof=dict(status='POST_RUN_REAL_RAW_LOG_VIEW_NATIVE_AUDITED',original_log=str(original),
        original_sha256=sha(original),raw_taskrunner=str(args.raw_taskrunner),raw_stdout_sha256=hashlib.sha256(raw).hexdigest(),
        raw_stderr=str(args.raw_stderr) if args.raw_stderr else None,raw_stderr_sha256=hashlib.sha256(err).hexdigest(),
        derived_sha256=sha(preview),native_recovery=recovery,source_verifier_sha256=sha(verifier),
        normalizer_sha256=sha(Path(__file__)),original_parent_result=json.loads((arm/'DRIVER_RESULT.json').read_text()),
        training_repeated=False,model_source_modified=False,**evidence)
    if args.install:
        archived=original.with_suffix('.txt.aggregated-original')
        if archived.exists():raise ValueError('Original archive already exists')
        original.rename(archived)
        selected=epoch/'logs/DERIVED_FROM_REAL_RAW_TASKRUNNER.txt';preview.replace(selected)
        proof.update(archived_original=str(archived),selected_derived=str(selected))
    else:proof['installed']=False
    (epoch/'LOG_RECONSTRUCTION_PROVENANCE.json').write_text(json.dumps(proof,indent=2)+'\n')
    print(json.dumps({k:v for k,v in proof.items() if k!='native_recovery'}))


if __name__=='__main__':main()
