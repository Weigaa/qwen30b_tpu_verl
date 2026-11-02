#!/usr/bin/env python3
"""Watch the actual campaign process, then audit all new N2 evidence once.

No inference, replanning, scoring retry, checkpoint deletion or paper edit.
Successful offline audit still requires primary-agent review before the
thread goal can be marked complete.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

from run_n2_f8_campaign import write


def signature(pid):
    try:
        fields=Path(f'/proc/{pid}/stat').read_text().rsplit(') ',1)[1].split()
        return None if fields[0]=='Z' else fields[19]
    except FileNotFoundError:return None


def ready(root):
    if json.loads((root/'STATE.json').read_text())['state']!='ALL_AFFECTED_N2_EXPERIMENTS_COMPLETE':
        raise RuntimeError('Main campaign did not complete; no final success claim')
    path=root/'ALL_DEPENDENCIES_COMPLETE.json'
    return path.exists() and json.loads(path.read_text())['status']=='ALL_NEW_F8_DEPENDENCIES_VALIDATED'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',required=True,type=Path)
    parser.add_argument('--controller-pid',required=True,type=int)
    args=parser.parse_args();root=args.root.resolve()
    ident=signature(args.controller_pid)
    if ident is None:raise RuntimeError('Specified controller is not live; inspect instead of assuming completion')
    argv=Path(f'/proc/{args.controller_pid}/cmdline').read_bytes().split(b'\0')
    if str(root).encode() not in argv or not any(
            p in argv for p in (b'run_n2_f8_campaign.py',b'recover_n2_seed3_epoch2.py',
                               b'resume_n2_postrun_summary.py')):
        raise RuntimeError('Controller does not belong to this campaign')
    write(root/'FINAL_N2_AUDIT_QUEUE_STATE.json',dict(state='WAITING_ACTUAL_CAMPAIGN_PROCESS',
        controller_pid=args.controller_pid,controller_start_identity=ident))
    while signature(args.controller_pid)==ident:time.sleep(15)
    for _ in range(5):
        if ready(root):break
        time.sleep(15)
    else:raise RuntimeError('Dependency validator has not completed; no final audit')
    command=[sys.executable,str(Path(__file__).with_name('audit_n2_f8_results.py')),
             '--root',str(root),'--require-complete']
    write(root/'FINAL_N2_AUDIT_QUEUE_STATE.json',dict(state='AUDITING_ALL_NEW_N2_RAW_EVIDENCE'))
    with (root/'final_new_n2_audit.log').open('x') as log:
        result=subprocess.run(command,cwd=Path(__file__).parent,stdout=log,stderr=subprocess.STDOUT)
    write(root/'FINAL_N2_AUDIT_PROCESS_RESULT.json',dict(command=command,exit_code=result.returncode))
    if result.returncode:raise RuntimeError('Raw evidence audit failed; inspect artifacts, do not rerun experiments')
    proof=json.loads((root/'NEW_N2_AUDIT_REPORT.json').read_text())
    if proof['status']!='ALL_NEW_N2_EXPERIMENTS_RAW_EVIDENCE_AUDITED' or proof['pending']:
        raise RuntimeError('Auditor did not prove the entire requested scope')
    write(root/'FINAL_N2_AUDIT_QUEUE_STATE.json',dict(state='OFFLINE_AUDIT_COMPLETE_REQUIRES_PRIMARY_REVIEW',
        report=str(root/'NEW_N2_AUDIT_REPORT.json'),summary=str(root/'NEW_N2_RESULTS.md')))


if __name__=='__main__':main()
