#!/usr/bin/env python3
"""Wait for the actual code queue, then audit all five evidence sets offline.

Does not launch or retry generation/scoring, remove checkpoints, modify the
paper, or declare the thread goal achieved. Final success still requires
the primary agent to inspect the audit and its requirement-level evidence.
"""
import json
from pathlib import Path
import subprocess
import sys
import time

CODE=Path(__file__).resolve().parent
ROOT=Path('/data/adafloor_graph_results/mlsys_v014_supplement_20261023T180000Z')
PRIOR_PID=3882051


def signature(pid):
    try:return Path(f'/proc/{pid}/stat').read_text().rsplit(') ',1)[1].split()[19]
    except FileNotFoundError:return None


def state(value,**extra):
    path=ROOT/'FINAL_AUDIT_QUEUE_STATE.json';temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(dict(state=value,time=time.time(),**extra),indent=2)+'\n')
    temporary.replace(path)


def gate(root=ROOT):
    queue=json.loads((root/'CODE_QUEUE_STATE.json').read_text())
    result=root/'code_task/CODE_PAIR_RESULT.json'
    if (queue['state']!='COMPLETE' or not result.is_file() or
            json.loads(result.read_text())['status']!='V014_CODE_DOMAIN_FROZEN_PAIR_COMPLETE'):
        raise RuntimeError('Actual code pair/scoring not complete; no final audit or automatic retry')


def main():
    old=signature(PRIOR_PID)
    while old is not None and signature(PRIOR_PID)==old:
        state('WAITING_FOR_ACTUAL_CODE_QUEUE',controller_pid=PRIOR_PID,controller_start_identity=old)
        time.sleep(15)
    gate()
    if (ROOT/'FINAL_AUDIT_PROCESS_RESULT.json').exists():
        raise RuntimeError('Refusing duplicate final-audit process; inspect its original result')
    state('AUDITING_ALL_FIVE_OFFLINE')
    command=[sys.executable,str(CODE/'audit_supplement_completion.py')]
    with (ROOT/'final_supplement_audit.log').open('x') as log:
        result=subprocess.run(command,cwd=CODE,stdout=log,stderr=subprocess.STDOUT)
    (ROOT/'FINAL_AUDIT_PROCESS_RESULT.json').write_text(json.dumps(dict(command=command,exit_code=result.returncode))+'\n')
    if result.returncode:
        raise RuntimeError('Offline final auditor failed; preserve artifacts, do not rerun NPU experiments')
    proof=ROOT/'FINAL_SUPPLEMENT_AUDIT.json'
    if (not proof.is_file() or json.loads(proof.read_text())['status']!=
            'ALL_FIVE_MINIMAL_V014_SUPPLEMENT_REQUIREMENTS_AUDITED'):
        raise RuntimeError('No complete requirement-level audit artifact despite process exit0')
    state('OFFLINE_AUDIT_COMPLETE_REQUIRES_PRIMARY_REVIEW',final_audit=str(proof))


if __name__=='__main__':
    try:main()
    except Exception as error:state('STOPPED_WITH_ERROR',error=repr(error));raise
