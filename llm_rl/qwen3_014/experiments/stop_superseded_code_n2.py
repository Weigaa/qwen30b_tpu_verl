#!/usr/bin/env python3
"""User-requested stop of the specific old-budget code-N2 process tree only."""
import json
import os
from pathlib import Path
import signal
import time

ROOT=Path('/workspace/adafloor_graph_results/mlsys_v014_code_natural_f2_seed401_full_20261025T123000Z')


def identity(pid):
    try:
        fields=Path(f'/proc/{pid}/stat').read_text().rsplit(') ',1)[1].split()
        return dict(pid=pid,ppid=int(fields[1]),start=fields[19],state=fields[0])
    except (FileNotFoundError,ProcessLookupError):return None


def main():
    pid=json.loads((ROOT/'STATE.json').read_text())['controller_pid']
    command=Path(f'/proc/{pid}/cmdline').read_bytes()
    if str(ROOT).encode() not in command or b'run_code_natural_f2.py' not in command:
        raise RuntimeError('Not the specifically owned old-budget controller')
    processes={int(p.name):identity(int(p.name)) for p in Path('/proc').iterdir() if p.name.isdigit()}
    owned={pid}
    while True:
        more={p for p,v in processes.items() if v and v['ppid'] in owned}
        if more.issubset(owned):break
        owned.update(more)
    receipt=dict(status='USER_REQUESTED_STOP_SUPERSEDED_F8_BUDGET_NOT_RUNTIME_FAILURE',
        utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
        reason='User requested stopping old288256 F8 code-N2 and rerunning affected experiments at349952.',
        old_budget=288256,new_candidate_budget=349952,
        processes=[processes[p] for p in sorted(owned)],artifacts_retained=True)
    (ROOT/'USER_REQUESTED_STOP.json').write_text(json.dumps(receipt,indent=2)+'\n')
    for value in receipt['processes']:
        now=identity(value['pid'])
        if now and now['start']==value['start']:
            try:os.kill(value['pid'],signal.SIGTERM)
            except ProcessLookupError:pass
    time.sleep(5)
    for value in receipt['processes']:
        now=identity(value['pid'])
        if now and now['start']==value['start'] and now['state']!='Z':
            try:os.kill(value['pid'],signal.SIGKILL)
            except ProcessLookupError:pass
    (ROOT/'STATE.json').write_text(json.dumps(dict(state=receipt['status'],
        utc=receipt['utc'],reason=receipt['reason'],stop_receipt='USER_REQUESTED_STOP.json'),indent=2)+'\n')
    print(json.dumps(dict(status=receipt['status'],owned_processes=len(owned))))


if __name__=='__main__':main()
