#!/usr/bin/env python3
"""Stop only this task's non-final device-metadata pilot, preserving evidence."""
import json
import os
from pathlib import Path
import signal
import time

ROOT=Path('/data/adafloor_graph_results/mlsys_v014_supplement_20261023T180000Z')


def stat(pid):
    try:
        fields=Path(f'/proc/{pid}/stat').read_text().rsplit(') ',1)[1].split()
        return dict(pid=pid,ppid=int(fields[1]),state=fields[0],start=fields[19])
    except (FileNotFoundError,ProcessLookupError):return None


def main():
    parent=3352390;child=3369983
    if b'run_eager_controls_when_ready.py' not in Path(f'/proc/{parent}/cmdline').read_bytes():
        raise RuntimeError('Not the owned pilot controller')
    if stat(child)['ppid']!=parent:raise RuntimeError('Pilot child identity changed')
    all_processes={int(p.name):stat(int(p.name)) for p in Path('/proc').iterdir() if p.name.isdigit()}
    owned={parent}
    while True:
        more={pid for pid,info in all_processes.items() if info and info['ppid'] in owned}
        if more.issubset(owned):break
        owned.update(more)
    record=dict(status='INTENTIONAL_PROTOCOL_ALIGNMENT_STOP_NOT_RUNTIME_FAILURE',time=time.time(),
        reason='Pilot uses EAGER_METADATA_SYNC_DEVICE1 while matched historical Graph uses0. Final configured-path comparison will use CPU metadata0 in both, same frozen source/checkpoint/input/KV and TQ2/TQ1. No performance conclusion from this pilot.',
        processes=[all_processes[pid] for pid in sorted(owned)],controllers=[])
    # Stop standby successors first so an intentional predecessor exit cannot
    # trigger evaluation or another NPU job. They own no running NPU child.
    for pid,name in [(3409384,b'run_code_pair_when_ready.py'),(3409149,b'run_eager_controls_when_ready.py')]:
        p=Path(f'/proc/{pid}/cmdline')
        if p.exists():
            if name not in p.read_bytes():raise RuntimeError('Standby controller identity changed')
            record['controllers'].append(stat(pid));os.kill(pid,signal.SIGTERM)
    (ROOT/'EAGER_DEVICE_META_PILOT_INTENTIONAL_STOP.json').write_text(json.dumps(record,indent=2)+'\n')
    for info in record['processes']:
        current=stat(info['pid'])
        if current and current['start']==info['start']:
            try:os.kill(info['pid'],signal.SIGTERM)
            except ProcessLookupError:pass
    print(json.dumps(dict(status=record['status'],owned_processes=len(owned),standby_controllers=len(record['controllers']))))


if __name__=='__main__':main()
