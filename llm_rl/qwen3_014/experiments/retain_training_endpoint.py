#!/usr/bin/env python3
"""Retain a committed epoch1 actor with hardlinks, without touching its job."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shutil
import time


def ready(epoch):
    logs=list((epoch/'logs').glob('*.txt'))
    if len(logs)!=1:return None
    text=logs[0].read_text(errors='replace')
    if not re.search(r'\[run\] end_time=\S+ exit_code=0',text):return None
    if len(re.findall(r'(?:^|\s)step:\d+\s+-',text))!=5:return None
    if not (epoch/'offline_planning_history.json').is_file():return None
    source=epoch/'checkpoints/qwen3moe_for_eagle3/global_step_5'
    weights=source/'actor/dist_ckpt'
    shards=list(weights.glob('*.distcp'))
    if len(shards)!=32 or not (weights/'.metadata').is_file():return None
    if any(p.stat().st_size==0 for p in shards):return None
    return source


def retain(source,destination):
    destination.parent.mkdir(parents=True,exist_ok=True)
    if source.stat().st_dev!=destination.parent.stat().st_dev:
        raise RuntimeError('Refusing a physical full-weight copy across filesystems')
    if destination.exists():raise RuntimeError('Refusing existing endpoint snapshot')
    temporary=destination.with_name(destination.name+f'.incomplete.{os.getpid()}')
    shutil.copytree(source,temporary,copy_function=os.link)
    files=[]
    for path in source.rglob('*'):
        if not path.is_file():continue
        target=temporary/path.relative_to(source)
        a,b=path.stat(),target.stat()
        if (a.st_dev,a.st_ino,a.st_size)!=(b.st_dev,b.st_ino,b.st_size):
            raise RuntimeError('Snapshot is not a complete hardlink identity')
        files.append(dict(path=str(path.relative_to(source)),bytes=a.st_size,inode=a.st_ino,device=a.st_dev))
    receipt=dict(status='COMMITTED_ENDPOINT_HARDLINKED',source=str(source),endpoint=str(destination),
        logical_epoch=1,additional_actor_updates_from_public_epoch0=5,
        seed=0,policy='natural_f2',actor_frozen=False,files=files,
        content_hashing_deferred_until_inference_job_finishes=True,
        created_utc=datetime.now(timezone.utc).isoformat(),
        scope='Endpoint retention only; neither held-out evaluation nor two-epoch completion is claimed.')
    (temporary/'RETENTION_RECEIPT.json').write_text(json.dumps(receipt,indent=2)+'\n')
    temporary.rename(destination)
    return receipt


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--epoch',type=Path,required=True)
    parser.add_argument('--destination',type=Path,required=True);parser.add_argument('--controller-pid',type=int,required=True)
    args=parser.parse_args();deadline=time.monotonic()+5*3600
    while time.monotonic()<deadline:
        source=ready(args.epoch)
        if source:
            print(json.dumps(retain(source,args.destination)),flush=True);return
        try:os.kill(args.controller_pid,0)
        except ProcessLookupError:raise RuntimeError('Controller exited before committed epoch1 endpoint was retained')
        time.sleep(10)
    raise RuntimeError('Endpoint retention observation deadline exceeded')


if __name__=='__main__':main()
