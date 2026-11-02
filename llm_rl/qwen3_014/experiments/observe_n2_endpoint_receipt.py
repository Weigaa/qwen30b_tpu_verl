#!/usr/bin/env python3
"""Retain tiny provenance before the validated private endpoint is removed.

Read-only observer of source/endpoint inode identities. Never launches an
NPU, copies weight data, retains weight hardlinks or modifies primary jobs.
"""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import time

from run_n2_f8_campaign import write


def capture(root):
    endpoint=root/'heldout_checkpoint/global_step_5'
    path=endpoint/'RETENTION_RECEIPT.json'
    if not path.exists():return False
    receipt=json.loads(path.read_text())
    expected=root/'graph_sidecar/seed_0/natural_f2/fig6/epoch_001_mode1_natural/checkpoints/qwen3moe_for_eagle3/global_step_5'
    if (receipt.get('status')!='COMMITTED_ENDPOINT_HARDLINKED' or
            receipt.get('source')!=str(expected) or receipt.get('endpoint')!=str(endpoint) or
            receipt.get('additional_actor_updates_from_public_epoch0')!=5 or
            receipt.get('actor_frozen') is not False or receipt.get('seed')!=0):
        raise RuntimeError('Endpoint provenance does not match requested new N2 epoch1')
    for row in receipt['files']:
        dest=endpoint/row['path'];actual=dest.stat()
        if (actual.st_dev,actual.st_ino,actual.st_size)!=(row['device'],row['inode'],row['bytes']):
            raise RuntimeError('Retained endpoint file identity changed')
        source=expected/row['path']
        if source.exists():
            original=source.stat()
            if (actual.st_dev,actual.st_ino,actual.st_size)!=(original.st_dev,original.st_ino,original.st_size):
                raise RuntimeError('Endpoint is not the actual trained actor hardlink')
    write(root/'ENDPOINT_RETENTION_AUDIT.json',dict(
        status='NEW_N2_EPOCH1_ENDPOINT_INODE_IDENTITY_OBSERVED',
        observed_utc=datetime.now(timezone.utc).isoformat(),receipt=receipt,
        weight_data_copied=False,extra_weight_hardlinks_created=False,
        note='Durable small provenance copy; original inference logs must separately prove actual checkpoint load and256-question evaluation.'))
    return True


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',required=True,type=Path)
    root=p.parse_args().root.resolve()
    if (root/'ENDPOINT_RETENTION_AUDIT.json').exists():return
    while True:
        if capture(root):return
        status=json.loads((root/'STATE.json').read_text())['state']
        if status=='STOPPED_WITH_ERROR':
            write(root/'ENDPOINT_OBSERVER_STATE.json',dict(state='PRIMARY_QUEUE_STOPPED_BEFORE_ENDPOINT',
                scope='Not an endpoint failure; inspect/restart observer after primary resume.'))
            return
        if status=='ALL_AFFECTED_N2_EXPERIMENTS_COMPLETE':
            raise RuntimeError('Endpoint provenance disappeared before observer captured it')
        time.sleep(5)


if __name__=='__main__':main()
