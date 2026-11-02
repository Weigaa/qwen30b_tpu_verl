#!/usr/bin/env python3
"""Passively preserve real TaskRunner logs missed by console aggregation.

Never rewrite a live primary log or infer successful completion from a
partial snapshot. This observer does not import torch or touch an NPU.
"""
import hashlib
import json
from pathlib import Path
import re
import time

ROOT=Path('/data/adafloor_graph_results/mlsys_v014_supplement_20261023T180000Z')
PID=3882051


def identity():
    try:return Path(f'/proc/{PID}/stat').read_text().rsplit(') ',1)[1].split()[19]
    except FileNotFoundError:return None


def snapshot():
    destination=ROOT/'code_task/raw_taskrunner_snapshots'
    destination.mkdir(parents=True,exist_ok=True)
    for directory in Path('/tmp').glob(f'f6_401_*_{PID}/ray/session_*/logs'):
        tag=directory.parents[2].name
        for source in directory.glob('worker-*.out'):
            with source.open('rb') as stream:
                # vLLM plugin startup lines precede Ray's actor header.
                if b':actor_name:TaskRunner\n' not in stream.read(65536):continue
                stream.seek(0);raw=stream.read()
            out=destination/tag/source.name;out.parent.mkdir(exist_ok=True)
            if out.exists() and out.read_bytes()==raw:continue
            temporary=out.with_suffix('.tmp');temporary.write_bytes(raw);temporary.replace(out)
            steps=[int(re.search(r'step:(\d+)\s+-',line)[1]) for line in raw.decode(errors='replace').splitlines()
                if re.search(r'step:\d+\s+-',line)]
            receipt=dict(status='RAW_LOG_SNAPSHOT_NOT_COMPLETION',source=str(source),snapshot=str(out),
                sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw),observed_metric_step_ids=steps,
                observer_time=time.time(),controller_pid=PID,completion_proven=False)
            proof=out.with_suffix('.snapshot.json');temporary=proof.with_suffix('.tmp')
            temporary.write_text(json.dumps(receipt,indent=2)+'\n');temporary.replace(proof)
            print(json.dumps({k:v for k,v in receipt.items() if k not in ('source','snapshot')}),flush=True)


def main():
    original=identity()
    while original is not None and identity()==original:
        snapshot();time.sleep(15)
    snapshot()


if __name__=='__main__':main()
