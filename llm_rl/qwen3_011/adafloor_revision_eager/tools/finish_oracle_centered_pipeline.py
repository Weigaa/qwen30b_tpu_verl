#!/usr/bin/env python3
"""Wait for existing online arms, then execute the approved replay controls."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, required=True)
    args = ap.parse_args()
    root = args.root.resolve()
    package = Path(__file__).resolve().parents[1]
    online = json.loads((root/'supervisor_reference_current.json').read_text())
    pid = online['pid']
    state = root/'pipeline_status.json'
    if state.exists():
        raise FileExistsError('pipeline already has a status record')
    def record(status, **extra):
        state.write_text(json.dumps({'status': status, 'pid': os.getpid(),
                                    'online_supervisor_pid': pid, **extra}, sort_keys=True, indent=2)+'\n')
    record('WAITING_FOR_EXISTING_ONLINE_WINDOWS')
    print(f'Waiting for online supervisor {pid}. No duplicate online run.', flush=True)
    try:
        while not (root/'complete_reference_current.json').exists():
            if (root/'failure_reference_current.json').exists():
                raise RuntimeError('online windows failed, evidence retained')
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                raise RuntimeError('online supervisor disappeared without completion')
            time.sleep(30)
        record('PREPARING_REFERENCE_BANK')
        with (root/'pipeline.log').open('w') as log:
            subprocess.run([sys.executable, str(package/'tools/build_oracle_reference_bank.py'),
                            '--root', str(root)], check=True, stdout=log, stderr=subprocess.STDOUT)
            record('CONTROLLED_REPLAYS_RUNNING')
            subprocess.run([sys.executable, str(package/'tools/run_oracle_centered_replays.py'),
                            '--root', str(root), '--stages', 'oracle', 'under_0p7', 'over_1p3', 'mixed'],
                           check=True, stdout=log, stderr=subprocess.STDOUT)
            record('FINAL_AUDIT_RUNNING')
            subprocess.run([sys.executable, str(package/'analysis_eval/summarize_oracle_centered_n2.py'),
                            '--root', str(root), '--output-dir', str(root/'analysis_final')],
                           check=True, stdout=log, stderr=subprocess.STDOUT)
        record('ALL_SIX_WINDOWS_AND_ANALYSIS_COMPLETE')
        print('Reference, current and four replay controls completed. Final evidence is ready.', flush=True)
    except Exception as error:
        record('PIPELINE_FAILED', error=str(error))
        raise


if __name__ == '__main__':
    main()
