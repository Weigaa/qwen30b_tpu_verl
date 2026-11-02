#!/usr/bin/env python3
"""Wait for the already-launched experiment, then run CPU-only result audits.

Never starts or restarts an NPU workload, and never changes its environment.
Heavy offline planning is deferred until both measured arms have finished.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-root', type=Path, required=True)
    args = parser.parse_args()
    root = args.run_root.resolve()
    package = Path(__file__).resolve().parents[1]
    supervisor = json.loads((root / 'run_process.json').read_text())['supervisor_pid']
    state = root / 'postprocess_status.json'
    if state.exists():
        raise FileExistsError('postprocessor already has a status record')

    def record(status, **values):
        state.write_text(json.dumps({'status': status, 'pid': os.getpid(),
                                     'experiment_supervisor_pid': supervisor, **values},
                                    sort_keys=True, indent=2) + '\n')

    record('WAITING_FOR_NPU_ARMS')
    print(f'Waiting for existing supervisor pid={supervisor}. No NPU launch.', flush=True)
    while not (root / 'completion.json').exists():
        if (root / 'failure.json').exists():
            failure = json.loads((root / 'failure.json').read_text())
            record('EXPERIMENT_FAILED', failure=failure)
            print(json.dumps(failure), flush=True)
            return 1
        try:
            os.kill(supervisor, 0)
        except ProcessLookupError:
            record('MISSING_COMPLETION_RECORD')
            return 1
        time.sleep(30)
    completion = json.loads((root / 'completion.json').read_text())
    if completion.get('status') != 'PASS':
        record('EXPERIMENT_NOT_VALIDATED', completion=completion)
        return 1
    record('CPU_ANALYSIS_RUNNING')
    cpu = root / 'paper_cpu_sensitivity.json'
    output = root / 'analysis_final'
    with (root / 'postprocess.log').open('w') as log:
        try:
            subprocess.run([sys.executable, str(package / 'analysis_eval/analyze_revision_forecast_sensitivity.py'),
                            '--scenario-root', str(root / 'scenarios'), '--output', str(cpu)],
                           check=True, stdout=log, stderr=subprocess.STDOUT)
            subprocess.run([sys.executable, str(package / 'analysis_eval/summarize_paper_aligned_forecast.py'),
                            '--run-root', str(root), '--output-dir', str(output)],
                           check=True, stdout=log, stderr=subprocess.STDOUT)
        except subprocess.CalledProcessError as error:
            record('POSTPROCESS_FAILED', exit_code=error.returncode)
            raise
    summary = json.loads((output / 'summary.json').read_text())
    record('ANALYSIS_PASS' if summary['status'] == 'PASS' else 'ANALYSIS_NOT_PASS',
           analysis_status=summary['status'], cpu_matrix=str(cpu), summary_dir=str(output))
    print(f"Offline audit finished: {summary['status']}, output={output}", flush=True)
    return 0 if summary['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
