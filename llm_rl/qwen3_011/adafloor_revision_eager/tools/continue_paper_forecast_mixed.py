#!/usr/bin/env python3
"""User-approved continuation of the already-planned Mixed pressure arm.

Preserve the original under-0.7 failure/audit. Continue only after that full
window exits cleanly and passes functional checks, except KV preemption.
Never edit model/runtime configuration or rerun under-0.7.
"""
import argparse
import importlib.util
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
    spec = importlib.util.spec_from_file_location('aligned_runner', package / 'tools/run_paper_aligned_natural_f2_forecast.py')
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    original_pid = json.loads((root / 'run_process.json').read_text())['supervisor_pid']
    state = root / 'mixed_continuation_status.json'
    if state.exists():
        raise FileExistsError('continuation already has a status record')

    def record(status, **values):
        state.write_text(json.dumps({'status': status, 'pid': os.getpid(),
                                     'original_supervisor_pid': original_pid,
                                     'user_approved_continue_with_preemptions': True,
                                     **values}, sort_keys=True, indent=2) + '\n')

    record('WAITING_FOR_ORIGINAL_WINDOW')
    print(f'Waiting for existing under-0.7 window pid={original_pid}. No rerun.', flush=True)
    while True:
        try:
            os.kill(original_pid, 0)
        except ProcessLookupError:
            break
        time.sleep(30)
    try:
        # Normal original completion means it already ran Mixed. Never duplicate.
        if (root / 'completion.json').exists():
            record('ORIGINAL_SEQUENCE_ALREADY_COMPLETE')
            return 0
        under = root / 'runs/under_0p7'
        exit_record = json.loads((under / 'exit.json').read_text())
        audit = json.loads((under / 'audit.json').read_text())
        errors = list(audit['errors'])
        if (exit_record['exit_code'] != 0 or audit['oom'] or audit['aborted']
                or audit['response_counts'] != [512] * 5 or audit['length_counts'] != [512] * 5
                or any(not e.startswith('preemptions=') for e in errors)):
            raise RuntimeError('under-0.7 failed functional/oom/abort validation, not only KV preemption')
        if (root / 'runs/mixed').exists():
            raise RuntimeError('Mixed has already been started, refusing duplicate NPU run')
        live = subprocess.run(['pgrep', '-f', '^python3 -m verl.trainer.main_ppo'], capture_output=True)
        if live.returncode == 0:
            raise RuntimeError('another trainer is live, refusing overlap')
        # Assert original paper runtime/launcher files have not changed.
        source_hashes = json.loads((root / 'code_sha256.json').read_text())
        for name, digest in source_hashes.items():
            if name.startswith('/') or name == 'forecast_scenario_generator':
                continue
            if runner.sha(runner.PAPER / name) != digest:
                raise RuntimeError(f'paper runtime source changed: {name}')
        write_json = runner.write_json
        write_json(root / 'mixed_continuation_protocol.json', {
            'reason': 'Explicit user approval to finish Mixed while retaining failed zero-preemption gates',
            'under_0p7_audit_status': audit['status'], 'under_0p7_preemptions': audit['preemption_count'],
            'original_failure_retained': (root / 'failure.json').exists(),
            'configuration_changes': [], 'seed': 505, 'variant': 'natural_f2',
            'checkpoint': str(runner.CHECKPOINT), 'supervisor_sha256': runner.sha(__file__),
        })
        record('MIXED_RUNNING')
        runner.launch(root, 'mixed', False)
        try:
            runner.audit_run(root, 'mixed')
        except RuntimeError:
            mixed = json.loads((root / 'runs/mixed/audit.json').read_text())
            if (mixed['oom'] or mixed['aborted'] or mixed['response_counts'] != [512] * 5
                    or mixed['length_counts'] != [512] * 5
                    or any(not e.startswith('preemptions=') for e in mixed['errors'])):
                raise
        # Strict safety statuses remain FAIL if either arm preempted. Completion
        # of measurement is a separate fact and is never written as safety PASS.
        write_json(root / 'measurement_completion.json', {
            'measurement_complete': True,
            'under_0p7_safety': json.loads((under / 'audit.json').read_text())['status'],
            'mixed_safety': json.loads((root / 'runs/mixed/audit.json').read_text())['status'],
            'original_failure_retained': (root / 'failure.json').exists(),
        })
        record('CPU_ANALYSIS_RUNNING')
        with (root / 'postprocess_after_mixed.log').open('w') as log:
            subprocess.run([sys.executable, str(package / 'analysis_eval/analyze_revision_forecast_sensitivity.py'),
                            '--scenario-root', str(root / 'scenarios'),
                            '--output', str(root / 'paper_cpu_sensitivity.json')], check=True,
                           stdout=log, stderr=subprocess.STDOUT)
            subprocess.run([sys.executable, str(package / 'analysis_eval/summarize_paper_aligned_forecast.py'),
                            '--run-root', str(root), '--output-dir', str(root / 'analysis_final')],
                           check=True, stdout=log, stderr=subprocess.STDOUT)
        record('MEASUREMENTS_AND_ANALYSIS_COMPLETE_SAFETY_STATUS_PRESERVED')
        print('Mixed and offline analyses complete. Original failure records retained.', flush=True)
        return 0
    except Exception as error:
        record('CONTINUATION_FAILED', error=str(error))
        raise


if __name__ == '__main__':
    raise SystemExit(main())
