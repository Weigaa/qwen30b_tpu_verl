#!/usr/bin/env python3
"""Prevent the old in-memory controller from launching a one-question eval."""
import json
from pathlib import Path

ROOT=Path('/data/adafloor_graph_results/mlsys_v014_supplement_20261023T180000Z')


if __name__=='__main__':
    command=Path('/proc/3352390/cmdline').read_bytes()
    if b'run_eager_controls_when_ready.py' not in command:raise RuntimeError('Original controller not live')
    output=ROOT/'heldout_results/vanilla'
    gate=output/'HELDOUT_RUNNING_vanilla_LAUNCH.json'
    gate.mkdir(parents=True,exist_ok=False)
    (gate/'CONTROL_GATE.json').write_text(json.dumps(dict(
        purpose='prevent_legacy_fraction_evaluation',actual_NPU_launch=False,
        controller_pid=3352390,training_child_pid=3369983,
        reason='Legacy loaded evaluation config inherits dataset_fraction0.005. Stop only at its existing duplicate-launch guard, after current training succeeds; successor uses fraction1.0 and all256 heldout questions.'),indent=2)+'\n')
    (output/'CONTROL_GATE_README.md').write_text(
        'This directory sentinel is planned control metadata, NOT a launched evaluation or result. '
        'Current training is unchanged and continues. The existing controller will refuse its evaluation before any NPU launch. '
        'The successor requires actual training exit0 and audits before moving this sentinel and running the corrected256-question evaluation.\n')
    print('Armed directory-only future evaluation gate; no process interrupted or experiment rerun')
