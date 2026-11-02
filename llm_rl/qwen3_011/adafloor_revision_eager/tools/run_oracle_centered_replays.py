#!/usr/bin/env python3
"""Execute the four controlled arms with the separately validated replay tree."""
import json
from pathlib import Path

import run_oracle_centered_n2 as runner

HERE = Path(__file__).resolve()
original_configure = runner.configure


def configure(root, stage):
    if stage not in ('oracle', 'under_0p7', 'over_1p3', 'mixed'):
        raise ValueError('this launcher is restricted to controlled replay arms')
    completion = json.loads((root / 'complete_reference_current.json').read_text())
    bank = json.loads((root / 'reference_bank/manifest.json').read_text())
    if not completion['measurement_complete'] or bank['status'] != 'EXACT_REFERENCE_BANK_VERIFIED':
        raise RuntimeError('online windows or reference bank incomplete')
    base = original_configure(root, stage)
    base.PAPER = root / 'paper_runtime_replay'
    base.DRIVER = base.PAPER / 'run_mode1_dynamic_length_aware_adaptive_floor2_natural_tailguard_reuse_epoch0_2epoch.sh'
    return base


def instrumented_sources(root, base):
    runtime = base.PAPER
    manifest = json.loads((runtime / 'ORACLE_SOURCE_MANIFEST.json').read_text())
    allowed = {'verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py',
               'vllm_ascend/worker/model_runner_v1.py'}
    actual = {}
    for name, before in manifest['files_sha256_before_instrumentation'].items():
        actual[name] = base.sha(runtime/name)
        if name not in allowed and actual[name] != before:
            raise RuntimeError(f'unexpected replay source modification: {name}')
    actual['verl/utils/oracle_control.py'] = base.sha(runtime/'verl/utils/oracle_control.py')
    actual[str(HERE)] = base.sha(HERE)
    actual[str(Path(runner.__file__).resolve())] = base.sha(runner.__file__)
    return actual


if __name__ == '__main__':
    runner.HERE = HERE
    runner.configure = configure
    runner.instrumented_sources = instrumented_sources
    runner.main()
