#!/usr/bin/env python3
"""Three-stage seed606 experiment using an isolated paper runtime.

Reference/current are genuine updated-policy generation. The other four arms
execute reference-token prefixes on the NPU and report controlled-replay rewards.
"""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys

HERE = Path(__file__).resolve()
PACKAGE = HERE.parents[1]
DEFAULT_ROOT = Path('/workspace/adafloor_revision_results/oracle_centered_n2_seed606_v1')
STAGES = ('reference', 'current', 'oracle', 'under_0p7', 'over_1p3', 'mixed')


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def configure(root, stage):
    base = module(PACKAGE / 'tools/run_paper_aligned_natural_f2_forecast.py', 'paper_base')
    base.PAPER = root / 'paper_runtime'
    base.DRIVER = base.PAPER / 'run_mode1_dynamic_length_aware_adaptive_floor2_natural_tailguard_reuse_epoch0_2epoch.sh'
    base.EXPERIMENT_KEYS = {'actor_rollout_ref.actor.optim.lr': '1e-6',
                           'actor_rollout_ref.rollout.seed': '606', 'trainer.save_freq': '-1'}
    original = base.historical_protocol
    def reference_protocol():
        reference = original()
        if stage == 'reference':
            reference['env']['VLLM_ROLLOUT_SHRINK_AWARE_SHORT_STEP_CAP_ENABLE'] = '0'
        return reference
    base.historical_protocol = reference_protocol
    return base


def instrumented_sources(root, base):
    runtime = root / 'paper_runtime'
    manifest = json.loads((runtime / 'ORACLE_SOURCE_MANIFEST.json').read_text())
    allowed = {'verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py',
               'vllm_ascend/worker/model_runner_v1.py'}
    actual = {}
    for name, before in manifest['files_sha256_before_instrumentation'].items():
        actual[name] = base.sha(runtime / name)
        if name not in allowed and actual[name] != before:
            raise RuntimeError(f'unexpected paper runtime modification: {name}')
    actual['verl/utils/oracle_control.py'] = base.sha(runtime / 'verl/utils/oracle_control.py')
    for path in (runtime / 'patches').rglob('*'):
        if path.is_file() and '__pycache__' not in path.parts:
            name = str(path.relative_to(runtime))
            actual[name] = base.sha(path)
            if actual[name] != base.sha(Path(manifest['source']) / name):
                raise RuntimeError(f'paper support source mismatch: {name}')
    actual[str(HERE)] = base.sha(HERE)
    actual['paper_boundary_runner'] = base.sha(PACKAGE / 'tools/run_paper_aligned_natural_f2_forecast.py')
    return actual


def launch(root, stage, capture, base):
    branch = 'preflight' if capture else 'runs'
    out = root / branch / stage
    if out.exists():
        raise FileExistsError(f'refusing duplicate stage: {out}')
    out.mkdir(parents=True)
    env = base.clean_environment(root, stage, capture)
    env.update(ADAFLOOR_PROTOCOL_TOOL=str(HERE), ADAFLOOR_ORACLE_STAGE=stage,
               ADAFLOOR_ORACLE_EXPERIMENT_ROOT=str(root),
               ADAFLOOR_ORACLE_CAPTURE_DIR=str(root / 'runs' / stage / 'raw'))
    if stage in ('reference', 'current'):
        env['DYNAMIC_INITIAL_BASELINE_DIR'] = str(base.COMMON)
    else:
        env['DYNAMIC_INITIAL_BASELINE_DIR'] = str(root / 'histories' / stage)
        env['ADAFLOOR_ORACLE_REPLAY_DIR'] = str(root / 'reference_bank')
    if stage == 'reference':
        env.update(DYNAMIC_DISABLE_TAIL_GUARD='1', DYNAMIC_SHORT_STEP_CAP_ENABLE='0',
                   DYNAMIC_EXPECT_NO_RESPONSE_CAPS='1')
    command = ['bash', str(base.DRIVER), 'trainer.resume_mode=resume_path',
               f'trainer.resume_from_path={base.CHECKPOINT}',
               'actor_rollout_ref.actor.optim.lr=1e-6', 'actor_rollout_ref.rollout.seed=606']
    with (out / 'driver.log').open('w') as log:
        child = subprocess.Popen(command, cwd=base.PAPER, env=env, stdout=log, stderr=subprocess.STDOUT)
        base.write_json(out / 'process.json', {'pid': child.pid, 'command': command, 'capture_only': capture})
        rc = child.wait()
    base.write_json(out / 'exit.json', {'exit_code': rc})
    if rc != (80 if capture else 0):
        raise RuntimeError(f'{stage} exit={rc}, capture_only={capture}')


def audit(root, stage, base):
    analyzer = module(PACKAGE / 'analysis_eval/summarize_paper_aligned_forecast.py', 'summary')
    run = root / 'runs' / stage / 'natural_f2/epoch_001_mode1_natural'
    result = analyzer.audit_epoch(run, False)
    if not result['measurement_complete']:
        raise RuntimeError(f'incomplete measurement: {result["errors"]}')
    if any(x['metrics'].get('actor/lr') != 1e-6 for x in result['steps']):
        raise RuntimeError('normal actor learning rate not verified')
    log = analyzer.clean(next((run / 'logs').glob('*.txt')).read_text(errors='replace'))
    gradients = re.findall(r'actor/grad_norm:\s*(' + analyzer.NUMBER + ')', log)
    if len(gradients) != 5 or any(float(x) <= 0 for x in gradients):
        raise RuntimeError('five nonzero actor gradient norms not verified')
    raw_hashes, identities = {}, set()
    all_records = []
    for step in range(1, 6):
        files = sorted((root / 'runs' / stage / 'raw' / f'step_{step:03d}').glob('rank_*.json'))
        if len(files) != 16:
            raise RuntimeError(f'step {step} missing raw rank artifacts')
        for path in files:
            data = json.loads(path.read_text())
            if len(data['records']) != 32:
                raise RuntimeError('expected 32 raw responses per rank')
            raw_hashes[str(path)] = base.sha(path)
            for record in data['records']:
                key = (record['prompt_key'], record['sample_index'])
                if key in identities:
                    raise RuntimeError('duplicate response identity')
                identities.add(key)
                if stage == 'reference' and record['sampling_cap'] != 16384:
                    raise RuntimeError('reference response has an adaptive cap')
                if stage not in ('reference', 'current') and not record.get('exact_prefix_match'):
                    raise RuntimeError('reference prefix not verified')
                all_records.append(record)
    result.update(stage=stage, actor_updates_verified=True,
                  execution_kind='online_updated_policy' if stage in ('reference', 'current') else 'controlled_reference_token_replay_with_actor_updates',
                  raw_responses=len(all_records), raw_generated_tokens=sum(x['raw_generated_tokens'] for x in all_records),
                  raw_artifact_sha256=raw_hashes)
    base.write_json(root / 'runs' / stage / 'audit.json', result)
    return result


def main():
    if sys.argv[1:2] == ['boundary']:
        root = Path(os.environ['ADAFLOOR_ORACLE_EXPERIMENT_ROOT'])
        base = configure(root, os.environ['ADAFLOOR_ORACLE_STAGE'])
        base.boundary(sys.argv[3:] if sys.argv[2:3] == ['--'] else sys.argv[2:])
        return
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    ap.add_argument('--stages', nargs='+', choices=STAGES, required=True)
    args = ap.parse_args()
    root = args.root.resolve()
    live = subprocess.run(['pgrep', '-f', '^python3 -m verl.trainer.main_ppo'], capture_output=True)
    if live.returncode == 0:
        raise RuntimeError('a trainer is already running')
    base = configure(root, args.stages[0])
    code = instrumented_sources(root, base)
    start = {'pid': os.getpid(), 'stages': args.stages, 'seed': 606,
             'code_sha256': code, 'common_history_sha256': base.sha(base.COMMON / 'offline_planning_history.json'),
             'checkpoint_marker_sha256': base.sha(base.CHECKPOINT / '.PRESERVE_COMMON_EPOCH0')}
    marker = root / ('supervisor_' + '_'.join(args.stages) + '.json')
    if marker.exists():
        raise FileExistsError(marker)
    base.write_json(marker, start)
    try:
        for stage in args.stages:
            base = configure(root, stage)
            print(f'[oracle-centered] preflight {stage}', flush=True)
            launch(root, stage, True, base)
            print(f'[oracle-centered] NPU {stage}', flush=True)
            launch(root, stage, False, base)
            result = audit(root, stage, base)
            print(f'[oracle-centered] complete {stage} safety={result["status"]} preemptions={result["preemption_count"]}', flush=True)
        if base.sha(base.COMMON / 'offline_planning_history.json') != start['common_history_sha256']:
            raise RuntimeError('public history changed')
        if base.sha(base.CHECKPOINT / '.PRESERVE_COMMON_EPOCH0') != start['checkpoint_marker_sha256']:
            raise RuntimeError('public checkpoint marker changed')
        base.write_json(root / ('complete_' + '_'.join(args.stages) + '.json'),
                        {'measurement_complete': True, 'stages': args.stages})
    except Exception as error:
        base.write_json(root / ('failure_' + '_'.join(args.stages) + '.json'), {'error': str(error)})
        raise


if __name__ == '__main__':
    main()
