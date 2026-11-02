#!/usr/bin/env python3
"""Build exact reference-token banks from the completed no-TG N2 run."""
import argparse
import copy
import errno
import hashlib
import json
import os
from pathlib import Path
import subprocess
import shutil
import sys


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, required=True)
    args = ap.parse_args()
    root = args.root.resolve()
    reference = root / 'runs/reference'
    audit = json.loads((reference / 'audit.json').read_text())
    if not audit['measurement_complete'] or not audit['actor_updates_verified']:
        raise RuntimeError('reference measurement incomplete')
    epoch = reference / 'natural_f2/epoch_001_mode1_natural'
    bank = root / 'reference_bank'
    if bank.exists() or (root / 'histories').exists():
        raise FileExistsError('reference bank or histories already exist')
    common = Path('/data/adafloor_shared_state/common_epoch0_probe_gpu09_kv380800_permanent/epoch_000_mode0_probe')
    current = json.loads((common / 'offline_planning_history.json').read_text())
    observed = {}
    hashes = {str(reference / 'audit.json'): sha(reference / 'audit.json'),
              str(common / 'offline_planning_history.json'): sha(common / 'offline_planning_history.json'),
              str(Path(__file__).resolve()): sha(__file__)}
    for step in range(1, 6):
        raw = {}
        for path in sorted((reference / 'raw' / f'step_{step:03d}').glob('rank_*.json')):
            data = json.loads(path.read_text())
            hashes[str(path)] = sha(path)
            for row in data['records']:
                if row['sampling_cap'] != 16384:
                    raise RuntimeError('reference has an adaptive response cap')
                key = (data['rank'], row['request_id'])
                if key in raw:
                    raise ValueError('duplicate rank/request identity')
                raw[key] = row
        path = epoch / 'rollout_data' / f'{step}.jsonl'
        hashes[str(path)] = sha(path)
        n = 0
        with path.open() as f:
            for line in f:
                row = json.loads(line)
                record = copy.deepcopy(raw.pop((int(row['rollout_rank']), str(row['request_id']))))
                tokens = record['token_ids']
                if row['responses'][:len(tokens)] != tokens:
                    raise RuntimeError('raw capture differs from trainer output prefix')
                key = record['prompt_key']
                group = observed.setdefault(key, {'prompt_key': key, 'input': row['input'],
                                                  'prompt_token_ids': record['prompt_token_ids'],
                                                  'ground_truth': row['gts'], 'responses': {}})
                if group['input'] != row['input'] or record['sample_index'] in group['responses']:
                    raise RuntimeError('prompt/sample identity mismatch')
                record.update(reference_step=step, reference_reward=float(row['score']))
                group['responses'][record['sample_index']] = record
                n += 1
        if raw or n != 512:
            raise RuntimeError(f'raw/trainer records not one-to-one at step {step}')
    by_input = {row['input']: row for row in observed.values()}
    if len(observed) != 160 or set(by_input) != {r['input'] for r in current['records']}:
        raise RuntimeError('reference prompt multiset differs from public epoch0')
    oracle = copy.deepcopy(current)
    oracle['length_definition'] = 'raw_generated_tokens'
    oracle['global_response_limit'] = 16384
    oracle['reference_scope'] = 'Exact uncapped-within-global-limit source for controlled token replay.'
    for row in oracle['records']:
        group = by_input[row['input']]
        if set(group['responses']) != set(range(16)):
            raise RuntimeError('missing response sample index')
        row['lengths'] = [len(group['responses'][i]['token_ids']) for i in range(16)]
    (bank / 'prompts').mkdir(parents=True)
    for key, group in sorted(observed.items()):
        group['responses'] = [group['responses'][i] for i in range(16)]
        (bank / 'prompts' / f'{key}.json').write_text(json.dumps(group, separators=(',', ':'))+'\n')
    oracle_path = bank / 'offline_planning_history.json'
    oracle_path.write_text(json.dumps(oracle, sort_keys=True, separators=(',', ':'))+'\n')
    manifest = {'status': 'EXACT_REFERENCE_BANK_VERIFIED', 'seed': 606, 'prompts': 160,
                'responses': 2560, 'input_sha256': hashes, 'oracle_history_sha256': sha(oracle_path),
                'prompt_artifact_sha256': {p.name: sha(p) for p in sorted((bank/'prompts').glob('*.json'))},
                'scope': 'Controlled replay of reference token prefixes. Online main arm has independent actor updates.'}
    (bank / 'manifest.json').write_text(json.dumps(manifest, sort_keys=True, indent=2)+'\n')
    script = Path(__file__).with_name('prepare_oracle_centered_histories.py')
    subprocess.run([sys.executable, str(script), '--current-history', str(common/'offline_planning_history.json'),
                    '--oracle-history', str(oracle_path), '--oracle-scope', 'controlled_token_replay',
                    '--seed', '606', '--output-root', str(root/'histories')], check=True)
    for condition in ('current', 'oracle', 'under_0p7', 'over_1p3', 'mixed'):
        source = common if condition == 'current' else epoch
        for folder in ('rollout_data', 'rollout_length'):
            dest = root / 'histories' / condition / folder
            dest.mkdir()
            for path in sorted((source/folder).iterdir()):
                if path.is_file():
                    try:
                        os.link(path, dest/path.name)
                    except OSError as error:
                        if error.errno != errno.EXDEV:
                            raise
                        # Public epoch0 can be on /data while experiment
                        # outputs are on /workspace. Preserve regular files
                        # for the original launcher's find-based checks.
                        shutil.copyfile(path, dest/path.name)
    print(json.dumps({'status': manifest['status'], 'responses': 2560}))


if __name__ == '__main__':
    main()
