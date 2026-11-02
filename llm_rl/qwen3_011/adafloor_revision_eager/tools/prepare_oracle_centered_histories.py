#!/usr/bin/env python3
"""Prepare five forecast histories without changing the source observations.

The oracle's scope is explicit. A reference trajectory is not asserted to be
the exact future of an independently updating actor. No model is imported.
"""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path


CONDITIONS = ('current', 'oracle', 'under_0p7', 'over_1p3', 'mixed')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate(history, *, realized=False):
    if history.get('schema_version') != 1 or history.get('responses_per_prompt') != 16:
        raise ValueError('expected schema 1, 16 responses per prompt')
    records = history.get('records', [])
    if len(records) != 160:
        raise ValueError('expected the full 160-prompt workload')
    identities = [row['input'] for row in records]
    if any(not isinstance(x, str) for x in identities) or len(set(identities)) != 160:
        raise ValueError('prompt identities must be unique strings')
    for row in records:
        lengths = row['lengths']
        if len(lengths) != 16 or any(not math.isfinite(x) or x < 1 for x in lengths):
            raise ValueError('invalid response length vector')
        if realized and any(int(x) != x or x > 16384 for x in lengths):
            raise ValueError('reference must contain integer raw lengths within the global limit')
    if realized and history.get('length_definition') != 'raw_generated_tokens':
        raise ValueError('reference needs audited raw generated lengths, not response-mask counts')
    return set(identities)


def derive(current, oracle, *, seed, oracle_scope):
    if oracle_scope not in ('reference_trajectory', 'per_step_lookahead', 'controlled_token_replay'):
        raise ValueError('oracle scope must be selected explicitly')
    if validate(current) != validate(oracle, realized=True):
        raise ValueError('current and oracle prompt sets differ')
    result = {}
    for condition in CONDITIONS:
        payload = copy.deepcopy(current if condition == 'current' else oracle)
        for row in payload['records']:
            if condition in ('current', 'oracle'):
                continue
            factor = {'under_0p7': 0.7, 'over_1p3': 1.3}.get(condition)
            if factor is None:
                digest = hashlib.sha256(f"{seed}:{row['input']}".encode()).digest()
                factor = 0.7 if digest[0] % 2 == 0 else 1.3
            row['lengths'] = [max(1, math.ceil(x * factor)) for x in row['lengths']]
        payload['forecast_experiment'] = {
            'condition': condition, 'seed': seed, 'oracle_scope': oracle_scope,
            'base': 'historical_predictor' if condition == 'current' else 'realized_oracle',
            'exact_future_of_measured_arm_verified': False,
        }
        if condition not in ('current', 'oracle'):
            payload['length_definition'] = 'forecast_from_scaled_raw_reference'
        result[condition] = payload
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--current-history', type=Path, required=True)
    parser.add_argument('--oracle-history', type=Path, required=True)
    parser.add_argument('--oracle-scope', choices=('reference_trajectory', 'per_step_lookahead', 'controlled_token_replay'), required=True)
    parser.add_argument('--seed', type=int, default=606)
    parser.add_argument('--output-root', type=Path, required=True)
    args = parser.parse_args()
    if args.output_root.exists():
        raise FileExistsError('existing evidence is never overwritten')
    hashes = {str(p.resolve()): sha(p) for p in (args.current_history, args.oracle_history)}
    result = derive(json.loads(args.current_history.read_text()),
                    json.loads(args.oracle_history.read_text()),
                    seed=args.seed, oracle_scope=args.oracle_scope)
    args.output_root.mkdir(parents=True)
    outputs = {}
    for condition, payload in result.items():
        folder = args.output_root / condition
        folder.mkdir()
        output = folder / 'offline_planning_history.json'
        output.write_text(json.dumps(payload, sort_keys=True, separators=(',', ':')) + '\n')
        outputs[condition] = {'path': str(output.resolve()), 'sha256': sha(output)}
    for path, digest in hashes.items():
        if sha(path) != digest:
            raise RuntimeError('source observation changed during preparation')
    manifest = {'seed': args.seed, 'oracle_scope': args.oracle_scope,
                'input_sha256': hashes, 'code_sha256': sha(__file__), 'outputs': outputs,
                'status': 'HISTORIES_PREPARED_NOT_EXECUTED'}
    (args.output_root / 'manifest.json').write_text(json.dumps(manifest, sort_keys=True, indent=2) + '\n')
    print(json.dumps({'status': manifest['status'], 'conditions': list(result)}))


if __name__ == '__main__':
    main()
