#!/usr/bin/env python3
"""Bind a disclosed synthetic length case to unmodified real raw-token prompts."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'analysis_eval'))
from build_revision_quota_counterexample import compare, planner, sha


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cases', type=Path, required=True)
    ap.add_argument('--case', default='figure4_scale90')
    ap.add_argument('--workload', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    if args.output.exists():
        raise ValueError('refusing overwrite')
    source = json.loads(args.cases.read_text())
    case = next(c for c in source['candidates'] if c['name'] == args.case)
    data = json.loads(args.workload.read_text())
    prompts = {}
    for request in data['requests']:
        key = request['prompt_hash']
        if key in prompts and prompts[key]['prompt_token_ids'] != request['prompt_token_ids']:
            raise ValueError('ambiguous raw-token prompt identity')
        prompts[key] = request
    if len(prompts) != 32 or len(case['profiles']) != 32:
        raise ValueError('need32 distinct raw-token prompts')
    # Fixed hash order, not tuned using any runtime outcome.
    rows = []
    for profile, prompt in zip(sorted(case['profiles'], key=lambda r: r['id']),
                               [prompts[k] for k in sorted(prompts)]):
        rows.append(dict(profile, prompt_tokens=len(prompt['prompt_token_ids']),
                         prompt_hash=prompt['prompt_hash'], prompt_token_ids=prompt['prompt_token_ids']))
    planner._progress = lambda _: None
    comparison = compare(rows, case['floor'], case['kv_cap_tokens_per_rank'], case['mu'])
    if not all(r['feasible'] for r in comparison['policies'].values()):
        raise ValueError('real prompt lengths make controlled case infeasible')
    requests = [dict(id=row['id'], prompt_hash=row['prompt_hash'], prompt_token_ids=row['prompt_token_ids'],
                     sample_index=idx, max_tokens=length,
                     request_seed=int.from_bytes(hashlib.sha256(f"996:{row['prompt_hash']}:{idx}".encode()).digest()[:4], 'big') % (2**31-1))
                for row in rows for idx, length in enumerate(row['lengths'])]
    payload = dict(status='CONTROLLED_LENGTH_RUNTIME_WORKLOAD_PREPARED', case=args.case,
                   origin=case['origin'], comparison=comparison, requests=requests,
                   input_sha256={str(p): sha(p) for p in (args.cases, args.workload, Path(__file__),
                       ROOT / 'analysis_eval/build_revision_quota_counterexample.py',
                       ROOT / 'tools/build_mode1_length_sorted_e2e_plan.py')},
                   scope='Real raw-token prompts, synthetic per-response lengths, forced ignore_eos termination. Mechanism experiment only; not reward, training convergence, natural workload frequency or measured release.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, sort_keys=True, indent=2) + '\n')
    print(json.dumps({k: comparison[k] for k in ('gain_rank_tokens', 'gain_percent', 'total_generated_tokens')}, sort_keys=True))


if __name__ == '__main__':
    main()
