#!/usr/bin/env python3
"""Validate native-sleep handoffs, not legacy manual-offload log strings.

Pure CPU/offline: no torch/ACL imports, synchronization or NPU allocation.
This proves observed successful handoffs/training, not a free-HBM threshold
or an instantaneous physical-memory peak; external DCMI evidence is separate.
"""
import argparse
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import re


def stamp(line):
    match = re.search(r'20\d\d-\d\d-\d\d \d\d:\d\d:\d\d,\d{3}', line)
    if not match:
        raise ValueError('Native audit is missing its original timestamp')
    return datetime.strptime(match[0], '%Y-%m-%d %H:%M:%S,%f')


def validate(text, steps=5, world_size=16, require_full=False):
    text = re.sub(r'\x1b\[[0-9;]*[mK]', '', text).replace('\r', '\n')
    if not re.search(r'\[run\] end_time=\S+ exit_code=0\s*$', text, re.M):
        raise ValueError('No successful trainer child exit')
    if re.search(r'OutOfMemoryError|NPU out of memory|Memory_Allocation_Failure|ACL_ERROR_RT_MEMORY_ALLOCATION|RayTaskError', text, re.I):
        raise ValueError('Runtime failure/OOM present')
    if 'canonical_offload_enabled=1' in text:
        raise ValueError('Shape-unsafe manual canonical offload used')
    audits = {}; handoffs = {}; ends = []; metrics = {}
    for line in text.splitlines():
        if 'MODE1_STEP_RUNTIME_AUDIT ' in line:
            row = json.JSONDecoder().raw_decode(line.split('MODE1_STEP_RUNTIME_AUDIT ', 1)[1])[0]
            key = row['step'], row['rank']
            if key in audits:
                raise ValueError('Duplicate native rank-step audit')
            audits[key] = row, stamp(line)
        match = re.search(r'\[Rank (\d+) \|.*Invalidated \d+ rollout ACL graph entries after native_sleep_before_release;', line)
        if match:
            handoffs.setdefault(int(match[1]), []).append(stamp(line))
        if '=> rollout_output_time_s:' in line:
            ends.append(stamp(line))
        match = re.search(r'(?:^|\s)step:(\d+)\s+-', line)
        if match:
            key = int(match[1])
            if key in metrics:
                raise ValueError('Duplicate actor update metric')
            metrics[key] = dict(re.findall(r'([A-Za-z0-9_./-]+):([^ ]+)', line))
    expected = {(s, r) for s in range(1, steps+1) for r in range(world_size)}
    if set(audits) != expected or len(ends) != steps or set(metrics) != set(range(1, steps+1)):
        raise ValueError('Incomplete native handoff/actor step coverage')
    verified = []
    for (step, rank), (row, completed_at) in sorted(audits.items()):
        if (not row['complete'] or row['kv_blocks'] <= 0 or
                row['kv_tokens'] != row['kv_blocks']*128 or
                row['worker_kv_blocks'] != row['kv_blocks']):
            raise ValueError('Incomplete inference or worker/scheduler KV mismatch')
        if require_full and (row.get('full_iterations', 0) <= 0 or
                'eager_iterations' not in row or 'global_prefill_iterations' not in row or
                row['eager_iterations'] != row['global_prefill_iterations']):
            raise ValueError('Missing FULL replay or unexpected non-FULL decode')
        matches = [t for t in handoffs.get(rank, []) if completed_at <= t <= ends[step-1]]
        if not matches:
            raise ValueError(f'No native sleep handoff after rank={rank} step={step} inference and before rollout return')
        verified.append(dict(step=step, rank=rank, native_sleep_handoff=min(matches).isoformat()))
    for row in metrics.values():
        gradient = float(row['actor/grad_norm'])
        if not math.isfinite(gradient) or gradient <= 0 or float(row['response/aborted_ratio']) != 0:
            raise ValueError('Actor update/aborted response evidence invalid')
    return dict(status='PASS', backend='native-sleep2-lifecycle1', steps=steps,
                rank_step_handoffs=len(verified), require_full=require_full,
                handoffs=verified, legacy_manual_free_guard_required=False,
                proves_free_hbm_threshold=False,
                note='Observed successful native handoffs and completed actor updates; physical peaks require separate DCMI evidence.')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--log', type=Path, required=True)
    p.add_argument('--expected-steps', type=int, default=5)
    p.add_argument('--world-size', type=int, default=16)
    p.add_argument('--require-full', action='store_true')
    p.add_argument('--output', type=Path)
    args = p.parse_args()
    raw = args.log.read_bytes()
    result = validate(raw.decode(errors='replace'), args.expected_steps, args.world_size, args.require_full)
    result.update(log=str(args.log), log_sha256=hashlib.sha256(raw).hexdigest(),
                  verifier_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    if args.output:
        args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='handoffs'}))


if __name__ == '__main__':
    main()
