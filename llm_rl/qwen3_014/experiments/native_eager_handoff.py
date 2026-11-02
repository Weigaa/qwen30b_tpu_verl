#!/usr/bin/env python3
"""Audit real native-sleep handoffs in eager logs, without graph-only events.

Offline only. Native allocator release counters are reported evidence, not an
independent device-memory peak measurement. The immutable training snapshot is
not changed. Its graph verifier remains authoritative for FULL graph runs.
"""
import argparse
from collections import defaultdict
from datetime import datetime, timedelta
import hashlib
import json
import math
from pathlib import Path
import re

ANSI = re.compile(r'\x1b\[[0-9;?]*[ -/]*[@-~]')
PID = re.compile(r'\(WorkerDict pid=(\d+)\)')
TIME = re.compile(r'(?:(20\d\d)-)?(\d\d)-(\d\d) (\d\d):(\d\d):(\d\d)(?:,(\d{3}))?')
SOURCE = Path('/workspace/cann-recipes-train/llm_rl/qwen3_014/qwen3_main')


def blocking_release_sources():
    """Resolve second-resolution clock overlap using the actual blocking RPC.

    This is not a timing tolerance: generate_sequences cannot return until
    trainer_mode has awaited native release. Preserve exact source hashes.
    """
    paths = [SOURCE/'verl/workers/megatron_workers.py',
        SOURCE/'verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py',
        SOURCE/'verl/trainer/ppo/ray_trainer.py']
    snippets = [
        ['loop.run_until_complete(self.trainer_mode())', 'await self.rollout.release()'],
        ['self.inference_engine.sleep(level=self.sleep_level)'],
        ['gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)'],
    ]
    result = {}
    for path, required in zip(paths, snippets):
        raw = path.read_bytes()
        if any(s not in raw.decode() for s in required):
            raise ValueError('Blocking native release source contract changed')
        result[str(path)] = hashlib.sha256(raw).hexdigest()
    return result


def stamp(line, year):
    m = TIME.search(line)
    if not m:
        raise ValueError('Missing original native-sleep timestamp')
    y, month, day, hour, minute, second, ms = m.groups()
    lo = datetime(int(y or year), int(month), int(day), int(hour), int(minute), int(second), int(ms or 0)*1000)
    return lo, lo + timedelta(milliseconds=1) if ms else lo + timedelta(seconds=1)


def validate(text, steps=5, world_size=16):
    blocking_sources = blocking_release_sources()
    text = ANSI.sub('', text).replace('\r', '\n')
    if not re.search(r'\[run\] end_time=\S+ exit_code=0\s*$', text, re.M):
        raise ValueError('No successful trainer child exit')
    if re.search(r'OutOfMemoryError|NPU out of memory|Memory_Allocation_Failure|ACL_ERROR_RT_MEMORY_ALLOCATION|RayTaskError', text, re.I):
        raise ValueError('Runtime failure/OOM present')
    if 'canonical_offload_enabled=1' in text:
        raise ValueError('Manual canonical offload used')
    year = int(re.search(r'(20\d\d)-\d\d-\d\d', text)[1])
    audits = {}; pids = {}; ends = []; metrics = {}
    sleeps = defaultdict(list); releases = defaultdict(list)
    for line in text.splitlines():
        for chunk in re.split(r'(?=\([^()\n]*pid=\d+\))', line):
            m = PID.match(chunk); pid = int(m[1]) if m else None
            if 'MODE1_STEP_RUNTIME_AUDIT ' in chunk:
                row = json.JSONDecoder().raw_decode(chunk.split('MODE1_STEP_RUNTIME_AUDIT ', 1)[1])[0]
                key = row['step'], row['rank']
                if key in audits or pid is None or pids.setdefault(pid, row['rank']) != row['rank']:
                    raise ValueError('Duplicate rank-step audit or changed worker/rank')
                audits[key] = row, stamp(chunk, year), pid
            if pid is not None and 'seconds to fall asleep.' in chunk:
                sleeps[pid].append(stamp(chunk, year))
            m = re.search(r'Sleep mode freed ([0-9.]+) GiB memory, ([0-9.]+) GiB memory is still in use', chunk)
            if m and pid is not None:
                releases[pid].append((stamp(chunk, year), float(m[1]), float(m[2])))
            if '=> rollout_output_time_s:' in chunk:
                ends.append(stamp(chunk, year))
            m = re.search(r'(?:^|\s)step:(\d+)\s+-', chunk)
            if m:
                step = int(m[1])
                if step in metrics:
                    raise ValueError('Duplicate actor update metric')
                metrics[step] = dict(re.findall(r'([A-Za-z0-9_./-]+):([^ ]+)', chunk))
    expected = {(s, r) for s in range(1, steps+1) for r in range(world_size)}
    if (set(audits) != expected or len(pids) != world_size or len(set(pids.values())) != world_size
            or len(ends) != steps or set(metrics) != set(range(1, steps+1))):
        raise ValueError('Incomplete native handoff/actor step coverage')
    ends.sort(); verified = []; used = set()
    for (step, rank), (row, completed, pid) in sorted(audits.items()):
        if (not row['complete'] or row['kv_blocks'] <= 0 or row['kv_tokens'] != row['kv_blocks']*128
                or row['worker_kv_blocks'] != row['kv_blocks'] or row.get('eager_iterations', 0) <= 0
                or row.get('full_iterations', 0) != 0):
            raise ValueError('Not complete eager inference with consistent KV')
        # Native logs have only second precision. A sleep timestamp sharing
        # the parent's return second is causally before return because the
        # actual worker RPC awaits it (sources recorded above). Never accept
        # an event whose earliest possible time is after the return.
        matches = [t for t in sleeps[pid] if completed[1] <= t[0] and t[0] <= ends[step-1][0]]
        if len(matches) != 1 or (pid, matches[0]) in used:
            raise ValueError(f'No unique post-inference native sleep for rank={rank} step={step}')
        t = matches[0]; used.add((pid, t))
        freed = [v for v in releases[pid] if completed[1] <= v[0][0] and v[0][1] <= t[1]]
        if len(freed) != 1 or not math.isfinite(freed[0][1]) or freed[0][1] <= 0:
            raise ValueError(f'No unique positive native memory release for rank={rank} step={step}')
        verified.append(dict(step=step, rank=rank, worker_pid=pid,
            native_sleep_completion_bounds=[v.isoformat() for v in t],
            native_sleep_completion_upper_bound_after_blocking_return=min(t[1], ends[step-1][1]).isoformat(),
            native_reported_freed_GiB=freed[0][1], native_reported_remaining_GiB=freed[0][2]))
    for row in metrics.values():
        gradient = float(row['actor/grad_norm'])
        if (not math.isfinite(gradient) or gradient <= 0 or float(row['actor/lr']) != 1e-6
                or float(row['response/aborted_ratio']) != 0):
            raise ValueError('Invalid actor update or aborted response evidence')
    return dict(status='PASS', mode='eager', backend='native-sleep2-lifecycle1',
        steps=steps, rank_step_handoffs=len(verified), handoffs=verified,
        graph_invalidation_event_required=False, proves_device_memory_peak=False,
        blocking_release_source_sha256=blocking_sources,
        evidence='Actual per-worker native sleep completion and positive allocator release counters; no synthetic graph events.')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--log', type=Path, required=True)
    p.add_argument('--expected-steps', type=int, default=5)
    p.add_argument('--world-size', type=int, default=16)
    p.add_argument('--output', type=Path)
    args = p.parse_args(); raw = args.log.read_bytes()
    result = validate(raw.decode(errors='replace'), args.expected_steps, args.world_size)
    result.update(log=str(args.log), log_sha256=hashlib.sha256(raw).hexdigest(),
        verifier_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    if args.output:
        args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k != 'handoffs'}))


if __name__ == '__main__':
    main()
