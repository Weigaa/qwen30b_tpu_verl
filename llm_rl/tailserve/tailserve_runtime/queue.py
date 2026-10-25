"""Durable single-broker queues and rank-owned event logs, using physical JSONL lines."""
import json
import os
from pathlib import Path
import re
import time

ID = re.compile(r'[A-Za-z0-9_.-]{1,96}\Z')


def validate_job(job, config, world_size):
    if not isinstance(job, dict) or set(job) - {'id', 'rank', 'prompt_token_ids', 'max_tokens', 'policy_version'}:
        raise ValueError('unknown request fields')
    if not isinstance(job.get('id'), str) or not ID.fullmatch(job['id']):
        raise ValueError('id must contain 1..96 ASCII letters/digits/_.-')
    rank = job.get('rank')
    if type(rank) is not int or not 0 <= rank < world_size:
        raise ValueError('invalid target rank')
    tokens = job.get('prompt_token_ids')
    if not isinstance(tokens, list) or not 0 < len(tokens) <= config.max_prompt_tokens:
        raise ValueError('prompt length outside the configured bound')
    if any(type(t) is not int or t < 0 for t in tokens):
        raise ValueError('prompt token IDs must be nonnegative integers')
    maximum = job.get('max_tokens', config.max_new_tokens)
    if type(maximum) is not int or not 0 < maximum <= config.max_new_tokens:
        raise ValueError('invalid max_tokens')
    version = job.get('policy_version', 'current')
    if not isinstance(version, str) or not version or len(version) > 256:
        raise ValueError('invalid policy_version')
    return dict(job, max_tokens=maximum, policy_version=version)


def append_record(path, record, timestamp=True):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Each event file has one rank writer. Broker queue writes have a lock.
    with path.open('a', encoding='utf-8') as stream:
        row = dict(time=time.time(), **record) if timestamp else record
        stream.write(json.dumps(row, ensure_ascii=False) + '\n')
        stream.flush()
        os.fsync(stream.fileno())


class Queue:
    def __init__(self, root, rank, world_size, config):
        self.root = Path(root)
        self.rank, self.world_size, self.config = rank, world_size, config
        self.path = self.root / 'queues' / f'rank_{rank}.jsonl'
        self.events = self.root / 'events' / f'rank_{rank}.jsonl'
        self.offset = 0
        self.pending = []
        self.seen = set()
        self.closed = set()
        if self.events.exists():
            latest = {}
            with self.events.open() as stream:
                for line in stream:
                    if line.strip():
                        row = json.loads(line)
                        if row.get('job_id'):
                            self.closed.add(row['job_id'])
                            latest[row['job_id']] = row
            for job_id, row in latest.items():
                if row['event'] == 'admitted':
                    self.emit('cancelled', job_id=job_id, policy_version=row.get('policy_version'),
                              reason='worker_restart_no_kv_reuse', counted_as_completion=False)

    def emit(self, event, **fields):
        append_record(self.events, dict(event=event, rank=self.rank, **fields))

    def poll(self):
        if not self.path.exists():
            return
        with self.path.open('rb') as stream:
            if self.path.stat().st_size < self.offset:
                raise RuntimeError('queue was truncated during an active service run')
            stream.seek(self.offset)
            # Bound per-round host work and avoid a producer starving RL.
            for _ in range(32):
                line = stream.readline(65537)
                if not line:
                    break
                if not line.endswith(b'\n'):
                    if len(line) > 65536:
                        raise ValueError('queue record too large')
                    break
                self.offset = stream.tell()
                try:
                    job = validate_job(json.loads(line), self.config, self.world_size)
                    if job['rank'] != self.rank:
                        raise ValueError('misrouted job')
                except (ValueError, TypeError) as exc:
                    self.emit('invalid_queue_record', error=str(exc))
                    continue
                if job['id'] in self.seen or job['id'] in self.closed:
                    continue
                self.seen.add(job['id'])
                self.pending.append(job)

    def candidate(self, version):
        while self.pending:
            job = self.pending[0]
            if job['policy_version'] in ('current', version):
                return job
            self.pending.pop(0)
            self.emit('rejected_version', job_id=job['id'], requested_version=job['policy_version'], policy_version=version)
        return None

    def take(self, version):
        job = self.candidate(version)
        if job:
            self.pending.pop(0)
        return job
