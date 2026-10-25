"""Local best-effort API. Queue operations only: this process never loads a model."""
import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import threading
import uuid

from .policy import Config
from .queue import append_record, validate_job


class Store:
    def __init__(self, root, world_size=16, config=None):
        self.root = Path(root)
        self.world_size = world_size
        if world_size <= 0:
            raise ValueError('world_size must be positive')
        self.config = config or Config.from_env()
        self.lock = threading.Lock()
        self.jobs = {}
        self.cursor = 0
        for p in (self.root / 'queues').glob('rank_*.jsonl'):
            with p.open() as f:
                for line in f:
                    if line.strip():
                        job = validate_job(json.loads(line), self.config, self.world_size)
                        if job['id'] in self.jobs:
                            raise ValueError('duplicate request ID in durable queue')
                        self.jobs[job['id']] = job

    def submit(self, request):
        with self.lock:
            if not isinstance(request, dict):
                raise ValueError('JSON object required')
            job = validate_job(dict(request, id=request.get('id', uuid.uuid4().hex),
                                    rank=request.get('rank', self.cursor % self.world_size)),
                               self.config, self.world_size)
            if job['id'] in self.jobs:
                raise ValueError('request ID already exists')
            append_record(self.root / 'queues' / f'rank_{job["rank"]}.jsonl', job, timestamp=False)
            self.jobs[job['id']] = job
            self.cursor += 1
            return dict(id=job['id'], rank=job['rank'], status='queued',
                        requested_policy_version=job['policy_version'], best_effort=True)

    def status(self, job_id):
        with self.lock:
            if job_id not in self.jobs:
                return None
            job = self.jobs[job_id]
        result = dict(id=job_id, rank=job['rank'], status='queued',
                      requested_policy_version=job['policy_version'], best_effort=True)
        path = self.root / 'events' / f'rank_{job["rank"]}.jsonl'
        if path.exists():
            with path.open() as f:
                for line in f:
                    if not line.endswith('\n'):
                        break
                    row = json.loads(line)
                    if row.get('job_id') == job_id:
                        result.update(row, status=row['event'])
        return result


def make_handler(store, auth_token=None):
    class Handler(BaseHTTPRequestHandler):
        def respond(self, status, body):
            data = json.dumps(body, ensure_ascii=False).encode()
            self.send_response(status)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.send_header('Content-Length', str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def authorized(self):
            if auth_token and self.headers.get('Authorization') != 'Bearer ' + auth_token:
                self.respond(401, {'error': 'unauthorized'})
                return False
            return True

        def do_GET(self):
            if not self.authorized():
                return
            if self.path == '/health':
                self.respond(200, {'status': 'queue_ready', 'model_loaded_here': False,
                                   'world_size': store.world_size, 'best_effort': True})
            elif self.path.startswith('/v1/tailserve/requests/'):
                result = store.status(self.path.rsplit('/', 1)[1])
                self.respond(200 if result else 404, result or {'error': 'unknown request'})
            else:
                self.respond(404, {'error': 'not found'})

        def do_POST(self):
            if not self.authorized():
                return
            if self.path != '/v1/tailserve/requests':
                self.respond(404, {'error': 'not found'})
                return
            try:
                size = int(self.headers.get('Content-Length', '0'))
                if not 0 < size <= 65536:
                    raise ValueError('body must be 1..65536 bytes')
                request = json.loads(self.rfile.read(size))
                result = store.submit(request)
                self.respond(202, result)
            except (ValueError, TypeError) as exc:
                self.respond(400, {'error': str(exc)})

        def log_message(self, *_):
            pass
    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spool', required=True)
    parser.add_argument('--world-size', type=int, default=16)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8099)
    parser.add_argument('--auth-token')
    args = parser.parse_args()
    if args.host not in ('127.0.0.1', 'localhost', '::1') and not args.auth_token:
        parser.error('non-local binding requires --auth-token')
    store = Store(args.spool, args.world_size)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(store, args.auth_token))
    try:
        server.serve_forever()
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
