#!/usr/bin/env python3
"""Copy the recorded paper source list plus native/runtime support assets."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--destination', type=Path, required=True)
    args = ap.parse_args()
    package = Path(__file__).resolve().parents[1]
    source = package.parent / 'qwen3_shrink_aware'
    snapshot = json.loads((package / 'PAPER_SOURCE_SNAPSHOT.json').read_text())
    if args.destination.exists():
        raise FileExistsError(args.destination)
    names = {r['relative_path'] for r in snapshot['copied_regular_files']}
    for support in ('r1_ascend', 'patches'):
        for path in (source / support).rglob('*'):
            if path.is_file() and '__pycache__' not in path.parts:
                names.add(str(path.relative_to(source)))
    for folder in ('vllm_ascend', 'vllm'):
        for path in (source / folder).glob('*.so'):
            names.add(str(path.relative_to(source)))
    records = {}
    for name in sorted(names):
        src = source / name
        if not src.is_file():
            raise FileNotFoundError(src)
        dst = args.destination / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        records[name] = sha(src)
        if sha(dst) != records[name]:
            raise RuntimeError(f'copy mismatch: {name}')
    manifest = {'source': str(source), 'destination': str(args.destination.resolve()),
                'files_sha256_before_instrumentation': records,
                'scope': 'Code-only copy of current paper tree, with native libraries and r1_ascend support. No checkpoint copy.'}
    (args.destination / 'ORACLE_SOURCE_MANIFEST.json').write_text(json.dumps(manifest, sort_keys=True, indent=2)+'\n')
    print(json.dumps({'copied_files': len(records), 'destination': str(args.destination)}))


if __name__ == '__main__':
    main()
