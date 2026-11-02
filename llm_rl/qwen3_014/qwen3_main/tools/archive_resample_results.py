#!/usr/bin/env python3
"""Move legacy result artifacts out of the implementation tree, with hashes."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

REPO = Path(__file__).resolve().parents[4]
SOURCE = REPO / 'llm_rl/qwen3_014/qwen3_main'
OLD_ARCHIVE = REPO / 'llm_rl_largefile/qwen3'
DESTINATION = Path('/data/adafloor_graph_results/qwen3_legacy_resample_archive')
PREFIX = 'resample_result_16k_bs32_n16_'


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def plan():
    records = []
    names = sorted(p.name for p in SOURCE.glob(PREFIX + '*') if p.is_dir())
    for name in names:
        for root in [SOURCE, OLD_ARCHIVE]:
            folder = root / name
            if not folder.exists():
                continue
            if folder.is_symlink():
                raise RuntimeError(f'Unexpected existing archive link: {folder}')
            for path in sorted(folder.rglob('*')):
                if path.is_symlink():
                    raise RuntimeError(f'Inspect symlink before moving: {path}')
                if path.is_dir():
                    continue
                if not path.is_file():
                    raise RuntimeError(f'Unexpected file type: {path}')
                if (path.suffix in {'.py', '.sh', '.yaml', '.yml', '.safetensors', '.pt', '.bin'}
                        or 'checkpoints' in path.relative_to(folder).parts):
                    raise RuntimeError(f'Not a disposable result artifact: {path}')
                destination = DESTINATION / name / path.relative_to(folder)
                sha = digest(path)
                if destination.exists() and digest(destination) != sha:
                    raise RuntimeError(f'Destination collision: {destination}')
                records.append(dict(source=str(path), destination=str(destination),
                                    bytes=path.stat().st_size, sha256=sha))
    by_destination = {}
    for row in records:
        prior = by_destination.setdefault(row['destination'], row['sha256'])
        if prior != row['sha256']:
            raise RuntimeError(f'Different files share destination: {row["destination"]}')
    return names, records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    names, records = plan()
    summary = dict(directories=len(names), files=len(records), bytes=sum(r['bytes'] for r in records),
                   destination=str(DESTINATION), source_code_or_checkpoints_moved=False)
    if not args.apply:
        print(json.dumps(dict(status='DRY_RUN', **summary), indent=2))
        return
    if not names:
        raise RuntimeError('No matching source directories remain')
    receipt = DESTINATION / 'MIGRATION_20261101.json'
    if receipt.exists():
        raise RuntimeError('Migration receipt already exists, refusing to overwrite')
    DESTINATION.mkdir(parents=True, exist_ok=True)
    # Verify every copy before removing any source file.
    for row in records:
        source, destination = Path(row['source']), Path(row['destination'])
        if digest(source) != row['sha256']:
            raise RuntimeError(f'Source changed during migration: {source}')
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            shutil.copy2(source, destination)
        if digest(destination) != row['sha256']:
            raise RuntimeError(f'Copied file hash differs: {destination}')
    for row in records:
        source = Path(row['source'])
        if digest(source) != row['sha256']:
            raise RuntimeError(f'Source changed before cleanup: {source}')
        source.unlink()
    # rmdir, rather than recursive deletion, fails if any unexpected file appeared.
    for name in names:
        for root in [SOURCE, OLD_ARCHIVE]:
            folder = root / name
            if not folder.exists():
                continue
            for child in sorted((p for p in folder.rglob('*') if p.is_dir()),
                                key=lambda p: len(p.parts), reverse=True):
                child.rmdir()
            folder.rmdir()
            if root == OLD_ARCHIVE:
                folder.symlink_to(DESTINATION / name, target_is_directory=True)
    for row in records:
        assert digest(Path(row['destination'])) == row['sha256']
    report = dict(status='COPIED_HASH_VERIFIED_AND_SOURCE_DIRECTORIES_REMOVED',
                  utc=datetime.now(timezone.utc).isoformat(), **summary,
                  preserved_original_archive_manifest=True,
                  old_archive_links_preserve_historical_paths=True,
                  implementation_tree_links_created=False,
                  directory_names=names, files_manifest=records)
    receipt.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'files_manifest'}, indent=2))


if __name__ == '__main__':
    main()
