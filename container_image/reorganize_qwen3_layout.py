#!/usr/bin/env python3
"""One-time version layout migration, preserving pre-merge source archives."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
import tarfile

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO / 'llm_rl'
ARCHIVE = Path('/data/adafloor_code_archives/qwen3_layout_before_20261101')
OLD011 = ['adafloor_gragh', 'adafloor_revision_eager', 'qwen3_shrink_aware',
          'qwen3_shrink_aware_sidecar', 'qwen3_old_machine_v11_bestmode345',
          'qwen3_shrink_aware_cleanup_20260626T182116Z', 'ref_wj_qwen3']
COPIES = ['qwen3', 'qwen3_graph_sidecar_v014', 'qwen3_graph_sidecar_models_v014',
          'qwen3_mlsys_supplement_v014', 'qwen3_mlsys_eager_controls_v014', 'ref_ziyi_qwen3']
PATHS = {name: 'qwen3_011/' + name for name in OLD011}
PATHS.update(qwen3='qwen3_014/qwen3_main',
             qwen3_graph_sidecar_v014='qwen3_014/qwen3_main',
             qwen3_mlsys_supplement_v014='qwen3_014/qwen3_main',
             qwen3_mlsys_eager_controls_v014='qwen3_014/qwen3_main',
             qwen3_graph_sidecar_models_v014='qwen3_014/qwen3_sidecar',
             mlsys_supplement_v014='qwen3_014/experiments')
TEXT = {'.py', '.sh', '.yaml', '.yml', '.toml', '.cfg', '.ini', '.md'}
SOURCE = {'.py', '.sh', '.yaml', '.yml', '.toml', '.cfg', '.ini', '.cpp', '.cc', '.h', '.hpp', '.patch', '.json'}
COMPONENTS = ['vllm', 'vllm_ascend', 'verl', 'megatron', 'mindspeed',
              'tools', 'tests', 'internal', 'patches', 'r1_ascend']
SKIP = {'.git', '__pycache__', '.pytest_cache', '.build', '.cache', '.triton',
        'evidence', 'outputs', 'record', 'record_old011', 'validation_results'}
REVIEWED_SUPERSEDED = {
    'qwen3_mlsys_supplement_v014': {
        'verl/utils/sidecar_restore_handshake.py', 'tools/sidecar_window_gate.py',
        'tools/run_v014_sidecar_training.py', 'tests/test_sidecar_window_gate.py',
        'tests/test_sidecar_training_natural_f2.py', 'internal/run_elastic_sidecar_infer.sh',
        'internal/watch_elastic_shrink_and_run_sidecar.sh'},
    'qwen3_mlsys_eager_controls_v014': {
        'internal/wj_train_grpo_qwen30b_a3b_16die_true_weight_regroup.sh'},
}


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def execution_files(folder):
    for component in COMPONENTS:
        for path in (folder / component).rglob('*'):
            if path.is_file() and path.suffix in SOURCE and not set(path.parts) & SKIP:
                yield path
    for path in folder.iterdir():
        if path.is_file() and path.suffix in SOURCE:
            yield path


def rewrite(path):
    old = path.read_text(errors='strict')
    text = old
    # Only source/config/documentation paths change. Raw experimental evidence
    # and its pinned original path strings/hashes are deliberately not edited.
    for name, new in sorted(PATHS.items(), key=lambda item: -len(item[0])):
        text = re.sub(r'llm_rl/' + re.escape(name) + r'(?![A-Za-z0-9_])',
                      'llm_rl/' + new, text)
    if path.is_relative_to(ROOT / 'qwen3_014'):
        destinations = {
            'qwen3': 'qwen3_main', 'qwen3_graph_sidecar_v014': 'qwen3_main',
            'qwen3_mlsys_supplement_v014': 'qwen3_main',
            'qwen3_mlsys_eager_controls_v014': 'qwen3_main',
            'qwen3_graph_sidecar_models_v014': 'qwen3_sidecar',
            'mlsys_supplement_v014': 'experiments',
            **{name: '../qwen3_011/' + name for name in OLD011},
        }
        for name, new in destinations.items():
            text = re.sub(r'(/\s*[\x22\x27])' + re.escape(name) + r'(?=/|[\x22\x27])',
                          lambda m: m[1] + new, text)
    if text != old:
        path.write_text(text)
        return dict(path=str(path.relative_to(ROOT)), before=hashlib.sha256(old.encode()).hexdigest(),
                    after=sha(path))


def main():
    for folder in [ROOT / 'qwen3_014', ROOT / 'qwen3_011']:
        if folder.exists():
            if any(folder.iterdir()):
                raise RuntimeError('Version layout already contains files, refusing to repeat migration')
            folder.rmdir()
    archives = []
    for name in COPIES:
        path = ARCHIVE / (name + '.tar.gz')
        if not path.is_file():
            raise RuntimeError(f'Missing required rollback archive: {path}')
        # Read the full archive, not only its header, before removing old copies.
        with tarfile.open(path, 'r:gz') as stream:
            members = stream.getmembers()
            assert members and all(Path(m.name).parts[0] == name for m in members)
            for member in members:
                if member.isfile():
                    with stream.extractfile(member) as payload:
                        digest = hashlib.file_digest(payload, 'sha256').hexdigest()
                    original = ROOT / member.name
                    assert original.is_file() and sha(original) == digest, original
        archives.append(dict(original=name, archive=str(path), sha256=sha(path)))
    version = ROOT / 'qwen3_014'
    version.mkdir()
    (ROOT / 'qwen3_011').mkdir()
    main_root = version / 'qwen3_main'
    merged = []
    # Basic graph sidecar extends the primary without replacing its paper or
    # measurement evidence. Supplements/eager copies must have no missed delta.
    base = ROOT / 'qwen3'
    side = ROOT / 'qwen3_graph_sidecar_v014'
    overlay = {p.relative_to(side): p for p in execution_files(side)}
    for folder in ['qwen3_mlsys_supplement_v014', 'qwen3_mlsys_eager_controls_v014']:
        for path in execution_files(ROOT / folder):
            relative = path.relative_to(ROOT / folder)
            digest = sha(path)
            candidates = [base / relative, overlay.get(relative)]
            if not any(p and p.is_file() and sha(p) == digest for p in candidates):
                if str(relative) not in REVIEWED_SUPERSEDED.get(folder, set()):
                    raise RuntimeError(f'Unreviewed supplement/eager delta: {path}')
    for relative, source in overlay.items():
        destination = base / relative
        if str(relative) in {
            'internal/wj_train_grpo_qwen30b_a3b_16die_true_weight_regroup.sh',
            'internal/wj_train_grpo_qwen30b_a3b_16die_true_weight_regroup_old.sh'}:
            # Merge the complete launcher, including sidecar watcher lifecycle,
            # while preserving ONLY the new /data default. Keeping the whole
            # old primary launcher would silently discard the sidecar branch.
            content = source.read_text().replace(
                'OUTPUT_ROOT=${OUTPUT_ROOT:-/workspace/cann-recipes-train/llm_rl/qwen3}',
                'OUTPUT_ROOT=${OUTPUT_ROOT:-/data/adafloor_graph_results/qwen3_runs}')
            merged.append(dict(path=str(relative), source='qwen3_graph_sidecar_v014',
                previous_sha256=sha(destination) if destination.exists() else None,
                merged_sha256=hashlib.sha256(content.encode()).hexdigest(),
                preserved_only_data_output_default=True))
            destination.write_text(content)
            continue
        if destination.exists() and sha(destination) == sha(source):
            continue
        merged.append(dict(path=str(relative), source='qwen3_graph_sidecar_v014',
                           previous_sha256=sha(destination) if destination.exists() else None,
                           merged_sha256=sha(source)))
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    base.rename(main_root)
    (ROOT / 'qwen3_graph_sidecar_models_v014').rename(version / 'qwen3_sidecar')
    (ROOT / 'mlsys_supplement_v014').rename(version / 'experiments')
    for name in OLD011:
        (ROOT / name).rename(ROOT / 'qwen3_011' / name)
    for name in ['qwen3_graph_sidecar_v014', 'qwen3_mlsys_supplement_v014',
                 'qwen3_mlsys_eager_controls_v014', 'ref_ziyi_qwen3']:
        shutil.rmtree(ROOT / name)
    edited = []
    for folder in [ROOT / 'qwen3_014', ROOT / 'qwen3_011', ROOT / 'tailserve']:
        for path in folder.rglob('*'):
            if (not path.is_file() or path.is_symlink() or path.suffix not in TEXT
                    or set(path.relative_to(folder).parts) & SKIP):
                continue
            try:
                record = rewrite(path)
            except UnicodeDecodeError:
                continue
            if record:
                edited.append(record)
    # Keep no old top-level aliases. Record their new locations explicitly.
    report = dict(utc=datetime.now(timezone.utc).isoformat(), rollback_archives=archives,
                  old_to_new_paths=PATHS, main_merge_sources=['qwen3', 'qwen3_graph_sidecar_v014',
                  'qwen3_mlsys_supplement_v014', 'qwen3_mlsys_eager_controls_v014'],
                  merged_execution_files=merged, relocated_text_files=edited,
                  reviewed_superseded_files={k: sorted(v) for k, v in REVIEWED_SUPERSEDED.items()},
                  historical_evidence_and_public_checkpoints_modified=False,
                  npu_experiments_started=False,
                  reference_v014_archived_only='ref_ziyi_qwen3',
                  independent_tailserve_preserved=True)
    receipt = version / 'LAYOUT_MIGRATION_20261101.json'
    receipt.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(dict(status='LAYOUT_CREATED', merged_files=len(merged),
                         rewritten_files=len(edited), receipt=str(receipt)), indent=2))


if __name__ == '__main__':
    main()
