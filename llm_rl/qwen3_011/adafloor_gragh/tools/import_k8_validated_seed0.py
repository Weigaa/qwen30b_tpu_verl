#!/usr/bin/env python3
"""Import the two validated k8 v2 arms, preserving their original provenance.

Only Planned admission changes in this continuation. No Planned outputs are
eligible for import. This is intentionally specific to this interrupted suite.
"""
import hashlib
import argparse
import json
from pathlib import Path
import shutil
import sys

SOURCE = Path('/data/adafloor_graph_results/qwen3_paper_graph_core3_k8_prompt_kv_v2')
EXPECTED = {
    'code_sha256.txt': 'c8d4ca0f5a71d5d7c7f7b8e69a17f5a75c3a040a6973488623551c300bdf29cd',
    'protocol.env': '1f9e64b7dfe5e50d61fecc716be0c96b9a58c837a0687484f50d392f3f9d8766',
    'run_manifest.tsv': '73bc6ebd5d9f12b35423cde7e8a45cb113c638e00c7e7558ef36e8b34ee5e4e4',
}
ALLOWED_CHANGES = {
    'run_qwen3_paper_graph_main_from_common_epoch0.sh',
    'run_mode1_local_length_sorted_e2e_adaptive_floor4.sh',
    'run_mode1_local_length_sorted_e2e_adaptive_floor2.sh',
    'build_mode1_length_sorted_e2e_plan.py',
}


def digest(p):
    with p.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('destination')
    parser.add_argument('--vanilla-only', action='store_true',
                        help='Import only unchanged Vanilla after changing AdaFloor admission.')
    args = parser.parse_args()
    dst = Path(args.destination).resolve()
    if dst == SOURCE.resolve() or not (dst/'protocol.env').is_file():
        raise ValueError('create a distinct destination with the runner dry-run first')
    if (dst/'IMPORTED_ARMS.json').exists():
        raise ValueError('import already exists')
    for name, sha in EXPECTED.items():
        if digest(SOURCE/name) != sha:
            raise ValueError(f'source provenance changed: {name}')
    for line in (SOURCE/'code_sha256.txt').read_text().splitlines():
        old, name = line.split(maxsplit=1)
        path = Path(name)
        if path.name not in ALLOWED_CHANGES and digest(path) != old:
            raise ValueError(f'unrelated execution code changed: {path}')
    src_protocol = dict(l.split('=',1) for l in (SOURCE/'protocol.env').read_text().splitlines())
    dst_protocol = dict(l.split('=',1) for l in (dst/'protocol.env').read_text().splitlines())
    for k, v in src_protocol.items():
        if args.vanilla_only and k == 'kv_admission_safety_factor':
            if (v,dst_protocol.get(k)) != ('1.16','1.0'):
                raise ValueError('unexpected admission-only protocol change')
            continue
        if k != 'created_at_utc' and dst_protocol.get(k) != v:
            raise ValueError(f'imported arm protocol differs: {k}')
    if not args.vanilla_only and dst_protocol.get('other_kv_peak_model') != 'synchronized':
        raise ValueError('import requires unchanged Vanilla/Natural planning semantics')
    imported = []
    manifest = dst/'run_manifest.tsv'
    if len(manifest.read_text().splitlines()) != 1:
        raise ValueError('destination must have an empty run manifest')
    policies = ('vanilla',) if args.vanilla_only else ('vanilla','natural_f2')
    for index, policy in enumerate(policies, 1):
        name = f'qwen3_graph_{policy}_seed0'
        src = SOURCE/'seed_0'/policy/name
        out = dst/'seed_0'/policy/name
        expected_row = f'{index}\t0\t{policy}\tcomplete\t{src}'
        if expected_row not in (SOURCE/'run_manifest.tsv').read_text().splitlines():
            raise ValueError('arm is not complete in source manifest')
        if not (src/'CHECKPOINTS_REMOVED_AFTER_VALIDATION.txt').is_file():
            raise ValueError('missing source cleanup validation')
        logs = sorted(src.glob('epoch_*/logs/*.txt'))
        if len(logs) != 2:
            raise ValueError('expected two completed epochs')
        for log in logs:
            text = log.read_text(errors='replace')
            if ('After trainer.fit' not in text or text.count('rollout_output_time_s:') != 5
                    or 'Preempting request' in text
                    or text.count('response/aborted_ratio:0.0') != 5):
                raise ValueError(f'source runtime validation failed: {log}')
        if out.exists() or any(src.rglob('global_step_*')):
            raise ValueError('destination occupied or source checkpoint not cleaned')
        files = {str(p.relative_to(src)): digest(p) for p in src.rglob('*') if p.is_file()}
        shutil.copytree(src, out)
        for rel, sha in files.items():
            if digest(out/rel) != sha:
                raise ValueError(f'import hash mismatch: {rel}')
        imported.append(dict(policy=policy, seed=0, source=str(src), destination=str(out), files_sha256=files))
        with manifest.open('a') as f:
            f.write(f'{index}\t0\t{policy}\tcomplete\t{out}\n')
    archive = dst/'imported_provenance'
    archive.mkdir()
    for name in EXPECTED:
        shutil.copy2(SOURCE/name, archive/name)
    (dst/'IMPORTED_ARMS.json').write_text(json.dumps(dict(
        source_provenance=EXPECTED, imported_arms=imported,
        reason=('Only unchanged Vanilla is imported; all Natural/Planned arms rerun under prompt-adaptive admission.'
                if args.vanilla_only else
                'Only Planned uses independent-prompt peak admission; imported Vanilla/Natural retain synchronized semantics.'),
    ), indent=2)+'\n')
    print(f'Imported validated seed0 {policies} with byte-for-byte artifact checks')


if __name__ == '__main__':
    main()
