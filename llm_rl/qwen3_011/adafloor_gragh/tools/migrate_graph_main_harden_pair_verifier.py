#!/usr/bin/env python3
"""Harden the post-run graph/eager verifier without changing execution code."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path


OLD_CODE_SHA = "d4d1cccbb4b7a179c902941a15da1b7fbb9b2db0ea3b34e9f69b5c93db0d1f64"
NEW_CODE_SHA = "ff165261dfe7be281df65f501aa5307910cc3546af9032cca3a88afb5332e8ce"
PROTOCOL_SHA = "a8e9b59771c65c04153fc750e4724f002b8c29f907671b9f111cb6517e9d8f9e"
RUN_MANIFEST_SHA = "aaed67a81e2c4b78f4f7bd607c8d59e0eebac189f3f37de8713e47d01775dad5"
COMPARATOR = "/workspace/cann-recipes-train/llm_rl/qwen3_011/adafloor_gragh/analysis_eval/compare_qwen3_paper_graph_vs_matched_eager.py"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def manifest(path: Path) -> dict[str, str]:
    rows: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        digest, source = line.split(maxsplit=1)
        if source in rows:
            raise RuntimeError(f"duplicate source path: {source}")
        rows[source] = digest
    return rows


def atomic_write(path: Path, data: bytes) -> None:
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        stream.write(data)
        temporary = Path(stream.name)
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--candidate-code", type=Path, required=True)
    parser.add_argument("--candidate-protocol", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    old_code = root / "code_sha256.txt"
    new_code = args.candidate_code.resolve()
    old_protocol = root / "protocol.env"
    new_protocol = args.candidate_protocol.resolve()
    run_manifest = root / "run_manifest.tsv"
    for path, expected in (
        (old_code, OLD_CODE_SHA),
        (new_code, NEW_CODE_SHA),
        (old_protocol, PROTOCOL_SHA),
        (new_protocol, PROTOCOL_SHA),
        (run_manifest, RUN_MANIFEST_SHA),
    ):
        actual = sha256(path)
        if actual != expected:
            raise RuntimeError(f"hash mismatch for {path}: {actual} != {expected}")
    if old_protocol.read_bytes() != new_protocol.read_bytes():
        raise RuntimeError("execution protocol changed")
    rows = run_manifest.read_text(encoding="utf-8").splitlines()
    expected_prefix = ("0\tvanilla", "0\tlengthsort", "0\tlengthsort_guard")
    if len(rows) != 4:
        raise RuntimeError("completed run prefix changed")
    for row, expected in zip(rows[1:], expected_prefix, strict=True):
        fields = row.split("\t")
        if "\t".join(fields[1:3]) != expected or fields[3] != "complete":
            raise RuntimeError(f"unexpected run row: {row}")
        arm_root = Path(fields[4])
        if not (arm_root / "CHECKPOINTS_REMOVED_AFTER_VALIDATION.txt").is_file():
            raise RuntimeError(f"cleanup marker missing: {arm_root}")

    old = manifest(old_code)
    new = manifest(new_code)
    changed = {path for path in old.keys() & new.keys() if old[path] != new[path]}
    if changed != {COMPARATOR} or set(old) != set(new):
        raise RuntimeError(f"unexpected manifest delta: changed={changed}")

    archive = root / "code_sha256.pre_pair_verifier_hardening.txt"
    audit_path = root / "PAIR_VERIFIER_HARDENING_MIGRATION.json"
    if archive.exists() and archive.read_bytes() != old_code.read_bytes():
        raise RuntimeError("existing archive is inconsistent")
    if not archive.exists():
        atomic_write(archive, old_code.read_bytes())
    audit = {
        "schema_version": 1,
        "status": "COMMITTED",
        "migration_scope": "post_run_verifier_only",
        "runtime_worker_code_changed": False,
        "orchestrator_code_changed": False,
        "execution_protocol_changed": False,
        "completed_run_artifacts_changed": False,
        "old_code_manifest_sha256": OLD_CODE_SHA,
        "new_code_manifest_sha256": NEW_CODE_SHA,
        "protocol_sha256": PROTOCOL_SHA,
        "run_manifest_sha256": RUN_MANIFEST_SHA,
        "changed_code_paths": sorted(changed),
        "new_gates": [
            "graph_and_eager_shared_source_hash_identity",
            "summary_hash_binding_to_pass_marker",
        ],
        "created_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "migration_tool_sha256": sha256(Path(__file__).resolve()),
    }
    atomic_write(audit_path, (json.dumps(audit, indent=2, sort_keys=True) + "\n").encode())
    atomic_write(old_code, new_code.read_bytes())
    print(json.dumps(audit, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
