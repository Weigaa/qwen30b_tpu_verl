#!/usr/bin/env python3
"""Add the matched-eager comparison control plane to a partial graph matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path


EXPECTED_OLD_CODE_SHA = "f99f226a850dc5d46aa3dcedfd9dae6febafe590b14beec753c60dbdc60033c0"
EXPECTED_NEW_CODE_SHA = "d4d1cccbb4b7a179c902941a15da1b7fbb9b2db0ea3b34e9f69b5c93db0d1f64"
EXPECTED_PROTOCOL_SHA = "a8e9b59771c65c04153fc750e4724f002b8c29f907671b9f111cb6517e9d8f9e"
EXPECTED_RUN_MANIFEST_SHA = "aaed67a81e2c4b78f4f7bd607c8d59e0eebac189f3f37de8713e47d01775dad5"
MAIN = "/workspace/cann-recipes-train/llm_rl/qwen3_011/adafloor_gragh/run_qwen3_paper_graph_main_from_common_epoch0.sh"
COMPARATOR = "/workspace/cann-recipes-train/llm_rl/qwen3_011/adafloor_gragh/analysis_eval/compare_qwen3_paper_graph_vs_matched_eager.py"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_code_manifest(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        digest, filename = line.split(maxsplit=1)
        if filename in result:
            raise RuntimeError(f"duplicate code path: {filename}")
        result[filename] = digest
    return result


def atomic_write(path: Path, content: bytes) -> None:
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        stream.write(content)
        temporary = Path(stream.name)
    os.replace(temporary, path)


def verify_completed_prefix(root: Path) -> None:
    manifest = root / "run_manifest.tsv"
    if sha256(manifest) != EXPECTED_RUN_MANIFEST_SHA:
        raise RuntimeError("run manifest changed from the verified three-arm prefix")
    rows = manifest.read_text(encoding="utf-8").splitlines()
    expected = (
        ("1", "0", "vanilla"),
        ("2", "0", "lengthsort"),
        ("3", "0", "lengthsort_guard"),
    )
    if len(rows) != len(expected) + 1:
        raise RuntimeError("run manifest does not contain exactly three completed arms")
    for row, (index, seed, policy) in zip(rows[1:], expected, strict=True):
        fields = row.split("\t")
        if fields[:4] != [index, seed, policy, "complete"] or len(fields) != 5:
            raise RuntimeError(f"unexpected completed-arm row: {row}")
        arm_root = Path(fields[4])
        marker = arm_root / "CHECKPOINTS_REMOVED_AFTER_VALIDATION.txt"
        if not marker.is_file() or marker.is_symlink():
            raise RuntimeError(f"cleanup marker missing for {policy}: {arm_root}")
        forbidden = [
            path for path in arm_root.rglob("*")
            if path.is_dir()
            and (path.name == "checkpoints" or path.name.startswith("global_step_"))
        ]
        if forbidden:
            raise RuntimeError(f"validated arm retains checkpoints: {forbidden}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--candidate-code", required=True, type=Path)
    parser.add_argument("--candidate-protocol", required=True, type=Path)
    args = parser.parse_args()

    root = args.root.resolve()
    old_code = root / "code_sha256.txt"
    old_protocol = root / "protocol.env"
    new_code = args.candidate_code.resolve()
    new_protocol = args.candidate_protocol.resolve()
    expected = (
        (old_code, EXPECTED_OLD_CODE_SHA),
        (new_code, EXPECTED_NEW_CODE_SHA),
        (old_protocol, EXPECTED_PROTOCOL_SHA),
        (new_protocol, EXPECTED_PROTOCOL_SHA),
        (root / "run_manifest.tsv", EXPECTED_RUN_MANIFEST_SHA),
    )
    for path, expected_digest in expected:
        actual = sha256(path)
        if actual != expected_digest:
            raise RuntimeError(f"hash mismatch for {path}: {actual} != {expected_digest}")
    if old_protocol.read_bytes() != new_protocol.read_bytes():
        raise RuntimeError("execution protocol changed during control-plane migration")
    verify_completed_prefix(root)

    old_paths = parse_code_manifest(old_code)
    new_paths = parse_code_manifest(new_code)
    changed = {
        path for path in old_paths.keys() & new_paths
        if old_paths[path] != new_paths[path]
    }
    added = set(new_paths) - set(old_paths)
    removed = set(old_paths) - set(new_paths)
    if changed != {MAIN} or added != {COMPARATOR} or removed:
        raise RuntimeError(
            f"unexpected code delta changed={changed} added={added} removed={removed}"
        )
    for path in (set(old_paths) & set(new_paths)) - {MAIN}:
        if old_paths[path] != new_paths[path]:
            raise RuntimeError(f"runtime code unexpectedly changed: {path}")

    archive = root / "code_sha256.pre_matched_eager_comparison.txt"
    audit_path = root / "ADD_MATCHED_EAGER_COMPARISON_MIGRATION.json"
    if archive.exists() and archive.read_bytes() != old_code.read_bytes():
        raise RuntimeError(f"existing archive is inconsistent: {archive}")
    if not archive.exists():
        atomic_write(archive, old_code.read_bytes())
    audit = {
        "schema_version": 1,
        "status": "COMMITTED",
        "migration_scope": "control_plane_and_post_run_analysis_only",
        "runtime_worker_code_changed": False,
        "execution_protocol_changed": False,
        "completed_run_artifacts_changed": False,
        "completed_arms_preserved": [
            "seed0/vanilla",
            "seed0/lengthsort",
            "seed0/lengthsort_guard",
        ],
        "old_code_manifest_sha256": EXPECTED_OLD_CODE_SHA,
        "new_code_manifest_sha256": EXPECTED_NEW_CODE_SHA,
        "protocol_sha256": EXPECTED_PROTOCOL_SHA,
        "run_manifest_sha256": EXPECTED_RUN_MANIFEST_SHA,
        "changed_code_paths": sorted(changed),
        "added_code_paths": sorted(added),
        "reason": (
            "Add a matched TQ1 eager execution mode and a fail-closed paired "
            "paper summarizer while preserving the frozen graph execution contract."
        ),
        "created_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "migration_tool_sha256": sha256(Path(__file__).resolve()),
    }
    atomic_write(audit_path, (json.dumps(audit, indent=2, sort_keys=True) + "\n").encode())
    atomic_write(old_code, new_code.read_bytes())
    print(json.dumps(audit, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
