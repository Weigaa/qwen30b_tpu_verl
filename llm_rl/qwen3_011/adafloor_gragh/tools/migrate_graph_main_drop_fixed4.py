#!/usr/bin/env python3
"""Remove the unexecuted Fixed4 arm from the frozen graph-main matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path


EXPECTED_OLD_CODE_SHA = "ddbe24988ca7c8a7546c37a021ead2e80d8d5bea3c90fd2aff5a981f1281b539"
EXPECTED_NEW_CODE_SHA = "f99f226a850dc5d46aa3dcedfd9dae6febafe590b14beec753c60dbdc60033c0"
EXPECTED_OLD_PROTOCOL_SHA = "478c6417d868d971bcce92924c3bf5464d083e164cd0836c24d726339827ebb4"
EXPECTED_NEW_PROTOCOL_SHA = "a8e9b59771c65c04153fc750e4724f002b8c29f907671b9f111cb6517e9d8f9e"
EXPECTED_RUN_MANIFEST_SHA = "aaed67a81e2c4b78f4f7bd607c8d59e0eebac189f3f37de8713e47d01775dad5"
MAIN = "/workspace/cann-recipes-train/llm_rl/qwen3_011/adafloor_gragh/run_qwen3_paper_graph_main_from_common_epoch0.sh"
FIXED4 = "/workspace/cann-recipes-train/llm_rl/qwen3_011/adafloor_gragh/run_baseline_kvsafe_fixed4_epoch1_2.sh"


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
        raise RuntimeError("run manifest does not contain exactly three arms")
    for row, (index, seed, policy) in zip(rows[1:], expected, strict=True):
        fields = row.split("\t")
        if fields[:4] != [index, seed, policy, "complete"] or len(fields) != 5:
            raise RuntimeError(f"unexpected completed-arm row: {row}")
        arm_root = Path(fields[4])
        if not (arm_root / "CHECKPOINTS_REMOVED_AFTER_VALIDATION.txt").is_file():
            raise RuntimeError(f"cleanup marker missing for {policy}: {arm_root}")
        forbidden = [
            path for path in arm_root.rglob("*")
            if path.is_dir() and (
                path.name == "checkpoints" or path.name.startswith("global_step_")
            )
        ]
        if forbidden:
            raise RuntimeError(f"validated arm retains checkpoints: {forbidden}")


def verify_fixed4_never_executed(root: Path) -> list[str]:
    fixed4_root = root / "seed_0/fixed4"
    if not fixed4_root.exists():
        return []
    unexpected = [
        path for path in fixed4_root.rglob("*")
        if path.is_file() or path.is_symlink()
    ]
    if unexpected:
        raise RuntimeError(f"Fixed4 contains execution artifacts: {unexpected}")
    return sorted(str(path.relative_to(root)) for path in fixed4_root.rglob("*") if path.is_dir())


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

    expected_hashes = (
        (old_code, EXPECTED_OLD_CODE_SHA),
        (new_code, EXPECTED_NEW_CODE_SHA),
        (old_protocol, EXPECTED_OLD_PROTOCOL_SHA),
        (new_protocol, EXPECTED_NEW_PROTOCOL_SHA),
    )
    for path, expected in expected_hashes:
        actual = sha256(path)
        if actual != expected:
            raise RuntimeError(f"hash mismatch for {path}: {actual} != {expected}")

    verify_completed_prefix(root)
    empty_fixed4_dirs = verify_fixed4_never_executed(root)

    old_paths = parse_code_manifest(old_code)
    new_paths = parse_code_manifest(new_code)
    changed = {
        path for path in old_paths.keys() & new_paths
        if old_paths[path] != new_paths[path]
    }
    added = set(new_paths) - set(old_paths)
    removed = set(old_paths) - set(new_paths)
    if changed != {MAIN} or added or removed != {FIXED4}:
        raise RuntimeError(
            f"unexpected code delta changed={changed} added={added} removed={removed}"
        )

    old_protocol_lines = old_protocol.read_text(encoding="utf-8").splitlines()
    new_protocol_lines = new_protocol.read_text(encoding="utf-8").splitlines()
    expected_removed_line = (
        "policies=vanilla,lengthsort,lengthsort_guard,fixed4,minskew,"
        "natural_f4,natural_f2,planned_f4"
    )
    expected_added = {
        "policies=vanilla,lengthsort,lengthsort_guard,minskew,natural_f4,natural_f2,planned_f4",
        "excluded_policy=fixed4",
        "fixed4_exclusion_reason=not_a_reasonable_comparison_strategy",
        "planned_arms=21",
        "planned_epochs=42",
    }
    removed_protocol = set(old_protocol_lines) - set(new_protocol_lines)
    added_protocol = set(new_protocol_lines) - set(old_protocol_lines)
    if removed_protocol != {expected_removed_line} or added_protocol != expected_added:
        raise RuntimeError(
            f"unexpected protocol delta removed={removed_protocol} added={added_protocol}"
        )

    code_archive = root / "code_sha256.pre_drop_fixed4.txt"
    protocol_archive = root / "protocol.pre_drop_fixed4.env"
    audit_path = root / "DROP_FIXED4_MATRIX_MIGRATION.json"
    for archive, source in ((code_archive, old_code), (protocol_archive, old_protocol)):
        if archive.exists() and archive.read_bytes() != source.read_bytes():
            raise RuntimeError(f"existing archive is inconsistent: {archive}")
        if not archive.exists():
            atomic_write(archive, source.read_bytes())

    audit = {
        "schema_version": 1,
        "status": "COMMITTED",
        "migration_scope": "unexecuted_matrix_arm_removal",
        "runtime_worker_code_changed": False,
        "completed_arms_preserved": ["seed0/vanilla", "seed0/lengthsort", "seed0/lengthsort_guard"],
        "excluded_policy": "fixed4",
        "excluded_before_execution": True,
        "empty_fixed4_directories": empty_fixed4_dirs,
        "old_code_manifest_sha256": EXPECTED_OLD_CODE_SHA,
        "new_code_manifest_sha256": EXPECTED_NEW_CODE_SHA,
        "old_protocol_sha256": EXPECTED_OLD_PROTOCOL_SHA,
        "new_protocol_sha256": EXPECTED_NEW_PROTOCOL_SHA,
        "run_manifest_sha256": EXPECTED_RUN_MANIFEST_SHA,
        "changed_code_paths": sorted(changed),
        "removed_code_paths": sorted(removed),
        "reason": (
            "Fixed4 was removed before execution because it is not a meaningful "
            "paper comparison under the graph-specific KV budget."
        ),
        "created_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "migration_tool_sha256": sha256(Path(__file__).resolve()),
    }
    atomic_write(audit_path, (json.dumps(audit, indent=2, sort_keys=True) + "\n").encode())
    atomic_write(old_code, new_code.read_bytes())
    atomic_write(old_protocol, new_protocol.read_bytes())
    print(json.dumps(audit, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
