#!/usr/bin/env python3
"""Atomically record the graph-main validation-only code contract migration."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path


EXPECTED_OLD_MANIFEST_SHA = "5cb0c43ae99ce968ea1892cb1320ee49e6a2549051b8e1503226982c0394a8ed"
EXPECTED_NEW_MANIFEST_SHA = "b70c3ab19cdf326dcae90a716dcbade8de6fc42fd1c43eb2b159071fd5c47f6e"
MAIN = "/workspace/cann-recipes-train/llm_rl/qwen3_011/adafloor_gragh/run_qwen3_paper_graph_main_from_common_epoch0.sh"
FAIR = "/workspace/cann-recipes-train/llm_rl/qwen3_011/adafloor_gragh/run_paper_fair_epoch1_2_from_common_epoch0.sh"
VERIFIER = "/workspace/cann-recipes-train/llm_rl/qwen3_011/adafloor_gragh/tools/verify_full_decode_preemption_recovery.py"
EXPECTED_CHANGED = {MAIN, FAIR}
EXPECTED_ADDED = {VERIFIER}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_manifest(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        digest, filename = line.split(maxsplit=1)
        if filename in result:
            raise RuntimeError(f"duplicate manifest path: {filename}")
        result[filename] = digest
    return result


def atomic_write(path: Path, content: bytes) -> None:
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        stream.write(content)
        temporary = Path(stream.name)
    os.replace(temporary, path)


def verify_completed_execution(root: Path) -> None:
    manifest = root / "run_manifest.tsv"
    rows = manifest.read_text(encoding="utf-8").splitlines()
    if len(rows) != 3:
        raise RuntimeError("run manifest is not the exact two-arm completed prefix")
    if not rows[1].startswith("1\t0\tvanilla\tcomplete\t"):
        raise RuntimeError("seed-0 Vanilla completion row is missing")
    if not rows[2].startswith("2\t0\tlengthsort\tcomplete\t"):
        raise RuntimeError("seed-0 LengthSort completion row is missing")

    arm = root / "seed_0/lengthsort_guard/qwen3_graph_lengthsort_guard_seed0"
    if (arm / "CHECKPOINTS_REMOVED_AFTER_VALIDATION.txt").exists():
        raise RuntimeError("LengthSort+TailGuard is already finalized")
    for epoch_tag in ("001", "002"):
        epochs = list(arm.glob(f"epoch_{epoch_tag}_*"))
        if len(epochs) != 1:
            raise RuntimeError(f"expected one completed epoch_{epoch_tag}")
        epoch = epochs[0]
        if len(list((epoch / "rollout_data").glob("*.jsonl"))) != 5:
            raise RuntimeError(f"epoch_{epoch_tag} does not have five rollouts")
        if len(list((epoch / "rollout_length").glob("length_*.txt"))) != 5:
            raise RuntimeError(f"epoch_{epoch_tag} does not have five length files")
        logs = list((epoch / "logs").glob("*.txt"))
        if len(logs) != 1 or "After trainer.fit" not in logs[0].read_text(
            encoding="utf-8", errors="replace"
        ):
            raise RuntimeError(f"epoch_{epoch_tag} is not execution-complete")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    args = parser.parse_args()

    root = args.root.resolve()
    old_path = root / "code_sha256.txt"
    new_path = args.candidate.resolve()
    archive = root / "code_sha256.pre_validation_fix.txt"
    audit_path = root / "CODE_CONTRACT_VALIDATION_MIGRATION.json"

    if sha256(old_path) != EXPECTED_OLD_MANIFEST_SHA:
        raise RuntimeError("formal root does not have the expected old code manifest")
    if sha256(new_path) != EXPECTED_NEW_MANIFEST_SHA:
        raise RuntimeError("candidate does not have the expected new code manifest")
    verify_completed_execution(root)

    old = parse_manifest(old_path)
    new = parse_manifest(new_path)
    changed = {path for path in old.keys() & new if old[path] != new[path]}
    added = set(new) - set(old)
    removed = set(old) - set(new)
    if changed != EXPECTED_CHANGED or added != EXPECTED_ADDED or removed:
        raise RuntimeError(
            f"unexpected contract delta changed={changed} added={added} removed={removed}"
        )
    for path in (old.keys() & new.keys()) - changed:
        if old[path] != new[path]:
            raise RuntimeError(f"runtime code drifted unexpectedly: {path}")

    old_bytes = old_path.read_bytes()
    if archive.exists():
        if archive.read_bytes() != old_bytes:
            raise RuntimeError("archived old code manifest is inconsistent")
    else:
        atomic_write(archive, old_bytes)

    audit = {
        "schema_version": 1,
        "status": "COMMITTED",
        "migration_scope": "post_run_validation_only",
        "runtime_execution_code_changed": False,
        "old_manifest_sha256": EXPECTED_OLD_MANIFEST_SHA,
        "new_manifest_sha256": EXPECTED_NEW_MANIFEST_SHA,
        "changed_paths": sorted(changed),
        "added_paths": sorted(added),
        "removed_paths": [],
        "reason": (
            "Accept KV preemption only after a fail-closed synchronized eager "
            "recovery and FULL graph resume audit."
        ),
        "created_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "migration_tool_sha256": sha256(Path(__file__).resolve()),
    }
    atomic_write(audit_path, (json.dumps(audit, indent=2, sort_keys=True) + "\n").encode())
    atomic_write(old_path, new_path.read_bytes())
    print(json.dumps(audit, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
