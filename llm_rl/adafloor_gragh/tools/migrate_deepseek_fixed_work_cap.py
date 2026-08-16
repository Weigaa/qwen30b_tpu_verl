#!/usr/bin/env python3
"""Migrate an authorized DeepSeek KV cap contract to fixed-work code."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
from pathlib import Path


SHA256_RE = re.compile(r"[0-9a-f]{64}")
EXECUTION_KEY = "DEEPSEEK_EXECUTION_CODE_SHA256"
AUTHORIZED_KEY = "DEEPSEEK_KV_CAP_AUTHORIZED_RUNTIME_EXECUTION_CODE_SHA256"
CONTINUATION_KEY = "DEEPSEEK_KV_CAP_CONTINUATION_EXECUTION_CODE_SHA256"
MIGRATED_KEYS = (EXECUTION_KEY, CONTINUATION_KEY)


class MigrationError(RuntimeError):
    pass


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _require_sha256(value: str, label: str) -> str:
    normalized = value.strip().lower()
    if SHA256_RE.fullmatch(normalized) is None:
        raise MigrationError(f"{label} is not a canonical SHA256")
    return normalized


def _read_contract(path: Path) -> tuple[bytes, list[str], dict[str, str]]:
    try:
        raw = path.read_bytes()
        text = raw.decode("ascii")
    except (OSError, UnicodeDecodeError) as error:
        raise MigrationError(f"cannot read source cap contract {path}: {error}") from error
    lines = text.splitlines(keepends=True)
    values: dict[str, str] = {}
    counts = {key: 0 for key in (EXECUTION_KEY, AUTHORIZED_KEY, CONTINUATION_KEY)}
    assignment = re.compile(r"^export ([A-Z0-9_]+)=([^\r\n]*)\r?\n?$")
    for line in lines:
        match = assignment.fullmatch(line)
        if match is None:
            continue
        key, value = match.groups()
        if key in counts:
            counts[key] += 1
            values[key] = _require_sha256(value, key)
    invalid = {key: count for key, count in counts.items() if count != 1}
    if invalid:
        raise MigrationError(
            "source cap contract must define each provenance hash exactly once, "
            f"found {invalid}"
        )
    if values[EXECUTION_KEY] != values[CONTINUATION_KEY]:
        raise MigrationError(
            "source cap execution and continuation hashes must identify the same "
            "authorized continuation"
        )
    return raw, lines, values


def _migrated_bytes(lines: list[str], current_execution_sha256: str) -> bytes:
    migrated: list[str] = []
    for line in lines:
        replaced = False
        for key in MIGRATED_KEYS:
            if line.startswith(f"export {key}="):
                newline = "\r\n" if line.endswith("\r\n") else "\n"
                migrated.append(f"export {key}={current_execution_sha256}{newline}")
                replaced = True
                break
        if not replaced:
            migrated.append(line)
    return "".join(migrated).encode("ascii")


def _write_new_or_identical(path: Path, raw: bytes, label: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        try:
            existing = path.read_bytes()
        except OSError as error:
            raise MigrationError(f"cannot read existing {label} {path}: {error}") from error
        if existing != raw:
            raise MigrationError(f"existing {label} is stale: {path}")
        return
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def migrate(
    source: Path,
    output: Path,
    manifest: Path,
    current_execution_sha256: str,
    common_root: Path,
) -> dict[str, object]:
    source = source.resolve()
    output = output.resolve()
    manifest = manifest.resolve()
    common_root = common_root.resolve()
    current = _require_sha256(
        current_execution_sha256, "fixed-work execution code SHA256"
    )
    if source == output:
        raise MigrationError("source and migrated cap paths must differ")
    source_raw, source_lines, source_values = _read_contract(source)
    migrated_raw = _migrated_bytes(source_lines, current)
    _write_new_or_identical(output, migrated_raw, "migrated cap contract")

    _, _, migrated_values = _read_contract(output)
    if migrated_values[EXECUTION_KEY] != current:
        raise MigrationError("migrated execution hash does not match fixed-work code")
    if migrated_values[CONTINUATION_KEY] != current:
        raise MigrationError("migrated continuation hash does not match fixed-work code")
    if migrated_values[AUTHORIZED_KEY] != source_values[AUTHORIZED_KEY]:
        raise MigrationError("migration changed the authorized runtime hash")

    payload: dict[str, object] = {
        "status": "PASS",
        "scope": "fixed_work_replay_and_preemption_accounting_only",
        "source_cap_env": str(source),
        "source_cap_env_sha256": _sha256(source_raw),
        "migrated_cap_env": str(output),
        "migrated_cap_env_sha256": _sha256(migrated_raw),
        "source_execution_code_sha256": source_values[EXECUTION_KEY],
        "authorized_runtime_execution_code_sha256": source_values[AUTHORIZED_KEY],
        "source_continuation_execution_code_sha256": source_values[
            CONTINUATION_KEY
        ],
        "fixed_work_execution_code_sha256": current,
        "common_epoch0_root": str(common_root),
        "kv_allocation_or_lifecycle_changed": False,
        "reason": (
            "The continuation adds exact-length replay metadata and preemption "
            "accounting. It does not change model residency, KV allocation, "
            "elastic groups, or the authorized physical capacities."
        ),
    }
    manifest_raw = (
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    ).encode("ascii")
    _write_new_or_identical(manifest, manifest_raw, "cap migration manifest")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--current-execution-sha256", required=True)
    parser.add_argument("--common-root", type=Path, required=True)
    args = parser.parse_args()
    try:
        payload = migrate(
            args.source,
            args.output,
            args.manifest,
            args.current_execution_sha256,
            args.common_root,
        )
    except MigrationError as error:
        parser.error(str(error))
    print(
        "migrated_cap="
        f"{payload['migrated_cap_env']} "
        f"sha256={payload['migrated_cap_env_sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
