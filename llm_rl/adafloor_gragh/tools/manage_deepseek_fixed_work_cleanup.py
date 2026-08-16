#!/usr/bin/env python3
"""Manage post-verification checkpoint cleanup for DeepSeek fixed-work runs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any


SCHEMA_VERSION = 1
RETAINED_MARKER = "CHECKPOINTS_RETAINED_AFTER_VALIDATION.txt"
REMOVED_MARKER = "CHECKPOINTS_REMOVED_AFTER_VALIDATION.txt"


class TransactionError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as error:
        raise TransactionError(f"cannot hash transaction input {path}: {error}") from error


def _require_sha256(value: str, label: str) -> str:
    normalized = value.strip().lower()
    if re.fullmatch(r"[0-9a-f]{64}", normalized) is None:
        raise TransactionError(f"{label} is not a canonical SHA256")
    return normalized


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise TransactionError(f"cannot read {label} {path}: {error}") from error
    if not isinstance(value, dict):
        raise TransactionError(f"{label} is not a JSON object: {path}")
    return value


def _atomic_write_new_or_identical(path: Path, payload: dict[str, Any]) -> None:
    raw = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != raw:
            raise TransactionError(f"existing cleanup transaction is stale: {path}")
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


def _validated_result(
    path: Path,
    label: str,
    allowed_statuses: tuple[str, ...],
    expected_lifecycle: str | None = None,
) -> dict[str, Any]:
    path = path.resolve()
    payload = _load_json(path, label)
    if payload.get("status") not in allowed_statuses:
        raise TransactionError(
            f"{label} status={payload.get('status')!r}, expected one of "
            f"{allowed_statuses}"
        )
    if expected_lifecycle is not None and payload.get("lifecycle") != expected_lifecycle:
        raise TransactionError(
            f"{label} lifecycle={payload.get('lifecycle')!r}, "
            f"expected {expected_lifecycle!r}"
        )
    return {"path": str(path), "sha256": _sha256(path)}


def _arm_record(run: Path) -> dict[str, str]:
    run = run.resolve()
    if not run.is_dir():
        raise TransactionError(f"arm run directory does not exist: {run}")
    return {"run_root": str(run)}


def build_payload(
    *,
    phase: str,
    mode: str,
    summary: Path,
    allowed_summary_statuses: tuple[str, ...],
    cap_env: Path,
    execution_code_sha256: str,
    vanilla_run: Path,
    adafloor_run: Path,
    adafloor_audit: Path,
) -> dict[str, Any]:
    if phase not in {"gate", "epoch"}:
        raise TransactionError(f"unsupported phase {phase!r}")
    if mode not in {"natural", "fixed"}:
        raise TransactionError(f"unsupported mode {mode!r}")
    execution = _require_sha256(execution_code_sha256, "execution code SHA256")
    cap_env = cap_env.resolve()
    if not cap_env.is_file():
        raise TransactionError(f"cap contract does not exist: {cap_env}")
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "PAIR_VERIFIED",
        "phase": phase,
        "mode": mode,
        "execution_code_sha256": execution,
        "cap_env": {"path": str(cap_env), "sha256": _sha256(cap_env)},
        "pair_summary": _validated_result(
            summary,
            "pair summary",
            allowed_summary_statuses,
        ),
        "adafloor_plan_runtime_audit": _validated_result(
            adafloor_audit,
            "AdaFloor plan/runtime audit",
            ("PASS",),
            "natural_f2",
        ),
        "arms": {
            "vanilla": _arm_record(vanilla_run),
            "adafloor": _arm_record(adafloor_run),
        },
    }


def _validate_recorded_inputs(payload: dict[str, Any]) -> None:
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise TransactionError("unsupported cleanup transaction schema")
    _require_sha256(str(payload.get("execution_code_sha256", "")), "recorded execution hash")
    for key in ("cap_env", "pair_summary", "adafloor_plan_runtime_audit"):
        record = payload.get(key)
        if not isinstance(record, dict):
            raise TransactionError(f"cleanup transaction lacks {key}")
        path_value = record.get("path")
        digest_value = record.get("sha256")
        if not isinstance(path_value, str) or not path_value:
            raise TransactionError(f"cleanup transaction has invalid {key} path")
        expected = _require_sha256(str(digest_value or ""), f"recorded {key} SHA256")
        if _sha256(Path(path_value)) != expected:
            raise TransactionError(f"cleanup transaction input changed: {path_value}")
    arms = payload.get("arms")
    if not isinstance(arms, dict) or set(arms) != {"vanilla", "adafloor"}:
        raise TransactionError("cleanup transaction has invalid arm set")
    for arm, record in arms.items():
        if not isinstance(record, dict) or not isinstance(record.get("run_root"), str):
            raise TransactionError(f"cleanup transaction has invalid {arm} arm")
        if not Path(record["run_root"]).resolve().is_dir():
            raise TransactionError(f"cleanup transaction arm disappeared: {arm}")


def _base_payload(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in payload.items()
        if key not in {"cleanup_markers", "status"}
    }


def prepare(pending: Path, committed: Path, expected: dict[str, Any]) -> None:
    pending = pending.resolve()
    committed = committed.resolve()
    if committed.exists():
        observed = _load_json(committed, "committed cleanup transaction")
        _validate_recorded_inputs(observed)
        if _base_payload(observed) != _base_payload(expected):
            raise TransactionError("committed cleanup transaction is stale")
        validate_committed_cleanup(observed)
        if pending.exists():
            pending.unlink()
        return
    if pending.exists():
        observed = _load_json(pending, "pending cleanup transaction")
        _validate_recorded_inputs(observed)
        if observed != expected:
            raise TransactionError("pending cleanup transaction is stale")
        return
    for arm, record in expected["arms"].items():
        run = Path(record["run_root"])
        if not (run / RETAINED_MARKER).is_file():
            raise TransactionError(f"{arm} lacks retained-checkpoint validation marker")
        if (run / REMOVED_MARKER).exists():
            raise TransactionError(f"{arm} was cleaned before pair verification")
    _atomic_write_new_or_identical(pending, expected)


def transaction_allows_removed_arm(
    pending: Path,
    committed: Path,
    arm: str,
    arm_run: Path,
    execution_code_sha256: str,
) -> None:
    if arm not in {"vanilla", "adafloor"}:
        raise TransactionError(f"unsupported arm {arm!r}")
    records = [path.resolve() for path in (committed, pending) if path.is_file()]
    if not records:
        raise TransactionError("removed checkpoints have no pair-verification transaction")
    payload = _load_json(records[0], "cleanup transaction")
    _validate_recorded_inputs(payload)
    if payload.get("execution_code_sha256") != _require_sha256(
        execution_code_sha256, "execution code SHA256"
    ):
        raise TransactionError("cleanup transaction execution hash is stale")
    recorded = payload["arms"][arm]["run_root"]
    if Path(recorded).resolve() != arm_run.resolve():
        raise TransactionError(f"cleanup transaction {arm} run path is stale")


def validate_committed_cleanup(payload: dict[str, Any]) -> None:
    if payload.get("status") != "COMMITTED":
        raise TransactionError("cleanup transaction is not committed")
    markers = payload.get("cleanup_markers")
    if not isinstance(markers, dict) or set(markers) != {"vanilla", "adafloor"}:
        raise TransactionError("committed cleanup transaction has invalid markers")
    for arm, arm_record in payload["arms"].items():
        run = Path(arm_record["run_root"])
        marker = run / REMOVED_MARKER
        expected = markers[arm]
        if not isinstance(expected, dict):
            raise TransactionError(f"committed cleanup marker record is invalid for {arm}")
        if str(marker.resolve()) != expected.get("path") or _sha256(marker) != expected.get(
            "sha256"
        ):
            raise TransactionError(f"committed cleanup marker changed for {arm}")
        if (run / RETAINED_MARKER).exists():
            raise TransactionError(f"{arm} still has a retained-checkpoint marker")
        if any(run.glob("epoch_*/checkpoints")):
            raise TransactionError(f"{arm} still contains epoch checkpoints")


def commit(pending: Path, committed: Path) -> dict[str, Any]:
    pending = pending.resolve()
    committed = committed.resolve()
    if committed.exists():
        payload = _load_json(committed, "committed cleanup transaction")
        _validate_recorded_inputs(payload)
        validate_committed_cleanup(payload)
        if pending.exists():
            pending.unlink()
        return payload
    payload = _load_json(pending, "pending cleanup transaction")
    _validate_recorded_inputs(payload)
    markers: dict[str, dict[str, str]] = {}
    for arm, arm_record in payload["arms"].items():
        run = Path(arm_record["run_root"])
        marker = run / REMOVED_MARKER
        if not marker.is_file():
            raise TransactionError(f"{arm} checkpoint cleanup is incomplete")
        if (run / RETAINED_MARKER).exists() or any(run.glob("epoch_*/checkpoints")):
            raise TransactionError(f"{arm} checkpoint cleanup is inconsistent")
        markers[arm] = {"path": str(marker.resolve()), "sha256": _sha256(marker)}
    committed_payload = {**payload, "status": "COMMITTED", "cleanup_markers": markers}
    _atomic_write_new_or_identical(committed, committed_payload)
    pending.unlink()
    return committed_payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--pending", type=Path, required=True)
    prepare_parser.add_argument("--committed", type=Path, required=True)
    prepare_parser.add_argument("--phase", required=True)
    prepare_parser.add_argument("--mode", required=True)
    prepare_parser.add_argument("--summary", type=Path, required=True)
    prepare_parser.add_argument(
        "--allowed-summary-status", action="append", required=True
    )
    prepare_parser.add_argument("--cap-env", type=Path, required=True)
    prepare_parser.add_argument("--execution-code-sha256", required=True)
    prepare_parser.add_argument("--vanilla-run", type=Path, required=True)
    prepare_parser.add_argument("--adafloor-run", type=Path, required=True)
    prepare_parser.add_argument("--adafloor-audit", type=Path, required=True)

    allowed_parser = subparsers.add_parser("allows-removed")
    allowed_parser.add_argument("--pending", type=Path, required=True)
    allowed_parser.add_argument("--committed", type=Path, required=True)
    allowed_parser.add_argument("--arm", required=True)
    allowed_parser.add_argument("--arm-run", type=Path, required=True)
    allowed_parser.add_argument("--execution-code-sha256", required=True)

    commit_parser = subparsers.add_parser("commit")
    commit_parser.add_argument("--pending", type=Path, required=True)
    commit_parser.add_argument("--committed", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "prepare":
            payload = build_payload(
                phase=args.phase,
                mode=args.mode,
                summary=args.summary,
                allowed_summary_statuses=tuple(args.allowed_summary_status),
                cap_env=args.cap_env,
                execution_code_sha256=args.execution_code_sha256,
                vanilla_run=args.vanilla_run,
                adafloor_run=args.adafloor_run,
                adafloor_audit=args.adafloor_audit,
            )
            prepare(args.pending, args.committed, payload)
        elif args.command == "allows-removed":
            transaction_allows_removed_arm(
                args.pending,
                args.committed,
                args.arm,
                args.arm_run,
                args.execution_code_sha256,
            )
        else:
            payload = commit(args.pending, args.committed)
            print(json.dumps(payload, sort_keys=True))
    except TransactionError as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
