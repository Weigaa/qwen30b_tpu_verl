#!/usr/bin/env python3
"""Validate an immutable Natural source before fixed-work replay."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any


SHA256_RE = re.compile(r"[0-9a-f]{64}")
VANILLA_RUN_NAME = "deepseek_v2_lite_vanilla_common_epoch0_epoch1_2"
ADAFLOOR_RUN_NAME = "deepseek_v2_lite_adafloor_n_f2_common_epoch0_epoch1_2"


class SourceValidationError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as error:
        raise SourceValidationError(f"cannot hash {path}: {error}") from error


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SourceValidationError(f"cannot read {label} {path}: {error}") from error
    if not isinstance(payload, dict):
        raise SourceValidationError(f"{label} is not a JSON object: {path}")
    return payload


def _require_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value.lower()) is None:
        raise SourceValidationError(f"{label} is not a canonical SHA256")
    return value.lower()


def _require_hashed_record(
    record: Any,
    expected_path: Path,
    label: str,
) -> None:
    if not isinstance(record, dict):
        raise SourceValidationError(f"cleanup transaction lacks {label}")
    recorded_path = record.get("path")
    if not isinstance(recorded_path, str):
        raise SourceValidationError(f"cleanup transaction has invalid {label} path")
    if Path(recorded_path).resolve() != expected_path.resolve():
        raise SourceValidationError(
            f"cleanup transaction {label} path differs from the immutable source"
        )
    expected_sha256 = _require_sha256(record.get("sha256"), f"{label} SHA256")
    if _sha256(expected_path) != expected_sha256:
        raise SourceValidationError(
            f"immutable source {label} changed after pair verification"
        )


def _resolve_epoch(run_root: Path) -> Path:
    candidates = sorted(run_root.glob("epoch_001_mode1_natural"))
    if len(candidates) != 1 or not (candidates[0] / "rollout_data").is_dir():
        raise SourceValidationError(
            f"expected one complete Natural epoch under {run_root}"
        )
    return candidates[0].resolve()


def validate_source(
    source_natural_root: Path,
    expected_execution_sha256: str,
    expected_recovery_sha256: str,
) -> dict[str, Any]:
    root = source_natural_root.expanduser().resolve()
    expected_execution = _require_sha256(
        expected_execution_sha256, "expected source execution SHA256"
    )
    expected_recovery = _require_sha256(
        expected_recovery_sha256, "expected source recovery SHA256"
    )
    if not root.is_dir():
        raise SourceValidationError(f"Natural source root does not exist: {root}")

    summary_path = root / "natural_epoch_summary.json"
    transaction_path = root / "PAIR_VERIFIED_CHECKPOINT_CLEANUP.json"
    vanilla_run = (root / VANILLA_RUN_NAME).resolve()
    adafloor_run = (root / ADAFLOOR_RUN_NAME).resolve()
    audit_path = adafloor_run / "DEEPSEEK_PLAN_RUNTIME_AUDIT.json"
    recovery_path = adafloor_run / "POSTVALIDATION_RECOVERY.json"
    summary = _load_json(summary_path, "Natural pair summary")
    transaction = _load_json(transaction_path, "cleanup transaction")
    audit = _load_json(audit_path, "AdaFloor plan/runtime audit")
    recovery = _load_json(recovery_path, "Natural post-validation recovery")
    if _sha256(recovery_path) != expected_recovery:
        raise SourceValidationError(
            "Natural source post-validation recovery SHA256 differs"
        )

    if summary.get("status") != "PASS" or summary.get("phase") != "epoch":
        raise SourceValidationError("Natural source pair summary is not a passing epoch")
    provenance = summary.get("provenance")
    if not isinstance(provenance, dict):
        raise SourceValidationError("Natural source summary lacks provenance")
    if provenance.get("execution_code_sha256") != expected_execution:
        raise SourceValidationError("Natural source summary execution hash is stale")
    common_value = provenance.get("common_root")
    cap_value = provenance.get("cap_env")
    if not isinstance(common_value, str) or not isinstance(cap_value, str):
        raise SourceValidationError("Natural source summary lacks common/cap paths")
    common_root = Path(common_value).resolve()
    cap_env = Path(cap_value).resolve()
    if not common_root.is_dir() or not cap_env.is_file():
        raise SourceValidationError("Natural source common root or cap contract disappeared")
    if _sha256(cap_env) != provenance.get("cap_env_sha256"):
        raise SourceValidationError("Natural source cap contract changed")

    if (
        transaction.get("status") != "COMMITTED"
        or transaction.get("phase") != "epoch"
        or transaction.get("mode") != "natural"
        or transaction.get("execution_code_sha256") != expected_execution
    ):
        raise SourceValidationError(
            "Natural source cleanup transaction is not a committed epoch pair"
        )
    _require_hashed_record(
        transaction.get("pair_summary"), summary_path, "pair_summary"
    )
    _require_hashed_record(
        transaction.get("adafloor_plan_runtime_audit"),
        audit_path,
        "adafloor_plan_runtime_audit",
    )
    _require_hashed_record(transaction.get("cap_env"), cap_env, "cap_env")
    arms = transaction.get("arms")
    if not isinstance(arms, dict) or set(arms) != {"vanilla", "adafloor"}:
        raise SourceValidationError("Natural source cleanup transaction has invalid arms")
    expected_runs = {"vanilla": vanilla_run, "adafloor": adafloor_run}
    markers = transaction.get("cleanup_markers")
    if not isinstance(markers, dict) or set(markers) != set(expected_runs):
        raise SourceValidationError("Natural source cleanup transaction lacks markers")
    for arm, expected_run in expected_runs.items():
        record = arms.get(arm)
        if (
            not isinstance(record, dict)
            or not isinstance(record.get("run_root"), str)
            or Path(record["run_root"]).resolve() != expected_run
        ):
            raise SourceValidationError(
                f"Natural source cleanup transaction has the wrong {arm} root"
            )
        removed = expected_run / "CHECKPOINTS_REMOVED_AFTER_VALIDATION.txt"
        _require_hashed_record(markers.get(arm), removed, f"{arm} cleanup marker")
        if (expected_run / "CHECKPOINTS_RETAINED_AFTER_VALIDATION.txt").exists():
            raise SourceValidationError(f"Natural source {arm} still has retained marker")
        if any(expected_run.glob("epoch_*/checkpoints")):
            raise SourceValidationError(f"Natural source {arm} still has checkpoints")

    if audit.get("status") != "PASS" or audit.get("lifecycle") != "natural_f2":
        raise SourceValidationError("Natural source AdaFloor audit is not passing")
    if Path(str(audit.get("run_root", ""))).resolve() != adafloor_run:
        raise SourceValidationError("Natural source AdaFloor audit has the wrong run root")
    adafloor_epoch = _resolve_epoch(adafloor_run)
    actual_plan = adafloor_epoch / "oracle" / "length_sorted_rank_plan.json"
    if not actual_plan.is_file():
        raise SourceValidationError("Natural source lacks the actual AdaFloor plan")
    if (
        recovery.get("status") != "PASS"
        or recovery.get("execution_code_sha256") != expected_execution
        or recovery.get("cap_env_sha256") != _sha256(cap_env)
        or recovery.get("formal_audit_sha256") != _sha256(audit_path)
    ):
        raise SourceValidationError(
            "Natural source post-validation recovery provenance is invalid"
        )
    plan_hashes = recovery.get("plan_sha256")
    if (
        not isinstance(plan_hashes, dict)
        or plan_hashes.get(str(actual_plan.resolve())) != _sha256(actual_plan)
    ):
        raise SourceValidationError(
            "Natural source actual plan differs from its recovery record"
        )
    request_audit = recovery.get("request_audit")
    artifact_hashes = (
        request_audit.get("artifact_sha256", {}).get("adafloor")
        if isinstance(request_audit, dict)
        else None
    )
    expected_artifacts = {
        (adafloor_epoch / "rollout_data" / f"{step}.jsonl").resolve()
        for step in range(1, 6)
    } | {
        (adafloor_epoch / "rollout_length" / f"length_{step}.txt").resolve()
        for step in range(1, 6)
    }
    if not isinstance(artifact_hashes, dict):
        raise SourceValidationError(
            "Natural source recovery lacks AdaFloor request artifact hashes"
        )
    recorded_artifacts: dict[Path, str] = {}
    for raw_path, raw_sha256 in artifact_hashes.items():
        if not isinstance(raw_path, str):
            raise SourceValidationError(
                "Natural source recovery has an invalid artifact path"
            )
        artifact_path = Path(raw_path).resolve()
        recorded_artifacts[artifact_path] = _require_sha256(
            raw_sha256, f"recovery artifact {artifact_path} SHA256"
        )
    if set(recorded_artifacts) != expected_artifacts:
        raise SourceValidationError(
            "Natural source recovery artifact set is not the five-step contract"
        )
    for artifact_path, expected_sha256 in recorded_artifacts.items():
        if _sha256(artifact_path) != expected_sha256:
            raise SourceValidationError(
                f"Natural source request artifact changed: {artifact_path}"
            )

    return {
        "schema_version": 1,
        "status": "PASS",
        "source_natural_root": str(root),
        "source_execution_code_sha256": expected_execution,
        "postvalidation_recovery": str(recovery_path.resolve()),
        "postvalidation_recovery_sha256": expected_recovery,
        "pair_summary": str(summary_path.resolve()),
        "pair_summary_sha256": _sha256(summary_path),
        "cleanup_transaction": str(transaction_path.resolve()),
        "cleanup_transaction_sha256": _sha256(transaction_path),
        "cap_env": str(cap_env),
        "cap_env_sha256": _sha256(cap_env),
        "common_root": str(common_root),
        "vanilla_run": str(vanilla_run),
        "adafloor_run": str(adafloor_run),
        "adafloor_epoch": str(adafloor_epoch),
        "adafloor_actual_plan": str(actual_plan.resolve()),
        "adafloor_actual_plan_sha256": _sha256(actual_plan),
    }


def _write_new_or_identical(path: Path, payload: dict[str, Any]) -> None:
    raw = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != raw:
            raise SourceValidationError(f"existing source validation is stale: {path}")
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-natural-root", type=Path, required=True)
    parser.add_argument("--expected-execution-sha256", required=True)
    parser.add_argument("--expected-recovery-sha256", required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        payload = validate_source(
            args.source_natural_root,
            args.expected_execution_sha256,
            args.expected_recovery_sha256,
        )
        if args.output:
            _write_new_or_identical(args.output.resolve(), payload)
    except SourceValidationError as error:
        parser.error(str(error))
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
