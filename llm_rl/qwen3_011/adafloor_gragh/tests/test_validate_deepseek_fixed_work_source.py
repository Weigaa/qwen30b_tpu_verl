from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from tools.validate_deepseek_fixed_work_source import (
    ADAFLOOR_RUN_NAME,
    VANILLA_RUN_NAME,
    SourceValidationError,
    validate_source,
)


EXECUTION_SHA256 = "b" * 64


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def _hashed(path: Path) -> dict[str, str]:
    return {"path": str(path.resolve()), "sha256": _sha256(path)}


def _source_fixture(tmp_path: Path) -> Path:
    root = tmp_path / "natural_epoch"
    common_root = tmp_path / "common_epoch0"
    common_root.mkdir()
    cap_env = tmp_path / "caps.env"
    cap_env.write_text("export CAP=1\n", encoding="utf-8")
    vanilla_run = root / VANILLA_RUN_NAME
    adafloor_run = root / ADAFLOOR_RUN_NAME
    for run in (vanilla_run, adafloor_run):
        run.mkdir(parents=True)
        (run / "CHECKPOINTS_REMOVED_AFTER_VALIDATION.txt").write_text(
            "removed\n", encoding="utf-8"
        )
    epoch = adafloor_run / "epoch_001_mode1_natural"
    (epoch / "rollout_data").mkdir(parents=True)
    (epoch / "rollout_length").mkdir()
    (epoch / "oracle").mkdir()
    actual_plan = epoch / "oracle" / "length_sorted_rank_plan.json"
    actual_plan.write_text(
        "[]\n", encoding="utf-8"
    )
    artifacts: list[Path] = []
    for step in range(1, 6):
        rollout_path = epoch / "rollout_data" / f"{step}.jsonl"
        length_path = epoch / "rollout_length" / f"length_{step}.txt"
        rollout_path.write_text(f'{{"step":{step}}}\n', encoding="utf-8")
        length_path.write_text(f"{step}\n", encoding="utf-8")
        artifacts.extend((rollout_path, length_path))
    audit_path = adafloor_run / "DEEPSEEK_PLAN_RUNTIME_AUDIT.json"
    _write_json(
        audit_path,
        {
            "status": "PASS",
            "lifecycle": "natural_f2",
            "run_root": str(adafloor_run.resolve()),
        },
    )
    recovery_path = adafloor_run / "POSTVALIDATION_RECOVERY.json"
    _write_json(
        recovery_path,
        {
            "status": "PASS",
            "execution_code_sha256": EXECUTION_SHA256,
            "cap_env_sha256": _sha256(cap_env),
            "formal_audit_sha256": _sha256(audit_path),
            "plan_sha256": {str(actual_plan.resolve()): _sha256(actual_plan)},
            "request_audit": {
                "artifact_sha256": {
                    "adafloor": {
                        str(path.resolve()): _sha256(path) for path in artifacts
                    }
                }
            },
        },
    )
    summary_path = root / "natural_epoch_summary.json"
    _write_json(
        summary_path,
        {
            "status": "PASS",
            "phase": "epoch",
            "provenance": {
                "execution_code_sha256": EXECUTION_SHA256,
                "common_root": str(common_root.resolve()),
                "cap_env": str(cap_env.resolve()),
                "cap_env_sha256": _sha256(cap_env),
            },
        },
    )
    transaction_path = root / "PAIR_VERIFIED_CHECKPOINT_CLEANUP.json"
    _write_json(
        transaction_path,
        {
            "status": "COMMITTED",
            "phase": "epoch",
            "mode": "natural",
            "execution_code_sha256": EXECUTION_SHA256,
            "pair_summary": _hashed(summary_path),
            "adafloor_plan_runtime_audit": _hashed(audit_path),
            "cap_env": _hashed(cap_env),
            "arms": {
                "vanilla": {"run_root": str(vanilla_run.resolve())},
                "adafloor": {"run_root": str(adafloor_run.resolve())},
            },
            "cleanup_markers": {
                "vanilla": _hashed(
                    vanilla_run / "CHECKPOINTS_REMOVED_AFTER_VALIDATION.txt"
                ),
                "adafloor": _hashed(
                    adafloor_run / "CHECKPOINTS_REMOVED_AFTER_VALIDATION.txt"
                ),
            },
        },
    )
    return root


def test_validate_source_accepts_committed_immutable_natural_pair(
    tmp_path: Path,
) -> None:
    root = _source_fixture(tmp_path)

    recovery = root / ADAFLOOR_RUN_NAME / "POSTVALIDATION_RECOVERY.json"
    result = validate_source(root, EXECUTION_SHA256, _sha256(recovery))

    assert result["status"] == "PASS"
    assert result["source_execution_code_sha256"] == EXECUTION_SHA256
    assert result["source_natural_root"] == str(root.resolve())
    assert result["adafloor_actual_plan_sha256"] == _sha256(
        Path(result["adafloor_actual_plan"])
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("status", "PENDING"),
        ("mode", "fixed"),
        ("execution_code_sha256", "c" * 64),
    ],
)
def test_validate_source_rejects_uncommitted_or_wrong_transaction(
    tmp_path: Path, field: str, value: str
) -> None:
    root = _source_fixture(tmp_path)
    transaction_path = root / "PAIR_VERIFIED_CHECKPOINT_CLEANUP.json"
    transaction = json.loads(transaction_path.read_text(encoding="utf-8"))
    transaction[field] = value
    _write_json(transaction_path, transaction)

    with pytest.raises(SourceValidationError, match="not a committed epoch pair"):
        recovery = root / ADAFLOOR_RUN_NAME / "POSTVALIDATION_RECOVERY.json"
        validate_source(root, EXECUTION_SHA256, _sha256(recovery))


def test_validate_source_rejects_changed_cleanup_marker(tmp_path: Path) -> None:
    root = _source_fixture(tmp_path)
    marker = root / VANILLA_RUN_NAME / "CHECKPOINTS_REMOVED_AFTER_VALIDATION.txt"
    marker.write_text("changed\n", encoding="utf-8")

    with pytest.raises(SourceValidationError, match="changed after pair verification"):
        recovery = root / ADAFLOOR_RUN_NAME / "POSTVALIDATION_RECOVERY.json"
        validate_source(root, EXECUTION_SHA256, _sha256(recovery))


def test_validate_source_rejects_wrong_arm_root(tmp_path: Path) -> None:
    root = _source_fixture(tmp_path)
    transaction_path = root / "PAIR_VERIFIED_CHECKPOINT_CLEANUP.json"
    transaction = json.loads(transaction_path.read_text(encoding="utf-8"))
    transaction["arms"]["adafloor"]["run_root"] = str(tmp_path / "other")
    _write_json(transaction_path, transaction)

    with pytest.raises(SourceValidationError, match="wrong adafloor root"):
        recovery = root / ADAFLOOR_RUN_NAME / "POSTVALIDATION_RECOVERY.json"
        validate_source(root, EXECUTION_SHA256, _sha256(recovery))


def test_validate_source_rejects_changed_request_artifact(tmp_path: Path) -> None:
    root = _source_fixture(tmp_path)
    recovery = root / ADAFLOOR_RUN_NAME / "POSTVALIDATION_RECOVERY.json"
    artifact = (
        root
        / ADAFLOOR_RUN_NAME
        / "epoch_001_mode1_natural"
        / "rollout_data"
        / "3.jsonl"
    )
    artifact.write_text('{"changed":true}\n', encoding="utf-8")

    with pytest.raises(SourceValidationError, match="request artifact changed"):
        validate_source(root, EXECUTION_SHA256, _sha256(recovery))


def test_validate_source_rejects_unpinned_recovery(tmp_path: Path) -> None:
    root = _source_fixture(tmp_path)

    with pytest.raises(SourceValidationError, match="recovery SHA256 differs"):
        validate_source(root, EXECUTION_SHA256, "c" * 64)
