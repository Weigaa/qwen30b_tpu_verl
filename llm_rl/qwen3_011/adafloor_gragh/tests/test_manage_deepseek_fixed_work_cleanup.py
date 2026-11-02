from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.manage_deepseek_fixed_work_cleanup import (
    REMOVED_MARKER,
    RETAINED_MARKER,
    TransactionError,
    build_payload,
    commit,
    prepare,
    transaction_allows_removed_arm,
)


EXECUTION_SHA256 = "a" * 64


def _json(path: Path, **values: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(values, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _fixture(tmp_path: Path) -> tuple[dict[str, object], Path, Path]:
    summary = _json(tmp_path / "summary.json", status="PASS", phase="gate")
    audit = _json(
        tmp_path / "adafloor" / "DEEPSEEK_PLAN_RUNTIME_AUDIT.json",
        status="PASS",
        lifecycle="natural_f2",
    )
    cap_env = tmp_path / "caps.env"
    cap_env.write_text("export CAP=1\n", encoding="utf-8")
    vanilla = tmp_path / "vanilla"
    adafloor = tmp_path / "adafloor"
    for run in (vanilla, adafloor):
        (run / "epoch_001" / "checkpoints").mkdir(parents=True)
        (run / RETAINED_MARKER).write_text("validated\n", encoding="utf-8")
    payload = build_payload(
        phase="gate",
        mode="fixed",
        summary=summary,
        allowed_summary_statuses=("PASS",),
        cap_env=cap_env,
        execution_code_sha256=EXECUTION_SHA256,
        vanilla_run=vanilla,
        adafloor_run=adafloor,
        adafloor_audit=audit,
    )
    return payload, vanilla, adafloor


def test_cleanup_requires_pair_verification_before_checkpoint_removal(
    tmp_path: Path,
) -> None:
    payload, vanilla, adafloor = _fixture(tmp_path)
    pending = tmp_path / ".pending.json"
    committed = tmp_path / "committed.json"
    prepare(pending, committed, payload)
    assert json.loads(pending.read_text(encoding="utf-8"))["status"] == "PAIR_VERIFIED"

    for arm, run in (("vanilla", vanilla), ("adafloor", adafloor)):
        transaction_allows_removed_arm(
            pending, committed, arm, run, EXECUTION_SHA256
        )
        (run / RETAINED_MARKER).unlink()
        (run / "epoch_001" / "checkpoints").rmdir()
        (run / REMOVED_MARKER).write_text(f"arm={arm}\n", encoding="utf-8")

    result = commit(pending, committed)
    assert result["status"] == "COMMITTED"
    assert not pending.exists()
    assert committed.is_file()
    assert commit(pending, committed) == result


def test_removed_checkpoint_marker_without_transaction_is_rejected(
    tmp_path: Path,
) -> None:
    _payload, vanilla, _adafloor = _fixture(tmp_path)
    with pytest.raises(TransactionError, match="no pair-verification transaction"):
        transaction_allows_removed_arm(
            tmp_path / ".pending.json",
            tmp_path / "committed.json",
            "vanilla",
            vanilla,
            EXECUTION_SHA256,
        )


def test_prepare_rejects_cleanup_that_precedes_pair_verification(tmp_path: Path) -> None:
    payload, vanilla, _adafloor = _fixture(tmp_path)
    (vanilla / RETAINED_MARKER).unlink()
    (vanilla / REMOVED_MARKER).write_text("removed\n", encoding="utf-8")
    with pytest.raises(TransactionError, match="lacks retained-checkpoint"):
        prepare(tmp_path / ".pending.json", tmp_path / "committed.json", payload)


def test_transaction_rejects_changed_summary_or_audit(tmp_path: Path) -> None:
    payload, vanilla, _adafloor = _fixture(tmp_path)
    pending = tmp_path / ".pending.json"
    committed = tmp_path / "committed.json"
    prepare(pending, committed, payload)
    Path(payload["pair_summary"]["path"]).write_text(
        '{"status":"FAIL"}\n', encoding="utf-8"
    )
    with pytest.raises(TransactionError, match="input changed"):
        transaction_allows_removed_arm(
            pending, committed, "vanilla", vanilla, EXECUTION_SHA256
        )


@pytest.mark.parametrize(
    ("summary_status", "audit_status", "lifecycle"),
    [("FAIL", "PASS", "natural_f2"), ("PASS", "FAIL", "natural_f2"), ("PASS", "PASS", "planned_f2")],
)
def test_payload_requires_passing_pair_and_natural_floor2_audit(
    tmp_path: Path,
    summary_status: str,
    audit_status: str,
    lifecycle: str,
) -> None:
    summary = _json(tmp_path / "summary.json", status=summary_status)
    audit = _json(tmp_path / "audit.json", status=audit_status, lifecycle=lifecycle)
    cap = tmp_path / "cap.env"
    cap.write_text("cap\n", encoding="utf-8")
    vanilla = tmp_path / "vanilla"
    adafloor = tmp_path / "adafloor"
    vanilla.mkdir()
    adafloor.mkdir()
    with pytest.raises(TransactionError):
        build_payload(
            phase="gate",
            mode="fixed",
            summary=summary,
            allowed_summary_statuses=("PASS",),
            cap_env=cap,
            execution_code_sha256=EXECUTION_SHA256,
            vanilla_run=vanilla,
            adafloor_run=adafloor,
            adafloor_audit=audit,
        )
