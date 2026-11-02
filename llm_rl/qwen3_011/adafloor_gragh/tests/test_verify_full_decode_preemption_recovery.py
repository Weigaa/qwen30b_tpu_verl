from pathlib import Path

import pytest

from tools.verify_full_decode_preemption_recovery import AuditError, audit_log


def write_log(path: Path, *, missing_resume_rank: int | None = None) -> Path:
    lines = ["Graph capturing finished", "Replaying aclgraph"]
    for step in range(5):
        lines.append(f"rollout_output_time_s: {step + 1}.0")
        lines.append("response/aborted_ratio:0.0")
    lines.append("Preempting request 7 for request 8")
    for rank in range(16):
        lines.append(
            "FULL_DECODE_ONLY DP rank entered synchronized eager recovery: "
            f"dp_rank={rank} num_tokens=64 global_with_prefill=1"
        )
    lines.append(
        "FULL_DECODE_ONLY local preemption recovery joined the "
        "DP-synchronized eager iteration: dp_rank=7 num_tokens=64 "
        "global_with_prefill=1"
    )
    for rank in range(16):
        if rank != missing_resume_rank:
            lines.append(
                "FULL_DECODE_ONLY DP rank resumed FULL graph after synchronized "
                f"eager recovery: dp_rank={rank} num_tokens=32"
            )
    lines.append("After trainer.fit")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_accepts_complete_world_recovery(tmp_path: Path) -> None:
    result = audit_log(write_log(tmp_path / "run.log"), 16, 5)
    assert result["status"] == "PASS"
    assert result["kv_preemption_events"] == 1
    assert result["synchronized_recovery_rounds"] == 1
    assert result["enter_events"] == result["full_resume_events"] == 16


def test_rejects_missing_rank_resume(tmp_path: Path) -> None:
    with pytest.raises(AuditError, match="without FULL resume"):
        audit_log(write_log(tmp_path / "run.log", missing_resume_rank=11), 16, 5)


def test_rejects_preemption_without_local_join(tmp_path: Path) -> None:
    log = write_log(tmp_path / "run.log")
    log.write_text(
        "\n".join(
            line for line in log.read_text(encoding="utf-8").splitlines()
            if "local preemption recovery joined" not in line
        ) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(AuditError, match="no local preemption recovery join"):
        audit_log(log, 16, 5)
