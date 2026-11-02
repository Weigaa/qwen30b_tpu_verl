from __future__ import annotations

import hashlib
import os
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "run_deepseek_v2_lite_fair_compare.sh"
PROFILE = ROOT / "internal" / "deepseek_v2_lite_batch64_workload_profile.sh"


def _profile_sha256() -> str:
    return hashlib.sha256(PROFILE.read_bytes()).hexdigest()


def _write_batch64_caps(
    tmp_path: Path,
    *,
    profile_id: str = "deepseek-v2-lite-chat-b64-n16-s5-v2",
) -> tuple[Path, Path]:
    common = tmp_path / "common"
    cap_env = tmp_path / "caps.env"
    values = {
        "DEEPSEEK_KV_CAP_MODEL_REVISION": (
            "85864749cd611b4353ce1decdb286193298f64c7"
        ),
        "DEEPSEEK_KV_CAP_EXECUTION_PROFILE": (
            "deepseek-v2-lite_tq1_a2aFalse_sharedFalse_deallocFalse_"
            "recomputeuniformx1_hccl800"
        ),
        "DEEPSEEK_KV_CAP_TRAIN_BATCH_SIZE": "64",
        "DEEPSEEK_KV_CAP_PROMPTS_PER_RANK": "4",
        "DEEPSEEK_KV_CAP_COMMON_STEPS": "5",
        "DEEPSEEK_KV_CAP_PROMPTS_TOTAL": "320",
        "DEEPSEEK_KV_CAP_EXPECTED_RESPONSES_PER_STEP": "1024",
        "DEEPSEEK_KV_CAP_DATASET_FRACTION": "0.01",
        "DEEPSEEK_KV_CAP_MAX_PROMPT_LENGTH": "1024",
        "DEEPSEEK_KV_CAP_MAX_RESPONSE_LENGTH": "16384",
        "DEEPSEEK_KV_CAP_MAX_NUM_BATCHED_TOKENS": "17408",
        "DEEPSEEK_KV_CAP_MAX_NUM_SEQS": "64",
        "DEEPSEEK_KV_CAP_GPU_MEMORY_UTILIZATION": "0.9",
        "DEEPSEEK_KV_CAP_ENFORCE_EAGER": "True",
        "DEEPSEEK_KV_CAP_BLOCK_SIZE": "128",
        "DEEPSEEK_KV_CAP_ROLLOUT_N": "16",
        "DEEPSEEK_KV_CAP_TARGET_RATIO": "1.0",
        "DEEPSEEK_KV_CAP_COMMON_EPOCH0_ROOT": str(common),
        "DEEPSEEK_KV_CAP_WORKLOAD_PROFILE_ID": profile_id,
        "DEEPSEEK_KV_CAP_WORKLOAD_PROFILE_SHA256": _profile_sha256(),
        "DEEPSEEK_KV_CAP_COMMON_PREEMPTION_POLICY": "record",
        "DEEPSEEK_VANILLA_KV_PHYSICAL_TOKENS": "380800",
        "DEEPSEEK_VANILLA_KV_ADMISSION_TOKENS": "360064",
    }
    cap_env.write_text(
        "\n".join(f"export {name}={value}" for name, value in values.items())
        + "\n",
        encoding="utf-8",
    )
    return common, cap_env


def _run(
    common: Path,
    cap_env: Path,
    *,
    prompts: int,
    expected_steps: int | None = None,
) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env.update(
        {
            "COMMON_EPOCH0_ROOT": str(common),
            "DEEPSEEK_KV_CAP_ENV": str(cap_env),
            "DEEPSEEK_WORKLOAD_PROFILE_PATH": str(PROFILE),
            "DEEPSEEK_VALIDATE_ASSETS_ON_LAUNCH": "0",
            "FAIR_PROMPTS_PER_EPOCH": str(prompts),
            "DEEPSEEK_FAIR_DATASET_FRACTION": (
                "0.001765" if prompts == 64 else "0.01"
            ),
        }
    )
    if expected_steps is not None:
        env["FAIR_EXPECTED_STEPS"] = str(expected_steps)
    return subprocess.run(
        ["bash", str(SCRIPT), "vanilla"],
        cwd=ROOT,
        env=env,
        check=False,
        text=True,
        capture_output=True,
    )


def test_batch32_defaults_remain_unset_only_defaults() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert "${COMMON_EPOCH0_TRAIN_BATCH_SIZE:-32}" in source
    assert "${COMMON_EPOCH0_ROLLOUT_N:-16}" in source
    assert "${COMMON_EPOCH0_PROMPTS_TOTAL:-160}" in source
    assert "${COMMON_EPOCH0_MAX_NUM_SEQS:-32}" in source
    assert '[[ -z "${FAIR_FREEZE_ACTOR+x}" ]]' in source
    assert '[[ -z "${DYNAMIC_DATASET_FRACTION+x}" ]]' in source
    assert (
        "DYNAMIC_SHORT_STEP_CAP_ENABLE=${DYNAMIC_SHORT_STEP_CAP_ENABLE:-1}"
        in source
    )
    assert "export DYNAMIC_SHORT_STEP_CAP_ENABLE=0" not in source


@pytest.mark.parametrize(("prompts", "steps"), [(64, 1), (320, 5)])
def test_batch64_accepts_gate_and_full_epoch_shapes(
    tmp_path: Path, prompts: int, steps: int
) -> None:
    common, cap_env = _write_batch64_caps(tmp_path)
    result = _run(common, cap_env, prompts=prompts, expected_steps=steps)

    assert result.returncode == 2
    assert "common epoch0 is not complete" in result.stderr
    assert "workload profile" not in result.stderr
    assert "FAIR_EXPECTED_STEPS" not in result.stderr


def test_batch64_rejects_non_gate_partial_epoch(tmp_path: Path) -> None:
    common, cap_env = _write_batch64_caps(tmp_path)
    result = _run(common, cap_env, prompts=128, expected_steps=2)

    assert result.returncode == 2
    assert "either one gate step or the full workload profile" in result.stderr


def test_batch64_rejects_inherited_expected_step_mismatch(tmp_path: Path) -> None:
    common, cap_env = _write_batch64_caps(tmp_path)
    result = _run(common, cap_env, prompts=64, expected_steps=5)

    assert result.returncode == 2
    assert (
        "FAIR_EXPECTED_STEPS must equal prompts_per_epoch / train_batch_size"
        in result.stderr
    )


def test_batch64_rejects_cap_profile_provenance_mismatch(tmp_path: Path) -> None:
    common, cap_env = _write_batch64_caps(tmp_path, profile_id="wrong-profile")
    result = _run(common, cap_env, prompts=64, expected_steps=1)

    assert result.returncode == 2
    assert "KV cap workload profile provenance mismatch" in result.stderr


def test_audit_uses_dynamic_expected_steps() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert '--expected-steps "$FAIR_EXPECTED_STEPS"' in source
    assert "--expected-steps 5" not in source
