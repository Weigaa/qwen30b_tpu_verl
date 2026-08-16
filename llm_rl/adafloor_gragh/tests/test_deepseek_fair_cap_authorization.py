from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "run_deepseek_v2_lite_fair_compare.sh"
RUNTIME_PROFILE = ROOT / "internal" / "deepseek_v2_lite_natural_f4_runtime_profile.sh"
N_F2_RUNTIME_PROFILE = (
    ROOT / "internal" / "deepseek_v2_lite_natural_f2_runtime_profile.sh"
)
P_F4_RUNTIME_PROFILE = (
    ROOT / "internal" / "deepseek_v2_lite_planned_f4_runtime_profile.sh"
)
P_F2_RUNTIME_PROFILE = (
    ROOT / "internal" / "deepseek_v2_lite_planned_f2_runtime_profile.sh"
)


def _profile_id(path: Path, key: str) -> str:
    match = re.search(
        rf"^export {re.escape(key)}=([^\s]+)$",
        path.read_text(encoding="utf-8"),
        re.M,
    )
    assert match is not None
    return match.group(1)


def _profile_closure_sha256(*paths: Path) -> str:
    command = [
        sys.executable,
        str(ROOT / "tools" / "hash_deepseek_runtime_profile.py"),
        "--root",
        str(ROOT),
    ]
    for path in paths:
        command.extend(("--profile", str(path)))
    return subprocess.check_output(command, cwd=ROOT, text=True).strip()


RUNTIME_PROFILE_ID = _profile_id(
    RUNTIME_PROFILE, "DEEPSEEK_N_F4_RUNTIME_PROFILE_ID"
)
RUNTIME_PROFILE_SHA256 = _profile_closure_sha256(RUNTIME_PROFILE)
N_F2_RUNTIME_PROFILE_ID = _profile_id(
    N_F2_RUNTIME_PROFILE, "DEEPSEEK_N_F2_RUNTIME_PROFILE_ID"
)
N_F2_RUNTIME_PROFILE_SHA256 = _profile_closure_sha256(
    RUNTIME_PROFILE, N_F2_RUNTIME_PROFILE
)
P_F4_RUNTIME_PROFILE_ID = _profile_id(
    P_F4_RUNTIME_PROFILE, "DEEPSEEK_P_F4_RUNTIME_PROFILE_ID"
)
P_F4_RUNTIME_PROFILE_SHA256 = _profile_closure_sha256(
    RUNTIME_PROFILE, P_F4_RUNTIME_PROFILE
)
P_F2_RUNTIME_PROFILE_ID = _profile_id(
    P_F2_RUNTIME_PROFILE, "DEEPSEEK_P_F2_RUNTIME_PROFILE_ID"
)
P_F2_RUNTIME_PROFILE_SHA256 = _profile_closure_sha256(
    RUNTIME_PROFILE, P_F4_RUNTIME_PROFILE, P_F2_RUNTIME_PROFILE
)
EXECUTION_CODE_SHA256 = subprocess.check_output(
    [
        sys.executable,
        str(ROOT / "tools" / "hash_deepseek_execution_code.py"),
        "--root",
        str(ROOT),
    ],
    text=True,
).strip()


def _write_n_f2_cap_env(
    tmp_path: Path, *, overrides: dict[str, str] | None = None
) -> tuple[Path, Path]:
    common = tmp_path / "common"
    values = {
        "DEEPSEEK_N_F2_KV_CAPS_VERIFIED": "1",
        "DEEPSEEK_KV_CAP_MODEL_REVISION": (
            "85864749cd611b4353ce1decdb286193298f64c7"
        ),
        "DEEPSEEK_KV_CAP_EXECUTION_PROFILE": (
            "deepseek-v2-lite_tq1_a2aFalse_sharedFalse_deallocFalse_"
            "recomputeuniformx1_hccl800"
        ),
        "DEEPSEEK_N_F2_RUNTIME_PROFILE": N_F2_RUNTIME_PROFILE_ID,
        "DEEPSEEK_N_F2_RUNTIME_PROFILE_SHA256": N_F2_RUNTIME_PROFILE_SHA256,
        "DEEPSEEK_EXECUTION_CODE_SHA256": EXECUTION_CODE_SHA256,
        "DEEPSEEK_KV_CAP_MAX_PROMPT_LENGTH": "1024",
        "DEEPSEEK_KV_CAP_MAX_RESPONSE_LENGTH": "16384",
        "DEEPSEEK_KV_CAP_MAX_NUM_BATCHED_TOKENS": "17408",
        "DEEPSEEK_KV_CAP_MAX_NUM_SEQS": "32",
        "DEEPSEEK_KV_CAP_GPU_MEMORY_UTILIZATION": "0.9",
        "DEEPSEEK_KV_CAP_ENFORCE_EAGER": "True",
        "DEEPSEEK_KV_CAP_BLOCK_SIZE": "128",
        "DEEPSEEK_KV_CAP_ROLLOUT_N": "16",
        "DEEPSEEK_KV_CAP_TARGET_RATIO": "1.0",
        "DEEPSEEK_KV_CAP_COMMON_EPOCH0_ROOT": str(common),
        "DEEPSEEK_N_F2_KV_ADMISSION_FLOOR2": "300032",
        "DEEPSEEK_N_F2_KV_ADMISSION_FLOOR4": "400000",
        "DEEPSEEK_N_F2_KV_ADMISSION_FLOOR8": "500096",
        "DEEPSEEK_N_F2_KV_ADMISSION_FLOOR16": "600064",
        "DEEPSEEK_N_F2_KV_PHYSICAL_FLOOR2": "309760",
        "DEEPSEEK_N_F2_KV_PHYSICAL_FLOOR4": "409728",
        "DEEPSEEK_N_F2_KV_PHYSICAL_FLOOR8": "509824",
        "DEEPSEEK_N_F2_KV_PHYSICAL_FLOOR16": "609792",
    }
    values.update(overrides or {})
    cap_env = tmp_path / "caps.env"
    cap_env.write_text(
        "\n".join(f"export {name}={value}" for name, value in values.items())
        + "\n",
        encoding="utf-8",
    )
    return common, cap_env


def _fair_env(common: Path, cap_env: Path) -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "COMMON_EPOCH0_ROOT": str(common),
            "DEEPSEEK_KV_CAP_ENV": str(cap_env),
            "DEEPSEEK_VALIDATE_ASSETS_ON_LAUNCH": "0",
        }
    )
    return env


def test_fair_wrapper_pins_formal_protocol_against_inherited_environment() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert "export FAIR_START_EPOCH=1" in source
    assert "export FAIR_TOTAL_EPOCHS=2" in source
    assert "export FAIR_TOTAL_EPOCHS=3" in source
    assert "export FAIR_FREEZE_ACTOR=0" in source
    assert "export DYNAMIC_DATASET_FRACTION=0.005" in source
    assert "export TRAIN_FILE_ORIG=/data/deepscaler/train.parquet" in source
    assert "export TEST_FILE=/data/deepscaler/test.parquet" in source
    assert "export LOCAL_TEST_LAUNCHER=$SCRIPT_DIR/internal/wj_train_grpo_deepseek_v2_lite_16die_adafloor.sh" in source
    assert "export VLLM_ASCEND_REGISTER_CUSTOM_MODELS=1" in source
    assert "export VERL_PAIRED_REQUEST_SAMPLING_SEEDS=1" in source
    assert "export MAX_CROSS_STEP_REPAIR_SWAPS=8" in source
    assert "export REPAIR_CANDIDATE_LIMIT=8" in source


def test_natural_floor4_profile_pins_expert_reload_and_export_paths() -> None:
    profile = RUNTIME_PROFILE.read_text(encoding="utf-8")
    expected = {
        "VLLM_ASCEND_CUSTOM_MODE1_STRICT": "1",
        "VLLM_ASCEND_MODE1_ENFORCE_TARGET_FLOOR_ON_WEIGHT_RELOAD": "1",
        "VLLM_ASCEND_MODE1_RELOAD_TRIM_ALLOCATOR": "0",
        "VLLM_ASCEND_MODE1_SKIP_PRE_WEIGHT_KV_RESUME": "1",
        "VLLM_ASCEND_MODE1_RELOAD_FORCE_FORMAT_CAST": "0",
        "VLLM_ASCEND_MODE1_RELOAD_SKIP_SAME_FORMAT_CAST": "1",
        "VLLM_ASCEND_MODE1_PRIMARY_RELOAD_MAP_AFTER_RESTORE": "0",
        "VLLM_ASCEND_MODE1_EMPTY_CACHE_BETWEEN_COMPACT_TENSORS": "1",
        "VLLM_ASCEND_MODE1_VALIDATE_EXPERT_MAPPING": "1",
        "VLLM_ASCEND_MODE1_VALIDATE_EXPERT_MAPPING_STRICT": "1",
        "VLLM_ASCEND_MODE1_PARITY_ALLTOALL_HOST_METADATA": "1",
        "VLLM_ASCEND_MODE1_USE_CPU_EXPORT_SLOT_MAP": "1",
        "VLLM_ASCEND_MODE1_SAFE_CPU_EXPORT": "0",
        "VLLM_ASCEND_MODE1_ALLOW_NPU_EXPORT_SLOT_FALLBACK": "0",
        "VLLM_ASCEND_MODE1_ALLOW_CONTIGUOUS_P2P_ALIAS_EXPORT": "1",
        "VLLM_ASCEND_MODE1_CACHE_P2P_ALIASES": "1",
        "VLLM_ASCEND_SHRINK_AWARE_TARGET_POLICY": "natural",
        "VLLM_ASCEND_ELASTIC_MIN_COMPUTE_GROUP_SIZE": "4",
        "VLLM_ASCEND_SHRINK_AWARE_STAGES": "8,4",
        "VLLM_ASCEND_SHRINK_AWARE_INTERMEDIATE_RANKS": "8,9,10,11,12,13,14,15",
        "VLLM_ASCEND_SHRINK_AWARE_FINAL_RANKS": "12,13,14,15",
        "VLLM_ROLLOUT_EARLY_STOP_ENABLE": "0",
        "VERL_SIDECAR_ENABLE": "0",
    }
    for name, value in expected.items():
        assert f"export {name}={value}" in profile


@pytest.mark.parametrize(
    ("variant", "message"),
    [
        ("fixed4", "Natural floor4 KV caps are not verified"),
        ("adafloor_n_f2", "Natural floor2 KV caps are not verified"),
        ("adafloor_p_f4", "Planned floor4 KV caps are not verified"),
        ("adafloor_p_f2", "Planned floor2 KV caps are not verified"),
    ],
)
def test_global_flag_does_not_authorize_unverified_lifecycle(
    tmp_path: Path, variant: str, message: str
) -> None:
    common = tmp_path / "common"
    cap_env = tmp_path / "caps.env"
    cap_env.write_text(
        "\n".join(
            [
                "export DEEPSEEK_KV_CAPS_VERIFIED=1",
                "export DEEPSEEK_N_F2_KV_CAPS_VERIFIED=0",
                "export DEEPSEEK_N_F4_KV_CAPS_VERIFIED=0",
                "export DEEPSEEK_P_F4_KV_CAPS_VERIFIED=0",
                "export DEEPSEEK_P_F2_KV_CAPS_VERIFIED=0",
                "export DEEPSEEK_KV_CAP_MODEL_REVISION=85864749cd611b4353ce1decdb286193298f64c7",
                "export DEEPSEEK_KV_CAP_EXECUTION_PROFILE=deepseek-v2-lite_tq1_a2aFalse_sharedFalse_deallocFalse_recomputeuniformx1_hccl800",
                f"export DEEPSEEK_N_F4_RUNTIME_PROFILE={RUNTIME_PROFILE_ID}",
                f"export DEEPSEEK_N_F4_RUNTIME_PROFILE_SHA256={RUNTIME_PROFILE_SHA256}",
                f"export DEEPSEEK_N_F2_RUNTIME_PROFILE={N_F2_RUNTIME_PROFILE_ID}",
                f"export DEEPSEEK_N_F2_RUNTIME_PROFILE_SHA256={N_F2_RUNTIME_PROFILE_SHA256}",
                f"export DEEPSEEK_P_F4_RUNTIME_PROFILE={P_F4_RUNTIME_PROFILE_ID}",
                f"export DEEPSEEK_P_F4_RUNTIME_PROFILE_SHA256={P_F4_RUNTIME_PROFILE_SHA256}",
                f"export DEEPSEEK_P_F2_RUNTIME_PROFILE={P_F2_RUNTIME_PROFILE_ID}",
                f"export DEEPSEEK_P_F2_RUNTIME_PROFILE_SHA256={P_F2_RUNTIME_PROFILE_SHA256}",
                f"export DEEPSEEK_EXECUTION_CODE_SHA256={EXECUTION_CODE_SHA256}",
                "export DEEPSEEK_KV_CAP_MAX_PROMPT_LENGTH=1024",
                "export DEEPSEEK_KV_CAP_MAX_RESPONSE_LENGTH=16384",
                "export DEEPSEEK_KV_CAP_MAX_NUM_BATCHED_TOKENS=17408",
                "export DEEPSEEK_KV_CAP_MAX_NUM_SEQS=32",
                "export DEEPSEEK_KV_CAP_GPU_MEMORY_UTILIZATION=0.9",
                "export DEEPSEEK_KV_CAP_ENFORCE_EAGER=True",
                "export DEEPSEEK_KV_CAP_BLOCK_SIZE=128",
                "export DEEPSEEK_KV_CAP_ROLLOUT_N=16",
                "export DEEPSEEK_KV_CAP_TARGET_RATIO=1.0",
                f"export DEEPSEEK_KV_CAP_COMMON_EPOCH0_ROOT={common}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    env = os.environ.copy()
    env.update(
        {
            "COMMON_EPOCH0_ROOT": str(common),
            "DEEPSEEK_KV_CAP_ENV": str(cap_env),
            "DEEPSEEK_VALIDATE_ASSETS_ON_LAUNCH": "0",
        }
    )

    result = subprocess.run(
        ["bash", str(SCRIPT), variant],
        cwd=ROOT,
        env=env,
        check=False,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 2
    assert message in result.stderr


def test_rejects_natural_floor4_runtime_profile_mismatch(tmp_path: Path) -> None:
    common = tmp_path / "common"
    cap_env = tmp_path / "caps.env"
    cap_env.write_text(
        "\n".join(
            [
                "export DEEPSEEK_N_F4_KV_CAPS_VERIFIED=0",
                "export DEEPSEEK_KV_CAP_MODEL_REVISION=85864749cd611b4353ce1decdb286193298f64c7",
                "export DEEPSEEK_KV_CAP_EXECUTION_PROFILE=deepseek-v2-lite_tq1_a2aFalse_sharedFalse_deallocFalse_recomputeuniformx1_hccl800",
                "export DEEPSEEK_N_F4_RUNTIME_PROFILE=wrong-profile",
                f"export DEEPSEEK_N_F4_RUNTIME_PROFILE_SHA256={RUNTIME_PROFILE_SHA256}",
                f"export DEEPSEEK_EXECUTION_CODE_SHA256={EXECUTION_CODE_SHA256}",
                "export DEEPSEEK_KV_CAP_MAX_PROMPT_LENGTH=1024",
                "export DEEPSEEK_KV_CAP_MAX_RESPONSE_LENGTH=16384",
                "export DEEPSEEK_KV_CAP_MAX_NUM_BATCHED_TOKENS=17408",
                "export DEEPSEEK_KV_CAP_MAX_NUM_SEQS=32",
                "export DEEPSEEK_KV_CAP_GPU_MEMORY_UTILIZATION=0.9",
                "export DEEPSEEK_KV_CAP_ENFORCE_EAGER=True",
                "export DEEPSEEK_KV_CAP_BLOCK_SIZE=128",
                "export DEEPSEEK_KV_CAP_ROLLOUT_N=16",
                "export DEEPSEEK_KV_CAP_TARGET_RATIO=1.0",
                f"export DEEPSEEK_KV_CAP_COMMON_EPOCH0_ROOT={common}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    env = os.environ.copy()
    env.update(
        {
            "COMMON_EPOCH0_ROOT": str(common),
            "DEEPSEEK_KV_CAP_ENV": str(cap_env),
            "DEEPSEEK_VALIDATE_ASSETS_ON_LAUNCH": "0",
        }
    )

    result = subprocess.run(
        ["bash", str(SCRIPT), "fixed4"],
        cwd=ROOT,
        env=env,
        check=False,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 2
    assert "runtime profile does not match" in result.stderr


def test_rejects_natural_floor4_runtime_profile_sha256_mismatch(
    tmp_path: Path,
) -> None:
    common = tmp_path / "common"
    cap_env = tmp_path / "caps.env"
    cap_env.write_text(
        "\n".join(
            [
                "export DEEPSEEK_N_F4_KV_CAPS_VERIFIED=0",
                "export DEEPSEEK_KV_CAP_MODEL_REVISION=85864749cd611b4353ce1decdb286193298f64c7",
                "export DEEPSEEK_KV_CAP_EXECUTION_PROFILE=deepseek-v2-lite_tq1_a2aFalse_sharedFalse_deallocFalse_recomputeuniformx1_hccl800",
                f"export DEEPSEEK_N_F4_RUNTIME_PROFILE={RUNTIME_PROFILE_ID}",
                f"export DEEPSEEK_N_F4_RUNTIME_PROFILE_SHA256={'0' * 64}",
                f"export DEEPSEEK_EXECUTION_CODE_SHA256={EXECUTION_CODE_SHA256}",
                "export DEEPSEEK_KV_CAP_MAX_PROMPT_LENGTH=1024",
                "export DEEPSEEK_KV_CAP_MAX_RESPONSE_LENGTH=16384",
                "export DEEPSEEK_KV_CAP_MAX_NUM_BATCHED_TOKENS=17408",
                "export DEEPSEEK_KV_CAP_MAX_NUM_SEQS=32",
                "export DEEPSEEK_KV_CAP_GPU_MEMORY_UTILIZATION=0.9",
                "export DEEPSEEK_KV_CAP_ENFORCE_EAGER=True",
                "export DEEPSEEK_KV_CAP_BLOCK_SIZE=128",
                "export DEEPSEEK_KV_CAP_ROLLOUT_N=16",
                "export DEEPSEEK_KV_CAP_TARGET_RATIO=1.0",
                f"export DEEPSEEK_KV_CAP_COMMON_EPOCH0_ROOT={common}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    env = os.environ.copy()
    env.update(
        {
            "COMMON_EPOCH0_ROOT": str(common),
            "DEEPSEEK_KV_CAP_ENV": str(cap_env),
            "DEEPSEEK_VALIDATE_ASSETS_ON_LAUNCH": "0",
        }
    )

    result = subprocess.run(
        ["bash", str(SCRIPT), "fixed4"],
        cwd=ROOT,
        env=env,
        check=False,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 2
    assert "runtime profile SHA256 does not match" in result.stderr


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        (
            {"DEEPSEEK_N_F2_RUNTIME_PROFILE": "wrong-profile"},
            "Natural floor2 runtime profile does not match its KV probes",
        ),
        (
            {"DEEPSEEK_N_F2_RUNTIME_PROFILE_SHA256": "0" * 64},
            "Natural floor2 runtime profile SHA256 does not match its KV probes",
        ),
        (
            {"DEEPSEEK_EXECUTION_CODE_SHA256": "0" * 64},
            "DeepSeek execution code SHA256 does not match its KV probes",
        ),
    ],
)
def test_rejects_natural_floor2_provenance_mismatch(
    tmp_path: Path, overrides: dict[str, str], message: str
) -> None:
    common, cap_env = _write_n_f2_cap_env(tmp_path, overrides=overrides)

    result = subprocess.run(
        ["bash", str(SCRIPT), "adafloor_n_f2"],
        cwd=ROOT,
        env=_fair_env(common, cap_env),
        check=False,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 2
    assert message in result.stderr


def test_natural_floor2_dry_run_pins_calibrated_runtime_path(
    tmp_path: Path,
) -> None:
    common, cap_env = _write_n_f2_cap_env(tmp_path)
    baseline = common / "epoch_000_mode0_probe"
    checkpoint = common / "global_step_5"
    (baseline / "rollout_data").mkdir(parents=True)
    (checkpoint / "actor").mkdir(parents=True)
    (checkpoint / ".PRESERVE_COMMON_EPOCH0").touch()
    (common / "DO_NOT_DELETE_COMMON_EPOCH0_CHECKPOINT").touch()
    (common / "reuse.env").write_text(
        "\n".join(
            [
                f"export DYNAMIC_INITIAL_BASELINE_DIR={baseline}",
                f"export DYNAMIC_INITIAL_RESUME_CKPT={checkpoint}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (common / "common_epoch0_metadata.env").write_text(
        "\n".join(
            [
                "export COMMON_EPOCH0_MODEL_PATH=/data/DeepSeek-V2-Lite-Chat",
                "export COMMON_EPOCH0_MODEL_REVISION=85864749cd611b4353ce1decdb286193298f64c7",
                "export COMMON_EPOCH0_DISTCP_PATH=/data/DeepSeek-V2-Lite-Chat_megatron_pp4_ep4",
                "export COMMON_EPOCH0_CHECKPOINT_MODEL_DIR_NAME=deepseek_v2_lite_chat",
                "export COMMON_EPOCH0_EXECUTION_PROFILE_USED=deepseek-v2-lite_tq1_a2aFalse_sharedFalse_deallocFalse_recomputeuniformx1_hccl800",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    env = _fair_env(common, cap_env)
    env.update(
        {
            "DEEPSEEK_FAIR_DRY_RUN": "1",
            "FAIR_OUTPUT_ROOT": str(tmp_path / "output"),
        }
    )

    result = subprocess.run(
        ["bash", str(SCRIPT), "adafloor_n_f2"],
        cwd=ROOT,
        env=env,
        check=False,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stderr
    assert (
        "Natural floor2 runtime_profile="
        f"{N_F2_RUNTIME_PROFILE_ID} "
        f"child={ROOT / 'run_mode1_local_length_sorted_e2e_adaptive_floor4.sh'} "
        "stages=8,4,2 final_ranks=14,15"
    ) in result.stdout
    assert "[fair rerun] dry run only" in result.stdout


def test_gate_force_is_scoped_to_one_step_natural_floor2() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    run_one_start = source.index("run_one() (")
    floor4_start = source.index("fixed4|adafloor_n_f4)", run_one_start)
    floor2_start = source.index(
        "minskew|adafloor_n_f2|adafloor_n_f2_noguard)", floor4_start
    )
    planned_start = source.index("adafloor_p_f4|adafloor_p_f2)", floor2_start)
    natural_floor4 = source[floor4_start:floor2_start]
    natural_floor2 = source[floor2_start:planned_start]

    assert "fair_force_selected_floor=${DEEPSEEK_FAIR_FORCE_SELECTED_FLOOR:-}" in source
    assert '"$variant" != adafloor_n_f2' in source
    assert '"$FAIR_EXPECTED_STEPS" != 1' in source
    assignment = "export DYNAMIC_FORCE_SELECTED_FLOOR=$fair_force_selected_floor"
    assert assignment not in natural_floor4
    assert assignment in natural_floor2


def test_fair_compare_uses_one_dataset_fraction_for_planner_and_vanilla() -> None:
    source = SCRIPT.read_text(encoding="utf-8")

    assert "fair_dataset_fraction=${DEEPSEEK_FAIR_DATASET_FRACTION:-}" in source
    assert "export FAIR_DATASET_FRACTION=$DYNAMIC_DATASET_FRACTION" in source
    assert "expected_fair_dataset_fraction=${DEEPSEEK_KV_PROBE_DATASET_FRACTION:-}" in source
    assert "expected_fair_dataset_fraction=$COMMON_EPOCH0_DATASET_FRACTION" in source
