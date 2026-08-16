from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path


ROOT = Path(__file__).parents[1]
INTERNAL = ROOT / "internal"


def _source(
    profile: Path,
    *names: str,
    env: dict[str, str] | None = None,
) -> list[str]:
    command = (
        "source \"$1\"; shift; "
        "for name in \"$@\"; do printf '%s\\n' \"${!name}\"; done"
    )
    result = subprocess.run(
        ["bash", "-c", command, "bash", str(profile), *names],
        cwd=ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.splitlines()


def _closure_digest(files: str) -> str:
    command = [
        "python3",
        str(ROOT / "tools/hash_deepseek_runtime_profile.py"),
        "--root",
        str(ROOT),
    ]
    for path in files.split(","):
        command.extend(("--profile", path))
    return subprocess.check_output(command, cwd=ROOT, text=True).strip()


def test_planned_floor4_profile_enables_residency_and_disables_sidecars() -> None:
    profile = INTERNAL / "deepseek_v2_lite_planned_f4_runtime_profile.sh"
    names = (
        "VLLM_ASCEND_SHRINK_AWARE_TARGET_POLICY",
        "VLLM_ASCEND_ELASTIC_MIN_COMPUTE_GROUP_SIZE",
        "VLLM_ASCEND_SHRINK_AWARE_STAGES",
        "VLLM_ASCEND_SHRINK_AWARE_FINAL_RANKS",
        "VLLM_ASCEND_MODE1_PARITY_FIXED_TOPOLOGY_REUSE",
        "VLLM_ASCEND_MODE1_PARITY_PRECREATE_PLANNED_FLOOR_GROUPS",
        "VLLM_ASCEND_MODE1_PARITY_CACHE_PLANNED_FLOOR_GROUPS",
        "VLLM_ASCEND_MODE1_PARITY_PRECREATE_COMM_CACHE",
        "VLLM_ASCEND_MODE1_PARITY_PRECREATE_DISPATCH_WARMUP",
        "VLLM_ASCEND_MODE1_PARITY_PREFILL_PLANNED_EXPERT_SLOTS",
        "VLLM_ASCEND_MODE1_TRAINING_MEMORY_GUARD",
        "VLLM_ASCEND_MODE1_TRAINING_MEMORY_GUARD_STRICT",
        "VERL_SIDECAR_ENABLE",
        "VLLM_ASCEND_ENABLE_DRAFT_TRAIN",
    )
    assert _source(profile, *names) == [
        "planned",
        "4",
        "8,4",
        "12,13,14,15",
        "1",
        "1",
        "1",
        "1",
        "1",
        "1",
        "1",
        "1",
        "0",
        "0",
    ]


def test_planned_floor2_profile_enables_every_stage() -> None:
    profile = INTERNAL / "deepseek_v2_lite_planned_f2_runtime_profile.sh"
    names = (
        "VLLM_ASCEND_SHRINK_AWARE_TARGET_POLICY",
        "VLLM_ASCEND_ELASTIC_MIN_COMPUTE_GROUP_SIZE",
        "VLLM_ASCEND_SHRINK_AWARE_STAGES",
        "VLLM_ASCEND_SHRINK_AWARE_FINAL_RANKS",
        "VLLM_ASCEND_MODE1_PARITY_DEFER_GROUP_DESTROY_ON_PRE_REBUILD",
        "VLLM_ASCEND_MODE1_PARITY_DEFER_GROUP_DESTROY_ON_STASH",
    )
    assert _source(profile, *names) == [
        "planned",
        "2",
        "8,4,2",
        "14,15",
        "1",
        "1",
    ]


def test_planned_profile_closures_are_ordered_and_content_bound() -> None:
    f4_profile = INTERNAL / "deepseek_v2_lite_planned_f4_runtime_profile.sh"
    f2_profile = INTERNAL / "deepseek_v2_lite_planned_f2_runtime_profile.sh"
    f4_files = _source(f4_profile, "DEEPSEEK_P_F4_RUNTIME_PROFILE_FILES")[0]
    f2_files = _source(f2_profile, "DEEPSEEK_P_F2_RUNTIME_PROFILE_FILES")[0]
    assert f4_files.split(",") == [
        "internal/deepseek_v2_lite_natural_f4_runtime_profile.sh",
        "internal/deepseek_v2_lite_planned_f4_runtime_profile.sh",
    ]
    assert f2_files.split(",") == [
        "internal/deepseek_v2_lite_natural_f4_runtime_profile.sh",
        "internal/deepseek_v2_lite_planned_f4_runtime_profile.sh",
        "internal/deepseek_v2_lite_planned_f2_runtime_profile.sh",
    ]
    for files in (f4_files, f2_files):
        digest = _closure_digest(files)
        assert len(digest) == 64
        int(digest, 16)
    assert _closure_digest(f4_files) != _closure_digest(f2_files)


def test_planned_profiles_preserve_explicit_measured_values() -> None:
    profile = INTERNAL / "deepseek_v2_lite_planned_f2_runtime_profile.sh"
    env = {
        "PATH": "/usr/bin:/bin",
        "DEEPSEEK_P_F2_HEADROOM_FLOOR2": "128",
        "DEEPSEEK_P_F2_HEADROOM_FLOOR4": "256",
        "DEEPSEEK_P_F2_HEADROOM_FLOOR8": "384",
        "DEEPSEEK_P_F2_HEADROOM_FLOOR16": "0",
        "DEEPSEEK_P_F2_TRAINING_MIN_FREE_MIB": "4096",
    }
    names = (
        "VLLM_ASCEND_MODE1_PARITY_PLANNED_FLOOR_GROUP_KV_HEADROOM_TOKENS_FLOOR2",
        "VLLM_ASCEND_MODE1_PARITY_PLANNED_FLOOR_GROUP_KV_HEADROOM_TOKENS_FLOOR4",
        "VLLM_ASCEND_MODE1_PARITY_PLANNED_FLOOR_GROUP_KV_HEADROOM_TOKENS_FLOOR8",
        "VLLM_ASCEND_MODE1_PARITY_PLANNED_FLOOR_GROUP_KV_HEADROOM_TOKENS_FLOOR16",
        "VLLM_ASCEND_MODE1_TRAINING_MIN_FREE_MIB",
    )
    assert _source(profile, *names, env=env) == [
        "128",
        "256",
        "384",
        "0",
        "4096",
    ]


def test_planned_calibration_wrappers_are_fail_closed_and_merge_canonical_caps() -> None:
    cases = {
        "f4": ("16 8 4", (4, 8, 16)),
        "f2": ("16 8 4 2", (2, 4, 8, 16)),
    }
    for suffix, (floor_loop, floors) in cases.items():
        path = ROOT / f"run_deepseek_v2_lite_planned_{suffix}_calibration.sh"
        text = path.read_text(encoding="utf-8")
        prefix = f"DEEPSEEK_P_{suffix.upper()}"
        assert f'for floor in {floor_loop}; do' in text
        assert f'--lifecycle planned_{suffix}' in text
        assert f'run_deepseek_v2_lite_kv_probe.sh" planned_{suffix}' in text
        assert f"{prefix}_TRAINING_MIN_FREE_MIB must be an explicitly measured" in text
        assert "hash_deepseek_runtime_profile.py" in text
        assert (
            f"CAP_ENV=${{{prefix}_KV_CAP_ENV:-"
            "${DEEPSEEK_KV_CAP_ENV:-$SCRIPT_DIR/deepseek_v2_lite_kv_caps.env}}"
        ) in text
        assert '--output "$CAP_ENV"' in text
        assert "--merge-existing" in text
        assert ".candidate.env" not in text
        assert ".adafloor_npu_exclusive.lock" in text
        assert "flock -n 9" in text
        assert "ALLOW_INFEASIBLE_PLAN" not in text
        assert '--expected-plan-response-cap "$PROBE_EXPECTED_PLAN_RESPONSE_CAP"' in text
        assert '--training-min-free-mib "$' + prefix + '_TRAINING_MIN_FREE_MIB"' in text
        for floor in floors:
            assert f"--planned-headroom-floor{floor}" in text


def test_profile_closure_digest_is_not_plain_leaf_digest() -> None:
    profile = INTERNAL / "deepseek_v2_lite_planned_f2_runtime_profile.sh"
    files = _source(profile, "DEEPSEEK_P_F2_RUNTIME_PROFILE_FILES")[0]
    closure = _closure_digest(files)
    leaf = hashlib.sha256(profile.read_bytes()).hexdigest()
    assert closure != leaf


def test_deepseek_planned_profile_rejects_unmeasured_training_guard() -> None:
    profile = INTERNAL / "deepseek_v2_lite_planned_f2_runtime_profile.sh"
    driver = ROOT / "run_mode1_dynamic_length_aware_adaptive_floor4_epochs.sh"
    result = subprocess.run(
        [
            "bash",
            "-c",
            'source "$1"; exec "$2"',
            "bash",
            str(profile),
            str(driver),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "requires an explicitly measured positive training HBM reserve" in result.stderr
