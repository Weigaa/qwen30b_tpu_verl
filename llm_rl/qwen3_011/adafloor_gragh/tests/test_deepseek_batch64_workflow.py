from __future__ import annotations

import json
import re
import subprocess
import sys
from copy import deepcopy
from pathlib import Path


ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "run_deepseek_v2_lite_batch64_paired.sh"
MODEL_ID = "deepseek-ai/DeepSeek-V2-Lite-Chat"
MODEL_REVISION = "85864749cd611b4353ce1decdb286193298f64c7"
MODEL_PATH = "/data/DeepSeek-V2-Lite-Chat"
DISTCP_PATH = "/data/DeepSeek-V2-Lite-Chat_megatron_pp4_ep4"


def _embedded_python(label: str) -> str:
    source = SCRIPT.read_text(encoding="utf-8")
    prefix = f"<<'{label}'\n"
    assert source.count(prefix) == 1
    body = source.split(prefix, 1)[1]
    code, suffix, _ = body.partition(f"\n{label}\n")
    assert suffix
    return code


def _run_embedded_python(label: str, *args: Path | str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-", *(str(arg) for arg in args)],
        input=_embedded_python(label),
        text=True,
        capture_output=True,
        check=False,
    )


def _shell_function(name: str) -> str:
    source = SCRIPT.read_text(encoding="utf-8")
    matches = list(
        re.finditer(rf"^{re.escape(name)}\(\) \{{\n", source, flags=re.MULTILINE)
    )
    assert len(matches) == 1
    body = source[matches[0].end() :]
    function, suffix, _ = body.partition("\n}\n")
    assert suffix
    return function


def _valid_semantic_output() -> dict:
    return {
        "model": MODEL_PATH,
        "load_format": "auto",
        "expert_parallel_size": 1,
        "max_tokens": 1024,
        "passed": True,
        "records": [
            {
                "label": "math_chat_primary",
                "semantic_smoke_pass": True,
                "answer_quality_pass": True,
                "finish_reason": "stop",
                "dialogue_continuation": False,
                "prompt_bos_count": 1,
            },
            {
                "label": "math_chat_secondary",
                "semantic_smoke_pass": True,
                "answer_quality_pass": True,
                "finish_reason": "stop",
                "dialogue_continuation": False,
                "prompt_bos_count": 1,
            },
        ],
    }


def _valid_asset_audit() -> dict:
    return {
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "model_path": MODEL_PATH,
        "distcp_path": DISTCP_PATH,
        "pipeline_model_parallel_size": 4,
        "expert_model_parallel_size": 4,
    }


def _with_value(document: dict, path: tuple[str | int, ...], value: object) -> dict:
    changed = deepcopy(document)
    cursor = changed
    for component in path[:-1]:
        cursor = cursor[component]
    cursor[path[-1]] = value
    return changed


def _validate_semantic_fixture(tmp_path: Path, semantic: dict, audit: dict) -> int:
    semantic_path = tmp_path / "semantic.json"
    audit_path = tmp_path / "audit.json"
    semantic_path.write_text(json.dumps(semantic), encoding="utf-8")
    audit_path.write_text(json.dumps(audit), encoding="utf-8")
    result = _run_embedded_python(
        "PY_SEMANTIC_OUTPUT_VALIDATE",
        semantic_path,
        audit_path,
        MODEL_ID,
        MODEL_REVISION,
        MODEL_PATH,
        DISTCP_PATH,
    )
    return result.returncode


def test_semantic_gate_accepts_only_the_pinned_output_and_assets(tmp_path: Path) -> None:
    semantic = _valid_semantic_output()
    audit = _valid_asset_audit()
    assert _validate_semantic_fixture(tmp_path, semantic, audit) == 0

    invalid_semantic = [
        _with_value(semantic, ("model",), "/data/wrong-model"),
        _with_value(semantic, ("load_format",), "safetensors"),
        _with_value(semantic, ("expert_parallel_size",), 2),
        _with_value(semantic, ("max_tokens",), 512),
        _with_value(semantic, ("passed",), False),
        _with_value(semantic, ("records",), semantic["records"][:1]),
        _with_value(semantic, ("records", 1, "label"), "math_chat_primary"),
        _with_value(semantic, ("records", 0, "semantic_smoke_pass"), False),
        _with_value(semantic, ("records", 0, "answer_quality_pass"), False),
        _with_value(semantic, ("records", 0, "finish_reason"), "length"),
        _with_value(semantic, ("records", 0, "dialogue_continuation"), True),
        _with_value(semantic, ("records", 0, "prompt_bos_count"), 2),
    ]
    for candidate in invalid_semantic:
        assert _validate_semantic_fixture(tmp_path, candidate, audit) != 0

    invalid_audits = [
        _with_value(audit, ("model_id",), "deepseek-ai/DeepSeek-V2-Lite"),
        _with_value(audit, ("model_revision",), "wrong-revision"),
        _with_value(audit, ("model_path",), "/data/wrong-model"),
        _with_value(audit, ("distcp_path",), "/data/wrong-distcp"),
        _with_value(audit, ("pipeline_model_parallel_size",), 2),
        _with_value(audit, ("expert_model_parallel_size",), 8),
    ]
    for candidate in invalid_audits:
        assert _validate_semantic_fixture(tmp_path, semantic, candidate) != 0


def test_semantic_complete_marker_requires_the_pinned_identity(tmp_path: Path) -> None:
    marker = tmp_path / "COMPLETE"
    expected_lines = [
        "COMPLETE DeepSeek-V2-Lite-Chat semantic gate",
        f"MODEL_ID={MODEL_ID}",
        f"MODEL_REVISION={MODEL_REVISION}",
        f"MODEL_PATH={MODEL_PATH}",
        f"DISTCP_PATH={DISTCP_PATH}",
    ]

    def validate(lines: list[str]) -> int:
        marker.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return _run_embedded_python(
            "PY_SEMANTIC_MARKER_VALIDATE",
            marker,
            MODEL_ID,
            MODEL_REVISION,
            MODEL_PATH,
            DISTCP_PATH,
        ).returncode

    assert validate(expected_lines) == 0
    for index in range(len(expected_lines)):
        changed = expected_lines.copy()
        changed[index] += ".stale"
        assert validate(changed) != 0
    assert validate([*expected_lines, "UNEXPECTED=stale"]) != 0


def test_converted_weight_marker_requires_the_pinned_assets(tmp_path: Path) -> None:
    marker = tmp_path / "COMPLETE"
    values = {
        "TASK_QUEUE_ENABLE": "2",
        "RECOMPUTE_METHOD": "uniform",
        "RECOMPUTE_NUM_LAYERS": "1",
        "TRAINING_STEPS": "1",
        "TRAIN_BATCH_SIZE": "32",
        "MAX_PROMPT_LENGTH": "1024",
        "MAX_RESPONSE_LENGTH": "32",
        "TAIL_VALIDATE_LEVEL_TOKENS_BY_STEP": "<unset>",
        "ROLLOUT_N": "1",
        "ROLLOUT_MAX_NUM_BATCHED_TOKENS": "1056",
        "ROLLOUT_MAX_NUM_SEQS": "32",
        "EXPECTED_ROWS": "32",
        "REQUIRE_SEMANTIC_OUTPUT": "0",
        "ROLLOUT_LOAD_FORMAT": "auto",
        "PRESERVE_INITIAL_HF_WEIGHTS": "0",
        "COMPARE_ONLINE_SYNC_TO_HF": "1",
        "MODEL_ID": MODEL_ID,
        "MODEL_REVISION": MODEL_REVISION,
        "MODEL_PATH": MODEL_PATH,
        "DISTCP_PATH": DISTCP_PATH,
    }

    def validate(candidate: dict[str, str]) -> int:
        lines = ["COMPLETE DeepSeek actor update probe"]
        lines.extend(f"{key}={value}" for key, value in candidate.items())
        marker.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return _run_embedded_python(
            "PY_CONVERTED_GATE_VALIDATE",
            marker,
            MODEL_ID,
            MODEL_REVISION,
            MODEL_PATH,
            DISTCP_PATH,
        ).returncode

    assert validate(values) == 0
    for key in ("MODEL_ID", "MODEL_REVISION", "MODEL_PATH", "DISTCP_PATH"):
        changed = values.copy()
        changed[key] += ".stale"
        assert validate(changed) != 0


def test_batch64_workflow_has_fail_closed_stages() -> None:
    source = SCRIPT.read_text(encoding="utf-8")

    for phase in (
        "common",
        "semantic",
        "recover-common",
        "trigger",
        "calibrate",
        "authorize",
        "gate",
        "epoch",
        "verify-gate",
        "verify-epoch",
    ):
        assert phase in source
    assert "EXECUTION_CODE_SHA256" in source
    assert "WORKLOAD_PROFILE_SHA256" in source
    assert "incomplete batch64 common epoch0 exists" in source
    assert "incomplete batch64 authorization exists" in source
    assert "COMMON_EPOCH0_KV_TOKENS_PER_RANK=auto" in source
    assert "DEEPSEEK_SHARED_FULL16_PHYSICAL_TOKENS" in source
    assert "DEEPSEEK_N_F2_KV_CAPS_VERIFIED=1" in source
    assert "FAIR_FREEZE_ACTOR=1" in source
    assert "FAIR_PROMPTS_PER_EPOCH=\"$prompts\"" in source
    assert "FAIR_EXPECTED_STEPS=\"$steps\"" in source
    assert "run_deepseek_v2_lite_fair_compare.sh" in source
    assert "verify_deepseek_batch64_pair.py" in source
    assert "DEEPSEEK_BATCH64_MIGRATE_POSTPROCESS_CODE" in source
    assert "DEEPSEEK_BATCH64_EXPECTED_OLD_EXECUTION_CODE_SHA256" in source
    assert "COMMON_EPOCH0_ROLLOUT_EXECUTION_CODE_SHA256" in source
    assert "CONTINUATION_EXECUTION_CODE_SHA256" in source
    assert "DEEPSEEK_BATCH64_MIGRATE_PRE_PAIR_CODE" in source
    assert "pre_pair_response_mask_and_natural_audit_correction" in source
    assert "DEEPSEEK_KV_CAP_VALIDATION_RESUME_AUDIT=1" in source
    assert "continuation execution code changed after migration" in source
    assert "postprocessing code migration is forbidden after downstream work starts" in source
    assert "common epoch0 recovery is forbidden after downstream work starts" in source
    assert "audit_deepseek_common_epoch0.py" in source
    assert "DeepSeek-V2-Lite-Chat semantic gate" in source
    assert "run_deepseek_v2_lite_semantic_smoke.sh" in source
    assert "run_deepseek_v2_lite_weight_compare_smoke.sh" in source
    assert "converted_weight_gate_complete" in source
    assert '[[ -f "$marker" ]] || return 1' in source
    assert "semantic_output_passes || return 1" in source
    assert "converted_weight_gate_complete || return 1" in source
    assert "assert_execution_code_unchanged" in source
    assert "refusing to rewrite a stale pair manifest" in source
    assert "batch64 epoch requires a completed one-step paired gate" in source
    assert "--expected-execution-code-sha256" in source
    assert "85864749cd611b4353ce1decdb286193298f64c7" in source
    assert "--expected-unique-prompts 317" in source
    assert "--expected-duplicate-occurrences 3" in source
    assert "--expected-distcp-count 32" in source
    assert "hash_deepseek_checkpoint.py" in source
    assert "FROZEN_CHECKPOINT_SHA256" in source


def test_deepseek_response_mask_uses_raw_vllm_lengths() -> None:
    source = (
        ROOT
        / "verl"
        / "workers"
        / "rollout"
        / "vllm_rollout"
        / "vllm_rollout_spmd.py"
    ).read_text(encoding="utf-8")

    assert "decoded_response_lengths.append(len(response_ids))" in source
    assert "< decoded_lengths.unsqueeze(1)" in source
    assert 'non_tensor_batch["decoded_response_length"]' in source
    assert 'non_tensor_batch["response_finish_reason"]' in source


def test_epoch_execution_and_verification_require_the_same_verified_gate() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    gate = _shell_function("require_verified_gate")
    run_pair = _shell_function("run_pair")
    verify_epoch = re.search(
        r"^    verify-epoch\)\n(?P<body>.*?)^        ;;$",
        source,
        flags=re.MULTILINE | re.DOTALL,
    )

    assert '[[ ! -f "$GATE_ROOT/paired_gate_summary.json" ]]' in gate
    assert "verify_pair gate" in gate
    assert "require_verified_gate" in run_pair
    assert source.count("require_verified_gate") == 3
    assert verify_epoch is not None
    verify_epoch_body = verify_epoch.group("body")
    assert (
        verify_epoch_body.index("load_pair_contract")
        < verify_epoch_body.index("require_verified_gate")
        < verify_epoch_body.index("verify_pair epoch")
    )


def test_calibration_and_authorization_guard_the_frozen_checkpoint() -> None:
    verify_digest = _shell_function("verify_checkpoint_digest")
    calibration = _shell_function("run_calibration")
    authorization = _shell_function("run_authorization")

    assert 'source "$COMMON_ROOT/reuse.env"' in verify_digest
    assert "stage_expected" in verify_digest
    assert calibration.count("verify_checkpoint_digest") == 2
    assert authorization.count("verify_checkpoint_digest") == 2
    calibration_first = calibration.index("verify_checkpoint_digest")
    calibration_last = calibration.rindex("verify_checkpoint_digest")
    assert (
        calibration_first
        < calibration.index('if [[ -f "$CALIBRATION_ROOT/COMPLETE"')
        < calibration.index("run_deepseek_v2_lite_natural_f2_calibration.sh")
        < calibration_last
    )
    assert 'verify_checkpoint_digest "$frozen_checkpoint_sha256"' in calibration
    authorization_first = authorization.index("verify_checkpoint_digest")
    authorization_last = authorization.rindex("verify_checkpoint_digest")
    assert (
        authorization_first
        < authorization.index('if [[ -f "$AUTHORIZATION_ROOT/COMPLETE"')
        < authorization.index("run_deepseek_v2_lite_kv_cap_validation.sh")
        < authorization_last
    )
    assert 'verify_checkpoint_digest "$frozen_checkpoint_sha256"' in authorization


def test_batch64_workflow_writes_complete_pair_manifest() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    required = {
        "DEEPSEEK_BATCH64_ARM",
        "DEEPSEEK_BATCH64_PHASE",
        "DEEPSEEK_WORKLOAD_PROFILE_ID",
        "DEEPSEEK_WORKLOAD_PROFILE_SHA256",
        "DEEPSEEK_BATCH64_COMMON_ROOT",
        "DEEPSEEK_BATCH64_FROZEN_CHECKPOINT",
        "DEEPSEEK_BATCH64_MODEL_PATH",
        "DEEPSEEK_BATCH64_MODEL_REVISION",
        "DEEPSEEK_BATCH64_EXECUTION_PROFILE",
        "DEEPSEEK_BATCH64_CAP_ENV_SHA256",
        "DEEPSEEK_BATCH64_EXECUTION_CODE_SHA256",
        "DEEPSEEK_BATCH64_FROZEN_CHECKPOINT_SHA256",
        "DEEPSEEK_BATCH64_PAIRED_REQUEST_SAMPLING_SEEDS",
        "DEEPSEEK_BATCH64_TRAIN_BATCH_SIZE",
        "DEEPSEEK_BATCH64_ROLLOUT_N",
        "DEEPSEEK_BATCH64_MAX_NUM_SEQS",
        "DEEPSEEK_BATCH64_MAX_PROMPT_LENGTH",
        "DEEPSEEK_BATCH64_MAX_RESPONSE_LENGTH",
        "DEEPSEEK_BATCH64_MAX_NUM_BATCHED_TOKENS",
        "DEEPSEEK_BATCH64_FULL16_PHYSICAL_TOKENS",
        "DEEPSEEK_BATCH64_TEMPERATURE",
        "DEEPSEEK_BATCH64_TOP_P",
        "DEEPSEEK_BATCH64_TOP_K",
        "DEEPSEEK_BATCH64_DATASET_FRACTION",
        "DEEPSEEK_BATCH64_FORCED_SELECTED_FLOOR",
    }
    for name in required:
        assert name in source


def test_batch64_gate_forces_only_adafloor_to_floor4() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    run_arm = _shell_function("run_pair_arm")
    dry_run = _shell_function("dry_run_pair")

    assert "GATE_ADAFLOOR_FORCE_SELECTED_FLOOR=4" in source
    assert '[[ "$phase" == gate && "$arm" == adafloor ]]' in run_arm
    assert "DEEPSEEK_FAIR_FORCE_SELECTED_FLOOR=" in run_arm
    assert '[[ "$phase" == gate && "$variant" == adafloor_n_f2 ]]' in dry_run
    assert "DEEPSEEK_FAIR_FORCE_SELECTED_FLOOR=" in dry_run
    assert "DEEPSEEK_FAIR_DATASET_FRACTION=" in run_arm
    assert "DEEPSEEK_FAIR_DATASET_FRACTION=" in dry_run
    assert 'printf \'%s\' "$DEEPSEEK_KV_PROBE_DATASET_FRACTION"' in source
    assert 'printf \'%s\' "$COMMON_EPOCH0_DATASET_FRACTION"' in source


def test_batch64_workflow_shell_syntax_and_status() -> None:
    subprocess.run(["bash", "-n", str(SCRIPT)], cwd=ROOT, check=True)
    result = subprocess.run(
        [str(SCRIPT), "status"],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    assert "common=" in result.stdout
    assert "semantic=" in result.stdout
    assert "authorization=" in result.stdout
    assert "gate=" in result.stdout
    assert "epoch=" in result.stdout
