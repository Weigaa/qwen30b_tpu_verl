from __future__ import annotations

import json
import hashlib
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).parents[1]


def test_common_epoch0_keeps_batch32_defaults_and_accepts_profile_values() -> None:
    runner = (
        ROOT / "run_common_epoch0_probe_gpu09_kv380800_permanent.sh"
    ).read_text(encoding="utf-8")

    assert 'TRAIN_STEPS="${COMMON_EPOCH0_TRAIN_STEPS:-5}"' in runner
    assert 'TRAIN_BATCH_SIZE="${COMMON_EPOCH0_TRAIN_BATCH_SIZE:-32}"' in runner
    assert 'ROLLOUT_N="${COMMON_EPOCH0_ROLLOUT_N:-16}"' in runner
    assert 'MAX_NUM_SEQS="${COMMON_EPOCH0_MAX_NUM_SEQS:-32}"' in runner
    assert 'DATASET_FRACTION="${COMMON_EPOCH0_DATASET_FRACTION:-0.005}"' in runner
    assert 'PREEMPTION_POLICY="${COMMON_EPOCH0_PREEMPTION_POLICY:-forbid}"' in runner
    assert 'EXPECTED_RESPONSES_PER_STEP=$((TRAIN_BATCH_SIZE * ROLLOUT_N))' in runner
    assert 'CHECKPOINT_PATH="$EPOCH_DIR/checkpoints/$CHECKPOINT_MODEL_DIR_NAME/global_step_$TRAIN_STEPS"' in runner
    assert 'TRAIN_BATCH_SIZE="$TRAIN_BATCH_SIZE"' in runner
    assert 'ROLLOUT_MAX_NUM_SEQS="$MAX_NUM_SEQS"' in runner
    assert 'ROLLOUT_N="$ROLLOUT_N"' in runner
    assert "preemption_policy == \"forbid\"" in runner
    assert "COMMON_EPOCH0_PROMPTS_TOTAL_USED" in runner
    assert "COMMON_EPOCH0_EXPECTED_RESPONSES_PER_STEP_USED" in runner
    assert "COMMON_EPOCH0_WORKLOAD_PROFILE_ID" in runner
    assert "COMMON_EPOCH0_WORKLOAD_PROFILE_SHA256" in runner
    assert "COMMON_EPOCH0_FINALIZE_EXISTING" in runner
    assert "COMMON_EPOCH0_ORIGINAL_EXECUTION_CODE_SHA256" in runner
    assert "COMMON_EPOCH0_EFFECTIVE_KV_TOKENS_PER_RANK" in runner
    assert "validating and finalizing existing training output" in runner
    assert 'history_overwrite_args=(--force)' in runner


def test_deepseek_batch64_workload_profile_contract() -> None:
    profile = ROOT / "internal" / "deepseek_v2_lite_batch64_workload_profile.sh"
    script = r'''
source "$1"
printf '%s\n' \
  "$DEEPSEEK_WORKLOAD_PROFILE_ID" \
  "$COMMON_EPOCH0_TRAIN_STEPS" \
  "$COMMON_EPOCH0_TRAIN_BATCH_SIZE" \
  "$COMMON_EPOCH0_ROLLOUT_N" \
  "$COMMON_EPOCH0_MAX_NUM_SEQS" \
  "$COMMON_EPOCH0_DATASET_FRACTION" \
  "$DEEPSEEK_KV_PROBE_DATASET_FRACTION" \
  "$COMMON_EPOCH0_PROMPTS_TOTAL" \
  "$COMMON_EPOCH0_EXPECTED_RESPONSES_PER_STEP" \
  "$COMMON_EPOCH0_PREEMPTION_POLICY"
'''
    result = subprocess.run(
        ["bash", "-c", script, "bash", str(profile)],
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stdout.splitlines() == [
        "deepseek-v2-lite-chat-b64-n16-s5-v2",
        "5",
        "64",
        "16",
        "64",
        "0.01",
        "0.001765",
        "320",
        "1024",
        "record",
    ]


def test_deepseek_common_wrapper_hashes_explicit_workload_profile() -> None:
    wrapper = (ROOT / "run_deepseek_v2_lite_common_epoch0.sh").read_text(
        encoding="utf-8"
    )

    assert "DEEPSEEK_WORKLOAD_PROFILE_PATH" in wrapper
    assert 'source "$WORKLOAD_PROFILE_PATH"' in wrapper
    assert 'sha256sum "$WORKLOAD_PROFILE_PATH"' in wrapper
    assert "COMMON_EPOCH0_WORKLOAD_PROFILE_ID" in wrapper
    assert "COMMON_EPOCH0_WORKLOAD_PROFILE_SHA256" in wrapper
    assert "DEEPSEEK_KV_CAP_WORKLOAD_PROFILE_ID" in wrapper
    assert "DEEPSEEK_KV_CAP_WORKLOAD_PROFILE_SHA256" in wrapper
    assert "cap_matches_workload" in wrapper
    assert "export COMMON_EPOCH0_DATASET_FRACTION=0.005" in wrapper


def test_batch64_profile_provenance_reaches_calibration_probe_and_authorization() -> None:
    calibration = (
        ROOT / "run_deepseek_v2_lite_natural_f2_calibration.sh"
    ).read_text(encoding="utf-8")
    probe = (ROOT / "run_deepseek_v2_lite_kv_probe.sh").read_text(
        encoding="utf-8"
    )
    authorization = (
        ROOT / "run_deepseek_v2_lite_kv_cap_validation.sh"
    ).read_text(encoding="utf-8")

    for source in (calibration, probe, authorization):
        assert "DEEPSEEK_WORKLOAD_PROFILE_PATH" in source
        assert 'source "$WORKLOAD_PROFILE_PATH"' in source
        assert 'sha256sum "$WORKLOAD_PROFILE_PATH"' in source

    for source in (probe, authorization):
        assert "COMMON_EPOCH0_WORKLOAD_PROFILE_ID" in source
        assert "COMMON_EPOCH0_WORKLOAD_PROFILE_SHA256" in source
        assert "COMMON_EPOCH0_PROMPTS_TOTAL_USED" in source
        assert "COMMON_EPOCH0_EXPECTED_RESPONSES_PER_STEP_USED" in source

    assert "DEEPSEEK_KV_CAP_WORKLOAD_PROFILE_ID" in authorization
    assert "DEEPSEEK_KV_CAP_WORKLOAD_PROFILE_SHA256" in authorization
    assert "DYNAMIC_SHORT_STEP_CAP_ENABLE=1" in authorization


def test_finalize_existing_common_epoch0_validates_before_commit(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "common"
    epoch = run_root / "epoch_000_mode0_probe"
    (epoch / "logs").mkdir(parents=True)
    (epoch / "rollout_data").mkdir()
    (epoch / "rollout_length").mkdir()
    checkpoint_root = epoch / "checkpoints" / "model"
    dist_ckpt = checkpoint_root / "global_step_1" / "actor" / "dist_ckpt"
    dist_ckpt.mkdir(parents=True)
    (checkpoint_root / "latest_checkpointed_iteration.txt").write_text(
        "1\n", encoding="utf-8"
    )
    (dist_ckpt / "rank0.distcp").write_bytes(b"checkpoint")
    (run_root / "INCOMPLETE").write_text("incomplete\n", encoding="utf-8")
    (epoch / "logs" / "run.txt").write_text(
        "training/global_step:1\n"
        "rollout_output_time_s: 1.25\n"
        "response/aborted_ratio:0.0\n"
        "Training Progress: 100%\n"
        "After trainer.fit\n",
        encoding="utf-8",
    )
    rows = [json.dumps({"input": f"prompt-{index}"}) for index in range(16)]
    (epoch / "rollout_data" / "1.jsonl").write_text(
        "\n".join(rows) + "\n", encoding="utf-8"
    )
    (epoch / "rollout_length" / "length_1.txt").write_text(
        "1\n" * 16, encoding="utf-8"
    )
    model = tmp_path / "model-input"
    distcp_input = tmp_path / "distcp-input"
    model.mkdir()
    distcp_input.mkdir()
    train_file = tmp_path / "train.parquet"
    test_file = tmp_path / "test.parquet"
    train_file.touch()
    test_file.touch()
    execution_sha256 = subprocess.check_output(
        [
            sys.executable,
            str(ROOT / "tools" / "hash_deepseek_execution_code.py"),
            "--root",
            str(ROOT),
        ],
        text=True,
    ).strip()
    empty_sha256 = hashlib.sha256(b"").hexdigest()
    contract = run_root / "common_epoch0_run_contract.env"
    contract.write_text(
        "\n".join(
            (
                "export COMMON_EPOCH0_RUN_CONTRACT_SCHEMA_VERSION=1",
                f"export COMMON_EPOCH0_RUN_CONTRACT_MODEL_PATH={model.resolve()}",
                "export COMMON_EPOCH0_RUN_CONTRACT_MODEL_REVISION=revision-test",
                f"export COMMON_EPOCH0_RUN_CONTRACT_DISTCP_PATH={distcp_input.resolve()}",
                "export COMMON_EPOCH0_RUN_CONTRACT_WORKLOAD_PROFILE_ID=unspecified",
                "export COMMON_EPOCH0_RUN_CONTRACT_WORKLOAD_PROFILE_SHA256=unspecified",
                f"export COMMON_EPOCH0_RUN_CONTRACT_EXECUTION_CODE_SHA256={execution_sha256}",
                f"export COMMON_EPOCH0_RUN_CONTRACT_TRAIN_FILE={train_file.resolve()}",
                f"export COMMON_EPOCH0_RUN_CONTRACT_TRAIN_FILE_SHA256={empty_sha256}",
                f"export COMMON_EPOCH0_RUN_CONTRACT_TEST_FILE={test_file.resolve()}",
                f"export COMMON_EPOCH0_RUN_CONTRACT_TEST_FILE_SHA256={empty_sha256}",
            )
        )
        + "\n",
        encoding="utf-8",
    )
    contract.chmod(0o444)
    contract_before = contract.read_bytes()

    env = os.environ.copy()
    env.update(
        {
            "COMMON_EPOCH0_OUTPUT_ROOT": str(tmp_path),
            "COMMON_EPOCH0_RUN_NAME": "common",
            "COMMON_EPOCH0_FINALIZE_EXISTING": "1",
            "CHECKPOINT_MODEL_DIR_NAME": "model",
            "MODEL_PATH": str(model),
            "MODEL_REVISION": "revision-test",
            "DISTCP_PATH": str(distcp_input),
            "TRAIN_FILE": str(train_file),
            "TEST_FILE": str(test_file),
            "COMMON_EPOCH0_TRAIN_STEPS": "1",
            "COMMON_EPOCH0_TRAIN_BATCH_SIZE": "16",
            "COMMON_EPOCH0_ROLLOUT_N": "1",
            "COMMON_EPOCH0_MAX_NUM_SEQS": "16",
            "COMMON_EPOCH0_PROMPTS_TOTAL": "16",
            "COMMON_EPOCH0_EXPECTED_RESPONSES_PER_STEP": "16",
            "COMMON_EPOCH0_KV_TOKENS_PER_RANK": "128",
            "COMMON_EPOCH0_PREEMPTION_POLICY": "record",
            "COMMON_EPOCH0_ORIGINAL_EXECUTION_CODE_SHA256": execution_sha256,
        }
    )
    result = subprocess.run(
        [str(ROOT / "run_common_epoch0_probe_gpu09_kv380800_permanent.sh")],
        cwd=ROOT,
        env=env,
        check=True,
        text=True,
        capture_output=True,
    )

    assert "validating and finalizing existing training output" in result.stdout
    assert not (run_root / "INCOMPLETE").exists()
    assert (run_root / "DO_NOT_DELETE_COMMON_EPOCH0_CHECKPOINT").is_file()
    assert (run_root / "reuse.env").is_file()
    assert (run_root / "common_epoch0_metadata.env").is_file()
    assert contract.read_bytes() == contract_before
    assert contract.stat().st_mode & 0o777 == 0o444
    assert (dist_ckpt.parent.parent / ".PRESERVE_COMMON_EPOCH0").is_file()
    history = json.loads((epoch / "offline_planning_history.json").read_text())
    assert history["prompt_count"] == 16
    assert history["prompt_occurrence_count"] == 16
