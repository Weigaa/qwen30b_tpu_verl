from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Callable

import pytest

from tools.audit_deepseek_common_epoch0 import (
    AuditError,
    _write_json_atomic,
    audit_common_epoch0,
)


PROFILE_ID = "deepseek-v2-lite-chat-b64-n16-s5-v2"
PROFILE_SHA256 = "c" * 64
COMMON_RUNTIME_SHA256 = "a" * 64
CONTINUATION_SHA256 = "b" * 64
MODEL_REVISION = "85864749cd611b4353ce1decdb286193298f64c7"
TEST_BOS_TOKEN_ID = 100000
ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "tools" / "audit_deepseek_common_epoch0.py"
RUNNER = ROOT / "run_common_epoch0_probe_gpu09_kv380800_permanent.sh"


def _write_env(path: Path, values: dict[str, str | int]) -> None:
    path.write_text(
        "".join(f"export {key}={value}\n" for key, value in values.items()),
        encoding="utf-8",
    )


def _write_test_tokenizer(model: Path) -> None:
    (model / "config.json").write_text(
        json.dumps({"bos_token_id": TEST_BOS_TOKEN_ID}) + "\n",
        encoding="utf-8",
    )
    (model / "tokenizer_config.json").write_text(
        json.dumps({"bos_token": {"content": "BOS"}}) + "\n",
        encoding="utf-8",
    )
    (model / "tokenizer.json").write_text(
        json.dumps(
            {
                "added_tokens": [
                    {"id": TEST_BOS_TOKEN_ID, "content": "BOS"}
                ]
            }
        )
        + "\n",
        encoding="utf-8",
    )


def _write_step(
    epoch: Path,
    step: int,
    occurrences: list[tuple[str, list[float]]],
) -> None:
    rollout_path = epoch / "rollout_data" / f"{step}.jsonl"
    length_path = epoch / "rollout_length" / f"length_{step}.txt"
    rollout_rows: list[str] = []
    length_rows: list[str] = []
    for prompt, lengths in occurrences:
        for index, length in enumerate(lengths):
            rollout_rows.append(
                json.dumps(
                    {
                        "input": prompt,
                        "output": f"response-{step}-{index}",
                        "prompts": [100001, 100001, TEST_BOS_TOKEN_ID, step, index],
                    },
                    ensure_ascii=True,
                )
            )
            length_rows.append(str(length))
    rollout_path.write_text("\n".join(rollout_rows) + "\n", encoding="utf-8")
    length_path.write_text("\n".join(length_rows) + "\n", encoding="utf-8")


def _make_common(tmp_path: Path) -> Path:
    experiment = tmp_path / "experiment"
    common = experiment / "common_epoch0"
    epoch = common / "epoch_000_mode0_probe"
    inputs = experiment / "inputs"
    model = inputs / "DeepSeek-V2-Lite-Chat"
    distcp_source = inputs / "DeepSeek-V2-Lite-Chat_megatron_pp4_ep4"
    model.mkdir(parents=True)
    _write_test_tokenizer(model)
    distcp_source.mkdir()
    train_file = inputs / "train.parquet"
    test_file = inputs / "test.parquet"
    train_file.write_bytes(b"frozen training dataset\n")
    test_file.write_bytes(b"frozen test dataset\n")
    (epoch / "rollout_data").mkdir(parents=True)
    (epoch / "rollout_length").mkdir()
    (epoch / "logs").mkdir()

    _write_step(epoch, 1, [("prompt-a", [1.0, 2.0]), ("prompt-b", [3.0, 4.0])])
    _write_step(epoch, 2, [("prompt-b", [5.0, 6.0]), ("prompt-c", [7.0, 8.0])])
    history = {
        "schema_version": 1,
        "steps": 2,
        "responses_per_prompt": 2,
        "prompt_count": 3,
        "prompt_occurrence_count": 4,
        "duplicate_prompt_occurrence_count": 1,
        "duplicate_prompt_policy": "latest_occurrence",
        "source_files": [
            {
                "rollout_data": "rollout_data/1.jsonl",
                "rollout_length": "rollout_length/length_1.txt",
            },
            {
                "rollout_data": "rollout_data/2.jsonl",
                "rollout_length": "rollout_length/length_2.txt",
            },
        ],
        "records": [
            {
                "input": "prompt-a",
                "lengths": [1.0, 2.0],
                "latest_logical_step": 1,
                "latest_source_step": 1,
            },
            {
                "input": "prompt-b",
                "lengths": [5.0, 6.0],
                "latest_logical_step": 2,
                "latest_source_step": 2,
            },
            {
                "input": "prompt-c",
                "lengths": [7.0, 8.0],
                "latest_logical_step": 2,
                "latest_source_step": 2,
            },
        ],
    }
    (epoch / "offline_planning_history.json").write_text(
        json.dumps(history) + "\n", encoding="utf-8"
    )

    log = "\n".join(
        (
            "rollout_output_time_s: 1.25 training/global_step:1 response/aborted_ratio:0.0",
            "Preempting request req-1 for request req-2",
            "rollout_output_time_s: 2.50 training/global_step:2 response/aborted_ratio:0.0",
            "Training Progress: 100%",
            "[TaskRunner] After trainer.fit(), about to finish run()",
        )
    )
    (epoch / "logs" / "train.txt").write_text(log + "\n", encoding="utf-8")

    checkpoint_root = epoch / "checkpoints" / "deepseek_v2_lite"
    checkpoint = checkpoint_root / "global_step_2"
    distcp = checkpoint / "actor" / "dist_ckpt"
    distcp.mkdir(parents=True)
    (distcp / "__0_0.distcp").write_bytes(b"abc")
    (distcp / "__0_1.distcp").write_bytes(b"defg")
    (checkpoint / ".PRESERVE_COMMON_EPOCH0").touch()
    (checkpoint_root / "latest_checkpointed_iteration.txt").write_text(
        "2\n", encoding="utf-8"
    )

    _write_env(
        common / "reuse.env",
        {
            "DYNAMIC_INITIAL_BASELINE_DIR": epoch,
            "BASELINE_INITIAL_RESUME_CKPT": checkpoint,
            "DYNAMIC_INITIAL_RESUME_CKPT": checkpoint,
        },
    )
    contract_path = common / "common_epoch0_run_contract.env"
    _write_env(
        contract_path,
        {
            "COMMON_EPOCH0_RUN_CONTRACT_SCHEMA_VERSION": 1,
            "COMMON_EPOCH0_RUN_CONTRACT_MODEL_PATH": model,
            "COMMON_EPOCH0_RUN_CONTRACT_MODEL_REVISION": MODEL_REVISION,
            "COMMON_EPOCH0_RUN_CONTRACT_DISTCP_PATH": distcp_source,
            "COMMON_EPOCH0_RUN_CONTRACT_WORKLOAD_PROFILE_ID": PROFILE_ID,
            "COMMON_EPOCH0_RUN_CONTRACT_WORKLOAD_PROFILE_SHA256": PROFILE_SHA256,
            "COMMON_EPOCH0_RUN_CONTRACT_EXECUTION_CODE_SHA256": (
                COMMON_RUNTIME_SHA256
            ),
            "COMMON_EPOCH0_RUN_CONTRACT_TRAIN_FILE": train_file,
            "COMMON_EPOCH0_RUN_CONTRACT_TRAIN_FILE_SHA256": hashlib.sha256(
                train_file.read_bytes()
            ).hexdigest(),
            "COMMON_EPOCH0_RUN_CONTRACT_TEST_FILE": test_file,
            "COMMON_EPOCH0_RUN_CONTRACT_TEST_FILE_SHA256": hashlib.sha256(
                test_file.read_bytes()
            ).hexdigest(),
        },
    )
    contract_path.chmod(0o444)
    _write_env(
        common / "common_epoch0_metadata.env",
        {
            "COMMON_EPOCH0_MODEL_PATH": model,
            "COMMON_EPOCH0_MODEL_REVISION": MODEL_REVISION,
            "COMMON_EPOCH0_DISTCP_PATH": distcp_source,
            "COMMON_EPOCH0_ORIGINAL_EXECUTION_CODE_SHA256": COMMON_RUNTIME_SHA256,
            "COMMON_EPOCH0_RUN_CONTRACT_SHA256": hashlib.sha256(
                contract_path.read_bytes()
            ).hexdigest(),
            "COMMON_EPOCH0_EFFECTIVE_KV_TOKENS_PER_RANK": 1280,
            "COMMON_EPOCH0_TRAIN_STEPS_USED": 2,
            "COMMON_EPOCH0_TRAIN_BATCH_SIZE_USED": 2,
            "COMMON_EPOCH0_ROLLOUT_N_USED": 2,
            "COMMON_EPOCH0_PROMPTS_TOTAL_USED": 4,
            "COMMON_EPOCH0_EXPECTED_RESPONSES_PER_STEP_USED": 4,
            "COMMON_EPOCH0_PREEMPTION_POLICY_USED": "record",
            "COMMON_EPOCH0_PREEMPTION_COUNT": 1,
            "COMMON_EPOCH0_WORKLOAD_PROFILE_ID": PROFILE_ID,
            "COMMON_EPOCH0_WORKLOAD_PROFILE_SHA256": PROFILE_SHA256,
            "COMMON_EPOCH0_TRAIN_FILE_USED": train_file,
            "COMMON_EPOCH0_TRAIN_FILE_SHA256": hashlib.sha256(
                train_file.read_bytes()
            ).hexdigest(),
            "COMMON_EPOCH0_TEST_FILE_USED": test_file,
            "COMMON_EPOCH0_TEST_FILE_SHA256": hashlib.sha256(
                test_file.read_bytes()
            ).hexdigest(),
        },
    )
    (common / "MEASURED_GPU_KV_CACHE_TOKENS_PER_RANK").write_text(
        "1280\n", encoding="utf-8"
    )
    (common / "DO_NOT_DELETE_COMMON_EPOCH0_CHECKPOINT").write_text(
        "PERMANENT COMMON EPOCH0 CHECKPOINT\n"
        f"Rollout history: {epoch}\n"
        f"Resume checkpoint: {checkpoint}\n",
        encoding="utf-8",
    )

    (experiment / "EXECUTION_CODE_SHA256").write_text(
        COMMON_RUNTIME_SHA256 + "\n", encoding="utf-8"
    )
    (experiment / "COMMON_EPOCH0_ROLLOUT_EXECUTION_CODE_SHA256").write_text(
        COMMON_RUNTIME_SHA256 + "\n", encoding="utf-8"
    )
    (experiment / "CONTINUATION_EXECUTION_CODE_SHA256").write_text(
        CONTINUATION_SHA256 + "\n", encoding="utf-8"
    )
    _write_env(
        experiment / "POSTPROCESS_CODE_MIGRATION.env",
        {
            "DEEPSEEK_BATCH64_COMMON_ROLLOUT_EXECUTION_CODE_SHA256": (
                COMMON_RUNTIME_SHA256
            ),
            "DEEPSEEK_BATCH64_CONTINUATION_EXECUTION_CODE_SHA256": (
                CONTINUATION_SHA256
            ),
            "DEEPSEEK_BATCH64_CODE_MIGRATION_SCOPE": "common_epoch0_postprocessing",
        },
    )
    return common


def _audit(common: Path, **overrides: object) -> dict:
    inputs = common.parent / "inputs"
    arguments = dict(
        common_root=common,
        expected_steps=2,
        expected_batch_size=2,
        expected_rollout_n=2,
        expected_workload_profile_id=PROFILE_ID,
        expected_workload_profile_sha256=PROFILE_SHA256,
        expected_common_runtime_sha256=COMMON_RUNTIME_SHA256,
        expected_continuation_sha256=CONTINUATION_SHA256,
        expected_model_path=inputs / "DeepSeek-V2-Lite-Chat",
        expected_model_revision=MODEL_REVISION,
        expected_distcp_path=inputs / "DeepSeek-V2-Lite-Chat_megatron_pp4_ep4",
        expected_train_file=inputs / "train.parquet",
        expected_test_file=inputs / "test.parquet",
        expected_unique_prompts=3,
        expected_duplicate_occurrences=1,
        expected_duplicate_policy="latest_occurrence",
        expected_preemption_policy="record",
        expected_preemption_count=1,
        expected_measured_kv_tokens=1280,
        expected_distcp_count=2,
        block_size=128,
    )
    arguments.update(overrides)
    return audit_common_epoch0(**arguments)


def test_audit_accepts_complete_recovered_common_and_records_hashes(
    tmp_path: Path,
) -> None:
    common = _make_common(tmp_path)
    payload = _audit(common)

    assert payload["status"] == "PASS"
    assert payload["run_contract"]["mode"] == "444"
    assert payload["run_contract"]["model_revision"] == MODEL_REVISION
    assert payload["run_contract"]["execution_code_sha256"] == (
        COMMON_RUNTIME_SHA256
    )
    history_path = (
        common / "epoch_000_mode0_probe" / "offline_planning_history.json"
    )
    assert payload["offline_planning_history"] == {
        "path": "offline_planning_history.json",
        "bytes": history_path.stat().st_size,
        "sha256": hashlib.sha256(history_path.read_bytes()).hexdigest(),
        "prompt_occurrence_count": 4,
        "unique_prompt_count": 3,
        "duplicate_prompt_occurrence_count": 1,
        "duplicate_prompt_policy": "latest_occurrence",
    }
    assert payload["checkpoint"]["distcp_shard_count"] == 2
    assert payload["checkpoint"]["distcp_total_bytes"] == 7
    assert payload["checkpoint"]["distcp_shards_hashed"] is False
    assert payload["training_log"]["preemption_count"] == 1
    assert all(
        artifact[kind]["sha256"]
        for artifact in payload["rollout_artifacts"]
        for kind in ("rollout_data", "rollout_length")
    )


def test_atomic_manifest_write_refuses_overwrite_without_force(tmp_path: Path) -> None:
    common = _make_common(tmp_path)
    payload = _audit(common)
    output = common / "COMMON_EPOCH0_RECOVERY_MANIFEST.json"

    _write_json_atomic(output, payload, force=False)
    assert json.loads(output.read_text(encoding="utf-8")) == payload
    with pytest.raises(AuditError, match="refusing to overwrite"):
        _write_json_atomic(output, payload, force=False)
    updated = payload | {"schema_version": 2}
    _write_json_atomic(output, updated, force=True)
    assert json.loads(output.read_text(encoding="utf-8"))["schema_version"] == 2
    assert not list(common.glob(f".{output.name}.tmp.*"))


def test_audit_records_and_enforces_rollout_quality(tmp_path: Path) -> None:
    common = _make_common(tmp_path)
    payload = _audit(
        common,
        max_response_length=8,
        max_clip_ratio=0.2,
        min_distinct_prompt_maxima=4,
    )
    assert payload["rollout_quality"] == {
        "max_response_length": 8,
        "clipped_response_count": 1,
        "response_count": 8,
        "clip_ratio": 0.125,
        "distinct_prompt_maxima": 4,
    }
    with pytest.raises(AuditError, match="response clip ratio"):
        _audit(common, max_response_length=8, max_clip_ratio=0.1)
    with pytest.raises(AuditError, match="distinct prompt maxima"):
        _audit(
            common,
            max_response_length=8,
            min_distinct_prompt_maxima=5,
        )


@pytest.mark.parametrize(("bos_count", "accepted"), ((0, False), (1, True), (2, False)))
def test_audit_requires_exactly_one_bos_per_left_padded_prompt(
    tmp_path: Path, bos_count: int, accepted: bool
) -> None:
    common = _make_common(tmp_path)
    path = common / "epoch_000_mode0_probe" / "rollout_data" / "1.jsonl"
    rows = path.read_text(encoding="utf-8").splitlines()
    first = json.loads(rows[0])
    first["prompts"] = [100001, 100001] + [TEST_BOS_TOKEN_ID] * bos_count + [42]
    rows[0] = json.dumps(first, ensure_ascii=True)
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")

    if accepted:
        assert _audit(common)["status"] == "PASS"
    else:
        with pytest.raises(AuditError, match="exactly one BOS token"):
            _audit(common)


def test_cli_atomically_writes_recovery_manifest(tmp_path: Path) -> None:
    common = _make_common(tmp_path)
    inputs = common.parent / "inputs"
    output = common / "COMMON_EPOCH0_RECOVERY_MANIFEST.json"
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--common-root",
            str(common),
            "--expected-steps",
            "2",
            "--expected-batch-size",
            "2",
            "--expected-rollout-n",
            "2",
            "--expected-workload-profile-id",
            PROFILE_ID,
            "--expected-workload-profile-sha256",
            PROFILE_SHA256,
            "--expected-common-runtime-sha256",
            COMMON_RUNTIME_SHA256,
            "--expected-continuation-sha256",
            CONTINUATION_SHA256,
            "--expected-model-path",
            str(inputs / "DeepSeek-V2-Lite-Chat"),
            "--expected-model-revision",
            MODEL_REVISION,
            "--expected-distcp-path",
            str(inputs / "DeepSeek-V2-Lite-Chat_megatron_pp4_ep4"),
            "--expected-train-file",
            str(inputs / "train.parquet"),
            "--expected-test-file",
            str(inputs / "test.parquet"),
            "--expected-unique-prompts",
            "3",
            "--expected-duplicate-occurrences",
            "1",
            "--expected-preemption-count",
            "1",
            "--expected-measured-kv-tokens",
            "1280",
            "--expected-distcp-count",
            "2",
            "--output",
            str(output),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    assert "PASS recovery_manifest=" in result.stdout
    assert json.loads(output.read_text(encoding="utf-8"))["status"] == "PASS"
    assert not list(common.glob(f".{output.name}.tmp.*"))


def _remove_marker(common: Path) -> None:
    (common / "DO_NOT_DELETE_COMMON_EPOCH0_CHECKPOINT").unlink()


def _break_rollout_rows(common: Path) -> None:
    path = common / "epoch_000_mode0_probe" / "rollout_data" / "2.jsonl"
    lines = path.read_text(encoding="utf-8").splitlines()
    path.write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")


def _stale_history(common: Path) -> None:
    path = common / "epoch_000_mode0_probe" / "offline_planning_history.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["records"][1]["lengths"] = [3.0, 4.0]
    payload["records"][1]["latest_logical_step"] = 1
    payload["records"][1]["latest_source_step"] = 1
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


def _add_abort(common: Path) -> None:
    path = common / "epoch_000_mode0_probe" / "logs" / "train.txt"
    path.write_text(
        path.read_text(encoding="utf-8").replace(
            "response/aborted_ratio:0.0", "response/aborted_ratio:0.5", 1
        ),
        encoding="utf-8",
    )


def _add_oom(common: Path) -> None:
    path = common / "epoch_000_mode0_probe" / "logs" / "train.txt"
    path.write_text(
        path.read_text(encoding="utf-8") + "NPU out of memory\n",
        encoding="utf-8",
    )


def _break_tracker(common: Path) -> None:
    tracker = (
        common
        / "epoch_000_mode0_probe"
        / "checkpoints"
        / "deepseek_v2_lite"
        / "latest_checkpointed_iteration.txt"
    )
    tracker.write_text("1\n", encoding="utf-8")


def _empty_shard(common: Path) -> None:
    shard = (
        common
        / "epoch_000_mode0_probe"
        / "checkpoints"
        / "deepseek_v2_lite"
        / "global_step_2"
        / "actor"
        / "dist_ckpt"
        / "__0_0.distcp"
    )
    shard.write_bytes(b"")


def _break_measured_kv(common: Path) -> None:
    (common / "MEASURED_GPU_KV_CACHE_TOKENS_PER_RANK").write_text(
        "1279\n", encoding="utf-8"
    )


def _break_continuation_hash(common: Path) -> None:
    (common.parent / "CONTINUATION_EXECUTION_CODE_SHA256").write_text(
        "d" * 64 + "\n", encoding="utf-8"
    )


def _remove_run_contract(common: Path) -> None:
    (common / "common_epoch0_run_contract.env").unlink()


def _make_run_contract_writable(common: Path) -> None:
    (common / "common_epoch0_run_contract.env").chmod(0o644)


def _change_run_contract_model_revision(common: Path) -> None:
    path = common / "common_epoch0_run_contract.env"
    path.chmod(0o644)
    path.write_text(
        path.read_text(encoding="utf-8").replace(MODEL_REVISION, "wrong-revision"),
        encoding="utf-8",
    )
    path.chmod(0o444)


def _change_metadata_model_revision(common: Path) -> None:
    path = common / "common_epoch0_metadata.env"
    path.write_text(
        path.read_text(encoding="utf-8").replace(MODEL_REVISION, "wrong-revision"),
        encoding="utf-8",
    )


def _change_training_dataset(common: Path) -> None:
    path = common.parent / "inputs" / "train.parquet"
    path.write_bytes(path.read_bytes() + b"mutated after rollout\n")


@pytest.mark.parametrize(
    ("mutate", "message"),
    (
        (_remove_marker, "completion marker is missing"),
        (_break_rollout_rows, "row count mismatch"),
        (_stale_history, "does not retain the latest prompt occurrence"),
        (_add_abort, "invalid aborted ratios"),
        (_add_oom, "OOM evidence"),
        (_break_tracker, "checkpoint tracker"),
        (_empty_shard, "missing or empty shards"),
        (_break_measured_kv, "positive block multiple"),
        (_break_continuation_hash, "continuation execution code SHA256"),
        (_remove_run_contract, "environment file does not exist"),
        (_make_run_contract_writable, "run contract must be read-only"),
        (_change_run_contract_model_revision, "MODEL_REVISION"),
        (_change_metadata_model_revision, "COMMON_EPOCH0_MODEL_REVISION"),
        (_change_training_dataset, "training dataset SHA256 mismatch"),
    ),
)
def test_audit_rejects_inconsistent_recovery_evidence(
    tmp_path: Path,
    mutate: Callable[[Path], None],
    message: str,
) -> None:
    common = _make_common(tmp_path)
    mutate(common)
    with pytest.raises(AuditError, match=message):
        _audit(common)


def test_runner_commits_contract_before_training_and_never_rewrites_recovery() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    contract_commit = source.index('mv -T "$contract_tmp" "$RUN_CONTRACT_ENV"')
    training_launch = source.index('"$SCRIPT_DIR/run_mode0_no_shrink_baseline.sh"')
    assert contract_commit < training_launch
    assert 'cmp -s "$expected_tmp" "$RUN_CONTRACT_ENV"' in source
    assert source.count('mv -T "$contract_tmp" "$RUN_CONTRACT_ENV"') == 1
    assert 'chmod 0444 "$contract_tmp"' in source
