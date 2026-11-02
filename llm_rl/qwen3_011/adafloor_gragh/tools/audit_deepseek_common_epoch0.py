#!/usr/bin/env python3
"""Audit a finalized DeepSeek common epoch0 and write a recovery manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shlex
import tempfile
from pathlib import Path
from typing import Any


HISTORY_FILENAME = "offline_planning_history.json"
COMPLETION_MARKER = "DO_NOT_DELETE_COMMON_EPOCH0_CHECKPOINT"
MEASURED_KV_FILENAME = "MEASURED_GPU_KV_CACHE_TOKENS_PER_RANK"
RUN_CONTRACT_FILENAME = "common_epoch0_run_contract.env"
DEFAULT_MODEL_PATH = Path("/data/DeepSeek-V2-Lite-Chat")
DEFAULT_MODEL_REVISION = "85864749cd611b4353ce1decdb286193298f64c7"
DEFAULT_DISTCP_PATH = Path("/data/DeepSeek-V2-Lite-Chat_megatron_pp4_ep4")
DEFAULT_TRAIN_FILE = Path("/data/deepscaler/train.parquet")
DEFAULT_TEST_FILE = Path("/data/deepscaler/test.parquet")
SHA256_RE = re.compile(r"[0-9a-f]{64}")
GLOBAL_STEP_RE = re.compile(r"training/global_step:([0-9]+)")
ROLLOUT_TIME_RE = re.compile(r"rollout_output_time_s:\s*([0-9.eE+-]+)")
ABORT_RE = re.compile(r"response/aborted_ratio:([0-9.eE+-]+)")
PREEMPT_RE = re.compile(r"preempting request|request preempted", re.IGNORECASE)
OOM_RE = re.compile(
    r"NPU out of memory|Memory_Allocation_Failure|"
    r"Failed to allocate[^\r\n]*NPU memory|OutOfMemoryError|"
    r"NPU memory is exhausted|ACL_ERROR_RT_MEMORY_ALLOCATION",
    re.IGNORECASE,
)
class AuditError(RuntimeError):
    """Raised when common epoch0 evidence violates the expected contract."""


def _fail(message: str) -> None:
    raise AuditError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_env(path: Path) -> dict[str, str]:
    if not path.is_file():
        _fail(f"environment file does not exist: {path}")
    values: dict[str, str] = {}
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].strip()
        key, separator, raw_value = line.partition("=")
        if not separator or not key.strip():
            _fail(f"{path}:{line_number} is not an environment assignment")
        key = key.strip()
        if key in values:
            _fail(f"{path}:{line_number} repeats {key}")
        try:
            words = shlex.split(raw_value, comments=True, posix=True)
        except ValueError as exc:
            raise AuditError(f"cannot parse {path}:{line_number}: {exc}") from exc
        if len(words) > 1:
            _fail(f"{path}:{line_number} contains multiple shell words")
        values[key] = words[0] if words else ""
    return values


def _required(values: dict[str, str], key: str, context: str) -> str:
    value = values.get(key)
    if value is None or value == "":
        _fail(f"{context} does not define {key}")
    return value


def _required_int(values: dict[str, str], key: str, context: str) -> int:
    value = _required(values, key, context)
    try:
        return int(value)
    except ValueError as exc:
        raise AuditError(f"{context} has invalid integer {key}={value!r}") from exc


def _require_value(
    values: dict[str, str], key: str, expected: str | int, context: str
) -> None:
    observed = _required(values, key, context)
    if observed != str(expected):
        _fail(
            f"{context} has {key}={observed!r}, expected {str(expected)!r}"
        )


def _require_path(recorded: str, expected: Path, label: str) -> None:
    if Path(recorded).expanduser().resolve() != expected.resolve():
        _fail(f"{label} path mismatch: recorded={recorded}, expected={expected}")


def _validate_sha256(value: str, label: str) -> str:
    if SHA256_RE.fullmatch(value) is None:
        _fail(f"{label} is not a lowercase SHA256 digest")
    return value


def _read_json(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        _fail(f"missing {label}: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise AuditError(f"invalid {label} JSON: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        _fail(f"{label} is not a JSON object: {path}")
    return payload


def _chat_bos_token_id(model_path: Path) -> int:
    config = _read_json(model_path / "config.json", "model config")
    tokenizer_config = _read_json(
        model_path / "tokenizer_config.json", "tokenizer config"
    )
    tokenizer = _read_json(model_path / "tokenizer.json", "tokenizer")
    config_bos = config.get("bos_token_id")
    if (
        isinstance(config_bos, bool)
        or not isinstance(config_bos, int)
        or config_bos < 0
    ):
        _fail("model config has no valid bos_token_id")
    bos_token = tokenizer_config.get("bos_token")
    if isinstance(bos_token, dict):
        bos_content = bos_token.get("content")
    else:
        bos_content = bos_token
    if not isinstance(bos_content, str) or not bos_content:
        _fail("tokenizer config has no valid bos_token")
    added_tokens = tokenizer.get("added_tokens")
    if not isinstance(added_tokens, list):
        _fail("tokenizer has no added_tokens list")
    tokenizer_bos_ids = [
        token.get("id")
        for token in added_tokens
        if isinstance(token, dict) and token.get("content") == bos_content
    ]
    if (
        len(tokenizer_bos_ids) != 1
        or isinstance(tokenizer_bos_ids[0], bool)
        or not isinstance(tokenizer_bos_ids[0], int)
        or tokenizer_bos_ids[0] < 0
    ):
        _fail("tokenizer does not map bos_token to exactly one valid token ID")
    if tokenizer_bos_ids[0] != config_bos:
        _fail(
            "model config bos_token_id differs from the tokenizer BOS token ID"
        )
    return config_bos


def _input_from_rollout_line(line: str, bos_token_id: int) -> str:
    try:
        parsed = json.loads(line)
    except (json.JSONDecodeError, TypeError) as exc:
        raise AuditError(f"invalid rollout JSON record: {exc}") from exc
    if not isinstance(parsed, dict):
        _fail("rollout record is not a JSON object")
    if "input" not in parsed:
        _fail("rollout record has no input field")
    prompts = parsed.get("prompts")
    if not isinstance(prompts, list) or not prompts:
        _fail("rollout record has no nonempty prompts list")
    if any(isinstance(token, bool) or not isinstance(token, int) for token in prompts):
        _fail("rollout record prompts contains a non-integer token ID")
    bos_count = sum(token == bos_token_id for token in prompts)
    if bos_count != 1:
        _fail(
            "rollout record prompt must contain exactly one BOS token "
            f"{bos_token_id}, found {bos_count}"
        )
    return str(parsed["input"])


def _read_lengths(path: Path) -> list[float]:
    values: list[float] = []
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        text = raw_line.strip()
        if not text:
            continue
        try:
            value = float(text)
        except ValueError as exc:
            raise AuditError(f"{path}:{line_number} is not numeric") from exc
        if not math.isfinite(value) or value < 0:
            _fail(f"{path}:{line_number} has invalid response length {value}")
        values.append(value)
    return values


def _audit_artifacts_and_history(
    epoch_dir: Path,
    steps: int,
    batch_size: int,
    rollout_n: int,
    bos_token_id: int,
    expected_unique_prompts: int | None,
    expected_duplicate_occurrences: int | None,
    expected_duplicate_policy: str,
    max_response_length: int | None,
    max_clip_ratio: float | None,
    min_distinct_prompt_maxima: int | None,
) -> dict[str, Any]:
    expected_rows = batch_size * rollout_n
    latest_by_prompt: dict[str, dict[str, Any]] = {}
    occurrence_count = 0
    duplicate_count = 0
    artifact_records: list[dict[str, Any]] = []
    all_lengths: list[float] = []
    prompt_maxima: list[float] = []

    rollout_dir = epoch_dir / "rollout_data"
    length_dir = epoch_dir / "rollout_length"
    rollout_files = sorted(rollout_dir.glob("*.jsonl"))
    length_files = sorted(length_dir.glob("length_*.txt"))
    if len(rollout_files) != steps or len(length_files) != steps:
        _fail(
            f"expected {steps} rollout/length files, found "
            f"{len(rollout_files)}/{len(length_files)}"
        )

    for step in range(1, steps + 1):
        rollout_path = rollout_dir / f"{step}.jsonl"
        length_path = length_dir / f"length_{step}.txt"
        if not rollout_path.is_file() or not length_path.is_file():
            _fail(f"missing rollout artifact pair for step {step}")
        prompt_inputs: list[str] = []
        with rollout_path.open("r", encoding="utf-8") as source:
            for line_number, line in enumerate(source, 1):
                if line.strip():
                    try:
                        prompt_inputs.append(
                            _input_from_rollout_line(line, bos_token_id)
                        )
                    except AuditError as exc:
                        raise AuditError(
                            f"{rollout_path}:{line_number}: {exc}"
                        ) from exc
        lengths = _read_lengths(length_path)
        if len(prompt_inputs) != expected_rows or len(lengths) != expected_rows:
            _fail(
                f"step {step} row count mismatch: rollout={len(prompt_inputs)} "
                f"length={len(lengths)} expected={expected_rows}"
            )
        for offset in range(0, expected_rows, rollout_n):
            prompt_group = prompt_inputs[offset : offset + rollout_n]
            prompt = prompt_group[0]
            if any(candidate != prompt for candidate in prompt_group[1:]):
                _fail(
                    f"step {step} rows {offset}:{offset + rollout_n} "
                    "do not belong to one prompt occurrence"
                )
            occurrence_count += 1
            if prompt in latest_by_prompt:
                duplicate_count += 1
            latest_by_prompt[prompt] = {
                "input": prompt,
                "lengths": [float(value) for value in lengths[offset : offset + rollout_n]],
                "latest_logical_step": step,
                "latest_source_step": step,
            }
            occurrence_lengths = lengths[offset : offset + rollout_n]
            all_lengths.extend(occurrence_lengths)
            prompt_maxima.append(max(occurrence_lengths))
        artifact_records.append(
            {
                "step": step,
                "rollout_data": {
                    "path": str(rollout_path.relative_to(epoch_dir)),
                    "rows": len(prompt_inputs),
                    "bytes": rollout_path.stat().st_size,
                    "sha256": _sha256(rollout_path),
                },
                "rollout_length": {
                    "path": str(length_path.relative_to(epoch_dir)),
                    "rows": len(lengths),
                    "bytes": length_path.stat().st_size,
                    "sha256": _sha256(length_path),
                },
            }
        )

    expected_occurrences = steps * batch_size
    if occurrence_count != expected_occurrences:
        _fail(
            f"history source has {occurrence_count} prompt occurrences, "
            f"expected {expected_occurrences}"
        )
    unique_count = len(latest_by_prompt)
    if duplicate_count != occurrence_count - unique_count:
        _fail("duplicate prompt accounting is internally inconsistent")
    if expected_unique_prompts is not None and unique_count != expected_unique_prompts:
        _fail(
            f"history source has {unique_count} unique prompts, "
            f"expected {expected_unique_prompts}"
        )
    if (
        expected_duplicate_occurrences is not None
        and duplicate_count != expected_duplicate_occurrences
    ):
        _fail(
            f"history source has {duplicate_count} duplicate occurrences, "
            f"expected {expected_duplicate_occurrences}"
        )

    history_path = epoch_dir / HISTORY_FILENAME
    history = _read_json(history_path, "offline planning history")
    expected_history_values: dict[str, Any] = {
        "schema_version": 1,
        "steps": steps,
        "responses_per_prompt": rollout_n,
        "prompt_count": unique_count,
        "prompt_occurrence_count": occurrence_count,
        "duplicate_prompt_occurrence_count": duplicate_count,
        "duplicate_prompt_policy": expected_duplicate_policy,
    }
    for key, expected in expected_history_values.items():
        if history.get(key) != expected:
            _fail(
                f"offline planning history has {key}={history.get(key)!r}, "
                f"expected {expected!r}"
            )
    source_files = history.get("source_files")
    expected_sources = [
        {
            "rollout_data": f"rollout_data/{step}.jsonl",
            "rollout_length": f"rollout_length/length_{step}.txt",
        }
        for step in range(1, steps + 1)
    ]
    if source_files != expected_sources:
        _fail("offline planning history source_files do not match raw artifacts")
    records = history.get("records")
    if not isinstance(records, list) or len(records) != unique_count:
        _fail("offline planning history records do not match prompt_count")
    history_by_prompt: dict[str, dict[str, Any]] = {}
    for index, record in enumerate(records):
        if not isinstance(record, dict) or not isinstance(record.get("input"), str):
            _fail(f"offline planning history record {index} is invalid")
        prompt = record["input"]
        if prompt in history_by_prompt:
            _fail("offline planning history repeats a prompt record")
        history_by_prompt[prompt] = record
    if set(history_by_prompt) != set(latest_by_prompt):
        _fail("offline planning history prompt keys do not match raw artifacts")
    for prompt, expected_record in latest_by_prompt.items():
        if history_by_prompt[prompt] != expected_record:
            _fail("offline planning history does not retain the latest prompt occurrence")

    quality: dict[str, Any] = {}
    if max_response_length is not None:
        clipped = sum(value >= max_response_length for value in all_lengths)
        clip_ratio = clipped / len(all_lengths)
        distinct_prompt_maxima = len(set(prompt_maxima))
        if max_clip_ratio is not None and clip_ratio > max_clip_ratio:
            _fail(
                f"response clip ratio {clip_ratio:.6f} exceeds "
                f"the allowed {max_clip_ratio:.6f}"
            )
        if (
            min_distinct_prompt_maxima is not None
            and distinct_prompt_maxima < min_distinct_prompt_maxima
        ):
            _fail(
                f"history has {distinct_prompt_maxima} distinct prompt maxima, "
                f"expected at least {min_distinct_prompt_maxima}"
            )
        quality = {
            "max_response_length": max_response_length,
            "clipped_response_count": clipped,
            "response_count": len(all_lengths),
            "clip_ratio": clip_ratio,
            "distinct_prompt_maxima": distinct_prompt_maxima,
        }

    return {
        "files": artifact_records,
        "history": {
            "path": str(history_path.relative_to(epoch_dir)),
            "bytes": history_path.stat().st_size,
            "sha256": _sha256(history_path),
            "prompt_occurrence_count": occurrence_count,
            "unique_prompt_count": unique_count,
            "duplicate_prompt_occurrence_count": duplicate_count,
            "duplicate_prompt_policy": expected_duplicate_policy,
        },
        "quality": quality,
    }


def _audit_log(
    epoch_dir: Path,
    steps: int,
    expected_preemption_policy: str,
    expected_preemption_count: int | None,
) -> dict[str, Any]:
    logs = sorted(
        (epoch_dir / "logs").glob("*.txt"),
        key=lambda path: (path.stat().st_mtime_ns, path.name),
    )
    if not logs:
        _fail(f"no common epoch0 training log under {epoch_dir / 'logs'}")
    log_path = logs[-1]
    text = log_path.read_text(encoding="utf-8", errors="replace")
    global_steps = [int(value) for value in GLOBAL_STEP_RE.findall(text)]
    expected_global_steps = list(range(1, steps + 1))
    if global_steps != expected_global_steps:
        _fail(f"global step metrics mismatch: {global_steps} != {expected_global_steps}")
    rollout_times = [float(value) for value in ROLLOUT_TIME_RE.findall(text)]
    if len(rollout_times) != steps or any(
        not math.isfinite(value) or value <= 0 for value in rollout_times
    ):
        _fail(f"invalid rollout timings: {rollout_times}")
    aborted_ratios = [float(value) for value in ABORT_RE.findall(text)]
    if len(aborted_ratios) != steps or any(value != 0.0 for value in aborted_ratios):
        _fail(f"invalid aborted ratios: {aborted_ratios}")
    for marker in ("Training Progress: 100%", "After trainer.fit"):
        if marker not in text:
            _fail(f"training log is missing completion marker {marker!r}")
    oom_matches = OOM_RE.findall(text)
    if oom_matches:
        _fail(f"training log contains NPU OOM evidence: {oom_matches[:5]}")
    preemption_count = sum(
        1 for line in text.splitlines() if PREEMPT_RE.search(line) is not None
    )
    if expected_preemption_policy == "forbid" and preemption_count:
        _fail("training log contains preemption evidence under forbid policy")
    if (
        expected_preemption_count is not None
        and preemption_count != expected_preemption_count
    ):
        _fail(
            f"training log has {preemption_count} preemption lines, "
            f"expected {expected_preemption_count}"
        )
    return {
        "path": str(log_path.relative_to(epoch_dir)),
        "bytes": log_path.stat().st_size,
        "sha256": _sha256(log_path),
        "global_steps": global_steps,
        "rollout_output_time_seconds": rollout_times,
        "aborted_ratios": aborted_ratios,
        "preemption_policy": expected_preemption_policy,
        "preemption_count": preemption_count,
        "completion_markers": ["Training Progress: 100%", "After trainer.fit"],
        "oom_evidence_count": 0,
    }


def _audit_checkpoint(
    checkpoint: Path,
    steps: int,
    expected_distcp_count: int | None,
) -> dict[str, Any]:
    if checkpoint.name != f"global_step_{steps}":
        _fail(f"checkpoint is not global_step_{steps}: {checkpoint}")
    if not (checkpoint / "actor").is_dir():
        _fail(f"checkpoint actor directory is missing: {checkpoint / 'actor'}")
    preserve = checkpoint / ".PRESERVE_COMMON_EPOCH0"
    if not preserve.is_file():
        _fail(f"checkpoint preservation marker is missing: {preserve}")
    tracker = checkpoint.parent / "latest_checkpointed_iteration.txt"
    if not tracker.is_file() or tracker.read_text(encoding="utf-8").strip() != str(steps):
        _fail(f"checkpoint tracker does not select global_step_{steps}: {tracker}")
    distcp_dir = checkpoint / "actor" / "dist_ckpt"
    if not distcp_dir.is_dir():
        _fail(f"distributed checkpoint directory is missing: {distcp_dir}")
    shards = sorted(distcp_dir.glob("*.distcp"))
    empty = [path.name for path in shards if path.stat().st_size <= 0]
    if not shards or empty:
        _fail(f"distributed checkpoint has missing or empty shards: {empty}")
    if expected_distcp_count is not None and len(shards) != expected_distcp_count:
        _fail(
            f"distributed checkpoint has {len(shards)} shards, "
            f"expected {expected_distcp_count}"
        )
    return {
        "path": str(checkpoint),
        "tracker_path": str(tracker),
        "tracker_iteration": steps,
        "distcp_directory": str(distcp_dir),
        "distcp_shard_count": len(shards),
        "distcp_total_bytes": sum(path.stat().st_size for path in shards),
        "distcp_shards_hashed": False,
    }


def _audit_runtime_provenance(
    common_root: Path,
    metadata: dict[str, str],
    expected_common_runtime_sha256: str,
    expected_continuation_sha256: str,
) -> dict[str, Any]:
    experiment_root = common_root.parent
    _require_value(
        metadata,
        "COMMON_EPOCH0_ORIGINAL_EXECUTION_CODE_SHA256",
        expected_common_runtime_sha256,
        "common metadata",
    )
    original_record = experiment_root / "EXECUTION_CODE_SHA256"
    if not original_record.is_file():
        _fail(f"missing original execution code record: {original_record}")
    original_recorded = original_record.read_text(encoding="utf-8").strip()
    if original_recorded != expected_common_runtime_sha256:
        _fail("experiment original execution code SHA256 mismatch")

    rollout_record = experiment_root / "COMMON_EPOCH0_ROLLOUT_EXECUTION_CODE_SHA256"
    continuation_record = experiment_root / "CONTINUATION_EXECUTION_CODE_SHA256"
    migration_path = experiment_root / "POSTPROCESS_CODE_MIGRATION.env"
    if expected_common_runtime_sha256 == expected_continuation_sha256:
        if continuation_record.is_file():
            observed = continuation_record.read_text(encoding="utf-8").strip()
            if observed != expected_continuation_sha256:
                _fail("continuation execution code SHA256 mismatch")
        migration_sha256 = None
    else:
        for path, expected, label in (
            (rollout_record, expected_common_runtime_sha256, "common rollout"),
            (continuation_record, expected_continuation_sha256, "continuation"),
        ):
            if not path.is_file() or path.read_text(encoding="utf-8").strip() != expected:
                _fail(f"{label} execution code SHA256 record mismatch: {path}")
        migration = _load_env(migration_path)
        _require_value(
            migration,
            "DEEPSEEK_BATCH64_COMMON_ROLLOUT_EXECUTION_CODE_SHA256",
            expected_common_runtime_sha256,
            "postprocess migration metadata",
        )
        _require_value(
            migration,
            "DEEPSEEK_BATCH64_CONTINUATION_EXECUTION_CODE_SHA256",
            expected_continuation_sha256,
            "postprocess migration metadata",
        )
        migration_sha256 = _sha256(migration_path)
    return {
        "common_rollout_execution_code_sha256": expected_common_runtime_sha256,
        "continuation_execution_code_sha256": expected_continuation_sha256,
        "migration_metadata_path": str(migration_path) if migration_path.is_file() else None,
        "migration_metadata_sha256": migration_sha256,
    }


def _audit_run_contract(
    common_root: Path,
    metadata: dict[str, str],
    expected_model_path: Path,
    expected_model_revision: str,
    expected_distcp_path: Path,
    expected_train_file: Path,
    expected_test_file: Path,
    expected_workload_profile_id: str,
    expected_workload_profile_sha256: str,
    expected_common_runtime_sha256: str,
) -> dict[str, Any]:
    path = common_root / RUN_CONTRACT_FILENAME
    contract = _load_env(path)
    mode = path.stat().st_mode & 0o777
    if mode & 0o222:
        _fail(f"common run contract must be read-only, mode={mode:o}: {path}")

    expected_model_path = expected_model_path.expanduser().resolve()
    expected_distcp_path = expected_distcp_path.expanduser().resolve()
    expected_train_file = expected_train_file.expanduser().resolve()
    expected_test_file = expected_test_file.expanduser().resolve()
    for expected, label in (
        (expected_model_path, "model"),
        (expected_distcp_path, "distributed checkpoint"),
        (expected_train_file, "training dataset"),
        (expected_test_file, "test dataset"),
    ):
        if not expected.exists():
            _fail(f"current expected {label} input does not exist: {expected}")
    if not expected_train_file.is_file() or not expected_test_file.is_file():
        _fail("current expected train and test datasets must be regular files")
    if not expected_model_revision:
        _fail("current expected model revision is empty")

    _require_value(
        contract,
        "COMMON_EPOCH0_RUN_CONTRACT_SCHEMA_VERSION",
        1,
        "common run contract",
    )
    for key, expected, label in (
        (
            "COMMON_EPOCH0_RUN_CONTRACT_MODEL_PATH",
            expected_model_path,
            "common run contract model",
        ),
        (
            "COMMON_EPOCH0_RUN_CONTRACT_DISTCP_PATH",
            expected_distcp_path,
            "common run contract distributed checkpoint",
        ),
        (
            "COMMON_EPOCH0_RUN_CONTRACT_TRAIN_FILE",
            expected_train_file,
            "common run contract training dataset",
        ),
        (
            "COMMON_EPOCH0_RUN_CONTRACT_TEST_FILE",
            expected_test_file,
            "common run contract test dataset",
        ),
    ):
        _require_path(_required(contract, key, "common run contract"), expected, label)
    for key, expected in (
        ("COMMON_EPOCH0_RUN_CONTRACT_MODEL_REVISION", expected_model_revision),
        (
            "COMMON_EPOCH0_RUN_CONTRACT_WORKLOAD_PROFILE_ID",
            expected_workload_profile_id,
        ),
        (
            "COMMON_EPOCH0_RUN_CONTRACT_WORKLOAD_PROFILE_SHA256",
            expected_workload_profile_sha256,
        ),
        (
            "COMMON_EPOCH0_RUN_CONTRACT_EXECUTION_CODE_SHA256",
            expected_common_runtime_sha256,
        ),
    ):
        _require_value(contract, key, expected, "common run contract")

    train_sha256 = _sha256(expected_train_file)
    test_sha256 = _sha256(expected_test_file)
    for key, expected, label in (
        (
            "COMMON_EPOCH0_RUN_CONTRACT_TRAIN_FILE_SHA256",
            train_sha256,
            "training dataset SHA256",
        ),
        (
            "COMMON_EPOCH0_RUN_CONTRACT_TEST_FILE_SHA256",
            test_sha256,
            "test dataset SHA256",
        ),
    ):
        observed = _validate_sha256(
            _required(contract, key, "common run contract"), label
        )
        if observed != expected:
            _fail(f"common run contract {label} mismatch")

    contract_sha256 = _sha256(path)
    for key, expected in (
        ("COMMON_EPOCH0_MODEL_PATH", expected_model_path),
        ("COMMON_EPOCH0_DISTCP_PATH", expected_distcp_path),
        ("COMMON_EPOCH0_TRAIN_FILE_USED", expected_train_file),
        ("COMMON_EPOCH0_TEST_FILE_USED", expected_test_file),
    ):
        _require_path(_required(metadata, key, "common metadata"), expected, key)
    for key, expected in (
        ("COMMON_EPOCH0_MODEL_REVISION", expected_model_revision),
        ("COMMON_EPOCH0_TRAIN_FILE_SHA256", train_sha256),
        ("COMMON_EPOCH0_TEST_FILE_SHA256", test_sha256),
        ("COMMON_EPOCH0_RUN_CONTRACT_SHA256", contract_sha256),
    ):
        _require_value(metadata, key, expected, "common metadata")

    return {
        "path": str(path),
        "sha256": contract_sha256,
        "mode": f"{mode:03o}",
        "model_path": str(expected_model_path),
        "model_revision": expected_model_revision,
        "distcp_path": str(expected_distcp_path),
        "workload_profile_id": expected_workload_profile_id,
        "workload_profile_sha256": expected_workload_profile_sha256,
        "execution_code_sha256": expected_common_runtime_sha256,
        "train_file": str(expected_train_file),
        "train_file_sha256": train_sha256,
        "test_file": str(expected_test_file),
        "test_file_sha256": test_sha256,
    }


def audit_common_epoch0(
    *,
    common_root: Path,
    expected_steps: int,
    expected_batch_size: int,
    expected_rollout_n: int,
    expected_workload_profile_id: str,
    expected_workload_profile_sha256: str,
    expected_common_runtime_sha256: str,
    expected_continuation_sha256: str,
    expected_model_path: Path = DEFAULT_MODEL_PATH,
    expected_model_revision: str = DEFAULT_MODEL_REVISION,
    expected_distcp_path: Path = DEFAULT_DISTCP_PATH,
    expected_train_file: Path = DEFAULT_TRAIN_FILE,
    expected_test_file: Path = DEFAULT_TEST_FILE,
    expected_unique_prompts: int | None = None,
    expected_duplicate_occurrences: int | None = None,
    expected_duplicate_policy: str = "latest_occurrence",
    expected_preemption_policy: str = "record",
    expected_preemption_count: int | None = None,
    expected_measured_kv_tokens: int | None = None,
    expected_distcp_count: int | None = None,
    block_size: int = 128,
    max_response_length: int | None = None,
    max_clip_ratio: float | None = None,
    min_distinct_prompt_maxima: int | None = None,
) -> dict[str, Any]:
    common_root = common_root.resolve()
    if expected_steps <= 0 or expected_batch_size <= 0 or expected_rollout_n <= 0:
        _fail("expected steps, batch size, and rollout n must be positive")
    if block_size <= 0:
        _fail("block size must be positive")
    if max_response_length is not None and max_response_length <= 0:
        _fail("maximum response length must be positive")
    if max_clip_ratio is not None and not 0.0 <= max_clip_ratio <= 1.0:
        _fail("maximum clip ratio must be within [0, 1]")
    if min_distinct_prompt_maxima is not None and min_distinct_prompt_maxima <= 0:
        _fail("minimum distinct prompt maxima must be positive")
    if expected_preemption_policy not in {"record", "forbid"}:
        _fail("expected preemption policy must be record or forbid")
    _validate_sha256(expected_workload_profile_sha256, "workload profile SHA256")
    _validate_sha256(expected_common_runtime_sha256, "common runtime SHA256")
    _validate_sha256(expected_continuation_sha256, "continuation SHA256")
    if not common_root.is_dir():
        _fail(f"common epoch0 root does not exist: {common_root}")
    marker = common_root / COMPLETION_MARKER
    if not marker.is_file():
        _fail(f"common epoch0 completion marker is missing: {marker}")
    if (common_root / "INCOMPLETE").exists():
        _fail("common epoch0 still has an INCOMPLETE marker")

    metadata_path = common_root / "common_epoch0_metadata.env"
    reuse_path = common_root / "reuse.env"
    metadata = _load_env(metadata_path)
    reuse = _load_env(reuse_path)
    expected_prompts = expected_steps * expected_batch_size
    expected_responses = expected_batch_size * expected_rollout_n
    for key, expected in (
        ("COMMON_EPOCH0_TRAIN_STEPS_USED", expected_steps),
        ("COMMON_EPOCH0_TRAIN_BATCH_SIZE_USED", expected_batch_size),
        ("COMMON_EPOCH0_ROLLOUT_N_USED", expected_rollout_n),
        ("COMMON_EPOCH0_PROMPTS_TOTAL_USED", expected_prompts),
        ("COMMON_EPOCH0_EXPECTED_RESPONSES_PER_STEP_USED", expected_responses),
        ("COMMON_EPOCH0_WORKLOAD_PROFILE_ID", expected_workload_profile_id),
        ("COMMON_EPOCH0_WORKLOAD_PROFILE_SHA256", expected_workload_profile_sha256),
        ("COMMON_EPOCH0_PREEMPTION_POLICY_USED", expected_preemption_policy),
    ):
        _require_value(metadata, key, expected, "common metadata")

    epoch_dir = Path(
        _required(reuse, "DYNAMIC_INITIAL_BASELINE_DIR", "common reuse metadata")
    ).expanduser().resolve()
    expected_epoch_dir = common_root / "epoch_000_mode0_probe"
    _require_path(str(epoch_dir), expected_epoch_dir, "common rollout history")
    baseline_checkpoint = Path(
        _required(reuse, "BASELINE_INITIAL_RESUME_CKPT", "common reuse metadata")
    ).expanduser().resolve()
    dynamic_checkpoint = Path(
        _required(reuse, "DYNAMIC_INITIAL_RESUME_CKPT", "common reuse metadata")
    ).expanduser().resolve()
    if baseline_checkpoint != dynamic_checkpoint:
        _fail("baseline and AdaFloor reuse checkpoints differ")
    checkpoint = dynamic_checkpoint

    measured_path = common_root / MEASURED_KV_FILENAME
    if not measured_path.is_file():
        _fail(f"measured KV capacity is missing: {measured_path}")
    try:
        measured_kv = int(measured_path.read_text(encoding="utf-8").strip())
    except ValueError as exc:
        raise AuditError(f"invalid measured KV capacity: {measured_path}") from exc
    if measured_kv <= 0 or measured_kv % block_size:
        _fail(f"measured KV capacity {measured_kv} is not a positive block multiple")
    if expected_measured_kv_tokens is not None and measured_kv != expected_measured_kv_tokens:
        _fail(
            f"measured KV capacity is {measured_kv}, "
            f"expected {expected_measured_kv_tokens}"
        )
    _require_value(
        metadata,
        "COMMON_EPOCH0_EFFECTIVE_KV_TOKENS_PER_RANK",
        measured_kv,
        "common metadata",
    )

    bos_token_id = _chat_bos_token_id(expected_model_path)
    artifacts = _audit_artifacts_and_history(
        epoch_dir,
        expected_steps,
        expected_batch_size,
        expected_rollout_n,
        bos_token_id,
        expected_unique_prompts,
        expected_duplicate_occurrences,
        expected_duplicate_policy,
        max_response_length,
        max_clip_ratio,
        min_distinct_prompt_maxima,
    )
    metadata_preemption_count = _required_int(
        metadata, "COMMON_EPOCH0_PREEMPTION_COUNT", "common metadata"
    )
    if (
        expected_preemption_count is not None
        and metadata_preemption_count != expected_preemption_count
    ):
        _fail("common metadata preemption count does not match the expected count")
    log = _audit_log(
        epoch_dir,
        expected_steps,
        expected_preemption_policy,
        metadata_preemption_count,
    )
    checkpoint_summary = _audit_checkpoint(
        checkpoint, expected_steps, expected_distcp_count
    )
    runtime = _audit_runtime_provenance(
        common_root,
        metadata,
        expected_common_runtime_sha256,
        expected_continuation_sha256,
    )
    run_contract = _audit_run_contract(
        common_root,
        metadata,
        expected_model_path,
        expected_model_revision,
        expected_distcp_path,
        expected_train_file,
        expected_test_file,
        expected_workload_profile_id,
        expected_workload_profile_sha256,
        expected_common_runtime_sha256,
    )

    marker_text = marker.read_text(encoding="utf-8")
    if str(epoch_dir) not in marker_text or str(checkpoint) not in marker_text:
        _fail("common completion marker does not name the audited history and checkpoint")
    return {
        "schema_version": 1,
        "status": "PASS",
        "common_epoch0_root": str(common_root),
        "contract": {
            "steps": expected_steps,
            "train_batch_size": expected_batch_size,
            "rollout_n": expected_rollout_n,
            "prompt_occurrences": expected_prompts,
            "responses_per_step": expected_responses,
            "workload_profile_id": expected_workload_profile_id,
            "workload_profile_sha256": expected_workload_profile_sha256,
            "block_size": block_size,
        },
        "runtime_provenance": runtime,
        "run_contract": run_contract,
        "completion_marker": {
            "path": str(marker),
            "sha256": _sha256(marker),
        },
        "reuse_env": {"path": str(reuse_path), "sha256": _sha256(reuse_path)},
        "metadata_env": {
            "path": str(metadata_path),
            "sha256": _sha256(metadata_path),
        },
        "measured_kv": {
            "path": str(measured_path),
            "sha256": _sha256(measured_path),
            "tokens_per_rank": measured_kv,
        },
        "rollout_artifacts": artifacts["files"],
        "offline_planning_history": artifacts["history"],
        "rollout_quality": artifacts["quality"],
        "training_log": log,
        "checkpoint": checkpoint_summary,
    }


def _write_json_atomic(path: Path, payload: dict[str, Any], force: bool) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not force:
        _fail(f"refusing to overwrite existing recovery manifest: {path}")
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.tmp.",
            delete=False,
        ) as handle:
            temporary_name = handle.name
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        if path.exists() and not force:
            _fail(f"refusing to overwrite existing recovery manifest: {path}")
        os.replace(temporary_name, path)
        temporary_name = None
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)


def _positive_int(value: str) -> int:
    converted = int(value)
    if converted <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return converted


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--common-root", required=True, type=Path)
    parser.add_argument("--expected-steps", required=True, type=_positive_int)
    parser.add_argument("--expected-batch-size", required=True, type=_positive_int)
    parser.add_argument("--expected-rollout-n", required=True, type=_positive_int)
    parser.add_argument("--expected-workload-profile-id", required=True)
    parser.add_argument("--expected-workload-profile-sha256", required=True)
    parser.add_argument("--expected-common-runtime-sha256", required=True)
    parser.add_argument("--expected-continuation-sha256", required=True)
    parser.add_argument(
        "--expected-model-path", type=Path, default=DEFAULT_MODEL_PATH
    )
    parser.add_argument(
        "--expected-model-revision", default=DEFAULT_MODEL_REVISION
    )
    parser.add_argument(
        "--expected-distcp-path", type=Path, default=DEFAULT_DISTCP_PATH
    )
    parser.add_argument(
        "--expected-train-file", type=Path, default=DEFAULT_TRAIN_FILE
    )
    parser.add_argument(
        "--expected-test-file", type=Path, default=DEFAULT_TEST_FILE
    )
    parser.add_argument("--expected-unique-prompts", type=_positive_int)
    parser.add_argument("--expected-duplicate-occurrences", type=int)
    parser.add_argument(
        "--expected-duplicate-policy", default="latest_occurrence"
    )
    parser.add_argument(
        "--expected-preemption-policy", choices=("record", "forbid"), default="record"
    )
    parser.add_argument("--expected-preemption-count", type=int)
    parser.add_argument("--expected-measured-kv-tokens", type=_positive_int)
    parser.add_argument("--expected-distcp-count", type=_positive_int)
    parser.add_argument("--block-size", type=_positive_int, default=128)
    parser.add_argument("--max-response-length", type=_positive_int)
    parser.add_argument("--max-clip-ratio", type=float)
    parser.add_argument("--min-distinct-prompt-maxima", type=_positive_int)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    for label in ("expected duplicate occurrences", "expected preemption count"):
        attribute = label.replace(" ", "_")
        value = getattr(args, attribute)
        if value is not None and value < 0:
            parser.error(f"{label} must be nonnegative")
    try:
        payload = audit_common_epoch0(
            common_root=args.common_root,
            expected_steps=args.expected_steps,
            expected_batch_size=args.expected_batch_size,
            expected_rollout_n=args.expected_rollout_n,
            expected_workload_profile_id=args.expected_workload_profile_id,
            expected_workload_profile_sha256=args.expected_workload_profile_sha256,
            expected_common_runtime_sha256=args.expected_common_runtime_sha256,
            expected_continuation_sha256=args.expected_continuation_sha256,
            expected_model_path=args.expected_model_path,
            expected_model_revision=args.expected_model_revision,
            expected_distcp_path=args.expected_distcp_path,
            expected_train_file=args.expected_train_file,
            expected_test_file=args.expected_test_file,
            expected_unique_prompts=args.expected_unique_prompts,
            expected_duplicate_occurrences=args.expected_duplicate_occurrences,
            expected_duplicate_policy=args.expected_duplicate_policy,
            expected_preemption_policy=args.expected_preemption_policy,
            expected_preemption_count=args.expected_preemption_count,
            expected_measured_kv_tokens=args.expected_measured_kv_tokens,
            expected_distcp_count=args.expected_distcp_count,
            block_size=args.block_size,
            max_response_length=args.max_response_length,
            max_clip_ratio=args.max_clip_ratio,
            min_distinct_prompt_maxima=args.min_distinct_prompt_maxima,
        )
        if args.output is not None:
            _write_json_atomic(args.output, payload, args.force)
            print(f"PASS recovery_manifest={args.output.resolve()}")
        else:
            print(json.dumps(payload, indent=2, sort_keys=True))
    except AuditError as exc:
        parser.exit(2, f"audit failed: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
