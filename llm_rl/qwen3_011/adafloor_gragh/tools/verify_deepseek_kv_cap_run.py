#!/usr/bin/env python3
"""Authorize lifecycle-specific DeepSeek KV caps with strict one-step runs."""

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

from audit_deepseek_n_f4_formal_run import (
    AuditError,
    DEFAULT_LIFECYCLE,
    LIFECYCLE_CONFIG,
    _validate_lifecycle,
    _validate_resize_calls,
    _runtime_profile_sha256,
    lifecycle_config,
    load_plans,
    parse_runtime_log,
    validate_plans,
    validate_training_health,
)
from hash_deepseek_execution_code import digest as execution_digest


ROOT = Path(__file__).resolve().parents[1]
FLOORS = tuple(LIFECYCLE_CONFIG[DEFAULT_LIFECYCLE]["floors"])
WORLD_SIZE = 16
BLOCK_SIZE = 128
RELEASE_AREA_UNIT = "rank_token_proxy"
DEEPSEEK_EOS_TOKEN_ID = 100001
EXPECTED_RESPONSES = 32 * 16
RUNTIME_PROFILE_PATH = Path(
    LIFECYCLE_CONFIG[DEFAULT_LIFECYCLE]["runtime_profile_path"]
)
PROFILE_ID_RE = re.compile(
    r"^export DEEPSEEK_N_F4_RUNTIME_PROFILE_ID=([^\s]+)$", re.M
)
COMMON_PROTOCOL = {
    "COMMON_EPOCH0_TRAIN_FILE_USED": "/data/deepscaler/train.parquet",
    "COMMON_EPOCH0_TEST_FILE_USED": "/data/deepscaler/test.parquet",
    "COMMON_EPOCH0_DATASET_FRACTION_USED": "0.005",
    "COMMON_EPOCH0_TRAIN_BATCH_SIZE_USED": "32",
    "COMMON_EPOCH0_ROLLOUT_N_USED": "16",
    "COMMON_EPOCH0_MAX_PROMPT_LENGTH_USED": "1024",
    "COMMON_EPOCH0_MAX_RESPONSE_LENGTH_USED": "16384",
    "COMMON_EPOCH0_MAX_NUM_BATCHED_TOKENS_USED": "17408",
    "COMMON_EPOCH0_MAX_NUM_SEQS_USED": "32",
    "COMMON_EPOCH0_GPU_MEMORY_UTILIZATION_USED": "0.9",
    "COMMON_EPOCH0_KV_BLOCK_SIZE_USED": "128",
    "COMMON_EPOCH0_TRAIN_STEPS_USED": "5",
}
WORKLOAD_FIELDS = {
    "train_batch_size": (
        "DEEPSEEK_KV_CAP_TRAIN_BATCH_SIZE",
        "COMMON_EPOCH0_TRAIN_BATCH_SIZE_USED",
        32,
    ),
    "rollout_n": (
        "DEEPSEEK_KV_CAP_ROLLOUT_N",
        "COMMON_EPOCH0_ROLLOUT_N_USED",
        16,
    ),
    "expected_responses": (
        "DEEPSEEK_KV_CAP_EXPECTED_RESPONSES_PER_STEP",
        "COMMON_EPOCH0_EXPECTED_RESPONSES_PER_STEP_USED",
        EXPECTED_RESPONSES,
    ),
    "max_num_seqs": (
        "DEEPSEEK_KV_CAP_MAX_NUM_SEQS",
        "COMMON_EPOCH0_MAX_NUM_SEQS_USED",
        32,
    ),
    "max_response_length": (
        "DEEPSEEK_KV_CAP_MAX_RESPONSE_LENGTH",
        "COMMON_EPOCH0_MAX_RESPONSE_LENGTH_USED",
        16384,
    ),
}
PROFILE_FIELDS = {
    "workload_profile_id": (
        "DEEPSEEK_KV_CAP_WORKLOAD_PROFILE_ID",
        "COMMON_EPOCH0_WORKLOAD_PROFILE_ID",
    ),
    "workload_profile_sha256": (
        "DEEPSEEK_KV_CAP_WORKLOAD_PROFILE_SHA256",
        "COMMON_EPOCH0_WORKLOAD_PROFILE_SHA256",
    ),
}
TAIL_GUARD_APPLIED_RE = re.compile(
    r"Shrink-aware tail-guard response cap:\s*"
    r"selected_floor=(?P<floor>\d+)\s+plan_cap=(?P<cap>\d+)\b"
)
INFER_START_RE = re.compile(
    r"rollout_worker_infer_start\s+rank=(?P<rank>\d+)\s+"
    r"step=(?P<step>\d+)\s+epoch=-?\d+\s+"
    r"batch_size=(?P<batch_size>\d+)\s+max_tokens=(?P<max_tokens>\d+)"
)


class AuthorizationError(RuntimeError):
    """Raised when any cap authorization invariant is not proven."""


def _fail(message: str) -> None:
    raise AuthorizationError(message)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_env(path: Path) -> dict[str, str]:
    if not path.is_file():
        _fail(f"environment file does not exist: {path}")
    values: dict[str, str] = {}
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].strip()
        key, separator, raw_value = line.partition("=")
        if not separator:
            _fail(f"{path}:{line_number} is not an environment assignment")
        try:
            parsed = shlex.split(raw_value, comments=True, posix=True)
        except ValueError as exc:
            raise AuthorizationError(f"cannot parse {path}:{line_number}: {exc}") from exc
        if len(parsed) > 1:
            _fail(f"{path}:{line_number} contains multiple shell words")
        values[key.strip()] = parsed[0] if parsed else ""
    return values


def _resolved(value: str) -> Path:
    return Path(value).expanduser().resolve()


def _parse_positive_int(value: str | int, label: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise AuthorizationError(f"{label} must be an integer") from exc
    if parsed <= 0:
        _fail(f"{label} must be positive")
    return parsed


def _resolve_workload(
    values: dict[str, str],
    metadata: dict[str, str],
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    overrides = overrides or {}
    workload: dict[str, Any] = {}
    for field, (cap_key, metadata_key, default) in WORKLOAD_FIELDS.items():
        candidates: list[tuple[str, str | int]] = []
        if cap_key in values:
            candidates.append((cap_key, values[cap_key]))
        if metadata_key in metadata:
            candidates.append((metadata_key, metadata[metadata_key]))
        if overrides.get(field) is not None:
            candidates.append((f"CLI {field}", overrides[field]))
        parsed = [
            (label, _parse_positive_int(raw, label))
            for label, raw in candidates
        ]
        if field == "expected_responses" and not parsed:
            workload[field] = None
            continue
        resolved = parsed[0][1] if parsed else int(default)
        for label, candidate in parsed[1:]:
            if candidate != resolved:
                _fail(
                    f"workload mismatch for {field}: {parsed[0][0]}="
                    f"{resolved}, {label}={candidate}"
                )
        workload[field] = resolved

    derived_responses = workload["train_batch_size"] * workload["rollout_n"]
    if workload["expected_responses"] is None:
        workload["expected_responses"] = derived_responses
    if workload["expected_responses"] != derived_responses:
        _fail(
            "expected responses per step must equal train batch size times "
            f"rollout n, got {workload['expected_responses']} versus "
            f"{derived_responses}"
        )
    if workload["max_num_seqs"] < workload["train_batch_size"]:
        _fail("max_num_seqs must be at least train_batch_size")

    for field, (cap_key, metadata_key) in PROFILE_FIELDS.items():
        candidates: list[tuple[str, str]] = []
        for label, raw in (
            (cap_key, values.get(cap_key)),
            (metadata_key, metadata.get(metadata_key)),
            (f"CLI {field}", overrides.get(field)),
        ):
            if raw is not None and str(raw).strip() not in ("", "unspecified"):
                candidates.append((label, str(raw).strip()))
        resolved = candidates[0][1] if candidates else None
        for label, candidate in candidates[1:]:
            if candidate != resolved:
                _fail(
                    f"workload mismatch for {field}: {candidates[0][0]}="
                    f"{resolved}, {label}={candidate}"
                )
        workload[field] = resolved

    profile_id = workload["workload_profile_id"]
    profile_sha256 = workload["workload_profile_sha256"]
    if (profile_id is None) != (profile_sha256 is None):
        _fail("workload profile ID and SHA256 must be provided together")
    if profile_sha256 is not None and not re.fullmatch(r"[0-9a-fA-F]{64}", profile_sha256):
        _fail("workload profile SHA256 must contain 64 hexadecimal characters")
    return workload


def _candidate_caps(
    values: dict[str, str], lifecycle: str = DEFAULT_LIFECYCLE
) -> tuple[dict[int, int], dict[int, int]]:
    config = lifecycle_config(lifecycle)
    prefix = str(config["prefix"])
    floors = tuple(config["floors"])
    verified_key = f"{prefix}_KV_CAPS_VERIFIED"
    if values.get(verified_key) != "0":
        _fail(f"candidate {verified_key} must equal 0")
    if values.get("DEEPSEEK_KV_CAP_TARGET_RATIO") != "1.0":
        _fail("candidate DEEPSEEK_KV_CAP_TARGET_RATIO must equal 1.0")
    if values.get("DEEPSEEK_KV_CAP_BLOCK_SIZE") != str(BLOCK_SIZE):
        _fail(f"candidate block size must equal {BLOCK_SIZE}")

    admission: dict[int, int] = {}
    physical: dict[int, int] = {}
    for floor in floors:
        try:
            admission[floor] = int(values[f"{prefix}_KV_ADMISSION_FLOOR{floor}"])
            physical[floor] = int(values[f"{prefix}_KV_PHYSICAL_FLOOR{floor}"])
        except (KeyError, ValueError) as exc:
            raise AuthorizationError(f"missing or invalid floor{floor} candidate cap") from exc
        for label, cap in (("admission", admission[floor]), ("physical", physical[floor])):
            if cap <= 0 or cap % BLOCK_SIZE:
                _fail(f"floor{floor} {label} cap must be a positive multiple of {BLOCK_SIZE}")
        if admission[floor] >= physical[floor]:
            _fail(f"floor{floor} admission cap must be below physical cap")
    if str(config["policy"]) == "planned":
        for floor in floors:
            key = f"{prefix}_HEADROOM_FLOOR{floor}"
            try:
                headroom = int(values[key])
            except (KeyError, ValueError) as exc:
                raise AuthorizationError(
                    f"missing or invalid Planned headroom {key}"
                ) from exc
            if headroom < 0 or headroom % BLOCK_SIZE:
                _fail(
                    f"{key} must be a nonnegative multiple of {BLOCK_SIZE}"
                )
        training_key = f"{prefix}_TRAINING_MIN_FREE_MIB"
        try:
            training_min_free_mib = int(values[training_key])
        except (KeyError, ValueError) as exc:
            raise AuthorizationError(
                f"missing or invalid Planned training reserve {training_key}"
            ) from exc
        if training_min_free_mib <= 0:
            _fail(f"{training_key} must be positive")
    return admission, physical


def _require_same_path(recorded: str | None, expected: Path, label: str) -> None:
    if not recorded or _resolved(recorded) != expected.resolve():
        _fail(f"{label} provenance mismatch")


def _validate_provenance(
    values: dict[str, str],
    common_root: Path,
    trigger_root: Path,
    lifecycle: str = DEFAULT_LIFECYCLE,
    workload_overrides: dict[str, Any] | None = None,
    expected_runtime_execution_sha256: str | None = None,
    expected_verification_code_sha256: str | None = None,
) -> tuple[dict[str, str], dict[str, Any]]:
    config = lifecycle_config(lifecycle)
    prefix = str(config["prefix"])
    common_root = common_root.resolve()
    trigger_root = trigger_root.resolve()
    _require_same_path(
        values.get("DEEPSEEK_KV_CAP_COMMON_EPOCH0_ROOT"), common_root, "common epoch0"
    )
    _require_same_path(
        values.get("DEEPSEEK_KV_CAP_PROBE_HISTORY_ROOT"), trigger_root, "trigger history"
    )

    metadata = _load_env(common_root / "common_epoch0_metadata.env")
    reuse = _load_env(common_root / "reuse.env")
    if not (common_root / "DO_NOT_DELETE_COMMON_EPOCH0_CHECKPOINT").is_file():
        _fail("common epoch0 completion marker is missing")
    workload = _resolve_workload(values, metadata, workload_overrides)
    protocol = dict(COMMON_PROTOCOL)
    protocol.update(
        {
            "COMMON_EPOCH0_DATASET_FRACTION_USED": values.get(
                "DEEPSEEK_KV_CAP_DATASET_FRACTION",
                protocol["COMMON_EPOCH0_DATASET_FRACTION_USED"],
            ),
            "COMMON_EPOCH0_TRAIN_BATCH_SIZE_USED": str(
                workload["train_batch_size"]
            ),
            "COMMON_EPOCH0_ROLLOUT_N_USED": str(workload["rollout_n"]),
            "COMMON_EPOCH0_MAX_PROMPT_LENGTH_USED": values.get(
                "DEEPSEEK_KV_CAP_MAX_PROMPT_LENGTH",
                protocol["COMMON_EPOCH0_MAX_PROMPT_LENGTH_USED"],
            ),
            "COMMON_EPOCH0_MAX_RESPONSE_LENGTH_USED": str(
                workload["max_response_length"]
            ),
            "COMMON_EPOCH0_MAX_NUM_BATCHED_TOKENS_USED": values.get(
                "DEEPSEEK_KV_CAP_MAX_NUM_BATCHED_TOKENS",
                protocol["COMMON_EPOCH0_MAX_NUM_BATCHED_TOKENS_USED"],
            ),
            "COMMON_EPOCH0_MAX_NUM_SEQS_USED": str(workload["max_num_seqs"]),
            "COMMON_EPOCH0_GPU_MEMORY_UTILIZATION_USED": values.get(
                "DEEPSEEK_KV_CAP_GPU_MEMORY_UTILIZATION",
                protocol["COMMON_EPOCH0_GPU_MEMORY_UTILIZATION_USED"],
            ),
            "COMMON_EPOCH0_KV_BLOCK_SIZE_USED": values.get(
                "DEEPSEEK_KV_CAP_BLOCK_SIZE",
                protocol["COMMON_EPOCH0_KV_BLOCK_SIZE_USED"],
            ),
            "COMMON_EPOCH0_TRAIN_STEPS_USED": values.get(
                "DEEPSEEK_KV_CAP_COMMON_STEPS",
                protocol["COMMON_EPOCH0_TRAIN_STEPS_USED"],
            ),
        }
    )
    if workload["workload_profile_id"] is not None:
        protocol.update(
            {
                "COMMON_EPOCH0_EXPECTED_RESPONSES_PER_STEP_USED": str(
                    workload["expected_responses"]
                ),
                "COMMON_EPOCH0_WORKLOAD_PROFILE_ID": workload[
                    "workload_profile_id"
                ],
                "COMMON_EPOCH0_WORKLOAD_PROFILE_SHA256": workload[
                    "workload_profile_sha256"
                ],
            }
        )
    if "DEEPSEEK_KV_CAP_PROMPTS_TOTAL" in values:
        protocol["COMMON_EPOCH0_PROMPTS_TOTAL_USED"] = values[
            "DEEPSEEK_KV_CAP_PROMPTS_TOTAL"
        ]
    for name, expected in protocol.items():
        recorded = metadata.get(name)
        if name.endswith("_FILE_USED"):
            if not recorded or _resolved(recorded) != Path(expected).resolve():
                _fail(f"common epoch0 protocol mismatch for {name}")
        elif recorded != expected:
            _fail(f"common epoch0 protocol mismatch for {name}")
    checkpoint = _resolved(reuse.get("DYNAMIC_INITIAL_RESUME_CKPT", ""))
    if not (checkpoint / "actor").is_dir() or not (checkpoint / ".PRESERVE_COMMON_EPOCH0").is_file():
        _fail(f"common epoch0 checkpoint is incomplete: {checkpoint}")

    trigger_files = {
        "history": trigger_root / "offline_planning_history.json",
        "manifest": trigger_root / "kv_probe_trigger_manifest.json",
        "subset": trigger_root / "rollout_data" / "1.jsonl",
    }
    trigger_keys = {
        "history": "DEEPSEEK_KV_CAP_PROBE_HISTORY_SHA256",
        "manifest": "DEEPSEEK_KV_CAP_PROBE_HISTORY_MANIFEST_SHA256",
        "subset": "DEEPSEEK_KV_CAP_PROBE_TRIGGER_SUBSET_SHA256",
    }
    trigger_hashes: dict[str, str] = {}
    for label, path in trigger_files.items():
        if not path.is_file():
            _fail(f"trigger {label} artifact is missing: {path}")
        trigger_hashes[label] = _sha256(path)
        if values.get(trigger_keys[label]) != trigger_hashes[label]:
            _fail(f"trigger {label} SHA256 mismatch")

    runtime_profile_path = Path(config["runtime_profile_path"])
    profile_id_key = str(config["runtime_profile_id_key"])
    profile_text = runtime_profile_path.read_text(encoding="utf-8")
    profile_match = re.search(
        rf"^export {re.escape(profile_id_key)}=([^\s]+)$", profile_text, re.M
    )
    if profile_match is None:
        _fail(f"{config['label']} runtime profile ID is missing")
    if values.get(f"{prefix}_RUNTIME_PROFILE") != profile_match.group(1):
        _fail(f"{config['label']} runtime profile ID mismatch")
    profile_sha256 = _runtime_profile_sha256(config["runtime_profile_files"])
    if values.get(f"{prefix}_RUNTIME_PROFILE_SHA256") != profile_sha256:
        _fail(f"{config['label']} runtime profile SHA256 mismatch")
    verification_code_sha256, _count = execution_digest(ROOT)
    runtime_execution_sha256 = values.get("DEEPSEEK_EXECUTION_CODE_SHA256")
    if expected_runtime_execution_sha256 is None and expected_verification_code_sha256 is None:
        if runtime_execution_sha256 != verification_code_sha256:
            _fail("DeepSeek execution code SHA256 mismatch")
    else:
        if not expected_runtime_execution_sha256 or not expected_verification_code_sha256:
            _fail("code migration requires both runtime and verification SHA256 values")
        for label, value in (
            ("runtime execution", expected_runtime_execution_sha256),
            ("verification code", expected_verification_code_sha256),
        ):
            if re.fullmatch(r"[0-9a-f]{64}", value) is None:
                _fail(f"invalid expected {label} SHA256")
        if runtime_execution_sha256 != expected_runtime_execution_sha256:
            _fail("DeepSeek runtime execution code SHA256 mismatch")
        if verification_code_sha256 != expected_verification_code_sha256:
            _fail("DeepSeek verification code SHA256 mismatch")
    if values.get("DEEPSEEK_KV_CAP_MODEL_REVISION") != metadata.get("COMMON_EPOCH0_MODEL_REVISION"):
        _fail("DeepSeek model revision provenance mismatch")
    if values.get("DEEPSEEK_KV_CAP_EXECUTION_PROFILE") != metadata.get(
        "COMMON_EPOCH0_EXECUTION_PROFILE_USED"
    ):
        _fail("DeepSeek execution profile provenance mismatch")
    return (
        {
            "checkpoint": str(checkpoint),
            "runtime_profile_sha256": profile_sha256,
            "runtime_execution_code_sha256": runtime_execution_sha256,
            "verification_code_sha256": verification_code_sha256,
            "trigger_history_sha256": trigger_hashes["history"],
            "trigger_manifest_sha256": trigger_hashes["manifest"],
            "trigger_subset_sha256": trigger_hashes["subset"],
        },
        workload,
    )


def _only_file(directory: Path, pattern: str, label: str) -> Path:
    files = sorted(directory.glob(pattern))
    if len(files) != 1:
        _fail(f"{directory}: expected one {label}, found {len(files)}")
    return files[0]


def _line_count(path: Path) -> int:
    with path.open(encoding="utf-8") as handle:
        return sum(1 for _ in handle)


def _validate_release_window(
    plan: dict[str, Any], floor: int, max_response_length: int
) -> int:
    if plan.get("tail_guard_enabled") is not True:
        _fail(f"floor{floor} authorization plan must enable TailGuard")
    try:
        tail_guard_cap = int(plan["tail_guard_response_cap"])
    except (KeyError, TypeError, ValueError) as exc:
        raise AuthorizationError(
            f"floor{floor} authorization plan has no valid TailGuard cap"
        ) from exc
    if tail_guard_cap <= 0 or tail_guard_cap >= max_response_length:
        _fail(
            f"floor{floor} TailGuard cap must be between 1 and "
            f"{max_response_length - 1}, "
            f"got {tail_guard_cap}"
        )
    if plan.get("length_prediction_mode") != "single_epoch_prompt_max":
        _fail(
            f"floor{floor} authorization plan must use one-epoch prompt-max "
            "prediction"
        )
    release = float(plan.get("release_area", -1))
    if not math.isfinite(release) or release < 0:
        _fail(f"floor{floor} plan has invalid release area {release}")
    if plan.get("release_area_unit") not in (None, RELEASE_AREA_UNIT):
        _fail(f"floor{floor} plan has an invalid release area unit")
    if floor == 16:
        if release != 0:
            _fail("floor16 authorization plan must have zero release area")
        return tail_guard_cap
    thresholds = plan.get("schedule_thresholds")
    predicted_exit = float(plan.get("predicted_step_exit", -1))
    if not isinstance(thresholds, list) or not thresholds or release <= 0:
        _fail(f"floor{floor} authorization plan has no positive release window")
    numeric = [float(value) for value in thresholds]
    if (
        not math.isfinite(predicted_exit)
        or any(not math.isfinite(value) for value in numeric)
        or max(numeric) >= predicted_exit
    ):
        _fail(f"floor{floor} shrink threshold does not precede predicted exit")
    return tail_guard_cap


def _validate_applied_tail_guard(
    text: str, floor: int, plan_cap: int, log_path: Path
) -> int:
    observed = [
        (int(match.group("floor")), int(match.group("cap")))
        for match in TAIL_GUARD_APPLIED_RE.finditer(text)
    ]
    if not observed:
        _fail(
            f"{log_path}: no runtime evidence that the TailGuard plan cap "
            "was applied"
        )
    mismatched = [item for item in observed if item != (floor, plan_cap)]
    if mismatched:
        _fail(
            f"{log_path}: runtime TailGuard floor/cap mismatch, expected "
            f"({floor}, {plan_cap}), observed {mismatched}"
        )
    return len(observed)


def _validate_applied_sampling_cap(
    text: str,
    floor: int,
    plan_cap: int,
    train_batch_size: int,
    log_path: Path,
) -> int:
    observed = [
        {
            key: int(match.group(key))
            for key in ("rank", "step", "batch_size", "max_tokens")
        }
        for match in INFER_START_RE.finditer(text)
    ]
    if len(observed) != WORLD_SIZE:
        _fail(
            f"{log_path}: floor{floor} has {len(observed)} infer-start cap "
            f"records, expected {WORLD_SIZE}"
        )
    ranks = [item["rank"] for item in observed]
    if sorted(ranks) != list(range(WORLD_SIZE)):
        _fail(f"{log_path}: floor{floor} infer-start ranks are {sorted(ranks)}")
    invalid = [
        item
        for item in observed
        if item["step"] != 1
        or item["batch_size"] != train_batch_size
        or item["max_tokens"] != plan_cap
    ]
    if invalid:
        _fail(
            f"{log_path}: floor{floor} did not apply max_tokens={plan_cap} "
            f"to every worker, invalid={invalid[:3]}"
        )
    return len(observed)


def _response_lengths(path: Path) -> list[int]:
    lengths: list[int] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, 1):
            value = raw_line.strip()
            if not value:
                _fail(f"{path}:{line_number} is empty")
            try:
                length = int(value)
            except ValueError as exc:
                raise AuthorizationError(
                    f"{path}:{line_number} is not an integer response length"
                ) from exc
            if length < 0:
                _fail(f"{path}:{line_number} has negative response length")
            lengths.append(length)
    return lengths


def _normalize_decoded_lengths(
    rollout_path: Path,
    length_path: Path,
    response_lengths: list[int],
    response_cap: int,
) -> tuple[list[int], int]:
    """Remove only the synthetic terminal pad counted by VERL.

    DeepSeek uses EOS as its padding token.  When vLLM reaches ``max_tokens``
    without emitting EOS, VERL pads the response tensor with EOS and
    ``get_response_mask`` includes the first padding EOS.  The resulting mask
    length is ``max_tokens + 1`` even though vLLM decoded only ``max_tokens``
    tokens.  The normalization below is accepted only when the padded tensor
    proves this exact boundary pattern.
    """

    overflow_lines = {
        line_number
        for line_number, length in enumerate(response_lengths, 1)
        if length > response_cap
    }
    if any(response_lengths[line_number - 1] != response_cap + 1 for line_number in overflow_lines):
        observed = max(response_lengths)
        _fail(
            f"{length_path}: observed response length {observed} exceeds "
            f"applied TailGuard plan cap {response_cap} by more than one"
        )
    if not overflow_lines:
        return list(response_lengths), 0

    normalized = list(response_lengths)
    seen: set[int] = set()
    with rollout_path.open(encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, 1):
            if line_number not in overflow_lines:
                continue
            try:
                row = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise AuthorizationError(
                    f"{rollout_path}:{line_number} is invalid JSON"
                ) from exc
            if not isinstance(row, dict):
                _fail(f"{rollout_path}:{line_number} is not a JSON object")
            responses = row.get("responses")
            mask = row.get("response_mask")
            if (
                not isinstance(responses, list)
                or not isinstance(mask, list)
                or len(responses) != len(mask)
                or len(responses) <= response_cap + 1
                or any(value not in (0, 1) for value in mask)
                or sum(mask) != response_cap + 1
                or any(value != 1 for value in mask[: response_cap + 1])
                or any(value != 0 for value in mask[response_cap + 1 :])
            ):
                _fail(
                    f"{rollout_path}:{line_number} does not prove a single "
                    "synthetic terminal padding token"
                )
            padding_token = responses[response_cap]
            if padding_token != DEEPSEEK_EOS_TOKEN_ID or any(
                token != padding_token for token in responses[response_cap:]
            ):
                _fail(
                    f"{rollout_path}:{line_number} has non-padding response "
                    "tokens beyond the applied cap"
                )
            normalized[line_number - 1] = response_cap
            seen.add(line_number)
    if seen != overflow_lines:
        missing = sorted(overflow_lines - seen)
        _fail(f"{rollout_path} lacks rows needed to audit overflow lines {missing[:3]}")
    return normalized, len(seen)


def _validate_partial_natural_lifecycle(
    shrinks: list[Any],
    restores: list[Any],
    plan: dict[str, Any],
    last_resize_done: dict[int, int],
) -> dict[str, Any]:
    expected_sizes: list[tuple[int, int]] = []
    current_size = WORLD_SIZE
    for target in plan["stage_sets"]:
        if len(target) < WORLD_SIZE:
            expected_sizes.append((current_size, len(target)))
            current_size = len(target)
    actual_signatures = {
        (event.current_ranks, event.target_ranks)
        for event in shrinks
        if event.step == 1
    }
    actual_sizes = {
        (len(current), len(target)) for current, target in actual_signatures
    }
    completed = len(actual_sizes)
    if completed > len(expected_sizes) or actual_sizes != set(expected_sizes[:completed]):
        _fail(
            "Natural authorization transitions are not a prefix of the "
            f"planned stage sizes, expected={expected_sizes}, "
            f"actual={sorted(actual_sizes)}"
        )
    partial_plan = {
        1: {
            "floor": plan["floor"],
            "stages": tuple(plan["stages"][:completed]),
            "stage_sets": tuple(plan["stage_sets"][:completed]),
        }
    }
    evidence = _validate_lifecycle(
        shrinks,
        restores,
        partial_plan,
        last_resize_done,
        rank_identity_known=False,
    )[1]
    evidence["completed_transition_count"] = completed
    evidence["planned_transition_count"] = len(expected_sizes)
    return evidence


def _validate_calibration_lifecycle(
    values: dict[str, str],
    prefix: str,
    floor: int,
    common_root: Path,
    trigger_root: Path,
    lifecycle: str,
    physical_cap: int,
) -> dict[str, Any]:
    key = f"{prefix}_KV_PROBE_FLOOR{floor}"
    summary_path = _resolved(values.get(key, ""))
    if not summary_path.is_file():
        _fail(f"floor{floor} calibration summary is missing: {summary_path}")
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise AuthorizationError(
            f"floor{floor} calibration summary is invalid JSON"
        ) from exc
    if not isinstance(summary, dict):
        _fail(f"floor{floor} calibration summary is not a JSON object")
    expected_fields = {
        "floor": floor,
        "lifecycle": lifecycle,
        "runtime_profile": values.get(f"{prefix}_RUNTIME_PROFILE"),
        "runtime_profile_sha256": values.get(f"{prefix}_RUNTIME_PROFILE_SHA256"),
        "execution_code_sha256": values.get("DEEPSEEK_EXECUTION_CODE_SHA256"),
    }
    for field, expected in expected_fields.items():
        if summary.get(field) != expected:
            _fail(
                f"floor{floor} calibration {field}={summary.get(field)!r}, "
                f"expected {expected!r}"
            )
    if _resolved(str(summary.get("common_epoch0_root", ""))) != common_root.resolve():
        _fail(f"floor{floor} calibration common epoch0 mismatch")
    if _resolved(str(summary.get("planning_history_root", ""))) != trigger_root.resolve():
        _fail(f"floor{floor} calibration planning history mismatch")
    observed_tokens = _parse_positive_int(
        summary.get("observed_tokens"), f"floor{floor} calibration observed tokens"
    )
    if observed_tokens < physical_cap:
        _fail(
            f"floor{floor} calibration observed {observed_tokens} KV tokens, "
            f"below candidate physical cap {physical_cap}"
        )
    log_path = _resolved(str(summary.get("log", "")))
    plan_path = _resolved(str(summary.get("plan_summary", "")))
    if not log_path.is_file() or not plan_path.is_file():
        _fail(f"floor{floor} calibration lifecycle artifacts are missing")
    raw = load_plans(plan_path, 1)[0]
    if int(raw.get("selected_floor", -1)) != floor:
        _fail(f"floor{floor} calibration plan selected another floor")
    calibration_plan = {
        1: {
            "floor": floor,
            "stages": tuple(int(value) for value in raw["shrink_stages"]),
            "stage_sets": tuple(
                tuple(int(rank) for rank in ranks)
                for ranks in raw["stage_survivor_ranks"]
            ),
        }
    }
    try:
        calls, shrinks, restores, _text = parse_runtime_log(log_path)
        last_resize_done = {1: max(call.done_position for call in calls)}
        runtime = _validate_lifecycle(
            shrinks,
            restores,
            calibration_plan,
            last_resize_done,
            rank_identity_known=str(lifecycle).startswith("planned_"),
        )[1]
    except (AuditError, ValueError) as exc:
        raise AuthorizationError(
            f"floor{floor} calibration lifecycle is invalid: {exc}"
        ) from exc
    return {
        "summary": str(summary_path),
        "summary_sha256": _sha256(summary_path),
        "log": str(log_path),
        "plan": str(plan_path),
        "observed_tokens": observed_tokens,
        "runtime_lifecycle": runtime,
    }


def _validate_floor_run(
    run_root: Path,
    floor: int,
    admission: dict[int, int],
    physical: dict[int, int],
    values: dict[str, str],
    workload: dict[str, Any],
    lifecycle: str = DEFAULT_LIFECYCLE,
) -> dict[str, Any]:
    config = lifecycle_config(lifecycle)
    prefix = str(config["prefix"])
    floor_root = run_root / f"floor{floor}"
    policy = str(config["policy"])
    epoch_name = f"epoch_001_mode1_{policy}"
    epoch_dirs = sorted(floor_root.glob(f"epoch_*_mode1_{policy}"))
    if [path.name for path in epoch_dirs] != [epoch_name]:
        _fail(f"floor{floor}: expected only {epoch_name}")
    epoch_dir = epoch_dirs[0]

    plan_path = epoch_dir / "oracle" / "length_sorted_rank_plan_summary.json"
    raw_plans = load_plans(plan_path, 1)
    try:
        plans = validate_plans(raw_plans, admission, physical, lifecycle)
    except AuditError as exc:
        raise AuthorizationError(str(exc)) from exc
    if plans[1]["floor"] != floor:
        _fail(f"floor{floor} run selected floor{plans[1]['floor']}")
    tail_guard_cap = _validate_release_window(
        raw_plans[0], floor, workload["max_response_length"]
    )
    history_dirs = raw_plans[0].get("length_prediction_baseline_dirs")
    expected_history = _resolved(values["DEEPSEEK_KV_CAP_PROBE_HISTORY_ROOT"])
    if (
        not isinstance(history_dirs, list)
        or len(history_dirs) != 1
        or not isinstance(history_dirs[0], str)
        or _resolved(history_dirs[0]) != expected_history
    ):
        _fail(
            f"floor{floor} authorization plan does not use the VERIFIED "
            "positive-release history"
        )

    log_path = _only_file(epoch_dir / "logs", "*.txt", "runtime log")
    try:
        calls, shrinks, restores, text = parse_runtime_log(log_path)
        validate_training_health(text, 1, log_path)
        tail_guard_log_markers = _validate_applied_tail_guard(
            text, floor, tail_guard_cap, log_path
        )
        sampling_cap_log_markers = _validate_applied_sampling_cap(
            text,
            floor,
            tail_guard_cap,
            workload["train_batch_size"],
            log_path,
        )
        last_resize_done = _validate_resize_calls(calls, plans)
        try:
            runtime_lifecycle = _validate_lifecycle(
                shrinks,
                restores,
                plans,
                last_resize_done,
                rank_identity_known=policy == "planned",
            )[1]
            lifecycle_evidence_source = "authorization"
            calibration_lifecycle = None
        except AuditError:
            if policy != "natural":
                raise
            runtime_lifecycle = _validate_partial_natural_lifecycle(
                shrinks,
                restores,
                plans[1],
                last_resize_done,
            )
            calibration_lifecycle = _validate_calibration_lifecycle(
                values,
                prefix,
                floor,
                _resolved(values["DEEPSEEK_KV_CAP_COMMON_EPOCH0_ROOT"]),
                _resolved(values["DEEPSEEK_KV_CAP_PROBE_HISTORY_ROOT"]),
                lifecycle,
                physical[floor],
            )
            lifecycle_evidence_source = "authorization_plus_calibration"
    except AuditError as exc:
        raise AuthorizationError(str(exc)) from exc

    rollout_path = _only_file(epoch_dir / "rollout_data", "*.jsonl", "rollout JSONL")
    length_path = _only_file(epoch_dir / "rollout_length", "length_*.txt", "length file")
    expected_responses = workload["expected_responses"]
    for artifact in (rollout_path, length_path):
        count = _line_count(artifact)
        if count != expected_responses:
            _fail(f"{artifact}: expected {expected_responses} rows, found {count}")
    response_lengths = _response_lengths(length_path)
    normalized_lengths, synthetic_terminal_pad_count = _normalize_decoded_lengths(
        rollout_path,
        length_path,
        response_lengths,
        tail_guard_cap,
    )
    observed_max_response = max(normalized_lengths)
    if observed_max_response > tail_guard_cap:
        _fail(
            f"floor{floor} observed response length {observed_max_response} "
            f"exceeds applied TailGuard plan cap {tail_guard_cap}"
        )

    planner_train = _only_file(epoch_dir / "oracle", "length_sorted_train.parquet", "planner train artifact")
    planner_sha256 = _sha256(planner_train)
    expected_planner_sha256 = values.get(
        f"{prefix}_KV_PROBE_PLANNER_TRAIN_SHA256_FLOOR{floor}"
    )
    if planner_sha256 != expected_planner_sha256:
        _fail(f"floor{floor} planner train artifact SHA256 mismatch")

    return {
        "floor": floor,
        "epoch_dir": str(epoch_dir.resolve()),
        "runtime_log": str(log_path.resolve()),
        "admission_cap": admission[floor],
        "physical_cap": physical[floor],
        "planner_train_sha256": planner_sha256,
        "release_area": float(raw_plans[0]["release_area"]),
        "release_area_unit": RELEASE_AREA_UNIT,
        "schedule_thresholds": raw_plans[0].get("schedule_thresholds", []),
        "predicted_step_exit": float(raw_plans[0]["predicted_step_exit"]),
        "tail_guard_response_cap": int(raw_plans[0]["tail_guard_response_cap"]),
        "tail_guard_log_markers": tail_guard_log_markers,
        "sampling_cap_log_markers": sampling_cap_log_markers,
        "recorded_max_response_mask_length": max(response_lengths),
        "observed_max_response_length": observed_max_response,
        "synthetic_terminal_pad_count": synthetic_terminal_pad_count,
        "resize_calls": len(calls),
        "shrink_events": len(shrinks),
        "restore_events": len(restores),
        "runtime_lifecycle": runtime_lifecycle,
        "lifecycle_evidence_source": lifecycle_evidence_source,
        "calibration_lifecycle": calibration_lifecycle,
    }


def validate_authorization(
    cap_env: Path,
    run_root: Path,
    common_root: Path,
    trigger_root: Path,
    lifecycle: str = DEFAULT_LIFECYCLE,
    workload_overrides: dict[str, Any] | None = None,
    expected_runtime_execution_sha256: str | None = None,
    expected_verification_code_sha256: str | None = None,
) -> dict[str, Any]:
    config = lifecycle_config(lifecycle)
    floors_to_validate = tuple(config["floors"])
    cap_env = cap_env.resolve()
    run_root = run_root.resolve()
    if not run_root.is_dir():
        _fail(f"authorization root does not exist: {run_root}")
    values = _load_env(cap_env)
    admission, physical = _candidate_caps(values, lifecycle)
    provenance, workload = _validate_provenance(
        values,
        common_root,
        trigger_root,
        lifecycle,
        workload_overrides,
        expected_runtime_execution_sha256,
        expected_verification_code_sha256,
    )
    floors = [
        _validate_floor_run(
            run_root,
            floor,
            admission,
            physical,
            values,
            workload,
            lifecycle,
        )
        for floor in floors_to_validate
    ]
    return {
        "status": "PASS",
        "protocol": (
            f"DeepSeek-V2-Lite {config['label']} strict one-step KV cap authorization"
        ),
        "lifecycle": lifecycle,
        "cap_env": str(cap_env),
        "cap_env_sha256_before_promotion": _sha256(cap_env),
        "run_root": str(run_root),
        "common_epoch0_root": str(common_root.resolve()),
        "trigger_root": str(trigger_root.resolve()),
        "world_size": WORLD_SIZE,
        "train_batch_size": workload["train_batch_size"],
        "rollout_n": workload["rollout_n"],
        "expected_responses_per_step": workload["expected_responses"],
        "max_prompt_length": int(
            values.get("DEEPSEEK_KV_CAP_MAX_PROMPT_LENGTH", "1024")
        ),
        "max_response_length": workload["max_response_length"],
        "max_num_batched_tokens": int(
            values.get("DEEPSEEK_KV_CAP_MAX_NUM_BATCHED_TOKENS", "17408")
        ),
        "max_num_seqs": workload["max_num_seqs"],
        "workload_profile_id": workload["workload_profile_id"],
        "workload_profile_sha256": workload["workload_profile_sha256"],
        "target_ratio": 1.0,
        "provenance": provenance,
        "floors": floors,
    }


def _write_atomic(path: Path, text: str, mode: int | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        if mode is not None:
            os.chmod(temporary_name, mode)
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def promote(
    cap_env: Path,
    run_root: Path,
    summary_path: Path,
    lifecycle: str = DEFAULT_LIFECYCLE,
) -> None:
    config = lifecycle_config(lifecycle)
    prefix = str(config["prefix"])
    floors = tuple(config["floors"])
    summary_sha256 = _sha256(summary_path)
    metadata_prefixes = (
        f"export {prefix}_KV_CAP_VALIDATION_RUN=",
        f"export {prefix}_KV_CAP_VALIDATION_SUMMARY=",
        f"export {prefix}_KV_CAP_VALIDATION_SUMMARY_SHA256=",
        f"export {prefix}_KV_CAP_VALIDATED_FLOORS=",
    )
    output: list[str] = []
    replaced = False
    for line in cap_env.read_text(encoding="utf-8").splitlines():
        if line.startswith(f"export {prefix}_KV_CAPS_VERIFIED="):
            output.append(f"export {prefix}_KV_CAPS_VERIFIED=1")
            replaced = True
        elif not line.startswith(metadata_prefixes):
            output.append(line)
    if not replaced:
        _fail(f"candidate cap file lacks {prefix}_KV_CAPS_VERIFIED")
    validated_floors = ",".join(str(floor) for floor in floors)
    output.extend(
        [
            f"export {prefix}_KV_CAP_VALIDATION_RUN="
            f"{shlex.quote(str(run_root.resolve()))}",
            f"export {prefix}_KV_CAP_VALIDATION_SUMMARY="
            f"{shlex.quote(str(summary_path.resolve()))}",
            f"export {prefix}_KV_CAP_VALIDATION_SUMMARY_SHA256={summary_sha256}",
            f"export {prefix}_KV_CAP_VALIDATED_FLOORS={validated_floors}",
        ]
    )
    _write_atomic(
        cap_env,
        "\n".join(output) + "\n",
        mode=cap_env.stat().st_mode & 0o777,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cap-env", required=True, type=Path)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--common-epoch0-root", required=True, type=Path)
    parser.add_argument("--trigger-root", required=True, type=Path)
    parser.add_argument(
        "--lifecycle",
        choices=tuple(LIFECYCLE_CONFIG),
        default=DEFAULT_LIFECYCLE,
    )
    parser.add_argument("--train-batch-size", type=int)
    parser.add_argument("--rollout-n", type=int)
    parser.add_argument("--expected-responses", type=int)
    parser.add_argument("--max-num-seqs", type=int)
    parser.add_argument("--max-response-length", type=int)
    parser.add_argument("--workload-profile-id")
    parser.add_argument("--workload-profile-sha256")
    parser.add_argument("--expected-runtime-execution-sha256")
    parser.add_argument("--expected-verification-code-sha256")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = args.output or args.run_root / "KV_CAP_AUTHORIZATION_SUMMARY.json"
    try:
        payload = validate_authorization(
            args.cap_env,
            args.run_root,
            args.common_epoch0_root,
            args.trigger_root,
            args.lifecycle,
            {
                "train_batch_size": args.train_batch_size,
                "rollout_n": args.rollout_n,
                "expected_responses": args.expected_responses,
                "max_num_seqs": args.max_num_seqs,
                "max_response_length": args.max_response_length,
                "workload_profile_id": args.workload_profile_id,
                "workload_profile_sha256": args.workload_profile_sha256,
            },
            args.expected_runtime_execution_sha256,
            args.expected_verification_code_sha256,
        )
        _write_atomic(output.resolve(), json.dumps(payload, indent=2, sort_keys=True) + "\n")
        promote(
            args.cap_env.resolve(),
            args.run_root.resolve(),
            output.resolve(),
            args.lifecycle,
        )
    except (AuthorizationError, AuditError, OSError, ValueError, KeyError) as exc:
        raise SystemExit(f"FAIL: {exc}") from exc
    floors = ", ".join(str(floor) for floor in lifecycle_config(args.lifecycle)["floors"])
    print(
        f"verified floors {floors} and promoted DeepSeek {args.lifecycle} "
        f"KV caps in {args.cap_env}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
