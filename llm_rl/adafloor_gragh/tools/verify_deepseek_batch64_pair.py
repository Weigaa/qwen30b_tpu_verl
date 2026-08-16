#!/usr/bin/env python3
"""Verify a request-matched DeepSeek batch-64 Vanilla/AdaFloor pair.

Each arm must contain exactly one epoch directory and a
``batch64_pair_manifest.env`` file, either at the supplied run root or inside
the epoch directory.  The manifest records the immutable experiment contract
that cannot be reconstructed safely from rollout output alone.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import os
import re
import shlex
import sys
import tempfile
from collections import Counter
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from tools.hash_deepseek_checkpoint import digest as checkpoint_digest  # noqa: E402


WORLD_SIZE = 16
ROLLOUT_N = 16
TRAIN_BATCH_SIZE = 64
RESPONSES_PER_STEP = TRAIN_BATCH_SIZE * ROLLOUT_N
BLOCK_SIZE = 128
MAX_PROMPT_LENGTH = 1024
MAX_RESPONSE_LENGTH = 16384
MAX_BATCHED_TOKENS = 17408
MAX_NUM_SEQS = 64
DEEPSEEK_EOS_TOKEN_ID = 100001
EXPECTED_PHASE_STEPS = {"gate": 1, "epoch": 5}
ALLOWED_FLOORS = (16, 8, 4, 2)
RELEASE_AREA_UNIT = "rank_token_proxy"
MANIFEST_NAME = "batch64_pair_manifest.env"

ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")
FLOAT = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
TIMESTAMP_RE = re.compile(
    r"(?:(\d{4})-)?(\d{2})-(\d{2})[ T](\d{2}):(\d{2}):(\d{2})(?:,(\d{3}))?"
)
ROLLOUT_RE = re.compile(r"rollout_output_time_s:\s*(%s)" % FLOAT)
ABORT_RE = re.compile(r"response/aborted_ratio:([0-9.eE+-]+)")
GLOBAL_STEP_RE = re.compile(r"training/global_step:([0-9]+)")
SCORE_RE = re.compile(r"critic/score/mean:([0-9.eE+-]+)")
WORKER_RE = re.compile(r"\((?:WorkerDict )?pid=(\d+)\)")
RANK_RE = re.compile(r"\[Rank (\d+)\s*\|\s*Local Rank")
RESIZE_RE = re.compile(
    r"rollout_worker_resize_start rank=(\d+) step=(\d+) epoch=-?\d+ "
    r"target_floor=(\d+) target_kv=(\d+)"
)
SHRINK_DONE_RE = re.compile(
    r"Elastic parallel shrink rpc done: global_rank=(\d+) active_ranks=\[([^]]*)\]"
)
TAIL_GUARD_RE = re.compile(
    r"Shrink-aware tail-guard response cap:\s*selected_floor=(\d+)\s+plan_cap=(\d+)\b"
)
PREEMPT_RE = re.compile(
    r"Preempting request\s+(\S+)\s+for request\s+(\S+)"
    r"(?:\s+discarded_computed_tokens=(\d+))?",
    re.IGNORECASE,
)
KWARGS_RE = re.compile(r"\bkwargs:\s*(\{[^\n]*\})")
INFER_START_RE = re.compile(
    r"rollout_worker_infer_start\s+rank=(\d+)\s+step=(\d+)\s+"
    r"epoch=-?\d+\s+batch_size=(\d+)\s+max_tokens=(\d+)"
)
OOM_RE = re.compile(
    r"out of memory|OutOfMemoryError|NPU memory is exhausted|"
    r"ACL_ERROR_RT_MEMORY_ALLOCATION",
    re.IGNORECASE,
)


class VerificationError(RuntimeError):
    """Raised when the paired experiment contract is not proven."""


def _fail(message: str) -> None:
    raise VerificationError(message)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_json_object(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        _fail(f"missing {label}: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise VerificationError(f"invalid {label} JSON: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        _fail(f"{label} is not a JSON object: {path}")
    return payload


def _chat_bos_token_id(model_path: Path) -> int:
    config = _read_json_object(model_path / "config.json", "model config")
    tokenizer_config = _read_json_object(
        model_path / "tokenizer_config.json", "tokenizer config"
    )
    tokenizer = _read_json_object(model_path / "tokenizer.json", "tokenizer")
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
        if not separator or not key.strip():
            _fail(f"{path}:{line_number} is not an environment assignment")
        try:
            words = shlex.split(raw_value, comments=True, posix=True)
        except ValueError as exc:
            raise VerificationError(f"cannot parse {path}:{line_number}: {exc}") from exc
        if len(words) > 1:
            _fail(f"{path}:{line_number} contains multiple shell words")
        key = key.strip()
        if key in values:
            _fail(f"{path}:{line_number} repeats {key}")
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
        converted = int(value)
    except ValueError as exc:
        raise VerificationError(f"{context} has invalid integer {key}={value!r}") from exc
    return converted


def _required_float(values: dict[str, str], key: str, context: str) -> float:
    value = _required(values, key, context)
    try:
        converted = float(value)
    except ValueError as exc:
        raise VerificationError(f"{context} has invalid number {key}={value!r}") from exc
    if not math.isfinite(converted):
        _fail(f"{context} has non-finite {key}")
    return converted


def _resolved(value: str) -> Path:
    return Path(value).expanduser().resolve()


def _require_path(recorded: str, expected: Path, label: str) -> None:
    if _resolved(recorded) != expected.resolve():
        _fail(f"{label} path mismatch: recorded={recorded}, expected={expected}")


def _only_file(directory: Path, pattern: str, label: str) -> Path:
    files = sorted(directory.glob(pattern))
    if len(files) != 1:
        _fail(f"{directory}: expected one {label}, found {len(files)}")
    return files[0]


def _resolve_epoch(run_dir: Path, arm: str) -> tuple[Path, Path]:
    run_dir = run_dir.resolve()
    if not run_dir.is_dir():
        _fail(f"{arm} run directory does not exist: {run_dir}")
    if (run_dir / "rollout_data").is_dir() and (run_dir / "logs").is_dir():
        epoch_dir = run_dir
        root = run_dir
    else:
        candidates = sorted(
            path
            for path in run_dir.glob("epoch_*")
            if path.is_dir() and (path / "rollout_data").is_dir() and (path / "logs").is_dir()
        )
        if len(candidates) != 1:
            _fail(f"{arm} run root must contain exactly one completed epoch, found {len(candidates)}")
        epoch_dir = candidates[0]
        root = run_dir
    if arm == "adafloor" and "mode1_natural" not in epoch_dir.name:
        _fail(f"AdaFloor epoch must use Natural mode1, got {epoch_dir.name}")
    manifest_candidates = [root / MANIFEST_NAME]
    if epoch_dir != root:
        manifest_candidates.append(epoch_dir / MANIFEST_NAME)
    manifests = [path for path in manifest_candidates if path.is_file()]
    if len(manifests) != 1:
        _fail(f"{arm} must contain exactly one {MANIFEST_NAME}, found {len(manifests)}")
    return epoch_dir, manifests[0]


def _validate_profile(path: Path) -> tuple[dict[str, str], str]:
    values = _load_env(path)
    expected = {
        "COMMON_EPOCH0_TRAIN_STEPS": 5,
        "COMMON_EPOCH0_TRAIN_BATCH_SIZE": TRAIN_BATCH_SIZE,
        "COMMON_EPOCH0_ROLLOUT_N": ROLLOUT_N,
        "COMMON_EPOCH0_MAX_NUM_SEQS": MAX_NUM_SEQS,
        "COMMON_EPOCH0_PROMPTS_TOTAL": 5 * TRAIN_BATCH_SIZE,
        "COMMON_EPOCH0_EXPECTED_RESPONSES_PER_STEP": RESPONSES_PER_STEP,
        "COMMON_EPOCH0_MAX_PROMPT_LENGTH": MAX_PROMPT_LENGTH,
        "COMMON_EPOCH0_MAX_RESPONSE_LENGTH": MAX_RESPONSE_LENGTH,
        "COMMON_EPOCH0_MAX_NUM_BATCHED_TOKENS": MAX_BATCHED_TOKENS,
    }
    for key, wanted in expected.items():
        got = _required_int(values, key, "workload profile")
        if got != wanted:
            _fail(f"batch64 workload profile requires {key}={wanted}, got {got}")
    profile_id = _required(values, "DEEPSEEK_WORKLOAD_PROFILE_ID", "workload profile")
    if re.fullmatch(r"[A-Za-z0-9._-]+", profile_id) is None:
        _fail("workload profile ID contains unsupported characters")
    return values, _sha256(path)


def _validate_common(
    common_root: Path,
    profile: dict[str, str],
    profile_sha256: str,
) -> dict[str, Any]:
    common_root = common_root.resolve()
    metadata_path = common_root / "common_epoch0_metadata.env"
    reuse_path = common_root / "reuse.env"
    marker = common_root / "DO_NOT_DELETE_COMMON_EPOCH0_CHECKPOINT"
    if not marker.is_file():
        _fail(f"common epoch0 completion marker is missing: {marker}")
    metadata = _load_env(metadata_path)
    reuse = _load_env(reuse_path)
    mapping = {
        "COMMON_EPOCH0_TRAIN_STEPS_USED": "COMMON_EPOCH0_TRAIN_STEPS",
        "COMMON_EPOCH0_TRAIN_BATCH_SIZE_USED": "COMMON_EPOCH0_TRAIN_BATCH_SIZE",
        "COMMON_EPOCH0_ROLLOUT_N_USED": "COMMON_EPOCH0_ROLLOUT_N",
        "COMMON_EPOCH0_MAX_NUM_SEQS_USED": "COMMON_EPOCH0_MAX_NUM_SEQS",
        "COMMON_EPOCH0_PROMPTS_TOTAL_USED": "COMMON_EPOCH0_PROMPTS_TOTAL",
        "COMMON_EPOCH0_EXPECTED_RESPONSES_PER_STEP_USED": (
            "COMMON_EPOCH0_EXPECTED_RESPONSES_PER_STEP"
        ),
        "COMMON_EPOCH0_MAX_PROMPT_LENGTH_USED": "COMMON_EPOCH0_MAX_PROMPT_LENGTH",
        "COMMON_EPOCH0_MAX_RESPONSE_LENGTH_USED": "COMMON_EPOCH0_MAX_RESPONSE_LENGTH",
        "COMMON_EPOCH0_MAX_NUM_BATCHED_TOKENS_USED": (
            "COMMON_EPOCH0_MAX_NUM_BATCHED_TOKENS"
        ),
    }
    for recorded_key, profile_key in mapping.items():
        recorded = _required(metadata, recorded_key, "common metadata")
        if recorded != profile[profile_key]:
            _fail(f"common metadata {recorded_key} does not match the workload profile")
    profile_id = profile["DEEPSEEK_WORKLOAD_PROFILE_ID"]
    if metadata.get("COMMON_EPOCH0_WORKLOAD_PROFILE_ID") != profile_id:
        _fail("common epoch0 workload profile ID mismatch")
    if metadata.get("COMMON_EPOCH0_WORKLOAD_PROFILE_SHA256") != profile_sha256:
        _fail("common epoch0 workload profile SHA256 mismatch")
    if metadata.get("COMMON_EPOCH0_PREEMPTION_POLICY_USED") != "record":
        _fail("batch64 common epoch0 must record rather than reject Vanilla preemption")
    checkpoint_values = {
        key: _resolved(_required(reuse, key, "common reuse metadata"))
        for key in ("BASELINE_INITIAL_RESUME_CKPT", "DYNAMIC_INITIAL_RESUME_CKPT")
    }
    if len(set(checkpoint_values.values())) != 1:
        _fail("common reuse metadata gives different Vanilla and AdaFloor checkpoints")
    checkpoint = next(iter(checkpoint_values.values()))
    if not (checkpoint / "actor").is_dir() or not (checkpoint / ".PRESERVE_COMMON_EPOCH0").is_file():
        _fail(f"frozen common checkpoint is incomplete: {checkpoint}")
    checkpoint_sha_path = common_root / "FROZEN_CHECKPOINT_SHA256"
    if not checkpoint_sha_path.is_file():
        _fail(f"frozen checkpoint SHA256 record is missing: {checkpoint_sha_path}")
    checkpoint_sha256 = checkpoint_sha_path.read_text(encoding="utf-8").strip()
    if re.fullmatch(r"[0-9a-f]{64}", checkpoint_sha256) is None:
        _fail("frozen checkpoint SHA256 record is invalid")
    if checkpoint_digest(checkpoint)[0] != checkpoint_sha256:
        _fail("frozen common checkpoint content does not match its SHA256 record")
    common_full16_text = _required(
        metadata, "COMMON_EPOCH0_KV_TOKENS_PER_RANK_USED", "common metadata"
    )
    if common_full16_text == "auto":
        full16: int | None = None
    else:
        try:
            full16 = int(common_full16_text)
        except ValueError as exc:
            raise VerificationError(
                "common Full16 KV capacity must be 'auto' or an integer"
            ) from exc
        if full16 <= 0 or full16 % BLOCK_SIZE:
            _fail("common Full16 physical KV capacity must be a positive block multiple")
    return {
        "metadata": metadata,
        "checkpoint": checkpoint,
        "checkpoint_sha256": checkpoint_sha256,
        "full16_physical_tokens": full16,
        "model_path": _resolved(_required(metadata, "COMMON_EPOCH0_MODEL_PATH", "common metadata")),
        "model_revision": _required(metadata, "COMMON_EPOCH0_MODEL_REVISION", "common metadata"),
        "execution_profile": _required(
            metadata, "COMMON_EPOCH0_EXECUTION_PROFILE_USED", "common metadata"
        ),
    }


def _validate_caps(
    cap_env: Path,
    profile: dict[str, str],
    profile_sha256: str,
    common_root: Path,
    common: dict[str, Any],
) -> dict[str, Any]:
    values = _load_env(cap_env)
    if values.get("DEEPSEEK_N_F2_KV_CAPS_VERIFIED") != "1":
        _fail("Natural floor2 KV caps are not verified")
    if values.get("DEEPSEEK_KV_CAP_WORKLOAD_PROFILE_ID") != profile["DEEPSEEK_WORKLOAD_PROFILE_ID"]:
        _fail("KV cap workload profile ID mismatch")
    if values.get("DEEPSEEK_KV_CAP_WORKLOAD_PROFILE_SHA256") != profile_sha256:
        _fail("KV cap workload profile SHA256 mismatch")
    _require_path(
        _required(values, "DEEPSEEK_KV_CAP_COMMON_EPOCH0_ROOT", "KV cap env"),
        common_root,
        "KV cap common epoch0",
    )
    if values.get("DEEPSEEK_KV_CAP_MODEL_REVISION") != common["model_revision"]:
        _fail("KV cap model revision mismatch")
    if values.get("DEEPSEEK_KV_CAP_EXECUTION_PROFILE") != common["execution_profile"]:
        _fail("KV cap execution profile mismatch")
    if values.get("DEEPSEEK_KV_CAP_BLOCK_SIZE") != str(BLOCK_SIZE):
        _fail(f"KV cap block size must equal {BLOCK_SIZE}")
    if values.get("DEEPSEEK_KV_CAP_ROLLOUT_N") != str(ROLLOUT_N):
        _fail(f"KV cap rollout n must equal {ROLLOUT_N}")
    shared = _required_int(values, "DEEPSEEK_KV_CAP_SHARED_FULL16_PHYSICAL_TOKENS", "KV cap env")
    vanilla = _required_int(values, "DEEPSEEK_VANILLA_KV_PHYSICAL_TOKENS", "KV cap env")
    floor16 = _required_int(values, "DEEPSEEK_N_F2_KV_PHYSICAL_FLOOR16", "KV cap env")
    if len({shared, vanilla, floor16}) != 1:
        _fail("Vanilla and Natural floor2 Full16 physical KV capacities differ")
    common_full16 = common["full16_physical_tokens"]
    if common_full16 is not None and common_full16 != shared:
        _fail("common, Vanilla, and Natural floor2 Full16 physical KV capacities differ")
    admission: dict[int, int] = {}
    physical: dict[int, int] = {}
    for floor in ALLOWED_FLOORS:
        admission[floor] = _required_int(
            values, f"DEEPSEEK_N_F2_KV_ADMISSION_FLOOR{floor}", "KV cap env"
        )
        physical[floor] = _required_int(
            values, f"DEEPSEEK_N_F2_KV_PHYSICAL_FLOOR{floor}", "KV cap env"
        )
        if (
            admission[floor] <= 0
            or physical[floor] <= 0
            or admission[floor] % BLOCK_SIZE
            or physical[floor] % BLOCK_SIZE
            or admission[floor] >= physical[floor]
        ):
            _fail(f"floor{floor} admission and physical capacities are invalid")

    validated_floors = _required(
        values, "DEEPSEEK_N_F2_KV_CAP_VALIDATED_FLOORS", "KV cap env"
    )
    if validated_floors != "16,8,4,2":
        _fail("Natural floor2 authorization did not validate floors 16,8,4,2")
    validation_run = _resolved(
        _required(values, "DEEPSEEK_N_F2_KV_CAP_VALIDATION_RUN", "KV cap env")
    )
    validation_summary = _resolved(
        _required(values, "DEEPSEEK_N_F2_KV_CAP_VALIDATION_SUMMARY", "KV cap env")
    )
    if not validation_run.is_dir():
        _fail(f"Natural floor2 validation run does not exist: {validation_run}")
    if not validation_summary.is_file():
        _fail(f"Natural floor2 validation summary does not exist: {validation_summary}")
    expected_summary_sha256 = _required(
        values, "DEEPSEEK_N_F2_KV_CAP_VALIDATION_SUMMARY_SHA256", "KV cap env"
    )
    observed_summary_sha256 = _sha256(validation_summary)
    if expected_summary_sha256 != observed_summary_sha256:
        _fail("Natural floor2 validation summary SHA256 mismatch")
    try:
        authorization = json.loads(validation_summary.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise VerificationError(
            f"invalid Natural floor2 validation summary: {validation_summary}"
        ) from exc
    if not isinstance(authorization, dict):
        _fail("Natural floor2 validation summary is not a JSON object")
    if authorization.get("status") != "PASS" or authorization.get("lifecycle") != "natural_f2":
        _fail("Natural floor2 validation summary is not a passing natural_f2 authorization")
    _require_path(
        str(authorization.get("run_root", "")), validation_run,
        "Natural floor2 validation run",
    )
    _require_path(
        str(authorization.get("common_epoch0_root", "")), common_root,
        "Natural floor2 validation common epoch0",
    )
    if authorization.get("workload_profile_id") != profile["DEEPSEEK_WORKLOAD_PROFILE_ID"]:
        _fail("Natural floor2 validation workload profile ID mismatch")
    if authorization.get("workload_profile_sha256") != profile_sha256:
        _fail("Natural floor2 validation workload profile SHA256 mismatch")
    expected_protocol = {
        "train_batch_size": TRAIN_BATCH_SIZE,
        "rollout_n": ROLLOUT_N,
        "expected_responses_per_step": RESPONSES_PER_STEP,
        "max_num_seqs": MAX_NUM_SEQS,
    }
    for key, expected in expected_protocol.items():
        if authorization.get(key) != expected:
            _fail(f"Natural floor2 validation summary has {key}={authorization.get(key)!r}")
    authorization_floors = authorization.get("floors")
    if not isinstance(authorization_floors, list) or len(authorization_floors) != 4:
        _fail("Natural floor2 validation summary has invalid floor evidence")
    observed_floors: list[int] = []
    for item in authorization_floors:
        if not isinstance(item, dict):
            _fail("Natural floor2 validation summary has invalid floor evidence")
        floor = item.get("floor")
        if floor not in ALLOWED_FLOORS:
            _fail("Natural floor2 validation summary has an unsupported floor")
        if item.get("admission_cap") != admission[floor] or item.get("physical_cap") != physical[floor]:
            _fail(f"Natural floor2 validation floor{floor} capacities are stale")
        observed_floors.append(floor)
    if observed_floors != list(ALLOWED_FLOORS):
        _fail(f"Natural floor2 validation floor order is {observed_floors}")
    return {
        "values": values,
        "sha256": _sha256(cap_env),
        "admission": admission,
        "physical": physical,
        "shared_full16": shared,
        "validation_run": validation_run,
        "validation_summary": validation_summary,
        "validation_summary_sha256": observed_summary_sha256,
    }


def _validate_manifest(
    path: Path,
    arm: str,
    phase: str,
    common_root: Path,
    common: dict[str, Any],
    profile: dict[str, str],
    profile_sha256: str,
    caps: dict[str, Any],
    expected_execution_code_sha256: str | None,
) -> dict[str, Any]:
    values = _load_env(path)
    expected_text = {
        "DEEPSEEK_BATCH64_ARM": arm,
        "DEEPSEEK_BATCH64_PHASE": phase,
        "DEEPSEEK_WORKLOAD_PROFILE_ID": profile["DEEPSEEK_WORKLOAD_PROFILE_ID"],
        "DEEPSEEK_WORKLOAD_PROFILE_SHA256": profile_sha256,
        "DEEPSEEK_BATCH64_MODEL_REVISION": common["model_revision"],
        "DEEPSEEK_BATCH64_EXECUTION_PROFILE": common["execution_profile"],
        "DEEPSEEK_BATCH64_CAP_ENV_SHA256": caps["sha256"],
        "DEEPSEEK_BATCH64_FROZEN_CHECKPOINT_SHA256": common[
            "checkpoint_sha256"
        ],
        "DEEPSEEK_BATCH64_PAIRED_REQUEST_SAMPLING_SEEDS": "1",
        "DEEPSEEK_BATCH64_FORCED_SELECTED_FLOOR": (
            "4" if phase == "gate" and arm == "adafloor" else "none"
        ),
    }
    if expected_execution_code_sha256 is not None:
        expected_text["DEEPSEEK_BATCH64_EXECUTION_CODE_SHA256"] = (
            expected_execution_code_sha256
        )
    for key, expected in expected_text.items():
        got = _required(values, key, f"{arm} manifest")
        if got != expected:
            _fail(f"{arm} manifest {key}={got!r}, expected {expected!r}")
    path_fields = {
        "DEEPSEEK_BATCH64_COMMON_ROOT": common_root,
        "DEEPSEEK_BATCH64_FROZEN_CHECKPOINT": common["checkpoint"],
        "DEEPSEEK_BATCH64_MODEL_PATH": common["model_path"],
    }
    for key, expected in path_fields.items():
        _require_path(_required(values, key, f"{arm} manifest"), expected, f"{arm} {key}")
    integer_fields = {
        "DEEPSEEK_BATCH64_TRAIN_BATCH_SIZE": TRAIN_BATCH_SIZE,
        "DEEPSEEK_BATCH64_ROLLOUT_N": ROLLOUT_N,
        "DEEPSEEK_BATCH64_MAX_NUM_SEQS": MAX_NUM_SEQS,
        "DEEPSEEK_BATCH64_MAX_PROMPT_LENGTH": MAX_PROMPT_LENGTH,
        "DEEPSEEK_BATCH64_MAX_RESPONSE_LENGTH": MAX_RESPONSE_LENGTH,
        "DEEPSEEK_BATCH64_MAX_NUM_BATCHED_TOKENS": MAX_BATCHED_TOKENS,
        "DEEPSEEK_BATCH64_FULL16_PHYSICAL_TOKENS": caps["shared_full16"],
        "DEEPSEEK_BATCH64_TOP_K": 50,
    }
    for key, expected in integer_fields.items():
        got = _required_int(values, key, f"{arm} manifest")
        if got != expected:
            _fail(f"{arm} manifest {key}={got}, expected {expected}")
    float_fields = {
        "DEEPSEEK_BATCH64_TEMPERATURE": 0.9,
        "DEEPSEEK_BATCH64_TOP_P": 0.9,
        "DEEPSEEK_BATCH64_DATASET_FRACTION": float(
            profile[
                "DEEPSEEK_KV_PROBE_DATASET_FRACTION"
                if phase == "gate"
                else "COMMON_EPOCH0_DATASET_FRACTION"
            ]
        ),
    }
    for key, expected in float_fields.items():
        got = _required_float(values, key, f"{arm} manifest")
        if not math.isclose(got, expected, rel_tol=0.0, abs_tol=1e-12):
            _fail(f"{arm} manifest {key}={got}, expected {expected}")
    execution_sha256 = _required(
        values, "DEEPSEEK_BATCH64_EXECUTION_CODE_SHA256", f"{arm} manifest"
    )
    if re.fullmatch(r"[0-9a-f]{64}", execution_sha256) is None:
        _fail(f"{arm} manifest has invalid execution code SHA256")
    return values


def _parse_timestamp(line: str) -> float | None:
    match = TIMESTAMP_RE.search(line)
    if match is None:
        return None
    return datetime(
        int(match.group(1) or 2026),
        int(match.group(2)),
        int(match.group(3)),
        int(match.group(4)),
        int(match.group(5)),
        int(match.group(6)),
        microsecond=int(match.group(7) or 0) * 1000,
    ).timestamp()


def _rank_list(value: str) -> tuple[int, ...]:
    if not value.strip():
        return ()
    try:
        return tuple(int(field.strip()) for field in value.split(","))
    except ValueError as exc:
        raise VerificationError(f"invalid rank list {value!r}") from exc


def _parse_log(path: Path, expected_steps: int, arm: str) -> dict[str, Any]:
    text = ANSI_RE.sub("", path.read_text(encoding="utf-8", errors="replace")).replace("\r", "\n")
    if OOM_RE.search(text):
        _fail(f"{arm} log contains an out-of-memory marker")
    aborts = [float(value) for value in ABORT_RE.findall(text)]
    if len(aborts) != expected_steps or any(value != 0.0 for value in aborts):
        _fail(f"{arm} abort ratios are {aborts}, expected {expected_steps} zeros")
    global_steps = [int(value) for value in GLOBAL_STEP_RE.findall(text)]
    if global_steps != list(range(1, expected_steps + 1)):
        _fail(f"{arm} global steps are {global_steps}")
    if "After trainer.fit" not in text:
        _fail(f"{arm} log has no trainer completion marker")

    rollout_events: list[tuple[float, float]] = []
    shrink_done: list[tuple[float, int, tuple[int, ...]]] = []
    preemptions: list[tuple[float | None, int, str, str, int | None]] = []
    pid_to_rank: dict[int, int] = {}
    kwargs: list[dict[str, Any]] = []
    for line in text.splitlines():
        timestamp = _parse_timestamp(line)
        rollout_match = ROLLOUT_RE.search(line)
        if rollout_match:
            if timestamp is None:
                _fail(f"{arm} rollout timing line has no timestamp")
            rollout_events.append((timestamp, float(rollout_match.group(1))))
        worker_prefixes = list(WORKER_RE.finditer(line))
        for index, worker_match in enumerate(worker_prefixes):
            pid = int(worker_match.group(1))
            chunk_end = (
                worker_prefixes[index + 1].start()
                if index + 1 < len(worker_prefixes)
                else len(line)
            )
            chunk = line[worker_match.end() : chunk_end]
            rank_match = RANK_RE.search(chunk)
            resize_match = RESIZE_RE.search(chunk)
            ranks = []
            if rank_match:
                ranks.append(int(rank_match.group(1)))
            if resize_match:
                ranks.append(int(resize_match.group(1)))
            if ranks and len(set(ranks)) != 1:
                _fail(f"{arm} worker pid {pid} has conflicting rank evidence")
            if ranks:
                rank = ranks[0]
                prior = pid_to_rank.setdefault(pid, rank)
                if prior != rank:
                    _fail(f"{arm} worker pid {pid} changes rank")
            preempt_match = PREEMPT_RE.search(chunk)
            if preempt_match:
                discarded = preempt_match.group(3)
                preemptions.append(
                    (
                        timestamp,
                        pid,
                        preempt_match.group(1),
                        preempt_match.group(2),
                        None if discarded is None else int(discarded),
                    )
                )
            kwargs_match = KWARGS_RE.search(chunk)
            if kwargs_match:
                try:
                    value = ast.literal_eval(kwargs_match.group(1))
                except (SyntaxError, ValueError) as exc:
                    raise VerificationError(f"cannot parse {arm} sampling kwargs") from exc
                if not isinstance(value, dict):
                    _fail(f"{arm} sampling kwargs are not a dictionary")
                kwargs.append(value)
        shrink_match = SHRINK_DONE_RE.search(line)
        if shrink_match:
            if timestamp is None:
                _fail("AdaFloor shrink completion line has no timestamp")
            shrink_done.append(
                (timestamp, int(shrink_match.group(1)), _rank_list(shrink_match.group(2)))
            )
    if len(rollout_events) != expected_steps:
        _fail(f"{arm} has {len(rollout_events)} rollout timings, expected {expected_steps}")
    if any(duration <= 0 or not math.isfinite(duration) for _end, duration in rollout_events):
        _fail(f"{arm} contains an invalid rollout duration")
    if arm == "adafloor" and preemptions:
        _fail(f"AdaFloor has {len(preemptions)} raw scheduler preemption events")
    return {
        "text": text,
        "rollout_events": rollout_events,
        "wall_s": sum(duration for _end, duration in rollout_events),
        "shrink_done": shrink_done,
        "preemptions": preemptions,
        "pid_to_rank": pid_to_rank,
        "sampling_kwargs": kwargs,
        "infer_start_caps": [
            tuple(map(int, match)) for match in INFER_START_RE.findall(text)
        ],
        "logged_scores": [float(value) for value in SCORE_RE.findall(text)],
        "resize_records": [tuple(map(int, match)) for match in RESIZE_RE.findall(text)],
        "abort_ratios": aborts,
        "oom_detected": False,
    }


def _validate_sampling(log: dict[str, Any], expected_steps: int, arm: str, caps: Iterable[int]) -> None:
    kwargs = log["sampling_kwargs"]
    if len(kwargs) != WORLD_SIZE:
        _fail(f"{arm} has {len(kwargs)} sampling kwargs, expected {WORLD_SIZE}")
    cap_list = list(caps)
    if len(cap_list) != expected_steps:
        _fail(f"{arm} response-cap evidence does not cover every step")
    observed_caps: Counter[int] = Counter()
    for index, value in enumerate(kwargs):
        expected = {"n": 1, "temperature": 0.9, "top_k": 50, "top_p": 0.9}
        for key, wanted in expected.items():
            got = value.get(key)
            if isinstance(wanted, float):
                valid = isinstance(got, (int, float)) and math.isclose(
                    float(got), wanted, rel_tol=0.0, abs_tol=1e-12
                )
            else:
                valid = got == wanted
            if not valid:
                _fail(f"{arm} sampling kwargs entry {index} has {key}={got!r}")
        max_tokens = value.get("max_tokens")
        if isinstance(max_tokens, bool) or not isinstance(max_tokens, int):
            _fail(f"{arm} sampling kwargs entry {index} has invalid max_tokens")
        observed_caps[max_tokens] += 1
    expected_per_worker = Counter({MAX_RESPONSE_LENGTH: WORLD_SIZE})
    if observed_caps != expected_per_worker:
        _fail(f"{arm} logged max_tokens {dict(observed_caps)} differs from expected {dict(expected_per_worker)}")

    infer_records = log["infer_start_caps"]
    expected_infer = {
        (rank, step): (TRAIN_BATCH_SIZE, cap_list[step - 1])
        for step in range(1, expected_steps + 1)
        for rank in range(WORLD_SIZE)
    }
    observed_infer: dict[tuple[int, int], tuple[int, int]] = {}
    for rank, step, batch_size, max_tokens in infer_records:
        key = (rank, step)
        if key in observed_infer:
            _fail(f"{arm} has duplicate infer-start evidence for rank {rank}, step {step}")
        observed_infer[key] = (batch_size, max_tokens)
    if observed_infer != expected_infer:
        missing = sorted(set(expected_infer) - set(observed_infer))
        extra = sorted(set(observed_infer) - set(expected_infer))
        wrong = sorted(
            key
            for key in set(observed_infer) & set(expected_infer)
            if observed_infer[key] != expected_infer[key]
        )
        _fail(
            f"{arm} applied sampling caps differ from the per-step contract, "
            f"missing={missing[:3]}, extra={extra[:3]}, wrong={wrong[:3]}"
        )


def _row_length(row: dict[str, Any], path: Path, line_number: int) -> int | None:
    mask = row.get("response_mask")
    if mask is not None:
        if not isinstance(mask, list) or any(value not in (0, 1) for value in mask):
            _fail(f"{path}:{line_number} has an invalid response_mask")
        return int(sum(mask))
    for key in ("response_length", "generated_tokens", "output_length", "response_token_count"):
        value = row.get(key)
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            return value
    return None


def _decoded_length(
    row: dict[str, Any],
    recorded_length: int,
    response_cap: int,
    path: Path,
    line_number: int,
) -> tuple[int, bool]:
    if recorded_length <= response_cap:
        return recorded_length, False
    if recorded_length != response_cap + 1:
        _fail(f"{path}:{line_number} response length exceeds its cap by more than one")

    responses = row.get("responses")
    mask = row.get("response_mask")
    if (
        not isinstance(responses, list)
        or not isinstance(mask, list)
        or len(responses) != len(mask)
        or len(responses) <= response_cap + 1
        or sum(mask) != response_cap + 1
        or any(value != 1 for value in mask[: response_cap + 1])
        or any(value != 0 for value in mask[response_cap + 1 :])
    ):
        _fail(
            f"{path}:{line_number} does not prove a single synthetic "
            "terminal padding token"
        )
    padding_token = responses[response_cap]
    if padding_token != DEEPSEEK_EOS_TOKEN_ID or any(
        token != padding_token for token in responses[response_cap:]
    ):
        _fail(f"{path}:{line_number} has non-padding response tokens beyond its cap")
    return response_cap, True


def _load_artifacts(
    epoch_dir: Path,
    expected_steps: int,
    response_caps: list[int],
    arm: str,
    bos_token_id: int,
) -> dict[str, Any]:
    rollout_dir = epoch_dir / "rollout_data"
    length_dir = epoch_dir / "rollout_length"
    identities: Counter[tuple[str, int, int]] = Counter()
    prompt_counts: Counter[str] = Counter()
    generated = 0
    rewards: list[float] = []
    cap_hits = 0
    synthetic_terminal_pad_count = 0
    step_stats: list[dict[str, Any]] = []
    for step in range(1, expected_steps + 1):
        rollout_path = rollout_dir / f"{step}.jsonl"
        length_path = length_dir / f"length_{step}.txt"
        if not rollout_path.is_file() or not length_path.is_file():
            _fail(f"{arm} step {step} rollout or length artifact is missing")
        try:
            lengths = [int(line.strip()) for line in length_path.read_text(encoding="utf-8").splitlines()]
        except ValueError as exc:
            raise VerificationError(f"{length_path} contains a non-integer length") from exc
        rows = rollout_path.read_text(encoding="utf-8").splitlines()
        if len(rows) != RESPONSES_PER_STEP or len(lengths) != RESPONSES_PER_STEP:
            _fail(
                f"{arm} step {step} must contain {RESPONSES_PER_STEP} responses, "
                f"got JSONL={len(rows)}, lengths={len(lengths)}"
            )
        step_tokens = 0
        step_rewards: list[float] = []
        step_hits = 0
        step_synthetic_terminal_pad_count = 0
        step_prompts: Counter[str] = Counter()
        step_samples: dict[str, Counter[int]] = {}
        for line_number, (raw_row, stored_length) in enumerate(zip(rows, lengths), 1):
            try:
                row = json.loads(raw_row)
            except json.JSONDecodeError as exc:
                raise VerificationError(f"invalid JSON at {rollout_path}:{line_number}") from exc
            if not isinstance(row, dict):
                _fail(f"{rollout_path}:{line_number} is not a JSON object")
            prompts = row.get("prompts")
            if not isinstance(prompts, list) or not prompts:
                _fail(f"{rollout_path}:{line_number} has no nonempty prompts list")
            if any(
                isinstance(token, bool) or not isinstance(token, int)
                for token in prompts
            ):
                _fail(
                    f"{rollout_path}:{line_number} prompts contains a non-integer token ID"
                )
            bos_count = sum(token == bos_token_id for token in prompts)
            if bos_count != 1:
                _fail(
                    f"{rollout_path}:{line_number} prompt must contain exactly one "
                    f"BOS token {bos_token_id}, found {bos_count}"
                )
            prompt_hash = row.get("rollout_prompt_hash")
            sample_index = row.get("rollout_sample_index")
            request_seed = row.get("rollout_request_seed")
            if not isinstance(prompt_hash, str) or not prompt_hash:
                _fail(f"{rollout_path}:{line_number} has no rollout_prompt_hash")
            if isinstance(sample_index, bool) or not isinstance(sample_index, int):
                _fail(f"{rollout_path}:{line_number} has no integer rollout_sample_index")
            if sample_index < 0 or sample_index >= ROLLOUT_N:
                _fail(
                    f"{rollout_path}:{line_number} rollout_sample_index is outside "
                    f"[0, {ROLLOUT_N})"
                )
            if isinstance(request_seed, bool) or not isinstance(request_seed, int):
                _fail(f"{rollout_path}:{line_number} has no integer rollout_request_seed")
            identity = (prompt_hash, sample_index, request_seed)
            identities[identity] += 1
            prompt_counts[prompt_hash] += 1
            step_prompts[prompt_hash] += 1
            step_samples.setdefault(prompt_hash, Counter())[sample_index] += 1
            measured_length = _row_length(row, rollout_path, line_number)
            if measured_length is None:
                measured_length = stored_length
            if measured_length != stored_length:
                _fail(
                    f"{rollout_path}:{line_number} response length {measured_length} "
                    f"does not match {length_path} value {stored_length}"
                )
            decoded_length, synthetic_terminal_pad = _decoded_length(
                row,
                stored_length,
                response_caps[step - 1],
                rollout_path,
                line_number,
            )
            if decoded_length < 0 or decoded_length > response_caps[step - 1]:
                _fail(f"{rollout_path}:{line_number} decoded length is outside its cap")
            score = row.get("score")
            if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(float(score)):
                _fail(f"{rollout_path}:{line_number} has no finite numeric score")
            step_tokens += decoded_length
            step_rewards.append(float(score))
            finish_reason = row.get("response_finish_reason")
            if finish_reason is not None and not isinstance(finish_reason, str):
                _fail(
                    f"{rollout_path}:{line_number} has a non-string "
                    "response_finish_reason"
                )
            if (
                decoded_length == response_caps[step - 1]
                and finish_reason == "length"
            ):
                step_hits += 1
            if synthetic_terminal_pad:
                step_synthetic_terminal_pad_count += 1
        step_prompt_occurrences = sum(step_prompts.values()) // ROLLOUT_N
        if (
            any(count <= 0 or count % ROLLOUT_N for count in step_prompts.values())
            or step_prompt_occurrences != TRAIN_BATCH_SIZE
        ):
            _fail(
                f"{arm} step {step} is not {TRAIN_BATCH_SIZE} prompt occurrences "
                f"with {ROLLOUT_N} samples each"
            )
        for prompt_hash, response_count in step_prompts.items():
            occurrences = response_count // ROLLOUT_N
            expected_samples = Counter({
                sample_index: occurrences
                for sample_index in range(ROLLOUT_N)
            })
            if step_samples[prompt_hash] != expected_samples:
                _fail(
                    f"{arm} step {step} prompt {prompt_hash!r} does not contain "
                    f"one complete sample-index set per occurrence"
                )
        generated += step_tokens
        rewards.extend(step_rewards)
        cap_hits += step_hits
        synthetic_terminal_pad_count += step_synthetic_terminal_pad_count
        step_stats.append(
            {
                "step": step,
                "responses": len(rows),
                "generated_tokens": step_tokens,
                "mean_reward": mean(step_rewards),
                "response_cap": response_caps[step - 1],
                "cap_hits": step_hits,
                "synthetic_terminal_pad_count": step_synthetic_terminal_pad_count,
            }
        )
    expected_prompt_occurrences = TRAIN_BATCH_SIZE * expected_steps
    prompt_occurrence_count = sum(prompt_counts.values()) // ROLLOUT_N
    if (
        any(count <= 0 or count % ROLLOUT_N for count in prompt_counts.values())
        or prompt_occurrence_count != expected_prompt_occurrences
    ):
        _fail(
            f"{arm} epoch is not {expected_prompt_occurrences} prompt occurrences "
            f"with {ROLLOUT_N} samples each"
        )
    return {
        "identities": identities,
        "identity_count": sum(identities.values()),
        "unique_identity_count": len(identities),
        "prompt_occurrence_count": prompt_occurrence_count,
        "unique_prompt_count": len(prompt_counts),
        "generated_tokens": generated,
        "mean_reward": mean(rewards),
        "cap_hits": cap_hits,
        "synthetic_terminal_pad_count": synthetic_terminal_pad_count,
        "steps": step_stats,
    }


def _validate_plans(
    epoch_dir: Path,
    expected_steps: int,
    caps: dict[str, Any],
    log: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[int]]:
    path = epoch_dir / "oracle" / "length_sorted_rank_plan_summary.json"
    if not path.is_file():
        _fail(f"AdaFloor plan summary is missing: {path}")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise VerificationError(f"invalid AdaFloor plan JSON: {path}") from exc
    if not isinstance(raw, list) or len(raw) != expected_steps:
        _fail(f"AdaFloor plan must contain {expected_steps} steps")
    plans: list[dict[str, Any]] = []
    response_caps: list[int] = []
    for expected_step, plan in enumerate(raw, 1):
        if not isinstance(plan, dict) or plan.get("step") != expected_step:
            _fail(f"AdaFloor plan entry {expected_step} has an invalid step")
        floor = plan.get("selected_floor")
        if floor not in ALLOWED_FLOORS or plan.get("feasible") is not True:
            _fail(f"AdaFloor step {expected_step} has no safe allowed floor")
        physical = caps["physical"][floor]
        admission = caps["admission"][floor]
        if plan.get("kv_cap") != physical or plan.get("kv_admission_cap") != admission:
            _fail(f"AdaFloor step {expected_step} does not use authorized floor{floor} caps")
        peak = plan.get("max_adjusted_rank_peak_tokens")
        if isinstance(peak, bool) or not isinstance(peak, (int, float)) or float(peak) > admission:
            _fail(f"AdaFloor step {expected_step} does not prove KV admission feasibility")
        response_cap = plan.get("tail_guard_response_cap")
        if (
            isinstance(response_cap, bool)
            or not isinstance(response_cap, int)
            or response_cap <= 0
            or response_cap > MAX_RESPONSE_LENGTH
        ):
            _fail(f"AdaFloor step {expected_step} has an invalid response cap")
        release_area = plan.get("release_area")
        if (
            isinstance(release_area, bool)
            or not isinstance(release_area, (int, float))
            or not math.isfinite(float(release_area))
            or float(release_area) < 0
        ):
            _fail(f"AdaFloor step {expected_step} has invalid release area")
        release_area_unit = plan.get("release_area_unit")
        if release_area_unit not in (None, RELEASE_AREA_UNIT):
            _fail(
                f"AdaFloor step {expected_step} has invalid release area unit "
                f"{release_area_unit!r}"
            )
        stages = plan.get("shrink_stages")
        survivors = plan.get("stage_survivor_ranks")
        expected_stage_sizes = [16] if floor == 16 else [
            stage for stage in (8, 4, 2) if stage >= floor
        ]
        if stages != expected_stage_sizes or not isinstance(survivors, list) or len(survivors) != len(stages):
            _fail(f"AdaFloor step {expected_step} has an invalid shrink schedule")
        prior = set(range(WORLD_SIZE))
        normalized_survivors: list[tuple[int, ...]] = []
        for stage, values in zip(stages, survivors):
            if not isinstance(values, list) or any(isinstance(value, bool) or not isinstance(value, int) for value in values):
                _fail(f"AdaFloor step {expected_step} has invalid survivor ranks")
            target = tuple(values)
            nested = set(target) == prior if stage == WORLD_SIZE else set(target) < prior
            if len(target) != stage or len(set(target)) != stage or not nested:
                _fail(f"AdaFloor step {expected_step} survivor sets are not strictly nested")
            prior = set(target)
            if stage < WORLD_SIZE:
                normalized_survivors.append(target)
        plans.append(
            {
                "step": expected_step,
                "floor": floor,
                "physical_cap": physical,
                "admission_cap": admission,
                "response_cap": response_cap,
                "release_area": float(release_area),
                "release_area_unit": RELEASE_AREA_UNIT,
                "survivors": normalized_survivors,
            }
        )
        response_caps.append(response_cap)

    records = log["resize_records"]
    expected_resize = {
        (plan["step"], rank): (plan["floor"], plan["physical_cap"])
        for plan in plans
        for rank in range(WORLD_SIZE)
    }
    observed_resize: dict[tuple[int, int], tuple[int, int]] = {}
    for rank, step, floor, physical in records:
        key = (step, rank)
        if key in observed_resize:
            _fail(f"duplicate AdaFloor runtime resize for step {step} rank {rank}")
        observed_resize[key] = (floor, physical)
    if observed_resize != expected_resize:
        missing = sorted(set(expected_resize) - set(observed_resize))
        extra = sorted(set(observed_resize) - set(expected_resize))
        wrong = sorted(
            key for key in set(observed_resize) & set(expected_resize)
            if observed_resize[key] != expected_resize[key]
        )
        _fail(f"AdaFloor runtime floor/cap evidence differs from plan, missing={missing[:4]}, extra={extra[:4]}, wrong={wrong[:4]}")
    applied_tail_guards = Counter(
        (int(floor), int(cap))
        for floor, cap in TAIL_GUARD_RE.findall(log["text"])
    )
    expected_tail_guards = Counter(
        (plan["floor"], plan["response_cap"]) for plan in plans
    )
    if applied_tail_guards != expected_tail_guards:
        _fail(
            "AdaFloor runtime TailGuard evidence differs from the plan, "
            f"observed={dict(applied_tail_guards)}, expected={dict(expected_tail_guards)}"
        )
    return plans, response_caps


def _released_rank_time(log: dict[str, Any], plans: list[dict[str, Any]]) -> dict[str, Any]:
    rollout_events = log["rollout_events"]
    shrink_done = log["shrink_done"]
    total = 0.0
    per_step: list[dict[str, Any]] = []
    previous_end = -float("inf")
    used_events = 0
    for plan, (rollout_end, rollout_duration) in zip(plans, rollout_events):
        rollout_start = rollout_end - rollout_duration
        window_start = max(previous_end, rollout_start - 1.0)
        events = [event for event in shrink_done if window_start < event[0] <= rollout_end]
        current = tuple(range(WORLD_SIZE))
        step_release = 0.0
        transitions: list[dict[str, Any]] = []
        for planned_target in plan["survivors"]:
            target_size = len(planned_target)
            candidate_targets = {
                event[2]
                for event in events
                if len(event[2]) == target_size and set(event[2]) < set(current)
            }
            if not candidate_targets:
                break
            if len(candidate_targets) != 1:
                _fail(
                    f"AdaFloor step {plan['step']} has multiple Natural "
                    f"survivor sets at floor{target_size}: {sorted(candidate_targets)}"
                )
            target = next(iter(candidate_targets))
            matches = [event for event in events if event[2] == target]
            ranks = [event[1] for event in matches]
            if len(matches) != len(current) or set(ranks) != set(current):
                _fail(
                    f"AdaFloor step {plan['step']} transition {len(current)}->{len(target)} "
                    f"has completion ranks {sorted(ranks)}"
                )
            coordinated = max(event[0] for event in matches)
            if coordinated > rollout_end:
                _fail(f"AdaFloor step {plan['step']} shrink completes after rollout")
            released = (len(current) - len(target)) * max(0.0, rollout_end - coordinated)
            step_release += released
            used_events += len(matches)
            transitions.append(
                {
                    "from": len(current),
                    "to": len(target),
                    "survivor_ranks": list(target),
                    "coordinated_completion_timestamp": coordinated,
                    "released_rank_seconds": released,
                }
            )
            current = target
        if not plan["survivors"] and events:
            _fail(f"AdaFloor floor16 step {plan['step']} contains shrink completion events")
        total += step_release
        per_step.append(
            {
                "step": plan["step"],
                "selected_floor": plan["floor"],
                "predicted_release_proxy_rank_tokens": plan["release_area"],
                "predicted_release_proxy_unit": plan["release_area_unit"],
                "coordinated_released_rank_seconds": step_release,
                "completed_transition_count": len(transitions),
                "transitions": transitions,
            }
        )
        previous_end = rollout_end
    if used_events != len(shrink_done):
        _fail("AdaFloor log contains shrink completion events outside planned transitions")
    return {"total_rank_seconds": total, "steps": per_step}


def _preemption_summary(
    log: dict[str, Any],
    *,
    all_responses_completed: bool,
    no_aborts: bool,
) -> dict[str, Any]:
    events = log["preemptions"]
    if not events:
        return {
            "raw_events": 0,
            "unique_preempted_request_count": 0,
            "unique_preempted_request_ids": [],
            "unique_preempted_request_ids_reason": None,
            "recomputed_kv_tokens": 0,
            "recomputed_kv_tokens_reason": None,
        }
    rollout_events = log["rollout_events"]
    windows = [
        (end - duration - 1.0, end, step)
        for step, (end, duration) in enumerate(rollout_events, 1)
    ]
    identities: set[str] = set()
    problems: list[str] = []
    discarded_counts: list[int | None] = []
    for timestamp, pid, request_id, _incoming, discarded in events:
        discarded_counts.append(discarded)
        rank = log["pid_to_rank"].get(pid)
        matching_steps = [] if timestamp is None else [step for start, end, step in windows if start < timestamp <= end]
        if rank is None:
            problems.append(f"pid {pid} has no reliable global-rank mapping")
        if len(matching_steps) != 1:
            problems.append(f"pid {pid} request {request_id} cannot be assigned to exactly one rollout step")
        if rank is not None and len(matching_steps) == 1:
            identities.add(f"step{matching_steps[0]}:rank{rank}:request{request_id}")
    if problems:
        unique_count: int | None = None
        unique_ids: list[str] | None = None
        reason: str | None = "; ".join(sorted(set(problems)))
    else:
        unique_count = len(identities)
        unique_ids = sorted(identities)
        reason = None
    if not all_responses_completed or not no_aborts:
        recomputed_tokens: int | None = None
        recomputed_reason: str | None = (
            "Discarded computed tokens imply recomputation only when every "
            "response completes and no response is aborted."
        )
    elif any(value is None for value in discarded_counts):
        recomputed_tokens = None
        recomputed_reason = (
            "At least one preemption event lacks discarded_computed_tokens, "
            "so recomputed KV work is unknown."
        )
    else:
        recomputed_tokens = sum(value for value in discarded_counts if value is not None)
        recomputed_reason = None
    return {
        "raw_events": len(events),
        "unique_preempted_request_count": unique_count,
        "unique_preempted_request_ids": unique_ids,
        "unique_preempted_request_ids_reason": reason,
        "recomputed_kv_tokens": recomputed_tokens,
        "recomputed_kv_tokens_reason": recomputed_reason,
    }


def _has_exercised_tailguard(summary: dict[str, Any]) -> bool:
    return any(
        step["response_cap"] < MAX_RESPONSE_LENGTH and step["cap_hits"] > 0
        for step in summary["steps"]
    )


def _tailguard_reduction_budget(summary: dict[str, Any]) -> int:
    return sum(
        step["cap_hits"] * (MAX_RESPONSE_LENGTH - step["response_cap"])
        for step in summary["steps"]
        if step["response_cap"] < MAX_RESPONSE_LENGTH
    )


def verify_pair(
    phase: str,
    vanilla_run_dir: Path,
    adafloor_run_dir: Path,
    common_root: Path,
    cap_env: Path,
    workload_profile_env: Path,
    expected_execution_code_sha256: str | None = None,
    enforce_work_comparability: bool = True,
) -> dict[str, Any]:
    if phase not in EXPECTED_PHASE_STEPS:
        _fail(f"phase must be one of {sorted(EXPECTED_PHASE_STEPS)}")
    expected_steps = EXPECTED_PHASE_STEPS[phase]
    profile, profile_sha256 = _validate_profile(workload_profile_env)
    common = _validate_common(common_root, profile, profile_sha256)
    bos_token_id = _chat_bos_token_id(common["model_path"])
    caps = _validate_caps(cap_env, profile, profile_sha256, common_root, common)
    vanilla_epoch, vanilla_manifest_path = _resolve_epoch(vanilla_run_dir, "vanilla")
    adafloor_epoch, adafloor_manifest_path = _resolve_epoch(adafloor_run_dir, "adafloor")
    vanilla_manifest = _validate_manifest(
        vanilla_manifest_path, "vanilla", phase, common_root, common, profile,
        profile_sha256, caps, expected_execution_code_sha256,
    )
    adafloor_manifest = _validate_manifest(
        adafloor_manifest_path, "adafloor", phase, common_root, common, profile,
        profile_sha256, caps, expected_execution_code_sha256,
    )
    sampling_keys = (
        "DEEPSEEK_BATCH64_TEMPERATURE",
        "DEEPSEEK_BATCH64_TOP_P",
        "DEEPSEEK_BATCH64_TOP_K",
        "DEEPSEEK_BATCH64_MAX_RESPONSE_LENGTH",
        "DEEPSEEK_BATCH64_PAIRED_REQUEST_SAMPLING_SEEDS",
        "DEEPSEEK_BATCH64_EXECUTION_CODE_SHA256",
        "DEEPSEEK_BATCH64_FROZEN_CHECKPOINT_SHA256",
    )
    if any(vanilla_manifest[key] != adafloor_manifest[key] for key in sampling_keys):
        _fail("Vanilla and AdaFloor manifests record different sampling contracts")

    vanilla_log_path = _only_file(vanilla_epoch / "logs", "*.txt", "Vanilla runtime log")
    adafloor_log_path = _only_file(adafloor_epoch / "logs", "*.txt", "AdaFloor runtime log")
    vanilla_log = _parse_log(vanilla_log_path, expected_steps, "vanilla")
    adafloor_log = _parse_log(adafloor_log_path, expected_steps, "adafloor")
    plans, adafloor_response_caps = _validate_plans(adafloor_epoch, expected_steps, caps, adafloor_log)
    if phase == "gate" and [plan["floor"] for plan in plans] != [4]:
        _fail("paired gate must execute the recorded AdaFloor floor4 safety gate")
    vanilla_response_caps = [MAX_RESPONSE_LENGTH] * expected_steps
    _validate_sampling(vanilla_log, expected_steps, "vanilla", vanilla_response_caps)
    _validate_sampling(adafloor_log, expected_steps, "adafloor", adafloor_response_caps)

    vanilla = _load_artifacts(
        vanilla_epoch,
        expected_steps,
        vanilla_response_caps,
        "vanilla",
        bos_token_id,
    )
    adafloor = _load_artifacts(
        adafloor_epoch,
        expected_steps,
        adafloor_response_caps,
        "adafloor",
        bos_token_id,
    )
    if vanilla["identities"] != adafloor["identities"]:
        only_vanilla = sorted(
            (identity, count)
            for identity, count in (
                vanilla["identities"] - adafloor["identities"]
            ).items()
        )
        only_adafloor = sorted(
            (identity, count)
            for identity, count in (
                adafloor["identities"] - vanilla["identities"]
            ).items()
        )
        _fail(
            "paired request identity multisets differ, "
            f"Vanilla-only={only_vanilla[:3]}, AdaFloor-only={only_adafloor[:3]}"
        )
    vanilla_tokens = vanilla["generated_tokens"]
    adafloor_tokens = adafloor["generated_tokens"]
    if vanilla_tokens:
        work_retention = adafloor_tokens / vanilla_tokens
    else:
        work_retention = 1.0
    work_difference = 1.0 - work_retention
    absolute_work_difference = abs(work_difference)
    exercised_tailguard = _has_exercised_tailguard(adafloor)
    tailguard_reduction_budget = _tailguard_reduction_budget(adafloor)
    work_tolerance_tokens = 0.01 * vanilla_tokens
    observed_reduction_tokens = vanilla_tokens - adafloor_tokens
    reduction_is_explained = (
        exercised_tailguard
        and observed_reduction_tokens
        <= work_tolerance_tokens + tailguard_reduction_budget + 1e-12
    )
    work_contract_satisfied = not (
        absolute_work_difference > 0.01 + 1e-12 and not (
            work_difference > 0.0 and reduction_is_explained
        )
    )
    if enforce_work_comparability and not work_contract_satisfied:
        _fail(
            "generated work differs by more than 1% without a matching "
            "AdaFloor TailGuard reduction"
        )

    release = _released_rank_time(adafloor_log, plans)
    if not any(plan["floor"] < WORLD_SIZE for plan in plans):
        _fail(f"paired {phase} does not select any shrinkable AdaFloor floor")
    completed_transitions = sum(
        step["completed_transition_count"] for step in release["steps"]
    )
    if completed_transitions < 1 or release["total_rank_seconds"] <= 0:
        _fail(
            f"paired {phase} does not complete a coordinated transition with positive released rank-time"
        )
    expected_response_count = expected_steps * RESPONSES_PER_STEP
    vanilla_preemption = _preemption_summary(
        vanilla_log,
        all_responses_completed=(
            vanilla["identity_count"] == expected_response_count
        ),
        no_aborts=all(value == 0.0 for value in vanilla_log["abort_ratios"]),
    )
    adafloor_preemption = _preemption_summary(
        adafloor_log,
        all_responses_completed=(
            adafloor["identity_count"] == expected_response_count
        ),
        no_aborts=all(value == 0.0 for value in adafloor_log["abort_ratios"]),
    )
    for summary, log, arm in (
        (vanilla, vanilla_log, "vanilla"),
        (adafloor, adafloor_log, "adafloor"),
    ):
        summary["rollout_wall_s"] = log["wall_s"]
        summary["aborted_responses"] = 0
        summary["abort_ratios"] = log["abort_ratios"]
        summary["oom_detected"] = log["oom_detected"]
        summary["work_normalized_throughput_tokens_per_s"] = (
            summary["generated_tokens"] / log["wall_s"]
        )
        logged_scores = log["logged_scores"]
        if logged_scores:
            if len(logged_scores) != expected_steps:
                _fail(f"{arm} has an incomplete set of logged reward means")
            artifact_scores = [step["mean_reward"] for step in summary["steps"]]
            if any(
                not math.isclose(a, b, rel_tol=0.0, abs_tol=5e-6)
                for a, b in zip(logged_scores, artifact_scores)
            ):
                _fail(f"{arm} logged rewards differ from rollout artifacts")
        summary.pop("identities")

    vanilla["preemption"] = vanilla_preemption
    adafloor["preemption"] = adafloor_preemption
    adafloor["selected_floors"] = [plan["floor"] for plan in plans]
    adafloor["predicted_release_proxy_rank_tokens"] = sum(
        plan["release_area"] for plan in plans
    )
    adafloor["predicted_release_proxy_unit"] = RELEASE_AREA_UNIT
    adafloor["coordinated_release"] = release
    throughput_ratio = (
        adafloor["work_normalized_throughput_tokens_per_s"]
        / vanilla["work_normalized_throughput_tokens_per_s"]
    )
    return {
        "status": "PASS" if work_contract_satisfied else "DIAGNOSTIC",
        "protocol": "DeepSeek-V2-Lite-Chat paired batch64 Vanilla Full16 versus AdaFloor Natural floor2",
        "phase": phase,
        "expected_steps": expected_steps,
        "responses_per_step": RESPONSES_PER_STEP,
        "request_identity_fields": [
            "rollout_prompt_hash",
            "rollout_sample_index",
            "rollout_request_seed",
        ],
        "request_identity_comparison": "multiset",
        "provenance": {
            "workload_profile": str(workload_profile_env.resolve()),
            "workload_profile_id": profile["DEEPSEEK_WORKLOAD_PROFILE_ID"],
            "workload_profile_sha256": profile_sha256,
            "common_root": str(common_root.resolve()),
            "frozen_checkpoint": str(common["checkpoint"]),
            "frozen_checkpoint_sha256": common["checkpoint_sha256"],
            "execution_code_sha256": vanilla_manifest[
                "DEEPSEEK_BATCH64_EXECUTION_CODE_SHA256"
            ],
            "model_path": str(common["model_path"]),
            "model_revision": common["model_revision"],
            "execution_profile": common["execution_profile"],
            "cap_env": str(cap_env.resolve()),
            "cap_env_sha256": caps["sha256"],
            "shared_full16_physical_tokens": caps["shared_full16"],
            "dataset_fraction": float(
                vanilla_manifest["DEEPSEEK_BATCH64_DATASET_FRACTION"]
            ),
            "natural_f2_validation_run": str(caps["validation_run"]),
            "natural_f2_validation_summary": str(caps["validation_summary"]),
            "natural_f2_validation_summary_sha256": caps[
                "validation_summary_sha256"
            ],
        },
        "vanilla": vanilla,
        "adafloor": adafloor,
        "comparison": {
            "paired_identity_multisets_equal": True,
            "paired_identity_count": vanilla["identity_count"],
            "paired_unique_identity_count": vanilla["unique_identity_count"],
            "generated_work_relative_difference": work_difference,
            "generated_work_absolute_relative_difference": (
                absolute_work_difference
            ),
            "work_tolerance_tokens": work_tolerance_tokens,
            "tailguard_reduction_budget_tokens": tailguard_reduction_budget,
            "tailguard_reduction_explains_difference": reduction_is_explained,
            "generated_work_contract_satisfied": work_contract_satisfied,
            "raw_timing_comparison_is_valid": work_contract_satisfied,
            "raw_timing_comparison_reason": (
                None
                if work_contract_satisfied
                else (
                    "The request identities match, but layout-dependent stochastic "
                    "generation changed realized token work by more than 1%. Use "
                    "the exact-work replay for the causal timing comparison."
                )
            ),
            "adafloor_generated_token_retention": work_retention,
            "adafloor_to_vanilla_work_normalized_throughput": throughput_ratio,
            "reward_mean_difference_adafloor_minus_vanilla": (
                adafloor["mean_reward"] - vanilla["mean_reward"]
            ),
        },
    }


def _write_atomic(path: Path, payload: dict[str, Any]) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=sorted(EXPECTED_PHASE_STEPS), required=True)
    parser.add_argument("--vanilla-run-dir", type=Path, required=True)
    parser.add_argument("--adafloor-run-dir", type=Path, required=True)
    parser.add_argument("--common-root", type=Path, required=True)
    parser.add_argument("--cap-env", type=Path, required=True)
    parser.add_argument("--workload-profile-env", type=Path, required=True)
    parser.add_argument("--expected-execution-code-sha256")
    parser.add_argument(
        "--allow-layout-induced-work-divergence",
        action="store_true",
        help="emit a DIAGNOSTIC summary instead of failing a noncomparable natural pair",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        summary = verify_pair(
            args.phase,
            args.vanilla_run_dir,
            args.adafloor_run_dir,
            args.common_root,
            args.cap_env,
            args.workload_profile_env,
            args.expected_execution_code_sha256,
            not args.allow_layout_induced_work_divergence,
        )
        _write_atomic(args.output, summary)
    except (OSError, VerificationError) as exc:
        try:
            _write_atomic(
                args.output,
                {
                    "status": "FAIL",
                    "phase": args.phase,
                    "error": str(exc),
                },
            )
        except OSError:
            pass
        print(f"FAIL: {exc}")
        return 1
    print(f"{summary['status']}: {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
