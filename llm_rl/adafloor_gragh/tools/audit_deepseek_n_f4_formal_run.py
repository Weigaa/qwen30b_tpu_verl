#!/usr/bin/env python3
"""Audit plan and runtime consistency for formal DeepSeek AdaFloor runs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shlex
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
WORLD_SIZE = 16
BLOCK_SIZE = 128
RELEASE_AREA_UNIT = "rank_token_proxy"
DEFAULT_LIFECYCLE = "natural_f4"
ALL_EXPECTED_STAGES = {
    16: (16,),
    8: (8,),
    4: (8, 4),
    2: (8, 4, 2),
}
LIFECYCLE_CONFIG = {
    "natural_f4": {
        "label": "Natural floor4",
        "prefix": "DEEPSEEK_N_F4",
        "floors": (16, 8, 4),
        "runtime_profile_path": (
            ROOT / "internal" / "deepseek_v2_lite_natural_f4_runtime_profile.sh"
        ),
        "runtime_profile_id_key": "DEEPSEEK_N_F4_RUNTIME_PROFILE_ID",
        "policy": "natural",
        "runtime_profile_files": (
            ROOT / "internal" / "deepseek_v2_lite_natural_f4_runtime_profile.sh",
        ),
    },
    "natural_f2": {
        "label": "Natural floor2",
        "prefix": "DEEPSEEK_N_F2",
        "floors": (16, 8, 4, 2),
        "runtime_profile_path": (
            ROOT / "internal" / "deepseek_v2_lite_natural_f2_runtime_profile.sh"
        ),
        "runtime_profile_id_key": "DEEPSEEK_N_F2_RUNTIME_PROFILE_ID",
        "policy": "natural",
        "runtime_profile_files": (
            ROOT / "internal" / "deepseek_v2_lite_natural_f4_runtime_profile.sh",
            ROOT / "internal" / "deepseek_v2_lite_natural_f2_runtime_profile.sh",
        ),
    },
    "planned_f4": {
        "label": "Planned floor4",
        "prefix": "DEEPSEEK_P_F4",
        "floors": (16, 8, 4),
        "policy": "planned",
        "runtime_profile_path": (
            ROOT / "internal" / "deepseek_v2_lite_planned_f4_runtime_profile.sh"
        ),
        "runtime_profile_id_key": "DEEPSEEK_P_F4_RUNTIME_PROFILE_ID",
        "runtime_profile_files": (
            ROOT / "internal" / "deepseek_v2_lite_natural_f4_runtime_profile.sh",
            ROOT / "internal" / "deepseek_v2_lite_planned_f4_runtime_profile.sh",
        ),
    },
    "planned_f2": {
        "label": "Planned floor2",
        "prefix": "DEEPSEEK_P_F2",
        "floors": (16, 8, 4, 2),
        "policy": "planned",
        "runtime_profile_path": (
            ROOT / "internal" / "deepseek_v2_lite_planned_f2_runtime_profile.sh"
        ),
        "runtime_profile_id_key": "DEEPSEEK_P_F2_RUNTIME_PROFILE_ID",
        "runtime_profile_files": (
            ROOT / "internal" / "deepseek_v2_lite_natural_f4_runtime_profile.sh",
            ROOT / "internal" / "deepseek_v2_lite_planned_f4_runtime_profile.sh",
            ROOT / "internal" / "deepseek_v2_lite_planned_f2_runtime_profile.sh",
        ),
    },
}

# Compatibility aliases for callers that imported the original floor4 constants.
ALLOWED_FLOORS = tuple(reversed(LIFECYCLE_CONFIG[DEFAULT_LIFECYCLE]["floors"]))
EXPECTED_STAGES = {
    floor: ALL_EXPECTED_STAGES[floor] for floor in ALLOWED_FLOORS
}

WORKER_PREFIX_RE = re.compile(r"\(WorkerDict pid=(\d+)\)")
RESIZE_START_RE = re.compile(
    r"rollout_worker_resize_start rank=(\d+) step=(\d+) epoch=(-?\d+) "
    r"target_floor=(\d+) target_kv=(\d+)"
)
RESIZE_DONE_RE = re.compile(
    r"rollout_worker_resize_done rank=(\d+) step=(\d+) epoch=(-?\d+)"
)
RESIZE_PLAN_RE = re.compile(
    r"Mode1 adaptive KV resize phase=plan_new_kv_done "
    r"target_tokens=(\d+) effective_target_tokens=(\d+) new_tokens=(\d+)"
)
RESIZE_SKIP_RE = re.compile(
    r"Skip mode1 adaptive KV resize because floor and KV target are unchanged: "
    r"old_tokens=(\d+) target_tokens=(\d+) effective_target_tokens=(\d+) "
    r"target_floor=(\d+)"
)
PRE_RESUME_START_RE = re.compile(
    r"Mode1 pre-resume KV cleanup phase=start target_floor=(\d+) "
    r"previous_floor=(\d+) target_policy=(\w+)"
)
PRE_RESUME_FLOOR_PREPARE_DONE_RE = re.compile(
    r"Mode1 pre-resume KV cleanup phase=floor_prepare_done target_floor=(\d+)"
)
PRE_RESUME_NATURAL_PRUNE_RE = re.compile(
    r"Mode1 natural KV resize runtime prune summary: rank=(\d+) "
    r"target_floor=(\d+) changed=(\d+)"
)
PRE_RESUME_NATURAL_PRUNE_DONE_RE = re.compile(
    r"Mode1 pre-resume KV cleanup phase=natural_runtime_prune_done "
    r"target_floor=(\d+)"
)
PRE_RESUME_DONE_RE = re.compile(
    r"Mode1 pre-resume KV cleanup phase=done target_floor=(\d+) "
    r"previous_floor=(\d+)"
)
PREFIX_CACHE_RESET_RE = re.compile(r"Successfully reset prefix cache")
AFTER_RESUME_KV_RE = re.compile(
    r"rollout_mode_after_resume_kv_cache rank=(\d+)"
)
AFTER_ENV_SYNC_RE = re.compile(
    r"rollout_worker_after_env_sync rank=(\d+) step=(\d+) epoch=(-?\d+)"
)
SHRINK_RE = re.compile(
    r"Shrink-aware staged trigger: stage=\S+ "
    r"current_local=\[([^]]*)\] unfinished_local=\[[^]]*\] "
    r"target_local=\[[^]]*\] target_global=\[([^]]*)\]"
)
RESTORE_RE = re.compile(
    r"Elastic full-world restore segmented timing: rank=(\d+) "
    r"restore_seq=(\d+) world_size=(\d+)"
)
PREEMPT_RE = re.compile(r"preempting request|request preempted", re.IGNORECASE)
OOM_RE = re.compile(
    r"out of memory|OutOfMemoryError|NPU memory is exhausted|ACL_ERROR_RT_MEMORY_ALLOCATION",
    re.IGNORECASE,
)
ABORT_RE = re.compile(r"response/aborted_ratio:([0-9.eE+-]+)")
ROLLOUT_OUTPUT_RE = re.compile(r"rollout_output_time_s:")
GLOBAL_STEP_RE = re.compile(r"training/global_step:([0-9]+)")


class AuditError(RuntimeError):
    """Raised when a formal run violates a plan or runtime invariant."""


@dataclass(frozen=True)
class ResizeCall:
    rank: int
    step: int
    target_floor: int
    target_kv: int
    start_position: int
    done_position: int
    outcome: str
    outcome_values: tuple[int, ...]


@dataclass(frozen=True)
class ShrinkEvent:
    logger_rank: int
    step: int
    current_ranks: tuple[int, ...]
    target_ranks: tuple[int, ...]
    position: int


@dataclass(frozen=True)
class RestoreEvent:
    logger_rank: int
    step: int
    restore_seq: int
    world_size: int
    position: int


def _fail(message: str) -> None:
    raise AuditError(message)


def lifecycle_config(lifecycle: str) -> dict[str, Any]:
    try:
        return LIFECYCLE_CONFIG[lifecycle]
    except KeyError as exc:
        choices = ", ".join(LIFECYCLE_CONFIG)
        raise AuditError(
            f"unsupported DeepSeek lifecycle {lifecycle!r}, expected one of {choices}"
        ) from exc


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _runtime_profile_id(path: Path, key: str) -> str:
    match = re.search(
        rf"^export {re.escape(key)}=([^\s]+)$",
        path.read_text(encoding="utf-8"),
        re.M,
    )
    if match is None:
        _fail(f"runtime profile {path} does not define {key}")
    return match.group(1)


def _runtime_profile_sha256(paths: Iterable[Path]) -> str:
    hasher = hashlib.sha256()
    seen: set[Path] = set()
    for raw_path in paths:
        path = raw_path.resolve()
        if path in seen:
            _fail(f"duplicate runtime profile source {path}")
        seen.add(path)
        if not path.is_file():
            _fail(f"runtime profile source does not exist: {path}")
        relative = path.relative_to(ROOT).as_posix().encode("utf-8")
        content = path.read_bytes()
        hasher.update(len(relative).to_bytes(8, "big"))
        hasher.update(relative)
        hasher.update(len(content).to_bytes(8, "big"))
        hasher.update(content)
    if not seen:
        _fail("runtime profile source closure is empty")
    return hasher.hexdigest()


def _parse_rank_list(value: str) -> tuple[int, ...]:
    if not value.strip():
        return ()
    try:
        return tuple(int(item.strip()) for item in value.split(","))
    except ValueError as exc:
        raise AuditError(f"invalid rank list {value!r}") from exc


def _require_int(value: Any, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        _fail(f"{context} must be an integer, got {value!r}")
    return value


def _require_number(value: Any, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail(f"{context} must be numeric, got {value!r}")
    converted = float(value)
    if not math.isfinite(converted):
        _fail(f"{context} must be finite, got {value!r}")
    return converted


def parse_expected_epochs(spec: str) -> tuple[int, ...]:
    epochs: set[int] = set()
    for field in spec.split(","):
        field = field.strip()
        if not field:
            continue
        if "-" in field:
            start_text, end_text = field.split("-", 1)
            start, end = int(start_text), int(end_text)
            if start > end:
                _fail(f"invalid descending epoch range {field!r}")
            epochs.update(range(start, end + 1))
        else:
            epochs.add(int(field))
    if not epochs or min(epochs) < 0:
        _fail(f"invalid expected epoch set {spec!r}")
    return tuple(sorted(epochs))


def load_cap_env(path: Path) -> dict[str, str]:
    if not path.is_file():
        _fail(f"cap environment file does not exist: {path}")
    values: dict[str, str] = {}
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].strip()
        if "=" not in line:
            _fail(f"{path}:{line_number} is not an environment assignment")
        key, raw_value = line.split("=", 1)
        key = key.strip()
        try:
            parsed = shlex.split(raw_value, comments=True, posix=True)
        except ValueError as exc:
            raise AuditError(f"cannot parse {path}:{line_number}: {exc}") from exc
        if len(parsed) > 1:
            _fail(f"{path}:{line_number} contains multiple shell words")
        values[key] = parsed[0] if parsed else ""
    return values


def validate_cap_env(
    values: dict[str, str], lifecycle: str = DEFAULT_LIFECYCLE
) -> tuple[dict[int, int], dict[int, int]]:
    config = lifecycle_config(lifecycle)
    prefix = str(config["prefix"])
    floors = tuple(config["floors"])
    verified_key = f"{prefix}_KV_CAPS_VERIFIED"
    if values.get(verified_key) != "1":
        _fail(f"{verified_key} must equal 1")
    if values.get("DEEPSEEK_KV_CAP_TARGET_RATIO") != "1.0":
        _fail("DEEPSEEK_KV_CAP_TARGET_RATIO must equal 1.0")
    if values.get("DEEPSEEK_KV_CAP_BLOCK_SIZE") != str(BLOCK_SIZE):
        _fail(f"DEEPSEEK_KV_CAP_BLOCK_SIZE must equal {BLOCK_SIZE}")
    validated_floors = ",".join(str(floor) for floor in floors)
    validated_key = f"{prefix}_KV_CAP_VALIDATED_FLOORS"
    if values.get(validated_key) != validated_floors:
        _fail(f"{validated_key} must equal {validated_floors}")

    profile_path = Path(config["runtime_profile_path"])
    profile_id_key = str(config["runtime_profile_id_key"])
    expected_profile_id = _runtime_profile_id(profile_path, profile_id_key)
    profile_key = f"{prefix}_RUNTIME_PROFILE"
    if values.get(profile_key) != expected_profile_id:
        _fail(f"{profile_key} does not match the {config['label']} runtime profile")
    profile_sha_key = f"{prefix}_RUNTIME_PROFILE_SHA256"
    profile_sha256 = _runtime_profile_sha256(config["runtime_profile_files"])
    if values.get(profile_sha_key) != profile_sha256:
        _fail(f"{profile_sha_key} does not match the runtime profile closure")

    if str(config["policy"]) == "planned":
        for floor in floors:
            headroom_key = f"{prefix}_HEADROOM_FLOOR{floor}"
            try:
                headroom = int(values[headroom_key])
            except (KeyError, ValueError) as exc:
                _fail(f"missing or invalid Planned headroom {headroom_key}")
            if headroom < 0 or headroom % BLOCK_SIZE:
                _fail(
                    f"{headroom_key} must be a nonnegative multiple of {BLOCK_SIZE}"
                )
        training_key = f"{prefix}_TRAINING_MIN_FREE_MIB"
        try:
            training_min_free_mib = int(values[training_key])
        except (KeyError, ValueError) as exc:
            _fail(f"missing or invalid Planned training reserve {training_key}")
        if training_min_free_mib <= 0:
            _fail(f"{training_key} must be positive")

    admission: dict[int, int] = {}
    physical: dict[int, int] = {}
    for floor in floors:
        admission_key = f"{prefix}_KV_ADMISSION_FLOOR{floor}"
        physical_key = f"{prefix}_KV_PHYSICAL_FLOOR{floor}"
        try:
            admission[floor] = int(values[admission_key])
            physical[floor] = int(values[physical_key])
        except KeyError as exc:
            _fail(f"missing cap environment value {exc.args[0]}")
        except ValueError as exc:
            raise AuditError(f"invalid integer cap for floor {floor}") from exc
        for label, cap in (("admission", admission[floor]), ("physical", physical[floor])):
            if cap <= 0 or cap % BLOCK_SIZE != 0:
                _fail(
                    f"floor{floor} {label} cap {cap} must be a positive multiple "
                    f"of {BLOCK_SIZE}"
                )
        if admission[floor] >= physical[floor]:
            _fail(
                f"floor{floor} admission cap {admission[floor]} must be below "
                f"physical cap {physical[floor]}"
            )
    return admission, physical


def load_plans(path: Path, expected_steps: int) -> list[dict[str, Any]]:
    if not path.is_file():
        _fail(f"plan summary does not exist: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise AuditError(f"invalid plan summary {path}: {exc}") from exc
    plans = payload.get("steps") if isinstance(payload, dict) else payload
    if not isinstance(plans, list):
        _fail(f"plan summary {path} must contain a list of steps")
    if len(plans) != expected_steps:
        _fail(f"plan summary {path} has {len(plans)} steps, expected {expected_steps}")
    if not all(isinstance(plan, dict) for plan in plans):
        _fail(f"plan summary {path} contains a non-object step")
    return plans


def validate_plans(
    plans: list[dict[str, Any]],
    admission: dict[int, int],
    physical: dict[int, int],
    lifecycle: str = DEFAULT_LIFECYCLE,
) -> dict[int, dict[str, Any]]:
    config = lifecycle_config(lifecycle)
    allowed_floors = frozenset(config["floors"])
    validated: dict[int, dict[str, Any]] = {}
    full_world = tuple(range(WORLD_SIZE))
    for expected_step, plan in enumerate(plans, 1):
        step = _require_int(plan.get("step"), f"plan index {expected_step} step")
        if step != expected_step:
            _fail(f"plan step order mismatch: expected {expected_step}, got {step}")
        if plan.get("feasible") is not True:
            _fail(f"step {step} plan is not feasible")
        if plan.get("rank_matching_policy") != "release_area":
            _fail(f"step {step} rank_matching_policy must equal release_area")
        if plan.get("release_area_unit") not in (None, RELEASE_AREA_UNIT):
            _fail(f"step {step} has an invalid release_area_unit")

        floor = _require_int(plan.get("selected_floor"), f"step {step} selected_floor")
        if floor not in allowed_floors:
            _fail(f"step {step} selected unsupported floor {floor}")
        expected_admission = admission[floor]
        expected_physical = physical[floor]
        if _require_number(plan.get("kv_admission_cap"), f"step {step} kv_admission_cap") != expected_admission:
            _fail(f"step {step} admission cap does not match floor{floor} cap environment")
        if _require_number(plan.get("kv_cap"), f"step {step} kv_cap") != expected_physical:
            _fail(f"step {step} physical cap does not match floor{floor} cap environment")
        peak = _require_number(
            plan.get("max_adjusted_rank_peak_tokens"),
            f"step {step} max_adjusted_rank_peak_tokens",
        )
        if peak > expected_admission:
            _fail(
                f"step {step} peak {peak} exceeds floor{floor} admission cap "
                f"{expected_admission}"
            )
        if "kv_admission_headroom_tokens" in plan:
            headroom = _require_number(
                plan["kv_admission_headroom_tokens"],
                f"step {step} kv_admission_headroom_tokens",
            )
            if abs(headroom - (expected_admission - peak)) > 1e-6:
                _fail(f"step {step} admission headroom is inconsistent")
        if "kv_physical_headroom_tokens" in plan:
            headroom = _require_number(
                plan["kv_physical_headroom_tokens"],
                f"step {step} kv_physical_headroom_tokens",
            )
            if abs(headroom - (expected_physical - peak)) > 1e-6:
                _fail(f"step {step} physical headroom is inconsistent")

        stages_raw = plan.get("shrink_stages")
        stage_sets_raw = plan.get("stage_survivor_ranks")
        if not isinstance(stages_raw, list) or not isinstance(stage_sets_raw, list):
            _fail(f"step {step} must include shrink_stages and stage_survivor_ranks lists")
        stages = tuple(_require_int(value, f"step {step} shrink stage") for value in stages_raw)
        expected_stages = ALL_EXPECTED_STAGES[floor]
        if stages != expected_stages:
            _fail(
                f"step {step} floor{floor} stages {stages} do not match "
                f"{expected_stages}"
            )
        if len(stage_sets_raw) != len(stages):
            _fail(f"step {step} stage survivor list count does not match stage count")

        stage_sets: list[tuple[int, ...]] = []
        prior_set = full_world
        for stage, raw_ranks in zip(stages, stage_sets_raw):
            if not isinstance(raw_ranks, list):
                _fail(f"step {step} stage {stage} survivor ranks must be a list")
            ranks = tuple(_require_int(rank, f"step {step} stage {stage} rank") for rank in raw_ranks)
            if len(ranks) != stage or len(set(ranks)) != stage:
                _fail(f"step {step} stage {stage} must contain {stage} unique ranks")
            if any(rank < 0 or rank >= WORLD_SIZE for rank in ranks):
                _fail(f"step {step} stage {stage} contains an out-of-range rank")
            if not set(ranks).issubset(prior_set):
                _fail(f"step {step} stage {stage} survivors are not nested")
            stage_sets.append(ranks)
            prior_set = ranks
        if floor == 16 and stage_sets[0] != full_world:
            _fail(f"step {step} floor16 survivor order must be the full world")
        if tuple(plan.get("intermediate_survivor_ranks", ())) != stage_sets[0]:
            _fail(f"step {step} intermediate_survivor_ranks is inconsistent")
        if tuple(plan.get("final_survivor_ranks", ())) != stage_sets[-1]:
            _fail(f"step {step} final_survivor_ranks is inconsistent")

        validated[step] = {
            "floor": floor,
            "admission_cap": expected_admission,
            "physical_cap": expected_physical,
            "peak": peak,
            "stages": stages,
            "stage_sets": tuple(stage_sets),
        }
    return validated


def validate_planning_features(
    plans: list[dict[str, Any]],
) -> str:
    modes: set[str] = set()
    history_counts: set[int] = set()
    for plan in plans:
        step = _require_int(plan.get("step"), "planning feature step")
        histories = plan.get("length_prediction_baseline_dirs")
        if (
            not isinstance(histories, list)
            or not histories
            or any(not isinstance(path, str) or not path for path in histories)
        ):
            _fail(f"step {step} has invalid length-prediction history provenance")
        history_count = len(histories)
        expected_mode = (
            "single_epoch_prompt_max"
            if history_count == 1
            else "prompt_max_ema_history"
        )
        mode = plan.get("length_prediction_mode")
        if mode != expected_mode:
            _fail(
                f"step {step} length_prediction_mode={mode!r}, expected "
                f"{expected_mode!r} for {history_count} history epochs"
            )
        modes.add(expected_mode)
        history_counts.add(history_count)

        max_response_len = _require_number(
            plan.get("max_response_len"), f"step {step} max_response_len"
        )
        tail_guard_cap = _require_int(
            plan.get("tail_guard_response_cap"),
            f"step {step} tail_guard_response_cap",
        )
        if tail_guard_cap <= 0 or tail_guard_cap > max_response_len:
            _fail(
                f"step {step} TailGuard cap {tail_guard_cap} is outside "
                f"the response range 1 through {max_response_len:g}"
            )
        expected_enabled = tail_guard_cap < max_response_len
        if plan.get("tail_guard_enabled") is not expected_enabled:
            _fail(f"step {step} TailGuard enable state is inconsistent with its cap")
        if plan.get("tail_guard_prompt_tail_stat") != "max_response_over_rollout_n":
            _fail(f"step {step} uses an unexpected TailGuard statistic")
    if len(modes) != 1 or len(history_counts) != 1:
        _fail("steps in one epoch disagree on history or prediction mode")
    return next(iter(modes))


def parse_runtime_log(path: Path) -> tuple[list[ResizeCall], list[ShrinkEvent], list[RestoreEvent], str]:
    text = path.read_text(encoding="utf-8", errors="replace")
    if PREEMPT_RE.search(text):
        _fail(f"preemption marker found in {path}")
    if OOM_RE.search(text):
        _fail(f"out-of-memory marker found in {path}")

    prefixes = list(WORKER_PREFIX_RE.finditer(text))
    pid_to_rank: dict[int, int] = {}
    last_step: dict[int, int] = {}
    active: dict[int, dict[str, Any]] = {}
    pre_resume: dict[int, dict[str, Any]] = {}
    calls: list[ResizeCall] = []
    shrinks: list[ShrinkEvent] = []
    restores: list[RestoreEvent] = []

    for index, prefix in enumerate(prefixes):
        pid = int(prefix.group(1))
        end = prefixes[index + 1].start() if index + 1 < len(prefixes) else len(text)
        chunk = text[prefix.end() : end]
        position = prefix.start()

        pre_start_match = PRE_RESUME_START_RE.search(chunk)
        if pre_start_match:
            target_floor, previous_floor = map(int, pre_start_match.groups()[:2])
            pre_resume[pid] = {
                "target_floor": target_floor,
                "previous_floor": previous_floor,
                "policy": pre_start_match.group(3),
                "start": position,
                "floor_prepare_done": None,
                "prune_rank": None,
                "prune_changed": None,
                "prune_done": None,
                "done": None,
                "prefix_reset": None,
                "after_resume_rank": None,
                "after_resume": None,
                "env_rank": None,
                "env_step": None,
                "env_sync": None,
            }

        evidence = pre_resume.get(pid)
        floor_prepare_match = PRE_RESUME_FLOOR_PREPARE_DONE_RE.search(chunk)
        if floor_prepare_match and evidence is not None:
            if int(floor_prepare_match.group(1)) != evidence["target_floor"]:
                _fail(f"worker pid {pid} changed pre-resume target floor")
            evidence["floor_prepare_done"] = position
        prune_match = PRE_RESUME_NATURAL_PRUNE_RE.search(chunk)
        if prune_match and evidence is not None:
            prune_rank, prune_floor, prune_changed = map(int, prune_match.groups())
            if prune_floor != evidence["target_floor"]:
                _fail(f"worker pid {pid} changed Natural prune target floor")
            evidence["prune_rank"] = prune_rank
            evidence["prune_changed"] = prune_changed
        prune_done_match = PRE_RESUME_NATURAL_PRUNE_DONE_RE.search(chunk)
        if prune_done_match and evidence is not None:
            if int(prune_done_match.group(1)) != evidence["target_floor"]:
                _fail(f"worker pid {pid} changed Natural prune completion floor")
            evidence["prune_done"] = position
        pre_done_match = PRE_RESUME_DONE_RE.search(chunk)
        if pre_done_match and evidence is not None:
            done_floor, done_previous = map(int, pre_done_match.groups())
            if (done_floor, done_previous) != (
                evidence["target_floor"],
                evidence["previous_floor"],
            ):
                _fail(f"worker pid {pid} changed pre-resume completion fields")
            evidence["done"] = position
        if PREFIX_CACHE_RESET_RE.search(chunk) and evidence is not None:
            evidence["prefix_reset"] = position
        after_resume_match = AFTER_RESUME_KV_RE.search(chunk)
        if after_resume_match and evidence is not None:
            evidence["after_resume_rank"] = int(after_resume_match.group(1))
            evidence["after_resume"] = position
        env_sync_match = AFTER_ENV_SYNC_RE.search(chunk)
        if env_sync_match and evidence is not None:
            evidence["env_rank"] = int(env_sync_match.group(1))
            evidence["env_step"] = int(env_sync_match.group(2))
            evidence["env_sync"] = position

        start_match = RESIZE_START_RE.search(chunk)
        if start_match:
            rank, step, _epoch, floor, target_kv = map(int, start_match.groups())
            if pid in active:
                _fail(f"worker pid {pid} begins a resize before completing the prior call")
            if pid in pid_to_rank and pid_to_rank[pid] != rank:
                _fail(f"worker pid {pid} changed rank from {pid_to_rank[pid]} to {rank}")
            pid_to_rank[pid] = rank
            last_step[pid] = step
            active[pid] = {
                "rank": rank,
                "step": step,
                "floor": floor,
                "target_kv": target_kv,
                "start_position": position,
                "outcome": None,
                "outcome_values": (),
                "pre_resume": dict(pre_resume.get(pid, {})),
            }

        plan_match = RESIZE_PLAN_RE.search(chunk)
        if plan_match:
            if pid not in active or active[pid]["outcome"] is not None:
                _fail(f"worker pid {pid} has an unpaired or duplicate resize plan outcome")
            active[pid]["outcome"] = "plan"
            active[pid]["outcome_values"] = tuple(map(int, plan_match.groups()))

        skip_match = RESIZE_SKIP_RE.search(chunk)
        if skip_match:
            if pid not in active or active[pid]["outcome"] is not None:
                _fail(f"worker pid {pid} has an unpaired or duplicate resize skip outcome")
            active[pid]["outcome"] = "skip"
            active[pid]["outcome_values"] = tuple(map(int, skip_match.groups()))

        done_match = RESIZE_DONE_RE.search(chunk)
        if done_match:
            rank, step, _epoch = map(int, done_match.groups())
            call = active.pop(pid, None)
            if call is None:
                _fail(f"worker pid {pid} completed a resize without a start")
            if (rank, step) != (call["rank"], call["step"]):
                _fail(f"worker pid {pid} resize start and completion fields differ")
            if call["outcome"] is None:
                evidence = call["pre_resume"]
                evidence_positions = [
                    evidence.get("start"),
                    evidence.get("floor_prepare_done"),
                    evidence.get("prune_done"),
                    evidence.get("done"),
                    evidence.get("prefix_reset"),
                    evidence.get("after_resume"),
                    evidence.get("env_sync"),
                ]
                valid_pre_resume = (
                    evidence.get("policy") == "natural"
                    and evidence.get("target_floor") == call["floor"]
                    and evidence.get("previous_floor") != call["floor"]
                    and evidence.get("prune_changed") == 1
                    and evidence.get("prune_rank") == rank
                    and evidence.get("after_resume_rank") == rank
                    and evidence.get("env_rank") == rank
                    and evidence.get("env_step") == step
                    and all(value is not None for value in evidence_positions)
                    and evidence_positions == sorted(evidence_positions)
                    and evidence_positions[-1] < call["start_position"]
                )
                if not valid_pre_resume:
                    _fail(
                        f"worker pid {pid} resize call has no plan, skip, or "
                        "complete pre-resume materialization outcome"
                    )
                call["outcome"] = "pre_resume_materialized"
                call["outcome_values"] = (
                    int(evidence["target_floor"]),
                    int(evidence["previous_floor"]),
                )
            calls.append(
                ResizeCall(
                    rank=rank,
                    step=step,
                    target_floor=call["floor"],
                    target_kv=call["target_kv"],
                    start_position=call["start_position"],
                    done_position=position,
                    outcome=call["outcome"],
                    outcome_values=call["outcome_values"],
                )
            )

        shrink_match = SHRINK_RE.search(chunk)
        if shrink_match:
            if pid not in pid_to_rank or pid not in last_step:
                _fail(f"worker pid {pid} emitted shrink evidence before resize context")
            shrinks.append(
                ShrinkEvent(
                    logger_rank=pid_to_rank[pid],
                    step=last_step[pid],
                    current_ranks=_parse_rank_list(shrink_match.group(1)),
                    target_ranks=_parse_rank_list(shrink_match.group(2)),
                    position=position,
                )
            )

        restore_match = RESTORE_RE.search(chunk)
        if restore_match:
            logged_rank, restore_seq, world_size = map(int, restore_match.groups())
            if pid not in pid_to_rank or pid not in last_step:
                _fail(f"worker pid {pid} emitted restore evidence before resize context")
            if logged_rank != pid_to_rank[pid]:
                _fail(f"worker pid {pid} restore rank differs from its resize rank")
            restores.append(
                RestoreEvent(
                    logger_rank=logged_rank,
                    step=last_step[pid],
                    restore_seq=restore_seq,
                    world_size=world_size,
                    position=position,
                )
            )

    if active:
        _fail(f"runtime log ends with incomplete resize calls for pids {sorted(active)}")
    return calls, shrinks, restores, text


def _validate_resize_calls(
    calls: list[ResizeCall], plans: dict[int, dict[str, Any]]
) -> dict[int, int]:
    by_key: dict[tuple[int, int], ResizeCall] = {}
    for call in calls:
        key = (call.step, call.rank)
        if key in by_key:
            _fail(f"duplicate resize call for step {call.step} rank {call.rank}")
        by_key[key] = call

    expected_keys = {(step, rank) for step in plans for rank in range(WORLD_SIZE)}
    if set(by_key) != expected_keys:
        missing = sorted(expected_keys - set(by_key))
        extra = sorted(set(by_key) - expected_keys)
        _fail(f"resize coverage mismatch, missing={missing[:8]}, extra={extra[:8]}")

    last_done: dict[int, int] = {}
    for (step, rank), call in sorted(by_key.items()):
        plan = plans[step]
        floor = plan["floor"]
        physical = plan["physical_cap"]
        if call.target_floor != floor or call.target_kv != physical:
            _fail(
                f"step {step} rank {rank} runtime target floor/KV "
                f"({call.target_floor}, {call.target_kv}) differs from plan "
                f"({floor}, {physical})"
            )
        if call.outcome == "plan":
            target, effective, new = call.outcome_values
            if (target, effective, new) != (physical, physical, physical):
                _fail(
                    f"step {step} rank {rank} planned KV resize values "
                    f"{call.outcome_values} must all equal physical cap {physical}"
                )
        elif call.outcome == "skip":
            old, target, effective, skip_floor = call.outcome_values
            if (old, target, effective, skip_floor) != (physical, physical, physical, floor):
                _fail(
                    f"step {step} rank {rank} skipped KV resize values "
                    f"{call.outcome_values} are inconsistent with floor{floor} cap {physical}"
                )
        elif call.outcome == "pre_resume_materialized":
            prepared_floor, previous_floor = call.outcome_values
            if prepared_floor != floor or previous_floor == floor:
                _fail(
                    f"step {step} rank {rank} pre-resume materialization values "
                    f"{call.outcome_values} are inconsistent with floor{floor}"
                )
        else:
            _fail(f"step {step} rank {rank} has unknown resize outcome {call.outcome}")
        last_done[step] = max(last_done.get(step, -1), call.done_position)
    return last_done


def _expected_transitions(plan: dict[str, Any]) -> list[tuple[tuple[int, ...], tuple[int, ...]]]:
    current = tuple(range(WORLD_SIZE))
    transitions: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
    for stage, target in zip(plan["stages"], plan["stage_sets"]):
        if stage < WORLD_SIZE:
            transitions.append((current, target))
            current = target
    return transitions


def _validate_lifecycle(
    shrinks: list[ShrinkEvent],
    restores: list[RestoreEvent],
    plans: dict[int, dict[str, Any]],
    last_resize_done: dict[int, int],
    *,
    rank_identity_known: bool = True,
) -> dict[int, dict[str, Any]]:
    result: dict[int, dict[str, Any]] = {}
    for step, plan in plans.items():
        step_shrinks = [event for event in shrinks if event.step == step]
        step_restores = [event for event in restores if event.step == step]
        expected = _expected_transitions(plan)
        expected_signatures = {(current, target) for current, target in expected}
        actual_signatures = {(event.current_ranks, event.target_ranks) for event in step_shrinks}
        if rank_identity_known:
            transitions = expected
            if actual_signatures != expected_signatures:
                _fail(
                    f"step {step} shrink transition signatures differ from the plan, "
                    f"expected={expected_signatures}, actual={actual_signatures}"
                )
        else:
            expected_sizes = [(len(current), len(target)) for current, target in expected]
            actual_by_size: dict[
                tuple[int, int], list[tuple[tuple[int, ...], tuple[int, ...]]]
            ] = {}
            for current, target in actual_signatures:
                if len(set(current)) != len(current) or len(set(target)) != len(target):
                    _fail(f"step {step} Natural transition contains duplicate ranks")
                if not set(target) < set(current):
                    _fail(f"step {step} Natural survivor ranks are not a strict subset")
                actual_by_size.setdefault((len(current), len(target)), []).append(
                    (current, target)
                )
            executed_count = len(actual_by_size)
            expected_prefix = expected_sizes[:executed_count]
            if (
                executed_count > len(expected_sizes)
                or set(actual_by_size) != set(expected_prefix)
                or any(len(values) != 1 for values in actual_by_size.values())
            ):
                _fail(
                    f"step {step} Natural transition sizes are not a safe plan "
                    f"prefix, planned={expected_sizes}, "
                    f"actual={sorted(actual_by_size)}"
                )
            transitions = []
            current = tuple(range(WORLD_SIZE))
            for transition_size in expected_prefix:
                actual_current, actual_target = actual_by_size[transition_size][0]
                if set(actual_current) != set(current):
                    _fail(
                        f"step {step} Natural transition chain is not nested at "
                        f"{transition_size[0]}->{transition_size[1]}"
                    )
                transitions.append((actual_current, actual_target))
                current = actual_target

        prior_end = last_resize_done[step]
        transition_summary: list[dict[str, Any]] = []
        for current, target in transitions:
            matching = [
                event
                for event in step_shrinks
                if event.current_ranks == current and event.target_ranks == target
            ]
            loggers = [event.logger_rank for event in matching]
            if len(matching) != len(current) or set(loggers) != set(current):
                _fail(
                    f"step {step} transition {len(current)}->{len(target)} has logger ranks "
                    f"{sorted(loggers)}, expected {sorted(current)}"
                )
            start = min(event.position for event in matching)
            end = max(event.position for event in matching)
            if start <= prior_end:
                _fail(f"step {step} shrink transition begins before the prior phase completes")
            prior_end = end
            transition_summary.append(
                {
                    "from": len(current),
                    "to": len(target),
                    "survivor_ranks": list(target),
                }
            )

        if transitions:
            restore_ranks = [event.logger_rank for event in step_restores]
            if len(step_restores) != WORLD_SIZE or set(restore_ranks) != set(range(WORLD_SIZE)):
                _fail(
                    f"step {step} full-world restore has ranks {sorted(restore_ranks)}, "
                    f"expected 0..{WORLD_SIZE - 1}"
                )
            restore_sequences = {event.restore_seq for event in step_restores}
            restore_worlds = {event.world_size for event in step_restores}
            if len(restore_sequences) != 1 or restore_worlds != {WORLD_SIZE}:
                _fail(f"step {step} restore sequence or world size is inconsistent")
            if min(event.position for event in step_restores) <= prior_end:
                _fail(f"step {step} restore begins before shrink transitions complete")
            restore_sequence: int | None = next(iter(restore_sequences))
        else:
            if step_shrinks or step_restores:
                _fail(f"step {step} floor16 plan must not shrink or restore")
            restore_sequence = None

        result[step] = {
            "selected_floor": plan["floor"],
            "executed_floor": (
                len(transitions[-1][1]) if transitions else WORLD_SIZE
            ),
            "planned_transition_count": len(expected),
            "executed_transition_count": len(transitions),
            "runtime_stages_are_safe_prefix": (
                transitions == expected
                if rank_identity_known
                else len(transitions) <= len(expected)
            ),
            "transitions": transition_summary,
            "restore_seq": restore_sequence,
        }
    extra_shrink_steps = sorted({event.step for event in shrinks} - set(plans))
    extra_restore_steps = sorted({event.step for event in restores} - set(plans))
    if extra_shrink_steps or extra_restore_steps:
        _fail(
            f"runtime contains lifecycle events for unexpected steps, "
            f"shrink={extra_shrink_steps}, restore={extra_restore_steps}"
        )
    return result


def validate_training_health(text: str, expected_steps: int, path: Path) -> None:
    abort_ratios = [float(value) for value in ABORT_RE.findall(text)]
    if len(abort_ratios) != expected_steps or any(value != 0.0 for value in abort_ratios):
        _fail(
            f"{path} must contain {expected_steps} zero abort ratios, got {abort_ratios}"
        )
    expected = list(range(1, expected_steps + 1))
    global_steps = [int(value) for value in GLOBAL_STEP_RE.findall(text)]
    rollout_outputs = len(ROLLOUT_OUTPUT_RE.findall(text))
    if rollout_outputs != expected_steps:
        _fail(f"{path} has {rollout_outputs} rollout outputs, expected {expected_steps}")
    if global_steps != expected:
        _fail(f"{path} global steps are {global_steps}, expected {expected}")
    if "After trainer.fit" not in text:
        _fail(f"{path} does not contain the trainer completion marker")


def _find_epoch_dirs(
    run_root: Path,
    expected_epochs: Iterable[int],
    lifecycle: str = DEFAULT_LIFECYCLE,
) -> dict[int, Path]:
    expected = set(expected_epochs)
    observed: dict[int, Path] = {}
    config = lifecycle_config(lifecycle)
    policy = str(config["policy"])
    pattern = re.compile(rf"epoch_(\d+)_mode1_{re.escape(policy)}$")
    for path in run_root.glob(f"epoch_*_mode1_{policy}"):
        match = pattern.fullmatch(path.name)
        if match:
            epoch = int(match.group(1))
            if epoch in observed:
                _fail(f"duplicate epoch directory for epoch {epoch}")
            observed[epoch] = path
    if set(observed) != expected:
        _fail(
            f"epoch directory set mismatch, expected={sorted(expected)}, "
            f"observed={sorted(observed)}"
        )
    return observed


def audit_run(
    run_root: Path,
    cap_env: Path,
    expected_epochs: tuple[int, ...],
    expected_steps: int,
    lifecycle: str = DEFAULT_LIFECYCLE,
) -> dict[str, Any]:
    config = lifecycle_config(lifecycle)
    run_root = run_root.resolve()
    cap_env = cap_env.resolve()
    if not run_root.is_dir():
        _fail(f"run root does not exist: {run_root}")
    cap_values = load_cap_env(cap_env)
    admission, physical = validate_cap_env(cap_values, lifecycle)
    epoch_dirs = _find_epoch_dirs(run_root, expected_epochs, lifecycle)

    epoch_results: list[dict[str, Any]] = []
    observed_prediction_modes: set[str] = set()
    for epoch in expected_epochs:
        epoch_dir = epoch_dirs[epoch]
        raw_plans = load_plans(
            epoch_dir / "oracle" / "length_sorted_rank_plan_summary.json",
            expected_steps,
        )
        prediction_mode = validate_planning_features(raw_plans)
        observed_prediction_modes.add(prediction_mode)
        plans = validate_plans(
            raw_plans,
            admission,
            physical,
            lifecycle,
        )
        logs = sorted((epoch_dir / "logs").glob("*.txt"))
        if len(logs) != 1:
            _fail(f"{epoch_dir} must contain exactly one runtime text log, found {len(logs)}")
        log_path = logs[0]
        calls, shrinks, restores, text = parse_runtime_log(log_path)
        validate_training_health(text, expected_steps, log_path)
        last_done = _validate_resize_calls(calls, plans)
        lifecycle_evidence = _validate_lifecycle(
            shrinks,
            restores,
            plans,
            last_done,
            rank_identity_known=str(config["policy"]) == "planned",
        )
        epoch_results.append(
            {
                "epoch": epoch,
                "epoch_dir": str(epoch_dir),
                "runtime_log": str(log_path),
                "length_prediction_mode": prediction_mode,
                "selected_floors": [plans[step]["floor"] for step in sorted(plans)],
                "resize_calls": len(calls),
                "shrink_events": len(shrinks),
                "restore_events": len(restores),
                "steps": [
                    lifecycle_evidence[step] for step in sorted(lifecycle_evidence)
                ],
            }
        )
    if len(expected_epochs) > 1 and "prompt_max_ema_history" not in observed_prediction_modes:
        _fail("multi-epoch formal run contains no EMA-based planning epoch")
    return {
        "status": "PASS",
        "protocol": (
            f"DeepSeek-V2-Lite AdaFloor {config['label']} formal plan/runtime audit"
        ),
        "lifecycle": lifecycle,
        "runtime_profile": cap_values[f"{config['prefix']}_RUNTIME_PROFILE"],
        "runtime_profile_sha256": cap_values[
            f"{config['prefix']}_RUNTIME_PROFILE_SHA256"
        ],
        "run_root": str(run_root),
        "cap_env": str(cap_env),
        "expected_epochs": list(expected_epochs),
        "expected_steps_per_epoch": expected_steps,
        "world_size": WORLD_SIZE,
        "admission_caps": {str(key): value for key, value in admission.items()},
        "physical_caps": {str(key): value for key, value in physical.items()},
        "epochs": epoch_results,
    }


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
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
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--cap-env", type=Path, required=True)
    parser.add_argument("--expected-epochs", default="1,2")
    parser.add_argument("--expected-steps", type=int, default=5)
    parser.add_argument(
        "--lifecycle",
        choices=tuple(LIFECYCLE_CONFIG),
        default=DEFAULT_LIFECYCLE,
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.expected_steps <= 0:
        parser.error("--expected-steps must be positive")
    try:
        payload = audit_run(
            args.run_root,
            args.cap_env,
            parse_expected_epochs(args.expected_epochs),
            args.expected_steps,
            args.lifecycle,
        )
    except (AuditError, OSError) as exc:
        parser.exit(1, f"FAIL: {exc}\n")
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    if args.output:
        _write_json_atomic(args.output.resolve(), payload)
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
