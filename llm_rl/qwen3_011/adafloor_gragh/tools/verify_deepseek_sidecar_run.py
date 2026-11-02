#!/usr/bin/env python3
"""Fail-closed audit for the DeepSeek-V2-Lite real-sidecar smoke run."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shlex
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
WORLD_SIZE = 16
RELEASE_AREA_UNIT = "rank_token_proxy"
ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
SHRINK_DONE_RE = re.compile(
    r"Elastic parallel shrink done: rank=(\d+) active_ranks=\[([^]]*)\]"
)
RESTORE_RE = re.compile(
    r"Elastic full-world restore segmented timing: rank=(\d+) "
    r"restore_seq=(\d+) world_size=(\d+)"
)
TRAINING_GUARD_RE = re.compile(
    r"Mode1 training memory guard: rank=(\d+) min_free_mib=(\d+) "
    r".*?free_after_bytes=(\d+)"
)
LOG_TIME_RE = re.compile(r"\bINFO\s+(\d{2})-(\d{2})\s+(\d{2}):(\d{2}):(\d{2})\b")
ABORT_RE = re.compile(r"response/aborted_ratio:([0-9.eE+-]+)")
PREEMPT_RE = re.compile(r"preempting request|request preempted", re.IGNORECASE)
OOM_RE = re.compile(
    r"out of memory|OutOfMemoryError|NPU memory is exhausted|"
    r"ACL_ERROR_RT_MEMORY_ALLOCATION",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class LifecycleSpec:
    key: str
    label: str
    prefix: str
    runtime_profile_path: Path
    policy: str
    target_floor: int
    floors: tuple[int, ...]
    stages: tuple[int, ...]

    @property
    def verified_key(self) -> str:
        return f"{self.prefix}_KV_CAPS_VERIFIED"

    @property
    def runtime_profile_id_key(self) -> str:
        return f"{self.prefix}_RUNTIME_PROFILE_ID"

    @property
    def runtime_profile_files_key(self) -> str:
        return f"{self.prefix}_RUNTIME_PROFILE_FILES"

    @property
    def recorded_runtime_profile_key(self) -> str:
        return f"{self.prefix}_RUNTIME_PROFILE"

    @property
    def recorded_runtime_profile_sha_key(self) -> str:
        return f"{self.prefix}_RUNTIME_PROFILE_SHA256"


LIFECYCLES = {
    "natural_f4": LifecycleSpec(
        key="natural_f4",
        label="Natural floor4",
        prefix="DEEPSEEK_N_F4",
        runtime_profile_path=(
            REPO_ROOT / "internal" / "deepseek_v2_lite_natural_f4_runtime_profile.sh"
        ),
        policy="natural",
        target_floor=4,
        floors=(4, 8, 16),
        stages=(8, 4),
    ),
    "natural_f2": LifecycleSpec(
        key="natural_f2",
        label="Natural floor2",
        prefix="DEEPSEEK_N_F2",
        runtime_profile_path=(
            REPO_ROOT / "internal" / "deepseek_v2_lite_natural_f2_runtime_profile.sh"
        ),
        policy="natural",
        target_floor=2,
        floors=(2, 4, 8, 16),
        stages=(8, 4, 2),
    ),
    "planned_f4": LifecycleSpec(
        key="planned_f4",
        label="Planned floor4",
        prefix="DEEPSEEK_P_F4",
        runtime_profile_path=(
            REPO_ROOT / "internal" / "deepseek_v2_lite_planned_f4_runtime_profile.sh"
        ),
        policy="planned",
        target_floor=4,
        floors=(4, 8, 16),
        stages=(8, 4),
    ),
    "planned_f2": LifecycleSpec(
        key="planned_f2",
        label="Planned floor2",
        prefix="DEEPSEEK_P_F2",
        runtime_profile_path=(
            REPO_ROOT / "internal" / "deepseek_v2_lite_planned_f2_runtime_profile.sh"
        ),
        policy="planned",
        target_floor=2,
        floors=(2, 4, 8, 16),
        stages=(8, 4, 2),
    ),
}


class VerificationError(RuntimeError):
    """Raised when a sidecar smoke artifact violates a required invariant."""


def _fail(message: str) -> None:
    raise VerificationError(message)


def _resolved(value: str | Path) -> Path:
    return Path(value).expanduser().resolve()


def _strip_ansi(value: str) -> str:
    return ANSI_RE.sub("", value).replace("\r", "")


def _parse_rank_csv(value: str, label: str) -> tuple[int, ...]:
    try:
        ranks = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise VerificationError(f"{label} contains a non-integer rank") from exc
    if len(set(ranks)) != len(ranks) or any(rank < 0 or rank >= WORLD_SIZE for rank in ranks):
        _fail(f"{label} is not a unique subset of ranks 0 through {WORLD_SIZE - 1}")
    return ranks


def load_env(path: Path) -> dict[str, str]:
    if not path.is_file():
        _fail(f"environment file does not exist: {path}")
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
        try:
            words = shlex.split(raw_value, comments=True, posix=True)
        except ValueError as exc:
            raise VerificationError(f"cannot parse {path}:{line_number}: {exc}") from exc
        if len(words) > 1:
            _fail(f"{path}:{line_number} contains multiple shell words")
        values[key.strip()] = words[0] if words else ""
    return values


def _positive_block_value(values: dict[str, str], name: str) -> int:
    try:
        value = int(values[name])
    except (KeyError, ValueError) as exc:
        raise VerificationError(f"missing or invalid {name}") from exc
    if value <= 0 or value % 128:
        _fail(f"{name} must be a positive multiple of 128")
    return value


def _nonnegative_block_value(values: dict[str, str], name: str) -> int:
    try:
        value = int(values[name])
    except (KeyError, ValueError) as exc:
        raise VerificationError(f"missing or invalid {name}") from exc
    if value < 0 or value % 128:
        _fail(f"{name} must be a nonnegative multiple of 128")
    return value


def _profile_export(path: Path, key: str) -> str:
    if not path.is_file():
        _fail(f"runtime profile does not exist: {path}")
    prefix = f"export {key}="
    matches = [
        line.strip()[len(prefix) :]
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip().startswith(prefix)
    ]
    if len(matches) != 1:
        _fail(f"{path} must export {key} exactly once")
    try:
        words = shlex.split(matches[0], comments=True, posix=True)
    except ValueError as exc:
        raise VerificationError(f"cannot parse {key} from {path}: {exc}") from exc
    if len(words) != 1:
        _fail(f"{key} in {path} must contain one shell word")
    return words[0]


def runtime_profile_provenance(lifecycle: str) -> tuple[str, str, tuple[str, ...]]:
    try:
        spec = LIFECYCLES[lifecycle]
    except KeyError as exc:
        raise VerificationError(f"unknown lifecycle: {lifecycle}") from exc
    profile_id = _profile_export(spec.runtime_profile_path, spec.runtime_profile_id_key)
    raw_files = _profile_export(
        spec.runtime_profile_path, spec.runtime_profile_files_key
    )
    profile_files = tuple(item.strip() for item in raw_files.split(",") if item.strip())
    if not profile_files:
        _fail(f"{spec.label} runtime profile declares an empty source closure")

    hasher = hashlib.sha256()
    seen: set[Path] = set()
    resolved_files: list[Path] = []
    for relative in profile_files:
        path = (REPO_ROOT / relative).resolve()
        try:
            canonical_name = path.relative_to(REPO_ROOT).as_posix()
        except ValueError as exc:
            raise VerificationError(
                f"{spec.label} runtime profile source is outside the repository: {path}"
            ) from exc
        if path in seen:
            _fail(f"{spec.label} runtime profile repeats source {canonical_name}")
        if not path.is_file():
            _fail(f"{spec.label} runtime profile source is missing: {path}")
        seen.add(path)
        resolved_files.append(path)
        name = canonical_name.encode("utf-8")
        content = path.read_bytes()
        hasher.update(len(name).to_bytes(8, "big"))
        hasher.update(name)
        hasher.update(len(content).to_bytes(8, "big"))
        hasher.update(content)
    if spec.runtime_profile_path.resolve() not in set(resolved_files):
        _fail(f"{spec.label} runtime profile closure omits its selected profile")
    return profile_id, hasher.hexdigest(), profile_files


def _validate_caps(
    values: dict[str, str], spec: LifecycleSpec
) -> tuple[
    dict[int, int],
    dict[int, int],
    dict[int, int],
    int | None,
    Path,
    Path,
    Path,
]:
    if values.get(spec.verified_key) != "1":
        _fail(f"DeepSeek {spec.label} KV caps are not VERIFIED")
    if values.get("DEEPSEEK_KV_CAP_TARGET_RATIO") != "1.0":
        _fail("DEEPSEEK_KV_CAP_TARGET_RATIO must equal 1.0")
    if values.get("DEEPSEEK_KV_CAP_BLOCK_SIZE") != "128":
        _fail("DEEPSEEK_KV_CAP_BLOCK_SIZE must equal 128")
    admission_caps: dict[int, int] = {}
    physical_caps: dict[int, int] = {}
    for floor in spec.floors:
        admission = _positive_block_value(
            values, f"{spec.prefix}_KV_ADMISSION_FLOOR{floor}"
        )
        physical = _positive_block_value(
            values, f"{spec.prefix}_KV_PHYSICAL_FLOOR{floor}"
        )
        if admission >= physical:
            _fail(f"floor{floor} admission capacity must be below physical capacity")
        admission_caps[floor] = admission
        physical_caps[floor] = physical

    planned_headroom: dict[int, int] = {}
    training_min_free_mib: int | None = None
    if spec.policy == "planned":
        for floor in spec.floors:
            planned_headroom[floor] = _nonnegative_block_value(
                values, f"{spec.prefix}_HEADROOM_FLOOR{floor}"
            )
        training_key = f"{spec.prefix}_TRAINING_MIN_FREE_MIB"
        try:
            training_min_free_mib = int(values[training_key])
        except (KeyError, ValueError) as exc:
            raise VerificationError(f"missing or invalid {training_key}") from exc
        if training_min_free_mib <= 0:
            _fail(f"{training_key} must be measured and positive")

    profile_id, profile_sha256, _profile_files = runtime_profile_provenance(spec.key)
    if values.get(spec.recorded_runtime_profile_key) != profile_id:
        _fail(f"{spec.label} runtime profile does not match its VERIFIED KV caps")
    if values.get(spec.recorded_runtime_profile_sha_key) != profile_sha256:
        _fail(f"{spec.label} runtime profile closure does not match its VERIFIED KV caps")

    common_text = values.get("DEEPSEEK_KV_CAP_COMMON_EPOCH0_ROOT", "")
    if not common_text:
        _fail("DeepSeek cap provenance does not name a common epoch0 root")
    common_root = _resolved(common_text)
    if not (common_root / "DO_NOT_DELETE_COMMON_EPOCH0_CHECKPOINT").is_file():
        _fail(f"common epoch0 completion marker is missing under {common_root}")
    reuse = load_env(common_root / "reuse.env")
    checkpoint_text = reuse.get("DYNAMIC_INITIAL_RESUME_CKPT", "")
    if not checkpoint_text:
        _fail("common epoch0 reuse environment has no resume checkpoint")
    checkpoint = _resolved(checkpoint_text)
    if not (checkpoint / "actor").is_dir() or not (
        checkpoint / ".PRESERVE_COMMON_EPOCH0"
    ).is_file():
        _fail(f"common epoch0 resume checkpoint is incomplete: {checkpoint}")
    trigger_text = values.get("DEEPSEEK_KV_CAP_PROBE_HISTORY_ROOT", "")
    if not trigger_text:
        _fail("DeepSeek cap provenance does not name its positive release history")
    trigger_root = _resolved(trigger_text)
    for relative in (
        "offline_planning_history.json",
        "kv_probe_trigger_manifest.json",
        "rollout_data/1.jsonl",
    ):
        if not (trigger_root / relative).is_file():
            _fail(f"positive release history is incomplete: {trigger_root / relative}")
    return (
        admission_caps,
        physical_caps,
        planned_headroom,
        training_min_free_mib,
        common_root,
        checkpoint,
        trigger_root,
    )


def _only_file(directory: Path, pattern: str, label: str) -> Path:
    files = sorted(directory.glob(pattern))
    if len(files) != 1:
        _fail(f"{directory}: expected exactly one {label}, found {len(files)}")
    return files[0]


def _load_plan(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise VerificationError(f"cannot read plan summary {path}: {exc}") from exc
    steps = payload.get("steps") if isinstance(payload, dict) else payload
    if not isinstance(steps, list) or len(steps) != 1 or not isinstance(steps[0], dict):
        _fail("sidecar smoke must contain exactly one planned step")
    return steps[0]


def _as_exact_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail(f"{label} must be numeric")
    converted = float(value)
    if not math.isfinite(converted) or not converted.is_integer():
        _fail(f"{label} must be an exact integer")
    return int(converted)


def _validate_plan(
    plan: dict[str, Any],
    admission: int,
    physical: int,
    trigger_root: Path,
    spec: LifecycleSpec,
) -> tuple[tuple[int, ...], ...]:
    if plan.get("feasible") is not True:
        _fail(f"the forced floor{spec.target_floor} sidecar plan is not feasible")
    if _as_exact_int(plan.get("step"), "plan step") != 1:
        _fail("sidecar smoke plan must be step 1")
    if _as_exact_int(plan.get("selected_floor"), "selected floor") != spec.target_floor:
        _fail(f"sidecar smoke did not select forced floor{spec.target_floor}")
    stages_raw = plan.get("shrink_stages")
    if not isinstance(stages_raw, list):
        _fail("plan shrink_stages must be a list")
    stages = tuple(_as_exact_int(value, "plan shrink stage") for value in stages_raw)
    if stages != spec.stages:
        path = " to ".join(str(value) for value in (WORLD_SIZE, *spec.stages))
        _fail(f"sidecar smoke must use the {path} {spec.policy} lifecycle")

    raw_stage_sets = plan.get("stage_survivor_ranks")
    if not isinstance(raw_stage_sets, list) or len(raw_stage_sets) != len(spec.stages):
        _fail("plan must contain one survivor set for every shrink stage")
    stage_sets: list[tuple[int, ...]] = []
    previous = set(range(WORLD_SIZE))
    for stage, raw_ranks in zip(spec.stages, raw_stage_sets):
        if not isinstance(raw_ranks, list) or any(
            isinstance(rank, bool) or not isinstance(rank, int) for rank in raw_ranks
        ):
            _fail(f"plan floor{stage} survivor ranks must be an integer list")
        ranks = _parse_rank_csv(
            ",".join(str(rank) for rank in raw_ranks),
            f"plan floor{stage} survivor ranks",
        )
        if len(ranks) != stage:
            _fail(f"plan floor{stage} survivor set must contain {stage} ranks")
        if not set(ranks).issubset(previous):
            _fail(f"plan floor{stage} survivor set is not nested")
        stage_sets.append(ranks)
        previous = set(ranks)
    if tuple(plan.get("intermediate_survivor_ranks", ())) != stage_sets[0]:
        _fail("plan intermediate survivor set differs from its first stage")
    survivors = tuple(plan.get("final_survivor_ranks", ()))
    if survivors != stage_sets[-1]:
        _fail("plan final survivor set differs from its last stage")
    if _as_exact_int(plan.get("kv_admission_cap"), "plan admission cap") != admission:
        _fail(
            f"plan floor{spec.target_floor} admission cap differs from the VERIFIED cap"
        )
    if _as_exact_int(plan.get("kv_cap"), "plan physical cap") != physical:
        _fail(
            f"plan floor{spec.target_floor} physical cap differs from the VERIFIED cap"
        )
    peak = float(plan.get("max_adjusted_rank_peak_tokens", math.inf))
    if not math.isfinite(peak) or peak > admission:
        _fail(f"plan exceeds its VERIFIED floor{spec.target_floor} admission capacity")
    release_area = float(plan.get("release_area", 0))
    if not math.isfinite(release_area) or release_area <= 0:
        _fail(
            f"forced floor{spec.target_floor} plan has no positive predicted release window"
        )
    if plan.get("release_area_unit") not in (None, RELEASE_AREA_UNIT):
        _fail("sidecar plan has an invalid release area unit")
    if plan.get("rank_matching_policy") != "release_area":
        _fail("sidecar smoke did not use release-area rank matching")
    if plan.get("tail_guard_enabled") is not True:
        _fail("sidecar smoke did not retain TailGuard")
    histories = plan.get("length_prediction_baseline_dirs")
    if not isinstance(histories, list) or trigger_root not in {
        _resolved(item) for item in histories if isinstance(item, str)
    }:
        _fail("plan provenance does not use the VERIFIED positive release history")
    return tuple(stage_sets)


def _validate_primary_log(
    text: str,
    common_checkpoint: Path,
    spec: LifecycleSpec,
    training_min_free_mib: int | None,
) -> tuple[tuple[tuple[int, ...], ...], list[str]]:
    clean = _strip_ansi(text)
    if PREEMPT_RE.search(clean):
        _fail("primary rollout contains a preemption")
    if OOM_RE.search(clean):
        _fail("primary rollout contains an OOM")
    aborts = [float(value) for value in ABORT_RE.findall(clean)]
    if not aborts or any(value != 0 for value in aborts):
        _fail("primary rollout does not report zero aborted responses")
    if "rollout_output_time_s:" not in clean:
        _fail("primary rollout did not complete")
    if "training/global_step:1" not in clean:
        _fail("primary run did not complete its single training step")
    if "After trainer.fit" not in clean:
        _fail("primary trainer did not exit normally")
    if f"Resuming from {common_checkpoint}" not in clean:
        _fail("primary run did not resume from the VERIFIED common epoch0 checkpoint")
    if spec.policy == "planned":
        if training_min_free_mib is None:
            _fail("Planned lifecycle has no VERIFIED training memory reserve")
        guard_records: dict[int, set[tuple[int, int]]] = {}
        for match in TRAINING_GUARD_RE.finditer(clean):
            rank, recorded_min_mib, free_after_bytes = map(int, match.groups())
            guard_records.setdefault(rank, set()).add(
                (recorded_min_mib, free_after_bytes)
            )
        if set(guard_records) != set(range(WORLD_SIZE)):
            _fail("Planned runtime did not execute the training memory guard on all ranks")
        required_bytes = training_min_free_mib * 1024 * 1024
        for rank, records in guard_records.items():
            if len(records) != 1:
                _fail(f"Planned rank {rank} has inconsistent training guard records")
            recorded_min_mib, free_after_bytes = next(iter(records))
            if recorded_min_mib != training_min_free_mib:
                _fail(f"Planned rank {rank} used the wrong training memory reserve")
            if free_after_bytes < required_bytes:
                _fail(f"Planned rank {rank} violated the training memory guard")

    sets_by_stage: dict[int, list[tuple[int, ...]]] = {
        stage: [] for stage in spec.stages
    }
    lines_by_stage: dict[int, list[str]] = {stage: [] for stage in spec.stages}
    observed_stage_sequence: list[int] = []
    for line in clean.splitlines():
        match = SHRINK_DONE_RE.search(line)
        if match is None:
            continue
        active = _parse_rank_csv(match.group(2), "runtime active ranks")
        stage = len(active)
        if stage not in sets_by_stage:
            _fail(f"primary log contains an unexpected floor{stage} shrink_done event")
        sets_by_stage[stage].append(active)
        lines_by_stage[stage].append(line)
        if not observed_stage_sequence or observed_stage_sequence[-1] != stage:
            observed_stage_sequence.append(stage)

    runtime_stage_sets: list[tuple[int, ...]] = []
    for stage in spec.stages:
        observed = sets_by_stage[stage]
        if not observed or len(set(observed)) != 1:
            _fail(
                f"primary log does not contain one consistent floor{stage} "
                "shrink_done set"
            )
        runtime_stage_sets.append(observed[0])
    if tuple(observed_stage_sequence) != spec.stages:
        _fail("primary shrink_done stages are not in the planned order")
    for previous, current in zip(
        (set(range(WORLD_SIZE)), *(set(ranks) for ranks in runtime_stage_sets[:-1])),
        runtime_stage_sets,
    ):
        if not set(current).issubset(previous):
            _fail("primary runtime survivor sets are not nested")

    restores: dict[int, set[int]] = {}
    for match in RESTORE_RE.finditer(clean):
        rank, restore_seq, world_size = map(int, match.groups())
        if world_size != WORLD_SIZE:
            _fail("primary restore did not return to the 16-rank world")
        restores.setdefault(restore_seq, set()).add(rank)
    if len(restores) != 1 or next(iter(restores.values())) != set(range(WORLD_SIZE)):
        _fail("primary run does not contain one complete 16-rank restore")
    return tuple(runtime_stage_sets), lines_by_stage[spec.target_floor]


def _parse_kv_lines(text: str) -> tuple[dict[str, list[str]], list[str]]:
    values: dict[str, list[str]] = {}
    lines = [_strip_ansi(line).strip() for line in text.splitlines()]
    for line in lines:
        if not line or line.startswith("{") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
            values.setdefault(key, []).append(value)
    return values, lines


def _one(values: dict[str, list[str]], key: str) -> str:
    found = values.get(key, [])
    if len(found) != 1:
        _fail(f"expected exactly one {key} event, found {len(found)}")
    return found[0]


def _validate_lease(
    lease_text: str,
    train_text: str,
    runtime_survivors: tuple[int, ...],
    spec: LifecycleSpec,
) -> tuple[tuple[int, ...], float]:
    values, lines = _parse_kv_lines(lease_text)
    if _one(values, "watch_start_trigger") != "shrink_done":
        _fail("sidecar watcher did not use the shrink_done trigger")
    if _one(values, "watch_expected_active_ranks") != str(spec.target_floor):
        _fail(
            "sidecar watcher did not require "
            f"{spec.target_floor} active primary ranks"
        )
    if _one(values, "watch_world_size") != str(WORLD_SIZE):
        _fail("sidecar watcher world size is not 16")
    if _one(values, "sidecar_devices_source") != "auto_from_inactive_ranks":
        _fail("sidecar device assignment was not derived from detached ranks")

    active = _parse_rank_csv(_one(values, "sidecar_active_ranks"), "lease active ranks")
    devices = _parse_rank_csv(_one(values, "sidecar_devices"), "lease sidecar devices")
    if active != runtime_survivors:
        _fail(
            "lease survivor ranks differ from the runtime "
            f"floor{spec.target_floor} survivor set"
        )
    expected_devices = tuple(rank for rank in range(WORLD_SIZE) if rank not in set(active))
    if devices != expected_devices:
        _fail("sidecar lease is not exactly the complement of active primary ranks")

    detected = float(_one(values, "shrink_window_detected_time"))
    start = float(_one(values, "sidecar_start_time"))
    if not math.isfinite(detected) or not math.isfinite(start) or start < detected:
        _fail("sidecar start time precedes the detected shrink_done event")
    copied = _one(values, "shrink_window_line")
    copied_match = SHRINK_DONE_RE.search(copied)
    if copied_match is None:
        _fail("lease trigger is not a shrink_done line")
    copied_active = _parse_rank_csv(
        copied_match.group(2), "lease shrink_done active ranks"
    )
    if copied_active != runtime_survivors:
        _fail("lease did not trigger on the final shrink_done stage")
    clean_train = _strip_ansi(train_text)
    if _strip_ansi(copied) not in clean_train:
        _fail("lease shrink_done trigger is absent from the primary runtime log")

    detected_index = next(i for i, line in enumerate(lines) if line.startswith("shrink_window_detected_time="))
    trigger_index = next(i for i, line in enumerate(lines) if line.startswith("shrink_window_line="))
    start_index = next(i for i, line in enumerate(lines) if line.startswith("sidecar_start_time="))
    if not detected_index < trigger_index < start_index:
        _fail("lease log does not place sidecar start after shrink_done detection")
    return devices, start


def _json_events(text: str) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for line in text.splitlines():
        line = _strip_ansi(line).strip()
        if not line.startswith("{"):
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and isinstance(value.get("event"), str):
            events.append(value)
    return events


def _single_event(events: list[dict[str, Any]], name: str) -> dict[str, Any]:
    matches = [event for event in events if event.get("event") == name]
    if len(matches) != 1:
        _fail(f"expected exactly one {name} event, found {len(matches)}")
    return matches[0]


def _validate_infer_log(
    text: str, leased_devices: tuple[int, ...], expected_model: Path
) -> tuple[tuple[int, ...], float, int]:
    values, _lines = _parse_kv_lines(text)
    if _one(values, "sidecar_exit_code") != "0":
        _fail("real sidecar inference did not exit successfully")
    if _one(values, "sidecar_killed_by_deadline") != "0":
        _fail("real sidecar inference was killed by its deadline")
    if _one(values, "sidecar_replica_count") != "1":
        _fail("short sidecar smoke must use exactly one replica")
    if _one(values, "sidecar_parallel_mode") != "dp":
        _fail("short sidecar smoke must use the single-rank DP layout")
    if _one(values, "sidecar_tensor_parallel_size") != "1":
        _fail("short sidecar smoke tensor parallel size must equal one")
    if _one(values, "sidecar_data_parallel_size") != "1":
        _fail("short sidecar smoke data parallel size must equal one")
    if _resolved(_one(values, "sidecar_model")) != expected_model:
        _fail("sidecar inference used an unexpected model")

    groups = [group.strip() for group in _one(values, "sidecar_device_groups").split(";") if group.strip()]
    if len(groups) != 1:
        _fail("short sidecar smoke must materialize one device group")
    used_devices = _parse_rank_csv(groups[0], "actual sidecar device group")
    if len(used_devices) != 1 or not set(used_devices).issubset(leased_devices):
        _fail("actual sidecar device group is not a single detached rank")

    end_time = float(_one(values, "sidecar_end_time"))
    if not math.isfinite(end_time):
        _fail("sidecar end time is not finite")
    events = _json_events(text)
    load = _single_event(events, "sidecar_load_start")
    if _resolved(str(load.get("model_path", ""))) != expected_model:
        _fail("sidecar load event used an unexpected model")
    visible = _parse_rank_csv(str(load.get("devices", "")), "sidecar visible devices")
    if visible != used_devices:
        _fail("sidecar process visible devices differ from its device group")
    done = _single_event(events, "sidecar_done")
    requests = _as_exact_int(done.get("num_requests"), "sidecar request count")
    output_tokens = _as_exact_int(done.get("num_output_tokens"), "sidecar output token count")
    if requests < 1 or output_tokens < 1:
        _fail("sidecar did not finish a request with generated tokens")
    return used_devices, end_time, output_tokens


def _restore_timestamp(train_text: str, reference_epoch_s: float) -> float:
    clean = _strip_ansi(train_text)
    restore_line = next(
        (
            line
            for line in clean.splitlines()
            if "Elastic full-world restore" in line or "Elastic parallel restore done" in line
        ),
        None,
    )
    if restore_line is None:
        _fail("primary log contains no observable full-world restore boundary")
    match = LOG_TIME_RE.search(restore_line)
    if match is None:
        _fail("cannot extract a timestamp from the first full-world restore event")
    month, day, hour, minute, second = map(int, match.groups())
    reference = datetime.fromtimestamp(reference_epoch_s, tz=timezone.utc)
    candidates: list[float] = []
    for year in (reference.year - 1, reference.year, reference.year + 1):
        try:
            candidates.append(
                datetime(year, month, day, hour, minute, second, tzinfo=timezone.utc).timestamp()
            )
        except ValueError:
            continue
    if not candidates:
        _fail("first restore event has an invalid timestamp")
    return min(candidates, key=lambda value: abs(value - reference_epoch_s))


def _validate_outputs(path: Path) -> tuple[int, int]:
    if not path.is_file():
        _fail(f"sidecar output file does not exist: {path}")
    records = 0
    generated_tokens = 0
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw_line.strip():
            continue
        try:
            row = json.loads(raw_line)
        except json.JSONDecodeError as exc:
            raise VerificationError(f"invalid sidecar output JSON at line {line_number}") from exc
        if not isinstance(row, dict) or row.get("sidecar_status") is not None:
            _fail(f"sidecar output line {line_number} is not a completed generation")
        outputs = row.get("outputs")
        if not isinstance(outputs, list) or not outputs:
            _fail(f"sidecar output line {line_number} has no completions")
        row_tokens = 0
        for output in outputs:
            if not isinstance(output, dict):
                _fail(f"sidecar output line {line_number} contains an invalid completion")
            row_tokens += _as_exact_int(
                output.get("token_ids_len"), f"sidecar output line {line_number} token count"
            )
        if row_tokens <= 0:
            _fail(f"sidecar output line {line_number} contains no generated tokens")
        records += 1
        generated_tokens += row_tokens
    if records == 0:
        _fail("sidecar output file is empty")
    return records, generated_tokens


def verify_run(
    run_root: Path,
    cap_env: Path,
    expected_model_path: Path,
    lifecycle: str = "natural_f4",
) -> dict[str, Any]:
    try:
        lifecycle_spec = LIFECYCLES[lifecycle]
    except KeyError as exc:
        raise VerificationError(f"unknown lifecycle: {lifecycle}") from exc
    run_root = run_root.resolve()
    expected_model = expected_model_path.resolve()
    values = load_env(cap_env.resolve())
    (
        admission_caps,
        physical_caps,
        planned_headroom,
        training_min_free_mib,
        common_root,
        common_checkpoint,
        trigger_root,
    ) = _validate_caps(values, lifecycle_spec)
    admission = admission_caps[lifecycle_spec.target_floor]
    physical = physical_caps[lifecycle_spec.target_floor]

    epoch_dirs = sorted(run_root.glob("epoch_*_mode1_*"))
    expected_epoch_name = f"epoch_001_mode1_{lifecycle_spec.policy}"
    if [path.name for path in epoch_dirs] != [expected_epoch_name]:
        _fail(f"sidecar smoke root must contain only {expected_epoch_name}")
    epoch_dir = epoch_dirs[0]
    plan_path = epoch_dir / "oracle" / "length_sorted_rank_plan_summary.json"
    plan = _load_plan(plan_path)
    plan_stage_sets = _validate_plan(
        plan, admission, physical, trigger_root, lifecycle_spec
    )

    train_log = _only_file(epoch_dir / "logs", "*.txt", "primary runtime log")
    train_text = train_log.read_text(encoding="utf-8", errors="replace")
    runtime_stage_sets, _shrink_lines = _validate_primary_log(
        train_text,
        common_checkpoint,
        lifecycle_spec,
        training_min_free_mib,
    )
    if runtime_stage_sets != plan_stage_sets:
        _fail("runtime survivor stages differ from the planned survivor stages")
    runtime_survivors = runtime_stage_sets[-1]

    sidecar_dir = epoch_dir / "sidecar"
    lease_log = sidecar_dir / "lease.log"
    infer_log = sidecar_dir / "infer.log"
    output_path = sidecar_dir / "outputs.jsonl"
    if not lease_log.is_file() or not infer_log.is_file():
        _fail(f"sidecar lease or inference log is missing under {sidecar_dir}")
    leased_devices, start_time = _validate_lease(
        lease_log.read_text(encoding="utf-8", errors="replace"),
        train_text,
        runtime_survivors,
        lifecycle_spec,
    )
    used_devices, end_time, logged_tokens = _validate_infer_log(
        infer_log.read_text(encoding="utf-8", errors="replace"), leased_devices, expected_model
    )
    records, output_tokens = _validate_outputs(output_path)
    if logged_tokens < output_tokens:
        _fail("sidecar completion log reports fewer tokens than the durable output")
    restore_time = _restore_timestamp(train_text, end_time)
    if end_time >= restore_time:
        _fail(
            "sidecar did not demonstrably terminate before the first full-world restore event"
        )

    detached = tuple(rank for rank in range(WORLD_SIZE) if rank not in set(runtime_survivors))
    return {
        "status": "PASS",
        "lifecycle": lifecycle_spec.label,
        "lifecycle_key": lifecycle_spec.key,
        "policy": lifecycle_spec.policy,
        "steps": 1,
        "selected_floor": lifecycle_spec.target_floor,
        "stages": list(lifecycle_spec.stages),
        "stage_survivor_ranks": [list(ranks) for ranks in runtime_stage_sets],
        "survivor_ranks": list(runtime_survivors),
        "detached_ranks": list(detached),
        "leased_ranks": list(leased_devices),
        "sidecar_used_ranks": list(used_devices),
        "sidecar_model": str(expected_model),
        "sidecar_output_records": records,
        "sidecar_output_tokens": output_tokens,
        "sidecar_start_epoch_s": start_time,
        "sidecar_end_epoch_s": end_time,
        "first_restore_log_epoch_s": restore_time,
        f"floor{lifecycle_spec.target_floor}_admission_tokens": admission,
        f"floor{lifecycle_spec.target_floor}_physical_tokens": physical,
        "planned_headroom_tokens": {
            str(floor): value for floor, value in planned_headroom.items()
        },
        "planned_training_min_free_mib": training_min_free_mib,
        "common_epoch0_root": str(common_root),
        "common_epoch0_checkpoint": str(common_checkpoint),
        "planning_history_root": str(trigger_root),
        "artifacts": {
            "run_root": str(run_root),
            "plan": str(plan_path),
            "primary_log": str(train_log),
            "lease_log": str(lease_log),
            "inference_log": str(infer_log),
            "sidecar_output": str(output_path),
            "cap_env": str(cap_env.resolve()),
        },
        "invariants": {
            "verified_lifecycle_caps": True,
            "verified_runtime_profile_closure": True,
            **(
                {"planned_training_guard": True}
                if lifecycle_spec.policy == "planned"
                else {}
            ),
            f"forced_floor{lifecycle_spec.target_floor}": True,
            "complete_staged_shrink": True,
            "start_after_shrink_done": True,
            "only_detached_ranks": True,
            "terminated_before_restore": True,
            "nonempty_real_generation": True,
            "primary_zero_preemption_oom_abort": True,
        },
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--cap-env", type=Path, required=True)
    parser.add_argument(
        "--lifecycle",
        choices=tuple(LIFECYCLES),
        default="natural_f4",
        help="Natural lifecycle to verify. Defaults to natural_f4.",
    )
    parser.add_argument(
        "--expected-model-path", type=Path, default=Path("/data/Qwen2.5-1.5B-Instruct")
    )
    parser.add_argument("--summary", type=Path)
    args = parser.parse_args(argv)
    summary_path = args.summary or args.run_root / "SIDECAR_SMOKE_SUMMARY.json"
    try:
        summary = verify_run(
            args.run_root,
            args.cap_env,
            args.expected_model_path,
            lifecycle=args.lifecycle,
        )
        _write_json(summary_path, summary)
    except VerificationError as exc:
        print(f"DeepSeek sidecar verification failed: {exc}", file=sys.stderr)
        return 2
    print(
        "DeepSeek sidecar verification PASS "
        f"lifecycle={summary['lifecycle_key']} "
        f"records={summary['sidecar_output_records']} "
        f"tokens={summary['sidecar_output_tokens']} "
        f"used_ranks={summary['sidecar_used_ranks']} "
        f"summary={summary_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
