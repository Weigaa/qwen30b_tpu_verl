"""Fail-closed loading and lookup for fixed-work rollout replay traces."""

from __future__ import annotations

import hashlib
import json
import os
import threading
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Any


SCHEMA_VERSION = 3
TRACE_FORMAT = "deepseek_batch64_fixed_work_replay"
TRACE_SHA256_ENV = "VERL_FIXED_WORK_REPLAY_SHA256"
EXPECTED_ROWS_PER_STEP = 1024
PROMPTS_PER_STEP = 64
RESPONSES_PER_PROMPT = 16
WORLD_SIZE = 16
PROMPTS_PER_RANK = PROMPTS_PER_STEP // WORLD_SIZE
MAX_RESPONSE_LENGTH = 16384
IDENTITY_FIELDS = (
    "rollout_prompt_hash",
    "rollout_sample_index",
    "rollout_request_seed",
)
SOURCE_LENGTH_FIELD = "source_decoded_response_length"
TARGET_LENGTH_FIELD = "target_response_length"

RequestIdentity = tuple[str, int, int]
OccurrenceKey = tuple[int, int]


class FixedWorkReplayError(RuntimeError):
    pass


class FixedWorkReplay:
    __slots__ = (
        "_path",
        "_trace_sha256",
        "_target_lengths_by_step",
        "_source_lengths_by_step",
        "_identities_by_ordinal",
        "_identity_ordinals_by_step",
        "_occurrences_by_ordinal",
        "_records_by_occurrence",
        "_prompt_occurrences_by_step",
        "_step_caps",
        "_steps",
        "_record_count",
        "_source_generated_tokens",
        "_target_generated_tokens",
        "_plan_path",
        "_plan_sha256",
    )

    def __init__(
        self,
        *,
        path: Path,
        trace_sha256: str,
        target_lengths_by_step: dict[int, tuple[int, ...]],
        source_lengths_by_step: dict[int, tuple[int, ...]],
        identities_by_ordinal: dict[int, tuple[RequestIdentity, ...]],
        occurrences_by_ordinal: dict[int, tuple[int, ...]],
        step_caps: dict[int, int],
        source_generated_tokens: int,
        target_generated_tokens: int,
        plan_path: Path | None,
        plan_sha256: str | None,
    ) -> None:
        self._path = path
        self._trace_sha256 = trace_sha256
        self._target_lengths_by_step = MappingProxyType(
            {
                step: MappingProxyType(dict(enumerate(lengths)))
                for step, lengths in sorted(target_lengths_by_step.items())
            }
        )
        self._source_lengths_by_step = MappingProxyType(
            {
                step: MappingProxyType(dict(enumerate(lengths)))
                for step, lengths in sorted(source_lengths_by_step.items())
            }
        )
        self._identities_by_ordinal = MappingProxyType(dict(identities_by_ordinal))
        self._occurrences_by_ordinal = MappingProxyType(dict(occurrences_by_ordinal))
        identity_ordinals_by_step: dict[
            int, Mapping[RequestIdentity, tuple[int, ...]]
        ] = {}
        for step, identities in sorted(identities_by_ordinal.items()):
            mutable_index: dict[RequestIdentity, list[int]] = {}
            for row_ordinal, identity in enumerate(identities):
                mutable_index.setdefault(identity, []).append(row_ordinal)
            identity_ordinals_by_step[step] = MappingProxyType(
                {
                    identity: tuple(ordinals)
                    for identity, ordinals in mutable_index.items()
                }
            )
        self._identity_ordinals_by_step = MappingProxyType(
            identity_ordinals_by_step
        )
        records_by_occurrence: dict[
            OccurrenceKey, tuple[int, int, RequestIdentity, int, int]
        ] = {}
        prompt_occurrences_by_step: dict[int, tuple[int, ...]] = {}
        for step, identities in sorted(identities_by_ordinal.items()):
            occurrences = occurrences_by_ordinal[step]
            targets = target_lengths_by_step[step]
            sources = source_lengths_by_step[step]
            ordered_occurrences: list[int] = []
            seen_occurrences: set[int] = set()
            for row_ordinal, (occurrence, identity) in enumerate(
                zip(occurrences, identities, strict=True)
            ):
                key = (occurrence, identity[1])
                if key in records_by_occurrence:
                    raise FixedWorkReplayError(
                        f"duplicate global stable request key: {key}"
                    )
                records_by_occurrence[key] = (
                    step,
                    row_ordinal,
                    identity,
                    sources[row_ordinal],
                    targets[row_ordinal],
                )
                if occurrence not in seen_occurrences:
                    ordered_occurrences.append(occurrence)
                    seen_occurrences.add(occurrence)
            prompt_occurrences_by_step[step] = tuple(ordered_occurrences)
        self._records_by_occurrence = MappingProxyType(records_by_occurrence)
        self._prompt_occurrences_by_step = MappingProxyType(
            prompt_occurrences_by_step
        )
        self._step_caps = MappingProxyType(dict(sorted(step_caps.items())))
        self._steps = tuple(self._target_lengths_by_step)
        self._record_count = sum(
            len(lengths) for lengths in self._target_lengths_by_step.values()
        )
        self._source_generated_tokens = source_generated_tokens
        self._target_generated_tokens = target_generated_tokens
        self._plan_path = plan_path
        self._plan_sha256 = plan_sha256

    @property
    def path(self) -> Path:
        return self._path

    @property
    def trace_sha256(self) -> str:
        return self._trace_sha256

    @property
    def steps(self) -> tuple[int, ...]:
        return self._steps

    @property
    def record_count(self) -> int:
        return self._record_count

    @property
    def source_generated_tokens(self) -> int:
        return self._source_generated_tokens

    @property
    def target_generated_tokens(self) -> int:
        return self._target_generated_tokens

    @property
    def adafloor_plan_path(self) -> Path | None:
        return self._plan_path

    @property
    def adafloor_plan_sha256(self) -> str | None:
        return self._plan_sha256

    def step_cap(self, step: int) -> int:
        try:
            return self._step_caps[step]
        except KeyError as error:
            raise FixedWorkReplayError(f"fixed-work trace has no step {step}") from error

    def target_lengths_for_step(self, step: int) -> Mapping[int, int]:
        try:
            return self._target_lengths_by_step[step]
        except KeyError as error:
            raise FixedWorkReplayError(f"fixed-work trace has no step {step}") from error

    def source_lengths_for_step(self, step: int) -> Mapping[int, int]:
        try:
            return self._source_lengths_by_step[step]
        except KeyError as error:
            raise FixedWorkReplayError(f"fixed-work trace has no step {step}") from error

    def targets_for_step(self, step: int) -> Mapping[RequestIdentity, int]:
        identities = self._identities_for_step(step)
        targets = self.target_lengths_for_step(step)
        by_identity: dict[RequestIdentity, int] = {}
        for row_ordinal, identity in enumerate(identities):
            if identity in by_identity:
                raise FixedWorkReplayError(
                    f"ambiguous request identity within step {step}: {identity}"
                )
            by_identity[identity] = targets[row_ordinal]
        return MappingProxyType(by_identity)

    def _identities_for_step(self, step: int) -> tuple[RequestIdentity, ...]:
        try:
            return self._identities_by_ordinal[step]
        except KeyError as error:
            raise FixedWorkReplayError(f"fixed-work trace has no step {step}") from error

    def identity_for_row(self, step: int, row_ordinal: int) -> RequestIdentity:
        if isinstance(row_ordinal, bool) or not isinstance(row_ordinal, int):
            raise FixedWorkReplayError("row_ordinal must be an integer")
        identities = self._identities_for_step(step)
        if not 0 <= row_ordinal < len(identities):
            raise FixedWorkReplayError(
                f"fixed-work trace has no row_ordinal={row_ordinal} in step {step}"
            )
        return identities[row_ordinal]

    def occurrence_for_row(self, step: int, row_ordinal: int) -> int:
        if isinstance(row_ordinal, bool) or not isinstance(row_ordinal, int):
            raise FixedWorkReplayError("row_ordinal must be an integer")
        try:
            occurrences = self._occurrences_by_ordinal[step]
        except KeyError as error:
            raise FixedWorkReplayError(
                f"fixed-work trace has no step {step}"
            ) from error
        if not 0 <= row_ordinal < len(occurrences):
            raise FixedWorkReplayError(
                f"fixed-work trace has no row_ordinal={row_ordinal} in step {step}"
            )
        return occurrences[row_ordinal]

    def prompt_occurrences_for_step(self, step: int) -> tuple[int, ...]:
        try:
            return self._prompt_occurrences_by_step[step]
        except KeyError as error:
            raise FixedWorkReplayError(
                f"fixed-work trace has no step {step}"
            ) from error

    def _record_for_occurrence(
        self,
        prompt_occurrence_ordinal: int,
        rollout_sample_index: int,
        rollout_prompt_hash: str,
        rollout_request_seed: int,
    ) -> tuple[int, int, RequestIdentity, int, int]:
        for value, label in (
            (prompt_occurrence_ordinal, "prompt_occurrence_ordinal"),
            (rollout_sample_index, "rollout_sample_index"),
            (rollout_request_seed, "rollout_request_seed"),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise FixedWorkReplayError(f"{label} must be a nonnegative integer")
        if not isinstance(rollout_prompt_hash, str) or not rollout_prompt_hash:
            raise FixedWorkReplayError(
                "rollout_prompt_hash must be a nonempty string"
            )
        key = (prompt_occurrence_ordinal, rollout_sample_index)
        try:
            record = self._records_by_occurrence[key]
        except KeyError as error:
            raise FixedWorkReplayError(
                "fixed-work trace has no stable request for "
                "prompt_occurrence_ordinal="
                f"{prompt_occurrence_ordinal} sample={rollout_sample_index}"
            ) from error
        expected_identity = record[2]
        observed_identity = (
            rollout_prompt_hash,
            rollout_sample_index,
            rollout_request_seed,
        )
        if observed_identity != expected_identity:
            raise FixedWorkReplayError(
                "fixed-work audit identity mismatch for stable request "
                "prompt_occurrence_ordinal="
                f"{prompt_occurrence_ordinal} sample={rollout_sample_index}: "
                f"expected {expected_identity}, got {observed_identity}"
            )
        return record

    def source_row_for_occurrence(
        self,
        prompt_occurrence_ordinal: int,
        rollout_sample_index: int,
        rollout_prompt_hash: str,
        rollout_request_seed: int,
    ) -> int:
        return self._record_for_occurrence(
            prompt_occurrence_ordinal,
            rollout_sample_index,
            rollout_prompt_hash,
            rollout_request_seed,
        )[1]

    def source_step_for_occurrence(
        self,
        prompt_occurrence_ordinal: int,
        rollout_sample_index: int,
        rollout_prompt_hash: str,
        rollout_request_seed: int,
    ) -> int:
        return self._record_for_occurrence(
            prompt_occurrence_ordinal,
            rollout_sample_index,
            rollout_prompt_hash,
            rollout_request_seed,
        )[0]

    def target_for_occurrence(
        self,
        prompt_occurrence_ordinal: int,
        rollout_sample_index: int,
        rollout_prompt_hash: str,
        rollout_request_seed: int,
    ) -> int:
        return self._record_for_occurrence(
            prompt_occurrence_ordinal,
            rollout_sample_index,
            rollout_prompt_hash,
            rollout_request_seed,
        )[4]

    def source_length_for_occurrence(
        self,
        prompt_occurrence_ordinal: int,
        rollout_sample_index: int,
        rollout_prompt_hash: str,
        rollout_request_seed: int,
    ) -> int:
        return self._record_for_occurrence(
            prompt_occurrence_ordinal,
            rollout_sample_index,
            rollout_prompt_hash,
            rollout_request_seed,
        )[3]

    def _unique_ordinal_for_identity(
        self,
        step: int,
        identity: RequestIdentity,
    ) -> int:
        try:
            identity_index = self._identity_ordinals_by_step[step]
        except KeyError as error:
            raise FixedWorkReplayError(f"fixed-work trace has no step {step}") from error
        try:
            ordinals = identity_index[identity]
        except (KeyError, TypeError) as error:
            raise FixedWorkReplayError(
                f"fixed-work trace has no request for "
                f"step={step} identity={identity}"
            ) from error
        if len(ordinals) != 1:
            raise FixedWorkReplayError(
                f"ambiguous request identity within step {step}: {identity} "
                f"appears at row_ordinals={ordinals}"
            )
        return ordinals[0]

    def target_for_identity(
        self,
        step: int,
        identity: RequestIdentity,
    ) -> int:
        row_ordinal = self._unique_ordinal_for_identity(step, identity)
        return self.target_for_request(step, row_ordinal, identity)

    def source_length_for_identity(
        self,
        step: int,
        identity: RequestIdentity,
    ) -> int:
        row_ordinal = self._unique_ordinal_for_identity(step, identity)
        return self.source_length_for_request(step, row_ordinal, identity)

    def target_for_request(
        self,
        step: int,
        row_ordinal: int,
        identity: RequestIdentity,
    ) -> int:
        expected_identity = self.identity_for_row(step, row_ordinal)
        if identity != expected_identity:
            raise FixedWorkReplayError(
                f"fixed-work identity mismatch at step={step} "
                f"row_ordinal={row_ordinal}: expected {expected_identity}, got {identity}"
            )
        return self.target_lengths_for_step(step)[row_ordinal]

    def source_length_for_request(
        self,
        step: int,
        row_ordinal: int,
        identity: RequestIdentity,
    ) -> int:
        expected_identity = self.identity_for_row(step, row_ordinal)
        if identity != expected_identity:
            raise FixedWorkReplayError(
                f"fixed-work identity mismatch at step={step} "
                f"row_ordinal={row_ordinal}: expected {expected_identity}, got {identity}"
            )
        return self.source_lengths_for_step(step)[row_ordinal]

    def decoded_target(
        self,
        step: int,
        rollout_prompt_hash: str,
        rollout_sample_index: int,
        rollout_request_seed: int,
    ) -> int:
        return self.target_for_identity(
            step,
            (
                rollout_prompt_hash,
                rollout_sample_index,
                rollout_request_seed,
            ),
        )


_CACHE_LOCK = threading.Lock()
_CACHE: dict[tuple[Path, str], FixedWorkReplay] = {}


def clear_fixed_work_replay_cache() -> None:
    with _CACHE_LOCK:
        _CACHE.clear()


def _require_int(value: Any, field: str, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise FixedWorkReplayError(f"{context}: {field} must be an integer")
    return value


def _require_sha256(value: Any, source: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise FixedWorkReplayError(f"missing SHA256 in {source}")
    normalized = value.strip().lower()
    if len(normalized) != 64 or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise FixedWorkReplayError(f"invalid SHA256 in {source}")
    return normalized


def _read_stable_bytes(path: Path) -> bytes:
    try:
        before = path.stat()
        raw = path.read_bytes()
        after = path.stat()
    except OSError as error:
        raise FixedWorkReplayError(f"cannot read fixed-work source {path}: {error}") from error
    before_identity = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    after_identity = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    )
    if before_identity != after_identity or len(raw) != after.st_size:
        raise FixedWorkReplayError(f"fixed-work source changed while reading: {path}")
    return raw


def _parse_step_caps(value: Any) -> dict[int, int]:
    if not isinstance(value, dict) or not value:
        raise FixedWorkReplayError("fixed-work manifest step_caps must be nonempty")
    caps: dict[int, int] = {}
    for raw_step, raw_cap in value.items():
        if not isinstance(raw_step, str) or not raw_step.isdecimal():
            raise FixedWorkReplayError("step_caps keys must be canonical step strings")
        step = int(raw_step)
        if str(step) != raw_step or step < 1:
            raise FixedWorkReplayError("step_caps keys must be canonical step strings")
        cap = _require_int(raw_cap, "response cap", f"step_caps[{raw_step!r}]")
        if not 1 <= cap <= MAX_RESPONSE_LENGTH:
            raise FixedWorkReplayError(
                f"step_caps[{raw_step!r}]={cap} is outside [1, {MAX_RESPONSE_LENGTH}]"
            )
        caps[step] = cap
    if sorted(caps) != list(range(1, max(caps) + 1)):
        raise FixedWorkReplayError("step_caps are not contiguous from 1")
    return caps


def _parse_plan_contract(
    raw: bytes, path: Path
) -> tuple[dict[int, int], dict[int, tuple[int, ...]]]:
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise FixedWorkReplayError(f"invalid AdaFloor plan JSON in {path}: {error}") from error
    if not isinstance(payload, list) or not payload:
        raise FixedWorkReplayError("AdaFloor plan must be a nonempty JSON list")
    caps: dict[int, int] = {}
    occurrences_by_step: dict[int, tuple[int, ...]] = {}
    for index, item in enumerate(payload):
        context = f"AdaFloor plan item {index}"
        if not isinstance(item, dict):
            raise FixedWorkReplayError(f"{context} must be an object")
        step = _require_int(item.get("step"), "step", context)
        cap = _require_int(
            item.get("tail_guard_response_cap"),
            "tail_guard_response_cap",
            context,
        )
        if step < 1 or not 1 <= cap <= MAX_RESPONSE_LENGTH:
            raise FixedWorkReplayError(f"{context} has invalid step or response cap")
        if step in caps:
            raise FixedWorkReplayError(f"AdaFloor plan contains duplicate step {step}")
        caps[step] = cap
        rank_map = item.get("rank_to_source_idx")
        expected_rank_keys = {str(rank) for rank in range(WORLD_SIZE)}
        if not isinstance(rank_map, dict) or set(rank_map) != expected_rank_keys:
            raise FixedWorkReplayError(
                f"{context} rank_to_source_idx must cover ranks "
                f"0..{WORLD_SIZE - 1}"
            )
        occurrences: list[int] = []
        for rank in range(WORLD_SIZE):
            values = rank_map[str(rank)]
            if not isinstance(values, list) or len(values) != PROMPTS_PER_RANK:
                raise FixedWorkReplayError(
                    f"{context} rank {rank} must contain {PROMPTS_PER_RANK} "
                    "source indices"
                )
            for value in values:
                occurrence = _require_int(
                    value, "source index", f"{context} rank {rank}"
                )
                if occurrence < 0:
                    raise FixedWorkReplayError(
                        f"{context} source indices must be nonnegative"
                    )
                occurrences.append(occurrence)
        if len(set(occurrences)) != PROMPTS_PER_STEP:
            raise FixedWorkReplayError(
                f"{context} does not identify {PROMPTS_PER_STEP} distinct "
                "prompt occurrences"
            )
        occurrences_by_step[step] = tuple(occurrences)
    if sorted(caps) != list(range(1, max(caps) + 1)):
        raise FixedWorkReplayError("AdaFloor plan steps are not contiguous from 1")
    all_occurrences = [
        occurrence
        for step in sorted(occurrences_by_step)
        for occurrence in occurrences_by_step[step]
    ]
    if len(set(all_occurrences)) != len(all_occurrences):
        raise FixedWorkReplayError(
            "AdaFloor plan reuses a prompt occurrence across rollout steps"
        )
    return caps, occurrences_by_step


def _validate_plan_source(
    metadata: Any,
    step_caps: Mapping[int, int],
    step_prompt_occurrences: Mapping[int, tuple[int, ...]],
) -> tuple[Path, str]:
    if metadata is None:
        raise FixedWorkReplayError(
            "fixed-work trace has no AdaFloor plan provenance"
        )
    if not isinstance(metadata, dict):
        raise FixedWorkReplayError("adafloor_plan_source must be an object or null")
    raw_path = metadata.get("path")
    if not isinstance(raw_path, str) or not raw_path:
        raise FixedWorkReplayError("adafloor_plan_source.path must be nonempty")
    path = Path(raw_path).expanduser().resolve()
    expected_sha256 = _require_sha256(
        metadata.get("sha256"), "adafloor_plan_source.sha256"
    )
    raw = _read_stable_bytes(path)
    actual_sha256 = hashlib.sha256(raw).hexdigest()
    if actual_sha256 != expected_sha256:
        raise FixedWorkReplayError(
            f"AdaFloor plan SHA256 mismatch for {path}: "
            f"expected {expected_sha256}, found {actual_sha256}"
        )
    plan_caps, plan_occurrences = _parse_plan_contract(raw, path)
    if plan_caps != dict(step_caps):
        raise FixedWorkReplayError(
            "AdaFloor plan caps do not match fixed-work manifest step_caps"
        )
    if plan_occurrences != dict(step_prompt_occurrences):
        raise FixedWorkReplayError(
            "AdaFloor plan prompt occurrences do not match the fixed-work trace"
        )
    return path, expected_sha256


def _validate_header(payload: dict[str, Any]) -> None:
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise FixedWorkReplayError(
            f"unsupported fixed-work schema_version={payload.get('schema_version')!r}"
        )
    if payload.get("format") != TRACE_FORMAT:
        raise FixedWorkReplayError(
            f"unexpected fixed-work format={payload.get('format')!r}"
        )
    source_run_dir = payload.get("source_run_dir")
    if not isinstance(source_run_dir, str) or not source_run_dir:
        raise FixedWorkReplayError(
            "fixed-work manifest source_run_dir must be a nonempty string"
        )
    expected_fields = {
        "row_ordinal_base": 0,
        "expected_rows_per_step": EXPECTED_ROWS_PER_STEP,
        "max_response_length": MAX_RESPONSE_LENGTH,
        "identity_fields": list(IDENTITY_FIELDS),
        "lookup_key_fields": [
            "prompt_occurrence_ordinal",
            "rollout_sample_index",
        ],
        "row_ordinal_semantics": "source_jsonl_physical_provenance_only",
        "prompt_occurrence_ordinal_source": (
            "adafloor_plan_source.rank_to_source_idx"
        ),
        "source_length_field": SOURCE_LENGTH_FIELD,
        "target_length_field": TARGET_LENGTH_FIELD,
    }
    for field, expected in expected_fields.items():
        if payload.get(field) != expected:
            raise FixedWorkReplayError(
                f"fixed-work manifest {field}={payload.get(field)!r}, "
                f"expected {expected!r}"
            )


def _validate_source_files(
    payload: dict[str, Any],
    aggregates_by_step: Mapping[int, tuple[int, int]],
) -> None:
    source_files = payload.get("source_files")
    if not isinstance(source_files, list):
        raise FixedWorkReplayError("fixed-work manifest source_files must be a list")
    source_steps: set[int] = set()
    for index, source in enumerate(source_files):
        context = f"source_files[{index}]"
        if not isinstance(source, dict):
            raise FixedWorkReplayError(f"{context} must be an object")
        step = _require_int(source.get("step"), "step", context)
        if step in source_steps:
            raise FixedWorkReplayError(f"duplicate source metadata for step {step}")
        source_steps.add(step)
        source_path = source.get("path")
        if not isinstance(source_path, str) or not source_path:
            raise FixedWorkReplayError(f"{context}.path must be nonempty")
        _require_sha256(source.get("sha256"), f"{context}.sha256")
        if source.get("row_count") != EXPECTED_ROWS_PER_STEP:
            raise FixedWorkReplayError(
                f"{context}: row_count must be {EXPECTED_ROWS_PER_STEP}"
            )
        if source.get("prompt_occurrence_count") != PROMPTS_PER_STEP:
            raise FixedWorkReplayError(
                f"{context}: prompt_occurrence_count must be {PROMPTS_PER_STEP}"
            )
        expected_aggregates = aggregates_by_step.get(step)
        if expected_aggregates is None:
            raise FixedWorkReplayError(f"{context}: step has no matching records")
        source_tokens = _require_int(
            source.get("source_generated_tokens"),
            "source_generated_tokens",
            context,
        )
        target_tokens = _require_int(
            source.get("target_generated_tokens"),
            "target_generated_tokens",
            context,
        )
        if (source_tokens, target_tokens) != expected_aggregates:
            raise FixedWorkReplayError(
                f"{context}: generated-token aggregates do not match records"
            )
    if source_steps != set(aggregates_by_step):
        raise FixedWorkReplayError(
            "source_files steps do not match fixed-work record steps"
        )


def _parse_trace(raw: bytes, path: Path, trace_sha256: str) -> FixedWorkReplay:
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise FixedWorkReplayError(f"invalid fixed-work JSON in {path}: {error}") from error
    if not isinstance(payload, dict):
        raise FixedWorkReplayError("fixed-work manifest must be a JSON object")
    _validate_header(payload)
    step_caps = _parse_step_caps(payload.get("step_caps"))
    raw_step_occurrences = payload.get("step_prompt_occurrences")
    if not isinstance(raw_step_occurrences, dict):
        raise FixedWorkReplayError(
            "fixed-work manifest step_prompt_occurrences must be an object"
        )
    step_prompt_occurrences: dict[int, tuple[int, ...]] = {}
    for raw_step, raw_occurrences in raw_step_occurrences.items():
        if not isinstance(raw_step, str) or not raw_step.isdecimal():
            raise FixedWorkReplayError(
                "step_prompt_occurrences keys must be canonical step strings"
            )
        step = int(raw_step)
        if str(step) != raw_step or step not in step_caps:
            raise FixedWorkReplayError(
                "step_prompt_occurrences contains an invalid step"
            )
        if not isinstance(raw_occurrences, list) or len(raw_occurrences) != PROMPTS_PER_STEP:
            raise FixedWorkReplayError(
                f"step {step} must list {PROMPTS_PER_STEP} prompt occurrences"
            )
        occurrences = tuple(
            _require_int(value, "prompt occurrence", f"step {step}")
            for value in raw_occurrences
        )
        if any(value < 0 for value in occurrences) or len(set(occurrences)) != len(
            occurrences
        ):
            raise FixedWorkReplayError(
                f"step {step} prompt occurrences must be distinct nonnegative integers"
            )
        step_prompt_occurrences[step] = occurrences
    if set(step_prompt_occurrences) != set(step_caps):
        raise FixedWorkReplayError(
            "step_prompt_occurrences steps do not match step_caps"
        )
    all_prompt_occurrences = [
        value
        for step in sorted(step_prompt_occurrences)
        for value in step_prompt_occurrences[step]
    ]
    if len(set(all_prompt_occurrences)) != len(all_prompt_occurrences):
        raise FixedWorkReplayError(
            "prompt occurrences must not be reused across rollout steps"
        )
    expected_occurrence_universe = set(
        range(len(step_caps) * PROMPTS_PER_STEP)
    )
    if set(all_prompt_occurrences) != expected_occurrence_universe:
        missing = sorted(expected_occurrence_universe - set(all_prompt_occurrences))
        extra = sorted(set(all_prompt_occurrences) - expected_occurrence_universe)
        raise FixedWorkReplayError(
            "prompt occurrence universe is not the source prefix, "
            f"missing={missing[:8]} extra={extra[:8]}"
        )

    records = payload.get("records")
    if not isinstance(records, list) or not records:
        raise FixedWorkReplayError("fixed-work manifest records must be a nonempty list")
    identities_by_ordinal_map: dict[int, dict[int, RequestIdentity]] = {}
    target_lengths_by_ordinal_map: dict[int, dict[int, int]] = {}
    source_lengths_by_ordinal_map: dict[int, dict[int, int]] = {}
    occurrences_by_ordinal_map: dict[int, dict[int, int]] = {}
    stable_keys_by_step: dict[int, set[OccurrenceKey]] = {}
    aggregates_by_step: dict[int, list[int]] = {}

    for index, record in enumerate(records):
        context = f"records[{index}]"
        if not isinstance(record, dict):
            raise FixedWorkReplayError(f"{context} must be an object")
        step = _require_int(record.get("step"), "step", context)
        row_ordinal = _require_int(record.get("row_ordinal"), "row_ordinal", context)
        if step not in step_caps:
            raise FixedWorkReplayError(f"{context}: step has no response cap")
        if row_ordinal < 0:
            raise FixedWorkReplayError(f"{context}: row_ordinal must be nonnegative")
        prompt_occurrence = _require_int(
            record.get("prompt_occurrence_ordinal"),
            "prompt_occurrence_ordinal",
            context,
        )
        if prompt_occurrence < 0:
            raise FixedWorkReplayError(
                f"{context}: prompt_occurrence_ordinal must be nonnegative"
            )

        prompt_hash = record.get("rollout_prompt_hash")
        if not isinstance(prompt_hash, str) or not prompt_hash.strip():
            raise FixedWorkReplayError(
                f"{context}: rollout_prompt_hash must be a nonempty string"
            )
        sample_index = _require_int(
            record.get("rollout_sample_index"), "rollout_sample_index", context
        )
        request_seed = _require_int(
            record.get("rollout_request_seed"), "rollout_request_seed", context
        )
        if sample_index < 0 or request_seed < 0:
            raise FixedWorkReplayError(
                f"{context}: sample index and request seed must be nonnegative"
            )

        source_length = _require_int(
            record.get(SOURCE_LENGTH_FIELD), SOURCE_LENGTH_FIELD, context
        )
        target_length = _require_int(
            record.get(TARGET_LENGTH_FIELD), TARGET_LENGTH_FIELD, context
        )
        if not 1 <= source_length <= MAX_RESPONSE_LENGTH:
            raise FixedWorkReplayError(
                f"{context}: {SOURCE_LENGTH_FIELD} is outside valid range"
            )
        expected_target = min(source_length, step_caps[step])
        if target_length != expected_target:
            raise FixedWorkReplayError(
                f"{context}: target_response_length={target_length}, "
                f"expected min(source, cap)={expected_target}"
            )

        ordinal_map = identities_by_ordinal_map.setdefault(step, {})
        if row_ordinal in ordinal_map:
            raise FixedWorkReplayError(
                f"duplicate row_ordinal={row_ordinal} within step {step}"
            )
        identity = (prompt_hash, sample_index, request_seed)
        ordinal_map[row_ordinal] = identity
        stable_key = (prompt_occurrence, sample_index)
        stable_keys = stable_keys_by_step.setdefault(step, set())
        if stable_key in stable_keys:
            raise FixedWorkReplayError(
                f"{context}: duplicate stable request key {stable_key}"
            )
        stable_keys.add(stable_key)
        occurrences_by_ordinal_map.setdefault(step, {})[
            row_ordinal
        ] = prompt_occurrence
        target_lengths_by_ordinal_map.setdefault(step, {})[
            row_ordinal
        ] = target_length
        source_lengths_by_ordinal_map.setdefault(step, {})[
            row_ordinal
        ] = source_length
        aggregates = aggregates_by_step.setdefault(step, [0, 0])
        aggregates[0] += source_length
        aggregates[1] += target_length

    identities_by_ordinal: dict[int, tuple[RequestIdentity, ...]] = {}
    target_lengths_by_step: dict[int, tuple[int, ...]] = {}
    source_lengths_by_step: dict[int, tuple[int, ...]] = {}
    occurrences_by_ordinal: dict[int, tuple[int, ...]] = {}
    expected_ordinals = set(range(EXPECTED_ROWS_PER_STEP))
    for step, ordinal_map in identities_by_ordinal_map.items():
        ordinals = set(ordinal_map)
        if ordinals != expected_ordinals:
            missing = sorted(expected_ordinals - ordinals)
            extra = sorted(ordinals - expected_ordinals)
            raise FixedWorkReplayError(
                f"step {step} row_ordinal values are not contiguous 0.."
                f"{EXPECTED_ROWS_PER_STEP - 1}; missing={missing[:8]} extra={extra[:8]}"
            )
        identities_by_ordinal[step] = tuple(
            ordinal_map[ordinal] for ordinal in range(EXPECTED_ROWS_PER_STEP)
        )
        target_lengths_by_step[step] = tuple(
            target_lengths_by_ordinal_map[step][ordinal]
            for ordinal in range(EXPECTED_ROWS_PER_STEP)
        )
        source_lengths_by_step[step] = tuple(
            source_lengths_by_ordinal_map[step][ordinal]
            for ordinal in range(EXPECTED_ROWS_PER_STEP)
        )
        occurrences_by_ordinal[step] = tuple(
            occurrences_by_ordinal_map[step][ordinal]
            for ordinal in range(EXPECTED_ROWS_PER_STEP)
        )
        expected_occurrence_rows = tuple(
            occurrence
            for occurrence in step_prompt_occurrences[step]
            for _sample in range(RESPONSES_PER_PROMPT)
        )
        if occurrences_by_ordinal[step] != expected_occurrence_rows:
            raise FixedWorkReplayError(
                f"step {step} occurrence rows do not match source physical provenance"
            )
        expected_stable_keys = {
            (occurrence, sample)
            for occurrence in step_prompt_occurrences[step]
            for sample in range(RESPONSES_PER_PROMPT)
        }
        if stable_keys_by_step.get(step) != expected_stable_keys:
            raise FixedWorkReplayError(
                f"step {step} does not contain every occurrence/sample pair exactly once"
            )

    steps = set(identities_by_ordinal)
    if steps != set(step_caps):
        raise FixedWorkReplayError("fixed-work record steps do not match step_caps")
    step_count = _require_int(payload.get("step_count"), "step_count", "manifest")
    record_count = _require_int(
        payload.get("record_count"), "record_count", "manifest"
    )
    source_total = _require_int(
        payload.get("source_generated_tokens"),
        "source_generated_tokens",
        "manifest",
    )
    target_total = _require_int(
        payload.get("target_generated_tokens"),
        "target_generated_tokens",
        "manifest",
    )
    computed_source_total = sum(values[0] for values in aggregates_by_step.values())
    computed_target_total = sum(values[1] for values in aggregates_by_step.values())
    if step_count != len(steps) or record_count != len(records):
        raise FixedWorkReplayError("fixed-work manifest counts do not match records")
    if (source_total, target_total) != (
        computed_source_total,
        computed_target_total,
    ):
        raise FixedWorkReplayError(
            "fixed-work manifest generated-token totals do not match records"
        )
    immutable_aggregates = {
        step: (values[0], values[1]) for step, values in aggregates_by_step.items()
    }
    _validate_source_files(payload, immutable_aggregates)
    plan_path, plan_sha256 = _validate_plan_source(
        payload.get("adafloor_plan_source"),
        step_caps,
        step_prompt_occurrences,
    )

    return FixedWorkReplay(
        path=path,
        trace_sha256=trace_sha256,
        target_lengths_by_step=target_lengths_by_step,
        source_lengths_by_step=source_lengths_by_step,
        identities_by_ordinal=identities_by_ordinal,
        occurrences_by_ordinal=occurrences_by_ordinal,
        step_caps=step_caps,
        source_generated_tokens=source_total,
        target_generated_tokens=target_total,
        plan_path=plan_path,
        plan_sha256=plan_sha256,
    )


def _revalidate_cached_plan(trace: FixedWorkReplay) -> None:
    _validate_plan_source(
        {
            "path": str(trace.adafloor_plan_path),
            "sha256": trace.adafloor_plan_sha256,
        },
        trace._step_caps,
        trace._prompt_occurrences_by_step,
    )


def load_fixed_work_replay(
    path: str | Path,
    *,
    expected_sha256: str | None = None,
) -> FixedWorkReplay:
    resolved = Path(path).expanduser().resolve()
    expected_source = "expected_sha256 argument"
    if expected_sha256 is None:
        expected_sha256 = os.environ.get(TRACE_SHA256_ENV)
        expected_source = TRACE_SHA256_ENV
    expected = _require_sha256(expected_sha256, expected_source)

    raw = _read_stable_bytes(resolved)
    actual = hashlib.sha256(raw).hexdigest()
    if actual != expected:
        raise FixedWorkReplayError(
            f"fixed-work trace SHA256 mismatch for {resolved}: "
            f"expected {expected}, found {actual}"
        )

    cache_key = (resolved, actual)
    with _CACHE_LOCK:
        cached = _CACHE.get(cache_key)
    if cached is not None:
        _revalidate_cached_plan(cached)
        return cached

    loaded = _parse_trace(raw, resolved, actual)
    with _CACHE_LOCK:
        stale_keys = [key for key in _CACHE if key[0] == resolved and key != cache_key]
        for key in stale_keys:
            del _CACHE[key]
        return _CACHE.setdefault(cache_key, loaded)
