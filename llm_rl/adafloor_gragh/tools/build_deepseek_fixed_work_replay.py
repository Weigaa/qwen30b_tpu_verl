#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


SCHEMA_VERSION = 3
TRACE_FORMAT = "deepseek_batch64_fixed_work_replay"
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
INPUT_LENGTH_FIELD = "decoded_response_length"
SOURCE_LENGTH_FIELD = "source_decoded_response_length"
TARGET_LENGTH_FIELD = "target_response_length"


class TraceValidationError(ValueError):
    pass


def _discover_step_files(source_run_dir: Path) -> list[tuple[int, Path]]:
    rollout_data = source_run_dir / "rollout_data"
    if not rollout_data.is_dir():
        raise TraceValidationError(f"missing rollout_data directory: {rollout_data}")

    by_step: dict[int, Path] = {}
    for path in rollout_data.glob("*.jsonl"):
        if not path.stem.isdecimal():
            continue
        step = int(path.stem)
        if step < 1:
            raise TraceValidationError(f"step filenames must start at 1: {path}")
        if step in by_step:
            raise TraceValidationError(
                f"multiple rollout files encode step {step}: "
                f"{by_step[step]} and {path}"
            )
        by_step[step] = path

    if not by_step:
        raise TraceValidationError(f"no numeric step JSONL files under {rollout_data}")
    steps = sorted(by_step)
    expected = list(range(1, steps[-1] + 1))
    if steps != expected:
        missing = sorted(set(expected) - set(steps))
        raise TraceValidationError(
            f"rollout steps are not contiguous from 1; missing steps: {missing}"
        )
    return [(step, by_step[step]) for step in steps]


def _require_int(record: dict[str, Any], field: str, context: str) -> int:
    value = record.get(field)
    if isinstance(value, bool) or not isinstance(value, int):
        raise TraceValidationError(f"{context}: {field} must be an integer")
    return value


def _load_plan_contract(
    plan_path: Path,
    rollout_steps: list[int],
) -> tuple[dict[int, int], dict[int, tuple[int, ...]], dict[str, str]]:
    resolved = plan_path.expanduser().resolve()
    try:
        raw = resolved.read_bytes()
    except OSError as error:
        raise TraceValidationError(f"cannot read AdaFloor plan {resolved}: {error}") from error
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise TraceValidationError(f"invalid AdaFloor plan JSON: {error}") from error
    if not isinstance(payload, list) or not payload:
        raise TraceValidationError("AdaFloor plan must be a nonempty JSON list")

    caps: dict[int, int] = {}
    occurrences_by_step: dict[int, tuple[int, ...]] = {}
    for index, item in enumerate(payload):
        context = f"AdaFloor plan item {index}"
        if not isinstance(item, dict):
            raise TraceValidationError(f"{context} must be an object")
        step = _require_int(item, "step", context)
        cap = _require_int(item, "tail_guard_response_cap", context)
        if step < 1:
            raise TraceValidationError(f"{context}: step must be positive")
        if not 1 <= cap <= MAX_RESPONSE_LENGTH:
            raise TraceValidationError(
                f"{context}: tail_guard_response_cap={cap} is outside "
                f"[1, {MAX_RESPONSE_LENGTH}]"
            )
        if step in caps:
            raise TraceValidationError(f"AdaFloor plan contains duplicate step {step}")
        caps[step] = cap
        rank_map = item.get("rank_to_source_idx")
        expected_rank_keys = {str(rank) for rank in range(WORLD_SIZE)}
        if not isinstance(rank_map, dict) or set(rank_map) != expected_rank_keys:
            raise TraceValidationError(
                f"{context}: rank_to_source_idx must cover ranks 0..{WORLD_SIZE - 1}"
            )
        occurrences: list[int] = []
        for rank in range(WORLD_SIZE):
            source_indices = rank_map[str(rank)]
            if (
                not isinstance(source_indices, list)
                or len(source_indices) != PROMPTS_PER_RANK
            ):
                raise TraceValidationError(
                    f"{context}: rank {rank} must contain exactly "
                    f"{PROMPTS_PER_RANK} source indices"
                )
            for source_index in source_indices:
                if (
                    isinstance(source_index, bool)
                    or not isinstance(source_index, int)
                    or source_index < 0
                ):
                    raise TraceValidationError(
                        f"{context}: rank_to_source_idx contains an invalid index"
                    )
                occurrences.append(source_index)
        if len(set(occurrences)) != PROMPTS_PER_STEP:
            raise TraceValidationError(
                f"{context}: rank_to_source_idx does not identify "
                f"{PROMPTS_PER_STEP} distinct prompt occurrences"
            )
        dataset_rank_map = item.get("rank_to_dataset_item_idx")
        if not isinstance(dataset_rank_map, dict) or set(dataset_rank_map) != expected_rank_keys:
            raise TraceValidationError(
                f"{context}: rank_to_dataset_item_idx must cover ranks "
                f"0..{WORLD_SIZE - 1}"
            )
        dataset_indices = [
            value
            for rank in range(WORLD_SIZE)
            for value in dataset_rank_map[str(rank)]
        ]
        expected_dataset_indices = list(
            range((step - 1) * PROMPTS_PER_STEP, step * PROMPTS_PER_STEP)
        )
        if dataset_indices != expected_dataset_indices:
            raise TraceValidationError(
                f"{context}: rank_to_dataset_item_idx does not encode the "
                "physical prompt order consumed by the rollout"
            )
        occurrences_by_step[step] = tuple(occurrences)

    plan_steps = sorted(caps)
    if plan_steps != list(range(1, plan_steps[-1] + 1)):
        raise TraceValidationError("AdaFloor plan steps are not contiguous from 1")
    if plan_steps != rollout_steps:
        raise TraceValidationError(
            f"AdaFloor plan steps {plan_steps} do not match rollout steps {rollout_steps}"
        )
    all_occurrences = [
        occurrence
        for step in sorted(occurrences_by_step)
        for occurrence in occurrences_by_step[step]
    ]
    if len(set(all_occurrences)) != len(all_occurrences):
        raise TraceValidationError(
            "AdaFloor plan reuses a prompt occurrence across rollout steps"
        )
    return caps, occurrences_by_step, {
        "path": str(resolved),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def _read_step(
    step: int,
    path: Path,
    response_cap: int,
    prompt_occurrences: tuple[int, ...],
) -> tuple[list[dict[str, object]], dict[str, object]]:
    digest = hashlib.sha256()
    records: list[dict[str, object]] = []
    source_tokens = 0
    target_tokens = 0

    with path.open("rb") as source:
        for row_ordinal, raw_line in enumerate(source):
            digest.update(raw_line)
            context = f"step={step} row={row_ordinal} file={path}"
            if not raw_line.strip():
                raise TraceValidationError(f"{context}: blank JSONL row")
            try:
                record = json.loads(raw_line)
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise TraceValidationError(f"{context}: invalid JSON: {error}") from error
            if not isinstance(record, dict):
                raise TraceValidationError(f"{context}: JSON row must be an object")

            prompt_hash = record.get("rollout_prompt_hash")
            if not isinstance(prompt_hash, str) or not prompt_hash.strip():
                raise TraceValidationError(
                    f"{context}: rollout_prompt_hash must be a nonempty string"
                )
            sample_index = _require_int(record, "rollout_sample_index", context)
            request_seed = _require_int(record, "rollout_request_seed", context)
            if sample_index < 0 or request_seed < 0:
                raise TraceValidationError(
                    f"{context}: sample index and request seed must be nonnegative"
                )

            source_length = _require_int(record, INPUT_LENGTH_FIELD, context)
            if not 1 <= source_length <= MAX_RESPONSE_LENGTH:
                raise TraceValidationError(
                    f"{context}: {INPUT_LENGTH_FIELD}={source_length} is outside "
                    f"[1, {MAX_RESPONSE_LENGTH}]"
                )
            embedded_step = record.get("step")
            if embedded_step is not None and embedded_step != step:
                raise TraceValidationError(
                    f"{context}: embedded step={embedded_step!r} does not match filename"
                )

            prompt_slot, expected_sample_index = divmod(
                row_ordinal, RESPONSES_PER_PROMPT
            )
            if sample_index != expected_sample_index:
                raise TraceValidationError(
                    f"{context}: rollout_sample_index={sample_index}, expected "
                    f"{expected_sample_index} from source physical order"
                )
            rollout_rank = _require_int(record, "rollout_rank", context)
            expected_rank = prompt_slot // PROMPTS_PER_RANK
            if rollout_rank != expected_rank:
                raise TraceValidationError(
                    f"{context}: rollout_rank={rollout_rank}, expected "
                    f"{expected_rank} from the actual plan"
                )
            prompt_occurrence_ordinal = prompt_occurrences[prompt_slot]
            embedded_occurrence = record.get("prompt_occurrence_ordinal")
            if (
                embedded_occurrence is not None
                and embedded_occurrence != prompt_occurrence_ordinal
            ):
                raise TraceValidationError(
                    f"{context}: embedded prompt_occurrence_ordinal="
                    f"{embedded_occurrence!r} differs from actual plan source "
                    f"index {prompt_occurrence_ordinal}"
                )

            target_length = min(source_length, response_cap)
            source_tokens += source_length
            target_tokens += target_length
            records.append(
                {
                    "step": step,
                    "row_ordinal": row_ordinal,
                    "prompt_occurrence_ordinal": prompt_occurrence_ordinal,
                    "rollout_prompt_hash": prompt_hash,
                    "rollout_sample_index": sample_index,
                    "rollout_request_seed": request_seed,
                    SOURCE_LENGTH_FIELD: source_length,
                    TARGET_LENGTH_FIELD: target_length,
                }
            )

    if len(records) != EXPECTED_ROWS_PER_STEP:
        raise TraceValidationError(
            f"step={step} file={path}: expected exactly "
            f"{EXPECTED_ROWS_PER_STEP} rows, found {len(records)}"
        )
    source_summary: dict[str, object] = {
        "step": step,
        "path": str(path),
        "sha256": digest.hexdigest(),
        "row_count": len(records),
        "source_generated_tokens": source_tokens,
        "target_generated_tokens": target_tokens,
        "prompt_occurrence_count": len(prompt_occurrences),
    }
    return records, source_summary


def build_manifest(
    source_run_dir: Path,
    adafloor_plan: Path,
) -> dict[str, object]:
    source_run_dir = source_run_dir.resolve()
    step_files = _discover_step_files(source_run_dir)
    rollout_steps = [step for step, _path in step_files]
    step_caps, occurrences_by_step, plan_source = _load_plan_contract(
        adafloor_plan, rollout_steps
    )
    all_records: list[dict[str, object]] = []
    source_files: list[dict[str, object]] = []

    for step, path in step_files:
        records, source_summary = _read_step(
            step,
            path,
            step_caps[step],
            occurrences_by_step[step],
        )
        source_summary["path"] = str(path.relative_to(source_run_dir))
        all_records.extend(records)
        source_files.append(source_summary)

    source_tokens = sum(
        int(source["source_generated_tokens"]) for source in source_files
    )
    target_tokens = sum(
        int(source["target_generated_tokens"]) for source in source_files
    )
    occurrence_universe = {
        occurrence
        for occurrences in occurrences_by_step.values()
        for occurrence in occurrences
    }
    expected_universe = set(range(len(rollout_steps) * PROMPTS_PER_STEP))
    if occurrence_universe != expected_universe:
        missing = sorted(expected_universe - occurrence_universe)
        extra = sorted(occurrence_universe - expected_universe)
        raise TraceValidationError(
            "actual plan prompt occurrence universe is not the source prefix, "
            f"missing={missing[:8]} extra={extra[:8]}"
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "format": TRACE_FORMAT,
        "source_run_dir": str(source_run_dir),
        "adafloor_plan_source": plan_source,
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
        "step_count": len(source_files),
        "record_count": len(all_records),
        "source_generated_tokens": source_tokens,
        "target_generated_tokens": target_tokens,
        "step_caps": {str(step): step_caps[step] for step in rollout_steps},
        "step_prompt_occurrences": {
            str(step): list(occurrences_by_step[step]) for step in rollout_steps
        },
        "source_files": source_files,
        "records": all_records,
    }


def write_manifest(payload: dict[str, object], output: Path, force: bool) -> None:
    if output.exists() and not force:
        raise TraceValidationError(f"refusing to overwrite existing output: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build a fail-closed fixed-work replay trace from DeepSeek batch64 "
            "rollout JSONL files."
        )
    )
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument(
        "--source-run-dir",
        dest="source_run_dir",
        type=Path,
        help="Source run directory containing rollout_data/<step>.jsonl",
    )
    source_group.add_argument(
        "--vanilla-dir",
        dest="legacy_source_run_dir",
        type=Path,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--adafloor-plan",
        type=Path,
        required=True,
        help=(
            "Actual list-form Natural plan providing TailGuard caps and the "
            "rank_to_source_idx prompt occurrence mapping"
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    try:
        source_run_dir = args.source_run_dir or args.legacy_source_run_dir
        payload = build_manifest(source_run_dir, args.adafloor_plan)
        write_manifest(payload, args.output, args.force)
    except (OSError, TraceValidationError) as error:
        parser.error(str(error))

    print(
        f"trace={args.output} steps={payload['step_count']} "
        f"records={payload['record_count']} "
        f"source_tokens={payload['source_generated_tokens']} "
        f"target_tokens={payload['target_generated_tokens']}"
    )


if __name__ == "__main__":
    main()
