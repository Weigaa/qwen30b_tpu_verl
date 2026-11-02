from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).parents[1]
TOOL = ROOT / "tools" / "build_deepseek_fixed_work_replay.py"
SPEC = importlib.util.spec_from_file_location(
    "build_deepseek_fixed_work_replay", TOOL
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _default_occurrences(step: int) -> list[int]:
    start = (step - 1) * MODULE.PROMPTS_PER_STEP
    return list(range(start, start + MODULE.PROMPTS_PER_STEP))


def _rank_map(values: list[int]) -> dict[str, list[int]]:
    return {
        str(rank): values[
            rank * MODULE.PROMPTS_PER_RANK : (rank + 1) * MODULE.PROMPTS_PER_RANK
        ]
        for rank in range(MODULE.WORLD_SIZE)
    }


def _plan_items(
    caps: list[int],
    occurrence_orders: list[list[int]] | None = None,
) -> list[dict[str, object]]:
    orders = occurrence_orders or [
        _default_occurrences(step) for step in range(1, len(caps) + 1)
    ]
    return [
        {
            "step": step,
            "tail_guard_response_cap": cap,
            "rank_to_source_idx": _rank_map(orders[step - 1]),
            "rank_to_dataset_item_idx": _rank_map(
                list(
                    range(
                        (step - 1) * MODULE.PROMPTS_PER_STEP,
                        step * MODULE.PROMPTS_PER_STEP,
                    )
                )
            ),
        }
        for step, cap in enumerate(caps, start=1)
    ]


def _write_plan(
    root: Path,
    caps: list[int],
    occurrence_orders: list[list[int]] | None = None,
) -> Path:
    path = root / "adafloor_plan.json"
    path.write_text(
        json.dumps(_plan_items(caps, occurrence_orders)) + "\n",
        encoding="utf-8",
    )
    return path


def _row(
    step: int,
    ordinal: int,
    prompt_occurrence: int,
    **overrides: object,
) -> dict[str, object]:
    prompt_slot, sample_index = divmod(ordinal, MODULE.RESPONSES_PER_PROMPT)
    record: dict[str, object] = {
        "step": step,
        "rollout_rank": prompt_slot // MODULE.PROMPTS_PER_RANK,
        "rollout_prompt_hash": f"prompt-{prompt_occurrence}",
        "rollout_sample_index": sample_index,
        "rollout_request_seed": prompt_occurrence * 1000 + sample_index,
        "decoded_response_length": ordinal + 1,
        "unrelated_large_payload": [ordinal, ordinal + 1],
    }
    record.update(overrides)
    return record


def _write_step(
    root: Path,
    step: int,
    *,
    occurrences: list[int] | None = None,
    row_count: int = MODULE.EXPECTED_ROWS_PER_STEP,
    mutations: dict[int, dict[str, object]] | None = None,
) -> Path:
    rollout_data = root / "rollout_data"
    rollout_data.mkdir(parents=True, exist_ok=True)
    prompt_occurrences = occurrences or _default_occurrences(step)
    rows = []
    for ordinal in range(row_count):
        prompt_slot = ordinal // MODULE.RESPONSES_PER_PROMPT
        prompt_occurrence = prompt_occurrences[prompt_slot]
        overrides = (mutations or {}).get(ordinal, {})
        rows.append(
            json.dumps(
                _row(step, ordinal, prompt_occurrence, **overrides),
                sort_keys=True,
            )
        )
    path = rollout_data / f"{step}.jsonl"
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return path


def _capped_total(cap: int) -> int:
    return sum(min(length, cap) for length in range(1, 1025))


def test_builds_schema_v3_manifest_with_global_occurrences(tmp_path: Path) -> None:
    first = _write_step(tmp_path, 1)
    second = _write_step(tmp_path, 2)
    plan = _write_plan(tmp_path, [100, 500])

    manifest = MODULE.build_manifest(tmp_path, plan)

    source_per_step = sum(range(1, 1025))
    assert manifest["schema_version"] == 3
    assert manifest["format"] == "deepseek_batch64_fixed_work_replay"
    assert manifest["source_run_dir"] == str(tmp_path.resolve())
    assert manifest["lookup_key_fields"] == [
        "prompt_occurrence_ordinal",
        "rollout_sample_index",
    ]
    assert manifest["row_ordinal_semantics"] == (
        "source_jsonl_physical_provenance_only"
    )
    assert manifest["prompt_occurrence_ordinal_source"] == (
        "adafloor_plan_source.rank_to_source_idx"
    )
    assert manifest["step_count"] == 2
    assert manifest["record_count"] == 2048
    assert manifest["step_caps"] == {"1": 100, "2": 500}
    assert manifest["step_prompt_occurrences"] == {
        "1": list(range(64)),
        "2": list(range(64, 128)),
    }
    assert manifest["source_generated_tokens"] == source_per_step * 2
    assert manifest["target_generated_tokens"] == (
        _capped_total(100) + _capped_total(500)
    )
    assert manifest["adafloor_plan_source"] == {
        "path": str(plan.resolve()),
        "sha256": hashlib.sha256(plan.read_bytes()).hexdigest(),
    }
    assert manifest["source_files"] == [
        {
            "step": 1,
            "path": "rollout_data/1.jsonl",
            "sha256": hashlib.sha256(first.read_bytes()).hexdigest(),
            "row_count": 1024,
            "source_generated_tokens": source_per_step,
            "target_generated_tokens": _capped_total(100),
            "prompt_occurrence_count": 64,
        },
        {
            "step": 2,
            "path": "rollout_data/2.jsonl",
            "sha256": hashlib.sha256(second.read_bytes()).hexdigest(),
            "row_count": 1024,
            "source_generated_tokens": source_per_step,
            "target_generated_tokens": _capped_total(500),
            "prompt_occurrence_count": 64,
        },
    ]
    assert manifest["records"][0] == {
        "step": 1,
        "row_ordinal": 0,
        "prompt_occurrence_ordinal": 0,
        "rollout_prompt_hash": "prompt-0",
        "rollout_sample_index": 0,
        "rollout_request_seed": 0,
        "source_decoded_response_length": 1,
        "target_response_length": 1,
    }
    assert manifest["records"][200]["prompt_occurrence_ordinal"] == 12
    assert manifest["records"][200]["target_response_length"] == 100


def test_plan_mapping_defines_occurrence_independent_of_source_step(
    tmp_path: Path,
) -> None:
    orders = [list(range(64, 128)), list(range(64))]
    _write_step(tmp_path, 1, occurrences=orders[0])
    _write_step(tmp_path, 2, occurrences=orders[1])
    plan = _write_plan(tmp_path, [100, 500], orders)

    manifest = MODULE.build_manifest(tmp_path, plan)

    assert manifest["records"][0]["step"] == 1
    assert manifest["records"][0]["prompt_occurrence_ordinal"] == 64
    assert manifest["records"][1024]["step"] == 2
    assert manifest["records"][1024]["prompt_occurrence_ordinal"] == 0


def test_duplicate_prompt_identity_is_disambiguated_by_occurrence(
    tmp_path: Path,
) -> None:
    duplicate_identity = {
        "rollout_prompt_hash": "duplicate-prompt",
        "rollout_sample_index": 0,
        "rollout_request_seed": 777,
    }
    _write_step(
        tmp_path,
        1,
        mutations={
            0: duplicate_identity,
            16: {**duplicate_identity, "decoded_response_length": 77},
        },
    )
    plan = _write_plan(tmp_path, [100])

    manifest = MODULE.build_manifest(tmp_path, plan)

    first = manifest["records"][0]
    second = manifest["records"][16]
    for field in MODULE.IDENTITY_FIELDS:
        assert second[field] == first[field]
    assert first["prompt_occurrence_ordinal"] == 0
    assert second["prompt_occurrence_ordinal"] == 1
    assert first["target_response_length"] == 1
    assert second["target_response_length"] == 77


def test_rejects_embedded_occurrence_that_disagrees_with_plan(
    tmp_path: Path,
) -> None:
    _write_step(
        tmp_path,
        1,
        mutations={32: {"prompt_occurrence_ordinal": 999}},
    )
    plan = _write_plan(tmp_path, [100])

    with pytest.raises(
        MODULE.TraceValidationError,
        match="embedded prompt_occurrence_ordinal=.*differs from actual plan",
    ):
        MODULE.build_manifest(tmp_path, plan)


def test_writes_atomically_and_refuses_overwrite(tmp_path: Path) -> None:
    _write_step(tmp_path, 1)
    plan = _write_plan(tmp_path, [100])
    payload = MODULE.build_manifest(tmp_path, plan)
    output = tmp_path / "trace" / "fixed_work.json"

    MODULE.write_manifest(payload, output, force=False)

    assert json.loads(output.read_text(encoding="utf-8")) == payload
    assert not (output.parent / f".{output.name}.tmp").exists()
    with pytest.raises(MODULE.TraceValidationError, match="refusing to overwrite"):
        MODULE.write_manifest(payload, output, force=False)


def test_rejects_step_with_wrong_row_count(tmp_path: Path) -> None:
    _write_step(tmp_path, 1, row_count=1023)
    plan = _write_plan(tmp_path, [100])

    with pytest.raises(MODULE.TraceValidationError, match="exactly 1024 rows"):
        MODULE.build_manifest(tmp_path, plan)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("rollout_prompt_hash", None, "nonempty string"),
        ("rollout_prompt_hash", "", "nonempty string"),
        ("rollout_sample_index", None, "must be an integer"),
        ("rollout_request_seed", None, "must be an integer"),
    ],
)
def test_rejects_incomplete_identity(
    tmp_path: Path, field: str, value: object, message: str
) -> None:
    _write_step(tmp_path, 1, mutations={7: {field: value}})
    plan = _write_plan(tmp_path, [100])

    with pytest.raises(MODULE.TraceValidationError, match=message):
        MODULE.build_manifest(tmp_path, plan)


@pytest.mark.parametrize("length", [0, 16385, 1.5, None])
def test_rejects_invalid_response_length(tmp_path: Path, length: object) -> None:
    _write_step(tmp_path, 1, mutations={9: {"decoded_response_length": length}})
    plan = _write_plan(tmp_path, [100])

    with pytest.raises(
        MODULE.TraceValidationError,
        match="outside|must be an integer",
    ):
        MODULE.build_manifest(tmp_path, plan)


@pytest.mark.parametrize("cap", [0, 16385, 1.5, None])
def test_rejects_invalid_plan_cap(tmp_path: Path, cap: object) -> None:
    _write_step(tmp_path, 1)
    items = _plan_items([100])
    items[0]["tail_guard_response_cap"] = cap
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps(items) + "\n", encoding="utf-8")

    with pytest.raises(MODULE.TraceValidationError, match="outside|must be an integer"):
        MODULE.build_manifest(tmp_path, plan)


def test_rejects_plan_with_duplicate_or_mismatched_steps(tmp_path: Path) -> None:
    _write_step(tmp_path, 1)
    _write_step(tmp_path, 2)
    plan = tmp_path / "plan.json"
    items = _plan_items([100, 200])
    items[1]["step"] = 1
    plan.write_text(json.dumps(items) + "\n", encoding="utf-8")
    with pytest.raises(MODULE.TraceValidationError, match="duplicate step"):
        MODULE.build_manifest(tmp_path, plan)

    _write_plan(tmp_path, [100])
    with pytest.raises(MODULE.TraceValidationError, match="do not match rollout steps"):
        MODULE.build_manifest(tmp_path, tmp_path / "adafloor_plan.json")


def test_rejects_non_list_plan(tmp_path: Path) -> None:
    _write_step(tmp_path, 1)
    plan = tmp_path / "plan.json"
    plan.write_text("{}\n", encoding="utf-8")

    with pytest.raises(MODULE.TraceValidationError, match="nonempty JSON list"):
        MODULE.build_manifest(tmp_path, plan)


def test_rejects_non_prefix_occurrence_universe(tmp_path: Path) -> None:
    occurrences = list(range(63)) + [64]
    _write_step(tmp_path, 1, occurrences=occurrences)
    plan = _write_plan(tmp_path, [100], [occurrences])

    with pytest.raises(MODULE.TraceValidationError, match="not the source prefix"):
        MODULE.build_manifest(tmp_path, plan)
