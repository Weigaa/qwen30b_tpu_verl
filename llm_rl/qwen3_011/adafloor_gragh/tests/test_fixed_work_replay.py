from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from verl.utils.fixed_work_replay import (
    TRACE_SHA256_ENV,
    FixedWorkReplayError,
    clear_fixed_work_replay_cache,
    load_fixed_work_replay,
)


PROMPTS_PER_STEP = 64
RESPONSES_PER_PROMPT = 16
PROMPTS_PER_RANK = 4
WORLD_SIZE = 16


def _default_occurrence_orders(step_count: int) -> list[list[int]]:
    return [
        list(range((step - 1) * PROMPTS_PER_STEP, step * PROMPTS_PER_STEP))
        for step in range(1, step_count + 1)
    ]


def _rank_map(values: list[int]) -> dict[str, list[int]]:
    return {
        str(rank): values[
            rank * PROMPTS_PER_RANK : (rank + 1) * PROMPTS_PER_RANK
        ]
        for rank in range(WORLD_SIZE)
    }


def _plan_items(
    caps: list[int], occurrence_orders: list[list[int]]
) -> list[dict[str, object]]:
    return [
        {
            "step": step,
            "tail_guard_response_cap": cap,
            "rank_to_source_idx": _rank_map(occurrence_orders[step - 1]),
            "rank_to_dataset_item_idx": _rank_map(
                list(
                    range(
                        (step - 1) * PROMPTS_PER_STEP,
                        step * PROMPTS_PER_STEP,
                    )
                )
            ),
        }
        for step, cap in enumerate(caps, start=1)
    ]


def _write_plan(
    path: Path,
    caps: list[int],
    occurrence_orders: list[list[int]] | None = None,
) -> dict[str, str]:
    orders = occurrence_orders or _default_occurrence_orders(len(caps))
    path.write_text(
        json.dumps(_plan_items(caps, orders)) + "\n",
        encoding="utf-8",
    )
    return {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def _records(
    step: int, cap: int, prompt_occurrences: list[int]
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for prompt_slot, occurrence in enumerate(prompt_occurrences):
        for sample_index in range(RESPONSES_PER_PROMPT):
            row_ordinal = prompt_slot * RESPONSES_PER_PROMPT + sample_index
            source_length = row_ordinal + 1
            records.append(
                {
                    "step": step,
                    "row_ordinal": row_ordinal,
                    "prompt_occurrence_ordinal": occurrence,
                    "rollout_prompt_hash": f"prompt-{occurrence}",
                    "rollout_sample_index": sample_index,
                    "rollout_request_seed": occurrence * 1000 + sample_index,
                    "source_decoded_response_length": source_length,
                    "target_response_length": min(source_length, cap),
                }
            )
    return records


def _payload(
    tmp_path: Path,
    caps: list[int] | None = None,
    occurrence_orders: list[list[int]] | None = None,
) -> dict[str, object]:
    resolved_caps = caps or [16384]
    orders = occurrence_orders or _default_occurrence_orders(len(resolved_caps))
    plan_source = _write_plan(tmp_path / "plan.json", resolved_caps, orders)
    records = [
        record
        for step, cap in enumerate(resolved_caps, start=1)
        for record in _records(step, cap, orders[step - 1])
    ]
    source_total_per_step = sum(range(1, 1025))
    target_totals = [
        sum(min(length, cap) for length in range(1, 1025))
        for cap in resolved_caps
    ]
    return {
        "schema_version": 3,
        "format": "deepseek_batch64_fixed_work_replay",
        "source_run_dir": "/test/source-run",
        "adafloor_plan_source": plan_source,
        "row_ordinal_base": 0,
        "expected_rows_per_step": 1024,
        "max_response_length": 16384,
        "identity_fields": [
            "rollout_prompt_hash",
            "rollout_sample_index",
            "rollout_request_seed",
        ],
        "lookup_key_fields": [
            "prompt_occurrence_ordinal",
            "rollout_sample_index",
        ],
        "row_ordinal_semantics": "source_jsonl_physical_provenance_only",
        "prompt_occurrence_ordinal_source": (
            "adafloor_plan_source.rank_to_source_idx"
        ),
        "source_length_field": "source_decoded_response_length",
        "target_length_field": "target_response_length",
        "step_count": len(resolved_caps),
        "record_count": len(records),
        "source_generated_tokens": source_total_per_step * len(resolved_caps),
        "target_generated_tokens": sum(target_totals),
        "step_caps": {
            str(step): cap for step, cap in enumerate(resolved_caps, start=1)
        },
        "step_prompt_occurrences": {
            str(step): orders[step - 1]
            for step in range(1, len(resolved_caps) + 1)
        },
        "source_files": [
            {
                "step": step,
                "path": f"rollout_data/{step}.jsonl",
                "sha256": f"{step:064x}",
                "row_count": 1024,
                "source_generated_tokens": source_total_per_step,
                "target_generated_tokens": target_totals[step - 1],
                "prompt_occurrence_count": 64,
            }
            for step in range(1, len(resolved_caps) + 1)
        ],
        "records": records,
    }


def _write(path: Path, payload: dict[str, object]) -> str:
    path.write_text(
        json.dumps(payload, ensure_ascii=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _adjust_aggregates(
    payload: dict[str, object], step: int, source_delta: int, target_delta: int
) -> None:
    payload["source_generated_tokens"] = (
        int(payload["source_generated_tokens"]) + source_delta
    )
    payload["target_generated_tokens"] = (
        int(payload["target_generated_tokens"]) + target_delta
    )
    source = payload["source_files"][step - 1]
    source["source_generated_tokens"] = (
        int(source["source_generated_tokens"]) + source_delta
    )
    source["target_generated_tokens"] = (
        int(source["target_generated_tokens"]) + target_delta
    )


@pytest.fixture(autouse=True)
def _clean_cache() -> None:
    clear_fixed_work_replay_cache()
    yield
    clear_fixed_work_replay_cache()


def test_loads_schema_v3_and_queries_global_occurrence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "trace.json"
    payload = _payload(tmp_path, [100, 500])
    digest = _write(path, payload)
    monkeypatch.setenv(TRACE_SHA256_ENV, digest.upper())

    trace = load_fixed_work_replay(path)

    assert trace.path == path.resolve()
    assert trace.trace_sha256 == digest
    assert trace.steps == (1, 2)
    assert trace.record_count == 2048
    assert trace.step_cap(1) == 100
    assert trace.source_generated_tokens == payload["source_generated_tokens"]
    assert trace.target_generated_tokens == payload["target_generated_tokens"]
    assert trace.adafloor_plan_path == (tmp_path / "plan.json").resolve()
    identity = ("prompt-12", 8, 12008)
    assert trace.identity_for_row(1, 200) == identity
    assert trace.occurrence_for_row(1, 200) == 12
    assert trace.source_row_for_occurrence(12, 8, *identity[::2]) == 200
    assert trace.source_step_for_occurrence(12, 8, *identity[::2]) == 1
    assert trace.source_length_for_occurrence(12, 8, *identity[::2]) == 201
    assert trace.target_for_occurrence(12, 8, *identity[::2]) == 100
    assert trace.source_lengths_for_step(1)[200] == 201
    assert trace.target_lengths_for_step(1)[200] == 100
    with pytest.raises(FixedWorkReplayError, match="audit identity mismatch"):
        trace.target_for_occurrence(12, 8, "wrong", 12008)
    with pytest.raises(FixedWorkReplayError, match="no row_ordinal"):
        trace.identity_for_row(1, 1024)


def test_global_lookup_is_not_keyed_by_source_step(tmp_path: Path) -> None:
    orders = [list(range(64, 128)), list(range(64))]
    path = tmp_path / "trace.json"
    payload = _payload(tmp_path, [100, 500], orders)
    digest = _write(path, payload)

    trace = load_fixed_work_replay(path, expected_sha256=digest)

    first_identity = ("prompt-76", 8, 76008)
    second_identity = ("prompt-12", 8, 12008)
    assert trace.source_step_for_occurrence(76, 8, *first_identity[::2]) == 1
    assert trace.source_row_for_occurrence(76, 8, *first_identity[::2]) == 200
    assert trace.target_for_occurrence(76, 8, *first_identity[::2]) == 100
    assert trace.source_step_for_occurrence(12, 8, *second_identity[::2]) == 2
    assert trace.source_row_for_occurrence(12, 8, *second_identity[::2]) == 200
    assert trace.target_for_occurrence(12, 8, *second_identity[::2]) == 201


def test_duplicate_prompt_identity_is_unambiguous_with_occurrence_key(
    tmp_path: Path,
) -> None:
    path = tmp_path / "trace.json"
    payload = _payload(tmp_path, [100])
    first = payload["records"][0]
    second = payload["records"][16]
    duplicate_identity = {
        "rollout_prompt_hash": "duplicate-prompt",
        "rollout_sample_index": 0,
        "rollout_request_seed": 777,
    }
    first.update(duplicate_identity)
    second.update(duplicate_identity)
    second["source_decoded_response_length"] = 77
    second["target_response_length"] = 77
    _adjust_aggregates(payload, 1, source_delta=60, target_delta=60)
    digest = _write(path, payload)

    trace = load_fixed_work_replay(path, expected_sha256=digest)

    assert trace.target_for_occurrence(0, 0, "duplicate-prompt", 777) == 1
    assert trace.target_for_occurrence(1, 0, "duplicate-prompt", 777) == 77
    assert trace.source_row_for_occurrence(0, 0, "duplicate-prompt", 777) == 0
    assert trace.source_row_for_occurrence(1, 0, "duplicate-prompt", 777) == 16
    with pytest.raises(FixedWorkReplayError, match="ambiguous"):
        trace.target_for_identity(1, ("duplicate-prompt", 0, 777))


def test_requires_plan_provenance_even_for_full_caps(tmp_path: Path) -> None:
    path = tmp_path / "trace.json"
    payload = _payload(tmp_path)
    payload["adafloor_plan_source"] = None
    digest = _write(path, payload)

    with pytest.raises(FixedWorkReplayError, match="no AdaFloor plan provenance"):
        load_fixed_work_replay(path, expected_sha256=digest)


def test_requires_expected_sha256_environment_value(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "trace.json"
    _write(path, _payload(tmp_path))
    monkeypatch.delenv(TRACE_SHA256_ENV, raising=False)

    with pytest.raises(FixedWorkReplayError, match="missing SHA256"):
        load_fixed_work_replay(path)


def test_rejects_trace_sha256_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "trace.json"
    _write(path, _payload(tmp_path))
    monkeypatch.setenv(TRACE_SHA256_ENV, "f" * 64)

    with pytest.raises(FixedWorkReplayError, match="SHA256 mismatch"):
        load_fixed_work_replay(path)


def test_cache_does_not_mask_trace_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "trace.json"
    payload = _payload(tmp_path)
    first_digest = _write(path, payload)
    monkeypatch.setenv(TRACE_SHA256_ENV, first_digest)
    first = load_fixed_work_replay(path)
    assert load_fixed_work_replay(path) is first

    record = payload["records"][0]
    record["source_decoded_response_length"] = 99
    record["target_response_length"] = 99
    _adjust_aggregates(payload, 1, source_delta=98, target_delta=98)
    second_digest = _write(path, payload)

    with pytest.raises(FixedWorkReplayError, match="SHA256 mismatch"):
        load_fixed_work_replay(path)
    monkeypatch.setenv(TRACE_SHA256_ENV, second_digest)
    second = load_fixed_work_replay(path)
    assert second is not first
    assert second.target_for_occurrence(0, 0, "prompt-0", 0) == 99


def test_cache_does_not_mask_plan_change(tmp_path: Path) -> None:
    trace_path = tmp_path / "trace.json"
    payload = _payload(tmp_path, [100])
    digest = _write(trace_path, payload)
    first = load_fixed_work_replay(trace_path, expected_sha256=digest)
    assert load_fixed_work_replay(trace_path, expected_sha256=digest) is first

    _write_plan(tmp_path / "plan.json", [101])

    with pytest.raises(FixedWorkReplayError, match="plan SHA256 mismatch"):
        load_fixed_work_replay(trace_path, expected_sha256=digest)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("schema_version", 2, "schema_version"),
        ("format", "other", "format"),
        ("source_run_dir", "", "source_run_dir"),
        ("row_ordinal_base", 1, "row_ordinal_base"),
        ("expected_rows_per_step", 64, "expected_rows_per_step"),
        ("lookup_key_fields", ["step", "row_ordinal"], "lookup_key_fields"),
        ("row_ordinal_semantics", "lookup_key", "row_ordinal_semantics"),
        ("source_length_field", "other", "source_length_field"),
        ("target_length_field", "other", "target_length_field"),
    ],
)
def test_rejects_invalid_manifest_header(
    tmp_path: Path,
    field: str,
    value: object,
    message: str,
) -> None:
    path = tmp_path / "trace.json"
    payload = _payload(tmp_path)
    payload[field] = value
    digest = _write(path, payload)

    with pytest.raises(FixedWorkReplayError, match=message):
        load_fixed_work_replay(path, expected_sha256=digest)


def test_rejects_noncontiguous_row_ordinals(tmp_path: Path) -> None:
    path = tmp_path / "trace.json"
    payload = _payload(tmp_path)
    payload["records"][-1]["row_ordinal"] = 1024
    digest = _write(path, payload)

    with pytest.raises(FixedWorkReplayError, match="not contiguous"):
        load_fixed_work_replay(path, expected_sha256=digest)


def test_rejects_target_not_equal_to_source_cap_minimum(tmp_path: Path) -> None:
    path = tmp_path / "trace.json"
    payload = _payload(tmp_path, [100])
    payload["records"][200]["target_response_length"] = 101
    digest = _write(path, payload)

    with pytest.raises(FixedWorkReplayError, match=r"min\(source, cap\)"):
        load_fixed_work_replay(path, expected_sha256=digest)


@pytest.mark.parametrize(
    ("scope", "field"),
    [
        ("manifest", "source_generated_tokens"),
        ("manifest", "target_generated_tokens"),
        ("source", "source_generated_tokens"),
        ("source", "target_generated_tokens"),
    ],
)
def test_rejects_incorrect_aggregates(
    tmp_path: Path, scope: str, field: str
) -> None:
    path = tmp_path / "trace.json"
    payload = _payload(tmp_path)
    owner = payload if scope == "manifest" else payload["source_files"][0]
    owner[field] = int(owner[field]) + 1
    digest = _write(path, payload)

    with pytest.raises(FixedWorkReplayError, match="totals|aggregates"):
        load_fixed_work_replay(path, expected_sha256=digest)


def test_rejects_plan_metadata_that_disagrees_with_caps(tmp_path: Path) -> None:
    path = tmp_path / "trace.json"
    payload = _payload(tmp_path, [100])
    payload["adafloor_plan_source"] = _write_plan(
        tmp_path / "other-plan.json", [101]
    )
    digest = _write(path, payload)

    with pytest.raises(FixedWorkReplayError, match="plan caps do not match"):
        load_fixed_work_replay(path, expected_sha256=digest)


def test_rejects_plan_metadata_that_disagrees_with_occurrences(
    tmp_path: Path,
) -> None:
    path = tmp_path / "trace.json"
    payload = _payload(tmp_path, [100])
    rotated = [list(range(1, 64)) + [0]]
    payload["adafloor_plan_source"] = _write_plan(
        tmp_path / "other-plan.json", [100], rotated
    )
    digest = _write(path, payload)

    with pytest.raises(
        FixedWorkReplayError,
        match="plan prompt occurrences do not match",
    ):
        load_fixed_work_replay(path, expected_sha256=digest)


def test_rejects_non_prefix_occurrence_universe(tmp_path: Path) -> None:
    orders = [list(range(63)) + [64]]
    path = tmp_path / "trace.json"
    payload = _payload(tmp_path, [100], orders)
    digest = _write(path, payload)

    with pytest.raises(FixedWorkReplayError, match="not the source prefix"):
        load_fixed_work_replay(path, expected_sha256=digest)


def test_rejects_duplicate_global_occurrence_sample_key(tmp_path: Path) -> None:
    path = tmp_path / "trace.json"
    payload = _payload(tmp_path)
    payload["records"][16]["prompt_occurrence_ordinal"] = 0
    digest = _write(path, payload)

    with pytest.raises(FixedWorkReplayError, match="duplicate stable request key"):
        load_fixed_work_replay(path, expected_sha256=digest)


def test_rejects_source_occurrence_rows_out_of_physical_order(tmp_path: Path) -> None:
    path = tmp_path / "trace.json"
    payload = _payload(tmp_path)
    first_block = payload["records"][:16]
    second_block = payload["records"][16:32]
    for record in first_block:
        record["prompt_occurrence_ordinal"] = 1
    for record in second_block:
        record["prompt_occurrence_ordinal"] = 0
    digest = _write(path, payload)

    with pytest.raises(
        FixedWorkReplayError,
        match="occurrence rows do not match source physical provenance",
    ):
        load_fixed_work_replay(path, expected_sha256=digest)


def test_rejects_nonstring_source_sha256(tmp_path: Path) -> None:
    path = tmp_path / "trace.json"
    payload = _payload(tmp_path)
    payload["source_files"][0]["sha256"] = 123
    digest = _write(path, payload)

    with pytest.raises(FixedWorkReplayError, match="SHA256"):
        load_fixed_work_replay(path, expected_sha256=digest)
