from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import tools.verify_deepseek_fixed_work_pair as verifier
from tools.verify_deepseek_batch64_pair import VerificationError
from tools.verify_deepseek_fixed_work_pair import _validate_arm_rows
from verl.utils.fixed_work_replay import FixedWorkReplay, load_fixed_work_replay


RESPONSES_PER_STEP = 1024
PROMPTS_PER_STEP = 64
SAMPLES_PER_PROMPT = 16
WORLD_SIZE = 16


def _identity(occurrence: int, sample: int, duplicate_prompt: bool) -> tuple[str, int, int]:
    prompt_hash = (
        "duplicate-prompt"
        if duplicate_prompt and occurrence in (0, 1)
        else f"prompt-{occurrence}"
    )
    seed_prompt = 0 if duplicate_prompt and occurrence in (0, 1) else occurrence
    return prompt_hash, sample, 10_000 + seed_prompt * SAMPLES_PER_PROMPT + sample


def _write_trace(
    tmp_path: Path,
    *,
    steps: int = 1,
    duplicate_prompt: bool = False,
) -> tuple[Path, FixedWorkReplay]:
    plan: list[dict[str, object]] = []
    records: list[dict[str, object]] = []
    source_files: list[dict[str, object]] = []
    step_occurrences: dict[str, list[int]] = {}
    source_total = 0
    for step in range(1, steps + 1):
        occurrences = list(
            range((step - 1) * PROMPTS_PER_STEP, step * PROMPTS_PER_STEP)
        )
        step_occurrences[str(step)] = occurrences
        plan.append(
            {
                "step": step,
                "tail_guard_response_cap": 16384,
                "rank_to_source_idx": {
                    str(rank): occurrences[rank * 4 : (rank + 1) * 4]
                    for rank in range(WORLD_SIZE)
                },
                "rank_to_dataset_item_idx": {
                    str(rank): occurrences[rank * 4 : (rank + 1) * 4]
                    for rank in range(WORLD_SIZE)
                },
            }
        )
        step_source = 0
        for prompt_slot, occurrence in enumerate(occurrences):
            for sample in range(SAMPLES_PER_PROMPT):
                row_ordinal = prompt_slot * SAMPLES_PER_PROMPT + sample
                identity = _identity(occurrence, sample, duplicate_prompt)
                source_length = occurrence % 3 + 1
                source_total += source_length
                step_source += source_length
                records.append(
                    {
                        "step": step,
                        "row_ordinal": row_ordinal,
                        "prompt_occurrence_ordinal": occurrence,
                        "rollout_prompt_hash": identity[0],
                        "rollout_sample_index": identity[1],
                        "rollout_request_seed": identity[2],
                        "source_decoded_response_length": source_length,
                        "target_response_length": source_length,
                    }
                )
        source_files.append(
            {
                "step": step,
                "path": f"rollout_data/{step}.jsonl",
                "sha256": "a" * 64,
                "row_count": RESPONSES_PER_STEP,
                "prompt_occurrence_count": PROMPTS_PER_STEP,
                "source_generated_tokens": step_source,
                "target_generated_tokens": step_source,
            }
        )

    plan_path = tmp_path / "length_sorted_rank_plan.json"
    plan_path.write_text(json.dumps(plan, sort_keys=True) + "\n", encoding="utf-8")
    plan_sha256 = hashlib.sha256(plan_path.read_bytes()).hexdigest()
    trace_path = tmp_path / "fixed_work_trace.json"
    payload = {
        "schema_version": 3,
        "format": "deepseek_batch64_fixed_work_replay",
        "source_run_dir": "/test/source-run",
        "adafloor_plan_source": {
            "path": str(plan_path.resolve()),
            "sha256": plan_sha256,
        },
        "row_ordinal_base": 0,
        "expected_rows_per_step": RESPONSES_PER_STEP,
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
        "prompt_occurrence_ordinal_source": "adafloor_plan_source.rank_to_source_idx",
        "source_length_field": "source_decoded_response_length",
        "target_length_field": "target_response_length",
        "step_count": steps,
        "record_count": steps * RESPONSES_PER_STEP,
        "source_generated_tokens": source_total,
        "target_generated_tokens": source_total,
        "step_caps": {str(step): 16384 for step in range(1, steps + 1)},
        "step_prompt_occurrences": step_occurrences,
        "source_files": source_files,
        "records": records,
    }
    trace_path.write_text(
        json.dumps(payload, ensure_ascii=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    digest = hashlib.sha256(trace_path.read_bytes()).hexdigest()
    return trace_path, load_fixed_work_replay(trace_path, expected_sha256=digest)


def _write_rollout_rows(
    epoch_dir: Path,
    trace: FixedWorkReplay,
    runtime_occurrences: dict[int, list[int]] | None = None,
) -> Path:
    rollout_dir = epoch_dir / "rollout_data"
    rollout_dir.mkdir(parents=True)
    last_path: Path | None = None
    for step in trace.steps:
        occurrences = (
            list(trace.prompt_occurrences_for_step(step))
            if runtime_occurrences is None
            else runtime_occurrences[step]
        )
        rows: list[str] = []
        for occurrence in occurrences:
            for sample in range(SAMPLES_PER_PROMPT):
                source_step = next(
                    candidate_step
                    for candidate_step in trace.steps
                    if occurrence
                    in trace.prompt_occurrences_for_step(candidate_step)
                )
                source_prompt_slot = trace.prompt_occurrences_for_step(
                    source_step
                ).index(occurrence)
                source_row = source_prompt_slot * SAMPLES_PER_PROMPT + sample
                identity = trace.identity_for_row(source_step, source_row)
                source = trace.source_length_for_occurrence(
                    occurrence, sample, identity[0], identity[2]
                )
                target = trace.target_for_occurrence(
                    occurrence, sample, identity[0], identity[2]
                )
                ordinal = len(rows)
                rows.append(
                    json.dumps(
                        {
                            "fixed_work_replay_row_ordinal": ordinal,
                            "prompt_occurrence_ordinal": occurrence,
                            "rollout_prompt_hash": identity[0],
                            "rollout_sample_index": identity[1],
                            "rollout_request_seed": identity[2],
                            "fixed_work_replay_source_row_ordinal": source_row,
                            "fixed_work_replay_source_step": source_step,
                            "fixed_work_replay_source_length": source,
                            "fixed_work_replay_target_length": target,
                            "fixed_work_replay_trace_sha256": trace.trace_sha256,
                            "decoded_response_length": target,
                            "response_finish_reason": "length",
                            "response_mask": [1] * target,
                        },
                        sort_keys=True,
                    )
                )
        last_path = rollout_dir / f"{step}.jsonl"
        last_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    assert last_path is not None
    return last_path


def _fixture(
    tmp_path: Path,
    *,
    steps: int = 1,
    duplicate_prompt: bool = False,
    runtime_occurrences: dict[int, list[int]] | None = None,
) -> tuple[Path, FixedWorkReplay, Path]:
    _, trace = _write_trace(
        tmp_path, steps=steps, duplicate_prompt=duplicate_prompt
    )
    epoch_dir = tmp_path / "epoch_001"
    rollout_path = _write_rollout_rows(epoch_dir, trace, runtime_occurrences)
    return epoch_dir, trace, rollout_path


def _replace_row(path: Path, ordinal: int, **updates: object) -> None:
    rows = path.read_text(encoding="utf-8").splitlines()
    row = json.loads(rows[ordinal])
    row.update(updates)
    rows[ordinal] = json.dumps(row, sort_keys=True)
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def test_validate_arm_rows_accepts_exact_work(tmp_path: Path) -> None:
    epoch_dir, trace, _ = _fixture(tmp_path)

    summary = _validate_arm_rows(epoch_dir, trace, 1, "vanilla")

    assert summary["responses"] == RESPONSES_PER_STEP
    assert summary["source_tokens"] == trace.source_generated_tokens
    assert summary["replayed_tokens"] == trace.target_generated_tokens
    assert summary["stable_request_count"] == RESPONSES_PER_STEP


def test_validate_arm_rows_disambiguates_duplicate_prompt_text_by_occurrence(
    tmp_path: Path,
) -> None:
    epoch_dir, trace, _ = _fixture(tmp_path, duplicate_prompt=True)

    summary = _validate_arm_rows(epoch_dir, trace, 1, "adafloor")

    assert summary["responses"] == RESPONSES_PER_STEP
    assert summary["stable_request_count"] == RESPONSES_PER_STEP


def test_lengthsort_accepts_cross_step_occurrence_remapping_but_adafloor_rejects(
    tmp_path: Path,
) -> None:
    source_step_1 = list(range(64))
    source_step_2 = list(range(64, 128))
    source_step_1[0], source_step_2[0] = source_step_2[0], source_step_1[0]
    epoch_dir, trace, _ = _fixture(
        tmp_path,
        steps=2,
        runtime_occurrences={1: source_step_1, 2: source_step_2},
    )

    summary = _validate_arm_rows(epoch_dir, trace, 2, "vanilla")
    assert summary["stable_request_count"] == 2 * RESPONSES_PER_STEP
    with pytest.raises(VerificationError, match="moved across its source plan step"):
        _validate_arm_rows(epoch_dir, trace, 2, "adafloor")


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"decoded_response_length": 2}, "length contract differs"),
        (
            {"response_finish_reason": "stop"},
            "did not finish at its fixed-work target",
        ),
    ],
)
def test_validate_arm_rows_rejects_length_or_finish_reason_mismatch(
    tmp_path: Path,
    updates: dict[str, object],
    message: str,
) -> None:
    epoch_dir, trace, rollout_path = _fixture(tmp_path)
    _replace_row(rollout_path, 0, **updates)

    with pytest.raises(VerificationError, match=message):
        _validate_arm_rows(epoch_dir, trace, 1, "adafloor")


def test_verify_pair_requires_fixed_adafloor_executed_plan_exact_sha(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    trace_path, trace = _write_trace(tmp_path)
    vanilla_epoch = tmp_path / "vanilla_epoch"
    adafloor_epoch = tmp_path / "adafloor_epoch"
    executed_plan = adafloor_epoch / "oracle" / "length_sorted_rank_plan.json"
    executed_plan.parent.mkdir(parents=True)
    assert trace.adafloor_plan_path is not None
    executed_plan.write_bytes(trace.adafloor_plan_path.read_bytes())
    arm_summary = {
        "responses": RESPONSES_PER_STEP,
        "source_tokens": trace.source_generated_tokens,
        "replayed_tokens": trace.target_generated_tokens,
        "stable_request_count": RESPONSES_PER_STEP,
        "stable_request_multiset_sha256": "d" * 64,
        "steps": [],
    }

    monkeypatch.setattr(
        verifier,
        "_validate_fixed_manifest",
        lambda _run, arm, _phase, _trace, _sha: (
            vanilla_epoch if arm == "vanilla" else adafloor_epoch
        ),
    )
    monkeypatch.setattr(
        verifier,
        "_validate_arm_rows",
        lambda _epoch, _trace, _steps, _arm: dict(arm_summary),
    )
    monkeypatch.setattr(
        verifier,
        "verify_pair",
        lambda *_args, **_kwargs: {
            "vanilla": {"generated_tokens": trace.target_generated_tokens},
            "adafloor": {"generated_tokens": trace.target_generated_tokens},
        },
    )
    kwargs = {
        "phase": "gate",
        "vanilla_run_dir": tmp_path / "vanilla",
        "adafloor_run_dir": tmp_path / "adafloor",
        "common_root": tmp_path / "common",
        "cap_env": tmp_path / "caps.env",
        "workload_profile_env": tmp_path / "profile.env",
        "trace_path": trace_path,
        "trace_sha256": trace.trace_sha256,
        "expected_execution_code_sha256": "e" * 64,
    }

    result = verifier.verify_fixed_pair(**kwargs)
    assert result["fixed_work"][
        "source_and_fixed_adafloor_plan_exactly_equal"
    ] is True

    changed_plan = json.loads(executed_plan.read_text(encoding="utf-8"))
    changed_plan[0]["rank_to_source_idx"]["0"][0], changed_plan[0][
        "rank_to_source_idx"
    ]["1"][0] = (
        changed_plan[0]["rank_to_source_idx"]["1"][0],
        changed_plan[0]["rank_to_source_idx"]["0"][0],
    )
    executed_plan.write_text(
        json.dumps(changed_plan, sort_keys=True) + "\n", encoding="utf-8"
    )
    with pytest.raises(VerificationError, match="executed plan differs"):
        verifier.verify_fixed_pair(**kwargs)
