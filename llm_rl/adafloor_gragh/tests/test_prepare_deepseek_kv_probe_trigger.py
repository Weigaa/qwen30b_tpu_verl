from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest


TOOL = (
    Path(__file__).parents[1]
    / "tools"
    / "prepare_deepseek_kv_probe_trigger.py"
)
SPEC = importlib.util.spec_from_file_location("prepare_deepseek_kv_probe_trigger", TOOL)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _source(tmp_path: Path, *, varied_maxima: bool = True) -> Path:
    root = tmp_path / "source"
    (root / "rollout_data").mkdir(parents=True)
    (root / "rollout_length").mkdir()
    history_records = []
    source_files = []
    for step in (1, 2):
        rows = []
        lengths = []
        for prompt_index in range(32):
            prompt = f"step-{step}-prompt-{prompt_index}"
            values = [5.0] * 16
            if step == 2:
                if prompt_index < 16:
                    maximum = 5.0
                elif prompt_index < 24:
                    maximum = 17.0
                elif prompt_index < 28:
                    maximum = 33.0
                else:
                    maximum = 64.0
                if not varied_maxima:
                    maximum = 64.0
                values = [maximum] * 16
            for sample, value in enumerate(values):
                rows.append(
                    json.dumps(
                        {
                            "input": prompt,
                            "step": step,
                            "sample": sample,
                            "responses": [1] * int(value),
                        }
                    )
                )
                lengths.append(str(int(value)))
            history_records.append({"input": prompt, "lengths": values})
        (root / "rollout_data" / f"{step}.jsonl").write_text(
            "\n".join(rows) + "\n", encoding="utf-8"
        )
        (root / "rollout_length" / f"length_{step}.txt").write_text(
            "\n".join(lengths) + "\n", encoding="utf-8"
        )
        source_files.append(
            {
                "rollout_data": f"rollout_data/{step}.jsonl",
                "rollout_length": f"rollout_length/length_{step}.txt",
            }
        )
    (root / "offline_planning_history.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "steps": 2,
                "responses_per_prompt": 16,
                "prompt_count": 64,
                "source_files": source_files,
                "records": history_records,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return root


def _train_file(tmp_path: Path, *, rows: int = 40) -> Path:
    path = tmp_path / "train.parquet"
    pd.DataFrame(
        {
            "prompt": [
                [{"role": "user", "content": f"train-prompt-{index}"}]
                for index in range(rows)
            ],
            "source_index": list(range(rows)),
        }
    ).to_parquet(path, index=False)
    return path


def test_builds_and_verifies_deterministic_single_step_history(
    tmp_path: Path,
) -> None:
    source = _source(tmp_path)
    train_file = _train_file(tmp_path)
    output = tmp_path / "trigger"

    manifest = MODULE.prepare(source, output, train_file=train_file)

    assert manifest["schema_version"] == 2
    assert manifest["source_step"] == 2
    assert manifest["prompt_count"] == 32
    assert manifest["responses_per_prompt"] == 16
    assert manifest["max_response"] == 64
    assert manifest["distinct_prompt_maxima"] == [5, 17, 33, 64]
    assert manifest["target_dataset"]["selected_source_indices"] == list(range(32))
    assert manifest["positive_release_profile"]["schedule_thresholds"] == [
        5.0,
        17.0,
        33.0,
    ]
    output_inputs = MODULE._read_rollout_inputs(
        output / "rollout_data" / "1.jsonl"
    )
    target_inputs, _ = MODULE._target_prompt_inputs(
        train_file,
        dataset_fraction=MODULE.DEFAULT_DATASET_FRACTION,
        tokenizer_path=None,
    )
    assert list(dict.fromkeys(output_inputs)) == target_inputs
    assert not any(value.startswith("step-2-prompt-") for value in output_inputs)
    assert (output / "rollout_length" / "length_1.txt").read_bytes() == (
        source / "rollout_length" / "length_2.txt"
    ).read_bytes()
    history = json.loads((output / "offline_planning_history.json").read_text())
    assert history["steps"] == 1
    assert history["prompt_count"] == 32
    assert len(history["records"]) == 32
    assert [record["input"] for record in history["records"]] == target_inputs
    assert history["source_files"] == [
        {
            "rollout_data": "rollout_data/1.jsonl",
            "rollout_length": "rollout_length/length_1.txt",
        }
    ]
    assert MODULE.verify(output) == manifest


def test_repeated_builds_are_byte_identical(tmp_path: Path) -> None:
    source = _source(tmp_path)
    train_file = _train_file(tmp_path)
    first = tmp_path / "trigger-first"
    second = tmp_path / "trigger-second"

    MODULE.prepare(source, first, train_file=train_file)
    MODULE.prepare(source, second, train_file=train_file)

    for relative_path in (
        "rollout_data/1.jsonl",
        "rollout_length/length_1.txt",
        "offline_planning_history.json",
        "kv_probe_trigger_manifest.json",
    ):
        assert (first / relative_path).read_bytes() == (
            second / relative_path
        ).read_bytes()


def test_builds_batch64_trigger_from_both_source_steps(tmp_path: Path) -> None:
    source = _source(tmp_path)
    train_file = _train_file(tmp_path, rows=80)
    output = tmp_path / "trigger-batch64"
    spec = MODULE.TriggerSpec(
        prompt_count=64,
        responses_per_prompt=16,
        max_response=64,
        source_steps=(1, 2),
    )

    manifest = MODULE.prepare(
        source,
        output,
        train_file=train_file,
        dataset_fraction=0.8,
        spec=spec,
    )

    assert manifest["source_step"] is None
    assert manifest["source_steps"] == [1, 2]
    assert manifest["prompt_count"] == 64
    assert manifest["prompts_per_rank"] == 4
    assert manifest["row_count"] == 1024
    assert manifest["positive_release_profile"]["schedule_thresholds"] == [
        5.0,
        5.0,
        17.0,
    ]
    assert len(manifest["positive_release_profile"]["paired_rank_loads"]) == 16
    assert sum(1 for _ in (output / "rollout_data" / "1.jsonl").open()) == 1024
    history = json.loads((output / "offline_planning_history.json").read_text())
    assert history["prompt_count"] == 64
    assert len(history["records"]) == 64
    assert MODULE.verify(output, expected_spec=spec) == manifest


def test_verify_rejects_tampered_output_hash(tmp_path: Path) -> None:
    output = tmp_path / "trigger"
    MODULE.prepare(
        _source(tmp_path),
        output,
        train_file=_train_file(tmp_path),
    )
    with (output / "rollout_length" / "length_1.txt").open(
        "a", encoding="utf-8"
    ) as handle:
        handle.write("5\n")

    with pytest.raises(ValueError, match="hash or size mismatch"):
        MODULE.verify(output)


def test_verify_rejects_changed_source_history(tmp_path: Path) -> None:
    source = _source(tmp_path)
    output = tmp_path / "trigger"
    MODULE.prepare(source, output, train_file=_train_file(tmp_path))
    with (source / "offline_planning_history.json").open(
        "a", encoding="utf-8"
    ) as handle:
        handle.write("\n")

    with pytest.raises(ValueError, match="source file hashes"):
        MODULE.verify(output)


def test_build_rejects_single_prompt_maximum(tmp_path: Path) -> None:
    source = _source(tmp_path, varied_maxima=False)

    with pytest.raises(ValueError, match="distinct per-prompt maxima"):
        MODULE.prepare(
            source,
            tmp_path / "trigger",
            train_file=_train_file(tmp_path),
        )


def test_build_refuses_to_overwrite_output(tmp_path: Path) -> None:
    source = _source(tmp_path)
    output = tmp_path / "trigger"
    output.mkdir()

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        MODULE.prepare(source, output, train_file=_train_file(tmp_path))


def test_verify_rejects_changed_planner_training_subset(tmp_path: Path) -> None:
    source = _source(tmp_path)
    train_file = _train_file(tmp_path)
    output = tmp_path / "trigger"
    MODULE.prepare(source, output, train_file=train_file)

    changed = pd.read_parquet(train_file)
    changed.at[0, "prompt"] = [
        {"role": "user", "content": "different-first-prompt"}
    ]
    changed.to_parquet(train_file, index=False)

    with pytest.raises(ValueError, match="target dataset or rendered prompts"):
        MODULE.verify(output)


def test_build_rejects_a_fraction_larger_than_one_step(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="exactly cover the planner subset"):
        MODULE.prepare(
            _source(tmp_path),
            tmp_path / "trigger",
            train_file=_train_file(tmp_path),
            dataset_fraction=1.0,
        )
