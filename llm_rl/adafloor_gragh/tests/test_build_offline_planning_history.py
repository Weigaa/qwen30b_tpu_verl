from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


TOOL = (
    Path(__file__).parents[1]
    / "tools"
    / "build_offline_planning_history.py"
)
SPEC = importlib.util.spec_from_file_location("build_offline_planning_history", TOOL)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _write_step(
    root: Path,
    step: int,
    occurrences: list[tuple[str, list[float]]],
) -> None:
    rollout_data = root / "rollout_data"
    rollout_length = root / "rollout_length"
    rollout_data.mkdir(parents=True, exist_ok=True)
    rollout_length.mkdir(parents=True, exist_ok=True)

    rows: list[str] = []
    lengths: list[str] = []
    for prompt, values in occurrences:
        for sample, value in enumerate(values):
            rows.append(
                json.dumps(
                    {
                        "input": prompt,
                        "sample": sample,
                        "responses": [1] * int(value),
                    }
                )
            )
            lengths.append(str(value))
    (rollout_data / f"{step}.jsonl").write_text(
        "\n".join(rows) + "\n", encoding="utf-8"
    )
    (rollout_length / f"length_{step}.txt").write_text(
        "\n".join(lengths) + "\n", encoding="utf-8"
    )


def test_repeated_prompt_keeps_latest_occurrence(tmp_path: Path) -> None:
    repeated = "repeated-prompt"
    _write_step(
        tmp_path,
        1,
        [
            (repeated, [11.0] * 16),
            ("step-1-only", [21.0] * 16),
        ],
    )
    _write_step(
        tmp_path,
        2,
        [
            ("step-2-only", [31.0] * 16),
            (repeated, [41.0 + sample for sample in range(16)]),
        ],
    )

    history = MODULE._build_history(tmp_path, steps=2, responses_per_prompt=16)

    assert history["schema_version"] == 1
    assert history["steps"] == 2
    assert history["responses_per_prompt"] == 16
    assert history["prompt_count"] == 3
    assert history["prompt_occurrence_count"] == 4
    assert history["duplicate_prompt_occurrence_count"] == 1
    assert history["duplicate_prompt_policy"] == "latest_occurrence"
    assert history["source_files"] == [
        {
            "rollout_data": "rollout_data/1.jsonl",
            "rollout_length": "rollout_length/length_1.txt",
        },
        {
            "rollout_data": "rollout_data/2.jsonl",
            "rollout_length": "rollout_length/length_2.txt",
        },
    ]

    records = {record["input"]: record for record in history["records"]}
    assert records[repeated] == {
        "input": repeated,
        "lengths": [41.0 + sample for sample in range(16)],
        "latest_logical_step": 2,
        "latest_source_step": 2,
    }
    assert records["step-1-only"]["latest_source_step"] == 1
    assert records["step-2-only"]["latest_source_step"] == 2


def test_rejects_mixed_prompt_chunk(tmp_path: Path) -> None:
    _write_step(
        tmp_path,
        1,
        [
            ("prompt-a", [1.0] * 8),
            ("prompt-b", [2.0] * 8),
        ],
    )

    with pytest.raises(RuntimeError, match="do not belong to one prompt occurrence"):
        MODULE._build_history(tmp_path, steps=1, responses_per_prompt=16)


def test_rejects_incomplete_prompt_chunk(tmp_path: Path) -> None:
    _write_step(tmp_path, 1, [("prompt-a", [1.0] * 15)])

    with pytest.raises(RuntimeError, match="is not divisible by responses_per_prompt=16"):
        MODULE._build_history(tmp_path, steps=1, responses_per_prompt=16)
