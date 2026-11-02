#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any


MANIFEST_FILENAME = "kv_probe_trigger_manifest.json"
HISTORY_FILENAME = "offline_planning_history.json"
SOURCE_STEPS = 2
SOURCE_STEP = 2
PROMPT_COUNT = 32
RESPONSES_PER_PROMPT = 16
ROW_COUNT = PROMPT_COUNT * RESPONSES_PER_PROMPT
MAX_RESPONSE = 64
DEFAULT_DATASET_FRACTION = 0.0009


class TriggerSpec:
    __slots__ = (
        "prompt_count",
        "responses_per_prompt",
        "max_response",
        "source_steps",
    )

    def __init__(
        self,
        prompt_count: int = PROMPT_COUNT,
        responses_per_prompt: int = RESPONSES_PER_PROMPT,
        max_response: int = MAX_RESPONSE,
        source_steps: tuple[int, ...] = (SOURCE_STEP,),
    ) -> None:
        self.prompt_count = int(prompt_count)
        self.responses_per_prompt = int(responses_per_prompt)
        self.max_response = int(max_response)
        self.source_steps = tuple(int(step) for step in source_steps)
        if self.prompt_count <= 0 or self.prompt_count % 16:
            raise ValueError("prompt_count must be a positive multiple of 16")
        if self.responses_per_prompt <= 0 or self.max_response <= 0:
            raise ValueError("responses_per_prompt and max_response must be positive")
        if not self.source_steps or any(step <= 0 for step in self.source_steps):
            raise ValueError("source_steps must contain positive step indices")
        if len(set(self.source_steps)) != len(self.source_steps):
            raise ValueError("source_steps must not contain duplicates")

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, TriggerSpec):
            return NotImplemented
        return (
            self.prompt_count,
            self.responses_per_prompt,
            self.max_response,
            self.source_steps,
        ) == (
            other.prompt_count,
            other.responses_per_prompt,
            other.max_response,
            other.source_steps,
        )

    def __repr__(self) -> str:
        return (
            "TriggerSpec("
            f"prompt_count={self.prompt_count}, "
            f"responses_per_prompt={self.responses_per_prompt}, "
            f"max_response={self.max_response}, "
            f"source_steps={self.source_steps!r})"
        )

    @property
    def row_count(self) -> int:
        return self.prompt_count * self.responses_per_prompt

    @property
    def prompts_per_rank(self) -> int:
        return self.prompt_count // 16


DEFAULT_SPEC = TriggerSpec()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_record(path: Path) -> dict[str, int | str]:
    return {"sha256": _sha256(path), "bytes": path.stat().st_size}


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read JSON from {path}: {exc}") from exc


def _prompt_to_input(prompt: Any, tokenizer: Any | None = None) -> str:
    if hasattr(prompt, "tolist"):
        prompt = prompt.tolist()
    if tokenizer is not None:
        token_ids = tokenizer.apply_chat_template(
            prompt,
            add_generation_prompt=True,
            tokenize=True,
        )
        return str(tokenizer.decode(token_ids, skip_special_tokens=True))
    if (
        isinstance(prompt, list)
        and len(prompt) == 1
        and isinstance(prompt[0], dict)
    ):
        return f"user\n{prompt[0].get('content', '')}\nassistant\n"
    return str(prompt)


def _target_prompt_inputs(
    train_file: Path,
    *,
    dataset_fraction: float,
    tokenizer_path: Path | None,
    spec: TriggerSpec = DEFAULT_SPEC,
) -> tuple[list[str], dict[str, Any]]:
    train_file = train_file.resolve()
    if not train_file.is_file():
        raise FileNotFoundError(f"missing planner training file: {train_file}")
    if not 0.0 < dataset_fraction <= 1.0:
        raise ValueError(
            f"dataset_fraction must be in (0, 1], got {dataset_fraction}"
        )

    import pandas as pd

    frame = pd.read_parquet(train_file, columns=["prompt"])
    planner_sample_count = max(
        int(len(frame) * dataset_fraction),
        spec.prompt_count,
    )
    if planner_sample_count != spec.prompt_count:
        raise ValueError(
            "the single-step trigger must exactly cover the planner subset: "
            f"dataset_rows={len(frame)} fraction={dataset_fraction} "
            f"planner_rows={planner_sample_count} expected={spec.prompt_count}"
        )

    tokenizer = None
    resolved_tokenizer_path: str | None = None
    if tokenizer_path is not None:
        tokenizer_path = tokenizer_path.resolve()
        if not tokenizer_path.exists():
            raise FileNotFoundError(
                f"missing planner tokenizer path: {tokenizer_path}"
            )
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            trust_remote_code=True,
            local_files_only=True,
        )
        resolved_tokenizer_path = str(tokenizer_path)

    inputs = [
        _prompt_to_input(frame.iloc[index]["prompt"], tokenizer=tokenizer)
        for index in range(spec.prompt_count)
    ]
    if len(set(inputs)) != spec.prompt_count:
        raise ValueError(
            "planner training subset must contain "
            f"{spec.prompt_count} unique rendered prompts"
        )
    prompt_digest = hashlib.sha256(
        json.dumps(
            inputs,
            ensure_ascii=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    metadata = {
        "train_file": str(train_file),
        "train_file_record": _file_record(train_file),
        "dataset_rows": len(frame),
        "dataset_fraction": dataset_fraction,
        "selected_source_indices": list(range(spec.prompt_count)),
        "tokenizer_path": resolved_tokenizer_path,
        "rendered_prompts_sha256": prompt_digest,
    }
    return inputs, metadata


def _read_rollout_inputs(path: Path) -> list[str]:
    inputs: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"invalid JSONL record at {path}:{line_number}: {exc}"
                ) from exc
            if not isinstance(row, dict) or "input" not in row:
                raise ValueError(
                    f"rollout record at {path}:{line_number} has no input field"
                )
            inputs.append(str(row["input"]))
    return inputs


def _read_lengths(path: Path) -> list[float]:
    lengths: list[float] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            value = line.strip()
            if not value:
                continue
            try:
                length = float(value)
            except ValueError as exc:
                raise ValueError(
                    f"invalid response length at {path}:{line_number}: {value!r}"
                ) from exc
            if not math.isfinite(length) or length < 0 or not length.is_integer():
                raise ValueError(
                    f"response length at {path}:{line_number} must be a "
                    f"nonnegative integer, got {value!r}"
                )
            lengths.append(length)
    return lengths


def _group_lengths(
    inputs: list[str],
    lengths: list[float],
    *,
    context: str,
    spec: TriggerSpec = DEFAULT_SPEC,
    expected_prompt_count: int | None = None,
    require_max_response: bool = True,
    require_distinct_maxima: bool = True,
) -> dict[str, list[float]]:
    prompt_count = (
        spec.prompt_count if expected_prompt_count is None else expected_prompt_count
    )
    row_count = prompt_count * spec.responses_per_prompt
    if len(inputs) != row_count or len(lengths) != row_count:
        raise ValueError(
            f"{context} must contain {row_count} nonempty JSONL and length rows, "
            f"got rows={len(inputs)} lengths={len(lengths)}"
        )
    grouped: dict[str, list[float]] = defaultdict(list)
    for prompt_input, length in zip(inputs, lengths, strict=True):
        grouped[prompt_input].append(length)
    if len(grouped) != prompt_count:
        raise ValueError(
            f"{context} must contain {prompt_count} unique prompts, "
            f"got {len(grouped)}"
        )
    invalid = {
        prompt: len(values)
        for prompt, values in grouped.items()
        if len(values) != spec.responses_per_prompt
    }
    if invalid:
        prompt, count = next(iter(invalid.items()))
        raise ValueError(
            f"{context} prompt {prompt[:120]!r} has {count} responses, "
            f"expected {spec.responses_per_prompt}"
        )
    maxima = {max(values) for values in grouped.values()}
    if require_max_response and max(lengths) != spec.max_response:
        raise ValueError(
            f"{context} maximum response must be {spec.max_response}, "
            f"got {max(lengths)}"
        )
    if require_distinct_maxima and len(maxima) < 2:
        raise ValueError(
            f"{context} must contain at least two distinct per-prompt maxima"
        )
    return dict(grouped)


def _history_payload(
    grouped: dict[str, list[float]], spec: TriggerSpec = DEFAULT_SPEC
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "steps": 1,
        "responses_per_prompt": spec.responses_per_prompt,
        "prompt_count": spec.prompt_count,
        "source_files": [
            {
                "rollout_data": "rollout_data/1.jsonl",
                "rollout_length": "rollout_length/length_1.txt",
            }
        ],
        "records": [
            {"input": prompt_input, "lengths": values}
            for prompt_input, values in grouped.items()
        ],
    }


def _rebind_profiles(
    source_grouped: dict[str, list[float]],
    target_inputs: list[str],
) -> tuple[dict[str, list[float]], list[dict[str, str]]]:
    if len(source_grouped) != len(target_inputs):
        raise ValueError(
            "source length profile and target prompt count mismatch: "
            f"profiles={len(source_grouped)} prompts={len(target_inputs)}"
        )
    rebound: dict[str, list[float]] = {}
    bindings: list[dict[str, str]] = []
    for (source_input, values), target_input in zip(
        source_grouped.items(), target_inputs, strict=True
    ):
        rebound[target_input] = list(values)
        bindings.append(
            {
                "source_input_sha256": hashlib.sha256(
                    source_input.encode("utf-8")
                ).hexdigest(),
                "target_input_sha256": hashlib.sha256(
                    target_input.encode("utf-8")
                ).hexdigest(),
            }
        )
    return rebound, bindings


def _positive_release_profile(
    grouped: dict[str, list[float]], spec: TriggerSpec = DEFAULT_SPEC
) -> dict[str, Any]:
    prompt_maxima = sorted(float(max(values)) for values in grouped.values())
    # Adjacent equal-load grouping is the deterministic trigger assignment.
    # The 8th, 12th, and 14th completed ranks activate the 16-to-8, 8-to-4,
    # and 4-to-2 transitions for either two or four prompts per rank.
    rank_loads = [
        max(prompt_maxima[index:index + spec.prompts_per_rank])
        for index in range(0, len(prompt_maxima), spec.prompts_per_rank)
    ]
    thresholds = [rank_loads[index - 1] for index in (8, 12, 14)]
    predicted_exit = rank_loads[-1]
    if any(value >= predicted_exit for value in thresholds):
        raise ValueError(
            "source length profile does not preserve a positive release "
            "window through floor2: "
            f"thresholds={thresholds} predicted_exit={predicted_exit}"
        )
    return {
        "paired_rank_loads": rank_loads,
        "schedule_thresholds": thresholds,
        "predicted_step_exit": predicted_exit,
    }


def _write_trigger_rows(
    rollout_path: Path,
    length_path: Path,
    grouped: dict[str, list[float]],
) -> None:
    with rollout_path.open("w", encoding="utf-8") as rollout_handle, length_path.open(
        "w", encoding="utf-8"
    ) as length_handle:
        for prompt_input, values in grouped.items():
            for value in values:
                rollout_handle.write(
                    json.dumps(
                        {"input": prompt_input},
                        ensure_ascii=True,
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                    + "\n"
                )
                length_handle.write(f"{int(value)}\n")


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(
            payload,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n",
        encoding="utf-8",
    )


def _validate_history_payload(
    payload: Any,
    grouped: dict[str, list[float]],
    *,
    path: Path,
    spec: TriggerSpec = DEFAULT_SPEC,
) -> None:
    if not isinstance(payload, dict):
        raise ValueError(f"history is not a JSON object: {path}")
    expected = _history_payload(grouped, spec)
    if payload != expected:
        raise ValueError(
            f"history content does not match the selected rollout step: {path}"
        )


def _validate_source(
    source_root: Path,
    spec: TriggerSpec = DEFAULT_SPEC,
) -> tuple[dict[str, list[float]], dict[str, Any]]:
    source_root = source_root.resolve()
    history_path = source_root / HISTORY_FILENAME
    selected_paths = [
        (
            step,
            source_root / "rollout_data" / f"{step}.jsonl",
            source_root / "rollout_length" / f"length_{step}.txt",
        )
        for step in spec.source_steps
    ]
    for path in (history_path, *(path for item in selected_paths for path in item[1:])):
        if not path.is_file():
            raise FileNotFoundError(f"missing source file: {path}")

    history = _read_json(history_path)
    if not isinstance(history, dict):
        raise ValueError(f"source history is not a JSON object: {history_path}")
    expected_meta = {
        "schema_version": 1,
        "responses_per_prompt": spec.responses_per_prompt,
    }
    for key, expected in expected_meta.items():
        if history.get(key) != expected:
            raise ValueError(
                f"source history {key} mismatch: expected={expected!r} "
                f"actual={history.get(key)!r}"
            )
    history_steps = history.get("steps")
    history_prompt_count = history.get("prompt_count")
    if not isinstance(history_steps, int) or history_steps < max(spec.source_steps):
        raise ValueError("source history does not contain every requested source step")
    if not isinstance(history_prompt_count, int) or history_prompt_count < spec.prompt_count:
        raise ValueError("source history does not contain enough prompts")
    source_files = history.get("source_files")
    if not isinstance(source_files, list) or len(source_files) != history_steps:
        raise ValueError(
            f"source history must describe exactly {history_steps} source steps"
        )
    grouped: dict[str, list[float]] = {}
    source_records: dict[str, Any] = {HISTORY_FILENAME: _file_record(history_path)}
    for step, rollout_path, length_path in selected_paths:
        expected_files = {
            "rollout_data": f"rollout_data/{step}.jsonl",
            "rollout_length": f"rollout_length/length_{step}.txt",
        }
        if source_files[step - 1] != expected_files:
            raise ValueError(
                f"source history step {step} file mapping mismatch: "
                f"expected={expected_files!r} actual={source_files[step - 1]!r}"
            )
        inputs = _read_rollout_inputs(rollout_path)
        lengths = _read_lengths(length_path)
        if len(inputs) % spec.responses_per_prompt:
            raise ValueError(f"source step {step} row count is not prompt aligned")
        step_prompt_count = len(inputs) // spec.responses_per_prompt
        current = _group_lengths(
            inputs,
            lengths,
            context=f"source step {step}",
            spec=spec,
            expected_prompt_count=step_prompt_count,
            require_max_response=False,
            require_distinct_maxima=False,
        )
        duplicate = set(grouped).intersection(current)
        if duplicate:
            raise ValueError(f"duplicate prompts across selected source steps: {len(duplicate)}")
        grouped.update(current)
        source_records[expected_files["rollout_data"]] = _file_record(rollout_path)
        source_records[expected_files["rollout_length"]] = _file_record(length_path)
    if len(grouped) != spec.prompt_count:
        raise ValueError(
            f"selected source steps contain {len(grouped)} prompts, "
            f"expected {spec.prompt_count}"
        )
    combined_maxima = {max(values) for values in grouped.values()}
    if max(combined_maxima) != spec.max_response:
        raise ValueError(
            f"selected source maximum response must be {spec.max_response}, "
            f"got {max(combined_maxima)}"
        )
    if len(combined_maxima) < 2:
        raise ValueError(
            "selected source steps must contain at least two distinct "
            "per-prompt maxima"
        )

    records = history.get("records")
    if not isinstance(records, list) or len(records) != history_prompt_count:
        raise ValueError(
            f"source history must contain {history_prompt_count} records"
        )
    history_by_prompt: dict[str, list[float]] = {}
    for record in records:
        if (
            not isinstance(record, dict)
            or "input" not in record
            or "lengths" not in record
        ):
            raise ValueError(f"malformed source history record in {history_path}")
        prompt_input = str(record["input"])
        values = [float(value) for value in record["lengths"]]
        if prompt_input in history_by_prompt:
            raise ValueError(
                f"duplicate prompt in source history: {prompt_input[:120]!r}"
            )
        if len(values) != spec.responses_per_prompt:
            raise ValueError(
                f"source history prompt {prompt_input[:120]!r} has "
                f"{len(values)} lengths"
            )
        history_by_prompt[prompt_input] = values
    if set(grouped) - set(history_by_prompt):
        raise ValueError(
            "selected source step contains prompts absent from source history"
        )
    for prompt_input, values in grouped.items():
        if history_by_prompt[prompt_input] != values:
            raise ValueError(
                "source history lengths disagree with a selected step for "
                f"prompt {prompt_input[:120]!r}"
            )
    return grouped, source_records


def prepare(
    source_root: Path,
    output_root: Path,
    *,
    train_file: Path,
    dataset_fraction: float = DEFAULT_DATASET_FRACTION,
    tokenizer_path: Path | None = None,
    spec: TriggerSpec = DEFAULT_SPEC,
) -> dict[str, Any]:
    source_root = source_root.resolve()
    output_root = output_root.resolve()
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output_root}")
    grouped, source_records = _validate_source(source_root, spec)
    target_inputs, target_dataset = _target_prompt_inputs(
        train_file,
        dataset_fraction=dataset_fraction,
        tokenizer_path=tokenizer_path,
        spec=spec,
    )
    rebound, bindings = _rebind_profiles(grouped, target_inputs)
    release_profile = _positive_release_profile(rebound, spec)

    output_root.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output_root.name}.tmp-", dir=output_root.parent)
    )
    moved = False
    try:
        (temporary / "rollout_data").mkdir()
        (temporary / "rollout_length").mkdir()
        _write_trigger_rows(
            temporary / "rollout_data" / "1.jsonl",
            temporary / "rollout_length" / "length_1.txt",
            rebound,
        )
        history_path = temporary / HISTORY_FILENAME
        _write_json(history_path, _history_payload(rebound, spec))

        maxima = sorted({int(max(values)) for values in rebound.values()})
        output_files = {
            HISTORY_FILENAME: _file_record(history_path),
            "rollout_data/1.jsonl": _file_record(
                temporary / "rollout_data" / "1.jsonl"
            ),
            "rollout_length/length_1.txt": _file_record(
                temporary / "rollout_length" / "length_1.txt"
            ),
        }
        manifest = {
            "schema_version": 2,
            "artifact_type": "deepseek_kv_probe_trigger_history",
            "source_root": str(source_root),
            "source_history_sha256": source_records[HISTORY_FILENAME]["sha256"],
            "source_step": spec.source_steps[0] if len(spec.source_steps) == 1 else None,
            "source_steps": list(spec.source_steps),
            "source_files": source_records,
            "output_files": output_files,
            "row_count": spec.row_count,
            "prompt_count": spec.prompt_count,
            "responses_per_prompt": spec.responses_per_prompt,
            "prompts_per_rank": spec.prompts_per_rank,
            "max_response": spec.max_response,
            "distinct_prompt_maxima": maxima,
            "target_dataset": target_dataset,
            "prompt_profile_bindings": bindings,
            "positive_release_profile": release_profile,
        }
        _write_json(temporary / MANIFEST_FILENAME, manifest)
        temporary.rename(output_root)
        moved = True
    finally:
        if not moved and temporary.exists():
            shutil.rmtree(temporary)

    return verify(
        output_root,
        train_file=train_file,
        dataset_fraction=dataset_fraction,
        tokenizer_path=tokenizer_path,
        expected_spec=spec,
    )


def _expect_manifest_file_records(
    root: Path, records: Any, *, label: str
) -> None:
    if not isinstance(records, dict) or not records:
        raise ValueError(f"manifest {label} must be a nonempty object")
    for relative_path, expected in records.items():
        if not isinstance(relative_path, str) or not isinstance(expected, dict):
            raise ValueError(f"malformed manifest {label} entry")
        path = root / relative_path
        if not path.is_file():
            raise FileNotFoundError(f"missing {label} file: {path}")
        actual = _file_record(path)
        if actual != expected:
            raise ValueError(
                f"{label} hash or size mismatch for {path}: "
                f"expected={expected!r} actual={actual!r}"
            )


def verify(
    output_root: Path,
    *,
    train_file: Path | None = None,
    dataset_fraction: float | None = None,
    tokenizer_path: Path | None = None,
    expected_spec: TriggerSpec | None = None,
) -> dict[str, Any]:
    output_root = output_root.resolve()
    manifest_path = output_root / MANIFEST_FILENAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"missing trigger manifest: {manifest_path}")
    manifest = _read_json(manifest_path)
    if not isinstance(manifest, dict):
        raise ValueError(f"trigger manifest is not a JSON object: {manifest_path}")

    source_steps_raw = manifest.get("source_steps")
    if source_steps_raw is None:
        source_steps_raw = [manifest.get("source_step")]
    if not isinstance(source_steps_raw, list):
        raise ValueError("manifest source_steps must be a list")
    try:
        spec = TriggerSpec(
            prompt_count=int(manifest["prompt_count"]),
            responses_per_prompt=int(manifest["responses_per_prompt"]),
            max_response=int(manifest["max_response"]),
            source_steps=tuple(int(step) for step in source_steps_raw),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("manifest contains an invalid trigger specification") from exc
    if expected_spec is not None and spec != expected_spec:
        raise ValueError(
            f"trigger specification mismatch: expected={expected_spec} actual={spec}"
        )
    expected_scalars = {
        "schema_version": 2,
        "artifact_type": "deepseek_kv_probe_trigger_history",
        "row_count": spec.row_count,
        "prompt_count": spec.prompt_count,
        "responses_per_prompt": spec.responses_per_prompt,
        "max_response": spec.max_response,
    }
    if "prompts_per_rank" in manifest or spec != DEFAULT_SPEC:
        expected_scalars["prompts_per_rank"] = spec.prompts_per_rank
    for key, expected in expected_scalars.items():
        if manifest.get(key) != expected:
            raise ValueError(
                f"manifest {key} mismatch: expected={expected!r} "
                f"actual={manifest.get(key)!r}"
            )

    source_root_value = manifest.get("source_root")
    if not isinstance(source_root_value, str) or not source_root_value:
        raise ValueError("manifest source_root is missing")
    source_root = Path(source_root_value).resolve()
    grouped_source, source_records = _validate_source(source_root, spec)
    if manifest.get("source_files") != source_records:
        raise ValueError("manifest source file hashes do not match the source")
    if (
        manifest.get("source_history_sha256")
        != source_records[HISTORY_FILENAME]["sha256"]
    ):
        raise ValueError("manifest source_history_sha256 does not match the source")

    target_dataset = manifest.get("target_dataset")
    if not isinstance(target_dataset, dict):
        raise ValueError("manifest target_dataset is missing")
    recorded_train_file = target_dataset.get("train_file")
    recorded_fraction = target_dataset.get("dataset_fraction")
    recorded_tokenizer_path = target_dataset.get("tokenizer_path")
    if not isinstance(recorded_train_file, str) or not recorded_train_file:
        raise ValueError("manifest target train_file is missing")
    if train_file is None:
        train_file = Path(recorded_train_file)
    elif train_file.resolve() != Path(recorded_train_file).resolve():
        raise ValueError("requested train_file does not match trigger manifest")
    if dataset_fraction is None:
        dataset_fraction = float(recorded_fraction)
    elif not math.isclose(
        dataset_fraction,
        float(recorded_fraction),
        rel_tol=0.0,
        abs_tol=0.0,
    ):
        raise ValueError("requested dataset_fraction does not match trigger manifest")
    if tokenizer_path is None and recorded_tokenizer_path is not None:
        tokenizer_path = Path(recorded_tokenizer_path)
    elif tokenizer_path is not None:
        requested_tokenizer = str(tokenizer_path.resolve())
        if requested_tokenizer != recorded_tokenizer_path:
            raise ValueError("requested tokenizer_path does not match trigger manifest")
    target_inputs, actual_target_dataset = _target_prompt_inputs(
        train_file,
        dataset_fraction=dataset_fraction,
        tokenizer_path=tokenizer_path,
        spec=spec,
    )
    if actual_target_dataset != target_dataset:
        raise ValueError("manifest target dataset or rendered prompts have changed")
    expected_grouped, expected_bindings = _rebind_profiles(
        grouped_source,
        target_inputs,
    )
    if manifest.get("prompt_profile_bindings") != expected_bindings:
        raise ValueError("manifest prompt profile bindings do not match")

    expected_output_names = {
        HISTORY_FILENAME,
        "rollout_data/1.jsonl",
        "rollout_length/length_1.txt",
    }
    output_records = manifest.get("output_files")
    if (
        not isinstance(output_records, dict)
        or set(output_records) != expected_output_names
    ):
        raise ValueError(
            f"manifest output_files must be exactly {sorted(expected_output_names)}"
        )
    _expect_manifest_file_records(output_root, output_records, label="output")

    rollout_files = sorted((output_root / "rollout_data").glob("*.jsonl"))
    length_files = sorted((output_root / "rollout_length").glob("length_*.txt"))
    if rollout_files != [output_root / "rollout_data" / "1.jsonl"]:
        raise ValueError("trigger directory must contain only rollout_data/1.jsonl")
    if length_files != [output_root / "rollout_length" / "length_1.txt"]:
        raise ValueError(
            "trigger directory must contain only rollout_length/length_1.txt"
        )

    inputs = _read_rollout_inputs(rollout_files[0])
    lengths = _read_lengths(length_files[0])
    grouped_output = _group_lengths(
        inputs, lengths, context="trigger output", spec=spec
    )
    if grouped_output != expected_grouped:
        raise ValueError(
            "trigger output does not bind the selected source length profiles to "
            "the planner training subset"
        )
    _validate_history_payload(
        _read_json(output_root / HISTORY_FILENAME),
        grouped_output,
        path=output_root / HISTORY_FILENAME,
        spec=spec,
    )

    maxima = sorted({int(max(values)) for values in grouped_output.values()})
    if manifest.get("distinct_prompt_maxima") != maxima:
        raise ValueError("manifest distinct_prompt_maxima does not match output")
    release_profile = _positive_release_profile(grouped_output, spec)
    if manifest.get("positive_release_profile") != release_profile:
        raise ValueError("manifest positive_release_profile does not match output")
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build or verify the deterministic single-step history used to "
            "trigger DeepSeek KV floor probes."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build", help="build a single-step trigger history")
    build.add_argument("--source-root", type=Path, required=True)
    build.add_argument("--output-root", type=Path, required=True)
    build.add_argument("--train-file", type=Path, required=True)
    build.add_argument(
        "--dataset-fraction",
        type=float,
        default=DEFAULT_DATASET_FRACTION,
    )
    build.add_argument("--tokenizer-path", type=Path, default=None)
    build.add_argument("--prompt-count", type=int, default=PROMPT_COUNT)
    build.add_argument("--responses-per-prompt", type=int, default=RESPONSES_PER_PROMPT)
    build.add_argument("--max-response", type=int, default=MAX_RESPONSE)
    build.add_argument(
        "--source-steps",
        default=str(SOURCE_STEP),
        help="comma-separated one-based source step indices",
    )
    check = subparsers.add_parser("verify", help="verify an existing trigger history")
    check.add_argument("--output-root", type=Path, required=True)
    check.add_argument("--train-file", type=Path, default=None)
    check.add_argument("--dataset-fraction", type=float, default=None)
    check.add_argument("--tokenizer-path", type=Path, default=None)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.command == "build":
        spec = TriggerSpec(
            prompt_count=args.prompt_count,
            responses_per_prompt=args.responses_per_prompt,
            max_response=args.max_response,
            source_steps=tuple(
                int(value.strip())
                for value in args.source_steps.split(",")
                if value.strip()
            ),
        )
        manifest = prepare(
            args.source_root,
            args.output_root,
            train_file=args.train_file,
            dataset_fraction=args.dataset_fraction,
            tokenizer_path=args.tokenizer_path,
            spec=spec,
        )
    else:
        manifest = verify(
            args.output_root,
            train_file=args.train_file,
            dataset_fraction=args.dataset_fraction,
            tokenizer_path=args.tokenizer_path,
        )
    print(
        f"verified={args.output_root.resolve()} "
        f"prompts={manifest['prompt_count']} "
        f"responses_per_prompt={manifest['responses_per_prompt']} "
        f"max_response={manifest['max_response']}"
    )


if __name__ == "__main__":
    main()
