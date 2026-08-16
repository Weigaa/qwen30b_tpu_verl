#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


OFFLINE_PLANNING_HISTORY_FILENAME = "offline_planning_history.json"
_JSON_DECODER = json.JSONDecoder()


def _input_from_rollout_line(line: str) -> str:
    index = 0
    while index < len(line) and line[index].isspace():
        index += 1
    if index >= len(line) or line[index] != "{":
        raise ValueError("rollout record is not a JSON object")
    index += 1
    while index < len(line) and line[index].isspace():
        index += 1
    key, index = _JSON_DECODER.raw_decode(line, index)
    while index < len(line) and line[index].isspace():
        index += 1
    if index >= len(line) or line[index] != ":":
        raise ValueError("rollout record has no value for its first field")
    index += 1
    while index < len(line) and line[index].isspace():
        index += 1
    value, _ = _JSON_DECODER.raw_decode(line, index)
    if key != "input":
        value = json.loads(line)["input"]
    return str(value)


def _discover_rollout_data_files(
    rollout_data: Path, steps: int
) -> list[tuple[int, int, Path]]:
    exact = [
        (idx, idx, rollout_data / f"{idx}.jsonl")
        for idx in range(1, steps + 1)
    ]
    if all(path.exists() for _, _, path in exact):
        return exact
    candidates = [
        path for path in rollout_data.glob("*.jsonl") if path.stem.isdigit()
    ]
    candidates.sort(key=lambda path: int(path.stem))
    if len(candidates) < steps:
        raise FileNotFoundError(
            f"expected at least {steps} rollout files under {rollout_data}"
        )
    return [
        (logical_idx, int(path.stem), path)
        for logical_idx, path in enumerate(candidates[:steps], start=1)
    ]


def _build_history(
    baseline_dir: Path,
    steps: int,
    responses_per_prompt: int,
) -> dict[str, object]:
    records_by_prompt: dict[str, dict[str, object]] = {}
    prompt_occurrence_count = 0
    duplicate_prompt_occurrence_count = 0
    source_files: list[dict[str, str]] = []
    for logical_step, source_step, data_file in _discover_rollout_data_files(
        baseline_dir / "rollout_data", steps
    ):
        length_file = (
            baseline_dir / "rollout_length" / f"length_{source_step}.txt"
        )
        if not length_file.is_file():
            raise FileNotFoundError(f"missing response length file: {length_file}")
        lengths = [
            float(line.strip())
            for line in length_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        prompt_inputs: list[str] = []
        with data_file.open("r", encoding="utf-8") as source:
            for line in source:
                if line.strip():
                    prompt_inputs.append(_input_from_rollout_line(line))
        if len(prompt_inputs) != len(lengths):
            raise RuntimeError(
                f"step={source_step} row/length mismatch: "
                f"rows={len(prompt_inputs)} lengths={len(lengths)}"
            )

        if len(prompt_inputs) % responses_per_prompt:
            raise RuntimeError(
                f"logical_step={logical_step} source_step={source_step} "
                f"has {len(prompt_inputs)} rows, which is not divisible by "
                f"responses_per_prompt={responses_per_prompt}"
            )
        for offset in range(0, len(prompt_inputs), responses_per_prompt):
            prompt_group = prompt_inputs[offset:offset + responses_per_prompt]
            values = lengths[offset:offset + responses_per_prompt]
            prompt_input = prompt_group[0]
            if any(value != prompt_input for value in prompt_group[1:]):
                raise RuntimeError(
                    f"logical_step={logical_step} source_step={source_step} "
                    f"rows {offset}:{offset + responses_per_prompt} do not "
                    "belong to one prompt occurrence"
                )
            prompt_occurrence_count += 1
            if prompt_input in records_by_prompt:
                duplicate_prompt_occurrence_count += 1
            # A repeated dataset row is the same prediction key. Keep its most
            # recent observation because the actor may have changed between
            # steps and the final occurrence is closest to the next epoch.
            records_by_prompt[prompt_input] = {
                "input": prompt_input,
                "lengths": [float(value) for value in values],
                "latest_logical_step": logical_step,
                "latest_source_step": source_step,
            }
        source_files.append(
            {
                "rollout_data": str(data_file.relative_to(baseline_dir)),
                "rollout_length": str(length_file.relative_to(baseline_dir)),
            }
        )

    records = list(records_by_prompt.values())
    return {
        "schema_version": 1,
        "steps": steps,
        "responses_per_prompt": responses_per_prompt,
        "prompt_count": len(records),
        "prompt_occurrence_count": prompt_occurrence_count,
        "duplicate_prompt_occurrence_count": duplicate_prompt_occurrence_count,
        "duplicate_prompt_policy": "latest_occurrence",
        "source_files": source_files,
        "records": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build compact prompt and response length history for offline planning."
        )
    )
    parser.add_argument("--baseline-dir", action="append", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--responses-per-prompt", type=int, default=16)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    for baseline_dir in args.baseline_dir:
        start = time.perf_counter()
        output = baseline_dir / OFFLINE_PLANNING_HISTORY_FILENAME
        if output.exists() and not args.force:
            raise SystemExit(f"refusing to overwrite existing history: {output}")
        payload = _build_history(
            baseline_dir,
            args.steps,
            args.responses_per_prompt,
        )
        temporary = output.with_suffix(output.suffix + ".tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=True, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
        temporary.replace(output)
        print(
            f"history={output} prompts={payload['prompt_count']} "
            f"bytes={output.stat().st_size} "
            f"elapsed_seconds={time.perf_counter() - start:.6f}"
        )


if __name__ == "__main__":
    main()
