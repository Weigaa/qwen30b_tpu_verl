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
    records: list[dict[str, object]] = []
    seen_prompts: set[str] = set()
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

        grouped: dict[str, list[float]] = {}
        for prompt_input, length in zip(prompt_inputs, lengths, strict=True):
            grouped.setdefault(prompt_input, []).append(float(length))
        for prompt_input, values in grouped.items():
            if len(values) != responses_per_prompt:
                raise RuntimeError(
                    f"logical_step={logical_step} source_step={source_step} "
                    f"expected {responses_per_prompt} responses for "
                    f"{prompt_input[:120]!r}, got {len(values)}"
                )
            if prompt_input in seen_prompts:
                raise RuntimeError(
                    f"duplicate prompt across steps: {prompt_input[:120]!r}"
                )
            seen_prompts.add(prompt_input)
            records.append({"input": prompt_input, "lengths": values})
        source_files.append(
            {
                "rollout_data": str(data_file.relative_to(baseline_dir)),
                "rollout_length": str(length_file.relative_to(baseline_dir)),
            }
        )

    return {
        "schema_version": 1,
        "steps": steps,
        "responses_per_prompt": responses_per_prompt,
        "prompt_count": len(records),
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
