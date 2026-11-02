#!/usr/bin/env python3
"""Fail-closed verifier for the paired v0.14 AdaFloor execution gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any


FLOAT = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"


def _one(paths: list[Path], label: str) -> Path:
    if len(paths) != 1:
        raise RuntimeError(f"expected one {label}, found {len(paths)}: {paths}")
    return paths[0]


def _metric_series(
    text: str, label: str, pattern: str, expected_steps: int
) -> list[float]:
    values = [float(value) for value in re.findall(pattern, text)]
    if len(values) != expected_steps or any(
        not math.isfinite(value) for value in values
    ):
        raise RuntimeError(
            f"expected {expected_steps} finite {label} values, got {values}"
        )
    return values


def _rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open() as stream:
        for line_no, line in enumerate(stream, 1):
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"invalid JSON at {path}:{line_no}: {exc}") from exc
    if len(rows) != 32:
        raise RuntimeError(f"{path} has {len(rows)} rows, expected 32")
    return rows


def _lengths(path: Path) -> list[int]:
    try:
        lengths = [int(line.strip()) for line in path.read_text().splitlines()]
    except ValueError as exc:
        raise RuntimeError(f"invalid response length in {path}: {exc}") from exc
    if len(lengths) != 32:
        raise RuntimeError(f"{path} has {len(lengths)} lengths, expected 32")
    if min(lengths) < 1 or max(lengths) > 128:
        raise RuntimeError(
            f"unexpected generated work in {path}: n={len(lengths)} "
            f"total={sum(lengths)} min={min(lengths)} max={max(lengths)}")
    return lengths


def _step_files(directory: Path, pattern: str, expected_steps: int,
                label: str) -> list[Path]:
    paths = sorted(directory.glob(pattern))
    expected_names = (
        [f"{step}.jsonl" for step in range(1, expected_steps + 1)]
        if pattern == "*.jsonl"
        else [f"length_{step}.txt" for step in range(1, expected_steps + 1)]
    )
    if [path.name for path in paths] != expected_names:
        raise RuntimeError(
            f"unexpected {label} files: got {[path.name for path in paths]}, "
            f"expected {expected_names}"
        )
    return paths


def _arm(
    path: Path,
    mode: str,
    expected_steps: int,
    target_floor: int,
    target_policy: str,
) -> dict[str, Any]:
    log = _one(sorted((path / "logs").glob("*.txt")), f"{mode} log")
    outputs = _step_files(
        path / "rollout_data", "*.jsonl", expected_steps, f"{mode} rollout"
    )
    length_files = _step_files(
        path / "rollout_length",
        "length_*.txt",
        expected_steps,
        f"{mode} response lengths",
    )
    text = log.read_text(errors="replace")

    fatal_patterns = {
        "ray_task_error": r"RayTaskError",
        "npu_oom": r"(?:NPU out of memory|OutOfMemoryError)",
        "hccl_failure": r"(?:ERR\d+.*HCCL|HCCL.*(?:failed|error))",
        "application_fatal": r"UNKNOWN application exception",
    }
    faults = {name: len(re.findall(pattern, text, re.IGNORECASE))
              for name, pattern in fatal_patterns.items()}
    if any(faults.values()):
        raise RuntimeError(f"{mode} runtime faults: {faults}")
    if text.count("response/aborted_ratio:0.0") < expected_steps:
        raise RuntimeError(
            f"{mode} does not prove aborted_ratio=0 for {expected_steps} steps"
        )

    stage_ranks = {
        8: [8, 9, 10, 11, 12, 13, 14, 15],
        4: [12, 13, 14, 15],
        2: [14, 15],
    }
    expected_stages = [8, 4] if target_floor == 4 else [8, 4, 2]
    role_plan_markers = len(re.findall(
        rf"AdaFloor role plan loaded:.*target_policy={re.escape(target_policy)}(?:\s|$)",
        text,
    ))
    expected_role_plan_markers = 16 * expected_steps
    if role_plan_markers != expected_role_plan_markers:
        raise RuntimeError(
            f"{mode} does not prove target_policy={target_policy}: "
            f"role_plan_markers={role_plan_markers}/{expected_role_plan_markers}"
        )

    stage_markers = {}
    trigger_markers = {}
    previous_world_size = 16
    for stage in expected_stages:
        ranks = re.escape(str(stage_ranks[stage]))
        stage_markers[stage] = len(re.findall(
            rf"Elastic parallel shrink done:.*active_ranks={ranks}.*"
            rf"dp_size={stage}.*ep_size={stage}",
            text,
        ))
        trigger_markers[stage] = len(re.findall(
            rf"AdaFloor staged trigger:.*target={ranks}.*"
            rf"policy={re.escape(target_policy)}(?:\s|$)",
            text,
        ))
        expected_triggers = previous_world_size * expected_steps
        if trigger_markers[stage] != expected_triggers:
            raise RuntimeError(
                f"{mode} does not prove {target_policy} floor{stage} trigger: "
                f"{trigger_markers[stage]}/{expected_triggers}"
            )
        previous_world_size = stage
    restores = len(re.findall(
        r"Elastic parallel restore done:.*dp_size=16.*ep_size=16", text))
    expected_restores = 16 * expected_steps
    invalid_stages = {
        stage: (stage_markers[stage], stage * expected_steps)
        for stage in expected_stages
        if stage_markers[stage] != stage * expected_steps
    }
    if invalid_stages or restores != expected_restores:
        raise RuntimeError(
            f"{mode} incomplete floor{target_floor} lifecycle: "
            f"stages={stage_markers} invalid={invalid_stages} "
            f"restores={restores}/{expected_restores}")

    if mode == "graph":
        if text.count("FULL_DECODE_ONLY compilation enabled on NPU") < 16:
            raise RuntimeError("graph arm did not configure FULL_DECODE_ONLY on all ranks")
        if text.count("Replaying aclgraph") < 1:
            raise RuntimeError("graph arm has no ACLGraph replay evidence")
        for stage in expected_stages:
            ranks = str(stage_ranks[stage])
            expected = stage * expected_steps
            count = len(re.findall(
                rf"Elastic ACLGraph topology capture finished:.*active_ranks={re.escape(ranks)}",
                text))
            if count < expected:
                raise RuntimeError(
                    f"graph arm missing topology capture for {ranks}: {count}/{expected}")
        full_world_recaptures = text.count(
            "Finished recapturing rollout ACL graphs at "
            "generate_after_elastic_restore_and_resume"
        )
        expected_full_world_recaptures = 16 * (expected_steps - 1)
        if full_world_recaptures != expected_full_world_recaptures:
            raise RuntimeError(
                "graph arm has an unexpected number of next-step full-world "
                f"recaptures: {full_world_recaptures}/"
                f"{expected_full_world_recaptures}"
            )
    elif "Elastic ACLGraph topology capture" in text or "Replaying aclgraph" in text:
        raise RuntimeError("eager arm unexpectedly used ACLGraph")

    rollout_times = _metric_series(
        text,
        "rollout time",
        rf"=> rollout_output_time_s:\s*({FLOAT})",
        expected_steps,
    )
    throughputs = _metric_series(
        text,
        "rollout throughput",
        rf"=> rollouts speed tokens/s:\s*({FLOAT})",
        expected_steps,
    )
    rewards = _metric_series(
        text,
        "reward mean",
        rf"step:\d+[^\n]*critic/score/mean:({FLOAT})",
        expected_steps,
    )
    total_tokens = [
        int(value)
        for value in _metric_series(
            text,
            "total token count",
            rf"step:\d+[^\n]*perf/total_num_tokens:({FLOAT})",
            expected_steps,
        )
    ]

    step_results = []
    identities_by_step = []
    serialized_outputs_by_step = []
    decoded_lengths_by_step = []
    for step, (output, length_file) in enumerate(
        zip(outputs, length_files, strict=True), 1
    ):
        rows = _rows(output)
        decoded = _lengths(length_file)
        serialized_outputs = []
        identities = []
        for row in rows:
            required = {"request_id", "input", "output", "gts", "score", "step"}
            missing = required.difference(row)
            if missing:
                raise RuntimeError(f"{mode} rollout row misses {sorted(missing)}")
            if int(row["step"]) != step:
                raise RuntimeError(
                    f"{mode} rollout {output} contains row step={row['step']}"
                )
            serialized_outputs.append((row["output"], float(row["score"])))
            identities.append((str(row["request_id"]), row["input"], row["gts"]))
        if len(set(identities)) != 32:
            raise RuntimeError(f"{mode} step {step} request identities are not unique")
        step_results.append({
            "step": step,
            "output": str(output),
            "output_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
            "length_file": str(length_file),
            "length_sha256": hashlib.sha256(length_file.read_bytes()).hexdigest(),
            "rows": len(rows),
            "generated_response_tokens": sum(decoded),
            "prompt_and_response_tokens": total_tokens[step - 1],
            "rollout_time_s": rollout_times[step - 1],
            "logged_prompt_and_response_tokens_s": throughputs[step - 1],
            "response_tokens_s": sum(decoded) / rollout_times[step - 1],
            "reward_mean": rewards[step - 1],
        })
        identities_by_step.append(identities)
        serialized_outputs_by_step.append(serialized_outputs)
        decoded_lengths_by_step.append(decoded)

    return {
        "path": str(path),
        "log": str(log),
        "steps": step_results,
        "rows": sum(step["rows"] for step in step_results),
        "generated_response_tokens": sum(
            step["generated_response_tokens"] for step in step_results
        ),
        "prompt_and_response_tokens": sum(
            step["prompt_and_response_tokens"] for step in step_results
        ),
        "rollout_time_s": sum(rollout_times),
        "prompt_and_response_tokens_s": (
            sum(step["prompt_and_response_tokens"] for step in step_results)
            / sum(rollout_times)
        ),
        "response_tokens_s": (
            sum(step["generated_response_tokens"] for step in step_results)
            / sum(rollout_times)
        ),
        "target_floor": target_floor,
        "target_policy": target_policy,
        "role_plan_markers": role_plan_markers,
        "trigger_markers": {
            str(stage): trigger_markers[stage] for stage in expected_stages
        },
        "stage_markers": {
            str(stage): stage_markers[stage] for stage in expected_stages
        },
        "restore_markers": restores,
        "faults": faults,
        "identities": identities_by_step,
        "serialized_outputs": serialized_outputs_by_step,
        "decoded_lengths": decoded_lengths_by_step,
    }


def verify(
    eager_root: Path,
    graph_root: Path,
    expected_steps: int,
    target_floor: int = 4,
    target_policy: str = "planned",
) -> dict[str, Any]:
    if target_floor not in (2, 4):
        raise ValueError(f"target_floor must be 2 or 4, got {target_floor}")
    if target_policy not in ("planned", "natural"):
        raise ValueError(
            f"target_policy must be planned or natural, got {target_policy}"
        )
    eager = _arm(
        eager_root, "eager", expected_steps, target_floor, target_policy
    )
    graph = _arm(
        graph_root, "graph", expected_steps, target_floor, target_policy
    )
    if eager.pop("identities") != graph.pop("identities"):
        raise RuntimeError("eager and graph request identities differ")
    if eager.pop("serialized_outputs") != graph.pop("serialized_outputs"):
        raise RuntimeError("eager and graph serialized outputs differ")
    if eager.pop("decoded_lengths") != graph.pop("decoded_lengths"):
        raise RuntimeError("eager and graph decoded response lengths differ")
    if [step["output_sha256"] for step in eager["steps"]] != [
        step["output_sha256"] for step in graph["steps"]
    ]:
        raise RuntimeError("eager and graph rollout JSONL bytes differ")
    if [step["length_sha256"] for step in eager["steps"]] != [
        step["length_sha256"] for step in graph["steps"]
    ]:
        raise RuntimeError("eager and graph response-length bytes differ")
    if eager["generated_response_tokens"] != graph["generated_response_tokens"]:
        raise RuntimeError("eager and graph generated token totals differ")
    if eager["prompt_and_response_tokens"] != graph["prompt_and_response_tokens"]:
        raise RuntimeError("eager and graph prompt-and-response token totals differ")
    if [step["reward_mean"] for step in eager["steps"]] != [
        step["reward_mean"] for step in graph["steps"]
    ]:
        raise RuntimeError("eager and graph rewards differ")

    return {
        "schema_version": 4,
        "status": "PASS",
        "experiment": "qwen3_v014_adafloor_eager_tq2_vs_full_decode_tq1_gate",
        "claim_boundary": (
            f"{expected_steps} deterministic 16-NPU, 32-prompt, n=1 steps "
            f"verify v0.14 AdaFloor {target_policy} 16->8->4"
            f"{'->2' if target_floor == 2 else ''} shrink, full-world restore, and eager/graph "
            "serialized-output and decoded-length parity. The v0.14 rollout "
            "artifact does not retain token IDs. This is a functional gate, "
            "not a throughput result."
        ),
        "arms": {"eager_tq2": eager, "graph_tq1": graph},
        "paired": {
            "steps": expected_steps,
            "target_floor": target_floor,
            "target_policy": target_policy,
            "request_identity_equal": True,
            "serialized_outputs_equal": True,
            "decoded_lengths_equal": True,
            "rollout_jsonl_byte_equal": True,
            "length_file_byte_equal": True,
            "reward_equal": True,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path)
    parser.add_argument("--eager-root", type=Path)
    parser.add_argument("--graph-root", type=Path)
    parser.add_argument("--expected-steps", type=int, default=1)
    parser.add_argument("--target-floor", type=int, choices=(2, 4), default=4)
    parser.add_argument(
        "--target-policy", choices=("planned", "natural"), default="planned"
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.expected_steps < 1:
        parser.error("--expected-steps must be positive")
    if args.root is not None:
        if args.eager_root is not None or args.graph_root is not None:
            parser.error("use either --root or both explicit arm roots")
        eager_root = args.root / "eager_tq2"
        graph_root = args.root / "graph_tq1"
        default_output = args.root / "ADAFLOOR_V014_GATE_SUMMARY.json"
    else:
        if args.eager_root is None or args.graph_root is None:
            parser.error("--eager-root and --graph-root are both required without --root")
        eager_root = args.eager_root
        graph_root = args.graph_root
        default_output = graph_root / "ADAFLOOR_V014_GATE_SUMMARY.json"
    result = verify(
        eager_root.resolve(),
        graph_root.resolve(),
        args.expected_steps,
        args.target_floor,
        args.target_policy,
    )
    output = args.output or default_output
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
