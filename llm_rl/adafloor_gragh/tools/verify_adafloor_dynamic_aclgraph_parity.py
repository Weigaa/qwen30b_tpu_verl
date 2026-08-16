#!/usr/bin/env python3
"""Verify dynamic AdaFloor FULL_DECODE_ONLY graph/eager parity gates."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any


RESPONSES = 32
TOKENS = 640


class VerificationError(RuntimeError):
    pass


def fail(message: str) -> None:
    raise VerificationError(message)


def require_file(path: Path, label: str) -> Path:
    if path.is_symlink() or not path.is_file():
        fail(f"missing regular {label}: {path}")
    return path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact(path: Path, label: str) -> dict[str, Any]:
    path = require_file(path, label)
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def load_json(path: Path, label: str) -> Any:
    try:
        return json.loads(require_file(path, label).read_text("utf-8"))
    except json.JSONDecodeError as error:
        raise VerificationError(f"invalid {label} {path}: {error}") from error


def parse_env(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for number, line in enumerate(
        require_file(path, "protocol").read_text("utf-8").splitlines(), 1
    ):
        if not line or "=" not in line:
            fail(f"invalid protocol line {number}: {line!r}")
        key, value = line.split("=", 1)
        if not key or key in result:
            fail(f"duplicate or empty protocol key at line {number}")
        result[key] = value
    return result


def verify_protocol(root: Path, mode: str) -> dict[str, Any]:
    path = root / "protocol.env"
    values = parse_env(path)
    expected = {
        "schema_version": "1",
        "experiment": "qwen3_adafloor_full_decode_dynamic_gate",
        "execution_mode": mode,
        "stack": "vllm-0.11.0_vllm-ascend-0.11.0rc0_cann-8.5.0",
        "cudagraph_mode": "FULL_DECODE_ONLY" if mode == "graph" else "NONE",
        "capture_sizes": "2" if mode == "graph" else "none",
        "attention_graph": "true" if mode == "graph" else "false",
        "moe_graph": "true" if mode == "graph" else "false",
        "task_queue_enable": "1" if mode == "graph" else "2",
        "plan_steps": "2",
    }
    for key, expected_value in expected.items():
        if values.get(key) != expected_value:
            fail(
                f"{mode} protocol {key}={values.get(key)!r}, "
                f"expected {expected_value!r}"
            )
    policy = values.get("policy")
    if policy not in {"natural", "planned"}:
        fail(f"unsupported {mode} policy {policy!r}")
    try:
        executed_steps = int(values.get("executed_steps", ""))
        forced_floors = [int(value) for value in values.get("forced_floors", "").split(",")]
    except ValueError as error:
        raise VerificationError(f"invalid {mode} step/floor protocol") from error
    if executed_steps not in (1, 2):
        fail(f"{mode} executed_steps must be 1 or 2")
    if forced_floors not in ([2, 16], [4, 16]):
        fail(f"{mode} forced_floors are unsupported: {forced_floors}")
    if forced_floors[0] == 2:
        floor2_expected = {
            "profile_max_tokens": "2048",
            "profile_expected_moe_comm": "alltoall",
            "max_num_batched_tokens": "17408",
            "max_model_len": "17408",
            "kv_tokens_floor2": "98304",
        }
        for key, expected_value in floor2_expected.items():
            if values.get(key) != expected_value:
                fail(
                    f"{mode} protocol {key}={values.get(key)!r}, "
                    f"expected {expected_value!r}"
                )
    return {"values": values, "artifact": artifact(path, f"{mode} protocol")}


def verify_plan(
    root: Path,
    expected_floors: list[int],
) -> tuple[dict[str, Any], dict[str, Any]]:
    path = root / "oracle/length_sorted_rank_plan.json"
    plan = load_json(path, "rank plan")
    if not isinstance(plan, list) or len(plan) != 2:
        fail("rank plan must contain two steps")
    floors = [row.get("selected_floor") for row in plan if isinstance(row, dict)]
    stages = [row.get("shrink_stages") for row in plan if isinstance(row, dict)]
    expected_stages = [[8, 4, 2] if expected_floors[0] == 2 else [8, 4], [16]]
    if floors != expected_floors or stages != expected_stages:
        fail(f"rank plan floors/stages differ: {floors}/{stages}")
    if any(row.get("tail_guard_enabled") is not False for row in plan):
        fail("rank plan must disable TailGuard")
    return {
        "selected_floors": floors,
        "shrink_stages": stages,
        "tail_guard_enabled": False,
    }, artifact(path, "rank plan")


def read_outputs(
    root: Path,
    executed_steps: int,
) -> tuple[dict[tuple[int, int, str], dict[str, Any]], dict[str, Any]]:
    rows: dict[tuple[int, int, str], dict[str, Any]] = {}
    output_artifacts: dict[str, Any] = {}
    for step in range(1, executed_steps + 1):
        first_request_id = 2 * (step - 1)
        expected_request_ids = {
            str(first_request_id),
            str(first_request_id + 1),
        }
        expected_step_keys = {
            (rank, request_id)
            for rank in range(16)
            for request_id in expected_request_ids
        }
        path = require_file(root / f"rollout_data/{step}.jsonl", f"step-{step} JSONL")
        length_path = require_file(
            root / f"rollout_length/length_{step}.txt", f"step-{step} lengths"
        )
        try:
            lengths = [
                int(value) for value in length_path.read_text("utf-8").splitlines()
            ]
        except ValueError as error:
            raise VerificationError(
                f"step {step} length artifact contains a non-integer"
            ) from error
        step_rows: dict[tuple[int, str], dict[str, Any]] = {}
        with path.open("r", encoding="utf-8") as source:
            for number, line in enumerate(source, 1):
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as error:
                    raise VerificationError(
                        f"invalid step {step} JSONL row {number}: {error}"
                    ) from error
                rank = row.get("rollout_rank")
                request_id = row.get("request_id")
                if row.get("step") != step:
                    fail(f"step {step} row {number} carries the wrong step")
                if not isinstance(rank, int) or rank not in range(16):
                    fail(f"step {step} row {number} has invalid rollout_rank")
                if (
                    not isinstance(request_id, str)
                    or request_id not in expected_request_ids
                ):
                    fail(f"step {step} row {number} has invalid request_id")
                step_key = (rank, request_id)
                if step_key in step_rows:
                    fail(f"duplicate stable request key {(step, *step_key)}")
                length = row.get("decoded_response_length")
                responses = row.get("responses")
                mask = row.get("response_mask")
                if not isinstance(length, int) or length <= 0:
                    fail(f"step {step} row {number} has invalid decoded length")
                if not isinstance(responses, list) or not isinstance(mask, list):
                    fail(f"step {step} row {number} lacks response arrays")
                if len(responses) != len(mask) or sum(mask) != length:
                    fail(
                        f"step {step} row {number} response mask does not match "
                        "decoded length"
                    )
                if mask[:length] != [1] * length or any(mask[length:]):
                    fail(f"step {step} row {number} response mask is not contiguous")
                step_rows[step_key] = row
                rows[(step, rank, request_id)] = row
        if set(step_rows) != expected_step_keys or len(lengths) != RESPONSES:
            fail(f"step {step} does not contain two stable requests per rank")
        decoded = [row["decoded_response_length"] for row in step_rows.values()]
        if sorted(decoded) != sorted(lengths) or sum(decoded) != TOKENS:
            fail(f"step {step} lengths do not match the 640-token gate contract")
        output_artifacts[str(step)] = {
            "responses": len(step_rows),
            "generated_tokens": sum(decoded),
            "min_tokens": min(decoded),
            "max_tokens": max(decoded),
            "jsonl": artifact(path, f"step-{step} JSONL"),
            "lengths": artifact(length_path, f"step-{step} lengths"),
        }
    return rows, output_artifacts


def verify(eager_root: Path, graph_root: Path) -> dict[str, Any]:
    eager_root = eager_root.resolve()
    graph_root = graph_root.resolve()
    protocols = {
        "eager": verify_protocol(eager_root, "eager"),
        "graph": verify_protocol(graph_root, "graph"),
    }
    eager_values = protocols["eager"]["values"]
    graph_values = protocols["graph"]["values"]
    paired_protocol_fields = (
        "policy",
        "plan_steps",
        "executed_steps",
        "forced_floors",
        "tail_validation_tokens",
        "baseline",
    )
    if any(
        eager_values.get(key) != graph_values.get(key)
        for key in paired_protocol_fields
    ):
        fail("eager and graph paired protocol fields differ")
    policy = eager_values["policy"]
    executed_steps = int(eager_values["executed_steps"])
    expected_floors = [int(value) for value in eager_values["forced_floors"].split(",")]
    eager_plan, eager_plan_artifact = verify_plan(eager_root, expected_floors)
    graph_plan, graph_plan_artifact = verify_plan(graph_root, expected_floors)
    if eager_plan != graph_plan or eager_plan_artifact["sha256"] != graph_plan_artifact["sha256"]:
        fail("eager and graph rank plans differ")
    eager_dataset = artifact(
        eager_root / "oracle/length_sorted_train.parquet", "eager planner parquet"
    )
    graph_dataset = artifact(
        graph_root / "oracle/length_sorted_train.parquet", "graph planner parquet"
    )
    if eager_dataset["sha256"] != graph_dataset["sha256"]:
        fail("eager and graph planner parquets differ")

    eager_rows, eager_outputs = read_outputs(eager_root, executed_steps)
    graph_rows, graph_outputs = read_outputs(graph_root, executed_steps)
    if set(eager_rows) != set(graph_rows):
        fail("eager and graph stable request keys differ")
    mismatches = [key for key in sorted(eager_rows) if eager_rows[key] != graph_rows[key]]
    if mismatches:
        fail(f"eager and graph response rows differ at {mismatches[:4]}")
    byte_identical_by_step = {
        step: eager_outputs[step]["jsonl"]["sha256"]
        == graph_outputs[step]["jsonl"]["sha256"]
        for step in eager_outputs
    }
    if not all(byte_identical_by_step.values()):
        fail(
            "canonical response rows match but one or more JSONL files are not "
            "byte-identical"
        )

    smoke_path = graph_root / "ACLGRAPH_SMOKE_SUMMARY.json"
    smoke = load_json(smoke_path, "graph smoke summary")
    runtime = smoke.get("runtime", {}) if isinstance(smoke, dict) else {}
    if smoke.get("status") != "PASS":
        fail("graph smoke summary is not PASS")
    expected_runtime = {
        "cudagraph_mode": "FULL_DECODE_ONLY",
        "capture_sizes": [2],
        "resident_full_decode_graphs_per_active_rank": 1,
        "single_full_decode_graph_per_active_rank": True,
        "topology_graph_cache": False,
    }
    for key, value in expected_runtime.items():
        if runtime.get(key) != value:
            fail(f"graph smoke runtime {key}={runtime.get(key)!r}, expected {value!r}")
    faults = runtime.get("contract_fault_counts")
    if not isinstance(faults, dict) or any(faults.values()):
        fail(f"graph smoke contains runtime faults: {faults!r}")
    if runtime.get("target_policy") != policy:
        fail("graph smoke target policy differs from paired protocol")
    expected_separation = {
        "mixed_full_capture_events": 0,
        "mixed_full_replay_events": 0,
        "prefill_execution": "eager",
        "uniform_decode_execution": "FULL_DECODE_ONLY",
    }
    if runtime.get("prefill_decode_separation") != expected_separation:
        fail("graph smoke does not prove eager-prefill/full-decode separation")
    # shrink_stages contains floor labels, while capture cardinality is the
    # number of active workers at each topology.
    expected_capture_invocations = 16 + sum(eager_plan["shrink_stages"][0])
    if executed_steps == 2:
        expected_capture_invocations += 16
    if runtime.get("capture_invocations") != expected_capture_invocations:
        fail(
            "graph smoke capture invocations differ from the topology contract: "
            f"{runtime.get('capture_invocations')!r} vs "
            f"{expected_capture_invocations}"
        )
    step_2_workers = runtime.get("step_2_recapture_workers")
    if executed_steps == 2:
        if not isinstance(step_2_workers, list) or len(step_2_workers) != 16:
            fail("two-step graph smoke lacks 16-worker step-2 recapture")
    elif step_2_workers != []:
        fail("single-step graph smoke unexpectedly reports step-2 recapture")

    return {
        "schema_version": 1,
        "status": "PASS",
        "experiment": (
            f"adafloor_{policy}_f{expected_floors[0]}_"
            f"{executed_steps}step_single_aclgraph_parity"
        ),
        "roots": {"eager": str(eager_root), "graph": str(graph_root)},
        "configuration": {
            "policy": f"{policy.capitalize()}-F{expected_floors[0]}",
            "executed_steps": executed_steps,
            "eager": "eager+TASK_QUEUE_ENABLE=2",
            "graph": "FULL_DECODE_ONLY ACLGraph+TASK_QUEUE_ENABLE=1",
            "capture_sizes": [2],
            "resident_full_decode_graphs_per_active_rank": 1,
            "topology_graph_cache": False,
        },
        "plan": eager_plan,
        "correctness": {
            "stable_key": ["step", "rollout_rank", "request_id"],
            "paired_responses": RESPONSES * executed_steps,
            "generated_tokens_per_arm": TOKENS * executed_steps,
            "all_fields_equal": True,
            "valid_response_tokens_equal": True,
            "jsonl_byte_identical": all(byte_identical_by_step.values()),
            "jsonl_byte_identical_by_step": byte_identical_by_step,
            "jsonl_sha256_by_step": {
                step: eager_outputs[step]["jsonl"]["sha256"]
                for step in eager_outputs
            },
        },
        "graph_lifecycle": {
            key: runtime[key] for key in expected_runtime
        } | {
            "capture_invocations": runtime.get("capture_invocations"),
            "captured_graph_instances": runtime.get("captured_graph_instances"),
            "max_observed_capture_gib_per_worker": runtime.get(
                "max_observed_capture_gib_per_worker"
            ),
            "contract_fault_counts": faults,
            "prefill_decode_separation": expected_separation,
            "step_2_recapture_workers": step_2_workers,
        },
        "artifacts": {
            "verifier": artifact(Path(__file__).resolve(), "parity verifier"),
            "protocols": {mode: row["artifact"] for mode, row in protocols.items()},
            "plans": {"eager": eager_plan_artifact, "graph": graph_plan_artifact},
            "planner_parquets": {"eager": eager_dataset, "graph": graph_dataset},
            "outputs": {"eager": eager_outputs, "graph": graph_outputs},
            "graph_smoke_summary": artifact(smoke_path, "graph smoke summary"),
        },
        "claim_boundary": (
            f"This deterministic {executed_steps}-step gate proves output "
            f"parity for one {policy.capitalize()}-F{expected_floors[0]} "
            "shrink/restore execution on the pinned single-node stack. It does "
            "not measure production throughput or training convergence."
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eager-root", type=Path, required=True)
    parser.add_argument("--graph-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = args.output or args.graph_root / "DYNAMIC_ACLGRAPH_PARITY_SUMMARY.json"
    try:
        result = verify(args.eager_root, args.graph_root)
    except VerificationError as error:
        print(f"FAIL: {error}")
        return 1
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", "utf-8")
    print(f"PASS: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
