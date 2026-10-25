#!/usr/bin/env python3
"""Verify the six-arm v0.14 eager/graph one-step AdaFloor matrix."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any


FLOAT = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
EXPECTED_ARMS = [
    "eager_tq2_vanilla",
    "graph_tq1_vanilla",
    "graph_tq1_ada_n2",
    "eager_tq2_ada_n2",
    "eager_tq2_ada_p_f4",
    "graph_tq1_ada_p_f4",
]
STAGE_RANKS = {
    8: list(range(8, 16)),
    4: list(range(12, 16)),
    2: [14, 15],
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def one(paths: list[Path], label: str) -> Path:
    if len(paths) != 1:
        raise RuntimeError(f"expected one {label}, found {len(paths)}: {paths}")
    return paths[0]


def one_metric(text: str, label: str, pattern: str) -> float:
    values = [float(value) for value in re.findall(pattern, text)]
    if len(values) != 1 or not math.isfinite(values[0]):
        raise RuntimeError(f"expected one finite {label}, got {values}")
    return values[0]


def read_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"invalid JSON {path}:{line_no}: {exc}") from exc
    if len(rows) != 512:
        raise RuntimeError(f"{path} contains {len(rows)} rows, expected 512")
    return rows


def read_lengths(path: Path) -> list[int]:
    try:
        values = [int(line) for line in path.read_text().splitlines()]
    except ValueError as exc:
        raise RuntimeError(f"invalid response length in {path}") from exc
    if len(values) != 512 or min(values) < 1 or max(values) > 4608:
        raise RuntimeError(
            f"invalid work in {path}: rows={len(values)} "
            f"min={min(values, default=-1)} max={max(values, default=-1)}"
        )
    return values


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def validate_output_contract(
    rows: list[dict[str, Any]], lengths: list[int], name: str
) -> dict[str, Any]:
    """Validate both the stock v0.14 and extended AdaFloor JSONL schemas."""
    required_base = {"input", "gts", "output", "score", "step", "request_id"}
    extended = {
        "prompt_occurrence_ordinal", "rollout_sample_index", "rollout_prompt_hash",
        "rollout_request_seed", "decoded_response_length", "responses", "response_mask",
    }
    identities: set[tuple[str, str, str]] = set()
    lengths_by_identity: dict[tuple[str, str, str], int] = {}
    outputs_by_identity: dict[tuple[str, str, str], str] = {}
    scores_by_identity: dict[tuple[str, str, str], str] = {}
    prompt_counts: Counter[tuple[str, str]] = Counter()
    extended_rows = 0

    for row, length in zip(rows, lengths, strict=True):
        missing = required_base.difference(row)
        if missing:
            raise RuntimeError(f"{name} row misses {sorted(missing)}")
        if int(row["step"]) != 1:
            raise RuntimeError(f"{name} contains a non-step-1 row")
        input_text = str(row["input"])
        ground_truth = canonical_json(row["gts"])
        request_id = str(row["request_id"])
        key = (input_text, ground_truth, request_id)
        if key in identities:
            raise RuntimeError(f"{name} duplicate request identity {key}")
        identities.add(key)
        lengths_by_identity[key] = length
        outputs_by_identity[key] = str(row["output"])
        scores_by_identity[key] = canonical_json(row["score"])
        prompt_counts[(input_text, ground_truth)] += 1

        present_extended = extended.intersection(row)
        if present_extended:
            missing_extended = extended.difference(row)
            if missing_extended:
                raise RuntimeError(
                    f"{name} partially populated extended row misses "
                    f"{sorted(missing_extended)}"
                )
            extended_rows += 1
            decoded = int(row["decoded_response_length"])
            mask = [int(value) for value in row["response_mask"]]
            if decoded != length or decoded != sum(mask):
                raise RuntimeError(f"{name} invalid extended response length at {key}")
            if mask[:decoded] != [1] * decoded or any(mask[decoded:]):
                raise RuntimeError(f"{name} non-contiguous response mask at {key}")
            if len(row["responses"]) < decoded:
                raise RuntimeError(f"{name} response token prefix is too short at {key}")

    if len(identities) != 512:
        raise RuntimeError(f"{name} request identity count={len(identities)}/512")
    if len(prompt_counts) != 32 or set(prompt_counts.values()) != {16}:
        raise RuntimeError(f"{name} prompt replication contract failed")
    if extended_rows not in (0, 512):
        raise RuntimeError(f"{name} mixes base and extended JSONL rows")

    identity_digest = hashlib.sha256()
    for key in sorted(identities):
        identity_digest.update(canonical_json(key).encode("utf-8"))
        identity_digest.update(b"\n")
    return {
        "schema": "extended" if extended_rows else "v0.14_base",
        "identity_sha256": identity_digest.hexdigest(),
        "identities": identities,
        "lengths_by_identity": lengths_by_identity,
        "outputs_by_identity": outputs_by_identity,
        "scores_by_identity": scores_by_identity,
    }


def parse_rank_list(value: str) -> tuple[int, ...]:
    parsed = ast.literal_eval(value)
    if not isinstance(parsed, list) or not all(isinstance(rank, int) for rank in parsed):
        raise RuntimeError(f"invalid rank list: {value}")
    if len(set(parsed)) != len(parsed):
        raise RuntimeError(f"duplicate rank in list: {value}")
    return tuple(parsed)


def verify_shrink_transitions(
    text: str, name: str, method: str, graph: bool
) -> list[list[int]]:
    policy = "natural" if method == "ada_n2" else "planned"
    stage_sizes = [8, 4, 2] if method == "ada_n2" else [8, 4]
    trigger_pattern = re.compile(
        rf"AdaFloor staged trigger:.*?current=(\[[^\]]*\]).*?"
        rf"unfinished=(\[[^\]]*\]).*?target=(\[[^\]]*\]).*?"
        rf"policy={policy}(?:\s|$)"
    )
    triggers = [
        tuple(parse_rank_list(value) for value in match.groups())
        for match in trigger_pattern.finditer(text)
    ]
    shrink_pattern = re.compile(
        r"Elastic parallel shrink done:.*?active_ranks=(\[[^\]]*\]).*?"
        r"dp_size=(\d+).*?ep_size=(\d+)"
    )
    shrinks = [
        (parse_rank_list(ranks), int(dp_size), int(ep_size))
        for ranks, dp_size, ep_size in shrink_pattern.findall(text)
    ]
    capture_pattern = re.compile(
        r"Elastic ACLGraph topology capture finished:.*?"
        r"active_ranks=(\[[^\]]*\])"
    )
    captures = [parse_rank_list(ranks) for ranks in capture_pattern.findall(text)]

    current = tuple(range(16))
    actual_stages: list[list[int]] = []
    for stage_size in stage_sizes:
        stage_triggers = [record for record in triggers if record[0] == current]
        if len(stage_triggers) != len(current):
            raise RuntimeError(
                f"{name} has {len(stage_triggers)}/{len(current)} coordinated "
                f"trigger reports for {len(current)}->{stage_size}"
            )
        target_sets = {record[2] for record in stage_triggers}
        unfinished_sets = {record[1] for record in stage_triggers}
        if len(target_sets) != 1 or len(unfinished_sets) != 1:
            raise RuntimeError(f"{name} ranks disagree on the {stage_size}-rank transition")
        target = next(iter(target_sets))
        unfinished = next(iter(unfinished_sets))
        if len(target) != stage_size or not set(target).issubset(current):
            raise RuntimeError(f"{name} has invalid target ranks for floor{stage_size}: {target}")
        if policy == "natural":
            if target != unfinished:
                raise RuntimeError(
                    f"{name} Natural target does not match unfinished ranks: "
                    f"target={target} unfinished={unfinished}"
                )
        elif target != tuple(STAGE_RANKS[stage_size]):
            raise RuntimeError(
                f"{name} Planned target changed at floor{stage_size}: {target}"
            )

        stage_shrinks = [
            record for record in shrinks
            if record == (target, stage_size, stage_size)
        ]
        if len(stage_shrinks) != stage_size:
            raise RuntimeError(
                f"{name} has {len(stage_shrinks)}/{stage_size} shrink completions "
                f"for target={target}"
            )
        if graph:
            stage_captures = sum(capture == target for capture in captures)
            if stage_captures != stage_size:
                raise RuntimeError(
                    f"{name} has {stage_captures}/{stage_size} graph captures "
                    f"for target={target}"
                )
        actual_stages.append(list(target))
        current = target

    if len(triggers) != sum([16, *stage_sizes[:-1]]):
        raise RuntimeError(f"{name} has unexpected extra/missing transition triggers")
    if len(shrinks) != sum(stage_sizes):
        raise RuntimeError(f"{name} has unexpected extra/missing shrink completions")
    return actual_stages


def verify_arm(root: Path, arm: dict[str, Any]) -> dict[str, Any]:
    name = arm["name"]
    mode = arm["mode"]
    method = arm["method"]
    path = root / name
    manifest_path = path / "arm_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("status") != "complete" or manifest.get("name") != name:
        raise RuntimeError(f"invalid arm manifest: {manifest_path}")
    log_path = one(sorted((path / "logs").glob("*.txt")), f"{name} log")
    output_path = one(sorted((path / "rollout_data").glob("*.jsonl")), f"{name} output")
    length_path = one(sorted((path / "rollout_length").glob("length_*.txt")), f"{name} lengths")
    text = log_path.read_text(errors="replace")

    faults = {
        "ray_task_error": len(re.findall(r"RayTaskError", text)),
        "npu_oom": len(re.findall(r"NPU out of memory|OutOfMemoryError", text, re.I)),
        "hccl_failure": len(re.findall(r"ERR\d+.*HCCL|HCCL.*(?:failed|error)", text, re.I)),
        "application_fatal": len(re.findall(r"UNKNOWN application exception", text)),
    }
    if any(faults.values()):
        raise RuntimeError(f"{name} has contract-defined faults: {faults}")
    if "response/aborted_ratio:0.0" not in text:
        raise RuntimeError(f"{name} does not prove aborted_ratio=0")
    if "Resetting trainer progress after loading checkpoint weights" not in text:
        raise RuntimeError(f"{name} did not reset child-run trainer progress")

    stages: list[int] = []
    stage_survivor_ranks: list[list[int]] = []
    if method == "vanilla":
        if "Elastic parallel shrink done" in text:
            raise RuntimeError(f"{name} unexpectedly shrank")
    else:
        stages = [8, 4, 2] if method == "ada_n2" else [8, 4]
        policy = "natural" if method == "ada_n2" else "planned"
        if len(re.findall(
            rf"AdaFloor role plan loaded:.*target_policy={policy}(?:\s|$)", text
        )) != 16:
            raise RuntimeError(f"{name} lacks the 16-rank {policy} role plan")
        stage_survivor_ranks = verify_shrink_transitions(
            text, name, method, mode == "graph"
        )
        restores = len(re.findall(
            r"Elastic parallel restore done:.*dp_size=16.*ep_size=16", text
        ))
        if restores != 16:
            raise RuntimeError(f"{name} full-world restore count={restores}/16")

    if mode == "graph":
        if text.count("FULL_DECODE_ONLY compilation enabled on NPU") < 16:
            raise RuntimeError(f"{name} did not configure FULL_DECODE_ONLY on 16 ranks")
        if "Replaying aclgraph" not in text:
            raise RuntimeError(f"{name} has no ACLGraph replay evidence")
    elif "Replaying aclgraph" in text or "FULL_DECODE_ONLY compilation enabled on NPU" in text:
        raise RuntimeError(f"{name} eager arm unexpectedly used ACLGraph")

    rows = read_rows(output_path)
    lengths = read_lengths(length_path)
    output_contract = validate_output_contract(rows, lengths, name)

    rollout_s = one_metric(
        text, "rollout time", rf"=> rollout_output_time_s:\s*({FLOAT})"
    )
    reward = one_metric(
        text, "reward", rf"step:\d+[^\n]*critic/score/mean:({FLOAT})"
    )
    response_tokens = sum(lengths)
    return {
        "name": name,
        "mode": mode,
        "method": method,
        "log": str(log_path),
        "log_sha256": sha256(log_path),
        "output": str(output_path),
        "output_sha256": sha256(output_path),
        "lengths": str(length_path),
        "lengths_sha256": sha256(length_path),
        "responses": 512,
        "output_schema": output_contract["schema"],
        "request_identity_sha256": output_contract["identity_sha256"],
        "generated_response_tokens": response_tokens,
        "rollout_time_s": rollout_s,
        "response_tokens_s": response_tokens / rollout_s,
        "reward_mean": reward,
        "stages": stages,
        "stage_survivor_ranks": stage_survivor_ranks,
        "faults": faults,
        "_output_contract": output_contract,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    protocol_path = args.root / "protocol.json"
    manifest_path = args.root / "run_manifest.json"
    protocol = json.loads(protocol_path.read_text())
    completed = json.loads(manifest_path.read_text()) if manifest_path.exists() else []
    names = [entry["name"] for entry in completed]
    if names != EXPECTED_ARMS[:len(names)]:
        raise RuntimeError(f"run manifest is not a canonical prefix: {names}")
    if not args.allow_incomplete and names != EXPECTED_ARMS:
        raise RuntimeError(f"expected all six arms, got {names}")
    if args.allow_incomplete and len(names) > len(EXPECTED_ARMS):
        raise RuntimeError("run manifest contains extra arms")

    results = [verify_arm(args.root, arm) for arm in completed]
    reference = None
    reference_name = None
    for result in results:
        contract = result.pop("_output_contract")
        if reference is None:
            reference = contract
            reference_name = result["name"]
            result["comparison_to_reference"] = None
        elif contract["identities"] != reference["identities"]:
            raise RuntimeError(
                f"{result['name']} does not match common request identities"
            )
        else:
            identities = reference["identities"]
            result["comparison_to_reference"] = {
                "reference": reference_name,
                "matching_lengths": sum(
                    contract["lengths_by_identity"][key]
                    == reference["lengths_by_identity"][key]
                    for key in identities
                ),
                "matching_outputs": sum(
                    contract["outputs_by_identity"][key]
                    == reference["outputs_by_identity"][key]
                    for key in identities
                ),
                "matching_scores": sum(
                    contract["scores_by_identity"][key]
                    == reference["scores_by_identity"][key]
                    for key in identities
                ),
                "responses": len(identities),
            }
    status = "PASS" if names == EXPECTED_ARMS else "INCOMPLETE"
    summary = {
        "schema_version": 1,
        "status": status,
        "protocol": str(protocol_path),
        "protocol_sha256": sha256(protocol_path),
        "common_checkpoint": protocol["common_checkpoint"],
        "selected_dataset_sha256": protocol["selected_dataset_sha256"],
        "completed_arms": names,
        "pending_arms": EXPECTED_ARMS[len(names):],
        "arms": results,
    }
    output_path = args.root / "one_step_matrix_summary.json"
    output_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
