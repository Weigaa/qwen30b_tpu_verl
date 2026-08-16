#!/usr/bin/env python3
"""Verify the graph-produced epoch0 and full Qwen3 graph/eager matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any


SOURCE_ROOT = Path(__file__).resolve().parents[1]
WORLD_SIZE = 16
STEPS = 5
RESPONSES_PER_STEP = 512
SEED = 101
FULL16_KV_TOKENS = {"eager": 380800, "graph": 380800}
TASK_QUEUE_ENABLE = {"eager": 2, "graph": 1}
CAPTURE_SIZES = [1, 2, 4, 8, 16, 32]
ORDER = [
    ("vanilla", "eager"),
    ("vanilla", "graph"),
    ("lengthsort_guard", "graph"),
    ("lengthsort_guard", "eager"),
    ("planned", "eager"),
    ("planned", "graph"),
    ("natural", "graph"),
    ("natural", "eager"),
]
POLICIES = ("vanilla", "lengthsort_guard", "planned", "natural")
ANSI_ESCAPE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
WORKER_PID = re.compile(r"WorkerDict pid=(\d+)")
CAPTURE = re.compile(
    r"Graph capturing finished in (?P<seconds>\d+) secs, "
    r"took (?P<gib>[0-9]+(?:\.[0-9]+)?) GiB"
)
CAPTURE_SIZES_RE = re.compile(
    r'"cudagraph_capture_sizes":\[(?P<sizes>[0-9, ]+)\]'
)
SHRINK = re.compile(
    r"Elastic parallel shrink rpc done: global_rank=(?P<rank>\d+) "
    r"active_ranks=\[(?P<active>[^]]*)\]"
)
TOPOLOGY_CAPTURE = re.compile(
    r"Elastic full-MoE ACLGraph topology capture "
    r"(?P<phase>starting|finished): rank=(?P<rank>\d+) "
    r"active_ranks=\[(?P<active>[^]]*)\]"
)
RESIZE = re.compile(
    r"rollout_worker_resize_start rank=(?P<rank>\d+) step=(?P<step>\d+) "
    r"epoch=\d+ target_floor=(?P<floor>\d+) target_kv=(?P<kv>\d+)"
)
FORBIDDEN = {
    "ray_task_error": re.compile(r"\bRayTaskError\b"),
    "worker_crash": re.compile(r"\b(?:WorkerCrashedError|ActorDiedError)\b"),
    "npu_oom": re.compile(
        r"OutOfMemoryError|ACL_ERROR_RT_MEMORY_ALLOCATION|"
        r"NPU out of memory|Memory_Allocation_Failure",
        re.IGNORECASE,
    ),
    "hccl_failure": re.compile(
        r"HCCL.*(?:fail|error)|EJ0003|ERR\d+.*HCCL", re.IGNORECASE
    ),
    "aclgraph_failure": re.compile(
        r"ACLgraph sizes capture fail|stale input-address|"
        r"Skipping ACL graph capture|falling back to NONE|"
        r"error code:?\s*507011|CCU instruction address check error|"
        r"Not allow to synchronize captured-stream",
        re.IGNORECASE,
    ),
}


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


def artifact(path: Path) -> dict[str, Any]:
    path = require_file(path, "artifact")
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


def only_file(directory: Path, pattern: str, label: str) -> Path:
    matches = sorted(directory.glob(pattern))
    if len(matches) != 1:
        fail(f"expected one {label} in {directory}, found {len(matches)}")
    return require_file(matches[0], label)


def parse_env(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for number, line in enumerate(
        require_file(path, "protocol contract").read_text("utf-8").splitlines(), 1
    ):
        if not line or "=" not in line:
            fail(f"invalid protocol line {number}: {line!r}")
        key, value = line.split("=", 1)
        if not key or key in result:
            fail(f"duplicate or empty protocol key at line {number}")
        result[key] = value
    return result


def verify_code_contract(root: Path) -> dict[str, Any]:
    path = require_file(root / "code_sha256.txt", "code contract")
    rows = []
    for number, line in enumerate(path.read_text("utf-8").splitlines(), 1):
        parts = line.split(maxsplit=1)
        if len(parts) != 2 or not re.fullmatch(r"[0-9a-f]{64}", parts[0]):
            fail(f"invalid code-contract line {number}")
        expected, raw_path = parts
        source = require_file(Path(raw_path.strip()), "frozen source")
        actual = sha256(source)
        if actual != expected:
            fail(f"frozen source changed: {source}")
        rows.append({"path": str(source), "sha256": actual})
    if len(rows) < 20:
        fail(f"code contract covers only {len(rows)} files")
    return {"manifest": artifact(path), "files": rows}


def verify_protocol(root: Path) -> dict[str, Any]:
    path = root / "protocol.env"
    values = parse_env(path)
    expected = {
        "schema_version": "2",
        "experiment": "qwen3_aclgraph_full_epoch_matrix",
        "common_epoch0_mode": "graph_vanilla",
        "common_epoch0_actor_updated": "true",
        "common_epoch0_steps": "5",
        "seed": "101",
        "epoch1_actor_frozen": "true",
        "paired_request_sampling_seeds": "true",
        "policies": "vanilla,lengthsort_guard,planned,natural",
        "modes": "eager,graph",
        "launch_order": "vanilla:eager,vanilla:graph,lengthsort_guard:graph,lengthsort_guard:eager,planned:eager,planned:graph,natural:graph,natural:eager",
        "prompts_per_step": "32",
        "rollout_n": "16",
        "responses_per_step": "512",
        "steps": "5",
        "max_response_length": "16384",
        "tail_guard": "policy_default",
        "eager_full16_kv_tokens": "380800",
        "graph_full16_kv_tokens": "380800",
        "kv_bytes_per_token": "98304",
        "graph_capture_sizes": "[1,2,4,8,16,32]",
        "graph_capture_profile": "balanced",
        "graph_mode": "FULL_DECODE_ONLY",
        "graph_attention": "true",
        "graph_moe": "true",
        "eager_task_queue_enable": "2",
        "graph_task_queue_enable": "1",
        "torchair": "false",
        "sidecar": "false",
        "moe_shared_expert_overlap": "false",
    }
    for key, value in expected.items():
        if values.get(key) != value:
            fail(f"protocol {key}={values.get(key)!r}, expected {value!r}")
    common_root = values.get("common_epoch0_root", "")
    if not common_root or not Path(common_root).is_absolute():
        fail("protocol common_epoch0_root must be an absolute path")
    return {"values": values, "artifact": artifact(path)}


def read_manifest(root: Path, allow_incomplete: bool) -> list[dict[str, str]]:
    path = root / "run_manifest.tsv"
    if not path.exists():
        if allow_incomplete:
            return []
        fail("run manifest is missing")
    lines = require_file(path, "run manifest").read_text("utf-8").splitlines()
    if not lines or lines[0] != "policy\tmode\tstatus\tarm_manifest":
        fail("run manifest header is invalid")
    rows = []
    for number, line in enumerate(lines[1:], 2):
        fields = line.split("\t")
        if len(fields) != 4:
            fail(f"invalid run manifest line {number}")
        policy, mode, status, arm_manifest = fields
        if status != "complete":
            fail(f"non-complete run manifest row {number}")
        rows.append(
            {
                "policy": policy,
                "mode": mode,
                "status": status,
                "arm_manifest": arm_manifest,
            }
        )
    observed = [(row["policy"], row["mode"]) for row in rows]
    if observed != ORDER[: len(observed)] or len(rows) > len(ORDER):
        fail(f"run manifest order={observed}, expected prefix of {ORDER}")
    if not allow_incomplete and observed != ORDER:
        fail(f"run manifest is incomplete: {observed}")
    return rows


def parse_active(value: str) -> tuple[int, ...]:
    if not value.strip():
        return ()
    try:
        return tuple(int(item.strip()) for item in value.split(","))
    except ValueError as error:
        raise VerificationError(f"invalid active ranks: {value!r}") from error


def worker_pids(lines: list[str], marker: str) -> set[int]:
    pids = set()
    for line in lines:
        if marker not in line:
            continue
        match = WORKER_PID.search(line)
        if match is None:
            fail(f"marker lacks worker PID: {marker}")
        pids.add(int(match.group(1)))
    return pids


def metric_value(line: str, name: str) -> float:
    match = re.search(rf"(?:^| - ){re.escape(name)}:([0-9.eE+-]+)(?: - |$)", line)
    if match is None:
        fail(f"metric line lacks {name}")
    return float(match.group(1))


def verify_plan(path: Path, policy: str) -> dict[str, Any]:
    plan = load_json(path, "rank plan")
    if not isinstance(plan, list) or len(plan) != STEPS:
        fail(f"{policy} plan must contain five steps")
    floors = []
    caps = []
    stages = []
    for step, row in enumerate(plan, 1):
        if not isinstance(row, dict) or row.get("step") != step:
            fail(f"{policy} plan step {step} is invalid")
        if row.get("feasible") is not True:
            fail(f"{policy} plan step {step} is infeasible")
        floors.append(row.get("selected_floor"))
        caps.append(row.get("tail_guard_response_cap"))
        stages.append(row.get("shrink_stages"))
        if floor := row.get("selected_floor"):
            if floor not in (2, 4, 8, 16):
                fail(f"{policy} step {step} has invalid floor {floor}")
        if policy in ("lengthsort_guard", "planned", "natural"):
            expected_guard = step < 5
            if row.get("tail_guard_enabled") is not expected_guard:
                fail(f"{policy} step {step} TailGuard mismatch")
    if policy == "lengthsort_guard" and floors != [16] * STEPS:
        fail(f"LengthSort+TailGuard must remain full16, got {floors}")
    if policy == "planned" and any(floor < 4 for floor in floors):
        fail(f"Planned-F4 selected an unsupported floor: {floors}")
    if policy == "natural" and any(floor < 2 for floor in floors):
        fail(f"Natural-F2 selected an unsupported floor: {floors}")
    return {"selected_floors": floors, "tail_guard_caps": caps, "stages": stages}


def response_digest(tokens: list[Any]) -> str:
    return hashlib.sha256(
        json.dumps(tokens, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def canonical_digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def verify_outputs(epoch_dir: Path) -> tuple[dict[str, Any], dict[tuple[Any, ...], Any]]:
    outputs: dict[str, Any] = {}
    requests: dict[tuple[Any, ...], Any] = {}
    for step in range(1, STEPS + 1):
        jsonl = require_file(epoch_dir / "rollout_data" / f"{step}.jsonl", "rollout JSONL")
        lengths_path = require_file(
            epoch_dir / "rollout_length" / f"length_{step}.txt", "rollout lengths"
        )
        try:
            lengths = [int(item) for item in lengths_path.read_text("utf-8").splitlines()]
        except ValueError as error:
            raise VerificationError(f"invalid length artifact at step {step}") from error
        if len(lengths) != RESPONSES_PER_STEP:
            fail(f"step {step} has {len(lengths)} lengths")
        decoded = []
        occurrences: Counter[int] = Counter()
        with jsonl.open("r", encoding="utf-8") as source:
            for row_number, line in enumerate(source):
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as error:
                    raise VerificationError(
                        f"invalid JSONL step {step} row {row_number + 1}"
                    ) from error
                occurrence = row.get("prompt_occurrence_ordinal")
                sample = row.get("rollout_sample_index")
                length = row.get("decoded_response_length")
                responses = row.get("responses")
                mask = row.get("response_mask")
                prompt_hash = row.get("rollout_prompt_hash")
                request_seed = row.get("rollout_request_seed")
                if not isinstance(sample, int) or sample < 0:
                    fail(f"step {step} row {row_number + 1} lacks sample identity")
                if not isinstance(responses, list) or not isinstance(mask, list):
                    fail(f"step {step} row {row_number + 1} lacks response arrays")
                if length is None:
                    length = sum(mask)
                if not isinstance(length, int) or length <= 0:
                    fail(f"step {step} row {row_number + 1} has invalid length")
                if len(responses) != len(mask) or sum(mask) != length:
                    fail(f"step {step} row {row_number + 1} has invalid response mask")
                if mask[:length] != [1] * length or any(mask[length:]):
                    fail(f"step {step} row {row_number + 1} mask is not a prefix")
                if isinstance(occurrence, int) and occurrence >= 0:
                    key = ("occurrence", occurrence, sample)
                    occurrence_identity: Any = occurrence
                else:
                    if not isinstance(prompt_hash, str) or not prompt_hash:
                        fail(
                            f"step {step} row {row_number + 1} lacks occurrence and prompt hash"
                        )
                    if not isinstance(request_seed, int):
                        fail(
                            f"step {step} row {row_number + 1} lacks occurrence and request seed"
                        )
                    key = ("hash", prompt_hash, sample, request_seed)
                    occurrence_identity = prompt_hash
                if key in requests:
                    fail(f"duplicate global request identity {key}")
                identity = {
                    "step": step,
                    "prompt_hash": prompt_hash,
                    "request_seed": request_seed,
                    "input": canonical_digest(row.get("input")),
                    "gts": canonical_digest(row.get("gts")),
                }
                if identity["prompt_hash"] is None or identity["request_seed"] is None:
                    fail(f"step {step} row {row_number + 1} lacks paired seed fields")
                requests[key] = {
                    "identity": identity,
                    "length": length,
                    "tokens": response_digest(responses[:length]),
                    "text": canonical_digest(row.get("output")),
                    "score": row.get("score"),
                }
                decoded.append(length)
                occurrences[occurrence_identity] += 1
        if len(decoded) != RESPONSES_PER_STEP or decoded != lengths:
            fail(f"step {step} JSONL and length artifact differ")
        if len(occurrences) != 32 or set(occurrences.values()) != {16}:
            fail(f"step {step} occurrence coverage is invalid")
        outputs[str(step)] = {
            "responses": len(decoded),
            "generated_tokens": sum(decoded),
            "min_tokens": min(decoded),
            "max_tokens": max(decoded),
            "jsonl": artifact(jsonl),
            "lengths": artifact(lengths_path),
        }
    if len(requests) != STEPS * RESPONSES_PER_STEP:
        fail(f"arm has {len(requests)} stable requests")
    return outputs, requests


def verify_log(
    log_path: Path,
    mode: str,
    policy: str,
    plan: dict[str, Any] | None,
) -> dict[str, Any]:
    raw = require_file(log_path, "primary log").read_text("utf-8", errors="replace")
    text = ANSI_ESCAPE.sub("", raw)
    lines = text.splitlines()
    fault_counts = {name: len(pattern.findall(text)) for name, pattern in FORBIDDEN.items()}
    if any(fault_counts.values()):
        fail(f"runtime faults in {log_path}: {fault_counts}")
    if re.search(r"preempting request|request preempted", text, re.IGNORECASE):
        fail(f"KV preemption found in {log_path}")
    if text.count("rollout_output_time_s:") != STEPS:
        fail("rollout timing count is not five")
    if text.count("response/aborted_ratio:0.0") != STEPS:
        fail("zero-abort metric count is not five")
    for marker in ("Training Progress: 100%", "After trainer.fit"):
        if marker not in text:
            fail(f"missing completion marker {marker!r}")
    if "moe_shared_expert_overlap: True" in text:
        fail("runtime enabled invalid shared-expert overlap")
    if "moe_shared_expert_overlap: False" not in text:
        fail("runtime did not prove shared-expert overlap false")
    task_queue = TASK_QUEUE_ENABLE[mode]
    task_queue_pids = worker_pids(
        lines, f"'TASK_QUEUE_ENABLE': '{task_queue}'"
    )
    if len(task_queue_pids) != WORLD_SIZE:
        fail(
            f"{mode} TASK_QUEUE_ENABLE={task_queue} covers "
            f"only {len(task_queue_pids)} workers"
        )

    rollout_times = [
        float(match.group(1))
        for match in re.finditer(r"rollout_output_time_s:\s*([0-9.]+)", text)
    ]
    metric_lines = [line for line in lines if re.search(r"\bstep:\d+ - ", line)]
    if len(metric_lines) != STEPS:
        fail(f"found {len(metric_lines)} trainer metric lines")
    metrics = []
    for step, (line, rollout_time) in enumerate(zip(metric_lines, rollout_times), 1):
        if f"step:{step} - " not in line:
            fail("trainer metric steps are out of order")
        metrics.append(
            {
                "step": step,
                "rollout_seconds": rollout_time,
                "generation_seconds": metric_value(line, "timing_s/gen"),
                "step_seconds": metric_value(line, "timing_s/step"),
                "reward_mean": metric_value(line, "critic/rewards/mean"),
                "response_mean": metric_value(line, "response_length/mean"),
                "max_memory_allocated_gib": metric_value(
                    line, "perf/max_memory_allocated_gb"
                ),
                "max_memory_reserved_gib": metric_value(
                    line, "perf/max_memory_reserved_gb"
                ),
            }
        )

    kv_by_step = []
    if policy in ("planned", "natural"):
        if plan is None:
            fail(f"{policy} is missing a rank plan")
        resize_rows: dict[int, list[tuple[int, int, int]]] = {
            step: [] for step in range(1, STEPS + 1)
        }
        for match in RESIZE.finditer(text):
            resize_rows[int(match.group("step"))].append(
                (
                    int(match.group("rank")),
                    int(match.group("floor")),
                    int(match.group("kv")),
                )
            )
        for step in range(1, STEPS + 1):
            rows = resize_rows[step]
            if len(rows) != WORLD_SIZE or {rank for rank, _, _ in rows} != set(range(16)):
                fail(f"step {step} KV resize does not cover all ranks")
            floors = {floor for _, floor, _ in rows}
            kvs = {kv for _, _, kv in rows}
            if floors != {plan["selected_floors"][step - 1]} or len(kvs) != 1:
                fail(f"step {step} runtime floor/KV disagrees with plan")
            kv = next(iter(kvs))
            if next(iter(floors)) == 16 and kv != FULL16_KV_TOKENS[mode]:
                fail(
                    f"step {step} full16 KV={kv}, expected {FULL16_KV_TOKENS[mode]}"
                )
            kv_by_step.append(kv)
    else:
        kv_by_step = [FULL16_KV_TOKENS[mode]] * STEPS

    step_bounds = []
    for step in range(1, STEPS + 1):
        starts = [i for i, line in enumerate(lines) if f"driver_generate_start step={step}" in line]
        dones = [i for i, line in enumerate(lines) if f"driver_generate_done step={step}" in line]
        if len(starts) != 1 or len(dones) != 1 or starts[0] >= dones[0]:
            fail(f"step {step} driver generation bounds are invalid")
        step_bounds.append((starts[0], dones[0]))
    executed_floors = []
    transition_active_sets = []
    for start, done in step_bounds:
        events = [SHRINK.search(line) for line in lines[start:done]]
        active = [parse_active(match.group("active")) for match in events if match]
        unique_active = []
        for value in active:
            if value not in unique_active:
                unique_active.append(value)
        executed_floors.append(min((len(value) for value in unique_active), default=16))
        transition_active_sets.append([list(value) for value in unique_active])

    graph_summary: dict[str, Any]
    if mode == "graph":
        full_decode = worker_pids(lines, "FULL_DECODE_ONLY compilation enabled on NPU")
        native = worker_pids(lines, "Native FULL_DECODE_ONLY ACLGraph enabled")
        attention = worker_pids(
            lines,
            "FULL_DECODE_ONLY ACLGraph captured Attention KV write and paged read",
        )
        if len(full_decode) != WORLD_SIZE or len(native) != WORLD_SIZE:
            fail("native FULL_DECODE_ONLY markers do not cover all 16 workers")
        if len(attention) != WORLD_SIZE:
            fail("FULL_DECODE_ONLY Attention markers do not cover all 16 workers")
        if "enforce_eager=True" in text or "'enforce_eager': True" in text:
            fail("graph arm initialized an eager rollout engine")
        size_by_pid = {}
        for line in lines:
            if "non-default args:" not in line:
                continue
            pid = WORKER_PID.search(line)
            sizes = CAPTURE_SIZES_RE.search(line)
            if pid and sizes:
                size_by_pid[int(pid.group(1))] = [
                    int(item) for item in sizes.group("sizes").split(",")
                ]
        if len(size_by_pid) != WORLD_SIZE or {tuple(v) for v in size_by_pid.values()} != {
            tuple(CAPTURE_SIZES)
        }:
            fail("graph capture-size contract is incomplete")
        captures = [
            {"seconds": int(match.group("seconds")), "gib": float(match.group("gib"))}
            for match in CAPTURE.finditer(text)
        ]
        if len(captures) < WORLD_SIZE:
            fail(f"only {len(captures)} graph captures completed")
        if len(worker_pids(lines, "Elastic full-MoE ACLGraph replay:")) != WORLD_SIZE:
            fail("full-MoE graph replay did not cover all workers")
        topology_events: dict[tuple[int, ...], dict[str, set[int]]] = {}
        for match in TOPOLOGY_CAPTURE.finditer(text):
            active = parse_active(match.group("active"))
            topology_events.setdefault(active, {"starting": set(), "finished": set()})[
                match.group("phase")
            ].add(int(match.group("rank")))
        expected_topologies = {
            tuple(active)
            for step_sets in transition_active_sets
            for active in step_sets
            if len(active) < WORLD_SIZE
        }
        for active in expected_topologies:
            events = topology_events.get(active)
            if events is None or events["starting"] != set(active) or events["finished"] != set(active):
                fail(f"topology graph capture is incomplete for {active}")
        graph_summary = {
            "enabled": True,
            "capture_sizes": CAPTURE_SIZES,
            "capture_events": len(captures),
            "max_capture_gib_per_rank": max(row["gib"] for row in captures),
            "max_capture_seconds": max(row["seconds"] for row in captures),
            "topologies_captured": [list(value) for value in sorted(expected_topologies, key=len, reverse=True)],
            "worker_count": WORLD_SIZE,
        }
    else:
        eager_pids = set()
        for line in lines:
            if "non-default args:" in line and "'enforce_eager': True" in line:
                match = WORKER_PID.search(line)
                if match:
                    eager_pids.add(int(match.group(1)))
        if len(eager_pids) != WORLD_SIZE:
            fail(f"eager mode covers only {len(eager_pids)} workers")
        forbidden_markers = (
            "FULL_DECODE_ONLY compilation enabled on NPU",
            "Graph capturing finished",
            "Replaying aclgraph",
            "Elastic full-MoE ACLGraph replay",
        )
        present = [marker for marker in forbidden_markers if marker in text]
        if present:
            fail(f"eager arm contains graph runtime markers: {present}")
        graph_summary = {"enabled": False, "worker_count": WORLD_SIZE}

    return {
        "metrics": metrics,
        "rollout_seconds_total": sum(row["rollout_seconds"] for row in metrics),
        "step_seconds_total": sum(row["step_seconds"] for row in metrics),
        "kv_tokens_by_step": kv_by_step,
        "selected_floors": plan["selected_floors"] if plan else [16] * STEPS,
        "executed_floors": executed_floors,
        "transition_active_sets": transition_active_sets,
        "graph": graph_summary,
        "task_queue_enable": task_queue,
        "fault_counts": fault_counts,
        "preemptions": 0,
        "aborted_responses": 0,
        "known_nonfatal_noise": {
            "torch_dynamo_metrics": text.count(
                "Unexpected exception logging compilation metrics"
            ),
            "post_success_tbe_cleanup": text.count("main process disappeared"),
        },
    }


def verify_arm(root: Path, row: dict[str, str]) -> dict[str, Any]:
    manifest_path = require_file(Path(row["arm_manifest"]), "arm manifest")
    manifest = load_json(manifest_path, "arm manifest")
    policy, mode = row["policy"], row["mode"]
    expected_manifest = {
        "schema_version": 2,
        "experiment": "qwen3_aclgraph_full_epoch_matrix",
        "status": "PASS",
        "policy": policy,
        "mode": mode,
        "seed": SEED,
        "actor_frozen": True,
        "steps": STEPS,
        "responses_per_step": RESPONSES_PER_STEP,
        "full16_kv_tokens": FULL16_KV_TOKENS[mode],
        "task_queue_enable": TASK_QUEUE_ENABLE[mode],
        "graph_attention": mode == "graph",
        "graph_moe": mode == "graph",
        "graph_mode": "FULL_DECODE_ONLY" if mode == "graph" else "NONE",
        "graph_capture_sizes": CAPTURE_SIZES if mode == "graph" else [],
    }
    if not isinstance(manifest, dict):
        fail("arm manifest must be an object")
    for key, value in expected_manifest.items():
        if manifest.get(key) != value:
            fail(f"{policy}/{mode} manifest {key}={manifest.get(key)!r}, expected {value!r}")
    run_dir = Path(manifest["run_dir"])
    epoch_dir = Path(manifest["epoch_dir"])
    common_epoch0_root = Path(manifest["common_epoch0_root"])
    protocol_common_root = Path(parse_env(root / "protocol.env")["common_epoch0_root"])
    if common_epoch0_root.resolve() != protocol_common_root.resolve():
        fail(f"common epoch0 mismatch for {policy}/{mode}")
    if not run_dir.is_dir() or epoch_dir.parent != run_dir:
        fail(f"invalid run/epoch relationship for {policy}/{mode}")
    plan_path = Path(manifest["plan_file"]) if manifest.get("plan_file") else None
    if plan_path is not None:
        if plan_path.parent.parent != epoch_dir or sha256(plan_path) != manifest["plan_sha256"]:
            fail(f"plan provenance mismatch for {policy}/{mode}")
    elif policy in ("planned", "natural"):
        fail(f"{policy}/{mode} is missing a rank plan")
    cleanup = require_file(
        run_dir / "CHECKPOINTS_REMOVED_AFTER_VALIDATION.txt", "cleanup record"
    )
    if any(run_dir.glob("epoch_*/checkpoints")):
        fail(f"validated checkpoints remain for {policy}/{mode}")
    plan = verify_plan(plan_path, policy) if plan_path is not None else None
    log_path = only_file(epoch_dir / "logs", "*.txt", "primary log")
    outputs, requests = verify_outputs(epoch_dir)
    runtime = verify_log(log_path, mode, policy, plan)
    generated = sum(row["generated_tokens"] for row in outputs.values())
    runtime["generated_tokens_total"] = generated
    runtime["response_tokens_per_second"] = generated / runtime["rollout_seconds_total"]
    compilation = None
    if mode == "graph":
        from verify_adafloor_aclgraph_smoke import _verify_compilation_graphs

        compilation = _verify_compilation_graphs(
            root,
            log_path,
            True,
            attention_capture=True,
            native_full_decode=True,
        )
    return {
        "policy": policy,
        "mode": mode,
        "manifest": artifact(manifest_path),
        "run_dir": str(run_dir),
        "epoch_dir": str(epoch_dir),
        "plan": plan,
        "plan_artifact": artifact(plan_path) if plan_path is not None else None,
        "runtime": runtime,
        "outputs": outputs,
        "requests": requests,
        "compilation_graphs": compilation,
        "artifacts": {"log": artifact(log_path), "cleanup": artifact(cleanup)},
    }


def compare_pair(eager: dict[str, Any], graph: dict[str, Any]) -> dict[str, Any]:
    eager_plan = eager["plan_artifact"]
    graph_plan = graph["plan_artifact"]
    if (eager_plan is None) != (graph_plan is None):
        fail(f"{eager['policy']} graph/eager plan presence differs")
    if eager_plan is not None and eager_plan["sha256"] != graph_plan["sha256"]:
        fail(f"{eager['policy']} graph/eager plans differ")
    eager_requests = eager["requests"]
    graph_requests = graph["requests"]
    if set(eager_requests) != set(graph_requests):
        fail(f"{eager['policy']} graph/eager stable request sets differ")
    counts = Counter()
    for key in eager_requests:
        left, right = eager_requests[key], graph_requests[key]
        counts["requests"] += 1
        if left["identity"] != right["identity"]:
            fail(f"{eager['policy']} identity mismatch at {key}")
        counts["same_length"] += left["length"] == right["length"]
        counts["same_tokens"] += left["tokens"] == right["tokens"]
        counts["same_text"] += left["text"] == right["text"]
        counts["same_score"] += left["score"] == right["score"]
    if counts["same_length"] != counts["requests"]:
        fail(f"{eager['policy']} graph/eager response lengths differ")
    if counts["same_tokens"] != counts["requests"]:
        fail(f"{eager['policy']} graph/eager response tokens differ")
    if counts["same_text"] != counts["requests"]:
        fail(f"{eager['policy']} graph/eager decoded text differs")
    eager_rt, graph_rt = eager["runtime"], graph["runtime"]
    eager_tps = eager_rt["response_tokens_per_second"]
    graph_tps = graph_rt["response_tokens_per_second"]
    eager_time = eager_rt["rollout_seconds_total"]
    graph_time = graph_rt["rollout_seconds_total"]
    return {
        "policy": eager["policy"],
        "requests": counts["requests"],
        "identity_matches": counts["requests"],
        "same_length": counts["same_length"],
        "same_tokens": counts["same_tokens"],
        "same_text": counts["same_text"],
        "same_score": counts["same_score"],
        "eager_generated_tokens": eager_rt["generated_tokens_total"],
        "graph_generated_tokens": graph_rt["generated_tokens_total"],
        "eager_rollout_seconds": eager_time,
        "graph_rollout_seconds": graph_time,
        "rollout_time_delta_percent": (graph_time / eager_time - 1.0) * 100.0,
        "eager_response_tokens_per_second": eager_tps,
        "graph_response_tokens_per_second": graph_tps,
        "throughput_delta_percent": (graph_tps / eager_tps - 1.0) * 100.0,
        "work_difference_percent": (
            graph_rt["generated_tokens_total"] / eager_rt["generated_tokens_total"] - 1.0
        )
        * 100.0,
    }


def public_arm(arm: dict[str, Any]) -> dict[str, Any]:
    result = dict(arm)
    result.pop("requests", None)
    return result


def verify_common_epoch0(
    root: Path,
    protocol: dict[str, Any],
    allow_incomplete: bool,
) -> dict[str, Any] | None:
    common_root = Path(protocol["values"]["common_epoch0_root"])
    marker = common_root / "DO_NOT_DELETE_COMMON_EPOCH0_CHECKPOINT"
    if not marker.exists():
        if common_root.exists():
            fail(f"common epoch0 exists without completion marker: {common_root}")
        if allow_incomplete:
            return None
        fail(f"graph Vanilla common epoch0 is missing: {common_root}")
    require_file(marker, "common epoch0 completion marker")
    reuse = require_file(common_root / "reuse.env", "common epoch0 reuse contract")
    metadata = require_file(
        common_root / "common_epoch0_metadata.env", "common epoch0 metadata"
    )
    epoch_dir = common_root / "epoch_000_mode0_probe"
    if not epoch_dir.is_dir():
        fail(f"common epoch0 directory is missing: {epoch_dir}")
    reuse_text = reuse.read_text("utf-8")
    checkpoint_match = re.search(
        r"^export BASELINE_INITIAL_RESUME_CKPT=(.+)$", reuse_text, re.MULTILINE
    )
    if checkpoint_match is None:
        fail("common epoch0 reuse contract lacks checkpoint")
    checkpoint = Path(checkpoint_match.group(1).strip().strip("'\""))
    if not checkpoint.is_dir() or not (checkpoint / ".PRESERVE_COMMON_EPOCH0").is_file():
        fail(f"common epoch0 checkpoint is not preserved: {checkpoint}")
    if not (checkpoint / "actor" / "dist_ckpt").is_dir():
        fail("common epoch0 actor dist checkpoint is missing")
    log_path = only_file(epoch_dir / "logs", "*.txt", "common epoch0 log")
    outputs, _ = verify_outputs(epoch_dir)
    runtime = verify_log(log_path, "graph", "vanilla", None)
    generated = sum(row["generated_tokens"] for row in outputs.values())
    runtime["generated_tokens_total"] = generated
    runtime["response_tokens_per_second"] = generated / runtime["rollout_seconds_total"]
    return {
        "status": "PASS",
        "root": str(common_root.resolve()),
        "epoch_dir": str(epoch_dir.resolve()),
        "checkpoint": str(checkpoint.resolve()),
        "outputs": outputs,
        "runtime": runtime,
        "artifacts": {
            "completion": artifact(marker),
            "reuse": artifact(reuse),
            "metadata": artifact(metadata),
            "log": artifact(log_path),
        },
    }


def verify(root: Path, allow_incomplete: bool) -> dict[str, Any]:
    root = root.resolve()
    protocol = verify_protocol(root)
    code = verify_code_contract(root)
    common_epoch0 = verify_common_epoch0(root, protocol, allow_incomplete)
    rows = read_manifest(root, allow_incomplete)
    if rows and common_epoch0 is None:
        fail("epoch1 arms exist without a verified common graph epoch0")
    arms = [verify_arm(root, row) for row in rows]
    by_key = {(arm["policy"], arm["mode"]): arm for arm in arms}
    pairs = []
    for policy in POLICIES:
        eager = by_key.get((policy, "eager"))
        graph = by_key.get((policy, "graph"))
        if eager is not None and graph is not None:
            pairs.append(compare_pair(eager, graph))
    complete = common_epoch0 is not None and len(arms) == 8 and len(pairs) == 4
    if not allow_incomplete and not complete:
        fail("the graph epoch0 plus eight-arm experiment is incomplete")
    return {
        "schema_version": 2,
        "status": "PASS" if complete else "INCOMPLETE",
        "experiment": "qwen3_aclgraph_full_epoch_matrix",
        "root": str(root),
        "completed_arms": len(arms),
        "completed_pairs": len(pairs),
        "common_epoch0": common_epoch0,
        "protocol": protocol,
        "code_contract": code,
        "run_manifest": artifact(root / "run_manifest.tsv") if rows else None,
        "arms": [public_arm(arm) for arm in arms],
        "pairs": pairs,
        "kv_memory": {
            "bytes_per_token": 98304,
            "eager_full16_tokens": FULL16_KV_TOKENS["eager"],
            "graph_full16_tokens": FULL16_KV_TOKENS["graph"],
            "eager_full16_gib": FULL16_KV_TOKENS["eager"] * 98304 / 2**30,
            "graph_full16_gib": FULL16_KV_TOKENS["graph"] * 98304 / 2**30,
            "graph_saved_gib_per_rank": (
                FULL16_KV_TOKENS["eager"] - FULL16_KV_TOKENS["graph"]
            ) * 98304 / 2**30,
        },
        "claim_boundary": (
            "This matrix uses one graph-produced updated-policy checkpoint and one "
            "frozen epoch1 request seed. It establishes workload-sized graph "
            "functionality and reports descriptive graph/eager performance for four "
            "policies. It is not a multi-seed confidence interval, long-horizon "
            "training-convergence result, or multi-node result."
        ),
    }


def markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Qwen3 ACLGraph Full-Epoch Matrix",
        "",
        f"Status: **{summary['status']}**",
        "",
        f"Common graph Vanilla epoch0: {'PASS' if summary['common_epoch0'] else 'PENDING'}",
        f"Completed arms: {summary['completed_arms']}/8",
        f"Completed policy pairs: {summary['completed_pairs']}/4",
        "",
    ]
    if summary["pairs"]:
        lines.extend(
            [
                "| Policy | Eager rollout s | Graph rollout s | Time delta | Eager tok/s | Graph tok/s | Throughput delta | Exact tokens |",
                "|---|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in summary["pairs"]:
            lines.append(
                f"| {row['policy']} | {row['eager_rollout_seconds']:.3f} | "
                f"{row['graph_rollout_seconds']:.3f} | "
                f"{row['rollout_time_delta_percent']:+.2f}% | "
                f"{row['eager_response_tokens_per_second']:.3f} | "
                f"{row['graph_response_tokens_per_second']:.3f} | "
                f"{row['throughput_delta_percent']:+.2f}% | "
                f"{row['same_tokens']}/{row['requests']} |"
            )
        lines.append("")
    kv = summary["kv_memory"]
    lines.extend(
        [
            "## KV Contract",
            "",
            f"The paper-compatible eager arm uses {kv['eager_full16_tokens']:,} "
            f"full16 KV tokens ({kv['eager_full16_gib']:.2f} GiB/rank). The graph "
            f"arm uses {kv['graph_full16_tokens']:,} tokens "
            f"({kv['graph_full16_gib']:.2f} GiB/rank), reserving "
            f"{kv['graph_saved_gib_per_rank']:.2f} GiB/rank for graph state.",
            "",
            "## Claim Boundary",
            "",
            summary["claim_boundary"],
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument(
        "--json", type=Path, help="output JSON; defaults to ROOT/aclgraph_full_epoch_matrix_summary.json"
    )
    parser.add_argument(
        "--markdown", type=Path, help="output Markdown; defaults to ROOT/aclgraph_full_epoch_matrix_summary.md"
    )
    args = parser.parse_args()
    try:
        summary = verify(args.root, args.allow_incomplete)
    except VerificationError as error:
        print(f"FAIL: {error}")
        return 1
    json_path = args.json or args.root / "aclgraph_full_epoch_matrix_summary.json"
    markdown_path = args.markdown or args.root / "aclgraph_full_epoch_matrix_summary.md"
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    markdown_path.write_text(markdown(summary))
    print(f"{summary['status']}: {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
