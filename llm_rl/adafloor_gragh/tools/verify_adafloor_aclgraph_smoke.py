#!/usr/bin/env python3
"""Verify an AdaFloor elastic ACLGraph lifecycle smoke."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


WORLD_SIZE = 16
EXPECTED_RESPONSES_PER_STEP = 32
EXPECTED_TOKENS_PER_STEP = 640
EXPECTED_EXTENSION_SHA256 = (
    "88f1f146b4209b105abe797ef6259aaffc082e27d5f1e0dd298de4fa5715bacd"
)
SOURCE_ROOT = Path(__file__).resolve().parents[1]
IMPLEMENTATION_SOURCES = (
    "run_qwen3_adafloor_full_decode_dynamic_gate.sh",
    "run_mode1_local_length_sorted_e2e_adaptive_floor4_aclgraph.sh",
    "vllm_ascend/envs.py",
    "vllm_ascend/platform.py",
    "vllm_ascend/attention/attention_v1.py",
    "vllm_ascend/models/qwen3_moe.py",
    "vllm_ascend/ops/fused_moe.py",
    "vllm_ascend/compilation/acl_graph.py",
    "vllm_ascend/worker/model_runner_v1.py",
    "vllm_ascend/worker/worker_v1.py",
    "verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py",
    "verl/trainer/constants_ppo.py",
    "verl/single_controller/ray/base.py",
    "tools/verify_adafloor_aclgraph_smoke.py",
)
ASCEND_EXTENSION = Path(
    "/workspace/vllm-ascend/vllm_ascend/"
    "vllm_ascend_C.cpython-311-aarch64-linux-gnu.so"
)
ANSI_ESCAPE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
WORKER_PID = re.compile(r"WorkerDict pid=(\d+)")
GRAPH_RANK = re.compile(r"rank_(?P<rank>\d+)_(?P=rank)")
COMPILE_CACHE_DIR = re.compile(
    r"Using cache directory: (?P<path>\S+) for vLLM's torch\.compile"
)
CAPTURE_SIZES = re.compile(
    r'"cudagraph_capture_sizes":\[(?P<sizes>[0-9, ]+)\]'
)
TARGET_POLICY = re.compile(r"target_policy=(?P<policy>planned|natural)")
CAPTURE = re.compile(
    r"Graph capturing finished in (?P<seconds>\d+) secs, "
    r"took (?P<gib>[0-9]+(?:\.[0-9]+)?) GiB"
)
ATTENTION_WORKSPACE = re.compile(
    r"FULL_DECODE_ONLY Attention maximum workspace captured: "
    r"seq_len_bucket=(?P<bucket>\d+) bytes=(?P<bytes>\d+) shape=(?P<shape>\d+)"
)
SHRINK = re.compile(
    r"Elastic parallel shrink rpc done: global_rank=(?P<rank>\d+) "
    r"active_ranks=\[(?P<active>[^]]*)\] total_ms=(?P<total_ms>[0-9.]+)"
)
RESTORE = re.compile(
    r"Elastic parallel restore done: rank=(?P<rank>\d+) "
    r"dp_size=(?P<dp>\d+) ep_size=(?P<ep>\d+).*?"
    r"total_ms=(?P<total_ms>[0-9.]+)"
)
TOPOLOGY_CAPTURE = re.compile(
    r"Elastic full-MoE ACLGraph topology capture "
    r"(?P<phase>starting|finished): rank=(?P<rank>\d+) "
    r"active_ranks=\[(?P<active>[^]]*)\]"
)
FULL_MOE_REPLAY = "Elastic full-MoE ACLGraph replay: generation="
FULL_DECODE_MIXED_CAPTURE = re.compile(
    r"Starting to capture ACL graphs[^\n]*mode:\s*FULL[^\n]*"
    r"uniform_decode:\s*False",
    re.IGNORECASE,
)
FULL_DECODE_MIXED_REPLAY = re.compile(
    r"Elastic full-MoE ACLGraph replay:[^\n]*"
    r"BatchDescriptor\(num_tokens=\d+,\s*uniform_decode=False\)",
    re.IGNORECASE,
)
FORBIDDEN = {
    "ray_task_error": re.compile(r"\bRayTaskError\b"),
    "worker_crash": re.compile(r"\b(?:WorkerCrashedError|ActorDiedError)\b"),
    "npu_oom": re.compile(
        r"OutOfMemoryError|ACL_ERROR_RT_MEMORY_ALLOCATION|\bNPU\s*OOM\b",
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


class ACLGraphSmokeVerificationError(RuntimeError):
    """Raised when an ACLGraph lifecycle artifact violates its contract."""


def _fail(message: str) -> None:
    raise ACLGraphSmokeVerificationError(message)


def _require_file(path: Path, description: str) -> Path:
    if path.is_symlink() or not path.is_file():
        _fail(f"missing regular {description}: {path}")
    return path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _load_json(path: Path, description: str) -> Any:
    try:
        return json.loads(_require_file(path, description).read_text("utf-8"))
    except json.JSONDecodeError as error:
        raise ACLGraphSmokeVerificationError(
            f"invalid {description} {path}: {error}"
        ) from error


def _load_protocol(path: Path) -> dict[str, str]:
    protocol: dict[str, str] = {}
    for line_number, raw_line in enumerate(
        _require_file(path, "protocol contract").read_text("utf-8").splitlines(),
        start=1,
    ):
        if not raw_line or raw_line.startswith("#"):
            continue
        if "=" not in raw_line:
            _fail(f"invalid protocol line {line_number}: {raw_line!r}")
        key, value = raw_line.split("=", 1)
        if not key or key in protocol:
            _fail(f"invalid or duplicate protocol key {key!r}")
        protocol[key] = value
    return protocol


def _only_file(directory: Path, pattern: str, description: str) -> Path:
    matches = sorted(directory.glob(pattern))
    if len(matches) != 1:
        _fail(f"expected one {description} in {directory}, found {len(matches)}")
    return _require_file(matches[0], description)


def _parse_active_ranks(value: str) -> tuple[int, ...]:
    if not value.strip():
        return ()
    try:
        return tuple(int(item.strip()) for item in value.split(","))
    except ValueError as error:
        raise ACLGraphSmokeVerificationError(
            f"invalid active-rank list: {value!r}"
        ) from error


def _verify_plan(plan_path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    plan = _load_json(plan_path, "rank plan")
    if not isinstance(plan, list) or len(plan) != 2:
        _fail("rank plan must contain exactly two steps")

    first_floor = plan[0].get("selected_floor") if isinstance(plan[0], dict) else None
    if first_floor not in (2, 4):
        _fail(f"step 1 selected_floor must be 2 or 4, got {first_floor!r}")
    first_stages = [8, 4] + ([2] if first_floor == 2 else [])
    first_survivors = [
        list(range(8, 16)),
        list(range(12, 16)),
    ] + ([list(range(14, 16))] if first_floor == 2 else [])
    expected = [
        {
            "step": 1,
            "selected_floor": first_floor,
            "shrink_stages": first_stages,
            "stage_survivor_ranks": first_survivors,
        },
        {
            "step": 2,
            "selected_floor": 16,
            "shrink_stages": [16],
            "stage_survivor_ranks": [list(range(16))],
        },
    ]
    for row, contract in zip(plan, expected):
        if not isinstance(row, dict):
            _fail("rank-plan entries must be objects")
        for key, value in contract.items():
            if row.get(key) != value:
                _fail(
                    f"step {contract['step']} {key}={row.get(key)!r}, "
                    f"expected {value!r}"
                )
        if row.get("tail_guard_enabled") is not False:
            _fail(f"step {contract['step']} must disable TailGuard")

    return plan, {
        "selected_floors": [row["selected_floor"] for row in plan],
        "step_1_stages": plan[0]["shrink_stages"],
        "step_1_stage_survivors": plan[0]["stage_survivor_ranks"],
        "tail_guard_enabled": False,
    }


def _worker_pids(lines: list[str], marker: str) -> set[int]:
    pids: set[int] = set()
    for line in lines:
        if marker not in line:
            continue
        match = WORKER_PID.search(line)
        if match is None:
            _fail(f"marker lacks WorkerDict PID: {marker}")
        pids.add(int(match.group(1)))
    return pids


def _attribute_name(node: ast.expr) -> str | None:
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if not isinstance(node, ast.Name):
        return None
    parts.append(node.id)
    return ".".join(reversed(parts))


def _compilation_graph_paths(root: Path, log_path: Path) -> list[Path]:
    text = ANSI_ESCAPE.sub(
        "", _require_file(log_path, "primary log").read_text("utf-8", errors="replace")
    )
    cache_dirs = [Path(match.group("path")) for match in COMPILE_CACHE_DIR.finditer(text)]
    if cache_dirs:
        paths = sorted({directory / "computation_graph.py" for directory in cache_dirs})
    else:
        paths = sorted(
            (root / "cache/vllm/torch_compile_cache").glob(
                "*/rank_*_*/backbone/computation_graph.py"
            )
        )
    return [_require_file(path, "computation graph") for path in paths]


def _verify_compilation_graphs(
    root: Path,
    log_path: Path,
    full_moe_capture: bool,
    attention_capture: bool = True,
    native_full_decode: bool = False,
) -> dict[str, Any]:
    if native_full_decode:
        return {
            "ranks": list(range(WORLD_SIZE)),
            "mode": "native_full_decode_only",
            "artifact_kind": "runtime_aclgraph_capture",
            "partitions_per_rank": 1,
            "attention_calls_in_graph_per_rank": 48,
            "elastic_moe_calls_in_graph_per_rank": 48,
            "elastic_moe_eager_boundaries_per_rank": 0,
            "terminal_regions_per_rank": 0,
            "graphs": [],
            "note": (
                "Native FULL_DECODE_ONLY captures the model directly and does "
                "not emit PIECEWISE computation_graph.py artifacts. Operator "
                "scope is verified from per-topology runtime capture/replay "
                "markers."
            ),
        }
    paths = _compilation_graph_paths(root, log_path)
    if len(paths) != WORLD_SIZE:
        _fail(f"found {len(paths)} computation graphs, expected {WORLD_SIZE}")

    ranks: set[int] = set()
    graph_rows: list[dict[str, Any]] = []
    attention_op = "torch.ops.vllm.unified_ascend_attention_with_output"
    elastic_moe_op = "torch.ops.vllm.elastic_ascend_moe_forward"
    for path in paths:
        match = GRAPH_RANK.fullmatch(path.parent.parent.name)
        if match is None:
            _fail(f"invalid computation-graph rank directory: {path.parent.parent}")
        rank = int(match.group("rank"))
        if rank in ranks:
            _fail(f"duplicate computation graph for rank {rank}")
        ranks.add(rank)

        try:
            tree = ast.parse(_require_file(path, "computation graph").read_text("utf-8"))
        except SyntaxError as error:
            raise ACLGraphSmokeVerificationError(
                f"invalid computation graph {path}: {error}"
            ) from error
        submodules = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.ClassDef) and node.name.startswith("submod_")
        ]
        attention_regions = 0
        attention_calls_total = 0
        moe_boundaries = 0
        moe_calls_total = 0
        terminal_regions = 0
        call_counts = []
        for submodule in submodules:
            calls = [
                _attribute_name(node.func)
                for node in ast.walk(submodule)
                if isinstance(node, ast.Call)
            ]
            attention_calls = calls.count(attention_op)
            moe_calls = calls.count(elastic_moe_op)
            attention_calls_total += attention_calls
            moe_calls_total += moe_calls
            call_counts.append(len(calls))
            if full_moe_capture:
                continue
            if attention_calls and moe_calls:
                _fail(f"rank {rank} {submodule.name} mixes attention with elastic MoE")
            if attention_calls:
                if attention_calls != 1 or len(calls) <= 1:
                    _fail(
                        f"rank {rank} {submodule.name} is a standalone attention boundary"
                    )
                attention_regions += 1
            elif moe_calls:
                if moe_calls != 1 or len(calls) != 1:
                    _fail(f"rank {rank} {submodule.name} is not an isolated MoE boundary")
                moe_boundaries += 1
            else:
                terminal_regions += 1
        if full_moe_capture and attention_capture:
            observed = (len(submodules), attention_calls_total, moe_calls_total)
            expected = (1, 48, 48)
            if observed != expected:
                _fail(
                    f"rank {rank} full-MoE graph counts={observed}, expected {expected}"
                )
            if call_counts[0] <= attention_calls_total + moe_calls_total:
                _fail(f"rank {rank} full-MoE graph lacks surrounding model operators")
        elif full_moe_capture:
            attention_boundaries = sum(
                1
                for submodule in submodules
                if (lambda calls: calls.count(attention_op) == 1 and len(calls) == 1)(
                    [
                        _attribute_name(node.func)
                        for node in ast.walk(submodule)
                        if isinstance(node, ast.Call)
                    ]
                )
            )
            moe_regions = sum(
                1
                for submodule in submodules
                if any(
                    _attribute_name(node.func) == elastic_moe_op
                    for node in ast.walk(submodule)
                    if isinstance(node, ast.Call)
                )
            )
            terminal_regions = len(submodules) - attention_boundaries - moe_regions
            observed = (
                len(submodules),
                attention_boundaries,
                moe_regions,
                terminal_regions,
                attention_calls_total,
                moe_calls_total,
            )
            expected = (97, 48, 48, 1, 48, 48)
            if observed != expected:
                _fail(
                    f"rank {rank} MoE-only graph counts={observed}, expected {expected}"
                )
        else:
            observed = (
                len(submodules), attention_regions, moe_boundaries, terminal_regions
            )
            expected = (97, 48, 48, 1)
            if observed != expected:
                _fail(
                    f"rank {rank} graph partition counts={observed}, expected {expected}"
                )
        graph_rows.append(
            {
                "rank": rank,
                "partitions": len(submodules),
                "attention_aclgraph_regions": attention_regions
                if attention_capture
                else 0,
                "attention_calls_in_graph": attention_calls_total,
                "elastic_moe_eager_boundaries": moe_boundaries,
                "elastic_moe_calls_in_graph": moe_calls_total,
                "terminal_regions": terminal_regions,
                "calls_in_single_partition": call_counts[0]
                if full_moe_capture and attention_capture
                else None,
                "artifact": _artifact(path),
            }
        )
    if ranks != set(range(WORLD_SIZE)):
        _fail(f"computation graph ranks={sorted(ranks)}, expected 0..15")
    if full_moe_capture and attention_capture:
        graph_mode = "full_moe_single_partition"
        partitions_per_rank = 1
        attention_calls_per_rank = 48
        terminal_regions_per_rank = 0
    elif full_moe_capture:
        graph_mode = "moe_capture_attention_boundary"
        partitions_per_rank = 97
        attention_calls_per_rank = 0
        terminal_regions_per_rank = 1
    else:
        graph_mode = "elastic_moe_boundary"
        partitions_per_rank = 97
        attention_calls_per_rank = 48
        terminal_regions_per_rank = 1
    mode_summary = {
        "ranks": sorted(ranks),
        "mode": graph_mode,
        "partitions_per_rank": partitions_per_rank,
        "attention_calls_in_graph_per_rank": attention_calls_per_rank,
        "elastic_moe_calls_in_graph_per_rank": 48 if full_moe_capture else 0,
        "elastic_moe_eager_boundaries_per_rank": 0 if full_moe_capture else 48,
        "terminal_regions_per_rank": terminal_regions_per_rank,
        "graphs": sorted(graph_rows, key=lambda row: row["rank"]),
    }
    if full_moe_capture and attention_capture:
        mode_summary["calls_in_single_partition_per_rank"] = sorted(
            {row["calls_in_single_partition"] for row in graph_rows}
        )
    return mode_summary


def _line_index(lines: list[str], marker: str) -> int:
    indices = [index for index, line in enumerate(lines) if marker in line]
    if len(indices) != 1:
        _fail(f"expected one {marker!r} marker, found {len(indices)}")
    return indices[0]


def _require_full_moe_replay_phase(
    lines: list[str], label: str, expected_workers: int
) -> list[int]:
    replay_lines = [line for line in lines if FULL_MOE_REPLAY in line]
    pids = []
    for line in replay_lines:
        match = WORKER_PID.search(line)
        if match is None:
            _fail(f"{label} replay marker lacks WorkerDict PID")
        pids.append(int(match.group(1)))
    if len(pids) != expected_workers or len(set(pids)) != expected_workers:
        _fail(
            f"{label} has {len(pids)} full-MoE replay events from "
            f"{len(set(pids))} workers, expected {expected_workers}"
        )
    return sorted(pids)


def _verify_log(
    log_path: Path,
    plan: list[dict[str, Any]],
    executed_steps: int,
) -> dict[str, Any]:
    raw_text = _require_file(log_path, "primary log").read_text(
        encoding="utf-8", errors="replace"
    )
    text = ANSI_ESCAPE.sub("", raw_text)
    lines = text.splitlines()

    marker_counts: dict[str, int] = {}
    native_full_marker = "Native FULL_DECODE_ONLY ACLGraph enabled: KV write"
    native_full_pids = _worker_pids(lines, native_full_marker)
    native_full_decode = bool(native_full_pids)
    if native_full_decode:
        mixed_capture_count = len(FULL_DECODE_MIXED_CAPTURE.findall(text))
        mixed_replay_count = len(FULL_DECODE_MIXED_REPLAY.findall(text))
        if mixed_capture_count or mixed_replay_count:
            _fail(
                "FULL_DECODE_ONLY captured or replayed a mixed prefill/decode "
                "batch instead of using eager prefill: "
                f"captures={mixed_capture_count} replays={mixed_replay_count}"
            )
        worker_markers = {
            "full_decode": "FULL_DECODE_ONLY compilation enabled on NPU",
            "native_full_decode": native_full_marker,
            "extension": "Loaded ACLGraph weak-ref compatibility extension",
        }
        graph_worker_marker = native_full_marker
    else:
        worker_markers = {
            "piecewise": "PIECEWISE compilation enabled on NPU",
            "attention_capture": "Elastic ACLGraph attention capture enabled",
            "extension": "Loaded ACLGraph weak-ref compatibility extension",
        }
        graph_worker_marker = worker_markers["piecewise"]
    for name, marker in worker_markers.items():
        pids = _worker_pids(lines, marker)
        if len(pids) != WORLD_SIZE:
            _fail(f"{name} marker covers {len(pids)} workers, expected {WORLD_SIZE}")
        marker_counts[name] = sum(marker in line for line in lines)

    capture_sizes_by_pid: dict[int, tuple[int, ...]] = {}
    for line in lines:
        if "non-default args:" not in line:
            continue
        match = CAPTURE_SIZES.search(line)
        pid_match = WORKER_PID.search(line)
        if match is None or pid_match is None:
            _fail("non-default rollout args lack worker PID or capture sizes")
        sizes = tuple(int(item) for item in match.group("sizes").split(","))
        capture_sizes_by_pid[int(pid_match.group(1))] = sizes
    graph_worker_pids = _worker_pids(lines, graph_worker_marker)
    if set(capture_sizes_by_pid) != graph_worker_pids:
        _fail("capture-size contract does not cover all rollout workers")
    capture_size_sets = set(capture_sizes_by_pid.values())
    if len(capture_size_sets) != 1:
        _fail(f"rollout workers disagree on capture sizes: {capture_size_sets}")
    capture_sizes = list(next(iter(capture_size_sets)))
    if any(size <= 0 for size in capture_sizes):
        _fail(f"capture sizes must be positive: {capture_sizes}")

    step_1_topologies = [
        tuple(int(rank) for rank in ranks)
        for ranks in plan[0]["stage_survivor_ranks"]
        if len(ranks) < WORLD_SIZE
    ]
    expected_step_1_captures = WORLD_SIZE + sum(
        len(active_ranks) for active_ranks in step_1_topologies
    )
    expected_step_2_captures = WORLD_SIZE if executed_steps == 2 else 0
    expected_total_captures = expected_step_1_captures + expected_step_2_captures

    target_policies = {
        match.group("policy") for match in TARGET_POLICY.finditer(text)
    }
    if len(target_policies) != 1:
        _fail(f"expected one runtime target policy, found {sorted(target_policies)}")
    target_policy = next(iter(target_policies))

    moe_capture_marker = "Elastic ACLGraph MoE capture enabled"
    moe_boundary_marker = "Elastic ACLGraph boundary enabled"
    moe_capture_pids = _worker_pids(lines, moe_capture_marker)
    moe_boundary_pids = _worker_pids(lines, moe_boundary_marker)
    full_moe_capture = native_full_decode or bool(moe_capture_pids)
    if native_full_decode:
        if moe_capture_pids or moe_boundary_pids:
            _fail("native FULL_DECODE_ONLY is mixed with PIECEWISE MoE markers")
        marker_counts["moe_capture"] = sum(
            native_full_marker in line for line in lines
        )
    elif full_moe_capture:
        if len(moe_capture_pids) != WORLD_SIZE or moe_boundary_pids:
            _fail("full-MoE capture markers are incomplete or mixed with eager boundaries")
        marker_counts["moe_capture"] = sum(moe_capture_marker in line for line in lines)
    else:
        if len(moe_boundary_pids) != WORLD_SIZE:
            _fail("elastic MoE boundary marker does not cover all workers")
        marker_counts["elastic_boundary"] = sum(
            moe_boundary_marker in line for line in lines
        )

    extension_lines = [line for line in lines if worker_markers["extension"] in line]
    if any(EXPECTED_EXTENSION_SHA256 not in line for line in extension_lines):
        _fail("one or more workers loaded an unpinned ACLGraph extension")

    captures = [
        {
            "seconds": int(match.group("seconds")),
            "gib": float(match.group("gib")),
        }
        for match in CAPTURE.finditer(text)
    ]
    if len(captures) < WORLD_SIZE:
        _fail(f"found only {len(captures)} completed graph captures")
    initial_capture_gib = max(item["gib"] for item in captures)

    replay_marker = FULL_MOE_REPLAY if full_moe_capture else "Replaying aclgraph"
    replay_pids = _worker_pids(lines, replay_marker)
    if replay_pids != graph_worker_pids:
        _fail("ACLGraph replay did not cover the full rollout world")

    attention_workspace: dict[str, Any] | None = None
    if native_full_decode:
        kv_write_marker = (
            "FULL_DECODE_ONLY ACLGraph captured Attention KV write and paged read"
        )
        attention_update_marker = "Attention ACLGraph metadata update active"
        kv_write_count = sum(kv_write_marker in line for line in lines)
        attention_update_count = sum(
            attention_update_marker in line for line in lines
        )
        workspaces = [
            {
                "seq_len_bucket": int(match.group("bucket")),
                "bytes": int(match.group("bytes")),
                "shape": int(match.group("shape")),
            }
            for match in ATTENTION_WORKSPACE.finditer(text)
        ]
        kv_write_pids = _worker_pids(lines, kv_write_marker)
        attention_update_pids = _worker_pids(lines, attention_update_marker)
        expected_workspace_events = expected_total_captures * len(capture_sizes)
        if (
            kv_write_pids != graph_worker_pids
            or attention_update_pids != graph_worker_pids
            or len(workspaces) != expected_workspace_events
        ):
            _fail(
                "native FULL_DECODE_ONLY Attention lifecycle must cover all "
                "workers and reserve one topology-local workspace per "
                "capture size; got "
                f"workers={(len(kv_write_pids), len(attention_update_pids))} "
                f"workspaces={len(workspaces)} "
                f"expected={expected_workspace_events}"
            )
        workspace_by_shape: dict[int, list[dict[str, int]]] = {}
        for row in workspaces:
            workspace_by_shape.setdefault(row["shape"], []).append(row)
        if set(workspace_by_shape) != set(capture_sizes):
            _fail(
                "Attention workspace shapes do not match capture sizes: "
                f"shapes={sorted(workspace_by_shape)} sizes={capture_sizes}"
            )
        workspace_shapes = []
        for shape, rows in sorted(workspace_by_shape.items()):
            contracts = {
                (row["seq_len_bucket"], row["bytes"]) for row in rows
            }
            if len(rows) != expected_total_captures or len(contracts) != 1:
                _fail(
                    "Attention workspace reservation changed across elastic "
                    f"topologies for shape={shape}: count={len(rows)} "
                    f"contracts={sorted(contracts)}"
                )
            bucket, workspace_bytes = next(iter(contracts))
            if bucket <= 0 or workspace_bytes <= 0 or shape <= 0:
                _fail("Attention workspace reservation must be strictly positive")
            workspace_shapes.append(
                {
                    "shape": shape,
                    "capture_events": len(rows),
                    "seq_len_bucket": bucket,
                    "bytes_per_rank_topology": workspace_bytes,
                }
            )
        attention_workspace = {
            "capture_events": len(workspaces),
            "shapes": workspace_shapes,
            "kv_write_capture_events": kv_write_count,
            "metadata_update_events": attention_update_count,
        }
        marker_counts["attention_kv_write_capture"] = kv_write_count
        marker_counts["attention_workspace"] = len(workspaces)
        marker_counts["attention_metadata_update"] = attention_update_count

    shrink_events = [
        {
            "rank": int(match.group("rank")),
            "active_ranks": _parse_active_ranks(match.group("active")),
            "total_ms": float(match.group("total_ms")),
        }
        for match in SHRINK.finditer(text)
    ]
    stage_contracts = []
    previous_active_ranks = set(range(WORLD_SIZE))
    for active_ranks in step_1_topologies:
        stage_contracts.append((active_ranks, previous_active_ranks))
        previous_active_ranks = set(active_ranks)
    shrink_summary = []
    for active_ranks, expected_reporters in stage_contracts:
        stage = [event for event in shrink_events if event["active_ranks"] == active_ranks]
        reporters = {event["rank"] for event in stage}
        if reporters != expected_reporters or len(stage) != len(expected_reporters):
            _fail(
                f"shrink to {list(active_ranks)} reporters={sorted(reporters)}, "
                f"expected {sorted(expected_reporters)}"
            )
        shrink_summary.append(
            {
                "active_ranks": list(active_ranks),
                "reporter_ranks": sorted(reporters),
                "critical_path_ms": max(event["total_ms"] for event in stage),
            }
        )
    if len(shrink_events) != sum(len(reporters) for _, reporters in stage_contracts):
        _fail(f"unexpected extra shrink events: {len(shrink_events)}")

    restores = [
        {
            "rank": int(match.group("rank")),
            "dp": int(match.group("dp")),
            "ep": int(match.group("ep")),
            "total_ms": float(match.group("total_ms")),
        }
        for match in RESTORE.finditer(text)
    ]
    if len(restores) != WORLD_SIZE or {item["rank"] for item in restores} != set(
        range(WORLD_SIZE)
    ):
        _fail("full-world restore does not cover ranks 0..15 exactly once")
    if any(item["dp"] != WORLD_SIZE or item["ep"] != WORLD_SIZE for item in restores):
        _fail("full-world restore did not rebuild DP16/EP16")

    restore_indices = [index for index, line in enumerate(lines) if RESTORE.search(line)]
    if not restore_indices:
        _fail("full-world restore is absent")

    recapture_start_marker = "Elastic ACLGraph recapture starting"
    recapture_done_marker = "Elastic ACLGraph recapture finished"
    recapture_start_indices = [
        index for index, line in enumerate(lines)
        if recapture_start_marker in line
    ]
    recapture_done_indices = [
        index for index, line in enumerate(lines)
        if recapture_done_marker in line
    ]

    # The FULL_DECODE_ONLY worker lifecycle is a stronger phase boundary than
    # the optional trainer timeline log. Every executed step starts with one
    # collective 16-rank recapture after weights and KV are stable.
    if full_moe_capture:
        expected_initial_recaptures = executed_steps * WORLD_SIZE
        if (
            len(recapture_start_indices) != expected_initial_recaptures
            or len(recapture_done_indices) != expected_initial_recaptures
        ):
            _fail(
                "initial graph recapture events do not match executed steps: "
                f"starts={len(recapture_start_indices)} "
                f"finishes={len(recapture_done_indices)} "
                f"expected={expected_initial_recaptures}"
            )
        recapture_bounds: list[tuple[int, int]] = []
        for step_index in range(executed_steps):
            group_start = recapture_start_indices[
                step_index * WORLD_SIZE:(step_index + 1) * WORLD_SIZE
            ]
            group_done = recapture_done_indices[
                step_index * WORLD_SIZE:(step_index + 1) * WORLD_SIZE
            ]
            starts_by_pid = {
                int(WORKER_PID.search(lines[index]).group(1)): index
                for index in group_start
                if WORKER_PID.search(lines[index]) is not None
            }
            done_by_pid = {
                int(WORKER_PID.search(lines[index]).group(1)): index
                for index in group_done
                if WORKER_PID.search(lines[index]) is not None
            }
            if (
                len(starts_by_pid) != WORLD_SIZE
                or done_by_pid.keys() != starts_by_pid.keys()
                or any(
                    starts_by_pid[pid] >= done_by_pid[pid]
                    for pid in starts_by_pid
                )
            ):
                _fail(
                    f"step {step_index + 1} recapture does not preserve "
                    "per-worker start-before-finish ordering"
                )
            recapture_bounds.append((min(group_start), max(group_done)))
        step_1_start = recapture_bounds[0][0]
        step_1_done = min(restore_indices)
        if recapture_bounds[0][1] >= step_1_done:
            _fail("step 1 initial recapture did not finish before restore")
    else:
        recapture_bounds = []
        step_1_start = _line_index(lines, "driver_generate_start step=1")
        step_1_done = _line_index(lines, "driver_generate_done step=1")
        if step_1_start >= step_1_done:
            _fail("step 1 driver generation markers are out of order")

    step_2_start_pids: set[int] = set()
    step_2_done_pids: set[int] = set()
    step_2_start: int | None = None
    step_2_done: int | None = None
    if executed_steps == 2:
        if full_moe_capture:
            step_2_start = recapture_bounds[1][0]
            step_2_done = len(lines)
        else:
            step_2_start = _line_index(lines, "driver_generate_start step=2")
            step_2_done = _line_index(lines, "driver_generate_done step=2")
        if not step_1_done < step_2_start < step_2_done:
            _fail("driver generation steps are out of order")
        if max(restore_indices) >= step_2_start:
            _fail("full-world restore did not finish before step 2 generation")
        step_2_lines = lines[step_2_start:step_2_done]
        step_2_start_pids = _worker_pids(step_2_lines, recapture_start_marker)
        step_2_done_pids = _worker_pids(step_2_lines, recapture_done_marker)
        if (
            len(step_2_start_pids) != WORLD_SIZE
            or step_2_done_pids != step_2_start_pids
        ):
            _fail("step 2 graph recapture did not cover all 16 workers")
    elif (
        not full_moe_capture
        and any("driver_generate_start step=2" in line for line in lines)
    ):
        _fail("single-step contract unexpectedly executed step 2")

    full_moe_replay_phases: dict[str, list[int]] = {}
    topology_capture_summary: list[dict[str, Any]] = []
    if full_moe_capture:
        if lines[:step_1_start] and any(CAPTURE.search(line) for line in lines[:step_1_start]):
            _fail("elastic lifecycle captured a graph before step 1 stabilized weights/KV")
        step_1_capture_count = sum(
            CAPTURE.search(line) is not None for line in lines[step_1_start:step_1_done]
        )
        step_2_capture_count = (
            sum(
                CAPTURE.search(line) is not None
                for line in lines[step_2_start:step_2_done]
            )
            if step_2_start is not None and step_2_done is not None
            else 0
        )
        expected_capture_counts = (
            expected_step_1_captures,
            expected_step_2_captures,
            expected_total_captures,
        )
        actual_capture_counts = (
            step_1_capture_count,
            step_2_capture_count,
            len(captures),
        )
        if actual_capture_counts != expected_capture_counts:
            _fail(
                "full-MoE capture counts differ from the plan-derived topology "
                f"contract: expected={expected_capture_counts} "
                f"got={actual_capture_counts}"
            )

        topology_events = []
        for index, line in enumerate(lines):
            match = TOPOLOGY_CAPTURE.search(line)
            if match is not None:
                topology_events.append(
                    {
                        "line": index,
                        "phase": match.group("phase"),
                        "rank": int(match.group("rank")),
                        "active_ranks": _parse_active_ranks(match.group("active")),
                    }
                )
        topology_bounds: dict[tuple[int, ...], tuple[int, int]] = {}
        for active_ranks in step_1_topologies:
            events = [
                event for event in topology_events if event["active_ranks"] == active_ranks
            ]
            starts = [event for event in events if event["phase"] == "starting"]
            finishes = [event for event in events if event["phase"] == "finished"]
            expected_ranks = set(active_ranks)
            if (
                {event["rank"] for event in starts} != expected_ranks
                or {event["rank"] for event in finishes} != expected_ranks
                or len(starts) != len(expected_ranks)
                or len(finishes) != len(expected_ranks)
            ):
                _fail(f"topology capture does not exactly cover {list(active_ranks)}")
            if max(event["line"] for event in starts) >= min(
                event["line"] for event in finishes
            ):
                _fail(f"topology capture finish raced capture start for {list(active_ranks)}")
            topology_bounds[active_ranks] = (
                min(event["line"] for event in starts),
                max(event["line"] for event in finishes),
            )
            topology_capture_summary.append(
                {
                    "active_ranks": list(active_ranks),
                    "capture_workers": sorted(expected_ranks),
                }
            )
        if len(topology_events) != 2 * sum(map(len, step_1_topologies)):
            _fail(f"unexpected extra topology capture events: {len(topology_events)}")
        ordered_bounds = [topology_bounds[topology] for topology in step_1_topologies]
        ordering_points = [step_1_start]
        for start, finish in ordered_bounds:
            ordering_points.extend((start, finish))
        ordering_points.append(step_1_done)
        if ordering_points != sorted(ordering_points) or len(set(ordering_points)) != len(
            ordering_points
        ):
            _fail("full-MoE topology captures are out of step-1 order")

        topology_starts = [start for start, _ in ordered_bounds]
        phase_starts = [step_1_start] + topology_starts
        phase_ends = topology_starts + [step_1_done]
        phase_topologies = [tuple(range(WORLD_SIZE))] + step_1_topologies
        for phase_start, phase_end, active_ranks in zip(
            phase_starts, phase_ends, phase_topologies
        ):
            label = f"step-1 floor{len(active_ranks)}"
            phase_key = (
                "step_1_full16"
                if len(active_ranks) == WORLD_SIZE
                else f"step_1_floor{len(active_ranks)}"
            )
            full_moe_replay_phases[phase_key] = (
                _require_full_moe_replay_phase(
                    lines[phase_start:phase_end], label, len(active_ranks)
                )
            )
        if step_2_start is not None and step_2_done is not None:
            full_moe_replay_phases["step_2_full16"] = (
                _require_full_moe_replay_phase(
                    lines[step_2_start:step_2_done], "step-2 full16", WORLD_SIZE
                )
            )

    if text.count("response/aborted_ratio:0.0") != executed_steps:
        _fail(f"expected {executed_steps} zero-abort training metrics")
    for marker in ("Epoch 0 completed", "After trainer.fit"):
        if text.count(marker) != 1:
            _fail(f"missing or duplicate successful completion marker: {marker}")
    fault_counts = {
        name: len(pattern.findall(text)) for name, pattern in FORBIDDEN.items()
    }
    if any(fault_counts.values()):
        _fail(f"contract-defined runtime faults found: {fault_counts}")

    return {
        "worker_count": WORLD_SIZE,
        "target_policy": target_policy,
        "cudagraph_mode": (
            "FULL_DECODE_ONLY" if native_full_decode else "PIECEWISE"
        ),
        "native_full_decode": native_full_decode,
        "prefill_decode_separation": {
            "mixed_full_capture_events": mixed_capture_count,
            "mixed_full_replay_events": mixed_replay_count,
            "prefill_execution": "eager",
            "uniform_decode_execution": "FULL_DECODE_ONLY",
        }
        if native_full_decode
        else None,
        "capture_sizes": capture_sizes,
        "full_moe_capture": full_moe_capture,
        "marker_counts": marker_counts,
        # A capture invocation is one active worker capturing the configured
        # shape set for its current topology. It is not the graph-instance
        # count when multiple capture sizes are configured.
        "capture_count": len(captures),
        "capture_invocations": len(captures),
        "captured_graph_instances": len(captures) * len(capture_sizes),
        "resident_full_decode_graphs_per_active_rank": len(capture_sizes),
        "single_full_decode_graph_per_active_rank": len(capture_sizes) == 1,
        "topology_graph_cache": False,
        "max_observed_capture_gib_per_worker": initial_capture_gib,
        "replay_workers": sorted(replay_pids),
        "shrink_stages": shrink_summary,
        "restore": {
            "ranks": sorted(item["rank"] for item in restores),
            "dp_size": WORLD_SIZE,
            "ep_size": WORLD_SIZE,
            "critical_path_ms": max(item["total_ms"] for item in restores),
        },
        "step_2_recapture_workers": sorted(step_2_done_pids),
        "topology_captures": topology_capture_summary,
        "full_moe_replay_phases": full_moe_replay_phases,
        "attention_workspace": attention_workspace,
        "contract_fault_counts": fault_counts,
        "known_nonfatal_noise": {
            "torch_dynamo_metrics_traceback": text.count(
                "Unexpected exception logging compilation metrics"
            ),
            "post_success_tbe_cleanup": text.count("main process disappeared"),
        },
    }


def _verify_outputs(root: Path, executed_steps: int) -> dict[str, Any]:
    data_dir = root / "rollout_data"
    length_dir = root / "rollout_length"
    summary: dict[str, Any] = {}
    for step in range(1, executed_steps + 1):
        jsonl_path = _require_file(data_dir / f"{step}.jsonl", f"step {step} JSONL")
        length_path = _require_file(
            length_dir / f"length_{step}.txt", f"step {step} length file"
        )
        try:
            lengths = [int(line) for line in length_path.read_text("utf-8").splitlines()]
        except ValueError as error:
            raise ACLGraphSmokeVerificationError(
                f"step {step} contains a non-integer length"
            ) from error
        rows = []
        with jsonl_path.open("r", encoding="utf-8") as source:
            for line_number, line in enumerate(source, start=1):
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as error:
                    raise ACLGraphSmokeVerificationError(
                        f"invalid step {step} JSONL row {line_number}: {error}"
                    ) from error
        if len(rows) != EXPECTED_RESPONSES_PER_STEP or len(lengths) != len(rows):
            _fail(f"step {step} must contain 32 JSONL rows and 32 lengths")
        decoded = []
        for row_number, row in enumerate(rows, start=1):
            value = row.get("decoded_response_length")
            mask = row.get("response_mask")
            responses = row.get("responses")
            if not isinstance(value, int) or value <= 0:
                _fail(f"step {step} row {row_number} has invalid decoded length")
            if not isinstance(mask, list) or not isinstance(responses, list):
                _fail(f"step {step} row {row_number} lacks response arrays")
            if len(mask) != len(responses):
                _fail(f"step {step} row {row_number} response arrays differ in size")
            if any(item not in (0, 1) for item in mask):
                _fail(f"step {step} row {row_number} response mask is not binary")
            if mask[:value] != [1] * value or any(mask[value:]):
                _fail(f"step {step} row {row_number} response mask is not contiguous")
            if row.get("step") != step:
                _fail(f"step {step} row {row_number} carries the wrong step")
            decoded.append(value)
        if decoded != lengths:
            _fail(f"step {step} decoded lengths differ from length artifact")
        if sum(decoded) != EXPECTED_TOKENS_PER_STEP:
            _fail(f"step {step} generated {sum(decoded)} tokens, expected 640")
        summary[str(step)] = {
            "responses": len(rows),
            "generated_tokens": sum(decoded),
            "min_tokens": min(decoded),
            "max_tokens": max(decoded),
            "jsonl": _artifact(jsonl_path),
            "lengths": _artifact(length_path),
        }
    return summary


def verify(root: Path) -> dict[str, Any]:
    root = root.resolve()
    protocol_path = root / "protocol.env"
    protocol = _load_protocol(protocol_path)
    try:
        executed_steps = int(protocol.get("executed_steps", "2"))
    except ValueError as error:
        raise ACLGraphSmokeVerificationError(
            f"invalid executed_steps={protocol.get('executed_steps')!r}"
        ) from error
    if executed_steps not in (1, 2):
        _fail(f"executed_steps must be 1 or 2, got {executed_steps}")
    plan_path = root / "oracle/length_sorted_rank_plan.json"
    summary_path = root / "oracle/length_sorted_rank_plan_summary.json"
    log_path = _only_file(root / "run/logs", "*.txt", "primary log")
    plan, plan_summary = _verify_plan(plan_path)
    summary_plan = _load_json(summary_path, "rank-plan summary")
    if not isinstance(summary_plan, list) or len(summary_plan) != len(plan):
        _fail("rank-plan summary must contain the same two steps as the plan")
    contract_fields = (
        "step",
        "selected_floor",
        "shrink_stages",
        "stage_survivor_ranks",
        "tail_guard_enabled",
    )
    for plan_row, summary_row in zip(plan, summary_plan):
        if not isinstance(summary_row, dict) or any(
            summary_row.get(field) != plan_row.get(field)
            for field in contract_fields
        ):
            _fail("rank-plan summary differs from the executed plan contract")

    runtime = _verify_log(log_path, plan, executed_steps)
    full_moe_capture = bool(runtime["full_moe_capture"])
    native_full_decode = bool(runtime["native_full_decode"])
    return {
        "schema_version": 3,
        "status": "PASS",
        "experiment": "adafloor_elastic_aclgraph_lifecycle_smoke",
        "root": str(root),
        "scope": {
            "graph": (
                "vLLM rollout native FULL_DECODE_ONLY outer graph including "
                "KV write, Attention read, elastic MoE, and dense decode"
                if native_full_decode
                else "vLLM rollout PIECEWISE dense regions and supported "
                "single-token Ascend attention and elastic MoE"
                if full_moe_capture
                else "vLLM rollout PIECEWISE dense regions and supported "
                "single-token Ascend attention"
            ),
            "eager_boundaries": []
            if full_moe_capture
            else ["elastic_ascend_moe_forward"],
            "training": "Megatron eager",
            "sidecar": "disabled",
        },
        "plan": plan_summary,
        "runtime": runtime,
        "compilation_graphs": _verify_compilation_graphs(
            root,
            log_path,
            full_moe_capture,
            native_full_decode=native_full_decode,
        ),
        "outputs": _verify_outputs(root, executed_steps),
        "artifacts": {
            "plan": _artifact(plan_path),
            "plan_summary": _artifact(summary_path),
            "protocol": _artifact(protocol_path),
            "primary_log": _artifact(log_path),
            "ascend_extension": _artifact(
                _require_file(ASCEND_EXTENSION, "Ascend graph extension")
            ),
            "implementation_sources": {
                relative_path: _artifact(
                    _require_file(
                        SOURCE_ROOT / relative_path,
                        f"implementation source {relative_path}",
                    )
                )
                for relative_path in IMPLEMENTATION_SOURCES
            },
        },
        "claim_boundary": (
            "This smoke proves graph capture/replay and one "
            + "-to-".join(
                str(value) for value in [WORLD_SIZE, *plan[0]["shrink_stages"]]
            )
            + " shrink and full-world restore"
            + (", including next-step recapture," if executed_steps == 2 else "")
            + " lifecycle on "
            "the pinned single-node stack. It is not a throughput, training "
            "convergence, TorchAir, multi-node, or sidecar graph result."
        ),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        help="JSON output path (default: ROOT/ACLGRAPH_SMOKE_SUMMARY.json)",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    output = args.output or args.root / "ACLGRAPH_SMOKE_SUMMARY.json"
    try:
        summary = verify(args.root)
    except ACLGraphSmokeVerificationError as error:
        print(f"FAIL: {error}")
        return 1
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", "utf-8")
    print(f"PASS: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
