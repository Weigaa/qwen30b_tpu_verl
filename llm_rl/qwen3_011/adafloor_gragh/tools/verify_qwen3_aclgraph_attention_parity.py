#!/usr/bin/env python3
"""Verify paired Vanilla Full16 eager/Attention-ACLGraph rollout runs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import statistics
from pathlib import Path
from typing import Any


ANSI_ESCAPE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
WORKER_PID = re.compile(r"WorkerDict pid=(\d+)")
CAPTURE_MEMORY = re.compile(
    r"Graph capturing finished in \d+ secs, took (?P<gib>[0-9]+(?:\.[0-9]+)?) GiB"
)
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
    "ray_failure": re.compile(r"RayTaskError|WorkerCrashedError|ActorDiedError"),
    "npu_oom": re.compile(
        r"OutOfMemoryError|ACL_ERROR_RT_MEMORY_ALLOCATION|NPU out of memory",
        re.IGNORECASE,
    ),
    "hccl_failure": re.compile(
        r"HCCL.*(?:fail|error)|EJ0003|ERR\d+.*HCCL", re.IGNORECASE
    ),
    "aclgraph_failure": re.compile(
        r"stale (?:key|value)-cache address|stale block-table address|"
        r"Attention ACLGraph .* mismatch|Attention ACLGraph .* absent|"
        r"Skipping ACL graph capture|falling back to NONE|"
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


def parse_env(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for number, line in enumerate(
        require_file(path, "protocol").read_text("utf-8").splitlines(), 1
    ):
        if not line or "=" not in line:
            fail(f"invalid protocol line {number}: {line!r}")
        key, value = line.split("=", 1)
        if not key or key in values:
            fail(f"duplicate or empty protocol key at line {number}")
        values[key] = value
    return values


def load_json(path: Path, label: str) -> Any:
    try:
        return json.loads(require_file(path, label).read_text("utf-8"))
    except json.JSONDecodeError as error:
        raise VerificationError(f"invalid {label}: {path}: {error}") from error


def only_log(arm: Path) -> Path:
    logs = sorted((arm / "run" / "logs").glob("*.txt"))
    if len(logs) != 1:
        fail(f"expected one primary log under {arm}, found {len(logs)}")
    return require_file(logs[0], "primary log")


def marker_pids(text: str, marker: str) -> set[int]:
    pids = set()
    for line in text.splitlines():
        if marker not in line:
            continue
        match = WORKER_PID.search(line)
        if match is None:
            fail(f"graph marker lacks WorkerDict pid: {marker}")
        pids.add(int(match.group(1)))
    return pids


def verify_code_contract(path: Path) -> dict[str, Any]:
    rows = []
    for number, line in enumerate(
        require_file(path, "code contract").read_text("utf-8").splitlines(), 1
    ):
        parts = line.split(maxsplit=1)
        if len(parts) != 2 or not re.fullmatch(r"[0-9a-f]{64}", parts[0]):
            fail(f"invalid code contract line {number}")
        expected, raw_path = parts
        source = require_file(Path(raw_path), "contract source")
        actual = sha256(source)
        if actual != expected:
            fail(f"contract source changed: {source}")
        rows.append({"path": str(source.resolve()), "sha256": actual})
    if len(rows) < 8:
        fail(f"expected at least 8 frozen source files, found {len(rows)}")
    return {"artifact": artifact(path), "files": rows}


def verify_protocol(path: Path, mode: str) -> dict[str, str]:
    values = parse_env(path)
    expected = {
        "schema_version": "1",
        "mode": mode,
        "stack": "vllm-0.11.0_vllm-ascend-0.11.0rc0",
        "batch_size": "16",
        "rollout_n": "1",
        "decode_tokens": "64",
        "temperature": "0.0",
        "seed": "101",
        "attention_graph": "true" if mode == "graph" else "false",
    }
    for key, value in expected.items():
        if values.get(key) != value:
            fail(f"{mode} protocol {key}={values.get(key)!r}, expected {value!r}")
    if int(values.get("measure_steps", "0")) <= 0:
        fail(f"{mode} protocol has no measured steps")
    if mode == "eager":
        if values.get("moe_graph") != "false":
            fail("eager protocol unexpectedly enables MoE graph capture")
        if values.get("cudagraph_mode") != "NONE":
            fail("eager protocol must use cudagraph_mode=NONE")
    else:
        graph_mode = values.get("cudagraph_mode")
        if graph_mode not in {"PIECEWISE", "FULL_DECODE_ONLY"}:
            fail(f"unsupported graph parity mode: {graph_mode!r}")
        expected_moe = "true" if graph_mode == "FULL_DECODE_ONLY" else "false"
        if values.get("moe_graph") != expected_moe:
            fail(
                f"graph protocol moe_graph={values.get('moe_graph')!r}, "
                f"expected {expected_moe!r} for {graph_mode}"
            )
    return values


def verify_steps(path: Path, protocol: dict[str, str]) -> list[dict[str, Any]]:
    records = []
    for number, line in enumerate(
        require_file(path, "step records").read_text("utf-8").splitlines(), 1
    ):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise VerificationError(f"invalid step record {number}: {error}") from error
        records.append(row)
    warmup = int(protocol["warmup_steps"])
    measured = [row for row in records if row.get("phase") == "measure"]
    if len(records) != warmup + int(protocol["measure_steps"]):
        fail("step count differs from protocol")
    if len(measured) != int(protocol["measure_steps"]):
        fail("measured step count differs from protocol")
    for row in records:
        if row.get("batch_size") != 16 or row.get("rollout_n") != 1:
            fail("step batch contract mismatch")
        if row.get("generated_samples") != 16 or row.get("decode_tokens") != 1024:
            fail("step generated-work contract mismatch")
        hashes = row.get("response_row_token_sha256")
        if not isinstance(hashes, list) or len(hashes) != 16:
            fail("step lacks 16 per-response token hashes")
        if not all(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value)
                   for value in hashes):
            fail("step has invalid per-response token hash")
    return measured


def verify_log(
    path: Path, mode: str, protocol: dict[str, str]
) -> dict[str, Any]:
    text = ANSI_ESCAPE.sub("", require_file(path, "primary log").read_text("utf-8"))
    faults = {
        name: len(pattern.findall(text)) for name, pattern in FORBIDDEN.items()
    }
    if any(faults.values()):
        fail(f"{mode} runtime faults: {faults}")
    evidence: dict[str, Any] = {"artifact": artifact(path), "faults": faults}
    if mode == "graph":
        graph_mode = protocol["cudagraph_mode"]
        if graph_mode == "FULL_DECODE_ONLY":
            mixed_capture_count = len(FULL_DECODE_MIXED_CAPTURE.findall(text))
            mixed_replay_count = len(FULL_DECODE_MIXED_REPLAY.findall(text))
            if mixed_capture_count or mixed_replay_count:
                fail(
                    "FULL_DECODE_ONLY captured or replayed a mixed prefill/decode "
                    "batch instead of using eager prefill: "
                    f"captures={mixed_capture_count} replays={mixed_replay_count}"
                )
            markers = {
                "native_full_decode_config": (
                    "FULL_DECODE_ONLY compilation enabled on NPU"
                ),
                "full_decode_platform": (
                    "Native FULL_DECODE_ONLY ACLGraph enabled: KV write"
                ),
                "kv_write_capture": (
                    "FULL_DECODE_ONLY ACLGraph captured Attention KV write "
                    "and paged read inside the outer model graph"
                ),
                "maximum_attention_workspace": (
                    "FULL_DECODE_ONLY Attention maximum workspace captured: "
                    "seq_len_bucket=6144"
                ),
                "attention_update": "Attention ACLGraph metadata update active",
                "capture": "Graph capturing finished",
                "replay": "Replaying aclgraph",
            }
        else:
            markers = {
                "piecewise_boundary": "Elastic ACLGraph boundary enabled",
                "attention_update": "Attention ACLGraph metadata update active",
                "capture": "Graph capturing finished",
                "replay": "Replaying aclgraph",
            }
        marker_counts = {}
        for name, marker in markers.items():
            pids = marker_pids(text, marker)
            if len(pids) != 16:
                fail(f"graph {name} covered {len(pids)} workers, expected 16")
            marker_counts[name] = len(pids)
        if graph_mode == "FULL_DECODE_ONLY":
            if "dynamic KV write remains outside" in text:
                fail("full-decode log fell back to the PIECEWISE KV-write boundary")
            if "MoE/HCCL executes outside PIECEWISE ACLGraph" in text:
                fail("full-decode log fell back to the PIECEWISE MoE boundary")
        else:
            if ("Attention's dynamic KV write remains outside and its paged read "
                    "core executes in a nested ACLGraph") not in text:
                fail("graph log lacks nested Attention ACLGraph declaration")
            if "MoE/HCCL executes outside PIECEWISE ACLGraph" not in text:
                fail("graph log lacks MoE boundary declaration")
        memories = [float(match.group("gib")) for match in CAPTURE_MEMORY.finditer(text)]
        if not memories:
            fail("graph log lacks capture-memory measurements")
        evidence.update(
            marker_worker_counts=marker_counts,
            capture_memory_gib={
                "samples": len(memories),
                "min": min(memories),
                "max": max(memories),
                "mean": statistics.fmean(memories),
            },
        )
        if graph_mode == "FULL_DECODE_ONLY":
            evidence["prefill_decode_separation"] = {
                "mixed_full_capture_events": mixed_capture_count,
                "mixed_full_replay_events": mixed_replay_count,
                "prefill_execution": "eager",
                "uniform_decode_execution": "FULL_DECODE_ONLY",
            }
    elif "Replaying aclgraph" in text or "Graph capturing finished" in text:
        fail("eager log unexpectedly contains ACLGraph execution")
    return evidence


def mean(values: list[float]) -> float:
    if not values or not all(math.isfinite(value) for value in values):
        fail("invalid performance samples")
    return statistics.fmean(values)


def verify(root: Path) -> dict[str, Any]:
    arms: dict[str, Any] = {}
    for mode in ("eager", "graph"):
        arm = root / mode
        protocol_path = arm / "protocol.env"
        protocol = verify_protocol(protocol_path, mode)
        contract = verify_code_contract(arm / "code_sha256.txt")
        summary_path = arm / "run" / "summary_batch_16.json"
        summary = load_json(summary_path, "summary")
        steps_path = arm / "run" / "steps.jsonl"
        measured = verify_steps(steps_path, protocol)
        if summary.get("measure_steps") != len(measured):
            fail(f"{mode} summary measured-step mismatch")
        arms[mode] = {
            "protocol": protocol,
            "protocol_artifact": artifact(protocol_path),
            "code_contract": contract,
            "summary": summary,
            "summary_artifact": artifact(summary_path),
            "steps_artifact": artifact(steps_path),
            "measured": measured,
            "log": verify_log(only_log(arm), mode, protocol),
        }

    eager = arms["eager"]
    graph = arms["graph"]
    if eager["code_contract"]["files"] != graph["code_contract"]["files"]:
        fail("eager and graph used different frozen source files")
    for key in ("task_queue_enable", "warmup_steps", "measure_steps"):
        if eager["protocol"].get(key) != graph["protocol"].get(key):
            fail(f"paired protocol differs at {key}")
    if len(eager["measured"]) != len(graph["measured"]):
        fail("paired measured-step count mismatch")

    pairs = []
    for index, (eager_row, graph_row) in enumerate(
        zip(eager["measured"], graph["measured"], strict=True)
    ):
        for key in (
            "prompt_token_sha256",
            "response_token_sha256",
            "response_row_token_sha256",
            "generated_samples",
            "decode_tokens",
        ):
            if eager_row.get(key) != graph_row.get(key):
                fail(f"measured pair {index} differs at {key}")
        eager_wall = float(eager_row["wall_s"])
        graph_wall = float(graph_row["wall_s"])
        eager_inner = float(eager_row["timing"]["generation_timing/max"])
        graph_inner = float(graph_row["timing"]["generation_timing/max"])
        pairs.append(
            {
                "index": index,
                "prompt_token_sha256": eager_row["prompt_token_sha256"],
                "response_token_sha256": eager_row["response_token_sha256"],
                "responses": eager_row["generated_samples"],
                "response_tokens": eager_row["decode_tokens"],
                "outer_wall_seconds": {"eager": eager_wall, "graph": graph_wall},
                "inner_generation_seconds": {
                    "eager": eager_inner,
                    "graph": graph_inner,
                },
                "outer_wall_delta_percent": (graph_wall / eager_wall - 1.0) * 100.0,
                "inner_generation_delta_percent":
                    (graph_inner / eager_inner - 1.0) * 100.0,
            }
        )

    eager_walls = [row["outer_wall_seconds"]["eager"] for row in pairs]
    graph_walls = [row["outer_wall_seconds"]["graph"] for row in pairs]
    eager_inner = [row["inner_generation_seconds"]["eager"] for row in pairs]
    graph_inner = [row["inner_generation_seconds"]["graph"] for row in pairs]
    result = {
        "schema_version": 1,
        "status": "PASS",
        "scope": "Vanilla Full16 greedy rollout only",
        "correctness": {
            "paired_measured_batches": len(pairs),
            "responses_per_batch": 16,
            "tokens_per_response": 64,
            "all_prompt_tokens_equal": True,
            "all_response_tokens_equal": True,
            "per_response_token_hashes_equal": True,
            "full_decode_kv_write_graph_parity": (
                graph["protocol"]["cudagraph_mode"] == "FULL_DECODE_ONLY"
            ),
        },
        "performance": {
            "graph_mode": graph["protocol"]["cudagraph_mode"],
            "task_queue_enable": int(eager["protocol"]["task_queue_enable"]),
            "outer_wall_seconds_mean": {
                "eager": mean(eager_walls),
                "graph": mean(graph_walls),
            },
            "outer_wall_delta_percent":
                (mean(graph_walls) / mean(eager_walls) - 1.0) * 100.0,
            "inner_generation_seconds_mean": {
                "eager": mean(eager_inner),
                "graph": mean(graph_inner),
            },
            "inner_generation_delta_percent":
                (mean(graph_inner) / mean(eager_inner) - 1.0) * 100.0,
            "pairs": pairs,
        },
        "graph_evidence": graph["log"],
        "artifacts": {
            mode: {
                key: arms[mode][key]
                for key in (
                    "protocol_artifact",
                    "code_contract",
                    "summary_artifact",
                    "steps_artifact",
                )
            }
            for mode in ("eager", "graph")
        },
        "claim_boundary": (
            "This verifies one fixed Full16 greedy rollout shape on the pinned "
            "0.11 stack. The graph contract is "
            f"{graph['protocol']['cudagraph_mode']}. It does not verify "
            "AdaFloor shrink/restore, TQ2 graph execution, stochastic sampling, "
            "or end-to-end RL throughput."
        ),
    }
    return result


def markdown(result: dict[str, Any]) -> str:
    performance = result["performance"]
    correctness = result["correctness"]
    return "\n".join(
        [
            "# Qwen3 0.11 Native ACLGraph Parity",
            "",
            f"Status: **{result['status']}**",
            "",
            f"- Scope: {result['scope']}",
            f"- Paired measured batches: {correctness['paired_measured_batches']}",
            "- Exact prompt, per-response token, and batch-token parity: PASS",
            f"- Task queue: {performance['task_queue_enable']}",
            f"- Graph mode: {performance['graph_mode']}",
            "- Mean outer wall: "
            f"eager {performance['outer_wall_seconds_mean']['eager']:.6f} s, "
            f"graph {performance['outer_wall_seconds_mean']['graph']:.6f} s, "
            f"delta {performance['outer_wall_delta_percent']:+.3f}%",
            "- Mean inner generation: "
            f"eager {performance['inner_generation_seconds_mean']['eager']:.6f} s, "
            f"graph {performance['inner_generation_seconds_mean']['graph']:.6f} s, "
            f"delta {performance['inner_generation_delta_percent']:+.3f}%",
            "",
            f"Claim boundary: {result['claim_boundary']}",
            "",
        ]
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    result = verify(args.root.resolve())
    json_path = args.root / "attention_aclgraph_parity_summary.json"
    md_path = args.root / "attention_aclgraph_parity_summary.md"
    json_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    md_path.write_text(markdown(result))
    print(f"PASS: {json_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except VerificationError as error:
        raise SystemExit(f"FAIL: {error}") from error
