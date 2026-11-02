#!/usr/bin/env python3
"""Compare Qwen3 FULL_DECODE_ONLY [16,32] and [32] Vanilla runs."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


REQUIRED_EQUAL_PROTOCOL_KEYS = (
    "implementation_stack",
    "optimization_profile",
    "model_path",
    "distcp_path",
    "train_file_sha256",
    "test_file_sha256",
    "seed",
    "steps",
    "prompts_per_step",
    "rollout_n",
    "responses_per_step",
    "max_prompt_length",
    "max_response_length",
    "max_num_batched_tokens",
    "max_num_seqs",
    "kv_tokens_per_rank",
    "kv_blocks",
    "task_queue_enable",
    "cudagraph_mode",
    "attention_backend",
    "cudagraph_copy_inputs",
    "prefill_execution",
    "decode_execution",
    "torchair",
    "graphex",
    "tail_guard",
    "shrink",
    "sidecar",
    "native_sleep_mode",
    "rollout_memory_lifecycle",
    "reuse_aclgraph_after_weight_update",
    "async_scheduling",
    "prefix_caching",
    "chunked_prefill",
    "filtered_custom_opp_bundle_sha256",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_env(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise SystemExit(f"missing protocol: {path}")
    result: dict[str, str] = {}
    for line in path.read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise SystemExit(f"malformed protocol line in {path}: {line!r}")
        key, value = line.split("=", 1)
        if key in result:
            raise SystemExit(f"duplicate protocol key {key!r} in {path}")
        result[key] = value
    return result


def _read_code_manifest(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise SystemExit(f"missing code manifest: {path}")
    result: dict[str, str] = {}
    for line in path.read_text().splitlines():
        match = re.fullmatch(r"([0-9a-f]{64})  (/.+)", line)
        if match is None:
            raise SystemExit(f"malformed code-manifest line in {path}: {line!r}")
        digest, source_path = match.groups()
        if source_path in result:
            raise SystemExit(f"duplicate source path in {path}: {source_path}")
        result[source_path] = digest
    return result


def _load_summary(root: Path) -> tuple[Path, dict[str, Any]]:
    path = root / "full_decode_epoch0_summary.json"
    if not path.is_file():
        raise SystemExit(f"missing completed summary: {path}")
    payload = json.loads(path.read_text())
    if payload.get("status") != "PASS":
        raise SystemExit(f"summary is not PASS: {path}")
    arm = payload.get("full_decode_fia_tq1")
    if not isinstance(arm, dict):
        raise SystemExit(f"missing full_decode_fia_tq1 arm: {path}")
    return path, arm


def _epoch_dir(root: Path, protocol: dict[str, str]) -> Path:
    epoch = root / protocol["run_name"] / "epoch_000_mode0_probe"
    if not epoch.is_dir():
        raise SystemExit(f"missing epoch directory: {epoch}")
    return epoch


def _row_identity(row: dict[str, Any]) -> tuple[str, str, str, int]:
    prompt = json.dumps(row.get("input"), sort_keys=True, separators=(",", ":"))
    ground_truth = json.dumps(row.get("gts"), sort_keys=True, separators=(",", ":"))
    return (
        hashlib.sha256(prompt.encode()).hexdigest(),
        hashlib.sha256(ground_truth.encode()).hexdigest(),
        str(row.get("request_id")),
        int(row.get("rollout_rank", -1)),
    )


def _read_workload(epoch: Path, steps: int, expected_rows: int) -> dict[str, Any]:
    tokens_by_step: list[int] = []
    identities_by_step: list[Counter[tuple[str, str, str, int]]] = []
    lengths_by_step: list[list[int]] = []
    rank_lengths_by_step: list[dict[int, list[int]]] = []
    for step in range(1, steps + 1):
        length_path = epoch / "rollout_length" / f"length_{step}.txt"
        data_path = epoch / "rollout_data" / f"{step}.jsonl"
        lengths = [int(value) for value in length_path.read_text().split()]
        if len(lengths) != expected_rows:
            raise SystemExit(
                f"expected {expected_rows} lengths in {length_path}, got {len(lengths)}")
        identities: Counter[tuple[str, str, str, int]] = Counter()
        rank_lengths: dict[int, list[int]] = {}
        decoded_lengths: list[int] = []
        rows = 0
        with data_path.open() as handle:
            for line in handle:
                row = json.loads(line)
                identities[_row_identity(row)] += 1
                rank = int(row.get("rollout_rank", -1))
                decoded_length = int(row["decoded_response_length"])
                rank_lengths.setdefault(rank, []).append(decoded_length)
                decoded_lengths.append(decoded_length)
                rows += 1
        if rows != expected_rows:
            raise SystemExit(
                f"expected {expected_rows} rows in {data_path}, got {rows}")
        if Counter(decoded_lengths) != Counter(lengths):
            raise SystemExit(
                f"decoded_response_length disagrees with {length_path}")
        if sorted(rank_lengths) != list(range(16)):
            raise SystemExit(f"expected rollout ranks 0..15 in {data_path}")
        tokens_by_step.append(sum(lengths))
        lengths_by_step.append(lengths)
        rank_lengths_by_step.append(rank_lengths)
        identities_by_step.append(identities)
    return {
        "tokens_by_step": tokens_by_step,
        "lengths_by_step": lengths_by_step,
        "rank_lengths_by_step": rank_lengths_by_step,
        "identities_by_step": identities_by_step,
    }


def _eligible_rounds(rank_lengths: dict[int, list[int]]) -> dict[str, Any]:
    """Estimate DP-synchronous decode rounds eligible for the 16-token graph.

    At decode round t, the global graph shape can use bucket 16 only when every
    rank has at most 16 unfinished sequences.
    """
    ranks = list(rank_lengths.values())
    if not ranks or any(len(values) != 32 for values in ranks):
        raise SystemExit("each rollout rank must own exactly 32 responses")
    lengths = [length for values in ranks for length in values]
    max_rounds = max(lengths)
    eligible = 0
    for token_index in range(max_rounds):
        global_active = max(
            sum(length > token_index for length in rank_lengths)
            for rank_lengths in ranks)
        if global_active <= 16:
            eligible += 1
    return {
        "total_decode_rounds": max_rounds,
        "bucket16_eligible_rounds": eligible,
        "bucket16_eligible_fraction": eligible / max_rounds,
    }


def _find_log(epoch: Path) -> tuple[Path, str]:
    logs = list((epoch / "logs").glob("*.txt"))
    if len(logs) != 1:
        raise SystemExit(f"expected exactly one primary log under {epoch}, got {len(logs)}")
    return logs[0], logs[0].read_text(errors="replace")


def _assert_capture_evidence(text: str, expected_shapes: tuple[int, ...], label: str) -> None:
    normalized = re.sub(r"\s+", "", text)
    expected_list = "[" + ",".join(str(value) for value in sorted(expected_shapes, reverse=True)) + "]"
    marker = f"StartingtocaptureACLgraphsforcases:{expected_list},mode:FULL"
    if marker not in normalized:
        raise SystemExit(f"{label} log lacks capture-list marker {expected_list}")
    for shape in expected_shapes:
        if not re.search(
                rf"FULL_DECODE_ONLY Attention maximum workspace captured:.*shape={shape}\b",
                text):
            raise SystemExit(f"{label} log lacks FIA workspace capture for shape={shape}")


def _initial_capture_stats(text: str, rank_count: int = 16) -> dict[str, float | int]:
    matches = [
        (float(seconds), float(memory_gib))
        for seconds, memory_gib in re.findall(
            r"Graph capturing finished in ([0-9.]+) secs, took ([0-9.]+) GiB", text)
    ]
    if len(matches) < rank_count:
        raise SystemExit(
            f"expected at least {rank_count} graph-capture reports, got {len(matches)}")
    initial = matches[:rank_count]
    return {
        "rank_reports": len(initial),
        "seconds_min": min(value[0] for value in initial),
        "seconds_max": max(value[0] for value in initial),
        "seconds_mean": sum(value[0] for value in initial) / len(initial),
        "memory_gib_min": min(value[1] for value in initial),
        "memory_gib_max": max(value[1] for value in initial),
        "memory_gib_mean": sum(value[1] for value in initial) / len(initial),
    }


def _percent_ratio(candidate: float, baseline: float) -> float:
    return (candidate / baseline - 1.0) * 100.0


def compare(baseline_root: Path, candidate_root: Path,
            output_json: Path, output_md: Path) -> dict[str, Any]:
    baseline_protocol_path = baseline_root / "protocol.env"
    candidate_protocol_path = candidate_root / "protocol.env"
    baseline_code_path = baseline_root / "code_sha256.txt"
    candidate_code_path = candidate_root / "code_sha256.txt"
    baseline_protocol = _read_env(baseline_protocol_path)
    candidate_protocol = _read_env(candidate_protocol_path)
    if baseline_protocol.get("capture_sizes") != "[32]":
        raise SystemExit("baseline capture_sizes must be [32]")
    if candidate_protocol.get("capture_sizes") != "[16,32]":
        raise SystemExit("candidate capture_sizes must be [16,32]")
    for key in REQUIRED_EQUAL_PROTOCOL_KEYS:
        if baseline_protocol.get(key) != candidate_protocol.get(key):
            raise SystemExit(
                f"protocol mismatch for {key}: baseline={baseline_protocol.get(key)!r} "
                f"candidate={candidate_protocol.get(key)!r}")
    baseline_code = _read_code_manifest(baseline_code_path)
    candidate_code = _read_code_manifest(candidate_code_path)
    shared_sources = sorted(set(baseline_code) & set(candidate_code))
    if len(shared_sources) < 20:
        raise SystemExit(
            f"too few shared source files for a same-implementation pair: {len(shared_sources)}")
    mismatched_sources = [
        path for path in shared_sources
        if baseline_code[path] != candidate_code[path]
    ]
    if mismatched_sources:
        raise SystemExit(
            "shared implementation hashes differ: " + ", ".join(mismatched_sources))

    baseline_summary_path, baseline_arm = _load_summary(baseline_root)
    candidate_summary_path, candidate_arm = _load_summary(candidate_root)
    steps = int(candidate_protocol["steps"])
    expected_rows = int(candidate_protocol["responses_per_step"])
    baseline_epoch = _epoch_dir(baseline_root, baseline_protocol)
    candidate_epoch = _epoch_dir(candidate_root, candidate_protocol)
    baseline_work = _read_workload(baseline_epoch, steps, expected_rows)
    candidate_work = _read_workload(candidate_epoch, steps, expected_rows)
    for step, (baseline_ids, candidate_ids) in enumerate(zip(
            baseline_work["identities_by_step"],
            candidate_work["identities_by_step"], strict=True), start=1):
        if baseline_ids != candidate_ids:
            raise SystemExit(f"request identity multiset differs at step {step}")

    baseline_log, baseline_text = _find_log(baseline_epoch)
    candidate_log, candidate_text = _find_log(candidate_epoch)
    _assert_capture_evidence(baseline_text, (32,), "baseline")
    _assert_capture_evidence(candidate_text, (16, 32), "candidate")
    baseline_capture = _initial_capture_stats(baseline_text)
    candidate_capture = _initial_capture_stats(candidate_text)

    eligibility = [
        _eligible_rounds(rank_lengths)
        for rank_lengths in baseline_work["rank_lengths_by_step"]
    ]
    eligible_rounds = sum(item["bucket16_eligible_rounds"] for item in eligibility)
    total_rounds = sum(item["total_decode_rounds"] for item in eligibility)
    baseline_seconds = float(baseline_arm["rollout_seconds"])
    candidate_seconds = float(candidate_arm["rollout_seconds"])
    baseline_tokens = int(baseline_arm["generated_tokens"])
    candidate_tokens = int(candidate_arm["generated_tokens"])
    baseline_tps = float(baseline_arm["response_tokens_per_second"])
    candidate_tps = float(candidate_arm["response_tokens_per_second"])

    per_step = []
    for index in range(steps):
        base_s = float(baseline_arm["rollout_seconds_by_step"][index])
        cand_s = float(candidate_arm["rollout_seconds_by_step"][index])
        base_t = int(baseline_work["tokens_by_step"][index])
        cand_t = int(candidate_work["tokens_by_step"][index])
        per_step.append({
            "step": index + 1,
            "baseline_tokens": base_t,
            "candidate_tokens": cand_t,
            "generated_work_delta_percent": _percent_ratio(cand_t, base_t),
            "baseline_rollout_seconds": base_s,
            "candidate_rollout_seconds": cand_s,
            "rollout_seconds_delta_percent": _percent_ratio(cand_s, base_s),
            "baseline_response_tokens_per_second": base_t / base_s,
            "candidate_response_tokens_per_second": cand_t / cand_s,
            "response_tokens_per_second_delta_percent": _percent_ratio(
                cand_t / cand_s, base_t / base_s),
            **eligibility[index],
        })

    payload = {
        "schema_version": 1,
        "status": "PASS",
        "experiment": "qwen3_vanilla_full_decode_bucket_16_32_vs_32",
        "comparison_boundary": (
            "Same source implementation, model, seed, workload, and runtime protocol; "
            "only the explicit FULL_DECODE_ONLY capture-size list differs. Generated "
            "work is reported because stochastic trajectories need not be byte-identical."
        ),
        "baseline": {
            "capture_sizes": [32],
            "root": str(baseline_root.resolve()),
            "rollout_seconds": baseline_seconds,
            "generated_tokens": baseline_tokens,
            "response_tokens_per_second": baseline_tps,
            "mean_reward": float(baseline_arm["mean_reward"]),
            "initial_graph_capture": baseline_capture,
        },
        "candidate": {
            "capture_sizes": [16, 32],
            "root": str(candidate_root.resolve()),
            "rollout_seconds": candidate_seconds,
            "generated_tokens": candidate_tokens,
            "response_tokens_per_second": candidate_tps,
            "mean_reward": float(candidate_arm["mean_reward"]),
            "initial_graph_capture": candidate_capture,
        },
        "delta_candidate_vs_baseline": {
            "rollout_seconds_percent": _percent_ratio(candidate_seconds, baseline_seconds),
            "generated_work_percent": _percent_ratio(candidate_tokens, baseline_tokens),
            "response_tokens_per_second_percent": _percent_ratio(candidate_tps, baseline_tps),
            "mean_reward_absolute": (
                float(candidate_arm["mean_reward"]) - float(baseline_arm["mean_reward"])),
            "initial_graph_capture_memory_gib_mean_absolute": (
                float(candidate_capture["memory_gib_mean"])
                - float(baseline_capture["memory_gib_mean"])),
        },
        "bucket16_opportunity_from_baseline_lengths": {
            "eligible_decode_rounds": eligible_rounds,
            "total_decode_rounds": total_rounds,
            "eligible_fraction": eligible_rounds / total_rounds,
            "definition": (
                "A decode round is eligible when every DP rank has at most 16 "
                "unfinished local sequences, allowing the synchronized FULL graph "
                "dispatcher to select shape 16 instead of padding to 32."
            ),
        },
        "per_step": per_step,
        "provenance": {
            "baseline_protocol": str(baseline_protocol_path.resolve()),
            "baseline_protocol_sha256": _sha256(baseline_protocol_path),
            "candidate_protocol": str(candidate_protocol_path.resolve()),
            "candidate_protocol_sha256": _sha256(candidate_protocol_path),
            "baseline_summary": str(baseline_summary_path.resolve()),
            "baseline_summary_sha256": _sha256(baseline_summary_path),
            "candidate_summary": str(candidate_summary_path.resolve()),
            "candidate_summary_sha256": _sha256(candidate_summary_path),
            "baseline_code_manifest": str(baseline_code_path.resolve()),
            "baseline_code_manifest_sha256": _sha256(baseline_code_path),
            "candidate_code_manifest": str(candidate_code_path.resolve()),
            "candidate_code_manifest_sha256": _sha256(candidate_code_path),
            "shared_source_files_with_identical_hashes": len(shared_sources),
            "comparison_tool": str(Path(__file__).resolve()),
            "comparison_tool_sha256": _sha256(Path(__file__).resolve()),
            "baseline_log": str(baseline_log.resolve()),
            "candidate_log": str(candidate_log.resolve()),
        },
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    rows = "\n".join(
        f"| {item['step']} | {item['baseline_response_tokens_per_second']:.2f} | "
        f"{item['candidate_response_tokens_per_second']:.2f} | "
        f"{item['response_tokens_per_second_delta_percent']:+.2f}% | "
        f"{item['bucket16_eligible_fraction'] * 100:.1f}% |"
        for item in per_step)
    output_md.write_text(
        "# Qwen3 FULL_DECODE_ONLY Capture-Bucket Comparison\n\n"
        "Status: PASS\n\n"
        "| Capture sizes | Response tokens | Rollout seconds | Response tok/s | Mean reward |\n"
        "|---|---:|---:|---:|---:|\n"
        f"| [32] | {baseline_tokens} | {baseline_seconds:.3f} | {baseline_tps:.3f} | {float(baseline_arm['mean_reward']):.6f} |\n"
        f"| [16,32] | {candidate_tokens} | {candidate_seconds:.3f} | {candidate_tps:.3f} | {float(candidate_arm['mean_reward']):.6f} |\n\n"
        f"Candidate throughput delta: {_percent_ratio(candidate_tps, baseline_tps):+.2f}%. "
        f"Rollout-time delta: {_percent_ratio(candidate_seconds, baseline_seconds):+.2f}%. "
        f"Generated-work delta: {_percent_ratio(candidate_tokens, baseline_tokens):+.2f}%.\n\n"
        f"Based on the baseline response lengths, {eligible_rounds}/{total_rounds} "
        f"decode rounds ({eligible_rounds / total_rounds * 100:.1f}%) are eligible "
        "for the 16-token graph bucket.\n\n"
        f"Initial graph-capture memory reported per rank is "
        f"{baseline_capture['memory_gib_mean']:.3f} GiB for [32] and "
        f"{candidate_capture['memory_gib_mean']:.3f} GiB for [16,32].\n\n"
        "| Step | [32] tok/s | [16,32] tok/s | Delta | 16-bucket eligible rounds |\n"
        "|---:|---:|---:|---:|---:|\n"
        f"{rows}\n\n"
        "Request identity multisets match at every step. Capture and FIA workspace "
        "markers for both candidate shapes were verified. Generated work and reward "
        "remain reported rather than assuming stochastic output equality.\n"
    )
    print(output_md.read_text(), end="")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-md", type=Path)
    args = parser.parse_args()
    output_json = args.output_json or args.candidate_root / "full_decode_bucket_comparison.json"
    output_md = args.output_md or args.candidate_root / "full_decode_bucket_comparison.md"
    compare(args.baseline_root, args.candidate_root, output_json, output_md)


if __name__ == "__main__":
    main()
