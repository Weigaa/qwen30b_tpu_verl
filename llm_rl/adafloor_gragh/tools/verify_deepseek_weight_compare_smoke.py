#!/usr/bin/env python3
"""Verify the pinned DeepSeek-V2-Lite-Chat weight-comparison smoke run."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any


WORLD_SIZE = 16
EXPECTED_ROWS = 32
EXPECTED_PARAMETERS_PER_RANK = 324
EXPECTED_ROUTED_STREAMS_PER_RANK = 8
EXPECTED_MODEL_ID = "deepseek-ai/DeepSeek-V2-Lite-Chat"
EXPECTED_MODEL_REVISION = "85864749cd611b4353ce1decdb286193298f64c7"
EXPECTED_CATEGORY_TOTALS = {
    "embedding_or_head": 2,
    "attention": 135,
    "dense_mlp": 2,
    "norm": 55,
    "router": 26,
    "shared_expert": 52,
    "routed_expert": 52,
}
EXPECTED_COMPLETE_VALUES = {
    "TASK_QUEUE_ENABLE": "2",
    "RECOMPUTE_METHOD": "uniform",
    "RECOMPUTE_NUM_LAYERS": "1",
    "TRAINING_STEPS": "1",
    "TRAIN_BATCH_SIZE": "32",
    "MAX_PROMPT_LENGTH": "1024",
    "MAX_RESPONSE_LENGTH": "32",
    "TAIL_VALIDATE_LEVEL_TOKENS_BY_STEP": "<unset>",
    "ROLLOUT_N": "1",
    "ROLLOUT_MAX_NUM_BATCHED_TOKENS": "1056",
    "ROLLOUT_MAX_NUM_SEQS": "32",
    "EXPECTED_ROWS": "32",
    "REQUIRE_SEMANTIC_OUTPUT": "0",
    "ROLLOUT_LOAD_FORMAT": "auto",
    "PRESERVE_INITIAL_HF_WEIGHTS": "0",
    "COMPARE_ONLINE_SYNC_TO_HF": "1",
    "MODEL_ID": EXPECTED_MODEL_ID,
    "MODEL_REVISION": EXPECTED_MODEL_REVISION,
    "MOE_ALLTOALL_OVERLAP_COMM": "True",
    "MOE_SHARED_EXPERT_OVERLAP": "True",
    "DEALLOCATE_PIPELINE_OUTPUTS": "False",
    "ACTOR_PPO_MAX_TOKEN_LEN_PER_GPU": "17408",
    "ROLLOUT_LOG_PROB_MAX_TOKEN_LEN_PER_GPU": "17408",
    "KV_TOKENS_PER_RANK": "16384",
}

ANSI_ESCAPE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
CAPTURE_RE = re.compile(
    r"Captured initial DeepSeek HF weight samples: rank=(?P<rank>\d+) "
    r"params=(?P<params>\d+)"
)
COMPARE_RE = re.compile(
    r"DeepSeek online-sync HF comparison: rank=(?P<rank>\d+) "
    r"total=(?P<total>\d+) matched=(?P<matched>\d+) "
    r"mismatched=(?P<mismatched>\d+) "
    r"category_totals=(?P<category_totals>\{.*?\}) "
    r"category_mismatches=(?P<category_mismatches>\{.*?\}) "
    r"missing=(?P<missing>\[.*?\]) "
    r"sample_errors=(?P<sample_errors>\[.*?\]) "
    r"mismatch_samples=(?P<mismatch_samples>\[.*\])$"
)
ROUTED_RE = re.compile(
    r"DeepSeek layer-1 routed w13 comparison: rank=(?P<rank>\d+) "
    r"result=(?P<result>\{.*\})$"
)
PASS_RE = re.compile(
    r"DeepSeek online-sync HF comparison PASS: rank=(?P<rank>\d+) "
    r"total=(?P<total>\d+) routed_streams=(?P<streams>\d+)"
)
OOM_RE = re.compile(
    r"out of memory|OutOfMemoryError|ACL_ERROR_RT_MEMORY_ALLOCATION|"
    r"\b(?:NPU|CUDA)?\s*OOM\b",
    re.IGNORECASE,
)
PREEMPT_RE = re.compile(r"\bpreempt(?:ion|ed|ing|s)?\b", re.IGNORECASE)
FATAL_RE = re.compile(
    r"Traceback \(most recent call last\):|"
    r"\b(?:RayTaskError|WorkerCrashedError|ActorDiedError)\b|"
    r"\b(?:RuntimeError|ValueError|AssertionError|SystemExit|"
    r"ProcessLookupError):|"
    r"\[ERROR\]|\bFATAL\b|Fatal error|Segmentation fault|core dumped"
)


class WeightCompareVerificationError(RuntimeError):
    """Raised when a weight-comparison artifact violates its contract."""


def _fail(message: str) -> None:
    raise WeightCompareVerificationError(message)


def _require_file(path: Path, description: str) -> Path:
    if path.is_symlink():
        _fail(f"{description} must not be a symbolic link: {path}")
    if not path.is_file():
        _fail(f"missing {description}: {path}")
    return path


def _require_directory(path: Path, description: str) -> Path:
    if path.is_symlink():
        _fail(f"{description} must not be a symbolic link: {path}")
    if not path.is_dir():
        _fail(f"missing {description}: {path}")
    return path


def _sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _artifact(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _parse_complete(path: Path) -> dict[str, str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines or lines[0] != "COMPLETE DeepSeek actor update probe":
        _fail("COMPLETE marker has the wrong or missing header")
    values: dict[str, str] = {}
    for line_number, line in enumerate(lines[1:], start=2):
        if not line:
            continue
        if "=" not in line:
            _fail(f"invalid COMPLETE line {line_number}: {line!r}")
        key, value = line.split("=", 1)
        if not key or key in values:
            _fail(f"duplicate or empty COMPLETE key on line {line_number}")
        values[key] = value
    for key, expected in EXPECTED_COMPLETE_VALUES.items():
        if values.get(key) != expected:
            _fail(
                f"COMPLETE {key}={values.get(key)!r}, expected {expected!r}"
            )
    for key in ("MODEL_PATH", "DISTCP_PATH"):
        if not values.get(key):
            _fail(f"COMPLETE does not define {key}")
    return values


def _load_json(path: Path, description: str) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise WeightCompareVerificationError(
            f"invalid {description} {path}: {error}"
        ) from error


def _validate_model_assets(values: dict[str, str]) -> dict[str, Any]:
    model_path = _require_directory(Path(values["MODEL_PATH"]), "model path")
    distcp_path = _require_directory(Path(values["DISTCP_PATH"]), "distcp path")
    model_config_path = _require_file(model_path / "config.json", "model config")
    model_index_path = _require_file(
        model_path / "model.safetensors.index.json", "model weight index"
    )
    conversion_manifest_path = _require_file(
        distcp_path / ".adafloor_deepseek_v2_lite_manifest.json",
        "conversion manifest",
    )
    distcp_metadata_path = _require_file(
        distcp_path / ".metadata", "distcp metadata"
    )

    config = _load_json(model_config_path, "model config")
    if config.get("architectures") != ["DeepseekV2ForCausalLM"]:
        _fail("model config does not identify DeepseekV2ForCausalLM")
    if config.get("n_routed_experts") != 64:
        _fail("model config does not contain 64 routed experts")
    if config.get("num_experts_per_tok") != 6:
        _fail("model config does not use top-6 routed experts")

    manifest = _load_json(conversion_manifest_path, "conversion manifest")
    expected_manifest = {
        "architecture": "DeepseekV2ForCausalLM",
        "expert_model_parallel_size": 4,
        "model_id": EXPECTED_MODEL_ID,
        "model_revision": EXPECTED_MODEL_REVISION,
        "pipeline_model_parallel_size": 4,
        "world_size": WORLD_SIZE,
    }
    if manifest != expected_manifest:
        _fail(
            "conversion manifest does not match the pinned Chat PP4 EP4 contract"
        )
    hf_shards = sorted(model_path.glob("model-*-of-*.safetensors"))
    if len(hf_shards) != 4 or any(path.is_symlink() for path in hf_shards):
        _fail(f"expected four regular Hugging Face weight shards, found {len(hf_shards)}")
    distcp_shards = sorted(distcp_path.glob("__*.distcp"))
    if len(distcp_shards) != 32 or any(path.is_symlink() for path in distcp_shards):
        _fail(f"expected 32 regular distcp shards, found {len(distcp_shards)}")

    return {
        "model_id": EXPECTED_MODEL_ID,
        "model_revision": EXPECTED_MODEL_REVISION,
        "architecture": "DeepseekV2ForCausalLM",
        "hf_path": str(model_path.resolve()),
        "hf_weight_shards": len(hf_shards),
        "distcp_path": str(distcp_path.resolve()),
        "distcp_shards": len(distcp_shards),
        "pipeline_model_parallel_size": 4,
        "expert_model_parallel_size": 4,
        "artifacts": {
            "model_config": _artifact(model_config_path),
            "model_weight_index": _artifact(model_index_path),
            "conversion_manifest": _artifact(conversion_manifest_path),
            "distcp_metadata": _artifact(distcp_metadata_path),
        },
    }


def _literal(value: str, context: str, expected_type: type[Any]) -> Any:
    try:
        result = ast.literal_eval(value)
    except (SyntaxError, ValueError) as error:
        raise WeightCompareVerificationError(
            f"cannot parse {context}: {error}"
        ) from error
    if not isinstance(result, expected_type):
        _fail(f"{context} has type {type(result).__name__}")
    return result


def _add_rank_record(
    records: dict[int, Any], rank: int, value: Any, context: str
) -> None:
    if rank not in range(WORLD_SIZE):
        _fail(f"{context} reports invalid rank {rank}")
    if rank in records:
        _fail(f"duplicate {context} record for rank {rank}")
    records[rank] = value


def _require_all_ranks(records: dict[int, Any], context: str) -> None:
    expected = set(range(WORLD_SIZE))
    if set(records) != expected:
        _fail(
            f"{context} rank coverage differs, missing={sorted(expected - set(records))} "
            f"extra={sorted(set(records) - expected)}"
        )


def _validate_routed_record(rank: int, result: dict[str, Any]) -> list[int]:
    expected_experts = list(range(rank * 4, rank * 4 + 4))
    expected_map = {expert: slot for slot, expert in enumerate(expected_experts)}
    if result.get("global_to_local") != expected_map:
        _fail(f"rank {rank} layer-1 expert map differs from the EP16 contract")
    if result.get("expected_stream_count") != EXPECTED_ROUTED_STREAMS_PER_RANK:
        _fail(f"rank {rank} expected routed stream count differs")
    if result.get("captured_stream_count") != EXPECTED_ROUTED_STREAMS_PER_RANK:
        _fail(f"rank {rank} captured routed stream count differs")
    for key in ("missing_stream_names", "extra_stream_names"):
        if result.get(key) != []:
            _fail(f"rank {rank} has nonempty {key}")

    expected_stream_matches: dict[str, str] = {}
    expected_post_matches: dict[str, str] = {}
    for expert in expected_experts:
        for projection in ("gate_proj", "up_proj"):
            label = f"expert{expert}.{projection}"
            expected_stream_matches[
                f"model.layers.1.mlp.experts.{expert}.{projection}.weight"
            ] = label
            expected_post_matches[label] = label
    if result.get("stream_matches") != expected_stream_matches:
        _fail(f"rank {rank} routed input stream hashes do not match")
    if result.get("post_matches") != expected_post_matches:
        _fail(f"rank {rank} routed post-load hashes do not match")
    return expected_experts


def _metric(line: str, name: str) -> float:
    match = re.search(rf"(?:^| - ){re.escape(name)}:([^\s]+)", line)
    if match is None:
        _fail(f"actor-update metric line has no {name}")
    try:
        value = float(match.group(1))
    except ValueError as error:
        raise WeightCompareVerificationError(
            f"actor-update metric {name} is not numeric"
        ) from error
    if not math.isfinite(value):
        _fail(f"actor-update metric {name} is not finite")
    return value


def _validate_runtime_log(path: Path) -> dict[str, Any]:
    text = ANSI_ESCAPE.sub("", path.read_text(encoding="utf-8", errors="strict"))
    lines = text.splitlines()
    captures: dict[int, int] = {}
    comparisons: dict[int, dict[str, Any]] = {}
    routed: dict[int, dict[str, Any]] = {}
    passes: dict[int, tuple[int, int]] = {}

    for line in lines:
        capture_match = CAPTURE_RE.search(line)
        if capture_match:
            _add_rank_record(
                captures,
                int(capture_match.group("rank")),
                int(capture_match.group("params")),
                "HF sample capture",
            )
        compare_match = COMPARE_RE.search(line)
        if compare_match:
            record = {
                "total": int(compare_match.group("total")),
                "matched": int(compare_match.group("matched")),
                "mismatched": int(compare_match.group("mismatched")),
                "category_totals": _literal(
                    compare_match.group("category_totals"),
                    "category totals",
                    dict,
                ),
                "category_mismatches": _literal(
                    compare_match.group("category_mismatches"),
                    "category mismatches",
                    dict,
                ),
                "missing": _literal(
                    compare_match.group("missing"), "missing weights", list
                ),
                "sample_errors": _literal(
                    compare_match.group("sample_errors"), "sample errors", list
                ),
                "mismatch_samples": _literal(
                    compare_match.group("mismatch_samples"),
                    "mismatch samples",
                    list,
                ),
            }
            _add_rank_record(
                comparisons,
                int(compare_match.group("rank")),
                record,
                "sample comparison",
            )
        routed_match = ROUTED_RE.search(line)
        if routed_match:
            _add_rank_record(
                routed,
                int(routed_match.group("rank")),
                _literal(routed_match.group("result"), "routed result", dict),
                "routed comparison",
            )
        pass_match = PASS_RE.search(line)
        if pass_match:
            _add_rank_record(
                passes,
                int(pass_match.group("rank")),
                (int(pass_match.group("total")), int(pass_match.group("streams"))),
                "PASS",
            )

    for records, context in (
        (captures, "HF sample capture"),
        (comparisons, "sample comparison"),
        (routed, "routed comparison"),
        (passes, "PASS"),
    ):
        _require_all_ranks(records, context)

    for rank in range(WORLD_SIZE):
        if captures[rank] != EXPECTED_PARAMETERS_PER_RANK:
            _fail(f"rank {rank} captured {captures[rank]} parameters")
        comparison = comparisons[rank]
        if (
            comparison["total"] != EXPECTED_PARAMETERS_PER_RANK
            or comparison["matched"] != EXPECTED_PARAMETERS_PER_RANK
            or comparison["mismatched"] != 0
        ):
            _fail(f"rank {rank} parameter comparison did not match exactly")
        if comparison["category_totals"] != EXPECTED_CATEGORY_TOTALS:
            _fail(f"rank {rank} category totals differ")
        for key in (
            "category_mismatches",
            "missing",
            "sample_errors",
            "mismatch_samples",
        ):
            if comparison[key] not in ({}, []):
                _fail(f"rank {rank} has nonempty {key}")
        _validate_routed_record(rank, routed[rank])
        if passes[rank] != (
            EXPECTED_PARAMETERS_PER_RANK,
            EXPECTED_ROUTED_STREAMS_PER_RANK,
        ):
            _fail(f"rank {rank} PASS record differs")

    covered_experts = sorted(
        expert
        for rank in range(WORLD_SIZE)
        for expert in _validate_routed_record(rank, routed[rank])
    )
    if covered_experts != list(range(64)):
        _fail("layer-1 routed comparison does not cover experts 0 through 63")

    update_lines = [
        line
        for line in lines
        if "step:1 - actor/" in line and "training/global_step:" in line
    ]
    if len(update_lines) != 1:
        _fail(f"expected one actor-update metric line, found {len(update_lines)}")
    update_line = update_lines[0]
    actor_lr = _metric(update_line, "actor/lr")
    grad_norm = _metric(update_line, "actor/grad_norm")
    global_step = _metric(update_line, "training/global_step")
    training_epoch = _metric(update_line, "training/epoch")
    aborted_ratio = _metric(update_line, "response/aborted_ratio")
    update_actor_s = _metric(update_line, "timing_s/update_actor")
    response_mean = _metric(update_line, "response_length/mean")
    response_min = _metric(update_line, "response_length/min")
    response_max = _metric(update_line, "response_length/max")
    total_tokens = _metric(update_line, "perf/total_num_tokens")
    if not math.isclose(actor_lr, 1.0e-6, rel_tol=0.0, abs_tol=1.0e-15):
        _fail(f"actor learning rate is {actor_lr}, expected 1e-6")
    if grad_norm < 0.0:
        _fail("actor gradient norm is negative")
    if global_step != 1.0 or training_epoch != 0.0:
        _fail("actor update is not the single expected epoch-0 step")
    if aborted_ratio != 0.0:
        _fail(f"response abort ratio is {aborted_ratio}")
    if update_actor_s <= 0.0:
        _fail("actor update time is not positive")
    if (response_mean, response_min, response_max) != (32.0, 32.0, 32.0):
        _fail("runtime response-length metrics do not equal 32 tokens")

    epoch_matches = [
        (index, match)
        for index, line in enumerate(lines)
        if (match := re.search(r"Epoch 0 completed in ([0-9.eE+-]+) seconds\.", line))
    ]
    if len(epoch_matches) != 1:
        _fail(f"expected one epoch completion record, found {len(epoch_matches)}")
    epoch_index, epoch_match = epoch_matches[0]
    epoch_seconds = float(epoch_match.group(1))
    if not math.isfinite(epoch_seconds) or epoch_seconds <= 0.0:
        _fail("epoch completion time is invalid")

    # Torch Dynamo repeats its metrics_context prefix on every line of the known
    # nonfatal metrics traceback. TBE process-manager noise is permitted only
    # after the successful epoch boundary, so it cannot hide a training failure.
    unexpected_fatal: list[str] = []
    tbe_cleanup_text = (
        "[ERROR] TBE Subprocess[task_distribute] raise error[], "
        "main process disappeared!"
    )
    for index, line in enumerate(lines):
        if not FATAL_RE.search(line):
            continue
        if index <= epoch_index and "torch/_dynamo/metrics_context.py" in line:
            continue
        if index > epoch_index:
            if tbe_cleanup_text in line:
                continue
            if re.search(r"ProcessLookupError: \[Errno 3\] No such process$", line):
                continue
            if (
                "Traceback (most recent call last):" in line
                and index > 0
                and re.search(r"Process ForkServerProcess-\d+:$", lines[index - 1])
            ):
                continue
        unexpected_fatal.append(line)
    if unexpected_fatal:
        position = "before" if lines.index(unexpected_fatal[0]) <= epoch_index else "after"
        _fail(
            f"unexpected fatal runtime marker {position} epoch completion: "
            + unexpected_fatal[0]
        )
    oom_lines = [line for line in lines if OOM_RE.search(line)]
    if oom_lines:
        _fail("runtime log contains an OOM marker: " + oom_lines[0])
    preemption_lines = [line for line in lines if PREEMPT_RE.search(line)]
    if preemption_lines:
        _fail("runtime log contains a preemption marker: " + preemption_lines[0])

    return {
        "rank_count": WORLD_SIZE,
        "parameters_per_rank": EXPECTED_PARAMETERS_PER_RANK,
        "parameter_tensor_comparisons": (
            WORLD_SIZE * EXPECTED_PARAMETERS_PER_RANK
        ),
        "deterministic_samples_per_tensor_max": 33,
        "category_totals_per_rank": EXPECTED_CATEGORY_TOTALS,
        "category_mismatches": 0,
        "layer1_routed_experts_covered": covered_experts,
        "layer1_routed_stream_hash_matches": (
            WORLD_SIZE * EXPECTED_ROUTED_STREAMS_PER_RANK
        ),
        "layer1_post_load_tensor_hash_matches": (
            WORLD_SIZE * EXPECTED_ROUTED_STREAMS_PER_RANK
        ),
        "actor_update": {
            "steps": 1,
            "learning_rate": actor_lr,
            "gradient_norm": grad_norm,
            "update_seconds": update_actor_s,
            "response_aborted_ratio": aborted_ratio,
            "total_tokens_including_prompts": int(total_tokens),
        },
        "epoch_seconds": epoch_seconds,
        "oom_markers": 0,
        "preemption_markers": 0,
        "allowed_noise": {
            "torch_dynamo_metrics_trace_lines": sum(
                "torch/_dynamo/metrics_context.py" in line for line in lines
            ),
            "post_completion_tbe_cleanup_lines": sum(
                index > epoch_index
                and tbe_cleanup_text in line
                for index, line in enumerate(lines)
            ),
            "post_completion_process_lookup_errors": sum(
                index > epoch_index
                and re.search(
                    r"ProcessLookupError: \[Errno 3\] No such process$", line
                )
                is not None
                for index, line in enumerate(lines)
            ),
        },
    }


def _validate_rollout(epoch_dir: Path) -> dict[str, Any]:
    rollout_path = _require_file(
        epoch_dir / "rollout_data" / "1.jsonl", "rollout JSONL"
    )
    length_path = _require_file(
        epoch_dir / "rollout_length" / "length_1.txt", "rollout length file"
    )
    rows: list[dict[str, Any]] = []
    with rollout_path.open(encoding="utf-8") as source:
        for ordinal, line in enumerate(source):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise WeightCompareVerificationError(
                    f"invalid rollout JSON at row {ordinal}: {error}"
                ) from error
            if not isinstance(row, dict):
                _fail(f"rollout row {ordinal} is not an object")
            rows.append(row)
    if len(rows) != EXPECTED_ROWS:
        _fail(f"rollout has {len(rows)} rows, expected {EXPECTED_ROWS}")

    request_keys: list[tuple[int, str]] = []
    rank_counts: Counter[int] = Counter()
    rank_request_ids: dict[int, set[str]] = {
        rank: set() for rank in range(WORLD_SIZE)
    }
    for ordinal, row in enumerate(rows):
        if row.get("step") != 1:
            _fail(f"rollout row {ordinal} has the wrong step")
        if row.get("decoded_response_length") != 32:
            _fail(f"rollout row {ordinal} has the wrong decoded length")
        if row.get("response_finish_reason") != "length":
            _fail(f"rollout row {ordinal} did not finish at the length cap")
        responses = row.get("responses")
        response_mask = row.get("response_mask")
        if not isinstance(responses, list) or len(responses) != 32:
            _fail(f"rollout row {ordinal} has an invalid response tensor")
        if not isinstance(response_mask, list) or response_mask != [1] * 32:
            _fail(f"rollout row {ordinal} has an invalid response mask")
        if not isinstance(row.get("output"), str) or not row["output"].strip():
            _fail(f"rollout row {ordinal} has no visible output")
        score = row.get("score")
        if isinstance(score, bool) or not isinstance(score, (int, float)):
            _fail(f"rollout row {ordinal} has a nonnumeric score")
        if not math.isfinite(float(score)):
            _fail(f"rollout row {ordinal} has a nonfinite score")
        request_id = row.get("request_id")
        rank = row.get("rollout_rank")
        if not isinstance(request_id, str):
            _fail(f"rollout row {ordinal} has no string request ID")
        if isinstance(rank, bool) or not isinstance(rank, int):
            _fail(f"rollout row {ordinal} has no integer rollout rank")
        if rank not in range(WORLD_SIZE):
            _fail(f"rollout row {ordinal} has invalid rollout rank {rank}")
        request_keys.append((rank, request_id))
        rank_counts[rank] += 1
        rank_request_ids[rank].add(request_id)
    if len(set(request_keys)) != EXPECTED_ROWS:
        _fail("rollout rank and request-ID pairs are not unique")
    if rank_counts != Counter({rank: 2 for rank in range(WORLD_SIZE)}):
        _fail(f"rollout rank distribution differs: {dict(sorted(rank_counts.items()))}")
    for rank, request_ids in rank_request_ids.items():
        if request_ids != {"0", "1"}:
            _fail(f"rollout rank {rank} request IDs differ: {sorted(request_ids)}")

    try:
        lengths = [
            int(line)
            for line in length_path.read_text(encoding="utf-8").splitlines()
        ]
    except ValueError as error:
        raise WeightCompareVerificationError(
            "rollout length file contains a noninteger"
        ) from error
    if lengths != [32] * EXPECTED_ROWS:
        _fail("rollout length file is not exactly 32 rows of 32 tokens")
    return {
        "responses": EXPECTED_ROWS,
        "responses_per_rank": 2,
        "decoded_tokens_per_response": 32,
        "finish_reason": "length",
        "aborted_responses": 0,
        "artifacts": {
            "rollout_jsonl": _artifact(rollout_path),
            "rollout_lengths": _artifact(length_path),
        },
    }


def verify_run(run_root: Path) -> dict[str, Any]:
    run_root = _require_directory(run_root.resolve(), "run root")
    complete_path = _require_file(run_root / "COMPLETE", "COMPLETE marker")
    values = _parse_complete(complete_path)
    epoch_dir = _require_directory(
        run_root / "epoch_000_mode0_probe", "epoch directory"
    )
    log_dir = _require_directory(epoch_dir / "logs", "runtime log directory")
    log_paths = [path for path in log_dir.glob("*.txt") if path.is_file()]
    if len(log_paths) != 1:
        _fail(f"expected one runtime log, found {len(log_paths)}")
    log_path = _require_file(log_paths[0], "runtime log")

    model = _validate_model_assets(values)
    runtime = _validate_runtime_log(log_path)
    rollout = _validate_rollout(epoch_dir)
    artifacts = {
        "complete": _artifact(complete_path),
        "runtime_log": _artifact(log_path),
        **rollout.pop("artifacts"),
    }
    bundle_sha256 = hashlib.sha256(
        json.dumps(
            {name: item["sha256"] for name, item in sorted(artifacts.items())},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
    ).hexdigest()
    return {
        "schema_version": 1,
        "status": "PASS",
        "protocol": "DeepSeek-V2-Lite-Chat initial online-sync weight comparison",
        "run_root": str(run_root),
        "model": model,
        "complete_contract": {
            key: values[key]
            for key in (
                "TRAINING_STEPS",
                "TRAIN_BATCH_SIZE",
                "MAX_PROMPT_LENGTH",
                "MAX_RESPONSE_LENGTH",
                "ROLLOUT_N",
                "ROLLOUT_MAX_NUM_BATCHED_TOKENS",
                "ROLLOUT_MAX_NUM_SEQS",
                "EXPECTED_ROWS",
                "ROLLOUT_LOAD_FORMAT",
                "PRESERVE_INITIAL_HF_WEIGHTS",
                "COMPARE_ONLINE_SYNC_TO_HF",
            )
        },
        "weight_comparison": runtime,
        "rollout": rollout,
        "artifacts": artifacts,
        "artifact_bundle_sha256": bundle_sha256,
        "claim_boundary": {
            "supports": [
                "On all 16 rollout ranks, the initial Megatron-to-vLLM online sync matched the pinned Hugging Face model in shape, dtype, and up to 33 deterministic values for each of 324 named parameter tensors.",
                "Full-tensor hashes matched the incoming and post-load gate and up projections for every local layer-1 routed expert, covering experts 0 through 63.",
                "The synchronized model completed one 32-response short generation and one finite actor update without an observed OOM, preemption, or aborted response.",
            ],
            "does_not_support": [
                "The sampled comparison is not a full elementwise proof for every parameter tensor.",
                "The full-tensor routed check covers layer-1 w13 gate and up projections rather than every routed tensor in every layer.",
                "The smoke run does not validate post-update weight reload, long-response performance, adaptive shrinking, sidecar execution, or training quality.",
                "The run marker does not bind an execution-code hash, so source provenance must be recorded separately in the experiment ledger.",
            ],
        },
    }


def _write_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        help="defaults to RUN_ROOT/WEIGHT_COMPARE_SUMMARY.json",
    )
    args = parser.parse_args()
    output = args.output or args.run_root / "WEIGHT_COMPARE_SUMMARY.json"
    try:
        summary = verify_run(args.run_root)
        _write_atomic(output, summary)
    except (OSError, WeightCompareVerificationError) as error:
        parser.exit(2, f"weight-comparison verification failed: {error}\n")
    print(json.dumps(summary, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
