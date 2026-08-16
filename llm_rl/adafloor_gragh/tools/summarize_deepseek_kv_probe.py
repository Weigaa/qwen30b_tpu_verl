#!/usr/bin/env python3
"""Validate and summarize a model-specific DeepSeek KV capacity probe."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path

try:
    from audit_deepseek_n_f4_formal_run import (
        AuditError,
        _validate_lifecycle,
        parse_runtime_log,
    )
except ModuleNotFoundError:
    from tools.audit_deepseek_n_f4_formal_run import (
        AuditError,
        _validate_lifecycle,
        parse_runtime_log,
    )


ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
PID_RE = re.compile(r"\(WorkerDict pid=(\d+)\)")
START_RE = re.compile(
    r"Mode1 adaptive KV resize phase=start .*?target_floor=(\d+)"
)
RANK_RE = re.compile(r"Mode1 adaptive KV resize memory: rank=(\d+)\b")
DONE_RE = re.compile(
    r"Mode1 adaptive KV resize phase=plan_new_kv_done .*?new_tokens=(\d+)"
)
PREEMPT_RE = re.compile(r"preempting request|request preempted", re.I)
OOM_RE = re.compile(
    r"NPU out of memory|Memory_Allocation_Failure|"
    r"Failed to allocate[^\r\n]*NPU memory|OutOfMemoryError|"
    r"ACL_ERROR_RT_MEMORY_ALLOCATION",
    re.I,
)
ABORT_RE = re.compile(r"response/aborted_ratio:([0-9.eE+-]+)")
RELEASE_AREA_UNIT = "rank_token_proxy"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lifecycle", required=True)
    parser.add_argument("--floor", required=True, type=int)
    parser.add_argument("--world-size", default=16, type=int)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--log", required=True, type=Path)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--execution-profile", required=True)
    parser.add_argument("--runtime-profile", required=True)
    parser.add_argument("--runtime-profile-sha256", required=True)
    parser.add_argument("--execution-code-sha256", required=True)
    parser.add_argument("--max-prompt-length", required=True, type=int)
    parser.add_argument("--max-response-length", required=True, type=int)
    parser.add_argument("--max-num-batched-tokens", required=True, type=int)
    parser.add_argument("--max-num-seqs", required=True, type=int)
    parser.add_argument("--tail-guard-min-cap", required=True, type=int)
    parser.add_argument("--tail-guard-round-to", required=True, type=int)
    parser.add_argument("--gpu-memory-utilization", required=True, type=float)
    parser.add_argument("--enforce-eager", required=True)
    parser.add_argument("--block-size", required=True, type=int)
    parser.add_argument("--common-epoch0-root", required=True, type=Path)
    parser.add_argument("--planning-history-root", required=True, type=Path)
    parser.add_argument("--planning-history-sha256", required=True)
    parser.add_argument("--planning-history-manifest-sha256", required=True)
    parser.add_argument("--planning-trigger-subset-sha256", required=True)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _probe_plan(run_root: Path, floor: int) -> tuple[dict, Path]:
    summaries = sorted(run_root.rglob("length_sorted_rank_plan_summary.json"))
    if len(summaries) != 1:
        raise RuntimeError(
            f"expected one rank-plan summary under {run_root}, found {len(summaries)}"
        )
    plans = json.loads(summaries[0].read_text(encoding="utf-8"))
    if not isinstance(plans, list) or not plans:
        raise RuntimeError(f"invalid rank-plan summary: {summaries[0]}")
    if len(plans) != 1 or not isinstance(plans[0], dict):
        raise RuntimeError(f"expected one probe plan in {summaries[0]}")
    plan = plans[0]
    if plan.get("feasible") is not True:
        raise RuntimeError(f"probe plan is not feasible: {summaries[0]}")
    if int(plan.get("selected_floor", -1)) != floor:
        raise RuntimeError(
            f"forced floor mismatch: expected {floor}, "
            f"observed {plan.get('selected_floor')!r}"
        )
    release_area = float(plan.get("release_area", -1))
    if not math.isfinite(release_area) or release_area < 0:
        raise RuntimeError(f"invalid probe release area: {release_area}")
    if plan.get("release_area_unit") not in (None, RELEASE_AREA_UNIT):
        raise RuntimeError("invalid probe release area unit")
    thresholds = plan.get("schedule_thresholds")
    predicted_exit = float(plan.get("predicted_step_exit", -1))
    if floor < 16:
        if release_area <= 0:
            raise RuntimeError(
                f"floor{floor} probe does not provide positive release area"
            )
        if not isinstance(thresholds, list) or not thresholds:
            raise RuntimeError(f"floor{floor} probe has no release thresholds")
        numeric_thresholds = [float(value) for value in thresholds]
        if (
            not math.isfinite(predicted_exit)
            or any(not math.isfinite(value) for value in numeric_thresholds)
            or max(numeric_thresholds) >= predicted_exit
        ):
            raise RuntimeError(
                f"floor{floor} release thresholds must precede the predicted tail"
            )
    return plan, summaries[0]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _validate_probe_lifecycle(
    log_path: Path, plan: dict, floor: int, lifecycle: str
) -> dict:
    try:
        calls, shrinks, restores, _text = parse_runtime_log(log_path)
    except AuditError as error:
        raise RuntimeError(str(error)) from error
    calls_by_rank = {call.rank: call for call in calls if call.step == 1}
    if len(calls) != 16 or set(calls_by_rank) != set(range(16)):
        raise RuntimeError(
            f"floor{floor} probe must contain one KV resize call per rank"
        )
    if any(call.target_floor != floor for call in calls_by_rank.values()):
        raise RuntimeError(f"floor{floor} probe has a runtime target-floor mismatch")
    last_resize_done = {1: max(call.done_position for call in calls)}
    lifecycle_plan = {
        1: {
            "floor": floor,
            "stages": tuple(int(value) for value in plan["shrink_stages"]),
            "stage_sets": tuple(
                tuple(int(rank) for rank in ranks)
                for ranks in plan["stage_survivor_ranks"]
            ),
        }
    }
    try:
        return _validate_lifecycle(
            shrinks,
            restores,
            lifecycle_plan,
            last_resize_done,
            rank_identity_known=lifecycle.startswith("planned_"),
        )[1]
    except AuditError as error:
        raise RuntimeError(str(error)) from error


def _target_waves(text: str, target_floor: int) -> list[dict[int, int]]:
    current_floor: dict[int, int] = {}
    rank_by_pid: dict[int, int] = {}
    events: list[tuple[int, int]] = []

    for raw_line in text.splitlines():
        line = ANSI_RE.sub("", raw_line)
        pid_match = PID_RE.search(line)
        if pid_match is None:
            continue
        pid = int(pid_match.group(1))
        start_match = START_RE.search(line)
        if start_match is not None:
            current_floor[pid] = int(start_match.group(1))
        rank_match = RANK_RE.search(line)
        if rank_match is not None:
            rank_by_pid[pid] = int(rank_match.group(1))
        done_match = DONE_RE.search(line)
        if done_match is None or current_floor.get(pid) != target_floor:
            continue
        if pid not in rank_by_pid:
            raise RuntimeError(
                f"target-floor KV result for worker pid {pid} has no rank identity"
            )
        events.append((rank_by_pid[pid], int(done_match.group(1))))

    waves: list[dict[int, int]] = []
    wave: dict[int, int] = {}
    for rank, tokens in events:
        if rank in wave:
            waves.append(wave)
            wave = {}
        wave[rank] = tokens
    if wave:
        waves.append(wave)
    return waves


def summarize(
    lifecycle: str,
    floor: int,
    world_size: int,
    run_root: Path,
    log_path: Path,
    model_revision: str = "unknown",
    execution_profile: str = "unknown",
    runtime_profile: str = "unknown",
    runtime_profile_sha256: str = "unknown",
    execution_code_sha256: str = "unknown",
    max_prompt_length: int = 0,
    max_response_length: int = 0,
    max_num_batched_tokens: int = 0,
    max_num_seqs: int = 0,
    gpu_memory_utilization: float = 0.0,
    enforce_eager: str = "unknown",
    block_size: int = 128,
    common_epoch0_root: Path | None = None,
    planning_history_root: Path | None = None,
    planning_history_sha256: str = "unknown",
    planning_history_manifest_sha256: str = "unknown",
    planning_trigger_subset_sha256: str = "unknown",
    tail_guard_min_cap: int = 0,
    tail_guard_round_to: int = 0,
) -> dict:
    text = log_path.read_text(encoding="utf-8", errors="replace")
    if re.fullmatch(r"[0-9a-f]{64}", execution_code_sha256) is None:
        raise RuntimeError("invalid DeepSeek execution code SHA256")
    for label, value in (
        ("planning history", planning_history_sha256),
        ("planning history manifest", planning_history_manifest_sha256),
        ("planning trigger subset", planning_trigger_subset_sha256),
    ):
        if re.fullmatch(r"[0-9a-f]{64}", value) is None:
            raise RuntimeError(f"invalid DeepSeek {label} SHA256")
    if PREEMPT_RE.search(text):
        raise RuntimeError(f"KV preemption found in {log_path}")
    if OOM_RE.search(text):
        raise RuntimeError(f"NPU OOM found in {log_path}")
    aborted_ratios = [float(value) for value in ABORT_RE.findall(text)]
    if aborted_ratios != [0.0]:
        raise RuntimeError(
            f"missing or nonzero aborted response ratio in {log_path}: "
            f"{aborted_ratios}"
        )
    if text.count("rollout_output_time_s:") != 1:
        raise RuntimeError(f"probe did not complete exactly one rollout step: {log_path}")
    if tail_guard_min_cap <= 0 or tail_guard_round_to <= 0:
        raise RuntimeError("probe TailGuard min cap and round-to must be positive")

    plan, plan_path = _probe_plan(run_root, floor)
    raw_plan_response_cap = plan.get("tail_guard_response_cap")
    if (
        isinstance(raw_plan_response_cap, bool)
        or not isinstance(raw_plan_response_cap, int)
        or raw_plan_response_cap <= 0
        or raw_plan_response_cap > max_response_length
    ):
        raise RuntimeError(
            f"invalid actual plan response cap: {raw_plan_response_cap!r}"
        )
    runtime_lifecycle = _validate_probe_lifecycle(
        log_path, plan, floor, lifecycle
    )

    expected_ranks = set(range(world_size))
    waves = _target_waves(text, floor)
    complete_waves = [wave for wave in waves if expected_ranks <= wave.keys()]
    if not complete_waves:
        observed = [sorted(wave) for wave in waves]
        raise RuntimeError(
            f"no complete floor{floor} KV resize wave across ranks "
            f"0..{world_size - 1}; observed={observed}"
        )
    per_rank = {
        rank: min(wave[rank] for wave in complete_waves)
        for rank in sorted(expected_ranks)
    }
    invalid = {
        rank: tokens
        for rank, tokens in per_rank.items()
        if tokens <= 0 or tokens % block_size != 0
    }
    if invalid:
        raise RuntimeError(f"invalid KV capacities: {invalid}")
    planner_train_files = sorted(run_root.rglob("length_sorted_train.parquet"))
    if len(planner_train_files) != 1:
        raise RuntimeError(
            f"expected one planner train artifact, found {len(planner_train_files)}"
        )

    return {
        "lifecycle": lifecycle,
        "floor": floor,
        "world_size": world_size,
        "observed_tokens": min(per_rank.values()),
        "per_rank_tokens": per_rank,
        "complete_target_waves": len(complete_waves),
        "log": str(log_path.resolve()),
        "model_revision": model_revision,
        "execution_profile": execution_profile,
        "runtime_profile": runtime_profile,
        "runtime_profile_sha256": runtime_profile_sha256,
        "execution_code_sha256": execution_code_sha256,
        "max_prompt_length": max_prompt_length,
        "max_response_length": max_response_length,
        "max_num_batched_tokens": max_num_batched_tokens,
        "max_num_seqs": max_num_seqs,
        "probe_tail_guard_min_cap": tail_guard_min_cap,
        "probe_tail_guard_round_to": tail_guard_round_to,
        "gpu_memory_utilization": gpu_memory_utilization,
        "enforce_eager": enforce_eager,
        "block_size": block_size,
        "common_epoch0_root": str(common_epoch0_root.resolve())
        if common_epoch0_root is not None
        else "",
        "planning_history_root": str(planning_history_root.resolve())
        if planning_history_root is not None
        else "",
        "planning_history_sha256": planning_history_sha256,
        "planning_history_manifest_sha256": planning_history_manifest_sha256,
        "planning_trigger_subset_sha256": planning_trigger_subset_sha256,
        "planner_train_artifact": str(planner_train_files[0].resolve()),
        "planner_train_sha256": _sha256(planner_train_files[0]),
        "plan_summary": str(plan_path.resolve()),
        "plan_release_area": float(plan["release_area"]),
        "plan_release_area_unit": RELEASE_AREA_UNIT,
        "plan_schedule_thresholds": plan.get("schedule_thresholds", []),
        "plan_predicted_step_exit": float(plan["predicted_step_exit"]),
        "plan_tail_guard_response_cap": raw_plan_response_cap,
        "actual_plan_response_cap": raw_plan_response_cap,
        "runtime_lifecycle": runtime_lifecycle,
    }


def main() -> int:
    args = _parse_args()
    report = summarize(
        args.lifecycle,
        args.floor,
        args.world_size,
        args.run_root,
        args.log,
        args.model_revision,
        args.execution_profile,
        args.runtime_profile,
        args.runtime_profile_sha256,
        args.execution_code_sha256,
        args.max_prompt_length,
        args.max_response_length,
        args.max_num_batched_tokens,
        args.max_num_seqs,
        args.gpu_memory_utilization,
        args.enforce_eager,
        args.block_size,
        args.common_epoch0_root,
        args.planning_history_root,
        args.planning_history_sha256,
        args.planning_history_manifest_sha256,
        args.planning_trigger_subset_sha256,
        args.tail_guard_min_cap,
        args.tail_guard_round_to,
    )
    output = args.output or args.run_root / "kv_probe_summary.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    per_rank = ",".join(
        f"{rank}:{tokens}" for rank, tokens in report["per_rank_tokens"].items()
    )
    print(
        "DEEPSEEK_KV_PROBE_RESULT "
        f"lifecycle={report['lifecycle']} floor={report['floor']} "
        f"observed_tokens={report['observed_tokens']} "
        f"actual_plan_response_cap={report['actual_plan_response_cap']} "
        f"per_rank_tokens={per_rank} "
        f"complete_target_waves={report['complete_target_waves']} "
        f"summary={output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
