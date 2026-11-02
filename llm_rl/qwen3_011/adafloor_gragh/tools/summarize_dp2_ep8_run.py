#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path


PREEMPT_RE = re.compile(r"preempting request|request preempted", re.I)
OOM_RE = re.compile(
    r"NPU out of memory|Memory_Allocation_Failure|Failed to allocate.*NPU memory",
    re.I,
)
ABORT_RE = re.compile(r"response/aborted_ratio:(0\.[0-9]*[1-9]|[1-9])")
FLOAT = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
TIMESTAMP_RE = re.compile(
    r"(?:(\d{4})-)?(\d{2})-(\d{2})[ T](\d{2}):(\d{2}):(\d{2})(?:,(\d{3}))?"
)
ROLLOUT_RE = re.compile(r"rollout_output_time_s:\s*(%s)" % FLOAT)
SHRINK_DONE_RE = re.compile(
    r"Elastic parallel shrink (?:rpc done: global_rank|phase breakdown: rank)="
    r"(\d+) active_ranks=\[([^]]*)\]"
)
ROLLOUT_CALL_DONE_RE = re.compile(r"megatron_rollout_call_done rank=(\d+)")
TRAINING_BOUNDARY_CLEANUP_RE = re.compile(
    r"\[Rank\s+(\d+)\s+\|[^]]*\].*"
    r"Mode1 training-boundary MoE runtime cleanup"
)
EXPECTED_TRANSITION_SIZES = {
    8: [],
    4: [4],
    2: [4, 2],
}


def _latest_log(run_dir: Path) -> Path:
    logs = sorted(
        (run_dir / "logs").glob("*.txt"),
        key=lambda path: path.stat().st_mtime,
    )
    if not logs:
        raise FileNotFoundError(f"no training log under {run_dir / 'logs'}")
    return logs[-1]


def _step_files(directory: Path, pattern: str) -> list[Path]:
    files = list(directory.glob(pattern))
    files.sort(key=lambda path: int(re.findall(r"\d+", path.stem)[-1]))
    return files


def _timestamp(line: str) -> float | None:
    match = TIMESTAMP_RE.search(line)
    if not match:
        return None
    return datetime(
        int(match.group(1) or 2026),
        int(match.group(2)),
        int(match.group(3)),
        int(match.group(4)),
        int(match.group(5)),
        int(match.group(6)),
        microsecond=int(match.group(7) or 0) * 1000,
    ).timestamp()


def _runtime_events(log_text: str) -> tuple[
        list[tuple[float, float]],
        list[tuple[float, int, tuple[int, ...]]],
        list[tuple[float, int]],
        list[tuple[float, int]],
]:
    rollout_events = []
    shrink_events = []
    call_done_events = []
    cleanup_events = []
    for line in log_text.replace("\r", "\n").splitlines():
        timestamp = _timestamp(line)
        if timestamp is None:
            continue
        rollout_match = ROLLOUT_RE.search(line)
        if rollout_match:
            rollout_events.append((timestamp, float(rollout_match.group(1))))
        shrink_match = SHRINK_DONE_RE.search(line)
        if shrink_match:
            active = tuple(
                int(value) for value in re.findall(r"\d+", shrink_match.group(2))
            )
            shrink_events.append(
                (timestamp, int(shrink_match.group(1)), active))
        call_done_match = ROLLOUT_CALL_DONE_RE.search(line)
        if call_done_match:
            call_done_events.append((timestamp, int(call_done_match.group(1))))
        cleanup_match = TRAINING_BOUNDARY_CLEANUP_RE.search(line)
        if cleanup_match:
            cleanup_events.append((timestamp, int(cleanup_match.group(1))))
    return rollout_events, shrink_events, call_done_events, cleanup_events


def _rank_seconds_from_runtime(
    rollout_end: float,
    rollout_duration: float,
    previous_rollout_end: float,
    shrink_events: list[tuple[float, int, tuple[int, ...]]],
    call_done_events: list[tuple[float, int]],
    cleanup_events: list[tuple[float, int]] | None = None,
) -> dict[str, object]:
    rollout_start = rollout_end - rollout_duration
    window_start = max(previous_rollout_end, rollout_start - 1.0)
    step_shrinks = [
        event for event in shrink_events
        if window_start < event[0] <= rollout_end
    ]
    step_call_done = [
        event for event in call_done_events
        if window_start < event[0] <= rollout_end
    ]
    if not step_call_done:
        step_call_done = [
            event for event in (cleanup_events or [])
            if window_start < event[0] <= rollout_end
        ]
    done_by_rank = defaultdict(list)
    for timestamp, rank in step_call_done:
        done_by_rank[rank].append(timestamp)

    worker_finish = []
    intra_worker_area = 0.0
    transitions_by_worker = []
    for worker_id in range(2):
        original = tuple(range(worker_id * 8, (worker_id + 1) * 8))
        finish_candidates = [
            max(done_by_rank[rank]) for rank in original if done_by_rank[rank]
        ]
        finish = max(finish_candidates) if finish_candidates else rollout_end
        worker_finish.append(finish)

        grouped = defaultdict(lambda: defaultdict(list))
        original_set = set(original)
        for timestamp, rank, active in step_shrinks:
            if set(active).issubset(original_set):
                grouped[active][rank].append(timestamp)
        previous = original
        worker_transitions = []
        for active, events_by_rank in sorted(
                grouped.items(),
                key=lambda item: min(
                    min(values) for values in item[1].values()),
        ):
            if len(active) >= len(previous) or not set(active).issubset(previous):
                continue
            missing = [rank for rank in previous if rank not in events_by_rank]
            if missing:
                raise RuntimeError(
                    f"worker {worker_id} transition to {active} is missing "
                    f"completion events for ranks {missing}")
            coordinated = max(
                max(events_by_rank[rank]) for rank in previous)
            exited = tuple(rank for rank in previous if rank not in active)
            released = len(exited) * max(0.0, finish - coordinated)
            intra_worker_area += released
            worker_transitions.append({
                "target_global_ranks": list(active),
                "exited_global_ranks": list(exited),
                "coordinated_timestamp": coordinated,
                "released_rank_seconds_until_worker_finish": released,
            })
            previous = active
        transitions_by_worker.append(worker_transitions)

    # TLT-style reuse ends when the slower worker finishes generation. Keep the
    # later VERL result-collection interval separate from cross-worker slack.
    global_worker_finish = max(worker_finish)
    worker_level_area = sum(
        8.0 * max(0.0, global_worker_finish - finish)
        for finish in worker_finish
    )
    post_worker_control_area = (
        16.0 * max(0.0, rollout_end - global_worker_finish)
    )
    return {
        "rollout_wall_seconds": rollout_duration,
        "worker_finish_offset_seconds": [
            finish - rollout_start for finish in worker_finish
        ],
        "tlt_like_worker_level_rank_seconds": worker_level_area,
        "adafloor_intra_worker_rank_seconds": intra_worker_area,
        "total_hierarchical_rank_seconds": (
            worker_level_area + intra_worker_area),
        "post_worker_control_rank_seconds": post_worker_control_area,
        "transitions_by_worker": transitions_by_worker,
    }


def _actual_area(
    rank_finish: list[float],
    stage_survivor_ranks: list[list[int]],
) -> float:
    step_finish = max(rank_finish)
    previous = set(range(8))
    area = 0.0
    for survivors_raw in stage_survivor_ranks:
        survivors = set(map(int, survivors_raw))
        exiting = previous - survivors
        if exiting:
            threshold = max(rank_finish[rank] for rank in exiting)
            area += len(exiting) * max(0.0, step_finish - threshold)
        previous = survivors
    return area


def _validate_runtime_floors(
    workers: list[dict],
    transitions_by_worker: list[list[dict]],
    step_index: int,
) -> None:
    if len(transitions_by_worker) != len(workers):
        raise RuntimeError(
            f"step {step_index} has {len(transitions_by_worker)} runtime "
            f"workers, expected {len(workers)}")
    for worker_id, (worker, transitions) in enumerate(
            zip(workers, transitions_by_worker, strict=True)):
        floor = int(worker["selected_floor"])
        expected = EXPECTED_TRANSITION_SIZES[floor]
        actual = [
            len(transition["target_global_ranks"])
            for transition in transitions
        ]
        if actual != expected:
            raise RuntimeError(
                f"step {step_index} worker {worker_id} planned floor {floor} "
                f"requires runtime transitions {expected}, observed {actual}")


def summarize(run_dir: Path, plan_file: Path, expected_steps: int) -> dict:
    plans = json.loads(plan_file.read_text(encoding="utf-8"))
    if len(plans) != expected_steps:
        raise RuntimeError(
            f"plan has {len(plans)} steps, expected {expected_steps}")
    rollout_files = _step_files(run_dir / "rollout_data", "*.jsonl")
    length_files = _step_files(run_dir / "rollout_length", "length_*.txt")
    if len(rollout_files) != expected_steps or len(length_files) != expected_steps:
        raise RuntimeError(
            f"run has {len(rollout_files)} rollout files and "
            f"{len(length_files)} length files, expected {expected_steps}")

    log_file = _latest_log(run_dir)
    log_text = log_file.read_text(encoding="utf-8", errors="replace")
    preemptions = len(PREEMPT_RE.findall(log_text))
    oom_events = len(OOM_RE.findall(log_text))
    aborted = bool(ABORT_RE.search(log_text))
    if preemptions or oom_events or aborted:
        raise RuntimeError(
            "strict run validation failed with "
            f"preemptions={preemptions} oom_events={oom_events} "
            f"aborted_response={aborted} log={log_file}")
    rollout_events, shrink_events, call_done_events, cleanup_events = (
        _runtime_events(log_text)
    )
    if len(rollout_events) != expected_steps:
        raise RuntimeError(
            f"log has {len(rollout_events)} rollout timing events, "
            f"expected {expected_steps}")

    step_summaries = []
    previous_rollout_end = -float("inf")
    for step_index, (rollout_file, length_file, plan, rollout_event) in enumerate(
            zip(rollout_files, length_files, plans, rollout_events, strict=True),
            start=1):
        rollout_rows = sum(1 for line in rollout_file.open(
            "r", encoding="utf-8") if line.strip())
        lengths = [
            float(line.strip())
            for line in length_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if rollout_rows != 512 or len(lengths) != 512:
            raise RuntimeError(
                f"step {step_index} has rollout_rows={rollout_rows} "
                f"lengths={len(lengths)}, expected 512")

        rank_finish = [
            max(lengths[rank * 32:(rank + 1) * 32])
            for rank in range(16)
        ]
        worker_finish = [max(rank_finish[:8]), max(rank_finish[8:])]
        global_finish = max(worker_finish)
        worker_level_area = sum(
            8.0 * (global_finish - finish) for finish in worker_finish)
        intra_worker_area = 0.0
        workers = plan.get("worker_plans", [])
        if len(workers) != 2:
            raise RuntimeError(f"step {step_index} has no two-worker plan")
        for worker_id, worker in enumerate(workers):
            local_finish = rank_finish[worker_id * 8:(worker_id + 1) * 8]
            intra_worker_area += _actual_area(
                local_finish,
                worker["role_plan"]["stage_survivor_ranks"],
            )
        runtime_area = _rank_seconds_from_runtime(
            rollout_event[0],
            rollout_event[1],
            previous_rollout_end,
            shrink_events,
            call_done_events,
            cleanup_events,
        )
        _validate_runtime_floors(
            workers,
            runtime_area["transitions_by_worker"],
            step_index,
        )
        step_summaries.append({
            "step": step_index,
            "worker_finish_token_proxy": worker_finish,
            "global_finish_token_proxy": global_finish,
            "tlt_like_worker_level_area_token_proxy": worker_level_area,
            "adafloor_intra_worker_area_token_proxy": intra_worker_area,
            "total_hierarchical_area_token_proxy": (
                worker_level_area + intra_worker_area),
            "selected_floors": [
                int(worker["selected_floor"]) for worker in workers
            ],
            **runtime_area,
        })
        previous_rollout_end = rollout_event[0]

    return {
        "schema_version": 1,
        "run_dir": str(run_dir),
        "plan_file": str(plan_file),
        "log_file": str(log_file),
        "validation": {
            "steps": expected_steps,
            "responses_per_step": 512,
            "preemptions": preemptions,
            "oom_events": oom_events,
            "aborted_response": aborted,
        },
        "steps": step_summaries,
        "totals": {
            "tlt_like_worker_level_area_token_proxy": sum(
                step["tlt_like_worker_level_area_token_proxy"]
                for step in step_summaries),
            "adafloor_intra_worker_area_token_proxy": sum(
                step["adafloor_intra_worker_area_token_proxy"]
                for step in step_summaries),
            "total_hierarchical_area_token_proxy": sum(
                step["total_hierarchical_area_token_proxy"]
                for step in step_summaries),
            "tlt_like_worker_level_rank_seconds": sum(
                step["tlt_like_worker_level_rank_seconds"]
                for step in step_summaries),
            "adafloor_intra_worker_rank_seconds": sum(
                step["adafloor_intra_worker_rank_seconds"]
                for step in step_summaries),
            "total_hierarchical_rank_seconds": sum(
                step["total_hierarchical_rank_seconds"]
                for step in step_summaries),
            "post_worker_control_rank_seconds": sum(
                step["post_worker_control_rank_seconds"]
                for step in step_summaries),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--plan-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-steps", type=int, default=5)
    args = parser.parse_args()
    payload = summarize(args.run_dir, args.plan_file, args.expected_steps)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        "[dp2-ep8 summary] strict validation passed "
        f"output={args.output} totals={payload['totals']}"
    )


if __name__ == "__main__":
    main()
