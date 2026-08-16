#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from build_mode1_length_sorted_e2e_plan import (
    PAIR_ADJUSTED_PEAK,
    PAIR_LOAD_GAP,
    PAIR_RANK_LOAD,
    PAIR_RAW_PEAK,
    PAIR_SOURCE_INDICES,
    PAIR_TOKEN_SUM,
    PairMetric,
    RELEASE_AREA_UNIT,
    _ceil_to_multiple,
    _has_perfect_matching_under_adjusted_peak,
    _min_skew_matching_from_edges,
    _pair_metrics,
    _quota_aware_matching_oracle,
    _read_baseline_history,
    _read_ema_baseline_stats,
    _tail_guard_underestimate_ratio,
)
from build_mode1_optimized_rank_plan import PromptStats, _map_stats_to_dataset


GLOBAL_WORLD_SIZE = 16
WORKER_COUNT = 2
WORKER_WORLD_SIZE = 8
PROMPTS_PER_RANK = 2
PROMPTS_PER_WORKER = WORKER_WORLD_SIZE * PROMPTS_PER_RANK
PROMPTS_PER_STEP = WORKER_COUNT * PROMPTS_PER_WORKER
FLOORS = (2, 4, 8)
DEFAULT_FLOOR_KV_CAPS = {
    2: 131072.0,
    4: 280576.0,
    8: 377344.0,
}


@dataclass(frozen=True)
class EP8Plan:
    prompts: list[PromptStats]
    rank_to_source_idx: dict[int, list[int]]
    rank_loads: dict[int, float]
    rank_peak_loads: dict[int, float]
    rank_adjusted_peak_loads: dict[int, float]
    rank_token_sums: dict[int, float]
    selected_floor: int
    theoretical_floor: int
    kv_cap: float
    thresholds: tuple[float, ...]
    quotas: tuple[tuple[float, int], ...]
    release_area: float
    solver: str


def _parse_floor_caps(raw: str) -> dict[int, float]:
    result = dict(DEFAULT_FLOOR_KV_CAPS)
    for item in raw.split(","):
        floor_text, cap_text = item.strip().split(":", 1)
        floor = int(floor_text)
        if floor not in FLOORS:
            raise ValueError(
                f"unsupported EP8 floor {floor}, expected one of {FLOORS}")
        result[floor] = float(cap_text)
    if set(result) != set(FLOORS):
        raise ValueError(f"floor caps must cover {FLOORS}, got {result}")
    return result


def _role_for_floor(floor: int) -> dict[str, Any]:
    if floor == 8:
        stages = [8]
    elif floor == 4:
        stages = [4]
    elif floor == 2:
        stages = [4, 2]
    else:
        raise ValueError(f"invalid EP8 floor {floor}")
    stage_ranks = [
        list(range(WORKER_WORLD_SIZE - stage, WORKER_WORLD_SIZE))
        for stage in stages
    ]
    intermediate = stage_ranks[0]
    final = stage_ranks[-1]
    donor = sorted(set(range(WORKER_WORLD_SIZE)) - set(intermediate))
    wave2 = sorted(set(intermediate) - set(final))
    return {
        "donor_ranks": donor,
        "wave2_ranks": wave2,
        "intermediate_survivor_ranks": intermediate,
        "final_survivor_ranks": final,
        "stage_survivor_ranks": stage_ranks,
        "package_locality_score": 1.0,
        "fallback_reason": "manual_stage_ranks",
    }


def _schedule_candidates(
    floor: int,
    candidate_times: list[float],
    tail_time: float,
) -> list[tuple[tuple[float, ...], tuple[tuple[float, int], ...], float]]:
    schedules = []
    if floor == 8:
        return [((), (), 0.0)]
    if floor == 4:
        for threshold in candidate_times:
            schedules.append((
                (float(threshold),),
                ((float(threshold), 4),),
                4.0 * max(0.0, tail_time - threshold),
            ))
    elif floor == 2:
        for first in candidate_times:
            for second in candidate_times:
                if first > second:
                    continue
                schedules.append((
                    (float(first), float(second)),
                    ((float(first), 4), (float(second), 6)),
                    4.0 * max(0.0, tail_time - first)
                    + 2.0 * max(0.0, tail_time - second),
                ))
    schedules.sort(key=lambda item: (item[2], tuple(-x for x in item[0])),
                   reverse=True)
    return schedules


def _theoretical_floor(pair_metrics: list[PairMetric]) -> int:
    tail = max(item[PAIR_RANK_LOAD] for item in pair_metrics)
    tail_count = sum(
        1 for item in pair_metrics if item[PAIR_RANK_LOAD] >= tail)
    for floor in FLOORS:
        if tail_count <= floor:
            return floor
    return 8


def _solve_floor(
    group: list[PromptStats],
    floor: int,
    kv_cap: float,
    active_peak_safety_factor: float,
    max_response_len: float,
) -> tuple[list[PairMetric], tuple[float, ...],
           tuple[tuple[float, int], ...], float, str] | None:
    metrics = _pair_metrics(
        group,
        active_peak_safety_factor=active_peak_safety_factor,
        max_response_len=max_response_len,
    )
    allowed_edges = {
        edge for edge, item in metrics.items()
        if item[PAIR_ADJUSTED_PEAK] <= kv_cap
    }
    if not _has_perfect_matching_under_adjusted_peak(
            len(group), metrics, kv_cap):
        return None

    if floor == 8:
        matching = _min_skew_matching_from_edges(
            len(group), metrics, allowed_edges)
        if matching is None:
            return None
        pairs = [metrics[edge] for edge in matching]
        pairs.sort(key=lambda item: (
            item[PAIR_RANK_LOAD], item[PAIR_LOAD_GAP],
            item[PAIR_ADJUSTED_PEAK]))
        return pairs, (), (), 0.0, "kv_safe_full_ep8"

    candidate_times = sorted({
        metrics[edge][PAIR_RANK_LOAD] for edge in allowed_edges
    })
    tail_time = max(item.load for item in group)
    for thresholds, quotas, area in _schedule_candidates(
            floor, candidate_times, tail_time):
        if thresholds[-1] >= tail_time:
            continue
        matching = _quota_aware_matching_oracle(
            len(group), metrics, allowed_edges, quotas)
        if matching is None:
            continue
        pairs = [metrics[edge] for edge in matching]
        pairs.sort(key=lambda item: (
            item[PAIR_RANK_LOAD], item[PAIR_LOAD_GAP],
            item[PAIR_ADJUSTED_PEAK]))
        return (
            pairs,
            thresholds,
            quotas,
            float(area),
            "max_release_area_quota_matching",
        )
    return None


def solve_ep8(
    prompts: list[PromptStats],
    floor_caps: dict[int, float],
    active_peak_safety_factor: float,
    max_response_len: float,
) -> EP8Plan:
    if len(prompts) != PROMPTS_PER_WORKER:
        raise ValueError(
            f"EP8 requires {PROMPTS_PER_WORKER} prompts, got {len(prompts)}")

    probe = _solve_floor(
        prompts,
        8,
        floor_caps[8],
        active_peak_safety_factor,
        max_response_len,
    )
    if probe is None:
        max_adjusted = max(
            item[PAIR_ADJUSTED_PEAK]
            for item in _pair_metrics(
                prompts,
                active_peak_safety_factor=active_peak_safety_factor,
                max_response_len=max_response_len,
            ).values()
        )
        raise RuntimeError(
            "EP8 full-world plan is not KV safe. "
            f"floor8_cap={floor_caps[8]:.0f} candidate_peak={max_adjusted:.0f}")
    theoretical_floor = _theoretical_floor(probe[0])

    best = None
    for floor in FLOORS:
        if floor < theoretical_floor:
            continue
        candidate = _solve_floor(
            prompts,
            floor,
            floor_caps[floor],
            active_peak_safety_factor,
            max_response_len,
        )
        if candidate is None:
            continue
        score = (candidate[3], floor)
        if best is None or score > best[0]:
            best = (score, floor, candidate)
    if best is None:
        raise RuntimeError(
            "no strictly KV-safe EP8 floor exists for the worker prompt set")

    _, selected_floor, selected = best
    pairs, thresholds, quotas, area, solver = selected
    rank_to_source_idx = {}
    rank_loads = {}
    rank_peak_loads = {}
    rank_adjusted_peak_loads = {}
    rank_token_sums = {}
    for rank, pair in zip(range(WORKER_WORLD_SIZE), pairs, strict=True):
        rank_to_source_idx[rank] = [
            int(item) for item in pair[PAIR_SOURCE_INDICES]
        ]
        rank_loads[rank] = float(pair[PAIR_RANK_LOAD])
        rank_peak_loads[rank] = float(pair[PAIR_RAW_PEAK])
        rank_adjusted_peak_loads[rank] = float(pair[PAIR_ADJUSTED_PEAK])
        rank_token_sums[rank] = float(pair[PAIR_TOKEN_SUM])

    if any(len(ids) != PROMPTS_PER_RANK
           for ids in rank_to_source_idx.values()):
        raise AssertionError("EP8 plan did not assign two prompts per rank")
    if max(rank_adjusted_peak_loads.values()) > floor_caps[selected_floor]:
        raise AssertionError("selected EP8 plan exceeds its KV cap")
    return EP8Plan(
        prompts=list(prompts),
        rank_to_source_idx=rank_to_source_idx,
        rank_loads=rank_loads,
        rank_peak_loads=rank_peak_loads,
        rank_adjusted_peak_loads=rank_adjusted_peak_loads,
        rank_token_sums=rank_token_sums,
        selected_floor=int(selected_floor),
        theoretical_floor=int(theoretical_floor),
        kv_cap=float(floor_caps[selected_floor]),
        thresholds=tuple(thresholds),
        quotas=tuple(quotas),
        release_area=float(area),
        solver=str(solver),
    )


def _make_steps(
    stats: list[PromptStats],
    steps: int,
    grouping: str,
    seed: int,
) -> list[list[PromptStats]]:
    if len(stats) != steps * PROMPTS_PER_STEP:
        raise ValueError(
            f"expected {steps * PROMPTS_PER_STEP} prompts, got {len(stats)}")
    ordered = list(stats)
    if grouping == "random":
        random.Random(seed).shuffle(ordered)
    elif grouping == "length_sorted":
        ordered.sort(key=lambda item: (item.load, item.source_idx))
    else:
        raise ValueError(f"unsupported grouping {grouping}")
    return [
        ordered[index:index + PROMPTS_PER_STEP]
        for index in range(0, len(ordered), PROMPTS_PER_STEP)
    ]


def _partition_workers(
    step: list[PromptStats],
    grouping: str,
) -> list[list[PromptStats]]:
    if grouping == "random":
        return [step[:PROMPTS_PER_WORKER], step[PROMPTS_PER_WORKER:]]
    ordered = sorted(step, key=lambda item: (item.load, item.source_idx))
    return [ordered[0::2], ordered[1::2]]


def _minimum_kv_safe_pairing_peak(
    prompts: list[PromptStats],
    active_peak_safety_factor: float,
    max_response_len: float,
) -> float:
    metrics = _pair_metrics(
        prompts,
        active_peak_safety_factor=active_peak_safety_factor,
        max_response_len=max_response_len,
    )
    for threshold in sorted({
            float(item[PAIR_ADJUSTED_PEAK]) for item in metrics.values()
    }):
        if _has_perfect_matching_under_adjusted_peak(
                len(prompts), metrics, threshold):
            return threshold
    return float("inf")


def _step_minimum_peaks(
    step: list[PromptStats],
    grouping: str,
    active_peak_safety_factor: float,
    max_response_len: float,
) -> list[float]:
    return [
        _minimum_kv_safe_pairing_peak(
            worker_group,
            active_peak_safety_factor,
            max_response_len,
        )
        for worker_group in _partition_workers(step, grouping)
    ]


def _step_minimum_peaks_cached(
    step: list[PromptStats],
    grouping: str,
    active_peak_safety_factor: float,
    max_response_len: float,
    cache: dict[tuple[int, ...], list[float]],
) -> list[float]:
    key = tuple(int(item.source_idx) for item in step)
    cached = cache.get(key)
    if cached is None:
        cached = _step_minimum_peaks(
            step,
            grouping,
            active_peak_safety_factor,
            max_response_len,
        )
        cache[key] = cached
    return cached


def _repair_score(
    first: list[float],
    second: list[float],
    floor8_cap: float,
    first_step: list[PromptStats],
    second_step: list[PromptStats],
) -> tuple[float, float, float, float]:
    overflows = [
        max(0.0, peak - floor8_cap)
        for result in (first, second) for peak in result
    ]
    infeasible = sum(peak > floor8_cap for peak in (*first, *second))
    return (
        float(infeasible),
        float(sum(overflows)),
        float(max(overflows, default=0.0)),
        max(item.load for item in first_step)
        + max(item.load for item in second_step),
    )


def _repair_infeasible_steps(
    steps: list[list[PromptStats]],
    grouping: str,
    floor_caps: dict[int, float],
    active_peak_safety_factor: float,
    max_response_len: float,
    max_swaps: int,
    candidate_limit: int,
) -> tuple[list[list[EP8Plan]], list[dict[str, Any]]]:
    solve_cache: dict[tuple[int, ...], list[float]] = {}
    results = [
        _step_minimum_peaks_cached(
            step,
            grouping,
            active_peak_safety_factor,
            max_response_len,
            solve_cache,
        )
        for step in steps
    ]
    repairs: list[dict[str, Any]] = []
    for swap_index in range(max_swaps):
        bad_indices = [
            index for index, result in enumerate(results)
            if any(peak > floor_caps[8] for peak in result)
        ]
        if not bad_indices:
            break

        best = None
        for bad_index in reversed(bad_indices):
            bad_order = sorted(
                range(len(steps[bad_index])),
                key=lambda index: (
                    steps[bad_index][index].load,
                    steps[bad_index][index].peak_active_tokens,
                ),
            )[:candidate_limit]
            for previous_index in range(bad_index - 1, -1, -1):
                previous_order = sorted(
                    range(len(steps[previous_index])),
                    key=lambda index: (
                        steps[previous_index][index].load,
                        steps[previous_index][index].peak_active_tokens,
                    ),
                )[:candidate_limit]
                current_score = _repair_score(
                    results[bad_index],
                    results[previous_index],
                    floor_caps[8],
                    steps[bad_index],
                    steps[previous_index],
                )
                for bad_pos in bad_order:
                    outgoing = steps[bad_index][bad_pos]
                    for previous_pos in previous_order:
                        incoming = steps[previous_index][previous_pos]
                        if (incoming.load >= outgoing.load
                                or incoming.peak_active_tokens
                                >= outgoing.peak_active_tokens):
                            continue
                        candidate_bad = list(steps[bad_index])
                        candidate_previous = list(steps[previous_index])
                        candidate_bad[bad_pos] = incoming
                        candidate_previous[previous_pos] = outgoing
                        bad_result = _step_minimum_peaks_cached(
                            candidate_bad,
                            grouping,
                            active_peak_safety_factor,
                            max_response_len,
                            solve_cache,
                        )
                        previous_result = _step_minimum_peaks_cached(
                            candidate_previous,
                            grouping,
                            active_peak_safety_factor,
                            max_response_len,
                            solve_cache,
                        )
                        if (all(peak <= floor_caps[8]
                                for peak in results[previous_index])
                                and any(peak > floor_caps[8]
                                        for peak in previous_result)):
                            continue
                        score = _repair_score(
                            bad_result,
                            previous_result,
                            floor_caps[8],
                            candidate_bad,
                            candidate_previous,
                        )
                        if score[:3] >= current_score[:3]:
                            continue
                        tie_break = (
                            score,
                            bad_index - previous_index,
                            abs(outgoing.load - incoming.load),
                            int(outgoing.source_idx),
                            int(incoming.source_idx),
                        )
                        if best is None or tie_break < best[0]:
                            best = (
                                tie_break,
                                bad_index,
                                previous_index,
                                bad_pos,
                                previous_pos,
                                bad_result,
                                previous_result,
                            )
                            break
                    if best is not None:
                        break
                if best is not None:
                    break
            if best is not None:
                break
        if best is None:
            break

        (
            _tie_break,
            bad_index,
            previous_index,
            bad_pos,
            previous_pos,
            bad_result,
            previous_result,
        ) = best
        outgoing = steps[bad_index][bad_pos]
        incoming = steps[previous_index][previous_pos]
        steps[bad_index][bad_pos], steps[previous_index][previous_pos] = (
            incoming,
            outgoing,
        )
        results[bad_index] = bad_result
        results[previous_index] = previous_result
        repair = {
            "swap": swap_index + 1,
            "infeasible_step": bad_index + 1,
            "previous_step": previous_index + 1,
            "outgoing_source_idx": int(outgoing.source_idx),
            "incoming_source_idx": int(incoming.source_idx),
            "outgoing_load": float(outgoing.load),
            "incoming_load": float(incoming.load),
        }
        repairs.append(repair)
        print(
            "[dp2-ep8 repair] "
            f"swap={swap_index + 1} step{bad_index + 1}:src{outgoing.source_idx} "
            f"<-> step{previous_index + 1}:src{incoming.source_idx} "
            f"loads={outgoing.load:.2f}/{incoming.load:.2f}",
            flush=True,
        )

    failures = []
    for step_index, result in enumerate(results, start=1):
        for worker_id, minimum_peak in enumerate(result):
            if minimum_peak > floor_caps[8]:
                failures.append(
                    f"step={step_index} worker={worker_id} "
                    f"minimum_pairing_peak={minimum_peak:.0f} "
                    f"floor8_cap={floor_caps[8]:.0f}")
    if failures:
        raise RuntimeError(
            "bounded cross-step repair could not produce a KV-safe plan. "
            + "; ".join(failures))

    solved_steps = []
    for step in steps:
        solved_steps.append([
            solve_ep8(
                worker_group,
                floor_caps,
                active_peak_safety_factor,
                max_response_len,
            )
            for worker_group in _partition_workers(step, grouping)
        ])
    return solved_steps, repairs


def _worker_payload(
    worker_id: int,
    plan: EP8Plan,
    row_map: dict[int, int],
    tail_guard_cap: int,
    tail_guard_ratio: float,
    tail_guard_enabled: bool,
) -> dict[str, Any]:
    global_ranks = list(range(
        worker_id * WORKER_WORLD_SIZE,
        (worker_id + 1) * WORKER_WORLD_SIZE,
    ))
    role_plan = _role_for_floor(plan.selected_floor)
    return {
        "worker_id": worker_id,
        "global_ranks": global_ranks,
        "selected_floor": plan.selected_floor,
        "theoretical_floor": plan.theoretical_floor,
        "kv_cap": plan.kv_cap,
        "tail_guard_response_cap": tail_guard_cap,
        "tail_guard_enabled": tail_guard_enabled,
        "tail_guard_ratio": tail_guard_ratio,
        "tail_guard_predicted_step_exit": max(plan.rank_loads.values()),
        "shrink_stages": [len(ranks) for ranks in role_plan["stage_survivor_ranks"]],
        "role_plan": role_plan,
        "schedule_thresholds": list(plan.thresholds),
        "schedule_quotas": [list(item) for item in plan.quotas],
        "release_area": plan.release_area,
        "release_area_unit": RELEASE_AREA_UNIT,
        "rank_matching_solver": plan.solver,
        "rank_to_dataset_item_idx": {
            str(local_rank): [row_map[source_idx] for source_idx in source_ids]
            for local_rank, source_ids in plan.rank_to_source_idx.items()
        },
        "rank_to_source_idx": {
            str(rank): list(source_ids)
            for rank, source_ids in plan.rank_to_source_idx.items()
        },
        "rank_loads": {
            str(rank): value for rank, value in plan.rank_loads.items()
        },
        "rank_adjusted_peak_loads": {
            str(rank): value
            for rank, value in plan.rank_adjusted_peak_loads.items()
        },
    }


def build_plan(args: argparse.Namespace) -> None:
    baseline_dirs = [Path(item) for item in args.baseline_dir]
    history = _read_baseline_history(
        baseline_dirs,
        args.steps,
        args.responses_per_prompt,
        require_compact_history=args.require_compact_history,
    )
    stats_by_input = _read_ema_baseline_stats(history, args.length_ema_decay)
    tail_guard_ratio, ratio_samples = _tail_guard_underestimate_ratio(
        history,
        ema_decay=args.length_ema_decay,
        quantile=args.tail_guard_ratio_quantile,
        default_ratio=args.tail_guard_default_ratio,
        window=args.tail_guard_ratio_window,
    )
    full_df, stats = _map_stats_to_dataset(
        Path(args.train_file),
        stats_by_input,
        args.dataset_fraction,
        args.max_samples,
        args.steps * PROMPTS_PER_STEP,
    )
    steps = _make_steps(stats, args.steps, args.grouping, args.seed)
    floor_caps = _parse_floor_caps(args.floor_kv_caps)

    solved_steps, repairs = _repair_infeasible_steps(
        steps,
        args.grouping,
        floor_caps,
        args.active_peak_safety_factor,
        args.max_response_len,
        args.max_cross_step_repair_swaps,
        args.repair_candidate_limit,
    )
    for step_index, worker_plans in enumerate(solved_steps, start=1):
        print(
            f"[dp2-ep8 plan] step={step_index} grouping={args.grouping} "
            f"floors={[plan.selected_floor for plan in worker_plans]} "
            f"areas={[round(plan.release_area, 2) for plan in worker_plans]}",
            flush=True,
        )

    ordered_source_indices = []
    for worker_plans in solved_steps:
        for worker_plan in worker_plans:
            for local_rank in range(WORKER_WORLD_SIZE):
                ordered_source_indices.extend(
                    worker_plan.rank_to_source_idx[local_rank])
    if len(ordered_source_indices) != args.steps * PROMPTS_PER_STEP:
        raise AssertionError("hierarchical plan lost prompts")
    if len(set(ordered_source_indices)) != len(ordered_source_indices):
        raise AssertionError("hierarchical plan duplicated prompts")
    row_map = {
        int(source_idx): row_idx
        for row_idx, source_idx in enumerate(ordered_source_indices)
    }

    plan_payload = []
    summary_steps = []
    oracle = {}
    for step_index, worker_plans in enumerate(solved_steps, start=1):
        predicted_exit = max(
            max(plan.rank_loads.values()) for plan in worker_plans)
        raw_cap = tail_guard_ratio * predicted_exit
        tail_guard_cap = min(
            int(args.max_response_len),
            max(
                args.tail_guard_min_cap,
                _ceil_to_multiple(raw_cap, args.tail_guard_round_to),
            ),
        )
        if args.disable_tail_guard:
            tail_guard_cap = int(args.max_response_len)
        workers = [
            _worker_payload(
                worker_id,
                plan,
                row_map,
                tail_guard_cap,
                tail_guard_ratio,
                tail_guard_cap < int(args.max_response_len),
            )
            for worker_id, plan in enumerate(worker_plans)
        ]
        global_rank_map = {}
        for worker_id, worker_plan in enumerate(worker_plans):
            for local_rank, source_ids in worker_plan.rank_to_source_idx.items():
                global_rank = worker_id * WORKER_WORLD_SIZE + local_rank
                global_rank_map[str(global_rank)] = [
                    row_map[source_idx] for source_idx in source_ids
                ]
                for source_idx in source_ids:
                    oracle[str(row_map[source_idx])] = float(next(
                        item.load for item in worker_plan.prompts
                        if item.source_idx == source_idx))
        entry = {
            "schema_version": 1,
            "topology": "external_dp2_ep8",
            "grouping": args.grouping,
            "step": step_index,
            "selected_floor": min(
                plan.selected_floor for plan in worker_plans),
            "theoretical_floor": min(
                plan.theoretical_floor for plan in worker_plans),
            "kv_cap": min(plan.kv_cap for plan in worker_plans),
            "tail_guard_response_cap": tail_guard_cap,
            "tail_guard_enabled": tail_guard_cap < int(args.max_response_len),
            "tail_guard_ratio": tail_guard_ratio,
            "tail_guard_ratio_quantile": args.tail_guard_ratio_quantile,
            "tail_guard_ratio_sample_count": ratio_samples,
            "tail_guard_predicted_step_exit": predicted_exit,
            "release_area_unit": RELEASE_AREA_UNIT,
            "rank_to_dataset_item_idx": global_rank_map,
            "worker_plans": workers,
        }
        plan_payload.append(entry)
        summary_steps.append({
            "step": step_index,
            "grouping": args.grouping,
            "tail_guard_response_cap": tail_guard_cap,
            "workers": workers,
            "total_predicted_release_area": sum(
                plan.release_area for plan in worker_plans),
            "release_area_unit": RELEASE_AREA_UNIT,
            "predicted_worker_finish_gap": abs(
                max(worker_plans[0].rank_loads.values())
                - max(worker_plans[1].rank_loads.values())),
        })

    output_train = Path(args.output_train)
    output_plan = Path(args.output_plan)
    output_summary = Path(args.output_summary)
    output_oracle = Path(args.output_oracle)
    for path in (output_train, output_plan, output_summary, output_oracle):
        path.parent.mkdir(parents=True, exist_ok=True)
    full_df.iloc[ordered_source_indices].copy().to_parquet(
        output_train, index=False)
    output_plan.write_text(
        json.dumps(plan_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    output_summary.write_text(
        json.dumps({
            "schema_version": 1,
            "topology": "external_dp2_ep8",
            "grouping": args.grouping,
            "seed": args.seed,
            "floor_kv_caps": {
                str(key): value for key, value in floor_caps.items()
            },
            "active_peak_safety_factor": args.active_peak_safety_factor,
            "release_area_unit": RELEASE_AREA_UNIT,
            "cross_step_repairs": repairs,
            "steps": summary_steps,
        }, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    output_oracle.write_text(
        json.dumps(oracle, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"[dp2-ep8 plan] train={output_train}")
    print(f"[dp2-ep8 plan] plan={output_plan}")
    print(f"[dp2-ep8 plan] summary={output_summary}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build strict KV-safe hierarchical AdaFloor plans for DP2 EP8."
    )
    parser.add_argument("--baseline-dir", action="append", required=True)
    parser.add_argument("--train-file", required=True)
    parser.add_argument("--output-train", required=True)
    parser.add_argument("--output-plan", required=True)
    parser.add_argument("--output-summary", required=True)
    parser.add_argument("--output-oracle", required=True)
    parser.add_argument(
        "--grouping", choices=("random", "length_sorted"), required=True)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--responses-per-prompt", type=int, default=16)
    parser.add_argument("--dataset-fraction", type=float, default=0.005)
    parser.add_argument("--max-samples", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=20270731)
    parser.add_argument("--length-ema-decay", type=float, default=0.3)
    parser.add_argument(
        "--floor-kv-caps", default="2:131072,4:280576,8:377344")
    parser.add_argument("--active-peak-safety-factor", type=float, default=1.16)
    parser.add_argument("--max-response-len", type=float, default=16384.0)
    parser.add_argument("--tail-guard-ratio-quantile", type=float, default=0.95)
    parser.add_argument("--tail-guard-ratio-window", type=int, default=3)
    parser.add_argument("--tail-guard-default-ratio", type=float, default=1.20)
    parser.add_argument("--tail-guard-min-cap", type=int, default=4096)
    parser.add_argument("--tail-guard-round-to", type=int, default=512)
    parser.add_argument("--max-cross-step-repair-swaps", type=int, default=8)
    parser.add_argument("--repair-candidate-limit", type=int, default=8)
    parser.add_argument("--disable-tail-guard", action="store_true")
    parser.add_argument("--require-compact-history", action="store_true")
    build_plan(parser.parse_args())


if __name__ == "__main__":
    main()
