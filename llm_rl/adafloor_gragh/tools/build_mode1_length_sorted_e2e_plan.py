#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import networkx as nx
from scipy.optimize import Bounds, LinearConstraint, milp

from build_mode1_optimized_rank_plan import (
    OFFLINE_PLANNING_HISTORY_FILENAME,
    PromptStats,
    _map_stats_to_dataset,
    _peak_active_tokens,
    _read_baseline_stats,
)


ALL_RANKS = list(range(16))
DEFAULT_MAX_RESPONSE_LEN = 16384.0
DEFAULT_FLOOR_KV_CAPS = {
    2: 147456.0,
    4: 280576.0,
    8: 377344.0,
    16: 377344.0,
}


_PLAN_START_TIME = time.perf_counter()
_TIMING_PROFILE: dict[str, float] | None = None
_REPAIR_TIMING_DEPTH = 0


def _record_timing(component: str, start: float) -> None:
    if _TIMING_PROFILE is None or _REPAIR_TIMING_DEPTH > 0:
        return
    _TIMING_PROFILE[component] += time.perf_counter() - start


def _record_repair_timing(start: float) -> None:
    if _TIMING_PROFILE is None:
        return
    _TIMING_PROFILE["repair"] += time.perf_counter() - start


def _progress(message: str) -> None:
    elapsed = time.perf_counter() - _PLAN_START_TIME
    print(
        f"[mode1 length-sorted e2e plan][{elapsed:8.2f}s] {message}",
        flush=True,
    )


@dataclass(frozen=True)
class BatchPlan:
    prompts: list[PromptStats]
    rank_to_prompt_indices: dict[int, list[int]]
    rank_loads: dict[int, float]
    rank_load_gaps: dict[int, float]
    rank_peak_loads: dict[int, float]
    rank_adjusted_peak_loads: dict[int, float]
    rank_token_sums: dict[int, float]
    feasible: bool
    selected_floor: int = 4
    theoretical_floor: int = 4
    kv_cap: float = 280576.0
    schedule_thresholds: tuple[float, ...] = ()
    schedule_quotas: tuple[tuple[float, int], ...] = ()
    release_area: float = 0.0
    rank_matching_objective: str = "min_rank_internal_gap"
    rank_grouping_search_space: str = "all_two_prompt_pairs"
    shrink_stages: tuple[int, ...] = (8, 4)
    stage_survivor_ranks: tuple[tuple[int, ...], ...] = (
        tuple(range(8, 16)),
        tuple(range(12, 16)),
    )
    intermediate_survivor_ranks: tuple[int, ...] = tuple(range(8, 16))
    final_survivor_ranks: tuple[int, ...] = tuple(range(12, 16))


@dataclass(frozen=True)
class PromptBundle:
    source_indices: tuple[int, int]
    prompt_loads: tuple[float, float]
    lengths: tuple[float, ...]
    sum_len: float
    load: float
    order_index: int


PlanningUnit = PromptStats | PromptBundle
PairMetric = tuple[float, float, float, float, float, list[int]]
FLOOR_CANDIDATES = (2, 4, 8, 16)
RANK_MATCHING_POLICIES = (
    "release_area",
    "min_skew",
    "contiguous",
)
RELEASE_AREA_UNIT = "rank_token_proxy"
PAIR_RAW_PEAK = 0
PAIR_ADJUSTED_PEAK = 1
PAIR_TOKEN_SUM = 2
PAIR_RANK_LOAD = 3
PAIR_LOAD_GAP = 4
PAIR_SOURCE_INDICES = 5


def _stats_from_lengths(
    lengths: tuple[float, ...],
    *,
    predicted_tail: float | None = None,
) -> PromptStats:
    arr = np.asarray(lengths, dtype=np.float64)
    return PromptStats(
        source_idx=-1,
        mean=float(arr.mean()),
        p95=float(np.percentile(arr, 95)),
        max_len=float(arr.max()),
        sum_len=float(arr.sum()),
        peak_active_tokens=_peak_active_tokens(arr.tolist()),
        clip_count=int(np.sum(arr >= DEFAULT_MAX_RESPONSE_LEN)),
        lengths=tuple(float(item) for item in arr.tolist()),
        predicted_tail=predicted_tail,
    )


def _update_prompt_tail_ema(
    previous: float | None,
    observed: float,
    ema_decay: float,
) -> float:
    if previous is None:
        return float(observed)
    return float(ema_decay * previous + (1.0 - ema_decay) * observed)


def _read_baseline_history(
    baseline_dirs: list[Path],
    steps: int,
    responses_per_prompt: int,
    require_compact_history: bool = False,
) -> list[dict[str, PromptStats]]:
    if not baseline_dirs:
        raise ValueError("at least one --baseline-dir is required")
    history: list[dict[str, PromptStats]] = []
    for history_idx, baseline_dir in enumerate(baseline_dirs, start=1):
        history_file = baseline_dir / OFFLINE_PLANNING_HISTORY_FILENAME
        if require_compact_history and not history_file.is_file():
            raise FileNotFoundError(
                f"required offline planning history is missing: {history_file}"
            )
        _progress(
            "history read start "
            f"index={history_idx}/{len(baseline_dirs)} dir={baseline_dir}"
        )
        current = _read_baseline_stats(
            baseline_dir, steps, responses_per_prompt
        )
        _progress(
            "history read done "
            f"index={history_idx}/{len(baseline_dirs)} "
            f"dir={baseline_dir} prompts={len(current)}"
        )
        history.append(current)
    return history


def _read_ema_baseline_stats(
    baseline_history: list[dict[str, PromptStats]],
    ema_decay: float,
) -> dict[str, PromptStats]:
    if not baseline_history:
        raise ValueError("at least one historical epoch is required")
    if not 0.0 <= ema_decay <= 1.0:
        raise ValueError(f"--length-ema-decay must be in [0, 1], got {ema_decay}")
    if len(baseline_history) == 1:
        return dict(baseline_history[0])

    ema_by_input: dict[str, float] = {}
    latest_by_input: dict[str, PromptStats] = {}
    seen_count_by_input: dict[str, int] = {}
    for current in baseline_history:
        for prompt_input, stat in current.items():
            ema_by_input[prompt_input] = _update_prompt_tail_ema(
                ema_by_input.get(prompt_input),
                stat.max_len,
                ema_decay,
            )
            latest_by_input[prompt_input] = stat
            seen_count_by_input[prompt_input] = (
                seen_count_by_input.get(prompt_input, 0) + 1)

    _progress(
        "[mode1 length-sorted e2e plan] EMA length prediction "
        f"history_dirs={len(baseline_history)} decay={ema_decay} "
        f"unique_prompts={len(ema_by_input)} "
        f"reobserved_prompts={sum(1 for count in seen_count_by_input.values() if count > 1)}"
    )
    return {
        prompt_input: _stats_from_lengths(
            latest_by_input[prompt_input].lengths,
            predicted_tail=predicted_tail,
        )
        for prompt_input, predicted_tail in ema_by_input.items()
    }


def _adjacent_prompt_bundles(group: list[PromptStats]) -> list[PromptBundle]:
    if len(group) != 64:
        raise ValueError(
            f"adjacent prompt bundling requires 64 prompts, got {len(group)}")
    ordered = sorted(group, key=lambda item: (item.load, item.source_idx))
    source_indices = [int(item.source_idx) for item in ordered]
    if len(set(source_indices)) != len(source_indices):
        raise ValueError("adjacent prompt bundling requires unique source indices")

    bundles: list[PromptBundle] = []
    for order_index, offset in enumerate(range(0, len(ordered), 2)):
        left, right = ordered[offset:offset + 2]
        bundles.append(PromptBundle(
            source_indices=(int(left.source_idx), int(right.source_idx)),
            prompt_loads=(float(left.load), float(right.load)),
            lengths=tuple(left.lengths + right.lengths),
            sum_len=float(left.sum_len + right.sum_len),
            load=float(max(left.load, right.load)),
            order_index=int(order_index),
        ))
    return bundles


def _unit_source_indices(item: PlanningUnit) -> tuple[int, ...]:
    if isinstance(item, PromptBundle):
        return item.source_indices
    return (int(item.source_idx),)


def _unit_prompt_loads(item: PlanningUnit) -> tuple[float, ...]:
    if isinstance(item, PromptBundle):
        return item.prompt_loads
    return (float(item.load),)


def _unit_sort_key(item: PlanningUnit) -> tuple[float, int]:
    if isinstance(item, PromptBundle):
        return float(item.load), int(item.order_index)
    return float(item.load), int(item.source_idx)


def _pair_metrics(
    group: list[PlanningUnit],
    *,
    active_peak_safety_factor: float,
    max_response_len: float,
) -> dict[tuple[int, int], PairMetric]:
    timing_start = time.perf_counter()
    metrics: dict[tuple[int, int], PairMetric] = {}
    for first in range(len(group)):
        for second in range(first + 1, len(group)):
            left = group[first]
            right = group[second]
            response_lengths = left.lengths + right.lengths
            adjusted_lengths = tuple(
                min(float(length) * active_peak_safety_factor, max_response_len)
                for length in response_lengths
            )
            prompt_loads = _unit_prompt_loads(left) + _unit_prompt_loads(right)
            metrics[(first, second)] = (
                _peak_active_tokens(response_lengths),
                _peak_active_tokens(adjusted_lengths),
                float(left.sum_len + right.sum_len),
                float(max(prompt_loads)),
                float(max(prompt_loads) - min(prompt_loads)),
                list(_unit_source_indices(left) + _unit_source_indices(right)),
            )
    _record_timing("graph_construction", timing_start)
    return metrics


def _capacity_constrained_load_gap_matching(
    group: list[PlanningUnit],
    *,
    max_rank_peak_tokens: float,
    active_peak_safety_factor: float,
    max_response_len: float,
) -> tuple[list[PairMetric], bool]:
    metrics = _pair_metrics(
        group,
        active_peak_safety_factor=active_peak_safety_factor,
        max_response_len=max_response_len,
    )
    matching = _find_min_weight_matching_under_adjusted_peak(
        len(group), metrics, max_rank_peak_tokens)
    feasible = matching is not None
    if matching is None:
        threshold = _min_feasible_adjusted_peak_threshold(len(group), metrics)
        matching = _find_min_weight_matching_under_adjusted_peak(
            len(group), metrics, threshold)
    if matching is None:
        raise RuntimeError("failed to find length-sorted e2e rank matching")

    pairs = [metrics[pair] for pair in matching]
    pairs.sort(key=lambda item: (item[3], item[4], item[1], item[0], item[2]))
    return pairs, feasible


def _role_for_floor(
    floor: int,
) -> tuple[tuple[int, ...], tuple[tuple[int, ...], ...], tuple[int, ...], tuple[int, ...]]:
    floor = int(floor)
    if floor not in FLOOR_CANDIDATES:
        raise ValueError(f"unsupported adaptive floor: {floor}")
    if floor == 16:
        stages = (16,)
        stage_sets = (tuple(ALL_RANKS),)
    else:
        stages = tuple(size for size in (8, 4, 2) if size >= floor)
        stage_sets = tuple(tuple(range(16 - size, 16)) for size in stages)
    intermediate = stage_sets[0]
    final = stage_sets[-1]
    return stages, stage_sets, intermediate, final


def _theoretical_floor_from_rank_loads(rank_loads: dict[int, float]) -> int:
    max_load = max(rank_loads.values())
    tail_count = sum(1 for value in rank_loads.values() if value >= max_load)
    for floor in FLOOR_CANDIDATES:
        if tail_count <= floor:
            return floor
    return 16


def _min_feasible_adjusted_peak_threshold(
    item_count: int,
    metrics: dict[tuple[int, int], PairMetric],
) -> float:
    thresholds = sorted({item[1] for item in metrics.values()})
    best_threshold = thresholds[-1]
    lo, hi = 0, len(thresholds) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if _has_perfect_matching_under_adjusted_peak(
                item_count, metrics, thresholds[mid]):
            best_threshold = thresholds[mid]
            hi = mid - 1
        else:
            lo = mid + 1
    return best_threshold


def _find_min_weight_matching_under_adjusted_peak(
    item_count: int,
    metrics: dict[tuple[int, int], PairMetric],
    adjusted_peak_limit: float,
) -> set[tuple[int, int]] | None:
    timing_start = time.perf_counter()
    graph = nx.Graph()
    graph.add_nodes_from(range(item_count))
    for (
            first,
            second,
    ), (
            _peak,
            adjusted_peak,
            rank_sum,
            rank_load,
            load_gap,
            _source_indices,
    ) in metrics.items():
        if adjusted_peak > adjusted_peak_limit:
            continue
        weight = (
            load_gap * 1_000_000.0 +
            rank_load * 1_000.0 +
            adjusted_peak +
            rank_sum * 0.001
        )
        graph.add_edge(first, second, weight=float(weight))

    matching = nx.algorithms.matching.min_weight_matching(
        graph, weight="weight")
    _record_timing("graph_construction", timing_start)
    if len(matching) * 2 != item_count:
        return None
    return {(min(first, second), max(first, second)) for first, second in matching}


def _has_perfect_matching_under_adjusted_peak(
    item_count: int,
    metrics: dict[tuple[int, int], PairMetric],
    adjusted_peak_limit: float,
) -> bool:
    timing_start = time.perf_counter()
    graph = nx.Graph()
    graph.add_nodes_from(range(item_count))
    for (
            first,
            second,
    ), (
            _peak,
            adjusted_peak,
            _rank_sum,
            _rank_load,
            _load_gap,
            _source_indices,
    ) in metrics.items():
        if adjusted_peak <= adjusted_peak_limit:
            graph.add_edge(first, second, weight=1.0)
    matching = nx.algorithms.matching.max_weight_matching(
        graph, maxcardinality=True)
    _record_timing("graph_construction", timing_start)
    return len(matching) * 2 == item_count


def _min_skew_matching_from_edges(
    item_count: int,
    metrics: dict[tuple[int, int], PairMetric],
    allowed_edges: set[tuple[int, int]],
) -> set[tuple[int, int]] | None:
    timing_start = time.perf_counter()
    graph = nx.Graph()
    graph.add_nodes_from(range(item_count))
    for edge in allowed_edges:
        item = metrics[edge]
        weight = (
            item[PAIR_LOAD_GAP] * 1_000_000.0 +
            item[PAIR_RANK_LOAD] * 1_000.0 +
            item[PAIR_ADJUSTED_PEAK] +
            item[PAIR_TOKEN_SUM] * 0.001
        )
        graph.add_edge(edge[0], edge[1], weight=float(weight))
    matching = nx.algorithms.matching.min_weight_matching(
        graph, weight="weight")
    _record_timing("graph_construction", timing_start)
    if len(matching) * 2 != item_count:
        return None
    return {(min(first, second), max(first, second)) for first, second in matching}


def _minimax_adjusted_peak_matching(
    item_count: int,
    metrics: dict[tuple[int, int], PairMetric],
    adjusted_peak_limit: float,
) -> tuple[set[tuple[int, int]], float] | None:
    peak_threshold = _min_feasible_adjusted_peak_threshold(item_count, metrics)
    if peak_threshold > adjusted_peak_limit:
        return None
    allowed_edges = {
        edge for edge, item in metrics.items()
        if item[PAIR_ADJUSTED_PEAK] <= peak_threshold
    }
    matching = _min_skew_matching_from_edges(item_count, metrics, allowed_edges)
    if matching is None:
        return None
    return matching, float(peak_threshold)


def _schedule_candidates(
    floor: int,
    candidate_times: list[float],
    tail_time: float,
) -> list[dict[str, Any]]:
    timing_start = time.perf_counter()
    schedules: list[dict[str, Any]] = []
    if floor == 8:
        for a in candidate_times:
            schedules.append({
                "thresholds": (float(a),),
                "quotas": ((float(a), 8),),
                "release_area": float(8.0 * max(0.0, tail_time - a)),
            })
    elif floor == 4:
        for a in candidate_times:
            for b in candidate_times:
                if a > b:
                    continue
                schedules.append({
                    "thresholds": (float(a), float(b)),
                    "quotas": ((float(a), 8), (float(b), 12)),
                    "release_area": float(
                        8.0 * max(0.0, tail_time - a)
                        + 4.0 * max(0.0, tail_time - b)),
                })
    elif floor == 2:
        for a in candidate_times:
            for b in candidate_times:
                if a > b:
                    continue
                for c in candidate_times:
                    if b > c:
                        continue
                    schedules.append({
                        "thresholds": (float(a), float(b), float(c)),
                        "quotas": (
                            (float(a), 8),
                            (float(b), 12),
                            (float(c), 14),
                        ),
                        "release_area": float(
                            8.0 * max(0.0, tail_time - a)
                            + 4.0 * max(0.0, tail_time - b)
                            + 2.0 * max(0.0, tail_time - c)),
                    })
    elif floor == 16:
        schedules.append({
            "thresholds": (),
            "quotas": (),
            "release_area": 0.0,
        })
    else:
        raise ValueError(f"unsupported floor={floor}")
    schedules.sort(
        key=lambda item: (
            item["release_area"],
            tuple(-threshold for threshold in item["thresholds"]),
        ),
        reverse=True,
    )
    _record_timing("graph_construction", timing_start)
    return schedules


def _quota_aware_matching_oracle(
    item_count: int,
    metrics: dict[tuple[int, int], PairMetric],
    allowed_edges: set[tuple[int, int]],
    quotas: tuple[tuple[float, int], ...],
) -> set[tuple[int, int]] | None:
    if not quotas:
        return _min_skew_matching_from_edges(item_count, metrics, allowed_edges)

    construction_start = time.perf_counter()
    edges = sorted(allowed_edges)
    edge_count = len(edges)
    if edge_count == 0:
        return None

    # Objective is a tie-breaker inside a fixed release-area schedule. It keeps
    # rank-internal skew small after the quota constraints have guaranteed the
    # requested shrink thresholds.
    c = np.asarray([
        metrics[edge][PAIR_LOAD_GAP] * 1_000_000.0
        + metrics[edge][PAIR_RANK_LOAD] * 1_000.0
        + metrics[edge][PAIR_ADJUSTED_PEAK]
        + metrics[edge][PAIR_TOKEN_SUM] * 0.001
        for edge in edges
    ], dtype=np.float64)

    rows: list[list[float]] = []
    lower: list[float] = []
    upper: list[float] = []

    for node in range(item_count):
        row = [0.0] * edge_count
        for edge_idx, (first, second) in enumerate(edges):
            if first == node or second == node:
                row[edge_idx] = 1.0
        rows.append(row)
        lower.append(1.0)
        upper.append(1.0)

    for threshold, required_finished_ranks in quotas:
        row = [
            1.0 if metrics[edge][PAIR_RANK_LOAD] <= threshold else 0.0
            for edge in edges
        ]
        rows.append(row)
        lower.append(float(required_finished_ranks))
        upper.append(float(edge_count))

    constraints = LinearConstraint(
        np.asarray(rows, dtype=np.float64),
        np.asarray(lower, dtype=np.float64),
        np.asarray(upper, dtype=np.float64),
    )
    _record_timing("graph_construction", construction_start)
    solve_start = time.perf_counter()
    result = milp(
        c=c,
        integrality=np.ones(edge_count, dtype=np.int8),
        bounds=Bounds(np.zeros(edge_count), np.ones(edge_count)),
        constraints=constraints,
        options={"time_limit": 5.0, "mip_rel_gap": 0.0},
    )
    _record_timing("milp_solving", solve_start)
    if not result.success or result.x is None:
        return None
    selected = {
        edges[idx]
        for idx, value in enumerate(result.x)
        if value >= 0.5
    }
    if len(selected) * 2 != item_count:
        return None
    covered: set[int] = set()
    for first, second in selected:
        if first in covered or second in covered:
            return None
        covered.add(first)
        covered.add(second)
    if len(covered) != item_count:
        return None
    return selected


def _release_area_matching_for_floor(
    group: list[PlanningUnit],
    *,
    selected_floor: int,
    kv_cap: float,
    active_peak_safety_factor: float,
    max_response_len: float,
    rank_matching_policy: str = "release_area",
    allow_zero_release_censored_tail: bool = False,
) -> tuple[list[PairMetric], bool, tuple[float, ...], tuple[tuple[float, int], ...], float, str]:
    metrics = _pair_metrics(
        group,
        active_peak_safety_factor=active_peak_safety_factor,
        max_response_len=max_response_len,
    )
    item_count = len(group)
    timing_start = time.perf_counter()
    allowed_edges = {
        edge for edge, item in metrics.items()
        if item[PAIR_ADJUSTED_PEAK] <= kv_cap
    }
    _record_timing("graph_construction", timing_start)

    if rank_matching_policy == "contiguous":
        ordered_positions = sorted(
            range(item_count),
            key=lambda idx: _unit_sort_key(group[idx]),
        )
        matching = [
            (
                min(ordered_positions[idx], ordered_positions[idx + 1]),
                max(ordered_positions[idx], ordered_positions[idx + 1]),
            )
            for idx in range(0, item_count, 2)
        ]
        pairs = [metrics[pair] for pair in matching]
        thresholds, quotas, area = _fallback_schedule_from_pairs(
            pairs,
            selected_floor=selected_floor,
            tail_time=max(item.load for item in group),
        )
        feasible = all(edge in allowed_edges for edge in matching)
        return (
            pairs,
            feasible,
            thresholds,
            quotas,
            area,
            "contiguous_pairing",
        )

    if not _has_perfect_matching_under_adjusted_peak(item_count, metrics, kv_cap):
        threshold = _min_feasible_adjusted_peak_threshold(item_count, metrics)
        allowed_edges = {
            edge for edge, item in metrics.items()
            if item[PAIR_ADJUSTED_PEAK] <= threshold
        }
        matching = _min_skew_matching_from_edges(item_count, metrics, allowed_edges)
        if matching is None:
            raise RuntimeError("failed to find fallback matching")
        pairs = [metrics[pair] for pair in matching]
        pairs.sort(key=lambda item: (item[PAIR_RANK_LOAD], item[PAIR_LOAD_GAP], item[PAIR_ADJUSTED_PEAK]))
        return pairs, False, (), (), 0.0, "fallback_min_skew_over_cap"

    if rank_matching_policy == "min_skew":
        matching = _min_skew_matching_from_edges(
            item_count, metrics, allowed_edges)
        if matching is None:
            raise RuntimeError("failed to find KV-feasible min-skew matching")
        pairs = [metrics[pair] for pair in matching]
        pairs.sort(key=lambda item: (
            item[PAIR_RANK_LOAD],
            item[PAIR_LOAD_GAP],
            item[PAIR_ADJUSTED_PEAK],
        ))
        thresholds, quotas, area = _fallback_schedule_from_pairs(
            pairs,
            selected_floor=selected_floor,
            tail_time=max(item.load for item in group),
        )
        return (
            pairs,
            True,
            thresholds,
            quotas,
            area,
            "kv_feasible_min_skew",
        )

    if rank_matching_policy != "release_area":
        raise ValueError(
            f"unsupported rank_matching_policy={rank_matching_policy!r}; "
            f"expected one of {RANK_MATCHING_POLICIES}")

    if selected_floor == 16:
        result = _minimax_adjusted_peak_matching(item_count, metrics, kv_cap)
        if result is None:
            matching = _min_skew_matching_from_edges(item_count, metrics,
                                                    allowed_edges)
            objective = "floor16_min_skew"
        else:
            matching, _peak_threshold = result
            objective = "floor16_minimax_adjusted_peak_then_skew"
        if matching is None:
            raise RuntimeError("failed to find floor16 matching")
        pairs = [metrics[pair] for pair in matching]
        pairs.sort(key=lambda item: (
            item[PAIR_ADJUSTED_PEAK],
            item[PAIR_RANK_LOAD],
            item[PAIR_LOAD_GAP],
        ))
        return pairs, True, (), (), 0.0, objective

    candidate_times = sorted({
        item[PAIR_RANK_LOAD]
        for edge, item in metrics.items()
        if edge in allowed_edges
    })
    tail_time = max(item.load for item in group)
    for schedule in _schedule_candidates(selected_floor, candidate_times, tail_time):
        # A floor is only useful if its deepest shrink point happens before
        # the batch tail. For example, floor8 requires at least 8 ranks with
        # M_ij < T; if more than 8 ranks run to the tail, the only possible
        # threshold is a=T and no rank-time is actually released.
        if schedule["thresholds"] and schedule["thresholds"][-1] >= tail_time:
            continue
        matching = _quota_aware_matching_oracle(
            item_count,
            metrics,
            allowed_edges,
            tuple(schedule["quotas"]),
        )
        if matching is None:
            continue
        pairs = [metrics[pair] for pair in matching]
        pairs.sort(key=lambda item: (item[PAIR_RANK_LOAD], item[PAIR_LOAD_GAP], item[PAIR_ADJUSTED_PEAK]))
        return (
            pairs,
            True,
            tuple(float(item) for item in schedule["thresholds"]),
            tuple((float(threshold), int(quota)) for threshold, quota in schedule["quotas"]),
            float(schedule["release_area"]),
            "max_release_area_quota_matching",
        )

    matching = _min_skew_matching_from_edges(item_count, metrics, allowed_edges)
    if matching is None:
        raise RuntimeError("failed to find KV-feasible fallback matching")
    pairs = [metrics[pair] for pair in matching]
    pairs.sort(key=lambda item: (item[PAIR_RANK_LOAD], item[PAIR_LOAD_GAP], item[PAIR_ADJUSTED_PEAK]))
    fallback_thresholds = _fallback_schedule_from_pairs(
        pairs,
        selected_floor=selected_floor,
        tail_time=max(item.load for item in group),
    )
    if allow_zero_release_censored_tail and max_response_len < DEFAULT_MAX_RESPONSE_LEN:
        return (
            pairs,
            True,
            fallback_thresholds[0],
            fallback_thresholds[1],
            fallback_thresholds[2],
            "censored_tail_kv_feasible_min_skew",
        )
    return (
        pairs,
        False,
        fallback_thresholds[0],
        fallback_thresholds[1],
        fallback_thresholds[2],
        "fallback_min_skew_no_positive_release_schedule",
    )


def _fallback_schedule_from_pairs(
    pairs: list[PairMetric],
    *,
    selected_floor: int,
    tail_time: float,
) -> tuple[tuple[float, ...], tuple[tuple[float, int], ...], float]:
    loads = sorted(item[PAIR_RANK_LOAD] for item in pairs)
    if selected_floor == 8:
        thresholds = (float(loads[7]),)
        quotas = ((thresholds[0], 8),)
        area = 8.0 * max(0.0, tail_time - thresholds[0])
    elif selected_floor == 4:
        thresholds = (float(loads[7]), float(loads[11]))
        quotas = ((thresholds[0], 8), (thresholds[1], 12))
        area = (
            8.0 * max(0.0, tail_time - thresholds[0])
            + 4.0 * max(0.0, tail_time - thresholds[1])
        )
    elif selected_floor == 2:
        thresholds = (float(loads[7]), float(loads[11]), float(loads[13]))
        quotas = (
            (thresholds[0], 8),
            (thresholds[1], 12),
            (thresholds[2], 14),
        )
        area = (
            8.0 * max(0.0, tail_time - thresholds[0])
            + 4.0 * max(0.0, tail_time - thresholds[1])
            + 2.0 * max(0.0, tail_time - thresholds[2])
        )
    else:
        thresholds = ()
        quotas = ()
        area = 0.0
    return thresholds, quotas, float(area)


def _singleton_rank_metrics(
    group: list[PromptStats],
    *,
    active_peak_safety_factor: float,
    max_response_len: float,
) -> list[PairMetric]:
    if len(group) != len(ALL_RANKS):
        raise ValueError(
            f"singleton rank assignment requires {len(ALL_RANKS)} prompts, "
            f"got {len(group)}")
    metrics: list[PairMetric] = []
    for item in group:
        adjusted_lengths = tuple(
            min(float(length) * active_peak_safety_factor, max_response_len)
            for length in item.lengths
        )
        metrics.append((
            float(item.peak_active_tokens),
            float(_peak_active_tokens(adjusted_lengths)),
            float(item.sum_len),
            float(item.load),
            0.0,
            [int(item.source_idx)],
        ))
    return metrics


def _singleton_assignment_for_floor(
    group: list[PromptStats],
    *,
    selected_floor: int,
    kv_cap: float,
    active_peak_safety_factor: float,
    max_response_len: float,
    rank_matching_policy: str,
    allow_zero_release_censored_tail: bool,
) -> tuple[list[PairMetric], bool, tuple[float, ...], tuple[tuple[float, int], ...], float, str]:
    if rank_matching_policy not in RANK_MATCHING_POLICIES:
        raise ValueError(
            f"unsupported rank_matching_policy={rank_matching_policy!r}; "
            f"expected one of {RANK_MATCHING_POLICIES}")
    metrics = _singleton_rank_metrics(
        group,
        active_peak_safety_factor=active_peak_safety_factor,
        max_response_len=max_response_len,
    )
    feasible = all(item[PAIR_ADJUSTED_PEAK] <= kv_cap for item in metrics)
    if selected_floor == 16:
        metrics.sort(key=lambda item: (
            item[PAIR_ADJUSTED_PEAK],
            item[PAIR_RANK_LOAD],
            item[PAIR_SOURCE_INDICES][0],
        ))
        return (
            metrics,
            feasible,
            (),
            (),
            0.0,
            "floor16_singleton_minimax_adjusted_peak",
        )

    metrics.sort(key=lambda item: (
        item[PAIR_RANK_LOAD],
        item[PAIR_ADJUSTED_PEAK],
        item[PAIR_SOURCE_INDICES][0],
    ))
    if not feasible:
        return metrics, False, (), (), 0.0, "singleton_over_kv_cap"

    thresholds, quotas, area = _fallback_schedule_from_pairs(
        metrics,
        selected_floor=selected_floor,
        tail_time=max(item.load for item in group),
    )
    if rank_matching_policy == "release_area":
        positive_release = bool(thresholds and thresholds[-1] < max(
            item.load for item in group))
        if not positive_release and not (
                allow_zero_release_censored_tail
                and max_response_len < DEFAULT_MAX_RESPONSE_LEN):
            return (
                metrics,
                False,
                thresholds,
                quotas,
                area,
                "singleton_no_positive_release_schedule",
            )
        objective = "singleton_release_area_order"
    elif rank_matching_policy == "min_skew":
        objective = "singleton_min_skew_order"
    else:
        objective = "singleton_contiguous_order"
    return metrics, True, thresholds, quotas, area, objective


def _solve_one_batch(
    batch: list[PromptStats],
    *,
    max_rank_peak_tokens: float,
    floor_kv_caps: dict[int, float] | None = None,
    adaptive_floor: bool = False,
    min_adaptive_floor: int = 2,
    force_selected_floor: int | None = None,
    kv_safe_fixed_floor: int | None = None,
    rank_matching_policy: str = "release_area",
    active_peak_safety_factor: float,
    max_response_len: float,
    ignore_tail_ties_at_response_cap: bool = False,
) -> BatchPlan:
    if len(batch) not in (16, 32, 64):
        raise ValueError(
            f"one e2e batch requires 16, 32, or 64 prompts, got {len(batch)}")
    cap_by_floor = dict(floor_kv_caps or {})
    if not cap_by_floor:
        cap_by_floor = {4: float(max_rank_peak_tokens)}
    cap_by_floor.setdefault(16, float(max_rank_peak_tokens))
    if force_selected_floor is not None and force_selected_floor not in FLOOR_CANDIDATES:
        raise ValueError(
            f"unsupported forced floor={force_selected_floor}; "
            f"expected one of {FLOOR_CANDIDATES}")
    if kv_safe_fixed_floor is not None and kv_safe_fixed_floor not in FLOOR_CANDIDATES:
        raise ValueError(
            f"unsupported KV-safe fixed floor={kv_safe_fixed_floor}; "
            f"expected one of {FLOOR_CANDIDATES}")
    if force_selected_floor is not None and kv_safe_fixed_floor is not None:
        raise ValueError(
            "force_selected_floor and kv_safe_fixed_floor are mutually exclusive")

    if len(batch) == 16:
        planning_group: list[PlanningUnit] = list(batch)
        rank_grouping_search_space = "singleton_prompts"
        probe_pairs = _singleton_rank_metrics(
            batch,
            active_peak_safety_factor=active_peak_safety_factor,
            max_response_len=max_response_len,
        )
        probe_pairs.sort(key=lambda item: (
            item[PAIR_RANK_LOAD],
            item[PAIR_ADJUSTED_PEAK],
            item[PAIR_SOURCE_INDICES][0],
        ))

        def solve_floor(
            group: list[PromptStats],
            *,
            selected_floor: int,
            kv_cap: float,
            active_peak_safety_factor: float,
            max_response_len: float,
            rank_matching_policy: str,
            allow_zero_release_censored_tail: bool,
        ):
            return _singleton_assignment_for_floor(
                group,
                selected_floor=selected_floor,
                kv_cap=kv_cap,
                active_peak_safety_factor=active_peak_safety_factor,
                max_response_len=max_response_len,
                rank_matching_policy=rank_matching_policy,
                allow_zero_release_censored_tail=allow_zero_release_censored_tail,
            )
    else:
        if len(batch) == 64:
            planning_group = list(_adjacent_prompt_bundles(batch))
            rank_grouping_search_space = (
                "pairings_of_32_fixed_adjacent_two_prompt_bundles")
        else:
            planning_group = list(batch)
            rank_grouping_search_space = "all_two_prompt_pairs"
        probe_pairs, _probe_feasible = _capacity_constrained_load_gap_matching(
            planning_group,
            max_rank_peak_tokens=float("inf"),
            active_peak_safety_factor=active_peak_safety_factor,
            max_response_len=max_response_len,
        )
        solve_floor = _release_area_matching_for_floor
    probe_loads = {
        rank: float(rank_load)
        for rank, (
                _rank_peak,
                _adjusted_peak,
                _rank_sum,
                rank_load,
                _load_gap,
                _source_indices,
        ) in zip(ALL_RANKS, probe_pairs, strict=True)
    }
    theoretical_floor = _theoretical_floor_from_rank_loads(probe_loads)
    if ignore_tail_ties_at_response_cap and max_response_len < DEFAULT_MAX_RESPONSE_LEN:
        # A short max-response smoke test censors all longer generations at the
        # artificial cap. Many unrelated prompts can then have identical
        # max_len=max_response_len, which is not evidence that all their ranks
        # would be true tail ranks in an uncapped rollout. Keep the theoretical
        # value for reporting, but let the KV/release-area oracle choose the
        # actual floor instead of forcing floor16 from capped ties.
        theoretical_floor_for_selection = int(min_adaptive_floor)
    else:
        theoretical_floor_for_selection = int(theoretical_floor)
    if not adaptive_floor:
        selected_floor = 4
        kv_cap = float(max_rank_peak_tokens)
        (
            pairs,
            feasible,
            schedule_thresholds,
            schedule_quotas,
            release_area,
            rank_matching_objective,
        ) = solve_floor(
            planning_group,
            selected_floor=selected_floor,
            kv_cap=kv_cap,
            active_peak_safety_factor=active_peak_safety_factor,
            max_response_len=max_response_len,
            rank_matching_policy=rank_matching_policy,
            allow_zero_release_censored_tail=ignore_tail_ties_at_response_cap,
        )
    else:
        selected_floor = 16
        kv_cap = float(cap_by_floor.get(16, max_rank_peak_tokens))
        pairs = probe_pairs
        feasible = False
        schedule_thresholds = ()
        schedule_quotas = ()
        release_area = 0.0
        rank_matching_objective = "probe"
        if force_selected_floor is not None:
            candidate_floors = (int(force_selected_floor),)
        elif kv_safe_fixed_floor is not None:
            start_floor = max(
                int(kv_safe_fixed_floor),
                int(theoretical_floor_for_selection),
            )
            candidate_floors = tuple(
                floor for floor in (int(kv_safe_fixed_floor), 16)
                if floor >= start_floor)
        else:
            start_floor = max(int(min_adaptive_floor),
                              int(theoretical_floor_for_selection))
            candidate_floors = tuple(
                floor for floor in FLOOR_CANDIDATES if floor >= start_floor)
        best_candidate: tuple[
            tuple[float, int],
            int,
            float,
            list[PairMetric],
            bool,
            tuple[float, ...],
            tuple[tuple[float, int], ...],
            float,
            str,
        ] | None = None
        fallback_candidate: tuple[
            int,
            float,
            list[PairMetric],
            bool,
            tuple[float, ...],
            tuple[tuple[float, int], ...],
            float,
            str,
        ] | None = None
        for floor in candidate_floors:
            cap = float(cap_by_floor.get(floor, cap_by_floor.get(16, max_rank_peak_tokens)))
            (
                candidate_pairs,
                candidate_feasible,
                candidate_thresholds,
                candidate_quotas,
                candidate_release_area,
                candidate_objective,
            ) = solve_floor(
                planning_group,
                selected_floor=int(floor),
                kv_cap=cap,
                active_peak_safety_factor=active_peak_safety_factor,
                max_response_len=max_response_len,
                rank_matching_policy=rank_matching_policy,
                allow_zero_release_censored_tail=ignore_tail_ties_at_response_cap,
            )
            candidate = (
                int(floor),
                cap,
                candidate_pairs,
                candidate_feasible,
                candidate_thresholds,
                candidate_quotas,
                candidate_release_area,
                candidate_objective,
            )
            fallback_candidate = candidate
            if not candidate_feasible:
                continue

            # Compare the best schedule from every feasible floor. A deeper
            # floor can delay an earlier quorum enough to reduce total area.
            # Equal-area plans prefer the larger floor and fewer transitions.
            candidate_score = (
                float(candidate_release_area),
                int(floor),
            )
            if best_candidate is None or candidate_score > best_candidate[0]:
                best_candidate = (candidate_score, *candidate)

        if best_candidate is not None:
            (
                _candidate_score,
                selected_floor,
                kv_cap,
                pairs,
                feasible,
                schedule_thresholds,
                schedule_quotas,
                release_area,
                rank_matching_objective,
            ) = best_candidate
        elif fallback_candidate is not None:
            (
                selected_floor,
                kv_cap,
                pairs,
                feasible,
                schedule_thresholds,
                schedule_quotas,
                release_area,
                rank_matching_objective,
            ) = fallback_candidate

    rank_map: dict[int, list[int]] = {}
    loads: dict[int, float] = {}
    load_gaps: dict[int, float] = {}
    peak_loads: dict[int, float] = {}
    adjusted_peak_loads: dict[int, float] = {}
    token_sums: dict[int, float] = {}
    for rank, (
            rank_peak,
            adjusted_peak,
            rank_sum,
            rank_load,
            load_gap,
            source_indices,
    ) in zip(
            ALL_RANKS, pairs, strict=True):
        rank_map[int(rank)] = source_indices
        loads[int(rank)] = float(rank_load)
        load_gaps[int(rank)] = float(load_gap)
        peak_loads[int(rank)] = float(rank_peak)
        adjusted_peak_loads[int(rank)] = float(adjusted_peak)
        token_sums[int(rank)] = float(rank_sum)

    if len(batch) == 64:
        expected_sources = [int(item.source_idx) for item in batch]
        assigned_sources = [
            int(source_idx)
            for rank in ALL_RANKS
            for source_idx in rank_map[rank]
        ]
        if any(len(rank_map[rank]) != 4 for rank in ALL_RANKS):
            raise RuntimeError(
                "batch64 rank matching must assign exactly four prompts per rank")
        if (
            len(set(expected_sources)) != len(expected_sources)
            or len(set(assigned_sources)) != len(assigned_sources)
            or set(assigned_sources) != set(expected_sources)
        ):
            raise RuntimeError(
                "batch64 rank matching must cover every source prompt exactly once")

    shrink_stages, stage_sets, intermediate_ranks, final_ranks = _role_for_floor(selected_floor)
    return BatchPlan(
        prompts=list(batch),
        rank_to_prompt_indices=rank_map,
        rank_loads=loads,
        rank_load_gaps=load_gaps,
        rank_peak_loads=peak_loads,
        rank_adjusted_peak_loads=adjusted_peak_loads,
        rank_token_sums=token_sums,
        feasible=feasible,
        selected_floor=int(selected_floor),
        theoretical_floor=int(theoretical_floor),
        kv_cap=float(kv_cap),
        schedule_thresholds=tuple(schedule_thresholds),
        schedule_quotas=tuple(schedule_quotas),
        release_area=float(release_area),
        rank_matching_objective=str(rank_matching_objective),
        rank_grouping_search_space=rank_grouping_search_space,
        shrink_stages=shrink_stages,
        stage_survivor_ranks=stage_sets,
        intermediate_survivor_ranks=intermediate_ranks,
        final_survivor_ranks=final_ranks,
    )


def _max_adjusted_peak(plan: BatchPlan) -> float:
    return max(plan.rank_adjusted_peak_loads.values())


def _plan_overflow(plan: BatchPlan, max_rank_peak_tokens: float) -> float:
    return max(0.0, _max_adjusted_peak(plan) - max_rank_peak_tokens)


def _plan_effective_cap(plan: BatchPlan, fallback_cap: float) -> float:
    return float(plan.kv_cap if plan.kv_cap > 0 else fallback_cap)


def _plan_effective_overflow(plan: BatchPlan, fallback_cap: float) -> float:
    return _plan_overflow(plan, _plan_effective_cap(plan, fallback_cap))


def _solve_group(
    group: list[PromptStats],
    *,
    max_rank_peak_tokens: float,
    floor_kv_caps: dict[int, float] | None = None,
    adaptive_floor: bool = False,
    min_adaptive_floor: int = 2,
    force_selected_floor: int | None = None,
    kv_safe_fixed_floor: int | None = None,
    rank_matching_policy: str = "release_area",
    active_peak_safety_factor: float,
    max_response_len: float,
    ignore_tail_ties_at_response_cap: bool = False,
) -> BatchPlan:
    return _solve_one_batch(
        group,
        max_rank_peak_tokens=max_rank_peak_tokens,
        floor_kv_caps=floor_kv_caps,
        adaptive_floor=adaptive_floor,
        min_adaptive_floor=min_adaptive_floor,
        force_selected_floor=force_selected_floor,
        kv_safe_fixed_floor=kv_safe_fixed_floor,
        rank_matching_policy=rank_matching_policy,
        active_peak_safety_factor=active_peak_safety_factor,
        max_response_len=max_response_len,
        ignore_tail_ties_at_response_cap=ignore_tail_ties_at_response_cap,
    )


def _group_cache_key(group: list[PromptStats]) -> tuple[int, ...]:
    return tuple(int(item.source_idx) for item in group)


def _solve_group_cached(
    group: list[PromptStats],
    cache: dict[tuple[int, ...], BatchPlan],
    *,
    max_rank_peak_tokens: float,
    floor_kv_caps: dict[int, float] | None = None,
    adaptive_floor: bool = False,
    min_adaptive_floor: int = 2,
    force_selected_floor: int | None = None,
    kv_safe_fixed_floor: int | None = None,
    rank_matching_policy: str = "release_area",
    active_peak_safety_factor: float,
    max_response_len: float,
    ignore_tail_ties_at_response_cap: bool = False,
) -> BatchPlan:
    key = _group_cache_key(group)
    cached = cache.get(key)
    if cached is not None:
        return cached
    plan = _solve_group(
        group,
        max_rank_peak_tokens=max_rank_peak_tokens,
        floor_kv_caps=floor_kv_caps,
        adaptive_floor=adaptive_floor,
        min_adaptive_floor=min_adaptive_floor,
        force_selected_floor=force_selected_floor,
        kv_safe_fixed_floor=kv_safe_fixed_floor,
        rank_matching_policy=rank_matching_policy,
        active_peak_safety_factor=active_peak_safety_factor,
        max_response_len=max_response_len,
        ignore_tail_ties_at_response_cap=ignore_tail_ties_at_response_cap,
    )
    cache[key] = plan
    return plan


def _neighbor_indices(index: int, count: int) -> list[int]:
    neighbors: list[int] = []
    for distance in range(1, count):
        left = index - distance
        right = index + distance
        if left >= 0:
            neighbors.append(left)
        if right < count:
            neighbors.append(right)
    return neighbors


def _pair_repair_score(
    first: BatchPlan,
    second: BatchPlan,
    *,
    max_rank_peak_tokens: float,
) -> tuple[float, float, float, float, float, float]:
    first_overflow = _plan_effective_overflow(first, max_rank_peak_tokens)
    second_overflow = _plan_effective_overflow(second, max_rank_peak_tokens)
    return (
        float(not first.feasible) + float(not second.feasible),
        first_overflow + second_overflow,
        max(first_overflow, second_overflow),
        max(second.rank_loads.values()),
        max(first.rank_loads.values()),
        max(_max_adjusted_peak(first), _max_adjusted_peak(second)),
    )


def _best_single_swap_repair(
    groups: list[list[PromptStats]],
    plans: list[BatchPlan],
    bad_index: int,
    neighbor_index: int,
    *,
    solve_cache: dict[tuple[int, ...], BatchPlan],
    max_rank_peak_tokens: float,
    floor_kv_caps: dict[int, float] | None,
    adaptive_floor: bool,
    min_adaptive_floor: int,
    force_selected_floor: int | None,
    kv_safe_fixed_floor: int | None,
    rank_matching_policy: str,
    active_peak_safety_factor: float,
    max_response_len: float,
    repair_candidate_limit: int,
    ignore_tail_ties_at_response_cap: bool = False,
) -> tuple[int, int, BatchPlan, BatchPlan] | None:
    current_bad = plans[bad_index]
    current_neighbor = plans[neighbor_index]
    current_progress_score = (
        float(not current_bad.feasible) + float(not current_neighbor.feasible),
        _plan_effective_overflow(current_bad, max_rank_peak_tokens)
        + _plan_effective_overflow(current_neighbor, max_rank_peak_tokens),
        max(
            _plan_effective_overflow(current_bad, max_rank_peak_tokens),
            _plan_effective_overflow(current_neighbor, max_rank_peak_tokens),
        ),
    )

    best: tuple[
        tuple[float, float, float, float, float, float],
        int,
        int,
        BatchPlan,
        BatchPlan,
    ] | None = None
    bad_group = groups[bad_index]
    neighbor_group = groups[neighbor_index]
    bad_order = sorted(
        range(len(bad_group)),
        key=lambda idx: (bad_group[idx].load, bad_group[idx].peak_active_tokens),
    )
    bad_risk_order = sorted(
        range(len(bad_group)),
        key=lambda idx: (bad_group[idx].load, bad_group[idx].peak_active_tokens),
        reverse=True,
    )
    neighbor_order = sorted(
        range(len(neighbor_group)),
        key=lambda idx: (neighbor_group[idx].load, neighbor_group[idx].peak_active_tokens),
    )

    first_limit = min(repair_candidate_limit, len(bad_order))
    expanded_limit = min(max(repair_candidate_limit * 2, first_limit),
                         len(bad_order))
    candidate_rounds = []
    candidate_rounds.append((
        bad_order[:first_limit],
        neighbor_order[:first_limit],
    ))
    if expanded_limit > first_limit:
        candidate_rounds.append((
            bad_order[:expanded_limit],
            neighbor_order[:expanded_limit],
        ))
    candidate_rounds.append((
        bad_risk_order[:first_limit],
        neighbor_order[:expanded_limit],
    ))

    for bad_candidates, neighbor_candidates in candidate_rounds:
        best = None
        for bad_pos in bad_candidates:
            for neighbor_pos in neighbor_candidates:
                candidate_bad = list(bad_group)
                candidate_neighbor = list(neighbor_group)
                candidate_bad[bad_pos], candidate_neighbor[neighbor_pos] = (
                    candidate_neighbor[neighbor_pos],
                    candidate_bad[bad_pos],
                )
                bad_plan = _solve_group_cached(
                    candidate_bad,
                    solve_cache,
                    max_rank_peak_tokens=max_rank_peak_tokens,
                    floor_kv_caps=floor_kv_caps,
                    adaptive_floor=adaptive_floor,
                    min_adaptive_floor=min_adaptive_floor,
                    force_selected_floor=force_selected_floor,
                    kv_safe_fixed_floor=kv_safe_fixed_floor,
                    rank_matching_policy=rank_matching_policy,
                    active_peak_safety_factor=active_peak_safety_factor,
                    max_response_len=max_response_len,
                    ignore_tail_ties_at_response_cap=ignore_tail_ties_at_response_cap,
                )
                neighbor_plan = _solve_group_cached(
                    candidate_neighbor,
                    solve_cache,
                    max_rank_peak_tokens=max_rank_peak_tokens,
                    floor_kv_caps=floor_kv_caps,
                    adaptive_floor=adaptive_floor,
                    min_adaptive_floor=min_adaptive_floor,
                    force_selected_floor=force_selected_floor,
                    kv_safe_fixed_floor=kv_safe_fixed_floor,
                    rank_matching_policy=rank_matching_policy,
                    active_peak_safety_factor=active_peak_safety_factor,
                    max_response_len=max_response_len,
                    ignore_tail_ties_at_response_cap=ignore_tail_ties_at_response_cap,
                )
                if current_neighbor.feasible and not neighbor_plan.feasible:
                    continue
                score = _pair_repair_score(
                    bad_plan,
                    neighbor_plan,
                    max_rank_peak_tokens=max_rank_peak_tokens,
                )
                progress_score = score[:3]
                if progress_score >= current_progress_score:
                    continue
                if best is None or score < best[0]:
                    best = (score, bad_pos, neighbor_pos, bad_plan, neighbor_plan)

        if best is not None:
            _score, bad_pos, neighbor_pos, bad_plan, neighbor_plan = best
            return bad_pos, neighbor_pos, bad_plan, neighbor_plan

    return None


def _repair_infeasible_groups(
    groups: list[list[PromptStats]],
    *,
    max_rank_peak_tokens: float,
    floor_kv_caps: dict[int, float] | None,
    adaptive_floor: bool,
    min_adaptive_floor: int,
    force_selected_floor: int | None,
    kv_safe_fixed_floor: int | None,
    rank_matching_policy: str,
    active_peak_safety_factor: float,
    max_response_len: float,
    max_cross_step_repair_swaps: int,
    repair_candidate_limit: int,
    ignore_tail_ties_at_response_cap: bool = False,
) -> list[BatchPlan]:
    solve_cache: dict[tuple[int, ...], BatchPlan] = {}
    plans = [
        _solve_group_cached(
            group,
            solve_cache,
            max_rank_peak_tokens=max_rank_peak_tokens,
            floor_kv_caps=floor_kv_caps,
            adaptive_floor=adaptive_floor,
            min_adaptive_floor=min_adaptive_floor,
            force_selected_floor=force_selected_floor,
            kv_safe_fixed_floor=kv_safe_fixed_floor,
            rank_matching_policy=rank_matching_policy,
            active_peak_safety_factor=active_peak_safety_factor,
            max_response_len=max_response_len,
            ignore_tail_ties_at_response_cap=ignore_tail_ties_at_response_cap,
        )
        for group in groups
    ]
    global _REPAIR_TIMING_DEPTH
    repair_start = time.perf_counter()
    _REPAIR_TIMING_DEPTH += 1
    try:
        for swap_idx in range(max_cross_step_repair_swaps):
            infeasible = [
                idx for idx, plan in enumerate(plans) if not plan.feasible
            ]
            if not infeasible:
                break

            applied = False
            for bad_index in infeasible:
                for neighbor_index in _neighbor_indices(bad_index, len(groups)):
                    repair = _best_single_swap_repair(
                        groups,
                        plans,
                        bad_index,
                        neighbor_index,
                        solve_cache=solve_cache,
                        max_rank_peak_tokens=max_rank_peak_tokens,
                        floor_kv_caps=floor_kv_caps,
                        adaptive_floor=adaptive_floor,
                        min_adaptive_floor=min_adaptive_floor,
                        force_selected_floor=force_selected_floor,
                        kv_safe_fixed_floor=kv_safe_fixed_floor,
                        rank_matching_policy=rank_matching_policy,
                        active_peak_safety_factor=active_peak_safety_factor,
                        max_response_len=max_response_len,
                        repair_candidate_limit=repair_candidate_limit,
                        ignore_tail_ties_at_response_cap=(
                            ignore_tail_ties_at_response_cap
                        ),
                    )
                    if repair is None:
                        continue
                    bad_pos, neighbor_pos, bad_plan, neighbor_plan = repair
                    bad_item = groups[bad_index][bad_pos]
                    neighbor_item = groups[neighbor_index][neighbor_pos]
                    groups[bad_index][bad_pos], groups[neighbor_index][neighbor_pos] = (
                        neighbor_item,
                        bad_item,
                    )
                    plans[bad_index] = bad_plan
                    plans[neighbor_index] = neighbor_plan
                    print(
                        "[mode1 length-sorted e2e repair] "
                        f"swap={swap_idx + 1} step{bad_index + 1}:src{bad_item.source_idx} "
                        f"<-> step{neighbor_index + 1}:src{neighbor_item.source_idx} "
                        f"step{bad_index + 1}_adjusted_peak={_max_adjusted_peak(bad_plan):.2f} "
                        f"step{neighbor_index + 1}_adjusted_peak={_max_adjusted_peak(neighbor_plan):.2f} "
                        f"step{bad_index + 1}_floor={bad_plan.selected_floor} "
                        f"step{neighbor_index + 1}_floor={neighbor_plan.selected_floor}"
                    )
                    applied = True
                    break
                if applied:
                    break
            if not applied:
                break
    finally:
        _REPAIR_TIMING_DEPTH -= 1
        _record_repair_timing(repair_start)
    return plans


def _solve_batches(
    stats: list[PromptStats],
    batch_size: int,
    steps: int,
    *,
    max_rank_peak_tokens: float,
    floor_kv_caps: dict[int, float] | None,
    adaptive_floor: bool,
    min_adaptive_floor: int,
    force_selected_floor: int | None,
    force_selected_floors: list[int] | None = None,
    kv_safe_fixed_floor: int | None = None,
    rank_matching_policy: str = "release_area",
    active_peak_safety_factor: float,
    max_response_len: float,
    max_cross_step_repair_swaps: int,
    repair_candidate_limit: int,
    ignore_tail_ties_at_response_cap: bool = False,
) -> list[BatchPlan]:
    if batch_size not in (16, 32, 64):
        raise ValueError(
            f"length-sorted e2e plan expects batch_size=16, 32, or 64, "
            f"got {batch_size}")
    if len(stats) != batch_size * steps:
        raise ValueError(
            f"expected {batch_size * steps} prompt stats, got {len(stats)}")
    ordered = sorted(stats, key=lambda item: (item.load, item.source_idx))
    groups = [
        list(ordered[step * batch_size:(step + 1) * batch_size])
        for step in range(steps)
    ]
    if force_selected_floors is not None and len(force_selected_floors) != len(groups):
        raise ValueError(
            f"force_selected_floors length mismatch: "
            f"{len(force_selected_floors)} floors for {len(groups)} groups")
    if max_cross_step_repair_swaps <= 0:
        return [
            _solve_group(
                group,
                max_rank_peak_tokens=max_rank_peak_tokens,
                floor_kv_caps=floor_kv_caps,
                adaptive_floor=adaptive_floor,
                min_adaptive_floor=min_adaptive_floor,
                force_selected_floor=(
                    force_selected_floors[idx]
                    if force_selected_floors is not None
                    else force_selected_floor),
                kv_safe_fixed_floor=kv_safe_fixed_floor,
                rank_matching_policy=rank_matching_policy,
                active_peak_safety_factor=active_peak_safety_factor,
                max_response_len=max_response_len,
                ignore_tail_ties_at_response_cap=ignore_tail_ties_at_response_cap,
            ) for idx, group in enumerate(groups)
        ]
    if force_selected_floors is not None:
        return [
            _solve_group(
                group,
                max_rank_peak_tokens=max_rank_peak_tokens,
                floor_kv_caps=floor_kv_caps,
                adaptive_floor=adaptive_floor,
                min_adaptive_floor=min_adaptive_floor,
                force_selected_floor=force_selected_floors[idx],
                kv_safe_fixed_floor=kv_safe_fixed_floor,
                rank_matching_policy=rank_matching_policy,
                active_peak_safety_factor=active_peak_safety_factor,
                max_response_len=max_response_len,
                ignore_tail_ties_at_response_cap=ignore_tail_ties_at_response_cap,
            ) for idx, group in enumerate(groups)
        ]
    return _repair_infeasible_groups(
        groups,
        max_rank_peak_tokens=max_rank_peak_tokens,
        floor_kv_caps=floor_kv_caps,
        adaptive_floor=adaptive_floor,
        min_adaptive_floor=min_adaptive_floor,
        force_selected_floor=force_selected_floor,
        kv_safe_fixed_floor=kv_safe_fixed_floor,
        rank_matching_policy=rank_matching_policy,
        active_peak_safety_factor=active_peak_safety_factor,
        max_response_len=max_response_len,
        max_cross_step_repair_swaps=max_cross_step_repair_swaps,
        repair_candidate_limit=repair_candidate_limit,
        ignore_tail_ties_at_response_cap=ignore_tail_ties_at_response_cap,
    )


def _write_outputs(
    full_df: pd.DataFrame,
    plans: list[BatchPlan],
    output_train: Path,
    output_plan: Path,
    output_summary: Path,
    output_oracle: Path,
    *,
    max_rank_peak_tokens: float,
    active_peak_safety_factor: float,
    max_response_len: float,
    baseline_dirs: list[Path],
    length_ema_decay: float,
    tail_guard_ratio: float,
    tail_guard_ratio_quantile: float,
    tail_guard_ratio_window: int,
    tail_guard_sample_count: int,
    tail_guard_min_cap: int,
    tail_guard_round_to: int,
    rank_matching_policy: str,
    kv_safe_fixed_floor: int | None,
) -> None:
    if not isinstance(full_df.index, pd.RangeIndex) or list(full_df.index) != list(
        range(len(full_df))
    ):
        raise ValueError(
            "planner source_idx requires a zero-based contiguous dataset RangeIndex"
        )
    if rank_matching_policy == "release_area":
        matching_description = (
            "quota-aware exact matching that maximizes the predicted "
            "rank-token release proxy")
    elif rank_matching_policy == "min_skew":
        matching_description = (
            "KV-feasible matching that minimizes rank-internal predicted "
            "length skew without quorum constraints")
    else:
        matching_description = (
            "contiguous predicted-length order without KV-safe rank repair")
    if kv_safe_fixed_floor is not None:
        floor_description = (
            f"prefer fixed floor {kv_safe_fixed_floor} and fall back directly "
            "to floor 16 when it is theoretically or KV infeasible")
    else:
        floor_description = (
            "choose the theoretically and KV-feasible floor with the largest "
            "predicted release area"
        )

    ordered_source_indices: list[int] = []
    plan_payload: list[dict[str, Any]] = []
    row_map: dict[int, int] = {}
    row_loads: dict[str, float] = {}
    next_row = 0
    for step_idx, plan in enumerate(plans, start=1):
        expected_prompts_per_rank = len(plan.prompts) // len(ALL_RANKS)
        expected_sources = [int(item.source_idx) for item in plan.prompts]
        assigned_sources = [
            int(source_idx)
            for rank in ALL_RANKS
            for source_idx in plan.rank_to_prompt_indices[rank]
        ]
        if len(plan.prompts) % len(ALL_RANKS) != 0:
            raise ValueError(
                f"step {step_idx} prompt count is not divisible by rank count")
        if any(
            len(plan.rank_to_prompt_indices[rank]) != expected_prompts_per_rank
            for rank in ALL_RANKS
        ):
            raise ValueError(
                f"step {step_idx} does not assign an equal prompt count per rank")
        if (
            len(set(expected_sources)) != len(expected_sources)
            or len(set(assigned_sources)) != len(assigned_sources)
            or set(assigned_sources) != set(expected_sources)
        ):
            raise ValueError(
                f"step {step_idx} does not cover every source prompt exactly once")
        predicted_step_exit = max(plan.rank_loads[rank] for rank in ALL_RANKS)
        raw_tail_guard_cap = float(tail_guard_ratio) * float(predicted_step_exit)
        tail_guard_cap = min(
            int(max_response_len),
            max(
                int(tail_guard_min_cap),
                _ceil_to_multiple(raw_tail_guard_cap, int(tail_guard_round_to)),
            ),
        )
        tail_guard_enabled = int(tail_guard_cap) < int(max_response_len)
        load_by_source = {
            int(item.source_idx): float(item.load)
            for item in plan.prompts
        }
        for rank in ALL_RANKS:
            for source_idx in plan.rank_to_prompt_indices[rank]:
                ordered_source_indices.append(int(source_idx))
                row_map[int(source_idx)] = next_row
                row_loads[str(next_row)] = load_by_source[int(source_idx)]
                next_row += 1
        plan_payload.append({
            "step": step_idx,
            "selected_floor": int(plan.selected_floor),
            "theoretical_floor": int(plan.theoretical_floor),
            "kv_cap": float(plan.kv_cap),
            "schedule_thresholds": [
                float(item) for item in plan.schedule_thresholds
            ],
            "schedule_quotas": [
                [float(threshold), int(quota)]
                for threshold, quota in plan.schedule_quotas
            ],
            "release_area": float(plan.release_area),
            "release_area_unit": RELEASE_AREA_UNIT,
            "rank_matching_solver": str(plan.rank_matching_objective),
            "rank_matching_policy": str(rank_matching_policy),
            "rank_grouping_search_space": plan.rank_grouping_search_space,
            "prompts_per_rank": int(expected_prompts_per_rank),
            "kv_safe_fixed_floor": kv_safe_fixed_floor,
            "tail_guard_response_cap": int(tail_guard_cap),
            "tail_guard_enabled": bool(tail_guard_enabled),
            "tail_guard_ratio": float(tail_guard_ratio),
            "tail_guard_ratio_quantile": float(tail_guard_ratio_quantile),
            "tail_guard_ratio_window": int(tail_guard_ratio_window),
            "tail_guard_ratio_sample_count": int(tail_guard_sample_count),
            "tail_guard_predicted_step_exit": float(predicted_step_exit),
            "tail_guard_raw_cap": float(raw_tail_guard_cap),
            "tail_guard_min_cap": int(tail_guard_min_cap),
            "tail_guard_round_to": int(tail_guard_round_to),
            "tail_guard_formula": (
                "min(max_response_len, max(min_cap, "
                "ceil_to_multiple(ratio_q * predicted_step_exit, round_to)))"
            ),
            "shrink_stages": [int(stage) for stage in plan.shrink_stages],
            "stage_survivor_ranks": [
                [int(rank) for rank in ranks]
                for ranks in plan.stage_survivor_ranks
            ],
            "intermediate_survivor_ranks": [
                int(rank) for rank in plan.intermediate_survivor_ranks
            ],
            "final_survivor_ranks": [
                int(rank) for rank in plan.final_survivor_ranks
            ],
            "rank_to_dataset_item_idx": {
                str(rank): [
                    row_map[int(source_idx)]
                    for source_idx in plan.rank_to_prompt_indices[rank]
                ]
                for rank in ALL_RANKS
            },
            "rank_to_source_idx": {
                str(rank): [
                    int(source_idx)
                    for source_idx in plan.rank_to_prompt_indices[rank]
                ]
                for rank in ALL_RANKS
            },
            "rank_loads": {
                str(rank): float(plan.rank_loads[rank])
                for rank in ALL_RANKS
            },
            "rank_load_gaps": {
                str(rank): float(plan.rank_load_gaps[rank])
                for rank in ALL_RANKS
            },
            "rank_peak_loads": {
                str(rank): float(plan.rank_peak_loads[rank])
                for rank in ALL_RANKS
            },
            "rank_adjusted_peak_loads": {
                str(rank): float(plan.rank_adjusted_peak_loads[rank])
                for rank in ALL_RANKS
            },
            "rank_token_sums": {
                str(rank): float(plan.rank_token_sums[rank])
                for rank in ALL_RANKS
            },
            "feasible": bool(plan.feasible),
        })

    out_df = full_df.iloc[ordered_source_indices].copy()
    extra_info_values: list[dict[str, Any]] = []
    if "extra_info" in out_df.columns:
        raw_extra_info = out_df["extra_info"].tolist()
    else:
        raw_extra_info = [None] * len(out_df)
    for source_idx, extra_info in zip(
        ordered_source_indices, raw_extra_info, strict=True
    ):
        if extra_info is None:
            preserved: dict[str, Any] = {}
        elif isinstance(extra_info, dict):
            preserved = dict(extra_info)
        else:
            raise ValueError(
                "dataset extra_info must be a mapping or null, got "
                f"{type(extra_info).__name__} for source_idx={source_idx}"
            )
        for identity_field in ("index", "prompt_occurrence_ordinal"):
            if identity_field not in preserved:
                continue
            existing_identity = preserved[identity_field]
            if (
                isinstance(existing_identity, bool)
                or not isinstance(existing_identity, (int, np.integer))
                or int(existing_identity) != int(source_idx)
            ):
                raise ValueError(
                    f"dataset extra_info.{identity_field} already carries a "
                    "conflicting identity for "
                    f"source_idx={source_idx}: {existing_identity!r}"
                )
        preserved["index"] = int(source_idx)
        preserved["prompt_occurrence_ordinal"] = int(source_idx)
        extra_info_values.append(preserved)
    out_df["extra_info"] = extra_info_values
    output_train.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(output_train, index=False)
    output_plan.write_text(
        json.dumps(plan_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    output_oracle.write_text(
        json.dumps(row_loads, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")

    summary: list[dict[str, Any]] = []
    for step_idx, plan in enumerate(plans, start=1):
        loads = [plan.rank_loads[rank] for rank in ALL_RANKS]
        peaks = [plan.rank_peak_loads[rank] for rank in ALL_RANKS]
        adjusted_peaks = [
            plan.rank_adjusted_peak_loads[rank] for rank in ALL_RANKS
        ]
        load_gaps = [plan.rank_load_gaps[rank] for rank in ALL_RANKS]
        sums = [plan.rank_token_sums[rank] for rank in ALL_RANKS]
        predicted_step_exit = max(loads)
        raw_tail_guard_cap = float(tail_guard_ratio) * float(predicted_step_exit)
        tail_guard_cap = min(
            int(max_response_len),
            max(
                int(tail_guard_min_cap),
                _ceil_to_multiple(raw_tail_guard_cap, int(tail_guard_round_to)),
            ),
        )
        tail_guard_enabled = int(tail_guard_cap) < int(max_response_len)
        summary.append({
            "step": step_idx,
            "feasible": plan.feasible,
            "selected_floor": int(plan.selected_floor),
            "theoretical_floor": int(plan.theoretical_floor),
            "shrink_stages": [int(stage) for stage in plan.shrink_stages],
            "stage_survivor_ranks": [
                [int(rank) for rank in ranks]
                for ranks in plan.stage_survivor_ranks
            ],
            "intermediate_survivor_ranks": [
                int(rank) for rank in plan.intermediate_survivor_ranks
            ],
            "final_survivor_ranks": [
                int(rank) for rank in plan.final_survivor_ranks
            ],
            "rank_load_range": [min(loads), max(loads)],
            "rank_load_gap_range": [min(load_gaps), max(load_gaps)],
            "rank_load_gap_sum": sum(load_gaps),
            "rank_peak_range": [min(peaks), max(peaks)],
            "rank_adjusted_peak_range": [
                min(adjusted_peaks),
                max(adjusted_peaks),
            ],
            "rank_token_sum_range": [min(sums), max(sums)],
            "max_rank_peak_tokens": max(peaks),
            "max_adjusted_rank_peak_tokens": max(adjusted_peaks),
            "kv_cap": float(plan.kv_cap),
            "schedule_thresholds": [
                float(item) for item in plan.schedule_thresholds
            ],
            "schedule_quotas": [
                [float(threshold), int(quota)]
                for threshold, quota in plan.schedule_quotas
            ],
            "release_area": float(plan.release_area),
            "release_area_unit": RELEASE_AREA_UNIT,
            "rank_matching_solver": str(plan.rank_matching_objective),
            "rank_matching_policy": str(rank_matching_policy),
            "rank_grouping_search_space": plan.rank_grouping_search_space,
            "prompts_per_rank": int(len(plan.prompts) // len(ALL_RANKS)),
            "kv_safe_fixed_floor": kv_safe_fixed_floor,
            "tail_guard_response_cap": int(tail_guard_cap),
            "tail_guard_enabled": bool(tail_guard_enabled),
            "tail_guard_ratio": float(tail_guard_ratio),
            "tail_guard_ratio_quantile": float(tail_guard_ratio_quantile),
            "tail_guard_ratio_window": int(tail_guard_ratio_window),
            "tail_guard_ratio_sample_count": int(tail_guard_sample_count),
            "tail_guard_raw_cap": float(raw_tail_guard_cap),
            "tail_guard_min_cap": int(tail_guard_min_cap),
            "tail_guard_round_to": int(tail_guard_round_to),
            "tail_guard_prompt_tail_stat": "max_response_over_rollout_n",
            "tail_guard_ratio_definition": (
                "Q_q(max(1, actual_prompt_max_tail / "
                "predicted_prompt_max_tail)) over a sliding window of "
                "recent adjacent historical epochs"
            ),
            "active_peak_safety_factor": float(active_peak_safety_factor),
            "max_response_len": float(max_response_len),
            "length_prediction_mode": (
                "prompt_max_ema_history"
                if len(baseline_dirs) > 1 else "single_epoch_prompt_max"),
            "length_prediction_baseline_dirs": [
                str(path) for path in baseline_dirs
            ],
            "length_ema_decay": float(length_ema_decay),
            "kv_length_source": "latest_observed_response_multiset",
            "adjusted_peak_definition": "A(min(mu * response_len, max_response_len))",
            "rank_matching_objective": (
                "preserve length-sorted step buckets first; "
                f"{floor_description}; {matching_description}; repair across "
                "neighboring steps only when the configured policy permits it; "
                "the matching optimum is limited to rank_grouping_search_space"
            ),
            "predicted_step_exit": max(loads),
        })
    output_summary.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"[mode1 length-sorted e2e plan] train={output_train}")
    print(f"[mode1 length-sorted e2e plan] plan={output_plan}")
    print(f"[mode1 length-sorted e2e plan] oracle={output_oracle}")


def _parse_floor_kv_caps(value: str | None,
                         fallback_cap: float) -> dict[int, float]:
    caps = dict(DEFAULT_FLOOR_KV_CAPS)
    caps[16] = float(fallback_cap)
    if value:
        for item in value.split(","):
            item = item.strip()
            if not item:
                continue
            if ":" not in item:
                raise ValueError(
                    "floor KV cap entries must use FLOOR:CAP format, "
                    f"got {item!r}")
            floor_text, cap_text = item.split(":", 1)
            floor = int(floor_text)
            if floor not in FLOOR_CANDIDATES:
                raise ValueError(
                    f"unsupported floor KV cap floor={floor}; "
                    f"expected one of {FLOOR_CANDIDATES}")
            caps[floor] = float(cap_text)
    return caps


def _parse_forced_floors(value: str | None, steps: int) -> list[int] | None:
    if value is None:
        return None
    floors: list[int] = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        floor = int(item)
        if floor not in FLOOR_CANDIDATES:
            raise ValueError(
                f"unsupported forced floor={floor}; "
                f"expected one of {FLOOR_CANDIDATES}")
        floors.append(floor)
    if not floors:
        return None
    if len(floors) != steps:
        raise ValueError(
            f"--force-selected-floors length must match --steps: "
            f"got {len(floors)} floors for {steps} steps")
    return floors


def _ceil_to_multiple(value: float, multiple: int) -> int:
    if multiple <= 1:
        return int(np.ceil(value))
    return int(np.ceil(value / float(multiple)) * multiple)


def _tail_guard_underestimate_ratio(
    baseline_history: list[dict[str, PromptStats]],
    *,
    ema_decay: float,
    quantile: float,
    default_ratio: float,
    window: int,
) -> tuple[float, int]:
    """Calibrate prompt-level max-tail underestimation from history.

    For history dirs d0, d1, ..., dk, predict dj from the EMA of d0..d(j-1)
    and compare prompt-level max response lengths.  The returned ratio is the
    high quantile of max(1, actual_max / predicted_max) over the most recent
    `window` adjacent epoch transitions.
    """
    if len(baseline_history) < 2:
        return float(default_ratio), 0

    if not 0.0 <= quantile <= 1.0:
        raise ValueError(
            f"--tail-guard-ratio-quantile must be in [0, 1], got {quantile}")
    if window <= 0:
        raise ValueError(f"--tail-guard-ratio-window must be positive, got {window}")

    ema_by_input: dict[str, float] = {}
    transition_ratios: list[list[float]] = []
    eps = 1.0
    for idx, current in enumerate(baseline_history):
        if idx > 0:
            current_ratios: list[float] = []
            for prompt_input, actual in current.items():
                predicted = ema_by_input.get(prompt_input)
                if predicted is None:
                    continue
                actual_tail = float(actual.max_len)
                current_ratios.append(max(
                    1.0, actual_tail / max(predicted, eps)))
            if current_ratios:
                transition_ratios.append(current_ratios)

        for prompt_input, stat in current.items():
            ema_by_input[prompt_input] = _update_prompt_tail_ema(
                ema_by_input.get(prompt_input),
                stat.max_len,
                ema_decay,
            )

    recent_transitions = transition_ratios[-int(window):]
    ratios = [
        ratio
        for transition in recent_transitions
        for ratio in transition
    ]
    if not ratios:
        return float(default_ratio), 0
    return float(np.quantile(np.asarray(ratios, dtype=np.float64), quantile)), len(ratios)


def main() -> None:
    global _REPAIR_TIMING_DEPTH, _TIMING_PROFILE
    planner_main_start = time.perf_counter()
    phase_seconds: dict[str, float] = {}
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-dir", action="append", required=True)
    parser.add_argument("--length-ema-decay", type=float, default=0.3)
    parser.add_argument("--train-file", required=True)
    parser.add_argument(
        "--tokenizer-path",
        default=None,
        help=(
            "Optional local tokenizer used to render dataset prompts exactly "
            "as they appear in rollout history."),
    )
    parser.add_argument("--output-train", required=True)
    parser.add_argument("--output-plan", required=True)
    parser.add_argument("--output-summary", required=True)
    parser.add_argument("--output-oracle", required=True)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--responses-per-prompt", type=int, default=16)
    parser.add_argument("--dataset-fraction", type=float, default=0.005)
    parser.add_argument("--max-samples", type=int, default=-1)
    parser.add_argument(
        "--repeat-prompt-set-to-fill",
        action="store_true",
        help=(
            "Repeat the matched historical prompt set until batch_size * "
            "steps rows are available. This is only for short transition "
            "stress tests and is disabled for performance experiments."))
    parser.add_argument("--max-rank-peak-tokens", type=float, default=280576.0)
    parser.add_argument(
        "--adaptive-floor",
        action="store_true",
        help=(
            "Choose the release-area maximizing per-step final floor from "
            "2/4/8/16. The ordered prompt batches stay fixed unless a batch "
            "remains KV-infeasible even at "
            "floor16, in which case bounded neighbor repair may swap prompts."))
    parser.add_argument("--min-adaptive-floor", type=int, default=2)
    parser.add_argument(
        "--force-selected-floor",
        type=int,
        default=0,
        help=(
            "Force every step to use this selected floor. This is intended for "
            "runtime capacity probes, e.g. validating floor2 without allowing "
            "a tail-tied smoke-test step to expand back to floor16."))
    parser.add_argument(
        "--force-selected-floors",
        default=None,
        help=(
            "Comma-separated per-step selected floors, e.g. 2,2,4. This is "
            "intended for short transition probes and disables cross-step "
            "repair to preserve the requested floor sequence."))
    parser.add_argument(
        "--floor-kv-caps",
        default=None,
        help=(
            "Comma-separated floor:cap mapping, e.g. "
            "2:147456,4:280576,8:377344,16:377344."))
    parser.add_argument(
        "--rank-matching-policy",
        choices=RANK_MATCHING_POLICIES,
        default="release_area",
        help=(
            "Rank assignment objective inside each fixed length-sorted step. "
            "release_area is AdaFloor, min_skew is the matching ablation, and "
            "contiguous preserves pure length order."))
    parser.add_argument(
        "--kv-safe-fixed-floor",
        type=int,
        default=0,
        help=(
            "Prefer this fixed floor when it is theoretically and KV feasible, "
            "otherwise fall back directly to floor16."))
    parser.add_argument("--active-peak-safety-factor", type=float, default=1.16)
    parser.add_argument("--max-response-len", type=float, default=DEFAULT_MAX_RESPONSE_LEN)
    parser.add_argument(
        "--ignore-tail-ties-at-response-cap",
        action="store_true",
        help=(
            "For short max-response smoke tests, do not treat many "
            "max_len==max_response_len ties as proof that shrink is impossible. "
            "The original theoretical floor is still reported."))
    parser.add_argument("--max-cross-step-repair-swaps", type=int, default=8)
    parser.add_argument("--repair-candidate-limit", type=int, default=8)
    parser.add_argument("--tail-guard-ratio-quantile", type=float, default=0.95)
    parser.add_argument("--tail-guard-ratio-window", type=int, default=3)
    parser.add_argument("--tail-guard-default-ratio", type=float, default=1.20)
    parser.add_argument("--tail-guard-min-cap", type=int, default=4096)
    parser.add_argument("--tail-guard-round-to", type=int, default=512)
    parser.add_argument("--allow-infeasible", action="store_true")
    parser.add_argument(
        "--require-compact-history",
        action="store_true",
        help=(
            "Require each baseline directory to contain "
            f"{OFFLINE_PLANNING_HISTORY_FILENAME}."
        ),
    )
    parser.add_argument(
        "--timing-output",
        default=None,
        help="Optional path for a JSON breakdown of offline planning time.",
    )
    args = parser.parse_args()
    _REPAIR_TIMING_DEPTH = 0
    _TIMING_PROFILE = None
    if args.timing_output:
        _TIMING_PROFILE = {
            "graph_construction": 0.0,
            "milp_solving": 0.0,
            "repair": 0.0,
        }

    baseline_dirs = [Path(item) for item in args.baseline_dir]
    _progress(
        "start "
        f"baseline_dirs={len(baseline_dirs)} steps={args.steps} "
        f"batch_size={args.batch_size} responses_per_prompt={args.responses_per_prompt} "
        f"dataset_fraction={args.dataset_fraction} max_samples={args.max_samples}"
    )
    history_update_start = time.perf_counter()
    baseline_history = _read_baseline_history(
        baseline_dirs,
        args.steps,
        args.responses_per_prompt,
        require_compact_history=args.require_compact_history,
    )
    _progress("tail guard calibration start")
    tail_guard_ratio, tail_guard_sample_count = _tail_guard_underestimate_ratio(
        baseline_history,
        ema_decay=args.length_ema_decay,
        quantile=args.tail_guard_ratio_quantile,
        default_ratio=args.tail_guard_default_ratio,
        window=args.tail_guard_ratio_window,
    )
    print(
        "[mode1 length-sorted e2e plan] tail guard "
        f"ratio_q={args.tail_guard_ratio_quantile} "
        f"window={args.tail_guard_ratio_window} "
        f"ratio={tail_guard_ratio:.6f} samples={tail_guard_sample_count} "
        f"min_cap={args.tail_guard_min_cap} round_to={args.tail_guard_round_to}",
        flush=True,
    )
    _progress("baseline length prediction read start")
    stats_by_input = _read_ema_baseline_stats(
        baseline_history,
        args.length_ema_decay,
    )
    phase_seconds["history_update"] = (
        time.perf_counter() - history_update_start
    )
    _progress(
        "baseline length prediction read done "
        f"unique_prompts={len(stats_by_input)}"
    )
    _progress(f"dataset map start train_file={args.train_file}")
    phase_start = time.perf_counter()
    prompt_tokenizer = None
    if args.tokenizer_path:
        from transformers import AutoTokenizer

        prompt_tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer_path,
            trust_remote_code=True,
            local_files_only=True,
        )
    full_df, stats = _map_stats_to_dataset(
        Path(args.train_file),
        stats_by_input,
        args.dataset_fraction,
        args.max_samples,
        args.batch_size * args.steps,
        repeat_prompt_set_to_fill=args.repeat_prompt_set_to_fill,
        prompt_tokenizer=prompt_tokenizer,
    )
    phase_seconds["dataset_mapping"] = time.perf_counter() - phase_start
    _progress(
        "dataset map done "
        f"full_rows={len(full_df)} matched_stats={len(stats)}"
    )
    floor_kv_caps = _parse_floor_kv_caps(
        args.floor_kv_caps, args.max_rank_peak_tokens)
    force_selected_floors = _parse_forced_floors(
        args.force_selected_floors, args.steps)
    kv_safe_fixed_floor = (
        int(args.kv_safe_fixed_floor)
        if int(args.kv_safe_fixed_floor) > 0 else None)
    if kv_safe_fixed_floor is not None and (
            int(args.force_selected_floor) > 0
            or force_selected_floors is not None):
        raise ValueError(
            "--kv-safe-fixed-floor cannot be combined with forced floors")
    _progress(
        "solve batches start "
        f"adaptive_floor={args.adaptive_floor} min_floor={args.min_adaptive_floor} "
        f"force_floor={args.force_selected_floor} force_floors={force_selected_floors} "
        f"kv_safe_fixed_floor={kv_safe_fixed_floor} "
        f"rank_matching_policy={args.rank_matching_policy} "
        f"floor_kv_caps={floor_kv_caps}"
    )
    phase_start = time.perf_counter()
    plans = _solve_batches(
        stats,
        args.batch_size,
        args.steps,
        max_rank_peak_tokens=args.max_rank_peak_tokens,
        floor_kv_caps=floor_kv_caps,
        adaptive_floor=args.adaptive_floor,
        min_adaptive_floor=args.min_adaptive_floor,
        force_selected_floor=(
            int(args.force_selected_floor)
            if int(args.force_selected_floor) > 0 else None),
        force_selected_floors=force_selected_floors,
        kv_safe_fixed_floor=kv_safe_fixed_floor,
        rank_matching_policy=args.rank_matching_policy,
        active_peak_safety_factor=args.active_peak_safety_factor,
        max_response_len=args.max_response_len,
        max_cross_step_repair_swaps=args.max_cross_step_repair_swaps,
        repair_candidate_limit=args.repair_candidate_limit,
        ignore_tail_ties_at_response_cap=args.ignore_tail_ties_at_response_cap,
    )
    phase_seconds["plan_construction_and_solve"] = (
        time.perf_counter() - phase_start
    )
    _progress(
        "solve batches done "
        f"selected_floors={[int(plan.selected_floor) for plan in plans]}"
    )
    phase_start = time.perf_counter()
    infeasible_steps = [
        idx for idx, plan in enumerate(plans, start=1) if not plan.feasible
    ]
    if infeasible_steps and not args.allow_infeasible:
        worst_single = max(stats, key=lambda item: _peak_active_tokens(tuple(
            min(float(length) * args.active_peak_safety_factor,
                args.max_response_len)
            for length in item.lengths
        )))
        worst_single_adjusted_peak = _peak_active_tokens(tuple(
            min(float(length) * args.active_peak_safety_factor,
                args.max_response_len)
            for length in worst_single.lengths
        ))
        raise RuntimeError(
            "infeasible length-sorted e2e plan under "
            f"active_peak_safety_factor={args.active_peak_safety_factor}: "
            f"steps={infeasible_steps}, "
            f"max_response_len={args.max_response_len:.2f}, "
            f"worst_single_adjusted_peak={worst_single_adjusted_peak:.2f}, "
            f"worst_single_raw_peak={worst_single.peak_active_tokens:.2f}, "
            f"worst_single_source_idx={worst_single.source_idx}. "
            "Lower the safety factor, increase the KV cap, reduce rollout.n/max "
            "response length, or allow infeasible output explicitly.")
    phase_seconds["plan_validation"] = time.perf_counter() - phase_start
    _progress("write outputs start")
    phase_start = time.perf_counter()
    _write_outputs(
        full_df,
        plans,
        Path(args.output_train),
        Path(args.output_plan),
        Path(args.output_summary),
        Path(args.output_oracle),
        max_rank_peak_tokens=args.max_rank_peak_tokens,
        active_peak_safety_factor=args.active_peak_safety_factor,
        max_response_len=args.max_response_len,
        baseline_dirs=baseline_dirs,
        length_ema_decay=args.length_ema_decay,
        tail_guard_ratio=tail_guard_ratio,
        tail_guard_ratio_quantile=args.tail_guard_ratio_quantile,
        tail_guard_ratio_window=args.tail_guard_ratio_window,
        tail_guard_sample_count=tail_guard_sample_count,
        tail_guard_min_cap=args.tail_guard_min_cap,
        tail_guard_round_to=args.tail_guard_round_to,
        rank_matching_policy=args.rank_matching_policy,
        kv_safe_fixed_floor=kv_safe_fixed_floor,
    )
    phase_seconds["output_materialization"] = time.perf_counter() - phase_start
    _progress("write outputs done")

    planner_main_seconds = time.perf_counter() - planner_main_start
    measured_phase_seconds = sum(phase_seconds.values())
    if args.timing_output:
        assert _TIMING_PROFILE is not None
        component_seconds = {
            "history_update": phase_seconds["history_update"],
            **_TIMING_PROFILE,
        }
        timing_output = Path(args.timing_output)
        timing_output.parent.mkdir(parents=True, exist_ok=True)
        timing_payload = {
            "schema_version": 2,
            "clock": "time.perf_counter",
            "planner_main_seconds": planner_main_seconds,
            "component_seconds": component_seconds,
            "component_definitions": {
                "history_update": (
                    "Tail guard calibration and EMA update from historical "
                    "rollout traces."
                ),
                "graph_construction": (
                    "KV pair metrics, candidate schedule and constraint "
                    "construction, and NetworkX graph matching outside repair."
                ),
                "milp_solving": (
                    "Wall time inside SciPy milp calls outside repair."
                ),
                "repair": (
                    "Bounded cross step repair search, including any graph "
                    "and MILP work repeated by repair."
                ),
            },
            "phase_seconds": phase_seconds,
            "unattributed_seconds": max(
                0.0, planner_main_seconds - measured_phase_seconds
            ),
            "baseline_dirs": [str(item) for item in baseline_dirs],
            "train_file": str(args.train_file),
            "steps": int(args.steps),
            "batch_size": int(args.batch_size),
            "responses_per_prompt": int(args.responses_per_prompt),
            "selected_floors": [int(plan.selected_floor) for plan in plans],
            "rank_matching_policy": str(args.rank_matching_policy),
            "floor_kv_caps": {
                str(floor): float(cap)
                for floor, cap in floor_kv_caps.items()
            },
        }
        timing_output.write_text(
            json.dumps(timing_payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        _progress(f"timing output={timing_output}")


if __name__ == "__main__":
    main()
