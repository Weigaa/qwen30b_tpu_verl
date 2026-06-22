#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import networkx as nx

from build_mode1_optimized_rank_plan import (
    PromptStats,
    _map_stats_to_dataset,
    _peak_active_tokens,
    _read_baseline_stats,
)


ALL_RANKS = list(range(16))
DEFAULT_MAX_RESPONSE_LEN = 16384.0


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


PairMetric = tuple[float, float, float, float, float, list[int]]


def _pair_metrics(
    group: list[PromptStats],
    *,
    active_peak_safety_factor: float,
    max_response_len: float,
) -> dict[tuple[int, int], PairMetric]:
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
            metrics[(first, second)] = (
                _peak_active_tokens(response_lengths),
                _peak_active_tokens(adjusted_lengths),
                float(left.sum_len + right.sum_len),
                float(max(left.load, right.load)),
                float(abs(left.load - right.load)),
                [int(left.source_idx), int(right.source_idx)],
            )
    return metrics


def _capacity_constrained_load_gap_matching(
    group: list[PromptStats],
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
    if len(matching) * 2 != item_count:
        return None
    return {(min(first, second), max(first, second)) for first, second in matching}


def _has_perfect_matching_under_adjusted_peak(
    item_count: int,
    metrics: dict[tuple[int, int], PairMetric],
    adjusted_peak_limit: float,
) -> bool:
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
    return len(matching) * 2 == item_count


def _solve_one_batch(
    batch: list[PromptStats],
    *,
    max_rank_peak_tokens: float,
    active_peak_safety_factor: float,
    max_response_len: float,
) -> BatchPlan:
    if len(batch) != 32:
        raise ValueError(f"one e2e batch requires 32 prompts, got {len(batch)}")
    pairs, feasible = _capacity_constrained_load_gap_matching(
        batch,
        max_rank_peak_tokens=max_rank_peak_tokens,
        active_peak_safety_factor=active_peak_safety_factor,
        max_response_len=max_response_len,
    )

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

    return BatchPlan(
        prompts=list(batch),
        rank_to_prompt_indices=rank_map,
        rank_loads=loads,
        rank_load_gaps=load_gaps,
        rank_peak_loads=peak_loads,
        rank_adjusted_peak_loads=adjusted_peak_loads,
        rank_token_sums=token_sums,
        feasible=feasible,
    )


def _max_adjusted_peak(plan: BatchPlan) -> float:
    return max(plan.rank_adjusted_peak_loads.values())


def _plan_overflow(plan: BatchPlan, max_rank_peak_tokens: float) -> float:
    return max(0.0, _max_adjusted_peak(plan) - max_rank_peak_tokens)


def _solve_group(
    group: list[PromptStats],
    *,
    max_rank_peak_tokens: float,
    active_peak_safety_factor: float,
    max_response_len: float,
) -> BatchPlan:
    return _solve_one_batch(
        group,
        max_rank_peak_tokens=max_rank_peak_tokens,
        active_peak_safety_factor=active_peak_safety_factor,
        max_response_len=max_response_len,
    )


def _group_cache_key(group: list[PromptStats]) -> tuple[int, ...]:
    return tuple(int(item.source_idx) for item in group)


def _solve_group_cached(
    group: list[PromptStats],
    cache: dict[tuple[int, ...], BatchPlan],
    *,
    max_rank_peak_tokens: float,
    active_peak_safety_factor: float,
    max_response_len: float,
) -> BatchPlan:
    key = _group_cache_key(group)
    cached = cache.get(key)
    if cached is not None:
        return cached
    plan = _solve_group(
        group,
        max_rank_peak_tokens=max_rank_peak_tokens,
        active_peak_safety_factor=active_peak_safety_factor,
        max_response_len=max_response_len,
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
    first_overflow = _plan_overflow(first, max_rank_peak_tokens)
    second_overflow = _plan_overflow(second, max_rank_peak_tokens)
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
    active_peak_safety_factor: float,
    max_response_len: float,
    repair_candidate_limit: int,
) -> tuple[int, int, BatchPlan, BatchPlan] | None:
    current_bad = plans[bad_index]
    current_neighbor = plans[neighbor_index]
    current_progress_score = (
        float(not current_bad.feasible) + float(not current_neighbor.feasible),
        _plan_overflow(current_bad, max_rank_peak_tokens)
        + _plan_overflow(current_neighbor, max_rank_peak_tokens),
        max(
            _plan_overflow(current_bad, max_rank_peak_tokens),
            _plan_overflow(current_neighbor, max_rank_peak_tokens),
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
                    active_peak_safety_factor=active_peak_safety_factor,
                    max_response_len=max_response_len,
                )
                neighbor_plan = _solve_group_cached(
                    candidate_neighbor,
                    solve_cache,
                    max_rank_peak_tokens=max_rank_peak_tokens,
                    active_peak_safety_factor=active_peak_safety_factor,
                    max_response_len=max_response_len,
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
    active_peak_safety_factor: float,
    max_response_len: float,
    max_cross_step_repair_swaps: int,
    repair_candidate_limit: int,
) -> list[BatchPlan]:
    solve_cache: dict[tuple[int, ...], BatchPlan] = {}
    plans = [
        _solve_group_cached(
            group,
            solve_cache,
            max_rank_peak_tokens=max_rank_peak_tokens,
            active_peak_safety_factor=active_peak_safety_factor,
            max_response_len=max_response_len,
        )
        for group in groups
    ]
    for swap_idx in range(max_cross_step_repair_swaps):
        infeasible = [idx for idx, plan in enumerate(plans) if not plan.feasible]
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
                    active_peak_safety_factor=active_peak_safety_factor,
                    max_response_len=max_response_len,
                    repair_candidate_limit=repair_candidate_limit,
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
                    f"step{neighbor_index + 1}_adjusted_peak={_max_adjusted_peak(neighbor_plan):.2f}"
                )
                applied = True
                break
            if applied:
                break
        if not applied:
            break
    return plans


def _solve_batches(
    stats: list[PromptStats],
    batch_size: int,
    steps: int,
    *,
    max_rank_peak_tokens: float,
    active_peak_safety_factor: float,
    max_response_len: float,
    max_cross_step_repair_swaps: int,
    repair_candidate_limit: int,
) -> list[BatchPlan]:
    if batch_size != 32:
        raise ValueError(f"length-sorted e2e plan expects batch_size=32, got {batch_size}")
    if len(stats) != batch_size * steps:
        raise ValueError(
            f"expected {batch_size * steps} prompt stats, got {len(stats)}")
    ordered = sorted(stats, key=lambda item: (item.load, item.source_idx))
    groups = [
        list(ordered[step * batch_size:(step + 1) * batch_size])
        for step in range(steps)
    ]
    return _repair_infeasible_groups(
        groups,
        max_rank_peak_tokens=max_rank_peak_tokens,
        active_peak_safety_factor=active_peak_safety_factor,
        max_response_len=max_response_len,
        max_cross_step_repair_swaps=max_cross_step_repair_swaps,
        repair_candidate_limit=repair_candidate_limit,
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
) -> None:
    ordered_source_indices: list[int] = []
    plan_payload: list[dict[str, Any]] = []
    row_map: dict[int, int] = {}
    row_loads: dict[str, float] = {}
    next_row = 0
    for step_idx, plan in enumerate(plans, start=1):
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
        summary.append({
            "step": step_idx,
            "feasible": plan.feasible,
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
            "kv_cap": float(max_rank_peak_tokens),
            "active_peak_safety_factor": float(active_peak_safety_factor),
            "max_response_len": float(max_response_len),
            "adjusted_peak_definition": "A(min(mu * response_len, max_response_len))",
            "rank_matching_objective": (
                "minimize sum |m_i-m_j| among pairings with adjusted peak <= kv_cap; "
                "fallback to minimum feasible adjusted peak only when no cap-feasible "
                "perfect matching exists"
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-dir", required=True)
    parser.add_argument("--train-file", required=True)
    parser.add_argument("--output-train", required=True)
    parser.add_argument("--output-plan", required=True)
    parser.add_argument("--output-summary", required=True)
    parser.add_argument("--output-oracle", required=True)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--responses-per-prompt", type=int, default=16)
    parser.add_argument("--dataset-fraction", type=float, default=0.005)
    parser.add_argument("--max-samples", type=int, default=-1)
    parser.add_argument("--max-rank-peak-tokens", type=float, default=280576.0)
    parser.add_argument("--active-peak-safety-factor", type=float, default=1.16)
    parser.add_argument("--max-response-len", type=float, default=DEFAULT_MAX_RESPONSE_LEN)
    parser.add_argument("--max-cross-step-repair-swaps", type=int, default=8)
    parser.add_argument("--repair-candidate-limit", type=int, default=8)
    parser.add_argument("--allow-infeasible", action="store_true")
    args = parser.parse_args()

    stats_by_input = _read_baseline_stats(
        Path(args.baseline_dir), args.steps, args.responses_per_prompt)
    full_df, stats = _map_stats_to_dataset(
        Path(args.train_file),
        stats_by_input,
        args.dataset_fraction,
        args.max_samples,
        args.batch_size * args.steps,
    )
    plans = _solve_batches(
        stats,
        args.batch_size,
        args.steps,
        max_rank_peak_tokens=args.max_rank_peak_tokens,
        active_peak_safety_factor=args.active_peak_safety_factor,
        max_response_len=args.max_response_len,
        max_cross_step_repair_swaps=args.max_cross_step_repair_swaps,
        repair_candidate_limit=args.repair_candidate_limit,
    )
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
    )


if __name__ == "__main__":
    main()
