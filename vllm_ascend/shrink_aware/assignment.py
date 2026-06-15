from __future__ import annotations

import heapq
from dataclasses import dataclass
from typing import Mapping, Sequence

from vllm_ascend.shrink_aware.planner import RankRolePlan


@dataclass(frozen=True)
class PromptAssignment:
    prompt_index: int
    rank: int
    role: str
    predicted_load: float


@dataclass(frozen=True)
class PromptAssignmentPlan:
    assignments: list[PromptAssignment]
    per_rank_counts: dict[int, int]
    per_rank_predicted_load: dict[int, float]
    role_by_rank: dict[int, str]


def assign_prompts_to_ranks(
    predicted_lengths: Sequence[float],
    role_plan: RankRolePlan,
) -> PromptAssignmentPlan:
    loads = [float(length) for length in predicted_lengths]
    if any(load < 0 for load in loads):
        raise ValueError("predicted_lengths cannot contain negative values")

    rank_roles: list[tuple[int, str]] = (
        [(rank, "donor") for rank in role_plan.donor_ranks] +
        [(rank, "wave2") for rank in role_plan.wave2_ranks] +
        [(rank, "survivor") for rank in role_plan.final_survivor_ranks])
    if not rank_roles and loads:
        raise ValueError("at least one target rank is required for prompt assignment")

    role_targets = _role_targets(len(loads), role_plan)
    sorted_items = sorted(
        enumerate(loads), key=lambda item: (item[1], item[0]))
    role_buckets = {
        "donor": sorted_items[:role_targets["donor"]],
        "wave2": sorted_items[role_targets["donor"]:
                              role_targets["donor"] + role_targets["wave2"]],
        "survivor": sorted_items[role_targets["donor"] +
                                 role_targets["wave2"]:],
    }

    assignments: list[PromptAssignment] = []
    for role, ranks in (
        ("donor", role_plan.donor_ranks),
        ("wave2", role_plan.wave2_ranks),
        ("survivor", role_plan.final_survivor_ranks),
    ):
        assignments.extend(_assign_role(role_buckets[role], ranks, role))

    assignments.sort(key=lambda item: item.prompt_index)
    seen = [item.prompt_index for item in assignments]
    if seen != list(range(len(loads))):
        raise RuntimeError("internal error: not all prompts were assigned exactly once")

    per_rank_counts = {rank: 0 for rank, _ in rank_roles}
    per_rank_load = {rank: 0.0 for rank, _ in rank_roles}
    role_by_rank = {rank: role for rank, role in rank_roles}
    for item in assignments:
        per_rank_counts[item.rank] = per_rank_counts.get(item.rank, 0) + 1
        per_rank_load[item.rank] = (
            per_rank_load.get(item.rank, 0.0) + item.predicted_load)

    return PromptAssignmentPlan(
        assignments=assignments,
        per_rank_counts=per_rank_counts,
        per_rank_predicted_load=per_rank_load,
        role_by_rank=role_by_rank,
    )


def build_reorder_indices(assignments: Sequence[PromptAssignment],
                          rank_order: Sequence[int]) -> tuple[list[int], list[int]]:
    rank_position = {int(rank): pos for pos, rank in enumerate(rank_order)}
    ordered = sorted(
        range(len(assignments)),
        key=lambda idx: (
            rank_position.get(assignments[idx].rank, len(rank_position)),
            assignments[idx].prompt_index,
        ))
    inverse = [0 for _ in ordered]
    for new_pos, old_pos in enumerate(ordered):
        inverse[old_pos] = new_pos
    return ordered, inverse


def _role_targets(num_prompts: int, role_plan: RankRolePlan) -> dict[str, int]:
    if num_prompts == 0:
        return {"donor": 0, "wave2": 0, "survivor": 0}
    active_counts = {
        "donor": len(role_plan.donor_ranks),
        "wave2": len(role_plan.wave2_ranks),
        "survivor": len(role_plan.final_survivor_ranks),
    }
    total_ranks = sum(active_counts.values())
    if total_ranks <= 0:
        raise ValueError("rank role plan contains no assignable ranks")

    targets = {
        role: (num_prompts * count) // total_ranks
        for role, count in active_counts.items()
    }
    assigned = sum(targets.values())
    fractional = sorted(
        active_counts,
        key=lambda role: (
            -((num_prompts * active_counts[role]) % total_ranks),
            role,
        ))
    for role in fractional:
        if assigned >= num_prompts:
            break
        if active_counts[role] > 0:
            targets[role] += 1
            assigned += 1
    for role, count in active_counts.items():
        if count == 0 and targets[role] != 0:
            raise RuntimeError(f"internal error: assigned prompts to empty role {role}")
    return targets


def _assign_role(items: Sequence[tuple[int, float]],
                 ranks: Sequence[int],
                 role: str) -> list[PromptAssignment]:
    if not items:
        return []
    if not ranks:
        raise ValueError(f"cannot assign {len(items)} prompts to empty {role} ranks")
    heap = [(0.0, 0, int(rank)) for rank in ranks]
    heapq.heapify(heap)
    result: list[PromptAssignment] = []
    for prompt_index, load in sorted(items, key=lambda item: (-item[1], item[0])):
        rank_load, count, rank = heapq.heappop(heap)
        result.append(PromptAssignment(
            prompt_index=int(prompt_index),
            rank=int(rank),
            role=role,
            predicted_load=float(load),
        ))
        heapq.heappush(heap, (rank_load + float(load), count + 1, rank))
    return result


def rank_seconds_from_loads(
    per_rank_load: Mapping[int, float],
    active_ranks: Sequence[int],
    survivor_ranks: Sequence[int],
) -> float:
    if not active_ranks:
        return 0.0
    active_tail = max(float(per_rank_load.get(rank, 0.0)) for rank in active_ranks)
    reclaimable = 0.0
    survivor_set = set(int(rank) for rank in survivor_ranks)
    for rank in active_ranks:
        if int(rank) not in survivor_set:
            reclaimable += max(0.0, active_tail - float(per_rank_load.get(rank, 0.0)))
    return reclaimable
