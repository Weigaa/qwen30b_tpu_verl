from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

from vllm_ascend.shrink_aware.planner import RankRolePlan


@dataclass(frozen=True)
class StagedShrinkDecision:
    should_shrink: bool
    target_active_ranks: list[int]
    stage_name: str
    fallback_reason: Optional[str] = None


def decide_staged_shrink(
    *,
    enabled: bool,
    mode: str,
    current_active_ranks: Optional[Sequence[int]],
    unfinished_ranks: Sequence[int],
    role_plan: RankRolePlan,
    min_window_seconds: float = 0.0,
    estimated_window_seconds: Optional[float] = None,
    allow_target_size: bool = True,
) -> StagedShrinkDecision:
    if not enabled or (mode or "off").lower() != "staged":
        return StagedShrinkDecision(False, [], "off", "disabled")

    current = sorted(set(int(rank) for rank in (
        current_active_ranks
        if current_active_ranks is not None
        else range(_infer_world_size(role_plan)))))
    unfinished = set(int(rank) for rank in unfinished_ranks)
    intermediate = sorted(role_plan.intermediate_survivor_ranks)
    final = sorted(role_plan.final_survivor_ranks)

    if set(current) == set(range(_infer_world_size(role_plan))):
        exiting = sorted(set(current) - set(intermediate))
        stage_name = "donor"
        target = intermediate
    elif set(current) == set(intermediate):
        exiting = sorted(set(current) - set(final))
        stage_name = "wave2"
        target = final
    else:
        return StagedShrinkDecision(
            False, [], "complete", "current_active_ranks_not_a_planned_stage")

    if not exiting:
        return StagedShrinkDecision(False, [], stage_name, "no_exiting_ranks")
    if not set(target).issubset(set(current)):
        return StagedShrinkDecision(False, [], stage_name, "target_not_subset_of_current")
    blocked = sorted(set(exiting) & unfinished)
    if blocked:
        return StagedShrinkDecision(
            False, [], stage_name, f"exiting_ranks_unfinished:{blocked}")
    if not allow_target_size:
        return StagedShrinkDecision(
            False, [], stage_name, "target_size_rejected_by_runtime_checks")
    if estimated_window_seconds is not None and estimated_window_seconds < min_window_seconds:
        return StagedShrinkDecision(
            False, [], stage_name,
            f"window_too_short:{estimated_window_seconds:.3f}<{min_window_seconds:.3f}")

    return StagedShrinkDecision(True, target, stage_name, None)


def _infer_world_size(role_plan: RankRolePlan) -> int:
    ranks = (
        list(role_plan.donor_ranks) +
        list(role_plan.wave2_ranks) +
        list(role_plan.final_survivor_ranks))
    if not ranks:
        return 0
    return max(ranks) + 1
