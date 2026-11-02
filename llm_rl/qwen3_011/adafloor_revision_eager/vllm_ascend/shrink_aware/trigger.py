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
    target_policy: str = "natural",
) -> StagedShrinkDecision:
    if not enabled or (mode or "off").lower() != "staged":
        return StagedShrinkDecision(False, [], "off", "disabled")

    current = sorted(set(int(rank) for rank in (
        current_active_ranks
        if current_active_ranks is not None
        else range(_infer_world_size(role_plan)))))
    unfinished = set(int(rank) for rank in unfinished_ranks)
    stage_sets = [
        sorted(set(int(rank) for rank in ranks))
        for ranks in getattr(role_plan, "stage_survivor_ranks", None) or [
            role_plan.intermediate_survivor_ranks,
            role_plan.final_survivor_ranks,
        ]
    ]
    current_set = set(current)
    world_size = _infer_world_size(role_plan)
    planned_sizes = [world_size] + [len(stage) for stage in stage_sets]
    current_size = len(current)
    if current_size not in planned_sizes:
        return StagedShrinkDecision(
            False, [], "complete",
            f"current_active_size_not_a_planned_stage:{current_size}")

    target_size: int | None = None
    target_stage: list[int] | None = None
    stage_name = "complete"
    for idx, stage in enumerate(stage_sets):
        stage_size = len(stage)
        if stage_size < current_size:
            target_size = stage_size
            target_stage = stage
            stage_name = _stage_name(idx)
            break
    if target_size is None:
        return StagedShrinkDecision(
            False, [], "complete", "already_at_final_stage")
    unfinished_sorted = sorted(unfinished)
    if len(unfinished_sorted) > target_size:
        return StagedShrinkDecision(
            False, [], stage_name,
            f"unfinished_ranks_exceed_target_size:{len(unfinished_sorted)}>{target_size}")
    policy = (target_policy or "natural").lower().strip()
    if policy in ("planned", "fixed", "plan"):
        if target_stage is None:
            return StagedShrinkDecision(
                False, [], stage_name, "missing_planned_target_stage")
        target = list(target_stage)
        unfinished_outside_stage = sorted(set(unfinished_sorted) - set(target))
        if unfinished_outside_stage:
            return StagedShrinkDecision(
                False, [], stage_name,
                "unfinished_ranks_outside_planned_target:"
                f"{unfinished_outside_stage}")
    elif policy in ("natural", "actual", "rank_aware"):
        if len(unfinished_sorted) < target_size:
            return StagedShrinkDecision(
                False, [], stage_name,
                f"unfinished_ranks_below_target_size:{len(unfinished_sorted)}<{target_size}")
        target = unfinished_sorted
    else:
        return StagedShrinkDecision(
            False, [], stage_name,
            f"unsupported_target_policy:{target_policy}")
    exiting = sorted(current_set - set(target))

    if not exiting:
        return StagedShrinkDecision(False, [], stage_name, "no_exiting_ranks")
    if not set(target).issubset(set(current)):
        return StagedShrinkDecision(False, [], stage_name, "target_not_subset_of_current")
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


def _stage_name(idx: int) -> str:
    if idx == 0:
        return "donor"
    if idx == 1:
        return "wave2"
    return f"stage{idx + 1}"
