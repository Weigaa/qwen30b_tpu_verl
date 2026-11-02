"""Shrink-aware rollout planning helpers."""

from vllm_ascend.shrink_aware.assignment import (
    PromptAssignment,
    PromptAssignmentPlan,
    assign_prompts_to_ranks,
    build_reorder_indices,
)
from vllm_ascend.shrink_aware.planner import (
    RankRolePlan,
    default_package_topology,
    parse_rank_list,
    parse_rank_topology,
    parse_stage_survivor_ranks,
    plan_survivor_ranks,
)
from vllm_ascend.shrink_aware.trigger import (
    StagedShrinkDecision,
    decide_staged_shrink,
)

__all__ = [
    "PromptAssignment",
    "PromptAssignmentPlan",
    "RankRolePlan",
    "StagedShrinkDecision",
    "assign_prompts_to_ranks",
    "build_reorder_indices",
    "decide_staged_shrink",
    "default_package_topology",
    "parse_rank_list",
    "parse_rank_topology",
    "parse_stage_survivor_ranks",
    "plan_survivor_ranks",
]
