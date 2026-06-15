#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from vllm_ascend.shrink_aware import (
    assign_prompts_to_ranks,
    default_package_topology,
    parse_rank_list,
    parse_rank_topology,
    plan_survivor_ranks,
)
from vllm_ascend.shrink_aware.assignment import rank_seconds_from_loads
from vllm_ascend.shrink_aware.trigger import decide_staged_shrink


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Dry-run shrink-aware staged rollout scheduling.")
    parser.add_argument("--world-size", type=int, default=16)
    parser.add_argument("--shrink-stages", default="8,4")
    parser.add_argument("--policy", default="topology_aware",
                        choices=("topology_aware", "contiguous", "manual"))
    parser.add_argument("--package-topology", default="")
    parser.add_argument("--intermediate-survivor-ranks", default="")
    parser.add_argument("--final-survivor-ranks", default="")
    parser.add_argument("--lengths", default="synthetic",
                        help="'synthetic', comma lengths, or a JSON file path")
    parser.add_argument("--num-prompts", type=int, default=64)
    parser.add_argument("--min-shrink-window-seconds", type=float, default=1.0)
    args = parser.parse_args()

    stages = [int(item.strip()) for item in args.shrink_stages.split(",")
              if item.strip()]
    topology = parse_rank_topology(args.package_topology, args.world_size)
    if topology is None:
        topology = default_package_topology(args.world_size)
    role_plan = plan_survivor_ranks(
        world_size=args.world_size,
        shrink_stages=stages,
        package_topology=topology,
        policy=args.policy,
        intermediate_survivor_ranks=parse_rank_list(
            args.intermediate_survivor_ranks),
        final_survivor_ranks=parse_rank_list(args.final_survivor_ranks),
    )
    lengths = _load_lengths(args.lengths, args.num_prompts)
    assignment = assign_prompts_to_ranks(lengths, role_plan)
    active_ranks = list(range(args.world_size))
    reclaimable = rank_seconds_from_loads(
        assignment.per_rank_predicted_load,
        active_ranks,
        role_plan.final_survivor_ranks,
    )
    donor_decision = decide_staged_shrink(
        enabled=True,
        mode="staged",
        current_active_ranks=active_ranks,
        unfinished_ranks=role_plan.intermediate_survivor_ranks,
        role_plan=role_plan,
        min_window_seconds=args.min_shrink_window_seconds,
        estimated_window_seconds=args.min_shrink_window_seconds,
    )
    wave2_decision = decide_staged_shrink(
        enabled=True,
        mode="staged",
        current_active_ranks=role_plan.intermediate_survivor_ranks,
        unfinished_ranks=role_plan.final_survivor_ranks,
        role_plan=role_plan,
        min_window_seconds=args.min_shrink_window_seconds,
        estimated_window_seconds=args.min_shrink_window_seconds,
    )

    report = {
        "world_size": args.world_size,
        "shrink_stages": stages,
        "package_topology": role_plan.package_topology,
        "donor_ranks": role_plan.donor_ranks,
        "wave2_ranks": role_plan.wave2_ranks,
        "intermediate_survivor_ranks": role_plan.intermediate_survivor_ranks,
        "final_survivor_ranks": role_plan.final_survivor_ranks,
        "intermediate_survivor_packages": role_plan.intermediate_survivor_packages,
        "final_survivor_packages": role_plan.final_survivor_packages,
        "package_locality_score": role_plan.package_locality_score,
        "per_rank_prompt_count": assignment.per_rank_counts,
        "per_rank_predicted_load": assignment.per_rank_predicted_load,
        "estimated_completion_wave": {
            str(rank): _role_for_rank(rank, assignment.role_by_rank)
            for rank in active_ranks
        },
        "estimated_reclaimable_rank_seconds": reclaimable,
        "donor_wave_trigger": donor_decision.__dict__,
        "wave2_trigger": wave2_decision.__dict__,
        "fallback_reason": role_plan.fallback_reason,
    }
    print(json.dumps(report, indent=2, sort_keys=True))


def _load_lengths(spec: str, num_prompts: int) -> list[float]:
    if spec == "synthetic":
        return [float((idx % 16) + 1) for idx in range(num_prompts)]
    path = Path(spec)
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            return [float(value) for _, value in sorted(payload.items())]
        return [float(value) for value in payload]
    return [float(item.strip()) for item in spec.split(",") if item.strip()]


def _role_for_rank(rank: int, role_by_rank: dict[int, str]) -> str:
    return role_by_rank.get(rank, "unassigned")


if __name__ == "__main__":
    main()
