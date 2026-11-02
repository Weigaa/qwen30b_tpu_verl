"""Forecast-independent, closed-window KV admission for revision experiments.

This is NOT a hot mid-decode expansion implementation. Admission occurs before
generation and KV resizing. Each response's hard sampling limit must be honored,
no requests may arrive after admission, KV is independently block-rounded, and
the supplied effective caps must cover expert/communication workspace reserves.
Under these conditions EOS only decreases this conservative KV bound.
"""
from __future__ import annotations

import copy
import math


class UnsafeBoundedWorkload(ValueError):
    pass


def rank_upper_bounds(prompt_lengths, hard_response_cap, world_size=16, block_size=128):
    if (world_size <= 0 or block_size <= 0 or hard_response_cap <= 0
            or len(prompt_lengths) == 0 or len(prompt_lengths) % world_size):
        raise ValueError("invalid bounded-window dimensions")
    if any(not isinstance(x, int) or x < 0 for x in prompt_lengths):
        raise ValueError("prompt lengths must be nonnegative integer token counts")
    width = len(prompt_lengths) // world_size
    # Input already contains one row PER RESPONSE, after interleaved repeat
    # and the actual prompt-to-rank reorder. Do not multiply rollout.n twice.
    return [sum(((p + hard_response_cap + block_size - 1) // block_size) * block_size
                for p in prompt_lengths[r*width:(r+1)*width]) for r in range(world_size)]


def admit(plan, runtime, roles, prompt_lengths, hard_response_cap, caps,
          world_size=16, block_size=128):
    if set(caps) != {2, 4, 8, 16} or any(not math.isfinite(v) or v < 0 for v in caps.values()):
        raise ValueError("effective caps must be finite nonnegative TOKENS for 2/4/8/16")
    current = int(plan["selected_floor"])
    if current not in caps:
        raise ValueError("unsupported nominal floor")
    upper = rank_upper_bounds(prompt_lengths, hard_response_cap, world_size, block_size)
    selected = next((f for f in (2, 4, 8, 16)
                     if f >= current and caps[f] > 0 and max(upper) <= caps[f]), None)
    if selected is None:
        raise UnsafeBoundedWorkload(
            f"unsafe_workload_rejection before generation: worst_rank_tokens={max(upper)} "
            f"full16_cap_tokens={caps[16]} hard_response_cap={hard_response_cap}; "
            "defer/rebatch explicitly, never silently lower response caps")
    new_plan, new_runtime, new_roles = map(copy.deepcopy, (plan, runtime, roles))
    stages = plan.get("shrink_stages", runtime.get("shrink_stages", []))
    survivors = plan.get("stage_survivor_ranks", roles.get("stage_survivor_ranks", []))
    if len(stages) != len(survivors):
        raise ValueError("ambiguous original survivor schedule")
    retained = [(int(f), ranks) for f, ranks in zip(stages, survivors) if selected <= int(f) < world_size]
    if selected != world_size and not any(f == selected for f, _ in retained):
        raise ValueError("cannot invent an unprepared floor/topology")
    new_stages = [f for f, _ in retained]
    new_survivors = [r for _, r in retained]
    final = new_survivors[-1] if retained else list(range(world_size))
    intermediate = new_survivors[0] if retained else list(range(world_size))
    for target in (new_plan, new_runtime, new_roles):
        target.update(selected_floor=selected, floor=selected, kv_cap=float(caps[selected]),
                      shrink_stages=new_stages, stage_survivor_ranks=new_survivors,
                      final_survivor_ranks=final, intermediate_survivor_ranks=intermediate)
    evidence = {"nominal_floor": current, "selected_floor": selected,
                "hard_response_cap": hard_response_cap, "block_size_tokens": block_size,
                "rank_upper_tokens": upper, "cap_units": "tokens",
                "admission_headroom_tokens": caps[selected]-max(upper),
                "disposition": "safe_current_floor" if selected == current else
                "full16_fallback" if selected == 16 else "safe_floor_escalation",
                "hot_expansion": False, "new_per_token_consensus": False}
    return new_plan, new_runtime, new_roles, evidence
