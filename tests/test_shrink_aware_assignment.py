from vllm_ascend.shrink_aware import assign_prompts_to_ranks, plan_survivor_ranks


def test_prompt_assignment_roles_and_uniqueness():
    role_plan = plan_survivor_ranks(world_size=16, shrink_stages=[8, 4])
    lengths = list(range(1, 65))

    assignment = assign_prompts_to_ranks(lengths, role_plan)

    assert sorted(item.prompt_index for item in assignment.assignments) == list(range(64))

    donor_loads = [
        item.predicted_load for item in assignment.assignments
        if item.role == "donor"
    ]
    wave2_loads = [
        item.predicted_load for item in assignment.assignments
        if item.role == "wave2"
    ]
    survivor_loads = [
        item.predicted_load for item in assignment.assignments
        if item.role == "survivor"
    ]

    assert max(donor_loads) < min(wave2_loads)
    assert max(wave2_loads) < min(survivor_loads)
    assert {
        item.rank for item in assignment.assignments if item.role == "donor"
    }.issubset(set(role_plan.donor_ranks))
    assert {
        item.rank for item in assignment.assignments if item.role == "wave2"
    }.issubset(set(role_plan.wave2_ranks))
    assert {
        item.rank for item in assignment.assignments if item.role == "survivor"
    }.issubset(set(role_plan.final_survivor_ranks))


def test_prompt_assignment_balances_within_role():
    role_plan = plan_survivor_ranks(world_size=16, shrink_stages=[8, 4])
    lengths = [10.0] * 64

    assignment = assign_prompts_to_ranks(lengths, role_plan)

    donor_counts = [
        assignment.per_rank_counts[rank] for rank in role_plan.donor_ranks
    ]
    wave2_counts = [
        assignment.per_rank_counts[rank] for rank in role_plan.wave2_ranks
    ]
    survivor_counts = [
        assignment.per_rank_counts[rank]
        for rank in role_plan.final_survivor_ranks
    ]
    assert max(donor_counts) - min(donor_counts) <= 1
    assert max(wave2_counts) - min(wave2_counts) <= 1
    assert max(survivor_counts) - min(survivor_counts) <= 1
