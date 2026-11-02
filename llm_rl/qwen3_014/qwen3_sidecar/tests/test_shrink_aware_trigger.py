from vllm_ascend.shrink_aware import decide_staged_shrink, plan_survivor_ranks


def test_donor_completion_triggers_intermediate_only():
    plan = plan_survivor_ranks(world_size=16, shrink_stages=[8, 4])
    decision = decide_staged_shrink(
        enabled=True,
        mode="staged",
        current_active_ranks=list(range(16)),
        unfinished_ranks=plan.intermediate_survivor_ranks,
        role_plan=plan,
    )

    assert decision.should_shrink
    assert decision.stage_name == "donor"
    assert decision.target_active_ranks == plan.intermediate_survivor_ranks


def test_wave2_completion_triggers_final_only():
    plan = plan_survivor_ranks(world_size=16, shrink_stages=[8, 4])
    decision = decide_staged_shrink(
        enabled=True,
        mode="staged",
        current_active_ranks=plan.intermediate_survivor_ranks,
        unfinished_ranks=plan.final_survivor_ranks,
        role_plan=plan,
    )

    assert decision.should_shrink
    assert decision.stage_name == "wave2"
    assert decision.target_active_ranks == plan.final_survivor_ranks


def test_three_stage_floor2_triggers_halving_steps():
    plan = plan_survivor_ranks(
        world_size=16,
        shrink_stages=[8, 4, 2],
        policy="manual",
        stage_survivor_ranks=[
            [8, 9, 10, 11, 12, 13, 14, 15],
            [12, 13, 14, 15],
            [14, 15],
        ],
    )

    first = decide_staged_shrink(
        enabled=True,
        mode="staged",
        current_active_ranks=list(range(16)),
        unfinished_ranks=plan.stage_survivor_ranks[0],
        role_plan=plan,
    )
    second = decide_staged_shrink(
        enabled=True,
        mode="staged",
        current_active_ranks=plan.stage_survivor_ranks[0],
        unfinished_ranks=plan.stage_survivor_ranks[1],
        role_plan=plan,
    )
    third = decide_staged_shrink(
        enabled=True,
        mode="staged",
        current_active_ranks=plan.stage_survivor_ranks[1],
        unfinished_ranks=plan.stage_survivor_ranks[2],
        role_plan=plan,
    )

    assert first.should_shrink
    assert first.target_active_ranks == [8, 9, 10, 11, 12, 13, 14, 15]
    assert second.should_shrink
    assert second.target_active_ranks == [12, 13, 14, 15]
    assert third.should_shrink
    assert third.target_active_ranks == [14, 15]


def test_rank_aware_trigger_waits_when_unfinished_below_next_stage():
    plan = plan_survivor_ranks(
        world_size=16,
        shrink_stages=[8, 4, 2],
        policy="manual",
        stage_survivor_ranks=[
            [8, 9, 10, 11, 12, 13, 14, 15],
            [12, 13, 14, 15],
            [14, 15],
        ],
    )

    decision = decide_staged_shrink(
        enabled=True,
        mode="staged",
        current_active_ranks=list(range(16)),
        unfinished_ranks=[3, 12, 13, 14, 15],
        role_plan=plan,
    )

    assert not decision.should_shrink
    assert decision.fallback_reason == "unfinished_ranks_below_target_size:5<8"


def test_planned_target_waits_until_unfinished_subset_of_planned_stage():
    plan = plan_survivor_ranks(
        world_size=16,
        shrink_stages=[8, 4, 2],
        policy="manual",
        stage_survivor_ranks=[
            [8, 9, 10, 11, 12, 13, 14, 15],
            [12, 13, 14, 15],
            [14, 15],
        ],
    )

    decision = decide_staged_shrink(
        enabled=True,
        mode="staged",
        current_active_ranks=list(range(16)),
        unfinished_ranks=[3, 12, 13, 14, 15],
        role_plan=plan,
        target_policy="planned",
    )

    assert not decision.should_shrink
    assert (
        decision.fallback_reason
        == "unfinished_ranks_outside_planned_target:[3]"
    )


def test_planned_target_uses_fixed_stage_with_dummy_survivors():
    plan = plan_survivor_ranks(
        world_size=16,
        shrink_stages=[8, 4, 2],
        policy="manual",
        stage_survivor_ranks=[
            [8, 9, 10, 11, 12, 13, 14, 15],
            [12, 13, 14, 15],
            [14, 15],
        ],
    )

    donor = decide_staged_shrink(
        enabled=True,
        mode="staged",
        current_active_ranks=list(range(16)),
        unfinished_ranks=[14, 15],
        role_plan=plan,
        target_policy="planned",
    )
    wave2 = decide_staged_shrink(
        enabled=True,
        mode="staged",
        current_active_ranks=plan.stage_survivor_ranks[0],
        unfinished_ranks=[14],
        role_plan=plan,
        target_policy="planned",
    )

    assert donor.should_shrink
    assert donor.target_active_ranks == [8, 9, 10, 11, 12, 13, 14, 15]
    assert wave2.should_shrink
    assert wave2.target_active_ranks == [12, 13, 14, 15]


def test_rank_aware_trigger_uses_actual_ranks_at_threshold():
    plan = plan_survivor_ranks(
        world_size=16,
        shrink_stages=[8, 4, 2],
        policy="manual",
        stage_survivor_ranks=[
            [8, 9, 10, 11, 12, 13, 14, 15],
            [12, 13, 14, 15],
            [14, 15],
        ],
    )

    decision = decide_staged_shrink(
        enabled=True,
        mode="staged",
        current_active_ranks=list(range(16)),
        unfinished_ranks=[3, 6, 9, 10, 12, 13, 14, 15],
        role_plan=plan,
    )

    assert decision.should_shrink
    assert decision.stage_name == "donor"
    assert decision.target_active_ranks == [3, 6, 9, 10, 12, 13, 14, 15]


def test_rank_aware_trigger_waits_until_unfinished_fits_next_stage():
    plan = plan_survivor_ranks(
        world_size=16,
        shrink_stages=[8, 4, 2],
        policy="manual",
        stage_survivor_ranks=[
            [8, 9, 10, 11, 12, 13, 14, 15],
            [12, 13, 14, 15],
            [14, 15],
        ],
    )

    decision = decide_staged_shrink(
        enabled=True,
        mode="staged",
        current_active_ranks=plan.stage_survivor_ranks[0],
        unfinished_ranks=[8, 9, 10, 11, 12],
        role_plan=plan,
    )

    assert not decision.should_shrink
    assert decision.fallback_reason == "unfinished_ranks_exceed_target_size:5>4"


def test_rank_aware_trigger_waits_when_unfinished_exceeds_next_stage():
    plan = plan_survivor_ranks(world_size=16, shrink_stages=[8, 4])
    decision = decide_staged_shrink(
        enabled=True,
        mode="staged",
        current_active_ranks=list(range(16)),
        unfinished_ranks=plan.donor_ranks[:1] + plan.intermediate_survivor_ranks,
        role_plan=plan,
    )

    assert not decision.should_shrink
    assert decision.fallback_reason == "unfinished_ranks_exceed_target_size:9>8"


def test_disabled_preserves_existing_behavior_marker():
    plan = plan_survivor_ranks(world_size=16, shrink_stages=[8, 4])
    decision = decide_staged_shrink(
        enabled=False,
        mode="staged",
        current_active_ranks=list(range(16)),
        unfinished_ranks=plan.intermediate_survivor_ranks,
        role_plan=plan,
    )

    assert not decision.should_shrink
    assert decision.fallback_reason == "disabled"
