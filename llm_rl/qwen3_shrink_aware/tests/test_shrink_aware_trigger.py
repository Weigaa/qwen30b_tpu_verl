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


def test_unfinished_exiting_rank_blocks_release():
    plan = plan_survivor_ranks(world_size=16, shrink_stages=[8, 4])
    decision = decide_staged_shrink(
        enabled=True,
        mode="staged",
        current_active_ranks=list(range(16)),
        unfinished_ranks=plan.donor_ranks[:1] + plan.intermediate_survivor_ranks,
        role_plan=plan,
    )

    assert not decision.should_shrink
    assert decision.fallback_reason.startswith("exiting_ranks_unfinished")


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
