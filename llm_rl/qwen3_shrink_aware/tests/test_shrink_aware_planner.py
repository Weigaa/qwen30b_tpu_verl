import pytest

from vllm_ascend.shrink_aware import plan_survivor_ranks


def test_topology_aware_survivor_selection_16_ranks():
    topology = [[idx, idx + 1] for idx in range(0, 16, 2)]
    plan = plan_survivor_ranks(
        world_size=16,
        shrink_stages=[8, 4],
        package_topology=topology,
        policy="topology_aware",
    )

    assert len(plan.intermediate_survivor_ranks) == 8
    assert len(plan.final_survivor_ranks) == 4
    assert set(plan.final_survivor_ranks).issubset(
        set(plan.intermediate_survivor_ranks))
    assert len(plan.intermediate_survivor_packages) == 4
    assert len(plan.final_survivor_packages) == 2
    assert all(len(package) == 2 for package in plan.intermediate_survivor_packages)
    assert all(len(package) == 2 for package in plan.final_survivor_packages)
    assert plan.package_locality_score == 1.0


def test_manual_survivor_ranks_valid():
    plan = plan_survivor_ranks(
        world_size=16,
        shrink_stages=[8, 4],
        policy="manual",
        intermediate_survivor_ranks=[0, 1, 2, 3, 8, 9, 10, 11],
        final_survivor_ranks=[8, 9, 10, 11],
    )

    assert plan.intermediate_survivor_ranks == [0, 1, 2, 3, 8, 9, 10, 11]
    assert plan.final_survivor_ranks == [8, 9, 10, 11]
    assert plan.donor_ranks == [4, 5, 6, 7, 12, 13, 14, 15]


@pytest.mark.parametrize(
    "intermediate,final",
    [
        ([0, 1, 2], [0, 1, 2, 3]),
        ([0, 1, 2, 3, 4, 5, 6, 6], [0, 1, 2, 3]),
        ([0, 1, 2, 3, 4, 5, 6, 99], [0, 1, 2, 3]),
        ([0, 1, 2, 3, 4, 5, 6, 7], [8, 9, 10, 11]),
    ],
)
def test_manual_survivor_ranks_invalid(intermediate, final):
    with pytest.raises(ValueError):
        plan_survivor_ranks(
            world_size=16,
            shrink_stages=[8, 4],
            policy="manual",
            intermediate_survivor_ranks=intermediate,
            final_survivor_ranks=final,
        )
