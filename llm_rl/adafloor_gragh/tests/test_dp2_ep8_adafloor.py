from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path


TOOLS_DIR = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS_DIR))

from build_dp2_ep8_adafloor_plan import (  # noqa: E402
    _partition_workers,
    _role_for_floor,
    _worker_payload,
    solve_ep8,
)
from build_mode1_length_sorted_e2e_plan import (  # noqa: E402
    RELEASE_AREA_UNIT,
    _stats_from_lengths,
)
from summarize_dp2_ep8_run import (  # noqa: E402
    _rank_seconds_from_runtime,
    _validate_runtime_floors,
)
from verl.experimental.dataset.shrink_aware_assignment import (  # noqa: E402
    select_shrink_aware_worker_plan,
)


def _prompt(index: int, length: float):
    return replace(
        _stats_from_lengths(tuple([length] * 16)),
        source_idx=index,
    )


def test_ep8_solver_assigns_two_prompts_per_rank_and_respects_kv() -> None:
    prompts = [_prompt(index, float(index + 1)) for index in range(16)]

    plan = solve_ep8(
        prompts,
        {2: 1_000_000.0, 4: 1_000_000.0, 8: 1_000_000.0},
        active_peak_safety_factor=1.0,
        max_response_len=16384.0,
    )

    assert plan.selected_floor in (2, 4, 8)
    assert sorted(plan.rank_to_source_idx) == list(range(8))
    assert all(len(source_ids) == 2
               for source_ids in plan.rank_to_source_idx.values())
    assert max(plan.rank_adjusted_peak_loads.values()) <= plan.kv_cap

    row_map = {
        source_idx: source_idx
        for source_ids in plan.rank_to_source_idx.values()
        for source_idx in source_ids
    }
    payload = _worker_payload(0, plan, row_map, 4096, 1.2, True)
    assert payload["release_area_unit"] == RELEASE_AREA_UNIT


def test_length_sorted_worker_partition_balances_each_global_step() -> None:
    prompts = [_prompt(index, float(index + 1)) for index in range(32)]

    workers = _partition_workers(prompts, "length_sorted")

    assert [len(worker) for worker in workers] == [16, 16]
    assert max(item.load for item in workers[0]) == 31.0
    assert max(item.load for item in workers[1]) == 32.0


def test_floor2_role_plan_is_ep8_local() -> None:
    role_plan = _role_for_floor(2)

    assert role_plan["stage_survivor_ranks"] == [[4, 5, 6, 7], [6, 7]]
    assert role_plan["donor_ranks"] == [0, 1, 2, 3]
    assert role_plan["wave2_ranks"] == [4, 5]


def test_worker_plan_overlay_selects_second_external_dp_group() -> None:
    meta = {
        "shrink_aware_runtime": {"mode": "staged"},
        "shrink_aware_kv_plan": {},
        "shrink_aware_worker_plans": [
            {
                "worker_id": 0,
                "global_ranks": list(range(8)),
                "selected_floor": 4,
                "kv_cap": 280576,
                "shrink_stages": [4],
                "role_plan": _role_for_floor(4),
            },
            {
                "worker_id": 1,
                "global_ranks": list(range(8, 16)),
                "selected_floor": 2,
                "kv_cap": 131072,
                "shrink_stages": [4, 2],
                "role_plan": _role_for_floor(2),
            },
        ],
    }

    select_shrink_aware_worker_plan(meta, 13)

    assert meta["shrink_aware_runtime"]["selected_floor"] == 2
    assert meta["shrink_aware_runtime"]["kv_cap"] == 131072
    assert meta["shrink_aware_worker_context"] == {
        "worker_id": 1,
        "global_ranks": list(range(8, 16)),
        "global_rank": 13,
        "local_rank": 5,
    }


def test_runtime_accounting_separates_worker_and_intra_worker_area() -> None:
    shrink_events = []
    for rank in range(8):
        shrink_events.append((4.0, rank, (4, 5, 6, 7)))
    for rank in range(8, 16):
        shrink_events.append((6.0, rank, (12, 13, 14, 15)))
    call_done_events = [
        *((8.0, rank) for rank in range(8)),
        *((10.0, rank) for rank in range(8, 16)),
    ]

    result = _rank_seconds_from_runtime(
        rollout_end=12.0,
        rollout_duration=10.0,
        previous_rollout_end=-float("inf"),
        shrink_events=shrink_events,
        call_done_events=call_done_events,
    )

    assert result["tlt_like_worker_level_rank_seconds"] == 16.0
    assert result["adafloor_intra_worker_rank_seconds"] == 32.0
    assert result["total_hierarchical_rank_seconds"] == 48.0
    assert result["post_worker_control_rank_seconds"] == 32.0


def test_runtime_accounting_uses_cleanup_fallback() -> None:
    result = _rank_seconds_from_runtime(
        rollout_end=20.0,
        rollout_duration=20.0,
        previous_rollout_end=-float("inf"),
        shrink_events=[],
        call_done_events=[],
        cleanup_events=[
            *((10.0, rank) for rank in range(8)),
            *((20.0, rank) for rank in range(8, 16)),
        ],
    )

    assert result["tlt_like_worker_level_rank_seconds"] == 80.0
    assert result["post_worker_control_rank_seconds"] == 0.0
    assert result["worker_finish_offset_seconds"] == [10.0, 20.0]


def test_runtime_floor_validation_matches_planned_ladders() -> None:
    workers = [{"selected_floor": 2}, {"selected_floor": 4}]
    transitions = [
        [
            {"target_global_ranks": [4, 5, 6, 7]},
            {"target_global_ranks": [6, 7]},
        ],
        [{"target_global_ranks": [12, 13, 14, 15]}],
    ]

    _validate_runtime_floors(workers, transitions, step_index=1)


def test_runtime_floor_validation_rejects_missing_transition() -> None:
    workers = [{"selected_floor": 4}]

    try:
        _validate_runtime_floors(workers, [[]], step_index=3)
    except RuntimeError as exc:
        assert "planned floor 4" in str(exc)
        assert "observed []" in str(exc)
    else:
        raise AssertionError("missing floor4 transition was accepted")
