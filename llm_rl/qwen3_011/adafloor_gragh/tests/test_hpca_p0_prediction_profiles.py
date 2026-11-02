import math

import pytest

from tools.analyze_hpca_prediction_profiles import evaluate, inflate, peak


PROFILES = {0: [(5, 20), (5, 10)], 1: [(2, 15), (2, 5)]}
STEP = {"step": 1, "selected_floor": 2, "shrink_stages": [2],
        "stage_survivor_ranks": [[0]], "release_area": 5}


def test_peak_includes_prompt_and_inclusive_completion():
    assert peak(PROFILES[0]) == 30
    assert peak([(5, 0), (5, 0)]) == 10


def test_three_scenarios_are_distinct_and_local_ties_all_inflate():
    assert inflate(PROFILES, 1.4, "local_tail")[1] == [(2, 21), (2, 5)]
    assert inflate(PROFILES, 1.4, "hottest_rank")[1] == PROFILES[1]
    assert inflate(PROFILES, 1.4, "global")[1] == [(2, 21), (2, 7)]
    assert inflate({0: [(5, 20), (5, 20)]}, 1.4, "local_tail")[0] == [(5, 28), (5, 28)]


def test_floor_monotonic_and_rejection_not_preemption():
    caps = {2: 30, 4: 32, 8: 33, 16: 35}
    rows = [evaluate(STEP, PROFILES, caps, x, "global", 1, 100)
            for x in (1, 1.1, 1.16, 1.25, 1.4, 2)]
    assert rows[0]["selected_floor"] == 2
    assert rows[1]["selected_floor"] == 4
    assert rows[2]["selected_floor"] == 16
    assert rows[2]["disposition"] == "full16_fallback"
    assert rows[-1]["disposition"] == "unsafe_workload_rejection"
    assert rows[-1]["predicted_release_rank_token_proxy"] == 0
    floors = [row["selected_floor"] or math.inf for row in rows]
    assert floors == sorted(floors)
    assert "preemption" not in rows[-1]


def test_bound_does_not_hide_unsafe_inflation():
    row = evaluate(STEP, PROFILES, {2: 30, 4: 31, 8: 32, 16: 35}, 10, "global", 1.16, 20)
    assert row["response_bound_exceedance_count"] == 4
    assert row["disposition"] == "unsafe_workload_rejection"
    with pytest.raises(ValueError):
        inflate(PROFILES, float("nan"), "global")
