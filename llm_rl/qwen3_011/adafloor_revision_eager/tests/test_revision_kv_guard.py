import copy

import pytest

from verl.utils.revision_kv_guard import UnsafeBoundedWorkload, admit, rank_upper_bounds


PLAN = {"selected_floor": 4, "shrink_stages": [8, 4],
        "stage_survivor_ranks": [list(range(8, 16)), list(range(12, 16))]}
CAPS = {2: 0, 4: 133120, 8: 262656, 16: 380800}


def test_independent_prompt_response_block_rounding_and_no_double_n():
    assert rank_upper_bounds([1]*32, 128, 16, 128) == [512]*16


def test_known_caps_escalate_without_rewriting_inputs_or_changing_sampling_cap():
    before = copy.deepcopy(PLAN)
    p, r, roles, audit = admit(PLAN, {}, {}, [512]*512, 4096, CAPS)
    assert PLAN == before
    assert p["selected_floor"] == r["selected_floor"] == 8
    assert p["shrink_stages"] == [8]
    assert roles["final_survivor_ranks"] == list(range(8, 16))
    assert audit["disposition"] == "safe_floor_escalation"
    assert audit["hard_response_cap"] == 4096


def test_full16_fallback_and_unsafe_rejection_are_not_preemptions():
    assert admit(PLAN, {}, {}, [512]*512, 8192, CAPS)[3]["disposition"] == "full16_fallback"
    with pytest.raises(UnsafeBoundedWorkload, match="before generation"):
        admit(PLAN, {}, {}, [512]*512, 16384, CAPS)


def test_smaller_response_count_can_be_admitted_without_hidden_truncation():
    assert admit(PLAN, {}, {}, [512]*256, 4096, CAPS)[3]["selected_floor"] == 4


def test_cap_monotonicity_and_invalid_token_units():
    floors = [admit(PLAN, {}, {}, [512]*512, cap, CAPS)[3]["selected_floor"]
              for cap in (1024, 2048, 4096, 8192)]
    assert floors == sorted(floors)
    with pytest.raises(ValueError):
        admit(PLAN, {}, {}, [512]*512, 4096, {2: 0, 4: float('nan'), 8: 2, 16: 3})
