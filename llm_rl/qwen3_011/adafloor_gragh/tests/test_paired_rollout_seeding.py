import numpy as np
import pytest

from verl.utils.rollout_seeding import (
    build_rollout_sample_indices,
    derive_request_seed,
    prompt_token_hash,
    validate_fixed_work_restored_order,
    validate_prompt_occurrence_ordinals,
)


def test_interleaved_sample_indices_follow_each_prompt():
    actual = build_rollout_sample_indices(3, 4, interleave=True)
    np.testing.assert_array_equal(
        actual,
        np.array([0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3]),
    )


def test_stacked_sample_indices_follow_repeat_blocks():
    actual = build_rollout_sample_indices(3, 4, interleave=False)
    np.testing.assert_array_equal(
        actual,
        np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3]),
    )


def test_request_seed_is_stable_and_order_independent():
    prompt = [151643, 42, 7, 9]
    first = derive_request_seed(101, prompt, 3)
    second = derive_request_seed(101, list(prompt), 3)
    assert first == second
    assert first[0] == prompt_token_hash(prompt)
    assert 0 <= first[1] < (1 << 31) - 1


def test_request_seed_changes_with_each_identity_component():
    base = derive_request_seed(101, [1, 2, 3], 0)[1]
    assert derive_request_seed(202, [1, 2, 3], 0)[1] != base
    assert derive_request_seed(101, [1, 2, 4], 0)[1] != base
    assert derive_request_seed(101, [1, 2, 3], 1)[1] != base


@pytest.mark.parametrize("prompt_count,responses", [(-1, 2), (2, 0)])
def test_invalid_repeat_shape_is_rejected(prompt_count, responses):
    with pytest.raises(ValueError):
        build_rollout_sample_indices(prompt_count, responses, interleave=True)


def test_repeated_prompt_occurrences_preserve_duplicate_prompt_rows():
    actual = validate_prompt_occurrence_ordinals(
        [15, 15, 274, 274], responses_per_prompt=2
    )
    np.testing.assert_array_equal(actual, np.array([15, 15, 274, 274]))


def test_repeated_prompt_occurrences_reject_ambiguous_source_identity():
    with pytest.raises(ValueError, match="exactly rollout.n"):
        validate_prompt_occurrence_ordinals([0, 0, 0, 0], responses_per_prompt=2)


def test_fixed_work_restored_order_accepts_nonidentity_schedule_restore():
    expected_occurrences = np.array([10, 10, 20, 20])
    expected_samples = np.array([0, 1, 0, 1])
    schedule = np.array([2, 0, 3, 1])
    restore = np.argsort(schedule)

    validate_fixed_work_restored_order(
        expected_occurrences,
        expected_samples,
        expected_occurrences[schedule][restore],
        expected_samples[schedule][restore],
        np.arange(4)[schedule][restore],
    )


def test_fixed_work_restored_order_rejects_unrestored_output():
    with pytest.raises(ValueError, match="was not restored"):
        validate_fixed_work_restored_order(
            [10, 10, 20, 20],
            [0, 1, 0, 1],
            [20, 10, 20, 10],
            [0, 0, 1, 1],
            [2, 0, 3, 1],
        )
