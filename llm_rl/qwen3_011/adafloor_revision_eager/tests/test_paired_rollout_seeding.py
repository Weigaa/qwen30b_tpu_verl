import numpy as np
import pytest

from verl.utils.rollout_seeding import (
    build_rollout_sample_indices,
    derive_request_seed,
    prompt_token_hash,
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
