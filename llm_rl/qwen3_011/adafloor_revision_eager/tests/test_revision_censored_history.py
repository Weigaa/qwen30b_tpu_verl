import pytest

from verl.utils.revision_censored_history import update_tail


def test_repeated_censoring_cannot_artificially_lower_prior_tail():
    old = 12000.0
    for _ in range(4):
        record = update_tail(old, [2000,4096], [False,True], [4096,4096])
        assert record["tail_prediction"] >= old
        old = record["tail_prediction"]
    assert old == 12000


def test_uncensored_evidence_can_adapt_downward():
    assert update_tail(12000,[2000,3000],[False,False],[4096,4096])["tail_prediction"] == 5700


def test_censoring_greater_than_prior_moves_estimate_up():
    assert update_tail(2000,[4096],[True],[4096])["tail_prediction"] == 4096


def test_missing_reason_and_mask_padding_are_not_silently_accepted():
    with pytest.raises(ValueError):
        update_tail(3000,[4097],[True],[4096])
    with pytest.raises(ValueError):
        update_tail(3000,[4096],[None],[4096])
