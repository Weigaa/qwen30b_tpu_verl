import pytest
from verl.utils.revision_completion_audit import completion_record


def test_raw_tokens_exclude_padding_without_changing_them():
    ids = [7, 8]
    record = completion_record(ids, "stop", 8, 128, 4096, 20000)
    assert ids == [7, 8] and record["rollout_generated_tokens"] == 2
    assert record["rollout_censored"] is False


def test_sampling_cap_and_context_cap_are_distinguished():
    assert completion_record([1] * 512, "length", None, 100, 512, 1000)["rollout_sampling_cap"] == 512
    record = completion_record([1] * 900, "length", None, 100, 4096, 1000)
    assert record["rollout_sampling_cap"] == 900 and record["rollout_configured_cap"] == 4096


@pytest.mark.parametrize("count,reason", [(513, "stop"), (511, "length"), (5, None), (5, "abort")])
def test_ambiguous_or_failed_completions_fail_closed(count, reason):
    with pytest.raises(ValueError): completion_record([1] * count, reason, None, 100, 512, 1000)
