import importlib.util
from pathlib import Path
import pytest

spec = importlib.util.spec_from_file_location("censored_history_builder", Path(__file__).parents[1] / "tools/build_revision_censored_history.py")
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)


def row(length, reason, cap=4096):
    return {"input": "p", "rollout_generated_tokens": length,
            "rollout_finish_reason": reason, "rollout_sampling_cap": cap}


def test_censor_preserves_prior_and_ignores_mask_padding():
    rows = [dict(row(4096, "length"), response_mask=[1] * 4097), row(400, "stop")]
    record = builder.build(rows, {"p": 8000}, 2)["records"][0]
    assert record["lengths"] == [4096, 400]
    assert record["tail_prediction"] == 8000
    assert record["censored"] == [True, False]


def test_uncensored_uses_paper_ema_weights():
    record = builder.build([row(400, "stop"), row(600, "stop")], {"p": 1000}, 2)["records"][0]
    assert record["tail_prediction"] == 0.3 * 1000 + 0.7 * 600


@pytest.mark.parametrize("rows", [[{"input": "p", "response_mask": [1]}], [row(20, "abort")],
                                  [row(4000, "length")], [row(4097, "stop")]])
def test_inferred_or_invalid_observations_are_rejected(rows):
    with pytest.raises(ValueError): builder.build(rows, responses_per_prompt=1)


def test_context_limit_must_be_effective_not_configured_cap():
    record = builder.build([row(1024, "length", 1024)], responses_per_prompt=1)["records"][0]
    assert record["tail_lower_bound"] == 1024
