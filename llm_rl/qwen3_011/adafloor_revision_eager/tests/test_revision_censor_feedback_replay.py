import importlib.util
from pathlib import Path
import pytest

spec = importlib.util.spec_from_file_location("feedback", Path(__file__).parents[1]/"analysis_eval/replay_revision_censor_feedback.py")
module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)


def row(length, reason, cap):
    return {"input": "p", "rollout_generated_tokens": length, "rollout_finish_reason": reason,
            "rollout_sampling_cap": cap, "response_mask": [1] * (length+1)}


def test_capped_feedback_preserves_prior_but_does_not_recover_unseen_tail():
    result = module.replay([row(4096, "length", 4096)], [row(9000, "stop", 16384)],
                           [{"input": "p", "lengths": [8000]}], 1)
    assert result["old_prediction_below_prior_on_censored_prompts"] == 1
    assert result["corrected_below_prior_on_censored_prompts"] == 0
    assert result["rows"][0]["corrected_prediction"] == 8000
    assert result["rows"][0]["corrected_underestimates_off_observed"]


def test_global_cap_is_not_uncapped_truth():
    result = module.replay([row(4096, "length", 4096)], [row(16384, "length", 16384)],
                           [{"input": "p", "lengths": [8000]}], 1)
    assert not result["rows"][0]["off_observation_is_exact_tail_in_this_sample"]


def test_mask_only_history_cannot_pass():
    with pytest.raises(ValueError): module.replay([{"input": "p", "response_mask": [1]}], [], [], 1)
