import importlib.util
import json
import sys
from pathlib import Path

import pytest

analysis_dir = Path(__file__).parents[1] / "analysis_eval"
sys.path.insert(0, str(analysis_dir))
spec = importlib.util.spec_from_file_location("quality_audit", analysis_dir / "analyze_training_quality_curve.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_saved_tokens_distinguish_censoring_and_unexplained_short_response(tmp_path):
    rows = [{"responses": tokens, "response_mask": [1] * len(tokens), "score": 1}
            for tokens in ([1, 151645], [1, 2], [1], [1, 2, 3])]
    path = tmp_path / "rollout.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    result = module.load_step(path, 4, 2, 151645)
    assert result["eos_terminated_responses"] == 1
    assert result["cap_censored_responses"] == 1
    assert result["short_non_eos_responses"] == 1
    assert result["over_cap_responses"] == 1
    assert result["aborted_responses"] == 0
    assert module.load_step(path, 4, 2)["eos_terminated_responses"] is None


def test_noncontiguous_mask_cannot_fake_a_valid_eos(tmp_path):
    path = tmp_path / "rollout.jsonl"
    path.write_text(json.dumps({"responses": [1, 151645, 3], "response_mask": [1, 0, 1]}) + "\n")
    with pytest.raises(ValueError, match="non-contiguous"):
        module.load_step(path, 1, 3, 151645)


def test_inclusive_padding_eos_is_not_generation_overflow(tmp_path):
    path = tmp_path / "rollout.jsonl"
    path.write_text(json.dumps({"responses": [1, 2, 151643, 151643],
                               "response_mask": [1, 1, 1, 0], "score": 1}) + "\n")
    result = module.load_step(path, 1, 2, [151645, 151643], 151643)
    assert result["generated_tokens"] == 3
    assert result["normalized_generated_tokens"] == 2
    assert result["inclusive_padding_eos_count"] == 1
    assert result["cap_censored_responses"] == 1
    assert result["over_cap_responses"] == 0
