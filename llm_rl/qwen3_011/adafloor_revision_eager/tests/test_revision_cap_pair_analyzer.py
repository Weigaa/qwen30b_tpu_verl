import importlib.util
import json
from pathlib import Path
import pytest

spec = importlib.util.spec_from_file_location("cap_pair_analyzer", Path(__file__).parents[1] / "analysis_eval/analyze_revision_cap_pair.py")
analyzer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analyzer)


def synthetic_arm(root, bad=None):
    logs, artifacts = root / "child/logs", root / "child/rollout_data"
    logs.mkdir(parents=True); artifacts.mkdir()
    text = 'After trainer.fit\nresponse/aborted_ratio:0.0 actor/lr:0.0 timing_s/gen:1.0\n'
    text += 'Revision hard-bound KV admission: ' + json.dumps({"selected_floor": 16}) + '\n'
    text += ''.join(f'Revision actual KV capacity verified: rank={rank} actual_tokens=380800 required_tokens=280576 sampling_cap=16384 selected_floor=16\n' for rank in range(16))
    if bad == "preemption": text += 'Preempting request\n'
    (logs / "run.txt").write_text(text)
    rows = [{"rollout_prompt_hash": str(prompt), "rollout_sample_index": sample,
             "rollout_request_seed": sample, "grpo_group_uid": str(prompt), "score": 1.0,
             "grpo_reward_sum": 1.0, "grpo_outcome_advantage": 0.0,
             "rollout_generated_tokens": 2, "rollout_sampling_cap": 16384,
             "rollout_finish_reason": "stop"} for prompt in range(32) for sample in range(8)]
    if bad == "missing_metadata": rows[0].pop("rollout_generated_tokens")
    if bad == "wrong_advantage": rows[0]["grpo_outcome_advantage"] = 1.0
    if bad == "duplicate_identity": rows[1]["rollout_sample_index"] = 0
    (artifacts / "1.jsonl").write_text(''.join(json.dumps(row)+'\n' for row in rows))


def test_parser_validates_complete_grouped_arm(tmp_path):
    synthetic_arm(tmp_path)
    keyed, summary, paths = analyzer.read_arm(tmp_path)
    assert len(keyed) == 256 and summary["reward_mean"] == 1 and len(paths) == 2


@pytest.mark.parametrize("bad", ["preemption", "missing_metadata", "wrong_advantage", "duplicate_identity"])
def test_incomplete_or_unreproducible_arm_fails(tmp_path, bad):
    synthetic_arm(tmp_path, bad)
    with pytest.raises(ValueError): analyzer.read_arm(tmp_path)
