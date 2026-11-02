import importlib.util
from pathlib import Path


spec = importlib.util.spec_from_file_location(
    "summary", Path(__file__).parents[1] / "tools/summarize_native_sleep_run.py")
summary = importlib.util.module_from_spec(spec)
spec.loader.exec_module(summary)


def log_file(tmp_path, text):
    path = tmp_path / "epoch_001_vanilla" / "logs" / "run.txt"
    path.parent.mkdir(parents=True)
    path.write_text(text)
    return path


def test_capture_and_rollout_do_not_prove_actor_completion(tmp_path):
    result = summary.summarize_log(log_file(tmp_path,
        "Graph capturing finished\nrollouts speed tokens/s: 123.4\n"))
    assert result["rollout_tokens_per_second"] == [123.4]
    assert result["completed_step_metrics"] == []
    assert result["trainer_fit_returned"] is False
    assert result["exit_code"] is None


def test_read_every_epoch_log_not_latest_length_file(tmp_path):
    result = summary.summarize_log(log_file(tmp_path,
        "rollout_forward_begin dp_rank=0 iteration=256 local_tokens=2 "
        "global_prefill=False graph_mode=FULL\n"
        "step:1 - timing_s/update_actor:12\nAfter trainer.fit\n"
        "[run] end_time=2026-09-14T00:00:00 exit_code=0\n"))
    assert result["last_rank0_forward"] == {
        "iteration": 256, "global_prefill": False, "graph_mode": "FULL"}
    assert result["sampled_full_ranks"] == [0]
    assert result["completed_step_metrics"] == [1]
    assert result["trainer_fit_returned"] is True
    assert result["exit_code"] == 0


def test_failed_update_preserves_rollout_evidence(tmp_path):
    result = summary.summarize_log(log_file(tmp_path,
        "rollout_output_time_s: 12.5\nRuntimeError: NPU out of memory\n"
        "[run] end_time=2026-09-14T00:00:00 exit_code=1\n"))
    assert result["rollout_seconds"] == [12.5]
    assert result["exit_code"] == 1
    assert len(result["errors"]) == 1
    assert result["completed_step_metrics"] == []
