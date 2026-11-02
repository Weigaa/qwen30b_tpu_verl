import subprocess
from pathlib import Path

import pytest


@pytest.mark.parametrize("prefix,exit_code,oom,expected", [
    ("", 0, False, 0),
    ("path string is NULLpath string is NULL", 0, False, 0),
    ("", 1, False, 1),
    ("", 0, True, 1),
])
def test_native_v014_completion_record(tmp_path, prefix, exit_code, oom, expected):
    root = Path(__file__).parents[1]
    source = (root / "run_paper_fair_epoch1_2_from_common_epoch0.sh").read_text()
    functions = source[source.index("is_planned_variant() {"):
                       source.index("list_matching_processes() {")]
    epoch = tmp_path / "epoch_001_mode1_natural"
    for folder in ("logs", "rollout_data", "rollout_length"):
        (epoch / folder).mkdir(parents=True)
    (epoch / "rollout_data/1.jsonl").write_text('{}\n' * 512)
    (epoch / "rollout_length/length_1.txt").write_text('123\n' * 512)
    ckpt = epoch / "checkpoints/model"
    shards = ckpt / "global_step_1/actor/dist_ckpt"
    shards.mkdir(parents=True)
    (shards / "fake.distcp").touch()
    (ckpt / "latest_checkpointed_iteration.txt").write_text('1\n')
    (epoch / "logs/test.txt").write_text(
        "rollout_output_time_s: 123\n"
        "step:1 - rollout_output_time_s:123 - response/aborted_ratio:0.0\n"
        "Training Progress: 100%\n"
        + ("NPU out of memory\n" if oom else "")
        + f"{prefix}[run] end_time=2026-10-14T14:07:27+0800 exit_code={exit_code}\n")
    invocation = '''
variant=adafloor_n_f2
FAIR_TRAIN_STEPS=1
FAIR_EXPECTED_RESPONSES_PER_STEP=512
CHECKPOINT_MODEL_DIR_NAME=model
FAIR_EXPECTED_DISTCP_SHARDS=auto
FAIR_FREEZE_ACTOR=0
VERL_PAIRED_REQUEST_SAMPLING_SEEDS=0
validate_completed_epoch "$1" 001
'''
    result = subprocess.run(["bash", "-c", functions + invocation, "test", str(tmp_path)],
                            capture_output=True, text=True)
    assert result.returncode == expected, result.stderr
