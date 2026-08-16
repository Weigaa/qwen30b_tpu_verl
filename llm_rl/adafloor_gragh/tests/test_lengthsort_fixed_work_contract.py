from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).parents[1]
RUNNER = ROOT / "run_baseline_lengthsort_epoch1_2.sh"
EXECUTION_SHA256 = "a" * 64


def _trace(path: Path, steps: int) -> tuple[Path, str]:
    records: list[dict[str, object]] = []
    sources: list[dict[str, object]] = []
    plan: list[dict[str, object]] = []
    for step in range(1, steps + 1):
        occurrences = list(range((step - 1) * 64, step * 64))
        dataset_indices = list(range((step - 1) * 64, step * 64))
        rank_to_source_idx = {
            str(rank): occurrences[rank * 4 : (rank + 1) * 4]
            for rank in range(16)
        }
        rank_to_dataset_item_idx = {
            str(rank): dataset_indices[rank * 4 : (rank + 1) * 4]
            for rank in range(16)
        }
        plan.append(
            {
                "step": step,
                "tail_guard_response_cap": 16384,
                "rank_to_source_idx": rank_to_source_idx,
                "rank_to_dataset_item_idx": rank_to_dataset_item_idx,
            }
        )
        for prompt_slot, occurrence in enumerate(occurrences):
            for sample_index in range(16):
                ordinal = prompt_slot * 16 + sample_index
                records.append(
                    {
                        "step": step,
                        "row_ordinal": ordinal,
                        "prompt_occurrence_ordinal": occurrence,
                        "rollout_prompt_hash": f"prompt-{occurrence}",
                        "rollout_sample_index": sample_index,
                        "rollout_request_seed": occurrence * 1000 + sample_index,
                        "source_decoded_response_length": 1,
                        "target_response_length": 1,
                    }
                )
        sources.append(
            {
                "step": step,
                "path": f"rollout_data/{step}.jsonl",
                "sha256": f"{step:064x}",
                "row_count": 1024,
                "source_generated_tokens": 1024,
                "target_generated_tokens": 1024,
                "prompt_occurrence_count": 64,
            }
        )
    plan_path = path.with_suffix(".plan.json")
    plan_path.write_text(json.dumps(plan) + "\n", encoding="utf-8")
    payload = {
        "schema_version": 3,
        "format": "deepseek_batch64_fixed_work_replay",
        "source_run_dir": "/source",
        "adafloor_plan_source": {
            "path": str(plan_path.resolve()),
            "sha256": hashlib.sha256(plan_path.read_bytes()).hexdigest(),
        },
        "row_ordinal_base": 0,
        "expected_rows_per_step": 1024,
        "max_response_length": 16384,
        "identity_fields": [
            "rollout_prompt_hash",
            "rollout_sample_index",
            "rollout_request_seed",
        ],
        "lookup_key_fields": [
            "prompt_occurrence_ordinal",
            "rollout_sample_index",
        ],
        "row_ordinal_semantics": "source_jsonl_physical_provenance_only",
        "prompt_occurrence_ordinal_source": (
            "adafloor_plan_source.rank_to_source_idx"
        ),
        "source_length_field": "source_decoded_response_length",
        "target_length_field": "target_response_length",
        "step_count": steps,
        "record_count": steps * 1024,
        "source_generated_tokens": steps * 1024,
        "target_generated_tokens": steps * 1024,
        "step_caps": {str(step): 16384 for step in range(1, steps + 1)},
        "step_prompt_occurrences": {
            str(step): list(range((step - 1) * 64, step * 64))
            for step in range(1, steps + 1)
        },
        "source_files": sources,
        "records": records,
    }
    raw = (json.dumps(payload, separators=(",", ":")) + "\n").encode("ascii")
    path.write_bytes(raw)
    return path, hashlib.sha256(raw).hexdigest()


def _fake_tree(tmp_path: Path) -> Path:
    tree = tmp_path / "driver"
    (tree / "tools").mkdir(parents=True)
    shutil.copy2(RUNNER, tree / RUNNER.name)
    (tree / "tools" / "hash_deepseek_execution_code.py").write_text(
        f"print('{EXECUTION_SHA256}')\n", encoding="utf-8"
    )
    (tree / "tools" / "build_mode1_length_sorted_e2e_plan.py").write_text(
        """#!/usr/bin/env python3
import json
import sys
from pathlib import Path

def value(name):
    return sys.argv[sys.argv.index(name) + 1]

steps = int(value('--steps'))
batch = int(value('--batch-size'))
Path(value('--output-train')).touch()
Path(value('--output-plan')).write_text('[]\\n', encoding='utf-8')
Path(value('--output-oracle')).write_text('{}\\n', encoding='utf-8')
summary = [
    {
        'step': step,
        'rank_matching_policy': 'contiguous',
        'selected_floor': 16,
        'tail_guard_response_cap': 16384,
    }
    for step in range(1, steps + 1)
]
Path(value('--output-summary')).write_text(json.dumps(summary) + '\\n', encoding='utf-8')
Path(__file__).with_name('planner_args.txt').write_text(
    f'steps={steps}\\nbatch_size={batch}\\n', encoding='utf-8'
)
""",
        encoding="utf-8",
    )
    child = tree / "run_mode0_no_shrink_baseline.sh"
    child.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
steps=
for arg in "$@"; do
    case "$arg" in trainer.total_training_steps=*) steps=${arg#*=} ;; esac
done
[[ "$steps" =~ ^[1-9][0-9]*$ ]]
mkdir -p "$RECORD_DIR/rollout_data"
for (( step = 1; step <= steps; step++ )); do
    : > "$RECORD_DIR/rollout_data/$step.jsonl"
done
mkdir -p "$RECORD_DIR/checkpoints/$CHECKPOINT_MODEL_DIR_NAME/global_step_$steps/actor/dist_ckpt"
printf '%s\\n' \
    "train_batch_size=$TRAIN_BATCH_SIZE" \
    "max_num_seqs=$ROLLOUT_MAX_NUM_SEQS" \
    "save_freq=$TRAINER_SAVE_FREQ" \
    "training_steps=$steps" \
    > "$(dirname "$0")/child_contract.txt"
""",
        encoding="utf-8",
    )
    child.chmod(0o755)
    return tree


def _run_contract(tmp_path: Path, phase: str, steps: int) -> tuple[subprocess.CompletedProcess[str], Path]:
    tree = _fake_tree(tmp_path)
    trace, digest = _trace(tmp_path / f"{phase}.json", steps)
    train = tmp_path / "train.parquet"
    test = tmp_path / "test.parquet"
    train.touch()
    test.touch()
    model = tmp_path / "model"
    distcp = tmp_path / "distcp"
    model.mkdir()
    distcp.mkdir()
    history = tmp_path / "history"
    (history / "rollout_data").mkdir(parents=True)
    output = tmp_path / "output"
    env = {
        **os.environ,
        "PYTHONPATH": str(ROOT),
        "ADAFLOOR_LENGTHSORT_DRIVER_SNAPSHOT_ACTIVE": "1",
        "TRAIN_FILE_ORIG": str(train),
        "TEST_FILE": str(test),
        "MODEL_PATH": str(model),
        "DISTCP_PATH": str(distcp),
        "DYNAMIC_INITIAL_BASELINE_DIR": str(history),
        "DYNAMIC_OUTPUT_ROOT": str(output),
        "DYNAMIC_RUN_NAME": f"fixed_{phase}",
        "DYNAMIC_PLAN_STEPS": str(steps),
        "DYNAMIC_TRAIN_STEPS": str(steps),
        "DYNAMIC_START_EPOCH": "1",
        "DYNAMIC_TOTAL_EPOCHS": "2",
        "TRAIN_BATCH_SIZE": "64",
        "ROLLOUT_MAX_NUM_SEQS": "64",
        "ROLLOUT_N": "16",
        "CHECKPOINT_MODEL_DIR_NAME": "model",
        "BASELINE_ENABLE_TAIL_GUARD": "0",
        "DEEPSEEK_FIXED_WORK_LAUNCH_PROTOCOL": "deepseek_batch64_fixed_work_replay_v3",
        "DEEPSEEK_FIXED_WORK_PHASE": phase,
        "DEEPSEEK_FIXED_WORK_EXPECTED_STEPS": str(steps),
        "DEEPSEEK_FIXED_WORK_EXECUTION_CODE_SHA256": EXECUTION_SHA256,
        "DEEPSEEK_KV_CAP_TRAIN_BATCH_SIZE": "64",
        "DEEPSEEK_KV_CAP_ROLLOUT_N": "16",
        "DEEPSEEK_KV_CAP_MAX_NUM_SEQS": "64",
        "DEEPSEEK_KV_CAP_EXPECTED_RESPONSES_PER_STEP": "1024",
        "VERL_FIXED_WORK_REPLAY_TRACE": str(trace),
        "VERL_FIXED_WORK_REPLAY_SHA256": digest,
        "VERL_FIXED_WORK_REPLAY_REQUIRE_PLAN_CAP": "0",
    }
    result = subprocess.run(
        ["bash", str(tree / RUNNER.name)],
        cwd=tree,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    return result, tree


@pytest.mark.parametrize(("phase", "steps"), [("gate", 1), ("epoch", 5)])
def test_non_dry_fixed_work_launch_uses_exact_batch64_contract(
    tmp_path: Path, phase: str, steps: int
) -> None:
    result, tree = _run_contract(tmp_path, phase, steps)
    assert result.returncode == 0, result.stderr
    assert (tree / "tools" / "planner_args.txt").read_text(encoding="utf-8") == (
        f"steps={steps}\nbatch_size=64\n"
    )
    assert (tree / "child_contract.txt").read_text(encoding="utf-8") == (
        "train_batch_size=64\n"
        "max_num_seqs=64\n"
        f"save_freq={steps}\n"
        f"training_steps={steps}\n"
    )


def test_one_step_gate_is_rejected_without_complete_fixed_work_contract(
    tmp_path: Path,
) -> None:
    result, _tree = _run_contract(tmp_path, "gate", 1)
    assert result.returncode == 0
    # A missing protocol must fail before planning or launch.
    tree = _fake_tree(tmp_path / "missing")
    trace, digest = _trace(tmp_path / "missing-trace.json", 1)
    env = {
        **os.environ,
        "PYTHONPATH": str(ROOT),
        "ADAFLOOR_LENGTHSORT_DRIVER_SNAPSHOT_ACTIVE": "1",
        "DYNAMIC_PLAN_STEPS": "1",
        "DYNAMIC_TRAIN_STEPS": "1",
        "TRAIN_BATCH_SIZE": "64",
        "ROLLOUT_MAX_NUM_SEQS": "64",
        "ROLLOUT_N": "16",
        "VERL_FIXED_WORK_REPLAY_TRACE": str(trace),
        "VERL_FIXED_WORK_REPLAY_SHA256": digest,
    }
    rejected = subprocess.run(
        ["bash", str(tree / RUNNER.name)],
        cwd=tree,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert rejected.returncode == 2
    assert "invalid DeepSeek fixed-work launch protocol" in rejected.stderr
