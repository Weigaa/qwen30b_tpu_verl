from __future__ import annotations

import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "run_qwen3_vanilla_epoch0_piecewise_tq1.sh"


def test_piecewise_epoch0_dry_run_contract(tmp_path: Path) -> None:
    output = tmp_path / "piecewise"
    env = os.environ.copy()
    env["PIECEWISE_EPOCH0_ROOT"] = str(output)
    result = subprocess.run(
        [str(RUNNER), "dry-run"],
        cwd=ROOT,
        env=env,
        check=True,
        text=True,
        capture_output=True,
    )
    assert "dry run only; Ray and NPU were not started" in result.stdout
    assert "graph=PIECEWISE task_queue=1" in result.stdout
    assert "attention=eager_split_boundary moe=aclgraph" in result.stdout

    protocol = dict(
        line.split("=", 1)
        for line in (output / "protocol.env").read_text().splitlines()
    )
    assert protocol["seed"] == "0"
    assert protocol["actor_megatron_seed"] == "42"
    assert protocol["actor_lr"] == "1e-6"
    assert protocol["steps"] == "5"
    assert protocol["prompts_per_step"] == "32"
    assert protocol["rollout_n"] == "16"
    assert protocol["responses_per_step"] == "512"
    assert protocol["max_response_length"] == "16384"
    assert protocol["max_num_batched_tokens"] == "17408"
    assert protocol["max_num_seqs"] == "32"
    assert protocol["kv_tokens_per_rank"] == "380800"
    assert protocol["kv_blocks"] == "2975"
    assert protocol["task_queue_enable"] == "1"
    assert protocol["cudagraph_mode"] == "PIECEWISE"
    assert protocol["capture_sizes"] == "[1,2,4,8,16,32]"
    assert protocol["attention_execution"] == "eager_split_boundary"
    assert protocol["moe_execution"] == "piecewise_aclgraph"
    assert protocol["tail_guard"] == "false"
    assert protocol["shrink"] == "false"
    assert protocol["data_shuffle"] == "false"
    code_manifest = (output / "code_sha256.txt").read_text()
    assert len(code_manifest.splitlines()) >= 10
    assert "vllm/compilation/backends.py" in code_manifest

    before = (output / "protocol.env").read_bytes()
    subprocess.run([str(RUNNER), "dry-run"], cwd=ROOT, env=env, check=True)
    assert (output / "protocol.env").read_bytes() == before


def test_piecewise_epoch0_launcher_has_no_run_side_effect_in_dry_run() -> None:
    source = RUNNER.read_text()
    assert "VLLM_ASCEND_ELASTIC_ACLGRAPH_CAPTURE_ATTENTION=0" in source
    assert "VLLM_ASCEND_ELASTIC_ACLGRAPH_CAPTURE_MOE=1" in source
    assert "ADAFLOOR_ACLGRAPH_MODE=PIECEWISE" in source
    assert "TASK_QUEUE_ENABLE=1" in source
    assert "ROLLOUT_ENFORCE_EAGER=False" in source
    assert "actor_rollout_ref.rollout.seed=\"$SEED\"" in source
    assert 'if [[ "$ACTION" == dry-run ]]' in source


def test_piecewise_epoch0_launcher_shell_syntax() -> None:
    subprocess.run(["bash", "-n", str(RUNNER)], check=True)
