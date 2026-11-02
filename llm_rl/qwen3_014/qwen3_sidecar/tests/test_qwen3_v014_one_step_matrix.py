from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pyarrow.parquet as pq

from tools.verify_qwen3_v014_one_step_matrix import (
    validate_output_contract,
    verify_shrink_transitions,
)


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "run_qwen3_v014_fair_one_step_matrix.sh"
COMMON = Path(
    "/workspace/adafloor_graph_results/"
    "qwen3_vanilla_epoch0_full_decode_fia_v014opt_tq1_buckets16_32_seed0/"
    "common_epoch0_full_decode_fia_v014opt_tq1_buckets16_32_seed0"
)


def test_resume_reset_is_explicit_and_skips_dataloader_state() -> None:
    source = (ROOT / "verl/trainer/ppo/ray_trainer.py").read_text()
    reset = source.index("VERL_RESET_TRAINER_PROGRESS_AFTER_RESUME")
    reset_step = source.index("self.global_steps = 0", reset)
    early_return = source.index("return", reset_step)
    dataloader = source.index("# load dataloader,", reset)
    assert reset < reset_step < early_return < dataloader


def test_verifier_accepts_stock_v014_rollout_rows() -> None:
    rows = []
    lengths = []
    for prompt in range(32):
        for sample in range(16):
            rows.append({
                "input": f"prompt-{prompt}",
                "gts": str(prompt),
                "output": f"answer-{prompt}-{sample}",
                "request_id": sample,
                "score": float(sample == 0),
                "step": 1,
            })
            lengths.append(10 + sample)
    contract = validate_output_contract(rows, lengths, "base-fixture")
    assert contract["schema"] == "v0.14_base"
    assert len(contract["identities"]) == 512
    assert len(contract["lengths_by_identity"]) == 512


def test_verifier_accepts_dynamic_natural_survivor_sets() -> None:
    stages = [
        list(range(16)),
        [7, 9, 10, 11, 12, 13, 14, 15],
        [10, 12, 13, 15],
        [13, 15],
    ]
    lines = []
    for current, target in zip(stages[:-1], stages[1:], strict=True):
        for _ in current:
            lines.append(
                "AdaFloor staged trigger: stage=x "
                f"current={current} unfinished={target} target={target} policy=natural"
            )
        for rank in target:
            lines.append(
                "Elastic ACLGraph topology capture finished: "
                f"rank={rank} active_ranks={target}"
            )
            lines.append(
                "Elastic parallel shrink done: "
                f"rank={rank} active_ranks={target} "
                f"dp_size={len(target)} ep_size={len(target)}"
            )
    assert verify_shrink_transitions(
        "\n".join(lines), "natural-fixture", "ada_n2", True
    ) == stages[1:]


def test_runner_cpu_smoke_builds_the_shared_real_step(tmp_path: Path) -> None:
    assert COMMON.is_dir()
    subprocess.run(["bash", "-n", str(RUNNER)], check=True)
    result_root = tmp_path / "matrix"
    env = os.environ.copy()
    env["QWEN3_V014_FAIR_ROOT"] = str(result_root)
    completed = subprocess.run(
        ["bash", str(RUNNER), "smoke"],
        cwd=ROOT,
        env=env,
        check=True,
        text=True,
        capture_output=True,
    )
    assert "SMOKE_PASS" in completed.stdout
    protocol = json.loads((result_root / "protocol.json").read_text())
    assets = json.loads((result_root / "assets/asset_manifest.json").read_text())
    assert protocol["expected_responses"] == 512
    assert protocol["max_response_length"] == 6144
    assert protocol["source_plan_step"] == 2
    assert protocol["fused_moe_backend"] == "legacy"
    assert protocol["automatic_mode1_redundancy"] is True
    assert protocol["planned_topology_precreation"] is True
    assert protocol["graph"] == {
        "capture_sizes": [16, 32],
        "mode": "FULL_DECODE_ONLY",
        "task_queue": 1,
    }
    assert protocol["eager"] == {"enforce_eager": True, "task_queue": 2}
    assert assets["source_plan_step"] == 2
    assert len(assets["selected_source_indices"]) == 32
    assert pq.read_table(result_root / "assets/ada_n2_train.parquet").num_rows == 32
    assert pq.read_table(result_root / "assets/ada_p_f4_train.parquet").num_rows == 32
    n2 = json.loads((result_root / "assets/ada_n2_plan.json").read_text())[0]
    f4 = json.loads((result_root / "assets/ada_p_f4_plan.json").read_text())[0]
    assert n2["selected_floor"] == 2
    assert n2["source_plan_step"] == 2
    assert n2["shrink_stages"] == [8, 4, 2]
    assert f4["selected_floor"] == 4
    assert f4["source_plan_step"] == 2
    assert f4["shrink_stages"] == [8, 4]
    assert n2["tail_guard_enabled"] is False
    assert f4["tail_guard_enabled"] is False
    assert n2["release_area"] > 0
    assert f4["release_area"] > 0
    assert n2["release_area_unit"] == "rank_token_proxy"
    assert f4["release_area_unit"] == "rank_token_proxy"
    summary = json.loads((result_root / "one_step_matrix_summary.json").read_text())
    assert summary["status"] == "INCOMPLETE"
    assert summary["completed_arms"] == []
