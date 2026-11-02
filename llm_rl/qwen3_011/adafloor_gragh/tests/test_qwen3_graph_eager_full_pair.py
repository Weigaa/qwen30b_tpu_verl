from __future__ import annotations

import importlib.util
import inspect
import json
import os
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "run_qwen3_adafloor_graph_eager_full_pair.sh"
VERIFIER = ROOT / "tools/verify_qwen3_adafloor_graph_eager_full_pair.py"


def _module():
    spec = importlib.util.spec_from_file_location("graph_eager_verifier", VERIFIER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_runner_dry_run_seals_minimal_full_protocol(tmp_path: Path) -> None:
    root = tmp_path / "pair"
    environment = os.environ.copy()
    environment["ADAFLOOR_GRAPH_EAGER_PAIR_ROOT"] = str(root)
    result = subprocess.run(
        [str(RUNNER), "dry-run"],
        cwd=ROOT,
        env=environment,
        check=True,
        text=True,
        capture_output=True,
    )
    assert result.stdout.count("arm=") == 8
    assert [
        line.split()[0].removeprefix("arm=")
        for line in result.stdout.splitlines()
        if line.startswith("arm=")
    ] == [
        "vanilla/eager",
        "vanilla/graph",
        "lengthsort_guard/graph",
        "lengthsort_guard/eager",
        "planned/eager",
        "planned/graph",
        "natural/graph",
        "natural/eager",
    ]
    protocol = dict(
        line.split("=", 1)
        for line in (root / "protocol.env").read_text().splitlines()
    )
    assert protocol["eager_full16_kv_tokens"] == "380800"
    assert protocol["graph_full16_kv_tokens"] == "380800"
    assert protocol["eager_task_queue_enable"] == "2"
    assert protocol["graph_task_queue_enable"] == "1"
    assert protocol["schema_version"] == "2"
    assert protocol["experiment"] == "qwen3_aclgraph_full_epoch_matrix"
    assert protocol["common_epoch0_mode"] == "graph_vanilla"
    assert protocol["common_epoch0_actor_updated"] == "true"
    assert protocol["graph_mode"] == "FULL_DECODE_ONLY"
    assert protocol["graph_attention"] == "true"
    assert protocol["graph_moe"] == "true"
    assert protocol["graph_capture_sizes"] == "[1,2,4,8,16,32]"
    assert protocol["steps"] == "5"
    assert protocol["responses_per_step"] == "512"
    assert protocol["epoch1_actor_frozen"] == "true"
    assert len((root / "code_sha256.txt").read_text().splitlines()) >= 20

    # A same-content resume must preserve the immutable protocol byte for byte.
    before = (root / "protocol.env").read_bytes()
    subprocess.run(
        [str(RUNNER), "dry-run"], cwd=ROOT, env=environment, check=True
    )
    assert (root / "protocol.env").read_bytes() == before


def test_verifier_accepts_empty_incomplete_prefix(tmp_path: Path) -> None:
    root = tmp_path / "pair"
    environment = os.environ.copy()
    environment["ADAFLOOR_GRAPH_EAGER_PAIR_ROOT"] = str(root)
    subprocess.run([str(RUNNER), "dry-run"], cwd=ROOT, env=environment, check=True)
    subprocess.run(
        [str(VERIFIER), "--root", str(root), "--allow-incomplete"],
        cwd=ROOT,
        check=True,
    )
    summary = json.loads((root / "aclgraph_full_epoch_matrix_summary.json").read_text())
    assert summary["status"] == "INCOMPLETE"
    assert summary["common_epoch0"] is None
    assert summary["completed_arms"] == 0
    assert summary["kv_memory"]["eager_full16_gib"] == 34.86328125
    assert summary["kv_memory"]["graph_full16_gib"] == 34.86328125
    assert summary["kv_memory"]["graph_saved_gib_per_rank"] == 0.0


def test_plan_contract_distinguishes_planned_and_natural(tmp_path: Path) -> None:
    verifier = _module()
    caps = [5120, 7168, 10240, 14848, 16384]
    for policy, floors in {
        "planned": [4, 4, 8, 8, 16],
        "natural": [4, 4, 4, 4, 16],
    }.items():
        plan = []
        for step, (floor, cap) in enumerate(zip(floors, caps), 1):
            plan.append(
                {
                    "step": step,
                    "selected_floor": floor,
                    "tail_guard_enabled": step < 5,
                    "tail_guard_response_cap": cap,
                    "shrink_stages": [value for value in (8, 4) if value >= floor],
                    "feasible": True,
                }
            )
        path = tmp_path / f"{policy}.json"
        path.write_text(json.dumps(plan))
        observed = verifier.verify_plan(path, policy)
        assert observed["selected_floors"] == floors
        assert observed["tail_guard_caps"] == caps


def test_pair_comparison_reports_work_normalized_throughput() -> None:
    verifier = _module()
    requests = {
        (0, 0): {
            "identity": {"step": 1, "prompt_hash": "a", "request_seed": 1},
            "length": 10,
            "tokens": "same",
            "text": "same",
            "score": 1,
        }
    }
    eager = {
        "policy": "planned",
        "plan_artifact": {"sha256": "p"},
        "requests": requests,
        "runtime": {
            "generated_tokens_total": 100,
            "response_tokens_per_second": 10.0,
            "rollout_seconds_total": 10.0,
        },
    }
    graph = {
        "policy": "planned",
        "plan_artifact": {"sha256": "p"},
        "requests": requests,
        "runtime": {
            "generated_tokens_total": 100,
            "response_tokens_per_second": 12.5,
            "rollout_seconds_total": 8.0,
        },
    }
    result = verifier.compare_pair(eager, graph)
    assert result["same_tokens"] == 1
    assert result["rollout_time_delta_percent"] == pytest.approx(-20.0)
    assert result["throughput_delta_percent"] == pytest.approx(25.0)


def test_full_decode_verifier_does_not_require_piecewise_moe_marker() -> None:
    verifier = _module()
    source = inspect.getsource(verifier.verify_log)
    assert "Elastic ACLGraph MoE capture enabled" not in source
    assert "Elastic full-MoE ACLGraph replay:" in source
