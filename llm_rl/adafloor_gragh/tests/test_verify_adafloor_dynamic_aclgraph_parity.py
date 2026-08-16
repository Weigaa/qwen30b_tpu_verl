from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools/verify_adafloor_dynamic_aclgraph_parity.py"
SPEC = importlib.util.spec_from_file_location("dynamic_aclgraph_parity", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _write_arm(
    root: Path,
    mode: str,
    *,
    policy: str = "natural",
    floor: int = 2,
    executed_steps: int = 1,
) -> None:
    (root / "oracle").mkdir(parents=True)
    (root / "rollout_data").mkdir()
    (root / "rollout_length").mkdir()
    protocol = {
        "schema_version": "1",
        "experiment": "qwen3_adafloor_full_decode_dynamic_gate",
        "policy": policy,
        "execution_mode": mode,
        "stack": "vllm-0.11.0_vllm-ascend-0.11.0rc0_cann-8.5.0",
        "cudagraph_mode": "FULL_DECODE_ONLY" if mode == "graph" else "NONE",
        "capture_sizes": "2" if mode == "graph" else "none",
        "attention_graph": "true" if mode == "graph" else "false",
        "moe_graph": "true" if mode == "graph" else "false",
        "task_queue_enable": "1" if mode == "graph" else "2",
        "plan_steps": "2",
        "executed_steps": str(executed_steps),
        "forced_floors": f"{floor},16",
        "profile_max_tokens": "2048",
        "profile_expected_moe_comm": "alltoall",
        "max_num_batched_tokens": "17408",
        "max_model_len": "17408",
        "kv_tokens_floor2": "98304",
        "tail_validation_tokens": "8,16,32,64,64",
        "baseline": "/data/common_epoch0",
    }
    (root / "protocol.env").write_text(
        "".join(f"{key}={value}\n" for key, value in protocol.items()), "utf-8"
    )
    plan = [
        {
            "step": 1,
            "selected_floor": floor,
            "shrink_stages": [8, 4, 2] if floor == 2 else [8, 4],
            "tail_guard_enabled": False,
        },
        {
            "step": 2,
            "selected_floor": 16,
            "shrink_stages": [16],
            "tail_guard_enabled": False,
        },
    ]
    (root / "oracle/length_sorted_rank_plan.json").write_text(
        json.dumps(plan), "utf-8"
    )
    (root / "oracle/length_sorted_train.parquet").write_bytes(b"same parquet")
    for step in range(1, executed_steps + 1):
        rows = []
        lengths = []
        first_request_id = 2 * (step - 1)
        for rank in range(16):
            for sample_offset in (0, 1):
                request_id = str(first_request_id + sample_offset)
                length = rank + 5 if sample_offset == 0 else 35 - rank
                lengths.append(length)
                rows.append(
                    {
                        "step": step,
                        "rollout_rank": rank,
                        "request_id": request_id,
                        "input": f"prompt-{step}-{rank}-{request_id}",
                        "prompts": [step, rank, sample_offset],
                        "gts": str(rank),
                        "responses": list(range(length)) + [0] * (64 - length),
                        "response_mask": [1] * length + [0] * (64 - length),
                        "decoded_response_length": length,
                        "output": f"response-{step}-{rank}-{request_id}",
                        "score": 0,
                        "response_finish_reason": "length",
                    }
                )
        assert sum(lengths) == MODULE.TOKENS
        (root / f"rollout_data/{step}.jsonl").write_text(
            "".join(
                json.dumps(row, separators=(",", ":")) + "\n" for row in rows
            ),
            "utf-8",
        )
        (root / f"rollout_length/length_{step}.txt").write_text(
            "".join(f"{value}\n" for value in lengths), "utf-8"
        )
    if mode == "graph":
        (root / "ACLGRAPH_SMOKE_SUMMARY.json").write_text(
            json.dumps(
                {
                    "status": "PASS",
                    "runtime": {
                        "cudagraph_mode": "FULL_DECODE_ONLY",
                        "capture_sizes": [2],
                        "resident_full_decode_graphs_per_active_rank": 1,
                        "single_full_decode_graph_per_active_rank": True,
                        "topology_graph_cache": False,
                        "capture_invocations": (
                            16
                            + sum(plan[0]["shrink_stages"])
                            + (16 if executed_steps == 2 else 0)
                        ),
                        "captured_graph_instances": (
                            16
                            + sum(plan[0]["shrink_stages"])
                            + (16 if executed_steps == 2 else 0)
                        ),
                        "max_observed_capture_gib_per_worker": 1.87,
                        "target_policy": policy,
                        "step_2_recapture_workers": (
                            list(range(1000, 1016)) if executed_steps == 2 else []
                        ),
                        "prefill_decode_separation": {
                            "mixed_full_capture_events": 0,
                            "mixed_full_replay_events": 0,
                            "prefill_execution": "eager",
                            "uniform_decode_execution": "FULL_DECODE_ONLY",
                        },
                        "contract_fault_counts": {
                            "aclgraph_failure": 0,
                            "hccl_failure": 0,
                            "npu_oom": 0,
                        },
                    },
                }
            ),
            "utf-8",
        )


def _fixture(
    tmp_path: Path,
    *,
    policy: str = "natural",
    floor: int = 2,
    executed_steps: int = 1,
) -> tuple[Path, Path]:
    eager = tmp_path / "eager"
    graph = tmp_path / "graph"
    _write_arm(
        eager,
        "eager",
        policy=policy,
        floor=floor,
        executed_steps=executed_steps,
    )
    _write_arm(
        graph,
        "graph",
        policy=policy,
        floor=floor,
        executed_steps=executed_steps,
    )
    return eager, graph


def test_dynamic_single_graph_parity_passes(tmp_path: Path) -> None:
    eager, graph = _fixture(tmp_path)
    result = MODULE.verify(eager, graph)
    assert result["status"] == "PASS"
    assert result["correctness"]["paired_responses"] == 32
    assert result["correctness"]["jsonl_byte_identical"] is True
    assert result["graph_lifecycle"]["resident_full_decode_graphs_per_active_rank"] == 1
    assert result["graph_lifecycle"]["topology_graph_cache"] is False


def test_planned_f4_single_graph_parity_passes(tmp_path: Path) -> None:
    eager, graph = _fixture(tmp_path, policy="planned", floor=4)
    result = MODULE.verify(eager, graph)
    assert result["status"] == "PASS"
    assert result["configuration"]["policy"] == "Planned-F4"
    assert result["correctness"]["paired_responses"] == 32
    assert result["graph_lifecycle"]["capture_invocations"] == 28


@pytest.mark.parametrize(("policy", "floor"), [("planned", 4), ("natural", 2)])
def test_two_step_restore_recapture_parity_passes(
    tmp_path: Path,
    policy: str,
    floor: int,
) -> None:
    eager, graph = _fixture(
        tmp_path,
        policy=policy,
        floor=floor,
        executed_steps=2,
    )
    result = MODULE.verify(eager, graph)
    assert result["status"] == "PASS"
    assert result["configuration"]["executed_steps"] == 2
    assert result["correctness"]["paired_responses"] == 64
    assert result["correctness"]["generated_tokens_per_arm"] == 1280
    assert len(result["graph_lifecycle"]["step_2_recapture_workers"]) == 16


def test_two_step_parity_rejects_reused_step_one_request_ids(tmp_path: Path) -> None:
    eager, graph = _fixture(tmp_path, policy="planned", floor=4, executed_steps=2)
    for root in (eager, graph):
        path = root / "rollout_data/2.jsonl"
        rows = [json.loads(line) for line in path.read_text("utf-8").splitlines()]
        for row in rows:
            row["request_id"] = str(int(row["request_id"]) - 2)
        path.write_text(
            "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows),
            "utf-8",
        )
    with pytest.raises(MODULE.VerificationError, match="invalid request_id"):
        MODULE.verify(eager, graph)


def test_dynamic_single_graph_parity_rejects_token_difference(tmp_path: Path) -> None:
    eager, graph = _fixture(tmp_path)
    path = graph / "rollout_data/1.jsonl"
    rows = [json.loads(line) for line in path.read_text("utf-8").splitlines()]
    rows[17]["responses"][0] += 1
    path.write_text(
        "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows),
        "utf-8",
    )
    with pytest.raises(MODULE.VerificationError, match="response rows differ"):
        MODULE.verify(eager, graph)


def test_dynamic_single_graph_parity_rejects_multiple_shapes(tmp_path: Path) -> None:
    eager, graph = _fixture(tmp_path)
    path = graph / "protocol.env"
    path.write_text(path.read_text("utf-8").replace("capture_sizes=2", "capture_sizes=1,2"), "utf-8")
    with pytest.raises(MODULE.VerificationError, match="capture_sizes"):
        MODULE.verify(eager, graph)


def test_dynamic_single_graph_parity_rejects_topology_cache(tmp_path: Path) -> None:
    eager, graph = _fixture(tmp_path)
    path = graph / "ACLGRAPH_SMOKE_SUMMARY.json"
    summary = json.loads(path.read_text("utf-8"))
    summary["runtime"]["topology_graph_cache"] = True
    path.write_text(json.dumps(summary), "utf-8")
    with pytest.raises(MODULE.VerificationError, match="topology_graph_cache"):
        MODULE.verify(eager, graph)
