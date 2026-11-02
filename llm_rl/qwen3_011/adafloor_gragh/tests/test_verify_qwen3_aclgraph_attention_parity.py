from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
VERIFIER = ROOT / "tools/verify_qwen3_aclgraph_attention_parity.py"


def _module():
    spec = importlib.util.spec_from_file_location("attention_parity", VERIFIER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_arm(
    root: Path,
    mode: str,
    source_files: list[Path],
    graph_mode: str = "PIECEWISE",
) -> None:
    arm = root / mode
    run = arm / "run"
    logs = run / "logs"
    logs.mkdir(parents=True)
    protocol = {
        "schema_version": "1",
        "mode": mode,
        "stack": "vllm-0.11.0_vllm-ascend-0.11.0rc0",
        "task_queue_enable": "1",
        "batch_size": "16",
        "rollout_n": "1",
        "decode_tokens": "64",
        "temperature": "0.0",
        "seed": "101",
        "warmup_steps": "0",
        "measure_steps": "2",
        "attention_graph": "true" if mode == "graph" else "false",
        "moe_graph": (
            "true" if mode == "graph" and graph_mode == "FULL_DECODE_ONLY"
            else "false"
        ),
        "cudagraph_mode": graph_mode if mode == "graph" else "NONE",
    }
    (arm / "protocol.env").write_text(
        "".join(f"{key}={value}\n" for key, value in protocol.items())
    )
    import hashlib

    (arm / "code_sha256.txt").write_text(
        "".join(
            f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path}\n"
            for path in source_files
        )
    )
    rows = []
    for step in range(2):
        rows.append(
            {
                "batch_size": 16,
                "rollout_n": 1,
                "prompt_len": 512,
                "decode_len": 64,
                "step": step,
                "phase": "measure",
                "generated_samples": 16,
                "decode_tokens": 1024,
                "response_token_sha256": f"response-{step}",
                "response_row_token_sha256": [f"{index:064x}" for index in range(16)],
                "prompt_token_sha256": f"prompt-{step}",
                "wall_s": 10.0 if mode == "eager" else 8.0,
                "timing": {
                    "generation_timing/max": 6.0 if mode == "eager" else 5.0
                },
            }
        )
    (run / "steps.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows)
    )
    (run / "summary_batch_16.json").write_text(
        json.dumps({"measure_steps": 2})
    )
    if mode == "graph":
        lines = []
        for pid in range(100, 116):
            prefix = f"(WorkerDict pid={pid})"
            if graph_mode == "FULL_DECODE_ONLY":
                lines.extend(
                    [
                        f"{prefix} FULL_DECODE_ONLY compilation enabled on NPU",
                        f"{prefix} Native ACLGraph rollout configured: "
                        "mode=FULL_DECODE_ONLY capture_sizes=[1]",
                        f"{prefix} Native FULL_DECODE_ONLY ACLGraph enabled: KV write, "
                        "Attention read, MoE, and dense decode execute in the outer graph",
                        f"{prefix} FULL_DECODE_ONLY ACLGraph captured Attention KV write "
                        "and paged read inside the outer model graph",
                        f"{prefix} FULL_DECODE_ONLY Attention maximum workspace "
                        "captured: seq_len_bucket=6144 bytes=4096 shape=1",
                        f"{prefix} Attention ACLGraph metadata update active",
                        f"{prefix} Graph capturing finished in 1 secs, took 0.20 GiB",
                        f"{prefix} Replaying aclgraph",
                    ]
                )
            else:
                lines.extend(
                    [
                        f"{prefix} Elastic ACLGraph boundary enabled: elastic Ascend "
                        "MoE/HCCL executes outside PIECEWISE ACLGraph; Attention's "
                        "dynamic KV write remains outside and its paged read core "
                        "executes in a nested ACLGraph",
                        f"{prefix} Attention ACLGraph metadata update active",
                        f"{prefix} Graph capturing finished in 1 secs, took 0.20 GiB",
                        f"{prefix} Replaying aclgraph",
                    ]
                )
        log = "\n".join(lines) + "\n"
    else:
        log = "eager rollout complete\n"
    (logs / "primary.txt").write_text(log)


def _fixture(tmp_path: Path, graph_mode: str = "PIECEWISE") -> Path:
    root = tmp_path / "pair"
    sources = []
    for index in range(8):
        path = tmp_path / f"source_{index}.py"
        path.write_text(f"VALUE = {index}\n")
        sources.append(path)
    _write_arm(root, "eager", sources)
    _write_arm(root, "graph", sources, graph_mode=graph_mode)
    return root


def test_verifier_accepts_exact_token_pair(tmp_path: Path) -> None:
    result = _module().verify(_fixture(tmp_path))
    assert result["status"] == "PASS"
    assert result["correctness"]["paired_measured_batches"] == 2
    assert result["correctness"]["all_response_tokens_equal"] is True
    assert result["performance"]["outer_wall_delta_percent"] == pytest.approx(-20.0)
    assert result["performance"]["inner_generation_delta_percent"] == pytest.approx(
        -100.0 / 6.0
    )
    assert result["graph_evidence"]["marker_worker_counts"]["attention_update"] == 16


def test_verifier_accepts_full_decode_kv_write_pair(tmp_path: Path) -> None:
    result = _module().verify(_fixture(tmp_path, graph_mode="FULL_DECODE_ONLY"))
    assert result["status"] == "PASS"
    assert result["performance"]["graph_mode"] == "FULL_DECODE_ONLY"
    assert result["correctness"]["full_decode_kv_write_graph_parity"] is True
    assert result["graph_evidence"]["marker_worker_counts"]["kv_write_capture"] == 16


def test_verifier_rejects_full_decode_piecewise_fallback(tmp_path: Path) -> None:
    verifier = _module()
    root = _fixture(tmp_path, graph_mode="FULL_DECODE_ONLY")
    path = root / "graph/run/logs/primary.txt"
    path.write_text(
        path.read_text()
        + "Attention's dynamic KV write remains outside and its paged read core "
        "executes in a nested ACLGraph\n"
    )
    with pytest.raises(verifier.VerificationError, match="fell back"):
        verifier.verify(root)


@pytest.mark.parametrize(
    "bad_event",
    [
        "Starting to capture ACL graphs for cases: [2], mode: FULL, "
        "uniform_decode: False",
        "Elastic full-MoE ACLGraph replay: generation=0 "
        "batch_descriptor=BatchDescriptor(num_tokens=2, uniform_decode=False)",
    ],
)
def test_verifier_rejects_full_decode_mixed_prefill_graph(
    tmp_path: Path,
    bad_event: str,
) -> None:
    verifier = _module()
    root = _fixture(tmp_path, graph_mode="FULL_DECODE_ONLY")
    path = root / "graph/run/logs/primary.txt"
    path.write_text(path.read_text() + bad_event + "\n")
    with pytest.raises(verifier.VerificationError, match="mixed prefill/decode"):
        verifier.verify(root)


def test_verifier_rejects_response_token_mismatch(tmp_path: Path) -> None:
    verifier = _module()
    root = _fixture(tmp_path)
    path = root / "graph/run/steps.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[1]["response_row_token_sha256"][3] = "f" * 64
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(verifier.VerificationError, match="differs at"):
        verifier.verify(root)


def test_verifier_rejects_incomplete_graph_worker_coverage(tmp_path: Path) -> None:
    verifier = _module()
    root = _fixture(tmp_path)
    path = root / "graph/run/logs/primary.txt"
    path.write_text(
        "\n".join(
            line
            for line in path.read_text().splitlines()
            if not ("pid=115" in line and "metadata update" in line)
        )
        + "\n"
    )
    with pytest.raises(verifier.VerificationError, match="covered 15 workers"):
        verifier.verify(root)
