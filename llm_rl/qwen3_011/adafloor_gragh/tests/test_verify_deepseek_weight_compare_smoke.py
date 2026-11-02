from __future__ import annotations

import json
from pathlib import Path

import pytest

import tools.verify_deepseek_weight_compare_smoke as verifier


def _write(path: Path, content: str = "fixture\n") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _routed_result(rank: int) -> dict[str, object]:
    experts = list(range(rank * 4, rank * 4 + 4))
    stream_matches: dict[str, str] = {}
    post_matches: dict[str, str] = {}
    for expert in experts:
        for projection in ("gate_proj", "up_proj"):
            label = f"expert{expert}.{projection}"
            stream_matches[
                f"model.layers.1.mlp.experts.{expert}.{projection}.weight"
            ] = label
            post_matches[label] = label
    return {
        "global_to_local": {
            expert: slot for slot, expert in enumerate(experts)
        },
        "expected_stream_count": 8,
        "captured_stream_count": 8,
        "missing_stream_names": [],
        "extra_stream_names": [],
        "stream_matches": stream_matches,
        "post_matches": post_matches,
    }


def _runtime_log() -> list[str]:
    lines: list[str] = []
    for rank in range(verifier.WORLD_SIZE):
        lines.append(
            "Captured initial DeepSeek HF weight samples: "
            f"rank={rank} params={verifier.EXPECTED_PARAMETERS_PER_RANK}"
        )
        lines.append(
            "DeepSeek online-sync HF comparison: "
            f"rank={rank} total=324 matched=324 mismatched=0 "
            f"category_totals={verifier.EXPECTED_CATEGORY_TOTALS!r} "
            "category_mismatches={} missing=[] sample_errors=[] "
            "mismatch_samples=[]"
        )
        lines.append(
            "DeepSeek layer-1 routed w13 comparison: "
            f"rank={rank} result={_routed_result(rank)!r}"
        )
        lines.append(
            "DeepSeek online-sync HF comparison PASS: "
            f"rank={rank} total=324 routed_streams=8"
        )
    lines.extend(
        [
            "site-packages/torch/_dynamo/metrics_context.py:105] "
            "Traceback (most recent call last):",
            "step:1 - actor/grad_norm:10.638793608457652 - actor/lr:1e-06 "
            "- training/global_step:1 - training/epoch:0 "
            "- response/aborted_ratio:0.0 - timing_s/update_actor:8.398917 "
            "- response_length/mean:32.0 - response_length/min:32.0 "
            "- response_length/max:32.0 - perf/total_num_tokens:3457",
            "Epoch 0 completed in 42.40 seconds.",
            "[ERROR] TBE Subprocess[task_distribute] raise error[], "
            "main process disappeared!",
            "Process ForkServerProcess-1:",
            "Traceback (most recent call last):",
            "ProcessLookupError: [Errno 3] No such process",
        ]
    )
    return lines


def _make_fixture(tmp_path: Path) -> tuple[Path, Path]:
    model = tmp_path / "DeepSeek-V2-Lite-Chat"
    distcp = tmp_path / "DeepSeek-V2-Lite-Chat_megatron_pp4_ep4"
    run_root = tmp_path / "run"
    epoch = run_root / "epoch_000_mode0_probe"
    log_path = epoch / "logs" / "runtime.txt"

    _write(
        model / "config.json",
        json.dumps(
            {
                "architectures": ["DeepseekV2ForCausalLM"],
                "n_routed_experts": 64,
                "num_experts_per_tok": 6,
            }
        ),
    )
    _write(model / "model.safetensors.index.json", "{}\n")
    for shard in range(1, 5):
        _write(model / f"model-{shard:05d}-of-00004.safetensors")

    _write(
        distcp / ".adafloor_deepseek_v2_lite_manifest.json",
        json.dumps(
            {
                "architecture": "DeepseekV2ForCausalLM",
                "expert_model_parallel_size": 4,
                "model_id": verifier.EXPECTED_MODEL_ID,
                "model_revision": verifier.EXPECTED_MODEL_REVISION,
                "pipeline_model_parallel_size": 4,
                "world_size": 16,
            }
        ),
    )
    _write(distcp / ".metadata")
    for shard in range(32):
        _write(distcp / f"__{shard}_0.distcp")

    complete_values = {
        **verifier.EXPECTED_COMPLETE_VALUES,
        "MODEL_PATH": str(model),
        "DISTCP_PATH": str(distcp),
    }
    _write(
        run_root / "COMPLETE",
        "COMPLETE DeepSeek actor update probe\n"
        + "".join(f"{key}={value}\n" for key, value in complete_values.items()),
    )
    _write(log_path, "\n".join(_runtime_log()) + "\n")

    rollout_rows = []
    for ordinal in range(verifier.EXPECTED_ROWS):
        rollout_rows.append(
            {
                "output": f"answer {ordinal}",
                "score": 0.0,
                "step": 1,
                "request_id": str(ordinal % 2),
                "rollout_rank": ordinal // 2,
                "decoded_response_length": 32,
                "response_finish_reason": "length",
                "responses": list(range(32)),
                "response_mask": [1] * 32,
            }
        )
    _write(
        epoch / "rollout_data" / "1.jsonl",
        "".join(json.dumps(row) + "\n" for row in rollout_rows),
    )
    _write(epoch / "rollout_length" / "length_1.txt", "32\n" * 32)
    return run_root, log_path


def test_verify_run_accepts_complete_smoke_evidence(tmp_path: Path) -> None:
    run_root, _ = _make_fixture(tmp_path)

    result = verifier.verify_run(run_root)

    assert result["status"] == "PASS"
    assert result["model"]["model_revision"] == verifier.EXPECTED_MODEL_REVISION
    assert result["weight_comparison"]["rank_count"] == 16
    assert result["weight_comparison"]["parameter_tensor_comparisons"] == 5184
    assert result["weight_comparison"]["layer1_routed_experts_covered"] == list(
        range(64)
    )
    assert result["weight_comparison"]["actor_update"]["steps"] == 1
    assert result["weight_comparison"]["actor_update"]["learning_rate"] == 1e-6
    assert result["weight_comparison"]["actor_update"]["gradient_norm"] > 0
    assert result["rollout"]["responses"] == 32
    assert len(result["artifact_bundle_sha256"]) == 64
    assert result["weight_comparison"]["allowed_noise"] == {
        "torch_dynamo_metrics_trace_lines": 1,
        "post_completion_tbe_cleanup_lines": 1,
        "post_completion_process_lookup_errors": 1,
    }


def test_verify_run_rejects_missing_rank_pass(tmp_path: Path) -> None:
    run_root, log_path = _make_fixture(tmp_path)
    lines = log_path.read_text(encoding="utf-8").splitlines()
    lines = [
        line
        for line in lines
        if "comparison PASS: rank=15 " not in line
    ]
    _write(log_path, "\n".join(lines) + "\n")

    with pytest.raises(
        verifier.WeightCompareVerificationError,
        match="PASS rank coverage differs",
    ):
        verifier.verify_run(run_root)


def test_verify_run_rejects_category_mismatch(tmp_path: Path) -> None:
    run_root, log_path = _make_fixture(tmp_path)
    content = log_path.read_text(encoding="utf-8").replace(
        "'attention': 135", "'attention': 134", 1
    )
    _write(log_path, content)

    with pytest.raises(
        verifier.WeightCompareVerificationError,
        match="rank 0 category totals differ",
    ):
        verifier.verify_run(run_root)


def test_verify_run_rejects_precompletion_fatal(tmp_path: Path) -> None:
    run_root, log_path = _make_fixture(tmp_path)
    content = log_path.read_text(encoding="utf-8").replace(
        "step:1 - actor/", "RuntimeError: execution failed\nstep:1 - actor/"
    )
    _write(log_path, content)

    with pytest.raises(
        verifier.WeightCompareVerificationError,
        match="unexpected fatal runtime marker before epoch completion",
    ):
        verifier.verify_run(run_root)


def test_verify_run_rejects_unknown_postcompletion_fatal(tmp_path: Path) -> None:
    run_root, log_path = _make_fixture(tmp_path)
    _write(
        log_path,
        log_path.read_text(encoding="utf-8")
        + "RuntimeError: unrelated shutdown failure\n",
    )

    with pytest.raises(
        verifier.WeightCompareVerificationError,
        match="unexpected fatal runtime marker after epoch completion",
    ):
        verifier.verify_run(run_root)


@pytest.mark.parametrize("marker", ["NPU OOM", "request preempted"])
def test_verify_run_rejects_safety_marker(
    tmp_path: Path, marker: str
) -> None:
    run_root, log_path = _make_fixture(tmp_path)
    _write(log_path, log_path.read_text(encoding="utf-8") + marker + "\n")

    with pytest.raises(verifier.WeightCompareVerificationError):
        verifier.verify_run(run_root)


def test_verify_run_rejects_rollout_length_disagreement(tmp_path: Path) -> None:
    run_root, _ = _make_fixture(tmp_path)
    length_path = (
        run_root
        / "epoch_000_mode0_probe"
        / "rollout_length"
        / "length_1.txt"
    )
    _write(length_path, "32\n" * 31 + "31\n")

    with pytest.raises(
        verifier.WeightCompareVerificationError,
        match="rollout length file",
    ):
        verifier.verify_run(run_root)
