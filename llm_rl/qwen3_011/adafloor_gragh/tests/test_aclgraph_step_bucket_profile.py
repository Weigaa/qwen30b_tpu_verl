from __future__ import annotations

import importlib.util
import json
import subprocess
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SELECTOR = ROOT / "analysis_eval/select_aclgraph_step_buckets.py"
SUMMARIZER = ROOT / "analysis_eval/summarize_aclgraph_step_bucket_profile.py"
TRACE_BUILDER = ROOT / "analysis_eval/build_qwen3_aclgraph_profile_trace.py"
RUNNER = ROOT / "run_qwen3_aclgraph_step_bucket_profile.sh"


def _load_selector():
    spec = importlib.util.spec_from_file_location("bucket_selector", SELECTOR)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_summarizer():
    spec = importlib.util.spec_from_file_location("bucket_summarizer", SUMMARIZER)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_exact_dynamic_programming_bucket_selection() -> None:
    module = _load_selector()
    histogram = {1: 10, 4: 1, 32: 10}
    selected = module._optimal_sets(histogram)
    assert selected[1] == ([32], 338)
    # With two buckets, size 1 is the exact optimum for the heavy batch-1 mass.
    assert selected[2] == ([1, 32], 28)
    assert selected[3] == ([1, 4, 32], 0)


def test_selector_builds_step_subset_and_exact_candidates(tmp_path: Path) -> None:
    records = []
    for prompt in range(160):
        records.append({"input": f"prompt-{prompt}", "lengths": [32] * 16})
    history = tmp_path / "history.json"
    history.write_text(json.dumps({"records": records}), encoding="utf-8")
    train = tmp_path / "train.parquet"
    pd.DataFrame({"row": list(range(160))}).to_parquet(train, index=False)
    plan = tmp_path / "plan.json"
    subset = tmp_path / "subset.parquet"
    subprocess.run(
        [
            "python3",
            str(SELECTOR),
            "--history",
            str(history),
            "--train-file",
            str(train),
            "--source-step",
            "4",
            "--counts",
            "1,2,32",
            "--output-plan",
            str(plan),
            "--output-train",
            str(subset),
        ],
        check=True,
    )
    payload = json.loads(plan.read_text(encoding="utf-8"))
    assert payload["source_row_begin"] == 96
    assert payload["source_row_end_exclusive"] == 128
    assert [item["graph_count"] for item in payload["candidates"]] == [1, 2, 32]
    assert pd.read_parquet(subset)["row"].tolist() == list(range(96, 128))


def test_profile_runner_contract() -> None:
    subprocess.run(["bash", "-n", str(RUNNER)], check=True)
    text = RUNNER.read_text(encoding="utf-8")
    assert "FAIR_TRAIN_STEPS=1" in text
    assert "FAIR_FREEZE_ACTOR=1" in text
    assert "VERL_PAIRED_REQUEST_SAMPLING_SEEDS=1" in text
    assert "TASK_QUEUE_ENABLE=1" in text
    assert "ADAFLOOR_ACLGRAPH_MODE=FULL_DECODE_ONLY" in text
    assert "VANILLA_KV_TOKENS_PER_RANK=\"$full16_kv_tokens\"" in text
    assert "ACLGRAPH_BUCKET_PROFILE_RESERVE_MODE" in text
    assert "reserve_for_count" in text
    assert "SOURCE_STEP=${ACLGRAPH_BUCKET_PROFILE_SOURCE_STEP:-4}" in text
    assert 'kill -TERM "$sampler_pid"' in text
    assert "VERL_FIXED_WORK_REPLAY_TRACE=\"$FIXED_WORK_TRACE\"" in text
    assert "VERL_FIXED_WORK_REPLAY_REQUIRE_PLAN_CAP=0" in text
    assert "VERL_FIXED_WORK_REPLAY_ALLOW_DATASET_ITEM_IDX=1" in text
    assert "VERL_FIXED_WORK_REPLAY_FORCE_TOKEN_IDS=1" in text
    assert "FAIR_ROLLOUT_TEMPERATURE=0" in text
    assert "FAIR_ROLLOUT_TOP_P=1" in text
    assert "FAIR_ROLLOUT_TOP_K=-1" in text


def test_forced_token_replay_runs_after_sampler_before_state_writeback() -> None:
    runner = (
        ROOT / "vllm_ascend/worker/model_runner_v1.py"
    ).read_text(encoding="utf-8")
    sampled = runner.index("valid_sampled_token_ids = sampled_token_ids.tolist()")
    forced = runner.index(
        "valid_sampled_token_ids = self._maybe_force_fixed_work_token_ids("
    )
    state_write = runner.index("req_state.output_token_ids.extend(sampled_ids)")
    assert sampled < forced < state_write
    assert 'if not _env_flag("VERL_FIXED_WORK_REPLAY_FORCE_TOKEN_IDS")' in runner


def test_summary_requires_parity_for_forced_replay() -> None:
    text = SUMMARIZER.read_text(encoding="utf-8")
    assert 'protocol.get("forced_token_replay") == "true"' in text
    assert '"forced_token_replay": forced_token_replay' in text
    assert '"two_graph_acceptance_threshold_percent": 1.0' in text
    assert '"recommended_graph_count": recommended_count' in text
    assert '"requires_two_arm_reverse_order_confirmation"' in text


def test_capture_parser_accepts_forwarded_rollout_cycle_only() -> None:
    module = _load_summarizer()
    report = "Graph capturing finished in 11 secs, took 0.28 GiB\n"
    cycles = module._capture_cycles(report * 16)
    assert len(cycles) == 1
    assert cycles[0] == [(11.0, 0.28)] * 16


def test_capture_parser_rejects_incomplete_rank_cycle() -> None:
    module = _load_summarizer()
    report = "Graph capturing finished in 11 secs, took 0.28 GiB\n"
    try:
        module._capture_cycles(report * 15)
    except ValueError as exc:
        assert "15 reports" in str(exc)
    else:
        raise AssertionError("incomplete capture cycle was accepted")


def test_qwen3_profile_trace_builds_and_loads(tmp_path: Path) -> None:
    source = tmp_path / "1.jsonl"
    rows = []
    for occurrence in range(32):
        for sample in range(16):
            rows.append(
                {
                    "prompt_occurrence_ordinal": occurrence,
                    "rollout_sample_index": sample,
                    "rollout_prompt_hash": f"hash-{occurrence}",
                    "rollout_request_seed": occurrence * 100 + sample,
                    "decoded_response_length": occurrence + sample + 1,
                    "responses": list(range(occurrence + sample + 1)),
                }
            )
    source.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    trace = tmp_path / "trace.json"
    subprocess.run(
        [
            "python3",
            str(TRACE_BUILDER),
            "--source-rollout",
            str(source),
            "--output",
            str(trace),
        ],
        check=True,
    )
    import hashlib
    from verl.utils.fixed_work_replay import load_fixed_work_replay

    digest = hashlib.sha256(trace.read_bytes()).hexdigest()
    replay = load_fixed_work_replay(trace, expected_sha256=digest)
    assert replay.record_count == 512
    assert replay.target_generated_tokens == sum(
        row["decoded_response_length"] for row in rows
    )
    assert replay.target_for_occurrence(3, 4, "hash-3", 304) == 8
    assert replay.target_token_ids_for_occurrence(3, 4, "hash-3", 304) == tuple(
        range(8)
    )
