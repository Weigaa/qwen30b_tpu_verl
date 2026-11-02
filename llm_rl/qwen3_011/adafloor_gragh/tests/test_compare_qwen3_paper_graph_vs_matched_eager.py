import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "analysis_eval/compare_qwen3_paper_graph_vs_matched_eager.py"
MATCHED_WRAPPER = ROOT / "run_qwen3_paper_matched_eager_tq1_from_common_epoch0.sh"
SPEC = importlib.util.spec_from_file_location("graph_eager_compare", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def protocols() -> tuple[dict[str, str], dict[str, str]]:
    common = {
        "common_epoch0_root": "/data/common",
        "common_epoch0_reuse_env_sha256": "a" * 64,
        "common_epoch0_history_sha256": "b" * 64,
        "seeds": "0,2,3",
        "excluded_policy": "fixed4",
        "fixed4_exclusion_reason": "not_a_reasonable_comparison_strategy",
        "steps_per_epoch": "5",
        "actor_frozen": "false",
        "actor_lr": "1e-6",
        "prompts_per_epoch": "160",
        "train_batch_size": "32",
        "rollout_n": "16",
        "responses_per_step": "512",
        "max_response_length": "16384",
        "request_sampling": "global_seed_matched",
        "task_queue_enable": "1",
        "tailguard_policy": "matches_eager_per_policy",
        "kv_bytes_per_token": "98304",
        "kv_block_size_tokens": "128",
        "kv_admission_formula": "block_aware_prompt_plus_predicted_response_peak",
        "kv_admission_safety_factor": "1.16",
        "kv_admission_reserve_tokens_per_rank": "0",
        "kv_admission_prompt_tokenizer": "/data/Qwen3-30B-A3B",
        "kv_admission_scope": "release_area,minskew,contiguous",
        "kv_block_bytes": "12582912",
        "graph_reserve_bytes_per_npu": "2684354560",
        "graph_reserve_blocks_per_npu": "214",
        "graph_reserve_tokens_per_floor": "27392",
        "native_sleep_mode": "0",
        "async_scheduling": "false",
        "filtered_custom_opp": "true",
        "filtered_opp_bundle_sha256": "c" * 64,
        "completed_arm_cache_cleanup": "true",
        "common_epoch0_cleanup_forbidden": "true",
    }
    graph = {
        **common,
        "policies": ",".join(MODULE.GRAPH_POLICIES),
        "planned_arms": "21",
        "planned_epochs": "42",
        "epochs": "1,2",
        "graph_mode": "FULL_DECODE_ONLY",
        "capture_sizes": "[16,32]",
        "attention_graph": "true",
        "moe_graph": "true",
        "prefill_execution": "eager",
        "kv_formula": "graph_cap=eager_verified_cap-graph_reserve_tokens",
        "graph_full16_tokens": "353408",
        "graph_floor2_tokens": "103680",
        "graph_floor4_tokens": "253184",
        "graph_floor8_natural_tokens": "288256",
        "graph_floor8_floor4_policy_tokens": "349952",
    }
    eager = {
        **common,
        "policies": ",".join(MODULE.MATCHED_POLICIES),
        "planned_arms": "9",
        "planned_epochs": "9",
        "epochs": "1",
        "control_scope": "representative_vanilla_planned_f4_natural_f2",
        "comparison_difference": "aclgraph_execution_only",
        "execution_mode": "eager_matched",
        "graph_mode": "disabled",
        "capture_sizes": "none",
        "enforce_eager": "true",
        "attention_graph": "false",
        "moe_graph": "false",
        "prefix_caching": "true",
        "chunked_prefill": "true",
        "kv_formula": "matched_eager_cap=graph_cap",
        "reuse_seed0_vanilla": "0",
        "eager_full16_tokens": "353408",
        "eager_floor2_tokens": "103680",
        "eager_floor4_tokens": "253184",
        "eager_floor8_natural_tokens": "288256",
        "eager_floor8_floor4_policy_tokens": "349952",
        "matched_graph_full16_tokens": "353408",
        "matched_graph_floor2_tokens": "103680",
        "matched_graph_floor4_tokens": "253184",
        "matched_graph_floor8_natural_tokens": "288256",
        "matched_graph_floor8_floor4_policy_tokens": "349952",
    }
    return graph, eager


def summary(
    policies: tuple[str, ...],
    epochs_to_emit: tuple[int, ...],
    *,
    adafloor_preemptions: int = 0,
) -> dict:
    epochs = []
    trials = []
    for seed in MODULE.SEEDS:
        for policy in policies:
            preemptions = adafloor_preemptions if policy == "natural_f2" else 0
            for epoch in epochs_to_emit:
                epochs.append({
                    "trial_seed": seed,
                    "policy": policy,
                    "epoch": epoch,
                    "actor_lr": 1e-6,
                    "ooms": 0,
                    "max_aborted_ratio": 0.0,
                    "preemptions": preemptions,
                    "prompt_hash": "d" * 64,
                    "rollout_s": 100.0,
                    "full_s": 110.0,
                    "generated_tokens": 100000,
                    "work_normalized_tokens_s": 1000.0,
                    "reward": 0.8,
                    "released_rank_s": 10.0,
                })
            trials.append({
                "trial_seed": seed,
                "policy": policy,
                "epochs": len(epochs_to_emit),
                "actor_lr": 1e-6,
                "mean_epoch_rollout_s": 100.0,
                "work_normalized_tokens_s": 1000.0,
                "mean_reward": 0.8,
                "mean_generated_tokens": 100000.0,
                "mean_released_rank_s": 10.0,
                "preemptions": preemptions * 2,
            })
    return {
        "seeds": list(MODULE.SEEDS),
        "policies": list(policies),
        "raw_epochs": epochs,
        "raw_trials": trials,
    }


def test_protocols_prove_only_graph_and_kv_difference() -> None:
    graph, eager = protocols()
    caps = MODULE.validate_protocols(graph, eager)
    assert caps["graph_full16"] == 353408
    assert caps["eager_full16"] == 353408


def test_protocol_rejects_non_formula_kv_difference() -> None:
    graph, eager = protocols()
    eager["eager_floor2_tokens"] = "103808"
    try:
        MODULE.validate_protocols(graph, eager)
    except ValueError as exc:
        assert "matched eager and graph KV caps differ" in str(exc)
    else:
        raise AssertionError("invalid KV cap difference was accepted")


def test_summary_accepts_baseline_preemption_but_rejects_adafloor() -> None:
    baseline = summary(MODULE.GRAPH_POLICIES, MODULE.GRAPH_EPOCHS)
    baseline["raw_epochs"][2]["preemptions"] = 7
    MODULE.validate_summary(
        baseline, "baseline-preemption", MODULE.GRAPH_POLICIES, MODULE.GRAPH_EPOCHS
    )
    try:
        MODULE.validate_summary(
            summary(
                MODULE.GRAPH_POLICIES,
                MODULE.GRAPH_EPOCHS,
                adafloor_preemptions=1,
            ),
            "adafloor",
            MODULE.GRAPH_POLICIES,
            MODULE.GRAPH_EPOCHS,
        )
    except ValueError as exc:
        assert "AdaFloor preemption" in str(exc)
    else:
        raise AssertionError("AdaFloor preemption was accepted")


def test_comparison_source_has_paper_statistics_and_provenance() -> None:
    source = MODULE_PATH.read_text(encoding="utf-8")
    assert "T95_DF2 = 4.30265273" in source
    assert '"n_paired_seeds": 3' in source
    assert '"df": 2' in source
    assert "work_normalized_tokens_s" in source
    assert "prompt-set mismatch" in source
    assert '"provenance": provenance' in source
    assert "validate_code_manifests" in source
    assert "summary JSON hash does not match" in source


def test_code_manifests_must_match_except_eager_control_wrapper(tmp_path) -> None:
    shared = str(ROOT / "shared.py")
    wrapper = str(ROOT / "run_qwen3_paper_matched_eager_tq1_from_common_epoch0.sh")
    graph = tmp_path / "graph.code"
    eager = tmp_path / "eager.code"
    graph.write_text(f"{'a' * 64}  {shared}\n", encoding="utf-8")
    eager_summarizer = str(ROOT / "analysis_eval/summarize_matched_eager_controls.py")
    eager.write_text(
        f"{'a' * 64}  {shared}\n"
        f"{'b' * 64}  {wrapper}\n"
        f"{'d' * 64}  {eager_summarizer}\n",
        encoding="utf-8",
    )
    result = MODULE.validate_code_manifests(graph, eager)
    assert result["shared_source_hashes_identical"] is True
    eager.write_text(
        f"{'c' * 64}  {shared}\n"
        f"{'b' * 64}  {wrapper}\n"
        f"{'d' * 64}  {eager_summarizer}\n",
        encoding="utf-8",
    )
    try:
        MODULE.validate_code_manifests(graph, eager)
    except ValueError as exc:
        assert "frozen source hashes differ" in str(exc)
    else:
        raise AssertionError("source drift was accepted")


def test_full_cli_path_writes_pass_artifacts(tmp_path, monkeypatch) -> None:
    graph_protocol, eager_protocol = protocols()
    graph_protocol_path = tmp_path / "graph.protocol"
    eager_protocol_path = tmp_path / "eager.protocol"
    graph_protocol_path.write_text(
        "".join(f"{key}={value}\n" for key, value in graph_protocol.items()),
        encoding="utf-8",
    )
    eager_protocol_path.write_text(
        "".join(f"{key}={value}\n" for key, value in eager_protocol.items()),
        encoding="utf-8",
    )
    graph_summary = tmp_path / "graph.json"
    eager_summary = tmp_path / "eager.json"
    graph_summary.write_text(
        json.dumps(summary(MODULE.GRAPH_POLICIES, MODULE.GRAPH_EPOCHS)),
        encoding="utf-8",
    )
    eager_summary.write_text(
        json.dumps(summary(MODULE.MATCHED_POLICIES, MODULE.MATCHED_EPOCHS)),
        encoding="utf-8",
    )
    graph_code = tmp_path / "graph.code"
    eager_code = tmp_path / "eager.code"
    shared_source = str(ROOT / "shared.py")
    matched_wrapper = str(MATCHED_WRAPPER)
    graph_code.write_text(f"{'a' * 64}  {shared_source}\n", encoding="utf-8")
    eager_code.write_text(
        f"{'a' * 64}  {shared_source}\n"
        f"{'b' * 64}  {matched_wrapper}\n"
        f"{'c' * 64}  {ROOT / 'analysis_eval/summarize_matched_eager_controls.py'}\n",
        encoding="utf-8",
    )
    graph_pass = tmp_path / "graph.pass"
    eager_pass = tmp_path / "eager.pass"
    graph_pass.write_text(
        "status=PASS\ncompleted_arms=21\ncompleted_epochs=42\n"
        f"summary_json_sha256={MODULE.sha256(graph_summary)}\n",
        encoding="utf-8",
    )
    eager_pass.write_text(
        "status=PASS\ncompleted_arms=9\ncompleted_epochs=9\n"
        f"summary_json_sha256={MODULE.sha256(eager_summary)}\n",
        encoding="utf-8",
    )
    output = tmp_path / "output"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(MODULE_PATH),
            "--graph-summary", str(graph_summary),
            "--eager-summary", str(eager_summary),
            "--graph-protocol", str(graph_protocol_path),
            "--eager-protocol", str(eager_protocol_path),
            "--graph-code-manifest", str(graph_code),
            "--eager-code-manifest", str(eager_code),
            "--graph-pass", str(graph_pass),
            "--eager-pass", str(eager_pass),
            "--output-dir", str(output),
        ],
    )
    MODULE.main()
    payload = json.loads(
        (output / "full_graph_vs_matched_eager.json").read_text(encoding="utf-8")
    )
    assert payload["status"] == "PASS"
    assert len(payload["paired_by_seed"]) == 9
    assert len(payload["aggregate"]) == 3
    assert (output / "full_graph_vs_matched_eager.md").is_file()
