from pathlib import Path
import json
import math
import re
import subprocess


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "run_qwen3_paper_graph_main_from_common_epoch0.sh"
MATCHED_EAGER_RUNNER = (
    ROOT / "run_qwen3_paper_matched_eager_tq1_from_common_epoch0.sh"
)
MATCHED_EAGER_RESUME = (
    ROOT / "resume_qwen3_paper_matched_eager_after_floor2_headroom_fix.sh"
)
PAIRED_RUNNER = ROOT / "run_qwen3_paper_full_graph_vs_eager_paired.sh"
FAIR_RUNNER = ROOT / "run_paper_fair_epoch1_2_from_common_epoch0.sh"
FLOOR2_RUNNER = ROOT / "run_mode1_local_length_sorted_e2e_adaptive_floor2.sh"
FLOOR4_RUNNER = ROOT / "run_mode1_local_length_sorted_e2e_adaptive_floor4.sh"


def text() -> str:
    return RUNNER.read_text(encoding="utf-8")


def test_shell_syntax() -> None:
    subprocess.run(["bash", "-n", str(RUNNER)], check=True)
    subprocess.run(["bash", "-n", str(MATCHED_EAGER_RUNNER)], check=True)
    subprocess.run(["bash", "-n", str(MATCHED_EAGER_RESUME)], check=True)
    subprocess.run(["bash", "-n", str(PAIRED_RUNNER)], check=True)


def test_matches_eager_main_matrix() -> None:
    source = text()
    assert "SEEDS_CSV=${GRAPH_MAIN_SEEDS_CSV:-0,2,3}" in source
    assert "IFS=',' read -r -a SEEDS" in source
    assert (
        "POLICIES=(vanilla lengthsort lengthsort_guard minskew "
        "natural_f4 natural_f2 planned_f4)"
    ) in source
    assert 'FAIR_START_EPOCH=1' in source
    assert 'FORMAL_TOTAL_EPOCHS=3' in source
    assert 'FORMAL_TOTAL_EPOCHS=2' in source
    assert 'FAIR_TRAIN_STEPS=$smoke_steps' in source
    assert 'FAIR_FREEZE_ACTOR=0' in source
    assert 'actor_rollout_ref.rollout.seed="$seed"' in source
    assert 'actor_rollout_ref.actor.optim.lr=1e-6' in source
    assert 'PLANNED_ARMS=21' in source
    assert 'PLANNED_EPOCHS=42' in source
    assert "excluded_policy=fixed4" in source
    assert "fixed4_exclusion_reason=not_a_reasonable_comparison_strategy" in source
    assert "printf 'planned_arms=%s" in source
    assert "printf 'planned_epochs=%s" in source
    assert "fixed4)" not in source


def test_single_arm_execution_preserves_the_complete_protocol() -> None:
    source = text()
    assert "GRAPH_MAIN_ONLY_ARM" in source
    assert "must be a canonical seed:policy pair" in source
    assert 'if [[ -n "$ONLY_ARM" && "$ONLY_ARM" != "$seed:$policy" ]]' in source
    assert "paired schedule arm complete" in source


def test_paired_runner_counterbalances_graph_and_eager() -> None:
    source = PAIRED_RUNNER.read_text(encoding="utf-8")
    assert "seed0_graph_eager,seed2_eager_graph,seed3_graph_eager" in source
    assert "order=(eager_matched graph)" in source
    assert "order=(graph eager_matched)" in source
    assert "GRAPH_MAIN_REUSE_SEED0_VANILLA=1" in source
    assert 'GRAPH_MAIN_ONLY_ARM="$seed:$policy"' in source
    assert "full_graph_vs_matched_eager.json" in source
    assert "MATCHED_EAGER_POLICIES=(vanilla natural_f2 planned_f4)" in source
    assert "matched_eager_epochs=1" in source
    assert "matched_eager_arms=9" in source


def test_full_decode_graph_contract() -> None:
    source = text()
    required = (
        "ADAFLOOR_ACLGRAPH_MODE=FULL_DECODE_ONLY",
        "ADAFLOOR_GRAPH_CAPTURE_SIZES=$CAPTURE_SIZES",
        "VLLM_ENABLE_GRAPH_MODE=0",
        "TASK_QUEUE_ENABLE=1",
        "ROLLOUT_ENFORCE_EAGER=False",
        "VLLM_ASCEND_ELASTIC_ACLGRAPH_CAPTURE_ATTENTION=1",
        "VLLM_ASCEND_ELASTIC_ACLGRAPH_CAPTURE_MOE=1",
        "VLLM_ROLLOUT_NATIVE_SLEEP_MODE=0",
        "VLLM_ROLLOUT_ASYNC_SCHEDULING=false",
        "actor_rollout_ref.rollout.cudagraph_mode=FULL_DECODE_ONLY",
    )
    for item in required:
        assert item in source
    assert "DEFAULT_CAPTURE_SIZES='[16,32]'" in source
    assert "CAPTURE_SIZES=${GRAPH_MAIN_CAPTURE_SIZES:-$DEFAULT_CAPTURE_SIZES}" in source


def test_core3_profile_preserves_two_updated_policy_epochs() -> None:
    source = text()
    assert "POLICY_PROFILE=${GRAPH_MAIN_POLICY_PROFILE:-full}" in source
    assert "POLICIES=(vanilla natural_f2 planned_f4)" in source
    assert "PLANNED_ARMS=9" in source
    assert "PLANNED_EPOCHS=18" in source
    assert "FORMAL_EPOCHS=(1 2)" in source
    assert "GRAPH_MAIN_POLICY_PROFILE must be full or core3" in source
    assert "capture_plan_sha256" in source
    assert "PROTOCOL_WRAPPER" in source


def test_formula_driven_graph_kv_reserve() -> None:
    source = text()
    assert "KV_BLOCK_SIZE=128" in source
    assert "KV_BYTES_PER_TOKEN=98304" in source
    assert "GRAPH_RESERVE_BYTES=$PROFILE_DERIVED_RESERVE_BYTES" in source
    assert "GRAPH_MAIN_GRAPH_RESERVE_BYTES must be a positive integer" in source
    assert "qwen3_aclgraph_step_bucket_profile_all32.json" in source
    assert "GRAPH_RESERVE_MINIMUM_BYTES=2684354560" in source
    assert "GRAPH_RESERVE_QUANTUM_BYTES=536870912" in source
    assert "profile_capture_memory_gib_rank_max" in source
    assert "GRAPH_RESERVE_BLOCKS=$(((GRAPH_RESERVE_BYTES + KV_BLOCK_BYTES - 1) / KV_BLOCK_BYTES))" in source
    assert "GRAPH_RESERVE_TOKENS=$((GRAPH_RESERVE_BLOCKS * KV_BLOCK_SIZE))" in source
    block_bytes = 128 * 98304
    reserve_bytes = 2684354560
    reserve_blocks = (reserve_bytes + block_bytes - 1) // block_bytes
    assert reserve_blocks == 214
    assert reserve_blocks * 128 == 27392
    expected = {
        "GRAPH_FULL16": 353408,
        "GRAPH_F2": 103680,
        "GRAPH_F4": 253184,
        "GRAPH_F8_NATURAL": 288256,
        "GRAPH_F8_F4": 349952,
    }
    eager = {
        "GRAPH_FULL16": 380800,
        "GRAPH_F2": 131072,
        "GRAPH_F4": 280576,
        "GRAPH_F8_NATURAL": 315648,
        "GRAPH_F8_F4": 377344,
    }
    assert {key: value - 27392 for key, value in eager.items()} == expected
    assert "profile_rank_max_capture_rounded_up_0p5g_with_2p5g_minimum" in source
    assert "KV_ADMISSION_RESERVE_TOKENS=0" in source
    assert '"KV_ADMISSION_REQUIRE_PROMPT_TOKENS=1"' in source
    assert '"KV_ADMISSION_BLOCK_SIZE_TOKENS=$KV_BLOCK_SIZE"' in source
    assert '"PLANNER_TOKENIZER_PATH=/data/Qwen3-30B-A3B"' in source
    assert "kv_admission_formula=block_aware_prompt_plus_predicted_response_peak" in source


def test_prompt_inclusive_kv_admission_is_fail_closed_for_both_adafloor_paths() -> None:
    for runner in (FLOOR2_RUNNER, FLOOR4_RUNNER):
        source = runner.read_text(encoding="utf-8")
        assert 'KV_ADMISSION_REQUIRE_PROMPT_TOKENS="${KV_ADMISSION_REQUIRE_PROMPT_TOKENS:-0}"' in source
        assert 'planner_extra_args+=(--tokenizer-path "$PLANNER_TOKENIZER_PATH")' in source or \
            'PLAN_EXTRA_ARGS+=(--tokenizer-path "$PLANNER_TOKENIZER_PATH")' in source
        assert "--require-prompt-token-lengths" in source
        assert "prompt-inclusive KV admission requires PLANNER_TOKENIZER_PATH" in source
        assert '--kv-block-size-tokens "$KV_ADMISSION_BLOCK_SIZE_TOKENS"' in source


def test_profile_driven_reserve_schedule() -> None:
    profile = json.loads(
        (ROOT / "analysis_eval/qwen3_aclgraph_step_bucket_profile_all32.json")
        .read_text(encoding="utf-8")
    )
    measured = {
        int(arm["graph_count"]): float(arm["capture_memory_gib_rank_max"])
        for arm in profile["arms"]
    }
    assert set(measured) == set(range(1, 33))

    def reserve_gib(count: int) -> float:
        return max(2.5, math.ceil(measured[count] * 2) / 2)

    assert reserve_gib(1) == 2.5
    assert reserve_gib(2) == 2.5
    assert reserve_gib(8) == 2.5
    assert reserve_gib(10) == 2.5
    assert reserve_gib(16) == 3.0
    assert reserve_gib(32) == 6.0


def test_planned_headroom_is_separate() -> None:
    source = text()
    assert "planned_headroom_accounting=separate_after_graph_reserve" in source
    assert "planned_floor4_effective_tokens=%s" in source
    assert "$((GRAPH_F4 - 147456))" in source
    assert "$((GRAPH_F8_F4 - 114688))" in source
    assert "env -u FLOOR_KV_CAPS" in source


def test_paper_artifact_gates() -> None:
    source = text()
    assert "code_sha256.txt" in source
    assert "protocol.env" in source
    assert "run_manifest.tsv" in source
    assert "analysis_manifest.tsv" in source
    assert "summarize_existing_e2e_repetitions.py" in source
    for marker in (
        "FULL_DECODE_ONLY compilation enabled on NPU",
        "Native FULL_DECODE_ONLY ACLGraph enabled",
        "FULL_DECODE_ONLY ACLGraph captured Attention KV write and paged read",
        "Graph capturing finished",
        "Replaying aclgraph",
    ):
        assert marker in source
    assert re.search(r"actor/lr:1e-0\*6", source)


def test_preemption_smoke_uses_an_isolated_low_kv_budget() -> None:
    source = text()
    assert "SMOKE_KV_TOKENS=${GRAPH_MAIN_SMOKE_KV_TOKENS:-196608}" in source
    assert 'if [[ "$ACTION" == smoke ]]; then' in source
    assert '"VANILLA_KV_TOKENS_PER_RANK=$SMOKE_KV_TOKENS"' in source
    assert "non_authoritative_smoke_kv_tokens" in source
    assert "local execution_scope=authoritative" in source
    assert "execution_scope=smoke" in source
    assert "printf 'execution_scope=%s\\n'" in source
    assert "execution_action=%s" not in source


def test_preemption_recovery_adds_no_collective() -> None:
    source = text()
    assert "preemption_safe_full_decode_recovery=true" in source
    assert "preemption_recovery_signal=existing_dp_metadata_sync" in source
    assert "additional_preemption_recovery_collectives=0" in source
    assert "preemption_iteration_runtime_mode=eager" in source
    assert "next_pure_decode_iteration_runtime_mode=full_decode_only" in source
    assert (
        "qwen3_paper_graph_main_full_decode_prompt_kv_admission_tq1_buckets16_32_v3"
        in source
    )


def test_completed_seed0_vanilla_is_imported_fail_closed() -> None:
    source = text()
    assert "reuse_seed0_vanilla_default=1" in source
    assert "reuse_seed0_vanilla_default=0" in source
    assert "GRAPH_MAIN_REUSE_SEED0_VANILLA:-$reuse_seed0_vanilla_default" in source
    assert "REUSED_SEED0_VANILLA_ROOT" in source
    assert "EXPECTED_REUSED_CODE_MANIFEST_SHA256" in source
    assert "EXPECTED_REUSED_PROTOCOL_SHA256" in source
    assert "EXPECTED_REUSED_RUN_MANIFEST_SHA256" in source
    assert "EXPECTED_REUSED_CLEANUP_SHA256" in source
    assert "verify_reused_seed0_vanilla" in source
    assert "unexpected code drift while reusing seed-0 Vanilla" in source
    assert '"$FAIR_RUNNER"' in source
    assert "reused Vanilla unexpectedly contains preemption" in source
    assert "reused_arms.tsv" in source
    assert "imported validated seed=0 policy=vanilla without rerun" in source


def test_matched_eager_contract_is_minimal_and_differs_only_in_execution() -> None:
    source = text()
    wrapper = MATCHED_EAGER_RUNNER.read_text(encoding="utf-8")
    assert "GRAPH_MAIN_EXECUTION_MODE=eager_matched" in wrapper
    assert "GRAPH_MAIN_REUSE_SEED0_VANILLA=0" in wrapper
    assert "qwen3_paper_matched_eager_tq1_prompt_kv_admission_v4_minimal_controls" in source
    assert "POLICIES=(vanilla natural_f2 planned_f4)" in source
    assert "FORMAL_EPOCHS=(1)" in source
    assert "PLANNED_ARMS=9" in source
    assert "PLANNED_EPOCHS=9" in source
    assert '"TASK_QUEUE_ENABLE=1"' in source
    assert '"ROLLOUT_ENFORCE_EAGER=True"' in source
    assert '"VLLM_ASCEND_ELASTIC_ACLGRAPH=0"' in source
    assert '"VLLM_ASCEND_ELASTIC_ACLGRAPH_CAPTURE_ATTENTION=0"' in source
    assert '"VLLM_ASCEND_ELASTIC_ACLGRAPH_CAPTURE_MOE=0"' in source
    assert '"VLLM_ROLLOUT_REUSE_ACLGRAPH_AFTER_WEIGHT_UPDATE=0"' in source
    assert '"VLLM_ROLLOUT_ENABLE_PREFIX_CACHING=true"' in source
    assert '"VLLM_ROLLOUT_ENABLE_CHUNKED_PREFILL=true"' in source
    assert '"VLLM_ASCEND_USE_FILTERED_CUSTOM_OPP=1"' in source
    assert '"VLLM_ENABLE_MC2=1"' in source
    assert "ACTIVE_FULL16=$GRAPH_FULL16" in source
    assert "ACTIVE_F2=$GRAPH_F2" in source
    assert "ACTIVE_F4=$GRAPH_F4" in source
    assert "ACTIVE_F8_NATURAL=$GRAPH_F8_NATURAL" in source
    assert "ACTIVE_F8_F4=$GRAPH_F8_F4" in source
    assert '"VANILLA_KV_TOKENS_PER_RANK=$ACTIVE_FULL16"' in source
    assert '"VLLM_ASCEND_MODE1_PARITY_MAX_KV_TOKENS_FLOOR2=$ACTIVE_F2"' in source
    assert "comparison_difference=aclgraph_execution_only" in source


def test_matched_eager_resume_fixes_only_unreachable_planned_floor2() -> None:
    source = MATCHED_EAGER_RESUME.read_text(encoding="utf-8")
    assert "PLANNED_F4_FLOOR2_HEADROOM_CORRECTION.env" in source
    assert "planned_f4_reachable_floors_are_4_8_16" in source
    assert "reachable_floor_caps_changed=false" in source
    assert "graph_eager_pairing_changed=false" in source
    assert (
        "VLLM_ASCEND_MODE1_PARITY_PLANNED_FLOOR_GROUP_KV_HEADROOM_TOKENS_FLOOR2=0"
        in source
    )
    assert "sha256sum -c \"$CODE_MANIFEST\"" in source
    assert "refusing to remove nonempty failed arm" in source
    assert 'grep -Fqx -- "$expected_planned" "$RUN_MANIFEST"' in source
    assert "rmdir \"$FAILED_ARM_ROOT\"" in source


def test_matched_eager_verifier_rejects_any_graph_execution() -> None:
    source = text()
    for marker in (
        "matched eager runtime evidence missing",
        "matched eager prefix-cache evidence missing",
        "matched eager TQ1 evidence missing",
        "matched eager graph-disable evidence missing",
        "matched eager arm unexpectedly used ACLGraph",
    ):
        assert marker in source


def test_completed_arms_are_disk_safe() -> None:
    source = text()
    assert "MIN_FREE_GIB=${GRAPH_MAIN_MIN_FREE_GIB:-120}" in source
    assert "CHECKPOINTS_REMOVED_AFTER_VALIDATION.txt" in source
    assert "validated arm still contains checkpoint directories" in source
    assert "cleanup_transient_dirs" in source
    assert 'actual_cache" == "$common_real' in source
    assert "refusing unsafe cache cleanup" in source
    assert "disk_cleanup.tsv" in source
    assert "require_disk_headroom" in source


def test_completed_execution_is_finalized_without_npu_rerun() -> None:
    source = text()
    assert "arm_execution_complete" in source
    assert "finalize_existing_arm" in source
    assert '"FAIR_FINALIZE_EXISTING=1"' in source
    assert "finalizing completed execution without NPU rerun" in source
    assert "verify_full_decode_preemption_recovery.py" in source


def test_fair_runner_requires_verified_full_decode_recovery() -> None:
    source = FAIR_RUNNER.read_text(encoding="utf-8")
    assert "FULL_DECODE_PREEMPTION_VERIFIER" in source
    assert "adafloor_*) return 0" in source
    assert "lengthsort_guard|fixed4|minskew|adafloor_*" not in source
    assert 'ADAFLOOR_ACLGRAPH_MODE:-}" == "FULL_DECODE_ONLY"' in source
    assert "FULL_DECODE_PREEMPTION_RECOVERY_AUDIT.json" in source
    assert "without a complete synchronized FULL recovery audit" in source
    assert "despite safe runtime recovery" in source
