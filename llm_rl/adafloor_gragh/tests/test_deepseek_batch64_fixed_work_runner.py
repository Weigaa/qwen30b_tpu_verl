from pathlib import Path


ROOT = Path(__file__).parents[1]
RUNNER = ROOT / "run_deepseek_v2_lite_batch64_fixed_work.sh"
MODE1_RUNNER = ROOT / "run_mode1_local_length_sorted_e2e_adaptive_floor4.sh"


def test_fixed_runner_uses_lengthsort_for_control_in_run_and_dry_run() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    assert source.count("variant=lengthsort") == 3
    assert "local variant=lengthsort" in source
    assert "deepseek_v2_lite_lengthsort_common_epoch0_epoch1_2" in source


def test_fixed_runner_migrates_cap_provenance_fail_closed() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    assert "tools/migrate_deepseek_fixed_work_cap.py" in source
    assert "--current-execution-sha256 \"$EXECUTION_CODE_SHA256\"" in source
    assert "DEEPSEEK_KV_CAP_AUTHORIZED_RUNTIME_EXECUTION_CODE_SHA256" in source


def test_fixed_runner_uses_immutable_old_natural_source_and_new_hash_root() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    assert "p0_8b_batch64_fixed_work_b0f33dbe107e/natural_epoch" in source
    assert "b0f33dbe107e700fc606c6416065f2ae14ef80fbf4a3f68e7d16f6d38e870ca5" in source
    assert "de30e3fa6f9c72b1a4b4391190f42e194b1ec85c95eed747bb19b0610300a19b" in source
    assert "p0_8b_batch64_fixed_work_${EXECUTION_CODE_SHA256:0:12}" in source
    assert "validate_deepseek_fixed_work_source.py" in source
    assert "deepseek_batch64_fixed_work_replay_v3" in source


def test_fixed_runner_rebuilds_existing_trace_before_accepting_it() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    assert ".fixed_work_trace.rebuilt.XXXXXX" in source
    assert '--output "$rebuilt_trace"' in source
    assert "--force >&2" in source
    assert 'cmp -s "$rebuilt_trace" "$trace"' in source
    assert "existing fixed-work trace differs from source rebuild" in source


def test_fixed_adafloor_plan_is_checked_before_runtime_initialization() -> None:
    source = MODE1_RUNNER.read_text(encoding="utf-8")
    validation = source.index(
        "fixed-work source/executed plan SHA256"
    )
    plan_only_exit = source.index('if [[ "${MODE1_PLAN_ONLY:-0}" == "1" ]]')
    launcher = source.index('"$LAUNCHER"')
    assert validation < plan_only_exit < launcher
    assert "VERL_FIXED_WORK_REPLAY_REQUIRE_PLAN_CAP" in source
    assert "executed_plan_sha256 != source_plan_sha256" in source


def test_immutable_natural_source_is_validated_before_epoch_trace_build() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    function = source[source.index("run_fixed_epoch() {") : source.index("dry_run_gate() {")]
    assert function.index("validate_immutable_natural_source") < function.index(
        "build_trace epoch"
    )
    assert "run_arm vanilla epoch natural" not in source


def test_fixed_runner_builds_trace_from_actual_natural_run_plan() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    assert "resolve_actual_plan" in source
    assert 'epoch_dir/oracle/length_sorted_rank_plan.json' in source
    assert 'build_trace gate "$SOURCE_GATE_ADAFLOOR_RUN" "$actual_plan"' in source
    assert 'build_trace epoch "$natural_adafloor" "$actual_plan"' in source
    assert 'build_trace gate "$SOURCE_GATE_ADAFLOOR_RUN" "$GATE_PREFLIGHT_PLAN"' not in source
    assert 'build_trace epoch "$natural_adafloor" "$EPOCH_PREFLIGHT_PLAN"' not in source


def test_checkpoint_cleanup_is_committed_only_after_pair_verification() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    assert "FAIR_KEEP_COMPLETED_CHECKPOINTS=1" in source
    assert "manage_deepseek_fixed_work_cleanup.py\" prepare" in source
    assert "manage_deepseek_fixed_work_cleanup.py\" commit" in source
    fixed_gate = source.index("verify_fixed_pair gate")
    gate_cleanup = source.index("cleanup_verified_pair gate fixed")
    assert fixed_gate < gate_cleanup
    fixed_epoch = source.index("verify_fixed_pair epoch")
    epoch_cleanup = source.index("cleanup_verified_pair epoch fixed")
    assert fixed_epoch < epoch_cleanup
