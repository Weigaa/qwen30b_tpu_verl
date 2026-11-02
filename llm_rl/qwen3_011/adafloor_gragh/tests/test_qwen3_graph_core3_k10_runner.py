from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "run_qwen3_paper_graph_core3_k10_from_common_epoch0.sh"
MAIN_RUNNER = ROOT / "run_qwen3_paper_graph_main_from_common_epoch0.sh"
BUCKET_HELPER = ROOT / "tools/aclgraph_bucket_plan.py"
VANILLA_BUILDER = ROOT / "tools/build_vanilla_aclgraph_bucket_plan.py"
ROLLOUT_CONFIG = ROOT / "verl/workers/config/rollout.py"
ROLLOUT_YAML = ROOT / "verl/trainer/config/rollout/rollout.yaml"
PIECEWISE_BACKEND = ROOT / "vllm/compilation/cuda_piecewise_backend.py"
ROLLOUT_SPMD = ROOT / "verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py"
FLOOR4_RUNNER = ROOT / "run_mode1_local_length_sorted_e2e_adaptive_floor4.sh"
SHARED_SUMMARIZER = (
    ROOT.parent / "qwen3_shrink_aware/analysis_eval/summarize_existing_e2e_repetitions.py"
)
E2E_PARSER = ROOT.parent / "qwen3_shrink_aware/analysis_eval/plot_e2e_performance.py"


def test_shell_and_python_syntax() -> None:
    subprocess.run(["bash", "-n", str(RUNNER)], check=True)
    subprocess.run(
        ["python3", "-m", "py_compile", str(BUCKET_HELPER), str(VANILLA_BUILDER)],
        check=True,
    )


def test_epoch_planned_core3_contract() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    assert "GRAPH_MAIN_POLICY_PROFILE=core3" in source
    assert "GRAPH_MAIN_REUSE_SEED0_VANILLA=0" in source
    assert "GRAPH_MAIN_CAPTURE_SIZES='[32]'" in source
    assert "GRAPH_MAIN_ACLGRAPH_BUCKET_COUNT=10" in source
    assert "run_qwen3_paper_graph_main_from_common_epoch0.sh" in source
    assert "Vanilla Full16, AdaFloor-N F2, and AdaFloor-P F4" in source
    assert "seeds 0/2/3" in source
    assert "updated-policy epochs 1/2" in source
    assert "qwen3_paper_graph_core3_k10_minpadding_tq1_v2" in source
    assert "each epoch, every" in source
    assert "step gets its own 10-graph set" in source
    assert "smoke" in source


def test_smoke_can_select_one_core3_arm() -> None:
    source = MAIN_RUNNER.read_text(encoding="utf-8")
    assert 'ONLY_ARM" != "0:$smoke_policy' in source
    assert 'failed to select exactly one smoke arm' in source


def test_rollout_config_declares_epoch_bucket_plan_fields() -> None:
    dataclass_source = ROLLOUT_CONFIG.read_text(encoding="utf-8")
    yaml_source = ROLLOUT_YAML.read_text(encoding="utf-8")
    for field in ("aclgraph_bucket_plan_path", "aclgraph_bucket_count"):
        assert field in dataclass_source
        assert field in yaml_source


def test_code_contract_covers_bucket_runtime_fixes() -> None:
    source = MAIN_RUNNER.read_text(encoding="utf-8")
    assert '"$SCRIPT_DIR/verl/workers/config/rollout.py"' in source
    assert '"$SCRIPT_DIR/vllm/compilation/cuda_piecewise_backend.py"' in source


def test_admission_validates_only_selectable_floors() -> None:
    source = FLOOR4_RUNNER.read_text(encoding="utf-8")
    min_floor = source.index('MIN_ADAPTIVE_FLOOR="${MIN_ADAPTIVE_FLOOR:-4}"')
    cap_loop = source.index('for floor_cap in "2:$effective_floor2_cap"')
    assert min_floor < cap_loop
    loop_end = source.index("done", cap_loop)
    loop = source[cap_loop:loop_end]
    assert "if (( floor < MIN_ADAPTIVE_FLOOR )); then" in loop
    assert "continue" in loop
    assert "nonpositive capacity at floor=$floor" in loop


def test_core3_summary_does_not_require_absent_policies() -> None:
    source = SHARED_SUMMARIZER.read_text(encoding="utf-8")
    assert 'if "lengthsort_guard" in policies else []' in source
    assert 'if policy not in policies:' in source
    assert 'Across {len(epoch_rows)} epochs' in source


def test_oom_advisory_is_not_counted_as_an_observed_fault() -> None:
    source = E2E_PARSER.read_text(encoding="utf-8")
    assert 'if "capturing too many batch sizes can lead to OOM" not in line' in source
    assert "fault_text," in source


def test_static_piece_does_not_index_an_absent_symbolic_shape() -> None:
    source = PIECEWISE_BACKEND.read_text(encoding="utf-8")
    guard = source.index("if not self.sym_shape_indices:")
    dispatch = source.index("runtime_shape = args[self.sym_shape_indices[0]]")
    assert guard < dispatch
    assert "self.to_be_compiled_sizes.clear()" in source[guard:dispatch]
    assert "return self.compiled_graph_for_general_shape(*args)" in source[guard:dispatch]


def test_step_buckets_remain_ascending_until_capture_reverses_them() -> None:
    rollout_source = ROLLOUT_SPMD.read_text(encoding="utf-8")
    runner_source = (ROOT / "vllm_ascend/worker/model_runner_v1.py").read_text(
        encoding="utf-8"
    )
    assert "model_runner.aclgraph_batch_sizes = normalized" in rollout_source
    assert "model_runner.aclgraph_batch_sizes = list(reversed(normalized))" not in rollout_source
    assert "compilation_cases_decode = list(\n                    reversed(decode_cudagraph_batch_sizes))" in runner_source


def test_exact_dp_selects_ten_sorted_buckets_per_step() -> None:
    sys.path.insert(0, str(ROOT / "tools"))
    from aclgraph_bucket_plan import optimal_capture_sizes

    sizes, padding = optimal_capture_sizes({1: 100, 8: 80, 16: 50, 32: 20}, 10)
    assert len(sizes) == 10
    assert sizes == sorted(set(sizes))
    assert sizes[-1] == 32
    assert padding >= 0
