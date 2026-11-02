from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "run_qwen3_paper_graph_core3_k8_from_common_epoch0.sh"
MAIN_RUNNER = ROOT / "run_qwen3_paper_graph_main_from_common_epoch0.sh"


def test_shell_syntax() -> None:
    subprocess.run(["bash", "-n", str(RUNNER)], check=True)


def test_paper_grade_k8_core3_contract() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    assert "GRAPH_MAIN_POLICY_PROFILE=core3" in source
    assert "GRAPH_MAIN_REUSE_SEED0_VANILLA=0" in source
    assert "GRAPH_MAIN_CAPTURE_SIZES='[32]'" in source
    assert "GRAPH_MAIN_ACLGRAPH_BUCKET_COUNT=8" in source
    assert "run_qwen3_paper_graph_main_from_common_epoch0.sh" in source
    assert "Vanilla Full16, AdaFloor-N F2, and AdaFloor-P F4" in source
    assert "seeds 0/2/3" in source
    assert "updated-policy epochs 1/2" in source
    assert "qwen3_paper_graph_core3_k8_minpadding_tq1_v1" in source
    assert "step gets its own 8-graph set" in source
    assert "same permanent common epoch0 checkpoint" in source


def test_main_runner_retains_formal_schedule_and_cleanup() -> None:
    source = MAIN_RUNNER.read_text(encoding="utf-8")
    assert "SEEDS_CSV=${GRAPH_MAIN_SEEDS_CSV:-0,2,3}" in source
    assert "FORMAL_EPOCHS=(1 2)" in source
    assert "POLICIES=(vanilla natural_f2 planned_f4)" in source
    assert 'cleanup_transient_dirs "$seed" "$policy"' in source
    assert "common_epoch0_cleanup_forbidden=true" in source
