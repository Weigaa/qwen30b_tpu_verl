from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "run_qwen3_paper_graph_core3_k2_seed0_from_common_epoch0.sh"
MAIN_RUNNER = ROOT / "run_qwen3_paper_graph_main_from_common_epoch0.sh"
SUMMARIZER = (
    ROOT.parent
    / "qwen3_shrink_aware/analysis_eval/summarize_existing_e2e_repetitions.py"
)


def test_shell_syntax_and_minimal_protocol() -> None:
    subprocess.run(["bash", "-n", str(RUNNER)], check=True)
    source = RUNNER.read_text(encoding="utf-8")
    assert "GRAPH_MAIN_POLICY_PROFILE=core3" in source
    assert "GRAPH_MAIN_SEEDS_CSV=0" in source
    assert "GRAPH_MAIN_ACLGRAPH_BUCKET_COUNT=2" in source
    assert "GRAPH_MAIN_GRAPH_RESERVE_BASIS=matched_k10_4gib_for_bucket_count_isolation" in source
    assert "GRAPH_MAIN_GRAPH_RESERVE_BYTES=4294967296" in source
    assert "GRAPH_MAIN_CAPTURE_SIZES='[32]'" in source
    assert "GRAPH_MAIN_REUSE_SEED0_VANILLA=0" in source
    assert "same common" in source
    assert "4-GiB graph reserve" in source


def test_main_runner_derives_schedule_size_from_seed_subset() -> None:
    source = MAIN_RUNNER.read_text(encoding="utf-8")
    assert "SEEDS_CSV=${GRAPH_MAIN_SEEDS_CSV:-0,2,3}" in source
    assert "PLANNED_ARMS=$(( ${#SEEDS[@]} * ${#POLICIES[@]} ))" in source
    assert "PLANNED_EPOCHS=$(( PLANNED_ARMS * FORMAL_EPOCH_COUNT ))" in source
    assert "printf 'seeds=%s\\n' \"$SEEDS_CSV\"" in source


def test_single_seed_summary_has_descriptive_zero_spread() -> None:
    source = SUMMARIZER.read_text(encoding="utf-8")
    assert "def sample_stdev(values: list[float]) -> float:" in source
    assert "return stdev(values) if len(values) > 1 else 0.0" in source
    assert '"std": sample_stdev(values)' in source
    assert '"std_difference": sample_stdev(differences)' in source
