from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "run_qwen3_paper_graph_core3_k2_matched_2p5g_seed0_from_common_epoch0.sh"


def test_runner_is_valid_bash_and_locks_matched_two_graph_budget():
    subprocess.run(["bash", "-n", str(RUNNER)], check=True)
    source = RUNNER.read_text(encoding="utf-8")

    assert "run_protocol fixed_16_32 0 '[16,32]'" in source
    assert "run_protocol minpadding_k2 2 '[32]'" in source
    assert "GRAPH_MAIN_REUSE_SEED0_VANILLA=0" in source
    assert "GRAPH_MAIN_POLICY_PROFILE=core3" in source
    assert "GRAPH_MAIN_SEEDS_CSV=0" in source


def test_runner_uses_distinct_roots_and_reserve_provenance():
    source = RUNNER.read_text(encoding="utf-8")

    assert 'GRAPH_MAIN_ROOT="$RESULT_ROOT/$label"' in source
    assert "GRAPH_MAIN_GRAPH_RESERVE_BYTES" not in source
    assert "GRAPH_MAIN_GRAPH_RESERVE_BASIS" not in source
    assert "qwen3_paper_graph_core3_k2_matched_2p5g_seed0_tq1_v1" in source
