from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / 'run_qwen3_paper_graph_core3_k2_from_common_epoch0.sh'


def test_k2_rebuttal_wrapper_is_two_bucket_old_admission():
    source = SCRIPT.read_text()
    assert "GRAPH_MAIN_CAPTURE_SIZES='[16,32]'" in source
    assert 'GRAPH_MAIN_ACLGRAPH_BUCKET_COUNT=2' in source
    assert 'GRAPH_MAIN_KV_ADMISSION_MODEL=synchronized' in source
    assert 'GRAPH_MAIN_GRAPH_RESERVE_BYTES=4294967296' in source
    assert 'GRAPH_MAIN_REUSE_SEED0_VANILLA=0' in source
    subprocess.run(['bash', '-n', str(SCRIPT)], check=True)


def test_main_runner_allows_admission_mode_selection():
    source = (ROOT / 'run_qwen3_paper_graph_main_from_common_epoch0.sh').read_text()
    assert 'GRAPH_MAIN_KV_ADMISSION_MODEL' in source
    assert 'KV_ADMISSION_MODEL' in source
    assert 'ACTIVE_PEAK_SAFETY_FACTOR=$([[ "$KV_ADMISSION_MODEL" == synchronized ]]' in source
