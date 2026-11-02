from __future__ import annotations

import ast
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WRAPPER = ROOT / "run_qwen3_vanilla_epoch0_full_decode_fia_v014opt_tq1_buckets16_32.sh"
COMPARE_TOOL = ROOT / "tools" / "compare_qwen3_full_decode_buckets.py"


def test_bucket_wrapper_is_fixed_to_16_32() -> None:
    text = WRAPPER.read_text()
    assert "export FULL_DECODE_CAPTURE_SIZES='[16,32]'" in text
    assert "FULL_DECODE_EPOCH0_ROOT" in text
    assert "FULL_DECODE_BUCKET_BASELINE_ROOT" in text
    assert 'BASE_RUNNER="$SCRIPT_DIR/run_qwen3_vanilla_epoch0_full_decode_fia_tq1.sh"' in text
    assert "FULL_DECODE_OPTIMIZATION_PROFILE=v014_runtime_port" in text
    assert "VLLM_ROLLOUT_ASYNC_SCHEDULING=false" in text
    assert "VLLM_ASCEND_USE_FILTERED_CUSTOM_OPP=1" in text
    subprocess.run(["bash", "-n", str(WRAPPER)], check=True)


def test_comparator_has_pairing_and_work_guards() -> None:
    tree = ast.parse(COMPARE_TOOL.read_text())
    function_names = {
        node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
    }
    assert {"_read_workload", "_eligible_rounds", "_assert_capture_evidence", "_initial_capture_stats", "compare"} <= function_names
    assert "_read_code_manifest" in function_names
    text = COMPARE_TOOL.read_text()
    assert 'baseline_protocol.get("capture_sizes") != "[32]"' in text
    assert 'candidate_protocol.get("capture_sizes") != "[16,32]"' in text
    assert "request identity multiset differs" in text
    assert "generated_work_percent" in text
    assert "bucket16_eligible_fraction" in text
    assert "shared_source_files_with_identical_hashes" in text
    assert "initial_graph_capture_memory_gib_mean_absolute" in text
