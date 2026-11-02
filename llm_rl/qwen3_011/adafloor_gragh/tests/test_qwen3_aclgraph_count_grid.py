import json
import subprocess
from pathlib import Path

from analysis_eval.derive_aclgraph_reserve import derive


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "run_qwen3_aclgraph_count_grid_profile_derived.sh"
PROFILE = ROOT / "analysis_eval/qwen3_aclgraph_step_bucket_profile_all32.json"


def test_grid_runner_contract() -> None:
    subprocess.run(["bash", "-n", str(RUNNER)], check=True)
    source = RUNNER.read_text(encoding="utf-8")
    assert "COUNTS=1,2,4,8,16,32" in source
    assert "LAUNCH_ORDER=1,32,2,16,4,8" in source
    assert 'ACLGRAPH_BUCKET_PROFILE_COUNTS="$COUNTS"' in source
    assert 'ACLGRAPH_BUCKET_PROFILE_LAUNCH_ORDER="$LAUNCH_ORDER"' in source
    assert "ACLGRAPH_BUCKET_PROFILE_RESERVE_MODE=profile_derived" in source
    assert "ACLGRAPH_BUCKET_PROFILE_SOURCE_STEP=4" in source
    assert "bootstrap_arm_protocols" in source
    assert "Materialize those small contracts up front" in source
    assert 'cmp -s "$candidate" "$arm_dir/arm_protocol.env"' in source


def test_profile_derived_grid_reserves() -> None:
    expected_gib = {1: 2.5, 2: 2.5, 4: 2.5, 8: 2.5, 16: 3.0, 32: 6.0}
    for count, expected in expected_gib.items():
        result = derive(PROFILE, count, 2684354560, 536870912)
        assert result["reserve_bytes"] == int(expected * 1024**3)


def test_memory_profile_is_complete() -> None:
    payload = json.loads(PROFILE.read_text(encoding="utf-8"))
    assert {int(arm["graph_count"]) for arm in payload["arms"]} == set(
        range(1, 33)
    )
