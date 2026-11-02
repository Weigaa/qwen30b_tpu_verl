import os
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[1] / "run_qwen3_v014_sleep2_multistep_validation.sh"


@pytest.mark.parametrize("mode,tq", [("eager", 2), ("graph", 1)])
def test_dry_run_preserves_native_sleep_contract(mode, tq):
    result = subprocess.run(["bash", str(SCRIPT), mode, "dry-run"],
                            text=True, capture_output=True, check=True)
    assert f"mode={mode} tq={tq} steps=2 MC2_min_EP=2" in result.stdout
    assert "native_sleep=2 lifecycle=1 KV_first=1" in result.stdout
    assert "hccl_buffer_mib=640" in result.stdout
    source = SCRIPT.read_text()
    assert "VLLM_ENABLE_GRAPH_MODE=0" in source
    assert "VLLM_ROLLOUT_MANUAL_FREE_CACHE_ENGINE=0" in source
    assert "VLLM_ASCEND_ELASTIC_ACLGRAPH_DYNAMIC_DISABLE=0" in source
    assert "ADAFLOOR_ACLGRAPH_MODE=FULL_DECODE_ONLY" in source
    assert "FAIR_FREEZE_ACTOR=0" in source


@pytest.mark.parametrize("steps", ["0", "6", "bad"])
def test_rejects_unbounded_or_invalid_steps(steps):
    result = subprocess.run(["bash", str(SCRIPT), "eager", "dry-run"],
                            env={**os.environ, "SLEEP2_VALIDATION_STEPS": steps},
                            text=True, capture_output=True)
    assert result.returncode == 2
