from __future__ import annotations

import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).parents[1]
WRAPPER = ROOT / "run_deepseek_v2_lite_planned_floor2_threshold_two_step_smoke.sh"
SMOKE = ROOT / "run_deepseek_v2_lite_adafloor_smoke.sh"


def _run_wrapper(env_updates: dict[str, str]) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env.update(env_updates)
    return subprocess.run(
        ["bash", str(WRAPPER)],
        cwd=ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )


def test_planned_floor2_threshold_wrapper_pins_the_exact_gate() -> None:
    source = WRAPPER.read_text(encoding="utf-8")

    assert "does not accept overrides" in source
    assert "deepseek_v2_lite_planned_f2_runtime_profile.sh" in source
    assert "DEEPSEEK_ADAFLOOR_SMOKE_POLICY=planned" in source
    assert "DEEPSEEK_ADAFLOOR_SMOKE_FLOOR=2" in source
    assert "run_deepseek_v2_lite_floor2_threshold_two_step_smoke.sh" in source
    assert "DEEPSEEK_P_F2_TRAINING_MIN_FREE_MIB" in source
    assert "ALLOW_INFEASIBLE_PLAN is forbidden" in source
    assert "BASELINE_ALLOW_INFEASIBLE_PLAN=0" in source


def test_planned_floor2_threshold_wrapper_requires_measured_guard() -> None:
    env = os.environ.copy()
    env.pop("ALLOW_INFEASIBLE_PLAN", None)
    env.pop("DEEPSEEK_P_F2_TRAINING_MIN_FREE_MIB", None)
    result = subprocess.run(
        ["bash", str(WRAPPER)],
        cwd=ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "must be an explicitly measured positive integer" in result.stderr


def test_planned_floor2_threshold_wrapper_rejects_infeasible_escape_hatch() -> None:
    result = _run_wrapper(
        {
            "ALLOW_INFEASIBLE_PLAN": "1",
            "DEEPSEEK_P_F2_TRAINING_MIN_FREE_MIB": "4096",
        }
    )

    assert result.returncode == 2
    assert "ALLOW_INFEASIBLE_PLAN is forbidden" in result.stderr


def test_shared_smoke_validates_planned_runtime_evidence() -> None:
    source = SMOKE.read_text(encoding="utf-8")

    assert "DYNAMIC_SHRINK_POLICY=$SHRINK_POLICY" in source
    assert 'EPOCH_DIR="$RUN_ROOT/epoch_001_mode1_${SHRINK_POLICY}"' in source
    assert "Mode1 planned floor groups precreated before KV sizing:" in source
    assert "Mode1 training memory guard: rank=0" in source
    assert "Mode1 training-boundary full-world transient cleanup:" in source
    assert '"ALLOW_INFEASIBLE_PLAN=0"' in source


def test_planned_floor2_threshold_shell_is_valid() -> None:
    for path in (WRAPPER, SMOKE):
        result = subprocess.run(
            ["bash", "-n", str(path)],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
