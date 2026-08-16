from __future__ import annotations

import runpy
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).parents[1]
TOOL = ROOT / "tools" / "verify_deepseek_n_f2_kv_cap_run.py"
WRAPPER = ROOT / "run_deepseek_v2_lite_natural_f2_kv_cap_validation.sh"


def test_compatibility_cli_defaults_to_natural_floor2(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tools = str(ROOT / "tools")
    monkeypatch.syspath_prepend(tools)
    monkeypatch.setattr(sys, "argv", [str(TOOL), "--help"])
    with pytest.raises(SystemExit) as exc:
        runpy.run_path(str(TOOL), run_name="__main__")
    assert exc.value.code == 0
    assert sys.argv[1:3] == ["--lifecycle", "natural_f2"]


def test_shell_wrapper_delegates_to_generic_authorizer() -> None:
    source = WRAPPER.read_text(encoding="utf-8")
    assert 'run_deepseek_v2_lite_kv_cap_validation.sh" natural_f2' in source
    assert "ALLOW_INFEASIBLE_PLAN" not in source


def test_floor2_compatibility_entry_points_are_execution_hashed() -> None:
    source = (ROOT / "tools" / "hash_deepseek_execution_code.py").read_text(
        encoding="utf-8"
    )
    required = (
        "run_deepseek_v2_lite_natural_f2_calibration.sh",
        "run_deepseek_v2_lite_natural_f2_kv_cap_validation.sh",
        "tools/verify_deepseek_n_f2_kv_cap_run.py",
    )
    for relative in required:
        assert f'"{relative}"' in source
