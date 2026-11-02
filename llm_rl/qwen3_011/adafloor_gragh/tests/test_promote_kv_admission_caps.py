from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


TOOL = Path(__file__).parents[1] / "tools" / "promote_kv_admission_caps.py"
SPEC = importlib.util.spec_from_file_location("promote_kv_admission_caps", TOOL)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _write(path: Path, cap: int) -> None:
    path.write_text(
        json.dumps(
            [
                {
                    "step": 1,
                    "selected_floor": 16,
                    "kv_cap": cap,
                    "rank_adjusted_peak_loads": {"0": 100.0},
                }
            ]
        ),
        encoding="utf-8",
    )


def test_separates_admission_and_physical_caps(tmp_path: Path) -> None:
    plan = tmp_path / "plan.json"
    summary = tmp_path / "summary.json"
    _write(plan, 256)
    _write(summary, 256)
    MODULE.promote_files(plan, summary, "16:512")
    step = json.loads(plan.read_text(encoding="utf-8"))[0]
    assert step["kv_admission_cap"] == 256
    assert step["kv_cap"] == 512
    assert step["kv_admission_headroom_tokens"] == 156
    assert step["kv_physical_headroom_tokens"] == 412


def test_rejects_admission_above_physical_cap(tmp_path: Path) -> None:
    plan = tmp_path / "plan.json"
    summary = tmp_path / "summary.json"
    _write(plan, 512)
    _write(summary, 512)
    with pytest.raises(ValueError, match="admission cap"):
        MODULE.promote_files(plan, summary, "16:256")
