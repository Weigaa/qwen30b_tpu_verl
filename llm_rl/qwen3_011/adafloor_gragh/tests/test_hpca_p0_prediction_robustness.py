from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.analyze_hpca_prediction_robustness import evaluate_step, main as tool_main


CAPS = {2: 100.0, 4: 200.0, 8: 300.0, 16: 400.0}


def step(selected_floor: int = 2) -> dict:
    return {
        "step": 1,
        "selected_floor": selected_floor,
        "rank_adjusted_peak_loads": {"0": 90.0, "1": 70.0},
        "rank_peak_loads": {"0": 50.0, "1": 40.0},
        "release_area": 123.0,
    }


def test_inflation_one_preserves_nominal_floor() -> None:
    result = evaluate_step(step(), CAPS, 1.0, "global")
    assert result["selected_floor_under_stress"] == 2
    assert result["disposition"] == "safe_current_floor"


def test_stress_escalates_floor_without_calling_it_preemption() -> None:
    result = evaluate_step(step(), {2: 100.0, 4: 120.0, 8: 300.0, 16: 400.0}, 1.4, "global")
    assert result["selected_floor_under_stress"] == 4
    assert result["disposition"] == "safe_floor_escalation"


def test_full16_failure_is_explicit_unsafe_rejection() -> None:
    result = evaluate_step(step(16), {2: 50.0, 4: 60.0, 8: 70.0, 16: 100.0}, 2.0, "hottest_rank")
    assert result["selected_floor_under_stress"] is None
    assert result["disposition"] == "unsafe_workload_rejection"


def test_cli_is_deterministic_and_writes_only_output(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    plan = tmp_path / "plan.json"
    history = tmp_path / "history.json"
    plan.write_text(json.dumps([step()]) + "\n", encoding="utf-8")
    history.write_text(json.dumps({"records": []}) + "\n", encoding="utf-8")
    out = tmp_path / "out"
    monkeypatch.setattr(
        "sys.argv",
        [
            "analyze_hpca_prediction_robustness.py",
            "--case", "fixture", str(plan), str(history),
            "--caps", "2:100,4:200,8:300,16:400",
            "--factors", "1.0,1.4",
            "--output-dir", str(out),
        ],
    )
    tool_main()
    first = (out / "hpca_prediction_robustness.json").read_bytes()
    tool_main()
    assert first == (out / "hpca_prediction_robustness.json").read_bytes()
    assert plan.read_text(encoding="utf-8") == json.dumps([step()]) + "\n"
