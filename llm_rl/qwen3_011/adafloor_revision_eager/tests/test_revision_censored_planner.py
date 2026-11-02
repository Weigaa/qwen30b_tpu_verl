import importlib.util
import json
import sys
from pathlib import Path
import pytest

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from build_mode1_optimized_rank_plan import _read_offline_planning_history
from build_mode1_length_sorted_e2e_plan import _read_ema_baseline_stats, _history_tail_update

spec = importlib.util.spec_from_file_location("builder", ROOT / "tools/build_revision_censored_history.py")
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)


def write_history(path, schema, lengths, previous=None):
    if schema == 1:
        data = {"schema_version": 1, "records": [{"input": "p", "lengths": lengths}]}
    else:
        rows = [{"input": "p", "rollout_generated_tokens": length, "rollout_finish_reason": "length",
                 "rollout_sampling_cap": length} for length in lengths]
        data = builder.build(rows, {"p": previous}, responses_per_prompt=2)
    data.update(steps=1, responses_per_prompt=2)
    path.write_text(json.dumps(data))


def test_schema1_preserves_submitted_behavior(tmp_path, monkeypatch):
    monkeypatch.delenv("ADAFLOOR_REVISION_CENSORED_HISTORY", raising=False)
    path = tmp_path / "history.json"
    write_history(path, 1, [8000, 8000])
    assert _read_offline_planning_history(path, 1, 2)["p"].load == 8000


def test_capped_feedback_cannot_spuriously_lower_prior(tmp_path, monkeypatch):
    monkeypatch.setenv("ADAFLOOR_REVISION_CENSORED_HISTORY", "1")
    monkeypatch.setenv("ADAFLOOR_REVISION_HARD_KV_GUARD", "1")
    p0, p1 = tmp_path / "old.json", tmp_path / "capped.json"
    write_history(p0, 1, [8000, 8000])
    write_history(p1, 2, [4096, 4096], previous=8000)
    old = _read_offline_planning_history(p0, 1, 2)
    capped = _read_offline_planning_history(p1, 1, 2)
    assert capped["p"].censor_flags == (True, True)
    assert _read_ema_baseline_stats([old, capped], .3)["p"].load == 8000
    assert _history_tail_update(8000, capped["p"], .3) == 8000


def test_schema2_needs_optin_and_forecast_independent_kv_guard(tmp_path, monkeypatch):
    path = tmp_path / "capped.json"
    write_history(path, 2, [4096, 4096], previous=8000)
    monkeypatch.delenv("ADAFLOOR_REVISION_CENSORED_HISTORY", raising=False)
    with pytest.raises(RuntimeError): _read_offline_planning_history(path, 1, 2)
    monkeypatch.setenv("ADAFLOOR_REVISION_CENSORED_HISTORY", "1")
    monkeypatch.delenv("ADAFLOOR_REVISION_HARD_KV_GUARD", raising=False)
    with pytest.raises(RuntimeError, match="independent hard-cap"):
        _read_offline_planning_history(path, 1, 2)
