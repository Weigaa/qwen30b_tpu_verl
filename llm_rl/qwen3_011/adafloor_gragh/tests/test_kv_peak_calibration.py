from dataclasses import replace
from pathlib import Path
import json
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from kv_peak_calibration import calibrate_peak_factor
import build_mode1_length_sorted_e2e_plan as planner


def group():
    return dict(step=1, rank=0, source_indices=[0, 1])


def snapshot(length):
    return {0: ([length]*2, 1), 1: ([length]*2, 1)}


def test_cold_start_needs_no_future_observations():
    result = calibrate_peak_factor([snapshot(10)], [[]])
    assert result['factor'] == 1.3
    assert result['sample_count'] == 0
    assert result['mode'] == 'cold_start'


def test_factor_increases_from_raw_peak_ratio_not_old_inflated_estimate():
    r = calibrate_peak_factor([snapshot(10), snapshot(21)], [[], [group()]],
                              block_size_tokens=1)
    assert r['records'][0]['predicted_peak_tokens'] == 44
    assert r['records'][0]['actual_peak_tokens'] == 88
    assert r['factor'] == 2.04


def test_decline_is_limited_and_recent_window_can_forget_old_outlier():
    h = [snapshot(10), snapshot(21), snapshot(21)]
    r = calibrate_peak_factor(h, [[], [group()], [group()]], window=1,
                              previous_factor=2.2, block_size_tokens=1)
    assert r['max_observed_ratio'] == 1
    assert r['sample_count'] == 1
    assert r['factor'] == 2.15
    assert calibrate_peak_factor(h, [[], [group()], [group()]], window=2,
                                block_size_tokens=1)['factor'] == 2.04


def test_history_pair_is_compared_under_same_mapping():
    h = [{0: ([10],1), 1: ([100],1), 2: ([20],1)},
         {0: ([20],1), 1: ([200],1), 2: ([30],1)}]
    g = dict(step=2, rank=7, source_indices=[0,2])
    r = calibrate_peak_factor(h, [[],[g]], block_size_tokens=1)
    assert r['records'][0]['source_indices'] == [0,2]
    assert r['records'][0]['predicted_peak_tokens'] == 22
    assert r['records'][0]['actual_peak_tokens'] == 42


def test_missing_completed_plan_is_not_treated_as_cold_start():
    with pytest.raises(ValueError, match='missing completed'):
        calibrate_peak_factor([snapshot(10),snapshot(11)],[[],[]])


def test_multiplier_applies_once_to_peak_and_preserves_completion_curves(monkeypatch):
    monkeypatch.setattr(planner, '_KV_PEAK_MODEL', 'calibrated')
    monkeypatch.setattr(planner, '_KV_PEAK_FACTOR', 1.4)
    monkeypatch.setattr(planner, '_KV_BLOCK_SIZE_TOKENS', 1)
    a = replace(planner._stats_from_lengths((10.,10.)), source_idx=0, prompt_token_length=1)
    b = replace(planner._stats_from_lengths((20.,20.)), source_idx=1, prompt_token_length=1)
    m = planner._pair_metrics([a,b],active_peak_safety_factor=1,max_response_len=100)
    assert m[(0,1)][1] == 62  # ceil(44*1.4), not 64 from sum of local peaks
    assert planner._pair_metrics([a,b], active_peak_safety_factor=1.16,
                                 max_response_len=100)[(0,1)][1] == 62


def test_completed_history_loader_preserves_previous_factor_and_provenance(tmp_path):
    import pandas as pd
    frame = pd.DataFrame({'prompt':['a','b']})
    stats = [replace(planner._stats_from_lengths((10.,10.)),source_idx=i,
                     prompt_token_length=1) for i in range(2)]
    history = [{'a':stats[0], 'b':stats[1]}]*2
    dirs = [tmp_path/'epoch0',tmp_path/'epoch1']
    for d in dirs:
        d.mkdir(); (d/'offline_planning_history.json').write_text('{}')
    (dirs[1]/'oracle').mkdir()
    (dirs[1]/'oracle/length_sorted_rank_plan.json').write_text(json.dumps([{
        'step':1,'rank_to_source_idx':{'0':[0,1]},
        'kv_peak_calibration':{'factor':1.5}}]))
    result=planner._calibrate_from_history(history,dirs,frame,stats,None)
    assert result['factor'] == 1.45
    assert result['sample_count'] == 1
    assert len(result['sources']) == 3
    assert all(len(s['sha256'])==64 for s in result['sources'])
    # A partial history or different prompt subset must not silently calibrate.
    (dirs[1]/'oracle/length_sorted_rank_plan.json').write_text(json.dumps([{
        'step':1,'rank_to_source_idx':{'0':[0]}}]))
    with pytest.raises(ValueError,match='cover every prompt'):
        planner._calibrate_from_history(history,dirs,frame,stats,None)


def test_impossible_single_prompt_fails_before_cross_step_search(monkeypatch):
    monkeypatch.setattr(planner, '_KV_BLOCK_SIZE_TOKENS', 128)
    stat=replace(planner._stats_from_lengths((16384.,)*16),source_idx=137,
                 prompt_token_length=384)
    monkeypatch.setattr(planner, '_KV_PEAK_FACTOR', 1.4)
    with pytest.raises(RuntimeError,match='one prompt at Full16'):
        planner._check_single_prompt_capacity([stat],353408,16384)
    monkeypatch.setattr(planner, '_KV_PEAK_FACTOR', 1.3)
    assert planner._check_single_prompt_capacity([stat],353408,16384)['required'] <= 353408
