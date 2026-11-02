from dataclasses import replace
from pathlib import Path
import sys
import json
from types import SimpleNamespace

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from kv_peak_calibration import (
    calibrate_peak_factor, calibrate_prompt_factors, PromptAdaptivePeakModel,
    live_kv_peak,
)
import build_mode1_length_sorted_e2e_plan as planner


def state(histories, groups, previous=None):
    public=calibrate_peak_factor(histories,groups,block_size_tokens=1)
    public.update(calibrate_prompt_factors(histories,public,
                  previous_factors=previous,block_size_tokens=1))
    return public


def test_cold_start_is_exactly_public_factor_on_combined_curve():
    h=[{0:((10,10),1),1:((20,20),1)}]
    s=state(h,[[]]); model=PromptAdaptivePeakModel(h,s,block_size_tokens=1)
    assert s['prompt_factors']=={'0':1.3,'1':1.3}
    r=model.evaluate((0,1),100)
    assert r['weighted_peak_tokens']==56  # response lengths become 13 and 26
    assert r['pair_residual_factor']==1
    assert r['admission_peak_tokens']==56
    assert r['calibration_samples']==0


def test_sparse_local_estimates_remain_close_to_public_but_risk_can_rise():
    h=[{0:((10,10),1),1:((10,10),1)},
       {0:((20,20),1),1:((10,10),1)}]
    s=state(h,[[],[dict(step=1,rank=0,source_indices=[0,1])]])
    a=s['prompt_calibration']['0'];b=s['prompt_calibration']['1']
    assert a['sample_count']==b['sample_count']==1
    assert a['local_weight']==b['local_weight']==.25  # one transition, not 2 responses
    assert a['factor']>=a['local_target']
    assert a['factor']>b['factor']
    assert b['factor']>=1.25  # no more than .05 below the cold-start factor


def test_saved_prompt_factor_limits_downward_change():
    h=[{0:((10,10),1)}]*2
    s=state(h,[[],[dict(step=1,rank=0,source_indices=[0])]],{'0':1.6})
    assert s['prompt_factors']['0']==1.55


def test_migrating_from_public_mode_uses_preceding_public_factor():
    h=[{0:((10,10),1)}]*2
    public=calibrate_peak_factor(h,[[],[dict(step=1,rank=0,source_indices=[0])]],
                                 previous_factor=1.6,block_size_tokens=1)
    assert calibrate_prompt_factors(h,public,block_size_tokens=1)['prompt_factors']['0']==1.55


def test_pair_residual_catches_changed_overlap_and_checks_unseen_pairs():
    h=[{0:((10,10),1),1:((1,20),1)},
       {0:((10,10),1),1:((10,20),1)}]
    c=dict(prompt_factors={'0':1,'1':1.05},window=3,ratio_guard=1.02,quantum=.01)
    model=PromptAdaptivePeakModel(h,c,block_size_tokens=1)
    r=model.evaluate((0,1),100)
    assert r['pair_history_checks'][0]['calibrated_prediction']==33
    assert r['pair_history_checks'][0]['reconstructed_actual']==44
    assert r['pair_residual_factor']==1.36
    assert r['admission_peak_tokens']==60
    assert model.evaluate((1,0),100)==r


def test_heterogeneous_weighting_preserves_response_exit_positions():
    h=[{0:((10,10),1),1:((20,20),1)}]
    c=dict(prompt_factors={'0':1.1,'1':1.3},window=3,ratio_guard=1.02,quantum=.01)
    r=PromptAdaptivePeakModel(h,c,block_size_tokens=1).evaluate((0,1),100)
    assert r['weighted_peak_tokens']==54  # second prompt dominates at t=26
    assert r['admission_peak_tokens']==54


def test_pair_history_window_excludes_expired_transition():
    h=[{0:((10,),1)},{0:((100,),1)},{0:((100,),1)}]
    c=dict(prompt_factors={'0':1.3},window=1,ratio_guard=1.02,quantum=.01)
    r=PromptAdaptivePeakModel(h,c,block_size_tokens=1).evaluate((0,),1000)
    assert r['pair_residual_factor']==1
    assert r['pair_history_checks'][0]['history_index']==2


def test_planner_pair_metrics_use_new_model_without_extra_public_multiplier(monkeypatch):
    h=[{0:((10,10),1),1:((20,20),1)}]
    model=PromptAdaptivePeakModel(h,state(h,[[]]),block_size_tokens=1)
    monkeypatch.setattr(planner,'_KV_PEAK_MODEL','prompt_adaptive')
    monkeypatch.setattr(planner,'_KV_PROMPT_MODEL',model)
    monkeypatch.setattr(planner,'_KV_PEAK_FACTOR',1.3)
    group=[replace(planner._stats_from_lengths(tuple(h[0][i][0])),
                   source_idx=i,prompt_token_length=1) for i in range(2)]
    m=planner._pair_metrics(group,active_peak_safety_factor=1.16,max_response_len=100)
    assert m[(0,1)][planner.PAIR_ADJUSTED_PEAK]==56


def test_full16_single_check_uses_each_prompt_factor(monkeypatch):
    h=[{0:((100,),1),1:((10,),1)}]
    c=dict(prompt_factors={'0':1.1,'1':2},window=3,ratio_guard=1.02,quantum=.01)
    model=PromptAdaptivePeakModel(h,c,block_size_tokens=1)
    monkeypatch.setattr(planner,'_KV_PEAK_MODEL','prompt_adaptive')
    monkeypatch.setattr(planner,'_KV_PROMPT_MODEL',model)
    monkeypatch.setattr(planner,'_KV_PEAK_FACTOR',2)
    stats=[replace(planner._stats_from_lengths(tuple(h[0][i][0])),source_idx=i,
                   prompt_token_length=1) for i in range(2)]
    assert planner._check_single_prompt_capacity(stats,120,1000)['required']==112
    with pytest.raises(RuntimeError,match='one prompt at Full16'):
        planner._check_single_prompt_capacity(stats,100,1000)


def test_bundle_path_uses_all_four_distinct_prompts(monkeypatch):
    h=[{i:((10+i,20+i),1) for i in range(64)}]
    s=state(h,[[]]); model=PromptAdaptivePeakModel(h,s,block_size_tokens=128)
    monkeypatch.setattr(planner,'_KV_PEAK_MODEL','prompt_adaptive')
    monkeypatch.setattr(planner,'_KV_PROMPT_MODEL',model)
    items=[replace(planner._stats_from_lengths(tuple(v[0])),source_idx=i,
                   prompt_token_length=1) for i,v in h[0].items()]
    bundles=planner._adjacent_prompt_bundles(items)
    m=planner._pair_metrics(bundles,active_peak_safety_factor=1.16,max_response_len=1000)
    ids=bundles[0].source_indices+bundles[1].source_indices
    assert m[(0,1)][planner.PAIR_ADJUSTED_PEAK]==model.evaluate(ids,1000)['admission_peak_tokens']


def test_step_metadata_and_history_reload_preserve_individual_state(tmp_path,monkeypatch):
    import pandas as pd
    monkeypatch.setattr(planner,'_KV_PROMPT_MODEL',None)
    calibration={'factor':1.3,'prompt_factors':{'0':1.6,'1':1.3},
                 'prompt_calibration':{'0':{'factor':1.6},'1':{'factor':1.3}},
                 'records':[{'do_not_duplicate':True}]}
    monkeypatch.setattr(planner,'_KV_CALIBRATION',calibration)
    metadata=[planner._calibration_metadata(SimpleNamespace(prompts=[SimpleNamespace(source_idx=i)]))
              for i in range(2)]
    assert metadata[0]['prompt_factors']=={'0':1.6}
    assert metadata[1]['prompt_factors']=={'1':1.3}
    assert 'records' not in metadata[0]
    dirs=[tmp_path/'e0',tmp_path/'e1']
    for d in dirs:
        d.mkdir();(d/'offline_planning_history.json').write_text('{}')
    (dirs[1]/'oracle').mkdir()
    plans=[dict(step=i+1,rank_to_source_idx={'0':[i]},kv_peak_calibration=metadata[i]) for i in range(2)]
    (dirs[1]/'oracle/length_sorted_rank_plan.json').write_text(json.dumps(plans))
    stats=[replace(planner._stats_from_lengths((10.,10.)),source_idx=i,prompt_token_length=1) for i in range(2)]
    history=[{'a':stats[0],'b':stats[1]}]*2
    r=planner._calibrate_from_history(history,dirs,pd.DataFrame({'prompt':['a','b']}),
                                      stats,None,prompt_adaptive=True)
    assert r['prompt_factors']['0']==1.55
    assert r['prompt_factors']['1']==1.25
    assert planner._KV_PROMPT_MODEL.factors[0]==1.55


def test_formal_entrypoints_enable_prompt_adaptation_and_pair_checks():
    root=Path(__file__).resolve().parents[1]
    runner=(root/'run_qwen3_paper_graph_main_from_common_epoch0.sh').read_text()
    assert 'KV_ADMISSION_PEAK_MODEL=prompt_adaptive' in runner
    assert 'kv_prompt_prior_strength=3.0' in runner
    assert 'kv_admission_safety_factor=1.0' in runner
    for floor in (2,4):
        source=(root/f'run_mode1_local_length_sorted_e2e_adaptive_floor{floor}.sh').read_text()
        assert '--kv-prompt-prior-strength' in source
        assert '--kv-prompt-minimum-factor' in source


def test_repair_pruning_matches_exhaustive_pair_peak_feasibility(monkeypatch):
    h=[{i:((v,),1) for i,v in enumerate((10,10,20,20))}]
    c=dict(prompt_factors={str(i):1 for i in range(4)},window=3,ratio_guard=1.02,quantum=.01)
    model=PromptAdaptivePeakModel(h,c,block_size_tokens=1)
    monkeypatch.setattr(planner,'_KV_PEAK_MODEL','prompt_adaptive')
    monkeypatch.setattr(planner,'_KV_PROMPT_MODEL',model)
    items=[replace(planner._stats_from_lengths(v[0]),source_idx=i,prompt_token_length=1)
           for i,v in h[0].items()]
    matchings=[((0,1),(2,3)),((0,2),(1,3)),((0,3),(1,2))]
    for bound in (21,22,23,43):
        exhaustive=any(max(model.evaluate(pair,100)['admission_peak_tokens']
                           for pair in matching)<bound for matching in matchings)
        assert planner._can_reduce_full16_peak(items,bound,active_peak_safety_factor=1,
                                               max_response_len=100)==exhaustive
