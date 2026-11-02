from dataclasses import replace
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import build_mode1_length_sorted_e2e_plan as planner


def prompt(index, lengths):
    return replace(planner._stats_from_lengths(tuple(lengths)),
                   source_idx=index, prompt_token_length=1)


def test_admission_does_not_assume_prompt_peaks_are_separated(monkeypatch):
    monkeypatch.setattr(planner, "_KV_BLOCK_SIZE_TOKENS", 1)
    group = [prompt(0, [10, 10]), prompt(1, [20, 20])]
    monkeypatch.setattr(planner, "_KV_PEAK_MODEL", "synchronized")
    old = planner._pair_metrics(group, active_peak_safety_factor=1,
                                max_response_len=100)[(0, 1)][1]
    monkeypatch.setattr(planner, "_KV_PEAK_MODEL", "independent_prompts")
    new = planner._pair_metrics(group, active_peak_safety_factor=1,
                                max_response_len=100)[(0, 1)][1]
    assert old == 44
    assert new == 64  # 2*(1+10) + 2*(1+20)
    # A 50-token pool was admitted solely by assuming disjoint peak times.
    assert old <= 50 < new


def test_real_failed_pair_is_rejected_without_changing_memory_budget(monkeypatch):
    monkeypatch.setattr(planner, "_KV_BLOCK_SIZE_TOKENS", 128)
    monkeypatch.setattr(planner, "_KV_PEAK_MODEL", "independent_prompts")
    a = [2727,2818,2821,3034,3086,3151,3450,3639,3869,3939,4074,4593,4780,5166,5342,5469]
    b = [3352,3659,3705,3815,3878,3881,3965,4248,4318,4411,4419,4423,5213,5313,5552,5680]
    group = [replace(prompt(33,a),prompt_token_length=31),
             replace(prompt(53,b),prompt_token_length=34)]
    peak = planner._pair_metrics(group, active_peak_safety_factor=1.16,
                                 max_response_len=16384)[(0, 1)][1]
    assert peak == 116480
    assert peak > 105728


@pytest.mark.parametrize("factor", [1, 1.16])
def test_bundle_sums_each_prompt_not_each_bundle(monkeypatch, factor):
    monkeypatch.setattr(planner, "_KV_BLOCK_SIZE_TOKENS", 128)
    items = [prompt(i, [100+i, 200+i]) for i in range(64)]
    bundle = planner._adjacent_prompt_bundles(items)[0]
    args = dict(safety_factor=factor, max_response_len=16384)
    assert planner._independent_prompt_kv_peak(bundle, **args) == sum(
        planner._independent_prompt_kv_peak(items[i], **args)
        for i in bundle.source_indices)
