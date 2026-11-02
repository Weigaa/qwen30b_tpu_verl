"""CPU-only admissibility evidence; never change a benchmark's KV budget."""
import importlib.util
import json
import sys
from dataclasses import replace
from pathlib import Path

import pytest


def test_common_history_hard_prompts_have_no_graph_f4_edge_even_with_ideal_donors(monkeypatch):
    root=Path(__file__).parents[1]
    monkeypatch.syspath_prepend(str(root/'tools'))
    spec=importlib.util.spec_from_file_location('fixed4_budget_actual_planner',
                                              root/'tools/build_mode1_length_sorted_e2e_plan.py')
    module=importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules,spec.name,module)
    spec.loader.exec_module(module)
    history=Path('/data/adafloor_shared_state/common_epoch0_probe_gpu09_kv380800_permanent')
    if not (history/'epoch_000_mode0_probe/offline_planning_history.json').is_file():
        pytest.skip('requires the preserved public epoch0 history')
    rows=json.loads((history/'epoch_000_mode0_probe/offline_planning_history.json').read_text())['records']
    hard=[module._stats_from_lengths(tuple(rows[i]['lengths'])) for i in (2,56,137)]
    # Zero-length prompts and very short synthetic companions deliberately
    # UNDERestimate actual KV usage. Infeasibility here cannot be repaired
    # by exchanging the real, no-shorter prompts between epoch buckets.
    group=hard+[module._stats_from_lengths((100.,)*16) for _ in range(29)]
    group=[replace(p,source_idx=i,prompt_token_length=0) for i,p in enumerate(group)]
    metrics=module._pair_metrics(group,active_peak_safety_factor=1.16,
                                       max_response_len=16384)
    for hard_index in range(3):
        incident=[m[1] for pair,m in metrics.items() if hard_index in pair]
        assert min(incident)==262144
        assert all(value>253184 for value in incident)
    assert module._find_min_weight_matching_under_adjusted_peak(32,metrics,253184) is None
    assert module._find_min_weight_matching_under_adjusted_peak(32,metrics,280576) is not None
