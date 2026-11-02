import importlib.util
from pathlib import Path
import pytest

spec = importlib.util.spec_from_file_location("costs", Path(__file__).parents[1]/"analysis_eval/summarize_revision_transition_costs.py")
module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)


def test_parallel_latencies_are_max_not_sum():
    text = "\n".join(f"Elastic parallel shrink done: rank={rank} active_ranks=[0, 1] rebuild_ms={rank+1} refresh_ms=2 total_ms={10+rank}" for rank in range(2))
    rows = module.parse(text)
    assert rows[0]["critical_rank_total_ms"] == 11
    assert rows[0]["component_max_ms"]["rebuild_ms"] == 2


def test_missing_timed_ranks_are_not_a_complete_event():
    with pytest.raises(ValueError): module.parse("Elastic parallel shrink done: rank=0 active_ranks=[0, 1] total_ms=1")
