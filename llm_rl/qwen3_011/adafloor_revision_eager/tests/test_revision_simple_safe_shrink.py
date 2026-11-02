import importlib.util
from pathlib import Path

spec=importlib.util.spec_from_file_location('simple',Path(__file__).parents[1]/'analysis_eval/analyze_revision_simple_safe_shrink.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def test_prompt_kv_and_independent_blocks_are_counted():
    assert module.peak([[128],[128]],[1,1],1,128)==512


def test_greedy_rejects_dead_end_instead_of_calling_it_safe():
    metrics={(0,1):100,(0,2):1,(0,3):100,(1,2):100,(1,3):1,(2,3):100}
    assert module.greedy_pairs([0,1,2,3],1,metrics)==[(0,2),(1,3)]
    assert module.greedy_pairs([0,1,2,3],.5,metrics) is None


def test_release_is_quorum_proxy_not_independent_idle_sum():
    pairs=[(2*i,2*i+1) for i in range(16)];tails={i:i//2+1 for i in range(32)}
    assert module.release_proxy(pairs,tails,4)==8*(16-8)+4*(16-12)
