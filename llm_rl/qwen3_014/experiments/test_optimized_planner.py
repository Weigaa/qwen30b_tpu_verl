from dataclasses import replace
import itertools
import random
import pytest
from cached_planner import load
from optimized_planner import install


def metrics_for(n,seed):
    rng=random.Random(seed)
    return {(a,b):(1, rng.randint(1,20), rng.randint(30,60),
                    rng.randint(1,8), rng.randint(1,10),[a,b])
            for a in range(n) for b in range(a+1,n)}


def test_random_oracle_exact_matching_not_just_feasibility():
    original=load(); fast=load(); stats=install(fast,progress=False)
    for seed in range(8):
        n=8; metrics=metrics_for(n,seed)
        allowed={e for e,v in metrics.items() if v[1] <= 16}
        for quotas in ((),((1,2),),((4,3),),((7,4),),((3,1),(6,3)),((2,1),(5,2),(8,4))):
            expected=original._quota_aware_matching_oracle(n,metrics,allowed,quotas)
            actual=fast._quota_aware_matching_oracle(n,metrics,allowed,quotas)
            assert actual == expected
    assert stats['vertex_pruned'] > 0
    assert stats['original_oracle_calls'] > 0


def test_star_prefix_pruned_by_matching_not_vertex_count():
    m=load(); stats=install(m,progress=False)
    metrics=metrics_for(8,0)
    for edge,value in list(metrics.items()):
        metrics[edge]=(*value[:3],1 if 0 in edge else 9,*value[4:])
    # All8 vertices occur, but the early star can finish at most one rank.
    assert m._quota_aware_matching_oracle(8,metrics,set(metrics),((1,2),)) is None
    assert stats['matching_pruned'] == 1 and stats['original_oracle_calls'] == 0


def test_early_pairs_must_extend_to_a_full_perfect_matching():
    original=load();m=load();stats=install(m,progress=False)
    # Two disjoint early edges exist, but late vertices4/5 force nodes0/1
    # away from their early pair. A full perfect matching has <=one early edge.
    metrics={(0,1):(1,1,1,1,0,[0,1]),(2,3):(1,1,1,1,0,[2,3]),
             (0,4):(1,1,1,2,0,[0,4]),(1,5):(1,1,1,2,0,[1,5])}
    assert original._quota_aware_matching_oracle(6,metrics,set(metrics),((1,2),)) is None
    assert m._quota_aware_matching_oracle(6,metrics,set(metrics),((1,2),)) is None
    assert stats['matching_pruned']==1 and stats['original_oracle_calls']==0


def test_exhaustive_small_graphs_never_prune_feasible_matching():
    m=load(); install(m,progress=False)
    edges=list(itertools.combinations(range(4),2))
    for bits in range(1 << len(edges)):
        metrics={e:(1,1,1,1 if bits & (1<<i) else 2,0,list(e)) for i,e in enumerate(edges)}
        for q in (1,2):
            expected=any(sum(metrics[e][3]<=1 for e in matching) >= q for matching in (
                {(0,1),(2,3)},{(0,2),(1,3)},{(0,3),(1,2)}))
            result=m._quota_aware_matching_oracle(4,metrics,set(metrics),((1,q),))
            assert (result is not None) == expected


def test_pair_cache_exact_with_order_numeric_settings_and_prompt_content():
    original=load(); m=load(); stats=install(m,progress=False)
    group=[replace(m._stats_from_lengths(tuple(range(i+1,1600+i,100))),source_idx=i,
                   prompt_token_length=30+i) for i in range(6)]
    for seq in (group,group[::-1],group[1:]+group[:1]):
        for block in (1,128):
            original._KV_BLOCK_SIZE_TOKENS=m._KV_BLOCK_SIZE_TOKENS=block
            for safety,max_len in ((1.16,16384),(1.1,4096),(1.16,16384)):
                kwargs=dict(active_peak_safety_factor=safety,max_response_len=max_len)
                assert m._pair_metrics(seq,**kwargs) == original._pair_metrics(seq,**kwargs)
    assert stats['pair_hits'] > stats['pair_misses']
    changed=[replace(group[0],lengths=tuple([99]*16)),*group[1:]]
    kwargs=dict(active_peak_safety_factor=1.16,max_response_len=16384)
    assert m._pair_metrics(changed,**kwargs)==original._pair_metrics(changed,**kwargs)


def test_surviving_milp_receives_byte_identical_arrays_and_options():
    import numpy as np
    a=load();b=load();install(b,progress=False)
    metrics=metrics_for(8,8); quotas=((8,3),);calls=[]
    original_milp=a.milp
    def capture(**kw):
        calls.append(kw)
        return original_milp(**kw)
    a.milp=b.milp=capture
    assert a._quota_aware_matching_oracle(8,metrics,set(metrics),quotas)==b._quota_aware_matching_oracle(8,metrics,set(metrics),quotas)
    assert len(calls)==2
    for name in ('c','integrality'):
        assert np.array_equal(calls[0][name],calls[1][name])
    for name in ('A','lb','ub'):
        assert np.array_equal(getattr(calls[0]['constraints'],name),getattr(calls[1]['constraints'],name))
    assert calls[0]['options']==calls[1]['options']


def test_repair_bound_preserves_full_candidate_order_scores_and_ties():
    from types import SimpleNamespace as NS
    original=load(); fast=load();counters=install(fast,progress=False)
    rng=random.Random(20261025)
    original_calls=[];fast_calls=[]
    def solve(items,cache,*,call_log,**kw):
        key=tuple(v.source_idx for v in items)
        if key not in cache:
            call_log.append(key)
            peak=sum(v.load for v in items)/len(items)
            cache[key]=NS(feasible=peak<=100,kv_cap=100,
                rank_adjusted_peak_loads={0:peak},rank_loads={0:max(v.load for v in items)})
        return cache[key]
    original._solve_group_cached=lambda items,cache,**kw:solve(items,cache,call_log=original_calls,**kw)
    fast._solve_group_cached=lambda items,cache,**kw:solve(items,cache,call_log=fast_calls,**kw)
    kwargs=dict(max_rank_peak_tokens=100,floor_kv_caps={4:100,16:100},adaptive_floor=True,
        min_adaptive_floor=4,force_selected_floor=None,kv_safe_fixed_floor=None,
        rank_matching_policy='release_area',active_peak_safety_factor=1.16,max_response_len=16384,
        repair_candidate_limit=2,ignore_tail_ties_at_response_cap=False)
    for seed in range(40):
        values=([300,100,100,100,140,140,10,10] if seed==0
                else [rng.randint(20,300) for _ in range(8)])
        items=[NS(source_idx=i,load=x,peak_active_tokens=x) for i,x in enumerate(values)]
        groups=[items[:4],items[4:]]
        plans=[solve(g,{},call_log=[]) for g in groups]
        a=original._best_single_swap_repair(groups,plans,0,1,solve_cache={},**kwargs)
        b=fast._best_single_swap_repair(groups,plans,0,1,solve_cache={},**kwargs)
        assert a==b
    assert counters['repair_neighbor_pruned']>0
    assert len(fast_calls)<=len(original_calls)
