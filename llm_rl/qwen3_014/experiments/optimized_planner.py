#!/usr/bin/env python3
"""Exact pruning around the pinned planner, without changing its MILP inputs.

For quota(t,q), q disjoint early edges need >=2q incident early vertices,
and a full perfect matching must be able to contain >=q early edges. These
are necessary conditions only. Every remaining candidate calls the original oracle, with
its original objective, constraints, edge order, limits and tie breaking.
"""
from bisect import bisect_right
from collections import OrderedDict
from functools import lru_cache, wraps
import hashlib
import json
from pathlib import Path
import sys
import time

from cached_planner import load, SOURCE


def install(module, *, progress=True):
    stats = dict(oracle_calls=0, vertex_pruned=0, matching_pruned=0,
                 original_oracle_calls=0, contexts=0, pair_hits=0, pair_misses=0,
                 repair_neighbor_pruned=0)
    original_oracle = module._quota_aware_matching_oracle
    original_pairs = module._pair_metrics
    contexts = OrderedDict()
    last_report = [time.monotonic()]

    def report(force=False):
        now = time.monotonic()
        if progress and (force or now-last_report[0] >= 20):
            print(json.dumps(dict(event='exact_planner_progress', **stats)), flush=True)
            last_report[0] = now

    @lru_cache(maxsize=65536)
    def pair(left, right, safety, max_len, block):
        # PromptStats/PromptBundle are frozen dataclasses. All their semantic
        # fields participate in this key, as do the three numeric settings.
        stats['pair_misses'] += 1
        return original_pairs([left, right], active_peak_safety_factor=safety,
                              max_response_len=max_len)[(0,1)]

    @wraps(original_pairs)
    def pairs(group, *, active_peak_safety_factor, max_response_len):
        result = {}
        before = pair.cache_info()
        begin = time.perf_counter()
        # The original function records the time of cache misses. Account for
        # wrapper/cache-hit work separately to avoid charging misses twice.
        graph_before = (module._TIMING_PROFILE or {}).get('graph_construction',0)
        for i,left in enumerate(group):
            for j in range(i+1,len(group)):
                value = pair(left,group[j],active_peak_safety_factor,max_response_len,
                             module._KV_BLOCK_SIZE_TOKENS)
                result[(i,j)] = (*value[:-1],list(value[-1]))
        after = pair.cache_info()
        stats['pair_hits'] += after.hits-before.hits
        elapsed = time.perf_counter()-begin
        graph_inner = (module._TIMING_PROFILE or {}).get('graph_construction',0)-graph_before
        if module._TIMING_PROFILE is not None and module._REPAIR_TIMING_DEPTH == 0:
            module._TIMING_PROFILE['graph_construction'] += max(0,elapsed-graph_inner)
        return result

    def context(n, metrics, allowed):
        key = (id(metrics),n,frozenset(allowed))
        cached = contexts.get(key)
        if cached is not None:
            contexts.move_to_end(key)
            return cached
        # Retain metrics itself so Python cannot reuse its id while cached.
        by_time = sorted(allowed,key=lambda e:metrics[e][module.PAIR_RANK_LOAD])
        times=[]; masks=[]; mask=0
        for a,b in by_time:
            mask |= (1 << a) | (1 << b)
            times.append(metrics[(a,b)][module.PAIR_RANK_LOAD]); masks.append(mask)
        value = dict(metrics=metrics, edges=by_time, times=times, masks=masks,
                     matching_numbers={})
        contexts[key] = value
        stats['contexts'] += 1
        if len(contexts) > 16:
            contexts.popitem(last=False)
        return value

    @wraps(original_oracle)
    def oracle(n, metrics, allowed, quotas):
        stats['oracle_calls'] += 1
        if quotas:
            begin = time.perf_counter()
            ctx = context(n,metrics,allowed)
            prefixes=[]
            rejected=None
            for threshold,required in quotas:
                count=bisect_right(ctx['times'],threshold)
                vertices=ctx['masks'][count-1].bit_count() if count else 0
                if count < required or vertices < 2*required:
                    rejected='vertex_pruned'; break
                prefixes.append((count,required))
            if rejected is None:
                for count,required in prefixes:
                    maximum=ctx['matching_numbers'].get(count)
                    if maximum is None:
                        # Prefix-only matching can be too optimistic: its
                        # early pairs may strand late vertices. Maximize the
                        # number of early edges among FULL maximum-cardinality
                        # matchings instead. Integer0/1 weights give an exact
                        # upper bound for each quota, not a heuristic solution.
                        graph=module.nx.Graph()
                        graph.add_nodes_from(range(n))
                        for index,edge in enumerate(ctx['edges']):
                            graph.add_edge(*edge,weight=int(index<count))
                        full=module.nx.max_weight_matching(graph,maxcardinality=True,weight='weight')
                        maximum=(sum(graph[a][b]['weight'] for a,b in full)
                                 if len(full)*2==n else -1)
                        ctx['matching_numbers'][count]=maximum
                    if maximum < required:
                        rejected='matching_pruned'; break
            module._record_timing('graph_construction',begin)
            if rejected:
                stats[rejected] += 1
                report()
                return None
        stats['original_oracle_calls'] += 1
        result=original_oracle(n,metrics,allowed,quotas)
        report()
        return result

    original_repair=module._best_single_swap_repair
    def bounded_repair(groups, plans, bad_index, neighbor_index, **options):
        # Same rounds, candidate order, exact solver, scoring and tie-break as
        # the pinned repair. Only skip a neighbor solve when even an ideal
        # feasible zero-overflow neighbor cannot improve the current prefix.
        cache=options['solve_cache']
        limit=options['repair_candidate_limit']
        solve_options={k:v for k,v in options.items()
                       if k not in ('solve_cache','repair_candidate_limit')}
        cap=options['max_rank_peak_tokens']
        bad=plans[bad_index]; neighbor=plans[neighbor_index]
        overflow_bad=module._plan_effective_overflow(bad,cap)
        overflow_neighbor=module._plan_effective_overflow(neighbor,cap)
        current=(float(not bad.feasible)+float(not neighbor.feasible),
                 overflow_bad+overflow_neighbor,max(overflow_bad,overflow_neighbor))
        bg=groups[bad_index];ng=groups[neighbor_index]
        order=sorted(range(len(bg)),key=lambda i:(bg[i].load,bg[i].peak_active_tokens))
        risk=sorted(range(len(bg)),key=lambda i:(bg[i].load,bg[i].peak_active_tokens),reverse=True)
        other=sorted(range(len(ng)),key=lambda i:(ng[i].load,ng[i].peak_active_tokens))
        first=min(limit,len(order)); expanded=min(max(limit*2,first),len(order))
        rounds=[(order[:first],other[:first])]
        if expanded>first:rounds.append((order[:expanded],other[:expanded]))
        rounds.append((risk[:first],other[:expanded]))
        for left,right in rounds:
            best=None
            for x in left:
                for y in right:
                    cb=list(bg);cn=list(ng)
                    cb[x],cn[y]=cn[y],cb[x]
                    bp=module._solve_group_cached(cb,cache,**solve_options)
                    bo=module._plan_effective_overflow(bp,cap)
                    optimistic=(float(not bp.feasible),bo,bo)
                    if optimistic>=current:
                        stats['repair_neighbor_pruned']+=1
                        report()
                        continue
                    np=module._solve_group_cached(cn,cache,**solve_options)
                    if neighbor.feasible and not np.feasible:continue
                    score=module._pair_repair_score(bp,np,max_rank_peak_tokens=cap)
                    if score[:3]>=current:continue
                    if best is None or score<best[0]:
                        best=(score,x,y,bp,np)
            if best is not None:
                return best[1:]
        return None

    @wraps(original_repair)
    def repair(*args,**kwargs):
        begin=time.monotonic()
        if progress:
            print(json.dumps(dict(event='exact_repair_search_begin',
                bad_step=args[2]+1,neighbor_step=args[3]+1)),flush=True)
        result=bounded_repair(*args,**kwargs)
        if progress:
            print(json.dumps(dict(event='exact_repair_search_end',
                seconds=time.monotonic()-begin,found=result is not None)),flush=True)
        return result

    module._pair_metrics=pairs
    module._quota_aware_matching_oracle=oracle
    module._best_single_swap_repair=repair
    return stats


def main():
    if Path(sys.argv[1]).resolve() != SOURCE.resolve():
        raise RuntimeError('Unexpected original planner path')
    module=load()
    stats=install(module)
    sys.argv=[str(SOURCE),*sys.argv[2:]]
    print(json.dumps(dict(event='exact_planner_optimization',
        original_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        optimization_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        search_order_budget_objective_and_original_milp_unchanged=True)),flush=True)
    module.main()
    print(json.dumps(dict(event='exact_planner_final_stats',**stats)),flush=True)


if __name__ == '__main__':
    main()
