import random
from cached_planner import load


def test_peak_memoization_exact_for_lengths_prompts_and_block_sizes():
    m=load();rng=random.Random(0)
    for n in (1,16,32):
        for _ in range(30):
            lengths=tuple(float(rng.randint(0,16384)) for _ in range(n))
            prompts=tuple(rng.randint(0,1024) for _ in range(n))
            assert m._peak_active_tokens(lengths)==m._peak_active_tokens.__wrapped__(lengths)
            for block in (1,128):
                expected=m._peak_active_kv_tokens.__wrapped__(lengths,prompts,block_size_tokens=block)
                assert m._peak_active_kv_tokens(lengths,prompts,block_size_tokens=block)==expected
                assert m._peak_active_kv_tokens(lengths,prompts,block_size_tokens=block)==expected
    assert m._peak_active_kv_tokens.cache_info().hits>=180


def test_same_pair_metric_values_and_source_identity_after_cache():
    m=load();a=m._stats_from_lengths(tuple(range(100,1700,100)))
    b=m._stats_from_lengths(tuple(range(200,3400,200)))
    x=m._pair_metrics([a,b],active_peak_safety_factor=1.16,max_response_len=16384)
    m._peak_active_tokens=m._peak_active_tokens.__wrapped__
    m._peak_active_kv_tokens=m._peak_active_kv_tokens.__wrapped__
    y=m._pair_metrics([a,b],active_peak_safety_factor=1.16,max_response_len=16384)
    assert x==y
