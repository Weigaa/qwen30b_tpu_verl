# Pure EP Graph Baseline

This file records the currently validated qwen3 rollout baseline so we do not
lose the stable point while iterating on graph-mode memory optimizations.

## Stable Baseline

Validated log:

```text
llm_rl/qwen3/resample_result_16k_bs32_n16_async_oldcfg/logs/wjqwen30b-a3b-record_graph_save4eagle3_20260511150353.txt
```

Result summary:

```text
exit_code=0
rollout_output_time_s=959.699324
rollouts speed tokens/s=3301.769543
response_len_count=512 min=1034 mean=6112.613 p50=5199 p90=12738 p95=15999 max=16384
length finish count=25/512, about 4.9%
```

The important correctness signal is that graph mode no longer produces an
abnormally large fraction of responses capped at max response length.

## Reproduction Command

Run from:

```bash
cd /workspace/cann-recipes-train/llm_rl/qwen3
```

Command:

```bash
VLLM_ROLLOUT_PARALLEL_MODE=ep \
VLLM_ROLLOUT_ZIYI_ALIGN=1 \
VLLM_ROLLOUT_GRAPH_WITH_RESAMPLER=1 \
VLLM_ROLLOUT_UNSAFE_TRUE_GRAPH_WITH_RESAMPLER=1 \
VLLM_ROLLOUT_FAST_DEBUG=1 \
MAX_RESPONSE_LENGTH=16384 \
TRAIN_BATCH_SIZE=64 \
ROLLOUT_N=8 \
ROLLOUT_MAX_NUM_SEQS=64 \
VLLM_ROLLOUT_DEBUG_GENERATION=1 \
VLLM_ROLLOUT_DEBUG_ABORT_AFTER_GENERATE=1 \
TRAINER_TOTAL_EPOCHS=1 \
TRAINER_TOTAL_TRAINING_STEPS=1 \
bash internal/wj_train_grpo_qwen30b_a3b_16die_true_weight_regroup_async_oldcfg.sh
```

## Key Runtime Settings

```text
parallel mode: ep
rollout TP: 1
rollout DP: 16
vLLM EP: enabled
ALL_TO_ALL_RESHARD: 1
USE_ALLTOALL_OVERLAP: 1
graph mode: enabled
enforce_eager: False
async_scheduling: false
prefix caching: true
chunked prefill: true
max_num_batched_tokens: 17408
gpu_memory_utilization: 0.87
sleep level: 1
CaMem weight reload: 0
manual free cache engine: 0
filter empty weight shards: 0
invalidate ACL graph after weight update: 0
recapture ACL graph after weight update: 0
legacy attention: 0
legacy fused MoE: 0
data rebalance: 0
length balance: 0
```

## Ablation Plan

Start from the stable baseline above and add only one qwen3 optimization at a
time. Keep batch shape and generation settings unchanged when judging graph
correctness.

1. `VLLM_ROLLOUT_FILTER_EMPTY_WEIGHT_SHARDS=1`
2. `VLLM_ROLLOUT_CAMEM_WEIGHT_RELOAD=1`
3. `VLLM_ROLLOUT_SLEEP_LEVEL=2`
4. `VLLM_ROLLOUT_MANUAL_FREE_CACHE_ENGINE=1`
5. `VLLM_ROLLOUT_INVALIDATE_ACLGRAPH_AFTER_WEIGHT_UPDATE=1` together with `VLLM_ROLLOUT_RECAPTURE_ACLGRAPH_AFTER_WEIGHT_UPDATE=1`

Treat an experiment as suspicious if the capped response ratio rises far above
the stable baseline, or if the run starts showing the earlier "many responses
run to max length" pattern.

## Ablation Results

### 1. Empty EP Weight Shard Filter

Command delta:

```bash
VLLM_ROLLOUT_FILTER_EMPTY_WEIGHT_SHARDS=1
```

Validated log:

```text
llm_rl/qwen3/resample_result_16k_bs32_n16_async_oldcfg/logs/wjqwen30b-a3b-record_graph_save4eagle3_20260511153556.txt
```

Result summary:

```text
exit_code=0
rollout_output_time_s=956.273769
rollouts speed tokens/s=3313.597112
response_len_count=512 min=1034 mean=6112.613 p50=5199 p90=12738 p95=15999 max=16384
```

Conclusion: this optimization did not reproduce the graph-mode response length
anomaly. Its response-length distribution matched the stable baseline, while
runtime was effectively unchanged.

### 2. CaMem Weight Reload

Command delta, cumulative with the previous passing optimization:

```bash
VLLM_ROLLOUT_FILTER_EMPTY_WEIGHT_SHARDS=1
VLLM_ROLLOUT_CAMEM_WEIGHT_RELOAD=1
```

Validated log:

```text
llm_rl/qwen3/resample_result_16k_bs32_n16_async_oldcfg/logs/wjqwen30b-a3b-record_graph_save4eagle3_20260511155707.txt
```

Result summary:

```text
exit_code=0
rollout_output_time_s=962.174932
rollouts speed tokens/s=3293.274326
response_len_count=512 min=1034 mean=6112.613 p50=5199 p90=12738 p95=15999 max=16384
```

Conclusion: enabling CaMem weight reload together with empty-shard filtering
also did not reproduce the graph-mode response length anomaly.

### 3. Sleep Level 2

Command delta, cumulative with the previous passing optimizations:

```bash
MAX_RESPONSE_LENGTH=8192
VLLM_ROLLOUT_FILTER_EMPTY_WEIGHT_SHARDS=1
VLLM_ROLLOUT_CAMEM_WEIGHT_RELOAD=1
VLLM_ROLLOUT_SLEEP_LEVEL=2
```

Validated log:

```text
llm_rl/qwen3/resample_result_16k_bs32_n16_async_oldcfg/logs/wjqwen30b-a3b-record_graph_save4eagle3_20260511161735.txt
```

Result summary:

```text
exit_code=0
rollout_output_time_s=510.542393
rollouts speed tokens/s=7992.723143
response_len_count=512 min=207 mean=7893.703 p50=8192 p90=8192 p95=8192 max=8192
```

8k reference without sleep level 2:

```text
llm_rl/qwen3/resample_result_16k_bs32_n16_async_oldcfg/logs/wjqwen30b-a3b-record_graph_save4eagle3_20260511130741.txt
rollout_output_time_s=785.535809
rollouts speed tokens/s=3431.270693
response_len_count=512 min=829 mean=5188.160 p50=5034 p90=8192 p95=8192 max=8192
```

Conclusion: `VLLM_ROLLOUT_SLEEP_LEVEL=2` is a strong suspect for the graph-mode
response length anomaly. The 8k run shifted from a mixed stop/length
distribution to a near-cap distribution with `p50=8192` and many
`finish='length'` entries in the debug preview. Keep true graph rollout on
sleep level 1 unless explicitly testing this path.

### 4. Sleep-Level Guard Verification

After guarding true graph rollout to default back to sleep level 1, rerun the
same cumulative passing optimizations without explicitly setting sleep level:

```bash
MAX_RESPONSE_LENGTH=8192
VLLM_ROLLOUT_FILTER_EMPTY_WEIGHT_SHARDS=1
VLLM_ROLLOUT_CAMEM_WEIGHT_RELOAD=1
```

Validated log:

```text
llm_rl/qwen3/resample_result_16k_bs32_n16_async_oldcfg/logs/wjqwen30b-a3b-record_graph_save4eagle3_20260511163142.txt
```

Result summary:

```text
exit_code=0
rollout_output_time_s=495.411748
rollouts speed tokens/s=5387.859720
response_len_count=512 min=1034 mean=5137.033 p50=5199 p90=8192 p95=8192 max=8192
```

Conclusion: the graph response-length distribution returns to the normal 8k
reference shape when sleep level stays at 1, even with empty-shard filtering
and CaMem weight reload enabled. This strengthens the conclusion that sleep
level 2, not these two optimizations, caused the earlier near-cap distribution.

### 5. Manual Free Cache Engine

Command delta, cumulative with the passing optimizations and sleep level 1:

```bash
MAX_RESPONSE_LENGTH=8192
VLLM_ROLLOUT_FILTER_EMPTY_WEIGHT_SHARDS=1
VLLM_ROLLOUT_CAMEM_WEIGHT_RELOAD=1
VLLM_ROLLOUT_MANUAL_FREE_CACHE_ENGINE=1
```

Validated log:

```text
llm_rl/qwen3/resample_result_16k_bs32_n16_async_oldcfg/logs/wjqwen30b-a3b-record_graph_save4eagle3_20260511164512.txt
```

Result summary:

```text
exit_code=1
failure point: resume(tags=["kv_cache"]) -> init_cache_engine() -> initialize_from_config()
error: ValueError: Graph parameters have already been set!
```

Conclusion: manual KV/cache-engine rebuild is not currently compatible with
true graph rollout. It fails before generation because the reinitialize path
calls `set_graph_params()` a second time while ACL graph global parameters are
already live. This is a separate correctness blocker from the response-length
distribution anomaly.

### 6. Invalidate And Recapture ACL Graph

Command delta, cumulative with the passing optimizations, sleep level 1, and
manual cache-engine rebuild disabled:

```bash
MAX_RESPONSE_LENGTH=8192
VLLM_ROLLOUT_FILTER_EMPTY_WEIGHT_SHARDS=1
VLLM_ROLLOUT_CAMEM_WEIGHT_RELOAD=1
VLLM_ROLLOUT_MANUAL_FREE_CACHE_ENGINE=0
VLLM_ROLLOUT_INVALIDATE_ACLGRAPH_AFTER_WEIGHT_UPDATE=1
VLLM_ROLLOUT_RECAPTURE_ACLGRAPH_AFTER_WEIGHT_UPDATE=1
```

Validated log:

```text
llm_rl/qwen3/resample_result_16k_bs32_n16_async_oldcfg/logs/wjqwen30b-a3b-record_graph_save4eagle3_20260511165022.txt
```

Result summary:

```text
exit_code=0
rollout_output_time_s=505.506922
rollouts speed tokens/s=5280.262018
response_len_count=512 min=1034 mean=5137.033 p50=5199 p90=8192 p95=8192 max=8192
```

Conclusion: invalidate plus recapture did not reproduce the response-length
anomaly in the 8k screening run. Distribution is identical to the passing
sleep-level-1 reference. The extra recapture path is semantically safe in this
test, though it is slightly slower than the guarded sleep-level-1 run and its
necessity should be judged separately.

### 7. Async Scheduling

Command delta, cumulative with the passing optimizations, sleep level 1, and
manual cache-engine rebuild disabled:

```bash
MAX_RESPONSE_LENGTH=8192
VLLM_ROLLOUT_ASYNC_SCHEDULING=true
VLLM_ROLLOUT_FILTER_EMPTY_WEIGHT_SHARDS=1
VLLM_ROLLOUT_CAMEM_WEIGHT_RELOAD=1
VLLM_ROLLOUT_MANUAL_FREE_CACHE_ENGINE=0
VLLM_ROLLOUT_INVALIDATE_ACLGRAPH_AFTER_WEIGHT_UPDATE=0
VLLM_ROLLOUT_RECAPTURE_ACLGRAPH_AFTER_WEIGHT_UPDATE=0
```

Validated log:

```text
llm_rl/qwen3/resample_result_16k_bs32_n16_async_oldcfg/logs/wjqwen30b-a3b-record_graph_save4eagle3_20260511172558.txt
```

Result summary:

```text
exit_code=0 via VLLM_ROLLOUT_DEBUG_ABORT_AFTER_GENERATE=1
rollout_output_time_s=479.811400
rollouts speed tokens/s=5559.953344
response_len_count=512 min=1034 mean=5134.143 p50=5199 p90=8192 p95=8192 max=8192
```

Conclusion: enabling vLLM async scheduling did not reproduce the graph-mode
response length anomaly in the 8k screening run. The distribution is aligned
with the sleep-level-1 reference, and rollout time improved from 495.4s to
479.8s in this one-step diagnostic. This should still get a 16k confirmation
before treating async=true as the final default, but it is safe enough to keep
testing on the graph path.
