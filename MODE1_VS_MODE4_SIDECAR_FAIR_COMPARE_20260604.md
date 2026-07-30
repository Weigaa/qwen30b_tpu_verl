# Mode 1 vs Mode 4 Sidecar Fair Compare (2026-06-04)

## Setup

- Main model: `Qwen3-30B-A3B`
- Sidecar model: `Qwen2.5-1.5B-Instruct`
- Sidecar parallel mode: `dp`
- Same sidecar knobs on both runs:
  - `max_model_len=2048`
  - `max_tokens=1024`
  - `max_num_seqs=128`
  - `max_num_batched_tokens=65536`
  - `max_prompts_per_device=128`
  - `max_prompts=1024`
  - `generate_chunk_size=32`
  - `repeat_until_killed=1`
- Fairness constraints:
  - `mode=1` vs `mode=4`
  - both use `floor=8`
  - both use the same inactive-rank sidecar device set `0..7`

## Run Artifacts

- Compare root: `sidecar_runs/compare_mode1_vs_mode4_20260604021336`
- Mode 1 main log: [wjeagerqwen30b-a3b-with_draft_breakdown_20260604021336_elastic.txt](/workspace/cann-recipes-train/llm_rl/qwen3/wjeagerqwen30b-a3b-with_draft_breakdown_20260604021336_elastic.txt)
- Mode 4 main log: [wjeagerqwen30b-a3b-with_draft_breakdown_20260604022045_elastic.txt](/workspace/cann-recipes-train/llm_rl/qwen3/wjeagerqwen30b-a3b-with_draft_breakdown_20260604022045_elastic.txt)
- Mode 1 sidecar monitor: [monitor.log](/workspace/cann-recipes-train/llm_rl/qwen3/sidecar_runs/compare_mode1_vs_mode4_20260604021336/mode1/monitor.log)
- Mode 4 sidecar monitor: [monitor.log](/workspace/cann-recipes-train/llm_rl/qwen3/sidecar_runs/compare_mode1_vs_mode4_20260604021336/mode4/monitor.log)
- Mode 1 lease: [lease.log](/workspace/cann-recipes-train/llm_rl/qwen3/sidecar_runs/compare_mode1_vs_mode4_20260604021336/mode1/lease.log)
- Mode 4 lease: [lease.log](/workspace/cann-recipes-train/llm_rl/qwen3/sidecar_runs/compare_mode1_vs_mode4_20260604021336/mode4/lease.log)
- Mode 1 sidecar infer: [infer.log](/workspace/cann-recipes-train/llm_rl/qwen3/sidecar_runs/compare_mode1_vs_mode4_20260604021336/mode1/infer.log)
- Mode 4 sidecar infer: [infer.log](/workspace/cann-recipes-train/llm_rl/qwen3/sidecar_runs/compare_mode1_vs_mode4_20260604021336/mode4/infer.log)

## Headline Result

`mode=4` gives the sidecar a much longer lease window and much higher total completed work, but it also slows the main rollout a lot more than repaired `mode=1`.

## Main Rollout Impact

| Metric | Mode 1 | Mode 4 | Delta |
|---|---:|---:|---:|
| `rollout_output_time_s` | `100.748297 s` | `153.704050 s` | `+52.956 s` (`+52.6%`) |
| shrink total mean | `4009.70 ms` | `3391.18 ms` | `-618.52 ms` |
| shrink warmup mean | `1177.78 ms` | `0.25 ms` | `-1177.53 ms` |
| restore total mean | `2200.47 ms` | `477.33 ms` | `-1723.14 ms` |

Interpretation:

- `mode=4` 的 shrink/restore 控制面其实更快。
- 但它把 sidecar 可用窗口拉长了很多，主任务是在更晚的时候才发起 restore，所以最终 `rollout_output_time_s` 反而更长。

## Sidecar Performance Impact

| Metric | Mode 1 | Mode 4 | Delta |
|---|---:|---:|---:|
| lease to restore request | `60.499 s` | `116.739 s` | `+56.240 s` (`1.93x`) |
| watch window end | `63.208 s` | `119.648 s` | `+56.440 s` |
| completed prompts | `871` | `2385` | `+1514` (`2.74x`) |
| resume prompts | `89` | `79` | `-10` |
| output tokens | `219227` | `612509` | `+393282` (`2.79x`) |
| aggregate decode tok/s | `904.161` | `915.957` | `+1.3%` |
| tokens / lease-second | `3623.647` | `5246.824` | `+44.8%` |
| prompts / lease-second | `14.397` | `20.430` | `+41.9%` |
| avg shard load time | `18.270 s` | `21.152 s` | `+2.882 s` |
| avg shard total time | `49.025 s` | `105.947 s` | `+56.922 s` |

Interpretation:

- `mode=4` 并不是单卡 sidecar decode 速度大幅变快，`agg_tokens_per_infer_s` 只比 `mode=1` 高约 `1.3%`。
- 主要收益来自 sidecar 获得了更长的可运行窗口，所以总 prompt / total token 显著更高。
- 从“单位租期产出”看，`mode=4` 也更高，说明它不只是多跑更久，窗口利用率本身也更好一些。

## Extra Resident Memory Impact

### Confirmed from sidecar startup logs

At sidecar startup, per-device available memory in the 8 inactive ranks was:

- `mode=1`: mean `35.920 GiB`
- `mode=4`: mean `32.096 GiB`
- delta: `mode=4` has about `3.824 GiB` less free memory per sidecar device

This is the cleanest measured proxy in this experiment for "额外驻留显存" on the sidecar devices.

### Why the gap exists

- `mode=1` run uses `kv_cache_init_headroom=0`.
- `mode=4` run explicitly applies `kv_cache_init_headroom=1073741824` (`1 GiB`) and keeps the mode-4 remote-cache path enabled.
- In sidecar monitor logs, `mode=4` sidecar sees much larger KV cache capacity:
  - `mode=1`: mean `377,344` tokens
  - `mode=4`: mean `1,201,904` tokens
- So this is not just a bookkeeping difference. `mode=4` leaves noticeably less free device memory for sidecar because the main runtime retains more decode-related residency / cache capacity.

### What I can and cannot claim from this batch

Confirmed:

- sidecar-visible free memory on inactive ranks is lower by about `3.8 GiB / card` under `mode=4`
- `mode=4` applies a non-zero KV init headroom while `mode=1` does not

Not fully isolated from these logs alone:

- the exact split of that `3.8 GiB` into `1 GiB` headroom vs remote-cache residency vs other mode-4 runtime retention

## Root Practical Conclusion

If your objective is:

- maximize sidecar total work during the shrink window: `mode=4` wins clearly
- minimize impact on main PPO rollout latency: repaired `mode=1` wins clearly

So the trade-off is very direct now:

- `mode=1`: better main-task latency, smaller sidecar memory squeeze, shorter sidecar lease
- `mode=4`: worse main-task latency, larger sidecar memory squeeze, much better sidecar throughput over the whole shrink interval

## Recommendation

For a fair report, I would summarize it as:

1. `mode=4` reduces sidecar-available memory by about `3.8 GiB / inactive NPU` relative to repaired `mode=1` in this setup.
2. `mode=4` improves sidecar completed prompts by `2.74x` and output tokens by `2.79x` mainly by extending the sidecar execution window from `60.5 s` to `116.7 s`.
3. The cost is that main rollout time increases from `100.7 s` to `153.7 s`.
