# Elastic Shrink Context - 2026-03-18

This document summarizes the current state of the `16 -> 8` elastic shrink work for Qwen3 MoE on Ascend, including:

- what has already been fixed,
- what regressions were observed and ruled out,
- what code paths were changed,
- where the current investigation is focused,
- and what the next agent should check first.

Current repo state when this document was written:

- branch: `master`
- HEAD: `5e92c0a`

Stable checkpoint branch already saved and pushed:

- branch: `shrink_naive`
- purpose: first basically usable lossless `16 -> 8` zero-redundancy baseline


## 1. High-level Goal

The main target is:

- elastic MoE shrink from `16` ranks to `8` ranks,
- `lossless`,
- `global_redundant_expert_num = 0`,
- quality as close as possible to pre-shrink / non-elastic behavior,
- while reducing shrink-time NPU memory overhead and avoiding very large safety margins.


## 2. What Was Successfully Fixed Earlier

### 2.1 Lossless shrink correctness

Earlier, lossless shrink quality degradation was traced to a `log2phy` / slot-order mismatch:

- `new_log2phy_cpu` was built from original `assignments`
- but actual local expert slots after activation were ordered as:
  - preserved local experts first
  - imported experts later

This caused logical expert IDs to be mapped into the wrong dense physical slots.

This was fixed by rebuilding `runtime_log2phy_cpu` from the actual post-activation local expert order.

Key file:

- [worker_v1.py](/workspace/cann-recipes-train/llm_rl/qwen3/vllm_ascend/worker/worker_v1.py)

Important result:

- quality recovered close to baseline
- score returned to normal range
- large-scale repeated punctuation / whitespace tail degeneration mostly disappeared


### 2.2 Uniform expert assignment for `16 -> 8`

The earlier soft balancing algorithm could produce uneven local expert counts like `14 / 19`, even with redundancy `0`.

That was fixed by changing `16 -> 8` zero-redundancy shrink to:

- one inactive rank maps to one unique active rank
- all 8 experts from that inactive rank go to the same active rank
- final active ranks are enforced to have exactly `16` experts each

Key file:

- [worker_v1.py](/workspace/cann-recipes-train/llm_rl/qwen3/vllm_ascend/worker/worker_v1.py)


### 2.3 Lossless branch refreshes `moe_comm_method`

Lossless shrink originally updated runtime expert metadata but did not refresh the MoE communication path.

That was fixed by calling:

- `module.refresh_elastic_groups()`

after updating:

- runtime expert count
- runtime log2phy

This avoided stale dispatcher state after shrink.


### 2.4 Runtime alias / old expert tensor release

There was an earlier OOM where old NPU expert tensors were not actually released because runtime aliases still pointed to them.

This was addressed by clearing stale runtime alias references before rebuilding / offloading.

Key file:

- [fused_moe.py](/workspace/cann-recipes-train/llm_rl/qwen3/vllm_ascend/ops/fused_moe.py)


## 3. What Was Tried for Performance / Memory

### 3.1 Preallocate shrink capacity in runtime buffers

An intermediate approach preallocated shrink-time expert capacity in `runtime_w13_buffer/runtime_w2_buffer`.

That reduced shrink-time rebuild peak, but still left complexity around:

- loaded weights vs runtime buffers
- extra copies
- offload / export path interactions


### 3.2 Current experimental direction: loaded-only enlarged slots

The current experimental branch on `master` tries this approach:

- keep `loaded weights` as the main NPU carrier
- allocate loaded weights large enough for `16` expert slots up front
- initially use only the first `8`
- on shrink, fill the remaining `8` slots directly

Why this was attempted:

- avoid whole-tensor rebuild during shrink
- reduce CPU <-> NPU round trips
- reduce shrink-time peak memory

Key files:

- [fused_moe.py](/workspace/cann-recipes-train/llm_rl/qwen3/vllm_ascend/ops/fused_moe.py)
- [worker_v1.py](/workspace/cann-recipes-train/llm_rl/qwen3/vllm_ascend/worker/worker_v1.py)

Important caveat:

- this moves part of the shrink cost into startup-time static NPU footprint
- so startup memory and decode-time headroom behavior changed significantly


## 4. Important Recent Regressions and What They Actually Meant

### 4.1 Startup KV-cache OOM after loaded-only preallocation

Observed:

- startup passed profile, but KV cache init OOMed or became much tighter than expected

Interpretation:

- preallocating 16 expert slots in loaded weights increases startup static footprint
- `profile_run` only estimates memory based on its own execution path
- it does not perfectly capture later allocations / fragmentation / post-profile lazy workspaces

Conclusion:

- startup margin cannot be dropped too aggressively just because shrink expert capacity is preallocated


### 4.2 HCCL / AllToAll blind spot

A major issue was:

- `profile_run` and real first requests were taking different MoE communication paths
- profile could warm up `MC2`, while the first real prefill would use `AllToAll`
- then `AllToAll` HCCL workspace would appear late and cause failures

This motivated aligning profile and runtime communication path selection.


### 4.3 Important correction: later failures were not caused by the path alignment itself

After profile/runtime comm-type selection was aligned, later failures showed:

- the problem was not “alignment broke communication”
- the deeper issue was that profile happens before the real runtime memory / KV state fully matches the first live request

So alignment was still the right direction. It just did not solve all runtime blind spots.


## 5. Current Question: Why Does Current Implementation Preempt Much More Than `naive`?

This is the current active investigation.

Reference logs:

- current run with unexpected heavy preemption:
  - [wjeagerqwen30b-a3b-with_draft_breakdown_20260318121426_elastic.txt](/workspace/cann-recipes-train/llm_rl/qwen3/wjeagerqwen30b-a3b-with_draft_breakdown_20260318121426_elastic.txt)
- naive comparison:
  - [naive16-8_record.txt](/workspace/cann-recipes-train/llm_rl/qwen3/naive16-8_record.txt)

Observed facts:

1. Current run and naive run use the same Ascend scheduler path.
2. The user explicitly does **not** want to focus on `max_num_batched_tokens=1024` not taking effect for now.
3. In the current run, preemptions happen:
   - before shrink
   - right after the first real prefill returns
   - at the first decode scheduling stage
4. The KV cache headline size is not dramatically smaller than naive:
   - current: about `342,528`
   - naive: about `340,992`
5. Therefore the failure mode is not well explained by “KV cache total capacity is too small”.


## 6. Current Best Hypothesis

The strongest current hypothesis is:

- **decode-time KV allocation state is inconsistent**
- specifically, request-level token progress and block-manager ownership may be out of sync after prefill

What this would look like:

- request has `num_computed_tokens > 0`
- meaning prefill already finished and the prompt is computed
- but `KVCacheManager` / coordinator has too few or zero blocks attached to that request
- so when decode asks for one more token slot, `allocate_slots()` may think it needs to allocate far more than it should
- then the scheduler hits `new_blocks is None` and starts preempting, even though total cache capacity looks sufficient

This is much closer to the user’s intuition:

- “decode KV cache allocation is wrong”
- not “there is truly no space”


## 7. What Has Already Been Ruled Out for This Preemption Issue

### 7.1 Not a shrink-time issue

The heavy preemptions appear before any `16 -> 8` shrink logs.


### 7.2 Not a communication-operator issue

This preemption behavior is happening in scheduler KV allocation, not in MoE collectives.


### 7.3 Not lookahead over-reservation in this run

This was checked carefully.

Relevant code:

- base scheduler sets:
  - `self.num_lookahead_tokens = 0` when `speculative_config is None`
  - [scheduler.py:169](/workspace/cann-recipes-train/llm_rl/qwen3/vllm/v1/core/sched/scheduler.py:169)
  - [scheduler.py:171](/workspace/cann-recipes-train/llm_rl/qwen3/vllm/v1/core/sched/scheduler.py:171)
- current log shows:
  - `speculative_config=None`
  - in both current and naive runs

So even though Ascend scheduler currently passes `self.num_lookahead_tokens` directly into `allocate_slots()`:

- [vllm_ascend/core/scheduler.py:248](/workspace/cann-recipes-train/llm_rl/qwen3/vllm_ascend/core/scheduler.py:248)
- [vllm_ascend/core/scheduler.py:378](/workspace/cann-recipes-train/llm_rl/qwen3/vllm_ascend/core/scheduler.py:378)

that value is `0` in this run, so it does **not** explain the current preemption storm.


## 8. Relevant Code Paths for the Current Investigation

### 8.1 Where preemption happens

- [vllm_ascend/core/scheduler.py:375](/workspace/cann-recipes-train/llm_rl/qwen3/vllm_ascend/core/scheduler.py:375)
- [vllm_ascend/core/scheduler.py:392](/workspace/cann-recipes-train/llm_rl/qwen3/vllm_ascend/core/scheduler.py:392)

The trigger is:

- `self.kv_cache_manager.allocate_slots(...)` returns `None`


### 8.2 What `allocate_slots()` actually checks

- [kv_cache_manager.py:260](/workspace/cann-recipes-train/llm_rl/qwen3/vllm/v1/core/kv_cache_manager.py:260)
- [kv_cache_manager.py:271](/workspace/cann-recipes-train/llm_rl/qwen3/vllm/v1/core/kv_cache_manager.py:271)

It computes:

- `num_tokens_need_slot = min(num_computed_tokens + num_new_tokens + num_lookahead_tokens, max_model_len)`
- then `num_blocks_to_allocate`
- and returns `None` if:
  - `num_blocks_to_allocate > free_blocks`

This means:

- the problem is pure block accounting at this point
- not raw NPU OOM


### 8.3 How request block ownership is stored

- [kv_cache_manager.py:383](/workspace/cann-recipes-train/llm_rl/qwen3/vllm/v1/core/kv_cache_manager.py:383)
- [kv_cache_coordinator.py:176](/workspace/cann-recipes-train/llm_rl/qwen3/vllm/v1/core/kv_cache_coordinator.py:176)
- [single_type_kv_cache_manager.py:99](/workspace/cann-recipes-train/llm_rl/qwen3/vllm/v1/core/single_type_kv_cache_manager.py:99)
- [single_type_kv_cache_manager.py:123](/workspace/cann-recipes-train/llm_rl/qwen3/vllm/v1/core/single_type_kv_cache_manager.py:123)

The critical mapping is:

- `req_to_blocks[request_id]`

If request-level counters and this mapping diverge, decode behavior can be wrong even when the global block pool still has capacity.


## 9. New Diagnostic Logs Added for This Exact Issue

To verify whether request state and KV block ownership diverge at the prefill->decode boundary, the following logs were added in:

- [vllm_ascend/core/scheduler.py](/workspace/cann-recipes-train/llm_rl/qwen3/vllm_ascend/core/scheduler.py)

### 9.1 First decode KV state

Log name:

- `Elastic first decode KV state`

Printed fields:

- `req`
- `phase`
- `prompt_tokens`
- `num_tokens`
- `computed_tokens`
- `cached_tokens`
- `num_new_tokens`
- `owned_blocks`
- `free_blocks`
- `lookahead`
- `running`
- `waiting`
- `finished_prefill`


### 9.2 Explicit mismatch detector

Log name:

- `Elastic decode KV mismatch`

Trigger condition:

- `request.num_computed_tokens > 0`
- but `owned_blocks == 0`

If this appears, it is very strong evidence that request progress and block ownership diverged.


### 9.3 First preempt detail

Log name:

- `Elastic first preempt detail`

Printed fields:

- request identity
- computed/prompt/token counts
- cached_tokens
- owned_blocks
- free_blocks
- running/waiting/finished_prefill

This is intended to answer:

- was the first preempt caused by genuine lack of free blocks
- or by wrong per-request block state


## 10. Other One-Time Boundary Logs Already Present

These were added earlier to locate the first live-request failure boundary:

### Engine-side first live step

- [llm_engine.py](/workspace/cann-recipes-train/llm_rl/qwen3/vllm/v1/engine/llm_engine.py)
  - `Elastic first live step: entering engine_core.get_output`
  - `Elastic first live step: engine_core.get_output returned`

### First model execution

- [core.py](/workspace/cann-recipes-train/llm_rl/qwen3/vllm/v1/engine/core.py)
  - `Elastic first model exec: entering model_fn scheduled_tokens=... num_reqs=...`
  - `Elastic first model exec: model_fn returned`

### First token dispatch path

- [token_dispatcher.py](/workspace/cann-recipes-train/llm_rl/qwen3/vllm_ascend/ops/moe/token_dispatcher.py)
  - `Elastic first token_dispatch: entering`
  - `Elastic first dispatch_preprocess: entering _preprocess`
  - `Elastic first dispatch_preprocess: _preprocess returned`
  - `Elastic first preprocess: entering`
  - `Elastic first preprocess: token-count gather returned`
  - `Elastic first preprocess: returning`

These are all one-time logs to keep noise low.


## 11. What Should Be Checked First in the Next Run

If the next run still shows large early preemptions, search the log for:

1. `Elastic first decode KV state`
2. `Elastic decode KV mismatch`
3. `Elastic first preempt detail`
4. `Preempting request`
5. `Elastic first model exec: model_fn returned`

Interpretation:

- If `computed_tokens > 0` and `owned_blocks == 0`:
  - very likely a prefill->decode block ownership bug
- If `owned_blocks` looks reasonable but `free_blocks` is already tiny:
  - then we should inspect where blocks are being over-consumed globally
- If both `owned_blocks` and `free_blocks` look healthy but preempt still occurs:
  - then we need to inspect `get_num_blocks_to_allocate()` more closely with request-local block counts


## 12. Things the Next Agent Should Avoid Re-litigating

These have already been examined recently:

1. `max_num_batched_tokens=1024` not taking effect
   - true, but not enough to explain current decode-time preemption pattern by itself
2. speculative lookahead over-reservation
   - not the current explanation because `speculative_config=None`
3. shrink-time expert remapping quality issue
   - already fixed and previously validated
4. stale `log2phy` / dispatcher after shrink
   - already fixed
5. profile/runtime comm-type mismatch as the direct explanation for current preemptions
   - relevant historically, but not the best explanation for the current scheduler preemption behavior


## 13. Suggested Next Action

Do **not** blindly patch communication collectives for this issue.

The next sensible step is:

1. run once with the new scheduler KV-state logs,
2. capture:
   - `Elastic first decode KV state`
   - `Elastic decode KV mismatch`
   - `Elastic first preempt detail`
3. determine whether:
   - request progress and block ownership diverge, or
   - block ownership is intact and global free-block accounting is the real problem

Only after that should scheduler logic be changed.


## 14. Important Reference Logs Mentioned in This Round

- Current preemption investigation:
  - [wjeagerqwen30b-a3b-with_draft_breakdown_20260318121426_elastic.txt](/workspace/cann-recipes-train/llm_rl/qwen3/wjeagerqwen30b-a3b-with_draft_breakdown_20260318121426_elastic.txt)
- Earlier OOM / communication investigations:
  - [wjeagerqwen30b-a3b-with_draft_breakdown_20260317221808_elastic.txt](/workspace/cann-recipes-train/llm_rl/qwen3/wjeagerqwen30b-a3b-with_draft_breakdown_20260317221808_elastic.txt)
  - [wjeagerqwen30b-a3b-with_draft_breakdown_20260318001820_elastic.txt](/workspace/cann-recipes-train/llm_rl/qwen3/wjeagerqwen30b-a3b-with_draft_breakdown_20260318001820_elastic.txt)
  - [wjeagerqwen30b-a3b-with_draft_breakdown_20260318021726_elastic.txt](/workspace/cann-recipes-train/llm_rl/qwen3/wjeagerqwen30b-a3b-with_draft_breakdown_20260318021726_elastic.txt)
  - [wjeagerqwen30b-a3b-with_draft_breakdown_20260318114409_elastic.txt](/workspace/cann-recipes-train/llm_rl/qwen3/wjeagerqwen30b-a3b-with_draft_breakdown_20260318114409_elastic.txt)
- Naive good reference:
  - [naive16-8_record.txt](/workspace/cann-recipes-train/llm_rl/qwen3/naive16-8_record.txt)

