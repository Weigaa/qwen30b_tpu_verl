# Mode1 Floor4 Communication Group Lifecycle Fix

Date: 2026-06-17

## Background

Mode=1, floor=4 baseline random-batch runs repeatedly hit NPU OOM in later
steps. The failure happened inside `aclnnMoeDistributeDispatchV4`, where HCCL
tried to allocate about 1.68 GB for MC2 dispatch resources.

The key symptom was not KV cache growth. KV cache stayed fixed, but
`non_torch` NPU memory kept increasing across steps, leaving less free memory
before the next post-shrink MoE dispatch warmup.

## Symptom Before Fix

In the failing run:

- Log:
  `mode1_baseline_random_batch_floor4_threshold_multistep/logs/wjqwen30b-a3b-record_graph_save4eagle3_20260617143442.txt`
- KV cache:
  `321,408 tokens`
- Failure:
  `Memory_Allocation_Failure`, requested `1,678,770,176` bytes.

Representative rank8 memory before `16 -> 8` post-shrink MoE warmup:

| step | free bytes | non_torch bytes |
| --- | ---: | ---: |
| 1 | 4,599,361,536 | 12,002,820,096 |
| 2 | 2,043,961,344 | 14,277,816,320 |
| 3 | 1,878,994,944 | 14,440,781,824 |
| 4 | 1,414,021,120 | 14,903,009,792 |

At step 4, free memory was already lower than the HCCL allocation request, so
OOM was inevitable.

## Root Cause

The leaking path was the low-floor mode1 `stash_group` release path.

When ranks such as 8/9/10/11 left the 8-rank floor group during `8 -> 4`,
old DP/EP/MC2 groups were removed from live/cache lookups, but the old code
could route them through deferred destroy:

```text
Elastic parallel group destroy deferred ... reason=stash_group deferred_count=...
```

Even when raw device/cpu process groups were manually retired, the full
`GroupCoordinator.destroy()` path did not run. That is unsafe for HCCL/MC2
because communicator/operator-side resources may stay resident in non-torch
memory. The observed behavior was exactly that: `deferred_count` grew by step,
and `non_torch` grew with it.

## Fix Strategy

The lifecycle rule for mode1 floor4 should be:

- Keep full-world reusable groups.
- Keep only the current active floor group during the current shrink phase.
- Fully destroy old floor DP/EP/MC2 groups when leaving that floor.
- Do not defer `stash_group` floor-group destruction by default.
- Force cleanup/sync around post-shrink staging release and MoE warmup so
  memory logs reflect the real allocator state.

## Code Changes

### 1. Disable deferred destroy for `stash_group` by default

File:
`vllm_ascend/worker/worker_v1.py`

Key logic:

```python
if (reason == "stash_group" and not _env_flag(
        "VLLM_ASCEND_MODE1_PARITY_DEFER_GROUP_DESTROY_ON_STASH",
        "0")):
    return False
```

This makes stale floor groups use the full `group.destroy()` path unless
explicitly overridden.

### 2. Add explicit script default

File:
`run_mode1_local_baseline_random_batch.sh`

```bash
export VLLM_ASCEND_MODE1_PARITY_DEFER_GROUP_DESTROY_ON_STASH="${VLLM_ASCEND_MODE1_PARITY_DEFER_GROUP_DESTROY_ON_STASH:-0}"
```

### 3. Strengthen cleanup around shrink/warmup

File:
`run_mode1_local_baseline_random_batch.sh`

```bash
export VLLM_ASCEND_POST_SHRINK_STAGING_RELEASE_EMPTY_CACHE="${VLLM_ASCEND_POST_SHRINK_STAGING_RELEASE_EMPTY_CACHE:-1}"
export VLLM_ASCEND_POST_SHRINK_STAGING_RELEASE_SYNC="${VLLM_ASCEND_POST_SHRINK_STAGING_RELEASE_SYNC:-1}"
export VLLM_ASCEND_MODE1_PARITY_RELEASE_WARMUP_CACHE="${VLLM_ASCEND_MODE1_PARITY_RELEASE_WARMUP_CACHE:-1}"
export VLLM_ASCEND_MODE1_PARITY_PRE_MOE_WARMUP_EMPTY_CACHE="${VLLM_ASCEND_MODE1_PARITY_PRE_MOE_WARMUP_EMPTY_CACHE:-1}"
```

File:
`vllm_ascend/worker/worker_v1.py`

Before mode1 parity MoE warmup, run a controlled:

```python
gc.collect()
torch.npu.empty_cache()
torch.npu.synchronize()
```

and log before/after memory. The low-floor mode1 parity branch returns early,
so the warmup-cache release must also be applied inside that branch after the
MoE-only warmup:

```python
if _env_flag("VLLM_ASCEND_MODE1_PARITY_RELEASE_WARMUP_CACHE", "0"):
    gc.collect()
    torch.npu.empty_cache()
    torch.npu.synchronize()
```

Without this branch-local post-warmup cleanup, the script-level
`VLLM_ASCEND_MODE1_PARITY_RELEASE_WARMUP_CACHE=1` does not affect the mode1
parity path.

## Validation Result

After the fix:

- Log:
  `mode1_baseline_random_batch_floor4_threshold_multistep/logs/wjqwen30b-a3b-record_graph_save4eagle3_20260617151411.txt`
- The run finished through step 5 without OOM.
- No `Elastic parallel group destroy deferred ... reason=stash_group` lines
  appeared.
- `stash_group` old 8-rank groups used direct destroy fallback and cleanup.
- `drop_old_floor_group_deferred_groups=0` in the shrink breakdown.

Representative rank8 memory before `16 -> 8` post-shrink MoE warmup:

| step | free bytes | non_torch bytes |
| --- | ---: | ---: |
| 1 | 5,152,989,184 | 11,449,192,448 |
| 2 | 2,621,722,624 | 13,700,055,040 |
| 3 | 2,566,787,072 | 13,752,989,696 |
| 4 | 2,480,283,648 | 13,836,747,264 |
| 5 | 2,458,677,248 | 13,858,822,656 |

The first step materializes runtime/HCCL resources, so a one-time increase is
expected. After that, `non_torch` stabilizes around 13.7-13.9 GB instead of
continuing to climb until OOM.

## Full-Length Run Follow-Up

A later non-threshold full run failed in the second rollout:

- Log:
  `mode1_baseline_random_batch_floor4/logs/wjqwen30b-a3b-record_graph_save4eagle3_20260617153647.txt`
- KV cache:
  `321,408 tokens`
- Max sequence length:
  `17,408`
- Reported concurrency:
  `18.46x`
- Failing operator:
  `PagedAttentionOperation`
- Failure:
  `Memory_Allocation_Failure(EL0004): Failed to allocate memory requested by APP module`

This failure is different from the stale communication-group leak. The log has
`drop_old_floor_group_deferred_groups=0`, and there are no
`reason=stash_group` deferred-destroy lines.

The failure happened after shrinking to the 8-rank group
`[0, 3, 5, 6, 7, 10, 11, 13]`. Rank 6 had about 1.83 GB free immediately after
the pre-warmup cleanup, but only about 94 MB free after the post-shrink
MoE-only warmup:

```text
after_post_shrink_mode1_parity_pre_moe_warmup_cache_release rank=6 free_bytes=1830936576
after_post_shrink_mode1_parity_moe_warmup rank=6 free_bytes=93745152
```

The subsequent scheduler dump shows a long decode request:

```text
CachedRequestData(req_ids=['46'], num_computed_tokens=[10815])
```

So this full-run failure is best treated as a runtime APP/PagedAttention
workspace shortage under the full 16k response setting, not as proof that old
floor groups are still accumulating across steps.

The first mitigation is to ensure post-warmup cache release really runs in the
mode1 parity branch, then rerun the full experiment and compare the new
`after_post_shrink_mode1_parity_moe_warmup_cache_release` free bytes. If the
free space is still only a few hundred MB, the full-length configuration needs
explicit PagedAttention/APP workspace headroom, for example by reducing the
floor4 KV cap slightly. That is a capacity trade-off for full-length decode,
not a substitute for fixing communication-group lifecycle leaks.

### Follow-Up Validation After Post-Warmup Release

A rerun with branch-local post-warmup cache release still failed in the second
rollout:

- Log:
  `mode1_baseline_random_batch_floor4/logs/wjqwen30b-a3b-record_graph_save4eagle3_20260617164923.txt`
- KV cache:
  `321,408 tokens`
- Reported concurrency:
  `18.46x`
- Shrunk active ranks:
  `[3, 5, 6, 7, 10, 11, 12, 13]`

## Fixed-Topology Shrink-Aware Exception

Date: 2026-06-18

The baseline rule above intentionally releases old floor groups because each
step may choose different 8-rank and 4-rank survivor sets. In that random
survivor setting, caching old floor groups wastes HCCL/MC2 workspace and can
cause non-torch memory growth.

The shrink-aware 5:2:1 oracle experiment is different:

- Intermediate survivors are fixed: `[8, 9, 10, 11, 12, 13, 14, 15]`.
- Final survivors are fixed: `[8, 9, 10, 11]`.
- Therefore DP/EP/MC2 groups for 16, 8, and 4 ranks are reusable across
  rollout steps.

For this fixed topology, repeatedly destroying and recreating the same 8-rank
and 4-rank groups is unnecessary and can increase HCCL workspace pressure.
The preferred lifecycle is:

- Create each fixed floor group once.
- Keep floor DP/EP/MC2 groups in the elastic group cache.
- On restore to full world, do not evict the cached 8-rank or 4-rank groups.
- On the next shrink to the same rank set, hit the cache instead of creating a
  new communicator.
- Disable synthetic post-shrink MoE warmup by default; cached groups should be
  reused directly, and real decode can naturally materialize any remaining
  operator workspace.

The oracle script enables this fixed-topology path with:

```bash
export VLLM_ASCEND_MODE1_PARITY_FIXED_TOPOLOGY_REUSE=1
export VLLM_ASCEND_MODE1_PARITY_CACHE_FLOOR_GROUPS=1
export VLLM_ASCEND_MODE1_PARITY_KEEP_MC2_GROUP_CACHE=1
export VLLM_ASCEND_MODE1_PARITY_KEEP_FLOOR4_GROUP_CACHE=1
export VLLM_ASCEND_MODE1_PARITY_KEEP_FLOOR4_GROUP_KINDS=dp,ep,mc2
export VLLM_ASCEND_MODE1_PARITY_DROP_OLD_FLOOR_BEFORE_REBUILD=0
export VLLM_ASCEND_MODE1_PARITY_RELEASE_LIVE_OLD_FLOOR_ON_REBUILD=0
export VLLM_ASCEND_MODE1_PARITY_POST_SHRINK_WARMUP=0
export VLLM_ASCEND_REPEAT_POST_SHRINK_MOE_DISPATCH_WARMUP=0
```

The worker-side rule is guarded by
`VLLM_ASCEND_MODE1_PARITY_FIXED_TOPOLOGY_REUSE`, so baseline random-survivor
runs keep using the old-group release policy.

The new cleanup hook did run, but rank 6 still had only about 103 MB free:

```text
after_post_shrink_mode1_parity_moe_warmup_cache_release rank=6 free_bytes=103006208
```

The failure again happened in `PagedAttentionOperation`, this time with 13
decode-only requests around 9k tokens:

```text
CachedRequestData(... num_computed_tokens=[9157, ..., 9210])
```

This confirms that the full-length run needs more APP/PagedAttention workspace
than the original `gpu_memory_utilization=0.90` setup leaves after MC2/MoE
runtime resources are materialized. To preserve the expected floor4 KV budget,
the current local baseline keeps the `321408` token cap and first tries to
raise the rollout memory utilization:

```bash
export VLLM_ASCEND_MODE1_PARITY_MAX_KV_TOKENS_FLOOR4="${VLLM_ASCEND_MODE1_PARITY_MAX_KV_TOKENS_FLOOR4:-321408}"
ROLLOUT_GPU_MEMORY_UTILIZATION="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.95}"
```

For `max_seq_len=17408`, `321408` tokens keeps the target `18.46x`
concurrency. If `0.95` is still not enough, lowering the floor4 KV cap remains
the direct fallback, but it is a capacity trade-off rather than the preferred
first fix.

## How To Verify On Another Machine

Search the run log:

```bash
rg -n "Memory_Allocation_Failure|Failed to allocate|Elastic parallel group destroy deferred|reason=stash_group|before_post_shrink_mode1_parity_moe_warmup|training/global_step" LOGFILE
```

Expected:

- No `Memory_Allocation_Failure`.
- No `Elastic parallel group destroy deferred ... reason=stash_group`.
- `before_post_shrink_mode1_parity_moe_warmup` `non_torch` stabilizes after
  the first one or two steps.
- `drop_old_floor_group_deferred_groups=0`.

If full direct destroy causes long stalls on a different environment, the old
behavior can be restored for diagnosis with:

```bash
export VLLM_ASCEND_MODE1_PARITY_DEFER_GROUP_DESTROY_ON_STASH=1
```

But this should not be the default for mode1 floor4 because it risks HCCL
workspace accumulation.

## 2026-06-18 Conservative Oracle Fallback

The shrink-aware 5:2:1 oracle run briefly tried fixed-topology communicator
reuse because its 16 -> 8 -> 4 rank sets are deterministic. The first step
completed, but the second step failed before rollout prefill:

```text
NPUWorkspaceAllocator tried to allocate 216.00 MiB
current working operator name is SelfAttentionOperation
```

The important clue was that restore to full world kept stale floor groups on
survivor ranks:

```text
Elastic lightweight path keeps stale DP/EP/MC2 cache ... cached_ranks=(8..15)
```

After restore, survivor ranks carried about 16.2-16.5 GB `non_torch`, while
donor ranks were around 12.7-13.0 GB. The next full-world prefill then needed
attention workspace with only a few hundred MB free on rank 9. So fixed
topology reuse is correct as a concept, but keeping floor MC2/HCCL groups alive
through the next full-world prefill is too aggressive when the KV cap is near
the baseline budget.

For stability-first validation, use the conservative profile:

```bash
export VLLM_ASCEND_MODE1_PARITY_MAX_KV_TOKENS_FLOOR4="${VLLM_ASCEND_MODE1_PARITY_MAX_KV_TOKENS_FLOOR4:-280576}"
export VLLM_ASCEND_MODE1_PARITY_FIXED_TOPOLOGY_REUSE="${VLLM_ASCEND_MODE1_PARITY_FIXED_TOPOLOGY_REUSE:-0}"
export VLLM_ASCEND_MODE1_PARITY_KEEP_MC2_GROUP_CACHE="${VLLM_ASCEND_MODE1_PARITY_KEEP_MC2_GROUP_CACHE:-0}"
export VLLM_ASCEND_MODE1_PARITY_CACHE_FLOOR_GROUPS="${VLLM_ASCEND_MODE1_PARITY_CACHE_FLOOR_GROUPS:-0}"
export VLLM_ASCEND_MODE1_PARITY_KEEP_FLOOR4_GROUP_CACHE="${VLLM_ASCEND_MODE1_PARITY_KEEP_FLOOR4_GROUP_CACHE:-0}"
export VLLM_ASCEND_MODE1_PARITY_KEEP_FLOOR4_GROUP_KINDS="${VLLM_ASCEND_MODE1_PARITY_KEEP_FLOOR4_GROUP_KINDS:-dp,ep}"
export VLLM_ASCEND_MODE1_PARITY_DROP_OLD_FLOOR_BEFORE_REBUILD="${VLLM_ASCEND_MODE1_PARITY_DROP_OLD_FLOOR_BEFORE_REBUILD:-1}"
export VLLM_ASCEND_MODE1_PARITY_RELEASE_LIVE_OLD_FLOOR_ON_REBUILD="${VLLM_ASCEND_MODE1_PARITY_RELEASE_LIVE_OLD_FLOOR_ON_REBUILD:-1}"
```

`280576` tokens is exactly `2192 * 128`, about `16.12x` for
`max_seq_len=17408`. This is a deliberate capacity trade-off: the first goal is
to let the oracle run finish and verify the scheduling logic. Once stable, a
more nuanced optimization can keep only cheap fixed-topology caches, or release
MC2 before full-world prefill and recreate/reuse it only after the next shrink.

After the conservative run completed 5 steps without OOM, the next validation
is to restore the baseline KV budget while keeping the communication cache
policy conservative:

```bash
export VLLM_ASCEND_MODE1_PARITY_MAX_KV_TOKENS_FLOOR4="${VLLM_ASCEND_MODE1_PARITY_MAX_KV_TOKENS_FLOOR4:-288000}"
export VLLM_ASCEND_MODE1_PARITY_FIXED_TOPOLOGY_REUSE="${VLLM_ASCEND_MODE1_PARITY_FIXED_TOPOLOGY_REUSE:-0}"
export VLLM_ASCEND_MODE1_PARITY_KEEP_MC2_GROUP_CACHE="${VLLM_ASCEND_MODE1_PARITY_KEEP_MC2_GROUP_CACHE:-0}"
export VLLM_ASCEND_MODE1_PARITY_CACHE_FLOOR_GROUPS="${VLLM_ASCEND_MODE1_PARITY_CACHE_FLOOR_GROUPS:-0}"
export VLLM_ASCEND_MODE1_PARITY_KEEP_FLOOR4_GROUP_CACHE="${VLLM_ASCEND_MODE1_PARITY_KEEP_FLOOR4_GROUP_CACHE:-0}"
```

This isolates the remaining question: whether timely floor-group release alone
is enough to support the baseline `288000` token KV reservation for the
shrink-aware redistributed dataset.
