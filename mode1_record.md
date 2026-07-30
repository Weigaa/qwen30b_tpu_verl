# Custom Mode=1 Unified Reference

Last updated: 2026-06-07

This file merges the useful final content that originally lived in:

- `CUSTOM_MODE1_NATIVE_PARITY_DEBUG.md`
- `CUSTOM_MODE1_ZERO_HEADROOM_CODEDIFF_SUMMARY.md`
- `CUSTOM_MODE1_ZERO_HEADROOM_MIGRATION.md`

It is meant to be the single follow-up reference for:

- what zero-headroom mode=1 is supposed to look like,
- which files were changed,
- why the original custom path needed headroom,
- how custom mode=1 was made to match native-level KV budget,
- how to recognize regressions quickly from runtime logs.

The three original files remain the source history. This file is the compact
merged reference and also contains the mode=1 headroom / DP / EP / MC2
communication-group lifecycle notes.

## 1. Final Target

For custom Qwen3 mode=1:

- keep the custom Qwen3 + old `vllm_ascend.ops.fused_moe.AscendFusedMoE` path,
- do not route custom mode=1 through native/common MoE code,
- run `floor=8` and `floor=4` without generic headroom,
- use native-level KV budget instead of subtracting ad hoc safety memory,
- keep shrink / restore behavior correct across multi-step training.

Validated floor8 target behavior:

- `GPU KV cache size: 377,344 tokens`
- `Maximum concurrency for 17,408 tokens per request: 21.68x`
- no generic `7.5 GB` headroom family
- successful shrink / restore

## 2. What Originally Consumed the Extra Headroom

The old custom mode=1 path carried costs that native mode=1 did not carry.

### 2.1 Generic headroom that used to be reserved

| Cost source | Old reserve | Why it existed | Current mode=1 parity behavior |
| --- | ---: | --- | --- |
| post-shrink MoE dispatch | 0.5 GB | first floor-level MC2 / MoE dispatch workspace touch | skipped for `lossless_mode1_native_parity_ready=True`; no synthetic warmup by default |
| post-restore DP | 2.0 GB | restore-time full-world DP workspace | skipped for parity-ready mode=1 |
| post-restore EP | 2.0 GB | restore-time full-world EP workspace | skipped for parity-ready mode=1 |
| post-restore MoE dispatch | 2.0 GB | restore-time full-world MC2 / dispatch workspace | skipped for parity-ready mode=1; MC2 lifecycle is controlled instead |
| first-live-prefill | 1.0 GB | insurance against real prefill peak not covered by profile | skipped for parity-ready mode=1 after stale workspace overlap was removed |
| total | 7.5 GB | generic workaround, not a true fix | eliminated from steady-state mode=1 |

Important nuance: the first-live-prefill issue was not mathematically proven to
be impossible. The practical fix is that the large stale DP / EP / MC2 workspace
overlap is gone, so the residual first-live-prefill variation no longer pushes
the validated floor8 / floor4 / floor2 / floor1 runs over capacity.

### 2.2 Root causes behind that headroom

| Root cause | Old behavior | Fixed behavior |
| --- | --- | --- |
| Mode1 reused heavier mode2/mode3 logic | runtime expert buffers, hybrid state, CPU shadow logic still existed in mode=1 | mode=1 gets its own lightweight path |
| Runtime expert slots were treated as transient | shrink / restore needed extra materialization workspace | floor-target loaded slots are preallocated up front |
| Communication-group caches were not aligned with native | full-world EP reuse was poor and stale MC2 resources could survive too long | keep only the reusable full-world EP cache, drop stale MC2 caches |
| Worker headroom policy was class-based | custom class triggered generic reserves even when parity path was active | headroom skip is parity-gated |
| Synthetic post-shrink warmups were used as insurance | extra workspace was touched only because warmup forced it | warmup removed for mode=1 parity path |

## 3. Code Changes That Made Zero-Headroom Work

### 3.1 Keep custom Qwen3 on old `fused_moe`

File:

- `vllm_ascend/models/qwen3_moe.py`

Rule:

- custom Qwen3 stays on `vllm_ascend.ops.fused_moe.AscendFusedMoE`
- `common_fused_moe.py` is reference only, not the final execution path

### 3.2 Make mode=1 a true lightweight path

File:

- `vllm_ascend/ops/fused_moe.py`

Main changes:

- precompute floor-target loaded expert capacity before `create_weights(...)`
- keep `active_local_num_experts` at the primary local count
- keep `loaded_weight_capacity` at the floor-target count
- disable mode2/mode3 runtime state in mode=1:
  - no hybrid-tail activation
  - no CPU shadow ownership
  - no runtime `w13/w2` mode3 double-buffer dependency
- bind mode=1 runtime views directly to loaded slots
- import experts directly into loaded slots during shrink
- add per-layer `lossless_mode1_native_parity_ready`
- fail fast if mode=1 tries to fall back into heavy runtime-buffer behavior

### 3.3 Replace generic headroom with parity-gated worker logic

File:

- `vllm_ascend/worker/worker_v1.py`

Main changes:

- skip generic headrooms only when old custom mode=1 parity is actually ready
- the gate is not class-name based; the layer must report
  `lossless_mode1_native_parity_ready=True`
- `floor=8` parity path skips:
  - post-shrink MoE dispatch headroom
  - post-restore DP headroom
  - post-restore EP headroom
  - post-restore MoE dispatch headroom
  - first-live-prefill headroom
- `floor=4` starts from the same zero-generic-headroom policy

Key functions:

- `_module_is_mode1_lightweight_parity(...)`
- `_module_skips_generic_headroom(...)`
- `_estimate_post_shrink_moe_dispatch_headroom_bytes(...)`
- `_estimate_post_restore_dp_collective_headroom_bytes(...)`
- `_estimate_post_restore_ep_collective_headroom_bytes(...)`
- `_estimate_post_restore_moe_dispatch_headroom_bytes(...)`
- `_estimate_first_live_prefill_headroom_bytes(...)`
- `_estimate_custom_mode1_kv_materialize_headroom_bytes(...)`

Expected log snippets:

```text
Skipping post-shrink MoE dispatch headroom for elastic lightweight path ...
Skipping first-live-prefill headroom for elastic lightweight path ...
Skipping KV materialization headroom for elastic lightweight path ...
```

### 3.4 Align communication-group lifetime with native

Files:

- `vllm_ascend/worker/worker_v1.py`
- related MC2 / group-cache handling in the custom path

Main changes:

- keep reusable full-world DP / EP / MC2 restore caches
- cache current MC2 by rank signature, so the same stage does not recreate HCCL
- drop stale floor-world MC2 / DP / EP instead of accumulating old stages
- avoid synthetic warmups whose only job was to pre-touch a later workspace

The final mode=1 strategy is not "reserve more HBM". It is "avoid duplicate
communication workspaces".

#### 3.4.1 Shrink lifecycle

Entry point:

- `vllm_ascend/worker/worker_v1.py::rebuild_elastic_ep_group(...)`

Main sequence:

1. prepare lossless shrink payload
2. preload/import outgoing-rank expert weights into active-rank loaded slots
3. inactive ranks detach from elastic parallel groups
4. active ranks rebuild DP, EP, and MC2 over the real active-rank list
5. reset MoE communication setup cache
6. refresh model-side elastic parallel state
7. release post-shrink staging state
8. selectively drop stale cached floor-world groups after shrink
9. run only the warmups that are enabled for the current lightweight path

Important logs:

```text
Elastic parallel shrink phase breakdown ...
Elastic parallel shrink done: ... dp_size=<floor> ep_size=<floor>
Elastic post-shrink MoE dispatch warmup skipped ... reason=mode1_lightweight_parity_no_synthetic_warmup
```

`VLLM_ASCEND_MODE1_PARITY_DROP_STALE_CACHE_AFTER_SHRINK` only controls whether
this optional sweep runs immediately after shrink. It does **not** control the
mandatory stale cleanup after restore. In other words, setting it to `0` means
"do not sweep right after shrink"; it does not mean "keep stale floor MC2
forever".

When the shrink-time sweep is enabled, it must be selective: full-world DP / EP
/ MC2 are kept because restore will reuse them, while old floor-world groups
that are no longer current may be dropped. A shrink-time cleanup that drops the
full-world signature is a bug and can make restore cold-start HCCL / MC2 again.

#### 3.4.2 Restore lifecycle

Entry point:

- `vllm_ascend/worker/worker_v1.py::restore_elastic_parallel_groups(...)`

Main sequence:

1. reconcile group creation sequence
2. rebuild full-world DP
3. rebuild full-world EP
4. rebuild full-world MC2
5. reset MoE communication setup cache
6. refresh model-side full-world elastic parallel state
7. mark active ranks as full world
8. drop stale floor-world cached groups with
   `keep_group_ranks=tuple(range(world_size))`
9. optionally warm post-restore MC2 / AllToAll dispatch if configured

Important logs:

```text
Elastic parallel stale MC2 cache dropped across restore ...
Elastic parallel stale group cache dropped ...
Elastic parallel restore done: ... dp_size=16 ep_size=16
```

#### 3.4.3 Group cache rules

Key functions:

- `_should_cache_elastic_parallel_group(...)`
- `_should_keep_stale_mc2_cache_for_custom_mode1_parity(...)`
- `_should_keep_stale_group_cache_for_custom_mode1_parity(...)`
- `_should_drop_stale_group_cache_after_elastic_shrink(...)`
- `_drop_stale_cached_elastic_parallel_groups(...)`
- `_cleanup_after_elastic_mc2_group_destroy(...)`
- `_reset_elastic_moe_comm_setup_cache(...)`

`current group` means the communicator whose rank signature exactly matches the
current active-rank list. Examples:

- full-world: `(0, 1, ..., 15)`
- floor8 after shrink: e.g. `(0, 1, ..., 7)`
- floor4 after follow-up shrink: e.g. `(0, 1, 2, 3)`

MC2 current groups are cached by rank signature unless
`VLLM_ASCEND_DISABLE_ELASTIC_MC2_GROUP_CACHE=1` or diagnostic
`VLLM_ASCEND_MODE1_PARITY_SINGLE_LIVE_MC2_GROUP=1` is set. This matches the
native/common mode=1 pattern: do not repeatedly allocate a fresh MC2 resource
for the same rank signature.

`stale MC2` means an MC2 communicator whose rank signature no longer matches
the active group being used. For example, after restore to full world, the old
floor8 MC2 group is stale. Keeping both full-world MC2 and floor8 MC2 can add
several GiB of non-torch memory.

Default mode=1 behavior:

- cache current MC2 by rank signature
- keep full-world DP / EP / MC2 across shrink for restore reuse
- do not keep non-full-world stale MC2:
  `VLLM_ASCEND_MODE1_PARITY_KEEP_MC2_GROUP_CACHE=0`
- optionally and selectively drop stale floor-world caches immediately after shrink:
  `VLLM_ASCEND_MODE1_PARITY_DROP_STALE_CACHE_AFTER_SHRINK=1`
- after restore, always call
  `_drop_stale_cached_elastic_parallel_groups(tuple(range(world_size)))`; this
  is where floor MC2 becomes stale and must be cleaned
- when an MC2 cache is destroyed, also run MC2 cleanup hooks

Diagnostic escape hatches:

```bash
export VLLM_ASCEND_MODE1_PARITY_KEEP_MC2_GROUP_CACHE=1
export VLLM_ASCEND_MODE1_PARITY_SINGLE_LIVE_MC2_GROUP=1
export VLLM_ASCEND_DISABLE_ELASTIC_MC2_GROUP_CACHE=1
```

These are for diagnosis only, not the recommended zero-headroom path.

Full-world DP / EP / MC2 are the stale groups intentionally kept during shrink:

```bash
export VLLM_ASCEND_MODE1_PARITY_KEEP_FULLWORLD_DP_CACHE=1
export VLLM_ASCEND_MODE1_PARITY_KEEP_FULLWORLD_EP_CACHE=1
export VLLM_ASCEND_MODE1_PARITY_KEEP_FULLWORLD_MC2_CACHE=1
```

Reason:

- restore always returns to full world
- keeping full-world DP / EP / MC2 avoids paying the full recreate / warmup cost again
- unlike stale floor MC2 / DP / EP, these caches have a clear reuse point on restore

Only the full-world signatures should be kept this way. Stale floor EP / DP
groups should not accumulate. Mode=1 does not keep non-full-world stale DP / EP
groups by default.

#### 3.4.4 Why the 7.5 GB cost disappears

The cost did not disappear because the hardware stopped needing communication
workspace. It disappeared from KV-cache reservation because mode=1 now makes
those costs either profiled, reused, or freed at the right time.

| Cost source | Why it caused OOM before | Why no steady reserve is needed now |
| --- | --- | --- |
| DP restore workspace | a fresh full-world DP workspace could appear after KV profile | stale DP groups are not retained; restore lifecycle is explicit |
| EP restore workspace | EP full-world reuse was poor; restore could stack with old floor resources | full-world EP is intentionally reused; stale floor EP is not accumulated |
| MC2 restore / dispatch workspace | stale floor MC2 plus full-world MC2 could coexist | current MC2 is cached by rank signature; stale MC2 is dropped after restore |
| post-shrink MoE dispatch | synthetic warmup touched workspace as an insurance policy | mode=1 skips synthetic post-shrink MoE dispatch warmup by default |
| first-live-prefill | profile did not cover every real batch allocator detail | the large stale workspace overlap is removed, leaving enough real margin |
| KV materialization | old custom path held extra runtime/alias state before KV allocation | rollout KV caches and old tensor aliases are cleared before reinit |

### 3.5 Use explicit mode=1 KV cap

Files:

- `vllm/v1/core/kv_cache_utils.py`
- `vllm_ascend/worker/model_runner_v1.py`

Main changes:

- cap parity mode=1 KV budget explicitly with:
  - `VLLM_ASCEND_MODE1_PARITY_MAX_KV_TOKENS=377344`
- log KV memory before / after allocation:
  - `before_initialize_kv_cache_tensors`
  - `after_initialize_kv_cache_tensors`

### 3.6 Fix rollout cache cleanup between steps

File:

- `verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py`

Critical cleanup points:

- clear layer/module `kv_cache`
- clear `attn.impl.key_cache` / `attn.impl.value_cache`
- clear `mla_attn.impl.key_cache` / `mla_attn.impl.value_cache`
- clear worker-side `kv_caches`

This cleanup is what removed the old second-step `resume(kv_cache)` OOM in the
restored good run.

## 4. Recommended Runtime Environment

The restored good mode=1 floor8 baseline used:

- `custom_mode1_kv_headroom=0`
- `kv_cache_init_headroom=0`
- `mode1_native_kv_cap=377344`
- `mode1_keep_fullworld_ep_cache=1`
- `mode1_post_restore_alltoall_warmup=0`

For mode=1 by default:

- `VLLM_ASCEND_REGISTER_CUSTOM_MODELS=1`

Set `VLLM_ASCEND_REGISTER_CUSTOM_MODELS=0` only when explicitly comparing
against the native path.

Recommended mode=1 zero-headroom baseline:

```bash
export VLLM_ASCEND_REGISTER_CUSTOM_MODELS=${VLLM_ASCEND_REGISTER_CUSTOM_MODELS:-1}
export VLLM_ASCEND_ELASTIC_EXECUTION_MODE=1
export VLLM_ASCEND_ELASTIC_MIN_COMPUTE_GROUP_SIZE=8  # or 4 / 2 / 1

export VLLM_ASCEND_CUSTOM_MODE1_KV_MATERIALIZE_HEADROOM_BYTES=0
export VLLM_ASCEND_KV_CACHE_INIT_HEADROOM_BYTES=0
export VLLM_ASCEND_MODE1_PARITY_MAX_KV_TOKENS=377344

export VLLM_ASCEND_MODE1_PARITY_KEEP_FULLWORLD_EP_CACHE=1
export VLLM_ASCEND_MODE1_PARITY_KEEP_MC2_GROUP_CACHE=0
export VLLM_ASCEND_MODE1_PARITY_POST_RESTORE_ALLTOALL_WARMUP=0
export VLLM_ASCEND_MODE1_PARITY_POST_SHRINK_WARMUP=0
export VLLM_ASCEND_CUSTOM_MODE1_KV_DIAG=1
```

Leave the generic headroom env vars unset or explicitly zero:

```bash
export VLLM_ASCEND_POST_SHRINK_MOE_DISPATCH_HEADROOM_BYTES=0
export VLLM_ASCEND_POST_RESTORE_DP_HEADROOM_BYTES=0
export VLLM_ASCEND_POST_RESTORE_EP_HEADROOM_BYTES=0
export VLLM_ASCEND_POST_RESTORE_MOE_DISPATCH_HEADROOM_BYTES=0
export VLLM_ASCEND_FIRST_LIVE_PREFILL_HEADROOM_BYTES=0
export VLLM_ASCEND_EXTRA_ELASTIC_SAFETY_HEADROOM_BYTES=0
```

## 5. Known Good Runtime Signatures

Look for these logs:

- `Mode1 fixed-slot parity init ... parity_ready=True`
- `Capping custom mode1 parity KV blocks ... capped_tokens=377344`
- `GPU KV cache size: 377,344 tokens`
- `Maximum concurrency ... 21.68x`
- `Elastic parallel shrink done ... dp_size=8 ep_size=8`
- `Elastic parallel stale MC2 cache dropped across restore ...`
- `Elastic parallel stale group cache dropped ...`
- `Elastic parallel restore done ... dp_size=16 ep_size=16`

Healthy second-step memory signature:

- `before_initialize_kv_cache_tensors total_allocated` should be about
  `22.9~23.2 GB`

If second-step `before_initialize_kv_cache_tensors` returns to
`30.8~32 GB`, expect the old `370 MiB` KV materialization OOM to return.

## 6. 2026-06-04 Regression Diagnosis

New failing log:

- `wjeagerqwen30b-a3b-with_draft_breakdown_20260604141535589521380_elastic.txt`

Successful reference log:

- `wjeagerqwen30b-a3b-with_draft_breakdown_20260604001522700563326_elastic.txt`

### 6.1 What is still correct in the failing run

- floor8 zero-headroom KV planning is still active
- `GPU KV cache size: 377,344 tokens`
- step1 completes
- shrink / restore completes once

So the regression is not:

- loss of the parity KV cap
- loss of the headroom skip policy
- loss of the floor8 mode=1 shrink / restore path itself

### 6.2 Where it fails

It fails on step2:

- `resume(tags=["kv_cache"])`
- inside `initialize_kv_cache_tensors`
- OOM when allocating about `370 MiB`

### 6.3 Memory comparison: good vs failing run

| Stage | Good run | Failing run | Delta | Main interpretation |
| --- | ---: | ---: | ---: | --- |
| `after_offload_model_weights` total | `7.4~7.7 GB` | `12.8~13.1 GB` | `+5 GB` | non-torch/runtime resources already over-retained |
| `before_update_weights_load` total | `34.3~35.4 GB` | `35.5~36.7 GB` | `+1.2 GB` | slightly heavier entering weight reload |
| `after_update_weights_load` total | `27.2~27.9 GB` | `35.7~36.6 GB` | `+8.5 GB` | old tensors/buffers are not being released after weight load |
| step2 `before_initialize_kv_cache_tensors` total | `22.9~23.2 GB` | `31.5~32.0 GB` | `+8.5~9 GB` | second KV materialization starts from the bad plateau again |

### 6.4 Is it stale parameters or stale communication groups?

Short answer:

- **yes, some stale resources are almost certainly surviving**, but
- **the dominant regression in this run looks more like stale parameter /
  tensor / alias retention than stale communication-group retention alone**.

Reasoning:

#### A. Evidence for stale non-torch resources still surviving

At `after_offload_model_weights`:

- good run: about `7.4~7.7 GB`
- failing run: about `12.8~13.1 GB`

The extra `~5 GB` here is almost entirely non-torch memory.

That strongly suggests some runtime-side resources are still alive after
offload, such as:

- communication-group workspace,
- HCCL / MC2-related non-torch allocations,
- other backend runtime caches.

So it is reasonable to say that **not everything stale was deleted**.

#### B. But the larger second-step regression is torch-side

At `after_update_weights_load`:

- good run settles near `27~28 GB`
- failing run stays near `35.7~36.6 GB`

The extra `~8.5 GB` here is mostly explained by `torch_current`, not by
`non_torch`.

So the larger direct OOM driver is:

- stale weight tensors,
- stale GPU buffer aliases,
- stale module references,
- or another torch-visible parameter/buffer lifetime issue after
  `model.load_weights(weights)`.

### 6.5 Practical conclusion for this regression

This time the problem is **not just "stale communication groups"**.

The evidence supports a two-level diagnosis:

1. **Some stale runtime / communication-side resources do survive offload**
   because `after_offload_model_weights` is already about `5 GB` too high.
2. **The immediate OOM trigger is larger stale torch-side retention after
   `update_weights_load`**, because the successful run drops to `27~28 GB`
   there while the failing run remains at `35~36 GB`.

So if this regression is fixed, the first places to re-check are:

- `verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py`
  - `free_cache_engine()`
  - `offload_model_weights()`
  - `update_weights()`
- all strong references kept by:
  - `gpu_buffers`
  - layer/module KV cache aliases
  - attention / MLA cache objects
  - post-load parameter aliases or staging buffers

Communication-group cleanup is still important, but based on this log it does
not look like the whole `8.5~9 GB` regression can be explained by stale MC2 /
EP / DP groups alone.

## 7. Fast Failure Interpretation Guide

| Log symptom | Likely meaning |
| --- | --- |
| `GPU KV cache size` already below `377,344` in floor8 parity run | KV cap/headroom logic changed |
| generic headroom lines reappear | parity gating is broken or not active |
| step1 succeeds, step2 `resume(kv_cache)` OOM with `370 MiB` request | cache/weight lifetime cleanup regression |
| `after_offload_model_weights` is much higher than `7~8 GB` | stale runtime / non-torch resources survived offload |
| `after_update_weights_load` stays around `35~36 GB` instead of `27~28 GB` | stale torch-side weight/buffer retention after reload |
| shrink / restore communication time becomes hundreds of seconds | full-world communication groups were dropped too early or MC2 is being recreated every cycle |

## 8. Multi-Minute Communication Regression

Symptom:

- a mode=1 run does not immediately OOM,
- but shrink / restore communication suddenly takes hundreds of seconds,
- rollout wall time becomes obviously unreasonable even though the backend is
  still MC2.

Root cause we hit before:

- zero-headroom mode=1 needs to reuse the cached full-world DP / EP / MC2
  communicators when restoring from a shrunken group back to 16 ranks;
- an intermediate version dropped stale group caches immediately after shrink;
- that also destroyed the full-world communication groups that restore was
  supposed to reuse;
- restore then had to recreate MC2 / HCCL resources and re-run expensive
  dispatcher setup / warmup, which produced the multi-minute communication
  regression.

This is different from the OOM-side stale-cache issue:

- for memory, stale floor MC2 must not survive after restore;
- for speed, full-world groups must survive across shrink so restore can reuse
  them;
- the correct lifecycle is therefore asymmetric.

Correct mode=1 policy:

| Group/cache | During shrink | After restore |
| --- | --- | --- |
| full-world DP | keep, because restore will reuse it | current group again |
| full-world EP | keep, because restore will reuse it | current group again |
| full-world MC2 | keep/reuse by rank signature when possible | current group again |
| floor MC2 | current while shrunken | stale; drop after full-world restore |
| stale floor DP/EP | do not accumulate | drop unless explicitly diagnosed |

Important env defaults:

```bash
export VLLM_ASCEND_MODE1_PARITY_KEEP_FULLWORLD_DP_CACHE=1
export VLLM_ASCEND_MODE1_PARITY_KEEP_FULLWORLD_EP_CACHE=1
export VLLM_ASCEND_MODE1_PARITY_KEEP_FULLWORLD_MC2_CACHE=1
export VLLM_ASCEND_MODE1_PARITY_KEEP_MC2_GROUP_CACHE=0
export VLLM_ASCEND_MODE1_PARITY_DROP_STALE_CACHE_AFTER_SHRINK=1
export VLLM_ASCEND_MODE1_PARITY_SINGLE_LIVE_MC2_GROUP=0
export VLLM_ASCEND_DISABLE_ELASTIC_MC2_GROUP_CACHE=0
export VLLM_ASCEND_MODE1_PARITY_POST_RESTORE_ALLTOALL_WARMUP=0
```

`VLLM_ASCEND_MODE1_PARITY_DROP_STALE_CACHE_AFTER_SHRINK` is only the
post-shrink immediate sweep switch. It is not the post-restore floor-MC2 cleanup
switch. Restore always runs:

```python
_drop_stale_cached_elastic_parallel_groups(tuple(range(world_size)))
```

So:

- `DROP_STALE_CACHE_AFTER_SHRINK=0`: skip the immediate sweep after shrink, but
  still clean floor stale groups after restore.
- `DROP_STALE_CACHE_AFTER_SHRINK=1`: run an extra selective sweep after shrink;
  this must preserve full-world DP / EP / MC2 restore caches.

The dangerous settings for this regression are:

```bash
export VLLM_ASCEND_MODE1_PARITY_KEEP_FULLWORLD_DP_CACHE=0
export VLLM_ASCEND_MODE1_PARITY_KEEP_FULLWORLD_EP_CACHE=0
export VLLM_ASCEND_MODE1_PARITY_KEEP_FULLWORLD_MC2_CACHE=0
export VLLM_ASCEND_MODE1_PARITY_SINGLE_LIVE_MC2_GROUP=1
export VLLM_ASCEND_DISABLE_ELASTIC_MC2_GROUP_CACHE=1
```

These force the runtime closer to "destroy and recreate full-world
communicators every cycle", which is exactly what made restore communication
blow up to hundreds of seconds. `DROP_STALE_CACHE_AFTER_SHRINK=1` is expected
only when the keep-full-world guards above are active; it should mean
"selectively drop stale floor-world groups", not "drop every non-current
signature".

Logs to check:

- healthy shrink:
  `Elastic parallel shrink phase breakdown ... rebuild_mc2_ms=...`
- healthy restore:
  `Elastic parallel restore done ... rebuild_ms=...`
- expected stale cleanup:
  `Elastic parallel stale MC2 cache dropped across restore ...`
- suspicious path:
  `Elastic custom mode1 MC2 single-live-group destroying old group before rebuild`
- suspicious env banner:
  `mode1_keep_fullworld_*_cache=0` or single-live / disable-MC2-cache enabled

If this happens on another machine, first compare the run banner and confirm
that the mode=1 group-cache envs above match the known-good defaults. Then
check whether restore is rebuilding MC2 from scratch every step instead of
reusing the full-world rank signature.

## 9. Minimal Migration Order

If porting this behavior into an older branch:

1. keep custom Qwen3 on old `fused_moe`
2. add true mode=1 lightweight loaded-slot path in `fused_moe.py`
3. add `lossless_mode1_native_parity_ready`
4. parity-gate worker headroom skips
5. align MC2 / EP cache lifecycle with native
6. use explicit mode=1 KV token cap
7. restore rollout cache cleanup between steps
8. validate second-step `before_initialize_kv_cache_tensors` is back to
   `22.9~23.2 GB`

## 10. Note

The three older mode=1 reference documents were merged into this file and can be removed to avoid duplicated maintenance. This file is intended to be the single maintained reference going forward.
