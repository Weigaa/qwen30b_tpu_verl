# Mode1 Floor2 No-Trim KV Cap Pitfall

Date: 2026-07-03

## Context

After the floor4 shrink-aware path became stable, we tested a deeper
`16 -> 8 -> 4 -> 2` shrink path under the natural runtime policy. The goal was
to determine whether floor2 is practically usable and, if so, what KV-cache
budget can be used without triggering either memory failures or large step2+
latency regressions.

The relevant fast-validation setting is:

```text
VLLM_ASCEND_SHRINK_AWARE_TARGET_POLICY=natural
VLLM_ASCEND_SHRINK_AWARE_STAGES=8,4,2
VLLM_ASCEND_ELASTIC_MIN_COMPUTE_GROUP_SIZE=2
MIN_ADAPTIVE_FLOOR=2
FORCE_SELECTED_FLOOR=2
VERL_ELASTIC_TAIL_VALIDATE_LEVEL_TOKENS=256,512,640,768,896
MAX_RESPONSE_LENGTH=896
```

The test is a threshold-controlled smoke test, not a full-length quality run.
Its purpose is to isolate floor2 communication, restore, reload, and KV-cache
behavior while keeping runtime bounded.

## Initial Symptom

The first floor2 runs showed a sharp latency jump after the first rollout step.
With reload allocator trimming enabled, the observed rollout times were:

| step | rollout output time |
| ---: | ---: |
| 1 | 150.07 s |
| 2 | 224.48 s |
| 3 | 233.32 s |
| 4 | 235.91 s |
| 5 | 242.17 s |

This looked at first like a post-floor2 restore or full-world MoE/MC2 dispatcher
state problem. We considered several possible causes:

- adaptive KV resize or restore overhead;
- insufficient non-KV workspace after floor2;
- full-world MoE/MC2 dispatcher taking a slower path after restore;
- reload path accidentally loading or using all 64 physical expert slots instead
  of the active 8 experts per rank;
- allocator/cache cleanup during post-reload weight processing.

## Instrumentation

We added timing and diagnostic logs around:

- rollout weight update and model reload;
- `model.load_weights` call and post-load synchronization;
- Qwen3-MoE `load_weights` internals;
- MoE `process_weights_after_loading`;
- full-world restore group/cache state;
- MC2 dispatcher warmup;
- runtime slot-map state after restore.

The most important split was:

```text
update_total_s
model_load_s
model_load_call_s
model_load_sync_s
post_process_s
infer_s
```

This showed that the large step2+ latency was not primarily from inference or
from the load call itself. It concentrated in MoE post-processing:

| metric | trim enabled, step2+ | trim disabled, step2+ |
| --- | ---: | ---: |
| `update_total_s` | 69-71 s | 10.6-11.0 s |
| `model_load_s` | 61-63 s | 3.0-3.1 s |
| `post_process_s` | 59-61 s | 1.0-1.1 s |
| `infer_avg_s` | 85-99 s | 60-66 s |

## Root Cause

The dominant floor2 slowdown was caused by the reload allocator trim path, not
by floor2 computation itself.

In `vllm_ascend/ops/fused_moe.py`, the floor2 reload path defaulted to trimming
the NPU allocator around MoE weight post-processing:

```text
VLLM_ASCEND_MODE1_RELOAD_TRIM_ALLOCATOR=1  # default for floor <= 2
VLLM_ASCEND_MODE1_RELOAD_SYNC_ON_TRIM=1
```

The trim operation performs synchronization, Python garbage collection, and NPU
cache release:

```text
torch.npu.synchronize()
gc.collect()
torch.npu.empty_cache()
```

It was executed repeatedly during `process_weights_after_loading`, effectively
many times per reload across all MoE layers. This produced the stable ~60 s
post-process penalty seen after step1.

Disabling trim removed the pathological step2+ slowdown:

```text
VLLM_ASCEND_MODE1_RELOAD_TRIM_ALLOCATOR=0
VLLM_ASCEND_MODE1_RELOAD_SYNC_ON_TRIM=0
```

With cap `50000` and no trim, the rollout times became:

| step | rollout output time |
| ---: | ---: |
| 1 | 150.25 s |
| 2 | 141.67 s |
| 3 | 137.10 s |
| 4 | 143.05 s |
| 5 | 142.28 s |

This proves that the large step2+ latency was a cleanup-policy artifact. It was
not an inherent floor2 performance cost.

## False Leads Ruled Out

### No-op KV Resize

The no-op or repeated adaptive KV resize path was not the primary cause. Even
with a very conservative KV cap, the slow trim-enabled runs still showed the
same post-process penalty.

### Loading 64 Experts on the Critical Compute Path

Floor2 uses larger physical capacity for redundant expert slots, but logs show
that the active full-world execution shape still uses the expected active expert
ownership, not all 64 physical slots as active compute. The slow path was tied
to post-load cleanup and allocator trimming, not to full active computation over
64 experts per rank.

### Step2+ Latency at Higher Caps

At higher no-trim caps, failures did not appear as completed-but-slow steps.
They occurred inside step2 before `rollout_output_time_s` was emitted.

## No-Trim KV-Cap Sweep

We then swept the floor2 KV-cache budget with no trim enabled. The sweep script
was:

```text
run_mode1_local_length_sorted_e2e_adaptive_floor2_natural_notrim_cap_sweep.sh
```

Pass criteria:

- all 5 rollout steps complete;
- no `OOM`, `RuntimeError`, `Preempting`, or `Memory_Allocation` appears in the
  log;
- step2-step5 avoid the large latency regression:

```text
max(step2..step5 rollout_output_time_s) <= max(180 s, step1 + 45 s)
```

Sweep result:

| floor2 KV cap | result | rollout output time |
| ---: | --- | --- |
| 147456 | fail | step1 only: 152.13 s |
| 139264 | fail | step1 only: 151.02 s |
| 131072 | pass | 148.26, 138.93, 138.09, 145.25, 143.85 s |

The two higher caps did not fail by slowly completing step2. They failed before
step2 could emit `rollout_output_time_s`.

Observed failure signature:

```text
path string is NULL
```

For cap `147456`, vLLM also dumped V1 engine input and scheduler state after the
step2 shrink/restore path. The logs did not show a clean Python traceback,
explicit `OOM`, or `Preempting`. The most plausible interpretation is that the
larger KV reservation left insufficient workspace for the floor2
HCCL/MC2/TBE/runtime path, causing a lower-level runtime failure or hang during
step2 execution.

## Current Conclusion

For the threshold-controlled natural floor2 smoke test:

```text
largest verified stable no-trim floor2 KV cap = 131072 tokens
```

This value satisfies both correctness and latency criteria:

- all 5 steps complete;
- floor2 is reached;
- no fatal error appears in the log;
- step2-step5 remain around 138-145 s, with no post-step1 slowdown.

## Recommended Runtime Policy

For floor2 natural-policy experiments, use:

```text
VLLM_ASCEND_MODE1_RELOAD_TRIM_ALLOCATOR=0
VLLM_ASCEND_MODE1_RELOAD_SYNC_ON_TRIM=0
VLLM_ASCEND_MODE1_PARITY_MAX_KV_TOKENS_FLOOR2=131072
FLOOR_KV_CAPS=2:131072,4:280576,8:377344,16:377344
```

`131072` should be treated as the current validated floor2 cap for this short
validation setting. If full-length runs need a different response cap or batch
shape, the cap should be revalidated, because floor2 workspace demand is
sensitive to HCCL/MC2/TBE runtime behavior.

## Follow-Up

The next useful tests are:

1. Run a finer no-trim search between `131072` and `139264`, for example
   `135168` and `137216`.
2. Validate whether the same no-trim cap holds under full-length floor2 runs.
3. If memory failures reappear, replace repeated per-layer trim with a bounded
   policy such as "trim at most once per reload" rather than re-enabling the
   old per-layer trim path.

## 2026-07-05 Dynamic 2-2-4 Residual Storage Fix

New symptom:

- Fixed floor2 natural run with `KV cap=131072` can complete 5 steps and reaches
  clean restore memory state.
- Dynamic `2-2-4` probe fails around the following actor update with OOM even
  though the same floor2 cap is used.

Key comparison:

- Fixed floor2 at full-world restore entry:
  `torch_current ~= 0.2 GiB`, `free ~= 46-47 GiB`.
- Dynamic 2-2-4 at full-world restore entry:
  ranks 0-13 keep `torch_current ~= 27 GiB`, while ranks 14-15 are clean.

Interpretation:

The failure is not primarily caused by the floor4 KV cap. The dynamic path keeps
floor2 loaded expert NPU storage alive before returning to trainer/actor update.
The most suspicious reference was the runtime prefix view: it is a slice of the
loaded parameter, so checking `runtime_weight is old_weight` misses it and the
old storage remains pinned.

Code changes:

- `offload_lossless_loaded_weights_to_cpu()` now clears runtime weight views
  whenever there is no independent runtime buffer, not only when the runtime
  object is identical to the old parameter.
- full-world restore now performs a post-warmup loaded-weight offload for mode1
  low-floor runs, then logs module count, estimated NPU bytes, and offload time.
- the dynamic `2-2-4` probe enables aggressive stale parameter release and
  post-restore loaded-weight offload by default.

Validation target for the next run:

- `after_all_post_restore_warmups` should report nonzero
  `loaded_weight_offloaded_modules`;
- dirty ranks should disappear or drop close to fixed floor2 levels;
- actor update should no longer OOM after the floor2 rollout.

Follow-up from run `20260705130142`:

- The post-restore offload hook did execute and reported 48 modules per rank.
- However, `estimated_npu_bytes=0` and dirty ranks still kept
  `torch_current ~= 29 GiB`.
- This means the resident storage was no longer visible through the canonical
  `w13_weight/w2_weight` parameters, but was still held by shrink/runtime-only
  references.

Second fix:

- Count runtime buffers/views in the offload diagnostics.
- Call `release_mode1_full_world_transient_state()` before post-restore offload
  so `runtime_w13_buffer`, `runtime_w2_buffer`, export-slot aliases, saved
  prefix refs, and dispatch state are dropped before actor update.
- Make `offload_lossless_loaded_weights_to_cpu()` clear runtime buffers/views
  unconditionally after loaded weights are offloaded.

Additional diagnostics:

- `Mode1 full-restore offload storage snapshot` logs tracked NPU storage by
  category before release, after release, and after GC.
- Categories:
  `loaded_param`, `runtime_view`, `runtime_buffer`, `saved_prefix`,
  `cpu_shadow`, and `p2p_alias_cache`.
- If `post_gc_tracked_npu_bytes` is near zero but `torch_current` is still high,
  the remaining storage is outside these MoE module fields and the next step is
  enabling the heavier live tensor scan.

Follow-up from run `20260705132334`:

- The residual storage was conclusively identified as
  `_lossless_p2p_alias_cache`.
- `before_release` showed tens of GiB per rank in `p2p_alias_cache`, while
  `loaded_param=0`, `runtime_view=0`, and `runtime_buffer=0`.
- After release and GC, the tracked NPU storage dropped to zero and restore
  memory returned to a clean state (`torch_current` near zero, non-torch around
  the baseline runtime footprint).
- The run then failed at the next rollout `update_weights` path:
  `IndexError: index 0 is out of bounds for dimension 0 with size 0` in
  `fused_moe/layer.py::weight_loader`.

Interpretation:

The memory leak fix overreached. The post-restore cleanup called
`offload_lossless_loaded_weights_to_cpu()`, which replaces canonical
`w13_weight/w2_weight` with zero-row CPU Parameters. This is safe only if the
next path restores or reallocates those Parameters before `model.load_weights`.
In this SPMD rollout path, `model.load_weights` indexes the existing parameter
rows directly, so the zero-row Parameter breaks the next reload.

Third fix:

- Treat full-world post-restore cleanup as a transient-cache cleanup by
  default, not canonical loaded-weight offload.
- `release_mode1_full_world_transient_state()` now clears
  `_lossless_p2p_alias_cache` directly.
- Worker cleanup now logs `Mode1 full-restore transient cleanup` and defaults
  `canonical_offload_enabled=0`.
- Canonical loaded-weight CPU offload remains available only through the
  explicit opt-in environment variable
  `VLLM_ASCEND_MODE1_OFFLOAD_CANONICAL_LOADED_WEIGHTS_AFTER_FULL_RESTORE=1`.

Validation target for the next run:

- `before_release` may still show `p2p_alias_cache` bytes.
- `after_release` and `after_gc` should drop those bytes to zero.
- The log line should show `canonical_offload_enabled=0` and
  `offloaded_modules=0`.
- Step 2 should no longer fail in `model.load_weights` with a zero-row MoE
  parameter.

## 2026-07-05 Upward Floor Resize OOM

After the transient-cache cleanup fix, the dynamic probe moved further: fixed
floor2 steps could run, and the `2 -> 2 -> 4` probe reached the upward floor
transition. The next failure was an OOM during step3, when switching from
floor2 to floor4:

```text
old_tokens=131072
target_tokens=280576
previous_floor=2
target_floor=4
```

The traceback pointed into the upward cleanup path:

```text
resize_kv_cache_for_mode1_step
  -> prepare_mode1_step_floor_for_kv_resize
  -> _cleanup_mode1_upward_floor_residue_for_kv_resize
  -> shrink_lossless_loaded_weights_to_primary
  -> _allocate_formatted_buffer_like(old_w13, target_rows)
```

The allocator had only about 149 MiB free while the first compacted `w13`
allocation needed about 194 MiB. Earlier changes had already reduced the
compact peak by allocating/copying/replacing `w13` and `w2` sequentially, so the
remaining problem was not the compact implementation itself.

Root cause:

- `resize_kv_cache_for_mode1_step()` entered `floor_prepare_start` before
  releasing the previous step's KV cache.
- During an upward transition such as `2 -> 4`, `floor_prepare` compacts the
  floor2 MoE runtime/storage layout back toward the floor4 layout.
- That compact requires temporary NPU buffers.
- Because the old floor2 KV cache was still resident, the temporary compact
  buffer and old KV cache overlapped in HBM and caused OOM.

The old ordering was:

```text
resize_start
floor_prepare_start
  upward cleanup / MoE compact
floor_prepare_done
clear_old_kv_start
clear_old_kv_done
plan_new_kv
allocate_new_kv
initialize_cache
```

This ordering was acceptable when the compact peak fit in the free workspace,
but floor2 left too little slack for `2 -> 4`.

## Current Floor-Switching Order

The resize path now distinguishes upward floor transitions:

```text
upward_floor_transition =
    previous_floor is not None
    and target_floor is not None
    and target_floor > previous_floor
    and previous_floor > 0
```

For non-upward transitions, the normal order remains:

```text
resize_start
floor_prepare_start
floor_prepare_done
clear_old_kv_start
clear_old_kv_done
clear_stale_param_dicts
plan_new_kv
allocate_new_kv
initialize_cache
```

For upward transitions such as `2 -> 4`, `2 -> 8`, or `4 -> 8`, the old KV cache
is released before floor prepare:

```text
resize_start
clear_old_kv_before_floor_prepare_start
clear_old_kv_before_floor_prepare_done
floor_prepare_start
  upward cleanup / MoE compact
floor_prepare_done
clear_old_kv_skipped  # already released before floor prepare
clear_stale_param_dicts
plan_new_kv
allocate_new_kv
initialize_cache
```

The behavior is controlled by:

```text
VLLM_ASCEND_MODE1_CLEAR_OLD_KV_BEFORE_UPWARD_FLOOR_PREPARE=1
```

This is enabled by default. The old KV cache is only cleared early when the
engine is already at a safe step boundary: no live scheduler requests and an
empty batch queue. Therefore no active request KV migration is needed.

## Allocation and Release Semantics

The step-boundary resize now uses the following resource order:

1. Check that the scheduler has no live requests and the engine batch queue is
   empty.
2. Compute the effective target KV tokens and target blocks.
3. If this is an upward floor transition, release old KV first with
   `clear_kv_cache_for_resize`.
4. Prepare the target floor:
   - prune unneeded floor communication groups for the current policy;
   - for upward transitions, clean residue from the lower floor;
   - compact MoE loaded/runtime slots toward the target floor layout;
   - precreate the required floor groups without preloading planned expert
     slots in natural mode.
5. Skip the regular old-KV release if it already happened before floor prepare.
6. Clear stale parameter dictionaries if enabled.
7. Query currently available KV memory.
8. Plan the new KV cache.
9. Allocate the new KV cache with `initialize_from_config`.
10. Initialize scheduler/cache state with the new block count.

The key principle is:

```text
old KV must not overlap with upward floor compact temporary buffers
```

but:

```text
new KV should still be allocated only after floor prepare and stale cleanup
```

This keeps the peak lower on both sides: the old cache is gone before compact,
and the new larger floor4 cache is not allocated until after compact and stale
cleanup finish.

## Validation Result

The validation run

```text
mode1_dynamic_floor2_to_floor4_kv_probe
log: wjqwen30b-a3b-record_graph_save4eagle3_20260705154923.txt
```

completed all three probe steps:

| step | target floor | target KV tokens | rollout output time |
| ---: | ---: | ---: | ---: |
| 1 | 2 | 131072 | 96.54 s |
| 2 | 2 | 131072 | 85.98 s |
| 3 | 4 | 280576 | 121.26 s |

The log confirms the new order:

```text
clear_old_kv_before_floor_prepare_start target_floor=4 previous_floor=2
clear_old_kv_before_floor_prepare_done target_floor=4 previous_floor=2
floor_prepare_start target_floor=4
clear_old_kv_skipped reason=already_cleared_before_floor_prepare
```

The run reached `global_step_3`, saved the checkpoint, and printed:

```text
Epoch 0 completed in 576.80 seconds.
```

This validates that the OOM was caused by an ordering peak, not by the final
floor4 KV budget itself.

## Remaining Cost

The OOM is fixed, but the upward transition still has measurable overhead.
For the validated run, the `2 -> 4` resize took about 51 s. The dominant part is
the upward cleanup:

```text
upward_cleanup_ms ~= 33-41 s
cleanup_restore_layout_ms ~= 22-30 s
cleanup_stale_param_ms ~= 4.6-4.7 s
cleanup_sync_ms ~= 5.4-5.7 s
```

So the current state is:

- correctness: fixed for the 2-2-4 probe;
- final floor4 KV target: satisfied at 280576 tokens;
- remaining optimization target: reduce upward cleanup/compact latency, not
  memory correctness.

## 2026-07-05 Upward Cleanup Latency Reduction Trial

The first correct `2 -> 4` run still spent about 51 s inside step-boundary KV
resize. The largest component was not old-KV release; it was upward cleanup and
MoE compact:

```text
upward_cleanup_ms ~= 33-41 s
cleanup_restore_layout_ms ~= 22-30 s
cleanup_stale_param_ms ~= 4.6-4.7 s
cleanup_sync_ms ~= 5.4-5.7 s
cleanup_post_compact_sync_ms ~= 0.3-0.4 s
```

Inspection found several conservative allocator trims in the upward compact
path:

- optional `empty_cache()` before compact;
- optional `empty_cache()+synchronize()` every N compacted layers;
- `empty_cache()` between compacting `w13` and `w2` in each MoE layer;
- optional post-compact `empty_cache()+synchronize()`;
- a final cleanup `empty_cache()+synchronize()`.

After the old KV cache is released before upward floor prepare, the compact path
has more HBM headroom, so repeated trims are no longer required for the fast
probe. The probe script now defaults to a lighter policy:

```text
VLLM_ASCEND_MODE1_EMPTY_CACHE_BEFORE_UPWARD_COMPACT=0
VLLM_ASCEND_MODE1_EMPTY_CACHE_DURING_UPWARD_COMPACT=0
VLLM_ASCEND_MODE1_EMPTY_CACHE_AFTER_UPWARD_COMPACT=0
VLLM_ASCEND_MODE1_EMPTY_CACHE_BETWEEN_COMPACT_TENSORS=0
VLLM_ASCEND_MODE1_UPWARD_CLEANUP_FINAL_EMPTY_CACHE=0
VLLM_ASCEND_MODE1_UPWARD_COMPACT_RELEASE_EVERY_LAYERS=48
```

This does not skip the required final memory check. The later
`get_mode1_resize_available_kv_memory()` call still performs one
`empty_cache()+synchronize()` before planning the new KV cache. The intent is to
collapse many per-layer/per-batch trims into one resize-time memory query.

Expected validation signal:

- the run should still complete the `2 -> 2 -> 4` probe;
- `clear_old_kv_before_floor_prepare_*` should still appear before
  `floor_prepare_start`;
- `cleanup_pre_compact_empty_cache_ms`, `cleanup_post_compact_sync_ms`, and
  `cleanup_sync_ms` should drop close to zero;
- `cleanup_restore_layout_ms` should drop if per-layer
  `EMPTY_CACHE_BETWEEN_COMPACT_TENSORS=0` was a major contributor;
- if OOM returns, restore the trims one by one in this order:
  1. `VLLM_ASCEND_MODE1_EMPTY_CACHE_BETWEEN_COMPACT_TENSORS=1`
  2. `VLLM_ASCEND_MODE1_UPWARD_CLEANUP_FINAL_EMPTY_CACHE=1`
  3. `VLLM_ASCEND_MODE1_EMPTY_CACHE_AFTER_UPWARD_COMPACT=1`

Validation result from run `20260705163708`:

| step | target floor | target KV tokens | rollout output time |
| ---: | ---: | ---: | ---: |
| 1 | 2 | 131072 | 96.55 s |
| 2 | 2 | 131072 | 87.58 s |
| 3 | 4 | 280576 | 95.16 s |

The run completed `global_step_3` and printed:

```text
Epoch 0 completed in 557.17 seconds.
```

The `2 -> 4` resize also improved substantially:

| metric | before | after |
| --- | ---: | ---: |
| total KV resize elapsed | ~50-52 s | ~22-24 s |
| `upward_cleanup_ms` | ~33-41 s | ~5.5-5.9 s |
| `cleanup_restore_layout_ms` | ~22-30 s | ~0.7 s |
| `cleanup_sync_ms` | ~5.4-5.7 s | ~0.0 s |
| `cleanup_post_compact_sync_ms` | ~0.3-0.4 s | ~0.0 s |

The remaining upward cleanup time is now mostly stale parameter dictionary
cleanup:

```text
cleanup_stale_param_ms ~= 4.8-5.1 s
```

Conclusion: after old-KV early release, collapsing repeated allocator trims into
the later resize memory query is safe for this probe and removes most of the
previous `2 -> 4` transition latency.

## 2026-07-05: `2 -> 2 -> 4 -> 4` step4 OOM root cause

The short probe was extended to `2,2,4,4,16` to validate a same-floor `4 -> 4`
step before running the full natural experiment. The failure happened at the
start of step4, after step3 had completed successfully.

The important observation is that the cleanup after step3 did run:

```text
rollout_mode_before_stage_path_resume_kv_cache:
  torch_current ~= 0.2-0.3 GB
  non_torch ~= 15.0-15.6 GB
  free_bytes ~= 49.7-50.5 GB
```

So the root cause was not a totally missing rollout release. The OOM was caused
by the rebuild order in the staged rollout path:

```text
stage weights to CPU
offload actor weights
resume(kv_cache)
resume(weights)
update_weights(...)
```

For the first `2 -> 4` step, `resume(kv_cache)` still brought back the small
floor2 KV cache first:

```text
after resume_kv: torch_current ~= 13.1 GB
adaptive resize inside generate: 131072 -> 280576 tokens
```

This worked because rollout weights were loaded before the floor4 KV cache was
expanded to the final 280576-token budget.

For the next `4 -> 4` step, `resume(kv_cache)` rebuilt the full floor4 KV cache
immediately:

```text
after resume_kv: torch_current ~= 27.8 GB
free_bytes ~= 22.3-22.9 GB
```

Then `resume(weights)` failed while allocating a formatted NPU rollout weight
buffer:

```text
Failed to allocate formatted rollout GPU buffer for
model.layers.34.mlp.experts.w13_weight format=FRACTAL_NZ
NPU out of memory. Tried to allocate 386 MiB; only ~132 MiB free.
```

The final floor4 state is feasible, as shown by the first floor4 step. The bad
case is the transient peak caused by building the full floor4 KV cache before
rollout weights are onloaded.

Fix strategy:

- Keep trainer-mode rollout release enabled; do not preserve KV across train.
- Keep CPU staging of exported weights so actor weights are not resident during
  KV rebuild.
- During the staged path, bootstrap `resume(kv_cache)` with the floor2 KV cap
  (`131072`) when the target cap is larger.
- Immediately restore `VLLM_ASCEND_MODE1_PARITY_CURRENT_KV_TOKENS` to the real
  target after `resume(kv_cache)`.
- Let the existing adaptive resize path expand KV to the real step target
  during rollout generation.

The expected validation signal is:

```text
rollout_mode_stage_path_bootstrap_kv_resume target_kv=280576 bootstrap_kv=131072
rollout_mode_after_stage_path_resume_kv_cache_memory torch_current ~= 13 GB
rollout_mode_stage_path_restore_target_kv_env target_kv=280576
Mode1 adaptive KV cache resized ... old_tokens=131072 target_tokens=280576
```

## 2026-07-06: reload capacity must be a step-level invariant

The later `2 -> 2 -> 4 -> 4` probes showed that the bootstrap-KV ordering alone
is not sufficient.  The first `2 -> 4` floor4 step succeeds because the upward
cleanup path compacts the MoE loaded tensors to the target floor4 capacity:

```text
cleanup_compacted_layers=48
cleanup_cleared_stale_param_bytes ~= 28.99 GB
loaded_local=32 loaded_capacity=32 weight_rows=32
```

The next `4 -> 4` step is more revealing.  Since the floor does not change,
upward cleanup is skipped.  After full-world restore the memory is clean, but
after `update_weights` / `process_weights_after_loading` the rollout model can
again hold a 64-slot-equivalent loaded storage, which is valid for floor2 but
invalid for a floor4 KV budget.  This is why the second floor4 step fails even
though the immediately previous floor4 step succeeded.

The correct invariant is:

```text
after update_weights:
  floor2 -> loaded_capacity=64
  floor4 -> loaded_capacity=32
  floor8 -> loaded_capacity=16
  floor16 -> loaded_capacity=8
```

This invariant must be enforced in the weight reload path itself, not only in
the upward floor cleanup path.  The fix is to set
`VLLM_ASCEND_MODE1_WEIGHT_RELOAD_TARGET_FLOOR` from the planned step floor before
`update_weights`, and have `process_weights_after_loading` compact/check the
loaded tensors against that target floor immediately after reload.  Same-floor
steps then get the same memory contract as floor-changing steps, avoiding hidden
64-slot residue that only appears when the next KV rebuild asks for the floor4
budget.

One subtle bug: `process_weights_after_loading` used to run only when
`invalidate_lossless_runtime_state_for_reload()` returned true.  In same-floor
or already-restored paths, invalidation may return false even though
`update_weights` still refreshed the parameters.  That silently skipped the
target-floor capacity invariant.  For mode1/lossless modules, post-reload
processing must run on every `load_weights` call, independent of whether runtime
state was invalidated.

## 2026-07-06: final step4 failure root cause was DP metadata HCCL workspace

After fixing the obvious `2 -> 4` and `4 -> 4` memory-contract problems above,
one more failure remained. The run still failed when entering the next rollout
step, and at first it looked like another floor4 KV-cache or expert-storage
leak. The decisive traceback was:

```text
vllm_ascend/worker/model_runner_v1.py:_sync_metadata_across_dp
  max_tokens_across_dp = int(torch.max(num_tokens_across_dp).item())

current working operator name is HcclAllreduce
Memory_Allocation_Failure(EL0004):
  Failed to allocate [size:1678770176] bytes of NPU memory.
```

The failed operation was not a KV-cache allocation and not MoE expert compute.
It was the first real engine execution after the floor transition, where vLLM
syncs a few request metadata values across DP ranks. In mode1 this metadata
sync still used an NPU HCCL `all_reduce` by default. Because mode1 also skipped
post-shrink DP allreduce warmup by default, the HCCL workspace for that tiny
metadata collective was allocated lazily at the worst possible time: after the
rollout KV cache and rollout weights had already been rebuilt.

The requested HCCL workspace was about 1.68 GB. That is large relative to the
remaining safety margin after floor4/floor16 KV and weight reload, so it showed
up as a step4 OOM even though the floor4 KV budget itself was feasible.

The final root cause hierarchy is therefore:

1. Earlier failures exposed real transient memory-ordering and reload-capacity
   bugs. Those had to be fixed first:
   - do not rebuild large KV before rollout weights are staged safely;
   - enforce `loaded_capacity = world_size / target_floor * local_experts`
     after every `update_weights`, including same-floor steps.
2. The last remaining step4 failure was caused by NPU DP metadata sync using
   HCCL and lazily allocating a large communication workspace during the first
   post-shrink decode.

The correct fix is to route mode1 DP metadata sync through the CPU DP group:

```text
VLLM_ASCEND_MODE1_CPU_DP_METADATA_SYNC=1
```

This is safe because the metadata path only exchanges small integer values such
as `num_input_tokens` / maximum tokens across DP ranks. It is not on the
large-tensor decode or MoE communication path.

Implementation notes:

- `vllm_ascend/worker/model_runner_v1.py` now defaults mode1
  `VLLM_ASCEND_MODE1_CPU_DP_METADATA_SYNC` to `1`.
- `run_mode1_dynamic_floor2_to_floor4_kv_probe.sh` exports the same flag
  explicitly.
- `run_mode1_dynamic_length_aware_adaptive_floor2_natural_tailguard_reuse_epoch0_2epoch.sh`
  also exports the flag for the full natural validation path.

Validation run:

```text
log: mode1_dynamic_floor2_to_floor4_kv_probe/epoch_001_mode1_natural/logs/
     wjqwen30b-a3b-record_graph_save4eagle3_20260706124351.txt

actual floor/KV sequence:
  step1 -> floor2,  KV=131072
  step2 -> floor2,  KV=131072
  step3 -> floor4,  KV=280576
  step4 -> floor4,  KV=280576
  step5 -> floor16, KV=380800

rollout gen time:
  step1 111.18 s
  step2  97.10 s
  step3  95.52 s
  step4  93.85 s
  step5  66.34 s
```

The run completed all five steps and saved `global_step_5`. No
`HcclAllreduce`, `Memory_Allocation_Failure`, or `Failed to allocate` appeared
after enabling CPU DP metadata sync. There was also no measurable slowdown from
moving this metadata exchange to CPU; step2-step4 remained stable and slightly
decreased.

Takeaway:

For dynamic floor2/floor4 mode1 runs, a feasible KV budget is not enough. Any
post-shrink NPU collective that allocates workspace lazily can invalidate the
memory plan. Tiny control-plane collectives should stay on CPU, while NPU HCCL
workspace must either be warmed before KV sizing or included explicitly in the
KV-cap budget.

## 2026-07-07: floor2 correctness fix for expert weight slot writes

After the memory and HCCL-workspace issues were fixed, the floor2 path exposed a
separate correctness bug.  The short probe could complete, but the generated
responses were abnormal: many responses ran to the full response budget, and
some outputs degenerated into repeated punctuation or otherwise low-quality
text.  The scheduler and threshold-control plan were correct.  The failure was
in the expert weight reload / direct import chain.

The useful diagnostic question was:

```text
Did FusedMoE.weight_loader receive nonzero source weights for experts 1..7?
If yes, why did the target slots remain zero after writing param.data[expert_id]?
```

The logs showed that the source weights for experts 1..7 were nonzero, but the
target NPU formatted expert rows could still remain zero after writeback.  This
made `log2phy`, `expert_map`, and capacity checks look superficially valid while
the actual execution slots contained wrong data.  In other words, metadata was
consistent, but the loaded expert storage was not.

### Root cause

The buggy write pattern dropped the expert dimension before copying into Ascend
formatted NPU MoE weights:

```python
expert_data = param.data[expert_id]
```

For floor4 this happened not to surface clearly because the 32-slot storage and
older reload path usually kept enough layout coincidence for the active prefix.
Floor2 preallocates a larger 64-slot storage and exercises more direct-import
and reload-capacity transitions.  In that layout, writing through
`param.data[expert_id]` can fail to update the intended formatted row even when
the source tensor is nonzero.

The fix is to preserve the expert dimension for NPU formatted per-expert writes:

```python
preserve_expert_dim_write = (
    not full_load
    and param.data.device.type == "npu"
    and param.data.ndim == loaded_weight.ndim + 1)
expert_data = (param.data if full_load else
               param.data[expert_id:expert_id + 1]
               if preserve_expert_dim_write else param.data[expert_id])
```

This change lives in:

```text
vllm/model_executor/layers/fused_moe/layer.py::FusedMoE.weight_loader
```

For the custom direct-import/runtime materialization path, the same invariant
must be enforced.  The helper now copies into a one-row destination slice rather
than indexing away the expert dimension:

```python
def _copy_logical_row_to_weight_slot(dst_weight, dst_slot, src_row, *,
                                     non_blocking=False):
    target = dst_weight[int(dst_slot):int(dst_slot) + 1]
    source = src_row
    if source.ndim == target.ndim - 1:
        source = source.unsqueeze(0)
    if source.device != dst_weight.device or source.dtype != dst_weight.dtype:
        source = source.to(device=dst_weight.device, dtype=dst_weight.dtype)
    target.copy_(source, non_blocking=non_blocking)
```

This change lives in:

```text
vllm_ascend/ops/fused_moe.py::_copy_logical_row_to_weight_slot
```

### What not to regress

The `param.data[expert_id:expert_id + 1]` / one-row-slice write is the core
correctness fix.  It should not be reverted when optimizing floor2 speed.
Earlier attempts that only changed communication cleanup, KV cache order,
metadata sync, or log2phy rebuild could make the run finish, but they did not
fix the corrupted expert row content.

The safe optimization boundary is:

- keep expert-dimension-preserving writes;
- keep direct NPU import enabled when the slot mapping is valid;
- remove only diagnostics, redundant validation, temporary staging buffers, and
  expensive storage snapshots from the hot path;
- do not replace the one-row-slice write with `param.data[expert_id]` or
  `dst_weight[slot]`.

### Validation signal

A correct floor2 smoke test should show:

- the run completes without OOM or HCCL workspace failure;
- generated responses no longer collapse into repeated punctuation or all run to
  the full response budget;
- direct-import postcheck, when explicitly enabled, reports nonzero target-slot
  samples for imported experts;
- 5-step probes can then be used for performance validation, but performance
  tuning should not touch the expert-dimension write semantics.

## 2026-07-08 Floor2 Step5 Floor16 Preemption Slowdown

After the floor2 correctness fix, the full epoch1/epoch2 natural run completed,
but epoch1 step5 was still slower than the floor4 reference:

| run | step5 `generate_sequences` | rollout output time | preemptions |
| --- | ---: | ---: | ---: |
| floor4 natural tailguard reference | 1389.15 s | 1419.88 s | 0 |
| floor2 natural fullworld-hot | 1535.27 s | 1572.52 s | 3 |

At first this looked like another floor2 memory or full-world restore problem.
However, the focused `planfull + runtimeshort` diagnostics showed that step5
entered floor16 with the same planning and KV-cache state for floor2 and floor4:

```text
selected_floor = 16
kv_cap = 380800
total_blocks = 2975
free_blocks = 2974
used_blocks = 1
```

The prompt set and prompt-to-rank assignment were also identical between the
floor2 and floor4 epoch1 step5 plans.  Therefore the slowdown was not caused by
a smaller initial KV cache, a dirty scheduler block pool, or a different batch
assignment.

The actual trigger was runtime KV pressure during the long decode.  In the
floor2 run, rank15 generated a heavier tail than predicted by the epoch0
length history:

```text
planner max_adjusted_rank_peak_tokens = 373404
floor2 actual max active peak        = 397764
kv_cap                               = 380800
```

So the planner believed step5 had:

```text
380800 - 373404 = 7396 tokens
```

of headroom, but the actual floor2 decode needed:

```text
397764 - 380800 = 16964 tokens
```

above the cap.  The effective miss was:

```text
397764 / 373404 = 1.065
```

or about 6.5% beyond the already safety-adjusted prediction.  This explains the
rank15 preemption messages:

```text
Preempting request 159 for request 132
Preempting request 158 for request 146
Preempting request 158 for request 136
```

The key distinction is:

- initial KV capacity was the same for floor2 and floor4;
- runtime active KV occupancy was higher in the floor2 step5 decode;
- preemption came from long-tail generation exceeding the predicted adjusted
  peak, not from a failed KV resize or leaked KV blocks at step start.

For this batch, the previous safety factor was too low:

```text
ACTIVE_PEAK_SAFETY_FACTOR = 1.16
required extra factor     ~= 1.065
effective needed factor   ~= 1.16 * 1.065 = 1.236
```

We therefore raised the floor2 planner default to:

```text
ACTIVE_PEAK_SAFETY_FACTOR = 1.25
```

This change is intentionally conservative.  Floor16 has no shrink rank-time
benefit, so avoiding preemption should take priority over squeezing the rank
matching close to the nominal KV cap.  Future planner work should consider a
floor16-specific objective that minimizes the maximum adjusted active KV peak
before optimizing rank-time/load skew.

## 2026-07-28 Planned Floor4 Training-Boundary OOM

Planned floor4 could pass one seed and fail another at the first actor update
even though both runs used the same floor plan and KV budgets.  The failed run
completed step1 rollout in 418.84 seconds, but rank0 reached the training memory
guard with only 24.41 GB free and 25.54 GB allocated by PyTorch.  The failure
was deterministic for that generated-length trace.

The missing condition was not an unsafe planner floor or a larger KV cache.
Planned preparation materializes floor4 and floor8 expert runtime aliases before
decode.  When the stochastic step finishes before a live shrink RPC is needed,
the driver restore RPC is a no-op.  Cleanup attached only to the full-world
restore callback therefore never runs, leaving the prepared expert aliases live
through the rollout-to-training transition.  A different seed can trigger an
actual shrink and restore, which explains why the old fix appeared intermittent.

The cleanup must run at the unconditional rollout-to-training boundary.  In
`vLLMRollout.offload_model_weights()`, every mode1 MoE module now calls
`release_mode1_full_world_transient_state()` before `param.data` is switched to
the CPU model storage.  This preserves canonical expert weights while releasing
runtime views, buffers, saved prefixes, CPU shadows, export-slot caches, and P2P
alias caches.  Canonical loaded-weight offload remains disabled because changing
that storage is not shape-safe for the next restore.

The seed3 reproduction validated the missing path without a live shrink.  Rank0
released 17.55 GB at the new boundary.  The subsequent guard observed 39.57 GB
free and 11.04 GB allocated by PyTorch, compared with 24.41 GB free and 25.54 GB
allocated before the fix.  All 16 ranks passed the 28 GiB reserve and the actor
update completed.

Future full Planned floor4 runs must retain all of the following safeguards:

- unconditional transient cleanup in `offload_model_weights()`;
- transient cleanup enabled after an actual full restore as a second line of
  defense;
- canonical loaded-weight offload disabled;
- a strict 28 GiB per-rank training memory guard;
- suite preflight that checks cleanup ordering and overrides unsafe inherited
  environment values;
- result validation that requires one rank0 boundary cleanup and memory guard
  for each of the five steps in both epochs.

The complete seed3 validation covered both planned epochs and all ten training
steps.  Each epoch produced five rollouts, five actor metrics, five rank0
boundary cleanups, and five rank0 memory guards.  No response was aborted and
no OOM occurred.  During epoch2, rank0 entered training with 39.30 to 39.59 GB
free after each step, while PyTorch held 11.04 to 11.18 GB.  The common epoch0
checkpoint remained untouched, and the two generated checkpoints were removed
only after artifact and log validation passed.

### Do not execute a live mutable Bash file

The first complete validation trained successfully but its outer launcher
failed during finalization.  The driver and fair-runner files had been edited
after the original processes started.  Bash reads top-level script commands
incrementally, so the running shells resumed at stale byte offsets and reported
an unmatched quote and a command assembled from the middle of a Hydra argument.
This was not a model, HCCL, or memory failure.

All long-lived paper launchers now execute a same-directory immutable snapshot
of themselves.  The fair suite preflight rejects a launcher stack missing these
snapshot guards.  `FAIR_FINALIZE_EXISTING=1` can validate and clean an already
completed run without repeating training.  It applies the same five-step log,
artifact, memory-safety, and checkpoint-shard checks as the normal completion
path before deleting any checkpoint.

## 2026-07-29 Non-monotonic Floor Cycles Need Capacity Metadata Restore

The ordinary paper plans move through a mostly monotonic sequence of expert
capacities. A forced 20-step `2,4,8` transition stress exposed a state bug when
the next cycle returned from floor8 to floor2 and then attempted the deeper
`8 -> 4` materialization. The physical expert Parameters had already returned
to 64 rows after full-world reload, but `loaded_weight_capacity` still recorded
the previous floor8 capacity of 16. Direct import consequently refused rows 16
through 31 and the parity path attempted a prohibited heavyweight activation.

The reload-capacity check incorrectly treated this state as aligned whenever
both values were no larger than the floor2 target. Physical storage equal to
the target is now handled separately. The implementation restores only
`loaded_weight_capacity` from 16 to 64, finalizes mapping metadata, and validates
the active mapping. It does not allocate or copy another expert tensor. A
physical row count below the target now raises an explicit error instead of
silently entering a later fallback.

The fixed stress completed all 20 steps with 41 staged shrink events, 20
full-world restores, and 96 metadata restorations. Allocated and reserved NPU
memory stayed fixed at 57.069031 and 57.078125 GiB. No OOM, collective timeout,
heavyweight parity fallback, or aborted response occurred. The output is under
`revision_stress/transition_stress_summary.md` in the July 28 matched-trial
artifact root.

Stress runs intentionally repeat the 160-prompt history to cover more than five
steps. They therefore disable construction of a next-epoch offline history,
whose normal invariant is exactly one set of responses per prompt. The stress
launcher supports `STRESS_FINALIZE_EXISTING=1` so a completed training run can
be validated after a postprocessing-only failure without consuming NPU time.
