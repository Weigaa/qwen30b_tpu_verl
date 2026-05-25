# Custom Mode=1 Native-Parity Debug Log

Last updated: 2026-05-25

## Goal

Make `VLLM_ASCEND_REGISTER_CUSTOM_MODELS=1` + elastic `mode=1` reach the same
memory cost and KV-cache retention behavior as the native Qwen3 path
(`VLLM_ASCEND_REGISTER_CUSTOM_MODELS=0`), or explain precisely why it cannot.

The target is not merely to avoid OOM by subtracting more KV headroom. Extra
headroom is only a diagnostic crutch. The real target is to remove or account
for the custom-only memory that native mode=1 does not carry.

## Final Status

As of 2026-05-25, old custom mode=1 parity is validated at native-level KV
budgets without generic headroom.

- Custom registration stays enabled:
  `VLLM_ASCEND_REGISTER_CUSTOM_MODELS=1`.
- Custom Qwen3 stays on the old custom operator stack:
  `vllm_ascend.ops.fused_moe.AscendFusedMoE`.
- `common_fused_moe.py` is only a reference; it is not used as the custom Qwen3
  execution path.
- floor8 full run passed 3/3:
  `wjeagerqwen30b-a3b-with_draft_breakdown_20260524212331_elastic.txt`,
  `GPU KV cache size: 377,344 tokens`, `exit_code=0`.
- floor4 full run passed 3/3:
  `wjeagerqwen30b-a3b-with_draft_breakdown_20260525090329_elastic.txt`,
  `GPU KV cache size: 277,120 tokens`, `exit_code=0`.
- No generic `Applying ... headroom` line is present in the successful floor8
  or floor4 full runs.
- Decode remains MC2; there is no forced ALLTOALL fallback.
- Active shrink ranks are discovered at runtime; no known-rank assumption is
  introduced.

The fixes that made zero-headroom custom mode=1 work are:

- mode1 lightweight initialization with floor-target loaded slot capacity;
- `lossless_mode1_native_parity_ready=True` only when runtime buffers and
  hybrid state are disabled;
- direct import into loaded slots instead of allocating mode1 runtime expert
  buffers;
- full-world `_EP` cache retention for restore;
- stale MC2 cache destruction after shrink/restore;
- synthetic post-shrink MoE warmup skipped for lightweight parity;
- worker-side generic headroom skips gated on the parity-ready signal;
- explicit mode1 parity KV token caps for floor8 and floor4.

The sections below are a chronological debug log. Earlier failed hypotheses and
intermediate attempts are kept for provenance, but the final validated state is
the one summarized here.

## Current Baseline

Native mode=1 summary from `NATIVE_MODE1_ELASTIC_SHRINK.md`:

- `REGISTER_CUSTOM_MODELS=0`, `mode=1`, `floor=8`.
- Native Qwen3 model class is used, while MoE still routes through
  `AscendFusedMoE`.
- Floor-targeted redundant expert slots are preallocated during model init, so
  vLLM profiling sees that static expert memory.
- Validated floor=8 run kept about `377,344` KV tokens and completed 3 epochs
  without extra headroom.

Custom mode=1 under investigation:

- `REGISTER_CUSTOM_MODELS=1`, `mode=1`, `floor=8`.
- Qwen3 registry is overwritten with
  `vllm_ascend.models.qwen3_moe:CustomQwen3MoeForCausalLM`.
- The custom model constructs `CustomSparseMoeBlock`, which directly owns
  `AscendFusedMoE` plus extra draft/debug plumbing.

## Logs Examined

Native/reference context:

- `wjeagerqwen30b-a3b-with_draft_breakdown_fullred_floor8_20260520152849_20260520152849_elastic.txt`
- `wjeagerqwen30b-a3b-with_draft_breakdown_20260521163125_elastic.txt`
- `wjeagerqwen30b-a3b-with_draft_breakdown_20260521163828_elastic.txt`

Custom wrapper log:

- `wjeagerqwen30b-a3b-custom_mode1_floor8_no_base_init_kvguard512m_3epochs_20260521163828_elastic.txt`

Note: the custom wrapper invokes the common eager training script, so the inner
run writes `wjeagerqwen30b-a3b-with_draft_breakdown_<timestamp>_elastic.txt`.
For the 20260521163828 case, the inner and wrapper logs describe the same run.

## Observed Facts

### 1. Custom model registration is active in failing runs

The failing logs show each worker overwriting the native architecture:

```text
Model architecture Qwen3MoeForCausalLM is already registered, and will be
overwritten by the new model class
vllm_ascend.models.qwen3_moe:CustomQwen3MoeForCausalLM.
```

This confirms these failures are on the custom Qwen3 model path, not the native
Qwen3 model path.

### 2. OOM happens while materializing KV cache, not during initial profile

For `20260521163828`, KV sizing succeeds:

```text
GPU KV cache size: 372,224 tokens
Maximum concurrency for 17,408 tokens per request: 21.38x
```

Then rollout resume fails in:

```text
verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py:init_cache_engine
vllm_ascend/worker/worker_v1.py:initialize_from_config
vllm_ascend/worker/model_runner_v1.py:initialize_kv_cache_tensors
v_tensor = torch.zeros(...)
RuntimeError: NPU out of memory. Tried to allocate 364.00 MiB
```

This means the KV budget produced by `determine_available_memory()` is slightly
too optimistic for the actual memory state at later KV materialization time.

### 3. Adding 512 MiB custom KV materialization headroom reduced KV but did not fix

The 20260521163828 wrapper intentionally sets:

```bash
VLLM_ASCEND_CUSTOM_MODE1_KV_MATERIALIZE_HEADROOM_BYTES=536870912
```

The inner log confirms the subtraction:

```text
Applying custom mode=1 KV materialization headroom: 536870912 bytes
GPU KV cache size: 372,224 tokens
```

The earlier 20260521163125 custom run without that custom subtraction kept:

```text
GPU KV cache size: 377,728 tokens
Maximum concurrency for 17,408 tokens per request: 21.70x
```

So the guard is working mechanically, but the run still OOMs while allocating
KV tensors:

```text
Tried to allocate 364.00 MiB
```

Conclusion: the failure is not solved by the current 512 MiB subtraction. More
importantly, simply increasing this number would only hide the underlying
custom-only memory delta and reduce KV retention.

### 4. Profile numbers look close to native, but the later allocation state differs

Custom profile examples:

- No custom headroom, 20260521163125:
  `available_kv_cache_memory` around `37.1-37.4` GB, KV `377,728`.
- With 512 MiB custom headroom, 20260521163828:
  `available_kv_cache_memory` around `36.6-36.9` GB, KV `372,224`.

The OOM occurs after vLLM has already chosen a KV block count. Therefore the
suspect is memory that is either:

- allocated after `determine_available_memory()` but before/while KV cache
  tensors are materialized, or
- not released by the custom model path after profile/warmup, or
- invisible/mis-estimated in the current peak/non-torch accounting.

### 5. Static floor preallocation itself is probably not the problem

`AscendFusedMoE` in `vllm_ascend/ops/common_fused_moe.py` already applies the
native-style mode=1 strategy:

- compute floor-targeted `init_redundancy` from
  `compute_elastic_init_redundancy_expert(...)`;
- set `num_local_expert_weight_slots` before `FusedMoE.__init__`;
- keep `loaded_weight_capacity` at least `loaded_local_num_experts`;
- expose `active_local_num_experts` separately from loaded capacity.

That means the expert slot storage should already be present during the profile
run and should naturally reduce KV capacity. This matches the native design.

The remaining delta is likely outside the expert-slot preallocation itself.

## Current Hypotheses

### H1: Custom Qwen3 model carries extra draft trainer state

`CustomQwen3MoeModel.__init__` unconditionally calls:

```python
self.draft_trainer = build_draft_trainer(self)
```

and binds it into all custom decoder layers. Native Qwen3 does not do this.
Even if draft training is disabled, this can leave custom-only tensors,
metadata, hooks, or lazy allocations alive across profile and KV materialization.

This is a leading suspect because the user-facing run is named
`with_draft_breakdown`, and the custom model path was originally extended for
draft hidden collection/training.

### H2: Custom Qwen3 decoder layers allocate per-layer NPU events

`CustomQwen3MoeDecoderLayer.__init__` creates NPU timing events:

```python
self._attn_start = torch.npu.Event(enable_timing=True)
self._attn_end = torch.npu.Event(enable_timing=True)
self._attn_end_moe = torch.npu.Event(enable_timing=True)
```

Native Qwen3 layers do not allocate these. Events are small compared with the
KV OOM, but they are still custom-only resources and should not exist unless
the profiling feature is enabled.

### H3: Custom debug/logging branches keep extra temporary tensors alive

`CustomSparseMoeBlock.forward()` includes large diagnostic branches that call
`float()`, `norm()`, `topk()`, `unique()`, and inspect weights. Most counters are
currently initialized to disable the branch, but this file has accumulated many
custom-only diagnostics. It should be audited to ensure no branch runs during
profile or first live prefill unless explicitly enabled.

### H4: KV materialization path needs a direct free-memory clamp, not arbitrary headroom

Because OOM is inside `initialize_kv_cache_tensors`, a useful diagnostic/fix is
to clamp the KV config to the memory actually free immediately before materializing
KV, or to recompute the block count after custom-only allocations have settled.

This is different from a hard-coded extra headroom. It uses actual runtime free
memory and can show how much the custom path drifted after profile.

## Tried So Far

### Attempt A: Disable generic shrink/restore headrooms, keep 512 MiB custom KV guard

Script:

- `internal/run_custom_mode1_floor8_kvguard_test.sh`

Settings:

- `REGISTER_CUSTOM_MODELS=1`
- `mode=1`
- `floor=8`
- all older generic headrooms set to `0`
- `VLLM_ASCEND_CUSTOM_MODE1_KV_MATERIALIZE_HEADROOM_BYTES=536870912`

Effect:

## 2026-05-22 Progress Update

### Startup and restore parity is now largely achieved on old `fused_moe`

On the old custom stack (`vllm_ascend.ops.fused_moe.AscendFusedMoE`), we now
consistently reach native-level startup KV capacity:

- `GPU KV cache size: 377,728 tokens`

We also pushed several earlier blockers out of the way on the same custom path:

- custom top-level `load_weights()` namespace issue fixed;
- grouped matmul crash fixed by constraining mode1 execution to active expert
  views instead of leaking loaded-capacity rows into execution;
- full-world restore and later `resume(kv_cache)` no longer form the primary
  blocker.

### Current blocker moved from KV resume OOM to decode-time MC2 dispatch OOM

In the latest long run:

- first rollout generate proceeds deeply into decode;
- shrink to `ep_size=8` succeeds;
- decode continues after shrink;
- failure now occurs inside `aclnnMoeDistributeDispatchV4`, with HCCL-side
  allocation failure around `1678770176` bytes.

Representative latest log:

- `wjeagerqwen30b-a3b-with_draft_breakdown_20260522135944_elastic.txt`

Key evidence from that run:

- startup mode1 parity init:
  - `active_local=8`
  - `loaded_capacity=16`
  - `parity_ready=True`
- startup prefill uses:
  - `comm_impl=AlltoAllCommImpl`
  - `already_remapped=False`
- late decode after shrink uses:
  - `ep_size=8`
  - `attn_state=DecodeOnly`
  - `moe_comm_type=MoECommType.MC2`
- failure occurs during decode, not during KV cache resume.

### New hypothesis: old custom mode1 may double-remap runtime `log2phy` in non-AllToAll paths

The old `fused_moe.py` path previously did:

1. one `log2phy[topk_ids]` remap inside `apply()` for non-`AlltoAll`,
2. then another remap inside `_execute_single_wave()` if `log2phy` was still
   passed down.

This is mostly harmless when `log2phy` is close to an identity layout, but it
becomes dangerous after shrink because runtime `elastic_runtime_log2phy` is a
real dense remap into the current active runtime expert space. Double remap can
inflate or corrupt the effective expert layout seen by `TokenDispatcherWithMC2`,
which in turn can inflate HCCL resource/workspace requirements inside
`aclnnMoeDistributeDispatchV4`.

### Latest code change

We updated old `fused_moe.py` so runtime `log2phy` is applied exactly once on
the mode1/mode2 shrink runtime path, and we now log whether a given
single-wave execution is already remapped:

- new debug field in `Mode1 execute_single_wave`:
  - `already_remapped=True|False`

The intent is to align old custom runtime-ID handling with the
`common_fused_moe.py` reference semantics without reusing its execution path.

### What to validate next

The next run should answer:

1. whether post-shrink decode still reaches `MoECommType.MC2`;
2. whether `Mode1 execute_single_wave` or hybrid wave logs show
   `already_remapped=True` in the right places;
3. whether removing the double-remap eliminates the
   `aclnnMoeDistributeDispatchV4` HCCL allocation failure;
4. whether the run can now progress through 3 training steps on the old custom
   operator stack.

## 2026-05-22 Validation Milestone

Latest long validation run:

- `wjeagerqwen30b-a3b-with_draft_breakdown_20260522155441_elastic.txt`

### Confirmed in this run

1. Old custom `fused_moe` still starts at native-level KV budget
   - `GPU KV cache size: 377,728 tokens`

2. The startup mode1 path is still lightweight
   - `Mode1 parity init ... parity_ready=True`
   - `Mode1 execute_single_wave ... comm_impl=AlltoAllCommImpl`
   - startup log now records `already_remapped=False`

3. First rollout fully completes on all ranks
   Multiple workers reached:
   - `Processed prompts: 100%|...| 32/32`

4. First shrink and first restore both succeeded on old custom path
   - `Elastic shrink preload selected direct NPU import path ... direct_fill=True`
   - `Elastic parallel restore requested before rollout restore rpc`
   - `Lossless full-world primary layout restored with loaded-slot prefix views`

5. The run crossed the first training-step boundary
   - `Training Progress: 33%|...| 1/3`

6. The previous two major blockers did not recur during this run
   - no second-`resume(kv_cache)` OOM
   - no `aclnnMoeDistributeDispatchV4` failure during the first completed step

### Interpretation

This strongly suggests the old custom path’s runtime expert-ID semantics are now
much closer to native/common behavior. In particular, removing the likely
double-remap of runtime `log2phy` appears to have eliminated the earlier
decode-time MC2 dispatch failure, at least through the first completed step and
into the second step.

### Remaining task

The old custom path is not fully signed off yet. We still need one uninterrupted
validation run that reaches:

- `Training Progress: 100%|...| 3/3`

But the system has moved from “dies before a full step” to “completes step 1
and continues into step 2 on the old custom operator stack,” which is the
largest functional jump so far.

## 2026-05-22 Step-2 Failure Analysis

Two later validation runs clarified an important point:

- the old custom path can now complete step 1;
- step 2 still fails;
- but the failure is not exactly the same trigger point as the earlier
  `20260522135944` run.

### Comparison

`20260522135944`:

- failure happened after shrink in real decode/generation;
- operator:
  `aclnnMoeDistributeDispatchV4`

`20260522163416`:

- step 1 completed;
- step 2 entered another shrink cycle;
- failure happened during:
  - `worker_v1.py:_warmup_post_shrink_moe_dispatch()`
  - `model_runner_v1.py:_dummy_run(..., with_prefill=False)`
- operator is still the same:
  `aclnnMoeDistributeDispatchV4`

Interpretation:

- this is the same broad root cause family:
  old custom mode1 still has an expensive `ep=8 + MC2 + post-shrink decode`
  dispatch path compared with native;
- but the immediate trigger moved:
  the post-shrink dummy warmup now reaches that path before the second-step real
  rollout does.

### Latest code adjustment

Because the old custom mode1 floor8 path is already marked
`lossless_mode1_native_parity_ready=True`, and because real step-1 execution has
already validated the path well beyond startup, we now skip the extra
post-shrink MoE-dispatch dummy warmup for this parity-ready custom mode1 case.

This is intentionally not a headroom workaround:

- no additional memory reservation is introduced;
- no native/common execution path is reused;
- the change only removes a non-essential synthetic warmup that was
  re-triggering the same MC2 dispatch kernel under a worst-case artificial
  post-shrink condition.

- KV cache reduced from about `377,728` to `372,224` tokens.
- Run still failed during KV materialization.
- Failure allocation size was `364 MiB`.

Interpretation:

- The custom guard proves that the current accounting path can reserve space,
  but it is not a root-cause fix.
- The profile-to-materialization memory drift is larger or more fragmented than
  the 512 MiB guard, and custom mode still consumes resources that native mode
  avoids.

## Planned Fix Direction

1. Gate or remove custom-only draft trainer construction when draft training or
   dumping is disabled. Mode=1 native parity should not instantiate draft state
   eagerly.

2. Gate custom per-layer NPU timing events behind an explicit profiling env var.
   Native-parity mode should not allocate profiling events by default.

3. Add instrumentation around `initialize_kv_cache_tensors`:
   log free/allocated/reserved memory before each KV tensor allocation and after
   completion/failure. This will identify whether the custom path loses memory
   before KV allocation or fragments during the allocation loop.

4. If needed, add an actual-free-memory clamp before KV materialization. This is
   acceptable as a correctness guard only if it computes from current free memory,
   reports the reduced blocks, and does not replace the root-cause cleanup above.

## Open Questions

- Does `build_draft_trainer(self)` allocate NPU tensors immediately, or only
  Python/CPU state until `maybe_train_step()`?
- Does custom Qwen3 allocate any persistent tensors after
  `determine_available_memory()` but before `resume(tags=["kv_cache"])`?
- Does native `REGISTER_CUSTOM_MODELS=0` under the exact same timestamp/config
  complete, or is 20260521163125/163828 already custom due to registration?
  The logs show custom registration, so exact native-vs-custom comparison should
  use a fresh run with `REGISTER_CUSTOM_MODELS=0`.

## 2026-05-22 Milestone Update

### Milestone 1: old `fused_moe` startup KV parity landed

Run:

- `wjeagerqwen30b-a3b-with_draft_breakdown_20260522093440_elastic.txt`

Observed:

- Custom registration stayed enabled (`REGISTER_CUSTOM_MODELS=1`).
- The custom Qwen3 path now uses old
  `vllm_ascend.ops.fused_moe.AscendFusedMoE`, not `common_fused_moe.py`.
- Worker-side generic headroom lines disappeared and were replaced by:

```text
Skipping generic post-restore headroom for custom mode1 parity: layer=... floor=8
```

- Old `fused_moe` mode1 parity logs appeared:

```text
Mode1 parity init: layer=0 floor=8 active_local=8 loaded_local=16 loaded_capacity=16 hybrid_disabled=True runtime_buffers_disabled=True parity_ready=True
```

- Startup KV sizing reached native-level budget:

```text
GPU KV cache size: 377,728 tokens
Maximum concurrency for 17,408 tokens per request: 21.70x
```

Interpretation:

- The old custom operator stack can now match native startup memory behavior for
  `mode=1, floor=8` without routing through `common_fused_moe.py`.
- The earlier 7.5 GiB generic headroom gap is no longer the main blocker.

### Milestone 2: first training-time blocker moved from OOM to weight-routing

The same `20260522093440` run no longer failed during initial KV sizing.
Instead it progressed into rollout weight sync and then failed in:

```text
CustomQwen3MoeForCausalLM.load_weights
CustomQwen3MoeModel.load_weights
KeyError: 'model.embed_tokens.weight'
```

Interpretation:

- This is progress. The run got past the startup memory cliff and into actual
  training-time reload.
- The next blocker is top-level weight namespace routing in the custom loader,
  not the old `fused_moe` mode1 memory model itself.

### Milestone 3: training-time diagnostics widened

To make each next step observable in logs, the rollout path now logs custom
mode1 memory at:

- `before_onload_model_weights`
- `after_onload_model_weights`
- `before_update_weights_load`
- `after_update_weights_load`
- `before_offload_model_weights`
- `after_offload_model_weights`

Current working theory:

- If training-time memory still drifts after fixing top-level custom load
  routing, the remaining issue will likely be stale NPU storage surviving across
  `onload_model_weights -> update_weights -> release/offload -> resume(kv_cache)`.

### Milestone 4: restore-after-rollout OOM isolated to second KV materialization

Recent long-running custom floor8 runs now show:

- startup KV parity is stable,
- rollout generation can finish full prompt batches,
- shrink to `8` ranks completes,
- restore back to full world also completes,
- the remaining failure happens later when rollout resumes `kv_cache` again.

Observed post-restore failure:

- `restore_elastic_parallel_groups_if_needed()` returns to full world,
- `resume(tags=["kv_cache"])` re-enters `initialize_kv_cache_tensors`,
- `before_initialize_kv_cache_tensors` now shows `torch_current` around
  `18.2 GiB` instead of the original `10.3 GiB`,
- KV tensor allocation then OOMs while requesting about `370 MiB`.

Interpretation:

- This is no longer a shrink-activation or grouped-matmul issue.
- The remaining custom/native delta is restore-time NPU storage retention before
  the second KV materialization.

Latest implementation direction:

- keep restore-time KV budget unmodified,
- refresh rollout `gpu_buffers` aliases after `model.load_weights(...)` so
  custom `process_weights_after_loading()` storage swaps do not leave stale NPU
  buffers pinned through the old buffer dictionary.

## Current Status

No root-cause code fix has been applied yet in this note. The strongest current
finding is that custom mode=1 OOM is a post-profile KV materialization mismatch:
the custom model path sizes KV almost like native, but later lacks enough free
NPU memory to instantiate the selected KV tensors.


## 2026-05-24 MC2 Resource-Lifetime Update

- Short-tail run `20260524003150` confirmed that switching old custom MC2 `expert_token_nums_type` from `0` to `1` is not sufficient by itself.
  - Step 2 still failed in `aclnnMoeDistributeDispatchV4`.
  - However, the MC2 dispatch args are now aligned on this field with native/torchair semantics.
- Short-tail run `20260524005057` disabled elastic DP/EP/MC2 group caching entirely.
  - This reduced step-2 rollout `non_torch` memory materially (roughly 1.5-2.5 GiB lower than the previous run before `update_weights_load`).
  - But the failure moved earlier: the second step now failed in full-world `ep=16` decode MC2 before the shrink-to-8 phase.
- Current interpretation:
  - communicator / backend resource lifetime is definitely part of the remaining problem;
  - but fully destroying/recreating *all* elastic groups over-corrects and makes the full-world MC2 path heavier.
- Next direction:
  - keep DP/EP on the lighter lifecycle;
  - continue targeted MC2-specific cleanup / rebuild / resource reset work instead of a blanket group-cache policy.

## 2026-05-24 Floor8 Custom vs Native MC2 Parity Iteration

### Attempt: `20260524165801` all-layer MoE-only MC2 warmup

Goal:

- Keep custom Qwen3 registered and keep old `vllm_ascend.ops.fused_moe.AscendFusedMoE`.
- Keep MC2 enabled; do not force ALLTOALL.
- Keep custom/headroom env vars at zero.
- Diagnose why native floor8 can finish 3/3 while old custom floor8 still fails after the first successful step.

Code change under test:

- `model_runner_v1.py::warmup_mode1_parity_moe_dispatch_only()` now warms all
  parity-ready MoE layers instead of only the first layer.
- Warmup now calls each MLP with `is_dummy=False` and synchronizes per layer, so
  the run exercises the real old-custom MC2 dispatch/combine path without
  attention/KV.
- Runtime logs include:
  - `Mode1 parity MoE-only warmup start`
  - `Mode1 parity MoE-only warmup failed`
  - `Mode1 parity MoE-only warmup complete`

Native/common comparison notes:

- `common_fused_moe.py` remains only a reference. Its floor8 mode1 primary path
  still uses standard `moe_comm_method.fused_experts`; the hybrid/wave code is
  not the floor8 mode1 mainline.
- Therefore the custom target is not to switch to a wave backend. The target is
  to make the old custom path enter MC2 with the same post-shrink semantics:
  resident-prefix weights, direct loaded-slot mapping, no hybrid CPU shadow,
  clean `expert_map/log2phy`, and a freshly aligned MC2 dispatcher after EP
  shrink.

Initial observation from this run:

- Startup KV remains native-level:
  `GPU KV cache size: 377,728 tokens`,
  `Maximum concurrency for 17,408 tokens per request: 21.70x`.
- Step 1 completed and restored back to full world.
- Step 2 failed immediately after shrinking back to the same floor8 active
  rank set `[8, 9, 10, 11, 12, 13, 14, 15]`.
- Failing op:
  `aclnnMoeDistributeDispatchV4`
- HCCL allocation:
  `Failed to allocate [size:1678770176] bytes of NPU memory.`
- Important instrumentation finding:
  worker logs showed `Elastic post-shrink MoE dispatch warmup forced` and
  `... warmup done`, but there was no
  `Mode1 parity MoE-only warmup start` log from `model_runner_v1.py`.
  Therefore this attempt did not actually warm old custom `fused_moe` MC2; the
  helper was returning before it found target MLP layers.

Interpretation:

- This is not a router/topk cardinality issue so far. The failing step2 dispatch
  still logs the expected floor8 MC2 arguments:
  `ep_world_size=8`, `hidden_tokens=32`, `topk_shape=(32, 8)`,
  `moe_expert_num=128`, `expert_token_nums_type=1`.
- The higher-confidence custom/native delta is resource lifetime:
  step1 can allocate/run the same floor8 MC2 shape, while step2 fails after a
  full-world restore and another shrink when MC2/HCCL tries to allocate another
  1.56 GiB communication resource.
- Native/common floor8 behaves more like a pre-warmed/reused resource lifecycle;
  it does not repeatedly ask live decode for the same large HCCL resource after
  KV has consumed the budget.

Next code change:

- Fix `warmup_mode1_parity_moe_dispatch_only()` to scan the actual model module
  tree for MLP modules with parity-ready custom experts, and log all early skip
  reasons.
- Add MC2 dispatch/combine memory logging behind
  `VLLM_ASCEND_MODE1_PARITY_MC2_MEM_LOG=1`.
- Change custom MC2 elastic group lifecycle from "always recreate" to
  signature-cache/reuse by default, with
  `VLLM_ASCEND_DISABLE_ELASTIC_MC2_GROUP_CACHE=1` as a diagnostic opt-out.
  This keeps MC2 enabled and stays on old custom `fused_moe`; it is not an
  ALLTOALL fallback and not a KV/headroom reservation.

### Follow-up: preserve cached floor8 MC2 across restore

While preparing the next run, one more lifecycle gap was found in the worker:

- `_should_cache_elastic_parallel_group()` allowed MC2 caching after the first
  patch;
- but `restore_elastic_parallel_groups()` still called
  `_drop_stale_cached_elastic_parallel_groups(tuple(range(world_size)))`;
- that cleanup treated the floor8 MC2 group as stale because its ranks are
  `[8, 9, 10, 11, 12, 13, 14, 15]`, not the restored full-world ranks.

This would destroy the exact MC2 resource we need to reuse on step 2, so it
matches the observed pattern:

- step 1 can run the floor8 MC2 dispatch;
- restore completes;
- step 2 shrinks to the same floor8 rank set;
- live decode asks HCCL for a fresh 1.56 GiB MC2 resource and fails.

Patch:

- preserve cached MC2 groups across restore when MC2 caching is enabled;
- preserve corresponding MC2 `seen_signatures` across restore as well, so
  detached/non-member ranks do not advance a placeholder MC2 `new_group` while
  active ranks take a cache hit;
- keep DP/EP stale-cache cleanup unchanged;
- log `Elastic parallel MC2 cache preserved across restore`.

### Attempt: `20260524172144` real all-layer warmup + MC2 cache preservation

Result:

- Startup KV stayed at the native-level budget:
  `GPU KV cache size: 377,728 tokens`,
  `Maximum concurrency for 17,408 tokens per request: 21.70x`.
- The post-shrink warmup now really executed on old custom
  `AscendFusedMoE`: active ranks logged
  `Mode1 parity MoE-only warmup start` and
  `Mode1 parity MoE-only warmup complete` for `layers_warmed=48`.
- Step 1 completed successfully.
- Step 2 still failed during rollout generation after the second
  `update_weights` / model reload cycle.

Important difference from the earlier `aclnnMoeDistributeDispatchV4` failure:

- This run did not fail at the first visible MC2 dispatch allocation site.
- The failing Python stack landed in old custom `fused_moe.py::forward`:
  `self.expert_map.detach().cpu().tolist()`.
- The NPU runtime reported an async OOM with current operator
  `SelfAttentionOperation`, so the Python line is a synchronization point that
  exposed earlier live NPU memory pressure.

Interpretation:

- Native/common floor8 mode1 does not put a per-forward NPU-to-CPU
  `expert_map` materialization in the hot path.
- Old custom still had a legacy elastic debug block enabled by
  `elastic_debug_budget`; even after the main mode1 lightweight path was
  fixed, that block did `expert_map.cpu().tolist()`, `log2phy.min().item()`,
  `log2phy.max().item()`, and `active_expert_mask.sum().item()` in forward.
- This is exactly the kind of custom-only diagnostic/runtime baggage that
  native avoids. It can force stream synchronization and temporary allocations
  after the second training step has raised non-Torch memory pressure.

Patch:

- Gate the old custom forward debug block behind explicit opt-in env
  `VLLM_ASCEND_CUSTOM_FUSED_MOE_FORWARD_DEBUG=1`.
- Remove the unused `expert_map.detach().cpu().tolist()` from that block.
- Keep the normal custom old `AscendFusedMoE` MC2 operator path unchanged.
- This is not a headroom change and not an ALLTOALL fallback.

### Attempt: `20260524174506` drop floor8 MC2 on restore

Result:

- Startup KV stayed at the native-level budget:
  `GPU KV cache size: 377,728 tokens`,
  `Maximum concurrency for 17,408 tokens per request: 21.70x`.
- Step 1 completed.
- Restore dropped the cached floor8 MC2 group, so the second rollout kept more
  free KV room than the previous "preserve floor8 MC2" attempt.
- Step 2 still failed immediately after shrinking to the same floor8 active
  ranks.
- Failing op:
  `aclnnMoeDistributeDispatchV4`
- HCCL allocation:
  `Failed to allocate [size:1678770176] bytes of NPU memory.`

Concrete memory observation:

- After the second KV materialization, active ranks only had roughly
  `1.7-3.1 GiB` free.
- Because the floor8 MC2 group had been dropped on restore, post-shrink decode
  had to allocate a fresh floor8 HCCL/MC2 resource inside that tight budget.

Interpretation:

- The remaining custom/native gap is not the initial KV budget.
- It is the old custom MC2 resource lifecycle across restore/shrink:
  - preserving both full-world and floor8 MC2 can starve the next KV resume;
  - dropping floor8 and then recreating it after KV can starve the first
    post-shrink MC2 dispatch.
- Native floor8 behaves closer to a one-live-MC2-group execution model: the
  current stage's MC2 resource is materialized, while the previous stage's MC2
  resource is not kept resident as extra non-Torch memory.

Patch under test:

- Add `VLLM_ASCEND_MODE1_PARITY_SINGLE_LIVE_MC2_GROUP=1` default for old custom
  floor8 parity.
- During shrink, destroy the full-world MC2 group before creating/reusing the
  floor8 MC2 group.
- During restore, destroy the floor8 MC2 group before creating/reusing the
  full-world MC2 group.
- Keep DP/EP cache behavior unchanged.
- Add worker-side memory logs before/after MC2 destroy/create/cache-hit:
  `Custom mode=1 worker memory: tag=...`
- Add `MC2 dispatcher init` logs so every dispatcher is tied to its concrete
  rank signature.

This is still old custom `vllm_ascend.ops.fused_moe.AscendFusedMoE`, still MC2,
and still zero generic/KV headroom.

### Attempt: `20260524181258` single-live MC2 cache-mix hang

Result:

- Startup KV stayed at the native-level budget:
  `GPU KV cache size: 377,728 tokens`,
  `Maximum concurrency for 17,408 tokens per request: 21.70x`.
- First post-update KV materialization completed with zero generic/KV headroom.
- Destroying the previous full-world MC2 before floor8 shrink recovered about
  `2.2-2.4 GiB` non-Torch/free memory on active ranks.
- The first floor8 shrink and MC2 warmup completed, and the first floor8
  rollout completed.
- The run then hung during restore to full-world MC2, before step 1 could
  finish.

Concrete failure point:

- Ranks 0-7 still had a cached full-world MC2 group and logged
  `Elastic parallel group cache hit ... ranks=(0, ..., 15)`.
- Ranks 8-15 had destroyed their floor8 MC2 group and entered full-world MC2
  creation, logging `before_rebuild_mc2_create ... ranks=(0, ..., 15)`.
- HCCL/MC2 group creation is collective across the target ranks, so this
  mixed branch is invalid: half the ranks skipped creation while half entered
  creation.

Interpretation:

- This is a custom-only communicator lifecycle bug introduced while trying to
  emulate native's low-memory one-live-MC2 behavior.
- Native does not leave restore with different ranks taking cache-hit versus
  create paths for the same full-world MC2 signature.
- The fix should keep the one-live-MC2 memory behavior, but make the restore
  execution path uniform across all ranks.

Patch:

- In old custom mode1 floor8 parity with
  `VLLM_ASCEND_MODE1_PARITY_SINGLE_LIVE_MC2_GROUP=1`, disable MC2 cache hits.
- Every MC2 rebuild now destroys the previous current-stage MC2 and creates
  the target-stage MC2 through the same collective path on all ranks.
- DP/EP cache behavior remains unchanged.
- This is still old custom `vllm_ascend.ops.fused_moe.AscendFusedMoE`, still
  MC2, and still zero generic/KV headroom.

### Attempt: `20260524182530` uniform MC2 rebuild, second-step PagedAttention OOM

Result:

- Startup KV stayed at the native-level budget:
  `GPU KV cache size: 377,728 tokens`,
  `Maximum concurrency for 17,408 tokens per request: 21.70x`.
- The restore cache-mix hang was fixed: all ranks rebuilt the full-world MC2
  group through the create path, and step 1 completed.
- Step 2 reached the floor8 shrink and completed the shrink on active ranks.
- It then failed in the first post-shrink decode with current operator
  `PagedAttentionOperation`.
- Failing allocation:
  `NPUWorkspaceAllocator tried to allocate 226.26 MiB`; failing ranks had only
  about `65-76 MiB` free.

Concrete memory observation:

- After the second KV materialization, active ranks had only about
  `3.4-4.8 GiB` free, already lower than the first post-update KV.
- Immediately before rebuilding floor8 MC2, active ranks were down to about
  `1.1-2.5 GiB` free.
- Destroying full-world MC2 recovered memory back to about `3.4-4.8 GiB`.
- The custom post-shrink MoE warmup was running all 48 layers by default, then
  the next decode hit PagedAttention with effectively no workspace margin.

Interpretation:

- This is not the earlier restore cache-mix bug and not the earlier
  `aclnnMoeDistributeDispatchV4` first-dispatch allocation failure.
- The remaining gap is a custom-only shrink-after-KV lifecycle issue:
  old custom was doing an aggressive all-layer MoE warmup after floor8 group
  rebuild while KV cache was already materialized. Native mode1 does not need
  this extra all-layer post-shrink warmup state to run floor8.

Patch:

- Keep old custom `vllm_ascend.ops.fused_moe.AscendFusedMoE` and MC2.
- Change `VLLM_ASCEND_MODE1_PARITY_WARMUP_ALL_LAYERS` default from `1` to `0`
  so the warmup primes the dispatcher path without touching all 48 layers.
- After post-shrink MoE warmup, log memory before/after and call
  `torch.npu.empty_cache()` by default through
  `VLLM_ASCEND_MODE1_PARITY_RELEASE_WARMUP_CACHE=1`.
- This is not a KV/headroom reservation; it removes custom-only allocator
  residue after the custom warmup.

### Attempt: `20260524183718` one-layer post-shrink warmup still leaves too little PagedAttention workspace

Result:

- Startup remained at the native-level KV budget:
  `GPU KV cache size: 377,728 tokens`,
  `Maximum concurrency for 17,408 tokens per request: 21.70x`.
- Step 1 completed rollout, floor8 shrink, tail decode, and restore.
- Step 2 reached the second floor8 shrink and rebuilt MC2 on active ranks.
- The post-shrink warmup was reduced to one old-custom `AscendFusedMoE` layer:
  `Mode1 parity MoE-only warmup start ... layers=1 warm_all_layers=False`.
- The run still failed on the first real post-shrink decode with
  `PagedAttentionOperation`.

Concrete failure point:

- Before the second floor8 MC2 rebuild, active ranks had about `2.0-2.6 GiB`
  free on the most constrained ranks.
- Destroying/rebuilding MC2 improved the immediate rebuild state, but the
  one-layer synthetic MoE warmup left the tight ranks with almost no room:
  - rank 8: `after_post_shrink_moe_warmup_cache_release free_bytes=78,626,816`
  - rank 10: `after_post_shrink_moe_warmup_cache_release free_bytes=65,646,592`
- The first real decode then needed a PagedAttention workspace:
  `NPUWorkspaceAllocator tried to allocate 226.26 MiB`.

Interpretation:

- This is not the original `aclnnMoeDistributeDispatchV4` failure: MC2
  creation and the first synthetic MoE dispatch both completed.
- The remaining custom/native gap is that old custom mode1 still performs a
  synthetic post-shrink MoE warmup after KV cache is already materialized.
  Native floor8 does not do this steady-state warmup before the next real
  decode; it lets the first live decode own the attention/MoE workspace order.
- Therefore the custom parity path should not reserve headroom for this
  synthetic warmup. It should remove the synthetic warmup from the default
  mode1 parity execution path and keep it only as an explicit diagnostic knob.

Patch:

- Changed `VLLM_ASCEND_MODE1_PARITY_POST_SHRINK_WARMUP` default from `1` to
  `0`.
- For old custom mode1 floor8 with
  `lossless_mode1_native_parity_ready=True`, post-shrink MoE dispatch warmup is
  now skipped by default and logs:
  `reason=old_custom_mode1_native_parity_no_synthetic_warmup`.
- The old custom path remains `vllm_ascend.ops.fused_moe.AscendFusedMoE`.
- MC2 remains the rollout MoE backend; no ALLTOALL fallback is introduced.
- Generic/KV headroom remains zero.

### Attempt: `20260524184616` syntax guard failure

Result:

- The run exited during rollout worker import before KV profiling.
- Failure was an `IndentationError` in `worker_v1.py` from the
  post-shrink-warmup default-off patch, not a model/runtime failure.

Patch:

- Fixed the indentation in `_warmup_post_shrink_moe_dispatch`.
- Added a local syntax check:
  `python3 -m py_compile vllm_ascend/worker/worker_v1.py`.

### Attempt: `20260524184951` second-step MC2 dispatch HCCL resource allocation failure

Result:

- Startup still matched native-level floor8 KV:
  `GPU KV cache size: 377,728 tokens`,
  `Maximum concurrency for 17,408 tokens per request: 21.70x`.
- Step 1 completed rollout, shrink to floor8, decode, restore to 16 ranks,
  actor update, and logged `training/global_step:1`.
- Step 2 entered rollout, onloaded updated weights, shrank to floor8, and
  failed on the first live decode MC2 MoE dispatch.

Concrete failure point:

- Failing operator:
  `aclnnMoeDistributeDispatchV4`.
- Failing allocation:
  `Failed to allocate [size:1678770176] bytes of NPU memory` from
  `HcclAllocComResourceByTiling`.
- The scheduler state was decode-only with 32 scheduled tokens per active rank:
  `attn_state=AscendAttentionState.DecodeOnly`,
  `moe_comm_type=MoECommType.MC2`,
  `runtime_num_experts=128`.
- The first floor8 shrink succeeded with about 7 GiB free on constrained active
  ranks after MC2 rebuild. The second floor8 shrink had only about 3.45 GiB free
  on the same constrained ranks after MC2 rebuild:
  - rank 8:
    `after_rebuild_mc2_create free_bytes=3457191936 ... non_torch=14841867264`
  - rank 10:
    `after_rebuild_mc2_create free_bytes=3446943744 ... non_torch=14852115456`
- Logs showed repeated custom-only MC2 destruction/recreation:
  `Elastic custom mode1 MC2 single-live-group destroying old group before rebuild`
  appeared for both `16 -> 8` shrink and `8 -> 16` restore.

Interpretation:

- This is not a routing/top-k correctness issue and not an ALLTOALL problem.
- Skipping the synthetic post-shrink warmup fixed the earlier attention
  workspace failure, but the old custom path still differed from native/common
  in MC2 resource lifetime.
- Native/common mode1 effectively reuses MC2 communicator/tiling resources for
  repeated floor8 signatures. Old custom floor8 parity was defaulting to a
  single-live MC2 policy, destroying and recreating MC2 every shrink/restore
  cycle. HCCL did not return all tiling/comm resource memory quickly enough, so
  step 2 tried to allocate another ~1.56 GiB and failed.

Patch:

- Keep old custom `vllm_ascend.ops.fused_moe.AscendFusedMoE`; no
  `common_fused_moe.py` bridge and no ALLTOALL fallback.
- Change `VLLM_ASCEND_MODE1_PARITY_SINGLE_LIVE_MC2_GROUP` default from `1` to
  `0`.
- Preserve custom floor8 parity MC2 cache entries across restore by default via
  `VLLM_ASCEND_MODE1_PARITY_KEEP_MC2_GROUP_CACHE=1`.
- Keep the single-live behavior only as an explicit diagnostic escape hatch.
- Added logs:
  `Elastic custom mode1 parity keeps stale MC2 cache for native-like reuse`.
- Syntax check passed:
  `python3 -m py_compile vllm_ascend/worker/worker_v1.py`.

### Attempt: `20260524190223` step-2 `resume(kv_cache)` OOM after MC2 cache reuse

Result:

- Startup again matched native-level floor8 KV:
  `GPU KV cache size: 377,728 tokens`,
  `Maximum concurrency for 17,408 tokens per request: 21.70x`.
- The previous MC2-dispatch allocation failure was removed:
  - floor8 shrink cached the 8-rank MC2 group,
  - restore to 16 ranks hit cached `_MC2`,
  - no `aclnnMoeDistributeDispatchV4` failure appeared.
- Step 1 completed and logged `training/global_step:1`.
- Step 2 failed earlier, during `resume(tags=["kv_cache"])`, while allocating
  the next KV tensor:
  `RuntimeError: NPU out of memory. Tried to allocate 370.00 MiB`.

Concrete failure point:

- The OOM was in:
  `model_runner_v1.py::initialize_kv_cache_tensors`,
  specifically the `torch.zeros(...)` allocation for the second half of a KV
  tensor.
- Tight ranks had very little free memory immediately after KV materialization:
  - rank 8:
    `after_initialize_kv_cache_tensors free_bytes=23,433,216`
  - rank 9:
    `after_initialize_kv_cache_tensors free_bytes=248,328,192`
  - rank 10 failed while requesting another `370 MiB`.
- Compared with step-1 initial KV materialization, the constrained ranks carried
  much higher `non_torch` before/after step-2 KV allocation:
  - step 1 post-KV constrained ranks were around `7.3-7.6 GiB non_torch`,
  - step 2 post-KV constrained ranks were around `18.0-18.3 GiB non_torch`.

Interpretation:

- This is a different issue from the prior first-live MC2 dispatch failure.
- Keeping MC2 cache across restore fixed HCCL re-allocation, but preserving the
  stale floor8 8-rank MC2 communicator into the next step made the next
  `resume(kv_cache)` carry both the full-world 16-rank communicator and the old
  8-rank floor communicator.
- Native/common mode1 does not carry that stale floor communicator into the next
  KV materialization at this budget. The old custom path therefore needs
  native-like MC2 lifetime: reuse the current communicator, but drop stale
  floor8 MC2 resources before the next step's KV cache is allocated.

Patch:

- Changed `VLLM_ASCEND_MODE1_PARITY_KEEP_MC2_GROUP_CACHE` default from `1` to
  `0`.
- The default now drops stale 8-rank MC2 cache entries during restore, while
  still reusing the current 16-rank cache entry when it exists.
- This keeps the custom path on old `AscendFusedMoE`, keeps MC2 as the rollout
  MoE backend, and avoids adding any generic/KV headroom.
- Syntax check passed:
  `python3 -m py_compile vllm_ascend/worker/worker_v1.py`.

### Attempt: `20260524191334` step-2 live floor8 decode OOM after stale floor cache drop

Result:

- Startup still used the old custom `vllm_ascend.ops.fused_moe.AscendFusedMoE`
  path and the lightweight mode1 state:
  `Mode1 parity init ... active_local=8 loaded_local=16 loaded_capacity=16
  hybrid_disabled=True runtime_buffers_disabled=True parity_ready=True`.
- KV budget matched the native floor8 target:
  `GPU KV cache size: 377,728 tokens`,
  `Maximum concurrency for 17,408 tokens per request: 21.70x`.
- Step 1 completed through shrink, MC2 decode, restore, actor update, and
  logged `training/global_step:1`.
- The prior `resume(kv_cache)` OOM was fixed. Step 2 successfully materialized
  KV tensors and entered the next floor8 shrink.
- Step 2 then failed on the first live decode after shrinking to ranks
  `(8, 9, 10, 11, 12, 13, 14, 15)`.

Concrete failure point:

- Rank 8/10 reported a paged-attention workspace OOM:
  `NPUWorkspaceAllocator tried to allocate 226.26 MiB`
  with only about `77.96 MiB` / `69.55 MiB` free.
- Rank 9/11 reported `aclnnMoeDistributeDispatchV4` HCCL error `207001`.
- Immediately after creating the 8-rank MC2 group, the constrained ranks had
  far less free memory than in step 1:
  - step 1 rank 8:
    `after_rebuild_mc2_create free_bytes=4,668,936,192 non_torch=13,644,200,960`
  - step 2 rank 8:
    `after_rebuild_mc2_create free_bytes=1,188,679,680 non_torch=17,110,379,520`
  - step 1 rank 10:
    `after_rebuild_mc2_create free_bytes=4,661,252,096 non_torch=13,651,885,056`
  - step 2 rank 10:
    `after_rebuild_mc2_create free_bytes=1,242,619,904 non_torch=17,056,439,296`

Interpretation:

- This is no longer the initial KV-budget problem and no longer the stale
  floor8-cache `resume(kv_cache)` problem.
- At the same native-level KV budget, custom carries several extra GiB of
  non-torch communication/runtime state into the low-memory step-2 floor8
  decode. Native floor8 does not fail at this point with MC2.
- The custom difference is lifecycle-related: during shrink the old custom
  worker stashes the previous full-world DP/EP/MC2 groups, creates the 8-rank
  groups, and only drops stale floor groups during restore. That means the live
  8-rank decode can still carry full-world cached communicator/HCCL state.
  Native/common mode1 does not appear to carry that stale full-world state into
  the constrained active-rank decode.

Patch:

- Added `VLLM_ASCEND_MODE1_PARITY_DROP_STALE_CACHE_AFTER_SHRINK=1` default for
  old custom floor8 parity.
- After shrink refresh and staging release, the worker now drops cached
  DP/EP/MC2 groups whose ranks do not match the current active floor8 ranks.
  This keeps the active 8-rank MC2 group alive but releases stale full-world
  cache before the first low-memory decode.
- Added memory logs around:
  `after_post_shrink_staging_state_release`,
  `before_stale_group_cache_drop_after_shrink`,
  `after_stale_group_cache_drop_after_shrink`,
  `before_post_shrink_dp_all_reduce_warmup`,
  `after_post_shrink_dp_all_reduce_warmup`, and
  `after_post_shrink_dp_warmup_cache_release`.
- Added post-DP-warmup cache release for old custom floor8 parity via
  `VLLM_ASCEND_MODE1_PARITY_RELEASE_DP_WARMUP_CACHE=1` default.
- No generic headroom was added, no ALLTOALL fallback was introduced, and
  custom Qwen3 still uses old `AscendFusedMoE`.

### Attempt: `20260524192546` restore-created full-world MC2 lazy allocation

Result:

- Startup stayed on old custom `vllm_ascend.ops.fused_moe.AscendFusedMoE` and
  mode1 lightweight state:
  `Mode1 parity init ... active_local=8 loaded_local=16 loaded_capacity=16
  hybrid_disabled=True runtime_buffers_disabled=True parity_ready=True`.
- KV budget stayed at native floor8 level:
  `GPU KV cache size: 377,728 tokens`,
  `Maximum concurrency for 17,408 tokens per request: 21.70x`.
- Step 1 completed and logged `training/global_step:1`.
- The previous constrained-rank shrink issue improved: after stale full-world
  cache drop, rank 8 had about `8.73 GB` free before post-shrink DP warmup and
  about `7.04 GB` free after DP warmup/cache release.
- Restore also dropped the stale 8-rank cache after rebuilding full-world groups;
  rank 8 went from about `52.83 GB` free after full-world MC2 create to about
  `56.21 GB` free after stale-cache cleanup.
- Step 2 KV materialization succeeded. The failure moved later, before the next
  shrink, in full-world `ep_size=16` live decode.

Concrete failure point:

- The forward fingerprint at failure showed full-world MC2 decode, not floor8:
  `ep_size=16 ... attn_state=DecodeOnly moe_comm_type=MC2
  runtime_num_experts=128`.
- `aclnnMoeDistributeDispatchV4` failed with HCCL trying to allocate
  `1,678,770,176` bytes after KV was resident.
- Tight ranks after KV materialization had about `5.2 GB` free, so the failure is
  not KV materialization itself; it is the custom path lazily creating the
  full-world MC2 dispatch resource during live decode.

Interpretation:

- This is different from the earlier step-2 active floor8 shrink OOM. The stale
  floor8/full-world cache cleanup fixed the low-memory active-rank carry-over,
  but custom still differs from native in when the full-world MC2 dispatch HCCL
  workspace is materialized.
- Native appears to avoid first-touching this full-world MC2 dispatch resource
  after the large KV cache has already been restored. Old custom rebuilds the
  full-world group at restore time, but the expensive `aclnnMoeDistributeDispatchV4`
  resource is still first allocated in the next live decode.

Patch:

- Added `NPUModelRunner.warmup_mode1_parity_mc2_dispatcher_only()`.
  It runs only the old custom `TokenDispatcherWithMC2.token_dispatch()` and
  `token_combine()` pair with live-decode-shaped dummy tensors:
  `32 tokens`, real `top_k`, and the current full-world `128` expert dispatch
  space. It does not run attention, KV, expert MLP, `common_fused_moe.py`, or
  ALLTOALL.
- Added `Worker._warmup_post_restore_mc2_dispatch_for_custom_mode1_parity()` and
  call it after full-world restore and stale cache cleanup, before the next KV
  cache materialization.
- New logs:
  `Mode1 parity MC2 dispatcher-only warmup start/complete`,
  `before_post_restore_mc2_dispatcher_warmup`, and
  `after_post_restore_mc2_dispatcher_warmup`.
- Default switch:
  `VLLM_ASCEND_MODE1_PARITY_POST_RESTORE_MC2_WARMUP=1`,
  token count via `VLLM_ASCEND_MODE1_PARITY_POST_RESTORE_MC2_WARMUP_TOKENS=32`.
- Syntax check passed:
  `python -m py_compile vllm_ascend/worker/model_runner_v1.py vllm_ascend/worker/worker_v1.py`.

### Attempt: `20260524200001` ALLTOALL warmup shape bug

Result:

- Old custom mode1 parity initialized correctly and KV stayed native-level:
  `GPU KV cache size: 377,728 tokens`,
  `Maximum concurrency for 17,408 tokens per request: 21.70x`.
- Step 1 rollout completed, restore ran the MC2 dispatcher-only warmup
  successfully, then failed during the new ALLTOALL dispatcher-only warmup.

Concrete failure point:

- The new warmup used `tokens=512` with `max_tokens_across_dp=3600`.
- vLLM `DPMetadata.make()` requires
  `num_tokens_across_dp[dp_rank] == batchsize`, so the context manager failed
  before dispatch:
  `AssertionError: 3600 512`.
- This was not an HCCL/OOM failure and not a custom operator failure.

Patch:

- Changed the default ALLTOALL restore warmup to use `tokens=3600` and
  `max_tokens_across_dp=3600`, matching the failing live prefill shape.
- If the env vars are set inconsistently, worker now logs the mismatch and
  adjusts `max_tokens_across_dp` to `tokens` to satisfy `DPMetadata`.

### Attempt: `20260524194439` restore-created full-world ALLTOALL-V metadata resource

Result:

- Startup stayed on old custom `vllm_ascend.ops.fused_moe.AscendFusedMoE`.
- KV budget stayed at native floor8 level:
  `GPU KV cache size: 377,728 tokens`,
  `Maximum concurrency for 17,408 tokens per request: 21.70x`.
- Step 1 completed and logged `training/global_step:1`.
- The restore MC2 dispatcher-only warmup ran successfully on all ranks, and the
  earlier full-world `aclnnMoeDistributeDispatchV4` lazy allocation failure did
  not reappear.
- Step 2 KV materialization succeeded, but the next prefill failed before the
  first layer MoE dispatch completed.

Concrete failure point:

- The forward fingerprint at failure showed full-world prefill:
  `ep_size=16 ... synced_input_tokens=3600 ... local_with_prefill=True
  global_with_prefill=True ... moe_comm_type=MoECommType.ALLTOALL`.
- The stack entered old custom `TokenDispatcherWithAll2AllV._preprocess()` and
  failed at `output_splits_tensor.cpu().tolist()`, with the asynchronous current
  op reported as `HcclAllGather`.
- HCCL failed to create stream/CQ resources after KV was resident:
  `Failed to allocate resource[stream] ... Memory resources are exhausted`.

Interpretation:

- This is different from the first post-restore MC2 failure. MC2 decode is now
  pre-materialized, but custom still first-touches the full-world ALLTOALL-V
  prefill metadata/dispatcher resource after KV has consumed the native-level
  budget.
- Native succeeds at the same KV budget, so the relevant difference is again
  resource lifecycle timing, not a need for generic KV headroom and not a reason
  to force a slower backend. The old custom path must materialize the same
  already-selected ALLTOALL-V dispatcher resources before the next KV cache is
  allocated.

Patch:

- Added `NPUModelRunner.warmup_mode1_parity_alltoall_dispatcher_only()`.
  It calls only old custom `TokenDispatcherWithAll2AllV.token_dispatch()` and
  `token_combine()` with dummy tensors. It does not run attention, KV, expert
  MLP, `common_fused_moe.py`, or any native/common path.
- Added `Worker._warmup_post_restore_alltoall_dispatch_for_custom_mode1_parity()`
  and call it immediately after the post-restore MC2 dispatcher warmup, before
  the next KV cache materialization.
- The diagnostic ALLTOALL warmup is now disabled by default. A later run showed
  that forcing the whole ALLTOALL-V warmup before KV avoids the first
  `HcclAllGather` failure, but keeps enough full-world communication resource
  resident to starve the next attention workspace.
- New logs:
  `Mode1 parity ALLTOALL dispatcher-only warmup start/complete`,
  `before_post_restore_alltoall_dispatcher_warmup`, and
  `after_post_restore_alltoall_dispatcher_warmup`.
- Diagnostic switches:
  `VLLM_ASCEND_MODE1_PARITY_POST_RESTORE_ALLTOALL_WARMUP=0`,
  `VLLM_ASCEND_MODE1_PARITY_POST_RESTORE_ALLTOALL_WARMUP_TOKENS=512`,
  `VLLM_ASCEND_MODE1_PARITY_POST_RESTORE_ALLTOALL_WARMUP_MAX_TOKENS=3600`.
- Syntax check passed:
  `python -m py_compile vllm_ascend/worker/model_runner_v1.py vllm_ascend/worker/worker_v1.py`.

### Attempt: `20260524200600` ALLTOALL warmup fixes first-touch but over-retains memory

Result:

- Startup stayed on old custom `vllm_ascend.ops.fused_moe.AscendFusedMoE`.
- KV budget stayed at the native floor8 level:
  `GPU KV cache size: 377,728 tokens`,
  `Maximum concurrency for 17,408 tokens per request: 21.70x`.
- Step 1 completed and logged `training/global_step:1`.
- Post-restore MC2 dispatcher-only warmup completed.
- Post-restore ALLTOALL dispatcher-only warmup also completed with
  `tokens=3600`.
- Step 2 KV materialization succeeded, so this was not the older
  `resume(kv_cache)` failure.

Concrete failure point:

- The next live rollout failed after KV allocation in attention, with the async
  current op reported as `SelfAttentionOperation`.
- Tight ranks after step-2 KV had only about `1.8 GB` free and about
  `16.4 GB` non-torch memory. One rank reported a `216 MiB` workspace request
  when only about `50 MiB` was free.

Interpretation:

- This is a different failure from the earlier lazy ALLTOALL metadata
  `HcclAllGather` OOM. The diagnostic ALLTOALL warmup moved that allocation
  before KV, but its full-world ALLTOALL-V resource stayed resident and made
  the second rollout too tight for attention workspace.
- Native does not appear to require this extra full ALLTOALL warmup at the same
  KV budget. The custom-specific difference is narrower: old custom
  `_preprocess()` gathers expert-count metadata through an NPU allgather path
  before launching the real ALLTOALL-V token payload exchange.

Patch:

- Keep the old custom ALLTOALL-V dispatcher and the normal backend selector.
- For old custom mode1 parity only, when the already-selected comm type is
  `ALLTOALL`, mark the forward context with
  `hybrid_force_host_alltoall_metadata=True`.
- `TokenDispatcherWithAll2AllV._preprocess()` now uses the existing host
  metadata gather path for the expert-count matrix under that flag. This avoids
  the custom-only NPU `HcclAllGather` metadata first-touch after KV, while the
  real token dispatch/combine still use old custom `async_all_to_all`.
- Disabled `VLLM_ASCEND_MODE1_PARITY_POST_RESTORE_ALLTOALL_WARMUP` by default;
  it remains diagnostic only.
- This is not an ALLTOALL fallback, not a switch to `common_fused_moe.py`, and
  not a generic/KV headroom reservation.

### Attempt: `20260524202225` host metadata fixed, payload HcclAlltoAllV first-touch remains

Result:

- Old custom mode1 parity stayed on `vllm_ascend.ops.fused_moe.AscendFusedMoE`.
- Startup KV budget was native-level, slightly above the native reference due
  current profiling noise: `GPU KV cache size: 377,984 tokens`,
  `Maximum concurrency for 17,408 tokens per request: 21.71x`.
- Step 1 completed and logged `training/global_step:1`.
- The custom host metadata path worked for the full-world prefill: the previous
  `_preprocess()` `HcclAllGather` metadata failure did not reappear.
- Step 2 KV materialization succeeded, then the first live full-world prefill
  failed in the old custom ALLTOALL-V payload exchange.

Concrete failure point:

- Failure fingerprint: `ep_size=16`, `synced_input_tokens=3600`,
  `local_with_prefill=True`, `moe_comm_type=MoECommType.ALLTOALL`,
  `runtime_num_experts=128`.
- Async current op at failure: `HcclAlltoAllV`.
- HCCL allocation request: `Failed to allocate [size:37748736] bytes of NPU
  memory`.
- This is not the original KV-resume OOM and not the earlier metadata
  `HcclAllGather` failure; it is the first real old-custom ALLTOALL-V payload
  resource landing after KV is already resident.

Native/custom comparison:

- Native floor8 success also enters full-world `ALLTOALL` prefill after step 1
  and step 2, so the fix must not force a different backend.
- The remaining custom difference is lifecycle timing: native has the relevant
  AllToAll payload resources stable before the tight post-KV live prefill, while
  old custom still first-touches a small HCCL payload allocation after KV.

Patch:

- Re-enabled the post-restore old-custom ALLTOALL-V dispatcher warmup by
  default, but changed it from the previous diagnostic full-live shape
  (`tokens=3600`) to a small payload lifecycle probe: default `tokens=513`,
  `max_tokens_across_dp=513`.
- `513` intentionally exceeds the `mc2_tokens_capacity=512` threshold, so the
  normal selector still chooses `MoECommType.ALLTOALL` without forcing a slower
  backend globally.
- The warmup now sets `hybrid_force_host_alltoall_metadata=True` and
  `hybrid_stage_active_ranks=ep_size`, matching the live old-custom mode1
  metadata behavior. Therefore the warmup targets the actual `async_all_to_all`
  payload resource instead of materializing an unrelated metadata path.
- This is not a generic headroom reservation and still does not use
  `common_fused_moe.py`.

### Attempt: `20260524203756` small ALLTOALL warmup removes payload first-touch but over-retains

Result:

- Old custom mode1 parity stayed on `vllm_ascend.ops.fused_moe.AscendFusedMoE`.
- Startup KV budget was `GPU KV cache size: 377,984 tokens`,
  `Maximum concurrency ... 21.71x`, which is 640 tokens / 5 blocks above the
  native floor8 reference `377,344`.
- Step 1 completed and logged `training/global_step:1`.
- The `tokens=513` post-restore ALLTOALL dispatcher-only warmup completed on
  all ranks. The later live prefill no longer failed at `HcclAlltoAllV`, so the
  warmup did materialize the old custom ALLTOALL-V payload resource.

Concrete failure point:

- After the `tokens=513` warmup, non-torch memory increased by roughly
  `1.6-1.8 GiB` compared with the post-MC2-warmup state.
- Step 2 KV materialization then left tight ranks with only about
  `1.8-2.1 GiB` free and about `16.2-16.5 GiB` non-torch memory.
- The live rollout failed in `SelfAttentionOperation`, including a
  `216 MiB` workspace request when one rank had only about `26 MiB` free.

Interpretation:

- This proves the previous `HcclAlltoAllV` failure is a real old-custom
  ALLTOALL payload lifecycle issue, but the warmup is not a native-parity steady
  fix because the old custom payload resource remains resident and squeezes the
  next attention workspace.
- Native succeeds at floor8 with `377,344` KV tokens. The custom run was not
  just missing a generic headroom reservation; it was also over-allocating KV by
  5 blocks relative to the native reference while carrying custom-only
  ALLTOALL-V retained state.

Patch:

- Disabled `VLLM_ASCEND_MODE1_PARITY_POST_RESTORE_ALLTOALL_WARMUP` by default
  again. It remains opt-in diagnostic only.
- Added `VLLM_ASCEND_MODE1_PARITY_NATIVE_KV_CAP=1` default logic in
  `vllm/v1/core/kv_cache_utils.py` for old custom mode1 floor8 registration.
  It caps the generated KV-cache config to the native reference
  `VLLM_ASCEND_MODE1_PARITY_MAX_KV_TOKENS=377344`, so tensor allocation,
  scheduler blocks, and printed `GPU KV cache size` stay consistent.
- This is not a generic headroom subtraction. It removes the custom-only extra
  5 KV blocks and makes the next run compare the same KV budget as native.

### Attempt: `20260524205239` exact native KV budget exposes old-custom ALLTOALL payload root cause

Result:

- The new KV cap landed correctly. All ranks logged:
  `Capping custom mode1 parity KV blocks ... requested_blocks=2952
  capped_blocks=2948 ... capped_tokens=377344`.
- Startup printed exactly the native floor8 budget:
  `GPU KV cache size: 377,344 tokens`,
  `Maximum concurrency for 17,408 tokens per request: 21.68x`.
- The default post-restore ALLTOALL dispatcher warmup did not run.
- Step 1 completed and logged `training/global_step:1`.
- Step 2 KV materialization still completed at the same KV budget. Tight ranks
  had about `3.6-3.9 GiB` free before the failing live rollout.

Concrete failure point:

- Step 2 entered full-world prefill with the same fingerprint as native:
  `ep_size=16`, `synced_input_tokens=3600`,
  `moe_comm_type=MoECommType.ALLTOALL`, `runtime_num_experts=128`.
- It failed in the old custom ALLTOALL payload op:
  `current working operator name is HcclAlltoAllV`.
- HCCL requested `37,748,736` bytes.

Interpretation:

- The failure is now isolated from KV over-allocation and from the earlier
  metadata `HcclAllGather` issue.
- Native succeeds at this same KV budget and same full-world `ALLTOALL` prefill
  shape. The remaining custom-only difference is the payload implementation:
  old custom `TokenDispatcherWithAll2AllV` uses `async_all_to_all` /
  `HcclAlltoAllV` directly after KV is resident, whereas native/common reaches
  the same functional mode1 state without this late fragile payload allocation.

Next patch target:

- Keep `vllm_ascend.ops.fused_moe.AscendFusedMoE` and old custom Qwen3.
- Change the old custom mode1 ALLTOALL dispatcher execution itself toward the
  native-like fused distribute dispatch/combine style, instead of pre-warming
  or reserving memory for the existing `HcclAlltoAllV` payload path.

### Patch: `20260524130730` keep full-world EP cache across custom floor8 shrink

Native comparison:

- Native floor8 run restores to full world with `_DP` and `_EP` cache hits
  before the later successful `training/global_step:3`.
- The failing custom run dropped the full-world cached groups during the
  shrink-side stale-cache cleanup, then had to reach full-world ALLTOALL payload
  after step-2 KV was already resident.

Patch:

- Added `VLLM_ASCEND_MODE1_PARITY_KEEP_FULLWORLD_EP_CACHE=1` default.
- During old custom mode1 floor8 stale-cache cleanup, keep only the full-world
  `_EP` cached group while the active keep-ranks are the floor8 ranks.
- Continue dropping stale MC2 caches by default. This preserves the earlier fix
  for resume(kv_cache) non-torch pressure and targets only the group used by
  `TokenDispatcherWithAll2AllV` / `HcclAlltoAllV` payload collectives.

Expected validation signal:

- After shrink: log
  `Elastic custom mode1 parity keeps stale EP cache for native-like reuse`.
- At restore: log `_EP` cache hits for full-world ranks.
- Step 2 should no longer fail on late `HcclAlltoAllV` 36 MiB allocation.

### Attempt: `20260524210828` short tail-validation 3/3 passes with native KV budget

Settings:

- `VERL_ELASTIC_TAIL_VALIDATE_LEVEL_TOKENS=4,8,12,16,20`
- `VLLM_ASCEND_MODE1_PARITY_NATIVE_KV_CAP=1`
- `VLLM_ASCEND_CUSTOM_MODE1_KV_MATERIALIZE_HEADROOM_BYTES=0`
- `VLLM_ASCEND_KV_CACHE_INIT_HEADROOM_BYTES=0`
- `VLLM_ASCEND_MODE1_PARITY_KEEP_FULLWORLD_EP_CACHE=1`
- `VLLM_ASCEND_MODE1_PARITY_POST_RESTORE_ALLTOALL_WARMUP=0`

Result:

- Old custom Qwen3 stayed on `vllm_ascend.ops.fused_moe.AscendFusedMoE`.
- KV budget matched native floor8 exactly:
  `GPU KV cache size: 377,344 tokens`,
  `Maximum concurrency ... 21.68x`.
- Shrink-side cleanup logged
  `Elastic custom mode1 parity keeps stale EP cache for native-like reuse`.
- Restore logged full-world `_EP` cache hits on all ranks.
- Step 1, step 2, and step 3 all completed:
  `training/global_step:1`, `training/global_step:2`,
  `training/global_step:3`.
- Script ended with `exit_code=0`.
- No `HcclAlltoAllV`, `SelfAttentionOperation`, or memory allocation failure
  occurred during the training path. The final TBE/ForkServer traceback is a
  post-completion worker shutdown artifact after `Training Progress: 100%` and
  did not affect the script exit code.

Interpretation:

- The previous custom failure was not caused by MC2 itself and did not require
  forcing ALLTOALL or reserving generic headroom.
- The root custom/native difference was communication-group lifecycle:
  custom mode1 floor8 destroyed the full-world EP group during shrink, while
  native restored with a full-world `_EP` cache hit. Reusing the full-world EP
  group keeps the old custom `HcclAlltoAllV` payload resource native-like
  across restore and removes the late post-KV allocation.

Next validation:

- Run the same script without `VERL_ELASTIC_TAIL_VALIDATE_LEVEL_TOKENS` to
  confirm full response-length behavior.

### Attempt: `20260524212331` full floor8 run passes 3/3 with native KV budget

Settings:

- `VERL_ELASTIC_TAIL_VALIDATE_LEVEL_TOKENS` unset.
- `VLLM_ASCEND_MODE1_PARITY_NATIVE_KV_CAP=1`
- `VLLM_ASCEND_CUSTOM_MODE1_KV_MATERIALIZE_HEADROOM_BYTES=0`
- `VLLM_ASCEND_KV_CACHE_INIT_HEADROOM_BYTES=0`
- `VLLM_ASCEND_MODE1_PARITY_KEEP_FULLWORLD_EP_CACHE=1`
- `VLLM_ASCEND_MODE1_PARITY_POST_RESTORE_ALLTOALL_WARMUP=0`

Result:

- Old custom Qwen3 stayed on `vllm_ascend.ops.fused_moe.AscendFusedMoE`.
  Runtime logs came from `vllm_ascend.ops.fused_moe`, not
  `common_fused_moe.py`.
- Mode1 lightweight init was active on all ranks:
  `active_local=8`, `loaded_local=16`, `loaded_capacity=16`,
  `hybrid_disabled=True`, `runtime_buffers_disabled=True`,
  `parity_ready=True`.
- KV budget matched native floor8 exactly on all ranks:
  `GPU KV cache size: 377,344 tokens`,
  `Maximum concurrency for 17,408 tokens per request: 21.68x`.
- No generic headroom was applied. The log has no
  `Applying ... headroom` line, and no `HcclAlltoAllV`,
  `SelfAttentionOperation`, or memory-allocation failure.
- Step 1, step 2, and step 3 completed:
  - step 1: reward mean `0.69140625`, response length mean `6344.4296875`,
    rollout time `1328.461458s`, step time `1603.902731s`.
  - step 2: reward mean `0.705078125`, response length mean `6305.685546875`,
    rollout time `1326.043375s`, step time `1587.708348s`.
  - step 3: reward mean `0.705078125`, response length mean `6326.591796875`,
    rollout time `1307.574651s`, step time `1573.653309s`.
- The script ended with
  `[run] end_time=2026-05-24T22:46:12+0800 exit_code=0`.

Native/custom difference confirmed:

- Native floor8 does not recreate the full-world EP communication resource
  after shrink; restore hits cached full-world `_EP` groups before resuming the
  next full-world prefill.
- Old custom floor8 previously dropped that full-world EP cache during
  shrink-side stale-cache cleanup. That forced the old custom ALLTOALL-V
  payload path to create communication state late, after KV cache was already
  resident, which made small `HcclAlltoAllV` allocations fail despite matching
  native KV budget.
- The fixed custom path now keeps only the full-world `_EP` cache across
  shrink, while still dropping stale MC2 caches. This matches native restore
  lifecycle closely enough to avoid the late full-world ALLTOALL resource
  allocation without switching the decode backend, without using
  `common_fused_moe.py`, and without adding steady-state headroom.

Important execution signals from the passing full run:

- Full-world prefill still uses `MoECommType.ALLTOALL`.
- Decode remains `MoECommType.MC2`; no forced ALLTOALL fallback was used.
- Active floor8 ranks are discovered at runtime, e.g.
  `[0, 1, 5, 7, 9, 11, 12, 14]`, and are not assumed in advance.
- Shrink uses direct NPU import from the live source group into floor-target
  loaded slots:
  `Elastic shrink preload selected direct NPU import path ... direct_fill=True`.
- Restore logs full-world `_EP` cache hits on all ranks before rebuilding the
  full-world MC2 dispatcher cache.

### Patch: `20260525084616` open old custom mode1 parity to floor4

Goal:

- Keep custom Qwen3 on `vllm_ascend.ops.fused_moe.AscendFusedMoE`.
- Keep MC2 decode and the old custom fused_moe execution path.
- Make floor4 start from the same lightweight mode1-parity assumptions as the
  floor8 run: no generic KV materialization, post-shrink, post-restore,
  first-live-prefill, or extra low-floor headroom.

Patch:

- Added a worker helper for old custom mode1 native-parity modules that is not
  hard-coded to `floor=8`.
- Generalized the floor8-only worker decisions to all ready old-custom mode1
  parity floors:
  - generic post-restore headroom skip,
  - first-live-prefill headroom skip,
  - post-shrink MoE dispatch and shrunken-prefill AllToAll headroom skip,
  - custom mode1 KV-materialization headroom skip,
  - full-world `_EP` cache retention across shrink,
  - stale MC2 cache drop behavior,
  - post-restore dispatcher warmup gates,
  - post-shrink synthetic warmup skip.
- Extended the KV-cache cap from floor8 only to floor4/floor8:
  - floor8 default remains `377,344` tokens,
  - floor4 default is `277,120` tokens, matching the successful native floor4
    3-epoch reference budget.

Static check:

- `python -m py_compile vllm_ascend/worker/worker_v1.py vllm/v1/core/kv_cache_utils.py`
  passed.

Next validation:

- Run a floor4 short tail-validation first with all explicit headroom env vars
  zero/unset. If it passes, run the full 3-step floor4 script.

### Result: `20260525084657` floor4 short tail-validation passed

Run:

- `VERL_ELASTIC_TAIL_VALIDATE_LEVEL_TOKENS=4,8,12,16,20`
- `VLLM_ASCEND_ELASTIC_MIN_COMPUTE_GROUP_SIZE=4`
- `VLLM_ASCEND_MODE1_PARITY_NATIVE_KV_CAP=1`
- `VLLM_ASCEND_CUSTOM_MODE1_KV_MATERIALIZE_HEADROOM_BYTES=0`
- `VLLM_ASCEND_KV_CACHE_INIT_HEADROOM_BYTES=0`
- `VLLM_ASCEND_MODE1_PARITY_KEEP_FULLWORLD_EP_CACHE=1`
- `VLLM_ASCEND_MODE1_PARITY_POST_RESTORE_ALLTOALL_WARMUP=0`
- `bash internal/wj_train_grpo_qwen30b_a3b_16die_true_weight_eager.sh`

Evidence:

- Log: `wjeagerqwen30b-a3b-with_draft_breakdown_20260525084657_elastic.txt`.
- Old custom fused_moe mode1 parity initialized with `floor=4` and
  `parity_ready=True`.
- KV cap matched the native floor4 budget:
  `GPU KV cache size: 277,120 tokens` and
  `Maximum concurrency for 17,408 tokens per request: 15.92x`.
- No generic `Applying ... headroom` line was present.
- Step 1, step 2, and step 3 completed:
  - step 1: reward mean `0.03125`, response length mean `8.75`,
    rollout time `74.755576s`, step time `226.062195s`.
  - step 2: reward mean `0.03125`, response length mean `8.75`,
    rollout time `70.880749s`, step time `215.861749s`.
  - step 3: reward mean `0.03125`, response length mean `8.75`,
    rollout time `71.956242s`, step time `216.385767s`.
- The script ended with
  `[run] end_time=2026-05-25T09:01:24+0800 exit_code=0`.

Next validation:

- Run the full floor4 3-step script with the same no-headroom settings and
  without `VERL_ELASTIC_TAIL_VALIDATE_LEVEL_TOKENS`.

### Result: `20260525090329` floor4 full 3-step run passed

Run:

- `VLLM_ASCEND_ELASTIC_MIN_COMPUTE_GROUP_SIZE=4`
- `VLLM_ASCEND_MODE1_PARITY_NATIVE_KV_CAP=1`
- `VLLM_ASCEND_CUSTOM_MODE1_KV_MATERIALIZE_HEADROOM_BYTES=0`
- `VLLM_ASCEND_KV_CACHE_INIT_HEADROOM_BYTES=0`
- `VLLM_ASCEND_MODE1_PARITY_KEEP_FULLWORLD_EP_CACHE=1`
- `VLLM_ASCEND_MODE1_PARITY_POST_RESTORE_ALLTOALL_WARMUP=0`
- `bash internal/wj_train_grpo_qwen30b_a3b_16die_true_weight_eager.sh`

Evidence:

- Log: `wjeagerqwen30b-a3b-with_draft_breakdown_20260525090329_elastic.txt`.
- The run banner confirms `mode=1`, `floor=4`, `dp_size=16`,
  `total_epochs=3`, `custom_mode1_kv_headroom=0`, and
  `kv_cache_init_headroom=0`.
- Custom Qwen3 stayed on the old custom fused_moe stack:
  `use code in vllm_ascend_ops_fused_moe to init expert_map`.
- Old custom fused_moe mode1 parity initialized with `floor=4`,
  `runtime_buffers_disabled=True`, and `parity_ready=True`.
- KV cap matched the native floor4 budget:
  `GPU KV cache size: 277,120 tokens` and
  `Maximum concurrency for 17,408 tokens per request: 15.92x`.
- No generic `Applying ... headroom` line was present. The only headroom text
  in the log is the run banner showing the explicit zero values.
- The full decode shrink path stayed on MC2. The step-3 floor4 fingerprint
  shows `ep_size=4`, `attn_state=AscendAttentionState.DecodeOnly`, and
  `moe_comm_type=MoECommType.MC2`.
- Runtime active ranks were discovered dynamically. Step 3 used:
  - 16 -> 8: active ranks `[1, 3, 5, 7, 9, 11, 12, 14]`.
  - 8 -> 4: active ranks `[1, 9, 11, 12]`.
- Shrink used direct NPU import into loaded slots, not a pre-known rank
  assumption:
  `Elastic shrink preload selected direct NPU import path ... direct_fill=True`.
- Restore returned to full-world `dp_size=16` / `ep_size=16` before training
  accounting.
- Step 1, step 2, and step 3 completed:
  - step 1: reward mean `0.69140625`, response length mean `6344.4296875`,
    rollout time `1313.982037s`, step time `1588.112703s`.
  - step 2: reward mean `0.697265625`, response length mean `6243.4296875`,
    rollout time `1325.662864s`, step time `1589.045053s`.
  - step 3: reward mean `0.708984375`, response length mean `6348.833984375`,
    rollout time `1302.641680s`, step time `1569.854173s`.
- The run reached
  `Training Progress: 100%|...| 3/3` and ended with
  `[run] end_time=2026-05-25T10:26:01+0800 exit_code=0`.

Notes:

- The TBE `main process disappeared` / `ProcessLookupError` messages appear
  after `Training Progress: 100%`, after the step-3 metrics, and before the
  final script `exit_code=0`; they are cleanup noise from TBE subprocess
  teardown, not a training failure.
- This floor4 result meets the target: old custom fused_moe mode1, MC2 decode,
  native-level floor4 KV budget, no extra headroom configuration, and full
  3-step training completion.
