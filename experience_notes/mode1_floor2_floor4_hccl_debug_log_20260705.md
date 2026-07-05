# Mode1 Floor2/Floor4 HCCL and KV Debug Log

This note records the current state of the floor2 -> floor4 debugging thread.
Before making a new optimization in this path, read this file first and append
the new observation/result here.  The goal is to avoid repeating the same
experiments and to keep the failure hypotheses falsifiable.

## Scope

Target path:

- Mode: mode1 shrink-aware natural policy.
- Shrink stages: 16 -> 8 -> 4 -> 2.
- Short probe shape: selected floors `2,2,4`.
- Expected floor4 KV budget: `280576` tokens.
- Current quick script:
  - `run_mode1_dynamic_floor2_to_floor4_kv_probe.sh`
- Related full/adaptive script:
  - `run_mode1_local_length_sorted_e2e_adaptive_floor2.sh`

The immediate correctness target is not final throughput.  It is:

1. step1 and step2 can shrink to floor2 and restore full-world safely.
2. step3 can enter floor4 with the normal floor4 state:
   - assignments like `[32,32,32,32]`
   - expected floor4 import behavior, e.g. `cpu_imports=64 remote_imports=64`
   - KV cache size returns to the known floor4 budget `280576`.
3. No HCCL/MC2 `path string is NULL`, OOM, stale dispatcher state, or corrupted
   expert map/log2phy state.

## Known Good Baselines

### Floor4-only natural/tailguard

Earlier floor4 natural runs were able to run with floor4 KV cache size around
`280576` tokens.  In that configuration, floor4 only needed slots for 32 local
experts and did not expose the floor2 -> floor4 upward transition bug.

Representative log referenced during debugging:

- `mode1_dynamic_length_aware_adaptive_floor4_natural_tailguard_full3/...`

### Force floor2 no-trim small-cap probe

Floor2 can run when the test is simplified and KV is small enough, especially
with allocator trim disabled.  This showed that floor2 itself is not impossible,
but the floor2 -> restore -> next shrink/floor transition path is fragile.

Prior conclusion from:

- `mode1_floor2_notrim_kv_cap_pitfall_20260703.md`

Summary:

- allocator trim is not the whole root cause.
- reducing KV cache alone does not explain all latency/failure behavior.
- after floor2 restore, MC2/runtime state can enter a slower or inconsistent
  path.

## Current Failing Probe

Most recent probe directory:

- `mode1_dynamic_floor2_to_floor4_kv_probe/epoch_001_mode1_natural`

Important logs:

- `logs/wjqwen30b-a3b-record_graph_save4eagle3_20260705094159.txt`
- `logs/wjqwen30b-a3b-record_graph_save4eagle3_20260705095637.txt`

Both runs complete the first rollout and then fail after full-world restore with:

```text
path string is NULLpath string is NULL
```

The current probe has not yet reached a clean step3/floor4 KV validation in the
latest runs.

## Observations by Attempt

### Attempt A: direct NPU import/export optimization

Motivation:

- Floor2 shrink was spending unnecessary time packing/exporting expert tensors.
- We suspected slow export/pack or device mismatch in direct import.

Changes made:

- Added contiguous NPU alias/range export support in `vllm_ascend/ops/fused_moe.py`.
- Batched sender export path in `vllm_ascend/worker/worker_v1.py`.
- Fixed device mismatch in `log2phy[topk_ids]` path by ensuring maps/tensors are
  on the expected NPU device.

Result:

- This moved the failure forward.
- In latest logs, shrink import/export is no longer the main failure.
- Sender path shows very small export overhead and `send_pack_copy_ms=0.00`.

Representative latest log evidence:

```text
Elastic shrink stream import breakdown ... mode=direct_npu ...
send_export_view_ms ~= small
send_pack_copy_ms=0.00
recv_direct_slot_chunks>0
```

Status:

- Keep this optimization.  It appears helpful and not the current blocker.

### Attempt B: no-op/adaptive KV resize cleanup

Motivation:

- Floor2 later steps were slower than step1.
- Hypothesis was that repeated KV resize/restore and allocator trim caused
  extra latency or memory fragmentation.

Changes/experiments:

- Added no-op resize skip when block count is unchanged.
- Tested with allocator trim disabled.
- Tested smaller KV caps, including very small cap (`50000`).

Result:

- Disabling trim and reducing KV can make some floor2 tests run more smoothly.
- However, the floor2 -> floor4 transition still fails in the dynamic probe.
- Therefore allocator trim / KV size is not the sole root cause.

Status:

- Keep no-trim for probe scripts.
- Do not assume "more workspace" alone fixes the issue.

### Attempt C: upward floor cleanup before KV resize

Motivation:

- When moving from floor2 to floor4, floor2 materializes larger temporary expert
  residency than floor4 needs.
- If floor2 residue stays alive, step3/floor4 KV profiling sees too little free
  memory and cannot recover to `280576`.

Changes/experiments:

- Added cleanup/compact path for `target_floor > last_floor`.
- Intended semantics:
  - floor2 -> floor4 cleanup
  - floor2 -> floor8 cleanup
  - floor4 -> floor8 cleanup
- Cleanup should remove stale expert slot caches and compact to the capacity
  required by the next target floor, not merely primary 8.

Expected slot capacities:

- floor16: 8 local experts
- floor8: 16 local expert slots
- floor4: 32 local expert slots
- floor2: 64 local expert slots

Current concern:

- Some cleanup may be correct for tensor memory but unsafe for live HCCL/MC2
  runtime state.
- The test now fails before proving the floor4 KV budget is restored.

Status:

- Keep the idea, but treat communicator/runtime cleanup separately from tensor
  slot cleanup.

### Attempt D: full-restore low-floor group defer

Motivation:

- Earlier latest log (`20260705094159`) showed:

```text
Elastic mode1 low-floor stashed group direct destroy fallback:
  group_kind=dp/ep/mc2
  group_ranks=(14,15)
  reason=mode1_full_restore
  defer_env=1
  defer_sizes=1,2,4,8
```

- This explained why `path string is NULL` could happen after restore: the low
  floor HCCL group was directly destroyed while runtime/HCCL still had sensitive
  references.

Change made:

- `run_mode1_dynamic_floor2_to_floor4_kv_probe.sh` now exports:

```bash
VLLM_ASCEND_MODE1_PARITY_DEFER_GROUP_DESTROY_ON_FULL_RESTORE=1
```

- `run_mode1_local_length_sorted_e2e_adaptive_floor2.sh` defaults the same flag
  to `1`.
- Direct-destroy fallback logs now print:
  - `defer_env`
  - `defer_full_restore`
  - `defer_pre_rebuild`
  - `defer_stash`
  - `defer_sizes`

Result:

- This eliminated the visible `direct destroy fallback` pattern in the next run.
- But the latest run (`20260705095637`) still fails after full restore with
  `path string is NULL`.
- New evidence:

```text
Elastic parallel group retired immediately:
  rank=14 group_kind=mc2 group_ranks=(14,15)
  reason=mode1_full_restore
  destroy_device_pg=1
  destroy_cpu_pg=1
```

Interpretation:

- The previous "direct destroy fallback" was one real bug, but not the whole
  problem.
- Even the defer/retire path can still destroy the underlying device process
  group during full restore.
- That device PG destruction may still trigger HCCL `path string is NULL`.

Status:

- Do not simply toggle more defer flags and assume done.
- Next fix should distinguish:
  - removing Python references / active state
  - retiring wrapper objects
  - destroying underlying HCCL device PG

## Current Root-Cause Hypotheses

### H1: Device PG destruction during full restore is unsafe

Evidence:

- `path string is NULL` appears after full-world restore.
- Latest run no longer shows direct-destroy fallback, but does show:

```text
group_kind=mc2 group_ranks=(14,15) reason=mode1_full_restore
destroy_device_pg=1 destroy_cpu_pg=1
```

Likely issue:

- Full restore replaces the active group with full-world group, but HCCL/MC2 may
  still have references or delayed runtime use of the old low-floor device PG.
- Destroying the low-floor device PG immediately is unsafe.

What to test next:

- For `reason=mode1_full_restore`, do not destroy device PG for low-floor
  DP/EP/MC2 groups; quarantine wrappers/references only.
- Destroy only at process teardown or after a much later explicit safe point.
- Verify no `path string is NULL`.

### H2: MC2 dispatcher state is rebuilt with stale rank/group identity

Evidence:

- Full-world MC2 warmup args look superficially correct:
  - `group_ranks=(0..15)`
  - `ep_world_size=16`
  - `expert_map_len=128`
  - `topk_min=0 topk_max=127 topk_unique=128`
- But failure still happens after warmup/restore.

Possible cause:

- The dispatcher object or cached HCCL handle points to a stale device PG even
  while printed ranks look correct.

What to log/verify next:

- dispatcher id, group uid/name, device pg id before/after:
  - restore entry
  - after rebuild
  - after refresh
  - after MC2 warmup
  - before next shrink
- Confirm stale `(14,15)` dispatcher handles are not reachable.

### H3: tensor/expert state cleanup and communicator cleanup are conflated

Evidence:

- Upward floor cleanup is needed for KV recovery.
- But aggressively releasing or retiring runtime group state during full restore
  causes HCCL instability.

Rule going forward:

- Tensor/expert slot cleanup can happen before KV profiling.
- HCCL/MC2 group/device PG destruction should be conservative and delayed.

## Important Negative Results

- It is not simply "KV cache too large"; even small KV tests exposed later-step
  latency or restore instability.
- It is not simply "direct NPU import is slow"; optimized direct NPU import
  removed pack overhead but did not solve restore failure.
- It is not simply "allocator trim"; no-trim helped, but did not fix the
  floor2 -> floor4 dynamic probe.
- It is not enough to only avoid `direct destroy fallback`; the retire path can
  still destroy device PG.

## Guardrails for Future Changes

Before changing code again:

1. Check the latest log for these exact lines:
   - `direct destroy fallback`
   - `retired immediately`
   - `destroy_device_pg=1`
   - `path string is NULL`
   - `Elastic full-world restore segmented timing`
   - `MC2 dispatch args: phase=post_floor_restore`
2. If any low-floor group is destroyed during `mode1_full_restore`, confirm
   whether it destroys only Python references or the underlying device PG.
3. Do not re-run full 2-epoch or 3-epoch experiments for this bug.  Use the
   short probe first:

```bash
cd /workspace/cann-recipes-train/llm_rl/qwen3_shrink_aware
./run_mode1_dynamic_floor2_to_floor4_kv_probe.sh
```

4. Success criteria for the short probe:
   - no `path string is NULL`
   - no OOM
   - step3 starts
   - floor4 KV cache is `280576`
   - floor4 shrink state shows the expected 32-slot behavior

## Current Recommended Next Fix

Do not destroy low-floor device process groups during `mode1_full_restore`.

More precise target:

- For low-floor `(14,15)`, `(12..15)`, `(8..15)` groups:
  - remove from active/cached lookup if needed
  - quarantine Python wrapper/state
  - do not call underlying device PG destroy during restore
  - avoid any `destroy_device_pg=1` for `reason=mode1_full_restore`

Then rerun only the short probe and compare:

- previous latest failing log:
  - `20260705095637`
- new log:
  - should not contain `retired immediately ... reason=mode1_full_restore ...
    destroy_device_pg=1`
  - should not contain `path string is NULL`

## Timeline Snapshot

- 2026-07-03: floor2 no-trim/small-cap tests showed trim/KV were not the only
  issue.
- 2026-07-04: focused on floor2 -> floor4 KV recovery and stale tensor cleanup.
- 2026-07-05 09:41: direct destroy fallback during full restore identified.
- 2026-07-05 09:56: full-restore defer enabled, direct fallback removed, but
  failure persists through immediate device PG retire path.

## 2026-07-05 Floor4-vs-Floor2 Root-Cause Check

Compared logs:

- floor4 success:
  - `mode1_dynamic_length_aware_adaptive_floor4_natural_tailguard_full3/epoch_001_mode1_natural/logs/wjqwen30b-a3b-record_graph_save4eagle3_20260628223407.txt`
- floor2 -> floor4 failing probe:
  - `mode1_dynamic_floor2_to_floor4_kv_probe/epoch_001_mode1_natural/logs/wjqwen30b-a3b-record_graph_save4eagle3_20260705095637.txt`

Key result:

- The floor4 run also printed `path string is NULL`, but only after rollout 5
  was dumped, checkpoint `global_step_5` was saved, final metrics were printed,
  and `trainer.fit()` returned.  That is teardown noise after success.
- The floor2 probe printed the same symptom during the active run: step1
  generation and full-world restore completed, `rollout_output_time_s` was
  printed, but no rollout dump happened and step2 never started.

Therefore the regression is not "HCCL prints path string is NULL" in general.
It is that the floor2 path triggers the HCCL teardown/failure at an active
continuation point.

Code-path confirmation:

- `_quarantine_deferred_elastic_group()` still calls
  `_retire_group_wrapper_defer_pg_destroy()`.
- `_retire_group_wrapper_defer_pg_destroy()` defaults
  `VLLM_ASCEND_MODE1_PARITY_DESTROY_DEVICE_PG_ON_RETIRE=1`.
- Thus a "deferred" low-floor group can still destroy the raw device process
  group and then be logged as:

```text
Elastic parallel group retired immediately ... reason=mode1_full_restore
destroy_device_pg=1 destroy_cpu_pg=1
```

Why floor2 is different from old floor4:

- floor4-only runs never introduce the extra two-rank `(14,15)` floor group.
- floor2 adds a 4 -> 2 stage and then restores full-world from that two-rank
  DP/EP/MC2 state.
- The current logic tries to release/retire those low-floor groups during the
  active restore path.  Even when the direct destroy fallback is removed, the
  retire path still destroys device PG by default.

Current root-cause assessment:

- Most likely root cause: communicator lifecycle bug introduced by the floor2
  path, specifically active-run destruction/retirement of low-floor HCCL device
  process groups.
- Less likely root causes for this specific failure: direct NPU expert import,
  tensor slot capacity, or KV budget.  Those can affect performance/capacity,
  but they do not explain why the run dies immediately after restore before the
  next rollout dump.

Next fix direction:

- Split "remove stale group from Python live/cache lookup" from "destroy raw
  device PG".
- During an active run, especially for `reason=mode1_full_restore`, quarantine
  wrappers and cache entries but do not destroy low-floor device PG.
- Confirm with the short probe that no low-floor group logs
  `destroy_device_pg=1` before the program reaches step3/floor4 KV profiling.

## 2026-07-05 Fix Applied: Preserve Device PG on Full Restore

Code changes:

- Added `_should_preserve_device_pg_on_retire()` in `vllm_ascend/worker/worker_v1.py`.
- Extended `_retire_group_wrapper_defer_pg_destroy()` so it knows:
  - `group_kind`
  - `group_ranks`
  - `reason`
- For low-floor mode1 groups with `reason=mode1_full_restore`, the retire path
  now preserves the raw device process group when
  `VLLM_ASCEND_MODE1_PARITY_PRESERVE_DEVICE_PG_ON_FULL_RESTORE=1`.
- The group is still removed from live/cache/registry paths, and the wrapper's
  direct references are detached, but the raw HCCL device PG is kept in the
  deferred list.
- Added `preserve_device_pg` to retire/defer logs so future runs can verify the
  exact behavior.

Scripts updated:

- `run_mode1_dynamic_floor2_to_floor4_kv_probe.sh`
- `run_mode1_local_length_sorted_e2e_adaptive_floor2.sh`

Both now explicitly export:

```bash
VLLM_ASCEND_MODE1_PARITY_PRESERVE_DEVICE_PG_ON_FULL_RESTORE=1
```

Validation before rerun:

- `python -m py_compile vllm_ascend/worker/worker_v1.py` passed.
- `bash -n run_mode1_dynamic_floor2_to_floor4_kv_probe.sh` passed.
- `bash -n run_mode1_local_length_sorted_e2e_adaptive_floor2.sh` passed.

Expected next-run evidence:

- For `reason=mode1_full_restore`, logs should show:

```text
destroy_device_pg=0 preserve_device_pg=1
```

- The probe should no longer die immediately after step1 restore.  It should at
  least reach step2 and ideally step3 floor4 KV profiling.

## 2026-07-05 Follow-up: Raw PG Preservation Was Necessary But Not Sufficient

Latest checked log:

- `mode1_dynamic_floor2_to_floor4_kv_probe/epoch_001_mode1_natural/logs/wjqwen30b-a3b-record_graph_save4eagle3_20260705101801.txt`

Observed state:

- The previous fix did take effect for the floor2 `(14,15)` DP/EP/MC2 groups:

```text
destroy_device_pg=0 preserve_device_pg=1
```

- Step1 generation and full-world restore completed:

```text
driver_restore_rpc_done elapsed_s=3.418
rollout_output_time_s: 87.130728
```

- However, the run still exited before dumping the new rollout data and before
  entering step2.  The latest `rollout_data/1.jsonl` and `2.jsonl` files were
  older files from the previous run, not outputs from this run.

Root-cause refinement:

- The device PG is now preserved, so "destroying raw HCCL device PG" is no
  longer the direct failure point.
- The retire path still detached the group wrapper internals:
  - `communicator.device_group = None`
  - `communicator.cpu_group = None`
  - `del group.device_group`
  - `del group.cpu_group`
  - `group.device_communicator = None`
- It also still destroyed CPU PGs and could still run forced cleanup after a
  full-restore floor-group release.

That creates a dangerous half-retired state: the raw HCCL PG is kept alive, but
the `GroupCoordinator`/communicator object that owns the MC2 runtime-facing
references is partially dismantled during an active restore continuation point.
This matches the symptom: post-restore MC2 warmup succeeds, then the process
dies before the trainer can dump rollouts or proceed.

Fix applied:

- Added `VLLM_ASCEND_MODE1_PARITY_PRESERVE_GROUP_REFS_ON_FULL_RESTORE=1`.
- When full-world restore preserves a low-floor group, the retire path now
  quarantines the whole group object instead of stripping its PG references.
- CPU PG destruction is skipped in this preserved-wrapper mode.
- Added `VLLM_ASCEND_MODE1_PARITY_SKIP_CLEANUP_ON_PRESERVED_FULL_RESTORE=1` to
  avoid forced cleanup on this active restore path.
- Logs now include `preserve_group_refs`.

Expected next-run evidence:

```text
reason=mode1_full_restore
destroy_device_pg=0
destroy_cpu_pg=0
preserve_device_pg=1
preserve_group_refs=1
cleanup_ms=0.00
```

The short probe should at least reach the rollout dump after step1.  If it then
fails at step3 floor4 KV profiling, that is a different issue: preserved floor2
workspace pressure versus floor4 KV capacity.

## 2026-07-05 Reset Direction: Match Floor4 Direct Release Semantics

User correction:

- Natural mode should use the same group lifecycle for floor2 and floor4.
- The previous preserve/defer experiments were useful for isolating the
  failure mode, but they are not the target design because they make floor2
  release semantics diverge from the known-working floor4 path.

Floor4 successful semantics:

- On `mode1_pre_rebuild`, old floor groups are not cached.
- On `mode1_full_restore`, the live shrink-floor DP/EP/MC2 groups are not
  cached.
- The old group is released by:

```text
group.destroy()
_detach_destroyed_group_references(group)
_mark_retired_elastic_group(group)
_remove_elastic_parallel_group_registry_entry(...)
_cleanup_after_elastic_group_release(...)
```

- The target group is then rebuilt with `init_model_parallel_group(...)`.

Code reset:

- Default preservation is now disabled:

```text
VLLM_ASCEND_MODE1_PARITY_PRESERVE_DEVICE_PG_ON_FULL_RESTORE=0
VLLM_ASCEND_MODE1_PARITY_PRESERVE_GROUP_REFS_ON_FULL_RESTORE=0
VLLM_ASCEND_MODE1_PARITY_SKIP_CLEANUP_ON_PRESERVED_FULL_RESTORE=0
```

- Floor2 test scripts now default to direct destroy, matching the successful
  floor4 lifecycle:

```text
VLLM_ASCEND_MODE1_PARITY_DEFER_GROUP_DESTROY=0
VLLM_ASCEND_MODE1_PARITY_DEFER_GROUP_DESTROY_ON_FULL_RESTORE=0
VLLM_ASCEND_MODE1_PARITY_DEFER_GROUP_DESTROY_ON_PRE_REBUILD=0
VLLM_ASCEND_MODE1_PARITY_DEFER_GROUP_DESTROY_ON_STASH=0
```

New low-frequency diagnostics:

- Added `Mode1 direct group destroy start/done` logs around both direct destroy
  call sites.
- The logs include:
  - rank, attr, group kind, group ranks, reason
  - group id, communicator id
  - group/communicator device and CPU PG object ids
  - whether each ref existed before/after destroy
  - `destroy_call_ms`, `detach_ms`, `registry_ms`, total destroy time
  - whether cleanup is forced

Expected next-run interpretation:

- If the last `start` log has no matching `done`, the failure is inside
  `group.destroy()`.
- If `done` exists but no cleanup/restore completion follows, the failure is
  after detach/registry removal or during cleanup.
- If all direct destroy logs complete and full-world restore logs complete, but
  the run fails before rollout dump, the remaining culprit is after restore
  continuation rather than group release itself.
