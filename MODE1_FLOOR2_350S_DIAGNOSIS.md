# Mode=1 floor=2 350s shrink stall diagnosis

This note records what has been ruled out and what remains plausible for the local-only 350s stall when shrinking from 4 ranks to 2 ranks (`active_ranks=[14, 15]`). The goal is to avoid circling back to experiments that already failed.

## Final resolution, 2026-06-11

### Final verified run

The local fixed run is:

- Log: `/workspace/cann-recipes-train/llm_rl/qwen3_nm_model1_final/wjeagerqwen30b-a3b-with_draft_breakdown_20260611212131628667998_elastic.txt`
- Header confirms `mode=1 floor=2`.
- Mode1 init confirms `floor=2 active_local=8 loaded_local=64 loaded_capacity=64`.
- KV cache increased to `174,336 tokens`, with maximum concurrency `10.01x` for `17,408` tokens/request.
- Final `4->2` shrink to `active_ranks=[14, 15]` no longer has the 350s stall:
  - rank 12 sender: `send_export_runtime_slot_map_ms=0.00`, `send_export_ms=0.95`, `send_post_ms=57.36`, `wall_ms=58.96`.
  - rank 13 sender: `send_export_runtime_slot_map_ms=0.00`, `send_export_ms=0.90`, `send_post_ms=49.97`, `wall_ms=51.53`.
  - rank 14 receiver: `recv_direct_slot_chunks=4`, `wall_ms=54.75`.
  - rank 15 receiver: `recv_direct_slot_chunks=4`, `wall_ms=54.94`.
- Final active ranks finish the `4->2` shrink in about `1.3-1.4s`:
  - rank 15: `total_ms=1307.62`.
  - rank 14: `total_ms=1422.97`.
- End-to-end `rollout_output_time_s=95.579743`.

### Root cause identified

The 350s delay was not P2P bandwidth, KV cache capacity, CPU shadow transfer, rebuild, refresh, or warmup. It was triggered by the direct-NPU shrink export path when it read runtime expert slot metadata from NPU tensors.

The decisive bad-run evidence is from `wjeagerqwen30b-a3b-with_draft_breakdown_20260610153952567373251_elastic.txt`:

- rank 13 final `4->2` sender:
  - `send_export_ms=348974.99`
  - `send_export_source_select_ms=348967.14`
  - `send_export_runtime_slot_map_ms=348966.97`
  - `send_post_ms=55.89`
  - `send_wait_ms=0.19`
- rank 12 final `4->2` sender:
  - `send_export_ms=349057.77`
  - `send_export_source_select_ms=349050.31`
  - `send_export_runtime_slot_map_ms=349050.16`
  - `send_post_ms=60.79`
  - `send_wait_ms=0.18`

This pins the large delay to runtime slot-map lookup before P2P send is posted. The actual P2P post/wait was tens of ms or less, and the standalone rank-pair P2P/alias benchmark was healthy.

The most accurate conclusion is:

- The Python/C++ trigger point is the NPU scalar/runtime slot lookup in direct-NPU export.
- On this local machine, that lookup acts as a synchronization boundary and exposes a pending HCCL/NPU runtime state, causing an approximately 349-351s timeout/retry-scale wait.
- The exact underlying CANN/HCCL event that waits for 350s is not proven from Python logs alone; it would require CANN/HCCL slog/plog/profiler correlation. However, the responsible hot-path boundary is proven by timing fields and by the fix.

### Why identical remote branch behaved differently across machines

The same remote branch reproduced the stall locally, while another host ran it normally. The code therefore contained a latent environment-sensitive synchronization hazard rather than a purely Python-level logic mismatch.

The risky assumption in the remote code was that reading a tiny NPU scalar slot map during shrink export is cheap. On the healthy host, that read returned quickly. On this host, after the full rollout context and repeated `16->8->4->2` dynamic HCCL group transitions, the same read synchronized with an unhealthy or delayed runtime state and waited for the 350s timeout/retry window.

Likely contributors include CANN, driver, firmware, `torch_npu`, HCCL, rank/device topology, HCCL workspace availability, and allocator fragmentation differences. The fix removes this environment-sensitive synchronization point from the shrink hot path.

### Final fix

The final solution keeps the payload transfer NPU-to-NPU direct, but moves slot metadata lookup off the NPU hot path.

Key changes:

1. `vllm_ascend/ops/fused_moe.py`
   - Maintain CPU mirrors for expert slot maps such as `expert_map` / `loaded_expert_map`.
   - Use CPU metadata for `export_lossless_expert_npu_weights()` slot selection.
   - Avoid NPU scalar `.item()` fallback in the mode=1 export hot path unless explicitly forced for debugging.

2. `vllm_ascend/worker/worker_v1.py`
   - Use batched/chunked direct-NPU import/export for mode=1 shrink.
   - Default to batch direct-NPU import and disallow scalar direct-NPU import in the verified mode=1 path.
   - Add detailed timing fields proving whether CPU or NPU slot-map lookup was used:
     - `send_export_runtime_slot_map_ms`
     - `send_export_runtime_slot_cpu_hits`
     - `send_export_runtime_slot_npu_items`
     - `slot_map_expert_cpu_ready`
     - `slot_map_loaded_cpu_ready`

3. `worker_v1.py` / launch script
   - Disable mode=1 post-shrink DP all-reduce synthetic warmup by default.
   - This avoids a separate local HCCL workspace/OOM failure exposed after the 350s export stall was removed.

4. Launch script defaults
   - `VLLM_ASCEND_MODE1_BATCH_DIRECT_NPU_IMPORT=1`
   - `VLLM_ASCEND_MODE1_ALLOW_SCALAR_DIRECT_NPU_IMPORT=0`
   - `VLLM_ASCEND_MODE1_ALLOW_BATCH_INDEX_SELECT_EXPORT=0`
   - `VLLM_ASCEND_MODE1_DIRECT_NPU_IMPORT_BATCH_EXPERTS=8`
   - `VLLM_ASCEND_MODE1_PARITY_ENABLE_POST_SHRINK_DP_WARMUP=0`

### Why this fixes the issue

Before the fix, final `4->2` shrink effectively did:

1. Ask NPU slot-map tensors which slot each expert occupies.
2. That scalar read synchronized with local runtime/HCCL state and could wait about 350s.
3. Only after that did P2P send happen, which was actually fast.

After the fix:

1. CPU mirror metadata already knows the expert-to-slot mapping.
2. The shrink path constructs the direct NPU transfer from CPU metadata.
3. Payload still moves NPU-to-NPU; CPU is not used for weight payload transfer.
4. The dangerous NPU scalar synchronization boundary disappears.

This is why the latest fixed log has `send_export_runtime_slot_map_ms=0.00`, final shrink completes in about `1.3-1.4s`, KV cache increases to `174,336 tokens`, and rollout time improves to about `95.58s`.


## Current baseline

Known healthy reference:

- Log: `/workspace/cann-recipes-train/llm_rl/qwen3/ref_mode1_small.txt`
- Same mode/floor/data/model class, remote host.
- KV cache: `156,672 tokens`.
- Final 4->2 shrink `stream_import_ms` is about 1.2-1.5s on ranks 12-15.
- Final active ranks finish in about 4.0-4.1s.
- End-to-end `rollout_output_time_s=120.814821`.

Local abnormal behavior:

- Logs: `wjeagerqwen30b-a3b-with_draft_breakdown_20260610101504051821160_elastic.txt`, `wjeagerqwen30b-a3b-with_draft_breakdown_20260610111421673056302_elastic.txt`, `wjeagerqwen30b-a3b-with_draft_breakdown_20260610123219116207620_elastic.txt`.
- Same remote branch code also reproduces the stall on this machine.
- Final 4->2 shrink `stream_import_ms` is about 350-351s on ranks 12-15.
- `rollout_output_time_s` stays around 450-451s.
- The large time is concentrated in shrink preload/import, not rollout compute itself.

Representative local evidence from `20260610123219116207620`:

- `active_ranks=[14, 15]` entered at line 4771.
- `stream_import_ms=351245.59` on rank 15 at line 4791.
- `stream_import_ms=351283.87` on rank 14 at line 4797.
- `stream_import_ms=351513.65` on rank 13 at line 4803.
- `stream_import_ms=351551.76` on rank 12 at line 4809.
- Final active rank total remains about `353503ms` at lines 4879 and 4884.

## Code state after rollback

The diagnostic changes that altered the shrink path have been rolled back:

- `vllm_ascend/worker/worker_v1.py` is back to branch content for the old-floor group drop path.
- `internal/wj_train_grpo_qwen30b_a3b_16die_true_weight_eager.sh` is back to `VLLM_ASCEND_CUSTOM_MODE1_KV_MATERIALIZE_HEADROOM_BYTES=0`.
- `python3 -m py_compile vllm_ascend/worker/worker_v1.py` passed.
- `bash -n run_mode1_floor2_local.sh internal/wj_train_grpo_qwen30b_a3b_16die_true_weight_eager.sh` passed.

The remaining untracked files are independent local helpers/logs, not changes to the branch code path.

## Ruled out or deprioritized

### 1. Code branch mismatch as the primary cause

Evidence:

- The remote `nm_mode1_final` code was checked out into `qwen3_nm_model1_final` and run locally.
- The same 350s-class final 4->2 shrink reproduced locally.

Conclusion:

- The local slowdown is not explained only by our earlier local code modifications.
- Code differences can still matter for instrumentation, but identical branch behavior across machines points strongly at runtime/environment/topology/state interactions.

### 2. Raw rank-pair P2P bandwidth between 12/13 and 14/15

Evidence:

- Micro-benchmark: `rank_pair_p2p_alias_runs/20260610T004205652031398/bench.log`.
- Tested pairs: `12->14`, `13->15`, `12->15`, `13->14`.
- Payload per measured pair: about 28.99 GB.
- Results:
  - `12->14`: `send_total_ms=888.38`, `recv_total_ms=841.26`, `e2e_GBps=32.63`.
  - `13->15`: `send_total_ms=884.86`, `recv_total_ms=893.19`, `e2e_GBps=32.46`.
  - `12->15`: `send_total_ms=1027.97`, `recv_total_ms=1085.56`, `e2e_GBps=26.71`.
  - `13->14`: `send_total_ms=878.66`, `recv_total_ms=869.94`, `e2e_GBps=32.99`.

Conclusion:

- Bare HCCL P2P plus zero-offset alias is not intrinsically 350s slow on these rank pairs.
- The stall needs the full training/shrink context, model buffers, prior MoE dispatch, dynamic groups, or allocator state to reproduce.

### 3. KV cache capacity or simple memory headroom pressure

Evidence:

- With 4 GiB headroom in both mode1 materialization and KV init paths, the local run still stalled.
- Log: `wjeagerqwen30b-a3b-with_draft_breakdown_20260610101504051821160_elastic.txt`.
- Header line 2 shows `custom_mode1_kv_headroom=4294967296` and `kv_cache_init_headroom=4294967296`.
- Final 4->2 still shows `stream_import_ms=350421-350698ms` and `rollout_output_time_s=451.184473`.

Conclusion:

- Simple extra headroom does not avoid the stall.
- Memory pressure or fragmentation may still contribute, but not in a way fixed by reserving a few GiB at startup.

### 4. Never releasing the old floor group after 8->4

Evidence:

- Diagnostic run skipped old-floor group cache drop after shrinking to `[12, 13, 14, 15]`.
- Log: `wjeagerqwen30b-a3b-with_draft_breakdown_20260610103554839850013_elastic.txt`.
- Skip log appears around lines 4553-4594.
- The job did not reach the `[14, 15]` shrink.
- It failed during the subsequent dp=4 stage with an AICORE timeout in `MoeDistributeDispatchV2` around line 4862.

Conclusion:

- We cannot fix this by simply retaining old 8-rank floor groups.
- The stale groups/resources are still capable of corrupting or blocking normal dp=4 MoE dispatch.
- Old-floor cleanup is necessary before continuing execution at the new floor.

### 5. Simple NPU synchronize before/after old-floor group drop

Evidence:

- Run with sync before old-floor drop: `wjeagerqwen30b-a3b-with_draft_breakdown_20260610111421673056302_elastic.txt`.
- Run with sync before and after old-floor drop: `wjeagerqwen30b-a3b-with_draft_breakdown_20260610123219116207620_elastic.txt`.
- Sync times are consistently small, roughly 0.1-0.3ms.
- Final 4->2 still stalls with `stream_import_ms=350811-351107ms` in the before-sync run.
- Final 4->2 still stalls with `stream_import_ms=351245-351551ms` in the before+after-sync run.

Conclusion:

- The failure is not a simple “old group destroy raced an obvious outstanding stream” problem that `torch.npu.synchronize()` can drain at the old-floor drop site.
- If pending work is involved, it is likely bound to a different stream, communicator, runtime queue, or delayed error path that is only forced during the later export/import step.

### 6. Rebuild, refresh, warmup, or old-floor drop itself as the 350s component

Evidence:

- In the bad local 4->2 shrink, `rebuild_ms` is tens of ms, `refresh_ms` is hundreds of ms, `warmup_ms` is under about 1s, and `drop_old_floor_group_cache_ms` is about 0.5-0.7s.
- The dominant field is `preload_import_ms` / `stream_import_ms`, about 351s.

Conclusion:

- The 350s should be investigated inside the shrink preload/import/export path, especially direct-NPU expert weight export/import and its interaction with HCCL/allocator/runtime state.

### 7. CPU payload filtering or CPU-side staging as the main stall

Evidence:

- The bad 4->2 logs show `filter_cpu_payload_ms=0.02-0.03` and `store_ms<0.2ms`.
- The selected path is direct NPU import.

Conclusion:

- The 350s is not in CPU payload filtering or CPU store bookkeeping.
- The CPU shadow discussion is not the main path for this specific stall.

## Still plausible hypotheses

### A. Direct-NPU export from real model expert slots is context-sensitive

Why it remains plausible:

- Bare alias/P2P micro-benchmark is healthy.
- Full model shrink stalls only after prior rollout and after reaching the 4->2 transition.
- Earlier detailed comparisons mapped the 350s to `send_export_ms`, which is the export/alias side before P2P wait.

Next test:

- Add a narrow diagnostic around `export_lossless_expert_npu_weights()` / `_npu_zero_offset_alias_for_p2p()` using actual model modules on ranks 12 and 13 just before 4->2 transfer.
- Log per-layer and per-slot export time, storage offset, data pointer delta, NPU format, tensor shape, and whether alias construction or a sync inside export consumes the time.
- Run one variant that exports actual slots but does not send them, to separate export/alias from HCCL P2P.

Expected interpretation:

- If export-only stalls, focus on alias/format/allocator/model-buffer state.
- If export-only is fast but export+send stalls, focus on HCCL communicator or P2P interaction in the full group context.

### B. The stall is a hidden device/runtime timeout, not real data transfer

Why it remains plausible:

- 350s looks like a timeout/retry scale, not bandwidth for about 29GB.
- In the skip-old-group experiment, the runtime surfaced an AICORE timeout in `MoeDistributeDispatchV2`.
- The bad run may be waiting on a prior runtime or HCCL error that is only reported/synchronized during export/import.

Next test:

- Inspect Ascend slog/plog and host `dmesg` around the exact local timestamps `12:36:54-12:42:47` from the `202606101232` run.
- Search for HCCL, RDMA, AICORE, stream timeout, device reset, and retry messages.
- Compare with the remote healthy host over the same interval type.

Expected interpretation:

- If logs show a 300-350s HCCL/device timeout, prioritize environment/driver/HCCL/topology rather than Python-level ordering.

### C. HCCL/MC2 dynamic group state diverges after repeated shrink groups

Why it remains plausible:

- Raw P2P in one group is healthy.
- Full training builds many DP/EP/MC2 groups and caches across full-world, 8-rank, 4-rank, and 2-rank floors.
- Skipping old groups breaks dp=4 MoE dispatch, proving stale group state matters.
- Simple sync around destruction did not fix it, so the issue may be in group identity/refcount/cache selection rather than timing alone.

Next test:

- Log all cached DP/EP/MC2 group signatures and object IDs before and after each shrink.
- Confirm the active `[14,15]` group is newly built or cache-hit as expected, and that stale `[12,13,14,15]` groups are not selected by the export/import path.
- Compare group counts/signatures between local and remote healthy logs.
- Try one knob at a time:
  - `VLLM_ASCEND_DISABLE_ELASTIC_MC2_GROUP_CACHE=1`
  - `VLLM_ASCEND_MODE1_PARITY_SINGLE_LIVE_MC2_GROUP=1`
  - `VLLM_ASCEND_MODE1_PARITY_KEEP_MC2_GROUP_CACHE=1`

Expected interpretation:

- If a cache knob removes 350s, the bug is in communicator lifecycle/cache reuse.
- If all knobs preserve 350s, the problem is likely below cache policy or inside export/runtime.

### D. Rank/device topology or local HCCL fabric has a context-specific issue

Why it remains plausible:

- The same code is healthy on another machine but slow locally.
- Raw P2P is healthy in isolation, but full context may use different communicator setup, stream type, or HCCL algorithm.
- The affected final pairs are exactly `12->14` and `13->15` for 4->2 paired import.

Next test:

- Compare local and remote:
  - `npu-smi info`
  - `npu-smi topo -m` if available
  - CANN version
  - firmware/driver version
  - `torch`, `torch_npu`, `hccl` package versions
  - `HCCL_*`, `ASCEND_*`, `PYTORCH_NPU_*` environment variables
  - rank-to-device mapping and visible device ordering
- Extend the micro-benchmark to initialize the same sequence of full-world, 8-rank, 4-rank, and 2-rank groups before running the pair transfer.

Expected interpretation:

- If the extended group-sequence benchmark reproduces the stall locally but not remotely, the issue is HCCL dynamic group lifecycle/topology rather than model code.

### E. Allocator/memory fragmentation in real model process

Why it remains plausible:

- Startup headroom did not help, but the real process allocates KV cache, model weights, temporary MoE buffers, and dynamic aliases before 4->2.
- The micro-benchmark uses clean synthetic tensors and does not reproduce the model allocator state.

Next test:

- Log NPU memory summary before 8->4, after 8->4, before 4->2 preload, after 4->2 preload.
- Add a diagnostic that pre-allocates all export alias buffers once at model init and reuses them during shrink, avoiding allocator work in the hot path.
- Add export-only timing on real module buffers with and without constructing fresh zero-offset aliases.

Expected interpretation:

- If preallocated alias/staging buffers remove the 350s, focus on allocator/format-cast/data-ptr alias creation.

### F. Export path waits on prior MoE dispatch work from the dp=4 stage

Why it remains plausible:

- The 350s appears only at the later 4->2 shrink, after dp=4 MoE dispatch has run.
- Sync around old-floor group drop happens immediately after the previous shrink, not immediately before the later export operation.

Next test:

- Add an optional `torch.npu.synchronize()` at the very beginning of the 4->2 preload/import path, before the first actual export/alias operation.
- Log whether the 350s moves from `stream_import_ms` into this pre-export sync.

Expected interpretation:

- If pre-export sync takes 350s, the stall is pending device work from dp=4 compute/MoE dispatch.
- If pre-export sync is fast and export still takes 350s, the stall is inside export/alias/HCCL itself.

### G. Environment-level timeout/retry threshold around 350s

Why it remains plausible:

- Multiple runs consistently cluster around 350s.
- Consistent wall time suggests a timeout or retry threshold rather than normal data movement.

Next test:

- Search CANN/HCCL configuration for timeout defaults and compare remote/local env.
- Check whether changing HCCL timeout/debug settings changes the stall duration or logs a clearer error.

Expected interpretation:

- If the delay tracks a timeout setting, the primary issue is a lower-level HCCL/runtime failure masked as a long wait.

## Suggested next experiment order

1. Add pre-export sync at the start of the 4->2 preload path and see whether the 350s moves there.
2. Add export-only timing on actual model expert slots, without P2P send/recv.
3. Extend the P2P micro-benchmark to build the same full-world/8/4/2 group sequence before testing pairs.
4. Collect and compare Ascend runtime logs around the 350s window on local and healthy remote hosts.
5. Compare CANN/driver/torch_npu/HCCL/env/topology/rank mapping between the two machines.
6. If export-only points at alias/allocator, try preallocated alias/staging buffers for the exact expert slot shapes.

## Practical notes

- Do not use `MODE1_SKIP_OLD_FLOOR_DROP_BEFORE_NEXT_SIZE=2` again for performance runs. It corrupts/breaks the dp=4 stage before the final shrink.
- Do not spend more time on plain headroom-only sweeps unless memory logs show a new signal.
- Do not focus on CPU payload filtering/store for this stall; the measured CPU-side times are negligible.
- The highest-value instrumentation is now immediately before and inside the 4->2 direct-NPU export/import path.

## Instrumentation added for next run

The current local diagnostic patch adds narrow instrumentation for the final `active_ranks=[14, 15]` shrink only.

Recommended next run:

```bash
cd /workspace/cann-recipes-train/llm_rl/qwen3_nm_model1_final
MODE1_DIAG_PRELOAD_ENTRY_SYNC=1 \
MODE1_DIAG_DIRECT_NPU_EXPORT_DRY_RUN=1 \
MODE1_DIAG_EXPORT_INTERNAL_TIMING=1 \
./run_mode1_floor2_local.sh
```

What to inspect:

- `Elastic shrink preload entry sync`: if this line is about 350s, the stall is pending device work before shrink preload starts.
- `Elastic shrink direct NPU export dry-run`: if this line is about 350s before any real send/recv, the stall is actual model-slot export/alias without HCCL transfer.
- `Elastic shrink stream import breakdown` for `active_ranks=[14, 15]`:
  - `send_export_ms` / `send_export_alias_*_ms`: source-side export/alias path.
  - `send_post_ms`: `isend` posting path.
  - `send_wait_ms`: sender-side HCCL wait.
  - `recv_alias_ms`: target slot receive-buffer alias path.
  - `recv_post_ms`: `irecv` posting path.
  - `recv_wait_ms`: receiver-side HCCL wait.

Interpretation matrix:

- `preload entry sync ~=350s`: prior dp=4 compute/MoE work is not drained before shrink preload.
- `dry-run export ~=350s`, but entry sync is fast: direct export/zero-offset alias over real model slots is the culprit.
- `dry-run export fast`, `send_export_ms ~=350s`: export stalls only when executed in the real send loop, likely due to order/group/runtime state.
- `send_post_ms` or `recv_post_ms ~=350s`: HCCL P2P operation creation/posting is blocked.
- `send_wait_ms` or `recv_wait_ms ~=350s`: HCCL transfer/wait path is blocked.
- All fields small but total still 350s: add per-layer logging next; the stall is not in layer 0.

## 2026-06-10 14:39 diagnostic run result

Log: `wjeagerqwen30b-a3b-with_draft_breakdown_20260610143958411968344_elastic.txt`.

New conclusions:

- `Elastic shrink preload entry sync` is fast on final 4->2: rank 12-15 are about 1.47-4.52ms. This rules out a simple pending-device-work drain at preload entry.
- First dry-run export on source ranks is fast: rank 13 export is `0.43ms`, rank 12 export is `0.48ms` on layer 0. This rules out “any real model-slot export is always slow”.
- Receiver ranks 14/15 enter final 4->2 layer 0 quickly: `recv_alias_ms≈5.10ms`, `recv_post_ms≈57ms`, `recv_wait_ms≈0.15ms`, total about 63ms.
- Source ranks 12/13 spend almost the entire 350s in `send_export_ms`: rank 13 `send_export_ms=349438.43ms`, rank 12 `send_export_ms=349446.75ms`.
- Within the previously instrumented export internals, `send_export_map_ms`, `send_export_view_ms`, and `send_export_alias_*_ms` are tiny. Therefore the 350s is before those measurements, inside the export source-selection/runtime-slot lookup section.

Most likely narrowed culprit:

- `export_lossless_expert_npu_weights()` enters the branch that computes `runtime_slots = [int(self.expert_map[expert_id].item()) ...]` before the previous internal timers started.
- The first dry-run expert is fast, so the stall may be expert-specific, call-order-specific, or caused by a delayed runtime synchronization on a later `expert_map.item()` call.

Instrumentation added after this run:

- `source_select_ms` and `runtime_slot_map_ms` are now included in `_last_lossless_npu_export_debug`.
- `send_export_source_select_ms` and `send_export_runtime_slot_map_ms` are now aggregated in the layer breakdown.
- A per-expert slow-export log is emitted when a final 4->2 single expert export exceeds `VLLM_ASCEND_MODE1_DIAG_SLOW_EXPORT_LOG_MS`, default `1000ms`.

Recommended next run remains:

```bash
cd /workspace/cann-recipes-train/llm_rl/qwen3_nm_model1_final
MODE1_DIAG_PRELOAD_ENTRY_SYNC=1 \
MODE1_DIAG_DIRECT_NPU_EXPORT_DRY_RUN=1 \
MODE1_DIAG_EXPORT_INTERNAL_TIMING=1 \
./run_mode1_floor2_local.sh
```

Optional lower threshold if we want more per-expert detail:

```bash
MODE1_DIAG_SLOW_EXPORT_LOG_MS=100
```

Expected next interpretation:

- If slow-export logs show `runtime_slot_map_ms≈350s`, the next fix should avoid per-export NPU scalar `.item()` on `expert_map` in this path, likely by caching/exporting a CPU-side expert-to-slot map before the shrink transfer loop.
- If `source_select_ms≈350s` but `runtime_slot_map_ms` is small, inspect source tensor selection or runtime buffer state checks.
- If both are small but `export_ms` is still huge, add even earlier per-line timing or suspect a Python/runtime scheduling artifact around function entry/exit.

## 2026-06-10 follow-up patch: CPU slot-map export experiment

Why this is the next highest-probability check:

- Latest 14:39 run proved raw pair P2P, receiver `irecv`, sender `isend/wait`, and zero-offset alias are not the 350s source.
- The 350s remains inside source-side `send_export_ms` on ranks 12/13 during final `active_ranks=[14, 15]`.
- Previously timed `map/view/alias` sections inside export are tiny, so the remaining unmeasured region is source selection and runtime slot lookup.
- The suspicious operation is repeated NPU scalar access, especially `expert_map[expert_id].item()`, during the formal send loop. This can hide a device synchronization even when the actual P2P link is healthy.

Code changes now added:

- `export_lossless_expert_npu_weights()` now reports:
  - `source_select_ms`
  - `runtime_slot_map_ms`
  - `runtime_slot_lookup_source`
  - `export_slot_lookup_source`
- Worker stream-import breakdown now reports:
  - `slot_cache_refresh_ms`
  - `send_export_source_select_ms`
  - `send_export_runtime_slot_map_ms`
- New guarded experiment switch:
  - `VLLM_ASCEND_MODE1_USE_CPU_EXPORT_SLOT_MAP=1`
  - Wrapper alias: `MODE1_USE_CPU_EXPORT_SLOT_MAP=1`

Important: this does not move expert weights through CPU. It only snapshots tiny expert-to-slot metadata maps to CPU before the source-side direct-NPU export loop. Weight tensors still use NPU views and HCCL P2P.

Recommended next run:

```bash
cd /workspace/cann-recipes-train/llm_rl/qwen3_nm_model1_final
MODE1_USE_CPU_EXPORT_SLOT_MAP=1 \
MODE1_DIAG_PRELOAD_ENTRY_SYNC=1 \
MODE1_DIAG_DIRECT_NPU_EXPORT_DRY_RUN=1 \
MODE1_DIAG_EXPORT_INTERNAL_TIMING=1 \
MODE1_DIAG_SLOW_EXPORT_LOG_MS=100 \
./run_mode1_floor2_local.sh
```

Interpretation:

- If `slot_cache_refresh_ms` is small and `send_export_ms` drops from ~350s to seconds, the root cause is repeated NPU map scalar lookup during final 4->2 export. The production fix should maintain CPU-side expert slot metadata when elastic maps are built/refreshed, not snapshot from NPU at transfer time.
- If `slot_cache_refresh_ms` itself is ~350s, the first CPU/NPU map touch at final 4->2 is blocked by hidden synchronization. The next fix should move CPU map construction earlier, before the problematic shrink boundary, or derive it from existing CPU assignment payloads.
- If `slot_cache_refresh_ms` is small but `send_export_ms` still ~350s, inspect per-expert `Elastic shrink direct NPU slow export` logs. The culprit then is not map lookup and we should instrument source tensor slicing/alias return boundary even more narrowly.
- If `send_export_source_select_ms` or `send_export_runtime_slot_map_ms` accounts for the 350s with `lookup_source=npu_item`, the CPU slot-map switch should be promoted from diagnostic to the main mode=1 path after validating correctness.

## 2026-06-10 15:39 run: root cause confirmed to NPU map scalar lookup

Log: `wjeagerqwen30b-a3b-with_draft_breakdown_20260610153952567373251_elastic.txt`.

Critical lines:

- rank 13, final `active_ranks=[14, 15]`, slow expert `expert_id=9`:
  - `export_ms=348961.28ms`
  - `source_select_ms=348960.37ms`
  - `runtime_slot_map_ms=348960.34ms`
  - `map/view/alias` remain sub-ms to low-ms.
- rank 12, final `active_ranks=[14, 15]`, slow expert `expert_id=1`:
  - `export_ms=349043.55ms`
  - `source_select_ms=349042.78ms`
  - `runtime_slot_map_ms=349042.74ms`.
- The preceding dry-run expert is fast on both ranks:
  - rank 12 `expert_id=0`: `runtime_slot_map_ms=0.076ms`
  - rank 13 `expert_id=8`: `runtime_slot_map_ms=0.073ms`.

Conclusion:

The 350s is not HCCL P2P transfer, not `isend/irecv`, not zero-offset alias creation, and not rebuild/warmup. It is caused by source-side NPU scalar lookup in `export_lossless_expert_npu_weights()`, specifically the runtime map lookup equivalent to `self.expert_map[expert_id].item()` during the final 4->2 direct-NPU export loop.

Fix candidate now implemented:

- Mode=1 export now prefers CPU-side expert-to-slot metadata by default.
- The CPU map is maintained from Python `active_expert_ids` during lossless activation, so it does not need to be copied back from the NPU map tensor at final shrink time.
- Weight data remains direct NPU view + HCCL P2P. Only tiny metadata lookup moves to CPU.
- Safety switch:
  - default for mode=1: `VLLM_ASCEND_MODE1_USE_CPU_EXPORT_SLOT_MAP=1`
  - disable for A/B: `MODE1_USE_CPU_EXPORT_SLOT_MAP=0 ./run_mode1_floor2_local.sh`

Recommended verification run:

```bash
cd /workspace/cann-recipes-train/llm_rl/qwen3_nm_model1_final
MODE1_DIAG_PRELOAD_ENTRY_SYNC=1 \
MODE1_DIAG_DIRECT_NPU_EXPORT_DRY_RUN=1 \
MODE1_DIAG_EXPORT_INTERNAL_TIMING=1 \
MODE1_DIAG_SLOW_EXPORT_LOG_MS=100 \
./run_mode1_floor2_local.sh
```

Success criteria:

- Final `active_ranks=[14, 15]` source ranks should show `runtime_slot_lookup_source='expert_cpu'` in dry-run / slow-export debug.
- `send_export_runtime_slot_map_ms` should be near ms-level instead of ~349000ms.
- `send_export_ms` for final 4->2 should drop from ~349s to seconds-level.
- `rollout_output_time_s` should move back toward the remote healthy ~120s baseline.

## 2026-06-10 patch refinement: avoid diagnostic NPU map refresh on fixed path

After the root-cause confirmation, the fix was refined to avoid a false-positive workaround:

- `VLLM_ASCEND_MODE1_USE_CPU_EXPORT_SLOT_MAP=1` now uses CPU metadata maintained during activation.
- It no longer triggers NPU `expert_map.detach().cpu().tolist()` refresh before export.
- The old NPU-copy refresh is now behind a separate diagnostic-only switch:
  - `VLLM_ASCEND_MODE1_DIAG_REFRESH_NPU_EXPORT_SLOT_CACHE=1`
- The local run script defaults `VLLM_ASCEND_MODE1_USE_CPU_EXPORT_SLOT_MAP=1` for mode=1 testing.

Updated verification focus:

- `slot_cache_refresh_ms` should normally remain `0.00` because the fixed path does not copy the map from NPU.
- `runtime_slot_lookup_source` should be `expert_cpu` for final 4->2 source exports.
- `send_export_runtime_slot_map_ms` should be tiny; if it is still ~349s and lookup source is `npu_item`, the CPU metadata was not populated on the prior activation path.

## 2026-06-10 extra verification counters

Added one more diagnostic layer for the verification run:

- Layer-0 activation now logs `Mode1 CPU export slot map ready` when the CPU-side expert map is populated from Python `active_expert_ids`.
- Stream-import breakdown now includes:
  - `send_export_runtime_slot_cpu_hits`
  - `send_export_runtime_slot_npu_items`
  - `send_export_slot_cpu_hits`
  - `send_export_slot_npu_items`

Expected fixed final 4->2 signature:

- On source ranks 12/13, `send_export_runtime_slot_cpu_hits` should match the number of exported experts and `send_export_runtime_slot_npu_items=0`.
- If `send_export_runtime_slot_npu_items>0`, CPU metadata was missing or invalidated and export fell back to the known slow NPU scalar `.item()` path.
- If `send_export_runtime_slot_npu_items=0` but `send_export_ms` is still high, the remaining stall is no longer the map lookup and should be re-profiled from the new breakdown.

## 2026-06-10 update: latest verified signal

Latest diagnostic log:

- `/workspace/cann-recipes-train/llm_rl/qwen3_nm_model1_final/wjeagerqwen30b-a3b-with_draft_breakdown_20260610153952567373251_elastic.txt`

New evidence:

- The final 4->2 shrink still stalls at about 349s on source ranks 12 and 13.
- The stall is no longer just attributed to aggregate `send_export_ms`; it is inside `export_lossless_expert_npu_weights()` source selection.
- Rank 13, layer 0, expert 9:
  - `export_ms=348961.28`
  - `source_select_ms=348960.37`
  - `runtime_slot_map_ms=348960.34`
  - `map_ms=0.16`, `view_ms=0.07`, `alias_w13_ms=0.50`, `alias_w2_ms=0.09`
- Rank 12, layer 0, expert 1:
  - `export_ms=349043.55`
  - `source_select_ms=349042.78`
  - `runtime_slot_map_ms=349042.74`
  - `map_ms=0.12`, `view_ms=0.04`, `alias_w13_ms=0.44`, `alias_w2_ms=0.08`
- The dry-run first expert before the formal loop is fast:
  - rank 12 expert 0: `runtime_slot_map_ms=0.076ms`
  - rank 13 expert 8: `runtime_slot_map_ms=0.073ms`

Conclusion:

- The 350s wait is not raw HCCL P2P transfer, not send/recv wait, not zero-offset alias construction, and not CPU payload handling.
- The strongest current root-cause candidate is NPU scalar map lookup in the real export path: `expert_map[expert_id].item()` or an equivalent NPU map read is synchronizing on delayed NPU work from the preceding dp=4 stage.
- Because the formal loop stalls on the second exported expert but dry-run first expert is fast, a single early dry-run is insufficient to validate the full path.

Code changes made after this evidence:

- `vllm_ascend/ops/fused_moe.py`
  - Maintains CPU mirrors for `expert_map` and `loaded_expert_map` at initialization, restore, activation, and known map replacement points.
  - In mode=1 export, CPU slot-map lookup is the default.
  - In mode=1 export, fallback to NPU scalar `.item()` is disabled by default. If the CPU mirror is missing/stale, the code now raises a clear error instead of silently entering the known 350s stall path.
  - Added `lossless_export_slot_map_status()` to report CPU mirror readiness and tensor ids.
- `vllm_ascend/worker/worker_v1.py`
  - Logs direct-NPU export slot-map readiness on source ranks before final 4->2 export.
  - Adds breakdown fields for CPU/NPU slot lookup hits and slot-map readiness.

Next validation command:

```bash
cd /workspace/cann-recipes-train/llm_rl/qwen3_nm_model1_final
MODE1_DIAG_DIRECT_NPU_EXPORT_DRY_RUN=1 \
MODE1_DIAG_EXPORT_INTERNAL_TIMING=1 \
MODE1_DIAG_SLOW_EXPORT_LOG_MS=100 \
./run_mode1_floor2_local.sh
```

Expected next-run interpretation:

- If the run succeeds and `send_export_runtime_slot_npu_items=0`, the CPU slot-map fix removed the known `.item()` stall path. Check whether `rollout_output_time_s` returns near the remote healthy ~120s.
- If the run raises `Mode1 export slot CPU mirror is missing or stale`, the next target is the exact map replacement point named by that error's `layer`, `ep_rank`, `map_kind`, and tensor ids.
- If the run still stalls while `send_export_runtime_slot_npu_items=0`, the NPU map lookup has been eliminated and the next suspect moves to real model-buffer alias/export synchronization or delayed runtime errors from the preceding dp=4 MoE dispatch.

Temporary escape hatch:

```bash
VLLM_ASCEND_MODE1_ALLOW_NPU_EXPORT_SLOT_FALLBACK=1 ./run_mode1_floor2_local.sh
```

Use this only to reproduce the old behavior. It re-enables the known slow `.item()` fallback and is not the recommended path.

## 2026-06-11 update: latest local run narrowed the stall to first batched direct-NPU export

Latest log:

- `/workspace/cann-recipes-train/llm_rl/qwen3_nm_model1_final/wjeagerqwen30b-a3b-with_draft_breakdown_20260610235449290755923_elastic.txt`
- Run completed successfully, but still slow: `rollout_output_time_s=450.676006`, `timing_s/gen=450.672619`.
- Final `4->2` shrink remains the dominant component: active rank total is about `352.86s`.

New evidence from the added instrumentation:

- Preload-entry `torch.npu.synchronize()` at final `active_ranks=[14, 15]` is small: about `1.4-2.2ms`. The 350s does not move into the pre-export sync.
- Source ranks are the slow side:
  - rank 12 final layer 0: `send_export_ms=350586.67`, `send_post_ms=51.11`, `send_wait_ms=0.04`, `wall_ms=350638.79`.
  - rank 13 final layer 0: `send_export_ms=350507.39`, `send_post_ms=58.43`, `send_wait_ms=0.04`, `wall_ms=350567.09`.
- Receiver ranks are not slow:
  - rank 14 final layer 0: `recv_post_ms=59.94`, `recv_wait_ms=0.04`, `wall_ms=67.76`.
  - rank 15 final layer 0: `recv_post_ms=58.70`, `recv_wait_ms=0.03`, `wall_ms=66.35`.
- After layer 0, source rank export is normal:
  - rank 12 layers 1-47 are about `3.1-3.3ms` per layer.
  - rank 13 layers 1-47 are about `3.1-3.4ms` per layer.
- The dry-run single-expert export immediately before the slow path is also normal:
  - rank 12 layer 0 single expert dry-run: `export_ms=0.41`.
  - rank 13 layer 0 single expert dry-run: `export_ms=0.66`.

Conclusion:

- The 350s is no longer best described as generic HCCL P2P, group rebuild, refresh, or recv-side wait.
- It is specifically the first final-`4->2` source-side batched direct-NPU export for layer 0.
- The current batched path calls `export_lossless_expert_npu_weights(chunk_ids)` with multiple experts, which falls through to `source_w13.index_select(0, export_index)` / `source_w2.index_select(0, export_index)`.
- The single-expert path uses narrow slot views plus zero-offset aliases and is fast. Therefore the most plausible root cause is a context-sensitive first `index_select`/materialized gather on real expert buffers after the prior dp=4 stage.

Fix applied after this run:

- Default `VLLM_ASCEND_MODE1_BATCH_DIRECT_NPU_IMPORT` changed from `1` to `0` in `vllm_ascend/worker/worker_v1.py`.
- `run_mode1_floor2_local.sh` and `internal/wj_train_grpo_qwen30b_a3b_16die_true_weight_eager.sh` now explicitly default `VLLM_ASCEND_MODE1_BATCH_DIRECT_NPU_IMPORT=0`.
- This forces mode=1 floor=2 direct-NPU shrink import back to the single-expert zero-offset alias path, avoiding the batched `index_select` export path that produced the 350s layer-0 spike.

Next validation target:

- Run with batch disabled and the same diagnostic timing enabled.
- Expected result if this diagnosis is correct:
  - final `4->2` source rank layer 0 `send_export_ms` should drop from about `350,500ms` to per-expert alias-scale milliseconds;
  - final `4->2` `preload_import_ms` should be seconds, not hundreds of seconds;
  - `rollout_output_time_s` should return near the known healthy `~120s` range, allowing for local noise.

If batch-disabled mode is still slow:

- Check whether the slow field moves into repeated single-expert `send_export_ms`, `send_post_ms`, or `send_wait_ms`.
- If single-expert `send_export_ms` becomes slow, inspect `_npu_zero_offset_alias_for_p2p()` on the first real layer-0 export.
- If `send_post_ms`/`send_wait_ms` becomes slow, revisit HCCL communicator state only after confirming the export path is no longer using batched `index_select`.

## 2026-06-11 update: single-expert alias path removed layer-0 index_select stall but exposed P2P post storm

Latest batch-disabled validation log:

- `/workspace/cann-recipes-train/llm_rl/qwen3_nm_model1_final/wjeagerqwen30b-a3b-with_draft_breakdown_20260611012852387684070_elastic.txt`
- The run used `mode1_batch_direct_npu_import=0`.
- End-to-end is still slow: `rollout_output_time_s=451.247493`.

What changed compared with the previous run:

- The old layer-0 source-side batched export spike disappeared. Final `4->2` layer 0 source ranks are now about `5.5ms` export plus `55-61ms` first post, not `350s` export.
- However, a new 350s spike appears later in the same final `4->2` preload:
  - rank 12 layer 9: `send_export_ms=5.88`, `send_post_ms=350204.95`, `send_wait_ms=0.29`, `wall_ms=350211.52`.
  - rank 13 layer 9: `send_export_ms=6.18`, `send_post_ms=350142.11`, `send_wait_ms=0.25`, `wall_ms=350148.98`.
  - rank 14 layer 9: `recv_post_ms=350188.42`, `recv_wait_ms=0.27`, `wall_ms=350194.61`.
  - rank 15 layer 8: `recv_post_ms=350178.05`, `recv_wait_ms=0.17`, `wall_ms=350184.40`.

Updated interpretation:

- Disabling batched export proves the previous `index_select`/materialized gather path was one way to trigger the 350s stall.
- But falling back to 32 experts * 2 tensors * 48 layers as many tiny P2P ops creates a second failure mode: the communicator/runtime stalls in `isend`/`irecv` post around layer 8/9.
- Therefore the desired path is neither `index_select` batching nor fully scalar expert-by-expert P2P. It should batch only when the source slots are contiguous, using continuous slice views plus zero-offset aliases. That reduces P2P post count without materializing `index_select` gather buffers.

Fix applied after this run:

- `export_lossless_expert_npu_weights()` now handles multi-expert requests as follows:
  - if mapped local slots are contiguous, return continuous source tensor views wrapped with zero-offset P2P aliases;
  - if slots are non-contiguous, refuse the batched `index_select` path by default with a clear RuntimeError;
  - the old `index_select` fallback is only allowed when `VLLM_ASCEND_MODE1_ALLOW_BATCH_INDEX_SELECT_EXPORT=1` is explicitly set.
- `VLLM_ASCEND_MODE1_BATCH_DIRECT_NPU_IMPORT` default is restored to `1`, so the shrink path can send chunks again.
- `VLLM_ASCEND_MODE1_ALLOW_BATCH_INDEX_SELECT_EXPORT` defaults to `0` in both scripts.

Next validation target:

- Log header should show `mode1_batch_direct_npu_import=1` and `mode1_allow_batch_index_select_export=0`.
- Final `4->2` layer rows should show `chunks=4` on source ranks, with no 350s `send_export_ms`, `send_post_ms`, `recv_post_ms`, or `recv_wait_ms`.
- If any chunk has non-contiguous slots, the run should fail fast with the explicit non-contiguous batched export error rather than silently taking the known-bad `index_select` path.

## 2026-06-11 follow-up patch before next run: preserve source-slot order and receive directly into target ranges

Additional code audit after the batch-disabled run found two risks in the first continuous-alias batching patch:

- The shrink payload planner records inactive experts in source local-slot order, but `_stream_lossless_layer_cpu_import_weights()` rebuilt `export_ids_per_source_rank` and sorted by expert id. That can destroy source-slot contiguity before chunking.
- Batch recv allocated temporary `(chunk_rows, ...)` tensors, received into them, then copied into target slots. This is unnecessary when target slots are contiguous and can add allocator/copy/post overhead.

Follow-up fixes applied:

- `export_ids_per_source_rank[source_rank]` is now sorted by `payload["remote_import_source_slot"][expert_id]` first, then expert id. This preserves source-slot order for batching.
- Added `get_lossless_expert_npu_slot_range_recv_buffers(first_slot, rows)` to return zero-offset aliases for a contiguous target slot range.
- Batch direct-NPU recv now computes target slots before posting recv. If they are contiguous, it posts `irecv` directly into the target slot range alias; otherwise it falls back to temporary receive plus row copy.

Expected next run behavior:

- Final `4->2` source chunks should remain `chunks=4`, but each chunk should now use a continuous source view alias rather than `index_select`.
- Final `4->2` target ranks should avoid most temporary receive/copy overhead when target slots are contiguous.
- If source slots are not contiguous for a chunk, the run should fail fast with the explicit non-contiguous batched export error because `VLLM_ASCEND_MODE1_ALLOW_BATCH_INDEX_SELECT_EXPORT=0`.

## 2026-06-11 instrumentation update: prove continuous-alias batch path in the next log

Added layer-level counters to the direct-NPU import breakdown log:

- `send_export_contiguous_chunks`: number of source chunks exported as contiguous slot-range views plus zero-offset aliases.
- `send_export_index_select_chunks`: number of chunks that did not report contiguous source slots. With `VLLM_ASCEND_MODE1_ALLOW_BATCH_INDEX_SELECT_EXPORT=0`, this should stay zero because non-contiguous chunks fail fast before index_select.
- `recv_direct_slot_chunks`: number of target chunks received directly into contiguous target slot-range aliases.
- `recv_temp_buffer_chunks`: number of target chunks received into temporary buffers and copied afterwards.

Next successful fast run should show, for final `active_ranks=[14, 15]`:

- source ranks 12/13: `chunks=4`, `send_export_contiguous_chunks=4`, `send_export_index_select_chunks=0` per layer.
- target ranks 14/15: `chunks=4`, `recv_direct_slot_chunks=4`, `recv_temp_buffer_chunks=0` per layer.
- no layer with any `send_export_ms`, `send_post_ms`, `recv_post_ms`, or `recv_wait_ms` near 350000ms.

## 2026-06-11 update: latest local run and current fix

Latest analyzed local log:

- `wjeagerqwen30b-a3b-with_draft_breakdown_20260611012852387684070_elastic.txt`
- Header shows `mode1_batch_direct_npu_import=0`.
- End-to-end remains slow: `rollout_output_time_s=451.247493`.

What this run verified:

- Disabling batch direct NPU import removes the previous layer-0 multi-expert `index_select` export stall, but it does not solve floor=2 runtime.
- The 350s delay moves into fine-grained direct-NPU P2P posts at the final `[14, 15]` shrink. Representative evidence:
  - rank 12 layer 9: `send_export_ms=5.88`, `send_post_ms=350204.95`, `wall_ms=350211.52`.
  - rank 13 layer 9: `send_export_ms=6.18`, `send_post_ms=350142.11`, `wall_ms=350148.98`.
  - rank 14 layer 9: `recv_post_ms=350188.42`, `wall_ms=350194.61`.
  - rank 15 layer 8: `recv_post_ms=350178.05`, `wall_ms=350184.40`.
- Therefore both extremes are bad on this host:
  - old batch with non-contiguous `index_select` can stall inside source export/materialization;
  - scalar per-expert P2P avoids `index_select` but creates enough P2P submit/post overhead to hit the same 350s-class runtime behavior.

Current code fix:

- Batch direct NPU import is restored as the default: `VLLM_ASCEND_MODE1_BATCH_DIRECT_NPU_IMPORT=1`.
- The dangerous multi-expert `index_select` export path is disabled by default: `VLLM_ASCEND_MODE1_ALLOW_BATCH_INDEX_SELECT_EXPORT=0`.
- `export_lossless_expert_npu_weights()` now only returns batched NPU export tensors when requested experts occupy a contiguous source slot range; it returns zero-offset P2P aliases over that range instead of materializing with `index_select`.
- The receiver now writes contiguous chunks directly into the target loaded slots when target slots are contiguous, avoiding an extra temporary NPU buffer and copy.
- The worker-side batch planner now splits each source/target transfer into chunks whose source slots and target slots are both contiguous. This avoids fixed-size batches that accidentally span slot gaps, and it avoids falling back to the known slow path.

Validation expectations for the next run:

- Header must show `mode1_batch_direct_npu_import=1` and `mode1_allow_batch_index_select_export=0`.
- In final `[14, 15]` shrink, source ranks 12/13 should show nonzero `chunks`, `send_export_contiguous_chunks`, and zero `send_export_index_select_chunks` when export timing is enabled.
- Target ranks 14/15 should show nonzero `recv_direct_slot_chunks` and ideally zero `recv_temp_buffer_chunks`.
- No layer should show `send_export_ms`, `send_post_ms`, `recv_post_ms`, or `recv_wait_ms` around `350000ms`.
- If the run fails with missing or non-contiguous slot metadata, the next fix should be in payload planning/slot metadata, not in re-enabling `index_select`.

## 2026-06-11 pre-run hardening: normalize and validate import slot metadata

Before the next run, the direct-NPU batch preload path was hardened further:

- `cpu_import_source_rank`, `cpu_import_target_rank`, and `remote_import_source_slot` are normalized at stream-import entry to `int -> int` maps. This avoids key-type drift after object/broadcast/path changes.
- Batch planning now validates that every expert in a direct-NPU batch has both source-slot and target-slot metadata.
- Batch planning also validates that source and target slots are non-negative before posting P2P.

This is deliberately fail-fast. If the next run fails with the new metadata error, the root cause is in shrink payload construction/slot metadata, and the fix should be in the planner. Do not re-enable `VLLM_ASCEND_MODE1_ALLOW_BATCH_INDEX_SELECT_EXPORT`; that would reintroduce the already verified 350s export materialization risk.

## 2026-06-11 code audit: fixed stale `local_slot` leakage in paired redundant planner

A code audit found a concrete planner bug in the mode=1 paired redundant shrink branch used by the final `4->2` transition:

```python
for free_slot, (expert_id, _) in zip(free_loaded_slots, inactive_runtime_pairs):
    remote_source_rank, source_slot = _resolve_mode4_npu_source(
        inactive_rank, int(expert_id), int(local_slot))
```

The loop discarded the inactive rank's actual `local_slot` as `_`, but then passed `int(local_slot)` from an outer loop. This can write incorrect `remote_import_source_slot` values into the shrink payload.

Why this matters for the 350s fix:

- The current fast path relies on `remote_import_source_slot` to order and chunk NPU exports by physically contiguous source slots.
- If that metadata is stale or wrong, chunking can either become non-contiguous and fail fast, or worse, post large P2P transfers from the wrong source view order.
- This bug is independent of the already verified two bad transport paths, but it can prevent the new continuous-alias batch path from being valid.

Fix applied:

```python
for free_slot, (expert_id, inactive_local_slot) in zip(
        free_loaded_slots, inactive_runtime_pairs):
    remote_source_rank, source_slot = _resolve_mode4_npu_source(
        inactive_rank, int(expert_id), int(inactive_local_slot))
```

Validation expectation:

- In the next final `[14, 15]` shrink, `send_export_contiguous_chunks` should become nonzero and `send_export_index_select_chunks` should remain zero.
- If chunks are split into too many small pieces or fail metadata validation, inspect the layer-0 payload source slots before changing transport behavior.

## 2026-06-11 instrumentation: final 4->2 chunk-plan log

Added a narrow layer-0 chunk-plan log for the final `[14, 15]` direct-NPU batch path:

- log label: `Elastic shrink direct NPU batch chunk plan`
- emitted only when `len(active_ranks) == 2`, `layer == 0`, and `current_rank == source_rank`
- includes source rank, target rank, total experts, number of chunks, configured `batch_experts`, and the first chunk ranges:
  - `(first_expert, last_expert)`
  - `(first_source_slot, last_source_slot)`
  - `(first_target_slot, last_target_slot)`

Expected healthy shape for floor=2 final shrink is roughly four 8-expert chunks per inactive->active edge. If the plan shows many 1-row chunks, the job may still behave like scalar P2P even with batch enabled; then the next fix should focus on assignment/source-slot ordering rather than transport primitives.

## 2026-06-11 diagnostic counter fix: single-row chunks are contiguous aliases, not index_select

The export debug record now clears stale state at the start of each NPU export and marks the single-expert path as:

- `expert_count=1`
- `local_slots_contiguous=True`

This does not change data movement semantics. It prevents misleading logs where a legitimate 1-row source-slot view/alias could be counted as `send_export_index_select_chunks` simply because the single-row debug record lacked the `local_slots_contiguous` field.

Interpretation after this patch:

- `send_export_index_select_chunks > 0` should mean an actual unexpected non-contiguous/index-select-like export record, not merely a single-row chunk.
- If chunk-plan shows many 1-row chunks but `send_export_index_select_chunks=0`, the remaining issue is poor chunk formation / scalar P2P overhead, not index_select materialization.

## 2026-06-11 script observability: log direct-NPU batch size

Both the local wrapper and the internal training script now default and print:

- `VLLM_ASCEND_MODE1_DIRECT_NPU_IMPORT_BATCH_EXPERTS=8`
- log field: `mode1_direct_npu_import_batch_experts=8` / `direct_npu_import_batch_experts=8`

This makes the next run self-contained: the header will show whether the run used batch direct-NPU import, whether index-select export was forbidden, and the maximum experts per direct-NPU P2P chunk.

## 2026-06-11 helper: log summarizer for next validation

Added local helper:

```bash
python3 analyze_mode1_floor2_log.py <elastic_log>
```

It extracts:

- `rollout_output_time_s`
- key mode=1 header knobs
- final `[14, 15]` shrink breakdown row count
- max `send_export_ms`, `send_post_ms`, `recv_post_ms`, `recv_wait_ms`, `wall_ms`
- layer-0 `Elastic shrink direct NPU batch chunk plan` rows

Running it on `wjeagerqwen30b-a3b-with_draft_breakdown_20260611012852387684070_elastic.txt` reproduces the current known failure:

- `rollout_output_time_s=451.247493`
- `mode1_batch_direct_npu_import=0`
- `max_send_post_ms=350204.95 rank=12 layer=9`
- `max_recv_post_ms=350188.42 rank=14 layer=9`
- `chunk_plan_rows=0`

Use this helper immediately after the next run to decide whether the continuous batch path solved the 350s stall or whether the remaining problem is chunk formation / metadata.

## 2026-06-11 status after latest log review: root cause split into two verified bad paths

Rechecked the latest two local logs with `analyze_mode1_floor2_log.py`:

```bash
python3 analyze_mode1_floor2_log.py wjeagerqwen30b-a3b-with_draft_breakdown_20260610235449290755923_elastic.txt
python3 analyze_mode1_floor2_log.py wjeagerqwen30b-a3b-with_draft_breakdown_20260611012852387684070_elastic.txt
```

Verified facts:

- With the old batched direct-NPU import path, final `4->2` uses `chunks=4`, but rank 12 layer 0 stalls in `send_export_ms=350586.67ms`. This is the multi-expert materialized gather / `index_select`-class failure.
- With `mode1_batch_direct_npu_import=0`, the layer-0 export stall disappears, but final `4->2` still stalls because scalar per-expert P2P creates a submit/post storm:
  - rank 12 layer 9: `send_post_ms=350204.95ms`.
  - rank 13 layer 9: `send_post_ms=350142.11ms`.
  - rank 14 layer 9: `recv_post_ms=350188.42ms`.
  - rank 15 layer 8: `recv_post_ms=350178.05ms`.
- Therefore the remaining 350s source is not normal decode compute and not KV capacity. It is the final `4->2` direct-NPU preload/import transport path choosing one of two bad extremes:
  - non-contiguous/materialized batched export, or
  - scalar per-expert P2P submit/post.

Current intended fix:

- Keep batch direct-NPU import enabled.
- Forbid `index_select` fallback.
- Batch only physically contiguous source slots and physically contiguous target slots.
- Send source contiguous slot-range views through zero-offset aliases.
- Receive directly into target contiguous slot-range aliases.

Additional validation helper update:

- `analyze_mode1_floor2_log.py` now reports:
  - `counter_sums` for chunk/direct/index/temp counters.
  - `source_path_summary` and `target_path_summary`.
  - slow rows above `--slow-ms` threshold, including path counters.

The next run is expected to distinguish the outcome without manual log scanning:

- Healthy path:
  - header: `mode1_batch_direct_npu_import=1`, `mode1_allow_batch_index_select_export=0`, `mode1_direct_npu_import_batch_experts=8`.
  - final `[14, 15]` source rows: `send_export_contiguous_chunks > 0`, `send_export_index_select_chunks=0`.
  - final `[14, 15]` target rows: `recv_direct_slot_chunks > 0`, ideally `recv_temp_buffer_chunks=0`.
  - `slow_rows_threshold_ms=1000.00 count=0` for direct-NPU timing fields.
  - rollout should move back toward the known healthy ~120s baseline.
- If it fails fast with metadata/contiguity errors, fix payload slot metadata or assignment ordering. Do not re-enable index-select export.
- If it still has 350s post spikes despite contiguous chunks, the next root cause is HCCL/runtime post behavior for chunked zero-offset aliases in the full training context, not export materialization.

Suggested next run command:

```bash
cd /workspace/cann-recipes-train/llm_rl/qwen3_nm_model1_final

MODE1_DIAG_PRELOAD_ENTRY_SYNC=1 \
MODE1_DIAG_DIRECT_NPU_EXPORT_DRY_RUN=1 \
MODE1_DIAG_EXPORT_INTERNAL_TIMING=1 \
MODE1_DIAG_SLOW_EXPORT_LOG_MS=100 \
VLLM_ASCEND_MODE1_DIAG_PRELOAD_LAYER_TIMING=1 \
VLLM_ASCEND_MODE1_DIAG_SLOW_PRELOAD_LAYER_LOG_MS=100 \
VLLM_ASCEND_MODE1_BATCH_DIRECT_NPU_IMPORT=1 \
VLLM_ASCEND_MODE1_ALLOW_BATCH_INDEX_SELECT_EXPORT=0 \
VLLM_ASCEND_MODE1_DIRECT_NPU_IMPORT_BATCH_EXPERTS=8 \
TRAINER_TOTAL_EPOCHS=1 \
bash run_mode1_floor2_local.sh
```

After the run:

```bash
cd /workspace/cann-recipes-train/llm_rl/qwen3_nm_model1_final
latest=$(ls -t wjeagerqwen30b-a3b-with_draft_breakdown_*_elastic.txt | head -1)
python3 analyze_mode1_floor2_log.py "$latest"
```

## 2026-06-11 latest validation: transfer path fixed, remaining 350s moved to group lifecycle ordering

Validation log:

- `/workspace/cann-recipes-train/llm_rl/qwen3_nm_model1_final/wjeagerqwen30b-a3b-with_draft_breakdown_20260611111804038713003_elastic.txt`
- Run completed with exit code 0.
- Header showed `mode1_batch_direct_npu_import=1`, `mode1_allow_batch_index_select_export=0`, `mode1_direct_npu_import_batch_experts=8`.

What is now fixed:

- The final `[14, 15]` direct-NPU transport path no longer has 350s P2P/export spikes.
- `analyze_mode1_floor2_log.py` reported:
  - `max_send_export_ms=1.64`
  - `max_send_post_ms=67.06`
  - `max_recv_post_ms=64.38`
  - `slow_rows_threshold_ms=1000 count=0`
  - `send_export_contiguous_chunks=384`
  - `send_export_index_select_chunks=0`
  - `recv_direct_slot_chunks=384`
  - `recv_temp_buffer_chunks=0`
- This confirms the desired continuous source-slot alias plus direct target-slot receive path is active and removes both prior bad extremes: non-contiguous `index_select` batching and scalar P2P post storms.

Remaining abnormal cost:

- The same validation still had a 350s-class final shrink stall, but it moved out of the import/export/P2P fields:
  - inactive ranks 12/13 reported `drop_old_floor_group_cache_ms ~= 350s`.
  - active ranks 14/15 reported `refresh_ms ~= 350s`.
- The timestamps overlap. This points to a communication-group lifecycle ordering problem: inactive ranks begin destroying stale `[12, 13, 14, 15]` floor groups while active ranks are still refreshing the new `[14, 15]` MoE/MC2 runtime state.

Current code fix:

- Added a default-on CPU barrier before mode=1 old-floor group cache drop:
  - `VLLM_ASCEND_MODE1_PARITY_BARRIER_BEFORE_OLD_FLOOR_DROP=1`
- Added a new breakdown field:
  - `old_floor_group_drop_barrier_ms`
- Intent: all ranks finish the shrink refresh phase before any rank destroys old floor DP/EP/MC2 groups. This keeps stale groups from surviving into subsequent compute, but avoids active refresh and inactive destroy crossing HCCL lifecycle operations.

Next validation criteria:

- Header should show `mode1_barrier_before_old_floor_drop=1`.
- Final `[14, 15]` shrink should still have no slow transport rows in `analyze_mode1_floor2_log.py`.
- In `Elastic parallel shrink phase breakdown`:
  - `refresh_ms` on ranks 14/15 should fall back to normal sub-second/low-second scale.
  - `drop_old_floor_group_cache_ms` on ranks 12/13 should fall back to normal sub-second/low-second scale.
  - if any wait remains, it should be visible as `old_floor_group_drop_barrier_ms`, which means the cost is synchronization/waiting for refresh rather than old-group destroy itself.
- If `old_floor_group_drop_barrier_ms` becomes 350s while active `refresh_ms` remains 350s, the root is inside active `module.refresh_elastic_groups()` / `setup_moe_comm_method` and needs layer-level refresh timing.
- If the barrier eliminates the 350s, keep this ordering as the production mode=1 floor=2 lifecycle rule.

## 2026-06-11 follow-up: barrier group must be previous-active, not full-world

Failed validation log:

- `/workspace/cann-recipes-train/llm_rl/qwen3_nm_model1_final/wjeagerqwen30b-a3b-with_draft_breakdown_20260611115150580835386_elastic.txt`

Observed behavior:

- The first shrink `16 -> 8` completed successfully.
- `drop_old_floor_group_cache_ms` stayed at sub-ms scale, so the new pre-drop barrier did prevent the previous 350s old-group destroy stall at this stage.
- The run then failed during the next shrink `8 -> 4` at `rebuild_elastic_ep_group`, line corresponding to `old_floor_group_drop_barrier`.
- The stack ended in Gloo send/recv timeout or peer closed errors, all at the same barrier site.

Root cause:

- The barrier was incorrectly issued on `world_group.cpu_group`.
- After `16 -> 8`, ranks `0..7` have detached and do not participate in the next shrink RPC.
- During `8 -> 4`, only previous active/source ranks `8..15` enter `rebuild_elastic_ep_group`, so a full-world barrier waits forever for ranks that are no longer there.

Fix:

- Use the previous-active/source CPU group returned by `_get_shrink_source_group_state(world_group)` for the mode=1 old-floor drop barrier.
- This makes the barrier group match the ranks that are actually participating in the current shrink:
  - `16 -> 8`: source group is `0..15`
  - `8 -> 4`: source group is `8..15`
  - `4 -> 2`: source group is `12..15`
  - `2 -> 1`: source group is `14..15`

Validation target for the next user-run:

- The same command should now pass the `8 -> 4` shrink without Gloo timeout.
- `old_floor_group_drop_barrier_ms` may show real waiting time, but no rank should block on detached ranks.
- Keep checking that direct-NPU import remains on the fixed fast path: contiguous source-slot chunks and direct target-slot receives.

## 2026-06-11 latest log: direct import remains fixed, 350s is final stage=2 refresh/lifecycle

Validation log:

- `/workspace/cann-recipes-train/llm_rl/qwen3_nm_model1_final/wjeagerqwen30b-a3b-with_draft_breakdown_20260611130624896867555_elastic.txt`

Analyzer update:

- `analyze_mode1_floor2_log.py` previously only summarized final `[14, 15]` stream-import rows. That proved direct-NPU import was fast, but missed the slow shrink phase fields.
- The analyzer now also parses `Elastic parallel shrink phase breakdown` and reports per-stage max/mean timings.

What the latest log shows:

- Final `4 -> 2` direct-NPU expert import is still on the fast path:
  - `max_send_export_ms=1.18`
  - `max_send_post_ms=59.17`
  - `max_recv_post_ms=57.99`
  - `send_export_contiguous_chunks=384`
  - `send_export_index_select_chunks=0`
  - `recv_direct_slot_chunks=384`
  - `recv_temp_buffer_chunks=0`
  - no stream-import rows above 1000ms
- The 350s stall is not expert transfer. It is in final stage=2 shrink lifecycle:
  - active ranks 14/15: `refresh_ms ~= 350834ms`
  - inactive ranks 12/13: `old_floor_group_drop_barrier_ms ~= 350839ms`
- Therefore inactive ranks are mostly waiting for active refresh to finish. The active-side stall is inside per-layer `module.refresh_elastic_groups()` / `setup_moe_comm_method()` after final `4 -> 2` group rebuild.

Current code changes after this log:

- Added pre-refresh source-group barrier:
  - `VLLM_ASCEND_MODE1_PARITY_BARRIER_BEFORE_REFRESH=1`
  - aligns only previous active/source ranks after inactive detach/non-member sequence advancement and active group rebuild, before active ranks enter per-layer refresh.
- Moved old-floor cached group drop before active refresh by default:
  - `VLLM_ASCEND_MODE1_PARITY_DROP_OLD_FLOOR_BEFORE_REFRESH=1`
  - intended to release stale floor `[12,13,14,15]` DP/EP/MC2 cached groups before active ranks initialize/refresh the new `[14,15]` dispatcher/runtime state.
- Kept the old post-refresh drop path as fallback only when pre-refresh drop is disabled.
- Added optional slow-layer refresh diagnostics:
  - `VLLM_ASCEND_MODE1_DIAG_REFRESH_LAYER_TIMING=1`
  - `VLLM_ASCEND_MODE1_DIAG_SLOW_REFRESH_LAYER_LOG_MS=<threshold>`
  - logs `Elastic refresh module groups timing` for layer 0 and any layer whose refresh exceeds the threshold.

Next validation target:

- `phase_stage stage=2 max_refresh_ms` should drop from ~350s to normal sub-second/low-second scale.
- `phase_stage stage=2 max_old_floor_group_drop_barrier_ms` should no longer carry the 350s wait.
- If 350s remains, inspect `Elastic refresh module groups timing` to identify the exact layer/reason. Then optimize or narrow `setup_moe_comm_method()` for mode=1 floor=2 to avoid unnecessary comm-method/workspace reinitialization.

## 2026-06-11 follow-up: latest 350s is old-floor drop vs active refresh concurrency

Failed/slow validation log:

- `/workspace/cann-recipes-train/llm_rl/qwen3_nm_model1_final/wjeagerqwen30b-a3b-with_draft_breakdown_20260611134632646172972_elastic.txt`
- Launcher wrapper: `local_mode1_floor2_runs/20260611T134632+0800/launcher.log`

Analyzer note:

- Running `python3 analyze_mode1_floor2_log.py "$latest"` returned zero rows because `$latest` pointed at the local `launcher.log`.
- The real Ray worker shrink rows are in the `logfile=...wjeager..._elastic.txt` file referenced by the launcher log.
- `analyze_mode1_floor2_log.py` now follows `logfile=` automatically when a launcher log is passed.

Observed from the real worker log:

- `rollout_output_time_s=447.809048`, still inflated.
- Final `4 -> 2` direct-NPU import is fast, so the expert-transfer path is not the 350s culprit:
  - `max_send_export_ms=0.90`
  - `max_send_post_ms=58.07`
  - `max_recv_post_ms=61.52`
  - no stream-import row above 1000ms.
- The 350s appears in final stage=2 shrink lifecycle:
  - inactive ranks 12/13: `drop_old_floor_group_cache_ms ~= 351130ms`
  - active ranks 14/15: `refresh_ms ~= 351140ms`
  - `pre_refresh_source_barrier_ms` is only ~18ms, so the pre-refresh barrier itself is not the stall.

Root cause hypothesis refined:

- Moving old-floor cache drop before active refresh was correct, but there was no synchronization after that drop.
- Active ranks could finish old-floor drop quickly and enter per-layer `refresh_elastic_groups()` while inactive ranks were still destroying stale floor `[12,13,14,15]` groups.
- That creates exactly the observed split: inactive ranks block inside stale group destroy; active ranks block inside refresh/setup on the new `[14,15]` state.
- A post-drop barrier is needed before refresh starts.

Important implementation constraint:

- The previous active/source DP CPU group may itself be one of the stale groups being destroyed when shrinking from `[12,13,14,15]` to `[14,15]`.
- Therefore the post-drop barrier must not use `source_cpu_group_for_barrier` after the drop.
- A full-world collective barrier is also invalid for later shrink stages because detached ranks do not participate.

Fix implemented:

- Added `_mode1_source_cpu_p2p_barrier(source_ranks, world_group, tag)`.
- It uses the stable full-world CPU group for a tiny root fan-in/fan-out P2P barrier, but only among the previous active/source ranks.
- Replaced mode=1 pre-refresh and old-floor-drop barriers with this P2P barrier.
- Added a new post-drop barrier immediately after `drop_old_floor_group_cache_before_refresh`, before active ranks enter refresh.
- Added `post_old_floor_group_drop_barrier_ms` to the shrink phase breakdown and analyzer.
- Mode=4 remote-cache barriers are intentionally unchanged.

Validation target for next run:

- `stage=2 max_refresh_ms` should no longer be ~351s.
- `stage=2 max_drop_old_floor_group_cache_ms` should no longer be ~351s due to concurrent active refresh.
- If a residual stall remains, it should now be isolated to one of:
  - `drop_old_floor_group_cache_ms` if stale group destruction itself is inherently slow;
  - `post_old_floor_group_drop_barrier_ms` if ranks finish drop at very different times;
  - `refresh_ms` with slow-layer logs if active refresh is independently slow.

## 2026-06-11 targeted fix after 14:24 log: isolate old-floor drop internals

Latest analyzed log:

- `/workspace/cann-recipes-train/llm_rl/qwen3_nm_model1_final/wjeagerqwen30b-a3b-with_draft_breakdown_20260611142404077372810_elastic.txt`

Observed facts:

- Final `4 -> 2` direct-NPU expert import is already fast:
  - `max_send_export_ms=0.87`
  - `max_send_post_ms=55.88`
  - `max_recv_post_ms=55.31`
  - no stream-import rows above 1000ms.
- The 350s stall has moved out of active refresh:
  - inactive ranks 12/13: `drop_old_floor_group_cache_ms ~= 350.7s`
  - active ranks 14/15: `post_old_floor_group_drop_barrier_ms ~= 350.0-350.4s`
  - active `refresh_ms` is now only about 396-409ms.
- Timeline clue:
  - ranks 12/13 print old-floor drop rows and summary around `14:29:14`;
  - their `after_old_floor_group_cache_drop_before_refresh` memory diagnostic appears only around `14:35:04`.
  - This points at old-floor-drop cleanup/diagnostic/GC/logging tail, not expert transfer.

Changes made now:

- `VLLM_ASCEND_CUSTOM_MODE1_KV_DIAG` now defaults to `0` in the internal script and local wrapper.
- `VLLM_ASCEND_MODE1_PARITY_OLD_FLOOR_DROP_VERBOSE` now defaults to `0`, suppressing per-group old-floor drop logs unless explicitly enabled.
- Added `VLLM_ASCEND_MODE1_PARITY_GC_AFTER_MC2_GROUP_DESTROY`, default `0`.
  - `_cleanup_after_elastic_mc2_group_destroy()` no longer unconditionally calls `gc.collect()` in the shrink critical path.
- Added detailed old-floor drop subfields to `Elastic parallel shrink phase breakdown` and to `analyze_mode1_floor2_log.py`:
  - `drop_old_floor_group_destroy_ms`
  - `drop_old_floor_group_cleanup_ms`
  - `drop_old_floor_group_cleanup_gc_ms`
  - `drop_old_floor_group_cleanup_empty_cache_ms`
  - `drop_old_floor_group_cleanup_sync_ms`
  - `drop_old_floor_group_cleanup_memory_log_ms`
  - `drop_old_floor_group_memory_log_ms`
  - `drop_old_floor_group_row_log_ms`
  - `drop_old_floor_group_summary_log_ms`
  - `drop_old_floor_group_signature_sweep_ms`
  - `drop_old_floor_group_dropped_groups`
  - `drop_old_floor_group_dropped_signatures`
- Analyzer header now prints the relevant switches:
  - `custom_mode1_kv_diag`
  - `old_floor_drop_verbose`
  - `mode1_gc_after_mc2_destroy`
  - `mode1_sync_after_mc2_destroy`

Next validation target:

- Header should show `custom_mode1_kv_diag=0 old_floor_drop_verbose=0 mode1_gc_after_mc2_destroy=0 mode1_sync_after_mc2_destroy=0`.
- `stage=2 max_drop_old_floor_group_cache_ms` should drop from ~350s.
- If it does not, the new subfields should tell whether the culprit is actual `group.destroy()`, cleanup GC, memory logging, row/summary logging, or signature sweep.
- `stage=2 post_old_floor_group_drop_barrier_ms` should become small if inactive old-floor drop finishes quickly.

## Codex workflow guardrail for future mode=1 floor=2 work

Before making any further code change for the mode=1 floor=2 350s issue:

- Re-read this file first, at least the latest sections, to avoid repeating already-tested directions.
- Record every meaningful attempt here, including failed attempts and negative results.
- For each attempt, include:
  - the log file or command/result that motivated the change;
  - the concrete hypothesis being tested;
  - files/functions changed;
  - expected validation signal;
  - actual validation result when available;
  - whether the direction should be continued or abandoned.
- Do not rely on memory alone for this debugging thread. If a change affects the diagnosis path, append it here before handing control back to the user.
- Prefer new changes that disambiguate untested root causes over repeating broad cleanup/diagnostic toggles that have already been tried.

Current priority after the latest 14:24 log:

1. Validate whether disabling mode1 KV memory diagnostics, old-floor verbose logs, and MC2-destroy GC removes the stage=2 350s old-floor-drop tail.
2. If 350s remains, use the newly added old-floor-drop subfields to identify whether the remaining stall is actual `group.destroy()`, cleanup, logging, or a barrier wait.
3. Only after that decide whether to defer/delay stale group destroy or change communicator lifecycle semantics.

## 2026-06-11 latest pasted analysis: hidden tail before post-drop barrier

Input evidence from the pasted latest analysis and worker-log excerpt:

- Latest final `4 -> 2` direct-NPU expert transfer is fast again:
  - source/import rows show `send_export_ms` around ms-level to tens of ms;
  - no stream-import rows explain the 350s anymore.
- Final stage=2 shrink shows the slow split as:
  - inactive ranks 12/13: `hidden_tail_ms ~= 351288ms`, while `post_old_floor_group_drop_barrier_ms` is only `1-3ms`;
  - active ranks 14/15: `post_old_floor_group_drop_barrier_ms ~= 351326ms`, while `hidden_tail_ms` is only about `10ms`.
- Old-floor drop internals are small in the already-recorded fields:
  - `drop_old_floor_group_cache_ms ~= 46.9ms` on inactive ranks;
  - `drop_old_floor_group_destroy_ms`, cleanup, signature sweep, and logging subfields are tiny.
- This rules out the previously suspected direct NPU export path and the old-floor group destroy body as the current 350s culprit.

Interpretation:

- The active ranks are blocked in the post-old-floor-drop source-rank P2P barrier because inactive/source ranks enter that barrier roughly 351s later.
- The inactive/source ranks are not spending that time in any existing named phase; it is counted as `hidden_tail_ms`.
- Therefore the problem is now in one of the unaccounted regions before the post-drop barrier, or in logging/stdout/backpressure around those regions.
- The two highest-risk unaccounted regions are:
  - detached-rank `_advance_group_creation_sequence_for_non_member()` for DP/EP/MC2 placeholder groups;
  - old-floor drop wrapper work around `_drop_old_floor_cached_elastic_parallel_groups_after_mode1_shrink()`, especially before/after memory diagnostic calls and outer call wall time that were not separated from the function's internal returned `total_ms`.

Changes made for the next run:

- `_advance_group_creation_sequence_for_non_member()` now returns detailed timing stats instead of being a blind/unaccounted call:
  - `non_member_sequence_ms`
  - `non_member_sequence_dp_ms`
  - `non_member_sequence_ep_ms`
  - `non_member_sequence_mc2_ms`
  - `non_member_sequence_device_new_group_ms`
  - `non_member_sequence_cpu_new_group_ms`
  - `non_member_sequence_destroy_ms`
  - `non_member_sequence_signature_ms`
  - `non_member_sequence_log_ms`
  - `non_member_sequence_groups`
- The non-member sequence reuse/advanced logs are no longer unconditional.
  - Default: quiet.
  - Enable all logs with `VLLM_ASCEND_ELASTIC_NON_MEMBER_SEQUENCE_VERBOSE=1`.
  - Slow logs are still emitted automatically if the sequence itself exceeds `VLLM_ASCEND_ELASTIC_NON_MEMBER_SEQUENCE_SLOW_LOG_MS`, default `1000`.
- Old-floor drop wrapper timing is now explicit:
  - `drop_old_floor_group_call_wall_ms`
  - `old_floor_pre_memory_log_ms`
  - `old_floor_post_memory_log_ms`
- `hidden_tail_ms` now subtracts `non_member_sequence_ms`, `drop_old_floor_group_call_wall_ms`, and old-floor pre/post memory-log time.
- The post-old-floor-drop P2P barrier in the before-refresh path was moved to immediately after stale group drop completes and before `after_old_floor_group_cache_drop_before_refresh` memory diagnostics.
  - This preserves the real synchronization requirement: active ranks wait for all source ranks to finish stale group destruction.
  - It avoids letting optional memory diagnostics or log backpressure hold active ranks before refresh.
- `analyze_mode1_floor2_log.py` now reports the new fields.

Expected validation signal:

- If the change fixes the 350s, stage=2 should show:
  - `post_old_floor_group_drop_barrier_ms` no longer around 351s on active ranks;
  - `hidden_tail_ms` no longer around 351s on inactive ranks;
  - rollout time should move back toward the healthy ~120s class.
- If 350s remains, it should no longer be hidden. It should appear in one of:
  - `non_member_sequence_*` fields: placeholder group ordering/new_group/logging is the culprit;
  - `drop_old_floor_group_call_wall_ms` but not internal `drop_old_floor_group_cache_ms`: outer call/return or Python/runtime scheduling is suspect;
  - `old_floor_pre_memory_log_ms` / `old_floor_post_memory_log_ms`: memory diagnostic/logging is still active or blocking;
  - `post_old_floor_group_drop_barrier_ms` with all inactive pre-barrier fields small: then the barrier implementation itself or source-rank participation is suspect.

Validation command to give the user:

```bash
cd /workspace/cann-recipes-train/llm_rl/qwen3_nm_model1_final
./run_mode1_floor2_local.sh
```

Analyzer command after the run:

```bash
cd /workspace/cann-recipes-train/llm_rl/qwen3_nm_model1_final
latest="$(ls -td local_mode1_floor2_runs/* | head -1)/launcher.log"
python3 analyze_mode1_floor2_log.py "$latest"
```

Do not re-enable verbose sequence logs unless this run remains inconclusive; the goal is to avoid making Ray/stdout backpressure part of the critical path again.

## 2026-06-11 16:15 aborted run: call-wall/internal-total split confirms wrapper-finalizer path

Latest pasted evidence from the aborted run:

- Inactive/source ranks 12/13:
  - `drop_old_floor_group_call_wall_ms ~= 349.89s`
  - but inside the same function, `drop_old_floor_group_cache_ms ~= 49ms`
  - `drop_old_floor_group_destroy_ms < 0.3ms`
  - cleanup, memory-log, row-log, summary-log, signature-sweep are all tiny
  - `hidden_tail_ms ~= 10ms`
- Active ranks 14/15:
  - `drop_old_floor_group_call_wall_ms ~= 1.2ms`
  - `post_old_floor_group_drop_barrier_ms ~= 349.89s`

Conclusion:

- Active ranks are not the root cause; they are simply waiting in the post-drop source-rank barrier.
- Non-member sequence is not the root cause; it is sub-ms.
- The explicit `group.destroy()` body is not the root cause; `drop_old_floor_group_destroy_ms` is sub-ms.
- The old-floor-drop function's internal accounting is not slow, but the caller observes ~350s waiting for the function call to return.
- This strongly points to Python frame/local-reference teardown after the function's `return` value is prepared. In practice, the stale `GroupCoordinator` wrapper was popped from cache and explicitly destroyed, but its `device_communicator` still held `cpu_group`/`device_group` references. When the wrapper/local variable becomes unreachable at function return, low-level ProcessGroup cleanup/finalization can occur outside the explicit `group.destroy()` timer.

Code inspection supporting this:

- `vllm.distributed.parallel_state.GroupCoordinator.destroy()` calls `self.device_communicator.destroy()` and destroys/deletes `self.device_group` / `self.cpu_group`.
- On Ascend, `NPUCommunicator.destroy()` is a no-op.
- `DeviceCommunicatorBase` stores `self.cpu_group` and `self.device_group`.
- Therefore the communicator can retain stale ProcessGroup references even after `GroupCoordinator.destroy()` deletes the wrapper's direct attributes.

Fix implemented:

- Added `_detach_destroyed_group_references(group)`.
  - It clears `group.device_communicator.device_group` and `.cpu_group` when present.
  - It then clears `group.device_communicator` and `group.mq_broadcaster` on the wrapper.
- Called this helper immediately after every explicit `GroupCoordinator.destroy()` in worker elastic paths:
  - `_destroy_group_if_present()`
  - `_drop_old_floor_cached_elastic_parallel_groups_after_mode1_shrink()`
  - `_stash_group_if_present()` destroy branch
  - `_drop_stale_cached_elastic_parallel_groups()`
- Added `drop_old_floor_group_release_ref_ms` in the old-floor-drop body by explicitly setting the local `group = None` after destroy/reference-detach.
  - This should make any remaining finalizer/ref-release delay visible inside the function rather than as `call_wall - internal_total` mystery time.
- Updated `analyze_mode1_floor2_log.py` to report `drop_old_floor_group_release_ref_ms`.

Expected next validation signal:

- Best case:
  - `stage=2 drop_old_floor_group_call_wall_ms` drops from ~350s to ms-level on ranks 12/13.
  - `post_old_floor_group_drop_barrier_ms` also drops on ranks 14/15.
- If still slow:
  - If `drop_old_floor_group_release_ref_ms ~= 350s`, finalizer/ref-release is confirmed directly and we should stop destroying stale wrappers in shrink, instead moving them to a long-lived delayed-release quarantine.
  - If `drop_old_floor_group_call_wall_ms ~= 350s` but both `drop_old_floor_group_cache_ms` and `release_ref_ms` are tiny, then the delay is likely at function return/return-value handoff or a Python/runtime scheduling artifact; the next mitigation should also quarantine destroyed wrappers until after restore/process end.

Validation command remains:

```bash
cd /workspace/cann-recipes-train/llm_rl/qwen3_nm_model1_final
./run_mode1_floor2_local.sh
```

Then analyze:

```bash
latest="$(ls -td local_mode1_floor2_runs/* | head -1)/launcher.log"
python3 analyze_mode1_floor2_log.py "$latest"
```

## 2026-06-11 16:36 full log: 350s is inside floor4 ProcessGroup destroy, not transfer/barrier/ref-release

Analyzed log:

- `/workspace/cann-recipes-train/llm_rl/qwen3_nm_model1_final/wjeagerqwen30b-a3b-with_draft_breakdown_20260611163659764437651_elastic.txt`

Analyzer summary:

```text
rollout_output_time_s=446.883083
header mode=1 floor=2 mode1_batch_direct_npu_import=1 mode1_allow_batch_index_select_export=0 mode1_direct_npu_import_batch_experts=8 mode1_native_kv_cap=377344 mode1_gc_after_mc2_destroy=0 mode1_sync_after_mc2_destroy=0
final_4to2_breakdown_rows=4
max_total_ms=57.83
slow_rows_threshold_ms=1000.00 count=0
stage=2 max_drop_old_floor_group_call_wall_ms=351098.41
stage=2 max_drop_old_floor_group_cache_ms=351098.40
stage=2 max_drop_old_floor_group_destroy_ms=351097.04
stage=2 max_drop_old_floor_group_release_ref_ms=0.01
stage=2 max_post_old_floor_group_drop_barrier_ms=41.04
stage=2 max_hidden_tail_ms=9.49
```

What this disproves:

- The final direct-NPU expert import is not the 350s source. The largest 4->2 direct-NPU layer-0 import row is about 58ms, with no rows above 1s.
- The active-rank post-drop barrier is not the 350s source in this run; it is only about 41ms max.
- Python/local reference release is not the 350s source in this run; `drop_old_floor_group_release_ref_ms` is about 0.01ms.
- GC / empty-cache / sync are not the 350s source; they are disabled or near zero.

Updated conclusion:

- The 350s delay is now directly inside explicit stale floor4 `GroupCoordinator.destroy()` / underlying `torch.distributed.destroy_process_group` during 4->2 shrink.
- This is a lower-level ProcessGroup/HCCL destroy pathology for the old 4-rank floor group, not an expert-transfer pathology.
- The previous wrapper-reference-detach fix was useful instrumentation, but it does not solve this latest case because the stall moved from mysterious function-return wall time into explicit `destroy_ms`.

Fix implemented after this log:

- Added targeted mode1/floor2 floor4 cache retention:
  - new helper `_should_keep_mode1_floor4_group_cache(...)` in `worker_v1.py`;
  - default env: `VLLM_ASCEND_MODE1_PARITY_KEEP_FLOOR4_GROUP_CACHE=1`;
  - applies only when a mode1 lightweight parity module is configured with `max_floor=2`;
  - applies only to cached `dp` / `ep` / `mc2` groups whose rank tuple length is exactly 4;
  - does not keep floor8 stale groups, because keeping the 8-rank old floor group was previously shown to destabilize the following dp=4 stage.
- `_drop_old_floor_cached_elastic_parallel_groups_after_mode1_shrink(...)` now skips destroying floor4 stale groups under this narrow condition.
- `_drop_stale_cached_elastic_parallel_groups(...)` uses the same helper, so the retained floor4 cache can persist across restore and be reused on the next step rather than destroyed later in another hot path.
- Added observability:
  - `run_mode1_floor2_local.sh` exports and prints `mode1_keep_floor4_group_cache`;
  - `internal/wj_train_grpo_qwen30b_a3b_16die_true_weight_eager.sh` exports and prints it;
  - `analyze_mode1_floor2_log.py` extracts it from headers.

Expected validation signal:

- Header should include `mode1_keep_floor4_group_cache=1`.
- At stage=2, old floor group destruction should be skipped:
  - `drop_old_floor_group_dropped_groups=0` or at least no floor4 group destroy;
  - `drop_old_floor_group_destroy_ms` should be near 0 rather than ~351000ms;
  - `drop_old_floor_group_call_wall_ms` should be near 0 rather than ~351000ms;
  - `post_old_floor_group_drop_barrier_ms` should remain small.
- If 350s still appears after this fix, the next check should be whether it moved to a later stale-cache cleanup/restore path. Do not go back to expert-transfer tuning unless the direct import breakdown itself becomes slow again.

Static checks after the fix:

```bash
python3 -m py_compile vllm_ascend/worker/worker_v1.py analyze_mode1_floor2_log.py
bash -n run_mode1_floor2_local.sh internal/wj_train_grpo_qwen30b_a3b_16die_true_weight_eager.sh
git diff --check -- vllm_ascend/worker/worker_v1.py analyze_mode1_floor2_log.py run_mode1_floor2_local.sh internal/wj_train_grpo_qwen30b_a3b_16die_true_weight_eager.sh MODE1_FLOOR2_350S_DIAGNOSIS.md
```

All passed locally.

Next validation command for the user:

```bash
cd /workspace/cann-recipes-train/llm_rl/qwen3_nm_model1_final
./run_mode1_floor2_local.sh
```

Analyzer command after the run:

```bash
cd /workspace/cann-recipes-train/llm_rl/qwen3_nm_model1_final
latest="$(ls -td local_mode1_floor2_runs/* | head -1)/launcher.log"
python3 analyze_mode1_floor2_log.py "$latest"
```

## 2026-06-11 low-floor communicator lifecycle update

Latest log inspected:

- `/workspace/cann-recipes-train/llm_rl/qwen3_nm_model1_final/wjeagerqwen30b-a3b-with_draft_breakdown_20260611170916338365363_elastic.txt`
- Header still reports `mode=1 floor=2`; no successful `floor=1` log was found in this repo snapshot, so the fix is aimed at the shared low-floor mode1 path (`4->2` and `2->1`).

New evidence:

- Keeping the floor4 group cache avoided the previous `drop_old_floor_group_destroy_ms ~= 350s` symptom, but the next post-shrink DP warmup failed with a HCCL/allocator OOM trying to allocate about `1.56 GiB`.
- The direct-NPU expert import itself was fast in that run: no layer-level import row was close to the 350s scale.
- Therefore, retaining stale floor4 communicators is not a valid fix. It trades the destroy stall for old/new communicator workspace overlap and low-floor OOM.

Current code change:

- Do not keep floor4 group cache by default (`VLLM_ASCEND_MODE1_PARITY_KEEP_FLOOR4_GROUP_CACHE=0`). If enabled for diagnostics, it is kind-allowlisted and excludes MC2 by default.
- Add `VLLM_ASCEND_MODE1_PARITY_PRE_REBUILD_DESTROY_FLOOR_GROUP_SIZES=2,4` and destroy stale live DP/EP/MC2 groups of those sizes in `_stash_group_if_present()` before the new low-floor group is created.
- Keep the after-shrink old-floor cache drop as a sweep, because active live groups only enter the cache during rebuild.
- Skip synthetic mode1 post-shrink DP all-reduce warmup by default (`VLLM_ASCEND_MODE1_PARITY_ENABLE_POST_SHRINK_DP_WARMUP=0`) so the diagnostic warmup does not allocate a new DP workspace before the low-floor communicator lifecycle has settled.
- Add per-group destroy logging from `_stash_group_if_present()` so the next run can distinguish DP/EP/MC2 live-group destroy stalls from cache-drop stalls.

Why this is different from failed attempts:

- It does not retain stale low-floor groups.
- It does not rely on extra headroom.
- It changes the ordering from “create/prime new low-floor collectives while stale low-floor groups still occupy HBM” to “complete expert migration, destroy stale low-floor live groups while rebuilding, then create the new group.”

Next validation command:

- Floor=2 regression: `./run_mode1_floor2_local.sh`
- Floor=1 target path: `VLLM_ASCEND_ELASTIC_MIN_COMPUTE_GROUP_SIZE=1 ./run_mode1_floor2_local.sh`


## 2026-06-11 deferred low-floor destroy + CPU DP metadata sync

Additional conclusion from the 20260611 logs:

- `202606111515...`: inactive ranks dropped/destroyed old floor groups quickly, but active ranks waited about `351s` in `post_old_floor_group_drop_barrier_ms`.
- `202606111636...`: when all source ranks reached old-floor cache drop together, `drop_old_floor_group_destroy_ms` itself became about `351s` on every source rank.
- `202606111709...`: keeping floor4 cache avoided the old-floor destroy delay, but the next low-floor `HcclAllreduce` failed/OOMed.

Interpretation:

- The 350s is a low-level HCCL communicator destroy/allreduce lifecycle stall, not expert weight transfer.
- Synchronous `GroupCoordinator.destroy()` for stale low-floor groups is unsafe in the shrink RPC hot path.
- Fully retaining old low-floor groups is also unsafe if the first post-shrink metadata sync uses the NPU DP group, because it can force a new HCCL workspace while stale low-floor resources are still resident.

Current attempt:

- Add `VLLM_ASCEND_MODE1_PARITY_DEFER_GROUP_DESTROY=1` and `VLLM_ASCEND_MODE1_PARITY_DEFER_DESTROY_FLOOR_GROUP_SIZES=1,2,4,8`.
- Old low-floor groups are removed from vLLM state/cache so they cannot be reused, but their Python objects are quarantined in `_elastic_deferred_destroy_groups` instead of synchronously calling `torch.distributed.destroy_process_group()` inside shrink.
- Keep `VLLM_ASCEND_MODE1_PARITY_ENABLE_POST_SHRINK_DP_WARMUP=0` so synthetic DP warmup does not force HCCL allocation during shrink.
- Add `VLLM_ASCEND_MODE1_CPU_DP_METADATA_SYNC=1` so post-shrink DP metadata sync uses the CPU group for mode=1, matching the existing mode3/4/5 safety pattern and avoiding the first decode HcclAllreduce allocation/OOM.

Expected validation signal:

- `drop_old_floor_group_destroy_ms` should stay near `0` in shrink phase breakdowns.
- Logs should include `Elastic parallel group destroy deferred` for stale floor groups.
- No `post_old_floor_group_drop_barrier_ms ~= 350000` and no `HcclAllreduce` OOM immediately after shrink.
- The run should move back toward the reference `rollout_output_time_s ~= 120s`; if not, inspect the next largest phase rather than revisiting expert transfer.


## 2026-06-11 stale group destroy quarantine coverage update

Latest evidence reviewed:

- `wjeagerqwen30b-a3b-with_draft_breakdown_20260611163659764437651_elastic.txt` still shows the core stall in communicator lifecycle, not expert transfer:
  - direct NPU import max wall time is about `58ms`;
  - stage=2 `drop_old_floor_group_destroy_ms ~= 351s` across source ranks;
  - stage=4 has smaller but same-shape destroy stalls around hundreds of ms.
- `wjeagerqwen30b-a3b-with_draft_breakdown_20260611170916338365363_elastic.txt` shows that retaining stale floor4 groups can avoid the immediate destroy stall, but it then fails on post-shrink NPU DP `HcclAllreduce` with a `~1.56GiB` workspace allocation. Therefore, keeping stale groups reachable is not a valid fix.

Correction to earlier attempt:

- The synchronous pre-rebuild destroy idea is superseded. It still relies on `GroupCoordinator.destroy()` in the shrink/restore hot path, which the logs now prove is the unsafe operation.
- The active fix is quarantine/defer, not synchronous destroy.

Code changes in this iteration:

- Extended mode=1 low-floor deferred destroy coverage to all three relevant stale group removal paths:
  - old-floor cache drop after shrink;
  - stashed live group removal before low-floor rebuild;
  - generic stale cache sweep after restore.
- These paths now remove stale groups from vLLM state/cache/signatures so they cannot be reused, but keep the Python group object in `_elastic_deferred_destroy_groups` instead of calling `group.destroy()` synchronously.
- Full-restore low-floor live group handling now also uses the same defer rule; otherwise the 350s stall could simply move from shrink to restore.
- Mode=1 DP metadata sync remains forced to CPU by default (`VLLM_ASCEND_MODE1_CPU_DP_METADATA_SYNC=1`), and synthetic post-shrink DP all-reduce warmup remains disabled by default (`VLLM_ASCEND_MODE1_PARITY_ENABLE_POST_SHRINK_DP_WARMUP=0`). This avoids the post-shrink NPU DP `HcclAllreduce`/workspace allocation failure seen when stale floor4 resources remain resident.

Expected validation signal for the next user run:

- Header includes `mode1_defer_group_destroy=1`, `mode1_defer_destroy_floor_sizes=1,2,4,8`, `mode1_cpu_dp_metadata_sync=1`, and `mode1_enable_post_shrink_dp_warmup=0`.
- Logs include `Elastic parallel group destroy deferred` for stale floor groups.
- Shrink breakdown has `drop_old_floor_group_destroy_ms=0` and no `drop_old_floor_group_call_wall_ms ~= 350000`.
- No post-shrink `HcclAllreduce`/OOM immediately after stage=4 or stage=2.
- If runtime is still slow, the next largest phase should be inspected; do not return to expert-transfer tuning unless direct import rows become slow again.

Static validation after this edit:

```bash
python3 -m py_compile vllm_ascend/worker/worker_v1.py vllm_ascend/worker/model_runner_v1.py analyze_mode1_floor2_log.py
bash -n run_mode1_floor2_local.sh internal/wj_train_grpo_qwen30b_a3b_16die_true_weight_eager.sh
git diff --check -- vllm_ascend/worker/worker_v1.py vllm_ascend/worker/model_runner_v1.py run_mode1_floor2_local.sh internal/wj_train_grpo_qwen30b_a3b_16die_true_weight_eager.sh analyze_mode1_floor2_log.py MODE1_FLOOR2_350S_DIAGNOSIS.md
```

All passed locally.

## 2026-06-11 runtime-guarded quarantine fix

Latest user-provided log analyzed:

- `wjeagerqwen30b-a3b-with_draft_breakdown_20260611163659764437651_elastic.txt`

Confirmed facts:

- The direct NPU expert import path is not the 350s bottleneck. In the 4->2 shrink rows, max direct import wall time is about `58ms`, while `drop_old_floor_group_destroy_ms` is about `351s`.
- Earlier logs split the same stall between inactive ranks doing `drop_old_floor_group_call_wall_ms` and active ranks waiting in `post_old_floor_group_drop_barrier_ms`. Once all source ranks reached old-floor drop together, the time moved directly into `drop_old_floor_group_destroy_ms`.
- Keeping the stale floor4 group cache avoids that destroy stall, but then the next NPU DP allreduce/warmup can OOM while old HCCL workspace is still resident. So stale groups must be removed from vLLM state/cache, but their low-level destroy must not happen synchronously inside the shrink hot path.

Bug in the previous mitigation:

- Several mode=1 lifecycle guards still depended on `_has_mode1_lightweight_parity_module()` scanning the model modules.
- During shrink/restore/detach transitions, that scan is not a reliable control-plane predicate. If it returns false on any rank, that rank can fall back to the dangerous path: synchronous old-floor `group.destroy()`, NPU DP warmup, or synthetic MoE warmup.

Code changes in this iteration:

- Added/used `_is_mode1_low_floor_elastic_runtime()` as the authoritative guard for low-floor mode=1 lifecycle behavior: execution mode is 1, elastic shrink is enabled, and configured floor is <= 8.
- Switched old-floor cache drop, stale full-world EP keep checks, pre-rebuild live-group removal, full-restore live-group removal, rebuild-time old-floor drop/barrier, and after-shrink old-floor drop branches to the runtime guard instead of module scanning.
- Kept `VLLM_ASCEND_MODE1_PARITY_DEFER_GROUP_DESTROY=1` default. Old DP/EP/MC2 floor groups are removed from parallel state/cache/signatures and quarantined in `_elastic_deferred_destroy_groups`, not synchronously destroyed.
- Kept `VLLM_ASCEND_MODE1_CPU_DP_METADATA_SYNC=1` default so decode metadata sync uses CPU DP group in mode=1.
- Kept `VLLM_ASCEND_MODE1_PARITY_ENABLE_POST_SHRINK_DP_WARMUP=0` default and added a runtime-guarded skip for post-shrink MoE synthetic warmup (`reason=mode1_low_floor_runtime_no_synthetic_warmup`).
- Updated cleanup defaults so mode=1 low-floor does not call `empty_cache()+synchronize()` just because module scanning failed.

Expected next-run signals:

- Header contains `mode1_defer_group_destroy=1`, `mode1_defer_destroy_floor_sizes=1,2,4,8`, `mode1_enable_post_shrink_dp_warmup=0`, and `mode1_cpu_dp_metadata_sync=1`.
- Logs include `Elastic parallel group destroy deferred` for stale low-floor DP/EP/MC2 groups.
- Shrink phase rows show `drop_old_floor_group_destroy_ms=0` and no `drop_old_floor_group_call_wall_ms ~= 350000`.
- Warmup rows show `Elastic post-shrink DP all_reduce warmup skipped` and `Elastic post-shrink MoE dispatch warmup skipped ... reason=mode1_low_floor_runtime_no_synthetic_warmup`.
- If runtime is still slow, the next target should be the new largest phase in `analyze_mode1_floor2_log.py`; do not revisit expert transfer unless import rows become slow again.

## 2026-06-11 defer scope narrowed to avoid carrying floor8 HCCL workspace

Latest visible evidence:

- `202606111636...` confirms the main 350s delay is still stale low-floor `GroupCoordinator.destroy()` / HCCL ProcessGroup destroy, not expert import:
  - 4->2 direct-NPU import max wall is about 58ms;
  - stage=2 `drop_old_floor_group_destroy_ms ~= 351s`.
- `202606111709...` with `mode1_keep_floor4_group_cache=1` confirms that retaining low-floor groups can remove the immediate destroy delay, but then the following post-shrink DP warmup hits `HcclAllreduce` OOM while old communicator resources remain resident.
- Therefore the correct direction is not expert-copy tuning and not permanent stale-group retention. The shrink path must avoid synchronous low-floor HCCL destroy, but also avoid carrying unnecessary larger old communicators into lower floors.

New code change:

- Narrow default `VLLM_ASCEND_MODE1_PARITY_DEFER_DESTROY_FLOOR_GROUP_SIZES` from `1,2,4,8` to `1,2,4`.
  - Reason: historical 8->4 old floor group destroy is only hundreds of ms, not 350s. Deferring the old 8-rank groups keeps extra HCCL workspace alive and increases 4->2 / 2->1 OOM risk.
  - Keep deferring size 4/2/1 groups because the 4->2 path is the confirmed 350s destroy site.
- Added `drop_old_floor_group_deferred_groups` to `Elastic parallel shrink phase breakdown` so the next run can prove whether a stage actually deferred stale groups or simply had no stale groups in cache.
- Updated `analyze_mode1_floor2_log.py` to report:
  - deferred destroy row count and distribution by kind/size;
  - warmup skip lines;
  - HCCL/OOM/timeout error signal lines.

Expected validation signal:

- Header shows `mode1_defer_destroy_floor_sizes=1,2,4`.
- Stage=8 should no longer defer/delay old 8-rank groups; this should reduce later HBM pressure.
- Stage=4/2/1 should show nonzero `drop_old_floor_group_deferred_groups`, while `drop_old_floor_group_destroy_ms` remains near 0.
- Logs should include post-shrink DP warmup skip and no `HcclAllreduce` OOM.
- If a new OOM appears, inspect whether stale MC2/EP/DP groups are being deferred too broadly rather than returning to expert import tuning.

## 2026-06-11 retired-group idempotence fix

Latest user-provided evidence:

- The 350s cost is still concentrated in stale low-floor communicator lifecycle, not expert import.
- Logs showed inactive/source ranks spending time in `drop_old_floor_group_call_wall_ms` or `drop_old_floor_group_destroy_ms`, while active ranks wait at the corresponding post-drop barrier.
- A keep-floor4-cache experiment removes that synchronous destroy but keeps HCCL workspace resident long enough to trigger later allreduce/workspace OOM. Therefore permanent stale-group retention is not viable.

New failure mode found by static code review:

- `drop_old_floor_cached_elastic_parallel_groups_after_mode1_shrink()` can pop and destroy/defer a cached stale group while the current parallel-state module still holds the same group object.
- The following `_stash_group_if_present()` then sees the same live object. If the object was already destroyed, the previous code could re-cache this retired wrapper; if already deferred, it could append another deferred reference.
- This creates stale/ghost communicator references and can explain apparently random later slowdowns or workspace pressure even after direct import became fast.

Code changes in this iteration:

- Added `_elastic_deferred_destroy_group_ids` and `_elastic_retired_group_ids` to make stale group handling idempotent by object identity.
- `_quarantine_deferred_elastic_group()` now returns whether it actually appended a new deferred group and refuses duplicate quarantine of the same object.
- Sync `group.destroy()` paths now mark the group as retired immediately after destroy and reference detachment.
- `_stash_group_if_present()` now checks whether the group is already retired/deferred. If yes, it only removes any cache/signature reference and clears the live state pointer; it does not re-cache, re-destroy, or re-defer the object.
- Added `_remove_cached_elastic_parallel_group_reference()` so live-state stashing can remove duplicate cache entries for the same object before release.
- Fixed restore stale-cache cleanup to handle DP/EP/MC2 uniformly; MC2 only gets extra memory logging.
- Broadened default `VLLM_ASCEND_MODE1_PARITY_PRE_REBUILD_DESTROY_FLOOR_GROUP_SIZES` from `2,4` to `2,4,8`, while keeping `VLLM_ASCEND_MODE1_PARITY_DEFER_DESTROY_FLOOR_GROUP_SIZES=1,2,4`.
  - Rationale: old 8-rank groups should not be re-cached, but historical 8-rank destroy cost is sub-second, so they should be synchronously retired instead of deferred into later low-floor stages.
  - Old 4/2/1 groups remain deferred because the 4->2 path is the confirmed 350s destroy site.

Expected validation signal for next run:

- Header shows `mode1_pre_rebuild_destroy_floor_sizes=2,4,8` and `mode1_defer_destroy_floor_sizes=1,2,4`.
- Stage=4 old 8-rank groups should be released/retired without being re-cached.
- Stage=2 old 4-rank groups should show deferred release, not `drop_old_floor_group_destroy_ms ~= 350000`.
- Analyzer should show deferred destroy rows but no HCCL allreduce OOM from post-shrink warmup.
- If runtime is still slow, inspect whether the next largest phase is a remaining live old-group destroy or a barrier waiting for ranks that did not reach the same lifecycle branch.

Static validation after this edit:

```bash
python3 -m py_compile vllm_ascend/worker/worker_v1.py vllm_ascend/worker/model_runner_v1.py analyze_mode1_floor2_log.py
bash -n run_mode1_floor2_local.sh internal/wj_train_grpo_qwen30b_a3b_16die_true_weight_eager.sh
```

Both passed locally.

Analyzer update in the same iteration:

- `analyze_mode1_floor2_log.py` now reports `stashed_release_rows` with distributions by reason/kind/size and `already_retired` count.
- Use this after the next run to confirm that stale live groups are no longer re-cached after the cache-drop path already handled the same object.

## 2026-06-11 split live-group release from cache-drop timing

New analysis from the latest visible failed run:

- `wjeagerqwen30b-a3b-with_draft_breakdown_20260611170916338365363_elastic.txt` is a diagnostic `mode=1 floor=2` run with `mode1_keep_floor4_group_cache=1`, not the final `floor=1` target.
- It avoids the 350s old-floor destroy, but then fails in `_warmup_post_shrink_dp_collectives()` with `HcclAllreduce` OOM allocating about 1.56GiB. This confirms permanent stale floor4 retention is not acceptable.
- The failure occurred in older code where DP warmup was still active. Current code keeps `VLLM_ASCEND_MODE1_PARITY_ENABLE_POST_SHRINK_DP_WARMUP=0`.

Important lifecycle correction:

- Previous default `DROP_OLD_FLOOR_BEFORE_REBUILD=1` let cache cleanup run before active ranks stashed their current live DP/EP/MC2 groups in `_rebuild_group()`.
- That ordering is backwards: cache is not the owner of the live parallel state. It can retire a group object while vLLM parallel state still points to it, forcing the later stashing path to rely on idempotence and risking ghost wrapper/cache behavior.
- New default: `VLLM_ASCEND_MODE1_PARITY_DROP_OLD_FLOOR_BEFORE_REBUILD=0`.
- Added a separate switch: `VLLM_ASCEND_MODE1_PARITY_RELEASE_LIVE_OLD_FLOOR_ON_REBUILD=1`.
  - This keeps live old-group release enabled during `_rebuild_group()`.
  - Cache sweeping remains enabled before refresh via `DROP_OLD_FLOOR_BEFORE_REFRESH=1`, after live state has been detached/rebuilt.
- Analyzer now reports `mode1_release_live_old_floor_on_rebuild` from the header.

Expected next-run signal:

- Header must show `mode1_drop_old_floor_before_rebuild=0` and `mode1_release_live_old_floor_on_rebuild=1`.
- `stashed_release_rows` should appear for old live groups during active rebuild, especially old 8/4/2 floor groups.
- `drop_old_floor_group_deferred_groups` should appear after live stashing, not before it.
- `drop_old_floor_group_destroy_ms` must not return to ~350000ms at stage 2 or stage 1.
- No post-shrink DP warmup `HcclAllreduce` should appear.

## 2026-06-11 align local validation with floor=1 objective

Current visible logs under `qwen3_nm_model1_final` are all `mode=1 floor=2`; there is no current floor=1 worker log in the local worktree. The active objective is floor=1, so the local wrapper must not keep silently defaulting to floor=2.

Change:

- `run_mode1_floor2_local.sh` now defaults `VLLM_ASCEND_ELASTIC_MIN_COMPUTE_GROUP_SIZE=1`.
- The main training script still defaults to 2 for compatibility, but the wrapper used for this diagnosis now targets the requested floor=1 unless explicitly overridden.

Expected next-run signal:

- Header must show `mode=1 floor=1`.
- Shrink stages should include 8, 4, 2, and 1.
- The previous 350s destroy must not appear at stage=2 or stage=1.

## 2026-06-11 retire stale wrappers without hot-path PG destroy

Latest evidence from `wjeagerqwen30b-a3b-with_draft_breakdown_20260611163659764437651_elastic.txt`:

- `rollout_output_time_s=446.883083`.
- Direct NPU import/export is healthy: final 4->2 `max_wall_ms` is about 58ms.
- The dominant cost is not expert transfer. It is old low-floor group cleanup:
  - stage=2 `drop_old_floor_group_call_wall_ms ~= 351098ms`.
  - stage=2 `drop_old_floor_group_destroy_ms ~= 351097ms`.
- Active ranks spend only secondary time in `post_old_floor_group_drop_barrier_ms`; they are waiting for ranks still blocked in old-group cleanup.

Related failed experiment from `wjeagerqwen30b-a3b-with_draft_breakdown_20260611170916338365363_elastic.txt`:

- Keeping floor4 group cache avoids the 350s destroy path, but later fails with `HcclAllreduce` memory allocation failure around 1.56GiB.
- This proves the fix cannot be permanent stale-group reuse/cache retention. Stale groups must disappear from all live/cache lookup paths.

Code change in this iteration:

- Added `_retire_group_wrapper_without_pg_destroy()`.
- `_quarantine_deferred_elastic_group()` now marks the group retired and calls the wrapper-retirement helper before appending the private deferred reference.
- The helper calls `device_communicator.destroy()` and clears communicator/broadcaster references, but intentionally does not call `torch.distributed.destroy_process_group()` from the shrink hot path.
- This keeps the semantic behavior we need: no cache hit/reuse of stale groups, no active live pointer to stale groups, and no synchronous HCCL ProcessGroup destroy during shrink.
- Restored `run_mode1_floor2_local.sh` default floor to 2, because the current investigation is explicitly mode=1 floor=2. Floor=1 can still be requested explicitly with `VLLM_ASCEND_ELASTIC_MIN_COMPUTE_GROUP_SIZE=1`.

Expected next-run signal:

- Header: `mode=1 floor=2`, `mode1_keep_floor4_group_cache=0`, `mode1_enable_post_shrink_dp_warmup=0`, `mode1_drop_old_floor_before_rebuild=0`, `mode1_release_live_old_floor_on_rebuild=1`.
- Stage=4/2 stale low-floor groups should produce `Elastic parallel group destroy deferred` / `Elastic parallel stashed group released` rows.
- Stage=2 `drop_old_floor_group_destroy_ms` should stay near 0, with nonzero `drop_old_floor_group_deferred_groups` instead of 350000ms destroy.
- No `HcclAllreduce` OOM should appear, because stale groups are no longer kept as reusable floor4 cache and post-shrink DP warmup is disabled by default.

If the next run still OOMs:

- Inspect whether deferred groups still retain raw ProcessGroup HCCL workspace too strongly. The next candidate is a stricter CPU-only barrier plus no device ProcessGroup creation for old non-member catch-up where safe, but this must be tested only after confirming the current retired-wrapper behavior.

## 2026-06-11 floor=1 objective correction and tombstone quarantine

Correction:

- The active objective is `mode=1, floor=1`, not floor=2.
- Restoring `run_mode1_floor2_local.sh` to default floor=2 was a misaligned diagnostic step. It has been reverted; the wrapper now defaults to `VLLM_ASCEND_ELASTIC_MIN_COMPUTE_GROUP_SIZE=1` again.
- The main training script still defaults to floor=2 for compatibility, so this diagnosis should use the local wrapper or explicitly set floor=1.

New code concern addressed:

- Previous quarantine stored the full `GroupCoordinator` object in `_elastic_deferred_destroy_groups`.
- That avoided hot-path `destroy_process_group()`, but it could still strongly reference `device_group`/`cpu_group` through the wrapper and keep HCCL resources resident into later floor=1/restore stages.
- `_elastic_deferred_destroy_groups` now stores only tombstone metadata: group id, kind, ranks, reason.
- `_retire_group_wrapper_without_pg_destroy()` now clears `device_communicator`, `mq_broadcaster`, and direct `device_group`/`cpu_group` attributes from the wrapper without explicitly calling `torch.distributed.destroy_process_group()` in the shrink hot path.
- Deferred logs now include `retire_wrapper_ms` and `dropped_pg_refs`, and the analyzer reports aggregate retire-wrapper timing.

Expected next-run signal for true target:

- Header must show `mode=1 floor=1`.
- Shrink stages should include 8, 4, 2, and 1.
- Stage=2 and stage=1 should not show `drop_old_floor_group_destroy_ms ~= 350000`.
- `deferred_destroy_rows` should be nonzero with low `retire_wrapper_ms`.
- `deferred_destroy_retire_wrapper_ms` should be small; if it is large, PG attribute release is triggering the same underlying stall and needs to be deferred differently.
- No `HcclAllreduce` / workspace OOM should occur after shrink.

## 2026-06-11 default defer coverage must include every low-floor stage

New conclusion from the latest diagnosis:

- The 350s regression is confirmed to be old low-floor communication group teardown, not expert import/export.
- `GroupCoordinator.destroy()` calls `device_communicator.destroy()` and then `torch.distributed.destroy_process_group()` for owned device/cpu process groups.
- Therefore any old floor DP/EP/MC2 group that reaches `group.destroy()` in `rebuild_elastic_ep_group()` can synchronously block the shrink RPC path.
- Previous defer default was `1,2,4`, while the floor=1 run traverses old group sizes `8 -> 4 -> 2 -> 1` across shrink/restore. Size 8 was not deferred by default, and size 1 was not included in the pre-rebuild live-release list.
- This explains why stale group cleanup could still enter direct destroy even after adding the defer/quarantine machinery.

Code change:

- Default `VLLM_ASCEND_MODE1_PARITY_DEFER_DESTROY_FLOOR_GROUP_SIZES` changed from `1,2,4` to `1,2,4,8` in both code and scripts.
- Default `VLLM_ASCEND_MODE1_PARITY_PRE_REBUILD_DESTROY_FLOOR_GROUP_SIZES` changed from `2,4,8` to `1,2,4,8` in both code and scripts.
- The intended behavior is now: all non-fullworld low-floor stale groups are removed from live/cache lookup paths and quarantined, but not synchronously destroyed during shrink.

Expected next-run signal:

- Header must show `mode=1 floor=1`, `mode1_defer_destroy_floor_sizes=1,2,4,8`, and `mode1_pre_rebuild_destroy_floor_sizes=1,2,4,8`.
- Stage 8/4/2/1 should show stale groups as `deferred_groups`, not `drop_old_floor_group_destroy_ms ~= 350000`.
- Analyzer should report nonzero `deferred_destroy_rows` and low `retire_wrapper_ms`.
- If there is still OOM, the next suspect is that retired wrapper attribute clearing drops the final raw ProcessGroup reference too early or too late; inspect `retire_wrapper_ms` and HBM before changing headroom.

## 2026-06-11 split device PG release from CPU/Gloo PG defer

Evidence from `wjeagerqwen30b-a3b-with_draft_breakdown_20260611170916338365363_elastic.txt`:

- This run used `mode1_keep_floor4_group_cache=1`; it avoided the 350s old-floor destroy path, but failed before rollout completion.
- Analyzer shows `drop_old_floor_group_destroy_ms=0` at stage=2, but later `HcclAllreduce` failed with `Memory_Allocation_Failure`.
- Therefore permanent stale floor4 cache/PG retention is not viable. It trades the 350s teardown stall for retained HCCL/HBM workspace and OOM.

New hypothesis:

- The 350s stall is likely dominated by CPU/Gloo or generic ProcessGroup teardown during `GroupCoordinator.destroy()`.
- The OOM from keep-cache is caused by keeping device/HCCL ProcessGroups alive.
- We need to remove stale groups from live/cache lookup paths, release device PG/HBM workspace, but avoid CPU/Gloo PG teardown inside the shrink RPC hot path.

Code change:

- `_retire_group_wrapper_without_pg_destroy()` now splits teardown:
  - optional `torch.npu.synchronize()` before device PG retirement;
  - explicit `torch.distributed.destroy_process_group(device_group)` for the device PG;
  - keep CPU/Gloo PG referenced in `_elastic_deferred_destroy_groups` so Python/Gloo teardown does not occur during shrink;
  - remove stale group from live/cache/signature paths so it cannot be reused.
- Deferred log now includes `npu_sync_ms`, `device_pg_destroy_ms`, `dropped_pg_refs`, and `deferred_cpu_pg_refs`.
- Analyzer now reports `deferred_destroy_npu_sync_ms` and `deferred_destroy_device_pg_destroy_ms`.

Expected next-run signal:

- Header: `mode=1 floor=1`, `mode1_keep_floor4_group_cache=0`, `mode1_defer_destroy_floor_sizes=1,2,4,8`, `mode1_sync_before_device_pg_retire=1`.
- Stage 8/4/2/1 should show `deferred_groups > 0` and `drop_old_floor_group_destroy_ms ~= 0`.
- `deferred_destroy_device_pg_destroy_ms` should be small-to-moderate, not 350s.
- No `HcclAllreduce` OOM should occur because stale HCCL/device PGs are explicitly destroyed.
- If OOM remains, device PG destroy is insufficient or too late. If 350s returns in `device_pg_destroy_ms`, the stall is HCCL destroy rather than CPU/Gloo.

### 2026-06-11 follow-up: collect device PG before clearing references

Review found a correctness gap in the split-retire helper:

- The helper cleared `communicator.device_group` before collecting all possible raw device ProcessGroup references.
- If the device communicator held the only useful reference, this could fail to release the HCCL/device PG and reproduce the OOM side of the failure.

Code change:

- `_retire_group_wrapper_defer_cpu_pg_destroy()` now first collects device PGs from both the `GroupCoordinator` wrapper and its `device_communicator`, deduplicates them by object id, then clears references, then destroys only those device PGs.
- CPU/Gloo PG references are intentionally kept in `_elastic_deferred_destroy_groups` to avoid CPU/Gloo teardown in the shrink RPC path.

Expected next-run signal remains:

- no 350s `drop_old_floor_group_destroy_ms`;
- nonzero `deferred_destroy_rows`;
- device PG destroy timing visible via `deferred_destroy_device_pg_destroy_ms`;
- no HCCL OOM.

### 2026-06-11 follow-up: defer ordinary stashed low-floor groups too

Additional static review found another possible direct-destroy path:

- `_stash_group_if_present()` only used the deferred retire path for the explicit `mode1_pre_rebuild` and `mode1_full_restore` cases.
- If a stale low-floor DP/EP/MC2 group reached the ordinary `stash_group` branch, it could still call `group.destroy()` directly.
- Even if this path is uncommon with current cache defaults, it is not aligned with the floor=1 objective: no stale 1/2/4/8 low-floor group should synchronously destroy ProcessGroups in the shrink/restore hot path.

Code change:

- Added `_should_defer_stashed_group_destroy_for_mode1()` and use it for all stashed group release reasons.
- It still delegates to `_should_defer_mode1_group_destroy()`, so it only applies to mode=1 low-floor non-fullworld stale DP/EP/MC2 groups whose size is in `1,2,4,8` and whose ranks differ from the rebuild target.
- This should not affect mode=3/4/5 or full-world groups.

Expected next-run signal:

- Any old low-floor live/cache group should produce `Elastic parallel group destroy deferred` instead of `Elastic parallel stashed group released ... destroy_ms=...` with a large value.

### 2026-06-11 follow-up: defer only raw CPU PGs, not full GroupCoordinator

Further static review found that keeping the full `GroupCoordinator` in `_elastic_deferred_destroy_groups` can still retain `device_communicator` and possible all2all/communication-buffer state. That risks reproducing the HCCL/HBM OOM observed in the keep-cache experiment.

Code change:

- `_retire_group_wrapper_defer_cpu_pg_destroy()` now extracts raw CPU/Gloo ProcessGroup references into a deduplicated list, clears both wrapper and communicator CPU/device group attributes, destroys only the device PGs, and returns the raw CPU PG list.
- `_elastic_deferred_destroy_groups` stores only `cpu_pgs` plus metadata, not the full `GroupCoordinator` object.
- This should prevent CPU/Gloo PG teardown in the shrink hot path while avoiding retention of device communicator/HCCL-related objects.

Expected signal:

- `deferred_cpu_pg_refs > 0` confirms CPU PGs are being held deliberately.
- `dropped_pg_refs > 0` and small `device_pg_destroy_ms` confirm device PGs are being released.
- Absence of HCCL OOM confirms full communicator retention was avoided.


## 2026-06-11 update: export stall removed, second 350s source identified

Latest checked log:

- `/workspace/cann-recipes-train/llm_rl/qwen3_nm_model1_final/wjeagerqwen30b-a3b-with_draft_breakdown_20260611170916338365363_elastic.txt`

What changed:

- Final 4->2 shrink no longer spends 350s in direct-NPU export/import.
- Source ranks 12/13 report `slot_map_expert_cpu_ready=1` and `slot_map_loaded_cpu_ready=1`.
- Final 4->2 layer-0 direct-NPU transfer is now about 59ms wall time:
  - rank 12: `send_export_ms=0.95`, `send_post_ms=57.22`, `wall_ms=58.98`
  - rank 13: `send_export_ms=0.95`, `send_post_ms=57.88`, `wall_ms=59.66`
- Full preload for final 4->2 is about 120-133ms per rank, not 350s.

New failure:

- Active ranks 14/15 later fail in `_warmup_post_shrink_dp_collectives()` while synchronizing a post-shrink `HcclAllreduce` warmup.
- The runtime reports `Memory_Allocation_Failure(EL0004)` for a `DeviceMemory` allocation of `1678770176` bytes.
- This indicates a second 350s-class source: post-shrink DP all_reduce warmup / HCCL workspace initialization, not expert-weight P2P transfer.

Code response:

- Mode=1 post-shrink DP all_reduce warmup is now disabled by default.
- It can be explicitly re-enabled with `VLLM_ASCEND_MODE1_PARITY_ENABLE_POST_SHRINK_DP_WARMUP=1` if needed for an A/B run.

Next validation expectation:

- The next run should show `Elastic post-shrink DP all_reduce warmup skipped ... reason=mode1_default_disabled` for mode=1 shrinks.
- If the job completes, compare `rollout_output_time_s` against the healthy reference around 120s.
- If the first real post-shrink DP metadata all_reduce later fails instead, then the all_reduce workspace must be reserved through KV/headroom or moved to CPU metadata sync instead of being merely warmed up.


### 2026-06-11 latest: old-floor destroy is still the slow point; fix default split-retire policy

Evidence from `wjeagerqwen30b-a3b-with_draft_breakdown_20260611163659764437651_elastic.txt`:

- Direct NPU expert transfer is healthy in final 4->2: max `wall_ms=58.18`, max `send_post_ms=56.43`, max `recv_post_ms=56.57`.
- The 350s stall is concentrated in old-floor group cleanup, not expert payload transfer: stage=2 `drop_old_floor_group_destroy_ms ~= 351097ms`, `drop_old_floor_group_call_wall_ms ~= 351098ms`.
- `deferred_destroy_rows=0`, so this run did not actually validate the deferred destroy path.

Evidence from `wjeagerqwen30b-a3b-with_draft_breakdown_20260611170916338365363_elastic.txt`:

- Keeping floor4 groups (`mode1_keep_floor4_group_cache=1`) avoids the stage=2 old-floor destroy cost but fails later with `HcclAllreduce` `Memory_Allocation_Failure`.
- Therefore retaining stale device/HCCL PGs is not viable; it prevents the stall only by keeping HCCL workspace alive and then OOMs later collectives.

Code change in this iteration:

- Default `VLLM_ASCEND_MODE1_PARITY_DESTROY_DEVICE_PG_ON_RETIRE=1`.
- The retire path removes stale wrappers/cache immediately, destroys device/HCCL PGs to free HBM/workspace, and defers only CPU/Gloo PG references.
- Added `destroy_device_pg` and `deferred_device_pg_refs` to deferred-destroy logs/analyzer.
- Added low-floor direct-destroy fallback warnings so a future 350s direct path is visible as a missed defer predicate rather than hidden in `drop_old_floor_group_destroy_ms`.

Expected next-run signal:

- Header includes `mode1_destroy_device_pg_on_retire=1`.
- `deferred_destroy_rows > 0`.
- Stage=4/2 old-floor `drop_old_floor_group_destroy_ms` is near 0.
- `deferred_destroy_device_pg_destroy_ms` is not 350s.
- No `HcclAllreduce` OOM. If OOM persists, device PG release is still incomplete or HCCL workspace belongs to another retained object. If `device_pg_destroy_ms` becomes 350s, the slow object is HCCL PG destroy rather than CPU/Gloo PG teardown.

## 2026-06-14 diagnostic: force old NPU scalar slot-map path and capture exact stall op

Goal:

- The stable mode=1 path avoids the 350s stall by using CPU mirrors for expert slot lookup.
- This diagnostic intentionally re-enables the old suspect path only in a dedicated launcher, so we can identify whether the 350s is spent in NPU scalar `.item()`, alias creation, `isend/irecv` post, or HCCL wait.

Code changes:

- `vllm_ascend/ops/fused_moe.py`
  - Added opt-in `Mode1 NPU scalar slot read begin/done` logs around the exact `export_map[expert_id].item()` fallback.
  - Added `scalar_item_count`, `scalar_item_ms_total`, `scalar_item_ms_max`, and `scalar_item_debug_head` to `_last_lossless_npu_export_debug`.
- `vllm_ascend/worker/worker_v1.py`
  - Batch and scalar direct-NPU shrink paths now log slow export records with the full export debug dict.
  - Batch and scalar P2P paths now log slow `send_post/send_wait` and `recv_post/recv_wait` timing records.
  - Stream import breakdown now includes scalar item count/total/max so `send_export_ms` can be attributed directly.
- `internal/wj_train_grpo_qwen30b_a3b_16die_true_weight_eager.sh`
  - `ASCEND_GLOBAL_LOG_LEVEL`, `ASCEND_GLOBAL_EVENT_ENABLE`, `ASCEND_SLOG_PRINT_TO_STDOUT`, `HCCL_CONNECT_TIMEOUT`, and `HCCL_EXEC_TIMEOUT` now respect external env overrides for diagnostic runs.
- New scripts:
  - `run_mode1_scalar_stall_diag.sh`
  - `analyze_mode1_scalar_stall_diag.py`

How to run the old suspect path:

```bash
cd /workspace/cann-recipes-train/llm_rl/qwen3_nm_model1_final
./run_mode1_scalar_stall_diag.sh --floor 2
```

Optional single-expert/non-batch repro:

```bash
cd /workspace/cann-recipes-train/llm_rl/qwen3_nm_model1_final
./run_mode1_scalar_stall_diag.sh --floor 2 --no-batch
```



Log layout for diagnostic runs:

- `diag_header.log`: wrapper summary and post-run quick commands.
- `launcher.log`: outer wrapper stdout/stderr from `run_mode1_floor2_local.sh`.
- `wjeagerqwen30b-a3b-with_draft_*_elastic.txt`: primary training/worker log; this is where the scalar/P2P breakdown lines are expected.
- `ascend_log/`: Ascend/HCCL runtime logs from `ASCEND_PROCESS_LOG_PATH`.

The diagnostic launcher sets `RUN_OUTPUT_DIR=<run_dir>`, so the primary `wjeager..._elastic.txt` log is written inside the same run directory. The analyzer accepts either the run directory or the primary train log and resolves the correct file automatically.

What the diagnostic launcher changes for this run only:

- `VLLM_ASCEND_MODE1_USE_CPU_EXPORT_SLOT_MAP=0`
- `VLLM_ASCEND_MODE1_ALLOW_NPU_EXPORT_SLOT_FALLBACK=1`
- `VLLM_ASCEND_MODE1_DIAG_SCALAR_TRACE=1`
- `VLLM_ASCEND_MODE1_DIAG_EXPORT_INTERNAL_TIMING=1`
- `VLLM_ASCEND_MODE1_DIAG_SLOW_EXPORT_LOG_MS=10`
- `VLLM_ASCEND_MODE1_DIAG_SLOW_P2P_LOG_MS=10`
- Ascend/HCCL logs are written under `mode1_scalar_stall_diag_runs/<timestamp>/ascend_log`.

How to analyze:

```bash
python3 /workspace/cann-recipes-train/llm_rl/qwen3_nm_model1_final/analyze_mode1_scalar_stall_diag.py \
  /workspace/cann-recipes-train/llm_rl/qwen3_nm_model1_final/mode1_scalar_stall_diag_runs/<run>
```

Interpretation:

- If `scalar_begin > scalar_done`, the last `Mode1 NPU scalar slot read begin` before termination is the blocked Python-visible op.
- If `scalar_item_ms_max` is around 350000ms and `send_export_ms` matches it, the stall is exposed by `export_map[expert_id].item()`, not by P2P payload transfer.
- If `send_wait_ms` or `recv_wait_ms` is around 350000ms while scalar item times are small, the stall is in HCCL P2P wait.
- If `send_post_ms` or `recv_post_ms` is around 350000ms, the stall is in HCCL op enqueue/post or communicator readiness.
- If Python logs show the stall in scalar `.item()`, then Ascend/HCCL logs from the same run should be searched around that timestamp for runtime wait/retry/timeout messages:

```bash
rg -n 'HCCL|hccl|timeout|retry|wait|error|failed|stream|synchronize' \
  /workspace/cann-recipes-train/llm_rl/qwen3_nm_model1_final/mode1_scalar_stall_diag_runs/<run>/ascend_log
```

Expected comparison with stable path:

- Stable path should keep `send_export_scalar_item_count=0` and `send_export_slot_cpu_hits > 0`.
- Diagnostic old path should produce `send_export_scalar_item_count > 0` and `Mode1 NPU scalar slot read ...` lines.
- If the 350s reproduces only in the diagnostic old path, the stable CPU slot-map fix remains valid while this run identifies the precise lower-level exposure point.
