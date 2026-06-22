# Mode5 Next Validation Checklist 2026-06-04

## Goal
Validate whether the current offline `mode=5 dual_source` host-side optimizations reduce:

- `submit_remote_npu_us`
- `submit_populate_us`
- `prefetch_total`
- and the gap to `max(prefetch_remote_npu_dev_ms, prefetch_cpu_dev_ms)`

for the heavy stages:

- `stage=2`
- `stage=1`

## Current best verified baseline

Log:
- `wjeagerqwen30b-a3b-with_draft_breakdown_20260604154651_elastic.txt`

Key result:
- `rollout_output_time_s = 270.998779`

## Offline optimizations now present in current worktree

1. Prefer same-package remote cache ranks for mode5
   - `(0,1) (2,3) ... (14,15)`

2. Batch CPU-side remote control message posting

3. Group slot order for mode5
   - resident local first
   - cpu-only next
   - remote experts grouped by remote rank / remote slot

4. Remote payload copy into runtime slot uses contiguous slot runs
   - not prefix-only bulk copy
   - not per-row scatter fallback for all non-prefix cases

5. Remote cache payload packing uses contiguous `(layer_idx, remote_slot)` runs
   - both in prepacked payload build
   - and in on-demand service send path

6. Parallel remote fetch no longer waits for all recv requests before any payload copy
   - now `wait + copy` per pending recv in order

7. Remote request/control CPU tensor buffers are reused
   - avoid repeated `torch.tensor(list)` allocation path
   - warmup path aligned too

## Do not test under these conditions

Do not start a new run if any of the following is true:

1. `npu-smi` shows active processes
2. `npu-smi` shows persistent residual HBM usage across all 16 cards even without processes
3. another user's whole-node workload is visible
4. HCCL bind failure is obviously caused by a dirty shared-machine window

## Required env expectations

Use the validated true `dual_source` path, not legacy runtime:

- `VLLM_ASCEND_ELASTIC_EXECUTION_MODE=5`
- `VLLM_ASCEND_MODE5_USE_LEGACY_CPU_SHADOW_RUNTIME=0`
- `VLLM_ASCEND_MODE5_PREFER_SAME_PACKAGE_REMOTE=1`
- `VLLM_ASCEND_MODE5_GROUP_CPU_REMOTE_SLOT_ORDER=1`
- `VLLM_ASCEND_MODE3_TIMING_LOG=1`
- `VLLM_ASCEND_MODE3_TIMING_SYNC=1`
- `VLLM_ASCEND_MODE3_VERBOSE_TRANSFER_LOG=1`

Use the now-aligned single-control-message mode5 remote control path by default:

- `VLLM_ASCEND_MODE5_SINGLE_CONTROL_MESSAGE_REMOTE=1`

## First-pass validation metrics

For the newest successful `mode=5` log, extract and compare against baseline:

### Stage 2
- `prefetch_total p50/p90`
- `submit_populate_us p50/p90`
- `submit_remote_npu_us p50/p90`
- `submit_cpu_us p50/p90`
- `prefetch_remote_npu_dev_ms p50/p90`
- `prefetch_cpu_dev_ms p50/p90`

### Stage 1
- same fields as stage 2

## Success criterion for this next validation window

This round is a win if at least one of the following becomes true versus baseline `20260604154651`:

1. `submit_populate_us` falls further at stage 1 and stage 2 without reducing expert counts
2. `submit_remote_npu_us` falls further at stage 1 and stage 2
3. `prefetch_total - max(remote_npu_dev_ms, cpu_dev_ms)` gap shrinks further
4. `rollout_output_time_s < 270.998779`

## Expert-count invariants

Do not treat a run as valid if these change unexpectedly:

- `stage=2`: `8 local / 42 remote / 14 cpu`
- `stage=1`: `8 local / 90 remote / 30 cpu`

## If the run succeeds

Immediately record:

1. `rollout_output_time_s`
2. `exit_code`
3. stage-2/1 aggregate table
4. delta vs baseline `270.998779`

## If the run fails before timing appears

Check in this order:

1. shared-machine `NPU` occupancy / HBM residue
2. HCCL bind conflict
3. whether the newest log contains more than the two-line launcher header
4. only then inspect code regression possibilities
