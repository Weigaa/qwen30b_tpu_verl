# Mode5 Current Status 2026-06-04

## Scope

This note summarizes the current validated status of `mode=5` in the Qwen3 elastic rollout path, the optimizations that are already proven useful, the experiments that are still unvalidated, and the exact next verification path once the machine becomes available again.

Workspace:
- `vllm_ascend/ops/fused_moe.py`
- `vllm_ascend/worker/worker_v1.py`
- `verl/single_controller/ray/base.py`
- `internal/wj_train_grpo_qwen30b_a3b_16die_true_weight_eager.sh`

## Objective

The optimization target remains:
- reduce end-to-end prefetch time for true `mode=5 dual_source`
- increase overlap between CPU->NPU and remote-NPU prefetch
- prefer same-package remote NPU transfers (`0-1, 2-3, ..., 14-15`)
- move runtime behavior closer to:
  - `prefetch time ~= max(npu-npu, cpu-npu)`

## Current Best Validated Result

Current best validated true `dual_source` run:
- log: `wjeagerqwen30b-a3b-with_draft_breakdown_20260604154651_elastic.txt`
- `rollout_output_time_s = 270.998779`
- `exit_code=0`

Comparison to earlier validated baselines:
- `20260604105119`: `295.626066`
- `20260604135215`: `277.125605`
- current best: `270.998779`

This means the optimized `mode=5 dual_source` path has already improved by about `24.63s` versus the early validated true-dual-source baseline.

## Validated Effective Optimizations

The following optimizations are already validated on the true `mode=5 dual_source` path and should be treated as the stable mainline:

1. CPU submit moved ahead of remote fetch submit
- file: `vllm_ascend/ops/fused_moe.py`
- goal: reduce host-side serial submit chain

2. Prefer same-package remote source ranks
- file: `vllm_ascend/worker/worker_v1.py`
- idea: for mode5, prefer remote NPU source in the same package (`0-1`, `2-3`, ...)

3. Batched remote control-message submission/wait
- file: `vllm_ascend/worker/worker_v1.py`
- idea: send all CPU-side control messages first, then wait, then issue irecv

4. Grouped slot order
- file: `vllm_ascend/ops/fused_moe.py`
- env: `VLLM_ASCEND_MODE5_GROUP_CPU_REMOTE_SLOT_ORDER`
- slot order becomes closer to:
  - resident local NPU
  - CPU-only experts
  - remote experts grouped by remote rank / remote slot

5. Remote payload bulk slot fill
- file: `vllm_ascend/ops/fused_moe.py`
- remote payload copy changed from row-by-row copy into bulk run copy using contiguous runs

These changes are the reason the validated best result reached `270.998779`.

## What Was Proven by Timing

On the validated best run, the main improvement came from reducing host-side prefetch organization cost, especially:
- `submit_populate_us`
- `submit_cpu_us`

Representative validated effect:

### Stage 2
Old validated best (`20260604135215`):
- `prefetch_total p50/p90 = 11.000 / 12.199 ms`
- `submit_populate p50/p90 = 4.032 / 4.765 ms`
- `submit_cpu p50/p90 = 2.394 / 2.472 ms`

Improved validated run (`20260604154651`):
- `prefetch_total p50/p90 = 7.264 / 7.900 ms`
- `submit_populate p50/p90 = 1.909 / 2.310 ms`
- `submit_cpu p50/p90 = 0.106 / 0.123 ms`

### Stage 1
Old validated best (`20260604135215`):
- `prefetch_total p50/p90 = 20.657 / 20.831 ms`
- `submit_populate p50/p90 = 7.401 / 7.479 ms`
- `submit_cpu p50/p90 = 4.790 / 4.810 ms`

Improved validated run (`20260604154651`):
- `prefetch_total p50/p90 = 13.691 / 14.287 ms`
- `submit_populate p50/p90 = 3.086 / 3.569 ms`
- `submit_cpu p50/p90 = 0.110 / 0.118 ms`

Interpretation:
- the mainline changes already pushed runtime behavior significantly closer to the intended overlap model
- the remaining gap is still mostly host/control/runtime organization overhead, not simply raw device-side bandwidth

## Current Unvalidated Experiment

There is an additional experiment in `vllm_ascend/worker/worker_v1.py`:
- `VLLM_ASCEND_MODE5_SINGLE_CONTROL_MESSAGE_REMOTE`

Idea:
- for mode5 only, merge remote fetch `shape + request` control messages into one fixed-size CPU control tensor
- intended goal: further reduce `submit_remote_npu_us` and therefore reduce `submit_populate_us`

Current status:
- code path exists
- it is now gated behind the explicit env flag above
- launcher default is set to `0`
- this experiment is NOT yet validated by a successful performance run

Reason for gating it off by default:
- recent attempts with this path did not enter a valid training/timing phase
- those runs only emitted the two-line elastic log header and never reached `Mode5 timing`
- therefore no trustworthy performance conclusion can be drawn from them yet

## Why the Experiment Is Disabled by Default

To keep the next rerun clean, the launcher now defaults to:
- `VLLM_ASCEND_MODE5_SINGLE_CONTROL_MESSAGE_REMOTE=0`

This ensures:
- default reruns go through the already validated mainline path
- the experimental control-message optimization is only enabled intentionally
- we do not accidentally mix an unverified startup-path experiment into the next performance validation

## Machine / Verification State

Current external limitation:
- the machine is frequently occupied by another full-device job or device-side residual occupancy not attributable to this container
- because of this, some recent reruns were invalid and should not be used as performance evidence

Current rule for future verification:
- only rerun when NPU is clearly free
- if NPU becomes occupied again, stop and wait rather than continuing blind retries

## Recommended Next Validation Order

Once the machine becomes available again:

1. First rerun the stable validated mainline only
- keep `VLLM_ASCEND_MODE5_SINGLE_CONTROL_MESSAGE_REMOTE=0`
- confirm the mainline still reproduces near the current best validated behavior
- expected target zone:
  - close to `rollout_output_time_s = 270.998779`

2. Only after that, run a single-variable experiment
- set:
  - `VLLM_ASCEND_MODE5_SINGLE_CONTROL_MESSAGE_REMOTE=1`
- compare only against the validated mainline on:
  - `submit_remote_npu_us`
  - `submit_populate_us`
  - `prefetch_total`
  - `rollout_output_time_s`

This keeps the next experiment interpretable.

## Commit Hygiene Note

The current workspace contains many unrelated and historical modifications beyond the mode5 optimization path.
For a safe push, the recommended commit scope is:
- this document
- `vllm_ascend/ops/fused_moe.py`
- `vllm_ascend/worker/worker_v1.py`
- `internal/wj_train_grpo_qwen30b_a3b_16die_true_weight_eager.sh`
- optionally `verl/single_controller/ray/base.py` only if the team still wants the HCCL base-port allocator change kept

It is not recommended to blindly push the entire dirty worktree as one commit.
