# Mode3 Native Iteration Log

Purpose: keep a durable record of every mode=3 native/common change and run,
so we do not repeat ineffective optimizations.

## Baselines

- Reference custom log: `mode3-record.txt`
  - Path source: old `llm_rl/qwen3` custom/legacy fused_moe path.
  - Result: `rollout_output_time_s=212.526065`, `timing_s/gen=212.52104927494656`.
  - Tail timing from log: stage=8 comm resolution to restore is about 166s.
  - Per-layer timing sample from the log:
    - `current_compute_wall_ms` mean about 0.901 ms, p50 about 0.624 ms.
    - `fused_wall_ms` mean about 0.860 ms, p50 about 0.582 ms.
    - `prefetch_dev_ms` mean about 2.353 ms, p50 about 2.108 ms.
    - `prefetch_cpu_path=direct_async`, `source_from_npu=8`, `source_from_cpu=8`.
- Current branch custom mode=3, same threshold setup:
  - Log: `wjeagerqwen30b-a3b-with_draft_breakdown_20260528132918_elastic.txt`.
  - Result: `rollout_output_time_s=288.298`, `timing_s/gen=288.2942`.
  - Tail after shrink8 done: about 222.190s.
  - Important conclusion: current branch custom is already much slower than
    the old 212s reference. Do not treat the whole 100s delta as native-only.
- Current branch native/common mode=3, graph=0/task_queue=2:
  - Log: `wjeagerqwen30b-a3b-with_draft_breakdown_20260528131306_elastic.txt`.
  - Result: `rollout_output_time_s=311.861`, `timing_s/gen=311.8577`.
  - Tail after shrink8 done: about 242.096s.
  - Gap to current custom: about 23.6s.
- Current branch native/common mode=3, graph=1/task_queue=1:
  - Log: `wjeagerqwen30b-a3b-with_draft_breakdown_20260528142243_elastic.txt`.
  - Result: `rollout_output_time_s=325.399776`.
  - Conclusion: graph=1/task_queue=1 made native slower in this branch.
    Do not repeat as a speed optimization without a new reason.
- Current branch native/common mode=3, layer local buffer plus
  `EXPERT_TOKEN_NUMS_TYPE=1`:
  - Log: `wjeagerqwen30b-a3b-with_draft_breakdown_20260528134438_elastic.txt`.
  - Result: `rollout_output_time_s=308.188`.
  - Conclusion: small improvement vs 311.861, still far from 210s.

## Code Changes

### 2026-05-28: Native/common mode=3 optional remap verification

Intent:

- Remove a native/common hot-path synchronization that custom mode=3 does not
  perform in its fused-experts path.
- Current native/common mode=3 computes `dispatch_topk_ids` and then checks
  `torch.any(dispatch_topk_ids < 0)`. The `if torch.any(...)` condition can
  synchronize the NPU stream every MoE layer.
- The old/custom mode=3 timing shows `remap_wall_ms` around 0.023 ms; native
  performance-like timing with `TIMING_SYNC=0` showed `remap_wall_ms` p50
  around 4.357 ms and large outliers.

Planned files:

- `vllm_ascend/ops/common_fused_moe.py`

Planned change:

- Add a debug-only helper controlled by
  `VLLM_ASCEND_MODE3_VERIFY_REMAP=1`.
- In native/common mode=3 hot paths, skip the `torch.any` negative-id check by
  default while preserving the ability to re-enable it for diagnostics.

Applied change:

- Added `_env_flag` and `_verify_mode3_remapped_ids`.
- Replaced unconditional negative-id checks after `dispatch_log2phy[...]` in:
  - mode2 native single-dispatch
  - mode3 native fused-experts
  - mode3 native single-dispatch
- Default behavior now avoids the NPU-stream sync; diagnostics can restore it
  with `VLLM_ASCEND_MODE3_VERIFY_REMAP=1`.

Validation plan:

- `python3 -m py_compile vllm_ascend/ops/common_fused_moe.py`
- Run threshold-limited native/common mode=3 with timing disabled:
  `REGISTER_CUSTOM_MODELS=0`, mode=3, floor=8, graph=0, task_queue=2.

Validation:

- `python3 -m py_compile vllm_ascend/ops/common_fused_moe.py` passed.

Next run:

- Native/common mode=3, floor=8, timing disabled, `TASK_QUEUE_ENABLE=2`,
  `VLLM_ENABLE_GRAPH_MODE=0`, default allocator.
- Goal: verify whether removing the unconditional remap check improves native
  perf from the current 300-312s band toward current custom (~288s).

Result:

- Log: `wjeagerqwen30b-a3b-with_draft_breakdown_20260528153733_elastic.txt`.
- Exit: `exit_code=0`.
- `rollout_output_time_s=309.809762`, `timing_s/gen=309.8062064209953`.
- Shrink RPC summary: n=16, min 3999.27 ms, max 22890.72 ms,
  avg 13781.61 ms.
- Runtime fingerprint counts:
  - `runtime_num_experts=64`: 16 lines.
  - `runtime_num_experts=16`: 15 lines.
  - `selected=MoECommType.MC2`: 8 lines.

Conclusion:

- Removing the hot-path remap verification is not a material speed fix.
- Keep the debug gate because it avoids an unnecessary sync in normal native
  mode=3, but do not spend more iterations on remap verification unless new
  evidence appears.
- Main 210s gap remains shared with current custom:
  - current custom: `rollout_output_time_s=288.297638`, shrink max 21698.5 ms.
  - current native: `rollout_output_time_s=309.809762`, shrink max 22890.72 ms.
  - old reference: `rollout_output_time_s=212.526065`, shrink max 12565.62 ms.
- Next investigation should compare current shared `Mode3DoubleBufferManager`
  and post-shrink mode3 setup with the old `llm_rl/qwen3` reference, not keep
  repeating allocator/remap-only runs.

### 2026-05-28: Make key runtime env values overrideable

Intent:

- Compare current mode=3 native/common against the old 210s-like runtime
  environment without permanently changing the default script behavior.
- First candidate is allocator/runtime drift because current custom and current
  native both show slow mode=3 prefetch, while old reference had much faster
  `prefetch_npu_dev_ms` and `prefetch_cpu_dev_ms`.

Planned files:

- `internal/wj_train_grpo_qwen30b_a3b_16die_true_weight_eager.sh`

Planned change:

- Keep current defaults, but allow external overrides for:
  - `PYTORCH_NPU_ALLOC_CONF`
  - `HCCL_ASYNC_ERROR_HANDLING`

Validation plan:

- `bash -n internal/wj_train_grpo_qwen30b_a3b_16die_true_weight_eager.sh`
- Run a threshold-limited native/common mode=3 timing sample with the old
  allocator config:
  `PYTORCH_NPU_ALLOC_CONF=garbage_collection_threshold:0.6,max_split_size_mb:24`.

Applied change:

- `PYTORCH_NPU_ALLOC_CONF` now uses
  `${PYTORCH_NPU_ALLOC_CONF:-"expandable_segments:True"}`.
- `HCCL_ASYNC_ERROR_HANDLING` now uses
  `${HCCL_ASYNC_ERROR_HANDLING:-1}`.
- Other existing dirty script changes were left untouched.

Validation:

- `bash -n internal/wj_train_grpo_qwen30b_a3b_16die_true_weight_eager.sh`
  passed.

Next run:

- Native/common mode=3, floor=8, timing enabled, `TASK_QUEUE_ENABLE=2`,
  `VLLM_ENABLE_GRAPH_MODE=0`.
- Only environment hypothesis changed from the current baseline:
  `PYTORCH_NPU_ALLOC_CONF=garbage_collection_threshold:0.6,max_split_size_mb:24`.
- Goal: check whether `prefetch_npu_dev_ms` drops from current ~6.5-7.0 ms
  toward the old reference ~0.15 ms and whether total prefetch approaches
  ~2 ms.

Result:

- Log: `wjeagerqwen30b-a3b-with_draft_breakdown_20260528152021_elastic.txt`.
- Exit: `exit_code=0`.
- `rollout_output_time_s=299.997961`, `timing_s/gen=299.99451052211225`.
- Because this run used `VLLM_ASCEND_MODE3_TIMING_SYNC=0`, device event
  fields such as `prefetch_npu_dev_ms` are intentionally `-1` and cannot be
  used for this run.
- Timing summary from 384 native/common mode=3 lines:
  - `fused_wall_ms`: mean 0.388, p50 0.344, p90 0.533.
  - `current_compute_wall_ms`: mean 0.391, p50 0.346, p90 0.537.
  - `remap_wall_ms`: mean 21.323, p50 4.357, p90 4.848, p99 763.424.
  - `prefetch_submit_us`: p50 546.050, p90 740.450, p99 196330.000.
  - `submit_npu_us`: p50 207.400, p90 297.400.
  - `submit_cpu_direct_async_us`: p50 135.600, p90 178.300.

Conclusion:

- Old allocator alone does not restore the 210s behavior.
- It may reduce some host-side submit costs, but total rollout is still about
  300s.
- New strongest native-specific gap: native/common `remap_wall_ms` is about
  4.3 ms p50, while current custom timing control had `remap_wall_ms` about
  0.023 ms p50. Investigate native/common output remap next.
- Do not keep repeating allocator-only runs without a new variable.

### 2026-05-28: Native/common mode=3 timing instrumentation

Files:

- `vllm_ascend/ops/common_fused_moe.py`

Change:

- Added native/common timing logs for `_execute_mode3_fused_experts_hybrid`.
- The log prefix is `Native common Mode3 timing fused-experts`.
- The fields mirror the custom path timing closely:
  - bind wait and ready wait
  - prefetch status and submit breakdown
  - NPU/CPU prefetch device timing
  - compute/fused wall and device timing
  - remap timing
  - dispatch expert count and `expert_token_nums_type`
- Timing only runs when `VLLM_ASCEND_MODE3_TIMING_LOG=1`; regular perf runs
  should keep timing disabled.

Validation:

- `python3 -m py_compile vllm_ascend/ops/common_fused_moe.py` passed.
- Timing sample run completed:
  - Log: `wjeagerqwen30b-a3b-with_draft_breakdown_20260528144853_elastic.txt`.
  - Command intent: native/common mode=3, floor=8, timing enabled.
  - Caveat: environment inherited `TASK_QUEUE_ENABLE=1`; use this run only
    for timing diagnosis, not final speed comparison.
  - Exit: `exit_code=0`.
  - Result: `rollout_output_time_s=325.258811`,
    `timing_s/gen=325.2553398311138`.
  - Analyze: `tail_after_shrink8_done_s=240.501`,
    `shrink_to_8_rpc_max_ms=25425.910`.
  - Timing summary from 384 lines:
    - `fused_wall_ms`: mean 0.570, p50 0.528, p90 0.623, max 2.605.
    - `current_compute_wall_ms`: mean 0.635, p50 0.591, p90 0.703.
    - `prefetch_dev_ms`: mean 11.558, p50 6.985, p90 7.360,
      p99 221.052, max 231.169.
    - `prefetch_npu_dev_ms`: mean 7.016, p50 6.867, p90 7.246.
    - `prefetch_cpu_dev_ms`: mean 4.790, p50 4.833, p90 5.243.
    - `prefetch_submit_us`: p50 673.100, p90 840.700, p99 216238.900.
  - Conclusion:
    - Native fused compute is not the main gap; `fused_wall_ms` p50 is close
      to old custom reference p50 0.582 ms.
    - The obvious gap is prefetch: native/common timing p50 about 6.985 ms vs
      old custom reference p50 about 2.108 ms.
    - There are large prefetch submit/NPU outliers, so investigate double-buffer
      prefetch/slot population before changing graph/task queue again.

### 2026-05-28: Current-branch custom timing control

Files:

- No code changes.

Run:

- Log: `wjeagerqwen30b-a3b-with_draft_breakdown_20260528150433_elastic.txt`.
- Env was explicit:
  - `VLLM_ASCEND_REGISTER_CUSTOM_MODELS=1`
  - `TASK_QUEUE_ENABLE=2`
  - `VLLM_ENABLE_GRAPH_MODE=0`
  - mode=3, floor=8, timing enabled
- Exit: `exit_code=0`.

Results:

- `rollout_output_time_s=301.295211`, `timing_s/gen=301.2916855928488`.
- Analyze: `tail_after_shrink8_done_s=233.542`,
  `shrink_to_8_rpc_max_ms=23082.940`.
- Timing summary from 384 lines:
  - `prefetch_dev_ms`: mean 11.796, p50 6.652, p90 8.747,
    p99 226.664, max 243.403.
  - `prefetch_npu_dev_ms`: mean 6.956, p50 6.551, p90 8.577.
  - `prefetch_cpu_dev_ms`: mean 4.873, p50 4.494, p90 6.580.
  - `fused_wall_ms`: mean 1.286, p50 0.808, p90 1.748.
  - `remap_wall_ms`: mean 0.023, p50 0.022.

Conclusion:

- Slow prefetch is not native/common-specific. Current custom uses the same
  current `Mode3DoubleBufferManager` and also has `prefetch_dev_ms` p50 around
  6.6 ms.
- The old 212s reference had `prefetch_dev_ms` p50 around 2.108 ms. Therefore
  the 210s gap should be investigated as shared manager/environment drift first.
- Native fused compute is not worse than custom in this branch; native
  `fused_wall_ms` p50 was 0.528 ms vs current custom p50 0.808 ms in timing
  runs.

## Do Not Repeat Without New Evidence

- Do not use `VLLM_ENABLE_GRAPH_MODE=1` as a native speed fix. It produced
  `325.399776s`, slower than graph=0/task_queue=2.
- Do not use the non-eager rollout override. Prior run failed with:
  `graph_batch_sizes_init is valid only when Torchair graph mode is enabled`.
- Do not claim native is 100s slower than custom from the old 212s log alone.
  Current custom in this branch is 288s, so most of the 212s gap is likely
  branch/env/code drift shared by custom and native.
- Do not repeat allocator-only testing. Old allocator with otherwise current
  native env produced about `299.998s`, still far from 210s.

### 2026-05-28: Old/current mode3 prefetch comparison

Files:

- No code changes.

Observation:

- The old `mode3-record.txt` 212s reference and the current timing logs use the
  same mode3 direct CPU slot path:
  - `prefetch_cpu_path=direct_async`
  - `source_from_npu=8`
  - `source_from_cpu=8`
  - `layer_local_buffer=0`
  - `dispatch_num_experts=128`
- Old reference timing:
  - `prefetch_npu_dev_ms` around 0.14-0.18 ms on normal layers.
  - `prefetch_cpu_dev_ms` around 1.4-2.7 ms.
  - `remap_wall_ms` around 0.022-0.027 ms.
  - `bind_wait_mode=device_event` after the first layer.
- Current branch custom timing control:
  - `prefetch_npu_dev_ms` around 6.3-8.8 ms on normal layers.
  - `prefetch_cpu_dev_ms` around 2.3-6.9 ms.
  - `remap_wall_ms` still around 0.02-0.03 ms.
- Current branch native/common timing:
  - `prefetch_npu_dev_ms` around 6.4-7.1 ms on normal layers.
  - `prefetch_cpu_dev_ms` around 4.3-4.9 ms.
  - Native remap is slower than custom in the timing run, but the dominant
    old/current difference is shared NPU prefetch copy latency.

Conclusion:

- The 210s reference is not explained by a custom-vs-native mode3 algorithm
  difference alone. The shared current mode3 prefetch path is much slower than
  the old reference, even when `REGISTER_CUSTOM_MODELS=1`.
- The code comparison of `Mode3DoubleBufferManager` shows the core copy path is
  still broadly the old direct CPU slot path; the strongest remaining
  hypothesis is a runtime/env or imported-stack difference that changes NPU
  copy scheduling.

Next run:

- Native/common, threshold-limited mode=3, with an old-reference-like runtime
  environment:
  - `PYTORCH_NPU_ALLOC_CONF=garbage_collection_threshold:0.6,max_split_size_mb:24`
  - `TASK_QUEUE_ENABLE=1`
  - `VLLM_ENABLE_GRAPH_MODE=1`
  - `HCCL_ASYNC_ERROR_HANDLING=0`
- Timing enabled for diagnosis.
- Goal: check whether `prefetch_npu_dev_ms` falls back toward the old
  0.15 ms band. If not, stop repeating runtime-env toggles and inspect the
  model/weight initialization path that produces `layer.w13_weight`.

Result:

- Log: `wjeagerqwen30b-a3b-with_draft_breakdown_20260528155411_elastic.txt`.
- Exit: `exit_code=0`.
- `rollout_output_time_s=312.418472`, `timing_s/gen=312.4150642398745`.
- Timing summary from 384 native/common mode3 lines:
  - `prefetch_dev_ms`: mean 13.636, p50 6.966, p90 7.805,
    p99 308.481, max 385.531.
  - `prefetch_npu_dev_ms`: mean 7.594, p50 6.836, p90 7.559,
    p99 10.855, max 102.504.
  - `prefetch_cpu_dev_ms`: mean 4.985, p50 4.798, p90 5.700.
  - `fused_wall_ms`: mean 0.586, p50 0.541, p90 0.642.
  - `remap_wall_ms`: mean 0.073, p50 0.064, p90 0.077.
  - `bind_wait_us`: p50 90.5 us, p90 109.1 us.
- Runtime fingerprint counts:
  - `runtime_num_experts=128`: 64 lines.
  - `runtime_num_experts=64`: 16 lines.
  - `runtime_num_experts=16`: 15 lines.
  - `selected=MoECommType.MC2`: 8 lines.

Conclusion:

- Old-reference-like env did not restore the old 0.15 ms NPU prefetch behavior.
- `TASK_QUEUE_ENABLE=1`, `VLLM_ENABLE_GRAPH_MODE=1`, old allocator, and
  `HCCL_ASYNC_ERROR_HANDLING=0` together are not a solution for the current
  branch native/common mode3 path.
- Do not repeat old-env combo testing without a new code change. Continue with
  code/layout differences and native-specific gap versus current custom.

## Next Checks

- Parse the current native timing sample and compare:
  - native `fused_wall_ms` vs old custom `fused_wall_ms`
  - native `prefetch_cpu_dev_ms` and `prefetch_dev_ms` vs old custom
  - bind/ready wait frequency
- Run a current-branch custom timing sample with explicit env. This is needed
  to decide whether slow prefetch is native/common-specific or comes from the
  shared current `Mode3DoubleBufferManager`.
- Run final perf comparisons with explicit env:
  - `TASK_QUEUE_ENABLE=2`
  - `VLLM_ENABLE_GRAPH_MODE=0`
  - `VLLM_ASCEND_MODE3_TIMING_LOG=0`
  - `VLLM_ASCEND_MODE3_TIMING_SYNC=0`

## Iteration Discipline

- Every code change and every performance run for native/common mode3 must be
  recorded in this file.
- Before changing code, record the hypothesis, touched files, and expected
  signal.
- After each run, record the log path, exit status, rollout timing, and whether
  the result should be repeated.
- Failed or non-improving experiments should be kept in the "Do Not Repeat"
  list unless new evidence changes the hypothesis.

### 2026-05-28: Native/common log2phy dispatch A/B

Files planned:

- `vllm_ascend/ops/common_fused_moe.py`

Hypothesis:

- Current native/common mode3 fused path materializes
  `dispatch_log2phy[logical_topk_ids]` before calling `fused_experts`.
- Current custom mode3 routes through the shared wave helper and passes
  `log2phy` into the communication path. The current branch custom run is still
  slow versus the old 212s reference, but it is faster than native/common
  (`288s` vs about `310s`).
- Add an env-gated native/common A/B path that passes logical ids plus `log2phy`
  to `fused_experts`, leaving the default behavior unchanged.

Expected signal:

- If native/common loses time mostly in the explicit pre-remap path, enabling
  `VLLM_ASCEND_MODE3_NATIVE_LOG2PHY_DISPATCH=1` should move native/common
  closer to current custom.
- If runtime is unchanged, stop repeating remap-dispatch-only experiments and
  continue with the shared prefetch/layout gap.

Code change:

- Added env gate `VLLM_ASCEND_MODE3_NATIVE_LOG2PHY_DISPATCH`.
- Default remains old native/common behavior:
  `topk_ids=dispatch_log2phy[logical_topk_ids]`, `log2phy=None`.
- When enabled, native/common passes `topk_ids=logical_topk_ids` and
  `log2phy=dispatch_log2phy` to `fused_experts`.
- Timing logs now include `native_log2phy_dispatch`.

Static validation:

- `python3 -m py_compile vllm_ascend/ops/common_fused_moe.py`: passed.

Run planned:

- Threshold-limited native/common mode=3.
- Enable only `VLLM_ASCEND_MODE3_NATIVE_LOG2PHY_DISPATCH=1` for the A/B.
- Keep timing disabled for the first performance signal.

Invalid run:

- Log: `/workspace/cann-recipes-train/wjeagerqwen30b-a3b-with_draft_breakdown_20260528161312_elastic.txt`.
- Exit: `exit_code=1`.
- Reason: launched from repository root, so the script set `HOME=$(pwd)` to
  `/workspace/cann-recipes-train`; Python then failed with
  `ModuleNotFoundError: No module named 'verl'`.
- This is not a mode3 result. Re-run from
  `llm_rl/ref_wj_qwen3/qwen30b_tpu_verl`.

Result:

- Log:
  `wjeagerqwen30b-a3b-with_draft_breakdown_20260528161353_elastic.txt`.
- Exit: `exit_code=0`.
- `rollout_output_time_s=315.522487`.
- `timing_s/gen=315.5190747869201`.
- Shrink RPC summary: n=16, min=4698.22 ms, max=29617.86 ms,
  avg=17567.21 ms.
- Runtime fingerprint counts:
  - `runtime_num_experts=128`: 64.
  - `runtime_num_experts=64`: 16.
  - `runtime_num_experts=16`: 15.
  - `selected=MoECommType.MC2`: 8.
- Timing log was disabled, so there are no native/common mode3 timing lines.
- Torch Dynamo emitted 16 compile tracebacks, but the run completed
  successfully.

Conclusion:

- Passing `log2phy` into native/common `fused_experts` did not improve
  performance; it was slower than the prior native/common timing-off result
  (`309.809762s`) and much slower than the current-branch custom control
  (`288.297638s`).
- Keep `VLLM_ASCEND_MODE3_NATIVE_LOG2PHY_DISPATCH` as an off-by-default
  diagnostic switch only.
- Do not repeat remap/log2phy-dispatch-only experiments unless a new semantic
  mismatch is found.

### 2026-05-28: Shared manager prefetch/copy investigation

Files inspected:

- `vllm_ascend/ops/fused_moe.py`
- `llm_rl/qwen3/vllm_ascend/ops/fused_moe_legacy.py`

Observations:

- The old 212s reference and current slow runs both submit NPU resident expert
  copies quickly on the host, but device time differs sharply:
  - Old reference: `prefetch_npu_dev_ms` around `0.14-0.18 ms`.
  - Current native/custom: `prefetch_npu_dev_ms` around `6-8 ms`.
- This points away from Python submit overhead and toward source/target weight
  layout, formatted-buffer copy behavior, or stream dependencies.
- Current manager differs from the legacy file in one notable slot reuse
  semantic: `prepare_slot()` now returns immediately when `_slot_matches()` is
  true, even if `slot.inflight_prefetch` is still true. The legacy file only
  reused a matching slot when `not slot.inflight_prefetch`.
- Do not patch the in-flight condition yet; first test the more direct copy
  hypothesis below.

Run planned:

- Threshold-limited native/common mode=3.
- Set `VLLM_ASCEND_MODE3_BULK_NPU_COPY=0` to compare per-row resident NPU
  copies against the current bulk slice copy.
- Enable mode3 timing for the first diagnostic run so
  `prefetch_npu_dev_ms` can be compared directly.

Result:

- Log:
  `wjeagerqwen30b-a3b-with_draft_breakdown_20260528162851_elastic.txt`.
- Exit: `exit_code=1`.
- The run was manually interrupted after the early timing samples showed a
  clear regression; it was not allowed to burn the full 25 minute timeout.
- No `rollout_output_time_s` was produced.
- Timing summary before interruption:
  - `prefetch_npu_dev_ms`: n=376, mean=48.447 ms, p50=48.114 ms,
    p90=49.810 ms, max=55.742 ms.
  - `prefetch_cpu_dev_ms`: n=376, mean=5.275 ms, p50=4.940 ms,
    p90=6.556 ms, max=12.585 ms.
  - `prefetch_dev_ms`: n=376, mean=53.123 ms, p50=48.242 ms,
    p90=49.962 ms, max=268.393 ms.
  - `fused_wall_ms`: n=384, mean=0.382 ms, p50=0.362 ms,
    p90=0.473 ms, max=0.782 ms.
- Shrink RPC summary: n=16, min=4186.78 ms, max=21645.40 ms,
  avg=13271.95 ms.

Conclusion:

- Per-row resident NPU copy is much worse than the current bulk slice copy.
- The current 6-8 ms `prefetch_npu_dev_ms` gap is not caused by the bulk-copy
  optimization itself.
- Keep `VLLM_ASCEND_MODE3_BULK_NPU_COPY=1` for future runs.
- Do not repeat `VLLM_ASCEND_MODE3_BULK_NPU_COPY=0` unless the copy layout is
  changed substantially.

### 2026-05-28: Restore legacy in-flight slot reuse guard

Files planned:

- `vllm_ascend/ops/fused_moe.py`

Hypothesis:

- The current shared mode3 double-buffer manager returns immediately from
  `prepare_slot()` when `_slot_matches()` is true, even if the slot still has
  `slot.inflight_prefetch=True`.
- The old 212s legacy implementation only reused a matching slot when
  `not slot.inflight_prefetch`.
- Restoring this guard may repair a scheduling/ready-event semantic mismatch in
  the shared manager. This is more plausible than remap/log2phy-only changes
  because it is a direct difference between the current branch and the old
  custom run that reached about 212s.

Expected signal:

- With bulk NPU copy enabled and timing enabled, `prefetch_npu_dev_ms` should
  move down from the current 6-8 ms range if this was causing stale or
  incorrectly reused in-flight prefetch state.
- If runtime and timing do not improve, keep the change only if it is
  semantically safer; otherwise revert this specific experiment before trying
  another direction.

Code change:

- Changed `Mode3DoubleBufferManager.prepare_slot()` from:
  `if self._slot_matches(slot, layer): return slot`
  to:
  `if self._slot_matches(slot, layer) and not slot.inflight_prefetch: return slot`.
- This restores the legacy slot-reuse guard exactly at the identified semantic
  difference.

Static validation:

- `python3 -m py_compile vllm_ascend/ops/fused_moe.py`: passed.

Run planned:

- Threshold-limited native/common mode=3.
- Keep `VLLM_ASCEND_MODE3_BULK_NPU_COPY=1`.
- Keep `VLLM_ASCEND_MODE3_NATIVE_LOG2PHY_DISPATCH=0`.
- Enable timing for the first diagnostic signal.

Result:

- Log:
  `wjeagerqwen30b-a3b-with_draft_breakdown_20260528164706_elastic.txt`.
- Exit: `exit_code=0`.
- `rollout_output_time_s=307.334235`.
- `timing_s/gen=307.3307797680609`.
- Timing summary:
  - `prefetch_npu_dev_ms`: n=376, mean=6.726 ms, p50=6.626 ms,
    p90=7.062 ms, p99=9.010 ms, max=10.160 ms.
  - `prefetch_cpu_dev_ms`: n=376, mean=4.778 ms, p50=4.659 ms,
    p90=5.108 ms, p99=8.933 ms, max=9.872 ms.
  - `prefetch_dev_ms`: n=376, mean=11.426 ms, p50=6.741 ms,
    p90=7.172 ms, p99=227.983 ms, max=230.325 ms.
  - `fused_wall_ms`: n=384, mean=0.346 ms, p50=0.331 ms,
    p90=0.432 ms, max=0.702 ms.
  - `remap_wall_ms`: n=384, mean=0.057 ms, p50=0.053 ms,
    p90=0.070 ms, max=0.154 ms.
- Shrink RPC summary: n=16, min=4015.26 ms, max=22848.50 ms,
  avg=13838.82 ms.
- Restore summary: n=16, min=96.37 ms, max=2620.09 ms,
  avg=1160.15 ms.
- Tail signals:
  - `Elastic shrink delayed for MC2 compatibility`: 7.
  - `single-rank tail blocked`: 7.

Conclusion:

- Restoring the legacy in-flight slot reuse guard made the code semantically
  closer to the 212s reference and gave a small end-to-end improvement versus
  the 309.809762s native/common timing-off baseline, but it did not fix the
  main performance gap.
- `prefetch_npu_dev_ms` remains in the 6-7 ms range instead of the old
  0.14-0.18 ms range, so the dominant slowdown is still in the shared
  resident NPU prefetch/device-copy path or its stream dependencies.
- Keep this guard for now as a safer semantic match, but do not repeat this
  isolated A/B as a speed fix.

### 2026-05-28: Timing-off control after legacy guard

Files planned:

- No code changes.

Hypothesis:

- The in-flight guard run above used `VLLM_ASCEND_MODE3_TIMING_LOG=1` and
  `VLLM_ASCEND_MODE3_TIMING_SYNC=1`.
- The timing samples show that the resident NPU copy remains slow, but the
  complete end-to-end number should also be measured without timing events so
  we do not mistake instrumentation overhead for mode3 behavior.

Expected signal:

- If timing instrumentation is a major source of slowdown, disabling timing
  should move the guarded native/common run substantially below 307s.
- If the run remains near 307-310s, the slow path is real and not just the
  diagnostic event/log machinery.

Run planned:

- Threshold-limited native/common mode=3.
- Keep the legacy in-flight guard.
- Keep `VLLM_ASCEND_MODE3_BULK_NPU_COPY=1`.
- Disable mode3 timing logs and timing sync.

Result:

- Log:
  `wjeagerqwen30b-a3b-with_draft_breakdown_20260528170023_elastic.txt`.
- Exit: `exit_code=0`.
- `rollout_output_time_s=308.989395`.
- `timing_s/gen=308.9859507079236`.
- Mode3 timing lines: 0.
- Shrink RPC summary: n=16, min=4726.02 ms, max=23601.98 ms,
  avg=14376.93 ms.
- Restore summary: n=16, min=94.87 ms, max=3508.21 ms,
  avg=1355.70 ms.
- Runtime fingerprint counts:
  - `runtime_num_experts=128`: 64.
  - `runtime_num_experts=64`: 16.
  - `runtime_num_experts=16`: 15.
  - `selected=MoECommType.MC2`: 8.
- Tail signals:
  - `Elastic shrink delayed for MC2 compatibility`: 7.
  - `single-rank tail blocked`: 7.

Conclusion:

- Disabling mode3 timing/log sync does not restore the 212s behavior; the
  timing-off guarded run remains at about 309s.
- Timing instrumentation can distort local progress bars and sampled device
  timings, but the end-to-end native/common gap is real.
- The current native/common run is still dominated by the same tail behavior:
  MC2 compatibility delays and single-rank tail blocking after shrinking to
  the configured floor.
- Do not repeat timing-off-only controls as an optimization.

### 2026-05-28: Cross-run comparison against the 212s reference

Files inspected:

- `mode3-record.txt`
- `wjeagerqwen30b-a3b-with_draft_breakdown_20260528170023_elastic.txt`
- `wjeagerqwen30b-a3b-with_draft_breakdown_20260528132918_elastic.txt`
- `wjeagerqwen30b-a3b-with_draft_breakdown_20260528150433_elastic.txt`

Observations:

- The old 212s reference and current runs have the same generated workload:
  - `response_length/mean=441.0`
  - `response_length/max=897.0`
  - `response_length/min=257.0`
  - `perf/total_num_tokens=263872`
- Therefore the 212s -> 309s gap is not caused by longer generated responses
  or a different token count.
- Tail mechanism counts are also the same:
  - `Elastic shrink delayed for MC2 compatibility`: 7.
  - `single-rank tail blocked`: 7.
  - `runtime_num_experts=128`: 64.
  - `runtime_num_experts=64`: 16.
  - `runtime_num_experts=16`: 15.
  - `selected=MoECommType.MC2`: 8.
- The difference is in speed, not in the high-level elastic state sequence:
  - Old reference:
    `rollout_output_time_s=212.526065`,
    `timing_per_token_ms/gen=0.9412248851817007`,
    progress durations min/p50/max = 20s/98s/191s,
    shrink RPC avg = 8277.31 ms.
  - Current native/common timing-off:
    `rollout_output_time_s=308.989395`,
    `timing_per_token_ms/gen=1.3684539341868782`,
    progress durations min/p50/max = 28s/143s/277s,
    shrink RPC avg = 14376.93 ms.
  - Current custom timing-off/timing-control examples:
    `288.297638s` and `301.295211s`, so current-branch custom is also much
    slower than the old 212s reference.

Conclusion:

- Do not attribute the whole 212s -> 309s gap to native/common mode3.
- The current branch/env has a shared slowdown versus the old custom reference,
  plus a smaller native/common-vs-current-custom gap.
- Next useful work should separate:
  - shared old-vs-current slowdown: shrink RPC time, resident copy/device
    throughput, and per-token decode throughput;
  - native-specific slowdown: current custom 288s vs current native 309s.

### 2026-05-28: Align local custom with remote-master 212s source

User clarification:

- The 212s mode=3 custom run was produced by the current remote `master`
  branch code.
- Therefore the immediate goal is not to guess optimizations, but to compare
  this working tree's current `ref_wj_qwen3/qwen30b_tpu_verl` source against
  the remote-master/custom source that produced `mode3-record.txt`.

Plan:

- Confirm the remote/master commit and locate the source tree used by the 212s
  run.
- Compare custom mode=3 code first, especially `vllm_ascend/ops/fused_moe.py`,
  worker/engine shrink logic, and the run scripts/env.
- Align the local custom path with the 212s source before moving any changes
  into native/common mode=3.
- Record every code change and every validation run in this file before
  repeating or extending an experiment.

Findings so far:

- `mode3-record.txt` was produced from the `llm_rl/qwen3` tree, not the
  current `ref_wj_qwen3/qwen30b_tpu_verl` tree.
- In `llm_rl/qwen3`, the custom fused MoE path selected by elastic shrink is
  `vllm_ascend/ops/fused_moe_legacy.py`.
- The current `ref_wj_qwen3/qwen30b_tpu_verl` custom path uses the much larger
  `vllm_ascend/ops/fused_moe.py`, with additional mode1 fixes, timing hooks,
  active weight views, slot expert map caching, layer-local-buffer support, and
  remapped-id handling.
- Script/env defaults also differ materially:
  - 212s source used `TASK_QUEUE_ENABLE=1`, `VLLM_ENABLE_GRAPH_MODE=1`,
    `PYTORCH_NPU_ALLOC_CONF=garbage_collection_threshold:0.6,max_split_size_mb:24`,
    and long HCCL timeouts.
  - Current ref_wj script defaults to `TASK_QUEUE_ENABLE=2`,
    `VLLM_ENABLE_GRAPH_MODE=0`, `PYTORCH_NPU_ALLOC_CONF=expandable_segments:True`,
    and shorter HCCL timeouts.

Next code change:

- Add an explicit env-gated custom compatibility path in the current ref_wj
  tree so that `VLLM_ASCEND_CUSTOM_FUSED_MOE_IMPL=legacy_master` resolves
  `AscendFusedMoE` to a copied `llm_rl/qwen3` legacy implementation.
- Keep default behavior unchanged.
- Validate this path before using the aligned custom implementation as the
  template for native/common mode=3.

Implemented:

- Copied `llm_rl/qwen3/vllm_ascend/ops/fused_moe_legacy.py` to
  `llm_rl/ref_wj_qwen3/qwen30b_tpu_verl/vllm_ascend/ops/fused_moe_master_legacy.py`.
- Added `VLLM_ASCEND_CUSTOM_FUSED_MOE_IMPL=legacy_master` aliasing at the end
  of the current ref_wj `vllm_ascend/ops/fused_moe.py`.
- `py_compile` passes for both files.

Static import issue:

- The copied legacy file imports `get_pcp_group`, but the ref_wj vLLM
  `vllm.distributed.parallel_state` does not expose that symbol.
- Patch the copied compatibility file only: import `get_pcp_group`
  opportunistically and return `None` when absent. This keeps the copied
  source close to the 212s implementation while allowing ref_wj import.

Implemented follow-up:

- Patched `fused_moe_master_legacy.py` so `get_pcp_group` is optional.
- Re-ran `py_compile`: pass.
- Import check with
  `VLLM_ASCEND_CUSTOM_FUSED_MOE_IMPL=legacy_master` shows:
  - `AscendFusedMoE.__module__ = vllm_ascend.ops.fused_moe_master_legacy`
  - `AscendUnquantizedFusedMoEMethod.__module__ = vllm_ascend.ops.fused_moe_master_legacy`

Validation run planned:

- Run the ref_wj script with custom model registration enabled and the copied
  legacy fused MoE implementation selected:
  - `VLLM_ASCEND_REGISTER_CUSTOM_MODELS=1`
  - `VLLM_ASCEND_CUSTOM_FUSED_MOE_IMPL=legacy_master`
  - `VLLM_ASCEND_ELASTIC_EXECUTION_MODE=3`
  - `VLLM_ASCEND_ELASTIC_MIN_COMPUTE_GROUP_SIZE=8`
  - `VLLM_ASCEND_ELASTIC_HYBRID_RESIDENT_EXPERT_SLOTS=8`
- Align runtime env toward the 212s reference:
  - `TASK_QUEUE_ENABLE=1`
  - `VLLM_ENABLE_GRAPH_MODE=1`
  - `PYTORCH_NPU_ALLOC_CONF=garbage_collection_threshold:0.6,max_split_size_mb:24`
  - `HCCL_ASYNC_ERROR_HANDLING=0`
  - `HCCL_CONNECT_TIMEOUT=7200`
  - `HCCL_EXEC_TIMEOUT=7200`

Validation run result:

- Log: `wjeagerqwen30b-a3b-with_draft_breakdown_20260528210455_elastic.txt`
- Result: failed during rollout model initialization.
- Positive signal: all 16 workers logged
  `Using legacy_master custom fused MoE implementation`.
- Failure:
  `TypeError: FusedMoE.__init__() got an unexpected keyword argument 'pcp_size'`.
- Cause: the copied 212s/qwen3 legacy class targets a newer/different vLLM
  base `FusedMoE` API that accepts `pcp_size` and related fields; the ref_wj
  base `FusedMoE` does not.

Next code change:

- Patch only `fused_moe_master_legacy.py` compatibility glue:
  remove unsupported base-class kwargs from `super().__init__` and construct
  `FusedMoEParallelConfig` without `pcp_size_`.
- Do not change mode=3 transfer/buffer logic.

Implemented:

- Removed unsupported `pcp_size`, `is_act_and_mul`, `expert_mapping`,
  `n_shared_experts`, `routing_method_type`, and `router_logits_dtype` kwargs
  from the copied legacy file's base `FusedMoE.__init__` call.
- Removed `pcp_size_` from the copied legacy file's
  `FusedMoEParallelConfig.make` call.
- `py_compile` passes.
- Import check still resolves `AscendFusedMoE` to
  `vllm_ascend.ops.fused_moe_master_legacy`.

Second validation result:

- Log: `wjeagerqwen30b-a3b-with_draft_breakdown_20260528210920_elastic.txt`
- Result: failed during rollout model initialization after progressing past
  the previous base-API keyword issue.
- Failure:
  `KeyError: "attribute 'expert_map' already exists"`.
- Cause: the copied qwen3 legacy class defines an `expert_map` property proxy
  for a vLLM API where `expert_map` is backed by `_expert_map`; the ref_wj base
  `FusedMoE.__init__` directly registers an `expert_map` buffer, so the class
  property collides with base buffer registration.

Next code change:

- Remove the copied compatibility file's `expert_map` property proxy.
- Keep using ref_wj's direct `expert_map` buffer semantics.

Implemented:

- Removed `AscendFusedMoE.expert_map` property proxy from
  `fused_moe_master_legacy.py`.
- `py_compile` passes.
- Import check:
  - `AscendFusedMoE.__module__ = vllm_ascend.ops.fused_moe_master_legacy`
  - `hasattr(AscendFusedMoE, "expert_map") = False`

Correction after user clarification:

- The desired scope is strictly the local
  `llm_rl/ref_wj_qwen3/qwen30b_tpu_verl` implementation:
  - local custom: `vllm_ascend/ops/fused_moe.py`
  - local native/common: `vllm_ascend/ops/common_fused_moe.py`
- The `llm_rl/qwen3` tree and `mode3-record.txt` should be used as reference
  evidence only, not as an implementation to copy into the local tree.
- The `legacy_master` compatibility branch was therefore a wrong direction.

Cleanup:

- Stopped the in-flight `legacy_master` validation run:
  `wjeagerqwen30b-a3b-with_draft_breakdown_20260528211321_elastic.txt`
  ended with `exit_code=134` due to manual termination.
- Removed the `VLLM_ASCEND_CUSTOM_FUSED_MOE_IMPL` alias block from local
  `vllm_ascend/ops/fused_moe.py`.
- Deleted the copied `vllm_ascend/ops/fused_moe_master_legacy.py`.

Do not repeat:

- Do not add a copied qwen3 legacy implementation into the ref_wj tree.
- Continue by comparing the local current custom/native mode=3 code against
  the behavior observed in `mode3-record.txt`, then implement the missing
  native/common pieces locally.

### 2026-05-28: Correct repository scope

User correction:

- The implementation target is the local nested repository:
  `llm_rl/ref_wj_qwen3/qwen30b_tpu_verl`.
- The 212s reference should be compared against this repository's own
  `origin/master`, not the outer repository and not `llm_rl/qwen3`.

Nested repository state:

- Workdir: `llm_rl/ref_wj_qwen3/qwen30b_tpu_verl`
- Current branch: `v11_newversion`
- Current HEAD: `af3889d Implement custom mode1 zero-headroom parity`
- Upstream: `origin/v11_newversion`
- Reference branch: `origin/master`
- `origin/master`: `3c28439 Add elastic sidecar utilization profiling`
- Relevant history on `origin/master` includes:
  `5889609 Optimize mode3 fused experts runtime path`

Current local modified files in scope:

- `internal/wj_train_grpo_qwen30b_a3b_16die_true_weight_eager.sh`
- `vllm/v1/engine/llm_engine.py`
- `vllm_ascend/ops/common_fused_moe.py`
- `vllm_ascend/ops/fused_moe.py`
- `MODE3_NATIVE_ITERATION_LOG.md`

Comparison summary against nested `origin/master`:

- `vllm_ascend/ops/fused_moe.py`: about 634 changed lines.
- `vllm_ascend/ops/common_fused_moe.py`: about 3069 changed lines.
- The next comparison must first isolate custom mode=3 behavior in local
  `fused_moe.py` versus `origin/master:vllm_ascend/ops/fused_moe.py`.

Scope going forward:

- First align local custom mode=3 to nested `origin/master`.
- Then implement/adjust local native/common mode=3 in
  `vllm_ascend/ops/common_fused_moe.py` using the aligned local custom path as
  the reference.

### 2026-05-28: Local-only custom/native mode=3 restart

Clarification applied:

- All implementation and validation must happen under
  `llm_rl/ref_wj_qwen3/qwen30b_tpu_verl`.
- The `mode3-record.txt` 212s run is only a performance reference. It is not
  an instruction to copy code from the sibling `llm_rl/qwen3` tree.

Function-level comparison against nested `origin/master`:

- Local custom `Mode3DoubleBufferManager` is effectively aligned with
  `origin/master` for the mode=3 double-buffer hot path:
  - `__init__`: same
  - `_populate_slot`: same
  - `bind_current_layer`: same
  - `prefetch_next_layer`: same
- The only local custom difference in `prepare_slot` is the additional guard
  that avoids returning a matching slot while `inflight_prefetch=True`.
- The custom mode=3 execution methods are aligned with `origin/master`:
  - `_execute_mode3_fused_experts_hybrid`: same
  - `_execute_mode3_single_dispatch_hybrid`: same
  - `_execute_mode3_single_rank_allgather_hybrid`: same

Implication:

- The current custom mode=3 performance gap is unlikely to be caused by a
  missing double-buffer implementation in `vllm_ascend/ops/fused_moe.py`.
- Next validation should run the local custom path with the same mode=3/floor=8
  constraints and comparable data/model paths before changing custom code
  again.

Change:

- Made `DISTCP_PATH` overrideable in
  `internal/wj_train_grpo_qwen30b_a3b_16die_true_weight_eager.sh` so the local
  script can reproduce reference-style paths without editing the script each
  time.

Do not repeat:

- Do not copy or import a sibling-tree legacy fused MoE implementation.
- Do not repeat the `VLLM_ASCEND_MODE3_NATIVE_LOG2PHY_DISPATCH=1` experiment
  unless there is a new correctness reason; it was slower in previous native
  validation.
- Do not repeat `VLLM_ASCEND_MODE3_BULK_NPU_COPY=0` as a performance attempt;
  it was only diagnostic and was much slower.

### 2026-05-28: Local custom mode=3 base initializer A/B

Files planned:

- `vllm_ascend/ops/fused_moe.py`

Hypothesis:

- Local `Mode3DoubleBufferManager` and mode=3 execution methods are already
  aligned with nested `origin/master`.
- The largest local custom difference is `AscendFusedMoE.__init__`: current
  branch bypasses `FusedMoE.__init__` for mode1 zero-headroom work, while
  `origin/master` calls the base initializer first.
- That base initializer may affect the NPU weight/buffer layout or allocator
  state used by mode=3 resident expert copies. The symptom matches this:
  local custom mode=3 has `prefetch_npu_dev_ms` p50 about `6.69 ms`, while
  the 212s reference has p50 about `0.158 ms`.

Planned change:

- Add an env-gated local custom branch:
  `VLLM_ASCEND_CUSTOM_MODE3_USE_BASE_INIT=1` by default.
- When `elastic_execution_mode == 3`, call `FusedMoE.__init__` first, matching
  nested `origin/master`.
- Keep the current manual initializer for other modes, especially mode=1, so
  the zero-headroom parity fix is not reverted.

Expected signal:

- Custom mode=3 should still run successfully.
- If the initializer/layout hypothesis is correct, local custom
  `prefetch_npu_dev_ms` should move toward the reference `0.15 ms` band and
  rollout time should move substantially below the current `327.965s` timing
  run.

Applied change:

- `AscendFusedMoE.__init__` now computes
  `elastic_execution_mode` before initialization.
- For custom mode=3, `VLLM_ASCEND_CUSTOM_MODE3_USE_BASE_INIT=1` causes the
  class to call `FusedMoE.__init__`, matching nested `origin/master`.
- For other modes, including mode=1, the current manual initializer remains.
- Added a one-time layer0 log:
  `Custom AscendFusedMoE init path: mode=... use_base_init=...`.

Static validation:

- `python3 -m py_compile vllm_ascend/ops/fused_moe.py`: passed.

Run planned:

- Local custom mode=3, floor=8, timing enabled.
- Explicit env:
  `VLLM_ASCEND_REGISTER_CUSTOM_MODELS=1`,
  `VLLM_ASCEND_CUSTOM_MODE3_USE_BASE_INIT=1`.

Result:

- Log:
  `wjeagerqwen30b-a3b-with_draft_breakdown_20260528215856_elastic.txt`.
- Exit: `exit_code=0`.
- `rollout_output_time_s=316.394690`.
- `timing_s/gen=316.3910218770616`.
- `timing_per_token_ms/gen=1.4012499197361359`.
- `perf/total_num_tokens=263872`.
- Layer0 logs confirmed the local custom mode=3 base initializer path:
  `Custom AscendFusedMoE init path: mode=3 use_base_init=1`.
- Timing summary from the mode3 fused-experts samples:
  - `prefetch_npu_dev_ms`: normal layers remain around 6-8 ms.
  - `prefetch_cpu_dev_ms`: normal layers remain around 2-6 ms.
  - `prefetch_dev_ms`: normal layers remain around 6-8 ms, with first-layer
    outliers.
  - `submit_npu_us`: normal layers are still mostly a few hundred us, so the
    slowdown is not primarily Python-side submit overhead.

Conclusion:

- Calling the local base `FusedMoE.__init__` for custom mode=3 does not restore
  the 212s reference behavior.
- The dominant issue remains the shared device-side resident NPU copy latency:
  current local custom is still 6-8 ms, while the reference log is about
  0.14-0.18 ms.
- Changed `VLLM_ASCEND_CUSTOM_MODE3_USE_BASE_INIT` to default off (`0`) so the
  ineffective path remains available only as a diagnostic A/B switch.

Do not repeat:

- Do not retry custom mode=3 base-init alignment as a speed fix unless a new
  layout signal appears. It ran successfully but produced `316.394690s`, still
  far from the 212s reference.

### 2026-05-28: Local qwen30b_tpu_verl scope correction and copy-event diagnosis plan

Scope correction:

- All custom/native mode=3 implementation and experiments must target this local repo only:
  `/workspace/cann-recipes-train/llm_rl/ref_wj_qwen3/qwen30b_tpu_verl`.
- `mode3-record.txt` remains useful as a 212s behavior/perf reference, but it was generated from `/workspace/cann-recipes-train/llm_rl/qwen3`; do not copy code from that sibling tree into this repo.
- Use this repo's `origin/master` only as the code comparison baseline.

Current signal:

- Local custom mode=3 hot path in `vllm_ascend/ops/fused_moe.py` is already very close to this repo's `origin/master` for the shared double-buffer path.
- The local custom run with base initializer enabled completed, but remained slow: `316.394690s`; base-init A/B is not a fix.
- Reference 212s run and current local custom run both use stage=8, source_from_npu=8/source_from_cpu=8, direct async CPU path, and skip mode3 post-shrink warmup.
- The main visible gap remains `prefetch_npu_dev_ms`: reference normal layers are about `0.14-0.18 ms`; current local normal layers are about `6-8 ms`.
- Host submit for NPU resident copies is only hundreds of microseconds in both cases, so the next question is whether the NPU event is measuring real copy latency or stream backlog before the copy event.

Planned local-only diagnostic change:

- Add an off-by-default `VLLM_ASCEND_MODE3_COPY_FORMAT_DIAG=1` diagnostic to the shared `Mode3DoubleBufferManager` in local `vllm_ascend/ops/fused_moe.py`.
- Log real weight/slot/slice metadata: NPU format, shape, stride, storage offset, contiguity, data_ptr.
- Add fine-grained NPU timing around each of the W13 and W2 resident NPU copy groups, so `prefetch_npu_dev_ms` can be split into W13 vs W2 and checked for stream backlog.
- Because native/common mode=3 reuses the same manager, this single diagnostic applies to both custom and native paths.

Applied local diagnostic change:

- Updated local `vllm_ascend/ops/fused_moe.py` shared `Mode3DoubleBufferManager` only.
- Added `VLLM_ASCEND_MODE3_COPY_FORMAT_DIAG=1` and `VLLM_ASCEND_MODE3_COPY_FORMAT_DIAG_FIRST_N` controls.
- The diagnostic logs real slot/source metadata and first copy-run view metadata for W13/W2.
- Added W13/W2 NPU event pairs around resident NPU copy groups.
- Updated local `vllm_ascend/ops/common_fused_moe.py` native/common timing output to print the new W13/W2 split when the diagnostic is enabled.
- Default behavior is unchanged when `VLLM_ASCEND_MODE3_COPY_FORMAT_DIAG=0`.

Diagnostic run result:

- Run log: `wjeagerqwen30b-a3b-with_draft_breakdown_20260528222824_elastic.txt`.
- Exit: `exit_code=1`.
- Failure cause: diagnostic logging bug, not a mode=3 data-path failure:
  `NameError: name 'prefetch_npu_w13_dev_ms' is not defined` in local
  `vllm_ascend/ops/fused_moe.py` custom timing-detail log.
- Useful signal before failure:
  - Real `slot.w13`, `slot.w2`, `layer.w13_weight`, and `layer.w2_weight`
    are all `format=FRACTAL_NZ`.
  - First resident NPU copy run is contiguous: `assignments=8 runs=1`,
    `first_run=(0, 0, 8)`, storage offset 0 for both source and destination
    views.
  - The earlier plain ND synthetic copy benchmark is not representative of
    this mode3 path.
- Fix next: define W13/W2 split timing variables in the custom fused-experts
  timing block before logging them, then rerun the same local custom
  diagnostic.

Diagnostic fix applied:

- Fixed local custom timing detail by defining `prefetch_npu_w13_dev_ms` and `prefetch_npu_w2_dev_ms` before the diagnostic detail log.
- Static validation: `python3 -m py_compile vllm_ascend/ops/fused_moe.py vllm_ascend/ops/common_fused_moe.py` passed.

Rerun planned:

- Same local custom mode=3 threshold diagnostic as the failed run.
- Keep `VLLM_ASCEND_MODE3_COPY_FORMAT_DIAG=1`.

Rerun result:

- Log:
  `wjeagerqwen30b-a3b-with_draft_breakdown_20260528223626_elastic.txt`.
- Exit: `exit_code=0`.
- `rollout_output_time_s=327.039699`.
- `timing_s/gen=327.0361759378575`.
- `timing_per_token_ms/gen=1.4483957621964352`.
- `perf/total_num_tokens=263872`.
- The total runtime is diagnostic-heavy because `COPY_FORMAT_DIAG` logs many
  large metadata lines, so use this run mainly for device-copy attribution.
- Copy format signal:
  - `slot.w13`, `slot.w2`, `layer.w13_weight`, and `layer.w2_weight` are all
    `FRACTAL_NZ`.
  - The resident NPU copy is one contiguous run:
    `assignments=8 runs=1 first_run=(0, 0, 8)`.
  - Source and destination views have storage offset 0 and are marked
    contiguous.
- Copy timing split:
  - Normal-layer `npu_total_ms` remains around 6-8 ms.
  - W13 dominates: about 4-7 ms.
  - W2 is consistently about 1.7-2.0 ms.
  - This confirms the slow path is the device-side resident NPU copy into the
    mode3 runtime slot, not just host submit overhead.

Conclusion:

- Local custom mode=3 is functionally running, but its resident NPU copy path
  is much slower than the 212s reference.
- The shared manager is copying contiguous `FRACTAL_NZ` slices, so the next
  useful local-only experiment is to make the runtime double-buffer layout
  configurable and test whether using a plain contiguous runtime slot avoids
  the slow `FRACTAL_NZ` copy while preserving fused-experts correctness.

Next code change:

- Add an off-by-default A/B switch in local `vllm_ascend/ops/fused_moe.py`:
  `VLLM_ASCEND_MODE3_RUNTIME_BUFFER_FORMAT=formatted|plain`.
- Default stays `formatted`, matching current behavior.
- With `plain`, allocate mode3 runtime double-buffer slots with plain
  `torch.empty` instead of `torch_npu.npu_format_cast(...)`.
- First validate custom mode=3 with threshold limit and timing enabled.

Implemented:

- Added `_allocate_plain_buffer_like(...)` in local
  `vllm_ascend/ops/fused_moe.py`.
- Added shared manager env switch:
  `VLLM_ASCEND_MODE3_RUNTIME_BUFFER_FORMAT`.
  - `formatted`: current/default behavior.
  - `plain`: allocate mode3 runtime double-buffer slots as plain contiguous
    NPU tensors.
- The switch is local to the shared `Mode3DoubleBufferManager`, so it affects
  both local custom mode=3 and local native/common mode=3 when enabled.
- Added a diagnostic log of the requested/runtime buffer format when
  `VLLM_ASCEND_MODE3_COPY_FORMAT_DIAG=1`.

Static validation:

- `python3 -m py_compile vllm_ascend/ops/fused_moe.py vllm_ascend/ops/common_fused_moe.py`
  passed.

Run planned:

- Local custom mode=3 threshold run.
- Env:
  - `VLLM_ASCEND_REGISTER_CUSTOM_MODELS=1`
  - `VLLM_ASCEND_MODE3_RUNTIME_BUFFER_FORMAT=plain`
  - `VLLM_ASCEND_MODE3_COPY_FORMAT_DIAG=1`
  - timing sync enabled for W13/W2 copy attribution
- Expected signal:
  - If fused-experts requires formatted weights, the run should fail early.
  - If plain runtime slots are accepted, check whether `npu_w13_ms` and
    `npu_w2_ms` drop from the current `4-7 ms` and `~1.8 ms` bands.

Run result:

- Log:
  `wjeagerqwen30b-a3b-with_draft_breakdown_20260528225128_elastic.txt`.
- Scope:
  local `llm_rl/ref_wj_qwen3/qwen30b_tpu_verl` code only.
- Exit: `exit_code=0`.
- Path: local custom mode=3 (`VLLM_ASCEND_REGISTER_CUSTOM_MODELS=1`).
- Env delta:
  - `VLLM_ASCEND_MODE3_RUNTIME_BUFFER_FORMAT=plain`
  - `VLLM_ASCEND_MODE3_COPY_FORMAT_DIAG=1`
  - timing sync enabled
- Result:
  - `rollout_output_time_s=179.111114`.
  - `timing_s/gen=179.107659782283`.
  - `timing_per_token_ms/gen=0.793241832227373`.
  - `perf/total_num_tokens=263872`.
  - `response_length/mean=441.0`, `max=897.0`, `min=257.0`.
- Format signal:
  - Runtime slot tensors are `ND`.
  - Layer resident weights remain `FRACTAL_NZ`.
  - One contiguous local NPU run is copied per layer:
    `assignments=8 runs=1 first_run=(0, 0, 8)`.
- Timing signal:
  - First layer still includes initialization noise
    (`prefetch_npu_dev_ms=7.409`).
  - Normal layers drop to roughly:
    - `prefetch_npu_dev_ms=0.28-0.45 ms`.
    - `npu_w13_ms=0.20-0.37 ms`.
    - `npu_w2_ms=0.07-0.09 ms`.
    - `prefetch_cpu_dev_ms=1.3-2.2 ms`.
- Conclusion:
  - The prior local custom 300s+ runs were dominated by copying into
    `FRACTAL_NZ` runtime slots.
  - Plain contiguous runtime slots are accepted by local custom fused-experts
    and materially outperform the 212s reference on this threshold run.

Next run:

- Validate the same shared mode3 manager path in local native/common mode3.
- Use `VLLM_ASCEND_REGISTER_CUSTOM_MODELS=0`.
- Keep `VLLM_ASCEND_MODE3_RUNTIME_BUFFER_FORMAT=plain`.
- Keep timing on, but turn off verbose copy-format diagnostics unless needed.

Run result:

- Log:
  `wjeagerqwen30b-a3b-with_draft_breakdown_20260528230246_elastic.txt`.
- Scope:
  local `llm_rl/ref_wj_qwen3/qwen30b_tpu_verl` code only.
- Exit: `exit_code=0`.
- Path: local native/common mode3
  (`VLLM_ASCEND_REGISTER_CUSTOM_MODELS=0`).
- Env delta:
  - `VLLM_ASCEND_MODE3_RUNTIME_BUFFER_FORMAT=plain`
  - `VLLM_ASCEND_MODE3_COPY_FORMAT_DIAG=1`
  - `VLLM_ASCEND_MODE3_COPY_FORMAT_DIAG_FIRST_N=1`
  - timing sync enabled
- Result:
  - `rollout_output_time_s=162.803968`.
  - `timing_s/gen=162.80058594001457`.
  - `timing_per_token_ms/gen=0.7210201687394353`.
  - `perf/total_num_tokens=263872`.
  - `response_length/mean=441.0`, `max=897.0`, `min=257.0`.
- Format signal:
  - Native/common path uses the shared local
    `Mode3DoubleBufferManager`.
  - Runtime slot tensors are `ND`.
  - Layer resident weights remain `FRACTAL_NZ`.
- Timing signal:
  - First layer still includes setup noise
    (`npu_total_ms` around 6.6-8.0 ms).
  - Normal layers are around:
    - `npu_total_ms=0.30-0.49 ms`.
    - `npu_w13_ms=0.21-0.41 ms`.
    - `npu_w2_ms=0.07-0.09 ms`.
    - `cpu_ms=1.3-1.9 ms`.
- Notes:
  - Torch dynamo emitted `TypeError: '>=' not supported between instances of
    'NoneType' and 'int'` while logging compilation metrics, but the run
    completed successfully. Treat this as noisy metrics logging, not a mode3
    failure.
  - At the time of this run, the script still wrote rollout records under the
    existing `RECORD_DIR=/workspace/cann-recipes-train/llm_rl/qwen3/record`,
    but the code under test and log file were the local `qwen30b_tpu_verl`
    tree. This default was corrected in the later path-ownership fix below.

Code change:

- Set shared `Mode3DoubleBufferManager` default runtime slot format to
  `plain`.
- Keep `VLLM_ASCEND_MODE3_RUNTIME_BUFFER_FORMAT=formatted` as an explicit
  rollback switch.
- This affects both local custom mode3 and local native/common mode3 because
  both paths use the same manager implementation.

Static validation:

- `python3 -m py_compile vllm_ascend/ops/fused_moe.py vllm_ascend/ops/common_fused_moe.py`
  passed after changing the default.

Next run:

- Validate native/common mode3 with the new default, without explicitly
  setting `VLLM_ASCEND_MODE3_RUNTIME_BUFFER_FORMAT`.
- Keep timing on.
- Turn copy-format diagnostics off for a cleaner speed signal.

Result:

- Log: `wjeagerqwen30b-a3b-with_draft_breakdown_20260528231352_elastic.txt`.
- Exit: `exit_code=0`.
- Scope: local `llm_rl/ref_wj_qwen3/qwen30b_tpu_verl` code only.
- Path: local native/common mode3
  (`VLLM_ASCEND_REGISTER_CUSTOM_MODELS=0`).
- Env delta:
  - `VLLM_ASCEND_ELASTIC_EXECUTION_MODE=3`
  - `VLLM_ASCEND_ELASTIC_MIN_COMPUTE_GROUP_SIZE=8`
  - no explicit `VLLM_ASCEND_MODE3_RUNTIME_BUFFER_FORMAT`
  - timing sync/log enabled for diagnostics
- Result:
  - `rollout_output_time_s=163.597700`.
  - `timing_s/gen=163.59379811026156`.
  - `timing_per_token_ms/gen=0.7245331903267678`.
  - `perf/total_num_tokens=263872`.
  - `response_length/mean=441.0`, `max=897.0`, `min=257.0`.
- Timing signal:
  - First layer still includes setup noise
    (`prefetch_npu_dev_ms` around 6.9-7.8 ms).
  - Normal layers show the default plain runtime slot behavior:
    - `prefetch_npu_dev_ms` around 0.29-0.42 ms.
    - `prefetch_cpu_dev_ms` around 1.30-1.45 ms.
    - `submit_npu_us` around 210-340 us on sampled normal layers.

Conclusion:

- The new default `plain` runtime buffer format is active without an explicit
  env override.
- Native/common mode3 now runs through the threshold-limited validation and is
  faster than the 212s custom reference target on the same token-count run.
- Keep `VLLM_ASCEND_MODE3_RUNTIME_BUFFER_FORMAT=formatted` only as a rollback
  switch for diagnosing formatted-slot behavior.

### 2026-05-28: Keep run artifacts under local qwen30b_tpu_verl

Intent:

- Correct the path ownership mismatch pointed out by the user.
- Both custom mode3 and native/common mode3 work for this task must be under
  local `llm_rl/ref_wj_qwen3/qwen30b_tpu_verl`.
- Avoid defaulting rollout/output artifacts into sibling `llm_rl/qwen3`.

Applied change:

- In `internal/wj_train_grpo_qwen30b_a3b_16die_true_weight_eager.sh`,
  changed `RECORD_DIR` default to `${HOME}/record`, while preserving external
  `RECORD_DIR=...` override.
- In `internal/wj_train_grpo_qwen30b_a3b_16die_true_weight_regroup.sh`,
  changed `OUTPUT_ROOT` default to `$(pwd)`, while preserving external
  `OUTPUT_ROOT=...` override.

Validation plan:

- `bash -n internal/wj_train_grpo_qwen30b_a3b_16die_true_weight_eager.sh`
- `bash -n internal/wj_train_grpo_qwen30b_a3b_16die_true_weight_regroup.sh`

Validation:

- `bash -n internal/wj_train_grpo_qwen30b_a3b_16die_true_weight_eager.sh`
  passed.
- `bash -n internal/wj_train_grpo_qwen30b_a3b_16die_true_weight_regroup.sh`
  passed.
- `python3 -m py_compile vllm_ascend/ops/fused_moe.py vllm_ascend/ops/common_fused_moe.py`
  passed after the mode3 code changes.
