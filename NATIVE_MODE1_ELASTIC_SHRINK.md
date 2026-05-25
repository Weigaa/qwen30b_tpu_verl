# Native Mode=1 Elastic Shrink Implementation

## Summary

This note summarizes the current native `mode=1` lossless elastic shrink path for Qwen3-30B-A3B. The key switch is to keep vLLM's native `Qwen3MoeForCausalLM` model path enabled and avoid registering the custom Qwen3 model override:

```bash
export VLLM_ASCEND_REGISTER_CUSTOM_MODELS=0
export VLLM_ASCEND_ELASTIC_EXECUTION_MODE=1
export VLLM_ASCEND_ELASTIC_MIN_COMPUTE_GROUP_SIZE=8  # or 4
```

In this mode, the model class remains the native vLLM implementation, while the MoE execution still goes through vLLM-Ascend's `AscendFusedMoE` layer. The shrink implementation is therefore concentrated in the fused MoE layer and worker parallel-group rebuild code rather than in a custom `Qwen3MoeForCausalLM` subclass.

The practical result from current runs is:

| Setting | Extra headroom | KV cache | Max concurrency for 17,408 tokens | 3 epochs | Preemption |
| --- | ---: | ---: | ---: | --- | --- |
| floor=8 | 0 | 377,344 tokens | 21.68x | success | none observed |
| floor=4 | 0 | 277,120 tokens | 15.92x | success in full validation | none observed |

This is cheaper than the earlier custom model path because native `mode=1` does not need the custom runtime expert buffers, CPU shadow state, or hybrid/fixed-slot machinery that existed to support other shrink/offload modes.

## Entry Points

The training script enables elastic shrink through:

- `internal/wj_train_grpo_qwen30b_a3b_16die_true_weight_eager.sh`
- `VLLM_ASCEND_ELASTIC_EXECUTION_MODE=1`
- `VLLM_ASCEND_ELASTIC_MIN_COMPUTE_GROUP_SIZE=<floor>`
- `VERL_SIDECAR_ENABLE=0` for the pure training/shrink validation runs

The vLLM-Ascend plugin only registers custom model classes when explicitly requested:

- `vllm_ascend/__init__.py`
- `register_model_loader()`
- `VLLM_ASCEND_REGISTER_CUSTOM_MODELS`

With `VLLM_ASCEND_REGISTER_CUSTOM_MODELS=0`, `Qwen3MoeForCausalLM` resolves to vLLM's native model registry entry in `vllm/model_executor/models/registry.py`, not `vllm_ascend.models.qwen3_moe:CustomQwen3MoeForCausalLM`.

## Native Model, Ascend MoE

The native model path still constructs MoE layers through vLLM's fused MoE abstraction:

- `vllm/model_executor/models/qwen3_moe.py`
- `vllm/model_executor/layers/fused_moe/layer.py`
- `vllm_ascend/ops/fused_moe.py`

The important separation is:

1. The transformer/model class is native vLLM.
2. The Ascend MoE layer owns expert placement, redundant slots, runtime expert maps, and fused dispatch.
3. Elastic shrink is applied by rebuilding DP/EP/MC2 communication groups and updating the MoE layer's runtime mapping.

This keeps the fast native Qwen3 path while still allowing the Ascend fused MoE implementation to provide lossless elastic behavior.

## Slot Preallocation

For mode=1, the code computes the amount of redundant expert capacity required by the configured floor. The core logic is in:

- `vllm_ascend/envs.py`
  - `compute_elastic_init_redundancy_expert()`
- `vllm_ascend/ops/fused_moe.py`
  - `_get_configured_elastic_min_compute_group_size()`
  - `_get_reserved_local_expert_slots_for_floor()`
  - `_get_lossless_loaded_slot_capacity()`

The slot rule is:

```text
reserved local expert slots = logical_num_experts / floor
```

For Qwen3 MoE with 128 logical experts:

| Floor | Local slots per participating rank |
| ---: | ---: |
| 16 | 8 |
| 8 | 16 |
| 4 | 32 |
| 2 | 64 |
| 1 | 128 |

The mode=1 native path preallocates the target slot count during model initialization. This means the extra expert storage is already visible to vLLM memory profiling, so it naturally reduces the KV-cache budget without requiring a second manual subtraction for the static expert memory.

## Runtime Shrink Flow

The runtime shrink flow is driven by the worker:

- `vllm_ascend/worker/worker_v1.py`
  - `rebuild_elastic_ep_group()`
  - `_prepare_lossless_shrink_payload()`
  - `_preload_lossless_shrink_import_weights()`
  - `_refresh_elastic_parallel_state()`
  - `_warmup_post_shrink_moe_dispatch()`

The high-level sequence is:

1. Scheduler detects that some ranks no longer have unfinished requests.
2. Active ranks are selected, respecting the configured floor and MC2 divisibility constraints.
3. The worker prepares a lossless shrink payload from the previous active group.
4. Active ranks rebuild DP, EP, and MC2 groups over the new active rank list.
5. Inactive ranks advance the same group-creation sequence as non-members, so collective order stays aligned.
6. MoE layers refresh their runtime expert map and active local expert count.
7. A small post-shrink dummy decode warms up MoE dispatch/HCCL workspaces.
8. Decode continues on the smaller active group.
9. Before rollout restore, all ranks rebuild full 16-rank DP/EP groups.

The important property is that expert identity is preserved. The runtime map changes which local slot a logical expert resolves to, but the expert weights are not approximated, dropped, quantized differently, or recomputed. That is why this path is lossless from the model-output perspective.

## Why It Costs Less Than Custom Mode=1

The earlier custom model route implemented shrink inside a custom Qwen3 model/MoE stack. It carried several categories of extra memory or staging state:

- `runtime_w13_buffer` / `runtime_w2_buffer`
- CPU shadow copies for expert import/offload
- hybrid/fixed-slot buffers
- post-restore DP/EP collective headroom
- first-live-prefill and restore workspaces

Those buffers are useful for hybrid or offload-style modes, especially mode=2/mode=3, but native `mode=1` does not need most of them. In mode=1 the intended strategy is simpler:

```text
preallocate enough NPU expert slots for the configured floor
then switch the runtime expert map and active communication group
```

Because the slot memory is allocated before profile, vLLM accounts for it as normal model memory. After the 2026-05-25 reverse parity update, native/common mode=1 lightweight layers also skip the generic transient headroom set when the layer explicitly reports `lossless_mode1_native_parity_ready=True`.

No extra headroom is needed for floor=8. The current floor=4 full 3-step validation also runs with the generic headroom set at zero.

## Headroom Interpretation

There are two different memory categories:

1. Static expert slot memory.

   This is preallocated in `AscendFusedMoE` and is already reflected in vLLM's memory profile. It should not be subtracted again as manual headroom.

2. Transient runtime workspace.

   This includes HCCL/MoE dispatch workspaces that appear after an actual shrink and are not always fully exercised by the initial profile. The old native floor=4 run covered this with a 2 GiB post-shrink MoE dispatch reservation. The current parity path instead avoids the synthetic warmup and stale-cache overlap that made the reservation necessary.

For clean floor=4 native mode=1 experiments, keep unrelated headrooms at zero unless specifically testing them:

```bash
export VLLM_ASCEND_POST_RESTORE_DP_HEADROOM_BYTES=0
export VLLM_ASCEND_POST_RESTORE_EP_HEADROOM_BYTES=0
export VLLM_ASCEND_POST_RESTORE_MOE_DISPATCH_HEADROOM_BYTES=0
export VLLM_ASCEND_FIRST_LIVE_PREFILL_HEADROOM_BYTES=0
export VLLM_ASCEND_FIRST_LIVE_PREFILL_LOW_FLOOR_HEADROOM_BYTES=0
export VLLM_ASCEND_EXTRA_ELASTIC_SAFETY_HEADROOM_BYTES=0
export VLLM_ASCEND_FLOOR_PREALLOC_HEADROOM_SAFETY_BYTES=0
```

Note that these are the actual names read by `worker_v1.py`; similarly named older variables will not override the defaults.

## Validation Results

### floor=8, no extra headroom

Log:

- `wjeagerqwen30b-a3b-with_draft_breakdown_20260521004908_elastic.txt`

Observed:

- KV cache: 377,344 tokens
- Max concurrency for 17,408 tokens/request: 21.68x
- Epoch rollout times: 1125.60s, 1117.99s, 1125.43s
- 3 epochs completed, exit code 0
- No `preempt`, `Failed to allocate`, or `RuntimeError` entries observed in the relevant run summary

### floor=4, 2 GiB post-shrink headroom

Log:

- `wjeagerqwen30b-a3b-with_draft_breakdown_floor4_postshrink2g_only_3epochs_20260521110430_elastic.txt`

Observed:

- Applied headroom: 2,147,483,648 bytes
- KV cache: 277,120 tokens
- Max concurrency for 17,408 tokens/request: 15.92x
- Epoch rollout times: 1137.54s, 1135.70s, 1147.05s
- 3 epochs completed, exit code 0
- No `preempt`, `Failed to allocate`, or `RuntimeError` entries observed in the relevant run summary

### floor=4, zero headroom short-threshold validation

Log:

- `wjeagerqwen30b-a3b-with_draft_breakdown_20260525132003_elastic.txt`

Command shape:

- `VLLM_ASCEND_REGISTER_CUSTOM_MODELS=0`
- `VLLM_ASCEND_ELASTIC_EXECUTION_MODE=1`
- `VLLM_ASCEND_ELASTIC_MIN_COMPUTE_GROUP_SIZE=4`
- generic headroom env vars set to `0`
- `VERL_ELASTIC_TAIL_VALIDATE_LEVEL_TOKENS=4,8,12,16,20`

Observed:

- KV cache: 277,120 tokens
- Max concurrency for 17,408 tokens/request: 15.92x
- Progress: `Training Progress: 100%|...| 3/3`
- Step rollout times: 70.51s, 66.53s, 63.24s
- 3 steps completed, exit code 0
- No `Applying ... headroom`, `Failed to allocate`, `OOM`, or `RuntimeError` entries observed in the run summary

### floor=4, zero headroom full validation

Log:

- `wjeagerqwen30b-a3b-with_draft_breakdown_20260525133811_elastic.txt`

Command shape:

- `VLLM_ASCEND_REGISTER_CUSTOM_MODELS=0`
- `VLLM_ASCEND_ELASTIC_EXECUTION_MODE=1`
- `VLLM_ASCEND_ELASTIC_MIN_COMPUTE_GROUP_SIZE=4`
- `VLLM_ASCEND_MODE1_PARITY_NATIVE_KV_CAP=1`
- generic headroom env vars set to `0`
- full tail validation thresholds, with `VERL_ELASTIC_TAIL_VALIDATE_LEVEL_TOKENS` unset

Observed:

- KV cache: 277,120 tokens
- Max concurrency for 17,408 tokens/request: 15.92x
- Progress: `Training Progress: 100%|...| 3/3`
- Step rollout times: 1236.11s, 1257.05s, 1263.97s
- Final step reward mean: 0.6953125
- Final step response length mean: 6273.77734375
- Final step aborted ratio: 0.0
- 3 steps completed, exit code 0
- No `Applying ... headroom`, `Failed to allocate`, `OOM`, `RuntimeError`, or `Error executing job` entries observed in the run summary

## Accuracy and Quality

Native `mode=1` is expected to preserve output quality because it is a lossless expert placement change:

- The logical expert set is unchanged.
- Router decisions still refer to the same logical expert ids.
- `loaded_expert_map` maps logical ids to preallocated physical slots.
- Shrink changes the active communication group and expert ownership, not the expert parameters.
- The same native Qwen3 model path is used, avoiding custom model overrides that could accidentally replace the fast path or dispatch different operators.

Empirically, the successful floor=8 and floor=4 runs completed normal rollout/training cycles without runtime quality guards failing. For paper tables, accuracy should still be reported from the standard answer-extraction/reward pipeline, but the implementation itself does not introduce an approximation mechanism.

## Recommended Usage

For the current native mode=1 implementation:

```bash
export VLLM_ASCEND_REGISTER_CUSTOM_MODELS=0
export VLLM_ASCEND_ELASTIC_EXECUTION_MODE=1
export VLLM_ASCEND_ELASTIC_MIN_COMPUTE_GROUP_SIZE=8
```

Use floor=8 with no additional headroom.

For floor=4:

```bash
export VLLM_ASCEND_REGISTER_CUSTOM_MODELS=0
export VLLM_ASCEND_ELASTIC_EXECUTION_MODE=1
export VLLM_ASCEND_ELASTIC_MIN_COMPUTE_GROUP_SIZE=4
export VLLM_ASCEND_POST_SHRINK_MOE_DISPATCH_LOW_FLOOR_HEADROOM_BYTES=0
```

Keep other headrooms disabled unless validating a separate failure mode. The parity-ready native/common mode=1 path should skip the generic headroom set.

## Main Takeaway

For mode=1, the native path is the better baseline than the older custom model path. It implements the same lossless semantic target, keeps the native Qwen3 fast path, accounts static expert redundancy during profile, and now follows the same lightweight no-headroom floor=4 accounting model as the optimized custom path. This makes it both simpler and cheaper while preserving correctness.

## 2026-05-25 Reverse Parity Update: Native floor=4 zero headroom

Goal: reuse the custom floor=4 zero-headroom lessons on the native/common
`AscendFusedMoE` path without switching native to the old custom operator stack
and without changing the MC2 backend.

Implemented changes:

- `common_fused_moe.py` now marks native/common mode=1 lossless redundant
  layers with `lossless_mode1_native_parity_ready=True` only when hybrid/runtime
  buffers are disabled and loaded slots cover the floor-targeted expert set.
- `worker_v1.py` now treats old custom and native/common mode=1 lightweight
  layers uniformly for zero-headroom accounting:
  post-restore collectives, first-live-prefill, post-shrink MoE dispatch,
  post-shrink prefill AllToAll, extra elastic safety, and KV materialization
  headroom are skipped only after the per-layer parity-ready flag is present.
- Native/common keeps the same group-cache lifecycle that fixed custom floor=4:
  keep the full-world `_EP` cache for restore reuse, but continue dropping stale
  MC2 cache after shrink so floor and full-world MC2 workspaces are not kept
  simultaneously.
- The mode1 parity KV block cap is no longer gated on
  `VLLM_ASCEND_REGISTER_CUSTOM_MODELS=1`, so native floor=4 can use the same
  277,120-token budget target as the validated custom run.
- The training script now allows external override of
  `VLLM_ASCEND_REGISTER_CUSTOM_MODELS`, so `REGISTER_CUSTOM_MODELS=0` native
  runs can be launched directly from the same script.
- First verification after these changes, with short tail thresholds, completed
  native floor=4 3/3 with `exit_code=0`, `GPU KV cache size: 277,120 tokens`,
  and no generic headroom application lines.
- Full validation also completed native floor=4 3/3 with the same 277,120-token
  KV budget, `REGISTER_CUSTOM_MODELS=0`, MC2 unchanged, all generic headrooms
  at zero, and `exit_code=0`.

Validation attempt log:

- `2026-05-25T05:11:48Z`: patch landed and `py_compile` passed for
  `worker_v1.py`, `common_fused_moe.py`, and `kv_cache_utils.py`. Next step is
  native floor=4 no-headroom runtime validation.
- `2026-05-25T05:16:59Z`: short native floor=4 run
  `wjeagerqwen30b-a3b-with_draft_breakdown_20260525131221_elastic.txt` failed
  before OOM with a capacity invariant:
  `target_owned_local=32, loaded_weight_capacity=31`. This showed native/common
  was still sizing weight slots from the uneven redundant map rows, while
  floor=4 direct NPU import needs the floor-target capacity on every surviving
  rank. Follow-up patch makes common/native `num_local_expert_weight_slots` and
  `loaded_weight_capacity` at least `logical_num_experts / floor` for mode=1.
- `2026-05-25T05:20:03Z`: short-threshold native floor=4 run
  `wjeagerqwen30b-a3b-with_draft_breakdown_20260525132003_elastic.txt`
  completed 3/3 with `exit_code=0`, `GPU KV cache size: 277,120 tokens`, and
  no generic headroom lines.
- `2026-05-25T05:38:11Z` to `2026-05-25T06:57:45Z`: full native floor=4 run
  `wjeagerqwen30b-a3b-with_draft_breakdown_20260525133811_elastic.txt`
  completed 3/3 with `exit_code=0`, `GPU KV cache size: 277,120 tokens`,
  `Maximum concurrency: 15.92x`, and no generic headroom, OOM, or runtime
  failure entries in the run summary.
