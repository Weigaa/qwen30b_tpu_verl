# Custom Mode=1 Zero-Headroom Migration Guide

This note summarizes how to migrate the old custom Qwen3 mode=1 path so it can
run without generic KV-cache headroom while still using the custom operator
stack:

- custom model registration may stay enabled;
- Qwen3 custom MoE still uses `vllm_ascend.ops.fused_moe.AscendFusedMoE`;
- mode=1 stays on the old custom `fused_moe.py` implementation;
- no fallback to `common_fused_moe.py` is required.

The target execution model is:

```text
preallocate floor-target loaded expert slots before profile
shrink by switching active group + expert map
do not allocate mode1 runtime expert buffers
do not enable hybrid / CPU-shadow / mode3 state
reuse full-world EP where useful
drop stale MC2 workspaces aggressively
skip generic headroom only after parity-ready is true
```

## Core Principle

Old custom mode=1 needed headroom because it was not a true lightweight mode=1
path. It still carried pieces of mode2/mode3-era machinery:

- runtime expert buffers;
- CPU shadow/offload state;
- hybrid resident/cpu-only expert state;
- synthetic post-shrink warmups;
- stale MC2 group workspaces;
- generic post-restore DP/EP/MoE headroom.

The fix is not to reserve more memory. The fix is to make those extra runtime
costs disappear or become visible during vLLM profiling.

## 1. Keep Custom Qwen3 on Old `fused_moe`

The custom model path must keep importing the old operator stack:

```python
from vllm_ascend.ops.fused_moe import AscendFusedMoE
```

Do not route custom Qwen3 through `common_fused_moe.py`. That file can be used
as a behavior reference, but the implementation should remain in the old custom
`fused_moe.py`.

## 2. Preallocate Floor-Target Loaded Slots Before `create_weights`

In `vllm_ascend/ops/fused_moe.py`, mode=1 lossless redundant execution must
compute the loaded expert capacity before `create_weights(...)`.

The important fields are:

| Field | Meaning |
| --- | --- |
| `active_local_num_experts` | Primary EP-local expert count used by the active rank |
| `loaded_local_num_experts` | Number of logical experts actually loaded on this rank |
| `loaded_expert_map` | Logical expert id -> loaded physical slot |
| `loaded_weight_capacity` | Physical loaded slot capacity reserved in weights |
| `num_local_expert_weight_slots` | Number of expert rows passed to weight creation |

The migration rule is:

```python
if mode == 1 and lossless and redundant:
    loaded_local_num_experts, loaded_expert_map = (
        determine_redundant_replica_expert_map(...)
    )

    floor_capacity = ceil(global_num_experts / configured_floor)
    loaded_weight_capacity = max(loaded_local_num_experts, floor_capacity)

    num_local_expert_weight_slots = loaded_weight_capacity
    active_local_num_experts = primary_ep_local_experts

    create_weights(num_local_expert_weight_slots)
```

This makes the floor redundancy cost visible to vLLM memory profiling. It also
prevents shrink-time allocation of runtime expert buffers just to cover a
capacity mismatch.

## 3. Disable Mode2/Mode3 Runtime State in Mode=1

For mode=1, these fields must be cleared or left disabled:

```python
lossless_hybrid_cpu_swap_enabled = False
lossless_hybrid_active = False
lossless_hybrid_resident_capacity = 0
lossless_hybrid_owned_expert_ids = []
lossless_hybrid_resident_expert_ids = []
lossless_hybrid_cpu_only_expert_ids = []
lossless_hybrid_active_ranks = []
lossless_hybrid_rank_owned_expert_ids = []
lossless_hybrid_rank_resident_expert_ids = []
lossless_hybrid_rank_lru = []
lossless_hybrid_owner_rank_by_expert = None
lossless_hybrid_owner_global_rank_by_expert = None
lossless_hybrid_last_stats = {}
runtime_w13_buffer = None
runtime_w2_buffer = None
```

Mode=1 should not initialize CPU shadow ownership, mode3 slot managers, hybrid
tail state, or normal mode2/mode3 runtime buffers.

## 4. Add a Per-Layer Parity-Ready Flag

Add an authoritative flag on each custom MoE layer:

```python
lossless_mode1_native_parity_ready = False
```

Set it to true only when the layer is really in the lightweight state:

```python
lossless_mode1_native_parity_ready = (
    elastic_execution_mode == 1
    and elastic_moe_mode == "lossless"
    and loaded_weight_capacity >= floor_capacity
    and runtime_w13_buffer is None
    and runtime_w2_buffer is None
    and not lossless_hybrid_active
)
```

This flag is the worker-side gate for skipping headroom. Do not skip headroom
based only on class name or environment variables.

Log the state once per layer init or activation:

```text
mode=1 floor=<floor>
active_local_num_experts=<n>
loaded_local_num_experts=<n>
loaded_weight_capacity=<n>
hybrid_disabled=True
runtime_buffers_disabled=True
lossless_mode1_native_parity_ready=True
```

## 5. Activation After Shrink Uses Loaded Slots, Not Runtime Buffers

After shrink/import, the mode=1 path should prefer loaded-prefix / loaded-slot
views:

```text
update loaded_expert_map
copy imported experts into preallocated loaded slots if needed
bind runtime views to loaded weight prefix
keep runtime_w13_buffer/runtime_w2_buffer as None
```

Avoid this old pattern in mode=1:

```text
allocate runtime_w13_buffer/runtime_w2_buffer
copy all active experts into runtime buffer
bind runtime buffer as expert weights
```

Recommended debug assertions:

```python
assert runtime_w13_buffer is None
assert runtime_w2_buffer is None
assert loaded_weight_capacity >= target_owned_local
assert lossless_mode1_native_parity_ready
```

If mode=1 would allocate runtime buffers, log a high-signal warning and fail
fast in debug mode.

## 6. Worker Headroom Skips Must Be Parity-Gated

In `vllm_ascend/worker/worker_v1.py`, replace custom-class-based headroom skips
with a generic lightweight parity check.

The check should require:

```python
module.elastic_execution_mode == 1
module.elastic_moe_mode == "lossless"
module.lossless_mode1_native_parity_ready is True
module.runtime_w13_buffer is None
module.runtime_w2_buffer is None
module.lossless_hybrid_active is False
configured_floor in (4, 8)
```

When this is true, skip:

- post-shrink MoE dispatch headroom;
- post-shrink prefill AllToAll headroom;
- post-restore DP collective headroom;
- post-restore EP collective headroom;
- post-restore MoE dispatch headroom;
- first-live-prefill headroom;
- extra elastic safety headroom;
- custom mode=1 KV materialization headroom.

Expected logs:

```text
Skipping generic post-restore headroom for mode1 lightweight parity
Skipping post-shrink MoE dispatch headroom for mode1 lightweight parity
Skipping post-shrink prefill AllToAll headroom for mode1 lightweight parity
Skipping first-live-prefill headroom for mode1 lightweight parity
Skipping mode=1 KV materialization headroom for lightweight parity
```

If the parity-ready condition is false, keep the old conservative headroom path.

## 7. Skip Synthetic Post-Shrink MoE Warmup in Mode=1

Old synthetic warmup can itself create a memory peak:

```text
shrink to floor
old MC2/EP/DP cache may still be alive
run dummy MoE dispatch
dummy dispatch materializes MC2/MoE temporary workspace
workspace overlaps KV cache and stale communication workspaces
```

For lightweight mode=1, skip this warmup by default:

```python
if _module_is_mode1_lightweight_parity(module):
    if not force_mode1_parity_warmup:
        mark_signature_as_warmed(active_signature)
        log("mode1_lightweight_parity_no_synthetic_warmup")
        return
```

Expected log:

```text
Elastic post-shrink MoE dispatch warmup skipped:
reason=mode1_lightweight_parity_no_synthetic_warmup
```

Keep an environment variable to force the warmup only for diagnostics, not as
the default steady-state path.

## 8. Fix Communication Group Workspace Lifetime

This is the largest source of old OOMs. The correct policy is:

| Group | Policy |
| --- | --- |
| Full-world `_EP` | Keep/cache for restore reuse |
| `_DP` | Reuse through the normal group cache |
| Current MC2 signature | May be reused while it is current |
| Stale MC2 signatures | Drop aggressively |

MC2 is tied to the active-rank signature. A floor8 MC2 group and a floor4 MC2
group can each hold large HCCL workspaces. Do not keep multiple historical MC2
groups alive.

Recommended behavior:

```text
keep full-world EP cache
cache current MC2 by active-rank signature
after shrink/restore, destroy stale MC2 signatures
after MC2 destroy, run gc.collect + torch.npu.empty_cache + synchronize
```

Expected logs:

```text
Elastic mode1 lightweight parity keeps stale EP cache for native-like reuse
Elastic parallel stale MC2 cache dropped across restore
after_mc2_group_destroy_cleanup
```

The wording "stale EP" in the log means "not current during shrink but useful
for full-world restore". That full-world EP cache should be kept. Stale MC2
should not be kept.

## 9. Use Explicit Mode1 KV Token Caps

Do not rely on generic headroom to indirectly reduce KV cache.

Keep the base vLLM formula:

```python
num_blocks = available_memory // page_size // num_layers
```

Then cap mode=1 parity KV tokens explicitly:

```python
default_max_tokens_by_floor = {
    "4": 277120,
    "8": 377344,
}

max_blocks_per_group = max_tokens // (block_size * dcp_world_size)
max_blocks = max_blocks_per_group * num_kv_cache_groups
num_blocks = min(num_blocks, max_blocks)
```

Expected log:

```text
Capping mode1 parity KV blocks to floor4 budget:
requested_tokens=<n> capped_tokens=277120 max_tokens=277120
```

For floor8:

```text
capped_tokens=377344
```

## 10. Recommended Runtime Environment

For custom floor4/floor8 zero-headroom validation:

```bash
export VLLM_ASCEND_REGISTER_CUSTOM_MODELS=1
export VLLM_ASCEND_ELASTIC_EXECUTION_MODE=1
export VLLM_ASCEND_ELASTIC_MIN_COMPUTE_GROUP_SIZE=4
export VLLM_ASCEND_MODE1_PARITY_NATIVE_KV_CAP=1

export VLLM_ASCEND_CUSTOM_MODE1_KV_MATERIALIZE_HEADROOM_BYTES=0
export VLLM_ASCEND_KV_CACHE_INIT_HEADROOM_BYTES=0
export VLLM_ASCEND_POST_RESTORE_DP_HEADROOM_BYTES=0
export VLLM_ASCEND_POST_RESTORE_EP_HEADROOM_BYTES=0
export VLLM_ASCEND_POST_RESTORE_MOE_DISPATCH_HEADROOM_BYTES=0
export VLLM_ASCEND_FIRST_LIVE_PREFILL_HEADROOM_BYTES=0
export VLLM_ASCEND_FIRST_LIVE_PREFILL_LOW_FLOOR_HEADROOM_BYTES=0
export VLLM_ASCEND_POST_SHRINK_MOE_DISPATCH_HEADROOM_BYTES=0
export VLLM_ASCEND_POST_SHRINK_MOE_DISPATCH_LOW_FLOOR_HEADROOM_BYTES=0
export VLLM_ASCEND_POST_SHRINK_PREFILL_ALLTOALL_HEADROOM_BYTES=0
export VLLM_ASCEND_EXTRA_ELASTIC_SAFETY_HEADROOM_BYTES=0
export VLLM_ASCEND_FLOOR_PREALLOC_HEADROOM_SAFETY_BYTES=0
```

Use floor8 first, then floor4:

```bash
export VLLM_ASCEND_ELASTIC_MIN_COMPUTE_GROUP_SIZE=8
# validate

export VLLM_ASCEND_ELASTIC_MIN_COMPUTE_GROUP_SIZE=4
# validate
```

## 11. Validation Checklist

### floor8

Expected:

```text
custom Qwen3 registration present
GPU KV cache size: 377,344 tokens
Maximum concurrency for 17,408 tokens per request: 21.68x
no Applying ... headroom
no Failed to allocate / OOM / RuntimeError
Training Progress: 100%|...| 3/3
```

### floor4

Expected:

```text
custom Qwen3 registration present
GPU KV cache size: 277,120 tokens
Maximum concurrency for 17,408 tokens per request: 15.92x
no Applying ... headroom
no Failed to allocate / OOM / RuntimeError
Training Progress: 100%|...| 3/3
```

### Required parity logs

```text
lossless_mode1_native_parity_ready=True
runtime_buffers_disabled=True
hybrid_disabled=True
Skipping generic post-restore headroom for mode1 lightweight parity
Skipping post-shrink MoE dispatch headroom for mode1 lightweight parity
Skipping first-live-prefill headroom for mode1 lightweight parity
Elastic post-shrink MoE dispatch warmup skipped:
reason=mode1_lightweight_parity_no_synthetic_warmup
Elastic mode1 lightweight parity keeps stale EP cache for native-like reuse
Elastic parallel stale MC2 cache dropped across restore
```

## 12. Failure Interpretation

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| `target_owned_local > loaded_weight_capacity` | Floor-target slot capacity was not preallocated | Increase `loaded_weight_capacity` before `create_weights` |
| `runtime_w13_buffer` allocated in mode=1 | Layer fell back to old runtime-buffer path | Route activation through loaded slots; fail fast in debug |
| Headroom lines still appear | Worker did not detect parity-ready module | Check `lossless_mode1_native_parity_ready` and worker predicate |
| Step2/step3 resume OOM | Stale MC2 workspaces are accumulating | Drop stale MC2 after shrink/restore |
| Restore OOM | Full-world EP/DP/MC2 workspace recreated while stale floor workspace is alive | Keep full-world EP, release stale MC2 |
| Floor4 KV lower than 277,120 | Generic headroom still being subtracted | Confirm all headroom envs are zero and parity-ready is true |

## Final Summary

The old custom mode=1 path used headroom to compensate for unnecessary runtime
costs. The migrated path removes those costs:

```text
old:
  runtime expert buffers
  hybrid / CPU shadow state
  synthetic post-shrink warmup
  stale MC2 workspaces
  generic 7.5 GiB headroom

new:
  floor loaded slots preallocated before profile
  shrink only changes active group + expert map
  no runtime buffers in mode=1
  full-world EP reused
  stale MC2 dropped
  explicit KV token cap
  zero generic headroom
```

Once `lossless_mode1_native_parity_ready=True`, the worker can safely use the
zero-headroom path.
