# Custom Mode=1 Zero-Headroom Code-Diff Summary

Last updated: 2026-05-25

This document summarizes the code changes that moved the old custom Qwen3
`mode=1` path from requiring about `7.5 GiB` of extra KV headroom to running
floor8 and floor4 with zero generic headroom.

The final target is:

```text
REGISTER_CUSTOM_MODELS=1
custom Qwen3 model path
old vllm_ascend.ops.fused_moe.AscendFusedMoE operator stack
mode=1 lossless elastic shrink
floor8 and floor4 pass 3/3
no generic Applying ... headroom lines
MC2 decode remains enabled
```

`common_fused_moe.py` was used only as a behavior reference. The final custom
path does not route custom Qwen3 through `common_fused_moe.py`.

## Result

| Run | Floor | Custom path | Extra generic headroom | KV cache | Max concurrency | Result |
| --- | --- | --- | --- | --- | --- | --- |
| `wjeagerqwen30b-a3b-with_draft_breakdown_20260524212331_elastic.txt` | 8 | old custom `fused_moe.py` | 0 | `377,344` tokens | `21.68x` | 3/3, `exit_code=0` |
| `wjeagerqwen30b-a3b-with_draft_breakdown_20260525090329_elastic.txt` | 4 | old custom `fused_moe.py` | 0 | `277,120` tokens | `15.92x` | 3/3, `exit_code=0` |

## Memory Cost Removed

The original `7.5 GiB` was not a real model requirement. It was a set of
reservations hiding custom-only runtime costs.

| Old reservation | Old purpose | Why it disappeared |
| --- | --- | --- |
| `post-restore DP` ~= `2 GiB` | First DP collective after restore could allocate HCCL workspace after KV sizing | Full-world group lifecycle is made reusable/visible, and parity-ready mode1 skips this generic reservation |
| `post-restore EP` ~= `2 GiB` | First EP/MoE collective after restore could allocate workspace after KV sizing | Full-world `_EP` cache is retained for restore instead of destroyed and recreated |
| `post-restore MoE dispatch` ~= `2 GiB` | First full-world MoE dispatch after restore could allocate a separate workspace | MC2/EP lifecycle is aligned with native and stale non-current MC2 resources are dropped |
| `post-shrink MoE dispatch` ~= `0.5 GiB` | Synthetic post-shrink dispatch warmup could allocate extra workspace | Lightweight mode1 skips synthetic post-shrink MoE warmup; real decode allocates only the needed current group |
| `first-live-prefill` ~= `1 GiB` | Insurance for real first prefill shape not fully covered by synthetic profile | After stale communication workspaces and runtime buffers are removed, the real prefill peak fits in the profiled budget |

The important point: the fix was not "subtract 7.5 GiB from KV". The fix was to
remove the custom-only allocations and only cap KV to the native floor budget.

## File-Level Patch Summary

### 1. Keep Custom Qwen3 on Old `fused_moe`

File:

```text
vllm_ascend/models/qwen3_moe.py
```

Patch shape:

```diff
- from vllm_ascend.ops.common_fused_moe import AscendFusedMoE
+ from vllm_ascend.ops.fused_moe import AscendFusedMoE
```

Purpose:

- custom Qwen3 still uses the original custom operator stack;
- `common_fused_moe.py` remains only a reference oracle;
- no native/common shortcut is used to satisfy custom parity.

Also keep custom-only live diagnostics opt-in:

```diff
+ def _custom_mode1_debug_enabled() -> bool:
+     return _env_flag("VLLM_ASCEND_CUSTOM_MODE1_DEBUG", "0")
+
+ def _custom_mode1_timing_events_enabled() -> bool:
+     return _env_flag("VLLM_ASCEND_CUSTOM_MODE1_TIMING_EVENTS", "0")
```

Heavy debug branches in `CustomSparseMoeBlock.forward()` should only run when
explicitly enabled. This prevents diagnostic `float()`, `topk()`, `unique()`,
`norm()`, and NPU event allocations from becoming steady-state custom overhead.

### 2. Preallocate Floor-Target Loaded Expert Slots Before Weight Creation

File:

```text
vllm_ascend/ops/fused_moe.py
```

Patch shape:

```diff
  primary_mapping = determine_expert_map(
      self.ep_size, self.ep_rank, num_experts, layer_idx=self.layer_idx)

+ if self.elastic_moe_mode == "lossless":
+     self.global_redundant_expert_num = effective_init_redundancy_expert
+     self.loaded_local_num_experts, self.loaded_expert_map = (
+         determine_redundant_replica_expert_map(
+             num_experts,
+             self.ep_size,
+             self.ep_rank,
+             self.global_redundant_expert_num))
+     self.active_local_num_experts = primary_local_num_experts
+     self.local_num_experts = self.active_local_num_experts
+     self.expert_map = primary_expert_map
+     self.log2phy = primary_log2phy
+     self.primary_log2phy = primary_log2phy.clone()
```

Then before `create_weights(...)`:

```diff
- self.loaded_weight_capacity = int(self.loaded_local_num_experts)
+ self.loaded_weight_capacity = int(self.loaded_local_num_experts)
+ if self.elastic_moe_mode == "lossless":
+     prealloc_capacity = self._get_lossless_loaded_slot_capacity()
+     if prealloc_capacity > self.loaded_weight_capacity:
+         self.loaded_weight_capacity = prealloc_capacity

  moe_quant_params = {
-     "num_experts": self.local_num_experts,
+     "num_experts": self.loaded_weight_capacity,
      "hidden_size": hidden_size,
      ...
  }
  self.quant_method.create_weights(layer=self, **moe_quant_params)
```

Purpose:

- `active_local_num_experts` remains the primary EP-local active count;
- `loaded_weight_capacity` becomes the floor-target physical capacity;
- expert slot memory is allocated before vLLM memory profile;
- shrink no longer needs to allocate a large runtime expert buffer after KV
  sizing.

The helper that drives this:

```diff
+ def _get_lossless_loaded_slot_capacity(self) -> int:
+     loaded_local_num_experts = int(getattr(
+         self, "loaded_local_num_experts",
+         getattr(self, "active_local_num_experts", 0)))
+     if self.elastic_moe_mode != "lossless":
+         return loaded_local_num_experts
+     if getattr(self, "global_redundant_expert_num", 0) <= 0:
+         return self._get_zero_redundancy_prealloc_capacity()
+     if not self._is_followup_shrink_enabled():
+         return loaded_local_num_experts
+     if self._is_hybrid_cpu_swap_enabled():
+         return max(loaded_local_num_experts,
+                    self._get_hybrid_resident_capacity())
+     floor = self._get_configured_elastic_min_compute_group_size()
+     if floor is None:
+         return loaded_local_num_experts
+     return max(loaded_local_num_experts,
+                self._get_reserved_local_expert_slots_for_floor(floor))
```

### 3. Make Mode1 a True Lightweight Path

File:

```text
vllm_ascend/ops/fused_moe.py
```

Patch shape:

```diff
+ self.lossless_mode1_native_parity_ready = False
+ self.runtime_w13_buffer = None
+ self.runtime_w2_buffer = None
+ self.lossless_hybrid_cpu_swap_enabled = False
+ self.lossless_hybrid_active = False
```

When `lossless + mode=1`:

```diff
+ if self.elastic_moe_mode == "lossless" and self.elastic_execution_mode == 1:
+     self.lossless_hybrid_cpu_swap_enabled = False
+     self.lossless_hybrid_active = False
+     self.lossless_hybrid_resident_capacity = 0
+     self.lossless_hybrid_owned_expert_ids = []
+     self.lossless_hybrid_resident_expert_ids = []
+     self.lossless_hybrid_cpu_only_expert_ids = []
+     self.lossless_hybrid_active_ranks = []
+     self.lossless_hybrid_rank_owned_expert_ids = []
+     self.lossless_hybrid_rank_resident_expert_ids = []
+     self.lossless_hybrid_rank_lru = []
+     self.lossless_hybrid_owner_rank_by_expert = None
+     self.lossless_hybrid_owner_global_rank_by_expert = None
+     self.lossless_hybrid_last_stats = {}
+     self.lossless_cpu_shadow_local_slots = {}
+     self.lossless_mode1_native_parity_ready = True
+     logger.info(
+         "Mode1 parity init: layer=%s floor=%s active_local=%s "
+         "loaded_local=%s loaded_capacity=%s hybrid_disabled=True "
+         "runtime_buffers_disabled=True parity_ready=True",
+         ...)
```

Purpose:

- mode1 no longer carries mode2/mode3 hybrid tail state;
- no CPU-shadow ownership is created for normal mode1;
- no mode3 double-buffer state is initialized;
- `lossless_mode1_native_parity_ready=True` becomes the authoritative signal
  that worker-side headroom can be skipped.

### 4. Bind Runtime Views to Loaded Slots, Not Runtime Buffers

File:

```text
vllm_ascend/ops/fused_moe.py
```

Patch shape:

```diff
+ def set_lossless_runtime_prefix_views(self) -> None:
+     active = int(self.active_local_num_experts)
+     if active > int(self.w13_weight.shape[0]):
+         raise RuntimeError(...)
+     self.runtime_w13_weight = self.w13_weight[:active]
+     self.runtime_w2_weight = self.w2_weight[:active]
+     self.runtime_w13_buffer = None
+     self.runtime_w2_buffer = None
+     self.runtime_weight_capacity = max(
+         int(getattr(self, "loaded_weight_capacity", 0)),
+         int(self.w13_weight.shape[0]))
+     self.lossless_runtime_activated = True
```

Old behavior to avoid:

```diff
- self.runtime_w13_buffer = torch.empty(...)
- self.runtime_w2_buffer = torch.empty(...)
- self.runtime_w13_weight = self.runtime_w13_buffer[:active]
- self.runtime_w2_weight = self.runtime_w2_buffer[:active]
```

Purpose:

- runtime weight tensors alias the canonical loaded expert weights;
- the old large `runtime_w13_buffer/runtime_w2_buffer` allocation disappears
  from mode1;
- mode1 activation cost becomes metadata remap plus slot copy/import, not a
  second expert-weight storage allocation.

### 5. During Shrink, Import Experts Directly Into Loaded Slots

Files:

```text
vllm_ascend/worker/worker_v1.py
vllm_ascend/ops/fused_moe.py
```

Patch shape in `worker_v1.py`:

```diff
+ loaded_weight_capacity = int(getattr(module, "loaded_weight_capacity", 0))
+ can_direct_fill_loaded_slots = (
+     target_owned_local_expert_count > 0
+     and loaded_weight_capacity >= target_owned_local_expert_count
+     and module.w13_weight.shape[0] >= target_owned_local_expert_count
+     and module.w2_weight.shape[0] >= target_owned_local_expert_count
+ )
+
+ requires_redundant_direct_npu = (
+     int(getattr(module, "global_redundant_expert_num", 0)) > 0
+     and not use_hybrid_cpu_swap
+     and len(previous_active_ranks) == 2 * len(active_ranks)
+     and set(active_ranks).issubset(set(previous_active_ranks))
+     and target_owned_local_expert_count > 0
+     and envs_ascend.VLLM_ASCEND_ELASTIC_EXECUTION_MODE == 1
+ )
+
+ use_direct_npu_slot_import = (
+     requires_redundant_direct_npu and can_direct_fill_loaded_slots
+ )
```

Receive buffers are loaded expert slots:

```diff
+ if direct_fill_preallocated_loaded:
+     recv_w13, recv_w2 = module.get_lossless_expert_npu_slot_recv_buffers(
+         local_import_slot_by_expert[expert_id])
+ else:
+     recv_w13 = torch.empty(...)
+     recv_w2 = torch.empty(...)
```

Patch shape in `fused_moe.py`:

```diff
+ def get_lossless_expert_npu_slot_recv_buffers(
+         self, local_slot: int) -> tuple[torch.Tensor, torch.Tensor]:
+     recv_w13 = self.w13_weight[local_slot:local_slot + 1]
+     recv_w2 = self.w2_weight[local_slot:local_slot + 1]
+     return (_npu_zero_offset_alias_for_p2p(self.w13_weight, recv_w13),
+             _npu_zero_offset_alias_for_p2p(self.w2_weight, recv_w2))
```

And export sends a canonical NPU slot view:

```diff
+ def export_lossless_expert_npu_weights(self, expert_ids):
+     ...
+     export_w13 = source_w13[local_slot:local_slot + 1]
+     export_w2 = source_w2[local_slot:local_slot + 1]
+     return (_npu_zero_offset_alias_for_p2p(source_w13, export_w13),
+             _npu_zero_offset_alias_for_p2p(source_w2, export_w2))
```

Purpose:

- active ranks are still discovered dynamically;
- no assumption is made about which ranks will exit;
- expert movement remains NPU-to-NPU P2P;
- imported experts land directly in the final loaded slots;
- no temporary mode1 runtime expert buffer is needed.

### 6. Activate From Loaded Slots and Fail Fast on Heavy Fallback

File:

```text
vllm_ascend/ops/fused_moe.py
```

Patch shape:

```diff
+ if self.elastic_execution_mode == 1:
+     self.clear_lossless_hybrid_state()
+     self.lossless_hybrid_cpu_swap_enabled = False
+     if not self._can_materialize_lossless_loaded_prefix_slots_for_target(
+             target_active_local_num_experts,
+             source_local_ids,
+             cpu_expert_weights):
+         logger.warning(
+             "Mode1 parity path refused heavyweight activation fallback: ...")
+         if _env_flag("VLLM_ASCEND_CUSTOM_MODE1_STRICT", "1"):
+             raise RuntimeError(
+                 "Mode1 parity path would fall back to heavyweight "
+                 "runtime-buffer activation")
```

When slots are available:

```diff
+ if self._can_materialize_lossless_loaded_prefix_slots(
+         source_local_ids, cpu_expert_weights):
+     copy/import experts into self.w13_weight/self.w2_weight prefix
+     self.loaded_expert_map = self.expert_map.clone()
+     self.loaded_local_num_experts = new_local_num_experts
+     self.lossless_cpu_w13_weight = None
+     self.lossless_cpu_w2_weight = None
+     self.lossless_cpu_shadow_local_slots = {}
+     self.set_lossless_runtime_prefix_views()
+     self.lossless_mode1_native_parity_ready = True
+     return
```

If it falls through to generic runtime-buffer refresh:

```diff
  self.refresh_lossless_runtime_weights(...)
+ if self.elastic_execution_mode == 1:
+     self.lossless_mode1_native_parity_ready = False
```

Purpose:

- mode1 never silently re-enters the heavyweight runtime-buffer path;
- if a future regression allocates `runtime_w13_buffer/runtime_w2_buffer`, the
  parity flag turns false and worker headroom skips stop applying;
- debug mode can fail fast at the first incorrect activation.

### 7. Restore Full World With Primary Prefix Views

File:

```text
vllm_ascend/ops/fused_moe.py
```

Patch shape:

```diff
+ def restore_lossless_full_world_primary_layout(self) -> None:
+     self.clear_lossless_hybrid_state()
+     recompute loaded_expert_map
+     recompute primary expert_map/log2phy
+     self.local_num_experts = self.active_local_num_experts
+     self.moe_config.num_local_experts = self.active_local_num_experts
+     self.elastic_runtime_log2phy = None
+     self.lossless_cpu_import_expert_ids = []
+     self.set_lossless_runtime_prefix_views()
+     if self.elastic_execution_mode == 1:
+         self.lossless_mode1_native_parity_ready = True
+     logger.info(
+         "Lossless full-world primary layout restored with loaded-slot prefix views: ...")
```

Purpose:

- restore does not allocate runtime expert buffers;
- restore returns to the same lightweight layout used at init;
- full-world mode after restore remains parity-ready.

### 8. Replace Class-Based Headroom With Parity-Gated Headroom

File:

```text
vllm_ascend/worker/worker_v1.py
```

Patch shape:

```diff
+ def _module_is_mode1_lightweight_parity(module, max_floor=None) -> bool:
+     if not _is_ascend_fused_moe_module(module):
+         return False
+     if int(getattr(module, "elastic_execution_mode", 0)) != 1:
+         return False
+     if not _module_uses_lossless_elastic(module):
+         return False
+     if not _module_has_preallocated_redundant_slots(module):
+         return False
+     if _module_hybrid_cpu_swap_enabled(module):
+         return False
+     if not getattr(module, "lossless_mode1_native_parity_ready", False):
+         return False
+     if max_floor is None:
+         return True
+     floor = _module_configured_elastic_floor(module)
+     return floor is not None and int(floor) <= max_floor
```

Use this predicate to skip old generic headrooms:

```diff
+ if _module_is_mode1_lightweight_parity(module):
+     logger.info(
+         "Skipping generic post-restore headroom for mode1 lightweight parity: "
+         "layer=%s floor=%s path=%s", ...)
+     continue
```

The same parity gate is used for:

```text
post-restore DP headroom
post-restore EP headroom
post-restore MoE dispatch headroom
post-shrink MoE dispatch headroom
post-shrink prefill AllToAll headroom
first-live-prefill headroom
extra elastic safety headroom
custom mode1 KV materialization headroom
```

Purpose:

- headroom is skipped only when the layer proves it is in lightweight mode1;
- this works for both floor8 and floor4;
- if the custom path regresses to heavy state, the fallback reservations can
  still protect correctness.

### 9. Skip Synthetic Post-Shrink MoE Warmup

File:

```text
vllm_ascend/worker/worker_v1.py
```

Patch shape:

```diff
+ force_mode1_parity_warmup = _env_flag(
+     "VLLM_ASCEND_MODE1_PARITY_POST_SHRINK_WARMUP", "0")
+
+ for module in model.modules():
+     if _module_is_mode1_lightweight_parity(module):
+         if not force_mode1_parity_warmup:
+             warmed_signatures.add(active_signature)
+             logger.info(
+                 "Elastic post-shrink MoE dispatch warmup skipped: "
+                 "reason=mode1_lightweight_parity_no_synthetic_warmup")
+             return
```

Purpose:

- old custom mode1 was warming up an all-layer synthetic post-shrink MoE
  dispatch that native did not need;
- this consumed allocator/HCCL workspace in the tight post-KV window;
- the lightweight path lets the first real decode allocate only the current
  required workspace.

### 10. Align MC2 and EP Communication Group Lifecycle With Native

File:

```text
vllm_ascend/worker/worker_v1.py
```

Patch shape:

```diff
+ def _should_cache_elastic_parallel_group(self, attr_name: str) -> bool:
+     group_kind = self._normalize_elastic_parallel_group_kind(attr_name)
+     if group_kind != "mc2":
+         return True
+     return not _env_flag("VLLM_ASCEND_DISABLE_ELASTIC_MC2_GROUP_CACHE", "0")
```

Do not keep stale MC2 by default:

```diff
+ def _should_keep_stale_mc2_cache_for_custom_mode1_parity(
+         stale_group_ranks, keep_group_ranks) -> bool:
+     if stale_group_ranks == keep_group_ranks:
+         return False
+     if not _env_flag("VLLM_ASCEND_MODE1_PARITY_KEEP_MC2_GROUP_CACHE", "0"):
+         return False
+     return self._has_mode1_lightweight_parity_module()
```

Keep only full-world `_EP` cache across shrink/restore:

```diff
+ def _should_keep_stale_group_cache_for_custom_mode1_parity(
+         group_kind, stale_group_ranks, keep_group_ranks) -> bool:
+     if group_kind == "mc2":
+         return self._should_keep_stale_mc2_cache_for_custom_mode1_parity(...)
+     if group_kind != "ep":
+         return False
+     if not _env_flag("VLLM_ASCEND_MODE1_PARITY_KEEP_FULLWORLD_EP_CACHE", "1"):
+         return False
+     full_world_ranks = tuple(range(torch.distributed.get_world_size()))
+     return stale_group_ranks == full_world_ranks and keep_group_ranks != full_world_ranks
```

Drop stale cached groups after shrink/restore:

```diff
+ for ranks in stale_group_ranks:
+     group = groups_by_ranks.pop(ranks, None)
+     if group is not None:
+         group.destroy()
+         if group_kind == "mc2":
+             self._cleanup_after_elastic_mc2_group_destroy("drop_stale_cache")
+             logger.info(
+                 "Elastic parallel stale MC2 cache dropped across restore: ...")
```

Purpose:

- full-world `_EP` cache is useful because restore returns to the same full
  world repeatedly;
- stale floor MC2 caches are harmful because they can remain resident into the
  next step's KV materialization;
- current MC2 group signatures can be reused, but old non-current MC2
  workspaces must not accumulate;
- this removes the largest non-torch memory drift that caused step-2/step-3
  failures at native KV budget.

### 11. Add Mode1 KV Diagnostics

File:

```text
vllm_ascend/worker/model_runner_v1.py
```

Patch shape:

```diff
+ def _custom_mode1_kv_diag_enabled() -> bool:
+     return _env_flag("VLLM_ASCEND_CUSTOM_MODE1_KV_DIAG", "0")
+
+ def _log_custom_mode1_kv_memory(tag: str) -> None:
+     if not _custom_mode1_kv_diag_enabled():
+         return
+     if envs_ascend.VLLM_ASCEND_ELASTIC_EXECUTION_MODE != 1:
+         return
+     free_bytes, total_bytes = torch.npu.mem_get_info()
+     stats = torch_npu.npu.memory_stats()
+     logger.info(
+         "Custom mode=1 KV memory: tag=%s free_bytes=%s total_bytes=%s "
+         "torch_current=%s torch_reserved=%s non_torch=%s total_allocated=%s",
+         ...)
```

Around KV allocation:

```diff
  def initialize_kv_cache_tensors(...):
+     _log_custom_mode1_kv_memory("before_initialize_kv_cache_tensors")
      allocate raw KV tensors
+     _log_custom_mode1_kv_memory("after_raw_kv_tensor_alloc")
      bind KV cache
+     _log_custom_mode1_kv_memory("after_initialize_kv_cache_tensors")
```

Purpose:

- prove whether failures are caused by torch model tensors or non-torch HCCL /
  communication workspaces;
- verify that stale MC2/EP fixes really reduce non-torch memory;
- keep the diagnostics opt-in so successful mode1 runs do not pay overhead.

### 12. Cap KV to Native Floor Budgets, Not Generic Headroom

File:

```text
vllm/v1/core/kv_cache_utils.py
```

Patch shape:

```diff
+ def maybe_cap_mode1_parity_num_blocks(vllm_config, kv_cache_groups, num_blocks):
+     if not _env_flag("VLLM_ASCEND_MODE1_PARITY_NATIVE_KV_CAP", "1"):
+         return num_blocks
+     if os.getenv("VLLM_ASCEND_ELASTIC_EXECUTION_MODE", "0") != "1":
+         return num_blocks
+     floor = os.getenv("VLLM_ASCEND_ELASTIC_MIN_COMPUTE_GROUP_SIZE", "")
+     if floor not in ("4", "8"):
+         return num_blocks
+
+     default_max_tokens_by_floor = {
+         "4": 277120,
+         "8": 377344,
+     }
+     max_tokens = int(os.getenv(
+         f"VLLM_ASCEND_MODE1_PARITY_MAX_KV_TOKENS_FLOOR{floor}",
+         default_max_tokens_by_floor[floor]))
+     ...
+     return min(num_blocks, max_blocks)
```

Call it in both KV config branches:

```diff
  num_blocks = available_memory // page_size
  num_blocks = may_override_num_blocks(vllm_config, num_blocks)
+ num_blocks = maybe_cap_mode1_parity_num_blocks(
+     vllm_config, kv_cache_groups, num_blocks)
```

Purpose:

- native floor8 validated at `377,344` tokens;
- native/custom floor4 validated at `277,120` tokens;
- this is not subtracting arbitrary headroom;
- it clamps the final KV block count to the validated native floor budget so
  custom cannot over-allocate a few extra blocks due to slightly different
  profile noise.

## What Not To Change

Do not make mode1 parity by changing model registration defaults:

```diff
- # hide custom Qwen3 by default and silently use native
+ # not allowed as the solution
```

The final custom validation requires:

```text
VLLM_ASCEND_REGISTER_CUSTOM_MODELS=1
Qwen3MoeForCausalLM -> vllm_ascend.models.qwen3_moe:CustomQwen3MoeForCausalLM
Custom Qwen3 MoE -> vllm_ascend.ops.fused_moe.AscendFusedMoE
```

Also do not force `ALLTOALL` as a workaround. Successful runs kept MC2 decode.

## Expected Logs

Successful floor8/floor4 custom mode1 runs should show:

```text
Mode1 parity init: ... hybrid_disabled=True runtime_buffers_disabled=True parity_ready=True
Skipping generic post-restore headroom for mode1 lightweight parity
Skipping post-shrink MoE dispatch headroom for mode1 lightweight parity
Skipping post-shrink prefill AllToAll headroom for mode1 lightweight parity
Skipping first-live-prefill headroom for mode1 lightweight parity
Skipping mode=1 KV materialization headroom for lightweight parity
Elastic post-shrink MoE dispatch warmup skipped: ... mode1_lightweight_parity_no_synthetic_warmup
Elastic parallel stale MC2 cache dropped across restore
```

And should not show:

```text
Applying post-restore DP headroom
Applying post-restore EP headroom
Applying post-restore MoE dispatch headroom
Applying post-shrink MoE dispatch headroom
Applying first-live-prefill headroom
Applying custom mode=1 KV materialization headroom
HcclAlltoAllV fallback as the main decode path
```

## Validation Checklist

For custom floor8:

```text
VLLM_ASCEND_REGISTER_CUSTOM_MODELS=1
VLLM_ASCEND_ELASTIC_EXECUTION_MODE=1
VLLM_ASCEND_ELASTIC_MIN_COMPUTE_GROUP_SIZE=8
all generic headroom env vars unset or 0
expected KV: 377,344 tokens
expected max concurrency: 21.68x
expected result: 3/3, exit_code=0
```

For custom floor4:

```text
VLLM_ASCEND_REGISTER_CUSTOM_MODELS=1
VLLM_ASCEND_ELASTIC_EXECUTION_MODE=1
VLLM_ASCEND_ELASTIC_MIN_COMPUTE_GROUP_SIZE=4
all generic headroom env vars unset or 0
expected KV: 277,120 tokens
expected max concurrency: 15.92x
expected result: 3/3, exit_code=0
```

## Short Migration Order

Apply the changes in this order when porting to an older codebase:

1. Keep `qwen3_moe.py` on old `vllm_ascend.ops.fused_moe.AscendFusedMoE`.
2. Add floor-target `loaded_weight_capacity` before `create_weights`.
3. Disable mode1 hybrid / CPU-shadow / mode3 state.
4. Add and log `lossless_mode1_native_parity_ready`.
5. Replace runtime-buffer activation with loaded-slot prefix views.
6. Direct P2P imported experts into loaded slots during shrink.
7. Restore full-world layout using loaded-slot prefix views.
8. Gate all headroom skips on `lossless_mode1_native_parity_ready`.
9. Skip synthetic post-shrink MoE dispatch warmup for lightweight mode1.
10. Keep full-world `_EP` cache, drop stale MC2 cache, and reuse current MC2 signatures.
11. Add opt-in KV/non-torch memory diagnostics.
12. Add floor8/floor4 native KV token caps.

After step 12, floor8 and floor4 should behave like native in memory cost while
still executing through the old custom fused MoE operator stack.
