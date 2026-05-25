# Mode 3 Prefetch/Compute Overlap

This note summarizes the core implementation of elastic execution `mode=3`
for Qwen3-30B-A3B MoE inference in vLLM-Ascend. The relevant code is mainly in
`vllm_ascend/ops/fused_moe.py`.

## High-level Flow

Mode 3 overlaps current-layer MoE compute with next-layer expert weight
prefetch. The hot path is:

```text
enter layer i on each active rank
  -> bind_current_layer(i)
       wait for layer i runtime expert weights to be ready
       bind runtime_w13/runtime_w2/expert_map to the layer
  -> prefetch_next_layer(i)
       asynchronously prepare layer i+1 weights into the alternate slot
  -> execute layer i MoE compute
       token dispatch + expert MLP + token combine, or fused single-wave path
enter layer i+1
  -> usually hit the prefetched double-buffer slot
```

The overlap is implemented with NPU streams and events, not Python threads:

- current compute stream: runs the current layer's MoE computation.
- `prefetch_stream`: copies NPU-resident experts into the next runtime slot.
- `cpu_prefetch_stream`: copies CPU-shadow experts into NPU staging or runtime slots.
- `slot.ready_event`: connects prefetch completion to the next layer's compute stream.

## Entry Conditions

The mode 3 cross-layer double-buffer path is selected only when the layer is in
hybrid lossless mode and the communication path is compatible with MC2:

```python
if getattr(layer, "lossless_hybrid_active", False):
    if self._should_use_mode3_cross_layer_buffer_path(...):
        return self._execute_mode3_single_dispatch_hybrid(...)
```

The gate for this path checks:

```python
getattr(layer, "elastic_execution_mode", 0) == 3
getattr(layer, "lossless_hybrid_active", False)
shared_experts is None
forward_context / token_dispatcher uses MC2
```

Code references:

- `AscendUnquantizedFusedMoEMethod._should_use_mode3_cross_layer_buffer_path`
- `AscendUnquantizedFusedMoEMethod.apply` mode 3 dispatch branch

## Double-buffer Manager

Mode 3 uses one `Mode3DoubleBufferManager` per forward context. It owns two
runtime slots and chooses the slot by layer parity:

```python
self.slots = [_Mode3DoubleBufferSlot(), _Mode3DoubleBufferSlot()]
slot_id = int(layer.layer_idx) & 1
```

The manager is created from the forward context:

```python
prefetch_stream = getattr(forward_context, "moe_prefetch_stream", None)
model_instance = getattr(forward_context, "model_instance", None)
manager = Mode3DoubleBufferManager(model_instance, prefetch_stream)
forward_context.moe_double_buffer_manager = manager
```

Important mode 3 switches:

```text
VLLM_ASCEND_MODE3_ASYNC_NPU_PREFETCH
VLLM_ASCEND_MODE3_ASYNC_CPU_STAGE
VLLM_ASCEND_MODE3_ASYNC_CPU_PACK
VLLM_ASCEND_MODE3_DIRECT_CPU_SLOT
VLLM_ASCEND_MODE3_DEVICE_READY_WAIT
VLLM_ASCEND_MODE3_BULK_NPU_COPY
VLLM_ASCEND_MODE3_BULK_CPU_STAGE
VLLM_ASCEND_MODE3_BULK_CPU_DIRECT
VLLM_ASCEND_MODE3_LAYER_LOCAL_BUFFER
VLLM_ASCEND_MODE3_USE_FUSED_EXPERTS_PATH
```

## Per-rank Expert Ownership

Each active rank builds a local runtime expert buffer from the experts assigned
to that rank. The ordered expert list comes from the layer's hybrid state:

```python
rank_owned = getattr(layer, "lossless_hybrid_rank_owned_expert_ids", None)
local_rank_idx = getattr(layer, "lossless_hybrid_active_rank_index", -1)
local_owned_expert_ids = rank_owned[local_rank_idx]
```

During slot population, each expert is classified into one of two sources:

```python
primary_slots = layer.lossless_mode3_primary_prefix_local_slots
cpu_shadow_slots = layer.lossless_cpu_shadow_local_slots

for slot_idx, expert_id in enumerate(slot_expert_ids):
    source_local_slot = primary_slots.get(int(expert_id))
    if source_local_slot is not None:
        npu_assignments.append((slot_idx, int(source_local_slot)))
    else:
        cpu_slot = cpu_shadow_slots.get(int(expert_id))
        cpu_assignments.append((slot_idx, int(cpu_slot)))
```

- `npu_assignments`: expert rows already resident on this rank's NPU.
- `cpu_assignments`: expert rows that must be fetched from CPU shadow weights.

## Prefetch Submission

`prefetch_next_layer(current_layer)` finds the next layer and calls
`prepare_slot(next_layer, async_copy=True)`.

The essential async prefetch code is:

```python
current_stream = torch.npu.current_stream()

if self.cpu_prefetch_stream is not None:
    self.cpu_prefetch_stream.wait_stream(current_stream)

with torch.npu.stream(self.prefetch_stream):
    self.prefetch_stream.wait_stream(current_stream)

    self._populate_slot(
        slot,
        layer,
        async_copy=True,
        cpu_prefetch_stream=self.cpu_prefetch_stream,
        reason=reason,
    )

    if slot.has_async_cpu_direct or slot.has_async_cpu_pack:
        self.prefetch_stream.wait_event(slot.cpu_pack_event)
    elif slot.has_async_cpu_copy:
        self.prefetch_stream.wait_event(slot.cpu_ready_event)
        # pack CPU staging rows into final runtime slots here

    slot.ready_event.record()

slot.inflight_prefetch = True
```

This submits weight movement for the next layer and returns control to the
current-layer compute path without host synchronization.

## NPU-resident Expert Copy

NPU-resident rows are copied from the layer's resident prefix buffer into the
mode 3 runtime slot:

```python
self._ensure_slot_capacity(slot, layer)

self._copy_npu_assignment_runs(
    slot.w13,
    layer.w13_weight,
    npu_assignments,
    async_copy=async_copy,
)
self._copy_npu_assignment_runs(
    slot.w2,
    layer.w2_weight,
    npu_assignments,
    async_copy=async_copy,
)
```

When `VLLM_ASCEND_MODE3_LAYER_LOCAL_BUFFER=1`, the implementation can skip this
runtime copy and reuse the layer-local resident prefix buffer directly, but only
when every destination slot matches the source slot.

## CPU-shadow Expert Copy

CPU-only experts have three possible paths.

Direct CPU-to-runtime slot path:

```python
with torch.npu.stream(cpu_prefetch_stream):
    self._copy_cpu_assignments_to_runtime_direct(
        slot,
        cpu_assignments,
        cpu_w13,
        cpu_w2,
    )
    slot.cpu_pack_event.record()
```

CPU-to-NPU-stage plus async pack path:

```python
with torch.npu.stream(cpu_prefetch_stream):
    self._copy_cpu_assignments_to_stage(
        slot,
        cpu_assignments,
        cpu_w13,
        cpu_w2,
        async_copy=True,
    )

    if pack_to_runtime:
        self._copy_npu_assignment_runs(slot.w13, slot.cpu_stage_w13, stage_to_slot)
        self._copy_npu_assignment_runs(slot.w2, slot.cpu_stage_w2, stage_to_slot)
        slot.cpu_pack_event.record()
    else:
        slot.cpu_ready_event.record()
```

Deferred bind-time fill path:

```python
slot.needs_sync_cpu_fill = True
```

In the deferred path, `bind_current_layer()` later calls `_fill_pending_cpu_rows()`.

## Current-layer Binding

When a layer starts, `bind_current_layer(layer)` selects the expected slot. If
the slot was not prefetched or no longer matches the layer state, it falls back
to synchronous population:

```python
if not self._slot_matches(slot, layer):
    slot = self.prepare_slot(layer, slot_id, async_copy=False,
                             reason="sync_current")
```

If the slot is inflight, the current compute stream waits for the slot's ready
event:

```python
if slot.inflight_prefetch and self.enable_device_ready_wait:
    current_stream = torch.npu.current_stream()
    current_stream.wait_event(slot.ready_event)
    wait_mode = "device_event"
else:
    slot.ready_event.synchronize()
```

Then the runtime expert buffers are bound to the layer:

```python
layer.runtime_w13_weight = slot.w13[:slot.valid_expert_count]
layer.runtime_w2_weight = slot.w2[:slot.valid_expert_count]
layer.expert_map = slot.expert_map
layer.active_local_num_experts = slot.valid_expert_count
layer.local_num_experts = slot.valid_expert_count
layer.moe_config.num_local_experts = slot.valid_expert_count
layer.moe_config.num_experts = slot.valid_expert_count
layer.lossless_runtime_dispatch_log2phy = slot.dispatch_log2phy
layer.lossless_runtime_dispatch_num_experts = slot.dispatch_num_experts
```

`current_stream.wait_event(slot.ready_event)` is the key dependency: it protects
correctness while still letting prefetch work overlap with previous-layer compute.

## Compute Path

The fused-experts mode 3 path does:

```python
bound_slot = manager.bind_current_layer(layer)
next_prefetch_timing = manager.prefetch_next_layer(layer)

final_hidden_states = self._execute_single_wave(
    layer=layer,
    hidden_states=x,
    logical_topk_ids=logical_topk_ids,
    topk_weights=topk_weights,
    global_num_experts=dispatch_num_experts,
    log2phy=dispatch_log2phy,
    ...
)
```

The non-fused single-dispatch path follows the same ordering:

```python
bound_slot = manager.bind_current_layer(layer)
next_prefetch_timing = manager.prefetch_next_layer(layer)

dispatch_results = token_dispatcher.token_dispatch(...)
dispatched_output = unified_apply_mlp(
    hidden_states=dispatched_hidden_states,
    w1=layer.runtime_w13_weight,
    w2=layer.runtime_w2_weight,
    group_list=dispatched_group_counts,
    ...
)
final_hidden_states = token_dispatcher.token_combine(dispatched_output)
```

The ordering is intentional: next-layer prefetch is submitted before the current
layer's expensive MoE compute starts, so the prefetch streams can make progress
while the current stream is running dispatch/MLP/combine.

## Mental Model

For each active rank:

```text
layer i slot = i % 2
layer i+1 slot = (i + 1) % 2

time ---->

compute stream:   wait slot i ready | compute layer i ------------------------>
prefetch stream:                    | copy NPU rows for layer i+1 | ready_event
cpu stream:                         | copy CPU rows/stage/pack for layer i+1
```

If prefetch finishes before layer `i+1` starts, `bind_current_layer(i+1)` becomes
a cheap event wait or a near-zero hit. If not, the current stream waits on the
same `ready_event`, preserving correctness.

## Useful Debug Logs

Relevant log strings in `fused_moe.py`:

```text
Mode3 prefetch scheduled
Mode3 slot binding
Mode3 fused-experts execution
Mode3 single-dispatch execution
Mode3 timing fused-experts
Mode3 timing single-dispatch
```

Useful timing fields:

```text
bind_wait_us
bind_cpu_fill_us
ready_wait_dev_ms
prefetch_submit_us
prefetch_dev_ms
prefetch_npu_dev_ms
prefetch_cpu_dev_ms
prefetch_cpu_pack_dev_ms
current_compute_dev_ms
prefetch_minus_compute_dev_ms
source_from_npu
source_from_cpu
prefetch_cpu_path
```

