# System Contributions and Design Implications

Date: 2026-07-06

This note summarizes the system-level contributions distilled from the
floor2/floor4 shrink-aware rollout debugging.  The goal is to convert the
engineering lessons into paper-ready claims and design principles.

## Contribution 1: Communication-Workspace-Aware KV Planning

Shrink-aware rollout cannot size the KV cache using only model weights, runtime
activations, and the nominal decode workspace.  In elastic distributed
execution, communication libraries may allocate large workspaces lazily, and the
allocation point can occur after the KV cache has already consumed the apparent
free memory.

The floor2/floor4 probe exposed this issue clearly.  The floor4 KV cache budget
was feasible by itself, and the rollout weights could also be loaded under the
intended ordering.  However, the first post-shrink execution still failed when a
small DP metadata synchronization triggered an NPU HCCL allreduce workspace
allocation:

```text
_sync_metadata_across_dp
current working operator name is HcclAllreduce
Failed to allocate [size:1678770176] bytes of NPU memory
```

The failed collective synchronized only small metadata values, but it requested
about 1.68 GB of device-side HCCL workspace.  This invalidated the memory plan
even though the KV cache and model state were individually feasible.

Design implication:

> KV-cache planning for elastic LLM rollout must account for communication
> workspaces, including lazily allocated HCCL/MC2/TBE runtime buffers.  A memory
> configuration is not feasible unless all post-shrink communication workspaces
> are either pre-warmed before KV sizing or explicitly excluded from the critical
> device-memory path.

## Contribution 2: Control-Plane and Data-Plane Communication Separation

The debugging process showed that not all distributed communication should be
treated uniformly.  Large tensor collectives, MoE token dispatch, and expert
all-to-all are data-plane communication and should use the high-throughput NPU
communication path.  In contrast, request metadata synchronization, token-count
exchange, and step-state coordination are control-plane communication.

Routing control-plane metadata synchronization through NPU HCCL can introduce
large hidden workspace allocations with no meaningful throughput benefit.  We
therefore route mode1 DP metadata synchronization through the CPU process group:

```text
VLLM_ASCEND_MODE1_CPU_DP_METADATA_SYNC=1
```

This removes the hidden NPU HCCL workspace allocation from the critical memory
path.  In the validated `2 -> 2 -> 4 -> 4 -> 16` probe, the run completed all
five steps without the previous HCCL OOM, and step2-step4 did not slow down:

```text
step1 floor2  gen = 111.18 s
step2 floor2  gen =  97.10 s
step3 floor4  gen =  95.52 s
step4 floor4  gen =  93.85 s
step5 floor16 gen =  66.34 s
```

Design implication:

> Elastic inference systems should separate control-plane synchronization from
> data-plane collectives.  Small metadata exchanges should use CPU-side process
> groups when device-side collectives would introduce large workspace overheads.

Paper-ready statement:

> We identify a hidden memory coupling between control-plane collectives and
> data-plane KV-cache allocation.  Although metadata synchronization exchanges
> only a few integers, using device-side HCCL can lazily allocate GB-scale
> workspace and invalidate an otherwise feasible KV budget.  Our system routes
> control-plane metadata synchronization through CPU process groups while
> reserving NPU collectives for high-volume MoE and tensor communication.

## Contribution 3: Step-Level Memory Invariants for Dynamic Floors

The dynamic `2 -> 2 -> 4 -> 4` probe also revealed that memory invariants must
be enforced at every rollout step, not only when the floor changes.  Same-floor
steps can still reload weights, rebuild runtime state, and recreate expert
storage.  If the reload path silently restores a larger floor2-style expert
capacity during a floor4 step, the next KV allocation can fail even though no
floor transition occurred.

The required invariant is:

```text
after update_weights:
  floor2  -> loaded_capacity = 64
  floor4  -> loaded_capacity = 32
  floor8  -> loaded_capacity = 16
  floor16 -> loaded_capacity = 8
```

This invariant must be tied to the planned step floor and enforced immediately
after every weight reload.  It cannot be restricted to upward cleanup paths such
as `2 -> 4` or `4 -> 8`, because a same-floor step can still refresh parameters
and recreate a larger storage layout.

Design implication:

> Dynamic shrink-aware rollout should treat the target floor as a step-level
> memory contract.  Weight reload, expert storage layout, dispatcher metadata,
> communication-group lifecycle, and KV-cache sizing must all be derived from
> the same target floor.

## Contribution 4: Transient-Peak-Aware Restore Ordering

The first floor4 failure was not caused by the final floor4 state being
infeasible.  It was caused by a transient peak in the restore sequence:

```text
resume large KV cache
then resume rollout weights
```

Building the full floor4 KV cache before rollout weights were onloaded left
insufficient memory for formatted NPU weight buffers.  Reordering the staged
path so that KV resume can bootstrap with a smaller cap and then expand through
adaptive resize avoided this peak.

Design implication:

> Elastic restore paths should minimize transient peaks, not only final-state
> memory.  A feasible final memory layout can still fail if KV cache and
> formatted weight buffers overlap during restoration.

Paper-ready statement:

> We make restore ordering explicit in the memory scheduler.  Instead of
> restoring the final KV cache before weight onload, the system can bootstrap
> with a smaller KV reservation, reload weights under a lower peak, and then
> resize the KV cache to the target floor budget before generation.

## Contribution 5: Warmup as Memory Disclosure, Not Only Latency Hiding

Post-shrink warmup is often treated as a performance optimization for reducing
first-token latency.  The floor2/floor4 issue shows another role: warmup can
force lazy runtime allocations to appear before KV sizing.

If an HCCL/MC2/TBE workspace is first allocated during real decode, it can
violate the memory budget after the KV cache has already been committed.  A
communication-aware warmup can convert hidden runtime demand into explicit
memory demand.

Design implication:

> Warmup should be viewed as a memory-disclosure mechanism.  Critical
> communication paths should be warmed before final KV-cache sizing, or moved
> out of the device-memory critical path when they belong to the control plane.

## Summary of Paper-Level Claims

The floor2/floor4 debugging suggests the following system claims:

1. Shrink-aware RL rollout requires communication-workspace-aware KV planning.
2. Control-plane metadata synchronization should be decoupled from NPU
   data-plane collectives.
3. Target floor must be enforced as a step-level memory invariant across weight
   reload, expert storage, dispatcher state, and KV cache.
4. Restore ordering must minimize transient memory peaks, not just final
   resident memory.
5. Warmup serves as memory disclosure for lazy communication/runtime
   allocations, in addition to reducing first-use latency.

These points turn several debugging fixes into a coherent design principle:

> Elastic LLM rollout is a joint scheduling problem over compute floors,
> communication groups, expert storage, and KV cache.  Correctness and
> performance depend on making every hidden device-memory consumer explicit
> before committing the KV budget.
