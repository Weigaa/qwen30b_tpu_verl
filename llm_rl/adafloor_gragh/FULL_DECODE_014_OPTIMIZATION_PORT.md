# vLLM 0.14 runtime optimizations ported to the 0.11 FULL graph path

## Scope

The graph body remains the validated 0.11 native `FULL_DECODE_ONLY` ACLGraph
implementation with FIA maximum-workspace Attention, KV write, MoE/HCCL, and
dense layers captured together. This change ports compatible runtime and
scheduler optimizations from the fastest local 0.14 Qwen3 graph run. It does
not replace the 0.11 graph implementation with TorchAir or GraphEX.

## Ported optimizations

1. **Native CaMem sleep level 1.** Vanilla Full16 can retain stable parameter
   and KV-cache virtual addresses across rollout/training phase changes instead
   of replacing every parameter with manually allocated storage.
2. **Guarded graph reuse after a weight refresh.** Parameter and KV tensor
   address signatures are checked at runtime. Stable addresses reuse the
   captured graph. Any address change fails over to the existing safe graph
   invalidation and late-recapture path.
3. **Async v1 scheduling.** The 0.11 rollout schema and launcher now expose
   `async_scheduling`, rather than silently omitting the 0.14 setting.
4. **Prefix caching.** The rollout no longer hard-codes prefix caching off.
   This matters for GRPO's multiple samples of the same prompt.
5. **Chunked prefill end-to-end.** Both vLLM and the Ascend scheduler now see
   the same requested value instead of the Ascend scheduler forcing it off.
6. **Filtered custom OPP selection.** The optimized launcher injects the
   content-hashed filtered 0.14 OPP bundle. Graph-required local operators are
   retained while MoE dispatch/combine resolve to the faster system CANN
   implementations.
7. **Stable allocator policy and existing fast 0.11 kernels.** The optimized
   profile keeps `TASK_QUEUE_ENABLE=1`, disables expandable segments for the
   graph/native-sleep lifecycle, retains MC2, and enables the existing 0.11
   custom TopK/TopP sampler.

All additions are opt-in. Existing eager, AdaFloor, and baseline FULL graph
launchers retain their previous defaults.

## Deliberately not claimed as ported

- The 0.14 GraphEX/Inductor graph-pass and fusion pipeline depends on compiler
  interfaces absent from vLLM/vLLM-Ascend 0.11. Setting the 0.14 fusion
  environment variables in 0.11 would be a no-op, so this port does not do so.
- Native CaMem sleep is initially fail-closed for elastic shrink. Dynamic
  communicator and expert-layout remapping still uses the validated manual
  AdaFloor lifecycle. The optimized test entrypoint is therefore Vanilla
  Full16 only.
- The filtered OPP is a CANN runtime artifact shared with the local 0.14 tree,
  not copied source code. Its resolved path and bundle hash are recorded in
  the experiment protocol.

## Validation boundary

Static and unit tests validate configuration propagation, fail-closed gates,
pointer-guard fallback, launcher syntax, and dry-run contracts. Performance and
numerical parity still require the five-step NPU run. No speedup should be
claimed until that run passes and is compared against the same workload.
