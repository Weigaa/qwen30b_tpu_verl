# qwen3 Eager Rollout Performance Experiments

This file records qwen3 eager rollout perf experiments so we do not repeat
the same probes.  Before starting a new eager perf run, read this file and
only vary one clear hypothesis.

## Target

Reference old-stack log:

`../ref_wj_qwen3/qwen30b_tpu_verl/resample_result_16k_bs32_n16_baseline_ft/logs/wjqwen30b-a3b-record_graph_save4eagle3_20260507090220.txt`

Stable threshold-control target:

- `timing_s/generate_sequences`: about `56-58s`
- `timing_s/gen` / `rollout_output_time_s`: about `82-84s`
- `timing_per_token_ms/gen`: about `0.36-0.37ms`

## Validation Funnel

Use the cheapest experiment that can answer the current question:

1. Syntax / config inspection first.
2. One-step rollout-only / generate-only with targeted debug logs.
3. Three-step threshold-control run only when a knob shows promise.
4. Full 16k run only after threshold-control behavior is stable.

Do not run a full 16k validation for every hypothesis.  For profiling, stop a
run once it has reached the length bucket or stage needed to answer the
question.

Use `internal/run_eager_perf_probe.sh` to keep probe conditions identical:

```bash
# Fastest reliable A/B: one threshold-control generate-only step.
bash internal/run_eager_perf_probe.sh gen1

# Same workload with runner/MoE/decode-attention profile logs.
bash internal/run_eager_perf_probe.sh profile

# Only after gen1/profile looks promising.
bash internal/run_eager_perf_probe.sh threshold3

# Only after threshold3 is stable.
bash internal/run_eager_perf_probe.sh full16k
```

Prefer `OUTPUT_SUBDIR=... bash internal/run_eager_perf_probe.sh <mode>` when
recording a named hypothesis.  The wrapper delegates production configuration
to `wj_train_grpo_qwen30b_a3b_16die_true_weight_regroup_eager_fast.sh`, then
parses the newest log with `tools/profile_rollout_logs.py`.

## Current Best Behavior-Preserving qwen3 Eager Path

Threshold-control log:

`resample_result_16k_bs32_n16_eager_nolocalopp_threshold3_20260516003739/logs/wjqwen30b-a3b-record_graph_save4eagle3_20260516003740.txt`

Config highlights:

- `VLLM_ASCEND_EAGER_COMPILE=1`
- `VLLM_ASCEND_USE_LOCAL_CUSTOM_OPP=0`
- new split-MoE, not legacy fused MoE
- `VLLM_ASCEND_MC2_TOKENS_CAPACITY=512`
- `VLLM_ASCEND_ENABLE_FUSED_MC2=0`
- `VLLM_ROLLOUT_FORCE_ELASTIC_MOE_POLICY=1`
- `VLLM_ASCEND_ATTENTION_BLOCK_SIZE=64`
- `VLLM_ASCEND_FORCE_DIRECT_ATTENTION_IMPL=0`
- `VLLM_ASCEND_EAGER_METADATA_SYNC_DEVICE=1`
- `VLLM_ASCEND_FUSED_MOE_SIMPLE_MC2=1`
- `VLLM_ROLLOUT_ASYNC_SCHEDULING=true`
- `VLLM_ROLLOUT_ENABLE_CHUNKED_PREFILL=true`
- `VLLM_ROLLOUT_MAX_NUM_BATCHED_TOKENS=17408`
- `VLLM_ROLLOUT_MANUAL_FREE_CACHE_ENGINE=1`

Threshold-control metric:

- Step 1: `generate_sequences=68.099s`, `gen=85.265s`,
  `rollout_output_time_s=85.269s`.
- Step 2: `generate_sequences=60.054s`, `gen=71.735s`,
  `rollout_output_time_s=71.738s`.
- Step 3: `generate_sequences=61.615s`, `gen=73.032s`,
  `rollout_output_time_s=73.035s`.

Status: fastest threshold-control eager path so far.  It beats the old
threshold-control rollout target (`82-84s`) and brings pure
`generate_sequences` close to the old range (`56-58s`).  A full-16k validation
is still required before claiming end-to-end full-run parity with the old stack.

Latest behavior-preserving profile, after reverting direct-attention from the
production eager wrapper:

- Log:
  `resample_result_16k_bs32_n16_eager_behavior_base_profile_20260515/logs/wjqwen30b-a3b-record_graph_save4eagle3_20260515134527.txt`
- Step metric: `generate_sequences=94.760s`, `gen=112.082s`,
  `rollout_output_time_s=112.085s`, `speed=2371.485 tok/s`,
  `total_tokens=265808`.
- Decode runner profile: `forward=91.632ms`, `sample=1.961ms`,
  `prepare=2.877ms`, `post=0.574ms` over 7024 records.
- Real small-token MoE MC2 profile (`call>1`, `tokens=32`):
  `dispatch=0.863ms`, `mlp=0.096ms`, `combine=0.139ms`,
  `total=1.098ms` over 2672 records.
- Real 512-token MC2 profile (`call>1`, `tokens=512`):
  `dispatch=1.333ms`, `mlp=0.482ms`, `combine=0.230ms`,
  `total=2.046ms` over 496 records.
- Decode FIA op remained small:
  `DecodeOnly/fia/tokens<=32 call>1=0.093ms`.

Interpretation: this is the current comparable baseline for further code
experiments.  The gap is again in decode forward (`~91.6ms` here versus old
`~60-62ms`), with MC2 dispatch/combine a real but insufficient contributor.
Continue with real-path Qwen3 eager MoE / decoder-layer internals; do not
promote direct-attention results unless full16k behavior matches the old
response distribution.

Custom-op dispatch sanity check, 2026-05-15:

- Log:
  `resample_result_16k_bs32_n16_eager_customop_probe_20260515141248/logs/wjqwen30b-a3b-record_graph_save4eagle3_20260515141302.txt`
- Added `VLLM_QWEN3_EAGER_CLASS_PROBE=1` to print the actual instantiated
  Qwen3 eager classes and selected OOT custom-op registry once per worker.
- Result: the production eager path is not silently falling back to native
  high-level vLLM model modules.  The real classes are Ascend replacements for
  the hot modules:
  `AscendVocabParallelEmbedding`, `AscendParallelLMHead`,
  `AscendLogitsProcessor`, `AscendRMSNorm`, `AscendQKVParallelLinear`,
  `AscendRowParallelLinear`, `AscendRotaryEmbedding`, custom attention backend
  `AscendAttentionBackendImpl`, and `AscendFusedMoE` with
  `AscendUnquantizedFusedMoEMethod`.  The engine also prints
  `CompilationMode.VLLM_COMPILE`, backend
  `vllm_ascend.compilation.compiler_interface.AscendCompiler`, and
  `Using OOT custom backend for compilation`.
- Linear wrapper detail: `qkv_custom_op=None` and `o_proj_custom_op=None` on
  the TP=1 rollout path.  This is not by itself evidence of a bad fallback:
  both new and old TP=1 eager linear-op selection only attach extra
  `linear_op.py` wrappers for SP / OProj TP / MLP TP / matmul-allreduce cases.
  With TP=1 and no SP, attention QKV/O projections normally run through the
  layer's unquantized linear method.  Therefore the remaining performance gap
  should not be framed as "the main Ascend CustomOp registry was not used";
  it is more likely inside the concrete Ascend unquantized matmul, attention
  wrapper/backend, or split-MoE dispatch/combine implementation selected by
  those Ascend classes.
- Next diagnostic, if needed: use branch-level logs
  (`VLLM_ASCEND_MOE_COMM_DEBUG=1`,
  `VLLM_ASCEND_ATTENTION_STAGE_DEBUG=1`) rather than rechecking class
  replacement.

Branch-level custom-op sanity check, 2026-05-15:

- Log:
  `resample_result_16k_bs32_n16_eager_branchprobe_202605151418/logs/wjqwen30b-a3b-record_graph_save4eagle3_20260515141833.txt`
- Result: the current eager path is not accidentally selecting the wrong MoE
  communication family.  With `VLLM_ASCEND_MC2_TOKENS_CAPACITY=512`, observed
  dispatch was `512 -> MC2`, large prefill chunks such as `17408` and `4016`
  -> `ALLTOALL`, and decode-ish `32 -> MC2`.  This matches the old intended
  `<=512` MC2 / `>512` AllToAll policy.
- Result: current best eager decode attention is not silently falling back to a
  generic Python/paged compatibility path.  Branch logs show DecodeOnly mainly
  selecting `fia`, while long prefill chunks use FIA as well.
- Important interpretation: this does not mean the attention path matches old
  eager.  The real old eager `attention_v1.py` DecodeOnly path used
  `_npu_paged_attention`; the current qwen3 eager default is FIA-first unless
  `VLLM_ASCEND_USE_LEGACY_ATTENTION=1` or the paged-attention selector forces
  otherwise.  So the remaining question is no longer "did custom ops fail to
  load?" but "is the new FIA-first wrapper/backend slower for this Qwen3 decode
  workload than the old paged-attention eager path?"

Residual RMSNorm old-op probe, 2026-05-15:

- Env: `VLLM_ASCEND_FORCE_TORCH_NPU_ADD_RMS_NORM=1`, log:
  `resample_result_16k_bs32_n16_eager_gen1_torchnpu_addrmsnorm_20260515/logs/wjqwen30b-a3b-record_graph_save4eagle3_20260515150857.txt`.
- Hypothesis: graph/new-stack changes made the eager residual RMSNorm path use
  `_C_ascend.npu_add_rms_norm_bias(...)`, while old eager used
  `torch_npu.npu_add_rms_norm(...)`; switching back might reduce the
  `input_norm` / `post_norm` layer-profile gap.
- Result: no clear win.  The one-step rollout summary was
  `rollout_output_time_s=99.250s`, `speed=2678.159 tok/s`, with the same
  threshold token count `265,808`.  This is outside the current-best threshold
  band and did not produce a better `generate_sequences` record.
- Keep the env-gated code available for diagnostics only; do not add it to
  `eager_fast`.  The remaining behavior-preserving gap is still more strongly
  indicated by the attention wrapper/impl region than by residual RMSNorm.

Unquantized GEMM custom-op bypass probe, 2026-05-15:

- Env: `VLLM_ASCEND_DISABLE_UNQUANTIZED_GEMM_CUSTOMOP=1`, log:
  `resample_result_16k_bs32_n16_eager_gen1_disable_unquant_gemm_customop_20260515/logs/wjqwen30b-a3b-record_graph_save4eagle3_20260515212131.txt`.
- Hypothesis: new v0.14 patches `default_unquantized_gemm` into a
  `torch.ops.vllm.unquantized_gemm` custom-op while the old eager stack used
  the plain unquantized GEMM path; bypassing that custom-op might reduce the
  small-token QKV/O/router dense-linear overhead.
- Result: no benefit.  One-step threshold generate-only produced
  `rollout_output_time_s=101.151s`, `generate_sequences=84.003s`,
  `gen=101.148s`, `speed=2627.824 tok/s`, with the same threshold token count
  (`265,808`).  This is slower than the current best behavior-preserving
  threshold band.
- Interpretation: the production gap should not be attributed to the
  unquantized GEMM custom-op wrapper alone.  Continue with the attention
  wrapper/backend and real Qwen3 eager MoE runtime differences.

Direct-attention diagnostic:

- `VLLM_ASCEND_FORCE_DIRECT_ATTENTION_IMPL=1` gave the best threshold-control
  result so far:
  `resample_result_16k_bs32_n16_eager_direct_attn_impl_threshold3_20260515/logs/wjqwen30b-a3b-record_graph_save4eagle3_20260515112931.txt`
  with step2/3 `generate_sequences=65.077/64.487s`,
  `gen=76.723/75.852s`, and
  `rollout_output_time_s=76.732/75.855s`.
- However its full-16k validation is not comparable:
  `resample_result_16k_bs32_n16_eager_direct_attn_impl_full16k_20260515/logs/wjqwen30b-a3b-record_graph_save4eagle3_20260515131959.txt`
  produced only `1,134,416` tokens, `mean~=2215.7`, and no responses reaching
  the 16k cap, while the old full-16k reference produced `3,069,063` tokens,
  `mean~=5994.3`, and 28 capped samples.  The apparent
  `rollout_output_time_s=766.436s` is therefore not a valid performance win.
- Keep this knob as an explicit diagnostic only until a same-workload full16k
  run proves behavior-preserving.

Wrapper validation:

- `resample_result_16k_bs32_n16_eager_bestwrapper_genonly_20260514` pinned the
  same current-best knobs in `wj_train_grpo_qwen30b_a3b_16die_true_weight_regroup_async_oldcfg.sh`
  and ran `VLLM_ROLLOUT_DEBUG_GENERATE_ONLY=1`.
- Step2/3 were `generate_sequences=80.4520/80.8854s`,
  `gen=92.5157/92.7158s`, so the wrapper carries the intended knobs but still
  lands a little slower than the previous best run.  Treat the previous best as
  the high-water mark and keep investigating MoE internals / hidden run drift.

## Confirmed Preserved Behavior

MC2/AllToAll threshold is preserved in current qwen3 for A3 + EP when
`VLLM_ASCEND_ENABLE_FUSED_MC2=0`:

- `num_tokens <= VLLM_ASCEND_MC2_TOKENS_CAPACITY` -> `MC2`
- `num_tokens > VLLM_ASCEND_MC2_TOKENS_CAPACITY` -> `ALLTOALL`

With `VLLM_ASCEND_MC2_TOKENS_CAPACITY=512`, this matches the old intended
`<=512` MC2 behavior.

Runtime validation:

- `resample_result_16k_bs32_n16_eager_genonly_moecomm_debug_20260514/logs/wjqwen30b-a3b-record_graph_save4eagle3_20260514153825.txt`
  enabled `VLLM_ASCEND_MOE_COMM_DEBUG=1` on the current-best eager stack.
  The observed threshold-control first step had 512 comm-selection records
  across 16 workers: `MC2=480`, `ALLTOALL=32`.  Per worker the early pattern
  was two `512 -> MC2` calls, two large prefill calls (`17408` / `4016`) using
  `ALLTOALL`, then 28 small decode-ish `32 -> MC2` calls.  This confirms the
  MC2/AllToAll selector is behaving as intended; the remaining gap is not an
  accidental fallback to AllToAll for small decode tokens.

## Useful Positive Findings

- `VLLM_ASCEND_EAGER_COMPILE=1` is required.  Without it, qwen3 eager becomes
  too pure-eager and loses vLLM compile wrapper benefits.
- `VLLM_ROLLOUT_FORCE_ELASTIC_MOE_POLICY=1` helps the new split-MoE eager path.
- `VLLM_ASCEND_ATTENTION_BLOCK_SIZE=64` gives a small additional improvement
  when stacked on the MoE policy path.
- `VLLM_ASCEND_EAGER_METADATA_SYNC_DEVICE=1` gives a small improvement
  (~0.5-1.0s), but does not close the large `generate_sequences` gap.
- `VLLM_ASCEND_FUSED_MOE_SIMPLE_MC2=1` improves the current best stacked
  config.  It avoids the generic dataclass/metadata-dict MC2 path in
  split-MoE and keeps combine metadata directly on the dispatcher.  Keep it
  enabled for the next round unless a later correctness issue appears.

## Profiling Findings

- Real old-stack eager Qwen3-MoE path, corrected 2026-05-15:

  Do not use `ref_wj_qwen3/qwen30b_tpu_verl/vllm_ascend/torchair/models/qwen3_moe.py`
  as the eager reference path.  That file is registered through the torchair
  graph helper and is relevant to graph/torchair mode, not the old eager
  baseline.  The actual old eager registry is
  `ref_wj_qwen3/qwen30b_tpu_verl/vllm_ascend/models/__init__.py`, which maps
  `Qwen3MoeForCausalLM` to
  `vllm_ascend.models.qwen3_moe:CustomQwen3MoeForCausalLM`.

  In that real old eager path, non-ACLGraph decoder layers replace the generic
  `Qwen3MoeSparseMoeBlock` with
  `vllm_ascend.models.qwen3_moe.CustomSparseMoeBlock`.  That block constructs
  `AscendFusedMoE(..., reduce_results=False, layer_idx=...)` and calls the
  experts with explicit old-style arguments:
  `is_prefill`, `top_k`, `enable_force_load_balance`, `shared_experts=None`,
  and `is_dummy`.

  The current qwen3 eager path instead uses
  `vllm/model_executor/models/qwen3_moe.py::Qwen3MoeSparseMoeBlock`, which by
  default constructs the generic v0.14 `FusedMoE` wrapper and calls
  `self.experts(hidden_states=..., router_logits=...)`.  Even when
  `VLLM_QWEN3_MOE_ASCEND_LEGACY_INIT=1` swaps the constructor to
  `AscendFusedMoE`, the forward path still goes through the newer v0.14
  CustomOp/generic `forward_impl` abstraction.  Future comparisons should stay
  on this real old-eager vs new-eager axis, not the torchair model file.

  Combined old-eager-like constructor probe:
  `VLLM_QWEN3_MOE_ASCEND_LEGACY_INIT=1` plus
  `VLLM_QWEN3_MOE_REDUCE_RESULTS=0`, log
  `resample_result_16k_bs32_n16_eager_oldref_moeblock_combo_20260515011727/logs/wjqwen30b-a3b-record_graph_save4eagle3_20260515011728.txt`.
  It was valid and did activate both envs, but it was slower:
  `generate_sequences=86.545s`, `gen=103.960s`,
  `rollout_output_time_s=103.964s`.  This rules out "constructor only" as the
  missing old eager performance.  The remaining suspect is the newer
  `AscendFusedMoE.forward_impl -> prepare -> quant_method.apply ->
  finalize(context_metadata)` abstraction versus the old direct
  `AscendFusedMoE.forward -> prepare -> quant_method.apply -> finalize` path.
  Next probes should profile or optimize that real internal difference.

- Matched old-vs-new Qwen3 layer profile, 2026-05-15:

  The old eager layerprofile was run on the real old eager Qwen3 path
  `vllm_ascend/models/qwen3_moe.py`, not the torchair graph model.  Log:
  `../ref_wj_qwen3/qwen30b_tpu_verl/resample_result_16k_bs32_n16_old_layerprofile_20260515/logs/wjqwen30b-a3b-record_graph_save4eagle3_20260515085424.txt`.
  The comparable new layerprofile log is
  `resample_result_16k_bs32_n16_eager_layerprofile_20260515004648/logs/wjqwen30b-a3b-record_graph_save4eagle3_20260515004648.txt`.

  Stable decode-small-token (`tokens=32`, call>1) comparison:

  | stage | old | new | delta |
  |---|---:|---:|---:|
  | attention total | `0.980ms` | `1.509ms` | `+0.529ms` |
  | sparse_moe total | `2.366ms` | `2.438ms` | `+0.072ms` |
  | decoder_layer total | `3.811ms` | `5.255ms` | `+1.444ms` |

  This narrows the next useful probe: the remaining decode gap is not
  MoE-only.  The new attention block is materially slower for the real decode
  token shape, even though standalone decode FIA op timing can look tiny.  A
  focused `VLLM_ASCEND_USE_LEGACY_ATTENTION=1` A/B is therefore warranted on
  top of the current best eager stack.  The production `eager_fast` wrapper
  now defaults legacy attention to `0` but allows env override so this probe is
  real rather than silently clobbered by the script.

  Direct model-level MoE-forward bypass was also tested separately with
  `VLLM_QWEN3_DIRECT_MOE_FORWARD_IMPL=1`.  The first direct call failed under
  compiled eager with `AssertionError: Unexpected type <class
  'vllm.model_executor.layers.fused_moe.config.FusedMoEQuantConfig'>`.  A
  follow-up tried to wrap the direct call in a no-Dynamo island via
  `torch._dynamo.disable`, log
  `resample_result_16k_bs32_n16_eager_direct_moe_nodynamo_gen1_20260515/logs/wjqwen30b-a3b-record_graph_save4eagle3_20260515115840.txt`.
  That also failed before performance metrics with
  `torch._dynamo.exc.Unsupported: Skip calling torch.compiler.disable()d
  function`.  This is not a MoE performance result; it only shows that a
  disabled-function island cannot be called from vLLM's compiled-eager wrapper.
  The broken no-Dynamo branch was removed.  Future MoE probes should either
  keep the call inside the compiled graph or swap the expert module/forward
  contract at construction time.

  Old-style Ascend Qwen3Moe expert-stack probe, 2026-05-15:

  `VLLM_QWEN3_MOE_ASCEND_LEGACY_STACK=1` replaces the current generic
  v0.14 `FusedMoE` stack with the old-style `vllm_ascend.ops.fused_moe_legacy`
  `AscendFusedMoE` module and calls it with the real old eager forward
  contract (`is_prefill`, `top_k`, `enable_force_load_balance`,
  `shared_experts=None`, `is_dummy`).  This is the first probe that actually
  swaps both constructor and forward contract rather than only the constructor.

  First run:
  `resample_result_16k_bs32_n16_eager_legacy_stack_gen1_20260515/logs/wjqwen30b-a3b-record_graph_save4eagle3_20260515121653.txt`.
  All 16 rollout workers logged `Using old-style Ascend Qwen3Moe experts
  stack`, but initialization failed during the vLLM dummy/profile run with
  `AttributeError(['prepare'])`.  Root cause: the old-style MoE forward was
  receiving the new fused-MoE comm-method object, whose interface is
  `prepare(...)->(hidden, logits, mask, context_metadata)` plus
  `finalize(..., context_metadata)`, while the old stack expects
  `prepare(...)->(hidden, logits)` and `finalize(hidden, reduce_results)`.

  Follow-up run after routing `VLLM_QWEN3_MOE_ASCEND_LEGACY_STACK=1` through
  the old `vllm_ascend.ops.moe.moe_comm_method` registry:
  `resample_result_16k_bs32_n16_eager_legacy_stack_commbridge_gen1_20260515/logs/wjqwen30b-a3b-record_graph_save4eagle3_20260515122118.txt`.
  This removed the explicit `prepare` mismatch, but the compiled dummy/profile
  run still failed before metrics with `AttributeError([])` hidden behind
  `torch._dynamo.exc.Unsupported`.  No performance conclusion can be drawn yet.
  The next diagnostic should run the same legacy stack with vLLM eager compile
  disabled to expose the real Python exception/stack before deciding whether
  this old-stack bridge is viable.

  Two subsequent compiled-eager bridge attempts also failed before metrics:

  - `resample_result_16k_bs32_n16_eager_legacy_stack_tokdisp_compile_gen1_20260515/logs/wjqwen30b-a3b-record_graph_save4eagle3_20260515125623.txt`
  - `resample_result_16k_bs32_n16_eager_legacy_stack_maskalign_compile_gen1_20260515/logs/wjqwen30b-a3b-record_graph_save4eagle3_20260515130128.txt`

  The first skipped the Python-side MC2 mask length guard in the token
  dispatcher.  The second additionally tried to pass the post-`prepare()`
  aligned `moe_comm_method.mc2_mask` into the old-style quant/dispatch path.
  Both still failed in the actual NPU operator with
  `aclnnMoeDistributeDispatchV4` / `ERR00100`.  This means the current
  `fused_moe_legacy.py` bridge is not yet ABI-compatible with the old eager
  dispatch path.  Do not keep bypassing guards blindly: the failure moved from
  Python validation into the real dispatch op, which is a strong sign that one
  of the dispatch inputs is malformed.

  Important correction: `qwen3/vllm_ascend/ops/fused_moe_legacy.py` is not a
  faithful copy of the real old eager file
  `ref_wj_qwen3/qwen30b_tpu_verl/vllm_ascend/ops/fused_moe.py`.  It still
  contains many current-stack mode3/hybrid/dynamic-shrink changes and v0.14
  compatibility changes.  The next useful step is not another high-level
  script knob; it is either (1) a compile-off diagnostic with the same
  old-stack bridge to expose the first real non-Dynamo exception, or (2) a
  smaller, more faithful old-eager MoE shim focused on the old
  `AscendFusedMoE.forward -> old moe_comm_method.prepare -> quant_method.apply
  -> old finalize` ABI.

- Attention FIA detail probe, 2026-05-15:

  Log:
  `resample_result_16k_bs32_n16_eager_attn_fia_detail_20260515/logs/wjqwen30b-a3b-record_graph_save4eagle3_20260515105936.txt`.
  This was run on the current eager-fast stack with DecodeOnly FIA detail
  timers enabled.  It completed with `generate_sequences=91.007s`,
  `gen=108.238s`, `rollout_output_time_s=108.241s`, so the instrumentation
  itself is heavy and should not be used as a performance number.

  The useful finding is attribution: for `DecodeOnly/tokens=32`, raw
  `npu_fused_infer_attention_score` detail timing was tiny compared with the
  model-level attention gap.  Across 512 detailed calls, `detail_total`
  averaged about `0.109ms` (`p50=0.051ms`, `p90=0.243ms`), while the broader
  attention-stage timing was much larger and polluted by logging/sync.  This
  means the remaining attention gap is not simply "FIA kernel is slow".
  Future attention work should inspect wrapper/metadata/output-buffer shape and
  real old-vs-new decode path semantics rather than re-profiling the raw FIA
  op.

  A narrow follow-up tried `VLLM_ASCEND_FIA_USE_OUT=1` to make FIA write
  directly into vLLM's output buffer and avoid the intermediate
  `attn_output -> output[:num_tokens]` copy.  It is not viable.  Log:
  `resample_result_16k_bs32_n16_eager_fia_out_gen1_20260515/logs/wjqwen30b-a3b-record_graph_save4eagle3_20260515111147.txt`.
  The run failed before a rollout metric with
  `FusedInferAttentionScore do tiling failed` because the op requires the
  output tensor shape to exactly match the actual attention output
  (`[2128, 32, 128]` in the failing rank), while the vLLM wrapper supplies a
  larger padded output buffer (`[4016, 32, 128]`).  The diagnostic branch was
  removed after the failure.  Do not repeat direct `.out` FIA unless the output
  buffer passed to the op is sliced to an exact shape supported by torch_npu,
  and be careful because views/slices may not be accepted as `out` tensors.

  A second, more conservative `.out` probe only used the out variant when the
  decode output buffer shape exactly matched `num_tokens`.  Env:
  `VLLM_ASCEND_DECODE_FIA_USE_OUT=1`, log:
  `resample_result_16k_bs32_n16_eager_decode_fia_out_gen1_20260515/logs/wjqwen30b-a3b-record_graph_save4eagle3_20260515152604.txt`.
  It completed successfully, but was slower: `generate_sequences=85.445s`,
  `gen=102.593s`, `rollout_output_time_s=102.597s`.  This rules out the
  `attn_output -> output[:num_tokens]` copy as the main attention gap.  The
  exact-shape `.out` branch was removed from the production code path.

  After this, the attention hot path was cleaned so profiling/logging gate
  functions are only called when the corresponding debug env is enabled:
  `VLLM_ASCEND_ATTENTION_DEBUG`, `VLLM_ASCEND_ATTENTION_STAGE_DEBUG`,
  `VLLM_ASCEND_ATTENTION_WRAPPER_DEBUG`, or
  `VLLM_ASCEND_ATTENTION_FIA_DETAIL_DEBUG`.  This is a semantic no-op intended
  to remove per-layer/per-token Python branch overhead from production eager.

  Follow-up cleanup, 2026-05-15: the later sync-wall/FIA-detail probes left a
  few production-path calls that still invoked profiling gates or recorded NPU
  events even when debug envs were off.  The current code now guards
  `forward_fused_infer_attention`, `forward_paged_attention`, wrapper timing,
  sync timing, and FIA-detail timing with the corresponding module-level debug
  booleans before calling the gate functions or creating events.  MoE context
  debug was also changed from a per-call `os.getenv(...)` check to a
  module-level boolean, and the callsite now skips `_maybe_log_moe_context`
  entirely when disabled.  This is a behavior-preserving hot-path cleanup, not
  a new kernel strategy; validate with a cheap `gen1` probe before treating it
  as a performance win.

  Validation:
  `resample_result_16k_bs32_n16_eager_hotpath_cleanup_gen1_20260515/logs/wjqwen30b-a3b-record_graph_save4eagle3_20260515175700.txt`.
  Result: `generate_sequences=86.068s`, `gen=103.496s`,
  `rollout_output_time_s=103.499s` for the standard `265,808` threshold
  tokens.  This is better than the heavily profiled behavior baseline
  (`gen=112.082s`) but still slower than the current best three-step high-water
  mark (`gen~=93.5s`, `generate_sequences~=79.9s`).  Keep the cleanup as a
  lower-noise baseline, but do not count it as closing the old-stack gap.

  Follow-up dummy-skip env cleanup, 2026-05-15: moved the default-off
  `VLLM_ASCEND_QWEN3_DUMMY_SKIP_ATTENTION` check in
  `Qwen3MoeDecoderLayer.forward` from a per-call `os.getenv(...)` lookup to a
  module-level boolean.  This preserves behavior and removes one diagnostic
  branch from the hot path, but it is not a performance fix.  Log:
  `resample_result_16k_bs32_n16_eager_dummyenv_cleanup_gen1_20260515/logs/wjqwen30b-a3b-record_graph_save4eagle3_20260515191714.txt`.
  Result: `generate_sequences=86.116s`, `gen=103.275s`,
  `rollout_output_time_s=103.278s`, `speed=2573.717 tok/s`.  Keep the cleanup
  for hygiene, but do not treat this as closing the old-stack gap.

  Follow-up MC2 no-pad/tensor fast-path probes:

  - `resample_result_16k_bs32_n16_eager_mc2_nopad_fastpath_gen1_20260515/logs/wjqwen30b-a3b-record_graph_save4eagle3_20260515180353.txt`
    added a no-pad short path in `PrepareAndFinalizeWithMC2` for TP=1 decode
    calls where `padded_num_tokens == num_tokens`.  Result:
    `generate_sequences=85.709s`, `gen=103.030s`,
    `rollout_output_time_s=103.035s`, `speed=2579.787 tok/s`.
  - `VLLM_ASCEND_FUSED_MOE_TENSOR_FASTPATH=1`, log
    `resample_result_16k_bs32_n16_eager_tensor_fastpath_gen1_20260515/logs/wjqwen30b-a3b-record_graph_save4eagle3_20260515181603.txt`,
    additionally lets the simple-MC2 fused-experts path return the routed
    tensor directly instead of constructing a `FusedExpertsResult` object when
    events and dynamic EPLB metadata are not required.  Result:
    `generate_sequences=84.237s`, `gen=101.640s`,
    `rollout_output_time_s=101.644s`, `speed=2615.099 tok/s`.
  - Stable `threshold3` validation for the tensor fast path:
    `resample_result_16k_bs32_n16_eager_tensor_fastpath_threshold3_20260515/logs/wjqwen30b-a3b-record_graph_save4eagle3_20260515182516.txt`.
    Step2/3 were `generate_sequences=79.082/79.531s`,
    `gen=90.845/91.117s`, `rollout_output_time_s=90.848/91.120s`.
    This is stable and directionally positive versus noisy behavior baselines,
    but it does not beat the current high-water mark (`generate_sequences`
    low `77-79s`, `gen` low `88-91s`).  Do not promote
    `VLLM_ASCEND_FUSED_MOE_TENSOR_FASTPATH=1` to `eager_fast` yet.

  Interpretation: reducing the new v0.14 fused-MoE wrapper/object overhead is
  directionally positive but small.  It is not the main old-stack gap.  Keep
  this as a candidate optimization, but only promote it after it beats the
  current high-water mark in `threshold3` and preserves full16k behavior.

  Re-test after the torch_npu profiler comparison:
  `resample_result_16k_bs32_n16_eager_tensorfast_current_gen1_20260516002335/logs/wjqwen30b-a3b-record_graph_save4eagle3_20260516002336.txt`.
  Result: `generate_sequences=84.721s`, `gen=101.945s`,
  `rollout_output_time_s=101.948s`, `speed=2607.290 tok/s`.
  This remains below the high-water mark, so keep
  `VLLM_ASCEND_FUSED_MOE_TENSOR_FASTPATH=1` out of `eager_fast`.

- Attention sync-detail probe, 2026-05-15:

  Log:
  `resample_result_16k_bs32_n16_eager_fia_syncdetail_profile_20260515/logs/wjqwen30b-a3b-record_graph_save4eagle3_20260515174426.txt`.
  This probe inserted synchronization boundaries around FIA parameter setup,
  pre-op, op, and output-copy regions.  It produced 528 DecodeOnly/tokens=32
  samples with approximate means:
  `get_params_sync=1.43ms`, `pre_op_sync=1.45ms`, `op_sync=1.47ms`,
  `copy_sync=1.42ms`, `total_sync=5.77ms`.

  Interpretation: the probe is too perturbative for direct optimization.
  Synchronizing at every boundary redistributes outstanding async work across
  all measured sub-stages, so it does not identify one isolated expensive FIA
  sub-op.  Keep this as diagnostic evidence that wrapper/backend waits exist,
  but do not repeat this exact sync-detail experiment as a performance metric.
  Prefer lower-overhead A/B changes on the real eager path.

- Forward-kind profile, 2026-05-14:

  The current qwen3 eager path was profiled with
  `VLLM_ASCEND_FORWARD_KIND_DEBUG=1` on the same threshold-control segment as
  the old-stack profile.  Logs:

  - new:
    `resample_result_16k_bs32_n16_eager_forwardkind_20260514174550/logs/wjqwen30b-a3b-record_graph_save4eagle3_20260514174550.txt`
  - old:
    `../ref_wj_qwen3/qwen30b_tpu_verl/resample_result_16k_bs32_n16_old_profile_segment_20260514173129/logs/wjqwen30b-a3b-record_graph_save4eagle3_20260514173129.txt`

  The token count is identical (`265,808`), but new rollout is
  `107.853s` vs old `99.417s`.  The new forward-kind breakdown shows a
  structural dummy-forward overhead:

  | kind | tokens | avg |
  |---|---:|---:|
  | real DecodeOnly | `32` | `88.135ms` |
  | runtime_dummy | `32` | `85.108ms` |
  | real PrefillNoCache | `4016` | `962.781ms` |

  There are `240` real 32-token decode forwards and `240` runtime dummy
  32-token forwards.  That means ranks without local scheduled work are still
  entering a synchronized dummy path that is almost as expensive as real decode.
  Old eager code propagated `is_dummy` through
  `vllm_ascend.models.qwen3_moe.CustomSparseMoeBlock` into the old
  `vllm_ascend.ops.fused_moe.AscendFusedMoE` path, so we tested restoring that
  semantic behind
  `VLLM_ASCEND_DUMMY_FORWARD_IS_DUMMY=1`.

  Follow-up result: no benefit.  Log:
  `resample_result_16k_bs32_n16_eager_dummyisdummy_20260514/logs/wjqwen30b-a3b-record_graph_save4eagle3_20260514180401.txt`.
  It kept the same token count (`265,808`) and essentially the same rollout
  time (`107.928s` vs `107.853s`).  `runtime_dummy/tokens=32` regressed
  slightly from `85.108ms` to `86.057ms`, and `real/tokens=32` regressed from
  `88.135ms` to `90.686ms`.  The env-gated dummy-forward code was removed
  after this test.  Do not repeat this path unless the dummy branch can skip
  actual MoE/attention work rather than only toggling the MoE load-balance flag.

- Old-vs-new MoE stage profile, 2026-05-14:

  The current best qwen3 eager stack and the old reference stack now have
  comparable runner-level and MoE-stage profiles.  The stable finding is that
  the remaining gap is inside decode forward and is not caused by sampling,
  phase switching, output length, or vLLM scheduler shape.

  | run | decode forward | decode sample | decode prepare | decode post |
  |---|---:|---:|---:|---:|
  | old observe `old_runner_observe_clean_20260514163036` | `59.003ms` | n/a | `2.679ms` | `2.806ms` |
  | old MoE-stage `old_moe_stage_observe_20260514164046` | `61.584ms` | n/a | `2.740ms` | `2.813ms` |
  | new current-best observe `eager_profile_currentbest_observe_20260513` | `86.113ms` | `2.096ms` | `3.059ms` | `0.562ms` |
  | new MoE-stage `eager_profile_attn_moe_decode_20260514` | `88.664ms` | `1.956ms` | `2.893ms` | `0.578ms` |

  Initial MoE-stage timing looked like a direct MC2 shape mismatch:

  | run | comm/token shape | dispatch | MLP | combine | total |
  |---|---|---:|---:|---:|---:|
  | old `call>1` | `MC2/tokens=32` | `0.628ms` | `0.077ms` | `0.101ms` | `0.806ms` |
  | new `call>1` | `MC2/tokens=512` | `1.396ms` | `0.482ms` | `0.209ms` | `2.087ms` |

  Follow-up context profiling corrected this interpretation:

  - `resample_result_16k_bs32_n16_eager_moe_context_profile_20260514165224`
    enabled `VLLM_ASCEND_MOE_CONTEXT_DEBUG=1` and
    `VLLM_ASCEND_MOE_STAGE_DEBUG=1` on the current best eager path.
  - Compile/profile warmup calls did enter MC2 with `before_tokens=512` and
    `after_tokens=512`.
  - Real rollout decode calls entered MC2 with `before_tokens=32`,
    `after_tokens=32`, `mc2_mask_tokens=32`, `max_tokens_across_dp=32`, and
    `padded_num_tokens=32`.
  - The run summary showed real decode `MC2/tokens=32` around
    `dispatch=0.874ms`, `mlp=0.097ms`, `combine=0.145ms`,
    `total=1.115ms` over 656 records.  This is still slower than old
    `0.806ms`, but it is not a 512-token steady-state decode path.

  Corrected conclusion: do not keep chasing `PrepareAndFinalizeWithMC2`
  32-to-512 expansion for steady-state decode.  The remaining decode-forward
  gap is broader than MC2 token shape and needs layer-level decomposition
  across qkv/rope/attention/o_proj, norms, router/shared-expert, expert MoE,
  and dense fallback MLP.

- Profiling helper: `tools/profile_rollout_logs.py` now parses old/new rollout
  text logs without modifying either runtime.  It extracts step metrics,
  rollout speed, token totals, KV cache/concurrency, vLLM runner
  `ProfileExecuteDuration` records, and env-gated MoE stage timings.  Use this
  before starting any new perf experiment:

  ```bash
  python3 tools/profile_rollout_logs.py \
    ../ref_wj_qwen3/qwen30b_tpu_verl/resample_result_16k_bs32_n16_baseline_ft/logs/wjqwen30b-a3b-record_graph_save4eagle3_20260507090220.txt \
    ../ref_wj_qwen3/qwen30b_tpu_verl/resample_result_16k_bs32_n16_baseline_ft/logs/wjqwen30b-a3b-record_graph_save4eagle3_20260511235742.txt \
    resample_result_16k_bs32_n16_eager_fast/logs/wjqwen30b-a3b-record_graph_save4eagle3_20260514100236.txt \
    resample_result_16k_bs32_n16_eager_fast/logs/wjqwen30b-a3b-record_graph_save4eagle3_20260514110218.txt
  ```

- New-vs-old log profile snapshot, 2026-05-14:

  | run | workload | rollout | generate_sequences | gen ms/token | tokens | tok/s | KV / concurrency |
  |---|---|---:|---:|---:|---:|---:|---|
  | old `20260507090220` | threshold, avg 15 steps | `84.001s` | `57.762s` | `0.3720` | `265,040` last step | `3149.34` | `414080 / 23.79x` |
  | old `20260511235742` | full 16k, 1 step | `1131.425s` | `1097.994s` | `0.3735` | `3,069,063` | `2712.56` | `414080 / 23.79x` |
  | new eager `20260514100236` | full 16k, abort after rollout | `1575.765s` | n/a | n/a | `3,051,584` | `1936.57` | `395776 / 22.74x` |
  | new eager `20260514110218` | full 16k, abort after rollout | `1542.219s` | n/a | n/a | `3,051,584` | `1978.70` | `395776 / 22.74x` |
  | new eager mem085 `20260514154735` | full 16k, abort after rollout | `1562.019s` | n/a | n/a | `3,051,584` | `1953.62` | `409088 / 23.50x` |

  The full 16k runs generate nearly identical token counts (`3.07M` old vs
  `3.05M` new), so the full-step gap is not a length-distribution issue.
  Raising new eager `gpu_memory_utilization` to `0.85` almost matched old KV
  concurrency (`23.50x` vs `23.79x`) but did not improve rollout speed.  This
  rules out KV capacity/concurrency as the primary full16k bottleneck.

- Full 16k behavior diverges much more than the threshold workload.  The old
  full-step reference
  `../ref_wj_qwen3/qwen30b_tpu_verl/resample_result_16k_bs32_n16_baseline_ft/logs/wjqwen30b-a3b-record_graph_save4eagle3_20260511235742.txt`
  generated `3,069,063` total tokens in `1131.425s`
  (`2712.56 tok/s`, `timing_per_token_ms/gen=0.3735`).  The current eager-fast
  full-step log
  `resample_result_16k_bs32_n16_eager_fast/logs/wjqwen30b-a3b-record_graph_save4eagle3_20260514110218.txt`
  generated a very similar `3,051,584` tokens but took `1542.219s`
  (`1978.70 tok/s`).  Length distribution is comparable, so the full-step gap
  is not output length.  Since the threshold workload only differs by roughly
  8-10s while full 16k differs by roughly 400s, continue investigating
  long-context decode/attention-KV scaling in addition to short-token MoE MC2.
- Decode-length profile, 2026-05-14:

  Log:
  `resample_result_16k_bs32_n16_eager_decode_len_profile_20260514183101/logs/wjqwen30b-a3b-record_graph_save4eagle3_20260514183101.txt`.

  This was a full-16k one-step rollout-only profile, intentionally interrupted
  after reaching the `4097-8192` decode bucket.  It was enough to answer the
  question without burning a whole 16k step.  Parsed summary:

  | bucket | forward | sample | prepare | post | avg max seq | records |
  |---|---:|---:|---:|---:|---:|---:|
  | `<=1024` | `92.293ms` | `2.302ms` | `3.165ms` | `0.559ms` | `530.6` | `174` |
  | `1025-2048` | `92.352ms` | `2.289ms` | `3.374ms` | `0.556ms` | `1474.4` | `142` |
  | `2049-4096` | `92.315ms` | `2.219ms` | `3.536ms` | `0.548ms` | `2633.3` | `110` |
  | `4097-8192` | `90.254ms` | `2.012ms` | `4.566ms` | `0.535ms` | `4248.8` | `20` |

  Conclusion: the full-16k slowdown is not explained by an early
  sequence-length-dependent decode-forward blow-up.  Up through the first
  `4k+` bucket, forward remains a flat `~92ms` and even trends slightly lower
  when fewer active requests remain.  The primary unresolved bottleneck is the
  flat decode-forward implementation gap versus the old stack (`~60-62ms` old
  vs `~88-92ms` new), not KV capacity, token count, or early long-context
  scaling.

- Decode-only attention/MoE profile, 2026-05-14:

  Log:
  `resample_result_16k_bs32_n16_eager_decode_attn_moe_profile_20260514185009/logs/wjqwen30b-a3b-record_graph_save4eagle3_20260514185009.txt`.

  This used the current best eager stack with one threshold-control step,
  `VLLM_ROLLOUT_DEBUG_GENERATE_ONLY=1`,
  `VLLM_ASCEND_MODEL_EXECUTE_TIME_OBSERVE=1`,
  `VLLM_ASCEND_MOE_STAGE_DEBUG=1`, and decode-only attention stage profiling
  (`VLLM_ASCEND_ATTENTION_STAGE_DEBUG_STATE_FILTER=DecodeOnly`).  It produced
  the same token count as the old profile segment (`265,808`) and completed in
  `108.270s`.

  Comparable old profile:
  `../ref_wj_qwen3/qwen30b_tpu_verl/resample_result_16k_bs32_n16_old_moe_stage_observe_20260514164045/logs/wjqwen30b-a3b-record_graph_save4eagle3_20260514164046.txt`.

  | run | decode forward | decode sample | decode prepare | decode post |
  |---|---:|---:|---:|---:|
  | old | `61.534ms` | n/a | `2.738ms` | `2.812ms` |
  | new | `89.202ms` | `1.956ms` | `2.831ms` | `0.572ms` |

  Decode-only attention FIA is not the main gap:

  | run | attention state/op | op avg | p50 | p95 |
  |---|---|---:|---:|---:|
  | new | `DecodeOnly/fia/tokens<=32` | `0.087ms` | `0.047ms` | `0.183ms` |

  MoE remains a confirmed contributor but does not explain all of the gap:

  | run | comm/tokens | dispatch | MLP | combine | total |
  |---|---|---:|---:|---:|---:|
  | old | `MC2/tokens=32` | `0.628ms` | `0.077ms` | `0.101ms` | `0.806ms` |
  | new | `MC2/tokens=32` | `0.835ms` | `0.096ms` | `0.143ms` | `1.074ms` |

  Conclusion: do not chase attention decode op selection as the primary issue.
  The new stack's MoE MC2 path is slower, especially dispatch/combine, but
  the remaining `~28ms` decode-forward gap is larger than the measured MoE
  delta alone.  Continue profiling MoE implementation overhead plus non-MoE
  layer work (qkv/o_proj, norms, router/shared-expert, residual/reduce), not
  sampler, phase switch, or output length.

- `resample_result_16k_bs32_n16_eager_profile_currentbest_observe_20260513`
  enabled `VLLM_ASCEND_MODEL_EXECUTE_TIME_OBSERVE=1` on the current best eager
  path.  The built-in runner summary shows decode `forward` dominates:
  `forward ~= 84-87ms/step/worker`, while `Sample ~= 1.8-2.4ms`,
  `prepare input ~= 2.5-3.5ms`, and `post process ~= 0.5-0.6ms`.
  This rules out sampler and phase-switch as the remaining `generate_sequences`
  bottleneck.  Continue by splitting forward into attention and MoE, not by
  repeating sampler/scheduler experiments.
- `resample_result_16k_bs32_n16_eager_full16k_forcepa_profile_20260514`
  attempted to validate the old-stack-like paged-attention decode path under a
  full 16k workload with attention/MoE stage timers enabled.  The run was
  interrupted before a rollout metric because compiled-eager treated the early
  profiling gate as effectively static and logged/synchronized every decode
  step (`call=33,34,...55`), contaminating the result.  Do not use this log for
  throughput.  The profiling gate is now wrapped with `torch._dynamo.disable`
  when stage timing envs are enabled so future profiling should preserve the
  intended first-32/every-N cadence.
- Added lightweight env-gated stage timing to attention
  (`VLLM_ASCEND_ATTENTION_STAGE_DEBUG=1`) and throttled both attention and MoE
  stage timers so they only synchronize for the first 32 calls and then every
  interval.  Use this only in profiling runs; normal eager performance is
  unaffected when the flags are unset.
- `resample_result_16k_bs32_n16_eager_profile_attn_moe_decode_20260514`
  split the forward hot path with attention and MoE stage timers.  This run
  did not include every current-best knob, so use it for proportions rather
  than absolute throughput.  Stable non-warmup MoE calls were roughly
  `dispatch ~= 1.40ms`, `mlp ~= 0.48ms`, `combine ~= 0.21ms`,
  `total ~= 2.09ms`, while attention stage timings were tiny
  (`PrefillNoCache` FIA calls typically around `0.15-0.32ms`).  This reinforces
  that the remaining `generate_sequences` gap is inside the MoE MC2 path, not
  attention or sampler.
- Old-stack logs currently available do not contain MoE/attention stage timing,
  but the old repo already has native `ProfileExecuteDuration` support in
  `vllm_ascend/worker/model_runner_v1*.py`, gated by
  `VLLM_ASCEND_MODEL_EXECUTE_TIME_OBSERVE`.  Next real profiling step should
  run the old stack with this env enabled to capture old
  `prepare input / forward / post process` timing.  Only add old-stack MoE
  stage probes if runner-level `forward` confirms a comparable old-vs-new
  forward gap and the existing logs are still insufficient.

## Code Path Differences To Profile, Not Guess

- Old fast eager MoE uses the old `ops/moe/*` MC2 path and simpler
  `TokenDispatcherWithMC2` state:
  `global_bs=0`, `expert_token_nums_type=0`, A3 `x_active_mask` passed to both
  dispatch and combine, and no Python metadata dict between dispatch and
  combine.
- Current qwen3 eager uses the newer split-MoE
  `ops/fused_moe/*` path.  The current best already enables
  `VLLM_ASCEND_FUSED_MOE_SIMPLE_MC2=1` to avoid the generic dataclass/metadata
  path, but the implementation still differs from old in hot-path details:
  optional `expand_scales`, configurable expert count source, optional active
  mask modes, optional custom grouped-matmul-swiglu, and newer
  `AscendCompiler + npugraph_ex_config` compiled-eager wrapping.
- Therefore the next code-level bottleneck work should instrument or compare:
  1. old vs new runner-level `forward` with `VLLM_ASCEND_MODEL_EXECUTE_TIME_OBSERVE=1`;
  2. old vs new MC2 `dispatch / MLP / combine` for the same threshold workload;
  3. full16k decode attention/KV scaling only after MoE dispatch is accounted
     for, because current attention timers are much smaller than MoE MC2.

## Ruled Out / Do Not Repeat

- `VLLM_ROLLOUT_GPU_MEMORY_UTILIZATION=0.85` on top of the current-best eager
  stack: slower.  The first attempt
  `resample_result_16k_bs32_n16_eager_genonly_currentbest_mem085_20260514/...`
  was invalid because stale NPU worker processes were still occupying device
  memory and it was interrupted before metrics.  A clean re-test did complete:
  `resample_result_16k_bs32_n16_eager_genonly_currentbest_mem085_clean_20260514/logs/wjqwen30b-a3b-record_graph_save4eagle3_20260514150515.txt`.
  It increased KV cache size / concurrency from `395,776 / 22.74x` to
  `409,216 / 23.51x`, closer to the old full-step reference, but regressed
  threshold-control generate-only performance.  Step2/3:
  `generate_sequences=82.4088/80.8505s`, `gen=94.3276/92.5632s`,
  slower than current best `76.9058s` / `88.4519s`.  Keep the default
  `VLLM_ROLLOUT_GPU_MEMORY_UTILIZATION=0.83` for eager fast.
- `VLLM_ASCEND_EAGER_COMPILE_RANGE_SPLITS=512`: failed during rollout engine
  initialization.  This was a targeted test to align compiled-eager ranges
  with the MC2/AllToAll threshold.  The engine config did show
  `compile_ranges_split_points=[512, 17408]`, and both ranges compiled:
  `(1, 512)` and `(513, 17408)`.  However, vLLM profile-run then failed in
  `_dummy_sampler_run` with
  `TypeError: only integer scalar arrays can be converted to a scalar index`
  at `hidden_states = hidden_states[logit_indices]`.  This suggests splitting
  the AscendCompiler eager graph into token ranges changes dummy profile output
  / indexing assumptions before generation starts.  Do not repeat this knob
  without first fixing the profile-run indexing path.  Log:
  `resample_result_16k_bs32_n16_eager_genonly_currentbest_ranges512_20260514/logs/wjqwen30b-a3b-record_graph_save4eagle3_20260514152234.txt`.
- `VLLM_ASCEND_EAGER_COMPILE_ELIMINATE_NOOPS=1`: slower.  This isolated only
  the vLLM compile `eliminate_noops` pass on top of the current-best eager
  stack, without enabling the broader `fuse_norm_quant` / `fuse_act_quant`
  pass bundle that was already ruled out.  The pass was correctly enabled in
  engine config (`pass_config.eliminate_noops=True`), but step2/3 regressed to
  `generate_sequences=81.6028/81.5459s`, `gen=93.2681/93.0578s`,
  slower than the current best `76.9058s` / `88.4519s`.  Log:
  `resample_result_16k_bs32_n16_eager_genonly_currentbest_noops_20260514/logs/wjqwen30b-a3b-record_graph_save4eagle3_20260514152708.txt`.
  Do not enable this pass by default.
- `VLLM_ROLLOUT_ASYNC_SCHEDULING=false`: slower.
- Restoring the old wrapper patch alone: did not recover performance.
- `VLLM_ROLLOUT_ENABLE_CHUNKED_PREFILL=false` with `max_num_batched_tokens=17408`:
  slower than chunked prefill.
- `VLLM_ASCEND_MOE_CHUNK_STRICT_LT=1`: slower than the current `<=512` path.
  It was first tested on an older/non-current-best stack, then re-tested on top
  of the current-best eager stack
  (`moepolicy + block64 + device metadata + simple MC2`) to make sure this was
  not a stale conclusion.  The current-best retest entered
  `actor_rollout_generate_sequences` but produced no first-step metric after
  more than three minutes in rollout generation, while the current best
  normally reaches step1 around `generate_sequences=83.9473s` /
  `gen=101.3077s`.  The run was interrupted as a clear regression.  Log:
  `resample_result_16k_bs32_n16_eager_genonly_currentbest_strictlt_20260514/logs/wjqwen30b-a3b-record_graph_save4eagle3_20260514134940.txt`.
  Keep the current `max_tokens <= VLLM_CHUNK_MOE_SIZE` single-chunk behavior.
- `VLLM_ASCEND_MOE_SKIP_EMPTY_EXPERT_MLP=1`: slower / likely pathological on
  the current-best stack.  This env-gated code path tried to mirror the old
  single-chunk MC2 shortcut that skips `unified_apply_mlp` when
  `expert_tokens` is all zero.  On the current-best
  `moepolicy + block64 + device metadata + simple MC2` stack, the run entered
  `actor_rollout_generate_sequences` but produced no first-step metric after
  more than three minutes, while the current best normally reaches step1 around
  `generate_sequences=83.9473s` / `gen=101.3077s`.  The run was interrupted.
  Log:
  `resample_result_16k_bs32_n16_eager_genonly_skipempty_20260514/logs/wjqwen30b-a3b-record_graph_save4eagle3_20260514135500.txt`.
  Keep this env off; the shortcut is not compatible with the current metadata /
  dispatch semantics.
- Explicitly passing `with_prefill` into current `set_ascend_forward_context`:
  slower.
- `VLLM_ASCEND_FORCE_PAGED_ATTENTION_DECODE=1`: slower.
- `VLLM_ASCEND_FORCE_PAGED_ATTENTION_DECODE=1` on top of the current-best
  eager stack under the full 16k workload: severe regression.  This was the
  clean follow-up after fixing the profiling gate contamination.  It entered
  `actor_rollout_generate_sequences` but produced no first-step
  `rollout_output_time_s` even after roughly 25 minutes, already beyond the
  old full-step rollout time (`1131s`) and close to / beyond the current qwen3
  eager full-step window.  The run was interrupted.  Log:
  `resample_result_16k_bs32_n16_eager_full16k_forcepa_clean_20260514/logs/wjqwen30b-a3b-record_graph_save4eagle3_20260514142806.txt`.
  Do not force paged-attention decode for eager full16k; normal eager FIA is
  still the better baseline.
- `VLLM_ASCEND_MC2_USE_ACTIVE_MASK=1`: slower.
- `VLLM_ASCEND_ENABLE_FUSED_MC2=1`: not useful for this path.
- `VLLM_ASCEND_ENABLE_FUSED_MC2=2`: slower on top of the current best stack.
  This decode-only `dispatch_gmm_combine_decode` path was a plausible
  untested MoE-hotspot candidate, but it did not beat the normal MC2 path.
  Log:
  `resample_result_16k_bs32_n16_eager_genonly_fusedmc2_2_20260513/logs/wjqwen30b-a3b-record_graph_save4eagle3_20260514010737.txt`.
  Step2/3: `generate_sequences=83.0240/80.5506s`,
  `gen=94.9424/92.2662s`, slower than the current best
  `78.8466/76.9058s` and `90.6882/88.4519s`.
- `VLLM_ASCEND_ENABLE_MLAPO=0`: slower.
- `VLLM_ASCEND_EAGER_COMPILE_PASS_FUSION=1`: no useful gain.
- `VLLM_ASCEND_EAGER_COMPILE_ATTENTION_SPLIT=1`: no useful gain.
- `VLLM_ASCEND_EAGER_COMPILE_ATTENTION_SPLIT=1` on top of the current-best
  eager stack: clearly regressed after the attention-split implementation was
  fixed.  The engine config did show the intended standard attention graph
  boundaries (`vllm::unified_attention*`, `vllm::mla_forward`) while keeping
  `cudagraph_mode=NONE`, so this was a valid test rather than a no-op.  The
  run entered `actor_rollout_generate_sequences` but produced no first-step
  metric after more than three minutes, while the current best reaches step1
  around `generate_sequences=83.9473s` / `gen=101.3077s`.  The run was
  interrupted.  Log:
  `resample_result_16k_bs32_n16_eager_genonly_currentbest_attnsplit_20260514/logs/wjqwen30b-a3b-record_graph_save4eagle3_20260514151650.txt`.
  Do not repeat attention split on the AscendCompiler eager path.
- Forcing AllToAll MoE: much slower.
- `VLLM_CHUNK_MOE_SIZE=1024`: no stable gain.
- `VLLM_CHUNK_MOE_SIZE=2048`: regressed and can OOM.
- Legacy attention / legacy fused MoE alignment attempts: not the right
  high-performance path on the new stack.
- `VLLM_ASCEND_FUSED_MOE_SOFTMAX_TOPK=1`: no gain / slight regression.
  This tried to revive a fused `torch_npu.npu_moe_gating_top_k_softmax`
  routing path for Qwen3-MoE `norm_topk_prob=true`, but the remaining gap is
  not in router top-k.  Log:
  `resample_result_16k_bs32_n16_eager_genonly_softmax_topk_20260514/logs/wjqwen30b-a3b-record_graph_save4eagle3_20260514014228.txt`.
  Step2/3: `generate_sequences=79.9711/79.6577s`,
  `gen=91.7529/91.4225s`, slower than the current best
  `76.9058s` / `88.4519s`.
- `VLLM_QWEN3_MOE_ASCEND_LEGACY_INIT=1`: bad.  Log:
  `resample_result_16k_bs32_n16_eager_legacyinit_threshold_20260513/logs/wjqwen30b-a3b-record_graph_save4eagle3_20260513053302.txt`.
  Step1 was already `generate_sequences=89.0999s`, `gen=107.2469s`,
  `rollout=107.2503s`, and the run exited with `exit_code=1`.
- `VLLM_QWEN3_MOE_REDUCE_RESULTS=0`: slower.  Log:
  `resample_result_16k_bs32_n16_eager_reducefalse_threshold_20260513/logs/wjqwen30b-a3b-record_graph_save4eagle3_20260513050826.txt`.
  Step1: `generate_sequences=86.3230s`, `gen=104.2916s`,
  `rollout=104.2948s`.
- `VLLM_ASCEND_MC2_OMIT_EXPAND_SCALES=1`: bad.  It makes the first generation
  step run far longer than the current best (>160s without completing), so the
  new combine op expects/benefits from `expand_scales` even though the old
  stack did not pass it.
- Old MC2 contract combination
  `VLLM_ROLLOUT_MC2_USE_ACTIVE_MASK=1 +
  VLLM_ASCEND_MC2_OMIT_EXPAND_SCALES=1 +
  VLLM_ASCEND_MC2_EXPERT_NUM_SOURCE=num_experts`: slower.  This tested whether
  the old stack's A3 MC2 operator contract only works as a bundle, not as
  isolated knobs.  It did not.  Log:
  `resample_result_16k_bs32_n16_eager_genonly_oldmc2_contract_20260514/logs/wjqwen30b-a3b-record_graph_save4eagle3_20260514015454.txt`.
  Step1: `generate_sequences=87.2344s`, `gen=104.7858s`,
  `rollout=104.7896s`, slower than current-best step1
  `83.9473s` / `101.3077s`.
- `VLLM_ASCEND_MC2_LEGACY_SWIGLU_GROUP_INDEX=1`: bad.  It changed MC2
  `npu_dequant_swiglu_quant(group_index=...)` to the old stack's direct
  `group_list` behavior.  It was re-tested on top of the current-best
  `moepolicy + block64 + device metadata + simple MC2` stack and still
  regressed.  Log:
  `resample_result_16k_bs32_n16_eager_genonly_currentbest_legacy_swiglu_idx_20260514/logs/wjqwen30b-a3b-record_graph_save4eagle3_20260514020043.txt`.
  Step1: `generate_sequences=88.7564s`, `gen=106.2395s`,
  `rollout=106.2428s`, worse than current-best step1
  `83.9473s` / `101.3077s`.  Do not repeat this path.
- `VLLM_ASCEND_MC2_EXPERT_NUM_SOURCE=num_experts`: slower on top of the
  current best stack.  Log:
  `resample_result_16k_bs32_n16_eager_genonly_moepolicy_block64_devicemeta_simplemc2_expertnum_20260513/logs/wjqwen30b-a3b-record_graph_save4eagle3_20260513232318.txt`.
  Step2/3: `generate_sequences=80.3682/79.8304s`,
  `gen=92.3416/91.4739s`, slower than the current best
  `76.9058s` / `88.4519s`.  Keep the current expert-map based value.
- `VLLM_ROLLOUT_MC2_USE_ACTIVE_MASK=1`: slower on top of the current best
  stack.  Although the old A3 MC2 dispatcher always passed `x_active_mask`,
  enabling it here regressed.  Log:
  `resample_result_16k_bs32_n16_eager_genonly_moepolicy_block64_devicemeta_simplemc2_activemask_20260513/logs/wjqwen30b-a3b-record_graph_save4eagle3_20260513233155.txt`.
  Step2/3: `generate_sequences=79.1962/79.0581s`,
  `gen=91.0170/90.6925s`, slower than the current best
  `76.9058s` / `88.4519s`.  Keep active mask disabled.
- `VLLM_ASCEND_MC2_DISABLE_DISPATCH_EXPERT_SCALES=1`: no gain / slower.
  Log:
  `resample_result_16k_bs32_n16_eager_genonly_moepolicy_block64_devicemeta_simplemc2_no_dispatch_scales_20260514/logs/wjqwen30b-a3b-record_graph_save4eagle3_20260514002443.txt`.
  Step1: `generate_sequences=88.9440s`, `gen=106.2554s`, worse than the
  current-best step1 `83.9473s` / `101.3077s`.  The run was interrupted after
  step1.  Also, worker env did not show `HCCL_INTRA_PCIE_ENABLE=1` /
  `HCCL_INTRA_ROCE_ENABLE=0`, so this branch was probably not the active gap.


- `VLLM_ASCEND_EAGER_COMPILE_USE_INDUCTOR=1` with
  `VLLM_USE_STANDALONE_COMPILE=0`: cold-start compile is blocked by a very large
  inductor graph.  A follow-up controlled smoke reached `backend='inductor'`,
  completed Dynamo in roughly 7s per worker, and produced torchinductor
  `fx_graph_*` / IR debug artifacts, but no first-step metric appeared before
  interruption.  `py-spy` on a worker showed the hot stack inside
  `torch._inductor.scheduler.graph_partition ->
  get_graph_partition_symbol_inputs -> get_free_symbol_uses` during
  `_dummy_run` / KV-cache sizing, while NPUs were idle.  This does not look like
  a bad rollout behavior or Ray hang; it is CPU-side inductor partition/codegen
  scale.  Logs:
  `resample_result_16k_bs32_n16_eager_inductor_adaptor_smoke_20260514/logs/wjqwen30b-a3b-record_graph_save4eagle3_20260514012830.txt`,
  `resample_result_16k_bs32_n16_eager_inductor_cold_20260514120800/logs/wjqwen30b-a3b-record_graph_save4eagle3_20260514120801.txt`.
  Do not repeat the same full unsplit inductor graph.  If we revisit native
  inductor, first reduce graph size with attention/piecewise split boundaries
  or a smaller compile range, then test cache reuse.
- `VLLM_ASCEND_EAGER_COMPILE_USE_INDUCTOR=1 +
  VLLM_ASCEND_EAGER_COMPILE_ATTENTION_SPLIT=1` before the attention-split fix:
  inconclusive, because the diagnostic knob was not actually installing the
  standard attention split boundaries.  The platform code has since been fixed
  to reset `splitting_ops=None` before `set_splitting_ops_for_v1()`, so new
  logs should show `vllm::unified_attention*` split ops.
- `VLLM_ASCEND_EAGER_COMPILE_USE_INDUCTOR=1 +
  VLLM_ASCEND_EAGER_COMPILE_ATTENTION_SPLIT=1` after the split fix:
  progressed past the giant Python graph-partition bottleneck but failed in
  `torch.utils._triton.has_triton()` because the NPU process has no CUDA device
  capability.  Log:
  `resample_result_16k_bs32_n16_eager_inductor_attnsplit_2k_20260514122534/logs/wjqwen30b-a3b-record_graph_save4eagle3_20260514122534.txt`.
  This led to an experiment-only monkeypatch in `vllm/env_override.py` that
  returns `False` from `has_triton()` when
  `VLLM_ASCEND_EAGER_COMPILE_USE_INDUCTOR=1`.
- `VLLM_ASCEND_EAGER_COMPILE_USE_INDUCTOR=1 +
  VLLM_ASCEND_EAGER_COMPILE_ATTENTION_SPLIT=1` with the NPU `has_triton=False`
  patch: progressed into native NPU Triton/Inductor codegen but failed during
  model initialization with a vector-core exception in `aclnnMatmul` while
  compiling/running split graph `model__49_inference_51`.  Log:
  `resample_result_16k_bs32_n16_eager_inductor_attnsplit_notriton_2k_20260514122927/logs/wjqwen30b-a3b-record_graph_save4eagle3_20260514122927.txt`.
  `py-spy` showed this stage was no longer Python graph partitioning; it was
  inside `torch_npu._inductor.npu_triton_heuristics.autotune_to_one_config`
  and Ascend Triton codegen.  Do not repeat this exact setting.  If continuing
  native inductor, try to skip the torch_npu cpp-wrapper real-input pre-run
  (`VLLM_ASCEND_INDUCTOR_AUTOTUNE_AT_COMPILE_TIME=1`) or otherwise reduce
  NPU Triton autotune/codegen scope.
- `VLLM_ASCEND_EAGER_COMPILE_USE_INDUCTOR=1 +
  VLLM_ASCEND_EAGER_COMPILE_ATTENTION_SPLIT=1 +
  VLLM_ASCEND_INDUCTOR_AUTOTUNE_AT_COMPILE_TIME=1`: also failed before the
  first generation step with the same class of `aclnnMatmul` vector-core
  exception.  This forced `torch._inductor.config.triton.autotune_at_compile_time=True`
  to avoid torch_npu's cpp-wrapper real-input pre-run, but the generated NPU
  Triton/Inductor path still executed an invalid matmul kernel during model
  initialization.  Log:
  `resample_result_16k_bs32_n16_eager_inductor_attnsplit_compiletime_2k_20260514124138/logs/wjqwen30b-a3b-record_graph_save4eagle3_20260514124138.txt`.
  Treat native `backend='inductor'` as unstable for this Qwen3-MoE rollout
  path unless we first add a targeted fallback for the generated matmul/rmsnorm
  Triton kernels.  Do not spend more full-run time on generic native inductor
  without such a targeted fallback.
- `NPUGRAPH_EX_ENABLE_STATIC_KERNEL=True`: slower.  This tested whether
  enabling npugraph_ex static kernel on top of the current best compiled-eager
  path could recover old-stack-like execution.  It regressed first-step
  generate-only performance.  Log:
  `resample_result_16k_bs32_n16_eager_genonly_npugraph_static_20260514/logs/wjqwen30b-a3b-record_graph_save4eagle3_20260514020922.txt`.
  Step1: `generate_sequences=88.7896s`, `gen=106.0651s`,
  `rollout=106.0684s`, worse than current-best step1
  `83.9473s` / `101.3077s`.
- `HCCL_INTRA_PCIE_ENABLE=1 + HCCL_INTRA_ROCE_ENABLE=0`: slower.  This
  explicitly forced the intra-node HCCL route hints on top of the current-best
  eager path.  Worker environments confirmed both variables were inherited, so
  this was a valid route experiment rather than a missing-env run.  Log:
  `resample_result_16k_bs32_n16_eager_genonly_hcclintra_20260514/logs/wjqwen30b-a3b-record_graph_save4eagle3_20260514125609.txt`.
  Step1: `generate_sequences=88.3663s`, `gen=105.6949s`,
  `rollout=105.6982s`, worse than current-best step1
  `83.9473s` / `101.3077s`.  Do not repeat this path.
- `VLLM_ASCEND_MC2_USE_DISPATCH_ACTIVE_MASK=1 +
  VLLM_ASCEND_MC2_USE_COMBINE_ACTIVE_MASK=0`: bad.  This split the old A3
  `x_active_mask` behavior and only passed the active mask into dispatch on top
  of the current-best simple-MC2 path.  It stayed in
  `actor_rollout_init_model` for more than five minutes and produced no first
  step metric before interruption, while the current best reaches step1 in
  about 4m40s with `generate_sequences=83.9473s`.  Log:
  `resample_result_16k_bs32_n16_eager_genonly_dispatch_activemask_20260514/logs/wjqwen30b-a3b-record_graph_save4eagle3_20260514130544.txt`.
  Do not repeat dispatch-only active mask.
- `VLLM_ASCEND_MC2_USE_DISPATCH_ACTIVE_MASK=0 +
  VLLM_ASCEND_MC2_USE_COMBINE_ACTIVE_MASK=1`: slower.  This tested the other
  half of the old A3 active-mask contract by only passing `x_active_mask` to
  combine.  It completed 3 threshold-control generate-only steps, but regressed
  against the current best.  Log:
  `resample_result_16k_bs32_n16_eager_genonly_combine_activemask_20260514/logs/wjqwen30b-a3b-record_graph_save4eagle3_20260514131235.txt`.
  Step2/3: `generate_sequences=82.2615/81.4568s`,
  `gen=94.2832/93.3336s`, slower than the current best
  `76.9058s` / `88.4519s`.  Do not repeat combine-only active mask.
- `VLLM_ASCEND_MC2_EXPERT_TOKEN_NUMS_TYPE=1`: slower.  This tested whether
  matching the old MC2 dispatcher's configurable `expert_token_nums_type` /
  `group_list_type` contract would improve the new split-MoE simple-MC2 path.
  It did not.  Log:
  `resample_result_16k_bs32_n16_eager_genonly_mc2_tokentype1_20260514/logs/wjqwen30b-a3b-record_graph_save4eagle3_20260514022014.txt`.
  Step2/3: `generate_sequences=80.6628/80.6242s`,
  `gen=92.5930/92.4329s`, slower than the current best
  `76.9058s` / `88.4519s`.  Keep the default type `0`.
- `VLLM_ASCEND_MC2_LEGACY_DISPATCHER_INIT=1`: slower.  This recreated the old
  `moe_comm_method_new.py` MC2 construction style where
  `TokenDispatcherWithMC2()` is instantiated without passing `top_k`,
  `num_experts`, or `num_local_experts`.  The hypothesis was that old-stack
  MC2 might rely on `moe_expert_num=0` in the dispatch/combine ABI.  It did
  not help.  Log:
  `resample_result_16k_bs32_n16_eager_genonly_legacy_dispatcher_init_20260514/logs/wjqwen30b-a3b-record_graph_save4eagle3_20260514132317.txt`.
  Step2/3: `generate_sequences=81.8354/82.6254s`,
  `gen=93.9430/94.4456s`, slower than the current best
  `76.9058s` / `88.4519s`.  Do not repeat legacy dispatcher init.


- `VLLM_ASCEND_NPUGRAPH_EX_RUN_EAGERLY=0`: not viable with the current eager
  rollout task-queue setting.  This changed npugraph_ex compile from the
  default eager pre-run into actual NPU graph capture during model init, then
  failed before the first step with `RuntimeError: Do not support
  TASK_QUEUE_ENABLE = 2 during NPU graph capture`.  Log:
  `resample_result_16k_bs32_n16_eager_genonly_npugraph_no_runeager_20260514/logs/wjqwen30b-a3b-record_graph_save4eagle3_20260514023940.txt`.
  Keep the default `run_eagerly=1` for eager compile; do not repeat unless
  also changing task queue and accepting a graph-capture path.
- `VLLM_ASCEND_PRESELECT_MOE_COMM=1`: slower.  This only preselected
  `moe_comm_type` and reused `reserved_mc2_mask` outside
  `set_ascend_forward_context`, without reverting the rest of the context to
  old-stack behavior.  It did not reduce the remaining `generate_sequences`
  gap, so the overhead is not from recomputing the context metadata.  Log:
  `resample_result_16k_bs32_n16_eager_genonly_preselect_moecomm_20260514/logs/wjqwen30b-a3b-record_graph_save4eagle3_20260514024813.txt`.
  Step2/3: `generate_sequences=83.0763/82.3038s`,
  `gen=94.8845/94.0725s`, slower than the current best
  `76.9058s` / `88.4519s`.
- `VLLM_ASCEND_ROLLOUT_WEIGHT_PREFETCH=1` with MoE `gate_up` prefetch ratio
  `0.8` and attention prefetch ratios disabled: slower.  This tested the
  generic vLLM-Ascend `weight_prefetch_config` path for MoE gate-up weights on
  top of the current-best eager stack.  It reached the first threshold-control
  generate-only step but regressed to
  `generate_sequences=88.5259s`, `gen=105.7749s`,
  `rollout_output_time_s=105.7781s`; current-best step2/3 are
  `76.9058s` / `88.4519s`.  The run was interrupted after step1 because it was
  already clearly slower and had no sign of a fast warmup effect.  Log:
  `resample_result_16k_bs32_n16_eager_genonly_weightprefetch_20260514/logs/wjqwen30b-a3b-record_graph_save4eagle3_20260514133934.txt`.
  Do not enable MoE gate-up prefetch by default.
- `VLLM_ASCEND_USE_COMMON_FUSED_MOE=1`: failed during model initialization.
  This tested whether the old v0.11-style `common_fused_moe + ops/moe`
  wrapper was closer to the reference eager path than the current v0.14
  `ops/fused_moe` wrapper.  It was not viable with the current compiled-eager
  stack: workers entered torchair compiled graph execution and failed in
  `_C_ascend.npu_add_rms_norm_bias` with `aclnnAddRmsNormBias do tiling
  failed`.  Log:
  `resample_result_16k_bs32_n16_eager_genonly_commonmoe_20260514/logs/wjqwen30b-a3b-record_graph_save4eagle3_20260514140526.txt`.
  Do not repeat this path unless the common-MoE wrapper is isolated from
  torchair compilation or the RMSNorm graph break is fixed.

## In Progress / Planned
- 2026-05-15 diagnostic, corrected: isolate Qwen3 attention q/k RMSNorm cost.
  The real old eager reference is also Qwen3-MoE, so it also has
  `qkv -> q_norm/k_norm -> RoPE -> attention -> o_proj`.  The earlier note
  comparing against a Qwen2-MoE attention path was wrong and should not guide
  future work.  `VLLM_QWEN3_SKIP_QK_NORM=1` remains useful only as a
  numerically-invalid attribution probe; it must never be added to
  `eager_fast`.  Since old and new both execute q/k RMSNorm, the deployable
  optimization target is not "remove q/k norm", but reducing the current
  wrapper/layout/compile overhead around the same Qwen3 attention sequence.
- The matched old-vs-new MoE and decode-attention profile is complete.  The
  remaining gap is now scoped to decode forward internals:
  old runner forward is about `60-62ms`, current qwen3 eager is about
  `88-92ms`.  Decode-only attention FIA is tiny (`~0.09ms`) and should not be
  the next target.  New MC2 steady decode is slower than old
  (`~1.07ms` vs `~0.81ms` for the sampled dispatch+MLP+combine), but this
  alone is too small to explain the full gap.
- Next experiments should be narrow implementation A/B probes, not broad
  shell-flag sweeps:
  1. MC2 dispatcher/combine ABI differences between new
     `ops/fused_moe/token_dispatcher.py` and old `ops/moe/token_dispatcher.py`.
  2. Layer-level split of non-MoE forward work (qkv/o_proj, norms, router,
     shared expert, residual/reduce) if MC2-only probes do not move
     `ProfileExecuteDuration[Decode].forward`.
  3. Inductor cold/warm-cache validation remains inconclusive and should use
     `gen1` first, then repeat once with the same cache before judging.
- `VLLM_ASCEND_EAGER_OLD_FORWARD_CONTEXT=1` on top of the current best eager
  stack: not promising.  It reached rollout generation but produced no step
  metric after more than 4 minutes in the first threshold-control generate-only
  step, so it was interrupted.  Current-best step1 normally finishes around
  `101s` rollout / `84s generate_sequences`, so this was already clearly
  regressed before a full 3-step result.  Log:
  `resample_result_16k_bs32_n16_eager_genonly_oldctx_currentbest_20260514/logs/wjqwen30b-a3b-record_graph_save4eagle3_20260514124842.txt`.
  Do not repeat unless a code change makes old forward context selective rather
  than reverting the whole MoE-model eager context.

## Next Suspects

- Profiling note: vllm-ascend `ProfileExecuteDuration` is gated by
  `VLLM_ASCEND_MODEL_EXECUTE_TIME_OBSERVE`, while the shell script used to
  hard-set only `VLLM_MODEL_EXECUTE_TIME_OBSERVE=0`.  The script now bridges
  both names, so use `VLLM_MODEL_EXECUTE_TIME_OBSERVE=1` or
  `VLLM_ASCEND_MODEL_EXECUTE_TIME_OBSERVE=1` for internal prefill/decode
  breakdown runs.
- Current CPU/device metadata sync only gives a small gain, so remaining gap is
  more likely inside per-token forward/MoE/attention execution rather than
  rollout phase switching.
- Compare current new split-MoE dispatch/finalize implementation against old
  `moe_comm_method_new.py`, especially decode MC2 path internals, not just the
  selection threshold.  Existing evidence says the threshold is correct
  (`<=512` -> MC2, `>512` -> AllToAll), so the target is the dispatcher ABI and
  extra work around dispatch/combine.
- New best still differs from old MC2 dispatcher in a few hot-path details:
  old MC2 always passed A3 `x_active_mask`, used a fixed `global_bs=0`, and
  stored dispatch output directly.  `expand_scales` omission was tested and is
  not viable.
- `VLLM_ASCEND_NPUGRAPH_EX_ENABLE_QKNORM_ROPE_FUSION=1`: slower.  This wired the existing GraphEX qk-norm/RoPE replacement into the otherwise-empty `NpuGraphEXPassManager` and fixed the local `graph.recompiler()` typo to `graph.recompile()`.  The pass registered successfully on all 16 workers (`Configured 1 npu_graph_ex fusion passes`) and the run completed, so the path is functionally viable, but it regressed the current best.  Log:
  `resample_result_16k_bs32_n16_eager_gen1_npugraphex_qknorm_rope_20260514230950/logs/wjqwen30b-a3b-record_graph_save4eagle3_20260514230950.txt`.
  Step1: `generate_sequences=93.207s`, `gen=110.533s`, `rollout_output_time_s=110.536s`, slower than the current best one-step range (`generate_sequences` roughly low/mid 80s and best steady-state `76.9s`).  Keep this pass opt-in only; do not add it to `eager_fast`.

- `VLLM_QWEN3_USE_QKV_RMSNORM_ROPE=1`: not viable in its current manual form.
  This attempted a narrow fused `qkv split + q/k RMSNorm + RoPE` path using
  `torch.ops.vllm.qkv_rmsnorm_rope`, which should have preserved Qwen3
  numerics while addressing the q/k norm cost found by the skip-qk diagnostic.
  It did not reach rollout generation: the run stayed in
  `actor_rollout_init_model` for more than two hours and was interrupted.
  Log:
  `resample_result_16k_bs32_n16_eager_qkrope_manual_profile_20260515001249/logs/wjqwen30b-a3b-record_graph_save4eagle3_20260515001249.txt`.
  Do not repeat this exact manual fused path.  The useful conclusion remains
  that q/k norm is a real cost, but the viable route must be a lower-overhead
  norm kernel or a compiler pass that does not explode model initialization.
- `VLLM_QWEN3_SKIP_QK_NORM=1`: diagnostic only, numerically invalid, not a
  deployable optimization.  This bypassed Qwen3 attention q/k RMSNorm to
  measure how much q/k norm contributes to the current implementation's
  decode-forward time.  This does not represent an old-vs-new architectural
  difference, because the old eager Qwen3-MoE path also applies q/k RMSNorm.
  It confirmed that q/k norm is a visible cost in the new implementation, but
  not the whole answer and not a new best.  Log:
  `resample_result_16k_bs32_n16_eager_skipqknorm_profile_20260515000326/logs/wjqwen30b-a3b-record_graph_save4eagle3_20260515000327.txt`.
  Step1: `generate_sequences=86.250s`, `gen=103.647s`,
  `rollout_output_time_s=103.650s`; decode profile averaged
  `forward=84.088ms`, `sample=1.970ms`, `prepare=2.719ms`,
  `post=0.574ms`.  A comparable profile without the bypass was
  `forward=93.084ms`, `generate_sequences=95.717s`, so q/k norm explains a
  visible slice of the gap.  However, current best steady-state remains around
  `76.9s generate_sequences`, and the correct GraphEX qk-norm/RoPE fusion pass
  was slower, so do not add this to `eager_fast`.  The useful conclusion is:
  a correct low-overhead q/k-norm implementation/fusion is worth future work,
  but the immediate remaining gap still needs broader layer-level decomposition
  beyond attention norm.
- `VLLM_QWEN3_USE_TORCH_NPU_QK_RMSNORM=1`: no end-to-end gain.  This replaced
  only Qwen3 attention q/k RMSNorm calls with `torch_npu.npu_rms_norm`, while
  preserving the rest of the attention path.  The env was active on all
  rollout workers and did not fall back, but the one-step profile regressed.
  Log:
  `resample_result_16k_bs32_n16_eager_torchnpu_qknorm_profile_20260515001840/logs/wjqwen30b-a3b-record_graph_save4eagle3_20260515001855.txt`.
  Step1: `generate_sequences=94.131s`, `gen=111.428s`,
  `rollout_output_time_s=111.431s`; decode profile averaged
  `forward=92.016ms`, `sample=1.952ms`, `prepare=2.895ms`,
  `post=0.574ms`.  Some raw late-decode records dropped into the `70-80ms`
  range, but the run average did not improve.  Keep this probe off by default;
  q/k norm likely needs a true fused qkv/norm/RoPE path or broader layer
  fusion, not just swapping the standalone RMSNorm op.
- `VLLM_QWEN3_USE_VLLM_QK_RMSNORM=1`: no end-to-end gain.  This kept Qwen3
  q/k RMSNorm numerics but forced the new qwen3 model code to use the same
  `self.q_norm(...)` / `self.k_norm(...)` call shape as the old eager model,
  bypassing the newer optional `_qwen3_qk_rms_norm(...)` torch_npu probe layer.
  Log:
  `resample_result_16k_bs32_n16_eager_vllm_qknorm_gen1_20260515/logs/wjqwen30b-a3b-record_graph_save4eagle3_20260515192413.txt`.
  Result: `generate_sequences=85.736s`, `gen=102.911s`,
  `rollout_output_time_s=102.919s`.  This is not a current-best result, so the
  q/k norm helper wrapper is not the main gap.  The env-gated code was removed
  after the probe to keep the production attention path cleaner.
- `VLLM_QWEN3_PASS_ATTENTION_OUTPUT_SHAPE=1`: no end-to-end gain.  This
  explicitly passed `output_shape=q.shape` from `Qwen3MoeAttention.forward` to
  the new v0.14 `Attention.forward`, avoiding its dynamic `torch.Size(...)`
  construction from `num_tokens`.  Log:
  `resample_result_16k_bs32_n16_eager_attn_outputshape_gen1_20260515/logs/wjqwen30b-a3b-record_graph_save4eagle3_20260515193344.txt`.
  Result: `generate_sequences=82.714s`, `gen=99.840s`,
  `rollout_output_time_s=99.844s`.  This is slower than the current best path,
  so dynamic output-shape construction is not a meaningful part of the
  attention-wrapper gap.  The env-gated code was removed after the probe.
- `VLLM_QWEN3_USE_SIMPLE_ATTENTION_FORWARD=1`: no end-to-end gain.  This
  forced `Qwen3MoeAttention.forward` into an old-eager-like straight-line
  implementation (`qkv.split -> self.q_norm/self.k_norm -> RoPE -> Attention
  -> o_proj`) and bypassed the currently disabled qkv/rmsnorm/rope diagnostic
  branches.  Log:
  `resample_result_16k_bs32_n16_eager_simple_attn_forward_gen1_20260515/logs/wjqwen30b-a3b-record_graph_save4eagle3_20260515194053.txt`.
  Result: `generate_sequences=86.039s`, `gen=103.256s`,
  `rollout_output_time_s=103.260s`, slower than the current best path.  This
  rules out the disabled diagnostic branches in `Qwen3MoeAttention.forward` as
  the source of the compiled-eager slowdown.  The env-gated code was removed
  after the probe.
- `VLLM_ASCEND_QWEN3_DUMMY_SKIP_ATTENTION=1`: slower / no useful signal.
  This restored the old-stack dummy-forward shape where runtime dummy runs can
  skip decoder attention/norm and execute only the MoE branch.  The hypothesis
  was that current eager compile/profile dummy passes might be spending or
  capturing extra non-MoE work.  It did not improve rollout generation.
  Log:
  `resample_result_16k_bs32_n16_eager_dummy_skip_attn_profile_20260514/logs/wjqwen30b-a3b-record_graph_save4eagle3_20260514232915.txt`.
  Step1: `generate_sequences=93.351s`, `gen=110.627s`,
  `rollout_output_time_s=110.631s`; decode profile averaged
  `forward=89.679ms`, `sample=1.950ms`, `prepare=2.984ms`,
  `post=0.573ms`.  MC2 timings stayed in the same band as current best
  (`tokens=32 total=1.084ms`, `tokens=512 total=1.911ms`).  Keep this code
  env-gated and off by default; do not add it to `eager_fast`.
- `VLLM_QWEN3_MOE_ASCEND_LEGACY_INIT=1`: slower.  This switched the Qwen3 MoE
  block constructor back to the older `AscendFusedMoE` wrapper instead of the
  current generic `FusedMoE` wrapper.  It was a direct MoE-forward hypothesis,
  not a scheduler flag, but it regressed the profile run.  Log:
  `resample_result_16k_bs32_n16_eager_legacyinit_profile_20260514233953/logs/wjqwen30b-a3b-record_graph_save4eagle3_20260514233953.txt`.
  Step1: `generate_sequences=91.559s`, `gen=108.589s`,
  `rollout_output_time_s=108.596s`; decode profile averaged
  `forward=91.168ms`, `sample=1.949ms`, `prepare=2.857ms`,
  `post=0.572ms`.  MoE stage showed extra warmup/profile
  `ALLTOALL/tokens=512` records and no forward improvement.  Keep the current
  default `VLLM_QWEN3_MOE_ASCEND_LEGACY_INIT=0`.

- Direct DecodeOnly attention wrapper bypass probe, 2026-05-15:

  Env: `VLLM_QWEN3_DIRECT_DECODE_ATTENTION=1` on top of current `eager_fast`,
  log directory `resample_result_16k_bs32_n16_eager_direct_decode_attn_gen1_20260515004002`.

  Hypothesis: old custom Ascend Qwen3 called `self.attn.impl.forward(...)`
  directly for `DecodeOnly`, while current qwen3 goes through the generic
  `Attention` wrapper / opaque custom op.  A narrow opt-in model-level bypass
  was tested.

  Result: not viable.  The run entered `VLLM_COMPILE` and spent ~24-27s in
  torch.compile, then initialized KV cache, but did not reach rollout metrics;
  after several minutes the log stayed at KV-cache initialization and NPU
  AICore was idle.  The probe was interrupted and the opt-in code was removed.

  Conclusion: do not repeat the direct model-level attention-impl bypass under
  current compiled-eager.  It likely changes the Dynamo/AscendCompiler graph
  shape enough to break the stable compiled-eager path.  The old direct impl
  path is not a drop-in optimization for v0.14 eager.

- Nested Ascend attention custom-op boundary probe, 2026-05-15:

  Env: `VLLM_ASCEND_USE_NESTED_ATTENTION_OP=1` on top of current `eager_fast`,
  log `resample_result_16k_bs32_n16_eager_nested_attnop_profile_20260515/logs/wjqwen30b-a3b-record_graph_save4eagle3_20260515092911.txt`.

  Hypothesis: old eager Ascend attention had an outer `vllm.unified_attention_with_output`
  and an inner `vllm.unified_ascend_attention_with_output` PrivateUse1 custom-op boundary.
  New eager calls the Ascend impl directly from the generic attention op.  Reintroducing
  this inner boundary might recover the old attention-op timing.

  Result: valid but slower.  The run compiled and completed one threshold-control
  generate-only/profile step, but regressed to `generate_sequences=95.371s`,
  `gen=112.529s`, `rollout_output_time_s=112.532s`.  Decode profile averaged
  `forward=92.106ms`, `sample=1.961ms`, `prepare=2.934ms`, `post=0.574ms`.
  Attention FIA timing itself stayed tiny (`DecodeOnly/fia/tokens<=32 call>1=0.091ms`),
  so the extra custom-op boundary did not recover the layer-level attention total and
  instead worsened the compiled-eager graph shape.

  Conclusion: keep `VLLM_ASCEND_USE_NESTED_ATTENTION_OP=0` by default.  Do not repeat
  this path unless the compiler boundary can be changed without adding the nested
  runtime custom-op dispatch.

- `VLLM_SKIP_KV_TRANSFER_DECORATOR=1`: slower / no benefit.  This was a tiny
  env-gated probe in `vllm/attention/utils/kv_transfer_utils.py` to bypass the
  v0.14 KV-transfer decorator wrapper around `unified_attention_with_output`
  when no KV connector is active.  Hypothesis: the extra decorator checks might
  explain part of the new attention wrapper gap.  Log:
  `resample_result_16k_bs32_n16_eager_gen1_skip_kv_transfer_decorator_20260515/logs/wjqwen30b-a3b-record_graph_save4eagle3_20260515103306.txt`.
  Result: `generate_sequences=88.212s`, `gen=105.213s`,
  `rollout_output_time_s=105.217s`, slower than the current best gen1 range.
  Keep the env-gated bypass off by default and do not add it to `eager_fast`.
  The remaining attention gap is not explained by the Python KV-transfer
  decorator alone.

- `VLLM_ASCEND_USE_LEGACY_ATTENTION_OP_SCHEMA=1`: slower / no benefit.  This
  temporarily registered a v0.11-style five-argument
  `unified_attention_with_output_legacy_schema` custom op to test whether the
  newer optional `output_scale` / `output_block_scale` arguments and schema
  mutation annotation were hurting compiled-eager attention boundaries.  Log:
  `resample_result_16k_bs32_n16_eager_gen1_attn_legacy_schema_20260515/logs/wjqwen30b-a3b-record_graph_save4eagle3_20260515104202.txt`.
  Result: `generate_sequences=86.414s`, `gen=103.828s`,
  `rollout_output_time_s=103.832s`, slower than current best.  The diagnostic
  code was removed after the run.  Do not repeat this schema-only path.

- `VLLM_ASCEND_FORCE_DIRECT_ATTENTION_IMPL=1`: fast threshold diagnostic, not
  promoted to production `eager_fast`.
  This is a lower-level wrapper probe in
  `vllm/attention/layer.py::Attention.forward`, not the earlier model-level
  `VLLM_QWEN3_DIRECT_DECODE_ATTENTION` probe.  It bypasses the v0.14 opaque
  `torch.ops.vllm.unified_attention_with_output` wrapper and directly calls
  `self.impl.forward(...)`, matching the old eager wrapper shape more closely
  while preserving the normal Qwen3 model code and Ascend attention backend.
  One-step gen-only log:
  `resample_result_16k_bs32_n16_eager_direct_attn_impl_gen1_20260515/logs/wjqwen30b-a3b-record_graph_save4eagle3_20260515112433.txt`.
  Result: `generate_sequences=69.661s`, `gen=87.171s`,
  `rollout_output_time_s=87.180s`, `speed=2744.851 tok/s`.  This is the best
  one-step signal so far and close to the old threshold rollout range.

  Three-step threshold log:
  `resample_result_16k_bs32_n16_eager_direct_attn_impl_threshold3_20260515/logs/wjqwen30b-a3b-record_graph_save4eagle3_20260515112931.txt`.
  Step2/3 were `generate_sequences=65.077/64.487s`,
  `gen=76.723/75.852s`, `rollout_output_time_s=76.732/75.855s`.
  This is the first qwen3 eager path that clearly beats the old threshold
  rollout walltime, while still leaving a pure `generate_sequences` gap of
  about `7-8s`.

  Follow-up full16k validation:
  `resample_result_16k_bs32_n16_eager_direct_attn_impl_full16k_20260515/logs/wjqwen30b-a3b-record_graph_save4eagle3_20260515131959.txt`.
  It is not comparable to the old full16k reference because it generated only
  `1,134,416` tokens (`mean~=2215.7`, no 16k-capped samples), while the old
  full16k reference generated `3,069,063` tokens (`mean~=5994.3`, 28 capped
  samples).  Keep direct attention as a diagnostic knob only until the response
  distribution mismatch is explained.

- `VLLM_ASCEND_FORCE_DIRECT_PREFILL_ATTENTION_IMPL=1`: fast threshold
  diagnostic, but still not behavior-preserving on full16k.  This variant kept
  `DecodeOnly` on the opaque `torch.ops.vllm.unified_attention_with_output`
  path and bypassed the v0.14 attention custom-op boundary only for
  prefill/chunked-prefill states.  The intent was to keep decode KV/update
  semantics intact while testing whether direct attention's threshold win came
  mainly from the prefill-side wrapper.
  One-step log:
  `resample_result_16k_bs32_n16_eager_direct_prefill_attn_gen1_20260515/logs/wjqwen30b-a3b-record_graph_save4eagle3_20260515213333.txt`.
  Result: `generate_sequences=70.580s`, `gen=87.633s`,
  `rollout_output_time_s=87.636s`, `speed=2730.559 tok/s` with the expected
  threshold token count `239,296`.
  Three-step threshold log:
  `resample_result_16k_bs32_n16_eager_direct_prefill_attn_threshold3_20260515/logs/wjqwen30b-a3b-record_graph_save4eagle3_20260515213922.txt`.
  Step2/3 were `generate_sequences=64.394/64.074s`,
  `gen=76.059/75.659s`, `rollout_output_time_s=76.066/75.662s`, matching the
  direct-all threshold speedup.
  Full16k validation log:
  `resample_result_16k_bs32_n16_eager_direct_prefill_attn_full16k_20260515/logs/wjqwen30b-a3b-record_graph_save4eagle3_20260515214704.txt`.
  Result: not comparable.  It generated only `1,134,416` tokens and the probe
  summary reported `rollout_output_time_s=754.914s`, exactly the same
  short-output scale as the earlier direct-all full16k failure.  Therefore the
  semantic break is not limited to DecodeOnly bypass; bypassing the opaque
  attention boundary during prefill/chunked-prefill is enough to corrupt the
  later response distribution.  Do not promote this knob to `eager_fast`.
  Future attention work must preserve the `unified_attention_with_output`
  boundary side effects and optimize inside or immediately around that boundary
  rather than replacing it with `self.impl.forward(...)`.

- `VLLM_ASCEND_DECODE_FIA_USE_OUT=1`: slower / no benefit.  This was an
  exact-shape DecodeOnly FIA `.out` probe intended to test whether the
  `npu_fused_infer_attention_score` allocation plus `output[:num_tokens] =
  attn_output` copy explained the new attention wrapper gap.  Log:
  `resample_result_16k_bs32_n16_eager_decode_fia_out_gen1_20260515/logs/wjqwen30b-a3b-record_graph_save4eagle3_20260515152604.txt`.
  Result: `generate_sequences=85.445s`, `gen=102.593s`,
  `rollout_output_time_s=102.597s`, slower than the current best path.  The
  branch was removed after the run.  Do not repeat this path unless the NPU
  operator gains a compatible no-regression `.out` shape.

- `VLLM_ASCEND_DECODE_FIA_NO_MASK=1`: invalid.  This tested whether the
  current DecodeOnly FIA path could omit `atten_mask` like the old eager
  DecodeOnly paged-attention path did.  Probe:
  `resample_result_16k_bs32_n16_eager_probe_gen1_20260515210346/logs/wjqwen30b-a3b-record_graph_save4eagle3_20260515210346.txt`.
  Result: first decode step failed in `aclnnFusedInferAttentionScoreV3`
  (`ERR00100`, error code `561002`) across workers.  This confirms the current
  compiled-eager FIA V3 shape requires the mask argument; do not retry this
  as a perf knob.

- `VLLM_ASCEND_DECODE_FAST_METADATA=1`: slower / no benefit.  This returned a
  reduced `AscendMetadata` for non-legacy `DecodeOnly` attention to skip
  device-side `query_start_loc`, `query_lens`, SWA mask, and some split-count
  fields while preserving the current FIA mask contract.  Probe:
  `resample_result_16k_bs32_n16_eager_probe_gen1_20260515211006/logs/wjqwen30b-a3b-record_graph_save4eagle3_20260515211006.txt`.
  Result: valid but slower, `generate_sequences=83.823s`,
  `gen=101.047s`, `rollout_output_time_s=101.051s`, speed
  `2630.447 tok/s`.  This rules out the extra DecodeOnly metadata fields as a
  direct win on the current compiled-eager path.  The probe branch was removed
  after the run.

- Attention hotpath cleanup probe: slower / no benefit.  This temporarily
  skipped the disabled debug-gate helper calls in the attention hot path
  (`_should_capture_attention_stage_timing`,
  `_should_capture_attention_wrapper_timing`,
  `_should_capture_attention_fia_detail_timing`, `_maybe_log_attention_path`)
  when all corresponding debug envs were off.  Log:
  `resample_result_16k_bs32_n16_eager_attn_hotpath_clean_gen1_20260515/logs/wjqwen30b-a3b-record_graph_save4eagle3_20260515153731.txt`.
  Result: `generate_sequences=89.822s`, `gen=107.194s`,
  `rollout_output_time_s=107.198s`, slower than current best.  Likely changed
  the compiled-eager graph/cache shape more than it saved Python overhead.  The
  code was reverted; do not repeat as a generic cleanup.

- `VLLM_ASCEND_REUSE_QUERY_START_LOC=1`: slower / no benefit.  This reused the
  device-side `common_attn_metadata.query_start_loc` instead of the old/new
  default CPU-to-NPU copy from `query_start_loc_cpu`.  Log:
  `resample_result_16k_bs32_n16_eager_reuse_qsl_gen1_20260515/logs/wjqwen30b-a3b-record_graph_save4eagle3_20260515154419.txt`.
  Result: `generate_sequences=88.375s`, `gen=105.567s`,
  `rollout_output_time_s=105.576s`, slower than current best.  This rules out
  query-start-location H2D copy as the main attention gap.  The probe branch
  was removed after the run.

- `VLLM_ASCEND_USE_ASCEND_ATTENTION_OP=1`: slower / no benefit.  This routed
  `Attention.forward` directly to the Ascend-specific
  `torch.ops.vllm.unified_ascend_attention_with_output` custom op instead of
  the generic v0.14 `unified_attention_with_output`, while still preserving a
  custom-op boundary.  Log:
  `resample_result_16k_bs32_n16_eager_ascend_attn_op_gen1_20260515/logs/wjqwen30b-a3b-record_graph_save4eagle3_20260515155400.txt`.
  Result: `generate_sequences=86.481s`, `gen=103.513s`,
  `rollout_output_time_s=103.516s`, slower than current best.  This rules out
  the generic op name / outer wrapper alone as the main gap.  The more
  interesting signal remains the fast but behavior-changing direct-impl path.

- `VLLM_ASCEND_USE_ASCEND_ATTENTION_MUTABLE_KV_OP=1`: failed during Dynamo fake
  tensor tracing, not a usable probe as implemented.  The idea was to make the
  custom-op boundary explicitly mutate KV cache tensors instead of letting the
  op fetch the cache from `layer.kv_cache` internally.  First attempt failed
  because `kv_cache` was a tuple/list and `mutates_args` expects tensors.  The
  follow-up passed `key_cache` and `value_cache` separately, but Dynamo traced
  the profile/dummy run while `self.kv_cache[virtual_engine]` was still an empty
  fake tensor and failed at `self_kv_cache[0]`.  Log:
  `resample_result_16k_bs32_n16_eager_attn_mutable_kv2_gen1_20260515/logs/wjqwen30b-a3b-record_graph_save4eagle3_20260515160734.txt`.
  This does not prove explicit mutable KV is slow, but the implementation would
  require changing the Attention forward contract/profile path just to test it.
  Retired for now to avoid high-risk plumbing changes; revisit only if we can
  pass KV tensors into the custom op from an existing compile-safe path.

- `VLLM_ASCEND_FIA_TENSOR_SEQ_PARAMS=1`: slower / no benefit.  This narrowly
  tested whether the DecodeOnly FIA path was paying meaningful overhead for
  Python-list sequence metadata by passing tensor forms of `actual_seq_lengths`
  and `actual_seq_lengths_kv` instead.  Log:
  `resample_result_16k_bs32_n16_eager_fia_tensor_seq_gen1_20260515/logs/wjqwen30b-a3b-record_graph_save4eagle3_20260515161726.txt`.
  Result: `generate_sequences=120.816s`, `gen=138.173s`,
  `rollout_output_time_s=138.176s`, much slower than the current best path.
  The branch was reverted.  Do not repeat this path; the NPU FIA op either
  prefers the list form or the tensor metadata changes the compiled graph in an
  unfavorable way.

- `VLLM_ASCEND_LEAN_ATTENTION_MUTATES=1`: slower / no benefit.  This kept the
  real custom-op boundary but removed the unused `output_block_scale` entry from
  `mutates_args` for the attention output custom ops, matching the old eager
  Ascend op's `output`-only mutable declaration more closely.  Log:
  `resample_result_16k_bs32_n16_eager_lean_attn_mutates_gen1_20260515/logs/wjqwen30b-a3b-record_graph_save4eagle3_20260515162658.txt`.
  Result: `generate_sequences=90.996s`, `gen=108.183s`,
  `rollout_output_time_s=108.187s`, slower than current best.  The branch was
  reverted.  The unused output-quant mutable arg is not the main compiled-eager
  attention gap.

- `VLLM_ASCEND_FORCE_DIRECT_ATTENTION_IMPL=1` +
  `VLLM_ASCEND_ZERO_ATTENTION_OUTPUT=1`: mixed diagnostic, not a production
  win.  This tested whether the fast-but-behavior-changing direct attention
  path was exposing uninitialized padding from the new `torch.empty` attention
  output buffer.  Log:
  `resample_result_16k_bs32_n16_eager_direct_zero_gen1_20260515/logs/wjqwen30b-a3b-record_graph_save4eagle3_20260515163927.txt`.
  Result: `generate_sequences=72.262s`, `gen=89.894s`,
  `rollout_output_time_s=89.897s`.  It is slower than direct attention alone
  and does not immediately beat the current best threshold-control path.  The
  length/debug summaries still show repeated-prompt identical token tails,
  though the threshold caps make this a weak semantic check.  Keep the
  zero-output hook only as a follow-up diagnostic for direct-attention full16k
  semantics; do not enable it in normal eager.
  Full16k follow-up log:
  `resample_result_16k_bs32_n16_eager_direct_zero_full16k_20260515164442/logs/wjqwen30b-a3b-record_graph_save4eagle3_20260515164443.txt`.
  Result: still not behavior-preserving.  It generated only `1,134,416`
  tokens with `rollout_output_time_s=768.871s` and per-rank mean length
  `2137.5`; all sampled ranks reported `finish_reason_counts={'stop': 32}`
  and no 16k-capped samples.  This matches the earlier direct-attention
  full16k short-output failure.  Zero-initializing the output buffer is not the
  missing semantic fix; do not promote direct attention unless the hidden KV /
  metadata side-effect issue is solved.

- `VLLM_ASCEND_ZERO_ATTENTION_OUTPUT=1` on the normal behavior-preserving
  opaque attention path: slower / no benefit.  This tested the only small
  output-buffer difference between old eager `Attention.forward` and the new
  v0.14 wrapper that had not been cleanly isolated: old used
  `torch.zeros(...)` for the attention output buffer, while new uses
  `torch.empty(...)`.  Log:
  `resample_result_16k_bs32_n16_eager_zero_output_normal_gen1_20260515/logs/wjqwen30b-a3b-record_graph_save4eagle3_20260515184937.txt`.
  Result: `generate_sequences=83.667s`, `gen=100.992s`,
  `rollout_output_time_s=100.996s`, slower than the current best path.  This
  rules out uninitialized attention output padding as the behavior-preserving
  attention-boundary performance gap.

- `VLLM_ASCEND_FORCE_DIRECT_DECODE_ATTENTION_IMPL=1`: slower / no benefit.
  This kept prefill and chunked-prefill on the normal opaque
  `unified_attention_with_output` path, but bypassed that boundary only for
  `DecodeOnly` metadata.  It was meant to test whether direct-all attention's
  threshold speedup could be recovered without the full16k semantic break.
  Log:
  `resample_result_16k_bs32_n16_eager_direct_decode_impl_gen1_20260515b/logs/wjqwen30b-a3b-record_graph_save4eagle3_20260515190150.txt`.
  Result: `generate_sequences=86.597s`, `gen=104.265s`,
  `rollout_output_time_s=104.269s`, slower than the current best path.  This
  rules out a narrow DecodeOnly-only direct impl bypass as a candidate default;
  direct-all attention remains a non-behavior-preserving diagnostic only.

- `VLLM_ASCEND_USE_EXPLICIT_KV_ATTENTION_OP=1`: slower / no benefit.  This
  kept the behavior-preserving opaque attention custom-op boundary, but passed
  key/value cache tensors explicitly into a new env-gated custom op and marked
  them mutable, so the compiler could see KV-cache side effects instead of
  retrieving KV from Python forward context only.  The branch guarded profile
  / dummy runs by falling back when KV cache was still an empty placeholder, so
  it avoided the earlier fake-tensor failure.  Log:
  `resample_result_16k_bs32_n16_eager_explicit_kv_attn_gen1_20260515/logs/wjqwen30b-a3b-record_graph_save4eagle3_20260515170405.txt`.
  Result: `generate_sequences=88.346s`, `gen=105.372s`,
  `rollout_output_time_s=105.376s`, slower than current best.  This rules out
  the hidden-KV-cache mutation contract as the main recoverable attention gap.
  Keep this branch off by default and do not run full16k.

- `VLLM_ASCEND_FORCE_DIRECT_ATTENTION_IMPL=1` +
  `VLLM_ASCEND_DIRECT_ATTENTION_GRAPH_BREAK=1`: failed during init, not a
  usable semantic-repair path.  The intent was to keep the fast direct
  Attention impl shape but put the impl call behind a no-Dynamo island so KV
  cache side effects would not be incorrectly captured.  Log:
  `resample_result_16k_bs32_n16_eager_direct_graphbreak_gen1_20260515/logs/wjqwen30b-a3b-record_graph_save4eagle3_20260515171653.txt`.
  Result: Dynamo raised `Unsupported: Skip calling torch.compiler.disable()d
  function` from the compiled Qwen3 forward during the dummy/profile run.
  This confirms the current compiled-eager wrapper cannot use a simple
  `torch._dynamo.disable` attention island.  The branch was removed.  Do not
  repeat this exact graph-break variant.

- `VLLM_ASCEND_USE_CUDAGRAPH_UNSAFE_ATTENTION_TAG=1`: slower / no benefit.
  This restored the old-stack `tag_cudagraph_unsafe` registration tag on
  `unified_attention` and `unified_attention_with_output` while keeping the
  behavior-preserving custom-op boundary.  Log:
  `resample_result_16k_bs32_n16_eager_cudagraph_unsafe_tag_gen1_20260515/logs/wjqwen30b-a3b-record_graph_save4eagle3_20260515172148.txt`.
  Result: `generate_sequences=87.979s`, `gen=105.396s`,
  `rollout_output_time_s=105.400s`, slower than current best.  The missing
  old-stack custom-op tag does not explain the remaining attention gap.

- `VLLM_ASCEND_LEAN_ATTENTION_FORWARD=1`: slower / no benefit.
  This added a narrow Qwen3 eager hot path in `vllm/attention/layer.py` that
  kept the behavior-preserving `unified_attention_with_output` opaque custom-op
  boundary and KV side-effect semantics, but skipped some v0.14 generic
  output-shape and optional dispatch branches in the common unquantized
  decoder-attention case.  Log:
  `resample_result_16k_bs32_n16_eager_lean_attn_forward_gen1_20260515/logs/wjqwen30b-a3b-record_graph_save4eagle3_20260515195344.txt`.
  Result: `generate_sequences=84.705s`, `gen=101.965s`,
  `rollout_output_time_s=101.968s`, slower than the current best.
  Conclusion: the remaining attention gap is not explained by the outer
  `Attention.forward` shape/branch plumbing.  Keep this env off by default and
  continue investigating the real backend implementation / metadata path.

- `VLLM_ASCEND_LEAN_ATTENTION_BACKEND_FORWARD=1`: slower / no benefit.
  This added an even narrower hot path inside
  `vllm_ascend/attention/attention_v1.py::AscendAttentionBackendImpl.forward`:
  preserve `reshape_and_cache()` and `forward_impl()` semantics, but skip the
  default-off wrapper/sync timing setup and optional branches.  Log:
  `resample_result_16k_bs32_n16_eager_lean_attn_backend_gen1_20260515/logs/wjqwen30b-a3b-record_graph_save4eagle3_20260515200722.txt`.
  Result: `generate_sequences=85.232s`, `gen=102.766s`,
  `rollout_output_time_s=102.770s`, slower than the current best.
  Conclusion: the remaining gap is not explained by backend-forward Python
  branch plumbing either.  The probe code was removed; do not repeat this lean
  backend wrapper experiment.

### 2026-05-15 Legacy Qwen3Moe stack compatibility follow-up

- `VLLM_QWEN3_MOE_ASCEND_LEGACY_STACK=1`, compile disabled, after adding `use_aclgraph=False` to `AscendUnquantizedFusedMoEMethod`:
  - Log: `resample_result_16k_bs32_n16_eager_legacy_stack_useaclgraphfix_eagererr_gen1_20260515/logs/wjqwen30b-a3b-record_graph_save4eagle3_20260515123300.txt`
  - Result: engine init completed and entered generation; the run was manually SIGTERM-ed because compile-off is not a performance target.
  - Conclusion: `use_aclgraph` was a real interface mismatch and the minimal compatibility patch is valid enough to pass initialization.
- Same stack with `VLLM_ASCEND_EAGER_COMPILE=1`:
  - Log: `resample_result_16k_bs32_n16_eager_legacy_stack_useaclgraphfix_compile_gen1_20260515/logs/wjqwen30b-a3b-record_graph_save4eagle3_20260515123720.txt`
  - Result: failed in Dynamo during dummy/profile run at `_execute_single_wave`: `if torch.any(invalid_mask)` produced `Data-dependent branching`.
  - Action: made this remap validity check opt-out for the legacy-stack performance probe via `VLLM_QWEN3_MOE_SKIP_INVALID_TOPK_CHECK`, defaulting to enabled only when `VLLM_QWEN3_MOE_ASCEND_LEGACY_STACK=1`.
- Follow-up compiled legacy-stack probes after skipping the Python guards:
  - Logs:
    `resample_result_16k_bs32_n16_eager_legacy_stack_tokdisp_compile_gen1_20260515/logs/wjqwen30b-a3b-record_graph_save4eagle3_20260515125623.txt`
    and
    `resample_result_16k_bs32_n16_eager_legacy_stack_maskalign_compile_gen1_20260515/logs/wjqwen30b-a3b-record_graph_save4eagle3_20260515130128.txt`.
  - Result: both reached the real MC2 dispatch op and failed in
    `aclnnMoeDistributeDispatchV4` with `ERR00100`.  Aligning the local
    `mc2_mask` to `moe_comm_method.mc2_mask` after `prepare()` did not fix the
    ABI mismatch.
- Compile-off 2k diagnostic:
  - Log:
    `resample_result_16k_bs32_n16_eager_legacy_stack_compileoff_2k_gen1_20260515/logs/wjqwen30b-a3b-record_graph_save4eagle3_20260515131124.txt`.
  - Env: `VLLM_ASCEND_EAGER_COMPILE=0`,
    `VLLM_QWEN3_MOE_ASCEND_LEGACY_STACK=1`, `MAX_RESPONSE_LENGTH=2048`.
  - Result: initialized and entered `actor_rollout_generate_sequences`, but
    produced no metric or Python error after more than four minutes; one worker
    reached uninterruptible `D` state.  This is not a viable cheap path and
    supports the conclusion that the current `fused_moe_legacy.py` bridge is
    not ABI-compatible with the real old eager MoE runtime.
- Important correction: this is not a reason to compare against torchair.
  The real old eager path remains
  `ref_wj_qwen3/qwen30b_tpu_verl/vllm_ascend/models/qwen3_moe.py ->
  vllm_ascend.ops.fused_moe.AscendFusedMoE`.  The current
  `qwen3/vllm_ascend/ops/fused_moe_legacy.py` file is not a faithful copy of
  that old eager op file; it still carries current-stack hybrid/mode3 and debug
  changes.  Do not continue bypassing guards blindly.  If this direction is
  revisited, make a faithful old-eager op port first, then test with 2k
  compile-off before enabling compiled eager.

### 2026-05-15 Rejected: common fused MoE custom-op path

- Real-path correction: current qwen3 eager registers `FusedMoE` through
  `vllm_ascend.utils.register_ascend_customop`.  With the default env it uses
  `vllm_ascend.ops.fused_moe.fused_moe.AscendFusedMoE`, not the old flat
  `ops/fused_moe.py` file.
- `VLLM_ASCEND_USE_COMMON_FUSED_MOE=1` was considered because
  `common_fused_moe.forward_oot` has an old-style-looking shape
  (`select_experts -> row_idx -> moe_comm_method.fused_experts -> finalize`).
  Earlier probes already showed this path is not currently usable for the
  qwen3 eager workload: it failed during initialization or compiled execution
  due to interface/shape mismatches with the current v0.14 fused-MoE and
  `Qwen3Moe` call contract.
- Do not repeat this as a generic "old-style MoE" experiment.  If the old eager
  MoE runtime is revisited, port the real old file
  `ref_wj_qwen3/qwen30b_tpu_verl/vllm_ascend/ops/fused_moe.py` faithfully, then
  validate it with a 2k compile-off diagnostic before enabling compiled eager.

### 2026-05-15 Rejected: old-schema nested attention op boundary

- `VLLM_ASCEND_USE_OLD_NESTED_ATTENTION_OP=1` tested a more faithful old-eager
  inner attention custom-op boundary than the earlier generic
  `VLLM_ASCEND_USE_NESTED_ATTENTION_OP=1` probe.  The probe added a
  five-argument `unified_ascend_attention_with_output_old_schema(query, key,
  value, output, layer_name)` PrivateUse1 op, with only `output` marked
  mutable, matching the old eager Ascend attention schema shape more closely
  than the current seven-argument v0.14 helper with output quant placeholders.
- Log:
  `resample_result_16k_bs32_n16_eager_old_nested_attn_schema_gen1_20260515/logs/wjqwen30b-a3b-record_graph_save4eagle3_20260515202700.txt`.
- Result: valid but slower.  One-step threshold-control generate-only produced
  `generate_sequences=85.107s`, `gen=102.433s`,
  `rollout_output_time_s=102.437s`, `speed=2594.847 tok/s`.
- Conclusion: the missing old eager performance is not recovered by restoring
  the old five-argument nested Ascend attention custom-op schema/boundary.
  Keep this env off and do not extend it to `threshold3` or full16k.

### 2026-05-15 Rejected: direct MoE `forward_impl` bypass

- `VLLM_ASCEND_FUSED_MOE_DIRECT_FORWARD_IMPL=1` tested whether the generic
  vLLM `torch.ops.vllm.moe_forward(layer_name)` trampoline itself was a
  measurable part of the new eager overhead.  The probe overrode
  `AscendFusedMoE.forward_oot()` to directly call the same Ascend
  `forward_impl(hidden_states, router_logits)` and preserve the existing
  reduction semantics.
- Log:
  `resample_result_16k_bs32_n16_eager_direct_moe_forwardoot_gen1_20260515/logs/wjqwen30b-a3b-record_graph_save4eagle3_20260515204142.txt`.
- Result: failed before first rollout metric during the vLLM dummy/profile run.
  Dynamo raised
  `ConstraintViolationError: Constraints violated (L['input_ids'].size()[0])`
  because the direct call let MoE prepare/finalize logic specialize a dynamic
  batch/token dimension to the profile-run constant `512`.
- Interpretation: the `torch.ops.vllm.moe_forward` trampoline is not just
  cosmetic plumbing in compiled eager; it is an important no-compile / layer
  lookup boundary that prevents current MoE prepare/finalize dynamic-shape
  logic from being traced into the compiled Qwen3 forward.  Do not repeat this
  bypass.  Future MoE work should stay inside the valid compiled-eager boundary
  or make a faithful old-eager op port with its own compatibility validation.

### 2026-05-15 Rejected: old eager `value.contiguous()` before KV cache

- Env: `VLLM_ASCEND_ATTENTION_VALUE_CONTIGUOUS=1`.
- Hypothesis: old eager attention made `value` contiguous before
  reshape/cache; restoring that layout might reduce KV write/read or decode
  attention fixed overhead.
- Log:
  `resample_result_16k_bs32_n16_eager_value_contiguous_gen1_20260515/logs/wjqwen30b-a3b-record_graph_save4eagle3_20260515221729.txt`.
- Result: slower.  One-step threshold-control generate-only produced
  `generate_sequences=85.686s`, `gen=103.058s`,
  `rollout_output_time_s=103.065s`, `speed=2579.028 tok/s`.
- Conclusion: the old eager `value.contiguous()` behavior is not the missing
  performance piece.  Keep the env off and do not extend to threshold3.

### 2026-05-15 Profiling plan: torch_npu operator traces

- Added `VLLM_ROLLOUT_TORCH_NPU_PROFILE=1` support around only
  `self.inference_engine.generate(...)` in both new and old
  `verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py`.
- This is rank-gated by `VLLM_ROLLOUT_TORCH_NPU_PROFILE_RANK` and writes to
  `VLLM_ROLLOUT_TORCH_NPU_PROFILE_DIR`, so it can capture one rollout rank
  without profiling actor/ref/update phases or all 16 workers.
- The qwen3 probe wrapper now has `internal/run_eager_perf_probe.sh
  npu_profile`.
- Next step: collect one new-stack trace and one old-stack trace under the
  same threshold-control workload, then compare CPU/NPU op tables before
  making further code changes.  This should replace broad knob sweeps.

### 2026-05-16 torch_npu profiler: old baseline vs new eager

- Old baseline profiler was run from the correct old eager entrypoint:
  `../ref_wj_qwen3/qwen30b_tpu_verl/internal/wj_train_grpo_qwen30b_a3b_16die_true_weight_regroup.sh`.
  Do not compare against ziyi scripts or the old `torchair/` model path for
  eager.  The old profile log confirms `Resolved architecture:
  Qwen3MoeForCausalLM` and `Compilation disabled, using eager mode by default`.
- New profile:
  `resample_result_16k_bs32_n16_eager_npu_profile_new_20260515/npu_profile_rank_0/.../op_summary_slice_{0,1}_20260515230035.csv`.
- Old profile:
  `../ref_wj_qwen3/qwen30b_tpu_verl/resample_result_16k_bs32_n16_old_npu_profile_20260515_old_profile/npu_profile_rank_0/.../op_summary_20260516000635.csv`.
- NPU op totals in the profiled rank-0 rollout-generate window:

  | op | old total/count/avg | new total/count/avg | delta |
  |---|---:|---:|---:|
  | `MoeDistributeDispatchV2` | `34.034s / 42960 / 792us` | `47.925s / 43008 / 1114us` | `+13.891s` |
  | `MoeDistributeCombineV2` | `1.882s / 42960 / 44us` | `9.215s / 43008 / 214us` | `+7.333s` |
  | `GroupedMatmul` | `2.476s` | `2.965s` | `+0.489s` |
  | `MatMul/BatchMatMul` | `0.942s` | `2.142s` | `+1.200s` |
  | decode attention op | old `PagedAttentionMaskNdKernel=0.487s` | new `FusedInferAttentionScore=0.571s` | small |

- This is the first strong operator-level explanation for the threshold
  `generate_sequences` gap: the bulk of the new-vs-old delta is inside MC2
  `MoeDistributeDispatchV2` and especially `MoeDistributeCombineV2`.
  Call counts are almost identical, so this is not caused by calling MoE many
  extra times.  The same NPU op names are slower per call under the new input
  contract.
- CPU/API statistics support this interpretation.  Host-side
  `aclnnInnerMoeDistributeDispatchV2` / `CombineV2` times are comparable
  between old and new, while task duration differs substantially.  Therefore
  this is not primarily Python/host launch overhead; the MC2 kernels themselves
  are taking slower internal paths with the current new-stack arguments or
  tensor layouts.
- Routing top-k is not the main gap.  Old has `MoeGatingTopKSoftmax=0.680s`
  while new uses `Softmax + Topk` around `1.095s`, which is visible but far
  smaller than the `~21s` combined MC2 dispatch/combine delta.
- Attention is not the main gap in this profile.  The old-vs-new attention
  operator difference is below one second in the profiled window; do not spend
  more time on paged-attention/FIA toggles unless MC2 is accounted for.
- Next optimization target: compare and minimize the real MC2 dispatch/combine
  input contract in the current eager path
  `qwen3/vllm_ascend/ops/fused_moe/{fused_moe.py,moe_comm_method.py,token_dispatcher.py}`
  against the old eager path
  `ref_wj_qwen3/qwen30b_tpu_verl/vllm_ascend/ops/{fused_moe.py,moe/token_dispatcher.py}`.
  Already-tested MC2 knobs that should not be repeated blindly include
  `expand_scales` omission, active-mask variants, `expert_num_source`,
  `expert_token_nums_type`, legacy dispatcher init, and fused-MC2 mode `2`.

### 2026-05-16 local custom OPP A/B

- Hypothesis from the old-vs-new torch_npu profile: qwen3 new stack prepended
  `vllm_ascend/_cann_ops_custom/vendors/vllm-ascend` to
  `ASCEND_CUSTOM_OPP_PATH` / `LD_LIBRARY_PATH`, while the old eager baseline
  script does not.  The profiler showed identical op names but much slower NPU
  task duration for `MoeDistributeDispatchV2` / `MoeDistributeCombineV2`, so a
  different local OPP implementation was a plausible root cause.
- Added `VLLM_ASCEND_USE_LOCAL_CUSTOM_OPP` gate in the qwen3 regroup base
  script.  The fast eager script now defaults it to `0`.
- `gen1` result with only this change:
  `resample_result_16k_bs32_n16_eager_nolocalopp_gen1_20260516003229/logs/wjqwen30b-a3b-record_graph_save4eagle3_20260516003229.txt`.
  Metrics: `generate_sequences=69.227s`, `gen=86.385s`,
  `rollout_output_time_s=86.388s`, `speed=3076.918 tok/s`,
  `tokens=265808`.
- `threshold3` validation:
  `resample_result_16k_bs32_n16_eager_nolocalopp_threshold3_20260516003739/logs/wjqwen30b-a3b-record_graph_save4eagle3_20260516003740.txt`.
  Metrics:
  - Step 1: `generate_sequences=68.099s`, `gen=85.265s`,
    `rollout=85.269s`, `speed=3117.305 tok/s`.
  - Step 2: `generate_sequences=60.054s`, `gen=71.735s`,
    `rollout=71.738s`, `speed=3678.260 tok/s`.
  - Step 3: `generate_sequences=61.615s`, `gen=73.032s`,
    `rollout=73.035s`, `speed=3599.806 tok/s`.
- Interpretation: this is the first change that directly explains and removes
  most of the old-vs-new eager gap.  It strongly supports the profiler-derived
  diagnosis that the bundled local custom OPP was putting
  `MoeDistributeDispatchV2` / `MoeDistributeCombineV2` on a slower internal
  path.  Next checks: run `npu_profile` with
  `VLLM_ASCEND_USE_LOCAL_CUSTOM_OPP=0`, then run one full-16k validation to
  confirm response-length behavior and end-to-end parity.
- `npu_profile` confirmation used the old eager baseline from exactly
  `ref_wj_qwen3/qwen30b_tpu_verl/internal/wj_train_grpo_qwen30b_a3b_16die_true_weight_regroup.sh`
  and compared it with qwen3 local-OPP on/off profiles.  The exported summary
  is in `internal/eager_profiler_op_compare_20260516.tsv`.
  Key rank-0 NPU task totals:
  - Old ref eager: `MoeDistributeDispatchV2=34.034s`,
    `MoeDistributeCombineV2=1.882s`, total profiled task time `47.138s`.
  - qwen3 local OPP on: `MoeDistributeDispatchV2=47.925s`,
    `MoeDistributeCombineV2=9.215s`, total profiled task time `72.541s`.
  - qwen3 local OPP off: `MoeDistributeDispatchV2=34.421s`,
    `MoeDistributeCombineV2=4.596s`, total profiled task time `54.906s`.
- Conclusion: disabling qwen3's bundled local custom OPP almost exactly
  restores the old baseline's dispatch latency and removes the largest measured
  NPU-side gap.  Combine is still ~2.4x old-ref on this profile window, but its
  absolute residual gap is only ~2.7s versus the ~13.5s dispatch improvement.
  The remaining validation blocker is full-16k behavior, not another broad
  script-knob sweep.

### 2026-05-16 graph local-OPP MoE A/B

- Question: the bundled local OPP may have been introduced to support or speed
  graph mode, so fully disabling it for graph might be unsafe.  Test graph mode
  with and without the local MoE dispatch/combine registrations while keeping
  graph-required local ops such as `AddRmsNormBias`.
- Full `VLLM_ASCEND_USE_LOCAL_CUSTOM_OPP=0` graph smoke is not viable:
  `resample_result_16k_bs32_n16_graph_fast_nolocalopp_8k_platformgate_20260516/logs/wjqwen30b-a3b-record_graph_save4eagle3_20260516015554.txt`
  fails during init with
  `The binary_info_config.json ... does not support opType [AddRmsNormBias]`.
  Interpretation: graph mode depends on at least part of qwen3's bundled local
  OPP package.
- Built a temporary filtered OPP package at
  `vllm_ascend/_cann_ops_custom_moe_filtered/vendors/vllm-ascend`.  It keeps
  graph-required local ops but removes the local `MoeDispatchNormal` and
  `MoeCombineNormal` registrations/kernels so MC2 dispatch/combine can fall
  back to the system CANN implementation.
- 8k graph smoke:
  `resample_result_16k_bs32_n16_graph_fast_filteredopp_no_moe_8k_20260516/logs/wjqwen30b-a3b-record_graph_save4eagle3_20260516020151.txt`
  completed with `rollout_output_time_s=493.593s`, same range as the existing
  graph 8k local-OPP baselines (`~495-508s`).
- Full 16k graph generate-only A/B, both `bs=32,n=16,max_response=16k`, same
  token total (`3060464`, 512 samples, 34 samples >=16k):
  - Full local OPP including local MoE dispatch/combine:
    `resample_result_16k_bs32_n16_graph_fast_localopp_moe_16k_20260516/logs/wjqwen30b-a3b-record_graph_save4eagle3_20260516093238.txt`,
    `rollout_output_time_s=1010.729s`, `speed=3027.977 tok/s`.
  - Filtered local OPP without local MoE dispatch/combine:
    `resample_result_16k_bs32_n16_graph_fast_filteredopp_no_moe_16k_20260516/logs/wjqwen30b-a3b-record_graph_save4eagle3_20260516095502.txt`,
    `rollout_output_time_s=1004.890s`, `speed=3045.572 tok/s`.
- Interpretation: graph does require bundled local OPP as a package, but not
  the bundled local MoE dispatch/combine implementation for performance.
  Removing those local MoE registrations did not slow graph mode and was
  slightly faster in this clean 16k A/B (`-5.84s`, about `0.6%`).  Therefore
  the local OPP bundle should not be treated as all-or-nothing: keep graph
  required ops, but avoid the local MoE dispatch/combine implementation by
  default unless a future graph profile proves otherwise.
- Follow-up implementation: the qwen3 base script, `ray_start_npu.sh`, and
  `vllm_ascend/platform.py` now default to
  `vllm_ascend/_cann_ops_custom_moe_filtered/vendors/vllm-ascend` when local
  OPP is enabled.  This package keeps graph-required local ops, removes local
  `MoeDispatchNormal` / `MoeCombineNormal` kernel registration, and strips any
  stale full local OPP path from `ASCEND_CUSTOM_OPP_PATH`.
- Important second-order fix: the filtered package's `op_api/lib/libcust_opapi.so`
  still exports `aclnnMoeDispatchNormal` / `aclnnMoeCombineNormal`.  Prepending
  that directory to `LD_LIBRARY_PATH` can still intercept the MoE API symbols
  even though the filtered `ops-info` no longer registers the local MoE kernels.
  The default is now:
  - `ASCEND_CUSTOM_OPP_PATH` includes the filtered OPP package, so local
    `AddRmsNormBias` remains available.
  - `LD_LIBRARY_PATH` does **not** prepend the filtered `op_api/lib` unless
    `VLLM_ASCEND_USE_LOCAL_CUSTOM_OP_API_LIB=1` is explicitly set.
  - `eager_fast` defaults `VLLM_ASCEND_USE_LOCAL_CUSTOM_OPP=1` and
    `VLLM_ASCEND_USE_LOCAL_CUSTOM_OP_API_LIB=0`, i.e. filtered OPP registration
    plus system CANN opapi for MoE dispatch/combine.

### 2026-05-16 eager filtered-OPP cache-key fix

- Symptom after the filtered-OPP/opapi split: rerunning `eager_fast` printed the
  expected header values
  `VLLM_ASCEND_USE_LOCAL_CUSTOM_OPP=1`,
  `VLLM_ASCEND_USE_LOCAL_CUSTOM_OP_API_LIB=0`, and
  `VLLM_ASCEND_LOCAL_CUSTOM_OPP_PATH=.../_cann_ops_custom_moe_filtered/...`,
  but the full 16k run was still slow:
  `resample_result_16k_bs32_n16_eager_fast/logs/wjqwen30b-a3b-record_graph_save4eagle3_20260516222954.txt`,
  `rollout_output_time_s=1477.588s`, `speed=2065.246 tok/s`.
- Diagnosis: this was not a length-distribution issue. The run used a vLLM
  torch.compile cache directory whose `cache_key_factors.json` /
  `computation_graph.py` timestamps were from an older experiment, before the
  filtered-OPP/opapi split. Upstream vLLM's compile cache key hashed registered
  vLLM env vars but did not include qwen3/vllm-ascend custom-op provider envs
  such as `ASCEND_CUSTOM_OPP_PATH`,
  `VLLM_ASCEND_USE_LOCAL_CUSTOM_OPP`,
  `VLLM_ASCEND_USE_LOCAL_CUSTOM_OP_API_LIB`, or
  `VLLM_ASCEND_LOCAL_CUSTOM_OPP_PATH`.
- Fix: `vllm/envs.py::compile_factors()` now includes those Ascend custom-op
  provider envs in the cache key. This prevents stale compiled-eager artifacts
  created under the full local OPP or local `libcust_opapi.so` path from being
  reused after switching to filtered OPP + system CANN opapi.
- Additional guard: `eager_fast` now defaults
  `VLLM_CACHE_ROOT=${PWD}/.cache/vllm_eager_fast` so production eager artifacts
  are separated from historical graph/eager probing caches. Future validation
  should confirm that the first rerun creates fresh cache directories under
  `.cache/vllm_eager_fast/torch_compile_cache` and then re-check full-16k
  `rollout_output_time_s`.

### 2026-05-17 eager OPP rollback note

- The attempted `_cann_ops_custom_add_rms_only` package was too aggressive:
  it registered `npu_add_rms_norm_bias` but missed the complete tiling path,
  causing `Do not find tiling func of AddRmsNormBias` during compiled-eager
  profile run initialization. Do not use this package as the eager default.
- The filtered local OPP package is valid for graph and can initialize eager,
  but the only experiment that directly restored old-like threshold speed used
  `VLLM_ASCEND_USE_LOCAL_CUSTOM_OPP=0`.
- Therefore production `eager_fast` is reset to the strict no-local-OPP
  direction with a separate cache root
  `.cache/vllm_eager_fast_nolocalopp`. If this later fails with
  `AddRmsNormBias` on a changed compiled-eager path, the fix should target that
  fusion path explicitly rather than introducing another partial OPP bundle.
- Follow-up: strict no-local OPP did fail once because compiled eager still
  emitted `_C_ascend.npu_add_rms_norm_bias`, which the system CANN OPP cannot
  run. The deployable fix is to pair no-local OPP with
  `VLLM_ASCEND_FORCE_TORCH_NPU_ADD_RMS_NORM=1`, keeping residual RMSNorm on
  `torch_npu.npu_add_rms_norm` instead of the local-only AddRmsNormBias custom
  op. A cheap 8k/1-step smoke
  `resample_result_16k_bs32_n16_eager_fast_smoke_20260517015305` completed
  successfully with `rollout_output_time_s=80.616s`, confirming that this
  combination initializes and generates without local OPP.
