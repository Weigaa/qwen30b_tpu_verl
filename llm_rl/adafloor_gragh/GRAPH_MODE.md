# AdaFloor ACLGraph Mode

This repository supports an experimental `elastic_aclgraph` rollout profile
for the Qwen3 mode-1 floor4 path. The production default captures the 48 elastic
Ascend MoE layers and surrounding dense work while keeping the 48 Attention
calls as graph boundaries. A topology owns only the ACLGraph instances for the
configured batch descriptors. Before a `16 -> 8`, `8 -> 4`, or restore
transition, the runtime synchronizes and destroys the current ACL resources. It
rebuilds communication groups, expert ownership, and runtime layout before
binding fresh graphs to the new topology. Weight and KV-layout changes use the
same fail-closed invalidation and late-recapture lifecycle.

`VLLM_ASCEND_ELASTIC_ACLGRAPH_CAPTURE_ATTENTION=1` is a correctness-gated,
opt-in path on this pinned 0.11 stack. An early isolation run changed every
response because PIECEWISE replay reused stale boundary-input addresses. The
runtime now refreshes validated capture buffers before replay, and a formal
Vanilla Full16 greedy pair passes exact token parity. The option is not a
production default until combined MoE plus Attention and elastic lifecycle
parity pass. `VLLM_ASCEND_ELASTIC_ACLGRAPH_CAPTURE_MOE=0` retains the diagnostic
mode in which each elastic MoE forward is a graph boundary.

The supported entry point is
`run_mode1_local_length_sorted_e2e_adaptive_floor4_aclgraph.sh`. It is a thin
wrapper around `run_mode1_local_length_sorted_e2e_adaptive_floor4.sh`; planning,
training orchestration, floor selection, and all user Hydra overrides continue
to use the existing runner.

## Scope

The graph applies only to vLLM rollout model execution. Megatron actor and
reference training remain on their existing eager path. Sidecar inference and
TorchAir/GE graph execution are not supported by this profile. The wrapper
rejects sidecar or TorchAir settings instead of silently running a different
configuration.

`VLLM_ENABLE_GRAPH_MODE=0` is intentional. In this tree that variable controls
TorchAir, not ACLGraph. ACLGraph is selected by
`ROLLOUT_ENFORCE_EAGER=False`, an explicit capture-size list, and
`VLLM_ASCEND_ELASTIC_ACLGRAPH=1`.

The copied 0.11 source tree contains an older prebuilt `vllm_ascend_C` that is
not ABI-compatible with the installed PyTorch. ACLGraph needs the extension's
non-owning output alias operation, so the wrapper loads the compatible local
build and verifies its SHA256 in every worker. Set
`VLLM_ASCEND_ELASTIC_ACLGRAPH_EXTENSION` explicitly when using another
environment. Missing or changed extension bytes fail closed; graph mode never
falls back to retaining captured outputs with strong tensor references.

This is a functional compatibility path. A successful smoke does not by itself
establish a throughput result or a paper claim.

## Native Full-Decode FIA Path

`run_qwen3_vanilla_epoch0_full_decode_fia_tq1.sh` is the fixed-topology
Vanilla entry point for the optimized native full graph. It ports the relevant
vLLM-Ascend 0.14 Qwen full-decode protocol to this pinned 0.11 tree:

- prefill stays eager and cannot dispatch to the decode graph;
- uniform single-token decode captures KV write, FIA Attention, MoE/HCCL, and
  the surrounding dense model in one outer `FULL_DECODE_ONLY` ACLGraph;
- FIA calls `_npu_fused_infer_attention_score_get_max_workspace` once while a
  capture shape is created, retains that fixed workspace address, and uses
  `npu_fused_infer_attention_score.out` during graph-task updates;
- `actual_seq_lengths_q` and KV sequence lengths are built on the host, padded
  to the capture shape, and replaced before each replay without an NPU-to-host
  synchronization in every Attention layer;
- the outer graph uses model-runner-owned static input buffers. FULL replay
  does not copy symbolic inputs per token and fails if an input, KV cache, or
  block-table address differs from the captured address;
- dummy capture temporarily fills the persistent slot-mapping buffer with
  `PAD_SLOT_ID`, preventing capture from corrupting live KV slots, then restores
  the same buffer before runtime replay.

The default production-sized launcher captures only shape 32. The workload has
32 local decode sequences per rollout worker at maximum, and lower occupancies
are padded to that shape. This minimizes the retained graph count to one per
rank. A different explicit list can be supplied with
`FULL_DECODE_CAPTURE_SIZES`, at the cost of one retained full graph per listed
shape and per rank.

Run the fail-closed preflight without starting Ray or NPU:

```bash
cd /workspace/cann-recipes-train/llm_rl/adafloor_gragh
FULL_DECODE_EPOCH0_ROOT=/workspace/adafloor_graph_results/full_fia_trial \
  ./run_qwen3_vanilla_epoch0_full_decode_fia_tq1.sh dry-run
```

Run the five-step epoch0 on an exclusive 16-NPU node:

```bash
cd /workspace/cann-recipes-train/llm_rl/adafloor_gragh
FULL_DECODE_EPOCH0_ROOT=/workspace/adafloor_graph_results/full_fia_trial \
  ./run_qwen3_vanilla_epoch0_full_decode_fia_tq1.sh run
```

After a completed run, the same script validates five rollout artifacts,
runtime graph markers, zero abort/preemption/OOM evidence, and writes a
diagnostic comparison with the historical eager epoch0:

```bash
FULL_DECODE_EPOCH0_ROOT=/workspace/adafloor_graph_results/full_fia_trial \
  ./run_qwen3_vanilla_epoch0_full_decode_fia_tq1.sh summarize
```

The launcher fixes seed 0, five steps, 32 prompts per step, `n=16`, response
limit 16384, 380800 KV tokens per rank, and actor learning rate `1e-6`. It uses
`TASK_QUEUE_ENABLE=1`, disables TorchAir/GraphEX and sidecar execution, and
selects `VLLM_ASCEND_FULL_DECODE_ATTENTION_BACKEND=fia_max_workspace`.
The historical eager comparison is diagnostic because it predates the current
source snapshot. A paper-quality speedup requires a fresh matched eager arm.

This migration has CPU/static contract coverage but has not yet completed its
representative 16-NPU five-step run. Until that run demonstrates exact workload
and output correctness plus the required runtime markers, it is an executable
candidate rather than an accepted performance result.

The pinned CANN 8.5.0, vLLM 0.11.0, and vLLM-Ascend 0.11.0rc0 stack passed a
fixed16 MoE-only capture parity test and earlier two-step lifecycle smokes.
The Planned and Natural roots are respectively
`/data/adafloor_graph_moe_planned_smoke/20260811T085812Z/lifecycle_full_moe_aclgraph`
and
`/data/adafloor_graph_moe_natural_smoke/20260811T091000Z/lifecycle_full_moe_aclgraph`.
The earlier lifecycle summaries are `PASS`, but their Attention-inclusive token
outputs are not accepted as correctness evidence. The decisive operator
isolation is under
`/data/adafloor_shared_state/qwen3_aclgraph_operator_isolation_20260812T_formal`.
For the same 16 prompts and 1,024 greedy output tokens, MoE-only capture has the
same response SHA256 as eager, while Attention-only capture has a different
SHA256. The MoE-only micro-run reduced measured inner generation time from
15.695 s to 9.586 s. This tiny single run establishes the safe operator boundary,
not a paper performance result. Its Attention failure is superseded by the
boundary-buffer fix and the formal pair under
`/data/adafloor_shared_state/qwen3_vanilla_full16_eager_vs_attention_aclgraph_tq1_20260812T_formal`.
That pair compares one warmup and three measured batches of 16 greedy requests,
with 64 output tokens per request. All prompt and response tokens match exactly.
Attention graph execution reduces mean inner generation time from 9.398 s to
8.547 s, but graph lifecycle work increases complete benchmark rollout wall
from 34.190 s to 37.898 s. It is correctness evidence and an optimization
diagnostic, not a paper speedup result.

## Wrapper Contract

The wrapper supplies these defaults:

| Setting | Default | Meaning |
| --- | --- | --- |
| `ADAFLOOR_GRAPH_MODE` | `elastic_aclgraph` | Select the supported hybrid graph/eager lifecycle. |
| `ADAFLOOR_GRAPH_CAPTURE_PROFILE` | `balanced` | Select `memory_saver`, `balanced`, or `full_coverage`. |
| `ADAFLOOR_GRAPH_CAPTURE_SIZES` | unset | Optional explicit list. When set, it overrides the selected profile. |
| `ADAFLOOR_GRAPH_BASE_RUNNER` | `run_mode1_local_length_sorted_e2e_adaptive_floor4.sh` | Existing runner to execute. Relative paths resolve from this directory. |
| `VLLM_ASCEND_ELASTIC_ACLGRAPH` | `1` | Enable AdaFloor-aware ACLGraph lifecycle handling. |
| `VLLM_ASCEND_ELASTIC_ACLGRAPH_CAPTURE_ATTENTION` | `0` | Keep Attention outside ACLGraph by default. The opt-in fixed16 Attention path passes exact greedy parity, but combined MoE and elastic lifecycle parity remain pending. |
| `VLLM_ASCEND_ELASTIC_ACLGRAPH_CAPTURE_MOE` | `1` | Capture elastic MoE dispatch, expert compute, communication, and combine. Set `0` for the isolated-MoE-boundary diagnostic mode. |
| `VLLM_ASCEND_ELASTIC_ACLGRAPH_EXTENSION` | `/workspace/vllm-ascend/vllm_ascend/vllm_ascend_C.cpython-311-aarch64-linux-gnu.so` | PyTorch-ABI-compatible Ascend custom-op library. The wrapper checks it before Ray starts. |
| `VLLM_ASCEND_ELASTIC_ACLGRAPH_EXTENSION_SHA256` | computed by wrapper | Pin the exact extension bytes passed to every Ray worker. |
| `VLLM_ENABLE_GRAPH_MODE` | `0` | Disable the distinct TorchAir path. |
| `TASK_QUEUE_ENABLE` | `1` | Use the task-queue setting validated for ACLGraph capture. |
| `ROLLOUT_ENFORCE_EAGER` | `False` | Permit vLLM ACLGraph construction. |
| `VERL_SIDECAR_ENABLE` | `0` | Keep the unsupported sidecar path disabled. |
| `VERL_HCCL_IF_BASE_PORT_START` | `12000` | Start the fail-closed four-phase HCCL port allocator below its 16K window limit. |
| `VERL_MASTER_PORT_START` | `28416` | Keep Ray worker master ports outside that HCCL allocation window. |
| `actor_rollout_ref.actor.megatron.override_transformer_config.moe_shared_expert_overlap` | `False` | Keep the Qwen3 training-side post-rollout path valid when the model has no shared experts. This does not change rollout graph coverage. |

Capture profiles make the graph-coverage and HBM tradeoff explicit:

| Profile | Capture sizes | Intended use |
| --- | --- | --- |
| `memory_saver` | `[1,2,4,8]` | Verified lifecycle/debug profile. Larger local decode batches fall back to eager until they enter the covered range. |
| `balanced` | `[1,2,4,8,16,32]` | Verified default. Covers the common low and mid decode occupancy while preserving more HBM headroom than full coverage. |
| `full_coverage` | `[1,2,4,8,16,32,64]` | Covers a 64-request local decode batch, including the common 64-prompt-by-16-response full-world layout. Use only after checking HBM headroom on the target model. |

Capture memory is per rollout worker. The two-shape `[1,2]` lifecycle
smoke reported a maximum allocator delta of 1.64 GiB in Planned and 3.25 GiB in
Natural. The difference includes policy-specific expert residency and memory
pool state, so it is not a portable estimate of graph memory. These numbers are
validation observations, not throughput results. The runtime logs
`Graph capturing finished ... took ... GiB` for every worker, so the selected
profile must still be checked against the minimum HBM headroom of the intended
floor before a performance run.

Graph count has two distinct meanings. The compiled model is one FX partition
per worker. At runtime the smoke keeps two ACLGraph instances per active worker,
one for each of batch sizes 1 and 2. It never retains graph sets for full16,
floor8, and floor4 simultaneously because those sets bind different HCCL
communicators and stale weight/KV addresses. Production `n=16` workloads need
more batch descriptors, commonly powers of two through 64. Reducing that list
further trades graph coverage for eager fallback or padding work.

The capture-size override is passed to the base runner before user arguments,
so an explicit final Hydra argument can replace it when diagnosing one exact
shape. The environment variable is preferred for normal use:

```bash
ADAFLOOR_GRAPH_CAPTURE_PROFILE=memory_saver \
  ./run_mode1_local_length_sorted_e2e_adaptive_floor4_aclgraph.sh \
  trainer.total_training_steps=4
```

## Minimal NPU Smoke

Both checks require an exclusive 16-NPU node. Run them sequentially. Store all
outputs and caches outside this repository and outside formal paper artifacts:

```bash
TARGET=/workspace/cann-recipes-train/llm_rl/adafloor_gragh
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
RUN_ROOT=/data/adafloor_graph_smoke/$STAMP
RAY_SOCKET_ROOT=/tmp/ag_$$
mkdir -p \
  "$RUN_ROOT"/{tmp,fixed16_aclgraph,lifecycle_16_8_4_restore_16} \
  "$RUN_ROOT"/cache/{huggingface,triton,torchair,ascend,ascend_work} \
  "$RAY_SOCKET_ROOT"
export XDG_CACHE_HOME="$RUN_ROOT/cache"
export HF_HOME="$RUN_ROOT/cache/huggingface"
export TRITON_CACHE_DIR="$RUN_ROOT/cache/triton"
export TORCHAIR_CACHE_HOME="$RUN_ROOT/cache/torchair"
export ASCEND_CACHE_PATH="$RUN_ROOT/cache/ascend"
export ASCEND_WORK_PATH="$RUN_ROOT/cache/ascend_work"
export RAY_TMPDIR="$RAY_SOCKET_ROOT"
export TMPDIR="$RUN_ROOT/tmp"
cd "$TARGET"
```

First validate fixed full-world ACLGraph initialization and replay with one
rollout-only batch. Sixteen prompts with `n=1` produce one local request per
rollout worker, so only shape 1 is needed:

```bash
OUTPUT_ROOT="$RUN_ROOT/fixed16_aclgraph" \
OUTPUT_SUBDIR=run \
BATCH_SIZES=16 \
ROLLOUT_N=1 \
DECODE_TOKENS=16 \
MAX_PROMPT_LENGTH=512 \
ROLLOUT_MAX_NUM_BATCHED_TOKENS=1024 \
VERL_ROLLOUT_BENCH_WARMUP_STEPS=0 \
VERL_ROLLOUT_BENCH_MEASURE_STEPS=1 \
VLLM_ASCEND_ELASTIC_ACLGRAPH=1 \
VLLM_ASCEND_ELASTIC_ACLGRAPH_CAPTURE_ATTENTION=0 \
VLLM_ASCEND_ELASTIC_ACLGRAPH_CAPTURE_MOE=1 \
VLLM_ENABLE_GRAPH_MODE=0 \
TASK_QUEUE_ENABLE=1 \
ROLLOUT_ENFORCE_EAGER=False \
VERL_HCCL_IF_BASE_PORT_START=12000 \
VERL_MASTER_PORT_START=28416 \
ADAFLOOR_TRAIN_LAUNCHER_SNAPSHOT_ACTIVE=1 \
./run_rollout_decode_batchsize_benchmark.sh \
  'actor_rollout_ref.rollout.cudagraph_capture_sizes=[1]' \
  actor_rollout_ref.rollout.ignore_eos=True
```

Then validate the shortest complete lifecycle with two training steps. Step 1
uses a floor-4 plan and executes `16 -> 8 -> 4`. Its post-rollout restore
returns all workers to DP16/EP16. Step 2 selects floor 16 and proves that the
restored world can recapture and run a fresh graph. The complete model forward
remains graph-backed at each validated floor.
The compact common epoch0 history is read-only; generated plans and logs stay
under `RUN_ROOT`.

```bash
COMMON_HISTORY=/data/adafloor_shared_state/common_epoch0_probe_gpu09_kv380800_permanent/epoch_000_mode0_probe
REPO_ROOT="$RUN_ROOT/lifecycle_16_8_4_restore_16" \
PATCH_TREE="$TARGET" \
LOCAL_TEST_LAUNCHER="$TARGET/internal/wj_train_grpo_qwen30b_a3b_16die_true_weight_regroup.sh" \
OUTPUT_ROOT="$RUN_ROOT/lifecycle_16_8_4_restore_16" \
OUTPUT_SUBDIR=run \
PLAN_DIR="$RUN_ROOT/lifecycle_16_8_4_restore_16/oracle" \
BASELINE_DIRS="$COMMON_HISTORY" \
REQUIRE_COMPACT_HISTORY=1 \
PLAN_STEPS=2 \
FORCE_SELECTED_FLOORS=4,16 \
TRAIN_BATCH_SIZE=32 \
ROLLOUT_N=1 \
MAX_PROMPT_LENGTH=512 \
MAX_RESPONSE_LENGTH=128 \
MAX_RESPONSE_LEN=128 \
IGNORE_TAIL_TIES_AT_RESPONSE_CAP=1 \
ROLLOUT_MAX_NUM_BATCHED_TOKENS=640 \
ROLLOUT_MAX_NUM_SEQS=32 \
VERL_ELASTIC_TAIL_VALIDATE_LEVEL_TOKENS_BY_STEP='8,16,32,64,64;8,16,32,64,64' \
TAIL_GUARD_MIN_CAP=64 \
TAIL_GUARD_ROUND_TO=64 \
SAVE_CKPT_ENABLE=0 \
SAVE_DRAFT_HIDDEN_ENABLE=0 \
TRAINER_SAVE_FREQ=-1 \
ADAFLOOR_GRAPH_CAPTURE_SIZES='[1,2]' \
VLLM_ASCEND_ELASTIC_ACLGRAPH_CAPTURE_MOE=1 \
ADAFLOOR_FLOOR4_CHILD_SNAPSHOT_ACTIVE=1 \
ADAFLOOR_TRAIN_LAUNCHER_SNAPSHOT_ACTIVE=1 \
./run_mode1_local_length_sorted_e2e_adaptive_floor4_aclgraph.sh \
  trainer.total_training_steps=2 \
  trainer.rollout_data_dir="$RUN_ROOT/lifecycle_16_8_4_restore_16/rollout_data" \
  trainer.rollout_length_dir="$RUN_ROOT/lifecycle_16_8_4_restore_16/rollout_length" \
  'trainer.logger=["console"]' \
  actor_rollout_ref.rollout.ignore_eos=True \
  actor_rollout_ref.actor.use_kl_loss=False \
  algorithm.use_kl_in_reward=False

python3 tools/verify_adafloor_aclgraph_smoke.py \
  --root "$RUN_ROOT/lifecycle_16_8_4_restore_16"
```

`IGNORE_TAIL_TIES_AT_RESPONSE_CAP=1` is only a planning-feasibility setting
for this artificially short response-cap smoke. Do not use it for performance
experiments or paper measurements.

The fixed check normally takes about 5 to 10 minutes. The two-step lifecycle
check normally takes about 12 to 20 minutes with the short caps above. Allow a
30-minute timeout while graph recapture is still experimental.

Keep `RAY_TMPDIR` under a short `/tmp` path. Ray creates Unix-domain sockets
below this directory, and Linux rejects socket paths longer than 107 bytes.

## Acceptance

The fixed check passes only when all of the following are present:

- `PIECEWISE compilation enabled on NPU` and `using only ACL Graph mode`.
- `Starting to capture ACL graphs`, followed by `Graph capturing finished`.
- `Replaying aclgraph` during the measured rollout.
- `summary_batch_16.json` reports `batch_size=16`, `rollout_n=1`, one measured
  step, and 256 total decode tokens.

The lifecycle check additionally requires:

- `oracle/length_sorted_rank_plan_summary.json` contains the selected floors
  `[4, 16]` in that order. Step 1 stages are `[8, 4]` with survivor sets
  `[8..15]` and `[12..15]`.
- `Elastic ACLGraph MoE capture enabled` is emitted on all 16 workers and the
  Attention-capture marker is absent.
- The generated FX program contains 48 standalone Attention boundaries and 48
  graph regions containing the elastic-MoE call on every rank. Runtime replay
  is observed at full16, floor8, floor4, and restored full16.
- Each rollout weight refresh emits `Elastic ACLGraph invalidated ... after
  update_weights_after_finalize`, then `Elastic ACLGraph recapture starting
  ...` and `Elastic ACLGraph recapture finished ...` before live generation.
- The step-1 post-rollout transition completes full-world group, layout,
  weight, and KV restore before step 2.
- Step 2 recaptures a newly valid ACLGraph after full-world restore and then
  completes generation at floor 16.
- Two training-step records are emitted, both contain 32 responses and exactly
  640 generated tokens, and the process exits successfully.
- `tools/verify_adafloor_aclgraph_smoke.py` writes a `PASS` result to
  `ACLGRAPH_SMOKE_SUMMARY.json`.

Both checks fail on `Compilation disabled, using eager mode by default`,
`Skipping ACL graph capture`, `falling back to NONE`, `ACLgraph sizes capture
fail`, stale input-address assertions, Ray worker failures, HCCL errors, NPU
OOM, an unrecognized runtime traceback, or a timeout. The verifier separately
reports the known Torch Dynamo metrics-logging traceback and post-success TBE
cleanup noise. Keep the full logs even after a pass; a zero exit status alone
is not sufficient evidence of graph execution.

## What Remains Outside ACLGraph

The MoE and surrounding dense portions used by validated batch descriptors are
graph captured. Attention remains outside only because the old runtime failed
token parity, not because Attention is intrinsically ungraphable. The following
work remains intentionally outside because it mutates or selects the execution
topology rather than replaying a fixed tensor program:

- planner and Natural floor decisions, host request scheduling, quorum polling,
  and survivor selection;
- HCCL/MC2/DP/EP communicator creation and destruction, expert migration,
  ownership-map updates, and resident-slot preparation;
- actor weight load/offload/update, KV-cache allocation or resize, graph-pool
  rotation, cache invalidation, and full-world restore;
- tokenization, request admission, output collation, reward computation,
  checkpointing, and Megatron training.

These operations are not useful graph candidates because replaying their side
effects would reuse stale process groups, addresses, or control decisions.
Other omissions are implementation coverage rather than fundamental limits.
Batch descriptors outside the configured capture-size list fall back to eager
and can be covered by adding shapes at a memory cost. vLLM 0.11 keeps sampling
and some scheduler-side NPU work outside the compiled model callable, although
newer runtimes could capture more of it. Async rollout is rejected because its
elastic graph lifecycle is not wired yet. Sidecar inference remains eager and
separate because short leases may not amortize capture. Megatron training graph
support is a different project due to offload, recomputation, and dynamic batch
settings; rollout ACLGraph does not imply training graph execution.

## Performance Boundary

The graph results above are functional smokes, not a paper speedup measurement.
They include cold compilation and topology recapture in very short 640-token
steps. Planned transition critical paths were 4.213 s and 3.335 s, while
Natural used 13.576 s and 15.393 s. The low-floor graph recapture portion was
about one second; the additional Natural time came primarily from runtime
expert and communication-state preparation. A paper-ready graph performance
claim still requires warmed, matched eager/ACLGraph runs with representative
response lengths and counterbalanced order.
