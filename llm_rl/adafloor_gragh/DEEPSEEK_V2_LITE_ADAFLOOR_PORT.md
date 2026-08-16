# DeepSeek-V2-Lite AdaFloor Port

This directory contains the DeepSeek-V2-Lite-Chat implementation of AdaFloor.
The model is pinned to revision
`85864749cd611b4353ce1decdb286193298f64c7`. The Hugging Face checkpoint and
the converted PP4 EP4 Megatron distributed checkpoint are expected at
`/data/DeepSeek-V2-Lite-Chat` and
`/data/DeepSeek-V2-Lite-Chat_megatron_pp4_ep4`.

Natural and Planned execution both support the complete EP16 shrink hierarchy
`16 -> 8 -> 4 -> 2`. A floor4 lifecycle permits candidates 16, 8, and 4. A
floor2 lifecycle additionally permits candidate 2. Natural and Planned use
separate runtime profiles, KV measurements, authorization flags, and formal
audits because their resident state and memory capacities are not
interchangeable.

This document distinguishes implemented support from runtime validation. A
feature listed as implemented is not a performance result. Formal results may
be used only after the exact lifecycle has passed calibration, strict
authorization, and the paired plan/runtime audit under its recorded source
hash.

## Model Contract

The port accepts the pinned DeepSeek-V2-Lite checkpoint rather than an
arbitrary DeepSeek-family model.

| Field | Required value |
| --- | --- |
| Architecture | `DeepseekV2ForCausalLM` |
| Precision | BF16 |
| Transformer layers | 27 |
| Hidden size | 2048 |
| Attention heads | 16 |
| Routed experts | 64 |
| Shared experts | 2 |
| Active routed experts per token | 6 |
| First dense layers | 1 |
| MoE intermediate size | 1408 |
| MLA KV LoRA rank | 512 |

AdaFloor redistributes only the 64 routed experts represented by
`FusedMoE.w13_weight` and `FusedMoE.w2_weight`. The two shared experts remain
dense shared MLP parameters and are excluded from `log2phy` and redundant
expert-slot accounting. With rollout EP16, an active rank requires 4, 8, 16,
or 32 routed-expert slots at floors 16, 8, 4, or 2. These values are derived
from the model configuration. The elastic buffer path does not use a
128-expert fallback.

Training follows the Qwen3 checkpoint flow. The local
`prepare_deepseek_v2_lite_assets.sh` launcher converts the pinned Hugging Face
weights to a PP4 EP4 Megatron distributed checkpoint. Actor and reference
training load that distributed checkpoint. `MODEL_PATH` remains the tokenizer,
generation-config, and vLLM construction source. Every DeepSeek launcher
validates the conversion manifest and pins the actor and reference distcp
options after user Hydra arguments are processed.

## Parallel and Workload Contract

The port uses one 16-NPU node.

| Phase | Configuration |
| --- | --- |
| Megatron training | TP1, PP4, EP4, ETP1 |
| Pipeline layers | 6, 7, 7, 7 |
| vLLM rollout | DP16, TP1, EP16 |
| Natural candidates | 16, 8, 4, 2 |
| Planned candidates | 16, 8, 4, 2 |

The paper workload uses 64 prompts per step and 16 responses per prompt. Each
step therefore contains 1024 responses. The prompt and response limits are
1024 and 16384 tokens. vLLM uses a 17408-token batching limit, 64 sequences, a
128-token KV block, 0.9 memory utilization, and eager execution. Sampling uses
temperature 0.9, top-p 0.9, and top-k 50.

The Natural comparison resumes a frozen actor after five common epoch0 actor
updates and executes one five-step epoch. A second replay assigns every stable
prompt-occurrence and sample identity the same generated length in LengthSort
Full16 and AdaFloor Natural floor2. It therefore removes generated-work
differences from the runtime comparison.

## Lifecycle Contracts

The four lifecycle profiles are independent authorization domains.

| Lifecycle | Policy | Allowed floors | Profile | Verification flag |
| --- | --- | --- | --- | --- |
| `natural_f4` | Completion-driven survivors | 16, 8, 4 | `internal/deepseek_v2_lite_natural_f4_runtime_profile.sh` | `DEEPSEEK_N_F4_KV_CAPS_VERIFIED` |
| `natural_f2` | Completion-driven survivors | 16, 8, 4, 2 | `internal/deepseek_v2_lite_natural_f2_runtime_profile.sh` | `DEEPSEEK_N_F2_KV_CAPS_VERIFIED` |
| `planned_f4` | Predetermined survivor residency | 16, 8, 4 | `internal/deepseek_v2_lite_planned_f4_runtime_profile.sh` | `DEEPSEEK_P_F4_KV_CAPS_VERIFIED` |
| `planned_f2` | Predetermined survivor residency | 16, 8, 4, 2 | `internal/deepseek_v2_lite_planned_f2_runtime_profile.sh` | `DEEPSEEK_P_F2_KV_CAPS_VERIFIED` |

Natural selects survivor ranks when completions reveal the live quorum.
Planned fixes the survivor topology before generation and retains the expert,
group, dispatcher, and communication state required by that topology. Planned
therefore has lifecycle-specific KV headroom and a measured training-memory
reserve. Neither its capacities nor its authorization may be inherited from
Natural.

Profile provenance is a SHA256 over the ordered profile-source closure. This
binds a derived floor2 or Planned profile to every profile that it sources.
Execution provenance separately hashes the DeepSeek AdaFloor code path.

## Implemented AdaFloor Functions

The DeepSeek port contains the complete AdaFloor control path for EP16.

1. The planner length-sorts prompts, enforces per-lifecycle KV admission caps,
   evaluates every allowed floor, and applies release-area rank matching.
   Calibration can force one floor to obtain independent evidence. Formal runs
   leave floor selection adaptive.
2. TailGuard derives guarded response caps from historical prompt-level maximum
   response lengths. Strict authorization requires TailGuard to be enabled and
   requires its cap to remain below the 16384-token response limit.
3. The first predicted epoch uses one historical epoch with
   `single_epoch_prompt_max`. Later epochs use prompt-level EMA history. The
   formal auditor rejects inconsistent history provenance and requires an EMA
   planning epoch in a multi-epoch run.
4. Natural and Planned can execute each legal staged transition. A floor2 plan
   performs `16 -> 8`, `8 -> 4`, and `4 -> 2`. The runtime validates nested
   survivor sets and expert mappings at every stage.
5. Each rollout step restores the complete 16-rank world before the actor
   update. The auditor requires one coherent restore wave on all 16 ranks after
   every shrinking step.
6. Actor updates use the PP4 EP4 Megatron checkpoint. Updated routed tensors are
   expanded into DeepSeek vLLM expert names before the next rollout.
7. Rank-time can execute an inference sidecar after a verified shrink. The
   dedicated launcher supports all four lifecycle profiles. It verifies that
   the sidecar starts on a detached rank after shrink, produces real output,
   and terminates before full-world restore. Planned sidecar runs additionally
   enforce their authorized headroom and strict training-memory reserve.

The formal paths keep `ALLOW_INFEASIBLE_PLAN` unset, require a native KV target
ratio of 1.0, and fail if a requested KV target cannot be allocated.

## Calibration and Authorization

All lifecycle capacities are written to the canonical
`deepseek_v2_lite_kv_caps.env`. Calibration updates only its own lifecycle
section through `--merge-existing`. A source-hash change invalidates previously
verified lifecycle flags rather than silently reusing stale capacities.

| Lifecycle | Calibration | Strict authorization |
| --- | --- | --- |
| `natural_f4` | `run_deepseek_v2_lite_natural_f4_calibration.sh` | `run_deepseek_v2_lite_kv_cap_validation.sh natural_f4` |
| `natural_f2` | `run_deepseek_v2_lite_natural_f2_calibration.sh` | `run_deepseek_v2_lite_natural_f2_kv_cap_validation.sh` |
| `planned_f4` | `run_deepseek_v2_lite_planned_f4_calibration.sh` | `run_deepseek_v2_lite_planned_f4_kv_cap_validation.sh` |
| `planned_f2` | `run_deepseek_v2_lite_planned_f2_calibration.sh` | `run_deepseek_v2_lite_planned_f2_kv_cap_validation.sh` |

Each calibration runs a plan-only preflight and an isolated physical-capacity
probe for every floor in that lifecycle. It records the common epoch0,
positive-release trigger history, planner artifact, profile closure, execution
hash, physical KV capacity, and admission reserve. Planned calibration also
requires measured per-floor resident-state headroom and a positive
training-memory reserve.

Strict authorization runs one independent 1024-response job for every
candidate floor. Each job resumes the same common epoch0 actor checkpoint and
must match its forced plan. The verifier binds physical KV resize evidence to
the matching calibration lifecycle, TailGuard, prediction history, rollout
artifacts, and the absence of preemption, abort, and OOM. Only that verifier
can promote the lifecycle-specific flag to 1.

After authorization, the paper comparison runs one frozen five-step epoch.
`tools/audit_deepseek_n_f4_formal_run.py` checks the plan against runtime resize,
safe-prefix shrink, full-world restore, health, TailGuard, and prediction
evidence. The implementation also supports later EMA-planned epochs, but the
current DeepSeek paper result does not claim that additional validation.

## Current Validation Status

The status below reflects artifacts present on 2026-08-09. A result is complete
only when its explicit verifier and cleanup transaction pass.

| Path | Current evidence | Status boundary |
| --- | --- | --- |
| Chat checkpoint selection | The semantic gate produces normal mathematical responses. The Base checkpoint was rejected because chat-formatted prompts caused pathological long continuations. | Chat is the only checkpoint used for performance or portability claims. |
| HF-to-MCore conversion and weight loading | `chat_weight_compare_smoke_20260809T_formal` matches 324 sampled tensor groups on every rank and covers all 64 routed experts in the layer-1 routed streams. It also completes generation and one actor update. | This verifies the sampled conversion and online-sync path. It is not a full post-update tensor equivalence proof. |
| Common epoch0 | The batch-64 common epoch0 completes five actor updates and preserves its hashed `global_step_5` checkpoint. | This frozen checkpoint is the sole source for calibration and paired runs. |
| Natural floor2 KV authorization | `KV_CAP_AUTHORIZATION_SUMMARY.json` reports `PASS` for floors 16, 8, 4, and 2 with 1024 responses per floor. | Natural floor2 is the only DeepSeek lifecycle authorized for the paper result. |
| Natural five-step pair | The official summary reports 5120 matched responses per arm. AdaFloor releases 15093.756 rank-s and both arms complete without preemption, OOM, or abort. | This is a complete-system comparison. TailGuard reduces generated work by 5.18 percent. |
| Equal-work five-step replay | LengthSort Full16 and AdaFloor each execute exactly 2509950 generated tokens. AdaFloor releases 15627.780 rank-s with a 5.52 percent rollout-time cost. The audit passes and cleanup is committed. | This removes generated-work differences. The control is LengthSort Full16, not Vanilla. |
| Natural response quality | Strict final-box accuracy is 12.66 percent for Vanilla and 12.56 percent for AdaFloor across 5120 matched responses. The paired difference has a prompt-cluster 95 percent interval from -0.72 to +0.53 percentage points. | This supports response-level comparability for one frozen epoch, not prompt-level noninferiority or training convergence. |
| Token correctness | All 3072 responses in the three steps that execute shrink match the Full16 response token streams exactly. | This is a paired diagnostic for the observed run, not a universal bitwise-determinism claim. |
| Planned residency, DeepSeek sidecar, and multi-epoch EMA | The code paths and fail-closed checks are implemented. | No current DeepSeek runtime claim is made for these optional variants. |

The authorized Natural capacities are 614144 physical and 585600 admission
tokens at floor16. Floors 8, 4, and 2 use 563840 physical and 535296 admission
tokens per rank. The floor16 admission cap holds 33 complete 17408-token
sequences. The batch-64 workload therefore exercises the measured KV admission
logic without assuming that Vanilla must preempt. Neither formal baseline
preempts in the observed run.

The fail-closed evidence ledger is
`../qwen3_shrink_aware/analysis_eval/deepseek_v2_lite_chat_evidence.json`.
It binds the weight, semantic, common-epoch, KV authorization, Natural,
equal-work, quality, token-correctness, cleanup, and release-unit artifacts by
SHA256. The corresponding readable summary is
`../qwen3_shrink_aware/analysis_eval/deepseek_v2_lite_chat_evidence.md`.

## Remaining Optional Validation

The required second-model evidence is complete. Additional NPU work is needed
only if the paper expands its DeepSeek claims.

1. Reverse the equal-work launch order to measure order sensitivity in the
   observed primary-throughput cost.
2. Add a near-cap Natural floor2 stress before claiming near-cap safety.
3. Add independent seeds to make a DeepSeek confidence-interval claim.
4. Compare TailGuard on and off to isolate DeepSeek quality effects.
5. Run a later updated-policy epoch to validate the EMA path on this model.
6. Calibrate Planned residency before making a DeepSeek Planned or sidecar
   claim.

These optional experiments must use new output roots and preserve the verified
Natural evidence. They do not replace the three-seed Qwen training trajectories
that support the paper's main performance and quality claims.

## Runtime Acceptance Checks

Before using a DeepSeek result, verify all of the following.

1. The model registry reports `DeepseekV2ForCausalLM` and 64 routed experts.
2. Initial local routed-expert count is 4 at EP16.
3. Active ranks contain 8, 16, and 32 routed slots at floors 8, 4, and 2.
4. The cap file contains the matching lifecycle profile ID, profile-closure
   hash, execution hash, common epoch0, trigger-history hashes, and verified
   flag.
5. Every selected floor is KV feasible and every runtime resize equals the
   authorized physical cap on all 16 ranks.
6. Every required shrink stage uses the nested survivor sets in the plan.
7. Every shrinking step completes a 16-rank restore before actor update.
8. Every actor update reports nonzero routed weights for all expected experts
   and no weight-loader or expert-map error.
9. TailGuard and prediction-mode fields match the recorded history count.
10. Logs and rollout artifacts contain no NaN, unexpected truncation,
    preemption, aborted response, OOM, timeout, or incomplete step.
11. A sidecar result additionally proves useful output after shrink and clean
    termination before full-world restore.

Static tests establish configuration, provenance, topology arithmetic, and
fail-closed validation behavior. They cannot establish physical KV capacity,
Planned resident-state headroom, HCCL stability, full-workload performance, or
sidecar throughput. Those remain lifecycle-specific runtime gates.
