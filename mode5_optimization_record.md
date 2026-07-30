# Mode=5 Hybrid CPU+Remote-NPU Double Buffer Optimization Record

This note records the mode=5 optimization attempts, failure modes, and the
current guardrails. Read this before starting a new mode=5 optimization so we
do not repeat the same bad directions.

## Goal

Mode=5 should implement a hybrid double-buffer path:

- Before first shrink: behave like the baseline EP path, no hybrid buffer.
- After shrink, active ranks keep rollout compute.
- Missing experts for the runtime buffer are split between:
  - local NPU resident experts
  - remote NPU cache experts
  - CPU-shadow experts
- The intended mode=5 direction is to combine CPU-NPU and remote-NPU bandwidth.
- For the current threshold-validation setup, the target is first to run
  `16 -> 8 -> 4 -> 2 -> 1` stably for 3 steps, then optimize speed.

## Test Command

Use threshold validation first:

```bash
TRAINER_TOTAL_EPOCHS=3 \
VERL_ELASTIC_TAIL_VALIDATE_LEVEL_TOKENS=256,512,640,768,896 \
VLLM_ASCEND_ELASTIC_EXECUTION_MODE=5 \
bash internal/wj_train_grpo_qwen30b_a3b_16die_true_weight_eager.sh
```

Do not switch to full validation until threshold mode is stable.

## Target Semantics

Mode=5 is supposed to be a true hybrid path, not a remote-only special case.

With the current default:

```bash
VLLM_ASCEND_MODE5_REMOTE_EXPERT_FRACTION=0.5
```

the intended layer-0 split after shrink is:

- `stage=8`: local `8`, remote-NPU `4`, CPU `4`
- `stage=4`: local `8`, remote-NPU `12`, CPU `12`
- `stage=2`: local `8`, remote-NPU `28`, CPU `28`
- `stage=1`: local `8`, remote-NPU `60`, CPU `60`

This is the semantics we should optimize toward and validate.

## Current Observed Semantics In The Latest Stabilized Run

For the currently stabilized-but-not-yet-correct mode=5 path seen in
`wjeagerqwen30b-a3b-with_draft_breakdown_20260602120429_elastic.txt`,
layer-0 activation followed this pattern instead:

- `stage=8`: `owned_local=16`, `primary_prefix_rows=8`,
  `remote_npu_local=8`, `cpu_only_local=0`
- `stage=4`: `owned_local=32`, `primary_prefix_rows=8`,
  `remote_npu_local=24`, `cpu_only_local=0`
- `stage=2`: `owned_local=64`, `primary_prefix_rows=8`,
  `remote_npu_local=56`, `cpu_only_local=0`
- `stage=1`: `owned_local=128`, `primary_prefix_rows=8`,
  `remote_npu_local=120`, `cpu_only_local=0`

This is not the desired mode=5 target. It is a remote-NPU-only special case
that happened because the current payload split forced all missing experts onto
the remote-NPU path in order to avoid the CPU-shadow export leak.

Rule:

- do not mistake this stabilized remote-only path for the correct mode=5 goal
- the next real mode=5 task is to restore the intended `50% remote + 50% CPU`
  split while keeping the run stable

## Known Bad Runs And Root Causes

### 1. Remote cache listener / stop-sentinel mismatch

Symptom:

- shrink to low floor appeared to succeed, then cache ranks hung in
  `cpu_group recv`
- final error looked like Gloo/GPU timeout, for example:

```text
Mode5 remote cache service failed: ... Timed out waiting ... recv
```

Representative log:

- `wjeagerqwen30b-a3b-with_draft_breakdown_20260602100844_elastic.txt`

Root cause:

- mode=5 reused the mode=4 remote-cache service startup too broadly
- cache listeners were started on ranks inferred from the full
  `mode4_remote_source_rank` view, including experts that were logically
  assigned to the CPU-shadow path
- stop sentinels were also broadcast too broadly
- result: some ranks entered the cache-service recv loop even though no active
  rank would ever send them requests

Fix direction:

- only start cache listeners for ranks that actually own
  `mode5_remote_experts`
- only send stop sentinels to those cache ranks

Rule:

- do not infer mode=5 cache listeners from the full inactive rank set
- do not infer them from all `cpu_import_source_rank` entries
- use only the real `mode5_remote_experts -> remote source rank` subset

### 2. CPU-shadow export leak for experts that never had CPU shadow

Symptom:

- run failed during preload/import with:

```text
RuntimeError: CPU export requested experts that are not present in the CPU shadow
```

Representative logs:

- `wjeagerqwen30b-a3b-with_draft_breakdown_20260602114952_elastic.txt`
- `wjeagerqwen30b-a3b-with_draft_breakdown_20260602115651_elastic.txt`

Typical missing experts:

- `missing=[64]`
- `missing=[72]`
- `missing=[80]`
- `missing=[88]`

Root cause:

- mode=5 had a payload split bug:
  - `mode5_remote_experts`
  - `mode5_cpu_import_experts`
  were derived from the logical import payload
- but some experts selected for the CPU path had no CPU shadow on the chosen
  `source_rank`; they only existed on the remote-NPU cache path
- later, `_export_lossless_expert_cpu_weight_batch()` correctly refused to
  export them because they were absent from `lossless_cpu_shadow_local_slots`

Stability fix we applied:

- treat "missing CPU shadow on source rank" as a hard rule:
  - such experts are temporarily forced into `mode5_remote_experts`
  - they are removed from `mode5_cpu_import_experts`

Important caveat:

- this was a stability fix, not the final desired mode=5 behavior
- it prevents the crash, but it can collapse mode=5 into a remote-only path
  if the chosen CPU-shadow source ranks are wrong

Rule:

- do not patch this later by adding a downstream fallback
- the correct place to fix it is still the payload split itself
- but the long-term fix is not "force everything remote"
- the long-term fix is: choose CPU-path source ranks that truly own CPU shadow
  for the experts assigned to the CPU half

### 3. Low-floor shrink suddenly became extremely slow

Symptom:

- shrink to stage=1 jumped to tens of seconds
- example:

```text
refresh_ms=44348.94 total_ms=57805.88
```

Representative log:

- `wjeagerqwen30b-a3b-with_draft_breakdown_20260602100844_elastic.txt`

Root cause:

- this was tied to the remote-cache listener / payload ownership confusion
- too many ranks were treated as cache/service participants
- shrink refresh and cache-service orchestration became much heavier than
  needed

Current healthy numbers:

From `wjeagerqwen30b-a3b-with_draft_breakdown_20260602120429_elastic.txt`:

- `16 -> 8`: `total_ms ~= 3.9s - 4.3s`
- `8 -> 4`: `total_ms ~= 3.0s - 3.1s`
- `4 -> 2`: `total_ms ~= 2.5s`
- `2 -> 1`: `total_ms ~= 2.66s`

Rule:

- if low-floor shrink jumps back to tens of seconds, first inspect:
  - cache listener ownership
  - stop sentinel target set
  - mode5 payload partition
- do not start by adding more warmup or more headroom

## Current Guardrails

### Cache service

- cache listener ranks must be derived from `mode5_remote_experts` only
- stop sentinels should target `self._mode4_owner_to_cache_ranks` if present
- do not fall back to "all inactive ranks" unless there is no precise mapping

### CPU shadow

- mode=5 CPU shadow should only contain experts that truly use the CPU path
- remote-NPU-backed experts should not be materialized into the CPU shadow
  just for convenience

### Payload partition

- `mode5_remote_experts` and `mode5_cpu_import_experts` must be mutually
  coherent
- if an expert lacks CPU shadow on its selected `source_rank`, it must be
  forced remote for stability
- but the optimization goal is to minimize such forced-remote cases by picking
  correct CPU-shadow source ranks for the CPU half

### Performance debugging order

If a future mode=5 run regresses, check in this order:

1. Did the run still follow the intended activation semantics at stage 8/4/2/1?
   - target mode=5 semantics are `local 8 + remote half + CPU half`
   - if `cpu_only_local=0` at all stages, the run may be stable but mode=5 is
     not actually exercising the CPU half
2. Did cache listeners only start on necessary cache ranks?
3. Did any expert leak into CPU export without a CPU shadow?
4. Did shrink totals jump back to multi-second or multi-tens-of-second outliers?
5. Only after the above, inspect bandwidth split or transfer overlap.

## Acceptance Checklist

Before calling a mode=5 change "good enough":

- `16 -> 8 -> 4 -> 2 -> 1` all appear in the log.
- No `Mode5 remote cache service failed`.
- No `CPU export requested experts that are not present in the CPU shadow`.
- No `RuntimeError` / `Traceback`.
- Shrink totals stay in the current healthy range, not tens of seconds.
- Restore completes cleanly.
- The expert split matches the target mode=5 ratio:
  - `stage=8`: local `8`, remote `4`, CPU `4`
  - `stage=4`: local `8`, remote `12`, CPU `12`
  - `stage=2`: local `8`, remote `28`, CPU `28`
  - `stage=1`: local `8`, remote `60`, CPU `60`
- Then verify `rollout_output_time_s` and multi-step continuity.

## Useful Logs

Bad / reference logs:

- `wjeagerqwen30b-a3b-with_draft_breakdown_20260602100844_elastic.txt`
  - remote-cache listener mismatch, huge shrink
- `wjeagerqwen30b-a3b-with_draft_breakdown_20260602114952_elastic.txt`
  - CPU-shadow export leak
- `wjeagerqwen30b-a3b-with_draft_breakdown_20260602115651_elastic.txt`
  - CPU-shadow export leak still present before payload split fix

Current stabilized-but-semantically-incomplete reference:

- `wjeagerqwen30b-a3b-with_draft_breakdown_20260602120429_elastic.txt`
  - `16 -> 8 -> 4 -> 2 -> 1` all observed
  - no CPU-shadow export leak
  - no remote-cache service failure
  - shrink timings back to normal
  - but CPU half collapsed to `0`, so this is not the final mode=5 target

## Files To Inspect First

- `vllm_ascend/worker/worker_v1.py`
  - `_prepare_lossless_shrink_payload`
  - `_mode5_select_remote_experts_by_target`
  - `_mode5_cpu_import_experts`
  - `_start_mode4_remote_cache_service`
  - `_send_mode4_remote_cache_stop_sentinels`
  - `_stream_lossless_hybrid_import_weights_p2p`
  - `_preload_lossless_shrink_import_weights`
- `vllm_ascend/ops/fused_moe.py`
  - CPU-shadow build path
  - `export_lossless_expert_cpu_weights`
  - mode5 runtime activation state

## Practical Rule For Future Mode=5 Work

When optimizing mode=5 again:

- first restore the correct hybrid split semantics
- then preserve correctness of ownership and path partition
- then preserve shrink/restore latency
- only then optimize the CPU-vs-remote-NPU traffic ratio

If a change breaks semantics, do not hide it with fallback. Fix the ownership
or payload split where the mistake begins.
