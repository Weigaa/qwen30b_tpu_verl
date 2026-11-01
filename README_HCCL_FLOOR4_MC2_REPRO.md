# HCCL floor=4 MC2 communicator destroy latency reproduction

This branch preserves the minimal code and scripts used to reproduce a repeated
4-rank MC2/HCCL communicator lifecycle issue observed in mode=1 elastic rollout.

## Symptom

When `VLLM_ASCEND_ELASTIC_EXECUTION_MODE=1` and the minimum active compute group
is `floor=4`, repeated shrink/restore cycles that create and retire 4-rank MC2
communicators can produce very long restore/rebuild latency. The slow phase is
usually reported as `_MC2` stash/quarantine/rebuild time, for example:

```text
Elastic stash slow timing: rank=10 attr=_MC2 group_kind=mc2 group_ranks=(10, 11, 12, 13) ... quarantine_ms=127885.23
Elastic parallel restore done: rank=10 ... rebuild_mc2_ms=127886.76 ...
rollout_output_time_s: 229.519047
```

The explicit `destroy_process_group(device_pg)` path may appear fast, while the
old 4-rank communicator state is still able to cause a later 80s-120s wait or
higher non-torch memory pressure. This is why the issue is not visible as a
simple Python exception at the destroy call site.

## Main reproduction script

Run from this directory on a 16-NPU Atlas A3 environment with local model and
dataset paths already configured in the training script:

```bash
cd /workspace/cann-recipes-train/llm_rl/qwen3_true_mode5_a3cfdc2
bash run_mode1_floor4_overlap4group_quiesce_destroy_test.sh
```

The default pattern is intentionally partial-overlap floor=4 groups:

```text
step 1 floor4 group: usually [12, 13, 14, 15]
step 2 floor4 group: usually [10, 11, 12, 13]
overlap: [12, 13]
```

This makes the HCCL/MC2 residual-state behavior easier to reproduce than using
disjoint 4-rank groups.

Useful variants:

```bash
# Baseline floor=4 threshold multistep run.
bash run_mode1_floor4_threshold_multistep_test.sh

# Rotate floor4 groups between disjoint or partially-overlapping groups.
bash run_mode1_floor4_rotate4group_destroy_test.sh

# Compare with floor=8. This usually avoids the pathological floor=4 behavior.
bash run_mode1_floor8_threshold_multistep_test.sh
```

## Important environment variables

The scripts set most required values. These are the most useful knobs when
narrowing the issue:

```bash
# Core mode/floor.
export VLLM_ASCEND_ELASTIC_EXECUTION_MODE=1
export VLLM_ASCEND_ELASTIC_MIN_COMPUTE_GROUP_SIZE=4

# Force threshold-controlled short multistep reproduction.
export BASELINE_TOTAL_TRAINING_STEPS=3
export DATASET_FRACTION=0.005
export VERL_ELASTIC_TAIL_VALIDATE_LEVEL_TOKENS=256,512,640,768,896

# Rotate floor4 survivor groups across rollout calls.
export VERL_ELASTIC_TAIL_VALIDATE_ROTATE_BUCKETS=1
export VERL_ELASTIC_TAIL_VALIDATE_ROTATE_MODES=tail,shift2

# Keep floor4 MC2 cache disabled to exercise create/retire behavior.
export VLLM_ASCEND_MODE1_PARITY_KEEP_FLOOR4_GROUP_CACHE=0
export VLLM_ASCEND_MODE1_PARITY_KEEP_FLOOR4_GROUP_KINDS=mc2

# Diagnostic quiesce-before-destroy path. This did not fully eliminate the issue,
# but it makes the intended subgroup drain explicit in the logs.
export VLLM_ASCEND_MODE1_PARITY_FLOOR4_MC2_QUIESCE_BEFORE_DESTROY=1
export VLLM_ASCEND_MODE1_PARITY_FLOOR4_MC2_QUIESCE_OP=all_reduce
```

## Analyze a run

After a run, analyze the generated elastic log:

```bash
python analyze_floor4_mc2_log.py wjeagerqwen30b-a3b-with_draft_breakdown_YYYYMMDDHHMMSS_elastic.txt --top 30
```

Or analyze the latest matching log in this directory:

```bash
bash analyze_latest_floor4_mc2_log.sh
```

The most important fields are:

```text
rollout_output_time_s
floor4_mc2_slow_stash
floor4_mc2_quarantine_exit
top_stash_total_ms
top_quarantine_total_ms
Elastic parallel restore done ... rebuild_mc2_ms=...
```

## Known observations from this branch

1. `floor=8` is much less prone to the pathological wait.
2. `floor=4` with disjoint 4-rank groups can run better than overlapping groups.
3. Partially-overlapping or repeated floor=4 MC2 groups often trigger 80s-120s waits.
4. Explicit device PG destroy can be fast, but later wrapper/state cleanup or the next
   restore/rebuild path can still block.
5. Retaining destroyed PG references can hide the long wait but increases non-torch
   memory pressure and can lead to OOM after several rollout steps.
6. Aggressively dropping/trimming destroyed PG references can reintroduce the long wait
   or expose CANN/HCCL residual-state behavior in a later restore.

## Files added for reproduction

```text
run_mode1_floor4_overlap4group_quiesce_destroy_test.sh
run_mode1_floor4_overlap4group_destroy_test.sh
run_mode1_floor4_rotate4group_destroy_test.sh
run_mode1_floor4_threshold_multistep_test.sh
run_mode1_floor8_threshold_multistep_test.sh
analyze_floor4_mc2_log.py
analyze_latest_floor4_mc2_log.sh
```

The core instrumentation and lifecycle logic is in:

```text
vllm_ascend/worker/worker_v1.py
verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py
internal/wj_train_grpo_qwen30b_a3b_16die_true_weight_eager.sh
```

## Experimental workaround: floor4 MC2 live pool

This branch also contains an opt-in workaround that avoids hot-path floor4 MC2
communicator teardown. Enable it with:

```bash
bash run_mode1_floor4_livepool_overlap4group_test.sh
```

Then analyze the newest live-pool run:

```bash
bash analyze_latest_floor4_livepool_log.sh --top 30
```

or pass a log explicitly:

```bash
bash analyze_latest_floor4_livepool_log.sh /path/to/log.txt --top 30
```

or manually:

```bash
export VLLM_ASCEND_MODE1_PARITY_FLOOR4_MC2_LIVE_POOL=1
export VLLM_ASCEND_MODE1_PARITY_KEEP_FLOOR4_GROUP_CACHE=1
export VLLM_ASCEND_MODE1_PARITY_KEEP_FLOOR4_GROUP_KINDS=mc2
export VLLM_ASCEND_MODE1_PARITY_FLOOR4_MC2_LIVE_POOL_RANK_SETS='12-15;10-13'
export VLLM_ASCEND_MODE1_PARITY_FLOOR4_MC2_LIVE_POOL_MAX_GROUPS=8
export VLLM_ASCEND_MODE1_PARITY_DEFER_FLOOR4_MC2_DEVICE_PG_DESTROY=0
export VLLM_ASCEND_MODE1_PARITY_FLOOR4_MC2_KEEP_DESTROYED_PG_REF=0
export VLLM_ASCEND_MODE1_PARITY_FLOOR4_MC2_QUIESCE_BEFORE_DESTROY=0
```

The live-pool test uses explicit validation buckets by default:

```bash
export VERL_ELASTIC_TAIL_VALIDATE_ROTATE_MODES=floor4:12-15,floor4:10-13
```

`floor4:a-b` is a validation-only selector that maps exactly those four real
ranks into the final floor=4 tail bucket. This is more precise than `shiftN`
when checking whether overlapping 4-rank MC2 groups reuse stale HCCL state.

`VLLM_ASCEND_MODE1_PARITY_FLOOR4_MC2_LIVE_POOL_RANK_SETS` bounds the live pool
to known 4-rank candidates. Leave it empty only for exploratory diagnostics;
otherwise a run with many distinct floor4 combinations can trade destroy
latency for resident non-torch/HCCL memory growth.

`VLLM_ASCEND_MODE1_PARITY_FLOOR4_MC2_LIVE_POOL_MAX_GROUPS` is a guardrail: if a
new floor4 MC2 candidate would exceed the live-pool budget, the worker raises a
clear error before creating another HCCL group instead of silently falling back
to the slow destroy/recreate path.

Use the helper below to generate bounded candidate pools:

```bash
python generate_floor4_livepool_rank_sets.py --mode sliding --step 2
python generate_floor4_livepool_rank_sets.py --mode blocks
python generate_floor4_livepool_rank_sets.py --mode custom --custom '12-15;10-13'
```

`--mode all` enumerates every 4-rank subset of the world. On 16 ranks that is
1820 candidates, so it is useful for reasoning or tiny standalone repros, but
it is not a safe default for the full training job because the live-pool
tradeoff is resident HCCL/non-torch memory.

Expected evidence if the workaround is effective:

```text
floor4_mc2_live_pool_cache_store > 0
floor4_mc2_live_pool_cache_hit > 0
floor4_mc2_live_pool_placeholder_store > 0 when detached ranks consume floor4 MC2 placeholders
live_pool_max_combined_pool_size stays bounded for the test pattern
floor4_mc2_live_pool_over_budget_events == 0 for the default short test
floor4_mc2_live_pool_rank_set_not_allowed == 0
floor4_mc2_live_pool_capacity_exceeded == 0
floor4_mc2_binding_mismatch == 0
floor4_mc2_slow_stash is 0 or much lower than the non-live-pool run
live_pool_missed_slow_stash_ranks is empty
live_pool_retired_after_seen_ranks is empty
top_stash_total_ms no longer contains 80s-120s floor4 MC2 quarantine entries
```

`floor4_mc2_binding_mismatch` comes from the short
`Elastic floor4 MC2 binding state` log. A zero value means the first MoE layer's
`moe_config.mc2_group` was not observably pointing at a different 4-rank group
after refresh. If that value is zero but HCCL still reports long floor4 destroy
or restore waits, the evidence points away from a Python stale-wrapper bug and
toward residual communicator state inside the HCCL/CANN runtime.

If `floor4_mc2_live_pool_retire_skip` or
`floor4_mc2_live_pool_quarantine_skip` is nonzero, the live-pool guard caught a
floor4 MC2 group that would otherwise have entered the generic release path.
That is useful evidence when debugging stray destroy paths; the main success
criteria are still bounded pool size and disappearance of slow floor4 MC2 stash
or quarantine timings.

Tradeoff: floor4 MC2 ProcessGroup objects remain resident for the process
lifetime, so non-torch memory should be monitored. The live-pool script logs
`Elastic floor4 MC2 live pool` with real and placeholder pool sizes; by default
it warns once a worker sees 8 resident 4-rank MC2 groups across both pools. If
many different 4-rank combinations are used, reduce
`VLLM_ASCEND_MODE1_PARITY_MAX_KV_TOKENS`, increase the cooldown between
overlapping groups, or avoid unbounded rank-set rotation.

The live-pool test intentionally disables the older destroyed-PG-ref retention,
async trim, and quiesce-before-destroy experiments. Mixing those paths makes it
unclear whether a run avoided the bad HCCL teardown path or merely postponed it.

`floor4_mc2_live_pool_placeholder_signature_seen_no_pool` is a diagnostic, not
a hard failure. The non-member path uses `seen_signatures` to mean "the
new_group sequence has already been consumed"; it does not always imply that a
placeholder ProcessGroup object must still be resident. Treat it as suspicious
only when it correlates with nonzero `floor4_mc2_slow_stash`,
`live_pool_missed_slow_stash_ranks`, or `live_pool_retired_after_seen_ranks`.

### Rank-set pattern isolation test

Use `run_mode1_floor4_rankset_pattern_test.sh` to change only the explicit
floor=4 survivor sequence while keeping the same live-pool and logging setup:

```bash
FLOOR4_PATTERN=disjoint bash run_mode1_floor4_rankset_pattern_test.sh
FLOOR4_PATTERN=overlap  bash run_mode1_floor4_rankset_pattern_test.sh
FLOOR4_PATTERN=repeat   bash run_mode1_floor4_rankset_pattern_test.sh
```

Then run:

```bash
bash analyze_latest_floor4_livepool_log.sh --top 30
```

The analyzer prints `floor4_rank_sequence` and
`floor4_rank_transition_summary=repeat=...,overlap=...,disjoint=...`. Compare
slow floor4 MC2 stash/rebuild timings across these patterns:

- `disjoint` slow but `overlap`/`repeat` slow: residual HCCL state is likely tied
  to rank overlap or reuse of previous 4-rank communicators.
- all patterns slow: any floor4 MC2 communicator lifecycle may be affected.
- only `repeat` slow: exact-rank reuse is the strongest trigger.

### Strict rank-set protection

When the floor4 MC2 live-pool is enabled, unknown 4-rank MC2 rank sets now fail
fast by default:

```bash
export VLLM_ASCEND_MODE1_PARITY_FLOOR4_MC2_LIVE_POOL_STRICT_RANK_SETS=1
```

This is intentional. If a 4-rank MC2 tuple is not in
`VLLM_ASCEND_MODE1_PARITY_FLOOR4_MC2_LIVE_POOL_RANK_SETS`, silently creating it
outside the live pool can fall back to the original HCCL destroy/recreate path
and hide the real failure behind another 70s-120s wait. For exploratory runs,
set the rank-set pool explicitly, or temporarily set strict mode to `0` only
when you want to observe the fallback behavior.

For bounded "arbitrary 4-rank" diagnostics, generate a candidate pool and pass
it into the live-pool test:

```bash
python generate_floor4_livepool_rank_sets.py --mode all --max-groups 8
FLOOR4_PATTERN=all FLOOR4_MAX_GROUPS=8 bash run_mode1_floor4_rankset_pattern_test.sh
```

Do not use the full 16C4 pool in a normal training process unless memory headroom
has been measured; keeping all 1820 floor4 MC2 ProcessGroups resident would trade
HCCL teardown latency for non-torch memory pressure.

### Rank-set-specific floor4 MC2 group names

The live-pool test enables a second opt-in mitigation:

```bash
export VLLM_ASCEND_MODE1_PARITY_FLOOR4_MC2_UNIQUE_GROUP_NAME=1
```

With this enabled, a floor4 MC2 group created for ranks `(10, 11, 12, 13)` uses
an effective name like `mc2_r10_11_12_13` instead of reusing the bare `mc2` name
for every floor4 rank set. This is meant to test whether the long HCCL destroy
or restore waits are tied to communicator state being internally keyed by a
reused group name/sequence. It is opt-in and only affects mode=1 low-floor
floor4 MC2 groups.

The analyzer prints `floor4_group_names_by_ranks`; each rank tuple should map to
a rank-set-specific name when this mitigation is active. If unique group names
remove the long waits, the likely culprit is stale HCCL/CANN communicator state
associated with group-name or sequence reuse. If waits remain with
`floor4_mc2_binding_mismatch=0`, the issue is deeper than Python wrapper reuse
and should be reported as an HCCL communicator lifecycle problem.

Note: in this codebase, `group_name` is used by vLLM's `GroupCoordinator`
`unique_name` and custom-op lookup/device-communicator wrapper. PyTorch
`torch.distributed.new_group()` itself is still called without a name. Therefore
`VLLM_ASCEND_MODE1_PARITY_FLOOR4_MC2_UNIQUE_GROUP_NAME=1` is a useful experiment
for wrapper/custom-op identity, but it may not change the internal HCCL
ProcessGroup name such as `group_name_327` shown in CANN logs.

The analyzer also prints `floor4_device_pg_ids_by_ranks`. If cache hits reuse the
same `device_pg_id` for a rank tuple, Python is not recreating the device
ProcessGroup for that tuple. If HCCL still reports long waits with stable
`device_pg_id` reuse and zero `floor4_mc2_binding_mismatch`, the remaining state
is below the Python wrapper layer.

### Optional fallback for unpooled floor4 rank sets

If the live pool cannot cover every possible 4-rank tuple, an unpooled floor4
rank set should not silently use MC2 and re-enter the bad communicator lifecycle.
For diagnostics or safer exploratory runs, enable:

```bash
export VLLM_ASCEND_MODE1_PARITY_FLOOR4_MC2_FALLBACK_ON_UNPOOLED=1
```

When EP size is 4 and the current EP rank tuple is not listed in
`VLLM_ASCEND_MODE1_PARITY_FLOOR4_MC2_LIVE_POOL_RANK_SETS`, model-runner MC2
selection is disabled for that batch. The existing platform logic then falls
back to ALLTOALL or ALLGATHER depending on SoC/model settings. The forward
context also keeps `fused_moe_state` aligned with the selected non-MC2 method so
it does not re-enable MC2 later in the forward path.

The analyzer reports `floor4_mc2_fallback_on_unpooled` and
`fallback_floor4_ranks`. A nonzero value means the run avoided MC2 for at least
one floor4 tuple because it was outside the safe live-pool candidate set.

Caveat: the fallback decision currently happens in model-runner communication
selection. If the worker has already rebuilt an unpooled floor4 MC2 ProcessGroup
before forward selection, the analyzer reports
`floor4_mc2_unpooled_rebuild_before_fallback` and `unpooled_rebuild_floor4_ranks`.
A nonzero value means the fallback avoided MC2 dispatch, but it was too late to
avoid MC2 ProcessGroup creation. The next fix should then move the unpooled
fallback earlier into the elastic group rebuild/refresh path, or provide a
refresh mode that does not require `get_mc2_group()` for non-MC2 forwards.

## 2026-06-30: worker-level unpooled floor4 MC2 skip

A later diagnostic showed that model-runner fallback can be too late: the worker
may already rebuild `ascend_ps._MC2` for an unpooled floor=4 rank set before the
MoE communication method is selected.  In that case the run still creates the
problematic 4-rank HCCL MC2 communicator even if the forward later falls back to
alltoall/allgather.

The current mitigation adds a worker-level guard:

```bash
export VLLM_ASCEND_MODE1_PARITY_FLOOR4_MC2_FALLBACK_ON_UNPOOLED=1
export VLLM_ASCEND_MODE1_PARITY_FLOOR4_MC2_LIVE_POOL=1
export VLLM_ASCEND_MODE1_PARITY_FLOOR4_MC2_LIVE_POOL_RANK_SETS='12-15;10-13'
```

For floor=4 MC2 rank sets that are not in the live-pool whitelist, the worker now
skips `_MC2` rebuild before per-layer refresh.  If the currently bound `_MC2` is
itself an old 4-rank group, it is detached from `ascend_ps._MC2` and kept in a
private quarantine list to avoid Python GC destroying it on the hot path.  If the
currently bound `_MC2` is a full-world or 8-rank group, it is left in place; the
non-MC2 forward path must not consult it.

Expected analyzer signals:

```text
floor4_mc2_rebuild_skipped_for_unpooled > 0
floor4_mc2_unpooled_rebuild_before_fallback == 0
floor4_mc2_binding_mismatch == 0
```

If `floor4_mc2_unpooled_rebuild_before_fallback` is still nonzero, MC2 creation
is still happening before fallback and the slow HCCL destroy/recreate path has
not been fully avoided.

Recommended validation for arbitrary floor4 rank sets:

```bash
FLOOR4_PATTERN=all FLOOR4_MAX_GROUPS=6 \
FLOOR4_LIVE_POOL_RANK_SETS='12-15;10-13' \
VLLM_ASCEND_MODE1_PARITY_FLOOR4_MC2_FALLBACK_ON_UNPOOLED=1 \
bash run_mode1_floor4_rankset_pattern_test.sh
```

This rotates through six explicit floor=4 rank sets but only permits two of them
in the MC2 live pool.  The remaining four-rank groups should skip worker-level
`_MC2` rebuild and use non-MC2 MoE communication.  This is the closest current
repo-level test for the target behavior: arbitrary 4-rank groups across steps
should not access stale MC2 groups and should not enter the HCCL 4-rank MC2
destroy/recreate slow path.

To test the strongest fallback mode, disable the floor4 MC2 live pool entirely:

```bash
FLOOR4_PATTERN=all FLOOR4_MAX_GROUPS=6 \
FLOOR4_DISABLE_LIVE_POOL=1 \
VLLM_ASCEND_MODE1_PARITY_FLOOR4_MC2_FALLBACK_ON_UNPOOLED=1 \
bash run_mode1_floor4_rankset_pattern_test.sh
```

With fallback enabled, an empty `VLLM_ASCEND_MODE1_PARITY_FLOOR4_MC2_LIVE_POOL_RANK_SETS`
means no floor=4 rank set is allowed to create MC2.  This differs from the pure
live-pool diagnostic mode, where an empty whitelist historically meant every
floor=4 tuple could be pooled.  The expected analyzer evidence is a nonzero
`floor4_mc2_rebuild_skipped_for_unpooled`, nonzero
`floor4_mc2_fallback_on_unpooled`, and no floor4 MC2 slow stash/rebuild entries.
