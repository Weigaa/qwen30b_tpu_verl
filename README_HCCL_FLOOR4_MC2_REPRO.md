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
