#!/usr/bin/env bash
set -euo pipefail

# The suite can run for days. Execute an immutable same-directory copy so live
# edits cannot alter later variants or the final suite state transition.
if [[ "${ADAFLOOR_FAIR_SUITE_SNAPSHOT_ACTIVE:-0}" != "1" ]]; then
    fair_suite_source=$(realpath "${BASH_SOURCE[0]}")
    fair_suite_snapshot=$(mktemp "${fair_suite_source}.run-snapshot.XXXXXX")
    cp -- "$fair_suite_source" "$fair_suite_snapshot"
    chmod 700 "$fair_suite_snapshot"
    set +e
    ADAFLOOR_FAIR_SUITE_SNAPSHOT_ACTIVE=1 \
        "$fair_suite_snapshot" "$@"
    fair_suite_rc=$?
    set -e
    rm -f -- "$fair_suite_snapshot"
    exit "$fair_suite_rc"
fi

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR"

SINGLE_RUNNER="$SCRIPT_DIR/run_paper_fair_epoch1_2_from_common_epoch0.sh"
COMMON_EPOCH0_ROOT="${COMMON_EPOCH0_ROOT:-/data/adafloor_shared_state/common_epoch0_probe_gpu09_kv380800_permanent}"
RUN_STAMP=$(date -u +%Y%m%dT%H%M%SZ)
SUITE_ROOT="${FAIR_SUITE_ROOT:-/data/adafloor_shared_state/paper_fair_all_common_epoch0_${RUN_STAMP}}"
RESULT_ROOT="$SUITE_ROOT/results"
DRIVER_LOG_ROOT="$SUITE_ROOT/driver_logs"
STATE_ROOT="$SUITE_ROOT/state"
STATUS_FILE="$SUITE_ROOT/SUITE_STATUS.tsv"
MANIFEST_FILE="$SUITE_ROOT/SUITE_MANIFEST.txt"
MIN_FREE_GB="${FAIR_MIN_FREE_GB:-240}"
ROLLOUT_SEED="${FAIR_ROLLOUT_SEED:-0}"
DRY_RUN="${FAIR_SUITE_DRY_RUN:-${FAIR_DRY_RUN:-0}}"
RESUME="${FAIR_SUITE_RESUME:-0}"

DEFAULT_VARIANTS=(
    vanilla
    lengthsort
    lengthsort_guard
    fixed4
    minskew
    adafloor_n_f4
    adafloor_n_f2
    adafloor_p_f4
    adafloor_n_f2_noguard
)

usage() {
    cat <<'EOF'
Usage:
  ./run_paper_fair_all_epoch1_2_from_common_epoch0.sh [VARIANT ...]

With no arguments, the suite runs these variants in order:
  vanilla lengthsort lengthsort_guard fixed4 minskew
  adafloor_n_f4 adafloor_n_f2 adafloor_p_f4 adafloor_n_f2_noguard

Each variant reuses the preserved public epoch0 and runs epochs 1 and 2. Its
results are written to a separate directory under SUITE_ROOT/results. The
per-epoch checkpoints are deleted only after the single-variant launcher has
validated both epochs. Failed runs stop the suite and retain their checkpoints.

Useful environment variables:
  FAIR_SUITE_ROOT=/data/...       Fixed output root, required for resume
  FAIR_SUITE_RESUME=1             Resume a previously created suite root
  FAIR_SUITE_DRY_RUN=1            Print the execution plan without training
  FAIR_MIN_FREE_GB=240            Free-space threshold checked before each run
  FAIR_ROLLOUT_SEED=0             vLLM engine seed shared by all variants
  FAIR_VARIANTS="vanilla ..."     Variant list used when no arguments are given

Examples:
  ./run_paper_fair_all_epoch1_2_from_common_epoch0.sh

  FAIR_SUITE_ROOT=/data/adafloor_shared_state/paper_repeat_seed2 \
  FAIR_ROLLOUT_SEED=2 \
  ./run_paper_fair_all_epoch1_2_from_common_epoch0.sh vanilla adafloor_n_f2

  FAIR_SUITE_ROOT=/data/adafloor_shared_state/paper_repeat_seed2 \
  FAIR_SUITE_RESUME=1 \
  ./run_paper_fair_all_epoch1_2_from_common_epoch0.sh
EOF
}

run_name_for_variant() {
    case "$1" in
        vanilla) RUN_NAME=baseline_vanilla_common_epoch0_epoch1_2 ;;
        lengthsort) RUN_NAME=baseline_lengthsort_noguard_common_epoch0_epoch1_2 ;;
        lengthsort_guard) RUN_NAME=baseline_lengthsort_tailguard_common_epoch0_epoch1_2 ;;
        fixed4) RUN_NAME=baseline_fixed4_tailguard_common_epoch0_epoch1_2 ;;
        minskew) RUN_NAME=baseline_minskew_tailguard_common_epoch0_epoch1_2 ;;
        adafloor_n_f4) RUN_NAME=adafloor_natural_floor4_tailguard_common_epoch0_epoch1_2 ;;
        adafloor_n_f2) RUN_NAME=adafloor_natural_floor2_tailguard_common_epoch0_epoch1_2 ;;
        adafloor_p_f4) RUN_NAME=adafloor_planned_floor4_tailguard_common_epoch0_epoch1_2 ;;
        adafloor_n_f2_noguard) RUN_NAME=adafloor_natural_floor2_noguard_common_epoch0_epoch1_2 ;;
        *)
            echo "[fair suite] unknown variant: $1" >&2
            usage >&2
            return 2
            ;;
    esac
}

check_common_epoch0() {
    local reuse_env="$COMMON_EPOCH0_ROOT/reuse.env"

    if [[ ! -f "$COMMON_EPOCH0_ROOT/DO_NOT_DELETE_COMMON_EPOCH0_CHECKPOINT" \
          || ! -f "$reuse_env" ]]; then
        echo "[fair suite] common epoch0 is incomplete: $COMMON_EPOCH0_ROOT" >&2
        return 1
    fi

    # shellcheck disable=SC1090
    source "$reuse_env"
    COMMON_EPOCH0_CHECKPOINT="${DYNAMIC_INITIAL_RESUME_CKPT:-}"
    if [[ ! -d "${DYNAMIC_INITIAL_BASELINE_DIR:-}/rollout_data" \
          || ! -d "$COMMON_EPOCH0_CHECKPOINT/actor" \
          || ! -f "$COMMON_EPOCH0_CHECKPOINT/.PRESERVE_COMMON_EPOCH0" ]]; then
        echo "[fair suite] reuse.env does not reference a preserved epoch0" >&2
        return 1
    fi
}

validate_rollout_seed_config() {
    local config_dir="$SCRIPT_DIR/verl/trainer/config"

    if ! PYTHONWARNINGS=ignore python3 -c '
from hydra import compose, initialize_config_dir
from verl.utils.config import omega_conf_to_dataclass
import sys

with initialize_config_dir(config_dir=sys.argv[1], version_base=None):
    cfg = compose(
        config_name="ppo_megatron_trainer.yaml",
        overrides=[
            "actor_rollout_ref.rollout.name=vllm",
            f"actor_rollout_ref.rollout.seed={sys.argv[2]}",
        ],
    )
rollout = omega_conf_to_dataclass(cfg.actor_rollout_ref.rollout)
assert rollout.seed == int(sys.argv[2])
' "$config_dir" "$ROLLOUT_SEED"; then
        echo "[fair suite] rollout seed schema preflight failed" >&2
        return 1
    fi
    echo "[fair suite] rollout seed schema valid seed=$ROLLOUT_SEED"
}

validate_low_peak_optimizer_fix() {
    local optimizer_file="$SCRIPT_DIR/megatron/core/optimizer/optimizer.py"

    if ! python3 - "$optimizer_file" <<'PY'
import ast
import sys
from pathlib import Path

path = Path(sys.argv[1])
tree = ast.parse(path.read_text())
for node in tree.body:
    if not isinstance(node, ast.ClassDef) or node.name != "ChainedOptimizer":
        continue
    for child in node.body:
        if not isinstance(child, ast.FunctionDef) or child.name != "count_zeros":
            continue
        source = ast.get_source_segment(path.read_text(), child) or ""
        if "log_num_zeros_in_grad" in source and "if not any(" in source:
            raise SystemExit(0)
raise SystemExit(1)
PY
    then
        echo "[fair suite] missing low-peak ChainedOptimizer.count_zeros guard" >&2
        echo "[fair suite] refusing to run Planned floor4 with the known training-peak OOM path" >&2
        return 1
    fi
    echo "[fair suite] low-peak optimizer guard valid"
}

validate_shell_snapshot_guards() {
    local spec
    local file
    local marker

    for spec in \
        "run_paper_fair_all_epoch1_2_from_common_epoch0.sh:ADAFLOOR_FAIR_SUITE_SNAPSHOT_ACTIVE" \
        "run_paper_fair_epoch1_2_from_common_epoch0.sh:ADAFLOOR_FAIR_RUNNER_SNAPSHOT_ACTIVE" \
        "run_mode1_dynamic_length_aware_adaptive_floor4_epochs.sh:ADAFLOOR_DYNAMIC_DRIVER_SNAPSHOT_ACTIVE" \
        "run_mode1_local_length_sorted_e2e_adaptive_floor4.sh:ADAFLOOR_FLOOR4_CHILD_SNAPSHOT_ACTIVE" \
        "run_mode1_local_length_sorted_e2e_adaptive_floor2.sh:ADAFLOOR_FLOOR2_CHILD_SNAPSHOT_ACTIVE" \
        "run_baseline_vanilla_epoch1_2_from_common_epoch0.sh:ADAFLOOR_VANILLA_DRIVER_SNAPSHOT_ACTIVE" \
        "run_baseline_lengthsort_epoch1_2.sh:ADAFLOOR_LENGTHSORT_DRIVER_SNAPSHOT_ACTIVE" \
        "internal/wj_train_grpo_qwen30b_a3b_16die_true_weight_regroup.sh:ADAFLOOR_TRAIN_LAUNCHER_SNAPSHOT_ACTIVE"; do
        file=${spec%%:*}
        marker=${spec#*:}
        if ! grep -qF "$marker" "$SCRIPT_DIR/$file" \
           || ! grep -qF '.run-snapshot.' "$SCRIPT_DIR/$file"; then
            echo "[fair suite] missing immutable shell snapshot guard: $file" >&2
            return 1
        fi
    done
    echo "[fair suite] immutable shell snapshot guards valid"
}

validate_planned_memory_safety() {
    local dry_output
    local required_line
    local rollout_file="$SCRIPT_DIR/verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py"

    for required_line in \
        'export VLLM_ASCEND_MODE1_RELEASE_MOE_RUNTIME_BEFORE_TRAINING=1' \
        'export VLLM_ASCEND_MODE1_TRAINING_MEMORY_GUARD=1' \
        'export VLLM_ASCEND_MODE1_TRAINING_MEMORY_GUARD_STRICT=1' \
        'export VLLM_ASCEND_MODE1_OFFLOAD_LOADED_WEIGHTS_AFTER_FULL_RESTORE=1' \
        'export VLLM_ASCEND_MODE1_OFFLOAD_CANONICAL_LOADED_WEIGHTS_AFTER_FULL_RESTORE=0'; do
        if ! grep -qF "$required_line" \
                "$SCRIPT_DIR/run_mode1_dynamic_length_aware_adaptive_floor4_epochs.sh"; then
            echo "[fair suite] Planned F4 safety preflight is missing: $required_line" >&2
            return 1
        fi
    done
    if ! python3 - "$rollout_file" <<'PY'
import ast
import sys
from pathlib import Path

path = Path(sys.argv[1])
source = path.read_text()
tree = ast.parse(source)
for node in ast.walk(tree):
    if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        continue
    if node.name != "offload_model_weights":
        continue
    function_source = ast.get_source_segment(source, node) or ""
    release_pos = function_source.find("release_mode1_full_world_transient_state")
    cpu_offload_pos = function_source.find("params.data = self.cpu_model[name]")
    log_pos = function_source.find(
        "Mode1 training-boundary full-world transient cleanup:")
    if 0 <= release_pos < cpu_offload_pos and log_pos >= 0:
        raise SystemExit(0)
raise SystemExit(1)
PY
    then
        echo "[fair suite] Planned F4 safety preflight is missing or misordering the unconditional training-boundary transient cleanup" >&2
        return 1
    fi

    if ! dry_output=$(
        env \
            COMMON_EPOCH0_ROOT="$COMMON_EPOCH0_ROOT" \
            FAIR_OUTPUT_ROOT="$SUITE_ROOT/.planned_f4_safety_preflight" \
            FAIR_DRY_RUN=1 \
            VLLM_ASCEND_MODE1_OFFLOAD_LOADED_WEIGHTS_AFTER_FULL_RESTORE=0 \
            VLLM_ASCEND_MODE1_OFFLOAD_CANONICAL_LOADED_WEIGHTS_AFTER_FULL_RESTORE=1 \
            VLLM_ASCEND_MODE1_TRAINING_MIN_FREE_MIB=1 \
            "$SINGLE_RUNNER" adafloor_p_f4 2>&1
    ); then
        echo "[fair suite] Planned F4 safety dry-run failed" >&2
        echo "$dry_output" >&2
        return 1
    fi
    if ! grep -qF 'training_boundary_transient_cleanup=unconditional' <<< "$dry_output" \
       || ! grep -qF 'full_restore_transient_cleanup=1' <<< "$dry_output" \
       || ! grep -qF 'canonical_loaded_weight_offload=0' <<< "$dry_output" \
       || ! grep -qF 'planned_training_min_free_mib=28672' <<< "$dry_output"; then
        echo "[fair suite] Planned F4 launcher did not override an unsafe inherited environment" >&2
        echo "$dry_output" >&2
        return 1
    fi
    echo "[fair suite] Planned F4 memory safety preflight valid"
}

check_free_space() {
    local available_kb
    local available_gb

    available_kb=$(df -Pk "$RESULT_ROOT" | awk 'NR == 2 {print $4}')
    available_gb=$((available_kb / 1024 / 1024))
    echo "[fair suite] disk_available_gb=$available_gb required_gb=$MIN_FREE_GB"
    if (( available_gb < MIN_FREE_GB )); then
        echo "[fair suite] insufficient disk space before the next variant" >&2
        return 1
    fi
}

verify_clean_result() {
    local run_dir="$1"
    local epoch_tag
    local epoch_dirs=()
    local checkpoint_dir

    if [[ ! -f "$run_dir/CHECKPOINTS_REMOVED_AFTER_VALIDATION.txt" ]]; then
        echo "[fair suite] missing successful-cleanup record: $run_dir" >&2
        return 1
    fi

    for epoch_tag in 001 002; do
        mapfile -t epoch_dirs < <(
            find "$run_dir" -mindepth 1 -maxdepth 1 -type d \
                -name "epoch_${epoch_tag}_*" -print 2>/dev/null | sort
        )
        if (( ${#epoch_dirs[@]} != 1 )); then
            echo "[fair suite] expected one epoch_${epoch_tag} result under $run_dir" >&2
            return 1
        fi
        if (( $(find "${epoch_dirs[0]}/rollout_data" -maxdepth 1 -type f \
                    -name '*.jsonl' 2>/dev/null | wc -l) != 5 )); then
            echo "[fair suite] epoch_${epoch_tag} rollout artifacts are incomplete" >&2
            return 1
        fi
        if (( $(find "${epoch_dirs[0]}/rollout_length" -maxdepth 1 -type f \
                    -name 'length_*.txt' 2>/dev/null | wc -l) != 5 )); then
            echo "[fair suite] epoch_${epoch_tag} length artifacts are incomplete" >&2
            return 1
        fi
    done

    checkpoint_dir=$(find "$run_dir" -type d -name checkpoints -print -quit 2>/dev/null)
    if [[ -n "$checkpoint_dir" ]]; then
        echo "[fair suite] completed result still contains checkpoints: $checkpoint_dir" >&2
        return 1
    fi

    check_common_epoch0
}

append_status() {
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$1" "$2" "$3" "$4" "$5" "$6" >> "$STATUS_FILE"
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

if ! [[ "$MIN_FREE_GB" =~ ^[0-9]+$ && "$ROLLOUT_SEED" =~ ^[0-9]+$ ]]; then
    echo "[fair suite] FAIR_MIN_FREE_GB and FAIR_ROLLOUT_SEED must be nonnegative integers" >&2
    exit 2
fi
if [[ ! -x "$SINGLE_RUNNER" ]]; then
    echo "[fair suite] missing executable single-variant runner: $SINGLE_RUNNER" >&2
    exit 2
fi

if (( $# > 0 )); then
    variants=("$@")
elif [[ -n "${FAIR_VARIANTS:-}" ]]; then
    read -r -a variants <<< "$FAIR_VARIANTS"
else
    variants=("${DEFAULT_VARIANTS[@]}")
fi
if (( ${#variants[@]} == 0 )); then
    echo "[fair suite] no variants selected" >&2
    exit 2
fi
for variant in "${variants[@]}"; do
    run_name_for_variant "$variant"
done

check_common_epoch0
validate_rollout_seed_config
validate_low_peak_optimizer_fix
validate_shell_snapshot_guards
validate_planned_memory_safety

if [[ "$DRY_RUN" == "1" ]]; then
    echo "[fair suite] dry run only"
    echo "[fair suite] suite_root=$SUITE_ROOT"
    echo "[fair suite] common_epoch0=$COMMON_EPOCH0_ROOT"
    echo "[fair suite] rollout_seed=$ROLLOUT_SEED"
    for variant in "${variants[@]}"; do
        run_name_for_variant "$variant"
        echo "[fair suite] variant=$variant output=$RESULT_ROOT/$RUN_NAME"
        printf '[fair suite] command='
        printf '%q ' env \
            "COMMON_EPOCH0_ROOT=$COMMON_EPOCH0_ROOT" \
            "FAIR_OUTPUT_ROOT=$RESULT_ROOT" \
            FAIR_KEEP_COMPLETED_CHECKPOINTS=0 \
            FAIR_DRY_RUN=0 \
            "$SINGLE_RUNNER" "$variant" \
            "actor_rollout_ref.rollout.seed=$ROLLOUT_SEED"
        printf '\n'
    done
    exit 0
fi

if [[ -e "$SUITE_ROOT" && "$RESUME" != "1" ]]; then
    echo "[fair suite] suite root already exists: $SUITE_ROOT" >&2
    echo "[fair suite] set FAIR_SUITE_RESUME=1 to resume it" >&2
    exit 2
fi
if [[ -e "$SUITE_ROOT" && "$RESUME" == "1" && ! -f "$MANIFEST_FILE" ]]; then
    echo "[fair suite] refusing to resume a directory without a suite manifest: $SUITE_ROOT" >&2
    exit 2
fi

mkdir -p "$RESULT_ROOT" "$DRIVER_LOG_ROOT" "$STATE_ROOT"
if [[ ! -f "$STATUS_FILE" ]]; then
    printf 'variant\tstatus\tstarted_utc\tfinished_utc\trc\tresult_dir\n' > "$STATUS_FILE"
fi

planner_hash=$(sha256sum "$SCRIPT_DIR/tools/build_mode1_length_sorted_e2e_plan.py" | awk '{print $1}')
runner_hash=$(sha256sum "$SINGLE_RUNNER" | awk '{print $1}')
suite_runner_hash=$(sha256sum "${BASH_SOURCE[0]}" | awk '{print $1}')
rollout_schema_hash=$(sha256sum "$SCRIPT_DIR/verl/workers/config/rollout.py" | awk '{print $1}')
rollout_yaml_hash=$(sha256sum "$SCRIPT_DIR/verl/trainer/config/rollout/rollout.yaml" | awk '{print $1}')
optimizer_hash=$(sha256sum "$SCRIPT_DIR/megatron/core/optimizer/optimizer.py" | awk '{print $1}')
dynamic_driver_hash=$(sha256sum "$SCRIPT_DIR/run_mode1_dynamic_length_aware_adaptive_floor4_epochs.sh" | awk '{print $1}')
planned_wrapper_hash=$(sha256sum "$SCRIPT_DIR/run_mode1_dynamic_length_aware_adaptive_floor4_planned_full3.sh" | awk '{print $1}')
mode1_worker_hash=$(sha256sum "$SCRIPT_DIR/vllm_ascend/worker/worker_v1.py" | awk '{print $1}')
rollout_spmd_hash=$(sha256sum "$SCRIPT_DIR/verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py" | awk '{print $1}')
megatron_workers_hash=$(sha256sum "$SCRIPT_DIR/verl/workers/megatron_workers.py" | awk '{print $1}')
floor4_child_hash=$(sha256sum "$SCRIPT_DIR/run_mode1_local_length_sorted_e2e_adaptive_floor4.sh" | awk '{print $1}')
floor2_child_hash=$(sha256sum "$SCRIPT_DIR/run_mode1_local_length_sorted_e2e_adaptive_floor2.sh" | awk '{print $1}')
vanilla_driver_hash=$(sha256sum "$SCRIPT_DIR/run_baseline_vanilla_epoch1_2_from_common_epoch0.sh" | awk '{print $1}')
lengthsort_driver_hash=$(sha256sum "$SCRIPT_DIR/run_baseline_lengthsort_epoch1_2.sh" | awk '{print $1}')
train_launcher_hash=$(sha256sum "$SCRIPT_DIR/internal/wj_train_grpo_qwen30b_a3b_16die_true_weight_regroup.sh" | awk '{print $1}')
if [[ ! -f "$MANIFEST_FILE" ]]; then
    git_commit=$(git -C "$SCRIPT_DIR" rev-parse HEAD 2>/dev/null || printf 'unknown')
    {
        printf 'created_at_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
        printf 'suite_root=%s\n' "$SUITE_ROOT"
        printf 'common_epoch0_root=%s\n' "$COMMON_EPOCH0_ROOT"
        printf 'common_epoch0_checkpoint=%s\n' "$COMMON_EPOCH0_CHECKPOINT"
        printf 'rollout_seed=%s\n' "$ROLLOUT_SEED"
        printf 'variants=%s\n' "${variants[*]}"
        printf 'minimum_free_gb=%s\n' "$MIN_FREE_GB"
        printf 'git_commit=%s\n' "$git_commit"
        printf 'single_runner_sha256=%s\n' "$runner_hash"
        printf 'suite_runner_sha256=%s\n' "$suite_runner_hash"
        printf 'planner_sha256=%s\n' "$planner_hash"
        printf 'rollout_schema_sha256=%s\n' "$rollout_schema_hash"
        printf 'rollout_yaml_sha256=%s\n' "$rollout_yaml_hash"
        printf 'optimizer_sha256=%s\n' "$optimizer_hash"
        printf 'dynamic_driver_sha256=%s\n' "$dynamic_driver_hash"
        printf 'planned_wrapper_sha256=%s\n' "$planned_wrapper_hash"
        printf 'mode1_worker_sha256=%s\n' "$mode1_worker_hash"
        printf 'rollout_spmd_sha256=%s\n' "$rollout_spmd_hash"
        printf 'megatron_workers_sha256=%s\n' "$megatron_workers_hash"
        printf 'floor4_child_sha256=%s\n' "$floor4_child_hash"
        printf 'floor2_child_sha256=%s\n' "$floor2_child_hash"
        printf 'vanilla_driver_sha256=%s\n' "$vanilla_driver_hash"
        printf 'lengthsort_driver_sha256=%s\n' "$lengthsort_driver_hash"
        printf 'train_launcher_sha256=%s\n' "$train_launcher_hash"
        printf 'completed_checkpoint_policy=delete_after_both_epochs_validate\n'
        printf 'failed_checkpoint_policy=retain_and_stop\n'
        printf 'common_epoch0_preserved=true\n'
    } > "$MANIFEST_FILE"
else
    if ! grep -qFx "common_epoch0_root=$COMMON_EPOCH0_ROOT" "$MANIFEST_FILE"; then
        echo "[fair suite] resume root was created with a different common epoch0" >&2
        exit 2
    fi
    if ! grep -qFx "rollout_seed=$ROLLOUT_SEED" "$MANIFEST_FILE"; then
        echo "[fair suite] resume root was created with a different rollout seed" >&2
        exit 2
    fi
    if ! grep -qFx "single_runner_sha256=$runner_hash" "$MANIFEST_FILE" \
       || ! grep -qFx "suite_runner_sha256=$suite_runner_hash" "$MANIFEST_FILE" \
       || ! grep -qFx "planner_sha256=$planner_hash" "$MANIFEST_FILE" \
       || ! grep -qFx "rollout_schema_sha256=$rollout_schema_hash" "$MANIFEST_FILE" \
       || ! grep -qFx "rollout_yaml_sha256=$rollout_yaml_hash" "$MANIFEST_FILE" \
       || ! grep -qFx "optimizer_sha256=$optimizer_hash" "$MANIFEST_FILE" \
       || ! grep -qFx "dynamic_driver_sha256=$dynamic_driver_hash" "$MANIFEST_FILE" \
       || ! grep -qFx "planned_wrapper_sha256=$planned_wrapper_hash" "$MANIFEST_FILE" \
       || ! grep -qFx "mode1_worker_sha256=$mode1_worker_hash" "$MANIFEST_FILE" \
       || ! grep -qFx "rollout_spmd_sha256=$rollout_spmd_hash" "$MANIFEST_FILE" \
       || ! grep -qFx "megatron_workers_sha256=$megatron_workers_hash" "$MANIFEST_FILE" \
       || ! grep -qFx "floor4_child_sha256=$floor4_child_hash" "$MANIFEST_FILE" \
       || ! grep -qFx "floor2_child_sha256=$floor2_child_hash" "$MANIFEST_FILE" \
       || ! grep -qFx "vanilla_driver_sha256=$vanilla_driver_hash" "$MANIFEST_FILE" \
       || ! grep -qFx "lengthsort_driver_sha256=$lengthsort_driver_hash" "$MANIFEST_FILE" \
       || ! grep -qFx "train_launcher_sha256=$train_launcher_hash" "$MANIFEST_FILE"; then
        echo "[fair suite] launcher, planner, or rollout config changed since this suite was created" >&2
        echo "[fair suite] use a new FAIR_SUITE_ROOT to avoid mixing code revisions" >&2
        exit 2
    fi
fi

echo "[fair suite] suite_root=$SUITE_ROOT"
echo "[fair suite] variants=${variants[*]}"
echo "[fair suite] common_epoch0_checkpoint=$COMMON_EPOCH0_CHECKPOINT"
echo "[fair suite] rollout_seed=$ROLLOUT_SEED"

total=${#variants[@]}
for index in "${!variants[@]}"; do
    variant="${variants[$index]}"
    ordinal=$((index + 1))
    run_name_for_variant "$variant"
    run_dir="$RESULT_ROOT/$RUN_NAME"
    driver_log="$DRIVER_LOG_ROOT/$(printf '%02d' "$ordinal")_${variant}.log"
    state_file="$STATE_ROOT/${variant}.complete"

    echo "[fair suite] [$ordinal/$total] variant=$variant"
    if [[ -e "$run_dir" ]]; then
        if verify_clean_result "$run_dir"; then
            now=$(date -u +%Y-%m-%dT%H:%M:%SZ)
            echo "[fair suite] already complete, skipping: $run_dir"
            append_status "$variant" SKIPPED "$now" "$now" 0 "$run_dir"
            continue
        fi
        echo "[fair suite] existing result is incomplete; refusing to overwrite: $run_dir" >&2
        exit 3
    fi

    if ! check_free_space; then
        now=$(date -u +%Y-%m-%dT%H:%M:%SZ)
        append_status "$variant" BLOCKED "$now" "$now" 5 "$run_dir"
        exit 5
    fi

    started=$(date -u +%Y-%m-%dT%H:%M:%SZ)
    set +e
    env \
        COMMON_EPOCH0_ROOT="$COMMON_EPOCH0_ROOT" \
        FAIR_OUTPUT_ROOT="$RESULT_ROOT" \
        FAIR_KEEP_COMPLETED_CHECKPOINTS=0 \
        FAIR_DRY_RUN=0 \
        "$SINGLE_RUNNER" "$variant" \
        "actor_rollout_ref.rollout.seed=$ROLLOUT_SEED" \
        2>&1 | tee "$driver_log"
    run_rc=${PIPESTATUS[0]}
    set -e
    finished=$(date -u +%Y-%m-%dT%H:%M:%SZ)

    if (( run_rc != 0 )); then
        append_status "$variant" FAILED "$started" "$finished" "$run_rc" "$run_dir"
        echo "[fair suite] variant=$variant failed rc=$run_rc" >&2
        echo "[fair suite] checkpoints retained for diagnosis; suite stopped" >&2
        exit "$run_rc"
    fi
    if ! verify_clean_result "$run_dir"; then
        append_status "$variant" INVALID "$started" "$finished" 4 "$run_dir"
        echo "[fair suite] variant=$variant returned success but post-run validation failed" >&2
        exit 4
    fi

    {
        printf 'variant=%s\n' "$variant"
        printf 'completed_at_utc=%s\n' "$finished"
        printf 'result_dir=%s\n' "$run_dir"
        printf 'checkpoints_removed=true\n'
        printf 'common_epoch0_preserved=true\n'
    } > "$state_file"
    append_status "$variant" SUCCESS "$started" "$finished" 0 "$run_dir"
    echo "[fair suite] variant=$variant complete size=$(du -sh "$run_dir" | awk '{print $1}')"
done

{
    printf 'completed_at_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf 'variants=%s\n' "${variants[*]}"
    printf 'common_epoch0_preserved=true\n'
    printf 'all_completed_checkpoints_removed=true\n'
} > "$SUITE_ROOT/SUITE_COMPLETE.txt"

echo "[fair suite] all selected variants completed"
echo "[fair suite] results=$RESULT_ROOT"
echo "[fair suite] status=$STATUS_FILE"
