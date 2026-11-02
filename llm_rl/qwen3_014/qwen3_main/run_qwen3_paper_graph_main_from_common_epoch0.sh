#!/usr/bin/env bash
set -euo pipefail

# Paper protocol for the graph counterpart of the Qwen3 eager main study.
# The workload, common epoch0, policies, updated-policy trajectory, and global
# seeds match the eager study. Only the rollout execution mode and the graph
# memory reservation differ.

if [[ "${ADAFLOOR_GRAPH_MAIN_SNAPSHOT_ACTIVE:-0}" != "1" ]]; then
    source_path=$(realpath "${BASH_SOURCE[0]}")
    snapshot=$(mktemp "${source_path}.run-snapshot.XXXXXX")
    cp -- "$source_path" "$snapshot"
    chmod 700 "$snapshot"
    set +e
    ADAFLOOR_GRAPH_MAIN_SNAPSHOT_ACTIVE=1 "$snapshot" "$@"
    rc=$?
    set -e
    rm -f -- "$snapshot"
    exit "$rc"
fi

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR"

ACTION=${1:-dry-run}
case "$ACTION" in
    dry-run|run|verify|smoke) ;;
    -h|--help)
        cat <<'EOF'
Usage:
  ./run_qwen3_paper_graph_main_from_common_epoch0.sh dry-run
  ./run_qwen3_paper_graph_main_from_common_epoch0.sh run
  ./run_qwen3_paper_graph_main_from_common_epoch0.sh verify
  ./run_qwen3_paper_graph_main_from_common_epoch0.sh smoke

`run` executes the paper matrix: seven policies, global seeds 0/2/3,
and sequential updated-policy epochs 1 and 2 from the preserved eager common
epoch0. `smoke` is non-authoritative and runs seed-0 Vanilla with one rollout
step so initialization/capture can be checked and interrupted early.

Set GRAPH_MAIN_ROOT to select a result root. The default run root is stable so
an interrupted paper run can resume only after completed arms. An incomplete
arm is rejected and must be diagnosed or moved aside explicitly.
EOF
        exit 0
        ;;
    *) echo "unknown action: $ACTION" >&2; exit 2 ;;
esac
EXECUTION_MODE=${GRAPH_MAIN_EXECUTION_MODE:-graph}
case "$EXECUTION_MODE" in
    graph|eager_matched) ;;
    *)
        echo "GRAPH_MAIN_EXECUTION_MODE must be graph or eager_matched" >&2
        exit 2
        ;;
esac

COMMON_EPOCH0_ROOT=${COMMON_EPOCH0_ROOT:-/data/adafloor_shared_state/common_epoch0_probe_gpu09_kv380800_permanent}
if [[ "$ACTION" == smoke ]]; then
    default_root="/workspace/adafloor_graph_results/qwen3_paper_${EXECUTION_MODE}_main_smoke_$(date -u +%Y%m%dT%H%M%SZ)"
elif [[ "$EXECUTION_MODE" == eager_matched ]]; then
    default_root=/workspace/adafloor_graph_results/qwen3_paper_matched_eager_tq1_prompt_kv_admission_v4_minimal_controls
else
    # The preemption-safe dispatcher changes the frozen runtime code contract.
    # Use a fresh result root instead of mixing it with the earlier step-5
    # failure, whose immutable manifest names the old dispatcher.
    default_root=/workspace/adafloor_graph_results/qwen3_paper_graph_main_full_decode_prompt_kv_admission_tq1_buckets16_32_v3
fi
RESULT_ROOT=${GRAPH_MAIN_ROOT:-$default_root}
if [[ "$EXECUTION_MODE" == graph ]]; then
    reuse_seed0_vanilla_default=1
else
    reuse_seed0_vanilla_default=0
fi
REUSE_SEED0_VANILLA=${GRAPH_MAIN_REUSE_SEED0_VANILLA:-$reuse_seed0_vanilla_default}
REUSED_EXPERIMENT_ROOT=${GRAPH_MAIN_REUSED_EXPERIMENT_ROOT:-/workspace/adafloor_graph_results/qwen3_paper_full_graph_vs_eager_prompt_kv_admission_v3/graph}
REUSED_SEED0_VANILLA_ROOT="$REUSED_EXPERIMENT_ROOT/seed_0/vanilla/qwen3_graph_vanilla_seed0"
REUSE_MANIFEST="$RESULT_ROOT/reused_arms.tsv"
EXPECTED_REUSED_CODE_MANIFEST_SHA256=fa220204c518e6f73b378a4337b241ad0f7ad3e71d8a08a69e44466acc2683fb
EXPECTED_REUSED_PROTOCOL_SHA256=b98982509a4917617bf79cac265a57520bf921d6d53e834f20cd3a240e6c591c
EXPECTED_REUSED_RUN_MANIFEST_SHA256=e4dd5a77d139e4fe959d837bed090dc7c043ae717abe9805e8a0861bf94a22f7
EXPECTED_REUSED_CLEANUP_SHA256=26eff423efe595d7137dea2d34bb8c97dd55736362928f8fe54d185c74faa493

FAIR_RUNNER="$SCRIPT_DIR/run_paper_fair_epoch1_2_from_common_epoch0.sh"
GRAPH_WRAPPER="$SCRIPT_DIR/run_mode1_local_length_sorted_e2e_adaptive_floor4_aclgraph.sh"
SHARED_SUMMARIZER="$SCRIPT_DIR/../qwen3_shrink_aware/analysis_eval/summarize_existing_e2e_repetitions.py"
EAGER_CONTROL_SUMMARIZER="$SCRIPT_DIR/analysis_eval/summarize_matched_eager_controls.py"
if [[ "$EXECUTION_MODE" == graph ]]; then
    SUMMARIZER=$SHARED_SUMMARIZER
else
    SUMMARIZER=$EAGER_CONTROL_SUMMARIZER
fi
PAIR_COMPARATOR="$SCRIPT_DIR/analysis_eval/compare_qwen3_paper_graph_vs_matched_eager.py"
FULL_DECODE_PREEMPTION_VERIFIER="$SCRIPT_DIR/tools/verify_full_decode_preemption_recovery.py"
EXTENSION=${VLLM_ASCEND_ELASTIC_ACLGRAPH_EXTENSION:-/workspace/vllm-ascend/vllm_ascend/vllm_ascend_C.cpython-311-aarch64-linux-gnu.so}
FILTERED_OPP=${VLLM_ASCEND_FILTERED_CUSTOM_OPP_PATH:-$SCRIPT_DIR/../qwen3/vllm_ascend/_cann_ops_custom_moe_filtered/vendors/vllm-ascend}

CODE_CONTRACT="$RESULT_ROOT/code_sha256.txt"
PROTOCOL_CONTRACT="$RESULT_ROOT/protocol.env"
RUN_MANIFEST="$RESULT_ROOT/run_manifest.tsv"
ANALYSIS_MANIFEST="$RESULT_ROOT/analysis_manifest.tsv"
ANALYSIS_DIR="$RESULT_ROOT/analysis_eval"
DISK_CLEANUP_LOG="$RESULT_ROOT/disk_cleanup.tsv"
MIN_FREE_GIB=${GRAPH_MAIN_MIN_FREE_GIB:-120}
SMOKE_KV_TOKENS=${GRAPH_MAIN_SMOKE_KV_TOKENS:-196608}
POLICY_PROFILE=${GRAPH_MAIN_POLICY_PROFILE:-full}
CAPTURE_PLAN=${GRAPH_MAIN_CAPTURE_PLAN:-}
PROTOCOL_WRAPPER=${GRAPH_MAIN_PROTOCOL_WRAPPER:-}
ACLGRAPH_BUCKET_COUNT=${GRAPH_MAIN_ACLGRAPH_BUCKET_COUNT:-0}
if ! [[ "$ACLGRAPH_BUCKET_COUNT" =~ ^[0-9]+$ ]] \
   || (( ACLGRAPH_BUCKET_COUNT < 0 || ACLGRAPH_BUCKET_COUNT > 32 )); then
    echo "GRAPH_MAIN_ACLGRAPH_BUCKET_COUNT must be in [0,32]" >&2
    exit 2
fi

SEEDS_CSV=${GRAPH_MAIN_SEEDS_CSV:-0,2,3}
IFS=',' read -r -a SEEDS <<< "$SEEDS_CSV"
(( ${#SEEDS[@]} > 0 )) || {
    echo "GRAPH_MAIN_SEEDS_CSV must contain at least one seed" >&2
    exit 2
}
declare -A seen_seeds=()
for seed in "${SEEDS[@]}"; do
    [[ "$seed" =~ ^(0|2|3)$ ]] || {
        echo "GRAPH_MAIN_SEEDS_CSV supports only the canonical seeds 0,2,3" >&2
        exit 2
    }
    [[ -z "${seen_seeds[$seed]:-}" ]] || {
        echo "GRAPH_MAIN_SEEDS_CSV contains duplicate seed=$seed" >&2
        exit 2
    }
    seen_seeds[$seed]=1
done
if [[ "$EXECUTION_MODE" == graph ]]; then
    case "$POLICY_PROFILE" in
        full)
            POLICIES=(vanilla lengthsort lengthsort_guard minskew natural_f4 natural_f2 planned_f4)
            PLANNED_ARMS=21
            PLANNED_EPOCHS=42
            POLICIES_CSV=vanilla,lengthsort,lengthsort_guard,minskew,natural_f4,natural_f2,planned_f4
            ;;
        core3)
            POLICIES=(vanilla natural_f2 planned_f4)
            PLANNED_ARMS=9
            PLANNED_EPOCHS=18
            POLICIES_CSV=vanilla,natural_f2,planned_f4
            ;;
        *) echo "GRAPH_MAIN_POLICY_PROFILE must be full or core3" >&2; exit 2 ;;
    esac
    FORMAL_EPOCHS=(1 2)
    FORMAL_EPOCH_COUNT=2
    FORMAL_TOTAL_EPOCHS=3
    EPOCHS_CSV=1,2
else
    [[ "$POLICY_PROFILE" == full ]] || {
        echo "GRAPH_MAIN_POLICY_PROFILE applies only to graph execution" >&2
        exit 2
    }
    # Minimal causal controls agreed for the paper. These three operating
    # points isolate the graph effect without repeating every system ablation.
    POLICIES=(vanilla natural_f2 planned_f4)
    FORMAL_EPOCHS=(1)
    FORMAL_EPOCH_COUNT=1
    FORMAL_TOTAL_EPOCHS=2
    PLANNED_ARMS=9
    PLANNED_EPOCHS=9
    POLICIES_CSV=vanilla,natural_f2,planned_f4
    EPOCHS_CSV=1
fi
PLANNED_ARMS=$(( ${#SEEDS[@]} * ${#POLICIES[@]} ))
PLANNED_EPOCHS=$(( PLANNED_ARMS * FORMAL_EPOCH_COUNT ))
ONLY_ARM=${GRAPH_MAIN_ONLY_ARM:-}
DEFAULT_CAPTURE_SIZES='[16,32]'
CAPTURE_SIZES=${GRAPH_MAIN_CAPTURE_SIZES:-$DEFAULT_CAPTURE_SIZES}
KV_BLOCK_SIZE=128
KV_BYTES_PER_TOKEN=98304
KV_ADMISSION_RESERVE_TOKENS=0
KV_BLOCK_BYTES=$((KV_BLOCK_SIZE * KV_BYTES_PER_TOKEN))
GRAPH_MEMORY_PROFILE=${GRAPH_MAIN_GRAPH_MEMORY_PROFILE:-$SCRIPT_DIR/analysis_eval/qwen3_aclgraph_step_bucket_profile_all32.json}
GRAPH_RESERVE_TOOL="$SCRIPT_DIR/analysis_eval/derive_aclgraph_reserve.py"
GRAPH_RESERVE_MINIMUM_BYTES=2684354560
GRAPH_RESERVE_QUANTUM_BYTES=536870912

EFFECTIVE_GRAPH_COUNT=$(python3 - "$CAPTURE_SIZES" "$ACLGRAPH_BUCKET_COUNT" <<'PY'
import json, sys
sizes = json.loads(sys.argv[1])
if (not isinstance(sizes, list) or not sizes or
        any(not isinstance(value, int) or value < 1 or value > 32 for value in sizes) or
        sizes != sorted(set(sizes)) or sizes[-1] != 32):
    raise SystemExit("GRAPH_MAIN_CAPTURE_SIZES must be a sorted unique JSON list in [1,32] ending at 32")
adaptive_count = int(sys.argv[2])
print(adaptive_count if adaptive_count else len(sizes))
PY
)

read -r GRAPH_PROFILE_CAPTURE_GIB PROFILE_DERIVED_RESERVE_BYTES \
        PROFILE_RAW_CAPTURE_BYTES PROFILE_MINIMUM_RESERVE_BYTES < <(
    python3 "$GRAPH_RESERVE_TOOL" \
        --profile "$GRAPH_MEMORY_PROFILE" \
        --graph-count "$EFFECTIVE_GRAPH_COUNT" \
        --minimum-reserve-bytes "$GRAPH_RESERVE_MINIMUM_BYTES" \
        --quantum-bytes "$GRAPH_RESERVE_QUANTUM_BYTES" \
        --format tsv
)

if [[ -n "${GRAPH_MAIN_GRAPH_RESERVE_BYTES:-}" ]]; then
    GRAPH_RESERVE_BYTES=$GRAPH_MAIN_GRAPH_RESERVE_BYTES
    GRAPH_RESERVE_SOURCE=explicit_override
else
    GRAPH_RESERVE_BYTES=$PROFILE_DERIVED_RESERVE_BYTES
    GRAPH_RESERVE_SOURCE=profile_derived
fi
[[ "$GRAPH_RESERVE_BYTES" =~ ^[1-9][0-9]*$ ]] || {
    echo "GRAPH_MAIN_GRAPH_RESERVE_BYTES must be a positive integer" >&2
    exit 2
}
GRAPH_RESERVE_BLOCKS=$(((GRAPH_RESERVE_BYTES + KV_BLOCK_BYTES - 1) / KV_BLOCK_BYTES))
GRAPH_RESERVE_TOKENS=$((GRAPH_RESERVE_BLOCKS * KV_BLOCK_SIZE))
GRAPH_RESERVE_BASIS=${GRAPH_MAIN_GRAPH_RESERVE_BASIS:-profile_rank_max_capture_rounded_up_0p5g_with_2p5g_minimum}

if [[ -n "$ONLY_ARM" ]]; then
    canonical_policies="|$(IFS='|'; echo "${POLICIES[*]}")|"
    [[ "$ONLY_ARM" =~ ^(0|2|3):([a-z0-9_]+)$ \
       && "$canonical_policies" == *"|${BASH_REMATCH[2]}|"* ]] || {
        echo "GRAPH_MAIN_ONLY_ARM must be a canonical seed:policy pair" >&2
        exit 2
    }
fi

# Eager capacities established by the original main protocol.
# Keep the same full16 KV capacity as the working 0.11 run.  Memory pressure
# must be handled by the rollout release path rather than by changing capacity.
EAGER_FULL16=${GRAPH_MAIN_EAGER_FULL16:-380800}
EAGER_F2=131072
EAGER_F4=280576
EAGER_F8_NATURAL=315648
EAGER_F8_F4=377344

GRAPH_FULL16=$((EAGER_FULL16 - GRAPH_RESERVE_TOKENS))
GRAPH_F2=$((EAGER_F2 - GRAPH_RESERVE_TOKENS))
GRAPH_F4=$((EAGER_F4 - GRAPH_RESERVE_TOKENS))
GRAPH_F8_NATURAL=$((EAGER_F8_NATURAL - GRAPH_RESERVE_TOKENS))
GRAPH_F8_F4=$((EAGER_F8_F4 - GRAPH_RESERVE_TOKENS))
# Both causal arms use the graph-safe capacities. Giving eager its larger
# native budget would mix graph execution with KV-capacity differences.
ACTIVE_FULL16=$GRAPH_FULL16
ACTIVE_F2=$GRAPH_F2
ACTIVE_F4=$GRAPH_F4
ACTIVE_F8_NATURAL=$GRAPH_F8_NATURAL
ACTIVE_F8_F4=$GRAPH_F8_F4

sha256_file() { sha256sum "$1" | awk '{print $1}'; }

variant_for_policy() {
    case "$1" in
        vanilla|lengthsort|lengthsort_guard|minskew) printf '%s\n' "$1" ;;
        natural_f4) printf 'adafloor_n_f4\n' ;;
        natural_f2) printf 'adafloor_n_f2\n' ;;
        planned_f4) printf 'adafloor_p_f4\n' ;;
        *) return 1 ;;
    esac
}

base_runner_for_policy() {
    case "$1" in
        natural_f4) printf '%s\n' "$SCRIPT_DIR/run_mode1_dynamic_length_aware_adaptive_floor4_natural_full3.sh" ;;
        natural_f2) printf '%s\n' "$SCRIPT_DIR/run_mode1_dynamic_length_aware_adaptive_floor2_natural_tailguard_reuse_epoch0_2epoch.sh" ;;
        planned_f4) printf '%s\n' "$SCRIPT_DIR/run_mode1_dynamic_length_aware_adaptive_floor4_planned_full3.sh" ;;
        *) printf '\n' ;;
    esac
}

run_name() { printf 'qwen3_%s_%s_seed%s\n' "$EXECUTION_MODE" "$1" "$2"; }
arm_root() { printf '%s/seed_%s/%s\n' "$RESULT_ROOT" "$1" "$2"; }
reuse_seed0_vanilla_enabled() {
    [[ "$EXECUTION_MODE" == graph && "$ACTION" != smoke \
       && "$REUSE_SEED0_VANILLA" == 1 ]]
}

run_root() {
    if [[ "$1" == 0 && "$2" == vanilla ]] \
            && reuse_seed0_vanilla_enabled; then
        printf '%s\n' "$REUSED_SEED0_VANILLA_ROOT"
    else
        printf '%s/%s\n' "$(arm_root "$1" "$2")" "$(run_name "$2" "$1")"
    fi
}

require_inputs() {
    local path
    local -a required_paths=(
        "$COMMON_EPOCH0_ROOT/DO_NOT_DELETE_COMMON_EPOCH0_CHECKPOINT" \
        "$COMMON_EPOCH0_ROOT/reuse.env" \
        "$COMMON_EPOCH0_ROOT/epoch_000_mode0_probe/offline_planning_history.json" \
        "$COMMON_EPOCH0_ROOT/epoch_000_mode0_probe/checkpoints/qwen3moe_for_eagle3/global_step_5" \
        "$FAIR_RUNNER" "$SUMMARIZER" "$FILTERED_OPP"
    )
    if [[ "$EXECUTION_MODE" == graph ]]; then
        required_paths+=("$GRAPH_WRAPPER" "$EXTENSION")
    fi
    if [[ -n "$CAPTURE_PLAN" ]]; then
        required_paths+=("$CAPTURE_PLAN")
    fi
    if [[ -n "$PROTOCOL_WRAPPER" ]]; then
        required_paths+=("$PROTOCOL_WRAPPER")
    fi
    required_paths+=("$GRAPH_MEMORY_PROFILE" "$GRAPH_RESERVE_TOOL")
    for path in "${required_paths[@]}"; do
        [[ -e "$path" ]] || { echo "missing required input: $path" >&2; exit 2; }
    done
    if reuse_seed0_vanilla_enabled; then
        for path in \
            "$REUSED_EXPERIMENT_ROOT/code_sha256.txt" \
            "$REUSED_EXPERIMENT_ROOT/protocol.env" \
            "$REUSED_EXPERIMENT_ROOT/run_manifest.tsv" \
            "$REUSED_SEED0_VANILLA_ROOT/CHECKPOINTS_REMOVED_AFTER_VALIDATION.txt"; do
            [[ -f "$path" && ! -L "$path" ]] || {
                echo "missing reusable Vanilla artifact: $path" >&2
                exit 2
            }
        done
    fi
    (( GRAPH_RESERVE_BLOCKS > 0 \
       && GRAPH_RESERVE_TOKENS == GRAPH_RESERVE_BLOCKS * KV_BLOCK_SIZE )) || {
        echo "invalid graph reserve calculation" >&2
        exit 2
    }
    for value in "$ACTIVE_FULL16" "$ACTIVE_F2" "$ACTIVE_F4" \
                 "$ACTIVE_F8_NATURAL" "$ACTIVE_F8_F4"; do
        (( value > 0 && value % KV_BLOCK_SIZE == 0 )) || {
            echo "graph KV capacity is not block aligned: $value" >&2
            exit 2
        }
    done
    [[ "$MIN_FREE_GIB" =~ ^[1-9][0-9]*$ ]] || {
        echo "GRAPH_MAIN_MIN_FREE_GIB must be a positive integer" >&2
        exit 2
    }
    if [[ "$ACTION" == smoke ]] \
            && { (( SMOKE_KV_TOKENS <= 0 )) \
                 || (( SMOKE_KV_TOKENS % KV_BLOCK_SIZE != 0 )); }; then
        echo "GRAPH_MAIN_SMOKE_KV_TOKENS must be positive and block aligned" >&2
        exit 2
    fi
}

install_or_verify_immutable() {
    local candidate=$1 destination=$2 label=$3
    if [[ -e "$destination" ]]; then
        if [[ ! -f "$destination" || -L "$destination" ]] \
                || ! cmp -s -- "$candidate" "$destination"; then
            rm -f -- "$candidate"
            echo "immutable $label changed: $destination" >&2
            return 1
        fi
        rm -f -- "$candidate"
    else
        mv -- "$candidate" "$destination"
    fi
}

seal_contracts() {
    mkdir -p "$RESULT_ROOT"
    local code_candidate protocol_candidate created
    code_candidate=$(mktemp "$RESULT_ROOT/.code.XXXXXX")
    protocol_candidate=$(mktemp "$RESULT_ROOT/.protocol.XXXXXX")
    local code_paths=(
        "$SCRIPT_DIR/run_qwen3_paper_graph_main_from_common_epoch0.sh"
        "$FAIR_RUNNER"
        "$GRAPH_WRAPPER"
        "$SCRIPT_DIR/run_baseline_vanilla_epoch1_2_from_common_epoch0.sh"
        "$SCRIPT_DIR/run_baseline_lengthsort_epoch1_2.sh"
        "$SCRIPT_DIR/run_baseline_minskew_epoch1_2.sh"
        "$SCRIPT_DIR/run_mode1_dynamic_length_aware_adaptive_floor4_natural_full3.sh"
        "$SCRIPT_DIR/run_mode1_dynamic_length_aware_adaptive_floor2_natural_tailguard_reuse_epoch0_2epoch.sh"
        "$SCRIPT_DIR/run_mode1_dynamic_length_aware_adaptive_floor4_planned_full3.sh"
        "$SCRIPT_DIR/run_mode0_no_shrink_baseline.sh"
        "$SCRIPT_DIR/run_mode1_dynamic_length_aware_adaptive_floor4_epochs.sh"
        "$SCRIPT_DIR/run_mode1_local_length_sorted_e2e_adaptive_floor4.sh"
        "$SCRIPT_DIR/run_mode1_local_length_sorted_e2e_adaptive_floor2.sh"
        "$SCRIPT_DIR/internal/wj_train_grpo_qwen30b_a3b_16die_true_weight_regroup.sh"
        "$SCRIPT_DIR/tools/build_mode1_length_sorted_e2e_plan.py"
        "$SCRIPT_DIR/tools/kv_peak_calibration.py"
        "$SCRIPT_DIR/tools/build_mode1_optimized_rank_plan.py"
        "$SCRIPT_DIR/tools/aclgraph_bucket_plan.py"
        "$SCRIPT_DIR/tools/build_vanilla_aclgraph_bucket_plan.py"
        "$SCRIPT_DIR/tools/promote_kv_admission_caps.py"
        "$FULL_DECODE_PREEMPTION_VERIFIER"
        "$SHARED_SUMMARIZER"
        "$PAIR_COMPARATOR"
        "$GRAPH_RESERVE_TOOL"
        "$SCRIPT_DIR/../qwen3_shrink_aware/analysis_eval/plot_e2e_performance.py"
        "$SCRIPT_DIR/verl/experimental/dataset/shrink_aware_assignment.py"
        "$SCRIPT_DIR/verl/trainer/ppo/ray_trainer.py"
        "$SCRIPT_DIR/verl/trainer/config/rollout/rollout.yaml"
        "$SCRIPT_DIR/verl/workers/config/rollout.py"
        "$SCRIPT_DIR/verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py"
        "$SCRIPT_DIR/vllm/compilation/piecewise_backend.py"
        "$SCRIPT_DIR/vllm_ascend/platform.py"
        "$SCRIPT_DIR/vllm_ascend/attention/attention_v1.py"
        "$SCRIPT_DIR/vllm/model_executor/models/qwen3_moe.py"
        "$SCRIPT_DIR/vllm_ascend/compilation/acl_graph.py"
        "$SCRIPT_DIR/vllm_ascend/worker/model_runner_v1.py"
        "$SCRIPT_DIR/vllm_ascend/worker/worker_v1.py"
    )
    if [[ -n "$PROTOCOL_WRAPPER" ]]; then
        code_paths+=("$PROTOCOL_WRAPPER")
    fi
    if [[ "$EXECUTION_MODE" == eager_matched ]]; then
        code_paths+=(
            "$SCRIPT_DIR/run_qwen3_paper_matched_eager_tq1_from_common_epoch0.sh"
            "$EAGER_CONTROL_SUMMARIZER"
        )
    fi
    sha256sum "${code_paths[@]}" > "$code_candidate"
    if [[ -f "$PROTOCOL_CONTRACT" && ! -L "$PROTOCOL_CONTRACT" ]]; then
        created=$(sed -n 's/^created_at_utc=//p' "$PROTOCOL_CONTRACT")
    else
        created=$(date -u +%Y-%m-%dT%H:%M:%SZ)
    fi
    local execution_scope=authoritative
    if [[ "$ACTION" == smoke ]]; then
        execution_scope=smoke
    fi
    {
        printf 'schema_version=1\n'
        printf 'created_at_utc=%s\n' "$created"
        if [[ "$EXECUTION_MODE" == graph ]]; then
            printf 'experiment=qwen3_paper_graph_main_from_common_eager_epoch0\n'
        else
            printf 'experiment=qwen3_paper_matched_eager_tq1_from_common_eager_epoch0\n'
            printf 'execution_mode=eager_matched\n'
            printf 'matched_graph_protocol=FULL_DECODE_ONLY_TQ1_buckets16_32\n'
            printf 'comparison_difference=aclgraph_execution_only\n'
            printf 'control_scope=representative_vanilla_planned_f4_natural_f2\n'
        fi
        # dry-run, run, and verify are control-plane actions over one protocol.
        printf 'execution_scope=%s\n' "$execution_scope"
        printf 'common_epoch0_root=%s\n' "$COMMON_EPOCH0_ROOT"
        printf 'common_epoch0_reuse_env_sha256=%s\n' "$(sha256_file "$COMMON_EPOCH0_ROOT/reuse.env")"
        printf 'common_epoch0_history_sha256=%s\n' "$(sha256_file "$COMMON_EPOCH0_ROOT/epoch_000_mode0_probe/offline_planning_history.json")"
        printf 'seeds=%s\n' "$SEEDS_CSV"
        printf 'policies=%s\n' "$POLICIES_CSV"
        printf 'excluded_policy=fixed4\n'
        printf 'fixed4_exclusion_reason=not_a_reasonable_comparison_strategy\n'
        printf 'planned_arms=%s\n' "$PLANNED_ARMS"
        printf 'planned_epochs=%s\n' "$PLANNED_EPOCHS"
        printf 'epochs=%s\n' "$EPOCHS_CSV"
        printf 'steps_per_epoch=5\n'
        printf 'actor_frozen=false\n'
        printf 'native_sleep=true\n'
        printf 'sleep_level=%s\n' "${GRAPH_MAIN_SLEEP_LEVEL:-2}"
        printf 'manual_free_cache_engine=0\n'
        printf 'camem_weight_reload=true\n'
        printf 'actor_lr=1e-6\n'
        printf 'prompts_per_epoch=160\n'
        printf 'train_batch_size=32\n'
        printf 'rollout_n=16\n'
        printf 'responses_per_step=512\n'
        printf 'max_response_length=16384\n'
        printf 'request_sampling=global_seed_matched\n'
        printf 'policy_profile=%s\n' "$POLICY_PROFILE"
        if [[ "$EXECUTION_MODE" == graph ]]; then
            printf 'graph_mode=FULL_DECODE_ONLY\n'
            printf 'adaptive_bucket_count=%s\n' "$ACLGRAPH_BUCKET_COUNT"
            if (( ACLGRAPH_BUCKET_COUNT > 0 )); then
                printf 'capture_sizes=per_step_plan\n'
                printf 'bootstrap_capture_sizes=%s\n' "$CAPTURE_SIZES"
                printf 'capture_selection=per_epoch_per_step_predicted_minimum_padding\n'
            else
                printf 'capture_sizes=%s\n' "$CAPTURE_SIZES"
                printf 'capture_selection=fixed\n'
            fi
            printf 'task_queue_enable=1\n'
            printf 'attention_graph=true\n'
            printf 'moe_graph=true\n'
            printf 'prefill_execution=eager\n'
            if [[ -n "$CAPTURE_PLAN" ]]; then
                printf 'capture_plan=%s\n' "$(realpath "$CAPTURE_PLAN")"
                printf 'capture_plan_sha256=%s\n' "$(sha256_file "$CAPTURE_PLAN")"
            fi
        else
            printf 'graph_mode=disabled\n'
            printf 'capture_sizes=none\n'
            printf 'task_queue_enable=1\n'
            printf 'enforce_eager=true\n'
            printf 'attention_graph=false\n'
            printf 'moe_graph=false\n'
            printf 'prefix_caching=true\n'
            printf 'chunked_prefill=true\n'
        fi
        printf 'tailguard_policy=matches_eager_per_policy\n'
        printf 'kv_bytes_per_token=%s\n' "$KV_BYTES_PER_TOKEN"
        printf 'kv_block_size_tokens=%s\n' "$KV_BLOCK_SIZE"
        printf 'kv_admission_formula=block_aware_prompt_plus_predicted_response_peak\n'
        printf 'kv_admission_safety_factor=1.0\n'
        printf 'kv_admission_safety_factor_role=per_response_inflation_disabled\n'
        printf 'kv_admission_reserve_tokens_per_rank=%s\n' "$KV_ADMISSION_RESERVE_TOKENS"
        printf 'adafloor_kv_peak_model=prompt_adaptive_synchronized_with_pair_residual\n'
        printf 'kv_prompt_prior_strength=3.0\n'
        printf 'kv_prompt_minimum_factor=1.10\n'
        printf 'kv_peak_initial_factor=1.30\n'
        printf 'kv_peak_minimum_factor=1.20\n'
        printf 'kv_peak_ratio_guard=1.02\n'
        printf 'kv_peak_window=3\n'
        printf 'kv_peak_quantum=0.01\n'
        printf 'kv_peak_max_decrease_per_epoch=0.05\n'
        printf 'kv_admission_prompt_tokenizer=/data/Qwen3-30B-A3B\n'
        printf 'kv_admission_scope=release_area,minskew,contiguous\n'
        printf 'kv_block_bytes=%s\n' "$KV_BLOCK_BYTES"
        printf 'graph_reserve_bytes_per_npu=%s\n' "$GRAPH_RESERVE_BYTES"
        printf 'graph_reserve_blocks_per_npu=%s\n' "$GRAPH_RESERVE_BLOCKS"
        printf 'graph_reserve_tokens_per_floor=%s\n' "$GRAPH_RESERVE_TOKENS"
        printf 'graph_reserve_basis=%s\n' "$GRAPH_RESERVE_BASIS"
        printf 'graph_reserve_source=%s\n' "$GRAPH_RESERVE_SOURCE"
        printf 'effective_graph_count=%s\n' "$EFFECTIVE_GRAPH_COUNT"
        printf 'graph_memory_profile=%s\n' "$(realpath "$GRAPH_MEMORY_PROFILE")"
        printf 'graph_memory_profile_sha256=%s\n' "$(sha256_file "$GRAPH_MEMORY_PROFILE")"
        printf 'profile_capture_memory_gib_rank_max=%s\n' "$GRAPH_PROFILE_CAPTURE_GIB"
        printf 'profile_minimum_reserve_bytes=%s\n' "$PROFILE_MINIMUM_RESERVE_BYTES"
        printf 'profile_raw_capture_bytes=%s\n' "$PROFILE_RAW_CAPTURE_BYTES"
        printf 'profile_reserve_quantum_bytes=%s\n' "$GRAPH_RESERVE_QUANTUM_BYTES"
        if [[ "$EXECUTION_MODE" == graph ]]; then
            printf 'kv_formula=graph_cap=eager_verified_cap-graph_reserve_tokens\n'
            printf 'graph_full16_tokens=%s\n' "$GRAPH_FULL16"
            if [[ "$ACTION" == smoke ]]; then
                printf 'non_authoritative_smoke_kv_tokens=%s\n' "$SMOKE_KV_TOKENS"
            fi
            printf 'graph_floor2_tokens=%s\n' "$GRAPH_F2"
            printf 'graph_floor4_tokens=%s\n' "$GRAPH_F4"
            printf 'graph_floor8_natural_tokens=%s\n' "$GRAPH_F8_NATURAL"
            printf 'graph_floor8_floor4_policy_tokens=%s\n' "$GRAPH_F8_F4"
            printf 'planned_floor4_effective_tokens=%s\n' "$((GRAPH_F4 - 147456))"
            printf 'planned_floor8_effective_tokens=%s\n' "$((GRAPH_F8_F4 - 114688))"
            printf 'planned_headroom_accounting=separate_after_graph_reserve\n'
        else
            printf 'kv_formula=matched_eager_cap=graph_cap\n'
            printf 'eager_full16_tokens=%s\n' "$ACTIVE_FULL16"
            printf 'eager_floor2_tokens=%s\n' "$ACTIVE_F2"
            printf 'eager_floor4_tokens=%s\n' "$ACTIVE_F4"
            printf 'eager_floor8_natural_tokens=%s\n' "$ACTIVE_F8_NATURAL"
            printf 'eager_floor8_floor4_policy_tokens=%s\n' "$ACTIVE_F8_F4"
            printf 'matched_graph_full16_tokens=%s\n' "$GRAPH_FULL16"
            printf 'matched_graph_floor2_tokens=%s\n' "$GRAPH_F2"
            printf 'matched_graph_floor4_tokens=%s\n' "$GRAPH_F4"
            printf 'matched_graph_floor8_natural_tokens=%s\n' "$GRAPH_F8_NATURAL"
            printf 'matched_graph_floor8_floor4_policy_tokens=%s\n' "$GRAPH_F8_F4"
        fi
        printf 'native_sleep_mode=0\n'
        printf 'async_scheduling=false\n'
        if [[ "$EXECUTION_MODE" == graph ]]; then
            printf 'preemption_safe_full_decode_recovery=true\n'
            printf 'preemption_recovery_signal=existing_dp_metadata_sync\n'
            printf 'additional_preemption_recovery_collectives=0\n'
            printf 'preemption_iteration_runtime_mode=eager\n'
            printf 'next_pure_decode_iteration_runtime_mode=full_decode_only\n'
        else
            printf 'preemption_recovery_runtime_mode=eager\n'
        fi
        printf 'filtered_custom_opp=true\n'
        printf 'minimum_free_gib_before_arm=%s\n' "$MIN_FREE_GIB"
        printf 'completed_arm_cache_cleanup=true\n'
        printf 'common_epoch0_cleanup_forbidden=true\n'
        printf 'reuse_seed0_vanilla=%s\n' "$REUSE_SEED0_VANILLA"
        if reuse_seed0_vanilla_enabled; then
            printf 'reused_seed0_vanilla_root=%s\n' "$REUSED_SEED0_VANILLA_ROOT"
            printf 'reused_code_manifest_sha256=%s\n' "$EXPECTED_REUSED_CODE_MANIFEST_SHA256"
            printf 'reused_protocol_sha256=%s\n' "$EXPECTED_REUSED_PROTOCOL_SHA256"
            printf 'reused_run_manifest_sha256=%s\n' "$EXPECTED_REUSED_RUN_MANIFEST_SHA256"
        fi
        if [[ "$EXECUTION_MODE" == graph ]]; then
            printf 'extension_sha256=%s\n' "$(sha256_file "$EXTENSION")"
        fi
        printf 'filtered_opp_bundle_sha256=%s\n' "$FILTERED_OPP_BUNDLE_SHA256"
    } > "$protocol_candidate"
    install_or_verify_immutable "$code_candidate" "$CODE_CONTRACT" code
    install_or_verify_immutable "$protocol_candidate" "$PROTOCOL_CONTRACT" protocol
}

manifest_status() {
    local seed=$1 policy=$2
    [[ -f "$RUN_MANIFEST" ]] || return 1
    awk -F '\t' -v s="$seed" -v p="$policy" \
        'NR>1 && $2==s && $3==p && $4=="complete" {found=1} END {exit !found}' \
        "$RUN_MANIFEST"
}

initialize_manifest() {
    local created=0
    if [[ ! -e "$RUN_MANIFEST" ]]; then
        printf 'launch_index\ttrial_seed\tpolicy\tstatus\trun_dir\n' > "$RUN_MANIFEST"
        created=1
    fi
    [[ $(head -n 1 "$RUN_MANIFEST") == $'launch_index\ttrial_seed\tpolicy\tstatus\trun_dir' ]] || {
        echo "invalid run manifest header: $RUN_MANIFEST" >&2
        exit 2
    }
    if reuse_seed0_vanilla_enabled; then
        verify_reused_seed0_vanilla
        if (( created )); then
            printf '1\t0\tvanilla\tcomplete\t%s\n' \
                "$REUSED_SEED0_VANILLA_ROOT" >> "$RUN_MANIFEST"
        fi
        grep -Fxq $'1\t0\tvanilla\tcomplete\t'"$REUSED_SEED0_VANILLA_ROOT" \
            "$RUN_MANIFEST" || {
                echo "run manifest does not contain the audited reused Vanilla arm" >&2
                exit 2
            }
    fi
}

free_gib() {
    df -PB1 "$RESULT_ROOT" | awk 'NR==2 {print int($4 / 1024 / 1024 / 1024)}'
}

require_disk_headroom() {
    local available
    available=$(free_gib)
    if ! [[ "$available" =~ ^[0-9]+$ ]] || (( available < MIN_FREE_GIB )); then
        echo "insufficient free space before next arm: available=${available:-unknown}GiB required=${MIN_FREE_GIB}GiB filesystem=$RESULT_ROOT" >&2
        return 1
    fi
    echo "[graph main] disk headroom available=${available}GiB required=${MIN_FREE_GIB}GiB"
}

cleanup_transient_dirs() {
    local seed=$1 policy=$2 cache_dir=$3 ray_tmp=$4
    local expected_cache actual_cache common_real
    expected_cache=$(realpath -m "$(arm_root "$seed" "$policy")/cache")
    actual_cache=$(realpath -m "$cache_dir")
    common_real=$(realpath -m "$COMMON_EPOCH0_ROOT")
    if [[ "$actual_cache" != "$expected_cache" \
          || "$actual_cache" == "$common_real" \
          || "$actual_cache" == "$common_real/"* \
          || -L "$cache_dir" ]]; then
        echo "refusing unsafe cache cleanup: $cache_dir" >&2
        return 1
    fi
    if [[ -e "$cache_dir" ]]; then
        rm -rf -- "$cache_dir"
    fi
    if [[ "$ray_tmp" != /tmp/ag_main_* || -L "$ray_tmp" ]]; then
        echo "refusing unsafe Ray temp cleanup: $ray_tmp" >&2
        return 1
    fi
    if [[ -e "$ray_tmp" ]]; then
        rm -rf -- "$ray_tmp"
    fi
}

record_disk_cleanup() {
    local seed=$1 policy=$2 available
    available=$(free_gib)
    if [[ ! -e "$DISK_CLEANUP_LOG" ]]; then
        printf 'completed_at_utc\ttrial_seed\tpolicy\tcheckpoint_dirs_remaining\tcache_removed\tfree_gib\n' > "$DISK_CLEANUP_LOG"
    fi
    printf '%s\t%s\t%s\t0\ttrue\t%s\n' \
        "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$seed" "$policy" "$available" \
        >> "$DISK_CLEANUP_LOG"
    echo "[graph main] post-arm cleanup seed=$seed policy=$policy free=${available}GiB"
}

record_complete() {
    local launch_index=$1 seed=$2 policy=$3 root=$4
    if manifest_status "$seed" "$policy"; then
        return 0
    fi
    printf '%s\t%s\t%s\tcomplete\t%s\n' \
        "$launch_index" "$seed" "$policy" "$root" >> "$RUN_MANIFEST"
}

policy_kv_env() {
    local policy=$1
    KV_ENV=(
        "KV_ADMISSION_PEAK_MODEL=prompt_adaptive"
        "ACTIVE_PEAK_SAFETY_FACTOR=1.0"
        "KV_PROMPT_PRIOR_STRENGTH=3.0"
        "KV_PROMPT_MINIMUM_FACTOR=1.10"
        "KV_PEAK_INITIAL_FACTOR=1.30"
        "KV_PEAK_MINIMUM_FACTOR=1.20"
        "KV_PEAK_RATIO_GUARD=1.02"
        "KV_PEAK_WINDOW=3"
        "VANILLA_KV_TOKENS_PER_RANK=$ACTIVE_FULL16"
        "VANILLA_KV_ADMISSION_TOKENS_PER_RANK=$ACTIVE_FULL16"
        "VLLM_ASCEND_MODE1_PARITY_MAX_KV_TOKENS=$ACTIVE_FULL16"
        "VLLM_ASCEND_MODE1_PARITY_MAX_KV_TOKENS_FLOOR16=$ACTIVE_FULL16"
    )
    if [[ "$ACTION" == smoke && "$policy" == vanilla ]]; then
        KV_ENV=(
            "VANILLA_KV_TOKENS_PER_RANK=$SMOKE_KV_TOKENS"
            "VANILLA_KV_ADMISSION_TOKENS_PER_RANK=$SMOKE_KV_TOKENS"
            "VLLM_ASCEND_MODE1_PARITY_MAX_KV_TOKENS=$SMOKE_KV_TOKENS"
            "VLLM_ASCEND_MODE1_PARITY_MAX_KV_TOKENS_FLOOR16=$SMOKE_KV_TOKENS"
        )
    fi
    case "$policy" in
        vanilla|lengthsort|lengthsort_guard) ;;
        minskew|natural_f2)
            KV_ENV+=(
                "VLLM_ASCEND_MODE1_PARITY_MAX_KV_TOKENS_FLOOR2=$ACTIVE_F2"
                "VLLM_ASCEND_MODE1_PARITY_MAX_KV_TOKENS_FLOOR4=$ACTIVE_F4"
                "VLLM_ASCEND_MODE1_PARITY_MAX_KV_TOKENS_FLOOR8=$ACTIVE_F8_NATURAL"
            )
            ;;
        natural_f4|planned_f4)
            KV_ENV+=(
                "VLLM_ASCEND_MODE1_PARITY_MAX_KV_TOKENS_FLOOR2=$ACTIVE_F2"
                "VLLM_ASCEND_MODE1_PARITY_MAX_KV_TOKENS_FLOOR4=$ACTIVE_F4"
                "VLLM_ASCEND_MODE1_PARITY_MAX_KV_TOKENS_FLOOR8=$ACTIVE_F8_F4"
            )
            ;;
        *) echo "unknown policy: $policy" >&2; return 2 ;;
    esac
}

verify_smoke_arm() {
    local policy=$1 root=$2 epoch_dir log count bucket_plan
    local -a smoke_epoch_dirs=()
    mapfile -t smoke_epoch_dirs < <(
        find "$root" -mindepth 1 -maxdepth 1 -type d -name 'epoch_001_*' | sort
    )
    (( ${#smoke_epoch_dirs[@]} == 1 )) || {
        echo "smoke expected one epoch directory: $root" >&2; return 1;
    }
    epoch_dir=${smoke_epoch_dirs[0]}
    count=$(find "$epoch_dir/rollout_data" -maxdepth 1 -type f -name '*.jsonl' | wc -l)
    (( count == 1 )) || { echo "smoke expected one rollout file: $epoch_dir" >&2; return 1; }
    count=$(find "$epoch_dir/rollout_length" -maxdepth 1 -type f -name 'length_*.txt' | wc -l)
    (( count == 1 )) || { echo "smoke expected one length file: $epoch_dir" >&2; return 1; }
    log=$(find "$epoch_dir/logs" -maxdepth 1 -type f -name '*.txt' -printf '%T@ %p\n' \
        | sort -n | tail -1 | cut -d' ' -f2-)
    [[ -n "$log" ]] || { echo "smoke missing primary log: $epoch_dir" >&2; return 1; }
    for marker in \
        'FULL_DECODE_ONLY compilation enabled on NPU' \
        'Native FULL_DECODE_ONLY ACLGraph enabled' \
        'FULL_DECODE_ONLY ACLGraph captured Attention KV write and paged read' \
        'Graph capturing finished' \
        'Replaying aclgraph' \
        'response/aborted_ratio:0.0' \
        'After trainer.fit'; do
        grep -qF "$marker" "$log" || {
            echo "smoke marker missing ($marker): $log" >&2; return 1;
        }
    done
    if (( ACLGRAPH_BUCKET_COUNT > 0 )); then
        bucket_plan=$(find "$epoch_dir/oracle" -maxdepth 1 -type f \
            \( -name 'aclgraph_bucket_plan.json' -o -name 'length_sorted_rank_plan.json' \) \
            -print | head -1)
        [[ -n "$bucket_plan" ]] || {
            echo "smoke missing epoch-planned bucket artifact: $epoch_dir" >&2; return 1;
        }
        python3 - "$bucket_plan" "$ACLGRAPH_BUCKET_COUNT" <<'PY'
import json, sys
entries = json.load(open(sys.argv[1], encoding="utf-8"))
count = int(sys.argv[2])
if not isinstance(entries, list) or not entries:
    raise SystemExit("bucket plan must be a non-empty list")
sizes = entries[0].get("aclgraph_capture_sizes")
if not isinstance(sizes, list) or len(sizes) != count or sizes != sorted(set(sizes)) or sizes[-1] != 32:
    raise SystemExit(f"invalid first-step bucket set: {sizes}")
PY
        grep -qF 'Elastic ACLGraph configured epoch-planned step buckets' "$log" || {
            echo "smoke did not apply epoch-planned buckets: $log" >&2; return 1;
        }
    fi
    if [[ "$policy" == natural_f2 || "$policy" == planned_f4 ]]; then
        grep -qF 'Elastic parallel shrink done' "$log" || {
            echo "smoke did not exercise AdaFloor shrink: $log" >&2; return 1;
        }
    fi
    echo "[graph main smoke] verified policy=$policy root=$root"
}

epoch_dirs_for_arm() {
    local root=$1 epoch
    for epoch in "${FORMAL_EPOCHS[@]}"; do
        find "$root" -mindepth 1 -maxdepth 1 -type d \
            -name "epoch_$(printf '%03d' "$epoch")_*" | sort
    done
}

verify_arm() {
    local seed=$1 policy=$2 root=$3
    [[ -f "$root/CHECKPOINTS_REMOVED_AFTER_VALIDATION.txt" ]] || {
        echo "missing checkpoint cleanup marker: $root" >&2; return 1;
    }
    if find "$root" -type d \
            \( -name checkpoints -o -name 'global_step_*' \) \
            -print -quit | grep -q .; then
        echo "validated arm still contains checkpoint directories: $root" >&2
        return 1
    fi
    local epoch_dirs epoch_dir log count bucket_plan
    mapfile -t epoch_dirs < <(epoch_dirs_for_arm "$root")
    (( ${#epoch_dirs[@]} == FORMAL_EPOCH_COUNT )) || {
        echo "expected $FORMAL_EPOCH_COUNT epoch directories: $root" >&2; return 1;
    }
    for epoch_dir in "${epoch_dirs[@]}"; do
        count=$(find "$epoch_dir/rollout_data" -maxdepth 1 -type f -name '*.jsonl' | wc -l)
        (( count == 5 )) || { echo "expected five rollout files: $epoch_dir" >&2; return 1; }
        count=$(find "$epoch_dir/rollout_length" -maxdepth 1 -type f -name 'length_*.txt' | wc -l)
        (( count == 5 )) || { echo "expected five length files: $epoch_dir" >&2; return 1; }
        log=$(find "$epoch_dir/logs" -maxdepth 1 -type f -name '*.txt' -printf '%T@ %p\n' \
            | sort -n | tail -1 | cut -d' ' -f2-)
        [[ -n "$log" ]] || { echo "missing primary log: $epoch_dir" >&2; return 1; }
        if [[ "$EXECUTION_MODE" == graph ]]; then
            grep -q 'FULL_DECODE_ONLY compilation enabled on NPU' "$log" || return 1
            grep -q 'Native FULL_DECODE_ONLY ACLGraph enabled' "$log" || return 1
            grep -q 'FULL_DECODE_ONLY ACLGraph captured Attention KV write and paged read' "$log" || return 1
            grep -q 'Graph capturing finished' "$log" || return 1
            grep -q 'Replaying aclgraph' "$log" || return 1
            if (( ACLGRAPH_BUCKET_COUNT > 0 )); then
                bucket_plan=$(find "$epoch_dir/oracle" -maxdepth 1 -type f \
                    \( -name 'aclgraph_bucket_plan.json' -o -name 'length_sorted_rank_plan.json' \) \
                    -print | head -1)
                [[ -n "$bucket_plan" ]] || {
                    echo "missing epoch bucket plan: $epoch_dir" >&2; return 1;
                }
                python3 - "$bucket_plan" "$ACLGRAPH_BUCKET_COUNT" <<'PY'
import json, sys
entries = json.load(open(sys.argv[1], encoding="utf-8"))
count = int(sys.argv[2])
if not isinstance(entries, list) or len(entries) != 5:
    raise SystemExit("formal bucket plan must contain five step entries")
for expected_step, entry in enumerate(entries, 1):
    sizes = entry.get("aclgraph_capture_sizes")
    if entry.get("step") != expected_step or entry.get("aclgraph_bucket_count") != count:
        raise SystemExit(f"invalid bucket metadata at step {expected_step}")
    if not isinstance(sizes, list) or len(sizes) != count or sizes != sorted(set(sizes)) or sizes[-1] != 32:
        raise SystemExit(f"invalid capture sizes at step {expected_step}: {sizes}")
PY
                count=$(grep -cF 'Applied epoch-planned ACLGraph buckets:' "$log" || true)
                (( count == 5 )) || {
                    echo "expected five planned bucket selections, got $count: $log" >&2; return 1;
                }
                grep -qF 'Elastic ACLGraph configured epoch-planned step buckets' "$log" || {
                    echo "worker did not apply epoch-planned buckets: $log" >&2; return 1;
                }
            fi
        else
            grep -q "'enforce_eager': True" "$log" || {
                echo "matched eager runtime evidence missing: $log" >&2; return 1;
            }
            grep -q "'enable_prefix_caching': True" "$log" || {
                echo "matched eager prefix-cache evidence missing: $log" >&2; return 1;
            }
            grep -q "'TASK_QUEUE_ENABLE': '1'" "$log" || {
                echo "matched eager TQ1 evidence missing: $log" >&2; return 1;
            }
            grep -q "'VLLM_ASCEND_ELASTIC_ACLGRAPH': '0'" "$log" || {
                echo "matched eager graph-disable evidence missing: $log" >&2; return 1;
            }
            if grep -Eq 'FULL_DECODE_ONLY compilation enabled on NPU|Native FULL_DECODE_ONLY ACLGraph enabled|Graph capturing finished|Replaying aclgraph' "$log"; then
                echo "matched eager arm unexpectedly used ACLGraph: $log" >&2
                return 1
            fi
        fi
        grep -Eq 'actor/lr:1e-0*6([^0-9.]|$)' "$log" || {
            echo "updated-policy actor/lr evidence missing: $log" >&2; return 1;
        }
    done
    echo "[graph main] verified seed=$seed policy=$policy"
}

arm_execution_complete() {
    local root=$1 epoch_dir log count
    local -a epoch_dirs=()
    mapfile -t epoch_dirs < <(epoch_dirs_for_arm "$root")
    (( ${#epoch_dirs[@]} == FORMAL_EPOCH_COUNT )) || return 1
    for epoch_dir in "${epoch_dirs[@]}"; do
        count=$(find "$epoch_dir/rollout_data" -maxdepth 1 -type f \
            -name '*.jsonl' | wc -l)
        (( count == 5 )) || return 1
        count=$(find "$epoch_dir/rollout_length" -maxdepth 1 -type f \
            -name 'length_*.txt' | wc -l)
        (( count == 5 )) || return 1
        log=$(find "$epoch_dir/logs" -maxdepth 1 -type f -name '*.txt' \
            -printf '%T@ %p\n' | sort -n | tail -1 | cut -d' ' -f2-)
        [[ -n "$log" ]] || return 1
        [[ $(grep -cF 'rollout_output_time_s:' "$log" || true) == 5 ]] \
            || return 1
        grep -qF 'Training Progress: 100%' "$log" || return 1
        grep -qF 'After trainer.fit' "$log" || return 1
    done
}

finalize_existing_arm() {
    local variant=$1 arm_dir=$2 name=$3
    local -a finalize_env=()
    if [[ "$EXECUTION_MODE" == graph ]]; then
        finalize_env+=("ADAFLOOR_ACLGRAPH_MODE=FULL_DECODE_ONLY")
    else
        finalize_env+=(
            "TASK_QUEUE_ENABLE=1"
            "ROLLOUT_ENFORCE_EAGER=True"
            "VLLM_ASCEND_ELASTIC_ACLGRAPH=0"
        )
    fi
    env "${finalize_env[@]}" \
        "COMMON_EPOCH0_ROOT=$COMMON_EPOCH0_ROOT" \
        "FAIR_OUTPUT_ROOT=$arm_dir" \
        "FAIR_RUN_NAME=$name" \
        "DYNAMIC_RUN_NAME=$name" \
        "FAIR_START_EPOCH=1" \
        "FAIR_TOTAL_EPOCHS=$FORMAL_TOTAL_EPOCHS" \
        "FAIR_TRAIN_STEPS=5" \
        "FAIR_TRAIN_BATCH_SIZE=32" \
        "FAIR_ROLLOUT_N=16" \
        "FAIR_MAX_RESPONSE_LENGTH=16384" \
        "FAIR_PROMPTS_PER_EPOCH=160" \
        "FAIR_FREEZE_ACTOR=0" \
        "FAIR_KEEP_COMPLETED_CHECKPOINTS=0" \
        "FAIR_FINALIZE_EXISTING=1" \
        "VERL_PAIRED_REQUEST_SAMPLING_SEEDS=0" \
        "$FAIR_RUNNER" "$variant"
}

verify_reused_seed0_vanilla() {
    local code_manifest="$REUSED_EXPERIMENT_ROOT/code_sha256.txt"
    local protocol="$REUSED_EXPERIMENT_ROOT/protocol.env"
    local source_manifest="$REUSED_EXPERIMENT_ROOT/run_manifest.tsv"
    local cleanup="$REUSED_SEED0_VANILLA_ROOT/CHECKPOINTS_REMOVED_AFTER_VALIDATION.txt"
    local actual candidate log

    actual=$(sha256_file "$code_manifest")
    [[ "$actual" == "$EXPECTED_REUSED_CODE_MANIFEST_SHA256" ]] || {
        echo "reused code manifest hash mismatch: $actual" >&2; return 1;
    }
    actual=$(sha256_file "$protocol")
    [[ "$actual" == "$EXPECTED_REUSED_PROTOCOL_SHA256" ]] || {
        echo "reused protocol hash mismatch: $actual" >&2; return 1;
    }
    actual=$(sha256_file "$source_manifest")
    [[ "$actual" == "$EXPECTED_REUSED_RUN_MANIFEST_SHA256" ]] || {
        echo "reused run manifest hash mismatch: $actual" >&2; return 1;
    }
    actual=$(sha256_file "$cleanup")
    [[ "$actual" == "$EXPECTED_REUSED_CLEANUP_SHA256" ]] || {
        echo "reused cleanup marker hash mismatch: $actual" >&2; return 1;
    }
    grep -Fxq $'1\t0\tvanilla\tcomplete\t'"$REUSED_SEED0_VANILLA_ROOT" \
        "$source_manifest" || {
            echo "source manifest does not verify seed-0 Vanilla" >&2; return 1;
        }

    # Only the top-level orchestrator and the preemption recovery dispatcher
    # may differ. Every other frozen execution file must still match the
    # source run byte for byte.
    local -a mismatches=()
    mapfile -t mismatches < <(
        sha256sum -c "$code_manifest" 2>&1 \
            | sed -n 's/: FAILED$//p' \
            | sort
    )
    local -a expected_mismatches=(
        "$PAIR_COMPARATOR"
        "$SCRIPT_DIR/run_qwen3_paper_graph_main_from_common_epoch0.sh"
    )
    mapfile -t expected_mismatches < <(printf '%s\n' "${expected_mismatches[@]}" | sort)
    [[ "${mismatches[*]}" == "${expected_mismatches[*]}" ]] || {
        echo "unexpected code drift while reusing seed-0 Vanilla" >&2
        printf 'expected mismatch: %s\n' "${expected_mismatches[@]}" >&2
        printf 'actual mismatch: %s\n' "${mismatches[@]}" >&2
        return 1
    }

    verify_arm 0 vanilla "$REUSED_SEED0_VANILLA_ROOT"
    while IFS= read -r log; do
        ! grep -q 'Preempting request' "$log" || {
            echo "reused Vanilla unexpectedly contains preemption: $log" >&2
            return 1
        }
        [[ $(grep -c 'response/aborted_ratio:0.0' "$log") == 5 ]] || {
            echo "reused Vanilla does not contain five zero-abort steps: $log" >&2
            return 1
        }
        [[ $(grep -c 'After trainer.fit' "$log") == 1 ]] || {
            echo "reused Vanilla did not finish its epoch: $log" >&2
            return 1
        }
    done < <(find "$REUSED_SEED0_VANILLA_ROOT" -type f -path '*/logs/*.txt' | sort)

    candidate=$(mktemp "$RESULT_ROOT/.reused_arms.XXXXXX")
    {
        printf 'trial_seed\tpolicy\tsource_run_dir\tsource_code_manifest_sha256\tsource_protocol_sha256\tsource_run_manifest_sha256\tcleanup_marker_sha256\tpreemption_events\taborted_responses\n'
        printf '0\tvanilla\t%s\t%s\t%s\t%s\t%s\t0\t0\n' \
            "$REUSED_SEED0_VANILLA_ROOT" \
            "$EXPECTED_REUSED_CODE_MANIFEST_SHA256" \
            "$EXPECTED_REUSED_PROTOCOL_SHA256" \
            "$EXPECTED_REUSED_RUN_MANIFEST_SHA256" \
            "$EXPECTED_REUSED_CLEANUP_SHA256"
    } > "$candidate"
    install_or_verify_immutable "$candidate" "$REUSE_MANIFEST" reused-arm
    echo "[graph main] imported validated seed=0 policy=vanilla without rerun"
}

run_arm() {
    local launch_index=$1 seed=$2 policy=$3 smoke_steps=${4:-5}
    local variant base arm_dir root name cache_dir ray_tmp fair_total_epochs
    variant=$(variant_for_policy "$policy")
    base=$(base_runner_for_policy "$policy")
    arm_dir=$(arm_root "$seed" "$policy")
    root=$(run_root "$seed" "$policy")
    name=$(run_name "$policy" "$seed")
    cache_dir="$arm_dir/cache"
    ray_tmp="/tmp/ag_main_${seed}_${policy}_${$}"
    fair_total_epochs=$FORMAL_TOTAL_EPOCHS
    if (( smoke_steps != 5 )); then
        fair_total_epochs=2
    fi

    if manifest_status "$seed" "$policy"; then
        verify_arm "$seed" "$policy" "$root"
        cleanup_transient_dirs "$seed" "$policy" "$cache_dir" "$ray_tmp"
        echo "[graph main] already complete seed=$seed policy=$policy"
        return 0
    fi
    if [[ -e "$root" ]]; then
        if (( smoke_steps == 5 )) \
                && [[ -f "$root/CHECKPOINTS_REMOVED_AFTER_VALIDATION.txt" ]]; then
            verify_arm "$seed" "$policy" "$root"
            cleanup_transient_dirs "$seed" "$policy" "$cache_dir" "$ray_tmp"
            seal_contracts
            record_complete "$launch_index" "$seed" "$policy" "$root"
            record_disk_cleanup "$seed" "$policy"
            echo "[graph main] recovered completed arm seed=$seed policy=$policy"
            return 0
        fi
        if (( smoke_steps == 5 )) && arm_execution_complete "$root"; then
            echo "[graph main] finalizing completed execution without NPU rerun seed=$seed policy=$policy"
            finalize_existing_arm "$variant" "$arm_dir" "$name"
            verify_arm "$seed" "$policy" "$root"
            cleanup_transient_dirs "$seed" "$policy" "$cache_dir" "$ray_tmp"
            seal_contracts
            record_complete "$launch_index" "$seed" "$policy" "$root"
            record_disk_cleanup "$seed" "$policy"
            echo "[graph main] recovered and finalized arm seed=$seed policy=$policy"
            return 0
        fi
        echo "incomplete arm exists; move it aside or select a new GRAPH_MAIN_ROOT: $root" >&2
        return 1
    fi
    mkdir -p "$arm_dir" "$ray_tmp" "$cache_dir"/{xdg,hf,triton,torchair,ascend,work}
    policy_kv_env "$policy"
    seal_contracts
    require_disk_headroom

    local run_env=(
        "COMMON_EPOCH0_ROOT=$COMMON_EPOCH0_ROOT"
        "FAIR_OUTPUT_ROOT=$arm_dir"
        "FAIR_RUN_NAME=$name"
        "DYNAMIC_RUN_NAME=$name"
        "FAIR_START_EPOCH=1"
        "FAIR_TOTAL_EPOCHS=$fair_total_epochs"
        "FAIR_TRAIN_STEPS=$smoke_steps"
        "FAIR_TRAIN_BATCH_SIZE=32"
        "FAIR_ROLLOUT_N=16"
        "FAIR_MAX_RESPONSE_LENGTH=16384"
        "FAIR_PROMPTS_PER_EPOCH=160"
        "FAIR_FREEZE_ACTOR=0"
        "FAIR_KEEP_COMPLETED_CHECKPOINTS=0"
        "VERL_PAIRED_REQUEST_SAMPLING_SEEDS=0"
        # AdaFloor elastic ACLGraph intentionally disables TorchAir's outer
        # graph gate; native FULL_DECODE_ONLY ACLGraph is selected explicitly
        # by the Hydra cudagraph_mode below.
        "VLLM_ENABLE_GRAPH_MODE=0"
        "TASK_QUEUE_ENABLE=1"
        "VLLM_ROLLOUT_NATIVE_SLEEP_MODE=1"
        # Match the working 0.11 rollout/actor handoff: level 2 releases
        # inference KV/graph allocations before actor.update_policy.
        "VLLM_ROLLOUT_SLEEP_LEVEL=${GRAPH_MAIN_SLEEP_LEVEL:-2}"
        # Use the 0.11 working weight-pool reload path.  Without it v0.14
        # sleep reports 0 GiB released and leaves inference allocations live.
        "VLLM_ROLLOUT_CAMEM_WEIGHT_RELOAD=1"
            "VLLM_ROLLOUT_MANUAL_FREE_CACHE_ENGINE=0"
            "VLLM_ROLLOUT_INVALIDATE_ACLGRAPH_AFTER_WEIGHT_UPDATE=0"
            "VLLM_ROLLOUT_DELAY_GRAPH_CAPTURE_UNTIL_WEIGHT_LOAD=0"
            "VLLM_ROLLOUT_CAPTURE_GRAPH_AFTER_WEIGHT_LOAD=0"
            "VLLM_ROLLOUT_DEFER_KV_CACHE_UNTIL_WEIGHT_LOAD=0"
        "VLLM_ROLLOUT_FILTER_EMPTY_WEIGHT_SHARDS=1"
            "VLLM_ROLLOUT_ASYNC_SCHEDULING=false"
        "VLLM_ROLLOUT_ENABLE_PREFIX_CACHING=true"
        "VLLM_ROLLOUT_ENABLE_CHUNKED_PREFILL=true"
        "VLLM_ASCEND_USE_FILTERED_CUSTOM_OPP=1"
        "VLLM_ASCEND_FILTERED_CUSTOM_OPP_PATH=$FILTERED_OPP"
        "VLLM_ASCEND_FILTERED_CUSTOM_OPP_BUNDLE_SHA256=$FILTERED_OPP_BUNDLE_SHA256"
        "VLLM_ASCEND_FORCE_ALLTOALL_MOE=0"
        "VLLM_ASCEND_MC2_MIN_EP_SIZE=2"
        "PLANNER_TOKENIZER_PATH=/data/Qwen3-30B-A3B"
        "KV_ADMISSION_REQUIRE_PROMPT_TOKENS=1"
        "KV_ADMISSION_BLOCK_SIZE_TOKENS=$KV_BLOCK_SIZE"
        "KV_ADMISSION_RESERVE_TOKENS=$KV_ADMISSION_RESERVE_TOKENS"
        "VLLM_ASCEND_ENABLE_TOPK_TOPP_OPTIMIZATION=1"
        "VLLM_ENABLE_MC2=1"
        "VERL_SIDECAR_ENABLE=0"
        "PYTORCH_NPU_ALLOC_CONF=garbage_collection_threshold:0.6,max_split_size_mb:128"
        "VERL_HCCL_IF_BASE_PORT_START=12000"
        "VERL_MASTER_PORT_START=28416"
        "RAY_TMPDIR=$ray_tmp"
        "XDG_CACHE_HOME=$cache_dir/xdg"
        "HF_HOME=$cache_dir/hf"
        "TRITON_CACHE_DIR=$cache_dir/triton"
        "TORCHAIR_CACHE_HOME=$cache_dir/torchair"
        "ASCEND_CACHE_PATH=$cache_dir/ascend"
        "ASCEND_WORK_PATH=$cache_dir/work"
    )
    local -a mode_hydra_args=()
    if [[ "$EXECUTION_MODE" == graph ]]; then
        run_env+=(
            "ADAFLOOR_ACLGRAPH_MODE=FULL_DECODE_ONLY"
            "VLLM_ROLLOUT_TASK_QUEUE_ENABLE=1"
            "TASK_QUEUE_ENABLE=1"
            "ADAFLOOR_GRAPH_CAPTURE_SIZES=$CAPTURE_SIZES"
            "ROLLOUT_ENFORCE_EAGER=False"
            "VLLM_ASCEND_ELASTIC_ACLGRAPH=1"
            "VLLM_ASCEND_ELASTIC_ACLGRAPH_CAPTURE_ATTENTION=1"
            "VLLM_ASCEND_ELASTIC_ACLGRAPH_CAPTURE_MOE=1"
            # Select dummy-run shape ordering explicitly. This does not
            # disable the engine's active-rank collective.
            "VLLM_ASCEND_ZIYI_GRAPH_DP_SYNC=${VLLM_ASCEND_ZIYI_GRAPH_DP_SYNC:-0}"
            "VLLM_ASCEND_ALLOW_LAZY_ACLGRAPH_CAPTURE=0"
            "VLLM_ASCEND_ELASTIC_ACLGRAPH_EXTENSION=$(readlink -f "$EXTENSION")"
            "VLLM_ASCEND_ELASTIC_ACLGRAPH_EXTENSION_SHA256=$(sha256_file "$EXTENSION")"
            # The dynamic training budget must fit a complete 17408-token
            # sequence. Recompute all layers (at most 13 per PP stage) to
            # reduce retained activations, rather than setting an invalid
            # token budget below the sequence length.
            "ACTOR_PPO_MAX_TOKEN_LEN_PER_GPU=${ACTOR_PPO_MAX_TOKEN_LEN_PER_GPU:-20480}"
            "ACTOR_RECOMPUTE_NUM_LAYERS=${ACTOR_RECOMPUTE_NUM_LAYERS:-13}"
            "VLLM_ASCEND_USE_TOPK_TOPP_CUSTOM=${VLLM_ASCEND_USE_TOPK_TOPP_CUSTOM:-0}"
            "VLLM_ROLLOUT_ASYNC_SCHEDULING=${VLLM_ROLLOUT_ASYNC_SCHEDULING:-false}"
            "VLLM_ASCEND_RUNTIME_PROGRESS_INTERVAL=${VLLM_ASCEND_RUNTIME_PROGRESS_INTERVAL:-256}"
            "VLLM_ASCEND_ACLGRAPH_BUCKET_COUNT=$ACLGRAPH_BUCKET_COUNT"
            "VERL_FULL_DECODE_MAX_UPDATE_ACTOR_SECONDS="
            "VERL_FULL_DECODE_MAX_UPDATE_ACTOR_MS_PER_TOKEN=0.08"
        )
        if [[ "${VLLM_ROLLOUT_NATIVE_SLEEP_GRAPH_LIFECYCLE:-0}" == "1" \
              && "${VLLM_ROLLOUT_NATIVE_SLEEP_GRAPH_RETAIN:-0}" != "1" ]]; then
            run_env+=(
                # Restore the large KV mapping before CaMem weight shards.
                # Weight-first remap reproducibly OOMs on some ranks.
                "VLLM_ROLLOUT_NATIVE_SLEEP_GRAPH_KV_FIRST=1"
                "VLLM_ROLLOUT_INVALIDATE_ACLGRAPH_AFTER_WEIGHT_UPDATE=1"
                "VLLM_ROLLOUT_RECAPTURE_ACLGRAPH_AFTER_WEIGHT_UPDATE=1"
                "VLLM_ROLLOUT_REUSE_ACLGRAPH_AFTER_WEIGHT_UPDATE=0"
            )
        else
            run_env+=(
                "VLLM_ROLLOUT_INVALIDATE_ACLGRAPH_AFTER_WEIGHT_UPDATE=0"
                "VLLM_ROLLOUT_RECAPTURE_ACLGRAPH_AFTER_WEIGHT_UPDATE=0"
                "VLLM_ROLLOUT_REUSE_ACLGRAPH_AFTER_WEIGHT_UPDATE=1"
            )
        fi
        mode_hydra_args+=(
            actor_rollout_ref.rollout.cudagraph_mode=FULL_DECODE_ONLY
            "actor_rollout_ref.rollout.cudagraph_capture_sizes=$CAPTURE_SIZES"
            "actor_rollout_ref.rollout.async_scheduling=${VLLM_ROLLOUT_ASYNC_SCHEDULING:-false}"
        )
        if [[ -n "$base" ]]; then
            run_env+=("FAIR_ADAFLOOR_TARGET=$GRAPH_WRAPPER" "ADAFLOOR_GRAPH_BASE_RUNNER=$base")
        fi
    else
        run_env+=(
            "ROLLOUT_ENFORCE_EAGER=True"
            "VLLM_ASCEND_ELASTIC_ACLGRAPH=0"
            "VLLM_ASCEND_ELASTIC_ACLGRAPH_CAPTURE_ATTENTION=0"
            "VLLM_ASCEND_ELASTIC_ACLGRAPH_CAPTURE_MOE=0"
            "VLLM_ROLLOUT_REUSE_ACLGRAPH_AFTER_WEIGHT_UPDATE=0"
        )
        if [[ -n "$base" ]]; then
            run_env+=("FAIR_ADAFLOOR_TARGET=$base")
        fi
    fi
    run_env+=("${KV_ENV[@]}")

    echo "[graph main] launch=$launch_index seed=$seed policy=$policy variant=$variant root=$root"
    set +e
    env -u FLOOR_KV_CAPS "${run_env[@]}" \
        "$FAIR_RUNNER" "$variant" \
        actor_rollout_ref.rollout.seed="$seed" \
        actor_rollout_ref.actor.optim.lr=1e-6 \
        "${mode_hydra_args[@]}" \
        ++actor_rollout_ref.actor.megatron.override_transformer_config.moe_shared_expert_overlap=False
    arm_rc=$?
    set -e
    if (( arm_rc != 0 )); then
        cleanup_transient_dirs "$seed" "$policy" "$cache_dir" "$ray_tmp"
        echo "[graph main] arm failed seed=$seed policy=$policy rc=$arm_rc; validated checkpoints were not deleted" >&2
        return "$arm_rc"
    fi

    if (( smoke_steps == 5 )); then
        verify_arm "$seed" "$policy" "$root"
        cleanup_transient_dirs "$seed" "$policy" "$cache_dir" "$ray_tmp"
        seal_contracts
        record_complete "$launch_index" "$seed" "$policy" "$root"
        record_disk_cleanup "$seed" "$policy"
    else
        verify_smoke_arm "$policy" "$root"
        cleanup_transient_dirs "$seed" "$policy" "$cache_dir" "$ray_tmp"
    fi
}

write_analysis_manifest() {
    local candidate seed policy root epoch_dir epoch
    candidate=$(mktemp "$RESULT_ROOT/.analysis_manifest.XXXXXX")
    printf 'trial_seed\tpolicy\tepoch\tepoch_dir\n' > "$candidate"
    for seed in "${SEEDS[@]}"; do
        for policy in "${POLICIES[@]}"; do
            root=$(run_root "$seed" "$policy")
            verify_arm "$seed" "$policy" "$root"
            while IFS= read -r epoch_dir; do
                epoch=$(basename "$epoch_dir" | sed -E 's/^epoch_0*([12])_.*/\1/')
                printf '%s\t%s\t%s\t%s\n' "$seed" "$policy" "$epoch" "$epoch_dir" >> "$candidate"
            done < <(epoch_dirs_for_arm "$root")
        done
    done
    if [[ -e "$ANALYSIS_MANIFEST" ]]; then
        cmp -s "$candidate" "$ANALYSIS_MANIFEST" || {
            rm -f "$candidate"; echo "analysis manifest changed" >&2; return 1;
        }
        rm -f "$candidate"
    else
        mv "$candidate" "$ANALYSIS_MANIFEST"
    fi
}

run_final_analysis() {
    write_analysis_manifest
    python3 "$SUMMARIZER" --manifest "$ANALYSIS_MANIFEST" --output-dir "$ANALYSIS_DIR"
        printf 'status=PASS\ncompleted_arms=%s\ncompleted_epochs=%s\nsummary_json_sha256=%s\n' \
        "$PLANNED_ARMS" "$PLANNED_EPOCHS" \
        "$(sha256_file "$ANALYSIS_DIR/summary.json")" > "$RESULT_ROOT/GRAPH_MAIN_PASS.txt"
    echo "[graph main] PASS summary=$ANALYSIS_DIR/summary.md"
}

require_inputs
FILTERED_OPP=$(realpath "$FILTERED_OPP")
FILTERED_OPP_BUNDLE_SHA256=$(find "$FILTERED_OPP" -type f -print0 \
    | sort -z | xargs -0 sha256sum | sha256sum | awk '{print $1}')
unset VLLM_ROLLOUT_SLEEP_LEVEL MEGATRON_CI_DISABLE_EXPANDABLE_SEGMENTS FLOOR_KV_CAPS
seal_contracts
initialize_manifest

if [[ "$ACTION" == dry-run ]]; then
    echo "root=$RESULT_ROOT"
    echo "common_epoch0=$COMMON_EPOCH0_ROOT"
    echo "seeds=${SEEDS[*]} policies=${POLICIES[*]} epochs=$EPOCHS_CSV actor_frozen=false"
    echo "execution_mode=$EXECUTION_MODE tq=1"
    if [[ "$EXECUTION_MODE" == graph ]]; then
        echo "graph=FULL_DECODE_ONLY bootstrap_capture_sizes=$CAPTURE_SIZES adaptive_bucket_count=$ACLGRAPH_BUCKET_COUNT attention=true moe=true"
    else
        echo "graph=disabled enforce_eager=true prefix_caching=true"
    fi
    echo "graph_reserve=${GRAPH_RESERVE_BYTES}B=${GRAPH_RESERVE_BLOCKS}blocks=${GRAPH_RESERVE_TOKENS}tokens/floor"
    echo "caps full16=$ACTIVE_FULL16 f2=$ACTIVE_F2 f4=$ACTIVE_F4 f8_natural=$ACTIVE_F8_NATURAL f8_f4=$ACTIVE_F8_F4"
    exit 0
fi

if [[ "$ACTION" == verify ]]; then
    seal_contracts
    run_final_analysis
    exit 0
fi

if [[ "$ACTION" == smoke ]]; then
    if [[ "$POLICY_PROFILE" == core3 ]]; then
        smoke_index=0
        smoke_selected=0
        for smoke_policy in vanilla natural_f2 planned_f4; do
            smoke_index=$((smoke_index + 1))
            if [[ -n "$ONLY_ARM" && "$ONLY_ARM" != "0:$smoke_policy" ]]; then
                continue
            fi
            run_arm "$smoke_index" 0 "$smoke_policy" 1
            smoke_selected=$((smoke_selected + 1))
        done
        if [[ -n "$ONLY_ARM" && "$smoke_selected" != 1 ]]; then
            echo "failed to select exactly one smoke arm: $ONLY_ARM" >&2
            exit 2
        fi
    else
        run_arm 1 0 vanilla 1
    fi
    echo "[graph main smoke] complete (non-authoritative) root=$RESULT_ROOT"
    exit 0
fi

launch_index=0
selected_arms=0
for seed in "${SEEDS[@]}"; do
    for policy in "${POLICIES[@]}"; do
        launch_index=$((launch_index + 1))
        if [[ -n "$ONLY_ARM" && "$ONLY_ARM" != "$seed:$policy" ]]; then
            continue
        fi
        run_arm "$launch_index" "$seed" "$policy" 5
        selected_arms=$((selected_arms + 1))
    done
done
if [[ -n "$ONLY_ARM" ]]; then
    (( selected_arms == 1 )) || {
        echo "failed to select exactly one formal arm: $ONLY_ARM" >&2
        exit 2
    }
    echo "[graph main] paired schedule arm complete mode=$EXECUTION_MODE arm=$ONLY_ARM"
    exit 0
fi
run_final_analysis
