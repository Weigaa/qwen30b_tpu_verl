#!/usr/bin/env bash
set -euo pipefail

# Protect post-run validation from edits to this long-running launcher.
if [[ "${ADAFLOOR_FULL_DECODE_RUNNER_SNAPSHOT_ACTIVE:-0}" != "1" ]]; then
    source_path=$(realpath "${BASH_SOURCE[0]}")
    snapshot=$(mktemp "${source_path}.run-snapshot.XXXXXX")
    cp -- "$source_path" "$snapshot"
    chmod 700 "$snapshot"
    set +e
    ADAFLOOR_FULL_DECODE_RUNNER_SNAPSHOT_ACTIVE=1 "$snapshot" "$@"
    rc=$?
    set -e
    rm -f -- "$snapshot"
    exit "$rc"
fi

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR"

ACTION=${1:-dry-run}
SELECTION=${2:-all}
case "$ACTION" in
    dry-run|run|verify) ;;
    -h|--help)
        cat <<'EOF'
Usage:
  ./run_qwen3_adafloor_full_decode_fia_v014opt_tq1.sh dry-run [planned|natural|all]
  ./run_qwen3_adafloor_full_decode_fia_v014opt_tq1.sh run [planned|natural|all]
  ./run_qwen3_adafloor_full_decode_fia_v014opt_tq1.sh verify

Runs one frozen-actor, five-step representative epoch for AdaFloor Planned-F4
and/or Natural-F2 from the optimized [16,32] FULL_DECODE_ONLY graph Vanilla
epoch0. Prefill remains eager. Decode includes KV write, FIA Attention, dense
layers, and elastic MoE inside native ACLGraph. Each topology captures only
the [16,32] shape set on its active ranks.

No NPU work is started by dry-run or verify.
EOF
        exit 0
        ;;
    *) echo "unknown action: $ACTION" >&2; exit 2 ;;
esac
if [[ "$SELECTION" != "planned" && "$SELECTION" != "natural" \
        && "$SELECTION" != "all" ]]; then
    echo "selection must be planned, natural, or all: $SELECTION" >&2
    exit 2
fi
if [[ "$ACTION" == verify && "$SELECTION" != all ]]; then
    echo "verify does not accept a policy selection" >&2
    exit 2
fi

COMMON_EPOCH0_ROOT=${COMMON_EPOCH0_ROOT:-/workspace/adafloor_graph_results/qwen3_vanilla_epoch0_full_decode_fia_v014opt_tq1_buckets16_32_seed0/common_epoch0_full_decode_fia_v014opt_tq1_buckets16_32_seed0}
RESULT_ROOT=${ADAFLOOR_FULL_DECODE_RESULT_ROOT:-/workspace/adafloor_graph_results/qwen3_adafloor_full_decode_fia_v014opt_tq1_buckets16_32_seed0}
CAPTURE_SIZES='[16,32]'
SEED=0
STEPS=5
RESPONSES_PER_STEP=512
FULL16_KV_TOKENS=380800
GRAPH_WRAPPER="$SCRIPT_DIR/run_mode1_local_length_sorted_e2e_adaptive_floor4_aclgraph.sh"
FAIR_RUNNER="$SCRIPT_DIR/run_paper_fair_epoch1_2_from_common_epoch0.sh"
VERIFY_TOOL="$SCRIPT_DIR/tools/verify_qwen3_adafloor_full_decode.py"
EXTENSION=${VLLM_ASCEND_ELASTIC_ACLGRAPH_EXTENSION:-/workspace/vllm-ascend/vllm_ascend/vllm_ascend_C.cpython-311-aarch64-linux-gnu.so}
FILTERED_OPP=${VLLM_ASCEND_FILTERED_CUSTOM_OPP_PATH:-$SCRIPT_DIR/../qwen3/vllm_ascend/_cann_ops_custom_moe_filtered/vendors/vllm-ascend}
CODE_CONTRACT="$RESULT_ROOT/code_sha256.txt"
PROTOCOL_CONTRACT="$RESULT_ROOT/protocol.env"
RUN_MANIFEST="$RESULT_ROOT/run_manifest.tsv"

base_runner() {
    case "$1" in
        planned) printf '%s\n' "$SCRIPT_DIR/run_mode1_dynamic_length_aware_adaptive_floor4_planned_full3.sh" ;;
        natural) printf '%s\n' "$SCRIPT_DIR/run_mode1_dynamic_length_aware_adaptive_floor2_natural_tailguard_reuse_epoch0_2epoch.sh" ;;
        *) return 1 ;;
    esac
}

variant_name() {
    case "$1" in
        planned) printf 'adafloor_p_f4\n' ;;
        natural) printf 'adafloor_n_f2\n' ;;
        *) return 1 ;;
    esac
}

run_name() {
    printf 'qwen3_adafloor_%s_full_decode_fia_tq1_buckets16_32_seed0\n' "$1"
}

sha256_file() {
    sha256sum "$1" | awk '{print $1}'
}

require_inputs() {
    local path
    for path in \
        "$COMMON_EPOCH0_ROOT/DO_NOT_DELETE_COMMON_EPOCH0_CHECKPOINT" \
        "$COMMON_EPOCH0_ROOT/reuse.env" \
        "$COMMON_EPOCH0_ROOT/epoch_000_mode0_probe/offline_planning_history.json" \
        "$COMMON_EPOCH0_ROOT/epoch_000_mode0_probe/checkpoints/qwen3moe_for_eagle3/global_step_5" \
        "$GRAPH_WRAPPER" "$FAIR_RUNNER" "$VERIFY_TOOL" "$EXTENSION" \
        "$FILTERED_OPP"; do
        [[ -e "$path" ]] || { echo "missing required input: $path" >&2; exit 2; }
    done
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
        return 0
    fi
    mv -- "$candidate" "$destination"
}

seal_contracts() {
    mkdir -p "$RESULT_ROOT"
    local code_candidate protocol_candidate created
    code_candidate=$(mktemp "$RESULT_ROOT/.code.XXXXXX")
    protocol_candidate=$(mktemp "$RESULT_ROOT/.protocol.XXXXXX")
    local code_paths=(
        "$SCRIPT_DIR/run_qwen3_adafloor_full_decode_fia_v014opt_tq1.sh"
        "$VERIFY_TOOL"
        "$FAIR_RUNNER"
        "$GRAPH_WRAPPER"
        "$SCRIPT_DIR/run_mode1_dynamic_length_aware_adaptive_floor4_planned_full3.sh"
        "$SCRIPT_DIR/run_mode1_dynamic_length_aware_adaptive_floor2_natural_tailguard_reuse_epoch0_2epoch.sh"
        "$SCRIPT_DIR/run_mode1_local_length_sorted_e2e_adaptive_floor4.sh"
        "$SCRIPT_DIR/run_mode1_local_length_sorted_e2e_adaptive_floor2.sh"
        "$SCRIPT_DIR/tools/verify_qwen3_adafloor_graph_eager_full_pair.py"
        "$SCRIPT_DIR/tools/build_mode1_length_sorted_e2e_plan.py"
        "$SCRIPT_DIR/verl/experimental/dataset/shrink_aware_assignment.py"
        "$SCRIPT_DIR/verl/trainer/ppo/ray_trainer.py"
        "$SCRIPT_DIR/verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py"
        "$SCRIPT_DIR/vllm_ascend/platform.py"
        "$SCRIPT_DIR/vllm_ascend/attention/attention_v1.py"
        "$SCRIPT_DIR/vllm_ascend/models/qwen3_moe.py"
        "$SCRIPT_DIR/vllm_ascend/ops/fused_moe.py"
        "$SCRIPT_DIR/vllm_ascend/compilation/acl_graph.py"
        "$SCRIPT_DIR/vllm_ascend/worker/model_runner_v1.py"
        "$SCRIPT_DIR/vllm_ascend/worker/worker_v1.py"
    )
    sha256sum "${code_paths[@]}" > "$code_candidate"
    if [[ -f "$PROTOCOL_CONTRACT" && ! -L "$PROTOCOL_CONTRACT" ]]; then
        created=$(sed -n 's/^created_at_utc=//p' "$PROTOCOL_CONTRACT")
    else
        created=$(date -u +%Y-%m-%dT%H:%M:%SZ)
    fi
    {
        printf 'schema_version=1\n'
        printf 'created_at_utc=%s\n' "$created"
        printf 'experiment=qwen3_adafloor_full_decode_fia_v014opt_tq1\n'
        printf 'common_epoch0_root=%s\n' "$COMMON_EPOCH0_ROOT"
        printf 'common_epoch0_mode=graph_vanilla_buckets16_32\n'
        printf 'policies=planned,natural\n'
        printf 'planned_variant=adafloor_p_f4\n'
        printf 'natural_variant=adafloor_n_f2\n'
        printf 'actor_frozen=true\n'
        printf 'seed=%s\n' "$SEED"
        printf 'steps=%s\n' "$STEPS"
        printf 'prompts_per_step=32\n'
        printf 'rollout_n=16\n'
        printf 'responses_per_step=%s\n' "$RESPONSES_PER_STEP"
        printf 'max_response_length=16384\n'
        printf 'max_num_seqs=32\n'
        printf 'full16_kv_tokens=%s\n' "$FULL16_KV_TOKENS"
        printf 'planned_floor4_kv_tokens=280576\n'
        printf 'planned_floor8_kv_tokens=377344\n'
        printf 'natural_floor2_kv_tokens=98304\n'
        printf 'natural_floor4_kv_tokens=280576\n'
        printf 'natural_floor8_kv_tokens=315648\n'
        printf 'graph_mode=FULL_DECODE_ONLY\n'
        printf 'capture_sizes=%s\n' "$CAPTURE_SIZES"
        printf 'cudagraph_copy_inputs=false\n'
        printf 'attention_backend=fia_max_workspace\n'
        printf 'attention_graph=true\n'
        printf 'moe_graph=true\n'
        printf 'prefill_execution=eager\n'
        printf 'task_queue_enable=1\n'
        printf 'max_update_actor_ms_per_token=0.08\n'
        printf 'native_sleep_mode=0\n'
        printf 'async_scheduling=false\n'
        printf 'torchair=false\n'
        printf 'graphex=false\n'
        printf 'sidecar=false\n'
        printf 'tail_guard=enabled\n'
        printf 'filtered_opp_bundle_sha256=%s\n' "$FILTERED_OPP_BUNDLE_SHA256"
    } > "$protocol_candidate"
    install_or_verify_immutable "$code_candidate" "$CODE_CONTRACT" 'code contract'
    install_or_verify_immutable "$protocol_candidate" "$PROTOCOL_CONTRACT" 'protocol contract'
}

manifest_complete() {
    local policy=$1
    [[ -f "$RUN_MANIFEST" ]] && awk -F '\t' -v p="$policy" \
        'NR > 1 && $1 == p && $2 == "complete" {found=1} END {exit !found}' \
        "$RUN_MANIFEST"
}

record_complete() {
    local policy=$1 manifest=$2
    if [[ ! -f "$RUN_MANIFEST" ]]; then
        printf 'policy\tstatus\tarm_manifest\n' > "$RUN_MANIFEST"
    fi
    if manifest_complete "$policy"; then
        return 0
    fi
    if awk -F '\t' -v p="$policy" \
        'NR > 1 && $1 == p {found=1} END {exit !found}' "$RUN_MANIFEST"; then
        echo "non-complete or duplicate manifest row for $policy" >&2
        return 1
    fi
    printf '%s\tcomplete\t%s\n' "$policy" "$manifest" >> "$RUN_MANIFEST"
}

write_arm_manifest() {
    local policy=$1 run_dir=$2 arm_dir=$3
    local manifest="$arm_dir/arm_manifest.json"
    mapfile -t epoch_dirs < <(find "$run_dir" -mindepth 1 -maxdepth 1 -type d -name 'epoch_001_*' -print)
    (( ${#epoch_dirs[@]} == 1 )) || { echo "expected one epoch_001 directory in $run_dir" >&2; return 1; }
    local epoch_dir=${epoch_dirs[0]}
    local plan_file="$epoch_dir/oracle/length_sorted_rank_plan.json"
    local log_file
    log_file=$(find "$epoch_dir/logs" -maxdepth 1 -type f -name '*.txt' -printf '%T@ %p\n' | sort -n | tail -1 | cut -d' ' -f2-)
    [[ -f "$plan_file" && -f "$log_file" \
        && -f "$run_dir/CHECKPOINTS_REMOVED_AFTER_VALIDATION.txt" ]] || {
        echo "completed arm artifacts are incomplete: $run_dir" >&2
        return 1
    }
    local tmp
    tmp=$(mktemp "$arm_dir/.manifest.XXXXXX")
    python3 - "$tmp" "$policy" "$run_dir" "$epoch_dir" "$plan_file" "$log_file" \
        "$COMMON_EPOCH0_ROOT" "$CAPTURE_SIZES" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

out, policy, run_dir, epoch_dir, plan, log, common, sizes = sys.argv[1:]
def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
payload = {
    "schema_version": 1,
    "experiment": "qwen3_adafloor_full_decode_fia_v014opt_tq1",
    "policy": policy,
    "variant": "adafloor_p_f4" if policy == "planned" else "adafloor_n_f2",
    "run_dir": str(Path(run_dir).resolve()),
    "epoch_dir": str(Path(epoch_dir).resolve()),
    "plan_file": str(Path(plan).resolve()),
    "plan_sha256": digest(plan),
    "log_file": str(Path(log).resolve()),
    "common_epoch0_root": str(Path(common).resolve()),
    "seed": 0,
    "steps": 5,
    "responses_per_step": 512,
    "actor_frozen": True,
    "graph_mode": "FULL_DECODE_ONLY",
    "capture_sizes": json.loads(sizes),
    "attention_graph": True,
    "moe_graph": True,
    "task_queue_enable": 1,
    "tail_guard_enabled": True,
    "checkpoint_cleanup_marker": str(Path(run_dir, "CHECKPOINTS_REMOVED_AFTER_VALIDATION.txt").resolve()),
}
Path(out).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY
    mv -- "$tmp" "$manifest"
    printf '%s\n' "$manifest"
}

run_policy() {
    local policy=$1
    local variant base arm_dir name run_dir cache_dir ray_tmp manifest
    local floor2_kv floor4_kv floor8_kv
    variant=$(variant_name "$policy")
    base=$(base_runner "$policy")
    arm_dir="$RESULT_ROOT/$policy"
    name=$(run_name "$policy")
    run_dir="$arm_dir/$name"
    cache_dir="$arm_dir/cache"
    ray_tmp="/tmp/ag_fd_${policy:0:1}_${$}"
    if [[ "$policy" == natural ]]; then
        # The graph-safe F2 capacity was established by the Natural-F2
        # correctness gate after aligning profile tokens and MC2 allocation.
        # A future eager control must use this same capacity.
        floor2_kv=98304
        floor4_kv=280576
        floor8_kv=315648
    else
        floor2_kv=98304
        floor4_kv=280576
        floor8_kv=377344
    fi

    if manifest_complete "$policy"; then
        echo "[AdaFloor FULL graph] already complete: $policy"
        return 0
    fi
    if [[ -e "$run_dir" ]]; then
        if [[ -f "$run_dir/CHECKPOINTS_REMOVED_AFTER_VALIDATION.txt" ]]; then
            manifest=$(write_arm_manifest "$policy" "$run_dir" "$arm_dir")
            record_complete "$policy" "$manifest"
            return 0
        fi
        echo "incomplete arm exists; use a fresh RESULT_ROOT: $run_dir" >&2
        return 1
    fi
    mkdir -p "$arm_dir" "$ray_tmp" "$cache_dir"/{xdg,hf,triton,torchair,ascend,work}
    echo "[AdaFloor FULL graph] starting policy=$policy output=$run_dir"
    seal_contracts
    env \
        COMMON_EPOCH0_ROOT="$COMMON_EPOCH0_ROOT" \
        FAIR_OUTPUT_ROOT="$arm_dir" \
        FAIR_RUN_NAME="$name" \
        DYNAMIC_RUN_NAME="$name" \
        FAIR_START_EPOCH=1 FAIR_TOTAL_EPOCHS=2 FAIR_TRAIN_STEPS="$STEPS" \
        FAIR_TRAIN_BATCH_SIZE=32 FAIR_ROLLOUT_N=16 \
        FAIR_MAX_RESPONSE_LENGTH=16384 FAIR_PROMPTS_PER_EPOCH=160 \
        FAIR_FREEZE_ACTOR=1 FAIR_KEEP_COMPLETED_CHECKPOINTS=0 \
        VERL_PAIRED_REQUEST_SAMPLING_SEEDS=1 \
        FAIR_ADAFLOOR_TARGET="$GRAPH_WRAPPER" \
        ADAFLOOR_GRAPH_BASE_RUNNER="$base" \
        ADAFLOOR_ACLGRAPH_MODE=FULL_DECODE_ONLY \
        ADAFLOOR_GRAPH_CAPTURE_SIZES="$CAPTURE_SIZES" \
        VLLM_ENABLE_GRAPH_MODE=0 TASK_QUEUE_ENABLE=1 ROLLOUT_ENFORCE_EAGER=False \
        VLLM_ASCEND_ELASTIC_ACLGRAPH=1 \
        VLLM_ASCEND_ELASTIC_ACLGRAPH_CAPTURE_ATTENTION=1 \
        VLLM_ASCEND_ELASTIC_ACLGRAPH_CAPTURE_MOE=1 \
        VLLM_ASCEND_ELASTIC_ACLGRAPH_EXTENSION="$(readlink -f "$EXTENSION")" \
        VLLM_ASCEND_ELASTIC_ACLGRAPH_EXTENSION_SHA256="$(sha256_file "$EXTENSION")" \
        VLLM_ASCEND_MODE1_PARITY_MAX_KV_TOKENS="$FULL16_KV_TOKENS" \
        VLLM_ASCEND_MODE1_PARITY_MAX_KV_TOKENS_FLOOR2="$floor2_kv" \
        VLLM_ASCEND_MODE1_PARITY_MAX_KV_TOKENS_FLOOR4="$floor4_kv" \
        VLLM_ASCEND_MODE1_PARITY_MAX_KV_TOKENS_FLOOR8="$floor8_kv" \
        VLLM_ROLLOUT_NATIVE_SLEEP_MODE=0 \
        VLLM_ROLLOUT_REUSE_ACLGRAPH_AFTER_WEIGHT_UPDATE=1 \
        VLLM_ROLLOUT_CAMEM_WEIGHT_RELOAD=0 \
        VLLM_ROLLOUT_FILTER_EMPTY_WEIGHT_SHARDS=1 \
        VLLM_ROLLOUT_ASYNC_SCHEDULING=false \
        VLLM_ROLLOUT_ENABLE_PREFIX_CACHING=true \
        VLLM_ROLLOUT_ENABLE_CHUNKED_PREFILL=true \
        VLLM_ASCEND_USE_FILTERED_CUSTOM_OPP=1 \
        VLLM_ASCEND_FILTERED_CUSTOM_OPP_PATH="$FILTERED_OPP" \
        VLLM_ASCEND_FILTERED_CUSTOM_OPP_BUNDLE_SHA256="$FILTERED_OPP_BUNDLE_SHA256" \
        VLLM_ASCEND_FORCE_ALLTOALL_MOE=0 VLLM_ASCEND_MC2_MIN_EP_SIZE=2 \
        VLLM_ASCEND_ENABLE_TOPK_TOPP_OPTIMIZATION=1 VLLM_ENABLE_MC2=1 \
        VERL_SIDECAR_ENABLE=0 VERL_FULL_DECODE_MAX_UPDATE_ACTOR_SECONDS= \
        VERL_FULL_DECODE_MAX_UPDATE_ACTOR_MS_PER_TOKEN=0.08 \
        PYTORCH_NPU_ALLOC_CONF=expandable_segments:True \
        VERL_HCCL_IF_BASE_PORT_START=12000 VERL_MASTER_PORT_START=28416 \
        RAY_TMPDIR="$ray_tmp" XDG_CACHE_HOME="$cache_dir/xdg" \
        HF_HOME="$cache_dir/hf" TRITON_CACHE_DIR="$cache_dir/triton" \
        TORCHAIR_CACHE_HOME="$cache_dir/torchair" \
        ASCEND_CACHE_PATH="$cache_dir/ascend" ASCEND_WORK_PATH="$cache_dir/work" \
        "$FAIR_RUNNER" "$variant" \
        actor_rollout_ref.rollout.seed="$SEED" \
        actor_rollout_ref.rollout.cudagraph_mode=FULL_DECODE_ONLY \
        "actor_rollout_ref.rollout.cudagraph_capture_sizes=$CAPTURE_SIZES" \
        actor_rollout_ref.actor.megatron.override_transformer_config.moe_shared_expert_overlap=False
    seal_contracts
    manifest=$(write_arm_manifest "$policy" "$run_dir" "$arm_dir")
    record_complete "$policy" "$manifest"
    python3 "$VERIFY_TOOL" --root "$RESULT_ROOT" --allow-incomplete
}

require_inputs
FILTERED_OPP=$(realpath "$FILTERED_OPP")
FILTERED_OPP_BUNDLE_SHA256=$(find "$FILTERED_OPP" -type f -print0 \
    | sort -z | xargs -0 sha256sum | sha256sum | awk '{print $1}')
unset VLLM_ROLLOUT_SLEEP_LEVEL MEGATRON_CI_DISABLE_EXPANDABLE_SEGMENTS
seal_contracts

if [[ "$ACTION" == dry-run ]]; then
    echo "root=$RESULT_ROOT"
    echo "common_epoch0=$COMMON_EPOCH0_ROOT"
    echo "graph=FULL_DECODE_ONLY capture_sizes=$CAPTURE_SIZES attention=true moe=true tq=1"
    for policy in planned natural; do
        if [[ "$SELECTION" == all || "$SELECTION" == "$policy" ]]; then
            echo "arm=$policy variant=$(variant_name "$policy") base=$(base_runner "$policy") run_name=$(run_name "$policy")"
        fi
    done
    exit 0
fi
if [[ "$ACTION" == verify ]]; then
    exec python3 "$VERIFY_TOOL" --root "$RESULT_ROOT"
fi

for policy in planned natural; do
    if [[ "$SELECTION" == all || "$SELECTION" == "$policy" ]]; then
        run_policy "$policy"
    fi
done
seal_contracts
if [[ "$SELECTION" == all ]]; then
    exec python3 "$VERIFY_TOOL" --root "$RESULT_ROOT"
fi
exec python3 "$VERIFY_TOOL" --root "$RESULT_ROOT" --allow-incomplete
