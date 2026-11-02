#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR"

ACTION=${1:-plan}
case "$ACTION" in
    plan|run|summarize) ;;
    -h|--help)
        cat <<'EOF'
Usage:
  ./run_qwen3_aclgraph_step_bucket_profile.sh plan
  ./run_qwen3_aclgraph_step_bucket_profile.sh run
  ./run_qwen3_aclgraph_step_bucket_profile.sh summarize

Profiles one frozen-actor Vanilla Full16 step using exact minimum-padding
FULL_DECODE_ONLY capture sets selected from epoch0 predicted lengths. The
default source is step 4, and the measured graph counts are 1..8, 12, and 32.
EOF
        exit 0
        ;;
    *) echo "unknown action: $ACTION" >&2; exit 2 ;;
esac

ROOT=${ACLGRAPH_BUCKET_PROFILE_ROOT:-/workspace/adafloor_graph_results/qwen3_aclgraph_bucket_count_step4_seed0_v7_forced_token_fixed_work}
COMMON_EPOCH0_ROOT=${COMMON_EPOCH0_ROOT:-/data/adafloor_shared_state/common_epoch0_probe_gpu09_kv380800_permanent}
SOURCE_HISTORY=${ACLGRAPH_BUCKET_PROFILE_HISTORY:-$COMMON_EPOCH0_ROOT/epoch_000_mode0_probe/offline_planning_history.json}
SOURCE_TRAIN=${ACLGRAPH_BUCKET_PROFILE_TRAIN_FILE:-/data/deepscaler/train.parquet}
SOURCE_STEP=${ACLGRAPH_BUCKET_PROFILE_SOURCE_STEP:-4}
COUNTS=${ACLGRAPH_BUCKET_PROFILE_COUNTS:-1,2,3,4,5,6,7,8,12,32}
# Interleave small and large sets so thermal/time drift is not monotonic in k.
LAUNCH_ORDER=${ACLGRAPH_BUCKET_PROFILE_LAUNCH_ORDER:-1,32,2,12,3,8,4,7,5,6}
SELECTOR="$SCRIPT_DIR/analysis_eval/select_aclgraph_step_buckets.py"
SUMMARIZER="$SCRIPT_DIR/analysis_eval/summarize_aclgraph_step_bucket_profile.py"
TRACE_BUILDER="$SCRIPT_DIR/analysis_eval/build_qwen3_aclgraph_profile_trace.py"
FAIR_RUNNER="$SCRIPT_DIR/run_paper_fair_epoch1_2_from_common_epoch0.sh"
SAMPLER="$SCRIPT_DIR/internal/sample_npu_util.py"
FILTERED_OPP=${VLLM_ASCEND_FILTERED_CUSTOM_OPP_PATH:-$SCRIPT_DIR/../qwen3/vllm_ascend/_cann_ops_custom_moe_filtered/vendors/vllm-ascend}
EXTENSION=${VLLM_ASCEND_ELASTIC_ACLGRAPH_EXTENSION:-/workspace/vllm-ascend/vllm_ascend/vllm_ascend_C.cpython-311-aarch64-linux-gnu.so}
PROFILE_PLAN="$ROOT/profile_plan.json"
PROFILE_TRAIN="$ROOT/source_step_${SOURCE_STEP}_train.parquet"
FIXED_WORK_SOURCE=${ACLGRAPH_BUCKET_PROFILE_FIXED_WORK_SOURCE:-/workspace/adafloor_graph_results/qwen3_aclgraph_bucket_count_step4_seed0_v2/arms/count_01/qwen3_aclgraph_bucket_profile_k01/epoch_001_mode0_vanilla/rollout_data/1.jsonl}
FIXED_WORK_TRACE="$ROOT/fixed_work_trace.json"
PROTOCOL="$ROOT/protocol.env"
CODE_CONTRACT="$ROOT/code_sha256.txt"
KV_BLOCK_SIZE=128
KV_BYTES_PER_TOKEN=98304
EAGER_FULL16_TOKENS=380800
KV_BLOCK_BYTES=$((KV_BLOCK_SIZE * KV_BYTES_PER_TOKEN))
RESERVE_MODE=${ACLGRAPH_BUCKET_PROFILE_RESERVE_MODE:-fixed_5p6g}
EXPLICIT_CAPTURE_SETS=${ACLGRAPH_BUCKET_PROFILE_EXPLICIT_CAPTURE_SETS:-}
EXTRA_CODE_CONTRACT=${ACLGRAPH_BUCKET_PROFILE_EXTRA_CODE_CONTRACT:-}
GRAPH_MEMORY_PROFILE=${ACLGRAPH_BUCKET_PROFILE_GRAPH_MEMORY_PROFILE:-$SCRIPT_DIR/analysis_eval/qwen3_aclgraph_step_bucket_profile_all32.json}
RESERVE_TOOL="$SCRIPT_DIR/analysis_eval/derive_aclgraph_reserve.py"
GRAPH_RESERVE_MINIMUM_BYTES=2684354560
GRAPH_RESERVE_QUANTUM_BYTES=536870912
FIXED_GRAPH_RESERVE_BYTES=$((56 * 1024 * 1024 * 1024 / 10))

require_file() {
    [[ -f "$1" ]] || { echo "missing required file: $1" >&2; exit 2; }
}

require_dir() {
    [[ -d "$1" ]] || { echo "missing required directory: $1" >&2; exit 2; }
}

for path in "$SOURCE_HISTORY" "$SOURCE_TRAIN" "$SELECTOR" "$SUMMARIZER" \
        "$TRACE_BUILDER" "$FIXED_WORK_SOURCE" \
        "$FAIR_RUNNER" "$SAMPLER" "$EXTENSION" "$GRAPH_MEMORY_PROFILE" \
        "$RESERVE_TOOL" \
        "$COMMON_EPOCH0_ROOT/DO_NOT_DELETE_COMMON_EPOCH0_CHECKPOINT" \
        "$COMMON_EPOCH0_ROOT/reuse.env"; do
    require_file "$path"
done
require_dir "$FILTERED_OPP"
require_dir "$COMMON_EPOCH0_ROOT/epoch_000_mode0_probe/checkpoints/qwen3moe_for_eagle3/global_step_5"

case "$RESERVE_MODE" in
    fixed_5p6g|profile_derived) ;;
    *) echo "ACLGRAPH_BUCKET_PROFILE_RESERVE_MODE must be fixed_5p6g or profile_derived" >&2; exit 2 ;;
esac

FILTERED_OPP=$(realpath "$FILTERED_OPP")
EXTENSION=$(realpath "$EXTENSION")
FILTERED_OPP_BUNDLE_SHA256=$(find "$FILTERED_OPP" -type f -print0 \
    | sort -z | xargs -0 sha256sum | sha256sum | awk '{print $1}')
EXTENSION_SHA256=$(sha256sum "$EXTENSION" | awk '{print $1}')

normalized_csv() {
    tr ',' '\n' <<< "$1" | sed '/^[[:space:]]*$/d' | sed 's/[[:space:]]//g' | sort -n -u | paste -sd, -
}

reserve_for_count() {
    local count=$1 reserve_bytes measured minimum_bytes raw_bytes
    if [[ "$RESERVE_MODE" == profile_derived ]]; then
        read -r measured reserve_bytes raw_bytes minimum_bytes < <(
            python3 "$RESERVE_TOOL" \
                --profile "$GRAPH_MEMORY_PROFILE" --graph-count "$count" \
                --minimum-reserve-bytes "$GRAPH_RESERVE_MINIMUM_BYTES" \
                --quantum-bytes "$GRAPH_RESERVE_QUANTUM_BYTES" --format tsv
        )
    else
        measured=na
        minimum_bytes=na
        reserve_bytes=$FIXED_GRAPH_RESERVE_BYTES
        raw_bytes=$FIXED_GRAPH_RESERVE_BYTES
    fi
    local blocks=$(((reserve_bytes + KV_BLOCK_BYTES - 1) / KV_BLOCK_BYTES))
    local tokens=$((blocks * KV_BLOCK_SIZE))
    local full16=$((EAGER_FULL16_TOKENS - tokens))
    (( full16 > 0 )) || { echo "k=$count graph reserve exhausts Full16 KV" >&2; return 1; }
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$reserve_bytes" "$blocks" "$tokens" "$full16" \
        "$measured" "$minimum_bytes" "$raw_bytes"
}

if [[ $(normalized_csv "$COUNTS") != $(normalized_csv "$LAUNCH_ORDER") ]]; then
    echo "launch order must contain exactly the requested graph counts" >&2
    exit 2
fi

install_or_verify() {
    local candidate=$1 destination=$2 label=$3
    if [[ -e "$destination" ]]; then
        if [[ ! -f "$destination" || -L "$destination" ]] \
                || ! cmp -s "$candidate" "$destination"; then
            rm -f "$candidate"
            echo "immutable $label changed: $destination" >&2
            exit 2
        fi
        rm -f "$candidate"
    else
        mv "$candidate" "$destination"
        chmod a-w "$destination"
    fi
}

prepare_contracts() {
    mkdir -p "$ROOT"
    local -a selector_args=(
        --history "$SOURCE_HISTORY" \
        --train-file "$SOURCE_TRAIN" \
        --source-step "$SOURCE_STEP" \
        --counts "$COUNTS" \
        --output-plan "$PROFILE_PLAN.tmp" \
        --output-train "$PROFILE_TRAIN.tmp"
    )
    if [[ -n "$EXPLICIT_CAPTURE_SETS" ]]; then
        selector_args+=(--capture-sets-json "$EXPLICIT_CAPTURE_SETS")
    fi
    python3 "$SELECTOR" "${selector_args[@]}"
    python3 - "$PROFILE_PLAN.tmp" "$PROFILE_TRAIN" <<'PY'
import json, sys
from pathlib import Path
path = Path(sys.argv[1])
payload = json.loads(path.read_text(encoding="utf-8"))
payload["profile_train_file"] = str(Path(sys.argv[2]).resolve())
path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
PY
    if [[ -f "$PROFILE_PLAN" ]]; then
        cmp -s "$PROFILE_PLAN.tmp" "$PROFILE_PLAN" || {
            echo "profile plan changed under existing root" >&2; exit 2;
        }
        rm -f "$PROFILE_PLAN.tmp"
    else
        mv "$PROFILE_PLAN.tmp" "$PROFILE_PLAN"
    fi
    if [[ -f "$PROFILE_TRAIN" ]]; then
        cmp -s "$PROFILE_TRAIN.tmp" "$PROFILE_TRAIN" || {
            echo "profile train subset changed under existing root" >&2; exit 2;
        }
        rm -f "$PROFILE_TRAIN.tmp"
    else
        mv "$PROFILE_TRAIN.tmp" "$PROFILE_TRAIN"
    fi

    local trace_candidate
    trace_candidate=$(mktemp "$ROOT/.fixed_work_trace.XXXXXX")
    python3 "$TRACE_BUILDER" \
        --source-rollout "$FIXED_WORK_SOURCE" \
        --output "$trace_candidate"
    install_or_verify "$trace_candidate" "$FIXED_WORK_TRACE" fixed-work-trace

    local protocol_candidate code_candidate
    protocol_candidate=$(mktemp "$ROOT/.protocol.XXXXXX")
    code_candidate=$(mktemp "$ROOT/.code.XXXXXX")
    {
        printf 'schema_version=1\n'
        printf 'experiment=qwen3_aclgraph_bucket_count_single_step\n'
        printf 'common_epoch0_root=%s\n' "$COMMON_EPOCH0_ROOT"
        printf 'source_history=%s\n' "$(realpath "$SOURCE_HISTORY")"
        printf 'source_history_sha256=%s\n' "$(sha256sum "$SOURCE_HISTORY" | awk '{print $1}')"
        printf 'source_train=%s\n' "$(realpath "$SOURCE_TRAIN")"
        printf 'source_train_sha256=%s\n' "$(sha256sum "$SOURCE_TRAIN" | awk '{print $1}')"
        printf 'source_step=%s\n' "$SOURCE_STEP"
        printf 'graph_counts=%s\n' "$(normalized_csv "$COUNTS")"
        printf 'launch_order=%s\n' "$LAUNCH_ORDER"
        if [[ -n "$EXPLICIT_CAPTURE_SETS" ]]; then
            printf 'capture_set_policy=prespecified\n'
            printf 'explicit_capture_sets=%s\n' "$EXPLICIT_CAPTURE_SETS"
        fi
        printf 'seed=0\nactor_frozen=true\nsteps=1\nprompts=32\nrollout_n=16\n'
        printf 'sampling=greedy\ntemperature=0\ntop_p=1\ntop_k=-1\n'
        printf 'forced_token_replay=true\n'
        printf 'graph_mode=FULL_DECODE_ONLY\ntask_queue_enable=1\n'
        printf 'graph_reserve_mode=%s\n' "$RESERVE_MODE"
        if [[ "$RESERVE_MODE" == fixed_5p6g ]]; then
            printf 'full16_kv_tokens=%s\n' "$((EAGER_FULL16_TOKENS - ((FIXED_GRAPH_RESERVE_BYTES + KV_BLOCK_BYTES - 1) / KV_BLOCK_BYTES) * KV_BLOCK_SIZE))"
            printf 'graph_reserve_bytes=%s\n' "$FIXED_GRAPH_RESERVE_BYTES"
        else
            printf 'graph_reserve_formula=profile_rank_max_capture_rounded_up_0p5g_with_2p5g_minimum\n'
            printf 'graph_memory_profile=%s\n' "$(realpath "$GRAPH_MEMORY_PROFILE")"
            printf 'graph_memory_profile_sha256=%s\n' "$(sha256sum "$GRAPH_MEMORY_PROFILE" | awk '{print $1}')"
            printf 'graph_reserve_minimum_bytes=%s\n' "$GRAPH_RESERVE_MINIMUM_BYTES"
            printf 'graph_reserve_quantum_bytes=%s\n' "$GRAPH_RESERVE_QUANTUM_BYTES"
        fi
        printf 'native_sleep_mode=0\nasync_scheduling=false\n'
        printf 'filtered_opp_bundle_sha256=%s\n' "$FILTERED_OPP_BUNDLE_SHA256"
        printf 'ascend_extension_sha256=%s\n' "$EXTENSION_SHA256"
        printf 'profile_plan_sha256=%s\n' "$(sha256sum "$PROFILE_PLAN" | awk '{print $1}')"
        printf 'profile_train_sha256=%s\n' "$(sha256sum "$PROFILE_TRAIN" | awk '{print $1}')"
        printf 'fixed_work_source=%s\n' "$(realpath "$FIXED_WORK_SOURCE")"
        printf 'fixed_work_source_sha256=%s\n' "$(sha256sum "$FIXED_WORK_SOURCE" | awk '{print $1}')"
        printf 'fixed_work_trace_sha256=%s\n' "$(sha256sum "$FIXED_WORK_TRACE" | awk '{print $1}')"
    } > "$protocol_candidate"
    local -a code_paths=(
        "$SCRIPT_DIR/run_qwen3_aclgraph_step_bucket_profile.sh" \
        "$SELECTOR" "$SUMMARIZER" "$TRACE_BUILDER" "$FAIR_RUNNER" \
        "$RESERVE_TOOL" \
        "$SCRIPT_DIR/verl/utils/fixed_work_replay.py" \
        "$SCRIPT_DIR/verl/trainer/ppo/ray_trainer.py" \
        "$SCRIPT_DIR/verl/trainer/constants_ppo.py" \
        "$SCRIPT_DIR/verl/single_controller/ray/base.py" \
        "$SCRIPT_DIR/run_baseline_vanilla_epoch1_2_from_common_epoch0.sh" \
        "$SCRIPT_DIR/run_mode0_no_shrink_baseline.sh" \
        "$SCRIPT_DIR/verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py" \
        "$SCRIPT_DIR/vllm_ascend/compilation/acl_graph.py" \
        "$SCRIPT_DIR/vllm_ascend/worker/model_runner_v1.py"
    )
    if [[ -n "$EXTRA_CODE_CONTRACT" ]]; then
        require_file "$EXTRA_CODE_CONTRACT"
        code_paths+=("$EXTRA_CODE_CONTRACT")
    fi
    sha256sum "${code_paths[@]}" > "$code_candidate"
    install_or_verify "$protocol_candidate" "$PROTOCOL" protocol
    install_or_verify "$code_candidate" "$CODE_CONTRACT" code
}

capture_sizes_for_count() {
    python3 - "$PROFILE_PLAN" "$1" <<'PY'
import json, sys
plan = json.load(open(sys.argv[1]))
count = int(sys.argv[2])
for item in plan["candidates"]:
    if int(item["graph_count"]) == count:
        print(json.dumps(item["capture_sizes"], separators=(",", ":")))
        break
else:
    raise SystemExit(f"graph count {count} is absent from profile plan")
PY
}

run_count() {
    local count=$1 capture_sizes arm_dir run_name run_dir cache_root ray_tmp
    local reserve_bytes reserve_blocks reserve_tokens full16_kv_tokens
    local measured_capture_gib minimum_reserve_bytes raw_capture_bytes
    capture_sizes=$(capture_sizes_for_count "$count")
    IFS=$'\t' read -r reserve_bytes reserve_blocks reserve_tokens \
        full16_kv_tokens measured_capture_gib minimum_reserve_bytes \
        raw_capture_bytes < <(reserve_for_count "$count")
    arm_dir="$ROOT/arms/$(printf 'count_%02d' "$count")"
    run_name=$(printf 'qwen3_aclgraph_bucket_profile_k%02d' "$count")
    run_dir="$arm_dir/$run_name"
    cache_root="$arm_dir/cache"
    ray_tmp=$(printf '/tmp/ag_bp_%02d_%s' "$count" "$$")
    if [[ -f "$run_dir/CHECKPOINTS_REMOVED_AFTER_VALIDATION.txt" ]]; then
        echo "[bucket profile] already complete k=$count sizes=$capture_sizes"
        return 0
    fi
    if [[ -e "$run_dir" ]]; then
        echo "incomplete profile arm exists, use a fresh root or inspect it: $run_dir" >&2
        return 1
    fi
    mkdir -p "$arm_dir" "$cache_root"/{xdg,hf,triton,torchair,ascend,work} "$ray_tmp"
    {
        printf 'graph_count=%s\n' "$count"
        printf 'capture_sizes=%s\n' "$capture_sizes"
        printf 'graph_reserve_mode=%s\n' "$RESERVE_MODE"
        printf 'graph_reserve_bytes=%s\n' "$reserve_bytes"
        printf 'graph_reserve_blocks=%s\n' "$reserve_blocks"
        printf 'graph_reserve_tokens=%s\n' "$reserve_tokens"
        printf 'full16_kv_tokens=%s\n' "$full16_kv_tokens"
        printf 'profile_capture_memory_gib_rank_max=%s\n' "$measured_capture_gib"
        printf 'profile_minimum_reserve_bytes=%s\n' "$minimum_reserve_bytes"
        printf 'profile_raw_capture_bytes=%s\n' "$raw_capture_bytes"
    } > "$arm_dir/arm_protocol.env"
    python3 "$SAMPLER" --output "$arm_dir/npu_memory.csv" --interval 1 &
    local sampler_pid=$!
    stop_sampler() {
        if kill -0 "$sampler_pid" 2>/dev/null; then
            kill -TERM "$sampler_pid" 2>/dev/null || true
            wait "$sampler_pid" 2>/dev/null || true
        fi
    }
    trap 'stop_sampler; exit 130' INT
    trap 'stop_sampler; exit 143' TERM
    echo "[bucket profile] starting k=$count sizes=$capture_sizes reserve=${reserve_bytes}B full16_kv=$full16_kv_tokens"
    set +e
    env -u FLOOR_KV_CAPS \
        COMMON_EPOCH0_ROOT="$COMMON_EPOCH0_ROOT" \
        FAIR_OUTPUT_ROOT="$arm_dir" FAIR_RUN_NAME="$run_name" \
        FAIR_START_EPOCH=1 FAIR_TOTAL_EPOCHS=2 FAIR_TRAIN_STEPS=1 \
        FAIR_TRAIN_BATCH_SIZE=32 FAIR_ROLLOUT_N=16 \
        FAIR_MAX_RESPONSE_LENGTH=16384 FAIR_PROMPTS_PER_EPOCH=32 \
        FAIR_DATASET_FRACTION=1.0 FAIR_FREEZE_ACTOR=1 \
        FAIR_ROLLOUT_TEMPERATURE=0 FAIR_ROLLOUT_TOP_P=1 FAIR_ROLLOUT_TOP_K=-1 \
        FAIR_KEEP_COMPLETED_CHECKPOINTS=0 \
        VERL_PAIRED_REQUEST_SAMPLING_SEEDS=1 \
        VERL_FIXED_WORK_REPLAY_TRACE="$FIXED_WORK_TRACE" \
        VERL_FIXED_WORK_REPLAY_SHA256="$(sha256sum "$FIXED_WORK_TRACE" | awk '{print $1}')" \
        VERL_FIXED_WORK_REPLAY_REQUIRE_PLAN_CAP=0 \
        VERL_FIXED_WORK_REPLAY_ALLOW_DATASET_ITEM_IDX=1 \
        VERL_FIXED_WORK_REPLAY_FORCE_TOKEN_IDS=1 \
        TRAIN_FILE="$PROFILE_TRAIN" VANILLA_KV_TOKENS_PER_RANK="$full16_kv_tokens" \
        ADAFLOOR_ACLGRAPH_MODE=FULL_DECODE_ONLY \
        ADAFLOOR_GRAPH_CAPTURE_SIZES="$capture_sizes" \
        ROLLOUT_ENFORCE_EAGER=False VLLM_ENABLE_GRAPH_MODE=0 \
        VLLM_ASCEND_ELASTIC_ACLGRAPH=1 \
        VLLM_ASCEND_ELASTIC_ACLGRAPH_CAPTURE_ATTENTION=1 \
        VLLM_ASCEND_ELASTIC_ACLGRAPH_CAPTURE_MOE=1 \
        VLLM_ASCEND_ELASTIC_ACLGRAPH_EXTENSION="$EXTENSION" \
        VLLM_ASCEND_ELASTIC_ACLGRAPH_EXTENSION_SHA256="$EXTENSION_SHA256" \
        VLLM_ROLLOUT_REUSE_ACLGRAPH_AFTER_WEIGHT_UPDATE=1 \
        VLLM_ROLLOUT_NATIVE_SLEEP_MODE=0 VLLM_ROLLOUT_CAMEM_WEIGHT_RELOAD=0 \
        VLLM_ROLLOUT_FILTER_EMPTY_WEIGHT_SHARDS=1 \
        VLLM_ROLLOUT_ASYNC_SCHEDULING=false \
        VLLM_ROLLOUT_ENABLE_PREFIX_CACHING=true \
        VLLM_ROLLOUT_ENABLE_CHUNKED_PREFILL=true \
        VLLM_ASCEND_USE_FILTERED_CUSTOM_OPP=1 \
        VLLM_ASCEND_FILTERED_CUSTOM_OPP_PATH="$FILTERED_OPP" \
        VLLM_ASCEND_FILTERED_CUSTOM_OPP_BUNDLE_SHA256="$FILTERED_OPP_BUNDLE_SHA256" \
        VLLM_ASCEND_FORCE_ALLTOALL_MOE=0 VLLM_ASCEND_MC2_MIN_EP_SIZE=2 \
        VLLM_ASCEND_ENABLE_TOPK_TOPP_OPTIMIZATION=1 VLLM_ENABLE_MC2=1 \
        TASK_QUEUE_ENABLE=1 VERL_SIDECAR_ENABLE=0 \
        PYTORCH_NPU_ALLOC_CONF=expandable_segments:True \
        VERL_HCCL_IF_BASE_PORT_START=12000 VERL_MASTER_PORT_START=28416 \
        RAY_TMPDIR="$ray_tmp" XDG_CACHE_HOME="$cache_root/xdg" \
        HF_HOME="$cache_root/hf" TRITON_CACHE_DIR="$cache_root/triton" \
        TORCHAIR_CACHE_HOME="$cache_root/torchair" \
        ASCEND_CACHE_PATH="$cache_root/ascend" ASCEND_WORK_PATH="$cache_root/work" \
        "$FAIR_RUNNER" vanilla \
        actor_rollout_ref.rollout.seed=0 \
        actor_rollout_ref.rollout.cudagraph_mode=FULL_DECODE_ONLY \
        "actor_rollout_ref.rollout.cudagraph_capture_sizes=$capture_sizes" \
        actor_rollout_ref.actor.megatron.override_transformer_config.moe_shared_expert_overlap=False \
        2>&1 | tee "$arm_dir/runner.log"
    local rc=${PIPESTATUS[0]}
    set -e
    stop_sampler
    trap - INT TERM
    rm -rf "$cache_root" "$ray_tmp"
    if (( rc != 0 )); then
        echo "[bucket profile] k=$count failed rc=$rc" >&2
        return "$rc"
    fi
    prepare_contracts
    python3 "$SUMMARIZER" --root "$ROOT" --allow-incomplete
}

prepare_contracts
if [[ "$ACTION" == plan ]]; then
    cat "$PROFILE_PLAN"
    exit 0
fi
if [[ "$ACTION" == summarize ]]; then
    exec python3 "$SUMMARIZER" --root "$ROOT"
fi

IFS=',' read -r -a order <<< "$LAUNCH_ORDER"
for count in "${order[@]}"; do
    run_count "$count"
done
prepare_contracts
python3 "$SUMMARIZER" --root "$ROOT"
