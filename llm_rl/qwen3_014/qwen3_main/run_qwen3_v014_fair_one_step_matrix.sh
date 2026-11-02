#!/usr/bin/env bash
# Fair six-arm one-step matrix: eager/graph x Vanilla/Ada-N2/Ada-P-F4.
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
BASE_SCRIPT="$ROOT/internal/wj_train_grpo_qwen30b_a3b_16die_true_weight_regroup.sh"
BUILDER="$ROOT/tools/build_qwen3_v014_one_step_assets.py"
VERIFIER="$ROOT/tools/verify_qwen3_v014_fair_one_step_matrix.py"
ACTION=${1:-smoke}
case "$ACTION" in
    smoke|run|verify) ;;
    *) echo "usage: $0 {smoke|run|verify}" >&2; exit 2 ;;
esac

COMMON_ROOT=${QWEN3_V014_COMMON_EPOCH0_ROOT:-/workspace/adafloor_graph_results/qwen3_vanilla_epoch0_full_decode_fia_v014opt_tq1_buckets16_32_seed0/common_epoch0_full_decode_fia_v014opt_tq1_buckets16_32_seed0}
COMMON_CKPT="$COMMON_ROOT/epoch_000_mode0_probe/checkpoints/qwen3moe_for_eagle3/global_step_5"
SOURCE_TRAIN=${QWEN3_V014_SOURCE_TRAIN:-/data/deepscaler/train.parquet}
STAMP=${QWEN3_V014_FAIR_STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}
MATRIX_ROOT=${QWEN3_V014_FAIR_ROOT:-/workspace/adafloor_graph_results/qwen3_v014_fair_one_step_matrix_${STAMP}}
SOURCE_PLAN_STEP=${QWEN3_V014_FAIR_SOURCE_PLAN_STEP:-2}
MAX_RESPONSE_LENGTH=${QWEN3_V014_FAIR_MAX_RESPONSE_LENGTH:-6144}
ASSET_DIR="$MATRIX_ROOT/assets"
PROTOCOL="$MATRIX_ROOT/protocol.json"
RUN_MANIFEST="$MATRIX_ROOT/run_manifest.json"

ARMS=(
    eager_tq2_vanilla
    graph_tq1_vanilla
    graph_tq1_ada_n2
    eager_tq2_ada_n2
    eager_tq2_ada_p_f4
    graph_tq1_ada_p_f4
)
if [[ -n "${QWEN3_V014_FAIR_ARMS:-}" ]]; then
    IFS=, read -r -a ARMS <<< "$QWEN3_V014_FAIR_ARMS"
fi

sha256() {
    sha256sum "$1" | awk '{print $1}'
}

prepare_contract() {
    mkdir -p "$MATRIX_ROOT"
    python3 "$BUILDER" \
        --common-root "$COMMON_ROOT" \
        --train-file "$SOURCE_TRAIN" \
        --output-dir "$ASSET_DIR" \
        --source-plan-step "$SOURCE_PLAN_STEP" \
        --max-response-length "$MAX_RESPONSE_LENGTH" \
        --floor-kv-caps 2:111232,4:111232,8:377344,16:377344 \
        --max-rank-peak-tokens 377344 \
        > "$MATRIX_ROOT/asset_build_report.json"

    local code_tmp="$MATRIX_ROOT/code_sha256.txt.tmp"
    : > "$code_tmp"
    local file
    for file in \
        run_qwen3_v014_fair_one_step_matrix.sh \
        tools/build_qwen3_v014_one_step_assets.py \
        tools/verify_qwen3_v014_fair_one_step_matrix.py \
        tools/build_mode1_length_sorted_e2e_plan.py \
        tools/build_mode1_optimized_rank_plan.py \
        tools/aclgraph_bucket_plan.py \
        internal/wj_train_grpo_qwen30b_a3b_16die_true_weight_regroup.sh \
        verl/trainer/ppo/ray_trainer.py \
        verl/experimental/dataset/shrink_aware_assignment.py \
        verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py \
        vllm/v1/engine/llm_engine.py \
        vllm_ascend/worker/worker_v1.py \
        vllm_ascend/worker/model_runner_v1.py \
        vllm_ascend/ops/fused_moe_legacy.py \
        vllm_ascend/eplb/core/eplb_utils.py \
        vllm_ascend/ops/fused_moe/__init__.py \
        vllm_ascend/envs.py \
        vllm_ascend/ascend_forward_context.py; do
        printf '%s  %s\n' "$(sha256 "$ROOT/$file")" "$file" >> "$code_tmp"
    done

    local protocol_tmp="$MATRIX_ROOT/protocol.json.tmp"
    python3 - "$ASSET_DIR/asset_manifest.json" "$COMMON_CKPT" \
        "$code_tmp" "${ARMS[@]}" > "$protocol_tmp" <<'PY'
import hashlib, json, sys
from pathlib import Path
asset_path, checkpoint, code_path = map(Path, sys.argv[1:4])
arms = sys.argv[4:]
allowed = {f'{mode}_{method}' for mode in ('eager_tq2', 'graph_tq1')
           for method in ('vanilla', 'ada_n2', 'ada_p_f4')}
if not arms or len(set(arms)) != len(arms) or set(arms) - allowed:
    raise SystemExit(f'invalid arm selection: {arms}')
asset = json.loads(asset_path.read_text())
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
print(json.dumps({
    "schema_version": 2,
    "experiment": "qwen3_v014_fair_eager_graph_one_step_matrix",
    "expected_arms": arms,
    "common_checkpoint": str(checkpoint.resolve()),
    "common_history_sha256": asset["common_history_sha256"],
    "selected_source_indices": asset["selected_source_indices"],
    "source_plan_step": asset["source_plan_step"],
    "method_assets": asset["plans"],
    "prompts": 32,
    "responses_per_prompt": 16,
    "expected_responses": 512,
    "max_response_length": asset["max_response_length"],
    "num_gpu_blocks_override": 869,
    "kv_block_size": 128,
    "kv_capacity_tokens": 111232,
    "seed": 0,
    "actor_lr": 0.0,
    "tail_guard_enabled": False,
    "regroup_enabled": False,
    "fused_moe_backend": "legacy",
    "automatic_mode1_redundancy": True,
    "planned_topology_precreation": True,
    "elastic_camem_weight_reload": True,
    "elastic_graph_recapture_after_weight_update": True,
    "async_scheduling": False,
    "prefix_caching": False,
    "chunked_prefill": True,
    "graph": {"task_queue": 1, "mode": "FULL_DECODE_ONLY", "capture_sizes": [16, 32]},
    "eager": {"task_queue": 2, "enforce_eager": True},
    "code_manifest_sha256": sha(code_path),
}, indent=2, sort_keys=True))
PY

    if [[ -e "$PROTOCOL" ]]; then
        cmp -s "$PROTOCOL" "$protocol_tmp" || {
            echo "protocol/code contract changed for existing root: $MATRIX_ROOT" >&2
            exit 2
        }
        cmp -s "$MATRIX_ROOT/code_sha256.txt" "$code_tmp" || {
            echo "code contract changed for existing root: $MATRIX_ROOT" >&2
            exit 2
        }
        rm -f "$protocol_tmp" "$code_tmp"
    else
        mv "$protocol_tmp" "$PROTOCOL"
        mv "$code_tmp" "$MATRIX_ROOT/code_sha256.txt"
        printf '[]\n' > "$RUN_MANIFEST"
    fi
}

arm_parts() {
    local name=$1
    case "$name" in
        eager_tq2_vanilla) printf 'eager\tvanilla\n' ;;
        graph_tq1_vanilla) printf 'graph\tvanilla\n' ;;
        graph_tq1_ada_n2) printf 'graph\tada_n2\n' ;;
        eager_tq2_ada_n2) printf 'eager\tada_n2\n' ;;
        eager_tq2_ada_p_f4) printf 'eager\tada_p_f4\n' ;;
        graph_tq1_ada_p_f4) printf 'graph\tada_p_f4\n' ;;
        *) return 2 ;;
    esac
}

print_config() {
    local name=$1 mode method
    IFS=$'\t' read -r mode method < <(arm_parts "$name")
    printf 'arm=%s mode=%s method=%s source_plan_step=%s prompts=32 n=16 max_response=%s checkpoint=%s\n' \
        "$name" "$mode" "$method" "$SOURCE_PLAN_STEP" "$MAX_RESPONSE_LENGTH" "$COMMON_CKPT"
}

append_complete_manifest() {
    local name=$1 mode=$2 method=$3 plan=$4
    python3 - "$RUN_MANIFEST" "$name" "$mode" "$method" "$plan" <<'PY'
import json, os, sys
from pathlib import Path
manifest, name, mode, method, plan = sys.argv[1:]
path = Path(manifest)
rows = json.loads(path.read_text())
rows.append({
    "name": name,
    "mode": mode,
    "method": method,
    "plan": plan or None,
    "status": "complete",
})
tmp = path.with_suffix(".tmp")
tmp.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n")
os.replace(tmp, path)
PY
}

run_arm() (
    set -euo pipefail
    local name=$1 mode method
    IFS=$'\t' read -r mode method < <(arm_parts "$name")
    local arm_dir="$MATRIX_ROOT/$name"
    [[ ! -e "$arm_dir" ]] || {
        echo "refusing to reuse arm directory: $arm_dir" >&2
        exit 2
    }

    export OUTPUT_ROOT="$MATRIX_ROOT"
    export OUTPUT_SUBDIR="$name"
    local dataset="$ASSET_DIR/ada_p_f4_train.parquet"
    local oracle="$ASSET_DIR/ada_p_f4_oracle.json"
    if [[ "$method" == ada_n2 ]]; then
        dataset="$ASSET_DIR/ada_n2_train.parquet"
        oracle="$ASSET_DIR/ada_n2_oracle.json"
    fi
    export TRAIN_FILE="$dataset"
    export TEST_FILE="$dataset"
    export TRAIN_BATCH_SIZE=32
    export ACTOR_PPO_MINI_BATCH_SIZE=32
    export DATASET_FRACTION=1.0
    export MAX_RESPONSE_LENGTH="$MAX_RESPONSE_LENGTH"
    export ROLLOUT_N=16
    export ROLLOUT_MAX_NUM_SEQS=32
    export TRAINER_TOTAL_EPOCHS=1
    export TRAINER_TOTAL_TRAINING_STEPS=1
    export SAVE_CKPT_ENABLE=0
    export GIT_TRACK_RUN_LOGS=0
    export VERL_RESET_TRAINER_PROGRESS_AFTER_RESUME=1
    export VLLM_EPOCH_LENGTH_REGROUP_ENABLE=0
    export VLLM_EAGER_BASELINE_NO_RESAMPLE=0
    export VLLM_ROLLOUT_EARLY_STOP_ENABLE=0
    export VLLM_ROLLOUT_SHRINK_AWARE_SHORT_STEP_CAP_ENABLE=0
    export VLLM_ROLLOUT_DATA_REBALANCE=0
    export VLLM_ROLLOUT_LENGTH_BALANCE=0
    export VLLM_ROLLOUT_GRAPH_SPREAD_REPEATS=0
    # Elastic scheduling is synchronous in v0.14. Keep Vanilla on the same
    # scheduler so the graph/eager and shrink deltas remain causal.
    export VLLM_ROLLOUT_ASYNC_SCHEDULING=false
    export VLLM_ROLLOUT_ENABLE_PREFIX_CACHING=false
    export VLLM_ROLLOUT_ENABLE_CHUNKED_PREFILL=true
    export VLLM_ROLLOUT_GPU_MEMORY_UTILIZATION=0.83
    export VLLM_ROLLOUT_MANUAL_FREE_CACHE_ENGINE=0
    export VLLM_ROLLOUT_FREE_CACHE_ENGINE=True
    export VLLM_ROLLOUT_SLEEP_LEVEL=1
    export VLLM_ROLLOUT_CAMEM_WEIGHT_RELOAD=0
    export VLLM_ROLLOUT_DIVERSIFY_SAMPLING_SEED=1
    export VLLM_ASCEND_MC2_TOKENS_CAPACITY=512
    export VLLM_ASCEND_FUSED_MOE_BACKEND=legacy
    export VLLM_ASCEND_USE_LEGACY_FUSED_MOE=1
    export VLLM_ASCEND_INIT_REDUNDANCY_EXPERT=0
    export VLLM_ASCEND_FUSED_MOE_SIMPLE_MC2=1
    export VLLM_ASCEND_ATTENTION_BLOCK_SIZE=64
    export VLLM_ASCEND_EAGER_METADATA_SYNC_DEVICE=1
    export VLLM_ASCEND_FORCE_TORCH_NPU_ADD_RMS_NORM=1
    export VLLM_ASCEND_EAGER_COMPILE_PASS_FUSION=0

    if [[ "$mode" == graph ]]; then
        export VLLM_ASCEND_USE_LOCAL_CUSTOM_OPP=1
        export VLLM_ASCEND_USE_LOCAL_CUSTOM_OP_API_LIB=1
        export VLLM_ENABLE_GRAPH_MODE=1
        export VLLM_ASCEND_ELASTIC_ACLGRAPH=1
        export ROLLOUT_ENFORCE_EAGER=False
        export VLLM_ROLLOUT_TASK_QUEUE_ENABLE=1
        export TASK_QUEUE_ENABLE=1
        export VLLM_ROLLOUT_GRAPH_WITH_RESAMPLER=0
        export VLLM_ROLLOUT_UNSAFE_TRUE_GRAPH_WITH_RESAMPLER=1
        export VLLM_ASCEND_ALLOW_LAZY_ACLGRAPH_CAPTURE=0
        export VLLM_ROLLOUT_INVALIDATE_ACLGRAPH_AFTER_WEIGHT_UPDATE=0
        export VLLM_ROLLOUT_RECAPTURE_ACLGRAPH_AFTER_WEIGHT_UPDATE=0
        export VLLM_ROLLOUT_DELAY_GRAPH_CAPTURE_UNTIL_WEIGHT_LOAD=0
        export VLLM_ROLLOUT_CAPTURE_GRAPH_AFTER_WEIGHT_LOAD=0
    else
        export VLLM_ASCEND_USE_LOCAL_CUSTOM_OPP=0
        export VLLM_ASCEND_USE_LOCAL_CUSTOM_OP_API_LIB=0
        export VLLM_ENABLE_GRAPH_MODE=0
        export VLLM_ASCEND_ELASTIC_ACLGRAPH=0
        export ROLLOUT_ENFORCE_EAGER=True
        export VLLM_ROLLOUT_TASK_QUEUE_ENABLE=2
        export TASK_QUEUE_ENABLE=2
    fi

    local plan=""
    local policy="planned"
    local floor=16
    local stages=""
    local stage_ranks="[]"
    local intermediate=""
    local final=""
    local scheduling=false
    if [[ "$method" == ada_n2 ]]; then
        plan="$ASSET_DIR/ada_n2_plan.json"
        policy=natural
        floor=2
        scheduling=true
    elif [[ "$method" == ada_p_f4 ]]; then
        plan="$ASSET_DIR/ada_p_f4_plan.json"
        policy=planned
        floor=4
        scheduling=true
    fi

    if [[ "$scheduling" == true ]]; then
        IFS=$'\t' read -r stages stage_ranks intermediate final < <(
            python3 - "$plan" <<'PY'
import json, sys
p = json.load(open(sys.argv[1]))[0]
stages = ",".join(map(str, p["shrink_stages"]))
stage_ranks = json.dumps(p["stage_survivor_ranks"], separators=(",", ":"))
intermediate = ",".join(map(str, p["intermediate_survivor_ranks"]))
final = ",".join(map(str, p["final_survivor_ranks"]))
print(stages, stage_ranks, intermediate, final, sep="\t")
PY
        )
    fi

    export VLLM_ASCEND_ENABLE_ELASTIC_PARALLEL_SHRINK=$([[ "$scheduling" == true ]] && echo 1 || echo 0)
    export VLLM_ASCEND_ELASTIC_EXECUTION_MODE=$([[ "$scheduling" == true ]] && echo 1 || echo 0)
    export VLLM_ASCEND_ELASTIC_MOE_MODE=lossless
    export VLLM_ASCEND_ELASTIC_MIN_COMPUTE_GROUP_SIZE=$floor
    export VLLM_ASCEND_MC2_MIN_EP_SIZE=$floor
    export VLLM_ASCEND_SHRINK_AWARE_ENABLE=$([[ "$scheduling" == true ]] && echo 1 || echo 0)
    export VLLM_ASCEND_SHRINK_AWARE_MODE=staged
    export VLLM_ASCEND_SHRINK_AWARE_STAGES=${stages:-16}
    export VLLM_ASCEND_SHRINK_AWARE_SURVIVOR_POLICY=manual
    export VLLM_ASCEND_SHRINK_AWARE_TARGET_POLICY=$policy
    export VLLM_ASCEND_SHRINK_AWARE_INTERMEDIATE_RANKS=$intermediate
    export VLLM_ASCEND_SHRINK_AWARE_FINAL_RANKS=$final
    export VLLM_ASCEND_SHRINK_AWARE_STAGE_RANKS=$stage_ranks
    export VLLM_ASCEND_SHRINK_AWARE_MIN_WINDOW_SECONDS=0
    export VLLM_ASCEND_SHRINK_AWARE_LOGGING=1
    export VLLM_ASCEND_MODE1_PRECREATE_PLANNED_FLOOR_GROUPS=$(
        [[ "$method" == ada_p_f4 ]] && echo 1 || echo 0
    )
    export VLLM_ASCEND_MODE1_PRECREATE_COMM_WARMUP=1
    if [[ "$scheduling" == true ]]; then
        export VLLM_ROLLOUT_CAMEM_WEIGHT_RELOAD=1
        export VLLM_ROLLOUT_PHASE_MEMORY_LOG=1
        export VERL_LOGGING_LEVEL=INFO
        export VLLM_ROLLOUT_STAGE_DEBUG=1
        if [[ "$mode" == graph ]]; then
            export VLLM_ROLLOUT_INVALIDATE_ACLGRAPH_AFTER_WEIGHT_UPDATE=1
            export VLLM_ROLLOUT_RECAPTURE_ACLGRAPH_AFTER_WEIGHT_UPDATE=1
        fi
    fi

    if [[ "$floor" == 2 ]]; then
        export VLLM_ASCEND_POST_SHRINK_MOE_DISPATCH_HEADROOM_BYTES=0
        export VLLM_ASCEND_POST_SHRINK_MOE_DISPATCH_LOW_FLOOR_HEADROOM_BYTES=0
        export VLLM_ASCEND_POST_SHRINK_PREFILL_ALLTOALL_HEADROOM_BYTES=0
        export VLLM_ASCEND_POST_RESTORE_DP_HEADROOM_BYTES=0
        export VLLM_ASCEND_POST_RESTORE_EP_HEADROOM_BYTES=0
        export VLLM_ASCEND_POST_RESTORE_MOE_DISPATCH_HEADROOM_BYTES=0
        export VLLM_ASCEND_FIRST_LIVE_PREFILL_HEADROOM_BYTES=0
        export VLLM_ASCEND_FIRST_LIVE_PREFILL_LOW_FLOOR_HEADROOM_BYTES=0
        export VLLM_ASCEND_EXTRA_ELASTIC_SAFETY_HEADROOM_BYTES=0
        export VLLM_ASCEND_FLOOR_PREALLOC_HEADROOM_SAFETY_BYTES=0
    fi

    local hydra=(
        trainer.resume_mode=resume_path
        "trainer.resume_from_path=$COMMON_CKPT"
        actor_rollout_ref.actor.optim.lr=0.0
        actor_rollout_ref.actor.use_kl_loss=False
        algorithm.use_kl_in_reward=False
        actor_rollout_ref.rollout.seed=0
        actor_rollout_ref.rollout.temperature=0.9
        actor_rollout_ref.rollout.top_p=0.9
        actor_rollout_ref.rollout.top_k=50
        actor_rollout_ref.rollout.ignore_eos=False
        actor_rollout_ref.rollout.max_model_len=$((MAX_RESPONSE_LENGTH + 1024))
        actor_rollout_ref.rollout.max_num_batched_tokens=$((MAX_RESPONSE_LENGTH + 1024))
        "+actor_rollout_ref.rollout.engine_kwargs.vllm.num_gpu_blocks_override=869"
        "+actor_rollout_ref.rollout.engine_kwargs.vllm.block_size=128"
        "actor_rollout_ref.rollout.shrink_aware.enable_shrink_aware_scheduling=$scheduling"
        +actor_rollout_ref.actor.megatron.override_transformer_config.moe_shared_expert_overlap=False
        'trainer.logger=["console"]'
    )
    if [[ "$scheduling" == true ]]; then
        hydra+=(
            actor_rollout_ref.rollout.shrink_aware.shrink_aware_mode=staged
            "actor_rollout_ref.rollout.shrink_aware.shrink_stages=[$stages]"
            actor_rollout_ref.rollout.shrink_aware.survivor_selection_policy=manual
            "actor_rollout_ref.rollout.shrink_aware.target_policy=$policy"
            "actor_rollout_ref.rollout.shrink_aware.intermediate_survivor_ranks=[$intermediate]"
            "actor_rollout_ref.rollout.shrink_aware.final_survivor_ranks=[$final]"
            "actor_rollout_ref.rollout.shrink_aware.stage_survivor_ranks=$stage_ranks"
            actor_rollout_ref.rollout.shrink_aware.length_prediction_source=oracle_trace
            "actor_rollout_ref.rollout.shrink_aware.oracle_trace_path=$oracle"
            actor_rollout_ref.rollout.shrink_aware.assignment_policy=optimized_rank_plan
            "actor_rollout_ref.rollout.shrink_aware.optimized_rank_plan_path=$plan"
            actor_rollout_ref.rollout.shrink_aware.min_shrink_window_seconds=0
            actor_rollout_ref.rollout.shrink_aware.enable_shrink_aware_logging=true
        )
    fi
    if [[ "$mode" == graph ]]; then
        hydra+=(
            actor_rollout_ref.rollout.cudagraph_mode=FULL_DECODE_ONLY
            'actor_rollout_ref.rollout.cudagraph_capture_sizes=[16,32]'
        )
    else
        hydra+=(
            actor_rollout_ref.rollout.cudagraph_mode=null
            actor_rollout_ref.rollout.cudagraph_capture_sizes=null
        )
    fi

    printf '[matrix] starting %s mode=%s method=%s\n' "$name" "$mode" "$method"
    bash "$BASE_SCRIPT" "${hydra[@]}"

    python3 - "$arm_dir" "$name" "$mode" "$method" "$plan" <<'PY'
import hashlib, json, sys
from pathlib import Path
arm_dir, name, mode, method, plan = sys.argv[1:]
payload = {
    "schema_version": 1,
    "status": "complete",
    "name": name,
    "mode": mode,
    "method": method,
    "plan": plan or None,
    "plan_sha256": hashlib.sha256(Path(plan).read_bytes()).hexdigest() if plan else None,
}
Path(arm_dir, "arm_manifest.json").write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n"
)
PY
    append_complete_manifest "$name" "$mode" "$method" "$plan"
)

prepare_contract

if [[ "$ACTION" == smoke ]]; then
    python3 -m py_compile "$BUILDER" "$VERIFIER" "$ROOT/verl/trainer/ppo/ray_trainer.py"
    bash -n "$0" "$BASE_SCRIPT"
    python3 "$VERIFIER" --root "$MATRIX_ROOT" --allow-incomplete > "$MATRIX_ROOT/smoke_verifier.json"
    for arm in "${ARMS[@]}"; do print_config "$arm"; done
    echo "SMOKE_PASS root=$MATRIX_ROOT"
    exit 0
fi

if [[ "$ACTION" == verify ]]; then
    exec python3 "$VERIFIER" --root "$MATRIX_ROOT"
fi

completed_count=$(python3 -c 'import json,sys; print(len(json.load(open(sys.argv[1]))))' "$RUN_MANIFEST")
if [[ "$completed_count" -ne 0 ]]; then
    echo "run requires a fresh smoke-only root; completed_arm_count=$completed_count" >&2
    exit 2
fi
for arm in "${ARMS[@]}"; do
    run_arm "$arm"
    python3 "$VERIFIER" --root "$MATRIX_ROOT" --allow-incomplete \
        > "$MATRIX_ROOT/partial_summary.json"
done
python3 "$VERIFIER" --root "$MATRIX_ROOT" > "$MATRIX_ROOT/final_verifier_output.json"
echo "MATRIX_PASS root=$MATRIX_ROOT"
