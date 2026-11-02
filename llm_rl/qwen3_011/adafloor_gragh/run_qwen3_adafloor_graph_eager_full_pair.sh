#!/usr/bin/env bash
set -euo pipefail

if [[ "${ADAFLOOR_GRAPH_EAGER_PAIR_SNAPSHOT_ACTIVE:-0}" != "1" ]]; then
    source_path=$(realpath "${BASH_SOURCE[0]}")
    snapshot=$(mktemp "${source_path}.run-snapshot.XXXXXX")
    cp -- "$source_path" "$snapshot"
    chmod 700 "$snapshot"
    set +e
    ADAFLOOR_GRAPH_EAGER_PAIR_SNAPSHOT_ACTIVE=1 "$snapshot" "$@"
    rc=$?
    set -e
    rm -f -- "$snapshot"
    exit "$rc"
fi

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR"

PAIR_ROOT="${ADAFLOOR_GRAPH_EAGER_PAIR_ROOT:-/workspace/adafloor_graph_results/qwen3_aclgraph_full_epoch_matrix_$(date -u +%Y%m%dT%H%M%SZ)}"
COMMON_EPOCH0_NAME=common_epoch0_graph_vanilla
COMMON_EPOCH0_ROOT="${COMMON_EPOCH0_ROOT:-$PAIR_ROOT/$COMMON_EPOCH0_NAME}"
SEED=101
EAGER_FULL_KV_TOKENS=380800
GRAPH_FULL_KV_TOKENS=380800
EAGER_TASK_QUEUE_ENABLE=2
GRAPH_TASK_QUEUE_ENABLE=1
CAPTURE_SIZES='[1,2,4,8,16,32]'
EXPECTED_RESPONSES_PER_STEP=512
STEPS=5
CODE_CONTRACT="$PAIR_ROOT/code_sha256.txt"
PROTOCOL_CONTRACT="$PAIR_ROOT/protocol.env"
RUN_MANIFEST="$PAIR_ROOT/run_manifest.tsv"
VERIFY_TOOL="$SCRIPT_DIR/tools/verify_qwen3_adafloor_graph_eager_full_pair.py"
GRAPH_WRAPPER="$SCRIPT_DIR/run_mode1_local_length_sorted_e2e_adaptive_floor4_aclgraph.sh"
ASCEND_EXTENSION="${VLLM_ASCEND_ELASTIC_ACLGRAPH_EXTENSION:-/workspace/vllm-ascend/vllm_ascend/vllm_ascend_C.cpython-311-aarch64-linux-gnu.so}"

CODE_CONTRACT_PATHS=(
    "$SCRIPT_DIR/run_qwen3_adafloor_graph_eager_full_pair.sh"
    "$VERIFY_TOOL"
    "$SCRIPT_DIR/run_paper_fair_epoch1_2_from_common_epoch0.sh"
    "$SCRIPT_DIR/run_common_epoch0_probe_gpu09_kv380800_permanent.sh"
    "$SCRIPT_DIR/run_mode0_no_shrink_baseline.sh"
    "$SCRIPT_DIR/run_baseline_vanilla_epoch1_2_from_common_epoch0.sh"
    "$SCRIPT_DIR/run_baseline_lengthsort_epoch1_2.sh"
    "$GRAPH_WRAPPER"
    "$SCRIPT_DIR/run_mode1_dynamic_length_aware_adaptive_floor4_planned_full3.sh"
    "$SCRIPT_DIR/run_mode1_dynamic_length_aware_adaptive_floor2_natural_tailguard_reuse_epoch0_2epoch.sh"
    "$SCRIPT_DIR/run_mode1_dynamic_length_aware_adaptive_floor4_epochs.sh"
    "$SCRIPT_DIR/run_mode1_local_length_sorted_e2e_adaptive_floor4.sh"
    "$SCRIPT_DIR/internal/wj_train_grpo_qwen30b_a3b_16die_true_weight_regroup.sh"
    "$SCRIPT_DIR/internal/wj_train_grpo_qwen30b_a3b_16die_true_weight_eager_baseline_util.sh"
    "$SCRIPT_DIR/tools/build_mode1_length_sorted_e2e_plan.py"
    "$SCRIPT_DIR/tools/build_mode1_optimized_rank_plan.py"
    "$SCRIPT_DIR/verl/experimental/dataset/shrink_aware_assignment.py"
    "$SCRIPT_DIR/verl/trainer/ppo/ray_trainer.py"
    "$SCRIPT_DIR/verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py"
    "$SCRIPT_DIR/vllm_ascend/envs.py"
    "$SCRIPT_DIR/vllm_ascend/platform.py"
    "$SCRIPT_DIR/vllm_ascend/attention/attention_v1.py"
    "$SCRIPT_DIR/vllm_ascend/models/qwen3_moe.py"
    "$SCRIPT_DIR/vllm_ascend/ops/fused_moe.py"
    "$SCRIPT_DIR/vllm_ascend/compilation/acl_graph.py"
    "$SCRIPT_DIR/vllm_ascend/worker/model_runner_v1.py"
    "$SCRIPT_DIR/vllm_ascend/worker/worker_v1.py"
)

usage() {
    cat <<'EOF'
Usage:
  ./run_qwen3_adafloor_graph_eager_full_pair.sh [all|common|dry-run|verify]

Builds one trained, five-step graph Vanilla epoch0, then runs frozen-actor
five-step graph/eager epoch1 pairs for Vanilla, LengthSort+TailGuard,
AdaFloor Planned F4, and AdaFloor Natural F2. Every epoch1 arm starts from the
same graph-produced checkpoint and uses the same request seed and workload.
The eager arms use TASK_QUEUE_ENABLE=2. The graph arms use
TASK_QUEUE_ENABLE=1 and native FULL_DECODE_ONLY ACLGraph with Attention and
elastic MoE inside the decode graph.
EOF
}

action="${1:-all}"
case "$action" in
    all|common|dry-run|verify) ;;
    -h|--help) usage; exit 0 ;;
    *) usage >&2; exit 2 ;;
esac

sha256_file() {
    sha256sum "$1" | awk '{print $1}'
}

install_or_verify_immutable() {
    local candidate="$1" destination="$2" label="$3"
    if [[ -e "$destination" ]]; then
        if [[ ! -f "$destination" || -L "$destination" ]] \
           || ! cmp -s -- "$candidate" "$destination"; then
            echo "[graph/eager pair] immutable $label changed: $destination" >&2
            rm -f -- "$candidate"
            return 1
        fi
        rm -f -- "$candidate"
        return 0
    fi
    if ln -- "$candidate" "$destination" 2>/dev/null; then
        rm -f -- "$candidate"
        return 0
    fi
    if [[ -f "$destination" && ! -L "$destination" ]] \
       && cmp -s -- "$candidate" "$destination"; then
        rm -f -- "$candidate"
        return 0
    fi
    echo "[graph/eager pair] cannot publish immutable $label" >&2
    rm -f -- "$candidate"
    return 1
}

seal_contracts() {
    mkdir -p "$PAIR_ROOT"
    local code_candidate protocol_candidate created
    code_candidate=$(mktemp "$PAIR_ROOT/.code.XXXXXX")
    protocol_candidate=$(mktemp "$PAIR_ROOT/.protocol.XXXXXX")
    sha256sum "${CODE_CONTRACT_PATHS[@]}" > "$code_candidate"
    if [[ -f "$PROTOCOL_CONTRACT" && ! -L "$PROTOCOL_CONTRACT" ]]; then
        created=$(sed -n 's/^created_at_utc=//p' "$PROTOCOL_CONTRACT")
    else
        created=$(date -u +%Y-%m-%dT%H:%M:%SZ)
    fi
    {
        printf 'schema_version=2\n'
        printf 'created_at_utc=%s\n' "$created"
        printf 'experiment=qwen3_aclgraph_full_epoch_matrix\n'
        printf 'common_epoch0_root=%s\n' "$COMMON_EPOCH0_ROOT"
        printf 'common_epoch0_mode=graph_vanilla\n'
        printf 'common_epoch0_actor_updated=true\n'
        printf 'common_epoch0_steps=%s\n' "$STEPS"
        printf 'seed=%s\n' "$SEED"
        printf 'epoch1_actor_frozen=true\n'
        printf 'paired_request_sampling_seeds=true\n'
        printf 'policies=vanilla,lengthsort_guard,planned,natural\n'
        printf 'modes=eager,graph\n'
        printf 'launch_order=vanilla:eager,vanilla:graph,lengthsort_guard:graph,lengthsort_guard:eager,planned:eager,planned:graph,natural:graph,natural:eager\n'
        printf 'prompts_per_step=32\n'
        printf 'rollout_n=16\n'
        printf 'responses_per_step=%s\n' "$EXPECTED_RESPONSES_PER_STEP"
        printf 'steps=%s\n' "$STEPS"
        printf 'max_response_length=16384\n'
        printf 'tail_guard=policy_default\n'
        printf 'eager_full16_kv_tokens=%s\n' "$EAGER_FULL_KV_TOKENS"
        printf 'graph_full16_kv_tokens=%s\n' "$GRAPH_FULL_KV_TOKENS"
        printf 'kv_bytes_per_token=98304\n'
        printf 'graph_capture_sizes=%s\n' "$CAPTURE_SIZES"
        printf 'graph_capture_profile=balanced\n'
        printf 'graph_mode=FULL_DECODE_ONLY\n'
        printf 'graph_attention=true\n'
        printf 'graph_moe=true\n'
        printf 'eager_task_queue_enable=%s\n' "$EAGER_TASK_QUEUE_ENABLE"
        printf 'graph_task_queue_enable=%s\n' "$GRAPH_TASK_QUEUE_ENABLE"
        printf 'torchair=false\n'
        printf 'sidecar=false\n'
        printf 'moe_shared_expert_overlap=false\n'
    } > "$protocol_candidate"
    install_or_verify_immutable "$code_candidate" "$CODE_CONTRACT" "code contract"
    install_or_verify_immutable "$protocol_candidate" "$PROTOCOL_CONTRACT" "protocol contract"
}

require_inputs() {
    local path
    for path in \
        "$VERIFY_TOOL" "$GRAPH_WRAPPER" "$ASCEND_EXTENSION" \
        /data/Qwen3-30B-A3B /data/Qwen3-30B-A3B_megatron \
        /data/deepscaler/train.parquet /data/deepscaler/test.parquet; do
        [[ -e "$path" ]] || { echo "missing required input: $path" >&2; exit 2; }
    done
    local kv
    for kv in "$EAGER_FULL_KV_TOKENS" "$GRAPH_FULL_KV_TOKENS"; do
        (( kv % 128 == 0 )) || {
            echo "full16 KV capacity must be block aligned: $kv" >&2
            exit 2
        }
    done
}

arm_order() {
    printf '%s\n' \
        'vanilla eager' 'vanilla graph' \
        'lengthsort_guard graph' 'lengthsort_guard eager' \
        'planned eager' 'planned graph' \
        'natural graph' 'natural eager'
}

variant_for_policy() {
    case "$1" in
        vanilla) printf 'vanilla\n' ;;
        lengthsort_guard) printf 'lengthsort_guard\n' ;;
        planned) printf 'adafloor_p_f4\n' ;;
        natural) printf 'adafloor_n_f2\n' ;;
        *) return 1 ;;
    esac
}

base_runner_for_policy() {
    case "$1" in
        vanilla) printf '%s\n' "$SCRIPT_DIR/run_baseline_vanilla_epoch1_2_from_common_epoch0.sh" ;;
        lengthsort_guard) printf '%s\n' "$SCRIPT_DIR/run_baseline_lengthsort_epoch1_2.sh" ;;
        planned) printf '%s\n' "$SCRIPT_DIR/run_mode1_dynamic_length_aware_adaptive_floor4_planned_full3.sh" ;;
        natural) printf '%s\n' "$SCRIPT_DIR/run_mode1_dynamic_length_aware_adaptive_floor2_natural_tailguard_reuse_epoch0_2epoch.sh" ;;
        *) return 1 ;;
    esac
}

run_name_for_arm() {
    printf 'qwen3_adafloor_%s_%s_seed%s_full5\n' "$1" "$2" "$SEED"
}

arm_manifest_path() {
    printf '%s/%s/%s/arm_manifest.json\n' "$PAIR_ROOT" "$1" "$2"
}

manifest_has_arm() {
    local policy="$1" mode="$2"
    [[ -f "$RUN_MANIFEST" ]] && awk -F '\t' -v p="$policy" -v m="$mode" \
        'NR > 1 && $1 == p && $2 == m && $3 == "complete" {found=1} END {exit !found}' \
        "$RUN_MANIFEST"
}

write_arm_manifest() {
    local policy="$1" mode="$2" arm_root="$3" run_dir="$4"
    local epoch_dir plan_file manifest tmp full_kv task_queue graph_attention graph_moe
    manifest=$(arm_manifest_path "$policy" "$mode")
    mapfile -t epoch_dirs < <(find "$run_dir" -mindepth 1 -maxdepth 1 -type d -name 'epoch_001_*' -print)
    (( ${#epoch_dirs[@]} == 1 )) || { echo "expected one epoch directory in $run_dir" >&2; return 1; }
    epoch_dir="${epoch_dirs[0]}"
    plan_file="$epoch_dir/oracle/length_sorted_rank_plan.json"
    if [[ ! -f "$plan_file" ]]; then
        plan_file=""
    fi
    [[ -f "$run_dir/CHECKPOINTS_REMOVED_AFTER_VALIDATION.txt" ]] || {
        echo "arm did not finish fair validation: $run_dir" >&2
        return 1
    }
    if [[ "$mode" == graph ]]; then
        full_kv=$GRAPH_FULL_KV_TOKENS
        task_queue=$GRAPH_TASK_QUEUE_ENABLE
        graph_attention=true
        graph_moe=true
    else
        full_kv=$EAGER_FULL_KV_TOKENS
        task_queue=$EAGER_TASK_QUEUE_ENABLE
        graph_attention=false
        graph_moe=false
    fi
    tmp=$(mktemp "$arm_root/.arm_manifest.XXXXXX")
    python3 - "$tmp" "$policy" "$mode" "$run_dir" "$epoch_dir" "$plan_file" \
        "$SEED" "$full_kv" "$CAPTURE_SIZES" "$task_queue" \
        "$graph_attention" "$graph_moe" "$COMMON_EPOCH0_ROOT" <<'PY'
import hashlib
import json
import pathlib
import sys

(output, policy, mode, run_dir, epoch_dir, plan_file, seed, kv, sizes,
 task_queue, graph_attention, graph_moe, common_epoch0_root) = sys.argv[1:]
plan_path = pathlib.Path(plan_file) if plan_file else None
payload = {
    "schema_version": 2,
    "experiment": "qwen3_aclgraph_full_epoch_matrix",
    "status": "PASS",
    "policy": policy,
    "mode": mode,
    "seed": int(seed),
    "actor_frozen": True,
    "steps": 5,
    "responses_per_step": 512,
    "full16_kv_tokens": int(kv),
    "task_queue_enable": int(task_queue),
    "graph_attention": graph_attention == "true",
    "graph_moe": graph_moe == "true",
    "graph_mode": "FULL_DECODE_ONLY" if mode == "graph" else "NONE",
    "graph_capture_sizes": json.loads(sizes) if mode == "graph" else [],
    "common_epoch0_root": str(pathlib.Path(common_epoch0_root).resolve()),
    "run_dir": str(pathlib.Path(run_dir).resolve()),
    "epoch_dir": str(pathlib.Path(epoch_dir).resolve()),
    "plan_file": str(plan_path.resolve()) if plan_path else None,
    "plan_sha256": hashlib.sha256(plan_path.read_bytes()).hexdigest() if plan_path else None,
}
pathlib.Path(output).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY
    if [[ -e "$manifest" ]]; then
        cmp -s -- "$tmp" "$manifest" || { echo "arm manifest changed: $manifest" >&2; rm -f "$tmp"; return 1; }
        rm -f "$tmp"
    else
        mv -- "$tmp" "$manifest"
    fi
}

record_complete() {
    local policy="$1" mode="$2" manifest="$3"
    if [[ ! -f "$RUN_MANIFEST" ]]; then
        printf 'policy\tmode\tstatus\tarm_manifest\n' > "$RUN_MANIFEST"
    fi
    if manifest_has_arm "$policy" "$mode"; then
        return 0
    fi
    if awk -F '\t' -v p="$policy" -v m="$mode" \
        'NR>1 && $1==p && $2==m {found=1} END {exit !found}' "$RUN_MANIFEST"; then
        echo "duplicate or non-complete manifest row for $policy/$mode" >&2
        return 1
    fi
    printf '%s\t%s\tcomplete\t%s\n' "$policy" "$mode" "$manifest" >> "$RUN_MANIFEST"
}

run_common_epoch0() {
    local marker="$COMMON_EPOCH0_ROOT/DO_NOT_DELETE_COMMON_EPOCH0_CHECKPOINT"
    local common_parent common_name cache_root ray_tmp extension_sha
    if [[ -f "$marker" && -f "$COMMON_EPOCH0_ROOT/reuse.env" ]]; then
        echo "[graph/eager pair] graph Vanilla common epoch0 already complete: $COMMON_EPOCH0_ROOT"
        return 0
    fi
    if [[ -e "$COMMON_EPOCH0_ROOT" ]]; then
        echo "[graph/eager pair] incomplete common epoch0 exists; use a fresh pair root: $COMMON_EPOCH0_ROOT" >&2
        return 1
    fi
    common_parent=$(dirname "$COMMON_EPOCH0_ROOT")
    common_name=$(basename "$COMMON_EPOCH0_ROOT")
    cache_root="$PAIR_ROOT/cache/common_epoch0"
    ray_tmp="/tmp/ag_ge_common_${$}"
    extension_sha=$(sha256_file "$ASCEND_EXTENSION")
    mkdir -p "$common_parent" "$ray_tmp" \
        "$cache_root"/{xdg,hf,triton,torchair,ascend,work}
    echo "[graph/eager pair] starting trained graph Vanilla common epoch0: $COMMON_EPOCH0_ROOT"
    seal_contracts
    env \
        COMMON_EPOCH0_OUTPUT_ROOT="$common_parent" \
        COMMON_EPOCH0_RUN_NAME="$common_name" \
        COMMON_EPOCH0_TRAIN_STEPS="$STEPS" \
        COMMON_EPOCH0_TRAIN_BATCH_SIZE=32 \
        COMMON_EPOCH0_ROLLOUT_N=16 \
        COMMON_EPOCH0_MAX_NUM_SEQS=32 \
        COMMON_EPOCH0_MAX_RESPONSE_LENGTH=16384 \
        COMMON_EPOCH0_MAX_NUM_BATCHED_TOKENS=17408 \
        COMMON_EPOCH0_KV_TOKENS_PER_RANK="$GRAPH_FULL_KV_TOKENS" \
        COMMON_EPOCH0_PREEMPTION_POLICY=forbid \
        COMMON_EPOCH0_WORKLOAD_PROFILE_ID=qwen3_aclgraph_full_epoch_v1 \
        COMMON_EPOCH0_WORKLOAD_PROFILE_SHA256="$(sha256_file "$PROTOCOL_CONTRACT")" \
        COMMON_EPOCH0_EXECUTION_PROFILE=full_decode_only_tq1_balanced \
        COMMON_EPOCH0_ORIGINAL_EXECUTION_CODE_SHA256="$(sha256_file "$CODE_CONTRACT")" \
        VERL_PAIRED_REQUEST_SAMPLING_SEEDS=1 \
        VLLM_ENABLE_GRAPH_MODE=0 \
        TASK_QUEUE_ENABLE="$GRAPH_TASK_QUEUE_ENABLE" \
        ROLLOUT_ENFORCE_EAGER=False \
        VLLM_ASCEND_ELASTIC_ACLGRAPH=1 \
        VLLM_ASCEND_ELASTIC_ACLGRAPH_CAPTURE_ATTENTION=1 \
        VLLM_ASCEND_ELASTIC_ACLGRAPH_CAPTURE_MOE=1 \
        VLLM_ASCEND_ELASTIC_ACLGRAPH_EXTENSION="$(readlink -f "$ASCEND_EXTENSION")" \
        VLLM_ASCEND_ELASTIC_ACLGRAPH_EXTENSION_SHA256="$extension_sha" \
        VLLM_ASCEND_MODE1_PARITY_MAX_KV_TOKENS="$GRAPH_FULL_KV_TOKENS" \
        VERL_SIDECAR_ENABLE=0 \
        VERL_HCCL_IF_BASE_PORT_START=12000 \
        VERL_MASTER_PORT_START=28416 \
        RAY_TMPDIR="$ray_tmp" \
        XDG_CACHE_HOME="$cache_root/xdg" \
        HF_HOME="$cache_root/hf" \
        TRITON_CACHE_DIR="$cache_root/triton" \
        TORCHAIR_CACHE_HOME="$cache_root/torchair" \
        ASCEND_CACHE_PATH="$cache_root/ascend" \
        ASCEND_WORK_PATH="$cache_root/work" \
        "$SCRIPT_DIR/run_common_epoch0_probe_gpu09_kv380800_permanent.sh" \
        actor_rollout_ref.rollout.seed="$SEED" \
        actor_rollout_ref.rollout.cudagraph_mode=FULL_DECODE_ONLY \
        "actor_rollout_ref.rollout.cudagraph_capture_sizes=$CAPTURE_SIZES" \
        actor_rollout_ref.actor.megatron.override_transformer_config.moe_shared_expert_overlap=False
    seal_contracts
}

run_arm() {
    local policy="$1" mode="$2"
    local arm_root="$PAIR_ROOT/$policy/$mode"
    local run_name run_dir variant base manifest ray_tmp
    run_name=$(run_name_for_arm "$policy" "$mode")
    run_dir="$arm_root/$run_name"
    variant=$(variant_for_policy "$policy")
    base=$(base_runner_for_policy "$policy")
    manifest=$(arm_manifest_path "$policy" "$mode")
    ray_tmp="/tmp/ag_ge_${policy:0:1}${mode:0:1}_${$}"

    if manifest_has_arm "$policy" "$mode"; then
        [[ -f "$manifest" ]] || { echo "run manifest lacks arm artifact: $manifest" >&2; return 1; }
        echo "[graph/eager pair] already complete: $policy/$mode"
        return 0
    fi
    if [[ -d "$run_dir" ]]; then
        if [[ -f "$run_dir/CHECKPOINTS_REMOVED_AFTER_VALIDATION.txt" ]]; then
            echo "[graph/eager pair] recovering completed fair arm metadata: $policy/$mode"
            write_arm_manifest "$policy" "$mode" "$arm_root" "$run_dir"
            record_complete "$policy" "$mode" "$manifest"
            return 0
        fi
        echo "[graph/eager pair] incomplete arm exists; use a fresh root: $run_dir" >&2
        return 1
    fi
    mkdir -p "$arm_root" "$ray_tmp"

    common_env=(
        COMMON_EPOCH0_ROOT="$COMMON_EPOCH0_ROOT"
        FAIR_OUTPUT_ROOT="$arm_root"
        FAIR_RUN_NAME="$run_name"
        DYNAMIC_RUN_NAME="$run_name"
        FAIR_START_EPOCH=1
        FAIR_TOTAL_EPOCHS=2
        FAIR_TRAIN_STEPS="$STEPS"
        FAIR_TRAIN_BATCH_SIZE=32
        FAIR_ROLLOUT_N=16
        FAIR_MAX_RESPONSE_LENGTH=16384
        FAIR_PROMPTS_PER_EPOCH=160
        FAIR_FREEZE_ACTOR=1
        FAIR_KEEP_COMPLETED_CHECKPOINTS=0
        VERL_PAIRED_REQUEST_SAMPLING_SEEDS=1
        VLLM_ENABLE_GRAPH_MODE=0
        VERL_SIDECAR_ENABLE=0
        VERL_HCCL_IF_BASE_PORT_START=12000
        VERL_MASTER_PORT_START=28416
        RAY_TMPDIR="$ray_tmp"
    )
    hydra_args=(
        actor_rollout_ref.rollout.seed="$SEED"
        actor_rollout_ref.actor.megatron.override_transformer_config.moe_shared_expert_overlap=False
    )
    echo "[graph/eager pair] starting $policy/$mode output=$run_dir"
    seal_contracts
    if [[ "$mode" == graph ]]; then
        cache_root="$arm_root/cache"
        mkdir -p "$cache_root"/{xdg,hf,triton,torchair,ascend,work}
        env "${common_env[@]}" \
            FAIR_ADAFLOOR_TARGET="$GRAPH_WRAPPER" \
            ADAFLOOR_GRAPH_BASE_RUNNER="$base" \
            ADAFLOOR_ACLGRAPH_MODE=FULL_DECODE_ONLY \
            ADAFLOOR_GRAPH_CAPTURE_PROFILE=balanced \
            ADAFLOOR_GRAPH_CAPTURE_SIZES="$CAPTURE_SIZES" \
            VLLM_ASCEND_MODE1_PARITY_MAX_KV_TOKENS="$GRAPH_FULL_KV_TOKENS" \
            TASK_QUEUE_ENABLE="$GRAPH_TASK_QUEUE_ENABLE" \
            VLLM_ASCEND_ELASTIC_ACLGRAPH=1 \
            VLLM_ASCEND_ELASTIC_ACLGRAPH_CAPTURE_ATTENTION=1 \
            VLLM_ASCEND_ELASTIC_ACLGRAPH_CAPTURE_MOE=1 \
            VLLM_ASCEND_ELASTIC_ACLGRAPH_EXTENSION="$(readlink -f "$ASCEND_EXTENSION")" \
            VLLM_ASCEND_ELASTIC_ACLGRAPH_EXTENSION_SHA256="$(sha256_file "$ASCEND_EXTENSION")" \
            ROLLOUT_ENFORCE_EAGER=False \
            XDG_CACHE_HOME="$cache_root/xdg" \
            HF_HOME="$cache_root/hf" \
            TRITON_CACHE_DIR="$cache_root/triton" \
            TORCHAIR_CACHE_HOME="$cache_root/torchair" \
            ASCEND_CACHE_PATH="$cache_root/ascend" \
            ASCEND_WORK_PATH="$cache_root/work" \
            "$SCRIPT_DIR/run_paper_fair_epoch1_2_from_common_epoch0.sh" \
            "$variant" \
            actor_rollout_ref.rollout.cudagraph_mode=FULL_DECODE_ONLY \
            "actor_rollout_ref.rollout.cudagraph_capture_sizes=$CAPTURE_SIZES" \
            "${hydra_args[@]}"
    else
        env "${common_env[@]}" \
            FAIR_ADAFLOOR_TARGET="$base" \
            VLLM_ASCEND_MODE1_PARITY_MAX_KV_TOKENS="$EAGER_FULL_KV_TOKENS" \
            TASK_QUEUE_ENABLE="$EAGER_TASK_QUEUE_ENABLE" \
            VLLM_ASCEND_ELASTIC_ACLGRAPH=0 \
            VLLM_ASCEND_ELASTIC_ACLGRAPH_CAPTURE_ATTENTION=0 \
            VLLM_ASCEND_ELASTIC_ACLGRAPH_CAPTURE_MOE=0 \
            ROLLOUT_ENFORCE_EAGER=True \
            "$SCRIPT_DIR/run_paper_fair_epoch1_2_from_common_epoch0.sh" \
            "$variant" "${hydra_args[@]}"
    fi
    seal_contracts
    write_arm_manifest "$policy" "$mode" "$arm_root" "$run_dir"
    record_complete "$policy" "$mode" "$manifest"
    python3 "$VERIFY_TOOL" --root "$PAIR_ROOT" --allow-incomplete
}

require_inputs
seal_contracts

if [[ "$action" == dry-run ]]; then
    echo "root=$PAIR_ROOT"
    echo "common_epoch0=$COMMON_EPOCH0_ROOT mode=graph_vanilla steps=$STEPS"
    while read -r policy mode; do
        echo "arm=$policy/$mode run_name=$(run_name_for_arm "$policy" "$mode") variant=$(variant_for_policy "$policy") base=$(base_runner_for_policy "$policy")"
    done < <(arm_order)
    exit 0
fi
if [[ "$action" == common ]]; then
    run_common_epoch0
    exec python3 "$VERIFY_TOOL" --root "$PAIR_ROOT" --allow-incomplete
fi
if [[ "$action" == verify ]]; then
    exec python3 "$VERIFY_TOOL" --root "$PAIR_ROOT"
fi

run_common_epoch0
python3 "$VERIFY_TOOL" --root "$PAIR_ROOT" --allow-incomplete
while read -r policy mode; do
    run_arm "$policy" "$mode"
done < <(arm_order)
seal_contracts
exec python3 "$VERIFY_TOOL" --root "$PAIR_ROOT"
