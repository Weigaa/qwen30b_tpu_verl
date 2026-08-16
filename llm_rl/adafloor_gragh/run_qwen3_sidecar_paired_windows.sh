#!/usr/bin/env bash
set -euo pipefail

if [[ "${ADAFLOOR_SIDECAR_PAIR_SNAPSHOT_ACTIVE:-0}" != "1" ]]; then
    source_path=$(realpath "${BASH_SOURCE[0]}")
    snapshot=$(mktemp "${source_path}.run-snapshot.XXXXXX")
    cp -- "$source_path" "$snapshot"
    chmod 700 "$snapshot"
    set +e
    ADAFLOOR_SIDECAR_PAIR_SNAPSHOT_ACTIVE=1 "$snapshot" "$@"
    rc=$?
    set -e
    rm -f -- "$snapshot"
    exit "$rc"
fi

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR"

COMMON_EPOCH0_ROOT="${COMMON_EPOCH0_ROOT:-/data/adafloor_shared_state/common_epoch0_probe_gpu09_kv380800_permanent}"
SIDECAR_PAIR_ROOT="${SIDECAR_PAIR_ROOT:-/data/adafloor_shared_state/qwen3_sidecar_paired_windows_$(date -u +%Y%m%dT%H%M%SZ)}"
SIDECAR_PAIR_SEEDS="${SIDECAR_PAIR_SEEDS:-101 202 303}"
SIDECAR_PAIR_MIN_FREE_GIB="${SIDECAR_PAIR_MIN_FREE_GIB:-100}"
SIDECAR_MODEL="${SIDECAR_MODEL:-/data/Qwen2.5-1.5B-Instruct}"
SIDECAR_DATA="${SIDECAR_DATA:-/data/gsm8k}"
SIDECAR_MODEL_REVISION="${SIDECAR_MODEL_REVISION:-a3c2dc17129625b1e51caf21ab486d32d1f12982}"
SIDECAR_MODEL_WEIGHTS_SHA256="dd924a11b4c220f385b51ffa522daea7c9f3d850e31b162bb5661df483c6d3ee"
SIDECAR_DATASET_SHA256="2c55b5db39cdd7ac17dbb7e1f64d7f8d04b81058702146d65047a9b36a581a75"
PRIMARY_HCCL_ALLOCATOR_START=12000
MANIFEST="$SIDECAR_PAIR_ROOT/run_manifest.tsv"
CODE_CONTRACT="$SIDECAR_PAIR_ROOT/code_sha256.txt"
PROTOCOL_CONTRACT="$SIDECAR_PAIR_ROOT/protocol.env"

CODE_CONTRACT_PATHS=(
    "$SCRIPT_DIR/run_qwen3_sidecar_paired_windows.sh"
    "$SCRIPT_DIR/run_paper_fair_epoch1_2_from_common_epoch0.sh"
    "$SCRIPT_DIR/run_mode1_dynamic_length_aware_adaptive_floor4_planned_full3.sh"
    "$SCRIPT_DIR/run_mode1_dynamic_length_aware_adaptive_floor4_epochs.sh"
    "$SCRIPT_DIR/run_mode1_local_length_sorted_e2e_adaptive_floor4.sh"
    "$SCRIPT_DIR/internal/wj_train_grpo_qwen30b_a3b_16die_true_weight_regroup.sh"
    "$SCRIPT_DIR/internal/watch_elastic_shrink_and_run_sidecar.sh"
    "$SCRIPT_DIR/internal/run_elastic_sidecar_infer.sh"
    "$SCRIPT_DIR/tools/build_mode1_length_sorted_e2e_plan.py"
    "$SCRIPT_DIR/tools/build_mode1_optimized_rank_plan.py"
    "$SCRIPT_DIR/tools/verify_qwen3_sidecar_pairs.py"
    "$SCRIPT_DIR/verl/experimental/dataset/shrink_aware_assignment.py"
    "$SCRIPT_DIR/verl/utils/rollout_seeding.py"
    "$SCRIPT_DIR/verl/utils/sidecar_restore_handshake.py"
    "$SCRIPT_DIR/verl/trainer/ppo/ray_trainer.py"
    "$SCRIPT_DIR/verl/single_controller/ray/base.py"
    "$SCRIPT_DIR/verl/workers/megatron_workers.py"
    "$SCRIPT_DIR/verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py"
    "$SCRIPT_DIR/vllm/v1/engine/llm_engine.py"
    "$SCRIPT_DIR/vllm_ascend/worker/worker_v1.py"
)

install_or_verify_immutable_file() {
    local candidate="$1"
    local destination="$2"
    local label="$3"

    if [[ -e "$destination" ]]; then
        if [[ ! -f "$destination" || -L "$destination" ]] \
           || ! cmp -s -- "$candidate" "$destination"; then
            echo "[sidecar pair] immutable $label differs from the existing experiment root: $destination" >&2
            rm -f -- "$candidate"
            return 1
        fi
        rm -f -- "$candidate"
        return 0
    fi

    # A same-filesystem hard link publishes the fully written candidate without
    # replacing a contract another process may have installed concurrently.
    if ln -- "$candidate" "$destination" 2>/dev/null; then
        rm -f -- "$candidate"
        return 0
    fi
    if [[ -f "$destination" && ! -L "$destination" ]] \
       && cmp -s -- "$candidate" "$destination"; then
        rm -f -- "$candidate"
        return 0
    fi
    echo "[sidecar pair] could not atomically install immutable $label: $destination" >&2
    rm -f -- "$candidate"
    return 1
}

root_has_arm_state() {
    if find "$SIDECAR_PAIR_ROOT" -mindepth 1 -maxdepth 1 -type d \
            -name 'seed_*' -print -quit | grep -q .; then
        return 0
    fi
    [[ -f "$MANIFEST" ]] && (( $(wc -l < "$MANIFEST") > 1 ))
}

protocol_created_at() {
    if [[ ! -e "$PROTOCOL_CONTRACT" ]]; then
        date -u +%Y-%m-%dT%H:%M:%SZ
        return 0
    fi
    [[ -f "$PROTOCOL_CONTRACT" && ! -L "$PROTOCOL_CONTRACT" ]] || {
        echo "[sidecar pair] invalid immutable protocol path: $PROTOCOL_CONTRACT" >&2
        return 1
    }
    local values=()
    mapfile -t values < <(
        sed -n 's/^created_at_utc=//p' "$PROTOCOL_CONTRACT"
    )
    if (( ${#values[@]} != 1 )) \
       || [[ ! "${values[0]}" =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z$ ]]; then
        echo "[sidecar pair] immutable protocol has an invalid created_at_utc" >&2
        return 1
    fi
    printf '%s\n' "${values[0]}"
}

seal_or_verify_root_contracts() {
    mkdir -p "$SIDECAR_PAIR_ROOT"
    if root_has_arm_state \
       && { [[ ! -f "$CODE_CONTRACT" || -L "$CODE_CONTRACT" ]] \
            || [[ ! -f "$PROTOCOL_CONTRACT" || -L "$PROTOCOL_CONTRACT" ]]; }; then
        echo "[sidecar pair] existing arm state is missing an immutable root contract" >&2
        return 1
    fi

    local code_candidate protocol_candidate created_at
    code_candidate=$(mktemp "$SIDECAR_PAIR_ROOT/.code_sha256.txt.XXXXXX")
    protocol_candidate=$(mktemp "$SIDECAR_PAIR_ROOT/.protocol.env.XXXXXX")
    if ! sha256sum "${CODE_CONTRACT_PATHS[@]}" > "$code_candidate"; then
        rm -f -- "$code_candidate" "$protocol_candidate"
        return 1
    fi
    if ! created_at=$(protocol_created_at); then
        rm -f -- "$code_candidate" "$protocol_candidate"
        return 1
    fi
    {
        printf 'created_at_utc=%s\n' "$created_at"
        printf 'common_epoch0_root=%s\n' "$COMMON_EPOCH0_ROOT"
        printf 'seeds=%s\n' "$SIDECAR_PAIR_SEEDS"
        printf 'orders=101:off,on 202:on,off 303:off,on\n'
        printf 'planned_residency=true\n'
        printf 'floor=floor4\n'
        printf 'tail_guard=false\n'
        printf 'actor_frozen=true\n'
        printf 'paired_request_sampling_seeds=true\n'
        printf 'planner_prompts=160\n'
        printf 'plan_steps=5\n'
        printf 'executed_prompts=32\n'
        printf 'steps_per_run=1\n'
        printf 'fast_step_subset=false\n'
        printf 'source_plan_step=1\n'
        printf 'sidecar_trigger_active_ranks=8\n'
        printf 'sidecar_model=%s\n' "$SIDECAR_MODEL"
        printf 'sidecar_model_revision=%s\n' "$SIDECAR_MODEL_REVISION"
        printf 'sidecar_data=%s/train.parquet\n' "$SIDECAR_DATA"
        printf 'sidecar_parallelism=TP1x8\n'
        printf 'sidecar_stop_ack_timeout_seconds=60\n'
        printf 'sidecar_require_active_lease_before_restore=true\n'
        printf 'sidecar_require_shrink_quorum=true\n'
        printf 'sidecar_shrink_quorum_size=16\n'
        printf 'eager_weight_sync_group_init=false\n'
        printf 'primary_hccl_allocator_start=%s\n' "$PRIMARY_HCCL_ALLOCATOR_START"
        printf 'primary_moe_shared_expert_overlap=false\n'
    } > "$protocol_candidate"

    install_or_verify_immutable_file \
        "$code_candidate" "$CODE_CONTRACT" "code hash manifest" || {
        rm -f -- "$protocol_candidate"
        return 1
    }
    install_or_verify_immutable_file \
        "$protocol_candidate" "$PROTOCOL_CONTRACT" "protocol manifest"
}

usage() {
    cat <<'EOF'
Usage:
  ./run_qwen3_sidecar_paired_windows.sh [all|dry-run|verify]

Runs three prespecified one-window pairs of Planned residency floor4 without
TailGuard. Each arm executes original planner step 1 and starts eight
Qwen2.5-1.5B replicas on the first eight ranks released by the 16-to-8
transition. The only within-pair difference is sidecar execution.
EOF
}

action="${1:-all}"
case "$action" in
    all|dry-run|verify) ;;
    -h|--help) usage; exit 0 ;;
    *) usage >&2; exit 2 ;;
esac

require_inputs() {
    local required
    for required in \
        "$COMMON_EPOCH0_ROOT/DO_NOT_DELETE_COMMON_EPOCH0_CHECKPOINT" \
        "$COMMON_EPOCH0_ROOT/reuse.env" \
        "$SIDECAR_MODEL/config.json" \
        "$SIDECAR_MODEL/tokenizer.json" \
        "$SIDECAR_MODEL/model.safetensors" \
        "$SIDECAR_DATA/train.parquet"; do
        [[ -f "$required" ]] || {
            echo "missing required input: $required" >&2
            exit 2
        }
    done
    [[ "$SIDECAR_PAIR_MIN_FREE_GIB" =~ ^[0-9]+$ ]] || {
        echo "SIDECAR_PAIR_MIN_FREE_GIB must be a nonnegative integer" >&2
        exit 2
    }
    [[ "$SIDECAR_PAIR_SEEDS" == "101 202 303" ]] || {
        echo "the paper protocol requires SIDECAR_PAIR_SEEDS='101 202 303'" >&2
        exit 2
    }
    [[ "$(sha256sum "$SIDECAR_MODEL/model.safetensors" | awk '{print $1}')" == \
        "$SIDECAR_MODEL_WEIGHTS_SHA256" ]] || {
        echo "sidecar model weights do not match the frozen protocol" >&2
        exit 2
    }
    [[ "$(sha256sum "$SIDECAR_DATA/train.parquet" | awk '{print $1}')" == \
        "$SIDECAR_DATASET_SHA256" ]] || {
        echo "sidecar dataset does not match the frozen protocol" >&2
        exit 2
    }
    local allocated_hccl_base
    if ! allocated_hccl_base=$(
        HCCL_IF_BASE_PORT="$PRIMARY_HCCL_ALLOCATOR_START" \
        VERL_HCCL_IF_BASE_PORT_START="$PRIMARY_HCCL_ALLOCATOR_START" \
        python3 - <<'PY'
from verl.single_controller.ray.base import _alloc_hccl_if_base_port

print(_alloc_hccl_if_base_port())
PY
    ); then
        echo "primary HCCL allocator preflight failed" >&2
        exit 2
    fi
    [[ "$allocated_hccl_base" == "$PRIMARY_HCCL_ALLOCATOR_START" ]] || {
        echo "primary HCCL allocator selected unexpected base $allocated_hccl_base" >&2
        exit 2
    }
}

arm_order_for_seed() {
    case "$1" in
        101) printf '%s\n' off on ;;
        202) printf '%s\n' on off ;;
        303) printf '%s\n' off on ;;
        *) return 1 ;;
    esac
}

arm_run_name() {
    printf 'primary_planned_f4_noguard_seed%s_%s' "$1" "$2"
}

record_status() {
    local seed="$1" arm="$2" order="$3" run_dir="$4" status="$5"
    local tmp="$MANIFEST.tmp.$$"
    local rows="$MANIFEST.rows.$$"
    awk -F '\t' -v seed="$seed" -v arm="$arm" \
        'NR > 1 && !($1 == seed && $2 == arm)' "$MANIFEST" > "$rows"
    printf '%s\t%s\t%s\t%s\t%s\n' \
        "$seed" "$arm" "$order" "$run_dir" "$status" >> "$rows"
    {
        printf 'seed\tarm\tlaunch_order\trun_dir\tstatus\n'
        sort -t $'\t' -k3,3n "$rows"
    } > "$tmp"
    mv -f -- "$tmp" "$MANIFEST"
    rm -f -- "$rows"
}

check_disk() {
    local free_kib required_kib
    free_kib=$(df --output=avail -k "$SIDECAR_PAIR_ROOT" | tail -1 | tr -d '[:space:]')
    required_kib=$((SIDECAR_PAIR_MIN_FREE_GIB * 1024 * 1024))
    if ! [[ "$free_kib" =~ ^[0-9]+$ ]] || (( free_kib < required_kib )); then
        echo "insufficient disk: free_kib=${free_kib:-unknown} required_kib=$required_kib" >&2
        return 1
    fi
}

write_arm_manifest() {
    local seed="$1" arm="$2" order="$3" arm_root="$4" run_dir="$5"
    local epoch_dir plan_path plan_sha cleanup_path
    cleanup_path="$run_dir/CHECKPOINTS_REMOVED_AFTER_VALIDATION.txt"
    epoch_dir="$run_dir/epoch_001_mode1_planned"
    plan_path="$epoch_dir/oracle/length_sorted_rank_plan.json"
    [[ -f "$cleanup_path" && -f "$plan_path" ]] || return 1
    plan_sha=$(sha256sum "$plan_path" | awk '{print $1}')
    python3 - \
        "$arm_root/sidecar_pair_manifest.json" "$seed" "$arm" "$order" \
        "$run_dir" "$epoch_dir" "$plan_path" "$plan_sha" \
        "$SIDECAR_MODEL" "$SIDECAR_MODEL_REVISION" \
        "$SIDECAR_MODEL_WEIGHTS_SHA256" \
        "$SIDECAR_DATA/train.parquet" "$PRIMARY_HCCL_ALLOCATOR_START" <<'PY'
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

(
    output_arg,
    seed_arg,
    arm,
    order_arg,
    run_arg,
    epoch_arg,
    plan_arg,
    plan_sha,
    model_arg,
    model_revision,
    model_weights_sha,
    data_arg,
    primary_hccl_allocator_start,
) = sys.argv[1:]

def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()

output = Path(output_arg)
payload = {
    "schema_version": 1,
    "experiment": "qwen2_5_1_5b_planned_floor4_noguard_sidecar_pair",
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
    "seed": int(seed_arg),
    "request_seed": int(seed_arg),
    "arm": arm,
    "launch_order": int(order_arg),
    "planned": True,
    "planned_residency": True,
    "target_floor": 4,
    "tail_guard_enabled": False,
    "expected_responses": 512,
    "expected_primary_responses": 512,
    "primary_prompts": 32,
    "planner_prompts": 160,
    "planner_steps": 5,
    "executed_steps": 1,
    "fast_step_subset": False,
    "source_plan_step": 1,
    "responses_per_prompt": 16,
    "actor_frozen": True,
    "paired_request_sampling_seeds": True,
    "sidecar_enabled": arm == "on",
    "sidecar_model": str(Path(model_arg).resolve()),
    "sidecar_model_path": str(Path(model_arg).resolve()),
    "sidecar_model_revision": model_revision,
    "sidecar_model_weights_sha256": model_weights_sha,
    "sidecar_tensor_parallel_size": 1,
    "sidecar_replica_count": 8,
    "sidecar_trigger_active_ranks": 8,
    "sidecar_data_split": "train",
    "sidecar_temperature": 0.0,
    "sidecar_top_p": 1.0,
    "sidecar_max_tokens": 4096,
    "sidecar_stop_ack_timeout_seconds": 60,
    "sidecar_require_active_lease_before_restore": True,
    "sidecar_require_shrink_quorum": True,
    "sidecar_shrink_quorum_size": 16,
    "eager_weight_sync_group_init": False,
    "primary_hccl_allocator_start": int(primary_hccl_allocator_start),
    "primary_moe_shared_expert_overlap": False,
    "run_dir": str(Path(run_arg).resolve()),
    "epoch_dir": str(Path(epoch_arg).resolve()),
    "plan_file": str(Path(plan_arg).resolve().relative_to(output.parent.resolve())),
    "plan_path": str(Path(plan_arg).resolve()),
    "plan_sha256": plan_sha,
    "model_config_sha256": digest(Path(model_arg) / "config.json"),
    "model_tokenizer_sha256": digest(Path(model_arg) / "tokenizer.json"),
    "dataset_path": str(Path(data_arg).resolve()),
    "sidecar_dataset_path": str(Path(data_arg).resolve()),
    "sidecar_dataset_split": "train",
    "dataset_sha256": digest(Path(data_arg)),
}
temporary = output.with_suffix(output.suffix + ".tmp")
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
temporary.replace(output)
PY
}

run_arm() {
    local seed="$1" arm="$2" order="$3"
    local seed_root arm_root run_name run_dir cleanup_record enable_sidecar
    seed_root="$SIDECAR_PAIR_ROOT/seed_$seed"
    arm_root="$seed_root/$arm"
    run_name=$(arm_run_name "$seed" "$arm")
    run_dir="$arm_root/$run_name"
    cleanup_record="$run_dir/CHECKPOINTS_REMOVED_AFTER_VALIDATION.txt"
    enable_sidecar=0
    [[ "$arm" == "on" ]] && enable_sidecar=1

    mkdir -p "$arm_root"
    if [[ -f "$cleanup_record" ]] \
       && grep -qFx 'validated_epochs=001' "$cleanup_record" \
       && [[ -f "$arm_root/sidecar_pair_manifest.json" ]]; then
        echo "[sidecar pair] already complete seed=$seed arm=$arm"
        record_status "$seed" "$arm" "$order" "$run_dir" complete
        return 0
    fi
    if [[ -e "$run_dir" ]]; then
        echo "incomplete existing arm requires inspection: $run_dir" >&2
        record_status "$seed" "$arm" "$order" "$run_dir" incomplete
        return 3
    fi

    check_disk
    record_status "$seed" "$arm" "$order" "$run_dir" running
    echo "[sidecar pair] start seed=$seed arm=$arm order=$order output=$run_dir"

    COMMON_EPOCH0_ROOT="$COMMON_EPOCH0_ROOT" \
    FAIR_OUTPUT_ROOT="$arm_root" \
    DYNAMIC_RUN_NAME="$run_name" \
    FAIR_START_EPOCH=1 \
    FAIR_TOTAL_EPOCHS=2 \
    FAIR_PROMPTS_PER_EPOCH=160 \
    FAIR_TRAIN_BATCH_SIZE=32 \
    FAIR_TRAIN_STEPS=1 \
    FAIR_ROLLOUT_N=16 \
    FAIR_MAX_RESPONSE_LENGTH=16384 \
    FAIR_FREEZE_ACTOR=1 \
    FAIR_KEEP_COMPLETED_CHECKPOINTS=0 \
    VERL_PAIRED_REQUEST_SAMPLING_SEEDS=1 \
    DYNAMIC_DISABLE_TAIL_GUARD=1 \
    DYNAMIC_EXPECT_NO_RESPONSE_CAPS=1 \
    DYNAMIC_SHORT_STEP_CAP_ENABLE=0 \
    VERL_SIDECAR_ENABLE="$enable_sidecar" \
    VERL_SIDECAR_START_ONCE=1 \
    VERL_SIDECAR_STOP_ACK_TIMEOUT_SECONDS=60 \
    VERL_SIDECAR_REQUIRE_ACTIVE_LEASE_BEFORE_RESTORE=1 \
    VERL_SIDECAR_REQUIRE_SHRINK_QUORUM=1 \
    VERL_SIDECAR_SHRINK_QUORUM_SIZE=16 \
    VERL_SIDECAR_WORLD_SIZE=16 \
    VERL_SIDECAR_EXPECTED_ACTIVE_RANKS=8 \
    VERL_SIDECAR_MODEL_PATH="$SIDECAR_MODEL" \
    VERL_SIDECAR_PROMPTS_FILE="$SIDECAR_DATA" \
    VERL_SIDECAR_DATA_SPLIT=train \
    VERL_SIDECAR_PARALLEL_MODE=dp \
    VERL_SIDECAR_TENSOR_PARALLEL_SIZE=1 \
    VERL_SIDECAR_DATA_PARALLEL_SIZE=1 \
    VERL_SIDECAR_REPLICA_COUNT=8 \
    VERL_SIDECAR_ENABLE_EXPERT_PARALLEL=0 \
    VERL_SIDECAR_N=1 \
    VERL_SIDECAR_TEMPERATURE=0.0 \
    VERL_SIDECAR_TOP_P=1.0 \
    VERL_SIDECAR_GPU_MEMORY_UTILIZATION=0.80 \
    VERL_SIDECAR_MAX_MODEL_LEN=6144 \
    VERL_SIDECAR_MAX_TOKENS=4096 \
    VERL_SIDECAR_MAX_NUM_SEQS=128 \
    VERL_SIDECAR_MAX_NUM_BATCHED_TOKENS=65536 \
    VERL_SIDECAR_MAX_PROMPTS=1024 \
    VERL_SIDECAR_MAX_PROMPTS_PER_REPLICA=128 \
    VERL_SIDECAR_GENERATE_CHUNK_SIZE=128 \
    VERL_SIDECAR_REPEAT_UNTIL_KILLED=1 \
    VERL_SIDECAR_STREAM_CHECKPOINT=1 \
    VERL_SIDECAR_RESET_OUTPUT_ON_START=0 \
    VERL_MEGATRON_EAGER_WEIGHT_SYNC_GROUP_INIT=0 \
    HCCL_IF_BASE_PORT="$PRIMARY_HCCL_ALLOCATOR_START" \
    VERL_HCCL_IF_BASE_PORT_START="$PRIMARY_HCCL_ALLOCATOR_START" \
    "$SCRIPT_DIR/run_paper_fair_epoch1_2_from_common_epoch0.sh" \
        adafloor_p_f4 \
        "actor_rollout_ref.rollout.seed=$seed" \
        "actor_rollout_ref.actor.megatron.override_transformer_config.moe_shared_expert_overlap=False"

    # Reject an arm whose execution overlapped a code or protocol change.
    seal_or_verify_root_contracts
    write_arm_manifest "$seed" "$arm" "$order" "$arm_root" "$run_dir"
    record_status "$seed" "$arm" "$order" "$run_dir" complete
}

verify_results() {
    python3 "$SCRIPT_DIR/tools/verify_qwen3_sidecar_pairs.py" \
        --root "$SIDECAR_PAIR_ROOT" "$@"
}

require_inputs

if [[ "$action" == "verify" ]]; then
    verify_results
    exit 0
fi

if [[ "$action" == "dry-run" ]]; then
    COMMON_EPOCH0_ROOT="$COMMON_EPOCH0_ROOT" \
    FAIR_OUTPUT_ROOT="$SIDECAR_PAIR_ROOT/dry_run" \
    DYNAMIC_RUN_NAME=primary_planned_f4_noguard_dry_run \
    FAIR_START_EPOCH=1 FAIR_TOTAL_EPOCHS=2 \
    FAIR_PROMPTS_PER_EPOCH=160 FAIR_TRAIN_BATCH_SIZE=32 FAIR_TRAIN_STEPS=1 \
    FAIR_ROLLOUT_N=16 \
    FAIR_MAX_RESPONSE_LENGTH=16384 FAIR_FREEZE_ACTOR=1 \
    VERL_PAIRED_REQUEST_SAMPLING_SEEDS=1 \
    DYNAMIC_DISABLE_TAIL_GUARD=1 DYNAMIC_EXPECT_NO_RESPONSE_CAPS=1 \
    DYNAMIC_SHORT_STEP_CAP_ENABLE=0 VERL_SIDECAR_ENABLE=1 \
    VERL_SIDECAR_REQUIRE_SHRINK_QUORUM=1 \
    VERL_SIDECAR_SHRINK_QUORUM_SIZE=16 VERL_SIDECAR_WORLD_SIZE=16 \
    VERL_SIDECAR_EXPECTED_ACTIVE_RANKS=8 FAIR_DRY_RUN=1 \
    VERL_MEGATRON_EAGER_WEIGHT_SYNC_GROUP_INIT=0 \
    HCCL_IF_BASE_PORT="$PRIMARY_HCCL_ALLOCATOR_START" \
    VERL_HCCL_IF_BASE_PORT_START="$PRIMARY_HCCL_ALLOCATOR_START" \
    "$SCRIPT_DIR/run_paper_fair_epoch1_2_from_common_epoch0.sh" \
        adafloor_p_f4 \
        actor_rollout_ref.rollout.seed=101 \
        actor_rollout_ref.actor.megatron.override_transformer_config.moe_shared_expert_overlap=False
    exit 0
fi

mkdir -p "$SIDECAR_PAIR_ROOT"
seal_or_verify_root_contracts
if [[ ! -f "$MANIFEST" ]]; then
    printf 'seed\tarm\tlaunch_order\trun_dir\tstatus\n' > "$MANIFEST"
fi

launch_order=0
for seed in $SIDECAR_PAIR_SEEDS; do
    while IFS= read -r arm; do
        launch_order=$((launch_order + 1))
        seal_or_verify_root_contracts
        run_arm "$seed" "$arm" "$launch_order"
    done < <(arm_order_for_seed "$seed")
    if [[ -f "$SCRIPT_DIR/tools/verify_qwen3_sidecar_pairs.py" ]]; then
        seal_or_verify_root_contracts
        verify_results --allow-incomplete
    fi
done

seal_or_verify_root_contracts
verify_results
echo "[sidecar pair] complete summary=$SIDECAR_PAIR_ROOT/sidecar_pair_summary.md"
