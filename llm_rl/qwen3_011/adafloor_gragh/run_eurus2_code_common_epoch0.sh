#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR"

WORKLOAD_DIR="${EURUS2_CODE_WORKLOAD_DIR:-/data/eurus2_rl_code/validation_code_paired_160}"
TRAIN_FILE="$WORKLOAD_DIR/train.parquet"
TEST_FILE="$WORKLOAD_DIR/test.parquet"
COMMON_OUTPUT_ROOT="${EURUS2_COMMON_OUTPUT_ROOT:-/data/adafloor_shared_state}"
COMMON_RUN_NAME="${EURUS2_COMMON_RUN_NAME:-common_epoch0_eurus2_code_validation_frozen_gpu09_kv380800_permanent}"
COMMON_ROOT="$COMMON_OUTPUT_ROOT/$COMMON_RUN_NAME"
ROLLOUT_SEED="${EURUS2_ROLLOUT_SEED:-401}"

if [[ ! -f "$WORKLOAD_DIR/manifest.json" \
      || ! -f "$TRAIN_FILE" || ! -f "$TEST_FILE" ]]; then
    "$SCRIPT_DIR/run_prepare_eurus2_code_workload.sh"
fi

if [[ -f "$COMMON_ROOT/DO_NOT_DELETE_COMMON_EPOCH0_CHECKPOINT" \
      && -f "$COMMON_ROOT/reuse.env" ]]; then
    echo "[eurus2 code epoch0] complete state already exists: $COMMON_ROOT"
    exit 0
fi
if [[ -e "$COMMON_ROOT" ]]; then
    echo "[eurus2 code epoch0] incomplete destination exists: $COMMON_ROOT" >&2
    echo "Remove only that incomplete code-specific directory before retrying." >&2
    exit 2
fi

echo "[eurus2 code epoch0] frozen actor history capture"
echo "[eurus2 code epoch0] workload=$WORKLOAD_DIR output=$COMMON_ROOT seed=$ROLLOUT_SEED"

VERL_PAIRED_REQUEST_SAMPLING_SEEDS=1 \
COMMON_EPOCH0_OUTPUT_ROOT="$COMMON_OUTPUT_ROOT" \
COMMON_EPOCH0_RUN_NAME="$COMMON_RUN_NAME" \
COMMON_EPOCH0_DATASET_FRACTION=1.0 \
COMMON_EPOCH0_MIN_FREE_GB="${EURUS2_COMMON_MIN_FREE_GB:-100}" \
TRAIN_FILE="$TRAIN_FILE" \
TEST_FILE="$TEST_FILE" \
"$SCRIPT_DIR/run_common_epoch0_probe_gpu09_kv380800_permanent.sh" \
    custom_reward_function.path=null \
    reward_model.reward_manager=prime \
    actor_rollout_ref.actor.optim.lr=0.0 \
    actor_rollout_ref.rollout.seed="$ROLLOUT_SEED"

cp -- "$WORKLOAD_DIR/manifest.json" "$COMMON_ROOT/eurus2_code_workload_manifest.json"
cat > "$COMMON_ROOT/FROZEN_CODE_EPOCH0_PROTOCOL.txt" <<EOF
dataset=PRIME-RL/Eurus-2-RL-Data
split=validation
ability=code
prompts=160
steps=5
prompts_per_step=32
responses_per_prompt=16
actor_learning_rate=0.0
reward_manager=prime
custom_reward_function=null
paired_request_sampling_seeds=1
rollout_seed=$ROLLOUT_SEED
EOF
echo "[eurus2 code epoch0] complete: $COMMON_ROOT"
