#!/usr/bin/env bash
# One actual DeepScaleR RL update with bounded same-model GSM8K tail inference.
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$ROOT"
ACTION=${1:-dry-run}
case "$ACTION" in dry-run|run) shift || true ;; *) exit 2 ;; esac
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
export TAILSERVE_RUN_ID=${TAILSERVE_RUN_ID:-deepscaler_gsm8k_$(date -u +%Y%m%dT%H%M%SZ)}
export OUTPUT_ROOT=${OUTPUT_ROOT:-/data/tailserve_results}
export OUTPUT_SUBDIR=$TAILSERVE_RUN_ID
export TAILSERVE_SPOOL=${TAILSERVE_SPOOL:-$OUTPUT_ROOT/${TAILSERVE_RUN_ID}_spool}
export TRAIN_FILE=/data/deepscaler/train.parquet
export TEST_FILE=/data/deepscaler/test.parquet
# Resolve these before preparing queues, using the same values as the engine
# launcher. A standalone queue producer must not silently keep old short caps.
export MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-1024}
export MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-16384}
export ROLLOUT_MAX_NUM_BATCHED_TOKENS=${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-17408}
export TAILSERVE_PER_RANK_TOKENS=${TAILSERVE_PER_RANK_TOKENS:-$ROLLOUT_MAX_NUM_BATCHED_TOKENS}
export TAILSERVE_MAX_AUX_RANKS=${TAILSERVE_MAX_AUX_RANKS:-16}
export TAILSERVE_GLOBAL_TOKENS=${TAILSERVE_GLOBAL_TOKENS:-$((TAILSERVE_MAX_AUX_RANKS * TAILSERVE_PER_RANK_TOKENS))}
export TAILSERVE_MAX_PROMPT_TOKENS=${TAILSERVE_MAX_PROMPT_TOKENS:-$MAX_PROMPT_LENGTH}
export TAILSERVE_MAX_NEW_TOKENS=${TAILSERVE_MAX_NEW_TOKENS:-$MAX_RESPONSE_LENGTH}
export TAILSERVE_RL_STEPS=${TAILSERVE_RL_STEPS:-1}
export SAVE_CKPT_ENABLE=0
export GIT_TRACK_RUN_LOGS=0
if [[ "$ACTION" == dry-run ]]; then
    echo "GSM8K: first ${GSM8K_REQUESTS:-256} rows, round-robin EP ranks, no length selection"
    echo 'Stop condition: global RL request count zero; abort auxiliary immediately, never wait for its EOS'
    exec bash "$ROOT/run_tailserve.sh" dry-run "$@"
fi
[[ ! -e "$OUTPUT_ROOT/$OUTPUT_SUBDIR" ]] || { echo 'run directory already exists' >&2; exit 2; }
[[ ! -e "$TAILSERVE_SPOOL" ]] || { echo 'spool already exists' >&2; exit 2; }
python tools/prepare_gsm8k_queue.py --spool "$TAILSERVE_SPOOL" \
    --tokenizer "${MODEL_PATH:-/data/Qwen3-30B-A3B}" \
    --count "${GSM8K_REQUESTS:-256}" --max-tokens "$TAILSERVE_MAX_NEW_TOKENS"
set +e
bash "$ROOT/run_tailserve.sh" run "$@"
rc=$?
set -e
python tools/summarize_tailserve.py --spool "$TAILSERVE_SPOOL" \
    --rl-output "$OUTPUT_ROOT/$OUTPUT_SUBDIR" --save "$TAILSERVE_SPOOL/report.json"
echo "[TailServe] training_exit_code=$rc report=$TAILSERVE_SPOOL/report.json"
exit "$rc"
