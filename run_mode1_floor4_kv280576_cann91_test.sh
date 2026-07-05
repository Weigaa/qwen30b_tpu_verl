#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR"

VENV_PATH="${VENV_PATH:-$SCRIPT_DIR/.venv_cann91_a3_torch29}"
CANN91_ENV="${CANN91_ENV:-/workspace/cann-recipes-train/llm_rl/qwen3/use_cann91_a3_with_nnal91.sh}"

if [[ ! -f "$VENV_PATH/bin/activate" ]]; then
    echo "missing venv: $VENV_PATH" >&2
    exit 2
fi
if [[ ! -f "$CANN91_ENV" ]]; then
    echo "missing CANN 9.1 A3 env script: $CANN91_ENV" >&2
    exit 2
fi

source "$VENV_PATH/bin/activate"
source "$CANN91_ENV"

export USE_CANN91_A3_WITH_NNAL91=1
export ANALYZE_MODE1_THRESHOLD_LOG="${ANALYZE_MODE1_THRESHOLD_LOG:-1}"

# Default to the 17,408-token-request / 16.12x validation budget, but allow
# quick A/B runs with: KV_TOKENS=240000 bash this_script.sh
KV_TOKENS="${KV_TOKENS:-280576}"
export VLLM_ASCEND_MODE1_PARITY_MAX_KV_TOKENS="$KV_TOKENS"
export VLLM_ASCEND_MODE1_PARITY_MAX_KV_TOKENS_FLOOR2="$KV_TOKENS"
export VLLM_ASCEND_MODE1_PARITY_MAX_KV_TOKENS_FLOOR4="$KV_TOKENS"
export VLLM_ASCEND_MODE1_PARITY_MAX_KV_TOKENS_FLOOR8="$KV_TOKENS"

# Keep this deterministic for validation runs: the log should show KV_TOKENS.
export VLLM_ASCEND_MODE1_ADAPTIVE_KV_RESIZE="${VLLM_ASCEND_MODE1_ADAPTIVE_KV_RESIZE:-0}"

RUN_TAG="${RUN_TAG:-$(date -u +%Y%m%dT%H%M%SZ)_cann91_kv${KV_TOKENS}_floor4}"
RUN_DIR="$SCRIPT_DIR/mode1_baseline_random_batch_floor4_threshold_multistep/$RUN_TAG"
mkdir -p "$RUN_DIR"

{
    echo "[kv${KV_TOKENS} test] start_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "[kv${KV_TOKENS} test] python=$(command -v python)"
    python - <<'PY'
import os
import torch
import torch_npu
import vllm
import vllm_ascend

print("torch", torch.__version__)
print("torch_npu", torch_npu.__version__)
print("vllm", getattr(vllm, "__file__", None))
print("vllm_ascend", getattr(vllm_ascend, "__file__", None))
for key in (
    "ASCEND_HOME_PATH",
    "USE_CANN91_A3_WITH_NNAL91",
    "VLLM_ASCEND_MODE1_PARITY_MAX_KV_TOKENS",
    "VLLM_ASCEND_MODE1_PARITY_MAX_KV_TOKENS_FLOOR4",
    "VLLM_ASCEND_MODE1_ADAPTIVE_KV_RESIZE",
):
    print(key, os.environ.get(key))
PY

    set +e
    bash "$SCRIPT_DIR/run_mode1_local_baseline_random_batch_threshold_multistep.sh" "$@"
    rc=$?
    set -e
    echo "[kv${KV_TOKENS} test] end_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ) exit_code=$rc"
    exit "$rc"
} 2>&1 | tee "$RUN_DIR/launcher.log"
