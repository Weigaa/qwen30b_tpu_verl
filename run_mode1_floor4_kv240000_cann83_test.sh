#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR"

CANN83_TOOLKIT_ENV="${CANN83_TOOLKIT_ENV:-/usr/local/Ascend/ascend-toolkit/set_env.sh}"
CANN83_ATB_ENV="${CANN83_ATB_ENV:-/usr/local/Ascend/nnal/atb/set_env.sh}"
CANN83_ASDSIP_ENV="${CANN83_ASDSIP_ENV:-/usr/local/Ascend/nnal/asdsip/set_env.sh}"

for f in "$CANN83_TOOLKIT_ENV" "$CANN83_ATB_ENV" "$CANN83_ASDSIP_ENV"; do
    if [[ ! -f "$f" ]]; then
        echo "missing old CANN/NNAL env script: $f" >&2
        exit 2
    fi
done

# Make this wrapper robust after a CANN 9.1 shell has been sourced: remove the
# local 9.1 overlay paths before sourcing the system 8.3 stack.
strip_cann91_paths() {
    local value="${1:-}"
    python3 - "$value" <<'PY'
import sys
value = sys.argv[1]
parts = [p for p in value.split(':') if p]
blocked = (
    '/workspace/cann-recipes-train/llm_rl/qwen3/cann_master_9.1',
    '/workspace/cann-recipes-train/llm_rl/qwen3/cann_master_9.1_a3',
    '/workspace/cann-recipes-train/llm_rl/qwen3/Ascend_nnal_9.1.0_beta3',
    '/workspace/cann-recipes-train/llm_rl/qwen3/nnal_9.1.0_beta3_a3',
)
print(':'.join(p for p in parts if not p.startswith(blocked)))
PY
}

export LD_LIBRARY_PATH="$(strip_cann91_paths "${LD_LIBRARY_PATH:-}")"
export PYTHONPATH="$(strip_cann91_paths "${PYTHONPATH:-}")"
unset USE_CANN91_A3_WITH_NNAL91
unset NNAL91_ROOT
unset ATB91_HOME_PATH
unset ATB_HOME_PATH
unset ASDSIP_HOME_PATH
unset ASCEND_HOME_PATH
unset ASCEND_TOOLKIT_HOME
unset TOOLCHAIN_HOME
unset VIRTUAL_ENV
# Avoid inheriting the 2.9 venv bin path if the caller has it activated.
export PATH="$(python3 - <<'PY'
import os
blocked = (
    '/workspace/cann-recipes-train/llm_rl/qwen3_true_mode5_shrinkaware_staged/.venv_cann91_a3_torch29/bin',
    '/workspace/cann-recipes-train/llm_rl/qwen3_true_mode5_shrinkaware_staged/.venv_cann91_a3_torch271/bin',
)
print(':'.join(p for p in os.environ.get('PATH', '').split(':') if p and not p.startswith(blocked)))
PY
)"

# Old ATB/ASDSIP set_env.sh reads optional shell variables such as ZSH_VERSION
# without guarding them, so source these scripts with nounset temporarily off.
set +u
source "$CANN83_TOOLKIT_ENV"
source "$CANN83_ATB_ENV"
source "$CANN83_ASDSIP_ENV"
set -u

export ANALYZE_MODE1_THRESHOLD_LOG="${ANALYZE_MODE1_THRESHOLD_LOG:-1}"

KV_TOKENS="${KV_TOKENS:-240000}"
export VLLM_ASCEND_MODE1_PARITY_MAX_KV_TOKENS="$KV_TOKENS"
export VLLM_ASCEND_MODE1_PARITY_MAX_KV_TOKENS_FLOOR2="$KV_TOKENS"
export VLLM_ASCEND_MODE1_PARITY_MAX_KV_TOKENS_FLOOR4="$KV_TOKENS"
export VLLM_ASCEND_MODE1_PARITY_MAX_KV_TOKENS_FLOOR8="$KV_TOKENS"
export VLLM_ASCEND_MODE1_ADAPTIVE_KV_RESIZE="${VLLM_ASCEND_MODE1_ADAPTIVE_KV_RESIZE:-0}"

RUN_TAG="${RUN_TAG:-$(date -u +%Y%m%dT%H%M%SZ)_cann83_kv${KV_TOKENS}_floor4}"
RUN_DIR="$SCRIPT_DIR/mode1_baseline_random_batch_floor4_threshold_multistep/$RUN_TAG"
mkdir -p "$RUN_DIR"

{
    echo "[old-cann83-kv${KV_TOKENS} test] start_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "[old-cann83-kv${KV_TOKENS} test] python=$(command -v python3)"
    python3 - <<'PY'
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
    "ASCEND_TOOLKIT_HOME",
    "USE_CANN91_A3_WITH_NNAL91",
    "VIRTUAL_ENV",
    "VLLM_ASCEND_MODE1_PARITY_MAX_KV_TOKENS",
    "VLLM_ASCEND_MODE1_PARITY_MAX_KV_TOKENS_FLOOR4",
    "VLLM_ASCEND_MODE1_ADAPTIVE_KV_RESIZE",
):
    print(key, os.environ.get(key))
print("LD_LIBRARY_PATH_head", ":".join(os.environ.get("LD_LIBRARY_PATH", "").split(":")[:8]))
PY

    set +e
    bash "$SCRIPT_DIR/run_mode1_local_baseline_random_batch_threshold_multistep.sh" "$@"
    rc=$?
    set -e
    echo "[old-cann83-kv${KV_TOKENS} test] end_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ) exit_code=$rc"
    exit "$rc"
} 2>&1 | tee "$RUN_DIR/launcher.log"
