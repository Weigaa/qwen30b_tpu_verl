#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR"

DATA_ROOT="${EURUS2_CODE_DATA_ROOT:-/data/eurus2_rl_code}"
RAW_DIR="$DATA_ROOT/raw"
WORKLOAD_DIR="${EURUS2_CODE_WORKLOAD_DIR:-$DATA_ROOT/validation_code_paired_160}"
MODEL_PATH="${MODEL_PATH:-/data/Qwen3-30B-A3B}"
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
HF_REVISION="${EURUS2_HF_REVISION:-main}"
SOURCE_FILE="$RAW_DIR/validation.parquet"

mkdir -p "$RAW_DIR"
echo "[eurus2 code data] endpoint=$HF_ENDPOINT revision=$HF_REVISION"
echo "[eurus2 code data] source=$SOURCE_FILE output=$WORKLOAD_DIR"

HF_ENDPOINT="$HF_ENDPOINT" python3 - "$RAW_DIR" "$HF_REVISION" <<'PY'
import sys
from huggingface_hub import hf_hub_download

raw_dir, revision = sys.argv[1:]
path = hf_hub_download(
    repo_id="PRIME-RL/Eurus-2-RL-Data",
    filename="validation.parquet",
    repo_type="dataset",
    revision=revision,
    local_dir=raw_dir,
)
print(f"downloaded={path}")
PY

prepare_args=(
    --input "$SOURCE_FILE"
    --output-dir "$WORKLOAD_DIR"
    --model-path "$MODEL_PATH"
    --source-split validation
    --train-samples 160
    --test-samples 64
    --max-prompt-length 1024
    --selection-seed "${EURUS2_SELECTION_SEED:-20260730}"
)
if [[ "${EURUS2_PREPARE_FORCE:-0}" == "1" ]]; then
    prepare_args+=(--force)
elif [[ -f "$WORKLOAD_DIR/manifest.json" \
        && -f "$WORKLOAD_DIR/train.parquet" \
        && -f "$WORKLOAD_DIR/test.parquet" ]]; then
    echo "[eurus2 code data] existing prepared workload retained"
    exit 0
fi

python3 "$SCRIPT_DIR/tools/prepare_eurus2_code_subset.py" "${prepare_args[@]}"
