#!/usr/bin/env bash
set -euo pipefail

MODEL_ID=${QWEN2_5_1_5B_MODEL_ID:-Qwen/Qwen2.5-1.5B-Instruct}
MODEL_DIR=${QWEN2_5_1_5B_MODEL_DIR:-/data/Qwen2.5-1.5B-Instruct}

if [[ -f "${MODEL_DIR}/config.json" ]]; then
    echo "Qwen2.5-1.5B sidecar model already exists at ${MODEL_DIR}"
    exit 0
fi

mkdir -p "${MODEL_DIR}"
if command -v modelscope >/dev/null 2>&1; then
    modelscope download --model "${MODEL_ID}" --local_dir "${MODEL_DIR}"
elif command -v hf >/dev/null 2>&1; then
    hf download "${MODEL_ID}" --local-dir "${MODEL_DIR}"
elif command -v huggingface-cli >/dev/null 2>&1; then
    huggingface-cli download "${MODEL_ID}" --local-dir "${MODEL_DIR}"
else
    echo "modelscope, hf, or huggingface-cli is required to download ${MODEL_ID}" >&2
    exit 2
fi

[[ -f "${MODEL_DIR}/config.json" ]] || {
    echo "download finished without ${MODEL_DIR}/config.json" >&2
    exit 1
}
echo "Qwen2.5-1.5B sidecar model prepared at ${MODEL_DIR}"
