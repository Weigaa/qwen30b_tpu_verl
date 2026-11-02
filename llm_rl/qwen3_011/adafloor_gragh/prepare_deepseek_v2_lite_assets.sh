#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR"

MODEL_ID=${DEEPSEEK_MODEL_ID:-deepseek-ai/DeepSeek-V2-Lite-Chat}
MODEL_REVISION=${DEEPSEEK_MODEL_REVISION:-85864749cd611b4353ce1decdb286193298f64c7}
MODEL_PATH=${MODEL_PATH:-/data/DeepSeek-V2-Lite-Chat}
DISTCP_PATH=${DISTCP_PATH:-/data/DeepSeek-V2-Lite-Chat_megatron_pp4_ep4}
DOWNLOAD_BACKEND=${DEEPSEEK_DOWNLOAD_BACKEND:-auto}
DOWNLOAD_RETRIES=${DEEPSEEK_DOWNLOAD_RETRIES:-20}
DOWNLOAD_RETRY_DELAY_SECONDS=${DEEPSEEK_DOWNLOAD_RETRY_DELAY_SECONDS:-10}
CONVERT_NPROC=${DEEPSEEK_CONVERT_NPROC:-16}
CONVERT_PP_SIZE=${DEEPSEEK_CONVERT_PP_SIZE:-4}
CONVERT_EP_SIZE=${DEEPSEEK_CONVERT_EP_SIZE:-4}
MASTER_PORT=${MASTER_PORT:-29621}

usage() {
    cat <<'EOF'
Usage: ./prepare_deepseek_v2_lite_assets.sh ACTION

Actions:
  download   Download the pinned DeepSeek-V2-Lite-Chat into MODEL_PATH
  validate   Validate the Hugging Face model assets
  convert    Convert the model to a PP4 x EP4 Megatron distributed checkpoint
  all        Download, validate, convert, and validate the result

This script performs real network or NPU work only when invoked explicitly.
EOF
}

validate_hf() {
    python3 "$SCRIPT_DIR/tools/validate_deepseek_v2_lite_assets.py" \
        --model-path "$MODEL_PATH" \
        --expected-model-id "$MODEL_ID" \
        --expected-revision "$MODEL_REVISION"
}

validate_all() {
    local distcp_path=${1:-$DISTCP_PATH}
    python3 "$SCRIPT_DIR/tools/validate_deepseek_v2_lite_assets.py" \
        --model-path "$MODEL_PATH" \
        --distcp-path "$distcp_path" \
        --expected-model-id "$MODEL_ID" \
        --expected-revision "$MODEL_REVISION" \
        --expected-pp-size "$CONVERT_PP_SIZE" \
        --expected-ep-size "$CONVERT_EP_SIZE"
}

write_conversion_manifest() {
    local output_path=$1
    python3 - "$output_path" "$MODEL_ID" "$MODEL_REVISION" \
        "$CONVERT_PP_SIZE" "$CONVERT_EP_SIZE" "$CONVERT_NPROC" <<'PY'
import json
import sys
from pathlib import Path

output, model_id, revision, pp_size, ep_size, world_size = sys.argv[1:]
path = Path(output) / ".adafloor_deepseek_v2_lite_manifest.json"
payload = {
    "model_id": model_id,
    "model_revision": revision,
    "architecture": "DeepseekV2ForCausalLM",
    "pipeline_model_parallel_size": int(pp_size),
    "expert_model_parallel_size": int(ep_size),
    "world_size": int(world_size),
}
temporary = path.with_suffix(path.suffix + ".tmp")
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
temporary.replace(path)
PY
}

download_model() {
    if [[ "$MODEL_ID" != "deepseek-ai/DeepSeek-V2-Lite-Chat" ]]; then
        echo "refusing a noncanonical checkpoint: $MODEL_ID" >&2
        exit 2
    fi
    mkdir -p "$MODEL_PATH"
    local backend="$DOWNLOAD_BACKEND"
    if [[ "$backend" == auto ]]; then
        if command -v hf >/dev/null 2>&1; then
            backend=hf
        elif command -v huggingface-cli >/dev/null 2>&1; then
            backend=huggingface
        else
            echo "no supported downloader found. Install huggingface_hub." >&2
            exit 2
        fi
    fi
    if (( DOWNLOAD_RETRIES < 1 )); then
        echo "DEEPSEEK_DOWNLOAD_RETRIES must be positive" >&2
        exit 2
    fi
    local attempt
    for ((attempt = 1; attempt <= DOWNLOAD_RETRIES; attempt++)); do
        if [[ "$backend" == hf ]]; then
            if hf download "$MODEL_ID" --revision "$MODEL_REVISION" \
                --local-dir "$MODEL_PATH"; then
                return 0
            fi
        elif [[ "$backend" == huggingface ]]; then
            if huggingface-cli download "$MODEL_ID" --revision "$MODEL_REVISION" \
                --local-dir "$MODEL_PATH"; then
                return 0
            fi
        else
            echo "unsupported DEEPSEEK_DOWNLOAD_BACKEND=$backend" >&2
            exit 2
        fi
        if (( attempt < DOWNLOAD_RETRIES )); then
            echo "download attempt $attempt failed, resuming in ${DOWNLOAD_RETRY_DELAY_SECONDS}s" >&2
            sleep "$DOWNLOAD_RETRY_DELAY_SECONDS"
        fi
    done
    echo "download failed after $DOWNLOAD_RETRIES attempts" >&2
    return 1
}

convert_model() {
    validate_hf
    if (( CONVERT_NPROC != CONVERT_PP_SIZE * CONVERT_EP_SIZE )); then
        echo "conversion world size must equal PP x EP" >&2
        echo "nproc=$CONVERT_NPROC pp=$CONVERT_PP_SIZE ep=$CONVERT_EP_SIZE" >&2
        exit 2
    fi
    if [[ -d "$DISTCP_PATH" && -n "$(find "$DISTCP_PATH" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]]; then
        echo "refusing to overwrite nonempty checkpoint directory: $DISTCP_PATH" >&2
        exit 2
    fi
    local staging_path=${DISTCP_PATH}.incomplete
    if [[ -e "$staging_path" ]]; then
        echo "refusing to overwrite incomplete checkpoint staging directory: $staging_path" >&2
        exit 2
    fi
    if [[ -d "$DISTCP_PATH" ]]; then
        rmdir "$DISTCP_PATH"
    fi
    mkdir -p "$staging_path"
    USE_ALLTOALL_OVERLAP=1 MASTER_PORT="$MASTER_PORT" \
        torchrun --nproc_per_node="$CONVERT_NPROC" \
        "$SCRIPT_DIR/converter_hf_to_mcore.py" \
        --hf_model_path "$MODEL_PATH" \
        --output_path "$staging_path" \
        --pp_size "$CONVERT_PP_SIZE" \
        --ep_size "$CONVERT_EP_SIZE" \
        --use_cpu_initialization \
        --trust_remote_code
    write_conversion_manifest "$staging_path"
    validate_all "$staging_path"
    mv "$staging_path" "$DISTCP_PATH"
    validate_all
}

action=${1:-}
case "$action" in
    download)
        download_model
        validate_hf
        ;;
    validate)
        if [[ -d "$DISTCP_PATH" ]]; then
            validate_all
        else
            validate_hf
        fi
        ;;
    convert)
        convert_model
        ;;
    all)
        download_model
        validate_hf
        convert_model
        ;;
    -h|--help|"")
        usage
        ;;
    *)
        usage >&2
        exit 2
        ;;
esac
