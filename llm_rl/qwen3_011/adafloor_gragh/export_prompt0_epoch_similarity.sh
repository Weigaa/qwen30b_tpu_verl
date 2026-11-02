#!/usr/bin/env bash
set -euo pipefail

ARTIFACT_ROOT="${1:-moe_stats/prompt_token_layer_artifacts}"
EPOCH_A="${2:-0}"
EPOCH_B="${3:-1}"
PROMPT_INDEX="${4:-0}"
OUT_CSV="${5:-moe_stats/prompt${PROMPT_INDEX}_epoch${EPOCH_A}_vs_epoch${EPOCH_B}_all_token_layer_similarity.csv}"

python analyze_prompt_token_artifact.py \
  --artifact-root "${ARTIFACT_ROOT}" \
  --epoch-a "${EPOCH_A}" \
  --epoch-b "${EPOCH_B}" \
  --prompt-index "${PROMPT_INDEX}" \
  --export-all-token-layer-csv \
  --output-csv "${OUT_CSV}"

echo "CSV written to ${OUT_CSV}"
