#!/usr/bin/env bash
set -euo pipefail

INPUT_CSV="${1:-moe_stats/prompt0_epoch0_vs_epoch1_all_token_layer_similarity.csv}"
FILTERED_CSV="${2:-moe_stats/prompt0_epoch0_vs_epoch1_comparable_only.csv}"
POSITION_PLOT="${3:-moe_stats/prompt0_epoch0_vs_epoch1_position_mean.png}"
LAYERS_PLOT="${4:-moe_stats/prompt0_epoch0_vs_epoch1_random5layers.png}"
SEED="${5:-42}"

python process_prompt_similarity_csv.py \
  "${INPUT_CSV}" \
  --filtered-csv "${FILTERED_CSV}" \
  --position-plot "${POSITION_PLOT}" \
  --layers-plot "${LAYERS_PLOT}" \
  --num-random-layers 5 \
  --seed "${SEED}"

echo "Filtered CSV written to ${FILTERED_CSV}"
echo "Position mean plot written to ${POSITION_PLOT}"
echo "Random 5 layers plot written to ${LAYERS_PLOT}"
