#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_DIR=$(cd "${SCRIPT_DIR}/.." && pwd)
cd "${REPO_DIR}"

FLOORS=${FULL_REDUNDANCY_FLOORS:-"8 4 2 1"}
RUN_ID=${FULL_REDUNDANCY_RUN_ID:-$(date +%Y%m%d%H%M%S)}
OUT_DIR=${FULL_REDUNDANCY_OUT_DIR:-"full_redundancy_runs/${RUN_ID}"}
TRAIN_SCRIPT=${FULL_REDUNDANCY_TRAIN_SCRIPT:-"internal/wj_train_grpo_qwen30b_a3b_16die_true_weight_eager.sh"}
SUMMARY_SCRIPT=${FULL_REDUNDANCY_SUMMARY_SCRIPT:-"internal/summarize_full_redundancy_logs.py"}

mkdir -p "${OUT_DIR}"
MANIFEST="${OUT_DIR}/manifest.tsv"
SWEEP_LOG="${OUT_DIR}/sweep.log"

printf "floor\tstart_time\tend_time\texit_code\tlogfile\n" > "${MANIFEST}"

echo "[fullred sweep] run_id=${RUN_ID} floors=${FLOORS} out_dir=${OUT_DIR} train_script=${TRAIN_SCRIPT}" | tee -a "${SWEEP_LOG}"

for floor in ${FLOORS}; do
    profile_mode="breakdown_fullred_floor${floor}_${RUN_ID}"
    start_time=$(date '+%Y-%m-%dT%H:%M:%S%z')
    echo "[fullred sweep] start floor=${floor} start_time=${start_time}" | tee -a "${SWEEP_LOG}"

    set +e
    VERL_SIDECAR_ENABLE=0 \
    VLLM_ASCEND_ELASTIC_EXECUTION_MODE=1 \
    VLLM_ASCEND_ELASTIC_MIN_COMPUTE_GROUP_SIZE="${floor}" \
    VLLM_ASCEND_FULL_REDUNDANCY_EXPERIMENT_LOG=1 \
    DRAFT_PROFILE_MODE="${profile_mode}" \
    bash "${TRAIN_SCRIPT}" "$@" 2>&1 | tee -a "${OUT_DIR}/floor_${floor}.console.log"
    exit_code=${PIPESTATUS[0]}
    set -e

    end_time=$(date '+%Y-%m-%dT%H:%M:%S%z')
    logfile=$(ls -t "wjeagerqwen30b-a3b-with_draft_${profile_mode}_"*_elastic.txt 2>/dev/null | head -n 1 || true)
    if [[ -n "${logfile}" ]]; then
        ln -sf "$(realpath "${logfile}")" "${OUT_DIR}/floor_${floor}.train.log"
        python3 "${SUMMARY_SCRIPT}" "${logfile}" --markdown > "${OUT_DIR}/floor_${floor}.summary.md" || true
    fi
    printf "%s\t%s\t%s\t%s\t%s\n" "${floor}" "${start_time}" "${end_time}" "${exit_code}" "${logfile}" >> "${MANIFEST}"
    echo "[fullred sweep] end floor=${floor} end_time=${end_time} exit_code=${exit_code} logfile=${logfile}" | tee -a "${SWEEP_LOG}"
done

logs=()
while IFS=$'\t' read -r floor _start _end _exit logfile; do
    [[ "${floor}" == "floor" ]] && continue
    [[ -n "${logfile}" ]] && logs+=("${logfile}")
done < "${MANIFEST}"

if (( ${#logs[@]} > 0 )); then
    python3 "${SUMMARY_SCRIPT}" "${logs[@]}" --markdown > "${OUT_DIR}/summary.md" || true
    python3 "${SUMMARY_SCRIPT}" "${logs[@]}" --csv > "${OUT_DIR}/summary.csv" || true
    python3 "${SUMMARY_SCRIPT}" "${logs[@]}" --json > "${OUT_DIR}/summary.json" || true
fi

echo "[fullred sweep] done manifest=${MANIFEST} summary=${OUT_DIR}/summary.md" | tee -a "${SWEEP_LOG}"
