#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'USAGE'
Usage:
  ./run_mode1_mode4_pressure_compare.sh [--floors 2,4,8] [--modes 4,1] [--scale 0.96] [--dry-run]
  COMPARE_RUN_DIR=/path/to/run ./run_mode1_mode4_pressure_compare.sh --analyze-only
  ./run_mode1_mode4_pressure_compare.sh --analyze-only  # analyze latest run

Purpose:
  Compare mode=1 vs mode=4 under high-KV pressure. For each floor, this script
  uses the existing repeated-halving bucket cap hook to make the last bucket
  approach the known mode=4 KV limit without intentionally preempting mode=4.
  With 16 ranks, that last bucket is rank 15 only. The same pressure is then
  run under mode=1, where preemption is expected for these floors.

Environment overrides:
  COMPARE_RUN_ROOT=mode1_mode4_pressure_compare_runs
  COMPARE_RUN_DIR=/path/to/existing_or_new_run_dir
  PATCH_TREE=/path/to/code/tree
  COMPARE_FLOORS=2,4,8
  COMPARE_MODES=4,1
  COMPARE_SCALE=0.96
  COMPARE_MAX_PROMPT_LENGTH=1024
  COMPARE_MAX_RESPONSE_LENGTH=16384
  COMPARE_MAX_NUM_SEQS=32
  COMPARE_TARGET_RANK=15  # fixed by 16-rank repeated-halving buckets
  COMPARE_LOW_CAP=256
  COMPARE_PORT_BASE_HCCL=48641
  COMPARE_PORT_BASE_MASTER=28640
  COMPARE_DRY_RUN=1
  COMPARE_RESUME=1
  COMPARE_FORCE=1

Per-floor cap overrides:
  COMPARE_RESPONSE_CAP_FLOOR2=10900
  COMPARE_RESPONSE_CAP_FLOOR4=11400
  COMPARE_RESPONSE_CAP_FLOOR8=11900

Notes:
  - The launcher is copied to the run directory and patched there only, so the
    repository's existing scripts and training logic are not modified.
  - The pressure control uses the existing VERL_ELASTIC_TAIL_VALIDATE_LEVEL_TOKENS
    bucket path. No additional rollout code path is required.
  - The temporary launcher supports ROLLOUT_IGNORE_EOS=True and LOG_HOME=case_dir
    to force max-token pressure and keep logs under the compare run directory.
USAGE
}

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PATCH_TREE=${PATCH_TREE:-$SCRIPT_DIR}
LAUNCHER_SRC="$PATCH_TREE/internal/wj_train_grpo_qwen30b_a3b_16die_true_weight_eager.sh"
PARSER="$PATCH_TREE/parse_mode1_mode4_pressure_compare.py"

floors=${COMPARE_FLOORS:-2,4,8}
modes=${COMPARE_MODES:-4,1}
scale=${COMPARE_SCALE:-0.96}
dry_run=${COMPARE_DRY_RUN:-0}
resume=${COMPARE_RESUME:-1}
force=${COMPARE_FORCE:-0}
analyze_only=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --floors)
            [[ $# -ge 2 ]] || { echo "missing value for --floors" >&2; exit 2; }
            floors="$2"
            shift 2
            ;;
        --scale)
            [[ $# -ge 2 ]] || { echo "missing value for --scale" >&2; exit 2; }
            scale="$2"
            shift 2
            ;;
        --modes)
            [[ $# -ge 2 ]] || { echo "missing value for --modes" >&2; exit 2; }
            modes="$2"
            shift 2
            ;;
        --dry-run)
            dry_run=1
            shift
            ;;
        --analyze-only)
            analyze_only=1
            shift
            ;;
        --resume)
            resume=1
            shift
            ;;
        --no-resume)
            resume=0
            shift
            ;;
        --force)
            force=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

if [[ ! -f "$LAUNCHER_SRC" ]]; then
    echo "launcher not found: $LAUNCHER_SRC" >&2
    exit 2
fi
if [[ ! -f "$PARSER" ]]; then
    echo "parser not found: $PARSER" >&2
    exit 2
fi

stamp=$(date -u +%Y%m%dT%H%M%SZ)
run_root=${COMPARE_RUN_ROOT:-$PATCH_TREE/mode1_mode4_pressure_compare_runs}

if [[ "$analyze_only" == "1" ]]; then
    if [[ -n "${COMPARE_RUN_DIR:-}" ]]; then
        run_dir="$COMPARE_RUN_DIR"
    else
        run_dir=$({
            find "$run_root" -mindepth 1 -maxdepth 1 -type d -printf '%T@ %p\n' 2>/dev/null \
                | sort -nr \
                | awk '{print substr($0, index($0,$2))}' \
                | while IFS= read -r candidate; do
                    if find "$candidate" -mindepth 2 -maxdepth 2 -name case.env -print -quit 2>/dev/null | grep -q .; then
                        printf '%s\n' "$candidate"
                        break
                    fi
                done
        } || true)
        if [[ -z "$run_dir" ]]; then
            echo "no existing compare run found under: $run_root" >&2
            echo "set COMPARE_RUN_DIR=/path/to/run or run a compare case first" >&2
            exit 2
        fi
    fi
    if [[ ! -d "$run_dir" ]]; then
        echo "compare run dir not found: $run_dir" >&2
        exit 2
    fi
    python3 "$PARSER" "$run_dir" --write
    echo "[compare] analyzed: $run_dir"
    echo "[compare] summary: $run_dir/summary.md"
    echo "[compare] csv: $run_dir/summary.csv"
    exit 0
fi

run_dir=${COMPARE_RUN_DIR:-$run_root/$stamp}
mkdir -p "$run_dir"

compare_bucket_sizes="8,4,2,1,1"
compare_bucket_ranges="[0-7],[8-11],[12-13],[14],[15]"
target_rank=${COMPARE_TARGET_RANK:-15}
if [[ "$target_rank" != "15" ]]; then
    echo "COMPARE_TARGET_RANK=$target_rank is unsupported by the bucket-only harness." >&2
    echo "For world_size=16, VERL_ELASTIC_TAIL_VALIDATE_LEVEL_TOKENS buckets are [0-7],[8-11],[12-13],[14],[15], so only rank15 can be isolated without a rollout code hook." >&2
    exit 2
fi

launcher="$run_dir/_compare_launcher.sh"
python3 - "$LAUNCHER_SRC" "$launcher" <<'PY'
from pathlib import Path
import sys
src = Path(sys.argv[1])
dst = Path(sys.argv[2])
text = src.read_text()
old_log = 'logfile="${HOME}/wjeagerqwen30b-a3b-with_draft_${DRAFT_PROFILE_MODE}_${time}${elastic_suffix}.txt"\n'
new_log = 'LOG_HOME=${LOG_HOME:-${HOME}}\nmkdir -p "${LOG_HOME}"\nlogfile="${LOG_HOME}/wjeagerqwen30b-a3b-with_draft_${DRAFT_PROFILE_MODE}_${time}${elastic_suffix}.txt"\n'
if old_log not in text:
    raise SystemExit('could not patch logfile line in launcher copy')
text = text.replace(old_log, new_log, 1)
old = 'actor_rollout_ref.rollout.ignore_eos=False \\\n'
new = 'actor_rollout_ref.rollout.ignore_eos="${ROLLOUT_IGNORE_EOS:-False}" \\\n'
if old not in text:
    raise SystemExit('could not patch ignore_eos line in launcher copy')
text = text.replace(old, new, 1)
dst.write_text(text)
dst.chmod(0o755)
PY

cat > "$run_dir/run_config.env" <<EOF_CONFIG
PATCH_TREE=$PATCH_TREE
LAUNCHER_SRC=$LAUNCHER_SRC
COMPARE_FLOORS=$floors
COMPARE_MODES=$modes
COMPARE_SCALE=$scale
COMPARE_MAX_PROMPT_LENGTH=${COMPARE_MAX_PROMPT_LENGTH:-1024}
COMPARE_MAX_RESPONSE_LENGTH=${COMPARE_MAX_RESPONSE_LENGTH:-16384}
COMPARE_MAX_NUM_SEQS=${COMPARE_MAX_NUM_SEQS:-32}
COMPARE_TARGET_RANK=$target_rank
COMPARE_BUCKET_SIZES=$compare_bucket_sizes
COMPARE_BUCKET_RANGES=$compare_bucket_ranges
COMPARE_LOW_CAP=${COMPARE_LOW_CAP:-256}
EOF_CONFIG

mode4_kv_tokens() {
    case "$1" in
        2) echo "397824" ;;
        4) echo "414848" ;;
        8) echo "431872" ;;
        *) echo "unsupported floor for mode4 KV target: $1" >&2; return 2 ;;
    esac
}

mode1_reference_kv_tokens() {
    local floor=$1
    local override_var="COMPARE_MODE1_KV_FLOOR${floor}"
    local override=${!override_var:-}
    if [[ -n "$override" ]]; then
        echo "$override"
        return 0
    fi
    case "$floor" in
        2) echo "174208" ;;
        4) echo "321408" ;;
        8) echo "395520" ;;
        *) echo "unsupported floor for mode1 KV reference: $floor" >&2; return 2 ;;
    esac
}

calc_response_cap() {
    local floor=$1
    local override_var="COMPARE_RESPONSE_CAP_FLOOR${floor}"
    local override=${!override_var:-}
    if [[ -n "$override" ]]; then
        echo "$override"
        return 0
    fi
    local kv prompt seq
    kv=$(mode4_kv_tokens "$floor")
    prompt=${COMPARE_MAX_PROMPT_LENGTH:-1024}
    seq=${COMPARE_MAX_NUM_SEQS:-32}
    awk -v kv="$kv" -v scale="$scale" -v seq="$seq" -v prompt="$prompt" 'BEGIN {
        cap = int((kv * scale) / seq - prompt)
        if (cap < 1) cap = 1
        print cap
    }'
}

build_cap_list() {
    local floor=$1
    local high=$2
    local low=${COMPARE_LOW_CAP:-256}
    case "$floor" in
        2|4|8) echo "${low},${low},${low},${low},${high}" ;;
        *) echo "unsupported floor: $floor" >&2; return 2 ;;
    esac
}

run_case() {
    local mode=$1
    local floor=$2
    local cap_list=$3
    local high_cap=$4
    local case_dir="$run_dir/floor${floor}_mode${mode}"
    mkdir -p "$case_dir"

    local hccl_base=${COMPARE_PORT_BASE_HCCL:-48641}
    local master_base=${COMPARE_PORT_BASE_MASTER:-28640}
    local case_offset=$((floor * 10 + mode))
    local hccl_port=$((hccl_base + case_offset * 100))
    local master_port=$((master_base + case_offset * 100))

    if [[ "$dry_run" != "1" && "$resume" == "1" && "$force" != "1" && -f "$case_dir/case.env" && -f "$case_dir/launcher.log" ]]; then
        if grep -q 'launcher_status=0' "$case_dir/case.env" \
           && grep -q 'rollout_output_time_s:' "$case_dir/launcher.log"; then
            echo "[compare] skip completed case: mode=$mode floor=$floor (use --force to rerun)"
            python3 "$PARSER" "$run_dir" --write >/dev/null || true
            return 0
        fi
    fi

    local prompt_len=${COMPARE_MAX_PROMPT_LENGTH:-1024}
    local response_len=${COMPARE_MAX_RESPONSE_LENGTH:-16384}
    local max_num_seqs=${COMPARE_MAX_NUM_SEQS:-32}
    local target_kv
    target_kv=$(mode4_kv_tokens "$floor")
    local mode1_ref_kv
    mode1_ref_kv=$(mode1_reference_kv_tokens "$floor")
    local estimated_high_rank_tokens=$((max_num_seqs * (prompt_len + high_cap)))
    local pressure_ratio
    pressure_ratio=$(awk -v est="$estimated_high_rank_tokens" -v kv="$target_kv" 'BEGIN {
        if (kv > 0) printf "%.6f", est / kv
    }')
    local mode1_pressure_ratio
    mode1_pressure_ratio=$(awk -v est="$estimated_high_rank_tokens" -v kv="$mode1_ref_kv" 'BEGIN {
        if (kv > 0) printf "%.6f", est / kv
    }')
    local mode1_pressure_margin_tokens=$((estimated_high_rank_tokens - mode1_ref_kv))

    cat > "$case_dir/case.env" <<EOF_CASE
mode=$mode
floor=$floor
target_mode4_kv_tokens=$target_kv
mode1_reference_kv_tokens=$mode1_ref_kv
scale=$scale
prompt_length=$prompt_len
max_response_length=$response_len
max_num_seqs=$max_num_seqs
high_response_cap=$high_cap
estimated_high_rank_tokens=$estimated_high_rank_tokens
estimated_pressure_ratio_to_mode4_kv=$pressure_ratio
estimated_pressure_ratio_to_mode1_ref_kv=$mode1_pressure_ratio
estimated_mode1_pressure_margin_tokens=$mode1_pressure_margin_tokens
target_rank=$target_rank
cap_list=$cap_list
launcher_log=$case_dir/launcher.log
case_dir=$case_dir
EOF_CASE

    echo "[compare] mode=$mode floor=$floor target_rank=$target_rank high_cap=$high_cap mode4_pressure_ratio=${pressure_ratio} mode1_ref_pressure_ratio=${mode1_pressure_ratio} mode1_margin_tokens=${mode1_pressure_margin_tokens} cap_list=$cap_list case_dir=$case_dir"

    if [[ "$dry_run" == "1" ]]; then
        echo "dry_run=1" >> "$case_dir/case.env"
        return 0
    fi

    set +e
    (
        cd "$PATCH_TREE"
        export LOG_HOME="$case_dir"
        export RECORD_DIR="$case_dir/record"
        export VLLM_ASCEND_ELASTIC_EXECUTION_MODE="$mode"
        export VLLM_ASCEND_ELASTIC_MIN_COMPUTE_GROUP_SIZE="$floor"
        if [[ "$mode" == "4" ]]; then
            export VLLM_ASCEND_MODE4_RUNTIME_MIN_COMPUTE_GROUP_SIZE="$floor"
        else
            unset VLLM_ASCEND_MODE4_RUNTIME_MIN_COMPUTE_GROUP_SIZE || true
        fi
        export VLLM_ASCEND_MODE1_PARITY_MAX_KV_TOKENS=0
        export VERL_ELASTIC_TAIL_VALIDATE_LEVEL_TOKENS="$cap_list"
        export ROLLOUT_IGNORE_EOS=${ROLLOUT_IGNORE_EOS:-True}
        export MAX_PROMPT_LENGTH=${COMPARE_MAX_PROMPT_LENGTH:-1024}
        export MAX_RESPONSE_LENGTH=${COMPARE_MAX_RESPONSE_LENGTH:-16384}
        export ROLLOUT_MAX_NUM_SEQS=${COMPARE_MAX_NUM_SEQS:-32}
        export ROLLOUT_MAX_MODEL_LEN=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))
        export ROLLOUT_MAX_NUM_BATCHED_TOKENS=${COMPARE_MAX_NUM_BATCHED_TOKENS:-$ROLLOUT_MAX_MODEL_LEN}
        export TRAINER_TOTAL_EPOCHS=${COMPARE_TRAINER_TOTAL_EPOCHS:-1}
        export HCCL_IF_BASE_PORT="$hccl_port"
        export MASTER_PORT="$master_port"
        export VERL_HCCL_IF_BASE_PORT_START="$hccl_port"
        export PYTHONPATH="$PATCH_TREE${PYTHONPATH:+:$PYTHONPATH}"
        bash "$launcher"
    ) 2>&1 | tee "$case_dir/launcher.log"
    local status=${PIPESTATUS[0]}
    set -e
    echo "launcher_status=$status" >> "$case_dir/case.env"

    python3 "$PARSER" "$run_dir" --write >/dev/null || true
    return 0
}

IFS=',' read -r -a floor_array <<< "$floors"
IFS=',' read -r -a mode_array <<< "$modes"
for mode in "${mode_array[@]}"; do
    mode=$(echo "$mode" | tr -d '[:space:]')
    case "$mode" in
        1|4) ;;
        *) echo "unsupported compare mode: $mode; expected 1 or 4" >&2; exit 2 ;;
    esac
done
for mode in "${mode_array[@]}"; do
    mode=$(echo "$mode" | tr -d '[:space:]')
    for floor in "${floor_array[@]}"; do
        floor=$(echo "$floor" | tr -d '[:space:]')
        [[ -n "$floor" ]] || continue
        case "$floor" in
            2|4|8) ;;
            *) echo "unsupported floor: $floor; expected 2,4,8" >&2; exit 2 ;;
        esac
        high_cap=$(calc_response_cap "$floor")
        cap_list=$(build_cap_list "$floor" "$high_cap")
        run_case "$mode" "$floor" "$cap_list" "$high_cap"
    done
done

python3 "$PARSER" "$run_dir" --write

echo "[compare] done: $run_dir"
echo "[compare] summary: $run_dir/summary.md"
echo "[compare] csv: $run_dir/summary.csv"
