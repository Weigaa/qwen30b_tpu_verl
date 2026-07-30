#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

latest_elastic_log() {
  ls -1t wjeagerqwen30b-a3b-with_draft_*_elastic.txt 2>/dev/null | head -n 1 || true
}

before_latest="$(latest_elastic_log)"
run_log="mode1_floor2_first_step_$(date -u +%Y%m%dT%H%M%SZ).launcher.log"

echo "[run+analyze] previous_latest=${before_latest:-none}"
echo "[run+analyze] launcher_log=${run_log}"
echo "[run+analyze] expected target: rollout_output_time_s close to remote healthy ~120s"
echo "[run+analyze] key check: final [14,15] should not spend ~350000ms in mode1_drain/send_export/preload"

# The child script also tees to its own timestamped mode1_floor2_first_step_*.log.
# This outer log preserves everything even if the job is interrupted.
bash ./run_mode1_floor2_first_step_test.sh 2>&1 | tee "${run_log}"

after_latest="$(latest_elastic_log)"
if [[ -n "${after_latest}" && "${after_latest}" != "${before_latest}" ]]; then
  analyze_log="${after_latest}"
else
  analyze_log="${after_latest:-${run_log}}"
  echo "[run+analyze] warning: no newer elastic log detected; analyzing ${analyze_log}" >&2
fi

if [[ -z "${analyze_log}" || ! -f "${analyze_log}" ]]; then
  echo "[run+analyze] no log found to analyze" >&2
  exit 1
fi

echo "[run+analyze] analyze_log=${analyze_log}"
echo
echo "===== FINAL SHRINK ====="
python3 internal/analyze_mode1_final_shrink.py "${analyze_log}"
echo
echo "===== SHRINK STAGES ====="
python3 internal/analyze_mode1_shrink_stages.py "${analyze_log}"
