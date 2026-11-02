#!/usr/bin/env bash
set -euo pipefail
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$root"
# CacheConfig's fraction is per vLLM instance. Preserve the parked primary's
# allocation as external usage, then independently clamp by physical free HBM.
export VERL_SIDECAR_COLOCATED_MEMORY_ACCOUNTING=1
snapshot=$(mktemp "$root/internal/.sidecar_snapshot.XXXXXX.run")
cp -- "$root/internal/run_elastic_sidecar_infer.sh" "$snapshot"
trap 'rm -f -- "$snapshot"' EXIT
bash "$snapshot"
