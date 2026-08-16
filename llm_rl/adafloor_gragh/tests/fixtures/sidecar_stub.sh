#!/usr/bin/env bash
set -euo pipefail

: "${SIDECAR_STUB_RECORD:?SIDECAR_STUB_RECORD is required}"
: "${VERL_SIDECAR_STOP_FILE:?VERL_SIDECAR_STOP_FILE is required}"

printf 'devices=%s\n' "${VERL_SIDECAR_NPU_DEVICES:-}" >> "${SIDECAR_STUB_RECORD}"
while [[ ! -f "${VERL_SIDECAR_STOP_FILE}" ]]; do
    sleep 0.05
done
printf 'stopped=1\n' >> "${SIDECAR_STUB_RECORD}"
