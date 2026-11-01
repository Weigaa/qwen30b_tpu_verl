#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
VARIANT=${1:?usage: select_vllm_ascend_native.sh <torch29_cann85|torch27_cann83>}

case "${VARIANT}" in
  torch29_cann85)
    BACKUP_ROOT="${ROOT}/.runtime_backups/torch29_cann85_vllm0140rc1_20260903/vllm_ascend"
    KERNEL_LIB64_SOURCE="${BACKUP_ROOT}/lib64/libvllm_ascend_kernels.so"
    ;;
  torch27_cann83)
    BACKUP_ROOT="${ROOT}/.runtime_backups/torch27_cann83_20260903/vllm_ascend"
    KERNEL_LIB64_SOURCE="${BACKUP_ROOT}/libvllm_ascend_kernels.lib64.so"
    ;;
  *)
    printf 'unsupported native runtime variant: %s\n' "${VARIANT}" >&2
    exit 2
    ;;
esac

EXTENSION_NAME="vllm_ascend_C.cpython-311-aarch64-linux-gnu.so"

select_artifact() {
  local source=$1
  local destination=$2
  local temporary

  if [ ! -f "${source}" ]; then
    printf 'missing native runtime artifact: %s\n' "${source}" >&2
    exit 1
  fi
  if cmp -s "${source}" "${destination}"; then
    return
  fi

  temporary="${destination}.runtime-select.$$"
  install -m 0755 "${source}" "${temporary}"
  mv -f "${temporary}" "${destination}"
}

select_artifact \
  "${BACKUP_ROOT}/${EXTENSION_NAME}" \
  "${ROOT}/vllm_ascend/${EXTENSION_NAME}"
select_artifact \
  "${BACKUP_ROOT}/libvllm_ascend_kernels.so" \
  "${ROOT}/vllm_ascend/libvllm_ascend_kernels.so"
select_artifact \
  "${KERNEL_LIB64_SOURCE}" \
  "${ROOT}/vllm_ascend/lib64/libvllm_ascend_kernels.so"

printf '[qwen3 native runtime] selected=%s\n' "${VARIANT}"
