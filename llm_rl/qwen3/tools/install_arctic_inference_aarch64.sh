#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
VENV=${QWEN3_CANN85_TORCH29_VENV:-${ROOT}/.venv_cann85_a3_torch29}
PYTHON=${VENV}/bin/python
PIP_INDEX_URL=${PIP_INDEX_URL:-https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple}

test -x "${PYTHON}"

TMPDIR=$(mktemp -d /tmp/qwen3-arctic-inference.XXXXXX)
trap 'rm -rf "${TMPDIR}"' EXIT

TORCH_DEVICE_BACKEND_AUTOLOAD=0 "${PYTHON}" -m pip download \
  --no-deps --no-binary=:all: --no-build-isolation \
  arctic-inference==0.1.1 -d "${TMPDIR}" -i "${PIP_INDEX_URL}"
tar -xf "${TMPDIR}/arctic_inference-0.1.1.tar.gz" -C "${TMPDIR}"
ARCTIC_SETUP="${TMPDIR}/arctic_inference-0.1.1/setup.py" "${PYTHON}" - <<'PY'
import os
from pathlib import Path

path = Path(os.environ["ARCTIC_SETUP"])
source = path.read_text()
old = '''ext_modules.append(
     CMakeExtension("arctic_inference.suffix_decoding._C",
                   "csrc/suffix_decoding"),
)
'''
new = '''if os.environ.get("ARCTIC_INFERENCE_SKIP_NATIVE", "").lower() not in (
        "1", "true", "on"):
    ext_modules.append(
        CMakeExtension("arctic_inference.suffix_decoding._C",
                       "csrc/suffix_decoding"),
    )
'''
if source.count(old) != 1:
    raise RuntimeError("unexpected arctic-inference 0.1.1 setup.py content")
path.write_text(source.replace(old, new))
PY

TORCH_DEVICE_BACKEND_AUTOLOAD=0 ARCTIC_INFERENCE_SKIP_NATIVE=1 \
  "${PYTHON}" -m pip install --no-deps --no-build-isolation \
  "${TMPDIR}/arctic_inference-0.1.1"
