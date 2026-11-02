#!/usr/bin/env bash
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
exec python3 -u "$HERE/tools/run_mlsys_fig6.py" "${1:-dry-run}" "${@:2}"
