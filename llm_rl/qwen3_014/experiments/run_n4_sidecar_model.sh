#!/usr/bin/env bash
set -euo pipefail
# Only independent Sidecar processes use this model-extension runtime.
# The primary RL process retains the original, evidence-pinned runtime.
root=/workspace/cann-recipes-train/llm_rl/qwen3_014/qwen3_sidecar
export PYTHONPATH="$root:${PYTHONPATH:-}"
# Python -m prepends cwd ahead of PYTHONPATH. The lease watcher is launched
# from the primary RL checkout, whose vllm lacks the legacy PanGu adapter.
# Isolate cwd as well as PYTHONPATH for nested registry/worker subprocesses.
cd "$root"
exec bash "$root/tools/run_frozen_sidecar.sh"
