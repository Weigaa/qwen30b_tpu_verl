# Torch 2.7 + vLLM 0.14 Local Runtime

This repository uses a repository-local Python environment for the Qwen3
baseline launcher. It does not replace the Python packages used by sibling
repositories.

## Selected stack

- Python: 3.11
- Torch: 2.7.1+cpu
- torch-npu: 2.7.1 development build
- CANN toolkit: 8.3.RC1
- ATB/ASDSIP: host 8.3 installation
- vLLM: 0.14.1 Python source from this repository
- vLLM-Ascend: 0.14 Python source from this repository
- vLLM-Ascend native extension/custom operators: rebuilt against Torch 2.7
  and CANN 8.3

The local venv is `.venv_torch27_vllm014`. It uses
`--system-site-packages` only to reuse the host's existing Torch 2.7,
torch-npu 2.7, and training dependencies. The vLLM 0.14 plugin metadata is
installed in the local venv, so the global vLLM/vLLM-Ascend metadata is not
modified.

## Run

The baseline launcher activates and verifies the runtime automatically:

```bash
cd /workspace/cann-recipes-train/llm_rl/qwen3_014_baseline/llm_rl/qwen3
bash run_qwen3_baseline_eager_tq2.sh
```

For an interactive shell, source the helper first:

```bash
source tools/activate_torch27_vllm014.sh
```

The launcher exits before starting Ray if any of Torch, torch-npu, vLLM,
vLLM-Ascend, local source origins, or the rebuilt native extension does not
match the expected runtime.

Torch 2.7 cannot run the vLLM-Ascend 0.14 eager compilation path used by the
Torch 2.9 stack. The launcher therefore sets
`VLLM_ASCEND_EAGER_COMPILE=0` while retaining vLLM eager execution.

Run only when all 16 NPU dies are free. A busy rank can make distributed KV
cache initialization fail even when the software stack is valid. The launcher
checks the `npu-smi info` process table before importing the model and refuses
to start when it finds an existing NPU process. Set `QWEN3_ALLOW_BUSY_NPU=1`
only when sharing the devices is intentional.
