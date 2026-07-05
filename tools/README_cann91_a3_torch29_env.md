# CANN 9.1 A3 + torch/torch_npu 2.9 venv bootstrap

This setup is for running the existing checked-out `verl`, `vllm`, and `vllm_ascend` code with an isolated CANN 9.1 A3/910C + torch/torch_npu 2.9 runtime.
It does not clone or pin repository code.

## Files

- `tools/bootstrap_cann91_a3_torch29_env.sh`: one-shot bootstrap script.
- `use_cann91_a3_torch29.sh`: generated source-able environment script.
- `.venv_cann91_a3_torch29`: generated Python venv.

## Required external assets

On a fresh machine, provide one of these two options.

Option A: existing CANN/NNAL installs:

```bash
CANN_ROOT=/path/to/cann-9.1.0/cann \
NNAL_ROOT=/path/to/Ascend_nnal_9.1.0_beta3/nnal \
bash tools/bootstrap_cann91_a3_torch29_env.sh
```

Option B: local installers:

```bash
CANN_TOOLKIT_RUN=/path/Ascend-cann-toolkit_9.1.0_linux-aarch64.run \
CANN_A3_OPS_RUN=/path/Ascend-cann-A3-ops_9.1.0_linux-aarch64.run \
NNAL_ROOT=/path/Ascend_nnal_9.1.0_beta3/nnal \
bash tools/bootstrap_cann91_a3_torch29_env.sh
```

The script installs CANN into an isolated directory by default and does not overwrite `/usr/local/Ascend`.

## Default Python stack

- `torch==2.9.0`
- `torch-npu==2.9.0`
- `torchvision==0.24.0`
- `numpy==1.26.4`
- Python `3.11`, AArch64

The default pip index is Huawei Cloud PyPI mirror. Override with `PYPI_INDEX=...` if needed.

## Use after bootstrap

```bash
cd /workspace/cann-recipes-train/llm_rl/qwen3_true_mode5_shrinkaware_staged
source .venv_cann91_a3_torch29/bin/activate
source ./use_cann91_a3_torch29.sh
KV_TOKENS=240000 bash run_mode1_floor4_kv280576_cann91_test.sh
```

## Notes

- Do not source old CANN 8.3 env after `use_cann91_a3_torch29.sh`.
- Do not source ASDSIP together with ATB unless you know the specific NNAL package requires it.
- Keep using the repository-local `verl`, `vllm`, and `vllm_ascend` paths.

## Default launcher behavior

`run_mode1_local_baseline_random_batch_threshold_multistep.sh` and
`run_mode1_local_baseline_random_batch.sh` now source the CANN 9.1 A3 +
torch/torch_npu 2.9 runtime by default inside this repository.  To intentionally
run with the caller-provided old stack, use:

```bash
USE_CANN91_A3_WITH_NNAL91=0 bash run_mode1_local_baseline_random_batch_threshold_multistep.sh
```

## 2026-07-05 Floor2 KV147456 sweep false-negative pitfall

Context: floor=2 KV cap sweep was used to verify the CANN 9.1 A3 + torch/torch_npu 2.9 runtime.  A rerun at `KV_TOKENS=147456` completed all 5 rollout steps, but the sweep initially still reported `pass=0`.

There were two separate issues:

- Real launcher drift before restoration: `internal/wj_train_grpo_qwen30b_a3b_16die_true_weight_regroup.sh` had temporarily accumulated an embedded completion-validation block.  That changed the known-good launcher hash from `f15afee3631a47bf1f06c12895fcc0bfb1222fe09289d98db7527f947d20fcfc` and produced runs that only reached `rollout_generate_start step=1` without completing the rollout.  Restoring the regroup launcher to the known-good hash fixed this part.
- Sweep false negative after restoration: `run_mode1_floor_kv_cap_sweep.sh` used a case-insensitive bare `OOM` grep.  This matched benign log words such as `headroom` from messages like `Skipping ... headroom for elastic lightweight path`, so a successful run was counted as `hard_errors_160`.  The matcher now uses a strict OOM/OutOfMemory pattern and no longer treats `headroom` as an OOM.

Validation evidence from the restored run:

- Log: `mode1_kv_cap_sweep_floor2_kv147456_20260705T050237Z/logs/wjqwen30b-a3b-record_graph_save4eagle3_20260705130237.txt`
- Launcher hash: `f15afee3631a47bf1f06c12895fcc0bfb1222fe09289d98db7527f947d20fcfc`
- Actual KV cache: `147,456 tokens`
- Completed rollout steps: `5/5`
- Rollout times: `102.175013, 91.095824, 91.067494, 91.327055, 89.656894` seconds
- Process exit: `run_exit_code=0`

Rule of thumb: for KV-cap sweeps, judge success primarily by `exit_code=0`, completion validator success, actual KV reaching the requested cap, and no strict pre/post-run fatal errors.  Do not use a bare case-insensitive `OOM` grep; it will match `headroom` and invalidate otherwise successful runs.
