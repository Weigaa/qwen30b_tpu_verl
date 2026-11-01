# Qwen3 vLLM 0.14 runtime

`run_qwen3_baseline_eager_tq2.sh` is locked to one repository-local runtime:

- Python 3.11 with `include-system-site-packages = false`
- PyTorch 2.9.0, torch_npu 2.9.0 and torchvision 0.24.0
- vLLM 0.14.1 (`0.14.1+empty`) and vLLM-Ascend 0.14.0rc1
- CANN runtime, HCCL and OPP 8.5.0 for A3
- NNAL/ATB 8.5.0.B160
- this host's `ascend910_9392` target
- HCCL host-side collective expansion (`HCCL_OP_EXPANSION_MODE=HOST`)

The clean venv is `.venv_cann85_a3_torch29`. Its predecessor that inherited
global site-packages is retained as `.venv_cann85_a3_torch29_mixed_20260903`
for rollback evidence, but must not be activated.

The selected CANN tree is:

```text
/workspace/cann-recipes-train/llm_rl/qwen3/cann_8.5.0_a3/cann-8.5.0
```

Do not use `cann_8.5.0_a3_stable`: that directory contains mixed or profiled
HCCL artifacts. Do not source system CANN, CANN 9.1, ASDSIP, or another
repository's vLLM paths in the same process. The activation script discards
inherited Python/CANN library paths and enables `PYTHONNOUSERSITE=1`.

## Dependency contract

The Python runtime is constrained by
`tools/cann85_a3_torch29_runtime_constraints.txt`. vLLM and vLLM-Ascend are
ordinary (non-editable) installs in the venv, while imports of `verl`, `vllm`
and `vllm_ascend` intentionally resolve to this repository.

vLLM 0.14.1 and vLLM-Ascend 0.14.0rc1 have two incompatible upstream
requirements. The Ascend plugin's constraints win because they select the NPU
implementations used here:

- `xgrammar==0.1.30` (vLLM declares `==0.1.29`)
- `opencv-python-headless==4.11.0.86` (vLLM declares `>=4.13.0`)

The plugin also lists `transformers<=4.57.1` only in its isolated build-system
requirements, while its runtime metadata requires `transformers>=4.57.3`.
This runtime builds without isolation and uses the repository's
`transformers==4.57.6` pin.

`arctic-inference==0.1.1` is installed by
`tools/install_arctic_inference_aarch64.sh`. Its unused CUDA extension does not
build on this AArch64 host, so the helper installs the same Python package with
that native extension disabled. It is not one of the enabled vLLM plugins.

## Native artifacts

The active vLLM-Ascend extension was rebuilt against this exact venv, CANN 8.5
and `ascend910_9392`. Its immutable runtime copy is under:

```text
.runtime_backups/torch29_cann85_vllm0140rc1_20260903/vllm_ascend
```

`tools/select_vllm_ascend_native.sh torch29_cann85` installs that variant, and
the runtime audit verifies its SHA-256 values and dynamic-library resolution.
Older native variants remain under `.runtime_backups` for rollback.

## Preflight

Activate and run the audit with:

```bash
source tools/activate_cann85_a3_torch29.sh
export VLLM_PLUGINS=ascend,ascend_kv_connector,ascend_model_loader,ascend_service_profiling
export VLLM_ASCEND_USE_LOCAL_CUSTOM_OPP=0
export VLLM_ASCEND_USE_LOCAL_CUSTOM_OP_API_LIB=0
python3 tools/check_cann85_a3_torch29_runtime.py
```

The audit fails closed on package versions and locations, editable installs,
foreign paths, CANN/ATB metadata, native hashes, unresolved libraries and host
driver compatibility. A read-only import diagnostic can bypass only the driver
version check:

```bash
QWEN3_ALLOW_UNVALIDATED_DRIVER=1 python3 tools/check_cann85_a3_torch29_runtime.py
```

The training launcher unsets that override and cannot use it to bypass the
driver gate.

The AIV/AICPU HCCL expansion path fails a standalone collective on this host
with runtime error `507018`. The activation script therefore selects `HOST`
expansion before `torch_npu` is imported. Every non-dry training launch also
runs a 16-rank smoke test covering all-reduce, object all-gather,
reduce-scatter, all-to-all, broadcast, barrier, and communicator teardown.
The trainer is not started unless every operation succeeds. The preflight can
be run directly with:

```bash
bash tools/run_hccl_collective_smoke.sh
```

`QWEN3_SKIP_HCCL_PREFLIGHT=1` is available only when intentionally avoiding
the startup cost after this exact host and runtime have already been checked.

The eager baseline preserves the remote run's paired settings
`TASK_QUEUE_ENABLE=2` and `VLLM_ROLLOUT_MANUAL_FREE_CACHE_ENGINE=1`. A local
diagnostic run changed both settings and left 13-15 GiB resident after rollout;
that non-parity run then failed at a 189.37 MiB actor workspace allocation.

## Current host driver

The CANN 8.5 version mapping supports Ascend HDK 25.0.x, 25.2.x, 25.3.x and
25.5.x. This host's `25.0.rc1.3` driver is therefore accepted by the formal
audit; upgrading the host driver is not required for this stack.

Validate without starting training:

```bash
bash run_qwen3_baseline_eager_tq2.sh dry-run
```
