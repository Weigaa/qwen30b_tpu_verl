from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "run_qwen3_baseline_eager_tq2.sh"
RUNTIME = ROOT / "tools" / "activate_cann85_a3_torch29.sh"
AUDIT = ROOT / "tools" / "check_cann85_a3_torch29_runtime.py"
HCCL_SMOKE = ROOT / "tools" / "run_hccl_collective_smoke.sh"
HCCL_SMOKE_WORKER = ROOT / "tools" / "hccl_collective_smoke.py"
CONSTRAINTS = ROOT / "tools" / "cann85_a3_torch29_runtime_constraints.txt"
NATIVE_SELECTOR = ROOT / "tools" / "select_vllm_ascend_native.sh"
WORKER = ROOT / "verl" / "single_controller" / "base" / "worker.py"


def test_baseline_removes_adafloor_rollout_config() -> None:
    source = SCRIPT.read_text()

    assert "~actor_rollout_ref.rollout.shrink_aware" in source
    assert "actor_rollout_ref.rollout.shrink_aware." not in source


def test_baseline_execution_contract() -> None:
    source = SCRIPT.read_text()

    assert 'source "${ROOT}/tools/activate_cann85_a3_torch29.sh"' in source
    assert "unset ASCEND_RT_VISIBLE_DEVICES NPU_VISIBLE_DEVICES" in source
    assert "export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1" in source
    assert "export HCCL_OP_EXPANSION_MODE=HOST" in source
    assert 'bash "${ROOT}/tools/run_hccl_collective_smoke.sh"' in source
    assert "QWEN3_SKIP_HCCL_PREFLIGHT" in source
    assert "HCCL_OP_EXPANSION_MODE=AIV" not in source
    assert 'QWEN3_ALLOW_BUSY_NPU' in source
    assert 'NPU processes already exist; refusing to start' in source
    assert 'python3 "${ROOT}/tools/check_cann85_a3_torch29_runtime.py"' in source
    assert "unset QWEN3_ALLOW_UNVALIDATED_DRIVER" in source
    assert "export TASK_QUEUE_ENABLE=2" in source
    assert "export VLLM_ENABLE_GRAPH_MODE=0" in source
    assert "export VLLM_PLUGINS=ascend," in source
    assert 'print("TORCH_VERSION=" + torch.__version__' in source
    assert "sed -n 's/^TORCH_VERSION=//p'" in source
    assert "if (major, minor) >= (2, 9)" in source
    assert "export VLLM_ASCEND_EAGER_COMPILE=0" in source
    assert "actor_rollout_ref.rollout.enforce_eager=True" in source
    assert "actor_rollout_ref.rollout.n=16" in source
    assert "data.train_batch_size=32" in source
    assert "export VLLM_ROLLOUT_MANUAL_FREE_CACHE_ENGINE=1" in source
    assert "RAY_TMPDIR" in source
    assert "/home/sharedata/qwen3_ray_tmp" in source
    assert source.index("export VLLM_ASCEND_USE_LOCAL_CUSTOM_OPP=0") < source.index(
        'python3 "${ROOT}/tools/check_cann85_a3_torch29_runtime.py"'
    )


def test_repository_local_runtime_contract() -> None:
    source = RUNTIME.read_text()

    assert ".venv_cann85_a3_torch29" in source
    assert 'source "${QWEN3_RUNTIME_VENV}/bin/activate"' in source
    assert "/cann_8.5.0_a3" in source
    assert "/cann_8.5.0_a3_stable" not in source
    assert 'source "${QWEN3_CANN85_HOME}/set_env.sh"' in source
    assert 'source "${QWEN3_NNAL85_ATB_ENV}" --cxx_abi=1' in source
    assert "asdsip/set_env.sh" not in source
    assert "torch29_cann85" in source
    assert "if ! bash" in source
    assert "failed to select the Torch 2.9/CANN 8.5" in source
    assert 'export QWEN3_LOCAL_IMPORT_ROOT="${QWEN3_RUNTIME_ROOT}"' in source
    assert 'for package in ("torch", "torch_npu")' in source
    assert "QWEN3_PYTORCH_LIB_PATHS" in source
    assert "vllm_ascend/lib64" in source
    assert "PYTHONNOUSERSITE=1" in source
    assert "PIP_REQUIRE_VIRTUALENV=1" in source
    assert "QWEN3_CANN_PYTHONPATH" in source
    assert "ascend910_9392" in source
    assert "export HCCL_OP_EXPANSION_MODE=HOST" in source
    assert "${QWEN3_DRIVER_ROOT}/lib64/driver" in source
    assert "/cann_master_9\\.1/" in source


def test_runtime_audit_is_fail_closed() -> None:
    source = AUDIT.read_text()

    assert '"torch": "==2.9.0"' in source
    assert '"torch-npu": "==2.9.0"' in source
    assert '"vllm": "==0.14.1+empty"' in source
    assert '"vllm-ascend": "==0.14.0rc1"' in source
    assert '"transformers": "==4.57.6"' in source
    assert "metadata resolved outside the selected venv" in source
    assert "is installed editable" in source
    assert "gd7de043d5" in source
    assert "g52d4acfa5" in source
    assert "VLLM_PLUGINS must be" in source
    assert "unexpected native artifact" in source
    assert "supported_driver_lines = {(25, 0), (25, 2), (25, 3), (25, 5)}" in source
    assert "only documented upstream conflicts remain" in source
    assert "check_ascend_python_runtime(cann_home)" in source
    assert "check_worker_import_contract()" in source
    assert "RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1 is required" in source
    assert "HCCL_OP_EXPANSION_MODE=HOST is required" in source


def test_hccl_preflight_covers_training_collectives() -> None:
    launcher = HCCL_SMOKE.read_text()
    worker = HCCL_SMOKE_WORKER.read_text()

    assert "QWEN3_HCCL_SMOKE_NPROC:-16" in launcher
    assert "QWEN3_HCCL_SMOKE_TIMEOUT_SECONDS:-300" in launcher
    assert "QWEN3_HCCL_SMOKE_IF_BASE_PORT" in launcher
    assert "export HCCL_IF_BASE_PORT=" in launcher
    assert "timeout --signal=TERM --kill-after=30" in launcher
    assert "HCCL_OP_EXPANSION_MODE=HOST" in launcher
    for operation in (
        "all_reduce",
        "all_gather_object",
        "reduce_scatter_tensor",
        "all_to_all_single",
        "broadcast",
        "barrier",
    ):
        assert operation in worker
    assert "dist.destroy_process_group()" in worker


def test_runtime_lock_and_native_variant_are_exact() -> None:
    constraints = CONSTRAINTS.read_text()
    selector = NATIVE_SELECTOR.read_text()

    for requirement in (
        "torch==2.9.0",
        "torch-npu==2.9.0",
        "torchvision==0.24.0",
        "transformers==4.57.6",
        "lark==1.2.2",
        "prometheus-fastapi-instrumentator==7.1.0",
        "starlette==0.50.0",
    ):
        assert requirement in constraints
    assert "torch29_cann85_vllm0140rc1_20260903" in selector


def test_ray_npu_binding_is_fail_closed() -> None:
    source = WORKER.read_text()

    assert "import logging" in source
    assert "Expected one Ray {device_name} ID per worker" in source
    assert "ASCEND_RT_VISIBLE_DEVICES to be unset before ray.init()" in source
    assert "get_torch_device().current_device() != int(local_rank)" in source
