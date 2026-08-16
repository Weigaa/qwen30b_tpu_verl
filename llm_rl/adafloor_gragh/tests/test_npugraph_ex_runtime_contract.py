from pathlib import Path
import re

from omegaconf import OmegaConf


ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / "npugraph_ex_runtime"
SMOKE = ROOT / "run_qwen3_npugraph_ex_fixed16_smoke.sh"
PLANNER_RUNNER = ROOT / "run_mode1_local_length_sorted_e2e_adaptive_floor4.sh"


def test_runtime_versions_and_official_npugraph_backend_are_pinned() -> None:
    assert "__version__ = version = '0.14.1'" in (
        RUNTIME / "vllm/_version.py"
    ).read_text()
    assert "__version__ = version = '0.14.0rc1'" in (
        RUNTIME / "vllm_ascend/_version.py"
    ).read_text()
    compiler = (RUNTIME / "vllm_ascend/compilation/compiler_interface.py").read_text()
    assert "import torchair" in compiler
    assert "torchair.get_npu_backend" in compiler
    assert 'config.mode = "reduce-overhead"' in compiler
    assert (RUNTIME / "vllm_ascend/shrink_aware/__init__.py").is_file()
    assert (RUNTIME / "vllm_ascend/shrink_aware/assignment.py").is_file()
    assert (RUNTIME / "vllm_ascend/shrink_aware/planner.py").is_file()
    assert (RUNTIME / "vllm_ascend/shrink_aware/trigger.py").is_file()
    hdp = (RUNTIME / "patches/verl/utils/hybrid_data_parallel/hdp.py").read_text()
    assert "def set_hdp_max_token_len" in hdp
    assert (RUNTIME / "patches/verl/features/rollout_optimize/__init__.py").is_file()
    assert (RUNTIME / "patches/vllm_ascend/spec_decode/sam_proposer.py").is_file()


def test_npugraph_runtime_rollout_config_accepts_adafloor_contract() -> None:
    config = OmegaConf.load(RUNTIME / "verl/trainer/config/rollout/rollout.yaml")
    assert config.seed == 0
    assert config.shrink_aware.enable_shrink_aware_scheduling is False
    assert config.shrink_aware.shrink_aware_mode == "off"
    assert list(config.shrink_aware.shrink_stages) == [8, 4]

    dataclass_source = (RUNTIME / "verl/workers/config/rollout.py").read_text()
    assert "seed: int = 0" in dataclass_source
    assert "shrink_aware: dict = field(default_factory=dict)" in dataclass_source


def test_legacy_moe_expert_map_setter_accepts_vllm_014_plain_attribute() -> None:
    source = (RUNTIME / "vllm_ascend/ops/fused_moe_legacy.py").read_text()
    assert 'elif "_expert_map" in self.__dict__:' in source
    assert 'object.__setattr__(self, "_expert_map", value)' in source


def test_legacy_moe_invalid_topk_check_does_not_branch_during_compile() -> None:
    source = (RUNTIME / "vllm_ascend/ops/fused_moe_legacy.py").read_text()
    assert "and not torch.compiler.is_compiling()" in source
    assert "if torch.any(invalid_mask):" in source
    assert "Invalid remapped topk_ids" in source


def test_planner_and_execution_runtime_can_be_selected_independently() -> None:
    source = PLANNER_RUNNER.read_text()
    assert 'RUNTIME_TREE="${ADAFLOOR_RUNTIME_TREE:-$PATCH_TREE}"' in source
    assert 'CONFIG_DIR="${CONFIG_DIR:-$RUNTIME_TREE/verl/trainer/config}"' in source
    assert 'PYTHONPATH="$RUNTIME_TREE:$PATCH_TREE' in source
    assert 'cd "$RUNTIME_TREE"' in source
    assert '"$PATCH_TREE/tools/build_mode1_length_sorted_e2e_plan.py"' in source


def test_rollout_defers_chunked_prefill_token_limit_validation_to_vllm() -> None:
    rollout = (
        RUNTIME
        / "verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py"
    ).read_text()
    scheduler = (RUNTIME / "vllm/config/scheduler.py").read_text()
    assert "The older VERL-side check had that condition" in rollout
    assert "Enable chunked prefill, max_num_batched_tokens is smaller" not in rollout
    assert "and not self.enable_chunked_prefill" in scheduler


def test_unsupported_add_rms_norm_bias_uses_supported_graph_fallback() -> None:
    smoke = SMOKE.read_text()
    layernorm = (RUNTIME / "vllm_ascend/ops/layernorm.py").read_text()
    constants = (RUNTIME / "verl/trainer/constants_ppo.py").read_text()
    ray_base = (RUNTIME / "verl/single_controller/ray/base.py").read_text()
    key = "VLLM_ASCEND_FORCE_TORCH_NPU_ADD_RMS_NORM"
    assert f"export {key}=1" in smoke
    assert key in constants
    assert key in ray_base
    assert f'"{key}", "0"' in layernorm
    assert "torch_npu.npu_add_rms_norm(" in layernorm


def test_cann85_grouped_matmul_group_list_is_normalized_to_int64() -> None:
    smoke = SMOKE.read_text()
    for relative_path in (
        "vllm_ascend/ops/fused_moe/moe_mlp.py",
        "vllm_ascend/ops/moe/moe_mlp.py",
    ):
        moe_mlp = (RUNTIME / relative_path).read_text()
        assert "if group_list.dtype != torch.int64:" in moe_mlp
        assert "torch_npu.npu_dtype_cast(group_list, torch.int64)" in moe_mlp
        assert "group_list = group_list.to(dtype=torch.int64)" not in moe_mlp
        assert f'"$RUNTIME_TREE/{relative_path}"' in smoke
    assert '"$RUNTIME_TREE/vllm_ascend/ops/fused_moe/moe_mlp.py"' in smoke


def test_npugraph_ex_has_opt_in_legacy_moe_boundary_but_official_smoke_disables_it() -> None:
    smoke = SMOKE.read_text()
    legacy = (RUNTIME / "vllm_ascend/ops/fused_moe_legacy.py").read_text()
    qwen = (RUNTIME / "vllm/model_executor/models/qwen3_moe.py").read_text()
    constants = (RUNTIME / "verl/trainer/constants_ppo.py").read_text()
    ray_base = (RUNTIME / "verl/single_controller/ray/base.py").read_text()
    key = "VLLM_ASCEND_NPUGRAPH_EX_MOE_CUSTOM_OP_BOUNDARY"
    assert f"export {key}=0" in smoke
    assert key in constants
    assert key in ray_base
    assert 'op_name="ascend_legacy_moe_forward"' in legacy
    assert "fake_impl=ascend_legacy_moe_forward_fake" in legacy
    assert 'dispatch_key="PrivateUse1"' in legacy
    assert "torch.ops.vllm.ascend_legacy_moe_forward(" in qwen
    assert '"$RUNTIME_TREE/vllm/model_executor/models/qwen3_moe.py"' in smoke


def test_official_graph_runtime_disables_nz_weight_layout_before_worker_import() -> None:
    source = SMOKE.read_text()
    common_moe = (RUNTIME / "vllm_ascend/ops/common_fused_moe.py").read_text()
    legacy_moe = (RUNTIME / "vllm_ascend/ops/fused_moe_legacy.py").read_text()
    rollout = (
        RUNTIME
        / "verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py"
    ).read_text()
    assert "export VLLM_ASCEND_ENABLE_NZ=0" in source
    assert "'enable_nz=false'" in source
    assert 'os.environ["VLLM_ASCEND_ENABLE_NZ"] = "0"' in rollout
    assert "maybe_trans_nz(layer.w13_weight.data)" in common_moe
    assert "maybe_trans_nz(layer.w2_weight.data)" in common_moe
    assert "ACL_FORMAT_FRACTAL_NZ" not in common_moe
    assert "maybe_trans_nz(layer.w13_weight.data)" in legacy_moe
    assert "maybe_trans_nz(layer.w2_weight.data)" in legacy_moe
    assert "ACL_FORMAT_FRACTAL_NZ" not in legacy_moe
    assert '"$RUNTIME_TREE/vllm_ascend/ops/common_fused_moe.py"' in source
    assert '"$RUNTIME_TREE/vllm_ascend/ops/fused_moe_legacy.py"' in source


def test_graph_runtime_materializes_moe_group_lists_as_int64() -> None:
    legacy_dispatch = (
        RUNTIME / "vllm_ascend/ops/moe/token_dispatcher.py"
    ).read_text()
    legacy_mlp = (RUNTIME / "vllm_ascend/ops/moe/moe_mlp.py").read_text()
    current_dispatch = (
        RUNTIME / "vllm_ascend/ops/fused_moe/token_dispatcher.py"
    ).read_text()
    current_mlp = (
        RUNTIME / "vllm_ascend/ops/fused_moe/moe_mlp.py"
    ).read_text()

    for source in (legacy_dispatch, current_dispatch):
        assert "torch_npu.npu_dtype_cast(expert_tokens, torch.int64)" in source
        assert "expert_tokens = expert_tokens.to(torch.int64)" not in source
    for source in (legacy_mlp, current_mlp):
        assert "if group_list.dtype != torch.int64:" in source
        assert "torch_npu.npu_dtype_cast(group_list, torch.int64)" in source

    smoke = SMOKE.read_text()
    assert '"$RUNTIME_TREE/vllm_ascend/ops/moe/token_dispatcher.py"' in smoke
    assert '"$RUNTIME_TREE/vllm_ascend/ops/fused_moe/token_dispatcher.py"' in smoke
    assert "'moe_group_list_materialization=npu_dtype_cast_int64'" in smoke


def test_fixed16_smoke_uses_npugraph_ex_without_shrink_and_rejects_task_queue_2() -> None:
    source = SMOKE.read_text()
    assert 'RUNTIME_TREE="$SCRIPT_DIR/npugraph_ex_runtime"' in source
    assert "FORCE_SELECTED_FLOORS=16,16,16,16,16" in source
    assert "FAST_STEP_SUBSET_STEPS=1" in source
    assert "VLLM_ENABLE_GRAPH_MODE=1" in source
    assert "ROLLOUT_ENFORCE_EAGER=False" in source
    assert "TASK_QUEUE_MODE=${ADAFLOOR_NPUGRAPH_TASK_QUEUE_ENABLE:-1}" in source
    assert "TASK_QUEUE_ENABLE=2 is unsupported by torch-npu NPUGraph capture" in source
    assert 'if [[ "$TASK_QUEUE_MODE" != 1 ]]' in source
    assert 'TASK_QUEUE_ENABLE="$TASK_QUEUE_MODE"' in source
    assert 'VLLM_ROLLOUT_TASK_QUEUE_ENABLE="$TASK_QUEUE_MODE"' in source
    assert "CAPTURE_SIZES=${ADAFLOOR_NPUGRAPH_CAPTURE_SIZES:-'[1,2,4,8,16,32]'}" in source
    assert "GPU_MEMORY_UTILIZATION=${ADAFLOOR_NPUGRAPH_GPU_MEMORY_UTILIZATION:-0.9}" in source
    assert "MAX_NUM_BATCHED_TOKENS=${ADAFLOOR_NPUGRAPH_MAX_NUM_BATCHED_TOKENS:-17408}" in source
    assert 'ROLLOUT_GPU_MEMORY_UTILIZATION="$GPU_MEMORY_UTILIZATION"' in source
    assert 'ROLLOUT_MAX_NUM_BATCHED_TOKENS="$MAX_NUM_BATCHED_TOKENS"' in source
    assert "MAX_JOBS=${MAX_JOBS:-1}" in source
    assert "TE_PARALLEL_COMPILER=${TE_PARALLEL_COMPILER:-1}" in source
    assert "MAX_COMPILE_CORE_NUMBER=${MAX_COMPILE_CORE_NUMBER:-1}" in source
    assert "VLLM_ROLLOUT_DELAY_GRAPH_CAPTURE_UNTIL_WEIGHT_LOAD=1" in source
    assert "VLLM_ROLLOUT_CAPTURE_GRAPH_AFTER_WEIGHT_LOAD=1" in source
    assert re.search(
        r'actor_rollout_ref\.rollout\.cudagraph_capture_sizes="\$CAPTURE_SIZES"',
        source,
    )
