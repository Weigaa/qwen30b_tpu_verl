import ast
import json
import os
import subprocess
from pathlib import Path

from tools.verify_qwen3_adafloor_v014_gate import verify


ROOT = Path(__file__).resolve().parents[1]


def _function(path: Path, name: str) -> ast.FunctionDef:
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"missing function {name} in {path}")


def test_rollout_dataclass_accepts_default_off_shrink_config() -> None:
    source = (ROOT / "verl/workers/config/rollout.py").read_text()
    assert "shrink_aware: dict = field(default_factory=dict)" in source


def test_graph_is_invalidated_before_group_change_and_recaptured_after() -> None:
    worker = ROOT / "vllm_ascend/worker/worker_v1.py"
    rebuild = _function(worker, "rebuild_elastic_ep_group")
    restore = _function(worker, "restore_elastic_parallel_groups")

    calls = [
        (node.func.attr, node.lineno)
        for node in ast.walk(rebuild)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in {
            "_invalidate_elastic_aclgraphs",
            "_rebuild_group",
            "_refresh_elastic_parallel_state",
            "_recapture_elastic_aclgraphs",
        }
    ]
    first = {}
    for name, line in calls:
        first.setdefault(name, line)
    assert first["_invalidate_elastic_aclgraphs"] < first["_rebuild_group"]
    assert first["_rebuild_group"] < first["_refresh_elastic_parallel_state"]
    assert first["_refresh_elastic_parallel_state"] < first["_recapture_elastic_aclgraphs"]

    restore_calls = {
        node.func.attr: node.lineno
        for node in ast.walk(restore)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    assert restore_calls["_invalidate_elastic_aclgraphs"] < restore_calls["_rebuild_group"]

    engine_source = (ROOT / "vllm/v1/engine/llm_engine.py").read_text()
    env_source = (ROOT / "vllm_ascend/envs.py").read_text()
    assert "or envs_ascend.VLLM_ASCEND_ELASTIC_ACLGRAPH" in engine_source
    assert '"VLLM_ASCEND_ELASTIC_ACLGRAPH"' in env_source


def test_ascend_worker_accepts_v014_split_execute_sample_protocol() -> None:
    worker_path = ROOT / "vllm_ascend/worker/worker_v1.py"
    execute = _function(worker_path, "execute_model")
    none_returns = [
        node
        for node in ast.walk(execute)
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.Compare)
        and isinstance(node.test.left, ast.Name)
        and node.test.left.id == "output"
        and any(isinstance(op, ast.Is) for op in node.test.ops)
        and any(isinstance(value, ast.Constant) and value.value is None for value in node.test.comparators)
    ]
    assert none_returns

    init_device = _function(worker_path, "init_device")
    init_calls = [
        node
        for node in ast.walk(init_device)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "init_workspace_manager"
    ]
    assert len(init_calls) == 1
    assert "self.device = self._init_device()" in worker_path.read_text()


def test_lossless_expert_p2p_uses_zero_offset_aliases() -> None:
    moe_source = (ROOT / "vllm_ascend/ops/fused_moe_legacy.py").read_text()
    worker_source = (ROOT / "vllm_ascend/worker/worker_v1.py").read_text()
    assert "def _npu_zero_offset_alias_for_p2p" in moe_source
    assert "torch_npu.npu_change_data_ptr" in moe_source
    assert "def get_lossless_expert_npu_slot_recv_buffers" in moe_source
    assert "get_lossless_expert_npu_slot_recv_buffers" in worker_source
    assert (
        "                mapping_mismatch = 0\n"
        "                if new_log2phy_cpu is not None:" in worker_source
    )


def test_lossless_restore_uses_host_owned_primary_row_count() -> None:
    moe_path = ROOT / "vllm_ascend/ops/fused_moe_legacy.py"
    row_count = _function(moe_path, "_get_lossless_primary_prefix_row_count")
    restore = _function(moe_path, "restore_lossless_full_world_primary_layout")

    row_count_source = ast.unparse(row_count)
    restore_source = ast.unparse(restore)
    assert "_lossless_primary_prefix_row_count_host" in row_count_source
    assert row_count_source.index("_lossless_primary_prefix_row_count_host") < row_count_source.index(
        "elastic_original_expert_map"
    )
    assert "self.elastic_original_expert_map =" in restore_source
    assert "self._lossless_primary_prefix_row_count_host =" in restore_source


def test_elastic_legacy_moe_is_an_opaque_full_graph_boundary() -> None:
    moe_source = (ROOT / "vllm_ascend/ops/fused_moe_legacy.py").read_text()
    model_source = (ROOT / "vllm/model_executor/models/qwen3_moe.py").read_text()
    assert 'op_name="elastic_ascend_moe_forward"' in moe_source
    assert 'dispatch_key="PrivateUse1"' in moe_source
    assert "fake_impl=elastic_ascend_moe_forward_fake" in moe_source
    assert "forward_context.no_compile_layers[layer_name]" in moe_source
    assert 'type(self.experts).__module__.endswith("fused_moe_legacy")' in model_source
    assert "and not vllm_config.model_config.enforce_eager" in model_source
    assert "cudagraph_mode.has_full_cudagraphs()" in model_source
    assert "torch.ops.vllm.elastic_ascend_moe_forward" in model_source
    assert "not self.skip_invalid_topk_check and not self.use_aclgraph" in moe_source
    assert "self.use_aclgraph = (" in moe_source
    assert "cudagraph_mode.has_full_cudagraphs()" in moe_source
    assert "if torch.any(mask):" not in moe_source


def test_graph_cache_releases_npugraph_and_renews_pool() -> None:
    source = (ROOT / "vllm_ascend/compilation/acl_graph.py").read_text()
    assert 'hasattr(aclgraph, "reset")' in source
    assert "aclgraph.reset()" in source
    assert "platform_cls._global_graph_pool = None" in source
    assert "renew_pool: bool = False" in source
    assert "def release_aclgraph_pool()" in source
    assert "def rebind_aclgraph_pool()" in source
    assert "reset_npugraph_trees" in source
    assert "_strong_lifecycle_output" in source


def test_full_decode_recapture_isolates_live_kv_slots() -> None:
    source = (ROOT / "vllm_ascend/worker/model_runner_v1.py").read_text()
    assert "def _isolate_live_kv_during_aclgraph_capture" in source
    assert '"block_tables", [])' in source
    assert "mapping.fill_(PAD_SLOT_ID)" in source
    assert "mapping.copy_(saved)" in source
    capture = _function(ROOT / "vllm_ascend/worker/model_runner_v1.py",
                        "capture_model")
    assert any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "_isolate_live_kv_during_aclgraph_capture"
        for node in ast.walk(capture)
    )


def test_restore_defers_graph_recapture_until_next_generate() -> None:
    worker = (ROOT / "vllm_ascend/worker/worker_v1.py").read_text()
    rollout = (ROOT / "verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py").read_text()
    assert "self.model_runner._elastic_aclgraph_needs_recapture = True" in worker
    assert '"generate_after_elastic_restore_and_resume"' in rollout
    assert "force=True" in rollout
    assert "self.model_runner._elastic_aclgraph_needs_recapture = False" in rollout


def test_gate_modes_are_workload_matched() -> None:
    launcher = ROOT / "run_qwen3_adafloor_v014_gate.sh"
    source = launcher.read_text()
    assert "export TRAIN_BATCH_SIZE=32" in source
    assert "export ROLLOUT_N=1" in source
    assert "export VERL_ELASTIC_TAIL_VALIDATE_LEVEL_TOKENS=8,16,32,64,64" in source
    assert "actor_rollout_ref.rollout.seed=101" in source
    assert "actor_rollout_ref.actor.optim.lr=0.0" in source
    assert "+actor_rollout_ref.actor.megatron.override_transformer_config.moe_shared_expert_overlap=False" in source
    assert "actor_rollout_ref.rollout.shrink_aware.assignment_policy=$ASSIGNMENT_POLICY" in source
    assert '"actor_rollout_ref.rollout.shrink_aware.stage_survivor_ranks=$STAGE_RANKS"' in source
    subprocess.run(["bash", "-n", str(launcher)], check=True)

    eager = subprocess.run(
        ["bash", str(launcher), "dry-run"],
        cwd=ROOT,
        env={**os.environ, "ADAFLOOR_V014_GATE_MODE": "eager"},
        check=True,
        text=True,
        capture_output=True,
    ).stdout
    graph = subprocess.run(
        ["bash", str(launcher), "dry-run"],
        cwd=ROOT,
        env={**os.environ, "ADAFLOOR_V014_GATE_MODE": "graph"},
        check=True,
        text=True,
        capture_output=True,
    ).stdout
    assert "task_queue_enable=2" in eager
    assert "manual_free_cache_engine=0" in eager
    assert "sleep_level=1" in eager
    assert "cudagraph_mode=NONE" in eager
    assert "assignment_policy=manual_5_2_1" in eager
    assert "task_queue_enable=1" in graph
    assert "cudagraph_mode=FULL_DECODE_ONLY" in graph
    assert "capture_sizes=[2]" in graph
    assert "export VLLM_ASCEND_ELASTIC_ACLGRAPH=1" in source
    assert "export VLLM_ASCEND_MC2_MIN_EP_SIZE=$TARGET_FLOOR" in source
    assert "export VLLM_ROLLOUT_GPU_MEMORY_UTILIZATION=0.83" in source

    floor2 = subprocess.run(
        ["bash", str(launcher), "dry-run"],
        cwd=ROOT,
        env={
            **os.environ,
            "ADAFLOOR_V014_GATE_MODE": "graph",
            "ADAFLOOR_V014_GATE_FLOOR": "2",
        },
        check=True,
        text=True,
        capture_output=True,
    ).stdout
    assert "target_floor=2" in floor2
    assert "manual_free_cache_engine=0" in floor2
    assert "sleep_level=1" in floor2
    assert "shrink_stages=16_to_8_to_4_to_2" in floor2
    assert "shrink_stages=[8,4,2]" in floor2
    assert "final_survivor_ranks=[14,15]" in floor2
    assert "stage_survivor_ranks=[[8,9,10,11,12,13,14,15],[12,13,14,15],[14,15]]" in floor2
    assert "assignment_policy=default" in floor2
    assert "num_gpu_blocks_override=869" in floor2
    assert "+actor_rollout_ref.rollout.engine_kwargs.vllm.num_gpu_blocks_override=869" in floor2
    assert "VLLM_ASCEND_POST_SHRINK_MOE_DISPATCH_LOW_FLOOR_HEADROOM_BYTES=0" in source
    assert "VLLM_ASCEND_POST_SHRINK_PREFILL_ALLTOALL_HEADROOM_BYTES=0" in source
    assert "VLLM_ASCEND_POST_RESTORE_DP_HEADROOM_BYTES=0" in source
    assert "VLLM_ASCEND_POST_RESTORE_EP_HEADROOM_BYTES=0" in source
    assert "VLLM_ASCEND_POST_RESTORE_MOE_DISPATCH_HEADROOM_BYTES=0" in source
    assert "VLLM_ASCEND_FIRST_LIVE_PREFILL_LOW_FLOOR_HEADROOM_BYTES=0" in source
    assert "VLLM_ASCEND_EXTRA_ELASTIC_SAFETY_HEADROOM_BYTES=0" in source
    assert "VLLM_ASCEND_FLOOR_PREALLOC_HEADROOM_SAFETY_BYTES=0" in source

    natural_floor2 = subprocess.run(
        ["bash", str(launcher), "dry-run"],
        cwd=ROOT,
        env={
            **os.environ,
            "ADAFLOOR_V014_GATE_MODE": "graph",
            "ADAFLOOR_V014_GATE_FLOOR": "2",
            "ADAFLOOR_V014_GATE_TARGET_POLICY": "natural",
        },
        check=True,
        text=True,
        capture_output=True,
    ).stdout
    assert "target_policy=natural" in natural_floor2
    assert "shrink_aware.target_policy=natural" in natural_floor2


def test_gate_rejects_unknown_target_policy() -> None:
    launcher = ROOT / "run_qwen3_adafloor_v014_gate.sh"
    result = subprocess.run(
        ["bash", str(launcher), "dry-run"],
        cwd=ROOT,
        env={
            **os.environ,
            "ADAFLOOR_V014_GATE_TARGET_POLICY": "oracle",
        },
        text=True,
        capture_output=True,
    )
    assert result.returncode == 2
    assert "must be planned or natural" in result.stderr


def test_tail_validation_caps_reach_task_and_rollout_workers() -> None:
    constants = (ROOT / "verl/trainer/constants_ppo.py").read_text()
    ray_base = (ROOT / "verl/single_controller/ray/base.py").read_text()

    assert '"VERL_ELASTIC_TAIL_VALIDATE_"' in constants
    assert '"VERL_ELASTIC_TAIL_VALIDATE_"' in ray_base
    assert "_ray_worker_passthrough_env().items()" in ray_base
    assert "import os" in ray_base


def test_multistage_survivor_sets_reach_the_training_scheduler() -> None:
    config = (ROOT / "verl/trainer/config/rollout/rollout.yaml").read_text()
    assignment = (
        ROOT / "verl/experimental/dataset/shrink_aware_assignment.py"
    ).read_text()

    assert "stage_survivor_ranks: null" in config
    assert '_cfg_get(\n        shrink_cfg, "stage_survivor_ranks", None)' in assignment
    assert "stage_survivor_ranks=stage_survivor_ranks" in assignment


def test_rollout_plan_selection_uses_worker_rank_without_default_process_group() -> None:
    source = (
        ROOT / "verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py"
    ).read_text()
    assert 'global_rank = int(os.environ.get("RANK", "-1"))' in source
    assert 'if _env_flag("VLLM_ASCEND_SHRINK_AWARE_ENABLE"):' in source
    assert "preserving the " in source
    assert 'os.environ["VLLM_ASCEND_SHRINK_AWARE_ENABLE"] = "0"' in source


def test_gate_verifier_uses_v014_serialized_rollout_schema() -> None:
    source = (ROOT / "tools/verify_qwen3_adafloor_v014_gate.py").read_text()
    assert '"length_*.txt"' in source
    assert "def _step_files(" in source
    assert 'required = {"request_id", "input", "output", "gts", "score", "step"}' in source
    assert 'row["decoded_response_length"]' not in source
    assert 'row["responses"]' not in source
    assert '"rollout_jsonl_byte_equal": True' in source
    assert "does not retain token IDs" in source
    assert "expected_steps" in source


def _write_verifier_arm(
    root: Path,
    mode: str,
    steps: int,
    target_floor: int = 4,
    target_policy: str = "planned",
) -> None:
    (root / "logs").mkdir(parents=True)
    (root / "rollout_data").mkdir()
    (root / "rollout_length").mkdir()
    lines = []
    if mode == "graph":
        lines.extend(["FULL_DECODE_ONLY compilation enabled on NPU"] * 16)
        lines.append("Replaying aclgraph")
    for step in range(1, steps + 1):
        lines.extend([
            "AdaFloor role plan loaded: "
            f"target_policy={target_policy} locality=1.0000"
        ] * 16)
        lines.extend([
            "AdaFloor staged trigger: stage=donor "
            "target=[8, 9, 10, 11, 12, 13, 14, 15] "
            f"policy={target_policy}"
        ] * 16)
        for rank in range(8, 16):
            lines.append(
                "Elastic parallel shrink done: "
                f"rank={rank} active_ranks=[8, 9, 10, 11, 12, 13, 14, 15] "
                "dp_size=8 ep_size=8"
            )
            if mode == "graph":
                lines.append(
                    "Elastic ACLGraph topology capture finished: "
                    f"rank={rank} active_ranks=[8, 9, 10, 11, 12, 13, 14, 15]"
                )
        lines.extend([
            "AdaFloor staged trigger: stage=wave2 "
            "target=[12, 13, 14, 15] "
            f"policy={target_policy}"
        ] * 8)
        for rank in range(12, 16):
            lines.append(
                "Elastic parallel shrink done: "
                f"rank={rank} active_ranks=[12, 13, 14, 15] "
                "dp_size=4 ep_size=4"
            )
            if mode == "graph":
                lines.append(
                    "Elastic ACLGraph topology capture finished: "
                    f"rank={rank} active_ranks=[12, 13, 14, 15]"
                )
        if target_floor == 2:
            lines.extend([
                "AdaFloor staged trigger: stage=stage3 target=[14, 15] "
                f"policy={target_policy}"
            ] * 4)
            for rank in range(14, 16):
                lines.append(
                    "Elastic parallel shrink done: "
                    f"rank={rank} active_ranks=[14, 15] dp_size=2 ep_size=2"
                )
                if mode == "graph":
                    lines.append(
                        "Elastic ACLGraph topology capture finished: "
                        f"rank={rank} active_ranks=[14, 15]"
                    )
        lines.extend(
            f"Elastic parallel restore done: rank={rank} dp_size=16 ep_size=16"
            for rank in range(16)
        )
        if mode == "graph" and step < steps:
            lines.extend([
                "Finished recapturing rollout ACL graphs at "
                "generate_after_elastic_restore_and_resume"
            ] * 16)
        lines.append(f"=> rollout_output_time_s: {10 + step}.0")
        lines.append(f"=> rollouts speed tokens/s: {64 - step}.0")
        lines.append(
            f"step:{step} - critic/score/mean:0.5 - response/aborted_ratio:0.0 "
            "- perf/total_num_tokens:1024"
        )
        rows = [
            {
                "request_id": str(index),
                "input": f"prompt-{index}",
                "output": f"answer-{step}-{index}",
                "gts": f"gt-{index}",
                "score": float(index == 0),
                "step": step,
            }
            for index in range(32)
        ]
        (root / "rollout_data" / f"{step}.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows)
        )
        (root / "rollout_length" / f"length_{step}.txt").write_text(
            "".join("21\n" for _ in rows)
        )
    (root / "logs" / "run.txt").write_text("\n".join(lines) + "\n")


def test_gate_verifier_accepts_exact_two_step_pair(tmp_path: Path) -> None:
    eager = tmp_path / "eager"
    graph = tmp_path / "graph"
    _write_verifier_arm(eager, "eager", 2)
    _write_verifier_arm(graph, "graph", 2)

    result = verify(eager, graph, 2)

    assert result["status"] == "PASS"
    assert result["paired"]["steps"] == 2
    assert result["arms"]["eager_tq2"]["rows"] == 64
    assert result["arms"]["graph_tq1"]["restore_markers"] == 32


def test_gate_verifier_accepts_floor2_pair(tmp_path: Path) -> None:
    eager = tmp_path / "eager"
    graph = tmp_path / "graph"
    _write_verifier_arm(eager, "eager", 1, target_floor=2)
    _write_verifier_arm(graph, "graph", 1, target_floor=2)

    result = verify(eager, graph, 1, target_floor=2)

    assert result["status"] == "PASS"
    assert result["paired"]["target_floor"] == 2
    assert result["arms"]["eager_tq2"]["stage_markers"] == {
        "8": 8,
        "4": 4,
        "2": 2,
    }


def test_gate_verifier_accepts_natural_floor2_pair(tmp_path: Path) -> None:
    eager = tmp_path / "eager"
    graph = tmp_path / "graph"
    _write_verifier_arm(
        eager, "eager", 1, target_floor=2, target_policy="natural"
    )
    _write_verifier_arm(
        graph, "graph", 1, target_floor=2, target_policy="natural"
    )

    result = verify(
        eager, graph, 1, target_floor=2, target_policy="natural"
    )

    assert result["status"] == "PASS"
    assert result["paired"]["target_policy"] == "natural"
    assert result["arms"]["graph_tq1"]["trigger_markers"] == {
        "8": 16,
        "4": 8,
        "2": 4,
    }
