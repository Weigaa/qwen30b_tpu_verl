import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace as NS

import pytest

ROOT = Path(__file__).parents[1]
OLD = ROOT.parent / "../qwen3_011/adafloor_gragh"


def load(path,name):
    spec = importlib.util.spec_from_file_location(name,path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_request_seed_algorithm_is_identical_to_v011():
    new = load(ROOT/"verl/utils/rollout_seeding.py","newseed")
    old = load(OLD/"verl/utils/rollout_seeding.py","oldseed")
    for seed in [0,2,3,42,-1]:
        for sample in range(16):
            assert new.derive_request_seed(seed,[1,2048,151644,42],sample)==old.derive_request_seed(seed,[1,2048,151644,42],sample)


def test_step_buckets_update_padding_and_runner_capture_list(monkeypatch):
    module = load(ROOT/"vllm_ascend/compilation/step_buckets.py","buckets")
    calls = []
    graph = ModuleType("vllm_ascend.compilation.acl_graph")
    graph.disable_aclgraph_dispatch = lambda runner:calls.append("disable")
    graph.reconfigure_graph_params = lambda sizes:calls.append(("task_tables",sizes))
    monkeypatch.setitem(sys.modules,graph.__name__,graph)
    config = NS(cudagraph_capture_sizes=[32],max_cudagraph_capture_size=32,
                compute_bs_to_padded_graph_size=lambda:calls.append("padding"))
    runner=NS(compilation_config=config,cudagraph_batch_sizes=[32])
    sizes=[1,4,8,13,17,25,29,32]
    assert module.configure_step_buckets(runner,sizes)
    assert calls==["disable","padding",("task_tables",set(sizes))]
    assert runner.cudagraph_batch_sizes==sizes and config.cudagraph_capture_sizes==sizes
    assert not module.configure_step_buckets(runner,sizes)
    next_sizes=[3,6,10,18,22,28,30,32]
    assert module.configure_step_buckets(runner,next_sizes)
    assert calls[-1]==("task_tables",set(next_sizes))
    with pytest.raises(ValueError):module.configure_step_buckets(runner,[32,1])


@pytest.mark.parametrize("fault",[None,"rank","seed","kv","bucket","cap","floor"])
def test_rank_input_audit_rejects_all_protocol_mismatches(tmp_path,monkeypatch,fault):
    module=load(ROOT/"verl/utils/frozen_graph_inputs.py","audit")
    seedmod=load(ROOT/"verl/utils/rollout_seeding.py","seed")
    monkeypatch.setitem(sys.modules,"verl.utils.rollout_seeding",seedmod)
    plan=dict(step=1,rank_to_dataset_item_idx={"0":[0,1]},kv_cap=103680,
              aclgraph_capture_sizes=[1,4,8,13,17,25,29,32],tail_guard_response_cap=5120,selected_floor=2)
    file=tmp_path/"length_sorted_rank_plan.json"
    file.write_text(json.dumps([plan]))
    manifest=dict(files={file.name:hashlib.sha256(file.read_bytes()).hexdigest()},
                  rollout_n=2,seed=0,bundle_id="test")
    (tmp_path/"manifest.json").write_text(json.dumps(manifest))
    monkeypatch.setenv("ADAFLOOR_FROZEN_INPUTS_DIR",str(tmp_path))
    batch=NS(meta_info={"global_steps":1,"shrink_aware_runtime":{"selected_floor":2}},
             non_tensor_batch={"dataset_item_idx":[0,0,1,1],"rollout_sample_index":[0,1,0,1]})
    inputs=[{"prompt_token_ids":[1,2]}]*4
    seeds=[seedmod.derive_request_seed(0,[1,2],s)[1] for s in [0,1,0,1]]
    kwargs=dict(rank=0,actual_kv_tokens=103680,actual_capture_sizes=plan["aclgraph_capture_sizes"],
                actual_response_cap=5120,logger=NS(warning=lambda *args:None))
    if fault=="rank":batch.non_tensor_batch["dataset_item_idx"]=[1,1,0,0]
    if fault=="seed":seeds[0]+=1
    if fault=="kv":kwargs["actual_kv_tokens"]=131072
    if fault=="bucket":kwargs["actual_capture_sizes"]=[16,32]
    if fault=="cap":kwargs["actual_response_cap"]=4096
    if fault=="floor":batch.meta_info["shrink_aware_runtime"]["selected_floor"]=4
    if fault:
        with pytest.raises(RuntimeError):module.audit_frozen_rank_inputs(batch,inputs,seeds,**kwargs)
    else:module.audit_frozen_rank_inputs(batch,inputs,seeds,**kwargs)


def test_frozen_loader_detects_artifact_tampering(tmp_path):
    module=load(ROOT/"tools/frozen_graph_inputs.py","bundle")
    (tmp_path/"input.json").write_text("bad")
    (tmp_path/"manifest.json").write_text(json.dumps({"files":{"input.json":"wrong"}}))
    with pytest.raises(RuntimeError,match="hash mismatch"):module.verify(tmp_path)


def test_both_child_launchers_skip_planner_in_frozen_mode():
    for repo in [ROOT,OLD]:
        source=(repo/"run_mode1_local_length_sorted_e2e_adaptive_floor2.sh").read_text()
        assert 'python3 "$ADAFLOOR_FROZEN_INPUT_TOOL" install' in source
        assert "frozen plans prohibit physical-cap rewriting" in source
    source=(ROOT/"run_qwen3_frozen_graph_pair.sh").read_text()
    assert "HCCL_BUFFSIZE=800" in source and "EAGER_METADATA_SYNC_DEVICE=0" in source
    assert "VERL_PAIRED_REQUEST_SAMPLING_SEEDS=1" in source
    assert "FAIR_TRAIN_STEPS=5" in source
    import ast
    trainer=ast.parse((ROOT/"verl/trainer/ppo/ray_trainer.py").read_text())
    calls=[node for node in ast.walk(trainer) if isinstance(node,ast.Call)
           and isinstance(node.func,ast.Name) and node.func.id=="maybe_apply_aclgraph_bucket_plan"]
    assert calls, "Importing the bucket helper without calling it loses step metadata"


@pytest.mark.parametrize("repo",[ROOT,OLD])
def test_shared_seed_overrides_compose_on_both_hydra_versions(repo):
    from hydra import compose,initialize_config_dir
    with initialize_config_dir(config_dir=str(repo/"verl/trainer/config"),version_base=None):
        config=compose(config_name="ppo_megatron_trainer",overrides=[
            "actor_rollout_ref.rollout.seed=0","actor_rollout_ref.actor.megatron.seed=42",
            "actor_rollout_ref.ref.megatron.seed=42","++data.seed=42",
            "++actor_rollout_ref.rollout.engine_kwargs.vllm.num_gpu_blocks_override=810"])
    assert config.data.seed==42 and config.actor_rollout_ref.rollout.seed==0


def test_result_reader_requires_complete_runtime_audit_and_real_zero_preemptions(tmp_path):
    module=load(ROOT/"tools/compare_frozen_graph_pair.py","comparison")
    epoch=tmp_path/"run/epoch_001_mode1_natural"
    for directory in ["logs","rollout_data","rollout_length"]:(epoch/directory).mkdir(parents=True)
    (epoch/"rollout_data/1.jsonl").write_text((json.dumps({"score":1})+"\n")*512)
    (epoch/"rollout_length/length_1.txt").write_text("42\n"*512)
    lines=["[run] end_time=2026-10-16T00:00:00+0800 exit_code=0",
           "=> rollout_output_time_s: 10","=> rollouts speed tokens/s: 2200"]
    for rank in range(16):
        lines.append("FROZEN_GRAPH_INPUT_AUDIT "+json.dumps({"step":1,"rank":rank}))
        lines.append("FROZEN_GRAPH_GENERATION_AUDIT "+json.dumps({"step":1,"rank":rank,"complete":True,"preemptions":0}))
    logfile=epoch/"logs/test.txt"
    logfile.write_text("\n".join(lines))
    audits,metrics=module.read_arm(tmp_path,1)
    assert len(audits)==16 and metrics[0]["reward"]==1
    logfile.write_text("\n".join(lines).replace('"preemptions": 0','"preemptions": 1',1))
    with pytest.raises(RuntimeError,match="preemption"):module.read_arm(tmp_path,1)


@pytest.mark.parametrize("arm,allocator,override",[
    ("v011","expandable_segments:True","none (v011 cold-init and per-step KV policy)"),
    ("v014","expandable_segments:False","++actor_rollout_ref.rollout.engine_kwargs.vllm.num_gpu_blocks_override=810")])
def test_initialization_allocator_contract_is_not_copied_between_sleep_backends(arm,allocator,override):
    import subprocess
    result=subprocess.run(["bash",str(ROOT/"run_qwen3_frozen_graph_pair.sh"),arm,"dry-run"],
                          text=True,capture_output=True,check=True)
    assert f"allocator_init={allocator} initial_kv_override={override}" in result.stdout
    assert "KV=103680,253184,253184,253184,353408" in result.stdout
    assert "HCCL=800" in result.stdout
    assert "sampler=AscendSampler topk_topp_optimization=1 local_custom_topk=0" in result.stdout
    assert "allocator_phase_policy=rollout:False,trainer:True actual_NPU_setter=enabled" in result.stdout


def test_fair_launcher_preserves_fast_sampler_and_matching_checkpoint_frequency():
    source=(ROOT/"run_qwen3_frozen_graph_pair.sh").read_text()
    assert "export VLLM_ASCEND_ENABLE_TOPK_TOPP_OPTIMIZATION=1" in source
    assert "ENABLE_TOPK_TOPP_OPTIMIZATION=0" not in source
    assert "trainer.save_freq=1" in source


def test_comparison_rejects_missing_or_generic_sampler_evidence(tmp_path):
    module=load(ROOT/"tools/compare_frozen_graph_pair.py","sampler_evidence")
    path=tmp_path/"run/epoch_001_mode1_natural/logs"
    path.mkdir(parents=True)
    with pytest.raises(RuntimeError,match="operator audits"):
        module.require_optimized_v011_sampler(tmp_path)
    record=dict(sampler="AscendSampler",filter="torch_npu.npu_top_k_top_p")
    (path/"test.txt").write_text(("FROZEN_SAMPLER_OPERATOR_AUDIT "+json.dumps(record)+"\n")*16)
    assert len(module.require_optimized_v011_sampler(tmp_path))==16
    (path/"test.txt").write_text(("FROZEN_SAMPLER_OPERATOR_AUDIT "+json.dumps(dict(sampler="Sampler",filter="pytorch_sort"))+"\n")*16)
    with pytest.raises(RuntimeError,match="operator audits"):
        module.require_optimized_v011_sampler(tmp_path)


@pytest.mark.parametrize("repo", [ROOT, OLD])
def test_allocator_helper_reaches_npu_setter_and_audits_only_success(repo, monkeypatch):
    import ast
    import os
    source = ast.parse((repo / "verl/utils/device.py").read_text())
    function = next(n for n in source.body if isinstance(n, ast.FunctionDef)
                    and n.name == "set_expandable_segments")
    calls, records = [], []
    def setter(settings):
        calls.append(settings)
    namespace = dict(is_cuda_available=False, is_npu_available=True,
                     torch=NS(npu=NS(memory=NS(_set_allocator_settings=setter,
                                              get_allocator_backend=lambda: "native")),
                              distributed=NS(is_initialized=lambda: True, get_rank=lambda: 3)),
                     os=os, json=json, logger=NS(warning=lambda fmt, value: records.append(json.loads(value))))
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(repo), "exec"), namespace)
    monkeypatch.setenv("ADAFLOOR_FROZEN_INPUTS_DIR", "/frozen")
    namespace["set_expandable_segments"](False)
    namespace["set_expandable_segments"](True)
    assert calls == ["expandable_segments:False", "expandable_segments:True"]
    assert [(r["rank"],r["phase"],r["expandable_segments"]) for r in records] == [
        (3, "rollout", False), (3, "trainer", True)]
    def failing_setter(settings):
        raise RuntimeError("setter failed")
    namespace["torch"].npu.memory._set_allocator_settings = failing_setter
    with pytest.raises(RuntimeError, match="setter failed"):
        namespace["set_expandable_segments"](False)
    assert len(records) == 2


def test_allocator_phase_comparison_fails_closed(tmp_path):
    module = load(ROOT / "tools/compare_frozen_graph_pair.py", "allocator_evidence")
    path = tmp_path / "run/epoch_001_mode1_natural/logs"
    path.mkdir(parents=True)
    with pytest.raises(RuntimeError, match="Missing"):
        module.require_v011_allocator_phases(tmp_path, 5)
    records = [dict(rank=r, phase=phase, expandable_segments=(phase == "trainer"),
                    applied=True, backend="native")
               for r in range(16) for phase in ("rollout", "trainer") for _ in range(5)]
    logfile = path / "test.txt"
    logfile.write_text("\n".join("NPU_ALLOCATOR_PHASE_AUDIT " + json.dumps(r) for r in records))
    assert module.require_v011_allocator_phases(tmp_path, 5)["v011_phase_setter"] == "PASS"
    records[0]["expandable_segments"] = True
    logfile.write_text("\n".join("NPU_ALLOCATOR_PHASE_AUDIT " + json.dumps(r) for r in records))
    with pytest.raises(RuntimeError, match="Invalid"):
        module.require_v011_allocator_phases(tmp_path, 5)
