import ast
from contextlib import contextmanager
import importlib.util
from pathlib import Path
from types import ModuleType, SimpleNamespace as NS

import pytest

ROOT = Path(__file__).parents[1]


def load_capacity():
    spec = importlib.util.spec_from_file_location("capacity", ROOT / "vllm/v1/engine/mode1_kv_capacity.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def config(blocks=1024):
    return NS(num_blocks=blocks,
              kv_cache_groups=[NS(kv_cache_spec=NS(block_size=128))],
              kv_cache_tensors=[NS(size=blocks * 131072, shared_by=["layer0"])])


@pytest.mark.parametrize("tokens,blocks", [(131072,1024),(280576,2192),(380800,2975)])
def test_exact_uniform_block_and_tensor_capacity(tokens, blocks):
    old = config()
    new, actual = load_capacity().resize_uniform_kv_configs([old], tokens)
    assert actual == tokens and new[0].num_blocks == blocks
    assert new[0].kv_cache_tensors[0].size == blocks * 131072
    assert old.num_blocks == 1024


def test_block_rounding_and_unsupported_layouts():
    module = load_capacity()
    assert module.resize_uniform_kv_configs([config()], 131073)[1] == 131072
    hybrid = config()
    hybrid.kv_cache_groups *= 2
    with pytest.raises(ValueError, match="uniform"):
        module.resize_uniform_kv_configs([hybrid], 131072)
    malformed = config()
    malformed.kv_cache_tensors[0].size += 1
    with pytest.raises(ValueError, match="integral"):
        module.resize_uniform_kv_configs([malformed], 131072)


def test_capacity_pool_reentry_does_not_remap_retired_other_capacity():
    path = ROOT / "vllm_ascend/device_allocator/camem.py"
    cls = next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.ClassDef) and n.name == "CaMemAllocator")
    methods = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in {"use_memory_pool", "select_step_pool"}]
    events = []
    @contextmanager
    def enter(pool):
        events.append(("enter", pool))
        yield
    env = dict(contextmanager=contextmanager, CaMemAllocator=NS(default_tag="default"),
               create_and_map=lambda h: events.append(("map",h)),
               torch=NS(npu=NS(memory=NS(use_mem_pool=enter))))
    exec(compile(ast.Module(body=methods, type_ignores=[]), str(path), "exec"), env)
    old = NS(tag="weights", parked=True, handle=1, pool_key="weights:step_capacity=2")
    new = NS(tag="weights", parked=True, handle=2, pool_key="weights:step_capacity=4")
    owner = NS(current_tag="default", pointer_to_data={1:old,2:new}, allocator_and_pools={"weights:step_capacity=4":("new_pool",None)})
    env["select_step_pool"](owner,"weights",4)
    with env["use_memory_pool"](owner,"weights"):
        assert owner.current_tag == "weights"
    assert events == [("map",2),("enter","new_pool")]
    assert old.parked and not new.parked


def test_native_prepare_is_before_wake_and_no_manual_offload():
    worker = (ROOT / "verl/workers/megatron_workers.py").read_text()
    start = worker.index('with simple_timer("rollout_mode_before_generate"')
    region = worker[start:worker.index('with simple_timer("generate_sequences"', start)]
    assert region.index("prepare_step(prompts.meta_info)") < region.index("self.rollout_mode()")
    storage = (ROOT / "vllm_ascend/worker/mode1_step_storage.py").read_text()
    assert "offload_model_weights" not in storage
    assert "parameter.data = torch.empty(0" in storage
    assert 'allocator.use_memory_pool(tag="weights")' in storage
    launcher = (ROOT / "run_qwen3_v014_sleep2_multistep_validation.sh").read_text()
    assert "VLLM_ASCEND_MODE1_ADAPTIVE_KV_RESIZE=1" in launcher
    assert "VLLM_ASCEND_MODE1_PREEMPTION_AUDIT=1" in launcher
    assert "380800" in launcher


def test_preemption_counter_does_not_depend_on_log_stats(monkeypatch):
    import os
    path = ROOT / "vllm/v1/core/sched/scheduler.py"
    tree = ast.parse(path.read_text())
    method = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "_preempt_request")
    env = dict(Request=object, RequestStatus=NS(RUNNING="running",PREEMPTED="preempted"),
               os=os, logger=NS(warning=lambda *args:None))
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), env)
    owner = NS(kv_cache_manager=NS(free=lambda r:None), encoder_cache_manager=NS(free=lambda r:None),
               log_stats=False, waiting=NS(prepend_request=lambda r:None))
    request = NS(status="running",num_computed_tokens=42,spec_token_ids=[],num_preemptions=0,request_id="r")
    monkeypatch.setenv("VLLM_ASCEND_MODE1_PREEMPTION_AUDIT","1")
    env["_preempt_request"](owner,request,0.0)
    assert owner._adafloor_preemption_count == 1
    assert request.num_preemptions == 1 and request.num_computed_tokens == 0


@pytest.mark.parametrize("sleeping,live,queue", [(False,0,[]),(True,1,[]),(True,0,[1])])
def test_engine_rejects_non_boundary_resize_before_mutation(sleeping, live, queue):
    path = ROOT / "vllm/v1/engine/core.py"
    method = next(n for n in ast.walk(ast.parse(path.read_text()))
                  if isinstance(n, ast.FunctionDef) and n.name=="resize_kv_cache_for_mode1_step")
    # The real pure sizing module is provided without importing the NPU stack.
    import sys
    from unittest.mock import patch
    env = dict()
    exec(compile(ast.Module(body=[method],type_ignores=[]),str(path),"exec"),env)
    owner = NS(model_executor=NS(is_sleeping=sleeping),
               scheduler=NS(get_num_unfinished_requests=lambda:live,requests={}),batch_queue=queue)
    with patch.dict(sys.modules,{"vllm.v1.engine.mode1_kv_capacity":load_capacity()}):
        with pytest.raises(RuntimeError):
            env[method.name](owner,280576,4)


@pytest.mark.parametrize("fails", [False, True])
def test_runtime_audit_reports_actual_counter_and_completion(fails):
    import contextlib
    import json
    path = ROOT / "verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py"
    method = next(n for n in ast.parse(path.read_text()).body
                  if isinstance(n, ast.FunctionDef) and n.name=="_mode1_runtime_audit")
    messages = []
    env = dict(contextlib=contextlib,json=json,_env_flag=lambda name:True,
               os=NS(getenv=lambda name:"4"),
               torch=NS(distributed=NS(get_rank=lambda:3)),
               logger=NS(warning=lambda fmt,data:messages.append(json.loads(data))))
    exec(compile(ast.Module(body=[method],type_ignores=[]),str(path),"exec"),env)
    scheduler = NS(_adafloor_preemption_count=5)
    core = NS(scheduler=scheduler,kv_cache_configs=[config(2192)])
    rollout = NS(inference_engine=NS(llm_engine=NS(engine_core=NS(engine_core=core))),model_runner=NS())
    try:
        with env[method.name](rollout,{"global_steps":3}):
            scheduler._adafloor_preemption_count += 2
            rollout.model_runner._mode1_forward_audit["full_iterations"] = 42
            if fails: raise RuntimeError("test generation failure")
    except RuntimeError:
        assert fails
    assert len(messages)==1
    assert messages[0]["preemptions"]==2
    assert messages[0]["kv_tokens"]==280576
    assert messages[0]["complete"] is not fails
    assert messages[0]["rank"]==3 and messages[0]["full_iterations"]==42


def test_engine_capacity_rebuilds_scheduler_but_defers_allocation():
    import sys
    from unittest.mock import patch
    path = ROOT / "vllm/v1/engine/core.py"
    method = next(n for n in ast.walk(ast.parse(path.read_text()))
                  if isinstance(n, ast.FunctionDef) and n.name=="resize_kv_cache_for_mode1_step")
    calls = []
    config_cache = NS(num_gpu_blocks=1024,num_gpu_blocks_override=1024,block_size=128)
    def scheduler_factory(**kwargs):
        return NS(get_num_unfinished_requests=lambda:0,requests={},kwargs=kwargs)
    owner = NS(model_executor=NS(is_sleeping=True),batch_queue=None,
               scheduler=scheduler_factory(),kv_cache_configs=[config()],
               vllm_config=NS(cache_config=config_cache,
                              parallel_config=NS(decode_context_parallel_size=1,prefill_context_parallel_size=1),
                              scheduler_config=NS(get_scheduler_cls=lambda:scheduler_factory)),
               structured_output_manager=object(),_include_finished_set=True,log_stats=False,
               collective_rpc=lambda name,args:calls.append((name,args)))
    env = dict(generate_scheduler_kv_cache_config=lambda configs:configs[0],
               logger=NS(info=lambda *args:None))
    exec(compile(ast.Module(body=[method],type_ignores=[]),str(path),"exec"),env)
    with patch.dict(sys.modules,{"vllm.v1.engine.mode1_kv_capacity":load_capacity()}):
        assert env[method.name](owner,280576,4)
        assert not env[method.name](owner,280576,4)
    assert len(calls)==1 and calls[0][0]=="prepare_mode1_native_step_storage"
    assert calls[0][1][0].num_blocks==2192 and calls[0][1][1:]==(4,True)
    assert config_cache.num_gpu_blocks==2192
    assert owner.scheduler.kwargs["include_finished_set"] is True
