"""CPU-only check that the actual rollout audit resets diagnostic progress."""
import ast
import contextlib
import os
from pathlib import Path
from types import SimpleNamespace as NS


def test_direct_epoch_progress_restarts_at_each_generate(monkeypatch):
    source=Path(__file__).parents[1]/'verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py'
    tree=ast.parse(source.read_text())
    function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_mode1_runtime_audit')
    namespace=dict(contextlib=contextlib,os=os,_env_flag=lambda key:True,
        torch=NS(distributed=NS(get_rank=lambda:0)),json=__import__('json'),
        logger=NS(warning=lambda *args:None))
    exec(compile(ast.Module(body=[function],type_ignores=[]),str(source),'exec'),namespace)
    config=NS(num_blocks=1838,kv_cache_groups=[NS(kv_cache_spec=NS(block_size=128))])
    core=NS(scheduler=NS(_adafloor_preemption_count=0),kv_cache_configs=[config])
    runner=NS(_runtime_progress_step=10496,kv_cache_config=config)
    rollout=NS(model_runner=runner,inference_engine=NS(llm_engine=NS(engine_core=NS(engine_core=core))))
    monkeypatch.setenv('VERL_SIDECAR_DIRECT_STEP_EVENTS','1')
    for step in (2,3,4,5):
        with namespace['_mode1_runtime_audit'](rollout,{'global_steps':step}):
            assert runner._runtime_progress_step==0
            runner._runtime_progress_step=9000+step
    monkeypatch.delenv('VERL_SIDECAR_DIRECT_STEP_EVENTS')
    with namespace['_mode1_runtime_audit'](rollout,{'global_steps':6}):
        assert runner._runtime_progress_step==9005
