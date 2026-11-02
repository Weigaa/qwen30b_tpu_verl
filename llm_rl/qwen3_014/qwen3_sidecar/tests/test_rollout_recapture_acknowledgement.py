"""Sleep wake and elastic restore must not capture the same topology twice."""
import ast
import os
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest


def _load_helper(monkeypatch, enabled=True):
    path = Path(__file__).parents[1] / "verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py"
    tree = ast.parse(path.read_text())
    helper = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                  and n.name == "_recapture_rollout_aclgraphs")
    graph = ModuleType("vllm_ascend.compilation.acl_graph")
    graph.enable_aclgraph_dispatch = lambda runner: None
    graph.reconfigure_graph_params = lambda sizes: None
    monkeypatch.setitem(sys.modules, graph.__name__, graph)
    env = dict(Any=Any, os=os, _rollout_aclgraph_enabled=lambda runner: enabled,
               torch=SimpleNamespace(npu=SimpleNamespace(synchronize=lambda: None)),
               logger=SimpleNamespace(warning=lambda *args: None))
    exec(compile(ast.Module(body=[helper], type_ignores=[]), str(path), "exec"), env)
    return env[helper.name]


def test_successful_wake_capture_acknowledges_pending_topology_capture(monkeypatch):
    capture = _load_helper(monkeypatch)
    calls = []
    runner = SimpleNamespace(_elastic_aclgraph_needs_recapture=True,
                             capture_model=lambda: calls.append("capture"))
    capture(runner, "rollout_mode_after_resume_kv_cache", force=True)
    assert calls == ["capture"]
    assert runner._elastic_aclgraph_needs_recapture is False
    # generate_sequences' topology gate must now skip a second capture.
    if runner._elastic_aclgraph_needs_recapture:
        capture(runner, "generate_after_elastic_restore_and_resume", force=True)
    assert calls == ["capture"]


def test_failed_capture_does_not_acknowledge_topology_request(monkeypatch):
    capture = _load_helper(monkeypatch)
    def fail():
        raise RuntimeError("capture failed")
    runner = SimpleNamespace(_elastic_aclgraph_needs_recapture=True, capture_model=fail)
    with pytest.raises(RuntimeError, match="capture failed"):
        capture(runner, "wake", force=True)
    assert runner._elastic_aclgraph_needs_recapture is True


def test_disabled_graph_does_not_acknowledge_topology_request(monkeypatch):
    capture = _load_helper(monkeypatch, enabled=False)
    runner = SimpleNamespace(_elastic_aclgraph_needs_recapture=True,
                             capture_model=lambda: pytest.fail("must not capture"))
    capture(runner, "wake", force=True)
    assert runner._elastic_aclgraph_needs_recapture is True
