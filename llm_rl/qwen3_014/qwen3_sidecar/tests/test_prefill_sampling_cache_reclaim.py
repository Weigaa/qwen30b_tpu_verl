"""Exercise the sampling-boundary guard without initializing a NPU."""
import ast
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize("prefill,eager,native,expected", [
    (True, True, "1", True),
    (False, True, "1", False),
    (True, False, "1", False),
    (True, True, "0", False),
])
def test_reclaim_only_native_sleep_eager_prefill(prefill, eager, native, expected):
    path = Path(__file__).parents[1] / "vllm_ascend/worker/model_runner_v1.py"
    tree = ast.parse(path.read_text())
    guards = [node for node in ast.walk(tree)
              if isinstance(node, ast.If)
              and "self.with_prefill" in ast.unparse(node.test)
              and "VLLM_ROLLOUT_NATIVE_SLEEP_MODE" in ast.unparse(node.test)]
    assert len(guards) == 1
    calls = []
    npu = SimpleNamespace(
        synchronize=lambda: calls.append("sync"),
        mem_get_info=lambda: (calls.append("free") or (2**30, 64 * 2**30)),
        memory_reserved=lambda: (calls.append("reserved") or 2**30),
        empty_cache=lambda: calls.append("empty"),
    )
    env = {
        "self": SimpleNamespace(with_prefill=prefill, dp_rank=0,
                                model_config=SimpleNamespace(enforce_eager=eager)),
        "os": SimpleNamespace(getenv=lambda name, default: native),
        "torch": SimpleNamespace(npu=npu),
        "logger": SimpleNamespace(info=lambda *args: calls.append("log")),
    }
    exec(compile(ast.Module(body=guards, type_ignores=[]), str(path), "exec"), env)
    assert calls == (["sync", "free", "reserved", "empty", "free", "reserved", "log"]
                     if expected else [])
