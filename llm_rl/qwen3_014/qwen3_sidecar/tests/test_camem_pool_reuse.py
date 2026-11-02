import ast
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest


def test_tag_pools_reused_and_tag_restored_on_exception():
    path = Path(__file__).parents[1] / "vllm_ascend/device_allocator/camem.py"
    tree = ast.parse(path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "CaMemAllocator")
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "use_memory_pool")
    created, reused = [], []

    @contextmanager
    def create(*callbacks):
        pair = (object(), object())
        created.append(pair)
        yield pair

    @contextmanager
    def use(pool):
        reused.append(pool)
        yield

    env = {"contextmanager": contextmanager,
           "use_memory_pool_with_allocator": create,
           "torch": SimpleNamespace(npu=SimpleNamespace(memory=SimpleNamespace(use_mem_pool=use))),
           "CaMemAllocator": SimpleNamespace(default_tag="default")}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), env)
    owner = SimpleNamespace(current_tag="default", allocator_and_pools={}, pointer_to_data={},
                            python_malloc_callback=None, python_free_callback=None)
    for tag in ("weights", "kv_cache", "weights", "weights"):
        with env["use_memory_pool"](owner, tag):
            assert owner.current_tag == tag
        assert owner.current_tag == "default"
    assert len(created) == 2
    assert reused == [created[0][0], created[0][0]]
    with pytest.raises(ValueError):
        with env["use_memory_pool"](owner, "weights"):
            raise ValueError("failed reload")
    assert owner.current_tag == "default"
