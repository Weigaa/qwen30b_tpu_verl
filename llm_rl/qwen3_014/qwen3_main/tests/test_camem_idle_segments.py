import ast
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest


def methods(env):
    path = Path(__file__).parents[1] / "vllm_ascend/device_allocator/camem.py"
    cls = next(n for n in ast.parse(path.read_text()).body
               if isinstance(n, ast.ClassDef) and n.name == "CaMemAllocator")
    selected = [n for n in cls.body if isinstance(n, ast.FunctionDef)
                and n.name in {"park_idle_weight_segments", "use_memory_pool", "sleep", "wake_up"}]
    env["contextmanager"] = contextmanager
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"), env)
    return env


def test_only_wholly_idle_weights_segments_can_be_parked():
    unmapped = []
    samples = [dict(address=i, active_size=0, blocks=[dict(state="inactive")]) for i in range(1, 7)]
    samples[1]["active_size"] = 1
    samples[2]["blocks"] = [dict(state="active_awaiting_free")]
    records = {i: SimpleNamespace(tag="weights", parked=False, handle=(0, 100, i, i)) for i in range(1, 6)}
    records[4].tag = "kv_cache"
    records[5].parked = True
    owner = SimpleNamespace(pointer_to_data=records)
    env = methods({
        "_park_idle_enabled": True,
        "unmap_and_release": lambda h: unmapped.append(h[2]),
        "logger": SimpleNamespace(info=lambda *args: None),
        "torch": SimpleNamespace(npu=SimpleNamespace(
            synchronize=lambda: None, mem_get_info=lambda: (1000, 2000),
            memory_snapshot=lambda: samples,
            memory=SimpleNamespace(MemPoolContext=SimpleNamespace(active_pool=lambda: None))))})
    result = env["park_idle_weight_segments"](owner)
    assert unmapped == [1]
    assert result["segments"] == 1 and result["released_bytes"] == 100
    assert records[1].parked
    assert not records[2].parked and not records[3].parked and not records[4].parked
    assert env["park_idle_weight_segments"](owner)["segments"] == 0


def test_pool_reentry_maps_parked_blocks_before_allocating():
    events = []
    @contextmanager
    def use(pool):
        events.append("enter")
        yield
    data = SimpleNamespace(tag="weights", parked=True, handle=(0, 100, 1, 1))
    env = methods({"create_and_map": lambda h: events.append("map"),
                   "torch": SimpleNamespace(npu=SimpleNamespace(memory=SimpleNamespace(use_mem_pool=use)))})
    owner = SimpleNamespace(pointer_to_data={1: data}, current_tag="default",
                            allocator_and_pools={"weights": (object(), object())})
    with env["use_memory_pool"](owner, "weights"):
        assert not data.parked
    assert events == ["map", "enter"]
    assert owner.current_tag == "default"


def test_native_sleep_and_wake_skip_parked_segments():
    calls = []
    parked = SimpleNamespace(tag="weights", parked=True, handle=(0, 100, 1, 1), cpu_backup_tensor=None)
    live = SimpleNamespace(tag="weights", parked=False, handle=(0, 100, 2, 2), cpu_backup_tensor=None)
    kv = SimpleNamespace(tag="kv_cache", parked=False, handle=(0, 100, 3, 3), cpu_backup_tensor=None)
    owner = SimpleNamespace(pointer_to_data={1: parked, 2: live, 3: kv})
    env = methods({"unmap_and_release": lambda h: calls.append(("unmap", h[2])),
                   "create_and_map": lambda h: calls.append(("map", h[2]))})
    env["sleep"](owner, offload_tags=())
    env["wake_up"](owner, ["kv_cache"])
    env["wake_up"](owner, ["weights"])
    assert calls == [("unmap", 2), ("unmap", 3), ("map", 3), ("map", 2)]


def test_do_not_park_during_an_active_pool_context():
    env = methods({"_park_idle_enabled": True,
                   "torch": SimpleNamespace(npu=SimpleNamespace(memory=SimpleNamespace(
                       MemPoolContext=SimpleNamespace(active_pool=lambda: object()))))})
    with pytest.raises(RuntimeError, match="currently active"):
        env["park_idle_weight_segments"](SimpleNamespace())
