import importlib.util
import sys
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace


def test_only_regenerable_training_caches_are_cleared(monkeypatch):
    path = Path(__file__).parents[1] / "verl/utils/training_cache_lifecycle.py"
    spec = importlib.util.spec_from_file_location("training_cache_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    @lru_cache(maxsize=32)
    def forward(x):
        return object()
    forward(1)
    forward(2)
    inv_freq = object()
    owner = SimpleNamespace(RotaryEmbedding=SimpleNamespace(forward=forward, inv_freq=inv_freq))
    monkeypatch.setitem(sys.modules, "megatron.core.models.common.embeddings.rotary_pos_embedding", owner)
    unrelated = {"semantic_state": object()}
    monkeypatch.setitem(sys.modules, "vllm_ascend.attention", SimpleNamespace(cache=unrelated))
    monkeypatch.delitem(sys.modules, "megatron.core.models.common.embeddings.yarn_rotary_pos_embedding", raising=False)
    monkeypatch.delitem(sys.modules, "megatron.core.transformer.utils", raising=False)
    monkeypatch.setattr(module, "torch", SimpleNamespace(npu=SimpleNamespace(
        synchronize=lambda: None, empty_cache=lambda: None,
        mem_get_info=lambda: (100, 200), memory_allocated=lambda: 50)))
    report = module.clear_training_tensor_caches()
    assert report["caches"][0]["entries"] == 2
    assert forward.cache_info().currsize == 0
    assert owner.RotaryEmbedding.inv_freq is inv_freq
    assert len(unrelated) == 1
    assert module.clear_training_tensor_caches()["caches"] == []


def test_global_communication_scratch_is_recreated_on_demand(monkeypatch):
    import torch
    path = Path(__file__).parents[1] / "verl/utils/training_cache_lifecycle.py"
    spec = importlib.util.spec_from_file_location("scratch_cache_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    scratch = SimpleNamespace(buffer={('mpu', torch.float32): torch.ones(16)})
    state = SimpleNamespace(_GLOBAL_MEMORY_BUFFER=scratch, communication_group=object())
    group = state.communication_group
    monkeypatch.setitem(sys.modules, "megatron.core.parallel_state", state)
    device = SimpleNamespace(synchronize=lambda: None, empty_cache=lambda: None,
                             mem_get_info=lambda: (100, 200), memory_allocated=lambda: 50)
    monkeypatch.setattr(module, "torch", SimpleNamespace(npu=device, Tensor=torch.Tensor))
    report = module.clear_training_tensor_caches()
    entry = next(x for x in report['caches'] if x['name'] == 'megatron.GlobalMemoryBuffer')
    assert entry['buffers'][0]['bytes'] == 64
    assert state._GLOBAL_MEMORY_BUFFER is scratch and scratch.buffer == {}
    assert state.communication_group is group


def test_training_dispatch_transients_not_semantic_tables(monkeypatch):
    import torch
    path = Path(__file__).parents[1] / "verl/utils/training_cache_lifecycle.py"
    spec = importlib.util.spec_from_file_location("dispatch_cache_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    cls = type("TrainingDispatcher", (), {"__module__": "mindspeed.token_dispatcher"})
    dispatch = cls()
    dispatch.routing_map = torch.ones(8, 8, dtype=torch.bool)
    dispatch.local_expert_indices = [0, 1]
    dispatch.expert_ids_per_ep_rank = torch.arange(8)
    persistent = dispatch.expert_ids_per_ep_rank
    layer = SimpleNamespace(token_dispatcher=dispatch)
    model = SimpleNamespace(named_modules=lambda: [("mlp", layer)])
    device = SimpleNamespace(synchronize=lambda: None, empty_cache=lambda: None,
                             mem_get_info=lambda: (100, 200), memory_allocated=lambda: 50)
    monkeypatch.setattr(module, "torch", SimpleNamespace(npu=device, Tensor=torch.Tensor))
    module.clear_training_tensor_caches([model])
    assert dispatch.routing_map is None
    assert dispatch.local_expert_indices == [0, 1]
    assert dispatch.expert_ids_per_ep_rank is persistent
