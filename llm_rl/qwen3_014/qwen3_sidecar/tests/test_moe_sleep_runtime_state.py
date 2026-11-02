import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

spec = importlib.util.spec_from_file_location(
    "moe_runtime_state", Path(__file__).parents[1]
    / "vllm_ascend/ops/fused_moe/runtime_state.py")
state = importlib.util.module_from_spec(spec)
spec.loader.exec_module(state)


@pytest.mark.parametrize("local_experts", [8, 16, 32, 64])
def test_recover_discarded_table_preserving_address(local_experts):
    ids = torch.tensor([i % local_experts for i in range(128)], dtype=torch.int32)
    original = ids.clone()
    address = ids.data_ptr()
    ids.fill_(-123)  # simulate discarded native-sleep allocation contents
    assert state.restore_expert_ids_(ids, local_experts) == 128
    assert torch.equal(ids, original)
    assert ids.data_ptr() == address
    assert state.restore_expert_ids_(ids, local_experts) == 0


def test_restore_view_preserves_shared_storage():
    storage = torch.full((132,), -1, dtype=torch.int32)
    ids = storage[2:130]
    address = ids.data_ptr()
    state.restore_expert_ids_(ids, 8)
    assert ids.data_ptr() == address
    assert storage[:2].tolist() == [-1, -1]
    assert storage[-2:].tolist() == [-1, -1]
    assert ids[:9].tolist() == [0, 1, 2, 3, 4, 5, 6, 7, 0]


@pytest.mark.parametrize("backend", ["moe", "fused_moe"])
def test_weight_wake_repairs_the_loaded_backend_registry(monkeypatch, backend):
    names = [f"vllm_ascend.ops.{b}.moe_comm_method" for b in ("moe", "fused_moe")]
    for name in names:
        monkeypatch.delitem(sys.modules, name, raising=False)
    ids = torch.zeros(128, dtype=torch.int32)
    alias = ids.view(16, 8)
    address = ids.data_ptr()
    dispatcher = SimpleNamespace(expert_ids_per_ep_rank=ids, num_local_experts=8)
    method = SimpleNamespace(token_dispatcher=dispatcher)
    # Alias the same dispatcher under two keys; it must be restored only once.
    name = f"vllm_ascend.ops.{backend}.moe_comm_method"
    monkeypatch.setitem(sys.modules, name, SimpleNamespace(
        _MoECommMethods={"alltoall": method, "alias": method}))
    reports = state.restore_loaded_moe_runtime_state_after_sleep()
    assert len(reports) == 1
    assert reports[0]["registry"] == name
    assert reports[0]["changed"] == 112
    assert dispatcher.expert_ids_per_ep_rank is ids
    assert ids.data_ptr() == address
    assert alias.tolist() == [list(range(8))] * 16
    assert state.restore_loaded_moe_runtime_state_after_sleep()[0]["changed"] == 0
    assert names[1 if backend == "moe" else 0] not in sys.modules


def test_both_loaded_registries_are_repaired_independently(monkeypatch):
    tables = []
    for backend, local_experts in (("moe", 8), ("fused_moe", 16)):
        ids = torch.full((128,), -1, dtype=torch.int32)
        tables.append((ids, ids.data_ptr(), local_experts))
        dispatcher = SimpleNamespace(expert_ids_per_ep_rank=ids,
                                     num_local_experts=local_experts)
        monkeypatch.setitem(sys.modules,
                            f"vllm_ascend.ops.{backend}.moe_comm_method",
                            SimpleNamespace(_MoECommMethods={
                                "alltoall": SimpleNamespace(token_dispatcher=dispatcher)}))
    reports = state.restore_loaded_moe_runtime_state_after_sleep()
    assert len(reports) == 2
    assert [report["changed"] for report in reports] == [128, 128]
    for ids, address, local_experts in tables:
        assert ids.data_ptr() == address
        assert ids.tolist() == [i % local_experts for i in range(128)]
