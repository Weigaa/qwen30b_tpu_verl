import ast
from pathlib import Path

import pytest
import torch


@pytest.mark.parametrize("redundancy,local_count,replicas", [(384, 32, 4), (896, 64, 8)])
def test_mode1_replica_layout_preserves_primary_and_exact_capacity(
    monkeypatch, redundancy, local_count, replicas
):
    monkeypatch.setenv("VLLM_ASCEND_ELASTIC_EXECUTION_MODE", "1")
    path = Path(__file__).resolve().parents[1] / "vllm_ascend/eplb/core/eplb_utils.py"
    tree = ast.parse(path.read_text())
    selected = [n for n in tree.body if isinstance(n, ast.FunctionDef)
                and n.name in {"_determine_primary_rank", "build_redundant_replica_expert_map"}]
    scope = {"torch": torch}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"), scope)
    mapping = scope["build_redundant_replica_expert_map"](128, 16, redundancy)
    assert ((mapping >= 0).sum(dim=1) == local_count).all()
    assert ((mapping >= 0).sum(dim=0) == replicas).all()
    for rank in range(16):
        assert mapping[rank, rank * 8:(rank + 1) * 8].tolist() == list(range(8))
        assert sorted(mapping[rank][mapping[rank] >= 0].tolist()) == list(range(local_count))
