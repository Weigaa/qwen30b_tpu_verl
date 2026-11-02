import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

path = Path(__file__).parents[1] / "vllm/model_executor/layers/fused_moe/layer.py"
tree = ast.parse(path.read_text())
cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "FusedMoE")
methods = [n for n in cls.body if isinstance(n, ast.FunctionDef)
           and n.name in ("_load_w13", "_load_w2")]
namespace = {"torch": torch}
exec(compile(ast.Module(body=methods, type_ignores=[]), str(path), "exec"), namespace)


@pytest.mark.parametrize("preserve", [False, True])
@pytest.mark.parametrize("transposed", [False, True])
def test_slot_and_projection_offsets_with_tp_sharding(preserve, transposed):
    owner = SimpleNamespace(moe_config=SimpleNamespace(is_act_and_mul=True))
    w13 = torch.zeros(8, 6, 4)
    w2 = torch.zeros(8, 4, 3)
    if transposed:
        w13, w2 = w13.transpose(1, 2), w2.transpose(1, 2)
    for slot in range(8):
        for shard, marker in (("w1", slot + 1), ("w3", slot + 11)):
            source = torch.full((6, 4), float(marker))
            source[:3] = -99  # tp_rank=1 must select the second half
            if transposed:
                source = source.t()
            view = w13[slot:slot + 1] if preserve else w13[slot]
            namespace["_load_w13"](owner, view, int(transposed), shard, source, 1)
        source = torch.full((4, 6), float(slot + 21))
        source[:, :3] = -99
        if transposed:
            source = source.t()
        view = w2[slot:slot + 1] if preserve else w2[slot]
        namespace["_load_w2"](owner, view, int(not transposed), source, 1)
    if transposed:
        w13, w2 = w13.transpose(1, 2), w2.transpose(1, 2)
    for slot in range(8):
        assert torch.all(w13[slot, :3] == slot + 1)
        assert torch.all(w13[slot, 3:] == slot + 11)
        assert torch.all(w2[slot] == slot + 21)
