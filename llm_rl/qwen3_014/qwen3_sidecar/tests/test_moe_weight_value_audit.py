import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

spec = importlib.util.spec_from_file_location(
    "weight_value_audit", Path(__file__).parents[1]
    / "vllm_ascend/ops/fused_moe/weight_value_audit.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


@pytest.mark.parametrize("transposed", [False, True])
def test_audit_loader_and_execution_layout(transposed):
    gate = torch.arange(12).reshape(3, 4).float()
    up = gate + 100
    down = gate.t() + 200
    w13 = torch.cat((gate, up))
    w2 = down
    if transposed:
        w13, w2 = w13.t(), w2.t()
    experts = SimpleNamespace(w13_weight=w13.unsqueeze(0), w2_weight=w2.unsqueeze(0),
                              ep_rank=0, _map_global_expert_id_to_local_expert_id=lambda i: 0)
    model = SimpleNamespace(model=SimpleNamespace(layers=[SimpleNamespace(
        mlp=SimpleNamespace(experts=experts))]))
    audit = module.MoEWeightValueAudit(model)
    list(audit.capture((f"model.layers.0.mlp.experts.0.{p}.weight", value)
                      for p, value in (("gate_proj", gate), ("up_proj", up), ("down_proj", down))))
    assert audit.check("test")["checked"] == 3
    experts.w13_weight.zero_()
    with pytest.raises(RuntimeError, match="weight value audit failed"):
        audit.check("corrupted")
