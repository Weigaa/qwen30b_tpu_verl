import importlib.util
from pathlib import Path
import torch
import pytest

spec=importlib.util.spec_from_file_location("hfmap",Path(__file__).parents[1]/"tools/audit_revision_common_hf_export.py")
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def fixture():
    cfg={"num_hidden_layers":1,"num_experts":1,"num_attention_heads":4,"num_key_value_heads":2,"head_dim":1,"hidden_size":2}
    state={key:torch.zeros(1) for key in module.KNOWN}
    for key in module.KNOWN:
        if key.startswith(module.LAYER):state[key]=torch.zeros(1,2)
    state[module.LAYER+"self_attention.linear_qkv.weight"]=torch.arange(16).reshape(1,8,2)
    state[module.LAYER+"mlp.experts.experts.linear_fc1.weight"]=torch.arange(8).reshape(1,1,4,2)
    state[module.LAYER+"mlp.experts.experts.linear_fc2.weight"]=torch.zeros(1,1,2,2)
    return state,cfg


def test_grouped_qkv_and_expert_gate_up_are_not_naively_chunked():
    state,cfg=fixture();out=dict(module.expected_weights(state,cfg))
    assert out["model.layers.0.self_attn.q_proj.weight"].tolist()==[[0,1],[2,3],[8,9],[10,11]]
    assert out["model.layers.0.self_attn.k_proj.weight"].tolist()==[[4,5],[12,13]]
    assert out["model.layers.0.self_attn.v_proj.weight"].tolist()==[[6,7],[14,15]]
    assert out["model.layers.0.mlp.experts.0.up_proj.weight"].tolist()==[[4,5],[6,7]]


def test_unknown_tensor_layout_rejected():
    state,cfg=fixture();state["extra"]=torch.zeros(1)
    with pytest.raises(ValueError):list(module.expected_weights(state,cfg))
