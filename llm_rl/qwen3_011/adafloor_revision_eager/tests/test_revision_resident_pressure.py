import importlib.util
import json
from pathlib import Path
import pytest

ROOT=Path(__file__).parents[1]
spec=importlib.util.spec_from_file_location("pressure",ROOT/"analysis_eval/analyze_revision_resident_pressure.py")
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
MODEL={"torch_dtype":"bfloat16","num_hidden_layers":48,"hidden_size":2048,
       "moe_intermediate_size":768,"num_experts":128,"num_attention_heads":32,
       "num_key_value_heads":4,"head_dim":128}
CAPS=json.loads((ROOT/"analysis_eval/paper_effective_kv_caps.json").read_text())


def test_logical_geometry_and_no_double_reservation():
    result=module.sweep(MODEL,CAPS,75776,[1])
    assert result["geometry"]["kv_bytes_per_token"]==96*1024
    assert result["rows"][0]["floors"][2]["logical_resident_bytes"]==27*2**30
    assert result["rows"][0]["floors"][2]["remaining_cap_tokens"]==131072


def test_floor_degrades_and_unsafe_is_not_full16_success():
    result=module.sweep(MODEL,CAPS,75776,[1,2,4,8])
    assert [r["selected_floor"] for r in result["rows"] if r["policy"]=="natural"]==[2,4,8,16]
    result=module.sweep(MODEL,CAPS,264192,[8])
    assert all(r["disposition"]=="unsafe_workload_rejection" for r in result["rows"])


def test_byte_caps_are_not_silently_accepted():
    bad=dict(CAPS,units="bytes")
    with pytest.raises(ValueError):module.sweep(MODEL,bad,1,[1])
