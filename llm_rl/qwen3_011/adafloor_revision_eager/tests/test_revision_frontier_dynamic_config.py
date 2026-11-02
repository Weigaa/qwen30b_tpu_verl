"""Check the launch contract without starting a model or accelerator."""
import ast
from pathlib import Path


def test_dynamic_refreshes_ascend_singleton_like_paper_rollout():
    root=Path(__file__).parents[1]
    code=ast.parse((root/'tools/run_revision_frontier_primary_window.py').read_text())
    updates=[node for node in ast.walk(code) if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute)
             and isinstance(node.func.value,ast.Name) and node.func.value.id=='extra' and node.func.attr=='update']
    assert len(updates)==1
    values={keyword.arg:keyword.value for keyword in updates[0].keywords}
    assert isinstance(values['refresh'],ast.Constant) and values['refresh'].value is True
    assert isinstance(values['elastic_moe_mode'],ast.Attribute) and values['elastic_moe_mode'].attr=='VLLM_ASCEND_ELASTIC_MOE_MODE'
    assert isinstance(values['init_redundancy_expert'],ast.Call)


def test_post_shrink_warmup_controls_match_verified_eager_launcher():
    root=Path(__file__).parents[1]
    code=ast.parse((root/'tools/run_revision_frontier_primary_window.py').read_text())
    values={keyword.arg:keyword.value.value for node in ast.walk(code) if isinstance(node,ast.Call)
            for keyword in node.keywords if isinstance(keyword.value,ast.Constant)}
    assert values['VLLM_ASCEND_MODE1_PARITY_PRE_MOE_WARMUP_EMPTY_CACHE']=='1'
    assert values['VLLM_ASCEND_MODE1_PARITY_POST_SHRINK_WARMUP']=='0'
    assert values['VLLM_ASCEND_MODE1_PARITY_POST_SHRINK_WARMUP_KIND']=='mc2_dispatcher_only'
