import importlib.util
from pathlib import Path
import pytest
from verl.utils.rollout_seeding import prompt_token_hash

spec=importlib.util.spec_from_file_location("workload",Path(__file__).parents[1]/"tools/prepare_revision_frontier_workload.py")
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def rows():
    result=[]
    for prompt in range(32):
        ids=[100,prompt,200]
        for sample in range(8):
            result.append({"prompts":[999,999]+ids,"rollout_prompt_hash":prompt_token_hash(ids),
                "rollout_sample_index":sample,"rollout_request_seed":1000+prompt*8+sample,
                "rollout_rank":prompt//2,"input":str(prompt),"gts":"0"})
    return result


def test_identical_requests_repartitioned_without_ghost_primary_work():
    result=module.prepare(rows(),999)
    assert result["request_count"]==256 and result["ep8_requests_per_rank"]==32
    assert result["ep16_requests_per_rank"]==16
    assert all(len(row["prompt_token_ids"])==3 for row in result["requests"])


def test_mismatched_tokenization_is_not_silently_accepted():
    data=rows();data[0]["prompts"][-1]=201
    with pytest.raises(ValueError):module.prepare(data,999)
