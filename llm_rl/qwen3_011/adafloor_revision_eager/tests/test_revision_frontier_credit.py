import importlib.util
from pathlib import Path
import pytest

spec=importlib.util.spec_from_file_location('frontier',Path(__file__).parents[1]/'analysis_eval/analyze_revision_frontier_windows.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def test_late_work_not_counted_and_boundary_consistent():
    rows=[{'completion_time':110,'credited':True},{'completion_time':120,'credited':True},{'completion_time':121,'credited':False}]
    assert len(module.credit(rows,100,20))==2


def test_false_credit_or_invalid_clock_rejected():
    with pytest.raises(ValueError):module.credit([{'completion_time':121,'credited':True}],100,20)
    with pytest.raises(ValueError):module.credit([{'completion_time':99}],100,20)


def test_torn_last_record_does_not_corrupt_next_shard_or_gain_credit(tmp_path):
    import json
    first=(json.dumps({'sidecar_epoch':0,'shard_index':0,'prompt_id':0})+'\n').encode()+b'{"unfinished":'
    second=(json.dumps({'sidecar_epoch':0,'shard_index':1,'prompt_id':1})+'\n').encode()
    (tmp_path/'outputs.jsonl.shard0').write_bytes(first);(tmp_path/'outputs.jsonl.shard1').write_bytes(second)
    (tmp_path/'outputs.jsonl').write_bytes(first+second)
    rows,excluded,paths=module.read_sidecar_shards(tmp_path)
    assert len(rows)==2 and len(excluded)==1 and len(paths)==2


def test_bad_committed_middle_record_must_fail(tmp_path):
    value=b'{}\nnot-json\n'
    (tmp_path/'outputs.jsonl.shard0').write_bytes(value);(tmp_path/'outputs.jsonl').write_bytes(value)
    with pytest.raises((ValueError,KeyError)):module.read_sidecar_shards(tmp_path)


def test_detached_rpc_completions_are_real_quorum_evidence_not_duplicate_votes():
    text='Elastic parallel shrink done: rank=1 active_ranks=[1]\n'
    text+='Elastic parallel shrink rpc done: global_rank=0 active_ranks=[1]\n'
    text+='Elastic parallel shrink rpc done: global_rank=1 active_ranks=[1]\n'
    assert module.shrink_reporters(text,[1])=={0,1}
