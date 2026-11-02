import importlib.util
import json
from pathlib import Path
import pytest

HERE=Path(__file__).parent


def module(name,file):
    s=importlib.util.spec_from_file_location(name,HERE/file)
    m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m


def test_exact_two_eager_methods_keep_graph_budgets():
    m=module('eager_flags_test','eager_controls.py')
    for policy in ('vanilla','planned_f4'):
        env,cmd=m.build_training(policy)
        assert env['TASK_QUEUE_ENABLE']=='2' and env['VLLM_ASCEND_ELASTIC_ACLGRAPH']=='0'
        assert env['VLLM_ASCEND_EAGER_METADATA_SYNC_DEVICE']=='0'
        assert env['VLLM_ASCEND_MODE1_CPU_DP_METADATA_SYNC']=='1'
        assert env['VLLM_ROLLOUT_SLEEP_LEVEL']=='2' and env['VLLM_ROLLOUT_MANUAL_FREE_CACHE_ENGINE']=='0'
        assert env['FAIR_TOTAL_EPOCHS']=='2' and env['FAIR_FREEZE_ACTOR']=='0'
        assert env['FAIR_KEEP_COMPLETED_CHECKPOINTS']=='1'
        assert env['VANILLA_KV_TOKENS_PER_RANK']=='380800'
        assert env['VLLM_ASCEND_MODE1_PARITY_MAX_KV_TOKENS_FLOOR4']=='280576'
        assert cmd[-2:]==['actor_rollout_ref.rollout.enforce_eager=True','actor_rollout_ref.rollout.cudagraph_mode=NONE']
    with pytest.raises(ValueError):m.build_training('natural_f2')


def test_val_only_loads_explicit_endpoint_not_common_again(tmp_path):
    m=module('eval_flags_test','eager_controls.py');endpoint=tmp_path/'trained/global_step_5'
    env,cmd=m.build_evaluation(endpoint,tmp_path/'evaluation')
    assert f'trainer.resume_from_path={endpoint}' in cmd
    assert '++trainer.val_only=True' in cmd and 'trainer.save_freq=-1' in cmd
    assert 'actor_rollout_ref.rollout.val_kwargs.do_sample=False' in cmd
    assert 'actor_rollout_ref.rollout.val_kwargs.n=1' in cmd
    assert env['VLLM_ASCEND_ENABLE_ELASTIC_PARALLEL_SHRINK']=='0'
    assert env['TASK_QUEUE_ENABLE']=='2' and env['VLLM_ROLLOUT_SLEEP_LEVEL']=='2'
    assert env['DATASET_FRACTION']=='1.0'
    assert env['VLLM_ASCEND_EAGER_METADATA_SYNC_DEVICE']=='0'
    assert env['TRAIN_FILE']==str(m.EVAL_TRAIN) and env['TEST_FILE']==str(m.HELDOUT)


def test_fixed_hash_quartiles_are_deterministic_and_not_score_selected():
    m=module('heldout_select_test','prepare_heldout.py')
    rows=[dict(prompt_tokens=i+1,prompt_sha256=f'{i:064x}',source_index=i) for i in range(24)]
    a=m.choose(rows,8);b=m.choose(list(reversed(rows)),8)
    assert a==b and len({r['source_index'] for r in a})==8
    assert [sum(r['stratum']==i for r in a) for i in range(4)]==[2,2,2,2]


def test_empty_eval_answer_is_counted_not_suppressed(tmp_path):
    m=module('heldout_output_test','eager_controls.py');m.ROOT=tmp_path
    prep=tmp_path/'heldout_prepared';prep.mkdir();examples=[]
    for i in range(256):examples.append(dict(expected_decoded_input=f'question{i}',ground_truth=str(i),prompt_sha256=f'{i:064x}'))
    m.HELDOUT=prep/'heldout256.parquet'
    (prep/'manifest.json').write_text(json.dumps(dict(examples=examples,input_sha256={str(m.HELDOUT):'heldout'})))
    out=tmp_path/'eval';(out/'samples').mkdir(parents=True);(out/'evaluation/logs').mkdir(parents=True)
    endpoint=tmp_path/'actual_trained/global_step_5'
    log='\n'.join('Loaded sharded model checkpoint from '+str(endpoint) for _ in range(16))
    (out/'evaluation/logs/run.txt').write_text(log+'\nvalidation generation end\n[run] end_time=2026-10-24T08:00:00+0800 exit_code=0\n')
    rows=[dict(input=x['expected_decoded_input'],gts=x['ground_truth'],score=0 if i==0 else 1,
               output='' if i==0 else str(i)) for i,x in enumerate(examples)]
    (out/'samples/0.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in rows))
    result=m.parse_evaluation(out,endpoint)
    assert result['questions']==256 and result['empty_answers']==1
    assert result['mean_score']==255/256
