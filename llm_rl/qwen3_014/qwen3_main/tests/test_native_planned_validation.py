import importlib.util
import json
from pathlib import Path
import subprocess

import pytest

ROOT=Path(__file__).parents[1]
spec=importlib.util.spec_from_file_location('native_planned',ROOT/'tools/validate_native_planned_memory.py')
native=importlib.util.module_from_spec(spec);spec.loader.exec_module(native)


def fixture_log():
    lines=['Initializing a V1 LLM engine (v0.14)']
    for rank in range(16):
        row=dict(step=1,rank=rank,complete=True,kv_blocks=100,kv_tokens=12800,
                 worker_kv_blocks=100,full_iterations=10,eager_iterations=1,
                 global_prefill_iterations=1)
        lines.append(f'[Rank {rank} | Local Rank 0] 2026-10-18 10:00:00,100 MODE1_STEP_RUNTIME_AUDIT '+json.dumps(row))
        lines.append(f'[Rank {rank} | Local Rank 0] 2026-10-18 10:00:00,200 Invalidated 0 rollout ACL graph entries after native_sleep_before_release; renew_pool=False.')
    lines += ['2026-10-18 10:00:00,300 => rollout_output_time_s: 1.0',
              'step:1 - response/aborted_ratio:0.0 - actor/grad_norm:0.1 - actor/lr:1e-6',
              'Training Progress: 100%', '[run] end_time=2026-10-18T10:00:01+0800 exit_code=0']
    return '\n'.join(lines)+'\n'


def test_native_handoff_proof_is_not_legacy_guard_or_physical_threshold():
    report=native.validate(fixture_log(),1,16,True)
    assert report['rank_step_handoffs']==16
    assert report['proves_free_hbm_threshold'] is False
    assert report['legacy_manual_free_guard_required'] is False


@pytest.mark.parametrize('kind', ['handoff','rank','exit','oom','decode','gradient','before_inference'])
def test_reject_incomplete_or_unsafe_native_evidence(kind):
    text=fixture_log()
    if kind=='handoff':text='\n'.join(x for x in text.splitlines() if not ('[Rank 5 |' in x and 'Invalidated' in x))
    if kind=='rank':text='\n'.join(x for x in text.splitlines() if not ('[Rank 5 |' in x and 'MODE1_STEP' in x))
    if kind=='exit':text=text.replace('exit_code=0','exit_code=1')
    if kind=='oom':text+='NPU out of memory\n'
    if kind=='decode':text=text.replace('"eager_iterations": 1','"eager_iterations": 2')
    if kind=='gradient':text=text.replace('actor/grad_norm:0.1','actor/grad_norm:nan')
    if kind=='before_inference':text=text.replace('10:00:00,200','10:00:00,050')
    with pytest.raises(ValueError):native.validate(text,1,16,True)


@pytest.mark.parametrize('native_enabled,expected', [(True,0),(False,1)])
def test_fair_validator_uses_correct_memory_backend(tmp_path,native_enabled,expected):
    source=(ROOT/'run_paper_fair_epoch1_2_from_common_epoch0.sh').read_text()
    functions=source[source.index('is_planned_variant() {'):source.index('list_matching_processes() {')]
    epoch=tmp_path/'epoch_001_mode1_planned'
    for name in ('logs','rollout_data','rollout_length'):(epoch/name).mkdir(parents=True)
    (epoch/'logs/test.txt').write_text(fixture_log())
    (epoch/'rollout_data/1.jsonl').write_text('{}\n'*512)
    (epoch/'rollout_length/length_1.txt').write_text('1\n'*512)
    cp=epoch/'checkpoints/model';(cp/'global_step_1/actor/dist_ckpt').mkdir(parents=True)
    (cp/'global_step_1/actor/dist_ckpt/fake.distcp').write_bytes(b'weight')
    (cp/'latest_checkpointed_iteration.txt').write_text('1')
    invocation=f'''
SCRIPT_DIR={ROOT}
variant=adafloor_p_f4
FAIR_TRAIN_STEPS=1
FAIR_EXPECTED_RESPONSES_PER_STEP=512
CHECKPOINT_MODEL_DIR_NAME=model
FAIR_EXPECTED_DISTCP_SHARDS=auto
FAIR_PLANNED_MIN_FREE_MIB_FLOOR=28672
FAIR_FREEZE_ACTOR=0
VERL_PAIRED_REQUEST_SAMPLING_SEEDS=0
VLLM_ROLLOUT_NATIVE_SLEEP_MODE={int(native_enabled)}
VLLM_ROLLOUT_SLEEP_LEVEL=2
VLLM_ROLLOUT_NATIVE_SLEEP_GRAPH_LIFECYCLE=1
ADAFLOOR_ACLGRAPH_MODE=FULL_DECODE_ONLY
validate_completed_epoch "$1" 001
'''
    result=subprocess.run(['bash','-c',functions+invocation,'test',str(tmp_path)],capture_output=True,text=True)
    assert result.returncode==expected,result.stderr


def test_dynamic_validator_accepts_native_but_requires_handoffs(tmp_path):
    source=(ROOT/'run_mode1_dynamic_length_aware_adaptive_floor4_epochs.sh').read_text()
    function=source[source.index('validate_planned_epoch_memory_safety() {'):source.index('build_offline_planning_history() {')]
    (tmp_path/'logs').mkdir();log=tmp_path/'logs/test.txt';log.write_text(fixture_log())
    invocation=f'''
REPO_ROOT={ROOT}
DYNAMIC_SHRINK_POLICY=planned
VLLM_ROLLOUT_NATIVE_SLEEP_MODE=1
VLLM_ROLLOUT_SLEEP_LEVEL=2
VLLM_ROLLOUT_NATIVE_SLEEP_GRAPH_LIFECYCLE=1
ADAFLOOR_ACLGRAPH_MODE=FULL_DECODE_ONLY
validate_planned_epoch_memory_safety "$1" test 1
'''
    good=subprocess.run(['bash','-c',function+invocation,'test',str(tmp_path)],capture_output=True,text=True)
    assert good.returncode==0,good.stderr
    log.write_text(fixture_log().replace('native_sleep_before_release','wrong_reason'))
    bad=subprocess.run(['bash','-c',function+invocation,'test',str(tmp_path)],capture_output=True,text=True)
    assert bad.returncode==3
