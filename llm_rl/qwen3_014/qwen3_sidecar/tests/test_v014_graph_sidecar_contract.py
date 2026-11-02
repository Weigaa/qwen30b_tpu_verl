from pathlib import Path
import importlib.util
import json
import pytest

ROOT=Path(__file__).parents[1]


def test_timeline_messages_use_python_logger_not_tracking():
    import ast
    tree=ast.parse((ROOT/'verl/trainer/ppo/ray_trainer.py').read_text())
    calls=[node for node in ast.walk(tree) if isinstance(node,ast.Call)
           and node.args and isinstance(node.args[0],ast.Constant)
           and isinstance(node.args[0].value,str)
           and node.args[0].value.startswith('Mode1 step timeline: driver_')]
    assert len(calls)==4
    for node in calls:
        assert isinstance(node.func,ast.Attribute)
        assert node.func.value.id=='_stats_logger'


def test_planned_policy_is_passed_to_metadata_not_just_environment():
    child=(ROOT/'run_mode1_local_length_sorted_e2e_adaptive_floor4.sh').read_text()
    wrapper=(ROOT/'tools/run_v014_sidecar_arm.py').read_text()
    assert 'shrink_aware.target_policy="$VLLM_ASCEND_SHRINK_AWARE_TARGET_POLICY"' in child
    assert 'actor_rollout_ref.rollout.shrink_aware.target_policy=planned' in wrapper


def test_taskrunner_explicitly_forwards_policy_and_sidecar(monkeypatch):
    from verl.trainer.constants_ppo import get_ppo_ray_runtime_env
    monkeypatch.setenv('VLLM_ASCEND_SHRINK_AWARE_TARGET_POLICY','planned')
    monkeypatch.setenv('VERL_SIDECAR_ENABLE','1')
    values=get_ppo_ray_runtime_env()['env_vars']
    assert values['VLLM_ASCEND_SHRINK_AWARE_TARGET_POLICY']=='planned'
    assert values['VERL_SIDECAR_ENABLE']=='1'


def test_sidecar_has_independent_tq_and_excludes_primary_graph_controls():
    source=(ROOT/'internal/run_elastic_sidecar_infer.sh').read_text()
    assert 'TASK_QUEUE_ENABLE="${VERL_SIDECAR_TASK_QUEUE_ENABLE:-2}"' in source
    assert 'ADAFLOOR_*|VLLM_ROLLOUT_*|VLLM_ASCEND_MODE1_*|VLLM_ASCEND_ELASTIC_*|VLLM_ASCEND_SHRINK_AWARE_*' in source


@pytest.mark.parametrize('old,new,message',[
    ('target_policy=planned','target_policy=natural','Planned survivors'),
    ('"full_iterations": 10','"full_iterations": 0','KV audit mismatch'),
    ('"max_tokens": 16384','"max_tokens": 4096','sampling caps'),
])
def test_native_verifier_rejects_false_graph_or_policy(tmp_path,old,new,message):
    spec=importlib.util.spec_from_file_location('native_sidecar_fixtures',ROOT/'tests/test_verify_qwen3_sidecar_pairs.py')
    fixture=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixture)
    arm=tmp_path/'seed_101/off'
    fixture._primary(arm,101,'off')
    log=next(arm.rglob('logs/*.txt'))
    text=log.read_text();assert old in text
    log.write_text(text.replace(old,new,1))
    with pytest.raises(fixture.MODULE.VerificationError,match=message):
        fixture.MODULE._primary_artifact(arm,101,'off')


def test_full_platform_banner_is_not_an_oom_but_actual_exception_is(tmp_path):
    spec=importlib.util.spec_from_file_location('native_sidecar_warning_fixture',ROOT/'tests/test_verify_qwen3_sidecar_pairs.py')
    fixture=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixture)
    arm=tmp_path/'seed_101/off';fixture._primary(arm,101,'off')
    log=next(arm.rglob('logs/*.txt'))
    banner='WARNING 10-23 00:00:00 [platform.py:558]             * (Out of Memory) errors or inference hangs. If you encounter such issues,\n'
    text=log.read_text();log.write_text(text+banner)
    fixture.MODULE._primary_artifact(arm,101,'off')
    log.write_text(text+banner+'torch.OutOfMemoryError: NPU out of memory\n')
    with pytest.raises(fixture.MODULE.VerificationError,match='contains an OOM'):
        fixture.MODULE._primary_artifact(arm,101,'off')


def test_all_sixteen_native_restore_completions_are_required(tmp_path):
    spec=importlib.util.spec_from_file_location('native_sidecar_restore_fixture',ROOT/'tests/test_verify_qwen3_sidecar_pairs.py')
    fixture=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixture)
    arm=tmp_path/'seed_101/off';fixture._primary(arm,101,'off')
    log=next(arm.rglob('logs/*.txt'));text=log.read_text()
    log.write_text(text.replace('restore done: rank=15 ','restore done: rank=14 '))
    with pytest.raises(fixture.MODULE.VerificationError,match='full-world restore'):
        fixture.MODULE._primary_artifact(arm,101,'off')
