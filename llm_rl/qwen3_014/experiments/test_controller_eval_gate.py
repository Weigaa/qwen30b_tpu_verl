import importlib.util
from pathlib import Path
import pytest


def test_directory_gate_prevents_any_launch_before_idle_or_NPU(monkeypatch,tmp_path):
    file=Path(__file__).parent/'run_eager_controls_when_ready.py'
    spec=importlib.util.spec_from_file_location('eval_gate_test_controller',file)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    label='HELDOUT_RUNNING_vanilla';(tmp_path/(label+'_LAUNCH.json')).mkdir()
    def must_not_call(*a,**k):raise AssertionError('A gated evaluation must not launch or touch NPU')
    monkeypatch.setattr(module.subprocess,'run',must_not_call);monkeypatch.setattr(module,'idle',must_not_call)
    with pytest.raises(RuntimeError,match='Refusing existing launch evidence'):
        module.launch(['unused'],{},tmp_path,tmp_path,label)


def test_post_validation_recovery_refuses_other_failures(tmp_path):
    import json
    from run_eager_controls_when_ready import recover_eager_validation_only
    with pytest.raises(RuntimeError,match='no automatic retry'):
        recover_eager_validation_only(None,tmp_path,'vanilla','unused')
    (tmp_path/'EAGER_TRAINING_planned_f4_RESULT.json').write_text(json.dumps({'exit_code':1}))
    (tmp_path/'EAGER_TRAINING_planned_f4.log').write_text('runtime failed')
    with pytest.raises(RuntimeError,match='bounded known eager'):
        recover_eager_validation_only(None,tmp_path,'planned_f4','unused')
