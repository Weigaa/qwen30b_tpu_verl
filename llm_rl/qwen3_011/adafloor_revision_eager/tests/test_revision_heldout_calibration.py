import importlib.util
from pathlib import Path
import pytest

spec=importlib.util.spec_from_file_location('calibration',Path(__file__).parents[1]/'analysis_eval/analyze_revision_heldout_calibration.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def fixture():return [{str(i):[1000]*16 for i in range(32)},{str(i):[1200]*16 for i in range(32)},{str(i):[1400]*16 for i in range(32)}]


def test_current_labels_do_not_leak_into_fitted_ratio():
    histories=fixture();base=module.replay(histories,.95,3)
    histories[2]['0']=[16000]*16;changed=module.replay(histories,.95,3)
    assert base[1]['rho']==changed[1]['rho'] and base[1]['step_caps']==changed[1]['step_caps']
    assert base[0]['calibration_samples']==0 and base[0]['rho']==1.2


def test_quantile_margin_cannot_decrease_for_same_history():
    histories=fixture();histories[1]['0']=[3000]*16
    assert module.replay(histories,.90,1)[1]['rho']<=module.replay(histories,.99,1)[1]['rho']


def test_changed_prompt_set_is_rejected():
    histories=fixture();histories[1].pop('0')
    with pytest.raises(ValueError):module.replay(histories,.95,3)
