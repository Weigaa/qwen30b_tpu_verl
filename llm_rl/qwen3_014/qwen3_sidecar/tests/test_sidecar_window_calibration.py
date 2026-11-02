import importlib.util
from pathlib import Path
import pytest

ROOT=Path(__file__).parents[1]
spec=importlib.util.spec_from_file_location('calibrated_gate',ROOT/'tools/sidecar_window_gate.py')
gate=importlib.util.module_from_spec(spec);spec.loader.exec_module(gate)


def history(tmp_path):
    predictions=(4085,5724,8390,12360,16384)
    plans=[dict(step=i,rank_loads={str(rank):value for rank in range(16)})
           for i,value in enumerate(predictions,1)]
    for index,length in enumerate((4186,7932,11850),1):
        (tmp_path/f'length_{index}.txt').write_text((str(length)+'\n')*512)
    return plans


def test_cold_start_does_not_invent_calibration(tmp_path):
    plans=history(tmp_path)
    prediction,report=gate.calibrate_exit_from_prior_steps(plans,1,4085,tmp_path)
    assert prediction==4085 and report['sources']==[]


def test_step4_reads_only_completed_steps_and_respects_cap(tmp_path):
    plans=history(tmp_path)
    (tmp_path/'length_4.txt').write_text('must never read current responses')
    (tmp_path/'length_5.txt').write_text('must never read future responses')
    prediction,report=gate.calibrate_exit_from_prior_steps(plans,4,12360,tmp_path)
    assert prediction==16384
    assert [row['step'] for row in report['sources']]==[1,2,3]
    assert 1.40<report['ratio']<1.42
    assert report['current_or_future_step_lengths_read'] is False
    assert (prediction-11264)*0.0390625>=80
    assert (4085-3328)*0.03125<80


def test_incomplete_prior_step_fails_closed(tmp_path):
    plans=history(tmp_path)
    (tmp_path/'length_2.txt').write_text('100\n')
    with pytest.raises(ValueError,match='Incomplete'):
        gate.calibrate_exit_from_prior_steps(plans,4,12360,tmp_path)


def test_missing_prior_step_is_not_filled_from_future(tmp_path):
    plans=history(tmp_path)
    (tmp_path/'length_2.txt').unlink()
    with pytest.raises(FileNotFoundError):
        gate.calibrate_exit_from_prior_steps(plans,4,12360,tmp_path)


def test_short_tailguard_history_excludes_proven_first_padding_eos(tmp_path):
    plans=[dict(step=1,rank_loads={str(r):4000 for r in range(16)},
                tail_guard_enabled=True,tail_guard_response_cap=5120)]
    (tmp_path/'length_1.txt').write_text('5121\n'*512)
    prediction,report=gate.calibrate_exit_from_prior_steps(plans,2,6000,tmp_path,hard_cap=7168)
    assert prediction==7168
    assert report['sources'][0]['actual_max']==5120
    assert report['sources'][0]['synthetic_first_eos_excluded']==512
