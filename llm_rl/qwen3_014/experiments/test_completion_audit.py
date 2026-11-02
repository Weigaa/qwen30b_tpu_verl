import pytest
from audit_supplement_completion import prerequisites,same


def test_no_missing_results_cannot_be_claimed_complete(tmp_path):
    assert len(prerequisites(tmp_path))==5
    (tmp_path/'code_task').mkdir()
    (tmp_path/'code_task/CODE_PAIR_RESULT.json').write_text('{}')
    assert len(prerequisites(tmp_path))==4


def test_reconstructed_values_fail_closed_on_nonfinite_or_changed_metrics():
    same(10.0,10.0,'unchanged')
    for value in (float('nan'),float('inf'),11.0):
        with pytest.raises(ValueError):same(value,10.0,'changed')
