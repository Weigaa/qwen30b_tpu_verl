import importlib.util
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "analysis_eval"))

spec = importlib.util.spec_from_file_location("frozen_final", Path(__file__).parents[1]
                                            / "analysis_eval/finalize_hpca_planned_frozen.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_paired_interval_is_three_seed_t_not_unpaired():
    rows = [{"trial_seed": s, "policy": p, "rollout_s": 100 + (i if p == "planned_release" else 0)}
            for i, s in enumerate((101, 202, 303), start=1)
            for p in ("planned_release", "planned_minskew")]
    actual = module.paired(rows, "rollout_s")
    assert actual["paired_mean"] == 2
    assert actual["sample_sd"] == 1
    assert actual["ci95"][0] == pytest.approx(2-4.30265273/(3**0.5))
    with pytest.raises(ValueError):
        module.paired(rows[:-1], "rollout_s")
    with pytest.raises(ValueError):
        module.paired(rows + [rows[0]], "rollout_s")
