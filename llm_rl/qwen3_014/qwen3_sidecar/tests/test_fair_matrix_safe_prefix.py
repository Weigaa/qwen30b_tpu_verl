import pytest

from tools.verify_qwen3_v014_fair_one_step_matrix import (
    verify_restore_boundary, verify_shrink_transitions)


def events(stages, policy="natural", graph=True):
    lines = []
    current = list(range(16))
    for target in stages:
        lines += [f"AdaFloor staged trigger: current={current} unfinished={target} "
                  f"target={target} policy={policy}"] * len(current)
        lines += [f"Elastic parallel shrink done: active_ranks={target} "
                  f"dp_size={len(target)} ep_size={len(target)}"] * len(target)
        if graph:
            lines += [f"Elastic ACLGraph topology capture finished: active_ranks={target}"] * len(target)
        current = target
    return "\n".join(lines)


@pytest.mark.parametrize("stages", [[], [list(range(8,16))],
    [list(range(8,16)), list(range(12,16))],
    [list(range(8,16)), list(range(12,16)), [14,15]]])
def test_natural_completed_prefix_is_normal(stages):
    assert verify_shrink_transitions(events(stages), "natural", "ada_n2", True) == stages


def test_planned_prefix_is_normal():
    stages = [list(range(8,16)), list(range(12,16))]
    assert verify_shrink_transitions(events(stages[:1], "planned"), "planned", "ada_p_f4",
                                     True, stages) == stages[:1]


def test_started_transition_still_requires_completion():
    text = events([list(range(8,16))])
    text = text.replace("Elastic parallel shrink done:", "missing:", 1)
    with pytest.raises(RuntimeError, match="shrink completions"):
        verify_shrink_transitions(text, "broken", "ada_n2", True)


def test_skipping_a_stage_is_not_a_safe_prefix():
    with pytest.raises(RuntimeError, match="invalid target"):
        verify_shrink_transitions(events([list(range(12,16))]), "skipped", "ada_n2", True)


def test_no_shrink_needs_unchanged_state_not_restore_rpc():
    text = "\n".join(
        f"Elastic parallel restore requested after rollout: rank={rank} "
        "scaled_once=False active_ranks=None" for rank in range(16))
    verify_restore_boundary(text, "no-shrink", shrank=False)
    with pytest.raises(RuntimeError, match="restore boundary"):
        verify_restore_boundary(text, "missing-restore", shrank=True)
    with pytest.raises(RuntimeError, match="restore boundary"):
        verify_restore_boundary(text.replace("rank=15", "rank=14"),
                                "missing-rank", shrank=False)


def test_no_shrink_with_no_boundary_evidence_fails():
    with pytest.raises(RuntimeError, match="restore boundary"):
        verify_restore_boundary("", "missing-boundary", shrank=False)
