from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path


TOOLS_DIR = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS_DIR))

from build_mode1_length_sorted_e2e_plan import _stats_from_lengths  # noqa: E402
from build_tailbatch_composability_plans import build_tailbatch_groups  # noqa: E402


def _stat(source_idx: int, load: int):
    return replace(
        _stats_from_lengths((float(load),)),
        source_idx=source_idx,
    )


def test_tailbatch_keeps_four_short_rounds_and_fifo_long_queue() -> None:
    stats = []
    for pool in range(4):
        pool_stats = [
            _stat(pool * 40 + offset, 40 - offset)
            for offset in range(40)
        ]
        stats.extend(reversed(pool_stats))

    groups, pools = build_tailbatch_groups(stats)

    assert len(groups) == 5
    assert all(len(group) == 32 for group in groups)
    for pool in range(4):
        expected_short = list(range(pool * 40 + 8, pool * 40 + 40))[::-1]
        expected_deferred = list(range(pool * 40, pool * 40 + 8))[::-1]
        assert [item.source_idx for item in groups[pool]] == expected_short
        assert pools[pool]["deferred_source_indices"] == expected_deferred
    assert [item.source_idx for item in groups[4]] == [
        source_idx
        for pool in range(4)
        for source_idx in list(range(pool * 40, pool * 40 + 8))[::-1]
    ]
    assert sorted(item.source_idx for group in groups for item in group) == list(
        range(160)
    )


def test_tailbatch_layout_is_independent_of_input_order() -> None:
    ordered = [_stat(source_idx, source_idx % 17 + 1) for source_idx in range(160)]
    first, _ = build_tailbatch_groups(ordered)
    second, _ = build_tailbatch_groups(list(reversed(ordered)))

    assert [
        [item.source_idx for item in group] for group in first
    ] == [
        [item.source_idx for item in group] for group in second
    ]
