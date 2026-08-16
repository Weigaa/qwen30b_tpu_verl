from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

import pytest


TOOLS_DIR = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS_DIR))

from build_mode1_length_sorted_e2e_plan import (  # noqa: E402
    PAIR_ADJUSTED_PEAK,
    PAIR_LOAD_GAP,
    PAIR_RAW_PEAK,
    PAIR_SOURCE_INDICES,
    RELEASE_AREA_UNIT,
    _adjacent_prompt_bundles,
    _pair_metrics,
    _peak_active_tokens,
    _solve_one_batch,
    _stats_from_lengths,
    _write_outputs,
)


def _prompt(source_idx: int, load: float):
    lengths = tuple(
        float(max(1, int(load) + (sample_idx % 3) - 1))
        for sample_idx in range(16)
    )
    return replace(
        _stats_from_lengths(lengths, predicted_tail=float(load)),
        source_idx=int(source_idx),
    )


@pytest.fixture(scope="module")
def batch64():
    loads = [1.0] * 32 + [2.0] * 16 + [3.0] * 8 + [4.0] * 8
    return [_prompt(source_idx, load) for source_idx, load in enumerate(loads)]


@pytest.fixture(scope="module")
def forced_floor_plans(batch64):
    caps = {2: 10000.0, 4: 10000.0, 8: 10000.0, 16: 10000.0}
    return {
        floor: _solve_one_batch(
            batch64,
            max_rank_peak_tokens=10000.0,
            floor_kv_caps=caps,
            adaptive_floor=True,
            min_adaptive_floor=2,
            force_selected_floor=floor,
            active_peak_safety_factor=1.25,
            max_response_len=8.0,
        )
        for floor in (2, 4, 8, 16)
    }


def test_batch64_builds_fixed_adjacent_two_prompt_bundles() -> None:
    prompts = [
        _prompt(source_idx, float(64 - source_idx))
        for source_idx in range(64)
    ]

    bundles = _adjacent_prompt_bundles(prompts)

    expected_order = sorted(
        prompts, key=lambda item: (item.load, item.source_idx))
    expected_pairs = [
        (
            expected_order[offset].source_idx,
            expected_order[offset + 1].source_idx,
        )
        for offset in range(0, 64, 2)
    ]
    assert len(bundles) == 32
    assert [bundle.source_indices for bundle in bundles] == expected_pairs
    assert all(len(bundle.lengths) == 32 for bundle in bundles)


def test_batch64_edge_uses_all_four_prompts_for_exact_kv_peak() -> None:
    prompts = [_prompt(source_idx, float(source_idx + 1)) for source_idx in range(64)]
    by_source = {item.source_idx: item for item in prompts}
    bundles = _adjacent_prompt_bundles(prompts)

    metrics = _pair_metrics(
        bundles,
        active_peak_safety_factor=1.5,
        max_response_len=20.0,
    )
    edge = metrics[(0, 1)]
    sources = list(bundles[0].source_indices + bundles[1].source_indices)
    raw_lengths = tuple(
        length
        for source_idx in sources
        for length in by_source[source_idx].lengths
    )
    adjusted_lengths = tuple(min(length * 1.5, 20.0) for length in raw_lengths)
    source_loads = [by_source[source_idx].load for source_idx in sources]

    assert len(metrics) == 496
    assert len(raw_lengths) == 64
    assert edge[PAIR_SOURCE_INDICES] == sources
    assert edge[PAIR_RAW_PEAK] == pytest.approx(
        _peak_active_tokens(raw_lengths))
    assert edge[PAIR_ADJUSTED_PEAK] == pytest.approx(
        _peak_active_tokens(adjusted_lengths))
    assert edge[PAIR_LOAD_GAP] == pytest.approx(
        max(source_loads) - min(source_loads))


def test_batch64_forced_floors_preserve_coverage_and_quotas(
    batch64,
    forced_floor_plans,
) -> None:
    expected_sources = {item.source_idx for item in batch64}
    bundles = _adjacent_prompt_bundles(batch64)
    fixed_bundles = {bundle.source_indices for bundle in bundles}
    by_source = {item.source_idx: item for item in batch64}

    for floor, plan in forced_floor_plans.items():
        assert plan.selected_floor == floor
        assert plan.feasible
        assert plan.rank_grouping_search_space == (
            "pairings_of_32_fixed_adjacent_two_prompt_bundles")

        assigned = []
        for rank in range(16):
            rank_sources = plan.rank_to_prompt_indices[rank]
            assert len(rank_sources) == 4
            assert tuple(rank_sources[:2]) in fixed_bundles
            assert tuple(rank_sources[2:]) in fixed_bundles
            assigned.extend(rank_sources)

            raw_lengths = tuple(
                length
                for source_idx in rank_sources
                for length in by_source[source_idx].lengths
            )
            adjusted_lengths = tuple(
                min(length * 1.25, 8.0) for length in raw_lengths)
            assert len(raw_lengths) == 64
            assert plan.rank_peak_loads[rank] == pytest.approx(
                _peak_active_tokens(raw_lengths))
            assert plan.rank_adjusted_peak_loads[rank] == pytest.approx(
                _peak_active_tokens(adjusted_lengths))

        assert len(assigned) == 64
        assert len(set(assigned)) == 64
        assert set(assigned) == expected_sources

        for threshold, quota in plan.schedule_quotas:
            completed = sum(
                load <= threshold for load in plan.rank_loads.values())
            assert completed >= quota
        if floor == 16:
            assert plan.schedule_quotas == ()
            assert plan.release_area == 0.0
        else:
            assert plan.schedule_quotas
            assert plan.schedule_thresholds[-1] < max(
                plan.rank_loads.values())
            assert plan.release_area > 0.0


def test_batch64_writes_four_prompt_rank_plan_and_search_space(
    tmp_path: Path,
    batch64,
    forced_floor_plans,
) -> None:
    pandas = pytest.importorskip("pandas")
    full_df = pandas.DataFrame({
        "source": list(range(64)),
        "prompt": [f"prompt-{index}" for index in range(64)],
        "extra_info": [
            {"answer": f"answer-{index}"} for index in range(64)
        ],
    })
    output_train = tmp_path / "length_sorted_train.parquet"
    output_plan = tmp_path / "length_sorted_rank_plan.json"
    output_summary = tmp_path / "length_sorted_rank_plan_summary.json"
    output_oracle = tmp_path / "length_sorted_length_oracle.json"

    write_kwargs = dict(
        max_rank_peak_tokens=10000.0,
        active_peak_safety_factor=1.25,
        max_response_len=8.0,
        baseline_dirs=[tmp_path / "history"],
        length_ema_decay=0.3,
        tail_guard_ratio=1.0,
        tail_guard_ratio_quantile=0.95,
        tail_guard_ratio_window=3,
        tail_guard_sample_count=64,
        tail_guard_min_cap=1,
        tail_guard_round_to=1,
        rank_matching_policy="release_area",
        kv_safe_fixed_floor=None,
    )
    _write_outputs(
        full_df,
        [forced_floor_plans[2]],
        output_train,
        output_plan,
        output_summary,
        output_oracle,
        **write_kwargs,
    )

    plan = json.loads(output_plan.read_text(encoding="utf-8"))[0]
    summary = json.loads(output_summary.read_text(encoding="utf-8"))[0]
    rank_map = plan["rank_to_dataset_item_idx"]
    source_map = plan["rank_to_source_idx"]
    flattened_dataset_ids = [
        item for rank in map(str, range(16)) for item in rank_map[rank]
    ]
    flattened_source_ids = [
        item for rank in map(str, range(16)) for item in source_map[rank]
    ]

    assert all(len(rank_map[str(rank)]) == 4 for rank in range(16))
    assert all(len(source_map[str(rank)]) == 4 for rank in range(16))
    assert flattened_dataset_ids == list(range(64))
    assert len(set(flattened_source_ids)) == 64
    assert set(flattened_source_ids) == set(range(64))
    written = pandas.read_parquet(output_train)
    assert written["source"].tolist() == flattened_source_ids
    assert [item["index"] for item in written["extra_info"]] == (
        flattened_source_ids
    )
    assert [item["answer"] for item in written["extra_info"]] == [
        f"answer-{source_idx}" for source_idx in flattened_source_ids
    ]
    assert len(json.loads(output_oracle.read_text(encoding="utf-8"))) == 64
    assert plan["prompts_per_rank"] == 4
    assert summary["prompts_per_rank"] == 4
    assert plan["rank_grouping_search_space"] == (
        "pairings_of_32_fixed_adjacent_two_prompt_bundles")
    assert summary["rank_grouping_search_space"] == (
        "pairings_of_32_fixed_adjacent_two_prompt_bundles")
    assert plan["release_area_unit"] == RELEASE_AREA_UNIT
    assert summary["release_area_unit"] == RELEASE_AREA_UNIT
    assert "limited to rank_grouping_search_space" in summary[
        "rank_matching_objective"]

    conflicting_df = full_df.copy(deep=True)
    conflicting_extra_info = list(conflicting_df["extra_info"])
    conflicting_extra_info[7] = {
        **conflicting_extra_info[7],
        "index": 0,
    }
    conflicting_df["extra_info"] = conflicting_extra_info
    with pytest.raises(ValueError, match="conflicting identity"):
        _write_outputs(
            conflicting_df,
            [forced_floor_plans[2]],
            tmp_path / "conflicting.parquet",
            tmp_path / "conflicting_plan.json",
            tmp_path / "conflicting_summary.json",
            tmp_path / "conflicting_oracle.json",
            **write_kwargs,
        )
