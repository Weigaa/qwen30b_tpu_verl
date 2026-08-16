from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import pytest


TOOLS_DIR = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS_DIR))

from build_mode1_length_sorted_e2e_plan import (  # noqa: E402
    _read_ema_baseline_stats,
    _solve_one_batch,
    _stats_from_lengths,
    _tail_guard_underestimate_ratio,
)
from build_mode1_optimized_rank_plan import _read_offline_planning_history
from build_mode1_optimized_rank_plan import _map_stats_to_dataset


def test_prompt_tail_ema_uses_epoch_max_and_latest_kv_lengths() -> None:
    epoch0 = {"prompt": _stats_from_lengths((100.0, 1.0))}
    epoch1 = {"prompt": _stats_from_lengths((60.0, 90.0))}

    predicted = _read_ema_baseline_stats([epoch0, epoch1], ema_decay=0.3)

    stat = predicted["prompt"]
    assert stat.load == pytest.approx(93.0)
    assert stat.predicted_tail == pytest.approx(93.0)
    assert stat.max_len == pytest.approx(90.0)
    assert stat.lengths == (60.0, 90.0)


def test_tail_guard_calibrates_against_prompt_tail_ema() -> None:
    history = [
        {"prompt": _stats_from_lengths((100.0, 1.0))},
        {"prompt": _stats_from_lengths((60.0, 90.0))},
        {"prompt": _stats_from_lengths((186.0, 50.0))},
    ]

    ratio, sample_count = _tail_guard_underestimate_ratio(
        history,
        ema_decay=0.3,
        quantile=1.0,
        default_ratio=1.0,
        window=2,
    )

    assert ratio == pytest.approx(2.0)
    assert sample_count == 2


@pytest.mark.parametrize(
    "floor4_area,floor8_area,expected_floor",
    [
        (80.0, 100.0, 8),
        (100.0, 100.0, 8),
    ],
)
def test_adaptive_floor_maximizes_release_area_across_floors(
    monkeypatch: pytest.MonkeyPatch,
    floor4_area: float,
    floor8_area: float,
    expected_floor: int,
) -> None:
    batch = [
        replace(
            _stats_from_lengths((float(index + 1),)),
            source_idx=index,
        )
        for index in range(32)
    ]
    probe_pairs = [
        (
            1.0,
            1.0,
            1.0,
            float(rank + 1),
            0.0,
            [2 * rank, 2 * rank + 1],
        )
        for rank in range(16)
    ]

    monkeypatch.setattr(
        "build_mode1_length_sorted_e2e_plan._capacity_constrained_load_gap_matching",
        lambda *args, **kwargs: (probe_pairs, True),
    )

    release_by_floor = {
        4: (floor4_area, (8.0, 12.0)),
        8: (floor8_area, (7.0,)),
        16: (0.0, ()),
    }

    def solve_floor(*args, selected_floor: int, **kwargs):
        area, thresholds = release_by_floor[selected_floor]
        quotas = tuple(
            (threshold, quota)
            for threshold, quota in zip(
                thresholds,
                {4: (8, 12), 8: (8,), 16: ()}[selected_floor],
                strict=True,
            )
        )
        return (
            probe_pairs,
            True,
            thresholds,
            quotas,
            area,
            f"floor{selected_floor}",
        )

    monkeypatch.setattr(
        "build_mode1_length_sorted_e2e_plan._release_area_matching_for_floor",
        solve_floor,
    )

    plan = _solve_one_batch(
        batch,
        max_rank_peak_tokens=1000.0,
        floor_kv_caps={4: 1000.0, 8: 1000.0, 16: 1000.0},
        adaptive_floor=True,
        min_adaptive_floor=4,
        active_peak_safety_factor=1.0,
        max_response_len=16384.0,
    )

    assert plan.selected_floor == expected_floor
    assert plan.release_area == pytest.approx(max(floor4_area, floor8_area))


def test_singleton_batch_assigns_one_prompt_per_rank() -> None:
    batch = [
        replace(
            _stats_from_lengths((float(index + 1), float(index + 2))),
            source_idx=index,
        )
        for index in range(16)
    ]

    plan = _solve_one_batch(
        batch,
        max_rank_peak_tokens=1000.0,
        floor_kv_caps={2: 1000.0, 4: 1000.0, 8: 1000.0, 16: 1000.0},
        adaptive_floor=True,
        min_adaptive_floor=2,
        active_peak_safety_factor=1.0,
        max_response_len=16384.0,
    )

    assert sorted(
        source_idx
        for indices in plan.rank_to_prompt_indices.values()
        for source_idx in indices
    ) == list(range(16))
    assert all(len(indices) == 1 for indices in plan.rank_to_prompt_indices.values())
    assert plan.feasible


def test_compact_history_reuses_prompt_set_across_step_counts_and_rollout_n(
    tmp_path: Path,
) -> None:
    history_file = tmp_path / "offline_planning_history.json"
    history_file.write_text(
        '{"schema_version":1,"steps":5,"responses_per_prompt":16,'
        '"records":[{"input":"prompt","lengths":['
        + ",".join(str(value) for value in range(1, 17))
        + "]}]}",
        encoding="utf-8",
    )

    stats = _read_offline_planning_history(
        history_file,
        steps=10,
        responses_per_prompt=8,
    )

    assert stats["prompt"].lengths == tuple(float(value) for value in range(1, 9))


def test_transition_stress_can_repeat_the_historical_prompt_set(
    tmp_path: Path,
) -> None:
    pandas = pytest.importorskip("pandas")
    train_file = tmp_path / "train.parquet"
    pandas.DataFrame({
        "prompt": [
            [{"role": "user", "content": "first"}],
            [{"role": "user", "content": "second"}],
        ]
    }).to_parquet(train_file, index=False)
    stats_by_input = {
        "user\nfirst\nassistant\n": _stats_from_lengths((1.0, 2.0)),
        "user\nsecond\nassistant\n": _stats_from_lengths((3.0, 4.0)),
    }

    repeated_df, repeated_stats = _map_stats_to_dataset(
        train_file,
        stats_by_input,
        dataset_fraction=1.0,
        max_samples=-1,
        min_samples=6,
        repeat_prompt_set_to_fill=True,
    )

    assert len(repeated_df) == 6
    assert [item.source_idx for item in repeated_stats] == list(range(6))
    assert [item.max_len for item in repeated_stats] == [2.0, 4.0] * 3


def test_dataset_mapping_ignores_duplicate_after_executed_prefix(
    tmp_path: Path,
) -> None:
    pandas = pytest.importorskip("pandas")
    train_file = tmp_path / "train.parquet"
    pandas.DataFrame({
        "prompt": [
            [{"role": "user", "content": "repeated"}],
            [{"role": "user", "content": "middle"}],
            [{"role": "user", "content": "repeated"}],
            [{"role": "user", "content": "outside"}],
            [{"role": "user", "content": "repeated"}],
        ]
    }).to_parquet(train_file, index=False)
    stats_by_input = {
        "user\nrepeated\nassistant\n": _stats_from_lengths((1.0, 2.0)),
        "user\nmiddle\nassistant\n": _stats_from_lengths((3.0, 4.0)),
    }

    full_df, mapped = _map_stats_to_dataset(
        train_file,
        stats_by_input,
        dataset_fraction=1.0,
        max_samples=-1,
        min_samples=3,
    )

    assert len(full_df) == 5
    assert [item.source_idx for item in mapped] == [0, 1, 2]
    assert [item.max_len for item in mapped] == [2.0, 4.0, 2.0]


def test_dataset_mapping_uses_model_chat_template(
    tmp_path: Path,
) -> None:
    pandas = pytest.importorskip("pandas")
    train_file = tmp_path / "train.parquet"
    pandas.DataFrame({
        "prompt": [[{"role": "user", "content": "first"}]],
    }).to_parquet(train_file, index=False)

    class DeepSeekLikeTokenizer:

        def apply_chat_template(
            self, prompt, *, add_generation_prompt: bool, tokenize: bool
        ):
            assert add_generation_prompt
            assert tokenize
            return [prompt[0]["content"]]

        def decode(self, token_ids, *, skip_special_tokens: bool) -> str:
            assert skip_special_tokens
            return f"User: {token_ids[0]}\n\nAssistant:"

    stats_by_input = {
        "User: first\n\nAssistant:": _stats_from_lengths((5.0, 7.0)),
    }
    _, mapped = _map_stats_to_dataset(
        train_file,
        stats_by_input,
        dataset_fraction=1.0,
        max_samples=-1,
        min_samples=1,
        prompt_tokenizer=DeepSeekLikeTokenizer(),
    )

    assert len(mapped) == 1
    assert mapped[0].max_len == 7.0
