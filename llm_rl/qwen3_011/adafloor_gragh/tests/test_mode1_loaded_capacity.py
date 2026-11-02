from types import SimpleNamespace

import pytest
import torch

from vllm_ascend.ops.fused_moe import AscendFusedMoE


def _fake_layer(rows: int, loaded_capacity: int) -> SimpleNamespace:
    calls = {"finalize": 0, "validate": [], "shrink": []}
    layer = SimpleNamespace(
        elastic_moe_mode="lossless",
        elastic_execution_mode=1,
        layer_idx=0,
        w13_weight=torch.empty((rows, 1)),
        w2_weight=torch.empty((rows, 1)),
        loaded_weight_capacity=loaded_capacity,
        _get_reserved_local_expert_slots_for_floor=lambda floor: 128 // floor,
        _finalize_mode1_full_world_compact_metadata=lambda: calls.__setitem__(
            "finalize", calls["finalize"] + 1
        ),
        _validate_mode1_lossless_active_mapping=lambda reason: calls[
            "validate"
        ].append(reason),
        shrink_lossless_loaded_weights_to_primary=lambda min_capacity: calls[
            "shrink"
        ].append(min_capacity)
        or True,
    )
    layer.calls = calls
    return layer


def test_reload_restores_capacity_metadata_when_rows_already_exist(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VLLM_ASCEND_MODE1_WEIGHT_RELOAD_TARGET_FLOOR", "2")
    layer = _fake_layer(rows=64, loaded_capacity=16)

    changed = AscendFusedMoE.enforce_mode1_target_floor_loaded_capacity_after_reload(
        layer, reason="test"
    )

    assert changed
    assert layer.loaded_weight_capacity == 64
    assert layer.calls["shrink"] == []
    assert layer.calls["finalize"] == 1
    assert layer.calls["validate"] == ["test:metadata_aligned"]


def test_reload_keeps_aligned_capacity_without_compaction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VLLM_ASCEND_MODE1_WEIGHT_RELOAD_TARGET_FLOOR", "4")
    layer = _fake_layer(rows=32, loaded_capacity=32)

    changed = AscendFusedMoE.enforce_mode1_target_floor_loaded_capacity_after_reload(
        layer, reason="test"
    )

    assert not changed
    assert layer.calls["shrink"] == []
    assert layer.calls["validate"] == ["test:metadata_aligned"]


def test_reload_compacts_rows_above_target(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VLLM_ASCEND_MODE1_WEIGHT_RELOAD_TARGET_FLOOR", "8")
    layer = _fake_layer(rows=64, loaded_capacity=32)

    changed = AscendFusedMoE.enforce_mode1_target_floor_loaded_capacity_after_reload(
        layer, reason="test"
    )

    assert changed
    assert layer.calls["shrink"] == [16]
    assert layer.calls["validate"] == ["test:capacity_enforced"]


def test_reload_rejects_target_larger_than_parameter_storage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VLLM_ASCEND_MODE1_WEIGHT_RELOAD_TARGET_FLOOR", "2")
    layer = _fake_layer(rows=16, loaded_capacity=16)

    with pytest.raises(RuntimeError, match="insufficient physical expert rows"):
        AscendFusedMoE.enforce_mode1_target_floor_loaded_capacity_after_reload(
            layer, reason="test"
        )
