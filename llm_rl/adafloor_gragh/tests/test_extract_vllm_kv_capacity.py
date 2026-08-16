from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


TOOL = Path(__file__).parents[1] / "tools" / "extract_vllm_kv_capacity.py"
SPEC = importlib.util.spec_from_file_location("extract_vllm_kv_capacity", TOOL)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _log(tmp_path: Path, capacities: list[int]) -> Path:
    path = tmp_path / "run.log"
    path.write_text(
        "\n".join(
            f"(WorkerDict pid={100 + rank}) GPU KV cache size: "
            f"{capacity:,} tokens"
            for rank, capacity in enumerate(capacities)
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def test_extracts_minimum_across_all_ranks(tmp_path: Path) -> None:
    capacities = [614144] * 16
    capacities[7] = 613888
    assert MODULE.extract(_log(tmp_path, capacities), 16, 128) == 613888


def test_rejects_missing_rank_capacity(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="expected 16.*found 15"):
        MODULE.extract(_log(tmp_path, [614144] * 15), 16, 128)


@pytest.mark.parametrize("capacity", [0, 614145])
def test_rejects_invalid_capacity(tmp_path: Path, capacity: int) -> None:
    capacities = [614144] * 16
    capacities[3] = capacity
    with pytest.raises(ValueError, match="invalid KV capacities"):
        MODULE.extract(_log(tmp_path, capacities), 16, 128)
