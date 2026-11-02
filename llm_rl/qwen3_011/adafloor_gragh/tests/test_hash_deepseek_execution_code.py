from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


TOOL = Path(__file__).parents[1] / "tools" / "hash_deepseek_execution_code.py"
SPEC = importlib.util.spec_from_file_location("hash_deepseek_execution_code", TOOL)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

DIRECT_EXECUTION_DEPENDENCIES = (
    "run_common_epoch0_probe_gpu09_kv380800_permanent.sh",
    "run_baseline_vanilla_epoch1_2_from_common_epoch0.sh",
    "run_baseline_kvsafe_fixed4_epoch1_2.sh",
    "run_baseline_minskew_epoch1_2.sh",
    "run_mode1_dynamic_length_aware_adaptive_floor4_planned_full3.sh",
    "deepscaler.py",
    "tools/extract_vllm_kv_capacity.py",
    "tools/verify_deepseek_batch64_pair.py",
    "tools/build_deepseek_fixed_work_replay.py",
    "tools/verify_deepseek_fixed_work_pair.py",
    "tools/validate_deepseek_fixed_work_source.py",
    "tools/manage_deepseek_fixed_work_cleanup.py",
    "tools/migrate_deepseek_fixed_work_cap.py",
    "tools/build_mode1_length_sorted_e2e_plan.py",
    "tools/build_mode1_optimized_rank_plan.py",
    "tools/verify_deepseek_sidecar_run.py",
    "tools/hash_deepseek_checkpoint.py",
    "tools/summarize_mode1_comm_cache_log.py",
    "run_deepseek_v2_lite_batch64_fixed_work.sh",
)
ASSET_SOURCE_DEPENDENCIES = (
    "converter_hf_to_mcore.py",
    "prepare_deepseek_v2_lite_assets.sh",
)
DIRECT_DEPENDENCIES = DIRECT_EXECUTION_DEPENDENCIES + ASSET_SOURCE_DEPENDENCIES


def _tree(tmp_path: Path) -> Path:
    for relative in MODULE.TREE_ROOTS:
        path = tmp_path / relative
        path.mkdir(parents=True)
        (path / "source.py").write_text(f"value = {relative!r}\n", encoding="utf-8")
        (path / "ignored.pyc").write_bytes(b"ignored")
    for relative in MODULE.EXACT_FILES:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"# {relative}\n", encoding="utf-8")
    return tmp_path


def test_digest_is_stable_and_tracks_execution_source(tmp_path: Path) -> None:
    root = _tree(tmp_path)
    first, count = MODULE.digest(root)
    assert len(first) == 64
    assert count > 0
    assert MODULE.digest(root) == (first, count)
    target = root / "vllm_ascend" / "source.py"
    target.write_text("value = 'changed'\n", encoding="utf-8")
    second, second_count = MODULE.digest(root)
    assert second != first
    assert second_count == count


def test_digest_ignores_generated_binary_files(tmp_path: Path) -> None:
    root = _tree(tmp_path)
    first = MODULE.digest(root)
    (root / "vllm" / "cache.so").write_bytes(b"generated")
    (root / "verl" / "__pycache__").mkdir()
    (root / "verl" / "__pycache__" / "source.py").write_text(
        "generated = True\n", encoding="utf-8"
    )
    assert MODULE.digest(root) == first


def test_digest_ignores_runtime_script_snapshots(tmp_path: Path) -> None:
    root = _tree(tmp_path)
    first = MODULE.digest(root)
    snapshot = root / "internal" / "launcher.sh.run-snapshot.ABC123"
    snapshot.write_text("#!/usr/bin/env bash\nexit 99\n", encoding="utf-8")
    assert MODULE.digest(root) == first


def test_batch64_direct_dependencies_are_in_the_combined_hash() -> None:
    expected_execution = set(DIRECT_EXECUTION_DEPENDENCIES)
    expected_assets = set(ASSET_SOURCE_DEPENDENCIES)

    assert expected_execution <= set(MODULE.EXECUTION_EXACT_FILES)
    assert expected_assets <= set(MODULE.ASSET_SOURCE_FILES)
    assert expected_execution | expected_assets <= set(MODULE.EXACT_FILES)


@pytest.mark.parametrize(
    "relative",
    DIRECT_DEPENDENCIES,
)
def test_digest_tracks_direct_dependency_content(
    tmp_path: Path, relative: str
) -> None:
    root = _tree(tmp_path)
    first, count = MODULE.digest(root)
    target = root / relative
    original = target.read_text(encoding="utf-8")

    target.write_text(original + "# changed\n", encoding="utf-8")

    second, second_count = MODULE.digest(root)
    assert second != first
    assert second_count == count


@pytest.mark.parametrize(
    "relative",
    DIRECT_DEPENDENCIES,
)
def test_missing_direct_dependency_fails_closed(
    tmp_path: Path, relative: str
) -> None:
    root = _tree(tmp_path)
    (root / relative).unlink()

    with pytest.raises(ValueError, match="missing execution source file"):
        MODULE.digest(root)
