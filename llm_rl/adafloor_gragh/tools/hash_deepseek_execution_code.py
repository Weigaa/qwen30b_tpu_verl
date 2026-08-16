#!/usr/bin/env python3
"""Hash the source files that define DeepSeek AdaFloor runs and assets."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


SOURCE_SUFFIXES = {
    ".c",
    ".cc",
    ".cpp",
    ".h",
    ".hpp",
    ".json",
    ".py",
    ".sh",
    ".toml",
    ".yaml",
    ".yml",
}
TREE_ROOTS = (
    "verl",
    "vllm",
    "vllm_ascend",
    "megatron",
    "mindspeed",
    "r1_ascend",
    "internal",
)
EXECUTION_EXACT_FILES = (
    # Common epoch0 and the paired baseline launch chain live at repository
    # root, outside the recursively hashed implementation trees.
    "run_common_epoch0_probe_gpu09_kv380800_permanent.sh",
    "run_baseline_vanilla_epoch1_2_from_common_epoch0.sh",
    "run_baseline_kvsafe_fixed4_epoch1_2.sh",
    "run_baseline_minskew_epoch1_2.sh",
    "run_mode1_dynamic_length_aware_adaptive_floor4_planned_full3.sh",
    # Reward and post-run validation directly affect accepted measurements.
    "deepscaler.py",
    "tools/extract_vllm_kv_capacity.py",
    "tools/verify_deepseek_batch64_pair.py",
    "tools/build_deepseek_fixed_work_replay.py",
    "tools/verify_deepseek_fixed_work_pair.py",
    "tools/validate_deepseek_fixed_work_source.py",
    "tools/manage_deepseek_fixed_work_cleanup.py",
    "tools/migrate_deepseek_fixed_work_cap.py",
    "tools/verify_deepseek_sidecar_run.py",
    "run_deepseek_v2_lite_kv_probe.sh",
    "run_deepseek_v2_lite_natural_f4_calibration.sh",
    "run_deepseek_v2_lite_natural_f2_calibration.sh",
    "run_deepseek_v2_lite_kv_cap_validation.sh",
    "run_deepseek_v2_lite_natural_f2_kv_cap_validation.sh",
    "run_deepseek_v2_lite_fair_compare.sh",
    "run_deepseek_v2_lite_batch64_fixed_work.sh",
    "run_baseline_lengthsort_epoch1_2.sh",
    "run_mode0_no_shrink_baseline.sh",
    "run_mode1_dynamic_length_aware_adaptive_floor4_epochs.sh",
    "run_mode1_dynamic_length_aware_adaptive_floor4_natural_full3.sh",
    "run_mode1_dynamic_length_aware_adaptive_floor2_natural_tailguard_reuse_epoch0_2epoch.sh",
    "run_mode1_dynamic_length_aware_adaptive_floor2_natural_noguard_reuse_epoch0_2epoch.sh",
    "run_mode1_local_length_sorted_e2e_adaptive_floor4.sh",
    "run_mode1_local_length_sorted_e2e_adaptive_floor2.sh",
    "run_paper_fair_epoch1_2_from_common_epoch0.sh",
    "tools/build_offline_planning_history.py",
    "tools/build_mode1_length_sorted_e2e_plan.py",
    "tools/build_mode1_optimized_rank_plan.py",
    "tools/build_mode1_fast_step_subset.py",
    "tools/promote_kv_admission_caps.py",
    "tools/generate_deepseek_kv_caps.py",
    "tools/summarize_deepseek_kv_probe.py",
    "tools/summarize_mode1_comm_cache_log.py",
    "tools/verify_deepseek_kv_cap_run.py",
    "tools/verify_deepseek_n_f2_kv_cap_run.py",
    "tools/hash_deepseek_execution_code.py",
    "tools/hash_deepseek_checkpoint.py",
    "tools/hash_deepseek_runtime_profile.py",
    "tools/audit_deepseek_n_f4_formal_run.py",
    "tools/audit_deepseek_common_epoch0.py",
    "tools/smoke_deepseek_v2_lite_vllm.py",
    "tools/validate_deepseek_v2_lite_assets.py",
    "tools/prepare_deepseek_kv_probe_trigger.py",
    "run_deepseek_v2_lite_threshold_adafloor_two_step_tq1_smoke.sh",
    "run_deepseek_v2_lite_threshold_actor_two_step_tq1_smoke.sh",
)

# Asset preparation is outside the online execution path, but a conversion
# source change must invalidate the combined digest used to name experiments.
ASSET_SOURCE_FILES = (
    "converter_hf_to_mcore.py",
    "prepare_deepseek_v2_lite_assets.sh",
)

EXACT_FILES = EXECUTION_EXACT_FILES + ASSET_SOURCE_FILES

EXACT_GLOBS = ("run_deepseek_v2_lite_*.sh",)


def source_files(root: Path) -> list[Path]:
    files: set[Path] = set()
    for relative in TREE_ROOTS:
        tree = root / relative
        if not tree.is_dir():
            raise ValueError(f"missing execution source tree: {tree}")
        for path in tree.rglob("*"):
            if (
                path.is_file()
                and path.suffix in SOURCE_SUFFIXES
                and "__pycache__" not in path.parts
                and ".run-snapshot." not in path.name
            ):
                files.add(path)
    for relative in EXACT_FILES:
        path = root / relative
        if not path.is_file():
            raise ValueError(f"missing execution source file: {path}")
        files.add(path)
    for pattern in EXACT_GLOBS:
        matches = sorted(root.glob(pattern))
        if not matches:
            raise ValueError(f"execution source glob matched no files: {pattern}")
        files.update(path for path in matches if path.is_file())
    return sorted(files, key=lambda path: path.relative_to(root).as_posix())


def digest(root: Path) -> tuple[str, int]:
    root = root.resolve()
    hasher = hashlib.sha256()
    files = source_files(root)
    for path in files:
        relative = path.relative_to(root).as_posix().encode("utf-8")
        content = path.read_bytes()
        hasher.update(len(relative).to_bytes(8, "big"))
        hasher.update(relative)
        hasher.update(len(content).to_bytes(8, "big"))
        hasher.update(content)
    return hasher.hexdigest(), len(files)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--show-count", action="store_true")
    args = parser.parse_args()
    value, count = digest(args.root)
    if args.show_count:
        print(f"{value} {count}")
    else:
        print(value)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
