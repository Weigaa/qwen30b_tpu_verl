from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


TOOL = Path(__file__).parents[1] / "tools" / "summarize_deepseek_kv_probe.py"
SPEC = importlib.util.spec_from_file_location("deepseek_kv_probe_summary", TOOL)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _make_run(
    tmp_path: Path,
    floor: int,
    *,
    preempt: bool = False,
    waves: int = 1,
) -> tuple[Path, Path]:
    oracle = tmp_path / "epoch_001_mode1_natural" / "oracle"
    oracle.mkdir(parents=True)
    stage_sets = {
        4: [list(range(8, 16)), list(range(12, 16))],
        8: [list(range(8, 16))],
        16: [list(range(16))],
    }[floor]
    stages = {4: [8, 4], 8: [8], 16: [16]}[floor]
    (oracle / "length_sorted_rank_plan_summary.json").write_text(
        json.dumps(
            [
                {
                    "step": 1,
                    "feasible": True,
                    "selected_floor": floor,
                    "release_area": 0.0 if floor == 16 else 100.0,
                    "release_area_unit": "rank_token_proxy",
                    "predicted_step_exit": 64.0,
                    "schedule_thresholds": [] if floor == 16 else [17.0, 33.0][
                        : len(stages)
                    ],
                    "shrink_stages": stages,
                    "stage_survivor_ranks": stage_sets,
                    "tail_guard_response_cap": 64,
                }
            ]
        ),
        encoding="utf-8",
    )
    (oracle / "length_sorted_train.parquet").write_bytes(b"fixed-trigger")
    lines = ["GPU KV cache size: 999,936 tokens"]
    for wave in range(waves):
        for rank in range(16):
            pid = 100 + rank
            tokens = 620032 - rank * 128 - wave * 256
            lines.extend(
                [
                    f"(WorkerDict pid={pid}) rollout_worker_resize_start rank={rank} step=1 epoch=1 target_floor={floor} target_kv=1000000000",
                    f"(WorkerDict pid={pid}) Mode1 adaptive KV resize phase=start old_tokens=2048 target_tokens=1000000000 effective_target_tokens=1000000000 target_floor={floor}",
                    f"(WorkerDict pid={pid}) Mode1 adaptive KV resize memory: rank={rank} free_bytes=1 available_for_kv=1",
                    f"(WorkerDict pid={pid}) GPU KV cache size: {tokens:,} tokens",
                    f"(WorkerDict pid={pid}) Mode1 adaptive KV resize phase=plan_new_kv_done target_tokens=1000000000 effective_target_tokens=1000000000 new_tokens={tokens} new_blocks=1",
                    f"(WorkerDict pid={pid}) rollout_worker_resize_done rank={rank} step=1 epoch=1 target_floor={floor} target_kv=1000000000",
                ]
            )
    current = list(range(16))
    for target in stage_sets:
        if len(target) == 16:
            continue
        for rank in current:
            lines.append(
                f"(WorkerDict pid={100 + rank}) Shrink-aware staged trigger: "
                f"stage=test current_local={current} unfinished_local=[] "
                f"target_local={target} target_global={target}"
            )
        current = target
    if floor < 16:
        for rank in range(16):
            lines.append(
                f"(WorkerDict pid={100 + rank}) Elastic full-world restore segmented "
                f"timing: rank={rank} restore_seq=1 world_size=16"
            )
    lines.append("Unexpected exception logging compilation metrics TypeError")
    if preempt:
        lines.append("Preempting request 7 for request 8")
    lines.extend(["response/aborted_ratio:0.0", "rollout_output_time_s: 1.0"])
    log = tmp_path / "probe.log"
    log.write_text("\n".join(lines), encoding="utf-8")
    return tmp_path, log


def test_uses_target_resize_wave_not_cold_init_or_last_value(tmp_path: Path) -> None:
    run_root, log = _make_run(tmp_path, 4)
    report = MODULE.summarize(
        "natural_f4",
        4,
        16,
        run_root,
        log,
        runtime_profile="runtime-a",
        runtime_profile_sha256="a" * 64,
        execution_code_sha256="b" * 64,
        max_num_seqs=32,
        max_response_length=16384,
        tail_guard_min_cap=64,
        tail_guard_round_to=64,
        gpu_memory_utilization=0.9,
        enforce_eager="True",
        planning_history_sha256="c" * 64,
        planning_history_manifest_sha256="d" * 64,
        planning_trigger_subset_sha256="e" * 64,
    )
    assert report["observed_tokens"] == 618112
    assert report["per_rank_tokens"][0] == 620032
    assert report["per_rank_tokens"][15] == 618112
    assert report["runtime_profile"] == "runtime-a"
    assert report["runtime_profile_sha256"] == "a" * 64
    assert report["max_num_seqs"] == 32
    assert report["probe_tail_guard_min_cap"] == 64
    assert report["probe_tail_guard_round_to"] == 64
    assert report["actual_plan_response_cap"] == 64
    assert report["plan_release_area_unit"] == "rank_token_proxy"
    assert report["gpu_memory_utilization"] == 0.9
    assert report["enforce_eager"] == "True"


def test_rejects_preemption(tmp_path: Path) -> None:
    run_root, log = _make_run(tmp_path, 4, preempt=True)
    with pytest.raises(RuntimeError, match="preemption"):
        MODULE.summarize(
            "natural_f4", 4, 16, run_root, log,
            execution_code_sha256="b" * 64,
            planning_history_sha256="c" * 64,
            planning_history_manifest_sha256="d" * 64,
            planning_trigger_subset_sha256="e" * 64,
            max_response_length=16384,
            tail_guard_min_cap=64,
            tail_guard_round_to=64,
        )


def test_records_complete_floor4_lifecycle(tmp_path: Path) -> None:
    run_root, log = _make_run(tmp_path, 4)
    report = MODULE.summarize(
        "natural_f4", 4, 16, run_root, log,
        execution_code_sha256="b" * 64,
        planning_history_sha256="c" * 64,
        planning_history_manifest_sha256="d" * 64,
        planning_trigger_subset_sha256="e" * 64,
        max_response_length=16384,
        tail_guard_min_cap=64,
        tail_guard_round_to=64,
    )
    assert report["observed_tokens"] == 618112
    assert report["runtime_lifecycle"]["transitions"] == [
        {"from": 16, "to": 8, "survivor_ranks": list(range(8, 16))},
        {"from": 8, "to": 4, "survivor_ranks": list(range(12, 16))},
    ]


@pytest.mark.parametrize(
    "replacement",
    ["", "response/aborted_ratio:1e-3"],
)
def test_rejects_missing_or_nonzero_aborted_ratio(
    tmp_path: Path, replacement: str
) -> None:
    run_root, log = _make_run(tmp_path, 4)
    text = log.read_text(encoding="utf-8").replace(
        "response/aborted_ratio:0.0", replacement
    )
    log.write_text(text, encoding="utf-8")
    with pytest.raises(RuntimeError, match="missing or nonzero aborted response ratio"):
        MODULE.summarize(
            "natural_f4", 4, 16, run_root, log,
            execution_code_sha256="b" * 64,
            planning_history_sha256="c" * 64,
            planning_history_manifest_sha256="d" * 64,
            planning_trigger_subset_sha256="e" * 64,
            max_response_length=16384,
            tail_guard_min_cap=64,
            tail_guard_round_to=64,
        )


def test_rejects_additional_oom_spelling(tmp_path: Path) -> None:
    run_root, log = _make_run(tmp_path, 4)
    with log.open("a", encoding="utf-8") as handle:
        handle.write("\nACL_ERROR_RT_MEMORY_ALLOCATION\n")
    with pytest.raises(RuntimeError, match="NPU OOM"):
        MODULE.summarize(
            "natural_f4", 4, 16, run_root, log,
            execution_code_sha256="b" * 64,
            planning_history_sha256="c" * 64,
            planning_history_manifest_sha256="d" * 64,
            planning_trigger_subset_sha256="e" * 64,
            max_response_length=16384,
            tail_guard_min_cap=64,
            tail_guard_round_to=64,
        )


@pytest.mark.parametrize(("min_cap", "round_to"), [(0, 64), (64, 0), (-1, 64)])
def test_rejects_nonpositive_probe_tailguard_settings(
    tmp_path: Path, min_cap: int, round_to: int
) -> None:
    run_root, log = _make_run(tmp_path, 4)
    with pytest.raises(RuntimeError, match="must be positive"):
        MODULE.summarize(
            "natural_f4", 4, 16, run_root, log,
            execution_code_sha256="b" * 64,
            max_response_length=16384,
            tail_guard_min_cap=min_cap,
            tail_guard_round_to=round_to,
            planning_history_sha256="c" * 64,
            planning_history_manifest_sha256="d" * 64,
            planning_trigger_subset_sha256="e" * 64,
        )
