from __future__ import annotations

import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest


ROOT = Path(__file__).parents[1]
TOOL = ROOT / "tools" / "verify_deepseek_sidecar_run.py"
SPEC = importlib.util.spec_from_file_location("verify_deepseek_sidecar_run", TOOL)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)
LAUNCHER = ROOT / "run_deepseek_v2_lite_sidecar_smoke.sh"

ADMISSION = {2: 290048, 4: 390016, 8: 490112, 16: 604416}
PHYSICAL = {2: 300032, 4: 400000, 8: 500096, 16: 614144}
F4_STAGE_SETS = (tuple(range(8, 16)), tuple(range(12, 16)))
F2_STAGE_SETS = (*F4_STAGE_SETS, (14, 15))
STAGE_SETS = {
    "natural_f4": F4_STAGE_SETS,
    "natural_f2": F2_STAGE_SETS,
    "planned_f4": F4_STAGE_SETS,
    "planned_f2": F2_STAGE_SETS,
}


def _timestamp(hour: int, minute: int, second: int) -> float:
    return datetime(2026, 8, 6, hour, minute, second, tzinfo=timezone.utc).timestamp()


def test_launcher_binds_all_lifecycles_to_profile_closure_and_policy() -> None:
    text = LAUNCHER.read_text(encoding="utf-8")

    for lifecycle in MODULE.LIFECYCLES:
        assert f"    {lifecycle})" in text
    assert "hash_deepseek_runtime_profile.py" in text
    assert "sha256sum" not in text
    assert "export DYNAMIC_SHRINK_POLICY=$SHRINK_POLICY" in text
    assert "${CAP_PREFIX}_HEADROOM_FLOOR${floor}" in text
    assert "${CAP_PREFIX}_TRAINING_MIN_FREE_MIB" in text


def _make_tree(
    tmp_path: Path, lifecycle: str = "natural_f4"
) -> tuple[Path, Path, Path]:
    lifecycle_spec = MODULE.LIFECYCLES[lifecycle]
    stage_sets = STAGE_SETS[lifecycle]
    survivors = stage_sets[-1]
    detached = tuple(rank for rank in range(16) if rank not in set(survivors))

    common = tmp_path / "common"
    (common / "epoch_000_mode0_probe").mkdir(parents=True)
    checkpoint = (
        common / "epoch_000_mode0_probe" / "checkpoints" / "deepseek" / "global_step_5"
    )
    (checkpoint / "actor").mkdir(parents=True)
    (checkpoint / ".PRESERVE_COMMON_EPOCH0").write_text(
        "preserve\n", encoding="utf-8"
    )
    (common / "reuse.env").write_text(
        f"export DYNAMIC_INITIAL_RESUME_CKPT={checkpoint}\n", encoding="utf-8"
    )
    (common / "DO_NOT_DELETE_COMMON_EPOCH0_CHECKPOINT").write_text(
        "complete\n", encoding="utf-8"
    )
    trigger = tmp_path / "positive_release_history"
    (trigger / "rollout_data").mkdir(parents=True)
    (trigger / "offline_planning_history.json").write_text("{}\n", encoding="utf-8")
    (trigger / "kv_probe_trigger_manifest.json").write_text("{}\n", encoding="utf-8")
    (trigger / "rollout_data" / "1.jsonl").write_text("{}\n", encoding="utf-8")

    profile_id, profile_sha256, _profile_files = MODULE.runtime_profile_provenance(
        lifecycle
    )
    cap_lines = [
        f"export {lifecycle_spec.verified_key}=1",
        "export DEEPSEEK_KV_CAP_TARGET_RATIO=1.0",
        "export DEEPSEEK_KV_CAP_BLOCK_SIZE=128",
        f"export {lifecycle_spec.recorded_runtime_profile_key}={profile_id}",
        f"export {lifecycle_spec.recorded_runtime_profile_sha_key}={profile_sha256}",
        f"export DEEPSEEK_KV_CAP_COMMON_EPOCH0_ROOT={common}",
        f"export DEEPSEEK_KV_CAP_PROBE_HISTORY_ROOT={trigger}",
    ]
    for floor in lifecycle_spec.floors:
        cap_lines.extend(
            [
                f"export {lifecycle_spec.prefix}_KV_ADMISSION_FLOOR{floor}="
                f"{ADMISSION[floor]}",
                f"export {lifecycle_spec.prefix}_KV_PHYSICAL_FLOOR{floor}="
                f"{PHYSICAL[floor]}",
            ]
        )
    if lifecycle_spec.policy == "planned":
        cap_lines.extend(
            f"export {lifecycle_spec.prefix}_HEADROOM_FLOOR{floor}=128"
            for floor in lifecycle_spec.floors
        )
        cap_lines.append(
            f"export {lifecycle_spec.prefix}_TRAINING_MIN_FREE_MIB=4096"
        )
    cap_env = tmp_path / "caps.env"
    cap_env.write_text("\n".join(cap_lines) + "\n", encoding="utf-8")

    run_root = tmp_path / "run"
    epoch = run_root / f"epoch_001_mode1_{lifecycle_spec.policy}"
    for relative in ("oracle", "logs", "sidecar"):
        (epoch / relative).mkdir(parents=True)
    target_floor = lifecycle_spec.target_floor
    plan = {
        "step": 1,
        "feasible": True,
        "selected_floor": target_floor,
        "shrink_stages": list(lifecycle_spec.stages),
        "stage_survivor_ranks": [list(ranks) for ranks in stage_sets],
        "intermediate_survivor_ranks": list(stage_sets[0]),
        "final_survivor_ranks": list(survivors),
        "kv_admission_cap": ADMISSION[target_floor],
        "kv_cap": PHYSICAL[target_floor],
        "max_adjusted_rank_peak_tokens": ADMISSION[target_floor] - 128,
        "release_area": 200.0,
        "release_area_unit": "rank_token_proxy",
        "rank_matching_policy": "release_area",
        "tail_guard_enabled": True,
        "length_prediction_baseline_dirs": [str(trigger)],
    }
    (epoch / "oracle" / "length_sorted_rank_plan_summary.json").write_text(
        json.dumps([plan]), encoding="utf-8"
    )

    primary_lines: list[str] = []
    final_shrink_line = ""
    for offset, active in enumerate(stage_sets):
        active_text = ", ".join(str(rank) for rank in active)
        final_shrink_line = (
            f"(WorkerDict pid=10) INFO 08-06 00:00:{50 + offset:02d} "
            "[worker_v1.py:1] Elastic parallel shrink done: "
            f"rank={active[0]} active_ranks=[{active_text}] "
            f"dp_size={len(active)} ep_size={len(active)} total_ms=1.0"
        )
        primary_lines.append(final_shrink_line)
    for rank in range(16):
        primary_lines.append(
            "(WorkerDict pid=10) INFO 08-06 00:02:00 [worker_v1.py:1] "
            f"Elastic full-world restore segmented timing: rank={rank} "
            "restore_seq=1 world_size=16 total_ms=2.0"
        )
    if lifecycle_spec.policy == "planned":
        for rank in range(16):
            primary_lines.append(
                "(WorkerDict pid=10) Mode1 training memory guard: "
                f"rank={rank} min_free_mib=4096 cleanup_triggered=0 "
                "free_before_bytes=8589934592 free_after_bytes=8589934592 "
                "torch_allocated_bytes=1 torch_reserved_bytes=1 "
                "non_torch_bytes=1 cleanup_methods=0 cleanup_topologies=0 "
                "cleanup_tensors=0 cleanup_tensor_bytes=0"
            )
    primary_lines.extend(
        [
            f"(TaskRunner pid=1) Resuming from {checkpoint}",
            "response/aborted_ratio:0.0",
            "rollout_output_time_s: 60.0",
            "training/global_step:1",
            "[TaskRunner] After trainer.fit(), about to finish run()",
        ]
    )
    (epoch / "logs" / "primary.txt").write_text(
        "\n".join(primary_lines) + "\n", encoding="utf-8"
    )

    lease_lines = [
        f"watch_start_time={_timestamp(0, 0, 0)}",
        f"watch_expected_active_ranks={target_floor}",
        "watch_world_size=16",
        "watch_start_trigger=shrink_done",
        f"shrink_window_detected_time={_timestamp(0, 1, 1)}",
        f"shrink_window_line={final_shrink_line}",
        "sidecar_devices_source=auto_from_inactive_ranks",
        f"sidecar_start_time={_timestamp(0, 1, 2)}",
        "sidecar_active_ranks=" + ",".join(str(rank) for rank in survivors),
        "sidecar_devices=" + ",".join(str(rank) for rank in detached),
        "sidecar_pid=1234",
        f"sidecar_exit_time={_timestamp(0, 1, 31)} sidecar_exit_code=0",
        f"watch_end_time={_timestamp(0, 1, 32)}",
    ]
    (epoch / "sidecar" / "lease.log").write_text(
        "\n".join(lease_lines) + "\n", encoding="utf-8"
    )

    model = tmp_path / "Qwen2.5-1.5B-Instruct"
    model.mkdir()
    load_event = {
        "event": "sidecar_load_start",
        "model_path": str(model),
        "devices": str(detached[0]),
        "num_prompts": 1,
    }
    done_event = {
        "event": "sidecar_done",
        "num_requests": 1,
        "num_output_tokens": 8,
    }
    infer_lines = [
        "sidecar_devices=" + ",".join(str(rank) for rank in detached),
        "sidecar_replica_count=1",
        "sidecar_parallel_mode=dp",
        "sidecar_tensor_parallel_size=1",
        "sidecar_data_parallel_size=1",
        f"sidecar_device_groups={detached[0]}",
        "sidecar_model=" + str(model),
        json.dumps(load_event),
        json.dumps(done_event),
        f"sidecar_end_time={_timestamp(0, 1, 30)}",
        "sidecar_exit_code=0",
        "sidecar_killed_by_deadline=0",
    ]
    (epoch / "sidecar" / "infer.log").write_text(
        "\n".join(infer_lines) + "\n", encoding="utf-8"
    )
    output = {
        "prompt_id": 0,
        "prompt": "question",
        "outputs": [
            {"text": "answer", "token_ids_len": 8, "finish_reason": "length"}
        ],
    }
    (epoch / "sidecar" / "outputs.jsonl").write_text(
        json.dumps(output) + "\n", encoding="utf-8"
    )
    return run_root, cap_env, model


def test_default_natural_f4_remains_compatible(tmp_path: Path) -> None:
    run_root, cap_env, model = _make_tree(tmp_path)

    summary = MODULE.verify_run(run_root, cap_env, model)

    assert summary["status"] == "PASS"
    assert summary["lifecycle_key"] == "natural_f4"
    assert summary["selected_floor"] == 4
    assert summary["survivor_ranks"] == list(STAGE_SETS["natural_f4"][-1])
    assert len(summary["detached_ranks"]) == 12
    assert summary["sidecar_used_ranks"] == [0]
    assert summary["sidecar_output_records"] == 1
    assert summary["sidecar_output_tokens"] == 8
    assert all(summary["invariants"].values())


def test_accepts_natural_f2_full_lifecycle_and_real_sidecar(tmp_path: Path) -> None:
    run_root, cap_env, model = _make_tree(tmp_path, "natural_f2")

    summary = MODULE.verify_run(
        run_root, cap_env, model, lifecycle="natural_f2"
    )

    assert summary["status"] == "PASS"
    assert summary["lifecycle_key"] == "natural_f2"
    assert summary["selected_floor"] == 2
    assert summary["stages"] == [8, 4, 2]
    assert summary["stage_survivor_ranks"] == [
        list(ranks) for ranks in STAGE_SETS["natural_f2"]
    ]
    assert summary["survivor_ranks"] == [14, 15]
    assert len(summary["detached_ranks"]) == 14
    assert summary["sidecar_used_ranks"] == [0]
    assert all(summary["invariants"].values())


@pytest.mark.parametrize(
    ("lifecycle", "floor", "survivors", "detached_count"),
    [
        ("planned_f4", 4, [12, 13, 14, 15], 12),
        ("planned_f2", 2, [14, 15], 14),
    ],
)
def test_accepts_planned_lifecycle_with_residency_guards(
    tmp_path: Path,
    lifecycle: str,
    floor: int,
    survivors: list[int],
    detached_count: int,
) -> None:
    run_root, cap_env, model = _make_tree(tmp_path, lifecycle)

    summary = MODULE.verify_run(run_root, cap_env, model, lifecycle=lifecycle)

    assert summary["status"] == "PASS"
    assert summary["lifecycle_key"] == lifecycle
    assert summary["policy"] == "planned"
    assert summary["selected_floor"] == floor
    assert summary["survivor_ranks"] == survivors
    assert len(summary["detached_ranks"]) == detached_count
    assert summary["planned_training_min_free_mib"] == 4096
    assert set(summary["planned_headroom_tokens"].values()) == {128}
    assert all(summary["invariants"].values())


def test_wrong_lifecycle_cannot_reuse_another_verified_cap_set(tmp_path: Path) -> None:
    run_root, cap_env, model = _make_tree(tmp_path, "natural_f4")

    with pytest.raises(MODULE.VerificationError, match="not VERIFIED"):
        MODULE.verify_run(run_root, cap_env, model, lifecycle="planned_f4")


def test_planned_caps_cannot_authorize_natural_runtime_artifacts(tmp_path: Path) -> None:
    run_root, cap_env, model = _make_tree(tmp_path, "planned_f4")
    planned_epoch = run_root / "epoch_001_mode1_planned"
    planned_epoch.rename(run_root / "epoch_001_mode1_natural")

    with pytest.raises(MODULE.VerificationError, match="epoch_001_mode1_planned"):
        MODULE.verify_run(run_root, cap_env, model, lifecycle="planned_f4")


def test_rejects_invalid_planned_headroom(tmp_path: Path) -> None:
    run_root, cap_env, model = _make_tree(tmp_path, "planned_f2")
    cap_env.write_text(
        cap_env.read_text(encoding="utf-8").replace(
            "DEEPSEEK_P_F2_HEADROOM_FLOOR2=128",
            "DEEPSEEK_P_F2_HEADROOM_FLOOR2=1",
        ),
        encoding="utf-8",
    )

    with pytest.raises(MODULE.VerificationError, match="nonnegative multiple"):
        MODULE.verify_run(run_root, cap_env, model, lifecycle="planned_f2")


def test_rejects_missing_planned_training_guard(tmp_path: Path) -> None:
    run_root, cap_env, model = _make_tree(tmp_path, "planned_f4")
    cap_env.write_text(
        cap_env.read_text(encoding="utf-8").replace(
            "DEEPSEEK_P_F4_TRAINING_MIN_FREE_MIB=4096",
            "DEEPSEEK_P_F4_TRAINING_MIN_FREE_MIB=0",
        ),
        encoding="utf-8",
    )

    with pytest.raises(MODULE.VerificationError, match="measured and positive"):
        MODULE.verify_run(run_root, cap_env, model, lifecycle="planned_f4")


def test_rejects_planned_runtime_without_training_guard_evidence(tmp_path: Path) -> None:
    run_root, cap_env, model = _make_tree(tmp_path, "planned_f4")
    log = run_root / "epoch_001_mode1_planned" / "logs" / "primary.txt"
    log.write_text(
        "\n".join(
            line
            for line in log.read_text(encoding="utf-8").splitlines()
            if "Mode1 training memory guard" not in line
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(MODULE.VerificationError, match="guard on all ranks"):
        MODULE.verify_run(run_root, cap_env, model, lifecycle="planned_f4")


def test_rejects_unverified_selected_lifecycle_caps(tmp_path: Path) -> None:
    run_root, cap_env, model = _make_tree(tmp_path, "natural_f2")
    cap_env.write_text(
        cap_env.read_text(encoding="utf-8").replace(
            "DEEPSEEK_N_F2_KV_CAPS_VERIFIED=1",
            "DEEPSEEK_N_F2_KV_CAPS_VERIFIED=0",
        ),
        encoding="utf-8",
    )

    with pytest.raises(MODULE.VerificationError, match="not VERIFIED"):
        MODULE.verify_run(run_root, cap_env, model, lifecycle="natural_f2")


def test_rejects_mismatched_runtime_profile_closure(tmp_path: Path) -> None:
    run_root, cap_env, model = _make_tree(tmp_path, "natural_f2")
    cap_env.write_text(
        cap_env.read_text(encoding="utf-8").replace(
            "DEEPSEEK_N_F2_RUNTIME_PROFILE_SHA256=",
            "DEEPSEEK_N_F2_RUNTIME_PROFILE_SHA256=stale",
        ),
        encoding="utf-8",
    )

    with pytest.raises(MODULE.VerificationError, match="profile closure"):
        MODULE.verify_run(run_root, cap_env, model, lifecycle="natural_f2")


def test_rejects_missing_floor2_runtime_stage(tmp_path: Path) -> None:
    run_root, cap_env, model = _make_tree(tmp_path, "natural_f2")
    log = run_root / "epoch_001_mode1_natural" / "logs" / "primary.txt"
    lines = log.read_text(encoding="utf-8").splitlines()
    log.write_text(
        "\n".join(line for line in lines if "active_ranks=[14, 15]" not in line)
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(MODULE.VerificationError, match="floor2 shrink_done"):
        MODULE.verify_run(run_root, cap_env, model, lifecycle="natural_f2")


def test_rejects_sidecar_device_overlapping_primary_survivor(tmp_path: Path) -> None:
    run_root, cap_env, model = _make_tree(tmp_path, "natural_f2")
    lease = run_root / "epoch_001_mode1_natural" / "sidecar" / "lease.log"
    lease.write_text(
        lease.read_text(encoding="utf-8").replace(
            "sidecar_devices=0,1,2,3,4,5,6,7,8,9,10,11,12,13",
            "sidecar_devices=0,1,2,3,4,5,6,7,8,9,10,11,12,14",
        ),
        encoding="utf-8",
    )

    with pytest.raises(MODULE.VerificationError, match="complement"):
        MODULE.verify_run(run_root, cap_env, model, lifecycle="natural_f2")


def test_rejects_sidecar_that_ends_after_restore(tmp_path: Path) -> None:
    run_root, cap_env, model = _make_tree(tmp_path)
    infer = run_root / "epoch_001_mode1_natural" / "sidecar" / "infer.log"
    infer.write_text(
        infer.read_text(encoding="utf-8").replace(
            f"sidecar_end_time={_timestamp(0, 1, 30)}",
            f"sidecar_end_time={_timestamp(0, 2, 1)}",
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        MODULE.VerificationError, match="before the first full-world restore"
    ):
        MODULE.verify_run(run_root, cap_env, model)


def test_rejects_empty_or_skipped_sidecar_output(tmp_path: Path) -> None:
    run_root, cap_env, model = _make_tree(tmp_path)
    output = run_root / "epoch_001_mode1_natural" / "sidecar" / "outputs.jsonl"
    output.write_text(
        json.dumps(
            {
                "prompt_id": 0,
                "sidecar_status": "context_overflow_skipped",
                "outputs": [{"text": "", "token_ids_len": 0}],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(MODULE.VerificationError, match="not a completed generation"):
        MODULE.verify_run(run_root, cap_env, model)


def test_rejects_primary_preemption(tmp_path: Path) -> None:
    run_root, cap_env, model = _make_tree(tmp_path)
    log = run_root / "epoch_001_mode1_natural" / "logs" / "primary.txt"
    log.write_text(
        log.read_text(encoding="utf-8")
        + "preempting request due to insufficient KV\n",
        encoding="utf-8",
    )

    with pytest.raises(MODULE.VerificationError, match="preemption"):
        MODULE.verify_run(run_root, cap_env, model)
