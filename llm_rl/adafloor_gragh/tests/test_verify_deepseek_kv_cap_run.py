from __future__ import annotations

import hashlib
import importlib.util
import json
import re
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).parents[1]
TOOL = ROOT / "tools" / "verify_deepseek_kv_cap_run.py"
SPEC = importlib.util.spec_from_file_location("verify_deepseek_kv_cap_run", TOOL)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.path.insert(0, str(ROOT / "tools"))
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

FLOORS = (16, 8, 4)
PHYSICAL = {2: 300032, 4: 400000, 8: 500096, 16: 614144}
ADMISSION = {2: 290048, 4: 390016, 8: 490112, 16: 604416}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _update_env(path: Path, updates: dict[str, str | int]) -> None:
    values = MODULE._load_env(path)
    values.update({key: str(value) for key, value in updates.items()})
    path.write_text(
        "\n".join(f"export {key}={value}" for key, value in values.items()) + "\n",
        encoding="utf-8",
    )


def _rank_text(ranks: tuple[int, ...]) -> str:
    return ", ".join(str(rank) for rank in ranks)


def _runtime_log(
    floor: int,
    *,
    missing_resize_rank: int | None = None,
    wrong_target_rank: int | None = None,
    missing_shrink: tuple[int, int] | None = None,
    missing_restore_rank: int | None = None,
    health_marker: str = "",
    tail_guard_cap: int | None = 4096,
    batch_size: int = 32,
) -> str:
    lines: list[str] = []
    full_world = tuple(range(16))
    floor8 = tuple(range(8, 16))
    floor4 = tuple(range(12, 16))
    floor2 = tuple(range(14, 16))
    physical = PHYSICAL[floor]
    if tail_guard_cap is not None:
        lines.append(
            "Shrink-aware tail-guard response cap: "
            f"selected_floor={floor} plan_cap={tail_guard_cap} "
            "ratio=1.0 ratio_q=1.0 predicted_step_exit=64"
        )
    for rank in range(16):
        pid = 1000 + rank
        if tail_guard_cap is not None:
            lines.append(
                f"(WorkerDict pid={pid}) [mode1_timeline] "
                f"rollout_worker_infer_start rank={rank} step=1 epoch=0 "
                f"batch_size={batch_size} max_tokens={tail_guard_cap} total_s=0.002"
            )
        if rank == missing_resize_rank:
            continue
        target = physical + 128 if rank == wrong_target_rank else physical
        lines.extend(
            [
                f"(WorkerDict pid={pid}) [mode1_timeline] "
                f"rollout_worker_resize_start rank={rank} step=1 epoch=0 "
                f"target_floor={floor} target_kv={target} total_s=0.000",
                f"(WorkerDict pid={pid}) INFO Mode1 adaptive KV resize "
                "phase=plan_new_kv_done "
                f"target_tokens={target} effective_target_tokens={target} "
                f"new_tokens={target} new_blocks={target // 128}",
                f"(WorkerDict pid={pid}) [mode1_timeline] "
                f"rollout_worker_resize_done rank={rank} step=1 epoch=0 "
                "elapsed_s=0.001 total_s=0.001",
            ]
        )
    transitions: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
    if floor == 8:
        transitions = [(full_world, floor8)]
    elif floor == 4:
        transitions = [(full_world, floor8), (floor8, floor4)]
    elif floor == 2:
        transitions = [
            (full_world, floor8),
            (floor8, floor4),
            (floor4, floor2),
        ]
    for current, target in transitions:
        for rank in current:
            if missing_shrink == (len(target), rank):
                continue
            pid = 1000 + rank
            lines.append(
                f"(WorkerDict pid={pid}) INFO Shrink-aware staged trigger: "
                f"stage=donor current_local=[{_rank_text(current)}] "
                f"unfinished_local=[{_rank_text(target)}] "
                f"target_local=[{_rank_text(target)}] "
                f"target_global=[{_rank_text(target)}]"
            )

    if floor < 16:
        for rank in range(16):
            if rank == missing_restore_rank:
                continue
            pid = 1000 + rank
            lines.append(
                f"(WorkerDict pid={pid}) INFO Elastic full-world restore "
                f"segmented timing: rank={rank} restore_seq=1 world_size=16 "
                "rebuild_ms=1.0 total_ms=2.0"
            )
    lines.extend(
        [
            "response/aborted_ratio:0.0",
            "rollout_output_time_s: 1.0",
            "training/global_step:1",
            "After trainer.fit",
        ]
    )
    if health_marker:
        lines.append(health_marker)
    return "\n".join(lines) + "\n"


def _plan(floor: int, baseline_dir: Path | None = None) -> dict[str, object]:
    peak = ADMISSION[floor] - 128
    if floor == 16:
        stages = [16]
        sets = [list(range(16))]
        thresholds: list[int] = []
        release = 0
    elif floor == 8:
        stages = [8]
        sets = [list(range(8, 16))]
        thresholds = [17]
        release = 376
    elif floor == 4:
        stages = [8, 4]
        sets = [list(range(8, 16)), list(range(12, 16))]
        thresholds = [17, 33]
        release = 504
    else:
        stages = [8, 4, 2]
        sets = [
            list(range(8, 16)),
            list(range(12, 16)),
            list(range(14, 16)),
        ]
        thresholds = [17, 33, 49]
        release = 536
    return {
        "step": 1,
        "selected_floor": floor,
        "feasible": True,
        "rank_matching_policy": "release_area",
        "kv_admission_cap": ADMISSION[floor],
        "kv_cap": PHYSICAL[floor],
        "max_adjusted_rank_peak_tokens": peak,
        "kv_admission_headroom_tokens": ADMISSION[floor] - peak,
        "kv_physical_headroom_tokens": PHYSICAL[floor] - peak,
        "shrink_stages": stages,
        "stage_survivor_ranks": sets,
        "intermediate_survivor_ranks": sets[0],
        "final_survivor_ranks": sets[-1],
        "release_area": release,
        "release_area_unit": "rank_token_proxy",
        "schedule_thresholds": thresholds,
        "predicted_step_exit": 64,
        "tail_guard_response_cap": 4096,
        "tail_guard_enabled": True,
        "length_prediction_mode": "single_epoch_prompt_max",
        "length_prediction_baseline_dirs": (
            [str(baseline_dir.resolve())] if baseline_dir is not None else []
        ),
        "kv_length_source": "latest_observed_response_multiset",
    }


def _authorization_tree(
    tmp_path: Path, lifecycle: str = "natural_f4"
) -> tuple[Path, Path, Path, Path]:
    config = MODULE.lifecycle_config(lifecycle)
    floors = tuple(config["floors"])
    prefix = config["prefix"]
    common = tmp_path / "common"
    checkpoint = common / "epoch_000_mode0_probe" / "checkpoints" / "deepseek_v2_lite" / "global_step_5"
    (checkpoint / "actor").mkdir(parents=True)
    (checkpoint / ".PRESERVE_COMMON_EPOCH0").write_text("preserve\n", encoding="utf-8")
    (common / "DO_NOT_DELETE_COMMON_EPOCH0_CHECKPOINT").write_text("complete\n", encoding="utf-8")
    (common / "reuse.env").write_text(
        f"export DYNAMIC_INITIAL_RESUME_CKPT={checkpoint}\n", encoding="utf-8"
    )
    metadata = {
        **MODULE.COMMON_PROTOCOL,
        "COMMON_EPOCH0_MODEL_REVISION": "revision",
        "COMMON_EPOCH0_EXECUTION_PROFILE_USED": "execution-profile",
    }
    (common / "common_epoch0_metadata.env").write_text(
        "\n".join(f"export {key}={value}" for key, value in metadata.items()) + "\n",
        encoding="utf-8",
    )

    trigger = tmp_path / "trigger"
    (trigger / "rollout_data").mkdir(parents=True)
    (trigger / "offline_planning_history.json").write_text("{}\n", encoding="utf-8")
    (trigger / "kv_probe_trigger_manifest.json").write_text("{}\n", encoding="utf-8")
    (trigger / "rollout_data" / "1.jsonl").write_text("{}\n", encoding="utf-8")

    run_root = tmp_path / "authorization"
    planner_hashes: dict[int, str] = {}
    policy = str(config["policy"])
    for floor in floors:
        epoch = run_root / f"floor{floor}" / f"epoch_001_mode1_{policy}"
        for directory in ("logs", "rollout_data", "rollout_length", "oracle"):
            (epoch / directory).mkdir(parents=True, exist_ok=True)
        (epoch / "logs" / "run.txt").write_text(_runtime_log(floor), encoding="utf-8")
        (epoch / "rollout_data" / "1.jsonl").write_text("{}\n" * 512, encoding="utf-8")
        (epoch / "rollout_length" / "length_1.txt").write_text("1\n" * 512, encoding="utf-8")
        (epoch / "oracle" / "length_sorted_rank_plan_summary.json").write_text(
            json.dumps([_plan(floor, trigger)]), encoding="utf-8"
        )
        planner = epoch / "oracle" / "length_sorted_train.parquet"
        planner.write_bytes(f"planner-floor{floor}".encode())
        planner_hashes[floor] = _sha256(planner)

    runtime_profile_path = Path(config["runtime_profile_path"])
    profile_text = runtime_profile_path.read_text(encoding="utf-8")
    profile_id = re.search(
        rf"^export {config['runtime_profile_id_key']}=([^\s]+)$",
        profile_text,
        re.M,
    )
    assert profile_id is not None
    execution_sha256, _count = MODULE.execution_digest(ROOT)
    cap_env = tmp_path / "caps.env"
    lines = [
        "export DEEPSEEK_KV_CAPS_VERIFIED=0",
        "export DEEPSEEK_N_F4_KV_CAPS_VERIFIED=0",
        "export DEEPSEEK_N_F2_KV_CAPS_VERIFIED=0",
        "export DEEPSEEK_P_F4_KV_CAPS_VERIFIED=0",
        "export DEEPSEEK_P_F2_KV_CAPS_VERIFIED=0",
        "export DEEPSEEK_KV_CAP_TARGET_RATIO=1.0",
        "export DEEPSEEK_KV_CAP_BLOCK_SIZE=128",
        f"export DEEPSEEK_KV_CAP_COMMON_EPOCH0_ROOT={common}",
        f"export DEEPSEEK_KV_CAP_PROBE_HISTORY_ROOT={trigger}",
        f"export DEEPSEEK_KV_CAP_PROBE_HISTORY_SHA256={_sha256(trigger / 'offline_planning_history.json')}",
        f"export DEEPSEEK_KV_CAP_PROBE_HISTORY_MANIFEST_SHA256={_sha256(trigger / 'kv_probe_trigger_manifest.json')}",
        f"export DEEPSEEK_KV_CAP_PROBE_TRIGGER_SUBSET_SHA256={_sha256(trigger / 'rollout_data' / '1.jsonl')}",
        f"export {prefix}_RUNTIME_PROFILE={profile_id.group(1)}",
        f"export {prefix}_RUNTIME_PROFILE_SHA256="
        f"{MODULE._runtime_profile_sha256(config['runtime_profile_files'])}",
        f"export DEEPSEEK_EXECUTION_CODE_SHA256={execution_sha256}",
        "export DEEPSEEK_KV_CAP_MODEL_REVISION=revision",
        "export DEEPSEEK_KV_CAP_EXECUTION_PROFILE=execution-profile",
    ]
    if policy == "planned":
        lines.append(f"export {prefix}_TRAINING_MIN_FREE_MIB=4096")
        lines.extend(
            f"export {prefix}_HEADROOM_FLOOR{floor}=0" for floor in floors
        )
    for floor in floors:
        lines.extend(
            [
                f"export {prefix}_KV_ADMISSION_FLOOR{floor}={ADMISSION[floor]}",
                f"export {prefix}_KV_PHYSICAL_FLOOR{floor}={PHYSICAL[floor]}",
                f"export {prefix}_KV_PROBE_PLANNER_TRAIN_SHA256_FLOOR{floor}="
                f"{planner_hashes[floor]}",
            ]
        )
    cap_env.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return cap_env, run_root, common, trigger


def test_validates_three_independent_runs_before_promotion(tmp_path: Path) -> None:
    cap_env, run_root, common, trigger = _authorization_tree(tmp_path)

    summary = MODULE.validate_authorization(cap_env, run_root, common, trigger)
    assert summary["status"] == "PASS"
    assert [item["floor"] for item in summary["floors"]] == [16, 8, 4]
    assert [item["resize_calls"] for item in summary["floors"]] == [16, 16, 16]
    assert [item["restore_events"] for item in summary["floors"]] == [0, 16, 16]
    assert all(
        item["release_area_unit"] == "rank_token_proxy"
        for item in summary["floors"]
    )

    summary_path = run_root / "KV_CAP_AUTHORIZATION_SUMMARY.json"
    summary_path.write_text(json.dumps(summary) + "\n", encoding="utf-8")
    MODULE.promote(cap_env, run_root, summary_path)
    promoted = cap_env.read_text(encoding="utf-8")
    assert "export DEEPSEEK_N_F4_KV_CAPS_VERIFIED=1" in promoted
    assert "export DEEPSEEK_KV_CAPS_VERIFIED=0" in promoted
    assert "export DEEPSEEK_N_F2_KV_CAPS_VERIFIED=0" in promoted
    assert "export DEEPSEEK_P_F4_KV_CAPS_VERIFIED=0" in promoted
    assert "export DEEPSEEK_P_F2_KV_CAPS_VERIFIED=0" in promoted
    assert "export DEEPSEEK_N_F4_KV_CAP_VALIDATED_FLOORS=16,8,4" in promoted


def test_parameterizes_batch64_workload_and_profile_from_provenance(
    tmp_path: Path,
) -> None:
    cap_env, run_root, common, trigger = _authorization_tree(tmp_path)
    profile_id = "deepseek-v2-lite-chat-b64-n16-s5-v2"
    profile_sha256 = "a" * 64
    _update_env(
        common / "common_epoch0_metadata.env",
        {
            "COMMON_EPOCH0_TRAIN_BATCH_SIZE_USED": 64,
            "COMMON_EPOCH0_EXPECTED_RESPONSES_PER_STEP_USED": 1024,
            "COMMON_EPOCH0_MAX_NUM_SEQS_USED": 64,
            "COMMON_EPOCH0_MAX_RESPONSE_LENGTH_USED": 8192,
            "COMMON_EPOCH0_WORKLOAD_PROFILE_ID": profile_id,
            "COMMON_EPOCH0_WORKLOAD_PROFILE_SHA256": profile_sha256,
        },
    )
    _update_env(
        cap_env,
        {
            "DEEPSEEK_KV_CAP_TRAIN_BATCH_SIZE": 64,
            "DEEPSEEK_KV_CAP_ROLLOUT_N": 16,
            "DEEPSEEK_KV_CAP_EXPECTED_RESPONSES_PER_STEP": 1024,
            "DEEPSEEK_KV_CAP_MAX_NUM_SEQS": 64,
            "DEEPSEEK_KV_CAP_MAX_RESPONSE_LENGTH": 8192,
            "DEEPSEEK_KV_CAP_WORKLOAD_PROFILE_ID": profile_id,
            "DEEPSEEK_KV_CAP_WORKLOAD_PROFILE_SHA256": profile_sha256,
        },
    )
    for floor in FLOORS:
        epoch = run_root / f"floor{floor}" / "epoch_001_mode1_natural"
        (epoch / "logs" / "run.txt").write_text(
            _runtime_log(floor, batch_size=64), encoding="utf-8"
        )
        (epoch / "rollout_data" / "1.jsonl").write_text(
            "{}\n" * 1024, encoding="utf-8"
        )
        (epoch / "rollout_length" / "length_1.txt").write_text(
            "4096\n" * 1024, encoding="utf-8"
        )

    summary = MODULE.validate_authorization(cap_env, run_root, common, trigger)

    assert summary["train_batch_size"] == 64
    assert summary["rollout_n"] == 16
    assert summary["expected_responses_per_step"] == 1024
    assert summary["max_num_seqs"] == 64
    assert summary["max_response_length"] == 8192
    assert summary["workload_profile_id"] == profile_id
    assert summary["workload_profile_sha256"] == profile_sha256
    assert all(item["observed_max_response_length"] == 4096 for item in summary["floors"])
    assert all(item["tail_guard_log_markers"] == 1 for item in summary["floors"])


def test_rejects_cli_workload_override_that_disagrees_with_cap_env(
    tmp_path: Path,
) -> None:
    cap_env, run_root, common, trigger = _authorization_tree(tmp_path)

    with pytest.raises(MODULE.AuthorizationError, match="workload mismatch for train_batch_size"):
        MODULE.validate_authorization(
            cap_env,
            run_root,
            common,
            trigger,
            workload_overrides={"train_batch_size": 64},
        )


def test_records_separate_runtime_and_verification_hashes_for_audited_migration(
    tmp_path: Path,
) -> None:
    cap_env, run_root, common, trigger = _authorization_tree(tmp_path)
    runtime_sha256 = "a" * 64
    verification_sha256, _count = MODULE.execution_digest(ROOT)
    _update_env(
        cap_env,
        {"DEEPSEEK_EXECUTION_CODE_SHA256": runtime_sha256},
    )

    summary = MODULE.validate_authorization(
        cap_env,
        run_root,
        common,
        trigger,
        expected_runtime_execution_sha256=runtime_sha256,
        expected_verification_code_sha256=verification_sha256,
    )

    assert summary["provenance"]["runtime_execution_code_sha256"] == runtime_sha256
    assert summary["provenance"]["verification_code_sha256"] == verification_sha256


def test_validates_natural_floor2_and_promotes_only_its_prefix(
    tmp_path: Path,
) -> None:
    cap_env, run_root, common, trigger = _authorization_tree(
        tmp_path, "natural_f2"
    )

    summary = MODULE.validate_authorization(
        cap_env, run_root, common, trigger, lifecycle="natural_f2"
    )
    assert summary["status"] == "PASS"
    assert summary["lifecycle"] == "natural_f2"
    assert [item["floor"] for item in summary["floors"]] == [16, 8, 4, 2]
    assert [item["resize_calls"] for item in summary["floors"]] == [16] * 4
    assert [item["shrink_events"] for item in summary["floors"]] == [0, 16, 24, 28]
    assert [item["restore_events"] for item in summary["floors"]] == [0, 16, 16, 16]

    summary_path = run_root / "KV_CAP_AUTHORIZATION_SUMMARY.json"
    summary_path.write_text(json.dumps(summary) + "\n", encoding="utf-8")
    MODULE.promote(
        cap_env, run_root, summary_path, lifecycle="natural_f2"
    )
    promoted = cap_env.read_text(encoding="utf-8")
    assert "export DEEPSEEK_N_F2_KV_CAPS_VERIFIED=1" in promoted
    assert "export DEEPSEEK_N_F4_KV_CAPS_VERIFIED=0" in promoted
    assert "export DEEPSEEK_P_F4_KV_CAPS_VERIFIED=0" in promoted
    assert "export DEEPSEEK_P_F2_KV_CAPS_VERIFIED=0" in promoted
    assert "export DEEPSEEK_N_F2_KV_CAP_VALIDATED_FLOORS=16,8,4,2" in promoted
    assert "DEEPSEEK_N_F4_KV_CAP_VALIDATED_FLOORS" not in promoted


@pytest.mark.parametrize(
    ("lifecycle", "expected_floors"),
    [
        ("planned_f4", [16, 8, 4]),
        ("planned_f2", [16, 8, 4, 2]),
    ],
)
def test_validates_planned_lifecycle_and_promotes_only_its_prefix(
    tmp_path: Path, lifecycle: str, expected_floors: list[int]
) -> None:
    cap_env, run_root, common, trigger = _authorization_tree(tmp_path, lifecycle)

    summary = MODULE.validate_authorization(
        cap_env, run_root, common, trigger, lifecycle=lifecycle
    )
    assert summary["status"] == "PASS"
    assert summary["lifecycle"] == lifecycle
    assert [item["floor"] for item in summary["floors"]] == expected_floors
    assert all(
        item["runtime_lifecycle"]["selected_floor"] == item["floor"]
        for item in summary["floors"]
    )

    summary_path = run_root / "KV_CAP_AUTHORIZATION_SUMMARY.json"
    summary_path.write_text(json.dumps(summary) + "\n", encoding="utf-8")
    MODULE.promote(
        cap_env, run_root, summary_path, lifecycle=lifecycle
    )
    promoted = cap_env.read_text(encoding="utf-8")
    prefix = MODULE.lifecycle_config(lifecycle)["prefix"]
    assert f"export {prefix}_KV_CAPS_VERIFIED=1" in promoted
    assert (
        f"export {prefix}_KV_CAP_VALIDATED_FLOORS="
        + ",".join(str(floor) for floor in expected_floors)
    ) in promoted
    for other_prefix in ("DEEPSEEK_N_F4", "DEEPSEEK_N_F2", "DEEPSEEK_P_F4", "DEEPSEEK_P_F2"):
        if other_prefix != prefix:
            assert f"export {other_prefix}_KV_CAPS_VERIFIED=0" in promoted


def test_natural_floor2_requires_floor2_authorization_run(tmp_path: Path) -> None:
    cap_env, run_root, common, trigger = _authorization_tree(
        tmp_path, "natural_f2"
    )
    missing = run_root / "floor2" / "epoch_001_mode1_natural"
    missing.rename(run_root / "floor2" / "epoch_001_mode1_natural.missing")

    with pytest.raises(MODULE.AuthorizationError, match="floor2"):
        MODULE.validate_authorization(
            cap_env, run_root, common, trigger, lifecycle="natural_f2"
        )
    assert "export DEEPSEEK_N_F2_KV_CAPS_VERIFIED=0" in cap_env.read_text(
        encoding="utf-8"
    )


@pytest.mark.parametrize(
    ("floor", "log_text", "message"),
    [
        (16, _runtime_log(16, missing_resize_rank=15), "resize coverage mismatch"),
        (8, _runtime_log(8, wrong_target_rank=7), "runtime target floor/KV"),
        (4, _runtime_log(4, missing_shrink=(4, 8)), "logger ranks"),
        (8, _runtime_log(8, missing_restore_rank=15), "full-world restore"),
        (4, _runtime_log(4, health_marker="request preempted"), "preemption marker"),
        (4, _runtime_log(4, health_marker="OutOfMemoryError"), "out-of-memory marker"),
    ],
)
def test_rejects_any_invalid_floor_without_promotion(
    tmp_path: Path, floor: int, log_text: str, message: str
) -> None:
    cap_env, run_root, common, trigger = _authorization_tree(tmp_path)
    log_path = run_root / f"floor{floor}" / "epoch_001_mode1_natural" / "logs" / "run.txt"
    log_path.write_text(log_text, encoding="utf-8")

    with pytest.raises(MODULE.AuthorizationError, match=message):
        MODULE.validate_authorization(cap_env, run_root, common, trigger)
    assert "export DEEPSEEK_N_F4_KV_CAPS_VERIFIED=0" in cap_env.read_text(encoding="utf-8")


def test_rejects_missing_runtime_tail_guard_application_marker(tmp_path: Path) -> None:
    cap_env, run_root, common, trigger = _authorization_tree(tmp_path)
    log_path = run_root / "floor8" / "epoch_001_mode1_natural" / "logs" / "run.txt"
    log_path.write_text(
        _runtime_log(8, tail_guard_cap=None), encoding="utf-8"
    )

    with pytest.raises(MODULE.AuthorizationError, match="TailGuard plan cap was applied"):
        MODULE.validate_authorization(cap_env, run_root, common, trigger)


def test_rejects_observed_response_above_tail_guard_plan_cap(tmp_path: Path) -> None:
    cap_env, run_root, common, trigger = _authorization_tree(tmp_path)
    length_path = (
        run_root
        / "floor8"
        / "epoch_001_mode1_natural"
        / "rollout_length"
        / "length_1.txt"
    )
    length_path.write_text("4097\n" + "1\n" * 511, encoding="utf-8")

    with pytest.raises(MODULE.AuthorizationError, match="synthetic terminal padding"):
        MODULE.validate_authorization(cap_env, run_root, common, trigger)


def test_normalizes_only_proven_synthetic_terminal_padding_eos(
    tmp_path: Path,
) -> None:
    cap_env, run_root, common, trigger = _authorization_tree(tmp_path)
    epoch = run_root / "floor8" / "epoch_001_mode1_natural"
    rows = (epoch / "rollout_data" / "1.jsonl").read_text(
        encoding="utf-8"
    ).splitlines()
    rows[0] = json.dumps(
        {
            "responses": [7] * 4096 + [100001] * 4,
            "response_mask": [1] * 4097 + [0] * 3,
        }
    )
    (epoch / "rollout_data" / "1.jsonl").write_text(
        "\n".join(rows) + "\n", encoding="utf-8"
    )
    lengths = (epoch / "rollout_length" / "length_1.txt").read_text(
        encoding="utf-8"
    ).splitlines()
    lengths[0] = "4097"
    (epoch / "rollout_length" / "length_1.txt").write_text(
        "\n".join(lengths) + "\n", encoding="utf-8"
    )

    summary = MODULE.validate_authorization(cap_env, run_root, common, trigger)
    floor8 = next(item for item in summary["floors"] if item["floor"] == 8)
    assert floor8["recorded_max_response_mask_length"] == 4097
    assert floor8["observed_max_response_length"] == 4096
    assert floor8["synthetic_terminal_pad_count"] == 1
    assert floor8["sampling_cap_log_markers"] == 16


def test_rejects_missing_floor_run_without_promotion(tmp_path: Path) -> None:
    cap_env, run_root, common, trigger = _authorization_tree(tmp_path)
    missing = run_root / "floor8" / "epoch_001_mode1_natural"
    missing.rename(run_root / "floor8" / "epoch_001_mode1_natural.missing")

    with pytest.raises(MODULE.AuthorizationError, match="floor8"):
        MODULE.validate_authorization(cap_env, run_root, common, trigger)
    assert "export DEEPSEEK_N_F4_KV_CAPS_VERIFIED=0" in cap_env.read_text(encoding="utf-8")


def test_rejects_planner_artifact_not_bound_to_probe(tmp_path: Path) -> None:
    cap_env, run_root, common, trigger = _authorization_tree(tmp_path)
    planner = run_root / "floor8" / "epoch_001_mode1_natural" / "oracle" / "length_sorted_train.parquet"
    planner.write_bytes(b"different prompt mapping")

    with pytest.raises(MODULE.AuthorizationError, match="planner train artifact SHA256"):
        MODULE.validate_authorization(cap_env, run_root, common, trigger)


def test_validation_launcher_uses_independent_strict_one_step_runs() -> None:
    source = (ROOT / "run_deepseek_v2_lite_kv_cap_validation.sh").read_text(encoding="utf-8")
    assert "natural_f4)" in source
    assert "natural_f2)" in source
    assert "planned_f4)" in source
    assert "planned_f2)" in source
    assert "floors=(16 8 4)" in source
    assert "floors=(16 8 4 2)" in source
    assert 'for floor in "${floors[@]}"; do' in source
    assert '--lifecycle "$lifecycle"' in source
    assert 'require_value "${prefix}_KV_CAPS_VERIFIED" 0' in source
    assert 'if [[ "$policy" == planned ]]; then' in source
    assert "export DYNAMIC_PLAN_STEPS=1" in source
    assert "export DYNAMIC_TRAIN_STEPS=1" in source
    assert "export DYNAMIC_INITIAL_BASELINE_DIR=$TRIGGER_ROOT" in source
    assert "trainer.resume_from_path=$DYNAMIC_INITIAL_RESUME_CKPT" in source
    assert "export VLLM_ASCEND_MODE1_PARITY_NATIVE_KV_CAP=1" in source
    assert "export VLLM_ASCEND_MODE1_ADAPTIVE_KV_FAIL_ON_UNMET_TARGET=1" in source
    assert "export VLLM_ASCEND_MODE1_ADAPTIVE_KV_MIN_TARGET_RATIO=1.0" in source
    assert "DYNAMIC_RUNTIME_TAIL_VALIDATE_LEVEL_TOKENS_BY_STEP=" not in source
    assert "4,8,16,16,16" not in source
    assert "ALLOW_INFEASIBLE_PLAN=1" not in source
