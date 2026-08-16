import importlib.util
import json
import sys
from pathlib import Path

import pytest


MODULE_PATH = Path(__file__).parents[1] / "tools" / "audit_deepseek_n_f4_formal_run.py"
SPEC = importlib.util.spec_from_file_location("audit_deepseek_n_f4_formal_run", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
AUDIT = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = AUDIT
SPEC.loader.exec_module(AUDIT)

ADMISSION = {2: 290048, 4: 390016, 8: 490112, 16: 604416}
PHYSICAL = {2: 300032, 4: 400000, 8: 500096, 16: 614144}
STAGE_SETS = {
    2: [
        [8, 9, 10, 11, 12, 13, 14, 15],
        [12, 13, 14, 15],
        [14, 15],
    ],
    4: [[8, 9, 10, 11, 12, 13, 14, 15], [12, 13, 14, 15]],
    8: [[8, 9, 10, 11, 12, 13, 14, 15]],
    16: [list(range(16))],
}
STAGES = {2: [8, 4, 2], 4: [8, 4], 8: [8], 16: [16]}


def _write_caps(path: Path, lifecycle: str = "natural_f4") -> None:
    config = AUDIT.lifecycle_config(lifecycle)
    prefix = config["prefix"]
    floors = config["floors"]
    profile_path = Path(config["runtime_profile_path"])
    profile_id = AUDIT._runtime_profile_id(
        profile_path, config["runtime_profile_id_key"]
    )
    lines = [
        f"export {prefix}_KV_CAPS_VERIFIED=1",
        "export DEEPSEEK_KV_CAP_TARGET_RATIO=1.0",
        "export DEEPSEEK_KV_CAP_BLOCK_SIZE=128",
        f"export {prefix}_KV_CAP_VALIDATED_FLOORS="
        + ",".join(str(floor) for floor in floors),
        f"export {prefix}_RUNTIME_PROFILE={profile_id}",
        f"export {prefix}_RUNTIME_PROFILE_SHA256="
        f"{AUDIT._runtime_profile_sha256(config['runtime_profile_files'])}",
    ]
    if config["policy"] == "planned":
        lines.append(f"export {prefix}_TRAINING_MIN_FREE_MIB=4096")
        lines.extend(
            f"export {prefix}_HEADROOM_FLOOR{floor}=0" for floor in floors
        )
    for floor in floors:
        lines.extend(
            [
                f"export {prefix}_KV_ADMISSION_FLOOR{floor}={ADMISSION[floor]}",
                f"export {prefix}_KV_PHYSICAL_FLOOR{floor}={PHYSICAL[floor]}",
            ]
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _plan(step: int, floor: int, history_dirs: list[str]) -> dict:
    peak = float(ADMISSION[floor]) - 128.5
    sets = STAGE_SETS[floor]
    return {
        "step": step,
        "feasible": True,
        "selected_floor": floor,
        "rank_matching_policy": "release_area",
        "release_area_unit": "rank_token_proxy",
        "kv_admission_cap": float(ADMISSION[floor]),
        "kv_cap": float(PHYSICAL[floor]),
        "max_adjusted_rank_peak_tokens": peak,
        "kv_admission_headroom_tokens": 128.5,
        "kv_physical_headroom_tokens": PHYSICAL[floor] - peak,
        "shrink_stages": STAGES[floor],
        "stage_survivor_ranks": sets,
        "intermediate_survivor_ranks": sets[0],
        "final_survivor_ranks": sets[-1],
        "length_prediction_baseline_dirs": history_dirs,
        "length_prediction_mode": (
            "single_epoch_prompt_max"
            if len(history_dirs) == 1
            else "prompt_max_ema_history"
        ),
        "max_response_len": 16384,
        "tail_guard_response_cap": 4096,
        "tail_guard_enabled": True,
        "tail_guard_prompt_tail_stat": "max_response_over_rollout_n",
    }


def _worker(pid: int, message: str) -> str:
    return f"(WorkerDict pid={pid}) {message}\n"


def _runtime_log(floors: list[int], epoch: int) -> str:
    lines: list[str] = []
    restore_seq = 100 * epoch
    pids = {rank: 10000 * epoch + rank for rank in range(16)}
    for step, floor in enumerate(floors, 1):
        cap = PHYSICAL[floor]
        for rank in range(16):
            pid = pids[rank]
            lines.append(
                _worker(
                    pid,
                    f"rollout_worker_resize_start rank={rank} step={step} epoch={epoch} "
                    f"target_floor={floor} target_kv={cap}",
                )
            )
            lines.append(
                _worker(
                    pid,
                    "Mode1 adaptive KV resize phase=plan_new_kv_done "
                    f"target_tokens={cap} effective_target_tokens={cap} new_tokens={cap}",
                )
            )
            lines.append(
                _worker(
                    pid,
                    f"rollout_worker_resize_done rank={rank} step={step} epoch={epoch} "
                    f"target_floor={floor} target_kv={cap}",
                )
            )

        current = list(range(16))
        for target in STAGE_SETS[floor]:
            if len(target) == 16:
                continue
            for logger_rank in current:
                lines.append(
                    _worker(
                        pids[logger_rank],
                        "Shrink-aware staged trigger: stage=test "
                        f"current_local={current} unfinished_local=[] "
                        f"target_local={target} target_global={target}",
                    )
                )
            current = target
        if floor < 16:
            restore_seq += 1
            for rank in range(16):
                lines.append(
                    _worker(
                        pids[rank],
                        "Elastic full-world restore segmented timing: "
                        f"rank={rank} restore_seq={restore_seq} world_size=16",
                    )
                )
        lines.append(
            f"rollout_output_time_s:1.0 - response/aborted_ratio:0.0 - "
            f"training/global_step:{step}\n"
        )
    lines.append("After trainer.fit\n")
    return "".join(lines)


def _make_run(
    tmp_path: Path, lifecycle: str = "natural_f4"
) -> tuple[Path, Path]:
    run_root = tmp_path / "run"
    cap_env = tmp_path / "caps.env"
    _write_caps(cap_env, lifecycle)
    if lifecycle.endswith("_f2"):
        epoch_floors = {1: [2, 4, 8, 16, 2], 2: [8, 2, 16, 4, 2]}
    else:
        epoch_floors = {1: [4, 8, 16, 4, 16], 2: [8, 16, 4, 8, 16]}
    policy = str(AUDIT.lifecycle_config(lifecycle)["policy"])
    for epoch, floors in epoch_floors.items():
        epoch_dir = run_root / f"epoch_{epoch:03d}_mode1_{policy}"
        (epoch_dir / "oracle").mkdir(parents=True)
        (epoch_dir / "logs").mkdir()
        history_dirs = [str(tmp_path / "history" / "epoch0")]
        if epoch > 1:
            history_dirs.append(str(tmp_path / "history" / "epoch1"))
        plans = [
            _plan(step, floor, history_dirs) for step, floor in enumerate(floors, 1)
        ]
        (epoch_dir / "oracle" / "length_sorted_rank_plan_summary.json").write_text(
            json.dumps(plans), encoding="utf-8"
        )
        (epoch_dir / "logs" / "runtime.txt").write_text(
            _runtime_log(floors, epoch), encoding="utf-8"
        )
    return run_root, cap_env


def _plans(run_root: Path, epoch: int) -> tuple[Path, list[dict]]:
    path = (
        run_root
        / f"epoch_{epoch:03d}_mode1_natural"
        / "oracle"
        / "length_sorted_rank_plan_summary.json"
    )
    return path, json.loads(path.read_text(encoding="utf-8"))


def _log_path(run_root: Path, epoch: int) -> Path:
    return run_root / f"epoch_{epoch:03d}_mode1_natural" / "logs" / "runtime.txt"


def test_audit_accepts_complete_plan_runtime_evidence(tmp_path: Path) -> None:
    run_root, cap_env = _make_run(tmp_path)

    result = AUDIT.audit_run(run_root, cap_env, (1, 2), 5)

    assert result["status"] == "PASS"
    assert [epoch["selected_floors"] for epoch in result["epochs"]] == [
        [4, 8, 16, 4, 16],
        [8, 16, 4, 8, 16],
    ]
    assert [epoch["resize_calls"] for epoch in result["epochs"]] == [80, 80]
    assert result["epochs"][0]["steps"][0]["transitions"][-1]["to"] == 4
    assert result["epochs"][0]["steps"][2]["transitions"] == []


def test_audit_accepts_natural_floor2_lifecycle(tmp_path: Path) -> None:
    run_root, cap_env = _make_run(tmp_path, "natural_f2")

    result = AUDIT.audit_run(
        run_root, cap_env, (1, 2), 5, lifecycle="natural_f2"
    )

    assert result["status"] == "PASS"
    assert result["lifecycle"] == "natural_f2"
    assert result["admission_caps"] == {
        "16": ADMISSION[16],
        "8": ADMISSION[8],
        "4": ADMISSION[4],
        "2": ADMISSION[2],
    }
    assert result["epochs"][0]["selected_floors"] == [2, 4, 8, 16, 2]
    assert [
        transition["to"]
        for transition in result["epochs"][0]["steps"][0]["transitions"]
    ] == [8, 4, 2]
    assert result["epochs"][0]["shrink_events"] == 96


@pytest.mark.parametrize("lifecycle", ["planned_f4", "planned_f2"])
def test_audit_accepts_planned_lifecycle(
    tmp_path: Path, lifecycle: str
) -> None:
    run_root, cap_env = _make_run(tmp_path, lifecycle)

    result = AUDIT.audit_run(
        run_root, cap_env, (1, 2), 5, lifecycle=lifecycle
    )

    assert result["status"] == "PASS"
    assert result["lifecycle"] == lifecycle
    assert all("_mode1_planned" in epoch["epoch_dir"] for epoch in result["epochs"])


def test_audit_rejects_cap_env_for_another_lifecycle(tmp_path: Path) -> None:
    run_root, cap_env = _make_run(tmp_path, "natural_f2")

    with pytest.raises(AUDIT.AuditError, match="DEEPSEEK_N_F4_KV_CAPS_VERIFIED"):
        AUDIT.audit_run(run_root, cap_env, (1, 2), 5)


def test_audit_rejects_runtime_profile_mismatch(tmp_path: Path) -> None:
    run_root, cap_env = _make_run(tmp_path, "natural_f2")
    cap_env.write_text(
        cap_env.read_text(encoding="utf-8").replace(
            "export DEEPSEEK_N_F2_RUNTIME_PROFILE_SHA256=",
            "export DEEPSEEK_N_F2_RUNTIME_PROFILE_SHA256=stale-",
        ),
        encoding="utf-8",
    )

    with pytest.raises(AUDIT.AuditError, match="RUNTIME_PROFILE_SHA256"):
        AUDIT.audit_run(
            run_root, cap_env, (1, 2), 5, lifecycle="natural_f2"
        )


def test_natural_floor4_rejects_floor2_plan(tmp_path: Path) -> None:
    run_root, cap_env = _make_run(tmp_path)
    path, plans = _plans(run_root, 1)
    plans[0]["selected_floor"] = 2
    path.write_text(json.dumps(plans), encoding="utf-8")

    with pytest.raises(AUDIT.AuditError, match="selected unsupported floor 2"):
        AUDIT.audit_run(run_root, cap_env, (1, 2), 5)


def test_audit_rejects_infeasible_plan(tmp_path: Path) -> None:
    run_root, cap_env = _make_run(tmp_path)
    path, plans = _plans(run_root, 1)
    plans[0]["feasible"] = False
    path.write_text(json.dumps(plans), encoding="utf-8")

    with pytest.raises(AUDIT.AuditError, match="step 1 plan is not feasible"):
        AUDIT.audit_run(run_root, cap_env, (1, 2), 5)


def test_audit_rejects_runtime_floor_mismatch(tmp_path: Path) -> None:
    run_root, cap_env = _make_run(tmp_path)
    log = _log_path(run_root, 1)
    text = log.read_text(encoding="utf-8").replace(
        "rank=0 step=1 epoch=1 target_floor=4 target_kv=400000",
        "rank=0 step=1 epoch=1 target_floor=8 target_kv=500096",
        2,
    )
    log.write_text(text, encoding="utf-8")

    with pytest.raises(AUDIT.AuditError, match="runtime target floor/KV"):
        AUDIT.audit_run(run_root, cap_env, (1, 2), 5)


def test_audit_rejects_resize_above_physical_cap(tmp_path: Path) -> None:
    run_root, cap_env = _make_run(tmp_path)
    log = _log_path(run_root, 1)
    text = log.read_text(encoding="utf-8").replace(
        "effective_target_tokens=400000 new_tokens=400000",
        "effective_target_tokens=400000 new_tokens=400128",
        1,
    )
    log.write_text(text, encoding="utf-8")

    with pytest.raises(AUDIT.AuditError, match="must all equal physical cap"):
        AUDIT.audit_run(run_root, cap_env, (1, 2), 5)


def test_audit_rejects_inconsistent_natural_shrink_target(tmp_path: Path) -> None:
    run_root, cap_env = _make_run(tmp_path)
    log = _log_path(run_root, 1)
    text = log.read_text(encoding="utf-8").replace(
        "target_global=[8, 9, 10, 11, 12, 13, 14, 15]",
        "target_global=[0, 1, 2, 3, 4, 5, 6, 7]",
        1,
    )
    log.write_text(text, encoding="utf-8")

    with pytest.raises(AUDIT.AuditError, match="not a safe plan prefix"):
        AUDIT.audit_run(run_root, cap_env, (1, 2), 5)


def test_audit_accepts_runtime_selected_natural_survivor_identity(
    tmp_path: Path,
) -> None:
    del tmp_path
    full = tuple(range(16))
    runtime_target = tuple(range(8))
    shrinks = [
        AUDIT.ShrinkEvent(rank, 1, full, runtime_target, 10 + rank)
        for rank in full
    ]
    restores = [
        AUDIT.RestoreEvent(rank, 1, 1, 16, 100 + rank)
        for rank in full
    ]
    plans = {
        1: {
            "floor": 8,
            "stages": (8,),
            "stage_sets": (tuple(range(8, 16)),),
        }
    }

    result = AUDIT._validate_lifecycle(
        shrinks,
        restores,
        plans,
        {1: 0},
        rank_identity_known=False,
    )

    assert result[1]["transitions"][0]["survivor_ranks"] == list(range(8))


@pytest.mark.parametrize(
    ("prefix_length", "executed_floor"),
    [(0, 16), (1, 8), (2, 4), (3, 2)],
)
def test_natural_runtime_accepts_safe_stage_prefix(
    prefix_length: int,
    executed_floor: int,
) -> None:
    full = tuple(range(16))
    stage_sets = tuple(tuple(values) for values in STAGE_SETS[2])
    current = full
    shrinks = []
    position = 10
    for target in stage_sets[:prefix_length]:
        shrinks.extend(
            AUDIT.ShrinkEvent(rank, 1, current, target, position + rank)
            for rank in current
        )
        position += 20
        current = target
    restores = []
    if prefix_length:
        restores = [
            AUDIT.RestoreEvent(rank, 1, 1, 16, position + rank)
            for rank in full
        ]
    plans = {1: {"floor": 2, "stages": (8, 4, 2), "stage_sets": stage_sets}}

    result = AUDIT._validate_lifecycle(
        shrinks,
        restores,
        plans,
        {1: 0},
        rank_identity_known=False,
    )

    assert result[1]["executed_floor"] == executed_floor
    assert result[1]["executed_transition_count"] == prefix_length
    assert result[1]["runtime_stages_are_safe_prefix"] is True


def test_natural_runtime_rejects_stage_skip() -> None:
    full = tuple(range(16))
    floor8 = tuple(STAGE_SETS[2][0])
    floor2 = tuple(STAGE_SETS[2][2])
    shrinks = [
        AUDIT.ShrinkEvent(rank, 1, full, floor8, 10 + rank)
        for rank in full
    ] + [
        AUDIT.ShrinkEvent(rank, 1, floor8, floor2, 40 + rank)
        for rank in floor8
    ]
    restores = [
        AUDIT.RestoreEvent(rank, 1, 1, 16, 80 + rank) for rank in full
    ]
    plans = {
        1: {
            "floor": 2,
            "stages": (8, 4, 2),
            "stage_sets": tuple(tuple(values) for values in STAGE_SETS[2]),
        }
    }

    with pytest.raises(AUDIT.AuditError, match="not a safe plan prefix"):
        AUDIT._validate_lifecycle(
            shrinks,
            restores,
            plans,
            {1: 0},
            rank_identity_known=False,
        )


def _pre_resume_materialized_log(include_prefix_reset: bool = True) -> str:
    messages = [
        "Mode1 pre-resume KV cleanup phase=start target_floor=8 "
        "previous_floor=2 target_policy=natural",
        "Mode1 pre-resume KV cleanup phase=floor_prepare_done target_floor=8",
        "Mode1 natural KV resize runtime prune summary: rank=0 "
        "target_floor=8 changed=1",
        "Mode1 pre-resume KV cleanup phase=natural_runtime_prune_done "
        "target_floor=8",
        "Mode1 pre-resume KV cleanup phase=done target_floor=8 previous_floor=2",
    ]
    if include_prefix_reset:
        messages.append("Successfully reset prefix cache")
    messages.extend(
        [
            "rollout_mode_after_resume_kv_cache rank=0 elapsed_s=1.0",
            "rollout_worker_after_env_sync rank=0 step=5 epoch=0 total_s=0.0",
            "rollout_worker_resize_start rank=0 step=5 epoch=0 "
            "target_floor=8 target_kv=500096",
            "rollout_worker_resize_done rank=0 step=5 epoch=0",
        ]
    )
    return "".join(_worker(1234, message) for message in messages)


def test_runtime_parser_accepts_pre_resume_materialized_resize(
    tmp_path: Path,
) -> None:
    log = tmp_path / "runtime.txt"
    log.write_text(_pre_resume_materialized_log(), encoding="utf-8")

    calls, _shrinks, _restores, _text = AUDIT.parse_runtime_log(log)

    assert len(calls) == 1
    assert calls[0].outcome == "pre_resume_materialized"
    assert calls[0].outcome_values == (8, 2)


def test_runtime_parser_rejects_incomplete_pre_resume_evidence(
    tmp_path: Path,
) -> None:
    log = tmp_path / "runtime.txt"
    log.write_text(
        _pre_resume_materialized_log(include_prefix_reset=False),
        encoding="utf-8",
    )

    with pytest.raises(AUDIT.AuditError, match="complete pre-resume"):
        AUDIT.parse_runtime_log(log)


def test_audit_rejects_incomplete_full_world_restore(tmp_path: Path) -> None:
    run_root, cap_env = _make_run(tmp_path)
    log = _log_path(run_root, 1)
    text = log.read_text(encoding="utf-8").replace(
        "(WorkerDict pid=10015) Elastic full-world restore segmented timing: "
        "rank=15 restore_seq=101 world_size=16\n",
        "",
        1,
    )
    log.write_text(text, encoding="utf-8")

    with pytest.raises(AUDIT.AuditError, match="full-world restore has ranks"):
        AUDIT.audit_run(run_root, cap_env, (1, 2), 5)


def test_audit_rejects_preemption_and_unexpected_epoch(tmp_path: Path) -> None:
    run_root, cap_env = _make_run(tmp_path)
    log = _log_path(run_root, 1)
    log.write_text(log.read_text(encoding="utf-8") + "Preempting request due to KV pressure\n")

    with pytest.raises(AUDIT.AuditError, match="preemption marker"):
        AUDIT.audit_run(run_root, cap_env, (1, 2), 5)

    with pytest.raises(AUDIT.AuditError, match="epoch directory set mismatch"):
        AUDIT.audit_run(run_root, cap_env, (1,), 5)
