from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd
import pytest


ROOT = Path(__file__).parents[1]
TOOL = ROOT / "tools" / "verify_qwen3_sidecar_pairs.py"
SPEC = importlib.util.spec_from_file_location("verify_qwen3_sidecar_pairs", TOOL)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _dataset(root: Path) -> Path:
    path = root / "gsm8k" / "train.parquet"
    path.parent.mkdir(parents=True)
    frame = pd.DataFrame(
        {
            "prompt": [
                [{"role": "user", "content": f"Question {index}"}]
                for index in range(3)
            ],
            "reward_model": [
                {"style": "rule", "ground_truth": str(index)}
                for index in range(3)
            ],
        }
    )
    frame.to_parquet(path)
    return path


def _primary(
    arm_dir: Path,
    seed: int,
    arm: str,
    *,
    length_shift: bool = False,
) -> None:
    epoch = (
        arm_dir
        / f"primary_planned_f4_noguard_seed{seed}_{arm}"
        / "epoch_001_mode1_planned"
    )
    log = epoch / "logs" / "train.txt"
    log.parent.mkdir(parents=True)
    rollout_time = 100.0 + (seed % 10) + (2.0 if arm == "on" else 0.0)
    log.write_text(
        "\n".join(
            [
                "TransformerConfig(moe_shared_expert_intermediate_size=None, "
                "moe_shared_expert_overlap=False, n_shared_experts=None)",
                "Elastic parallel shrink done: rank=8 active_ranks=[8, 9, 10, 11, 12, 13, 14, 15]",
                "Elastic parallel shrink done: rank=12 active_ranks=[12, 13, 14, 15]",
                "Shrink-aware tail-guard response cap: selected_floor=4 plan_cap=16384 "
                "ratio=None ratio_q=None predicted_step_exit=None sampling_override=False",
                "Mode1 step timeline: driver_generate_done step=1 epoch=0 elapsed_s=9.000 "
                "driver_generate_done_time=12.000000000",
                *(
                    [
                        "Sidecar pre-restore release acknowledged: request_id=request-1 "
                        "lease_id=1 elapsed_s=7.000"
                    ]
                    if arm == "on"
                    else []
                ),
                "Mode1 step timeline: driver_restore_rpc_start "
                "driver_restore_rpc_start_time=21.000000000",
                f"=> rollout_output_time_s: {rollout_time}",
                "step:1 - response/aborted_ratio:0.0 - training/global_step:1",
                "Elastic full-world restore segmented timing: rank=0 restore_seq=1 world_size=16",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    rows: list[dict[str, object]] = []
    lengths: list[int] = []
    for index in range(MODULE.EXPECTED_RESPONSES):
        occurrence = index // 16
        sample_index = index % 16
        length = 1 + index % 3
        if length_shift and index == 0:
            length += 1
        lengths.append(length)
        rows.append(
            {
                "request_id": str(index % 32),
                "input": f"seed-{seed}-question-{occurrence}",
                "gts": str(occurrence),
                "prompt_occurrence_ordinal": occurrence,
                "rollout_sample_index": sample_index,
                "rollout_request_seed": seed * 1_000_000 + occurrence * 16 + sample_index,
                "rollout_prompt_hash": hashlib.blake2b(
                    f"seed-{seed}-question-{occurrence}".encode("utf-8"),
                    digest_size=16,
                ).hexdigest(),
                "rollout_rank": occurrence // 2,
                "response_mask": [1] * length + [0] * (4 - length),
                "responses": [occurrence * 100 + sample_index] * 4,
                "output": f"answer-{occurrence}-{sample_index}",
                "step": 1,
            }
        )
    _write_jsonl(epoch / "rollout_data" / "1.jsonl", rows)
    length_file = epoch / "rollout_length" / "length_1.txt"
    length_file.parent.mkdir(parents=True)
    length_file.write_text("".join(f"{value}\n" for value in lengths), encoding="utf-8")


def _sidecar(arm_dir: Path, dataset: Path, *, overlap: bool = False) -> None:
    epoch = next(arm_dir.rglob("epoch_001_mode1_planned"))
    sidecar = epoch / "sidecar"
    sidecar.mkdir(parents=True)
    sidecar.joinpath("lease.log").write_text(
        "\n".join(
            [
                "watch_expected_active_ranks=8",
                "watch_world_size=16",
                "watch_require_active_lease=1",
                "watch_require_shrink_quorum=1",
                "watch_shrink_quorum_size=16",
                *[
                    "shrink_quorum_progress_time="
                    f"{9.0 + rank * 0.01:.9f} "
                    "active_ranks=8,9,10,11,12,13,14,15 "
                    f"reporter={rank} quorum_count={rank + 1} quorum_required=16"
                    for rank in range(16)
                ],
                "quorum_ranks=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 "
                "active_ranks=8,9,10,11,12,13,14,15",
                "quorum_count=16 quorum_required=16 "
                "active_ranks=8,9,10,11,12,13,14,15",
                "coordinated_start_time=10.000000000 "
                "active_ranks=8,9,10,11,12,13,14,15 "
                "quorum_ranks=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 "
                "quorum_count=16",
                "shrink_window_detected_time=10.0 active_count=8 coordinated=1",
                "shrink_window_line=Elastic parallel shrink done: rank=8 "
                "active_ranks=[8, 9, 10, 11, 12, 13, 14, 15]",
                "sidecar_start_time=10.1",
                "sidecar_active_ranks=8,9,10,11,12,13,14,15",
                "sidecar_devices=0,1,2,3,4,5,6,7",
                "sidecar_deadline_signal_deferred_time=12.500000000 "
                "reason=trainer_stop_request_is_authoritative",
                "trainer_stop_request_time=13.000000000 watcher_observed_time=13.1 "
                "request_id=request-1 lease_id=1",
                "sidecar_stop_request_time=13.2 reason=pre_restore_handshake pid=123",
                f"sidecar_artifacts_durable_time=19.700000000 "
                f"output={sidecar / 'outputs.jsonl'} state_dir={sidecar / 'state'}",
                "sidecar_exit_confirmed_time=19.800000000 reason=pre_restore_handshake "
                "pid=123 process_group_alive=0 request_id=request-1 lease_id=1",
                "watcher_restore_ack_time=19.900000000 request_id=request-1 "
                "lease_id=1 status=released",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    used = "0;1;2;3;4;5;6;12" if overlap else "0;1;2;3;4;5;6;7"
    unused = "7" if overlap else ""
    sidecar.joinpath("infer.log").write_text(
        "\n".join(
            [
                f"sidecar_device_groups={used}",
                f"sidecar_unused_devices={unused}",
                "sidecar_model=/data/Qwen2.5-1.5B-Instruct",
                json.dumps(
                    {
                        "event": "sidecar_sampling_params",
                        "n": 1,
                        "temperature": 0.0,
                        "top_p": 1.0,
                        "max_tokens": 4096,
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    outputs: list[dict[str, object]] = []
    completed: list[dict[str, object]] = []
    for index in range(3):
        text = f"reasoning #### {index if index < 2 else 999}"
        outputs.append(
            {
                "sidecar_epoch": 0,
                "iteration": 0,
                "chunk_start": 0,
                "prompt_id": index,
                "prompt_source": f"{dataset}:{index}",
                "prompt": f"user: Question {index}",
                "outputs": [
                    {
                        "text": text,
                        "delta_text": text,
                        "token_ids_len": 5 + index,
                        "resume_prefix_text_len": 0,
                        "finish_reason": "stop",
                    }
                ],
            }
        )
        completed.append(
            {
                "time": 13.5 if index == 2 else 10.5 + 0.5 * index,
                "sidecar_epoch": 0,
                "prompt_id": index,
                "output_file": str(sidecar / "outputs.jsonl"),
            }
        )
    _write_jsonl(sidecar / "outputs.jsonl", outputs)
    state = sidecar / "state"
    _write_jsonl(state / "completed.shard0.jsonl", completed)
    _write_jsonl(
        state / "resume.shard0.jsonl",
        [
            {
                "time": 19.5,
                "sidecar_epoch": 0,
                "prompt_id": 10,
                "partial_text": "reasoning",
                "token_ids_len": 3,
            }
        ],
    )
    handshake = sidecar / "restore_handshake"
    handshake.mkdir()
    handshake.joinpath("active_lease").write_text(
        "lease_id=1\n"
        "state=released\n"
        "pid=123\n"
        "devices=0,1,2,3,4,5,6,7\n"
        "update_time=19.850000000\n",
        encoding="utf-8",
    )
    handshake.joinpath("stop_request").write_text(
        "request_id=request-1\n"
        "lease_id=1\n"
        "observed_state=running\n"
        "request_time=13.000000000\n",
        encoding="utf-8",
    )
    handshake.joinpath("stop_ack").write_text(
        "request_id=request-1\n"
        "lease_id=1\n"
        "status=released\n"
        "reason=sidecar_process_group_exited\n"
        "request_time=13.000000000\n"
        "ack_time=19.900000000\n",
        encoding="utf-8",
    )


def _arm(
    root: Path,
    dataset: Path,
    seed: int,
    arm: str,
    *,
    length_shift: bool = False,
    overlap: bool = False,
) -> None:
    arm_dir = root / f"seed_{seed}" / arm
    arm_dir.mkdir(parents=True)
    plan = arm_dir / "step1_plan.json"
    plan_rows = []
    for step in range(1, 6):
        plan_rows.append(
            {
                "step": step,
                "selected_floor": 4 if step == 1 else 8,
                "feasible": True,
                "tail_guard_enabled": False,
                "intermediate_survivor_ranks": [8, 9, 10, 11, 12, 13, 14, 15],
                "final_survivor_ranks": [12, 13, 14, 15] if step == 1 else [8, 9, 10, 11, 12, 13, 14, 15],
                "shrink_stages": [8, 4] if step == 1 else [8],
                "stage_survivor_ranks": (
                    [
                        [8, 9, 10, 11, 12, 13, 14, 15],
                        [12, 13, 14, 15],
                    ]
                    if step == 1
                    else [[8, 9, 10, 11, 12, 13, 14, 15]]
                ),
                "rank_to_source_idx": {
                    str(rank): [
                        (step - 1) * 32 + rank * 2,
                        (step - 1) * 32 + rank * 2 + 1,
                    ]
                    for rank in range(16)
                },
            }
        )
    _write_json(plan, plan_rows)
    _write_json(
        arm_dir / MODULE.MANIFEST_NAME,
        {
            "schema_version": 1,
            "experiment": "qwen2_5_1_5b_planned_floor4_noguard_sidecar_pair",
            "seed": seed,
            "arm": arm,
            "launch_order": MODULE.EXPECTED_LAUNCH_ORDER[(seed, arm)],
            "planned": True,
            "planned_residency": True,
            "target_floor": 4,
            "tail_guard_enabled": False,
            "expected_responses": 512,
            "request_seed": seed,
            "planner_prompts": 160,
            "planner_steps": 5,
            "executed_steps": 1,
            "source_plan_step": 1,
            "fast_step_subset": False,
            "plan_file": "step1_plan.json",
            "plan_sha256": _sha256(plan),
            "sidecar_enabled": arm == "on",
            "sidecar_tensor_parallel_size": 1,
            "sidecar_replica_count": 8,
            "sidecar_trigger_active_ranks": 8,
            "sidecar_temperature": 0.0,
            "sidecar_top_p": 1.0,
            "sidecar_max_tokens": 4096,
            "sidecar_stop_ack_timeout_seconds": 60,
            "sidecar_require_active_lease_before_restore": True,
            "sidecar_require_shrink_quorum": True,
            "sidecar_shrink_quorum_size": 16,
            "eager_weight_sync_group_init": False,
            "primary_hccl_allocator_start": MODULE.PRIMARY_HCCL_ALLOCATOR_START,
            "primary_moe_shared_expert_overlap": False,
            "primary_prompts": 32,
            "responses_per_prompt": 16,
            "actor_frozen": True,
            "paired_request_sampling_seeds": True,
            "sidecar_model_path": "/data/Qwen2.5-1.5B-Instruct",
            "sidecar_model_revision": "a3c2dc17129625b1e51caf21ab486d32d1f12982",
            "sidecar_model_weights_sha256": MODULE.SIDECAR_MODEL_WEIGHTS_SHA256,
            "sidecar_dataset_path": str(dataset),
            "sidecar_dataset_split": "train",
            "run_dir": str(
                (
                    arm_dir
                    / f"primary_planned_f4_noguard_seed{seed}_{arm}"
                ).resolve()
            ),
        },
    )
    _primary(arm_dir, seed, arm, length_shift=length_shift)
    if arm == "on":
        _sidecar(arm_dir, dataset, overlap=overlap)


def _experiment(
    root: Path,
    *,
    seeds: tuple[int, ...] = MODULE.SEEDS,
    shifted_seed: int | None = None,
    overlap_seed: int | None = None,
) -> Path:
    dataset = _dataset(root)
    for seed in seeds:
        _arm(root, dataset, seed, "off")
        _arm(
            root,
            dataset,
            seed,
            "on",
            length_shift=seed == shifted_seed,
            overlap=seed == overlap_seed,
        )
    (root / MODULE.CODE_PROVENANCE_NAME).write_text(
        f"{_sha256(TOOL)}  {TOOL.resolve()}\n",
        encoding="utf-8",
    )
    (root / MODULE.PROTOCOL_NAME).write_text(
        "created_at_utc=2026-08-09T00:00:00Z\n"
        "common_epoch0_root=/data/common_epoch0\n"
        "seeds=101 202 303\n"
        "orders=101:off,on 202:on,off 303:off,on\n"
        "planned_residency=true\n"
        "floor=floor4\n"
        "tail_guard=false\n"
        "actor_frozen=true\n"
        "paired_request_sampling_seeds=true\n"
        "planner_prompts=160\n"
        "plan_steps=5\n"
        "executed_prompts=32\n"
        "steps_per_run=1\n"
        "fast_step_subset=false\n"
        "source_plan_step=1\n"
        "sidecar_trigger_active_ranks=8\n"
        "sidecar_model=/data/Qwen2.5-1.5B-Instruct\n"
        "sidecar_model_revision=a3c2dc17129625b1e51caf21ab486d32d1f12982\n"
        f"sidecar_data={dataset}\n"
        "sidecar_parallelism=TP1x8\n"
        "sidecar_stop_ack_timeout_seconds=60\n"
        "sidecar_require_active_lease_before_restore=true\n"
        "sidecar_require_shrink_quorum=true\n"
        "sidecar_shrink_quorum_size=16\n"
        "eager_weight_sync_group_init=false\n"
        f"primary_hccl_allocator_start={MODULE.PRIMARY_HCCL_ALLOCATOR_START}\n"
        "primary_moe_shared_expert_overlap=false\n",
        encoding="utf-8",
    )
    ordered_keys = sorted(MODULE.EXPECTED_LAUNCH_ORDER, key=MODULE.EXPECTED_LAUNCH_ORDER.get)
    manifest_lines = [MODULE.RUN_MANIFEST_HEADER]
    for seed, arm in ordered_keys:
        if seed not in seeds:
            continue
        run_dir = (
            root
            / f"seed_{seed}"
            / arm
            / f"primary_planned_f4_noguard_seed{seed}_{arm}"
        ).resolve()
        manifest_lines.append(
            f"{seed}\t{arm}\t{MODULE.EXPECTED_LAUNCH_ORDER[(seed, arm)]}\t"
            f"{run_dir}\tcomplete"
        )
    (root / MODULE.RUN_MANIFEST_NAME).write_text(
        "\n".join(manifest_lines) + "\n", encoding="utf-8"
    )
    return root


def test_complete_three_pair_summary_and_outputs(tmp_path: Path) -> None:
    root = _experiment(tmp_path / "pairs")

    assert MODULE.main(["--root", str(root)]) == 0

    summary = json.loads((root / MODULE.SUMMARY_JSON).read_text(encoding="utf-8"))
    assert summary["status"] == "PASS"
    assert summary["completed_pairs"] == 3
    assert summary["aggregate"]["sidecar_completed_queries"] == 9
    assert summary["aggregate"]["sidecar_completed_before_trainer_stop"] == 6
    assert summary["aggregate"]["sidecar_completed_during_shutdown"] == 3
    assert summary["aggregate"]["sidecar_partial_queries_at_restore"] == 3
    assert summary["aggregate"]["sidecar_strict_accuracy"] == pytest.approx(2 / 3)
    assert summary["aggregate"]["sidecar_strict_accuracy_before_trainer_stop"] == 1.0
    delta = summary["aggregate"]["primary_throughput_delta_percent"]
    assert delta["n"] == 3
    assert delta["t95_df"] == 2
    assert len(delta["ci95"]) == 2
    assert (root / MODULE.SUMMARY_MD).is_file()


def test_rejects_generated_length_mismatch(tmp_path: Path) -> None:
    root = _experiment(tmp_path / "pairs", shifted_seed=202)

    with pytest.raises(MODULE.VerificationError, match="generated response lengths differ"):
        MODULE.verify(root)


def test_rejects_duplicate_occurrence_sample_identity(tmp_path: Path) -> None:
    root = _experiment(tmp_path / "pairs")
    rollout = next((root / "seed_101" / "on").rglob("rollout_data/1.jsonl"))
    rows = MODULE._read_jsonl(rollout, "fixture rollout")
    rows[1]["prompt_occurrence_ordinal"] = rows[0]["prompt_occurrence_ordinal"]
    rows[1]["rollout_sample_index"] = rows[0]["rollout_sample_index"]
    _write_jsonl(rollout, rows)

    with pytest.raises(MODULE.VerificationError, match="duplicate primary request identity"):
        MODULE.verify(root)


def test_rejects_paired_response_token_mismatch(tmp_path: Path) -> None:
    root = _experiment(tmp_path / "pairs")
    rollout = next((root / "seed_101" / "on").rglob("rollout_data/1.jsonl"))
    rows = MODULE._read_jsonl(rollout, "fixture rollout")
    rows[0]["responses"][0] += 1
    _write_jsonl(rollout, rows)

    with pytest.raises(MODULE.VerificationError, match="generated response token IDs differ"):
        MODULE.verify(root)


def test_rejects_plan_without_unique_ordered_step_one(tmp_path: Path) -> None:
    root = _experiment(tmp_path / "pairs")
    arm = root / "seed_101" / "on"
    plan = arm / "step1_plan.json"
    payload = json.loads(plan.read_text(encoding="utf-8"))
    payload[0]["step"] = 2
    _write_json(plan, payload)
    manifest_path = arm / MODULE.MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["plan_sha256"] = _sha256(plan)
    _write_json(manifest_path, manifest)

    with pytest.raises(MODULE.VerificationError, match="planner steps"):
        MODULE.verify(root)


def test_rejects_tailguard_cap_in_primary_log(tmp_path: Path) -> None:
    root = _experiment(tmp_path / "pairs")
    log = next((root / "seed_101" / "on").rglob("logs/train.txt"))
    with log.open("a", encoding="utf-8") as handle:
        handle.write(
            "Shrink-aware tail-guard response cap: selected_floor=4 plan_cap=4096 "
            "sampling_override=True\n"
        )

    with pytest.raises(MODULE.VerificationError, match="TailGuard"):
        MODULE.verify(root)


def test_rejects_shared_expert_overlap_in_primary_runtime(tmp_path: Path) -> None:
    root = _experiment(tmp_path / "pairs")
    log = next((root / "seed_101" / "off").rglob("logs/train.txt"))
    text = log.read_text(encoding="utf-8").replace(
        "moe_shared_expert_overlap=False",
        "moe_shared_expert_overlap=True",
    )
    log.write_text(text, encoding="utf-8")

    with pytest.raises(
        MODULE.VerificationError,
        match="does not prove moe_shared_expert_overlap=False",
    ):
        MODULE.verify(root)


def test_rejects_sidecar_survivor_device_overlap(tmp_path: Path) -> None:
    root = _experiment(tmp_path / "pairs", overlap_seed=303)

    with pytest.raises(MODULE.VerificationError, match="primary survivor rank"):
        MODULE.verify(root)


def test_rejects_sidecar_without_strict_shrink_quorum(tmp_path: Path) -> None:
    root = _experiment(tmp_path / "pairs")
    lease = next((root / "seed_101" / "on").rglob("sidecar/lease.log"))
    text = lease.read_text(encoding="utf-8").replace(
        "watch_require_shrink_quorum=1", "watch_require_shrink_quorum=0"
    )
    lease.write_text(text, encoding="utf-8")

    with pytest.raises(MODULE.VerificationError, match="strict shrink quorum"):
        MODULE.verify(root)


def test_rejects_incomplete_coordinated_quorum_ranks(tmp_path: Path) -> None:
    root = _experiment(tmp_path / "pairs")
    lease = next((root / "seed_101" / "on").rglob("sidecar/lease.log"))
    full = "quorum_ranks=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 "
    incomplete = "quorum_ranks=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14 "
    text = lease.read_text(encoding="utf-8").replace(full, incomplete, 1)
    lease.write_text(text, encoding="utf-8")

    with pytest.raises(MODULE.VerificationError, match="final quorum ranks"):
        MODULE.verify(root)


def test_rejects_sidecar_start_before_coordinated_quorum(tmp_path: Path) -> None:
    root = _experiment(tmp_path / "pairs")
    lease = next((root / "seed_101" / "on").rglob("sidecar/lease.log"))
    text = lease.read_text(encoding="utf-8").replace(
        "coordinated_start_time=10.000000000",
        "coordinated_start_time=10.200000000",
    ).replace(
        "shrink_window_detected_time=10.0 active_count=8 coordinated=1",
        "shrink_window_detected_time=10.2 active_count=8 coordinated=1",
    )
    lease.write_text(text, encoding="utf-8")

    with pytest.raises(MODULE.VerificationError, match="started before"):
        MODULE.verify(root)


@pytest.mark.parametrize(
    "replacement",
    [
        "shrink_window_detected_time=10.0 active_count=7 coordinated=1",
        "shrink_window_detected_time=10.0 active_count=8 coordinated=0",
    ],
)
def test_rejects_noncoordinated_shrink_window_event(
    tmp_path: Path, replacement: str
) -> None:
    root = _experiment(tmp_path / "pairs")
    lease = next((root / "seed_101" / "on").rglob("sidecar/lease.log"))
    text = lease.read_text(encoding="utf-8").replace(
        "shrink_window_detected_time=10.0 active_count=8 coordinated=1",
        replacement,
    )
    lease.write_text(text, encoding="utf-8")

    with pytest.raises(MODULE.VerificationError, match="shrink-window event"):
        MODULE.verify(root)


def test_rejects_duplicate_shrink_window_event(tmp_path: Path) -> None:
    root = _experiment(tmp_path / "pairs")
    lease = next((root / "seed_101" / "on").rglob("sidecar/lease.log"))
    with lease.open("a", encoding="utf-8") as handle:
        handle.write(
            "shrink_window_detected_time=10.0 active_count=8 coordinated=1\n"
        )

    with pytest.raises(MODULE.VerificationError, match="found 2"):
        MODULE.verify(root)


def test_rejects_shrink_window_timestamp_before_quorum(tmp_path: Path) -> None:
    root = _experiment(tmp_path / "pairs")
    lease = next((root / "seed_101" / "on").rglob("sidecar/lease.log"))
    text = lease.read_text(encoding="utf-8").replace(
        "shrink_window_detected_time=10.0 active_count=8 coordinated=1",
        "shrink_window_detected_time=9.9 active_count=8 coordinated=1",
    )
    lease.write_text(text, encoding="utf-8")

    with pytest.raises(MODULE.VerificationError, match="quorum timestamp"):
        MODULE.verify(root)


def test_allow_incomplete_reports_one_pair_but_final_rejects(tmp_path: Path) -> None:
    root = _experiment(tmp_path / "pairs", seeds=(101,))

    assert MODULE.main(["--root", str(root), "--allow-incomplete"]) == 0
    summary = json.loads((root / MODULE.SUMMARY_JSON).read_text(encoding="utf-8"))
    assert summary["status"] == "INCOMPLETE"
    assert summary["completed_pairs"] == 1
    assert [item["seed"] for item in summary["pending"]] == [202, 303]
    assert summary["aggregate"]["primary_throughput_delta_percent"]["ci95"] is None
    with pytest.raises(MODULE.VerificationError, match="exactly six rows"):
        MODULE.verify(root)


@pytest.mark.parametrize(
    ("name", "message"),
    [
        (MODULE.CODE_PROVENANCE_NAME, "missing code provenance"),
        (MODULE.PROTOCOL_NAME, "missing root protocol"),
        (MODULE.RUN_MANIFEST_NAME, "missing run manifest"),
    ],
)
def test_requires_root_provenance_files(
    tmp_path: Path, name: str, message: str
) -> None:
    root = _experiment(tmp_path / "pairs", seeds=(101,))
    (root / name).unlink()

    with pytest.raises(MODULE.VerificationError, match=message):
        MODULE.verify(root, allow_incomplete=True)


def test_rejects_run_manifest_out_of_fixed_order(tmp_path: Path) -> None:
    root = _experiment(tmp_path / "pairs", seeds=(101,))
    path = root / MODULE.RUN_MANIFEST_NAME
    lines = path.read_text(encoding="utf-8").splitlines()
    path.write_text("\n".join([lines[0], lines[2], lines[1]]) + "\n", encoding="utf-8")

    with pytest.raises(MODULE.VerificationError, match="fixed launch-order prefix"):
        MODULE.verify(root, allow_incomplete=True)


def test_rejects_noncomplete_run_manifest_status(tmp_path: Path) -> None:
    root = _experiment(tmp_path / "pairs", seeds=(101,))
    path = root / MODULE.RUN_MANIFEST_NAME
    text = path.read_text(encoding="utf-8").replace("\tcomplete\n", "\trunning\n", 1)
    path.write_text(text, encoding="utf-8")

    with pytest.raises(MODULE.VerificationError, match="non-complete status"):
        MODULE.verify(root, allow_incomplete=True)


def test_rejects_duplicate_run_manifest_row(tmp_path: Path) -> None:
    root = _experiment(tmp_path / "pairs", seeds=(101,))
    path = root / MODULE.RUN_MANIFEST_NAME
    lines = path.read_text(encoding="utf-8").splitlines()
    path.write_text("\n".join([*lines, lines[1]]) + "\n", encoding="utf-8")

    with pytest.raises(MODULE.VerificationError, match="repeats seed/arm"):
        MODULE.verify(root, allow_incomplete=True)


def test_rejects_root_and_arm_run_dir_mismatch(tmp_path: Path) -> None:
    root = _experiment(tmp_path / "pairs", seeds=(101,))
    path = root / MODULE.RUN_MANIFEST_NAME
    lines = path.read_text(encoding="utf-8").splitlines()
    columns = lines[1].split("\t")
    columns[3] += "_wrong"
    lines[1] = "\t".join(columns)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    with pytest.raises(MODULE.VerificationError, match="root and arm run_dir"):
        MODULE.verify(root, allow_incomplete=True)


def test_rejects_root_protocol_mismatch(tmp_path: Path) -> None:
    root = _experiment(tmp_path / "pairs", seeds=(101,))
    path = root / MODULE.PROTOCOL_NAME
    text = path.read_text(encoding="utf-8").replace("floor=floor4", "floor=floor2")
    path.write_text(text, encoding="utf-8")

    with pytest.raises(MODULE.VerificationError, match="floor='floor4'"):
        MODULE.verify(root, allow_incomplete=True)
