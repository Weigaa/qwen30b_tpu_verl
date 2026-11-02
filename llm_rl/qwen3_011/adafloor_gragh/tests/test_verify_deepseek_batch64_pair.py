from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

from tools.hash_deepseek_checkpoint import digest as checkpoint_digest


ROOT = Path(__file__).parents[1]
TOOL = ROOT / "tools" / "verify_deepseek_batch64_pair.py"
SPEC = importlib.util.spec_from_file_location("verify_deepseek_batch64_pair", TOOL)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

PHYSICAL = {16: 614144, 8: 500096, 4: 400000, 2: 300032}
ADMISSION = {16: 596736, 8: 490112, 4: 390016, 2: 290048}
EXECUTION_SHA256 = "d" * 64
TEST_BOS_TOKEN_ID = 100000


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_env(path: Path, values: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(f"export {key}={value}\n" for key, value in values.items()),
        encoding="utf-8",
    )


def _write_test_tokenizer(model: Path) -> None:
    (model / "config.json").write_text(
        json.dumps({"bos_token_id": TEST_BOS_TOKEN_ID}) + "\n",
        encoding="utf-8",
    )
    (model / "tokenizer_config.json").write_text(
        json.dumps({"bos_token": {"content": "BOS"}}) + "\n",
        encoding="utf-8",
    )
    (model / "tokenizer.json").write_text(
        json.dumps(
            {
                "added_tokens": [
                    {"id": TEST_BOS_TOKEN_ID, "content": "BOS"}
                ]
            }
        )
        + "\n",
        encoding="utf-8",
    )


def _profile(root: Path) -> Path:
    path = root / "batch64_workload_profile.env"
    _write_env(
        path,
        {
            "DEEPSEEK_WORKLOAD_PROFILE_ID": "deepseek-v2-lite-b64-n16-s5-test",
            "COMMON_EPOCH0_TRAIN_STEPS": 5,
            "COMMON_EPOCH0_TRAIN_BATCH_SIZE": 64,
            "COMMON_EPOCH0_ROLLOUT_N": 16,
            "COMMON_EPOCH0_MAX_NUM_SEQS": 64,
            "COMMON_EPOCH0_DATASET_FRACTION": 0.01,
            "DEEPSEEK_KV_PROBE_DATASET_FRACTION": 0.001765,
            "COMMON_EPOCH0_PROMPTS_TOTAL": 320,
            "COMMON_EPOCH0_EXPECTED_RESPONSES_PER_STEP": 1024,
            "COMMON_EPOCH0_MAX_PROMPT_LENGTH": 1024,
            "COMMON_EPOCH0_MAX_RESPONSE_LENGTH": 16384,
            "COMMON_EPOCH0_MAX_NUM_BATCHED_TOKENS": 17408,
            "COMMON_EPOCH0_PREEMPTION_POLICY": "record",
        },
    )
    return path


def _common(root: Path, profile: Path) -> tuple[Path, Path, Path]:
    common = root / "common"
    checkpoint = common / "global_step_5"
    (checkpoint / "actor").mkdir(parents=True)
    distcp = checkpoint / "actor" / "dist_ckpt"
    distcp.mkdir()
    (distcp / ".metadata").write_bytes(b"metadata")
    (distcp / "__0_0.distcp").write_bytes(b"checkpoint")
    (checkpoint / ".PRESERVE_COMMON_EPOCH0").touch()
    (common / "DO_NOT_DELETE_COMMON_EPOCH0_CHECKPOINT").touch()
    model = root / "model"
    model.mkdir()
    _write_test_tokenizer(model)
    _write_env(
        common / "reuse.env",
        {
            "BASELINE_INITIAL_RESUME_CKPT": checkpoint,
            "DYNAMIC_INITIAL_RESUME_CKPT": checkpoint,
        },
    )
    _write_env(
        common / "common_epoch0_metadata.env",
        {
            "COMMON_EPOCH0_MODEL_PATH": model,
            "COMMON_EPOCH0_MODEL_REVISION": "revision-test",
            "COMMON_EPOCH0_KV_TOKENS_PER_RANK_USED": PHYSICAL[16],
            "COMMON_EPOCH0_TRAIN_STEPS_USED": 5,
            "COMMON_EPOCH0_PROMPTS_TOTAL_USED": 320,
            "COMMON_EPOCH0_EXPECTED_RESPONSES_PER_STEP_USED": 1024,
            "COMMON_EPOCH0_PREEMPTION_POLICY_USED": "record",
            "COMMON_EPOCH0_WORKLOAD_PROFILE_ID": "deepseek-v2-lite-b64-n16-s5-test",
            "COMMON_EPOCH0_WORKLOAD_PROFILE_SHA256": _sha256(profile),
            "COMMON_EPOCH0_EXECUTION_PROFILE_USED": "profile-test",
            "COMMON_EPOCH0_TRAIN_BATCH_SIZE_USED": 64,
            "COMMON_EPOCH0_ROLLOUT_N_USED": 16,
            "COMMON_EPOCH0_MAX_PROMPT_LENGTH_USED": 1024,
            "COMMON_EPOCH0_MAX_RESPONSE_LENGTH_USED": 16384,
            "COMMON_EPOCH0_MAX_NUM_BATCHED_TOKENS_USED": 17408,
            "COMMON_EPOCH0_MAX_NUM_SEQS_USED": 64,
        },
    )
    (common / "FROZEN_CHECKPOINT_SHA256").write_text(
        checkpoint_digest(checkpoint)[0] + "\n", encoding="utf-8"
    )
    return common, checkpoint, model


def _caps(root: Path, common: Path, profile: Path) -> Path:
    path = root / "caps.env"
    validation_run = root / "natural_f2_authorization"
    validation_run.mkdir()
    validation_summary = validation_run / "KV_CAP_AUTHORIZATION_SUMMARY.json"
    validation_summary.write_text(
        json.dumps(
            {
                "status": "PASS",
                "lifecycle": "natural_f2",
                "run_root": str(validation_run),
                "common_epoch0_root": str(common),
                "train_batch_size": 64,
                "rollout_n": 16,
                "expected_responses_per_step": 1024,
                "max_num_seqs": 64,
                "workload_profile_id": "deepseek-v2-lite-b64-n16-s5-test",
                "workload_profile_sha256": _sha256(profile),
                "floors": [
                    {
                        "floor": floor,
                        "physical_cap": PHYSICAL[floor],
                        "admission_cap": ADMISSION[floor],
                    }
                    for floor in MODULE.ALLOWED_FLOORS
                ],
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    values: dict[str, object] = {
        "DEEPSEEK_KV_CAPS_VERIFIED": 0,
        "DEEPSEEK_N_F2_KV_CAPS_VERIFIED": 1,
        "DEEPSEEK_N_F2_KV_CAP_VALIDATION_RUN": validation_run,
        "DEEPSEEK_N_F2_KV_CAP_VALIDATION_SUMMARY": validation_summary,
        "DEEPSEEK_N_F2_KV_CAP_VALIDATION_SUMMARY_SHA256": _sha256(
            validation_summary
        ),
        "DEEPSEEK_N_F2_KV_CAP_VALIDATED_FLOORS": "16,8,4,2",
        "DEEPSEEK_KV_CAP_WORKLOAD_PROFILE_ID": "deepseek-v2-lite-b64-n16-s5-test",
        "DEEPSEEK_KV_CAP_WORKLOAD_PROFILE_SHA256": _sha256(profile),
        "DEEPSEEK_KV_CAP_COMMON_EPOCH0_ROOT": common,
        "DEEPSEEK_KV_CAP_MODEL_REVISION": "revision-test",
        "DEEPSEEK_KV_CAP_EXECUTION_PROFILE": "profile-test",
        "DEEPSEEK_KV_CAP_BLOCK_SIZE": 128,
        "DEEPSEEK_KV_CAP_ROLLOUT_N": 16,
        "DEEPSEEK_KV_CAP_SHARED_FULL16_PHYSICAL_TOKENS": PHYSICAL[16],
        "DEEPSEEK_VANILLA_KV_PHYSICAL_TOKENS": PHYSICAL[16],
    }
    for floor in MODULE.ALLOWED_FLOORS:
        values[f"DEEPSEEK_N_F2_KV_PHYSICAL_FLOOR{floor}"] = PHYSICAL[floor]
        values[f"DEEPSEEK_N_F2_KV_ADMISSION_FLOOR{floor}"] = ADMISSION[floor]
    _write_env(path, values)
    return path


def _manifest(
    root: Path,
    arm: str,
    phase: str,
    common: Path,
    checkpoint: Path,
    model: Path,
    profile: Path,
    caps: Path,
) -> None:
    _write_env(
        root / MODULE.MANIFEST_NAME,
        {
            "DEEPSEEK_BATCH64_ARM": arm,
            "DEEPSEEK_BATCH64_PHASE": phase,
            "DEEPSEEK_WORKLOAD_PROFILE_ID": "deepseek-v2-lite-b64-n16-s5-test",
            "DEEPSEEK_WORKLOAD_PROFILE_SHA256": _sha256(profile),
            "DEEPSEEK_BATCH64_COMMON_ROOT": common,
            "DEEPSEEK_BATCH64_FROZEN_CHECKPOINT": checkpoint,
            "DEEPSEEK_BATCH64_MODEL_PATH": model,
            "DEEPSEEK_BATCH64_MODEL_REVISION": "revision-test",
            "DEEPSEEK_BATCH64_EXECUTION_PROFILE": "profile-test",
            "DEEPSEEK_BATCH64_CAP_ENV_SHA256": _sha256(caps),
            "DEEPSEEK_BATCH64_EXECUTION_CODE_SHA256": EXECUTION_SHA256,
            "DEEPSEEK_BATCH64_FROZEN_CHECKPOINT_SHA256": checkpoint_digest(
                checkpoint
            )[0],
            "DEEPSEEK_BATCH64_PAIRED_REQUEST_SAMPLING_SEEDS": 1,
            "DEEPSEEK_BATCH64_TRAIN_BATCH_SIZE": 64,
            "DEEPSEEK_BATCH64_ROLLOUT_N": 16,
            "DEEPSEEK_BATCH64_MAX_NUM_SEQS": 64,
            "DEEPSEEK_BATCH64_MAX_PROMPT_LENGTH": 1024,
            "DEEPSEEK_BATCH64_MAX_RESPONSE_LENGTH": 16384,
            "DEEPSEEK_BATCH64_MAX_NUM_BATCHED_TOKENS": 17408,
            "DEEPSEEK_BATCH64_FULL16_PHYSICAL_TOKENS": PHYSICAL[16],
            "DEEPSEEK_BATCH64_TEMPERATURE": 0.9,
            "DEEPSEEK_BATCH64_TOP_P": 0.9,
            "DEEPSEEK_BATCH64_TOP_K": 50,
            "DEEPSEEK_BATCH64_DATASET_FRACTION": (
                0.001765 if phase == "gate" else 0.01
            ),
            "DEEPSEEK_BATCH64_FORCED_SELECTED_FLOOR": (
                4 if phase == "gate" and arm == "adafloor" else "none"
            ),
        },
    )


def _plan(step: int, floor: int, response_cap: int) -> dict[str, object]:
    stages = [stage for stage in (8, 4, 2) if stage >= floor] if floor < 16 else [16]
    survivor_sets = {
        16: list(range(16)),
        8: list(range(8, 16)),
        4: list(range(12, 16)),
        2: list(range(14, 16)),
    }
    return {
        "step": step,
        "selected_floor": floor,
        "feasible": True,
        "kv_cap": PHYSICAL[floor],
        "kv_admission_cap": ADMISSION[floor],
        "max_adjusted_rank_peak_tokens": ADMISSION[floor] - 128,
        "tail_guard_response_cap": response_cap,
        "release_area": 40.0 if floor < 16 else 0.0,
        "release_area_unit": "rank_token_proxy",
        "shrink_stages": stages,
        "stage_survivor_ranks": [survivor_sets[stage] for stage in stages],
    }


def _timestamp(second: int) -> str:
    return f"2026-08-06 12:{second // 60:02d}:{second % 60:02d},000"


def _log(
    arm: str,
    steps: int,
    floors: list[int],
    response_caps: list[int],
    *,
    tail_guard_caps: list[int] | None = None,
    vanilla_preemption: bool = True,
    adafloor_preemption: bool = False,
) -> str:
    lines: list[str] = []
    for rank in range(16):
        lines.append(
            f"(WorkerDict pid={1000 + rank}) [Rank {rank} | Local Rank 0] worker ready"
        )
    for step in range(1, steps + 1):
        base = (step - 1) * 20
        max_tokens = response_caps[step - 1]
        for rank in range(16):
            pid = 1000 + rank
            if step == 1:
                lines.append(
                    f"(WorkerDict pid={pid}) kwargs: {{'n': 1, 'max_tokens': {max_tokens}, "
                    "'temperature': 0.9, 'top_k': 50, 'top_p': 0.9}"
                )
            applied_cap = (
                tail_guard_caps[step - 1]
                if arm == "adafloor" and tail_guard_caps is not None
                else max_tokens
            )
            lines.append(
                f"(WorkerDict pid={pid}) [mode1_timeline] "
                f"rollout_worker_infer_start rank={rank} step={step} epoch=0 "
                f"batch_size=64 max_tokens={applied_cap} total_s=0.1"
            )
            if arm == "adafloor":
                lines.append(
                    f"(WorkerDict pid={pid}) rollout_worker_resize_start rank={rank} "
                    f"step={step} epoch=0 target_floor={floors[step - 1]} "
                    f"target_kv={PHYSICAL[floors[step - 1]]}"
                )
        if arm == "adafloor" and floors[step - 1] in (8, 4, 2):
            target = ", ".join(str(rank) for rank in range(8, 16))
            for rank in range(16):
                lines.append(
                    f"(WorkerDict pid={1000 + rank}) INFO {_timestamp(base + 6)} "
                    "Elastic parallel shrink rpc done: "
                    f"global_rank={rank} active_ranks=[{target}]"
                )
        if (arm == "vanilla" and vanilla_preemption) or (
            arm == "adafloor" and adafloor_preemption
        ):
            lines.append(
                f"(WorkerDict pid=1000) INFO {_timestamp(base + 5)} "
                "[scheduler.py:500] Preempting request 7 for request 2 "
                "discarded_computed_tokens=512"
            )
        lines.extend(
            [
                *(
                    [
                        "Shrink-aware tail-guard response cap: "
                        f"selected_floor={floors[step - 1]} "
                        f"plan_cap={tail_guard_caps[step - 1]}"
                    ]
                    if arm == "adafloor" and tail_guard_caps is not None
                    else []
                ),
                "response/aborted_ratio:0.0",
                f"critic/score/mean:{0.5:.6f}",
                f"{_timestamp(base + 10)} rollout_output_time_s: 10.0",
                f"training/global_step:{step}",
            ]
        )
    lines.append("After trainer.fit")
    return "\n".join(lines) + "\n"


def _artifacts(
    epoch: Path,
    steps: int,
    arm: str,
    *,
    generated_length: int = 4,
    finish_reason: str = "stop",
) -> None:
    rollout_dir = epoch / "rollout_data"
    length_dir = epoch / "rollout_length"
    rollout_dir.mkdir(parents=True, exist_ok=True)
    length_dir.mkdir(parents=True, exist_ok=True)
    for step in range(1, steps + 1):
        rows: list[str] = []
        lengths: list[str] = []
        prompt_offset = (step - 1) * 64
        for prompt_index in range(64):
            for sample_index in range(16):
                global_prompt = prompt_offset + prompt_index
                row = {
                    "rollout_prompt_hash": f"prompt-{global_prompt:03d}",
                    "rollout_sample_index": sample_index,
                    "rollout_request_seed": global_prompt * 16 + sample_index + 100,
                    "response_mask": [1] * generated_length,
                    "decoded_response_length": generated_length,
                    "response_finish_reason": finish_reason,
                    "score": 0.5,
                    "output": f"{arm}-output-{global_prompt}-{sample_index}",
                    "prompts": [
                        100001,
                        100001,
                        TEST_BOS_TOKEN_ID,
                        global_prompt + 1,
                    ],
                }
                rows.append(json.dumps(row, sort_keys=True))
                lengths.append(str(generated_length))
        if arm == "adafloor":
            rows.reverse()
            lengths.reverse()
        (rollout_dir / f"{step}.jsonl").write_text("\n".join(rows) + "\n", encoding="utf-8")
        (length_dir / f"length_{step}.txt").write_text(
            "\n".join(lengths) + "\n", encoding="utf-8"
        )


def _replace_prompt_identity(epoch: Path, source_prompt: int, target_prompt: int) -> None:
    replaced = 0
    for path in sorted((epoch / "rollout_data").glob("*.jsonl")):
        rows: list[str] = []
        for raw_row in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(raw_row)
            if row["rollout_prompt_hash"] == f"prompt-{source_prompt:03d}":
                sample_index = row["rollout_sample_index"]
                row["rollout_prompt_hash"] = f"prompt-{target_prompt:03d}"
                row["rollout_request_seed"] = target_prompt * 16 + sample_index + 100
                replaced += 1
            rows.append(json.dumps(row, sort_keys=True))
        path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    assert replaced == 16


def _fixture(tmp_path: Path, phase: str = "gate", floor: int = 4) -> dict[str, Path]:
    steps = MODULE.EXPECTED_PHASE_STEPS[phase]
    profile = _profile(tmp_path)
    common, checkpoint, model = _common(tmp_path, profile)
    caps = _caps(tmp_path, common, profile)
    floors = [floor] * steps
    response_caps = [8] * steps
    roots: dict[str, Path] = {}
    for arm in ("vanilla", "adafloor"):
        root = tmp_path / arm
        epoch_name = "epoch_001_mode0_full16" if arm == "vanilla" else "epoch_001_mode1_natural"
        epoch = root / epoch_name
        (epoch / "logs").mkdir(parents=True)
        _manifest(root, arm, phase, common, checkpoint, model, profile, caps)
        caps_for_log = [16384] * steps
        (epoch / "logs" / "run.txt").write_text(
            _log(
                arm,
                steps,
                floors,
                caps_for_log,
                tail_guard_caps=response_caps if arm == "adafloor" else None,
            ),
            encoding="utf-8",
        )
        _artifacts(epoch, steps, arm)
        if arm == "adafloor":
            oracle = epoch / "oracle"
            oracle.mkdir()
            (oracle / "length_sorted_rank_plan_summary.json").write_text(
                json.dumps(
                    [_plan(step, floors[step - 1], response_caps[step - 1]) for step in range(1, steps + 1)]
                ),
                encoding="utf-8",
            )
        roots[arm] = root
    roots.update({"profile": profile, "common": common, "caps": caps})
    return roots


def _verify(paths: dict[str, Path], phase: str = "gate") -> dict[str, object]:
    return MODULE.verify_pair(
        phase,
        paths["vanilla"],
        paths["adafloor"],
        paths["common"],
        paths["caps"],
        paths["profile"],
        EXECUTION_SHA256,
    )


def test_gate_accepts_vanilla_preemption_and_different_output_bytes(tmp_path: Path) -> None:
    summary = _verify(_fixture(tmp_path))

    assert summary["status"] == "PASS"
    assert summary["comparison"]["paired_identity_count"] == 1024
    assert summary["comparison"]["paired_identity_multisets_equal"] is True
    assert summary["vanilla"]["preemption"]["raw_events"] == 1
    assert summary["vanilla"]["preemption"]["unique_preempted_request_ids"] == [
        "step1:rank0:request7"
    ]
    assert summary["vanilla"]["preemption"]["recomputed_kv_tokens"] == 512
    assert summary["vanilla"]["preemption"]["recomputed_kv_tokens_reason"] is None
    assert summary["adafloor"]["preemption"]["raw_events"] == 0
    assert summary["adafloor"]["preemption"]["recomputed_kv_tokens"] == 0
    assert summary["vanilla"]["aborted_responses"] == 0
    assert summary["adafloor"]["oom_detected"] is False
    assert summary["adafloor"]["selected_floors"] == [4]
    assert summary["adafloor"]["predicted_release_proxy_rank_tokens"] == 40.0
    assert summary["adafloor"]["predicted_release_proxy_unit"] == (
        "rank_token_proxy"
    )
    assert "planned_release_area_rank_seconds" not in summary["adafloor"]
    assert summary["adafloor"]["coordinated_release"]["total_rank_seconds"] == 32.0
    release_step = summary["adafloor"]["coordinated_release"]["steps"][0]
    assert release_step["predicted_release_proxy_rank_tokens"] == 40.0
    assert release_step["predicted_release_proxy_unit"] == "rank_token_proxy"
    assert "planned_release_area" not in release_step


def test_pair_rejects_an_incorrect_release_area_unit(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    plan_path = (
        paths["adafloor"]
        / "epoch_001_mode1_natural"
        / "oracle"
        / "length_sorted_rank_plan_summary.json"
    )
    plans = json.loads(plan_path.read_text(encoding="utf-8"))
    plans[0]["release_area_unit"] = "rank_seconds"
    plan_path.write_text(json.dumps(plans), encoding="utf-8")

    with pytest.raises(MODULE.VerificationError, match="release area unit"):
        _verify(paths)


@pytest.mark.parametrize(("bos_count", "accepted"), ((0, False), (1, True), (2, False)))
def test_pair_requires_exactly_one_bos_per_left_padded_prompt(
    tmp_path: Path, bos_count: int, accepted: bool
) -> None:
    paths = _fixture(tmp_path)
    path = (
        paths["vanilla"]
        / "epoch_001_mode0_full16"
        / "rollout_data"
        / "1.jsonl"
    )
    rows = path.read_text(encoding="utf-8").splitlines()
    first = json.loads(rows[0])
    first["prompts"] = [100001, 100001] + [TEST_BOS_TOKEN_ID] * bos_count + [42]
    rows[0] = json.dumps(first, sort_keys=True)
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")

    if accepted:
        assert _verify(paths)["status"] == "PASS"
    else:
        with pytest.raises(MODULE.VerificationError, match="exactly one BOS token"):
            _verify(paths)


def test_epoch_phase_requires_five_steps_of_1024_responses(tmp_path: Path) -> None:
    summary = _verify(_fixture(tmp_path, phase="epoch", floor=8), phase="epoch")

    assert summary["expected_steps"] == 5
    assert summary["comparison"]["paired_identity_count"] == 5120
    assert summary["comparison"]["paired_unique_identity_count"] == 5120
    assert summary["vanilla"]["prompt_occurrence_count"] == 320
    assert summary["vanilla"]["unique_prompt_count"] == 320
    assert summary["adafloor"]["selected_floors"] == [8] * 5
    assert summary["adafloor"]["coordinated_release"]["total_rank_seconds"] == 160.0


def test_epoch_accepts_three_prompts_repeated_across_steps(tmp_path: Path) -> None:
    paths = _fixture(tmp_path, phase="epoch", floor=8)
    for arm, epoch_name in (
        ("vanilla", "epoch_001_mode0_full16"),
        ("adafloor", "epoch_001_mode1_natural"),
    ):
        epoch = paths[arm] / epoch_name
        for source, target in ((256, 0), (257, 64), (258, 128)):
            _replace_prompt_identity(epoch, source, target)

    summary = _verify(paths, phase="epoch")

    assert summary["request_identity_comparison"] == "multiset"
    assert summary["comparison"]["paired_identity_multisets_equal"] is True
    assert summary["comparison"]["paired_identity_count"] == 5120
    assert summary["comparison"]["paired_unique_identity_count"] == 5072
    for arm in ("vanilla", "adafloor"):
        assert summary[arm]["prompt_occurrence_count"] == 320
        assert summary[arm]["unique_prompt_count"] == 317


def test_rejects_selected_floor_without_completed_transition(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    log = paths["adafloor"] / "epoch_001_mode1_natural" / "logs" / "run.txt"
    lines = [
        line for line in log.read_text(encoding="utf-8").splitlines()
        if "Elastic parallel shrink rpc done" not in line
    ]
    log.write_text("\n".join(lines) + "\n", encoding="utf-8")

    with pytest.raises(MODULE.VerificationError, match="does not complete a coordinated transition"):
        _verify(paths)


def test_accepts_runtime_selected_natural_survivor_identity(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    log = paths["adafloor"] / "epoch_001_mode1_natural" / "logs" / "run.txt"
    text = log.read_text(encoding="utf-8").replace(
        "active_ranks=[8, 9, 10, 11, 12, 13, 14, 15]",
        "active_ranks=[0, 1, 2, 3, 4, 5, 6, 7]",
    )
    log.write_text(text, encoding="utf-8")

    summary = _verify(paths)

    transition = summary["adafloor"]["coordinated_release"]["steps"][0][
        "transitions"
    ][0]
    assert transition["survivor_ranks"] == list(range(8))


def test_rejects_epoch_without_any_shrinkable_floor(tmp_path: Path) -> None:
    paths = _fixture(tmp_path, phase="epoch", floor=16)

    with pytest.raises(MODULE.VerificationError, match="does not select any shrinkable"):
        _verify(paths, phase="epoch")


def test_interleaved_worker_prefixes_do_not_misattribute_preemption(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    log = paths["vanilla"] / "epoch_001_mode0_full16" / "logs" / "run.txt"
    text = log.read_text(encoding="utf-8").replace(
        "(WorkerDict pid=1000) INFO 2026-08-06 12:00:05,000 "
        "[scheduler.py:500] Preempting request 7 for request 2 "
        "discarded_computed_tokens=512",
        "(WorkerDict pid=9999) unrelated output "
        "(WorkerDict pid=1001) [Rank 1 | Local Rank 0] "
        "INFO 2026-08-06 12:00:05,000 [scheduler.py:500] "
        "Preempting request 7 for request 2 discarded_computed_tokens=512",
    )
    log.write_text(text, encoding="utf-8")

    summary = _verify(paths)

    assert summary["vanilla"]["preemption"]["unique_preempted_request_ids"] == [
        "step1:rank1:request7"
    ]


def test_rejects_cross_arm_identity_mismatch(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    path = paths["adafloor"] / "epoch_001_mode1_natural" / "rollout_data" / "1.jsonl"
    rows = path.read_text(encoding="utf-8").splitlines()
    first = json.loads(rows[0])
    first["rollout_request_seed"] += 1_000_000
    rows[0] = json.dumps(first)
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")

    with pytest.raises(MODULE.VerificationError, match="identity multisets differ"):
        _verify(paths)


def test_rejects_stale_execution_code_manifest(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    manifest = paths["adafloor"] / MODULE.MANIFEST_NAME
    text = manifest.read_text(encoding="utf-8").replace(
        EXECUTION_SHA256, "e" * 64
    )
    manifest.write_text(text, encoding="utf-8")

    with pytest.raises(MODULE.VerificationError, match="EXECUTION_CODE_SHA256"):
        _verify(paths)


def test_rejects_mutated_frozen_checkpoint(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    reuse = MODULE._load_env(paths["common"] / "reuse.env")
    checkpoint = Path(reuse["DYNAMIC_INITIAL_RESUME_CKPT"])
    shard = checkpoint / "actor" / "dist_ckpt" / "__0_0.distcp"
    shard.write_bytes(shard.read_bytes() + b"tampered")

    with pytest.raises(MODULE.VerificationError, match="checkpoint content"):
        _verify(paths)


def test_rejects_identity_multiplicity_mismatch_when_sets_match(tmp_path: Path) -> None:
    paths = _fixture(tmp_path, phase="epoch", floor=8)
    vanilla_epoch = paths["vanilla"] / "epoch_001_mode0_full16"
    adafloor_epoch = paths["adafloor"] / "epoch_001_mode1_natural"
    for source, target in ((128, 0), (192, 0), (256, 64)):
        _replace_prompt_identity(vanilla_epoch, source, target)
    for source, target in ((128, 64), (192, 64), (256, 0)):
        _replace_prompt_identity(adafloor_epoch, source, target)

    with pytest.raises(MODULE.VerificationError, match="identity multisets differ"):
        _verify(paths, phase="epoch")


def test_rejects_duplicate_identity_within_one_arm(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    path = paths["vanilla"] / "epoch_001_mode0_full16" / "rollout_data" / "1.jsonl"
    rows = path.read_text(encoding="utf-8").splitlines()
    rows[1] = rows[0]
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")

    with pytest.raises(MODULE.VerificationError, match="complete sample-index set"):
        _verify(paths)


def test_accepts_duplicate_prompt_occurrence_within_step(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    for arm, epoch_name in (
        ("vanilla", "epoch_001_mode0_full16"),
        ("adafloor", "epoch_001_mode1_natural"),
    ):
        _replace_prompt_identity(paths[arm] / epoch_name, 1, 0)

    summary = _verify(paths)

    assert summary["comparison"]["paired_identity_multisets_equal"] is True
    assert summary["comparison"]["paired_identity_count"] == 1024
    assert summary["comparison"]["paired_unique_identity_count"] == 1008
    for arm in ("vanilla", "adafloor"):
        assert summary[arm]["prompt_occurrence_count"] == 64
        assert summary[arm]["unique_prompt_count"] == 63


def test_rejects_adafloor_scheduler_preemption(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    log = paths["adafloor"] / "epoch_001_mode1_natural" / "logs" / "run.txt"
    text = log.read_text(encoding="utf-8")
    text += (
        "(WorkerDict pid=1000) INFO 2026-08-06 12:00:05,000 "
        "[scheduler.py:500] Preempting request 7 for request 2\n"
    )
    log.write_text(text, encoding="utf-8")

    with pytest.raises(MODULE.VerificationError, match="AdaFloor has 1 raw"):
        _verify(paths)


def test_rejects_unexplained_generated_work_reduction(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    epoch = paths["adafloor"] / "epoch_001_mode1_natural"
    _artifacts(epoch, 1, "adafloor", generated_length=3)

    with pytest.raises(MODULE.VerificationError, match="differs by more than 1%"):
        _verify(paths)


def test_accepts_tailguard_work_reduction_and_reports_retention(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    vanilla_epoch = paths["vanilla"] / "epoch_001_mode0_full16"
    adafloor_epoch = paths["adafloor"] / "epoch_001_mode1_natural"
    _artifacts(vanilla_epoch, 1, "vanilla", generated_length=10)
    _artifacts(
        adafloor_epoch,
        1,
        "adafloor",
        generated_length=8,
        finish_reason="length",
    )

    summary = _verify(paths)

    assert summary["comparison"]["generated_work_relative_difference"] == pytest.approx(0.2)
    assert summary["comparison"]["adafloor_generated_token_retention"] == pytest.approx(0.8)
    assert summary["adafloor"]["cap_hits"] == 1024


def test_shared_max_length_hits_do_not_explain_work_reduction() -> None:
    summary = {
        "steps": [
            {"response_cap": MODULE.MAX_RESPONSE_LENGTH, "cap_hits": 3},
        ]
    }

    assert MODULE._has_exercised_tailguard(summary) is False


def test_reduced_cap_hit_is_tailguard_evidence() -> None:
    summary = {
        "steps": [
            {"response_cap": MODULE.MAX_RESPONSE_LENGTH - 1, "cap_hits": 1},
        ]
    }

    assert MODULE._has_exercised_tailguard(summary) is True


def test_stop_at_reduced_cap_is_not_tailguard_evidence(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    vanilla_epoch = paths["vanilla"] / "epoch_001_mode0_full16"
    adafloor_epoch = paths["adafloor"] / "epoch_001_mode1_natural"
    _artifacts(vanilla_epoch, 1, "vanilla", generated_length=10)
    _artifacts(adafloor_epoch, 1, "adafloor", generated_length=8)

    with pytest.raises(MODULE.VerificationError, match="differs by more than 1%"):
        _verify(paths)


def test_tailguard_budget_is_bounded() -> None:
    summary = {
        "steps": [
            {
                "response_cap": MODULE.MAX_RESPONSE_LENGTH - 128,
                "cap_hits": 2,
            },
        ]
    }

    assert MODULE._tailguard_reduction_budget(summary) == 256


def test_rejects_adafloor_generated_work_more_than_one_percent_above_vanilla(
    tmp_path: Path,
) -> None:
    paths = _fixture(tmp_path)
    adafloor_epoch = paths["adafloor"] / "epoch_001_mode1_natural"
    _artifacts(adafloor_epoch, 1, "adafloor", generated_length=5)

    with pytest.raises(MODULE.VerificationError, match="differs by more than 1%"):
        _verify(paths)


def test_accepts_small_adafloor_generated_work_increase(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    epoch = paths["adafloor"] / "epoch_001_mode1_natural"
    rollout_path = epoch / "rollout_data" / "1.jsonl"
    length_path = epoch / "rollout_length" / "length_1.txt"
    rows = rollout_path.read_text(encoding="utf-8").splitlines()
    lengths = length_path.read_text(encoding="utf-8").splitlines()
    first = json.loads(rows[0])
    first["response_mask"] = [1] * 5
    rows[0] = json.dumps(first, sort_keys=True)
    lengths[0] = "5"
    rollout_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    length_path.write_text("\n".join(lengths) + "\n", encoding="utf-8")

    summary = _verify(paths)

    assert summary["comparison"]["adafloor_generated_token_retention"] > 1.0
    assert summary["comparison"][
        "generated_work_absolute_relative_difference"
    ] < 0.01


def test_normalizes_proven_synthetic_terminal_padding_token(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    for arm, epoch_name in (
        ("vanilla", "epoch_001_mode0_full16"),
        ("adafloor", "epoch_001_mode1_natural"),
    ):
        epoch = paths[arm] / epoch_name
        rollout_path = epoch / "rollout_data" / "1.jsonl"
        length_path = epoch / "rollout_length" / "length_1.txt"
        rows = rollout_path.read_text(encoding="utf-8").splitlines()
        lengths = length_path.read_text(encoding="utf-8").splitlines()
        row = json.loads(rows[0])
        cap = 16384 if arm == "vanilla" else 8
        if arm == "adafloor":
            row["responses"] = [7] * cap + [100001] * 4
            row["response_mask"] = [1] * (cap + 1) + [0] * 3
            rows[0] = json.dumps(row, sort_keys=True)
            lengths[0] = str(cap + 1)
        else:
            row["response_mask"] = [1] * 10
            rows[0] = json.dumps(row, sort_keys=True)
            lengths[0] = "10"
        rollout_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
        length_path.write_text("\n".join(lengths) + "\n", encoding="utf-8")

    summary = _verify(paths)

    assert summary["adafloor"]["synthetic_terminal_pad_count"] == 1
    assert summary["adafloor"]["steps"][0]["synthetic_terminal_pad_count"] == 1
    assert summary["adafloor"]["steps"][0]["generated_tokens"] == 4100


def test_rejects_unproven_one_token_cap_overrun(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    epoch = paths["adafloor"] / "epoch_001_mode1_natural"
    rollout_path = epoch / "rollout_data" / "1.jsonl"
    length_path = epoch / "rollout_length" / "length_1.txt"
    rows = rollout_path.read_text(encoding="utf-8").splitlines()
    lengths = length_path.read_text(encoding="utf-8").splitlines()
    row = json.loads(rows[0])
    row["response_mask"] = [1] * 9
    rows[0] = json.dumps(row, sort_keys=True)
    lengths[0] = "9"
    rollout_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    length_path.write_text("\n".join(lengths) + "\n", encoding="utf-8")

    with pytest.raises(MODULE.VerificationError, match="synthetic terminal"):
        _verify(paths)


def test_rejects_full16_capacity_mismatch(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    metadata = paths["common"] / "common_epoch0_metadata.env"
    text = metadata.read_text(encoding="utf-8").replace(
        f"COMMON_EPOCH0_KV_TOKENS_PER_RANK_USED={PHYSICAL[16]}",
        "COMMON_EPOCH0_KV_TOKENS_PER_RANK_USED=614016",
    )
    metadata.write_text(text, encoding="utf-8")

    with pytest.raises(MODULE.VerificationError, match="Full16 physical KV capacities differ"):
        _verify(paths)


def test_global_verified_flag_remains_zero(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)

    assert "DEEPSEEK_KV_CAPS_VERIFIED=0" in paths["caps"].read_text(encoding="utf-8")
    assert _verify(paths)["status"] == "PASS"


def test_rejects_tampered_natural_floor2_authorization_summary(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    summary_path = tmp_path / "natural_f2_authorization" / "KV_CAP_AUTHORIZATION_SUMMARY.json"
    summary_path.write_text("{}\n", encoding="utf-8")

    with pytest.raises(MODULE.VerificationError, match="summary SHA256 mismatch"):
        _verify(paths)


def test_accepts_auto_profiled_common_capacity_when_pair_caps_match(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    metadata = paths["common"] / "common_epoch0_metadata.env"
    text = metadata.read_text(encoding="utf-8").replace(
        f"COMMON_EPOCH0_KV_TOKENS_PER_RANK_USED={PHYSICAL[16]}",
        "COMMON_EPOCH0_KV_TOKENS_PER_RANK_USED=auto",
    )
    metadata.write_text(text, encoding="utf-8")

    assert _verify(paths)["status"] == "PASS"


def test_unmapped_preemption_reports_unique_ids_as_null() -> None:
    summary = MODULE._preemption_summary(
        {
            "preemptions": [(None, 999, "7", "2", 64)],
            "pid_to_rank": {},
            "rollout_events": [(100.0, 10.0)],
        },
        all_responses_completed=True,
        no_aborts=True,
    )

    assert summary["raw_events"] == 1
    assert summary["unique_preempted_request_count"] is None
    assert summary["unique_preempted_request_ids"] is None
    assert "no reliable global-rank mapping" in summary["unique_preempted_request_ids_reason"]
    assert summary["recomputed_kv_tokens"] == 64


def test_no_preemption_reports_zero_recomputed_tokens() -> None:
    summary = MODULE._preemption_summary(
        {
            "preemptions": [],
            "pid_to_rank": {},
            "rollout_events": [(100.0, 10.0)],
        },
        all_responses_completed=True,
        no_aborts=True,
    )

    assert summary["raw_events"] == 0
    assert summary["unique_preempted_request_count"] == 0
    assert summary["recomputed_kv_tokens"] == 0
    assert summary["recomputed_kv_tokens_reason"] is None


def test_multiple_preemptions_sum_each_discarded_token_count() -> None:
    summary = MODULE._preemption_summary(
        {
            "preemptions": [
                (95.0, 1000, "7", "2", 128),
                (96.0, 1000, "7", "3", 64),
            ],
            "pid_to_rank": {1000: 0},
            "rollout_events": [(100.0, 10.0)],
        },
        all_responses_completed=True,
        no_aborts=True,
    )

    assert summary["raw_events"] == 2
    assert summary["unique_preempted_request_count"] == 1
    assert summary["unique_preempted_request_ids"] == [
        "step1:rank0:request7"
    ]
    assert summary["recomputed_kv_tokens"] == 192
    assert summary["recomputed_kv_tokens_reason"] is None


@pytest.mark.parametrize(
    ("all_responses_completed", "no_aborts"),
    ((False, True), (True, False)),
)
def test_discard_count_requires_completed_nonaborted_responses(
    all_responses_completed: bool,
    no_aborts: bool,
) -> None:
    summary = MODULE._preemption_summary(
        {
            "preemptions": [(95.0, 1000, "7", "2", 128)],
            "pid_to_rank": {1000: 0},
            "rollout_events": [(100.0, 10.0)],
        },
        all_responses_completed=all_responses_completed,
        no_aborts=no_aborts,
    )

    assert summary["raw_events"] == 1
    assert summary["unique_preempted_request_count"] == 1
    assert summary["recomputed_kv_tokens"] is None
    assert "only when every response completes" in summary[
        "recomputed_kv_tokens_reason"
    ]


def test_legacy_preemption_without_discard_count_reports_unknown(
    tmp_path: Path,
) -> None:
    paths = _fixture(tmp_path)
    log = paths["vanilla"] / "epoch_001_mode0_full16" / "logs" / "run.txt"
    text = log.read_text(encoding="utf-8").replace(
        " discarded_computed_tokens=512", ""
    )
    log.write_text(text, encoding="utf-8")

    summary = _verify(paths)["vanilla"]["preemption"]

    assert summary["raw_events"] == 1
    assert summary["unique_preempted_request_count"] == 1
    assert summary["recomputed_kv_tokens"] is None
    assert "lacks discarded_computed_tokens" in summary[
        "recomputed_kv_tokens_reason"
    ]
