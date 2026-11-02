from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import pytest


TOOL = Path(__file__).parents[1] / "tools" / "generate_deepseek_kv_caps.py"
SPEC = importlib.util.spec_from_file_location("generate_deepseek_kv_caps", TOOL)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _probe(
    path: Path,
    common: Path,
    history: Path,
    floor: int,
    physical: int,
    lifecycle: str = "natural_f4",
    max_num_seqs: int = 32,
) -> Path:
    import hashlib

    payload = {
        "lifecycle": lifecycle,
        "floor": floor,
        "world_size": 16,
        "observed_tokens": physical,
        "per_rank_tokens": {str(rank): physical for rank in range(16)},
        "complete_target_waves": 1,
        "model_revision": "revision",
        "execution_profile": "profile",
        "runtime_profile": "runtime-profile",
        "runtime_profile_sha256": "a" * 64,
        "execution_code_sha256": "b" * 64,
        "max_prompt_length": 1024,
        "max_response_length": 16384,
        "max_num_batched_tokens": 17408,
        "max_num_seqs": max_num_seqs,
        "probe_tail_guard_min_cap": 64,
        "probe_tail_guard_round_to": 64,
        "plan_tail_guard_response_cap": 64,
        "actual_plan_response_cap": 64,
        "gpu_memory_utilization": 0.9,
        "enforce_eager": "True",
        "block_size": 128,
        "common_epoch0_root": str(common),
        "planning_history_root": str(history),
        "planning_history_sha256": hashlib.sha256(
            (history / "offline_planning_history.json").read_bytes()
        ).hexdigest(),
        "planning_history_manifest_sha256": hashlib.sha256(
            (history / "kv_probe_trigger_manifest.json").read_bytes()
        ).hexdigest(),
        "planning_trigger_subset_sha256": hashlib.sha256(
            (history / "rollout_data" / "1.jsonl").read_bytes()
        ).hexdigest(),
        "planner_train_sha256": "c" * 64,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


@pytest.mark.parametrize(
    ("lifecycle", "prefix", "floor_capacities"),
    [
        ("natural_f4", "DEEPSEEK_N_F4", {4: 400000, 8: 500096, 16: 614144}),
        (
            "natural_f2",
            "DEEPSEEK_N_F2",
            {2: 300032, 4: 400000, 8: 500096, 16: 614144},
        ),
    ],
)
def test_generates_separate_admission_and_physical_caps(
    tmp_path: Path,
    lifecycle: str,
    prefix: str,
    floor_capacities: dict[int, int],
) -> None:
    common = tmp_path / "common"
    rollout = common / "epoch_000_mode0_probe" / "rollout_data"
    rollout.mkdir(parents=True)
    model = tmp_path / "model"
    model.mkdir()
    history = tmp_path / "probe_history"
    history.mkdir()
    (history / "offline_planning_history.json").write_text(
        '{"schema_version": 1}\n', encoding="utf-8"
    )
    (history / "kv_probe_trigger_manifest.json").write_text(
        '{"schema_version": 1}\n', encoding="utf-8"
    )
    (history / "rollout_data").mkdir()
    (history / "rollout_data" / "1.jsonl").write_text(
        '{"input": "trigger"}\n', encoding="utf-8"
    )
    (model / "config.json").write_text(
        json.dumps({"pad_token_id": None, "eos_token_id": 100001}),
        encoding="utf-8",
    )
    (common / "common_epoch0_metadata.env").write_text(
        "\n".join(
            [
                f"export COMMON_EPOCH0_MODEL_PATH={model}",
                "export COMMON_EPOCH0_MODEL_REVISION=revision",
                "export COMMON_EPOCH0_EXECUTION_PROFILE_USED=profile",
                "export COMMON_EPOCH0_TRAIN_FILE_USED=/data/deepscaler/train.parquet",
                "export COMMON_EPOCH0_TEST_FILE_USED=/data/deepscaler/test.parquet",
                "export COMMON_EPOCH0_DATASET_FRACTION_USED=0.005",
                "export COMMON_EPOCH0_TRAIN_BATCH_SIZE_USED=32",
                "export COMMON_EPOCH0_ROLLOUT_N_USED=16",
                "export COMMON_EPOCH0_MAX_PROMPT_LENGTH_USED=1024",
                "export COMMON_EPOCH0_MAX_RESPONSE_LENGTH_USED=16384",
                "export COMMON_EPOCH0_MAX_NUM_BATCHED_TOKENS_USED=17408",
                "export COMMON_EPOCH0_MAX_NUM_SEQS_USED=32",
                "export COMMON_EPOCH0_GPU_MEMORY_UTILIZATION_USED=0.9",
                "export COMMON_EPOCH0_KV_BLOCK_SIZE_USED=128",
                "export COMMON_EPOCH0_TRAIN_STEPS_USED=5",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (common / "MEASURED_GPU_KV_CACHE_TOKENS_PER_RANK").write_text(
        "614144\n", encoding="utf-8"
    )
    prompt_lengths = list(range(10, 170))
    for step in range(5):
        rows = []
        for prompt_index in range(step * 32, (step + 1) * 32):
            length = prompt_lengths[prompt_index]
            prompt_tokens = [100001] * (1024 - length) + list(range(length))
            for sample in range(16):
                rows.append(
                    json.dumps(
                        {
                            "input": f"prompt-{prompt_index}",
                            "prompts": prompt_tokens,
                            "sample": sample,
                        }
                    )
                )
        (rollout / f"{step + 1}.jsonl").write_text(
            "\n".join(rows) + "\n", encoding="utf-8"
        )
    summaries = {
        floor: _probe(
            tmp_path / f"floor{floor}.json",
            common,
            history,
            floor,
            physical,
            lifecycle,
        )
        for floor, physical in floor_capacities.items()
    }
    args = argparse.Namespace(
        lifecycle=lifecycle,
        common_epoch0_root=common,
        model_path=model,
        floor2_summary=summaries.get(2),
        floor4_summary=summaries[4],
        floor8_summary=summaries[8],
        floor16_summary=summaries[16],
        output=tmp_path / "caps.env",
        block_size=128,
        rollout_n=16,
        target_ratio=1.0,
        runtime_profile="runtime-profile",
        runtime_profile_sha256="a" * 64,
        execution_code_sha256="b" * 64,
        probe_history_root=history,
    )
    output = MODULE.generate(args)
    assert "export DEEPSEEK_KV_CAPS_VERIFIED=0" in output
    assert f"export {prefix}_KV_CAPS_VERIFIED=0" in output
    assert f"export {prefix}_RUNTIME_PROFILE=runtime-profile" in output
    assert f"export {prefix}_RUNTIME_PROFILE_SHA256={'a' * 64}" in output
    assert "export DEEPSEEK_KV_CAP_MAX_NUM_SEQS=32" in output
    assert "export DEEPSEEK_KV_CAP_PROBE_TAIL_GUARD_MIN_CAP=64" in output
    assert "export DEEPSEEK_KV_CAP_PROBE_TAIL_GUARD_ROUND_TO=64" in output
    assert "export DEEPSEEK_KV_CAP_PROBE_PLAN_RESPONSE_CAP=64" in output
    assert "export DEEPSEEK_KV_CAP_PROMPT_RESERVE_TOKENS=5392" in output
    assert "export DEEPSEEK_KV_CAP_BLOCK_RESERVE_TOKENS=4224" in output
    assert f"export {prefix}_KV_PHYSICAL_FLOOR16=614144" in output
    assert f"export {prefix}_KV_ADMISSION_FLOOR16=604416" in output
    assert "export DEEPSEEK_VANILLA_KV_ADMISSION_TOKENS=604416" in output
    assert (
        f"export {prefix}_KV_PROBE_PLANNER_TRAIN_SHA256_FLOOR4="
        + "c" * 64
    ) in output
    if lifecycle == "natural_f2":
        assert "export DEEPSEEK_N_F2_KV_PHYSICAL_FLOOR2=300032" in output
        assert "export DEEPSEEK_N_F2_KV_ADMISSION_FLOOR2=290304" in output
        assert "DEEPSEEK_N_F4_" not in output
    else:
        assert "DEEPSEEK_N_F2_" not in output


def test_natural_f2_requires_floor2_summary(tmp_path: Path) -> None:
    args = argparse.Namespace(
        lifecycle="natural_f2",
        common_epoch0_root=tmp_path,
        model_path=tmp_path,
        floor2_summary=None,
        floor4_summary=tmp_path / "floor4.json",
        floor8_summary=tmp_path / "floor8.json",
        floor16_summary=tmp_path / "floor16.json",
        output=tmp_path / "caps.env",
        block_size=128,
        rollout_n=16,
        target_ratio=1.0,
        runtime_profile="runtime-profile",
        runtime_profile_sha256="a" * 64,
        execution_code_sha256="b" * 64,
        probe_history_root=tmp_path / "probe_history",
    )
    with pytest.raises(ValueError, match="floor2-summary"):
        MODULE.generate(args)


def test_batch64_uses_four_prompt_reserve_and_shared_full16_cap(
    tmp_path: Path,
) -> None:
    common = tmp_path / "common"
    rollout = common / "epoch_000_mode0_probe" / "rollout_data"
    rollout.mkdir(parents=True)
    model = tmp_path / "model"
    model.mkdir()
    history = tmp_path / "probe_history"
    (history / "rollout_data").mkdir(parents=True)
    (history / "offline_planning_history.json").write_text(
        '{"schema_version": 1}\n', encoding="utf-8"
    )
    (history / "kv_probe_trigger_manifest.json").write_text(
        '{"schema_version": 1}\n', encoding="utf-8"
    )
    (history / "rollout_data" / "1.jsonl").write_text(
        '{"input": "trigger"}\n', encoding="utf-8"
    )
    (model / "config.json").write_text(
        json.dumps({"pad_token_id": None, "eos_token_id": 100001}),
        encoding="utf-8",
    )
    profile_id = "deepseek-v2-lite-chat-b64-n16-s5-v2"
    profile_sha256 = "d" * 64
    (common / "common_epoch0_metadata.env").write_text(
        "\n".join(
            [
                f"export COMMON_EPOCH0_MODEL_PATH={model}",
                "export COMMON_EPOCH0_MODEL_REVISION=revision",
                "export COMMON_EPOCH0_EXECUTION_PROFILE_USED=profile",
                "export COMMON_EPOCH0_TRAIN_FILE_USED=/data/deepscaler/train.parquet",
                "export COMMON_EPOCH0_TEST_FILE_USED=/data/deepscaler/test.parquet",
                "export COMMON_EPOCH0_DATASET_FRACTION_USED=0.01",
                "export COMMON_EPOCH0_TRAIN_BATCH_SIZE_USED=64",
                "export COMMON_EPOCH0_ROLLOUT_N_USED=16",
                "export COMMON_EPOCH0_MAX_PROMPT_LENGTH_USED=1024",
                "export COMMON_EPOCH0_MAX_RESPONSE_LENGTH_USED=16384",
                "export COMMON_EPOCH0_MAX_NUM_BATCHED_TOKENS_USED=17408",
                "export COMMON_EPOCH0_MAX_NUM_SEQS_USED=64",
                "export COMMON_EPOCH0_GPU_MEMORY_UTILIZATION_USED=0.9",
                "export COMMON_EPOCH0_KV_BLOCK_SIZE_USED=128",
                "export COMMON_EPOCH0_TRAIN_STEPS_USED=5",
                "export COMMON_EPOCH0_PROMPTS_TOTAL_USED=320",
                "export COMMON_EPOCH0_EXPECTED_RESPONSES_PER_STEP_USED=1024",
                "export COMMON_EPOCH0_PREEMPTION_POLICY_USED=record",
                f"export COMMON_EPOCH0_WORKLOAD_PROFILE_ID={profile_id}",
                "export COMMON_EPOCH0_WORKLOAD_PROFILE_SHA256="
                f"{profile_sha256}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (common / "MEASURED_GPU_KV_CACHE_TOKENS_PER_RANK").write_text(
        "614144\n", encoding="utf-8"
    )
    prompt_lengths = list(range(10, 330))
    repeated_prompt_sources = {64: 0, 128: 1, 192: 2}
    for step in range(5):
        rows = []
        for prompt_index in range(step * 64, (step + 1) * 64):
            prompt_source = repeated_prompt_sources.get(prompt_index, prompt_index)
            length = prompt_lengths[prompt_source]
            prompt_tokens = [100001] * (1024 - length) + list(range(length))
            for sample in range(16):
                rows.append(
                    json.dumps(
                        {
                            "input": f"prompt-{prompt_source}",
                            "prompts": prompt_tokens,
                            "sample": sample,
                        }
                    )
                )
        (rollout / f"{step + 1}.jsonl").write_text(
            "\n".join(rows) + "\n", encoding="utf-8"
        )

    capacities = {2: 400000, 4: 500096, 8: 650112, 16: 700032}
    summaries = {
        floor: _probe(
            tmp_path / f"floor{floor}.json",
            common,
            history,
            floor,
            physical,
            "natural_f2",
            max_num_seqs=64,
        )
        for floor, physical in capacities.items()
    }
    args = argparse.Namespace(
        lifecycle="natural_f2",
        common_epoch0_root=common,
        model_path=model,
        floor2_summary=summaries[2],
        floor4_summary=summaries[4],
        floor8_summary=summaries[8],
        floor16_summary=summaries[16],
        output=tmp_path / "caps.env",
        block_size=128,
        rollout_n=16,
        train_batch_size=64,
        max_num_seqs=64,
        dataset_fraction="0.01",
        common_steps=5,
        prompts_total=320,
        max_prompt_length=1024,
        max_response_length=16384,
        max_num_batched_tokens=17408,
        gpu_memory_utilization=0.9,
        world_size=16,
        workload_profile_id=profile_id,
        workload_profile_sha256=profile_sha256,
        common_preemption_policy="record",
        shared_full16_physical_tokens=614144,
        target_ratio=1.0,
        runtime_profile="runtime-profile",
        runtime_profile_sha256="a" * 64,
        execution_code_sha256="b" * 64,
        probe_history_root=history,
    )
    output = MODULE.generate(args)
    assert "export DEEPSEEK_KV_CAP_TRAIN_BATCH_SIZE=64" in output
    assert "export DEEPSEEK_KV_CAP_PROMPTS_PER_RANK=4" in output
    assert "export DEEPSEEK_KV_CAP_MAX_NUM_SEQS=64" in output
    assert "export DEEPSEEK_KV_CAP_EXPECTED_RESPONSES_PER_STEP=1024" in output
    assert "export DEEPSEEK_KV_CAP_PROMPT_RESERVE_TOKENS=20960" in output
    assert "export DEEPSEEK_KV_CAP_BLOCK_RESERVE_TOKENS=8320" in output
    assert "export DEEPSEEK_N_F2_KV_PROBED_PHYSICAL_FLOOR16=700032" in output
    assert "export DEEPSEEK_N_F2_KV_PHYSICAL_FLOOR16=614144" in output
    assert "export DEEPSEEK_N_F2_KV_ADMISSION_FLOOR16=584832" in output
    assert "export DEEPSEEK_VANILLA_KV_ADMISSION_TOKENS=584832" in output
    assert "export DEEPSEEK_KV_CAP_SHARED_FULL16_PHYSICAL_TOKENS=614144" in output
    assert f"export DEEPSEEK_KV_CAP_WORKLOAD_PROFILE_ID={profile_id}" in output
    assert (
        "export DEEPSEEK_KV_CAP_WORKLOAD_PROFILE_SHA256=" + profile_sha256
        in output
    )


def test_batch64_shared_full16_fails_when_natural_probe_is_smaller(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = argparse.Namespace(
        lifecycle="natural_f2",
        common_epoch0_root=tmp_path / "common",
        model_path=tmp_path / "model",
        floor2_summary=tmp_path / "floor2.json",
        floor4_summary=tmp_path / "floor4.json",
        floor8_summary=tmp_path / "floor8.json",
        floor16_summary=tmp_path / "floor16.json",
        output=tmp_path / "caps.env",
        block_size=128,
        rollout_n=16,
        train_batch_size=64,
        max_num_seqs=64,
        dataset_fraction="0.01",
        common_steps=5,
        prompts_total=320,
        max_prompt_length=1024,
        max_response_length=16384,
        max_num_batched_tokens=17408,
        gpu_memory_utilization=0.9,
        world_size=16,
        workload_profile_id="batch64",
        workload_profile_sha256="d" * 64,
        common_preemption_policy="record",
        shared_full16_physical_tokens=614144,
        target_ratio=1.0,
        runtime_profile="runtime-profile",
        runtime_profile_sha256="a" * 64,
        execution_code_sha256="b" * 64,
        probe_history_root=tmp_path / "history",
    )
    monkeypatch.setattr(
        MODULE,
        "load_env",
        lambda _path: {
            **MODULE.expected_common_protocol(MODULE.workload_protocol(args)),
            "COMMON_EPOCH0_MODEL_PATH": str(args.model_path),
            "COMMON_EPOCH0_MODEL_REVISION": "revision",
            "COMMON_EPOCH0_EXECUTION_PROFILE_USED": "profile",
        },
    )
    monkeypatch.setattr(Path, "is_file", lambda _path: True)
    monkeypatch.setattr(Path, "read_bytes", lambda _path: b"fixture")
    monkeypatch.setattr(
        MODULE,
        "load_probe",
        lambda path, floor, **_kwargs: {
            "observed_tokens": 600064 if floor == 16 else 400000,
            "probe_tail_guard_min_cap": 64,
            "probe_tail_guard_round_to": 64,
            "actual_plan_response_cap": 64,
            "plan_tail_guard_response_cap": 64,
            "planner_train_sha256": "c" * 64,
        },
    )
    monkeypatch.setattr(MODULE, "infer_pad_token_id", lambda _path: 0)
    monkeypatch.setattr(
        MODULE,
        "prompt_lengths",
        lambda *_args, **_kwargs: list(range(329, 9, -1)),
    )
    original_read_text = Path.read_text

    def fake_read_text(path: Path, *read_args, **read_kwargs) -> str:
        if path.name == "MEASURED_GPU_KV_CACHE_TOKENS_PER_RANK":
            return "614144\n"
        return original_read_text(path, *read_args, **read_kwargs)

    monkeypatch.setattr(Path, "read_text", fake_read_text)
    with pytest.raises(ValueError, match="below the shared Full16 cap"):
        MODULE.generate(args)


def test_rejects_floor_specific_actual_plan_response_caps() -> None:
    reports = {
        2: {"actual_plan_response_cap": 64},
        4: {"actual_plan_response_cap": 128},
    }
    with pytest.raises(ValueError, match="disagree on actual plan response cap"):
        MODULE.uniform_positive_probe_value(
            reports, "actual_plan_response_cap", "actual plan response cap"
        )


def test_rejects_common_epoch_with_different_rollout_protocol(
    tmp_path: Path,
) -> None:
    common = tmp_path / "common"
    common.mkdir()
    model = tmp_path / "model"
    model.mkdir()
    (common / "common_epoch0_metadata.env").write_text(
        "\n".join(
            f"export {name}={('8' if name == 'COMMON_EPOCH0_ROLLOUT_N_USED' else value)}"
            for name, value in MODULE.COMMON_PROTOCOL.items()
        )
        + f"\nexport COMMON_EPOCH0_MODEL_PATH={model}\n"
        "export COMMON_EPOCH0_MODEL_REVISION=revision\n"
        "export COMMON_EPOCH0_EXECUTION_PROFILE_USED=profile\n",
        encoding="utf-8",
    )
    args = argparse.Namespace(
        common_epoch0_root=common,
        model_path=model,
        floor4_summary=tmp_path / "floor4.json",
        floor8_summary=tmp_path / "floor8.json",
        floor16_summary=tmp_path / "floor16.json",
        output=tmp_path / "caps.env",
        block_size=128,
        rollout_n=16,
        target_ratio=1.0,
        runtime_profile="runtime-profile",
        runtime_profile_sha256="a" * 64,
        execution_code_sha256="b" * 64,
        probe_history_root=tmp_path / "probe_history",
    )
    with pytest.raises(
        ValueError,
        match="COMMON_EPOCH0_ROLLOUT_N_USED",
    ):
        MODULE.generate(args)


def test_rejects_probe_from_different_runtime_profile_content(tmp_path: Path) -> None:
    common = tmp_path / "common"
    common.mkdir()
    history = tmp_path / "probe_history"
    history.mkdir()
    (history / "offline_planning_history.json").write_text(
        '{"schema_version": 1}\n', encoding="utf-8"
    )
    (history / "kv_probe_trigger_manifest.json").write_text(
        '{"schema_version": 1}\n', encoding="utf-8"
    )
    (history / "rollout_data").mkdir()
    (history / "rollout_data" / "1.jsonl").write_text(
        '{"input": "trigger"}\n', encoding="utf-8"
    )
    probe = _probe(tmp_path / "floor4.json", common, history, 4, 400000)
    with pytest.raises(ValueError, match="runtime_profile_sha256"):
        MODULE.load_probe(
            probe,
            4,
            common_root=common.resolve(),
            model_revision="revision",
            execution_profile="profile",
            runtime_profile="runtime-profile",
            runtime_profile_sha256="b" * 64,
            execution_code_sha256="b" * 64,
            probe_history_root=history.resolve(),
            probe_history_sha256=__import__("hashlib").sha256(
                (history / "offline_planning_history.json").read_bytes()
            ).hexdigest(),
            probe_history_manifest_sha256=__import__("hashlib").sha256(
                (history / "kv_probe_trigger_manifest.json").read_bytes()
            ).hexdigest(),
            probe_trigger_subset_sha256=__import__("hashlib").sha256(
                (history / "rollout_data" / "1.jsonl").read_bytes()
            ).hexdigest(),
            block_size=128,
        )
