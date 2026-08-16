#!/usr/bin/env python3
"""Generate conservative, lifecycle-specific DeepSeek KV capacity settings."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shlex
from pathlib import Path


LIFECYCLE_CONFIG = {
    "natural_f4": {
        "floors": (4, 8, 16),
        "prefix": "DEEPSEEK_N_F4",
        "label": "Natural floor4",
    },
    "natural_f2": {
        "floors": (2, 4, 8, 16),
        "prefix": "DEEPSEEK_N_F2",
        "label": "Natural floor2",
    },
    "planned_f4": {
        "floors": (4, 8, 16),
        "prefix": "DEEPSEEK_P_F4",
        "label": "Planned floor4",
    },
    "planned_f2": {
        "floors": (2, 4, 8, 16),
        "prefix": "DEEPSEEK_P_F2",
        "label": "Planned floor2",
    },
}
LIFECYCLE_PREFIXES = tuple(
    config["prefix"] for config in LIFECYCLE_CONFIG.values()
)
COMMON_PROTOCOL = {
    "COMMON_EPOCH0_TRAIN_FILE_USED": "/data/deepscaler/train.parquet",
    "COMMON_EPOCH0_TEST_FILE_USED": "/data/deepscaler/test.parquet",
    "COMMON_EPOCH0_DATASET_FRACTION_USED": "0.005",
    "COMMON_EPOCH0_TRAIN_BATCH_SIZE_USED": "32",
    "COMMON_EPOCH0_ROLLOUT_N_USED": "16",
    "COMMON_EPOCH0_MAX_PROMPT_LENGTH_USED": "1024",
    "COMMON_EPOCH0_MAX_RESPONSE_LENGTH_USED": "16384",
    "COMMON_EPOCH0_MAX_NUM_BATCHED_TOKENS_USED": "17408",
    "COMMON_EPOCH0_MAX_NUM_SEQS_USED": "32",
    "COMMON_EPOCH0_GPU_MEMORY_UTILIZATION_USED": "0.9",
    "COMMON_EPOCH0_KV_BLOCK_SIZE_USED": "128",
    "COMMON_EPOCH0_TRAIN_STEPS_USED": "5",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lifecycle",
        choices=tuple(LIFECYCLE_CONFIG),
        default="natural_f4",
    )
    parser.add_argument("--common-epoch0-root", required=True, type=Path)
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--floor2-summary", type=Path)
    parser.add_argument("--floor4-summary", required=True, type=Path)
    parser.add_argument("--floor8-summary", required=True, type=Path)
    parser.add_argument("--floor16-summary", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--block-size",
        default=int(os.environ.get("VLLM_KV_BLOCK_SIZE", "128")),
        type=int,
    )
    parser.add_argument(
        "--rollout-n",
        default=int(os.environ.get("COMMON_EPOCH0_ROLLOUT_N", "16")),
        type=int,
    )
    parser.add_argument(
        "--train-batch-size",
        default=int(os.environ.get("COMMON_EPOCH0_TRAIN_BATCH_SIZE", "32")),
        type=int,
    )
    parser.add_argument(
        "--max-num-seqs",
        default=int(os.environ.get("COMMON_EPOCH0_MAX_NUM_SEQS", "32")),
        type=int,
    )
    parser.add_argument(
        "--dataset-fraction",
        default=os.environ.get("COMMON_EPOCH0_DATASET_FRACTION", "0.005"),
    )
    parser.add_argument(
        "--common-steps",
        default=int(os.environ.get("COMMON_EPOCH0_TRAIN_STEPS", "5")),
        type=int,
    )
    parser.add_argument(
        "--prompts-total",
        default=(
            int(os.environ["COMMON_EPOCH0_PROMPTS_TOTAL"])
            if "COMMON_EPOCH0_PROMPTS_TOTAL" in os.environ
            else None
        ),
        type=int,
    )
    parser.add_argument(
        "--max-prompt-length",
        default=int(os.environ.get("COMMON_EPOCH0_MAX_PROMPT_LENGTH", "1024")),
        type=int,
    )
    parser.add_argument(
        "--max-response-length",
        default=int(os.environ.get("COMMON_EPOCH0_MAX_RESPONSE_LENGTH", "16384")),
        type=int,
    )
    parser.add_argument(
        "--max-num-batched-tokens",
        default=int(
            os.environ.get("COMMON_EPOCH0_MAX_NUM_BATCHED_TOKENS", "17408")
        ),
        type=int,
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        default=float(
            os.environ.get("COMMON_EPOCH0_GPU_MEMORY_UTILIZATION", "0.9")
        ),
        type=float,
    )
    parser.add_argument("--world-size", default=16, type=int)
    parser.add_argument(
        "--workload-profile-id",
        default=os.environ.get("DEEPSEEK_WORKLOAD_PROFILE_ID"),
    )
    parser.add_argument(
        "--workload-profile-sha256",
        default=os.environ.get("DEEPSEEK_WORKLOAD_PROFILE_SHA256"),
    )
    parser.add_argument(
        "--common-preemption-policy",
        default=os.environ.get("COMMON_EPOCH0_PREEMPTION_POLICY", "forbid"),
        choices=("forbid", "record"),
    )
    parser.add_argument(
        "--shared-full16-physical-tokens",
        default=(
            int(os.environ["DEEPSEEK_SHARED_FULL16_PHYSICAL_TOKENS"])
            if "DEEPSEEK_SHARED_FULL16_PHYSICAL_TOKENS" in os.environ
            else None
        ),
        type=int,
    )
    parser.add_argument("--target-ratio", default=1.0, type=float)
    parser.add_argument("--runtime-profile", required=True)
    parser.add_argument("--runtime-profile-sha256", required=True)
    parser.add_argument("--execution-code-sha256", required=True)
    parser.add_argument("--probe-history-root", required=True, type=Path)
    parser.add_argument("--expected-plan-response-cap", type=int)
    parser.add_argument("--planned-headroom-floor2", type=int)
    parser.add_argument("--planned-headroom-floor4", type=int)
    parser.add_argument("--planned-headroom-floor8", type=int)
    parser.add_argument("--planned-headroom-floor16", type=int)
    parser.add_argument("--training-min-free-mib", type=int)
    parser.add_argument(
        "--merge-existing",
        action="store_true",
        help="replace only this lifecycle in an existing unified cap file",
    )
    return parser.parse_args()


def _arg(args: argparse.Namespace, name: str, default):
    value = getattr(args, name, None)
    return default if value is None else value


def _number_text(value: float | str) -> str:
    return format(float(value), ".15g")


def workload_protocol(args: argparse.Namespace) -> dict[str, int | float | str | None]:
    train_batch_size = int(_arg(args, "train_batch_size", 32))
    rollout_n = int(_arg(args, "rollout_n", 16))
    world_size = int(_arg(args, "world_size", 16))
    common_steps = int(_arg(args, "common_steps", 5))
    prompts_total = int(
        _arg(args, "prompts_total", common_steps * train_batch_size)
    )
    max_num_seqs = int(_arg(args, "max_num_seqs", 32))
    block_size = int(_arg(args, "block_size", 128))
    max_prompt_length = int(_arg(args, "max_prompt_length", 1024))
    max_response_length = int(_arg(args, "max_response_length", 16384))
    max_num_batched_tokens = int(
        _arg(args, "max_num_batched_tokens", 17408)
    )
    gpu_memory_utilization = float(
        _arg(args, "gpu_memory_utilization", 0.9)
    )
    dataset_fraction = _number_text(_arg(args, "dataset_fraction", "0.005"))
    workload_profile_id = _arg(args, "workload_profile_id", None)
    workload_profile_sha256 = _arg(args, "workload_profile_sha256", None)
    common_preemption_policy = str(
        _arg(args, "common_preemption_policy", "forbid")
    )
    if workload_profile_id == "unspecified":
        workload_profile_id = None
    if workload_profile_sha256 == "unspecified":
        workload_profile_sha256 = None

    positive = {
        "train batch size": train_batch_size,
        "rollout n": rollout_n,
        "world size": world_size,
        "common steps": common_steps,
        "prompts total": prompts_total,
        "max num seqs": max_num_seqs,
        "block size": block_size,
        "max prompt length": max_prompt_length,
        "max response length": max_response_length,
        "max num batched tokens": max_num_batched_tokens,
    }
    for label, value in positive.items():
        if value <= 0:
            raise ValueError(f"{label} must be positive, got {value}")
    if world_size != 16:
        raise ValueError(f"DeepSeek AdaFloor requires world size 16, got {world_size}")
    if train_batch_size % world_size:
        raise ValueError(
            f"train batch size {train_batch_size} is not divisible by world size "
            f"{world_size}"
        )
    if prompts_total != common_steps * train_batch_size:
        raise ValueError(
            "prompts total must equal common steps times train batch size"
        )
    if not 0.0 < float(dataset_fraction) <= 1.0:
        raise ValueError(f"dataset fraction must be in (0, 1], got {dataset_fraction}")
    if not 0.0 < gpu_memory_utilization <= 1.0:
        raise ValueError(
            "GPU memory utilization must be in (0, 1], got "
            f"{gpu_memory_utilization}"
        )
    if common_preemption_policy not in ("forbid", "record"):
        raise ValueError(
            "common preemption policy must be forbid or record, got "
            f"{common_preemption_policy!r}"
        )
    if workload_profile_id is None:
        if workload_profile_sha256 is not None:
            raise ValueError("workload profile SHA256 requires a profile ID")
    else:
        if re.fullmatch(r"[A-Za-z0-9._-]+", str(workload_profile_id)) is None:
            raise ValueError("invalid workload profile ID")
        if re.fullmatch(r"[0-9a-f]{64}", str(workload_profile_sha256 or "")) is None:
            raise ValueError("invalid workload profile SHA256")

    return {
        "train_batch_size": train_batch_size,
        "rollout_n": rollout_n,
        "world_size": world_size,
        "common_steps": common_steps,
        "prompts_total": prompts_total,
        "expected_responses_per_step": train_batch_size * rollout_n,
        "prompts_per_rank": train_batch_size // world_size,
        "max_num_seqs": max_num_seqs,
        "block_size": block_size,
        "max_prompt_length": max_prompt_length,
        "max_response_length": max_response_length,
        "max_num_batched_tokens": max_num_batched_tokens,
        "gpu_memory_utilization": gpu_memory_utilization,
        "dataset_fraction": dataset_fraction,
        "workload_profile_id": workload_profile_id,
        "workload_profile_sha256": workload_profile_sha256,
        "common_preemption_policy": common_preemption_policy,
    }


def expected_common_protocol(
    workload: dict[str, int | float | str | None],
) -> dict[str, str]:
    expected = {
        "COMMON_EPOCH0_TRAIN_FILE_USED": "/data/deepscaler/train.parquet",
        "COMMON_EPOCH0_TEST_FILE_USED": "/data/deepscaler/test.parquet",
        "COMMON_EPOCH0_DATASET_FRACTION_USED": str(workload["dataset_fraction"]),
        "COMMON_EPOCH0_TRAIN_BATCH_SIZE_USED": str(workload["train_batch_size"]),
        "COMMON_EPOCH0_ROLLOUT_N_USED": str(workload["rollout_n"]),
        "COMMON_EPOCH0_MAX_PROMPT_LENGTH_USED": str(workload["max_prompt_length"]),
        "COMMON_EPOCH0_MAX_RESPONSE_LENGTH_USED": str(
            workload["max_response_length"]
        ),
        "COMMON_EPOCH0_MAX_NUM_BATCHED_TOKENS_USED": str(
            workload["max_num_batched_tokens"]
        ),
        "COMMON_EPOCH0_MAX_NUM_SEQS_USED": str(workload["max_num_seqs"]),
        "COMMON_EPOCH0_GPU_MEMORY_UTILIZATION_USED": _number_text(
            workload["gpu_memory_utilization"]
        ),
        "COMMON_EPOCH0_KV_BLOCK_SIZE_USED": str(workload["block_size"]),
        "COMMON_EPOCH0_TRAIN_STEPS_USED": str(workload["common_steps"]),
    }
    if workload["workload_profile_id"] is not None:
        expected.update(
            {
                "COMMON_EPOCH0_PROMPTS_TOTAL_USED": str(workload["prompts_total"]),
                "COMMON_EPOCH0_EXPECTED_RESPONSES_PER_STEP_USED": str(
                    workload["expected_responses_per_step"]
                ),
                "COMMON_EPOCH0_WORKLOAD_PROFILE_ID": str(
                    workload["workload_profile_id"]
                ),
                "COMMON_EPOCH0_WORKLOAD_PROFILE_SHA256": str(
                    workload["workload_profile_sha256"]
                ),
                "COMMON_EPOCH0_PREEMPTION_POLICY_USED": str(
                    workload["common_preemption_policy"]
                ),
            }
        )
    return expected


def load_env(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:]
        name, separator, raw_value = line.partition("=")
        if not separator:
            raise ValueError(f"invalid environment line in {path}: {raw_line!r}")
        parsed = shlex.split(raw_value, posix=True)
        values[name] = parsed[0] if parsed else ""
    return values


def load_probe(
    path: Path,
    floor: int,
    *,
    common_root: Path,
    model_revision: str,
    execution_profile: str,
    runtime_profile: str,
    runtime_profile_sha256: str,
    execution_code_sha256: str,
    probe_history_root: Path,
    probe_history_sha256: str,
    probe_history_manifest_sha256: str,
    probe_trigger_subset_sha256: str,
    block_size: int,
    lifecycle: str = "natural_f4",
    world_size: int = 16,
    max_prompt_length: int = 1024,
    max_response_length: int = 16384,
    max_num_batched_tokens: int = 17408,
    max_num_seqs: int = 32,
    gpu_memory_utilization: float = 0.9,
) -> dict:
    report = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "lifecycle": lifecycle,
        "floor": floor,
        "world_size": world_size,
        "model_revision": model_revision,
        "execution_profile": execution_profile,
        "runtime_profile": runtime_profile,
        "runtime_profile_sha256": runtime_profile_sha256,
        "execution_code_sha256": execution_code_sha256,
        "max_prompt_length": max_prompt_length,
        "max_response_length": max_response_length,
        "max_num_batched_tokens": max_num_batched_tokens,
        "max_num_seqs": max_num_seqs,
        "gpu_memory_utilization": gpu_memory_utilization,
        "enforce_eager": "True",
        "block_size": block_size,
    }
    for key, value in expected.items():
        if report.get(key) != value:
            raise ValueError(
                f"{path}: {key}={report.get(key)!r}, expected {value!r}"
            )
    if Path(report.get("common_epoch0_root", "")).resolve() != common_root:
        raise ValueError(f"{path}: common epoch0 provenance mismatch")
    if Path(report.get("planning_history_root", "")).resolve() != probe_history_root:
        raise ValueError(f"{path}: planning history provenance mismatch")
    if report.get("planning_history_sha256") != probe_history_sha256:
        raise ValueError(f"{path}: planning history SHA256 mismatch")
    if (
        report.get("planning_history_manifest_sha256")
        != probe_history_manifest_sha256
    ):
        raise ValueError(f"{path}: planning history manifest SHA256 mismatch")
    if report.get("planning_trigger_subset_sha256") != probe_trigger_subset_sha256:
        raise ValueError(f"{path}: planning trigger subset SHA256 mismatch")
    if re.fullmatch(r"[0-9a-f]{64}", str(report.get("planner_train_sha256", ""))) is None:
        raise ValueError(f"{path}: invalid planner train artifact SHA256")
    if int(report.get("complete_target_waves", 0)) < 1:
        raise ValueError(f"{path}: no complete target resize wave")
    per_rank = report.get("per_rank_tokens")
    if not isinstance(per_rank, dict) or len(per_rank) != world_size:
        raise ValueError(f"{path}: expected capacities for {world_size} ranks")
    physical = int(report.get("observed_tokens", 0))
    rank_minimum = min(int(value) for value in per_rank.values())
    if physical != rank_minimum or physical <= 0 or physical % block_size:
        raise ValueError(f"{path}: invalid physical capacity {physical}")
    return report


def infer_pad_token_id(model_path: Path) -> int:
    config = json.loads((model_path / "config.json").read_text(encoding="utf-8"))
    pad_token_id = config.get("pad_token_id")
    if pad_token_id is None:
        pad_token_id = config.get("eos_token_id")
    if isinstance(pad_token_id, list):
        if len(pad_token_id) != 1:
            raise ValueError("model config has ambiguous pad/eos token IDs")
        pad_token_id = pad_token_id[0]
    if not isinstance(pad_token_id, int):
        raise ValueError("model config has no usable pad/eos token ID")
    return pad_token_id


def prompt_lengths(
    common_root: Path,
    pad_token_id: int,
    rollout_n: int,
    *,
    prompts_total: int = 160,
    common_steps: int = 5,
) -> list[int]:
    rollout_dir = common_root / "epoch_000_mode0_probe" / "rollout_data"
    files = sorted(rollout_dir.glob("*.jsonl"), key=lambda path: int(path.stem))
    if len(files) != common_steps:
        raise ValueError(
            f"{rollout_dir}: expected {common_steps} rollout JSONL files"
        )
    if prompts_total % common_steps:
        raise ValueError(
            f"prompts_total={prompts_total} is not divisible by "
            f"common_steps={common_steps}"
        )
    prompts_per_step = prompts_total // common_steps
    lengths: list[int] = []
    for path in files:
        step_rows: list[tuple[str, int]] = []
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                tokens = row.get("prompts")
                prompt = row.get("input")
                if not isinstance(tokens, list) or not tokens or not isinstance(prompt, str):
                    raise ValueError(f"{path}: rollout row lacks prompt tokens")
                first_valid = next(
                    (index for index, token in enumerate(tokens) if token != pad_token_id),
                    len(tokens),
                )
                length = len(tokens) - first_valid
                if length <= 0:
                    raise ValueError(f"{path}: empty tokenized prompt")
                step_rows.append((prompt, length))
        expected_step_rows = prompts_per_step * rollout_n
        if len(step_rows) != expected_step_rows:
            raise ValueError(
                f"{path}: expected {expected_step_rows} rows, got {len(step_rows)}"
            )
        seen_step_prompts: set[str] = set()
        for offset in range(0, len(step_rows), rollout_n):
            occurrence = step_rows[offset:offset + rollout_n]
            prompt = occurrence[0][0]
            if any(value[0] != prompt for value in occurrence[1:]):
                raise ValueError(
                    f"{path}: rows {offset}:{offset + rollout_n} do not "
                    "belong to one prompt occurrence"
                )
            if prompt in seen_step_prompts:
                raise ValueError(f"{path}: duplicate prompt occurrence within one step")
            seen_step_prompts.add(prompt)
            length = occurrence[0][1]
            if any(value[1] != length for value in occurrence[1:]):
                raise ValueError(
                    f"{path}: inconsistent token count for one prompt occurrence"
                )
            lengths.append(length)
        if len(seen_step_prompts) != prompts_per_step:
            raise ValueError(
                f"{path}: expected {prompts_per_step} prompt occurrences, "
                f"got {len(seen_step_prompts)}"
            )
    if len(lengths) != prompts_total:
        raise ValueError(
            f"{rollout_dir}: expected {prompts_total} prompt occurrences, "
            f"got {len(lengths)}"
        )
    return sorted(lengths, reverse=True)


def floor_to_block(value: float, block_size: int) -> int:
    return math.floor(value / block_size) * block_size


def shell_value(value: str | int | float | Path) -> str:
    return shlex.quote(str(value))


def uniform_positive_probe_value(
    reports: dict[int, dict], key: str, label: str
) -> int:
    values: dict[int, int] = {}
    for floor, report in reports.items():
        raw_value = report.get(key)
        if isinstance(raw_value, bool) or not isinstance(raw_value, int):
            raise ValueError(f"floor{floor}: invalid {label} {raw_value!r}")
        if raw_value <= 0:
            raise ValueError(f"floor{floor}: invalid {label} {raw_value}")
        values[floor] = raw_value
    unique = set(values.values())
    if len(unique) != 1:
        raise ValueError(f"probe floors disagree on {label}: {values}")
    return next(iter(unique))


def generate(args: argparse.Namespace) -> str:
    common_root = args.common_epoch0_root.resolve()
    workload = workload_protocol(args)
    block_size = int(workload["block_size"])
    rollout_n = int(workload["rollout_n"])
    target_ratio = float(_arg(args, "target_ratio", 1.0))
    lifecycle = getattr(args, "lifecycle", "natural_f4")
    if lifecycle not in LIFECYCLE_CONFIG:
        raise ValueError(f"unsupported DeepSeek lifecycle {lifecycle!r}")
    lifecycle_config = LIFECYCLE_CONFIG[lifecycle]
    floors = lifecycle_config["floors"]
    prefix = lifecycle_config["prefix"]
    label = lifecycle_config["label"]
    floor2_summary = getattr(args, "floor2_summary", None)
    if 2 in floors and floor2_summary is None:
        raise ValueError(f"{lifecycle} requires --floor2-summary")
    if block_size <= 0 or rollout_n <= 0:
        raise ValueError("block size and rollout n must be positive")
    if target_ratio != 1.0:
        raise ValueError("strict KV-safe generation requires target ratio 1.0")
    if not args.runtime_profile.strip():
        raise ValueError(f"{label} runtime profile must be nonempty")
    if re.fullmatch(r"[0-9a-f]{64}", args.runtime_profile_sha256) is None:
        raise ValueError(f"{label} runtime profile SHA256 is invalid")
    if re.fullmatch(r"[0-9a-f]{64}", args.execution_code_sha256) is None:
        raise ValueError("DeepSeek execution code SHA256 is invalid")
    metadata = load_env(common_root / "common_epoch0_metadata.env")
    for name, expected in expected_common_protocol(workload).items():
        recorded = metadata.get(name)
        if name.endswith("_FILE_USED"):
            matches = recorded is not None and Path(recorded).resolve() == Path(
                expected
            ).resolve()
        else:
            matches = recorded == expected
        if not matches:
            raise ValueError(
                f"common epoch0 protocol mismatch for {name}: "
                f"recorded={recorded!r}, expected={expected!r}"
            )
    model_revision = metadata["COMMON_EPOCH0_MODEL_REVISION"]
    execution_profile = metadata["COMMON_EPOCH0_EXECUTION_PROFILE_USED"]
    if Path(metadata["COMMON_EPOCH0_MODEL_PATH"]).resolve() != args.model_path.resolve():
        raise ValueError("common epoch0 model path does not match --model-path")
    probe_history_root = args.probe_history_root.resolve()
    probe_history_file = probe_history_root / "offline_planning_history.json"
    probe_history_manifest = probe_history_root / "kv_probe_trigger_manifest.json"
    probe_trigger_subset = probe_history_root / "rollout_data" / "1.jsonl"
    if not probe_history_file.is_file():
        raise ValueError(f"missing DeepSeek probe history: {probe_history_file}")
    for path in (probe_history_manifest, probe_trigger_subset):
        if not path.is_file():
            raise ValueError(f"missing DeepSeek probe trigger artifact: {path}")
    probe_history_sha256 = hashlib.sha256(probe_history_file.read_bytes()).hexdigest()
    probe_history_manifest_sha256 = hashlib.sha256(
        probe_history_manifest.read_bytes()
    ).hexdigest()
    probe_trigger_subset_sha256 = hashlib.sha256(
        probe_trigger_subset.read_bytes()
    ).hexdigest()

    summaries: dict[int, Path] = {
        4: args.floor4_summary.resolve(),
        8: args.floor8_summary.resolve(),
        16: args.floor16_summary.resolve(),
    }
    if 2 in floors:
        assert floor2_summary is not None
        summaries[2] = floor2_summary.resolve()
    reports = {
        floor: load_probe(
            path,
            floor,
            common_root=common_root,
            model_revision=model_revision,
            execution_profile=execution_profile,
            runtime_profile=args.runtime_profile,
            runtime_profile_sha256=args.runtime_profile_sha256,
            execution_code_sha256=args.execution_code_sha256,
            probe_history_root=probe_history_root,
            probe_history_sha256=probe_history_sha256,
            probe_history_manifest_sha256=probe_history_manifest_sha256,
            probe_trigger_subset_sha256=probe_trigger_subset_sha256,
            block_size=block_size,
            lifecycle=lifecycle,
            world_size=int(workload["world_size"]),
            max_prompt_length=int(workload["max_prompt_length"]),
            max_response_length=int(workload["max_response_length"]),
            max_num_batched_tokens=int(workload["max_num_batched_tokens"]),
            max_num_seqs=int(workload["max_num_seqs"]),
            gpu_memory_utilization=float(workload["gpu_memory_utilization"]),
        )
        for floor, path in summaries.items()
    }
    probe_tail_guard_min_cap = uniform_positive_probe_value(
        reports, "probe_tail_guard_min_cap", "TailGuard min cap"
    )
    probe_tail_guard_round_to = uniform_positive_probe_value(
        reports, "probe_tail_guard_round_to", "TailGuard round-to"
    )
    actual_plan_response_cap = uniform_positive_probe_value(
        reports, "actual_plan_response_cap", "actual plan response cap"
    )
    if (
        getattr(args, "expected_plan_response_cap", None) is not None
        and actual_plan_response_cap != args.expected_plan_response_cap
    ):
        raise ValueError(
            "probe plan response cap mismatch: "
            f"observed={actual_plan_response_cap} "
            f"expected={args.expected_plan_response_cap}"
        )
    expected_plan_response_cap = getattr(args, "expected_plan_response_cap", None)
    if (
        expected_plan_response_cap is not None
        and actual_plan_response_cap != expected_plan_response_cap
    ):
        raise ValueError(
            f"actual plan response cap {actual_plan_response_cap} does not match "
            f"the expected probe protocol cap {expected_plan_response_cap}"
        )
    for floor, report in reports.items():
        if report.get("plan_tail_guard_response_cap") != actual_plan_response_cap:
            raise ValueError(
                f"floor{floor}: plan response cap fields disagree in probe summary"
            )
    if actual_plan_response_cap > int(workload["max_response_length"]):
        raise ValueError(
            f"actual plan response cap exceeds max response length: "
            f"{actual_plan_response_cap}"
        )
    pad_token_id = infer_pad_token_id(args.model_path)
    lengths = prompt_lengths(
        common_root,
        pad_token_id,
        rollout_n,
        prompts_total=int(workload["prompts_total"]),
        common_steps=int(workload["common_steps"]),
    )
    prompts_per_rank = int(workload["prompts_per_rank"])
    prompt_reserve = rollout_n * sum(lengths[:prompts_per_rank])
    block_reserve = (prompts_per_rank * rollout_n + 1) * block_size
    probed_physical = {
        floor: int(reports[floor]["observed_tokens"]) for floor in floors
    }
    physical = dict(probed_physical)

    measured_path = common_root / "MEASURED_GPU_KV_CACHE_TOKENS_PER_RANK"
    vanilla_physical = int(measured_path.read_text(encoding="utf-8").strip())
    if vanilla_physical <= 0 or vanilla_physical % block_size:
        raise ValueError(f"invalid common epoch0 physical KV capacity {vanilla_physical}")
    shared_full16 = _arg(args, "shared_full16_physical_tokens", None)
    if shared_full16 is not None:
        shared_full16 = int(shared_full16)
        if not lifecycle.startswith("natural_"):
            raise ValueError(
                "shared Full16 physical cap is only valid for Natural lifecycles"
            )
        if shared_full16 <= 0 or shared_full16 % block_size:
            raise ValueError(
                "shared Full16 physical cap must be a positive block multiple"
            )
        if shared_full16 != vanilla_physical:
            raise ValueError(
                "shared Full16 physical cap must equal the common epoch0 "
                f"Vanilla capacity {vanilla_physical}, got {shared_full16}"
            )
        if probed_physical[16] < shared_full16:
            raise ValueError(
                "Natural floor16 probe capacity is below the shared Full16 cap: "
                f"probe={probed_physical[16]} shared={shared_full16}"
            )
        physical[16] = shared_full16
    planned_headroom: dict[int, int] = {floor: 0 for floor in floors}
    training_min_free_mib: int | None = None
    if lifecycle.startswith("planned_"):
        for floor in floors:
            value = getattr(args, f"planned_headroom_floor{floor}", None)
            if value is None or value < 0 or value % block_size:
                raise ValueError(
                    f"{label} floor{floor} headroom must be an explicit "
                    f"nonnegative multiple of {block_size}"
                )
            planned_headroom[floor] = value
        training_min_free_mib = getattr(args, "training_min_free_mib", None)
        if training_min_free_mib is None or training_min_free_mib <= 0:
            raise ValueError(
                f"{label} requires an explicit positive training minimum free MiB"
            )
    elif any(
        getattr(args, f"planned_headroom_floor{floor}", None) is not None
        for floor in (2, 4, 8, 16)
    ) or getattr(args, "training_min_free_mib", None) is not None:
        raise ValueError("planned memory arguments are invalid for a Natural lifecycle")

    admission = {
        floor: floor_to_block(
            target_ratio * physical[floor]
            - prompt_reserve
            - block_reserve
            - planned_headroom[floor],
            block_size,
        )
        for floor in floors
    }
    for floor in floors:
        if admission[floor] <= 0 or admission[floor] >= physical[floor]:
            raise ValueError(
                f"floor{floor}: invalid admission={admission[floor]} "
                f"for physical={physical[floor]}"
            )

    vanilla_admission = floor_to_block(
        vanilla_physical - prompt_reserve - block_reserve,
        block_size,
    )
    if vanilla_admission <= 0 or vanilla_admission >= vanilla_physical:
        raise ValueError(
            f"invalid common epoch0 admission KV capacity {vanilla_admission}"
        )

    values: list[tuple[str, str | int | float | Path]] = [
        ("DEEPSEEK_KV_CAPS_VERIFIED", 0),
        (f"{prefix}_KV_CAPS_VERIFIED", 0),
        ("DEEPSEEK_KV_CAP_MODEL_REVISION", model_revision),
        ("DEEPSEEK_KV_CAP_EXECUTION_PROFILE", execution_profile),
        (f"{prefix}_RUNTIME_PROFILE", args.runtime_profile),
        (f"{prefix}_RUNTIME_PROFILE_SHA256", args.runtime_profile_sha256),
        ("DEEPSEEK_EXECUTION_CODE_SHA256", args.execution_code_sha256),
        ("DEEPSEEK_KV_CAP_TRAIN_BATCH_SIZE", workload["train_batch_size"]),
        ("DEEPSEEK_KV_CAP_PROMPTS_PER_RANK", prompts_per_rank),
        ("DEEPSEEK_KV_CAP_COMMON_STEPS", workload["common_steps"]),
        ("DEEPSEEK_KV_CAP_PROMPTS_TOTAL", workload["prompts_total"]),
        (
            "DEEPSEEK_KV_CAP_EXPECTED_RESPONSES_PER_STEP",
            workload["expected_responses_per_step"],
        ),
        ("DEEPSEEK_KV_CAP_DATASET_FRACTION", workload["dataset_fraction"]),
        ("DEEPSEEK_KV_CAP_MAX_PROMPT_LENGTH", workload["max_prompt_length"]),
        ("DEEPSEEK_KV_CAP_MAX_RESPONSE_LENGTH", workload["max_response_length"]),
        (
            "DEEPSEEK_KV_CAP_MAX_NUM_BATCHED_TOKENS",
            workload["max_num_batched_tokens"],
        ),
        ("DEEPSEEK_KV_CAP_MAX_NUM_SEQS", workload["max_num_seqs"]),
        ("DEEPSEEK_KV_CAP_PROBE_TAIL_GUARD_MIN_CAP", probe_tail_guard_min_cap),
        ("DEEPSEEK_KV_CAP_PROBE_TAIL_GUARD_ROUND_TO", probe_tail_guard_round_to),
        ("DEEPSEEK_KV_CAP_PROBE_PLAN_RESPONSE_CAP", actual_plan_response_cap),
        (
            "DEEPSEEK_KV_CAP_GPU_MEMORY_UTILIZATION",
            workload["gpu_memory_utilization"],
        ),
        ("DEEPSEEK_KV_CAP_ENFORCE_EAGER", "True"),
        ("DEEPSEEK_KV_CAP_BLOCK_SIZE", block_size),
        ("DEEPSEEK_KV_CAP_ROLLOUT_N", rollout_n),
        ("DEEPSEEK_KV_CAP_TARGET_RATIO", target_ratio),
        ("DEEPSEEK_KV_CAP_PROMPT_RESERVE_TOKENS", prompt_reserve),
        ("DEEPSEEK_KV_CAP_BLOCK_RESERVE_TOKENS", block_reserve),
        ("DEEPSEEK_KV_CAP_COMMON_EPOCH0_ROOT", common_root),
        ("DEEPSEEK_KV_CAP_PROBE_HISTORY_ROOT", probe_history_root),
        ("DEEPSEEK_KV_CAP_PROBE_HISTORY_SHA256", probe_history_sha256),
        (
            "DEEPSEEK_KV_CAP_PROBE_HISTORY_MANIFEST_SHA256",
            probe_history_manifest_sha256,
        ),
        ("DEEPSEEK_KV_CAP_PROBE_TRIGGER_SUBSET_SHA256", probe_trigger_subset_sha256),
        ("DEEPSEEK_VANILLA_KV_PHYSICAL_TOKENS", vanilla_physical),
        ("DEEPSEEK_VANILLA_KV_ADMISSION_TOKENS", vanilla_admission),
    ]
    if workload["workload_profile_id"] is not None:
        values.extend(
            [
                (
                    "DEEPSEEK_KV_CAP_WORKLOAD_PROFILE_ID",
                    workload["workload_profile_id"],
                ),
                (
                    "DEEPSEEK_KV_CAP_WORKLOAD_PROFILE_SHA256",
                    workload["workload_profile_sha256"],
                ),
                (
                    "DEEPSEEK_KV_CAP_COMMON_PREEMPTION_POLICY",
                    workload["common_preemption_policy"],
                ),
            ]
        )
    if shared_full16 is not None:
        values.append(
            ("DEEPSEEK_KV_CAP_SHARED_FULL16_PHYSICAL_TOKENS", shared_full16)
        )
    for floor in floors:
        values.extend(
            [
                (f"{prefix}_KV_ADMISSION_FLOOR{floor}", admission[floor]),
                (f"{prefix}_KV_PHYSICAL_FLOOR{floor}", physical[floor]),
                (
                    f"{prefix}_KV_PROBED_PHYSICAL_FLOOR{floor}",
                    probed_physical[floor],
                ),
                (f"{prefix}_KV_PROBE_FLOOR{floor}", summaries[floor]),
                (
                    f"{prefix}_KV_PROBE_PLANNER_TRAIN_SHA256_FLOOR{floor}",
                    reports[floor]["planner_train_sha256"],
                ),
            ]
        )
        if lifecycle.startswith("planned_"):
            values.append(
                (f"{prefix}_HEADROOM_FLOOR{floor}", planned_headroom[floor])
            )
    if training_min_free_mib is not None:
        values.append((f"{prefix}_TRAINING_MIN_FREE_MIB", training_min_free_mib))
    values.append(("DEEPSEEK_EXPECTED_DISTCP_SHARDS", "auto"))
    lines = [
        f"# Generated from DeepSeek-V2-Lite common epoch0 and {label} probes.",
        f"# Keep {prefix.removeprefix('DEEPSEEK_')} VERIFIED=0 until strict "
        "per-floor KV cap authorization passes.",
    ]
    lines.extend(f"export {name}={shell_value(value)}" for name, value in values)
    return "\n".join(lines) + "\n"


def _assignment_name(line: str) -> str | None:
    stripped = line.strip()
    if stripped.startswith("export "):
        stripped = stripped[7:].strip()
    name, separator, _value = stripped.partition("=")
    if not separator or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
        return None
    return name


def merge_existing(existing: str, generated: str, prefix: str) -> str:
    generated_lines = generated.rstrip().splitlines()
    generated_names = {
        name for line in generated_lines if (name := _assignment_name(line))
    }
    existing_values = load_env_text(existing)
    generated_values = load_env_text(generated)
    code_changed = (
        bool(existing_values.get("DEEPSEEK_EXECUTION_CODE_SHA256"))
        and existing_values.get("DEEPSEEK_EXECUTION_CODE_SHA256")
        != generated_values.get("DEEPSEEK_EXECUTION_CODE_SHA256")
    )

    kept: list[str] = []
    for line in existing.rstrip().splitlines():
        name = _assignment_name(line)
        if name is None:
            kept.append(line)
            continue
        selected_metadata = name.startswith(f"{prefix}_KV_CAP_VALIDATION")
        if name in generated_names or name.startswith(f"{prefix}_") or selected_metadata:
            continue
        if code_changed and any(name.startswith(f"{other}_KV_CAP_VALIDATION") for other in LIFECYCLE_PREFIXES):
            continue
        if code_changed and name in {
            f"{other}_KV_CAPS_VERIFIED" for other in LIFECYCLE_PREFIXES
        }:
            kept.append(f"export {name}=0")
            continue
        kept.append(line)
    while kept and not kept[-1].strip():
        kept.pop()
    return "\n".join(kept + ["", *generated_lines]) + "\n"


def load_env_text(text: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw_line in text.splitlines():
        name = _assignment_name(raw_line)
        if name is None:
            continue
        raw_value = raw_line.strip()
        if raw_value.startswith("export "):
            raw_value = raw_value[7:].strip()
        raw_value = raw_value.split("=", 1)[1]
        parsed = shlex.split(raw_value, posix=True)
        values[name] = parsed[0] if parsed else ""
    return values


def main() -> int:
    args = parse_args()
    output = generate(args)
    prefix = LIFECYCLE_CONFIG[args.lifecycle]["prefix"]
    if args.merge_existing and args.output.is_file():
        output = merge_existing(
            args.output.read_text(encoding="utf-8"), output, prefix
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(output, encoding="utf-8")
    temporary.replace(args.output)
    print(f"wrote candidate DeepSeek KV caps to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
