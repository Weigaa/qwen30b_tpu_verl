#!/usr/bin/env python3
"""Fail-closed verification for the Qwen2.5-1.5B sidecar paired experiment.

The experiment contract is intentionally narrow.  Seeds 101, 202, and 303
each contain one sidecar-off arm and one sidecar-on arm.  Every arm executes
one Planned floor4 step without TailGuard and produces 512 primary responses.
The verifier accepts results only when the paired request identities, response
lengths, and executed plan match exactly.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
import tempfile
from collections import Counter
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from verl.utils.reward_score import gsm8k  # noqa: E402


SEEDS = (101, 202, 303)
ARMS = ("off", "on")
EXPECTED_LAUNCH_ORDER = {
    (101, "off"): 1,
    (101, "on"): 2,
    (202, "on"): 3,
    (202, "off"): 4,
    (303, "off"): 5,
    (303, "on"): 6,
}
WORLD_SIZE = 16
TARGET_FLOOR = 4
SIDECAR_TRIGGER_FLOOR = 8
PRIMARY_HCCL_ALLOCATOR_START = 12000
SIDECAR_MODEL_PATH = Path("/data/Qwen2.5-1.5B-Instruct")
SIDECAR_MODEL_REVISION = "a3c2dc17129625b1e51caf21ab486d32d1f12982"
SIDECAR_MODEL_WEIGHTS_SHA256 = (
    "dd924a11b4c220f385b51ffa522daea7c9f3d850e31b162bb5661df483c6d3ee"
)
EXPECTED_RESPONSES = 512
MAX_RESPONSE_LENGTH = 16384
T_CRITICAL_95_DF2 = 4.302652729911275
MANIFEST_NAME = "sidecar_pair_manifest.json"
SUMMARY_JSON = "sidecar_pair_summary.json"
SUMMARY_MD = "sidecar_pair_summary.md"
CODE_PROVENANCE_NAME = "code_sha256.txt"
PROTOCOL_NAME = "protocol.env"
RUN_MANIFEST_NAME = "run_manifest.tsv"
RUN_MANIFEST_HEADER = "seed\tarm\tlaunch_order\trun_dir\tstatus"

ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")
FLOAT = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
ROLLOUT_TIME_RE = re.compile(r"rollout_output_time_s:\s*(%s)" % FLOAT)
ABORT_RE = re.compile(r"response/aborted_ratio:([0-9.eE+-]+)")
SHRINK_RE = re.compile(
    r"Elastic parallel shrink done:.*?active_ranks=\[([^]]*)\]"
)
PREEMPT_RE = re.compile(r"\bPreempting request\b|\brequest preempted\b", re.I)
OOM_RE = re.compile(
    r"out of memory|OutOfMemoryError|NPU memory is exhausted|"
    r"ACL_ERROR_RT_MEMORY_ALLOCATION",
    re.I,
)
TAIL_GUARD_LINE_RE = re.compile(r"(?m)^.*Shrink-aware tail-guard response cap[^\n]*$")
MOE_SHARED_EXPERT_OVERLAP_RE = re.compile(
    r"\bmoe_shared_expert_overlap=(True|False)\b"
)
SHA256_RE = re.compile(r"[0-9a-f]{64}")
PROMPT_HASH_RE = re.compile(r"[0-9a-f]{32}")
GENERATE_DONE_RE = re.compile(
    r"Mode1 step timeline: driver_generate_done\b[^\n]*?"
    r"driver_generate_done_time=(%s)" % FLOAT
)
RESTORE_START_RE = re.compile(
    r"Mode1 step timeline: driver_restore_rpc_start\b[^\n]*?"
    r"driver_restore_rpc_start_time=(%s)" % FLOAT
)
PRIMARY_ACK_RE = re.compile(
    r"Sidecar pre-restore release acknowledged:\s*request_id=(\S+)\s+lease_id=(\S+)"
)


class VerificationError(RuntimeError):
    """Raised when an artifact does not prove the experiment contract."""


def _fail(message: str) -> None:
    raise VerificationError(message)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _strip_ansi(text: str) -> str:
    return ANSI_RE.sub("", text).replace("\r", "")


def _read_json(path: Path, label: str) -> Any:
    if not path.is_file():
        _fail(f"missing {label}: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise VerificationError(f"invalid {label} JSON: {path}: {exc}") from exc


def _read_jsonl(path: Path, label: str) -> list[dict[str, Any]]:
    if not path.is_file():
        _fail(f"missing {label}: {path}")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, 1):
            line = raw.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise VerificationError(
                    f"invalid JSON at {path}:{line_number}: {exc}"
                ) from exc
            if not isinstance(row, dict):
                _fail(f"{path}:{line_number} is not a JSON object")
            rows.append(row)
    return rows


def _only(paths: Iterable[Path], label: str) -> Path:
    items = sorted(paths)
    if len(items) != 1:
        _fail(f"expected exactly one {label}, found {len(items)}")
    return items[0]


def _required(manifest: dict[str, Any], key: str, expected: Any | None = None) -> Any:
    if key not in manifest:
        _fail(f"manifest does not define {key}")
    value = manifest[key]
    if expected is not None and value != expected:
        _fail(f"manifest requires {key}={expected!r}, got {value!r}")
    return value


def _resolve_recorded_path(value: Any, arm_dir: Path, label: str) -> Path:
    if not isinstance(value, str) or not value:
        _fail(f"manifest {label} must be a nonempty path")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = arm_dir / path
    return path.resolve()


def _load_code_provenance(root: Path) -> Path:
    path = root / CODE_PROVENANCE_NAME
    if not path.is_file():
        _fail(f"missing code provenance: {path}")
    entries: set[Path] = set()
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines:
        _fail("code provenance is empty")
    for line_number, line in enumerate(lines, 1):
        match = re.fullmatch(r"([0-9a-f]{64})  (.+)", line)
        if match is None:
            _fail(f"{path}:{line_number} is not sha256sum output")
        source = Path(match.group(2)).expanduser()
        if not source.is_absolute():
            _fail(f"{path}:{line_number} does not record an absolute source path")
        canonical = source.resolve()
        if canonical in entries:
            _fail(f"code provenance repeats source path {canonical}")
        entries.add(canonical)
    return path


def _load_protocol(root: Path) -> tuple[Path, dict[str, str]]:
    path = root / PROTOCOL_NAME
    protocol = _read_fields(path, "root protocol")
    fixed = {
        "seeds": "101 202 303",
        "orders": "101:off,on 202:on,off 303:off,on",
        "planned_residency": "true",
        "floor": "floor4",
        "tail_guard": "false",
        "actor_frozen": "true",
        "paired_request_sampling_seeds": "true",
        "planner_prompts": "160",
        "plan_steps": "5",
        "executed_prompts": "32",
        "steps_per_run": "1",
        "fast_step_subset": "false",
        "source_plan_step": "1",
        "sidecar_trigger_active_ranks": "8",
        "sidecar_model_revision": SIDECAR_MODEL_REVISION,
        "sidecar_parallelism": "TP1x8",
        "sidecar_stop_ack_timeout_seconds": "60",
        "sidecar_require_active_lease_before_restore": "true",
        "sidecar_require_shrink_quorum": "true",
        "sidecar_shrink_quorum_size": "16",
        "eager_weight_sync_group_init": "false",
        "primary_hccl_allocator_start": str(PRIMARY_HCCL_ALLOCATOR_START),
        "primary_moe_shared_expert_overlap": "false",
    }
    for key, expected in fixed.items():
        if protocol.get(key) != expected:
            _fail(f"root protocol requires {key}={expected!r}, got {protocol.get(key)!r}")
    for key in ("created_at_utc", "common_epoch0_root", "sidecar_model", "sidecar_data"):
        if not protocol.get(key):
            _fail(f"root protocol does not define {key}")
    if Path(protocol["sidecar_model"]).expanduser().resolve() != SIDECAR_MODEL_PATH.resolve():
        _fail("root protocol sidecar model path differs from the frozen model")
    return path, protocol


def _load_run_manifest(
    root: Path,
    *,
    allow_incomplete: bool,
) -> tuple[Path, list[dict[str, Any]]]:
    path = root / RUN_MANIFEST_NAME
    if not path.is_file():
        _fail(f"missing run manifest: {path}")
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines or lines[0] != RUN_MANIFEST_HEADER:
        _fail(f"run manifest header must be exactly {RUN_MANIFEST_HEADER!r}")
    expected_order = sorted(EXPECTED_LAUNCH_ORDER, key=EXPECTED_LAUNCH_ORDER.get)
    rows: list[dict[str, Any]] = []
    seen: set[tuple[int, str]] = set()
    for line_number, line in enumerate(lines[1:], 2):
        columns = line.split("\t")
        if len(columns) != 5:
            _fail(f"{path}:{line_number} must contain exactly five tab-separated fields")
        seed_raw, arm, order_raw, run_dir_raw, status = columns
        try:
            seed = int(seed_raw)
            order = int(order_raw)
        except ValueError as exc:
            raise VerificationError(
                f"{path}:{line_number} has a non-integer seed or launch order"
            ) from exc
        key = (seed, arm)
        if key in seen:
            _fail(f"run manifest repeats seed/arm {seed}/{arm}")
        seen.add(key)
        if len(rows) >= len(expected_order) or key != expected_order[len(rows)]:
            _fail(f"run manifest row {line_number} is not the fixed launch-order prefix")
        expected_launch_order = EXPECTED_LAUNCH_ORDER[key]
        if order != expected_launch_order:
            _fail(
                f"run manifest {seed}/{arm} launch order is {order}, "
                f"expected {expected_launch_order}"
            )
        if status != "complete":
            _fail(f"run manifest {seed}/{arm} has non-complete status {status!r}")
        run_dir_path = Path(run_dir_raw).expanduser()
        if not run_dir_path.is_absolute():
            _fail(f"run manifest {seed}/{arm} run_dir is not absolute")
        rows.append(
            {
                "seed": seed,
                "arm": arm,
                "launch_order": order,
                "run_dir": run_dir_path.resolve(),
                "status": status,
            }
        )
    if len(rows) % 2:
        _fail("run manifest contains an incomplete off/on pair")
    if allow_incomplete:
        if len(rows) not in (0, 2, 4, 6):
            _fail("incremental run manifest is not a complete-pair launch prefix")
    elif len(rows) != len(expected_order):
        _fail(f"final run manifest must contain exactly six rows, found {len(rows)}")
    return path, rows


def _validate_protocol_manifest(
    protocol: dict[str, str], manifest: dict[str, Any], arm_dir: Path
) -> None:
    scalar_pairs = {
        "planner_prompts": "planner_prompts",
        "plan_steps": "planner_steps",
        "executed_prompts": "primary_prompts",
        "steps_per_run": "executed_steps",
        "source_plan_step": "source_plan_step",
        "sidecar_trigger_active_ranks": "sidecar_trigger_active_ranks",
        "sidecar_stop_ack_timeout_seconds": "sidecar_stop_ack_timeout_seconds",
        "sidecar_shrink_quorum_size": "sidecar_shrink_quorum_size",
        "primary_hccl_allocator_start": "primary_hccl_allocator_start",
    }
    for protocol_key, manifest_key in scalar_pairs.items():
        if int(protocol[protocol_key]) != manifest.get(manifest_key):
            _fail(
                f"root protocol {protocol_key} differs from arm manifest {manifest_key}"
            )
    boolean_pairs = {
        "planned_residency": "planned_residency",
        "tail_guard": "tail_guard_enabled",
        "actor_frozen": "actor_frozen",
        "paired_request_sampling_seeds": "paired_request_sampling_seeds",
        "fast_step_subset": "fast_step_subset",
        "sidecar_require_active_lease_before_restore": (
            "sidecar_require_active_lease_before_restore"
        ),
        "sidecar_require_shrink_quorum": "sidecar_require_shrink_quorum",
        "eager_weight_sync_group_init": "eager_weight_sync_group_init",
        "primary_moe_shared_expert_overlap": "primary_moe_shared_expert_overlap",
    }
    for protocol_key, manifest_key in boolean_pairs.items():
        protocol_value = protocol[protocol_key] == "true"
        if protocol_value is not manifest.get(manifest_key):
            _fail(
                f"root protocol {protocol_key} differs from arm manifest {manifest_key}"
            )
    model_path = _resolve_recorded_path(
        manifest["sidecar_model_path"], arm_dir, "sidecar_model_path"
    )
    dataset_path = _resolve_recorded_path(
        manifest["sidecar_dataset_path"], arm_dir, "sidecar_dataset_path"
    )
    if Path(protocol["sidecar_model"]).expanduser().resolve() != model_path:
        _fail("root protocol sidecar model differs from arm manifest")
    if Path(protocol["sidecar_data"]).expanduser().resolve() != dataset_path:
        _fail("root protocol sidecar data differs from arm manifest")
    if protocol["sidecar_model_revision"] != manifest["sidecar_model_revision"]:
        _fail("root protocol sidecar model revision differs from arm manifest")


def _parse_rank_list(value: str, label: str) -> tuple[int, ...]:
    try:
        ranks = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise VerificationError(f"{label} contains a non-integer rank") from exc
    if len(set(ranks)) != len(ranks):
        _fail(f"{label} contains duplicate ranks")
    if any(rank < 0 or rank >= WORLD_SIZE for rank in ranks):
        _fail(f"{label} is not a subset of ranks 0 through {WORLD_SIZE - 1}")
    return ranks


def _find_step_plan(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, list) or len(payload) != 5:
        _fail("executed planner output must be a JSON list containing exactly five steps")
    if any(not isinstance(entry, dict) for entry in payload):
        _fail("executed planner output contains a non-object step")
    steps = [entry.get("step") for entry in payload]
    if steps != [1, 2, 3, 4, 5]:
        _fail(f"executed planner steps must be [1, 2, 3, 4, 5], got {steps}")
    candidates = [entry for entry in payload if entry.get("step") == 1]
    if len(candidates) != 1:
        _fail(f"planner output must contain exactly one step 1 record, found {len(candidates)}")
    plan = candidates[0]
    if plan.get("selected_floor") != TARGET_FLOOR:
        _fail(f"step 1 plan selected floor {plan.get('selected_floor')}, expected floor4")
    if plan.get("feasible") is not True:
        _fail("step 1 plan is not marked feasible")
    if plan.get("tail_guard_enabled") is not False:
        _fail("step 1 plan does not explicitly disable TailGuard")
    final_ranks = plan.get("final_survivor_ranks")
    if not isinstance(final_ranks, list) or len(final_ranks) != TARGET_FLOOR:
        _fail("step 1 plan does not contain four final survivor ranks")
    _parse_rank_list(",".join(str(item) for item in final_ranks), "plan final survivors")
    intermediate_ranks = plan.get("intermediate_survivor_ranks")
    if not isinstance(intermediate_ranks, list) or len(intermediate_ranks) != 8:
        _fail("step 1 plan does not contain eight intermediate survivor ranks")
    _parse_rank_list(
        ",".join(str(item) for item in intermediate_ranks),
        "plan intermediate survivors",
    )
    stage_survivors = plan.get("stage_survivor_ranks")
    if (
        not isinstance(stage_survivors, list)
        or len(stage_survivors) != 2
        or any(not isinstance(ranks, list) for ranks in stage_survivors)
    ):
        _fail("step 1 plan does not contain floor8 and floor4 stage survivor sets")
    if stage_survivors[0] != intermediate_ranks or stage_survivors[1] != final_ranks:
        _fail("step 1 stage survivor sets disagree with its intermediate and final survivors")
    return plan


def _validate_no_tailguard_override(text: str, label: str) -> None:
    for match in TAIL_GUARD_LINE_RE.finditer(text):
        line = match.group(0)
        cap_match = re.search(r"\bplan_cap=(\d+)\b", line)
        override_match = re.search(r"\bsampling_override=(True|False)\b", line)
        if cap_match is None or override_match is None:
            _fail(f"{label} contains an ambiguous TailGuard cap record")
        cap = int(cap_match.group(1))
        override = override_match.group(1) == "True"
        if override or cap < MAX_RESPONSE_LENGTH:
            _fail(
                f"{label} applied a TailGuard sampling override: "
                f"plan_cap={cap} sampling_override={override}"
            )


def _load_manifest(arm_dir: Path, seed: int, arm: str) -> tuple[dict[str, Any], Path, dict[str, Any]]:
    manifest_path = arm_dir / MANIFEST_NAME
    manifest = _read_json(manifest_path, "pair manifest")
    if not isinstance(manifest, dict):
        _fail(f"pair manifest is not an object: {manifest_path}")
    _required(manifest, "schema_version", 1)
    _required(
        manifest,
        "experiment",
        "qwen2_5_1_5b_planned_floor4_noguard_sidecar_pair",
    )
    _required(manifest, "seed", seed)
    _required(manifest, "arm", arm)
    _required(manifest, "launch_order", EXPECTED_LAUNCH_ORDER[(seed, arm)])
    _required(manifest, "planned", True)
    _required(manifest, "planned_residency", True)
    _required(manifest, "target_floor", TARGET_FLOOR)
    _required(manifest, "tail_guard_enabled", False)
    _required(manifest, "expected_responses", EXPECTED_RESPONSES)
    _required(manifest, "request_seed", seed)
    _required(manifest, "planner_prompts", 160)
    _required(manifest, "planner_steps", 5)
    _required(manifest, "executed_steps", 1)
    _required(manifest, "source_plan_step", 1)
    _required(manifest, "fast_step_subset", False)
    _required(manifest, "primary_prompts", 32)
    _required(manifest, "responses_per_prompt", 16)
    _required(manifest, "actor_frozen", True)
    _required(manifest, "paired_request_sampling_seeds", True)
    _required(manifest, "sidecar_enabled", arm == "on")
    _required(manifest, "sidecar_tensor_parallel_size", 1)
    _required(manifest, "sidecar_replica_count", 8)
    _required(manifest, "sidecar_trigger_active_ranks", SIDECAR_TRIGGER_FLOOR)
    _required(manifest, "sidecar_temperature", 0.0)
    _required(manifest, "sidecar_top_p", 1.0)
    _required(manifest, "sidecar_max_tokens", 4096)
    _required(manifest, "sidecar_stop_ack_timeout_seconds", 60)
    _required(manifest, "sidecar_require_active_lease_before_restore", True)
    _required(manifest, "sidecar_require_shrink_quorum", True)
    _required(manifest, "sidecar_shrink_quorum_size", WORLD_SIZE)
    _required(manifest, "eager_weight_sync_group_init", False)
    _required(
        manifest,
        "primary_hccl_allocator_start",
        PRIMARY_HCCL_ALLOCATOR_START,
    )
    _required(manifest, "primary_moe_shared_expert_overlap", False)
    _required(manifest, "sidecar_dataset_split", "train")
    _required(manifest, "sidecar_model_revision", SIDECAR_MODEL_REVISION)
    _required(
        manifest,
        "sidecar_model_weights_sha256",
        SIDECAR_MODEL_WEIGHTS_SHA256,
    )
    model_path = _resolve_recorded_path(
        _required(manifest, "sidecar_model_path"), arm_dir, "sidecar_model_path"
    )
    if model_path != SIDECAR_MODEL_PATH.resolve():
        _fail(
            f"sidecar model must be {SIDECAR_MODEL_PATH}, got {model_path}"
        )
    _resolve_recorded_path(_required(manifest, "run_dir"), arm_dir, "run_dir")

    plan_sha = _required(manifest, "plan_sha256")
    if not isinstance(plan_sha, str) or SHA256_RE.fullmatch(plan_sha) is None:
        _fail("manifest plan_sha256 is not a lowercase SHA-256 digest")
    plan_path = _resolve_recorded_path(
        _required(manifest, "plan_file"), arm_dir, "plan_file"
    )
    if not plan_path.is_file():
        _fail(f"recorded plan file does not exist: {plan_path}")
    actual_sha = _sha256(plan_path)
    if actual_sha != plan_sha:
        _fail(f"plan SHA-256 mismatch: recorded={plan_sha}, actual={actual_sha}")
    step_plan = _find_step_plan(_read_json(plan_path, "executed plan"))
    return manifest, plan_path, step_plan


def _integer_field(row: dict[str, Any], key: str, label: str) -> int:
    value = row.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        _fail(f"{label} does not contain an integer {key}")
    return value


def _canonical(value: Any) -> str:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as exc:
        raise VerificationError("primary identity field is not JSON serializable") from exc


def _plan_prompt_occurrences(plan: dict[str, Any]) -> tuple[int, ...]:
    assignment = plan.get("rank_to_source_idx")
    if not isinstance(assignment, dict) or set(assignment) != {
        str(rank) for rank in range(WORLD_SIZE)
    }:
        _fail("step 1 plan does not contain rank_to_source_idx for all 16 ranks")
    occurrences: list[int] = []
    for rank in range(WORLD_SIZE):
        values = assignment[str(rank)]
        if not isinstance(values, list) or len(values) != 2:
            _fail(f"step 1 rank_to_source_idx rank {rank} must contain two prompts")
        for value in values:
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                _fail("step 1 rank_to_source_idx contains an invalid occurrence")
            occurrences.append(value)
    if len(set(occurrences)) != 32:
        _fail("step 1 rank_to_source_idx does not contain 32 unique occurrences")
    return tuple(sorted(occurrences))


def _primary_artifact(arm_dir: Path, seed: int, arm: str) -> dict[str, Any]:
    expected_parent = f"primary_planned_f4_noguard_seed{seed}_{arm}"
    epoch = _only(arm_dir.rglob("epoch_001_mode1_planned"), f"{arm} primary epoch")
    if epoch.parent.name != expected_parent:
        _fail(f"primary epoch parent is {epoch.parent.name!r}, expected {expected_parent!r}")

    log_path = _only((epoch / "logs").glob("*.txt"), f"{arm} primary log")
    text = _strip_ansi(log_path.read_text(encoding="utf-8", errors="replace"))
    if OOM_RE.search(text):
        _fail(f"{seed}/{arm} primary log contains an OOM")
    if PREEMPT_RE.search(text):
        _fail(f"{seed}/{arm} primary log contains request preemption")
    _validate_no_tailguard_override(text, f"{seed}/{arm} primary log")
    overlap_values = MOE_SHARED_EXPERT_OVERLAP_RE.findall(text)
    if not overlap_values or set(overlap_values) != {"False"}:
        _fail(
            f"{seed}/{arm} primary runtime does not prove "
            "moe_shared_expert_overlap=False"
        )
    if "Elastic full-world restore" not in text and "restore requested" not in text:
        _fail(f"{seed}/{arm} primary log does not prove full-world restore")

    generate_done_matches = list(GENERATE_DONE_RE.finditer(text))
    restore_start_matches = list(RESTORE_START_RE.finditer(text))
    if len(generate_done_matches) != 1 or len(restore_start_matches) != 1:
        _fail(f"{seed}/{arm} does not contain one generation-done and restore-start timestamp")
    generate_done_time = float(generate_done_matches[0].group(1))
    restore_start_time = float(restore_start_matches[0].group(1))
    if not generate_done_time < restore_start_time:
        _fail(f"{seed}/{arm} restore did not start after generation")
    primary_acks = list(PRIMARY_ACK_RE.finditer(text))
    if arm == "on":
        if len(primary_acks) != 1:
            _fail(f"{seed}/on does not contain one sidecar release acknowledgement")
        if not (
            generate_done_matches[0].end()
            < primary_acks[0].start()
            < restore_start_matches[0].start()
        ):
            _fail(f"{seed}/on primary acknowledgement is outside the pre-restore interval")
    elif primary_acks:
        _fail(f"{seed}/off unexpectedly performed a sidecar restore handshake")

    rollout_times = [float(item) for item in ROLLOUT_TIME_RE.findall(text)]
    if len(rollout_times) != 1 or not math.isfinite(rollout_times[0]) or rollout_times[0] <= 0:
        _fail(f"{seed}/{arm} must contain one positive rollout_output_time_s")
    aborted = [float(item) for item in ABORT_RE.findall(text)]
    if len(aborted) != 1 or aborted[0] != 0.0:
        _fail(f"{seed}/{arm} does not prove response/aborted_ratio=0")

    shrink_sets = [
        _parse_rank_list(raw, f"{seed}/{arm} shrink active ranks")
        for raw in SHRINK_RE.findall(text)
    ]
    unique_sets: list[tuple[int, ...]] = []
    for ranks in shrink_sets:
        if ranks not in unique_sets:
            unique_sets.append(ranks)
    shrink_sizes = [len(ranks) for ranks in unique_sets]
    if shrink_sizes not in ([8], [8, 4]):
        _fail(
            f"{seed}/{arm} must execute Planned 16-to-8 shrink with optional 8-to-4, "
            f"got {shrink_sizes}"
        )

    rollout_path = _only((epoch / "rollout_data").glob("*.jsonl"), "primary rollout JSONL")
    length_path = _only((epoch / "rollout_length").glob("length_*.txt"), "primary length file")
    try:
        lengths = [
            int(line.strip())
            for line in length_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except ValueError as exc:
        raise VerificationError(f"{length_path} contains a non-integer length") from exc
    if len(lengths) != EXPECTED_RESPONSES:
        _fail(f"{seed}/{arm} has {len(lengths)} lengths, expected {EXPECTED_RESPONSES}")
    if any(length <= 0 or length > MAX_RESPONSE_LENGTH for length in lengths):
        _fail(f"{seed}/{arm} contains a response length outside 1..{MAX_RESPONSE_LENGTH}")

    rows = _read_jsonl(rollout_path, "primary rollout output")
    if len(rows) != EXPECTED_RESPONSES:
        _fail(f"{seed}/{arm} has {len(rows)} rollout rows, expected {EXPECTED_RESPONSES}")
    identities: list[tuple[int, int]] = []
    lengths_by_identity: dict[tuple[int, int], int] = {}
    tokens_by_identity: dict[tuple[int, int], tuple[int, ...]] = {}
    request_fingerprints: dict[tuple[int, int], tuple[str, str, str, int]] = {}
    samples_by_occurrence: dict[int, set[int]] = {}
    occurrence_fingerprints: dict[int, tuple[str, str, str]] = {}
    for line_number, (row, length) in enumerate(zip(rows, lengths, strict=True), 1):
        label = f"{seed}/{arm} rollout row {line_number}"
        occurrence = _integer_field(row, "prompt_occurrence_ordinal", label)
        sample_index = _integer_field(row, "rollout_sample_index", label)
        request_seed = _integer_field(row, "rollout_request_seed", label)
        rollout_rank = _integer_field(row, "rollout_rank", label)
        if occurrence < 0:
            _fail(f"{label} has a negative prompt occurrence")
        if sample_index < 0 or sample_index >= 16:
            _fail(f"{label} rollout_sample_index is outside 0..15")
        if request_seed < 0:
            _fail(f"{label} has a negative rollout_request_seed")
        if rollout_rank < 0 or rollout_rank >= WORLD_SIZE:
            _fail(f"{label} rollout_rank is outside 0..15")
        if row.get("step") != 1:
            _fail(f"{label} does not belong to executed runtime step 1")
        request_id = row.get("request_id")
        if request_id is None or str(request_id) == "":
            _fail(f"{label} has no request_id provenance")
        prompt_hash = row.get("rollout_prompt_hash")
        if not isinstance(prompt_hash, str) or PROMPT_HASH_RE.fullmatch(prompt_hash) is None:
            _fail(f"{label} has no lowercase 128-bit rollout_prompt_hash")
        for key in ("input", "gts"):
            if key not in row:
                _fail(f"{label} does not contain {key}")
        canonical_input = _canonical(row["input"])
        canonical_gts = _canonical(row["gts"])
        occurrence_fingerprint = (prompt_hash, canonical_input, canonical_gts)
        previous_occurrence = occurrence_fingerprints.setdefault(
            occurrence, occurrence_fingerprint
        )
        if previous_occurrence != occurrence_fingerprint:
            _fail(f"{seed}/{arm} occurrence {occurrence} changes prompt identity")
        identity = (occurrence, sample_index)
        if identity in lengths_by_identity:
            _fail(f"{seed}/{arm} contains duplicate primary request identity {identity}")
        mask = row.get("response_mask")
        if not isinstance(mask, list):
            _fail(f"{seed}/{arm} rollout row has no response_mask list")
        try:
            mask_values = [int(value) for value in mask]
        except (TypeError, ValueError) as exc:
            raise VerificationError(
                f"{seed}/{arm} response_mask contains a non-integer value"
            ) from exc
        if any(value not in (0, 1) for value in mask_values):
            _fail(f"{seed}/{arm} response_mask contains a value outside 0 or 1")
        if sum(mask_values) != length:
            _fail(f"{seed}/{arm} response_mask does not match length file for {identity}")
        responses = row.get("responses")
        if not isinstance(responses, list) or len(responses) != len(mask):
            _fail(f"{seed}/{arm} response token IDs do not align with response_mask")
        try:
            generated_ids = [
                int(token)
                for token, keep in zip(responses, mask_values, strict=True)
                if keep != 0
            ]
        except (TypeError, ValueError) as exc:
            raise VerificationError(
                f"{seed}/{arm} responses contains a non-integer token ID"
            ) from exc
        identities.append(identity)
        lengths_by_identity[identity] = length
        tokens_by_identity[identity] = tuple(generated_ids)
        request_fingerprints[identity] = (
            prompt_hash,
            canonical_input,
            canonical_gts,
            request_seed,
        )
        samples_by_occurrence.setdefault(occurrence, set()).add(sample_index)
    if len(samples_by_occurrence) != 32:
        _fail(
            f"{seed}/{arm} has {len(samples_by_occurrence)} prompt occurrences, expected 32"
        )
    expected_samples = set(range(16))
    incomplete = {
        occurrence: sorted(samples)
        for occurrence, samples in samples_by_occurrence.items()
        if samples != expected_samples
    }
    if incomplete:
        occurrence = next(iter(incomplete))
        _fail(
            f"{seed}/{arm} occurrence {occurrence} does not contain samples 0..15"
        )

    total_tokens = sum(lengths)
    return {
        "epoch_dir": str(epoch),
        "log_file": str(log_path),
        "rollout_file": str(rollout_path),
        "length_file": str(length_path),
        "response_count": len(rows),
        "generated_response_tokens": total_tokens,
        "rollout_time_s": rollout_times[0],
        "response_token_throughput": total_tokens / rollout_times[0],
        "driver_generate_done_time": generate_done_time,
        "driver_restore_rpc_start_time": restore_start_time,
        "restore_ack_request_id": primary_acks[0].group(1) if primary_acks else None,
        "restore_ack_lease_id": primary_acks[0].group(2) if primary_acks else None,
        "request_identity_multiset": Counter(identities),
        "lengths_by_identity": lengths_by_identity,
        "tokens_by_identity": tokens_by_identity,
        "request_fingerprints": request_fingerprints,
        "prompt_occurrences": sorted(samples_by_occurrence),
        "stage8_survivor_ranks": list(unique_sets[0]),
        "deepest_runtime_survivor_ranks": list(unique_sets[-1]),
        "runtime_reached_floor4": shrink_sizes == [8, 4],
    }


def _kv_lines(path: Path) -> dict[str, list[str]]:
    if not path.is_file():
        _fail(f"missing sidecar lease log: {path}")
    values: dict[str, list[str]] = {}
    for raw in _strip_ansi(path.read_text(encoding="utf-8", errors="replace")).splitlines():
        key, separator, value = raw.partition("=")
        if separator:
            values.setdefault(key.strip(), []).append(value.strip())
    return values


def _one_value(values: dict[str, list[str]], key: str) -> str:
    items = values.get(key, [])
    if len(items) != 1:
        _fail(f"lease log must contain {key} exactly once, found {len(items)}")
    return items[0]


def _float_value(values: dict[str, list[str]], key: str) -> float:
    try:
        result = float(_one_value(values, key))
    except ValueError as exc:
        raise VerificationError(f"lease log {key} is not numeric") from exc
    if not math.isfinite(result):
        _fail(f"lease log {key} is not finite")
    return result


def _timed_event(
    text: str,
    key: str,
    *,
    required_fields: dict[str, str] | None = None,
) -> tuple[float, dict[str, str]]:
    candidates: list[tuple[float, dict[str, str]]] = []
    pattern = re.compile(rf"(?m)^{re.escape(key)}=({FLOAT})(?:\s+(.*))?$")
    for match in pattern.finditer(text):
        fields: dict[str, str] = {}
        for item in (match.group(2) or "").split():
            field, separator, value = item.partition("=")
            if separator:
                fields[field] = value
        if required_fields and any(fields.get(name) != value for name, value in required_fields.items()):
            continue
        candidates.append((float(match.group(1)), fields))
    if len(candidates) != 1:
        _fail(
            f"lease log must contain exactly one {key} event matching "
            f"{required_fields or {}}, found {len(candidates)}"
        )
    timestamp, fields = candidates[0]
    if not math.isfinite(timestamp):
        _fail(f"lease event {key} has a non-finite timestamp")
    return timestamp, fields


def _timed_events(text: str, key: str) -> list[tuple[float, dict[str, str]]]:
    events: list[tuple[float, dict[str, str]]] = []
    pattern = re.compile(rf"(?m)^{re.escape(key)}=({FLOAT})(?:\s+(.*))?$")
    for match in pattern.finditer(text):
        fields: dict[str, str] = {}
        for item in (match.group(2) or "").split():
            field, separator, value = item.partition("=")
            if separator:
                fields[field] = value
        events.append((float(match.group(1)), fields))
    return events


def _read_fields(path: Path, label: str) -> dict[str, str]:
    if not path.is_file():
        _fail(f"missing {label}: {path}")
    fields: dict[str, str] = {}
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        key, separator, value = raw.partition("=")
        if not separator or not key:
            _fail(f"{path}:{line_number} is not a key-value field")
        if key in fields:
            _fail(f"{path}:{line_number} repeats {key}")
        fields[key] = value
    return fields


def _infer_setting(text: str, key: str) -> str:
    matches = re.findall(rf"(?m)^{re.escape(key)}=(.*)$", text)
    unique = list(dict.fromkeys(item.strip() for item in matches))
    if len(unique) != 1:
        _fail(f"sidecar infer log must contain one stable {key}, found {unique}")
    return unique[0]


def _json_events(text: str) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line.startswith("{"):
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and "event" in value:
            events.append(value)
    return events


def _prompt_to_text(value: Any) -> str:
    if value is None:
        return ""
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        if "content" in value:
            return _prompt_to_text(value["content"])
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, (list, tuple)):
        parts: list[str] = []
        for item in value:
            if isinstance(item, dict) and "content" in item:
                role = item.get("role")
                content = _prompt_to_text(item["content"])
                parts.append(f"{role}: {content}" if role else content)
            else:
                parts.append(_prompt_to_text(item))
        return "\n".join(part for part in parts if part)
    return str(value)


def _dataset_frame(path: Path, cache: dict[Path, Any]) -> Any:
    if path in cache:
        return cache[path]
    if not path.is_file() or path.suffix.lower() != ".parquet":
        _fail(f"sidecar GSM8K train parquet does not exist: {path}")
    try:
        import pandas as pd
    except ImportError as exc:
        raise VerificationError("pandas is required to verify GSM8K accuracy") from exc
    frame = pd.read_parquet(path)
    for column in ("prompt", "reward_model"):
        if column not in frame.columns:
            _fail(f"sidecar dataset does not contain {column}: {path}")
    cache[path] = frame
    return frame


def _state_rows(state_dir: Path, pattern: str, label: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(state_dir.glob(pattern)):
        rows.extend(_read_jsonl(path, label))
    return rows


def _sidecar_artifact(
    arm_dir: Path,
    manifest: dict[str, Any],
    primary: dict[str, Any],
    dataset_cache: dict[Path, Any],
) -> dict[str, Any]:
    sidecar_dir = Path(primary["epoch_dir"]) / "sidecar"
    duplicate_candidates = [
        path
        for path in (arm_dir / "sidecar", Path(primary["epoch_dir"]).parent / "sidecar")
        if path.exists() and path.resolve() != sidecar_dir.resolve()
    ]
    if duplicate_candidates:
        _fail(f"ambiguous second sidecar artifact directory: {duplicate_candidates[0]}")
    lease_path = sidecar_dir / "lease.log"
    infer_path = sidecar_dir / "infer.log"
    output_path = sidecar_dir / "outputs.jsonl"
    state_dir = sidecar_dir / "state"
    if not state_dir.is_dir():
        _fail(f"missing sidecar state directory: {state_dir}")

    lease = _kv_lines(lease_path)
    if _one_value(lease, "watch_expected_active_ranks") != str(SIDECAR_TRIGGER_FLOOR):
        _fail("watcher was not configured to start at the floor8 release window")
    if _one_value(lease, "watch_world_size") != str(WORLD_SIZE):
        _fail("watcher world size is not 16")
    if _one_value(lease, "watch_require_active_lease") != "1":
        _fail("watcher did not require an active running lease before restore")
    if _one_value(lease, "watch_require_shrink_quorum") != "1":
        _fail("watcher did not require a strict shrink quorum")
    if _one_value(lease, "watch_shrink_quorum_size") != str(WORLD_SIZE):
        _fail("watcher shrink quorum size is not 16")
    lease_text = _strip_ansi(lease_path.read_text(encoding="utf-8", errors="replace"))
    start = _float_value(lease, "sidecar_start_time")
    detected, detected_fields = _timed_event(
        lease_text, "shrink_window_detected_time"
    )
    if detected_fields != {
        "active_count": str(SIDECAR_TRIGGER_FLOOR),
        "coordinated": "1",
    }:
        _fail("shrink-window event does not prove a coordinated floor8 release")
    coordinated_start, coordinated_fields = _timed_event(
        lease_text, "coordinated_start_time"
    )
    if detected != coordinated_start:
        _fail("shrink-window timestamp differs from the coordinated quorum timestamp")
    if start < coordinated_start:
        _fail("sidecar started before the coordinated 16-rank shrink quorum")
    trainer_stop, trainer_stop_fields = _timed_event(lease_text, "trainer_stop_request_time")
    if lease.get("sidecar_deadline_signal_time"):
        _fail("formal sidecar arm was stopped by the legacy log-text deadline path")
    for deferred_time, deferred_fields in _timed_events(
        lease_text, "sidecar_deadline_signal_deferred_time"
    ):
        if deferred_fields.get("reason") != "trainer_stop_request_is_authoritative":
            _fail("sidecar legacy deadline deferral has an unexpected reason")
        if deferred_time > trainer_stop:
            _fail("sidecar legacy deadline was deferred only after the trainer stop request")
    exit_confirmed, exit_fields = _timed_event(
        lease_text,
        "sidecar_exit_confirmed_time",
        required_fields={"reason": "pre_restore_handshake", "process_group_alive": "0"},
    )
    watcher_ack, watcher_ack_fields = _timed_event(
        lease_text,
        "watcher_restore_ack_time",
        required_fields={"status": "released"},
    )
    generate_done = float(primary["driver_generate_done_time"])
    restore_start = float(primary["driver_restore_rpc_start_time"])
    durable_candidates = [
        event
        for event in _timed_events(lease_text, "sidecar_artifacts_durable_time")
        if trainer_stop <= event[0] <= exit_confirmed
    ]
    if len(durable_candidates) != 1:
        _fail(
            "lease log must contain one artifact-durability event between trainer stop "
            f"and confirmed process exit, found {len(durable_candidates)}"
        )
    artifacts_durable, durable_fields = durable_candidates[0]
    if (
        Path(durable_fields.get("output", "")).expanduser().resolve()
        != output_path.resolve()
        or Path(durable_fields.get("state_dir", "")).expanduser().resolve()
        != state_dir.resolve()
    ):
        _fail("artifact-durability event does not name the verified output and state paths")
    if not (
        detected
        <= start
        < generate_done
        < trainer_stop
        <= artifacts_durable
        <= exit_confirmed
        <= watcher_ack
        < restore_start
    ):
        _fail(
            "sidecar handshake order must be shrink, start, generation done, trainer stop, "
            "sidecar exit, watcher acknowledgement, then restore RPC"
        )
    if any(
        timestamp < trainer_stop
        for timestamp, _ in _timed_events(lease_text, "sidecar_release_confirmed_time")
    ):
        _fail("sidecar active lease was released before the trainer stop request")
    request_id = trainer_stop_fields.get("request_id")
    lease_id = trainer_stop_fields.get("lease_id")
    if not request_id or not lease_id:
        _fail("trainer stop event does not record request_id and lease_id")
    if watcher_ack_fields.get("request_id") != request_id or watcher_ack_fields.get("lease_id") != lease_id:
        _fail("watcher acknowledgement does not match the trainer stop request")
    if exit_fields.get("request_id") != request_id or exit_fields.get("lease_id") != lease_id:
        _fail("confirmed sidecar exit does not match the trainer stop request")
    if primary["restore_ack_request_id"] != request_id or primary["restore_ack_lease_id"] != lease_id:
        _fail("primary acknowledgement does not match the watcher handshake")
    shrink_line = _one_value(lease, "shrink_window_line")
    match = SHRINK_RE.search(shrink_line)
    if match is None:
        _fail("sidecar lease does not retain the triggering shrink line")
    active = _parse_rank_list(match.group(1), "lease active ranks")
    if len(active) != SIDECAR_TRIGGER_FLOOR:
        _fail("sidecar did not start after the 16-to-8 shrink")
    if list(active) != primary["stage8_survivor_ranks"]:
        _fail("lease survivor ranks differ from the primary floor8 survivors")
    explicit_active = _parse_rank_list(
        _one_value(lease, "sidecar_active_ranks"), "explicit lease active ranks"
    )
    if explicit_active != active:
        _fail("explicit sidecar active ranks differ from the triggering shrink")

    active_csv = ",".join(str(rank) for rank in active)
    expected_quorum = tuple(range(WORLD_SIZE))
    expected_quorum_csv = ",".join(str(rank) for rank in expected_quorum)
    if coordinated_fields.get("active_ranks") != active_csv:
        _fail("coordinated quorum active set differs from the planned floor8 stage")
    if coordinated_fields.get("quorum_count") != str(WORLD_SIZE):
        _fail("coordinated start does not record a complete 16-rank quorum")
    coordinated_ranks = _parse_rank_list(
        coordinated_fields.get("quorum_ranks", ""), "coordinated quorum ranks"
    )
    if coordinated_ranks != expected_quorum:
        _fail("coordinated quorum ranks are not exactly ranks 0 through 15")

    rank_record = re.fullmatch(
        r"([0-9,]+)\s+active_ranks=([0-9,]+)",
        _one_value(lease, "quorum_ranks"),
    )
    if rank_record is None:
        _fail("lease final quorum-rank record is malformed")
    final_quorum_ranks = _parse_rank_list(rank_record.group(1), "final quorum ranks")
    if final_quorum_ranks != expected_quorum or rank_record.group(2) != active_csv:
        _fail("lease final quorum ranks or active set differ from the coordinated start")

    count_record = re.fullmatch(
        r"(\d+)\s+quorum_required=(\d+)\s+active_ranks=([0-9,]+)",
        _one_value(lease, "quorum_count"),
    )
    if count_record is None:
        _fail("lease final quorum-count record is malformed")
    if (
        int(count_record.group(1)) != WORLD_SIZE
        or int(count_record.group(2)) != WORLD_SIZE
        or count_record.group(3) != active_csv
    ):
        _fail("lease final quorum count does not prove 16 of 16 ranks")

    progress = _timed_events(lease_text, "shrink_quorum_progress_time")
    target_progress = [
        (timestamp, fields)
        for timestamp, fields in progress
        if fields.get("active_ranks") == active_csv
    ]
    if len(target_progress) != WORLD_SIZE:
        _fail(
            "lease must contain 16 unique progress events for the coordinated active set"
        )
    try:
        reporters = [int(fields["reporter"]) for _, fields in target_progress]
        counts = [int(fields["quorum_count"]) for _, fields in target_progress]
        required = [int(fields["quorum_required"]) for _, fields in target_progress]
    except (KeyError, ValueError) as exc:
        raise VerificationError("lease shrink-quorum progress evidence is malformed") from exc
    if sorted(reporters) != list(range(WORLD_SIZE)) or counts != list(
        range(1, WORLD_SIZE + 1)
    ):
        _fail("lease shrink-quorum progress does not uniquely cover ranks 0 through 15")
    if required != [WORLD_SIZE] * WORLD_SIZE:
        _fail("lease shrink-quorum progress does not consistently require 16 ranks")
    if any(timestamp > coordinated_start for timestamp, _ in target_progress):
        _fail("lease records shrink-quorum progress after coordinated sidecar start")
    if expected_quorum_csv != coordinated_fields.get("quorum_ranks"):
        _fail("coordinated quorum rank ordering is not canonical")

    detached = _parse_rank_list(_one_value(lease, "sidecar_devices"), "lease detached ranks")
    expected_detached = tuple(rank for rank in range(WORLD_SIZE) if rank not in set(active))
    if set(detached) != set(expected_detached):
        _fail("lease detached ranks are not the complement of floor8 survivors")
    if lease.get("sidecar_force_kill_time"):
        _fail("sidecar required SIGKILL at the restore boundary")

    handshake_dir = sidecar_dir / "restore_handshake"
    active_fields = _read_fields(handshake_dir / "active_lease", "sidecar active lease")
    stop_fields = _read_fields(handshake_dir / "stop_request", "sidecar stop request")
    ack_fields = _read_fields(handshake_dir / "stop_ack", "sidecar stop acknowledgement")
    if stop_fields.get("request_id") != request_id or stop_fields.get("lease_id") != lease_id:
        _fail("persisted stop request differs from lease-log handshake")
    if ack_fields.get("request_id") != request_id or ack_fields.get("lease_id") != lease_id:
        _fail("persisted stop acknowledgement differs from lease-log handshake")
    if ack_fields.get("status") != "released" or ack_fields.get("reason") != "sidecar_process_group_exited":
        _fail("persisted stop acknowledgement does not prove sidecar process-group exit")
    try:
        active_update_time = float(active_fields["update_time"])
        stop_file_time = float(stop_fields["request_time"])
        ack_request_time = float(ack_fields["request_time"])
        ack_file_time = float(ack_fields["ack_time"])
    except (KeyError, ValueError) as exc:
        raise VerificationError("persisted handshake contains an invalid timestamp") from exc
    if not (
        active_fields.get("lease_id") == lease_id
        and active_fields.get("state") == "released"
        and exit_confirmed <= active_update_time <= watcher_ack
        and stop_file_time == trainer_stop
        and ack_request_time == trainer_stop
        and exit_confirmed <= ack_file_time <= watcher_ack
    ):
        _fail("persisted handshake timestamps differ from the verified lease interval")

    if not infer_path.is_file():
        _fail(f"missing sidecar infer log: {infer_path}")
    infer_text = _strip_ansi(infer_path.read_text(encoding="utf-8", errors="replace"))
    if OOM_RE.search(infer_text):
        _fail("sidecar infer log contains an OOM")
    if PREEMPT_RE.search(infer_text):
        _fail("sidecar infer log contains request preemption")
    groups_raw = _infer_setting(infer_text, "sidecar_device_groups")
    groups = [
        _parse_rank_list(group, "sidecar device group")
        for group in groups_raw.split(";")
        if group.strip()
    ]
    if not groups:
        _fail("sidecar infer log contains no device group")
    used = tuple(dict.fromkeys(rank for group in groups for rank in group))
    if len(used) != sum(len(group) for group in groups):
        _fail("sidecar device groups overlap")
    if not set(used).issubset(set(detached)) or set(used).intersection(active):
        _fail("sidecar used a primary survivor rank")
    unused = _parse_rank_list(
        _infer_setting(infer_text, "sidecar_unused_devices"), "sidecar unused ranks"
    )
    if set(used).union(unused) != set(detached) or set(used).intersection(unused):
        _fail("sidecar used and unused device accounting does not cover detached ranks")
    if len(groups) != 8 or any(len(group) != 1 for group in groups):
        _fail("Qwen2.5-1.5B must use eight independent one-rank replicas")
    if len(used) != 8 or unused:
        _fail(
            "Qwen2.5-1.5B DP8 must use all eight ranks detached at the 16-to-8 transition"
        )
    model_path = _resolve_recorded_path(
        _required(manifest, "sidecar_model_path"), arm_dir, "sidecar_model_path"
    )
    logged_model = Path(_infer_setting(infer_text, "sidecar_model")).expanduser().resolve()
    if logged_model != model_path:
        _fail(f"sidecar model path mismatch: manifest={model_path}, log={logged_model}")

    events = _json_events(infer_text)
    sampling = [event for event in events if event.get("event") == "sidecar_sampling_params"]
    if not sampling:
        _fail("sidecar infer log has no sampling-parameter event")
    if any(
        event.get("n") != 1
        or float(event.get("temperature", math.nan)) != 0.0
        or float(event.get("top_p", math.nan)) != 1.0
        or int(event.get("max_tokens", -1)) != 4096
        for event in sampling
    ):
        _fail("sidecar sampling must use n=1, temperature=0, top_p=1, and max_tokens=4096")

    dataset_path = _resolve_recorded_path(
        _required(manifest, "sidecar_dataset_path"), arm_dir, "sidecar_dataset_path"
    )
    frame = _dataset_frame(dataset_path, dataset_cache)
    output_rows = _read_jsonl(output_path, "sidecar output")
    completed_rows = _state_rows(state_dir, "completed.shard*.jsonl", "sidecar completed state")
    if not completed_rows:
        _fail("sidecar state contains no completed rows")
    completed_times: dict[tuple[int, int], float] = {}
    for row in completed_rows:
        key = (int(row.get("sidecar_epoch", 0)), int(row["prompt_id"]))
        if key in completed_times:
            _fail(f"sidecar completed state repeats {key}")
        completed_times[key] = float(row["time"])
        if not start <= completed_times[key] <= exit_confirmed:
            _fail(f"sidecar completion {key} lies outside its occupied interval")

    resume_rows = _state_rows(state_dir, "resume.shard*.jsonl", "sidecar resume state")
    partial_keys: set[tuple[int, int]] = set()
    for row in resume_rows:
        key = (int(row.get("sidecar_epoch", 0)), int(row["prompt_id"]))
        if key in partial_keys:
            _fail(f"sidecar resume state repeats {key}")
        partial_keys.add(key)
        timestamp = float(row["time"])
        if not start <= timestamp <= exit_confirmed:
            _fail(f"sidecar partial {key} lies outside its lease shutdown")
    for path in state_dir.glob("inflight.shard*.json"):
        payload = _read_json(path, "sidecar inflight state")
        if payload.get("prompt_ids"):
            _fail(f"sidecar left uncheckpointed inflight prompts in {path}")

    output_keys: set[tuple[int, int]] = set()
    complete_count = 0
    completed_before_stop = 0
    completed_during_shutdown = 0
    skipped_count = 0
    correct_count = 0
    correct_before_stop = 0
    correct_during_shutdown = 0
    output_tokens = 0
    output_tokens_before_stop = 0
    output_tokens_during_shutdown = 0
    for row in output_rows:
        key = (int(row.get("sidecar_epoch", 0)), int(row["prompt_id"]))
        if key in output_keys:
            _fail(f"sidecar output repeats {key}")
        output_keys.add(key)
        if key not in completed_times:
            _fail(f"sidecar output {key} has no completed-state record")
        outputs = row.get("outputs")
        if not isinstance(outputs, list) or len(outputs) != 1 or not isinstance(outputs[0], dict):
            _fail(f"sidecar output {key} does not contain exactly one completion")
        completion = outputs[0]
        status = row.get("sidecar_status")
        finish_reason = completion.get("finish_reason")
        if status is not None or (isinstance(finish_reason, str) and "skipped" in finish_reason):
            skipped_count += 1
            continue
        if finish_reason not in ("stop", "length"):
            _fail(f"sidecar output {key} has nonfinal finish_reason={finish_reason!r}")
        if int(completion.get("resume_prefix_text_len", 0)) != 0:
            _fail("this one-lease experiment must not count resumed completions")
        token_count = completion.get("token_ids_len")
        if isinstance(token_count, bool) or not isinstance(token_count, int) or token_count <= 0:
            _fail(f"sidecar output {key} has invalid token_ids_len")

        source = row.get("prompt_source")
        if not isinstance(source, str) or ":" not in source:
            _fail(f"sidecar output {key} has invalid prompt_source")
        source_path_raw, row_index_raw = source.rsplit(":", 1)
        try:
            row_index = int(row_index_raw)
        except ValueError as exc:
            raise VerificationError(f"sidecar output {key} has invalid dataset row") from exc
        if Path(source_path_raw).expanduser().resolve() != dataset_path:
            _fail(f"sidecar output {key} is not sourced from the recorded GSM8K train file")
        if row_index < 0 or row_index >= len(frame):
            _fail(f"sidecar output {key} references dataset row {row_index} out of range")
        dataset_row = frame.iloc[row_index]
        if row.get("prompt") != _prompt_to_text(dataset_row["prompt"]):
            _fail(f"sidecar output {key} prompt differs from GSM8K train row {row_index}")
        reward_model = dataset_row["reward_model"]
        if not isinstance(reward_model, dict) or "ground_truth" not in reward_model:
            _fail(f"GSM8K train row {row_index} has no ground truth")
        text = completion.get("text")
        if not isinstance(text, str):
            _fail(f"sidecar output {key} has no completion text")
        score = gsm8k.compute_score(
            solution_str=text,
            ground_truth=str(reward_model["ground_truth"]),
            method="strict",
        )
        is_correct = int(score == 1.0)
        correct_count += is_correct
        complete_count += 1
        output_tokens += token_count
        if completed_times[key] <= trainer_stop:
            completed_before_stop += 1
            correct_before_stop += is_correct
            output_tokens_before_stop += token_count
        else:
            completed_during_shutdown += 1
            correct_during_shutdown += is_correct
            output_tokens_during_shutdown += token_count

    if output_keys != set(completed_times):
        _fail("sidecar output rows do not exactly match completed state")
    if output_keys.intersection(partial_keys):
        _fail("sidecar prompt is simultaneously complete and partial")
    if complete_count == 0:
        _fail("sidecar completed no usable GSM8K query")

    free_window_s = trainer_stop - start
    drain_s = exit_confirmed - trainer_stop
    occupied_s = exit_confirmed - start
    return {
        "lease_log": str(lease_path),
        "infer_log": str(infer_path),
        "output_file": str(output_path),
        "state_dir": str(state_dir),
        "lease_seconds": free_window_s,
        "free_window_seconds": free_window_s,
        "drain_seconds": drain_s,
        "occupied_seconds": occupied_s,
        "trainer_stop_request_time": trainer_stop,
        "sidecar_exit_confirmed_time": exit_confirmed,
        "watcher_restore_ack_time": watcher_ack,
        "driver_restore_rpc_start_time": restore_start,
        "restore_handshake_request_id": request_id,
        "restore_handshake_lease_id": lease_id,
        "survivor_ranks": list(active),
        "detached_ranks": list(detached),
        "used_ranks": list(used),
        "unused_detached_ranks": list(unused),
        "detached_rank_seconds": free_window_s * len(detached),
        "sidecar_used_rank_seconds": free_window_s * len(used),
        "free_window_detached_rank_seconds": free_window_s * len(detached),
        "free_window_sidecar_used_rank_seconds": free_window_s * len(used),
        "drain_detached_rank_seconds": drain_s * len(detached),
        "drain_sidecar_used_rank_seconds": drain_s * len(used),
        "occupied_detached_rank_seconds": occupied_s * len(detached),
        "occupied_sidecar_used_rank_seconds": occupied_s * len(used),
        "completed_queries": complete_count,
        "completed_before_trainer_stop": completed_before_stop,
        "completed_during_shutdown": completed_during_shutdown,
        "partial_queries": len(partial_keys),
        "skipped_queries": skipped_count,
        "completed_output_tokens": output_tokens,
        "completed_output_tokens_before_trainer_stop": output_tokens_before_stop,
        "completed_output_tokens_during_shutdown": output_tokens_during_shutdown,
        "strict_correct_queries": correct_count,
        "strict_accuracy": correct_count / complete_count,
        "strict_correct_before_trainer_stop": correct_before_stop,
        "strict_accuracy_before_trainer_stop": (
            correct_before_stop / completed_before_stop
            if completed_before_stop
            else None
        ),
        "strict_correct_during_shutdown": correct_during_shutdown,
    }


def _descriptive(values: list[float], *, final: bool) -> dict[str, Any]:
    if not values:
        return {
            "n": 0,
            "mean": None,
            "min": None,
            "max": None,
            "sample_sd": None,
            "t95_df": None,
            "ci95": None,
        }
    result: dict[str, Any] = {
        "n": len(values),
        "mean": mean(values),
        "min": min(values),
        "max": max(values),
        "sample_sd": stdev(values) if len(values) >= 2 else None,
        "t95_df": 2 if final else None,
        "ci95": None,
    }
    if final:
        if len(values) != len(SEEDS):
            _fail("df=2 paired confidence interval requires all three seeds")
        half_width = T_CRITICAL_95_DF2 * stdev(values) / math.sqrt(len(values))
        result["ci95"] = [result["mean"] - half_width, result["mean"] + half_width]
        result["t95_critical"] = T_CRITICAL_95_DF2
    return result


def _public_primary(primary: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in primary.items()
        if key not in {
            "request_identity_multiset",
            "lengths_by_identity",
            "tokens_by_identity",
            "request_fingerprints",
            "prompt_occurrences",
        }
    }


def verify(root: Path, allow_incomplete: bool = False) -> dict[str, Any]:
    root = root.expanduser().resolve()
    if not root.is_dir():
        _fail(f"experiment root does not exist: {root}")
    code_provenance_path = _load_code_provenance(root)
    protocol_path, protocol = _load_protocol(root)
    run_manifest_path, run_rows = _load_run_manifest(
        root, allow_incomplete=allow_incomplete
    )
    run_rows_by_key = {(row["seed"], row["arm"]): row for row in run_rows}
    pairs: list[dict[str, Any]] = []
    pending: list[dict[str, Any]] = []
    dataset_cache: dict[Path, Any] = {}

    for seed in SEEDS:
        seed_dir = root / f"seed_{seed}"
        missing = [arm for arm in ARMS if (seed, arm) not in run_rows_by_key]
        if missing:
            if allow_incomplete:
                pending.append({"seed": seed, "missing_arms": missing})
                continue
            _fail(f"seed {seed} is missing arms: {', '.join(missing)}")

        artifacts: dict[str, Any] = {}
        manifests: dict[str, dict[str, Any]] = {}
        plans: dict[str, dict[str, Any]] = {}
        for arm in ARMS:
            arm_dir = seed_dir / arm
            manifest, plan_path, step_plan = _load_manifest(arm_dir, seed, arm)
            _validate_protocol_manifest(protocol, manifest, arm_dir)
            primary = _primary_artifact(arm_dir, seed, arm)
            recorded_run_dir = _resolve_recorded_path(
                manifest["run_dir"], arm_dir, "run_dir"
            )
            expected_run_dir = (
                arm_dir / f"primary_planned_f4_noguard_seed{seed}_{arm}"
            ).resolve()
            if recorded_run_dir != expected_run_dir:
                _fail(f"seed {seed}/{arm} manifest run_dir is not the canonical arm run")
            if run_rows_by_key[(seed, arm)]["run_dir"] != recorded_run_dir:
                _fail(f"seed {seed}/{arm} root and arm run_dir records differ")
            if Path(primary["epoch_dir"]).parent.resolve() != recorded_run_dir:
                _fail(f"seed {seed}/{arm} epoch is outside its recorded run_dir")
            if tuple(primary["prompt_occurrences"]) != _plan_prompt_occurrences(step_plan):
                _fail(f"seed {seed}/{arm} rollout occurrences differ from the step 1 plan")
            if list(step_plan["stage_survivor_ranks"][0]) != primary["stage8_survivor_ranks"]:
                _fail(f"seed {seed}/{arm} floor8 survivors differ from the plan")
            if (
                primary["runtime_reached_floor4"]
                and list(step_plan["final_survivor_ranks"])
                != primary["deepest_runtime_survivor_ranks"]
            ):
                _fail(f"seed {seed}/{arm} floor4 survivors differ from the plan")
            sidecar = None
            if arm == "on":
                sidecar = _sidecar_artifact(arm_dir, manifest, primary, dataset_cache)
            elif (Path(primary["epoch_dir"]) / "sidecar").exists():
                _fail(f"seed {seed}/off unexpectedly contains sidecar artifacts")
            manifests[arm] = manifest
            plans[arm] = {"path": str(plan_path), "sha256": manifest["plan_sha256"]}
            artifacts[arm] = {"primary": primary, "sidecar": sidecar}

        for key in (
            "request_seed",
            "plan_sha256",
            "sidecar_model_path",
            "sidecar_model_revision",
            "sidecar_model_weights_sha256",
            "sidecar_dataset_path",
            "sidecar_dataset_split",
        ):
            if manifests["off"].get(key) != manifests["on"].get(key):
                _fail(f"seed {seed} pair differs in manifest field {key}")
        off_primary = artifacts["off"]["primary"]
        on_primary = artifacts["on"]["primary"]
        if off_primary["request_identity_multiset"] != on_primary["request_identity_multiset"]:
            _fail(f"seed {seed} off/on request identity multisets differ")
        if off_primary["lengths_by_identity"] != on_primary["lengths_by_identity"]:
            _fail(f"seed {seed} off/on generated response lengths differ")
        if off_primary["request_fingerprints"] != on_primary["request_fingerprints"]:
            _fail(
                f"seed {seed} off/on prompt hashes, inputs, ground truths, or request seeds differ"
            )
        if off_primary["tokens_by_identity"] != on_primary["tokens_by_identity"]:
            _fail(f"seed {seed} off/on generated response token IDs differ")
        if off_primary["generated_response_tokens"] != on_primary["generated_response_tokens"]:
            _fail(f"seed {seed} off/on generated token totals differ")
        for key in (
            "stage8_survivor_ranks",
            "deepest_runtime_survivor_ranks",
            "runtime_reached_floor4",
        ):
            if off_primary[key] != on_primary[key]:
                _fail(f"seed {seed} off/on runtime transition field {key} differs")

        off_throughput = off_primary["response_token_throughput"]
        on_throughput = on_primary["response_token_throughput"]
        throughput_delta = 100.0 * (on_throughput / off_throughput - 1.0)
        time_delta = 100.0 * (
            on_primary["rollout_time_s"] / off_primary["rollout_time_s"] - 1.0
        )
        pairs.append(
            {
                "seed": seed,
                "plan_sha256": manifests["off"]["plan_sha256"],
                "off": _public_primary(off_primary),
                "on": _public_primary(on_primary),
                "sidecar": artifacts["on"]["sidecar"],
                "paired_primary_throughput_delta_percent": throughput_delta,
                "paired_primary_throughput_cost_percent": -throughput_delta,
                "paired_rollout_time_delta_percent": time_delta,
            }
        )

    final = len(pairs) == len(SEEDS) and not pending
    if not final and not allow_incomplete:
        _fail("paired experiment is incomplete")
    throughput_deltas = [row["paired_primary_throughput_delta_percent"] for row in pairs]
    time_deltas = [row["paired_rollout_time_delta_percent"] for row in pairs]
    total_queries = sum(row["sidecar"]["completed_queries"] for row in pairs)
    total_correct = sum(row["sidecar"]["strict_correct_queries"] for row in pairs)
    free_window_queries = sum(
        row["sidecar"]["completed_before_trainer_stop"] for row in pairs
    )
    free_window_correct = sum(
        row["sidecar"]["strict_correct_before_trainer_stop"] for row in pairs
    )
    summary = {
        "schema_version": 1,
        "status": "PASS" if final else "INCOMPLETE",
        "experiment": "Qwen2.5-1.5B Planned floor4 no-TailGuard sidecar paired evidence",
        "root": str(root),
        "provenance": {
            "code_sha256_file": str(code_provenance_path),
            "code_sha256_file_sha256": _sha256(code_provenance_path),
            "protocol_file": str(protocol_path),
            "protocol_file_sha256": _sha256(protocol_path),
            "run_manifest_file": str(run_manifest_path),
            "run_manifest_file_sha256": _sha256(run_manifest_path),
            "completed_run_manifest_rows": len(run_rows),
        },
        "required_seeds": list(SEEDS),
        "completed_pairs": len(pairs),
        "pending": pending,
        "contract": {
            "steps_per_arm": 1,
            "planner_prompts": 160,
            "planner_steps": 5,
            "executed_steps": 1,
            "source_plan_step": 1,
            "fast_step_subset": False,
            "sidecar_trigger_floor": SIDECAR_TRIGGER_FLOOR,
            "strict_shrink_quorum_size": WORLD_SIZE,
            "primary_responses_per_arm": EXPECTED_RESPONSES,
            "planned": True,
            "target_floor": TARGET_FLOOR,
            "tail_guard_enabled": False,
            "paired_request_length_and_token_id_match": True,
            "paired_request_identity": (
                "prompt_occurrence_ordinal,rollout_sample_index"
            ),
            "gsm8k_scoring": "verl.utils.reward_score.gsm8k.compute_score(method=strict)",
        },
        "pairs": pairs,
        "aggregate": {
            "primary_throughput_delta_percent": _descriptive(throughput_deltas, final=final),
            "primary_rollout_time_delta_percent": _descriptive(time_deltas, final=final),
            "sidecar_completed_queries": total_queries,
            "sidecar_completed_before_trainer_stop": free_window_queries,
            "sidecar_completed_during_shutdown": sum(
                row["sidecar"]["completed_during_shutdown"] for row in pairs
            ),
            "sidecar_completed_output_tokens": sum(
                row["sidecar"]["completed_output_tokens"] for row in pairs
            ),
            "sidecar_completed_output_tokens_before_trainer_stop": sum(
                row["sidecar"]["completed_output_tokens_before_trainer_stop"]
                for row in pairs
            ),
            "sidecar_completed_output_tokens_during_shutdown": sum(
                row["sidecar"]["completed_output_tokens_during_shutdown"]
                for row in pairs
            ),
            "sidecar_partial_queries_at_restore": sum(
                row["sidecar"]["partial_queries"] for row in pairs
            ),
            "sidecar_skipped_queries": sum(
                row["sidecar"]["skipped_queries"] for row in pairs
            ),
            "sidecar_strict_correct_queries": total_correct,
            "sidecar_strict_accuracy": total_correct / total_queries if total_queries else None,
            "sidecar_strict_correct_before_trainer_stop": free_window_correct,
            "sidecar_strict_accuracy_before_trainer_stop": (
                free_window_correct / free_window_queries if free_window_queries else None
            ),
            "detached_rank_seconds": sum(
                row["sidecar"]["detached_rank_seconds"] for row in pairs
            ),
            "sidecar_used_rank_seconds": sum(
                row["sidecar"]["sidecar_used_rank_seconds"] for row in pairs
            ),
            "drain_detached_rank_seconds": sum(
                row["sidecar"]["drain_detached_rank_seconds"] for row in pairs
            ),
            "drain_sidecar_used_rank_seconds": sum(
                row["sidecar"]["drain_sidecar_used_rank_seconds"] for row in pairs
            ),
            "occupied_detached_rank_seconds": sum(
                row["sidecar"]["occupied_detached_rank_seconds"] for row in pairs
            ),
            "occupied_sidecar_used_rank_seconds": sum(
                row["sidecar"]["occupied_sidecar_used_rank_seconds"] for row in pairs
            ),
        },
    }
    return summary


def _fmt(value: Any, digits: int = 2) -> str:
    if value is None:
        return "NA"
    return f"{float(value):.{digits}f}"


def render_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Qwen2.5-1.5B Sidecar Paired Evidence",
        "",
        f"Status: **{summary['status']}**",
        "",
        "Each row is a request-matched one-step Planned floor4 run without TailGuard. "
        "Primary throughput counts generated response tokens only.",
        "",
        "| Seed | Off tokens/s | On tokens/s | Delta (%) | Rollout time delta (%) | "
        "Free-window queries | Drain completions | Total completions | "
        "Strict accuracy (%) | Free-window used rank-s | Free-window detached rank-s |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for pair in summary["pairs"]:
        lines.append(
            "| {seed} | {off} | {on} | {delta} | {time_delta} | {queries} | "
            "{drain} | {total} | {accuracy} | {used} | {detached} |".format(
                seed=pair["seed"],
                off=_fmt(pair["off"]["response_token_throughput"]),
                on=_fmt(pair["on"]["response_token_throughput"]),
                delta=_fmt(pair["paired_primary_throughput_delta_percent"]),
                time_delta=_fmt(pair["paired_rollout_time_delta_percent"]),
                queries=pair["sidecar"]["completed_before_trainer_stop"],
                drain=pair["sidecar"]["completed_during_shutdown"],
                total=pair["sidecar"]["completed_queries"],
                accuracy=_fmt(100.0 * pair["sidecar"]["strict_accuracy"]),
                used=_fmt(pair["sidecar"]["sidecar_used_rank_seconds"]),
                detached=_fmt(pair["sidecar"]["detached_rank_seconds"]),
            )
        )
    aggregate = summary["aggregate"]
    delta = aggregate["primary_throughput_delta_percent"]
    lines.extend(
        [
            "",
            "## Aggregate",
            "",
            f"Paired primary throughput delta mean is {_fmt(delta['mean'])}% "
            f"with minimum {_fmt(delta['min'])}% and maximum {_fmt(delta['max'])}%.",
        ]
    )
    if delta["ci95"] is not None:
        lines.append(
            f"The two-sided t 95% CI is [{_fmt(delta['ci95'][0])}%, "
            f"{_fmt(delta['ci95'][1])}%] with df=2."
        )
    lines.extend(
        [
            f"Free-window sidecar queries total "
            f"{aggregate['sidecar_completed_before_trainer_stop']} with strict GSM8K "
            f"accuracy {_fmt(100.0 * aggregate['sidecar_strict_accuracy_before_trainer_stop'])}%.",
            f"Shutdown drain adds {aggregate['sidecar_completed_during_shutdown']} completions. "
            f"All {aggregate['sidecar_completed_queries']} completions have strict GSM8K "
            f"accuracy {_fmt(100.0 * aggregate['sidecar_strict_accuracy'])}%.",
            f"Completed sidecar output tokens total {aggregate['sidecar_completed_output_tokens']}.",
            f"Free-window sidecar-used rank-time totals "
            f"{_fmt(aggregate['sidecar_used_rank_seconds'])} rank-s. "
            f"Free-window detached capacity totals "
            f"{_fmt(aggregate['detached_rank_seconds'])} rank-s.",
            f"Shutdown drain occupies "
            f"{_fmt(aggregate['drain_sidecar_used_rank_seconds'])} used rank-s.",
            f"Restore leaves {aggregate['sidecar_partial_queries_at_restore']} checkpointed partial "
            f"queries and records {aggregate['sidecar_skipped_queries']} skipped queries.",
            "",
        ]
    )
    if summary["pending"]:
        lines.extend(["## Pending", ""])
        for row in summary["pending"]:
            lines.append(
                f"Seed {row['seed']} is missing {', '.join(row['missing_arms'])}."
            )
        lines.append("")
    return "\n".join(lines)


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        handle.write(text)
        temporary = Path(handle.name)
    temporary.replace(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="paired experiment root")
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="summarize completed pairs while later seeds are still missing",
    )
    args = parser.parse_args(argv)
    try:
        summary = verify(args.root, allow_incomplete=args.allow_incomplete)
        root = args.root.expanduser().resolve()
        json_path = root / SUMMARY_JSON
        markdown_path = root / SUMMARY_MD
        _atomic_write(json_path, json.dumps(summary, indent=2, sort_keys=True) + "\n")
        _atomic_write(markdown_path, render_markdown(summary))
    except VerificationError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 2
    print(f"{summary['status']}: {json_path}")
    print(f"{summary['status']}: {markdown_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
