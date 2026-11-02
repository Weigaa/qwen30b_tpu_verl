#!/usr/bin/env python3
"""Verify representative Planned and Natural FULL_DECODE_ONLY graph arms."""

from __future__ import annotations

import argparse
import importlib.util
import json
import statistics
from pathlib import Path
from typing import Any


SOURCE_ROOT = Path(__file__).resolve().parents[1]
LEGACY_VERIFIER = SOURCE_ROOT / "tools/verify_qwen3_adafloor_graph_eager_full_pair.py"
POLICIES = ("planned", "natural")
CAPTURE_SIZES = [16, 32]


class VerificationError(RuntimeError):
    pass


def fail(message: str) -> None:
    raise VerificationError(message)


def load_legacy():
    spec = importlib.util.spec_from_file_location("full_graph_matrix_verifier", LEGACY_VERIFIER)
    if spec is None or spec.loader is None:
        fail(f"cannot load verifier support: {LEGACY_VERIFIER}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.CAPTURE_SIZES = CAPTURE_SIZES
    module.SEED = 0
    return module


def load_json(path: Path, label: str) -> Any:
    if path.is_symlink() or not path.is_file():
        fail(f"missing regular {label}: {path}")
    try:
        return json.loads(path.read_text("utf-8"))
    except json.JSONDecodeError as error:
        raise VerificationError(f"invalid {label}: {path}: {error}") from error


def read_manifest(root: Path) -> dict[str, Path]:
    path = root / "run_manifest.tsv"
    if not path.exists():
        return {}
    lines = path.read_text("utf-8").splitlines()
    if not lines or lines[0] != "policy\tstatus\tarm_manifest":
        fail("run manifest header is invalid")
    result: dict[str, Path] = {}
    for number, line in enumerate(lines[1:], 2):
        fields = line.split("\t")
        if len(fields) != 3:
            fail(f"invalid run manifest line {number}")
        policy, status, manifest = fields
        if policy not in POLICIES or policy in result or status != "complete":
            fail(f"invalid run manifest row {number}: {fields}")
        result[policy] = Path(manifest)
    return result


def verify_protocol(root: Path, legacy) -> dict[str, str]:
    values = legacy.parse_env(root / "protocol.env")
    expected = {
        "schema_version": "1",
        "experiment": "qwen3_adafloor_full_decode_fia_v014opt_tq1",
        "common_epoch0_mode": "graph_vanilla_buckets16_32",
        "policies": "planned,natural",
        "planned_variant": "adafloor_p_f4",
        "natural_variant": "adafloor_n_f2",
        "actor_frozen": "true",
        "seed": "0",
        "steps": "5",
        "prompts_per_step": "32",
        "rollout_n": "16",
        "responses_per_step": "512",
        "max_response_length": "16384",
        "max_num_seqs": "32",
        "full16_kv_tokens": "380800",
        "planned_floor4_kv_tokens": "280576",
        "planned_floor8_kv_tokens": "377344",
        "natural_floor2_kv_tokens": "98304",
        "natural_floor4_kv_tokens": "280576",
        "natural_floor8_kv_tokens": "315648",
        "graph_mode": "FULL_DECODE_ONLY",
        "capture_sizes": "[16,32]",
        "cudagraph_copy_inputs": "false",
        "attention_backend": "fia_max_workspace",
        "attention_graph": "true",
        "moe_graph": "true",
        "prefill_execution": "eager",
        "task_queue_enable": "1",
        "max_update_actor_ms_per_token": "0.08",
        "native_sleep_mode": "0",
        "async_scheduling": "false",
        "torchair": "false",
        "graphex": "false",
        "sidecar": "false",
        "tail_guard": "enabled",
    }
    for key, expected_value in expected.items():
        if values.get(key) != expected_value:
            fail(f"protocol {key}={values.get(key)!r}, expected {expected_value!r}")
    common = Path(values.get("common_epoch0_root", ""))
    required_common = (
        common / "DO_NOT_DELETE_COMMON_EPOCH0_CHECKPOINT",
        common / "reuse.env",
        common / "epoch_000_mode0_probe/offline_planning_history.json",
        common / "epoch_000_mode0_probe/checkpoints/qwen3moe_for_eagle3/global_step_5",
    )
    for path in required_common:
        if not path.exists():
            fail(f"common epoch0 artifact is missing: {path}")
    legacy.verify_code_contract(root)
    return values


def verify_arm(policy: str, manifest_path: Path, legacy) -> dict[str, Any]:
    manifest = load_json(manifest_path, f"{policy} arm manifest")
    expected = {
        "schema_version": 1,
        "experiment": "qwen3_adafloor_full_decode_fia_v014opt_tq1",
        "policy": policy,
        "variant": "adafloor_p_f4" if policy == "planned" else "adafloor_n_f2",
        "seed": 0,
        "steps": 5,
        "responses_per_step": 512,
        "actor_frozen": True,
        "graph_mode": "FULL_DECODE_ONLY",
        "capture_sizes": CAPTURE_SIZES,
        "attention_graph": True,
        "moe_graph": True,
        "task_queue_enable": 1,
        "tail_guard_enabled": True,
    }
    for key, expected_value in expected.items():
        if manifest.get(key) != expected_value:
            fail(f"{policy} manifest {key}={manifest.get(key)!r}, expected {expected_value!r}")
    epoch_dir = Path(manifest["epoch_dir"])
    plan_path = Path(manifest["plan_file"])
    log_path = Path(manifest["log_file"])
    cleanup = Path(manifest["checkpoint_cleanup_marker"])
    if not cleanup.is_file():
        fail(f"{policy} checkpoint cleanup marker is missing")
    plan = legacy.verify_plan(plan_path, policy)
    outputs, _requests = legacy.verify_outputs(epoch_dir)
    runtime = legacy.verify_log(log_path, "graph", policy, plan)
    if policy == "planned" and min(plan["selected_floors"]) < 4:
        fail("Planned-F4 selected a floor below four")
    if policy == "natural" and min(plan["selected_floors"]) < 2:
        fail("Natural-F2 selected an invalid floor")
    expected_kv = (
        {4: 280576, 8: 377344, 16: 380800}
        if policy == "planned"
        else {2: 98304, 4: 280576, 8: 315648, 16: 380800}
    )
    for step, (floor, kv_tokens) in enumerate(
        zip(plan["selected_floors"], runtime["kv_tokens_by_step"]), 1
    ):
        if kv_tokens != expected_kv[floor]:
            fail(
                f"{policy} step {step} floor{floor} KV={kv_tokens}, "
                f"expected {expected_kv[floor]}"
            )
    generated_tokens = sum(row["generated_tokens"] for row in outputs.values())
    rollout_seconds = runtime["rollout_seconds_total"]
    if generated_tokens <= 0 or rollout_seconds <= 0:
        fail(f"{policy} has invalid generated work or rollout time")
    rewards = [row["reward_mean"] for row in runtime["metrics"]]
    return {
        "status": "PASS",
        "policy": policy,
        "variant": manifest["variant"],
        "selected_floors": plan["selected_floors"],
        "executed_floors": runtime["executed_floors"],
        "transition_active_sets": runtime["transition_active_sets"],
        "kv_tokens_by_step": runtime["kv_tokens_by_step"],
        "generated_response_tokens": generated_tokens,
        "rollout_seconds": rollout_seconds,
        "response_tokens_per_second": generated_tokens / rollout_seconds,
        "step_seconds": runtime["step_seconds_total"],
        "mean_runtime_reward": statistics.fmean(rewards),
        "reward_by_step": rewards,
        "graph": runtime["graph"],
        "aborted_responses": runtime["aborted_responses"],
        "preemptions": runtime["preemptions"],
        "fault_counts": runtime["fault_counts"],
        "plan": legacy.artifact(plan_path),
        "log": legacy.artifact(log_path),
        "manifest": legacy.artifact(manifest_path),
        "cleanup_marker": legacy.artifact(cleanup),
    }


def write_summary(root: Path, summary: dict[str, Any]) -> None:
    json_path = root / "adafloor_full_decode_summary.json"
    md_path = root / "adafloor_full_decode_summary.md"
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    lines = [
        "# Qwen3 AdaFloor FULL_DECODE_ONLY Summary",
        "",
        f"Status: {summary['status']}",
        "",
        "| Policy | Floors selected | Floors executed | Response tokens | Rollout seconds | Response tok/s | Mean reward |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for policy in POLICIES:
        arm = summary["arms"].get(policy)
        if arm is None:
            continue
        lines.append(
            f"| {policy} | {arm['selected_floors']} | {arm['executed_floors']} | "
            f"{arm['generated_response_tokens']} | {arm['rollout_seconds']:.3f} | "
            f"{arm['response_tokens_per_second']:.3f} | {arm['mean_runtime_reward']:.6f} |"
        )
    lines.extend(
        [
            "",
            "Both arms use native FULL_DECODE_ONLY ACLGraph with [16,32] capture sizes.",
            "Prefill and topology control remain eager. Summary status is PASS only when",
            "both Planned-F4 and Natural-F2 complete all five steps and strict graph,",
            "output, transition, safety, and checkpoint-cleanup checks pass.",
            "",
        ]
    )
    md_path.write_text("\n".join(lines))


def verify(root: Path, allow_incomplete: bool) -> dict[str, Any]:
    legacy = load_legacy()
    protocol = verify_protocol(root, legacy)
    manifests = read_manifest(root)
    arms = {
        policy: verify_arm(policy, path, legacy)
        for policy, path in manifests.items()
    }
    missing = [policy for policy in POLICIES if policy not in arms]
    if missing and not allow_incomplete:
        fail(f"missing completed policies: {missing}")
    summary = {
        "schema_version": 1,
        "status": "PASS" if not missing else "INCOMPLETE",
        "experiment": "qwen3_adafloor_full_decode_fia_v014opt_tq1",
        "root": str(root.resolve()),
        "common_epoch0_root": protocol["common_epoch0_root"],
        "capture_sizes": CAPTURE_SIZES,
        "completed_policies": sorted(arms),
        "missing_policies": missing,
        "arms": arms,
    }
    write_summary(root, summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    try:
        summary = verify(args.root, args.allow_incomplete)
    except (VerificationError, RuntimeError) as error:
        parser.error(str(error))
    print(json.dumps({
        "status": summary["status"],
        "completed_policies": summary["completed_policies"],
        "root": summary["root"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
