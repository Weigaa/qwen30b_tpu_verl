#!/usr/bin/env python3
"""Verify an exact-work DeepSeek batch-64 LengthSort and AdaFloor replay."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from tools.verify_deepseek_batch64_pair import (
    EXPECTED_PHASE_STEPS,
    RESPONSES_PER_STEP,
    VerificationError,
    _load_env,
    _resolve_epoch,
    _write_atomic,
    verify_pair,
)
from verl.utils.fixed_work_replay import (
    FixedWorkReplay,
    FixedWorkReplayError,
    load_fixed_work_replay,
)


PROTOCOL = "deepseek_batch64_fixed_work_replay_v3"


def _fail(message: str) -> None:
    raise VerificationError(message)


def _require_int(value: Any, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        _fail(f"{context} must be an integer")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _validate_fixed_manifest(
    run_dir: Path,
    arm: str,
    phase: str,
    trace_path: Path,
    trace_sha256: str,
) -> Path:
    epoch_dir, manifest_path = _resolve_epoch(run_dir, arm)
    values = _load_env(manifest_path)
    expected = {
        "DEEPSEEK_BATCH64_FIXED_WORK_PROTOCOL": PROTOCOL,
        "DEEPSEEK_BATCH64_FIXED_WORK_TRACE": str(trace_path.resolve()),
        "DEEPSEEK_BATCH64_FIXED_WORK_TRACE_SHA256": trace_sha256,
        "DEEPSEEK_BATCH64_PHASE": phase,
        "DEEPSEEK_BATCH64_ARM": arm,
    }
    for key, expected_value in expected.items():
        if values.get(key) != expected_value:
            _fail(
                f"{arm} fixed-work manifest {key}={values.get(key)!r}, "
                f"expected {expected_value!r}"
            )
    return epoch_dir


def _validate_arm_rows(
    epoch_dir: Path,
    trace: FixedWorkReplay,
    expected_steps: int,
    arm: str,
) -> dict[str, Any]:
    observed_tokens = 0
    observed_source_tokens = 0
    stable_request_keys: list[tuple[int, int]] = []
    per_step: list[dict[str, Any]] = []
    expected_trace_sha256 = trace.trace_sha256
    rollout_dir = epoch_dir / "rollout_data"
    for step in range(1, expected_steps + 1):
        path = rollout_dir / f"{step}.jsonl"
        if not path.is_file():
            _fail(f"missing {arm} fixed-work rollout artifact {path}")
        rows = 0
        step_tokens = 0
        step_source_tokens = 0
        step_source_steps: set[int] = set()
        with path.open(encoding="utf-8") as handle:
            for file_ordinal, line in enumerate(handle):
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as error:
                    raise VerificationError(
                        f"invalid {arm} fixed-work JSON at step={step} "
                        f"row={file_ordinal}: {error}"
                    ) from error
                if not isinstance(row, dict):
                    _fail(
                        f"{arm} fixed-work step={step} row={file_ordinal} "
                        "is not an object"
                    )
                ordinal = _require_int(
                    row.get("fixed_work_replay_row_ordinal"),
                    f"{arm} step={step} fixed_work_replay_row_ordinal",
                )
                if ordinal != file_ordinal:
                    _fail(
                        f"{arm} step={step} row order differs from replay ordinal, "
                        f"file_row={file_ordinal} replay_row={ordinal}"
                    )
                occurrence = _require_int(
                    row.get("prompt_occurrence_ordinal"),
                    f"{arm} step={step} prompt_occurrence_ordinal",
                )
                prompt_hash = row.get("rollout_prompt_hash")
                if not isinstance(prompt_hash, str) or not prompt_hash:
                    _fail(f"{arm} step={step} row={ordinal} has no prompt hash")
                identity = (
                    prompt_hash,
                    _require_int(
                        row.get("rollout_sample_index"),
                        f"{arm} step={step} sample index",
                    ),
                    _require_int(
                        row.get("rollout_request_seed"),
                        f"{arm} step={step} request seed",
                    ),
                )
                target = trace.target_for_occurrence(
                    occurrence,
                    identity[1],
                    identity[0],
                    identity[2],
                )
                source = trace.source_length_for_occurrence(
                    occurrence,
                    identity[1],
                    identity[0],
                    identity[2],
                )
                expected_source_row = trace.source_row_for_occurrence(
                    occurrence,
                    identity[1],
                    identity[0],
                    identity[2],
                )
                expected_source_step = trace.source_step_for_occurrence(
                    occurrence,
                    identity[1],
                    identity[0],
                    identity[2],
                )
                recorded_source_step = _require_int(
                    row.get("fixed_work_replay_source_step"),
                    f"{arm} step={step} fixed-work source step",
                )
                if recorded_source_step != expected_source_step:
                    _fail(
                        f"{arm} runtime step={step} occurrence={occurrence} "
                        f"records source step {recorded_source_step}, expected "
                        f"{expected_source_step}"
                    )
                if arm == "adafloor" and expected_source_step != step:
                    _fail(
                        "fixed AdaFloor request moved across its source plan step, "
                        f"occurrence={occurrence} source_step="
                        f"{expected_source_step} runtime_step={step}"
                    )
                step_source_steps.add(expected_source_step)
                recorded_source_row = _require_int(
                    row.get("fixed_work_replay_source_row_ordinal"),
                    f"{arm} step={step} fixed-work source row ordinal",
                )
                if recorded_source_row != expected_source_row:
                    _fail(
                        f"{arm} step={step} occurrence={occurrence} "
                        f"sample={identity[1]} records source row "
                        f"{recorded_source_row}, expected {expected_source_row}"
                    )
                recorded_source = _require_int(
                    row.get("fixed_work_replay_source_length"),
                    f"{arm} step={step} fixed-work source length",
                )
                recorded_target = _require_int(
                    row.get("fixed_work_replay_target_length"),
                    f"{arm} step={step} fixed-work target length",
                )
                decoded = _require_int(
                    row.get("decoded_response_length"),
                    f"{arm} step={step} decoded response length",
                )
                if (recorded_source, recorded_target, decoded) != (
                    source,
                    target,
                    target,
                ):
                    _fail(
                        f"{arm} step={step} row={ordinal} length contract differs, "
                        f"source={recorded_source}/{source} "
                        f"target={recorded_target}/{target} decoded={decoded}"
                    )
                if row.get("fixed_work_replay_trace_sha256") != expected_trace_sha256:
                    _fail(
                        f"{arm} step={step} row={ordinal} records the wrong trace hash"
                    )
                if row.get("response_finish_reason") != "length":
                    _fail(
                        f"{arm} step={step} row={ordinal} did not finish at its "
                        "fixed-work target"
                    )
                response_mask = row.get("response_mask")
                if not isinstance(response_mask, list):
                    _fail(f"{arm} step={step} row={ordinal} has no response mask")
                if sum(response_mask) != target:
                    _fail(
                        f"{arm} step={step} row={ordinal} response mask does not "
                        "match the replay target"
                    )
                step_source_tokens += source
                step_tokens += target
                stable_request_keys.append((occurrence, identity[1]))
                rows += 1
        if rows != RESPONSES_PER_STEP:
            _fail(
                f"{arm} fixed-work step={step} has {rows} rows, "
                f"expected {RESPONSES_PER_STEP}"
            )
        observed_tokens += step_tokens
        observed_source_tokens += step_source_tokens
        per_step.append(
            {
                "step": step,
                "responses": rows,
                "source_tokens": step_source_tokens,
                "replayed_tokens": step_tokens,
                "source_plan_steps": sorted(step_source_steps),
                "source_plan_response_caps": sorted(
                    {trace.step_cap(source_step) for source_step in step_source_steps}
                ),
            }
        )
    if observed_tokens != trace.target_generated_tokens:
        _fail(f"{arm} fixed-work token total differs from trace")
    if observed_source_tokens != trace.source_generated_tokens:
        _fail(f"{arm} fixed-work source-token total differs from trace")
    expected_stable_keys = {
        (occurrence, sample)
        for step in trace.steps
        for occurrence in trace.prompt_occurrences_for_step(step)
        for sample in range(16)
    }
    if len(stable_request_keys) != len(set(stable_request_keys)):
        _fail(f"{arm} fixed-work artifacts contain duplicate stable requests")
    if set(stable_request_keys) != expected_stable_keys:
        _fail(f"{arm} fixed-work stable request multiset differs from trace")
    stable_key_sha256 = hashlib.sha256(
        json.dumps(sorted(stable_request_keys), separators=(",", ":")).encode(
            "ascii"
        )
    ).hexdigest()
    return {
        "responses": expected_steps * RESPONSES_PER_STEP,
        "source_tokens": observed_source_tokens,
        "replayed_tokens": observed_tokens,
        "stable_request_count": len(stable_request_keys),
        "stable_request_multiset_sha256": stable_key_sha256,
        "steps": per_step,
    }


def verify_fixed_pair(
    *,
    phase: str,
    vanilla_run_dir: Path,
    adafloor_run_dir: Path,
    common_root: Path,
    cap_env: Path,
    workload_profile_env: Path,
    trace_path: Path,
    trace_sha256: str,
    expected_execution_code_sha256: str,
) -> dict[str, Any]:
    trace = load_fixed_work_replay(
        trace_path,
        expected_sha256=trace_sha256,
    )
    expected_steps = EXPECTED_PHASE_STEPS[phase]
    if trace.steps != tuple(range(1, expected_steps + 1)):
        _fail(
            f"fixed-work trace steps={trace.steps}, expected 1..{expected_steps}"
        )
    vanilla_epoch = _validate_fixed_manifest(
        vanilla_run_dir, "vanilla", phase, trace_path, trace.trace_sha256
    )
    adafloor_epoch = _validate_fixed_manifest(
        adafloor_run_dir, "adafloor", phase, trace_path, trace.trace_sha256
    )
    summary = verify_pair(
        phase,
        vanilla_run_dir,
        adafloor_run_dir,
        common_root,
        cap_env,
        workload_profile_env,
        expected_execution_code_sha256,
    )
    vanilla_fixed = _validate_arm_rows(
        vanilla_epoch, trace, expected_steps, "vanilla"
    )
    adafloor_fixed = _validate_arm_rows(
        adafloor_epoch, trace, expected_steps, "adafloor"
    )
    for field in (
        "responses",
        "source_tokens",
        "replayed_tokens",
        "stable_request_count",
        "stable_request_multiset_sha256",
    ):
        if vanilla_fixed[field] != adafloor_fixed[field]:
            _fail(f"fixed-work arms differ in global work field {field}")
    if summary["vanilla"]["generated_tokens"] != trace.target_generated_tokens:
        _fail("base Vanilla token accounting differs from fixed-work trace")
    if summary["adafloor"]["generated_tokens"] != trace.target_generated_tokens:
        _fail("base AdaFloor token accounting differs from fixed-work trace")
    if trace.adafloor_plan_path is None or trace.adafloor_plan_sha256 is None:
        _fail("fixed-work trace lacks source AdaFloor plan provenance")
    executed_plan = adafloor_epoch / "oracle" / "length_sorted_rank_plan.json"
    if not executed_plan.is_file():
        _fail(f"fixed AdaFloor arm lacks its executed raw plan: {executed_plan}")
    executed_plan_sha256 = _sha256(executed_plan)
    if executed_plan_sha256 != trace.adafloor_plan_sha256:
        _fail(
            "fixed AdaFloor executed plan differs from the source Natural plan, "
            f"source_sha256={trace.adafloor_plan_sha256} "
            f"executed_sha256={executed_plan_sha256}"
        )
    summary["protocol"] = (
        "DeepSeek-V2-Lite-Chat exact-work batch64 replay, LengthSort Full16 versus "
        "AdaFloor Natural floor2"
    )
    summary["fixed_work"] = {
        "protocol": PROTOCOL,
        "trace": str(trace.path),
        "trace_sha256": trace.trace_sha256,
        "trace_file_sha256": _sha256(trace.path),
        "adafloor_plan": (
            None if trace.adafloor_plan_path is None else str(trace.adafloor_plan_path)
        ),
        "adafloor_plan_sha256": trace.adafloor_plan_sha256,
        "source_generated_tokens": trace.source_generated_tokens,
        "replayed_generated_tokens_per_arm": trace.target_generated_tokens,
        "arms_exactly_equal": True,
        "stable_occurrence_multiset_equal": (
            vanilla_fixed["stable_request_multiset_sha256"]
            == adafloor_fixed["stable_request_multiset_sha256"]
        ),
        "source_and_fixed_adafloor_plan_exactly_equal": True,
        "fixed_adafloor_executed_plan": str(executed_plan.resolve()),
        "fixed_adafloor_executed_plan_sha256": executed_plan_sha256,
        "per_arm": {
            "vanilla": vanilla_fixed,
            "adafloor": adafloor_fixed,
        },
        "reward_scope": (
            "Replay reward is diagnostic only because exact-length forcing "
            "suppresses natural stopping. Natural-generation runs provide "
            "the quality comparison."
        ),
    }
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=sorted(EXPECTED_PHASE_STEPS), required=True)
    parser.add_argument("--vanilla-run-dir", type=Path, required=True)
    parser.add_argument("--adafloor-run-dir", type=Path, required=True)
    parser.add_argument("--common-root", type=Path, required=True)
    parser.add_argument("--cap-env", type=Path, required=True)
    parser.add_argument("--workload-profile-env", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--trace-sha256", required=True)
    parser.add_argument("--expected-execution-code-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = verify_fixed_pair(
            phase=args.phase,
            vanilla_run_dir=args.vanilla_run_dir,
            adafloor_run_dir=args.adafloor_run_dir,
            common_root=args.common_root,
            cap_env=args.cap_env,
            workload_profile_env=args.workload_profile_env,
            trace_path=args.trace,
            trace_sha256=args.trace_sha256,
            expected_execution_code_sha256=args.expected_execution_code_sha256,
        )
        _write_atomic(args.output, result)
    except (FixedWorkReplayError, OSError, VerificationError) as error:
        try:
            _write_atomic(
                args.output,
                {"status": "FAIL", "phase": args.phase, "error": str(error)},
            )
        except OSError:
            pass
        print(f"FAIL: {error}")
        return 1
    print(f"PASS: {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
