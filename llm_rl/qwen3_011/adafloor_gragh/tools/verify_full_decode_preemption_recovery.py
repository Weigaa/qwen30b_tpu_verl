#!/usr/bin/env python3
"""Fail-closed audit for synchronized FULL_DECODE preemption recovery."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
from collections import Counter
from pathlib import Path


PREEMPT = re.compile(r"preempting request|request preempted", re.IGNORECASE)
ENTER = re.compile(
    r"FULL_DECODE_ONLY DP rank entered synchronized eager recovery: "
    r"dp_rank=(\d+)"
)
RESUME = re.compile(
    r"FULL_DECODE_ONLY DP rank resumed FULL graph after synchronized "
    r"eager recovery: dp_rank=(\d+)"
)
LOCAL_JOIN = re.compile(
    r"FULL_DECODE_ONLY local preemption recovery joined the "
    r"DP-synchronized eager iteration: dp_rank=(\d+)"
)
ABORT = re.compile(r"response/aborted_ratio:([0-9.eE+-]+)")
FATAL = re.compile(
    r"RayTaskError|RuntimeError|NPU out of memory|Memory_Allocation_Failure|"
    r"OutOfMemoryError|ACL_ERROR_RT_MEMORY_ALLOCATION|ERR\d+"
)


class AuditError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit_log(log_path: Path, world_size: int, expected_steps: int) -> dict:
    lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    active = {rank: False for rank in range(world_size)}
    enters: Counter[int] = Counter()
    resumes: Counter[int] = Counter()
    local_joins: Counter[int] = Counter()
    preemption_indices: list[int] = []
    local_join_indices: list[int] = []

    for index, line in enumerate(lines):
        if PREEMPT.search(line):
            preemption_indices.append(index)
        if match := ENTER.search(line):
            rank = int(match.group(1))
            if rank not in active:
                raise AuditError(f"enter event has invalid dp_rank={rank}")
            if active[rank]:
                raise AuditError(f"dp_rank={rank} entered recovery twice without resume")
            active[rank] = True
            enters[rank] += 1
        if match := RESUME.search(line):
            rank = int(match.group(1))
            if rank not in active:
                raise AuditError(f"resume event has invalid dp_rank={rank}")
            if not active[rank]:
                raise AuditError(f"dp_rank={rank} resumed FULL without entering recovery")
            active[rank] = False
            resumes[rank] += 1
        if match := LOCAL_JOIN.search(line):
            rank = int(match.group(1))
            if rank not in active:
                raise AuditError(f"local join has invalid dp_rank={rank}")
            if not active[rank]:
                raise AuditError(f"dp_rank={rank} joined recovery outside an eager round")
            local_joins[rank] += 1
            local_join_indices.append(index)

    if not preemption_indices:
        raise AuditError("log contains no KV preemption event")
    if not local_join_indices:
        raise AuditError("log contains no local preemption recovery join")
    if max(local_join_indices) <= max(preemption_indices):
        raise AuditError("last KV preemption has no subsequent local recovery join")
    if any(active.values()):
        pending = sorted(rank for rank, value in active.items() if value)
        raise AuditError(f"ranks ended in eager recovery without FULL resume: {pending}")

    expected_ranks = set(range(world_size))
    if set(enters) != expected_ranks or set(resumes) != expected_ranks:
        raise AuditError("enter/resume events do not cover every DP rank")
    enter_values = set(enters.values())
    resume_values = set(resumes.values())
    if len(enter_values) != 1 or len(resume_values) != 1:
        raise AuditError("DP ranks observed different recovery-round counts")
    if enters != resumes:
        raise AuditError("per-rank enter and FULL-resume counts differ")
    rounds = next(iter(enter_values))
    if rounds <= 0 or sum(enters.values()) != rounds * world_size:
        raise AuditError("recovery events do not form complete DP-world rounds")

    abort_values = [float(match.group(1)) for line in lines if (match := ABORT.search(line))]
    if len(abort_values) != expected_steps or any(value != 0 for value in abort_values):
        raise AuditError("missing or nonzero aborted-response metric")
    if sum("rollout_output_time_s:" in line for line in lines) != expected_steps:
        raise AuditError("rollout metric count does not match expected steps")
    if sum("After trainer.fit" in line for line in lines) != 1:
        raise AuditError("trainer completion marker is missing or duplicated")
    if not any("Graph capturing finished" in line for line in lines):
        raise AuditError("FULL graph capture marker is missing")
    if not any("Replaying aclgraph" in line for line in lines):
        raise AuditError("ACLGraph replay marker is missing")
    fatal_lines = [line for line in lines if FATAL.search(line)]
    if fatal_lines:
        raise AuditError(f"contract-defined fatal marker found: {fatal_lines[0]}")

    return {
        "schema_version": 1,
        "status": "PASS",
        "log_path": str(log_path.resolve()),
        "log_sha256": _sha256(log_path),
        "world_size": world_size,
        "expected_steps": expected_steps,
        "kv_preemption_events": len(preemption_indices),
        "synchronized_recovery_rounds": rounds,
        "enter_events": sum(enters.values()),
        "full_resume_events": sum(resumes.values()),
        "local_preemption_join_events": sum(local_joins.values()),
        "local_preemption_join_ranks": sorted(local_joins),
        "aborted_responses": 0,
        "claim_boundary": (
            "This audit verifies run-specific synchronized eager recovery and "
            "subsequent FULL graph resume. It does not make preemption free."
        ),
    }


def _write_json(path: Path, result: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write("\n")
        temporary = Path(stream.name)
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--world-size", type=int, default=16)
    parser.add_argument("--expected-steps", type=int, default=5)
    args = parser.parse_args()
    try:
        result = audit_log(args.log, args.world_size, args.expected_steps)
    except AuditError as error:
        parser.error(str(error))
    _write_json(args.output, result)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
