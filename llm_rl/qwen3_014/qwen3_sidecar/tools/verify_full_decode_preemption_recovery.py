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


def audit_native_v014(log_path: Path, world_size: int, expected_steps: int) -> dict:
    """Audit native v0.14 metadata-driven recovery, not old v0.11 joins.

    The native runner OR-reduces with_prefill in existing DP metadata sync;
    it does not emit the legacy join/enter/resume strings. Do not fabricate
    those events or infer that absent legacy strings mean no recovery.
    """
    text = re.sub(r"\x1b\[[0-9;]*[mK]", "", log_path.read_text(errors="replace"))
    lines = text.replace("\r", "\n").splitlines()
    if not re.search(r"Initializing a V1 LLM engine \(v0\.14", text):
        raise AuditError("native-v014 audit requires actual v0.14 engine evidence")
    if not re.search(r"\[run\] end_time=\S+ exit_code=0", text):
        raise AuditError("native-v014 successful child completion missing")
    if len(re.findall(r"=> rollout_output_time_s:\s*[\d.]+", text)) != expected_steps:
        raise AuditError("native-v014 missing rollout metrics")
    aborts = [float(x) for x in ABORT.findall(text)]
    if len(aborts) != expected_steps or any(aborts):
        raise AuditError("native-v014 missing/nonzero abort metrics")
    if re.search(r"RayTaskError|RuntimeError:|NPU out of memory|Memory_Allocation_Failure|OutOfMemoryError",text):
        raise AuditError("native-v014 fatal runtime exception")
    # Elastic DP ranks are renumbered inside each survivor group. The Ray
    # worker stays on its original global rank, recorded in its runtime
    # audit. Attribute progress to that physical worker, not its local DP id.
    worker_ranks = {}
    for line in lines:
        pid = re.search(r"\(WorkerDict pid=(\d+)\)", line)
        if pid and "MODE1_STEP_RUNTIME_AUDIT " in line:
            a = json.JSONDecoder().raw_decode(line.split("MODE1_STEP_RUNTIME_AUDIT ",1)[1])[0]
            rank = int(a["rank"])
            if pid[1] in worker_ranks and worker_ranks[pid[1]] != rank:
                raise AuditError("native-v014 worker PID changed global rank")
            worker_ranks[pid[1]] = rank
    records, preemptions, full_progress, audit_progress = {}, {}, {}, {}
    worker_steps = {}
    step = 1
    for index, line in enumerate(lines):
        pid_match = re.search(r"\(WorkerDict pid=(\d+)\)", line)
        pid = pid_match[1] if pid_match else None
        if "TAILGUARD_CAP_AUDIT " in line and pid:
            cap = json.JSONDecoder().raw_decode(line.split("TAILGUARD_CAP_AUDIT ",1)[1])[0]
            if pid in worker_ranks and int(cap["rank"]) != worker_ranks[pid]:
                raise AuditError("native-v014 cap/global rank attribution mismatch")
            worker_steps[pid] = int(cap["step"])
        if "=> rollout_output_time_s:" in line:
            step += 1
        if match := re.search(r"Mode1 KV request preempted: rank=(\d+) step=(\d+)",line):
            key = (int(match[2]), int(match[1]))
            if pid in worker_ranks:
                if key[1] != worker_ranks[pid]:
                    raise AuditError("native-v014 preemption/global rank attribution mismatch")
                worker_steps[pid] = key[0]
            preemptions.setdefault(key, []).append(index)
        if match := re.search(r"rollout_forward_begin dp_rank=(\d+).*global_prefill=(True|False) graph_mode=(\w+)",line):
            if match[2] == "True" and match[3] == "FULL":
                raise AuditError("native-v014 prefill incorrectly entered FULL replay")
            if match[2] == "False" and match[3] != "FULL":
                raise AuditError("native-v014 observed eager/non-FULL decode")
            if match[3] == "FULL":
                global_rank = worker_ranks.get(pid, int(match[1]))
                source_step = worker_steps.get(pid, 1) if pid in worker_ranks else step
                full_progress[(source_step, global_rank)] = index
        marker = "MODE1_STEP_RUNTIME_AUDIT "
        if marker in line:
            a = json.JSONDecoder().raw_decode(line.split(marker,1)[1])[0]
            key = (int(a["step"]), int(a["rank"]))
            if key in records:
                raise AuditError("native-v014 duplicated rank-step audit")
            records[key] = a
            audit_progress[key] = index
            if pid:
                # A worker's own stream preserves its generation/audit
                # ordering even when Ray forwards the TaskRunner late.
                worker_steps[pid] = int(a["step"]) + 1
    expected = {(s,r) for s in range(1,expected_steps+1) for r in range(world_size)}
    if set(records) != expected:
        raise AuditError("native-v014 incomplete rank-step generation audits")
    for key,a in records.items():
        if (not a["complete"] or a.get("full_iterations",0) <= 0
                or a.get("eager_iterations") != a.get("global_prefill_iterations")
                or a.get("global_prefill_iterations",-1) < a.get("local_prefill_iterations",0)
                or a.get("worker_kv_blocks") != a.get("kv_blocks")):
            raise AuditError(f"native-v014 inconsistent eager/FULL/KV audit: {key}")
        if a["preemptions"] != len(preemptions.get(key,[])):
            raise AuditError(f"native-v014 scheduler/log preemption count mismatch: {key}")
        if a["preemptions"]:
            if a["local_prefill_iterations"] <= 1:
                raise AuditError(f"native-v014 preempted rank lacks recompute evidence: {key}")
            if audit_progress[key] <= max(preemptions[key]):
                raise AuditError(f"native-v014 recovery completion precedes last preemption: {key}")
            if key not in full_progress:
                raise AuditError(f"native-v014 lacks sampled FULL execution for preempted rank-step: {key}")
    if not preemptions:
        raise AuditError("native-v014 log has no preemptions to audit")
    # FULL heartbeats are sampled, whereas every execute_model updates the
    # counters checked above. A late preemption may have no later heartbeat;
    # a request may also finish during recompute prefill. Neither requires an
    # eager decode. Do NOT claim aggregate counters prove post-event replay.
    recovery_evidence = [dict(step=k[0], rank=k[1],
        post_preemption_full_sampled=full_progress[k] > max(preemptions[k]),
        completed_recompute=True, extra_eager_decode_iterations=0)
        for k in sorted(preemptions)]
    return dict(schema_version=3,status="PASS",backend="native_v014",
                log_path=str(log_path.resolve()),log_sha256=_sha256(log_path),
                world_size=world_size,expected_steps=expected_steps,rank_step_audits=len(records),
                kv_preemption_events=sum(map(len,preemptions.values())),
                preempted_rank_steps=[list(k) for k in sorted(preemptions)],
                recovery_evidence=recovery_evidence,
                claim_boundary="Completed native metadata-driven recompute, FULL execution in each rank-step, and no eager decode by exhaustive mode counters. Post-preemption FULL is directly observed only where post_preemption_full_sampled is true; no legacy synchronized-round or unsampled replay claim.")


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
    parser.add_argument("--backend", choices=("legacy", "native-v014"), default="legacy")
    args = parser.parse_args()
    try:
        auditor = audit_native_v014 if args.backend == "native-v014" else audit_log
        result = auditor(args.log, args.world_size, args.expected_steps)
    except AuditError as error:
        parser.error(str(error))
    _write_json(args.output, result)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
