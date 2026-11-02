#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR"

ACTION=${1:-run}
case "$ACTION" in
    run|verify) ;;
    -h|--help)
        cat <<'EOF'
Usage:
  ./run_qwen3_full_decode_preemption_recovery_smoke.sh run
  GRAPH_PREEMPTION_SMOKE_ROOT=/path/to/result \
    ./run_qwen3_full_decode_preemption_recovery_smoke.sh verify

This is a non-authoritative one-step functionality smoke. It deliberately
reduces the Vanilla KV cache to trigger recompute preemption, then verifies
that all DP ranks use synchronized eager recovery and return to FULL decode.
It is not a throughput experiment.
EOF
        exit 0
        ;;
    *) echo "unknown action: $ACTION" >&2; exit 2 ;;
esac

MAIN_RUNNER="$SCRIPT_DIR/run_qwen3_paper_graph_main_from_common_epoch0.sh"
ROOT=${GRAPH_PREEMPTION_SMOKE_ROOT:-/workspace/adafloor_graph_results/qwen3_full_decode_preemption_recovery_smoke_$(date -u +%Y%m%dT%H%M%SZ)}
SMOKE_KV_TOKENS=${GRAPH_PREEMPTION_SMOKE_KV_TOKENS:-196608}

verify_smoke() {
    python3 - "$ROOT" "$SMOKE_KV_TOKENS" "$MAIN_RUNNER" <<'PY'
from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()
kv_tokens = int(sys.argv[2])
runner = Path(sys.argv[3]).resolve()

logs = sorted(root.glob(
    "seed_0/vanilla/qwen3_graph_vanilla_seed0/"
    "epoch_001_mode0_vanilla/logs/*.txt"
))
if len(logs) != 1:
    raise SystemExit(f"expected one primary log, found {len(logs)} under {root}")
log = logs[0]
text = log.read_text(encoding="utf-8", errors="replace")
lines = text.splitlines()

for marker in (
    "FULL_DECODE_ONLY compilation enabled on NPU",
    "Native FULL_DECODE_ONLY ACLGraph enabled",
    "Graph capturing finished",
    "Replaying aclgraph",
    "Preempting request",
    "FULL_DECODE_ONLY local preemption recovery joined the DP-synchronized eager iteration",
    "FULL_DECODE_ONLY DP rank entered synchronized eager recovery",
    "FULL_DECODE_ONLY DP rank resumed FULL graph after synchronized eager recovery",
    "After trainer.fit",
):
    if marker not in text:
        raise SystemExit(f"missing smoke marker {marker!r} in {log}")

for pattern in (
    r"FULL_DECODE_ONLY DP recovery iteration unexpectedly dispatched a graph",
    r"FULL_DECODE_ONLY runtime dummy recovery iteration unexpectedly dispatched a graph",
    r"FULL_DECODE_ONLY DP long-tail iteration did not dispatch",
    r"Refusing a mixed FULL/eager collective sequence",
    r"NPU out of memory|Memory_Allocation_Failure|OutOfMemoryError",
    r"\bHCCL\b[^\n]*(?:failed|failure|fatal|error)",
    r"(?:\bERROR\b|\bERR\d+\b)[^\n]*\bHCCL\b",
    r"RayTaskError|WorkerCrashedError",
):
    if re.search(pattern, text, flags=re.IGNORECASE):
        raise SystemExit(f"forbidden smoke pattern {pattern!r} in {log}")

aborted = re.findall(r"response/aborted_ratio:([0-9.eE+-]+)", text)
if aborted != ["0.0"]:
    raise SystemExit(f"expected one zero aborted ratio, got {aborted}")

preemptions = sum("Preempting request" in line for line in lines)
entered_ranks = {
    int(match.group(1))
    for line in lines
    if (match := re.search(
        r"DP rank entered synchronized eager recovery: dp_rank=(\d+)", line
    ))
}
if entered_ranks != set(range(16)):
    raise SystemExit(
        f"synchronized eager entry did not cover all DP ranks: {sorted(entered_ranks)}"
    )

# Prove that at least one actual preemption recovery round made every rank
# take the eager path.  The entry marker is emitted only after the local
# dispatcher has returned CUDAGraphMode.NONE, so grouping the markers by the
# existing synchronized token count and wall-clock second audits the runtime
# modes without introducing another collective into the measured process.
entry_event = re.compile(
    r"INFO \d\d-\d\d (\d\d:\d\d:\d\d).*"
    r"DP rank entered synchronized eager recovery: dp_rank=(\d+) "
    r"num_tokens=(\d+)"
)
local_event = re.compile(
    r"INFO \d\d-\d\d (\d\d:\d\d:\d\d).*"
    r"local preemption recovery joined the DP-synchronized eager iteration: "
    r"dp_rank=(\d+) num_tokens=(\d+)"
)
entry_groups: dict[tuple[str, int], set[int]] = {}
local_groups: set[tuple[str, int]] = set()
for line in lines:
    if match := entry_event.search(line):
        key = (match.group(1), int(match.group(3)))
        entry_groups.setdefault(key, set()).add(int(match.group(2)))
    if match := local_event.search(line):
        local_groups.add((match.group(1), int(match.group(3))))
synchronized_preemption_rounds = [
    {
        "timestamp": key[0],
        "synced_num_tokens": key[1],
        "eager_dp_ranks": sorted(ranks),
    }
    for key, ranks in sorted(entry_groups.items())
    if key in local_groups and ranks == set(range(16))
]
if not synchronized_preemption_rounds:
    raise SystemExit(
        "no preemption recovery iteration locally verified eager mode on all DP ranks"
    )

local_recovery = re.compile(
    r"local preemption recovery joined the DP-synchronized eager iteration: "
    r"dp_rank=(\d+)"
)
resume = re.compile(
    r"DP rank resumed FULL graph after synchronized eager recovery: "
    r"dp_rank=(\d+)"
)
recovery_pairs: list[dict[str, int]] = []
for index, line in enumerate(lines):
    match = local_recovery.search(line)
    if not match:
        continue
    rank = int(match.group(1))
    resumed_at = next(
        (
            later
            for later in range(index + 1, len(lines))
            if (resume_match := resume.search(lines[later]))
            and int(resume_match.group(1)) == rank
        ),
        None,
    )
    if resumed_at is not None:
        recovery_pairs.append(
            {"dp_rank": rank, "fallback_line": index + 1, "resume_line": resumed_at + 1}
        )
if not recovery_pairs:
    raise SystemExit("no preempted rank returned from synchronized eager to FULL")

epoch = log.parents[1]
rollouts = sorted((epoch / "rollout_data").glob("*.jsonl"))
lengths = sorted((epoch / "rollout_length").glob("length_*.txt"))
if len(rollouts) != 1 or len(lengths) != 1:
    raise SystemExit("expected one rollout JSONL and one length artifact")
rollout_rows = sum(1 for _ in rollouts[0].open(encoding="utf-8"))
length_rows = sum(1 for _ in lengths[0].open(encoding="utf-8"))
if rollout_rows != 512 or length_rows != 512:
    raise SystemExit(
        f"expected 512 response rows, got rollout={rollout_rows} length={length_rows}"
    )

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

summary = {
    "schema_version": 1,
    "status": "PASS",
    "claim_boundary": "non-authoritative preemption-recovery functionality smoke",
    "root": str(root),
    "kv_tokens_per_rank": kv_tokens,
    "capture_sizes": [16, 32],
    "preemption_log_events": preemptions,
    "synchronized_eager_entry_dp_ranks": sorted(entered_ranks),
    "synchronized_preemption_eager_rounds": synchronized_preemption_rounds,
    "preemption_to_full_recovery_examples": recovery_pairs,
    "responses": rollout_rows,
    "aborted_ratio": 0.0,
    "primary_log": str(log),
    "primary_log_sha256": sha256(log),
    "rollout_sha256": sha256(rollouts[0]),
    "runner_sha256": sha256(runner),
}
summary_path = root / "PREEMPTION_RECOVERY_SMOKE_SUMMARY.json"
summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
(root / "PREEMPTION_RECOVERY_SMOKE_PASS.txt").write_text(
    "status=PASS\n"
    f"summary_sha256={sha256(summary_path)}\n"
    f"preemption_log_events={preemptions}\n"
    f"responses={rollout_rows}\n"
)
print(json.dumps(summary, indent=2, sort_keys=True))
PY
}

if [[ "$ACTION" == run ]]; then
    if [[ -e "$ROOT" ]]; then
        echo "smoke root already exists: $ROOT" >&2
        exit 2
    fi
    GRAPH_MAIN_ROOT="$ROOT" \
    GRAPH_MAIN_SMOKE_KV_TOKENS="$SMOKE_KV_TOKENS" \
        "$MAIN_RUNNER" smoke
fi

verify_smoke
echo "[preemption recovery smoke] PASS root=$ROOT"
