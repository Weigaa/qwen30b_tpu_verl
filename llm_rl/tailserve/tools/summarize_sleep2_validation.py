#!/usr/bin/env python3
"""Report evidence, without conflating process completion with model accuracy."""
import argparse
import ast
import hashlib
import json
import re
import statistics
from pathlib import Path


def summarize(root):
    results = []
    for epoch in sorted(root.glob("*/epoch_*")):
        if not epoch.is_dir():
            continue
        text = "\n".join(p.read_text(errors="replace") for p in sorted((epoch / "logs").glob("*.txt")))
        text = re.sub(r"\x1b\[[0-9;]*[mK]", "", text)
        rollout_times = re.findall(r"=> rollout_output_time_s:\s*([\d.]+)", text)
        rollout_rates = re.findall(r"=> rollouts speed tokens/s:\s*([\d.]+)", text)
        plan_file = epoch / "oracle/length_sorted_rank_plan.json"
        plans = {int(p["step"]): p for p in json.loads(plan_file.read_text())} if plan_file.exists() else {}
        audits = []
        for line in text.splitlines():
            if "MODE1_STEP_RUNTIME_AUDIT " in line:
                audits.append(json.JSONDecoder().raw_decode(
                    line.split("MODE1_STEP_RUNTIME_AUDIT ", 1)[1])[0])
        shrinks = re.findall(r"shrink rpc done: global_rank=(\d+) active_ranks=(\[[\d, ]+\])", text)
        targets = sorted({tuple(ast.literal_eval(ranks)) for _, ranks in shrinks}, key=lambda x: (-len(x), x))
        steps = []
        for p in sorted((epoch / "rollout_data").glob("*.jsonl"), key=lambda p: int(p.stem)):
            rows = [json.loads(line) for line in p.open() if line.strip()]
            length_file = epoch / "rollout_length" / f"length_{p.stem}.txt"
            lengths = [float(v) for v in length_file.read_text().split()] if length_file.exists() else []
            step_id = int(p.stem)
            plan = plans.get(step_id, {})
            cap = plan.get("tail_guard_response_cap")
            stats_file = epoch / "rollout_length" / f"stats_{p.stem}.json"
            stats = json.loads(stats_file.read_text()) if stats_file.exists() else {}
            step_audits = [a for a in audits if a["step"] == step_id]
            steps.append({"step": step_id, "rows": len(rows),
                          "reward_mean": statistics.mean(r["score"] for r in rows) if rows else None,
                          "reward_one_count": sum(r["score"] == 1 for r in rows),
                          "think_outputs": sum("<think>" in r["output"] for r in rows),
                          "length_mean": statistics.mean(lengths) if lengths else None,
                          "length_min": min(lengths) if lengths else None,
                          "length_max": max(lengths) if lengths else None,
                          "tailguard_cap": cap,
                          "at_tailguard": sum(v >= cap for v in lengths) if cap else None,
                          "planned_floor": plan.get("selected_floor"),
                          "runtime_audit_ranks": sorted(a["rank"] for a in step_audits),
                          "runtime_audit_complete": len(step_audits) == 16 and len({a["rank"] for a in step_audits}) == 16 and all(a["complete"] for a in step_audits),
                          "kv_capacities_actual": sorted({a["kv_tokens"] for a in step_audits}),
                          "preemptions_counted": sum(a["preemptions"] for a in step_audits) if step_audits else None,
                          "runtime_audits": step_audits,
                          "rollout_seconds": float(rollout_times[step_id - 1]) if len(rollout_times) >= step_id else None,
                          "logged_tokens_per_second": float(rollout_rates[step_id - 1]) if len(rollout_rates) >= step_id else None,
                          "reported_response_tokens_per_second": (sum(lengths) / float(rollout_times[step_id - 1])) if lengths and len(rollout_times) >= step_id else None,
                          "timing_raw": stats.get("timing_raw", {}),
                          "output_sha256": hashlib.sha256(p.read_bytes()).hexdigest()})
        exits = re.findall(r"\[run\] end_time=\S+ exit_code=(\d+)", text)
        results.append({"epoch": str(epoch), "steps": steps,
                        "completed_shrink_targets": targets,
                        "min_completed_floor": min(map(len, targets), default=16),
                        "full_dispatch_log_count": len(re.findall(r"rollout_forward_begin[^\n]+graph_mode=FULL", text)),
                        "replay_marker_count": text.count("Replaying aclgraph"),
                        "topology_capture_completions": [
                            {"rank": int(rank), "active_ranks": ast.literal_eval(ranks)}
                            for rank, ranks in re.findall(
                                r"Elastic ACLGraph topology capture finished: rank=(\d+) active_ranks=(\[[\d, ]+\])", text)],
                        "preemption_log_count": len(re.findall(r"preempting request|request preempted", text, re.I)),
                        "weight_audit_pass_records": len(re.findall(r"MOE_WEIGHT_VALUE_AUDIT[^\n]+'failed': 0", text)),
                        "exit_code": int(exits[-1]) if exits else None,
                        "errors": [l[:800] for l in text.splitlines() if re.search(
                            r"RuntimeError:|NPU out of memory|aclrtMallocPhysical failed|AssertionError:", l)]})
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    print(json.dumps(summarize(args.root), ensure_ascii=False, indent=2))
