#!/usr/bin/env python3
"""Summarize observed rollout/actor evidence without inferring success from liveness."""
import argparse
import json
import re
from pathlib import Path


def summarize_log(path: Path) -> dict:
    text = path.read_text(errors="replace")
    text = re.sub(r"\x1b\[[0-9;]*[mK]", "", text)
    forwards = re.findall(
        r"rollout_forward_begin dp_rank=(\d+) iteration=(\d+).*?"
        r"global_prefill=(True|False) graph_mode=(\w+)", text)
    rank0 = [row for row in forwards if row[0] == "0"]
    exit_codes = re.findall(r"\[run\] end_time=\S+ exit_code=(\d+)", text)
    speeds = re.findall(r"rollouts speed tokens/s:\s*([\d.]+)", text)
    times = re.findall(r"rollout_output_time_s:\s*([\d.]+)", text)
    # Completion must come from the trainer, not graph capture or process launch.
    step_ids = sorted(set(map(int, re.findall(r"\bstep:(\d+)\s+-", text))))
    epoch = path.parent.parent
    rollout_files = sorted((epoch / "rollout_data").glob("*.jsonl"))
    lengths = []
    for length_file in sorted((epoch / "rollout_length").glob("length_*.txt")):
        values = [int(line) for line in length_file.read_text().splitlines()
                  if line.strip().isdigit()]
        lengths.append({"file": length_file.name, "responses": len(values),
                        "mean": sum(values) / len(values) if values else None,
                        "max": max(values) if values else None,
                        "at_16384": sum(v == 16384 for v in values)})
    return {
        "log": str(path),
        "exit_code": int(exit_codes[-1]) if exit_codes else None,
        "trainer_fit_returned": "After trainer.fit" in text,
        "completed_step_metrics": step_ids,
        "rollout_seconds": list(map(float, times)),
        "rollout_tokens_per_second": list(map(float, speeds)),
        "last_rank0_forward": ({"iteration": int(rank0[-1][1]),
                                "global_prefill": rank0[-1][2] == "True",
                                "graph_mode": rank0[-1][3]} if rank0 else None),
        "sampled_full_ranks": sorted({int(r[0]) for r in forwards if r[3] == "FULL"}),
        "sleep_freed_gib": list(map(float, re.findall(
            r"Sleep mode freed ([\d.]+) GiB", text))),
        "rollout_files": [str(p) for p in rollout_files],
        "length_files": lengths,
        "errors": [line[:800] for line in text.splitlines()
                   if re.search(r"(?:RuntimeError:|AssertionError:|NPU out of memory|"
                                r"ACLGraph weight storage changed)", line)],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    paths = sorted(p for p in args.root.rglob("logs/*.txt")
                   if p.parent.parent == args.root
                   or p.parent.parent.name.startswith("epoch_"))
    if not paths:
        parser.error(f"no epoch training logs under {args.root}")
    print(json.dumps([summarize_log(p) for p in paths], indent=2))


if __name__ == "__main__":
    main()
