#!/usr/bin/env python3
"""Verify BOTH arms' actual requests/resources before showing performance."""
import argparse
import json
import re
from pathlib import Path


def read_arm(root, steps):
    epochs = list(root.glob("*/epoch_001_mode1_natural"))
    if len(epochs) != 1: raise RuntimeError(f"Missing/ambiguous epoch: {root}")
    epoch = epochs[0]
    text = "\n".join(p.read_text(errors="replace") for p in (epoch / "logs").glob("*.txt"))
    text = re.sub(r"\x1b\[[0-9;]*[mK]", "", text)
    if not re.search(r"\[run\] end_time=\S+ exit_code=0", text):
        raise RuntimeError(f"Arm not completed successfully: {root}")
    audits = {}
    generation_audits = {}
    for line in text.splitlines():
        if "FROZEN_GRAPH_INPUT_AUDIT " in line:
            a = json.JSONDecoder().raw_decode(line.split("FROZEN_GRAPH_INPUT_AUDIT ", 1)[1])[0]
            key = (a["step"], a["rank"])
            if key in audits: raise RuntimeError(f"Duplicate input audit: {key}")
            audits[key] = a
        for marker in ("MODE1_STEP_RUNTIME_AUDIT ", "FROZEN_GRAPH_GENERATION_AUDIT "):
            if marker in line:
                a = json.JSONDecoder().raw_decode(line.split(marker,1)[1])[0]
                key = (a["step"],a["rank"])
                if key in generation_audits: raise RuntimeError("Duplicate generation audit")
                generation_audits[key] = a
    expected = {(s,r) for s in range(1,steps+1) for r in range(16)}
    if set(audits) != expected: raise RuntimeError(f"Incomplete frozen audit: {root}")
    if set(generation_audits) != expected or not all(x["complete"] for x in generation_audits.values()):
        raise RuntimeError(f"Incomplete generation/preemption audit: {root}")
    if any(x["preemptions"] for x in generation_audits.values()):
        raise RuntimeError(f"KV preemption observed in matched arm: {root}")
    if re.search(r"RuntimeError:|NPU out of memory|OutOfMemoryError|RayTaskError",text):
        raise RuntimeError(f"Runtime error in matched arm: {root}")
    times = list(map(float, re.findall(r"=> rollout_output_time_s:\s*([\d.]+)", text)))
    rates = list(map(float, re.findall(r"=> rollouts speed tokens/s:\s*([\d.]+)", text)))
    if len(times) != steps or len(rates) != steps: raise RuntimeError("Missing per-step metrics")
    output = []
    for i in range(1,steps+1):
        rows = [json.loads(x) for x in (epoch / f"rollout_data/{i}.jsonl").read_text().splitlines()]
        lengths = [float(x) for x in (epoch / f"rollout_length/length_{i}.txt").read_text().split()]
        if len(rows) != 512 or len(lengths) != 512: raise RuntimeError("Incomplete responses")
        output.append(dict(step=i,seconds=times[i-1],tokens_per_second=rates[i-1],
                           reward=sum(x["score"] for x in rows)/512,mean_length=sum(lengths)/512))
    return audits, output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root",type=Path)
    parser.add_argument("--expected-steps",type=int,default=5)
    args = parser.parse_args()
    a,x = read_arm(args.root / "v011",args.expected_steps)
    b,y = read_arm(args.root / "v014",args.expected_steps)
    require_optimized_v011_sampler(args.root / "v011")
    allocator_audit = require_v011_allocator_phases(args.root / "v011", args.expected_steps)
    if a != b: raise RuntimeError("Actual frozen request/rank/seed/KV/bucket audits DIFFER")
    result = dict(input_audit="PASS",compared_rank_steps=len(a),v011=x,v014=y,
                  allocator_audit=allocator_audit,
                  claim_boundary="Same frozen inputs and request seeds; stochastic outputs/actor updates may diverge")
    (args.root / "comparison.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))


def require_optimized_v011_sampler(root):
    text="\n".join(p.read_text(errors="replace") for p in root.glob("*/epoch_001_mode1_natural/logs/*.txt"))
    records=[]
    for line in text.splitlines():
        if "FROZEN_SAMPLER_OPERATOR_AUDIT " in line:
            records.append(json.JSONDecoder().raw_decode(line.split("FROZEN_SAMPLER_OPERATOR_AUDIT ",1)[1])[0])
    if len(records)!=16 or any(r.get("sampler")!="AscendSampler"
            or r.get("filter")!="torch_npu.npu_top_k_top_p" for r in records):
        raise RuntimeError("v0.11 is missing16 actual fused Ascend sampler operator audits")
    return records


def require_v011_allocator_phases(root, steps):
    counts = {(rank, phase): 0 for rank in range(16) for phase in ("rollout", "trainer")}
    for path in root.glob("*/epoch_001_mode1_natural/logs/*.txt"):
        for line in path.read_text(errors="replace").splitlines():
            marker = "NPU_ALLOCATOR_PHASE_AUDIT "
            if marker not in line:
                continue
            record = json.JSONDecoder().raw_decode(line.split(marker, 1)[1])[0]
            key = (record["rank"], record["phase"])
            if (key not in counts or record.get("applied") is not True
                    or record.get("backend") != "native"
                    or record.get("expandable_segments") is not (key[1] == "trainer")):
                raise RuntimeError("Invalid NPU allocator phase audit")
            counts[key] += 1
    if any(count < steps for count in counts.values()):
        raise RuntimeError("Missing per-rank rollout/trainer NPU allocator phase audits")
    return dict(v011_phase_setter="PASS", rollout_expandable_segments=False,
                trainer_expandable_segments=True,
                initialization_difference="v011=True,v014=False; existing segments are not converted",
                v014_evidence="unchanged NPU setter and call sites; standalone real-allocation probe")


if __name__=="__main__": main()
