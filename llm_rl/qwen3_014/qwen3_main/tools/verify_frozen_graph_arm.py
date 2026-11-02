#!/usr/bin/env python3
"""Validate retained runtime seed evidence without rewriting rollout outputs."""
import argparse
import hashlib
import json
import struct
from pathlib import Path

from compare_frozen_graph_pair import read_arm
from frozen_graph_inputs import verify


def validate(root, bundle, steps):
    manifest = verify(bundle)
    plans = {int(p["step"]):p for p in json.loads((bundle/"length_sorted_rank_plan.json").read_text())}
    audits, metrics = read_arm(root,steps)
    hashes = {}
    for (step,rank), a in audits.items():
        p=plans[step]
        if (a["bundle_id"]!=manifest["bundle_id"] or a["floor"]!=p["selected_floor"]
                or a["kv_tokens"]!=int(p["kv_cap"]) or a["capture_sizes"]!=p["aclgraph_capture_sizes"]
                or a["tailguard"]!=p["tail_guard_response_cap"]):
            raise RuntimeError("Runtime plan/resource evidence differs from frozen bundle")
        expected=[(int(i),s) for i in p["rank_to_dataset_item_idx"][str(rank)] for s in range(16)]
        actual=[(r["dataset_item_idx"],r["sample_index"]) for r in a["requests"]]
        if actual!=expected:raise RuntimeError("Actual frozen rank/request order differs")
        for r in a["requests"]:
            h=r["prompt_hash"]
            if hashes.setdefault(r["dataset_item_idx"],h)!=h:
                raise RuntimeError("Different prompt hashes for the same frozen dataset row")
            digest=hashlib.blake2b(digest_size=8,person=b"adafloor.seed")
            digest.update(struct.pack("<Q",manifest["seed"] & ((1<<64)-1)))
            digest.update(bytes.fromhex(h))
            digest.update(struct.pack("<Q",r["sample_index"]))
            if r["seed"]!=int.from_bytes(digest.digest(),"little")%((1<<31)-1):
                raise RuntimeError("Runtime request seed derivation mismatch")
    return dict(status="PASS",bundle_id=manifest["bundle_id"],rank_step_audits=len(audits),
                verified_requests=sum(len(a["requests"]) for a in audits.values()),
                evidence="generation-time frozen input audits + complete zero-preemption generation audits",
                original_rollout_json_unchanged=True,metrics=metrics)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root",type=Path)
    parser.add_argument("bundle",type=Path)
    parser.add_argument("--steps",type=int,default=5)
    args=parser.parse_args()
    result=validate(args.root,args.bundle,args.steps)
    (args.root/"FROZEN_RUNTIME_AUDIT_PASS.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))


if __name__=="__main__":main()
