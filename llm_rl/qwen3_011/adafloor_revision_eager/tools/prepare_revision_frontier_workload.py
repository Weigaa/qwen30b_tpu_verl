#!/usr/bin/env python3
"""Prepare identical raw-token requests for real EP8 and EP16 controls."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from verl.utils.rollout_seeding import prompt_token_hash


def prepare(rows, pad_token_id, expected_prompts=32, n=8):
    requests = []
    by_prompt = defaultdict(set)
    identities = set()
    for row in rows:
        ids = list(row["prompts"])
        while ids and ids[0] == pad_token_id: ids.pop(0)
        if not ids or prompt_token_hash(ids) != row["rollout_prompt_hash"]:
            raise ValueError("raw prompt hash mismatch; do not substitute a new chat template")
        key = (row["rollout_prompt_hash"], int(row["rollout_sample_index"]))
        if key in identities: raise ValueError("duplicate request identity")
        identities.add(key); by_prompt[key[0]].add(key[1])
        requests.append({"prompt_token_ids": ids, "prompt_hash": key[0], "sample_index": key[1],
                         "request_seed": int(row["rollout_request_seed"]),
                         "original_ep16_owner": int(row["rollout_rank"]),
                         "input": row["input"], "ground_truth": row["gts"]})
    if len(by_prompt) != expected_prompts or any(indices != set(range(n)) for indices in by_prompt.values()):
        raise ValueError("expected exact prompt/sample identity set")
    if len(requests) % 16 or len(requests) % 8: raise ValueError("both launch worlds need equal request chunks")
    # Group by original rank to preserve the original EP16 local assignment.
    # The EP8 static arm receives consecutive pairs of those chunks: all real
    # requests run on8 ranks from the beginning, with no discarded ghost batch.
    requests.sort(key=lambda row: (row["original_ep16_owner"], row["prompt_hash"], row["sample_index"]))
    width = len(requests)//16
    if any(sum(row["original_ep16_owner"] == rank for row in requests) != width for rank in range(16)):
        raise ValueError("original EP16 ownership is not uniform")
    return {"schema_version": 1, "prompt_count": len(by_prompt), "request_count": len(requests),
            "responses_per_prompt": n, "pad_token_id": pad_token_id, "requests": requests,
            "ep16_requests_per_rank": len(requests)//16, "ep8_requests_per_rank": len(requests)//8}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--rollout",type=Path,required=True)
    parser.add_argument("--model-config",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists() or args.output.resolve() in (args.rollout.resolve(),args.model_config.resolve()):
        raise ValueError("refusing overwrite/source mutation")
    rows=[json.loads(line) for line in args.rollout.read_text().splitlines() if line.strip()]
    payload=prepare(rows,json.loads(args.model_config.read_text())["pad_token_id"])
    payload["input_sha256"]={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (args.rollout,args.model_config)}
    payload["code_sha256"]={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (Path(__file__),Path(__file__).parents[1]/"verl/utils/rollout_seeding.py")}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(payload,ensure_ascii=True,sort_keys=True,separators=(",",":"))+"\n")
    print(json.dumps({key:payload[key] for key in ("prompt_count","request_count","ep16_requests_per_rank","ep8_requests_per_rank")},sort_keys=True))


if __name__=="__main__":main()
