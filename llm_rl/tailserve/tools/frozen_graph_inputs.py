#!/usr/bin/env python3
"""Freeze historical input artifacts once; install/check without replanning."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

FILES = ("length_sorted_train.parquet", "length_sorted_rank_plan.json",
         "length_sorted_rank_plan_summary.json", "length_sorted_length_oracle.json")


def verify(root):
    manifest = json.loads((root / "manifest.json").read_text())
    for name, digest in manifest["files"].items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != digest:
            raise RuntimeError(f"Frozen input hash mismatch: {name}")
    return manifest


def create(source, root):
    if root.exists():
        raise RuntimeError(f"Refusing to overwrite frozen bundle: {root}")
    import pyarrow.parquet as pq
    rows = pq.read_table(source / FILES[0]).to_pylist()
    plans = json.loads((source / FILES[1]).read_text())
    if len(rows) != 160 or len(plans) != 5:
        raise ValueError("Expected160 prompts and5 frozen steps")
    for step, plan in enumerate(plans, 1):
        if plan["step"] != step or len(plan["aclgraph_capture_sizes"]) != 8:
            raise ValueError("Expected an ordered5-step K8 graph plan")
        for rank in range(16):
            positions = plan["rank_to_dataset_item_idx"][str(rank)]
            if [rows[int(i)]["extra_info"]["index"] for i in positions] != plan["rank_to_source_idx"][str(rank)]:
                raise ValueError("Frozen parquet does not match original prompt/rank plan")
    root.mkdir(parents=True)
    hashes = {}
    for name in FILES:
        shutil.copyfile(source / name, root / name)
        hashes[name] = hashlib.sha256((root / name).read_bytes()).hexdigest()
        (root / name).chmod(0o444)
    manifest = dict(files=hashes, source=str(source), steps=5, prompts=160,
                    batch_size=32, rollout_n=16, seed=0,
                    common_epoch0="/data/adafloor_shared_state/common_epoch0_probe_gpu09_kv380800_permanent",
                    workload="normal stochastic RL,actor updates enabled;not frozen outputs")
    manifest["bundle_id"] = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    (root / "manifest.json").chmod(0o444)
    return verify(root)


def install(root, destination):
    manifest = verify(root)
    destination.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        target = destination / name
        if target.exists():
            if hashlib.sha256(target.read_bytes()).hexdigest() != manifest["files"][name]:
                raise RuntimeError(f"Refusing different existing input: {target}")
        else:
            shutil.copyfile(root / name, target)
            target.chmod(0o444)
    print(json.dumps(manifest, sort_keys=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["create", "verify", "install"])
    parser.add_argument("root", type=Path)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--destination", type=Path)
    args = parser.parse_args()
    if args.action == "create": print(json.dumps(create(args.source, args.root), sort_keys=True))
    elif args.action == "install": install(args.root, args.destination)
    else: print(json.dumps(verify(args.root), sort_keys=True))


if __name__ == "__main__": main()
