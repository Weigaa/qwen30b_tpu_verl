"""Fail-closed audit of shared frozen input/plan and rank-local requests."""
import hashlib
import json
import os
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=4)
def load_bundle(directory):
    root = Path(directory)
    manifest = json.loads((root / "manifest.json").read_text())
    for name, digest in manifest["files"].items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != digest:
            raise RuntimeError(f"Frozen experiment input changed: {name}")
    plans = json.loads((root / "length_sorted_rank_plan.json").read_text())
    return manifest, {int(plan["step"]): plan for plan in plans}


def audit_frozen_rank_inputs(prompts, inputs, seeds, *, rank, actual_kv_tokens,
                             actual_capture_sizes, actual_response_cap, logger):
    directory = os.getenv("ADAFLOOR_FROZEN_INPUTS_DIR")
    if not directory:
        return
    manifest, plans = load_bundle(directory)
    step = int(prompts.meta_info["global_steps"])
    plan = plans[step]
    ids = prompts.non_tensor_batch.get("dataset_item_idx")
    samples = prompts.non_tensor_batch.get("rollout_sample_index")
    if ids is None or samples is None:
        raise RuntimeError("Frozen inputs require physical dataset IDs and sample indices")
    identities = [(int(i), int(s)) for i, s in zip(ids, samples, strict=True)]
    expected = [(int(i), s) for i in plan["rank_to_dataset_item_idx"][str(rank)]
                for s in range(manifest["rollout_n"])]
    if identities != expected:
        raise RuntimeError(f"Frozen prompt/rank/order mismatch: step={step} rank={rank} "
                           f"actual={identities} expected={expected}")
    if int(actual_kv_tokens) != int(plan["kv_cap"]):
        raise RuntimeError(f"Frozen KV mismatch: {actual_kv_tokens} != {plan['kv_cap']}")
    if list(actual_capture_sizes) != plan["aclgraph_capture_sizes"]:
        raise RuntimeError("Frozen graph bucket mismatch")
    if int(actual_response_cap) != int(plan["tail_guard_response_cap"]):
        raise RuntimeError("Frozen runtime response cap mismatch")
    if int(prompts.meta_info["shrink_aware_runtime"]["selected_floor"]) != int(plan["selected_floor"]):
        raise RuntimeError("Frozen runtime floor mismatch")
    from verl.utils.rollout_seeding import derive_request_seed
    expected_seeds = [derive_request_seed(manifest["seed"], x["prompt_token_ids"], s)[1]
                      for x, (_, s) in zip(inputs, identities, strict=True)]
    if seeds != expected_seeds:
        raise RuntimeError("Frozen per-request seed mismatch")
    requests = [dict(dataset_item_idx=i, sample_index=s,
                     prompt_hash=derive_request_seed(manifest["seed"], x["prompt_token_ids"], s)[0],
                     seed=seed)
                for (i, s), x, seed in zip(identities, inputs, seeds, strict=True)]
    logger.warning("FROZEN_GRAPH_INPUT_AUDIT %s", json.dumps(dict(
        step=step, rank=rank, bundle_id=manifest["bundle_id"],
        floor=plan["selected_floor"], kv_tokens=int(actual_kv_tokens),
        capture_sizes=list(actual_capture_sizes), tailguard=plan["tail_guard_response_cap"],
        requests=requests), sort_keys=True))
