"""Revision diagnostics: current-load expert fingerprints, not full byte hashes.

Gather expected fingerprints ONCE per full-world weight load; checks on active
survivors require no collective. Scope is sampled values of every mapped expert
in every detected MoE layer, TP1 only; this does not establish unsampled equality.
"""
import hashlib
import json
import logging
import os
import struct
import time

import torch
import torch.distributed as dist

LOG = logging.getLogger(__name__)


def enabled():
    return os.getenv("ADAFLOOR_REVISION_EXPERT_AUDIT", "0") == "1"


def _modules(model):
    return [(name, m) for name, m in model.named_modules()
            if hasattr(m, "expert_map") and hasattr(m, "w13_weight") and hasattr(m, "loaded_expert_map")]


def fingerprint(tensor, slot, expected_shape=None):
    row = tensor[int(slot)].detach()
    if expected_shape is not None and tuple(row.shape) != tuple(expected_shape):
        if row.ndim == 2 and tuple(row.transpose(0,1).shape) == tuple(expected_shape):
            row = row.transpose(0,1)
        else:
            raise RuntimeError(f"expert logical shape differs: {tuple(row.shape)} != {expected_shape}")
    flat = row.reshape(-1)
    count = min(int(os.getenv("ADAFLOOR_REVISION_EXPERT_AUDIT_VALUES", "16")), flat.numel())
    if count < 1:
        raise RuntimeError("empty expert row")
    offsets = sorted({round(i*(flat.numel()-1)/max(count-1,1)) for i in range(count)})
    vals = flat[torch.tensor(offsets,device=flat.device)].float().cpu().tolist()
    return {"shape": list(row.shape), "sample_count": len(vals),
            "sha256": hashlib.sha256(b''.join(struct.pack('<f',v) for v in vals)).hexdigest()}


def capture_loaded(model):
    if not enabled():
        return
    from vllm.distributed.parallel_state import get_world_group
    start = time.perf_counter()
    version = int(getattr(model, "_revision_weight_version", 0)) + 1
    local = {}
    modules = _modules(model)
    if not modules:
        raise RuntimeError("expert audit found no canonical MoE maps")
    for name, m in modules:
        mapping = m.loaded_expert_map.detach().cpu().tolist()
        for expert, slot in enumerate(mapping):
            if int(slot) >= 0:
                key = f"{name}:{expert}"
                local[key] = {"w13": fingerprint(m.w13_weight,slot), "w2": fingerprint(m.w2_weight,slot)}
    group = get_world_group().cpu_group
    gathered = [None]*dist.get_world_size(group=group)
    dist.all_gather_object(gathered, {"version":version,"rows":local}, group=group)
    expected = {}
    for record in gathered:
        if record["version"] != version:
            raise RuntimeError("weight-load versions disagree across ranks")
        for key, value in record["rows"].items():
            if key in expected and expected[key] != value:
                raise RuntimeError(f"canonical fingerprints disagree: {key}")
            expected[key] = value
    for name, m in modules:
        for expert in range(len(m.expert_map)):
            if f"{name}:{expert}" not in expected:
                raise RuntimeError(f"missing current canonical expert: {name}:{expert}")
    model._revision_weight_version = version
    model._revision_expected_expert_fingerprints = expected
    model._revision_serving_weights_ready = True
    LOG.warning("Revision expert load audit: rank=%s version=%s layers=%s global_experts=%s sample_values=%s elapsed_ms=%.3f",
                dist.get_rank(), version,len(modules),len(expected),os.getenv("ADAFLOOR_REVISION_EXPERT_AUDIT_VALUES","16"),
                1000*(time.perf_counter()-start))


def check_active(model, tag):
    if not enabled():
        return
    expected = getattr(model,"_revision_expected_expert_fingerprints",None)
    version = getattr(model,"_revision_weight_version",None)
    if expected is None or version is None:
        raise RuntimeError("no current-load golden expert fingerprints")
    if not getattr(model, "_revision_serving_weights_ready", False):
        raise RuntimeError("metadata-only restored model must reload current actor weights before serving")
    start = time.perf_counter()
    checked = 0
    for name, m in _modules(model):
        for expert, slot in enumerate(m.expert_map.detach().cpu().tolist()):
            if int(slot) < 0:
                continue
            golden = expected[f"{name}:{expert}"]
            for attr, key in (("w13_weight","w13"),("w2_weight","w2")):
                actual = fingerprint(getattr(m,attr),slot,golden[key]["shape"])
                if actual != golden[key]:
                    raise RuntimeError(f"stale/corrupt active expert fingerprint: tag={tag} version={version} {name}:{expert}:{key}")
            checked += 1
    LOG.warning("Revision expert active audit: rank=%s tag=%s version=%s checked_experts=%s mismatches=0 scope=sampled_values elapsed_ms=%.3f",
                dist.get_rank(),tag,version,checked,1000*(time.perf_counter()-start))


def mark_restored_nonserving(model):
    if not enabled():
        return
    # Existing mode1 restore intentionally restores the full-world logical map
    # without copying CPU-stashed prefix rows back; that copy can corrupt
    # formatted tensors. It releases transient views before actor training.
    # Do not compare unused restored rows as if they were serving weights.
    model._revision_serving_weights_ready = False
    LOG.warning("Revision expert restore boundary: rank=%s version=%s state=nonserving_requires_actor_reload content_check=not_applicable",
                dist.get_rank(),getattr(model,"_revision_weight_version",None))
