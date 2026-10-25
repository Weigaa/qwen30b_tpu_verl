"""Opt-in source-to-execution audit for unquantized Qwen MoE reloads."""
import re

import torch


class MoEWeightValueAudit:
    def __init__(self, model):
        self.experts = model.model.layers[0].mlp.experts
        self.expected = {}

    def capture(self, weights):
        for name, value in weights:
            match = re.fullmatch(
                r"model\.layers\.0\.mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight",
                name)
            if match:
                expert_id = int(match[1])
                if self.experts._map_global_expert_id_to_local_expert_id(expert_id) >= 0:
                    self.expected[(expert_id, match[2])] = value.detach().cpu().clone()
            yield name, value

    def install_loader_probe(self):
        self._loaders = []
        for parameter in (self.experts.w13_weight, self.experts.w2_weight):
            original = parameter.weight_loader
            self._loaders.append((parameter, original))
            def probe(param, loaded_weight, weight_name, *, shard_id,
                      expert_id, return_success=False, _original=original):
                slot = self.experts._map_global_expert_id_to_local_expert_id(expert_id)
                result = _original(param, loaded_weight, weight_name,
                                   shard_id=shard_id, expert_id=expert_id,
                                   return_success=return_success)
                if slot >= 0:
                    data = param[slot]
                    if shard_id != "w2":
                        size = data.shape[0] // 2
                        data = data.narrow(0, size if shard_id == "w3" else 0, size)
                    print("MOE_LOADER_WRITE", {
                        "rank": self.experts.ep_rank, "expert": expert_id,
                        "slot": slot, "shard": shard_id,
                        "loader_owner_is_experts": getattr(_original, "__self__", None) is self.experts,
                        "parameter_ptr": param.data_ptr(), "view_ptr": data.data_ptr(),
                        "source_shape": tuple(loaded_weight.shape), "view_shape": tuple(data.shape),
                        "source_head": loaded_weight.detach().reshape(-1)[:8].cpu().float().tolist(),
                        "target_head": data.detach().reshape(-1)[:8].cpu().float().tolist(),
                    }, flush=True)
                return result
            parameter.weight_loader = probe

    def restore_loader_probe(self):
        for parameter, original in self._loaders:
            parameter.weight_loader = original
        self._loaders = []

    def check(self, stage):
        if not self.expected:
            raise RuntimeError("MoE value audit captured no local layer-0 weights")
        failures = []
        for (expert_id, projection), expected in self.expected.items():
            slot = self.experts._map_global_expert_id_to_local_expert_id(expert_id)
            parameter = (self.experts.w2_weight if projection == "down_proj"
                         else self.experts.w13_weight)
            actual = parameter[slot].detach().cpu()
            if projection == "down_proj":
                if actual.shape != expected.shape:
                    actual = actual.t()
            else:
                width, hidden = expected.shape
                if actual.shape == (hidden, 2 * width):
                    actual = actual.t()
                offset = width if projection == "up_proj" else 0
                actual = actual[offset:offset + width]
            if actual.shape != expected.shape or not torch.equal(actual, expected):
                failures.append({
                    "expert": expert_id, "slot": slot, "projection": projection,
                    "actual_shape": tuple(actual.shape), "expected_shape": tuple(expected.shape),
                    "max_abs": (float((actual.float() - expected.float()).abs().max())
                                if actual.shape == expected.shape else None),
                    "actual_head": actual.reshape(-1)[:8].float().tolist(),
                    "expected_head": expected.reshape(-1)[:8].float().tolist(),
                    "head_matches": [str(key) for key, value in self.expected.items()
                                     if torch.equal(actual.reshape(-1)[:8], value.reshape(-1)[:8])],
                })
        report = {"stage": stage, "ep_rank": self.experts.ep_rank,
                  "checked": len(self.expected), "failed": len(failures),
                  "details": failures[:3]}
        print(f"MOE_WEIGHT_VALUE_AUDIT {report}", flush=True)
        if failures:
            raise RuntimeError(f"MoE weight value audit failed: {report}")
        return report
