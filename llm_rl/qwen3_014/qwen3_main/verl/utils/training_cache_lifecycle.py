"""Retire reconstructible training-only caches before co-located rollout."""
import gc
import sys

import torch


def clear_training_tensor_caches(models=()):
    """Clear only explicit deterministic caches, never model/optimizer state.

    Called after training has completed and before rollout weights/KV wake.
    Cached rotary tensors otherwise survive parameter offload, often pinning
    much larger partly-free allocator segments across the rollout boundary.
    """
    targets = (
        ("megatron.core.models.common.embeddings.rotary_pos_embedding", "RotaryEmbedding", "forward"),
        ("megatron.core.models.common.embeddings.yarn_rotary_pos_embedding", "YarnRotaryEmbedding", "forward"),
        ("megatron.core.transformer.utils", None, "get_default_causal_mask"),
    )
    device = torch.npu
    device.synchronize()
    free_before = device.mem_get_info()[0]
    allocated_before = device.memory_allocated()
    cleared = []
    # Megatron/MindSpeed dispatchers are plain Python objects, so their
    # last-microbatch activations are missed by parameter/buffer offload.
    transient_fields = (
        "routing_map", "probs", "local_map", "local_probs",
        "reversed_local_input_permutation_mapping",
        "reversed_global_input_permutation_mapping",
        "global_input_tokens_local_experts_indices",
        "num_global_tokens_per_local_expert", "tokens_per_expert",
    )
    seen = set()
    for model in models:
        for name, layer in model.named_modules():
            if (type(layer).__name__ == "TransformerBlock"
                    and type(layer).__module__.startswith("megatron.")
                    and isinstance(getattr(layer, "input_tensor", None), torch.Tensor)):
                value = layer.input_tensor
                cleared.append(dict(name=name + ".input_tensor",
                                    bytes=value.numel() * value.element_size()))
                layer.input_tensor = None
                del value
            dispatcher = getattr(layer, "token_dispatcher", None)
            for obj in (dispatcher, getattr(dispatcher, "disaptor", None)):
                if obj is None or id(obj) in seen:
                    continue
                seen.add(id(obj))
                if not type(obj).__module__.startswith(("megatron.", "mindspeed.")):
                    continue
                for field in transient_fields:
                    value = vars(obj).get(field)
                    if isinstance(value, torch.Tensor):
                        cleared.append(dict(name=name + ".token_dispatcher." + field,
                                            bytes=value.numel() * value.element_size()))
                        setattr(obj, field, None)
                    del value
    parallel_state = sys.modules.get("megatron.core.parallel_state")
    scratch = getattr(getattr(parallel_state, "_GLOBAL_MEMORY_BUFFER", None), "buffer", None)
    if isinstance(scratch, dict) and scratch:
        details = [{"key": str(key), "bytes": value.numel() * value.element_size()}
                   for key, value in scratch.items() if isinstance(value, torch.Tensor)]
        cleared.append(dict(name="megatron.GlobalMemoryBuffer", entries=len(scratch),
                            buffers=details))
        # This is uninitialized scratch returned by get_tensor(), not state.
        # Preserve the GlobalMemoryBuffer object; its next use reallocates.
        scratch.clear()
    for module_name, class_name, function_name in targets:
        module = sys.modules.get(module_name)
        if module is None:
            continue
        owner = getattr(module, class_name, None) if class_name else module
        function = getattr(owner, function_name, None)
        if not callable(getattr(function, "cache_clear", None)):
            continue
        info = function.cache_info()
        if info.currsize:
            function.cache_clear()
            cleared.append(dict(name=f"{module_name}.{class_name or ''}.{function_name}",
                                entries=info.currsize))
    gc.collect()
    device.empty_cache()
    device.synchronize()
    report = dict(caches=cleared, allocated_before=allocated_before,
                  allocated_after=device.memory_allocated(),
                  physical_free_before=free_before,
                  physical_free_after=device.mem_get_info()[0])
    print(f"TRAINING_CACHE_RELEASE {report}", flush=True)
    return report
