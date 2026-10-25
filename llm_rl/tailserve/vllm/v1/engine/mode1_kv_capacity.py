# SPDX-License-Identifier: Apache-2.0
"""Pure step-capacity planning shared by native-sleep RL and host tests."""
from copy import deepcopy


def resize_uniform_kv_configs(configs, target_tokens: int):
    """Resize Qwen full-attention configs exactly, without profiling/clamping.

    Same token-cap/block rounding as the v0.11 mode1 path. Hybrid attention
    and context-parallel layouts need their own sizing rules: fail closed
    rather than silently assigning an incorrect scheduler capacity.
    """
    if not configs or target_tokens <= 0:
        raise ValueError("A positive step KV token cap and existing configs are required")
    result = deepcopy(configs)
    for config in result:
        if len(config.kv_cache_groups) != 1 or config.num_blocks <= 0:
            raise ValueError("Native step resize currently supports one uniform KV group")
        block_size = config.kv_cache_groups[0].kv_cache_spec.block_size
        blocks = target_tokens // block_size
        if blocks <= 0:
            raise ValueError("KV target is smaller than one block")
        for tensor in config.kv_cache_tensors:
            if tensor.size % config.num_blocks:
                raise ValueError("KV tensor size is not an integral number of pages")
            tensor.size = tensor.size // config.num_blocks * blocks
        config.num_blocks = blocks
    return result, result[0].num_blocks * result[0].kv_cache_groups[0].kv_cache_spec.block_size
