from vllm_ascend import envs as envs_ascend

_use_legacy_fused_moe = (
    envs_ascend.VLLM_ASCEND_ENABLE_ELASTIC_PARALLEL_SHRINK
    or envs_ascend.VLLM_ASCEND_USE_LEGACY_FUSED_MOE
)

if _use_legacy_fused_moe:
    from vllm_ascend.ops.fused_moe_legacy import (AscendFusedMoE,
                                                  AscendUnquantizedFusedMoEMethod)
    from vllm_ascend.ops.fused_moe.fused_moe import AscendSharedFusedMoE
else:
    from vllm_ascend.ops.fused_moe.fused_moe import (
        AscendFusedMoE, AscendSharedFusedMoE, AscendUnquantizedFusedMoEMethod)

__all__ = [
    "AscendFusedMoE",
    "AscendSharedFusedMoE",
    "AscendUnquantizedFusedMoEMethod",
]
