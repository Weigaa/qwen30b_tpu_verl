from vllm_ascend import envs as envs_ascend

FUSED_MOE_BACKEND = envs_ascend.resolve_fused_moe_backend()
_use_legacy_fused_moe = FUSED_MOE_BACKEND == "legacy"

if _use_legacy_fused_moe:
    from vllm_ascend.ops.fused_moe_legacy import (AscendFusedMoE,
                                                  AscendUnquantizedFusedMoEMethod)
    from vllm_ascend.ops.fused_moe.fused_moe import AscendSharedFusedMoE
else:
    from vllm_ascend.ops.fused_moe.fused_moe import (
        AscendFusedMoE, AscendSharedFusedMoE, AscendUnquantizedFusedMoEMethod)

__all__ = [
    "FUSED_MOE_BACKEND",
    "AscendFusedMoE",
    "AscendSharedFusedMoE",
    "AscendUnquantizedFusedMoEMethod",
]
