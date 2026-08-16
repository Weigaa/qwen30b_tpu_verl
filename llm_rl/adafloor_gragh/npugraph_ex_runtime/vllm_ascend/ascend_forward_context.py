import math
import logging
import os
from collections import Counter
from contextlib import contextmanager
from enum import Enum
from typing import Any

import torch
from vllm.config import CUDAGraphMode, VllmConfig
from vllm.distributed import get_dp_group, get_ep_group, get_tensor_model_parallel_world_size
from vllm.forward_context import BatchDescriptor, get_forward_context, set_forward_context

import vllm_ascend.envs as envs_ascend
from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.utils import (
    AscendDeviceType,
    enable_sp,
    flashcomm2_enable,
    get_ascend_device_type,
    has_layer_idx,
    is_drafter_moe_model,
    is_moe_model,
    speculative_enable_dispatch_gmm_combine_decode,
)

logger = logging.getLogger(__name__)


def _use_old_eager_forward_context() -> bool:
    return os.getenv("VLLM_ASCEND_EAGER_OLD_FORWARD_CONTEXT", "0").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _use_old_qwen3_moe_stack() -> bool:
    return os.getenv("VLLM_QWEN3_MOE_ASCEND_LEGACY_STACK", "0").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


class MoECommType(Enum):
    ALLGATHER = 0
    MC2 = 1
    ALLTOALL = 2
    FUSED_MC2 = 3
    NAIVE_MULTICAST = 4


class FusedMoEState(Enum):
    AllGather = 0
    All2All = 1
    MC2 = 2
    AllGatherEP = 3
    NaiveMulticast = 4
    All2AllSeq = 5


def _to_legacy_fused_moe_state(moe_comm_type: MoECommType | None) -> FusedMoEState:
    if moe_comm_type == MoECommType.ALLGATHER:
        return FusedMoEState.AllGather
    if moe_comm_type == MoECommType.ALLTOALL:
        return FusedMoEState.All2All
    if moe_comm_type == MoECommType.MC2:
        return FusedMoEState.MC2
    if moe_comm_type == MoECommType.FUSED_MC2:
        return FusedMoEState.MC2
    if moe_comm_type == MoECommType.NAIVE_MULTICAST:
        return FusedMoEState.NaiveMulticast
    return FusedMoEState.AllGather


@contextmanager
def set_ascend_forward_context(
    attn_metadata: Any,
    vllm_config: VllmConfig,
    virtual_engine: int = 0,
    num_tokens: int = 0,
    num_tokens_across_dp: torch.Tensor | None = None,
    with_prefill: bool = True,
    in_profile_run: bool = False,
    num_actual_tokens: int | None = None,
    aclgraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE,
    batch_descriptor: BatchDescriptor | None = None,
    model_instance: torch.nn.Module = None,
    is_draft_model=False,
    prefetch_stream: torch.npu.Stream | None = None,
    moe_prefetch_stream: torch.npu.Stream | None = None,
    moe_comm_type: MoECommType | None = None,
    reserved_mc2_mask: torch.Tensor | None = None,
):
    """A context manager that stores the current forward context,
    can be attention metadata, etc.
    We add some additional param into forward_context.
    """
    with set_forward_context(
        attn_metadata,
        vllm_config,
        virtual_engine=virtual_engine,
        num_tokens=num_tokens,
        num_tokens_across_dp=num_tokens_across_dp,
        cudagraph_runtime_mode=aclgraph_runtime_mode,
        batch_descriptor=batch_descriptor,
    ):
        forward_context = get_forward_context()

        if (
            envs_ascend.VLLM_ASCEND_ENABLE_ELASTIC_PARALLEL_SHRINK
            or envs_ascend.VLLM_ASCEND_USE_LEGACY_FUSED_MOE
            or envs_ascend.VLLM_ASCEND_USE_COMMON_FUSED_MOE
            or _use_old_qwen3_moe_stack()
        ):
            from vllm_ascend.ops.moe.moe_comm_method import get_moe_comm_method
        else:
            from vllm_ascend.ops.fused_moe.moe_comm_method import get_moe_comm_method

        if moe_comm_type is None:
            moe_comm_type = select_moe_comm_method(
                num_tokens,
                vllm_config,
                is_draft_model,
                with_prefill=with_prefill,
            )
        forward_context.moe_comm_type = moe_comm_type
        forward_context.selected_moe_comm_type = moe_comm_type
        forward_context.moe_comm_method = get_moe_comm_method(moe_comm_type)
        forward_context.fused_moe_state = _to_legacy_fused_moe_state(moe_comm_type)
        forward_context.with_prefill = with_prefill

        tp_world_size = get_tensor_model_parallel_world_size()

        forward_context.in_profile_run = in_profile_run

        # NOTE: This cannot be set using set_forward_context
        # due to multiple warmups before actual capturing
        forward_context.capturing = False

        # TODO: remove it when torch_npu.npu_mm_reduce_scatter_base supports tp_size >= 16.
        mmrs_fusion = tp_world_size <= 8

        # set for sequence parallelism, 1000 is the batch size concurrency threshold
        # for enabling the flashcomm_v1 or sequence_parallelism feature.
        # Currently, it is an empirical value. In normal scenarios, if the concurrency
        # exceeds this threshold, the performance benefits can be maximized.
        # Conversely, if the concurrency is below the threshold,
        # the performance may degrade due to the switching of communication methods.

        # main model and drafter model may have different architecture
        is_context_moe_model = is_drafter_moe_model(vllm_config) if is_draft_model else is_moe_model(vllm_config)
        use_old_eager_forward_context = _use_old_eager_forward_context()
        if is_context_moe_model:
            if use_old_eager_forward_context:
                sp_enabled = (
                    enable_sp(vllm_config)
                    and tp_world_size > 1
                    and num_tokens is not None
                    and num_tokens > 1000
                )
            else:
                sp_enabled = enable_sp(vllm_config) and num_tokens is not None
            mmrs_fusion = False
        elif is_draft_model:
            # TODO: for dense drafter, `sp` is redundant and is not compatible with `dp` and `graph`.
            # Disable it to avoid more problems.
            sp_enabled = False
        else:
            sp_enabled = enable_sp(vllm_config) and num_tokens is not None and num_tokens > 1000

        forward_context.mmrs_fusion = mmrs_fusion
        forward_context.num_tokens = num_tokens
        forward_context.sp_enabled = sp_enabled
        # TODO(Levi-JQ): another PR to normalize the enabling logic for sp/fc2
        forward_context.flashcomm_v2_enabled = (
            not use_old_eager_forward_context
            and flashcomm2_enable()
            and tp_world_size > 1
            and num_tokens is not None
        )

        if forward_context.sp_enabled or forward_context.flashcomm_v2_enabled:
            pad_size = (tp_world_size - (num_tokens % tp_world_size)) % tp_world_size
            forward_context.pad_size = pad_size

        # set this for rope forward_oot using
        forward_context.is_first_layer = True

        # set layer_idx to enable optimization features that depend on this information.
        # This is only applicable to models that contain these necessary attributes.
        forward_context.layer_idx = None
        if has_layer_idx(model_instance):
            forward_context.layer_idx = model_instance.model.start_layer
        forward_context.prefetch_stream = prefetch_stream
        forward_context.moe_prefetch_stream = moe_prefetch_stream
        forward_context.moe_double_buffer_manager = None

        # TODO(rjg-lyh): refactor mlp weight prefetch method
        # set for mlp weight prefetch
        prefetch_mlp_enabled = (
            (not use_old_eager_forward_context or bool(int(os.getenv("VLLM_ASCEND_ENABLE_DENSE_OPTIMIZE", "0"))))
            and envs_ascend.VLLM_ASCEND_ENABLE_PREFETCH_MLP
            and forward_context.layer_idx is not None
            and num_tokens is not None
            and num_tokens < 500
        )
        if prefetch_mlp_enabled:
            forward_context.prefetch_mlp_gate_up_proj = False
            forward_context.prefetch_mlp_down_proj = False
        forward_context.prefetch_mlp_enabled = prefetch_mlp_enabled
        forward_context.model_instance = model_instance
        forward_context.is_draft_model = is_draft_model

        if num_tokens is None and attn_metadata is not None:
            num_tokens = attn_metadata.num_actual_tokens

        dp_world_size = get_dp_group().world_size
        if dp_world_size > 1 and forward_context.dp_metadata is not None:
            max_tokens_across_dp = forward_context.dp_metadata.max_tokens_across_dp_cpu.item()
            if forward_context.sp_enabled or forward_context.flashcomm_v2_enabled:
                padded_length = (max_tokens_across_dp + tp_world_size - 1) // tp_world_size * tp_world_size
                pad_size = padded_length - num_tokens
                forward_context.padded_length = padded_length
                forward_context.pad_size = pad_size
        else:
            max_tokens_across_dp = num_tokens

        forward_context.max_tokens_across_dp = max_tokens_across_dp

        if num_tokens is not None:
            if num_actual_tokens is None:
                num_actual_tokens = num_tokens
            # NOTE: token num which need to pad to when mc2
            forward_context.padded_num_tokens = math.ceil(max_tokens_across_dp / tp_world_size) * tp_world_size
            if reserved_mc2_mask is None:
                reserved_mc2_mask = get_mc2_mask()
            if reserved_mc2_mask is not None:
                mc2_mask = reserved_mc2_mask[: forward_context.padded_num_tokens]
                mc2_mask[:num_actual_tokens] = True
                mc2_mask[num_actual_tokens:] = False
                forward_context.mc2_mask = mc2_mask

        try:
            yield
        finally:
            pass


_mc2_tokens_capacity: int | None = None
_reserved_mc2_mask: torch.Tensor | None = None
_sin: torch.Tensor | None = None
_cos: torch.Tensor | None = None
_moe_comm_debug_total: int = 0
_moe_comm_debug_summary: Counter[tuple[str, str]] = Counter()


def _token_bucket(num_tokens: int) -> str:
    if num_tokens <= 0:
        return "0"
    if num_tokens <= 32:
        return "<=32"
    if num_tokens <= 64:
        return "33-64"
    if num_tokens <= 128:
        return "65-128"
    if num_tokens <= 256:
        return "129-256"
    if num_tokens <= 512:
        return "257-512"
    if num_tokens <= 1024:
        return "513-1024"
    return ">1024"


def _maybe_log_moe_comm_selection(num_tokens: int,
                                  mc2_tokens_capacity: int | None,
                                  moe_comm_type: MoECommType | None,
                                  with_prefill: bool | None = None) -> None:
    if os.getenv("VLLM_ASCEND_MOE_COMM_DEBUG", "0").lower() not in {
            "1", "true", "yes", "on"}:
        return

    global _moe_comm_debug_total
    _moe_comm_debug_total += 1
    comm_name = "None" if moe_comm_type is None else moe_comm_type.name
    bucket = _token_bucket(num_tokens)
    _moe_comm_debug_summary[(comm_name, bucket)] += 1

    interval = int(os.getenv("VLLM_ASCEND_MOE_COMM_DEBUG_INTERVAL", "512"))
    if _moe_comm_debug_total <= 32 or (
            interval > 0 and _moe_comm_debug_total % interval == 0):
        summary = ", ".join(
            f"{comm}/{bucket}:{count}"
            for (comm, bucket), count in sorted(
                _moe_comm_debug_summary.items()))
        logger.info(
            "MoE comm select pid=%s call=%d num_tokens=%s bucket=%s "
            "with_prefill=%s mc2_capacity=%s selected=%s summary={%s}",
            os.getpid(),
            _moe_comm_debug_total,
            num_tokens,
            bucket,
            with_prefill,
            mc2_tokens_capacity,
            comm_name,
            summary,
        )


def set_mc2_tokens_capacity(vllm_config, max_num_reqs, uniform_decode_query_len):
    global _mc2_tokens_capacity
    if _mc2_tokens_capacity is not None:
        return
    configured_capacity = envs_ascend.VLLM_ASCEND_MC2_TOKENS_CAPACITY
    if configured_capacity > 0:
        max_num_tokens = configured_capacity
    elif envs_ascend.VLLM_ASCEND_USE_LEGACY_FUSED_MOE:
        # The old eager rollout stack reserved 512 MC2 mask entries.  The new
        # formula can shrink this to max_num_reqs (32 here), which makes mixed
        # small-token batches fall back to AllToAll much earlier than before.
        max_num_tokens = 512
    elif vllm_config.compilation_config.cudagraph_capture_sizes:
        max_num_tokens = vllm_config.compilation_config.max_cudagraph_capture_size
    else:
        # NOTE: To save memory, we cap the max number of tokens to 512.
        max_num_tokens = min(max_num_reqs * uniform_decode_query_len, 512)
    tp_size = vllm_config.parallel_config.tensor_parallel_size
    # Use integer arithmetic for ceiling division.
    num_tokens_per_tp_rank = (max_num_tokens + tp_size - 1) // tp_size
    _mc2_tokens_capacity = num_tokens_per_tp_rank * tp_size


def get_mc2_tokens_capacity():
    return _mc2_tokens_capacity


def set_mc2_mask(vllm_config, device):
    global _reserved_mc2_mask
    if _reserved_mc2_mask is not None:
        return
    if is_moe_model(vllm_config):
        _reserved_mc2_mask = torch.zeros(get_mc2_tokens_capacity(), dtype=torch.bool, device=device)
    else:
        _reserved_mc2_mask = None


def get_mc2_mask():
    return _reserved_mc2_mask


def select_moe_comm_method(num_tokens: int,
                           vllm_config: VllmConfig,
                           is_draft_model=False,
                           with_prefill: bool = False) -> MoECommType | None:
    """Select the MoE communication method according to parallel settings,
    device generation, token count, and quantization.

    1. Non-MoE models return `None`.
    2. Without expert parallel, fall back to all-gather.
    3. On A2 with expert parallel, pick MC2 when tokens fit the MC2 capacity
       and the DP size is large enough; otherwise use all-gather.
    4. On A3 with expert parallel, prefer fused MC2 when using w8a8_dynamic
       quantization with small EP size, no dynamic_eplb, and not in MTP
       mode; otherwise use MC2 within capacity or all-to-all.

    Args:
        num_tokens (int): The number of tokens in the current batch.
        vllm_config (VllmConfig): Runtime configuration for the model.
        is_draft_model (bool): Whether the model runs in MTP mode (disables fused MC2).
        with_prefill (bool): Whether the current batch contains prefill tokens.

    Raises:
        ValueError: If the soc version is unsupported.

    Returns:
        MoECommType | None: The selected MoE communication method.
    """
    if not is_moe_model(vllm_config):
        return None
    mc2_tokens_capacity = get_mc2_tokens_capacity()
    soc_version = get_ascend_device_type()
    quant_type = getattr(
        vllm_config.model_config.hf_text_config,
        "moe_quantize",
        getattr(vllm_config.model_config.hf_text_config, "quantize", None),
    )

    force_alltoall_moe = envs_ascend.VLLM_ASCEND_FORCE_ALLTOALL_MOE
    if (
        force_alltoall_moe
        and vllm_config.parallel_config.enable_expert_parallel
        and get_ep_group().world_size > 1
    ):
        moe_comm_type = MoECommType.ALLTOALL
    elif not vllm_config.parallel_config.enable_expert_parallel or get_ep_group().world_size == 1:
        moe_comm_type = MoECommType.ALLGATHER
    elif soc_version in {AscendDeviceType.A2}:
        if (
            num_tokens <= mc2_tokens_capacity
            and vllm_config.parallel_config.world_size_across_dp / vllm_config.parallel_config.pipeline_parallel_size
            >= 16
        ):
            moe_comm_type = MoECommType.MC2
        else:
            moe_comm_type = MoECommType.ALLGATHER

    elif soc_version in {AscendDeviceType.A3}:
        dynamic_eplb = get_ascend_config().eplb_config.dynamic_eplb
        # TODO: drop the EP-size guard when dispatch_ffn_combine supports larger EP sizes
        # TODO: drop speculative method guard when dispatch_gmm_combine_decode supports w16a16
        fused_mc2_enable = envs_ascend.VLLM_ASCEND_ENABLE_FUSED_MC2 and quant_type == "w8a8_dynamic"
        dispatch_ffn_combine_enable = get_ep_group().world_size <= 32 and (not is_draft_model) and (not dynamic_eplb)
        if num_tokens <= mc2_tokens_capacity:
            fused_decode_enable = fused_mc2_enable
            if envs_ascend.VLLM_ASCEND_ENABLE_FUSED_MC2 == 1:
                fused_decode_enable = fused_mc2_enable and dispatch_ffn_combine_enable
            elif envs_ascend.VLLM_ASCEND_ENABLE_FUSED_MC2 == 2:
                fused_decode_enable = fused_mc2_enable and speculative_enable_dispatch_gmm_combine_decode(vllm_config)
            moe_comm_type = MoECommType.FUSED_MC2 if fused_decode_enable else MoECommType.MC2
        else:
            fused_prefill_enable = fused_mc2_enable
            if envs_ascend.VLLM_ASCEND_ENABLE_FUSED_MC2 == 1:
                fused_prefill_enable = fused_mc2_enable and dispatch_ffn_combine_enable
            elif envs_ascend.VLLM_ASCEND_ENABLE_FUSED_MC2 == 2:
                fused_prefill_enable = False
            moe_comm_type = MoECommType.FUSED_MC2 if fused_prefill_enable else MoECommType.ALLTOALL

    else:
        raise ValueError(f"Unsupported soc_version: {soc_version}")
    _maybe_log_moe_comm_selection(num_tokens, mc2_tokens_capacity,
                                  moe_comm_type, with_prefill)
    return moe_comm_type
