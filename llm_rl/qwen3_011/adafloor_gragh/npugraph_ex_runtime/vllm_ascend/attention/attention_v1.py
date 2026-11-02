#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#

import logging
import os
import time
from collections import Counter
from dataclasses import dataclass
from enum import Enum
from typing import ClassVar

import torch
import torch_npu
import vllm.envs as envs_vllm
from vllm.config import VllmConfig, get_current_vllm_config
from vllm.forward_context import ForwardContext, get_forward_context
from vllm.utils.math_utils import cdiv
from vllm.utils.torch_utils import direct_register_custom_op
from vllm.v1.attention.backend import (  # type: ignore
    AttentionBackend,
    AttentionCGSupport,
    AttentionImpl,
    AttentionLayer,
    AttentionMetadataBuilder,
    AttentionType,
)
from vllm.v1.attention.backends.registry import (  # type: ignore
    AttentionBackendEnum,
    register_backend,
)
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_cache_interface import AttentionSpec, CrossAttentionSpec

import vllm_ascend.envs as envs_ascend
from vllm_ascend.attention.attention_mask import AttentionMaskBuilder
from vllm_ascend.attention.context_parallel.common_cp import AscendMetadataForDecode, AscendMetadataForPrefill
from vllm_ascend.attention.utils import (
    AscendCommonAttentionMetadata,
    enable_cp,
    split_decodes_and_prefills,
    using_paged_attention,
)
from vllm_ascend.compilation.acl_graph import (
    get_draft_graph_params,
    get_graph_params,
    update_draft_graph_params_workspaces,
    update_graph_params_workspaces,
)
from vllm_ascend.device.device_op import DeviceOperator
from vllm_ascend.ops.flashcomm2_oshard_manager import flashcomm2_oshard_manager
from vllm_ascend.utils import weak_ref_tensors

# default max value of sliding window size
SWA_INT_MAX = 2147483647
logger = logging.getLogger(__name__)
_LEGACY_ATTENTION_LOGGED = False
_ATTENTION_DEBUG_TOTAL = 0
_ATTENTION_DEBUG_SUMMARY: Counter[tuple[str, str, str]] = Counter()


def _env_flag(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        logger.warning("Invalid integer env %s=%r, using %s", name, value, default)
        return default


_ATTENTION_DEBUG_ENABLED = _env_flag("VLLM_ASCEND_ATTENTION_DEBUG")
_ATTENTION_DEBUG_INTERVAL = int(os.getenv("VLLM_ASCEND_ATTENTION_DEBUG_INTERVAL", "512"))
_ATTENTION_STAGE_DEBUG_ENABLED = _env_flag("VLLM_ASCEND_ATTENTION_STAGE_DEBUG")
_ATTENTION_STAGE_DEBUG_INTERVAL = _env_int("VLLM_ASCEND_ATTENTION_STAGE_DEBUG_INTERVAL", 1024)
_ATTENTION_STAGE_DEBUG_SKIP = _env_int("VLLM_ASCEND_ATTENTION_STAGE_DEBUG_SKIP", 0)
_ATTENTION_STAGE_DEBUG_LIMIT = _env_int("VLLM_ASCEND_ATTENTION_STAGE_DEBUG_LIMIT", 0)
_ATTENTION_WRAPPER_DEBUG_ENABLED = _env_flag("VLLM_ASCEND_ATTENTION_WRAPPER_DEBUG")
_ATTENTION_WRAPPER_DEBUG_INTERVAL = _env_int("VLLM_ASCEND_ATTENTION_WRAPPER_DEBUG_INTERVAL", 1024)
_ATTENTION_WRAPPER_DEBUG_SKIP = _env_int("VLLM_ASCEND_ATTENTION_WRAPPER_DEBUG_SKIP", 0)
_ATTENTION_WRAPPER_DEBUG_LIMIT = _env_int("VLLM_ASCEND_ATTENTION_WRAPPER_DEBUG_LIMIT", 0)
_ATTENTION_SYNC_DEBUG_ENABLED = _env_flag("VLLM_ASCEND_ATTENTION_SYNC_DEBUG")
_ATTENTION_SYNC_DEBUG_INTERVAL = _env_int("VLLM_ASCEND_ATTENTION_SYNC_DEBUG_INTERVAL", 1024)
_ATTENTION_SYNC_DEBUG_SKIP = _env_int("VLLM_ASCEND_ATTENTION_SYNC_DEBUG_SKIP", 0)
_ATTENTION_SYNC_DEBUG_LIMIT = _env_int("VLLM_ASCEND_ATTENTION_SYNC_DEBUG_LIMIT", 0)
_ATTENTION_FIA_DETAIL_DEBUG_ENABLED = _env_flag("VLLM_ASCEND_ATTENTION_FIA_DETAIL_DEBUG")
_ATTENTION_FIA_DETAIL_DEBUG_INTERVAL = _env_int("VLLM_ASCEND_ATTENTION_FIA_DETAIL_DEBUG_INTERVAL", 1024)
_ATTENTION_FIA_DETAIL_DEBUG_SKIP = _env_int("VLLM_ASCEND_ATTENTION_FIA_DETAIL_DEBUG_SKIP", 0)
_ATTENTION_FIA_DETAIL_DEBUG_LIMIT = _env_int("VLLM_ASCEND_ATTENTION_FIA_DETAIL_DEBUG_LIMIT", 0)
_ATTENTION_VALUE_CONTIGUOUS = _env_flag("VLLM_ASCEND_ATTENTION_VALUE_CONTIGUOUS")
_ATTENTION_STAGE_DEBUG_STATE_FILTER = {
    value.strip()
    for value in os.getenv("VLLM_ASCEND_ATTENTION_STAGE_DEBUG_STATE_FILTER", "").split(",")
    if value.strip()
}
_ATTENTION_STAGE_DEBUG_OP_FILTER = {
    value.strip()
    for value in os.getenv("VLLM_ASCEND_ATTENTION_STAGE_DEBUG_OP_FILTER", "").split(",")
    if value.strip()
}
_FORCE_PAGED_ATTENTION_DECODE = _env_flag("VLLM_ASCEND_FORCE_PAGED_ATTENTION_DECODE")
_DISABLE_FULL_GRAPH_FIA = _env_flag("VLLM_ASCEND_DISABLE_FULL_GRAPH_FIA")
_DISABLE_FULL_GRAPH_PA = _env_flag("VLLM_ASCEND_DISABLE_FULL_GRAPH_PA")
_attention_stage_debug_total = 0
_attention_stage_debug_seen = 0
_attention_stage_debug_matched = 0
_attention_stage_debug_summary: Counter[tuple[str, str, str]] = Counter()
_attention_wrapper_debug_matched = 0
_attention_sync_debug_matched = 0
_attention_fia_detail_debug_matched = 0


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


def _maybe_log_attention_path(
    attn_state: "AscendAttentionState",
    op_name: str,
    num_tokens: int,
    num_decodes: int,
    num_prefills: int,
) -> None:
    if not _ATTENTION_DEBUG_ENABLED:
        return
    global _ATTENTION_DEBUG_TOTAL
    _ATTENTION_DEBUG_TOTAL += 1
    bucket = _token_bucket(num_tokens)
    key = (attn_state.name, op_name, bucket)
    _ATTENTION_DEBUG_SUMMARY[key] += 1

    if _ATTENTION_DEBUG_TOTAL <= 32 or (
        _ATTENTION_DEBUG_INTERVAL > 0 and _ATTENTION_DEBUG_TOTAL % _ATTENTION_DEBUG_INTERVAL == 0
    ):
        summary = ", ".join(
            f"{state}/{op}/{tok}:{count}"
            for (state, op, tok), count in _ATTENTION_DEBUG_SUMMARY.most_common(12)
        )
        logger.info(
            "Attention path select #%s state=%s op=%s tokens=%s bucket=%s "
            "decodes=%s prefills=%s summary={%s}",
            _ATTENTION_DEBUG_TOTAL,
            attn_state.name,
            op_name,
            num_tokens,
            bucket,
            num_decodes,
            num_prefills,
            summary,
        )


def _maybe_new_timing_event() -> torch.npu.Event | None:
    try:
        return torch.npu.Event(enable_timing=True)
    except TypeError:
        return torch.npu.Event()


def _should_capture_attention_stage_timing(
    attn_state=None,
    op_name: str | None = None,
) -> bool:
    if not _ATTENTION_STAGE_DEBUG_ENABLED:
        return False
    state_name = getattr(attn_state, "name", "")
    if _ATTENTION_STAGE_DEBUG_STATE_FILTER and state_name not in _ATTENTION_STAGE_DEBUG_STATE_FILTER:
        return False
    if _ATTENTION_STAGE_DEBUG_OP_FILTER and op_name not in _ATTENTION_STAGE_DEBUG_OP_FILTER:
        return False

    global _attention_stage_debug_seen, _attention_stage_debug_matched
    _attention_stage_debug_seen += 1
    _attention_stage_debug_matched += 1
    if _ATTENTION_STAGE_DEBUG_LIMIT:
        if _attention_stage_debug_matched > _ATTENTION_STAGE_DEBUG_SKIP + _ATTENTION_STAGE_DEBUG_LIMIT:
            return False
    if _attention_stage_debug_matched <= _ATTENTION_STAGE_DEBUG_SKIP:
        return False
    matched_after_skip = _attention_stage_debug_matched - _ATTENTION_STAGE_DEBUG_SKIP
    return matched_after_skip <= 32 or (
        _ATTENTION_STAGE_DEBUG_INTERVAL > 0
        and matched_after_skip % _ATTENTION_STAGE_DEBUG_INTERVAL == 0
    )


if _ATTENTION_STAGE_DEBUG_ENABLED:
    # Keep the profiling gate out of Dynamo graphs. Otherwise compiled-eager can
    # treat an early True result as static and record/log every decode call.
    _should_capture_attention_stage_timing = torch._dynamo.disable(
        _should_capture_attention_stage_timing)


def _should_capture_attention_wrapper_timing(attn_state=None) -> bool:
    if not _ATTENTION_WRAPPER_DEBUG_ENABLED:
        return False
    state_name = getattr(attn_state, "name", "")
    if _ATTENTION_STAGE_DEBUG_STATE_FILTER and state_name not in _ATTENTION_STAGE_DEBUG_STATE_FILTER:
        return False

    global _attention_wrapper_debug_matched
    _attention_wrapper_debug_matched += 1
    if _ATTENTION_WRAPPER_DEBUG_LIMIT:
        if _attention_wrapper_debug_matched > _ATTENTION_WRAPPER_DEBUG_SKIP + _ATTENTION_WRAPPER_DEBUG_LIMIT:
            return False
    if _attention_wrapper_debug_matched <= _ATTENTION_WRAPPER_DEBUG_SKIP:
        return False
    matched_after_skip = _attention_wrapper_debug_matched - _ATTENTION_WRAPPER_DEBUG_SKIP
    return matched_after_skip <= 32 or (
        _ATTENTION_WRAPPER_DEBUG_INTERVAL > 0
        and matched_after_skip % _ATTENTION_WRAPPER_DEBUG_INTERVAL == 0
    )


if _ATTENTION_WRAPPER_DEBUG_ENABLED:
    _should_capture_attention_wrapper_timing = torch._dynamo.disable(
        _should_capture_attention_wrapper_timing)


def _should_capture_attention_sync_timing(attn_state=None) -> bool:
    if not _ATTENTION_SYNC_DEBUG_ENABLED:
        return False
    state_name = getattr(attn_state, "name", "")
    if _ATTENTION_STAGE_DEBUG_STATE_FILTER and state_name not in _ATTENTION_STAGE_DEBUG_STATE_FILTER:
        return False

    global _attention_sync_debug_matched
    _attention_sync_debug_matched += 1
    if _ATTENTION_SYNC_DEBUG_LIMIT:
        if _attention_sync_debug_matched > _ATTENTION_SYNC_DEBUG_SKIP + _ATTENTION_SYNC_DEBUG_LIMIT:
            return False
    if _attention_sync_debug_matched <= _ATTENTION_SYNC_DEBUG_SKIP:
        return False
    matched_after_skip = _attention_sync_debug_matched - _ATTENTION_SYNC_DEBUG_SKIP
    return matched_after_skip <= 32 or (
        _ATTENTION_SYNC_DEBUG_INTERVAL > 0
        and matched_after_skip % _ATTENTION_SYNC_DEBUG_INTERVAL == 0
    )


if _ATTENTION_SYNC_DEBUG_ENABLED:
    # This intentionally creates a profiling-only synchronization boundary.
    # Keep it out of compiled graphs and never enable it in production probes.
    _should_capture_attention_sync_timing = torch._dynamo.disable(
        _should_capture_attention_sync_timing)


def _should_capture_attention_fia_detail_timing(attn_state=None) -> bool:
    if not _ATTENTION_FIA_DETAIL_DEBUG_ENABLED:
        return False
    state_name = getattr(attn_state, "name", "")
    if _ATTENTION_STAGE_DEBUG_STATE_FILTER and state_name not in _ATTENTION_STAGE_DEBUG_STATE_FILTER:
        return False

    global _attention_fia_detail_debug_matched
    _attention_fia_detail_debug_matched += 1
    if _ATTENTION_FIA_DETAIL_DEBUG_LIMIT:
        if _attention_fia_detail_debug_matched > (
            _ATTENTION_FIA_DETAIL_DEBUG_SKIP + _ATTENTION_FIA_DETAIL_DEBUG_LIMIT
        ):
            return False
    if _attention_fia_detail_debug_matched <= _ATTENTION_FIA_DETAIL_DEBUG_SKIP:
        return False
    matched_after_skip = _attention_fia_detail_debug_matched - _ATTENTION_FIA_DETAIL_DEBUG_SKIP
    return matched_after_skip <= 32 or (
        _ATTENTION_FIA_DETAIL_DEBUG_INTERVAL > 0
        and matched_after_skip % _ATTENTION_FIA_DETAIL_DEBUG_INTERVAL == 0
    )


if _ATTENTION_FIA_DETAIL_DEBUG_ENABLED:
    _should_capture_attention_fia_detail_timing = torch._dynamo.disable(
        _should_capture_attention_fia_detail_timing)


def _record_event() -> torch.npu.Event | None:
    event = _maybe_new_timing_event()
    if event is not None:
        torch.npu.current_stream().record_event(event)
    return event


def _elapsed_ms(start_event: torch.npu.Event, end_event: torch.npu.Event) -> float:
    end_event.synchronize()
    return float(start_event.elapsed_time(end_event))


def _sync_perf_counter() -> float:
    torch.npu.synchronize()
    return time.perf_counter()


def _maybe_log_attention_wrapper_timing(
    attn_state,
    num_tokens: int,
    reshape_cache_ms: float,
    forward_impl_ms: float,
    total_ms: float,
) -> None:
    logger.info(
        "Attention wrapper timing pid=%s state=%s tokens=%s "
        "reshape_cache_ms=%.3f forward_impl_ms=%.3f total_ms=%.3f",
        os.getpid(),
        getattr(attn_state, "name", "None"),
        num_tokens,
        reshape_cache_ms,
        forward_impl_ms,
        total_ms,
    )


def _maybe_log_attention_sync_timing(
    attn_state,
    num_tokens: int,
    reshape_sync_ms: float,
    forward_impl_sync_ms: float,
    total_sync_ms: float,
) -> None:
    logger.info(
        "Attention sync-wall timing pid=%s state=%s tokens=%s "
        "reshape_sync_ms=%.3f forward_impl_sync_ms=%.3f total_sync_ms=%.3f",
        os.getpid(),
        getattr(attn_state, "name", "None"),
        num_tokens,
        reshape_sync_ms,
        forward_impl_sync_ms,
        total_sync_ms,
    )


def _maybe_log_attention_stage_timing(
    *,
    attn_state: "AscendAttentionState",
    op_name: str,
    num_tokens: int,
    op_ms: float,
) -> None:
    global _attention_stage_debug_total
    _attention_stage_debug_total += 1
    bucket = _token_bucket(num_tokens)
    key = (attn_state.name, op_name, bucket)
    _attention_stage_debug_summary[key] += 1

    summary = ", ".join(
        f"{state}/{op}/{tok}:{count}"
        for (state, op, tok), count in _attention_stage_debug_summary.most_common(12)
    )
    logger.info(
        "Attention stage timing pid=%s call=%d state=%s op=%s tokens=%s "
        "op_ms=%.3f summary={%s}",
        os.getpid(),
        _attention_stage_debug_total,
        attn_state.name,
        op_name,
        num_tokens,
        op_ms,
        summary,
    )


def _maybe_log_attention_fia_detail_timing(
    *,
    attn_state: "AscendAttentionState",
    num_tokens: int,
    get_params_ms: float,
    pre_op_ms: float,
    op_ms: float,
    copy_ms: float,
    total_ms: float,
) -> None:
    logger.info(
        "Attention FIA detail timing pid=%s state=%s tokens=%s "
        "get_params_ms=%.3f pre_op_ms=%.3f op_ms=%.3f "
        "copy_ms=%.3f total_ms=%.3f",
        os.getpid(),
        attn_state.name,
        num_tokens,
        get_params_ms,
        pre_op_ms,
        op_ms,
        copy_ms,
        total_ms,
    )


def _maybe_log_attention_fia_sync_detail_timing(
    *,
    attn_state: "AscendAttentionState",
    num_tokens: int,
    get_params_ms: float,
    pre_op_ms: float,
    op_ms: float,
    copy_ms: float,
    total_ms: float,
) -> None:
    logger.info(
        "Attention FIA sync detail timing pid=%s state=%s tokens=%s "
        "get_params_sync_ms=%.3f pre_op_sync_ms=%.3f op_sync_ms=%.3f "
        "copy_sync_ms=%.3f total_sync_ms=%.3f",
        os.getpid(),
        attn_state.name,
        num_tokens,
        get_params_ms,
        pre_op_ms,
        op_ms,
        copy_ms,
        total_ms,
    )


def _pad_attention_seq_params(
    actual_seq_lengths_q: list[int], seq_lens: list[int], runtime_shape: int
) -> tuple[list[int], list[int]]:
    if not actual_seq_lengths_q:
        padded_actual_seq_lengths_q = [runtime_shape]
    else:
        last_val = actual_seq_lengths_q[-1]
        if last_val >= runtime_shape:
            padded_actual_seq_lengths_q = actual_seq_lengths_q
        else:
            interpolated = list(range(last_val + 1, runtime_shape + 1))
            padded_actual_seq_lengths_q = actual_seq_lengths_q + interpolated

    target_len = len(padded_actual_seq_lengths_q)
    if len(seq_lens) >= target_len:
        padded_seq_lens = seq_lens
    else:
        padded_seq_lens = seq_lens + [0] * (target_len - len(seq_lens))

    return padded_actual_seq_lengths_q, padded_seq_lens


@register_backend(AttentionBackendEnum.CUSTOM, "ASCEND")
class AscendAttentionBackend(AttentionBackend):
    accept_output_buffer: bool = True

    @staticmethod
    def get_name() -> str:
        # HACK(Ronald1995): vllm `initialize_kv_cache` method in model runner v2 make
        # attention name assertion, we just set name to FLASH_ATTN to avoid assertion error.
        # rectify this when vllm disable the assertion.
        return "CUSTOM" if not envs_vllm.VLLM_USE_V2_MODEL_RUNNER else "FLASH_ATTN"

    @staticmethod
    def get_impl_cls() -> type["AscendAttentionBackendImpl"]:
        if enable_cp():
            from vllm_ascend.attention.context_parallel.attention_cp import AscendAttentionCPImpl

            return AscendAttentionCPImpl
        return AscendAttentionBackendImpl

    @staticmethod
    def get_builder_cls() -> type["AscendAttentionMetadataBuilder"]:
        if enable_cp():
            from vllm_ascend.attention.context_parallel.attention_cp import AscendAttentionCPMetadataBuilder

            return AscendAttentionCPMetadataBuilder
        return AscendAttentionMetadataBuilder

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
    ) -> tuple[int, ...]:
        return (2, num_blocks, block_size, num_kv_heads, head_size)

    @staticmethod
    def swap_blocks(
        src_kv_cache: list[torch.Tensor],
        dst_kv_cache: list[torch.Tensor],
        src_to_dst: torch.Tensor,
    ) -> None:
        src_key_cache, src_value_cache = src_kv_cache[0], src_kv_cache[1]
        dst_key_cache, dst_value_cache = dst_kv_cache[0], dst_kv_cache[1]
        src_indices = src_to_dst[:, 0]
        dst_indices = src_to_dst[:, 1]

        dst_key_cache[dst_indices] = src_key_cache[src_indices].to(dst_key_cache.device)
        dst_value_cache[dst_indices] = src_value_cache[src_indices].to(dst_key_cache.device)

    @staticmethod
    def copy_blocks(
        kv_caches: list[torch.Tensor],
        src_to_dists: torch.Tensor,
    ) -> None:
        src_indices = src_to_dists[:, 0]
        dst_indices = src_to_dists[:, 1]

        for kv_cache in kv_caches:
            key_caches = kv_cache[0]
            value_caches = kv_cache[1]
            key_caches[dst_indices] = key_caches[src_indices]
            value_caches[dst_indices] = value_caches[src_indices]

    @staticmethod
    def get_supported_block_size() -> list[int]:
        block_size_override = os.getenv("VLLM_ASCEND_ATTENTION_BLOCK_SIZE")
        if block_size_override:
            return [int(block_size_override)]
        if envs_ascend.VLLM_ASCEND_USE_LEGACY_ATTENTION:
            return [64]
        return [128]


class AscendAttentionState(Enum):
    PrefillNoCache = 0
    PrefillCacheHit = 1
    DecodeOnly = 2
    ChunkedPrefill = 3
    SpecDecoding = 4


@dataclass
class AscendMetadata:
    """
    Per-layer attention metadata for Ascend FlashAttention backend.

    Contains attention masks, token counts, sequence lengths and KV cache
    related properties for attention computation.
    """

    # **************************** Basic Properties ************************** #
    attn_mask: torch.Tensor | None = None
    # Current state of this attention run.
    attn_state: AscendAttentionState = AscendAttentionState.ChunkedPrefill

    # Number of tokens excluding padding.
    num_actual_tokens_pcp_padded: int = 0
    num_actual_tokens: int = 0
    num_decode_tokens: int = 0
    num_prefills: int = 0
    num_decodes: int = 0

    # The sequence length per sequence. Sequence length means the computed
    # tokens + new tokens (is None if it is a decoding).
    # (batch_size,)
    # TODO(Angazenn): The following parameters are quite redundant and
    # contains similar information (such as seq_lens seq_lens_list). We
    # should simplified these parameters once attention schema in vLLM-Ascend
    # is unified.
    seq_lens: torch.Tensor = None
    seq_lens_list: list[int] = None  # type: ignore
    actual_seq_lengths_q: list[int] = None  # type: ignore

    query_start_loc: torch.Tensor = None
    query_lens: torch.Tensor = None
    # Maximum query length in the batch (None for decoding).
    max_query_len: int | None = None

    # ********************** KV Cache Related Properties ********************* #
    # Block addresses per sequence (Seq id -> list of physical block).
    # (batch_size, max_blocks_per_seq)
    block_tables: torch.Tensor = None

    # The indices of the token slots that input tokens will be stored into.
    # E.g., if `slot_mapping` is [35, 2, 17] and the block size is 16, the
    # three tokens are stored in the 3rd slot in block 2, 2nd slot in block 0,
    # and 1st slot in block 1, respectively.
    # (num_tokens,)
    slot_mapping: torch.Tensor = None
    # pcp
    prefill: AscendMetadataForPrefill | None = None
    # dcp
    decode_meta: AscendMetadataForDecode | None = None

    causal: bool = True
    # runner_type in model_config.
    model_runner_type: str = ""
    # prefill reshape_and_cache event
    reshape_cache_event: torch.npu.Event = None

    # sliding window attention mask
    swa_mask: torch.Tensor | None = None


class AscendAttentionMetadataBuilder(AttentionMetadataBuilder[AscendMetadata]):
    """
    Builder for constructing AscendMetadata from CommonAttentionMetadata.

    Handles attention mask generation and metadata preparation for
    Ascend FlashAttention backend.
    """

    # Does this backend/builder reorder the batch?
    # If not, set this to None. Otherwise set it to the query
    # length that will be pulled into the front of the batch.
    reorder_batch_threshold: ClassVar[int] = 1

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.compilation_config = vllm_config.compilation_config
        self.device = device
        self.max_num_blocks_per_req = cdiv(
            self.model_config.max_model_len, AscendAttentionBackend.get_supported_block_size()[0]
        )

        self.speculative_config = vllm_config.speculative_config
        self.decode_threshold = 1
        if self.speculative_config:
            spec_token_num = self.speculative_config.num_speculative_tokens
            self.decode_threshold += spec_token_num
            assert self.decode_threshold <= 16, (
                f"decode_threshold exceeded \
                npu_fused_infer_attention_score TND layout's limit of 16, \
                got {self.decode_threshold}"
            )

        AscendAttentionMetadataBuilder.reorder_batch_threshold = self.decode_threshold

        scheduler_config = vllm_config.scheduler_config
        self.chunked_prefill_enabled = scheduler_config.enable_chunked_prefill
        self.attn_mask_builder = AttentionMaskBuilder(self.device)

    @classmethod
    def get_cudagraph_support(
        cls: type["AscendAttentionMetadataBuilder"],
        vllm_config: VllmConfig,
        kv_cache_spec: AttentionSpec,
    ) -> AttentionCGSupport:
        # Explicit override in case the underlying builder specialized this getter.
        # @override omitted only because of mypy limitation due to type variable.
        return AttentionCGSupport.ALWAYS

    def reorder_batch(self, input_batch, scheduler_output: "SchedulerOutput") -> bool:
        return False

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: AscendCommonAttentionMetadata,
        fast_build: bool = False,
    ) -> AscendMetadata:
        num_reqs = common_attn_metadata.num_reqs
        num_actual_tokens = common_attn_metadata.num_actual_tokens
        query_start_loc_cpu = common_attn_metadata.query_start_loc_cpu[: num_reqs + 1]
        query_lens = query_start_loc_cpu[1:] - query_start_loc_cpu[:-1]

        num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens = split_decodes_and_prefills(
            common_attn_metadata, decode_threshold=self.decode_threshold
        )

        block_table = common_attn_metadata.block_table_tensor
        seq_lens = common_attn_metadata.seq_lens_cpu[:num_reqs]

        slot_mapping = common_attn_metadata.slot_mapping[:num_actual_tokens]
        if isinstance(self.kv_cache_spec, CrossAttentionSpec):
            seq_lens = common_attn_metadata.seq_lens
            slot_mapping = common_attn_metadata.slot_mapping.to(torch.int32)
        attn_state = common_attn_metadata.attn_state

        # Get attn_mask and swa_mask from singleton AttentionMaskBuilder.  The
        # legacy eager ops expect the old state-specific mask layouts, while
        # the newer FIA path uses a fixed splitfuse-style mask.
        if envs_ascend.VLLM_ASCEND_USE_LEGACY_ATTENTION:
            if attn_state == AscendAttentionState.DecodeOnly:
                attn_mask = None
            elif attn_state == AscendAttentionState.PrefillNoCache:
                max_seq_len = max(seq_lens.max().item(), 1)
                attn_mask = self.attn_mask_builder.get_attn_mask(
                    max_seq_len, self.model_config.dtype
                )
            elif attn_state == AscendAttentionState.PrefillCacheHit:
                attn_mask = self.attn_mask_builder.get_attn_mask(
                    128, self.model_config.dtype
                )
            elif envs_ascend.VLLM_ASCEND_LEGACY_ATTENTION_SPLITFUSE:
                attn_mask = self.attn_mask_builder.get_legacy_splitfuse_attn_mask(
                    seq_lens, common_attn_metadata.positions, self.model_config.dtype
                )
            else:
                attn_mask = self.attn_mask_builder.get_attention_mask(self.model_config)
        else:
            attn_mask = self.attn_mask_builder.get_attention_mask(self.model_config)

        swa_mask = None
        is_swa = hasattr(self.model_config.hf_text_config, "sliding_window")
        if self.model_config is not None and is_swa:
            swa_mask = self.attn_mask_builder.get_swa_mask(
                self.model_config.dtype, self.model_config.hf_text_config.sliding_window
            )

        query_start_loc = query_start_loc_cpu.pin_memory().to(self.device, non_blocking=True)

        attn_metadata = AscendMetadata(
            num_actual_tokens=num_actual_tokens,
            num_decode_tokens=num_decode_tokens,
            block_tables=block_table,
            query_start_loc=query_start_loc,
            query_lens=query_lens,
            seq_lens=seq_lens,
            seq_lens_list=seq_lens.tolist(),
            max_query_len=common_attn_metadata.max_query_len,
            actual_seq_lengths_q=query_start_loc_cpu[1:].tolist(),
            slot_mapping=slot_mapping,
            attn_mask=attn_mask,
            swa_mask=swa_mask,
            attn_state=attn_state,
            num_prefills=num_prefills,
            num_decodes=num_decodes,
            causal=common_attn_metadata.causal,
            model_runner_type=self.model_config.runner_type,
        )
        return attn_metadata

    def build_for_graph_capture(
        self,
        common_attn_metadata: AscendCommonAttentionMetadata,
        attn_state: AscendAttentionState = AscendAttentionState.DecodeOnly,
    ):
        if attn_state in (AscendAttentionState.DecodeOnly, AscendAttentionState.ChunkedPrefill):
            attn_metadata = self.build(
                common_prefix_len=0,
                common_attn_metadata=common_attn_metadata,
            )
        else:
            raise NotImplementedError(
                "Currently we only support building dummy metadata for DecodeOnly and ChunkedPrefill state"
            )

        attn_metadata.attn_state = attn_state
        return attn_metadata


class AscendAttentionBackendImpl(AttentionImpl):
    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: list[float] | None,
        sliding_window: int | None,
        kv_cache_dtype: str,
        logits_soft_cap: float | None,
        attn_type: str,
        kv_sharing_target_layer_name: str | None,
        **kwargs,
    ) -> None:
        self.vllm_config = get_current_vllm_config()
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_heads if num_kv_heads is None else num_kv_heads
        self.hidden_size = self.num_heads * self.head_size
        self.kv_cache_dtype = kv_cache_dtype
        self.sliding_window = sliding_window
        if alibi_slopes is not None:
            alibi_slopes = torch.tensor(alibi_slopes, dtype=torch.float32, device="npu")
        self.alibi_slopes = alibi_slopes
        self.attn_type = attn_type

        assert self.num_heads % self.num_kv_heads == 0
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads
        self.key_cache = None
        self.value_cache = None
        self.is_kv_producer = (
            self.vllm_config.kv_transfer_config is not None and self.vllm_config.kv_transfer_config.is_kv_producer
        )
        global _LEGACY_ATTENTION_LOGGED
        if envs_ascend.VLLM_ASCEND_USE_LEGACY_ATTENTION and not _LEGACY_ATTENTION_LOGGED:
            logger.info(
                "Using legacy Ascend attention eager op mix: block_size=%s "
                "decode=_npu_paged_attention chunked_prefill=%s",
                AscendAttentionBackend.get_supported_block_size()[0],
                (
                    "_npu_paged_attention_splitfuse"
                    if envs_ascend.VLLM_ASCEND_LEGACY_ATTENTION_SPLITFUSE
                    else "npu_fused_infer_attention_score"
                ),
            )
            _LEGACY_ATTENTION_LOGGED = True

    @staticmethod
    def update_graph_params(
        update_stream,
        forward_context,
        num_tokens,
        vllm_config,
        speculative_config=None,
        num_dcp_pcp_tokens=None,
    ):
        if using_paged_attention(num_tokens, vllm_config):
            # Paged Attention update logic
            if forward_context.is_draft_model:
                graph_params = get_draft_graph_params()
            else:
                graph_params = get_graph_params()
            with torch.npu.stream(update_stream):
                for key, param, handle, event in zip(
                    forward_context.attn_metadata,
                    graph_params.attn_params[num_tokens],
                    graph_params.handles[num_tokens],
                    graph_params.events[num_tokens],
                ):
                    (
                        query,
                        key_cache,
                        value_cache,
                        num_kv_heads,
                        num_heads,
                        scale,
                        block_table,
                        seq_lens,
                        output,
                    ) = param
                    seq_lens = forward_context.attn_metadata[key].seq_lens

                    workspace = torch_npu._npu_paged_attention_get_workspace(
                        query=query,
                        key_cache=key_cache,
                        value_cache=value_cache,
                        num_kv_heads=num_kv_heads,
                        num_heads=num_heads,
                        scale_value=scale,
                        block_table=block_table,
                        context_lens=seq_lens,
                        out=output,
                    )
                    torch.npu.graph_task_update_begin(update_stream, handle)
                    torch_npu._npu_paged_attention(
                        query=query,
                        key_cache=key_cache,
                        value_cache=value_cache,
                        num_kv_heads=num_kv_heads,
                        num_heads=num_heads,
                        scale_value=scale,
                        block_table=block_table,
                        context_lens=seq_lens,
                        out=output,
                        workspace=workspace,
                    )
                    torch.npu.graph_task_update_end(update_stream)
                    event.record(update_stream)
        else:
            # FIA update logic
            if forward_context.is_draft_model:
                graph_params = get_draft_graph_params()
                attn_metadata = forward_context.draft_attn_metadatas
                attn_keys = list(attn_metadata[0].keys())
            else:
                graph_params = get_graph_params()
                attn_metadata = forward_context.attn_metadata
                attn_keys = list(attn_metadata.keys())
            # For Qwen3-next, since the kv_cache_config has already categorized
            # linear_attn and self_attn, the attn_metadata is first arranged with
            # self_attn followed by linear_attn. Therefore, using zip directly
            # filters out the update operations for linear_attn.
            # TODO: We use a new variable `attn_keys` to ensure the loop count is
            # correct after get by `zip` because of the new structure of the attn_metadata
            # when running with the merged full eagle-graph. Should check it with Qwen3-next.
            num_layers = len(attn_keys)
            if num_layers == 0:
                return
            if forward_context.is_draft_model:
                attn_keys = attn_keys * (len(graph_params.attn_params[num_tokens]) // num_layers)
            attn_count = 0
            with torch.npu.stream(update_stream):
                for key, param, handle, event in zip(
                    attn_keys,
                    graph_params.attn_params[num_tokens],
                    graph_params.handles[num_tokens],
                    graph_params.events[num_tokens],
                ):
                    (
                        query,
                        key_cache,
                        value,
                        block_tables,
                        attn_mask,
                        block_size,
                        seq_lens,
                        query_start_loc,
                        num_kv_heads,
                        num_heads,
                        scale,
                        attn_output,
                        softmax_lse,
                    ) = param

                    if forward_context.is_draft_model:
                        draft_step = attn_count // num_layers
                        seq_lens = attn_metadata[draft_step][key].seq_lens_list
                        actual_seq_lengths_q = attn_metadata[draft_step][key].actual_seq_lengths_q
                        attn_count = attn_count + 1
                    else:
                        seq_lens = attn_metadata[key].seq_lens_list
                        actual_seq_lengths_q = attn_metadata[key].actual_seq_lengths_q
                    actual_seq_lengths_q, seq_lens = _pad_attention_seq_params(
                        actual_seq_lengths_q, seq_lens, num_tokens
                    )
                    torch.npu.graph_task_update_begin(update_stream, handle)
                    torch_npu.npu_fused_infer_attention_score.out(
                        query=query,
                        key=key_cache,
                        value=value,
                        block_table=block_tables,
                        atten_mask=attn_mask,
                        input_layout="TND",
                        block_size=block_size,
                        actual_seq_lengths=actual_seq_lengths_q,
                        actual_seq_lengths_kv=seq_lens,
                        num_key_value_heads=num_kv_heads,
                        num_heads=num_heads,
                        scale=scale,
                        sparse_mode=3,
                        workspace=graph_params.workspaces.get(num_tokens),
                        out=[attn_output, softmax_lse],
                    )
                    torch.npu.graph_task_update_end(update_stream)

                    event.record(update_stream)

    def process_weights_after_loading(self, act_dtype: torch.dtype):
        super().process_weights_after_loading(act_dtype)
        if flashcomm2_oshard_manager.flashcomm2_oshard_enable():
            flashcomm2_oshard_manager.post_process_after_loading()

    def full_graph_fia(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AscendMetadata,
        output: torch.Tensor,
    ) -> torch.Tensor:
        key, value, block_size, block_table, actual_seq_lengths_kv = self._get_fia_params(key, value, attn_metadata)

        num_tokens = attn_metadata.actual_seq_lengths_q[-1]
        forward_context = get_forward_context()
        if forward_context.is_draft_model:
            graph_params = get_draft_graph_params()
        else:
            graph_params = get_graph_params()
        actual_seq_lengths_q = attn_metadata.actual_seq_lengths_q
        # Prepare tensors for attention output
        # TODO: Refactor this to step-level instead of layer-level

        # Get workspace from cache or calculate it if not present.
        workspace = graph_params.workspaces.get(num_tokens)
        softmax_lse = torch.empty(1, dtype=query.dtype, device=query.device)
        if workspace is None:
            workspace = torch_npu._npu_fused_infer_attention_score_get_max_workspace(
                query=query,
                key=key,
                value=value,
                atten_mask=attn_metadata.attn_mask,
                block_table=block_table,
                input_layout="TND",
                block_size=block_size,
                actual_seq_lengths=actual_seq_lengths_q,
                actual_seq_lengths_kv=actual_seq_lengths_kv,
                num_key_value_heads=self.num_kv_heads,
                num_heads=self.num_heads,
                sparse_mode=3,
                scale=self.scale,
            )
            if forward_context.is_draft_model:
                update_draft_graph_params_workspaces(num_tokens, workspace)
            else:
                update_graph_params_workspaces(num_tokens, workspace)

        # Handle graph capturing mode
        stream = torch_npu.npu.current_stream()

        event = torch.npu.ExternalEvent()
        event.wait(stream)
        event.reset(stream)
        graph_params.events[num_tokens].append(event)
        graph_params.attn_params[num_tokens].append(
            (
                weak_ref_tensors(query),
                weak_ref_tensors(key),
                weak_ref_tensors(value),
                weak_ref_tensors(block_table),
                weak_ref_tensors(attn_metadata.attn_mask),
                block_size,
                actual_seq_lengths_kv,
                actual_seq_lengths_q,
                self.num_kv_heads,
                self.num_heads,
                self.scale,
                weak_ref_tensors(output),
                weak_ref_tensors(softmax_lse),
            )
        )

        torch.npu.graph_task_group_begin(stream)
        torch_npu.npu_fused_infer_attention_score.out(
            query=query,
            key=key,
            value=value,
            atten_mask=attn_metadata.attn_mask,
            block_table=block_table,
            input_layout="TND",
            block_size=block_size,
            actual_seq_lengths=actual_seq_lengths_q,
            actual_seq_lengths_kv=actual_seq_lengths_kv,
            num_key_value_heads=self.num_kv_heads,
            num_heads=self.num_heads,
            scale=self.scale,
            sparse_mode=3,
            workspace=workspace,
            out=[output, softmax_lse],
        )

        output = output.view(num_tokens, self.num_heads, self.head_size)

        handle = torch.npu.graph_task_group_end(stream)
        graph_params.handles[num_tokens].append(handle)
        return output, num_tokens

    def full_graph_pa(
        self,
        query: torch.Tensor,
        attn_metadata: AscendMetadata,
        output: torch.Tensor | None = None,
    ):
        graph_params = get_graph_params()
        forward_context: ForwardContext = get_forward_context()
        num_tokens = query.shape[0]
        if forward_context.capturing:
            # Get workspace from cache or calculate it if not present.
            workspace = graph_params.workspaces.get(num_tokens)
            if workspace is None:
                workspace = torch_npu._npu_paged_attention_get_workspace(
                    query=query,
                    key_cache=self.key_cache,
                    value_cache=self.value_cache,
                    num_kv_heads=self.num_kv_heads,
                    num_heads=self.num_heads,
                    scale_value=self.scale,
                    block_table=attn_metadata.block_tables,
                    context_lens=attn_metadata.seq_lens,
                    out=output,
                )
                update_graph_params_workspaces(num_tokens, workspace)

            # Handle graph capturing mode
            stream = torch_npu.npu.current_stream()

            event = torch.npu.ExternalEvent()
            event.wait(stream)
            event.reset(stream)
            graph_params.events[num_tokens].append(event)
            graph_params.attn_params[num_tokens].append(
                (
                    weak_ref_tensors(query),
                    weak_ref_tensors(self.key_cache),
                    weak_ref_tensors(self.value_cache),
                    self.num_kv_heads,
                    self.num_heads,
                    self.scale,
                    attn_metadata.block_tables,
                    attn_metadata.seq_lens,
                    weak_ref_tensors(output),
                )
            )

            torch.npu.graph_task_group_begin(stream)
            torch_npu._npu_paged_attention(
                query=query,
                key_cache=self.key_cache,
                value_cache=self.value_cache,
                num_kv_heads=self.num_kv_heads,
                num_heads=self.num_heads,
                scale_value=self.scale,
                block_table=attn_metadata.block_tables,
                context_lens=attn_metadata.seq_lens,
                out=output,
                workspace=workspace,
            )
            handle = torch.npu.graph_task_group_end(stream)
            graph_params.handles[num_tokens].append(handle)
            return output

    def _get_fia_params(self, key: torch.Tensor, value: torch.Tensor, attn_metadata: AscendMetadata):
        if attn_metadata.attn_state == AscendAttentionState.PrefillNoCache:
            block_size = 128
            block_table = None
            actual_seq_lengths_kv = attn_metadata.actual_seq_lengths_q
            if self.attn_type == AttentionType.ENCODER_DECODER:
                actual_seq_lengths_kv = torch.cumsum(attn_metadata.seq_lens, dim=0).tolist()
        elif attn_metadata.attn_state == AscendAttentionState.PrefillCacheHit:
            batch_size = attn_metadata.seq_lens.shape[0]
            block_table = attn_metadata.block_tables[:batch_size, :]
            num_block, block_size, _, _ = self.key_cache.shape  # type: ignore
            key = self.key_cache.view(  # type: ignore
                num_block, block_size, -1
            )
            value = self.value_cache.view(  # type: ignore
                num_block, block_size, -1
            )
            actual_seq_lengths_kv = attn_metadata.seq_lens_list
        elif attn_metadata.attn_state == AscendAttentionState.DecodeOnly:
            num_block, block_size, _, _ = self.key_cache.shape  # type: ignore
            key = self.key_cache.view(  # type: ignore
                num_block, block_size, -1
            )
            value = self.value_cache.view(  # type: ignore
                num_block, block_size, -1
            )
            block_table = attn_metadata.block_tables
            actual_seq_lengths_kv = attn_metadata.seq_lens_list
        # chunked prefill.
        else:
            num_block, block_size, _, _ = self.key_cache.shape  # type: ignore
            key = self.key_cache.view(  # type: ignore
                num_block, block_size, -1
            )
            value = self.value_cache.view(  # type: ignore
                num_block, block_size, -1
            )
            block_table = attn_metadata.block_tables
            actual_seq_lengths_kv = attn_metadata.seq_lens_list
        return key, value, block_size, block_table, actual_seq_lengths_kv

    def _forward_fia_slidingwindow(self, query: torch.Tensor, attn_metadata: AscendMetadata, output: torch.Tensor):
        batch_size = attn_metadata.seq_lens.shape[0]
        block_size = 128
        query = query.view(batch_size, 1, self.num_heads * self.head_size)
        key = self.key_cache
        value = self.value_cache
        if self.key_cache is not None and self.value_cache is not None:
            block_size = self.key_cache.shape[1]
            key = self.key_cache.flatten(2, 3).contiguous()
            value = self.value_cache.flatten(2, 3).contiguous()

        output, _ = torch_npu.npu_fused_infer_attention_score(
            query,
            key,
            value,
            num_heads=self.num_heads,
            num_key_value_heads=self.num_kv_heads,
            input_layout="BSH",
            block_size=block_size,
            pre_tokens=self.sliding_window,
            scale=self.scale,
            block_table=attn_metadata.block_tables,
            actual_seq_lengths=[1] * len(attn_metadata.seq_lens),
            actual_seq_lengths_kv=attn_metadata.seq_lens,
        )

        output = output.view(batch_size, self.num_heads, self.head_size)
        return output

    def forward_fused_infer_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AscendMetadata,
        output: torch.Tensor,
    ):
        stage_start_evt = None
        if _ATTENTION_STAGE_DEBUG_ENABLED and _should_capture_attention_stage_timing(
                attn_metadata.attn_state, "fia"):
            stage_start_evt = _record_event()
        forward_context: ForwardContext = get_forward_context()
        # we inherit ForwardContext in model runner v2, when enable model
        # runner v2, there is not capturing attribute in forward_context,
        # just use getattr to avoid attribute error.
        if getattr(forward_context, "capturing", False) and not _DISABLE_FULL_GRAPH_FIA:
            attn_output, num_tokens = self.full_graph_fia(query, key, value, attn_metadata, output)
            output[:num_tokens] = attn_output[:num_tokens]
            stage_end_evt = _record_event() if stage_start_evt is not None else None
            if stage_start_evt is not None and stage_end_evt is not None:
                _maybe_log_attention_stage_timing(
                    attn_state=attn_metadata.attn_state,
                    op_name="full_graph_fia",
                    num_tokens=num_tokens,
                    op_ms=_elapsed_ms(stage_start_evt, stage_end_evt),
                )
            return output
        if (
            attn_metadata.attn_state == AscendAttentionState.DecodeOnly
            and self.sliding_window is not None
            and attn_metadata.seq_lens.shape[0] == query.size(0)
        ):
            output = self._forward_fia_slidingwindow(query, attn_metadata, output)
            stage_end_evt = _record_event() if stage_start_evt is not None else None
            if stage_start_evt is not None and stage_end_evt is not None:
                _maybe_log_attention_stage_timing(
                    attn_state=attn_metadata.attn_state,
                    op_name="fia_slidingwindow",
                    num_tokens=query.shape[0],
                    op_ms=_elapsed_ms(stage_start_evt, stage_end_evt),
            )
            return output
        capture_fia_detail = False
        if _ATTENTION_FIA_DETAIL_DEBUG_ENABLED:
            capture_fia_detail = _should_capture_attention_fia_detail_timing(
                attn_metadata.attn_state)
        detail_start_evt = _record_event() if capture_fia_detail else None
        detail_start_sync = _sync_perf_counter() if capture_fia_detail else None
        key, value, block_size, block_table, actual_seq_lengths_kv = self._get_fia_params(key, value, attn_metadata)
        detail_after_params_evt = _record_event() if capture_fia_detail else None
        detail_after_params_sync = _sync_perf_counter() if capture_fia_detail else None
        num_tokens = attn_metadata.actual_seq_lengths_q[-1]
        query = query[:num_tokens]
        actual_seq_lengths_q = attn_metadata.actual_seq_lengths_q
        if (
            attn_metadata.attn_state == AscendAttentionState.PrefillNoCache
            and self.attn_type != AttentionType.ENCODER_DECODER
        ):
            key = key[:num_tokens]
            value = value[:num_tokens]
        detail_before_op_evt = _record_event() if capture_fia_detail else None
        detail_before_op_sync = _sync_perf_counter() if capture_fia_detail else None
        attn_output, _ = torch_npu.npu_fused_infer_attention_score(
            query=query,
            key=key,
            value=value,
            atten_mask=attn_metadata.attn_mask,
            block_table=block_table,
            input_layout="TND",
            block_size=block_size,
            actual_seq_lengths=actual_seq_lengths_q,
            actual_seq_lengths_kv=actual_seq_lengths_kv,
            num_key_value_heads=self.num_kv_heads,
            num_heads=self.num_heads,
            scale=self.scale,
            sparse_mode=3,
        )

        detail_after_op_evt = _record_event() if capture_fia_detail else None
        detail_after_op_sync = _sync_perf_counter() if capture_fia_detail else None
        attn_output = attn_output.view(num_tokens, self.num_heads, self.head_size)
        output[:num_tokens] = attn_output[:num_tokens]
        detail_after_copy_evt = _record_event() if capture_fia_detail else None
        detail_after_copy_sync = _sync_perf_counter() if capture_fia_detail else None
        if (
            capture_fia_detail
            and detail_start_evt is not None
            and detail_after_params_evt is not None
            and detail_before_op_evt is not None
            and detail_after_op_evt is not None
            and detail_after_copy_evt is not None
        ):
            _maybe_log_attention_fia_detail_timing(
                attn_state=attn_metadata.attn_state,
                num_tokens=num_tokens,
                get_params_ms=_elapsed_ms(detail_start_evt, detail_after_params_evt),
                pre_op_ms=_elapsed_ms(detail_after_params_evt, detail_before_op_evt),
                op_ms=_elapsed_ms(detail_before_op_evt, detail_after_op_evt),
                copy_ms=_elapsed_ms(detail_after_op_evt, detail_after_copy_evt),
                total_ms=_elapsed_ms(detail_start_evt, detail_after_copy_evt),
            )
        if (
            capture_fia_detail
            and detail_start_sync is not None
            and detail_after_params_sync is not None
            and detail_before_op_sync is not None
            and detail_after_op_sync is not None
            and detail_after_copy_sync is not None
        ):
            _maybe_log_attention_fia_sync_detail_timing(
                attn_state=attn_metadata.attn_state,
                num_tokens=num_tokens,
                get_params_ms=(detail_after_params_sync - detail_start_sync) * 1000,
                pre_op_ms=(detail_before_op_sync - detail_after_params_sync) * 1000,
                op_ms=(detail_after_op_sync - detail_before_op_sync) * 1000,
                copy_ms=(detail_after_copy_sync - detail_after_op_sync) * 1000,
                total_ms=(detail_after_copy_sync - detail_start_sync) * 1000,
            )
        stage_end_evt = _record_event() if stage_start_evt is not None else None
        if stage_start_evt is not None and stage_end_evt is not None:
            _maybe_log_attention_stage_timing(
                attn_state=attn_metadata.attn_state,
                op_name="fia",
                num_tokens=num_tokens,
                op_ms=_elapsed_ms(stage_start_evt, stage_end_evt),
            )
        return output

    def forward_paged_attention(
        self,
        query: torch.Tensor,
        attn_metadata: AscendMetadata,
        output: torch.Tensor | None = None,
    ) -> torch.Tensor:
        stage_start_evt = None
        if _ATTENTION_STAGE_DEBUG_ENABLED and _should_capture_attention_stage_timing(
                attn_metadata.attn_state, "paged_attention"):
            stage_start_evt = _record_event()
        forward_context: ForwardContext = get_forward_context()
        if forward_context.capturing and not _DISABLE_FULL_GRAPH_PA:
            output = self.full_graph_pa(query, attn_metadata, output)
            stage_end_evt = _record_event() if stage_start_evt is not None else None
            if stage_start_evt is not None and stage_end_evt is not None:
                _maybe_log_attention_stage_timing(
                    attn_state=attn_metadata.attn_state,
                    op_name="full_graph_paged_attention",
                    num_tokens=query.shape[0],
                    op_ms=_elapsed_ms(stage_start_evt, stage_end_evt),
                )
            return output
        torch_npu._npu_paged_attention(
            query=query,
            key_cache=self.key_cache,
            value_cache=self.value_cache,
            num_kv_heads=self.num_kv_heads,
            num_heads=self.num_heads,
            scale_value=self.scale,
            block_table=attn_metadata.block_tables,
            context_lens=attn_metadata.seq_lens,
            out=output,
        )
        stage_end_evt = _record_event() if stage_start_evt is not None else None
        if stage_start_evt is not None and stage_end_evt is not None:
            _maybe_log_attention_stage_timing(
                attn_state=attn_metadata.attn_state,
                op_name="paged_attention",
                num_tokens=query.shape[0],
                op_ms=_elapsed_ms(stage_start_evt, stage_end_evt),
            )
        return output

    def _forward_encoder_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AscendMetadata,
        _: torch.Tensor,
    ) -> torch.Tensor:
        assert attn_metadata is not None

        if attn_metadata.causal:
            # use sparse_mode 3 in causal scenario
            return torch_npu.npu_fusion_attention(
                query=query,
                key=key,
                value=value,
                head_num=self.num_heads,
                input_layout="TND",
                scale=self.scale,
                sparse_mode=3,
                atten_mask=attn_metadata.attn_mask,
                actual_seq_qlen=attn_metadata.actual_seq_lengths_q,
                actual_seq_kvlen=attn_metadata.actual_seq_lengths_q,
            )[0]
        else:
            # use default sparse_mode 0 in normal scenario, which means no mask works on it
            return torch_npu.npu_fusion_attention(
                query=query,
                key=key,
                value=value,
                head_num=self.num_heads,
                input_layout="TND",
                scale=self.scale,
                actual_seq_qlen=attn_metadata.actual_seq_lengths_q,
                actual_seq_kvlen=attn_metadata.actual_seq_lengths_q,
            )[0]

    def _forward_legacy_prefill_no_cache(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AscendMetadata,
        output: torch.Tensor,
    ) -> torch.Tensor:
        if attn_metadata.attn_mask is None:
            return self.forward_fused_infer_attention(query, key, value, attn_metadata, output)
        num_tokens = attn_metadata.num_actual_tokens
        torch_npu._npu_flash_attention(
            query=query[:num_tokens],
            key=key[:num_tokens],
            value=value[:num_tokens],
            mask=attn_metadata.attn_mask,
            seq_len=attn_metadata.seq_lens,
            scale_value=self.scale,
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            out=output[:num_tokens],
        )
        return output

    def _forward_legacy_prefill_cache_hit(
        self,
        query: torch.Tensor,
        attn_metadata: AscendMetadata,
        output: torch.Tensor,
    ) -> torch.Tensor:
        if attn_metadata.attn_mask is None:
            return self.forward_paged_attention(query, attn_metadata, output)
        batch_size = attn_metadata.query_lens.shape[0]
        block_table = attn_metadata.block_tables[:batch_size, :]
        torch_npu._npu_flash_attention_qlens(
            query=query,
            key_cache=self.key_cache,
            value_cache=self.value_cache,
            block_table=block_table,
            mask=attn_metadata.attn_mask,
            seq_len=attn_metadata.query_lens,
            context_lens=attn_metadata.seq_lens,
            num_kv_heads=self.num_kv_heads,
            num_heads=self.num_heads,
            scale_value=self.scale,
            out=output,
        )
        return output

    def _forward_legacy_decode_only(
        self,
        query: torch.Tensor,
        attn_metadata: AscendMetadata,
        output: torch.Tensor,
    ) -> torch.Tensor:
        if (
            self.sliding_window is not None
            and attn_metadata.seq_lens.shape[0] == query.size(0)
        ):
            return self._forward_fia_slidingwindow(query, attn_metadata, output)
        torch_npu._npu_paged_attention(
            query=query,
            key_cache=self.key_cache,
            value_cache=self.value_cache,
            num_kv_heads=self.num_kv_heads,
            num_heads=self.num_heads,
            scale_value=self.scale,
            block_table=attn_metadata.block_tables,
            context_lens=attn_metadata.seq_lens,
            out=output,
        )
        return output

    def _forward_legacy_chunked_prefill(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AscendMetadata,
        output: torch.Tensor,
    ) -> torch.Tensor:
        # The old eager rollout used splitfuse for normal V1 chunked-prefill,
        # but the op is more sensitive to newer V1 metadata layouts than decode
        # paged attention. Keep it behind a separate flag so we can preserve the
        # high-value legacy decode path while retaining a stable FIA prefill.
        if (
            not envs_ascend.VLLM_ASCEND_LEGACY_ATTENTION_SPLITFUSE
            or attn_metadata.attn_mask is None
            or self.head_size == 192
        ):
            return self.forward_fused_infer_attention(query, key, value, attn_metadata, output)
        torch_npu._npu_paged_attention_splitfuse(
            query=query,
            key_cache=self.key_cache,
            value_cache=self.value_cache,
            mask=attn_metadata.attn_mask,
            block_table=attn_metadata.block_tables,
            seq_len=attn_metadata.query_lens,
            context_lens=attn_metadata.seq_lens,
            num_kv_heads=self.num_kv_heads,
            num_heads=self.num_heads,
            scale_value=self.scale,
            out=output,
        )
        return output

    def forward_impl_legacy(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AscendMetadata,
        output: torch.Tensor,
    ) -> torch.Tensor:
        if attn_metadata.attn_state == AscendAttentionState.PrefillNoCache:
            return self._forward_legacy_prefill_no_cache(query, key, value, attn_metadata, output)
        if attn_metadata.attn_state == AscendAttentionState.PrefillCacheHit:
            return self._forward_legacy_prefill_cache_hit(query, attn_metadata, output)
        if attn_metadata.attn_state == AscendAttentionState.DecodeOnly:
            return self._forward_legacy_decode_only(query, attn_metadata, output)
        return self._forward_legacy_chunked_prefill(query, key, value, attn_metadata, output)

    def reshape_and_cache(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: tuple[torch.Tensor],
        attn_metadata: AscendMetadata,
    ):
        if len(kv_cache) > 1:
            if self.is_kv_producer:
                attn_metadata.reshape_cache_event = torch.npu.Event()
            if self.key_cache is None:
                self.key_cache, self.value_cache = kv_cache[0], kv_cache[1]
            slots = attn_metadata.slot_mapping
            encoder_decoder = self.attn_type == AttentionType.ENCODER_DECODER
            DeviceOperator.reshape_and_cache(
                key=key[: attn_metadata.num_actual_tokens] if not encoder_decoder else key,
                value=value[: attn_metadata.num_actual_tokens] if not encoder_decoder else value,
                key_cache=self.key_cache,
                value_cache=self.value_cache,
                slot_mapping=slots[: attn_metadata.num_actual_tokens] if not encoder_decoder else slots,
            )
            if self.is_kv_producer:
                attn_metadata.reshape_cache_event.record()
        return key, value

    def forward_impl(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: tuple[torch.Tensor],
        attn_metadata: AscendMetadata,
        output: torch.Tensor,
    ):
        if (
            envs_ascend.VLLM_ASCEND_USE_LEGACY_ATTENTION
            and self.attn_type == AttentionType.DECODER
        ):
            if attn_metadata.attn_state == AscendAttentionState.DecodeOnly:
                op_name = "legacy_paged_attention"
            elif attn_metadata.attn_state == AscendAttentionState.PrefillNoCache:
                op_name = "legacy_flash_attention" if attn_metadata.attn_mask is not None else "fia"
            elif attn_metadata.attn_state == AscendAttentionState.PrefillCacheHit:
                op_name = "legacy_flash_attention_qlens" if attn_metadata.attn_mask is not None else "paged_attention"
            elif (
                envs_ascend.VLLM_ASCEND_LEGACY_ATTENTION_SPLITFUSE
                and attn_metadata.attn_mask is not None
                and self.head_size != 192
            ):
                op_name = "legacy_paged_attention_splitfuse"
            else:
                op_name = "fia"
            if _ATTENTION_DEBUG_ENABLED:
                _maybe_log_attention_path(
                    attn_metadata.attn_state,
                    op_name,
                    query.shape[0],
                    attn_metadata.num_decodes,
                    attn_metadata.num_prefills,
                )
            return self.forward_impl_legacy(query, key, value, attn_metadata, output)
        num_tokens = query.shape[0]
        use_decode_pa = (
            attn_metadata.attn_state == AscendAttentionState.DecodeOnly
            and self.sliding_window is None
            and (_FORCE_PAGED_ATTENTION_DECODE or using_paged_attention(num_tokens, self.vllm_config))
        )
        if _ATTENTION_DEBUG_ENABLED:
            _maybe_log_attention_path(
                attn_metadata.attn_state,
                "paged_attention" if use_decode_pa else "fia",
                num_tokens,
                attn_metadata.num_decodes,
                attn_metadata.num_prefills,
            )
        if (
            use_decode_pa
        ):
            output = self.forward_paged_attention(query, attn_metadata, output)
        else:
            output = self.forward_fused_infer_attention(query, key, value, attn_metadata, output)

        return output

    def forward(
        self,
        layer: AttentionLayer,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: tuple[torch.Tensor],
        attn_metadata: AscendMetadata,
        output: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass with Ascend attention.
        Args:
            query: shape = [num_tokens, num_heads, head_size]
            key: shape = [num_tokens, num_kv_heads, head_size]
            value: shape = [num_tokens, num_kv_heads, head_size]
            kv_cache: shape =
                [2, num_blocks, block_size, num_kv_heads, head_size]
            attn_metadata: Metadata for attention.
        Returns:
            shape = [num_tokens, num_heads * head_size]
        """
        assert output is not None, "Output tensor must be provided."

        if output_scale is not None or output_block_scale is not None:
            raise NotImplementedError("fused output quantization is not yet supported for AscendAttentionBackendImpl")

        assert layer._k_scale_float == 1.0 and layer._v_scale_float == 1.0
        num_tokens = query.shape[0]
        if attn_metadata is None:
            return output.fill_(0)

        capture_wrapper_timing = False
        if _ATTENTION_WRAPPER_DEBUG_ENABLED:
            capture_wrapper_timing = _should_capture_attention_wrapper_timing(
                getattr(attn_metadata, "attn_state", None))
        capture_sync_timing = False
        if _ATTENTION_SYNC_DEBUG_ENABLED:
            capture_sync_timing = _should_capture_attention_sync_timing(
                getattr(attn_metadata, "attn_state", None))
        wrapper_start_evt = _record_event() if capture_wrapper_timing else None
        sync_start = _sync_perf_counter() if capture_sync_timing else None
        sync_after_reshape = None
        reshape_end_evt = None
        if key is not None and value is not None:
            if _ATTENTION_VALUE_CONTIGUOUS:
                # Old eager made the V tensor contiguous before reshape/cache.
                # Keep this as a diagnostic knob because it can affect NPU cache
                # write/read layout without changing attention numerics.
                value = value.contiguous()
            key, value = self.reshape_and_cache(key, value, kv_cache, attn_metadata)
            sync_after_reshape = _sync_perf_counter() if capture_sync_timing else None
            reshape_end_evt = _record_event() if capture_wrapper_timing else None
        elif capture_wrapper_timing or capture_sync_timing:
            sync_after_reshape = _sync_perf_counter() if capture_sync_timing else None
            reshape_end_evt = _record_event()
        # pooling model branch
        if attn_metadata.model_runner_type == "pooling":
            attn_output = self._forward_encoder_attention(query, key, value, attn_metadata, output)
            output[:num_tokens] = attn_output[:num_tokens]
            return output
        output = self.forward_impl(query, key, value, kv_cache, attn_metadata, output)
        if capture_wrapper_timing and wrapper_start_evt is not None and reshape_end_evt is not None:
            wrapper_end_evt = _record_event()
            _maybe_log_attention_wrapper_timing(
                getattr(attn_metadata, "attn_state", None),
                num_tokens,
                _elapsed_ms(wrapper_start_evt, reshape_end_evt),
                _elapsed_ms(reshape_end_evt, wrapper_end_evt),
                _elapsed_ms(wrapper_start_evt, wrapper_end_evt),
            )
        if capture_sync_timing and sync_start is not None and sync_after_reshape is not None:
            sync_end = _sync_perf_counter()
            _maybe_log_attention_sync_timing(
                getattr(attn_metadata, "attn_state", None),
                num_tokens,
                (sync_after_reshape - sync_start) * 1000,
                (sync_end - sync_after_reshape) * 1000,
                (sync_end - sync_start) * 1000,
            )
        return output


def unified_ascend_attention_with_output(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    output: torch.Tensor,
    layer_name: str,
    output_scale: torch.Tensor | None = None,
    output_block_scale: torch.Tensor | None = None,
) -> None:
    forward_context: ForwardContext = get_forward_context()
    attn_metadata = forward_context.attn_metadata
    if isinstance(attn_metadata, dict):
        attn_metadata = attn_metadata[layer_name]
    layer = forward_context.no_compile_layers[layer_name]
    kv_cache = layer.kv_cache[forward_context.virtual_engine]
    layer.impl.forward(
        layer,
        query,
        key,
        value,
        kv_cache,
        attn_metadata,
        output=output,
        output_scale=output_scale,
        output_block_scale=output_block_scale,
    )


def unified_ascend_attention_with_output_fake(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    output: torch.Tensor,
    layer_name: str,
    output_scale: torch.Tensor | None = None,
    output_block_scale: torch.Tensor | None = None,
) -> None:
    return


direct_register_custom_op(
    op_name="unified_ascend_attention_with_output",
    op_func=unified_ascend_attention_with_output,
    mutates_args=["output", "output_block_scale"],
    fake_impl=unified_ascend_attention_with_output_fake,
    dispatch_key="PrivateUse1",
)
