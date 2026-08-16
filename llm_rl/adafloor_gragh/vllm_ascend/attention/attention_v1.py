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

from dataclasses import dataclass
from enum import Enum
import os
from typing import ClassVar, List, Optional, Tuple, Type

import torch
import torch.nn as nn
import torch_npu
from vllm.attention.backends.abstract import (AttentionBackend, AttentionImpl,
                                              AttentionLayer, AttentionType)
from vllm.compilation.cuda_graph import CUDAGraphOptions
from vllm.config import CUDAGraphMode, VllmConfig, get_current_vllm_config
from vllm.forward_context import ForwardContext, get_forward_context
from vllm.logger import logger
from vllm.utils import cdiv, direct_register_custom_op
from vllm.v1.attention.backends.utils import AttentionCGSupport
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_cache_interface import AttentionSpec

from vllm_ascend.attention.utils import (AscendCommonAttentionMetadata,
                                         maybe_save_kv_layer_to_connector,
                                         wait_for_kv_layer_from_connector)
from vllm_ascend.compilation.acl_graph import (ACLGraphWrapper,
                                               get_graph_params,
                                               update_graph_params_workspace)
from vllm_ascend.ops.attention import vanilla_chunked_prefill
from vllm_ascend.utils import (ACL_FORMAT_FRACTAL_NZ, aligned_16, is_310p,
                               nd_to_nz_2d, nd_to_nz_spec, weak_ref_tensors)


_FULL_DECODE_ATTENTION_BACKEND = os.getenv(
    "VLLM_ASCEND_FULL_DECODE_ATTENTION_BACKEND",
    "fia_max_workspace",
).lower()
_FULL_DECODE_ATTENTION_BACKENDS = {"fia_max_workspace", "paged_attention"}


def _pad_attention_seq_params(
    actual_seq_lengths_q: List[int],
    seq_lens: List[int],
    runtime_shape: int,
) -> tuple[List[int], List[int]]:
    """Pad host sequence metadata to a captured uniform-decode shape."""
    if not actual_seq_lengths_q:
        padded_actual_seq_lengths_q = [runtime_shape]
    else:
        last_value = actual_seq_lengths_q[-1]
        if last_value >= runtime_shape:
            padded_actual_seq_lengths_q = actual_seq_lengths_q
        else:
            padded_actual_seq_lengths_q = actual_seq_lengths_q + list(
                range(last_value + 1, runtime_shape + 1))

    target_length = len(padded_actual_seq_lengths_q)
    if len(seq_lens) >= target_length:
        padded_seq_lens = seq_lens
    else:
        padded_seq_lens = seq_lens + [0] * (target_length - len(seq_lens))
    return padded_actual_seq_lengths_q, padded_seq_lens


class AscendAttentionBackend(AttentionBackend):
    accept_output_buffer: bool = True

    @staticmethod
    def get_name() -> str:
        return "ASCEND"

    @staticmethod
    def get_impl_cls() -> Type["AscendAttentionBackendImpl"]:
        return AscendAttentionBackendImpl

    @staticmethod
    def get_metadata_cls() -> Type["AscendMetadata"]:
        return AscendMetadata

    @staticmethod
    def get_builder_cls() -> type["AscendAttentionMetadataBuilder"]:
        return AscendAttentionMetadataBuilder

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
    ) -> Tuple[int, ...]:
        if is_310p():
            return (2, num_blocks, num_kv_heads * head_size // 16, block_size,
                    16)
        return (2, num_blocks, block_size, num_kv_heads, head_size)

    @staticmethod
    def get_bsh_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
    ) -> Tuple[int, ...]:
        return (2, num_blocks, block_size, num_kv_heads * head_size)

    @staticmethod
    def swap_blocks(
        src_kv_cache: List[torch.Tensor],
        dst_kv_cache: List[torch.Tensor],
        src_to_dst: torch.Tensor,
    ) -> None:
        src_key_cache, src_value_cache = src_kv_cache[0], src_kv_cache[1]
        dst_key_cache, dst_value_cache = dst_kv_cache[0], dst_kv_cache[1]
        src_indices = src_to_dst[:, 0]
        dst_indices = src_to_dst[:, 1]

        dst_key_cache[dst_indices] = src_key_cache[src_indices].to(
            dst_key_cache.device)
        dst_value_cache[dst_indices] = src_value_cache[src_indices].to(
            dst_key_cache.device)

    @staticmethod
    def copy_blocks(
        kv_caches: List[torch.Tensor],
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
        return [64]


class AscendAttentionState(Enum):
    PrefillNoCache = 0
    PrefillCacheHit = 1
    DecodeOnly = 2
    ChunkedPrefill = 3
    SpecDecoding = 4


@dataclass
class AscendMetadata:

    # **************************** Basic Properties ************************** #
    attn_mask: Optional[torch.Tensor] = None
    # Current state of this attention run.
    attn_state: AscendAttentionState = AscendAttentionState.ChunkedPrefill

    # Number of tokens excluding padding.
    num_actual_tokens: int = 0

    # The sequence length per sequence. Sequence length means the computed
    # tokens + new tokens (is None if it is a decoding).
    # (batch_size,)
    seq_lens: torch.Tensor = None
    # Host-side values used by FULL_DECODE_ONLY graph-task updates. Keeping
    # these lists in the metadata builder avoids an NPU-to-host sync in every
    # Attention layer during decode replay.
    seq_lens_list: Optional[List[int]] = None
    actual_seq_lengths_q: Optional[List[int]] = None

    query_start_loc: torch.Tensor = None
    query_lens: torch.Tensor = None
    # Maximum query length in the batch (None for decoding).
    max_query_len: Optional[int] = None

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

    # *************************** Other Properties *************************** #
    enable_dbo_across_dp: bool = False


class AscendAttentionMetadataBuilder:
    # Does this backend/builder support ACL Graphs for attention (default: no).
    aclgraph_support: ClassVar[AttentionCGSupport] = \
        AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE
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
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.device = device
        self.max_num_blocks_per_req = cdiv(
            self.model_config.max_model_len,
            AscendAttentionBackend.get_supported_block_size()[0])

    def reorder_batch(self, input_batch,
                      scheduler_output: "SchedulerOutput") -> bool:
        return False

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: AscendCommonAttentionMetadata,
        model: Optional[nn.Module] = None,
    ):
        num_reqs = common_attn_metadata.num_reqs
        num_actual_tokens = common_attn_metadata.num_actual_tokens
        query_start_loc_cpu = common_attn_metadata.query_start_loc_cpu[:
                                                                       num_reqs
                                                                       + 1]
        block_table = common_attn_metadata.block_table_tensor
        query_lens = query_start_loc_cpu[1:] - query_start_loc_cpu[:-1]
        seq_lens = common_attn_metadata.seq_lens_cpu[:num_reqs]
        seq_lens_list = seq_lens.tolist()
        actual_seq_lengths_q = query_start_loc_cpu[1:].tolist()
        slot_mapping = common_attn_metadata.slot_mapping[:num_actual_tokens]
        attn_mask = common_attn_metadata.attn_mask
        attn_state = common_attn_metadata.attn_state
        query_start_loc_cpu = common_attn_metadata.query_start_loc_cpu[:
                                                                       num_reqs
                                                                       + 1]
        query_start_loc = query_start_loc_cpu.to(self.device,
                                                 non_blocking=True)

        if is_310p():
            if attn_state == AscendAttentionState.PrefillNoCache:
                mask_nz = nd_to_nz_2d(attn_mask)
                attn_mask = torch_npu.npu_format_cast(mask_nz.contiguous(),
                                                      ACL_FORMAT_FRACTAL_NZ)
            elif attn_state == AscendAttentionState.ChunkedPrefill:
                mask_nz = nd_to_nz_spec(attn_mask)
                attn_mask = torch_npu.npu_format_cast(mask_nz.contiguous(),
                                                      ACL_FORMAT_FRACTAL_NZ)

        attn_metadata = AscendMetadata(
            num_actual_tokens=num_actual_tokens,
            block_tables=block_table,
            query_start_loc=query_start_loc,
            query_lens=query_lens,
            seq_lens=seq_lens,
            seq_lens_list=seq_lens_list,
            actual_seq_lengths_q=actual_seq_lengths_q,
            max_query_len=common_attn_metadata.max_query_len,
            slot_mapping=slot_mapping,
            attn_mask=attn_mask,
            attn_state=attn_state,
            enable_dbo_across_dp=common_attn_metadata.enable_dbo_across_dp)
        return attn_metadata

    def build_for_graph_capture(
        self,
        common_attn_metadata: AscendCommonAttentionMetadata,
        attn_state: AscendAttentionState = AscendAttentionState.DecodeOnly,
    ):
        if attn_state == AscendAttentionState.DecodeOnly:
            attn_metadata = self.build(
                common_prefix_len=0,
                common_attn_metadata=common_attn_metadata,
            )
        else:
            raise NotImplementedError(
                "Currently we only support building dummy metadata for DecodeOnly state"
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
        alibi_slopes: Optional[List[float]],
        sliding_window: Optional[int],
        kv_cache_dtype: str,
        logits_soft_cap: Optional[float],
        attn_type: str,
        kv_sharing_target_layer_name: Optional[str],
        **kwargs,
    ) -> None:
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_heads if num_kv_heads is None else num_kv_heads
        self.hidden_size = self.num_heads * self.head_size
        self.kv_cache_dtype = kv_cache_dtype
        self.sliding_window = sliding_window
        if alibi_slopes is not None:
            alibi_slopes = torch.tensor(alibi_slopes,
                                        dtype=torch.float32,
                                        device="npu")
        self.alibi_slopes = alibi_slopes
        self.attn_type = attn_type

        assert self.num_heads % self.num_kv_heads == 0
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads
        self.key_cache = None
        self.value_cache = None
        self._elastic_pa_graph: Optional[ACLGraphWrapper] = None
        self._elastic_pa_buffers: dict[int, tuple[torch.Tensor,
                                                  torch.Tensor]] = {}
        self._vllm_config = get_current_vllm_config()

    def _forward_prefill_no_cache(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AscendMetadata,
        output: Optional[torch.Tensor] = None,
        num_tokens=0,
    ) -> torch.Tensor:
        assert attn_metadata is not None
        assert attn_metadata.attn_mask is not None

        mask = attn_metadata.attn_mask

        if is_310p():
            # align q k v output tensors
            query = aligned_16(query)
            key = aligned_16(key)
            value = aligned_16(value)
            output = aligned_16(output)
            # do reformat in case of broadcasted tensors
            mask = mask.repeat(attn_metadata.seq_lens.size(0), 1, 1, 1)
            mask = torch_npu.npu_format_cast(mask.contiguous(),
                                             ACL_FORMAT_FRACTAL_NZ)

        torch_npu._npu_flash_attention(query=query,
                                       key=key,
                                       value=value,
                                       mask=mask,
                                       seq_len=attn_metadata.seq_lens,
                                       scale_value=self.scale,
                                       num_heads=self.num_heads,
                                       num_kv_heads=self.num_kv_heads,
                                       out=output)
        assert output is not None
        return output[:num_tokens, :, :]

    def _forward_prefill_cache_hit(
        self,
        query: torch.Tensor,
        attn_metadata: AscendMetadata,
        output: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        assert attn_metadata is not None
        assert attn_metadata.attn_mask is not None

        compress_mask = attn_metadata.attn_mask
        batch_size = attn_metadata.query_lens.shape[0]
        block_table = attn_metadata.block_tables[:batch_size, :]

        torch_npu._npu_flash_attention_qlens(
            query=query,
            key_cache=self.key_cache,
            value_cache=self.value_cache,
            block_table=block_table,
            mask=compress_mask,
            seq_len=attn_metadata.query_lens,
            context_lens=attn_metadata.seq_lens,
            num_kv_heads=self.num_kv_heads,
            num_heads=self.num_heads,
            scale_value=self.scale,
            out=output)
        return output

    def _forward_decode_only(
        self,
        query: torch.Tensor,
        attn_metadata: AscendMetadata,
        output: Optional[torch.Tensor] = None,
        layer_name: Optional[str] = None,
    ) -> torch.Tensor:
        forward_context = get_forward_context()
        if (_FULL_DECODE_ATTENTION_BACKEND not in
                _FULL_DECODE_ATTENTION_BACKENDS):
            raise RuntimeError(
                "VLLM_ASCEND_FULL_DECODE_ATTENTION_BACKEND must be one of "
                f"{sorted(_FULL_DECODE_ATTENTION_BACKENDS)}, got "
                f"{_FULL_DECODE_ATTENTION_BACKEND!r}")
        if (getattr(forward_context, "capturing", False)
                and forward_context.cudagraph_runtime_mode
                == CUDAGraphMode.FULL
                and _FULL_DECODE_ATTENTION_BACKEND == "fia_max_workspace"):
            return self._forward_decode_only_full_graph_fia(
                query, attn_metadata, output, layer_name)

        if is_310p():
            # seq_lens_tensor needs to be transferred to the device for 310P.
            attn_metadata.seq_lens = \
                attn_metadata.seq_lens.to(device=query.device)
        if self.sliding_window is not None and attn_metadata.seq_lens.shape[
                0] == query.size(0):
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
                actual_seq_lengths_kv=attn_metadata.seq_lens)

            output = output.view(batch_size, self.num_heads, self.head_size)
        else:
            graph_params = get_graph_params()
            forward_context: ForwardContext = get_forward_context()
            num_tokens = query.shape[0]
            if forward_context.capturing:
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
                        out=output)
                    update_graph_params_workspace(num_tokens, workspace)
                    # ``workspace is None`` is true once per capture shape.
                    # Log every recapture so elastic topology transitions are
                    # auditable without producing one line per model layer.
                    logger.info(
                        "FULL_DECODE_ONLY Attention maximum workspace "
                        "captured: seq_len_bucket=%s bytes=%s shape=%s",
                        int(attn_metadata.seq_lens.max().item()),
                        workspace.numel() * workspace.element_size(),
                        num_tokens,
                    )
                stream = torch_npu.npu.current_stream()

                event = torch.npu.ExternalEvent()
                event.wait(stream)
                event.reset(stream)
                graph_params.events[num_tokens].append(event)

                graph_params.attn_params[num_tokens].append((
                    "pa",
                    layer_name,
                    weak_ref_tensors(query),
                    weak_ref_tensors(self.key_cache),
                    weak_ref_tensors(self.value_cache),
                    self.num_kv_heads,
                    self.num_heads,
                    self.scale,
                    weak_ref_tensors(attn_metadata.block_tables),
                    attn_metadata.seq_lens,
                    weak_ref_tensors(output),
                ))

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
                    workspace=workspace)
                handle = torch.npu.graph_task_group_end(stream)
                graph_params.handles[num_tokens].append(handle)
            else:
                torch_npu._npu_paged_attention(
                    query=query,
                    key_cache=self.key_cache,
                    value_cache=self.value_cache,
                    num_kv_heads=self.num_kv_heads,
                    num_heads=self.num_heads,
                    scale_value=self.scale,
                    block_table=attn_metadata.block_tables,
                    context_lens=attn_metadata.seq_lens,
                    out=output)
        return output

    def _forward_decode_only_full_graph_fia(
        self,
        query: torch.Tensor,
        attn_metadata: AscendMetadata,
        output: torch.Tensor,
        layer_name: Optional[str],
    ) -> torch.Tensor:
        """Capture the 0.14 FIA full-decode task with one max workspace.

        The workspace address and size are fixed at capture. Only the host
        sequence-length lists are replaced by the graph-task update protocol
        before the queued outer graph executes. This removes the old 0.11
        path's per-layer, per-token paged-attention workspace query.
        """
        graph_params = get_graph_params()
        if graph_params is None:
            raise RuntimeError(
                "FULL_DECODE_ONLY FIA graph parameters are not initialized")
        if not hasattr(torch_npu,
                       "_npu_fused_infer_attention_score_get_max_workspace"):
            raise RuntimeError(
                "The installed torch_npu lacks the FIA max-workspace API "
                "required by FULL_DECODE_ONLY")
        if not hasattr(torch_npu.npu_fused_infer_attention_score, "out"):
            raise RuntimeError(
                "The installed torch_npu lacks the out= FIA API required by "
                "FULL_DECODE_ONLY")
        if self.key_cache is None or self.value_cache is None:
            raise RuntimeError(
                "FULL_DECODE_ONLY FIA capture requires initialized KV cache")
        if output is None:
            raise RuntimeError(
                "FULL_DECODE_ONLY FIA capture requires a stable output buffer")
        if attn_metadata.block_tables is None:
            raise RuntimeError(
                "FULL_DECODE_ONLY FIA capture requires a stable block table")
        if layer_name is None:
            raise RuntimeError(
                "FULL_DECODE_ONLY FIA capture requires the Attention layer name")

        num_tokens = query.shape[0]
        num_blocks, block_size, _, _ = self.key_cache.shape
        key_cache = self.key_cache.view(num_blocks, block_size, -1)
        value_cache = self.value_cache.view(num_blocks, block_size, -1)
        block_table = attn_metadata.block_tables
        sparse_mode = 3 if attn_metadata.attn_mask is not None else 0
        actual_seq_lengths_q, actual_seq_lengths_kv = (
            _pad_attention_seq_params(
                list(attn_metadata.actual_seq_lengths_q or []),
                list(attn_metadata.seq_lens_list or []),
                num_tokens,
            ))

        workspace = graph_params.workspaces.get(num_tokens)
        softmax_lse = torch.empty(1,
                                  dtype=query.dtype,
                                  device=query.device)
        if workspace is None:
            workspace = (
                torch_npu._npu_fused_infer_attention_score_get_max_workspace(
                    query=query,
                    key=key_cache,
                    value=value_cache,
                    atten_mask=attn_metadata.attn_mask,
                    block_table=block_table,
                    input_layout="TND",
                    block_size=block_size,
                    actual_seq_lengths=actual_seq_lengths_q,
                    actual_seq_lengths_kv=actual_seq_lengths_kv,
                    num_key_value_heads=self.num_kv_heads,
                    num_heads=self.num_heads,
                    sparse_mode=sparse_mode,
                    scale=self.scale,
                ))
            update_graph_params_workspace(num_tokens, workspace)
            max_seq_len = max(actual_seq_lengths_kv, default=0)
            logger.info(
                "FULL_DECODE_ONLY Attention maximum workspace captured: "
                "seq_len_bucket=%s bytes=%s shape=%s backend=fia",
                max_seq_len,
                workspace.numel() * workspace.element_size(),
                num_tokens,
            )

        stream = torch_npu.npu.current_stream()
        event = torch.npu.ExternalEvent()
        event.wait(stream)
        event.reset(stream)
        graph_params.events[num_tokens].append(event)
        graph_params.attn_params[num_tokens].append((
            "fia",
            layer_name,
            weak_ref_tensors(query),
            weak_ref_tensors(key_cache),
            weak_ref_tensors(value_cache),
            weak_ref_tensors(block_table),
            (weak_ref_tensors(attn_metadata.attn_mask)
             if attn_metadata.attn_mask is not None else None),
            sparse_mode,
            block_size,
            actual_seq_lengths_kv,
            actual_seq_lengths_q,
            self.num_kv_heads,
            self.num_heads,
            self.scale,
            weak_ref_tensors(output),
            weak_ref_tensors(softmax_lse),
        ))

        torch.npu.graph_task_group_begin(stream)
        torch_npu.npu_fused_infer_attention_score.out(
            query=query,
            key=key_cache,
            value=value_cache,
            atten_mask=attn_metadata.attn_mask,
            block_table=block_table,
            input_layout="TND",
            block_size=block_size,
            actual_seq_lengths=actual_seq_lengths_q,
            actual_seq_lengths_kv=actual_seq_lengths_kv,
            num_key_value_heads=self.num_kv_heads,
            num_heads=self.num_heads,
            scale=self.scale,
            sparse_mode=sparse_mode,
            workspace=workspace,
            out=[output, softmax_lse],
        )
        handle = torch.npu.graph_task_group_end(stream)
        graph_params.handles[num_tokens].append(handle)
        logger.info_once(
            "FULL_DECODE_ONLY FIA max-workspace Attention capture active")
        return output

    def _forward_decode_only_aclgraph(
        self,
        query: torch.Tensor,
        attn_metadata: AscendMetadata,
        output: torch.Tensor,
        layer_name: Optional[str],
    ) -> torch.Tensor:
        """Capture only the side-effect-free paged-attention read.

        The enclosing unified Attention custom op remains an FX splitting
        boundary. Its KV-cache write therefore runs eagerly with the live slot
        mapping on every token. Static staging buffers give the nested
        paged-attention ACLGraph stable query/output addresses, while the task
        update path supplies live sequence metadata before each replay.
        """
        forward_context = get_forward_context()
        if forward_context.cudagraph_runtime_mode != CUDAGraphMode.PIECEWISE:
            return self._forward_decode_only(query, attn_metadata, output,
                                             layer_name)

        num_tokens = query.shape[0]
        buffers = self._elastic_pa_buffers.get(num_tokens)
        if buffers is None:
            buffers = (torch.empty_like(query), torch.empty_like(output))
            self._elastic_pa_buffers[num_tokens] = buffers
        static_query, static_output = buffers
        if (static_query.shape != query.shape
                or static_query.dtype != query.dtype
                or static_output.shape != output.shape
                or static_output.dtype != output.dtype):
            raise RuntimeError(
                "Attention ACLGraph staging-buffer contract changed for "
                f"shape={num_tokens}")

        if self._elastic_pa_graph is None:
            self._elastic_pa_graph = ACLGraphWrapper(
                self._forward_decode_only,
                self._vllm_config,
                CUDAGraphMode.PIECEWISE,
                cudagraph_options=CUDAGraphOptions(
                    debug_log_enable=False,
                    gc_disable=True,
                    weak_ref_output=False,
                ),
            )

        static_query.copy_(query)
        static_result = self._elastic_pa_graph(static_query, attn_metadata,
                                               static_output, layer_name)
        output.copy_(static_result)
        return output

    def _forward_v1_style(
        self,
        query: torch.Tensor,
        attn_metadata: AscendMetadata,
        output: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Use chunked prefill for head size 192 scenario, like deepseek
        # paged_attention_splitfuse maybe crash at such scenario.
        # TODO: vanilla path will be removed after the kernel support
        # head_size 192 scenario.
        if self.head_size == 192:
            cu_seqlen_q = [0] + attn_metadata.query_lens.tolist()
            cu_seqlen_k = [0] + attn_metadata.seq_lens.tolist()
            cu_seqlen_q = torch.tensor(cu_seqlen_q, device=query.device)
            cu_seqlen_k = torch.tensor(cu_seqlen_k, device=query.device)
            cu_seqlen_q = torch.cumsum(cu_seqlen_q, dim=0)
            cu_seqlen_k = torch.cumsum(cu_seqlen_k, dim=0)
            max_seqlen_q = torch.max(attn_metadata.query_lens)
            max_seqlen_k = torch.max(attn_metadata.seq_lens)
            vanilla_chunked_prefill(output, query, self.key_cache,
                                    self.value_cache,
                                    attn_metadata.block_tables, cu_seqlen_q,
                                    cu_seqlen_k, max_seqlen_q, max_seqlen_k,
                                    self.scale, None, True)
            return output

        # Use paged attention.
        assert attn_metadata is not None
        assert attn_metadata.attn_mask is not None

        if is_310p():
            # Do reformat in case of broadcasted tensors.
            attn_metadata.attn_mask = \
                torch_npu.npu_format_cast(attn_metadata.attn_mask.contiguous(),
                                          ACL_FORMAT_FRACTAL_NZ)
            attn_metadata.seq_lens = \
                attn_metadata.seq_lens.to(device=query.device)

        if int(os.environ.get("USE_FIA_FOR_SPEC_DECODING", "0")):
            # TODO:The npu_fused_infer_attention_score op is planned to
            # be utilized in a wider range in upcoming versions.
            num_block, block_size, _, _ = self.key_cache.shape  # type: ignore
            key = self.key_cache.view(  # type: ignore
                num_block, block_size, -1)
            value = self.value_cache.view(  # type: ignore
                num_block, block_size, -1)

            output, _ = torch_npu.npu_fused_infer_attention_score(
                query=query,
                key=key,
                value=value,
                atten_mask=attn_metadata.attn_mask,
                block_table=attn_metadata.block_tables,
                input_layout="TND",
                block_size=block_size,
                actual_seq_lengths=attn_metadata.query_start_loc[1:],
                actual_seq_lengths_kv=attn_metadata.seq_lens,
                num_key_value_heads=self.num_kv_heads,
                num_heads=self.num_heads,
                scale=self.scale,
                sparse_mode=3,
            )
        else:
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
                out=output)
        return output

    def forward(
        self,
        layer: AttentionLayer,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: Tuple[torch.Tensor],
        attn_metadata: AscendMetadata,
        output: Optional[torch.Tensor] = None,
        trace_flag: bool = True,
        layer_name: Optional[str] = None,
    ) -> torch.Tensor:
        """Forward pass with Ascend attention.
        Args:
            query: shape = [batch_size, seq_len, num_heads * head_size]
            key: shape = [batch_size, seq_len, num_kv_heads * head_size]
            value: shape = [batch_size, seq_len, num_kv_heads * head_size]
            kv_cache: shape = [key_cache, value_cache]
                      key_cache = [num_blocks, block_size,
                                   num_kv_heads, head_size]
                      value_cache = [num_blocks, block_size,
                                     num_kv_heads, head_size]
            attn_metadata: Metadata for attention.
        Returns:
            shape = [batch_size * seq_len, num_heads, head_size]
        """
        num_tokens = query.shape[0]
        use_kv_cache_int8 = len(
            kv_cache) > 0 and kv_cache[0].dtype == torch.int8
        if output is None:
            output = torch.empty(num_tokens,
                                 self.num_heads,
                                 self.head_size,
                                 dtype=query.dtype,
                                 device=query.device)
        ori_output = output
        if trace_flag:
            torch.ops.vllm.unified_ascend_attention_with_output(
                query=query,
                key=key,
                value=value,
                output=output,
                layer_name=layer.layer_name)

        elif hasattr(layer, 'quant_method') and use_kv_cache_int8:
            output = layer.quant_method.apply(layer, query, key, value,
                                              kv_cache, attn_metadata,
                                              self.attn_type, self.scale,
                                              output)

        else:
            if attn_metadata is None:
                return output.view(num_tokens, self.hidden_size)
            num_actual_tokens = attn_metadata.num_actual_tokens
            assert layer._k_scale_float == 1.0 and layer._v_scale_float == 1.0
            attn_type = self.attn_type
            if attn_type != AttentionType.DECODER:
                raise NotImplementedError("Encoder self-attention and "
                                          "encoder/decoder cross-attention "
                                          "are not implemented for "
                                          "PallasAttentionBackendImpl")
            # View q k v to BSH.
            query = query.view(-1, self.num_heads, self.head_size)
            key = key.view(-1, self.num_kv_heads, self.head_size)
            value = value.view(-1, self.num_kv_heads, self.head_size)
            # TODO: Remove this contiguous in the future.
            value = value.contiguous()

            if len(kv_cache) > 1:
                if self.key_cache is None:
                    self.key_cache, self.value_cache = kv_cache[0], kv_cache[1]
                slots = attn_metadata.slot_mapping
                torch_npu._npu_reshape_and_cache(
                    key=key[:num_actual_tokens],
                    value=value[:num_actual_tokens],
                    key_cache=self.key_cache,
                    value_cache=self.value_cache,
                    slot_indices=slots)
                forward_context = get_forward_context()
                if (forward_context.capturing and
                        forward_context.cudagraph_runtime_mode ==
                        CUDAGraphMode.FULL):
                    logger.info_once(
                        "FULL_DECODE_ONLY ACLGraph captured Attention KV write "
                        "and paged read inside the outer model graph")

            # V0-Style scheduler situation.
            if attn_metadata.attn_state == AscendAttentionState.PrefillNoCache:
                output = self._forward_prefill_no_cache(
                    query, key, value, attn_metadata, output, num_tokens)
            elif attn_metadata.attn_state == \
                AscendAttentionState.PrefillCacheHit:
                output = self._forward_prefill_cache_hit(
                    query, attn_metadata, output)
            elif attn_metadata.attn_state == AscendAttentionState.DecodeOnly:
                if os.getenv(
                        "VLLM_ASCEND_ELASTIC_ACLGRAPH_CAPTURE_ATTENTION",
                        "0") == "1":
                    output = self._forward_decode_only_aclgraph(
                        query, attn_metadata, output, layer_name)
                else:
                    output = self._forward_decode_only(
                        query, attn_metadata, output, layer_name)
            # Normal V1 situation.
            else:
                if int(os.environ.get("USE_FIA_FOR_SPEC_DECODING", "0")):
                    # npu_fused_infer_attention_score does not support cases
                    # where query.shape[0] != attn_metadata.query_start_loc[-1].
                    # Thus we need unpad it here.
                    num_tokens = attn_metadata.query_start_loc[-1]
                    query = query[:num_tokens]
                output = self._forward_v1_style(query, attn_metadata, output)

        # to make in-place change to the output tensor
        if hasattr(layer, 'quant_method') and use_kv_cache_int8:
            output = output.view(num_tokens, self.num_heads, self.head_size)
        ori_output[:num_tokens, :, :] = output[:num_tokens, :, :]
        return output.view(num_tokens, self.hidden_size)


def unified_ascend_attention_with_output(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    output: torch.Tensor,
    layer_name: str,
) -> None:
    wait_for_kv_layer_from_connector(layer_name)
    forward_context: ForwardContext = get_forward_context()
    attn_metadata = forward_context.attn_metadata
    if isinstance(attn_metadata, dict):
        attn_metadata = attn_metadata[layer_name]
    self = forward_context.no_compile_layers[layer_name]
    kv_cache = self.kv_cache[forward_context.virtual_engine]
    self.impl.forward(self,
                      query,
                      key,
                      value,
                      kv_cache,
                      attn_metadata,
                      output,
                      trace_flag=False,
                      layer_name=layer_name)
    maybe_save_kv_layer_to_connector(layer_name, kv_cache)
    return


def unified_attention_with_output_fake(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    output: torch.Tensor,
    layer_name: str,
) -> None:
    return


direct_register_custom_op(
    op_name="unified_ascend_attention_with_output",
    op_func=unified_ascend_attention_with_output,
    mutates_args=["output"],
    fake_impl=unified_attention_with_output_fake,
    dispatch_key="PrivateUse1",
)
