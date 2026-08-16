# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import dataclasses
import gc
import os
import weakref
from contextlib import ExitStack
from dataclasses import dataclass
from typing import Any, Callable, Optional
from unittest.mock import patch

import torch
import torch_npu
import vllm.envs as envs
from vllm.compilation.counter import compilation_counter
from vllm.compilation.cuda_graph import CUDAGraphOptions
from vllm.compilation.monitor import validate_cudagraph_capturing_enabled
from vllm.config import CUDAGraphMode, VllmConfig
from vllm.forward_context import BatchDescriptor, get_forward_context
from vllm.logger import logger
from vllm.platforms import current_platform

from ..utils import weak_ref_tensors


_ACL_GRAPH_WRAPPERS: "weakref.WeakSet[ACLGraphWrapper]" = weakref.WeakSet()


@dataclasses.dataclass
class ACLGraphEntry:
    batch_descriptor: BatchDescriptor
    aclgraph: Optional[torch.npu.NPUGraph] = None
    output: Optional[Any] = None

    # for aclgraph debugging, track the input addresses
    # during capture, and check if they are the same during replay
    input_addresses: Optional[list[int]] = None
    input_tensors: Optional[list[torch.Tensor]] = None
    attention_task_range: Optional[tuple[int, int]] = None


class ACLGraphWrapper:
    """Wraps a runnable to add acl graph capturing and replaying ability. And
    provide attribute access to the underlying `runnable` via `__getattr__`.

    The workflow of this wrapper in the aclgraph dispatching is as follows:
    1. At initialization, a runtime mode is assigned to the wrapper (FULL or
    PIECEWISE).
    2. At runtime, the wrapper receives a runtime_mode and a
    batch_descriptor(key) from the forward context and blindly trust them
    for aclgraph dispatching.
    3. If runtime_mode is NONE or runtime_mode does not match the mode of the
    wrapper, just call the runnable directly.
    4. Otherwise, i.e., the runtime_mode matches the mode of the wrapper,
    the wrapper will perform aclgraph capture(if key does not exist, create
    a new entry and cache it) or replay (if key exists in the cache).

    In the elastic PIECEWISE path, split operators may return fresh activation
    buffers on every invocation. Those inputs are copied into the compatible
    tensors whose addresses were captured by NPUGraph. Other paths retain the
    upstream address-consistency contract.
    """

    def __init__(self,
                 runnable: Callable,
                 vllm_config: VllmConfig,
                 runtime_mode: CUDAGraphMode,
                 graph_pool: Any = None,
                 cudagraph_options: Optional[CUDAGraphOptions] = None):
        self.runnable = runnable
        self.vllm_config = vllm_config
        self.graph_pool = graph_pool
        self.runtime_mode = runtime_mode
        self.compilation_config = vllm_config.compilation_config

        self.first_run_finished = False
        self.is_debugging_mode = envs.VLLM_LOGGING_LEVEL == "DEBUG"
        self._cache_generation = 0
        self._logged_replay_generation = -1
        self._elastic_strong_outputs = (
            os.getenv("VLLM_ASCEND_ELASTIC_ACLGRAPH_STRONG_OUTPUTS", "0") == "1")

        # assert runtime_mode is not NONE(no aclgraph), otherwise, we don't
        # need to initialize a ACLGraphWrapper.
        assert self.runtime_mode != CUDAGraphMode.NONE
        if self.graph_pool is None:
            self.graph_pool = current_platform.get_global_graph_pool()

        if cudagraph_options is None:
            cudagraph_options = CUDAGraphOptions()
        self.aclgraph_options = cudagraph_options
        # the entries for different batch descriptors that we need to capture
        # aclgraphs for.
        self.concrete_aclgraph_entries: dict[BatchDescriptor, ACLGraphEntry]\
                                                                        = {}
        _ACL_GRAPH_WRAPPERS.add(self)

    def __getattr__(self, key: str):
        # allow accessing the attributes of the runnable.
        if hasattr(self.runnable, key):
            return getattr(self.runnable, key)
        raise AttributeError(f"Attribute {key} not exists in the runnable of "
                             f"aclgraph wrapper: {self.runnable}")

    def unwrap(self) -> Callable:
        # in case we need to access the original runnable.
        return self.runnable

    def clear_aclgraph_cache(self) -> int:
        """Drop graphs that may retain stale weight or communicator addresses."""
        num_entries = len(self.concrete_aclgraph_entries)
        for entry in self.concrete_aclgraph_entries.values():
            aclgraph = entry.aclgraph
            if aclgraph is not None and hasattr(aclgraph, "reset"):
                aclgraph.reset()
            entry.aclgraph = None
            entry.output = None
            entry.input_addresses = None
            entry.input_tensors = None
            entry.attention_task_range = None
        self.concrete_aclgraph_entries.clear()
        self.first_run_finished = False
        if num_entries:
            self._cache_generation = getattr(self, "_cache_generation", 0) + 1
            self._logged_replay_generation = -1
        return num_entries

    def __call__(self, *args, **kwargs):
        forward_context = get_forward_context()
        batch_descriptor = forward_context.batch_descriptor
        aclgraph_runtime_mode = forward_context.cudagraph_runtime_mode

        if aclgraph_runtime_mode == CUDAGraphMode.NONE or \
                            aclgraph_runtime_mode != self.runtime_mode:
            # CUDAGraphMode.NONE could mean the profile run, a warmup run, or
            # running without aclgraphs.
            # We do not trigger capture/replay if the runtime mode is not
            # matches. This enables properly dispatching to the correct
            # CUDAGraphWrapper when nesting multiple instances with different
            # runtime modes.
            return self.runnable(*args, **kwargs)

        if batch_descriptor not in self.concrete_aclgraph_entries:
            # create a new entry for this batch descriptor
            self.concrete_aclgraph_entries[batch_descriptor] = \
                ACLGraphEntry(batch_descriptor=batch_descriptor)

        entry = self.concrete_aclgraph_entries[batch_descriptor]

        if entry.aclgraph is None:
            if self.aclgraph_options.debug_log_enable:
                # Since we capture aclgraph for many different shapes and
                # capturing is fast, we don't need to log it for every
                # shape. E.g. we only log it for the first subgraph in
                # piecewise mode.
                logger.debug("Capturing a aclgraph on (%s,%s)",
                             self.runtime_mode.name, entry.batch_descriptor)
            # validate that aclgraph capturing is legal at this point.
            validate_cudagraph_capturing_enabled()

            input_addresses = [
                x.data_ptr() for x in args if isinstance(x, torch.Tensor)
            ]
            entry.input_addresses = input_addresses
            if (os.getenv("VLLM_ASCEND_ELASTIC_ACLGRAPH", "0") == "1"
                    and self.runtime_mode == CUDAGraphMode.PIECEWISE):
                # A PIECEWISE split op can allocate a fresh activation on every
                # invocation. NPUGraph replays the addresses seen at capture,
                # so keep those tensors alive and refresh only inputs whose
                # runtime address changed. Static weights retain their address
                # and incur no copy.
                entry.input_tensors = [
                    x for x in args if isinstance(x, torch.Tensor)
                ]
            aclgraph = torch.npu.NPUGraph()
            graph_params = get_graph_params()
            task_start = (len(graph_params.attn_params[
                batch_descriptor.num_tokens]) if graph_params is not None else 0)

            with ExitStack() as stack:
                if self.aclgraph_options.gc_disable:
                    # during every model forward for piecewise aclgraph
                    # mode, we will capture many pieces of aclgraphs
                    # (roughly one per layer). running gc again and again
                    # across layers will make the aclgraph capture very slow.
                    # therefore, we only run gc for the first graph,
                    # and disable gc for the rest of the graphs.
                    stack.enter_context(patch("gc.collect", lambda: None))
                    stack.enter_context(
                        patch("torch.npu.empty_cache", lambda: None))

                # mind-exploding: carefully manage the reference and memory.
                forward_context.capturing = True
                with torch.npu.graph(aclgraph, pool=self.graph_pool):
                    # `output` is managed by pytorch's aclgraph pool
                    output = self.runnable(*args, **kwargs)
                    if (self.aclgraph_options.weak_ref_output
                            and not self._elastic_strong_outputs):
                        # by converting it to weak ref,
                        # the original `output` will immediately be released
                        # to save memory. It is only safe to do this for
                        # the last graph in piecewise aclgraph mode, because
                        # the output of the last graph will not be used by
                        # any other acl graph.
                        output = weak_ref_tensors(output)

            # here we always use weak ref for the output
            # to save memory
            entry.output = (output if self._elastic_strong_outputs else
                            weak_ref_tensors(output))
            entry.aclgraph = aclgraph
            if (graph_params is not None
                    and self.runtime_mode == CUDAGraphMode.PIECEWISE):
                task_end = len(
                    graph_params.attn_params[batch_descriptor.num_tokens])
                if task_end > task_start:
                    entry.attention_task_range = (task_start, task_end)

            compilation_counter.num_cudagraph_captured += 1

            # important: we need to return the output, rather than
            # the weak ref of the output, so that pytorch can correctly
            # manage the memory during acl graph capture
            return output

        runtime_tensors = [x for x in args if isinstance(x, torch.Tensor)]
        if entry.input_tensors is not None:
            if len(runtime_tensors) != len(entry.input_tensors):
                raise RuntimeError(
                    "Elastic ACLGraph tensor input count changed between "
                    f"capture and replay: captured={len(entry.input_tensors)} "
                    f"runtime={len(runtime_tensors)}")
            changed_inputs = []
            for index, (captured, runtime) in enumerate(
                    zip(entry.input_tensors, runtime_tensors, strict=True)):
                if captured.data_ptr() == runtime.data_ptr():
                    continue
                if (captured.shape != runtime.shape
                        or captured.stride() != runtime.stride()
                        or captured.dtype != runtime.dtype
                        or captured.device != runtime.device):
                    raise RuntimeError(
                        "Elastic ACLGraph input changed incompatibly at "
                        f"tensor {index}: captured shape={tuple(captured.shape)} "
                        f"stride={captured.stride()} dtype={captured.dtype} "
                        f"device={captured.device}; runtime "
                        f"shape={tuple(runtime.shape)} stride={runtime.stride()} "
                        f"dtype={runtime.dtype} device={runtime.device}")
                captured.copy_(runtime)
                changed_inputs.append(index)
            if (changed_inputs and os.getenv(
                    "VLLM_ASCEND_ELASTIC_ACLGRAPH_DEBUG_INPUTS", "0") == "1"):
                graph_index = getattr(self.runnable,
                                      "piecewise_compile_index", "unknown")
                logger.warning_once(
                    "Elastic ACLGraph refreshed dynamic piecewise inputs: "
                    "graph_index=%s tensor_indices=%s",
                    graph_index,
                    tuple(changed_inputs),
                )

        if (self.runtime_mode == CUDAGraphMode.FULL
                and entry.input_tensors is None):
            new_input_addresses = [x.data_ptr() for x in runtime_tensors]
            if new_input_addresses != entry.input_addresses:
                raise RuntimeError(
                    "FULL_DECODE_ONLY requires stable model-input addresses; "
                    f"captured={entry.input_addresses}, "
                    f"runtime={new_input_addresses}")

        if self.is_debugging_mode:
            # check if the input addresses are the same
            new_input_addresses = [x.data_ptr() for x in runtime_tensors]
            expected_addresses = (
                [x.data_ptr() for x in entry.input_tensors]
                if entry.input_tensors is not None else entry.input_addresses)
            assert (entry.input_tensors is not None
                    or new_input_addresses == expected_addresses), (
                f"Input addresses for aclgraphs are different "
                f"during replay. Expected {expected_addresses}, "
                f"got {new_input_addresses}")

        if (os.getenv("VLLM_ASCEND_ELASTIC_ACLGRAPH_CAPTURE_MOE", "0") == "1"
                and self._logged_replay_generation != self._cache_generation):
            logger.info(
                "Elastic full-MoE ACLGraph replay: generation=%s "
                "batch_descriptor=%s",
                self._cache_generation,
                batch_descriptor,
            )
            self._logged_replay_generation = self._cache_generation
        else:
            logger.info_once("Replaying aclgraph")
        # Match vLLM-Ascend 0.14 PIECEWISE replay ordering. Runtime input
        # buffers and KV metadata can be updated on streams other than the one
        # used by NPUGraph replay. Finish the previous replay and all live-input
        # updates before starting this replay, even when Attention itself is a
        # splitting op outside the graph.
        torch.npu.synchronize()
        entry.aclgraph.replay()
        if os.getenv("VLLM_ASCEND_ELASTIC_ACLGRAPH_DEBUG_POST_SYNC", "0") == "1":
            torch.npu.synchronize()
        if (entry.attention_task_range is not None
                and self.runtime_mode == CUDAGraphMode.PIECEWISE):
            graph_params = get_graph_params()
            if graph_params is None:
                raise RuntimeError(
                    "Attention ACLGraph parameters disappeared before replay")
            update_attn_params(
                graph_params.update_stream,
                forward_context,
                batch_descriptor.num_tokens,
                task_range=entry.attention_task_range,
            )
        return entry.output


def update_attn_params(update_stream,
                       forward_context,
                       runtime_shape,
                       task_range: Optional[tuple[int, int]] = None):
    graph_params = get_graph_params()
    if graph_params is None:
        raise RuntimeError("Attention ACLGraph parameters are not initialized")
    metadata = forward_context.attn_metadata
    if not isinstance(metadata, dict):
        raise RuntimeError(
            "Attention ACLGraph requires per-layer metadata dict, got "
            f"{type(metadata).__name__}")
    if runtime_shape not in graph_params.attn_params:
        raise RuntimeError(
            f"Attention ACLGraph has no captured shape {runtime_shape}")
    params = graph_params.attn_params[runtime_shape]
    handles = graph_params.handles[runtime_shape]
    events = graph_params.events[runtime_shape]
    counts = (len(params), len(handles), len(events))
    if len(set(counts)) != 1:
        raise RuntimeError(
            "Attention ACLGraph layer/task mismatch: "
            f"metadata={len(metadata)} params={counts[0]} handles={counts[1]} "
            f"events={counts[2]} shape={runtime_shape}")
    if task_range is not None:
        start, end = task_range
        if not 0 <= start < end <= counts[0]:
            raise RuntimeError(
                f"Attention ACLGraph invalid task range {task_range} for "
                f"shape={runtime_shape} task_count={counts[0]}")
        params = params[start:end]
        handles = handles[start:end]
        events = events[start:end]
    expected = len(params)

    attention_backends: set[str] = set()
    with torch.npu.stream(update_stream):
        for param, handle, event in zip(params, handles, events, strict=True):
            attention_backend = (param[0] if param[0] in {"fia", "pa"}
                                 else "pa")
            attention_backends.add(attention_backend)
            offset = 1 if param[0] in {"fia", "pa"} else 0
            captured_layer_name = param[offset]
            if captured_layer_name is None:
                raise RuntimeError(
                    "Attention ACLGraph capture omitted the layer identity")
            if captured_layer_name not in metadata:
                raise RuntimeError(
                    "Attention ACLGraph captured layer is absent from live metadata: "
                    f"{captured_layer_name!r}")
            runtime_metadata = metadata[captured_layer_name]
            layer = forward_context.no_compile_layers.get(captured_layer_name)
            if layer is None:
                raise RuntimeError(
                    f"Attention ACLGraph layer {captured_layer_name!r} is absent from "
                    "static_forward_context")
            live_kv_cache = layer.kv_cache[forward_context.virtual_engine]
            if len(live_kv_cache) < 2:
                raise RuntimeError(
                    "Attention ACLGraph layer has no KV cache: "
                    f"{captured_layer_name!r}")
            key_cache, value_cache = live_kv_cache[:2]
            block_table = runtime_metadata.block_tables
            if attention_backend == "fia":
                (
                    _backend,
                    _layer_name,
                    query,
                    captured_key_cache,
                    captured_value_cache,
                    captured_block_table,
                    attn_mask,
                    sparse_mode,
                    block_size,
                    _captured_seq_lens,
                    _captured_actual_seq_lengths_q,
                    num_kv_heads,
                    num_heads,
                    scale,
                    output,
                    softmax_lse,
                ) = param
                num_blocks = key_cache.shape[0]
                live_key_cache = key_cache.view(num_blocks, block_size, -1)
                live_value_cache = value_cache.view(num_blocks, block_size, -1)
                if live_key_cache.data_ptr() != captured_key_cache.data_ptr():
                    raise RuntimeError(
                        "Attention ACLGraph stale FIA key-cache address for "
                        f"{captured_layer_name!r}")
                if live_value_cache.data_ptr() != captured_value_cache.data_ptr():
                    raise RuntimeError(
                        "Attention ACLGraph stale FIA value-cache address for "
                        f"{captured_layer_name!r}")
                if block_table.data_ptr() != captured_block_table.data_ptr():
                    raise RuntimeError(
                        "Attention ACLGraph stale FIA block-table address for "
                        f"{captured_layer_name!r}")
                runtime_sparse_mode = (
                    3 if runtime_metadata.attn_mask is not None else 0)
                if runtime_sparse_mode != sparse_mode:
                    raise RuntimeError(
                        "Attention ACLGraph FIA mask mode changed after capture: "
                        f"layer={captured_layer_name!r} "
                        f"captured_sparse_mode={sparse_mode} "
                        f"runtime_sparse_mode={runtime_sparse_mode}")
                from vllm_ascend.attention.attention_v1 import \
                    _pad_attention_seq_params
                actual_seq_lengths_q, seq_lens = _pad_attention_seq_params(
                    list(runtime_metadata.actual_seq_lengths_q or []),
                    list(runtime_metadata.seq_lens_list or []),
                    runtime_shape,
                )
                workspace = graph_params.workspaces.get(runtime_shape)
                if workspace is None:
                    raise RuntimeError(
                        "FULL_DECODE_ONLY FIA max workspace disappeared before "
                        f"replay for shape={runtime_shape}")
                torch.npu.graph_task_update_begin(update_stream, handle)
                torch_npu.npu_fused_infer_attention_score.out(
                    query=query,
                    key=live_key_cache,
                    value=live_value_cache,
                    atten_mask=attn_mask,
                    block_table=block_table,
                    input_layout="TND",
                    block_size=block_size,
                    actual_seq_lengths=actual_seq_lengths_q,
                    actual_seq_lengths_kv=seq_lens,
                    num_key_value_heads=num_kv_heads,
                    num_heads=num_heads,
                    scale=scale,
                    sparse_mode=sparse_mode,
                    workspace=workspace,
                    out=[output, softmax_lse],
                )
                torch.npu.graph_task_update_end(update_stream)
            else:
                pa_param = param[offset:]
                (
                    _layer_name,
                    query,
                    captured_key_cache,
                    captured_value_cache,
                    num_kv_heads,
                    num_heads,
                    scale,
                    captured_block_table,
                    _captured_seq_lens,
                    output,
                ) = pa_param
                seq_lens = runtime_metadata.seq_lens
                if key_cache.data_ptr() != captured_key_cache.data_ptr():
                    raise RuntimeError(
                        "Attention ACLGraph stale key-cache address for "
                        f"{captured_layer_name!r}")
                if value_cache.data_ptr() != captured_value_cache.data_ptr():
                    raise RuntimeError(
                        "Attention ACLGraph stale value-cache address for "
                        f"{captured_layer_name!r}")
                if block_table.data_ptr() != captured_block_table.data_ptr():
                    raise RuntimeError(
                        "Attention ACLGraph stale block-table address for "
                        f"{captured_layer_name!r}")
                # PA does not expose a max-workspace API in this 0.11 stack,
                # so retain the upstream fallback for PIECEWISE diagnostics.
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
    logger.info_once(
        "Attention ACLGraph metadata update active: shape=%s layers=%s "
        "backend=%s",
        runtime_shape,
        expected,
        ",".join(sorted(attention_backends)),
    )


@dataclass
class GraphParams:
    events: dict[int, list[torch.npu.ExternalEvent]]
    workspaces: dict[int, torch.Tensor]
    handles: dict[int, list[torch_npu._C._NPUTaskGroupHandle]]
    attn_params: dict[int, list[tuple]]
    update_stream: torch.npu.Stream


_graph_params: Optional[GraphParams] = None


def set_graph_params(aclgraph_capture_sizes: set[int],
                     update_stream: torch.npu.Stream):
    global _graph_params
    if _graph_params is not None:
        raise ValueError("Graph parameters have already been set!")
    _graph_params = GraphParams(
        {size: []
         for size in aclgraph_capture_sizes},
        {size: None
         for size in aclgraph_capture_sizes},
        {size: []
         for size in aclgraph_capture_sizes},
        {size: []
         for size in aclgraph_capture_sizes},
        update_stream,
    )


def update_graph_params_workspace(num_tokens: int,
                                  workspace: torch.Tensor) -> None:
    graph_params = get_graph_params()
    if graph_params is None:
        raise RuntimeError("Attention ACLGraph parameters are not initialized")
    graph_params.workspaces[num_tokens] = workspace


def get_graph_params():
    return _graph_params


def reset_graph_params_runtime_state() -> None:
    """Release task-update state owned by previously captured full graphs."""
    if _graph_params is None:
        return
    for values in _graph_params.events.values():
        values.clear()
    for values in _graph_params.handles.values():
        values.clear()
    for values in _graph_params.attn_params.values():
        values.clear()
    for num_tokens in _graph_params.workspaces:
        _graph_params.workspaces[num_tokens] = None


def disable_aclgraph_dispatch(model_runner: Any) -> int:
    """Fail closed until ``capture_model`` rebuilds the valid graph keys."""
    dispatcher = getattr(model_runner, "aclgraph_dispatcher", None)
    if dispatcher is None:
        return 0
    dispatcher.keys_initialized = False
    cleared_keys = 0
    for keys in getattr(dispatcher, "cudagraph_keys", {}).values():
        cleared_keys += len(keys)
        keys.clear()
    return cleared_keys


def renew_aclgraph_pool() -> int:
    """Move future captures off a pool that may retain collective graph state.

    Resetting an ``NPUGraph`` releases the graph itself, but the torch_npu 2.5
    runtime can keep a pool reference alive when the graph contains HCCL nodes.
    Reusing that token then fails capture with an internal ``use_count``
    assertion. A fresh pool is therefore part of topology and weight
    invalidation, not an optional memory optimization.
    """
    platform_cls = current_platform.__class__
    platform_cls._global_graph_pool = None
    graph_pool = current_platform.get_global_graph_pool()
    wrappers = list(_ACL_GRAPH_WRAPPERS)
    for wrapper in wrappers:
        wrapper.graph_pool = graph_pool
    return len(wrappers)


def clear_aclgraph_caches(root: Any = None,
                          *,
                          reset_graph_params: bool = True,
                          renew_pool: bool = True) -> int:
    """Release every live ACLGraphWrapper cache reachable in this process.

    Piecewise wrappers can be retained by torch.compile closures instead of
    regular ``nn.Module`` children. The weak registry therefore remains the
    authoritative fallback after walking the supplied model root.
    """
    # Never drop graph objects while queued replay work may still reference
    # their captured addresses. A synchronization failure must abort the
    # lifecycle transition instead of continuing with a partially cleared cache.
    torch.npu.synchronize()

    seen: set[int] = set()
    cleared_wrappers: set[int] = set()
    stack: list[Any] = [] if root is None else [root]
    cleared_entries = 0

    def clear_wrapper(wrapper: ACLGraphWrapper) -> None:
        nonlocal cleared_entries
        wrapper_id = id(wrapper)
        if wrapper_id in cleared_wrappers:
            return
        cleared_wrappers.add(wrapper_id)
        cleared_entries += wrapper.clear_aclgraph_cache()

    while stack:
        obj = stack.pop()
        if obj is None or isinstance(obj, (str, bytes, int, float, bool)):
            continue
        obj_id = id(obj)
        if obj_id in seen:
            continue
        seen.add(obj_id)

        if isinstance(obj, torch.Tensor):
            continue
        if isinstance(obj, ACLGraphWrapper):
            clear_wrapper(obj)
            stack.append(obj.runnable)
            continue
        if isinstance(obj, dict):
            stack.extend(obj.values())
            continue
        if isinstance(obj, (list, tuple, set)):
            stack.extend(obj)
            continue

        values: list[Any] = []
        if isinstance(obj, torch.nn.Module):
            values.extend(obj._modules.values())
            values.extend(vars(obj).values())
        elif hasattr(obj, "__dict__"):
            module_name = getattr(obj.__class__, "__module__", "")
            if module_name.startswith(("vllm", "vllm_ascend")):
                values.extend(vars(obj).values())
        stack.extend(values)

    for wrapper in list(_ACL_GRAPH_WRAPPERS):
        clear_wrapper(wrapper)

    if reset_graph_params:
        reset_graph_params_runtime_state()
    # Make every local graph object and captured output collectible before the
    # shared pool token is replaced. This ordering is required for graphs that
    # own HCCL task state on the pinned torch_npu runtime.
    gc.collect()
    if renew_pool:
        renew_aclgraph_pool()
    torch.npu.empty_cache()
    torch.npu.synchronize()
    return cleared_entries
