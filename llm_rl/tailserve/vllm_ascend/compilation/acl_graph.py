# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import dataclasses
import gc
import os
import weakref
from collections.abc import Callable
from contextlib import ExitStack
from dataclasses import dataclass
from typing import Any
from unittest.mock import patch

import torch
import torch_npu
import vllm.envs as envs
from vllm.compilation.counter import compilation_counter
from vllm.compilation.cuda_graph import CUDAGraphOptions
from vllm.compilation import monitor as cudagraph_monitor
from vllm.compilation.monitor import (set_cudagraph_capturing_enabled,
                                      validate_cudagraph_capturing_enabled)
from vllm.config import CUDAGraphMode, VllmConfig
from vllm.forward_context import BatchDescriptor, get_forward_context
from vllm.logger import logger
from vllm.platforms import current_platform

from ..utils import weak_ref_tensors


_ACL_GRAPH_WRAPPERS: "weakref.WeakSet[ACLGraphWrapper]" = weakref.WeakSet()

# Model/runtime semantics are separate from graph allocations.  A graph epoch
# may be discarded while MoE placement and Attention topology must survive.
_SEMANTIC_STATE_FIELDS = (
    "expert_map", "loaded_expert_map", "elastic_original_expert_map",
    "log2phy", "primary_log2phy", "elastic_runtime_log2phy",
    "active_expert_mask", "num_experts", "local_num_experts",
    "active_local_num_experts", "loaded_local_num_experts",
)


def _snapshot_graph_semantic_state(root: Any) -> list[tuple[Any, dict[str, Any]]]:
    if root is not None and not hasattr(root, "modules") and hasattr(root, "unwrap"):
        root = root.unwrap()
    if root is None or not hasattr(root, "modules"):
        return []
    state = []
    for module in root.modules():
        values = {}
        for field in _SEMANTIC_STATE_FIELDS:
            if not hasattr(module, field):
                continue
            value = getattr(module, field)
            if isinstance(value, torch.Tensor):
                values[field] = ("tensor", value.detach().cpu().clone(),
                                 value.device, value.dtype)
            elif value is not None and isinstance(value, (int, float, bool, str)):
                values[field] = ("scalar", value)
        if values:
            state.append((module, values))
    return state


def _restore_graph_semantic_state(state: list[tuple[Any, dict[str, Any]]]) -> None:
    for module, values in state:
        for field, saved in values.items():
            if saved[0] == "tensor":
                _, cpu_value, device, dtype = saved
                try:
                    setattr(module, field, cpu_value.to(device=device, dtype=dtype))
                except Exception as exc:
                    logger.warning("Unable to restore graph semantic field %s: %s",
                                   field, exc)
            else:
                setattr(module, field, saved[1])


@dataclasses.dataclass
class ACLGraphEntry:
    batch_descriptor: BatchDescriptor
    aclgraph: torch.npu.NPUGraph | None = None
    output: Any | None = None

    # for aclgraph debugging, track the input addresses
    # during capture, and check if they are the same during replay
    input_addresses: list[int] | None = None


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

    Note: ACLGraphWrapper does not store persistent buffers or copy any
    runtime inputs into that buffers for replay. We assume implementing them
    is done outside of the wrapper. That is because we do not make any
    assumption on the dynamic shape (batch size) of the runtime inputs, as a
    trade-off for staying orthogonal to compilation logic. Nevertheless,
    tracing and checking the input addresses to be consistent during replay is
    guaranteed when VLLM_LOGGING_LEVEL == "DEBUG".
    """

    def __init__(
        self,
        runnable: Callable,
        vllm_config: VllmConfig,
        runtime_mode: CUDAGraphMode,
        cudagraph_options: CUDAGraphOptions | None = None,
    ):
        self.runnable = runnable
        self.vllm_config = vllm_config
        self.runtime_mode = runtime_mode
        self.compilation_config = vllm_config.compilation_config

        self.first_run_finished = False
        self.is_debugging_mode = envs.VLLM_LOGGING_LEVEL == "DEBUG"

        # assert runtime_mode is not NONE(no aclgraph), otherwise, we don't
        # need to initialize a ACLGraphWrapper.
        assert self.runtime_mode != CUDAGraphMode.NONE
        self.graph_pool = current_platform.get_global_graph_pool()

        if cudagraph_options is None:
            cudagraph_options = CUDAGraphOptions()
        self.aclgraph_options = cudagraph_options
        # A graph recaptured after native-sleep must keep its output tensor
        # alive until the next invalidation.  The generic weak-output policy
        # is safe for one-shot serving captures, but after a weight/KV
        # transition it can leave replay returning a dead/stale output view.
        self._strong_lifecycle_output = (
            os.getenv("VLLM_ROLLOUT_NATIVE_SLEEP_GRAPH_LIFECYCLE", "0").lower()
            in ("1", "true", "yes", "on")
            and os.getenv("VLLM_ROLLOUT_NATIVE_SLEEP_GRAPH_STRONG_OUTPUT", "0").lower()
            in ("1", "true", "yes", "on")
        )
        # the entries for different batch descriptors that we need to capture
        # aclgraphs for.
        self.concrete_aclgraph_entries: dict[BatchDescriptor, ACLGraphEntry] = {}
        _ACL_GRAPH_WRAPPERS.add(self)

    def __getattr__(self, key: str):
        # allow accessing the attributes of the runnable.
        if hasattr(self.runnable, key):
            return getattr(self.runnable, key)
        raise AttributeError(f"Attribute {key} not exists in the runnable of aclgraph wrapper: {self.runnable}")

    def unwrap(self) -> Callable:
        # in case we need to access the original runnable.
        return self.runnable

    def clear_aclgraph_cache(self) -> int:
        """Release graphs that may retain stale tensor or HCCL resources."""
        num_entries = len(self.concrete_aclgraph_entries)
        for entry in self.concrete_aclgraph_entries.values():
            aclgraph = entry.aclgraph
            if aclgraph is not None and hasattr(aclgraph, "reset"):
                aclgraph.reset()
            entry.aclgraph = None
            entry.output = None
            entry.input_addresses = None
            if hasattr(entry, "input_tensors"):
                entry.input_tensors = None
            if hasattr(entry, "attention_task_range"):
                entry.attention_task_range = None
        self.concrete_aclgraph_entries.clear()
        self.first_run_finished = False
        return num_entries

    def __call__(self, *args, **kwargs):
        forward_context = get_forward_context()
        batch_descriptor = forward_context.batch_descriptor
        aclgraph_runtime_mode = forward_context.cudagraph_runtime_mode

        if aclgraph_runtime_mode == CUDAGraphMode.NONE or aclgraph_runtime_mode != self.runtime_mode:
            # CUDAGraphMode.NONE could mean the profile run, a warmup run, or
            # running without aclgraphs.
            # We do not trigger capture/replay if the runtime mode is not
            # matches. This enables properly dispatching to the correct
            # CUDAGraphWrapper when nesting multiple instances with different
            # runtime modes.
            return self.runnable(*args, **kwargs)

        if batch_descriptor not in self.concrete_aclgraph_entries:
            # create a new entry for this batch descriptor
            self.concrete_aclgraph_entries[batch_descriptor] = ACLGraphEntry(batch_descriptor=batch_descriptor)

        entry = self.concrete_aclgraph_entries[batch_descriptor]

        if entry.aclgraph is None:
            lazy_capture = (
                not cudagraph_monitor.cudagraph_capturing_enabled
                and os.getenv("VLLM_USE_V1", "0").lower()
                in ("1", "true", "yes", "on")
                and os.getenv("VLLM_ASCEND_ALLOW_LAZY_ACLGRAPH_CAPTURE",
                              "0").lower() in ("1", "true", "yes", "on")
            )
            if lazy_capture:
                logger.info_once(
                    "Temporarily allowing lazy ACL graph recapture after "
                    "runtime cache invalidation.",
                    scope="local",
                )
                set_cudagraph_capturing_enabled(True)
            if self.aclgraph_options.debug_log_enable:
                # Since we capture aclgraph for many different shapes and
                # capturing is fast, we don't need to log it for every
                # shape. E.g. we only log it for the first subgraph in
                # piecewise mode.
                logger.debug("Capturing a aclgraph on (%s,%s)", self.runtime_mode.name, entry.batch_descriptor)
            try:
                # validate that aclgraph capturing is legal at this point.
                validate_cudagraph_capturing_enabled()

                input_addresses = [x.data_ptr() for x in args if isinstance(x, torch.Tensor)]
                entry.input_addresses = input_addresses
                aclgraph = torch.npu.NPUGraph()

                with ExitStack() as stack:
                    if self.aclgraph_options.gc_disable:
                        # during every model forward for piecewise aclgraph
                        # mode, we will capture many pieces of aclgraphs
                        # (roughly one per layer). running gc again and again
                        # across layers will make the aclgraph capture very slow.
                        # therefore, we only run gc for the first graph,
                        # and disable gc for the rest of the graphs.
                        stack.enter_context(patch("gc.collect", lambda: None))
                        stack.enter_context(patch("torch.npu.empty_cache", lambda: None))

                    # mind-exploding: carefully manage the reference and memory.
                    forward_context.capturing = True
                    with torch.npu.graph(aclgraph, pool=self.graph_pool):
                        # `output` is managed by pytorch's aclgraph pool
                        output = self.runnable(*args, **kwargs)
                        if (self.aclgraph_options.weak_ref_output
                                and not self._strong_lifecycle_output):
                            # by converting it to weak ref,
                            # the original `output` will immediately be released
                            # to save memory. It is only safe to do this for
                            # the last graph in piecewise aclgraph mode, because
                            # the output of the last graph will not be used by
                            # any other acl graph.
                            output = weak_ref_tensors(output)
            finally:
                if lazy_capture:
                    set_cudagraph_capturing_enabled(False)

            # here we always use weak ref for the workspaces
            # to save memory
            global _graph_params
            global _draft_graph_params
            weak_ref_workspaces(_graph_params)
            weak_ref_workspaces(_draft_graph_params)

            # here we always use weak ref for the output
            # to save memory
            entry.output = (
                output if self._strong_lifecycle_output
                else weak_ref_tensors(output)
            )
            entry.aclgraph = aclgraph

            compilation_counter.num_cudagraph_captured += 1

            # important: we need to return the output, rather than
            # the weak ref of the output, so that pytorch can correctly
            # manage the memory during acl graph capture
            return output

        if self.is_debugging_mode:
            # check if the input addresses are the same
            new_input_addresses = [x.data_ptr() for x in args if isinstance(x, torch.Tensor)]
            assert new_input_addresses == entry.input_addresses, (
                f"Input addresses for aclgraphs are different "
                f"during replay. Expected {entry.input_addresses}, "
                f"got {new_input_addresses}"
            )

        logger.info_once("Replaying aclgraph")
        # In async scheduling or multi-threaded (MT) scenarios, it is possible that
        # the CPU's record event (from update_attn_params) for the iteration i completes
        # before the grph replay of iteration i-1.
        # To ensure proper ordering, we must call synchronize here before replaying,
        # so that update_attn_params only executes after the previous graph replay has fully completed.
        # If we do not in main model and in full-graph mode when using merge-eagle-graph,
        # we do not need to synchronize.
        use_eagle = (
            self.vllm_config.speculative_config.method in ("eagle", "eagle3")
            if self.vllm_config.speculative_config
            else False
        )
        # Match the v0.14 replay/event ordering. Only merged draft EAGLE
        # graphs omit this barrier; main-model FULL graphs still need it.
        if (self.runtime_mode != CUDAGraphMode.FULL
                or not forward_context.is_draft_model or not use_eagle):
            torch.npu.synchronize()
        # Propagate failures: silently running eagerly would invalidate both
        # correctness diagnostics and graph performance measurements.
        entry.aclgraph.replay()
        return entry.output


def weak_ref_workspaces(params):
    if params is None:
        return
    for num_tokens in params.workspaces:
        if params.workspaces[num_tokens] is None:
            continue
        params.workspaces[num_tokens] = weak_ref_tensors(params.workspaces[num_tokens])


def _reset_graph_params(params: "GraphParams | None") -> None:
    if params is None:
        return
    for num_tokens in params.events:
        params.events[num_tokens].clear()
    for num_tokens in params.handles:
        params.handles[num_tokens].clear()
    for num_tokens in params.attn_params:
        params.attn_params[num_tokens].clear()
    for num_tokens in params.workspaces:
        params.workspaces[num_tokens] = None


def reset_graph_params_runtime_state(include_draft: bool = True) -> None:
    """Clear runtime task groups captured inside ACL graph attention ops."""
    _reset_graph_params(_graph_params)
    if include_draft:
        _reset_graph_params(_draft_graph_params)


def _renew_aclgraph_pool() -> None:
    """Ensure captures after an elastic transition use a fresh graph pool."""
    platform_cls = current_platform.__class__
    old_pool = getattr(platform_cls, "_global_graph_pool", None)
    wrappers = list(_ACL_GRAPH_WRAPPERS)
    for wrapper in wrappers:
        if old_pool is None or wrapper.graph_pool == old_pool:
            wrapper.graph_pool = None
    platform_cls._global_graph_pool = None
    gc.collect()
    try:
        torch.npu.empty_cache()
    except RuntimeError as exc:
        # torch_npu may still report an internal capture counter while the
        # previous graph entry is being retired. The old pool has already
        # been detached; a failed cache flush must not abort the new capture.
        logger.warning("Skipping ACLGraph pool empty_cache during renewal: %s", exc)
    new_pool = current_platform.get_global_graph_pool()
    for wrapper in wrappers:
        wrapper.graph_pool = new_pool


def release_aclgraph_pool() -> None:
    """Drop the current ACLGraph pool before a native-sleep transition.

    Clearing ``ACLGraphEntry`` objects is not sufficient on torch_npu: the
    opaque pool token is retained by both ``Platform`` and every wrapper, and
    the allocator is then allowed to keep the graph's physical segments.  A
    native level-2 sleep subsequently unmaps KV pages while those segments
    are still resident, so the next ``wake_up(tags=["kv_cache"])`` can fail
    even though the Python graph cache is empty.

    Do not allocate a replacement pool here.  Rebinding is deliberately
    performed only after KV has been restored, immediately before capture.
    """
    platform_cls = current_platform.__class__
    wrappers = list(_ACL_GRAPH_WRAPPERS)
    for wrapper in wrappers:
        wrapper.graph_pool = None
    platform_cls._global_graph_pool = None
    gc.collect()
    # torch_npu's npugraph tree and Dynamo/Inductor caches can retain graph
    # task groups and their workspaces after NPUGraph.reset(). Clear them only
    # at the explicit native-sleep lifecycle boundary; the next capture will
    # rebuild the required compiled callable and pool.
    try:
        from torch_npu.npu._graph_tree import reset_npugraph_trees
        reset_npugraph_trees()
    except Exception as exc:
        logger.warning("Unable to reset torch_npu npugraph trees: %s", exc)
    try:
        torch.compiler.reset()
    except Exception as exc:
        logger.warning("Unable to reset torch compiler cache: %s", exc)
    try:
        torch.npu.synchronize()
    except Exception:
        pass
    try:
        torch.npu.empty_cache()
    except RuntimeError as exc:
        # A capture counter can briefly remain set while torch_npu retires a
        # graph.  The references are still detached; the caller will retry
        # the allocator flush after sleep has completed.
        logger.warning("ACLGraph pool flush deferred during release: %s", exc)


def renew_aclgraph_pool() -> int:
    """Rotate the graph pool after clearing all graph/task references.

    This is the v0.11 lifecycle operation.  Merely setting the platform pool
    to ``None`` and postponing allocation until wake-up leaves torch_npu's
    allocator with the old opaque pool token during native sleep.  Creating a
    fresh token after the old graph objects are collectible forces inactive
    graph segments through ``empty_cache`` before level-2 sleep remaps KV.
    """
    platform_cls = current_platform.__class__
    old_pool = getattr(platform_cls, "_global_graph_pool", None)
    wrappers = list(_ACL_GRAPH_WRAPPERS)
    for wrapper in wrappers:
        if old_pool is None or wrapper.graph_pool == old_pool:
            wrapper.graph_pool = None
    platform_cls._global_graph_pool = None
    gc.collect()
    try:
        torch.npu.empty_cache()
    except RuntimeError as exc:
        logger.warning("ACLGraph pool rotation flush deferred: %s", exc)
    # Match the v0.11 lifecycle: let the allocator retire inactive graph
    # segments before native sleep starts unmapping the large KV/weight pools.
    try:
        torch.npu.synchronize()
    except RuntimeError as exc:
        logger.warning("ACLGraph pool rotation synchronize deferred: %s", exc)
    new_pool = current_platform.get_global_graph_pool()
    for wrapper in wrappers:
        wrapper.graph_pool = new_pool
    return len(wrappers)


def rebind_aclgraph_pool() -> Any:
    """Create one fresh pool token after native-sleep KV restoration."""
    platform_cls = current_platform.__class__
    platform_cls._global_graph_pool = None
    new_pool = current_platform.get_global_graph_pool()
    for wrapper in list(_ACL_GRAPH_WRAPPERS):
        wrapper.graph_pool = new_pool
    return new_pool


def clear_aclgraph_caches(root: Any = None, *, reset_graph_params: bool = True,
                          renew_pool: bool = False) -> int:
    """Clear all ACLGraphWrapper caches reachable from ``root``.

    Piecewise wrappers are installed as plain Python attributes on compiled
    graph modules, not always as ``nn.Module`` children.  A small object-graph
    walk is therefore more reliable than ``named_modules()`` alone.  A global
    weak registry covers wrappers hidden inside torch.compile closures.
    """
    # Never retire a graph while replay/task updates may still be queued.
    # This is the ordering used by the known-good v0.11 lifecycle.
    torch.npu.synchronize()
    semantic_state = _snapshot_graph_semantic_state(root)

    seen: set[int] = set()
    cleared_wrappers: set[int] = set()
    stack: list[Any] = [] if root is None else [root]
    cleared = 0

    def clear_wrapper(wrapper: ACLGraphWrapper) -> None:
        nonlocal cleared
        wrapper_id = id(wrapper)
        if wrapper_id in cleared_wrappers:
            return
        cleared_wrappers.add(wrapper_id)
        cleared += wrapper.clear_aclgraph_cache()

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
    # Do not rotate until NPUGraph.reset() has run for every wrapper and all
    # references are collectible. This ordering is required for HCCL task
    # groups and avoids retaining the old physical graph segments.
    gc.collect()
    if renew_pool:
        _renew_aclgraph_pool()
    if renew_pool:
        torch.npu.empty_cache()
        torch.npu.synchronize()
    elif cleared and os.getenv("VLLM_ASCEND_ACLGRAPH_CLEAR_EMPTY_CACHE",
                              "0").lower() in ("1", "true", "yes", "on"):
        torch.npu.empty_cache()
    # Restore only model/runtime semantics. Graph entries, task handles,
    # workspaces and physical pool allocations remain retired.
    _restore_graph_semantic_state(semantic_state)
    if semantic_state:
        logger.info("Restored graph semantic state for %s modules after graph cleanup",
                    len(semantic_state))
    return cleared


def disable_aclgraph_dispatch(model_runner: Any) -> int:
    """Fail closed while FULL ACLGraphs are being rebuilt.

    Clearing wrapper entries alone is insufficient: the vLLM dispatcher still
    advertises the old ``BatchDescriptor`` keys, so the dummy forward inside a
    recapture can dispatch to a stale graph and nest replay inside capture.
    v0.11 explicitly cleared these keys before every floor/topology capture.
    """
    dispatcher = getattr(model_runner, "aclgraph_dispatcher", None)
    if dispatcher is None:
        dispatcher = getattr(model_runner, "cudagraph_dispatcher", None)
    if dispatcher is None:
        return 0
    dispatcher.keys_initialized = False
    cleared = 0
    for keys in getattr(dispatcher, "cudagraph_keys", {}).values():
        cleared += len(keys)
        keys.clear()
    return cleared


def enable_aclgraph_dispatch(model_runner: Any) -> None:
    """Recreate dispatcher keys after an atomic graph invalidation."""
    dispatcher = getattr(model_runner, "aclgraph_dispatcher", None)
    if dispatcher is None:
        dispatcher = getattr(model_runner, "cudagraph_dispatcher", None)
    if dispatcher is None:
        return
    config = getattr(model_runner, "vllm_config", None)
    compilation_config = getattr(model_runner, "compilation_config", None)
    if compilation_config is None and config is not None:
        compilation_config = config.compilation_config
    mode = getattr(compilation_config, "cudagraph_mode", None)
    if mode is None:
        return
    dispatcher.initialize_cudagraph_keys(
        mode, int(getattr(model_runner, "uniform_decode_query_len", 1)))


def update_full_graph_params(
    attn_backend,
    update_stream,
    forward_context,
    num_tokens,
    vllm_config,
    speculative_config=None,
    num_dcp_pcp_tokens=None,
):
    impl_cls = attn_backend.get_impl_cls()
    impl_cls.update_graph_params(
        update_stream,
        forward_context,
        num_tokens,
        vllm_config,
        speculative_config,
        num_dcp_pcp_tokens,
    )


@dataclass
class GraphParams:
    events: dict[int, list[torch.npu.ExternalEvent]]
    workspaces: dict[int, torch.Tensor]
    handles: dict[int, list[torch_npu._C._NPUTaskGroupHandle]]
    attn_params: dict[int, list[tuple]]


_graph_params: GraphParams | None = None


def set_graph_params(aclgraph_capture_sizes: list[int]):
    global _graph_params
    if _graph_params is not None:
        # Manual KV/cache-engine recreation re-enters model-runner
        # initialization in the same process.  Reusing the existing parameter
        # tables is safe when capture sizes are unchanged and avoids aborting
        # the first rollout after an actor update.
        existing = set(_graph_params.events)
        requested = set(aclgraph_capture_sizes)
        if existing == requested:
            return
        raise ValueError("Graph parameters have already been set with different capture sizes!")
    _graph_params = GraphParams(
        {size: [] for size in aclgraph_capture_sizes},
        {size: None for size in aclgraph_capture_sizes},
        {size: [] for size in aclgraph_capture_sizes},
        {size: [] for size in aclgraph_capture_sizes},
    )


def reconfigure_graph_params(aclgraph_capture_sizes: set[int],
                             update_stream: Any = None) -> None:
    """Rebuild FULL-decode Attention task tables for a new capture epoch.

    Clearing the old lists is not enough after a topology/floor transition:
    torch_npu task handles and workspace entries belong to the previous graph
    epoch. v0.11 replaced the parameter tables before every floor recapture;
    v0.14 must do the same for native ACLGraph.
    """
    global _graph_params
    sizes = {int(size) for size in aclgraph_capture_sizes}
    if not sizes or min(sizes) < 1:
        raise ValueError("ACLGraph capture sizes must be positive")
    _graph_params = GraphParams(
        {size: [] for size in sizes},
        {size: None for size in sizes},
        {size: [] for size in sizes},
        {size: [] for size in sizes},
    )


def update_graph_params_workspaces(num_tokens: int, workspace: torch.Tensor):
    global _graph_params
    if _graph_params is not None:
        _graph_params.workspaces[num_tokens] = workspace


def get_graph_params():
    return _graph_params


_draft_graph_params: GraphParams | None = None


def set_draft_graph_params(aclgraph_capture_sizes: list[int]):
    global _draft_graph_params
    if _draft_graph_params is not None:
        raise ValueError("DraftGraph parameters have already been set!")
    _draft_graph_params = GraphParams(
        {size: [] for size in aclgraph_capture_sizes},
        {size: None for size in aclgraph_capture_sizes},
        {size: [] for size in aclgraph_capture_sizes},
        {size: [] for size in aclgraph_capture_sizes},
    )


def update_draft_graph_params_workspaces(num_tokens: int, workspace: Any):
    global _draft_graph_params
    if _draft_graph_params is not None:
        _draft_graph_params.workspaces[num_tokens] = workspace


def get_draft_graph_params():
    return _draft_graph_params
