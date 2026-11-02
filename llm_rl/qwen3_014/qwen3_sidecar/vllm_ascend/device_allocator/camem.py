#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
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
#
# CANN-mem-based pytorch pluggable allocator to implement sleep mode.
#
import dataclasses
import os
from collections.abc import Callable
from contextlib import contextmanager
from typing import Any

import torch
from acl.rt import memcpy  # type: ignore # noqa: F401
from vllm.logger import logger


def find_loaded_library(lib_name) -> str | None:
    """
    According to according to https://man7.org/linux/man-pages/man5/proc_pid_maps.5.html,
    the file `/proc/self/maps` contains the memory maps of the process, which includes the
    shared libraries loaded by the process. We can use this file to find the path of the
    a loaded library.
    """  # noqa
    found_line = None
    with open("/proc/self/maps") as f:
        for line in f:
            if lib_name in line:
                found_line = line
                break
    if found_line is None:
        # the library is not loaded in the current process
        return None
    # if lib_name is libcudart, we need to match a line with:
    # address /path/to/libcudart-hash.so.11.0
    start = found_line.index("/")
    path = found_line[start:].strip()
    filename = path.split("/")[-1]
    assert filename.rpartition(".so")[0].startswith(lib_name), f"Unexpected filename: {filename} for library {lib_name}"
    return path


camem_available = False
try:
    from vllm_ascend.vllm_ascend_C import (  # type: ignore # noqa: F401
        init_module,
        python_create_and_map,
        python_unmap_and_release,
    )

    lib_name = find_loaded_library("vllm_ascend_C")
    camem_available = True
except ImportError as e:
    logger.warning("Failed to import vllm_ascend_C:%s. Sleep mode will be disabled. ", e)
    init_module = None
    python_create_and_map = None
    python_unmap_and_release = None
    lib_name = None
    libcudart = None

# py_device, py_alignedSize, py_d_mem, py_p_memHandle
HandleType = tuple[int, int, int, int]

# Isolated local shim: same native allocator, but physical map/unmap/free
# operations are idempotent so entirely idle cached segments can be parked.
_park_idle_enabled = os.getenv("VLLM_ASCEND_NATIVE_SLEEP_PARK_IDLE", "0") == "1"
_allocator_callback_keepalive = None
if _park_idle_enabled:
    from . import camem_lifecycle_C as _lifecycle
    init_module = _lifecycle.init_module
    python_create_and_map = _lifecycle.python_lifecycle_create_and_map
    python_unmap_and_release = _lifecycle.python_lifecycle_unmap_and_release
    lib_name = _lifecycle.__file__
    camem_available = True


@dataclasses.dataclass
class AllocationData:
    handle: HandleType
    tag: str
    cpu_backup_tensor: torch.Tensor | None = None
    parked: bool = False
    pool_key: str | None = None


def create_and_map(allocation_handle: HandleType) -> None:
    python_create_and_map(*allocation_handle)


def unmap_and_release(allocation_handle: HandleType) -> None:
    python_unmap_and_release(*allocation_handle)


def get_pluggable_allocator(
    python_malloc_fn: Callable[[tuple[int, int, int, int]], None],
    python_free_func: Callable[[int], tuple[int, int, int, int]],
) -> torch.npu.memory.NPUPluggableAllocator:
    global _allocator_callback_keepalive
    init_module(python_malloc_fn, python_free_func)
    # The upstream extension borrows these bound-method references. A reused
    # pool outlives the contextmanager that first supplied them, so pin both
    # callbacks for the allocator's entire singleton lifetime.
    _allocator_callback_keepalive = (python_malloc_fn, python_free_func)
    new_alloc = torch.npu.memory.NPUPluggableAllocator(
        lib_name, "lifecycle_malloc" if _park_idle_enabled else "my_malloc",
        "lifecycle_free" if _park_idle_enabled else "my_free")
    return new_alloc


@contextmanager
def use_memory_pool_with_allocator(
    python_malloc_fn: Callable[[tuple[int, int, int, int]], None],
    python_free_func: Callable[[int], tuple[int, int, int, int]],
):
    new_alloc = get_pluggable_allocator(python_malloc_fn, python_free_func)
    mem_pool = torch.npu.memory.MemPool(new_alloc._allocator)
    with torch.npu.memory.use_mem_pool(mem_pool):
        yield mem_pool, new_alloc


class CaMemAllocator:
    """
    A singleton class that manages a memory pool for CANN tensors.
    The memory in this pool can be offloaded or discarded when the
    allocator sleeps.
    Inside the `use_memory_pool(tag)` context, all tensors created will
    be allocated in the memory pool, and has the same tag as the
    tag passed to the context.
    When we call `sleep`, all tensors with the specified tag will be
    offloaded to CPU memory, and the rest of the tensors will be discarded.
    When we call `wake_up`, all tensors that are previously offloaded
    will be loaded back to GPU memory, and the rest of the tensors will
    have empty memory.
    Why it needs to be a singleton?
    When allocated tensors are garbage collected, PyTorch will call
    the free callback, which will call the `python_free_callback` method.
    The C-extension uses a global variable to store the function of an
    instance of this class. If we create multiple instances of this class,
    the global variable will be overwritten and the free callback will
    not work as expected.
    """

    instance = None
    default_tag: str = "default"

    @staticmethod
    def get_instance() -> "CaMemAllocator":
        """
        CaMemAllocator is a singleton class.
        We cannot call the constructor directly.
        Call this method to get the instance.
        """
        if CaMemAllocator.instance is None:
            CaMemAllocator.instance = CaMemAllocator()
        return CaMemAllocator.instance

    def __init__(self):
        conf = os.environ.get("PYTORCH_NPU_ALLOC_CONF", "")
        assert "expandable_segments:True" not in conf, (
            "Expandable segments are not compatible with memory pool. "
            "Please track https://github.com/pytorch/pytorch/issues/147851 "
            "for the latest updates."
        )

        self.pointer_to_data: dict[int, AllocationData] = {}
        self.current_tag: str = CaMemAllocator.default_tag
        self.allocator_and_pools: dict[str, Any] = {}

    def python_malloc_callback(self, allocation_handle: HandleType) -> None:
        """
        Internal method to store the allocation data
        when memory is allocated in the memory pool."""
        py_d_mem = allocation_handle[2]
        self.pointer_to_data[py_d_mem] = AllocationData(
            allocation_handle, self.current_tag,
            pool_key=getattr(self, "current_pool_key", self.current_tag))
        return

    def python_free_callback(self, ptr: int) -> HandleType:
        """
        Internal method to look up the allocation data
        when memory is freed in the memory pool."""
        data = self.pointer_to_data.pop(ptr)
        if data.cpu_backup_tensor is not None:
            data.cpu_backup_tensor = None
        return data.handle

    def sleep(self, offload_tags: tuple[str, ...] | str | None = None) -> None:
        """
        Put the allocator in sleep mode.
        All data in the memory allocation with the specified tag will be
        offloaded to CPU memory, and others will be discarded.
        :param offload_tags: The tags of the memory allocation that will be
            offloaded. The rest of the memory allocation will be discarded.
        """
        if offload_tags is None:
            # by default, allocated tensors are offloaded
            # when the allocator sleeps
            offload_tags = (CaMemAllocator.default_tag,)
        elif isinstance(offload_tags, str):
            offload_tags = (offload_tags,)

        assert isinstance(offload_tags, tuple)

        for ptr, data in self.pointer_to_data.items():
            if data.parked:
                continue  # no live tensor owns this already-unmapped segment
            handle = data.handle
            if data.tag in offload_tags:
                size_in_bytes = handle[1]
                cpu_backup_tensor = torch.empty(size_in_bytes, dtype=torch.uint8, device="cpu", pin_memory=True)
                cpu_ptr = cpu_backup_tensor.data_ptr()
                ACL_MEMCPY_DEVICE_TO_HOST = 2
                dest_max = cpu_ptr + size_in_bytes * 2
                memcpy(cpu_ptr, dest_max, ptr, size_in_bytes, ACL_MEMCPY_DEVICE_TO_HOST)
                data.cpu_backup_tensor = cpu_backup_tensor
            unmap_and_release(handle)

    def wake_up(self, tags: list[str] | None = None) -> None:
        """
        Wake up the allocator from sleep mode.
        All data that is previously offloaded will be loaded back to GPU
        memory, and the rest of the data will have empty memory."""
        for ptr, data in self.pointer_to_data.items():
            if not data.parked and (tags is None or data.tag in tags):
                handle = data.handle
                create_and_map(handle)
                if data.cpu_backup_tensor is not None:
                    cpu_backup_tensor = data.cpu_backup_tensor
                    if cpu_backup_tensor is not None:
                        size_in_bytes = cpu_backup_tensor.numel() * cpu_backup_tensor.element_size()
                        cpu_ptr = cpu_backup_tensor.data_ptr()
                        ACL_MEMCPY_HOST_TO_DEVICE = 1
                        dest_max = ptr + size_in_bytes * 2
                        memcpy(ptr, dest_max, cpu_ptr, size_in_bytes, ACL_MEMCPY_HOST_TO_DEVICE)
                        data.cpu_backup_tensor = None

    @contextmanager
    def use_memory_pool(self, tag: str | None = None):
        """
        A context manager to use the memory pool.
        All memory allocation created inside the context will be allocated
        in the memory pool, and has the specified tag.
        :param tag: The tag of the memory allocation. If None, the default tag
            will be used.
        """
        if tag is None:
            tag = CaMemAllocator.default_tag

        assert isinstance(tag, str)

        old_tag = self.current_tag
        old_pool_key = getattr(self, "current_pool_key", None)
        pool_key = getattr(self, "step_pool_keys", {}).get(tag, tag)
        self.current_tag = tag
        self.current_pool_key = pool_key
        try:
            for data in self.pointer_to_data.values():
                if (data.tag == tag and data.parked
                        and (getattr(data, "pool_key", None) or tag) == pool_key):
                    create_and_map(data.handle)
                    data.parked = False
            existing = self.allocator_and_pools.get(pool_key)
            if existing is None:
                with use_memory_pool_with_allocator(
                        self.python_malloc_callback, self.python_free_callback) as data:
                    # Keep both the pool and its allocator alive, and reuse
                    # this pool on subsequent RL reloads. A new pool cannot
                    # reuse the old pool's cached source-shard allocations,
                    # even though native wake has remapped those blocks.
                    self.allocator_and_pools[pool_key] = data
                    yield
            else:
                pool, _allocator = existing
                with torch.npu.memory.use_mem_pool(pool):
                    yield
        finally:
            self.current_tag = old_tag
            self.current_pool_key = old_pool_key

    def select_step_pool(self, tag: str, capacity: int) -> None:
        """Select capacity-specific native pools without reviving retired pools.

        Tags still identify native sleep/wake ownership. Only the allocation
        pool key changes; old live segments remain subject to normal wake,
        while old wholly-inactive segments stay physically parked. Keeping
        pool tokens alive avoids unsafe releasePool/re-entry on this runtime.
        """
        if tag not in ("weights", "kv_cache") or capacity <= 0:
            raise ValueError("Expected a positive weights/KV capacity")
        if not hasattr(self, "step_pool_keys"):
            self.step_pool_keys = {}
        self.step_pool_keys[tag] = f"{tag}:step_capacity={capacity}"

    def park_retired_step_segments(self, tags: tuple[str, ...]) -> dict[str, int]:
        """Park only wholly-inactive segments after step-boundary retirement.

        Unlike regular idle-weight parking this is allowed to retire an OLD
        KV pool, but only after graph, Attention and runner references have
        been cleared and PyTorch reports every block inactive. Never unmap
        live/partly-active tensors and never change communication semantics.
        """
        if not _park_idle_enabled:
            raise RuntimeError("Step capacity changes require the native lifecycle allocator")
        if torch.npu.memory.MemPoolContext.active_pool() is not None:
            raise RuntimeError("Cannot retire an active allocation pool")
        torch.npu.synchronize()
        count = released = 0
        for segment in torch.npu.memory_snapshot():
            data = self.pointer_to_data.get(segment["address"])
            if (data is None or data.tag not in tags or data.parked
                    or segment["active_size"] != 0
                    or any(b["state"] != "inactive" for b in segment["blocks"])):
                continue
            unmap_and_release(data.handle)
            data.parked = True
            count += 1
            released += data.handle[1]
        result = dict(segments=count, retired_capacity_bytes=released)
        logger.info("Native step pool retirement: tags=%s result=%s", tags, result)
        return result

    def park_idle_weight_segments(self) -> dict[str, int]:
        """Unmap only wholly idle weights-pool segments, preserving virtual VA.

        PyTorch still owns the cached blocks. Before entering their pool again
        use_memory_pool remaps them. Native wake skips these unused blocks;
        active weights, partial segments, KV, and graph pools are untouched.
        """
        if not _park_idle_enabled:
            return {}
        if torch.npu.memory.MemPoolContext.active_pool() is not None:
            raise RuntimeError("Cannot park a currently active allocation pool")
        torch.npu.synchronize()
        before = torch.npu.mem_get_info()[0]
        count = released = 0
        for segment in torch.npu.memory_snapshot():
            data = self.pointer_to_data.get(segment["address"])
            if (data is None or data.tag != "weights" or data.parked
                    or segment["active_size"] != 0
                    or any(b["state"] != "inactive" for b in segment["blocks"])):
                continue
            unmap_and_release(data.handle)
            data.parked = True
            count += 1
            released += data.handle[1]
        after = torch.npu.mem_get_info()[0]
        result = dict(segments=count, released_bytes=released,
                      physical_free_before=before, physical_free_after=after)
        logger.info("Native sleep idle-segment parking: %s", result)
        return result

    def get_current_usage(self) -> int:
        """
        Get the total number of bytes allocated in the memory pool.
        """
        sum_bytes: int = 0
        for ptr, data in self.pointer_to_data.items():
            handle = data.handle
            sum_bytes += handle[1]
        return sum_bytes
