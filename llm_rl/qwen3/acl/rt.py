"""Minimal pyACL runtime API implemented against CANN's libascendcl."""

from __future__ import annotations

import ctypes
import os
from functools import lru_cache
from pathlib import Path


ACL_SUCCESS = 0


def _runtime_library_path() -> Path:
    roots = (
        os.environ.get("QWEN3_CANN85_HOME"),
        os.environ.get("ASCEND_HOME_PATH"),
    )
    for root in roots:
        if not root:
            continue
        base = Path(root)
        for relative in (
            "aarch64-linux/lib64/libascendcl.so",
            "lib64/libascendcl.so",
        ):
            candidate = (base / relative).resolve()
            if candidate.is_file():
                return candidate
    raise RuntimeError(
        "libascendcl.so was not found in the selected CANN runtime; "
        "source tools/activate_cann85_a3_torch29.sh first"
    )


@lru_cache(maxsize=1)
def _runtime() -> ctypes.CDLL:
    library = ctypes.CDLL(str(_runtime_library_path()))
    library.aclrtGetSocName.argtypes = []
    library.aclrtGetSocName.restype = ctypes.c_char_p
    library.aclrtGetDeviceCount.argtypes = [ctypes.POINTER(ctypes.c_uint32)]
    library.aclrtGetDeviceCount.restype = ctypes.c_int
    library.aclrtGetMemInfo.argtypes = [
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_size_t),
        ctypes.POINTER(ctypes.c_size_t),
    ]
    library.aclrtGetMemInfo.restype = ctypes.c_int
    library.aclrtMemcpy.argtypes = [
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_int,
    ]
    library.aclrtMemcpy.restype = ctypes.c_int
    return library


def get_soc_name() -> str:
    name = _runtime().aclrtGetSocName()
    return name.decode("utf-8") if name else ""


def get_device_count() -> tuple[int, int]:
    count = ctypes.c_uint32()
    result = _runtime().aclrtGetDeviceCount(ctypes.byref(count))
    return count.value, int(result)


def get_mem_info(attr: int) -> tuple[int, int, int]:
    free = ctypes.c_size_t()
    total = ctypes.c_size_t()
    result = _runtime().aclrtGetMemInfo(
        int(attr), ctypes.byref(free), ctypes.byref(total)
    )
    return free.value, total.value, int(result)


def memcpy(
    destination: int,
    destination_max: int,
    source: int,
    count: int,
    kind: int,
) -> int:
    return int(
        _runtime().aclrtMemcpy(
            ctypes.c_void_p(destination),
            destination_max,
            ctypes.c_void_p(source),
            count,
            kind,
        )
    )


def runtime_library_path() -> str:
    """Return the resolved library path for fail-closed runtime audits."""
    return str(_runtime_library_path())
