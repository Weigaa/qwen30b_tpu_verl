#!/usr/bin/env python3
"""Build the local native-sleep mapping shim; no changes to installed packages."""
import os
import subprocess
import sysconfig
from pathlib import Path


def build():
    root = Path(__file__).resolve().parents[1] / "vllm_ascend/device_allocator"
    source = root / "camem_lifecycle.cpp"
    target = root / ("camem_lifecycle_C" + sysconfig.get_config_var("EXT_SUFFIX"))
    if target.exists() and target.stat().st_mtime >= source.stat().st_mtime:
        return target
    cann = Path(os.environ.get("ASCEND_HOME_PATH", "/usr/local/Ascend/ascend-toolkit/latest"))
    temporary = target.with_name(target.name + f".build-{os.getpid()}")
    subprocess.run(["g++", "-std=c++17", "-shared", "-fPIC", "-O2",
                    "-I" + sysconfig.get_path("include"), "-I" + str(cann / "include"),
                    str(source), "-L" + str(cann / "lib64"),
                    "-Wl,-rpath," + str(cann / "lib64"), "-lascendcl",
                    "-o", str(temporary)], check=True)
    temporary.replace(target)
    return target


if __name__ == "__main__":
    print(build())
