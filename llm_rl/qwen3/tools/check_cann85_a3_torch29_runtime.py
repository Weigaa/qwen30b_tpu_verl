#!/usr/bin/env python3
"""Fail closed when the Qwen3 vLLM 0.14 runtime is mixed or incomplete."""

from __future__ import annotations

import importlib.util
import json
import os
from importlib import metadata
from pathlib import Path
import re
import subprocess
import sys

from packaging.specifiers import SpecifierSet
from packaging.version import Version


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_NATIVE_SHA256 = {
    "vllm_ascend_C.cpython-311-aarch64-linux-gnu.so":
        "4d62a52033e9dca398c5713e51624e4c3631874da0dd77d7cfe302dcb8c7a7bd",
    "libvllm_ascend_kernels.so":
        "75266ebf055a6ace2ce0c80aa5e7c194517300ef84439b2ee93aed19f58f750d",
}
EXPECTED_PACKAGES = {
    "torch": "==2.9.0",
    "torch-npu": "==2.9.0",
    "torchvision": "==0.24.0",
    "vllm": "==0.14.1+empty",
    "vllm-ascend": "==0.14.0rc1",
    "numpy": "==1.26.4",
    "transformers": "==4.57.6",
    "compressed-tensors": "==0.13.0",
    "xgrammar": "==0.1.30",
    "opencv-python-headless": "==4.11.0.86",
    "fastapi": "==0.123.10",
    "starlette": "==0.50.0",
    "prometheus-fastapi-instrumentator": "==7.1.0",
    "lark": "==1.2.2",
    "pydantic": ">=2.12.0,<3",
    "depyf": "==0.20.0",
    "llguidance": ">=1.3.0,<1.4.0",
    "lm-format-enforcer": "==0.11.3",
    "outlines-core": "==0.2.11",
    "mistral-common": ">=1.8.8",
    "grpcio": ">=1.76.0,<2",
    "grpcio-reflection": ">=1.76.0,<2",
    "ray": ">=2.47.1,<=2.48.0",
    "triton-ascend": "==3.2.0",
    "arctic-inference": "==0.1.1",
    "anthropic": "==0.71.0",
    "model-hosting-container-standards": ">=0.1.10,<1",
}
REQUIRED_PACKAGES = (
    "pandas-stubs",
    "ml-dtypes",
    "tornado",
    "ijson",
    "mcp",
)
FORBIDDEN_PATH_PARTS = (
    "/usr/local/Ascend/ascend-toolkit/",
    "/usr/local/Ascend/nnal/",
    "/workspace/vllm",
    "/workspace/vllm-ascend",
    "/cann_master_9.1/",
    "/cann_8.5.0_a3_stable/",
    "/.venv_torch27_vllm014/",
)


def fail(message: str) -> None:
    raise RuntimeError(message)


def read(path: Path) -> str:
    if not path.is_file():
        fail(f"missing required runtime file: {path}")
    return path.read_text(errors="replace")


def check_package_versions() -> None:
    expected_venv = (ROOT / ".venv_cann85_a3_torch29").resolve()
    for name, specifier in EXPECTED_PACKAGES.items():
        try:
            distribution = metadata.distribution(name)
            installed = distribution.version
        except metadata.PackageNotFoundError:
            fail(f"missing Python package: {name}{specifier}")
        if Version(installed) not in SpecifierSet(specifier):
            fail(f"{name} mismatch: expected {specifier}, got {installed}")
        location = Path(distribution.locate_file("")).resolve()
        if not location.is_relative_to(expected_venv):
            fail(f"{name} metadata resolved outside the selected venv: {location}")
        direct_url = distribution._path / "direct_url.json"
        if direct_url.is_file():
            direct = json.loads(direct_url.read_text())
            if direct.get("dir_info", {}).get("editable"):
                fail(f"{name} is installed editable from {direct.get('url')}")
        print(f"[runtime audit] package {name}={installed}")
    for name in REQUIRED_PACKAGES:
        try:
            distribution = metadata.distribution(name)
            installed = distribution.version
        except metadata.PackageNotFoundError:
            fail(f"missing Python package: {name}")
        location = Path(distribution.locate_file("")).resolve()
        if not location.is_relative_to(expected_venv):
            fail(f"{name} metadata resolved outside the selected venv: {location}")
        print(f"[runtime audit] package {name}={installed}")


def check_local_imports() -> None:
    for package in ("verl", "vllm", "vllm_ascend"):
        spec = importlib.util.find_spec(package)
        origin = Path(spec.origin).resolve() if spec and spec.origin else None
        if origin is None or not origin.is_relative_to(ROOT):
            fail(f"{package} resolved outside this repository: {origin}")
        print(f"[runtime audit] local import {package}={origin}")

    source_contracts = {
        ROOT / "vllm/_version.py": ("0.14.1", "gd7de043d5"),
        ROOT / "vllm_ascend/_version.py": ("0.14.0rc1", "g52d4acfa5"),
    }
    for path, expected_values in source_contracts.items():
        source = read(path)
        missing = [value for value in expected_values if value not in source]
        if missing:
            fail(f"source version contract mismatch in {path}: missing {missing}")
        print(f"[runtime audit] source contract={path}: {', '.join(expected_values)}")


def check_environment() -> tuple[Path, Path, Path]:
    expected_stack = "vllm0141_vllm_ascend0140rc1_torch29_cann850_a3"
    if os.environ.get("QWEN3_RUNTIME_STACK") != expected_stack:
        fail("activate tools/activate_cann85_a3_torch29.sh before auditing")
    if os.environ.get("SOC_VERSION") != "ascend910_9392":
        fail(f"SOC_VERSION must match this host (ascend910_9392), got {os.environ.get('SOC_VERSION')!r}")
    if os.environ.get("HCCL_OP_EXPANSION_MODE") != "HOST":
        fail(
            "HCCL_OP_EXPANSION_MODE=HOST is required on this host: CANN 8.5 "
            "AIV expansion fails standalone HCCL collectives with error 507018"
        )
    expected_plugins = (
        "ascend,ascend_kv_connector,ascend_model_loader,"
        "ascend_service_profiling"
    )
    if os.environ.get("VLLM_PLUGINS") != expected_plugins:
        fail(f"VLLM_PLUGINS must be {expected_plugins}, got {os.environ.get('VLLM_PLUGINS')!r}")
    for name in ("VLLM_ASCEND_USE_LOCAL_CUSTOM_OPP", "VLLM_ASCEND_USE_LOCAL_CUSTOM_OP_API_LIB"):
        if os.environ.get(name) != "0":
            fail(f"{name}=0 must be set before imports")
    if os.environ.get("PYTHONNOUSERSITE") != "1":
        fail("PYTHONNOUSERSITE=1 is required to prevent user-site package leakage")
    if os.environ.get("RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES") != "1":
        fail(
            "RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1 is required "
            "for CANN 8.5 multi-rank HCCL"
        )
    if os.environ.get("ASCEND_RT_VISIBLE_DEVICES"):
        fail(
            "ASCEND_RT_VISIBLE_DEVICES must be unset when Ray no-set NPU "
            "binding is enabled"
        )

    expected_venv = (ROOT / ".venv_cann85_a3_torch29").resolve()
    actual_venv = Path(sys.prefix).resolve()
    if actual_venv != expected_venv:
        fail(f"wrong venv: expected {expected_venv}, got {actual_venv}")

    for variable in ("PATH", "PYTHONPATH", "LD_LIBRARY_PATH", "ASCEND_OPP_PATH"):
        value = os.environ.get(variable, "")
        for forbidden in FORBIDDEN_PATH_PARTS:
            if forbidden in value:
                fail(f"mixed runtime path in {variable}: found {forbidden} in {value}")

    cann_home = Path(os.environ["QWEN3_CANN85_HOME"]).resolve()
    atb_env = Path(os.environ["QWEN3_NNAL85_ATB_ENV"]).resolve()
    driver_root = Path(os.environ["QWEN3_DRIVER_ROOT"]).resolve()
    expected_cann = Path(
        "/workspace/cann-recipes-train/llm_rl/qwen3/cann_8.5.0_a3/cann-8.5.0"
    )
    if cann_home != expected_cann:
        fail(f"wrong CANN root: expected {expected_cann}, got {cann_home}")

    library_paths = os.environ.get("LD_LIBRARY_PATH", "").split(":")
    real_driver = str(driver_root / "lib64" / "driver")
    devlib_positions = [i for i, path in enumerate(library_paths) if path.endswith("/devlib")]
    if real_driver not in library_paths:
        fail(f"real driver library path is absent: {real_driver}")
    if devlib_positions and library_paths.index(real_driver) > min(devlib_positions):
        fail("CANN devlib stub precedes the real driver library")
    return cann_home, atb_env, driver_root


def check_cann(cann_home: Path, atb_env: Path, driver_root: Path) -> None:
    version_files = (
        cann_home / "share/info/runtime/version.info",
        cann_home / "share/info/hccl/version.info",
        cann_home / "opp/version.info",
    )
    for version_file in version_files:
        if "Version=8.5.0" not in read(version_file):
            fail(f"expected CANN 8.5.0 metadata in {version_file}")
        print(f"[runtime audit] CANN component={version_file}")

    atb_version = atb_env.parent / "8.5.0/version.info"
    if "Ascend-cann-atb Version : 8.5.0.B160" not in read(atb_version):
        fail(f"expected NNAL/ATB 8.5.0.B160 in {atb_version}")
    print(f"[runtime audit] NNAL/ATB={atb_version}: 8.5.0.B160")

    hccl = cann_home / "aarch64-linux/lib64/libhccl.so"
    if not hccl.is_file():
        fail(f"missing HCCL library: {hccl}")
    gcda = list(cann_home.rglob("*.gcda"))
    if gcda:
        fail(f"CANN tree contains compiler coverage state, first file: {gcda[0]}")
    dynamic = subprocess.run(
        ["readelf", "-d", str(hccl)], check=True, text=True, capture_output=True
    ).stdout
    if "jenkins" in dynamic.lower():
        fail(f"HCCL contains a build-host RUNPATH: {hccl}")

    driver_info = read(driver_root / "version.info")
    match = re.search(r"^package_version=(\S+)", driver_info, re.MULTILINE)
    driver_version = match.group(1) if match else "unknown"
    release = tuple(int(part) for part in re.findall(r"\d+", driver_version)[:2])
    supported_driver_lines = {(25, 0), (25, 2), (25, 3), (25, 5)}
    if (
        release not in supported_driver_lines
        and os.environ.get("QWEN3_ALLOW_UNVALIDATED_DRIVER") != "1"
    ):
        fail(
            "CANN 8.5.0 supports Ascend HDK 25.0.x, 25.2.x, 25.3.x and "
            f"25.5.x, but host driver is {driver_version}; install a supported "
            "driver before launching, or set "
            "QWEN3_ALLOW_UNVALIDATED_DRIVER=1 only for a diagnostic import check"
        )
    print(f"[runtime audit] driver package={driver_version}")


def check_ascend_python_runtime(cann_home: Path) -> None:
    for package in ("te", "tbe", "hccl"):
        spec = importlib.util.find_spec(package)
        origin = Path(spec.origin).resolve() if spec and spec.origin else None
        if origin is None or not origin.is_relative_to(cann_home):
            fail(f"{package} resolved outside the selected CANN runtime: {origin}")
        print(f"[runtime audit] CANN Python import {package}={origin}")

    import acl

    acl_origin = Path(acl.__file__).resolve()
    if not acl_origin.is_relative_to(ROOT / "acl"):
        fail(f"acl compatibility module resolved outside this repository: {acl_origin}")
    ascendcl = Path(acl.rt.runtime_library_path()).resolve()
    if not ascendcl.is_relative_to(cann_home):
        fail(f"acl compatibility module loaded a mixed CANN library: {ascendcl}")
    soc_name = acl.get_soc_name()
    if soc_name != "Ascend910_9392":
        fail(f"unexpected SoC from CANN runtime: {soc_name!r}")
    device_count, result = acl.rt.get_device_count()
    if result != 0 or device_count != 16:
        fail(
            "expected 16 NPUs from the selected CANN runtime, got "
            f"count={device_count}, result={result}"
        )
    print(
        f"[runtime audit] pyACL compatibility={acl_origin} "
        f"libascendcl={ascendcl} soc={soc_name} devices={device_count}"
    )


def check_worker_import_contract() -> None:
    imports = (
        "from mindspeed.core.transformer.moe.moe_feature.overlap."
        "moe_layer_overlap_allgather import MoELayerOverlapAllGather; "
        "from vllm_ascend.device_allocator.camem import CaMemAllocator; "
        "print('worker import contract ok')"
    )
    result = subprocess.run(
        [sys.executable, "-c", imports], text=True, capture_output=True
    )
    if result.returncode != 0:
        fail(
            "worker import contract failed before Ray launch:\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    print("[runtime audit] worker import contract: MindSpeed + CaMem ok")


def check_native_extension() -> None:
    native = ROOT / "vllm_ascend/vllm_ascend_C.cpython-311-aarch64-linux-gnu.so"
    if not native.is_file():
        fail(f"missing vLLM-Ascend native extension: {native}")
    import hashlib

    for artifact, expected in EXPECTED_NATIVE_SHA256.items():
        path = ROOT / "vllm_ascend" / artifact
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            fail(f"unexpected native artifact {path}: expected {expected}, got {actual}")
    ldd = subprocess.run(
        ["ldd", str(native)], check=True, text=True, capture_output=True
    ).stdout
    lowered = ldd.lower()
    if "not found" in lowered:
        fail(f"unresolved native dependency:\n{ldd}")
    for forbidden in FORBIDDEN_PATH_PARTS:
        if forbidden.lower() in lowered:
            fail(f"native extension resolves through a mixed path {forbidden}:\n{ldd}")

    import torch
    import torch_npu
    import vllm_ascend.vllm_ascend_C as native_ext

    native_origin = Path(native_ext.__file__).resolve()
    if native_origin != native.resolve():
        fail(f"wrong vLLM-Ascend native extension: {native_origin}")
    print(
        "[runtime audit] native import ok "
        f"torch={torch.__version__} torch_npu={torch_npu.__version__} so={native_origin}"
    )


def check_pip_consistency() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "pip", "check"], text=True, capture_output=True
    )
    if result.returncode == 0:
        print("[runtime audit] pip check: clean")
        return

    allowed = (
        "vllm 0.14.1+empty has requirement opencv-python-headless>=4.13.0",
        "vllm 0.14.1+empty has requirement xgrammar==0.1.29",
        "op-compile-tool 0.1.0 requires getopt",
        "op-compile-tool 0.1.0 requires inspect",
        "op-compile-tool 0.1.0 requires multiprocessing",
    )
    unexpected = [
        line
        for line in result.stdout.splitlines()
        if line.strip() and not line.startswith(allowed)
    ]
    if unexpected:
        fail("pip dependency check failed:\n" + "\n".join(unexpected))
    print("[runtime audit] pip check: only documented upstream conflicts remain")


def main() -> None:
    check_package_versions()
    check_local_imports()
    cann_home, atb_env, driver_root = check_environment()
    check_cann(cann_home, atb_env, driver_root)
    check_ascend_python_runtime(cann_home)
    check_native_extension()
    check_worker_import_contract()
    check_pip_consistency()
    print("[runtime audit] PASS: coherent vLLM 0.14 / Torch 2.9 / CANN 8.5 A3 stack")


if __name__ == "__main__":
    main()
