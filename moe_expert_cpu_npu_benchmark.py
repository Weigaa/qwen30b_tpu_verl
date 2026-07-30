#!/usr/bin/env python3
"""
Benchmark Qwen3 MoE expert MLP compute on Kunpeng CPU vs Ascend NPU.

The measured math is the rollout expert MLP core:

    expanded_hidden @ w13 -> SwiGLU -> @ w2

It intentionally excludes router top-k, all-to-all dispatch/combine, and CPU<->NPU
copies, so the result isolates expert-layer compute throughput.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import math
import multiprocessing as mp
import os
import platform
import statistics
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass
from typing import Iterable


MIB = 1024 * 1024

BUILTIN_QWEN3_30B_A3B_CONFIG = {
    "model_type": "qwen3_moe",
    "hidden_size": 2048,
    "intermediate_size": 6144,
    "moe_intermediate_size": 768,
    "num_experts": 128,
    "num_experts_per_tok": 8,
}


def find_default_model_config() -> str | None:
    env_candidates = [
        os.environ.get("MOE_BENCH_MODEL_CONFIG"),
        os.environ.get("MODEL_CONFIG"),
    ]
    for env_name in ("MODEL_PATH", "HF_MODEL_PATH"):
        env_path = os.environ.get(env_name)
        if env_path:
            env_candidates.append(os.path.join(env_path, "config.json"))

    candidates = [
        *env_candidates,
        "/home/data/Qwen3-30B-A3B/config.json",
        "/home/data/Qwen3-30B-A3B-Instruct-2507/config.json",
        "/data/Qwen3-30B-A3B/config.json",
        "/data/deepscaler/Qwen3-30B-A3B/config.json",
        find_repo_file("config.json"),
    ]
    for path in candidates:
        if path and os.path.exists(path):
            return path
    return None


def parse_int_list(spec: str) -> list[int]:
    values = [int(x.strip()) for x in spec.split(",") if x.strip()]
    if not values:
        raise argparse.ArgumentTypeError(f"empty integer list: {spec!r}")
    if any(v <= 0 for v in values):
        raise argparse.ArgumentTypeError(f"all values must be > 0: {spec!r}")
    return values


def parse_scope_list(spec: str) -> list[str]:
    scopes = [x.strip() for x in spec.split(",") if x.strip()]
    valid = {"ep-local", "full-layer"}
    bad = [x for x in scopes if x not in valid]
    if bad:
        raise argparse.ArgumentTypeError(
            f"invalid scope(s) {bad}; expected comma-separated subset of {sorted(valid)}"
        )
    return scopes


def parse_npu_list(spec: str) -> list[int]:
    if "," in spec:
        ids = [int(x.strip()) for x in spec.split(",") if x.strip()]
        if not ids:
            raise argparse.ArgumentTypeError(f"empty NPU list: {spec!r}")
        return ids
    count = int(spec)
    if count == 0:
        return [0]
    if count <= 0:
        raise argparse.ArgumentTypeError(f"NPU count must be > 0: {spec!r}")
    return list(range(count))


def parse_cpu_list(spec: str) -> list[int]:
    cpus: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_s, end_s = part.split("-", 1)
            start = int(start_s)
            end = int(end_s)
            if end < start:
                raise ValueError(f"invalid CPU range: {part}")
            cpus.extend(range(start, end + 1))
        else:
            cpus.append(int(part))
    return sorted(set(cpus))


def get_numa_cpu_lists() -> list[list[int]]:
    node_root = "/sys/devices/system/node"
    if not os.path.isdir(node_root):
        return []
    nodes = sorted(
        name for name in os.listdir(node_root)
        if name.startswith("node") and name[4:].isdigit()
    )
    cpu_lists: list[list[int]] = []
    for node in nodes:
        cpulist_path = os.path.join(node_root, node, "cpulist")
        try:
            with open(cpulist_path, "r", encoding="utf-8") as f:
                cpus = parse_cpu_list(f.read().strip())
        except Exception:
            cpus = []
        if cpus:
            cpu_lists.append(cpus)
    return cpu_lists


NUMA_CPU_BACKENDS = {
    "numpy-openblas-numa",
    "kml-sgemm-numa",
    "kml-sbgemm-numa",
    "kml-bgemm-batch-numa",
    "kml-bgemm-pack-numa",
}


def configure_worker_affinity_and_threads(cpu_ids: list[int], blas_threads: int) -> None:
    if cpu_ids:
        os.sched_setaffinity(0, set(cpu_ids))
        os.environ["GOMP_CPU_AFFINITY"] = " ".join(str(cpu) for cpu in cpu_ids)
    os.environ["OMP_NUM_THREADS"] = str(blas_threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(blas_threads)
    os.environ["GOTO_NUM_THREADS"] = str(blas_threads)
    # Let the explicit process affinity and GOMP_CPU_AFFINITY define placement.
    os.environ.pop("OMP_PROC_BIND", None)
    os.environ.pop("OMP_PLACES", None)


def split_evenly(total: int, parts: int) -> list[int]:
    base = total // parts
    rem = total % parts
    return [base + (1 if idx < rem else 0) for idx in range(parts)]


def find_default_kml_lib() -> str | None:
    env_path = os.environ.get("KML_BLAS_LIB")
    if env_path and os.path.exists(env_path):
        return env_path

    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(
            script_dir,
            "third_party/hpckit26/kml26_full/gcclib/sve/kblas/multi/libkblas.so.1.26.0.RC1",
        ),
        os.path.join(
            script_dir,
            "third_party/hpckit26/kml26_full/KunpengHPCKit-kml.26.0.RC1/gcclib/sve/kblas/multi/libkblas.so.1.26.0.RC1",
        ),
        os.path.join(
            script_dir,
            "third_party/hpckit26/kml26/gcclib/sve/kblas/multi/libkblas.so.1.26.0.RC1",
        ),
        os.path.join(
            script_dir,
            "third_party/hpckit26/kml26/KunpengHPCKit-kml.26.0.RC1/gcclib/sve/kblas/multi/libkblas.so.1.26.0.RC1",
        ),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return None


def find_repo_file(relative_path: str) -> str:
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), relative_path)


def find_default_kudnn_root() -> str | None:
    env_path = os.environ.get("KUDNN_ROOT")
    if env_path and os.path.exists(env_path):
        return env_path

    candidates = [
        find_repo_file("third_party/hpckit26/kudnn_gcc"),
        "/usr/local/kudnn",
        "/usr/local/KuDNN",
    ]
    for path in candidates:
        if os.path.exists(os.path.join(path, "lib", "libkudnn.so")):
            return path
    return None


def find_default_kupl_lib_dir() -> str | None:
    env_path = os.environ.get("KUPL_LIB_DIR")
    if env_path and os.path.exists(env_path):
        return env_path

    candidates = [
        find_repo_file("third_party/hpckit26/kupl_gcc/release/lib"),
        "/usr/local/kupl/release/lib",
        "/usr/local/kupl/lib",
    ]
    for path in candidates:
        if os.path.exists(os.path.join(path, "libkupl.so")):
            return path
    return None


def find_default_sdma_lib_dir() -> str | None:
    env_path = os.environ.get("SDMA_LIB_DIR")
    if env_path and os.path.exists(env_path):
        return env_path

    candidates = [
        find_repo_file("third_party/hpckit26"),
        "/usr/local/lib64",
        "/usr/local/lib",
        "/usr/lib64",
        "/lib64",
    ]
    for path in candidates:
        if os.path.exists(os.path.join(path, "libsdma_dk.so")):
            return path
    return None


def ensure_kudnn_worker(args: argparse.Namespace) -> str:
    worker_path = os.path.abspath(args.kudnn_worker)
    source_path = os.path.abspath(args.kudnn_worker_src)
    kudnn_root = args.kudnn_root or find_default_kudnn_root()
    kupl_lib_dir = args.kupl_lib_dir or find_default_kupl_lib_dir()
    kml_lib_path = args.kml_lib or find_default_kml_lib()
    sdma_lib_dir = args.sdma_lib_dir or find_default_sdma_lib_dir()

    missing = []
    if not kudnn_root:
        missing.append("KUDNN_ROOT/--kudnn-root")
    if not kupl_lib_dir:
        missing.append("KUPL_LIB_DIR/--kupl-lib-dir")
    if not kml_lib_path:
        missing.append("KML_BLAS_LIB/--kml-lib")
    if not sdma_lib_dir:
        missing.append("SDMA_LIB_DIR/--sdma-lib-dir")
    if missing:
        raise RuntimeError(f"missing KuDNN runtime inputs: {', '.join(missing)}")

    include_dir = os.path.join(kudnn_root, "include")
    kudnn_lib_dir = os.path.join(kudnn_root, "lib")
    kml_lib_dir = os.path.dirname(os.path.abspath(kml_lib_path))
    lib_dirs = [
        os.path.abspath(sdma_lib_dir),
        os.path.abspath(kudnn_lib_dir),
        os.path.abspath(kupl_lib_dir),
        os.path.abspath(kml_lib_dir),
    ]

    needs_build = args.rebuild_kudnn_worker or not os.path.exists(worker_path)
    if not needs_build:
        src_mtime = os.path.getmtime(source_path)
        needs_build = os.path.getmtime(worker_path) < src_mtime
    if needs_build:
        os.makedirs(os.path.dirname(worker_path) or ".", exist_ok=True)
        cmd = [
            args.cxx,
            "-O3",
            "-std=c++17",
            f"-I{include_dir}",
            source_path,
            f"-L{kudnn_lib_dir}",
            f"-L{kupl_lib_dir}",
            f"-L{kml_lib_dir}",
            f"-L{sdma_lib_dir}",
            "-lkudnn",
            "-lkupl",
            "-lkblas",
            "-lsdma_dk",
        ]
        for lib_dir in lib_dirs:
            cmd.append(f"-Wl,-rpath,{lib_dir}")
        cmd.extend(["-o", worker_path])
        subprocess.run(cmd, check=True)

    args.kudnn_root = os.path.abspath(kudnn_root)
    args.kupl_lib_dir = os.path.abspath(kupl_lib_dir)
    args.kml_lib = os.path.abspath(kml_lib_path)
    args.sdma_lib_dir = os.path.abspath(sdma_lib_dir)
    return worker_path


def read_model_config(path: str | None) -> tuple[dict, str]:
    if path:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f), os.path.abspath(path)
        print(
            f"[warn] model config not found: {path}; using built-in Qwen3-30B-A3B MoE sizes",
            file=sys.stderr,
        )
    return dict(BUILTIN_QWEN3_30B_A3B_CONFIG), "builtin:Qwen3-30B-A3B"


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * pct / 100.0
    lower = math.floor(pos)
    upper = math.ceil(pos)
    if lower == upper:
        return ordered[int(pos)]
    frac = pos - lower
    return ordered[lower] * (1.0 - frac) + ordered[upper] * frac


def format_float(value: float, digits: int = 3) -> str:
    if not math.isfinite(value):
        return "nan"
    return f"{value:.{digits}f}"


def dtype_nbytes(dtype_name: str) -> int:
    if dtype_name in {"float16", "bfloat16"}:
        return 2
    if dtype_name == "float32":
        return 4
    raise ValueError(f"unsupported dtype: {dtype_name}")


def torch_dtype(torch, dtype_name: str):
    if dtype_name == "bfloat16":
        return torch.bfloat16
    if dtype_name == "float16":
        return torch.float16
    if dtype_name == "float32":
        return torch.float32
    raise ValueError(f"unsupported dtype: {dtype_name}")


def run_command(args: list[str]) -> str:
    try:
        return subprocess.check_output(args, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return ""


def get_cpu_model() -> str:
    out = run_command(["lscpu"])
    for line in out.splitlines():
        if line.startswith("BIOS Model name:"):
            return line.split(":", 1)[1].strip()
        if line.startswith("Model name:"):
            value = line.split(":", 1)[1].strip()
            if value and value != "-":
                return value
    return platform.processor() or platform.machine()


def get_cpu_topology_summary() -> str:
    cpu_count = os.cpu_count() or len(os.sched_getaffinity(0))
    numa_nodes = get_numa_cpu_lists()
    if not numa_nodes:
        return f"{cpu_count} logical CPUs, NUMA unknown"
    node_sizes = ",".join(str(len(cpus)) for cpus in numa_nodes)
    return f"{cpu_count} logical CPUs, {len(numa_nodes)} NUMA nodes, cpus_per_node={node_sizes}"


def default_cpu_threads() -> list[int]:
    cpu_count = os.cpu_count() or len(os.sched_getaffinity(0))
    numa_nodes = get_numa_cpu_lists()
    node_size = min((len(cpus) for cpus in numa_nodes), default=cpu_count)
    candidates = [
        max(1, node_size // 2),
        node_size,
        node_size * 2,
        node_size * 4,
        cpu_count,
    ]
    return sorted({min(cpu_count, value) for value in candidates if value > 0})


def full_node_numa_thread_count(args: argparse.Namespace) -> int:
    cpu_lists = get_numa_cpu_lists()
    if not cpu_lists:
        return len(os.sched_getaffinity(0))
    if args.cpu_numa_nodes > 0:
        cpu_lists = cpu_lists[:args.cpu_numa_nodes]
    return sum(len(cpus) for cpus in cpu_lists)


def effective_cpu_thread_sweep(args: argparse.Namespace) -> list[int]:
    if args.cpu_backend in NUMA_CPU_BACKENDS and args.cpu_numa_layout == "full-node":
        return [full_node_numa_thread_count(args)]
    return list(args.cpu_threads)


@dataclass(frozen=True)
class BenchCase:
    scope: str
    hidden_size: int
    intermediate_size: int
    num_experts: int
    top_k: int
    ep_size: int
    num_tokens: int
    active_experts: int
    expanded_rows: int
    rows_per_expert: int

    @property
    def case_key(self) -> str:
        return f"{self.scope}/e{self.active_experts}/rpe{self.rows_per_expert}"

    @property
    def logical_flops(self) -> float:
        # Two GEMMs, FMA counted as 2 FLOPs:
        # w13: 2 * rows * H * (2I), w2: 2 * rows * I * H.
        return float(self.expanded_rows * 6 * self.hidden_size * self.intermediate_size)

    @property
    def weight_mib(self) -> float:
        element_bytes = dtype_nbytes("bfloat16")
        elements = self.active_experts * (
            self.hidden_size * 2 * self.intermediate_size
            + self.intermediate_size * self.hidden_size
        )
        return elements * element_bytes / MIB

    @property
    def activation_mib(self) -> float:
        element_bytes = dtype_nbytes("bfloat16")
        elements = self.expanded_rows * (
            self.hidden_size + 2 * self.intermediate_size + self.intermediate_size
            + self.hidden_size
        )
        return elements * element_bytes / MIB


@dataclass
class BenchResult:
    scope: str
    case_key: str
    device: str
    workers: str
    active_experts: int
    expanded_rows: int
    rows_per_expert: int
    median_ms: float
    min_ms: float
    p90_ms: float
    tflops: float


def make_case_for_tokens(scope: str, cfg: dict, args: argparse.Namespace) -> BenchCase:
    hidden_size = int(args.hidden_size or cfg.get("hidden_size", 2048))
    intermediate_size = int(args.moe_intermediate_size or cfg.get("moe_intermediate_size", 768))
    num_experts = int(args.num_experts or cfg.get("num_experts", 128))
    top_k = int(args.top_k or cfg.get("num_experts_per_tok", 8))
    ep_size = int(args.ep_size)
    num_tokens = int(args.tokens)

    if args.active_experts is not None:
        if len(args.active_experts) != 1:
            raise ValueError("--tokens mode accepts a single --active-experts value")
        active_experts = int(args.active_experts[0])
        if active_experts > num_experts:
            raise ValueError(
                f"active_experts={active_experts} cannot exceed num_experts={num_experts}")
        expanded_rows = num_tokens * top_k
    elif scope == "ep-local":
        if num_experts % ep_size:
            raise ValueError(f"num_experts={num_experts} must be divisible by ep_size={ep_size}")
        active_experts = num_experts // ep_size
        expanded_rows = num_tokens * top_k // ep_size
        if num_tokens * top_k % ep_size:
            raise ValueError("tokens * top_k must be divisible by ep_size for ep-local")
    elif scope == "full-layer":
        active_experts = num_experts
        expanded_rows = num_tokens * top_k
    else:
        raise ValueError(f"unsupported scope: {scope}")

    if expanded_rows % active_experts:
        raise ValueError(
            f"expanded_rows={expanded_rows} must be divisible by active_experts={active_experts}; "
            "choose a token count that gives balanced expert rows"
        )

    return BenchCase(
        scope=scope,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_experts=num_experts,
        top_k=top_k,
        ep_size=ep_size,
        num_tokens=num_tokens,
        active_experts=active_experts,
        expanded_rows=expanded_rows,
        rows_per_expert=expanded_rows // active_experts,
    )


def make_case_for_rows_per_expert(
    scope: str,
    rows_per_expert: int,
    active_experts_override: int | None,
    cfg: dict,
    args: argparse.Namespace,
) -> BenchCase:
    hidden_size = int(args.hidden_size or cfg.get("hidden_size", 2048))
    intermediate_size = int(args.moe_intermediate_size or cfg.get("moe_intermediate_size", 768))
    num_experts = int(args.num_experts or cfg.get("num_experts", 128))
    top_k = int(args.top_k or cfg.get("num_experts_per_tok", 8))
    ep_size = int(args.ep_size)

    if active_experts_override is not None:
        active_experts = int(active_experts_override)
        if active_experts > num_experts:
            raise ValueError(
                f"active_experts={active_experts} cannot exceed num_experts={num_experts}")
        expanded_rows = rows_per_expert * active_experts
        token_numer = expanded_rows
    elif scope == "ep-local":
        if num_experts % ep_size:
            raise ValueError(f"num_experts={num_experts} must be divisible by ep_size={ep_size}")
        active_experts = num_experts // ep_size
        expanded_rows = rows_per_expert * active_experts
        token_numer = expanded_rows * ep_size
    elif scope == "full-layer":
        active_experts = num_experts
        expanded_rows = rows_per_expert * active_experts
        token_numer = expanded_rows
    else:
        raise ValueError(f"unsupported scope: {scope}")

    if token_numer % top_k:
        raise ValueError(
            f"rows_per_expert={rows_per_expert} does not map to an integer "
            f"pre-routing token count for scope={scope}, top_k={top_k}, ep_size={ep_size}"
        )
    num_tokens = token_numer // top_k

    return BenchCase(
        scope=scope,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_experts=num_experts,
        top_k=top_k,
        ep_size=ep_size,
        num_tokens=num_tokens,
        active_experts=active_experts,
        expanded_rows=expanded_rows,
        rows_per_expert=rows_per_expert,
    )


def make_cases(cfg: dict, args: argparse.Namespace) -> list[BenchCase]:
    cases: list[BenchCase] = []
    for scope in args.scope:
        if args.rows_per_expert is None:
            cases.append(make_case_for_tokens(scope, cfg, args))
        else:
            active_expert_values = args.active_experts or [None]
            for active_experts in active_expert_values:
                for rows_per_expert in args.rows_per_expert:
                    cases.append(make_case_for_rows_per_expert(
                        scope, rows_per_expert, active_experts, cfg, args))
    return cases


def cpu_expert_mlp_bmm(torch, x, w13, w2):
    gate_up = torch.bmm(x, w13)
    gate, up = gate_up.chunk(2, dim=-1)
    activated = torch.nn.functional.silu(gate) * up
    return torch.bmm(activated, w2)


def cpu_expert_mlp_loop(torch, x, w13, w2):
    outputs = []
    for expert_idx in range(x.shape[0]):
        gate_up = x[expert_idx].matmul(w13[expert_idx])
        gate, up = gate_up.chunk(2, dim=-1)
        activated = torch.nn.functional.silu(gate) * up
        outputs.append(activated.matmul(w2[expert_idx]))
    return torch.stack(outputs, dim=0)


def run_openblas_child_from_stdin() -> int:
    import numpy as np

    payload = json.load(sys.stdin)
    active_experts = int(payload["active_experts"])
    rows_per_expert = int(payload["rows_per_expert"])
    hidden_size = int(payload["hidden_size"])
    intermediate_size = int(payload["intermediate_size"])
    warmup = int(payload["warmup"])
    iters = int(payload["iters"])

    shape_x = (rows_per_expert, hidden_size)
    shape_w13 = (hidden_size, 2 * intermediate_size)
    shape_w2 = (intermediate_size, hidden_size)

    # NumPy has no native bfloat16 matmul. This path is an optimistic CPU upper
    # bound using the fastest installed OpenBLAS FP32 GEMM backend. It runs in a
    # fresh Python process so OPENBLAS_NUM_THREADS is applied before NumPy loads.
    xs = [
        (np.random.randn(*shape_x).astype(np.float32) * np.float32(0.01))
        for _ in range(active_experts)
    ]
    w13 = [
        (np.random.randn(*shape_w13).astype(np.float32) * np.float32(0.01))
        for _ in range(active_experts)
    ]
    w2 = [
        (np.random.randn(*shape_w2).astype(np.float32) * np.float32(0.01))
        for _ in range(active_experts)
    ]

    def fn():
        outputs = []
        for expert_idx in range(active_experts):
            gate_up = xs[expert_idx] @ w13[expert_idx]
            gate = gate_up[:, :intermediate_size]
            up = gate_up[:, intermediate_size:]
            activated = (gate / (1.0 + np.exp(-gate))) * up
            outputs.append(activated @ w2[expert_idx])
        return outputs

    out = None
    for _ in range(warmup):
        out = fn()
    times_ms: list[float] = []
    for _ in range(iters):
        t0 = time.perf_counter()
        out = fn()
        t1 = time.perf_counter()
        times_ms.append((t1 - t0) * 1000.0)

    valid = (
        out is not None
        and len(out) == active_experts
        and out[0].shape == (rows_per_expert, hidden_size)
    )
    if not valid:
        raise RuntimeError("NumPy/OpenBLAS output shape validation failed")

    json.dump(
        {
            "times_ms": times_ms,
            "openblas_threads_env": os.environ.get("OPENBLAS_NUM_THREADS"),
        },
        sys.stdout,
    )
    sys.stdout.write("\n")
    return 0


def bench_cpu_numpy_openblas(case: BenchCase, args: argparse.Namespace, threads: int) -> BenchResult:
    payload = {
        "active_experts": case.active_experts,
        "rows_per_expert": case.rows_per_expert,
        "hidden_size": case.hidden_size,
        "intermediate_size": case.intermediate_size,
        "warmup": args.warmup,
        "iters": args.iters,
    }
    env = os.environ.copy()
    env["OPENBLAS_NUM_THREADS"] = str(threads)
    env["OMP_NUM_THREADS"] = str(threads)
    env["GOTO_NUM_THREADS"] = str(threads)
    proc = subprocess.run(
        [sys.executable, os.path.abspath(__file__), "--_openblas_child"],
        input=json.dumps(payload),
        text=True,
        capture_output=True,
        env=env,
        check=True,
    )
    child_result = json.loads(proc.stdout)
    times_ms = [float(value) for value in child_result["times_ms"]]
    median_ms = statistics.median(times_ms)
    return BenchResult(
        scope=case.scope,
        case_key=case.case_key,
        device="cpu-openblas-fp32",
        workers=str(threads),
        active_experts=case.active_experts,
        expanded_rows=case.expanded_rows,
        rows_per_expert=case.rows_per_expert,
        median_ms=median_ms,
        min_ms=min(times_ms),
        p90_ms=percentile(times_ms, 90),
        tflops=case.logical_flops / (median_ms / 1000.0) / 1e12,
    )


def numa_openblas_worker(
    worker_idx: int,
    cpu_ids: list[int],
    local_experts: int,
    rows_per_expert: int,
    hidden_size: int,
    intermediate_size: int,
    blas_threads: int,
    task_queue,
    result_queue,
) -> None:
    try:
        configure_worker_affinity_and_threads(cpu_ids, blas_threads)

        import numpy as np

        shape_x = (rows_per_expert, hidden_size)
        shape_w13 = (hidden_size, 2 * intermediate_size)
        shape_w2 = (intermediate_size, hidden_size)
        xs = [
            (np.random.randn(*shape_x).astype(np.float32) * np.float32(0.01))
            for _ in range(local_experts)
        ]
        w13 = [
            (np.random.randn(*shape_w13).astype(np.float32) * np.float32(0.01))
            for _ in range(local_experts)
        ]
        w2 = [
            (np.random.randn(*shape_w2).astype(np.float32) * np.float32(0.01))
            for _ in range(local_experts)
        ]

        def run_once():
            checksum = np.float32(0.0)
            for expert_idx in range(local_experts):
                gate_up = xs[expert_idx] @ w13[expert_idx]
                gate = gate_up[:, :intermediate_size]
                up = gate_up[:, intermediate_size:]
                activated = (gate / (1.0 + np.exp(-gate))) * up
                out = activated @ w2[expert_idx]
                checksum += out[0, 0]
            return float(checksum)

        result_queue.put({
            "worker": worker_idx,
            "event": "ready",
            "local_experts": local_experts,
            "cpu_count": len(cpu_ids),
            "blas_threads": blas_threads,
        })

        while True:
            cmd = task_queue.get()
            if cmd == "stop":
                break
            if cmd != "run":
                raise ValueError(f"unknown worker command: {cmd!r}")
            checksum = run_once()
            result_queue.put({
                "worker": worker_idx,
                "event": "done",
                "checksum": checksum,
            })
    except Exception:
        result_queue.put({
            "worker": worker_idx,
            "event": "error",
            "traceback": traceback.format_exc(),
        })


def numa_kml_sgemm_worker(
    worker_idx: int,
    cpu_ids: list[int],
    local_experts: int,
    rows_per_expert: int,
    hidden_size: int,
    intermediate_size: int,
    blas_threads: int,
    kml_lib_path: str,
    task_queue,
    result_queue,
) -> None:
    try:
        configure_worker_affinity_and_threads(cpu_ids, blas_threads)

        import numpy as np

        lib = ctypes.CDLL(kml_lib_path)
        lib.BlasSetNumThreads.argtypes = [ctypes.c_int]
        lib.BlasSetNumThreads.restype = None
        lib.BlasGetNumThreads.argtypes = []
        lib.BlasGetNumThreads.restype = ctypes.c_int
        lib.BlasGetParallel.argtypes = []
        lib.BlasGetParallel.restype = ctypes.c_int
        lib.cblas_sgemm.argtypes = [
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_float,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_float,
            ctypes.c_void_p,
            ctypes.c_int,
        ]
        lib.cblas_sgemm.restype = None
        lib.BlasSetNumThreads(int(blas_threads))

        cblas_row_major = 101
        cblas_no_trans = 111
        shape_x = (rows_per_expert, hidden_size)
        shape_w13 = (hidden_size, 2 * intermediate_size)
        shape_gate_up = (rows_per_expert, 2 * intermediate_size)
        shape_act = (rows_per_expert, intermediate_size)
        shape_w2 = (intermediate_size, hidden_size)
        shape_out = (rows_per_expert, hidden_size)
        xs = [
            np.ascontiguousarray(
                np.random.randn(*shape_x).astype(np.float32) * np.float32(0.01)
            )
            for _ in range(local_experts)
        ]
        w13 = [
            np.ascontiguousarray(
                np.random.randn(*shape_w13).astype(np.float32) * np.float32(0.01)
            )
            for _ in range(local_experts)
        ]
        gate_ups = [np.empty(shape_gate_up, dtype=np.float32) for _ in range(local_experts)]
        activations = [np.empty(shape_act, dtype=np.float32) for _ in range(local_experts)]
        w2 = [
            np.ascontiguousarray(
                np.random.randn(*shape_w2).astype(np.float32) * np.float32(0.01)
            )
            for _ in range(local_experts)
        ]
        outs = [np.empty(shape_out, dtype=np.float32) for _ in range(local_experts)]

        def ptr(array) -> int:
            return int(array.ctypes.data)

        def sgemm(m: int, n: int, k: int, a, lda: int, b, ldb: int, c, ldc: int) -> None:
            lib.cblas_sgemm(
                cblas_row_major,
                cblas_no_trans,
                cblas_no_trans,
                m,
                n,
                k,
                ctypes.c_float(1.0),
                ptr(a),
                lda,
                ptr(b),
                ldb,
                ctypes.c_float(0.0),
                ptr(c),
                ldc,
            )

        def run_once():
            checksum = np.float32(0.0)
            for expert_idx in range(local_experts):
                sgemm(
                    rows_per_expert,
                    2 * intermediate_size,
                    hidden_size,
                    xs[expert_idx],
                    hidden_size,
                    w13[expert_idx],
                    2 * intermediate_size,
                    gate_ups[expert_idx],
                    2 * intermediate_size,
                )
                gate = gate_ups[expert_idx][:, :intermediate_size]
                up = gate_ups[expert_idx][:, intermediate_size:]
                np.multiply(gate / (1.0 + np.exp(-gate)), up, out=activations[expert_idx])
                sgemm(
                    rows_per_expert,
                    hidden_size,
                    intermediate_size,
                    activations[expert_idx],
                    intermediate_size,
                    w2[expert_idx],
                    hidden_size,
                    outs[expert_idx],
                    hidden_size,
                )
                checksum += outs[expert_idx][0, 0]
            return float(checksum)

        result_queue.put({
            "worker": worker_idx,
            "event": "ready",
            "local_experts": local_experts,
            "cpu_count": len(cpu_ids),
            "blas_threads": lib.BlasGetNumThreads(),
            "blas_parallel": lib.BlasGetParallel(),
            "kml_lib": kml_lib_path,
        })

        while True:
            cmd = task_queue.get()
            if cmd == "stop":
                break
            if cmd != "run":
                raise ValueError(f"unknown worker command: {cmd!r}")
            checksum = run_once()
            result_queue.put({
                "worker": worker_idx,
                "event": "done",
                "checksum": checksum,
            })
    except Exception:
        result_queue.put({
            "worker": worker_idx,
            "event": "error",
            "traceback": traceback.format_exc(),
        })


def numa_kml_sbgemm_worker(
    worker_idx: int,
    cpu_ids: list[int],
    local_experts: int,
    rows_per_expert: int,
    hidden_size: int,
    intermediate_size: int,
    blas_threads: int,
    kml_lib_path: str,
    task_queue,
    result_queue,
) -> None:
    try:
        configure_worker_affinity_and_threads(cpu_ids, blas_threads)

        import numpy as np

        lib = ctypes.CDLL(kml_lib_path)
        lib.BlasSetNumThreads.argtypes = [ctypes.c_int]
        lib.BlasSetNumThreads.restype = None
        lib.BlasGetNumThreads.argtypes = []
        lib.BlasGetNumThreads.restype = ctypes.c_int
        lib.BlasGetParallel.argtypes = []
        lib.BlasGetParallel.restype = ctypes.c_int
        lib.cblas_sbgemm.argtypes = [
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_float,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_float,
            ctypes.c_void_p,
            ctypes.c_int,
        ]
        lib.cblas_sbgemm.restype = None
        lib.BlasSetNumThreads(int(blas_threads))

        cblas_row_major = 101
        cblas_no_trans = 111

        def f32_to_bf16_array(values):
            arr = np.ascontiguousarray(values.astype(np.float32, copy=False))
            return np.ascontiguousarray((arr.view(np.uint32) >> 16).astype(np.uint16))

        def f32_to_bf16_into(src, dst):
            arr = np.ascontiguousarray(src.astype(np.float32, copy=False))
            np.right_shift(arr.view(np.uint32), 16, out=dst)

        shape_x = (rows_per_expert, hidden_size)
        shape_w13 = (hidden_size, 2 * intermediate_size)
        shape_gate_up = (rows_per_expert, 2 * intermediate_size)
        shape_act = (rows_per_expert, intermediate_size)
        shape_w2 = (intermediate_size, hidden_size)
        shape_out = (rows_per_expert, hidden_size)
        xs = [
            f32_to_bf16_array(
                np.random.randn(*shape_x).astype(np.float32) * np.float32(0.01)
            )
            for _ in range(local_experts)
        ]
        w13 = [
            f32_to_bf16_array(
                np.random.randn(*shape_w13).astype(np.float32) * np.float32(0.01)
            )
            for _ in range(local_experts)
        ]
        gate_ups = [np.empty(shape_gate_up, dtype=np.float32) for _ in range(local_experts)]
        activations = [np.empty(shape_act, dtype=np.float32) for _ in range(local_experts)]
        activations_bf16 = [np.empty(shape_act, dtype=np.uint16) for _ in range(local_experts)]
        w2 = [
            f32_to_bf16_array(
                np.random.randn(*shape_w2).astype(np.float32) * np.float32(0.01)
            )
            for _ in range(local_experts)
        ]
        outs = [np.empty(shape_out, dtype=np.float32) for _ in range(local_experts)]

        def ptr(array) -> int:
            return int(array.ctypes.data)

        def sbgemm(m: int, n: int, k: int, a, lda: int, b, ldb: int, c, ldc: int) -> None:
            lib.cblas_sbgemm(
                cblas_row_major,
                cblas_no_trans,
                cblas_no_trans,
                m,
                n,
                k,
                ctypes.c_float(1.0),
                ptr(a),
                lda,
                ptr(b),
                ldb,
                ctypes.c_float(0.0),
                ptr(c),
                ldc,
            )

        def run_once():
            checksum = np.float32(0.0)
            for expert_idx in range(local_experts):
                sbgemm(
                    rows_per_expert,
                    2 * intermediate_size,
                    hidden_size,
                    xs[expert_idx],
                    hidden_size,
                    w13[expert_idx],
                    2 * intermediate_size,
                    gate_ups[expert_idx],
                    2 * intermediate_size,
                )
                gate = gate_ups[expert_idx][:, :intermediate_size]
                up = gate_ups[expert_idx][:, intermediate_size:]
                np.multiply(gate / (1.0 + np.exp(-gate)), up, out=activations[expert_idx])
                f32_to_bf16_into(activations[expert_idx], activations_bf16[expert_idx])
                sbgemm(
                    rows_per_expert,
                    hidden_size,
                    intermediate_size,
                    activations_bf16[expert_idx],
                    intermediate_size,
                    w2[expert_idx],
                    hidden_size,
                    outs[expert_idx],
                    hidden_size,
                )
                checksum += outs[expert_idx][0, 0]
            return float(checksum)

        result_queue.put({
            "worker": worker_idx,
            "event": "ready",
            "local_experts": local_experts,
            "cpu_count": len(cpu_ids),
            "blas_threads": lib.BlasGetNumThreads(),
            "blas_parallel": lib.BlasGetParallel(),
            "kml_lib": kml_lib_path,
        })

        while True:
            cmd = task_queue.get()
            if cmd == "stop":
                break
            if cmd != "run":
                raise ValueError(f"unknown worker command: {cmd!r}")
            checksum = run_once()
            result_queue.put({
                "worker": worker_idx,
                "event": "done",
                "checksum": checksum,
            })
    except Exception:
        result_queue.put({
            "worker": worker_idx,
            "event": "error",
            "traceback": traceback.format_exc(),
        })


def numa_kml_bgemm_batch_worker(
    worker_idx: int,
    cpu_ids: list[int],
    local_experts: int,
    rows_per_expert: int,
    hidden_size: int,
    intermediate_size: int,
    blas_threads: int,
    kml_lib_path: str,
    task_queue,
    result_queue,
) -> None:
    try:
        configure_worker_affinity_and_threads(cpu_ids, blas_threads)

        import numpy as np

        lib = ctypes.CDLL(kml_lib_path)
        lib.BlasSetNumThreads.argtypes = [ctypes.c_int]
        lib.BlasSetNumThreads.restype = None
        lib.BlasGetNumThreads.argtypes = []
        lib.BlasGetNumThreads.restype = ctypes.c_int
        lib.BlasGetParallel.argtypes = []
        lib.BlasGetParallel.restype = ctypes.c_int
        lib.cblas_bgemm_batch.argtypes = [
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_uint16),
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_uint16),
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_int),
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_int),
        ]
        lib.cblas_bgemm_batch.restype = None
        lib.BlasSetNumThreads(int(blas_threads))

        cblas_row_major = 101
        cblas_no_trans = 111

        def f32_to_bf16_scalar(value: float) -> int:
            return int((np.array([value], dtype=np.float32).view(np.uint32)[0] >> 16) & 0xFFFF)

        def f32_to_bf16_array(values):
            arr = np.ascontiguousarray(values.astype(np.float32, copy=False))
            return np.ascontiguousarray((arr.view(np.uint32) >> 16).astype(np.uint16))

        def f32_to_bf16_into(src, dst):
            arr = np.ascontiguousarray(src.astype(np.float32, copy=False))
            np.right_shift(arr.view(np.uint32), 16, out=dst)

        def bf16_to_f32_array(src, dst):
            src32 = np.asarray(src, dtype=np.uint16).astype(np.uint32)
            np.left_shift(src32, 16, out=dst.view(np.uint32))

        shape_x = (rows_per_expert, hidden_size)
        shape_w13 = (hidden_size, 2 * intermediate_size)
        shape_gate_up = (rows_per_expert, 2 * intermediate_size)
        shape_act = (rows_per_expert, intermediate_size)
        shape_w2 = (intermediate_size, hidden_size)
        shape_out = (rows_per_expert, hidden_size)
        xs = [
            f32_to_bf16_array(
                np.random.randn(*shape_x).astype(np.float32) * np.float32(0.01)
            )
            for _ in range(local_experts)
        ]
        w13 = [
            f32_to_bf16_array(
                np.random.randn(*shape_w13).astype(np.float32) * np.float32(0.01)
            )
            for _ in range(local_experts)
        ]
        gate_ups_bf16 = [np.empty(shape_gate_up, dtype=np.uint16) for _ in range(local_experts)]
        gate_ups = [np.empty(shape_gate_up, dtype=np.float32) for _ in range(local_experts)]
        activations = [np.empty(shape_act, dtype=np.float32) for _ in range(local_experts)]
        activations_bf16 = [np.empty(shape_act, dtype=np.uint16) for _ in range(local_experts)]
        w2 = [
            f32_to_bf16_array(
                np.random.randn(*shape_w2).astype(np.float32) * np.float32(0.01)
            )
            for _ in range(local_experts)
        ]
        outs_bf16 = [np.empty(shape_out, dtype=np.uint16) for _ in range(local_experts)]
        outs_head = np.empty((1,), dtype=np.float32)

        trans_a = (ctypes.c_int * 1)(cblas_no_trans)
        trans_b = (ctypes.c_int * 1)(cblas_no_trans)
        group_size = (ctypes.c_int * 1)(local_experts)
        alpha = (ctypes.c_uint16 * 1)(f32_to_bf16_scalar(1.0))
        beta = (ctypes.c_uint16 * 1)(f32_to_bf16_scalar(0.0))
        m_w13 = (ctypes.c_int * 1)(rows_per_expert)
        n_w13 = (ctypes.c_int * 1)(2 * intermediate_size)
        k_w13 = (ctypes.c_int * 1)(hidden_size)
        lda_w13 = (ctypes.c_int * 1)(hidden_size)
        ldb_w13 = (ctypes.c_int * 1)(2 * intermediate_size)
        ldc_w13 = (ctypes.c_int * 1)(2 * intermediate_size)
        m_w2 = (ctypes.c_int * 1)(rows_per_expert)
        n_w2 = (ctypes.c_int * 1)(hidden_size)
        k_w2 = (ctypes.c_int * 1)(intermediate_size)
        lda_w2 = (ctypes.c_int * 1)(intermediate_size)
        ldb_w2 = (ctypes.c_int * 1)(hidden_size)
        ldc_w2 = (ctypes.c_int * 1)(hidden_size)

        def ptr_array(arrays):
            return (ctypes.c_void_p * len(arrays))(
                *(ctypes.c_void_p(int(array.ctypes.data)) for array in arrays)
            )

        a_w13 = ptr_array(xs)
        b_w13 = ptr_array(w13)
        c_w13 = ptr_array(gate_ups_bf16)
        b_w2 = ptr_array(w2)
        c_w2 = ptr_array(outs_bf16)

        def bgemm_batch_w13() -> None:
            lib.cblas_bgemm_batch(
                cblas_row_major,
                trans_a,
                trans_b,
                m_w13,
                n_w13,
                k_w13,
                alpha,
                a_w13,
                lda_w13,
                b_w13,
                ldb_w13,
                beta,
                c_w13,
                ldc_w13,
                1,
                group_size,
            )

        def bgemm_batch_w2() -> None:
            lib.cblas_bgemm_batch(
                cblas_row_major,
                trans_a,
                trans_b,
                m_w2,
                n_w2,
                k_w2,
                alpha,
                ptr_array(activations_bf16),
                lda_w2,
                b_w2,
                ldb_w2,
                beta,
                c_w2,
                ldc_w2,
                1,
                group_size,
            )

        def run_once():
            checksum = np.float32(0.0)
            bgemm_batch_w13()
            for expert_idx in range(local_experts):
                bf16_to_f32_array(gate_ups_bf16[expert_idx], gate_ups[expert_idx])
                gate = gate_ups[expert_idx][:, :intermediate_size]
                up = gate_ups[expert_idx][:, intermediate_size:]
                np.multiply(gate / (1.0 + np.exp(-gate)), up, out=activations[expert_idx])
                f32_to_bf16_into(activations[expert_idx], activations_bf16[expert_idx])
            bgemm_batch_w2()
            for expert_idx in range(local_experts):
                bf16_to_f32_array(outs_bf16[expert_idx][:1, :1], outs_head)
                checksum += outs_head[0]
            return float(checksum)

        result_queue.put({
            "worker": worker_idx,
            "event": "ready",
            "local_experts": local_experts,
            "cpu_count": len(cpu_ids),
            "blas_threads": lib.BlasGetNumThreads(),
            "blas_parallel": lib.BlasGetParallel(),
            "kml_lib": kml_lib_path,
        })

        while True:
            cmd = task_queue.get()
            if cmd == "stop":
                break
            if cmd != "run":
                raise ValueError(f"unknown worker command: {cmd!r}")
            checksum = run_once()
            result_queue.put({
                "worker": worker_idx,
                "event": "done",
                "checksum": checksum,
            })
    except Exception:
        result_queue.put({
            "worker": worker_idx,
            "event": "error",
            "traceback": traceback.format_exc(),
        })


def numa_kml_bgemm_pack_worker(
    worker_idx: int,
    cpu_ids: list[int],
    local_experts: int,
    rows_per_expert: int,
    hidden_size: int,
    intermediate_size: int,
    blas_threads: int,
    kml_lib_path: str,
    task_queue,
    result_queue,
) -> None:
    try:
        configure_worker_affinity_and_threads(cpu_ids, blas_threads)

        import numpy as np

        lib = ctypes.CDLL(kml_lib_path)
        lib.BlasSetNumThreads.argtypes = [ctypes.c_int]
        lib.BlasSetNumThreads.restype = None
        lib.BlasGetNumThreads.argtypes = []
        lib.BlasGetNumThreads.restype = ctypes.c_int
        lib.BlasGetParallel.argtypes = []
        lib.BlasGetParallel.restype = ctypes.c_int
        lib.cblas_bgemm.argtypes = [
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_uint16,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_uint16,
            ctypes.c_void_p,
            ctypes.c_int,
        ]
        lib.cblas_bgemm.restype = None
        lib.cblas_bgemm_pack_get_size.argtypes = [
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
        ]
        lib.cblas_bgemm_pack_get_size.restype = ctypes.c_size_t
        lib.cblas_bgemm_pack.argtypes = [
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_void_p,
        ]
        lib.cblas_bgemm_pack.restype = None
        lib.cblas_bgemm_compute.argtypes = [
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_uint16,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_uint16,
            ctypes.c_void_p,
            ctypes.c_int,
        ]
        lib.cblas_bgemm_compute.restype = None
        lib.BlasSetNumThreads(int(blas_threads))

        cblas_row_major = 101
        cblas_no_trans = 111
        cblas_b = 152

        def f32_to_bf16_scalar(value: float) -> int:
            return int((np.array([value], dtype=np.float32).view(np.uint32)[0] >> 16) & 0xFFFF)

        def f32_to_bf16_array(values):
            arr = np.ascontiguousarray(values.astype(np.float32, copy=False))
            return np.ascontiguousarray((arr.view(np.uint32) >> 16).astype(np.uint16))

        def f32_to_bf16_into(src, dst):
            arr = np.ascontiguousarray(src.astype(np.float32, copy=False))
            np.right_shift(arr.view(np.uint32), 16, out=dst)

        def bf16_to_f32_array(src, dst):
            src32 = np.asarray(src, dtype=np.uint16).astype(np.uint32)
            np.left_shift(src32, 16, out=dst.view(np.uint32))

        shape_x = (rows_per_expert, hidden_size)
        shape_w13 = (hidden_size, 2 * intermediate_size)
        shape_gate_up = (rows_per_expert, 2 * intermediate_size)
        shape_act = (rows_per_expert, intermediate_size)
        shape_w2 = (intermediate_size, hidden_size)
        shape_out = (rows_per_expert, hidden_size)
        xs = [
            f32_to_bf16_array(
                np.random.randn(*shape_x).astype(np.float32) * np.float32(0.01)
            )
            for _ in range(local_experts)
        ]
        w13 = [
            f32_to_bf16_array(
                np.random.randn(*shape_w13).astype(np.float32) * np.float32(0.01)
            )
            for _ in range(local_experts)
        ]
        gate_ups_bf16 = [np.empty(shape_gate_up, dtype=np.uint16) for _ in range(local_experts)]
        gate_ups = [np.empty(shape_gate_up, dtype=np.float32) for _ in range(local_experts)]
        activations = [np.empty(shape_act, dtype=np.float32) for _ in range(local_experts)]
        activations_bf16 = [np.empty(shape_act, dtype=np.uint16) for _ in range(local_experts)]
        w2 = [
            f32_to_bf16_array(
                np.random.randn(*shape_w2).astype(np.float32) * np.float32(0.01)
            )
            for _ in range(local_experts)
        ]
        outs_bf16 = [np.empty(shape_out, dtype=np.uint16) for _ in range(local_experts)]
        outs_head = np.empty((1,), dtype=np.float32)

        alpha = f32_to_bf16_scalar(1.0)
        beta = f32_to_bf16_scalar(0.0)

        def ptr(array) -> int:
            return int(array.ctypes.data)

        def pack_b(weight, m: int, n: int, k: int, ld: int):
            packed_elems = lib.cblas_bgemm_pack_get_size(cblas_b, m, n, k)
            packed = np.empty(packed_elems, dtype=np.uint16)
            lib.cblas_bgemm_pack(
                cblas_row_major,
                cblas_b,
                cblas_no_trans,
                m,
                n,
                k,
                ptr(weight),
                ld,
                ptr(packed),
            )
            return packed

        packed_w13 = [
            pack_b(weight, rows_per_expert, 2 * intermediate_size, hidden_size, 2 * intermediate_size)
            for weight in w13
        ]
        packed_w2 = [
            pack_b(weight, rows_per_expert, hidden_size, intermediate_size, hidden_size)
            for weight in w2
        ]

        def bgemm_compute(m: int, n: int, k: int, a, lda: int, packed_b, c, ldc: int) -> None:
            lib.cblas_bgemm_compute(
                cblas_row_major,
                cblas_no_trans,
                cblas_no_trans,
                m,
                n,
                k,
                alpha,
                ptr(a),
                lda,
                ptr(packed_b),
                n,
                beta,
                ptr(c),
                ldc,
            )

        def run_once():
            checksum = np.float32(0.0)
            for expert_idx in range(local_experts):
                bgemm_compute(
                    rows_per_expert,
                    2 * intermediate_size,
                    hidden_size,
                    xs[expert_idx],
                    hidden_size,
                    packed_w13[expert_idx],
                    gate_ups_bf16[expert_idx],
                    2 * intermediate_size,
                )
                bf16_to_f32_array(gate_ups_bf16[expert_idx], gate_ups[expert_idx])
                gate = gate_ups[expert_idx][:, :intermediate_size]
                up = gate_ups[expert_idx][:, intermediate_size:]
                np.multiply(gate / (1.0 + np.exp(-gate)), up, out=activations[expert_idx])
                f32_to_bf16_into(activations[expert_idx], activations_bf16[expert_idx])
                bgemm_compute(
                    rows_per_expert,
                    hidden_size,
                    intermediate_size,
                    activations_bf16[expert_idx],
                    intermediate_size,
                    packed_w2[expert_idx],
                    outs_bf16[expert_idx],
                    hidden_size,
                )
                bf16_to_f32_array(outs_bf16[expert_idx][:1, :1], outs_head)
                checksum += outs_head[0]
            return float(checksum)

        result_queue.put({
            "worker": worker_idx,
            "event": "ready",
            "local_experts": local_experts,
            "cpu_count": len(cpu_ids),
            "blas_threads": lib.BlasGetNumThreads(),
            "blas_parallel": lib.BlasGetParallel(),
            "kml_lib": kml_lib_path,
        })

        while True:
            cmd = task_queue.get()
            if cmd == "stop":
                break
            if cmd != "run":
                raise ValueError(f"unknown worker command: {cmd!r}")
            checksum = run_once()
            result_queue.put({
                "worker": worker_idx,
                "event": "done",
                "checksum": checksum,
            })
    except Exception:
        result_queue.put({
            "worker": worker_idx,
            "event": "error",
            "traceback": traceback.format_exc(),
        })


def bench_cpu_numpy_openblas_numa(
    case: BenchCase,
    args: argparse.Namespace,
    total_threads: int,
) -> BenchResult:
    return bench_cpu_numa_processes(
        case,
        args,
        total_threads,
        device="cpu-openblas-numa-fp32",
        worker_target=numa_openblas_worker,
        extra_worker_args=(),
    )


def bench_cpu_kml_sgemm_numa(
    case: BenchCase,
    args: argparse.Namespace,
    total_threads: int,
) -> BenchResult:
    kml_lib_path = args.kml_lib or find_default_kml_lib()
    if not kml_lib_path:
        raise RuntimeError(
            "KML libkblas.so not found. Pass --kml-lib or set KML_BLAS_LIB."
        )
    return bench_cpu_numa_processes(
        case,
        args,
        total_threads,
        device="cpu-kml-sgemm-numa-fp32",
        worker_target=numa_kml_sgemm_worker,
        extra_worker_args=(os.path.abspath(kml_lib_path),),
    )


def bench_cpu_kml_sbgemm_numa(
    case: BenchCase,
    args: argparse.Namespace,
    total_threads: int,
) -> BenchResult:
    kml_lib_path = args.kml_lib or find_default_kml_lib()
    if not kml_lib_path:
        raise RuntimeError(
            "KML libkblas.so not found. Pass --kml-lib or set KML_BLAS_LIB."
        )
    return bench_cpu_numa_processes(
        case,
        args,
        total_threads,
        device="cpu-kml-sbgemm-numa-bf16-fp32",
        worker_target=numa_kml_sbgemm_worker,
        extra_worker_args=(os.path.abspath(kml_lib_path),),
    )


def bench_cpu_kml_bgemm_batch_numa(
    case: BenchCase,
    args: argparse.Namespace,
    total_threads: int,
) -> BenchResult:
    kml_lib_path = args.kml_lib or find_default_kml_lib()
    if not kml_lib_path:
        raise RuntimeError(
            "KML libkblas.so not found. Pass --kml-lib or set KML_BLAS_LIB."
        )
    return bench_cpu_numa_processes(
        case,
        args,
        total_threads,
        device="cpu-kml-bgemm-batch-numa-bf16",
        worker_target=numa_kml_bgemm_batch_worker,
        extra_worker_args=(os.path.abspath(kml_lib_path),),
    )


def bench_cpu_kml_bgemm_pack_numa(
    case: BenchCase,
    args: argparse.Namespace,
    total_threads: int,
) -> BenchResult:
    kml_lib_path = args.kml_lib or find_default_kml_lib()
    if not kml_lib_path:
        raise RuntimeError(
            "KML libkblas.so not found. Pass --kml-lib or set KML_BLAS_LIB."
        )
    return bench_cpu_numa_processes(
        case,
        args,
        total_threads,
        device="cpu-kml-bgemm-pack-numa-bf16",
        worker_target=numa_kml_bgemm_pack_worker,
        extra_worker_args=(os.path.abspath(kml_lib_path),),
    )


def kudnn_process_worker(
    worker_idx: int,
    cpu_ids: list[int],
    local_experts: int,
    rows_per_expert: int,
    hidden_size: int,
    intermediate_size: int,
    blas_threads: int,
    worker_path: str,
    ld_library_path: str,
    warmup: int,
    iters: int,
    task_queue,
    result_queue,
) -> None:
    try:
        if cpu_ids:
            os.sched_setaffinity(0, set(cpu_ids))

        env = os.environ.copy()
        env["LD_LIBRARY_PATH"] = ld_library_path
        env["OMP_NUM_THREADS"] = str(blas_threads)
        env["GOMP_CPU_AFFINITY"] = " ".join(str(cpu) for cpu in cpu_ids)

        result_queue.put({
            "worker": worker_idx,
            "event": "ready",
            "local_experts": local_experts,
            "cpu_count": len(cpu_ids),
            "blas_threads": blas_threads,
            "worker_path": worker_path,
        })

        while True:
            cmd = task_queue.get()
            if cmd == "stop":
                break
            if cmd != "run":
                raise ValueError(f"unknown worker command: {cmd!r}")

            proc = subprocess.run(
                [
                    worker_path,
                    str(local_experts),
                    str(rows_per_expert),
                    str(hidden_size),
                    str(intermediate_size),
                    str(blas_threads),
                    str(warmup),
                    str(iters),
                ],
                text=True,
                capture_output=True,
                env=env,
                check=True,
            )
            result_queue.put({
                "worker": worker_idx,
                "event": "done",
                "payload": json.loads(proc.stdout),
            })
    except Exception:
        result_queue.put({
            "worker": worker_idx,
            "event": "error",
            "traceback": traceback.format_exc(),
        })


def bench_cpu_kudnn_gemm_numa(
    case: BenchCase,
    args: argparse.Namespace,
    total_threads: int,
) -> BenchResult:
    worker_path = ensure_kudnn_worker(args)
    cpu_lists = get_numa_cpu_lists()
    if not cpu_lists:
        cpu_lists = [sorted(os.sched_getaffinity(0))]
    if args.cpu_numa_nodes > 0:
        cpu_lists = cpu_lists[:args.cpu_numa_nodes]

    worker_count = min(case.active_experts, len(cpu_lists), total_threads)
    if worker_count <= 0:
        raise RuntimeError("no CPU workers available for kudnn-gemm-numa")

    expert_counts = split_evenly(case.active_experts, worker_count)
    active_cpu_lists = cpu_lists[:worker_count]
    thread_counts = split_evenly(total_threads, worker_count)
    thread_counts = [
        max(1, min(thread_counts[idx], len(active_cpu_lists[idx])))
        for idx in range(worker_count)
    ]
    worker_cpu_lists = [
        active_cpu_lists[idx][:thread_counts[idx]]
        for idx in range(worker_count)
    ]

    kml_lib_dir = os.path.dirname(os.path.abspath(args.kml_lib))
    ld_parts = [
        os.path.abspath(args.sdma_lib_dir),
        os.path.join(os.path.abspath(args.kudnn_root), "lib"),
        os.path.abspath(args.kupl_lib_dir),
        kml_lib_dir,
        os.environ.get("LD_LIBRARY_PATH", ""),
    ]
    ld_library_path = ":".join(part for part in ld_parts if part)

    procs = []
    for idx in range(worker_count):
        env = os.environ.copy()
        env["LD_LIBRARY_PATH"] = ld_library_path
        env["OMP_NUM_THREADS"] = str(thread_counts[idx])
        env.pop("OMP_PROC_BIND", None)
        env.pop("OMP_PLACES", None)
        env.pop("GOMP_CPU_AFFINITY", None)

        def set_affinity(cpus=worker_cpu_lists[idx]) -> None:
            if cpus:
                os.sched_setaffinity(0, set(cpus))

        proc = subprocess.Popen(
            [
                worker_path,
                str(expert_counts[idx]),
                str(case.rows_per_expert),
                str(case.hidden_size),
                str(case.intermediate_size),
                str(thread_counts[idx]),
                str(args.warmup),
                str(args.iters),
            ],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            preexec_fn=set_affinity,
        )
        procs.append((idx, proc))

    worker_times: list[list[float]] = []
    try:
        for idx, proc in procs:
            stdout, stderr = proc.communicate()
            if proc.returncode != 0:
                raise RuntimeError(
                    f"kudnn worker {idx} failed with exit={proc.returncode}\n"
                    f"stdout:\n{stdout}\nstderr:\n{stderr}"
                )
            payload = json.loads(stdout)
            times = [float(v) for v in payload["times_ms"]]
            if len(times) != args.iters:
                raise RuntimeError(
                    f"kudnn worker {idx} returned {len(times)} timings, expected {args.iters}"
                )
            worker_times.append(times)
    finally:
        for _idx, proc in procs:
            if proc.poll() is None:
                proc.terminate()
                proc.wait(timeout=5)

    times_ms = [
        max(worker_times[worker_idx][iter_idx] for worker_idx in range(worker_count))
        for iter_idx in range(args.iters)
    ]
    median_ms = statistics.median(times_ms)
    return BenchResult(
        scope=case.scope,
        case_key=case.case_key,
        device="cpu-kudnn-gemm-numa-fp32",
        workers=f"{worker_count}x{','.join(map(str, thread_counts))}",
        active_experts=case.active_experts,
        expanded_rows=case.expanded_rows,
        rows_per_expert=case.rows_per_expert,
        median_ms=median_ms,
        min_ms=min(times_ms),
        p90_ms=percentile(times_ms, 90),
        tflops=case.logical_flops / (median_ms / 1000.0) / 1e12,
    )


def bench_cpu_kudnn_gemm_single(
    case: BenchCase,
    args: argparse.Namespace,
    total_threads: int,
) -> BenchResult:
    worker_path = ensure_kudnn_worker(args)
    kml_lib_dir = os.path.dirname(os.path.abspath(args.kml_lib))
    ld_parts = [
        os.path.abspath(args.sdma_lib_dir),
        os.path.join(os.path.abspath(args.kudnn_root), "lib"),
        os.path.abspath(args.kupl_lib_dir),
        kml_lib_dir,
        os.environ.get("LD_LIBRARY_PATH", ""),
    ]
    ld_library_path = ":".join(part for part in ld_parts if part)

    cpu_lists = get_numa_cpu_lists()
    if cpu_lists:
        flat_cpus = [cpu for node in cpu_lists for cpu in node]
    else:
        flat_cpus = sorted(os.sched_getaffinity(0))
    affinity_cpus = flat_cpus[:min(total_threads, len(flat_cpus))]

    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = ld_library_path
    env["OMP_NUM_THREADS"] = str(total_threads)
    env.pop("OMP_PROC_BIND", None)
    env.pop("OMP_PLACES", None)
    env.pop("GOMP_CPU_AFFINITY", None)

    def set_affinity() -> None:
        if affinity_cpus:
            os.sched_setaffinity(0, set(affinity_cpus))

    proc = subprocess.run(
        [
            worker_path,
            str(case.active_experts),
            str(case.rows_per_expert),
            str(case.hidden_size),
            str(case.intermediate_size),
            str(total_threads),
            str(args.warmup),
            str(args.iters),
        ],
        text=True,
        capture_output=True,
        env=env,
        preexec_fn=set_affinity,
        check=True,
    )
    payload = json.loads(proc.stdout)
    times_ms = [float(v) for v in payload["times_ms"]]
    median_ms = statistics.median(times_ms)
    return BenchResult(
        scope=case.scope,
        case_key=case.case_key,
        device="cpu-kudnn-gemm-single-fp32",
        workers=str(total_threads),
        active_experts=case.active_experts,
        expanded_rows=case.expanded_rows,
        rows_per_expert=case.rows_per_expert,
        median_ms=median_ms,
        min_ms=min(times_ms),
        p90_ms=percentile(times_ms, 90),
        tflops=case.logical_flops / (median_ms / 1000.0) / 1e12,
    )


def bench_cpu_numa_processes(
    case: BenchCase,
    args: argparse.Namespace,
    total_threads: int,
    device: str,
    worker_target,
    extra_worker_args: tuple,
    external_timing: bool = False,
) -> BenchResult:
    cpu_lists = get_numa_cpu_lists()
    if not cpu_lists:
        cpu_lists = [sorted(os.sched_getaffinity(0))]
    if args.cpu_numa_nodes > 0:
        cpu_lists = cpu_lists[:args.cpu_numa_nodes]

    if args.cpu_numa_layout == "split-threads":
        worker_count = min(case.active_experts, len(cpu_lists), total_threads)
        active_cpu_lists = cpu_lists[:worker_count]
        thread_counts = split_evenly(total_threads, worker_count)
        thread_counts = [
            max(1, min(thread_counts[idx], len(active_cpu_lists[idx])))
            for idx in range(worker_count)
        ]
        worker_cpu_lists = [
            active_cpu_lists[idx][:thread_counts[idx]]
            for idx in range(worker_count)
        ]
    else:
        worker_count = min(case.active_experts, len(cpu_lists))
        worker_cpu_lists = [list(cpus) for cpus in cpu_lists[:worker_count]]
        thread_counts = [len(cpus) for cpus in worker_cpu_lists]

    if worker_count <= 0:
        raise RuntimeError("no CPU workers available for numpy-openblas-numa")

    expert_counts = split_evenly(case.active_experts, worker_count)

    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue()
    task_queues = [ctx.Queue() for _ in range(worker_count)]
    workers = []
    for idx in range(worker_count):
        proc = ctx.Process(
            target=worker_target,
            args=(
                idx,
                worker_cpu_lists[idx],
                expert_counts[idx],
                case.rows_per_expert,
                case.hidden_size,
                case.intermediate_size,
                thread_counts[idx],
                *extra_worker_args,
                task_queues[idx],
                result_queue,
            ),
        )
        proc.start()
        workers.append(proc)

    try:
        ready = 0
        while ready < worker_count:
            msg = result_queue.get()
            if msg.get("event") == "ready":
                ready += 1
            elif msg.get("event") == "error":
                raise RuntimeError(msg.get("traceback", "worker failed"))

        def run_parallel_once() -> None:
            for q in task_queues:
                q.put("run")
            done = 0
            while done < worker_count:
                msg = result_queue.get()
                if msg.get("event") == "done":
                    done += 1
                elif msg.get("event") == "error":
                    raise RuntimeError(msg.get("traceback", "worker failed"))

        for _ in range(args.warmup):
            run_parallel_once()

        times_ms: list[float] = []
        for _ in range(args.iters):
            if external_timing:
                for q in task_queues:
                    q.put("run")
                done = 0
                worker_times: list[float] = []
                while done < worker_count:
                    msg = result_queue.get()
                    if msg.get("event") == "done":
                        done += 1
                        payload = msg.get("payload", {})
                        worker_times.extend(float(v) for v in payload.get("times_ms", []))
                    elif msg.get("event") == "error":
                        raise RuntimeError(msg.get("traceback", "worker failed"))
                times_ms.append(max(worker_times) if worker_times else float("nan"))
            else:
                t0 = time.perf_counter()
                run_parallel_once()
                t1 = time.perf_counter()
                times_ms.append((t1 - t0) * 1000.0)
    finally:
        for q in task_queues:
            q.put("stop")
        for proc in workers:
            proc.join(timeout=5)
            if proc.is_alive():
                proc.terminate()
                proc.join(timeout=2)

    median_ms = statistics.median(times_ms)
    return BenchResult(
        scope=case.scope,
        case_key=case.case_key,
        device=device,
        workers=f"{worker_count}x{','.join(map(str, thread_counts))}",
        active_experts=case.active_experts,
        expanded_rows=case.expanded_rows,
        rows_per_expert=case.rows_per_expert,
        median_ms=median_ms,
        min_ms=min(times_ms),
        p90_ms=percentile(times_ms, 90),
        tflops=case.logical_flops / (median_ms / 1000.0) / 1e12,
    )


def bench_cpu(torch, case: BenchCase, args: argparse.Namespace, threads: int) -> BenchResult:
    if args.cpu_backend == "numpy-openblas":
        return bench_cpu_numpy_openblas(case, args, threads)
    if args.cpu_backend == "numpy-openblas-numa":
        return bench_cpu_numpy_openblas_numa(case, args, threads)
    if args.cpu_backend == "kml-sgemm-numa":
        return bench_cpu_kml_sgemm_numa(case, args, threads)
    if args.cpu_backend == "kml-sbgemm-numa":
        return bench_cpu_kml_sbgemm_numa(case, args, threads)
    if args.cpu_backend == "kml-bgemm-batch-numa":
        return bench_cpu_kml_bgemm_batch_numa(case, args, threads)
    if args.cpu_backend == "kml-bgemm-pack-numa":
        return bench_cpu_kml_bgemm_pack_numa(case, args, threads)
    if args.cpu_backend == "kudnn-gemm-numa":
        return bench_cpu_kudnn_gemm_numa(case, args, threads)
    if args.cpu_backend == "kudnn-gemm-single":
        return bench_cpu_kudnn_gemm_single(case, args, threads)

    torch.set_num_threads(threads)
    dtype = torch_dtype(torch, args.dtype)
    shape_x = (case.active_experts, case.rows_per_expert, case.hidden_size)
    shape_w13 = (case.active_experts, case.hidden_size, 2 * case.intermediate_size)
    shape_w2 = (case.active_experts, case.intermediate_size, case.hidden_size)

    with torch.inference_mode():
        x = torch.randn(shape_x, dtype=dtype)
        w13 = torch.randn(shape_w13, dtype=dtype) * 0.01
        w2 = torch.randn(shape_w2, dtype=dtype) * 0.01

        if args.cpu_backend == "bmm":
            fn = lambda: cpu_expert_mlp_bmm(torch, x, w13, w2)
        else:
            fn = lambda: cpu_expert_mlp_loop(torch, x, w13, w2)

        out = None
        for _ in range(args.warmup):
            out = fn()
        times_ms: list[float] = []
        for _ in range(args.iters):
            t0 = time.perf_counter()
            out = fn()
            t1 = time.perf_counter()
            times_ms.append((t1 - t0) * 1000.0)

        if out is None or tuple(out.shape) != (
            case.active_experts,
            case.rows_per_expert,
            case.hidden_size,
        ):
            raise RuntimeError("CPU output shape validation failed")

    median_ms = statistics.median(times_ms)
    return BenchResult(
        scope=case.scope,
        case_key=case.case_key,
        device=f"cpu-{args.cpu_backend}",
        workers=str(threads),
        active_experts=case.active_experts,
        expanded_rows=case.expanded_rows,
        rows_per_expert=case.rows_per_expert,
        median_ms=median_ms,
        min_ms=min(times_ms),
        p90_ms=percentile(times_ms, 90),
        tflops=case.logical_flops / (median_ms / 1000.0) / 1e12,
    )


def new_npu_event(torch):
    try:
        return torch.npu.Event(enable_timing=True)
    except TypeError:
        return torch.npu.Event()


def bench_npu(torch, torch_npu, case: BenchCase, args: argparse.Namespace, npu_id: int) -> BenchResult:
    device = torch.device(f"npu:{npu_id}")
    torch.npu.set_device(device)
    dtype = torch_dtype(torch, args.dtype)
    shape_x = (case.expanded_rows, case.hidden_size)
    shape_w13 = (case.active_experts, case.hidden_size, 2 * case.intermediate_size)
    shape_w2 = (case.active_experts, case.intermediate_size, case.hidden_size)
    group_list = [
        (idx + 1) * case.rows_per_expert for idx in range(case.active_experts)
    ]

    with torch.inference_mode():
        x = torch.randn(shape_x, device=device, dtype=dtype)
        w13 = torch.randn(shape_w13, device=device, dtype=dtype) * 0.01
        w2 = torch.randn(shape_w2, device=device, dtype=dtype) * 0.01
        groups = torch.tensor(group_list, device=device, dtype=torch.int64)

        def fn():
            gate_up = torch_npu.npu_grouped_matmul(
                [x],
                [w13],
                group_list=groups,
                split_item=2,
                group_type=0,
                group_list_type=0,
            )[0]
            activated = torch_npu.npu_swiglu(gate_up)
            return torch_npu.npu_grouped_matmul(
                [activated],
                [w2],
                group_list=groups,
                split_item=2,
                group_type=0,
                group_list_type=0,
            )[0]

        out = None
        for _ in range(args.warmup):
            out = fn()
        torch.npu.synchronize(device)

        times_ms: list[float] = []
        for _ in range(args.iters):
            start = new_npu_event(torch)
            end = new_npu_event(torch)
            start.record()
            out = fn()
            end.record()
            torch.npu.synchronize(device)
            try:
                elapsed = start.elapsed_time(end)
            except Exception:
                elapsed = float("nan")
            if not math.isfinite(elapsed) or elapsed <= 0:
                # Fallback includes Python submission overhead, but keeps the script usable.
                t0 = time.perf_counter()
                out = fn()
                torch.npu.synchronize(device)
                t1 = time.perf_counter()
                elapsed = (t1 - t0) * 1000.0
            times_ms.append(elapsed)

        if out is None or tuple(out.shape) != (case.expanded_rows, case.hidden_size):
            raise RuntimeError("NPU output shape validation failed")

    median_ms = statistics.median(times_ms)
    return BenchResult(
        scope=case.scope,
        case_key=case.case_key,
        device=f"npu:{npu_id}",
        workers="1",
        active_experts=case.active_experts,
        expanded_rows=case.expanded_rows,
        rows_per_expert=case.rows_per_expert,
        median_ms=median_ms,
        min_ms=min(times_ms),
        p90_ms=percentile(times_ms, 90),
        tflops=case.logical_flops / (median_ms / 1000.0) / 1e12,
    )


def print_markdown_results(results: list[BenchResult]) -> None:
    print("")
    print("| case | scope | device | workers | experts | expanded rows | rows/expert | median ms | min ms | p90 ms | effective TFLOP/s |")
    print("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in results:
        print(
            f"| {row.case_key} | {row.scope} | {row.device} | {row.workers} | {row.active_experts} "
            f"| {row.expanded_rows} | {row.rows_per_expert} "
            f"| {format_float(row.median_ms)} | {format_float(row.min_ms)} "
            f"| {format_float(row.p90_ms)} | {format_float(row.tflops)} |"
        )

    by_scope: dict[tuple[str, int, int], list[BenchResult]] = {}
    for row in results:
        by_scope.setdefault(
            (row.scope, row.active_experts, row.rows_per_expert), []).append(row)

    print("")
    print("| case | comparison | speedup |")
    print("|---|---|---:|")
    for _case_key, rows in by_scope.items():
        npu_rows = [r for r in rows if r.device.startswith("npu:")]
        if not npu_rows:
            continue
        npu = min(npu_rows, key=lambda r: r.median_ms)
        for row in rows:
            if row is npu or row.device.startswith("npu:"):
                continue
            speedup = row.median_ms / npu.median_ms
            print(f"| {row.case_key} | {npu.device} vs {row.device}({row.workers}) | {format_float(speedup, 2)}x |")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare Qwen3 MoE expert MLP compute on CPU and NPU.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model-config",
        default=find_default_model_config(),
        help=(
            "HF model config used to derive hidden/intermediate/expert/top-k sizes. "
            "Defaults to MODEL_CONFIG/MODEL_PATH or the local /home/data Qwen3 path; "
            "if missing, built-in Qwen3-30B-A3B MoE sizes are used."
        ),
    )
    parser.add_argument(
        "--tokens",
        type=int,
        default=17408,
        help="Rollout input tokens before top-k expansion. The training script default is 1024+16384.",
    )
    parser.add_argument(
        "--rows-per-expert",
        "--expert-batch-sizes",
        dest="rows_per_expert",
        type=parse_int_list,
        default=None,
        help=(
            "Comma-separated per-expert MoE micro-batch sizes to sweep. "
            "For ep-local Qwen3-30B-A3B with 16-way EP, values "
            "512,256,128,64,32,16,8 correspond to pre-routing token counts "
            "8192,4096,2048,1024,512,256,128. Overrides --tokens."
        ),
    )
    parser.add_argument("--ep-size", type=int, default=16, help="Rollout expert parallel size.")
    parser.add_argument("--hidden-size", type=int, default=None, help="Override model hidden size.")
    parser.add_argument(
        "--moe-intermediate-size",
        type=int,
        default=None,
        help="Override MoE expert intermediate size.",
    )
    parser.add_argument("--num-experts", type=int, default=None, help="Override model expert count.")
    parser.add_argument("--top-k", type=int, default=None, help="Override experts per token.")
    parser.add_argument(
        "--active-experts",
        "--expert-counts",
        dest="active_experts",
        type=parse_int_list,
        default=None,
        help=(
            "Comma-separated number of experts resident/active on the measured "
            "CPU or NPU process. Use 8,16,32,64,128 to sweep the requested "
            "expert counts. With --rows-per-expert this forms a full cross "
            "product of expert count x per-expert batch size."
        ),
    )
    parser.add_argument(
        "--scope",
        type=parse_scope_list,
        default=parse_scope_list("ep-local"),
        help="'ep-local', 'full-layer', or comma-separated list.",
    )
    parser.add_argument(
        "--dtype",
        choices=["bfloat16", "float16", "float32"],
        default="bfloat16",
        help="Tensor dtype. bfloat16 matches Qwen3-30B-A3B rollout weights.",
    )
    parser.add_argument("--warmup", type=int, default=5, help="Warmup iterations.")
    parser.add_argument("--iters", type=int, default=20, help="Measured iterations.")
    parser.add_argument(
        "--cpu-threads",
        type=parse_int_list,
        default=default_cpu_threads(),
        help="Comma-separated CPU thread counts to sweep. Defaults are derived from local NUMA topology.",
    )
    parser.add_argument(
        "--cpu-backend",
        choices=[
            "bmm",
            "loop",
            "numpy-openblas",
            "numpy-openblas-numa",
            "kml-sgemm-numa",
            "kml-sbgemm-numa",
            "kml-bgemm-batch-numa",
            "kml-bgemm-pack-numa",
            "kudnn-gemm-numa",
            "kudnn-gemm-single",
        ],
        default="numpy-openblas-numa",
        help=(
            "CPU implementation. bmm/loop use PyTorch and the requested dtype. "
            "numpy-openblas uses FP32 OpenBLAS serial expert loop. "
            "numpy-openblas-numa shards experts across NUMA-bound worker processes. "
            "kml-sgemm-numa uses HPCKit KML cblas_sgemm in the same NUMA layout. "
            "kml-sbgemm-numa uses HPCKit KML cblas_sbgemm BF16 inputs with FP32 outputs. "
            "kml-bgemm-batch-numa uses HPCKit KML cblas_bgemm_batch with BF16 outputs. "
            "kml-bgemm-pack-numa pre-packs expert weights and uses HPCKit KML cblas_bgemm_compute. "
            "kudnn-gemm-numa uses KuDNN Gemm from HPCKit/KDNN with a NUMA worker layout. "
            "kudnn-gemm-single uses one KuDNN process for all experts."
        ),
    )
    parser.add_argument(
        "--cpu-numa-nodes",
        type=int,
        default=0,
        help=(
            "For *-numa CPU backends: limit worker placement to the first N NUMA "
            "nodes. 0 uses all detected NUMA nodes."
        ),
    )
    parser.add_argument(
        "--cpu-numa-layout",
        choices=["full-node", "split-threads"],
        default="full-node",
        help=(
            "NUMA worker layout for *-numa CPU backends. full-node starts one "
            "worker per NUMA node, binds it to all cores of that node, and splits "
            "experts across workers. split-threads preserves the older behavior "
            "that divides --cpu-threads across NUMA workers."
        ),
    )
    parser.add_argument(
        "--kml-lib",
        default=None,
        help=(
            "Path to KML libkblas.so for kml-* CPU backends. Defaults to "
            "KML_BLAS_LIB or local third_party/hpckit26 unpack."
        ),
    )
    parser.add_argument(
        "--kudnn-root",
        default=None,
        help="KuDNN root containing include/ and lib/. Defaults to KUDNN_ROOT or local HPCKit 26 unpack.",
    )
    parser.add_argument(
        "--kupl-lib-dir",
        default=None,
        help="Directory containing libkupl.so. Defaults to KUPL_LIB_DIR or local HPCKit 26 unpack.",
    )
    parser.add_argument(
        "--sdma-lib-dir",
        default=None,
        help=(
            "Directory containing libsdma_dk.so. Defaults to SDMA_LIB_DIR or local "
            "third_party/hpckit26 shim used only for GEMM-path testing."
        ),
    )
    parser.add_argument(
        "--kudnn-worker-src",
        default=find_repo_file("kudnn_moe_worker.cpp"),
        help="C++ source for the KuDNN MoE worker.",
    )
    parser.add_argument(
        "--kudnn-worker",
        default=find_repo_file("kudnn_moe_worker"),
        help="Compiled KuDNN MoE worker executable path.",
    )
    parser.add_argument("--cxx", default="g++", help="C++ compiler for the KuDNN worker.")
    parser.add_argument(
        "--rebuild-kudnn-worker",
        action="store_true",
        help="Force rebuild of the KuDNN worker executable.",
    )
    parser.add_argument(
        "--cpu-inter-op-threads",
        type=int,
        default=1,
        help="PyTorch CPU inter-op threads. Must be set before work starts.",
    )
    parser.add_argument("--skip-cpu", action="store_true", help="Skip CPU measurements.")
    parser.add_argument("--skip-npu", action="store_true", help="Skip NPU measurements.")
    parser.add_argument(
        "--npus",
        type=parse_npu_list,
        default=parse_npu_list("0"),
        help="NPU ids to test, or a count such as '2' for devices 0..1.",
    )
    parser.add_argument("--json-out", default=None, help="Optional path to write raw result JSON.")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    if raw_argv == ["--_openblas_child"]:
        return run_openblas_child_from_stdin()

    args = build_parser().parse_args(raw_argv)
    if args.warmup < 0 or args.iters <= 0:
        raise SystemExit("--warmup must be >= 0 and --iters must be > 0")

    cfg, model_config_source = read_model_config(args.model_config)
    args.model_config = model_config_source

    os.environ.setdefault("OMP_PROC_BIND", "close")
    os.environ.setdefault("OMP_PLACES", "cores")

    need_torch = (
        not args.skip_npu
        or (not args.skip_cpu and args.cpu_backend not in (
            "numpy-openblas",
            "numpy-openblas-numa",
            "kml-sgemm-numa",
            "kml-sbgemm-numa",
            "kml-bgemm-batch-numa",
            "kml-bgemm-pack-numa",
            "kudnn-gemm-numa",
            "kudnn-gemm-single",
        ))
    )
    torch = None
    if need_torch:
        import torch as imported_torch

        torch = imported_torch
        try:
            torch.set_num_interop_threads(args.cpu_inter_op_threads)
        except RuntimeError:
            pass

    torch_npu = None
    if not args.skip_npu:
        try:
            import torch_npu as imported_torch_npu

            torch_npu = imported_torch_npu
        except Exception as exc:
            print(f"[warn] torch_npu import failed, skipping NPU: {exc}", file=sys.stderr)
            args.skip_npu = True

    cases = make_cases(cfg, args)

    print("MoE expert MLP benchmark")
    print(f"model_config={model_config_source}")
    print(
        "model="
        f"H={cases[0].hidden_size}, I={cases[0].intermediate_size}, "
        f"E={cases[0].num_experts}, top_k={cases[0].top_k}, dtype={args.dtype}"
    )
    print(
        f"tokens={cases[0].num_tokens}, ep_size={cases[0].ep_size}, "
        "measured_ops=grouped(x@w13)->swiglu->grouped(@w2)"
    )
    if args.rows_per_expert is not None:
        print(
            "rows_per_expert_sweep="
            f"{','.join(map(str, args.rows_per_expert))} "
            "(overrides the single --tokens value)"
        )
    if args.active_experts is not None:
        print(
            "active_experts_sweep="
            f"{','.join(map(str, args.active_experts))} "
            "(overrides scope-derived expert count)"
        )
    print(f"cpu={get_cpu_model()}")
    print(f"cpu_topology={get_cpu_topology_summary()}")
    if torch is not None:
        npu_available = getattr(torch, "npu", None) is not None and torch.npu.is_available()
        print(f"torch={torch.__version__}, npu_available={npu_available}")
    else:
        print("torch=not imported for clean numpy-openblas CPU run")
    if torch_npu is not None:
        print(f"torch_npu={getattr(torch_npu, '__version__', 'unknown')}, npu_count={torch.npu.device_count()}")
    cpu_thread_sweep = effective_cpu_thread_sweep(args)
    print(f"cpu_backend={args.cpu_backend}, cpu_threads={','.join(map(str, cpu_thread_sweep))}")
    if args.cpu_backend in NUMA_CPU_BACKENDS:
        print(f"cpu_numa_layout={args.cpu_numa_layout}, cpu_numa_nodes={args.cpu_numa_nodes or 'all'}")
    for case in cases:
        print(
            f"case[{case.scope}]: active_experts={case.active_experts}, "
            f"expanded_rows={case.expanded_rows}, rows_per_expert={case.rows_per_expert}, "
            f"logical_flops={case.logical_flops / 1e9:.3f} GFLOP, "
            f"weight_bytes_bf16={case.weight_mib:.1f} MiB"
        )

    results: list[BenchResult] = []
    for case in cases:
        if not args.skip_npu:
            for npu_id in args.npus:
                print(f"[run] NPU case={case.scope} device=npu:{npu_id}", flush=True)
                results.append(bench_npu(torch, torch_npu, case, args, npu_id))
        if not args.skip_cpu:
            for threads in cpu_thread_sweep:
                print(f"[run] CPU case={case.scope} threads={threads}", flush=True)
                results.append(bench_cpu(torch, case, args, threads))

    print_markdown_results(results)

    if args.json_out:
        payload = {
            "model_config": args.model_config,
            "tokens": args.tokens,
            "rows_per_expert": args.rows_per_expert,
            "active_experts": args.active_experts,
            "ep_size": cases[0].ep_size,
            "dtype": args.dtype,
            "cpu": get_cpu_model(),
            "torch": torch.__version__ if torch is not None else None,
            "torch_npu": getattr(torch_npu, "__version__", None) if torch_npu is not None else None,
            "cases": [case.__dict__ for case in cases],
            "results": [row.__dict__ for row in results],
        }
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"\nwrote {args.json_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
