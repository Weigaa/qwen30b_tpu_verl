#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# This file is mainly Adapted from vllm-project/vllm/vllm/envs.py
# Copyright 2023 The vLLM team.
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

import os
from typing import Any, Callable, Dict, Optional

# The begin-* and end* here are used by the documentation generator
# to extract the used env vars.

# begin-env-vars-definition

def _get_optional_positive_power_of_two_env(name: str) -> Optional[int]:
    raw_value = os.getenv(name)
    if raw_value is None or raw_value == "":
        return None
    value = int(raw_value)
    if value <= 0 or (value & (value - 1)) != 0:
        raise ValueError(
            f"{name} must be a positive power of two, got {raw_value!r}.")
    return value


def _get_optional_positive_int_env(name: str) -> Optional[int]:
    raw_value = os.getenv(name)
    if raw_value is None or raw_value == "":
        return None
    value = int(raw_value)
    if value <= 0:
        raise ValueError(
            f"{name} must be a positive integer, got {raw_value!r}.")
    return value


def _get_optional_float_env(name: str) -> Optional[float]:
    raw_value = os.getenv(name)
    if raw_value is None or raw_value == "":
        return None
    return float(raw_value)


def _has_explicit_elastic_execution_mode() -> bool:
    raw_value = os.getenv("VLLM_ASCEND_ELASTIC_EXECUTION_MODE")
    return raw_value is not None and raw_value != ""


def _parse_elastic_execution_mode(raw_value: str) -> int:
    value = int(raw_value)
    if value not in (0, 1, 2, 3, 4, 5):
        raise ValueError(
            "VLLM_ASCEND_ELASTIC_EXECUTION_MODE must be one of 0, 1, 2, 3, 4, or 5, "
            f"got {raw_value!r}.")
    return value


def get_elastic_execution_mode() -> int:
    if _has_explicit_elastic_execution_mode():
        return _parse_elastic_execution_mode(
            os.environ["VLLM_ASCEND_ELASTIC_EXECUTION_MODE"])

    shrink_enabled = bool(
        int(os.getenv("VLLM_ASCEND_ENABLE_ELASTIC_PARALLEL_SHRINK", "0")))
    if not shrink_enabled:
        return 0

    elastic_moe_mode = os.getenv("VLLM_ASCEND_ELASTIC_MOE_MODE",
                                 "lossy").lower().strip()
    if elastic_moe_mode != "lossless":
        return 0

    init_redundancy_expert = int(
        os.getenv("VLLM_ASCEND_INIT_REDUNDANCY_EXPERT", "0"))
    return 1 if init_redundancy_expert > 0 else 2


def get_effective_elastic_parallel_shrink_enabled() -> bool:
    if _has_explicit_elastic_execution_mode():
        return get_elastic_execution_mode() != 0
    return bool(int(os.getenv("VLLM_ASCEND_ENABLE_ELASTIC_PARALLEL_SHRINK",
                              "0")))


def get_effective_elastic_moe_mode() -> str:
    if _has_explicit_elastic_execution_mode():
        return "lossless" if get_elastic_execution_mode() in (1, 2, 3, 4, 5) else "lossy"
    return os.getenv("VLLM_ASCEND_ELASTIC_MOE_MODE", "lossy").lower().strip()


def compute_elastic_init_redundancy_expert(logical_num_experts: int,
                                           initial_ep_size: int,
                                           explicit_value: Optional[int] = None
                                           ) -> int:
    if explicit_value is None:
        explicit_value = int(os.getenv("VLLM_ASCEND_INIT_REDUNDANCY_EXPERT",
                                       "0"))

    if not _has_explicit_elastic_execution_mode():
        return max(int(explicit_value), 0)

    mode = get_elastic_execution_mode()
    if mode == 0:
        return 0

    if logical_num_experts <= 0 or initial_ep_size <= 0:
        raise ValueError(
            "compute_elastic_init_redundancy_expert requires positive "
            f"logical_num_experts and initial_ep_size, got "
            f"{logical_num_experts} and {initial_ep_size}.")
    if logical_num_experts % initial_ep_size != 0:
        raise ValueError(
            f"logical_num_experts={logical_num_experts} must divide the "
            f"initial EP size={initial_ep_size}.")

    current_local_experts = logical_num_experts // initial_ep_size
    if mode == 1:
        min_compute_group_size = _get_optional_positive_power_of_two_env(
            "VLLM_ASCEND_ELASTIC_MIN_COMPUTE_GROUP_SIZE")
        if min_compute_group_size is None:
            return max(int(explicit_value), 0)
        if initial_ep_size % min_compute_group_size != 0:
            raise ValueError(
                "VLLM_ASCEND_ELASTIC_MIN_COMPUTE_GROUP_SIZE must divide the "
                f"initial EP size: {min_compute_group_size} vs {initial_ep_size}.")
        if logical_num_experts % min_compute_group_size != 0:
            raise ValueError(
                f"logical_num_experts={logical_num_experts} must divide the "
                f"configured floor={min_compute_group_size}.")
        target_local_experts = logical_num_experts // min_compute_group_size
    elif mode in (3, 4, 5):
        return 0
    else:
        hybrid_resident_slots = _get_optional_positive_int_env(
            "VLLM_ASCEND_ELASTIC_HYBRID_RESIDENT_EXPERT_SLOTS")
        if hybrid_resident_slots is None:
            return max(int(explicit_value), 0)
        target_local_experts = max(current_local_experts,
                                   int(hybrid_resident_slots))

    if target_local_experts <= current_local_experts:
        return 0
    return initial_ep_size * (target_local_experts - current_local_experts)


env_variables: Dict[str, Callable[[], Any]] = {
    # Unified elastic execution mode:
    # - 0: baseline dummy-run path
    # - 1: preloaded redundant experts only
    # - 2: preloaded redundant experts + CPU/NPU hybrid fallback
    # - 3: no redundancy + cross-layer double-buffer hybrid tail
    # - 4: no redundancy + cross-layer double-buffer remote-NPU expert cache
    # - 5: no redundancy + cross-layer double-buffer hybrid CPU+remote-NPU cache
    # When this is set, it overrides the older elastic enable / moe mode /
    # init redundancy expert knobs.
    "VLLM_ASCEND_ELASTIC_EXECUTION_MODE":
    lambda: get_elastic_execution_mode(),
    # max compile thread number for package building. Usually, it is set to
    # the number of CPU cores. If not set, the default value is None, which
    # means all number of CPU cores will be used.
    "MAX_JOBS":
    lambda: os.getenv("MAX_JOBS", None),
    # The build type of the package. It can be one of the following values:
    # Release, Debug, RelWithDebugInfo. If not set, the default value is Release.
    "CMAKE_BUILD_TYPE":
    lambda: os.getenv("CMAKE_BUILD_TYPE"),
    # Whether to compile custom kernels. If not set, the default value is True.
    # If set to False, the custom kernels will not be compiled. Please note that
    # the sleep mode feature will be disabled as well if custom kernels are not
    # compiled.
    "COMPILE_CUSTOM_KERNELS":
    lambda: bool(int(os.getenv("COMPILE_CUSTOM_KERNELS", "1"))),
    # The CXX compiler used for compiling the package. If not set, the default
    # value is None, which means the system default CXX compiler will be used.
    "CXX_COMPILER":
    lambda: os.getenv("CXX_COMPILER", None),
    # The C compiler used for compiling the package. If not set, the default
    # value is None, which means the system default C compiler will be used.
    "C_COMPILER":
    lambda: os.getenv("C_COMPILER", None),
    # The version of the Ascend chip. If not set, the default value is
    # ASCEND910B1(Available for A2 and A3 series). It's used for package building.
    # Please make sure that the version is correct.
    "SOC_VERSION":
    lambda: os.getenv("SOC_VERSION", "ASCEND910B1"),
    # If set, vllm-ascend will print verbose logs during compilation
    "VERBOSE":
    lambda: bool(int(os.getenv('VERBOSE', '0'))),
    # The home path for CANN toolkit. If not set, the default value is
    # /usr/local/Ascend/ascend-toolkit/latest
    "ASCEND_HOME_PATH":
    lambda: os.getenv("ASCEND_HOME_PATH", None),
    # The path for HCCL library, it's used by pyhccl communicator backend. If
    # not set, the default value is libhccl.so。
    "HCCL_SO_PATH":
    lambda: os.environ.get("HCCL_SO_PATH", None),
    # The version of vllm is installed. This value is used for developers who
    # installed vllm from source locally. In this case, the version of vllm is
    # usually changed. For example, if the version of vllm is "0.9.0", but when
    # it's installed from source, the version of vllm is usually set to "0.9.1".
    # In this case, developers need to set this value to "0.9.0" to make sure
    # that the correct package is installed.
    "VLLM_VERSION":
    lambda: os.getenv("VLLM_VERSION", None),
    # Whether to enable the trace recompiles from pytorch.
    "VLLM_ASCEND_TRACE_RECOMPILES":
    lambda: bool(int(os.getenv("VLLM_ASCEND_TRACE_RECOMPILES", '0'))),
    # Whether to enable fused_experts_allgather_ep. MoeInitRoutingV3 and
    # GroupedMatmulFinalizeRouting operators are combined to implement EP.
    "VLLM_ENABLE_FUSED_EXPERTS_ALLGATHER_EP":
    lambda: bool(int(os.getenv("VLLM_ENABLE_FUSED_EXPERTS_ALLGATHER_EP", '0'))
                 ),
    # Whether to enable elastic DP/EP/MC2 shrink during eager external
    # launcher rollout. Disabled by default so the original dummy-run path
    # remains unchanged unless explicitly turned on.
    "VLLM_ASCEND_ENABLE_ELASTIC_PARALLEL_SHRINK":
    lambda: get_effective_elastic_parallel_shrink_enabled(),
    # Elastic MoE strategy:
    # - lossy: shrink by masking experts that only exist on exited ranks
    # - lossless: require preloaded redundant experts and keep the original
    #   logical expert space after shrink.
    "VLLM_ASCEND_ELASTIC_MOE_MODE":
    lambda: get_effective_elastic_moe_mode(),
    # Explicit redundant expert replica count. This remains available for
    # backward compatibility, but the unified elastic execution mode may
    # derive a larger effective redundancy count from the configured floor.
    "VLLM_ASCEND_INIT_REDUNDANCY_EXPERT":
    lambda: int(os.getenv("VLLM_ASCEND_INIT_REDUNDANCY_EXPERT", "0")),
    # Fixed resident NPU expert slots per rank for elastic execution mode 2.
    # When set, mode 2 preloads enough experts to fill these slots and enters
    # CPU/NPU hybrid execution only when shrink would require more experts than
    # the configured resident capacity.
    "VLLM_ASCEND_ELASTIC_HYBRID_RESIDENT_EXPERT_SLOTS":
    lambda: _get_optional_positive_int_env(
        "VLLM_ASCEND_ELASTIC_HYBRID_RESIDENT_EXPERT_SLOTS"),
    # Whether to force expert-parallel MoE communication onto the AllToAll
    # path. This is mainly useful for performance comparisons against elastic
    # shrink, where MC2 would otherwise introduce another variable.
    "VLLM_ASCEND_FORCE_ALLTOALL_MOE":
    lambda: bool(int(os.getenv("VLLM_ASCEND_FORCE_ALLTOALL_MOE", '0'))),
    # Minimum EP size required before the runtime is allowed to select MC2.
    # Default to 2 so the elastic floor=2 hybrid path can keep using MC2
    # when the usual token-count / prefill fallbacks still allow it.
    "VLLM_ASCEND_MC2_MIN_EP_SIZE":
    lambda: int(os.getenv("VLLM_ASCEND_MC2_MIN_EP_SIZE", '2')),
    # Hybrid lossless shrink import mode used when active ranks <= 4.
    # - cpu_p2p: source rank sends CPU shadow weights directly over the CPU group
    # - npu_p2p_to_cpu: source rank sends NPU weights P2P; target stores them on CPU
    "VLLM_ASCEND_LOSSLESS_HYBRID_IMPORT_MODE":
    lambda: os.getenv("VLLM_ASCEND_LOSSLESS_HYBRID_IMPORT_MODE",
                      "cpu_p2p").lower(),
    # Optional chunk size for hybrid point-to-point expert import. Smaller
    # chunks reduce temporary memory pressure; 1 means per-expert streaming.
    "VLLM_ASCEND_LOSSLESS_HYBRID_IMPORT_CHUNK_EXPERTS":
    lambda: max(
        1, int(os.getenv("VLLM_ASCEND_LOSSLESS_HYBRID_IMPORT_CHUNK_EXPERTS",
                         "1"))),
    # Minimum active elastic compute-group size allowed for real shrink.
    # When unset, the runtime keeps the current stable behavior and does not
    # enable repeated real shrink stages by default.
    "VLLM_ASCEND_ELASTIC_MIN_COMPUTE_GROUP_SIZE":
    lambda: _get_optional_positive_power_of_two_env(
        "VLLM_ASCEND_ELASTIC_MIN_COMPUTE_GROUP_SIZE"),
    # Mode=4 keeps the initialization/preallocation floor separate from the
    # runtime tail floor. This lets validation keep floor=8 memory behavior
    # while still exercising 8->4->2->1 shrink stages.
    "VLLM_ASCEND_MODE4_RUNTIME_MIN_COMPUTE_GROUP_SIZE":
    lambda: _get_optional_positive_power_of_two_env(
        "VLLM_ASCEND_MODE4_RUNTIME_MIN_COMPUTE_GROUP_SIZE"),

    "VLLM_ASCEND_MODE5_RUNTIME_MIN_COMPUTE_GROUP_SIZE":
    lambda: _get_optional_positive_power_of_two_env(
        "VLLM_ASCEND_MODE5_RUNTIME_MIN_COMPUTE_GROUP_SIZE"),
    "VLLM_ASCEND_MODE5_REMOTE_EXPERT_FRACTION":
    lambda: float(os.getenv("VLLM_ASCEND_MODE5_REMOTE_EXPERT_FRACTION", "0.5")),
    "VLLM_ASCEND_SHRINK_AWARE_ENABLE":
    lambda: bool(int(os.getenv("VLLM_ASCEND_SHRINK_AWARE_ENABLE", "0"))),
    "VLLM_ASCEND_SHRINK_AWARE_MODE":
    lambda: os.getenv("VLLM_ASCEND_SHRINK_AWARE_MODE", "off").lower().strip(),
    "VLLM_ASCEND_SHRINK_AWARE_STAGES":
    lambda: os.getenv("VLLM_ASCEND_SHRINK_AWARE_STAGES", "8,4"),
    "VLLM_ASCEND_SHRINK_AWARE_SURVIVOR_POLICY":
    lambda: os.getenv("VLLM_ASCEND_SHRINK_AWARE_SURVIVOR_POLICY",
                      "topology_aware").lower().strip(),
    "VLLM_ASCEND_SHRINK_AWARE_TARGET_POLICY":
    lambda: os.getenv("VLLM_ASCEND_SHRINK_AWARE_TARGET_POLICY",
                      "natural").lower().strip(),
    "VLLM_ASCEND_SHRINK_AWARE_PACKAGE_TOPOLOGY":
    lambda: os.getenv("VLLM_ASCEND_SHRINK_AWARE_PACKAGE_TOPOLOGY", ""),
    "VLLM_ASCEND_SHRINK_AWARE_INTERMEDIATE_RANKS":
    lambda: os.getenv("VLLM_ASCEND_SHRINK_AWARE_INTERMEDIATE_RANKS", ""),
    "VLLM_ASCEND_SHRINK_AWARE_FINAL_RANKS":
    lambda: os.getenv("VLLM_ASCEND_SHRINK_AWARE_FINAL_RANKS", ""),
    "VLLM_ASCEND_SHRINK_AWARE_STAGE_RANKS":
    lambda: os.getenv("VLLM_ASCEND_SHRINK_AWARE_STAGE_RANKS", ""),
    "VLLM_ASCEND_SHRINK_AWARE_LENGTH_SOURCE":
    lambda: os.getenv("VLLM_ASCEND_SHRINK_AWARE_LENGTH_SOURCE",
                      "existing_regroup").lower().strip(),
    "VLLM_ASCEND_SHRINK_AWARE_MAX_OVERHEAD_RATIO":
    lambda: float(os.getenv("VLLM_ASCEND_SHRINK_AWARE_MAX_OVERHEAD_RATIO",
                            "1.10")),
    "VLLM_ASCEND_SHRINK_AWARE_MIN_WINDOW_SECONDS":
    lambda: float(os.getenv("VLLM_ASCEND_SHRINK_AWARE_MIN_WINDOW_SECONDS",
                            "1.0")),
    "VLLM_ASCEND_SHRINK_AWARE_LOGGING":
    lambda: bool(int(os.getenv("VLLM_ASCEND_SHRINK_AWARE_LOGGING", "0"))),
    "VLLM_ASCEND_SHRINK_AWARE_DRY_RUN":
    lambda: bool(int(os.getenv("VLLM_ASCEND_SHRINK_AWARE_DRY_RUN", "0"))),
    # Whether to enable DBO feature for deepseek model.
    "VLLM_ASCEND_ENABLE_DBO":
    lambda: bool(int(os.getenv("VLLM_ASCEND_ENABLE_DBO", '0'))),
    # Whether to enable the model execute time observe profile. Disable it when
    # running vllm ascend in production environment.
    "VLLM_ASCEND_MODEL_EXECUTE_TIME_OBSERVE":
    lambda: bool(int(os.getenv("VLLM_ASCEND_MODEL_EXECUTE_TIME_OBSERVE", '0'))
                 ),
    # Some models are optimized by vllm ascend. While in some case, e.g. rlhf
    # training, the optimized model may not be suitable. In this case, set this
    # value to False to disable the optimized model.
    "USE_OPTIMIZED_MODEL":
    lambda: bool(int(os.getenv('USE_OPTIMIZED_MODEL', '1'))),
    # The tolerance of the kv cache size, if the difference between the
    # actual kv cache size and the cached kv cache size is less than this value,
    # then the cached kv cache size will be used.
    "VLLM_ASCEND_KV_CACHE_MEGABYTES_FLOATING_TOLERANCE":
    lambda: int(
        os.getenv("VLLM_ASCEND_KV_CACHE_MEGABYTES_FLOATING_TOLERANCE", 64)),
    # Whether to enable the topk optimization. It's enabled by default. Please set to False if you hit any issue.
    # We'll remove this flag in the future once it's stable enough.
    "VLLM_ASCEND_ENABLE_TOPK_TOPP_OPTIMIZATION":
    lambda: bool(
        int(os.getenv("VLLM_ASCEND_ENABLE_TOPK_TOPP_OPTIMIZATION", '1'))),
    # `LLMDataDistCMgrConnector` required variable. `DISAGGREGATED_PREFILL_RANK_TABLE_PATH` is
    # used for llmdatadist to build the communication topology for kv cache transfer, it is
    # a required variable if `LLMDataDistCMgrConnector` is used as kv connector for disaggregated
    # pd. The rank table can be generated by adopting the script `gen_ranktable.sh`
    # in vllm_ascend's example folder.
    "DISAGGREGATED_PREFILL_RANK_TABLE_PATH":
    lambda: os.getenv("DISAGGREGATED_PREFILL_RANK_TABLE_PATH", None),
    # `LLMDataDistCMgrConnector` required variable. `VLLM_ASCEND_LLMDD_RPC_IP` is used as the
    # rpc communication listening ip, which will be used to receive the agent metadata from the
    # remote worker.
    "VLLM_ASCEND_LLMDD_RPC_IP":
    lambda: os.getenv("VLLM_ASCEND_LLMDD_RPC_IP", "0.0.0.0"),
    # `LLMDataDistCMgrConnector` required variable. `VLLM_ASCEND_LLMDD_RPC_PORT` is used as the
    # rpc communication listening port, which will be used to receive the agent metadata from the
    # remote worker.
    "VLLM_ASCEND_LLMDD_RPC_PORT":
    lambda: int(os.getenv("VLLM_ASCEND_LLMDD_RPC_PORT", 5557)),
    # Whether to enable mla_pa for deepseek mla decode, this flag will be removed after its available torch_npu is public accessible
    # and the mla_pa will be the default path of deepseek decode path.
    "VLLM_ASCEND_MLA_PA":
    lambda: int(os.getenv("VLLM_ASCEND_MLA_PA", 0)),
    # Whether to enable MatmulAllReduce fusion kernel when tensor parallel is enabled.
    # this feature is supported in A2, and eager mode will get better performance.
    "VLLM_ASCEND_ENABLE_MATMUL_ALLREDUCE":
    lambda: bool(int(os.getenv("VLLM_ASCEND_ENABLE_MATMUL_ALLREDUCE", '0'))),
    # Whether to enable FlashComm optimization when tensor parallel is enabled.
    # This feature will get better performance when concurrency is large.
    "VLLM_ASCEND_ENABLE_FLASHCOMM":
    lambda: bool(int(os.getenv("VLLM_ASCEND_ENABLE_FLASHCOMM", '0'))),
    # Whether to enable MLP weight prefetch, only used in small concurrency.
    "VLLM_ASCEND_ENABLE_PREFETCH_MLP":
    lambda: bool(int(os.getenv("VLLM_ASCEND_ENABLE_PREFETCH_MLP", '0'))),
    # buffer size for gate up prefetch
    "VLLM_ASCEND_MLP_GATE_UP_PREFETCH_SIZE":
    lambda: int(
        os.getenv("VLLM_ASCEND_MLP_GATE_UP_PREFETCH_SIZE", 18 * 1024 * 1024)),
    # buffer size for down proj prefetch
    "VLLM_ASCEND_MLP_DOWN_PREFETCH_SIZE":
    lambda: int(
        os.getenv("VLLM_ASCEND_MLP_DOWN_PREFETCH_SIZE", 18 * 1024 * 1024)),
    # Whether to enable dense model and general optimizations for better performance.
    # Since we modified the base parent class `linear`, this optimization is also applicable to other model types.
    # However, there might be hidden issues, and it is currently recommended to prioritize its use with dense models.
    "VLLM_ASCEND_ENABLE_DENSE_OPTIMIZE":
    lambda: bool(int(os.getenv("VLLM_ASCEND_ENABLE_DENSE_OPTIMIZE", '0'))),
    # Whether to enable mlp optimize when tensor parallel is enabled.
    # this feature in eager mode will get better performance.
    "VLLM_ASCEND_ENABLE_MLP_OPTIMIZE":
    lambda: bool(int(os.getenv("VLLM_ASCEND_ENABLE_MLP_OPTIMIZE", '0'))),
    # Determine the number of physical devices in a non-full-use scenario
    # caused by the initialization of the Mooncake connector.
    "PHYSICAL_DEVICES":
    lambda: os.getenv("PHYSICAL_DEVICES", None),
    # Whether to enable msMonitor tool to monitor the performance of vllm-ascend.
    "MSMONITOR_USE_DAEMON":
    lambda: bool(int(os.getenv("MSMONITOR_USE_DAEMON", '0'))),
    # Timeout (in seconds) for delayed KVCache block release. In the prefill
    # node, if a request is marked for delayed KV block release and the blocks
    # are not freed within this timeout, they will be forcibly released.
    "VLLM_ASCEND_KVCACHE_DELAY_FREE_TIMEOUT":
    lambda: int(os.getenv("VLLM_ASCEND_KVCACHE_DELAY_FREE_TIMEOUT", 250)),
    "VLLM_ASCEND_ENABLE_MLAPO":
    lambda: bool(int(os.getenv("VLLM_ASCEND_ENABLE_MLAPO", '0'))),
}

# end-env-vars-definition


def __getattr__(name: str):
    # lazy evaluation of environment variables
    if name in env_variables:
        return env_variables[name]()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return list(env_variables.keys())
