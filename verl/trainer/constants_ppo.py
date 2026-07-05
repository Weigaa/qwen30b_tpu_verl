# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

import json
import os

from ray._private.runtime_env.constants import RAY_JOB_CONFIG_JSON_ENV_VAR

MODE1_PARITY_RUNTIME_ENV_KEYS = (
    "VLLM_ASCEND_ELASTIC_EXECUTION_MODE",
    "VLLM_ASCEND_ELASTIC_MIN_COMPUTE_GROUP_SIZE",
    "VLLM_ASCEND_GROUP_SEQ_DIGEST_LOG",
    "VLLM_ASCEND_ELASTIC_HCCL_SEQ_DIGEST_LOG",
    "VLLM_ASCEND_ELASTIC_HCCL_SEQ_SUMMARY_LOG",
    "VLLM_ASCEND_MODE1_SEQ_CHECKPOINT_LOG",
    "VLLM_ASCEND_MOE_DISPATCH_SEQ_DIGEST_LOG",
    "VLLM_ASCEND_MOE_DISPATCH_SEQ_DIGEST_LIMIT",
    "VLLM_ASCEND_MOE_DISPATCH_SEQ_DIGEST_SAMPLE_EVERY",
    "VLLM_ASCEND_MOE_DISPATCH_SEQ_DIGEST_SLOW_MS",
    "VLLM_ASCEND_MODE1_STASH_CLASSIFY_LOG",
    "VLLM_ASCEND_MODE1_PARITY_DEFER_GROUP_DESTROY",
    "VLLM_ASCEND_MODE1_PARITY_DEFER_GROUP_DESTROY_ON_FULL_RESTORE",
    "VLLM_ASCEND_MODE1_PARITY_DEFER_GROUP_DESTROY_ON_PRE_REBUILD",
    "VLLM_ASCEND_MODE1_PARITY_DEFER_GROUP_DESTROY_ON_STASH",
    "VLLM_ASCEND_MODE1_PARITY_DEFER_DESTROY_FLOOR_GROUP_SIZES",
    "VLLM_ASCEND_MODE1_PARITY_DESTROY_DEVICE_PG_ON_RETIRE",
    "VLLM_ASCEND_MODE1_PARITY_DESTROY_CPU_PG_ON_RETIRE",
    "VLLM_ASCEND_MODE1_PARITY_DESTROY_CPU_PG_ON_NON_MEMBER_PLACEHOLDER",
    "VLLM_ASCEND_MODE1_FORCE_NON_MEMBER_PLACEHOLDER_NEW_GROUP",
    "VLLM_ASCEND_MODE1_ALLOW_REBUILD_CACHE_HIT_WITH_PLACEHOLDER",
    "VLLM_ASCEND_MODE1_PARITY_POST_RESTORE_MC2_WARMUP",
    "VLLM_ASCEND_MODE1_PARITY_POST_RESTORE_MC2_WARMUP_TOKENS",
    "VLLM_ASCEND_MODE1_PARITY_POST_RESTORE_ALLTOALL_WARMUP",
    "VLLM_ASCEND_MODE1_PARITY_POST_RESTORE_ALLTOALL_WARMUP_TOKENS",
    "VLLM_ASCEND_MODE1_PARITY_POST_RESTORE_ALLTOALL_WARMUP_MAX_TOKENS",
    "VLLM_ASCEND_MODE1_PARITY_POST_SHRINK_WARMUP",
    "VLLM_ASCEND_MODE1_PARITY_PRE_MOE_WARMUP_EMPTY_CACHE",
    "VLLM_ASCEND_MODE1_PARITY_RELEASE_WARMUP_CACHE",
    "VLLM_ASCEND_REPEAT_POST_SHRINK_MOE_DISPATCH_WARMUP",
    "VLLM_ASCEND_POST_SHRINK_STAGING_RELEASE_EMPTY_CACHE",
    "VLLM_ASCEND_POST_SHRINK_STAGING_RELEASE_SYNC",
    "VLLM_ASCEND_MODE1_PARITY_DROP_OLD_FLOOR_BEFORE_REFRESH",
    "VLLM_ASCEND_MODE1_PARITY_DROP_OLD_FLOOR_BEFORE_REBUILD",
    "VLLM_ASCEND_MODE1_PARITY_BARRIER_BEFORE_OLD_FLOOR_DROP",
    "VLLM_ASCEND_MODE1_PARITY_BARRIER_AFTER_OLD_FLOOR_DROP",
    "VLLM_ASCEND_MODE1_PARITY_RELEASE_LIVE_OLD_FLOOR_ON_REBUILD",
    "VLLM_ASCEND_MODE1_PARITY_FORCE_CLEANUP_AFTER_FLOOR_GROUP_RELEASE",
    "VLLM_ASCEND_MODE1_PARITY_MC2_MEM_LOG",
    "VLLM_ASCEND_MODE1_PARITY_DROP_STALE_CACHE_AFTER_SHRINK",
    "VLLM_ASCEND_MODE1_PARITY_KEEP_FLOOR4_GROUP_CACHE",
    "VLLM_ASCEND_MODE1_PARITY_KEEP_FLOOR4_GROUP_KINDS",
    "VLLM_ASCEND_MODE1_PARITY_KEEP_FULLWORLD_EP_CACHE",
    "VLLM_ASCEND_MODE1_PARITY_KEEP_MC2_GROUP_CACHE",
    "VLLM_ASCEND_DISABLE_ELASTIC_MC2_GROUP_CACHE",
    "VLLM_ASCEND_MODE1_PARITY_SINGLE_LIVE_MC2_GROUP",
    "VLLM_ASCEND_MODE1_CPU_DP_METADATA_SYNC",
    "VLLM_ASCEND_MODE1_PARITY_MAX_KV_TOKENS",
    "VLLM_ASCEND_MODE1_PARITY_MAX_KV_TOKENS_FLOOR4",
)

PPO_RAY_RUNTIME_ENV = {
    "env_vars": {
        "TOKENIZERS_PARALLELISM": "true",
        "NCCL_DEBUG": "WARN",
        "VLLM_LOGGING_LEVEL": "WARN",
        "VLLM_ALLOW_RUNTIME_LORA_UPDATING": "true",
        "CUDA_DEVICE_MAX_CONNECTIONS": "1",
        # To prevent hanging or crash during synchronization of weights between actor and rollout
        # in disaggregated mode. See:
        # https://docs.vllm.ai/en/latest/usage/troubleshooting.html?h=nccl_cumem_enable#known-issues
        # https://github.com/vllm-project/vllm/blob/c6b0a7d3ba03ca414be1174e9bd86a97191b7090/vllm/worker/worker_base.py#L445
        "NCCL_CUMEM_ENABLE": "0",
    },
}


def get_ppo_ray_runtime_env():
    """
    A filter function to return the PPO Ray runtime environment.
    To avoid repeat of some environment variables that are already set.
    """
    working_dir = (
        json.loads(os.environ.get(RAY_JOB_CONFIG_JSON_ENV_VAR, "{}")).get("runtime_env", {}).get("working_dir", None)
    )

    runtime_env = {
        "env_vars": PPO_RAY_RUNTIME_ENV["env_vars"].copy(),
        **({"working_dir": None} if working_dir is None else {}),
    }
    for key in list(runtime_env["env_vars"].keys()):
        if os.environ.get(key) is not None:
            runtime_env["env_vars"].pop(key, None)
    for key in MODE1_PARITY_RUNTIME_ENV_KEYS:
        if os.environ.get(key) is not None:
            runtime_env["env_vars"][key] = os.environ[key]
    return runtime_env
