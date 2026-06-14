set -ex

export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_NPU_ALLOC_CONF="expandable_segments:True"

export ASCEND_HOME_PATH=/usr/local/Ascend/ascend-toolkit
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/asdsip/set_env.sh
# ATB's set_env.sh can return non-zero under `set -e` when it auto-detects
# torch's CXX ABI and an internal grep misses. Passing the ABI explicitly
# avoids that false failure and keeps the environment initialization stable.
source /usr/local/Ascend/nnal/atb/set_env.sh --cxx_abi=1

export HYDRA_FULL_ERROR=1
#export ASCEND_LAUNCH_BLOCKING=1         
export RAY_DEDUP_LOGS=0                   

export ASCEND_GLOBAL_EVENT_ENABLE=0         
export ASCEND_SLOG_PRINT_TO_STDOUT=0       
export ASCEND_GLOBAL_LOG_LEVEL=3           

export HCCL_CONNECT_TIMEOUT=360
export HCCL_IF_BASE_PORT=${HCCL_IF_BASE_PORT:-64021}
export HCCL_EXEC_TIMEOUT=360
export CUDA_DEVICE_MAX_CONNECTIONS=1

if [[ -z "${GLOO_SOCKET_IFNAME:-}" ]]; then
    default_gloo_ifname=$(
        awk '$2 == "00000000" && $1 != "lo" { print $1; exit }' \
            /proc/net/route 2>/dev/null || true
    )
    if [[ -z "${default_gloo_ifname}" ]]; then
        default_gloo_ifname=$(
            for iface_path in /sys/class/net/*; do
                iface=$(basename "${iface_path}")
                [[ "${iface}" == "lo" ]] && continue
                [[ -r "${iface_path}/operstate" ]] || continue
                [[ "$(cat "${iface_path}/operstate")" == "up" ]] || continue
                echo "${iface}"
                break
            done
        )
    fi
    if [[ -n "${default_gloo_ifname}" ]]; then
        export GLOO_SOCKET_IFNAME="${default_gloo_ifname}"
    fi
fi
echo "[gloo] GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-}"

export MASTER_PORT=${MASTER_PORT:-23300}    # vllm port error
export D2D_DATA_TRANSFER=1
export VLLM_USE_V1=1
export PRINT_MEMORY=1
export USE_ALLTOALL_OVERLAP=1
export HCCL_OP_EXPANSION_MODE=AIV
export VLLM_LOGGING_LEVEL=INFO
export VLLM_ASCEND_FORCE_ALLTOALL_MOE=${VLLM_ASCEND_FORCE_ALLTOALL_MOE:-0}  # 1: force AllToAll for EP MoE, 0: allow MC2
if [[ "${VLLM_ASCEND_FORCE_ALLTOALL_MOE}" == "1" ]]; then
    export VLLM_ENABLE_MC2=0
else
export VLLM_ENABLE_MC2=1
fi
export VLLM_DP_SIZE=${VLLM_DP_SIZE:-16}                        # world_size // rollout.tp_sizebaseline dummy-run, 1: redundant experts only, 2: redundant experts + CPU/NPU hybrid tail
export HCCL_BUFFSIZE=800

export TASK_QUEUE_ENABLE=2

export VLLM_ENABLE_FIX_ROUTE=0    
export VLLM_MODEL_EXECUTE_TIME_OBSERVE=0     # decode prefill的耗时打印

#extra env in qwen3_235b_env.sh
# Recipe features
export VLLM_ENABLE_GRAPH_MODE=0             # 0: eager mode, 1: graph mode
export VLLM_ENABLE_EXPERT_PARALLEL=1        # Enable EP in vLLM rollout.
export VLLM_CHUNK_MOE_SIZE=512              # The minimum block size set for prefill computation partition.
export ALL_TO_ALL_RESHARD=1                 # Enable EP to reshard parameters with AllToAllV (without communication redundancy).
export USE_ALLTOALL_OVERLAP=1               # Enable to overlap communication in EP with computation to hide MoE communication latency. Should be consistent with model conversion config.
export VLLM_ENABLE_EPLB=0                   # 0: disable eplb, 1: enable eplb
export USE_HDP=0                            # 0: disable hdp, 1: enable hdp
export ROLLOUT_REBALANCE_ENABLE=0          # 0: disable rollout rebalance, 1: enable rollout rebalance

#关闭看门狗
export HCCL_ASYNC_ERROR_HANDLING=1

#Train Drafter开关
export VLLM_ASCEND_ENABLE_DRAFT_TRAIN=0
export VLLM_ASCEND_DRAFT_WARMUP_ON_INIT=1
export VLLM_ASCEND_DRAFT_LR=1e-4
export VLLM_ASCEND_DRAFT_REUSE_TARGET_EMB_LM=1
export VLLM_ASCEND_DRAFT_VOCAB_SIZE=4096
export VLLM_ASCEND_DRAFT_QUEUE_SIZE=4
export VLLM_ASCEND_DRAFT_MAX_SEQ_LEN=16384
export VLLM_ASCEND_DRAFT_TRAIN_DTYPE=bf16
export VLLM_ASCEND_DRAFT_ATTN_IMPL=sdpa
export VLLM_ASCEND_DRAFT_ATTN_CHUNK_SIZE=1024
export VLLM_ASCEND_DRAFT_LORA_ENABLE=0
export VLLM_ASCEND_DRAFT_LORA_BACKEND=custom
export VLLM_ASCEND_DRAFT_LORA_RANK=8
export VLLM_ASCEND_DRAFT_LORA_ALPHA=16
export VLLM_ASCEND_DRAFT_LORA_DROPOUT=0.0
export VLLM_ASCEND_DRAFT_LORA_TARGET_MODULES=q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj,fc
export VLLM_ASCEND_DRAFT_SPARSE_KL_ENABLE=1
export VLLM_ASCEND_DRAFT_SPARSE_KL_TOPK=64
export VLLM_ASCEND_DRAFT_COMPUTE_ACCURACY=0
# micro-step training: one dummy window only runs a seq chunk, then accumulate across windows
export VLLM_ASCEND_DRAFT_MICRO_SEQ_LEN=1
export VLLM_ASCEND_DRAFT_GRAD_ACCUM_STEPS=4096

# Draft profile mode:
#   breakdown   -> 正常训练中打印 draft 分段耗时（默认）
#   profile_only -> 只跑 draft profile step 后退出进程
export DRAFT_PROFILE_MODE=${DRAFT_PROFILE_MODE:-breakdown}

# 默认：在正常训练里看 draft 的分段耗时
export VLLM_ASCEND_DRAFT_PROFILE_ONLY=0
export VLLM_ASCEND_DRAFT_NPU_PROFILE=0
# export VLLM_ASCEND_DRAFT_NPU_PROFILE_STEPS=10
# export VLLM_ASCEND_DRAFT_NPU_PROFILE_WAIT=0
# export VLLM_ASCEND_DRAFT_NPU_PROFILE_WARMUP=2
# export VLLM_ASCEND_DRAFT_NPU_PROFILE_ACTIVE=4
# export VLLM_ASCEND_DRAFT_NPU_PROFILE_REPEAT=1
export VLLM_ASCEND_DRAFT_NPU_PROFILE_DIR=./result/profiler/draft_${DRAFT_PROFILE_MODE}
export VLLM_ASCEND_DRAFT_STARTUP_WARMUP_STEPS=5
export VLLM_ASCEND_DRAFT_WARMUP_STEPS=5
export VLLM_ASCEND_DRAFT_PROFILE_BREAKDOWN=1
export VLLM_ASCEND_DRAFT_PROFILE_SYNC=0
export VLLM_ASCEND_DRAFT_ASYNC_TRAIN=0
#配置native模式还是custom模式
export VLLM_ASCEND_REGISTER_CUSTOM_MODELS=${VLLM_ASCEND_REGISTER_CUSTOM_MODELS:-1}
export VLLM_ASCEND_CUSTOM_MODE1_KV_MATERIALIZE_HEADROOM_BYTES=0
export VLLM_ASCEND_CUSTOM_MODE1_DEBUG=0
export VLLM_ASCEND_CUSTOM_MODE1_TIMING_EVENTS=0
export VLLM_ASCEND_CUSTOM_MODE1_KV_DIAG=${VLLM_ASCEND_CUSTOM_MODE1_KV_DIAG:-0}

# 弹性执行模式:
# 0: baseline dummy-run
# 1: 冗余专家模式
# 2: CPU/NPU 混合模式
# 3: 无冗余专家 + 跨层双缓冲 hybrid tail
# 4: 无冗余专家 + 远程 NPU expert cache 双缓冲
# 5: 无冗余专家 + CPU shadow 和远程 NPU cache 混合双缓冲
#    使用建议:
#    - mode=3 主要面向 shrink 到 <=8 rank 之后的 MoE 主路径
#    - 若希望允许 2 -> 1 的 single-rank no-EP tail，保持
#      VLLM_ASCEND_ELASTIC_MIN_COMPUTE_GROUP_SIZE=1
#    - mode=3 不依赖冗余专家；运行时 expert double buffer 固定为 128 expert slots
#    - VLLM_ASCEND_ELASTIC_HYBRID_RESIDENT_EXPERT_SLOTS 在 mode=3 下只保留
#      primary prefix 语义，不再决定运行时 buffer 容量
export VLLM_ASCEND_ELASTIC_EXECUTION_MODE=${VLLM_ASCEND_ELASTIC_EXECUTION_MODE:-1}
VLLM_ASCEND_ELASTIC_MIN_COMPUTE_GROUP_SIZE_WAS_SET=0
if [[ -n "${VLLM_ASCEND_ELASTIC_MIN_COMPUTE_GROUP_SIZE+x}" ]]; then
    VLLM_ASCEND_ELASTIC_MIN_COMPUTE_GROUP_SIZE_WAS_SET=1
fi
# 弹性缩容的最小计算组:
#   1  -> 允许在 2-rank 阶段后进入 single-rank no-EP tail
#   2/4/8/16 -> 最多缩到该 floor 结束，不再进入 1-rank tail
export VLLM_ASCEND_ELASTIC_MIN_COMPUTE_GROUP_SIZE=${VLLM_ASCEND_ELASTIC_MIN_COMPUTE_GROUP_SIZE:-2}
# mode=2 时每个 rank 固定保留的 NPU resident expert 槽位数
# mode=3 时该值不控制双缓冲大小；当前 runtime double buffer 固定为 128 experts
export VLLM_ASCEND_ELASTIC_HYBRID_RESIDENT_EXPERT_SLOTS=${VLLM_ASCEND_ELASTIC_HYBRID_RESIDENT_EXPERT_SLOTS:-8}
# mode=1 floor=2 local-stability path:
# keep payload transfer as direct NPU->NPU, but avoid the NPU scalar slot-map
# lookup that can turn into a 350s runtime/HCCL synchronization on this host.
export VLLM_ASCEND_MODE1_PARITY_MAX_KV_TOKENS=${VLLM_ASCEND_MODE1_PARITY_MAX_KV_TOKENS:-377344}
export VLLM_ASCEND_MODE1_PARITY_KEEP_FULLWORLD_EP_CACHE=${VLLM_ASCEND_MODE1_PARITY_KEEP_FULLWORLD_EP_CACHE:-1}
export VLLM_ASCEND_MODE1_PARITY_KEEP_MC2_GROUP_CACHE=${VLLM_ASCEND_MODE1_PARITY_KEEP_MC2_GROUP_CACHE:-0}
export VLLM_ASCEND_MODE1_PARITY_DROP_STALE_CACHE_AFTER_SHRINK=${VLLM_ASCEND_MODE1_PARITY_DROP_STALE_CACHE_AFTER_SHRINK:-0}
export VLLM_ASCEND_MODE1_PARITY_SINGLE_LIVE_MC2_GROUP=${VLLM_ASCEND_MODE1_PARITY_SINGLE_LIVE_MC2_GROUP:-0}
export VLLM_ASCEND_DISABLE_ELASTIC_MC2_GROUP_CACHE=${VLLM_ASCEND_DISABLE_ELASTIC_MC2_GROUP_CACHE:-0}
export VLLM_ASCEND_MODE1_PARITY_POST_RESTORE_ALLTOALL_WARMUP=${VLLM_ASCEND_MODE1_PARITY_POST_RESTORE_ALLTOALL_WARMUP:-0}
export VLLM_ASCEND_MODE1_BATCH_DIRECT_NPU_IMPORT=${VLLM_ASCEND_MODE1_BATCH_DIRECT_NPU_IMPORT:-1}
export VLLM_ASCEND_MODE1_ALLOW_SCALAR_DIRECT_NPU_IMPORT=${VLLM_ASCEND_MODE1_ALLOW_SCALAR_DIRECT_NPU_IMPORT:-0}
export VLLM_ASCEND_MODE1_ALLOW_BATCH_INDEX_SELECT_EXPORT=${VLLM_ASCEND_MODE1_ALLOW_BATCH_INDEX_SELECT_EXPORT:-0}
export VLLM_ASCEND_MODE1_DIRECT_NPU_IMPORT_BATCH_EXPERTS=${VLLM_ASCEND_MODE1_DIRECT_NPU_IMPORT_BATCH_EXPERTS:-8}
export VLLM_ASCEND_POST_SHRINK_STAGING_RELEASE_EMPTY_CACHE=${VLLM_ASCEND_POST_SHRINK_STAGING_RELEASE_EMPTY_CACHE:-0}
export VLLM_ASCEND_POST_SHRINK_STAGING_RELEASE_SYNC=${VLLM_ASCEND_POST_SHRINK_STAGING_RELEASE_SYNC:-0}
export VLLM_ASCEND_MODE1_PARITY_SYNC_AFTER_MC2_GROUP_DESTROY=${VLLM_ASCEND_MODE1_PARITY_SYNC_AFTER_MC2_GROUP_DESTROY:-0}
export VLLM_ASCEND_MODE1_PARITY_GC_AFTER_MC2_GROUP_DESTROY=${VLLM_ASCEND_MODE1_PARITY_GC_AFTER_MC2_GROUP_DESTROY:-0}
export VLLM_ASCEND_MODE1_PARITY_RELEASE_DP_WARMUP_CACHE=${VLLM_ASCEND_MODE1_PARITY_RELEASE_DP_WARMUP_CACHE:-0}
export VLLM_ASCEND_MODE1_PARITY_RELEASE_WARMUP_CACHE=${VLLM_ASCEND_MODE1_PARITY_RELEASE_WARMUP_CACHE:-0}
export VLLM_ASCEND_MODE1_PARITY_BARRIER_BEFORE_REFRESH=${VLLM_ASCEND_MODE1_PARITY_BARRIER_BEFORE_REFRESH:-1}
export VLLM_ASCEND_MODE1_PARITY_DROP_OLD_FLOOR_BEFORE_REFRESH=${VLLM_ASCEND_MODE1_PARITY_DROP_OLD_FLOOR_BEFORE_REFRESH:-1}
export VLLM_ASCEND_MODE1_PARITY_DROP_OLD_FLOOR_BEFORE_REBUILD=${VLLM_ASCEND_MODE1_PARITY_DROP_OLD_FLOOR_BEFORE_REBUILD:-0}
export VLLM_ASCEND_MODE1_PARITY_RELEASE_LIVE_OLD_FLOOR_ON_REBUILD=${VLLM_ASCEND_MODE1_PARITY_RELEASE_LIVE_OLD_FLOOR_ON_REBUILD:-1}
export VLLM_ASCEND_MODE1_PARITY_PRE_REBUILD_DESTROY_FLOOR_GROUP_SIZES=${VLLM_ASCEND_MODE1_PARITY_PRE_REBUILD_DESTROY_FLOOR_GROUP_SIZES:-1,2,4,8}
export VLLM_ASCEND_MODE1_PARITY_DEFER_GROUP_DESTROY=${VLLM_ASCEND_MODE1_PARITY_DEFER_GROUP_DESTROY:-1}
export VLLM_ASCEND_MODE1_PARITY_DEFER_DESTROY_FLOOR_GROUP_SIZES=${VLLM_ASCEND_MODE1_PARITY_DEFER_DESTROY_FLOOR_GROUP_SIZES:-1,2,4,8}
export VLLM_ASCEND_MODE1_PARITY_SYNC_BEFORE_DEVICE_PG_RETIRE=${VLLM_ASCEND_MODE1_PARITY_SYNC_BEFORE_DEVICE_PG_RETIRE:-1}
export VLLM_ASCEND_MODE1_PARITY_DESTROY_DEVICE_PG_ON_RETIRE=${VLLM_ASCEND_MODE1_PARITY_DESTROY_DEVICE_PG_ON_RETIRE:-1}
export VLLM_ASCEND_MODE1_PARITY_ENABLE_POST_SHRINK_DP_WARMUP=${VLLM_ASCEND_MODE1_PARITY_ENABLE_POST_SHRINK_DP_WARMUP:-0}
export VLLM_ASCEND_MODE1_CPU_DP_METADATA_SYNC=${VLLM_ASCEND_MODE1_CPU_DP_METADATA_SYNC:-1}
export VLLM_ASCEND_MODE1_PARITY_BARRIER_BEFORE_OLD_FLOOR_DROP=${VLLM_ASCEND_MODE1_PARITY_BARRIER_BEFORE_OLD_FLOOR_DROP:-1}
export VLLM_ASCEND_MODE1_PARITY_OLD_FLOOR_DROP_VERBOSE=${VLLM_ASCEND_MODE1_PARITY_OLD_FLOOR_DROP_VERBOSE:-0}
export VLLM_ASCEND_MODE1_PARITY_KEEP_FLOOR4_GROUP_CACHE=${VLLM_ASCEND_MODE1_PARITY_KEEP_FLOOR4_GROUP_CACHE:-0}
export VLLM_ASCEND_MODE1_PARITY_KEEP_FLOOR4_GROUP_KINDS=${VLLM_ASCEND_MODE1_PARITY_KEEP_FLOOR4_GROUP_KINDS:-dp,ep}
if [[ "${VLLM_ASCEND_ELASTIC_EXECUTION_MODE}" == "1" \
      && "${VLLM_ASCEND_ELASTIC_MIN_COMPUTE_GROUP_SIZE}" == "2" \
      && ! "${VLLM_ASCEND_MODE1_ALLOW_SCALAR_DIRECT_NPU_IMPORT}" =~ ^(1|true|TRUE|yes|YES|on|ON)$ ]]; then
    export VLLM_ASCEND_MODE1_BATCH_DIRECT_NPU_IMPORT=1
fi
# mode=3 step-1: allow next-layer NPU resident experts to prefetch into the
# alternate runtime buffer. CPU-only experts still fill synchronously at bind.
export VLLM_ASCEND_MODE3_ASYNC_NPU_PREFETCH=${VLLM_ASCEND_MODE3_ASYNC_NPU_PREFETCH:-1}
# mode=3 keeps the configured floor for the first shrink, but the tail can
# continue shrinking below that floor when MC2-compatible active ranks remain.
export VLLM_ASCEND_MODE3_ALLOW_BELOW_FLOOR_TAIL=${VLLM_ASCEND_MODE3_ALLOW_BELOW_FLOOR_TAIL:-0}
# mode=3 step-2: prefetch CPU-only experts into a plain NPU staging buffer on
# a separate stream.
export VLLM_ASCEND_MODE3_ASYNC_CPU_STAGE=${VLLM_ASCEND_MODE3_ASYNC_CPU_STAGE:-1}
# mode=3 step-3: after CPU staging, pack CPU-only experts into their final
# runtime slots on the CPU staging stream. The main prefetch stream only waits
# for cpu_pack_event before marking the alternate buffer ready.
export VLLM_ASCEND_MODE3_ASYNC_CPU_PACK=${VLLM_ASCEND_MODE3_ASYNC_CPU_PACK:-1}
# mode=3 step-4: bind current layer by inserting a device-side wait on the
# slot ready event, instead of blocking Python with event.synchronize().
export VLLM_ASCEND_MODE3_DEVICE_READY_WAIT=${VLLM_ASCEND_MODE3_DEVICE_READY_WAIT:-1}
# mode=3 step-5: try direct CPU shadow row copies into the final runtime
# expert slots on the CPU prefetch stream, bypassing the staging buffer.
#
# Default to staging for the next A/B run so it can be compared with the
# previous direct-slot run. Override to 1 to restore direct CPU -> runtime slot.
export VLLM_ASCEND_MODE3_DIRECT_CPU_SLOT=${VLLM_ASCEND_MODE3_DIRECT_CPU_SLOT:-1}
# mode=3 step-6: coalesce contiguous expert slot copies into larger slice
# copies. CPU direct bulk is experimental because the runtime slot may use a
# formatted layout; set it to 0 to fall back to the proven per-expert direct
# copies while keeping NPU/staging bulk copies enabled.
export VLLM_ASCEND_MODE3_BULK_NPU_COPY=${VLLM_ASCEND_MODE3_BULK_NPU_COPY:-1}
export VLLM_ASCEND_MODE3_BULK_CPU_STAGE=${VLLM_ASCEND_MODE3_BULK_CPU_STAGE:-1}
export VLLM_ASCEND_MODE3_BULK_CPU_DIRECT=${VLLM_ASCEND_MODE3_BULK_CPU_DIRECT:-1}
# mode=3 experimental: reuse each layer's resident prefix weight buffer as the
# runtime buffer when resident NPU experts already occupy the required dense
# prefix slots. Keep disabled by default to preserve the strict two-runtime-
# buffer execution model.
export VLLM_ASCEND_MODE3_LAYER_LOCAL_BUFFER=${VLLM_ASCEND_MODE3_LAYER_LOCAL_BUFFER:-0}
# mode=3 dispatch/group optimization:
#   EXPERT_TOKEN_NUMS_TYPE=1 asks MC2 dispatch to return per-expert counts
#   directly, avoiding a per-layer cumulative->counts conversion.
#   ACTIVE_ROWS_SYNC=1 restores active_rows diagnostics but adds host sync.
export VLLM_ASCEND_MODE3_USE_FUSED_EXPERTS_PATH=${VLLM_ASCEND_MODE3_USE_FUSED_EXPERTS_PATH:-1}
export VLLM_ASCEND_MODE3_EXPERT_TOKEN_NUMS_TYPE=${VLLM_ASCEND_MODE3_EXPERT_TOKEN_NUMS_TYPE:-0}
export VLLM_ASCEND_MODE3_ACTIVE_ROWS_SYNC=${VLLM_ASCEND_MODE3_ACTIVE_ROWS_SYNC:-0}
# mode=3 deeper shrink stages lazily allocate MC2/HCCL dispatcher resources.
# Prime the dispatcher after each shrink without running full MoE/KV warmup.
export VLLM_ASCEND_MODE3_POST_SHRINK_MOE_WARMUP=${VLLM_ASCEND_MODE3_POST_SHRINK_MOE_WARMUP:-1}
export VLLM_ASCEND_MODE3_POST_SHRINK_MOE_WARMUP_TOKENS=${VLLM_ASCEND_MODE3_POST_SHRINK_MOE_WARMUP_TOKENS:-32}
# mode=3 floor=1 hits a large low-floor MC2/HCCL workspace at the 4->2
# transition. Keep this reservation mode3-only so mode1/mode2/mode4 KV sizing
# is untouched, and prefer stable MC2 over falling back to all2all.
export VLLM_ASCEND_MODE3_LOW_FLOOR_MC2_WORKSPACE_HEADROOM_BYTES=${VLLM_ASCEND_MODE3_LOW_FLOOR_MC2_WORKSPACE_HEADROOM_BYTES:-5368709120}
# mode=3 profile controls:
#   TRANSFER_LOG=0 closes high-frequency binding/prefetch logs.
#   TIMING_LOG/TIMING_SYNC default off for performance runs. Override them to
#   1 only when collecting compute-vs-prefetch diagnostics.
export VLLM_ASCEND_MODE3_TRANSFER_LOG=${VLLM_ASCEND_MODE3_TRANSFER_LOG:-0}
export VLLM_ASCEND_MODE3_TRANSFER_PLAN_LOG=${VLLM_ASCEND_MODE3_TRANSFER_PLAN_LOG:-0}
export VLLM_ASCEND_MODE3_TRANSFER_PLAN_FIRST_N=${VLLM_ASCEND_MODE3_TRANSFER_PLAN_FIRST_N:-4}
export VLLM_ASCEND_MODE3_TIMING_LOG=${VLLM_ASCEND_MODE3_TIMING_LOG:-0}
export VLLM_ASCEND_MODE3_TIMING_SYNC=${VLLM_ASCEND_MODE3_TIMING_SYNC:-0}
export VLLM_ASCEND_MODE3_TIMING_EVERY=${VLLM_ASCEND_MODE3_TIMING_EVERY:-1024}
export VLLM_ASCEND_MODE3_TIMING_FIRST_N=${VLLM_ASCEND_MODE3_TIMING_FIRST_N:-1}
export VLLM_ASCEND_MODE3_TIMING_LAYERS=${VLLM_ASCEND_MODE3_TIMING_LAYERS:-all}
#控制moe记录是否开启
export VLLM_MOE_PATTERN_STATS=${VLLM_MOE_PATTERN_STATS:-0}  # 1: enable MoE pattern stats collection, 0: disable
export VLLM_MOE_STATS=${VLLM_MOE_PATTERN_STATS}
export VLLM_MOE_STATS_DIR=${VLLM_MOE_STATS_DIR:-./moe_stats}
# Per-rank dummy waste accounting for baseline runs. Each dummy run emits one
# parseable line:
#   Dummy waste timing: rank=... dummy_wall_ms=... dummy_moe_selected_layers=... dummy_moe_effective_ms=... dummy_wasted_ms=...
# Aggregate by rank as:
#   wasted_time = sum(dummy_wasted_ms)
#   effective_time = rollout_wall_time - sum(dummy_wall_ms) + sum(dummy_moe_effective_ms)
# dummy_moe_effective_ms only includes MoE layers whose selected experts hit
# the local rank; dummy attention and unselected MoE layers are counted wasted.
export VLLM_ASCEND_DUMMY_WASTE_TIMING=${VLLM_ASCEND_DUMMY_WASTE_TIMING:-0}
export VLLM_ASCEND_DUMMY_WASTE_TIMING_SYNC=${VLLM_ASCEND_DUMMY_WASTE_TIMING_SYNC:-0}
export VLLM_ASCEND_DUMMY_WASTE_TIMING_PROFILE=${VLLM_ASCEND_DUMMY_WASTE_TIMING_PROFILE:-0}
export VLLM_ASCEND_DUMMY_WASTE_PROFILE_MARKERS=${VLLM_ASCEND_DUMMY_WASTE_PROFILE_MARKERS:-0}
export VLLM_ASCEND_ELASTIC_UTIL_LOG=${VLLM_ASCEND_ELASTIC_UTIL_LOG:-0}
export VLLM_ASCEND_ELASTIC_UTIL_BUCKET_STEPS=${VLLM_ASCEND_ELASTIC_UTIL_BUCKET_STEPS:-500}
export VLLM_ASCEND_FULL_REDUNDANCY_EXPERIMENT_LOG=${VLLM_ASCEND_FULL_REDUNDANCY_EXPERIMENT_LOG:-0}
# Mode=4 is still experimental. For validation runs, prefer a small, targeted
# stability budget instead of enabling broad generic elastic headrooms, which
# can over-shrink KV cache and introduce preemption.
export VLLM_ASCEND_MODE4_STABILITY_PROFILE=${VLLM_ASCEND_MODE4_STABILITY_PROFILE:-1}
if [[ ("${VLLM_ASCEND_ELASTIC_EXECUTION_MODE}" == "4" || "${VLLM_ASCEND_ELASTIC_EXECUTION_MODE}" == "5") && "${VLLM_ASCEND_MODE4_STABILITY_PROFILE}" == "1" ]]; then
    # For mode=4/5, the configured floor controls the maximum double-buffer
    # slot capacity, not full mode=1-style redundant expert residency.
    # Do not silently force floor=8; tests that need that behavior can set
    # VLLM_ASCEND_MODE4_STABILITY_FORCE_FLOOR explicitly.
    VLLM_ASCEND_MODE4_STABILITY_FORCE_FLOOR_WAS_SET=0
    if [[ -n "${VLLM_ASCEND_MODE4_STABILITY_FORCE_FLOOR+x}" ]]; then
        VLLM_ASCEND_MODE4_STABILITY_FORCE_FLOOR_WAS_SET=1
    fi
    # If the caller explicitly provides the configured floor, respect it.
    # Otherwise only override when a force floor is explicitly provided.
    if [[ "${VLLM_ASCEND_ELASTIC_MIN_COMPUTE_GROUP_SIZE_WAS_SET}" == "1" && "${VLLM_ASCEND_MODE4_STABILITY_FORCE_FLOOR_WAS_SET}" == "0" ]]; then
        echo "[mode4 stability] respect explicit elastic floor: ${VLLM_ASCEND_ELASTIC_MIN_COMPUTE_GROUP_SIZE}"
    elif [[ -n "${VLLM_ASCEND_MODE4_STABILITY_FORCE_FLOOR}" && "${VLLM_ASCEND_ELASTIC_MIN_COMPUTE_GROUP_SIZE}" != "${VLLM_ASCEND_MODE4_STABILITY_FORCE_FLOOR}" ]]; then
        echo "[mode4 stability] override elastic floor: ${VLLM_ASCEND_ELASTIC_MIN_COMPUTE_GROUP_SIZE} -> ${VLLM_ASCEND_MODE4_STABILITY_FORCE_FLOOR}"
        export VLLM_ASCEND_ELASTIC_MIN_COMPUTE_GROUP_SIZE=${VLLM_ASCEND_MODE4_STABILITY_FORCE_FLOOR}
    fi
    export VLLM_ASCEND_MODE4_RUNTIME_MIN_COMPUTE_GROUP_SIZE=${VLLM_ASCEND_MODE4_RUNTIME_MIN_COMPUTE_GROUP_SIZE:-1}
    export VLLM_ASCEND_MODE5_RUNTIME_MIN_COMPUTE_GROUP_SIZE=${VLLM_ASCEND_MODE5_RUNTIME_MIN_COMPUTE_GROUP_SIZE:-${VLLM_ASCEND_MODE4_RUNTIME_MIN_COMPUTE_GROUP_SIZE}}
    export VLLM_ASCEND_MODE5_REMOTE_EXPERT_FRACTION=${VLLM_ASCEND_MODE5_REMOTE_EXPERT_FRACTION:-0.75}
    # Derive the mode=5 remote/cpu split from the recorded pure-NPU vs pure-CPU
    # communication timings. The resulting expert share is
    # mean(cpu_ms / npu_ms) / (1 + mean(cpu_ms / npu_ms)), which balances the
    # two transfer paths by their measured effective bandwidth.
    export VLLM_ASCEND_MODE5_REMOTE_EXPERT_FRACTION_POLICY=${VLLM_ASCEND_MODE5_REMOTE_EXPERT_FRACTION_POLICY:-fixed}
    export VLLM_ASCEND_MODE5_REMOTE_COMM_MS_SERIES=${VLLM_ASCEND_MODE5_REMOTE_COMM_MS_SERIES:-2.494,3.031,5.075,5.458,8.973,9.809,17.944,19.030}
    export VLLM_ASCEND_MODE5_CPU_COMM_MS_SERIES=${VLLM_ASCEND_MODE5_CPU_COMM_MS_SERIES:-4.970,6.558,9.802,11.629,18.657,18.788,37.956,38.132}
export VLLM_ASCEND_MODE5_BALANCE_REMOTE_SOURCE_FANOUT=${VLLM_ASCEND_MODE5_BALANCE_REMOTE_SOURCE_FANOUT:-0}
export VLLM_ASCEND_MODE5_CPU_DP_METADATA_SYNC=${VLLM_ASCEND_MODE5_CPU_DP_METADATA_SYNC:-1}
export VLLM_ASCEND_MODE5_SINGLE_CONTROL_MESSAGE_REMOTE=${VLLM_ASCEND_MODE5_SINGLE_CONTROL_MESSAGE_REMOTE:-1}
    export VLLM_ASCEND_MODE4_ENABLE_GENERIC_HEADROOM=${VLLM_ASCEND_MODE4_ENABLE_GENERIC_HEADROOM:-0}
    export VLLM_ASCEND_MODE4_MOE_DISPATCH_HEADROOM_BYTES=${VLLM_ASCEND_MODE4_MOE_DISPATCH_HEADROOM_BYTES:-4294967296}
    export VLLM_ASCEND_MODE4_LOW_FLOOR_MC2_WORKSPACE_HEADROOM_BYTES=${VLLM_ASCEND_MODE4_LOW_FLOOR_MC2_WORKSPACE_HEADROOM_BYTES:-2147483648}
    # Mode=5 first priority is semantic correctness + stable 3-step execution.
    # Keep a more conservative low-floor MC2 workspace cushion for now; once
    # stage=8/4/2/1 all run stably with the intended CPU/NPU split, we can
    # shrink this budget back down with evidence.
    export VLLM_ASCEND_MODE5_LOW_FLOOR_MC2_WORKSPACE_HEADROOM_BYTES=${VLLM_ASCEND_MODE5_LOW_FLOOR_MC2_WORKSPACE_HEADROOM_BYTES:-6442450944}
    # Keep a slightly larger KV init cushion for mode=5 so the next rollout
    # step does not fall over on a small post-restore allocator fragment.
    export VLLM_ASCEND_KV_CACHE_INIT_HEADROOM_BYTES=${VLLM_ASCEND_KV_CACHE_INIT_HEADROOM_BYTES:-1073741824}
    export VLLM_ASCEND_MODE4_FORCE_POST_SHRINK_MOE_WARMUP=${VLLM_ASCEND_MODE4_FORCE_POST_SHRINK_MOE_WARMUP:-0}
    export VLLM_ASCEND_REPEAT_POST_SHRINK_MOE_DISPATCH_WARMUP=${VLLM_ASCEND_REPEAT_POST_SHRINK_MOE_DISPATCH_WARMUP:-0}
    # Keep the full-world communicator for remote-cache P2P, but do not keep
    # stale shrink-time DP/EP/MC2 groups across rollout steps. Reusing those
    # old compute groups can poison later mode=4 double-buffer dispatch state.
    export VLLM_ASCEND_MODE4_KEEP_STALE_DP_GROUP_CACHE=${VLLM_ASCEND_MODE4_KEEP_STALE_DP_GROUP_CACHE:-0}
    export VLLM_ASCEND_MODE4_KEEP_STALE_MC2_GROUP_CACHE=${VLLM_ASCEND_MODE4_KEEP_STALE_MC2_GROUP_CACHE:-0}
    export VLLM_ASCEND_MODE4_KEEP_STALE_EP_GROUP_CACHE=${VLLM_ASCEND_MODE4_KEEP_STALE_EP_GROUP_CACHE:-0}
    export VLLM_ASCEND_MODE5_KEEP_STALE_DP_GROUP_CACHE=${VLLM_ASCEND_MODE5_KEEP_STALE_DP_GROUP_CACHE:-0}
    export VLLM_ASCEND_MODE5_KEEP_STALE_MC2_GROUP_CACHE=${VLLM_ASCEND_MODE5_KEEP_STALE_MC2_GROUP_CACHE:-0}
    export VLLM_ASCEND_MODE5_KEEP_STALE_EP_GROUP_CACHE=${VLLM_ASCEND_MODE5_KEEP_STALE_EP_GROUP_CACHE:-0}
    export VLLM_ASCEND_MODE4_DROP_STALE_GROUP_CACHE_AFTER_SHRINK=${VLLM_ASCEND_MODE4_DROP_STALE_GROUP_CACHE_AFTER_SHRINK:-0}
    export VLLM_ASCEND_MODE4_BLOCK_PREFETCH_LAYERS=${VLLM_ASCEND_MODE4_BLOCK_PREFETCH_LAYERS:-1}
else
    if [[ "${VLLM_ASCEND_ELASTIC_EXECUTION_MODE}" == "3" ]]; then
        # Mode=3 keeps CPU-shadow + double-buffer state across shrink/restore.
        # Reserve a small KV re-init cushion so the next rollout step does not
        # fail on the final KV block after allocator/workspace fragmentation.
        export VLLM_ASCEND_KV_CACHE_INIT_HEADROOM_BYTES=${VLLM_ASCEND_KV_CACHE_INIT_HEADROOM_BYTES:-1073741824}
    else
        export VLLM_ASCEND_KV_CACHE_INIT_HEADROOM_BYTES=${VLLM_ASCEND_KV_CACHE_INIT_HEADROOM_BYTES:-0}
    fi
    export VLLM_ASCEND_MODE4_MOE_DISPATCH_HEADROOM_BYTES=${VLLM_ASCEND_MODE4_MOE_DISPATCH_HEADROOM_BYTES:-2147483648}
fi
echo "[moe pattern stats] enabled=${VLLM_MOE_PATTERN_STATS} dir=${VLLM_MOE_STATS_DIR} mode=${VLLM_ASCEND_ELASTIC_EXECUTION_MODE} floor=${VLLM_ASCEND_ELASTIC_MIN_COMPUTE_GROUP_SIZE} hybrid_slots=${VLLM_ASCEND_ELASTIC_HYBRID_RESIDENT_EXPERT_SLOTS} mode3_async_npu_prefetch=${VLLM_ASCEND_MODE3_ASYNC_NPU_PREFETCH} mode3_async_cpu_stage=${VLLM_ASCEND_MODE3_ASYNC_CPU_STAGE} mode3_async_cpu_pack=${VLLM_ASCEND_MODE3_ASYNC_CPU_PACK} mode3_direct_cpu_slot=${VLLM_ASCEND_MODE3_DIRECT_CPU_SLOT} mode3_bulk_npu_copy=${VLLM_ASCEND_MODE3_BULK_NPU_COPY} mode3_bulk_cpu_stage=${VLLM_ASCEND_MODE3_BULK_CPU_STAGE} mode3_bulk_cpu_direct=${VLLM_ASCEND_MODE3_BULK_CPU_DIRECT} mode3_layer_local_buffer=${VLLM_ASCEND_MODE3_LAYER_LOCAL_BUFFER} mode3_use_fused_experts_path=${VLLM_ASCEND_MODE3_USE_FUSED_EXPERTS_PATH} mode3_expert_token_nums_type=${VLLM_ASCEND_MODE3_EXPERT_TOKEN_NUMS_TYPE} mode3_active_rows_sync=${VLLM_ASCEND_MODE3_ACTIVE_ROWS_SYNC} mode3_device_ready_wait=${VLLM_ASCEND_MODE3_DEVICE_READY_WAIT} mode3_post_shrink_warmup=${VLLM_ASCEND_MODE3_POST_SHRINK_MOE_WARMUP} mode3_post_shrink_warmup_tokens=${VLLM_ASCEND_MODE3_POST_SHRINK_MOE_WARMUP_TOKENS} mode3_low_floor_mc2_workspace=${VLLM_ASCEND_MODE3_LOW_FLOOR_MC2_WORKSPACE_HEADROOM_BYTES} mode3_transfer_log=${VLLM_ASCEND_MODE3_TRANSFER_LOG} mode3_transfer_plan_log=${VLLM_ASCEND_MODE3_TRANSFER_PLAN_LOG} mode3_transfer_plan_first_n=${VLLM_ASCEND_MODE3_TRANSFER_PLAN_FIRST_N} mode3_timing_log=${VLLM_ASCEND_MODE3_TIMING_LOG} mode3_timing_sync=${VLLM_ASCEND_MODE3_TIMING_SYNC} mode3_timing_every=${VLLM_ASCEND_MODE3_TIMING_EVERY} mode3_timing_first_n=${VLLM_ASCEND_MODE3_TIMING_FIRST_N} mode3_timing_layers=${VLLM_ASCEND_MODE3_TIMING_LAYERS} dummy_waste_timing=${VLLM_ASCEND_DUMMY_WASTE_TIMING} dummy_waste_sync=${VLLM_ASCEND_DUMMY_WASTE_TIMING_SYNC} dummy_waste_profile=${VLLM_ASCEND_DUMMY_WASTE_TIMING_PROFILE} dummy_waste_markers=${VLLM_ASCEND_DUMMY_WASTE_PROFILE_MARKERS} elastic_util_log=${VLLM_ASCEND_ELASTIC_UTIL_LOG} elastic_util_bucket_steps=${VLLM_ASCEND_ELASTIC_UTIL_BUCKET_STEPS} custom_mode1_kv_headroom=${VLLM_ASCEND_CUSTOM_MODE1_KV_MATERIALIZE_HEADROOM_BYTES} kv_cache_init_headroom=${VLLM_ASCEND_KV_CACHE_INIT_HEADROOM_BYTES} mode4_stability=${VLLM_ASCEND_MODE4_STABILITY_PROFILE} mode4_force_floor=${VLLM_ASCEND_MODE4_STABILITY_FORCE_FLOOR:-} mode4_runtime_floor=${VLLM_ASCEND_MODE4_RUNTIME_MIN_COMPUTE_GROUP_SIZE:-} mode5_runtime_floor=${VLLM_ASCEND_MODE5_RUNTIME_MIN_COMPUTE_GROUP_SIZE:-} mode5_runtime_strategy=$(if [[ "${VLLM_ASCEND_MODE5_USE_LEGACY_CPU_SHADOW_RUNTIME:-0}" =~ ^(1|true|TRUE|yes|YES|on|ON)$ ]]; then echo legacy_cpu_shadow; else echo dual_source; fi) mode5_remote_fraction=${VLLM_ASCEND_MODE5_REMOTE_EXPERT_FRACTION:-} mode5_fraction_policy=${VLLM_ASCEND_MODE5_REMOTE_EXPERT_FRACTION_POLICY:-} mode5_balance_remote_source_fanout=${VLLM_ASCEND_MODE5_BALANCE_REMOTE_SOURCE_FANOUT:-} mode5_cpu_dp_metadata_sync=${VLLM_ASCEND_MODE5_CPU_DP_METADATA_SYNC:-} mode5_single_control_message_remote=${VLLM_ASCEND_MODE5_SINGLE_CONTROL_MESSAGE_REMOTE:-} mode4_generic_headroom=${VLLM_ASCEND_MODE4_ENABLE_GENERIC_HEADROOM:-0} mode4_moe_dispatch_headroom=${VLLM_ASCEND_MODE4_MOE_DISPATCH_HEADROOM_BYTES} mode4_low_floor_mc2_headroom=${VLLM_ASCEND_MODE4_LOW_FLOOR_MC2_WORKSPACE_HEADROOM_BYTES:-} mode5_low_floor_mc2_headroom=${VLLM_ASCEND_MODE5_LOW_FLOOR_MC2_WORKSPACE_HEADROOM_BYTES:-} mode4_block_prefetch_layers=${VLLM_ASCEND_MODE4_BLOCK_PREFETCH_LAYERS:-1}"
#模拟样本缩短规则
# export VERL_ELASTIC_TAIL_VALIDATE_LEVEL_TOKENS=4,8,12,16,20
export VERL_ELASTIC_TAIL_VALIDATE_LEVEL_TOKENS=${VERL_ELASTIC_TAIL_VALIDATE_LEVEL_TOKENS-256,512,640,768,896}
echo "[elastic tail validate] VERL_ELASTIC_TAIL_VALIDATE_LEVEL_TOKENS=${VERL_ELASTIC_TAIL_VALIDATE_LEVEL_TOKENS:-}"

if [ "${DRAFT_PROFILE_MODE}" = "profile_only" ]; then
    # 只关注 draft train 的耗时拆分，不进入整套 RL 训练
    export VLLM_ASCEND_DRAFT_PROFILE_ONLY=1
    export VLLM_ASCEND_DRAFT_PROFILE_ONLY_WARMUP_STEPS=2
    export VLLM_ASCEND_DRAFT_PROFILE_ONLY_STEPS=10
    export VLLM_ASCEND_DRAFT_NPU_PROFILE_STEPS=10
    export VLLM_ASCEND_DRAFT_STARTUP_WARMUP_STEPS=0
    export VLLM_ASCEND_DRAFT_WARMUP_STEPS=0
fi

#超时配置
export ACL_MDL_STREAM_SYNC_TIMEOUT=-1
export ACL_MDL_EVENT_SYNC_TIMEOUT=-1

HOME=$(pwd)
MODEL_PATH=${MODEL_PATH:-"/home/data/Qwen3-30B-A3B"}
CONFIG_DIR=${CONFIG_DIR:-"${HOME}/verl/trainer/config"}
DISTCP_PATH=${DISTCP_PATH:-"/home/data/Qwen3-30B-A3B_megatron"}
TRAIN_FILE=${TRAIN_FILE:-"/workspace/data/deepscaler/train.parquet"}
TEST_FILE=${TEST_FILE:-"/workspace/data/deepscaler/test.parquet"}
RECORD_DIR=${RECORD_DIR:-"${HOME}/record"}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-32}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-1024}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-16384}
ROLLOUT_N=${ROLLOUT_N:-16}
ROLLOUT_MAX_NUM_SEQS=${ROLLOUT_MAX_NUM_SEQS:-32}
ROLLOUT_MAX_MODEL_LEN=${ROLLOUT_MAX_MODEL_LEN:-$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))}
ROLLOUT_MAX_NUM_BATCHED_TOKENS=${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-${ROLLOUT_MAX_MODEL_LEN}}
ROLLOUT_GPU_MEMORY_UTILIZATION=${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.85}
ROLLOUT_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=${ROLLOUT_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-4}
REF_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=${REF_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-8}
TRAINER_TOTAL_EPOCHS=${TRAINER_TOTAL_EPOCHS:-1}
if (( ROLLOUT_MAX_NUM_BATCHED_TOKENS < ROLLOUT_MAX_MODEL_LEN )); then
    echo "[rollout config] bump ROLLOUT_MAX_NUM_BATCHED_TOKENS=${ROLLOUT_MAX_NUM_BATCHED_TOKENS} to ROLLOUT_MAX_MODEL_LEN=${ROLLOUT_MAX_MODEL_LEN}"
    ROLLOUT_MAX_NUM_BATCHED_TOKENS=${ROLLOUT_MAX_MODEL_LEN}
fi
mkdir -p "${RECORD_DIR}"

time=$(date +%Y%m%d%H%M%S)
elastic_suffix=""
if [ "${VLLM_ASCEND_ELASTIC_EXECUTION_MODE}" != "0" ]; then
    elastic_suffix="_elastic"
fi
logfile="${HOME}/wjeagerqwen30b-a3b-with_draft_${DRAFT_PROFILE_MODE}_${time}${elastic_suffix}.txt"

export VERL_SIDECAR_ENABLE=${VERL_SIDECAR_ENABLE:-0}
export VERL_SIDECAR_MODEL_PATH=${VERL_SIDECAR_MODEL_PATH:-"/home/data/Qwen2.5-1.5B-Instruct"}
export VERL_SIDECAR_PROMPTS_FILE=${VERL_SIDECAR_PROMPTS_FILE:-"/home/qiuzy/verl_dev/data/gsm8k"}
export VERL_SIDECAR_DATA_SPLIT=${VERL_SIDECAR_DATA_SPLIT:-train}
export VERL_SIDECAR_USE_SHORT_DATA=${VERL_SIDECAR_USE_SHORT_DATA:-0}
export VERL_SIDECAR_GPU_MEMORY_UTILIZATION=${VERL_SIDECAR_GPU_MEMORY_UTILIZATION:-0.90}
export VERL_SIDECAR_MAX_MODEL_LEN=${VERL_SIDECAR_MAX_MODEL_LEN:-2048}
export VERL_SIDECAR_MAX_NUM_SEQS=${VERL_SIDECAR_MAX_NUM_SEQS:-128}
export VERL_SIDECAR_MAX_NUM_BATCHED_TOKENS=${VERL_SIDECAR_MAX_NUM_BATCHED_TOKENS:-65536}
export VERL_SIDECAR_MAX_PROMPTS_PER_DEVICE=${VERL_SIDECAR_MAX_PROMPTS_PER_DEVICE:-128}
export VERL_SIDECAR_MAX_PROMPTS=${VERL_SIDECAR_MAX_PROMPTS:-1024}
export VERL_SIDECAR_MAX_TOKENS=${VERL_SIDECAR_MAX_TOKENS:-1024}
export VERL_SIDECAR_REPEAT_UNTIL_KILLED=${VERL_SIDECAR_REPEAT_UNTIL_KILLED:-1}
export VERL_SIDECAR_MAX_ITERATIONS=${VERL_SIDECAR_MAX_ITERATIONS:-0}
export VERL_SIDECAR_GENERATE_CHUNK_SIZE=${VERL_SIDECAR_GENERATE_CHUNK_SIZE:-32}
export VERL_SIDECAR_STREAM_CHECKPOINT=${VERL_SIDECAR_STREAM_CHECKPOINT:-1}
export VERL_SIDECAR_PARTIAL_SYNC_EVERY_STEPS=${VERL_SIDECAR_PARTIAL_SYNC_EVERY_STEPS:-0}
export VERL_SIDECAR_GRACEFUL_KILL_SECONDS=${VERL_SIDECAR_GRACEFUL_KILL_SECONDS:-3}
export VERL_SIDECAR_STATE_DIR=${VERL_SIDECAR_STATE_DIR:-"sidecar_runs/state/qwen25_15b_gsm8k_train"}
export VERL_SIDECAR_PARALLEL_MODE=${VERL_SIDECAR_PARALLEL_MODE:-dp}
export VERL_SIDECAR_LOG_DIR=${VERL_SIDECAR_LOG_DIR:-"sidecar_runs/${time}"}
sidecar_monitor_pid=""
cleanup_sidecar_monitor() {
    if [[ -n "${sidecar_monitor_pid}" ]] && kill -0 "${sidecar_monitor_pid}" 2>/dev/null; then
        kill "${sidecar_monitor_pid}" 2>/dev/null || true
        wait "${sidecar_monitor_pid}" 2>/dev/null || true
    fi
}

if [[ "${VERL_SIDECAR_ENABLE}" == "1" ]]; then
    mkdir -p "${VERL_SIDECAR_LOG_DIR}"
    : > "${logfile}"
    export VERL_SIDECAR_TRAIN_LOG="${logfile}"
    export VERL_SIDECAR_LEASE_LOG=${VERL_SIDECAR_LEASE_LOG:-"${VERL_SIDECAR_LOG_DIR}/lease.log"}
    export VERL_SIDECAR_LOG_FILE=${VERL_SIDECAR_LOG_FILE:-"${VERL_SIDECAR_LOG_DIR}/infer.log"}
    export VERL_SIDECAR_OUTPUT_FILE=${VERL_SIDECAR_OUTPUT_FILE:-"${VERL_SIDECAR_LOG_DIR}/outputs.jsonl"}
    sidecar_monitor_log=${VERL_SIDECAR_MONITOR_LOG:-"${VERL_SIDECAR_LOG_DIR}/monitor.log"}
    echo "[elastic sidecar] enabled=1 train_log=${VERL_SIDECAR_TRAIN_LOG} log_dir=${VERL_SIDECAR_LOG_DIR} lease_log=${VERL_SIDECAR_LEASE_LOG} sidecar_log=${VERL_SIDECAR_LOG_FILE} sidecar_output=${VERL_SIDECAR_OUTPUT_FILE} monitor_log=${sidecar_monitor_log} devices=${VERL_SIDECAR_NPU_DEVICES:-auto_from_inactive_ranks} parallel_mode=${VERL_SIDECAR_PARALLEL_MODE} model=${VERL_SIDECAR_MODEL_PATH} prompts=${VERL_SIDECAR_PROMPTS_FILE} data_split=${VERL_SIDECAR_DATA_SPLIT} use_short_data=${VERL_SIDECAR_USE_SHORT_DATA} state_dir=${VERL_SIDECAR_STATE_DIR} max_prompts=${VERL_SIDECAR_MAX_PROMPTS} max_prompts_per_device=${VERL_SIDECAR_MAX_PROMPTS_PER_DEVICE} generate_chunk_size=${VERL_SIDECAR_GENERATE_CHUNK_SIZE} stream_checkpoint=${VERL_SIDECAR_STREAM_CHECKPOINT} partial_sync_every_steps=${VERL_SIDECAR_PARTIAL_SYNC_EVERY_STEPS} graceful_kill_seconds=${VERL_SIDECAR_GRACEFUL_KILL_SECONDS} max_tokens=${VERL_SIDECAR_MAX_TOKENS} max_num_seqs=${VERL_SIDECAR_MAX_NUM_SEQS} max_num_batched_tokens=${VERL_SIDECAR_MAX_NUM_BATCHED_TOKENS} max_model_len=${VERL_SIDECAR_MAX_MODEL_LEN} gpu_memory_utilization=${VERL_SIDECAR_GPU_MEMORY_UTILIZATION} repeat_until_killed=${VERL_SIDECAR_REPEAT_UNTIL_KILLED} max_iterations=${VERL_SIDECAR_MAX_ITERATIONS}"
    internal/watch_elastic_shrink_and_run_sidecar.sh "${logfile}" >> "${sidecar_monitor_log}" 2>&1 &
    sidecar_monitor_pid=$!
    trap cleanup_sidecar_monitor EXIT
else
    echo "[elastic sidecar] enabled=0"
fi

{
    echo "[run] start_time=$(date '+%Y-%m-%dT%H:%M:%S%z') logfile=${logfile}"
    echo "[full redundancy experiment] enabled=${VLLM_ASCEND_FULL_REDUNDANCY_EXPERIMENT_LOG} sidecar=${VERL_SIDECAR_ENABLE} mode=${VLLM_ASCEND_ELASTIC_EXECUTION_MODE} floor=${VLLM_ASCEND_ELASTIC_MIN_COMPUTE_GROUP_SIZE} mode4_runtime_floor=${VLLM_ASCEND_MODE4_RUNTIME_MIN_COMPUTE_GROUP_SIZE:-} mode4_keep_stale_dp=${VLLM_ASCEND_MODE4_KEEP_STALE_DP_GROUP_CACHE:-} mode4_keep_stale_ep=${VLLM_ASCEND_MODE4_KEEP_STALE_EP_GROUP_CACHE:-} mode4_keep_stale_mc2=${VLLM_ASCEND_MODE4_KEEP_STALE_MC2_GROUP_CACHE:-} gloo_socket_ifname=${GLOO_SOCKET_IFNAME:-} hccl_if_base_port=${HCCL_IF_BASE_PORT:-} dp_size=${VLLM_DP_SIZE} draft_profile_mode=${DRAFT_PROFILE_MODE} model_path=${MODEL_PATH} train_file=${TRAIN_FILE} test_file=${TEST_FILE} train_batch_size=${TRAIN_BATCH_SIZE} max_num_seqs=${ROLLOUT_MAX_NUM_SEQS} max_num_batched_tokens=${ROLLOUT_MAX_NUM_BATCHED_TOKENS} max_prompt_length=${MAX_PROMPT_LENGTH} max_response_length=${MAX_RESPONSE_LENGTH} rollout_n=${ROLLOUT_N} gpu_memory_utilization=${ROLLOUT_GPU_MEMORY_UTILIZATION} total_epochs=${TRAINER_TOTAL_EPOCHS} custom_mode1_kv_headroom=${VLLM_ASCEND_CUSTOM_MODE1_KV_MATERIALIZE_HEADROOM_BYTES} kv_cache_init_headroom=${VLLM_ASCEND_KV_CACHE_INIT_HEADROOM_BYTES} mode1_native_kv_cap=${VLLM_ASCEND_MODE1_PARITY_MAX_KV_TOKENS} mode1_keep_fullworld_ep_cache=${VLLM_ASCEND_MODE1_PARITY_KEEP_FULLWORLD_EP_CACHE} mode1_keep_mc2_cache=${VLLM_ASCEND_MODE1_PARITY_KEEP_MC2_GROUP_CACHE} mode1_drop_stale_after_shrink=${VLLM_ASCEND_MODE1_PARITY_DROP_STALE_CACHE_AFTER_SHRINK} mode1_single_live_mc2=${VLLM_ASCEND_MODE1_PARITY_SINGLE_LIVE_MC2_GROUP} disable_elastic_mc2_cache=${VLLM_ASCEND_DISABLE_ELASTIC_MC2_GROUP_CACHE} mode1_post_restore_alltoall_warmup=${VLLM_ASCEND_MODE1_PARITY_POST_RESTORE_ALLTOALL_WARMUP} mode1_batch_direct_npu_import=${VLLM_ASCEND_MODE1_BATCH_DIRECT_NPU_IMPORT} mode1_allow_scalar_direct_npu_import=${VLLM_ASCEND_MODE1_ALLOW_SCALAR_DIRECT_NPU_IMPORT} mode1_allow_batch_index_select_export=${VLLM_ASCEND_MODE1_ALLOW_BATCH_INDEX_SELECT_EXPORT} mode1_direct_npu_import_batch_experts=${VLLM_ASCEND_MODE1_DIRECT_NPU_IMPORT_BATCH_EXPERTS} post_shrink_release_empty_cache=${VLLM_ASCEND_POST_SHRINK_STAGING_RELEASE_EMPTY_CACHE} post_shrink_release_sync=${VLLM_ASCEND_POST_SHRINK_STAGING_RELEASE_SYNC} mode1_sync_after_mc2_destroy=${VLLM_ASCEND_MODE1_PARITY_SYNC_AFTER_MC2_GROUP_DESTROY} mode1_gc_after_mc2_destroy=${VLLM_ASCEND_MODE1_PARITY_GC_AFTER_MC2_GROUP_DESTROY} mode1_release_dp_warmup_cache=${VLLM_ASCEND_MODE1_PARITY_RELEASE_DP_WARMUP_CACHE} mode1_release_warmup_cache=${VLLM_ASCEND_MODE1_PARITY_RELEASE_WARMUP_CACHE} mode1_keep_floor4_group_cache=${VLLM_ASCEND_MODE1_PARITY_KEEP_FLOOR4_GROUP_CACHE} mode1_keep_floor4_group_kinds=${VLLM_ASCEND_MODE1_PARITY_KEEP_FLOOR4_GROUP_KINDS} mode1_drop_old_floor_before_rebuild=${VLLM_ASCEND_MODE1_PARITY_DROP_OLD_FLOOR_BEFORE_REBUILD} mode1_release_live_old_floor_on_rebuild=${VLLM_ASCEND_MODE1_PARITY_RELEASE_LIVE_OLD_FLOOR_ON_REBUILD} mode1_defer_group_destroy=${VLLM_ASCEND_MODE1_PARITY_DEFER_GROUP_DESTROY} mode1_defer_destroy_floor_sizes=${VLLM_ASCEND_MODE1_PARITY_DEFER_DESTROY_FLOOR_GROUP_SIZES} mode1_sync_before_device_pg_retire=${VLLM_ASCEND_MODE1_PARITY_SYNC_BEFORE_DEVICE_PG_RETIRE} mode1_destroy_device_pg_on_retire=${VLLM_ASCEND_MODE1_PARITY_DESTROY_DEVICE_PG_ON_RETIRE} mode1_enable_post_shrink_dp_warmup=${VLLM_ASCEND_MODE1_PARITY_ENABLE_POST_SHRINK_DP_WARMUP} mode1_cpu_dp_metadata_sync=${VLLM_ASCEND_MODE1_CPU_DP_METADATA_SYNC}"
} | tee -a "${logfile}"

set -x

set +e
python3 -m verl.trainer.main_ppo --config-path="${CONFIG_DIR}" \
    --config-name='ppo_megatron_trainer.yaml'\
    algorithm.adv_estimator=grpo \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${TEST_FILE}" \
    data.train_batch_size="${TRAIN_BATCH_SIZE}" \
    data.max_prompt_length="${MAX_PROMPT_LENGTH}" \
    data.max_response_length="${MAX_RESPONSE_LENGTH}" \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.shuffle=False \
    +data.dataset_fraction=0.003\
    custom_reward_function.path=deepscaler.py \
    custom_reward_function.name=compute_score  \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.clip_grad=10000 \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.megatron.sequence_parallel=True \
    actor_rollout_ref.actor.megatron.expert_model_parallel_size=4 \
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=4 \
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=4 \
    actor_rollout_ref.actor.megatron.expert_tensor_parallel_size=1 \
    actor_rollout_ref.actor.megatron.param_offload=True \
    actor_rollout_ref.actor.megatron.grad_offload=True \
    actor_rollout_ref.actor.megatron.optimizer_offload=False \
    actor_rollout_ref.actor.megatron.use_dist_checkpointing=True \
    actor_rollout_ref.actor.megatron.dist_checkpointing_path="${DISTCP_PATH}" \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.load_weight=True \
    actor_rollout_ref.actor.megatron.override_transformer_config.recompute_granularity=full \
    actor_rollout_ref.actor.megatron.override_transformer_config.recompute_method=block \
    actor_rollout_ref.actor.megatron.override_transformer_config.recompute_num_layers=1 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="${ROLLOUT_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU}" \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization="${ROLLOUT_GPU_MEMORY_UTILIZATION}" \
    actor_rollout_ref.rollout.max_num_batched_tokens="${ROLLOUT_MAX_NUM_BATCHED_TOKENS}" \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.max_num_seqs="${ROLLOUT_MAX_NUM_SEQS}" \
    actor_rollout_ref.rollout.n="${ROLLOUT_N}" \
    actor_rollout_ref.rollout.temperature=0.9 \
    actor_rollout_ref.rollout.top_k=50 \
    actor_rollout_ref.rollout.top_p=0.9 \
    actor_rollout_ref.rollout.ignore_eos=False \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="${REF_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU}" \
    actor_rollout_ref.ref.megatron.param_offload=True \
    actor_rollout_ref.ref.load_weight=True \
    actor_rollout_ref.ref.megatron.use_dist_checkpointing=True \
    actor_rollout_ref.ref.megatron.dist_checkpointing_path="${DISTCP_PATH}" \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.balance_batch=False \
    trainer.device=npu \
    trainer.val_before_train=False \
    trainer.critic_warmup=0 \
    trainer.logger=['console','tensorboard'] \
    trainer.project_name='verl_grpo_example' \
    trainer.experiment_name='qwen3_30_verl_mindspeedllm_vllm' \
    trainer.n_gpus_per_node=16 \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=-1 \
    trainer.total_epochs="${TRAINER_TOTAL_EPOCHS}" \
    +trainer.rollout_data_dir="${RECORD_DIR}" \
    +trainer.rollout_length_dir="${RECORD_DIR}" \
    +actor_rollout_ref.actor.megatron.override_transformer_config.use_flash_attn=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.pipeline_num_transformer_layers=[[11],[13],[13],[11]] \
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_token_dispatcher_type='alltoall' \
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_alltoall_overlap_comm=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.use_fused_rotary_pos_emb=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.use_fused_swiglu=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.seq_length=2048 \
    +actor_rollout_ref.actor.megatron.override_transformer_config.num_layers_in_first_pipeline_stage=11 \
    +actor_rollout_ref.actor.megatron.override_transformer_config.num_layers_in_last_pipeline_stage=11 \
    +actor_rollout_ref.actor.megatron.override_transformer_config.swap_optimizer=True \
    "$@" >> "${logfile}" 2>&1
run_exit_code=$?
set -e

echo "[run] end_time=$(date '+%Y-%m-%dT%H:%M:%S%z') exit_code=${run_exit_code}" | tee -a "${logfile}"
exit "${run_exit_code}"
