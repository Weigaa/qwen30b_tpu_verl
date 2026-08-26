# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import time
from collections.abc import Callable, Mapping
from copy import copy
from typing import Any, cast

import torch
import torch.nn as nn
from typing_extensions import TypeVar

import vllm.envs as envs
from vllm.config import ParallelConfig, VllmConfig
from vllm.distributed import stateless_destroy_torch_distributed_process_group
from vllm.distributed.parallel_state import get_dp_group, get_world_group
from vllm.engine.arg_utils import EngineArgs
from vllm.inputs import PromptType
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.multimodal import MULTIMODAL_REGISTRY, MultiModalRegistry
from vllm.outputs import PoolingRequestOutput, RequestOutput
from vllm.plugins.io_processors import get_io_processor
from vllm.pooling_params import PoolingParams
from vllm.sampling_params import SamplingParams
from vllm.tasks import SupportedTask
from vllm.tokenizers import TokenizerLike, cached_tokenizer_from_config
from vllm.tracing import init_tracer
from vllm.usage.usage_lib import UsageContext
from vllm.v1.engine import EngineCoreRequest
from vllm.v1.engine.core_client import EngineCoreClient
from vllm.v1.engine.input_processor import InputProcessor
from vllm.v1.engine.output_processor import OutputProcessor
from vllm.v1.engine.parallel_sampling import ParentRequest
from vllm.v1.executor import Executor
from vllm.v1.metrics.loggers import StatLoggerFactory, StatLoggerManager
from vllm.v1.metrics.reader import Metric, get_metrics_snapshot
from vllm.v1.metrics.stats import IterationStats
from vllm.v1.utils import record_function_or_nullcontext
from vllm.v1.worker.worker_base import WorkerBase
from vllm.utils.moe_stats import moe_stats
from vllm_ascend import envs as envs_ascend
from vllm_ascend.shrink_aware import (
    decide_staged_shrink,
    parse_rank_list,
    parse_rank_topology,
    parse_stage_survivor_ranks,
    plan_survivor_ranks,
)

logger = init_logger(__name__)

_R = TypeVar("_R", default=Any)


class LLMEngine:
    """Legacy LLMEngine for backwards compatibility."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        executor_class: type[Executor],
        log_stats: bool,
        aggregate_engine_logging: bool = False,
        usage_context: UsageContext = UsageContext.ENGINE_CONTEXT,
        stat_loggers: list[StatLoggerFactory] | None = None,
        mm_registry: MultiModalRegistry = MULTIMODAL_REGISTRY,
        use_cached_outputs: bool = False,
        multiprocess_mode: bool = False,
    ) -> None:
        self.vllm_config = vllm_config
        self.observability_config = vllm_config.observability_config
        self.model_config = vllm_config.model_config
        self.cache_config = vllm_config.cache_config

        self.log_stats = log_stats

        parallel_config = vllm_config.parallel_config
        executor_backend = parallel_config.distributed_executor_backend

        self.external_launcher_dp = (
            parallel_config.data_parallel_size > 1
            and executor_backend == "external_launcher"
        )
        # important: init dp group before init the engine_core
        # In the decoupled engine case this is handled in EngineCoreProc.
        if (
            not multiprocess_mode
            and parallel_config.data_parallel_size > 1
            and not self.external_launcher_dp
        ):
            self.dp_group = parallel_config.stateless_init_dp_group()
        else:
            self.dp_group = None
        self.should_execute_dummy_batch = False

        if self.model_config.skip_tokenizer_init:
            tokenizer = None
        else:
            tokenizer = cached_tokenizer_from_config(self.model_config)

        self.input_processor = InputProcessor(self.vllm_config, tokenizer)
        self.io_processor = get_io_processor(
            self.vllm_config,
            self.model_config.io_processor_plugin,
        )

        # OutputProcessor (convert EngineCoreOutputs --> RequestOutput).
        self.output_processor = OutputProcessor(
            self.tokenizer,
            log_stats=self.log_stats,
            stream_interval=self.vllm_config.scheduler_config.stream_interval,
        )
        endpoint = self.observability_config.otlp_traces_endpoint
        if endpoint is not None:
            tracer = init_tracer("vllm.llm_engine", endpoint)
            self.output_processor.tracer = tracer

        # EngineCore (gets EngineCoreRequests and gives EngineCoreOutputs)
        self.engine_core = EngineCoreClient.make_client(
            multiprocess_mode=multiprocess_mode,
            asyncio_mode=False,
            vllm_config=vllm_config,
            executor_class=executor_class,
            log_stats=self.log_stats,
        )

        self.logger_manager: StatLoggerManager | None = None
        if self.log_stats:
            self.logger_manager = StatLoggerManager(
                vllm_config=vllm_config,
                custom_stat_loggers=stat_loggers,
                enable_default_loggers=log_stats,
                aggregate_engine_logging=aggregate_engine_logging,
            )
            self.logger_manager.log_engine_initialized()

        if not multiprocess_mode:
            # for v0 compatibility
            self.model_executor = self.engine_core.engine_core.model_executor  # type: ignore

        if self.external_launcher_dp:
            # If we use DP in external launcher mode, we reuse the
            # existing DP group used for data communication.
            self.dp_group = get_dp_group().cpu_group
            self.elastic_original_dp_world_size = torch.distributed.get_world_size(group=self.dp_group)
            rank_tensor = torch.tensor(
                [torch.distributed.get_rank()], dtype=torch.int64, device="cpu")
            gathered_ranks = [
                torch.empty_like(rank_tensor)
                for _ in range(self.elastic_original_dp_world_size)
            ]
            torch.distributed.all_gather(
                gathered_ranks, rank_tensor, group=self.dp_group)
            self.elastic_original_dp_global_ranks = tuple(
                int(item.item()) for item in gathered_ranks)
        else:
            self.elastic_original_dp_world_size = None
            self.elastic_original_dp_global_ranks = None

        self.elastic_ep_no_dummy = (
            envs_ascend.VLLM_ASCEND_ENABLE_ELASTIC_PARALLEL_SHRINK
            and self.external_launcher_dp
            and self.model_config.enforce_eager
            and self._is_followup_elastic_shrink_enabled()
        )
        self.elastic_ep_scaled_once = False
        self.elastic_ep_active_ranks: tuple[int, ...] | None = None
        self.elastic_ep_runtime_active_ranks: tuple[int, ...] | None = None
        self._shrink_aware_role_plan = None
        self._shrink_aware_role_plan_signature = None
        self._logged_followup_shrink_skip: tuple[int, ...] | None = None
        self._logged_mc2_delay = False
        self.backend_dp_dummy_flow = (
            self.external_launcher_dp
            and not self.elastic_ep_no_dummy
            and self.vllm_config.scheduler_config.async_scheduling
            and self.vllm_config.model_config.is_moe
        )

        # Don't keep the dummy data in memory
        self.reset_mm_cache()

    @classmethod
    def from_vllm_config(
        cls,
        vllm_config: VllmConfig,
        usage_context: UsageContext = UsageContext.ENGINE_CONTEXT,
        stat_loggers: list[StatLoggerFactory] | None = None,
        disable_log_stats: bool = False,
    ) -> "LLMEngine":
        return cls(
            vllm_config=vllm_config,
            executor_class=Executor.get_class(vllm_config),
            log_stats=(not disable_log_stats),
            usage_context=usage_context,
            stat_loggers=stat_loggers,
            multiprocess_mode=envs.VLLM_ENABLE_V1_MULTIPROCESSING,
        )

    @classmethod
    def from_engine_args(
        cls,
        engine_args: EngineArgs,
        usage_context: UsageContext = UsageContext.ENGINE_CONTEXT,
        stat_loggers: list[StatLoggerFactory] | None = None,
        enable_multiprocessing: bool = False,
    ) -> "LLMEngine":
        """Creates an LLM engine from the engine arguments."""

        # Create the engine configs.
        vllm_config = engine_args.create_engine_config(usage_context)
        executor_class = Executor.get_class(vllm_config)

        if envs.VLLM_ENABLE_V1_MULTIPROCESSING:
            logger.debug("Enabling multiprocessing for LLMEngine.")
            enable_multiprocessing = True

        # Create the LLMEngine.
        return cls(
            vllm_config=vllm_config,
            executor_class=executor_class,
            log_stats=not engine_args.disable_log_stats,
            usage_context=usage_context,
            stat_loggers=stat_loggers,
            multiprocess_mode=enable_multiprocessing,
        )

    def get_num_unfinished_requests(self) -> int:
        return self.output_processor.get_num_unfinished_requests()

    def has_unfinished_requests(self) -> bool:
        has_unfinished = self.output_processor.has_unfinished_requests()
        if self.dp_group is None:
            return has_unfinished or self.engine_core.dp_engines_running()
        if self.backend_dp_dummy_flow:
            # In async mode, keep the front-end lightweight and let the
            # back-end engine step own the global running-state / dummy
            # transition. Otherwise fast ranks can jump straight into
            # dummy-run collectives while slow ranks are still advancing
            # the previous real batch through the async batch queue.
            return has_unfinished or self.engine_core.dp_engines_running()
        return self.has_unfinished_requests_dp(has_unfinished)

    def _get_active_dp_global_ranks(self, has_unfinished: bool) -> list[int]:
        assert self.dp_group is not None
        local_global_rank = torch.distributed.get_rank()
        local_marker = local_global_rank if has_unfinished else -1
        local_tensor = torch.tensor([local_marker], dtype=torch.int64, device="cpu")
        dp_world_size = torch.distributed.get_world_size(group=self.dp_group)
        gathered = [torch.empty_like(local_tensor) for _ in range(dp_world_size)]
        torch.distributed.all_gather(gathered, local_tensor, group=self.dp_group)
        return [int(t.item()) for t in gathered if int(t.item()) >= 0]

    def _original_dp_global_ranks(self, dp_world_size: int) -> list[int]:
        ranks = self.elastic_original_dp_global_ranks
        if ranks:
            return [int(rank) for rank in ranks]
        return list(range(dp_world_size))

    def _dp_global_to_local_ranks(
        self, global_ranks: list[int], full_world_size: int
    ) -> list[int]:
        original = self._original_dp_global_ranks(full_world_size)
        local_by_global = {
            int(global_rank): local_rank
            for local_rank, global_rank in enumerate(original)
        }
        unknown = sorted(set(map(int, global_ranks)) - set(local_by_global))
        if unknown:
            raise ValueError(
                "AdaFloor active ranks are outside the original external-DP "
                f"group: active={global_ranks} original={original} "
                f"unknown={unknown}")
        return sorted(local_by_global[int(rank)] for rank in global_ranks)

    def _dp_local_to_global_ranks(
        self, local_ranks: list[int], full_world_size: int
    ) -> list[int]:
        original = self._original_dp_global_ranks(full_world_size)
        if any(rank < 0 or rank >= len(original) for rank in local_ranks):
            raise ValueError(
                f"AdaFloor local ranks {local_ranks} are invalid for "
                f"external-DP group {original}")
        return [original[int(rank)] for rank in local_ranks]

    def _get_shrink_aware_role_plan(self, dp_world_size: int):
        if not envs_ascend.VLLM_ASCEND_SHRINK_AWARE_ENABLE:
            return None
        full_world_size = self.elastic_original_dp_world_size or dp_world_size
        raw_stage_ranks = envs_ascend.VLLM_ASCEND_SHRINK_AWARE_STAGE_RANKS
        signature = (
            int(full_world_size),
            str(envs_ascend.VLLM_ASCEND_SHRINK_AWARE_STAGES),
            str(envs_ascend.VLLM_ASCEND_SHRINK_AWARE_SURVIVOR_POLICY),
            str(envs_ascend.VLLM_ASCEND_SHRINK_AWARE_PACKAGE_TOPOLOGY),
            str(envs_ascend.VLLM_ASCEND_SHRINK_AWARE_INTERMEDIATE_RANKS),
            str(envs_ascend.VLLM_ASCEND_SHRINK_AWARE_FINAL_RANKS),
            str(raw_stage_ranks),
        )
        if (
            self._shrink_aware_role_plan is not None
            and self._shrink_aware_role_plan_signature == signature
        ):
            return self._shrink_aware_role_plan

        stages = [
            int(item.strip())
            for item in str(
                envs_ascend.VLLM_ASCEND_SHRINK_AWARE_STAGES).split(",")
            if item.strip()
        ]
        stage_ranks = parse_stage_survivor_ranks(
            raw_stage_ranks, world_size=full_world_size)
        intermediate = parse_rank_list(
            envs_ascend.VLLM_ASCEND_SHRINK_AWARE_INTERMEDIATE_RANKS)
        final = parse_rank_list(
            envs_ascend.VLLM_ASCEND_SHRINK_AWARE_FINAL_RANKS)
        policy = envs_ascend.VLLM_ASCEND_SHRINK_AWARE_SURVIVOR_POLICY
        if stage_ranks is not None or (
            intermediate is not None and final is not None
        ):
            policy = "manual"
        plan = plan_survivor_ranks(
            world_size=full_world_size,
            shrink_stages=stages,
            package_topology=parse_rank_topology(
                envs_ascend.VLLM_ASCEND_SHRINK_AWARE_PACKAGE_TOPOLOGY,
                world_size=full_world_size,
            ),
            policy=policy,
            intermediate_survivor_ranks=intermediate,
            final_survivor_ranks=final,
            stage_survivor_ranks=stage_ranks,
        )
        self._shrink_aware_role_plan = plan
        self._shrink_aware_role_plan_signature = signature
        logger.info(
            "AdaFloor role plan loaded: stages=%s donor=%s final=%s "
            "target_policy=%s locality=%.4f",
            plan.stage_survivor_ranks,
            plan.donor_ranks,
            plan.final_survivor_ranks,
            envs_ascend.VLLM_ASCEND_SHRINK_AWARE_TARGET_POLICY,
            plan.package_locality_score,
        )
        return plan

    def _staged_shrink_target(
        self,
        natural_active_ranks: list[int],
        has_unfinished: bool,
        dp_world_size: int,
    ) -> tuple[bool, list[int]]:
        plan = self._get_shrink_aware_role_plan(dp_world_size)
        if plan is None:
            return False, natural_active_ranks

        full_world_size = self.elastic_original_dp_world_size or dp_world_size
        full_world_ranks = self._original_dp_global_ranks(full_world_size)
        current_global = (
            list(self.elastic_ep_active_ranks)
            if self.elastic_ep_active_ranks is not None
            else full_world_ranks
        )
        current_local = self._dp_global_to_local_ranks(
            current_global, full_world_size)
        unfinished_local = self._dp_global_to_local_ranks(
            natural_active_ranks, full_world_size)
        decision = decide_staged_shrink(
            enabled=True,
            mode=envs_ascend.VLLM_ASCEND_SHRINK_AWARE_MODE,
            current_active_ranks=current_local,
            unfinished_ranks=unfinished_local,
            role_plan=plan,
            min_window_seconds=(
                envs_ascend.VLLM_ASCEND_SHRINK_AWARE_MIN_WINDOW_SECONDS),
            estimated_window_seconds=None,
            allow_target_size=True,
            target_policy=(
                envs_ascend.VLLM_ASCEND_SHRINK_AWARE_TARGET_POLICY),
        )
        if decision.should_shrink:
            target_global = self._dp_local_to_global_ranks(
                decision.target_active_ranks, full_world_size)
            logger.info(
                "AdaFloor staged trigger: stage=%s current=%s unfinished=%s "
                "target=%s policy=%s",
                decision.stage_name,
                current_global,
                natural_active_ranks,
                target_global,
                envs_ascend.VLLM_ASCEND_SHRINK_AWARE_TARGET_POLICY,
            )
            return True, target_global

        current_rank = torch.distributed.get_rank()
        if not has_unfinished and current_rank in current_global:
            self.should_execute_dummy_batch = True
        return True, current_global

    def _should_delay_elastic_shrink_for_mc2(self, num_active_ranks: int, dp_world_size: int) -> bool:
        if num_active_ranks >= dp_world_size:
            return False
        if envs_ascend.VLLM_ASCEND_FORCE_ALLTOALL_MOE:
            return False
        if not self.vllm_config.parallel_config.enable_expert_parallel:
            return False

        hf_config = getattr(self.vllm_config.model_config, "hf_config", None)
        num_experts = getattr(hf_config, "num_experts", None)
        if not isinstance(num_experts, int) or num_experts <= 0:
            return False

        min_mc2_ep_size = envs_ascend.VLLM_ASCEND_MC2_MIN_EP_SIZE
        return num_active_ranks < min_mc2_ep_size or (num_experts % num_active_ranks) != 0

    def _should_enter_single_rank_no_ep_tail(self, num_active_ranks: int, dp_world_size: int) -> bool:
        if num_active_ranks != 1 or dp_world_size != 2:
            return False
        if envs_ascend.VLLM_ASCEND_ELASTIC_EXECUTION_MODE not in (2, 3):
            return False
        min_compute_group_size = self._get_configured_elastic_min_compute_group_size()
        if min_compute_group_size != 1:
            return False
        if not self.external_launcher_dp:
            return False
        if not self.vllm_config.parallel_config.enable_expert_parallel:
            return False
        return True

    @staticmethod
    def _is_power_of_two(value: int) -> bool:
        return value > 0 and (value & (value - 1)) == 0

    def _get_configured_elastic_min_compute_group_size(self) -> int | None:
        min_compute_group_size = envs_ascend.VLLM_ASCEND_ELASTIC_MIN_COMPUTE_GROUP_SIZE
        if min_compute_group_size is None:
            return None
        original_dp_world_size = self.elastic_original_dp_world_size
        if not isinstance(original_dp_world_size, int) or original_dp_world_size <= 0:
            raise ValueError(
                "VLLM_ASCEND_ELASTIC_MIN_COMPUTE_GROUP_SIZE requires external-launcher DP to be initialized."
            )
        if not self._is_power_of_two(original_dp_world_size):
            raise ValueError(
                "Elastic repeated shrink only supports power-of-two initial "
                f"DP size, got {original_dp_world_size}."
            )
        if min_compute_group_size > original_dp_world_size:
            raise ValueError(
                "VLLM_ASCEND_ELASTIC_MIN_COMPUTE_GROUP_SIZE cannot exceed "
                f"the initial DP size: {min_compute_group_size} > {original_dp_world_size}."
            )
        if original_dp_world_size % min_compute_group_size != 0:
            raise ValueError(
                "VLLM_ASCEND_ELASTIC_MIN_COMPUTE_GROUP_SIZE must divide the "
                f"initial DP size: {min_compute_group_size} vs {original_dp_world_size}."
            )
        return min_compute_group_size

    def _is_followup_elastic_shrink_enabled(self) -> bool:
        if not envs_ascend.VLLM_ASCEND_ENABLE_ELASTIC_PARALLEL_SHRINK:
            return False
        original_dp_world_size = self.elastic_original_dp_world_size
        if not isinstance(original_dp_world_size, int) or original_dp_world_size <= 1:
            return False
        min_compute_group_size = self._get_configured_elastic_min_compute_group_size()
        if min_compute_group_size is None:
            return True
        if min_compute_group_size >= original_dp_world_size:
            logger.info(
                "Elastic shrink disabled because configured floor matches the initial DP size: floor=%s initial_dp_size=%s",
                min_compute_group_size,
                original_dp_world_size,
            )
            return False
        return True

    def has_unfinished_requests_dp(self, has_unfinished: bool) -> bool:
        active_global_ranks = self._get_active_dp_global_ranks(has_unfinished)
        self.elastic_ep_runtime_active_ranks = tuple(active_global_ranks) if active_global_ranks else None

        if self.elastic_ep_no_dummy:
            aggregated_has_unfinished = len(active_global_ranks) > 0
            if not aggregated_has_unfinished:
                return False

            assert self.dp_group is not None
            dp_world_size = torch.distributed.get_world_size(group=self.dp_group)
            shrink_aware_active, staged_active_ranks = (
                self._staged_shrink_target(
                    active_global_ranks, has_unfinished, dp_world_size))
            if shrink_aware_active:
                active_global_ranks = staged_active_ranks
            if len(active_global_ranks) < dp_world_size:
                num_active_ranks = len(active_global_ranks)
                current_global_rank = torch.distributed.get_rank()
                allow_single_rank_no_ep_tail = self._should_enter_single_rank_no_ep_tail(
                    num_active_ranks, dp_world_size
                )
                min_mc2_ep_size = envs_ascend.VLLM_ASCEND_MC2_MIN_EP_SIZE
                min_compute_group_size = self._get_configured_elastic_min_compute_group_size()
                num_experts = None
                hf_config = getattr(self.vllm_config.model_config, "hf_config", None)
                if hf_config is not None:
                    num_experts = getattr(hf_config, "num_experts", None)
                delay_for_min_ep = num_active_ranks < min_mc2_ep_size
                delay_for_divisibility = (
                    isinstance(num_experts, int) and num_experts > 0 and (num_experts % num_active_ranks) != 0
                )
                if self._should_delay_elastic_shrink_for_mc2(num_active_ranks, dp_world_size) and not allow_single_rank_no_ep_tail:
                    if not has_unfinished:
                        if not self._logged_mc2_delay:
                            self._logged_mc2_delay = True
                            logger.info(
                                "Elastic shrink delayed for MC2 compatibility: global_rank=%s active_ranks=%s active_count=%s dp_world_size=%s delay_for_min_ep=%s delay_for_divisibility=%s local_has_unfinished=%s scaled_once=%s prev_active_ranks=%s",
                                current_global_rank,
                                active_global_ranks,
                                num_active_ranks,
                                dp_world_size,
                                delay_for_min_ep,
                                delay_for_divisibility,
                                has_unfinished,
                                self.elastic_ep_scaled_once,
                                self.elastic_ep_active_ranks,
                            )
                            if delay_for_min_ep and num_active_ranks == 1:
                                logger.info(
                                    "Elastic single-rank tail blocked: global_rank=%s surviving_rank=%s prev_active_ranks=%s current_group_size=%s configured_min_compute_group_size=%s configured_min_ep_size=%s tail_mode=no_ep",
                                    current_global_rank,
                                    active_global_ranks[0] if active_global_ranks else None,
                                    self.elastic_ep_active_ranks,
                                    dp_world_size,
                                    min_compute_group_size,
                                    min_mc2_ep_size,
                                )
                        self.should_execute_dummy_batch = True
                    return True

                active_ranks_tuple = tuple(active_global_ranks)
                if self.elastic_ep_active_ranks != active_ranks_tuple:
                    skip_reason = None
                    if min_compute_group_size is None:
                        if (
                            self.elastic_ep_active_ranks is not None
                            and len(active_global_ranks) < len(self.elastic_ep_active_ranks)
                            and set(active_global_ranks).issubset(set(self.elastic_ep_active_ranks))
                        ):
                            skip_reason = "default_single_shrink"
                    else:
                        if not self._is_power_of_two(dp_world_size):
                            skip_reason = f"current_group_not_power_of_two:{dp_world_size}"
                        elif num_active_ranks != (dp_world_size // 2):
                            skip_reason = f"non_halving_step:{dp_world_size}->{num_active_ranks}"
                        elif num_active_ranks < min_compute_group_size and not allow_single_rank_no_ep_tail:
                            skip_reason = f"below_configured_floor:{min_compute_group_size}"

                    if skip_reason is not None:
                        if self._logged_followup_shrink_skip != active_ranks_tuple:
                            logger.info(
                                "Elastic follow-up shrink skipped: global_rank=%s unfinished_active_ranks=%s prev_active_ranks=%s current_group_size=%s min_compute_group_size=%s reason=%s local_has_unfinished=%s",
                                current_global_rank,
                                active_global_ranks,
                                self.elastic_ep_active_ranks,
                                dp_world_size,
                                min_compute_group_size,
                                skip_reason,
                                has_unfinished,
                            )
                            self._logged_followup_shrink_skip = active_ranks_tuple
                        if current_global_rank not in active_global_ranks:
                            self.should_execute_dummy_batch = True
                        return True

                    if not has_unfinished:
                        if allow_single_rank_no_ep_tail:
                            logger.info(
                                "Elastic single-rank tail allowed: global_rank=%s surviving_rank=%s prev_active_ranks=%s current_group_size=%s configured_min_compute_group_size=%s tail_mode=no_ep",
                                current_global_rank,
                                active_global_ranks[0] if active_global_ranks else None,
                                self.elastic_ep_active_ranks,
                                dp_world_size,
                                min_compute_group_size,
                            )
                        logger.info(
                            "Elastic EP rank exit start: global_rank=%s active_ranks=%s",
                            current_global_rank,
                            active_global_ranks,
                        )

                    self.elastic_ep_active_ranks = active_ranks_tuple
                    self.elastic_ep_scaled_once = True
                    self._logged_followup_shrink_skip = None
                    rebuild_start_t = time.perf_counter()
                    self.engine_core.collective_rpc("rebuild_elastic_ep_group", args=(active_global_ranks,))
                    if self.external_launcher_dp:
                        if current_global_rank in active_global_ranks:
                            self.dp_group = get_dp_group().cpu_group
                        else:
                            self.dp_group = None
                            self.should_execute_dummy_batch = False
                    logger.info(
                        "Elastic parallel shrink rpc done: global_rank=%s active_ranks=%s total_ms=%.2f",
                        current_global_rank,
                        active_global_ranks,
                        (time.perf_counter() - rebuild_start_t) * 1000.0,
                    )
            return has_unfinished or self.should_execute_dummy_batch

        aggregated_has_unfinished = len(active_global_ranks) > 0
        if not has_unfinished and aggregated_has_unfinished:
            self.should_execute_dummy_batch = True
        return aggregated_has_unfinished

    def restore_elastic_parallel_groups_if_needed(self) -> None:
        if not self.elastic_ep_no_dummy:
            return
        local_need_restore = int(self.elastic_ep_scaled_once or self.elastic_ep_active_ranks is not None)
        need_restore = bool(local_need_restore)
        if torch.distributed.is_initialized():
            restore_tensor = torch.tensor([local_need_restore], dtype=torch.int32)
            torch.distributed.all_reduce(
                restore_tensor,
                op=torch.distributed.ReduceOp.MAX,
                group=get_world_group().cpu_group,
            )
            need_restore = bool(int(restore_tensor.item()))
        if not need_restore:
            return
        restore_start_t = time.perf_counter()
        self.engine_core.collective_rpc("restore_elastic_parallel_groups")
        if self.external_launcher_dp:
            self.dp_group = get_dp_group().cpu_group
        self.elastic_ep_scaled_once = False
        self.elastic_ep_active_ranks = None
        self.elastic_ep_runtime_active_ranks = None
        self._logged_followup_shrink_skip = None
        self._logged_mc2_delay = False
        self._shrink_aware_role_plan = None
        self._shrink_aware_role_plan_signature = None
        logger.info(
            "Elastic parallel groups restored: global_rank=%s total_ms=%.2f",
            torch.distributed.get_rank(),
            (time.perf_counter() - restore_start_t) * 1000.0,
        )

    @classmethod
    def validate_outputs(cls, outputs, output_type):
        return outputs

    def get_supported_tasks(self) -> tuple[SupportedTask, ...]:
        return self.engine_core.get_supported_tasks()

    def abort_request(self, request_ids: list[str], internal: bool = False) -> None:
        """Remove request_ids from EngineCore and Detokenizer."""

        request_ids = self.output_processor.abort_requests(request_ids, internal)
        self.engine_core.abort_requests(request_ids)

    def add_request(
        self,
        request_id: str,
        prompt: EngineCoreRequest | PromptType,
        params: SamplingParams | PoolingParams,
        arrival_time: float | None = None,
        lora_request: LoRARequest | None = None,
        tokenization_kwargs: dict[str, Any] | None = None,
        trace_headers: Mapping[str, str] | None = None,
        priority: int = 0,
        prompt_text: str | None = None,
    ) -> None:
        # Validate the request_id type.
        if not isinstance(request_id, str):
            raise TypeError(f"request_id must be a string, got {type(request_id)}")

        # Process raw inputs into the request.
        if isinstance(prompt, EngineCoreRequest):
            request = prompt
            if request_id != request.request_id:
                logger.warning_once(
                    "AsyncLLM.add_request() was passed a request_id parameter that "
                    "does not match the EngineCoreRequest.request_id attribute. The "
                    "latter will be used, and the former will be ignored."
                )
        else:
            assert prompt_text is None
            request = self.input_processor.process_inputs(
                request_id,
                prompt,
                params,
                arrival_time,
                lora_request,
                tokenization_kwargs,
                trace_headers,
                priority,
            )
            if isinstance(prompt, str):
                prompt_text = prompt
            elif isinstance(prompt, Mapping):
                prompt_text = cast(str | None, prompt.get("prompt"))

        self.input_processor.assign_request_id(request)

        # Use cloned params that may have been updated in process_inputs()
        params = request.params

        n = params.n if isinstance(params, SamplingParams) else 1

        if n == 1:
            # Make a new RequestState and queue.
            self.output_processor.add_request(request, prompt_text, None, 0)
            # Add the request to EngineCore.
            self.engine_core.add_request(request)
            return

        # Fan out child requests (for n>1).
        parent_req = ParentRequest(request)
        for idx in range(n):
            request_id, child_params = parent_req.get_child_info(idx)
            child_request = request if idx == n - 1 else copy(request)
            child_request.request_id = request_id
            child_request.sampling_params = child_params

            # Make a new RequestState and queue.
            self.output_processor.add_request(
                child_request, prompt_text, parent_req, idx
            )
            # Add the request to EngineCore.
            self.engine_core.add_request(child_request)

    def step(self) -> list[RequestOutput | PoolingRequestOutput]:
        active_rank_count = None
        if self.elastic_ep_runtime_active_ranks is not None:
            active_rank_count = len(self.elastic_ep_runtime_active_ranks)
        elif self.elastic_ep_active_ranks is not None:
            active_rank_count = len(self.elastic_ep_active_ranks)
        elif isinstance(self.elastic_original_dp_world_size, int):
            active_rank_count = self.elastic_original_dp_world_size
        elif self.dp_group is not None:
            active_rank_count = torch.distributed.get_world_size(group=self.dp_group)
        else:
            active_rank_count = int(getattr(self.vllm_config.parallel_config, "data_parallel_size", 1))
        moe_stats.set_step_context(active_rank_count=active_rank_count)

        if self.should_execute_dummy_batch:
            self.should_execute_dummy_batch = False
            self.engine_core.execute_dummy_batch()
            return []

        # 1) Get EngineCoreOutput from the EngineCore.
        with record_function_or_nullcontext("llm_engine step: get_output"):
            outputs = self.engine_core.get_output()

        # 2) Process EngineCoreOutputs.
        with record_function_or_nullcontext("llm_engine step: process_outputs"):
            iteration_stats = IterationStats() if self.log_stats else None
            processed_outputs = self.output_processor.process_outputs(
                outputs.outputs,
                engine_core_timestamp=outputs.timestamp,
                iteration_stats=iteration_stats,
            )
            self.output_processor.update_scheduler_stats(outputs.scheduler_stats)

        # 3) Abort any reqs that finished due to stop strings.
        with record_function_or_nullcontext("llm_engine step: abort_requests"):
            self.engine_core.abort_requests(processed_outputs.reqs_to_abort)

        # 4) Record stats
        with record_function_or_nullcontext("llm_engine step: record_stats"):
            if self.logger_manager is not None and outputs.scheduler_stats is not None:
                self.logger_manager.record(
                    scheduler_stats=outputs.scheduler_stats,
                    iteration_stats=iteration_stats,
                    mm_cache_stats=self.input_processor.stat_mm_cache(),
                )
                self.do_log_stats_with_interval()

        return processed_outputs.request_outputs

    def start_profile(self):
        self.engine_core.profile(True)

    def stop_profile(self):
        self.engine_core.profile(False)

    def reset_mm_cache(self):
        self.input_processor.clear_mm_cache()
        self.engine_core.reset_mm_cache()

    def reset_prefix_cache(
        self, reset_running_requests: bool = False, reset_connector: bool = False
    ) -> bool:
        return self.engine_core.reset_prefix_cache(
            reset_running_requests, reset_connector
        )

    def sleep(self, level: int = 1):
        self.engine_core.sleep(level)

        if self.logger_manager is not None:
            self.logger_manager.record_sleep_state(1, level)

    def wake_up(self, tags: list[str] | None = None):
        self.engine_core.wake_up(tags)

        if self.logger_manager is not None:
            self.logger_manager.record_sleep_state(0, 0)

    def is_sleeping(self) -> bool:
        return self.engine_core.is_sleeping()

    def get_metrics(self) -> list[Metric]:
        assert self.log_stats, "Stat logging disabled"
        return get_metrics_snapshot()

    @property
    def tokenizer(self) -> TokenizerLike | None:
        return self.input_processor.tokenizer

    def get_tokenizer(self) -> TokenizerLike:
        if self.tokenizer is None:
            raise ValueError(
                "Unable to get tokenizer because `skip_tokenizer_init=True`"
            )

        return self.tokenizer

    def do_log_stats(self) -> None:
        """Log stats if logging is enabled."""
        if self.logger_manager:
            self.logger_manager.log()

    def do_log_stats_with_interval(self) -> None:
        """Log stats when the time interval has passed."""
        now = time.time()
        if not hasattr(self, "_last_log_time"):
            self._last_log_time = now
        if now - self._last_log_time >= envs.VLLM_LOG_STATS_INTERVAL:
            self.do_log_stats()
            self._last_log_time = now

    def add_lora(self, lora_request: LoRARequest) -> bool:
        """Load a new LoRA adapter into the engine for future requests."""
        return self.engine_core.add_lora(lora_request)

    def remove_lora(self, lora_id: int) -> bool:
        """Remove an already loaded LoRA adapter."""
        return self.engine_core.remove_lora(lora_id)

    def list_loras(self) -> set[int]:
        """List all registered adapters."""
        return self.engine_core.list_loras()

    def pin_lora(self, lora_id: int) -> bool:
        """Prevent an adapter from being evicted."""
        return self.engine_core.pin_lora(lora_id)

    def collective_rpc(
        self,
        method: str | Callable[[WorkerBase], _R],
        timeout: float | None = None,
        args: tuple = (),
        kwargs: dict[str, Any] | None = None,
    ) -> list[_R]:
        return self.engine_core.collective_rpc(method, timeout, args, kwargs)

    def apply_model(self, func: Callable[[nn.Module], _R]) -> list[_R]:
        return self.collective_rpc("apply_model", args=(func,))

    def __del__(self):
        dp_group = getattr(self, "dp_group", None)
        if dp_group is not None and not self.external_launcher_dp:
            stateless_destroy_torch_distributed_process_group(dp_group)
