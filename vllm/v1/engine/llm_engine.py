# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import time
from collections.abc import Mapping
from copy import copy
from typing import Any, Callable, Optional, Union

import torch.nn as nn
from typing_extensions import TypeVar

import vllm.envs as envs
import vllm_ascend.envs as envs_ascend
from vllm.config import ParallelConfig, VllmConfig
from vllm.distributed import stateless_destroy_torch_distributed_process_group
from vllm.distributed.parallel_state import get_dp_group
from vllm.engine.arg_utils import EngineArgs
from vllm.inputs import PromptType
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.multimodal import MULTIMODAL_REGISTRY, MultiModalRegistry
from vllm.outputs import PoolingRequestOutput, RequestOutput
from vllm.pooling_params import PoolingParams
from vllm.sampling_params import SamplingParams
from vllm.tasks import SupportedTask
from vllm.tracing import init_tracer
from vllm.transformers_utils.tokenizer import (AnyTokenizer,
                                               init_tokenizer_from_configs)
from vllm.usage.usage_lib import UsageContext
from vllm.utils import Device
from vllm.v1.engine.core_client import EngineCoreClient
from vllm.v1.engine.output_processor import OutputProcessor
from vllm.v1.engine.parallel_sampling import ParentRequest
from vllm.v1.engine.processor import Processor
from vllm.v1.executor.abstract import Executor
from vllm.v1.metrics.loggers import StatLoggerFactory, StatLoggerManager
from vllm.v1.metrics.reader import Metric, get_metrics_snapshot
from vllm.v1.metrics.stats import IterationStats
from vllm.v1.worker.worker_base import WorkerBase
import torch

logger = init_logger(__name__)

_R = TypeVar("_R", default=Any)


class LLMEngine:
    """Legacy LLMEngine for backwards compatibility."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        executor_class: type[Executor],
        log_stats: bool,
        usage_context: UsageContext = UsageContext.ENGINE_CONTEXT,
        stat_loggers: Optional[list[StatLoggerFactory]] = None,
        mm_registry: MultiModalRegistry = MULTIMODAL_REGISTRY,
        use_cached_outputs: bool = False,
        multiprocess_mode: bool = False,
    ) -> None:
        if not envs.VLLM_USE_V1:
            raise ValueError(
                "Using V1 LLMEngine, but envs.VLLM_USE_V1=False. "
                "This should not happen. As a workaround, try using "
                "LLMEngine.from_vllm_config(...) or explicitly set "
                "VLLM_USE_V1=0 or 1 and report this issue on Github.")

        if stat_loggers is not None:
            raise NotImplementedError(
                "Passing StatLoggers to LLMEngine in V1 is not yet supported. "
                "Set VLLM_USE_V1=0 and file and issue on Github.")

        self.vllm_config = vllm_config
        self.observability_config = vllm_config.observability_config
        self.model_config = vllm_config.model_config
        self.cache_config = vllm_config.cache_config

        self.log_stats = log_stats

        executor_backend = (
            self.vllm_config.parallel_config.distributed_executor_backend)
        parallel_config = vllm_config.parallel_config
        self.external_launcher_dp = (parallel_config.data_parallel_size > 1 and
                                     executor_backend == "external_launcher")
        # important: init dp group before init the engine_core
        print("multiprocess_mode:", multiprocess_mode, "external_launcher_dp:",
              self.external_launcher_dp)
        # In the decoupled engine case this is handled in EngineCoreProc.
        if not multiprocess_mode and parallel_config.data_parallel_size > 1 \
            and not self.external_launcher_dp:
            self.dp_group = parallel_config.stateless_init_dp_group()
        else:
            self.dp_group = None
        self.should_execute_dummy_batch = False

        if self.model_config.skip_tokenizer_init:
            self.tokenizer = None
        else:
            # Tokenizer (+ ensure liveness if running in another process).
            self.tokenizer = init_tokenizer_from_configs(
                model_config=vllm_config.model_config)

        # Processor (convert Inputs --> EngineCoreRequests)
        self.processor = Processor(vllm_config=vllm_config,
                                   tokenizer=self.tokenizer,
                                   mm_registry=mm_registry)

        # OutputProcessor (convert EngineCoreOutputs --> RequestOutput).
        self.output_processor = OutputProcessor(self.tokenizer,
                                                log_stats=self.log_stats)
        if self.observability_config.otlp_traces_endpoint is not None:
            tracer = init_tracer(
                "vllm.llm_engine",
                self.observability_config.otlp_traces_endpoint)
            self.output_processor.tracer = tracer

        # EngineCore (gets EngineCoreRequests and gives EngineCoreOutputs)
        self.engine_core = EngineCoreClient.make_client(
            multiprocess_mode=multiprocess_mode,
            asyncio_mode=False,
            vllm_config=vllm_config,
            executor_class=executor_class,
            log_stats=self.log_stats,
        )

        self.logger_manager: Optional[StatLoggerManager] = None
        if self.log_stats:
            self.logger_manager = StatLoggerManager(
                vllm_config=vllm_config,
                custom_stat_loggers=stat_loggers,
                enable_default_loggers=log_stats,
            )
            self.logger_manager.log_engine_initialized()

        if not multiprocess_mode:
            # for v0 compatibility
            self.model_executor = self.engine_core.engine_core.model_executor  # type: ignore

        if self.external_launcher_dp:
            # If we use DP in external launcher mode, we reuse the
            # existing DP group used for data communication.
            # 当前进程在 dp_pg 里的 rank
            self.dp_group = get_dp_group().cpu_group
            dp_pg = self.dp_group
            dp_rank = torch.distributed.get_rank(group=dp_pg)
            # dp_pg 的总大小（DP 组里有多少个进程）
            dp_size = torch.distributed.get_world_size(group=dp_pg)
            print(f"[DP Info] dp_group rank = {dp_rank}, dp_group world_size = {dp_size}")

        # Elastic no-dummy path is only enabled for eager external-launcher DP.
        self.elastic_ep_no_dummy = (
            envs_ascend.VLLM_ASCEND_ENABLE_ELASTIC_PARALLEL_SHRINK
            and self.external_launcher_dp and self.model_config.enforce_eager)
        self.elastic_ep_scaled_once = False
        self.elastic_ep_active_ranks: Optional[tuple[int, ...]] = None
        
        #new code here
        self.dummy_times = 0
        self.total_step_times = 0
        # Don't keep the dummy data in memory
        self.reset_mm_cache()

    @classmethod
    def from_vllm_config(
        cls,
        vllm_config: VllmConfig,
        usage_context: UsageContext = UsageContext.ENGINE_CONTEXT,
        stat_loggers: Optional[list[StatLoggerFactory]] = None,
        disable_log_stats: bool = False,
    ) -> "LLMEngine":
        return cls(vllm_config=vllm_config,
                   executor_class=Executor.get_class(vllm_config),
                   log_stats=(not disable_log_stats),
                   usage_context=usage_context,
                   stat_loggers=stat_loggers,
                   multiprocess_mode=envs.VLLM_ENABLE_V1_MULTIPROCESSING)

    @classmethod
    def from_engine_args(
        cls,
        engine_args: EngineArgs,
        usage_context: UsageContext = UsageContext.ENGINE_CONTEXT,
        stat_loggers: Optional[list[StatLoggerFactory]] = None,
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
        return cls(vllm_config=vllm_config,
                   executor_class=executor_class,
                   log_stats=not engine_args.disable_log_stats,
                   usage_context=usage_context,
                   stat_loggers=stat_loggers,
                   multiprocess_mode=enable_multiprocessing)

    def get_num_unfinished_requests(self) -> int:
        return self.output_processor.get_num_unfinished_requests()

    def has_unfinished_requests(self) -> bool:
        has_unfinished = self.output_processor.has_unfinished_requests()
        if self.dp_group is None:
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

    def _should_delay_elastic_shrink_for_mc2(self, num_active_ranks: int,
                                             dp_world_size: int) -> bool:
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
        return (num_active_ranks < min_mc2_ep_size
                or (num_experts % num_active_ranks) != 0)

    def has_unfinished_requests_dp(self, has_unfinished: bool) -> bool:
        if self.elastic_ep_no_dummy:
            active_global_ranks = self._get_active_dp_global_ranks(
                has_unfinished)
            aggregated_has_unfinished = len(active_global_ranks) > 0
            if not aggregated_has_unfinished:
                return False

            assert self.dp_group is not None
            dp_world_size = torch.distributed.get_world_size(group=self.dp_group)
            if len(active_global_ranks) < dp_world_size:
                num_active_ranks = len(active_global_ranks)
                min_mc2_ep_size = envs_ascend.VLLM_ASCEND_MC2_MIN_EP_SIZE
                num_experts = None
                hf_config = getattr(self.vllm_config.model_config, "hf_config",
                                    None)
                if hf_config is not None:
                    num_experts = getattr(hf_config, "num_experts", None)
                delay_for_min_ep = num_active_ranks < min_mc2_ep_size
                delay_for_divisibility = (
                    isinstance(num_experts, int) and num_experts > 0
                    and (num_experts % num_active_ranks) != 0)
                if self._should_delay_elastic_shrink_for_mc2(
                        num_active_ranks, dp_world_size):
                    if not has_unfinished:
                        logger.info(
                            "Elastic shrink delayed for MC2 compatibility: global_rank=%s active_ranks=%s active_count=%s dp_world_size=%s delay_for_min_ep=%s delay_for_divisibility=%s local_has_unfinished=%s scaled_once=%s prev_active_ranks=%s",
                            torch.distributed.get_rank(),
                            active_global_ranks,
                            num_active_ranks,
                            dp_world_size,
                            delay_for_min_ep,
                            delay_for_divisibility,
                            has_unfinished,
                            self.elastic_ep_scaled_once,
                            self.elastic_ep_active_ranks)
                        self.should_execute_dummy_batch = True
                    return True
                active_ranks_tuple = tuple(active_global_ranks)
                if self.elastic_ep_active_ranks != active_ranks_tuple:
                    if not has_unfinished:
                        logger.info(
                            "Elastic EP rank exit start: global_rank=%s active_ranks=%s",
                            torch.distributed.get_rank(), active_global_ranks)
                    self.elastic_ep_active_ranks = active_ranks_tuple
                    self.elastic_ep_scaled_once = True
                    rebuild_start_t = time.perf_counter()
                    self.engine_core.collective_rpc(
                        "rebuild_elastic_ep_group",
                        args=(active_global_ranks, ))
                    if self.external_launcher_dp:
                        current_global_rank = torch.distributed.get_rank()
                        if current_global_rank in active_global_ranks:
                            self.dp_group = get_dp_group().cpu_group
                        else:
                            self.dp_group = None
                    logger.info(
                        "Elastic parallel shrink rpc done: global_rank=%s active_ranks=%s total_ms=%.2f",
                        torch.distributed.get_rank(), active_global_ranks,
                        (time.perf_counter() - rebuild_start_t) * 1000.0)
            return has_unfinished

        aggregated_has_unfinished = ParallelConfig.has_unfinished_dp(
            self.dp_group, has_unfinished)
        if not has_unfinished and aggregated_has_unfinished:
            self.should_execute_dummy_batch = True
        return aggregated_has_unfinished

    def restore_elastic_parallel_groups_if_needed(self) -> None:
        if not self.elastic_ep_no_dummy or not self.elastic_ep_scaled_once:
            return
        restore_start_t = time.perf_counter()
        self.engine_core.collective_rpc("restore_elastic_parallel_groups")
        if self.external_launcher_dp:
            self.dp_group = get_dp_group().cpu_group
        self.elastic_ep_scaled_once = False
        self.elastic_ep_active_ranks = None
        logger.info(
            "Elastic parallel groups restored: global_rank=%s total_ms=%.2f",
            torch.distributed.get_rank(),
            (time.perf_counter() - restore_start_t) * 1000.0)

    @classmethod
    def validate_outputs(cls, outputs, output_type):
        return outputs

    def get_supported_tasks(self) -> tuple[SupportedTask, ...]:
        return self.engine_core.get_supported_tasks()

    def abort_request(self, request_ids: list[str]) -> None:
        """Remove request_ids from EngineCore and Detokenizer."""

        request_ids = self.output_processor.abort_requests(request_ids)
        self.engine_core.abort_requests(request_ids)

    def add_request(
        self,
        request_id: str,
        prompt: PromptType,
        params: Union[SamplingParams, PoolingParams],
        arrival_time: Optional[float] = None,
        lora_request: Optional[LoRARequest] = None,
        tokenization_kwargs: Optional[dict[str, Any]] = None,
        trace_headers: Optional[Mapping[str, str]] = None,
        priority: int = 0,
    ) -> None:
        # Validate the request_id type.
        if not isinstance(request_id, str):
            raise TypeError(
                f"request_id must be a string, got {type(request_id)}")

        # Process raw inputs into the request.
        prompt_str, request = self.processor.process_inputs(
            request_id, prompt, params, arrival_time, lora_request,
            tokenization_kwargs, trace_headers, priority)

        n = params.n if isinstance(params, SamplingParams) else 1

        if n == 1:
            # Make a new RequestState and queue.
            self.output_processor.add_request(request, prompt_str, None, 0)
            # Add the request to EngineCore.
            self.engine_core.add_request(request)
            return

        # Fan out child requests (for n>1).
        parent_req = ParentRequest(request_id, params)
        for idx in range(n):
            request_id, params = parent_req.get_child_info(idx)
            child_request = request if idx == n - 1 else copy(request)
            child_request.request_id = request_id
            child_request.sampling_params = params

            # Make a new RequestState and queue.
            self.output_processor.add_request(child_request, prompt_str,
                                              parent_req, idx)
            # Add the request to EngineCore.
            self.engine_core.add_request(child_request)

    def step(self) -> Union[list[RequestOutput], list[PoolingRequestOutput]]:
        self.total_step_times += 1

        if self.should_execute_dummy_batch:
            self.should_execute_dummy_batch = False
            #do something replace execute dummy batch
            #record dummy_run time
            # print("rank", torch.distributed.get_rank(group=self.dp_group), "executed dummy batch")
            # begin_time = time.time()
            self.engine_core.execute_dummy_batch()
            # end=time.time()
            # self.dummy_times += 1
            # print("rank", torch.distributed.get_rank(group=self.dp_group), "dummy_times:", self.dummy_times)
            # print("rank", torch.distributed.get_rank(group=self.dp_group), "total_step_times:", self.total_step_times)
            # print("rank", torch.distributed.get_rank(group=self.dp_group), "dummy batch time:", end-begin_time)
            return []

        # 1) Get EngineCoreOutput from the EngineCore.
        outputs = self.engine_core.get_output()

        # 2) Process EngineCoreOutputs.
        iteration_stats = IterationStats() if self.log_stats else None
        processed_outputs = self.output_processor.process_outputs(
            outputs.outputs,
            engine_core_timestamp=outputs.timestamp,
            iteration_stats=iteration_stats)

        # 3) Abort any reqs that finished due to stop strings.
        self.engine_core.abort_requests(processed_outputs.reqs_to_abort)

        # 4) Record stats
        if self.logger_manager is not None:
            assert outputs.scheduler_stats is not None
            self.logger_manager.record(
                scheduler_stats=outputs.scheduler_stats,
                iteration_stats=iteration_stats,
            )
            self.do_log_stats_with_interval()

        return processed_outputs.request_outputs

    def get_vllm_config(self):
        return self.vllm_config

    def get_model_config(self):
        return self.model_config

    def start_profile(self):
        self.engine_core.profile(True)

    def stop_profile(self):
        self.engine_core.profile(False)

    def reset_mm_cache(self):
        self.processor.clear_cache()
        self.engine_core.reset_mm_cache()

    def reset_prefix_cache(self, device: Optional[Device] = None):
        self.engine_core.reset_prefix_cache()

    def sleep(self, level: int = 1):
        self.engine_core.sleep(level)

    def wake_up(self, tags: Optional[list[str]] = None):
        self.engine_core.wake_up(tags)

    def is_sleeping(self) -> bool:
        return self.engine_core.is_sleeping()

    def get_metrics(self) -> list[Metric]:
        assert self.log_stats, "Stat logging disabled"
        return get_metrics_snapshot()

    def get_tokenizer(self) -> AnyTokenizer:
        if self.tokenizer is None:
            raise ValueError("Unable to get tokenizer because "
                             "skip_tokenizer_init is True")

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

    def collective_rpc(self,
                       method: Union[str, Callable[[WorkerBase], _R]],
                       timeout: Optional[float] = None,
                       args: tuple = (),
                       kwargs: Optional[dict[str, Any]] = None) -> list[_R]:
        return self.engine_core.collective_rpc(method, timeout, args, kwargs)

    def apply_model(self, func: Callable[[nn.Module], _R]) -> list[_R]:
        return self.collective_rpc("apply_model", args=(func, ))

    def __del__(self):
        if dp_group := getattr(self, "dp_group",
                               None) and not self.external_launcher_dp:
            stateless_destroy_torch_distributed_process_group(dp_group)
