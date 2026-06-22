import contextlib
import os
import queue
import random
import threading
import time
import weakref
from collections import deque
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import AdamW
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config

from vllm_ascend.draft.model.qwen3_eagle3 import (
    LoRALinear,
    Qwen3ModelEagle3,
    apply_lora_to_linear_modules,
)

try:
    from peft import LoraConfig, get_peft_model
except Exception:
    LoraConfig = None
    get_peft_model = None


def _get_env_flag(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _coerce_int(value: object, default: int) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _coerce_float(value: object, default: float) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


@dataclass
class DraftTrainerConfig:
    enabled: bool
    warmup_on_init: bool
    startup_warmup_steps: int
    step_budget_ms: float
    lr: float
    prediction_length: int
    max_seq_len: int
    draft_vocab_size: int
    queue_size: int
    async_train: bool
    profile_sync: bool
    profile_breakdown: bool
    reuse_target_emb_lm: bool
    profile_only: bool
    attn_impl: str
    attn_chunk_size: int
    sparse_kl_enabled: bool
    sparse_kl_topk: int
    micro_seq_len: int
    grad_accum_steps: int
    fastrl_concat_enabled: bool
    fastrl_concat_batch_size: int
    fastrl_concat_recent_samples: int
    fastrl_concat_window_len: int


@dataclass
class DraftSample:
    input_ids: torch.Tensor
    positions: Optional[torch.Tensor]
    hidden_states: torch.Tensor
    response_mask: Optional[torch.Tensor] = None


@dataclass
class DraftSeqBuffer:
    input_ids: list[torch.Tensor]
    positions: list[torch.Tensor]
    hidden_states: list[torch.Tensor]
    response_masks: list[torch.Tensor]


@dataclass
class DraftStepBatch:
    req_ids: list[str]
    query_start_loc: torch.Tensor
    input_ids: torch.Tensor
    positions: torch.Tensor
    hidden_states: torch.Tensor
    response_start_locs: Optional[torch.Tensor] = None


@dataclass
class DraftFinalizeReqs:
    req_ids: list[str]


@dataclass
class DraftTrainTask:
    layer_idx: int
    enqueue_ts: float
    done_event: Optional[threading.Event] = None
    error: Optional[BaseException] = None


class DraftTrainer:
    """Train an Eagle-3 draft model during dummy-stage attention gaps."""

    def __init__(self, model: nn.Module, config: DraftTrainerConfig):
        self._model_ref = weakref.ref(model)
        self.config = config
        if self.config.profile_only:
            # Ensure timing includes device sync when profiling only.
            self.config.profile_sync = True
            self.config.warmup_on_init = False
            self.config.async_train = False
        self._initialized = False
        self.device = torch.device("cpu")
        self.train_dtype = self._resolve_train_dtype()
        self.eagle3_model: Optional[Qwen3ModelEagle3] = None
        self.optimizer: Optional[AdamW] = None
        self._target_embed_tokens: Optional[nn.Module] = None
        self._target_lm_head: Optional[nn.Module] = None
        self._warmup_done = False
        self._startup_warmup_done = False
        self._profile_only_done = False
        self.num_calls = 0
        self.num_over_budget = 0
        self.num_train_steps = 0
        self.last_loss: Optional[float] = None
        self.num_samples_enqueued = 0
        self.num_samples_built = 0
        self.num_samples_dropped = 0
        self.num_samples_empty = 0
        self._latest_sample: Optional[DraftSample] = None
        self._profile_reuse_random_batch = _get_env_flag(
            "VLLM_ASCEND_DRAFT_PROFILE_REUSE_RANDOM_BATCH", False)
        self._profile_random_batch_cache: dict[int, Tuple[torch.Tensor, ...]] = {}
        self._active_train_batch: Optional[Tuple[torch.Tensor, ...]] = None
        self._active_train_seq_len = 0
        self._active_train_cursor = 0
        self._active_train_valid_tokens = 1.0
        self._grad_accum_counter = 0
        self._req_buffers: dict[str, DraftSeqBuffer] = {}
        self._sample_pool_lock = threading.Lock()
        self._sample_pool: deque[DraftSample] = deque(
            maxlen=max(1, int(self.config.fastrl_concat_recent_samples)))
        self._event_queue: "queue.Queue[Optional[object]]" = queue.Queue(
            maxsize=self.config.queue_size)
        self._ready_queue: "queue.Queue[DraftSample]" = queue.Queue(
            maxsize=self.config.queue_size)
        self._builder_thread: Optional[threading.Thread] = None
        self._train_queue: "queue.Queue[Optional[DraftTrainTask]]" = queue.Queue(
            maxsize=1)
        self._train_thread: Optional[threading.Thread] = None
        self._train_lock = threading.Lock()
        self._profile_steps_done = 0
        self._lora_enabled = _get_env_flag("VLLM_ASCEND_DRAFT_LORA_ENABLE",
                                           False)
        self._lora_applied = False
        self._lora_replaced_linear = 0
        self._lora_backend = os.getenv("VLLM_ASCEND_DRAFT_LORA_BACKEND",
                                       "peft").strip().lower()
        self._lora_target_modules = {
            name.strip()
            for name in os.getenv(
                "VLLM_ASCEND_DRAFT_LORA_TARGET_MODULES",
                "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj,fc",
            ).split(",") if name.strip()
        }
        self._lora_rank = max(
            1, _coerce_int(os.getenv("VLLM_ASCEND_DRAFT_LORA_RANK"), 8))
        self._lora_alpha = max(
            1.0, _coerce_float(os.getenv("VLLM_ASCEND_DRAFT_LORA_ALPHA"),
                               float(self._lora_rank)))
        self._lora_dropout = max(
            0.0, _coerce_float(os.getenv("VLLM_ASCEND_DRAFT_LORA_DROPOUT"), 0.0))
        self._dump_enabled = _get_env_flag(
            "VLLM_ASCEND_DRAFT_DUMP_ENABLE", False)
        self._dump_dir = os.getenv(
            "VLLM_ASCEND_DRAFT_DUMP_DIR",
            "./result/draft_dataset",
        ).strip()
        self._dump_every = max(
            1, _coerce_int(os.getenv("VLLM_ASCEND_DRAFT_DUMP_EVERY"), 1))
        self._dump_hidden_dtype = self._resolve_dump_hidden_dtype()
        self._dump_seen = 0
        self._dump_saved = 0
        self._capture_enabled = self.config.enabled or self._dump_enabled
        if self._dump_enabled:
            os.makedirs(self._dump_dir, exist_ok=True)
            print(
                "[DraftTrainer] data dump "
                f"enabled={self._dump_enabled} dir={self._dump_dir} "
                f"every={self._dump_every} "
                f"hidden_dtype={self._dump_hidden_dtype}"
            )
        if (self.config.enabled and self.config.warmup_on_init
                and not self.config.reuse_target_emb_lm
                and not self.config.profile_only):
            print("[DraftTrainer] warmup triggered by __init__")
            self.warmup()

    def _resolve_train_dtype(self) -> torch.dtype:
        dtype_name = os.getenv("VLLM_ASCEND_DRAFT_TRAIN_DTYPE", "bf16").strip().lower()
        if dtype_name in {"bf16", "bfloat16"}:
            return torch.bfloat16
        if dtype_name in {"fp16", "float16", "half"}:
            return torch.float16
        if dtype_name in {"fp32", "float32", "float"}:
            return torch.float32
        print(
            "[DraftTrainer] Unknown VLLM_ASCEND_DRAFT_TRAIN_DTYPE="
            f"{dtype_name}, fallback to bf16")
        return torch.bfloat16

    def _resolve_dump_hidden_dtype(self) -> torch.dtype:
        dtype_name = os.getenv(
            "VLLM_ASCEND_DRAFT_DUMP_HIDDEN_DTYPE", "bf16").strip().lower()
        if dtype_name in {"bf16", "bfloat16"}:
            return torch.bfloat16
        if dtype_name in {"fp16", "float16", "half"}:
            return torch.float16
        if dtype_name in {"fp32", "float32", "float"}:
            return torch.float32
        print(
            "[DraftTrainer] Unknown VLLM_ASCEND_DRAFT_DUMP_HIDDEN_DTYPE="
            f"{dtype_name}, fallback to bf16")
        return torch.bfloat16

    def _get_profile_random_batch(self, seq_len: int) -> Tuple[torch.Tensor, ...]:
        seq_len = max(1, min(seq_len, self.config.max_seq_len))
        cached = self._profile_random_batch_cache.get(seq_len)
        if cached is None:
            cached = self._random_training_batch(seq_len=seq_len)
            self._profile_random_batch_cache[seq_len] = cached
        return cached

    def _clone_batch_tensors(self, batch: Tuple[torch.Tensor, ...]) -> Tuple[torch.Tensor, ...]:
        return tuple(t.clone() for t in batch)

    def _forward_draft_batch(
        self,
        batch: Tuple[torch.Tensor, ...],
        prediction_length: int,
    ) -> Tuple[list[torch.Tensor], list[float]]:
        if self.eagle3_model is None:
            raise RuntimeError("Draft model is not initialized.")
        if len(batch) == 5:
            return self.eagle3_model(
                base_model_hidden_states=batch[0],
                input_ids=batch[1],
                position_ids=batch[2],
                target=batch[3],
                loss_mask=batch[4],
                use_cache=False,
                prediction_length=prediction_length,
            )
        if len(batch) == 6:
            return self.eagle3_model(
                base_model_hidden_states=batch[0],
                input_ids=batch[1],
                position_ids=batch[2],
                target=None,
                target_topk_idx=batch[3],
                target_topk_vals=batch[4],
                loss_mask=batch[5],
                use_cache=False,
                prediction_length=prediction_length,
            )
        raise RuntimeError(f"Unexpected draft batch format: len={len(batch)}")

    def bind_target_layers(
        self,
        embed_tokens: Optional[nn.Module],
        lm_head: Optional[nn.Module],
    ) -> None:
        """Attach target embedding/LM head weights and freeze them."""
        self._target_embed_tokens = embed_tokens
        self._target_lm_head = lm_head
        if not self._initialized:
            self._lazy_init()
        if self._initialized:
            self._attach_target_layers()
            self._configure_trainable_params()
            if self.eagle3_model is not None:
                self.optimizer = AdamW(
                    [p for p in self.eagle3_model.parameters()
                     if p.requires_grad],
                    lr=self.config.lr,
                )
            if self.config.enabled and self.config.warmup_on_init and not self._warmup_done:
                print("[DraftTrainer] warmup triggered by bind_target_layers")
                self.warmup()
            if (self.config.enabled and self.config.startup_warmup_steps > 0
                    and not self._startup_warmup_done
                    and not self.config.profile_only):
                self._startup_warmup()
        if self.config.profile_only:
            self._run_profile_only()

    def _run_profile_only(self) -> None:
        if self._profile_only_done:
            return
        self._profile_only_done = True
        if not self._initialized:
            self._lazy_init()
        if self.eagle3_model is None or self.optimizer is None:
            return

        warmup_steps = _coerce_int(
            os.getenv("VLLM_ASCEND_DRAFT_PROFILE_ONLY_WARMUP_STEPS", "1"), 1)
        warmup_steps = max(0, warmup_steps)
        steps = _coerce_int(
            os.getenv("VLLM_ASCEND_DRAFT_PROFILE_ONLY_STEPS", "1"), 1)
        steps = max(1, steps)

        def _run_one_step(tag: str) -> int:
            start = time.perf_counter()
            sync_before_ms = 0.0
            sync_after_ms = 0.0
            sync_fwd_before_ms = 0.0
            sync_fwd_after_ms = 0.0
            sync_bwd_before_ms = 0.0
            sync_bwd_after_ms = 0.0
            sync_opt_before_ms = 0.0
            sync_opt_after_ms = 0.0
            batch_ms = 0.0
            clone_ms = 0.0
            compute_ms = 0.0
            fwd_ms = 0.0
            bwd_ms = 0.0
            opt_ms = 0.0
            if self.config.profile_sync:
                sync_start = time.perf_counter()
                self._synchronize_device()
                sync_before_ms = (time.perf_counter() - sync_start) * 1000.0
            with torch.inference_mode(False), torch.enable_grad():
                batch_start = time.perf_counter()
                batch = self._next_training_batch()
                if batch is None:
                    if self._profile_reuse_random_batch:
                        batch = self._get_profile_random_batch(
                            seq_len=max(1, self.config.max_seq_len))
                    else:
                        batch = self._random_training_batch(
                            seq_len=max(1, self.config.max_seq_len))
                batch_ms = (time.perf_counter() - batch_start) * 1000.0
                clone_start = time.perf_counter()
                batch = self._clone_batch_tensors(batch)
                clone_ms = (time.perf_counter() - clone_start) * 1000.0
                seq_len = int(batch[1].shape[1])
                if self.config.profile_sync:
                    sync_start = time.perf_counter()
                    self._synchronize_device()
                    sync_fwd_before_ms = (time.perf_counter() - sync_start) * 1000.0
                fwd_start = time.perf_counter()
                losses, _ = self._forward_draft_batch(
                    batch=batch,
                    prediction_length=self.config.prediction_length,
                )
                loss = torch.stack(losses).mean()
                fwd_ms = (time.perf_counter() - fwd_start) * 1000.0
                if self.config.profile_sync:
                    sync_start = time.perf_counter()
                    self._synchronize_device()
                    sync_fwd_after_ms = (time.perf_counter() - sync_start) * 1000.0

                if self.config.profile_sync:
                    sync_start = time.perf_counter()
                    self._synchronize_device()
                    sync_bwd_before_ms = (time.perf_counter() - sync_start) * 1000.0
                bwd_start = time.perf_counter()
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                bwd_ms = (time.perf_counter() - bwd_start) * 1000.0
                if self.config.profile_sync:
                    sync_start = time.perf_counter()
                    self._synchronize_device()
                    sync_bwd_after_ms = (time.perf_counter() - sync_start) * 1000.0

                if self.config.profile_sync:
                    sync_start = time.perf_counter()
                    self._synchronize_device()
                    sync_opt_before_ms = (time.perf_counter() - sync_start) * 1000.0
                opt_start = time.perf_counter()
                self.optimizer.step()
                opt_ms = (time.perf_counter() - opt_start) * 1000.0
                if self.config.profile_sync:
                    sync_start = time.perf_counter()
                    self._synchronize_device()
                    sync_opt_after_ms = (time.perf_counter() - sync_start) * 1000.0

                compute_ms = fwd_ms + bwd_ms + opt_ms
                self.num_train_steps += 1
            if self.config.profile_sync:
                sync_start = time.perf_counter()
                self._synchronize_device()
                sync_after_ms = (time.perf_counter() - sync_start) * 1000.0
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            rank = self._get_rank()
            print(
                "[DraftTrainer] profile-only "
                f"rank={rank} step={self.num_train_steps} "
                f"seq_len={seq_len} elapsed_ms={elapsed_ms:.3f} "
                f"tag={tag}")
            if self.config.profile_breakdown:
                sync_total_ms = (
                    sync_before_ms + sync_fwd_before_ms + sync_fwd_after_ms +
                    sync_bwd_before_ms + sync_bwd_after_ms + sync_opt_before_ms +
                    sync_opt_after_ms + sync_after_ms)
                accounted_ms = (
                    batch_ms + clone_ms + compute_ms + sync_total_ms)
                residual_ms = max(0.0, elapsed_ms - accounted_ms)
                print(
                    "[DraftTrainer] profile-only breakdown "
                    f"rank={rank} step={self.num_train_steps} tag={tag} "
                    f"sync_before_ms={sync_before_ms:.3f} "
                    f"sync_fwd_before_ms={sync_fwd_before_ms:.3f} "
                    f"sync_fwd_after_ms={sync_fwd_after_ms:.3f} "
                    f"sync_bwd_before_ms={sync_bwd_before_ms:.3f} "
                    f"sync_bwd_after_ms={sync_bwd_after_ms:.3f} "
                    f"sync_opt_before_ms={sync_opt_before_ms:.3f} "
                    f"sync_opt_after_ms={sync_opt_after_ms:.3f} "
                    f"sync_total_ms={sync_total_ms:.3f} "
                    f"batch_ms={batch_ms:.3f} "
                    f"clone_ms={clone_ms:.3f} "
                    f"fwd_ms={fwd_ms:.3f} "
                    f"bwd_ms={bwd_ms:.3f} "
                    f"opt_ms={opt_ms:.3f} "
                    f"compute_ms={compute_ms:.3f} "
                    f"sync_after_ms={sync_after_ms:.3f} "
                    f"residual_ms={residual_ms:.3f} "
                    f"total_ms={elapsed_ms:.3f}")
            return seq_len

        for warm_step in range(warmup_steps):
            _run_one_step(tag=f"warmup-{warm_step + 1}/{warmup_steps}")

        prof_ctx, prof_enabled = self._get_profiler_context()
        with prof_ctx as prof:
            for step_idx in range(steps):
                _run_one_step(tag=f"profile-{step_idx + 1}/{steps}")
                if prof_enabled:
                    prof.step()
                    self._profile_steps_done += 1
        if self.eagle3_model is not None:
            vocab_size = int(getattr(self.eagle3_model.config, "vocab_size", -1))
            draft_vocab = int(getattr(self.eagle3_model.config, "draft_vocab_size", -1))
            lm_shape = None
            try:
                lm_shape = tuple(self.eagle3_model.lm_head.weight.shape)
            except Exception:
                lm_shape = None
            print(
                "[DraftTrainer] profile-only "
                f"device={self.device} vocab_size={vocab_size} "
                f"draft_vocab_size={draft_vocab} lm_head_shape={lm_shape}")
        print(
            "[DraftTrainer] profile-only mode: exiting process after "
            f"{steps} step(s).")
        os._exit(0)

    def _get_rank(self) -> int:
        if (torch.distributed.is_available()
                and torch.distributed.is_initialized()):
            return torch.distributed.get_rank()
        return -1

    def _synchronize_device(self) -> None:
        if self.device.type == "cpu":
            return
        if self.device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize(self.device)
            return
        npu = getattr(torch, "npu", None)
        if npu is not None and hasattr(npu, "synchronize"):
            npu.synchronize()

    def _ensure_builder_thread(self) -> None:
        if self._builder_thread is not None:
            return
        self._builder_thread = threading.Thread(
            target=self._dataset_builder_loop,
            name="draft-dataset-builder",
            daemon=True,
        )
        self._builder_thread.start()

    def _dataset_builder_loop(self) -> None:
        while True:
            event = self._event_queue.get()
            if event is None:
                return
            if isinstance(event, DraftStepBatch):
                self._append_step_batch(event)
            elif isinstance(event, DraftFinalizeReqs):
                self._finalize_req_ids(event.req_ids)

    def _ensure_train_thread(self) -> None:
        if self._train_thread is not None:
            return
        self._train_thread = threading.Thread(
            target=self._train_loop,
            name="draft-train-worker",
            daemon=True,
        )
        self._train_thread.start()

    def _train_loop(self) -> None:
        while True:
            task = self._train_queue.get()
            if task is None:
                return
            queue_wait_ms = (time.perf_counter() - task.enqueue_ts) * 1000.0
            with self._train_lock:
                try:
                    self._train_once(task.layer_idx, queue_wait_ms=queue_wait_ms)
                except BaseException as exc:
                    task.error = exc
                finally:
                    if task.done_event is not None:
                        task.done_event.set()

    def _get_profiler_context(self):
        if not _get_env_flag("VLLM_ASCEND_DRAFT_NPU_PROFILE", False):
            return contextlib.nullcontext(), None
        if self._profile_steps_done >= _coerce_int(
                os.getenv("VLLM_ASCEND_DRAFT_NPU_PROFILE_STEPS", "1"), 1):
            return contextlib.nullcontext(), None
        try:
            import torch_npu  # type: ignore
        except Exception:
            print("[DraftTrainer] torch_npu not available; skip profiling.")
            return contextlib.nullcontext(), None

        rank = self._get_rank()
        out_dir = os.getenv(
            "VLLM_ASCEND_DRAFT_NPU_PROFILE_DIR",
            f"./result/profiler/draft_rank_{rank}",
        )
        experimental_config = torch_npu.profiler._ExperimentalConfig(
            export_type=[torch_npu.profiler.ExportType.Text],
            profiler_level=torch_npu.profiler.ProfilerLevel.Level0,
            msprof_tx=False,
            aic_metrics=torch_npu.profiler.AiCMetrics.AiCoreNone,
            l2_cache=False,
            op_attr=False,
            data_simplification=False,
            record_op_args=False,
            gc_detect_threshold=None,
        )
        wait_steps = _coerce_int(
            os.getenv("VLLM_ASCEND_DRAFT_NPU_PROFILE_WAIT", "0"), 0)
        warmup_steps = _coerce_int(
            os.getenv("VLLM_ASCEND_DRAFT_NPU_PROFILE_WARMUP", "0"), 0)
        active_steps = _coerce_int(
            os.getenv("VLLM_ASCEND_DRAFT_NPU_PROFILE_ACTIVE", "1"), 1)
        repeat_steps = _coerce_int(
            os.getenv("VLLM_ASCEND_DRAFT_NPU_PROFILE_REPEAT", "1"), 1)

        ctx = torch_npu.profiler.profile(
            activities=[
                torch_npu.profiler.ProfilerActivity.CPU,
                torch_npu.profiler.ProfilerActivity.NPU,
            ],
            schedule=torch_npu.profiler.schedule(
                wait=wait_steps,
                warmup=warmup_steps,
                active=active_steps,
                repeat=repeat_steps,
            ),
            on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(out_dir),
            record_shapes=False,
            profile_memory=False,
            with_stack=False,
            with_modules=False,
            with_flops=False,
            experimental_config=experimental_config,
        )
        return ctx, True

    def _train_once(self, layer_idx: int, queue_wait_ms: float = 0.0) -> None:
        if not self.config.enabled:
            return
        if not self._initialized:
            self._lazy_init()
        if self.eagle3_model is None or self.optimizer is None:
            return

        start = time.perf_counter()
        sync_before_ms = 0.0
        if self.config.profile_sync:
            sync_start = time.perf_counter()
            self._synchronize_device()
            sync_before_ms = (time.perf_counter() - sync_start) * 1000.0
        sync_fwd_before_ms = 0.0
        sync_fwd_after_ms = 0.0
        sync_bwd_before_ms = 0.0
        sync_bwd_after_ms = 0.0
        sync_opt_before_ms = 0.0
        sync_opt_after_ms = 0.0
        prof_ctx, prof_enabled = self._get_profiler_context()
        batch_ms = 0.0
        clone_ms = 0.0
        compute_ms = 0.0
        fwd_ms = 0.0
        bwd_ms = 0.0
        opt_ms = 0.0
        seq_len = 0
        full_seq_len = 0
        micro_start = 0
        micro_end = 0
        micro_batch_end = True
        micro_valid_tokens = 0.0
        full_valid_tokens = 0.0
        opt_applied = False
        accum_before = self._grad_accum_counter
        accum_after = self._grad_accum_counter
        use_micro_seq = int(self.config.micro_seq_len) > 0
        grad_accum_steps = max(1, int(self.config.grad_accum_steps))
        with prof_ctx as prof:
            with torch.inference_mode(False), torch.enable_grad():
                batch_start = time.perf_counter()
                if use_micro_seq:
                    micro_info = self._next_micro_training_batch()
                    if micro_info is None:
                        return
                    (batch, full_seq_len, micro_start, micro_end, micro_batch_end,
                     micro_valid_tokens, full_valid_tokens) = micro_info
                    seq_len = micro_end - micro_start
                else:
                    batch = self._next_training_batch()
                    if batch is None:
                        return
                    seq_len = int(batch[1].shape[1])
                    full_seq_len = seq_len
                    micro_start = 0
                    micro_end = seq_len
                    micro_batch_end = True
                    micro_valid_tokens = float(batch[-1].sum().item())
                    full_valid_tokens = max(1.0, micro_valid_tokens)
                batch_ms = (time.perf_counter() - batch_start) * 1000.0
                # Defensive: break inference tensor propagation from dummy_run.
                clone_start = time.perf_counter()
                batch = self._clone_batch_tensors(batch)
                clone_ms = (time.perf_counter() - clone_start) * 1000.0
                if self.config.profile_sync:
                    sync_start = time.perf_counter()
                    self._synchronize_device()
                    sync_fwd_before_ms = (time.perf_counter() - sync_start) * 1000.0
                fwd_start = time.perf_counter()
                losses, _ = self._forward_draft_batch(
                    batch=batch,
                    prediction_length=self.config.prediction_length,
                )
                loss = torch.stack(losses).mean()
                if use_micro_seq:
                    if full_valid_tokens > 0.0 and micro_valid_tokens > 0.0:
                        loss = loss * (micro_valid_tokens / full_valid_tokens)
                    else:
                        loss = loss * 0.0
                fwd_ms = (time.perf_counter() - fwd_start) * 1000.0
                if self.config.profile_sync:
                    sync_start = time.perf_counter()
                    self._synchronize_device()
                    sync_fwd_after_ms = (time.perf_counter() - sync_start) * 1000.0

                if self.config.profile_sync:
                    sync_start = time.perf_counter()
                    self._synchronize_device()
                    sync_bwd_before_ms = (time.perf_counter() - sync_start) * 1000.0
                bwd_start = time.perf_counter()
                if use_micro_seq:
                    if self._grad_accum_counter == 0:
                        self.optimizer.zero_grad(set_to_none=True)
                    if micro_valid_tokens > 0.0:
                        loss.backward()
                        self._grad_accum_counter += 1
                else:
                    self.optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                bwd_ms = (time.perf_counter() - bwd_start) * 1000.0
                if self.config.profile_sync:
                    sync_start = time.perf_counter()
                    self._synchronize_device()
                    sync_bwd_after_ms = (time.perf_counter() - sync_start) * 1000.0

                if use_micro_seq:
                    should_step = (
                        self._grad_accum_counter >= grad_accum_steps
                        or (micro_batch_end and self._grad_accum_counter > 0))
                else:
                    should_step = True

                if should_step:
                    if self.config.profile_sync:
                        sync_start = time.perf_counter()
                        self._synchronize_device()
                        sync_opt_before_ms = (time.perf_counter() - sync_start) * 1000.0
                    opt_start = time.perf_counter()
                    self.optimizer.step()
                    self.optimizer.zero_grad(set_to_none=True)
                    opt_ms = (time.perf_counter() - opt_start) * 1000.0
                    if self.config.profile_sync:
                        sync_start = time.perf_counter()
                        self._synchronize_device()
                        sync_opt_after_ms = (time.perf_counter() - sync_start) * 1000.0
                    opt_applied = True
                    self.num_train_steps += 1
                    if use_micro_seq:
                        self._grad_accum_counter = 0

                compute_ms = fwd_ms + bwd_ms + opt_ms
                accum_after = self._grad_accum_counter
            if prof_enabled:
                prof.step()
                self._profile_steps_done += 1

        sync_after_ms = 0.0
        if self.config.profile_sync:
            sync_start = time.perf_counter()
            self._synchronize_device()
            sync_after_ms = (time.perf_counter() - sync_start) * 1000.0
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        rank = self._get_rank()
        print(
            "[DraftTrainer] "
            f"rank={rank} step={self.num_train_steps} layer={layer_idx} "
            f"seq_len={seq_len} "
            f"elapsed_ms={elapsed_ms:.3f}")
        if self.config.profile_breakdown:
            sync_total_ms = (
                sync_before_ms + sync_fwd_before_ms + sync_fwd_after_ms +
                sync_bwd_before_ms + sync_bwd_after_ms + sync_opt_before_ms +
                sync_opt_after_ms + sync_after_ms)
            accounted_ms = (
                batch_ms + clone_ms + compute_ms + sync_total_ms)
            residual_ms = max(0.0, elapsed_ms - accounted_ms)
            print(
                "[DraftTrainer] breakdown "
                f"rank={rank} step={self.num_train_steps} layer={layer_idx} "
                f"queue_wait_ms={queue_wait_ms:.3f} "
                f"sync_before_ms={sync_before_ms:.3f} "
                f"sync_fwd_before_ms={sync_fwd_before_ms:.3f} "
                f"sync_fwd_after_ms={sync_fwd_after_ms:.3f} "
                f"sync_bwd_before_ms={sync_bwd_before_ms:.3f} "
                f"sync_bwd_after_ms={sync_bwd_after_ms:.3f} "
                f"sync_opt_before_ms={sync_opt_before_ms:.3f} "
                f"sync_opt_after_ms={sync_opt_after_ms:.3f} "
                f"sync_total_ms={sync_total_ms:.3f} "
                f"batch_ms={batch_ms:.3f} "
                f"clone_ms={clone_ms:.3f} "
                f"fwd_ms={fwd_ms:.3f} "
                f"bwd_ms={bwd_ms:.3f} "
                f"opt_ms={opt_ms:.3f} "
                f"compute_ms={compute_ms:.3f} "
                f"sync_after_ms={sync_after_ms:.3f} "
                f"residual_ms={residual_ms:.3f} "
                f"micro_range={micro_start}:{micro_end}/{full_seq_len} "
                f"micro_valid={micro_valid_tokens:.1f} "
                f"full_valid={full_valid_tokens:.1f} "
                f"accum={accum_before}->{accum_after}/{grad_accum_steps} "
                f"opt_applied={int(opt_applied)} "
                f"total_ms={elapsed_ms:.3f}")

        if self.config.step_budget_ms > 0:
            if elapsed_ms > self.config.step_budget_ms:
                self.num_over_budget += 1
        if self.config.profile_only:
            print(
                "[DraftTrainer] profile-only mode: disabling further training.")
            self.config.enabled = False

    def _append_step_batch(self, batch: DraftStepBatch) -> None:
        query_start_loc = batch.query_start_loc
        num_reqs = len(batch.req_ids)
        if num_reqs == 0:
            return
        total_tokens = int(query_start_loc[num_reqs].item())
        if total_tokens <= 0:
            return

        input_ids = batch.input_ids[:total_tokens]
        positions = batch.positions[:total_tokens]
        hidden_states = batch.hidden_states

        if input_ids.ndim > 1:
            input_ids = input_ids.reshape(-1)
        if positions.ndim > 1:
            positions = positions.reshape(-1)
        if hidden_states.ndim == 3:
            hidden_states = hidden_states.reshape(-1, hidden_states.shape[-1])

        for i, req_id in enumerate(batch.req_ids):
            start = int(query_start_loc[i].item())
            end = int(query_start_loc[i + 1].item())
            if end <= start:
                continue
            ids_slice = input_ids[start:end].clone()
            pos_slice = positions[start:end].clone()
            hs_slice = hidden_states[start:end].clone().detach()
            seg_len = end - start
            resp_start = 0
            if batch.response_start_locs is not None and i < int(
                    batch.response_start_locs.numel()):
                resp_start = int(batch.response_start_locs[i].item())
            resp_start = max(0, min(resp_start, seg_len))
            resp_mask = torch.zeros(seg_len, dtype=torch.float32)
            if resp_start < seg_len:
                resp_mask[resp_start:] = 1.0
            buf = self._req_buffers.get(req_id)
            if buf is None:
                buf = DraftSeqBuffer(input_ids=[],
                                     positions=[],
                                     hidden_states=[],
                                     response_masks=[])
                self._req_buffers[req_id] = buf
            buf.input_ids.append(ids_slice)
            buf.positions.append(pos_slice)
            buf.hidden_states.append(hs_slice)
            buf.response_masks.append(resp_mask)

    def _dump_sample(self, req_id: str, sample: DraftSample) -> None:
        if not self._dump_enabled:
            return
        self._dump_seen += 1
        if (self._dump_seen - 1) % self._dump_every != 0:
            return
        rank = self._get_rank()
        hidden_states = sample.hidden_states.to(
            dtype=self._dump_hidden_dtype).cpu()
        if sample.response_mask is not None:
            loss_mask = sample.response_mask.to(dtype=torch.float32).clone()
            if loss_mask.ndim > 1:
                loss_mask = loss_mask.reshape(-1)
        else:
            loss_mask = torch.ones_like(sample.input_ids,
                                        dtype=torch.float32)
        if loss_mask.numel() > 0:
            loss_mask[-1] = 0.0
        payload = {
            "req_id": req_id,
            "rank": rank,
            "seq_len": int(sample.input_ids.shape[0]),
            "input_ids": sample.input_ids.cpu(),
            "positions": (sample.positions.cpu()
                          if sample.positions is not None else None),
            "hidden_states": hidden_states,
            "response_mask": (sample.response_mask.cpu()
                              if sample.response_mask is not None else None),
            "loss_mask": loss_mask.cpu(),
        }
        file_name = f"rank_{rank}_sample_{self._dump_seen:012d}.pt"
        path = os.path.join(self._dump_dir, file_name)
        try:
            torch.save(payload, path)
            self._dump_saved += 1
            if self._dump_saved % 100 == 0:
                print(
                    "[DraftTrainer] data dump progress "
                    f"rank={rank} saved={self._dump_saved} "
                    f"seen={self._dump_seen} dir={self._dump_dir}")
        except Exception as exc:
            print(
                "[DraftTrainer] data dump failed "
                f"path={path} err={exc}")

    def _finalize_req_ids(self, req_ids: list[str]) -> None:
        for req_id in req_ids:
            buf = self._req_buffers.pop(req_id, None)
            if buf is None or not buf.input_ids or not buf.hidden_states:
                self.num_samples_empty += 1
                continue
            input_ids = torch.cat(buf.input_ids, dim=0)
            positions = torch.cat(buf.positions,
                                  dim=0) if buf.positions else None
            hidden_states = torch.cat(buf.hidden_states, dim=0)
            response_mask = (torch.cat(buf.response_masks, dim=0)
                             if buf.response_masks else None)
            sample = DraftSample(
                input_ids=input_ids,
                positions=positions,
                hidden_states=hidden_states,
                response_mask=response_mask,
            )
            self._dump_sample(req_id, sample)
            with self._sample_pool_lock:
                self._sample_pool.append(sample)
            if self.config.enabled:
                try:
                    self._ready_queue.put_nowait(sample)
                    self.num_samples_built += 1
                except queue.Full:
                    self.num_samples_dropped += 1
            else:
                self.num_samples_built += 1

    def record_step_batch(
        self,
        req_ids: list[str],
        query_start_loc_cpu: torch.Tensor,
        input_ids_cpu: torch.Tensor,
        positions_cpu: torch.Tensor,
        hidden_states: torch.Tensor,
        response_start_locs_cpu: Optional[torch.Tensor] = None,
    ) -> None:
        if not self._capture_enabled:
            return
        if not req_ids:
            return
        self._ensure_builder_thread()
        try:
            self._event_queue.put_nowait(
                DraftStepBatch(
                    req_ids=list(req_ids),
                    query_start_loc=query_start_loc_cpu.detach().clone(),
                    input_ids=input_ids_cpu.detach().clone(),
                    positions=positions_cpu.detach().clone(),
                    hidden_states=hidden_states.detach(),
                    response_start_locs=(None if response_start_locs_cpu is None
                                         else response_start_locs_cpu.detach()
                                         .clone()),
                ))
            self.num_samples_enqueued += 1
        except queue.Full:
            self.num_samples_dropped += 1

    def finalize_requests(self, req_ids: list[str]) -> None:
        if not self._capture_enabled:
            return
        if not req_ids:
            return
        self._ensure_builder_thread()
        try:
            self._event_queue.put_nowait(DraftFinalizeReqs(req_ids=list(req_ids)))
        except queue.Full:
            self.num_samples_dropped += 1

    def _build_eagle3_config(self, main_model: nn.Module) -> Qwen3Config:
        base_cfg = getattr(main_model, "config", None)
        if base_cfg is None:
            raise RuntimeError("Draft trainer requires main model config.")

        vocab_size = _coerce_int(getattr(base_cfg, "vocab_size", None), 151936)
        hidden_size = _coerce_int(getattr(base_cfg, "hidden_size", None), 4096)
        intermediate_size = _coerce_int(
            getattr(base_cfg, "intermediate_size", None), hidden_size * 4)
        num_heads = _coerce_int(getattr(base_cfg, "num_attention_heads", None),
                                32)
        num_kv_heads = _coerce_int(
            getattr(base_cfg, "num_key_value_heads", None), num_heads)
        max_position_embeddings = _coerce_int(
            getattr(base_cfg, "max_position_embeddings", None), 2048)
        rms_norm_eps = _coerce_float(getattr(base_cfg, "rms_norm_eps", None),
                                     1e-6)
        pad_token_id = _coerce_int(getattr(base_cfg, "pad_token_id", None), 0)
        attention_bias = bool(getattr(base_cfg, "attention_bias", False))
        attn_impl = getattr(base_cfg, "_attn_implementation", None) or getattr(
            base_cfg, "attn_implementation", None)
        if attn_impl not in {"eager", "sdpa", "flash_attention_2"}:
            attn_impl = "eager"
        override_attn_impl = (self.config.attn_impl or "").strip()
        if override_attn_impl in {"eager", "sdpa", "flash_attention_2"}:
            attn_impl = override_attn_impl

        cfg = Qwen3Config(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_hidden_layers=1,
            num_attention_heads=num_heads,
            num_key_value_heads=num_kv_heads,
            max_position_embeddings=max_position_embeddings,
            rms_norm_eps=rms_norm_eps,
            pad_token_id=pad_token_id,
            attention_bias=attention_bias,
            use_cache=False,
            output_attentions=False,
            output_hidden_states=False,
        )
        # Some transformers versions expose `use_return_dict` as a read-only
        # property, so we set `return_dict` defensively after init.
        try:
            cfg.return_dict = False
        except (AttributeError, TypeError):
            pass
        cfg._attn_implementation = attn_impl
        cfg.draft_attn_chunk_size = max(0, int(self.config.attn_chunk_size))
        cfg.target_hidden_size = hidden_size
        if self.config.reuse_target_emb_lm:
            cfg.draft_vocab_size = vocab_size
        else:
            cfg.draft_vocab_size = min(self.config.draft_vocab_size, vocab_size)
        cfg.draft_sparse_kl_enabled = bool(self.config.sparse_kl_enabled)
        cfg.draft_sparse_kl_topk = max(1, int(self.config.sparse_kl_topk))
        cfg.draft_vloss_weight = float(
            os.getenv("VLLM_ASCEND_DRAFT_VLOSS_WEIGHT", "0.5"))
        cfg.draft_ploss_weight = float(
            os.getenv("VLLM_ASCEND_DRAFT_PLOSS_WEIGHT", "0.5"))
        cfg.draft_compute_accuracy = _get_env_flag(
            "VLLM_ASCEND_DRAFT_COMPUTE_ACCURACY", False)
        return cfg

    def _next_training_batch(self) -> Optional[Tuple[torch.Tensor, ...]]:
        if self.config.fastrl_concat_enabled:
            batch = self._next_fastrl_concat_batch()
            if batch is not None:
                return batch
        try:
            sample = self._ready_queue.get_nowait()
            self._latest_sample = sample
        except queue.Empty:
            sample = self._latest_sample
        if sample is None:
            return None
        return self._sample_to_training_batch(sample)

    def _next_fastrl_concat_batch(self) -> Optional[Tuple[torch.Tensor, ...]]:
        with self._sample_pool_lock:
            available = list(self._sample_pool)
        if not available:
            return None

        effective_bs = max(1, min(int(self.config.fastrl_concat_batch_size), 4))
        min_required = min(2, max(1, effective_bs // 2))
        if len(available) >= effective_bs:
            items = random.sample(available, effective_bs)
        elif len(available) >= min_required:
            items = available
        else:
            return None

        input_ids_list: list[torch.Tensor] = []
        positions_list: list[torch.Tensor] = []
        hidden_states_list: list[torch.Tensor] = []
        response_mask_list: list[torch.Tensor] = []
        for sample in items:
            seq_len = int(sample.input_ids.numel())
            seq_len = min(seq_len, int(sample.hidden_states.shape[0]))
            if sample.positions is not None:
                seq_len = min(seq_len, int(sample.positions.numel()))
            seq_len = min(seq_len, int(self.config.max_seq_len))
            if seq_len <= 0:
                continue

            input_ids = sample.input_ids[:seq_len]
            hidden_states = sample.hidden_states[:seq_len]
            if sample.positions is not None:
                positions = sample.positions[:seq_len]
            else:
                positions = torch.arange(seq_len, dtype=torch.long)
            if sample.response_mask is not None:
                response_mask = sample.response_mask[:seq_len].float()
                if response_mask.ndim > 1:
                    response_mask = response_mask.reshape(-1)
            else:
                response_mask = torch.ones(seq_len, dtype=torch.float32)

            max_len = min(seq_len, int(self.config.fastrl_concat_window_len))
            nonzero = torch.nonzero(response_mask).flatten()
            if nonzero.numel() > 0:
                resp_start_idx = int(nonzero[0].item())
                resp_end_idx = int(nonzero[-1].item()) + 1
                start = max(0, min(resp_start_idx, seq_len - max_len))
                if resp_end_idx - start > max_len:
                    start = resp_end_idx - max_len
                end = min(seq_len, start + max_len)
            else:
                start = max(0, seq_len - max_len)
                end = seq_len
            if end <= start:
                continue

            input_ids_list.append(input_ids[start:end])
            positions_list.append(positions[start:end])
            hidden_states_list.append(hidden_states[start:end])
            response_mask_list.append(response_mask[start:end])

        if not input_ids_list:
            return None
        input_ids = torch.cat(input_ids_list, dim=0)
        positions = torch.cat(positions_list, dim=0)
        hidden_states = torch.cat(hidden_states_list, dim=0)
        response_mask = torch.cat(response_mask_list, dim=0)
        return self._build_training_batch(input_ids=input_ids,
                                          hidden_states=hidden_states,
                                          positions=positions,
                                          response_mask=response_mask)

    def _reset_active_train_batch(self) -> None:
        self._active_train_batch = None
        self._active_train_seq_len = 0
        self._active_train_cursor = 0
        self._active_train_valid_tokens = 1.0

    def _slice_training_batch(
        self,
        batch: Tuple[torch.Tensor, ...],
        start: int,
        end: int,
    ) -> Tuple[torch.Tensor, ...]:
        if len(batch) == 5:
            return (
                batch[0][:, start:end, :],
                batch[1][:, start:end],
                batch[2][:, start:end],
                batch[3][:, start:end, :],
                batch[4][:, start:end, :],
            )
        if len(batch) == 6:
            return (
                batch[0][:, start:end, :],
                batch[1][:, start:end],
                batch[2][:, start:end],
                batch[3][:, start:end, :],
                batch[4][:, start:end, :],
                batch[5][:, start:end, :],
            )
        raise RuntimeError(f"Unexpected draft batch format: len={len(batch)}")

    def _next_micro_training_batch(
        self,
    ) -> Optional[tuple[Tuple[torch.Tensor, ...], int, int, int, bool, float, float]]:
        micro_seq_len = int(self.config.micro_seq_len)
        if micro_seq_len <= 0:
            return None

        while True:
            if (self._active_train_batch is None
                    or self._active_train_cursor >= self._active_train_seq_len):
                full_batch = self._next_training_batch()
                if full_batch is None:
                    self._reset_active_train_batch()
                    return None
                full_seq_len = int(full_batch[1].shape[1])
                if full_seq_len <= 0:
                    continue
                self._active_train_batch = full_batch
                self._active_train_seq_len = full_seq_len
                self._active_train_cursor = 0
                loss_mask = full_batch[-1]
                self._active_train_valid_tokens = max(
                    1.0, float(loss_mask.sum().item()))

            start = self._active_train_cursor
            end = min(start + micro_seq_len, self._active_train_seq_len)
            micro_batch = self._slice_training_batch(self._active_train_batch,
                                                     start, end)
            self._active_train_cursor = end
            batch_done = (self._active_train_cursor >= self._active_train_seq_len)
            micro_valid_tokens = float(micro_batch[-1].sum().item())
            full_valid_tokens = float(self._active_train_valid_tokens)
            if batch_done:
                self._reset_active_train_batch()
            return (micro_batch, self._active_train_seq_len, start, end,
                    batch_done, micro_valid_tokens, full_valid_tokens)

    def _sample_to_training_batch(
        self,
        sample: DraftSample,
    ) -> Optional[Tuple[torch.Tensor, ...]]:
        input_ids = sample.input_ids
        hidden_states = sample.hidden_states
        positions = sample.positions
        response_mask = sample.response_mask

        seq_len = int(input_ids.numel())
        seq_len = min(seq_len, int(hidden_states.shape[0]))
        if positions is not None:
            seq_len = min(seq_len, int(positions.numel()))
        if seq_len <= 0:
            return None
        return self._build_training_batch(input_ids=input_ids[:seq_len],
                                          hidden_states=hidden_states[:seq_len],
                                          positions=(positions[:seq_len]
                                                     if positions is not None else None),
                                          response_mask=(response_mask[:seq_len]
                                                         if response_mask is not None else None))

    def _build_training_batch(
        self,
        input_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        positions: Optional[torch.Tensor],
        response_mask: Optional[torch.Tensor],
    ) -> Optional[Tuple[torch.Tensor, ...]]:
        if self.eagle3_model is None:
            return None
        cfg = self.eagle3_model.config
        vocab_size = int(cfg.vocab_size)
        target_hidden_size = int(getattr(cfg, "target_hidden_size", cfg.hidden_size))

        seq_len = int(input_ids.numel())
        seq_len = min(seq_len, int(hidden_states.shape[0]))
        if positions is not None:
            seq_len = min(seq_len, int(positions.numel()))
        seq_len = min(seq_len, int(self.config.max_seq_len))
        if seq_len <= 0:
            return None

        input_ids = input_ids[:seq_len].to(self.device, dtype=torch.long)
        input_ids = input_ids.clamp(0, vocab_size - 1)
        input_ids = input_ids.unsqueeze(0)

        hidden_states = hidden_states[:seq_len].to(self.device,
                                                   dtype=self.train_dtype)
        hidden_states = hidden_states.clone().unsqueeze(0)

        if response_mask is not None:
            response_mask = response_mask[:seq_len]
            if response_mask.ndim > 1:
                response_mask = response_mask.reshape(-1)
            response_mask = response_mask.to(self.device, dtype=torch.float32)
        else:
            response_mask = torch.ones(seq_len,
                                       device=self.device,
                                       dtype=torch.float32)

        if hidden_states.shape[-1] == target_hidden_size:
            target_hidden_states = hidden_states
        elif hidden_states.shape[-1] == target_hidden_size * 3:
            target_hidden_states = hidden_states[..., :target_hidden_size]
        else:
            if hidden_states.shape[-1] > target_hidden_size:
                target_hidden_states = hidden_states[..., :target_hidden_size]
            else:
                pad = target_hidden_size - hidden_states.shape[-1]
                target_hidden_states = F.pad(hidden_states, (0, pad))
        base_model_hidden_states = torch.cat([
            target_hidden_states, target_hidden_states, target_hidden_states
        ],
                                             dim=-1)

        if positions is not None:
            position_ids = positions[:seq_len].to(self.device,
                                                  dtype=torch.long)
            position_ids = position_ids.unsqueeze(0)
        else:
            position_ids = torch.arange(seq_len, device=self.device,
                                        dtype=torch.long).unsqueeze(0)

        target_ids = input_ids.clone()
        if seq_len > 1:
            target_ids[:, :-1] = input_ids[:, 1:]
        loss_mask = torch.zeros(1,
                                seq_len,
                                1,
                                device=self.device,
                                dtype=torch.float32)
        if seq_len > 1:
            loss_mask[:, :-1, 0] = response_mask[1:]

        sparse_enabled = bool(getattr(cfg, "draft_sparse_kl_enabled", True))
        topk = max(1, int(getattr(cfg, "draft_sparse_kl_topk", 64)))
        if sparse_enabled:
            target_topk_idx = torch.zeros(
                1, seq_len, topk, device=self.device, dtype=torch.long)
            target_topk_vals = torch.full(
                (1, seq_len, topk),
                -1e9,
                device=self.device,
                dtype=torch.float32,
            )
            target_topk_idx[..., 0] = target_ids
            target_topk_vals[..., 0] = 0.0
            return (
                base_model_hidden_states,
                input_ids,
                position_ids,
                target_topk_idx,
                target_topk_vals,
                loss_mask,
            )

        target_hidden = target_hidden_states.clone()
        if seq_len > 1:
            target_hidden[:, :-1, :] = target_hidden_states[:, 1:, :]
        return (base_model_hidden_states, input_ids, position_ids,
                target_hidden, loss_mask)

    def _lazy_init(self) -> None:
        if self._initialized:
            return
        main_model = self._model_ref()
        if main_model is None:
            raise RuntimeError("Main model is released before draft init.")

        param = next(main_model.parameters(), None)
        if param is not None:
            self.device = param.device
        self.train_dtype = self._resolve_train_dtype()

        eagle3_cfg = self._build_eagle3_config(main_model)
        print(
            "[DraftTrainer] config "
            f"vocab_size={int(getattr(eagle3_cfg, 'vocab_size', -1))} "
            f"max_seq_len={self.config.max_seq_len} "
            f"attn_impl={getattr(eagle3_cfg, '_attn_implementation', 'unknown')} "
            f"attn_chunk_size={int(getattr(eagle3_cfg, 'draft_attn_chunk_size', 0))} "
            f"micro_seq_len={int(self.config.micro_seq_len)} "
            f"grad_accum_steps={int(self.config.grad_accum_steps)}")
        self.eagle3_model = Qwen3ModelEagle3(eagle3_cfg).to(self.device,
                                                            dtype=self.train_dtype)
        self.eagle3_model.train()
        self._maybe_enable_lora()
        self._attach_target_layers()
        self._configure_trainable_params()
        self.optimizer = AdamW(
            [p for p in self.eagle3_model.parameters() if p.requires_grad],
            lr=self.config.lr,
        )
        self._initialized = True

    def _freeze_module(self, module: Optional[nn.Module]) -> None:
        if module is None:
            return
        for param in module.parameters(recurse=True):
            param.requires_grad = False

    def _attach_target_layers(self) -> None:
        if self.eagle3_model is None:
            return
        if self.config.reuse_target_emb_lm:
            if self._target_embed_tokens is not None:
                self.eagle3_model.embed_tokens = self._target_embed_tokens
            if (self._target_lm_head is not None
                    and hasattr(self._target_lm_head, "weight")):
                target_weight = self._target_lm_head.weight
                if (hasattr(self.eagle3_model.lm_head, "weight")
                        and self.eagle3_model.lm_head.weight.shape
                        == target_weight.shape):
                    self.eagle3_model.lm_head.weight = target_weight
                else:
                    print(
                        "[DraftTrainer] Skip LM head reuse due to shape mismatch: "
                        f"{self.eagle3_model.lm_head.weight.shape} vs "
                        f"{target_weight.shape}")
        # Freeze embedding + LM head; only train mid layers.
        self._freeze_module(self.eagle3_model.embed_tokens)
        self._freeze_module(self.eagle3_model.lm_head)

    def _maybe_enable_lora(self) -> None:
        if not self._lora_enabled or self.eagle3_model is None or self._lora_applied:
            return
        backend = self._lora_backend
        replaced = 0
        if backend == "peft":
            if LoraConfig is None or get_peft_model is None:
                raise RuntimeError(
                    "LoRA backend=peft but `peft` is not available in this environment.")
            lora_cfg = LoraConfig(
                r=self._lora_rank,
                lora_alpha=int(self._lora_alpha),
                target_modules=sorted(self._lora_target_modules),
                lora_dropout=float(self._lora_dropout),
                bias="none",
            )
            self.eagle3_model = get_peft_model(self.eagle3_model, lora_cfg)
            for _, module in self.eagle3_model.named_modules():
                if hasattr(module, "lora_A") and hasattr(module, "lora_B"):
                    replaced += 1
        elif backend == "custom":
            replaced = apply_lora_to_linear_modules(
                self.eagle3_model,
                rank=self._lora_rank,
                alpha=self._lora_alpha,
                dropout=self._lora_dropout,
                target_module_names=self._lora_target_modules,
            )
        else:
            raise RuntimeError(
                f"Unknown VLLM_ASCEND_DRAFT_LORA_BACKEND={backend}, expected peft/custom.")
        self._lora_replaced_linear = replaced
        self._lora_applied = True
        print(
            "[DraftTrainer] LoRA "
            f"enabled={self._lora_enabled} rank={self._lora_rank} "
            f"alpha={self._lora_alpha:.2f} dropout={self._lora_dropout:.3f} "
            f"backend={backend} targets={sorted(self._lora_target_modules)} "
            f"replaced_linear={replaced}"
        )
        self._log_lora_matrix_shapes()

    def _log_lora_matrix_shapes(self) -> None:
        if self.eagle3_model is None:
            return
        printed = 0
        for name, module in self.eagle3_model.named_modules():
            if isinstance(module, LoRALinear):
                base_w_shape = tuple(module.base_linear.weight.shape)
                a_shape = tuple(module.lora_a.weight.shape)
                b_shape = tuple(module.lora_b.weight.shape)
                print(
                    "[DraftTrainer] LoRA matrix "
                    f"module={name} base_weight={base_w_shape} "
                    f"lora_a={a_shape} lora_b={b_shape}")
                printed += 1
                continue
            if hasattr(module, "lora_A") and hasattr(module, "lora_B"):
                base_w_shape = None
                if hasattr(module, "base_layer") and hasattr(
                        module.base_layer, "weight"):
                    base_w_shape = tuple(module.base_layer.weight.shape)
                lora_a = getattr(module, "lora_A", {})
                lora_b = getattr(module, "lora_B", {})
                adapter_name = "default"
                if hasattr(lora_a, "keys"):
                    keys = list(lora_a.keys())
                    if keys:
                        adapter_name = keys[0]
                a_shape = None
                b_shape = None
                try:
                    a_mod = lora_a[adapter_name]
                    if hasattr(a_mod, "weight"):
                        a_shape = tuple(a_mod.weight.shape)
                except Exception:
                    pass
                try:
                    b_mod = lora_b[adapter_name]
                    if hasattr(b_mod, "weight"):
                        b_shape = tuple(b_mod.weight.shape)
                except Exception:
                    pass
                print(
                    "[DraftTrainer] LoRA matrix "
                    f"module={name} adapter={adapter_name} "
                    f"base_weight={base_w_shape} lora_a={a_shape} lora_b={b_shape}")
                printed += 1
        if printed == 0:
            print("[DraftTrainer] LoRA matrix: no LoRA modules found.")

    def _configure_trainable_params(self) -> None:
        if self.eagle3_model is None:
            return
        if not self._lora_enabled or self._lora_replaced_linear <= 0:
            return
        for param in self.eagle3_model.parameters():
            param.requires_grad = False
        for name, param in self.eagle3_model.named_parameters():
            lower_name = name.lower()
            if ".lora_a." in lower_name or ".lora_b." in lower_name:
                param.requires_grad = True

    def _random_training_batch(self, seq_len: int) -> Tuple[torch.Tensor, ...]:
        assert self.eagle3_model is not None
        cfg = self.eagle3_model.config
        batch = 1
        seq_len = max(1, min(seq_len, self.config.max_seq_len))

        base_model_hidden_states = torch.randn(
            batch,
            seq_len,
            int(cfg.target_hidden_size) * 3,
            device=self.device,
            dtype=self.train_dtype,
        )
        input_ids = torch.randint(
            low=0,
            high=int(cfg.vocab_size),
            size=(batch, seq_len),
            device=self.device,
            dtype=torch.long,
        )
        position_ids = torch.arange(seq_len,
                                    device=self.device,
                                    dtype=torch.long).unsqueeze(0)
        loss_mask = torch.ones(batch,
                               seq_len,
                               1,
                               device=self.device,
                               dtype=torch.float32)
        sparse_enabled = bool(getattr(cfg, "draft_sparse_kl_enabled", True))
        topk = max(1, int(getattr(cfg, "draft_sparse_kl_topk", 64)))
        if sparse_enabled:
            target_topk_idx = torch.randint(
                low=0,
                high=int(cfg.vocab_size),
                size=(batch, seq_len, topk),
                device=self.device,
                dtype=torch.long,
            )
            target_topk_vals = torch.randn(
                batch,
                seq_len,
                topk,
                device=self.device,
                dtype=torch.float32,
            )
            return (
                base_model_hidden_states,
                input_ids,
                position_ids,
                target_topk_idx,
                target_topk_vals,
                loss_mask,
            )

        target = torch.randn(
            batch,
            seq_len,
            int(cfg.target_hidden_size),
            device=self.device,
            dtype=self.train_dtype,
        )
        return base_model_hidden_states, input_ids, position_ids, target, loss_mask

    def _run_warmup_steps(self, steps: int, tag_prefix: str) -> None:
        if self.eagle3_model is None or self.optimizer is None:
            return
        steps = max(1, steps)
        for warm_step in range(steps):
            sync_before_ms = 0.0
            sync_after_ms = 0.0
            batch_ms = 0.0
            compute_ms = 0.0
            if self.config.profile_sync:
                sync_start = time.perf_counter()
                self._synchronize_device()
                sync_before_ms = (time.perf_counter() - sync_start) * 1000.0
            start = time.perf_counter()
            with torch.inference_mode(False), torch.enable_grad():
                batch_start = time.perf_counter()
                batch = self._next_training_batch()
                if batch is None:
                    seq_len = max(4, self.config.max_seq_len)
                    batch = self._random_training_batch(seq_len=seq_len)
                batch_ms = (time.perf_counter() - batch_start) * 1000.0
                losses, _ = self._forward_draft_batch(
                    batch=batch,
                    prediction_length=1,
                )
                loss = torch.stack(losses).mean()
                compute_start = time.perf_counter()
                self.optimizer.zero_grad(set_to_none=True)  # type: ignore[arg-type]
                loss.backward()
                self.optimizer.step()  # type: ignore[union-attr]
                self.optimizer.zero_grad(set_to_none=True)  # type: ignore[arg-type]
                compute_ms = (time.perf_counter() - compute_start) * 1000.0
            if self.config.profile_sync:
                sync_start = time.perf_counter()
                self._synchronize_device()
                sync_after_ms = (time.perf_counter() - sync_start) * 1000.0
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            rank = self._get_rank()
            print(
                "[DraftTrainer] Warmup completed "
                f"rank={rank} step={warm_step + 1}/{steps} "
                f"elapsed_ms={elapsed_ms:.3f} tag={tag_prefix}")
            if self.config.profile_breakdown:
                print(
                    "[DraftTrainer] warmup_breakdown "
                    f"rank={rank} step={warm_step + 1}/{steps} "
                    f"sync_before_ms={sync_before_ms:.3f} "
                    f"batch_ms={batch_ms:.3f} "
                    f"compute_ms={compute_ms:.3f} "
                    f"sync_after_ms={sync_after_ms:.3f} "
                    f"total_ms={elapsed_ms:.3f} tag={tag_prefix}")

    def _startup_warmup(self) -> None:
        if self._startup_warmup_done: 
            return
        if not self._initialized:
            self._lazy_init()
        if self.eagle3_model is None or self.optimizer is None:
            return
        steps = max(1, self.config.startup_warmup_steps)
        self._run_warmup_steps(steps, "startup")
        self._startup_warmup_done = True

    def warmup(self) -> None:
        self._lazy_init()
        if self.eagle3_model is None or self.optimizer is None:
            return
        steps = _coerce_int(
            os.getenv("VLLM_ASCEND_DRAFT_WARMUP_STEPS", "1"), 1)
        self._run_warmup_steps(steps, "warmup")
        self._warmup_done = True

    def maybe_train_step(
        self,
        layer_idx: int,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
    ) -> None:
        del hidden_states, positions
        self.num_calls += 1
        if not self.config.enabled:
            return
        if self.config.profile_only:
            if self._profile_only_done:
                return
            self._profile_only_done = True
        if torch.is_inference_mode_enabled():
            self._ensure_train_thread()
            if self.config.async_train:
                try:
                    self._train_queue.put_nowait(
                        DraftTrainTask(layer_idx=layer_idx,
                                       enqueue_ts=time.perf_counter()))
                except queue.Full:
                    return
            else:
                task = DraftTrainTask(layer_idx=layer_idx,
                                      enqueue_ts=time.perf_counter(),
                                      done_event=threading.Event())
                self._train_queue.put(task)
                task.done_event.wait()
                if task.error is not None:
                    raise task.error
        else:
            self._train_once(layer_idx, queue_wait_ms=0.0)


def build_draft_trainer(model: nn.Module) -> Optional[DraftTrainer]:
    config = DraftTrainerConfig(
        enabled=_get_env_flag("VLLM_ASCEND_ENABLE_DRAFT_TRAIN", False),
        warmup_on_init=_get_env_flag("VLLM_ASCEND_DRAFT_WARMUP_ON_INIT", True),
        startup_warmup_steps=max(
            0, int(os.getenv("VLLM_ASCEND_DRAFT_STARTUP_WARMUP_STEPS", "0"))),
        step_budget_ms=float(os.getenv("VLLM_ASCEND_DRAFT_STEP_BUDGET_MS",
                                       "0")),
        lr=float(os.getenv("VLLM_ASCEND_DRAFT_LR", "1e-4")),
        prediction_length=max(
            1, int(os.getenv("VLLM_ASCEND_DRAFT_PREDICTION_LEN", "1"))),
        max_seq_len=max(1, int(os.getenv("VLLM_ASCEND_DRAFT_MAX_SEQ_LEN",
                                         "16"))),
        draft_vocab_size=max(
            2, int(os.getenv("VLLM_ASCEND_DRAFT_VOCAB_SIZE", "4096"))),
        queue_size=max(
            1, int(os.getenv("VLLM_ASCEND_DRAFT_QUEUE_SIZE", "4"))),
        async_train=_get_env_flag("VLLM_ASCEND_DRAFT_ASYNC_TRAIN", True),
        profile_sync=_get_env_flag("VLLM_ASCEND_DRAFT_PROFILE_SYNC", False),
        profile_breakdown=_get_env_flag(
            "VLLM_ASCEND_DRAFT_PROFILE_BREAKDOWN", False),
        reuse_target_emb_lm=_get_env_flag(
            "VLLM_ASCEND_DRAFT_REUSE_TARGET_EMB_LM", True),
        profile_only=_get_env_flag(
            "VLLM_ASCEND_DRAFT_PROFILE_ONLY", False),
        attn_impl=os.getenv("VLLM_ASCEND_DRAFT_ATTN_IMPL", "").strip(),
        attn_chunk_size=max(
            0, int(os.getenv("VLLM_ASCEND_DRAFT_ATTN_CHUNK_SIZE", "0"))),
        sparse_kl_enabled=_get_env_flag(
            "VLLM_ASCEND_DRAFT_SPARSE_KL_ENABLE", True),
        sparse_kl_topk=max(
            1, int(os.getenv("VLLM_ASCEND_DRAFT_SPARSE_KL_TOPK", "64"))),
        micro_seq_len=max(
            0, int(os.getenv("VLLM_ASCEND_DRAFT_MICRO_SEQ_LEN", "0"))),
        grad_accum_steps=max(
            1, int(os.getenv("VLLM_ASCEND_DRAFT_GRAD_ACCUM_STEPS", "1"))),
        fastrl_concat_enabled=_get_env_flag(
            "VLLM_ASCEND_DRAFT_FASTRL_CONCAT_ENABLE", True),
        fastrl_concat_batch_size=max(
            1, int(os.getenv("VLLM_ASCEND_DRAFT_FASTRL_CONCAT_BATCH_SIZE", "4"))),
        fastrl_concat_recent_samples=max(
            1, int(os.getenv("VLLM_ASCEND_DRAFT_FASTRL_CONCAT_RECENT_SAMPLES", "64"))),
        fastrl_concat_window_len=max(
            1, int(os.getenv("VLLM_ASCEND_DRAFT_FASTRL_CONCAT_WINDOW_LEN", "512"))),
    )
    dump_enabled = _get_env_flag("VLLM_ASCEND_DRAFT_DUMP_ENABLE", False)
    if not config.enabled and not config.profile_only and not dump_enabled:
        return None
    return DraftTrainer(model=model, config=config)
