#!/usr/bin/env python3
import argparse
import os
import time

import torch
from transformers import AutoConfig
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config

from vllm_ascend.draft.model.qwen3_eagle3 import Qwen3ModelEagle3


def _coerce_int(value: object, default: int) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _require_npu():
    try:
        import torch_npu  # noqa: F401
    except Exception as exc:  # pragma: no cover - env specific
        raise RuntimeError("torch_npu is required for NPU profiling.") from exc
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        raise RuntimeError("NPU device is not available.")


def _build_config(model_path: str):
    base_cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    vocab_size = _coerce_int(getattr(base_cfg, "vocab_size", None), 151936)
    hidden_size = _coerce_int(getattr(base_cfg, "hidden_size", None), 4096)
    intermediate_size = _coerce_int(
        getattr(base_cfg, "intermediate_size", None), hidden_size * 4)
    num_hidden_layers = _coerce_int(
        getattr(base_cfg, "num_hidden_layers",
                getattr(base_cfg, "num_layers", None)), 1)
    num_heads = _coerce_int(getattr(base_cfg, "num_attention_heads", None), 32)
    num_kv_heads = _coerce_int(
        getattr(base_cfg, "num_key_value_heads", None), num_heads)
    max_position_embeddings = _coerce_int(
        getattr(base_cfg, "max_position_embeddings", None), 2048)
    rms_norm_eps = float(getattr(base_cfg, "rms_norm_eps", 1e-6))
    pad_token_id = _coerce_int(getattr(base_cfg, "pad_token_id", None), 0)
    attention_bias = bool(getattr(base_cfg, "attention_bias", False))
    sliding_window = getattr(base_cfg, "sliding_window", None)

    cfg = Qwen3Config(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_hidden_layers=num_hidden_layers,
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
    cfg.draft_vocab_size = vocab_size
    cfg.target_hidden_size = hidden_size
    if not hasattr(cfg, "layer_types"):
        cfg.layer_types = ["full_attention"] * num_hidden_layers
    if sliding_window is not None:
        cfg.sliding_window = sliding_window
    # Force eager attention for stability in profiling.
    cfg._attn_implementation = "eager"
    cfg.attn_implementation = "eager"
    return cfg


def main():
    parser = argparse.ArgumentParser(
        description="Profile a single Eagle3 training step on NPU.")
    parser.add_argument(
        "--model-path",
        default=os.getenv("MODEL_PATH", ""),
        help="HF model path for Qwen3 config (required).")
    parser.add_argument("--seq-len", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--prediction-length", type=int, default=1)
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Enable torch_npu profiler.")
    parser.add_argument(
        "--profile-dir",
        default="./result/profiler/draft_profile",
        help="Profiler output directory.")
    args = parser.parse_args()

    if not args.model_path:
        raise RuntimeError("--model-path is required for real config values.")

    _require_npu()
    device = torch.device("npu")

    cfg = _build_config(args.model_path)
    model = Qwen3ModelEagle3(cfg).to(device, dtype=torch.float32)
    model.train()

    batch = args.batch_size
    seq_len = max(1, args.seq_len)
    vocab_size = int(cfg.vocab_size)
    hidden_size = int(cfg.hidden_size)

    base_model_hidden_states = torch.randn(
        batch,
        seq_len,
        hidden_size * 3,
        device=device,
        dtype=torch.float32,
    )
    input_ids = torch.randint(
        low=0,
        high=vocab_size,
        size=(batch, seq_len),
        device=device,
        dtype=torch.long,
    )
    position_ids = torch.arange(seq_len, device=device,
                                dtype=torch.long).unsqueeze(0).repeat(batch, 1)
    target = torch.randn(
        batch,
        seq_len,
        vocab_size,
        device=device,
        dtype=torch.float32,
    )
    loss_mask = torch.ones(batch,
                           seq_len,
                           1,
                           device=device,
                           dtype=torch.float32)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    prof_ctx = None
    if args.profile:
        import torch_npu
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
        prof_ctx = torch_npu.profiler.profile(
            activities=[
                torch_npu.profiler.ProfilerActivity.CPU,
                torch_npu.profiler.ProfilerActivity.NPU,
            ],
            schedule=torch_npu.profiler.schedule(
                wait=0, warmup=0, active=max(1, args.steps), repeat=1),
            on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(
                args.profile_dir),
            record_shapes=False,
            profile_memory=False,
            with_stack=False,
            with_modules=False,
            with_flops=False,
            experimental_config=experimental_config,
        )

    if prof_ctx is None:
        class _Null:
            def __enter__(self):  # noqa: D401
                return None
            def __exit__(self, exc_type, exc, tb):
                return False
        prof_ctx = _Null()

    def _train_one_step():
        losses, _ = model(
            base_model_hidden_states=base_model_hidden_states,
            input_ids=input_ids,
            position_ids=position_ids,
            target=target,
            loss_mask=loss_mask,
            use_cache=False,
            prediction_length=args.prediction_length,
        )
        loss = torch.stack(losses).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    # Warmup to avoid first-step overhead skewing timing.
    for _ in range(max(0, args.warmup_steps)):
        _train_one_step()
    torch.npu.synchronize()

    times = []
    with prof_ctx as prof:
        for _ in range(max(1, args.steps)):
            torch.npu.synchronize()
            start = time.perf_counter()
            _train_one_step()
            torch.npu.synchronize()
            times.append((time.perf_counter() - start) * 1000.0)
            if prof is not None:
                prof.step()

    avg_ms = sum(times) / len(times)
    min_ms = min(times)
    max_ms = max(times)
    print(
        f"[draft-profile] device={device} "
        f"vocab_size={vocab_size} hidden_size={hidden_size} "
        f"seq_len={seq_len} batch={batch} "
        f"steps={len(times)} avg_ms={avg_ms:.3f} "
        f"min_ms={min_ms:.3f} max_ms={max_ms:.3f}")


if __name__ == "__main__":
    main()
