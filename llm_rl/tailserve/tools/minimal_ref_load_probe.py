import argparse
import os
import traceback

import torch
import torch.distributed
from omegaconf import OmegaConf

from megatron.core import parallel_state as mpu
from verl.utils.torch_dtypes import PrecisionType
from verl.utils.megatron_utils import McoreModuleWrapperConfig, make_megatron_module, unwrap_model
from verl.utils.model import load_mcore_dist_weights
from verl.workers.megatron_workers import ActorRolloutRefWorker


MODEL_PATH = "/data/Qwen3-30B-A3B"
DISTCP_PATH = "/data/Qwen3-30B-A3B_megatron"


def build_actor_rollout_ref_config():
    override_tf = {
        "recompute_granularity": "full",
        "recompute_method": "block",
        "recompute_num_layers": 1,
        "use_flash_attn": True,
        "pipeline_num_transformer_layers": [[11], [13], [13], [11]],
        "moe_token_dispatcher_type": "alltoall",
        "moe_alltoall_overlap_comm": True,
        "use_fused_rotary_pos_emb": True,
        "use_fused_swiglu": True,
        "seq_length": 2048,
        "num_layers_in_first_pipeline_stage": 11,
        "num_layers_in_last_pipeline_stage": 11,
        "swap_optimizer": True,
    }
    base_megatron = {
        "tensor_model_parallel_size": 4,
        "pipeline_model_parallel_size": 4,
        "virtual_pipeline_model_parallel_size": None,
        "context_parallel_size": 1,
        "expert_model_parallel_size": 4,
        "expert_tensor_parallel_size": 1,
        "sequence_parallel": True,
        "use_distributed_optimizer": True,
        "use_dist_checkpointing": True,
        "dist_checkpointing_path": DISTCP_PATH,
        "seed": 42,
        "override_ddp_config": {},
        "override_transformer_config": override_tf,
        "override_mcore_model_config": {},
        "use_mbridge": False,
        "forward_only": False,
        "strategy": "megatron",
        "param_offload": False,
        "grad_offload": False,
        "optimizer_offload": False,
    }
    actor_cfg = {
        "load_weight": True,
        "ppo_mini_batch_size": 32,
        "ppo_epochs": 1,
        "shuffle": False,
        "data_loader_seed": 1,
        "use_fused_kernels": False,
        "recompute_old_log_prob": True,
        "use_kl_loss": True,
        "kl_loss_coef": 0.001,
        "kl_loss_type": "low_var_kl",
        "entropy_coeff": 0.0,
        "loss_agg_mode": "token-mean",
        "policy_loss": {},
        "profiler": {},
        "megatron": base_megatron,
        "optim": {
            "lr": 1e-6,
            "clip_grad": 10000,
            "betas": [0.9, 0.999],
            "weight_decay": 0.01,
            "weight_decay_incr_style": "constant",
            "lr_decay_style": "constant",
            "lr_decay_steps": None,
            "min_lr": 0.0,
            "lr_warmup_steps": -1,
            "lr_warmup_init": 0.0,
            "lr_warmup_steps_ratio": 0.0,
            "lr_wsd_decay_steps": None,
            "lr_wsd_decay_style": "exponential",
            "total_training_steps": -1,
            "use_checkpoint_opt_param_scheduler": False,
            "override_optimizer_config": {},
            "optimizer": "adam",
        },
    }
    ref_megatron = dict(base_megatron)
    ref_megatron["use_distributed_optimizer"] = False
    ref_cfg = {
        "load_weight": True,
        "log_prob_micro_batch_size_per_gpu": 8,
        "shuffle": False,
        "data_loader_seed": 1,
        "use_fused_kernels": False,
        "recompute_old_log_prob": True,
        "use_kl_loss": False,
        "entropy_coeff": 0.0,
        "loss_agg_mode": "token-mean",
        "policy_loss": {},
        "profiler": {},
        "megatron": ref_megatron,
    }
    rollout_cfg = {"n": 16, "log_prob_micro_batch_size_per_gpu": 4, "profiler": {}}
    config = {
        "model": {
            "path": MODEL_PATH,
            "trust_remote_code": False,
            "override_config": {},
        },
        "actor": actor_cfg,
        "ref": ref_cfg,
        "rollout": rollout_cfg,
        "nccl_timeout": 600,
    }
    return OmegaConf.create(config)


def prepare_worker(worker):
    worker.param_dtype = torch.bfloat16
    worker.dtype = PrecisionType.to_dtype(worker.param_dtype)


def summarize_moe_classes(parallel_model):
    model = unwrap_model(parallel_model[0])
    layer = model.decoder.layers[0]
    parts = [f"mlp={type(layer.mlp).__name__}"]
    experts = getattr(layer.mlp, "experts", None)
    if experts is not None:
        parts.append(f"experts={type(experts).__name__}")
        for attr in ("linear_fc1", "linear_fc2"):
            obj = getattr(experts, attr, None)
            if obj is not None:
                parts.append(f"{attr}={type(obj).__name__}")
    return ", ".join(parts)


def run_actor_init(worker):
    print(f"[probe] actor init start rank={worker.rank} pid={os.getpid()}", flush=True)
    worker.init_model()
    print(
        f"[probe] actor init done rank={worker.rank} pid={os.getpid()} "
        f"{summarize_moe_classes(worker.actor_module)}",
        flush=True,
    )


def run_ref_init(worker):
    print(f"[probe] ref init start rank={worker.rank} pid={os.getpid()}", flush=True)
    worker.init_model()
    print(
        f"[probe] ref init done rank={worker.rank} pid={os.getpid()} "
        f"{summarize_moe_classes(worker.ref_module)}",
        flush=True,
    )


def build_actor_module_only(worker, do_load: bool):
    override_model_config = {}
    override_transformer_config = OmegaConf.to_container(
        OmegaConf.create(worker.config.actor.megatron.get("override_transformer_config", {}))
    )
    override_ddp_config = OmegaConf.to_container(
        OmegaConf.create(worker.config.actor.megatron.get("override_ddp_config", {}))
    )
    worker._init_hf_config_and_tf_config(
        worker.config.model.path,
        worker.config.model.path,
        worker.dtype,
        override_model_config,
        override_transformer_config,
        worker.config.model.get("trust_remote_code", False),
        worker.config.actor.megatron.use_mbridge,
    )
    wrap_config = McoreModuleWrapperConfig(
        is_value_model=False,
        share_embeddings_and_output_weights=worker.share_embeddings_and_output_weights,
        wrap_with_ddp=True,
        use_distributed_optimizer=worker.config.actor.megatron.use_distributed_optimizer,
    )
    actor_module = make_megatron_module(
        wrap_config=wrap_config,
        tf_config=worker.tf_config,
        hf_config=worker.hf_config,
        bridge=worker.bridge,
        override_model_config=override_model_config,
        override_ddp_config=override_ddp_config,
    )
    print(
        f"[probe] actor module built rank={worker.rank} pid={os.getpid()} do_load={do_load} "
        f"{summarize_moe_classes(actor_module)}",
        flush=True,
    )
    if do_load:
        load_mcore_dist_weights(
            actor_module,
            worker.config.actor.megatron.dist_checkpointing_path,
            is_value_model=False,
        )
        print(f"[probe] actor module load done rank={worker.rank} pid={os.getpid()}", flush=True)
    return actor_module


def cleanup():
    try:
        if mpu.model_parallel_is_initialized():
            mpu.destroy_model_parallel()
    except Exception:
        traceback.print_exc()
    try:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
    except Exception:
        traceback.print_exc()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=[
            "actor_then_ref_same_process",
            "ref_only_fresh_process",
            "actor_build_no_load_then_ref_same_process",
            "actor_load_no_wrapper_then_ref_same_process",
        ],
        required=True,
    )
    args = parser.parse_args()

    config = build_actor_rollout_ref_config()
    exit_code = 0
    try:
        if args.mode == "actor_then_ref_same_process":
            actor_worker = ActorRolloutRefWorker(config=config, role="actor")
            prepare_worker(actor_worker)
            run_actor_init(actor_worker)

            ref_worker = ActorRolloutRefWorker(config=config, role="ref")
            prepare_worker(ref_worker)
            run_ref_init(ref_worker)
        elif args.mode == "actor_build_no_load_then_ref_same_process":
            actor_worker = ActorRolloutRefWorker(config=config, role="actor")
            prepare_worker(actor_worker)
            build_actor_module_only(actor_worker, do_load=False)

            ref_worker = ActorRolloutRefWorker(config=config, role="ref")
            prepare_worker(ref_worker)
            run_ref_init(ref_worker)
        elif args.mode == "actor_load_no_wrapper_then_ref_same_process":
            actor_worker = ActorRolloutRefWorker(config=config, role="actor")
            prepare_worker(actor_worker)
            build_actor_module_only(actor_worker, do_load=True)

            ref_worker = ActorRolloutRefWorker(config=config, role="ref")
            prepare_worker(ref_worker)
            run_ref_init(ref_worker)
        elif args.mode == "ref_only_fresh_process":
            ref_worker = ActorRolloutRefWorker(config=config, role="ref")
            prepare_worker(ref_worker)
            run_ref_init(ref_worker)
    except Exception:
        traceback.print_exc()
        exit_code = 1
    finally:
        cleanup()
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
