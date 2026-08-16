# AdaFloor Sidecar Integration

This repository is a compact copy of `qwen3_shrink_aware`. It preserves the
current floor2 correctness and memory fixes while excluding experiment logs,
checkpoints, caches, and generated paper artifacts.

The sidecar workflow was audited against remote commit
`c7d405007dde03ee5d836f703dce0e0a886b4c60`. The histories are unrelated, so
the commit was not merged wholesale. Only the sidecar-facing behavior was
ported onto the newer local runtime.

## Included behavior

* A watcher starts sidecar inference only after the training log reports a
  completed shrink to the expected active rank count.
* The watcher derives sidecar devices from inactive rollout ranks.
* A restore or rollout completion marker requests graceful sidecar shutdown
  before training resumes.
* Sidecar output is checkpointed in chunks and can resume after interruption.
* Training elastic headroom variables are removed from the sidecar process.
* Sidecar and training use separate dynamically checked MASTER and HCCL ports.
* The current `regroup` launcher keeps sidecar disabled unless
  `VERL_SIDECAR_ENABLE=1` is explicitly exported.

## Assets

The validated local defaults are `/data/Qwen3-8B` and `/data/gsm8k`. The
optional Pangu Pro MoE path is `/data/pangu-pro-moe-model`.

## Configuration check

This command does not import vLLM and does not access an NPU.

```bash
VERL_SIDECAR_CONFIG_ONLY=1 \
VERL_SIDECAR_MODEL_PATH=/data/Qwen3-8B \
VERL_SIDECAR_PROMPTS_FILE=/data/gsm8k \
VERL_SIDECAR_NPU_DEVICES=8,9,10,11,12,13,14,15 \
VERL_SIDECAR_PARALLEL_MODE=hybrid \
VERL_SIDECAR_TENSOR_PARALLEL_SIZE=8 \
VERL_SIDECAR_REPLICA_COUNT=1 \
internal/run_elastic_sidecar_infer.sh
```

The full experiment entry point is
`run_adafloor_natural_floor2_with_qwen3_8b_sidecar.sh`. Set
`SIDECAR_DRY_RUN=1` to inspect the common epoch0 and output configuration
without starting training.

The incremental 16 to 8 to 4 to 2 workflow is documented in
`MULTISTAGE_SIDECAR.md`. Its Qwen2.5-1.5B entry point uses TP1 independent
replicas and launches only on ranks newly released by each transition.
