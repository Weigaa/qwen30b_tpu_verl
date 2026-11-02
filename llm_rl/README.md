# AdaFloor code layout

- [qwen3_014](qwen3_014/README.md) contains the current vLLM 0.14.1 implementations.
  Use `qwen3_014/qwen3_main` for the primary AdaFloor runtime and basic sidecar.
  Use `qwen3_014/qwen3_sidecar` for multi-model auxiliary inference.
- [qwen3_011](qwen3_011/README.md) preserves the separate vLLM 0.11 implementations.
- `tailserve` is an independent in-engine serving prototype, not the AdaFloor
  multi-model sidecar implementation. `deepseek` and `qwen2_5` remain separate recipes.

Results and checkpoints belong under `/data`, not beside implementation directories.
The active MLSys paper is in
`qwen3_014/qwen3_main/analysis_eval/mlsys2027_two_metrics_v2/`.
