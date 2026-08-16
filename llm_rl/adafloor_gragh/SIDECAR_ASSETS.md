# Sidecar Assets

Validated on 2026-07-31 without loading a model or accessing an NPU.

| Asset | Local path | Verified identity | Verification |
| --- | --- | --- | --- |
| Multi-stage serving model | `/data/Qwen2.5-1.5B-Instruct` | `Qwen/Qwen2.5-1.5B-Instruct` | The 3,087,467,144-byte safetensors file is readable and contains 338 tensors. The tokenizer and generation configuration are present. |
| Serving model | `/data/Qwen3-8B` | `Qwen/Qwen3-8B` | Five indexed safetensor shards are present. Their total size is 16,381,516,776 bytes. |
| Evaluation data | `/data/gsm8k` | `openai/gsm8k` with the `main` subset | `train.parquet` has 7,473 rows and `test.parquet` has 1,319 rows. |
| Optional serving model | `/data/pangu-pro-moe-model` | Pangu Pro MoE | All 29 indexed safetensor shards are present. |

The Qwen3-8B directory records ModelScope revision
`7c9709d23bd2136dac1d6ea1fe30f4107d681cd6` for its weight shards. The GSM8K
conversion summary identifies `openai/gsm8k` as the data source and preserves
both train and test splits.
