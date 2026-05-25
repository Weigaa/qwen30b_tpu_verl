# Redundancy Floor Sweep and Full-Redundancy Check

Run ID: `20260520152849`

Command:

```bash
FULL_REDUNDANCY_FLOORS="8 4 2 1" ./internal/run_full_redundancy_floor_sweep.sh
```

Common settings:

- Training script: `internal/wj_train_grpo_qwen30b_a3b_16die_true_weight_eager.sh`
- Sidecar: disabled (`VERL_SIDECAR_ENABLE=0`, `[elastic sidecar] enabled=0`)
- Elastic mode: `VLLM_ASCEND_ELASTIC_EXECUTION_MODE=1`
- NPU count: 16
- Rollout config: `max_num_seqs=32`, `max_num_batched_tokens=1024`, `max_prompt_length=1024`, `max_response_length=16384`, `rollout_n=16`
- Memory utilization: `actor_rollout_ref.rollout.gpu_memory_utilization=0.85`
- Dataset fraction: `data.dataset_fraction=0.003`
- Main summary artifacts:
  - `full_redundancy_runs/20260520152849/summary.md`
  - `full_redundancy_runs/20260520152849/summary.csv`
  - `full_redundancy_runs/20260520152849/manifest.tsv`

## Important Correction

The `floor=8/4/2` rows in this sweep are **not strict full expert
redundancy**. They are floor-targeted redundant-slot runs.

With `VLLM_ASCEND_ELASTIC_EXECUTION_MODE=1` and
`VLLM_ASCEND_ELASTIC_MIN_COMPUTE_GROUP_SIZE=<floor>`,
`vllm_ascend/envs.py::compute_elastic_init_redundancy_expert` computes:

```text
current_local_experts = logical_num_experts / initial_ep_size = 128 / 16 = 8
target_local_experts  = logical_num_experts / floor
```

So the tested floors mean:

- `floor=8`: enough slots for `16 -> 8`, about `16` experts/layer/rank.
- `floor=4`: enough slots for `16 -> 4`, about `32` experts/layer/rank.
- `floor=2`: enough slots for `16 -> 2`, about `64` experts/layer/rank.
- `floor=1`: strict full redundancy, all `128` experts/layer/rank.

This is why the measured max concurrency for `floor=4` is much higher than
older full-redundancy-like measurements such as `6.07x`: this run only
preallocated about one quarter of all experts per layer on each rank, not all
experts. The strict all-expert case is represented by `floor=1`, and it OOMs
before KV-cache sizing.

There is a second comparability issue in run `20260520152849`: the Qwen3 fast
path uses the vLLM native `FusedMoE` module, while the KV-cache headroom
detector only recognized `vllm_ascend.ops.fused_moe.AscendFusedMoE`. As a
result, this run did **not** subtract the post-shrink/post-restore safety
headrooms that older shrink-to-8 logs used:

```text
post-shrink MoE dispatch headroom: 536870912 bytes
post-restore DP collective headroom: 2147483648 bytes
post-restore EP collective headroom: 2147483648 bytes
post-restore MoE dispatch headroom: 2147483648 bytes
first-live-prefill activation headroom: 1073741824 bytes
```

Those reservations total `8053063680` bytes, or `7.5 GiB`. The measured
`floor=8` available KV memory in this run is `37101472256` bytes. Subtracting
the same headrooms gives `29048408576` bytes, which is within `179359744`
bytes of the older `28869048832` byte shrink-to-8 log. Therefore the apparent
`21.68x` vs `16.85x` difference is mostly a missing-headroom accounting issue,
not a different expert-slot strategy.

Code has been patched after this run so that `worker_v1.py` recognizes both
Ascend MoE and vLLM native `FusedMoE` for elastic headroom detection. Future
floor sweeps should be rerun before using the table as final paper data.

## Instrumentation Added

The sweep used extra logging to make the HBM boundary visible:

- Run metadata in the training script: sidecar state, elastic mode, floor, rollout lengths, and batch settings.
- Redundant-slot logs in fused MoE layer construction: per-layer `loaded_capacity`, weight shapes, per-expert bytes, and total expert-slot bytes.
- HBM profile logs in `determine_available_memory`: total HBM, profile peak, post-profile free memory, non-torch allocation, and available KV-cache memory.
- Scheduler preemption logs: request preemptions with KV-cache usage.
- Summary parser: `internal/summarize_full_redundancy_logs.py`.

## Results

| floor | actual redundancy semantics | observed consolidation | outcome | rollout_s | slot cap per layer | expert-slot GiB per rank | peak GiB | available KV GiB | KV tokens | max concurrency | preemptions | OOM |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 8 | floor-targeted, not full | 16 -> 8 | success | 1156.63 | 16 inferred | ~6.75 inferred | 17.53 | 34.55 | 377344 | 21.68x | 0 | no |
| 4 | floor-targeted, not full | 16 -> 8 -> 4 | success | 1188.80 | 32/33 measured | 13.50-13.92 measured | 24.70 | 27.38 | 299008 | 17.18x | 0 | no |
| 2 | floor-targeted, not full | 16 -> 8 -> 4, then failure | OOM after heavy KV preemption | N/A | 63-65 measured | 26.58-27.42 measured | 38.20 | 13.88 | 151552 | 8.71x | 52 | yes |
| 1 | strict full expert redundancy | failed during init/profile | OOM before KV cache sizing | N/A | 128 measured | 54.00 measured | N/A | N/A | N/A | N/A | 0 | yes |

Notes:

- Run `20260520152849` is missing shrink/restore KV safety headroom for the
  Qwen3 fast path. The raw `floor=8` concurrency is therefore too optimistic
  relative to older shrink-to-8 logs. With the older 7.5 GiB headroom applied,
  `floor=8` would be close to the previously observed `16.85x`.
- The `floor=8` slot-cap line was not emitted in this run, but the capacity follows the same deterministic slot rule as the measured `floor=4/2/1` runs. Its HBM and KV-cache profile were measured.
- Per expert per layer memory is `9437184` bytes, or `0.0087890625 GiB`.
- The 48-layer expert-slot totals are:
  - `floor=4`: typical `32 * 0.0087890625 * 48 = 13.5 GiB`; max observed `33` slots gives `13.921875 GiB`
  - `floor=2`: typical `64 * 0.0087890625 * 48 = 27.0 GiB`; max observed `65` slots gives `27.421875 GiB`
  - `floor=1`: `128 * 0.0087890625 * 48 = 54.0 GiB`
- The measured profile peak increases almost exactly with the extra expert slots:
  - `floor=4` peak: `24.70 GiB`
  - `floor=2` peak: `38.20 GiB`
  - Delta: `13.50 GiB`, matching the extra `32` expert slots per layer across `48` layers.

## Failure Evidence

`floor=2` successfully initialized KV cache but had only `151552` KV tokens, or `8.71x` concurrency for a `17408` token request. During rollout it repeatedly hit KV pressure:

- Preemptions: `52`
- Max logged KV usage: `96.9569%`
- Final OOM example:

```text
NPUWorkspaceAllocator tried to allocate 226.26 MiB
(NPU 0; 61.27 GiB total capacity; 59.46 MiB free)
```

This happened after shrink had reached the `ep_size=4` stage, before a stable `16 -> 2` execution completed.

`floor=1` failed earlier, during initialization/profile run. The model loaded full expert slots on each rank:

```text
loaded_capacity=128
Loading model weights took 56.8812 GB
```

Then the dummy/profile run OOMed before KV-cache sizing:

```text
NPUWorkspaceAllocator tried to allocate 20.00 MiB
(NPU 0; 61.27 GiB total capacity; 112.41 MiB free)
```

## Paper Takeaway

These logs support two separate claims:

1. Floor-targeted preallocation is feasible through `floor=4`, but becomes
   marginal at `floor=2`.
2. Strict full expert redundancy, where every rank preallocates all 128 experts
   per layer, is infeasible in this configuration.

The current data supports:

- `16 -> 8` with floor-targeted redundancy works normally.
- `16 -> 4` with floor-targeted redundancy still works in this run and does not show KV-cache preemption.
- `floor=2` sharply reduces available KV cache, causes repeated preemption, and eventually OOMs.
- `floor=1`, the true full-redundancy case, is infeasible even before KV-cache sizing.

Suggested wording:

```latex
Measured on the Qwen-30B MoE workload, floor-targeted redundant expert
preallocation is feasible for 16$\to$8 and 16$\to$4 consolidation under
the tested rollout configuration. In this mode, the available KV-cache
budget drops from 34.55 GiB at floor 8 to 27.38 GiB at floor 4, and then
to only 13.88 GiB at floor 2. At floor 2, the scheduler records 52
KV-cache preemptions with peak KV usage near 97\%, and the run
eventually fails with an NPU out-of-memory error. In contrast, strict
full expert redundancy corresponds to floor 1: each rank pre-allocates
all 128 experts per layer, loads about 56.88 GB of model weights, and
fails during the profiling run before KV-cache allocation. These
measurements show that HBM capacity, rather than communication
correctness, is the binding constraint on lossless consolidation depth.
```

For Eq. `P_e * N_e + K + A <= C`, a useful measured substitution is:

- `C ~= 61.27 GiB`
- `P_e ~= 0.008789 GiB` per expert per layer
- `N_e = loaded_capacity * 48 layers`
- `K` is the allocatable KV-cache budget after model/profile peak
- The residual non-expert/model/activation term is roughly `20 GiB` in successful configurations.

This gives:

- `floor=4`: `13.92 + 27.38 + ~20 <= 61.27`, feasible.
- `floor=2`: `27.42 + 13.88 + ~20 ~= 61.3`, marginal; preemption and OOM observed.
- `floor=1`: `54.00 + ~20 > 61.27`, infeasible before KV allocation.
