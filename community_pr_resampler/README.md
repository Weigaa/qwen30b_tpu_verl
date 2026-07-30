# Length-Aware Resampler PR Bundle

This folder isolates the resampler-related changes so we can submit them as a focused community PR.

It contains three files:

- `0001-length-aware-resampler.patch`: the minimal code patch
- `README.md`: usage notes
- `run_resampler_example.sh`: an example launch script based on your regroup script

## What is included

The patch adds four resampler-specific pieces:

1. `LengthAwareEpochSampler`
   A curriculum sampler that reorders prompts at the start of each epoch using previous rollout response lengths.

2. Stable sample ids in the dataset
   Adds `dataset_item_idx` to each dataset row so rollout stats can be aggregated back to the same sample across epochs.

3. Dataset alignment to full batches
   Floors the train dataset size to a multiple of `gen_batch_size` so each epoch sees the same prompt set when `drop_last=True`.

4. Optional rollout long-tail guard
   Uses the sampler's expected response length to cap `max_tokens` for a batch:
   `cap = min(response_length, max(min_tokens, factor * expected_len))`

## Apply the patch

From the repository root:

```bash
git apply community_pr_resampler/0001-length-aware-resampler.patch
```

If you want to preview first:

```bash
git apply --check community_pr_resampler/0001-length-aware-resampler.patch
```

## How to enable the resampler

Set the custom sampler in Hydra config:

```bash
data.sampler.class_path=pkg://verl.experimental.dataset.length_bucket_sampler
data.sampler.class_name=LengthAwareEpochSampler
+data.sampler.bucket_size=1024
+data.sampler.ema_decay=0.7
+data.sampler.shuffle_batch_blocks=True
```

Recommended settings:

- `data.dataloader_num_workers=0`
- `data.shuffle=False`
- keep `drop_last=True` in the training dataloader

`dataloader_num_workers=0` is recommended because the sampler updates its ordering using rollout stats on the main process between epochs. Keeping a single worker avoids stale sampler state and makes behavior deterministic.

## Optional rollout long-tail guard

These environment variables enable early stopping for obviously overlong batches:

```bash
export VLLM_ROLLOUT_EARLY_STOP_ENABLE=1
export VLLM_ROLLOUT_EARLY_STOP_FACTOR=2.0
export VLLM_ROLLOUT_EARLY_STOP_MIN_TOKENS=10000
```

Default behavior:

- enabled by default
- only active when the dataloader sampler is an `AbstractCurriculumSampler`
- if the computed cap is greater than or equal to the rollout `response_length`, it does nothing

## Expected behavior

- Epoch 1 behaves like baseline because there is no rollout-length history yet.
- From epoch 2 onward, prompts with similar estimated response lengths are grouped into nearby batches.
- This reduces long/short mixing inside generation batches and typically improves rollout throughput.

## Notes for PR scope

This bundle intentionally excludes unrelated local changes such as:

- request id logging
- checkpoint save logic
- draft training and profiling changes
- elastic EP or MoE experiments

That keeps the PR review surface small and centered on the resampler optimization itself.
