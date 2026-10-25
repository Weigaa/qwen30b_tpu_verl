# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import annotations

from collections.abc import Iterator, Sized
from typing import Optional

import numpy as np
import torch
from omegaconf import DictConfig

from verl import DataProto
from verl.experimental.dataset.sampler import AbstractCurriculumSampler


class LengthAwareEpochSampler(AbstractCurriculumSampler):
    """Group prompts with similar expected response lengths into nearby batches.

    The sampler consumes rollout statistics from previous steps via `update(batch)`.
    At the next epoch iterator build, it reorders dataset rows by the estimated
    response length. This reduces long/short mixing inside each generation batch.
    """

    def __init__(self, data_source: Sized, data_config: DictConfig):
        self.data_source = data_source
        self.data_config = data_config
        self.num_samples = len(data_source)

        self.shuffle = bool(data_config.get("shuffle", True))
        self.seed = int(data_config.get("seed", 1))
        self.batch_size = int(data_config.get("gen_batch_size", data_config.train_batch_size))

        sampler_cfg = data_config.get("sampler", {})
        self.bucket_size = int(sampler_cfg.get("bucket_size", 512))
        self.ema_decay = float(sampler_cfg.get("ema_decay", 0.7))
        self.shuffle_batch_blocks = bool(sampler_cfg.get("shuffle_batch_blocks", True))
        self.default_length = float(
            sampler_cfg.get("default_length", max(1, int(data_config.get("max_response_length", 1024) // 2)))
        )

        self._epoch = 0
        self._length_estimate = np.full((self.num_samples,), self.default_length, dtype=np.float32)
        self._seen = np.zeros((self.num_samples,), dtype=np.int64)

    def __len__(self) -> int:
        return self.num_samples

    def __iter__(self) -> Iterator[int]:
        rng = np.random.default_rng(self.seed + self._epoch)
        self._epoch += 1

        order = np.arange(self.num_samples, dtype=np.int64)
        if self._seen.any():
            est = self._length_estimate[order]
            # coarser bucketing first, then fine-grained length, with random tie-break
            bucket = np.floor(est / max(1, self.bucket_size)).astype(np.int64)
            tie_break = rng.random(order.shape[0], dtype=np.float32)
            sort_idx = np.lexsort((tie_break, est, bucket))
            order = order[sort_idx]

            if self.shuffle_batch_blocks and self.batch_size > 0:
                blocks = [order[i : i + self.batch_size] for i in range(0, len(order), self.batch_size)]
                rng.shuffle(blocks)
                order = np.concatenate(blocks, axis=0)
        elif self.shuffle:
            order = rng.permutation(order)

        return iter(order.tolist())

    def update(self, batch: DataProto) -> None:
        if "response_mask" not in batch.batch:
            return

        sample_ids = self._extract_sample_ids(batch)
        if sample_ids is None or sample_ids.size == 0:
            return

        lengths = batch.batch["response_mask"].sum(-1)
        if isinstance(lengths, torch.Tensor):
            lengths = lengths.detach().cpu().numpy()
        else:
            lengths = np.asarray(lengths)
        lengths = lengths.astype(np.float32, copy=False)

        sample_ids = sample_ids.astype(np.int64, copy=False)
        valid = (sample_ids >= 0) & (sample_ids < self.num_samples)
        if not np.any(valid):
            return

        sample_ids = sample_ids[valid]
        lengths = lengths[valid]

        uniq, inv = np.unique(sample_ids, return_inverse=True)
        sum_len = np.zeros((uniq.shape[0],), dtype=np.float64)
        cnt = np.zeros((uniq.shape[0],), dtype=np.int64)
        np.add.at(sum_len, inv, lengths)
        np.add.at(cnt, inv, 1)
        mean_len = sum_len / np.maximum(cnt, 1)

        old = self._length_estimate[uniq]
        seen = self._seen[uniq]
        updated = np.where(seen > 0, self.ema_decay * old + (1.0 - self.ema_decay) * mean_len, mean_len)
        self._length_estimate[uniq] = updated.astype(np.float32)
        self._seen[uniq] = seen + 1

    def _extract_sample_ids(self, batch: DataProto) -> Optional[np.ndarray]:
        for key in ("dataset_item_idx", "index"):
            if key in batch.non_tensor_batch:
                values = batch.non_tensor_batch[key]
                if isinstance(values, torch.Tensor):
                    values = values.detach().cpu().numpy()
                else:
                    values = np.asarray(values)
                return values
        return None
