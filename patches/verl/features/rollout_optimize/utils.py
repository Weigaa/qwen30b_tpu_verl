# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
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

import functools
import torch
from vllm.logger import logger


class CheckCounter:
    """
    A counter class to check if a specific number of calls has reached a threshold.
    """
    def __init__(self, threshold=100):
        self.total = 0  # Tracks the total number of times `check()` has been called since instantiation.
        self.counter = 0  # Tracks the count for the current cycle, resets after reaching the threshold.
        self.threshold = threshold  # The configured threshold for the counter to reset.

    def check(self):
        self.total += 1
        self.counter += 1
        if self.counter >= self.threshold:
            self.reset()
            return True
        return False

    def reset(self):
        self.counter = 0


class _Cache:
    global_rank: int = None


def rank_log_info(msg, force=False):
    rank = _Cache.global_rank
    if rank is None:
        try:
            rank = _Cache.global_rank = torch.distributed.get_rank()
        except (RuntimeError, ValueError, ImportError):
            pass
        if rank is None:
            logger.info(f'[Rank=None]: {msg}')
            return
    # In non-forced mode, only rank 0 logs by default.
    if not force and rank not in [0]:
        return
    logger.info(f'[Rank={rank}]: {msg}')


def _patch_cls_method(original_method, before_callback=None, after_callback=None):
    @functools.wraps(original_method)
    def patched_method(self, *args, **kwargs):
        if before_callback and callable(before_callback):
            modified_params = before_callback(self, *args, **kwargs)
            if isinstance(modified_params, tuple) and len(modified_params) == 2:
                args, kwargs = modified_params

        ret = original_method(self, *args, **kwargs)

        if after_callback and callable(after_callback):
            final_ret = after_callback(self, ret, *args, **kwargs)
            if final_ret is not None:
                return final_ret

        return ret

    return patched_method


class _HookUtils:
    """
    A decorator-based toolkit for hooking into class methods.
    """
    @staticmethod
    def before(cls, method_name):
        original_method = getattr(cls, method_name)

        def decorator(callback):
            setattr(cls, method_name, _patch_cls_method(original_method, before_callback=callback))
            return callback

        return decorator

    @staticmethod
    def after(cls, method_name):
        original_method = getattr(cls, method_name)

        def decorator(callback):
            setattr(cls, method_name, _patch_cls_method(original_method, after_callback=callback))
            return callback

        return decorator


# Provides namespaced decorators like `@hook.before` and `@hook.after`
# to add logic without modifying the original class source code.
hook = _HookUtils()
