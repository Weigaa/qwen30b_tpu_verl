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

import dataclasses
import os


@dataclasses.dataclass
class RolloutRebalanceConfig:
    enable = int(os.environ.get("ROLLOUT_REBALANCE_ENABLE", "0"))  # Whether to enable RolloutRebalance
    check_interval = 1000  # Interval for performing RolloutRebalance checks in decode steps

    # Whether to enable multi-graphs. If disabled, it will still perform rollout rebalance 
    # based on the pre-compiled graph, but no significant improvement. 
    multi_graph = True
    graph_batch_sizes = [128, 64, 48, 32, 16, 8, 4]  # Graph size configuration for pre-compiled graphs.

    profile = True  # Whether to print during inference.
    profile_interval = 100   # Interval for printing logs.
