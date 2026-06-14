#
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
# This file is a part of the vllm-ascend project.
#
# Todo: Once https://github.com/vllm-project/vllm/issues/22246 is merged in vllm. Remove eplb utils.
import random

import torch
from vllm.logger import logger


def _determine_primary_rank(global_expert_num, world_size, expert_id):
    if world_size == 1:
        return 0
    local_num_experts = global_expert_num // world_size
    split_point = local_num_experts * (world_size - 1)
    if expert_id >= split_point:
        return world_size - 1
    return expert_id // local_num_experts


def _can_use_balanced_cyclic_replica_layout(global_expert_num, world_size,
                                            global_redundant_expert_num):
    redundant = max(int(global_redundant_expert_num), 0)
    if (global_expert_num <= 0 or world_size <= 1 or redundant <= 0
            or global_expert_num % world_size != 0
            or redundant % global_expert_num != 0):
        return False
    replica_rounds = redundant // global_expert_num
    return replica_rounds <= (world_size - 1)


def build_redundant_replica_expert_map(global_expert_num,
                                       world_size,
                                       global_redundant_expert_num,
                                       prefer_balanced_cyclic: bool = False):
    expert_map_all = torch.full((world_size, global_expert_num),
                                -1,
                                dtype=torch.int32)
    next_local_ids = [0 for _ in range(world_size)]

    for expert_id in range(global_expert_num):
        primary_rank = _determine_primary_rank(global_expert_num, world_size,
                                               expert_id)
        expert_map_all[primary_rank, expert_id] = next_local_ids[primary_rank]
        next_local_ids[primary_rank] += 1

    # Mode=1 floor-targeted redundancy wants an exact loaded-slot capacity on
    # every rank. When the number of primary experts is already evenly split
    # and the requested replica rounds are integral, assign each replica round
    # with a fixed cyclic shift away from the primary rank. This preserves the
    # per-expert uniqueness constraint while keeping every rank's slot count
    # exactly aligned to the floor target (e.g. 32 for floor=4, 64 for floor=2).
    if (prefer_balanced_cyclic
            and _can_use_balanced_cyclic_replica_layout(
                global_expert_num, world_size, global_redundant_expert_num)):
        replica_rounds = max(int(global_redundant_expert_num), 0) // global_expert_num
        for replica_round in range(replica_rounds):
            shift = replica_round + 1
            for expert_id in range(global_expert_num):
                primary_rank = _determine_primary_rank(global_expert_num,
                                                       world_size, expert_id)
                target_rank = (primary_rank + shift) % world_size
                if int(expert_map_all[target_rank, expert_id].item()) >= 0:
                    raise RuntimeError(
                        "Balanced cyclic replica placement hit a duplicate "
                        f"assignment for expert_id={expert_id}, "
                        f"primary_rank={primary_rank}, target_rank={target_rank}, "
                        f"replica_round={replica_round}.")
                expert_map_all[target_rank, expert_id] = next_local_ids[target_rank]
                next_local_ids[target_rank] += 1
        return expert_map_all

    for replica_idx in range(max(int(global_redundant_expert_num), 0)):
        expert_id = replica_idx % global_expert_num
        replica_round = replica_idx // global_expert_num
        primary_rank = _determine_primary_rank(global_expert_num, world_size,
                                               expert_id)
        candidate_ranks = [
            rank for rank in range(world_size)
            if int(expert_map_all[rank, expert_id].item()) < 0
        ]
        if not candidate_ranks:
            break
        rotate_start = (primary_rank + 1 + replica_round) % world_size
        candidate_ranks.sort(
            key=lambda rank: (next_local_ids[rank],
                              (rank - rotate_start) % world_size))
        target_rank = candidate_ranks[0]
        expert_map_all[target_rank, expert_id] = next_local_ids[target_rank]
        next_local_ids[target_rank] += 1

    return expert_map_all


def determine_default_expert_map(global_expert_num, world_size, rank_id,
                                 global_redundant_expert_num):
    if world_size == 1:
        local_ids = torch.arange(global_expert_num, dtype=torch.int32)
        return (global_expert_num, local_ids)

    local_num_experts = global_expert_num // world_size

    expert_map = torch.full((global_expert_num, ), -1, dtype=torch.int32)

    if rank_id < world_size - 1:
        start = rank_id * local_num_experts
        end = (rank_id + 1) * local_num_experts
        local_count = local_num_experts
    else:
        start = rank_id * local_num_experts
        end = global_expert_num
        local_count = global_expert_num - rank_id * local_num_experts

    if isinstance(global_redundant_expert_num,
                  int) and rank_id < global_redundant_expert_num:
        local_count += 1
        if end < global_expert_num:
            end += 1
        else:
            start -= 1

    if isinstance(local_count, int):
        local_ids = torch.arange(local_count, dtype=torch.int32)
        expert_map[start:end] = local_ids

    return (local_count, expert_map)


def determine_redundant_replica_expert_map(global_expert_num, world_size,
                                           rank_id,
                                           global_redundant_expert_num,
                                           prefer_balanced_cyclic: bool = False):
    expert_map_all = build_redundant_replica_expert_map(
        global_expert_num,
        world_size,
        global_redundant_expert_num,
        prefer_balanced_cyclic=prefer_balanced_cyclic)
    local_count = int((expert_map_all[rank_id] != -1).sum().item())
    return local_count, expert_map_all[rank_id]


def generate_log2phy_map(expert_map):
    num_local_experts = expert_map.max() + 1
    log2phy_map = expert_map.clone()
    num_ranks, num_global_expert = log2phy_map.shape

    row_indices = torch.arange(num_ranks).view(-1, 1).expand(num_ranks, \
                                                             num_global_expert) * num_local_experts
    log2phy_map[log2phy_map != -1] += row_indices[log2phy_map != -1]

    for idx in range(num_global_expert):
        positive_rank_idx = torch.where(log2phy_map[:, idx] != -1)[0]
        negative_rank_idx = torch.where(log2phy_map[:, idx] == -1)[0]
        num_rank_holding_expert = positive_rank_idx.size(0)

        if num_rank_holding_expert == 0:
            log2phy_map[:, idx] = torch.full((num_ranks, ),
                                             0,
                                             dtype=log2phy_map.dtype)

        if num_rank_holding_expert == 1:
            log2phy_map[negative_rank_idx, idx] = torch.full(
                (num_ranks - 1, ),
                log2phy_map[positive_rank_idx, idx].item(),
                dtype=log2phy_map.dtype)
        else:
            try:
                random_list = [
                    random.choice(log2phy_map[positive_rank_idx, idx])
                    for _ in range(num_ranks - num_rank_holding_expert)
                ]
                log2phy_map[negative_rank_idx,
                            idx] = torch.tensor(random_list,
                                                dtype=log2phy_map.dtype)
            except Exception as e:
                logger.error(f"Fail to get log2phy_map: {str(e)}")

    return log2phy_map


def determine_default_log2phy_map(global_expert_num, world_size, rank_id,
                                  global_redundant_expert_num):
    if world_size == 1:
        local_ids = torch.arange(global_expert_num, dtype=torch.int32)
        expert_map_all = local_ids.unsqueeze(0).expand(world_size, -1)
        log2phy_map_all = generate_log2phy_map(expert_map_all)
        return log2phy_map_all[rank_id]

    local_num_experts = global_expert_num // world_size

    expert_map_all = torch.full((world_size, global_expert_num),
                                -1,
                                dtype=torch.int32)

    for r in range(world_size):
        if r < world_size - 1:
            start = r * local_num_experts
            end = (r + 1) * local_num_experts
            local_count = local_num_experts
        else:
            start = r * local_num_experts
            end = global_expert_num
            local_count = global_expert_num - r * local_num_experts

        if isinstance(global_redundant_expert_num,
                      int) and rank_id < global_redundant_expert_num:
            local_count += 1
            if end < global_expert_num:
                end += 1
            else:
                start -= 1

        if isinstance(local_count, int):
            local_ids = torch.arange(local_count, dtype=torch.int32)
            expert_map_all[r, start:end] = local_ids

    log2phy_map_all = generate_log2phy_map(expert_map_all)

    return log2phy_map_all[rank_id]


def determine_redundant_replica_log2phy_map(global_expert_num, world_size,
                                            rank_id,
                                            global_redundant_expert_num,
                                            prefer_balanced_cyclic: bool = False):
    expert_map_all = build_redundant_replica_expert_map(
        global_expert_num,
        world_size,
        global_redundant_expert_num,
        prefer_balanced_cyclic=prefer_balanced_cyclic)
    log2phy_map_all = generate_log2phy_map(expert_map_all)
    return log2phy_map_all[rank_id]
