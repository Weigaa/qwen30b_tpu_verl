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
import json
import os.path
from collections import defaultdict
import random

import numpy as np
import torch
from vllm.logger import logger


def expert_file_to_tensor(expert_map_path, layer_id):
    with open(expert_map_path) as f:
        data = json.load(f)
    physical_count = 0
    device_data = []
    if layer_id > data["moe_layer_count"]:
        raise ValueError("Invalid EPLB Table")
    if layer_id == data["moe_layer_count"]:
        logger.warning("Init expert map of mtp/eagle when using sample.")
        return None, None
    for device in data["layer_list"][layer_id]["device_list"]:
        physical_count += len(device["device_expert"])
        device_data.append(device["device_expert"])
    global_placement = torch.tensor(device_data, dtype=torch.int32)
    return global_placement, physical_count


def generate_global_placement(n_expert, ep_size, n_redundant):
    all_experts = np.arange(n_expert)
    groups = np.array_split(all_experts, ep_size)
    for i in range(n_redundant):
        j = i % ep_size + 1
        if len(groups[-j]) == 0:
            groups[-j] = np.append(groups[-j], j)
        else:
            groups[-j] = np.append(groups[-j], (groups[-j][-1] + 1) % n_expert)
    return torch.tensor(groups, dtype=torch.int32)


def init_eplb_config(eplb_config, layer_id, moe_config):
    expert_map_path = eplb_config.expert_map_path
    n_experts = moe_config.num_experts
    ep_size = moe_config.ep_size
    global_placement = None
    eplb_enable = eplb_config.dynamic_eplb
    n_redundant = eplb_config.num_redundant_experts if eplb_enable else 0
    if expert_map_path:
        if not (os.path.exists(expert_map_path) and os.access(expert_map_path, os.R_OK)):
            raise ValueError("Invalid EPLB path")
        eplb_enable = True
        global_placement, physical_count = expert_file_to_tensor(expert_map_path, layer_id)
        if physical_count is not None:
            n_redundant = physical_count - n_experts
            if not moe_config.supports_eplb:
                raise ValueError("Eplb supports only w8a8_dynamic quantization.")
        else:
            eplb_enable = False

    if global_placement is None:
        global_placement = generate_global_placement(n_experts, ep_size, n_redundant)

    if ep_size == 1:
        assert not eplb_enable, "EPLB must used in expert parallelism."
        return None, None, None, n_redundant
    global_expert_map = []
    for rankid in range(ep_size):
        expert_map = torch.full((n_experts,), -1, dtype=torch.int32)
        local_placement = global_placement[rankid]
        expert_map[local_placement] = torch.arange(local_placement.shape[0], dtype=torch.int32)
        global_expert_map.append(expert_map)
        if rankid == moe_config.ep_rank:
            local_expert_map = expert_map.npu()
    log2phy = generate_log2phy_map(global_expert_map, moe_config.ep_rank).npu() if eplb_enable else None

    return torch.stack(global_expert_map), local_expert_map, log2phy, n_redundant


def generate_log2phy_map(global_expert_map, ep_rank):
    log2phy_map = defaultdict(list)
    valid_count = torch.sum(global_expert_map[0] != -1)
    for rankid, map_per_rank in enumerate(global_expert_map):
        for idx, val in enumerate(map_per_rank):
            val = val.item()
            if val != -1:
                log2phy_map[idx].append(val + rankid * valid_count)

    for key in log2phy_map:
        num_of_duplications = len(log2phy_map[key])
        log2phy_map[key] = log2phy_map[key][ep_rank % num_of_duplications]

    log2phy_map = torch.scatter(
        torch.zeros(len(log2phy_map), dtype=torch.int32),
        0,
        torch.tensor(list(log2phy_map), dtype=torch.int64),
        torch.tensor(list(log2phy_map.values()), dtype=torch.int32),
    )

    return log2phy_map


def _determine_primary_rank(global_expert_num, world_size, expert_id):
    if world_size == 1:
        return 0
    local_num_experts = global_expert_num // world_size
    split_point = local_num_experts * (world_size - 1)
    if expert_id >= split_point:
        return world_size - 1
    return expert_id // local_num_experts


def build_redundant_replica_expert_map(global_expert_num, world_size, global_redundant_expert_num):
    expert_map_all = torch.full((world_size, global_expert_num), -1, dtype=torch.int32)
    next_local_ids = [0 for _ in range(world_size)]

    for expert_id in range(global_expert_num):
        primary_rank = _determine_primary_rank(global_expert_num, world_size, expert_id)
        expert_map_all[primary_rank, expert_id] = next_local_ids[primary_rank]
        next_local_ids[primary_rank] += 1

    for replica_idx in range(max(int(global_redundant_expert_num), 0)):
        expert_id = replica_idx % global_expert_num
        replica_round = replica_idx // global_expert_num
        primary_rank = _determine_primary_rank(global_expert_num, world_size, expert_id)
        candidate_ranks = [
            rank for rank in range(world_size)
            if int(expert_map_all[rank, expert_id].item()) < 0
        ]
        if not candidate_ranks:
            break
        rotate_start = (primary_rank + 1 + replica_round) % world_size
        candidate_ranks.sort(
            key=lambda rank: (next_local_ids[rank], (rank - rotate_start) % world_size)
        )
        target_rank = candidate_ranks[0]
        expert_map_all[target_rank, expert_id] = next_local_ids[target_rank]
        next_local_ids[target_rank] += 1

    return expert_map_all


def determine_default_expert_map(global_expert_num, world_size, rank_id, global_redundant_expert_num):
    if world_size == 1:
        local_ids = torch.arange(global_expert_num, dtype=torch.int32)
        return global_expert_num, local_ids

    local_num_experts = global_expert_num // world_size
    expert_map = torch.full((global_expert_num,), -1, dtype=torch.int32)

    if rank_id < world_size - 1:
        start = rank_id * local_num_experts
        end = (rank_id + 1) * local_num_experts
        local_count = local_num_experts
    else:
        start = rank_id * local_num_experts
        end = global_expert_num
        local_count = global_expert_num - rank_id * local_num_experts

    if isinstance(global_redundant_expert_num, int) and rank_id < global_redundant_expert_num:
        local_count += 1
        if end < global_expert_num:
            end += 1
        else:
            start -= 1

    local_ids = torch.arange(local_count, dtype=torch.int32)
    expert_map[start:end] = local_ids
    return local_count, expert_map


def determine_redundant_replica_expert_map(global_expert_num, world_size, rank_id, global_redundant_expert_num):
    expert_map_all = build_redundant_replica_expert_map(
        global_expert_num, world_size, global_redundant_expert_num
    )
    local_count = int((expert_map_all[rank_id] != -1).sum().item())
    return local_count, expert_map_all[rank_id]


def generate_redundant_log2phy_map(expert_map):
    num_local_experts = expert_map.max() + 1
    log2phy_map = expert_map.clone()
    num_ranks, num_global_expert = log2phy_map.shape

    row_indices = torch.arange(num_ranks).view(-1, 1).expand(num_ranks, num_global_expert) * num_local_experts
    log2phy_map[log2phy_map != -1] += row_indices[log2phy_map != -1]

    for idx in range(num_global_expert):
        positive_rank_idx = torch.where(log2phy_map[:, idx] != -1)[0]
        negative_rank_idx = torch.where(log2phy_map[:, idx] == -1)[0]
        num_rank_holding_expert = positive_rank_idx.size(0)

        if num_rank_holding_expert == 0:
            log2phy_map[:, idx] = torch.full((num_ranks,), 0, dtype=log2phy_map.dtype)
            continue

        if num_rank_holding_expert == 1:
            log2phy_map[negative_rank_idx, idx] = torch.full(
                (num_ranks - 1,),
                log2phy_map[positive_rank_idx, idx].item(),
                dtype=log2phy_map.dtype,
            )
            continue

        try:
            random_list = [
                random.choice(log2phy_map[positive_rank_idx, idx]).item()
                for _ in range(num_ranks - num_rank_holding_expert)
            ]
            log2phy_map[negative_rank_idx, idx] = torch.tensor(random_list, dtype=log2phy_map.dtype)
        except Exception as exc:
            logger.error("Fail to get log2phy_map: %s", exc)

    return log2phy_map


def determine_default_log2phy_map(global_expert_num, world_size, rank_id, global_redundant_expert_num):
    if world_size == 1:
        local_ids = torch.arange(global_expert_num, dtype=torch.int32)
        expert_map_all = local_ids.unsqueeze(0).expand(world_size, -1)
        log2phy_map_all = generate_redundant_log2phy_map(expert_map_all)
        return log2phy_map_all[rank_id]

    local_num_experts = global_expert_num // world_size
    expert_map_all = torch.full((world_size, global_expert_num), -1, dtype=torch.int32)

    for rank in range(world_size):
        if rank < world_size - 1:
            start = rank * local_num_experts
            end = (rank + 1) * local_num_experts
            local_count = local_num_experts
        else:
            start = rank * local_num_experts
            end = global_expert_num
            local_count = global_expert_num - rank * local_num_experts

        if isinstance(global_redundant_expert_num, int) and rank_id < global_redundant_expert_num:
            local_count += 1
            if end < global_expert_num:
                end += 1
            else:
                start -= 1

        local_ids = torch.arange(local_count, dtype=torch.int32)
        expert_map_all[rank, start:end] = local_ids

    log2phy_map_all = generate_redundant_log2phy_map(expert_map_all)
    return log2phy_map_all[rank_id]


def determine_redundant_replica_log2phy_map(global_expert_num, world_size, rank_id, global_redundant_expert_num):
    expert_map_all = build_redundant_replica_expert_map(
        global_expert_num, world_size, global_redundant_expert_num
    )
    log2phy_map_all = generate_redundant_log2phy_map(expert_map_all)
    return log2phy_map_all[rank_id]
