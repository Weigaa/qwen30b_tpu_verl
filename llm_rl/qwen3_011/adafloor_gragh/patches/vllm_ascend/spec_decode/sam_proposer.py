# Copyright (c) 2025 Huawei Technologies Co., Ltd.	
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

import os
import torch
import numpy as np
from patches.vllm.v1.spec_decode.sam import SAM
from vllm_ascend.spec_decode.interface import Proposer, SpecDcodeType


class SAMProposer(Proposer):
    def __init__(self, vllm_config, device, runner):
        self.n_predicts = vllm_config.speculative_config.num_speculative_tokens
        self.all_proposers: dict[int, SAM] = {}
        self.name = SpecDcodeType.SAM
        self.device = device
        self.runner = runner

    def propose(self,
                request_id: int,
                old_token_ids,
                new_token_ids,
                num_sampled_ids):
        if self.all_proposers.get(request_id, None) is None:
            self.all_proposers[request_id] = SAM(n_predicts=self.n_predicts)
            self.all_proposers[request_id].add_tokens(old_token_ids)
        if num_sampled_ids > 1:
            self.all_proposers[request_id].add_tokens(new_token_ids[:-1])
        query_token = new_token_ids[-1]
        index_dyn, _ = self.all_proposers[request_id].lookup(query_token)
        seq = self.all_proposers[request_id].gen_draft(index_dyn, query_token)
        self.all_proposers[request_id].add_tokens(new_token_ids[-1:])
        return np.array(seq)

    def load_model(self, *args, **kwargs):
        pass

    @torch.inference_mode()
    def dummy_run(self, *args, **kwargs):
        pass

    def generate_token_ids(self, valid_sampled_token_ids, *args, **kwargs):
        draft_token_ids = []
        for i, sampled_ids in enumerate(valid_sampled_token_ids):
            num_sampled_ids = len(sampled_ids)
            if not num_sampled_ids:
                # Skip speculative decoding.
                draft_token_ids.append([])
                continue

            # Skip requests that require top-p, top-k, etc.
            req_id = self.runner.input_batch.req_ids[i]
            if req_id in self.runner.input_batch.spec_decode_unsupported_reqs:
                draft_token_ids.append([])
                continue

            # Add sampled_token_ids to token_ids_cpu.
            end_idx = self.runner.input_batch.num_tokens_no_spec[i]
            start_idx = end_idx - num_sampled_ids

            drafter_output = self.propose(
                req_id,
                self.runner.input_batch.token_ids_cpu[i, :start_idx],
                self.runner.input_batch.token_ids_cpu[i, start_idx:end_idx],
                num_sampled_ids
            )

            if drafter_output is None or len(drafter_output) == 0:
                draft_token_ids.append([])
            else:
                draft_token_ids.append(drafter_output.tolist())
        return draft_token_ids
