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

from mathruler.grader import extract_boxed_content


def extract_solution(solution_str, ground_truth):
    extract_output = extract_boxed_content(solution_str)
    if ground_truth in solution_str or extract_output == ground_truth:
        return 1.0
    return 0.0


def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    """The scoring function for deepscaler.

    Reference: Trung, Luong
    Args:
        solution_str (str): the solution text
        ground_truth (str): the ground truth
        method: the method to use to compute the score. Defaults to "strict".
        format_score: the format of the score. Defaults to 0.0.
        score: the score. Defaults to 1.0.
    """
    score = extract_solution(solution_str=solution_str, ground_truth=ground_truth)
    return score
