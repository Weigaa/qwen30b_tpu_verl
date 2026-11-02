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

#!/bin/bash
set -ex

FRAMEWORK_ROOT=${QWEN3_FRAMEWORK_ROOT:-/workspace}

rm -rf ./verl ./r1_ascend ./converter_hf_to_mcore.py ./megatron ./mindspeed ./vllm ./vllm_ascend

cp -r "${FRAMEWORK_ROOT}/verl/verl" ./
cp -r "${FRAMEWORK_ROOT}/verl/recipe/r1_ascend" ./
cp "${FRAMEWORK_ROOT}/verl/scripts/converter_hf_to_mcore.py" ./
cp "${FRAMEWORK_ROOT}/verl/recipe/dapo/config/"* ./verl/trainer/config/
cp "${FRAMEWORK_ROOT}/verl/recipe/dapo/"*py ./verl/trainer/
mkdir -p ./megatron
cp -r "${FRAMEWORK_ROOT}/Megatron-LM/megatron/core" ./megatron/core
cp -r "${FRAMEWORK_ROOT}/MindSpeed/mindspeed" ./
cp -r "${FRAMEWORK_ROOT}/vllm/vllm" ./
cp -r "${FRAMEWORK_ROOT}/vllm-ascend/vllm_ascend" ./

ls -l
