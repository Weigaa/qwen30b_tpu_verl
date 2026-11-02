# Copyright (c) 2025, HUAWEI CORPORATION.  All rights reserved.
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

_BATCH_HDP_GROUP = None


def get_batch_hdp_group():
    global _BATCH_HDP_GROUP
    return _BATCH_HDP_GROUP


def set_batch_hdp_group(batch_hdp_group):
    global _BATCH_HDP_GROUP
    _BATCH_HDP_GROUP = batch_hdp_group


def clean_hdp_group():
    global _BATCH_HDP_GROUP
    _BATCH_HDP_GROUP = None
