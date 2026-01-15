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
set -o pipefail

echo "Applying patches in numerical order..."

find ./patches -type f -name "*.patch" | \
sort -V | \

## apply patch
while IFS= read -r PATCH_FILE; do
    # skip empty lines
    [[ -z "$PATCH_FILE" ]] && continue
    PATCH_REL_PATH=$(realpath --relative-to=. "$PATCH_FILE")
    
    echo -n "Applying $PATCH_REL_PATH ... "

    git apply --ignore-whitespace "$PATCH_REL_PATH"

    if [ $? -ne 0 ]; then
        echo "[FAIL]: $PATCH_REL_PATH" >&2
        exit 1
    fi
    echo "[SUCCESS]: $PATCH_REL_PATH"
done
