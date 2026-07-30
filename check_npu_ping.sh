#!/usr/bin/env bash
set -euo pipefail

HCCN_TOOL=/usr/local/Ascend/driver/tools/hccn_tool
NUM_NPU=${1:-8}

declare -a IPS

echo "===== Collect NPU HCCN IPs ====="

for i in $(seq 0 $((NUM_NPU - 1))); do
    echo "----- NPU $i -----"
    out=$($HCCN_TOOL -i "$i" -ip -g)
    echo "$out"

    ip=$(echo "$out" | grep -Eo '([0-9]{1,3}\.){3}[0-9]{1,3}' | head -n 1)

    if [ -z "${ip:-}" ]; then
        echo "[ERROR] Cannot parse IP for NPU $i"
        exit 1
    fi

    IPS[$i]=$ip
done

echo
echo "===== IP Summary ====="
for i in $(seq 0 $((NUM_NPU - 1))); do
    echo "NPU $i -> ${IPS[$i]}"
done

echo
echo "===== Pairwise Ping Test ====="

failed=0

for src in $(seq 0 $((NUM_NPU - 1))); do
    for dst in $(seq 0 $((NUM_NPU - 1))); do
        if [ "$src" -eq "$dst" ]; then
            continue
        fi

        echo
        echo ">>> NPU $src ping NPU $dst: ${IPS[$dst]}"

        if $HCCN_TOOL -i "$src" -ping -g address "${IPS[$dst]}"; then
            echo "[OK] NPU $src -> NPU $dst"
        else
            echo "[FAIL] NPU $src -> NPU $dst"
            failed=1
        fi
    done
done

echo
if [ "$failed" -eq 0 ]; then
    echo "===== All NPU HCCN ping tests passed ====="
else
    echo "===== Some NPU HCCN ping tests failed ====="
    exit 1
fi
