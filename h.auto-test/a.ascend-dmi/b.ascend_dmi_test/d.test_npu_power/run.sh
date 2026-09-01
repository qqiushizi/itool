#!/bin/bash
set -e

cd "$(dirname "$0")"

echo "开始执行 NPU 温度/电压/功耗监控"
source /usr/local/Ascend/toolbox/set_env.sh
ascend-dmi -p -q
