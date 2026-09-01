#!/bin/bash
set -e

cd "$(dirname "$0")"

echo "开始执行 Device-to-Host 带宽测试"
source /usr/local/Ascend/toolbox/set_env.sh
ascend-dmi --bw -t d2h -q
