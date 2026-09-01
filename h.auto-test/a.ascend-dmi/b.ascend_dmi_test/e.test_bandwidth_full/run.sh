#!/bin/bash
set -e

cd "$(dirname "$0")"

echo "开始执行 H2D + D2H 双向带宽测试"
source /usr/local/Ascend/toolbox/set_env.sh
ascend-dmi --bw -q
