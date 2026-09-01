#!/bin/bash
set -e

cd "$(dirname "$0")"

echo "开始执行 P2P 带宽测试"
source /usr/local/Ascend/toolbox/set_env.sh
ascend-dmi --bw -t p2p -ds 0 -dd 1 -q

