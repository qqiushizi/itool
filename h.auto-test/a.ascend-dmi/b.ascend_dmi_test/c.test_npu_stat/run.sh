#!/bin/bash
set -e

cd "$(dirname "$0")"

echo 开始执行 NPU 状态检查
source /usr/local/Ascend/toolbox/set_env.sh
ascend-dmi --info
