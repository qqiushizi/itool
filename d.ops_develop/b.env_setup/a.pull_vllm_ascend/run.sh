#!/bin/bash
# ============================================================
# 从 vllm-ascend 拉取 CANN 镜像 (用户指定 CANN 版本)
# 用法:
#   CANN_VERSION=8.1.rc1 CHIP=910b bash run.sh
#   IMAGE=quay.io/ascend/cann:xxx bash run.sh   # 完全自定义镜像
# 说明:
#   - 官方镜像命名: quay.io/ascend/cann:<版本>-<芯片>-<系统>-<python>
#     如: 8.1.rc1-910b-ubuntu22.04-py3.10
#   - 可查全部 tag: curl -s 'https://quay.io/v2/ascend/cann/tags/list'
#   - 拉取后额外打本地短标签 cann-<芯片>:<版本>, 便于 b.run_container 使用。
# ============================================================
set -euo pipefail

command -v docker >/dev/null 2>&1 || { echo -e "\033[0;31m未找到 docker, 请先安装。\033[0m" >&2; exit 1; }

CANN_VERSION="${1:-${CANN_VERSION:-8.1.rc1}}"
CHIP="${CHIP:-910b}"
OS_TAG="${OS_TAG:-ubuntu22.04}"
PY_TAG="${PY_TAG:-py3.10}"

# 官方镜像(可整体用 IMAGE 覆盖)
IMAGE="${IMAGE:-quay.io/ascend/cann:${CANN_VERSION}-${CHIP}-${OS_TAG}-${PY_TAG}}"
LOCAL_TAG="cann-${CHIP}:${CANN_VERSION}"

echo "=========================================================="
echo " 拉取 vllm-ascend CANN 镜像"
echo "   镜像: $IMAGE"
echo "   本地标签: $LOCAL_TAG"
echo "=========================================================="

docker pull "$IMAGE"
docker tag "$IMAGE" "$LOCAL_TAG"

echo ""
echo -e "\033[0;32m镜像已就绪:\033[0m"
docker images | grep -E "ascend/cann|$LOCAL_TAG" || true
echo ""
echo "下一步实例化容器:"
echo "  bash d.ops_develop/b.env_setup/b.run_container/run.sh $LOCAL_TAG my_ascend"
