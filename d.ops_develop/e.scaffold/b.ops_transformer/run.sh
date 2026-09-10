#!/bin/bash
# ============================================================
# 拉取 ops-transformer 源码 (CANN 进阶算子库, 厚重但完善)
# 用法:
#   bash run.sh [目标目录]
# 说明:
#   - 官方仓库: https://gitcode.com/cann/ops-transformer.git (GitCode)
#   - 可用 REPO 环境变量覆盖为内部镜像。
#   - 拉取后打印编译/安装指导。
# ============================================================
set -euo pipefail

CYAN='\033[0;36m'; GREEN='\033[0;32m'; RESET='\033[0m'

REPO="${REPO:-https://gitcode.com/cann/ops-transformer.git}"
DIR="${1:-${DIR:-ops-transformer}}"

command -v git >/dev/null 2>&1 || { echo "未找到 git。" >&2; exit 1; }

echo "=========================================================="
echo " 拉取 ops-transformer (CANN 进阶算子库)"
echo "   仓库: $REPO"
echo "   目录: $DIR"
echo "=========================================================="

if [ -d "$DIR/.git" ]; then
    echo "已存在, 执行 git pull..."
    (cd "$DIR" && git pull --ff-only)
else
    git clone "$REPO" "$DIR"
fi

echo ""
echo -e "${GREEN}✔ 源码已就绪: $DIR${RESET}"
echo ""
echo -e "${CYAN}===== 编译 / 安装指导 =====${RESET}"
cat <<GUIDE
1) 加载 CANN 环境:
     source /usr/local/Ascend/ascend-toolkit/set_env.sh

2) 安装依赖(会检测 cmake/gcc/python 依赖并自动补装):
     cd $DIR
     bash install_deps.sh

3) 编译(产物输出到 output/ 与 build_out/):
     bash build.sh
   # 可选编译目标: RELEASE_TARGETS=(ophost opapi opgraph onnxplugin)

4) 按仓库 docs/ 接入 torch_npu / vllm-ascend(见 docs/QUICKSTART.md)。
GUIDE
