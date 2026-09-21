#!/bin/bash
# ============================================================
# ⑧-C 联网更新 skill_store 仓库
# 有网络时，从官方总仓 Ascend/agent-skills 拉取最新 skill，
# 重新构建 b.skill_store/skills.tar.gz。
#
# 说明:
#   - 本功能更新的是「仓库」本身（skills.tar.gz），不是直接装进 opencode
#   - 更新后，用户再用 b.skill_store/run.sh 选装最新 skill
#   - 需要 git + python3 + 网络（gitcode 可达）
#
# 用法:
#   bash c.update_skill/run.sh          # 拉最新总仓并重打包
#
# 环境变量:
#   SKILL_REPO   # 总仓地址，默认 https://gitcode.com/Ascend/agent-skills.git
# ============================================================
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd || echo "$PWD")
STORE_DIR="$SCRIPT_DIR/../b.skill_store"
STORE_TAR="$STORE_DIR/skills.tar.gz"
BUILD_PY="$SCRIPT_DIR/build_store.py"

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; RESET='\033[0m'

SKILL_REPO="${SKILL_REPO:-https://gitcode.com/Ascend/agent-skills.git}"
WORK_DIR="$SCRIPT_DIR/.agent-skills-cache"

echo -e "${CYAN}==> 联网更新 skill_store 仓库${RESET}"
echo "    总仓 : $SKILL_REPO"
echo ""

# 前置检查
for c in git python3; do
    if ! command -v "$c" >/dev/null 2>&1; then
        echo -e "${RED}[ERROR] 缺少命令: $c${RESET}" >&2
        exit 1
    fi
done

# 拉取/更新总仓
if [ -d "$WORK_DIR/.git" ]; then
    echo "==> 更新已存在的总仓缓存..."
    if ! git -C "$WORK_DIR" pull --ff-only --quiet; then
        echo -e "${YELLOW}[WARN] pull 失败，尝试重新 clone${RESET}"
        rm -rf "$WORK_DIR"
    fi
fi
if [ ! -d "$WORK_DIR/.git" ]; then
    echo "==> 浅克隆官方总仓（含子模块）..."
    if ! git clone --depth 1 --recurse-submodules --quiet "$SKILL_REPO" "$WORK_DIR"; then
        echo ""
        echo -e "${RED}===== 拉取总仓失败 =====${RESET}"
        echo "可能原因：网络无法访问 gitcode（超时/418 被风控）"
        echo -e "${YELLOW}离线环境请直接用 b.skill_store/run.sh 选装${RESET}"
        exit 1
    fi
fi

# 重打包
echo "==> 重新构建 skills.tar.gz..."
if ! python3 "$BUILD_PY" "$WORK_DIR" "$STORE_TAR"; then
    echo -e "${RED}[ERROR] 打包失败${RESET}" >&2
    exit 1
fi

echo ""
echo -e "${GREEN}[OK] skill_store 仓库已更新。${RESET}"
echo "  下一步: bash $STORE_DIR/run.sh  选装最新 skill"