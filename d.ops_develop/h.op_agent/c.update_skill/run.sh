#!/bin/bash
# ============================================================
# ⑧-C 联网获取 / 更新 Skill（官方 Ascend/agent-skills 总仓）
# 有网络时，从官方总仓拉取最新 skill 并安装到 opencode。
#
# 说明:
#   - 总仓 Ascend/agent-skills 是官方 skill 分发市场，覆盖
#     CANNBot / PyTorch / MindStudio / vllm-ascend / verl 等全模块
#   - 通过 Agent Skills 开放标准 CLI (npx skills) 安装
#   - 需要 Node.js 18+ 与网络（gitcode 可达）
#
# 用法:
#   bash c.update_skill/run.sh                # 安装全部最新 skill
#   bash c.update_skill/run.sh <skill名>      # 安装指定 skill
#   bash c.update_skill/run.sh list           # 列出可安装 skill
#
# 环境变量:
#   SKILL_REPO    # 总仓地址，默认 https://gitcode.com/Ascend/agent-skills.git
# ============================================================
set -euo pipefail

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; WHITE='\033[1;37m'; RESET='\033[0m'

SKILL_REPO="${SKILL_REPO:-https://gitcode.com/Ascend/agent-skills.git}"
CMD="${1:-}"

# 前置检查：npx
check_npx() {
    if ! command -v npx >/dev/null 2>&1; then
        echo -e "${RED}[ERROR] 未检测到 npx，需要 Node.js 18+。${RESET}" >&2
        echo "  离线环境请改用 b.skill_store/run.sh（离线选装）" >&2
        return 1
    fi
    return 0
}

echo -e "${CYAN}==> 联网获取 / 更新 Skill${RESET}"
echo "    总仓 : $SKILL_REPO"
echo ""

check_npx || exit 1

case "$CMD" in
    list)
        echo "==> 列出官方总仓可安装 skill（可能需要一点时间）"
        npx -y skills add "$SKILL_REPO" --list
        ;;
    "")
        echo "==> 安装全部最新 skill 到 opencode"
        echo -e "${YELLOW}    注意：这会安装全部模块（含训练/集群等可能用不到的）${RESET}"
        echo ""
        if ! npx -y skills add "$SKILL_REPO" --skill '*' --agent '*' -y; then
            echo ""
            echo -e "${RED}===== 安装失败 =====${RESET}"
            echo "可能原因："
            echo "  1. 网络无法访问 gitcode（连接超时 / 418 被风控）"
            echo "  2. 缺少 Node.js 18+"
            echo ""
            echo -e "${YELLOW}建议：离线环境请用 b.skill_store/run.sh${RESET}"
            exit 1
        fi
        echo ""
        echo -e "${GREEN}[OK] 已安装官方最新 skill。${RESET}"
        ;;
    *)
        echo "==> 安装指定 skill: $CMD"
        if ! npx -y skills add "$SKILL_REPO" --skill "$CMD" --agent opencode -y; then
            echo ""
            echo -e "${RED}===== 安装失败 =====${RESET}"
            echo "可能原因：网络无法访问 gitcode / skill 名不存在"
            echo -e "${YELLOW}可运行: $0 list 查看可安装 skill 名${RESET}"
            exit 1
        fi
        echo ""
        echo -e "${GREEN}[OK] 已安装: $CMD${RESET}"
        ;;
esac