#!/bin/bash
# ============================================================
# ⑧-B 安装最新 CANNBot skills（失败即报错，不自动兜底）
#
# 目标：把官方最新 CANNBot skills/agents 安装到 opencode。
# 若安装失败，明确说明原因（网络 / 环境），不做任何离线兜底，
# 避免让用户误以为装到的是最新版。
#
# 环境变量:
#   CANNBOT_TOOL=opencode|claude|trae|cursor|...     # 默认 opencode
#   CANNBOT_LEVEL=project|global                     # 默认 global
#   CANNBOT_USE_BUNDLE=1                             # 显式转为离线快照安装(旧版, 非最新)
#   CANNBOT_ARGS="--all"                             # 传给 install-helper 的额外参数
# ============================================================
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd || echo "$PWD")
A_INSTALL_DIR="$SCRIPT_DIR/../a.install_opencode"
TARBALL="$A_INSTALL_DIR/opencode.tar.gz"

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; RESET='\033[0m'

TOOL="${CANNBOT_TOOL:-opencode}"
LEVEL="${CANNBOT_LEVEL:-global}"
ARGS="${CANNBOT_ARGS:---all}"

CONFIG_DST="${XDG_CONFIG_HOME:-$HOME/.config}/opencode"
MANIFEST="$CONFIG_DST/cannbot-manifest.json"

# 是否已安装（有 skills 目录且非空即视为已装）
is_installed() {
    [ -d "$CONFIG_DST/skills" ] && [ -n "$(find "$CONFIG_DST/skills" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null || true)" ]
}

detect_current_version() {
    CURRENT_VERSION=""
    if [ -f "$MANIFEST" ]; then
        CURRENT_VERSION=$(python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); print(d.get("version","?"))' "$MANIFEST" 2>/dev/null || echo '')
    fi
}

print_manifest() {
    brand=$(python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); print(d.get("brand","CANNBot"))' "$MANIFEST" 2>/dev/null || echo CANNBot)
    version=$(python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); print(d.get("version","?"))' "$MANIFEST" 2>/dev/null || echo '?')
    echo "    品牌        : $brand"
    echo "    版本        : $version"
}

verify_install() {
    if [ -d "$CONFIG_DST/skills" ]; then
        N=$(find "$CONFIG_DST/skills" -mindepth 1 -maxdepth 1 | wc -l | tr -d ' ')
    else
        N=0
    fi
    A=$(find "$CONFIG_DST/agents" -mindepth 1 -maxdepth 1 2>/dev/null | wc -l | tr -d ' ')
    echo "    skills 数量: $N"
    echo "    agents 数量: $A"
    [ "$N" -gt 0 ]
}

# 安装官方最新版。返回 0 成功 / 非 0 失败。
install_latest() {
    # 1) 本机已有 install-helper
    if command -v install-helper >/dev/null 2>&1; then
        if is_installed; then
            echo "==> 检测到 install-helper，执行更新到最新"
            install-helper update --tool "$TOOL" --level "$LEVEL" --yes
        else
            echo "==> 检测到 install-helper，执行完整安装"
            install-helper install ${ARGS} --tool "$TOOL" --level "$LEVEL" --yes
        fi
        return $?
    fi

    # 2) 有 npx（Node 18+）
    if command -v npx >/dev/null 2>&1; then
        echo "==> 检测到 npx，通过 @cannbot-ai/install-helper 安装最新"
        if is_installed; then
            npx -y @cannbot-ai/install-helper update --tool "$TOOL" --level "$LEVEL" --yes
        else
            npx -y @cannbot-ai/install-helper install ${ARGS} --tool "$TOOL" --level "$LEVEL" --yes
        fi
        return $?
    fi

    # 3) 无 install-helper / npx：尝试用官方 install.sh 引导安装 helper（需联网）
    echo "==> 未检测到 install-helper / npx，尝试通过官方 install.sh 安装 helper"
    if ! curl -fsSL --max-time 60 https://raw.gitcode.com/cann/cannbot-skills/raw/master/install.sh | bash; then
        echo -e "${RED}[ERROR] 官方 install.sh 拉取/执行失败。${RESET}" >&2
        return 1
    fi
    if command -v install-helper >/dev/null 2>&1; then
        install-helper install ${ARGS} --tool "$TOOL" --level "$LEVEL" --yes
        return $?
    fi
    echo -e "${RED}[ERROR] install.sh 执行后仍未找到 install-helper。${RESET}" >&2
    return 1
}

# 显式离线快照安装（仅 CANNBOT_USE_BUNDLE=1 时使用，非默认行为）
bundled_install() {
    echo "==> 使用绿色包内置 CANNBot skills/agents（离线快照，非最新）"
    OPCODE_ROOT="${OPCODE_ROOT:-}"
    if [ -n "$OPCODE_ROOT" ] && [ -x "$OPCODE_ROOT/bin/opencode" ]; then
        :
    elif [ -x "$A_INSTALL_DIR/opencode/bin/opencode" ]; then
        OPCODE_ROOT="$A_INSTALL_DIR/opencode"
    elif [ -f "$TARBALL" ]; then
        echo "==> 解压 opencode 绿色包"
        tar -xzf "$TARBALL" -C "$A_INSTALL_DIR"
        OPCODE_ROOT="$A_INSTALL_DIR/opencode"
    elif [ -x "$SCRIPT_DIR/../../../f.installation/b.install_coder/b.install_opencode/opencode/bin/opencode" ]; then
        OPCODE_ROOT="$SCRIPT_DIR/../../../f.installation/b.install_coder/b.install_opencode/opencode"
    fi

    if [ -z "$OPCODE_ROOT" ] || [ ! -d "$OPCODE_ROOT/config/opencode" ]; then
        echo -e "${RED}[ERROR] 未找到绿色包中的 CANNBot skills。${RESET}" >&2
        return 1
    fi

    CONFIG_SRC="$OPCODE_ROOT/config/opencode"
    mkdir -p "$(dirname "$CONFIG_DST")"
    if [ -L "$CONFIG_DST" ]; then
        echo "==> 用户配置目录已是链接: $(readlink "$CONFIG_DST")"
    elif [ -e "$CONFIG_DST" ]; then
        echo "==> 用户配置目录已存在，保留原内容并安装 skills/agents"
        mkdir -p "$CONFIG_DST/skills" "$CONFIG_DST/agents"
        for item in "$CONFIG_SRC"/AGENTS.md "$CONFIG_SRC"/opencode.json "$CONFIG_SRC"/cannbot-manifest.json; do
            [ -e "$item" ] || continue
            name=$(basename "$item")
            dst="$CONFIG_DST/$name"
            if [ -e "$dst" ] || [ -L "$dst" ]; then continue; fi
            ln -s "$item" "$dst"
        done
        for sub in skills agents; do
            for item in "$CONFIG_SRC/$sub"/*; do
                [ -e "$item" ] || continue
                name=$(basename "$item")
                dst="$CONFIG_DST/$sub/$name"
                if [ -e "$dst" ] || [ -L "$dst" ]; then continue; fi
                ln -s "$item" "$dst"
            done
        done
    else
        echo "==> 链接整套 opencode 配置（含 skills/agents）"
        ln -s "$CONFIG_SRC" "$CONFIG_DST"
    fi
    return 0
}

# 失败原因说明
explain_failure() {
    echo ""
    echo -e "${RED}===== 安装最新 CANNBot 失败 =====${RESET}"
    echo ""
    echo "未安装到官方最新版，现有配置保持不变（如有）。"
    echo ""
    echo "可能原因："
    echo "  1. 网络无法访问官方源（gitcode.com / npm registry，连接超时或被墙）"
    echo "     → 请确认网络或代理，稍后重试"
    echo "  2. 缺少 Node.js 18+（无 npx 命令）"
    echo "     → 安装 Node 18+ 后重试"
    echo "  3. 缺少 install-helper 且无法自动下载安装"
    echo "     → 参考官方：curl -fsSL https://raw.gitcode.com/cann/cannbot-skills/raw/master/install.sh | bash"
    echo ""
    echo -e "${YELLOW}注意：本脚本不会自动回退到旧快照。${RESET}"
    echo "如需显式使用仓库内置离线快照（v1.1.0，旧版），请手动执行："
    echo "  CANNBOT_USE_BUNDLE=1 $0"
    echo ""
}

# ============================================================
# 主流程
# ============================================================

# 显式离线模式：仅当用户主动设置 CANNBOT_USE_BUNDLE=1 时才走
if [ "${CANNBOT_USE_BUNDLE:-0}" = "1" ]; then
    echo -e "${YELLOW}==> 显式离线快照模式${RESET}"
    echo "    注意：这是仓库内置的 v1.1.0 旧快照，非官方最新版。"
    echo ""
    if bundled_install && verify_install; then
        print_manifest
        echo ""
        echo -e "${GREEN}[OK] 离线快照安装完成。${RESET}"
        exit 0
    else
        echo -e "${RED}[ERROR] 离线快照安装失败。${RESET}" >&2
        exit 1
    fi
fi

echo -e "${CYAN}==> CANNBot skills 安装（目标：官方最新版）${RESET}"
echo "    目标工具 : $TOOL"
echo "    安装级别 : $LEVEL"
echo ""

if is_installed; then
    detect_current_version
    echo "==> 已检测到本地安装，将尝试更新到最新，当前版本:"
    print_manifest
    echo ""
fi

if install_latest; then
    echo ""
    if verify_install; then
        print_manifest
        echo ""
        echo -e "${GREEN}[OK] 已安装官方最新版。${RESET}"
        exit 0
    else
        echo -e "${RED}[ERROR] 安装命令执行完成，但未检测到 skills，可能安装有误。${RESET}" >&2
        exit 1
    fi
else
    explain_failure
    exit 1
fi