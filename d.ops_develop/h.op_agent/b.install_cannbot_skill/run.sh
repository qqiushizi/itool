#!/bin/bash
# ============================================================
# ⑧-B 检查 / 更新 CANNBot skills
# 检查当前已安装的 CANNBot 版本，若官方有更新则升级到最新。
#
# 说明:
#   - CANNBot 官方为滚动更新(按日期迭代，无固定版本号文件)，
#     因此"检查是否有新版"直接交给官方 install-helper update，
#     由它比对并决定是否需要升级。
#   - 首次使用(尚未安装)会先走完整安装，再进入更新检查。
#   - 网络不可用/无 install-helper 时，可回退到绿色包离线安装；
#     此时无法检查官方最新版，仅提示当前版本。
#
# 环境变量:
#   CANNBOT_TOOL=opencode|claude|trae|cursor|...     # 默认 opencode
#   CANNBOT_LEVEL=project|global                     # 默认 global
#   CANNBOT_USE_BUNDLE=1                             # 强制使用本地绿色包 skills
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
USE_BUNDLE="${CANNBOT_USE_BUNDLE:-0}"

CONFIG_DST="${XDG_CONFIG_HOME:-$HOME/.config}/opencode"
MANIFEST="$CONFIG_DST/cannbot-manifest.json"

# 读取当前已安装版本；结果放 CURRENT_VERSION（未安装则为空）
detect_current_version() {
    CURRENT_VERSION=""
    if [ -f "$MANIFEST" ]; then
        CURRENT_VERSION=$(python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); print(d.get("version","?"))' "$MANIFEST" 2>/dev/null || echo '')
    fi
}

# 是否已安装（有 skills 目录且非空即视为已装）
is_installed() {
    [ -d "$CONFIG_DST/skills" ] && [ -n "$(find "$CONFIG_DST/skills" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null || true)" ]
}

# 官方 install-helper 安装（首次安装用）
official_install_helper() {
    if command -v install-helper >/dev/null 2>&1; then
        install-helper install ${ARGS} --tool "$TOOL" --level "$LEVEL" --yes
        return $?
    fi
    if command -v npx >/dev/null 2>&1; then
        npx -y @cannbot-ai/install-helper install ${ARGS} --tool "$TOOL" --level "$LEVEL" --yes
        return $?
    fi
    echo "==> 未检测到 install-helper / npx，使用官方 install.sh 引导安装"
    curl -fsSL https://raw.gitcode.com/cann/cannbot-skills/raw/master/install.sh | bash
}

# 官方 install-helper 更新（检查并升级到最新）
official_install_helper_update() {
    if command -v install-helper >/dev/null 2>&1; then
        install-helper update --tool "$TOOL" --level "$LEVEL" --yes
        return $?
    fi
    if command -v npx >/dev/null 2>&1; then
        npx -y @cannbot-ai/install-helper update --tool "$TOOL" --level "$LEVEL" --yes
        return $?
    fi
    return 1
}

bundled_install() {
    echo "==> 使用绿色包内置 CANNBot skills/agents（离线快照）"
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

verify_install() {
    echo ""
    echo "==> 校验安装结果"
    if [ -d "$CONFIG_DST/skills" ]; then
        N=$(find "$CONFIG_DST/skills" -mindepth 1 -maxdepth 1 | wc -l | tr -d ' ')
    else
        N=0
    fi
    A=$(find "$CONFIG_DST/agents" -mindepth 1 -maxdepth 1 2>/dev/null | wc -l | tr -d ' ')
    echo "    skills 数量: $N"
    echo "    agents 数量: $A"
    if [ "$N" -eq 0 ]; then
        echo -e "${RED}[WARN] 未在 $CONFIG_DST/skills 下检测到 skills。${RESET}" >&2
        return 1
    fi
}

print_manifest() {
    brand=$(python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); print(d.get("brand","CANNBot"))' "$MANIFEST" 2>/dev/null || echo CANNBot)
    version=$(python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); print(d.get("version","?"))' "$MANIFEST" 2>/dev/null || echo '?')
    echo "    品牌        : $brand"
    echo "    版本        : $version"
}

# ============================================================
# 主流程
# ============================================================
echo -e "${CYAN}==> CANNBot skills 检查 / 更新${RESET}"
echo "    目标工具 : $TOOL"
echo "    安装级别 : $LEVEL"
echo ""

# 强制绿色包：不检查版本，直接离线安装
if [ "$USE_BUNDLE" = "1" ]; then
    echo "==> 已设置 CANNBOT_USE_BUNDLE=1，强制使用绿色包离线安装"
    bundled_install
    verify_install
    print_manifest
    echo ""
    echo "==> 已离线安装绿色包快照；无法检查官方最新版。"
    exit 0
fi

detect_current_version

if ! is_installed; then
    echo -e "${YELLOW}==> 尚未安装 CANNBot skills，先执行完整安装${RESET}"
    if ! official_install_helper; then
        echo ""
        echo -e "${YELLOW}[WARN] 官方 installer 安装失败或不可用，回退到绿色包内置 skills。${RESET}"
        bundled_install
    fi
    verify_install
    detect_current_version
    echo ""
    echo -e "${GREEN}[OK] 首次安装完成。${RESET}"
    print_manifest
    exit 0
fi

# 已安装：显示当前版本，再尝试官方更新
echo "==> 已安装 CANNBot skills，当前版本:"
print_manifest
echo ""

if official_install_helper_update; then
    detect_current_version
    echo ""
    echo -e "${GREEN}[OK] 更新检查完成，当前版本:${RESET}"
    print_manifest
else
    echo ""
    echo -e "${YELLOW}[提示] 未检测到 install-helper / npx，无法在线检查官方最新版。${RESET}"
    echo "        当前为本地已安装版本；如需离线兜底可运行:"
    echo "          CANNBOT_USE_BUNDLE=1 $0"
    print_manifest
fi

echo ""
echo "==> 校验安装结果"
verify_install