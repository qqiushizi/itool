#!/bin/bash
# ============================================================
# ⑧-B 安装 CANNBot skills
# 优先使用官方 CANNBot 安装助手 @cannbot-ai/install-helper；
# 也支持官方 install.sh 引导；网络不可用时自动回退到绿色包内置 skills。
#
# 环境变量:
#   CANNBOT_TOOL=opencode|claude|trae|cursor|...     # 默认 opencode
#   CANNBOT_LEVEL=project|global                     # 默认 global
#   CANNBOT_USE_BUNDLE=1                             # 强制使用本地绿色包 skills
#   CANNBOT_ARGS="--all"                             # 额外参数
# ============================================================
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd || echo "$PWD")
A_INSTALL_DIR="$SCRIPT_DIR/../a.install_opencode"
TARBALL="$A_INSTALL_DIR/opencode.tar.gz"

TOOL="${CANNBOT_TOOL:-opencode}"
LEVEL="${CANNBOT_LEVEL:-global}"
ARGS="${CANNBOT_ARGS:---all}"
USE_BUNDLE="${CANNBOT_USE_BUNDLE:-0}"

CONFIG_DST="${XDG_CONFIG_HOME:-$HOME/.config}/opencode"

echo "==> CANNBot skills 安装"
echo "    目标工具 : $TOOL"
echo "    安装级别 : $LEVEL"
echo ""

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

bundled_install() {
    echo "==> 使用绿色包内置 CANNBot skills/agents"
    # 定位 opencode 绿色包
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
        echo "[ERROR] 未找到绿色包中的 CANNBot skills。" >&2
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
        echo "[WARN] 未在 $CONFIG_DST/skills 下检测到 skills。" >&2
        return 1
    fi
}

if [ "$USE_BUNDLE" = "1" ]; then
    echo "==> 已设置 CANNBOT_USE_BUNDLE=1，强制使用绿色包内置 skills"
    bundled_install
elif official_install_helper; then
    :
else
    echo ""
    echo "[WARN] 官方 CANNBot installer 安装失败或不可用，回退到绿色包内置 skills。"
    bundled_install
fi

verify_install

MANIFEST="$CONFIG_DST/cannbot-manifest.json"
brand=$(python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); print(d.get("brand","CANNBot"))' "$MANIFEST" 2>/dev/null || echo CANNBot)
version=$(python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); print(d.get("version","?"))' "$MANIFEST" 2>/dev/null || echo '?')
echo "    品牌        : $brand"
echo "    版本        : $version"
echo ""
echo "[OK] CANNBot skills 安装完成。"
