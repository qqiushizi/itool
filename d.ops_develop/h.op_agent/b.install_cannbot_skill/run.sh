#!/bin/bash
# ============================================================
# ⑧-B 安装 CANNBot skills/agents
# 把 opencode 绿色包里的 CANNBot 算子开发技能集安装到用户配置目录。
# 运行前需要先由 a.install_opencode 完成解压，或设置 OPCODE_ROOT。
# 用法:
#   ./run.sh                      # 安装 opencode + CANNBot skills/agents
#   OPCODE_ROOT=/path/to/opencode ./run.sh
#   SKILL_ONLY=1 ./run.sh         # 只链接 skills/agents, 不装 opencode 可执行文件
#   PREFIX=$HOME/.local ./run.sh  # 自定义安装前缀
# ============================================================
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd || echo "$PWD")
A_INSTALL_DIR="$SCRIPT_DIR/../a.install_opencode"
TARBALL="$A_INSTALL_DIR/opencode.tar.gz"

resolve_opencode_root() {
    local root
    if [ -n "${OPCODE_ROOT:-}" ] && [ -x "$OPCODE_ROOT/bin/opencode" ]; then
        root="$OPCODE_ROOT"
    elif [ -x "$A_INSTALL_DIR/opencode/bin/opencode" ]; then
        root="$A_INSTALL_DIR/opencode"
    elif [ -x "$SCRIPT_DIR/../../../f.installation/b.install_coder/b.install_opencode/opencode/bin/opencode" ]; then
        root="$SCRIPT_DIR/../../../f.installation/b.install_coder/b.install_opencode/opencode"
    else
        root=""
    fi
    OPCODE_ROOT="$root"
}

ensure_extracted() {
    if [ -n "$OPCODE_ROOT" ]; then
        return 0
    fi
    if [ ! -f "$TARBALL" ]; then
        echo "[ERROR] 找不到 opencode 安装包: $TARBALL" >&2
        echo "        请先运行 a.install_opencode, 或通过 OPCODE_ROOT 指定 opencode 目录。" >&2
        exit 1
    fi
    echo "==> 解压 opencode 绿色包到: $A_INSTALL_DIR"
    tar -xzf "$TARBALL" -C "$A_INSTALL_DIR"
    resolve_opencode_root
}

resolve_opencode_root
ensure_extracted

if [ -z "$OPCODE_ROOT" ] || [ ! -x "$OPCODE_ROOT/bin/opencode" ]; then
    echo "[ERROR] 未找到 opencode 可执行文件。" >&2
    exit 1
fi

CONFIG_SRC="$OPCODE_ROOT/config/opencode"
[ -d "$CONFIG_SRC" ] || {
    echo "[ERROR] 未找到配置目录: $CONFIG_SRC" >&2
    exit 1
}

CONFIG_DST="${XDG_CONFIG_HOME:-$HOME/.config}/opencode"
echo "==> CANNBot skills/agents 安装"
echo "    opencode 根目录 : $OPCODE_ROOT"
echo "    skill 配置源     : $CONFIG_SRC"
echo "    用户配置目录     : $CONFIG_DST"
echo ""

# 安装 opencode 可执行文件（若需要）
if [ "${SKILL_ONLY:-0}" != "1" ]; then
    echo "==> 安装 opencode 可执行链接"
    cd "$OPCODE_ROOT"
    if [ "$(id -u)" -eq 0 ]; then
        export PREFIX="${PREFIX:-/usr/local}"
    else
        export PREFIX="${PREFIX:-$HOME/.local}"
    fi
    bash ./install.sh
    echo ""
else
    echo "==> 跳过 opencode 可执行文件安装（SKILL_ONLY=1）"
    echo ""
fi

# 确保用户配置目录存在并链入 CANNBot skills/agents
mkdir -p "$(dirname "$CONFIG_DST")"
if [ -L "$CONFIG_DST" ]; then
    echo "==> 用户配置目录已是链接: $(readlink "$CONFIG_DST")"
elif [ -e "$CONFIG_DST" ]; then
    echo "==> 用户配置目录已存在，将保留原内容并单独安装 CANNBot 配置"
    mkdir -p "$CONFIG_DST/skills" "$CONFIG_DST/agents"
    for item in "$CONFIG_SRC"/AGENTS.md "$CONFIG_SRC"/opencode.json "$CONFIG_SRC"/cannbot-manifest.json; do
        [ -e "$item" ] || continue
        name=$(basename "$item")
        dst="$CONFIG_DST/$name"
        if [ -e "$dst" ] || [ -L "$dst" ]; then
            continue
        fi
        ln -s "$item" "$dst"
    done
    for sub in skills agents; do
        for item in "$CONFIG_SRC/$sub"/*; do
            [ -e "$item" ] || continue
            name=$(basename "$item")
            dst="$CONFIG_DST/$sub/$name"
            if [ -e "$dst" ] || [ -L "$dst" ]; then
                continue
            fi
            ln -s "$item" "$dst"
        done
    done
else
    echo "==> 链接整套 opencode 配置（含 skills/agents）"
    ln -s "$CONFIG_SRC" "$CONFIG_DST"
fi

echo ""
echo "==> 校验安装结果"
if [ -d "$CONFIG_DST/skills" ]; then
    N=$(find "$CONFIG_DST/skills" -mindepth 1 -maxdepth 1 | wc -l | tr -d ' ')
else
    N=0
fi
A=$(find "$CONFIG_DST/agents" -mindepth 1 -maxdepth 1 2>/dev/null | wc -l | tr -d ' ')
MANIFEST="$CONFIG_DST/cannbot-manifest.json"
brand=$(python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); print(d.get("brand","CANNBot"))' "$MANIFEST" 2>/dev/null || echo CANNBot)
version=$(python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); print(d.get("version","?"))' "$MANIFEST" 2>/dev/null || echo '?')
echo "    品牌            : $brand"
echo "    版本            : $version"
echo "    skills 数量     : $N"
echo "    agents 数量     : $A"

echo ""
echo "[OK] CANNBot skills 安装完成。"
echo "     启动 opencode:"
if command -v opencode >/dev/null 2>&1; then
    echo "       opencode"
else
    echo "       $OPCODE_ROOT/start.sh"
fi
