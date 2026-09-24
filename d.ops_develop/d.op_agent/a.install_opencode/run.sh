#!/bin/bash
# ============================================================
# 安装 OpenCode 绿色版（便携版）
# 抽包/安装 opencode，为后续 CANNBot skill 安装提供基础运行时。
# 用法:
#   ./run.sh                       # 解压并启动 opencode TUI（便携，不修改系统）
#   ./run.sh run "你好"            # 解压后执行单条指令
#   ./run.sh ensure                # 仅解压，不启动（供 b.skill_store 自动调用）
#   INSTALL=1 ./run.sh             # 解压并安装到系统(可选 PREFIX/SUDO)
# ============================================================
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd || echo "$PWD")
TARBALL="$SCRIPT_DIR/opencode.tar.gz"
OPCODE_ROOT="$SCRIPT_DIR/opencode"
BIN="$OPCODE_ROOT/bin/opencode"
CONFIG_DIR="$OPCODE_ROOT/config"

# ---------- 解压 ----------
if [ ! -x "$BIN" ]; then
    if [ ! -f "$TARBALL" ]; then
        echo "[ERROR] 找不到安装包: $TARBALL" >&2
        exit 1
    fi
    echo "正在解压 opencode 绿色安装包..."
    tar -xzf "$TARBALL" -C "$SCRIPT_DIR"
else
    echo "检测到 opencode 已解压: $OPCODE_ROOT"
    echo "(如需重新解压, 请手动删除或改名 $OPCODE_ROOT)"
fi

if [ ! -x "$BIN" ]; then
    echo "[ERROR] 解压后未找到可执行文件: $BIN" >&2
    exit 1
fi

# ---------- 仅解压（供 b.skill_store 等调用，不启动） ----------
if [ "${1:-}" = "ensure" ] || [ "${EXTRACT_ONLY:-0}" = "1" ]; then
    echo "[OK] opencode 已就绪: $BIN"
    echo "     skills 目录: $CONFIG_DIR/opencode/skills"
    exit 0
fi

# ---------- 可选：安装到系统 ----------
if [ "${INSTALL:-0}" = "1" ]; then
    cd "$OPCODE_ROOT"
    echo "==> 安装 opencode（PREFIX=${PREFIX:-/usr/local}）"
    if [ "$(id -u)" -ne 0 ] && [ -z "${PREFIX:-}" ]; then
        echo "==> 当前非 root，自动使用用户目录: \$HOME/.local"
        export PREFIX="$HOME/.local"
    fi
    bash ./install.sh
    echo ""
    echo "[OK] 安装完成"
    echo "     验证:        ${PREFIX:-/usr/local}/bin/opencode --version"
    exit 0
fi

# ---------- 便携启动 ----------
echo "==> 启动 opencode（绿色便携模式，不修改系统）"

# 建立全局便携别名 iopencode（可选，失败不阻断）
if [ -w /usr/bin ] 2>/dev/null && [ ! -e /usr/bin/iopencode ]; then
    ln -sf "$BIN" /usr/bin/iopencode 2>/dev/null || true
fi

# 与绿色包内 start.sh 等价：把 XDG_CONFIG_HOME 指向 config/，
# 让 opencode 读取 config/opencode/（含 opencode.json / AGENTS.md / skills/）
export XDG_CONFIG_HOME="$CONFIG_DIR"
export PATH="$OPCODE_ROOT/bin:$PATH"

exec "$BIN" "$@"
