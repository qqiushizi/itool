#!/bin/bash
# ============================================================
# ⑧-A 安装 OpenCode 绿色版（便携版）
# 抽包/安装 opencode，为后续 CANNBot skill 安装提供基础运行时。
# 用法:
#   ./run.sh                       # 解压并启动 opencode TUI
#   ./run.sh run "你好"            # 解压后执行单条指令
#   INSTALL=1 ./run.sh             # 解压并安装到系统(可选 PREFIX/SUDO)
# ============================================================
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd || echo "$PWD")
TARBALL="$SCRIPT_DIR/opencode.tar.gz"
OPCODE_ROOT="$SCRIPT_DIR/opencode"

if [ ! -f "$TARBALL" ]; then
    echo "[ERROR] 找不到安装包: $TARBALL" >&2
    exit 1
fi

cd "$SCRIPT_DIR"

if [ ! -x "$OPCODE_ROOT/bin/opencode" ]; then
    echo "正在解压 opencode 绿色安装包..."
    tar -xzf "$TARBALL"
else
    echo "检测到 opencode 已解压: $OPCODE_ROOT"
    echo "(如需重新解压, 请手动删除或改名 $OPCODE_ROOT)"
fi

if [ ! -x "$OPCODE_ROOT/bin/opencode" ]; then
    echo "[ERROR] 解压后未找到可执行文件: $OPCODE_ROOT/bin/opencode" >&2
    exit 1
fi

if [ "${INSTALL:-0}" = "1" ]; then
    cd "$OPCODE_ROOT"
    echo "==> 安装 opencode（PREFIX=${PREFIX:-/usr/local}）"
    if [ "$(id -u)" -ne 0 ] && [ -z "${PREFIX:-}" ]; then
        echo "==> 当前非 root，自动使用用户目录: \$HOME/.local"
        export PREFIX="$HOME/.local"
    fi
    bash ./install.sh
    BIN_CMD="$PREFIX/bin/opencode"
    PORTABLE_CMD="$PREFIX/bin/iopencode"
    echo ""
    echo "[OK] 安装完成"
    echo "     验证:        $BIN_CMD --version"
    echo "     便携启动:    $PORTABLE_CMD"
    exit 0
fi

echo "==> 启动 opencode（绿色便携模式，不修改系统）"
exec "$OPCODE_ROOT/start.sh" "$@"
