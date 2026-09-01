#!/bin/bash
set -euo pipefail

# 设置 ASCEND_GLOBAL_LOG_LEVEL 日志级别
#   0  DEBUG     调试级别
#   1  INFO      信息级别
#   2  WARNING   警告级别
#   3  ERROR     错误级别(默认)
#   4  NULL      不输出日志

usage() {
    echo "用法: $0 [0|1|2|3|4]"
    echo "不带参数则进入交互选择,回车默认 3 (ERROR)。"
    echo "  0  DEBUG     - 调试级别"
    echo "  1  INFO      - 信息级别"
    echo "  2  WARNING   - 警告级别"
    echo "  3  ERROR     - 错误级别(默认)"
    echo "  4  NULL      - 不输出日志"
    exit 1
}

level_name() {
    case "$1" in
        0) echo "DEBUG" ;;
        1) echo "INFO" ;;
        2) echo "WARNING" ;;
        3) echo "ERROR" ;;
        4) echo "NULL" ;;
        *) echo "UNKNOWN" ;;
    esac
}

# 解析输入:支持参数传入,否则交互选择
LEVEL=""
if [ $# -gt 1 ]; then
    usage
fi
if [ $# -eq 1 ]; then
    case "$1" in
        -h|--help) usage ;;
        *) LEVEL="$1" ;;
    esac
else
    echo "请选择 ASCEND_GLOBAL_LOG_LEVEL 日志级别:"
    echo "  0  DEBUG     - 调试级别"
    echo "  1  INFO      - 信息级别"
    echo "  2  WARNING   - 警告级别"
    echo "  3  ERROR     - 错误级别(默认)"
    echo "  4  NULL      - 不输出日志"
    read -p "输入数字(0-4),回车默认 3: " LEVEL
    LEVEL="${LEVEL:-3}"
fi

# 校验级别
if ! [[ "$LEVEL" =~ ^[0-4]$ ]]; then
    echo "❌ 错误:级别必须是 0-4 的整数,当前输入: ${LEVEL}"
    usage
fi

NAME="$(level_name "$LEVEL")"

# 设置环境变量
export ASCEND_GLOBAL_LOG_LEVEL="$LEVEL"

echo ""
echo "✅ 已设置 ASCEND_GLOBAL_LOG_LEVEL=${LEVEL} (${NAME})"
echo ""
echo "当前生效的命令(复制到终端即可生效):"
echo "  export ASCEND_GLOBAL_LOG_LEVEL=${LEVEL}"
echo ""
echo "💡 提示:"
echo "  - 直接执行(./run.sh)时 export 仅在本进程有效,不会影响当前终端。"
echo "  - 请将上面的 export 行复制到终端,或用 source 执行本脚本使其在当前终端生效。"
echo "  - 如需永久生效,可将该行加入 ~/.bashrc。"
