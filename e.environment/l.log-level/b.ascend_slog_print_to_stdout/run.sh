#!/bin/bash
set -euo pipefail

# 设置 ASCEND_SLOG_PRINT_TO_STDOUT 开关(Host 侧应用类日志输出方式)
#   1 / on   开启:日志直接打印到 stdout,不再写入 log 文件
#   0 / off  关闭:日志写入 log 文件(默认行为)

usage() {
    echo "用法: $0 [1|0|on|off]"
    echo "不带参数则进入交互选择,回车默认 0 (关闭)。"
    echo "  1 / on   开启:日志打印到 stdout,不写入 log 文件"
    echo "  0 / off  关闭:日志写入 log 文件(默认)"
    exit 1
}

# 归一化输入为 1/0
normalize() {
    case "$1" in
        1|on|ON|On)   echo 1 ;;
        0|off|OFF|Off) echo 0 ;;
        *) return 1 ;;
    esac
}

# 解析输入:支持参数传入,否则交互选择
VAL=""
if [ $# -gt 1 ]; then
    usage
fi
if [ $# -eq 1 ]; then
    case "$1" in
        -h|--help) usage ;;
        *) VAL="$(normalize "$1")" || { echo "❌ 错误:无效输入 ${1}"; usage; } ;;
    esac
else
    echo "请选择 ASCEND_SLOG_PRINT_TO_STDOUT:"
    echo "  1  开启:Host 侧应用类日志直接打印到 stdout,不写入 log 文件"
    echo "  0  关闭:日志写入 log 文件(默认)"
    read -p "输入 1/0,回车默认 0: " INPUT
    INPUT="${INPUT:-0}"
    VAL="$(normalize "$INPUT")" || { echo "❌ 错误:无效输入 ${INPUT}"; usage; }
fi

# 设置环境变量
export ASCEND_SLOG_PRINT_TO_STDOUT="$VAL"

if [ "$VAL" = "1" ]; then
    DESC="开启:日志打印到 stdout,不写入 log 文件"
else
    DESC="关闭:日志写入 log 文件"
fi

echo ""
echo "✅ 已设置 ASCEND_SLOG_PRINT_TO_STDOUT=${VAL} (${DESC})"
echo ""
echo "当前生效的命令(复制到终端即可生效):"
echo "  export ASCEND_SLOG_PRINT_TO_STDOUT=${VAL}"
echo ""
echo "💡 提示:"
echo "  - 直接执行(./run.sh)时 export 仅在本进程有效,不会影响当前终端。"
echo "  - 请将上面的 export 行复制到终端,或用 source 执行本脚本使其在当前终端生效。"
echo "  - 如需永久生效,可将该行加入 ~/.bashrc。"
