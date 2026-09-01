#!/bin/bash
#!/bin/bash

dir1="/dockerdata/ascend/log"
dir2="/root/ascend/log"

echo "====================================="
echo "日志目录选择："
[ -d "$dir1" ] && echo "1. $dir1 [目录存在]" || echo "1. $dir1 [目录不存在]"
[ -d "$dir2" ] && echo "2. $dir2 [目录存在]" || echo "2. $dir2 [目录不存在]"
echo "====================================="
read -p "输入序号(1/2)：" sel

case $sel in
    1) plog_dir="$dir1" ;;
    2) plog_dir="$dir2" ;;
    *) echo "无效输入"; exit 1 ;;
esac

# 不存在则自动创建
if [ ! -d "$plog_dir" ]; then
    echo "目录不存在，正在创建 $plog_dir"
    mkdir -p "$plog_dir"
fi

echo "最终日志目录：$plog_dir"

tar -zcf plog-$(date +%Y-%m-%d-%H-%M-%S).tar.gz -C / ${plog_dir#/}
