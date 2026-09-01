#!/bin/bash
echo "========== GPU Memory / Resource Info =========="
echo ""

echo "[Free Memory]"
free -h

echo ""
echo "[Disk Usage]"
df -h | head -15

echo ""
echo "[CPU Info]"
lscpu 2>/dev/null | head -20 || cat /proc/cpuinfo 2>/dev/null | head -30

echo ""
echo "[Memory Info]"
cat /proc/meminfo 2>/dev/null | head -10

echo ""
echo "[Process List - Top Memory]"
ps aux --sort=-%mem 2>/dev/null | head -15 || echo "ps aux not available"

echo ""
echo "[NPU Memory]"
if command -v npu-smi 2>/dev/null; then
    npu-smi info -m 2>/dev/null || npu-smi info 2>/dev/null | head -20
else
    echo "npu-smi not available"
fi

echo ""
echo "[Device Files]"
ls -la /dev/davinci* /dev/ascend* /dev/hisi* 2>/dev/null || echo "No accelerator device files"

echo ""
echo "================================================"
