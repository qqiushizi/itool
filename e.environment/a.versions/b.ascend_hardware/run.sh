#!/bin/bash
echo "========== Ascend Hardware Info =========="
echo ""

echo "[Ascend NPU Info]"
npuaieinfo=$(which npuinfo 2>/dev/null || which npu-smi 2>/dev/null)
if [ -n "$npuaieinfo" ]; then
    npu-smi info 2>/dev/null || npuinfo 2>/dev/null
else
    echo "NPU tool not found in PATH"
fi

echo ""
echo "[lspci NPU Device]"
lspci 2>/dev/null | grep -i ascend || echo "lspci not available or no Ascend device found"

echo ""
echo "[dmesg NPU Boot Log]"
dmesg 2>/dev/null | grep -i npu | tail -20 || echo "dmesg not accessible"

echo ""
echo "[Ascend Device Files]"
ls -la /dev/ascend* 2>/dev/null || ls -la /dev/davinci* 2>/dev/null || echo "No Ascend device files found"

echo ""
echo "[sysfs NPU Info]"
if [ -d /sys/class/ascend ]; then
    find /sys/class/ascend -type f -name "*" 2>/dev/null | head -30 | xargs -I{} sh -c 'echo "{}:"; cat {} 2>/dev/null'
else
    echo "/sys/class/ascend not found"
fi

echo ""
echo "[HWMlog Config]"
cat /etc/ascend_install.info 2>/dev/null || echo "No ascend_install.info"

echo ""
echo "[CANN Version from env]"
echo "ASCEND_HOME: $ASCEND_HOME"
echo "ASCEND_TOOLKIT_HOME: $ASCEND_TOOLKIT_HOME"
ls $ASCEND_HOME/version 2>/dev/null || ls $ASCEND_TOOLKIT_HOME/version 2>/dev/null || echo "Version file not found"

echo ""
echo "==========================================="
