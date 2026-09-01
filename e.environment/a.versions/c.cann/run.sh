#!/bin/bash
echo "========== CANN Version Info =========="
echo ""

echo "[CANN Environment Variables]"
echo "ASCEND_HOME: $ASCEND_HOME"
echo "ASCEND_TOOLKIT_HOME: $ASCEND_TOOLKIT_HOME"
echo "ASCEND_OPP_PATH: $ASCEND_OPP_PATH"
echo "ASCEND_AICPU_PATH: $ASCEND_AICPU_PATH"
echo "LD_LIBRARY_PATH (contains ascend): $(echo $LD_LIBRARY_PATH | tr ':' '\n' | grep -i ascend | head -5)"

echo ""
echo "[CANN Version Files]"
for path in "$ASCEND_HOME" "$ASCEND_TOOLKIT_HOME" "/usr/local/Ascend" "/home/ascend"; do
    if [ -d "$path" ]; then
        echo "--- Found CANN root: $path ---"
        ls -la "$path/version" 2>/dev/null && cat "$path/version" 2>/dev/null
        ls -la "$path"/version.txt 2>/dev/null && cat "$path"/version.txt 2>/dev/null
    fi
done

echo ""
echo "[CANN Installed Packages]"
pip3 list 2>/dev/null | grep -i "hccl\|opp\|aicpu\|cann" || echo "No CANN packages found via pip"
dpkg -l 2>/dev/null | grep -i ascend | head -20 || rpm -qa 2>/dev/null | grep -i ascend | head -20 || echo "No CANN packages found via dpkg/rpm"

echo ""
echo "[CANN Setup Status]"
if [ -f "$ASCEND_HOME/set_env.sh" ]; then
    echo "CANN env script exists: $ASCEND_HOME/set_env.sh"
elif [ -f "$ASCEND_TOOLKIT_HOME/set_env.sh" ]; then
    echo "CANN env script exists: $ASCEND_TOOLKIT_HOME/set_env.sh"
else
    echo "CANN set_env.sh not found"
fi

echo ""
echo "[Driver Version]"
cat /etc/ascend_install.info 2>/dev/null || echo "Driver info not available"

echo ""
echo "========================================="
