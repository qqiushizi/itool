#!/bin/bash
echo "========== OS / Compiler / GLIBC Info =========="
echo ""

echo "[OS Version]"
cat /etc/os-release 2>/dev/null || echo "Not found"
cat /etc/lsb-release 2>/dev/null || echo "Not found"
uname -a

echo ""
echo "[Compiler Versions]"
echo "GCC: $(gcc --version 2>/dev/null | head -1 || echo 'Not installed')"
echo "G++: $(g++ --version 2>/dev/null | head -1 || echo 'Not installed')"
echo "NVCC: $(nvcc --version 2>/dev/null | tail -1 || echo 'Not installed')"
echo "CXX: $CXX"
echo "CC: $CC"

echo ""
echo "[GLIBC Version]"
ldd --version 2>/dev/null | head -1
ls -la /lib*/libc.so* 2>/dev/null | head -5

echo ""
echo "[Python Version]"
python3 --version 2>/dev/null || python --version 2>/dev/null || echo "Not installed"
which python3 python 2>/dev/null

echo ""
echo "[CMake Version]"
cmake --version 2>/dev/null | head -1 || echo "Not installed"

echo ""
echo "[Make Version]"
make --version 2>/dev/null | head -1 || echo "Not installed"

echo ""
echo "[Git Version]"
git --version 2>/dev/null || echo "Not installed"

echo ""
echo "=================================================="
