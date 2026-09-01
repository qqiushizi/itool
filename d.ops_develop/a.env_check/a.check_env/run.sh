#!/bin/bash
# ============================================================
# 算子开发环境检查
# 功能: 先打印环境变量, 再检查开发算子所需组件是否存在, 最后给出修复建议。
# 检查组件: TOOLKIT / OPP / torch / torch_npu / pybind11 / cmake / gcc
# 说明: 纯检查, 不修改系统; 缺失项只打印建议命令, 不自动安装。
# ============================================================
set +e   # 检查类脚本: 单个命令失败不应中断

# 颜色
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; WHITE='\033[1;37m'; RESET='\033[0m'

section() { echo; echo -e "  ${CYAN}===== $1 =====${RESET}"; }
have()    { command -v "$1" >/dev/null 2>&1; }

# ============================================================
# 1. 环境变量 (先打印)
# ============================================================
section "环境变量 (Ascend / 编译相关)"
ENV_VARS="ASCEND_HOME_PATH ASCEND_TOOLKIT_HOME ASCEND_OPP_PATH ASCEND_AICPU_PATH \
ASCEND_RUNTIME_HOME ASCEND_DRIVER_PATH LD_LIBRARY_PATH PATH PYTHONPATH \
CMAKE_PREFIX_PATH CC CXX"
for v in $ENV_VARS; do
    val="$(eval echo \"\$$v\")"
    if [ -n "$val" ]; then
        echo -e "  ${WHITE}$v${RESET} = $val"
    else
        echo -e "  ${WHITE}$v${RESET} = ${YELLOW}(未设置)${RESET}"
    fi
done

# ============================================================
# 2. 组件检查
# ============================================================
section "组件检查 (算子开发依赖)"

print_ok()     { echo -e "  ${GREEN}[ OK ]${RESET} $1  ${2}"; }
print_missing() {
    echo -e "  ${RED}[缺失]${RESET} $1"
    if [ -n "$2" ]; then echo -e "          ${YELLOW}建议: $2${RESET}"; fi
}

# ---- TOOLKIT (ascend-toolkit) ----
TOOLKIT_DIR=""
[ -n "${ASCEND_HOME_PATH:-}" ] && TOOLKIT_DIR="$ASCEND_HOME_PATH"
if [ -z "$TOOLKIT_DIR" ] && [ -d /usr/local/Ascend/ascend-toolkit/latest ]; then
    TOOLKIT_DIR=/usr/local/Ascend/ascend-toolkit/latest
fi
if [ -n "$TOOLKIT_DIR" ] && [ -d "$TOOLKIT_DIR" ]; then
    TV=$(cat "$TOOLKIT_DIR/version.cfg" 2>/dev/null | grep -i version | head -1 | tr -d ' ')
    print_ok "TOOLKIT (ascend-toolkit)" "路径: $TOOLKIT_DIR ${TV:+($TV)}"
else
    print_missing "TOOLKIT (ascend-toolkit)" "安装 CANN toolkit, 或用 b.download_cann 下载后安装"
fi

# ---- OPP (算子原型库) ----
OPP_DIR="${ASCEND_OPP_PATH:-}"
[ -z "$OPP_DIR" ] && [ -n "$TOOLKIT_DIR" ] && OPP_DIR="$TOOLKIT_DIR/opp"
if [ -n "$OPP_DIR" ] && [ -d "$OPP_DIR" ]; then
    print_ok "OPP (算子原型库)" "路径: $OPP_DIR"
else
    print_missing "OPP (算子原型库)" "设置 ASCEND_OPP_PATH 指向 <toolkit>/opp"
fi

# ---- torch ----
if python3 -c "import torch" 2>/dev/null; then
    TV=$(python3 -c "import torch; print(torch.__version__)" 2>/dev/null)
    print_ok "torch" "$TV"
else
    print_missing "torch" "pip install torch (昇腾环境需配套 torch/torch_npu 版本)"
fi

# ---- torch_npu ----
if python3 -c "import torch_npu" 2>/dev/null; then
    TV=$(python3 -c "import torch_npu; print(torch_npu.__version__)" 2>/dev/null)
    print_ok "torch_npu" "$TV"
else
    print_missing "torch_npu" "pip install torch_npu (版本需与 torch/CANN 匹配)"
fi

# ---- pybind11 ----
if python3 -c "import pybind11" 2>/dev/null; then
    TV=$(python3 -c "import pybind11; print(pybind11.__version__)" 2>/dev/null)
    print_ok "pybind11" "$TV"
elif have pybind11-config; then
    TV=$(pybind11-config --version 2>/dev/null)
    print_ok "pybind11" "$TV"
else
    print_missing "pybind11" "pip install pybind11"
fi

# ---- cmake ----
if have cmake; then
    TV=$(cmake --version 2>/dev/null | head -1)
    print_ok "cmake" "$TV"
else
    print_missing "cmake" "apt install -y cmake  或  pip install cmake"
fi

# ---- gcc ----
if have gcc; then
    TV=$(gcc --version 2>/dev/null | head -1)
    print_ok "gcc" "$TV"
else
    print_missing "gcc" "apt install -y build-essential"
fi

section "检查完成"
echo -e "  ${YELLOW}提示:${RESET} 缺失项可按上面「建议」修复;"
echo -e "  ${YELLOW}      缺少 CANN 可运行:${RESET} d.ops_develop/a.env_check/b.download_cann/run.sh"
