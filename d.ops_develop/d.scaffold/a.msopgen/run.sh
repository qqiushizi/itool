#!/bin/bash
# ============================================================
# 基于 msopgen 生成算子工程 (轻量)
# 用法:
#   bash run.sh [op.json] [arch]
#   默认: op.json 为当前目录下, arch=910B
# 说明:
#   - msopgen 来自 CANN toolkit(位于 <toolkit>/bin 或 <toolkit>/python/site-packages/bin)。
#   - 生成 AscendC 算子工程后, 打印编译/安装指导。
# ============================================================
set -euo pipefail

CYAN='\033[0;36m'; GREEN='\033[0;32m'; RED='\033[0;31m'; RESET='\033[0m'

OP_JSON="${1:-op.json}"
ARCH_IN="${2:-910B}"
# 兼容 bash3: 小写转换用 tr
ARCH=$(printf '%s' "$ARCH_IN" | tr '[:upper:]' '[:lower:]')
OUT_DIR="${OUT_DIR:-op_workspace}"

[ -f "$OP_JSON" ] || { echo -e "${RED}找不到 $OP_JSON, 请先运行 c.design/a.op_spec 生成。${RESET}" >&2; exit 1; }

# ---- 定位 msopgen ----
MSOPGEN="$(command -v msopgen 2>/dev/null || true)"
if [ -z "$MSOPGEN" ] && [ -n "${ASCEND_HOME_PATH:-}" ]; then
    MSOPGEN=$(find "$ASCEND_HOME_PATH" -type f -name msopgen 2>/dev/null | head -1 || true)
fi
if [ -z "$MSOPGEN" ]; then
    echo -e "${RED}未找到 msopgen。${RESET}" >&2
    echo "请先安装 CANN toolkit 并执行: source /usr/local/Ascend/ascend-toolkit/set_env.sh" >&2
    exit 1
fi

echo "=========================================================="
echo " msopgen 生成算子工程"
echo "   msopgen : $MSOPGEN"
echo "   op.json : $OP_JSON"
echo "   arch    : ai_core-$ARCH"
echo "   out     : $OUT_DIR"
echo "=========================================================="

"$MSOPGEN" gen -i "$OP_JSON" -f pytorch -c "ai_core-$ARCH" -out "$OUT_DIR"

echo ""
echo -e "${GREEN}✔ 工程已生成: $OUT_DIR${RESET}"
echo ""
echo -e "${CYAN}===== 编译 / 安装指导 =====${RESET}"
cat <<GUIDE
1) 进入算子内核目录编译(生成 .so):
     cd $OUT_DIR/op_kernel
     mkdir -p build && cd build
     cmake .. -DCMAKE_CXX_COMPILER=g++ && make -j
   (具体子目录以实际生成为准: op_host 为 host 侧, op_kernel 为 kernel 侧)

2) 将生成的 .so 安装到算子包目录(路径随 CANN 版本略有差异):
     cp *.so \${ASCEND_OPP_PATH}/vendors/$(whoami)/op_impl/ai_core/tbe/op_tiling/

3) 重新加载后即可被框架调用(可配合 torchbind 接入 PyTorch)。
GUIDE
