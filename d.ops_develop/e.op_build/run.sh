#!/bin/bash
# ============================================================
# 算子工程构建 (msopgen)
#
# 功能:
#   1) 到 d.ops_develop/workspace 目录下查找 op.json
#   2) 使用 CANN msopgen 根据 op.json 建立 AscendC 算子工程
#   3) 输出到 d.ops_develop/workspace/op_build_<算子名>
#
# 用法:
#   bash d.ops_develop/e.op_build/run.sh
#   bash d.ops_develop/e.op_build/run.sh d.ops_develop/workspace/op_design_AddCustom/op.json
#   bash d.ops_develop/e.op_build/run.sh <op.json> 910B
#
# 常用环境变量:
#   OPS_WORKSPACE  统一工作目录，默认 d.ops_develop/workspace
#   OUT_DIR        覆盖 msopgen 输出目录
#   REPO_ROOT      itool 仓库根目录，默认自动识别
# ============================================================
set -euo pipefail

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; WHITE='\033[1;37m'; RESET='\033[0m'

read_def() {
    local prompt="$1" def="$2"
    printf "  %s [%s]: " "$prompt" "$def"
    IFS= read -r REPLY || REPLY=""
    [ -z "$REPLY" ] && REPLY="$def"
}

# ---------- 统一工作区：d.ops_develop/workspace ----------
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd || echo "$PWD")
REPO_ROOT="${ITOOL_REPO_ROOT:-}"
if [ -z "$REPO_ROOT" ] && [ -f "$PWD/itool.sh" ]; then
    REPO_ROOT="$PWD"
fi
if [ -z "$REPO_ROOT" ] && command -v git >/dev/null 2>&1; then
    REPO_ROOT=$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel 2>/dev/null || true)
fi
if [ -z "$REPO_ROOT" ]; then
    d="$SCRIPT_DIR"
    while [ "$d" != "/" ]; do
        if [ -f "$d/itool.sh" ]; then REPO_ROOT="$d"; break; fi
        d=$(dirname "$d")
    done
fi
[ -z "$REPO_ROOT" ] && REPO_ROOT="$SCRIPT_DIR"
OPS_WORKSPACE="${OPS_WORKSPACE:-$REPO_ROOT/d.ops_develop/workspace}"
mkdir -p "$OPS_WORKSPACE"

# ---------- 解析 op.json ----------
OP_JSON="${1:-}"
ARCH_IN="${2:-910B}"

# msopgen -c 需要的是 ai_core-ascendXXX 形式，例如 ai_core-ascend910B。
# 允许用户输入 910B / ascend910B / ai_core-ascend910B。
normalize_soc() {
    local s="${1:-}"
    case "$s" in
        ai_core-*) s="${s#ai_core-}" ;;
        ai_core* ) s="${s#ai_core}" ;;
    esac
    case "$s" in
        [Aa]scend*) s="${s#[Aa]scend}" ;;
    esac
    case "$s" in
        910B|910b) printf 'ascend910B'; return 0 ;;
        910A|910a) printf 'ascend910A'; return 0 ;;
        910)       printf 'ascend910';  return 0 ;;
        310P|310p) printf 'ascend310P'; return 0 ;;
        310)       printf 'ascend310';  return 0 ;;
        *)         printf '%s' "$s";    return 0 ;;
    esac
}
SOC_UNIT=$(normalize_soc "$ARCH_IN")
COMPUTE_UNIT="ai_core-$SOC_UNIT"

if [ -n "$OP_JSON" ]; then
    if [ -d "$OP_JSON" ]; then
        CANDIDATE=$(find "$OP_JSON" -maxdepth 1 -name op.json -type f 2>/dev/null | head -1 || true)
        [ -n "$CANDIDATE" ] && OP_JSON="$CANDIDATE"
    fi
    [ -f "$OP_JSON" ] || { echo -e "${RED}找不到 op.json: $OP_JSON${RESET}" >&2; exit 1; }
else
    JSON_FILES=()
    while IFS= read -r -d '' f; do
        JSON_FILES+=("$f")
    done < <(find "$OPS_WORKSPACE" -name op.json -type f -print0 2>/dev/null | sort -z)
    if [ ${#JSON_FILES[@]} -eq 0 ]; then
        echo -e "${RED}在 $OPS_WORKSPACE 下没有找到 op.json。${RESET}" >&2
        echo "请先运行算子需求分析：" >&2
        echo "  bash d.ops_develop/d.op_design/run.sh" >&2
        exit 1
    fi
    if [ ${#JSON_FILES[@]} -eq 1 ]; then
        OP_JSON="${JSON_FILES[0]}"
        echo -e "  ${GREEN}[自动选择]${RESET} $OP_JSON"
    else
        echo ""
        echo -e "  ${CYAN}找到多个 op.json，请选择:${RESET}"
        for ((i=0; i<${#JSON_FILES[@]}; i++)); do
            printf '    %3d) %s\n' "$((i+1))" "${JSON_FILES[$i]}"
        done
        while true; do
            read_def "请选择编号" "1"
            n="$REPLY"
            if [ "$n" -ge 1 ] 2>/dev/null && [ "$n" -le "${#JSON_FILES[@]}" ] 2>/dev/null; then
                OP_JSON="${JSON_FILES[$((n-1))]}"
                break
            fi
            echo -e "  ${RED}编号无效。${RESET}"
        done
    fi
fi

# ---------- 提取算子名 ----------
OP_NAME=""
if command -v python3 >/dev/null 2>&1; then
    OP_NAME=$(python3 -c 'import json,sys; d=json.load(open(sys.argv[1], encoding="utf-8")); d=d[0] if isinstance(d,list) else d; print(d.get("op","") or "")' "$OP_JSON" 2>/dev/null || true)
fi
[ -n "$OP_NAME" ] || OP_NAME=$(basename "$(dirname "$OP_JSON")" | sed 's/^op_design_//')
[ -n "$OP_NAME" ] || OP_NAME="Custom"
OUT_DIR="${OUT_DIR:-$OPS_WORKSPACE/op_build_${OP_NAME}}"

# ---------- 定位 msopgen ----------
MSOPGEN="$(command -v msopgen 2>/dev/null || true)"
if [ -z "$MSOPGEN" ] && [ -n "${ASCEND_HOME_PATH:-}" ]; then
    MSOPGEN=$(find "$ASCEND_HOME_PATH" -type f -name msopgen 2>/dev/null | head -1 || true)
fi
if [ -z "$MSOPGEN" ]; then
    echo -e "${RED}未找到 msopgen。${RESET}" >&2
    echo "请先安装/激活 CANN toolkit 环境，例如：" >&2
    echo "  source /usr/local/Ascend/ascend-toolkit/set_env.sh" >&2
    exit 1
fi

echo ""
echo -e "  ${WHITE}==========================================================${RESET}"
echo -e "  ${WHITE} msopgen 生成算子工程${RESET}"
echo -e "  ${WHITE}==========================================================${RESET}"
printf '  %-16s: %s\n' "msopgen" "$MSOPGEN"
printf '  %-16s: %s\n' "op.json" "$OP_JSON"
printf '  %-16s: %s\n' "arch" "$COMPUTE_UNIT"
printf '  %-16s: %s\n' "输出目录" "$OUT_DIR"
echo ""

mkdir -p "$OUT_DIR"
if [ -n "$(find "$OUT_DIR" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null || true)" ]; then
    echo -e "  ${YELLOW}[警告]${RESET} 输出目录已存在且非空: $OUT_DIR"
    echo -e "  ${YELLOW}msopgen 可能直接覆盖或报错；如需保留旧工程，请先重命名/清理该目录。${RESET}"
fi

"$MSOPGEN" gen -i "$OP_JSON" -f tf -lan cpp -c "$COMPUTE_UNIT" -out "$OUT_DIR"

echo ""
echo -e "  ${GREEN}✔ 算子工程已生成:${RESET} $OUT_DIR"
echo ""
echo -e "  ${CYAN}===== 编译 / 安装指导 =====${RESET}"
cat <<GUIDE
1) 进入算子内核目录编译(生成 .so):
     cd $OUT_DIR/op_kernel
     mkdir -p build && cd build
     cmake .. -DCMAKE_CXX_COMPILER=g++ && make -j

2) 将生成的 .so 安装到算子包目录(路径随 CANN 版本略有差异):
     cp *.so \${ASCEND_OPP_PATH}/vendors/$(whoami)/op_impl/ai_core/tbe/op_tiling/

3) 后续需要 torch 接入时，可回到 itool 仓库继续扩展工作区功能。
GUIDE
