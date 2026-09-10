#!/bin/bash
# ============================================================
# ② 算子开发环境检查 (单脚本, 只读)
#
# 用途:
#   在“当前 shell 所在机器/容器”内检查算子开发所需软件版本是否匹配,
#   并汇总识别到的 Ascend 芯片型号。
#   宿主机与容器通用: 不再区分 b.check_container, 减少用户选择。
#
# 检查内容(只保留环境版本匹配所需的项目):
#   1) 运行位置 / 系统 / Python
#   2) Ascend 芯片型号 (npu-smi, lspci/驱动目录兜底)
#   3) CANN 安装目录与 toolkit/OPP 版本
#   4) torch / torch_npu 版本
#   5) 兼容性矩阵判断: Python ↔ torch ↔ torch_npu ↔ CANN ↔ 芯片
#
# 用法:
#   bash d.ops_develop/b.env_check/run.sh
#
# 说明:
#   - 纯 bash + 标准命令, 兼容 bash 3.2。
#   - 本脚本只读, 不安装/不修复/不修改系统。
#   - 矩阵为常见昇腾客户机的经验值; 不同现场需以昇腾官方配套表为准。
# ============================================================
set -o pipefail

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; WHITE='\033[1;37m'; RESET='\033[0m'

section() { echo; echo -e "  ${CYAN}===== $1 =====${RESET}"; }
ok()      { echo -e "  ${GREEN}[ 匹配 ]${RESET} $1"; }
warn()    { echo -e "  ${YELLOW}[ 警告 ]${RESET} $1"; }
bad()     { echo -e "  ${RED}[不匹配]${RESET} $1"; }
info()    { echo -e "  ${WHITE}[ 信息 ]${RESET} $1"; }
have()    { command -v "$1" >/dev/null 2>&1; }
envval()  { eval "printf '%s' \"\${$1:-}\""; }

# ------------------------------------------------------------
# 在已激活或未激活 CANN 环境下尝试执行 python 表达式, 返回 stdout。
# 优先当前 shell, 失败时若有 set_env.sh 则在子 shell source 后重试。
# ------------------------------------------------------------
py_eval() {
    local code="$1" out=""
    out=$(python3 -c "$code" 2>/dev/null) && { printf '%s' "$out"; return 0; }
    if [ -n "$SETENV" ] && [ -f "$SETENV" ]; then
        out=$( ( . "$SETENV" >/dev/null 2>&1; python3 -c "$code" 2>/dev/null ) )
        [ -n "$out" ] && { printf '%s' "$out"; return 0; }
    fi
    return 1
}

# 数字化版本: 取前两段 major.minor; 不成功则返回原值。
maj_min() {
    printf '%s' "$1" | grep -oE '^[0-9]+\.[0-9]+' | head -1
}
ver_ge() {
    # version_ge A B  =>  A >= B (仅比较 major.minor)
    local a b
    a=$(maj_min "$1"); b=$(maj_min "$2")
    [ -z "$a" ] || [ -z "$b" ] && return 1
    awk -v x="$a" -v y="$b" 'BEGIN{exit !(x+0>=y+0)}'
}

echo ""
echo -e "  ${WHITE}════════════════════════════════════════════════════════════${RESET}"
echo -e "  ${WHITE}  ② 算子开发环境检查(单脚本, 宿主机/容器通用, 只读)${RESET}"
echo -e "  ${WHITE}════════════════════════════════════════════════════════════${RESET}"

# ============================================================
# 1. 运行位置 / 系统 / Python
# ============================================================
section "1. 运行位置 / 系统 / Python"

IN_CONTAINER=0
[ -f /.dockerenv ] && IN_CONTAINER=1
grep -qE 'docker|containerd|kubepods' /proc/1/cgroup 2>/dev/null && IN_CONTAINER=1
if [ "$IN_CONTAINER" = "1" ]; then
    RUN_POS="容器内"
else
    RUN_POS="宿主机"
fi
# 常见: 容器内常设 ASCEND_RUNTIME_MOUNTS；宿主机一般没有。
if [ -n "$(envval ASCEND_RUNTIME_MOUNTS)" ] && [ "$IN_CONTAINER" = "0" ]; then RUN_POS="疑似容器"; fi

SYS_ARCH=$(uname -m 2>/dev/null || echo unknown)
SYS_PRETTY=""
if [ -f /etc/os-release ]; then
    SYS_PRETTY=$(sed -n 's/^PRETTY_NAME="\{0,1\}\([^"]*\)"\{0,1\}/\1/p' /etc/os-release | head -1)
fi
[ -z "$SYS_PRETTY" ] && SYS_PRETTY="unknown"

PY_BIN=$(command -v python3 2>/dev/null || true)
PY_VER=""
if [ -n "$PY_BIN" ]; then
    PY_VER=$(python3 -c 'import sys; print(sys.version.split()[0])' 2>/dev/null || true)
fi
[ -z "$PY_VER" ] && PY_VER="未知"

printf '  %-14s: %s\n' "运行位置" "$RUN_POS"
printf '  %-14s: %s\n' "主机名" "$(hostname 2>/dev/null || echo unknown)"
printf '  %-14s: %s\n' "系统" "$SYS_PRETTY ($SYS_ARCH)"
printf '  %-14s: %s\n' "Python" "${PY_VER:-未知}${PY_BIN:+  ($PY_BIN)}"

# ============================================================
# 2. Ascend 芯片型号识别
# ============================================================
section "2. Ascend 芯片型号"

CHIP=""
CHIP_SOURCE=""
CHIP_LIST=""
N_DEV=0

NPU_INFO=""
if have npu-smi; then
    NPU_INFO=$(npu-smi info 2>/dev/null || true)
    if [ -n "$NPU_INFO" ]; then
        CHIP_SOURCE="npu-smi info"
        CHIP_LIST=$(printf '%s\n' "$NPU_INFO" | grep -oiE '910[a-zA-Z]|950|310p|310' 2>/dev/null | tr '[:lower:]' '[:upper:]' | sort -u | tr '\n' ',')
        CHIP_LIST=$(printf '%s' "$CHIP_LIST" | sed 's/,$//')
    fi
fi

if [ -z "$CHIP_LIST" ]; then
    if have lspci; then
        LSPCI_ASC=$(lspci 2>/dev/null | grep -i 'ascend\|processing accelerators' || true)
        if [ -n "$LSPCI_ASC" ]; then
            CHIP_SOURCE="lspci(ascend)"
            CHIP_LIST=$(printf '%s\n' "$LSPCI_ASC" | grep -oiE '910[a-zA-Z]|950|310p|310' 2>/dev/null | tr '[:lower:]' '[:upper:]' | sort -u | tr '\n' ',')
            CHIP_LIST=$(printf '%s' "$CHIP_LIST" | sed 's/,$//')
        fi
    fi
fi

if [ -z "$CHIP_LIST" ] && [ -f /etc/ascend_install.info ]; then
    DRV_INFO=$(cat /etc/ascend_install.info 2>/dev/null || true)
    CHIP_LIST=$(printf '%s\n' "$DRV_INFO" | grep -oiE '910[a-zA-Z]|950|310p|310' 2>/dev/null | tr '[:lower:]' '[:upper:]' | sort -u | tr '\n' ',')
    CHIP_LIST=$(printf '%s' "$CHIP_LIST" | sed 's/,$//')
    [ -n "$CHIP_LIST" ] && CHIP_SOURCE="/etc/ascend_install.info"
fi

if [ -n "$CHIP_LIST" ]; then
    case "$CHIP_LIST" in
        *950*)  CHIP="950";;
        *910C*) CHIP="910C";;
        *910B*) CHIP="910B";;
        *910A*) CHIP="910A";;
        *310P*) CHIP="310P";;
        *310*)  CHIP="310";;
        *)      CHIP="${CHIP_LIST%%,*}";;
    esac
else
    CHIP="未知"
    CHIP_SOURCE="未识别"
fi

N_DEV=$(ls /dev/davinci* 2>/dev/null | wc -l | tr -d ' ')
N_DEV=${N_DEV:-0}

if [ "$CHIP" != "未知" ]; then
    ok "识别到芯片型号: $CHIP"
    printf '  %-14s: %s\n' "识别来源" "$CHIP_SOURCE"
    printf '  %-14s: %s\n' "识别原始词" "${CHIP_LIST:-$CHIP}"
    printf '  %-14s: %s\n' "NPU 设备数" "$N_DEV"
else
    warn "未能识别芯片型号 (尝试 npu-smi / lspci / /etc/ascend_install.info 均未找到 910/950/310 关键字)"
    printf '  %-14s: %s\n' "NPU 设备数" "$N_DEV"
fi

if have npu-smi; then
    DRIVER_NPU=$(npu-smi -v 2>/dev/null | head -1 || true)
    [ -n "$DRIVER_NPU" ] && printf '  %-14s: %s\n' "npu-smi 版本" "$DRIVER_NPU"
fi

# ============================================================
# 3. CANN toolkit 版本
# ============================================================
section "3. CANN toolkit 版本"

TOOLKIT_DIR=""
CANDIDATES=()
[ -n "$(envval ASCEND_HOME_PATH)" ] && CANDIDATES+=("$(envval ASCEND_HOME_PATH)")
[ -n "$(envval ASCEND_TOOLKIT_HOME)" ] && CANDIDATES+=("$(envval ASCEND_TOOLKIT_HOME)")
CANDIDATES+=("/usr/local/Ascend/ascend-toolkit/latest" "/usr/local/Ascend/ascend-toolkit" "/usr/local/Ascend")

FOUND_DIRS=()
for d in "${CANDIDATES[@]}"; do
    [ -n "$d" ] && [ -d "$d" ] && FOUND_DIRS+=("$d")
done

# 去重(保序)
UNIQ_DIRS=()
for d in "${FOUND_DIRS[@]}"; do
    found=0
    for u in "${UNIQ_DIRS[@]}"; do [ "$u" = "$d" ] && found=1 && break; done
    [ "$found" = "0" ] && UNIQ_DIRS+=("$d")
done
FOUND_DIRS=("${UNIQ_DIRS[@]}")

if [ ${#FOUND_DIRS[@]} -gt 0 ]; then
    for d in "${FOUND_DIRS[@]}"; do
        echo -e "  ${GREEN}•${RESET} $d"
        [ -n "$TOOLKIT_DIR" ] || TOOLKIT_DIR="$d"
    done
else
    bad "未发现 CANN toolkit 安装目录"
fi

CANN_VER=""
SETENV=""
print_verfile() { if [ -f "$1" ]; then echo "        --- $1 ---"; sed 's/^/        /' "$1" 2>/dev/null | head -12; fi; }

if [ -n "$TOOLKIT_DIR" ] && [ -d "$TOOLKIT_DIR" ]; then
    for vf in version.cfg version version.info; do
        if [ -f "$TOOLKIT_DIR/$vf" ]; then
            print_verfile "$TOOLKIT_DIR/$vf"
            [ -n "$CANN_VER" ] && continue
            case "$vf" in
                version.cfg)
                    CANN_VER=$(grep -iE '^\s*version\s*=' "$TOOLKIT_DIR/$vf" 2>/dev/null | head -1 | sed 's/^[^=]*=//' | tr -d '[:space:]"')
                    ;;
                version|version.info)
                    CANN_VER=$(head -1 "$TOOLKIT_DIR/$vf" 2>/dev/null | sed 's/^[^=]*=//' | tr -d '[:space:]"')
                    ;;
            esac
        fi
    done
    [ -z "$CANN_VER" ] && CANN_VER="已安装(版本未知)"
fi

OPP_DIR="$(envval ASCEND_OPP_PATH)"
[ -z "$OPP_DIR" ] && [ -n "$TOOLKIT_DIR" ] && OPP_DIR="$TOOLKIT_DIR/opp"
if [ -n "$OPP_DIR" ] && [ -d "$OPP_DIR" ]; then
    echo -e "  ${GREEN}•${RESET} OPP(算子原型库): $OPP_DIR"
else
    [ -n "$TOOLKIT_DIR" ] && warn "未找到 OPP(算子原型库), 算子开发通常需要 kernels/opp"
fi

if [ -n "$(envval ASCEND_HOME_PATH)" ] && [ -n "$(envval ASCEND_TOOLKIT_HOME)" ]; then
    SOURCE_STATE="部分/已设置 ASCEND_HOME_PATH、ASCEND_TOOLKIT_HOME"
else
    SOURCE_STATE="未在当前 shell 检测到完整 ASCEND_* 激活变量"
fi

for p in "$TOOLKIT_DIR/set_env.sh" "$(envval ASCEND_HOME_PATH)/set_env.sh" /usr/local/Ascend/ascend-toolkit/set_env.sh /usr/local/Ascend/ascend-toolkit/latest/set_env.sh /usr/local/Ascend/set_env.sh; do
    if [ -n "$p" ] && [ -f "$p" ]; then SETENV="$p"; break; fi
done

if [ -n "$SETENV" ]; then
    ok "找到 CANN 激活脚本: $SETENV"
else
    warn "未找到 set_env.sh (未安装 toolkit, 或自定义安装路径)"
fi

# ============================================================
# 4. torch / torch_npu 版本
# ============================================================
section "4. torch / torch_npu 版本"

TORCH_VER=$(py_eval 'import torch; print(torch.__version__, end="")' 2>/dev/null || true)
TORCH_NPU_VER=$(py_eval 'import torch_npu; print(torch_npu.__version__, end="")' 2>/dev/null || true)
TORCH_FOUND=1; TORCH_NPU_FOUND=1
[ -z "$TORCH_VER" ] && TORCH_FOUND=0
[ -z "$TORCH_NPU_VER" ] && TORCH_NPU_FOUND=0

if [ "$TORCH_FOUND" = "1" ]; then
    ok "torch      : $TORCH_VER"
else
    bad "torch 未安装/不可导入"
fi
if [ "$TORCH_NPU_FOUND" = "1" ]; then
    ok "torch_npu  : $TORCH_NPU_VER"
else
    bad "torch_npu 未安装/不可导入 (当前 Python: ${PY_VER:-未知})"
fi

# ============================================================
# 5. 兼容性矩阵判断
# ============================================================
section "5. 兼容性矩阵判断"

MATRIX_OK=1
PY_TORCH_OK=1
TORCH_NPU_OK=1
CANN_CHIP_OK=1
RECOMMEND_NOTE=""

# -- 5.1 Python == torch 支持范围 ---------------------------------
if [ -n "$PY_VER" ] && [ "$TORCH_FOUND" = "1" ]; then
    PY_MAJMIN=$(maj_min "$PY_VER")
    case "$PY_MAJMIN" in
        3.8|3.9|3.10|3.11|3.12)
            ok "Python ${PY_VER} 在 torch 常见支持范围内"
            ;;
        "")
            warn "Python 版本未知, 跳过该项"
            ;;
        *)
            warn "Python ${PY_VER} 与当前 torch 组合不常见, 建议核对"
            PY_TORCH_OK=0
            ;;
    esac
else
    PY_TORCH_OK=0
    [ "$TORCH_FOUND" = "1" ] || warn "缺少 torch, 无法判断 Python ↔ torch"
fi

# -- 5.2 torch ↔ torch_npu 主版本 ---------------------------------
TORCH_MM=""
TORCH_NPU_MM=""
[ "$TORCH_FOUND" = "1" ] && TORCH_MM=$(maj_min "$TORCH_VER")
[ "$TORCH_NPU_FOUND" = "1" ] && TORCH_NPU_MM=$(maj_min "$TORCH_NPU_VER")

if [ "$TORCH_FOUND" = "1" ] && [ "$TORCH_NPU_FOUND" = "1" ]; then
    if [ -n "$TORCH_MM" ] && [ "$TORCH_MM" = "$TORCH_NPU_MM" ]; then
        ok "torch ${TORCH_VER} 与 torch_npu ${TORCH_NPU_VER} 主版本一致"
    else
        bad "torch(${TORCH_VER}) 与 torch_npu(${TORCH_NPU_VER}) 主版本不一致, 强烈建议保持同系列"
        TORCH_NPU_OK=0
    fi
else
    TORCH_NPU_OK=0
    warn "torch/torch_npu 缺失, 跳过主版本比对"
fi

# -- 5.3 芯片 ↔ CANN ↔ torch_npu 常见矩阵 -------------------------
if [ "$CHIP" != "未知" ] && [ -n "$CANN_VER" ] && [ "$TORCH_NPU_FOUND" = "1" ]; then
    CANN_MM=$(maj_min "$CANN_VER")
    case "$CHIP" in
        950)
            if [ -n "$CANN_MM" ] && ver_ge "$CANN_MM" "9.0" && [ "$TORCH_NPU_MM" = "2.6" ]; then
                ok "950 常见配套: CANN >= 9.0.0 + torch 2.6.x / torch_npu 2.6.x"
                RECOMMEND_NOTE="CANN ${CANN_VER} + torch_npu ${TORCH_NPU_VER}, 芯片 950: 常见可匹配"
            else
                bad "950 常见配套为 CANN >= 9.0.0 + torch_npu 2.6.x; 当前为例外组合"
                RECOMMEND_NOTE="建议 950 使用 CANN 9.0.0 与 torch 2.6.0/torch_npu 2.6.0.post2"
                CANN_CHIP_OK=0
            fi
            ;;
        910B|910C|910A)
            if [ -n "$CANN_MM" ] && [ "$TORCH_NPU_MM" = "2.1" ] ; then
                ok "910 系列常见配套: CANN 8.x + torch 2.1.x / torch_npu 2.1.x"
                RECOMMEND_NOTE="CANN ${CANN_VER} + torch_npu ${TORCH_NPU_VER}, 芯片 ${CHIP}: 常见可匹配"
            elif [ -n "$CANN_MM" ] && ver_ge "$CANN_MM" "9.0" && [ "$TORCH_NPU_MM" = "2.6" ]; then
                ok "910 系列较新配套: CANN 9.x + torch 2.6.x / torch_npu 2.6.x"
                RECOMMEND_NOTE="CANN ${CANN_VER} + torch_npu ${TORCH_NPU_VER}, 芯片 ${CHIP}: 常见可匹配"
            else
                bad "芯片 ${CHIP} 与 CANN(${CANN_VER})/torch_npu(${TORCH_NPU_VER}) 组合未命中常见矩阵"
                RECOMMEND_NOTE="建议 910 系列优先 CANN 8.1.RC1 + torch 2.1.0/torch_npu 2.1.0.post16, 或 CANN 9.0.0 + torch 2.6.0/torch_npu 2.6.0.post2"
                CANN_CHIP_OK=0
            fi
            ;;
        310P)
            if [ -n "$CANN_MM" ] && [ "$TORCH_NPU_MM" = "2.1" ]; then
                ok "310P 常见配套: CANN 8.x + torch 2.1.x / torch_npu 2.1.x"
                RECOMMEND_NOTE="CANN ${CANN_VER} + torch_npu ${TORCH_NPU_VER}, 芯片 310P: 常见可匹配"
            else
                bad "310P 常见配套为 CANN 8.x + torch_npu 2.1.x; 当前为例外组合"
                RECOMMEND_NOTE="建议 310P 使用 CANN 8.x + torch 2.1.0/torch_npu 2.1.0.post16"
                CANN_CHIP_OK=0
            fi
            ;;
        *)
            warn "芯片 ${CHIP} 不在内置矩阵中, 跳过芯片级判断"
            ;;
    esac
elif [ "$CHIP" = "未知" ]; then
    warn "芯片未知, 跳过芯片级矩阵判断"
elif [ "$TORCH_NPU_FOUND" = "0" ]; then
    CANN_CHIP_OK=0
    warn "torch_npu 未识别, 跳过芯片级矩阵判断"
elif [ -z "$CANN_VER" ]; then
    CANN_CHIP_OK=0
    warn "CANN 未识别, 跳过芯片级矩阵判断"
fi

# ============================================================
# 汇总报告
# ============================================================
echo ""
echo -e "  ${CYAN}════════════════════════════════════════════════════════════${RESET}"
echo -e "  ${WHITE}  【环境检查汇总报告】${RESET}"
echo -e "  ${CYAN}════════════════════════════════════════════════════════════${RESET}"
printf '  %-14s: %s\n' "运行位置" "$RUN_POS"
printf '  %-14s: %s\n' "芯片型号" "$CHIP${CHIP_LIST:+  ($CHIP_LIST)}"
printf '  %-14s: %s\n' "NPU 设备数" "$N_DEV"
printf '  %-14s: %s\n' "Python" "${PY_VER:-未知}"
printf '  %-14s: %s\n' "CANN" "${CANN_VER:-未识别}"
printf '  %-14s: %s\n' "安装目录" "${TOOLKIT_DIR:-未发现}"
printf '  %-14s: %s\n' "激活脚本" "${SETENV:-未找到}"
printf '  %-14s: %s\n' "torch" "${TORCH_VER:-未安装}"
printf '  %-14s: %s\n' "torch_npu" "${TORCH_NPU_VER:-未安装}"
printf '  %-14s: %s\n' "识别来源" "$CHIP_SOURCE"
echo -e "  ${CYAN}────────────────────────────────────────────────────────────${RESET}"

[ "$PY_TORCH_OK" = "0" ] && MATRIX_OK=0
[ "$TORCH_NPU_OK" = "0" ] && MATRIX_OK=0
[ "$CANN_CHIP_OK" = "0" ] && MATRIX_OK=0
if [ "$TORCH_FOUND" = "0" ] || [ "$TORCH_NPU_FOUND" = "0" ]; then MATRIX_OK=0; fi

if [ "$MATRIX_OK" = "1" ]; then
    echo -e "  ${GREEN}✅ 综合结论: 当前环境软件版本匹配, 可以继续算子开发。${RESET}"
else
    echo -e "  ${YELLOW}⚠ 综合结论: 存在缺失或组合不匹配, 请先补齐/调整后再开始算子开发。${RESET}"
fi

# 建议命令(只提示, 不自动执行)
echo -e "  ${YELLOW}下一步建议(按需手动执行):${RESET}"
if [ -z "$TOOLKIT_DIR" ]; then
    echo -e "    → 安装/激活 CANN:   bash d.ops_develop/c.install_cann/a.cann-9.1.0/run.sh"
fi
if [ -n "$SETENV" ]; then
    echo -e "    → 激活当前环境:     source $SETENV"
fi
if [ "$TORCH_FOUND" = "0" ] || [ "$TORCH_NPU_FOUND" = "0" ]; then
    echo -e "    → 安装 torch/torch_npu: 参考 e.environment/e.setenvs/setenvs.sh 内的 install_torch"
fi
[ -n "$RECOMMEND_NOTE" ] && echo -e "    → 版本建议:         $RECOMMEND_NOTE"
if [ "$MATRIX_OK" = "1" ]; then
    echo -e "    → 需求分析:         bash d.ops_develop/d.op_design/a.op_spec/run.sh"
    echo -e "    → 生成工程:         bash d.ops_develop/e.op_scaffold/a.msopgen/run.sh"
fi
echo ""
echo "完成时间: $(date '+%F %T')"
