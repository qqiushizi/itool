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
#   bash d.ops_develop/c.env_check/run.sh
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
# 多 Python 识别辅助:
# 客户容器里可能同时存在 /usr/bin/python3 和 /usr/local/python3.x 等
# 多个解释器。脚本会优先选择“能成功 import torch”的那个解释器来做
# 后续 torch / torch_npu 检测，避免明明装了 torch 却被判定为未安装。
# ------------------------------------------------------------
PY_CANDIDATES=()
add_py_candidate() {
    local p="$1" u
    [ -n "$p" ] || return 0
    [ -x "$p" ] || return 0
    for u in "${PY_CANDIDATES[@]}"; do
        [ "$u" = "$p" ] && return 0
    done
    PY_CANDIDATES+=("$p")
}
collect_python_candidates() {
    local d f OLDIFS
    add_py_candidate "$(command -v python3 2>/dev/null)"
    add_py_candidate "$(command -v python 2>/dev/null)"

    OLDIFS=$IFS
    IFS=':'
    for d in $PATH; do
        [ -n "$d" ] || continue
        [ -d "$d" ] || continue
        for f in "$d"/python3 "$d"/python "$d"/python3.*; do
            add_py_candidate "$f"
        done
    done
    IFS=$OLDIFS

    # 常见自定义/容器路径兜底
    for f in /usr/local/bin/python3 /usr/bin/python3 /usr/local/python3.12.13/bin/python3 /usr/local/python3.10*/bin/python3 /usr/local/python*/bin/python3; do
        add_py_candidate "$f"
    done
}
python_import_ok() {
    local p="$1"
    [ -n "$p" ] || return 1
    [ -x "$p" ] || return 1
    [ -n "$("$p" -c 'import torch; print(torch.__version__, end="")' 2>/dev/null)" ] && return 0
    return 1
}
resolve_active_python() {
    local p default_py
    default_py=$(command -v python3 2>/dev/null || true)
    # 默认 python3 如果已能 import torch，优先保持，避免切换到其他环境。
    if [ -n "$default_py" ] && [ -x "$default_py" ] && python_import_ok "$default_py"; then
        PY_BIN="$default_py"
        return 0
    fi
    # 否则在发现的候选解释器中选择第一个能 import torch 的。
    for p in "${PY_CANDIDATES[@]}"; do
        if python_import_ok "$p"; then
            PY_BIN="$p"
            return 0
        fi
    done
    # 都不行就退回系统默认 python3，并如实报告。
    PY_BIN="$default_py"
}

# ------------------------------------------------------------
# 在已激活或未激活 CANN 环境下尝试执行 python 表达式, 返回 stdout。
# 优先当前 shell, 失败时若有 set_env.sh 则在子 shell source 后重试。
# ------------------------------------------------------------
py_eval() {
    local code="$1" py out=""
    py="${PY_BIN:-python3}"
    out=$("$py" -c "$code" 2>/dev/null) && { printf '%s' "$out"; return 0; }
    if [ -n "$SETENV" ] && [ -f "$SETENV" ]; then
        out=$( ( . "$SETENV" >/dev/null 2>&1; "$py" -c "$code" 2>/dev/null ) )
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

collect_python_candidates
PY_BIN=""
PY_VER=""
resolve_active_python
PY_BIN="${PY_BIN:-$(command -v python3 2>/dev/null || true)}"
if [ -n "$PY_BIN" ] && [ -x "$PY_BIN" ]; then
    PY_VER=$("$PY_BIN" -c 'import sys; print(sys.version.split()[0])' 2>/dev/null || true)
fi
[ -z "$PY_VER" ] && PY_VER="未知"
PY_DEFAULT=$(command -v python3 2>/dev/null || true)

printf '  %-14s: %s\n' "运行位置" "$RUN_POS"
printf '  %-14s: %s\n' "主机名" "$(hostname 2>/dev/null || echo unknown)"
printf '  %-14s: %s\n' "系统" "$SYS_PRETTY ($SYS_ARCH)"
printf '  %-14s: %s\n' "Python" "${PY_VER:-未知}${PY_BIN:+  ($PY_BIN)}"
if [ -n "$PY_BIN" ] && [ -n "$PY_DEFAULT" ] && [ "$PY_BIN" != "$PY_DEFAULT" ]; then
    info "检测到多个 Python；已选择可导入 torch 的解释器: $PY_BIN"
elif [ -n "$PY_BIN" ] && [ "$PY_BIN" = "$PY_DEFAULT" ]; then
    info "torch 导入检测使用解释器: $PY_BIN"
fi

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

CANN_VER=""
TOOLKIT_DIR=""
VERSION_FILE=""
SETENV=""
OPP_DIR=""

UNIQ_DIRS=()
add_unique_dir() {
    local d="$1" found=0 u
    [ -n "$d" ] || return 0
    [ -d "$d" ] || return 0
    for u in "${UNIQ_DIRS[@]}"; do
        [ "$u" = "$d" ] && found=1 && break
    done
    [ "$found" = "0" ] && UNIQ_DIRS+=("$d")
}

# 1) 优先采用环境变量声明的 CANN 路径
for e in ASCEND_HOME_PATH ASCEND_TOOLKIT_HOME ASCEND_OPP_PATH; do
    add_unique_dir "$(envval "$e")"
done

# 2) 常用默认路径
for d in /usr/local/Ascend/ascend-toolkit/latest /usr/local/Ascend/ascend-toolkit /usr/local/Ascend; do
    add_unique_dir "$d"
done

# 3) 若只有 /usr/local/Ascend 这类父目录，也把下一级目录纳入，
#    以便找到 /usr/local/Ascend/ascend-toolkit/latest 等真实 toolkit 根目录
for base in /usr/local/Ascend /usr/local/Ascend/ascend-toolkit; do
    [ -d "$base" ] || continue
    for child in "$base"/*; do
        [ -d "$child" ] || continue
        add_unique_dir "$child"
    done
done

if [ ${#UNIQ_DIRS[@]} -gt 0 ]; then
    for d in "${UNIQ_DIRS[@]}"; do
        echo -e "  ${GREEN}•${RESET} 候选目录: $d"
    done
    TOOLKIT_DIR="${UNIQ_DIRS[0]}"
else
    bad "未发现 CANN toolkit 安装目录"
fi
# 仅保留可能是 CANN Toolkit 的目录；driver/firmware/nnal/hdk/opp 等目录不参与版本查找
SEARCH_DIRS=()
for d in "${UNIQ_DIRS[@]}"; do
    case "$d" in
        /usr/local/Ascend) continue ;;
        */driver|*/driver/*|*/firmware|*/firmware/*|*/nnal|*/nnal/*|*/hdk|*/hdk/*|*/opp|*/opp/*) continue ;;
    esac
    SEARCH_DIRS+=("$d")
done

# 从文件内容解析 CANN Toolkit 版本
# 优先读取 version.cfg 里的 toolkit_running_version，避免把 cann_running_version 等字段误报成 toolkit
parse_version_file() {
    local f="$1" v
    [ -f "$f" ] || return 1

    # 1) 首选：toolkit_running_version=9.1.0
    v=$(grep -iE '^[[:space:]]*toolkit_running_version[[:space:]]*=' "$f" 2>/dev/null | head -1 | sed 's/^[^=]*=//' | sed 's/^[[:space:]]*//')
    v=$(printf '%s' "$v" | grep -oiE '[0-9]+\.[0-9]+(\.[0-9]+)?([.-]?RC[0-9]+)?([.-]?B[0-9]+)?' | head -1)
    [ -n "$v" ] && { printf '%s' "$v"; return 0; }

    # 2) 次选：ascend_toolkit_version / toolkit_version / version
    v=$(grep -iE '^[[:space:]]*(ascend_toolkit_version|toolkit_version|version)[[:space:]]*=' "$f" 2>/dev/null | head -1 | sed 's/^[^=]*=//' | sed 's/^[[:space:]]*//')
    v=$(printf '%s' "$v" | grep -oiE '[0-9]+\.[0-9]+(\.[0-9]+)?([.-]?RC[0-9]+)?([.-]?B[0-9]+)?' | head -1)
    [ -n "$v" ] && { printf '%s' "$v"; return 0; }

    # 3) 兜底：文件第一处像版本号的片段
    v=$(grep -oiE '[0-9]+\.[0-9]+(\.[0-9]+)?([.-]?RC[0-9]+)?([.-]?B[0-9]+)?' "$f" 2>/dev/null | head -1)
    [ -n "$v" ] && { printf '%s' "$v"; return 0; }
    return 1
}

# 将 CANN 版本转换为可比较的数值串，避免 9.0.0 比 9.1.0 排在前面
ver_key() {
    printf '%s' "$1" | awk -F. '{printf "%012d%012d%012d", ($1+0), ($2+0), ($3+0)}'
}

BEST_VER=""
BEST_KEY=""
BEST_FILE=""
BEST_DIR=""
record_version() {
    local v="$1" file="$2" dir="$3" key
    key=$(ver_key "$v")
    if [ -z "$BEST_KEY" ]; then
        BEST_KEY="$key"
        BEST_VER="$v"
        BEST_FILE="$file"
        BEST_DIR="$dir"
    elif [[ "$key" > "$BEST_KEY" ]]; then
        BEST_KEY="$key"
        BEST_VER="$v"
        BEST_FILE="$file"
        BEST_DIR="$dir"
    fi
}

# 从文件中解析版本并记录
consider_version_file() {
    local vfile="$1" v
    v=$(parse_version_file "$vfile") || return 0
    record_version "$v" "$vfile" "$(dirname "$vfile")"
}

# 从目录名中解析版本并记录，例如 /usr/local/Ascend/cann-9.1.0
consider_dir_version() {
    local d="$1" v
    v=$(printf '%s' "$d" | grep -oiE 'cann-[0-9]+\.[0-9]+(\.[0-9]+)?([.-]?RC[0-9]+)?' | head -1 | sed 's/^cann-//' )
    [ -z "$v" ] && v=$(printf '%s' "$d" | grep -oE 'ascend-toolkit/[0-9]+\.[0-9]+(\.[0-9]+)?([.-]?RC[0-9]+)?' | head -1 | sed 's#.*/##')
    [ -n "$v" ] && record_version "$v" "" "$d"
}

# 4) 高优先级目录，直接认根目录 version.cfg / version / version.info
PRIORITY_DIRS=()
for e in ASCEND_TOOLKIT_HOME ASCEND_HOME_PATH; do
    pv="$(envval "$e")"
    [ -n "$pv" ] && [ -d "$pv" ] && PRIORITY_DIRS+=("$pv")
done
PRIORITY_DIRS+=("/usr/local/Ascend/ascend-toolkit/latest")

for d in "${PRIORITY_DIRS[@]}"; do
    [ -n "$d" ] && [ -d "$d" ] || continue
    for vf in "$d/version.cfg" "$d/version" "$d/version.info"; do
        consider_version_file "$vf"
    done
    [ -n "$BEST_VER" ] && break
done

# 5) 所有候选目录，只认根目录的 CANN 版本文件；version.info 放在 version.cfg/version 之后降低权重
if [ -z "$BEST_VER" ]; then
    for d in "${SEARCH_DIRS[@]}"; do
        [ -d "$d" ] || continue
        for vf in "$d/version.cfg" "$d/version"; do
            consider_version_file "$vf"
        done
    done
fi
if [ -z "$BEST_VER" ]; then
    for d in "${SEARCH_DIRS[@]}"; do
        [ -d "$d" ] || continue
        consider_version_file "$d/version.info"
    done
fi

# 6) 仍无法确定时，递归查找 version.cfg / version。
#    这里不递归 version.info，避免把 mindstudio-debugger 等子组件版本误认为 CANN。
if [ -z "$BEST_VER" ]; then
    for d in "${SEARCH_DIRS[@]}"; do
        [ -d "$d" ] || continue
        while IFS= read -r vf; do
            consider_version_file "$vf"
        done < <(find -H "$d" -maxdepth 4 \( -name driver -o -name firmware -o -name nnal -o -name hdk -o -name opp \) -prune -o -type f \( -name 'version.cfg' -o -name 'version' \) -print 2>/dev/null | sort)
    done
fi

# 7) 最后用目录名识别，例如 /usr/local/Ascend/cann-9.1.0 或 ascend-toolkit/9.1.0
if [ -z "$BEST_VER" ]; then
    for d in "${UNIQ_DIRS[@]}"; do
        consider_dir_version "$d"
    done
fi

# 8) 应用最佳识别结果，并把 TOOLKIT_DIR 修正为版本文件真正所在目录
if [ -n "$BEST_VER" ]; then
    CANN_VER="$BEST_VER"
    VERSION_FILE="$BEST_FILE"
    TOOLKIT_DIR="$BEST_DIR"
fi

if [ -n "$CANN_VER" ]; then
    ok "识别到 CANN Toolkit 版本: $CANN_VER"
    printf '  %-14s: %s\n' "安装目录" "$TOOLKIT_DIR"
    if [ -n "$VERSION_FILE" ]; then
        printf '  %-14s: %s\n' "版本文件" "$VERSION_FILE"
    else
        printf '  %-14s: %s\n' "版本来源" "目录名推断: $TOOLKIT_DIR"
    fi
else
    if [ -n "$TOOLKIT_DIR" ]; then
        warn "发现 CANN Toolkit 安装目录, 但未找到可用的 version 文件"
        printf '  %-14s: %s\n' "目录" "$TOOLKIT_DIR"
    else
        warn "未识别 CANN Toolkit 版本文件"
    fi
fi

# 5) OPP(算子原型库)识别
OPP_DIR="$(envval ASCEND_OPP_PATH)"
[ -z "$OPP_DIR" ] && [ -n "$TOOLKIT_DIR" ] && [ -d "$TOOLKIT_DIR/opp" ] && OPP_DIR="$TOOLKIT_DIR/opp"
if [ -n "$OPP_DIR" ] && [ -d "$OPP_DIR" ]; then
    echo -e "  ${GREEN}•${RESET} OPP(算子原型库): $OPP_DIR"
else
    [ -n "$TOOLKIT_DIR" ] && warn "未找到 OPP(算子原型库), 算子开发通常需要 kernels/opp"
fi

# 6) 激活环境与 set_env.sh
if [ -n "$(envval ASCEND_HOME_PATH)" ] && [ -n "$(envval ASCEND_TOOLKIT_HOME)" ]; then
    SOURCE_STATE="部分/已设置 ASCEND_HOME_PATH、ASCEND_TOOLKIT_HOME"
else
    SOURCE_STATE="未在当前 shell 检测到完整 ASCEND_* 激活变量"
fi

if [ -n "$TOOLKIT_DIR" ] && [ -f "$TOOLKIT_DIR/set_env.sh" ]; then
    SETENV="$TOOLKIT_DIR/set_env.sh"
fi
if [ -z "$SETENV" ] && [ -n "$TOOLKIT_DIR" ] && [ -f "$(dirname "$TOOLKIT_DIR")/set_env.sh" ]; then
    SETENV="$(dirname "$TOOLKIT_DIR")/set_env.sh"
fi
if [ -z "$SETENV" ]; then
    for p in "$(envval ASCEND_HOME_PATH)/set_env.sh" /usr/local/Ascend/ascend-toolkit/set_env.sh /usr/local/Ascend/ascend-toolkit/latest/set_env.sh /usr/local/Ascend/set_env.sh; do
        if [ -n "$p" ] && [ -f "$p" ]; then SETENV="$p"; break; fi
    done
fi

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
    warn "Toolkit 未识别, 跳过芯片级矩阵判断"
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
printf '  %-14s: %s\n' "Toolkit" "${CANN_VER:-未识别}"
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

# ============================================================
# 6. 环境补齐 / 下载安装
# ============================================================
section "6. 环境补齐 / 下载安装"

# ---------- 定位 itool 仓库根目录（用于跳转安装脚本） ----------
CHOICE_SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd || echo "$PWD")
CHOICE_REPO_ROOT="${ITOOL_REPO_ROOT:-}"
if [ -z "$CHOICE_REPO_ROOT" ] && [ -f "$PWD/itool.sh" ]; then CHOICE_REPO_ROOT="$PWD"; fi
if [ -z "$CHOICE_REPO_ROOT" ] && have git; then
    CHOICE_REPO_ROOT=$(git -C "$CHOICE_SCRIPT_DIR" rev-parse --show-toplevel 2>/dev/null || true)
fi
if [ -z "$CHOICE_REPO_ROOT" ]; then
    CHOICE_D="$CHOICE_SCRIPT_DIR"
    while [ "$CHOICE_D" != "/" ]; do
        if [ -f "$CHOICE_D/itool.sh" ]; then CHOICE_REPO_ROOT="$CHOICE_D"; break; fi
        CHOICE_D=$(dirname "$CHOICE_D")
    done
fi
[ -z "$CHOICE_REPO_ROOT" ] && CHOICE_REPO_ROOT="$CHOICE_SCRIPT_DIR"
CANN_BASE_DIR="$CHOICE_REPO_ROOT/d.ops_develop/d.install_cann"

# 按芯片推荐 CANN 版本与对应安装目录
RECOMMEND_CANN_VER="9.1.0"; RECOMMEND_CANN_DIR="a.cann-9.1.0"
case "$CHIP" in
    950)            RECOMMEND_CANN_VER="9.0.0";   RECOMMEND_CANN_DIR="b.cann-9.0.0" ;;
    910A|910B|910C) RECOMMEND_CANN_VER="8.1.RC1"; RECOMMEND_CANN_DIR="d.cann-8.1.RC1" ;;
    310P)           RECOMMEND_CANN_VER="8.1.RC1"; RECOMMEND_CANN_DIR="d.cann-8.1.RC1" ;;
esac

show_installed_versions() {
    echo ""
    echo -e "  ${CYAN}────────── 目前环境已安装版本 ──────────${RESET}"
    printf '  %-12s: %s\n' "芯片型号" "$CHIP${CHIP_LIST:+  ($CHIP_LIST)}"
    printf '  %-12s: %s\n' "Python" "${PY_VER:-未知}"
    printf '  %-12s: %s\n' "CANN/toolkit" "${CANN_VER:-未识别}"
    printf '  %-12s: %s\n' "安装目录" "${TOOLKIT_DIR:-未发现}"
    printf '  %-12s: %s\n' "激活脚本" "${SETENV:-未找到}"
    printf '  %-12s: %s\n' "torch" "${TORCH_VER:-未安装}"
    printf '  %-12s: %s\n' "torch_npu" "${TORCH_NPU_VER:-未安装}"
    echo -e "  ${CYAN}────────────────────────────────────────${RESET}"
}

run_cann_install() {
    local dir="$1" ver="$2" mode="$3"
    local script="$CANN_BASE_DIR/$dir/run.sh" log rc
    if [ ! -f "$script" ]; then
        warn "未找到安装脚本: $script"
        return 1
    fi
    echo ""
    if [ "$mode" = "auto" ]; then
        echo -e "  ${CYAN}按推荐自动下载/安装 CANN $ver ...${RESET}"
        log=$(mktemp "/tmp/itool_cann_install.XXXXXX" 2>/dev/null || echo "/tmp/itool_cann_install.log")
        env ITOOL_AUTO_DL=1 ITOOL_AUTO_INSTALL=1 QUIET=1 bash "$script" </dev/null >"$log" 2>&1
        rc=$?
        if [ $rc -eq 0 ]; then
            echo -e "  ${GREEN}✔ 按推荐下载安装成功。${RESET}"
            grep -E '✔|安装完成|激活脚本|安装与激活完成|acl OK' "$log" 2>/dev/null | tail -10 | sed 's/^/      /' || true
            return 0
        fi
        echo -e "  ${RED}✖ 按推荐下载安装失败(退出码 $rc)。${RESET}"
        echo -e "  ${YELLOW}失败原因(脚本最近输出):${RESET}"
        tail -n 15 "$log" 2>/dev/null | sed 's/^/      /'
        show_installed_versions
        return 1
    fi
    # 交互模式：直接接过去，由安装脚本接管输入输出
    echo -e "  ${CYAN}跳转安装 CANN $ver ...${RESET}"
    bash "$script"
    rc=$?
    if [ $rc -ne 0 ]; then
        warn "安装脚本退出码: $rc"
        show_installed_versions
    fi
    return $rc
}

# 保留有用的下一步提示
if [ -n "$SETENV" ]; then
    echo -e "  ${YELLOW}提示: 激活当前环境:${RESET} source $SETENV"
fi
[ -n "$RECOMMEND_NOTE" ] && echo -e "  ${YELLOW}版本建议:${RESET} $RECOMMEND_NOTE"

# ---------- 安装菜单 ----------
while true; do
    echo ""
    echo -e "  ${CYAN}请选择要下载/安装的环境组件（跳转 d.install_cann）:${RESET}"
    echo -e "    ${GREEN}1${RESET}) CANN 9.1.0   (较新稳定)"
    echo -e "    ${GREEN}2${RESET}) CANN 9.0.0   (稳定)"
    echo -e "    ${GREEN}3${RESET}) CANN 8.2.RC1 (旧芯片兼容)"
    echo -e "    ${GREEN}4${RESET}) CANN 8.1.RC1 (旧芯片兼容)"
    echo -e "    ${GREEN}r${RESET}) 按推荐下载安装 (CANN ${RECOMMEND_CANN_VER})"
    echo -e "    ${GREEN}s${RESET}) 跳过，不安装"
    printf '  请选择 (1/2/3/4/r/s) [s]: '
    IFS= read -r MENU_CHOICE || MENU_CHOICE="s"
    [ -n "$MENU_CHOICE" ] || MENU_CHOICE="s"

    case "$MENU_CHOICE" in
        1) run_cann_install "a.cann-9.1.0"   "9.1.0"   interactive ;;
        2) run_cann_install "b.cann-9.0.0"   "9.0.0"   interactive ;;
        3) run_cann_install "c.cann-8.2.RC1" "8.2.RC1" interactive ;;
        4) run_cann_install "d.cann-8.1.RC1" "8.1.RC1" interactive ;;
        r|R) run_cann_install "$RECOMMEND_CANN_DIR" "$RECOMMEND_CANN_VER" auto ;;
        s|S) echo -e "  ${WHITE}已跳过安装。${RESET}"; break ;;
        *)   warn "输入无效，请重新选择。"; continue ;;
    esac
    echo -e "  ${GREEN}[完成]${RESET} 安装完成后可重新运行本环境检查脚本复查版本。"
done

echo ""
echo "完成时间: $(date '+%F %T')"
