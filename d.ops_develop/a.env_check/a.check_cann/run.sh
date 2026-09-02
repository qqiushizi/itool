#!/bin/bash
# ============================================================
# ① 服务器 CANN 环境检查
# 功能: 汇总客户机上昇腾/CANN 的关键信息, 判断能否开始算子开发。
#   检查内容(agent 汇总):
#     1) 主机与 OS          主机名/内核/发行版/架构
#     2) 驱动(HDK)          /etc/ascend_install.info、npu-smi、/dev/davinci*
#     3) CANN 安装目录       扫描 /usr/local/Ascend、ASCEND_HOME_PATH 等
#     4) CANN 版本           toolkit / kernels(ops) / nnal 版本文件
#     5) 环境变量激活状态     ASCEND_* 变量、set_env.sh、LD_LIBRARY_PATH
#     6) Python 与框架       python/pip、acl、torch、torch_npu、vllm-ascend
#     7) Docker              docker 版本 + ascend 镜像
#   最后输出【汇总报告】: 安装目录 / 版本 / 激活状态 / 下一步建议。
# 说明: 纯检查, 不修改系统。
# ============================================================
set +e

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; WHITE='\033[1;37m'; RESET='\033[0m'

section() { echo; echo -e "  ${CYAN}===== $1 =====${RESET}"; }
have()    { command -v "$1" >/dev/null 2>&1; }
envval()  { eval "printf '%s' \"\${$1:-}\""; }
ok()      { echo -e "  ${GREEN}[ OK ]${RESET} $1"; }
warn()    { echo -e "  ${YELLOW}[ WARN]${RESET} $1"; }
bad()     { echo -e "  ${RED}[缺失]${RESET} $1"; }

# 汇总收集器(脚本内全局变量)
G_DIR=""; G_VERSION=""; G_ACTIVATED="0"; G_DRIVER=""

echo ""
echo -e "  ${WHITE}════════════════════════════════════════════════════${RESET}"
echo -e "  ${WHITE}  ① 服务器 CANN 环境检查${RESET}"
echo -e "  ${WHITE}════════════════════════════════════════════════════${RESET}"

# ============================================================
# 1. 主机与 OS
# ============================================================
section "1. 主机与 OS"
echo "  主机名   : $(hostname 2>/dev/null || echo unknown)"
echo "  内核     : $(uname -r 2>/dev/null)"
echo "  架构     : $(uname -m 2>/dev/null)"
( . /etc/os-release 2>/dev/null && echo "  发行版   : $PRETTY_NAME" ) || echo "  发行版   : (unknown)"
echo "  python3  : $(command -v python3 2>/dev/null || echo '(未找到)') $(python3 -c 'import sys;print(sys.version.split()[0])' 2>/dev/null)"

# ============================================================
# 2. 驱动(HDK)
# ============================================================
section "2. 驱动 (HDK / NPU 设备)"
if [ -f /etc/ascend_install.info ]; then
    G_DRIVER=$(grep -iE '^version|driver' /etc/ascend_install.info 2>/dev/null | head -1)
    ok "驱动安装信息 /etc/ascend_install.info 存在"
    echo "        $(cat /etc/ascend_install.info 2>/dev/null | sed 's/^/        /' | head -20)"
else
    bad "未找到 /etc/ascend_install.info (可能未安装驱动)"
fi

if have npu-smi; then
    ok "npu-smi 可用: $(npu-smi -v 2>/dev/null | head -1 || true)"
else
    warn "未找到 npu-smi (驱动工具不在 PATH)"
fi

N_DEV=$(ls /dev/davinci* 2>/dev/null | wc -l | tr -d ' ')
if [ "$N_DEV" -gt 0 ] 2>/dev/null; then
    ok "检测到 ${N_DEV} 个 NPU 设备: $(ls /dev/davinci* 2>/dev/null | tr '\n' ' ')"
else
    warn "未检测到 /dev/davinci* 设备 (无 NPU 或驱动未加载)"
fi

# ============================================================
# 3. CANN 安装目录
# ============================================================
section "3. CANN 安装目录"
TOOLKIT_DIR=""
# 候选安装根
CANDIDATES=()
[ -n "$(envval ASCEND_HOME_PATH)" ] && CANDIDATES+=("$(envval ASCEND_HOME_PATH)")
[ -n "$(envval ASCEND_TOOLKIT_HOME)" ] && CANDIDATES+=("$(envval ASCEND_TOOLKIT_HOME)")
CANDIDATES+=("/usr/local/Ascend/ascend-toolkit/latest" "/usr/local/Ascend/ascend-toolkit" "/usr/local/Ascend")

FOUND_DIRS=()
for d in "${CANDIDATES[@]}"; do
    [ -n "$d" ] && [ -e "$d" ] && FOUND_DIRS+=("$d")
done
# 去重
FOUND_DIRS=($(printf '%s\n' "${FOUND_DIRS[@]}" | awk '!seen[$0]++'))

if [ ${#FOUND_DIRS[@]} -eq 0 ]; then
    bad "未发现 CANN 安装目录"
else
    for d in "${FOUND_DIRS[@]}"; do
        if [ -d "$d" ]; then
            echo -e "  ${GREEN}•${RESET} $d"
            [ -n "$TOOLKIT_DIR" ] || TOOLKIT_DIR="$d"
        fi
    done
fi
G_DIR="$TOOLKIT_DIR"

# ============================================================
# 4. CANN 版本
# ============================================================
section "4. CANN 版本"
print_verfile() { # $1=路径
    if [ -f "$1" ]; then
        echo "        --- $1 ---"
        sed 's/^/        /' "$1" 2>/dev/null | head -15
    fi
}

if [ -n "$TOOLKIT_DIR" ] && [ -d "$TOOLKIT_DIR" ]; then
    for vf in version.cfg version version.info; do
        print_verfile "$TOOLKIT_DIR/$vf"
    done
    # 版本字符串抓取(优先 version.cfg)
    G_VERSION=$(grep -iE 'version' "$TOOLKIT_DIR/version.cfg" 2>/dev/null | head -1 | tr -d '[:space:]')
    [ -z "$G_VERSION" ] && G_VERSION=$(head -1 "$TOOLKIT_DIR/version" 2>/dev/null)
    [ -z "$G_VERSION" ] && G_VERSION="(见上方版本文件)"
else
    bad "无 toolkit 目录, 无法读取 CANN 版本"
fi

# kernels(ops) 版本
OPP_DIR="$(envval ASCEND_OPP_PATH)"
[ -z "$OPP_DIR" ] && [ -n "$TOOLKIT_DIR" ] && OPP_DIR="$TOOLKIT_DIR/opp"
if [ -n "$OPP_DIR" ] && [ -d "$OPP_DIR" ]; then
    echo -e "  ${GREEN}•${RESET} OPP(算子原型库): $OPP_DIR"
    print_verfile "$OPP_DIR/version.info"
else
    warn "未找到 OPP(算子原型库) 目录 (算子开发必需 kernels)"
fi

# ============================================================
# 5. 环境变量激活状态
# ============================================================
section "5. 环境变量激活状态"
ASCEND_VARS="ASCEND_HOME_PATH ASCEND_TOOLKIT_HOME ASCEND_OPP_PATH ASCEND_AICPU_PATH ASCEND_RUNTIME_HOME ASCEND_DRIVER_PATH"
for v in $ASCEND_VARS; do
    val="$(envval "$v")"
    if [ -n "$val" ]; then
        echo -e "  ${GREEN}[已设置]${RESET} $v = $val"
    else
        echo -e "  ${YELLOW}[未设置]${RESET} $v"
    fi
done

# set_env.sh 探测
SETENV=""
for p in "$TOOLKIT_DIR/set_env.sh" "$(envval ASCEND_HOME_PATH)/set_env.sh" /usr/local/Ascend/ascend-toolkit/set_env.sh; do
    [ -f "$p" ] && SETENV="$p" && break
done
if [ -n "$SETENV" ]; then
    ok "找到激活脚本: $SETENV"
    G_ACTIVATED="1"
else
    bad "未找到 set_env.sh (无法激活 CANN 环境)"
fi

# LD_LIBRARY_PATH 是否包含 ascend
LDHAS=$(printf '%s' "${LD_LIBRARY_PATH:-}" | tr ':' '\n' | grep -i ascend | head -1)
if [ -n "$LDHAS" ]; then
    ok "LD_LIBRARY_PATH 已包含 ascend 路径"
else
    warn "LD_LIBRARY_PATH 未包含 ascend 路径 (当前 shell 可能未 source set_env.sh)"
fi

# ============================================================
# 6. Python / 框架
# ============================================================
section "6. Python 与框架"
if python3 -c "import acl; print('acl', getattr(acl,'__version__','?'))" 2>/dev/null; then
    ok "acl 可导入 (CANN Python 接口正常)"
else
    warn "acl 导入失败 (CANN Python 包未装或环境未激活)"
fi

if python3 -c "import torch; print('torch', torch.__version__)" 2>/dev/null; then
    ok "torch 已安装: $(python3 -c "import torch;print(torch.__version__)" 2>/dev/null)"
else
    bad "torch 未安装 (算子开发/调试需要)"
fi

if python3 -c "import torch_npu; print('torch_npu', torch_npu.__version__)" 2>/dev/null; then
    ok "torch_npu 已安装: $(python3 -c "import torch_npu;print(torch_npu.__version__)" 2>/dev/null)"
else
    bad "torch_npu 未安装 (需与 torch/CANN 版本匹配)"
fi

for m in "vllm_ascend vllm-ascend" "pybind11 pybind11" "numpy numpy"; do
    set -- $m
    if python3 -c "import $1" 2>/dev/null; then
        echo -e "  ${GREEN}[ OK ]${RESET} $1"
    else
        echo -e "  ${YELLOW}[缺失]${RESET} $1 (${2})"
    fi
done

if have cmake; then
    ok "cmake: $(cmake --version 2>/dev/null | head -1)"
else
    warn "cmake 未找到"
fi
if have gcc; then
    ok "gcc: $(gcc --version 2>/dev/null | head -1)"
else
    warn "gcc 未找到"
fi

# ============================================================
# 7. Docker
# ============================================================
section "7. Docker"
if have docker; then
    ok "docker: $(docker --version 2>/dev/null)"
    echo "  镜像(ascend):"
    docker images 2>/dev/null | grep -iE 'ascend|cann' | sed 's/^/    /' || echo "    (无 ascend/cann 镜像)"
else
    warn "docker 未安装 (容器化开发需要, 见 b.env_setup)"
fi

# ============================================================
# 汇总报告
# ============================================================
echo ""
echo -e "  ${CYAN}════════════════════════════════════════════════════${RESET}"
echo -e "  ${WHITE}  【汇总报告】${RESET}"
echo -e "  ${CYAN}════════════════════════════════════════════════════${RESET}"
printf '  %-16s: %s\n' "安装目录" "${G_DIR:-未发现}"
printf '  %-16s: %s\n' "CANN 版本" "${G_VERSION:-未知}"
printf '  %-16s: %s\n' "激活脚本" "${SETENV:-未找到}"
printf '  %-16s: %s\n' "当前激活状态" "$([ "$G_ACTIVATED" = "1" ] && echo '已找到 set_env.sh' || echo '未激活')"
printf '  %-16s: %s\n' "驱动信息" "${G_DRIVER:-未检测到}"
printf '  %-16s: %s\n' "NPU 设备数" "${N_DEV:-0}"
echo -e "  ${CYAN}────────────────────────────────────────────────────${RESET}"
echo -e "  ${YELLOW}下一步建议:${RESET}"
if [ -z "$TOOLKIT_DIR" ]; then
    echo -e "    → 未安装 CANN, 请先: ② 安装 CANN (a.env_check/c.install_cann)"
elif [ "$G_ACTIVATED" != "1" ]; then
    echo -e "    → CANN 已装但缺少激活脚本, 检查: $TOOLKIT_DIR/set_env.sh"
else
    echo -e "    → 环境基本就绪, 可继续: ③ 拉取镜像 → ④ 起容器 → ⑤ 进容器检查"
fi
echo ""
echo "完成时间: $(date '+%F %T')"
