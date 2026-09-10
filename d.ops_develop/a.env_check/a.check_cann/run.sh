#!/bin/bash
# ============================================================
# ① 服务器 CANN 环境检查 (只读, 不做任何修复/安装)
# 检查内容:
#   1) 主机与 OS          主机名/内核/发行版/架构/python
#   2) 驱动(HDK)          /etc/ascend_install.info、npu-smi、/dev/davinci*
#   3) CANN 安装目录       扫描 /usr/local/Ascend、ASCEND_HOME_PATH 等
#   4) CANN 版本           toolkit / kernels(ops) 版本文件
#   5) 环境变量激活状态     ASCEND_* 变量、set_env.sh、LD_LIBRARY_PATH
#   6) Python/框架/编译链   acl、torch、torch_npu、pybind11、gcc/cmake
#   7) Docker              docker 版本 + ascend/cann 镜像
# 最后输出【汇总报告】+ 下一步建议(只提示, 不执行)。
# ============================================================
set -o pipefail

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; WHITE='\033[1;37m'; RESET='\033[0m'

section() { echo; echo -e "  ${CYAN}===== $1 =====${RESET}"; }
have()    { command -v "$1" >/dev/null 2>&1; }
envval()  { eval "printf '%s' \"\${$1:-}\""; }
ok()      { echo -e "  ${GREEN}[ OK ]${RESET} $1"; }
warn()    { echo -e "  ${YELLOW}[ WARN]${RESET} $1"; }
bad()     { echo -e "  ${RED}[缺失]${RESET} $1"; }

echo ""
echo -e "  ${WHITE}════════════════════════════════════════════════════${RESET}"
echo -e "  ${WHITE}  ① 服务器 CANN 环境检查 (只读)${RESET}"
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
    ok "驱动安装信息 /etc/ascend_install.info 存在"
    sed 's/^/        /' /etc/ascend_install.info 2>/dev/null | head -20
else
    bad "未找到 /etc/ascend_install.info (可能未安装驱动)"
fi
if have npu-smi; then
    ok "npu-smi: $(npu-smi -v 2>/dev/null | head -1 || true)"
else
    warn "未找到 npu-smi"
fi
N_DEV=$(ls /dev/davinci* 2>/dev/null | wc -l | tr -d ' ')
if [ "${N_DEV:-0}" -gt 0 ] 2>/dev/null; then
    ok "检测到 ${N_DEV} 个 NPU 设备: $(ls /dev/davinci* 2>/dev/null | tr '\n' ' ')"
else
    warn "未检测到 /dev/davinci* 设备"
fi

# ============================================================
# 3. CANN 安装目录
# ============================================================
section "3. CANN 安装目录"
TOOLKIT_DIR=""
CANDIDATES=()
[ -n "$(envval ASCEND_HOME_PATH)" ] && CANDIDATES+=("$(envval ASCEND_HOME_PATH)")
[ -n "$(envval ASCEND_TOOLKIT_HOME)" ] && CANDIDATES+=("$(envval ASCEND_TOOLKIT_HOME)")
CANDIDATES+=("/usr/local/Ascend/ascend-toolkit/latest" "/usr/local/Ascend/ascend-toolkit" "/usr/local/Ascend")
FOUND_DIRS=()
for d in "${CANDIDATES[@]}"; do
    [ -n "$d" ] && [ -e "$d" ] && FOUND_DIRS+=("$d")
done
FOUND_DIRS=($(printf '%s\n' "${FOUND_DIRS[@]}" | awk '!seen[$0]++'))
if [ ${#FOUND_DIRS[@]} -eq 0 ]; then
    bad "未发现 CANN 安装目录"
else
    for d in "${FOUND_DIRS[@]}"; do
        [ -d "$d" ] && { echo -e "  ${GREEN}•${RESET} $d"; [ -n "$TOOLKIT_DIR" ] || TOOLKIT_DIR="$d"; }
    done
fi

# ============================================================
# 4. CANN 版本
# ============================================================
section "4. CANN 版本"
G_VERSION=""
print_verfile() { if [ -f "$1" ]; then echo "        --- $1 ---"; sed 's/^/        /' "$1" 2>/dev/null | head -12; fi; }
if [ -n "$TOOLKIT_DIR" ] && [ -d "$TOOLKIT_DIR" ]; then
    for vf in version.cfg version version.info; do print_verfile "$TOOLKIT_DIR/$vf"; done
    G_VERSION=$(grep -iE 'version' "$TOOLKIT_DIR/version.cfg" 2>/dev/null | head -1 | tr -d '[:space:]')
    [ -z "$G_VERSION" ] && G_VERSION=$(head -1 "$TOOLKIT_DIR/version" 2>/dev/null)
    [ -z "$G_VERSION" ] && G_VERSION="(见上方版本文件)"
else
    bad "无 toolkit 目录, 无法读取 CANN 版本"
fi

OPP_DIR="$(envval ASCEND_OPP_PATH)"
[ -z "$OPP_DIR" ] && [ -n "$TOOLKIT_DIR" ] && OPP_DIR="$TOOLKIT_DIR/opp"
if [ -n "$OPP_DIR" ] && [ -d "$OPP_DIR" ]; then
    echo -e "  ${GREEN}•${RESET} OPP(算子原型库): $OPP_DIR"
    print_verfile "$OPP_DIR/version.info"
else
    warn "未找到 OPP(算子原型库) (算子开发需要 kernels)"
fi

# ============================================================
# 5. 环境变量激活状态
# ============================================================
section "5. 环境变量激活状态"
for v in ASCEND_HOME_PATH ASCEND_TOOLKIT_HOME ASCEND_OPP_PATH ASCEND_AICPU_PATH ASCEND_RUNTIME_HOME ASCEND_DRIVER_PATH; do
    val="$(envval "$v")"
    if [ -n "$val" ]; then
        echo -e "  ${GREEN}[已设置]${RESET} $v = $val"
    else
        echo -e "  ${YELLOW}[未设置]${RESET} $v"
    fi
done

SETENV=""
for p in "$TOOLKIT_DIR/set_env.sh" "$(envval ASCEND_HOME_PATH)/set_env.sh" /usr/local/Ascend/ascend-toolkit/set_env.sh /usr/local/Ascend/ascend-toolkit/latest/set_env.sh; do
    [ -f "$p" ] && SETENV="$p" && break
done
if [ -n "$SETENV" ]; then
    ok "找到 CANN 激活脚本: $SETENV"
else
    bad "未找到 set_env.sh (无法激活 CANN 环境)"
fi

LDHAS=$(printf '%s' "${LD_LIBRARY_PATH:-}" | tr ':' '\n' | grep -i ascend | head -1)
if [ -n "$LDHAS" ]; then
    ok "LD_LIBRARY_PATH 已包含 ascend 路径"
else
    warn "LD_LIBRARY_PATH 未包含 ascend 路径 (当前 shell 未 source set_env.sh)"
fi

# ============================================================
# 6. Python / 框架 / 编译链
# ============================================================
section "6. Python / 框架 / 编译链"
if python3 -c "import acl" 2>/dev/null; then
    ok "acl 可导入 (CANN Python 接口正常)"
else
    warn "acl 导入失败 (CANN Python 包未装或环境未激活)"
fi

if python3 -c "import torch" 2>/dev/null; then
    ok "torch: $(python3 -c "import torch;print(torch.__version__)" 2>/dev/null)"
else
    bad "torch 未安装"
fi
if python3 -c "import torch_npu" 2>/dev/null; then
    ok "torch_npu: $(python3 -c "import torch_npu;print(torch_npu.__version__)" 2>/dev/null)"
else
    bad "torch_npu 未安装"
fi
for m in "pybind11 pybind11" "numpy numpy"; do
    set -- $m
    python3 -c "import $1" 2>/dev/null \
        && echo -e "  ${GREEN}[ OK ]${RESET} $1" \
        || echo -e "  ${YELLOW}[缺失]${RESET} $1 (${2})"
done
if have cmake && have gcc; then
    echo -e "  ${GREEN}[ OK ]${RESET} cmake: $(cmake --version 2>/dev/null | head -1)"
    echo -e "  ${GREEN}[ OK ]${RESET} gcc:   $(gcc --version 2>/dev/null | head -1)"
else
    warn "缺少 cmake 或 gcc (算子编译需要)"
fi

# ============================================================
# 7. Docker
# ============================================================
section "7. Docker"
if have docker; then
    ok "docker: $(docker --version 2>/dev/null)"
else
    warn "docker 未安装 (容器化开发需要)"
fi

# ============================================================
# 汇总报告
# ============================================================
echo ""
echo -e "  ${CYAN}════════════════════════════════════════════════════${RESET}"
echo -e "  ${WHITE}  【汇总报告】${RESET}"
echo -e "  ${CYAN}════════════════════════════════════════════════════${RESET}"
printf '  %-16s: %s\n' "安装目录" "${TOOLKIT_DIR:-未发现}"
printf '  %-16s: %s\n' "CANN 版本" "${G_VERSION:-未知}"
printf '  %-16s: %s\n' "激活脚本" "${SETENV:-未找到}"
printf '  %-16s: %s\n' "OPP 算子库" "${OPP_DIR:-未找到}"
printf '  %-16s: %s\n' "NPU 设备数" "${N_DEV:-0}"
echo -e "  ${CYAN}────────────────────────────────────────────────────${RESET}"

# ============================================================
# 下一步建议 (只提示, 不执行)
# ============================================================
echo -e "  ${YELLOW}下一步建议(按需手动执行):${RESET}"
if [ -z "$TOOLKIT_DIR" ]; then
    echo -e "    → 未安装 CANN:        bash d.ops_develop/b.env_setup/b.install_cann/run.sh"
else
    echo -e "    → CANN 已安装。"
fi
if [ -n "$SETENV" ]; then
    echo -e "    → 激活当前环境:      source $SETENV"
elif [ -n "$TOOLKIT_DIR" ]; then
    echo -e "    → 未找到 set_env.sh, 请在 toolkit 目录下确认。"
fi
if ! have docker; then
    echo -e "    → 未安装 docker:      bash f.installation/a.install_ascend/a.install_docker/run.sh"
fi
echo -e "    → 拉取镜像:           bash d.ops_develop/b.env_setup/c.pull_image/run.sh"
echo -e "    → 起容器:             bash d.ops_develop/b.env_setup/d.run_container/run.sh"
echo -e "    → 进容器检查:        bash d.ops_develop/a.env_check/b.check_in_container/run.sh 容器名"
echo ""
echo "完成时间: $(date '+%F %T')"
