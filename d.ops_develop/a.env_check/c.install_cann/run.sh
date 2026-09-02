#!/bin/bash
# ============================================================
# ② CANN 安装 (交互式)
# 功能: 选择安装方式 → 确定安装位置 → 执行安装 → source 激活并验证。
#   安装方式:
#     1) toolkit + kernels(ops)  算子开发推荐, 不装驱动
#     2) 仅 toolkit              最小安装
#     3) 合一包(驱动+toolkit+其他) 单 .run, 含驱动(需确认与硬件/内核匹配)
# 非交互(自动化)可用环境变量:
#   INSTALL_METHOD=all|toolkit|combined
#   CANN_VERSION=8.1.RC1  CHIP=910b  ARCH=x86_64
#   INSTALL_DIR=/usr/local/Ascend/ascend-toolkit
#   PKG_DIR=./cann_pkgs   (包所在目录; 找不到则自动下载)
#   QUIET=1               (静默安装)
# ============================================================
set -uo pipefail

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; WHITE='\033[1;37m'; RESET='\033[0m'

have() { command -v "$1" >/dev/null 2>&1; }

ask() { # $1=提示 $2=默认值 ; 结果放 $REPLY
    local prompt="$1" def="$2"
    printf "  %s [%s]: " "$prompt" "$def"
    IFS= read -r REPLY || REPLY=""
    [ -z "$REPLY" ] && REPLY="$def"
}

confirm() { # $1=提示
    local ans
    printf "  %s [y/N]: " "$1"
    IFS= read -r ans || ans=""
    case "$ans" in y|Y|yes|YES) return 0;; *) return 1;; esac
}

# ---------- 参数(环境变量优先) ----------
CANN_VERSION="${CANN_VERSION:-}"
CHIP="${CHIP:-910b}"
ARCH="${ARCH:-$(uname -m)}"
INSTALL_METHOD="${INSTALL_METHOD:-}"
INSTALL_DIR="${INSTALL_DIR:-}"
PKG_DIR="${PKG_DIR:-}"
QUIET="${QUIET:-0}"

# 公网下载源
CANN_BASE_URL="${CANN_BASE_URL:-https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%20__VER__}"

echo ""
echo -e "  ${WHITE}════════════════════════════════════════════════════${RESET}"
echo -e "  ${WHITE}  ② CANN 安装${RESET}"
echo -e "  ${WHITE}════════════════════════════════════════════════════${RESET}"

# ---------- 1. 安装方式 ----------
if [ -z "$INSTALL_METHOD" ]; then
    echo ""
    echo -e "  ${CYAN}请选择安装方式:${RESET}"
    echo "    1) toolkit + kernels(ops)   [算子开发推荐]"
    echo "    2) 仅 toolkit"
    echo "    3) 合一包 (驱动 + toolkit + 其他, 单个 .run)"
    printf "  %s" "选择 [1]: "
    IFS= read -r REPLY || REPLY=""
    REPLY="${REPLY:-1}"
    case "$REPLY" in
        2) INSTALL_METHOD=toolkit ;;
        3) INSTALL_METHOD=combined ;;
        *) INSTALL_METHOD=all ;;
    esac
fi
case "$INSTALL_METHOD" in
    all)      METHOD_DESC="toolkit + kernels(ops)" ;;
    toolkit)  METHOD_DESC="仅 toolkit" ;;
    combined) METHOD_DESC="合一包(驱动+toolkit+其他)" ;;
    *) echo -e "${RED}未知安装方式: $INSTALL_METHOD${RESET}" >&2; exit 1 ;;
esac

# ---------- 2. 版本 / 芯片 ----------
if [ -z "$CANN_VERSION" ]; then
    ask "CANN 版本" "8.1.RC1"; CANN_VERSION="$REPLY"
fi
if [ "$INSTALL_METHOD" = "all" ]; then
    ask "芯片型号(910b/910a/950/310p)" "$CHIP"; CHIP="$REPLY"
fi

# 组包
case "$INSTALL_METHOD" in
    all)      PKGS=("Ascend-cann-toolkit_${CANN_VERSION}_linux-${ARCH}.run"
                    "Ascend-cann-kernels-${CHIP}_${CANN_VERSION}_linux-${ARCH}.run") ;;
    toolkit)  PKGS=("Ascend-cann-toolkit_${CANN_VERSION}_linux-${ARCH}.run") ;;
    combined) PKGS=("Ascend-cann-${CANN_VERSION}_linux-${ARCH}.run") ;;
esac

# ---------- 3. 定位/下载安装包 ----------
locate_pkg() { # $1=包名 ; 输出路径或空
    local name="$1" d
    for d in "${SEARCH_DIRS[@]}"; do
        if [ -f "$d/$name" ]; then echo "$d/$name"; return 0; fi
    done
    return 1
}

# 搜索目录: 显式 PKG_DIR, 本目录, ./cann_pkgs, $HOME/cann_pkgs, /tmp, /tmp/itool-*
SEARCH_DIRS=()
[ -n "$PKG_DIR" ] && SEARCH_DIRS+=("$PKG_DIR")
SEARCH_DIRS+=("$PWD" "$PWD/cann_pkgs" "$HOME/cann_pkgs" "/tmp")
for t in /tmp/itool-*; do [ -d "$t" ] && SEARCH_DIRS+=("$t"); done
# 去重
SEARCH_DIRS=($(printf '%s\n' "${SEARCH_DIRS[@]}" | awk '!seen[$0]++'))

PKG_PATHS=()
for p in "${PKGS[@]}"; do
    found=$(locate_pkg "$p" || true)
    if [ -n "$found" ]; then
        PKG_PATHS+=("$found")
        echo -e "  ${GREEN}[找到]${RESET} $found"
    else
        PKG_PATHS+=("")
        echo -e "  ${YELLOW}[缺失]${RESET} $p"
    fi
done

NEED_DL=0
for ((i=0; i<${#PKGS[@]}; i++)); do [ -z "${PKG_PATHS[$i]}" ] && NEED_DL=1; done
if [ "$NEED_DL" = "1" ]; then
    if [ -n "$PKG_DIR" ]; then
        : # 用户指定了目录, 仍可下载到该目录
    else
        PKG_DIR="$PWD/cann_pkgs"
    fi
    mkdir -p "$PKG_DIR"
    if [ -z "${ITOOL_AUTO_DL:-}" ]; then
        echo ""
        confirm "未找到全部安装包, 是否自动下载到 $PKG_DIR ?" || { echo -e "${RED}已取消。可先运行 b.download_cann 下载, 或用 PKG_DIR= 指定包目录。${RESET}" >&2; exit 1; }
    fi
    BASE="${CANN_BASE_URL//__VER__/${CANN_VERSION}}"
    for ((i=0; i<${#PKGS[@]}; i++)); do
        if [ -z "${PKG_PATHS[$i]}" ]; then
            p="${PKGS[$i]}"
            echo -e "  ${CYAN}下载: ${BASE}/${p}${RESET}"
            if have wget; then
                wget -c -O "$PKG_DIR/$p" "${BASE}/${p}"
            elif have curl; then
                curl -L -C - -o "$PKG_DIR/$p" "${BASE}/${p}"
            else
                echo -e "${RED}未找到 wget/curl。${RESET}" >&2; exit 1
            fi
            PKG_PATHS[$i]="$PKG_DIR/$p"
        fi
    done
fi

# ---------- 4. 安装位置 ----------
if [ -z "$INSTALL_DIR" ]; then
    if [ "$INSTALL_METHOD" = "combined" ]; then
        DEF="/usr/local/Ascend"
    else
        DEF="/usr/local/Ascend/ascend-toolkit"
    fi
    ask "安装位置" "$DEF"; INSTALL_DIR="$REPLY"
fi
mkdir -p "$INSTALL_DIR" 2>/dev/null || true

# ---------- 5. 执行安装 ----------
echo ""
echo -e "  ${CYAN}════════════════════════════════════════════════════${RESET}"
echo "  安装方式: $METHOD_DESC"
echo "  版本    : $CANN_VERSION   芯片: $CHIP   架构: $ARCH"
echo "  安装位置: $INSTALL_DIR"
echo -e "  ${CYAN}════════════════════════════════════════════════════${RESET}"

SUDO=""
if [ "$(id -u)" != "0" ]; then
    if have sudo; then
        SUDO="sudo"
    else
        echo -e "${YELLOW}非 root 且无 sudo, 尝试直接安装(可能失败)。${RESET}"
    fi
fi

QUIET_FLAG=""
[ "$QUIET" = "1" ] && QUIET_FLAG="--quiet"

for ((i=0; i<${#PKGS[@]}; i++)); do
    p="${PKG_PATHS[$i]}"
    echo ""
    echo -e "  ${CYAN}▶ 安装: $(basename "$p")${RESET}"
    chmod +x "$p" 2>/dev/null || true
    $SUDO "$p" --install --install-path="$INSTALL_DIR" $QUIET_FLAG
    rc=$?
    if [ $rc -ne 0 ]; then
        echo -e "${RED}安装失败(退出码 $rc): $(basename "$p")${RESET}" >&2
        exit $rc
    fi
done

echo ""
echo -e "${GREEN}✔ 安装完成。${RESET}"

# ---------- 6. source 激活并验证 ----------
echo ""
echo -e "  ${CYAN}===== 激活 CANN 环境 =====${RESET}"
SETENV=""
for s in "$INSTALL_DIR/set_env.sh" "$INSTALL_DIR/ascend-toolkit/set_env.sh" \
         "$INSTALL_DIR/latest/set_env.sh" /usr/local/Ascend/ascend-toolkit/set_env.sh; do
    [ -f "$s" ] && SETENV="$s" && break
done

if [ -z "$SETENV" ]; then
    echo -e "${YELLOW}未自动找到 set_env.sh, 请手动 source:${RESET}"
    echo "  source $INSTALL_DIR/set_env.sh"
    echo "  source /usr/local/Ascend/ascend-toolkit/set_env.sh"
    exit 0
fi

echo "  激活脚本: $SETENV"
# 在当前脚本进程内 source, 便于立即验证
# shellcheck disable=SC1090
source "$SETENV" 2>/dev/null || true

echo ""
echo -e "  ${CYAN}===== 验证 =====${RESET}"
if python3 -c "import acl; print('  acl OK, soc:', acl.get_soc_name())" 2>/dev/null; then
    echo -e "  ${GREEN}[ OK ]${RESET} CANN Python 接口可用, 环境已激活。"
else
    echo -e "  ${YELLOW}[ WARN]${RESET} acl 导入失败(可能未安装 python 包或版本不匹配)。"
    echo -e "          可先手动执行: source $SETENV && python3 -c \"import acl;print(acl.get_soc_name())\""
fi

echo ""
echo -e "  ${CYAN}===== 永久激活(写入 ~/.bashrc) =====${RESET}"
echo "  如需每次登录自动激活, 手动执行:"
echo -e "    ${WHITE}echo \"source $SETENV\" >> ~/.bashrc${RESET}"
if confirm "是否现在帮你写入 ~/.bashrc ?"; then
    if ! grep -qF "source $SETENV" "$HOME/.bashrc" 2>/dev/null; then
        echo "source $SETENV" >> "$HOME/.bashrc"
        echo -e "  ${GREEN}已写入 ~/.bashrc${RESET}"
    else
        echo -e "  ${YELLOW}已存在, 跳过。${RESET}"
    fi
fi

echo ""
echo -e "${GREEN}✔ 安装与激活完成。${RESET}"
echo "下一步: 拉取镜像 → bash d.ops_develop/b.env_setup/a.pull_image/run.sh"
