#!/bin/bash
# ============================================================
# ② CANN toolkit 安装 (下载 + 安装合并, 仅安装 toolkit)
#
# 功能:
#   选择/指定一个主流稳定 CANN version → 自动找包或下载 toolkit → 安装
#   → 自动 source set_env.sh 激活 → 验证 acl。
#
# 本次只安装 Ascend-cann-toolkit_<version>_linux-<arch>.run,
# 不安装 kernels/ops, 也不安装合一包/驱动。
#
# 可用环境变量(非交互):
#   CANN_VERSION=9.1.0
#   ARCH=x86_64|aarch64
#   INSTALL_DIR=/usr/local/Ascend/ascend-toolkit
#   PKG_DIR=./cann_pkgs        # 优先从这个目录找包, 找不到时也可下载到这里
#   CHECK_ONLY=1               # 只探测官方 URL 是否可达, 不下载/不安装
#   ITOOL_AUTO_DL=1            # 缺包时自动下载, 不再询问
#   QUIET=1                    # 安装时增加 --quiet
#   CANN_BASE_URL=https://内网镜像/CANN/CANN%20__VER__
#
# 官方公开源(默认):
#   https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/
#
# 已按官网资源做连通性校验的 toolkit 版本:
#   9.1.0 (推荐) / 9.0.0 / 8.2.RC1 / 8.1.RC1
# ============================================================
set -uo pipefail

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; WHITE='\033[1;37m'; RESET='\033[0m'

have()   { command -v "$1" >/dev/null 2>&1; }
ask()    { # $1=提示 $2=默认值 => $REPLY
    local prompt="$1" def="$2"
    printf "  %s [%s]: " "$prompt" "$def"
    IFS= read -r REPLY || REPLY=""
    [ -z "$REPLY" ] && REPLY="$def"
}
confirm(){ local ans; printf "  %s [y/N]: " "$1"; IFS= read -r ans || ans=""; case "$ans" in y|Y|yes|YES) return 0;; *) return 1;; esac; }

# ---------- 参数 ----------
ARCH="${ARCH:-$(uname -m)}"
case "$ARCH" in
    x86_64|amd64) ARCH="x86_64" ;;
    arm64|aarch64) ARCH="aarch64" ;;
    *) echo -e "${YELLOW}[ WARN]${RESET} 未知架构: $ARCH, 继续按其生成包名。" ;;
esac
INSTALL_DIR="${INSTALL_DIR:-}"
PKG_DIR="${PKG_DIR:-}"
QUIET="${QUIET:-0}"
CHECK_ONLY="${CHECK_ONLY:-0}"
CANN_VERSION="${CANN_VERSION:-}"
CANN_BASE_URL="${CANN_BASE_URL:-https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%20__VER__}"

SUPPORTED_VERSIONS=("9.1.0" "9.0.0" "8.2.RC1" "8.1.RC1")
SUPPORTED_DESC=("9.1.0   [默认/推荐]" "9.0.0" "8.2.RC1 [旧芯片兼容]" "8.1.RC1 [旧芯片兼容]")

valid_version() {
    local v="$1" x
    for x in "${SUPPORTED_VERSIONS[@]}"; do
        [ "$x" = "$v" ] && return 0
    done
    return 1
}

echo ""
echo -e "  ${WHITE}════════════════════════════════════════════════════════════${RESET}"
echo -e "  ${WHITE}  ② CANN toolkit 安装 (仅 toolkit, 下载 + 安装一步完成)${RESET}"
echo -e "  ${WHITE}════════════════════════════════════════════════════════════${RESET}"

# ---------- 1. 选择版本 ----------
if [ -z "$CANN_VERSION" ]; then
    echo ""
    echo -e "  ${CYAN}请选择 CANN toolkit 版本:${RESET}"
    local_i=0
    for desc in "${SUPPORTED_DESC[@]}"; do
        local_i=$((local_i+1))
        echo "    $local_i) $desc"
    done
    printf "  %s" "选择 [1]: "
    IFS= read -r REPLY || REPLY=""
    REPLY="${REPLY:-1}"
    case "$REPLY" in
        [1-9])
            CANN_VERSION="${SUPPORTED_VERSIONS[$((REPLY-1))]:-}"
            ;;
        *)
            CANN_VERSION="${SUPPORTED_VERSIONS[0]}"
            ;;
    esac
fi

if ! valid_version "$CANN_VERSION"; then
    echo -e "${RED}暂不支持的 CANN 版本: $CANN_VERSION${RESET}" >&2
    echo -e "  当前可选择: ${SUPPORTED_VERSIONS[*]}" >&2
    exit 1
fi

# ---------- 2. 组装包名 / URL ----------
PKG="Ascend-cann-toolkit_${CANN_VERSION}_linux-${ARCH}.run"
BASE="${CANN_BASE_URL//__VER__/${CANN_VERSION}}"
URL="${BASE}/${PKG}"

echo ""
echo -e "  ${CYAN}════════════════════════════════════════════════════════════${RESET}"
echo "  版本        : $CANN_VERSION"
echo "  架构        : $ARCH"
echo "  本次安装包  : $PKG"
echo "  安装内容    : 仅 Ascend-cann-toolkit (不含 kernels/ops/驱动)"
echo "  官方下载源  : $BASE"
echo -e "  ${CYAN}════════════════════════════════════════════════════════════${RESET}"

# ---------- 3. 只探测模式 ----------
probe_url() {
    local u="$1"
    if have curl; then
        curl -sS --max-time 30 -r 0-1023 -o /dev/null -w '%{http_code}' "$u" 2>/dev/null || echo ERR
    elif have wget; then
        wget -q --spider -S "$u" 2>&1 | awk '/HTTP\//{print $2}' | tail -1
    else
        echo NO_TOOL
    fi
}

if [ "$CHECK_ONLY" = "1" ]; then
    echo "  (仅探测 URL, 不下载/不安装)"
    code=$(probe_url "$URL")
    printf '  %-64s -> %s\n' "$PKG" "$code"
    echo ""
    echo "  说明: 200/206=可用; 000=网络不可达; 403/404=URL或版本可能不存在。"
    echo "  若内网无法访问公网, 可设置 CANN_BASE_URL 指向内网镜像。"
    exit 0
fi

# ---------- 4. 找包, 找不到则提示下载 ----------
mkdir -p "$HOME" 2>/dev/null || true
SEARCH_DIRS=()
[ -n "$PKG_DIR" ] && SEARCH_DIRS+=("$PKG_DIR")
SEARCH_DIRS+=("$PWD" "$PWD/cann_pkgs" "$HOME/cann_pkgs" "/tmp")
for t in /tmp/itool-*; do [ -d "$t" ] && SEARCH_DIRS+=("$t"); done
SEARCH_DIRS=($(printf '%s\n' "${SEARCH_DIRS[@]}" | awk '!seen[$0]++'))

PKG_PATH=""
for d in "${SEARCH_DIRS[@]}"; do
    if [ -n "$d" ] && [ -f "$d/$PKG" ]; then
        PKG_PATH="$d/$PKG"
        break
    fi
done

if [ -n "$PKG_PATH" ]; then
    echo -e "  ${GREEN}[找到]${RESET} $PKG_PATH"
else
    [ -n "$PKG_DIR" ] || PKG_DIR="$PWD/cann_pkgs"
    mkdir -p "$PKG_DIR"
    PKG_PATH="$PKG_DIR/$PKG"

    if [ -z "${ITOOL_AUTO_DL:-}" ]; then
        echo -e "  ${YELLOW}[缺失]${RESET} $PKG"
        echo ""
        confirm "是否自动下载 toolkit 到 $PKG_DIR ?" || { echo -e "${RED}已取消。${RESET}" >&2; exit 1; }
    fi

    echo ""
    echo -e "  ${CYAN}下载: $URL${RESET}"
    if have wget; then
        wget -c -O "$PKG_PATH" "$URL"
    elif have curl; then
        curl -L -C - -o "$PKG_PATH" "$URL"
    else
        echo -e "${RED}未找到 wget/curl, 无法下载。${RESET}" >&2
        exit 1
    fi
    rc=$?
    if [ $rc -ne 0 ] || [ ! -s "$PKG_PATH" ]; then
        echo -e "${RED}下载失败(退出码 $rc)。${RESET}" >&2
        echo "可先探测: CHECK_ONLY=1 CANN_VERSION=$CANN_VERSION bash d.ops_develop/b.env_setup/a.install_cann/run.sh" >&2
        exit 1
    fi
fi

# ---------- 5. 安装位置 / SUDO ----------
if [ -z "$INSTALL_DIR" ]; then
    ask "安装位置" "/usr/local/Ascend/ascend-toolkit"; INSTALL_DIR="$REPLY"
fi
mkdir -p "$INSTALL_DIR" 2>/dev/null || true

SUDO=""
if [ "$(id -u)" != "0" ]; then
    if have sudo; then
        SUDO="sudo"
    else
        echo -e "${YELLOW}非 root 且无 sudo, 将直接尝试安装(可能失败)。${RESET}"
    fi
fi

# ---------- 6. 执行安装 ----------
echo ""
echo -e "  ${CYAN}════════════════════════════════════════════════════════════${RESET}"
echo "  即将执行:"
echo "    $SUDO $PKG_PATH --install --install-path=$INSTALL_DIR $([[ "$QUIET" = "1" ]] && echo --quiet || true)"
echo -e "  ${CYAN}════════════════════════════════════════════════════════════${RESET}"

if [ -z "${ITOOL_AUTO_INSTALL:-}" ]; then
    confirm "确认开始安装?" || { echo -e "${RED}已取消。${RESET}" >&2; exit 1; }
fi

chmod +x "$PKG_PATH" 2>/dev/null || true
QUIET_FLAG=""
[ "$QUIET" = "1" ] && QUIET_FLAG="--quiet"
$SUDO "$PKG_PATH" --install --install-path="$INSTALL_DIR" $QUIET_FLAG
rc=$?
if [ $rc -ne 0 ]; then
    echo -e "${RED}安装失败(退出码 $rc)。${RESET}" >&2
    exit $rc
fi

echo ""
echo -e "${GREEN}✔ CANN toolkit 安装完成。${RESET}"

# ---------- 7. source 激活并验证 ----------
SETENV=""
for s in "$INSTALL_DIR/set_env.sh" "$INSTALL_DIR/latest/set_env.sh" "$INSTALL_DIR/ascend-toolkit/set_env.sh" \
         /usr/local/Ascend/ascend-toolkit/set_env.sh /usr/local/Ascend/ascend-toolkit/latest/set_env.sh; do
    [ -f "$s" ] && SETENV="$s" && break
done

echo ""
echo -e "  ${CYAN}===== 激活 CANN 环境 =====${RESET}"
if [ -z "$SETENV" ]; then
    echo -e "${YELLOW}未自动找到 set_env.sh, 请手动确认:${RESET}"
    echo "  source $INSTALL_DIR/set_env.sh"
    exit 0
fi

echo "  激活脚本: $SETENV"
# shellcheck disable=SC1090
source "$SETENV" 2>/dev/null || true

if python3 -c "import acl; print('  acl OK, soc:', acl.get_soc_name())" 2>/dev/null; then
    echo -e "  ${GREEN}[ OK ]${RESET} CANN Python 接口可用, 环境已激活。"
else
    echo -e "  ${YELLOW}[ WARN]${RESET} acl 导入失败(可能未安装 python 包或版本不匹配)。"
    echo "          手动验证: source $SETENV && python3 -c \"import acl;print(acl.get_soc_name())\""
fi

echo ""
echo -e "  ${CYAN}===== 永久激活(可选) =====${RESET}"
echo "  如需每次登录自动激活, 可执行:"
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
echo "下一步: 拉取镜像 → bash d.ops_develop/b.env_setup/b.pull_image/run.sh"
