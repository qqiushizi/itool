#!/bin/bash
# ============================================================
# 自动下载 CANN 包 (toolkit + kernel)
# 用法:
#   CANN_VERSION=8.1.RC1 CHIP=910b ARCH=x86_64 bash run.sh        # 真实下载
#   CANN_VERSION=8.1.RC1 CHIP=910b CHECK_ONLY=1 bash run.sh       # 只探测 URL 是否可达
#   CANN_BASE_URL=https://内网镜像/CANN/... bash run.sh           # 覆盖下载源
# 说明:
#   - 公开源(ascend OBS) toolkit/kernel 可直接下载(已验证 200/206)。
#   - 若公网不可达, 用 CANN_BASE_URL 覆盖为内网镜像。
#   - 下载产物保存到 ./cann_pkgs, 支持断点续传。
# ============================================================
set -euo pipefail

CANN_VERSION="${CANN_VERSION:-8.1.RC1}"
CHIP="${CHIP:-910b}"
ARCH="${ARCH:-$(uname -m)}"          # x86_64 / aarch64
OUT_DIR="${OUT_DIR:-./cann_pkgs}"
CHECK_ONLY="${CHECK_ONLY:-0}"
# 默认公开下载源(路径含 %20, 请勿随意改); 建议内部覆盖:
CANN_BASE_URL="${CANN_BASE_URL:-https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%20${CANN_VERSION}}"

mkdir -p "$OUT_DIR"

# 包名规则(华为官方命名, 已实测):
#   Ascend-cann-toolkit_<版本>_linux-<arch>.run
#   Ascend-cann-kernels-<芯片>_<版本>_linux-<arch>.run
TOOLKIT_PKG="Ascend-cann-toolkit_${CANN_VERSION}_linux-${ARCH}.run"
KERNEL_PKG="Ascend-cann-kernels-${CHIP}_${CANN_VERSION}_linux-${ARCH}.run"

have() { command -v "$1" >/dev/null 2>&1; }

probe_url() {  # $1=url ; 输出 "HTTP码 大小" 到 stdout
    local url="$1"
    if have curl; then
        curl -sS --max-time 30 -r 0-1023 -o /dev/null -w '%{http_code}' "$url" 2>/dev/null || echo ERR
    elif have wget; then
        wget -q --spider -S "$url" 2>&1 | awk '/HTTP\//{print $2}' | tail -1
    else
        echo "NO_TOOL"
    fi
}

if [ "$CHECK_ONLY" = "1" ]; then
    echo "=========================================================="
    echo " CANN 包 URL 探测(不下载)"
    echo "   版本: $CANN_VERSION   芯片: $CHIP   架构: $ARCH"
    echo "=========================================================="
    for pkg in "$TOOLKIT_PKG" "$KERNEL_PKG"; do
        url="${CANN_BASE_URL}/${pkg}"
        code=$(probe_url "$url")
        printf '  %-56s -> %s\n' "$pkg" "$code"
    done
    echo ""
    echo "说明: 200/206=可用; 403=无权或文件名不对; 000=网络不可达。"
    echo "正式下载: 去掉 CHECK_ONLY=1 重跑即可。"
    exit 0
fi

download() {
    local url="$1" dest="$2"
    echo -e "  \033[0;36m下载: $url\033[0m"
    if have wget; then
        wget -c -O "$dest" "$url"
    elif have curl; then
        curl -L -C - -o "$dest" "$url"
    else
        echo -e "  \033[0;31m未找到 wget/curl, 请先安装。\033[0m" >&2
        exit 1
    fi
}

echo "=========================================================="
echo " CANN 包自动下载"
echo "   版本: $CANN_VERSION   芯片: $CHIP   架构: $ARCH"
echo "   保存: $OUT_DIR"
echo "=========================================================="

download "${CANN_BASE_URL}/${TOOLKIT_PKG}" "$OUT_DIR/$TOOLKIT_PKG"
download "${CANN_BASE_URL}/${KERNEL_PKG}" "$OUT_DIR/$KERNEL_PKG"

echo ""
echo -e "\033[0;32m下载完成, 产物在 $OUT_DIR\033[0m"
ls -lh "$OUT_DIR"
echo ""
echo "下一步安装(需 root):"
echo "  chmod +x $OUT_DIR/$TOOLKIT_PKG && sudo $OUT_DIR/$TOOLKIT_PKG --install"
echo "  chmod +x $OUT_DIR/$KERNEL_PKG && sudo $OUT_DIR/$KERNEL_PKG --install"
echo "安装后执行: source /usr/local/Ascend/ascend-toolkit/set_env.sh"
