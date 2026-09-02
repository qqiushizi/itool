#!/bin/bash
# ============================================================
# 下载 CANN 包 (供 c.install_cann 安装使用)
# 用法:
#   CANN_VERSION=8.1.RC1 CHIP=910b ARCH=x86_64 bash run.sh           # 默认: toolkit + kernels
#   MODE=toolkit  ... bash run.sh                                    # 仅 toolkit
#   MODE=combined ... bash run.sh                                    # 合一包(驱动+toolkit+其他)
#   CHECK_ONLY=1  ... bash run.sh                                    # 只探测 URL, 不下载
#   CANN_BASE_URL=https://内网镜像/CANN/... bash run.sh               # 覆盖下载源
# 说明:
#   - 默认公开源: ascend-repo.obs.cn-east-2.myhuaweicloud.com (OBS)
#   - 包名规则(华为官方命名):
#       Ascend-cann-toolkit_<版本>_linux-<arch>.run        toolkit
#       Ascend-cann-kernels-<芯片>_<版本>_linux-<arch>.run 算子库(kernels)
#       Ascend-cann-<版本>_linux-<arch>.run                 合一包
#   - 产物保存到 ./cann_pkgs, 支持断点续传。
# ============================================================
set -euo pipefail

CANN_VERSION="${CANN_VERSION:-8.1.RC1}"
CHIP="${CHIP:-910b}"
ARCH="${ARCH:-$(uname -m)}"
OUT_DIR="${OUT_DIR:-./cann_pkgs}"
CHECK_ONLY="${CHECK_ONLY:-0}"
MODE="${MODE:-all}"                     # all=toolkit+kernels, toolkit=仅toolkit, combined=合一包
CANN_BASE_URL="${CANN_BASE_URL:-https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%20${CANN_VERSION}}"

mkdir -p "$OUT_DIR"

have() { command -v "$1" >/dev/null 2>&1; }

# 按 MODE 组装要下载的包
PKGS=()
case "$MODE" in
    toolkit)
        PKGS=("Ascend-cann-toolkit_${CANN_VERSION}_linux-${ARCH}.run")
        ;;
    combined)
        PKGS=("Ascend-cann-${CANN_VERSION}_linux-${ARCH}.run")
        ;;
    all|*)
        PKGS=("Ascend-cann-toolkit_${CANN_VERSION}_linux-${ARCH}.run"
              "Ascend-cann-kernels-${CHIP}_${CANN_VERSION}_linux-${ARCH}.run")
        ;;
esac

probe_url() {  # $1=url ; 输出 HTTP 码
    local url="$1"
    if have curl; then
        curl -sS --max-time 30 -r 0-1023 -o /dev/null -w '%{http_code}' "$url" 2>/dev/null || echo ERR
    elif have wget; then
        wget -q --spider -S "$url" 2>&1 | awk '/HTTP\//{print $2}' | tail -1
    else
        echo "NO_TOOL"
    fi
}

echo "=========================================================="
echo " CANN 包下载"
echo "   版本: $CANN_VERSION   芯片: $CHIP   架构: $ARCH"
echo "   模式: $MODE   保存: $OUT_DIR"
echo "=========================================================="

if [ "$CHECK_ONLY" = "1" ]; then
    echo " (仅探测 URL, 不下载)"
    echo "----------------------------------------------------------"
    for pkg in "${PKGS[@]}"; do
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

for pkg in "${PKGS[@]}"; do
    download "${CANN_BASE_URL}/${pkg}" "$OUT_DIR/$pkg"
done

echo ""
echo -e "\033[0;32m下载完成, 产物在 $OUT_DIR\033[0m"
ls -lh "$OUT_DIR"
echo ""
echo "下一步安装(见 a.env_check/c.install_cann):"
echo "  bash d.ops_develop/a.env_check/c.install_cann/run.sh"
