#!/bin/bash
# ============================================================
# ③ 镜像拉取 (quay.io/ascend/cann 可视化选择 tag)
# 功能: 从 quay.io 拉取 CANN 镜像, 可按 芯片/版本/系统/Python 筛选,
#       以编号列表可视化选择 tag 后 docker pull + 打本地短标签。
# 用法:
#   bash run.sh                                   # 交互式可视化选择
#   IMAGE=quay.io/ascend/cann:8.1.rc1-910b-ubuntu22.04-py3.10 bash run.sh   # 直接指定
#   CHIP=910b CANN_VERSION=8.1.rc1 OS_TAG=ubuntu22.04 PY_TAG=py3.10 bash run.sh
# 环境变量:
#   REGISTRY=quay.io/ascend/cann  镜像仓库(默认)
#   CHIP / CANN_VERSION / OS_TAG / PY_TAG  自动筛选(跳过交互)
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

command -v docker >/dev/null 2>&1 || { echo -e "${RED}未找到 docker, 请先安装。${RESET}" >&2; exit 1; }

REGISTRY="${REGISTRY:-quay.io/ascend/cann}"
IMAGE="${IMAGE:-}"

# ---------- 非交互: 直接指定 IMAGE ----------
if [ -n "$IMAGE" ]; then
    echo "=========================================================="
    echo " 镜像拉取"
    echo "   $IMAGE"
    echo "=========================================================="
    docker pull "$IMAGE"
    echo ""
    echo -e "${GREEN}✔ 拉取完成: $IMAGE${RESET}"
    exit 0
fi

# ---------- 拉取 tag 列表 ----------
echo -e "${CYAN}正在查询 ${REGISTRY} 的 tag 列表 ...${RESET}"
# quay.io/v2/<namespace>/<name>/tags/list ; REGISTRY 形如 quay.io/ascend/cann
API_HOST="${REGISTRY%%/*}"          # quay.io
API_PATH="${REGISTRY#*/}"           # ascend/cann
TAGS_JSON=$(curl -sS --max-time 30 "https://${API_HOST}/v2/${API_PATH}/tags/list?n=2000" 2>/dev/null)

if [ -z "$TAGS_JSON" ]; then
    echo -e "${RED}无法获取 tag 列表(网络不可达或仓库不存在): ${REGISTRY}${RESET}" >&2
    echo "可设置 IMAGE= 直接指定镜像后重试。" >&2
    exit 1
fi

# ---------- 解析 tags(优先 python3, 退化 grep) ----------
TAGS_FILE="/tmp/itool-cann-tags-$$.txt"
if have python3; then
    python3 - <<PY > "$TAGS_FILE"
import json,sys
try:
    data=json.loads('''$TAGS_JSON''')
except Exception:
    data={}
tags=data.get('tags') or data.get('Tags') or []
print('\n'.join(str(t) for t in tags))
PY
else
    printf '%s' "$TAGS_JSON" | grep -oE '"[^"]+"' | tr -d '"' > "$TAGS_FILE"
fi

TOTAL=$(wc -l < "$TAGS_FILE" | tr -d ' ')
if [ "$TOTAL" -eq 0 ]; then
    echo -e "${RED}仓库无可用 tag。${RESET}" >&2
    rm -f "$TAGS_FILE"
    exit 1
fi
echo -e "${GREEN}共获取 ${TOTAL} 个 tag。${RESET}"

# ---------- 筛选(交互, 环境变量可跳过) ----------
filter_tags() { # $1=关键字 ; 从 stdin 过滤
    if [ -n "$1" ]; then
        grep -E "$1" || true
    else
        cat
    fi
}

CHIP="${CHIP:-}"
CANN_VERSION="${CANN_VERSION:-}"
OS_TAG="${OS_TAG:-}"
PY_TAG="${PY_TAG:-}"

if [ -z "$CHIP" ] && [ -z "$CANN_VERSION" ] && [ -z "$OS_TAG" ] && [ -z "$PY_TAG" ]; then
    echo ""
    echo -e "  ${CYAN}按需筛选(直接回车=不限):${RESET}"
    ask "芯片(910b/910a/950/310p, 留空=全部)" "";   CHIP="$REPLY"
    ask "CANN 版本(如 8.1.rc1 / 9.0.0, 留空=全部)" ""; CANN_VERSION="$REPLY"
    ask "系统(如 ubuntu22.04 / openeuler22.03, 留空=全部)" ""; OS_TAG="$REPLY"
    ask "Python(如 py3.10 / py3.11, 留空=全部)" "";    PY_TAG="$REPLY"
fi

MATCHED=$(cat "$TAGS_FILE" | filter_tags "${CHIP:-}" | filter_tags "${CANN_VERSION:-}" | filter_tags "${OS_TAG:-}" | filter_tags "${PY_TAG:-}")
if [ -z "$MATCHED" ]; then
    echo -e "${RED}没有匹配的 tag, 请放宽筛选条件重试。${RESET}" >&2
    rm -f "$TAGS_FILE"
    exit 1
fi

# ---------- 可视化编号列表 ----------
mapfile() { :; } # bash3 无 mapfile; 用数组逐行读
TAG_ARRAY=()
while IFS= read -r line; do
    [ -n "$line" ] && TAG_ARRAY+=("$line")
done <<< "$MATCHED"
N=${#TAG_ARRAY[@]}

echo ""
echo -e "  ${CYAN}════════ 匹配的镜像 tag (${N}) ════════${RESET}"
for ((i=0; i<N; i++)); do
    printf "  %3d) %s\n" "$((i+1))" "${TAG_ARRAY[$i]}"
done
echo -e "  ${CYAN}────────────────────────────────────${RESET}"

if [ "$N" -eq 1 ]; then
    SEL=1
else
    printf "  选择编号 [%d]: " "$N"   # 默认选最后一个(通常最新)
    IFS= read -r ans || ans=""
    ans="${ans:-$N}"
    case "$ans" in
        ''|*[!0-9]*) SEL=1 ;;
        *) SEL=$ans ;;
    esac
    [ "$SEL" -lt 1 ] && SEL=1
    [ "$SEL" -gt "$N" ] && SEL="$N"
fi
CHOSEN="${TAG_ARRAY[$((SEL-1))]}"
FULL_IMAGE="${REGISTRY}:${CHOSEN}"

# 本地短标签: cann-<chip>:<version> (从 tag 解析)
short_chip=$(printf '%s' "$CHOSEN" | grep -oE '910[aAbB]|910|950|310[pP]|310' | head -1 | tr '[:upper:]' '[:lower:]')
short_ver=$(printf '%s' "$CHOSEN" | cut -d- -f1)
LOCAL_TAG="${short_chip:-cann}:${short_ver:-latest}"

echo ""
echo "=========================================================="
echo " 镜像拉取"
echo "   远端: $FULL_IMAGE"
echo "   本地标签: cann-${LOCAL_TAG}"
echo "=========================================================="
docker pull "$FULL_IMAGE"
docker tag "$FULL_IMAGE" "cann-${LOCAL_TAG}"

rm -f "$TAGS_FILE"
echo ""
echo -e "${GREEN}✔ 镜像已就绪:${RESET}"
docker images | grep -E "ascend/cann|cann-" | sed 's/^/  /'
echo ""
echo "下一步实例化容器:"
echo "  bash d.ops_develop/b.env_setup/b.run_container/run.sh cann-${LOCAL_TAG}"
