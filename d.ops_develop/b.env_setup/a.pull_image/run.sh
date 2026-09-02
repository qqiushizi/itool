#!/bin/bash
# ============================================================
# ③ 镜像拉取 (quay.io/ascend/cann 可视化选择 tag)
# 功能: 从 quay.io 拉取 CANN 镜像, 可按 芯片/版本/系统/Python 筛选,
#       以编号列表可视化选择 tag 后 docker pull + 打本地短标签。
# 网络健壮性:
#   - 依次尝试 quay.io 官方 API / Docker Registry v2 API
#   - 每个接口最多重试 3 次, 带超时
#   - 获取失败时提供: 重试 / 内置常见 tag / 手动输入镜像 三种兜底
# 用法:
#   bash run.sh                                   # 交互式可视化选择
#   IMAGE=quay.io/ascend/cann:8.1.rc1-910b-ubuntu22.04-py3.10 bash run.sh
#   CHIP=910b CANN_VERSION=8.1.rc1 OS_TAG=ubuntu22.04 PY_TAG=py3.10 bash run.sh
# 环境变量:
#   REGISTRY=quay.io/ascend/cann  镜像仓库(默认)
#   CHIP / CANN_VERSION / OS_TAG / PY_TAG  自动筛选(跳过交互)
# ============================================================
set -uo pipefail

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; WHITE='\033[1;37m'; RESET='\033[0m'

have() { command -v "$1" >/dev/null 2>&1; }
ask() { local p="$1" d="$2"; printf "  %s [%s]: " "$p" "$d"; IFS= read -r REPLY || REPLY=""; [ -z "$REPLY" ] && REPLY="$d"; }

command -v docker >/dev/null 2>&1 || { echo -e "${RED}未找到 docker, 请先安装。${RESET}" >&2; exit 1; }

REGISTRY="${REGISTRY:-quay.io/ascend/cann}"
IMAGE="${IMAGE:-}"

# ============================================================
# 直接指定 IMAGE (非交互)
# ============================================================
if [ -n "$IMAGE" ]; then
    echo "=========================================================="
    echo " 镜像拉取"
    echo "   $IMAGE"
    echo "=========================================================="
    docker pull "$IMAGE" || { echo -e "${RED}拉取失败: $IMAGE${RESET}" >&2; exit 1; }
    echo -e "${GREEN}✔ 拉取完成: $IMAGE${RESET}"
    exit 0
fi

# ============================================================
# 拉取 tag 列表 (多接口 + 重试)
# ============================================================
API_HOST="${REGISTRY%%/*}"   # quay.io
API_PATH="${REGISTRY#*/}"    # ascend/cann

fetch_tags() {
    local urls=() url i resp code body
    if [ "$API_HOST" = "quay.io" ]; then
        urls+=("https://quay.io/api/v1/repository/${API_PATH}/tag/?limit=2000&onlyActiveTags=true")
    fi
    urls+=("https://${API_HOST}/v2/${API_PATH}/tags/list?n=2000")

    for url in "${urls[@]}"; do
        for i in 1 2 3; do
            resp=$(curl -sS --connect-timeout 10 --max-time 45 -w $'\n%{http_code}' "$url" 2>/dev/null)
            code="${resp##*$'\n'}"
            body="${resp%$'\n'*}"
            if [ "$code" = "200" ] && [ -n "$body" ]; then
                printf '%s' "$body"
                return 0
            fi
            [ "$i" -lt 3 ] && sleep 1
        done
    done
    return 1
}

parse_tags() { # $1=输出文件
    local out="$1"
    if have python3; then
        TAGS_JSON="$TAGS_JSON" python3 - "$out" <<'PY'
import json, os, sys
raw = os.environ.get('TAGS_JSON', '')
try:
    data = json.loads(raw)
except Exception:
    data = {}
tags = data.get('tags') or []
res = []
for t in tags:
    if isinstance(t, dict):
        n = t.get('name') or t.get('manifest_digest')
        if n:
            res.append(str(n))
    else:
        res.append(str(t))
with open(sys.argv[1], 'w') as f:
    f.write('\n'.join(res))
    if res:
        f.write('\n')
PY
    else
        printf '%s' "$TAGS_JSON" | grep -oE '"[^"]+"' | tr -d '"' > "$out"
    fi
}

# 内置常见 tag(网络/接口不可达时的兜底)
COMMON_TAGS=(
    "8.1.rc1-910b-ubuntu22.04-py3.10"
    "8.1.rc1-910b-ubuntu24.04-py3.10"
    "8.1.rc1-910a-ubuntu22.04-py3.10"
    "8.1.rc1-310p-ubuntu22.04-py3.10"
    "9.0.0-910b-ubuntu22.04-py3.10"
    "9.0.0-910a-ubuntu22.04-py3.10"
    "9.1.0-910b-ubuntu22.04-py3.10"
)

TAGS_FILE="/tmp/itool-cann-tags-$$.txt"

echo -e "${CYAN}正在查询 ${REGISTRY} 的 tag 列表 ...${RESET}"
TAGS_JSON=""
TAGS_JSON="$(fetch_tags)"

if [ -z "$TAGS_JSON" ]; then
    echo ""
    echo -e "  ${RED}无法获取 tag 列表: ${REGISTRY}${RESET}"
    echo -e "  可能原因: 网络不通 / DNS 解析失败 / 仓库不存在 / 需要代理 / 接口被限流"
    echo ""
    echo -e "  兜底处理方式:"
    echo -e "    ${WHITE}[1]${RESET} 重试获取 tag 列表"
    echo -e "    ${WHITE}[2]${RESET} 使用内置常见 tag 列表选择"
    echo -e "    ${WHITE}[3]${RESET} 手动输入镜像或 tag"
    echo -e "    ${WHITE}[0]${RESET} 退出"
    echo ""
    if [ ! -t 0 ]; then
        echo -e "  ${YELLOW}非交互终端, 已退出。可: 用 IMAGE=... 指定镜像, 或 REGISTRY=内网源 重试。${RESET}"
        rm -f "$TAGS_FILE"
        exit 1
    fi
    printf "  请选择 [2]: "
    IFS= read -r fb || fb=""
    fb="${fb:-2}"
    case "$fb" in
        1)
            echo -e "  ${CYAN}重试获取 ...${RESET}"
            TAGS_JSON="$(fetch_tags)"
            if [ -z "$TAGS_JSON" ]; then
                echo -e "  ${RED}仍然失败。${RESET}" >&2
                echo -e "  建议: 检查网络/代理, 或 REGISTRY= 换成可达镜像源后重试。" >&2
                rm -f "$TAGS_FILE"
                exit 1
            fi
            ;;
        2)
            : > "$TAGS_FILE"
            for t in "${COMMON_TAGS[@]}"; do printf '%s\n' "$t" >> "$TAGS_FILE"; done
            ;;
        3)
            ask "请输入镜像(如 quay.io/ascend/cann:9.0.0-910b-ubuntu22.04-py3.10)或 tag" ""
            local v="$REPLY"
            [ -z "$v" ] && { echo -e "${RED}未输入, 退出。${RESET}" >&2; exit 1; }
            case "$v" in
                *:*|*/*) FULL_IMAGE="$v" ;;
                *)       FULL_IMAGE="${REGISTRY}:${v}" ;;
            esac
            CHOSEN="${FULL_IMAGE##*:}"
            docker pull "$FULL_IMAGE" || { echo -e "${RED}拉取失败: $FULL_IMAGE${RESET}" >&2; exit 1; }
            echo -e "${GREEN}✔ 拉取完成: $FULL_IMAGE${RESET}"
            rm -f "$TAGS_FILE"
            exit 0
            ;;
        *)
            echo -e "  ${YELLOW}已退出。${RESET}"
            rm -f "$TAGS_FILE"
            exit 0
            ;;
    esac
fi

# 若尚未生成 tag 文件(即 fetch 成功路径), 解析
if [ ! -s "$TAGS_FILE" ]; then
    parse_tags "$TAGS_FILE"
fi

TOTAL=$(wc -l < "$TAGS_FILE" | tr -d ' ')
if [ "$TOTAL" -eq 0 ]; then
    echo -e "${RED}仓库无可用 tag(或解析失败)。${RESET}" >&2
    echo -e "  可用 IMAGE= 直接指定镜像后重试。" >&2
    rm -f "$TAGS_FILE"
    exit 1
fi
echo -e "${GREEN}共获取 ${TOTAL} 个 tag。${RESET}"

# ============================================================
# 筛选(交互, 环境变量可跳过)
# ============================================================
filter_tags() { if [ -n "$1" ]; then grep -E "$1" || true; else cat; fi; }

CHIP="${CHIP:-}"; CANN_VERSION="${CANN_VERSION:-}"; OS_TAG="${OS_TAG:-}"; PY_TAG="${PY_TAG:-}"

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

# ============================================================
# 可视化编号列表
# ============================================================
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
    printf "  选择编号 [%d]: " "$N"
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

# 本地短标签: cann-<chip>:<version>
short_chip=$(printf '%s' "$CHOSEN" | grep -oE '910[aAbB]|910|950|310[pP]|310' | head -1 | tr '[:upper:]' '[:lower:]')
short_ver=$(printf '%s' "$CHOSEN" | cut -d- -f1)
LOCAL_TAG="${short_chip:-cann}:${short_ver:-latest}"

echo ""
echo "=========================================================="
echo " 镜像拉取"
echo "   远端: $FULL_IMAGE"
echo "   本地标签: cann-${LOCAL_TAG}"
echo "=========================================================="
docker pull "$FULL_IMAGE" || { echo -e "${RED}拉取失败(可能是镜像 tag 不存在): $FULL_IMAGE${RESET}" >&2; exit 1; }
docker tag "$FULL_IMAGE" "cann-${LOCAL_TAG}"

rm -f "$TAGS_FILE"
echo ""
echo -e "${GREEN}✔ 镜像已就绪:${RESET}"
docker images | grep -E "ascend/cann|cann-" | sed 's/^/  /'
echo ""
echo "下一步实例化容器:"
echo "  bash d.ops_develop/b.env_setup/b.run_container/run.sh cann-${LOCAL_TAG}"
