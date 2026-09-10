#!/bin/bash
# ============================================================
# ① 镜像拉取 + 容器实例化 (合并脚本)
#
# 功能:
#   1) 交互/环境变量选择一个 CANN 镜像并 docker pull
#   2) 为该镜像打本地短标签或直接使用 IMAGE=
#   3) 收集容器参数并生成可编辑 start_container.sh
#   4) 立即启动容器
#
# 用法:
#   bash run.sh                                             # 交互式
#   bash run.sh <镜像> [容器名] [工作目录]                    # 位置参数
#   IMAGE=quay.io/ascend/cann:9.0.0-910b-ubuntu22.04-py3.10 bash run.sh
#   CHIP=910b CANN_VERSION=9.0.0 OS_TAG=ubuntu22.04 PY_TAG=py3.10 bash run.sh
#   NAME=asc_dev WORK_DIR=/data/ops SHM_SIZE=16g bash run.sh
#
# 镜像相关环境变量:
#   IMAGE / REGISTRY / CHIP / CANN_VERSION / OS_TAG / PY_TAG
# 容器相关环境变量:
#   NAME / WORK_DIR / SHM_SIZE / EXTRA_ARGS / ITOOL_DEVICES
# ============================================================
set -uo pipefail

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; WHITE='\033[1;37m'; RESET='\033[0m'

have()    { command -v "$1" >/dev/null 2>&1; }
ask()     { local p="$1" d="$2"; printf "  %s [%s]: " "$p" "$d"; IFS= read -r REPLY || REPLY=""; [ -z "$REPLY" ] && REPLY="$d"; }
confirm() { local ans; printf "  %s [y/N]: " "$1"; IFS= read -r ans || ans=""; case "$ans" in y|Y|yes|YES) return 0;; *) return 1;; esac; }

command -v docker >/dev/null 2>&1 || { echo -e "${RED}未找到 docker, 请先安装。${RESET}" >&2; exit 1; }

# 镜像参数
REGISTRY="${REGISTRY:-quay.io/ascend/cann}"
IMAGE="${1:-${IMAGE:-}}"
CHIP="${CHIP:-}"
CANN_VERSION="${CANN_VERSION:-}"
OS_TAG="${OS_TAG:-}"
PY_TAG="${PY_TAG:-}"

# 容器参数
NAME="${2:-${NAME:-}}"
WORK_DIR="${3:-${WORK_DIR:-}}"
SHM_SIZE="${SHM_SIZE:-}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

IMAGE_TO_USE=""
TAGS_FILE="/tmp/itool-cann-tags-$$.txt"

echo ""
echo -e "  ${WHITE}════════════════════════════════════════════════════════════${RESET}"
echo -e "  ${WHITE}  ① CANN 镜像拉取 + 容器实例化（合并脚本）${RESET}"
echo -e "  ${WHITE}════════════════════════════════════════════════════════════${RESET}"

# ============================================================
# 第一部分: 选择并拉取镜像
# ============================================================
if [ -n "$IMAGE" ]; then
    echo ""
    echo -e "  ${CYAN}直接使用指定镜像, 先拉取:${RESET} $IMAGE"
    docker pull "$IMAGE" || { echo -e "${RED}拉取失败: $IMAGE${RESET}" >&2; exit 1; }
    IMAGE_TO_USE="$IMAGE"
else
    API_HOST="${REGISTRY%%/*}"
    API_PATH="${REGISTRY#*/}"

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

    parse_tags() {
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

    COMMON_TAGS=(
        "8.1.rc1-910b-ubuntu22.04-py3.10"
        "8.1.rc1-910b-ubuntu24.04-py3.10"
        "8.1.rc1-910a-ubuntu22.04-py3.10"
        "8.1.rc1-310p-ubuntu22.04-py3.10"
        "9.0.0-910b-ubuntu22.04-py3.10"
        "9.0.0-910a-ubuntu22.04-py3.10"
        "9.1.0-910b-ubuntu22.04-py3.10"
    )

    echo ""
    echo -e "  ${CYAN}正在查询 ${REGISTRY} 的 tag 列表 ...${RESET}"
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
                v="$REPLY"
                [ -z "$v" ] && { echo -e "${RED}未输入, 退出。${RESET}" >&2; exit 1; }
                case "$v" in
                    *:*|*/*) FULL_IMAGE="$v" ;;
                    *)       FULL_IMAGE="${REGISTRY}:${v}" ;;
                esac
                docker pull "$FULL_IMAGE" || { echo -e "${RED}拉取失败: $FULL_IMAGE${RESET}" >&2; exit 1; }
                IMAGE_TO_USE="$FULL_IMAGE"
                rm -f "$TAGS_FILE"
                ;;
            *)
                echo -e "  ${YELLOW}已退出。${RESET}"
                rm -f "$TAGS_FILE"
                exit 0
                ;;
        esac
    fi

    # 若尚未生成 tag 文件, 解析 fetch 结果
    if [ -n "$IMAGE_TO_USE" ]; then
        : # 手动镜像已就绪
    elif [ ! -s "$TAGS_FILE" ]; then
        parse_tags "$TAGS_FILE"

        TOTAL=$(wc -l < "$TAGS_FILE" | tr -d ' ')
        if [ "$TOTAL" -eq 0 ]; then
            echo -e "${RED}仓库无可用 tag(或解析失败)。${RESET}" >&2
            echo -e "  可用 IMAGE= 直接指定镜像后重试。" >&2
            rm -f "$TAGS_FILE"
            exit 1
        fi
        echo -e "${GREEN}共获取 ${TOTAL} 个 tag。${RESET}"

        # ---------- 筛选(环境变量可跳过) ----------
        filter_tags() { if [ -n "$1" ]; then grep -E "$1" || true; else cat; fi; }

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

        short_chip=$(printf '%s' "$CHOSEN" | grep -oE '910[aAbB]|910|950|310[pP]|310' | head -1 | tr '[:upper:]' '[:lower:]')
        short_ver=$(printf '%s' "$CHOSEN" | cut -d- -f1)
        LOCAL_TAG="${short_chip:-cann}:${short_ver:-latest}"

        echo ""
        echo -e "  ${CYAN}拉取镜像:${RESET} $FULL_IMAGE"
        echo -e "  ${CYAN}本地标签:${RESET} cann-${LOCAL_TAG}"
        docker pull "$FULL_IMAGE" || { echo -e "${RED}拉取失败(可能是镜像 tag 不存在): $FULL_IMAGE${RESET}" >&2; exit 1; }
        docker tag "$FULL_IMAGE" "cann-${LOCAL_TAG}"
        IMAGE_TO_USE="cann-${LOCAL_TAG}"
    else
        # fallback 2 已写入 TAGS_FILE; 继续进入同一个筛选/选择逻辑
        TOTAL=$(wc -l < "$TAGS_FILE" | tr -d ' ')
        if [ "$TOTAL" -eq 0 ]; then
            echo -e "${RED}备用 tag 列表为空。${RESET}" >&2
            rm -f "$TAGS_FILE"
            exit 1
        fi
        echo -e "${GREEN}使用内置 tag 列表: 共 ${TOTAL} 个。${RESET}"

        filter_tags() { if [ -n "$1" ]; then grep -E "$1" || true; else cat; fi; }

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

        short_chip=$(printf '%s' "$CHOSEN" | grep -oE '910[aAbB]|910|950|310[pP]|310' | head -1 | tr '[:upper:]' '[:lower:]')
        short_ver=$(printf '%s' "$CHOSEN" | cut -d- -f1)
        LOCAL_TAG="${short_chip:-cann}:${short_ver:-latest}"

        echo ""
        echo -e "  ${CYAN}拉取镜像:${RESET} $FULL_IMAGE"
        echo -e "  ${CYAN}本地标签:${RESET} cann-${LOCAL_TAG}"
        docker pull "$FULL_IMAGE" || { echo -e "${RED}拉取失败(可能是镜像 tag 不存在): $FULL_IMAGE${RESET}" >&2; exit 1; }
        docker tag "$FULL_IMAGE" "cann-${LOCAL_TAG}"
        IMAGE_TO_USE="cann-${LOCAL_TAG}"
    fi

    rm -f "$TAGS_FILE"
fi

if [ -z "$IMAGE_TO_USE" ]; then
    echo -e "${RED}未能确定要使用的镜像。${RESET}" >&2
    exit 1
fi

# ============================================================
# 第二部分: 收集容器参数并启动容器
# ============================================================
DEVICES=()
for d in /dev/davinci*; do
    [ -e "$d" ] && DEVICES+=("$d")
done
[ -n "${ITOOL_DEVICES:-}" ] && { DEVICES=(); for d in $ITOOL_DEVICES; do DEVICES+=("$d"); done; }

MGR_DEVS=()
[ -e /dev/davinci_manager ] && MGR_DEVS+=("/dev/davinci_manager")
[ -e /dev/devmm_svm ]      && MGR_DEVS+=("/dev/devmm_svm")
[ -e /dev/hisi_hdc ]       && MGR_DEVS+=("/dev/hisi_hdc")

MOUNTS=()
[ -e /usr/local/Ascend/driver ]      && MOUNTS+=("/usr/local/Ascend/driver:/usr/local/Ascend/driver")
[ -d /usr/local/dcmi ]               && MOUNTS+=("/usr/local/dcmi:/usr/local/dcmi")
[ -e /usr/local/bin/npu-smi ]        && MOUNTS+=("/usr/local/bin/npu-smi:/usr/local/bin/npu-smi")

echo ""
echo -e "  ${WHITE}════════════════════════════════════════════════════════════${RESET}"
echo -e "  ${WHITE}  容器实例化参数${RESET}"
echo -e "  ${WHITE}════════════════════════════════════════════════════════════${RESET}"

[ -z "$NAME" ]     && ask "容器名" "asc_dev" && NAME="$REPLY"
[ -z "$WORK_DIR" ] && ask "工作目录(映射到容器 /workspace)" "$(pwd)" && WORK_DIR="$REPLY"
[ -z "$SHM_SIZE" ] && ask "共享内存(--shm-size, 如 16g)" "16g" && SHM_SIZE="$REPLY"

if [ ${#DEVICES[@]} -gt 0 ]; then
    echo -e "  ${GREEN}[检测到]${RESET} NPU 设备: ${DEVICES[*]}"
else
    echo -e "  ${YELLOW}[警告]${RESET} 未发现 /dev/davinci* 设备, 将以无设备模式启动(仅编译可用)。"
fi

mkdir -p "$WORK_DIR"

START_SH="$WORK_DIR/start_container.sh"
cat > "$START_SH" <<SCRIPT_HEAD
#!/bin/bash
# ============================================================
# start_container.sh — 起容器脚本 (由 itool 生成, 可自行修改)
# 修改下方变量后执行:  bash $START_SH
# ============================================================
IMAGE="$IMAGE_TO_USE"
NAME="$NAME"
WORK_DIR="$WORK_DIR"
SHM_SIZE="$SHM_SIZE"
EXTRA_ARGS="$EXTRA_ARGS"

echo "== 停止旧容器(如存在) =="
docker stop  "\$NAME" 2>/dev/null || true
docker rm -f "\$NAME" 2>/dev/null || true

echo "== 启动容器: \$NAME (镜像 \$IMAGE) =="
docker run -itd \\
    --name "\$NAME" \\
    --network host \\
    --ipc host \\
    --privileged \\
    --shm-size "\$SHM_SIZE" \\
SCRIPT_HEAD

for d in "${DEVICES[@]}"; do
    printf '    --device "%s" \\\n' "$d" >> "$START_SH"
done
for d in "${MGR_DEVS[@]}"; do
    printf '    --device "%s" \\\n' "$d" >> "$START_SH"
done
for m in "${MOUNTS[@]}"; do
    printf '    -v "%s" \\\n' "$m" >> "$START_SH"
done

cat >> "$START_SH" <<'SCRIPT_TAIL'
    -v "$WORK_DIR":/workspace \
    -w /workspace \
    $EXTRA_ARGS \
    "$IMAGE" /bin/bash

echo "== 容器已启动 =="
docker ps --filter "name=$NAME" --format "  {{.Names}}  {{.Image}}  {{.Status}}"
echo ""
echo "进入容器:  docker exec -it $NAME bash"
SCRIPT_TAIL
chmod +x "$START_SH"

echo ""
echo -e "  ${GREEN}✔ 已生成起容器脚本:${RESET} $START_SH"
echo -e "  ${YELLOW}(可先用编辑器修改上方变量/设备/挂载, 再执行)${RESET}"
echo ""

echo -e "  ${CYAN}──── 执行起容器脚本 ────${RESET}"
bash "$START_SH"
rc=$?

echo ""
if [ $rc -eq 0 ]; then
    echo -e "  ${GREEN}✔ 容器实例化完成。${RESET}"
else
    echo -e "  ${RED}启动失败(退出码 $rc)。${RESET}" >&2
    exit $rc
fi

echo ""
echo -e "  ${CYAN}下一步:${RESET} 进入容器后执行环境检查 → docker exec -it $NAME bash"
echo -e "            容器内执行: bash d.ops_develop/b.env_check/run.sh"
