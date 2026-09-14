#!/bin/bash
# ============================================================
# ① 镜像拉取 + 容器实例化 (quay.io/ascend/cann 向导版)
#
# 功能:
#   1) 查询 quay.io/ascend/cann 官方可用 tag，客户可视化选择
#   2) 本机已有该镜像则直接使用；不存在则自动 docker pull
#   3) 配置容器名 / 工作目录 / 网络模式 / 是否 privileged / 共享内存
#   4) 自动识别当前机器 NPU 设备和 Ascend 挂载
#   5) 自动把当前 itool 仓库挂载到容器 /workspace/itool
#   6) 生成当前机器专用 start_container.sh，预览后询问是否立即启动
#
# 用法:
#   bash run.sh
#   IMAGE=quay.io/ascend/cann:9.1.0-910b-ubuntu22.04-py3.10 bash run.sh
#   bash run.sh quay.io/ascend/cann:9.1.0-910b-ubuntu22.04-py3.10 asc_dev
#
# 常用环境变量:
#   CANN_TAG_FILTER  官方 tag 筛选关键字，例如 9.1.0 / 910b / py3.10
#   IMAGE            显式指定镜像；如果只有 tag，会自动补 quay.io/ascend/cann 前缀
#   NAME             容器名
#   WORK_DIR         宿主机工作目录，默认 ~/ascend_ops_workspace
#   SHM_SIZE         共享内存，默认 16g
#   NET_MODE         host 或 bridge，默认 host
#   PRIVILEGED       yes 或 no，默认 yes
#   EXTRA_ARGS       额外 docker run 参数
# ============================================================
set -uo pipefail

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; WHITE='\033[1;37m'; RESET='\033[0m'

QUAY_REPO="quay.io/ascend/cann"
QUAY_TAGS_URL="https://quay.io/api/v1/repository/ascend/cann/tag"
TAG_LIMIT=100
MAX_PAGES=5

have() { command -v "$1" >/dev/null 2>&1; }
warn()  { echo -e "  ${YELLOW}[ 警告 ]${RESET} $1"; }
ask() {
    local p="$1" d="$2"
    printf "  %s [%s]: " "$p" "$d"
    IFS= read -r REPLY || REPLY=""
    [ -z "$REPLY" ] && REPLY="$d"
}
ask_yes() {
    local p="$1" d="$2"
    printf "  %s [%s]: " "$p" "$d"
    IFS= read -r REPLY || REPLY=""
    [ -z "$REPLY" ] && REPLY="$d"
    case "$REPLY" in
        y|Y|yes|YES) return 0 ;;
        n|N|no|NO)   return 1 ;;
        *) return 1 ;;
    esac
}
yesno() {
    [ "$1" = "yes" ] || [ "$1" = "y" ] || [ "$1" = "Y" ] || [ "$1" = "YES" ] && return 0 || return 1
}

# ---------- 当前 itool 仓库根目录 ----------
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd || echo "$PWD")
REPO_ROOT="${ITOOL_REPO_ROOT:-}"

if [ -z "$REPO_ROOT" ] && [ -f "$PWD/itool.sh" ]; then
    REPO_ROOT="$PWD"
fi
if [ -z "$REPO_ROOT" ] && have git; then
    REPO_ROOT=$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel 2>/dev/null || true)
fi
if [ -z "$REPO_ROOT" ]; then
    d="$SCRIPT_DIR"
    while [ "$d" != "/" ]; do
        if [ -f "$d/itool.sh" ]; then REPO_ROOT="$d"; break; fi
        d=$(dirname "$d")
    done
fi
[ -z "$REPO_ROOT" ] && REPO_ROOT="$SCRIPT_DIR"

HAVE_FULL_REPO=1
[ -f "$REPO_ROOT/d.ops_develop/b.env_check/run.sh" ] || HAVE_FULL_REPO=0

# ---------- tag 查询 ----------
TAGS=()
fetch_official_tags() {
    local page body
    TAGS=()
    for page in 1 2 3 4 5; do
        body=$(curl -fsS --max-time 30 "$QUAY_TAGS_URL/?limit=$TAG_LIMIT&page=$page&onlyActiveTags=true" 2>/dev/null) || break
        [ -n "$body" ] || break

        while IFS= read -r tag; do
            [ -n "$tag" ] || continue
            TAGS+=("$tag")
        done < <(printf '%s' "$body" | grep -oE '"name":[[:space:]]*"[^"]+"' 2>/dev/null | sed -E 's/.*"name":[[:space:]]*"([^"]+)".*/\1/')

        if printf '%s' "$body" | grep -q '"has_additional":[[:space:]]*false'; then
            break
        fi
    done
}

# ---------- 选择官方 tag ----------
choose_official_tag() {
    local keyword="${CANN_TAG_FILTER:-}"
    local filtered=() i n tag
    local show_count

    if [ ${#TAGS[@]} -eq 0 ]; then
        warn "未查询到 quay.io/ascend/cann 可用 tag（可能当前机器无法访问 quay.io）。"
        REPLY=""
        while [ -z "$REPLY" ]; do
            ask "请手动输入官方 tag，例如 9.1.0-910b-ubuntu22.04-py3.10" ""
            tag="$REPLY"
        done
        SELECTED_IMAGE="quay.io/ascend/cann:$tag"
        return 0
    fi

    while true; do
        filtered=()
        if [ -n "$keyword" ]; then
            for tag in "${TAGS[@]}"; do
                if printf '%s' "$tag" | grep -qiF "$keyword"; then
                    filtered+=("$tag")
                fi
            done
        else
            filtered=("${TAGS[@]}")
        fi

        if [ ${#filtered[@]} -eq 0 ]; then
            echo ""
            echo -e "  ${YELLOW}没有匹配「$keyword」的 tag。${RESET}"
            keyword=""
            continue
        fi

        if [ ${#filtered[@]} -gt 20 ]; then
            echo ""
            echo -e "  ${YELLOW}匹配到 ${#filtered[@]} 个 tag，先展示前 20 个。${RESET}"
            echo -e "  ${DIM}建议用更精确关键字筛选，例如: 9.1.0 / 910b / py3.10 / devel${RESET}"
            for ((i=0; i<20; i++)); do
                printf '    %3d) %s\n' "$((i+1))" "${filtered[$i]}"
            done
            echo ""
            ask "请输入更精确筛选关键字" "$keyword"
            keyword="$REPLY"
            continue
        fi

        echo ""
        echo -e "  ${CYAN}匹配到以下官方 tag:${RESET}"
        for ((i=0; i<${#filtered[@]}; i++)); do
            printf '    %3d) %s\n' "$((i+1))" "${filtered[$i]}"
        done
        echo ""
        ask "请选择 tag 编号" "1"
        n="$REPLY"
        if [ "$n" -ge 1 ] 2>/dev/null && [ "$n" -le "${#filtered[@]}" ]; then
            SELECTED_IMAGE="quay.io/ascend/cann:${filtered[$((n-1))]}"
            return 0
        fi
        echo -e "  ${RED}输入编号无效。${RESET}"
    done
}

# ---------- 颜色辅助 ----------
DIM='\033[2m'

echo ""
echo -e "  ${WHITE}════════════════════════════════════════════════════════════${RESET}"
echo -e "  ${WHITE}  ① 镜像拉取 + 容器实例化（quay.io/ascend/cann 官方向导）${RESET}"
echo -e "  ${WHITE}════════════════════════════════════════════════════════════${RESET}"

have docker || { echo -e "${RED}未找到 docker，请先安装。${RESET}" >&2; exit 1; }

# ==== 1. 确定镜像 ====
IMAGE_ARG="${1:-${IMAGE:-}}"
if [ -n "$IMAGE_ARG" ]; then
    if printf '%s' "$IMAGE_ARG" | grep -q '/'; then
        IMAGE_TO_USE="$IMAGE_ARG"
    else
        IMAGE_TO_USE="$QUAY_REPO:$IMAGE_ARG"
    fi
    echo -e "  ${CYAN}[镜像]${RESET} $IMAGE_TO_USE"
else
    echo -e "  ${CYAN}正在查询 $QUAY_REPO 官方可用 tag ...${RESET}"
    fetch_official_tags
    [ ${#TAGS[@]} -gt 0 ] && echo -e "  ${GREEN}[查询成功]${RESET} 共发现 ${#TAGS[@]} 个 tag"
    choose_official_tag
    IMAGE_TO_USE="$SELECTED_IMAGE"
    echo -e "  ${GREEN}[已选择镜像]${RESET} $IMAGE_TO_USE"
fi

# ==== 2. 容器配置 ====
NAME_USE="${2:-${NAME:-}}"
if [ -z "$NAME_USE" ]; then
    ask "请输入容器名称" "asc_dev"
    NAME_USE="$REPLY"
    [ -z "$NAME_USE" ] && NAME_USE="asc_dev"
fi

WORK_DIR_USE="${WORK_DIR:-${ITOOL_WORK_DIR:-$HOME/ascend_ops_workspace}}"
if [ -z "$WORK_DIR" ] && [ -z "$ITOOL_WORK_DIR" ]; then
    ask "宿主机工作目录(将挂载到 /workspace)" "$WORK_DIR_USE"
    WORK_DIR_USE="$REPLY"
fi
[ -z "$WORK_DIR_USE" ] && WORK_DIR_USE="$HOME/ascend_ops_workspace"

SHM_SIZE_USE="${SHM_SIZE:-16g}"
if [ -z "$SHM_SIZE" ]; then
    ask "共享内存大小" "$SHM_SIZE_USE"
    SHM_SIZE_USE="$REPLY"
fi

NET_MODE_USE="${NET_MODE:-host}"
if [ -z "$NET_MODE" ]; then
    while true; do
        ask "网络模式(host/bridge)" "$NET_MODE_USE"
        NET_MODE_USE="$REPLY"
        [ "$NET_MODE_USE" = "host" ] || [ "$NET_MODE_USE" = "bridge" ] && break
    done
fi

PRIV_ENV_SET=0
[ -n "${PRIVILEGED:-}" ] && PRIV_ENV_SET=1
PRIVILEGED_USE="${PRIVILEGED:-yes}"
if [ "$PRIV_ENV_SET" = "0" ]; then
    if ask_yes "是否使用 --privileged (y/n)" "$PRIVILEGED_USE"; then
        PRIVILEGED_USE="yes"
    else
        PRIVILEGED_USE="no"
    fi
fi

EXTRA_ARGS_USE="${EXTRA_ARGS:-}"

echo ""
echo -e "  ${CYAN}── 配置汇总 ──${RESET}"
printf '  %-20s: %s\n' "镜像" "$IMAGE_TO_USE"
printf '  %-20s: %s\n' "容器名" "$NAME_USE"
printf '  %-20s: %s\n' "宿主机工作目录" "$WORK_DIR_USE"
printf '  %-20s: %s\n' "容器工作目录" "/workspace"
printf '  %-20s: %s\n' "共享内存" "$SHM_SIZE_USE"
printf '  %-20s: %s\n' "网络模式" "$NET_MODE_USE"
printf '  %-20s: %s\n' "privileged" "$PRIVILEGED_USE"
if [ "$HAVE_FULL_REPO" = "1" ]; then
    printf '  %-20s: %s\n' "自动挂载 itool" "$REPO_ROOT → /workspace/itool"
else
    printf '  %-20s: %s\n' "自动挂载 itool" "未检测到完整仓库，跳过"
fi

# ==== 3. 本地检查/拉取镜像 ====
if docker image inspect "$IMAGE_TO_USE" >/dev/null 2>&1; then
    echo -e "  ${GREEN}[存在]${RESET} 本机已有该镜像，直接使用。"
else
    echo -e "  ${YELLOW}[不存在]${RESET} 开始拉取镜像: $IMAGE_TO_USE"
    docker pull "$IMAGE_TO_USE" || { echo -e "${RED}镜像拉取失败: $IMAGE_TO_USE${RESET}" >&2; exit 1; }
fi

IMAGE_ID=$(docker image inspect -f '{{.Id}}' "$IMAGE_TO_USE" 2>/dev/null || true)
[ -z "$IMAGE_ID" ] && { echo -e "${RED}无法获取镜像 ID。${RESET}" >&2; exit 1; }
echo -e "  ${GREEN}[镜像 ID]${RESET} $IMAGE_ID"

mkdir -p "$WORK_DIR_USE" || { echo -e "${RED}无法创建工作目录: $WORK_DIR_USE${RESET}" >&2; exit 1; }

# ==== 4. 生成 start_container.sh ====
START_SH="$WORK_DIR_USE/start_container.sh"
cat > "$START_SH" <<SCRIPT_HEAD
#!/bin/bash
# ============================================================
# start_container.sh — 由 itool 生成的当前机器专用起容器脚本
#
# 自动化挂载:
#   - NPU / Ascend 设备与驱动挂载：启动时按当前机器实际存在自动识别
#   - 工作目录: $WORK_DIR_USE -> /workspace
# ============================================================
IMAGE="$IMAGE_TO_USE"
IMAGE_ID="$IMAGE_ID"
NAME="$NAME_USE"
WORK_DIR="$WORK_DIR_USE"
SHM_SIZE="$SHM_SIZE_USE"
NET_MODE="$NET_MODE_USE"
PRIVILEGED="$PRIVILEGED_USE"
EXTRA_ARGS="$EXTRA_ARGS_USE"

RUN_IMAGE="\$IMAGE_ID"
[ -n "\$RUN_IMAGE" ] || RUN_IMAGE="\$IMAGE"

echo "== 停止旧容器(如存在) =="
docker stop  "\$NAME" 2>/dev/null || true
docker rm -f "\$NAME" 2>/dev/null || true

DOCKER_ARGS=(-itd --name "\$NAME" --ipc host --shm-size "\$SHM_SIZE" -w /workspace)
if [ "\$PRIVILEGED" = "yes" ]; then
    DOCKER_ARGS+=(--privileged)
fi
if [ "\$NET_MODE" = "host" ]; then
    DOCKER_ARGS+=(--network host)
fi

for d in /dev/davinci*; do
    [ -e "\$d" ] && DOCKER_ARGS+=(--device "\$d")
done
for d in /dev/davinci_manager /dev/devmm_svm /dev/hisi_hdc; do
    [ -e "\$d" ] && DOCKER_ARGS+=(--device "\$d")
done

[ -e /usr/local/Ascend/driver ] && DOCKER_ARGS+=(-v "/usr/local/Ascend/driver:/usr/local/Ascend/driver")
[ -d /usr/local/dcmi ]         && DOCKER_ARGS+=(-v "/usr/local/dcmi:/usr/local/dcmi")
[ -e /usr/local/bin/npu-smi ]  && DOCKER_ARGS+=(-v "/usr/local/bin/npu-smi:/usr/local/bin/npu-smi")

SCRIPT_HEAD

if [ "$HAVE_FULL_REPO" = "1" ]; then
    cat >> "$START_SH" <<SCRIPT_REPO_MOUNT
DOCKER_ARGS+=(-v "$REPO_ROOT:/workspace/itool")
SCRIPT_REPO_MOUNT
fi

cat >> "$START_SH" <<'SCRIPT_TAIL'

echo "== 启动容器: $NAME (镜像 $RUN_IMAGE) =="
docker run "${DOCKER_ARGS[@]}" \
    -v "$WORK_DIR":/workspace \
    $EXTRA_ARGS \
    "$RUN_IMAGE" /bin/bash

echo ""
echo "== 容器已启动 =="
docker ps --filter "name=$NAME" --format "  {{.Names}}  {{.Image}}  {{.Status}}"
echo ""
echo "进入容器:                 docker exec -it $NAME bash"
SCRIPT_TAIL

if [ "$HAVE_FULL_REPO" = "1" ]; then
    cat >> "$START_SH" <<'SCRIPT_TAIL_REPO'
echo "容器内检查环境:           cd /workspace/itool && bash d.ops_develop/b.env_check/run.sh"
SCRIPT_TAIL_REPO
fi

chmod +x "$START_SH"
echo ""
echo -e "  ${GREEN}✔ 已生成起容器脚本:${RESET} $START_SH"
echo -e "  ${YELLOW}(你可以先修改，再手动执行:  bash $START_SH)${RESET}"

# ==== 5. 询问是否立即启动 ====
echo ""
if ask_yes "是否立即启动容器" "N"; then
    bash "$START_SH"
    rc=$?
    if [ $rc -ne 0 ]; then
        echo -e "${RED}启动失败(退出码 $rc)。${RESET}" >&2
        exit $rc
    fi
else
    echo -e "  ${DIM}已跳过启动。后续手动执行: bash $START_SH${RESET}"
fi

echo ""
echo -e "  ${CYAN}下一步:${RESET}"
echo -e "    1) 进入容器:      docker exec -it $NAME_USE bash"
if [ "$HAVE_FULL_REPO" = "1" ]; then
    echo -e "    2) 环境检查:      cd /workspace/itool && bash d.ops_develop/b.env_check/run.sh"
else
    echo -e "    2) 请先把 itool 仓库拷贝/挂载进容器 /workspace/itool 后再执行环境检查"
fi
