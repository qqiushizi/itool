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

QUAY_REPO="${QUAY_REPO:-}"
QUAY_TAGS_URL=""
TAG_LIMIT=100
QUAY_CONNECT_TIMEOUT=5
QUAY_MAX_TIME=8
QUERY_PAGES=5
# quay.io 直连不通时按顺序尝试国内镜像；留空则禁止镜像 fallback
QUAY_MIRROR="${QUAY_MIRROR:-m.daocloud.io/quay.io quay.nju.edu.cn}"

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
pull_image_smart() {
    local image="$1" mirror mimg
    if docker pull "$image"; then
        return 0
    fi
    [ -n "$QUAY_MIRROR" ] || return 1
    for mirror in $QUAY_MIRROR; do
        mimg="${mirror}/${image#quay.io/}"
        [ "$mimg" = "$image" ] && continue
        echo -e "  ${YELLOW}[镜像加速]${RESET} 尝试: $mimg"
        if docker pull "$mimg"; then
            docker tag "$mimg" "$image" || { echo -e "${RED}镜像 tag 失败: $mimg -> $image${RESET}" >&2; return 1; }
            echo -e "  ${GREEN}[镜像加速]${RESET} 已打回官方 tag: $image"
            return 0
        fi
    done
    return 1
}

# 把 quay.io/ascend/xxx 转为 API 需要的 ascend/xxx
quay_repo_path() {
    local p="$1"
    p="${p#https://}"; p="${p#http://}"; p="${p#quay.io/}"; p="${p%/}"
    printf '%s' "$p"
}
set_quay_url() {
    local path
    path=$(quay_repo_path "$QUAY_REPO")
    QUAY_TAGS_URL="https://quay.io/api/v1/repository/$path/tag"
}
choose_official_repo() {
    while true; do
        echo ""
        echo -e "  ${CYAN}请选择要查询/拉取的官方 quay.io 仓库:${RESET}"
        echo -e "    ${GREEN}A${RESET}) quay.io/ascend/vllm-ascend   (vLLM Ascend 推理容器)"
        echo -e "    ${GREEN}B${RESET}) quay.io/ascend/cann          (CANN 算子开发容器)"
        ask "请选择 (A/B)" "A"
        case "$REPLY" in
            a|A|1) QUAY_REPO="quay.io/ascend/vllm-ascend"; break ;;
            b|B|2) QUAY_REPO="quay.io/ascend/cann"; break ;;
            *) continue ;;
        esac
    done
    set_quay_url
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
    for ((page=1; page<=QUERY_PAGES; page++)); do
        body=$(curl -fsSL --connect-timeout "$QUAY_CONNECT_TIMEOUT" --max-time "$QUAY_MAX_TIME" "$QUAY_TAGS_URL/?limit=$TAG_LIMIT&page=$page&onlyActiveTags=true" 2>/dev/null) || true
        if [ -z "$body" ]; then
            [ "$page" = "1" ] && warn "官方 tag 查询超时或不可达（连接时限 ${QUAY_CONNECT_TIMEOUT}s / 总时限 ${QUAY_MAX_TIME}s）。"
            [ "$page" = "1" ] && warn "为避免长时间卡住，已跳过远程查询；稍后请直接手动输入 tag。"
            break
        fi

        while IFS= read -r tag; do
            [ -n "$tag" ] || continue
            TAGS+=("$tag")
        done < <(printf '%s' "$body" | grep -oE '"name":[[:space:]]*"[^"]+"' 2>/dev/null | sed -E 's/.*"name":[[:space:]]*"([^"]+)".*/\1/')

        if printf '%s' "$body" | grep -q '"has_additional":[[:space:]]*false'; then
            break
        fi
    done
}

# ---------- 选择官方 tag（手动输入为主，查询只是辅助） ----------
choose_official_tag() {
    local keyword="${TAG_FILTER:-${CANN_TAG_FILTER:-}}"
    local filtered=() i n tag choice example="9.1.0-910b-ubuntu22.04-py3.10"
    local display_count

    case "$QUAY_REPO" in
        *vllm*) example="v0.27.1" ;;
    esac

    # 查询不到时直接手动输入
    if [ ${#TAGS[@]} -eq 0 ]; then
        warn "未查询到 $QUAY_REPO 可用 tag（可能当前机器无法访问 quay.io）。"
        while true; do
            ask "请输入要拉取的 tag 或完整镜像，例如 $example" ""
            tag="$REPLY"
            [ -n "$tag" ] && break
        done
        if printf '%s' "$tag" | grep -q '/'; then
            SELECTED_IMAGE="$tag"
        else
            SELECTED_IMAGE="$QUAY_REPO:$tag"
        fi
        return 0
    fi

    # 查询成功了也只是参考；用户可以任意手动输入 tag / 完整镜像
    filtered=()
    if [ -n "$keyword" ]; then
        for tag in "${TAGS[@]}"; do
            if printf '%s' "$tag" | grep -qiF "$keyword"; then
                filtered+=("$tag")
            fi
        done
    fi
    if [ ${#filtered[@]} -eq 0 ]; then
        filtered=("${TAGS[@]}")
    fi

    display_count=${#filtered[@]}
    [ "$display_count" -gt 20 ] && display_count=20

    echo ""
    echo -e "  ${CYAN}官方查询结果仅作参考，共 ${#filtered[@]} 个匹配 tag:${RESET}"
    for ((i=0; i<display_count; i++)); do
        printf '    %3d) %s
' "$((i+1))" "${filtered[$i]}"
    done
    echo ""

    while true; do
        printf '  请输入要拉取的 tag 或完整镜像，或输入 1-%s 选择参考项: ' "$display_count"
        IFS= read -r choice || choice=""
        if [ -z "$choice" ]; then
            echo -e "  ${RED}输入不能为空。${RESET}"
            continue
        fi
        if [ "$choice" -ge 1 ] 2>/dev/null && [ "$choice" -le "$display_count" ] 2>/dev/null; then
            SELECTED_IMAGE="$QUAY_REPO:${filtered[$((choice-1))]}"
            return 0
        fi
        # 非编号输入，一律当作手动 tag / 完整镜像
        if printf '%s' "$choice" | grep -q '/'; then
            SELECTED_IMAGE="$choice"
        else
            SELECTED_IMAGE="$QUAY_REPO:$choice"
        fi
        return 0
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

# 明确给了完整镜像地址时，不再查询官方仓库。
if [ -n "$IMAGE_ARG" ] && printf '%s' "$IMAGE_ARG" | grep -q '/'; then
    IMAGE_TO_USE="$IMAGE_ARG"
    echo -e "  ${CYAN}[镜像]${RESET} $IMAGE_TO_USE"
else
    # 未指定官方仓库时，交互选择 vllm-ascend / cann，或使用 QUAY_REPO 环境变量。
    [ -z "$QUAY_REPO" ] && choose_official_repo
    [ -z "$QUAY_REPO" ] && { echo -e "  ${RED}未确定 quay.io 官方仓库。${RESET}" >&2; exit 1; }
    set_quay_url

    if [ -n "$IMAGE_ARG" ]; then
        IMAGE_TO_USE="$QUAY_REPO:$IMAGE_ARG"
        echo -e "  ${CYAN}[镜像]${RESET} $IMAGE_TO_USE"
    else
        echo -e "  ${CYAN}正在查询 $QUAY_REPO 官方可用 tag ...${RESET}"
        fetch_official_tags
        [ ${#TAGS[@]} -gt 0 ] && echo -e "  ${GREEN}[查询成功]${RESET} 共发现 ${#TAGS[@]} 个 tag"
        choose_official_tag
        IMAGE_TO_USE="$SELECTED_IMAGE"
        echo -e "  ${GREEN}[已选择镜像]${RESET} $IMAGE_TO_USE"
    fi
fi

# ==== 2. 容器配置 ====
# 下面所有配置均支持环境变量预设；未预设时通过 A/B 选项引导，尽量不让客户手输参数。
chosen=""
pick_abcd() {
    local p="$1" d="$2"
    while true; do
        ask "$p" "$d"
        case "$REPLY" in
            a|A|1) chosen="A"; return 0 ;;
            b|B|2) chosen="B"; return 0 ;;
            c|C|3) chosen="C"; return 0 ;;
            d|D|4) chosen="D"; return 0 ;;
        esac
    done
}
pick_ab() {
    local p="$1" d="$2"
    while true; do
        ask "$p" "$d"
        case "$REPLY" in
            a|A|1) chosen="A"; return 0 ;;
            b|B|2) chosen="B"; return 0 ;;
        esac
    done
}

NAME_USE="${2:-${NAME:-}}"
if [ -z "$NAME_USE" ]; then
    echo ""
    echo -e "  ${CYAN}容器名称${RESET}"
    echo -e "    ${GREEN}A${RESET}) asc_dev（默认）"
    echo -e "    ${GREEN}B${RESET}) 自定义名称"
    pick_ab "请选择" "A"
    if [ "$chosen" = "A" ]; then
        NAME_USE="asc_dev"
    else
        ask "请输入容器名称" "asc_dev"
        NAME_USE="$REPLY"
        [ -z "$NAME_USE" ] && NAME_USE="asc_dev"
    fi
fi

DEFAULT_WORK_DIR="${HOME:-/root}/ascend_ops_workspace"
CWD_WORK_DIR="${PWD:-$(pwd)}/ascend_ops_workspace"
WORK_DIR_USE="${WORK_DIR:-${ITOOL_WORK_DIR:-}}"
if [ -z "$WORK_DIR_USE" ]; then
    echo ""
    echo -e "  ${CYAN}宿主机工作目录（将挂载到容器 /workspace）${RESET}"
    echo -e "    ${GREEN}A${RESET}) ${DEFAULT_WORK_DIR}（默认）"
    echo -e "    ${GREEN}B${RESET}) ${CWD_WORK_DIR}"
    echo -e "    ${GREEN}C${RESET}) 自定义路径"
    pick_abcd "请选择" "A"
    case "$chosen" in
        A) WORK_DIR_USE="$DEFAULT_WORK_DIR" ;;
        B) WORK_DIR_USE="$CWD_WORK_DIR" ;;
        C)
            while true; do
                ask "请输入宿主机工作目录" "$DEFAULT_WORK_DIR"
                WORK_DIR_USE="$REPLY"
                [ -n "$WORK_DIR_USE" ] && break
            done
            ;;
    esac
fi
[ -z "$WORK_DIR_USE" ] && WORK_DIR_USE="$DEFAULT_WORK_DIR"

SHM_SIZE_USE="${SHM_SIZE:-}"
if [ -z "$SHM_SIZE_USE" ]; then
    echo ""
    echo -e "  ${CYAN}容器共享内存 --shm-size${RESET}"
    echo -e "    ${GREEN}A${RESET}) 16g（默认，一般算子开发够用）"
    echo -e "    ${GREEN}B${RESET}) 32g"
    echo -e "    ${GREEN}C${RESET}) 64g"
    echo -e "    ${GREEN}D${RESET}) 自定义"
    pick_abcd "请选择" "A"
    case "$chosen" in
        A) SHM_SIZE_USE="16g" ;;
        B) SHM_SIZE_USE="32g" ;;
        C) SHM_SIZE_USE="64g" ;;
        D)
            while true; do
                ask "请输入共享内存大小" "16g"
                SHM_SIZE_USE="$REPLY"
                [ -n "$SHM_SIZE_USE" ] && break
            done
            ;;
    esac
fi

NET_MODE_USE="${NET_MODE:-}"
if [ -z "$NET_MODE_USE" ]; then
    echo ""
    echo -e "  ${CYAN}容器网络模式${RESET}"
    echo -e "    ${GREEN}A${RESET}) host（默认，推荐 NPU / vLLM 场景）"
    echo -e "    ${GREEN}B${RESET}) bridge"
    pick_ab "请选择" "A"
    [ "$chosen" = "A" ] && NET_MODE_USE="host" || NET_MODE_USE="bridge"
fi

PRIVILEGED_USE="${PRIVILEGED:-}"
if [ -z "$PRIVILEGED_USE" ]; then
    echo ""
    echo -e "  ${CYAN}是否使用 --privileged${RESET}"
    echo -e "    ${GREEN}A${RESET}) yes（默认，推荐）"
    echo -e "    ${GREEN}B${RESET}) no"
    pick_ab "请选择" "A"
    [ "$chosen" = "A" ] && PRIVILEGED_USE="yes" || PRIVILEGED_USE="no"
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
    pull_image_smart "$IMAGE_TO_USE" || { echo -e "${RED}镜像拉取失败: $IMAGE_TO_USE${RESET}" >&2; exit 1; }
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
