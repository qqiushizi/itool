#!/bin/bash
# ============================================================
# ① 镜像拉取 + 容器实例化 (简化版)
#
# 功能:
#   1) 用户只提供一个镜像地址
#   2) 本机存在该镜像则直接用; 不存在则自动 docker pull
#   3) 根据镜像 ID 和当前机器已有设备/挂载, 自动生成 start_container.sh
#   4) 宿主机工作目录挂载到容器 /workspace, 并在 /workspace 下工作
#
# 用法:
#   bash run.sh                                        # 交互提示输入镜像
#   bash run.sh quay.io/ascend/cann:9.0.0-910b-ubuntu22.04-py3.10
#   IMAGE=quay.io/ascend/cann:9.0.0-910b-ubuntu22.04-py3.10 bash run.sh
#   NAME=asc_dev WORK_DIR=/data/ops SHM_SIZE=16g bash run.sh
#
# 默认工作目录:
#   ~/ascend_ops_workspace (即 $HOME/ascend_ops_workspace)
#   可通过 WORK_DIR 或 ITOOL_WORK_DIR 覆盖
# ============================================================
set -uo pipefail

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; WHITE='\033[1;37m'; RESET='\033[0m'

have() { command -v "$1" >/dev/null 2>&1; }
ask()  { local p="$1" d="$2"; printf "  %s [%s]: " "$p" "$d"; IFS= read -r REPLY || REPLY=""; [ -z "$REPLY" ] && REPLY="$d"; }

command -v docker >/dev/null 2>&1 || { echo -e "${RED}未找到 docker, 请先安装。${RESET}" >&2; exit 1; }

IMAGE_TO_USE="${1:-${IMAGE:-}}"
NAME="${2:-${NAME:-}}"
WORK_DIR="${WORK_DIR:-${ITOOL_WORK_DIR:-$HOME/ascend_ops_workspace}}"
SHM_SIZE="${SHM_SIZE:-16g}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

echo ""
echo -e "  ${WHITE}════════════════════════════════════════════════════════════${RESET}"
echo -e "  ${WHITE}  ① 镜像拉取 + 容器实例化（简化版）${RESET}"
echo -e "  ${WHITE}════════════════════════════════════════════════════════════${RESET}"

# ---------- 1. 获取镜像地址 ----------
if [ -z "$IMAGE_TO_USE" ]; then
    while [ -z "$IMAGE_TO_USE" ]; do
        ask "请输入镜像地址(例如 quay.io/ascend/cann:9.0.0-910b-ubuntu22.04-py3.10)" ""
        IMAGE_TO_USE="$REPLY"
    done
fi

echo ""
echo -e "  ${CYAN}镜像地址:${RESET} $IMAGE_TO_USE"

# 容器名：位置参数/环境变量已指定则直接用；否则交互让用户自定义
if [ -z "$NAME" ]; then
    ask "请输入容器名称" "asc_dev"
    NAME="$REPLY"
    [ -z "$NAME" ] && NAME="asc_dev"
fi
echo -e "  ${GREEN}[容器名]${RESET} $NAME"

# ---------- 2. 本地检查, 不存在则拉取 ----------
if docker image inspect "$IMAGE_TO_USE" >/dev/null 2>&1; then
    echo -e "  ${GREEN}[存在]${RESET} 本机已有该镜像, 直接使用。"
else
    echo -e "  ${YELLOW}[不存在]${RESET} 开始拉取镜像: $IMAGE_TO_USE"
    docker pull "$IMAGE_TO_USE" || { echo -e "${RED}镜像拉取失败: $IMAGE_TO_USE${RESET}" >&2; exit 1; }
fi

IMAGE_ID=$(docker image inspect -f '{{.Id}}' "$IMAGE_TO_USE" 2>/dev/null || true)
if [ -z "$IMAGE_ID" ]; then
    echo -e "${RED}无法获取镜像 ID, 请检查镜像是否正常。${RESET}" >&2
    exit 1
fi
echo -e "  ${GREEN}[镜像 ID]${RESET} $IMAGE_ID"

# ---------- 3. 确定宿主机工作目录并创建 ----------
mkdir -p "$WORK_DIR" || { echo -e "${RED}无法创建工作目录: $WORK_DIR${RESET}" >&2; exit 1; }
echo -e "  ${GREEN}[工作目录]${RESET} $WORK_DIR  → 容器 /workspace"

# ---------- 4. 检测当前机器设备和挂载 ----------
DEVICES=()
for d in /dev/davinci*; do
    [ -e "$d" ] && DEVICES+=("$d")
done
[ -n "${ITOOL_DEVICES:-}" ] && { DEVICES=(); for d in $ITOOL_DEVICES; do DEVICES+=("$d"); done; }

MGR_DEVS=()
for d in /dev/davinci_manager /dev/devmm_svm /dev/hisi_hdc; do
    [ -e "$d" ] && MGR_DEVS+=("$d")
done

MOUNTS=()
[ -e /usr/local/Ascend/driver ] && MOUNTS+=("/usr/local/Ascend/driver:/usr/local/Ascend/driver")
[ -d /usr/local/dcmi ] && MOUNTS+=("/usr/local/dcmi:/usr/local/dcmi")
[ -e /usr/local/bin/npu-smi ] && MOUNTS+=("/usr/local/bin/npu-smi:/usr/local/bin/npu-smi")

echo ""
echo -e "  ${CYAN}────────────────────────────────────────────${RESET}"
echo -e "  容器名     : $NAME"
echo -e "  镜像 ID    : $IMAGE_ID"
echo -e "  工作目录   : $WORK_DIR → /workspace"
echo -e "  共享内存   : $SHM_SIZE"
if [ ${#DEVICES[@]} -gt 0 ]; then
    echo -e "  NPU 设备   : ${DEVICES[*]}"
else
    echo -e "  ${YELLOW}NPU 设备   : 未发现 /dev/davinci* (仅编译可用)${RESET}"
fi
echo -e "  ${CYAN}────────────────────────────────────────────${RESET}"

# ---------- 5. 生成当前机器专用起容器脚本 ----------
START_SH="$WORK_DIR/start_container.sh"
cat > "$START_SH" <<SCRIPT_HEAD
#!/bin/bash
# ============================================================
# start_container.sh — 当前机器专用起容器脚本
#
# 该脚本由 itool 根据生成时的机器设备和镜像 ID 生成。
# 适用于“当前这台机器”反复重启容器；不建议跨机器直接复制。
# ============================================================
IMAGE="$IMAGE_TO_USE"
IMAGE_ID="$IMAGE_ID"
NAME="$NAME"
WORK_DIR="$WORK_DIR"
SHM_SIZE="$SHM_SIZE"
EXTRA_ARGS="$EXTRA_ARGS"

RUN_IMAGE="\$IMAGE_ID"
[ -n "\$RUN_IMAGE" ] || RUN_IMAGE="\$IMAGE"

echo "== 停止旧容器(如存在) =="
docker stop  "\$NAME" 2>/dev/null || true
docker rm -f "\$NAME" 2>/dev/null || true

echo "== 启动容器: \$NAME (镜像 \$RUN_IMAGE) =="
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
    "$RUN_IMAGE" /bin/bash

echo ""
echo "== 容器已启动 =="
docker ps --filter "name=$NAME" --format "  {{.Names}}  {{.Image}}  {{.Status}}"
echo ""
echo "进入容器:  docker exec -it $NAME bash"
SCRIPT_TAIL

chmod +x "$START_SH"

echo ""
echo -e "  ${GREEN}✔ 已生成起容器脚本:${RESET} $START_SH"
echo -e "  ${YELLOW}(可直接修改后执行:  bash $START_SH)${RESET}"

# ---------- 6. 立即启动容器 ----------
echo ""
echo -e "  ${CYAN}──── 执行起容器脚本 ────${RESET}"
bash "$START_SH"
rc=$?

if [ $rc -ne 0 ]; then
    echo -e "${RED}启动失败(退出码 $rc)。${RESET}" >&2
    exit $rc
fi

echo ""
echo -e "  ${CYAN}下一步:${RESET} 进入容器 → docker exec -it $NAME bash"
echo -e "            容器内执行: bash d.ops_develop/b.env_check/run.sh"
