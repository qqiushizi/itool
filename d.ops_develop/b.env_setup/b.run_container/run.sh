#!/bin/bash
# ============================================================
# ④ 容器实例化 (交互式 + 生成可编辑起容器脚本)
# 功能:
#   - 交互收集 镜像/容器名/工作目录/挂载设备/共享内存 等参数
#   - 生成 start_container.sh (客户可自行修改后重复使用)
#   - 立即执行 start_container.sh 建立容器
# 用法:
#   bash run.sh [镜像] [容器名] [工作目录]
# 环境变量: IMAGE / NAME / WORK_DIR / SHM_SIZE / EXTRA_ARGS / ITOOL_DEVICES
# ============================================================
set -o pipefail

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; WHITE='\033[1;37m'; RESET='\033[0m'

ask() { # $1=提示 $2=默认值 ; 结果放 $REPLY
    local prompt="$1" def="$2"
    printf "  %s [%s]: " "$prompt" "$def"
    IFS= read -r REPLY || REPLY=""
    [ -z "$REPLY" ] && REPLY="$def"
}

command -v docker >/dev/null 2>&1 || { echo -e "${RED}未找到 docker。${RESET}" >&2; exit 1; }

IMAGE="${1:-${IMAGE:-}}"
NAME="${2:-${NAME:-}}"
WORK_DIR="${3:-${WORK_DIR:-}}"
SHM_SIZE="${SHM_SIZE:-}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

# 收集 NPU 设备
DEVICES=()
for d in /dev/davinci*; do
    [ -e "$d" ] && DEVICES+=("$d")
done
[ -n "${ITOOL_DEVICES:-}" ] && { DEVICES=(); for d in $ITOOL_DEVICES; do DEVICES+=("$d"); done; }

# 驱动管理设备(存在才挂载)
MGR_DEVS=()
[ -e /dev/davinci_manager ] && MGR_DEVS+=("/dev/davinci_manager")
[ -e /dev/devmm_svm ]      && MGR_DEVS+=("/dev/devmm_svm")
[ -e /dev/hisi_hdc ]       && MGR_DEVS+=("/dev/hisi_hdc")

# 驱动/工具挂载(存在才挂载)
MOUNTS=()
[ -e /usr/local/Ascend/driver ]      && MOUNTS+=("/usr/local/Ascend/driver:/usr/local/Ascend/driver")
[ -d /usr/local/dcmi ]               && MOUNTS+=("/usr/local/dcmi:/usr/local/dcmi")
[ -e /usr/local/bin/npu-smi ]        && MOUNTS+=("/usr/local/bin/npu-smi:/usr/local/bin/npu-smi")

echo ""
echo -e "  ${WHITE}════════════════════════════════════════════════════${RESET}"
echo -e "  ${WHITE}  ④ 容器实例化${RESET}"
echo -e "  ${WHITE}════════════════════════════════════════════════════${RESET}"

# ---------- 交互收集参数 ----------
[ -z "$IMAGE" ]    && ask "镜像" "cann-910b:8.1.rc1" && IMAGE="$REPLY"
[ -z "$NAME" ]     && ask "容器名" "asc_dev" && NAME="$REPLY"
[ -z "$WORK_DIR" ] && ask "工作目录(映射到容器 /workspace)" "$(pwd)" && WORK_DIR="$REPLY"
[ -z "$SHM_SIZE" ] && ask "共享内存(--shm-size, 如 16g)" "16g" && SHM_SIZE="$REPLY"

if [ ${#DEVICES[@]} -gt 0 ]; then
    echo -e "  ${GREEN}[检测到]${RESET} NPU 设备: ${DEVICES[*]}"
else
    echo -e "  ${YELLOW}[警告]${RESET} 未发现 /dev/davinci* 设备, 将以无设备模式启动(仅编译可用)。"
fi

mkdir -p "$WORK_DIR"

# ---------- 生成可编辑的起容器脚本 ----------
START_SH="$WORK_DIR/start_container.sh"
cat > "$START_SH" <<SCRIPT_HEAD
#!/bin/bash
# ============================================================
# start_container.sh — 起容器脚本 (由 itool 生成, 可自行修改)
# 修改下方变量后执行:  bash $START_SH
# ============================================================
IMAGE="$IMAGE"
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

# 写入 NPU 设备
for d in "${DEVICES[@]}"; do
    printf '    --device "%s" \\\n' "$d" >> "$START_SH"
done
# 写入驱动管理设备
for d in "${MGR_DEVS[@]}"; do
    printf '    --device "%s" \\\n' "$d" >> "$START_SH"
done
# 写入挂载
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

# ---------- 执行 ----------
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
echo -e "  ${CYAN}下一步:${RESET} 进入容器检查软件包 → bash d.ops_develop/b.env_setup/c.check_in_container/run.sh $NAME"
