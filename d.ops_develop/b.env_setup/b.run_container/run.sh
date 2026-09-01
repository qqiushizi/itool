#!/bin/bash
# ============================================================
# 实例化 Ascend CANN 容器 (vllm-ascend 镜像)
# 用法:
#   bash run.sh [镜像] [容器名] [工作目录]
#   默认: 镜像 cann-910b:8.1.RC1, 容器名 asc_dev, 工作目录 $(pwd)
# 说明:
#   - 自动枚举 /dev/davinci* 设备并挂载。
#   - 挂载驱动相关目录(npu-smi / dcmi / driver), 便于容器内访问昇腾设备。
# ============================================================
set -euo pipefail

command -v docker >/dev/null 2>&1 || { echo -e "\033[0;31m未找到 docker。\033[0m" >&2; exit 1; }

IMAGE="${1:-cann-910b:8.1.RC1}"
NAME="${2:-asc_dev}"
WORK_DIR="${3:-$(pwd)}"

# 收集 NPU 设备
DEVICES=()
for d in /dev/davinci*; do
    [ -e "$d" ] && DEVICES+=("--device" "$d")
done
if [ ${#DEVICES[@]} -eq 0 ]; then
    echo -e "\033[1;33m警告: 未发现 /dev/davinci* 设备, 将以无设备模式启动(仅编译可用)。\033[0m"
fi

echo "=========================================================="
echo " 实例化容器"
echo "   镜像: $IMAGE   容器名: $NAME   工作目录: $WORK_DIR"
echo "=========================================================="

# 驱动管理设备(存在才挂载)
DRIVER_MOUNTS=()
[ -e /dev/davinci_manager ] && DRIVER_MOUNTS+=("--device" "/dev/davinci_manager")
[ -e /dev/devmm_svm ]      && DRIVER_MOUNTS+=("--device" "/dev/devmm_svm")
[ -e /dev/hisi_hdc ]       && DRIVER_MOUNTS+=("--device" "/dev/hisi_hdc")
[ -e /usr/local/dcmi ]     && DRIVER_MOUNTS+=("-v" "/usr/local/dcmi:/usr/local/dcmi")
[ -e /usr/local/bin/npu-smi ] && DRIVER_MOUNTS+=("-v" "/usr/local/bin/npu-smi:/usr/local/bin/npu-smi")
[ -d /usr/local/Ascend/driver ] && DRIVER_MOUNTS+=("-v" "/usr/local/Ascend/driver:/usr/local/Ascend/driver")

docker run -itd --name "$NAME" \
    --network host --ipc=host --privileged \
    "${DEVICES[@]}" \
    "${DRIVER_MOUNTS[@]}" \
    -v "$WORK_DIR:/workspace" \
    -w /workspace \
    "$IMAGE" /bin/bash

echo ""
echo -e "\033[0;32m容器已启动:\033[0m"
docker ps --filter "name=$NAME" --format "  {{.Names}}  {{.Image}}  {{.Status}}"
echo ""
echo "进入容器:  docker exec -it $NAME bash"
