#!/bin/bash
set -euo pipefail

# 1. 交互选择镜像源
if [ $# -ne 0 ]; then
    echo "❌ 无需指定命令行参数，请直接执行：sudo $0"
    exit 1
fi

echo "请选择要使用的 APT 镜像源："
echo "  1) 清华大学镜像源（HTTPS）"
echo "  2) 阿里云镜像源（HTTP）"

while true; do
    if ! read -r -p "请输入选项 [1-2]: " SOURCE_OPTION; then
        echo
        echo "❌ 未读取到选择，已取消操作"
        exit 1
    fi

    case "${SOURCE_OPTION}" in
        1)
            SOURCE_TYPE="tsinghua"
            break
            ;;
        2)
            SOURCE_TYPE="aliyun"
            break
            ;;
        *)
            echo "❌ 无效选项，请输入 1 或 2"
            ;;
    esac
done

# 2. 校验root权限
if [ "$(id -u)" -ne 0 ]; then
    echo "❌ 错误：必须使用 sudo 或 root 执行脚本"
    exit 1
fi

# 3. 读取 /etc/os-release 获取 VERSION_ID
OS_RELEASE="/etc/os-release"
if [ ! -f "${OS_RELEASE}" ]; then
    echo "❌ 无法读取系统版本文件 ${OS_RELEASE}"
    exit 1
fi

# 提取 VERSION_ID，形如 20.04 / 22.04 / 24.04
VERSION_ID=$(grep -E '^VERSION_ID=' "${OS_RELEASE}" | cut -d'=' -f2 | tr -d '"')
echo "✅ 识别到Ubuntu版本号: ${VERSION_ID}"

# 映射版本号 → 系统代号 CODENAME
case "${VERSION_ID}" in
    "20.04")
        CODENAME="focal"
        ;;
    "22.04")
        CODENAME="jammy"
        ;;
    "24.04")
        CODENAME="noble"
        ;;
    *)
        echo "❌ 不支持的Ubuntu版本 ${VERSION_ID}"
        echo "仅支持 20.04 / 22.04 / 24.04"
        exit 1
        ;;
esac
echo "✅ 对应系统代号: ${CODENAME}"

# 4. 备份原sources.list（带时间戳，不覆盖旧备份）
SOURCE_FILE="/etc/apt/sources.list"
BAK_FILE="${SOURCE_FILE}.bak.$(date +%Y%m%d_%H%M%S)"
cp "${SOURCE_FILE}" "${BAK_FILE}"
echo "✅ 原源文件已备份至: ${BAK_FILE}"

# 5. 写入对应镜像源
if [ "${SOURCE_TYPE}" = "tsinghua" ]; then
    echo "🔄 正在写入清华大学镜像源(HTTPS)..."
    cat > "${SOURCE_FILE}" <<EOF
deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ ${CODENAME} main restricted universe multiverse
deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ ${CODENAME}-updates main restricted universe multiverse
deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ ${CODENAME}-backports main restricted universe multiverse
deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ ${CODENAME}-security main restricted universe multiverse
EOF
elif [ "${SOURCE_TYPE}" = "aliyun" ]; then
    echo "🔄 正在写入阿里云镜像源(HTTP)..."
    cat > "${SOURCE_FILE}" <<EOF
deb http://mirrors.aliyun.com/ubuntu/ ${CODENAME} main restricted universe multiverse
deb http://mirrors.aliyun.com/ubuntu/ ${CODENAME}-updates main restricted universe multiverse
deb http://mirrors.aliyun.com/ubuntu/ ${CODENAME}-backports main restricted universe multiverse
deb http://mirrors.aliyun.com/ubuntu/ ${CODENAME}-security main restricted universe multiverse
EOF
fi

# 6. 更新apt缓存
echo "🔄 刷新软件源索引..."
apt update

echo -e "\n🎉 APT源切换完成！"
echo "镜像源: ${SOURCE_TYPE}"
echo "系统版本: Ubuntu ${VERSION_ID} (${CODENAME})"
echo "备份文件: ${BAK_FILE}"
echo "如需升级全部软件包，执行：sudo apt upgrade -y"
