#!/bin/bash

# 定义文件路径
ENV_SCRIPT="/usr/local/Ascend/toolbox/set_env.sh"
# 安装包下载地址
PKG_URL="https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/MindCluster/MindCluster%2026.0.1.1/Ascend-mindx-toolbox_26.0.1.1_linux-x86_64.run?response-content-type=application/octet-stream"
PKG_NAME="/tmp/Ascend-mindx-toolbox_26.0.1.1_linux-x86_64.run"

# 判断环境脚本是否存在
if [ -f "${ENV_SCRIPT}" ]; then
    echo "检测到 ${ENV_SCRIPT}，直接加载环境变量"
    source "${ENV_SCRIPT}"
else
    echo "未检测到 ${ENV_SCRIPT}，开始下载安装包..."
    # 使用wget下载，若没有wget可替换curl
    if command -v wget &> /dev/null; then
        wget -O "${PKG_NAME}" "${PKG_URL}"
    elif command -v curl &> /dev/null; then
        curl -L -o "${PKG_NAME}" "${PKG_URL}"
    else
        echo "错误：系统未安装wget/curl，无法下载安装包"
        exit 1
    fi

    # 赋予执行权限
    chmod +x "${PKG_NAME}"
    echo "开始静默安装toolbox..."
    bash "${PKG_NAME}" --install -y

    # 校验安装后的环境文件
    if [ ! -f "${ENV_SCRIPT}" ]; then
        echo "安装完成后仍未找到 ${ENV_SCRIPT}，安装异常！"
        exit 1
    fi

    echo "加载新安装的环境变量"
    source "${ENV_SCRIPT}"
    echo "MindX Toolbox 安装并环境加载完成"
fi
