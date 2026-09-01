#!/bin/bash
set -euo pipefail

# 1. 获取用户输入PyTorch版本
read -p "请输入要安装的PyTorch版本(例如: 2.3.0、2.4.1): " TORCH_VERSION
if [[ -z "$TORCH_VERSION" ]]; then
    echo "错误：版本号不能为空！"
    exit 1
fi

# 2. 识别系统CPU架构
ARCH=$(uname -m)
echo "====================================="
echo "检测到系统架构: $ARCH"
echo "目标PyTorch版本: $TORCH_VERSION"
echo "====================================="

# 3. 判断安装渠道与包
if [[ "$ARCH" == "aarch64" ]]; then
    # ARM架构：强制CPU版本
    echo "当前为aarch64架构，仅安装CPU版本PyTorch"
    PIP_CMD="pip3 install torch==${TORCH_VERSION} torchvision torchaudio==${TORCH_VERSION} --index-url https://download.pytorch.org/whl/cpu"
elif [[ "$ARCH" == "x86_64" ]]; then
    # x86架构：检测NVIDIA显卡
    if command -v nvidia-smi &> /dev/null; then
        echo "检测到nvidia-smi，存在NVIDIA显卡，安装CUDA版本PyTorch"
        PIP_CMD="pip3 install torch==${TORCH_VERSION} torchvision torchaudio==${TORCH_VERSION} --index-url https://download.pytorch.org/whl/cu121"
    else
        echo "未检测到nvidia-smi，无NVIDIA显卡，安装CPU版本PyTorch"
        PIP_CMD="pip3 install torch==${TORCH_VERSION} torchvision torchaudio==${TORCH_VERSION} --index-url https://download.pytorch.org/whl/cpu"
    fi
else
    echo "不支持的系统架构: $ARCH"
    exit 1
fi

# 4. 打印pip命令并执行
echo -e "\n====================生成的安装命令===================="
echo "$PIP_CMD"
echo "======================================================"
echo -e "\n开始执行安装..."
eval "$PIP_CMD"

# 5. 验证安装结果
echo -e "\n安装完成。"

#pip install https://gitcode.com/Ascend/pytorch/releases/download/v26.1.0-beta.1-pytorch2.7.1/torch_npu-2.7.1.post5-cp310-cp310-manylinux_2_28_x86_64.whl
