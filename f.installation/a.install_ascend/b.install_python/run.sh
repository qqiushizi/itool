#!/bin/bash
set -e
# 自动识别CPU架构
ARCH=$(uname -m)
case $ARCH in
x86_64) PKG=Miniforge3-Linux-x86_64.sh ;;
aarch64) PKG=Miniforge3-Linux-aarch64.sh ;;
*) echo "错误：不支持当前架构 $ARCH"; exit 1 ;;
esac

# 下载安装包
#wget -c https://github.com/conda-forge/miniforge/releases/latest/download/$PKG
wget -c https://mirrors.tuna.tsinghua.edu.cn/github-release/conda-forge/miniforge/LatestRelease/$PKG
# 静默安装至家目录miniforge3
bash $PKG -b -p ~/miniforge3 -u
rm -f $PKG

# 初始化shell
~/miniforge3/bin/conda init bash
source ~/.bashrc

# 配置中科大镜像源
conda config --remove-key channels
conda config --add channels https://mirrors.ustc.edu.cn/anaconda/pkgs/main/
conda config --add channels https://mirrors.ustc.edu.cn/anaconda/pkgs/free/
conda config --add channels https://mirrors.ustc.edu.cn/anaconda/cloud/conda-forge/
conda config --set show_channel_urls yes

echo "Miniforge安装完成，中科大源已配置完毕！"
echo "验证命令：conda --version / mamba --version"
