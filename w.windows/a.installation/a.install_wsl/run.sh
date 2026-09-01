echo \# 启用适用于Linux的Windows子系统
echo dism.exe /online /enable-feature /featurename:Microsoft-Windows-Subsystem-Linux /all /norestart
echo \ 

echo \# 启用虚拟机平台（WSL 2必需）
echo dism.exe /online /enable-feature /featurename:VirtualMachinePlatform /all /norestart
echo \

echo \# 下载内核更新包,然后运行安装
echo https://wslstorestorage.blob.core.windows.net/wslblob/wsl_update_x64.msi
echo \

echo \# 重启电脑， 后设置wsl2为默认版本
echo wsl --set-default-version 2
echo \

echo \# 安装Ubuntu 24.04
echo 方案A
echo https://cloud-images.ubuntu.com/releases/24.04/release/ubuntu-24.04-server-cloudimg-amd64-wsl.rootfs.tar.gz；
echo mkdir C:\WSL\Ubuntu2404
echo wsl --import Ubuntu-24.04 C:\WSL\Ubuntu2404 C:\Users\admin\Downloads\ubuntu-24.04-server-cloudimg-amd64-wsl.rootfs.tar.gz --version 2
echo wsl -d Ubuntu-24.04
echo \

echo 方案B
echo 打开商店
echo 安装WSL UI
echo 安装Ubuntu-24.04
