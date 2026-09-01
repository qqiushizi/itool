#Ascend-cann为驱动、Toolkit合一包

TEMP_DIR=/tmp
wget "https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%209.0.0/Ascend-cann-toolkit_9.0.0_linux-x86_64.run?response-content-type=application/octet-stream" -O $TEMP_DIR/Ascend-cann-toolkit_9.0.0_linux-x86_64.run
bash $TEMP_DIR/Ascend-cann-toolkit_9.0.0_linux-x86_64.run --install --quiet

wget https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%209.0.0/Ascend-cann-910b-ops_9.0.0_linux-x86_64.run -O $TEMP_DIR/Ascend-cann-910b-ops_9.0.0_linux-x86_64.run
bash $TEMP_DIR/Ascend-cann-910b-ops_9.0.0_linux-x86_64.run --install --quiet

# 配置环境变量(如下命令以root用户为例，请以实际安装路径为准)
source /usr/local/Ascend/cann/set_env.sh

python3 -c "import acl;print(acl.get_soc_name())"
