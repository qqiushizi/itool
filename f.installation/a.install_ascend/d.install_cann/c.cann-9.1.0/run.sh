#https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%209.1.0/Ascend-cann-toolkit_9.1.0_linux-x86_64.run?response-content-type=application/octet-stream
#https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%209.1.0/Ascend-cann-910b-ops_9.1.0_linux-x86_64.run?response-content-type=application/octet-stream
#https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%209.1.0/Ascend-cann-nnal_9.1.0_linux-x86_64.run?response-content-type=application/octet-stream

TEMP_DIR=/tmp
TOOLKIT_RUN=$TEMP_DIR/Ascend-cann-toolkit_9.0.0_linux-x86_64.run
wget "https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%209.1.0/Ascend-cann-toolkit_9.1.0_linux-x86_64.run?response-content-type=application/octet-stream" -O $TOOLKIT_RUN
bash $TOOLKIT_RUN --install --quiet

OPS_RUN=$TEMP_DIR/Ascend-cann-910b-ops_9.1.0_linux-x86_64.run
wget "https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%209.1.0/Ascend-cann-910b-ops_9.1.0_linux-x86_64.run?response-content-type=application/octet-stream" -O $OPS_RUN
bash $OPS_RUN --install --quiet

NNAL_RUN=$TEMP_DIR/Ascend-cann-nnal_9.1.0_linux-x86_64.run
wget "https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%209.1.0/Ascend-cann-nnal_9.1.0_linux-x86_64.run?response-content-type=application/octet-stream" -O $NNAL_RUN
bash $NNAL_RUN --install --quiet

# 配置环境变量(如下命令以root用户为例，请以实际安装路径为准)
source /usr/local/Ascend/cann/set_env.sh

python3 -c "import acl;print(acl.get_soc_name())"
