wget https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%209.0.0/Ascend-cann-toolkit_9.0.0_linux-x86_64.run?response-content-type=application/octet-stream -O Ascend-cann-toolkit_9.0.0_linux-x86_64.run

wget https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%209.0.0/Ascend-cann-950-ops_9.0.0_linux-x86_64.run?response-content-type=application/octet-stream -O Ascend-cann-950-ops_9.0.0_linux-x86_64.run

echo "Start to install ascend-toolkit..."
bash Ascend-cann-toolkit_9.0.0_linux-x86_64.run --install

echo "Start to install cann ops..."
bash Ascend-cann-950-ops_9.0.0_linux-x86_64.run --install
