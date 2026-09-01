
if [ ! -f "/tmp/Ascend-cann-toolkit_8.5.0_linux-x86_64.run" ]; then
	wget "https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%208.5.0/Ascend-cann-toolkit_8.5.0_linux-x86_64.run?response-content-type=application/octet-stream" -O /tmp/Ascend-cann-toolkit_8.5.0_linux-x86_64.run
fi;

if [ ! -f "/tmp/Ascend-cann-910b-ops_8.5.0_linux-x86_64.run" ]; then
	wget "https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%208.5.0/Ascend-cann-910b-ops_8.5.0_linux-x86_64.run?response-content-type=application/octet-stream" -O /tmp/Ascend-cann-910b-ops_8.5.0_linux-x86_64.run
fi;

if [ ! -f "/tmp/Ascend-cann-nnal_8.5.0_linux-x86_64.run" ]; then
	wget "https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%208.5.0/Ascend-cann-nnal_8.5.0_linux-x86_64.run?response-content-type=application/octet-stream" -O /tmp/Ascend-cann-nnal_8.5.0_linux-x86_64.run
fi;

bash /tmp/Ascend-cann-toolkit_8.5.0_linux-x86_64.run --install -q

bash /tmp/Ascend-cann-910b-ops_8.5.0_linux-x86_64.run --install -q 

bash /tmp/Ascend-cann-nnal_8.5.0_linux-x86_64.run --install -q
