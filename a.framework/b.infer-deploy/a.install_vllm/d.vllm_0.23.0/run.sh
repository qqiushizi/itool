# Using apt-get with mirror
sed -i 's|ports.ubuntu.com|mirrors.tuna.tsinghua.edu.cn|g' /etc/apt/sources.list
apt-get update -y && apt-get install -y gcc g++ cmake libnuma-dev wget git curl jq
# Or using yum
# yum update -y && yum install -y gcc g++ cmake numactl-devel wget git curl jq
# Config pip mirror,only versions 0.11.0 and earlier are supported, if using a version later than 0.11.0, do not execute this command
pip config set global.index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple

# For torch-npu dev version or x86 machine
pip config set global.extra-index-url "https://download.pytorch.org/whl/cpu/"

# Install vllm-project/vllm. The newest supported version is v0.18.0.
pip install vllm==0.23.0

# Install vllm-project/vllm-ascend from pypi.
pip install vllm-ascend==0.23.0rc1
