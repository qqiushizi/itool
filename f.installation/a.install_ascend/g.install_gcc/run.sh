#!/bin/bash
set -e

sudo yum install -y scl-utils gcc-toolset-11-runtime gcc-toolset-11-gcc gcc-toolset-11-gcc-c++

grep -q 'gcc-toolset-11/enable' ~/.bashrc || \
echo 'source /opt/rh/gcc-toolset-11/enable' >> ~/.bashrc

source /opt/rh/gcc-toolset-11/enable

echo "当前 GCC 路径：$(which gcc)"
gcc --version
g++ --version
