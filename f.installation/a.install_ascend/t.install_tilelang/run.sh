echo "开始下载tilelang-0.1.1.10+linux.cann900-cp311-cp311-linux_x86_64.whl"
wget "https://github.com/tile-ai/tilelang-ascend/releases/download/v0.1.1.010-release/tilelang-0.1.1.10+linux.cann900-cp311-cp311-linux_x86_64.whl" -O /tmp/tilelang-0.1.1.10+linux.cann900-cp311-cp311-linux_x86_64.whl

echo "开始安装/tmp/tilelang-0.1.1.10+linux.cann900-cp311-cp311-linux_x86_64.whl"
pip install /tmp/tilelang-0.1.1.10+linux.cann900-cp311-cp311-linux_x86_64.whl
