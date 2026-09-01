#pip install https://gitcode.com/Ascend/pytorch/releases/download/v26.1.0-beta.1-pytorch2.7.1/torch_npu-2.7.1.post5-cp310-cp310-manylinux_2_28_x86_64.whl

# 注意：triton-ascend 3.2.0 及以下 Triton-Ascend和Triton 不能同时存在。需要先卸载社区 Triton，再安装 Triton-Ascend。
pip install triton-ascend==3.2.1 --extra-index-url=https://triton-ascend.osinfra.cn/pypi/simple

ROOT_PATH=/workspace
MindSpeed_MM_PATH=$ROOT_PATH/MindSpeed-LLM

source /usr/local/Ascend/cann/set_env.sh 
source /usr/local/Ascend/nnal/atb/set_env.sh 

cd $ROOT_PATH
git clone https://gitcode.com/ascend/MindSpeed.git
cd MindSpeed
git checkout 26.0.0_core_r0.12.1  # 切换分支至MindSpeed 26.0.0_core_r0.12.1
pip3 install -r requirements.txt 
pip3 install -e .
cd ..

git clone https://gitcode.com/ascend/MindSpeed-LLM.git 
git clone https://github.com/NVIDIA/Megatron-LM.git  # 从github下载Megatron-LM，请确保网络能访问
cd Megatron-LM
git checkout core_v0.12.1
cp -r megatron ../MindSpeed-LLM/
cd ../MindSpeed-LLM
git checkout 26.0.0
mkdir logs

pip3 install -r requirements.txt  # 安装其余依赖库

pip install loguru  matplotlib msguard  openpyxl absl-py  cloudpickle ml-dtypes tornado numpy==1.26.0
pip install opentelemetry-exporter-otlp-proto-http==1.33.1

cp -r imodels idatasets iscripts $MindSpeed_MM_PATH/
