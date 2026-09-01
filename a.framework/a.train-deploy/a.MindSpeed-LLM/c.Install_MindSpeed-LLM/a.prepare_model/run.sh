source ../setenv.sh

echo ====== 下载模型 ======
mkdir $MindSpeed_LLM_PATH/imodels
mkdir $MindSpeed_LLM_PATH/imodels/model_hf
mkdir $MindSpeed_LLM_PATH/imodels/model_mcore

echo "当前模型是：$Model_Name"

read -p "如需修改请输入新模型名称，直接回车保持默认：" input_name

# 判断输入是否非空，非空则覆盖
if [[ -n "$input_name" ]]; then
    Model_Name="$input_name"
    echo "已更新模型名称为：$Model_Name"
else
    echo "保持默认模型名称：$Model_Name"
fi

echo 检查模型配置文件：$MindSpeed_LLM_PATH/imodels/model_hf/$Model_Name/config.json

if [ -f "$MindSpeed_LLM_PATH/imodels/model_hf/$Model_Name/config.json" ]; then
    echo "模型文件存在, 跳过模型下载。"
else
    echo "文件不存在, 开始下载..."
    modelscope download --model $Model_Name  --local_dir $MindSpeed_LLM_PATH/imodels/model_hf/$Model_Name
fi

echo ====== 开始转换模型 ====
# 修改 ascend-toolkit 路径
export CUDA_DEVICE_MAX_CONNECTIONS=1
source /usr/local/Ascend/ascend-toolkit/set_env.sh

cd $MindSpeed_LLM_PATH
python convert_ckpt_v2.py \
    --load-model-type hf \
    --save-model-type mg \
    --target-tensor-parallel-size 1 \
    --target-pipeline-parallel-size 1 \
    --load-dir $MindSpeed_LLM_PATH/imodels/model_hf/$Model_Name/ \
    --save-dir $MindSpeed_LLM_PATH/imodels/model_mcore/$Model_Name \
    --model-type-hf qwen3
