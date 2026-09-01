source ../setenv.sh

echo ====== 下载模型 ======
mkdir $MindSpeed_MM_PATH/imodels

echo "====================================="
echo "当前模型是：$Model_Name"
read -p "如需修改请输入新名称，回车使用默认：" input_name

if [[ -n "$input_name" ]]; then
    Model_Name="$input_name"
    echo ">> 模型名称已修改为：$Model_Name"
else
    echo ">> 使用默认模型名称：$Model_Name"
fi
echo "====================================="

echo 检查模型配置文件：$MindSpeed_MM_PATH/imodels/model_hf/$Model_Name/config.json

if [ -f "$MindSpeed_MM_PATH/imodels/model_hf/$Model_Name/config.json" ]; then
    echo "模型文件存在, 跳过模型下载。"
else
    echo "文件不存在, 开始下载..."
    modelscope download --model $Model_Name  --local_dir $MindSpeed_MM_PATH/imodels/model_hf/$Model_Name
fi

echo ====== 开始转换模型 ====
mkdir $MindSpeed_MM_PATH/imodels/model_dcp

cd $MindSpeed_MM_PATH
mm-convert Qwen35Converter hf_to_dcp \
--hf_dir ./imodels/model_hf/$Model_Name/ \
--dcp_dir ./imodels/model_dcp/$Model_Name/ \
--tie_weight_mapping '{"lm_head.weight":"model.language_model.embed_tokens.weight"}'


# 生成启动脚本
#cp $MindSpeed_MM_PATH/examples/qwen3_5/finetune_qwen3_5_4B.sh $MindSpeed_MM_PATH/iscripts/finetune_qwen3_5_2B.sh
#cp $MindSpeed_MM_PATH/examples/qwen3_5/qwen3_5_4B_config.yaml $MindSpeed_MM_PATH/iscripts/qwen3_5_4B_config.yaml
