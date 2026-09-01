source ../../setenv.sh

mkdir $MindSpeed_LLM_PATH/idatasets
SOURCE_DATA=$MindSpeed_LLM_PATH/idatasets/train-00000-of-00001-a09b74b3ef9c3b56.parquet

if [ -f "$MindSpeed_LLM_PATH/idatasets/train-00000-of-00001-a09b74b3ef9c3b56.parquet" ]; then
    echo "数据文件存在，跳过下载。"
else
    echo "数据文件不存在, 开始下载..."

    # HuggingFace 数据集链接（择一获取）
    # wget https://huggingface.co/datasets/tatsu-lab/alpaca/resolve/main/data/train-00000-of-00001-a09b74b3ef9c3b56.parquet -O $SOURCE_DATA

    # ModelScope 数据集链接（择一获取）
    wget https://www.modelscope.cn/datasets/angelala00/tatsu-lab-alpaca/resolve/master/train-00000-of-00001-a09b74b3ef9c3b56.parquet  -O $SOURCE_DATA
fi

# 修改 ascend-toolkit 路径
source /usr/local/Ascend/ascend-toolkit/set_env.sh

cd $MindSpeed_LLM_PATH
python ./preprocess_data.py \
    --input $SOURCE_DATA \
    --tokenizer-name-or-path $MindSpeed_LLM_PATH/imodels/model_hf/$Model_Name/ \
    --tokenizer-type PretrainedFromHF \
    --handler-name GeneralPretrainHandler \
    --output-prefix $MindSpeed_LLM_PATH/idatasets/enwiki \
    --json-keys text \
    --workers 4 \
    --log-interval 1000
