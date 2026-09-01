source ../setenv.sh

cd $MindSpeed_MM_PATH
mkdir $MindSpeed_MM_PATH/idatasets
cd $MindSpeed_MM_PATH/idatasets

#使用真实数据集进行训练时，通常因为样本间序列长度不一，每一步迭代的时间会有所波动，且真实数据通常较大，有一定的下载和使用成本，因此在指定数据分辨率、序列长度的功能和性能测试场景，使用虚构数据可以更快的满足测试效果。
#当前仓库提供了一种构造指定配置图文数据的方法，虚构数据生成脚本使用指令如下：

source /usr/local/Ascend/ascend-toolkit/set_env.sh
SAVE_DIR=$MindSpeed_MM_PATH/idatasets/mocked_vl_data/
mkdir -p $SAVE_DIR

# 下方命令会生成包括512条样本的数据集，每条样本拥有10张1024*1024大小的图片以及16384的文本长度，--tokenizer_path需要指定当前待测模型的原始权重本地路径
cd $MindSpeed_MM_PATH
echo 开始生成虚拟数据集到：$SAVE_DIR
python mindspeed_mm/fsdp/tools/data_tool/generate_mock_data_for_vlmodel.py \
    --tokenizer_path $Model_Path_Root/$Model_Name \
    --pic_width 1024 \
    --pic_height 1024 \
    --num_pics 10 \
    --text_length 16384 \
    --num_samples 512 \
    --save_dir $SAVE_DIR

