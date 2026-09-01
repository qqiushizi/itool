#镜像：https://quay.io/repository/ascend/vllm-ascend?tab=tags&tag=latest
#权重：https://modelscope.cn/models/Eco-Tech/DeepSeek-V4-Flash-w8a8-mtp
#指导：https://docs.vllm.ai/projects/ascend/en/v0.13.0/tutorials/DeepSeek-V4.html

#A2
#wget http://xql-model.obs.cn-east-3.myhuaweicloud.com/images/vllm-ascend/v0.13.0rc3.tar.gz
#wget http://xql-model.obs.cn-east-3.myhuaweicloud.com/images/vllm-ascend/v0.13.0rc3-openeuler.tar.gz

#A3
#wget http://xql-model.obs.cn-east-3.myhuaweicloud.com/images/vllm-ascend/v0.13.0rc3-a3.tar.gz
#wget http://xql-model.obs.cn-east-3.myhuaweicloud.com/images/vllm-ascend/v0.13.0rc3-a3-openeuler.tar.gz

model_root_path=$LXY_ROOT_PATH/model
modelscope download --model Eco-Tech/DeepSeek-V4-Flash-w8a8-mtp --local_dir $model_root_path/DeepSeek-V4-Flash-w8a8-mtp
