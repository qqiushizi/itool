# 拉取镜像
source $ITOOL_FUNCTIONS

echo 镜像下载地址：https://www.hiascend.com/developer/ascendhub/detail/6857f6fc2cfa4a678710a7075426ee5e

echo MindSpeed-MM安装指南：https://gitcode.com/Ascend/MindSpeed-MM/blob/master/docs/zh/pytorch/installation.md#%E5%AE%89%E8%A3%85%E6%8C%87%E5%8D%97

echo 镜像：$ITOOL_MINDSPEED_MM_IMAGEID
read -p "请输入您要创建的容器名:" CONTAINER_NAME

# 初始化变量
DOCKER_MAP_PATH=""

echo "========== 映射路径输入工具 =========="
echo "输入完成后直接按回车即可结束输入"
echo "--------------------------------------"

# 循环输入
while true; do
    # 提示用户输入
    read -p "请输入映射路径: " input

    # 如果直接回车，结束循环
    if [[ -z "$input" ]]; then
        break
    fi

    # 判断是否包含 :，不包含则自动包裹 :
    if [[ "$input" != *":"* ]]; then
        result="${input}:${input}"
    else
        result="$input"
    fi

    # 拼接到总变量中
    DOCKER_MAP_PATH+=" -v ${result}"
done

# 输出最终结果
echo -e "\n========== 最终生成的映射路径 =========="
echo "$DOCKER_MAP_PATH"


idocker $ITOOL_MINDSPEED_MM_IMAGEID $CONTAINER_NAME "$DOCKER_MAP_PATH"


