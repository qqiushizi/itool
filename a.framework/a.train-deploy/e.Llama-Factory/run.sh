#!/bin/bash
###############################################################################
# Qwen3-VL-2B-Instruct SFT 一键自动化部署脚本
#
# 功能：
#   1. 检查 Ascend NPU 环境
#   2. 安装 Python 依赖 (modelscope, llamafactory, peft 等)
#   3. 下载基础模型 (Qwen3-VL-2B-Instruct)
#   4. 创建训练数据 (ShareGPT 格式)
#   5. 生成 LLaMAFactory 训练配置
#   6. 创建启动脚本和文档
#   7. (可选) 自动启动 SFT 训练
#
# 使用方法：
#   bash run.sh           # 仅部署，不训练
#   bash run.sh --train   # 部署后自动启动训练
#
# 系统要求：
#   - 华为 Ascend 910B NPU + CANN 9.0.0+
#   - 系统已安装 python3 (位于 PATH 中)
#   - Root 权限
###############################################################################

set -e

# ==================== 颜色输出 ====================
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info()  { echo -e "${BLUE}[INFO]${NC} $*"; }
log_ok()    { echo -e "${GREEN}[ OK ]${NC} $*"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC} $*"; }
log_err()   { echo -e "${RED}[FAIL]${NC} $*"; }

# ==================== 路径配置 ====================
PYTHON_BIN="python3"
LLAFACTORY_CLI="$(dirname "$(command -v python3)")/llamafactory-cli"
WORKSPACE="/workspace"
PROJECT_DIR="${WORKSPACE}/Llama-Factory"
MODEL_NAME="Qwen3-VL-2B-Instruct"
MODEL_REPO_ID="Qwen/${MODEL_NAME}"
MODEL_DIR="${PROJECT_DIR}/models/Qwen/${MODEL_NAME}"
DATA_DIR="${PROJECT_DIR}/data"
CONFIG_DIR="${PROJECT_DIR}/configs"
SAVES_DIR="${PROJECT_DIR}/saves/qwen3vl-sft"
LOG_DIR="${PROJECT_DIR}/logs"

# 修复 CANN 9.0.0 customize op 库（libcust_opmaster_rt2.0.so）的 libstdc++ 兼容性问题
# 现象：error: /lib/x86_64-linux-gnu/libstdc++.so.6: version `GLIBCXX_3.4.32' not found
# 原因：CANN 9.0.0 customize op 库用 GCC 13+ 编译，而系统（Ubuntu 22.04）默认 gcc 11.4 仅提供 GLIBCXX_3.4.30
# 解决：预加载 conda 自带的 GCC 15 libstdc++（提供 GLIBCXX_3.4.32+）
CUSTOM_LIBSTDCXX="/root/miniforge3/lib/libstdc++.so.6.0.34"
if [ -f "${CUSTOM_LIBSTDCXX}" ]; then
    log_ok "libstdc++ 兼容层就绪：${CUSTOM_LIBSTDCXX}"
else
    log_warn "未找到 ${CUSTOM_LIBSTDCXX}，libstdc++ 预加载将跳过"
    CUSTOM_LIBSTDCXX=""
fi

AUTO_TRAIN=false
for arg in "$@"; do
    if [ "$arg" = "--train" ] || [ "$arg" = "-t" ]; then
        AUTO_TRAIN=true
    fi
done

# ==================== 步骤 1: 检查 NPU 环境 ====================
echo ""
echo "=========================================================================="
echo "  步骤 1/7: 检查 Ascend NPU 环境"
echo "=========================================================================="

if ! command -v npu-smi &> /dev/null; then
    log_err "未检测到 npu-smi 工具，请确认已安装 CANN Toolkit"
    exit 1
fi

NPU_COUNT=$(npu-smi info 2>/dev/null | grep -c "910B" || echo "0")
if [ "$NPU_COUNT" -lt 1 ]; then
    log_err "未检测到 Ascend 910B NPU"
    exit 1
fi
log_ok "检测到 ${NPU_COUNT} 张 Ascend 910B NPU"

CANN_VERSION=$(ls /usr/local/Ascend/ 2>/dev/null | grep "^cann-" | head -1)
if [ -z "$CANN_VERSION" ]; then
    log_warn "未找到 CANN 安装目录，使用默认路径"
else
    log_ok "CANN 版本：${CANN_VERSION}"
fi

# ==================== 步骤 2: 检查 Python 环境 ====================
echo ""
echo "=========================================================================="
echo "  步骤 2/7: 检查 Python 和 torch_npu"
echo "=========================================================================="

PY_VERSION=$($PYTHON_BIN --version 2>&1 | awk '{print $2}')
log_ok "Python 版本：${PY_VERSION}"

TORCH_NPU_VER=$($PYTHON_BIN -c "import torch_npu; print(torch_npu.__version__)" 2>&1 || echo "未安装 torch_npu")
log_ok "torch_npu 版本：${TORCH_NPU_VER}"

# ==================== 步骤 3: 安装依赖 ====================
echo ""
echo "=========================================================================="
echo "  步骤 3/7: 安装 Python 依赖包"
echo "=========================================================================="

install_pkg() {
    local pkg=$1
    if $PYTHON_BIN -c "import ${pkg}" 2>/dev/null; then
        log_ok "${pkg} 已安装，跳过"
    else
        log_info "安装 ${pkg}..."
        $PYTHON_BIN -m pip install "${pkg}" \
            -i https://mirrors.huaweicloud.com/repository/pypi/simple \
            --quiet 2>&1 | tail -3
        log_ok "${pkg} 安装完成"
    fi
}

install_pkg "modelscope"
install_pkg "peft"
install_pkg "llamafactory" || true

# 验证 LLaMAFactory
if [ ! -x "$LLAFACTORY_CLI" ]; then
    log_warn "llamafactory-cli 未找到，尝试重新安装"
    $PYTHON_BIN -m pip install llamafactory \
        -i https://mirrors.huaweicloud.com/repository/pypi/simple --quiet
fi

if [ -x "$LLAFACTORY_CLI" ]; then
    log_ok "llamafactory-cli 可用"
else
    log_err "llamafactory-cli 安装失败"
    exit 1
fi

# ==================== 步骤 4: 创建项目结构 & 下载模型 ====================
echo ""
echo "=========================================================================="
echo "  步骤 4/7: 创建项目结构 & 下载模型"
echo "=========================================================================="

mkdir -p "${PROJECT_DIR}" "${DATA_DIR}" "${CONFIG_DIR}" "${SAVES_DIR}" "${LOG_DIR}"
log_ok "项目目录已创建：${PROJECT_DIR}"

# 下载模型，添加重试机制
if [ ! -f "${MODEL_DIR}/config.json" ]; then
    log_info "清理不完整的模型下载..."
    rm -rf "${MODEL_DIR}" 2>/dev/null || true
    mkdir -p "${MODEL_DIR}"

    log_info "下载模型 ${MODEL_REPO_ID} 到 ${MODEL_DIR} ..."
    MAX_RETRIES=3
    RETRY_COUNT=0

    while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
        RETRY_COUNT=$((RETRY_COUNT + 1))
        log_info "下载尝试 ${RETRY_COUNT}/${MAX_RETRIES}..."

        if modelscope download --model "${MODEL_REPO_ID}" --local_dir "${MODEL_DIR}" 2>&1 | tail -20; then
            if [ -f "${MODEL_DIR}/config.json" ]; then
                log_ok "模型下载完成"
                break
            fi
        fi

        log_warn "下载失败"
        if [ $RETRY_COUNT -lt $MAX_RETRIES ]; then
            rm -rf "${MODEL_DIR}"/* 2>/dev/null || true
            sleep 15
        fi
    done
fi

# 最后验证
if [ ! -f "${MODEL_DIR}/config.json" ]; then
    log_err "模型下载失败或文件未找到"
    ls -la "${MODEL_DIR}"/ 2>&1 || true
    exit 1
fi

# 清理未完成的下载文件
find "${MODEL_DIR}" -name "*.incomplete" -delete 2>/dev/null || true

log_info "模型大小：$(du -sh ${MODEL_DIR} | awk '{print $1}')"

# ==================== 步骤 5: 准备训练数据 ====================
echo ""
echo "=========================================================================="
echo "  步骤 5/7: 准备训练数据 (ShareGPT 格式)"
echo "=========================================================================="

cat > "${DATA_DIR}/train.json" << 'EOF'
[
  {"conversations": [{"from": "human", "value": "你好，请介绍下自己"}, {"from": "gpt", "value": "你好！我是 Qwen，一个由阿里云开发的大型语言模型。我可以帮你回答问题、写代码、写作文等。"}]},
  {"conversations": [{"from": "human", "value": "什么是华为昇腾 NPU？"}, {"from": "gpt", "value": "华为昇腾 NPU 是华为自研的人工智能处理器，专为深度学习推理和训练设计，性能强大。"}]},
  {"conversations": [{"from": "human", "value": "Python 有什么特点？"}, {"from": "gpt", "value": "Python 语法简洁优雅，库生态丰富，适合快速开发，广泛用于 AI、数据分析、Web 开发。"}]},
  {"conversations": [{"from": "human", "value": "什么是机器学习？"}, {"from": "gpt", "value": "机器学习是人工智能的一个分支，通过算法让计算机从数据中学习规律，无需明确编程。"}]},
  {"conversations": [{"from": "human", "value": "什么是深度学习？"}, {"from": "gpt", "value": "深度学习是机器学习的子集，使用多层神经网络处理复杂模式识别任务，如图像、语音、文本。"}]},
  {"conversations": [{"from": "human", "value": "1+1 等于多少？"}, {"from": "gpt", "value": "1+1 等于 2。"}]},
  {"conversations": [{"from": "human", "value": "你是谁？"}, {"from": "gpt", "value": "我是基于 Qwen3-VL 训练的语言模型助手，可以帮助你处理各种问题。"}]},
  {"conversations": [{"from": "human", "value": "今天天气怎么样？"}, {"from": "gpt", "value": "抱歉，我无法获取实时天气信息。你可以告诉我所在城市，我给你一些建议。"}]},
  {"conversations": [{"from": "human", "value": "怎么学习编程？"}, {"from": "gpt", "value": "建议：1. 选择一门语言如 Python；2. 学习基础语法；3. 动手做小项目；4. 阅读优秀代码；5. 加入社区交流。"}]},
  {"conversations": [{"from": "human", "value": "推荐一本书"}, {"from": "gpt", "value": "《Python 编程：从入门到实践》是一本非常适合 Python 初学者的书，内容通俗易懂，案例丰富。"}]}
]
EOF
log_ok "训练数据已创建：${DATA_DIR}/train.json (10 条对话)"

cat > "${DATA_DIR}/dataset_info.json" << 'EOF'
{
  "train": {
    "file_name": "train.json",
    "formatting": "sharegpt",
    "columns": {
      "messages": "conversations"
    }
  }
}
EOF
log_ok "数据集配置已创建"

# ==================== 步骤 6: 创建训练配置 ====================
echo ""
echo "=========================================================================="
echo "  步骤 6/7: 创建 LLaMAFactory 训练配置"
echo "=========================================================================="

cat > "${CONFIG_DIR}/qwen3vl_sft_single.yaml" << EOF
# Qwen3-VL-2B-Instruct SFT 训练配置 (单卡 Ascend NPU)
# 自动生成于 $(date '+%Y-%m-%d %H:%M:%S')

model_name_or_path: ${MODEL_DIR}
trust_remote_code: true

stage: sft
do_train: true
finetuning_type: lora
lora_rank: 8
lora_alpha: 16
lora_dropout: 0.1
lora_target: q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj

dataset_dir: ${DATA_DIR}
dataset: train
template: qwen3_vl
max_samples: 1000
overwrite_cache: true
preprocessing_num_workers: 2

output_dir: ${SAVES_DIR}
logging_steps: 1
save_steps: 50
save_total_limit: 2
plot_loss: true
overwrite_output_dir: true

per_device_train_batch_size: 1
gradient_accumulation_steps: 4
learning_rate: 5.0e-5
num_train_epochs: 3.0
max_grad_norm: 1.0
lr_scheduler_type: cosine
warmup_ratio: 0.03
bf16: true
flash_attn: auto

report_to: none
seed: 42
EOF
log_ok "训练配置已创建：${CONFIG_DIR}/qwen3vl_sft_single.yaml"

# ==================== 步骤 7: 创建启动脚本和文档 ====================
echo ""
echo "=========================================================================="
echo "  步骤 7/7: 创建启动脚本和文档"
echo "=========================================================================="

# 创建一键启动脚本
cat > "${PROJECT_DIR}/start_train.sh" << EOF
#!/bin/bash
# Qwen3-VL SFT 一键训练脚本
cd "\$(dirname "\$0")"

# 修复 CANN 9.0.0 customize op 库的 GLIBCXX_3.4.32 缺失（注入 GCC 15 的 libstdc++）
if [ -f "${CUSTOM_LIBSTDCXX}" ]; then
    export LD_PRELOAD="${CUSTOM_LIBSTDCXX}:\${LD_PRELOAD}"
fi

# 单卡训练（避免 16 卡分布式端口冲突）
export NPROC_PER_NODE=1
export HCCL_NPU_SOCKET_PORT_RANGE=60000-60100

# 启动训练
\$(dirname "\$(command -v python3)")/llamafactory-cli train configs/qwen3vl_sft_single.yaml
EOF
chmod +x "${PROJECT_DIR}/start_train.sh"
log_ok "启动脚本已创建：${PROJECT_DIR}/start_train.sh"

# 创建 README
cat > "${PROJECT_DIR}/README.md" << EOF
# Llama-Factory SFT 训练项目

## 快速开始

\`\`\`bash
# 启动训练
./start_train.sh

# 查看文档
cat TRAINING.md
\`\`\`

## 目录结构

\`\`\`
Llama-Factory/
├── configs/qwen3vl_sft_single.yaml  # 训练配置
├── data/                             # 训练数据
├── models/Qwen/Qwen3-VL-2B-Instruct/ # 基础模型
├── saves/qwen3vl-sft/                # 训练输出
├── logs/                             # 训练日志
├── start_train.sh                    # 启动脚本
├── TRAINING.md                       # 训练文档
└── README.md                         # 本文件
\`\`\`
EOF
log_ok "README.md 已创建"

# ==================== 完成 ====================
echo ""
echo "=========================================================================="
echo "  ✅ 部署完成!"
echo "=========================================================================="
echo ""
log_ok "项目目录：${PROJECT_DIR}"
log_ok "模型路径：${MODEL_DIR}"
log_ok "训练配置：${CONFIG_DIR}/qwen3vl_sft_single.yaml"
echo ""
log_info "下一步:"
echo "  1. 查看训练文档：cat ${PROJECT_DIR}/TRAINING.md"
echo "  2. 启动训练：     cd ${PROJECT_DIR} && ./start_train.sh"
echo ""

# ==================== 自动训练（可选） ====================
if [ "$AUTO_TRAIN" = true ]; then
    echo "=========================================================================="
    echo "  自动启动训练 (--train)"
    echo "=========================================================================="
    cd "${PROJECT_DIR}"
    if [ -n "${CUSTOM_LIBSTDCXX}" ] && [ -f "${CUSTOM_LIBSTDCXX}" ]; then
        export LD_PRELOAD="${CUSTOM_LIBSTDCXX}:${LD_PRELOAD}"
    fi
    export NPROC_PER_NODE=1
    export HCCL_NPU_SOCKET_PORT_RANGE=60000-60100
    $LLAFACTORY_CLI train "${CONFIG_DIR}/qwen3vl_sft_single.yaml" 2>&1 | tee "${LOG_DIR}/train_$(date +%Y%m%d_%H%M%S).log"
fi
