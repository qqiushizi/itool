#!/bin/bash
# ============================================================
# torchbind 接入工程 (torch CPU / torch_npu NPU / vllm_ascend)
# 生成一套可编译的 torch 扩展骨架:
#   <op>.cpp + setup.py           # CPU 侧 (torch CppExtension)
#   <op>_npu.cpp + setup_npu.py   # NPU 侧 (torch_npu NpuExtension + op_api)
#   README.md                     # 编译/安装/验证/vllm_ascend 接入说明
# 用法:
#   bash run.sh [算子名] [输出目录]
#   默认: 算子名 AddCustom, 输出目录 ./torchbind_AddCustom
# ============================================================
set -euo pipefail

CYAN='\033[0;36m'; GREEN='\033[0;32m'; RESET='\033[0m'

OP="${1:-AddCustom}"
OUT="${2:-torchbind_${OP}}"
mkdir -p "$OUT"

echo "=========================================================="
echo " 生成 torchbind 接入工程: $OUT"
echo " 算子名: $OP"
echo "=========================================================="

# ============================================================
# 1) CPU 侧示例 (torch::extension)
# ============================================================
cat > "$OUT/${OP}.cpp" <<CPP_EOF
#include <torch/extension.h>

// 示例自定义算子: ${OP} = x + y  (CPU)
// 真实场景把这里替换为你的 kernel 调用。
torch::Tensor ${OP}_forward(torch::Tensor x, torch::Tensor y) {
    return x + y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("${OP}", &${OP}_forward, "${OP} 自定义算子 (CPU)");
}
CPP_EOF

cat > "$OUT/setup.py" <<PY_EOF
from torch.utils.cpp_extension import BuildExtension, CppExtension
from setuptools import setup

setup(
    name="${OP}",
    version="0.1.0",
    ext_modules=[CppExtension("${OP}", ["${OP}.cpp"])],
    cmdclass={"build_ext": BuildExtension},
)
PY_EOF

# ============================================================
# 2) NPU 侧示例 (torch_npu NpuExtension + op_api 模板)
# ============================================================
cat > "$OUT/${OP}_npu.cpp" <<NPU_CPP_EOF
#include <torch/extension.h>
// torch_npu 常用头(版本间路径可能微调, 以实际安装为准)
#include <torch_npu/csrc/framework/utils/OpPreparation.h>
#include <torch_npu/csrc/framework/utils/OpAdapter.h>
#include <torch_npu/csrc/aten/NPUNativeFunctions.h>

// 示例 NPU 算子: ${OP} = x + y
// 方式一(本模板): 用 op_api 组合已有算子。
// 方式二(自研): 把函数体替换为调用你用 AscendC(msopgen) 编译出的 kernel,
//             例如 aclnn${OP} 或 dlopen 你的 .so 后调用其符号。
at::Tensor ${OP}_npu_forward(at::Tensor x, at::Tensor y) {
    // 1. 按 x 的元信息在 NPU 上分配输出张量
    at::Tensor out = at_npu::native::OpPreparation::apply_tensor(x);
    // 2. 调用 op_api 组合算子(此处为 add, alpha=1.0)
    at_npu::native::op_api::add_out(x, y, 1.0, out);
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("${OP}_npu", &${OP}_npu_forward, "${OP} NPU 自定义算子");
}
NPU_CPP_EOF

cat > "$OUT/setup_npu.py" <<NPU_PY_EOF
# 在 NPU 机器上构建: source /usr/local/Ascend/ascend-toolkit/set_env.sh 后再执行
from torch_npu.utils.cpp_extension import NpuExtension, BuildExtension
from setuptools import setup

setup(
    name="${OP}_npu",
    version="0.1.0",
    ext_modules=[NpuExtension("${OP}_npu", ["${OP}_npu.cpp"])],
    cmdclass={"build_ext": BuildExtension},
)
NPU_PY_EOF

# ============================================================
# 3) README
# ============================================================
cat > "$OUT/README.md" <<MD_EOF
# ${OP} torchbind 接入工程

## 目录

\`\`\`
${OP}.cpp          CPU 示例
setup.py           CPU 构建
${OP}_npu.cpp      NPU 示例(op_api 组合算子)
setup_npu.py       NPU 构建
\`\`\`

## 一、CPU 侧编译 & 验证

\`\`\`bash
cd ${OUT}
python setup.py install          # 或 pip install .
\`\`\`

\`\`\`python
import torch, ${OP}
x = torch.ones(4); y = torch.ones(4)
print(${OP}.${OP}(x, y))
\`\`\`

## 二、NPU 侧编译 & 验证

\`\`\`bash
# 1. 加载 CANN 环境
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# 2. 构建(会用 NpuExtension 走昇腾工具链)
cd ${OUT}
python setup_npu.py install

# 3. 验证
\`\`\`

\`\`\`python
import torch, torch_npu, ${OP}_npu
torch.npu.set_device(0)
x = torch.ones(4, device='npu'); y = torch.ones(4, device='npu')
print(${OP}_npu.${OP}_npu(x, y))
\`\`\`

## 三、接入自研 AscendC kernel

若 kernel 由 msopgen 生成(见 ../a.msopgen), 典型做法:

1. 编译出 \`.so\` 后, 在 \`${OP}_npu.cpp\` 中用 \`dlopen/dlSym\` 加载并调用;
2. 或把 kernel 的 host 入口直接 include 进来, 在 forward 里构造 tiling 并 launch;
3. 注册进 vllm_ascend 的算子映射后, vLLM 即可在 NPU 上调用本算子。
MD_EOF

echo ""
echo -e "${GREEN}✔ 已生成:${RESET}"
find "$OUT" -type f | sort
echo ""
echo -e "${CYAN}===== 编译 / 安装指导 =====${RESET}"
cat <<GUIDE
  # CPU
  cd $OUT && python setup.py install

  # NPU (需在昇腾机器上, 先 source set_env.sh)
  cd $OUT && python setup_npu.py install
GUIDE
