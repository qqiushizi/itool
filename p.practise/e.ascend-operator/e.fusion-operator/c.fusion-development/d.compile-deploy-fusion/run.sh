#!/bin/bash
# ============================================================
# 实验: d.compile-deploy-fusion
# 说明: 融合算子编译部署、aclnn 接口
# 模块: p.practise 第一性原理学习实验
# ============================================================
# 【第一性原理】
# 融合算子部署的 4 个层次:
#   1. AscendC kernel: __aicore__ 函数 (.cpp)
#   2. TIK/AscendC 编译: bisheng + clang -> .o -> .so
#   3. aclnn 接口: C++ 包装层, 对外暴露统一 API
#   4. Python 绑定: torch.ops 或 pybind11
# 编译工具链:
#   - bisheng: 毕昇编译器 (CANN 自带, 基于 clang)
#   - clang: LLVM 编译器前端
#   - CMake: 跨平台构建
#   - make: 增量编译
# 关键概念:
#   - aclnn: Ascend Compute Language Neural Network, 标准 C++ API
#   - 算子二进制: 不同芯片 (910/310) 需重编
#   - 算子兼容性: 跨 NPU 型号运行
# ============================================================
set -euo pipefail
echo "############################################################"
echo "# 实验: d.compile-deploy-fusion | 融合算子编译部署、aclnn 接口"
echo "############################################################"

python3 <<'PYEOF'
def hdr(n,total,t): print(f"\n{'='*60}\n【步骤 {n}/{total}】{t}\n{'='*60}")
def why(s): print("\n--- 做什么 & 为什么 ---\n  "+s.replace("\n","\n  "))
def res(s): print("\n--- 结果 ---\n  "+s.replace("\n","\n  "))
def mea(s): print("\n--- 解读 ---\n  "+s.replace("\n","\n  "))
TOTAL = 4

# --- 1. 编译工具链 ---
hdr(1,TOTAL,"编译工具链:bisheng + clang")
why("""AscendC 算子编译需要:
  - bisheng: 毕昇编译器 (CANN 自带, ARM 芯片)
  - clang: LLVM 前端 (处理 C++)
  - 链接器: 生成 .so
  - 头文件: kernel_operator.h (AscendC 库)
  - 标志: -O2, --std=c++17, -fopenmp""")
out = ["  阶段      工具         输入        输出"]
out.append("  1. 前端    clang        .cpp        .o (目标文件)")
out.append("  2. 优化    bisheng -O2  .o          .o (优化后)")
out.append("  3. 链接    clang        .o          .so (动态库)")
out.append("  4. 符号    strip        .so         .so (去符号)")
out.append("  5. 安装    cp           .so         OPP 目录")
res("\n".join(out))
mea("""编译选项关键点:
  -O2: 必开, 性能差 30%+
  --std=c++17: AscendC 库要求
  -fopenmp: 启用 OpenMP 并行 (CPU 调试用)
  -mcpu=tsv110: 指定目标芯片 (Ascend 910 = tsv110, 310 = tsv200)
  错误排查:
  - 找不到 kernel_operator.h: 检查 CANN 头文件路径
  - undefined reference: 漏链接库 (ascendcl)
  - 段错误: 内存越界, 多为 UB 太小""")

# --- 2. CMake 构建脚本 ---
hdr(2,TOTAL,"CMake 构建脚本")
why("""复杂项目用 CMake 管理, 单文件用 make 也行。""")
res("""# CMakeLists.txt
cmake_minimum_required(VERSION 3.16)
project(fused_ops LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# CANN 路径
if(NOT ENV{ASCEND_HOME})
  set(ASCEND_HOME "/usr/local/Ascend/ascend-toolkit/latest")
endif()
include_directories(${ASCEND_HOME}/include)
include_directories(${ASCEND_HOME}/include/aclnn)
include_directories(${ASCEND_HOME}/include/ascendc)

# AscendC 库
add_library(fused_linear_relu_add SHARED
    src/linear_relu_add.cpp
)
target_compile_options(fused_linear_relu_add PRIVATE
    -O2
    -fopenmp
    -mcpu=tsv110
)
target_link_libraries(fused_linear_relu_add PRIVATE
    ascendcl
    runtime
)

# 编译
# mkdir build && cd build
# cmake .. -DCMAKE_CXX_COMPILER=clang++
# make -j8
# 输出: libfused_linear_relu_add.so""")
mea("""CMake 要点:
  1. CANN 路径: 必填, 否则找不到头文件
  2. 编译选项: -O2 -mcpu=tsv110
  3. 链接: ascendcl, runtime 库
  4. 多芯片: 用 if(ENV{CHIP_TYPE}) 判断
  5. 多算子: 1 个 add_library per 算子
工具链配置:
  - CMAKE_CXX_COMPILER=clang++ (CANN 自带)
  - 否则 bisheng 不生效""")

# --- 3. aclnn 接口包装 ---
hdr(3,TOTAL,"aclnn 接口:标准 C++ 包装")
why("""aclnn = Ascend Compute Language NN, 是算子对外的标准 API。
作用:
  1. 跨语言: Python/C++/Java 都能调
  2. 跨平台: 910/310/390p 同接口
  3. 异步: 支持 stream, 不阻塞 CPU
aclnn 接口包含:
  - aclnnXxxGetWorkspaceSize: 算 workspace
  - aclnnXxx: 真正计算
  - 配套: 申请 device buffer, 拷贝数据""")
res("""// 文件: aclnn_linear_relu_add.cpp
#include \"aclnn/linear_relu_add.h\"

extern \"C\" aclnnStatus aclnnLinearReLUAddGetWorkspaceSize(
    const aclTensor* x, const aclTensor* w, const aclTensor* b, const aclTensor* z,
    const aclTensor* y, uint64_t* workspaceSize, aclOpExecutor** executor
) {
    // 1. 创建中间 tensor (累加器)
    auto c = aclCreateTensor(M, N, ACL_FLOAT);
    
    // 2. 算 workspace 大小
    *workspaceSize = 0;
    *executor = nullptr;
    
    // 3. 调度算子
    auto ret = aclnnFusedLinearReLUAddGetWorkspaceSize(
        x, w, b, c, z, y, workspaceSize, executor
    );
    return ret;
}

extern \"C\" aclnnStatus aclnnLinearReLUAdd(
    void* workspace, uint64_t workspaceSize,
    aclOpExecutor* executor, aclrtStream stream
) {
    return aclnnFusedLinearReLUAdd(workspace, workspaceSize, executor, stream);
}""")
mea("""aclnn 接口要点:
  1. GetWorkspaceSize + 计算 分开 (异步调度)
  2. workspace 在 device 上 (HBM)
  3. aclTensor 包装 device buffer + shape + dtype
  4. stream 必传 (NPU 是异步的)
  5. 返回值: ACL_SUCCESS/ACL_ERROR_xxx""")

# --- 4. Python 端调用 ---
hdr(4,TOTAL,"Python 端调用:torch.ops / pybind11")
why("""aclnn 是 C++, Python 调要包一层。两类方式:
  1. torch.ops (CANN 内置, 标准)
  2. pybind11 (自定义, 灵活)
torch.ops 路径:
  - 算子装在 OPP 目录
  - CANN 自动注册到 torch.ops
  - 业务代码 torch.ops.myops.xxx 调用""")
res("""# 方式 1: torch.ops (推荐)
import torch
y = torch.ops.myops.linear_relu_add(x, w, b, z)
# 框架自动处理 device 拷贝 + dtype 转换

# 方式 2: pybind11 (自定义 C++ 算子)
# 文件: bindings.cpp
#include <torch/extension.h>
#include \"aclnn_linear_relu_add.h\"

torch::Tensor linear_relu_add(
    torch::Tensor x, torch::Tensor w, torch::Tensor b, torch::Tensor z
) {
    // 转 aclTensor
    auto y = torch::empty({x.size(0), w.size(1)}, x.options());
    
    // 算 workspace
    uint64_t ws_size;
    aclOpExecutor* exec;
    aclnnLinearReLUAddGetWorkspaceSize(
        aclTensorFromTorch(x), aclTensorFromTorch(w),
        aclTensorFromTorch(b), aclTensorFromTorch(z),
        aclTensorFromTorch(y), &ws_size, &exec
    );
    
    // 申请 workspace + 算
    void* ws = nullptr;
    if (ws_size > 0) aclrtMalloc(&ws, ws_size, ACL_MEM_MALLOC_HUGE_FIRST);
    auto stream = aclrtGetCurrentStream();
    aclnnLinearReLUAdd(ws, ws_size, exec, stream);
    if (ws) aclrtFree(ws);
    
    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(\"linear_relu_add\", &linear_relu_add, \"Fused Linear+ReLU+Add\");
}

# 编译:
# python3 setup.py build_ext --inplace
# 调用:
import fused_ops
y = fused_ops.linear_relu_add(x, w, b, z)""")
mea("""Python 调用的工程选择:
  - 简单标准算子: torch.ops (CANN 已注册)
  - 自定义复杂算子: pybind11 (灵活, 性能好)
  - 跨语言项目: aclnn (统一 C++ 接口)
  性能注意:
  - 频繁调用 (per-step) 要避免 Python 开销
  - 大 batch 算 1 次比 多次 Python 调更优
  - 算子要 in-place 或返回新 tensor, 不混用""")
PYEOF

echo ""
echo "############################################################"
cat <<'EOF'
【整体总结】
- 小白:融合算子部署 4 层: AscendC kernel -> 编译 .so -> aclnn 接口 -> Python 调用;
  bisheng + clang 编译, CMake 管理; 跨芯片要重编;
  aclnn 是标准 C++ API, 异步 stream, 跨语言。
- 熟手:编译选项 -O2 -mcpu=tsv110 必开; aclnn 接口 GetWorkspaceSize + 算 分开;
  workspace 在 device HBM, 不用每次申请/释放; Python 调 2 选 1:
  torch.ops (标准) 或 pybind11 (自定义); 频繁调用要避免 Python 开销。
【进阶】完整走通 1 个融合算子: 写 .cpp -> CMake 构建 -> 装到 OPP ->
  写 Python 脚本调用, 测精度和性能; 试不同 -O 级别和 tile 大小, 找最优配置。
EOF
echo "############################################################"
