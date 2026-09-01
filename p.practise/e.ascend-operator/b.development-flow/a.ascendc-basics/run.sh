#!/bin/bash
# ============================================================
# 实验: a.ascendc-basics
# 说明: Ascend C 编程模型、tiling、API
# 模块: p.practise 第一性原理学习实验
# ============================================================
# 【第一性原理】
# Ascend C 是华为为昇腾 NPU 设计的 C++ 算子开发语言,关键概念:
#   1. Kernel 入口: __global__ __aicore__ void kernel()
#   2. Tiling 策略: 大问题切成小块, 在 UB 装下
#   3. 5 段流水线: MTE1/MTE2/Cube/MTE3/MTE4
#   4. 同步 API: SetFlag / WaitFlag 控制 pipe 依赖
#   5. 内存对象: GlobalTensor (HBM) / LocalTensor (UB)
# 类比:
#   - Kernel = CUDA __global__ 函数
#   - Block = AscendC 一个 AICore
#   - Tile = tile 大小 (UB 装得下的数据块)
#   - Tiling 策略 = grid 维度 / block 维度
# ============================================================
set -euo pipefail
echo "############################################################"
echo "# 实验: a.ascendc-basics | Ascend C 编程模型:kernel/tiling/API"
echo "############################################################"

python3 <<'PYEOF'
def hdr(n,total,t): print(f"\n{'='*60}\n【步骤 {n}/{total}】{t}\n{'='*60}")
def why(s): print("\n--- 做什么 & 为什么 ---\n  "+s.replace("\n","\n  "))
def res(s): print("\n--- 结果 ---\n  "+s.replace("\n","\n  "))
def mea(s): print("\n--- 解读 ---\n  "+s.replace("\n","\n  "))
TOTAL = 4

# --- 1. Hello World kernel ---
hdr(1,TOTAL,"Ascend C 最小算子:Hello World")
why("""最简 Ascend C kernel,展示 4 个必备元素:""")
res("""#include \"kernel_operator.h\"
using namespace AscendC;

extern \"C\" __global__ __aicore__ void add_custom(__gm__ float* x, __gm__ float* y, __gm__ float* z, uint32_t total) {
  // 1. 准备:每个核处理一块数据
  uint32_t block_idx = GetBlockIdx();          // 当前核 id
  uint32_t tile = 1024;                        // 每块 1024
  uint32_t offset = block_idx * tile;

  // 2. 局部内存 (UB)
  __local__ float local_x[1024];
  __local__ float local_y[1024];
  __local__ float local_z[1024];

  // 3. 搬数据 (MTE)
  DataCopy(local_x, x + offset, tile);
  DataCopy(local_y, y + offset, tile);

  // 4. 算 (Vector)
  Add(local_z, local_x, local_y, tile);

  // 5. 写回 (MTE)
  DataCopy(z + offset, local_z, tile);
}""")
mea("""4 个必备元素:
  1. __global__ __aicore__ 修饰: 表示是 AICore 上跑的 kernel
  2. __gm__ 修饰: GlobalTensor 指针, 指向 HBM
  3. __local__ 修饰: LocalTensor 数组, 实际在 UB
  4. DataCopy + Add: MTE 搬运 + Vector 计算""")

# --- 2. Tiling 策略 ---
hdr(2,TOTAL,"Tiling 策略:大矩阵切块")
why("""HBM 1GB 数据, UB 仅 256KB → 必须切块。
Tiling = 把大问题切多个 tile, 每个 tile 装入 UB 计算。
设计 tiling 时考虑:
  1. 每个 tile 装得进 UB
  2. tile 越大, 数据复用越多
  3. 边界 tile 要特殊处理 (不能整除时)""")
res("""例: GEMM C = A @ B, M=N=K=4096, tile 大小 128
  A tile  (128, 128) = 32 KB FP16  ← 装得下
  B tile  (128, 128) = 32 KB FP16
  C tile  (128, 128) = 32 KB FP16
  累加器 (128, 128) FP32 = 64 KB    ← 装得下
  合计:  160 KB ≤ 256 KB UB ✓
  
  Grid: (M/128) * (N/128) = 32 * 32 = 1024 个 Block""")
mea("""Tiling 设计经验:
  1. 选最大的 tile (但不超过 UB)
  2. 优先 128x128 (成熟) 或 256x128 (大数据)
  3. 边界 tile 用 padding 填 0
  4. 多维并行 (M 维 + N 维) 比单维更均匀""")

# --- 3. 常用 API 速查 ---
hdr(3,TOTAL,"常用 API 速查表")
why("""Ascend C API 三大类:""")
res("""搬运类 (MTE):
  DataCopy(dst, src, size)              HBM<->UB
  DataCopyPad(dst, src, pad_mode)       带 padding 的搬运
  SetGlobalBuffer / SetLocalBuffer      设置指针
  
计算类 (Vector/Cube):
  Add / Sub / Mul / Div                 逐元素
  Matmul                                矩阵乘 (Cube)
  Reduce / ReduceMax / ReduceSum        归约
  Cast                                  类型转换
  Exp / Log / Sqrt / Rsqrt              数学函数
  
同步类 (Pipe):
  SetFlag<PIPE_MTE2>(id)                设置同步事件
  WaitFlag<PIPE_MTE2>(id)               等待事件
  CrossCoreWaitFlag(id)                 跨核同步
  PipeBarrier<PIPE_V>()                 Pipe barrier""")
mea("""API 选择原则:
  - 数据搬运: 必用 MTE API
  - 矩阵乘: 必用 Matmul (走 Cube)
  - 元素级: Vector API
  - 同步: Set/WaitFlag 显式控制
  - 跨核: CrossCoreSync""")

# --- 4. 编译与运行 ---
hdr(4,TOTAL,"编译运行全流程")
why("""Ascend C 算子从源码到运行:""")
res("""开发流程:
  1. 写 .cpp 文件 (含 kernel 实现)
  2. 编译: bisheng compiler (.cpp -> .o)
  3. 链接: clang (+ runtime lib)
  4. 生成算子:  .so 文件
  5. 注册: 通过自定义算子机制注册
  6. 调用: torch_npu op 或 aclnn 接口

示例编译:
  $CANN_HOME/bin/bisheng \\
    --cce-aicore-arch=davinci-xxx \\
    -O2 -std=c++17 \\
    add_custom.cpp -o add_custom.o
  $CANN_HOME/bin/clang++ \\
    add_custom.o -lascendcl -o add_custom.so
  python3 -c \"import torch; torch.ops.myops.add_custom(x, y, z, n)\"""")
mea("""常见问题:
  - 编译慢 (5-10 分钟一次): 缓存 .o
  - 链接错: 检查 CANN 环境变量
  - 运行时 OOM: tile 太大
  - 数据错: 同步漏了 Set/WaitFlag
  - 性能差: profile 看是 MTE 还是 Compute 是瓶颈""")
PYEOF

echo ""
echo "############################################################"
cat <<'EOF'
【整体总结】
- 小白:Ascend C = 昇腾 NPU 的 C++ 算子开发语言;kernel 用 __aicore__
  修饰,数据用 __gm__ (HBM) 和 __local__ (UB);Tiling 切块让大矩阵装得下;
  编译用 bisheng + clang,生成 .so 后注册调用。
- 熟手:5 段流水线 (MTE1/MTE2/Cube/MTE3/MTE4) 显式同步;Tile 大小优先
  128x128,边界用 padding;Matmul API 走 Cube,元素级走 Vector;profile
  看是 MTE 还是 Compute 瓶颈。
【进阶】读 CANN 官方 sample, 找一个 elementwise 算子读懂;再尝试加 tiling
  和双缓冲,对比性能。
EOF
echo "############################################################"
