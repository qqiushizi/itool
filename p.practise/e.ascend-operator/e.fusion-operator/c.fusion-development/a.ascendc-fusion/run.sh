#!/bin/bash
# ============================================================
# 实验: a.ascendc-fusion
# 说明: Ascend C 融合算子开发流程、tiling、临时变量
# 模块: p.practise 第一性原理学习实验
# ============================================================
# 【第一性原理】
# Ascend C 写融合算子的 5 步:
#   1. 设计: 确定要融哪些算子, IO 收益多少
#   2. 切分: 大问题切 tile, UB 装得下
#   3. 写 kernel: __aicore__ 函数, 显式 Set/WaitFlag
#   4. 编译: bisheng + clang -> .so
#   5. 注册: op_info.cfg + asc_op_compiler
# 关键概念:
#   - Tiling 策略: 决定 tile 大小, 影响复用
#   - 临时变量: UB 上中转数据, 大小要省
#   - Pipe 同步: SetFlag/WaitFlag 控制 5 段
#   - 调度: scheduler 控制多核分配
# ============================================================
set -euo pipefail
echo "############################################################"
echo "# 实验: a.ascendc-fusion | AscendC 融合算子:开发流程 + tiling"
echo "############################################################"

python3 <<'PYEOF'
def hdr(n,total,t): print(f"\n{'='*60}\n【步骤 {n}/{total}】{t}\n{'='*60}")
def why(s): print("\n--- 做什么 & 为什么 ---\n  "+s.replace("\n","\n  "))
def res(s): print("\n--- 结果 ---\n  "+s.replace("\n","\n  "))
def mea(s): print("\n--- 解读 ---\n  "+s.replace("\n","\n  "))
TOTAL = 4

# --- 1. 融合算子开发 5 步 ---
hdr(1,TOTAL,"融合算子开发 5 步")
why("""从需求到上线的完整流程:""")
out = ["  步     动作                      输出"]
out.append("  1. 设计  确定融合算子,算子边界    设计文档")
out.append("  2. 切分  选 tile 大小,UB 装得下   tiling 策略")
out.append("  3. 写    __aicore__ kernel       .cpp 文件")
out.append("  4. 编译  bisheng + clang         .o + .so")
out.append("  5. 注册  op_info.cfg + 测试      算子可用")
res("\n".join(out))
mea("融合算子 vs 普通算子:\n  - 普通算子: 实现 1 个操作 (Add, ReLU)\n  - 融合算子: 实现 N 个操作 (Linear+ReLU+Add)\n  开发成本 2-3×, 但收益 1.3-2×")

# --- 2. Tiling 策略 ---
hdr(2,TOTAL,"Tiling 策略:决定性能")
why("""融合算子的 tiling 考虑更多:
  - 参与融合的算子,所有 tile 大小需一致
  - 中间结果 tile 也要装在 UB
  - 双缓冲需要 2× UB
  - 临时变量: scale, mask, indices 等""")
out = ["  算子                UB 占用 (单 tile)    备注"]
out.append("  Linear (128x128)     A 32K + B 32K + out 32K = 96K")
out.append("  + 累加器 FP32        +64K = 160K")
out.append("  + 临时 (bias)        +1K = 161K")
out.append("  + ReLU (无)         0 (in-place)")
out.append("  + 双缓冲            x2 = 322K  OOM!")
out.append("  降 tile 到 64x64    减半 = 161K 装得下")
res("\n".join(out))
mea("Tiling 经验:\n  - 简单融合 (Linear+ReLU): tile 不变\n  - 复杂融合 (Linear+ReLU+Add+LayerNorm): 需降 tile\n  - 双缓冲: 必降 tile\n  - 经验: 128x128 太大会 OOM, 试 64x64")

# --- 3. 临时变量管理 ---
hdr(3,TOTAL,"临时变量:UB 空间竞争")
why("""融合算子比普通算子多用临时变量:
  - mask (for attn)
  - indices (for sort/gather)
  - scale, shift (for norm)
  - 累加器 (matmul)
  - 反向中间值 (训练用)
策略:
  - 复用: 一个 UB 区在不同阶段用
  - 释放: 不用了立刻释放 (AscendC 不自动 GC)""")
out = ["  临时变量        大小     何时释放"]
out.append("  累加器          m1*n1*4  scale 前")
out.append("  scale 值         1       softmax 前")
out.append("  mask (attn)     seq*1    softmax 后")
out.append("  临时 mask       seq*1    dropout 后")
out.append("  临时累加 (RMS)  d        normalize 前")
out.append("  总 UB 占用 = 累加器 + 临时 =  ~120 KB (128x128 tile)")
res("\n".join(out))
mea("""实战技巧:
  1. 用 AscendC::LocalTensor 显式管理
  2. 不再用立刻用 FreeLocalTensor()
  3. 同阶段临时变量复用 1 个 UB 区
  4. 累加器通常最后释放 (后续可能用)""")

# --- 4. 完整例子:Linear+ReLU+Add 融合 ---
hdr(4,TOTAL,"完整例子:Linear+ReLU+Add")
why("""实战一个简单融合算子:""")
res("""// file: linear_relu_add.cpp
#include \"kernel_operator.h\"
using namespace AscendC;

extern \"C\" __global__ __aicore__ void linear_relu_add(
    __gm__ float* X, __gm__ float* W, __gm__ float* B, __gm__ float* Y,  // inputs
    uint32_t M, uint32_t N, uint32_t K
) {
  // Tiling
  uint32_t block_idx = GetBlockIdx();
  uint32_t tile_m = 128, tile_n = 128, tile_k = 32;
  uint32_t m_start = block_idx * tile_m;
  
  // 临时变量 (UB)
  __local__ float a_local[tile_m * tile_k];
  __local__ float b_local[tile_k * tile_n];
  __local__ float c_local[tile_m * tile_n];      // 累加器
  __local__ float y_local[tile_m * tile_n];
  
  // 累加器初始化为 0
  InitBuf(c_local, tile_m * tile_n);
  
  // K 维累加
  for (uint32_t k = 0; k < K; k += tile_k) {
    // 搬 A, B (MTE2)
    DataCopy(a_local, X + m_start * K + k, tile_m * tile_k);
    DataCopy(b_local, W + k * N, tile_k * tile_n);
    SetFlag<AscendC::HardEvent::MTE2_S>(0);
    WaitFlag<AscendC::HardEvent::MTE2_S>(0);
    
    // 算 matmul (Cube)
    Matmul(c_local, a_local, b_local, tile_m, tile_n, tile_k);
    SetFlag<AscendC::HardEvent::MAT_S>(1);
    WaitFlag<AscendC::HardEvent::MAT_S>(1);
  }
  
  // epilogue: + bias + ReLU + Add (Vector, fused)
  // (具体实现省略, 核心是 1 个 vector kernel 完成这 3 个操作)
  ...
  
  // 写回
  DataCopy(Y + m_start * N, y_local, tile_m * tile_n);
}""")
mea("完整 fused Linear+ReLU+Add = 1 个 __aicore__ kernel, 充分利用 Cube + Vector + 双发射。\n  上线流程: bisheng 编译 -> 注册 -> torch.ops.myops.linear_relu_add")
PYEOF

echo ""
echo "############################################################"
cat <<'EOF'
【整体总结】
- 小白:AscendC 融合算子 = __aicore__ 函数,多步合 1 步;tile 大小受 UB 限制;
  临时变量手动管理;编译用 bisheng+clang;注册用 op_info.cfg。
- 熟手:Tiling 决定性能,简单融合 tile 不变,复杂融合需降 tile;
  临时变量复用 + 显式释放;Set/WaitFlag 控制 pipe 同步;
  双发射让 Cube 和 Vector 并行是核心。
【进阶】读 CANN 官方 sample 中的 fused 算子;尝试写一个 Linear+Bias+ReLU
  算子,完整走 编译 -> 注册 -> Python 调用 流程。
EOF
echo "############################################################"
