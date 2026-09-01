#!/bin/bash
# ============================================================
# 实验: a.ascend-hardware
# 说明: 达芬奇架构、AI Core、Cube/Vector/AI CPU
# 模块: p.practise 第一性原理学习实验
# ============================================================
# 【第一性原理】
# 昇腾 NPU 基于 \"达芬奇 (Da Vinci)\" 架构,核心是 AI Core:
#   1. Cube 单元: 矩阵乘 (16x16x16 FP16/INT8),专门加速 GEMM
#   2. Vector 单元: 向量运算 (elementwise, reduction, activation)
#   3. AI CPU: 标量控制流 + 杂项算子 (与 ARM 协同)
# 类比 NVIDIA:
#   Cube  ≈ Tensor Core
#   Vector ≈ CUDA Core
#   AI CPU ≈ 标量单元
# 关键差异:
#   - 达芬奇是\"分离式\",Cube 和 Vector 独立调度
#   - 同一指令可同时调度 Cube 和 Vector (双发射)
#   - L2/UB/HBM 内存层次更细
# ============================================================
set -euo pipefail
echo "############################################################"
echo "# 实验: a.ascend-hardware | 达芬奇架构, AI Core, Cube/Vector"
echo "############################################################"

python3 <<'PYEOF'
def hdr(n,total,t): print(f"\n{'='*60}\n【步骤 {n}/{total}】{t}\n{'='*60}")
def why(s): print("\n--- 做什么 & 为什么 ---\n  "+s.replace("\n","\n  "))
def res(s): print("\n--- 结果 ---\n  "+s.replace("\n","\n  "))
def mea(s): print("\n--- 解读 ---\n  "+s.replace("\n","\n  "))
TOTAL = 4

# --- 1. 达芬奇架构总览 ---
hdr(1,TOTAL,"达芬奇架构总览")
why("""一个 AI Core 包含:
  - Cube 单元 (16×16×16):  矩阵乘
  - Vector 单元: 4096-bit 宽向量运算
  - AI CPU: 标量控制 + 杂项
  - 标量/向量寄存器
  - UB (Unified Buffer): 片上 SRAM (~256 KB)
  - L1 / L0 Cache
  - MTE (Memory Transfer Engine): 负责数据搬运""")
res("""AI Core 内部结构(简化):
  ┌─────────────────────────────────────┐
  │ L1/L0 Cache (几十 KB)              │
  │ ┌──────────┐  ┌──────────┐         │
  │ │ Cube     │  │ Vector   │         │
  │ │ 16x16x16 │  │ 4096-bit │         │
  │ │ FP16 GEMM│  │ SIMD     │         │
  │ └──────────┘  └──────────┘         │
  │ ┌──────────┐  ┌──────────┐         │
  │ │ AI CPU   │  │ MTE      │         │
  │ │ 标量     │  │ 数据搬运 │         │
  │ └──────────┘  └──────────┘         │
  │ UB (Unified Buffer, 256KB)         │
  └─────────────────────────────────────┘
  HBM  ──── (片外, 几十 GB)""")
mea("Cube + Vector 可同时工作(双发射),这是达芬奇的核心优势。\n  一条 \"gemv\" 可能同时跑 Cube 算 matmul + Vector 算 bias + activation。")

# --- 2. 算力对比 ---
hdr(2,TOTAL,"昇腾 vs NVIDIA 算力对比")
why("""同代产品对比(2023-2024):""")
res("""产品              FP16 (TFLOPS)   INT8 (TOPS)   HBM (GB)
  Atlas 800T A2     280             560            80
  Atlas 900I A3     560             1120           128
  A100              312             624            80
  H100              990             1970           80
  H200              989             1970           141""")
mea("""A3 (2024) FP16 560 TFLOPS,接近 H100 的 57%;
  INT8 1120 TOPS,达到 H100 的 57% 左右。
  差距在缩小,且昇腾有 \"集群扩展\" 优势 (Atlas 9000 384 卡集群)。""")

# --- 3. 内存层次 ---
hdr(3,TOTAL,"内存层次:HBM → L2 → L1 → UB → Register")
why("""类似 CUDA 的 global → L2 → L1 → shared → register,达芬奇有:
  - HBM: 片外, 80 GB, 带宽 2-3 TB/s
  - L2: 片上, 几十 MB
  - L1 / L0: 1-4 MB
  - UB (Unified Buffer): 256 KB, AI Core 私有, 极快
  - 寄存器: 几十 KB, 最快""")
res("""内存层次                大小     带宽/延迟       用途
  Register (REG)        32 KB    ~1 cycle      算子输入
  Unified Buffer (UB)    256 KB   ~10 cycles    算子中间结果
  L1 / L0 Cache         1-4 MB   ~30 cycles    tile 缓存
  L2 Cache              几十 MB  ~100 cycles   全局缓冲
  HBM                   80 GB    2-3 TB/s     权重 / 完整数据""")
mea("算子优化的核心: 尽量让数据在 UB/Register 复用, 减少 HBM 读写。\n  算子融合的本质 = 把多个小算子的中间结果留在 UB, 不回 HBM。")

# --- 4. AICore 工作模式 ---
hdr(4,TOTAL,"AICore 工作流:5 段流水线")
why("""一个算子在 AICore 跑的 5 段:""")
res("""MTE1 (数据搬运 1):   HBM → L1/UB
  MTE2 (数据搬运 2):   L1 → UB
  Cube/Vector 计算:     读 UB 写回 UB
  MTE3 (结果回写):      UB → L1
  MTE4 / DMA:           L1 → HBM""")
mea("""这 5 段可流水: 不同指令可重叠执行。
  优化算子 = 让 MTE 和计算 overlap, 减少空闲。
  实战: 用 AscendC 写算子时, 用 Pipe API 显式控制 pipeline stage。
  类比 CUDA: 同步 copy_ + compute kernel 的 stream。""")
PYEOF

echo ""
echo "############################################################"
cat <<'EOF'
【整体总结】
- 小白:达芬奇架构 = Cube 算矩阵乘 + Vector 算向量 + AI CPU 标量;
  双发射可同时跑;Atlas 800T A2 / 900I A3 算力接近同期 NVIDIA;
  优化算子 = 让数据在片上 (UB/L1) 复用, 减少 HBM 读写。
- 熟手:AI Core 5 段流水线 (MTE1/MTE2/Calc/MTE3/MTE4) 可深度流水;
  AscendC 用 Pipe API 显式控制;UB 256KB 是优化的\"主战场\";Cube 16x16x16
  是 GEMM 最小粒度, vector 是 4096-bit 宽。
【进阶】读 CANN 文档的 \"Tiling 最佳实践\";自己写一个 elementwise 算子
  用 AscendC 测 UB 复用 vs 不复用的性能差。
EOF
echo "############################################################"
