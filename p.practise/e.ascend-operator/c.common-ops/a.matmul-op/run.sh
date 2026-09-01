#!/bin/bash
# ============================================================
# 实验: a.matmul-op
# 说明: 矩阵乘算子、Cube、tiling
# 模块: p.practise 第一性原理学习实验
# ============================================================
# 【第一性原理】
# 矩阵乘 C = A @ B 是 LLM 80%+ 算力消耗的算子。
# 昇腾实现 = Cube 单元 (16x16x16 FP16 GEMM 单元)。
# 关键: 数据排布 (Layout) + 精度 + 算力利用率。
# 性能指标:
#   - Cube 利用率: 实际 FLOPS / 峰值 FLOPS
#   - 内存带宽: 实际 GB/s / 峰值 GB/s
#   - MTE 利用率: 搬运占计算的比例
# 目标: M=N=K=4096 FP16 GEMM 达到 70%+ Cube 利用率
# ============================================================
set -euo pipefail
if ! python3 -c "import numpy" 2>/dev/null; then python3 -m pip install --quiet numpy; fi
echo "############################################################"
echo "# 实验: a.matmul-op | 矩阵乘: Cube 单元 + Tiling + Layout"
echo "############################################################"

python3 <<'PYEOF'
import numpy as np, time
def hdr(n,total,t): print(f"\n{'='*60}\n【步骤 {n}/{total}】{t}\n{'='*60}")
def why(s): print("\n--- 做什么 & 为什么 ---\n  "+s.replace("\n","\n  "))
def res(s): print("\n--- 结果 ---\n  "+s.replace("\n","\n  "))
def mea(s): print("\n--- 解读 ---\n  "+s.replace("\n","\n  "))
TOTAL = 4

# --- 1. GEMM 算力 vs 访存 ---
hdr(1,TOTAL,"GEMM:算力 vs 访存下界")
why("""M=N=K=Nn, FP16:
  FLOPS = 2*N^3
  IO    = 3*N^2*2 (读 A,B, 写 C)
  算子强度 I = 2*N^3 / 6*N^2 = N/3
  N=4096 → I = 1365 FLOPS/Byte  (算力密集!)
  N=128  → I = 43  FLOPS/Byte   (访存密集)
  临界 N ≈ 312 (A2 拐点 156)""")
N_list = [128, 512, 2048, 4096]
out = ["  N    FLOPS         IO         I(FLOPS/Byte)    主导"]
for N in N_list:
    flops = 2*N**3
    io = 3*N*N*2
    I = flops/io
    dom = "算力" if I > 156 else "访存"
    out.append(f"  {N:4d}  {flops/1e9:.1f} G     {io/1024/1024:.1f} MB    {I:8.1f}         {dom}")
res("\n".join(out))
mea("""LLM 训练/推理的 GEMM 通常 N>=2048, 算力密集。
  小 N (如 batch=1 推理) 时 GEMM 反而是访存密集, 需要特别优化。""")

# --- 2. 实测 toy matmul ---
hdr(2,TOTAL,"CPU 模拟:GEMM 时间")
why("""M=N=K=2048, FP32, CPU 实测:""")
A = np.random.randn(2048, 2048).astype(np.float32)
B = np.random.randn(2048, 2048).astype(np.float32)
# warmup
C = A @ B
t = time.perf_counter()
for _ in range(3): C = A @ B
dt = (time.perf_counter() - t) / 3
flops = 2*2048**3
gflops = flops/dt/1e9
res(f"""CPU 2048x2048x2048 FP32 GEMM:
  耗时:    {dt*1000:.1f} ms
  算力:    {gflops:.1f} GFLOPS
  理论算力 (A2): 280 TFLOPS FP16 / 56 TFLOPS FP32
  → CPU 远低于 NPU""")
mea("CPU 是 8-10 GFLOPS,NPU 是 280 TFLOPS,差 30000×。\n  同样的算法,NPU 上比 CPU 快 30000×,这就是用 NPU 的意义。")

# --- 3. Layout 影响 ---
hdr(3,TOTAL,"数据排布:ND / NZ / zN")
why("""昇腾有 3 种主要数据排布:
  - ND (NCHW 默认): 行连续, 适合通用
  - NZ (16-align): 16 元素对齐, 适合 Cube (一次 16x16 读)
  - zN: NZ 的特例
  Cube 单元一次读 16x16 块, NZ 排布让读取更高效。""")
out = ["  Layout    读 16x16 块  适用场景"]
out.append("  ND        需要 scatter   通用 / 小算子")
out.append("  NZ        一次 16 连续  Cube GEMM (matmul)")
out.append("  zN        与 NZ 类似    Conv2D NCHW 输入")
res("\n".join(out))
mea("""最佳实践:
  - GEMM 输入用 NZ 排布
  - 其他算子用 ND
  - 框架 (torch_npu) 自动选择, 一般不用关心
  - 自研算子时手动指定 layout""")

# --- 4. Tiling + L1 复用 ---
hdr(4,TOTAL,"Tiling 优化:M × N × K 维切分")
why("""Cube 16x16x16 单元 = 最小算力粒度。
  实操: tile (128, 128) 输出 + K 维 32 步累加
  = 4 个 16x16 GEMM 在 N 维 + 8 个 K 步
  = 4*8 = 32 次 Cube 调用,每次 16x16x16""")
res("""tile 128x128x32 GEMM:
  N 维: 128 / 16 = 8 次 16x16 调用
  M 维: 128 / 16 = 8 次 16x16 调用
  K 维: 32 / 16 = 2 次累加
  总 Cube 调用: 8*8*2 = 128 次
  每次 16x16x16 = 8192 FLOPS
  总 FLOPS: 128 * 8192 = 1.05 M FLOPS  ≈ 2*128^3/1e6 = 4.2 M FLOPS ✓""")
mea("""Tiling 经验:
  1. tile 128x128 K=32 是稳的 sweet spot
  2. 大矩阵 (>=4096) 用 256x128 或 128x256
  3. 小矩阵 (< 256) 用 64x64 tile, 否则浪费
  4. K 维尽量大, 减少累加次数, 但要装得下 L1""")
PYEOF

echo ""
echo "############################################################"
cat <<'EOF'
【整体总结】
- 小白:矩阵乘是 LLM 80%+ 算力,昇腾用 Cube 单元算;Tile 切 128x128 K=32
  是稳的 sweet spot;数据排布 NZ 让 Cube 一次读 16x16 块,效率最高。
- 熟手:GEMM 是算力密集(N>=2048);Layout: ND 通用, NZ 给 Cube, zN 给 Conv;
  Cube 16x16x16 是最小粒度;大矩阵用 256x128 tile 提并行;msprof 看 Cube 利用
  率 70%+ 算优秀。
【进阶】用 AscendC 写一个 matmul 算子,从 64x64 tile 起步,试 128x128,256x128,
  对比 msprof 报告的 Cube 利用率。
EOF
echo "############################################################"
