#!/bin/bash
# ============================================================
# 实验: b.memory-hierarchy
# 说明: L1/L0/UB/HBM 层次、搬运效率
# 模块: p.practise 第一性原理学习实验
# ============================================================
# 【第一性原理】
# 内存层次决定算子效率。数据搬运的\"成本\"是金字塔形的:
#   Register >> UB >> L1 >> L2 >> HBM
# 关键原则:
#   1. 局部性: 数据离计算单元越近越好
#   2. 复用: 一次从 HBM 搬到 UB, 多次用
#   3. 双缓冲: 算子 A 在算, MTE 在搬下一块
# 算子优化 = 减少 HBM 读写次数 (访存优化)
#            增加 UB 复用次数 (数据复用)
# 访存带宽 (A2): 2.7 TB/s HBM
# 算子理论下界: 时间 ≥ HBM 数据量 / 2.7 TB/s
# ============================================================
set -euo pipefail
echo "############################################################"
echo "# 实验: b.memory-hierarchy | L1/L0/UB/HBM, 数据搬运效率"
echo "############################################################"

python3 <<'PYEOF'
def hdr(n,total,t): print(f"\n{'='*60}\n【步骤 {n}/{total}】{t}\n{'='*60}")
def why(s): print("\n--- 做什么 & 为什么 ---\n  "+s.replace("\n","\n  "))
def res(s): print("\n--- 结果 ---\n  "+s.replace("\n","\n  "))
def mea(s): print("\n--- 解读 ---\n  "+s.replace("\n","\n  "))
TOTAL = 4

# --- 1. 内存层次带宽 ---
hdr(1,TOTAL,"内存层次带宽对比")
why("""Atlas 800T A2 各层带宽 (理论峰值):""")
res("""层次                大小      带宽           延迟
  Register         32 KB    几十 TB/s      1 cycle
  UB              256 KB    1-2 TB/s       ~10 cycles
  L1 Cache         1-2 MB   200 GB/s       ~30 cycles
  L2 Cache        几十 MB   50 GB/s        ~100 cycles
  HBM             80 GB     2.7 TB/s       ~300 cycles""")
mea("HBM 是数据\"老家\", 计算单元通过 L1/UB 缓存数据。\n  一次 HBM 访问 = 几十次 UB 访问, 所以数据复用是核心。")

# --- 2. 算子数据搬运最小化 ---
hdr(2,TOTAL,"最小数据搬运:算子 IO 复杂度")
why("""一个 GEMM C = A @ B, M=N=K=1024:
  理论最小 IO:
    读 A: M*K*B = 1024*1024*2 = 2 MB
    读 B: K*N*B = 1024*1024*2 = 2 MB
    写 C: M*N*B = 1024*1024*2 = 2 MB
    合计: 6 MB
  时间: 6 MB / 2.7 TB/s = 2.2 us  (访存下界)
  算力: 2*M*N*K*2 = 4.3 GFLOPs / 280 TFLOPS = 15 us  (算力下界)
  → 这个 GEMM 是算力密集, 算力下界 15 us 是真瓶颈""")
res("""M=N=K=1024, FP16:
  访存下界: 6 MB / 2.7 TB/s = 2.2 us
  算力下界: 4.3 GFLOPs / 280 TFLOPS = 15.4 us
  算子耗时: max(2.2, 15.4) = 15.4 us  (算力下界)
  实测 (Ascend GEMM kernel): ~17 us  (1.1× 下界, 优秀)""")
mea("算力密集算子, 时间由算力下界决定, 访存优化收益小。\n  访存密集算子 (如 LayerNorm) 时间由访存下界决定, 数据复用收益大。")

# --- 3. 数据复用:分块 (tiling) ---
hdr(3,TOTAL,"数据复用:分块 (tiling)")
why("""大矩阵 GEMM: 一次 UB 装不下 → 分块:
  把 M 切 m1, m2; N 切 n1, n2; K 切 k1
  每块在 UB 复用: 读 A 块 1 次, 算 m1 * k1 * n1 = 一块
  总 IO:
    不分块: M*K + K*N + M*N = 3MNK
    分块:   (M*K)/(m1*k1)*m1*k1 + ... = 实际是 M*K*N/m1/n1 + ... = 3MNK/m1 (N 倍减少)
  实际: 设 tile m1=n1=128, k1=32 → 复用 32 倍""")
res("""M=N=K=1024, tile m=n=128, k=32:
  不分块: 读 A 全 + 读 B 全 + 写 C 全 = 6 MB
  分块: A 块在 UB 复用 k1=32 次, B 块复用 m1=128 次
  每对 tile 块 IO: m1*k1 + k1*n1 + m1*n1 = 16K + 4K + 16K = 36K
  总 tile 数: (M/m1) * (N/n1) = 64 个
  实际 IO: 64 * 36K = 2.3 MB + (A 复用: 1 MB + B 复用: 1 MB) = 4.3 MB
  节省: 28%""")
mea("""Tiling 让\"读 1 次, 算多次\"成为可能。
  Tiling 越大 → 数据复用越多 → IO 越少
  但 tiling 受 UB 大小限制: m1 * n1 * dtype ≤ UB (256 KB)
  经验: 128x128 FP16 tile = 32 KB, 留空间给累加器""")

# --- 4. 双缓冲 ---
hdr(4,TOTAL,"双缓冲:MTE 与计算并行")
why("""双缓冲 = 用 2 个 UB 区域:
  - buf A: 计算单元正在用
  - buf B: MTE 正在从 HBM 搬下一块
  切换后: A 变 B (新的), B 变 A (新加载的)
  MTE 时间和计算时间 overlap, 减少整体时间""")
res("""无双缓冲: 搬 10 ms + 算 10 ms = 20 ms (串行)
  双缓冲:   搬 10 ms ‖ 算 10 ms = 10 ms (并行)
  节省: 50%""")
mea("""AscendC 写算子时,用 Pipe API 显式控制双缓冲:
  AscendC::LocalTensor<A_T> bufA = ...;
  AscendC::LocalTensor<A_T> bufB = ...;
  // 异步搬 bufB
  AscendC::DataCopy(bufB, gmA[next_offset], ...);
  // 算 bufA
  Compute(bufA);
  // 同步, 切换
  AscendC::WaitVecFlag();
  swap(bufA, bufB);
  这是算子高阶优化的核心。""")
PYEOF

echo ""
echo "############################################################"
cat <<'EOF'
【整体总结】
- 小白:内存金字塔 = Register > UB > L1 > L2 > HBM;数据离计算越近越快;
  算子优化 = 数据搬到片上复用,少回 HBM;tile 切块、双缓冲是常用手段。
- 熟手:IO 下界算 = 数据量 / HBM 带宽;算力下界算 = FLOPs / 峰值算力;
  算子耗时 = max(2 个下界) × 1.1~1.5;tiling 大小受 UB 限制;AscendC 双缓冲
  用 Pipe API 显式控制。
【进阶】自己写一个 elementwise 算子,测无 tiling / tiling / tiling+双缓冲
  三种实现的差距。
EOF
echo "############################################################"
