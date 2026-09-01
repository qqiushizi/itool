#!/bin/bash
# ============================================================
# 实验: b.kernel-tiling
# 说明: tiling 切分策略、数据搬运
# 模块: p.practise 第一性原理学习实验
# ============================================================
# 【第一性原理】
# Tiling = 把大问题切成小块, 每块在片上 (UB/L1) 算。
#   为什么必须 tiling:
#     - UB 仅 256 KB, 1 GB 数据装不下
#     - 数据\"搬到片上, 复用\"是性能来源
#   4 维 tiling 决策:
#     1. M 维切多大 (输出行)
#     2. N 维切多大 (输出列)
#     3. K 维切多大 (归约深度)
#     4. 边界处理 (整除 + 残块)
#   Tiling 性能影响:
#     - 越大 → 数据复用越多 → IO 越少
#     - 越大 → UB 占用越多 → 块越少 → 并行度低
#     - 太大溢出 OOM
#   sweet spot: 128x128x32 (FP16, GEMM)
# ============================================================
set -euo pipefail
echo "############################################################"
echo "# 实验: b.kernel-tiling | Tiling 切分 + 数据搬运"
echo "############################################################"

python3 <<'PYEOF'
def hdr(n,total,t): print(f"\n{'='*60}\n【步骤 {n}/{total}】{t}\n{'='*60}")
def why(s): print("\n--- 做什么 & 为什么 ---\n  "+s.replace("\n","\n  "))
def res(s): print("\n--- 结果 ---\n  "+s.replace("\n","\n  "))
def mea(s): print("\n--- 解读 ---\n  "+s.replace("\n","\n  "))
TOTAL = 4

# --- 1. Tiling 必要性 ---
hdr(1,TOTAL,"为什么必须 tiling")
why("""HBM 80 GB, UB 256 KB = 312500 倍差距。
  1 GB 矩阵分元素搬运 = 250M 次 DataCopy, 不可接受。
  Tiling: 切 128x128 的小块, 每块在 UB 完成。
  切 1 GB 矩阵 (25000 个 tile) 比 不切 (250M 次) 快 10000×""")
res("""GEMM 1024x1024, FP16:
  不切:  一次算 1024x1024 矩阵乘, 中间结果要 1 GB 内存, UB 装不下
  切 128: 切 8x8 = 64 个 tile, 每块 16KB, UB 装得下
  IO:     切块:  64 * (128*128*2 + 128*128*2) = 4 MB IO
          不切: 1 GB IO
          减少 250×""")
mea("Tiling 是算子开发的\"门槛\", 不切没法跑, 切得好跑得快。")

# --- 2. Tiling 大小选择 ---
hdr(2,TOTAL,"Tile 大小:算力/IO/并行的折中")
why("""设 tile 大小为 m1*n1, K 维为 k1, FP16:
  A tile: m1*k1*2 bytes
  B tile: k1*n1*2 bytes
  C tile: m1*n1*2 bytes
  累加器: m1*n1*4 bytes
  合计: (m1*k1 + k1*n1 + m1*n1) * 2 + m1*n1*4 ≤ 256 KB""")
# 计算最大 tile
for m1, n1, k1 in [(64, 64, 32), (128, 128, 32), (128, 128, 64), (256, 128, 32), (128, 256, 32)]:
    a_tile = m1*k1*2
    b_tile = k1*n1*2
    c_tile = m1*n1*2
    acc = m1*n1*4
    total = a_tile + b_tile + c_tile + acc
    ok = "✓" if total <= 256*1024 else "✗ OOM"
    print(f"  {m1}x{n1}x{k1}: A={a_tile/1024:.1f}KB B={b_tile/1024:.1f}KB C={c_tile/1024:.1f}KB Acc={acc/1024:.1f}KB  合计={total/1024:.1f}KB {ok}")
res("""Tile 大小       A tile   B tile   C tile   Acc     合计     是否能装下
  64x64x32       4 KB     4 KB     8 KB     16 KB   32 KB    ✓
  128x128x32     8 KB     8 KB     32 KB    64 KB   112 KB   ✓
  128x128x64     16 KB    16 KB    32 KB    64 KB   128 KB   ✓
  256x128x32     16 KB    8 KB     64 KB    128 KB  216 KB   ✓
  128x256x32     8 KB     16 KB    64 KB    128 KB  216 KB   ✓""")
mea("""128x128x32 是常用 sweet spot:
  - 装得下, 余 144 KB 给双缓冲 + 临时
  - K=32 平衡 (K 小并行度高, K 大复用多)
  - 大数据集用 256x128""")

# --- 3. 边界 tile ---
hdr(3,TOTAL,"边界 tile:padding")
why("""问题矩阵 M=N=1023, tile=128:
  1023 / 128 = 7 余 127
  边界 tile 大小 127, 装不满 tile 大小
  解决: padding 到 128, 计算时只写 127 列
  或: 单独处理边界 (代码复杂, 性能可能更好)""")
res("""M=N=1023, tile=128:
  完整 tile: 7 × 7 = 49 个
  边界 tile: 边界 tile 个数 = (M%128>0) + (N%128>0) + (边界都 > 0) = 1+1+1=3
  实际 block 数: 7*7 + 边界修正 = 64 个
  边界用 padding 简化代码""")
mea("""实现:
  - AscendC 提供 DataCopyPad 自动 padding
  - 显式控制: tile 内只算到 min(M, (block_idx+1)*128) 终止
  - 边界处理 = 性能与简洁的权衡""")

# --- 4. Tiling 性能对比 ---
hdr(4,TOTAL,"Tiling 大小 vs 性能(经验)")
why("""M=N=K=4096 GEMM, A2 上不同 tile 大小耗时(估):""")
res("""Tile 大小       耗时 (us)   加速比    备注
  32x32x32        220         1.0×     baseline
  64x64x32        130         1.7×     复用增加
  128x128x32      95          2.3×     平衡点
  128x128x64      78          2.8×     复用更高
  256x128x32      75          2.9×     极致
  128x256x32      75          2.9×     同上
  256x256x32      OOM         -        UB 溢出""")
mea("""选择流程:
  1. 先算 UB 上限, 排除 OOM
  2. 128x128 起步
  3. msprof 测实际耗时
  4. 试 256x128 / 128x256 看是否更快
  5. 双缓冲需要 UB 再翻倍, 留余量
  6. 极致 (T3) 用 4 缓冲""")
PYEOF

echo ""
echo "############################################################"
cat <<'EOF'
【整体总结】
- 小白:Tiling = 把大问题切小块, 让数据装得下片上 (UB);Tile 越大复用越多
  但并行度低;128x128x32 是常用 sweet spot;边界用 padding 简化。
- 熟手:Tile 选 128x128 起步,profile 后试 256x128/128x256;双缓冲要 UB
  翻倍;边界用 DataCopyPad 自动 padding;msprof 看实际 tile 效果。
【进阶】自己写一个简单 matmul 算子, 试 64x64/128x128/256x128 三种 tile
  大小, profile 对比耗时。
EOF
echo "############################################################"
