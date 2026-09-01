#!/bin/bash
# ============================================================
# 实验: c.memory-optimize
# 说明: UB 复用、双缓冲、bank conflict
# 模块: p.practise 第一性原理学习实验
# ============================================================
# 【第一性原理】
# UB 256 KB 是算子最珍贵的资源, 三大问题:
#   1. 容量: 数据装不下 → 复用
#   2. 复用: 多个算子共享 UB → 复用策略
#   3. bank conflict: 多线程访同一 bank 冲突 → padding
# 优化技术:
#   - 双缓冲: 2 个 UB 区, 一个算一个搬
#   - 内存池: 申请/释放, 避免反复 alloc
#   - 数据复用: A 块在多个 N 块算中复用
#   - Bank conflict: padding 让访存错开
# ============================================================
set -euo pipefail
echo "############################################################"
echo "# 实验: c.memory-optimize | UB 复用 + 双缓冲 + bank conflict"
echo "############################################################"

python3 <<'PYEOF'
def hdr(n,total,t): print(f"\n{'='*60}\n【步骤 {n}/{total}】{t}\n{'='*60}")
def why(s): print("\n--- 做什么 & 为什么 ---\n  "+s.replace("\n","\n  "))
def res(s): print("\n--- 结果 ---\n  "+s.replace("\n","\n  "))
def mea(s): print("\n--- 解读 ---\n  "+s.replace("\n","\n  "))
TOTAL = 4

# --- 1. UB 容量限制 ---
hdr(1,TOTAL,"UB 容量 256 KB:能装多少")
why("""UB 是算子片上最快的存储, 但容量仅 256 KB。
  FP16 (2B): 256/2 = 128K 元素
  FP32 (4B): 256/4 = 64K 元素
  INT8 (1B): 256/1 = 256K 元素
  1024x1024 FP16 矩阵 = 2 MB, 远超 256 KB → 必须 tiling""")
res("""UB 能装的数据(单 tile):
  格式      大小     例子 (单 tile)
  FP16 32x32   2 KB     极小算子
  FP16 64x64   8 KB     小算子
  FP16 128x128 32 KB   标准 matmul
  FP16 128x256 64 KB   大 matmul
  FP32 128x128 64 KB   累加器
  总上限       256 KB    含双缓冲""")
mea("""经验: 128x128 FP16 tile + 128x128 FP32 累加器 ≈ 96 KB
  双缓冲: 96 * 2 = 192 KB  ← 仍装得下
  3 缓冲: 96 * 3 = 288 KB  ← OOM!
  双缓冲是甜点。""")

# --- 2. 双缓冲 ---
hdr(2,TOTAL,"双缓冲:MTE 与 Compute 并行")
why("""双缓冲 = 2 个 UB 区域, A 算时 MTE 搬 B, 切换后 B 算时 MTE 搬 A。
  无双缓冲: 搬 10 + 算 10 = 20 us
  双缓冲:   搬 10 ‖ 算 10 = 10 us (50% 节省)""")
out = ["  状态           单 tile 耗时  100 tile 总耗时  加速"]
out.append("  无缓冲         20 us        2000 us         1.0×")
out.append("  双缓冲         10 us        1000 us         2.0× (理论)")
out.append("  3 缓冲         8 us         800 us          2.5× (理论)")
out.append("  实际双缓冲     11 us        1100 us         1.8× (有同步开销)")
out.append("  实际 3 缓冲    10 us        1000 us         2.0× (要更多 UB)")
res("\n".join(out))
mea("""双缓冲实现 (AscendC 伪代码):
  LocalTensor<A_T> bufA[TILE_SIZE];
  LocalTensor<A_T> bufB[TILE_SIZE];
  DataCopy(bufB, gm[0], TILE);    // 先搬 B
  for (int i = 0; i < N; i++) {
    if (i+1 < N) DataCopy(bufA, gm[(i+1)*TILE], TILE);  // 预取下一个
    Compute(bufB);  // 算当前
    swap(bufA, bufB);
  }""")

# --- 3. 数据复用 ---
hdr(3,TOTAL,"数据复用:同一块数据算多次")
why("""复用 = 读 1 次, 用多次。
  例子: A[i, k] 在 K 维上被多个 C[i, n] 算用到 → 一次读, 多次用
  K 维越大, 复用越多。
  实际上 K 维所有 N 维输出共享 A[i, k] 一次读 → N 倍复用""")
out = ["  复用维度       复用倍数 (理论)     说明"]
out.append("  K 维复用       tile_N 倍          A 块在 N 维输出复用")
out.append("  M 维复用       tile_K 倍          B 块在 M 维输出复用")
out.append("  N 维复用       1 (无复用)         C 是输出, 不复用")
out.append("  双缓冲复用     2×                 切到 2 个 buffer")
out.append("  实际综合       tile_N × tile_K    (tile_N 是 N 维 tile 数)")
res("\n".join(out))
mea("""复用越多, IO 越少, 算力利用率越高。
  例: 128x128x32 tile
    A 块: 1 次读, 复用 N/128 次 (N=4096 → 32 次)
    B 块: 1 次读, 复用 M/128 次 (M=4096 → 32 次)
  这是 128x128 tile 性能好的核心原因。""")

# --- 4. Bank conflict ---
hdr(4,TOTAL,"Bank conflict:Vector 访存冲突")
why("""UB 分多个 bank (类似 GPU shared memory)。
  多个 vector 单元同时访同一 bank → 串行, 慢。
  解决: padding 让访存地址错开""")
out = ["  模式              时间   说明"]
out.append("  无 conflict       1×     各 vector 访不同 bank")
out.append("  全 conflict       8-16×  所有 vector 访同 bank")
out.append("  解决: padding     1×     padding 后访不同 bank")
out.append("")
out.append("  例: A 矩阵 128x128, 每行 128 元素, 32 bank")
out.append("    同一列不同行同时读 → 同一 bank!")
out.append("    解决: 每行加 1 元素 padding → 129 元素宽, 错开 bank")
res("\n".join(out))
mea("""Bank conflict 检测:
  - msprof 看 Vector pipe stall cycles 高
  - 解决: 加 padding (e.g. UB_SHAPE = (128, 129))
  - 8 元素对齐 32 字节, 也防 conflict
  
  AscendC 默认 TileShape 已考虑, 自研时注意""")
PYEOF

echo ""
echo "############################################################"
cat <<'EOF'
【整体总结】
- 小白:UB 仅 256 KB, 必须 tiling + 复用;双缓冲让搬运和计算并行, 加速 1.8-2×;
  bank conflict 是 vector 单元同时访同 bank 变慢, padding 解决。
- 熟手:128x128 FP16 tile + 128x128 FP32 累加器 ≈ 96 KB, 双缓冲装得下;
  K 维越大复用越多, 128x128x32 是 sweet spot;msprof 看 Vector stall cycles
  判断 bank conflict;padding 128 元素 → 129 元素宽防 conflict。
【进阶】用 AscendC 写一个 128x128x32 GEMM, 测无/有双缓冲、padding 前后
  的 msprof 指标变化。
EOF
echo "############################################################"
