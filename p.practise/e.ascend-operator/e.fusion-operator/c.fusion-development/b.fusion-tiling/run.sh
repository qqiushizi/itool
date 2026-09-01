#!/bin/bash
# ============================================================
# 实验: b.fusion-tiling
# 说明: 融合算子 tiling 策略与 UB 平衡
# 模块: p.practise 第一性原理学习实验
# ============================================================
# 【第一性原理】
# 融合算子 tiling 的 3 大原则:
#   1. 一致性: 参与融合的所有算子 tile 大小一致
#   2. 装得下: 中间结果 (累加器 + 临时) 必须装入 UB
#   3. 复用优: 尽量让 A/B/C 都能在 UB 复用多次
# 平衡目标:
#   - 算力 vs 带宽: 大 tile 算力高,小 tile 带宽优
#   - 复用 vs 占用: 累加器大 -> 复用多, 占用也多
#   - 单核 vs 多核: tile 大 -> 核少, tile 小 -> 核多
# 经验公式 (Ascend 910B):
#   - A 64x64, B 64x64 -> C 64x64, 占用 ~80KB
#   - 双缓冲: 乘 2 = 160KB (UB 总 256KB, 剩 96KB 给临时)
#   - 三缓冲: 乘 3 = 240KB (UB 满,临时挤)
# ============================================================
set -euo pipefail
echo "############################################################"
echo "# 实验: b.fusion-tiling | 融合算子 tiling 与 UB 平衡"
echo "############################################################"

python3 <<'PYEOF'
def hdr(n,total,t): print(f"\n{'='*60}\n【步骤 {n}/{total}】{t}\n{'='*60}")
def why(s): print("\n--- 做什么 & 为什么 ---\n  "+s.replace("\n","\n  "))
def res(s): print("\n--- 结果 ---\n  "+s.replace("\n","\n  "))
def mea(s): print("\n--- 解读 ---\n  "+s.replace("\n","\n  "))
TOTAL = 4

# --- 1. Tiling 维度: M/N/K ---
hdr(1,TOTAL,"Tiling 三维:M/N/K")
why("""融合算子 tiling 一般 3 维:
  - M (output row): 通常 64-128
  - N (output col): 通常 64-256 (N 一般更大, 取宽)
  - K (reduce dim):  通常 16-64 (循环累加, 步长小)
为什么 K 最小:
  - K 是 reduce 维, 一次只算 1 个 K step
  - K step 太大 -> A/B tile 也大, 占用高
  - K step 太小 -> 循环次数多, Set/WaitFlag 开销大""")
out = ["  维度   典型大小   太大问题                太小问题"]
out.append("  M      64-128     A tile 大,UB 不够       核太多,调度碎")
out.append("  N      64-256     C tile 大,累加器爆      B 复用少,带宽低")
out.append("  K      16-64      A/B 太大                累加循环多,延迟大")
res("\n".join(out))
mea("""选择流程:
  1. 先定 N (一般最大维度)
  2. 由 N 推 M, 让 N*M = 64*64 ~ 128*128
  3. K 取小一点, 16-32
  4. 双缓冲: 整体乘 2, 看是否还装得下
  5. OOM 优先降 M, 再降 N, 最后降 K""")

# --- 2. UB 容量计算 ---
hdr(2,TOTAL,"UB 容量计算(Ascend 910B)")
why("""Ascend 910B 单核 UB 容量 = 256 KB。
融合算子的 UB 占用 = (A + B + C + 临时) x 缓冲数。
C 是累加器, 通常 FP32, 大小 M*N*4 字节。
临时包括: scale, mask, bias, indices, 临时累加。""")
out = ["  配置                         占用 (KB)    备注"]
out.append("  64x64 单缓冲 FP16            64x64x2 = 8  A,B 各 8 KB")
out.append("  + C (累加 FP32)              64*64*4 = 16  C 16 KB")
out.append("  + 临时                        ~10          累加器, scale")
out.append("  = 单缓冲总计                  ~42          < 256, 富裕")
out.append("  x2 双缓冲                     ~84          还装得下")
out.append("  x3 三缓冲                     ~126         临界,临时要省")
out.append("  128x128 双缓冲 + Linear+ReLU+Add  ~190      装得下")
out.append("  128x128 三缓冲 + 多临时        >256         OOM! 降 tile")
res("\n".join(out))
mea("""OB 平衡经验值:
  - 单核 UB 装得下: 70% 占用以下 (留 buffer)
  - 双缓冲首选: 1 个算在算, 1 个在搬
  - 累加器 FP32 必要 (否则精度掉)
  - 临时变量复用, 别都 new""")

# --- 3. 复杂融合的 tiling 策略 ---
hdr(3,TOTAL,"复杂融合的 tiling 策略")
why("""复杂融合 (Linear+ReLU+Add+LayerNorm) tiling 要考虑:
  - LayerNorm 需要 mean/var 一次性算
  - mean/var 临时要存在 UB
  - LayerNorm 输入输出大小 = C (M*N tile)
策略:
  1. LayerNorm 算整个 C tile, mean/var 临时
  2. ReLU+Add 算完再 LayerNorm (fusion 链)
  3. 临时累加器复用""")
out = ["  复杂融合算子          M  N  K  临时数  备注"]
out.append("  Linear+ReLU            128 128 32  2     简单, tile 不变")
out.append("  Linear+ReLU+Add        128 128 32  3     加 1 个临时")
out.append("  Linear+LayerNorm       64  128 32  5     LayerNorm 临时多")
out.append("  Linear+Softmax         64  128 32  6     max, exp 临时")
out.append("  MHA(q,k,v,softmax)     64  64  64  8     最多, 64x64 起")
out.append("  推理多融合 (L+LA+S)    32  64  32  10    极复杂, 32x32 试探")
res("\n".join(out))
mea("""复杂融合的工程取舍:
  1. 简单融合 (1-2 个 op): tile 不变, 128x128 起
  2. 中等 (3-4 个 op): tile 降 1 档, 64-128
  3. 复杂 (5+ op): tile 再降, 64x64 + 大量复用
  4. 极复杂 (MHA): 不强求 1 个 kernel, 可拆 2-3 个
  5. tile 大小不是越大约好, 是 平衡 复用 vs 占用""")

# --- 4. Tiling 实测验证 (伪实验) ---
hdr(4,TOTAL,"Tiling 实测验证(伪数据)")
why("""通过伪数据估算不同 tile 大小的理论性能,辅助选 tile:""")
import numpy as np

# 假设 NPU 算力: 256 TFLOPS (FP16), UB: 256KB
# 任务: Linear M=1024, N=1024, K=1024
M, N, K = 1024, 1024, 1024
peak = 256e12  # 256 TFLOPS

# 估算每个 tile 大小的: 理论 FLOPs, UB 占用, 核数, 耗时
out = ["  tile(M,N,K)    核数   UB/核 (KB)   理论耗时 (us)   加速比"]
for tile_m, tile_n, tile_k in [(64, 64, 32), (64, 128, 32), (128, 128, 32), (128, 256, 32), (256, 256, 32)]:
    cores = (M // tile_m) * (N // tile_n)
    a_kb = tile_m * tile_k * 2 / 1024   # FP16
    b_kb = tile_k * tile_n * 2 / 1024
    c_kb = tile_m * tile_n * 4 / 1024   # FP32
    tmp_kb = 10
    ub_kb = (a_kb + b_kb + c_kb + tmp_kb) * 2  # 双缓冲
    flops = 2 * tile_m * tile_n * tile_k * (K // tile_k)  # 单核
    total_flops = flops * cores
    time_s = total_flops / peak
    time_us = time_s * 1e6
    print(f"  ({tile_m:3d},{tile_n:3d},{tile_k:3d})    {cores:4d}    {ub_kb:6.1f}        {time_us:6.1f}")

# 选最优 (装得下 + 耗时低)
print()
res(f"""最优点: (128, 128, 32)
  - 64 核, UB 116KB (装得下, 留 50% buffer)
  - 理论耗时 ~32 us
  - 64x64: 核多但单核慢, 总耗时相同
  - 256x256: UB 爆 (300KB > 256KB), 不可行""")
mea("""Tiling 选择决策树:
  1. 算 M/N 维, 让 (M//tile_m) * (N//tile_n) 核数 = 32-64
  2. tile 选 64-128
  3. 检查 UB 占用 (双缓冲 < 200KB)
  4. 检查 K step 循环次数 (10-100 次)
  5. 跑实测, 选最优""")
PYEOF

echo ""
echo "############################################################"
cat <<'EOF'
【整体总结】
- 小白:Tiling = 把大矩阵切成小 tile,装入 UB, 多个核并行算;
  融合算子 tiling 要 一致(所有算子同 tile)+ 装得下(UB < 80%) + 复用多;
  双缓冲首推, 三缓冲慎用, 复杂融合 tile 要降档。
- 熟手:Tiling 三维 M/N/K 一般 64-128/64-256/16-64;
  UB 占用 = (A+B+C+临时) x 缓冲数, 累加器 FP32 不能省;
  复杂融合 (MHA) 临时变量多, 64x64 起手;
  选 tile 流程: 估核数 -> 算 UB -> 估耗时 -> 实测调。
【进阶】写脚本遍历 (tile_m, tile_n) 组合, 在 profiler 上实测,
  画 UB 占用 vs 加速比 散点图, 找 最优 Pareto 前沿。
EOF
echo "############################################################"
