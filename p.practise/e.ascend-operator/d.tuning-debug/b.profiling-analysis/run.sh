#!/bin/bash
# ============================================================
# 实验: b.profiling-analysis
# 说明: profiling 分析方法:时间轴解读、算子耗时排序、Cube/Vector 占比、流水气泡、瓶颈定位决策树
# 模块: p.practise 第一性原理学习实验
# ============================================================
# 【第一性原理】
# Profiling 数据本身没价值, 解读才有价值。
# 解读 4 步:
#   1. 找 top-3 耗时 kernel
#   2. 看每个 kernel 的 Pipe 占比, 判断是算力/访存/同步瓶颈
#   3. 看时间轴的 bubble (空闲时间) 大小
#   4. 决策: 改算法 / 改 tiling / 加融合 / 加并行
# 决策树: 算法 → tiling → 双缓冲 → 融合 → 跨核优化
# ============================================================
set -euo pipefail
echo "############################################################"
echo "# 实验: b.profiling-analysis | 时间轴解读 + 瓶颈决策树"
echo "############################################################"

python3 <<'PYEOF'
def hdr(n,total,t): print(f"\n{'='*60}\n【步骤 {n}/{total}】{t}\n{'='*60}")
def why(s): print("\n--- 做什么 & 为什么 ---\n  "+s.replace("\n","\n  "))
def res(s): print("\n--- 结果 ---\n  "+s.replace("\n","\n  "))
def mea(s): print("\n--- 解读 ---\n  "+s.replace("\n","\n  "))
TOTAL = 4

# --- 1. 时间轴解读 ---
hdr(1,TOTAL,"时间轴:看懂 4 种\"段\"")
why("""msprof 时间轴上, 每个 kernel 内有 5 种段 (颜色):""")
out = ["  颜色       段              含义"]
out.append("  紫色       Cube             矩阵乘计算")
out.append("  绿色       Vector           元素级计算")
out.append("  黄色       MTE              数据搬运 (HBM ↔ UB)")
out.append("  蓝色       AI CPU           标量控制")
out.append("  白色       Stall            等待 (同步 stall)")
out.append("  灰色       Bubble           空闲 (无任务)")
res("\n".join(out))
mea("""看时间轴的 5 问:
  1. 哪段最长? = 优化目标
  2. 是否有大量白色 (stall)? = 同步有问题
  3. 是否有大量灰色 (bubble)? = 调度有 gap
  4. Cube 和 MTE 错开了吗? = 双缓冲是否生效
  5. Vector 是否和 Cube 并行? = 双发射是否生效""")

# --- 2. 算子耗时排序 ---
hdr(2,TOTAL,"Top-3 耗时算子分析模板")
why("""每个 LLM 推理 step 的算子耗时 (示例):""")
out = ["  排名  算子                  耗时(us)  占比   类型   优化方向"]
out.append("  1     matmul_qkv           120       25%    GEMM   已 Cube 80%, 接近极限")
out.append("  2     flash_attention      100       21%    访存   INT8 KV 减半")
out.append("  3     matmul_mlp_up         80       17%    GEMM   切 TP 或 INT8")
out.append("  4     matmul_mlp_down       80       17%    GEMM   切 TP 或 INT8")
out.append("  5     rmsnorm               30        6%    访存   融合到 GEMM")
out.append("  6     rope                  20        4%    访存   融合到 GEMM")
out.append("  7     silu                  15        3%    访存   融合到 GEMM")
out.append("  8     add (residual)        10        2%    访存   融合到 GEMM")
out.append("  9     其他                  20        4%    -      -")
out.append("  总                              475 us")
res("\n".join(out))
mea("""优化策略:
  1. matmul_qkv (25%): 已接近极限, 难优化
  2. flash_attention (21%): INT8 KV cache
  3. mlp_up/down (34%): 切 TP 或 INT8
  4. rmsnorm/rope/silu/add (15%): 全部融合 = 1 个 kernel
  5. 优化后总时间可能 350 us (-25%)""")

# --- 3. 流水气泡 ---
hdr(3,TOTAL,"流水气泡:4 类 bubble 来源")
why("""\"bubble\" = 算子内部空闲时间。常见 4 类:""")
out = ["  类型          原因                              优化"]
out.append("  1. 同步气泡  前后 Pipe 等待 (Set/WaitFlag)     检查依赖, 错开")
out.append("  2. 调度气泡  kernel launch 间隔                 CUDA Graph")
out.append("  3. 数据气泡  数据搬运与计算不重叠               双缓冲 / 多缓冲")
out.append("  4. 空闲气泡  UB 没充分利用, 算子太小           加大 tile / 合并 kernel")
res("\n".join(out))
mea("""识别 bubble:
  - 1. 同步气泡: msprof 看 Pipe stall cycles
  - 2. 调度气泡: timeline 间隙 (gray)
  - 3. 数据气泡: MTE 和 Compute 错开 (不是交叠)
  - 4. 空闲气泡: timeline 上 kernel 段 < 实际 tile 时间
  优化优先级: 3 > 1 > 2 > 4""")

# --- 4. 决策树 ---
hdr(4,TOTAL,"瓶颈定位决策树")
why("""按下面流程, 90% 瓶颈都能定位:""")
tree = """Q1: msprof 报告 kernel 占比最大? (top-3)
  ├── 算力密集 (Cube > 70%, MTE < 20%)
  │   → 算子已接近极限
  │   ├── tile 已最大?  → 切 TP / 切 batch / 换算法
  │   └── tile 可加大?  → 试 256x128 看 Cube 利用率
  ├── 访存密集 (MTE > 50%, Cube < 30%)
  │   → 数据搬运是瓶颈
  │   ├── 双缓冲已开?  → 试 3 缓冲
  │   ├── 可融合?       → 合并多个算子
  │   └── 数据复用?    → 加大 tile, 减少 IO
  ├── 同步密集 (Stall > 30%)
  │   → Pipe 等待
  │   ├── Set/WaitFlag 多?  → 检查依赖, 去掉冗余
  │   └── 跨核同步?        → 减少 cross-core
  └── 调度密集 (kernel 间隔大)
      → launch overhead
      ├── kernel 太多?  → 融合
      └── 走 CUDA Graph 减少 launch
"""
res(tree)
mea("""决策树应用:
  1. 跑 msprof, 找 top-3 kernel
  2. 每个 kernel 看 PipeUtil 分布, 走对应分支
  3. 优化后重测, 对比下界时间
  4. 反复迭代直到接近下界
  极限: 总时间 / 下界 = 1.1~1.3 算优秀""")
PYEOF

echo ""
echo "############################################################"
cat <<'EOF'
【整体总结】
- 小白:Profiling 数据 = 颜色 (Cube 紫 / Vec 绿 / MTE 黄) + 时间轴;bubble
  (白色 stall / 灰色空闲) 是优化点;top-3 耗时算子是优化目标;决策树按
  算力/访存/同步/调度四类瓶颈走。
- 熟手:占比排序后, 算力密集改 tile/切 TP, 访存密集开双缓冲/融合,
  同步密集查依赖, 调度密集用 CUDA Graph;总时间 / 下界 < 1.3 算优秀;
  msprof + chrome://tracing + 反复迭代是核心流程。
【进阶】msprof 跑自己的训练 step, 找 top-5 kernel, 按决策树优化每个,
  对比优化前后总时间。
EOF
echo "############################################################"
