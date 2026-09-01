#!/bin/bash
# ============================================================
# 实验: a.profiling
# 说明: msprof/profiling、瓶颈定位
# 模块: p.practise 第一性原理学习实验
# ============================================================
# 【第一性原理】
# 算子调优的起点是 profiling, 找到瓶颈:
#   1. 算力瓶颈: Cube 利用率 < 60% → 算子没打满
#   2. 访存瓶颈: MTE 时间 > 计算时间 → 搬运是瓶颈
#   3. 同步瓶颈: PipeBarrier 多 → 等待多
#   4. 调度瓶颈: 调度器 overhead 高 → kernel launch 太频繁
# 工具:
#   - msprof: 昇腾官方 profiler (类比 nsys)
#   - 算子级: msprof --application ...
#   - 火焰图: msprof --output=json 导 Chrome
# ============================================================
set -euo pipefail
echo "############################################################"
echo "# 实验: a.profiling | msprof 使用 + 瓶颈定位"
echo "############################################################"

python3 <<'PYEOF'
def hdr(n,total,t): print(f"\n{'='*60}\n【步骤 {n}/{total}】{t}\n{'='*60}")
def why(s): print("\n--- 做什么 & 为什么 ---\n  "+s.replace("\n","\n  "))
def res(s): print("\n--- 结果 ---\n  "+s.replace("\n","\n  "))
def mea(s): print("\n--- 解读 ---\n  "+s.replace("\n","\n  "))
TOTAL = 4

# --- 1. msprof 命令 ---
hdr(1,TOTAL,"msprof 命令速查")
why("""msprof 是昇腾 NPU 的 profiler, 类似 NVIDIA nsys:""")
res("""# 1. 应用级 profile (整应用)
msprof --application=\"./my_app\"

# 2. 算子级 profile (细粒度)
msprof --application=\"./my_app\" --aic-metrics=All

# 3. 输出 JSON 给 chrome trace
msprof --output=./output --application=\"./my_app\"
# 生成 prof_*.json, 用 chrome://tracing 看

# 4. 只看算子
msprof --sys-counter=0 --ai-core=on --ai-cpu=on \\
  --output=./prof -- ./my_app

# 5. Python 应用
msprof --application=\"python3 my.py\" \\
  --ai-core=on""")
mea("msprof vs nsys:\n  - msprof 还能采 AI Core 详细指标 (Pipe 利用率)\n  - nsys 适合 cross-platform (CPU+GPU)\n  - 国内项目几乎都用 msprof")

# --- 2. AI Core 详细指标 ---
hdr(2,TOTAL,"AI Core 详细指标解读")
why("""msprof --aic-metrics 采集的 AI Core 指标:""")
out = ["  指标                       反映              优化方向"]
out.append("  PipeUtilization_Cube       Cube 算力利用率    < 70% 则 tiling 太大")
out.append("  PipeUtilization_Vec        Vector 利用率      < 50% 则是访存密集")
out.append("  PipeUtilization_MTE        MTE 搬运利用率    过高则数据搬运是瓶颈")
out.append("  MemoryBandwidth            实际 HBM 带宽     < 50% 峰值则不访存密集")
out.append("  StallCycles                等待周期数         过多则同步有问题")
out.append("  InstructionIssued          指令发射数         -")
out.append("  CycleElapsed               总周期数           -")
res("\n".join(out))
mea("""看 msprof 火焰图:
  1. 横条是 kernel
  2. 颜色: 紫色=Cube, 绿色=MTE, 黄色=Vector
  3. 长度: 耗时
  4. 看哪个最长 = 优化目标""")

# --- 3. 瓶颈定位 5 步 ---
hdr(3,TOTAL,"5 步瓶颈定位流程")
why("""按下面流程, 90% 问题都能找到:""")
res("""步骤                          工具
  1. 跑 msprof, 导出 json       msprof --output=...
  2. 找耗时 top-5 kernel        chrome://tracing
  3. 看每个 kernel 的 Pipe 占比  msprof --aic-metrics
  4. 对比下界时间              max(IO 下界, 算力下界)
  5. 优化最大占比 kernel       重写 / 调参

示例:
  Kernel 1: GEMM 100 us, Cube 80%, MTE 15%, Vec 5%  → 算力下界 80us, 实际 80us, 完美
  Kernel 2: Add 30 us, MTE 90%, Vec 10%            → 访存下界 10us, 实际 30us, 有 IO 优化空间
  Kernel 3: LayerNorm 50 us, MTE 70%, Vec 30%      → 访存下界 20us, 实际 50us, 融合更好
""")
mea("""常见优化方向:
  - 算力利用率低: 加大 tile / 切 TP / 改算法
  - 访存利用率高: 算子融合 / 减少中间结果
  - 同步 stall 多: 检查 SetFlag/WaitFlag 依赖
  - Kernel 太多: 用 CUDA Graph / 图融合""")

# --- 4. 实战:msprof 报告示例 ---
hdr(4,TOTAL,"msprof 报告示例(伪)")
why("""一次 matmul kernel 的 msprof 报告:""")
out = ["  Kernel: my_matmul_128x128x32"]
out.append("  ───────────────")
out.append("  CycleElapsed:        1,234,567")
out.append("  PipeUtil_Cube:       75.3%   ← 良好")
out.append("  PipeUtil_Vec:        8.1%    ← 仅 epilogue (bias) 用")
out.append("  PipeUtil_MTE1:       12.5%   ← 读 A 块")
out.append("  PipeUtil_MTE2:       12.5%   ← 读 B 块")
out.append("  PipeUtil_MTE3:       5.0%    ← 写 C")
out.append("  MemoryBandwidth:     1.2 TB/s (峰值 2.7 TB/s) → 44%")
out.append("  CubeFLOPS:           210 TFLOPS (峰值 280) → 75%")
out.append("  ───────────────")
out.append("  结论: 算力下界 800 GFLOPS / 280 TFLOPS = 2.9 us")
out.append("        实际 1.2 ms, 远高于下界, 算子本身有优化空间")
out.append("        可能是 tile 不够大 / 双缓冲没开")
res("\n".join(out))
mea("""优化动作:
  1. 试 tile 256x128 看 Cube 利用率
  2. 试双缓冲看 MTE 时间是否能与 Cube overlap
  3. 读 SASS (Ascend 字节码) 看是否有冗余指令
  4. 多次跑看耗时稳定性 (一次 profile 不准)""")
PYEOF

echo ""
echo "############################################################"
cat <<'EOF'
【整体总结】
- 小白:msprof 是昇腾 profiler,看 kernel 耗时和 Pipe 利用率;Cube 利用率 70%+
  算优秀;MTE 利用率高 = 搬运瓶颈,需要融合;Chrome trace 火焰图找最耗 kernel。
- 熟手:AI Core 5 段 (Cube/Vec/MTE1/MTE2/MTE3) 都有利用率指标;对比下界
  (max(IO 下界, 算力下界)) 找差距;双缓冲、tile 大小、算子融合是常用优化;
  msprof + SASS + 多次 run 综合判断。
【进阶】msprof 跑自己的算子,看 Pipe 利用率, 识别最大耗时 kernel, 试 tile
  大小和双缓冲优化。
EOF
echo "############################################################"
