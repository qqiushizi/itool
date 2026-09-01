#!/bin/bash
# ============================================================
# 实验: a.fusion-profiling
# 说明: 融合算子 profiling、与未融合对比
# 模块: p.practise 第一性原理学习实验
# ============================================================
# 【第一性原理】
# 融合算子 profiling 要回答:
#   1. 真的快了吗? (vs 未融合基线)
#   2. 快在哪里? (内存、计算、调度)
#   3. 是否达到峰值? (算力利用率)
#   4. 瓶颈在哪? (memory-bound / compute-bound)
# 工具:
#   - msprof / npu profiler: 抓 NPU 各 stage 耗时
#   - torch.profiler: 端到端 timeline
#   - 自家工具: tile 维度利用率
# 关键指标:
#   - 算力利用率 (MFU): 实测 FLOPs / 理论峰值
#   - 带宽利用率 (MBU): 实测 bytes / 理论带宽
#   - Cube / Vector 利用率: 各自占比
# ============================================================
set -euo pipefail
echo "############################################################"
echo "# 实验: a.fusion-profiling | 融合算子 profiling 与对比"
echo "############################################################"

python3 <<'PYEOF'
def hdr(n,total,t): print(f"\n{'='*60}\n【步骤 {n}/{total}】{t}\n{'='*60}")
def why(s): print("\n--- 做什么 & 为什么 ---\n  "+s.replace("\n","\n  "))
def res(s): print("\n--- 结果 ---\n  "+s.replace("\n","\n  "))
def mea(s): print("\n--- 解读 ---\n  "+s.replace("\n","\n  "))
TOTAL = 4

# --- 1. Profiling 工具概览 ---
hdr(1,TOTAL,"Profiling 工具全景")
why("""Ascend 平台有 3 层 profiling:
  1. NPU 硬件层: msprof (抓每个 kernel 耗时)
  2. 框架层: torch.profiler (Python 调用栈 + NPU)
  3. 自定义: 算子内加 timer (精细到 Set/WaitFlag)
每一层给的信息不同:
  - msprof: kernel 粒度, 看 MTE/Cube/Vector 占比
  - torch.profiler: Python + NPU, 看哪段代码慢
  - 自家: 算子内 Set/WaitFlag 间隔, 看 pipe 是否阻塞""")
out = ["  工具           粒度         适用场景                输出"]
out.append("  msprof        kernel       找 NPU 瓶颈            时序图, 占比")
out.append("  torch.profiler Python+NPU  找 Python 瓶颈         timeline")
out.append("  nsight        细粒度      算子内部 pipe 阻塞     Set/Wait 间隔")
out.append("  自己加 timer  自定义       算子内部各阶段         阶段耗时")
res("\n".join(out))
mea("""Profiling 顺序:
  1. 先 torch.profiler 看 Python 端慢不慢
  2. Python 快就 msprof 看 NPU 端
  3. NPU 慢就自家工具看算子内部
  4. 找 cube/vector/mte 占比, 算 memory/compute bound
注意: 第一次跑要 warmup (10-20 次), 否则不稳定""")

# --- 2. 融合 vs 未融合对比 ---
hdr(2,TOTAL,"融合 vs 未融合:profile 对比")
why("""用 msprof 同时跑融合版和未融合版, 对比:
  - 端到端耗时
  - kernel 数 (融合: 1 个, 未融合: N 个)
  - HBM 带宽 (融合: 低, 未融合: 高)
  - 算子利用率 (融合: 接近 100%, 未融合: 中等)
伪数据: 7 层 transformer 推理""")
out = ["  指标              未融合       融合        提升"]
out.append("  Kernel 数          35          7           -5x")
out.append("  总耗时 (us)       280         130         2.15x")
out.append("  HBM 读 (KB)       850         320         2.66x")
out.append("  HBM 写 (KB)       320         120         2.66x")
out.append("  Cube 利用率       45%         78%         +33pp")
out.append("  Vector 利用率     22%         65%         +43pp")
out.append("  算力利用率 (MFU)  35%         65%         +30pp")
out.append("  显存 (中间 buffer) 12 MB      3 MB        -75%")
res("\n".join(out))
mea("""融合的 3 大收益 (profile 体现):
  1. 耗时降 50-70% (kernel 数 -5x)
  2. 带宽降 60% (中间 buffer 少)
  3. 利用率 +30pp (算子间无空隙, 双发射好)
注意: 融合后单 kernel 变大, 调试更难, 要 profile 配合""")

# --- 3. 算力/带宽利用率分析 ---
hdr(3,TOTAL,"MFU/MBU:Roofline 分析")
why("""Roofline 模型: 实测点 vs 算力/带宽峰值的比。
算力 bound (compute-bound): MFU 高, MBU 低 -> 算子是瓶颈
带宽 bound (memory-bound): MBU 高, MFU 低 -> 搬运是瓶颈
融合算子应该:
  - 简单融合 (Linear+ReLU): compute-bound, MFU 接近 100%
  - 复杂融合 (MHA): memory-bound, MBU 接近 100%""")
import numpy as np
out = ["  融合算子           算力类型     MFU    MBU    瓶颈"]
out.append("  Linear+ReLU        compute     88%    35%    Cube")
out.append("  Linear+LayerNorm   compute     75%    45%    Cube")
out.append("  Softmax            memory      20%    92%    HBM 读")
out.append("  MHA (q,k,v)        compute     80%    55%    Cube")
out.append("  MHA+Softmax        mixed       65%    75%    平衡")
out.append("  LayerNorm+RMSNorm  memory      15%    95%    累加器读")
res("\n".join(out))
mea("""Roofline 优化方向:
  1. compute-bound: 算子融合, 减少重复算
  2. memory-bound: 算子融合, 减少 HBM 读写
  3. 双向 bound: 拆分融合, 各自最优
  4. 都不是: 看 pipe 同步, Set/WaitFlag 等待
MFU > 80% 已经不错, 95% 算接近极限""")

# --- 4. 实际 profile 命令 ---
hdr(4,TOTAL,"msprof 实测命令")
why("""msprof 用法:""")
res("""# 1. 采集
msprof --application="python3 model.py" \
       --output=./profiling_output \
       --timeline=on \
       --aicpu=on \
       --aicore=on

# 2. 看 timeline (浏览器打开)
# 文件: profiling_output/timeline.json
# 在 chrome://tracing 打开

# 3. 看 summary
cat profiling_output/summary.txt
# 输出:
# Total time: 1.23 s
# NPU time: 1.15 s
# Kernel 数: 35
# Cube 利用率: 78%
# Vector 利用率: 65%

# 4. 找 hot kernel
python3 -c "
import json
with open('profiling_output/kernel_details.json') as f:
    data = json.load(f)
kernels = sorted(data['kernels'], key=lambda k: k['duration'], reverse=True)
for k in kernels[:5]:
    print(f\"{k['name']}: {k['duration']:.1f} us, cube={k['cube_util']:.0%}\")
"

# 5. 对比融合 vs 未融合
python3 compare.py --baseline unfused --fused fused""")
mea("""Profile 调试技巧:
  1. 第一次跑要 warmup (10-20 次), 排除初始化
  2. profile 模式会拖慢 2-3x, 正常
  3. 看 timeline 看各算子间隔 (gap), gap 大 = 调度问题
  4. 看 cube/vector 占比, 算是否平衡
  5. 找 hot kernel 优先优化 (Pareto)
  6. 一定要对比基线, 否则看不出提升""")
PYEOF

echo ""
echo "############################################################"
cat <<'EOF'
【整体总结】
- 小白:融合算子 profiling 看 3 件事: 端到端耗时、kernel 数、利用率;
  msprof 抓 NPU, torch.profiler 抓 Python, nsight 抓算子内部;
  一定要对比基线, 找 hot kernel 优先优化。
- 熟手:Roofline 模型判断 compute-bound (MFU 高) vs memory-bound (MBU 高);
  融合算子 MFU 应 > 80%, 简单融合 (Linear+ReLU) 接近 100%;
  profile 流程: warmup -> torch.profiler -> msprof -> 自家工具;
  cube/vector 平衡是性能关键。
【进阶】用 msprof 抓 1 个真实模型 (如 Llama 7B 推理), 对比 融合 vs 未融合,
  输出 timeline 截图 + 算力/带宽利用率报告; 找 hot kernel 优化, 迭代 3 轮。
EOF
echo "############################################################"
