#!/bin/bash
# ============================================================
# 实验: e.performance-model
# 说明: roofline、算力利用率、调优决策
# 模块: p.practise 第一性原理学习实验
# ============================================================
# 【第一性原理】
# Roofline 模型: 算子性能 = min(算力下界, 访存下界)
  - 算力下界:  FLOPs / FLOPS_peak
  - 访存下界:  Bytes / BW_peak
  - 实测时间 / 下界 = 1.0~1.5 算优秀
  决策: 算力密集 → 切 TP / 加大 tile / 换算法
        访存密集 → 双缓冲 / 融合 / 量化
# ============================================================
set -euo pipefail
echo "############################################################"
echo "# 实验: e.performance-model | Roofline + 调优决策"
echo "############################################################"

python3 <<'PYEOF'
def hdr(n,total,t): print(f"\n{'='*60}\n【步骤 {n}/{total}】{t}\n{'='*60}")
def why(s): print("\n--- 做什么 & 为什么 ---\n  "+s.replace("\n","\n  "))
def res(s): print("\n--- 结果 ---\n  "+s.replace("\n","\n  "))
def mea(s): print("\n--- 解读 ---\n  "+s.replace("\n","\n  "))
TOTAL = 4

# --- 1. Roofline 公式 ---
hdr(1,TOTAL,"Roofline 公式")
why("""给定算子:
  实测时间 T_actual
  算力下界 T_compute = FLOPs / FLOPS_peak
  访存下界 T_memory  = Bytes / BW_peak
  下界时间 T_lower = max(T_compute, T_memory)
  算子效率 η = T_lower / T_actual
  η > 0.7 优秀, η = 0.4-0.7 良好, η < 0.4 有优化空间""")
FLOPS_peak = 280e12   # A2 FP16
BW_peak = 2.7e12
out = [f"  A2 关键参数: FLOPS_peak = {FLOPS_peak/1e12:.0f} TFLOPS, BW = {BW_peak/1e12:.1f} TB/s"]
out.append("  拐点算子强度 I = FLOPS/BW = 104 FLOPs/Byte")
res("\n".join(out))
mea("Roofline 简单实用: 算一个算子的 I, 判断算力/访存密集, 决定优化方向。")

# --- 2. 算力利用率计算 ---
hdr(2,TOTAL,"算力利用率:实测 vs 理论")
why("""计算几个典型算子的算力利用率:""")
out = ["  算子               FLOPs        Bytes      I        主导    η (经验)"]
# Matmul 4096^3
N = 4096
flops_mm = 2*N**3
bytes_mm = 3*N*N*2
I_mm = flops_mm/bytes_mm
out.append(f"  Matmul 4096^3     {flops_mm/1e12:.1f} TF   {bytes_mm/1e9:.0f} GB   {I_mm:.0f}      算力     70-80%")
# Elementwise Add 4096^2
N = 4096
flops_ew = N*N
bytes_ew = 3*N*N*2
out.append(f"  Elementwise 4096^2  {flops_ew/1e9:.1f} G    {bytes_ew/1e6:.0f} MB   {flops_ew/bytes_ew:.2f}      访存     30-50%")
# Softmax 4096
flops_sm = 3*N  # max, exp, sum
bytes_sm = 2*N*2
out.append(f"  Softmax 4096        {flops_sm/1e3:.0f} k   {bytes_sm/1024:.0f} KB    {flops_sm/bytes_sm:.2f}      访存     40-60%")
# Attention 4096x4096
flops_at = 4*N*N
bytes_at = 8*N*N
out.append(f"  Attn 4096^2         {flops_at/1e9:.1f} G    {bytes_at/1e6:.0f} MB   {flops_at/bytes_at:.1f}        算力     60-75%")
res("\n".join(out))
mea("""LLM 算子特征:
  - Matmul: 算力密集, 容易 70-80% 算力
  - Elementwise: 访存密集, 难超 50% 算力 (但也不该)
  - Softmax: 访存密集, 30-40% 算力合理
  - Attention: 算力密集, 60-75% 算力""")

# --- 3. 调优决策 ---
hdr(3,TOTAL,"调优决策:按瓶颈选技术")
why("""4 大瓶颈 → 4 类技术:""")
out = ["  瓶颈        症状                优化技术                预期收益"]
out.append("  算力 算力  Cube 利用 < 60%      切 TP / 加大 tile / 量化  1.2-1.5×")
out.append("  访存 访存  MTE 占比 > 50%       双缓冲 / 融合 / 量化      1.5-2×")
out.append("  同步 同步  stall cycles > 30%   优化 Set/WaitFlag         1.1-1.3×")
out.append("  调度 调度  kernel launch 频繁   CUDA Graph / 图融合       1.2-1.5×")
res("\n".join(out))
mea("""实战顺序:
  1. 算法级: 切 TP / 切 batch / 量化
  2. Tile 级: 加大 tile 试 256x128
  3. Pipeline 级: 开双缓冲
  4. 融合级: 多个 elementwise 合一
  5. 调度级: CUDA Graph 减少 launch
  
  每步要 msprof 验证效果, 避免\"以为优化实际退化\"""")

# --- 4. 算子调优清单 ---
hdr(4,TOTAL,"算子调优 7 步检查清单")
why("""调优一个算子的标准流程:""")
out = ["  步     检查项                                工具/方法"]
out.append("  1. 算子边界: 极端 size 都对吗                  单元测试")
out.append("  2. 数值精度: 与 torch 对比 max abs error        numpy 比对")
out.append("  3. Pipe 利用: msprof 各 Pipe 占比              msprof")
out.append("  4. Roofline: 实测 vs 算力下界 vs 访存下界       算术")
out.append("  5. Tile 大小: 试 64/128/256 看哪个最快         msprof 多次")
out.append("  6. 双缓冲: 试 1/2/3 缓冲对比                   msprof")
out.append("  7. 融合: 能不能和上下游算子合                   看整体 timeline")
res("\n".join(out))
mea("""完成 7 步后:
  - η > 0.8 极致 (生产用)
  - η = 0.6-0.8 良好 (生产可用)
  - η = 0.4-0.6 一般 (有优化空间)
  - η < 0.4 差 (必须优化, 不建议上线)
  
  LLM 推理 7B 模型 decode 一步 η 整体 0.5-0.7 已算不错""")
PYEOF

echo ""
echo "############################################################"
cat <<'EOF'
【整体总结】
- 小白:Roofline 公式 = 实测 / max(算力下界, 访存下界);η > 0.7 算优秀;
  算力密集改 TP/tile, 访存密集开双缓冲/融合;7 步调优清单 = 边界 + 精度 +
  Pipe + Roofline + Tile + 缓冲 + 融合。
- 熟手:用 Roofline 算每个算子的 I, 决定优化方向;调优迭代 msprof 验证;
  η 0.6-0.8 是 LLM 推理现实目标;极致 (η > 0.8) 需要手写 + 双缓冲 + 融合 +
  切 TP + 量化 + 算法优化 6 步叠加。
【进阶】msprof 跑自己的 LLM 推理 step, 算每个 kernel 的 η, 识别 < 0.5 的
  按 7 步清单逐个优化。
EOF
echo "############################################################"
