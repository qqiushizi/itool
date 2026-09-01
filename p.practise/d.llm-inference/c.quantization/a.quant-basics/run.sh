#!/bin/bash
# ============================================================
# 实验: a.quant-basics
# 说明: 对称/非对称量化、scale/zero-point
# 模块: p.practise 第一性原理学习实验
# ============================================================
# 【第一性原理】
# 量化 = 把 FP32 数值映射到低位宽整数 (INT8/INT4):
#   1. 找范围: min, max
#   2. 算 scale: (max - min) / (qmax - qmin)
#   3. 算 zero_point: 让 0 能精确表示
#   4. 量化: q = round((x - zero_point) * scale)
#   5. 反量化: x = (q - zero_point) / scale
# 对称量化:  zero_point = 0, 范围对称 [-a, a]
#   scale = max(|x|) / 127  (INT8)
# 非对称量化: zero_point ≠ 0, 范围 [min, max]
#   scale = (max - min) / 255
# 对称实现简单,非对称精度高(分布偏时省一段)。
# ============================================================
set -euo pipefail
if ! python3 -c "import numpy" 2>/dev/null; then python3 -m pip install --quiet numpy; fi
echo "############################################################"
echo "# 实验: a.quant-basics | 对称 vs 非对称量化,scale/zero-point"
echo "############################################################"

python3 <<'PYEOF'
import numpy as np
def hdr(n,t,total): print(f"\n{'='*60}\n【步骤 {n}/{total}】{t}\n{'='*60}")
def why(s): print("\n--- 做什么 & 为什么 ---\n  "+s.replace("\n","\n  "))
def res(s): print("\n--- 结果 ---\n  "+s.replace("\n","\n  "))
def mea(s): print("\n--- 解读 ---\n  "+s.replace("\n","\n  "))
TOTAL = 4

# --- 1. 对称量化 INT8 ---
hdr(1,TOTAL,"对称 INT8 量化:scale = max(|x|)/127")
why("""设 FP32 权重范围 [-2.0, 2.0](对称,0 多)。INT8 范围 [-128, 127]。
  scale = max(|x|) / 127 = 2.0/127 ≈ 0.0157
  zero_point = 0 (对称)
  量化:  q = round(x / scale)
  反量化: x ≈ q * scale
  实测误差: scale 的一半 ≈ 0.0079""")
np.random.seed(0)
x = np.random.randn(8) * 0.5  # 大多在 [-1, 1]
x = np.clip(x, -2.0, 2.0)
scale_sym = 2.0 / 127
zp_sym = 0
q_sym = np.round(x / scale_sym).clip(-128, 127).astype(np.int8)
x_recon_sym = q_sym.astype(np.float32) * scale_sym
res(f"原始:        {x.round(3).tolist()}\n  scale={scale_sym:.4f}, zp={zp_sym}\n  量化后:      {q_sym.tolist()}\n  反量化:      {x_recon_sym.round(3).tolist()}\n  最大误差:    {np.max(np.abs(x - x_recon_sym)):.4f}")
mea("对称量化适合权重分布对称、0 居中。LLaMA 类模型权重近似高斯,适合。")

# --- 2. 非对称量化 INT8 ---
hdr(2,TOTAL,"非对称 INT8:scale = (max-min)/255, zp ≠ 0")
why("""激活值分布常偏(silu/relu 后都是正),非对称更省:
  scale = (max - min) / 255
  zero_point = -round(min / scale) - 128  (把 min 映射到 -128, max 到 127)
  量化: q = round(x / scale) + zero_point""")
x = np.array([0.1, 0.5, 0.3, 0.8, 0.05, 0.9, 0.4, 0.7])  # 全正
mx, mn = x.max(), x.min()
scale_asym = (mx - mn) / 255
zp_asym = int(-round(mn / scale_asym) - 128)
q_asym = np.round(x / scale_asym + zp_asym).clip(-128, 127).astype(np.int8)
x_recon_asym = (q_asym.astype(np.float32) - zp_asym) * scale_asym
res(f"原始:        {x.tolist()}\n  scale={scale_asym:.5f}, zp={zp_asym}\n  量化后:      {q_asym.tolist()}\n  反量化:      {x_recon_asym.round(4).tolist()}\n  最大误差:    {np.max(np.abs(x - x_recon_asym)):.5f}")
mea("""非对称把区间 [min, max] 精确映射到 [-128, 127],不浪费。
  对比:若用对称(范围 [-0.9, 0.9]),正值只用 128 段,负值 128 段浪费。
  激活值量化必用非对称。""")

# --- 3. 对比误差 ---
hdr(3,TOTAL,"对称 vs 非对称误差对比")
why("""相同数据,两种量化误差对比。""")
np.random.seed(0)
# 偏分布
x = np.abs(np.random.randn(1000)) * 2 + 0.1  # 偏右
# 对称
scale_sym = np.max(np.abs(x)) / 127
q_sym = np.round(x / scale_sym).clip(-128, 127).astype(np.int8)
r_sym = q_sym.astype(np.float32) * scale_sym
# 非对称
mx, mn = x.max(), x.min()
scale_asym = (mx - mn) / 255
zp_asym = int(-round(mn / scale_asym) - 128)
q_asym = np.round(x / scale_asym + zp_asym).clip(-128, 127).astype(np.int8)
r_asym = (q_asym.astype(np.float32) - zp_asym) * scale_asym
res(f"""1000 个偏右样本 [0.1, 4.5]:
  对称 INT8    MSE: {((x-r_sym)**2).mean():.4f}   误差均值: {np.abs(x-r_sym).mean():.4f}
  非对称 INT8  MSE: {((x-r_asym)**2).mean():.4f}   误差均值: {np.abs(x-r_asym).mean():.4f}
  提升:        {((1-((x-r_asym)**2).mean()/((x-r_sym)**2).mean()))*100:.0f}%""")
mea("非对称在偏分布数据上精度高 50%+;权重用对称,激活用非对称是主流。")

# --- 4. per-tensor vs per-channel ---
hdr(4,TOTAL,"per-tensor vs per-channel:粒度")
why("""per-tensor: 整个 tensor 共 1 个 scale/zp
per-channel: 每个 channel 单独 1 个 scale/zp
粒度越细 → 精度越高 → 算力略增""")
np.random.seed(0)
# 模拟 Linear 权重:shape (out=4, in=8)
W = np.random.randn(4, 8) * np.array([0.1, 1.0, 5.0, 0.5])[:, None]  # 不同 channel 不同尺度
# per-tensor
mx, mn = W.max(), W.min()
s_t = (mx - mn) / 255
zp_t = int(-round(mn/s_t) - 128)
q_t = np.round(W/s_t + zp_t).clip(-128,127).astype(np.int8)
r_t = (q_t.astype(np.float32) - zp_t) * s_t
# per-channel(out)
s_c = np.zeros(4); zp_c = np.zeros(4, dtype=int); q_c = np.zeros_like(W, dtype=np.int8)
for i in range(4):
    s_c[i] = (W[i].max() - W[i].min()) / 255
    zp_c[i] = int(-round(W[i].min()/s_c[i]) - 128)
    q_c[i] = np.round(W[i]/s_c[i] + zp_c[i]).clip(-128,127).astype(np.int8)
r_c = (q_c.astype(np.float32) - zp_c[:, None]) * s_c[:, None]
res(f"""权重 shape=(4,8),4 个 channel 尺度差异大(0.1~5):
  per-tensor    MSE: {((W-r_t)**2).mean():.4f}
  per-channel   MSE: {((W-r_c)**2).mean():.4f}
  精度提升:      {((1-((W-r_c)**2).mean()/((W-r_t)**2).mean()))*100:.0f}%""")
mea("per-channel 是 LLM 量化的默认选择,几乎不增加算力但精度大提升。\n  进一步:per-group(GPTQ)把 group 内 64/128 元素共享 scale/zp,精度更好。")
PYEOF

echo ""
echo "############################################################"
cat <<'EOF'
【整体总结】
- 小白:量化 = 把 FP32 映射到 INT8/INT4,显存省 2-4×;对称简单、非对称
  精度高;per-channel 几乎必用,per-group 是更精细的选择。
- 熟手:权重用对称 + per-channel,激活用非对称 + per-tensor;INT8 几乎
  不掉点,INT4 需 calibration;per-group (GPTQ) 是 INT4 标配。
【进阶】用 bitsandbytes 跑一次 LLaMA-7B INT8 推理,看 perplexity 变化。
EOF
echo "############################################################"
