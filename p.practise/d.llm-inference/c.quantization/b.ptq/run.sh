#!/bin/bash
# ============================================================
# 实验: b.ptq
# 说明: 训练后量化、校准、per-channel/per-tensor
# 模块: p.practise 第一性原理学习实验
# ============================================================
# 【第一性原理】
# PTQ (Post-Training Quantization) = 训完模型后再量化,不重新训练:
#   1. 准备 calibration 数据集(几十~几百样本)
#   2. 跑前向,统计每层激活分布
#   3. 选每层最优 scale/zp(最小化 KL 散度或 MSE)
#   4. 量化权重 + 激活 → INT8
# 优点: 不用训练,几分钟搞定
# 缺点: 校准不好精度掉;敏感层可能需要 fallback 到 FP16
# 校准算法:
#   - MinMax: 直接用 min/max,简单但 outlier 影响大
#   - Percentile (99%): 截断 outlier
#   - KL 散度: 让量化后分布最接近原始(TensorRT 用)
#   - MSE: 最小化反量化误差(AWQ/GPTQ)
# ============================================================
set -euo pipefail
if ! python3 -c "import numpy" 2>/dev/null; then python3 -m pip install --quiet numpy; fi
echo "############################################################"
echo "# 实验: b.ptq | 训练后量化:校准算法对比 + per-tensor/per-channel"
echo "############################################################"

python3 <<'PYEOF'
import numpy as np
def hdr(n,t,total): print(f"\n{'='*60}\n【步骤 {n}/{total}】{t}\n{'='*60}")
def why(s): print("\n--- 做什么 & 为什么 ---\n  "+s.replace("\n","\n  "))
def res(s): print("\n--- 结果 ---\n  "+s.replace("\n","\n  "))
def mea(s): print("\n--- 解读 ---\n  "+s.replace("\n","\n  "))
TOTAL = 4

# --- 1. 4 种校准算法 ---
hdr(1,TOTAL,"4 种校准算法:MinMax / Percentile / KL / MSE")
why("""激活值常有 outlier(几个极大值),直接 MinMax 会浪费大量区间。
Percentile 截断 1% outlier 改善;KL 散度让量化分布最接近原始;MSE 优化反量化误差。""")
np.random.seed(0)
# 模拟激活值:大部分 [-1, 1] + 几个 10x outlier
x = np.concatenate([np.random.randn(1000)*0.3, np.array([5, -5, 7, -6])])
def quant_int8(x, scale, zp):
    q = np.round(x/scale + zp).clip(-128, 127).astype(np.int8)
    return (q.astype(np.float32) - zp) * scale
def kl_div(p, q, bins=50):
    p_h, _ = np.histogram(p, bins=bins, density=True)
    q_h, _ = np.histogram(q, bins=bins, density=True)
    p_h = p_h + 1e-10; q_h = q_h + 1e-10
    p_h /= p_h.sum(); q_h /= q_h.sum()
    return np.sum(p_h * np.log(p_h / q_h))
results = []
# 1. MinMax
mx, mn = x.max(), x.min()
s = (mx-mn)/255; zp = int(-round(mn/s)-128)
r = quant_int8(x, s, zp); results.append(("MinMax", np.abs(x-r).mean()))
# 2. Percentile 99
p99, p1 = np.percentile(x, 99), np.percentile(x, 1)
s = (p99-p1)/255; zp = int(-round(p1/s)-128)
r = quant_int8(x, s, zp); results.append(("Percentile 99", np.abs(x-r).mean()))
# 3. KL: 试几个 threshold 取 KL 最小
best = (1e9, None, None)
for pct in [99, 99.5, 99.9, 99.95]:
    hi = np.percentile(np.abs(x), pct)
    s = hi/127; zp = 0
    r = quant_int8(x, s, zp)
    d = kl_div(x, r)
    if d < best[0]: best = (d, pct, r)
results.append((f"KL(best {best[1]}%)", np.abs(x-best[2]).mean()))
# 4. MSE: 网格搜索最优 scale
best_mse, best_s = 1e9, 0
for s_try in np.linspace(0.005, 0.5, 100):
    r = quant_int8(x, s_try, 0)
    m = np.mean((x - r)**2)
    if m < best_mse: best_mse, best_s = m, s_try
r = quant_int8(x, best_s, 0)
results.append(("MSE-optimal", np.abs(x-r).mean()))
res("1000 样本 + 4 outlier,4 种校准平均误差:\n  " + "\n  ".join(f"{n:20s} {e:.4f}" for n,e in results))
mea("MinMax 误差最大(outlier 拖累);Percentile 99/99.9 简单有效;KL/MSE 最优但需算。\n  实战:TensorRT 用 KL,LLaMA.cpp 用 Percentile,AWQ 用激活分布感知。")

# --- 2. per-tensor vs per-channel ---
hdr(2,TOTAL,"per-tensor vs per-channel")
why("""同一权重,两种粒度量化误差对比。per-channel 是 LLM 默认。""")
np.random.seed(0)
W = np.random.randn(64, 64) * np.linspace(0.1, 5.0, 64)  # 每行尺度递增
def quant_per_tensor(W, bits=8):
    qmax = 2**(bits-1)-1
    mx, mn = W.max(), W.min()
    s = (mx-mn)/(2*qmax)
    zp = int(-round(mn/s)-qmax-1)
    q = np.round(W/s + zp).clip(-qmax-1, qmax).astype(np.int8)
    return (q.astype(np.float32) - zp) * s
def quant_per_channel(W, bits=8, axis=0):
    qmax = 2**(bits-1)-1
    out = np.zeros_like(W)
    for i in range(W.shape[axis]):
        sl = [slice(None)]*W.ndim; sl[axis] = i
        sl = tuple(sl)
        sub = W[sl]
        mx, mn = sub.max(), sub.min()
        if mx==mn: continue
        s = (mx-mn)/(2*qmax)
        zp = int(-round(mn/s)-qmax-1)
        q = np.round(sub/s + zp).clip(-qmax-1, qmax).astype(np.int8)
        out[sl] = (q.astype(np.float32) - zp) * s
    return out
r_t = quant_per_tensor(W)
r_c = quant_per_channel(W)
res(f"""权重 (64,64),INT8:
  per-tensor    MSE: {((W-r_t)**2).mean():.4f}
  per-channel   MSE: {((W-r_c)**2).mean():.4f}
  提升:          {((1-((W-r_c)**2).mean()/((W-r_t)**2).mean()))*100:.0f}%""")
mea("per-channel 在异尺度通道下优势明显。LLM 量化默认 per-channel(对 Linear 输出维度)。")

# --- 3. 敏感层识别 ---
hdr(3,TOTAL,"敏感层:哪些层不该量化")
why("""不是所有层都该量化。经验:
  - Embedding/LM head: 词表大小,影响大,通常 FP16
  - attention Q/K/V: 精度敏感,敏感层
  - FFN up/down: 较稳
  - LayerNorm: 极敏感(归一化放大误差),必须 FP16
简单识别:看每层 INT8 量化前后 output 差异,差异 > 阈值就 fallback。""")
np.random.seed(0)
# 模拟 4 层
layers = ["embed", "attn_qk", "attn_v", "ffn", "ln", "lm_head"]
sensitivities = [0.3, 0.05, 0.12, 0.01, 0.4, 0.25]  # 模拟的敏感度
out = ["  层         敏感度  建议"]
for n, s in zip(layers, sensitivities):
    rec = "FP16" if s > 0.2 else ("混合 INT8" if s > 0.1 else "INT8 全量化")
    out.append(f"  {n:10s}  {s:.2f}    {rec}")
res("\n".join(out))
mea("""LLM INT8 量化经验:
  - lm_head / embed:  FP16 (敏感)
  - LayerNorm:        FP16 (必)
  - attention QKVO:   INT8 (稳)
  - FFN:              INT8 (稳)
  - 这样 mixed-precision 量化,几乎不掉点""")

# --- 4. PTQ vs QAT ---
hdr(4,TOTAL,"PTQ vs QAT (Quantization-Aware Training)")
why("""PTQ: 训完再量化,几分钟
QAT: 训练时模拟量化噪声,几小时
QAT 精度好但贵,PTQ 简单便宜是主流""")
res("""对比        PTQ           QAT
  时间       分钟            小时
  精度       良好(< 0.5% ppl) 优秀(< 0.1%)
  适用        多数情况         精度敏感(INT4)
  工具       bitsandbytes    LSQ / QLoRA 训练时模拟
  复现        高              中(随机种子敏感)""")
mea("LLM 推理几乎都用 PTQ(快、便宜)。QAT 在 INT3 及以下才用,因为精度掉太多。\n  QLoRA 训练 = 4bit 基座(QAT 风格的伪量化) + LoRA adapter,本实验后续会讲。")
PYEOF

echo ""
echo "############################################################"
cat <<'EOF'
【整体总结】
- 小白:PTQ = 训完模型再量化,用校准数据算 scale/zp,几分钟搞定;per-channel
  比 per-tensor 精度高,敏感层(LayerNorm/lm_head)留 FP16。
- 熟手:校准算法 MinMax < Percentile < KL < MSE;LLM 量化默认 mixed-precision
  (敏感层 FP16,其它 INT8);QAT 仅在 INT3/4 精度不够时用。
【进阶】用 bitsandbytes --load-in-8bit 跑 LLaMA-7B,看 perplexity 和显存。
EOF
echo "############################################################"
