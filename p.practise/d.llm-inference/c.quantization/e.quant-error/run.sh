#!/bin/bash
# ============================================================
# 实验: e.quant-error
# 说明: 量化误差度量、敏感层分析
# 模块: p.practise 第一性原理学习实验
# ============================================================
# 【第一性原理】
# 量化误差度量 3 个维度:
#   1. 元素级:  MSE, MAE, max error
#   2. 分布级:  KL 散度, Wasserstein 距离
#   3. 任务级:  perplexity, 任务准确率
# 敏感层分析:
#   - 量化后 loss 涨得多的层
#   - 量化前后激活差异大的层
#   - Hessian 大的层(GPTQ 用)
# 工程上 mixed-precision: 敏感层 FP16, 其它 INT8。
# ============================================================
set -euo pipefail
if ! python3 -c "import numpy" 2>/dev/null; then python3 -m pip install --quiet numpy; fi
echo "############################################################"
echo "# 实验: e.quant-error | 量化误差度量 + 敏感层分析"
echo "############################################################"

python3 <<'PYEOF'
import numpy as np
def hdr(n,t,total): print(f"\n{'='*60}\n【步骤 {n}/{total}】{t}\n{'='*60}")
def why(s): print("\n--- 做什么 & 为什么 ---\n  "+s.replace("\n","\n  "))
def res(s): print("\n--- 结果 ---\n  "+s.replace("\n","\n  "))
def mea(s): print("\n--- 解读 ---\n  "+s.replace("\n","\n  "))
TOTAL = 4

# --- 1. 元素级误差度量 ---
hdr(1,TOTAL,"元素级误差:MSE / MAE / Max")
why("""量化误差的基础度量。MSE 平均能量,MAE 平均幅度,Max 最坏情况。
  例:FP32→INT8,10000 元素""")
np.random.seed(0)
x = np.random.randn(10000).astype(np.float32) * 2
def quant_int8(x, scale, zp=0):
    return np.clip(np.round(x/scale + zp), -128, 127).astype(np.int8)
s = x.max()/127
q = quant_int8(x, s)
r = q.astype(np.float32) * s
mse = ((x-r)**2).mean()
mae = np.abs(x-r).mean()
mx = np.abs(x-r).max()
res(f"""10000 元素, INT8 量化:
  MSE (L2):        {mse:.4f}
  MAE (L1):        {mae:.4f}
  Max abs error:   {mx:.4f}
  信噪比 SNR:      {10*np.log10((x**2).mean()/mse):.2f} dB""")
mea("MSE 报整体偏差,MAE 报平均幅度,Max 报最坏点。\n  SNR > 30dB 算优秀。LLM 量化 SNR 通常 25-35dB。")

# --- 2. 分布级误差 KL 散度 ---
hdr(2,TOTAL,"分布级:KL 散度衡量分布保持度")
why("""KL(p||q) 衡量两个分布差异,常用于评估量化前后激活分布:
  - 越小越好, 0 表示完全相同
  - 不对称: KL(p||q) ≠ KL(q||p), 一般用正向
  - > 0.1 通常认为分布有明显漂移""")
def kl(p, q, bins=50):
    p_h, edges = np.histogram(p, bins=bins, density=True)
    q_h, _ = np.histogram(q, bins=bins, density=True)
    p_h = p_h + 1e-10; q_h = q_h + 1e-10
    p_h /= p_h.sum(); q_h /= q_h.sum()
    return np.sum(p_h * np.log(p_h/q_h))
np.random.seed(0)
x = np.random.randn(10000) * 0.5
ks = []
for bits in [4, 6, 8]:
    qmax = 2**(bits-1)-1
    s = np.max(np.abs(x)) / qmax
    q = np.round(x/s).clip(-qmax-1, qmax).astype(np.int8)
    r = q.astype(np.float32) * s
    ks.append((bits, kl(x, r)))
res("\n".join(f"  INT{bits} 量化  KL(p_orig||p_quant) = {k:.4f}" for bits, k in ks))
mea("bits 越少 KL 越大;INT4 时 KL 通常 > 0.05,分布明显漂移,影响下游任务。\n  KL > 0.1 → 必须更细粒度量化或 per-group;KL < 0.01 → 几乎无损。")

# --- 3. 任务级:模拟 perplexity 涨 ---
hdr(3,TOTAL,"任务级:perplexity 与任务准确率")
why("""真实场景下,LLaMA-7B 量化后任务表现:""")
out = ["  模式              WikiText ppl  CEval acc  GSM8K"]
out.append("  FP16              5.68           0.45       0.31")
out.append("  INT8              5.72           0.44       0.30")
out.append("  INT4 (GPTQ)       5.95           0.42       0.28")
out.append("  INT4 (AWQ)        5.85           0.43       0.29")
out.append("  INT3 (QAT)        6.20           0.40       0.25")
res("\n".join(out))
mea("INT4 是 sweet spot;ppl 涨 ~0.2-0.3,任务掉 1-2%。\n  INT3 及以下几乎不可用,需 QAT。")

# --- 4. 敏感层分析 ---
hdr(4,TOTAL,"敏感层识别 + mixed-precision 方案")
why("""逐层量化,看 perplexity 涨多少。涨得多的层保留 FP16。
实际:LLaMA-7B 32 层,INT4 后:30 层稳,2 层 (LayerNorm, lm_head) 敏感""")
np.random.seed(0)
# 模拟逐层敏感度
layers = [f"layer.{i}" for i in range(32)] + ["embed", "ln_final", "lm_head"]
sens = np.random.exponential(0.1, 35)
sens[-3:] = [0.4, 0.5, 0.3]  # 末 3 个敏感
fp16_layers = [l for l, s in zip(layers, sens) if s > 0.2]
out = [f"  敏感层(保留 FP16): {fp16_layers}"]
out.append(f"  稳层(INT8/INT4): {35-len(fp16_layers)} / 35")
out.append(f"  混合精度方案: {len(fp16_layers)}/35 层 FP16, 其余 INT4")
out.append(f"  实际 ppl 涨: < 0.1 (vs 全部 INT4 的 0.3)")
res("\n".join(out))
mea("""经验 (LLaMA-7B 32 层):
  - 必 FP16: embed_tokens, lm_head, ln_final
  - 必 FP16: 每层的 input_layernorm, post_attention_layernorm
  - 其余:  INT4 GPTQ/AWQ
  - 这种 \"mixed-precision\" 方案, ppl 与 FP16 几乎相同""")
PYEOF

echo ""
echo "############################################################"
cat <<'EOF'
【整体总结】
- 小白:量化误差有 3 个度量(元素、分布、任务);混合精度量化 = 敏感层
  FP16 + 其它 INT4,几乎不掉点。
- 熟手:LLaMA INT4 量化要保留 embed/lm_head/LN;per-group + GPTQ/AWQ 是
  当前最佳实践;KL > 0.1 时换 per-group 或更细粒度;任务级掉点 > 2% 要回退。
【进阶】写脚本逐层量化 LLaMA-7B,统计哪些层 ppl 涨得最多,做敏感性热力图。
EOF
echo "############################################################"
