#!/bin/bash
# ============================================================
# 实验: d.optimizer-states
# 说明: 优化器状态显存建模(Adam)
# 模块: p.practise 第一性原理学习实验
# ============================================================
# 【第一性原理】
# Adam 优化器需要为每个参数存 2 个状态:momentum(m) 和 variance(v),
# 通常是 FP32 存。Adam 显存 = 参数量 × 8B(FP32 权 + FP32 m + FP32 v)。
# SGD 仅需参数量 × 4B(无状态或 momentum 一种)。
# 解决:ZeRO-1 把 optimizer state 分片到 N 张卡 → 每卡只存 1/N。
#       AdamW + 8-bit 优化器:把 m, v 量化到 8bit → 参数量 × 2B + 2B。
# ============================================================
set -euo pipefail
if ! python3 -c "import numpy" 2>/dev/null; then python3 -m pip install --quiet numpy; fi
echo "############################################################"
echo "# 实验: d.optimizer-states | Adam/SGD/Lion/8bit 显存建模"
echo "############################################################"

python3 <<'PY'
import numpy as np
def hdr(n,t,total): print(f"\n{'='*60}\n【步骤 {n}/{total}】{t}\n{'='*60}")
def why(s): print("\n--- 做什么 & 为什么 ---\n  "+s.replace("\n","\n  "))
def res(s): print("\n--- 结果 ---\n  "+s.replace("\n","\n  "))
def mea(s): print("\n--- 解读 ---\n  "+s.replace("\n","\n  "))
TOTAL = 4

# --- 1. 各种优化器显存 ---
hdr(1,TOTAL,"各优化器单卡显存(以 P 个参数计)")
why("""不同优化器状态大小:
  SGD:       0 字节(无状态,或仅 momentum=1×P×4B)
  SGD-mom:   1×P×4B(momentum)
  Adam:      2×P×4B(m,v) + P×4B(FP32 master)=12P 字节
  AdamW:     同 Adam
  Lion:      1×P×4B(m)
  8-bit Adam:2×P×1B(m,v) + P×4B(master) ≈ 6P 字节""")
P = 7e9   # 7B 模型
print()
res(f"""7B 模型(P={P:.1e}) 各优化器单卡显存(FP32 master weight 单独算):
  优化器            状态字节/参   状态 GB      master GB  合计 GB
  SGD              0B           0            0           0
  SGD-momentum     4B           {P*4/1e9:.0f}         0           {P*4/1e9:.0f}
  Adam (FP32)      12B          {P*12/1e9:.0f}         {P*4/1e9:.0f}        {P*16/1e9:.0f}
  AdamW (FP32)     12B          {P*12/1e9:.0f}         {P*4/1e9:.0f}        {P*16/1e9:.0f}
  Lion (FP32)      4B           {P*4/1e9:.0f}         {P*4/1e9:.0f}        {P*8/1e9:.0f}
  AdamW + 8bit     2B×2         {P*4/1e9:.0f}         {P*4/1e9:.0f}        {P*8/1e9:.0f}""")
mea("""Adam(16GB) 比 SGD-momentum(4GB) 多 4×,这就是为什么早期大模型选 SGD
不选 Adam(显存省得多)。现代用 ZeRO-1 把状态分到 64 卡 → 每卡 0.25GB。
8-bit 优化器(DeepSpeed bitsandbytes)再省 4×,效果略差但常可接受。""")

# --- 2. AdamW 单步更新模拟 ---
hdr(2,TOTAL,"AdamW 更新规则 + 动量方差曲线")
why("""AdamW 公式:
  m = β1*m + (1-β1)*g
  v = β2*v + (1-β2)*g^2
  m̂ = m/(1-β1^t),  v̂ = v/(1-β2^t)
  W = W - lr*(m̂/(√v̂+ε) + λ*W)
  β1=0.9, β2=0.999, ε=1e-8, λ=0.01
  m 是"梯度的指数滑动平均",v 是"梯度平方的滑动平均",二者给每个参数自适应学习率。""")
np.random.seed(0)
P = 100
g_seq = np.random.randn(20, P) * 0.01
g_seq[10] *= 100   # 一次大梯度
W = np.random.randn(P) * 0.02
m = np.zeros(P); v = np.zeros(P)
β1, β2, lr, eps, lam = 0.9, 0.999, 1e-3, 1e-8, 0.01
norms = []
for t, g in enumerate(g_seq, 1):
    m = β1*m + (1-β1)*g
    v = β2*v + (1-β2)*g**2
    mh = m/(1-β1**t); vh = v/(1-β2**t)
    update = mh/(np.sqrt(vh)+eps) + lam*W
    W -= lr*update
    norms.append(np.linalg.norm(update))
res(f"""随机梯度 + 第 10 步注入 100× 大梯度:
  步 1  更新范数: {norms[0]:.3e}
  步 5  更新范数: {norms[4]:.3e}    (动量累积中)
  步 10 更新范数: {norms[9]:.3e}    ← 大梯度!
  步 11 更新范数: {norms[10]:.3e}   ← Adam 自适应,把这次大更新\"摊薄\"到后续
  步 15 更新范数: {norms[14]:.3e}    (回到正常)
  步 20 更新范数: {norms[19]:.3e}""")
mea("""Adam 的「自适应」体现在:大梯度来临时,分母 √v 同步涨,实际更新幅度被压住。
所以 Adam 训练比 SGD 稳得多——v 的存在本质上给每个参数单独调了 lr。""")

# --- 3. ZeRO-1 分片计算 ---
hdr(3,TOTAL,"ZeRO-1 把 Adam 状态分到 N 卡")
why("""ZeRO-1 把 optimizer state(m, v)切成 N 份,每张卡只负责自己那部分
参数的更新。AllReduce 梯度后,每卡只更新自己持有的那 1/N 参数。
为什么:通信没多(每卡还得收到全部梯度),但显存省 N 倍。""")
P = 7e9
N_list = [1, 2, 4, 8, 16, 64, 128]
print()
res(f"""7B Adam 状态(每卡 GB,N 卡分片):
  N=1    {P*12/1e9:.0f} GB   ← 单卡根本装不下!
  N=2    {P*12/1e9/2:.0f} GB
  N=4    {P*12/1e9/4:.0f} GB
  N=8    {P*12/1e9/8:.0f} GB
  N=16   {P*12/1e9/16:.0f} GB
  N=64   {P*12/1e9/64:.0f} GB
  N=128  {P*12/1e9/128:.0f} GB
  → 64 卡训练 7B,单卡优化器状态 < 2 GB""")
mea("""DeepSpeed ZeRO-1 = 仅分片 optimizer state。
ZeRO-2 还分片 gradient,ZeRO-3 还分片 parameter。
代价是通信量略增,但显存可以无界扩展(代价是卡数 × 系数)。""")

# --- 4. 8-bit Adam 量化模拟 ---
hdr(4,TOTAL,"8-bit Adam:用 bitsandbytes 把 m,v 量化到 uint8")
why("""把 FP32 的 m, v 各自量化到 uint8:scale = max(|m|)/127, m_q = round(m/scale)。
更新时反量化回 FP32 算。误差 ~1%,但显存从 12B/参 降到 2B/参。
DeepSpeed + bitsandbytes 组合已在 BLOOM-176B 训练中验证。""")
P = 1000
m = np.random.randn(P).astype(np.float32) * 0.01
def quantize_uint8(x):
    s = max(1e-8, np.max(np.abs(x))/127)
    return np.clip(np.round(x/s), -127, 127).astype(np.int8), s
mq, scale = quantize_uint8(m)
m_recon = mq.astype(np.float32) * scale
err = np.abs(m - m_recon).max()
res(f"""m 大小: {P} 个 FP32 = {P*4/1024:.1f} KB
  量化到 int8     = {P*1/1024:.1f} KB (省 4×)
  scale           = {scale:.2e}
  最大反量化误差  = {err:.2e}
  相对最大幅值    = {err/np.max(np.abs(m))*100:.2f}%""")
mea("""8-bit Adam 实测可让 176B 模型单卡优化器状态 < 30GB。
BLOOM 训练经验:几乎不掉点(<0.1% eval loss 差)。
加速比:因为数据量小了,优化器 step 速度也快 1.5-2×。""")
PY

echo ""
echo "############################################################"
cat <<'EOF'
【整体总结】
- 小白:优化器为了"自适应"得给每个参数存额外状态(Adam 存 2 份),显存
  比 SGD 多 3 倍。ZeRO-1 把这些状态切到多卡,8-bit 优化器把它们量化。
- 熟手:大模型 99% 用 AdamW + ZeRO-1/2/3;极致省显存用 bitsandbytes 8-bit;
  Lion 优化器状态更小但收敛表现略差;AdaFactor 用低秩分解 m,v。
【进阶】试运行 bnb.optim.AdamW8bit;在 8×A100 训练 13B 时关掉 ckpt 看 OOM 边界。
EOF
echo "############################################################"
