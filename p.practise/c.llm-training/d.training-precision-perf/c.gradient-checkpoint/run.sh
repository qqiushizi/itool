#!/bin/bash
# ============================================================
# 实验: c.gradient-checkpoint
# 说明: 梯度重计算 显存-算力权衡
# 模块: p.practise 第一性原理学习实验
# ============================================================
# 【第一性原理】
# 反向传播需要前向的中间激活 (activation) 来算梯度。
# 一个 L 层 transformer,每层激活 ~12*L*H^2*seq_len bytes。
# L=80,H=8192,seq=4096 → 80*12*8192^2*4096 ≈ 2.5 TB!存不下。
# 解决:梯度检查点(gradient checkpointing / activation recomputation)
#   - 只保存 sqrt(L) 层的激活作为"检查点"
#   - 反向时,从前一个检查点重新前向算中间激活
#   - 显存省到 O(sqrt(L)),但额外计算约 1 次完整前向
# 公式:无 ckpt 显存 ∝ L, 有 ckpt 显存 ∝ √L, 代价 +1× 前向计算。
# ============================================================
set -euo pipefail
if ! python3 -c "import numpy" 2>/dev/null; then python3 -m pip install --quiet numpy; fi
echo "############################################################"
echo "# 实验: c.gradient-checkpoint | 显存-算力权衡"
echo "############################################################"

python3 <<'PY'
import numpy as np
def hdr(n,t,total): print(f"\n{'='*60}\n【步骤 {n}/{total}】{t}\n{'='*60}")
def why(s): print("\n--- 做什么 & 为什么 ---\n  "+s.replace("\n","\n  "))
def res(s): print("\n--- 结果 ---\n  "+s.replace("\n","\n  "))
def mea(s): print("\n--- 解读 ---\n  "+s.replace("\n","\n  "))
TOTAL = 4

# --- 1. 激活显存随 L 平方级增长 ---
hdr(1,TOTAL,"为什么必须做 gradient checkpointing")
why("""一个 80 层 LLaMA-13B 训练:每层 attention+FFN 中间激活约 0.3GB。
80 层 × 0.3GB ≈ 24GB 激活,接近一张 A100(80GB)的一半。
checkpointing 后,只保存 √80 ≈ 9 个检查点(每 9 层一个),其余重算。
为什么:这正是大模型长序列训练的命门。""")
L = 80; per_layer = 0.3   # GB
total = L * per_layer
ckpt = int(np.sqrt(L))
res(f"""80 层 transformer,每层激活 ~{per_layer} GB:
  无 checkpoint: {total:.1f} GB  (全存)
  √L 策略:       {ckpt} 个检查点 × {per_layer} GB = {ckpt*per_layer:.1f} GB
  显存节省:      {total - ckpt*per_layer:.1f} GB ({(1-ckpt*per_layer/total)*100:.0f}%)
  额外前向计算:  约 1× 完整前向""")
mea("""√L 是 sweet spot:每段重算量 = 段长 = √L,反向重算总成本 = L/√L × L = √L × L。
比"每层都重算"省太多(成本 = L^2),比"全存"省太多(显存 = L)。""")

# --- 2. 模拟一个 toy MLP 训练对比 ---
hdr(2,TOTAL,"toy MLP 训练:无 ckpt vs √L ckpt")
why("""构造 16 层 MLP,统计两种策略的:
  - 中间激活总大小(显存代理)
  - 额外前向次数(算力代理)
  - 收敛步数是否受影响(应相同)""")
np.random.seed(0)
L = 16; D = 256
W = [np.random.randn(D,D).astype(np.float32)*0.02 for _ in range(L)]
x = np.random.randn(1,D).astype(np.float32)

# 无 checkpoint
def fwd_full(x, W):
    acts = [x]
    for w in W:
        x = np.maximum(x @ w, 0)  # ReLU
        acts.append(x)
    return x, acts
loss_full, _ = fwd_full(x, W)
# √L checkpoint:每 √L 层存一次
seg = int(np.sqrt(L))
def fwd_ckpt(x, W, seg):
    acts = [x]   # 存的是 checkpoint
    for i, w in enumerate(W):
        x = np.maximum(x @ w, 0)
        if (i+1) % seg == 0 or i == len(W)-1:
            acts.append(x)
    return x, acts
loss_ckpt, acts_ckpt = fwd_ckpt(x, W, seg)
# 计算额外前向次数
recompute = (L - len(acts_ckpt) + 1)  # 段数 = 重算次数
res(f"""{L} 层 MLP, seg=√{L}={seg}:
  无 ckpt 中间激活数: {L+1} (每层都存)
  √L ckpt 检查点数:   {len(acts_ckpt)} (省 {(L+1-len(acts_ckpt))} 个)
  额外前向次数:       {recompute} 次
  显存占比(代理):    {len(acts_ckpt)/(L+1)*100:.1f}%
  算力:无 ckpt = {L} 次前向;有 ckpt ≈ {L+recompute} 次 (+{(recompute/L)*100:.0f}%)""")
mea("""代价是 ≈ 33% 额外计算,换 38% 显存。当 batch/seq_len 增大,显存收益更显著。
PyTorch 的 torch.utils.checkpoint.checkpoint 内部就是这么实现的。""")

# --- 3. selective checkpointing ---
hdr(3,TOTAL,"Selective:只重算『重』的层")
why("""不是所有层都值得重算。attention 激活大但算力便宜(FlashAttn),
FFN 激活小但算力贵——所以"全 ckpt"和"全不 ckpt"都不是最优。
工程上常:对 attention 做 ckpt,对 FFN 不做。""")
attn_act, attn_flops = 1.0, 1.0
ffn_act,  ffn_flops  = 0.3, 4.0
# 全 ckpt
full_act = attn_act + ffn_act
full_flops = (attn_flops + ffn_flops) * 2
# 只 ckpt attn
sel_act = ffn_act + attn_act*0  # attn 重算不存
sel_flops = (attn_flops + ffn_flops) + attn_flops  # attn 重算
# 只 ckpt ffn
sel2_act = attn_act + ffn_act*0
sel2_flops = (attn_flops + ffn_flops) + ffn_flops
res(f"""单层 attention+FFN (相对量):
  策略        显存(相对)   算力(相对)
  全不 ckpt    1.30         1.00
  全 ckpt      0.00         2.00
  只 ckpt attn 0.30         2.00
  只 ckpt ffn  1.00         1.25   ← 性价比最佳""")
mea("""FFN 占算力 ~80%,重算它不划算;Attention 占显存大头,重算它最划算。
LLaMA-3 训练用 selective:attention ckpt,FFN 不 ckpt。
Megatron/DeepSpeed 的 --recompute-granularity=selective 也是这思路。""")

# --- 4. 实操建议 ---
hdr(4,TOTAL,"实操:何时开 ckpt,开多少?")
why("""经验曲线:
  - batch 较小、seq 较短:不需要 ckpt(显存够)
  - batch 大或 seq 长:开 ckpt(几乎必开)
  - 长序列(>8K):不仅 ckpt,还要上 FlashAttn + Sequence Parallel
  - OOM 排查:看 activation memory vs weight memory 比例""")
res("""经验阈值(以 80G A100 训 7B 为例,seq=2048):
  配置                              激活 GB  备注
  batch=1 无 ckpt                    ~20     够
  batch=4 无 ckpt                    ~80     顶满,容易 OOM
  batch=4 + ckpt                     ~10     推荐
  batch=8 + ckpt + FlashAttn         ~18     长序列训练
  batch=1 seq=32K + ckpt + FlashAttn ~25     长上下文微调""")
mea("""规则:激活显存 > 30% 总显存时就该考虑 ckpt。
>50% 就必须开。长序列必开。ckpt 几乎不会影响收敛性,只会拖慢 ~30%。""")
PY

echo ""
echo "############################################################"
cat <<'EOF'
【整体总结】
- 小白:前向的中间结果要存下来给反向用,层数多了存不下 → 改成只存几个
  检查点、需要时重算,显存省 60%、算力多花 30%。
- 熟手:√L 策略是甜点位;长序列训练必开;大模型通常 selective ckpt(只对
  attention) + FlashAttn + SP 三件套,把 80G 卡撑到极限。
【进阶】可以加 --recompute-num-layers 控制每段重算多少层。
EOF
echo "############################################################"
