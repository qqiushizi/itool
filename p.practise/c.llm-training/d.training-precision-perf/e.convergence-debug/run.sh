#!/bin/bash
# ============================================================
# 实验: e.convergence-debug
# 说明: 损失震荡/发散、梯度范数监控、调参
# 模块: p.practise 第一性原理学习实验
# ============================================================
# 【第一性原理】
# 训练"不收敛"通常表现为:
#   - loss 震荡不下降(学习率过大 / 梯度噪声大)
#   - loss 突然飙到 NaN/Inf(梯度爆炸 / 数值溢出)
#   - loss 卡 plateau(学习率过小 / 落入局部最优)
# 监控三件套:
#   1. 训练 loss 曲线
#   2. 梯度范数 ‖g‖ (检测爆炸/消失)
#   3. 学习率/权重/激活 直方图(检测分布漂移)
# 调参顺序:lr > warmup > batch > β > wd。
# ============================================================
set -euo pipefail
if ! python3 -c "import numpy" 2>/dev/null; then python3 -m pip install --quiet numpy; fi
echo "############################################################"
echo "# 实验: e.convergence-debug | 损失/梯度监控与调参"
echo "############################################################"

python3 <<'PY'
import numpy as np
def hdr(n,t,total): print(f"\n{'='*60}\n【步骤 {n}/{total}】{t}\n{'='*60}")
def why(s): print("\n--- 做什么 & 为什么 ---\n  "+s.replace("\n","\n  "))
def res(s): print("\n--- 结果 ---\n  "+s.replace("\n","\n  "))
def mea(s): print("\n--- 解读 ---\n  "+s.replace("\n","\n  "))
TOTAL = 4

# --- 1. 三种 loss 曲线:稳定 / 震荡 / 爆炸 ---
hdr(1,TOTAL,"三种典型 loss 曲线")
why("""模拟一个 toy 训练(LR 过小/合适/过大),看 loss 走势。
诊断:
  - LR 太小:loss 下降慢,曲线平
  - LR 合适:loss 平滑下降,尾部 plateau
  - LR 太大:loss 震荡,可能后期发散""")
def train(lr, steps=100, seed=0):
    np.random.seed(seed)
    x = np.random.randn(20)
    w = np.zeros(20)
    losses = []
    for t in range(steps):
        g = 2*(w - x) + np.random.randn(20)*0.1
        w -= lr*g
        losses.append(np.mean((w-x)**2))
    return losses
ls = train(0.01)
lm = train(0.1)
ll = train(1.0)
res(f"""toy MSE 训练 100 步:
  LR=0.01: loss 末值 = {ls[-1]:.4f}    (过小,慢)
  LR=0.1:  loss 末值 = {lm[-1]:.4f}    (合适)
  LR=1.0:  loss 末值 = {ll[-1]:.4f}    (过大,震荡)
  早期(步 20)loss:  LR=0.01→{ls[19]:.3f}  LR=0.1→{lm[19]:.3f}  LR=1.0→{ll[19]:.3f}""")
mea("""学习率是最关键的超参,大 10× 就能让 loss 从下降到震荡。
实操:用 LR range test 找最大值——把 LR 从 0 指数增大,选 loss 还没炸的
最大值再 ÷ 3~10 作为正式 LR。""")

# --- 2. 梯度范数监控 ---
hdr(2,TOTAL,"梯度范数 ‖g‖:检测爆炸/消失")
why("""每步打印 ‖g‖ = sqrt(Σ g_i^2)。爆炸:‖g‖ 突然从 1 → 1000。消失:
‖g‖ 从 0.01 → 1e-7(深层 RNN/Transformer 早期常见)。
为什么:梯度的尺度直接决定更新步长,监控它是防止发散的第一道关。""")
np.random.seed(0)
steps = 100
norms_stable = [np.exp(-t/50)*0.5 + 0.01 + np.random.randn()*0.005 for t in range(steps)]
norms_explode = norms_stable.copy(); norms_explode[60:65] = [200, 500, 1000, np.nan, np.nan]
norms_vanish = [10.0*(0.95**t) + 1e-8 for t in range(steps)]
res(f"""‖g‖ 监控示意:
  正常训练(步 1/30/60/80/100):
    {norms_stable[0]:.3e} / {norms_stable[30]:.3e} / {norms_stable[60]:.3e} / {norms_stable[80]:.3e} / {norms_stable[-1]:.3e}
  爆炸(步 60-64):  {norms_explode[60]:.1f} / {norms_explode[61]:.1f} / {norms_explode[62]:.1f} / {norms_explode[63]:.1f} / {norms_explode[64]:.1f}  ← NaN!
  消失(步 1/30/60/80/100):
    {norms_vanish[0]:.3e} / {norms_vanish[30]:.3e} / {norms_vanish[60]:.3e} / {norms_vanish[80]:.3e} / {norms_vanish[-1]:.3e}""")
mea("""爆炸 → 立刻降 LR 或开 grad clip(threshold 1.0)。
消失 → 检查初始化(尤其深 RNN)、激活函数(改 ReLU/GELU)、归一化(LayerNorm)。
clip-by-norm: g = g * min(1, threshold/‖g‖) — 简单有效,几乎所有 LLM 都开。""")

# --- 3. 模拟一次发散 → 排查 → 修复 ---
hdr(3,TOTAL,"故障排查决策树")
why("""训练翻车时按这个顺序排查,90% 问题都能定位:""")
res("""决策树:
  loss 突然 NaN
  ├─ 检查 ‖g‖ 是否爆炸 → 是 → ① 开 grad clip ② 降 LR ③ 检查 loss scale
  ├─ 检查数据是否有脏样本(超长 seq/全 0 label) → 清洗数据
  └─ 数值溢出(FP16 训练) → 开 dynamic loss scaling / 切 BF16

  loss 不下降
  ├─ LR 太小?试 LR range test
  ├─ 数据未 shuffle?开启 shuffle
  ├─ loss 函数不对?(分类用 CE 用了 MSE)→ 换 loss
  └─ 模型容量不够?加层/加宽

  loss 震荡不收敛
  ├─ batch 太小 → 增大 batch 或 grad accum
  ├─ LR 太大 → 降 LR + warmup
  └─ 优化器选错(用 SGD 训 transformer)→ 换 AdamW""")
mea("""经验:90% 的"训练不动"是 LR 或数据问题,模型/优化器问题很少。
大模型训练里 warmup 几乎必开(尤其 transformer)——前 2000 步让 LR 从 0 涨。""")

# --- 4. warmup + cosine LR 曲线 ---
hdr(4,TOTAL,"warmup + cosine 是标配")
why("""训练初期权重随机,大 LR 会让 loss 飞掉。warmup:前 T_w 步 LR 从 0 线性升到 peak。
cosine 退火:之后按余弦从 peak 降到 0。
为什么:前期稳,后期精细调整。LLaMA/PaLM/GLM 全用这套。""")
def lr_schedule(t, peak, warmup, total, min_lr_frac=0.1):
    if t < warmup:
        return peak * t / warmup
    prog = (t - warmup) / (total - warmup)
    return peak * (min_lr_frac + (1 - min_lr_frac) * 0.5 * (1 + np.cos(np.pi*prog)))
total = 1000; warmup = 50; peak = 1e-3
ckpts = [0, 10, 50, 100, 500, 999]
res(f"""peak=1e-3, warmup={warmup}, total={total}, min_lr_frac=0.1:
  step   lr
  0      {lr_schedule(0,peak,warmup,total):.2e}
  10     {lr_schedule(10,peak,warmup,total):.2e}
  50     {lr_schedule(50,peak,warmup,total):.2e}    ← warmup 结束,达到 peak
  100    {lr_schedule(100,peak,warmup,total):.2e}
  500    {lr_schedule(500,peak,warmup,total):.2e}
  999    {lr_schedule(999,peak,warmup,total):.2e}   ← 接近 min_lr""")
mea("""变体:linear decay(直线下降),cosine restart(周期重启),inverse sqrt
(Transformer 原论文:lr = peak * min(t^-0.5, t*warmup^-1.5))。
大模型 SFT 常用 cosine,pretrain 常用 inverse sqrt(不预设总步数)。""")
PY

echo ""
echo "############################################################"
cat <<'EOF'
【整体总结】
- 小白:训练不收敛先看 loss 曲线和梯度范数;调参顺序 lr → batch → 优化器参数。
- 熟手:必开 grad clip + warmup + cosine;监控 loss / ‖g‖ / lr / 激活范数;
  LR range test 找上限;翻车时按决策树排查,90% 是 LR/数据问题。
【进阶】用 wandb / tensorboard 看 loss/grad_norm/learning_rate 三件套;真出问题
前先小 batch 跑通 sanity check(loss 应能到 0)。
EOF
echo "############################################################"
