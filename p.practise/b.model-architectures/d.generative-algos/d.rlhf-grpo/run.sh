#!/bin/bash
# ============================================================
# 实验: d.rlhf-grpo
# 说明: RLHF/PPO、奖励模型、GRPO 算法概览
# 模块: p.practise/b.model-architectures  模型架构
# ============================================================
# 【第一性原理】
# RLHF 三步:① SFT 监督微调 → ② 训奖励模型 RM(给回答打分)→ ③ 用 RM 的奖励做 RL(PPO)微调策略。
# 策略梯度:最大化 E[r·log π(a|s)],即让高分回答的概率上升、低分下降。PPO 加 clip 防策略更新过猛。
# GRPO(DeepSeek):不要 critic,对同一 prompt 采一组回答,用组内"相对优势"(reward-均值)做基线,
# 省掉价值网络、更省显存。本实验用一个小策略演示策略梯度与 GRPO 的相对优势。
# ============================================================
set -euo pipefail
if ! python3 -c "import numpy" 2>/dev/null; then
  echo "未检测到 numpy,正在安装..."; python3 -m pip install --quiet numpy || { echo "请手动: pip install numpy"; exit 1; }
fi
echo "============================================================"
echo " 实验: RLHF / 策略梯度 / PPO / GRPO"
echo "============================================================"
python3 <<'PY'
import numpy as np
np.set_printoptions(precision=3, suppress=True)
rng=np.random.default_rng(0)
def softmax(x):
    x=x-x.max(); e=np.exp(x); return e/e.sum()
# 策略:对3个候选回答的概率
logits=np.array([1.0,2.0,0.5]); pi=softmax(logits)
rewards=np.array([0.2,0.8,0.1])   # RM 给的奖励
print("【1】RLHF 流程:SFT → 训奖励模型(RM)→ 用奖励做 RL 微调策略")
print(f"  当前策略 π={pi.round(3).tolist()}  奖励 r={rewards.tolist()}")

# 2 策略梯度:让高分回答概率↑
lr=1.0
adv=rewards-rewards.mean()       # 优势(减基线降方差)
new_logits=logits+lr*adv         # 梯度 ascent: logit += lr·advantage
pi_new=softmax(new_logits)
print(f"\n【2】策略梯度:advantage=r-均值={adv.round(3).tolist()}")
print(f"  更新后 π={pi_new.round(3).tolist()}  (高分回答2概率↑,低分↓)")
print("  解读:策略梯度按'优势'调整 logit,奖励高于均值的回答概率上升。减均值(基线)降方差。")

# 3 PPO clip & GRPO
print("\n【3】PPO 与 GRPO 的改进:")
print("  PPO:限制新旧策略比率 ratio=π_new/π_old 在 [1-ε,1+ε],防一步更新过猛导致崩溃。")
print("  GRPO:不要 critic,对同一 prompt 采 G 个回答,优势=(r_i - 组均值)/组std,省价值网络。")
# GRPO 示意:同一prompt采4个回答
g_rewards=np.array([0.1,0.5,0.9,0.3]); g_adv=(g_rewards-g_rewards.mean())/g_rewards.std()
print(f"  GRPO 示例:组内奖励={g_rewards.tolist()} → 相对优势={g_adv.round(3).tolist()}")
print("  解读:GRPO 用组内相对优势当基线,免去训练 critic,显存更省、实现更简,适合大模型 RL。")
PY
echo "============================================================"
cat <<'EOF'
【结论与进阶】
- 小白:RLHF=SFT+奖励模型+RL微调;策略梯度让高分回答概率升;PPO 防更新过猛;GRPO 用组内相对优势省掉 critic。
- 熟手:策略梯度方差大,基线/critic 降方差是关键;PPO 的 clip 是稳定性核心;
  GRPO 省价值网络、显存友好,是开源大模型 RL 的主流;奖励黑客(钻 RM 漏洞)是 RLHF 主要风险。
- 延伸:把奖励改成极端值看策略是否崩溃;对比有无基线时梯度的方差。
EOF
echo "============================================================"
