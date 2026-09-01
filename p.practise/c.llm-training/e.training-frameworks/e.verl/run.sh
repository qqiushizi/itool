#!/bin/bash
# ============================================================
# 实验: e.verl
# 说明: veRL 架构:RL 训练框架、Actor-Critic + Ray + Colocated 架构、与 SFT 训练框架解耦的接入方式
# 模块: p.practise 第一性原理学习实验
# ============================================================
# 【第一性原理】
# RLHF/PPO 需要 4 个模型:
#   - Actor(策略):  要训的 LLM
#   - Critic(价值): 单独训的 V(s)
#   - Reference:  冻结的 LLM,算 KL 散度的基准
#   - Reward:    奖励模型(打分)
# 显存 = 4×LLM,计算量也 4×。veRL 设计目标:
#   - 把 4 个模型灵活切到不同硬件池
#   - 用 Ray 做分布式调度
#   - Colocated: 4 个模型放同机多卡(显存够时)
#   - Decoupled: 4 个模型分散在多机(显存紧时)
# 创新: 与 SFT 框架解耦——veRL 只负责 RL 训练,SFT 用 HF/TF/任意框架
# ============================================================
set -euo pipefail
echo "############################################################"
echo "# 实验: e.verl | RL 训练:4 模型 + Ray 调度 + 与 SFT 解耦"
echo "############################################################"

python3 <<'PY'
def hdr(n,t,total): print(f"\n{'='*60}\n【步骤 {n}/{total}】{t}\n{'='*60}")
def why(s): print("\n--- 做什么 & 为什么 ---\n  "+s.replace("\n","\n  "))
def res(s): print("\n--- 结果 ---\n  "+s.replace("\n","\n  "))
def mea(s): print("\n--- 解读 ---\n  "+s.replace("\n","\n  "))
TOTAL = 4

# --- 1. 4 模型 + Actor-Critic ---
hdr(1,TOTAL,"RLHF 4 模型 + Actor-Critic 框架")
why("""PPO/GRPO 需要 4 个模型,这是和 SFT 最大区别:
  - Actor (策略):  我们要训的 LLM
  - Critic:       V(s) 网络,预测长期回报
  - Reference:    冻结 LLM,作为 KL 散度锚
  - Reward:       奖励模型,可以是 RM / rule-based / outcome
采样: 用 Actor rollout 一批 (prompt, response, log_prob) → 用 Reward 打分
训练: 用 PPO loss 同时更新 Actor + Critic
  Actor loss = -E[advantage * log_prob] + KL * (Actor||Reference)
  Critic loss = MSE(V(s), returns)""")
res("""Actor 一次 PPO step 的计算流:
  1. Rollout:    Actor 生成 response  →  (query, response, logp_old)
  2. Reward:     Reward model 打分   →  scores
  3. Value:      Critic 估 V(s)      →  values
  4. GAE:        用 scores + values  →  advantages, returns
  5. Update:     PPO-clip 更新 Actor + Critic
  6. KL:         Actor || Reference  →  KL penalty 加入 loss""")
mea("""4 个模型 + rollout + reward + update 一次 step 的算力是 SFT 的 ~5×。
这就是 RLHF 慢且贵的根因。GRPO 砍掉 Critic(改成 group-relative)省 ~30% 显存。""")

# --- 2. Ray 调度 + Colocated / Decoupled ---
hdr(2,TOTAL,"Ray 调度 + 两种部署模式")
why("""veRL 用 Ray 做分布式资源调度,把 4 个模型放到 Ray actor pool 里。
两种模式:
  - Colocated: 4 模型放同一节点的 4×N 卡
    优势: actor 不用走网络,通信快
    劣势: 显存全在这机,卡多才行
  - Decoupled: 4 模型分散在多机
    优势: 显存分散,小机也能跑
    劣势: rollout 需走网络,慢""")
res("""Colocated 模式(8×A100 训 7B):
  卡 0-3: Actor(7B, TP=4)        ← 主训目标
  卡 4-5: Critic(7B, TP=2)       ← 价值网络
  卡 6:   Reference(7B)            ← 冻结
  卡 7:   Reward(7B)               ← 奖励模型

Decoupled 模式(多机):
  Node1 (8 卡): Actor
  Node2 (4 卡): Critic
  Node3 (2 卡): Reference
  Node4 (2 卡): Reward
  → Ray 把 rollout / reward / update 任务通过 RPC 串起来""")
mea("""veRL 的创新是把\"哪张卡跑哪个模型\"完全配置化(yaml 改 actor_rollout_ref
  的 n_gpus 字段即可),让用户根据硬件池灵活调。
  替代方案:OpenRLHF / TRL 也支持多模型,但调度更死板。""")

# --- 3. GRPO:DeepSeek 提出的简化版 ---
hdr(3,TOTAL,"GRPO:无 Critic 的 PPO")
why("""GRPO (Group Relative Policy Optimization) 由 DeepSeek 提出,
  砍掉 Critic 网络,改成\"组内相对优势\":
  - 对每个 prompt 采样 G 个 response
  - G 个 response 的 reward 做归一化: A_i = (r_i - mean) / std
  - 用 A_i 当 advantage 训 Actor
  - 优: 省一个 7B 模型(~14G 显存)
  - 缺: 优势估计方差大,需要 G 较大""")
res("""PPO vs GRPO:
  项            PPO (Actor-Critic)        GRPO (DeepSeek)
  模型数        4 (Actor/Critic/Ref/RM)   3 (Actor/Ref/RM)
  Critic        必                          砍掉
  Advantage     GAE(TD)                   组内归一化
  KL 锚点      Ref model                  Ref model
  显存(7B)    ~120G                     ~90G
  训练速度      1×                         1.3×
  代表          InstructGPT               DeepSeek-R1""")
mea("""GRPO 已经成为 LLM RL 训练的事实标准(2024+),简单又省。
  进阶: DAPO / Dr. GRPO / GSPO 都是 GRPO 的改进变种。""")

# --- 4. 与 SFT 框架解耦的接入 ---
hdr(4,TOTAL,"与 SFT 解耦:veRL 只负责 RL,SFT 随便用")
why("""veRL 设计哲学:R 仅是 RL 训练框架,SFT 阶段用户自选(HF/transformers/
  LLaMA-Factory/任何)。
  流程:
    step 1. 任意 SFT 训好初始 Actor
    step 2. veRL 加载这个 Actor → 进入 RL
    step 3. RL 训完的 Actor 可独立部署
  这避免了\"RL 框架硬绑 SFT 框架\"的痛点""")
res("""veRL 工作流:
  ┌────────────────────────────────────┐
  │ 1. 准备数据(对话 + reward 标签)     │
  │ 2. 任意 SFT 训初始 Actor(HF/MS-LLM)│
  │ 3. 准备 reward model(也可用 rule)   │
  │ 4. veRL train:                      │
  │    config: 选定 actor/critic/ref 路径│
  │    algorithm: PPO / GRPO / REINFORCE│
  │    trainer: Ray + FSDP + vLLM       │
  │ 5. 训完 deploy 到 vLLM/SGLang       │
  └────────────────────────────────────┘""")
mea("""这种\"解耦\"是 veRL 与 OpenRLHF/TRL 的最大区别。
  优势: 不被 SFT 框架绑死;Actor 可以是任何来源的模型;
  代价: 用户需要自己保证 Actor 和 Reference 的 tokenizer 对齐。""")
PY

echo ""
echo "############################################################"
cat <<'EOF'
【整体总结】
- 小白:veRL = RLHF/PPO 训练框架,4 模型(Actor/Critic/Ref/RM)协调;
  Ray 调度;GRPO 砍掉 Critic 省显存;与 SFT 解耦,Actor 来自任意框架。
- 熟手:Colocated vs Decoupled 模式按显存/带宽选;FSDP 切 Actor,
  vLLM 加速 rollout;GRPO 是当前 RLHF 主流,DeepSeek-R1 用的就是它。
【进阶】用 veRL 跑一个 Qwen-7B + GRPO 训练(GSM8K 数学题)练手。
EOF
echo "############################################################"
