#!/bin/bash
# ============================================================
# 实验: c.peft-methods
# 说明: Adapter/Prefix/Prompt tuning 对比
# 模块: p.practise/c.llm-training  LLM 训练
# ============================================================
# 【第一性原理】
# PEFT(参数高效微调)只训少量额外参数,冻结基座,省资源、防遗忘。几种代表:
#  Adapter:在每层 Transformer 插入小瓶颈 MLP(降维→激活→升维),只训它;
#  Prefix/Prompt tuning:在输入或每层注意力前加可学习的"软提示"向量,只训这些前缀;
#  LoRA:低秩分解权重更新(见上一实验)。
# 共同点:基座冻结,只训极少参数(常 <1%)。本实验对比可训参数量与插入位置。
# ============================================================
set -euo pipefail
if ! python3 -c "import numpy" 2>/dev/null; then
  echo "未检测到 numpy,正在安装..."; python3 -m pip install --quiet numpy || { echo "请手动: pip install numpy"; exit 1; }
fi
echo "============================================================"
echo " 实验: PEFT / Adapter / Prefix / Prompt Tuning"
echo "============================================================"
python3 <<'PY'
import numpy as np
d=4096; L=32
full=L*d*d*3   # 基座全参(粗略,QKV+FFN)
print("【1】各 PEFT 方法的可训参数(基座冻结,只训额外参数):")
# Adapter:每层插一个 瓶颈 d→r→d
for r in [64]:
    adapter=L*(d*r+r+r*d)
    print(f"  Adapter(瓶颈r={r}): {adapter:,}  (占全参 {adapter/full*100:.2f}%)")
# Prefix tuning:每层加 len 个可学习前缀向量
for plen in [20]:
    prefix=L*plen*d
    print(f"  Prefix(len={plen}): {prefix:,}  (占全参 {prefix/full*100:.2f}%)")
# LoRA
for r in [16]:
    lora=L*2*d*r*4   # Q/K/V/O 各一对 A,B
    print(f"  LoRA(r={r}): {lora:,}  (占全参 {lora/full*100:.2f}%)")
print("  解读:三者可训参数都 <1%,基座冻结→省显存、防遗忘、可切换不同任务的轻量适配器。")

# 2 插入位置对比
print("\n【2】插入位置与机制对比:")
print("  Adapter:在 Transformer 层内插入小 MLP,改变层内计算(推理多一次小 MLP)")
print("  Prefix:在每层注意力 K/V 前拼可学习前缀,不改权重只改输入序列")
print("  Prompt:只在最输入加软提示,改动最小(甚至不进每层)")
print("  LoRA:旁路低秩更新权重,推理可合并回 W,零额外延迟")
print("  解读:LoRA 因可合并、无推理延迟,工程上最流行;Prefix/Prompt 改动小但效果常略逊。")

# 3 效果-成本权衡
print("\n【3】效果-成本权衡(直觉):")
print("  效果:全参 ≥ LoRA ≈ Adapter > Prefix > Prompt tuning")
print("  成本:全参 >> LoRA ≈ Adapter > Prefix > Prompt tuning")
print("  解读:LoRA 在效果和成本间平衡最好,是当前主流;任务越需深度适配,越倾向 LoRA/Adapter。")
PY
echo "============================================================"
cat <<'EOF'
【结论与进阶】
- 小白:PEFT 冻结基座只训极少额外参数;Adapter 插小MLP、Prefix/Prompt 加可学习前缀、LoRA 低秩更新权重,各占<1%参数。
- 熟手:LoRA 可合并回权重零延迟,工程首选;Prefix 改 K/V 前缀影响注意力,适合生成控制;
  Adapter 推理多一层计算;选型看任务适配深度、推理延迟、显存预算。
- 延伸:把 r/len 调大看参数占比升;对比各方法在下游任务的准确率。
EOF
echo "============================================================"
