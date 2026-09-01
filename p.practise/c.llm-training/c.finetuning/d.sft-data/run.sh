#!/bin/bash
# ============================================================
# 实验: d.sft-data
# 说明: 指令数据构造、loss masking
# 模块: p.practise/c.llm-training  LLM 训练
# ============================================================
# 【第一性原理】
# SFT(监督微调)用「指令-回答」对教模型听指令。数据格式:把 prompt 和 response 拼成一条序列,
# 但 loss 只算 response 部分(prompt 部分用 mask 屏蔽)——只学「怎么回答」,不学「怎么复述指令」。
# 这叫 loss masking / label masking。本实验演示序列拼接、label 构造与 mask。
# ============================================================
set -euo pipefail
if ! python3 -c "import numpy" 2>/dev/null; then
  echo "未检测到 numpy,正在安装..."; python3 -m pip install --quiet numpy || { echo "请手动: pip install numpy"; exit 1; }
fi
echo "============================================================"
echo " 实验: SFT / 指令数据 / loss masking"
echo "============================================================"
python3 <<'PY'
import numpy as np
np.set_printoptions(precision=0, suppress=True, linewidth=100)
# 模拟一条 SFT 样本:prompt + response(用 token id)
prompt=[10,11,12,13]          # 「请翻译这句话」
response=[20,21,22,23,24]     # 「翻译结果...」
seq=prompt+response
# label:把 prompt 位置置 -100(不计算 loss),response 位置是下一个 token(右移一位)
label=[-100]*len(prompt)+response
# 因果训练:输入是 seq[:-1], label 是 seq[1:] 的对应(mask prompt)
inp=seq[:-1]; lab=seq[1:]
lab_masked=[-100 if lab[i] in [-100] or i < len(prompt)-1 else lab[i] for i in range(len(lab))]
print("【1】SFT 样本:prompt+response 拼接,label 只在 response 区计算 loss")
print(f"  输入序列 = {inp}")
print(f"  原始label= {lab}")
print(f"  masked  = {lab_masked}  (-100 = 不计算 loss 的 prompt 区)")
print("  解读:模型只在「回答」部分计算 loss,学会生成回答而非复述指令;prompt 区被屏蔽。")

# 2 loss 只在 response
mask=np.array([0 if l==-100 else 1 for l in lab_masked])
print(f"\n【2】loss mask = {mask.tolist()}  (1=计算loss, 0=屏蔽)")
print(f"  有效 loss 位置 = {int(mask.sum())}/{len(mask)}  (只有 response 区)")
print("  解读:masking 让训练信号集中在回答质量上,prompt 长短不影响学习目标。")

# 3 数据质量
print("\n【3】SFT 数据质量要点:")
print("  - 指令多样、覆盖目标任务分布(指令跟随、多轮对话、代码…)")
print("  - 回答高质量、格式一致(拒答、CoT、工具调用按需)")
print("  - 数量不必巨大(几千~几万条常足够),质量>数量")
print("  解读:SFT 是把预训练模型「对齐」到听指令的关键;数据噪声会直接污染行为。")
PY
echo "============================================================"
cat <<'EOF'
【结论与进阶】
- 小白:SFT 用指令-回答对微调;把 prompt+response 拼序列,但 loss 只算 response 区(prompt 用 -100 屏蔽)。
- 熟手:loss masking 用 -100(label_smoothing 忽略);ChatML 等模板统一多轮对话格式;
  数据质量>数量;SFT 后常接 DPO/RLHF 进一步对齐; packing 多轮要按轮次 mask。
- 延伸:把 prompt 变长看有效 loss 比例下降;构造多轮对话的 mask(每轮回答区分别算)。
EOF
echo "============================================================"
