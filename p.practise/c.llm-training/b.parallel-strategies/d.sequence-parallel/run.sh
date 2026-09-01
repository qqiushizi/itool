#!/bin/bash
# ============================================================
# 实验: d.sequence-parallel
# 说明: SP/Context 并行、长序列切分
# 模块: p.practise/c.llm-training  LLM 训练
# ============================================================
# 【第一性原理】
# 序列并行(SP):长上下文时,把序列维度切到多卡,每卡只处理一段,降低单卡显存。
# 难点在注意力:某段要 attend 所有段→需跨卡通信交换 K/V(或用 Ring Attention 流式传)。
# Megatron SP:把 LayerNorm/激活的序列维也切分(非注意力部分),减少激活显存,层归一后再 AllGather 还原。
# 本实验演示序列切分 + 注意力跨段通信,并对比显存节省。
# ============================================================
set -euo pipefail
if ! python3 -c "import numpy" 2>/dev/null; then
  echo "未检测到 numpy,正在安装..."; python3 -m pip install --quiet numpy || { echo "请手动: pip install numpy"; exit 1; }
fi
echo "============================================================"
echo " 实验: 序列并行 / 长序列切分 / 跨段注意力"
echo "============================================================"
python3 <<'PY'
import numpy as np
np.set_printoptions(precision=3, suppress=True)
def softmax(x):
    x=x-x.max(axis=-1,keepdims=True); e=np.exp(x); return e/e.sum(axis=-1,keepdims=True)
rng=np.random.default_rng(0)
seq=8; d=4; P=2
Q=rng.standard_normal((seq,d)); K=rng.standard_normal((seq,d)); V=rng.standard_normal((seq,d))
full=softmax(Q@K.T/np.sqrt(d))@V
# 1 序列切分:Q 按段分到各卡,K/V 需全量(或跨卡交换)
seg=seq//P
Q0,Q1=Q[:seg],Q[seg:]
print("【1】序列切分:Q 按序列维切到2卡,每卡算自己段的注意力")
print(f"  完整注意力 shape={full.shape}")
# 每卡要 attend 全部 K/V → 需跨卡拿到对方的 K/V
out0=softmax(Q0@K.T/np.sqrt(d))@V; out1=softmax(Q1@K.T/np.sqrt(d))@V
sp=np.vstack([out0,out1])
print(f"  序列并行(每卡拿全量K/V)结果一致: {np.allclose(sp,full)}")
print("  解读:Q 切段后,每段要 attend 全部 K/V,必须跨卡交换 K/V(Ring Attention 流式传,省显存)。")

# 2 显存节省
print("\n【2】显存节省(激活随序列切分):")
for sp_n in [1,2,4]:
    per=(seq/sp_n)*d*4   # 每卡 Q 激活(FP32)
    print(f"  SP={sp_n}: 每卡序列长度={seq//sp_n}, Q激活≈{per} 元素 (单卡 {1/sp_n*100:.0f}%)")
print("  解读:序列维度切分让每卡激活/显存随 P 线性下降,使超长上下文训练成为可能。")

# 3 非注意力的 SP
print("\n【3】Megatron SP:LayerNorm/激活也按序列切,只在注意力前 AllGather 还原")
print("  解读:除注意力外的大部分算子(LayerNorm、Dropout、激活)按序列切分省激活;注意力时再聚合,综合降显存。")
PY
echo "============================================================"
cat <<'EOF'
【结论与进阶】
- 小白:序列并行把长序列切到多卡降显存;注意力要跨段交换 K/V(如 Ring Attention);Megatron SP 把非注意力层也切序列维。
- 熟手:Ring Attention 让 K/V 流式传递避免全量复制;Ulysses 用 AllToAll 重排注意力头;
  SP 对长上下文训练关键,与 TP/PP 正交可组合;通信量随序列长度和 P 变化。
- 延伸:把 seq 从8调到32看单卡显存压力;对比 Ring Attention 与朴素全量交换。
EOF
echo "============================================================"
