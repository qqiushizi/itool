#!/bin/bash
# ============================================================
# 实验: c.transformer-core
# 说明: 自注意力 QKV、多头、位置编码手算
# 模块: p.practise/b.model-architectures  模型架构
# ============================================================
# 【第一性原理】
# 注意力:每个词当查询 Q,去和所有词的键 K 打分(Q·K),softmax 成权重,再用权重加权所有词的值 V。
# 于是每个词的新表示=所有词的加权混合,谁相关谁的权重高→"上下文感知"。
# 多头:把 Q/K/V 切成 h 组并行做注意力再拼回,让模型从多个子空间同时建模不同关系。
# 位置编码:注意力本身没有顺序概念,要靠加位置编码告诉模型"第几个词"。
# 本实验手算一个 3 词句子的自注意力 + 多头 + 正弦位置编码。
# ============================================================
set -euo pipefail
if ! python3 -c "import numpy" 2>/dev/null; then
  echo "未检测到 numpy,正在安装..."; python3 -m pip install --quiet numpy || { echo "请手动: pip install numpy"; exit 1; }
fi
echo "============================================================"
echo " 实验: 自注意力 / QKV / 多头 / 位置编码"
echo "============================================================"
python3 <<'PY'
import numpy as np
np.set_printoptions(precision=3, suppress=True)
rng=np.random.default_rng(0)
def softmax(x):
    x=x-x.max(axis=-1,keepdims=True); e=np.exp(x); return e/e.sum(axis=-1,keepdims=True)
seq,d=3,4
# 1 自注意力 QKV
print("【1】自注意力:每个词当 Q,和所有词的 K 打分,加权 V")
X=rng.standard_normal((seq,d))
Wq=Wk=Wv=rng.standard_normal((d,d))
Q,K,V=X@Wq,X@Wk,X@Wv
scores=Q@K.T/np.sqrt(d); A=softmax(scores); O=A@V
print(f"  注意力权重 A(每行=某词对其他词的关注度)=\n{A}")
print(f"  输出 O(每个词=所有词值的加权混合)=\n{O}")
print("  解读:权重行softmax和为1;某词的新表示=按相关度混合所有词的V→上下文感知。√d 缩放防分数过大。")

# 2 多头:切成 h 组并行
print("\n【2】多头注意力:把 d=4 切成 2 个头(各 d=2)并行,再拼回")
h=2; dh=d//h
heads=[Q[:,i*dh:(i+1)*dh]@K[:,i*dh:(i+1)*dh].T/np.sqrt(dh) for i in range(h)]
heads=[softmax(m) for m in heads]
print(f"  头0权重=\n{heads[0]}\n  头1权重=\n{heads[1]}")
print("  解读:不同头关注不同关系(如头0看语法、头1看指代);并行后拼接,模型从多子空间同时建模。")

# 3 正弦位置编码
print("\n【3】正弦位置编码:给注意力补回顺序信息")
pe=np.zeros((seq,d))
for pos in range(seq):
    for i in range(d):
        pe[pos,i]=np.sin(pos/10000**(2*(i//2)/d)) if i%2==0 else np.cos(pos/10000**(2*((i-1)//2)/d))
print(f"  位置编码 PE(每行=一个位置的编码)=\n{pe}")
print("  解读:用不同频率的 sin/cos 给每个位置唯一编码;加到词向量上后,注意力就能感知'谁在前谁在后'。")
PY
echo "============================================================"
cat <<'EOF'
【结论与进阶】
- 小白:注意力=用 Q·K 打分、softmax 加权 V,让每个词融合上下文;多头=多组并行捕捉不同关系;
  注意力本身无序,靠位置编码补回顺序。
- 熟手:复杂度 O(seq²·d),长序列显存/算力吃紧(催生 FlashAttention、长上下文优化);
  缩放 1/√d 稳定 softmax 梯度;Q/K/V 共享或独立投影影响参数与表达力。
- 延伸:把 seq 从3调到8看注意力矩阵大小增长;去掉位置编码看模型能否区分词序。
EOF
echo "============================================================"
