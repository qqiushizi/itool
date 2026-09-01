#!/bin/bash
# ============================================================
# 实验: a.word-embeddings
# 说明: one-hot→word2vec→上下文向量、相似度演变
# 模块: p.practise/b.model-architectures  模型架构
# ============================================================
# 【第一性原理】
# one-hot 把每个词编成独热向量:维度=词表大小、任意两词正交(全不相似)、稀疏且无语义。
# word2vec 让模型在"预测上下文"中学会把意思相近的词映射到相近的稠密向量。
# 稠密向量的好处:维度低、能算相似度(余弦)、还涌现"类比关系"king-man+woman≈queen。
# 本实验对比 one-hot 与稠密嵌入,并用向量算术演示语义类比。
# ============================================================
set -euo pipefail
if ! python3 -c "import numpy" 2>/dev/null; then
  echo "未检测到 numpy,正在安装..."; python3 -m pip install --quiet numpy || { echo "请手动: pip install numpy"; exit 1; }
fi
echo "============================================================"
echo " 实验: 词向量 / one-hot / 稠密嵌入 / 语义类比"
echo "============================================================"
python3 <<'PY'
import numpy as np
np.set_printoptions(precision=3, suppress=True)
def cos(a,b): return float(a@b/(np.linalg.norm(a)*np.linalg.norm(b)+1e-12))
vocab=["king","queen","man","woman"]
# 1 one-hot:任意两词正交
print("【1】one-hot 编码(维度=词表大小,任意两词正交、无相似度):")
oh=np.eye(4)
for i in range(4):
    print(f"  {vocab[i]:<6} = {oh[i].astype(int).tolist()}")
print(f"  cos(king,queen)={cos(oh[0],oh[1]):.1f}  cos(king,man)={cos(oh[0],oh[2]):.1f}  (全=0,看不出谁更像谁)")
print("  解读:one-hot 把每个词隔开,没有任何'相似度'信息,且维度随词表膨胀。")

# 2 稠密嵌入:手设2维空间,维0≈性别、维1≈王室
print("\n【2】稠密嵌入(2维,学到的语义空间):")
emb={"king":np.array([0.9,0.8]),"queen":np.array([0.1,0.8]),
     "man":np.array([0.9,0.1]),"woman":np.array([0.1,0.1])}
for w in vocab: print(f"  {w:<6} = {emb[w].tolist()}")
print(f"  cos(king,queen)={cos(emb['king'],emb['queen']):.3f}  cos(king,man)={cos(emb['king'],emb['man']):.3f}")
print("  解读:king与man同性别(维0≈0.9)→更近;king与queen同王室(维1≈0.8)→也近。稠密向量能表达'像不像'。")

# 3 类比:king-man+woman ≈ queen
print("\n【3】语义类比:king - man + woman ≈ ?")
vec=emb["king"]-emb["man"]+emb["woman"]
print(f"  king-man+woman = {vec.round(3).tolist()}")
best=min(vocab,key=lambda w:np.linalg.norm(emb[w]-vec))
print(f"  最近词 = {best}  (queen 的向量 = {emb['queen'].tolist()})")
print("  解读:减去'男性'加上'女性'语义维度,king→queen,向量算术捕捉到了词与词的关系。\n  → 这就是词嵌入著名特性:方向≈语义关系(性别、时态、首都-国家…)。")
PY
echo "============================================================"
cat <<'EOF'
【结论与进阶】
- 小白:one-hot 稀疏且任意词都"不像";稠密词向量维度低、能算相似度,还支持类比(king-man+woman≈queen)。
- 熟手:word2vec(skip-gram/CBOW)靠"预测上下文"学嵌入,中心词与上下文词向量靠近;
  GloVe 用共现统计;现代 LLM 的 token 嵌入是这思想的超大规模版,且在训练中持续更新。
- 延伸:把嵌入维数从2调大看能否表达更多关系;思考为何"上下文预测"能学出语义。
EOF
echo "============================================================"
