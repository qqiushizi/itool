#!/bin/bash
# ============================================================
# 实验: a.tokenization
# 说明: BPE/WordPiece/SentencePiece 分词实验
# 模块: p.practise/c.llm-training  LLM 训练
# ============================================================
# 【第一性原理】
# 分词把文本切成 token(模型的最小单位)。词表大小固定下,要在"罕见词能拆开"和"常见词整体保留"间平衡。
# BPE:从字符起步,反复合并最高频的相邻对,直到词表满→能处理未登录词(拆成子词)。
# WordPiece:类似 BPE,但按"合并后似然增益最大"而非频率选对(BERT 用)。
# SentencePiece:把空格当普通字符,支持任意语言、可逆还原(GPT/T5 用)。
# 本实验手写一个迷你 BPE,演示合并过程,并对比不同分词的 token 数。
# ============================================================
set -euo pipefail
if ! python3 -c "import numpy" 2>/dev/null; then
  echo "未检测到 numpy,正在安装..."; python3 -m pip install --quiet numpy || { echo "请手动: pip install numpy"; exit 1; }
fi
echo "============================================================"
echo " 实验: 分词 / BPE / WordPiece / SentencePiece"
echo "============================================================"
python3 <<'PY'
import numpy as np
from collections import Counter
def get_pairs(vocab):
    pairs=Counter()
    for word,freq in vocab.items():
        sym=word.split()
        for i in range(len(sym)-1): pairs[sym[i],sym[i+1]]+=freq
    return pairs
def bpe_merge(vocab,n_merges):
    merges=[]
    for _ in range(n_merges):
        pairs=get_pairs(vocab)
        if not pairs: break
        best=max(pairs,key=pairs.get); merges.append(best)
        a,b=best; new=a+b
        vocab={k.replace(a+" "+b,new):v for k,v in vocab.items()}
    return merges
# 语料:单词拆成字符(空格分隔),频率=出现次数
corpus="low low low low low lower lower newest newest newest newest newest newest newest widest widest"
vocab={}
for w in corpus.split():
    vocab[" ".join(list(w))+"_"]=vocab.get(" ".join(list(w))+"_",0)+1   # _ 表示词尾
print("【1】迷你 BPE:从字符起步,反复合并最高频对")
print(f"  初始(字符级)词表片段示例: {dict(list(vocab.items())[:3])}")
merges=bpe_merge(dict(vocab),10)
print(f"  合并顺序(前6): {merges[:6]}")
print("  解读:BPE 把高频字符对(l+o→lo,lo+w→low…)逐步合并成子词,直到词表满。高频词整体保留,罕见词可拆成学过的子词。")

# 2 分词结果对比
def apply_bpe(word,merges):
    syms=list(word)+["_"]
    for a,b in merges:
        i=0
        while i<len(syms)-1:
            if syms[i]==a and syms[i+1]==b: syms[i:i+2]=[a+b]; 
            else: i+=1
    return syms
print("\n【2】BPE 分词结果(词表大小=%d):"%len(merges))
for w in ["lowest","newer","widely","unseen"]:
    toks=apply_bpe(w,merges)
    print(f"  {w:<8} → {toks}  ({len(toks)} tokens)")
print("  解读:'lowest'被拆成学过的 low+est;'unseen'部分能拆、部分留字符→能处理未登录词。")

# 3 词表大小权衡
print("\n【3】词表大小权衡(用同一语料测不同 merges 数):")
for nm in [3,6,10,15]:
    m=bpe_merge(dict(vocab),nm)
    avg=np.mean([len(apply_bpe(w,m)) for w in ["lowest","newer","widely"]])
    print(f"  merges={nm:<3}(词表≈{len(m)+len(set('lowertsnwid_'))}): 平均 {avg:.1f} tokens/词")
print("  解读:词表越大→常见词整体保留→token 数少、序列短;但词表大→嵌入层参数多、罕见 token 学不充分。")
PY
echo "============================================================"
cat <<'EOF'
【结论与进阶】
- 小白:BPE 从字符起步反复合并高频对,既能保留常见词、又能把罕见词拆成子词;WordPiece 按似然增益合并;SentencePiece 跨语言可逆。
- 熟手:词表大小权衡序列长度与嵌入参数;BBPE 在字节级避免未登录;GPT 用 BBPE、BERT 用 WordPiece、T5/Llama 用 SentencePiece;
  分词器决定 token 数,直接影响训练成本与多语言能力。
- 延伸:把 merges 从10调到30看 token 数下降;换中文语料看 SentencePiece 优势。
EOF
echo "============================================================"
