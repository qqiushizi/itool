#!/bin/bash
# ============================================================
# 实验: e.decoder-llm
# 说明: GPT 式 decoder、因果 mask、KV-cache 直觉
# 模块: p.practise/b.model-architectures  模型架构
# ============================================================
# 【第一性原理】
# GPT 是 decoder-only:自回归,每生成一个词只看它前面的词(不能偷看未来)→用因果 mask 实现。
# 因果 mask:把注意力分数的上三角(未来)设成 -∞,softmax 后权重为0。
# KV-cache:生成时把已算过的 K、V 存下来,新词只算自己的 q 去查历史 KV,避免每步重算→解码变快。
# 本实验演示因果 mask 的效果,并量化 KV-cache 带来的计算节省。
# ============================================================
set -euo pipefail
if ! python3 -c "import numpy" 2>/dev/null; then
  echo "未检测到 numpy,正在安装..."; python3 -m pip install --quiet numpy || { echo "请手动: pip install numpy"; exit 1; }
fi
echo "============================================================"
echo " 实验: Decoder-LLM / 因果 mask / 自回归 / KV-cache"
echo "============================================================"
python3 <<'PY'
import numpy as np
np.set_printoptions(precision=3, suppress=True)
rng=np.random.default_rng(2)
def softmax(x):
    x=x-x.max(axis=-1,keepdims=True); e=np.exp(x); return e/e.sum(axis=-1,keepdims=True)
seq=4; d=4
Q=rng.standard_normal((seq,d)); K=rng.standard_normal((seq,d))
scores=Q@K.T/np.sqrt(d)
# 1 无 mask vs 因果 mask
print("【1】因果 mask:禁止看未来(上三角置 -∞)")
mask=np.triu(np.ones((seq,seq),bool),1)          # 上三角=True(未来)
masked=scores.copy(); masked[mask]=-np.inf
A_full=softmax(scores); A_causal=softmax(masked)
print(f"  无mask权重(可看未来)=\n{A_full}\n  因果mask权重(只看过去)=\n{A_causal}")
print("  解读:因果mask后每个位置只加权自己和之前的词(下三角),未来权重为0→自回归生成的关键。")

# 2 自回归生成:逐步预测
print("\n【2】自回归生成:每步用已有词预测下一个(此处用随机权重示意概率):")
ctx=["BOS"]
for step in range(3):
    p=softmax(rng.standard_normal(5))
    nxt=["我","爱","猫","狗","。"][int(p.argmax())]
    ctx.append(nxt)
    print(f"  步{step}: 上下文={ctx[:-1]} → 预测 '{nxt}' (softmax argmax)")
print("  解读:每步把新词拼进上下文再预测下一个;序列越长上下文越大。")

# 3 KV-cache 节省:无cache每步重算全部,有cache只算新词的q查历史
print("\n【3】KV-cache 计算节省(seq 长度 N,逐步生成):")
for N in [64,256,1024]:
    no_cache=sum((i+1)**2*d for i in range(N))     # 无cache:每步从头重算 i+1 个token注意力 ≈ O(N³)
    with_cache=sum(i*d for i in range(1,N+1))       # 有cache:第i步只算新q查i个缓存KV ≈ O(N²)
    print(f"  N={N:<5}: 无cache≈{no_cache//1000:>7}k  有cache≈{with_cache//1000:>7}k  加速比≈{no_cache/with_cache:.0f}x")
print("  解读:无cache每步重算全部注意力(O(N²)每步×N步≈O(N³));有cache第i步只算新q查i个KV(O(N²)总)。\n  → KV-cache 把生成复杂度从 O(N³) 降到 ~O(N²),是高吞吐推理的基石。")
PY
echo "============================================================"
cat <<'EOF'
【结论与进阶】
- 小白:GPT 自回归生成,因果 mask 禁止看未来;KV-cache 把算过的 K/V 存下,新词只查不重算,生成更快。
- 熟手:无 KV-cache 解码 O(N³),有 cache O(N²);代价是显存随序列线性增长(催生 PagedAttention);
  prefill 阶段是计算密集(算所有 KV),decode 阶段是访存密集(每步读全部 KV)。
- 延伸:把 seq 调到 4096 看 KV-cache 显存压力;思考为何 decode 是访存瓶颈。
EOF
echo "============================================================"
