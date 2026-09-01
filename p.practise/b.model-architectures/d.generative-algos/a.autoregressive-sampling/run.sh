#!/bin/bash
# ============================================================
# 实验: a.autoregressive-sampling
# 说明: 贪心/beam/top-k/top-p/temperature 解码对比
# 模块: p.practise/b.model-architectures  模型架构
# ============================================================
# 【第一性原理】
# 自回归生成每步输出一个 logits,怎么从中选下一个 token 决定了生成风格:
#  贪心:每步取最大→确定、易重复;temperature:除以 T 软化/锐化分布,T大更随机;
#  top-k:只在前 k 个里选,砍掉长尾;top-p(核采样):选累计概率达 p 的最小集合,动态截断;
#  beam:同时维护 B 条序列,选综合概率最高的→偏高质量但少多样性。
# 本实验用一组 logits 演示各策略如何改变下一个 token 的分布与选择。
# ============================================================
set -euo pipefail
if ! python3 -c "import numpy" 2>/dev/null; then
  echo "未检测到 numpy,正在安装..."; python3 -m pip install --quiet numpy || { echo "请手动: pip install numpy"; exit 1; }
fi
echo "============================================================"
echo " 实验: 解码策略 / 贪心 / temperature / top-k / top-p / beam"
echo "============================================================"
python3 <<'PY'
import numpy as np
np.set_printoptions(precision=3, suppress=True)
rng=np.random.default_rng(0)
vocab=["猫","狗","鱼","鸟","车","房","的","了","呢","哈"]
logits=np.array([3.0,2.8,1.5,1.2,-2,-2,0.5,0.3,0.1,-0.5])
def softmax(x):
    x=x-x.max(); e=np.exp(x); return e/e.sum()
p=softmax(logits)
print("【1】原始分布:logits → softmax 概率")
for w,l,pr in zip(vocab,logits,p):
    print(f"  {w}: logit={l:+.1f}  p={pr:.3f}")
print(f"  贪心(argmax)→ '{vocab[p.argmax()]}'  (每步最大,确定但易重复)")

# 2 temperature
print("\n【2】temperature:除以 T,T>1 更随机,T<1 更确定")
for T in [0.5,1.0,2.0]:
    pt=softmax(logits/T)
    print(f"  T={T}: top1概率={pt.max():.3f}, 分布={pt.round(3).tolist()}")
print("  解读:T 小→分布尖锐(趋近贪心);T 大→分布平坦(更随机多样)。")

# 3 top-k / top-p
print("\n【3】top-k(只留前k)/ top-p(累计概率达p的最小集合):")
k=3; idx_k=np.argsort(p)[::-1][:k]; pk=p[idx_k]; pk=pk/pk.sum()
print(f"  top-k=3 候选={[vocab[i] for i in idx_k]} 重归一化={pk.round(3).tolist()}")
order=np.argsort(p)[::-1]; cum=np.cumsum(p[order]); cut=np.searchsorted(cum,0.9)+1
idx_p=order[:cut]; pp=p[idx_p]; pp=pp/pp.sum()
print(f"  top-p=0.9 候选={[vocab[i] for i in idx_p]} (累计到0.9即止) 重归一化={pp.round(3).tolist()}")
print("  解读:top-k 固定数量截断;top-p 动态截断(概率集中时少选、分散时多选),更自适应。")

# 4 beam search
print("\n【4】beam search:同时维护 B 条候选序列,选累计概率最高者")
B=2
seqs=[("",1.0)]
for step in range(2):
    cand=[]
    for s,score in seqs:
        for w,pr in zip(vocab,p):
            cand.append((s+w, score*pr))
    cand.sort(key=lambda x:-x[1]); seqs=cand[:B]
print(f"  beam=2, 2步后的候选: {[(s,round(float(sc),4)) for s,sc in seqs]}")
print("  解读:beam 保留多条高概率路径,比贪心更稳,但多样性低、计算量×B。")
PY
echo "============================================================"
cat <<'EOF'
【结论与进阶】
- 小白:贪心确定但易重复;temperature 调随机;top-k/top-p 砍长尾防胡言;beam 维护多条路径求高质量。
- 熟手:实际常组合 top-p + temperature;beam 适合翻译/摘要(求准),采样适合开放生成(求多样);
  repetition penalty 抑制重复;LLM 推理里这些都在 logits 后处理阶段完成。
- 延伸:把 logits 改得更平坦看 top-p 候选数变化;调 beam 宽度看质量与速度权衡。
EOF
echo "============================================================"
