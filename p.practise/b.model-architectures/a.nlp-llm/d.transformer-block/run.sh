#!/bin/bash
# ============================================================
# 实验: d.transformer-block
# 说明: 残差/LayerNorm/FFN 组装、参数量计算
# 模块: p.practise/b.model-architectures  模型架构
# ============================================================
# 【第一性原理】
# 一个 Transformer Block = 多头注意力 + 残差连接 + LayerNorm + 前馈网络(FFN)+ 残差 + LayerNorm。
# 残差 x+f(x):让梯度直通,解决深层难训练;LayerNorm:把每个样本的隐向量归一化,稳定激活分布;
# FFN:两层 MLP(升维→激活→降维),给非线性表达力(注意力只做线性混合)。
# 本实验手搭一个 block 跑前向,并精确计算各部分参数量。
# ============================================================
set -euo pipefail
if ! python3 -c "import numpy" 2>/dev/null; then
  echo "未检测到 numpy,正在安装..."; python3 -m pip install --quiet numpy || { echo "请手动: pip install numpy"; exit 1; }
fi
echo "============================================================"
echo " 实验: Transformer Block / 残差 / LayerNorm / FFN / 参数量"
echo "============================================================"
python3 <<'PY'
import numpy as np
np.set_printoptions(precision=3, suppress=True)
rng=np.random.default_rng(1)
def softmax(x):
    x=x-x.max(axis=-1,keepdims=True); e=np.exp(x); return e/e.sum(axis=-1,keepdims=True)
def layernorm(x,g,b,eps=1e-5):
    mu=x.mean(-1,keepdims=True); var=x.var(-1,keepdims=True)
    return g*(x-mu)/np.sqrt(var+eps)+b
d=8; dff=32; seq=4
X=rng.standard_normal((seq,d))
# 注意力
Wq=Wk=Wv=rng.standard_normal((d,d)); Wo=rng.standard_normal((d,d))
Q,K,V=X@Wq,X@Wk,X@Wv
A=softmax(Q@K.T/np.sqrt(d)); attn=A@V@Wo
# 残差+LayerNorm
x1=layernorm(X+attn,np.ones(d),np.zeros(d))
# FFN:升维→ReLU→降维 + 残差 + LayerNorm
W1=rng.standard_normal((d,dff)); W2=rng.standard_normal((dff,d))
ffn=np.maximum(0,x1@W1)@W2
x2=layernorm(x1+ffn,np.ones(d),np.zeros(d))
print("【1】一个 Transformer Block 前向(输入 shape=",X.shape,")")
print(f"  注意力后 + 残差 + LN → shape {x1.shape}, 均值≈{x1.mean():.3f}, 方差≈{x1.var():.3f}")
print(f"  FFN 后 + 残差 + LN → shape {x2.shape}, 均值≈{x2.mean():.3f}, 方差≈{x2.var():.3f}")
print("  解读:每经过 LN,每行被归一化到均值0方差1(再缩放),激活分布稳定,深层也能训。")

# 参数量
print("\n【2】参数量计算(d=%d, dff=%d):"%(d,dff))
attn_p=4*d*d              # Q,K,V,O 各 d×d
ln_p=2*2*d                # 两个 LN 各 γ,β
ffn_p=d*dff+dff*d
print(f"  注意力投影(QKVO): 4×{d}×{d} = {attn_p}")
print(f"  两个 LayerNorm: 2×2×{d} = {ln_p}")
print(f"  FFN(W1,W2): {d}×{dff}+{dff}×{d} = {ffn_p}")
print(f"  单 block 合计 = {attn_p+ln_p+ffn_p}  (不含 bias)")
print("  解读:FFN 升维(dff=4d 常用)是参数大头;残差/LN 几乎不加参数却关键。堆 L 层≈×L。")
PY
echo "============================================================"
cat <<'EOF'
【结论与进阶】
- 小白:一个 Block=注意力+残差+LayerNorm+FFN+残差+LayerNorm;残差让深层可训,LN 稳定分布,FFN 给非线性。
- 熟手:FFN 的 dff 常取 4d,是参数与算力大头;残差让梯度直通是深网络能训的根本;
  Post-LN vs Pre-LN 影响稳定性(现代多用 Pre-LN);参数量≈L×(4d²+8d+2d·dff)。
- 延伸:把 dff 从4d 改到2d 看参数变化;对比有/无残差时深层的梯度大小。
EOF
echo "============================================================"
