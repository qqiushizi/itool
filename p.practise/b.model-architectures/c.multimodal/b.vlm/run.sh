#!/bin/bash
# ============================================================
# 实验: b.vlm
# 说明: 视觉编码器 + LLM 融合、cross-attention
# 模块: p.practise/b.model-architectures  模型架构
# ============================================================
# 【第一性原理】
# VLM(视觉语言模型)=看图 + 说话。先用视觉编码器(如 ViT)把图变成一组视觉 token,
# 再把这组视觉 token 注入 LLM。注入方式分两类:
#  ① 融合式(cross-attention):文本 token 当 Q,视觉 token 当 K/V,让文本"看"图像;
#  ② 前缀式:把视觉 token 当作文本序列前缀直接拼进去(如 LLaVA)。
# 本实验演示 cross-attention 融合:文本查询去关注图像的哪些区域。
# ============================================================
set -euo pipefail
if ! python3 -c "import numpy" 2>/dev/null; then
  echo "未检测到 numpy,正在安装..."; python3 -m pip install --quiet numpy || { echo "请手动: pip install numpy"; exit 1; }
fi
echo "============================================================"
echo " 实验: VLM / 视觉编码 + LLM / cross-attention 融合"
echo "============================================================"
python3 <<'PY'
import numpy as np
np.set_printoptions(precision=3, suppress=True)
rng=np.random.default_rng(1)
def softmax(x):
    x=x-x.max(axis=-1,keepdims=True); e=np.exp(x); return e/e.sum(axis=-1,keepdims=True)
d=8; n_img=5; n_txt=3
# 1 视觉编码:图 → 5 个视觉 token(模拟 ViT patch 输出)
vis=rng.standard_normal((n_img,d))
print("【1】视觉编码器把图变成视觉 token 序列:",vis.shape,"(类比 ViT 的 patch 嵌入)")
# 2 文本 token
txt=rng.standard_normal((n_txt,d))
print(f"  文本 token 序列:{txt.shape}  (要生成的文本/问句的嵌入)")
# 3 cross-attention:文本当Q,视觉当K/V → 文本去"看"图像
Wq=Wk=Wv=rng.standard_normal((d,d))
Q=txt@Wq; K=vis@Wk; V=vis@Wv
A=softmax(Q@K.T/np.sqrt(d)); fused=A@V
print(f"\n【2】cross-attention 融合:文本Q × 视觉KV → 每个文本token关注图像区域")
print(f"  注意力权重(行=文本,列=视觉token)=\n{A.round(3)}")
print(f"  融合后文本表示 shape={fused.shape}  (注入了图像信息)")
print("  解读:每个文本 token 按相关度加权图像区域,生成'看着图说话'的表示。")

# 4 前缀式对比
print("\n【3】另一种融合:前缀式(LLaVA 风格)——把视觉 token 当文本前缀拼进去")
seq=np.vstack([vis,txt])
print(f"  拼接后序列 shape={seq.shape}  (前{n_img}是视觉token,后{n_txt}是文本token)")
print("  解读:视觉 token 和文本 token 共享同一个 Transformer,模型自己学如何用视觉信息。两种方式各有取舍。")
PY
echo "============================================================"
cat <<'EOF'
【结论与进阶】
- 小白:VLM=视觉编码器+LLM;视觉token 通过 cross-attention(文本Q查视觉KV)或前缀拼接注入 LLM,实现看图说话。
- 熟手:cross-attention 融合(flamingo)解耦视觉与语言,但参数多;前缀式(LLaVA)简单、复用 LLM,是主流;
  视觉 token 数量影响成本,可用 resampler/池化压缩;对齐质量取决于图文对数据。
- 延伸:增加视觉 token 数看注意力变化;对比前缀式与 cross-attention 的参数量。
EOF
echo "============================================================"
