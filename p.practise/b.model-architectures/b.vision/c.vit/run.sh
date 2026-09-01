#!/bin/bash
# ============================================================
# 实验: c.vit
# 说明: 图像分块→Transformer、cls token
# 模块: p.practise/b.model-architectures  模型架构
# ============================================================
# 【第一性原理】
# ViT 把 Transformer 从文本搬到图像:不逐像素卷积,而是把图切成固定大小的 patch(如16×16),
# 每个 patch 展平+线性投影成一个"词向量",再加一个可学习的 [CLS] token 汇总全局信息,
# 再加位置编码,然后送进标准 Transformer。图像分类读 [CLS] 的输出即可。
# 本实验演示:分块 → patch 嵌入 → 加 CLS 与位置编码 → 注意力聚合。
# ============================================================
set -euo pipefail
if ! python3 -c "import numpy" 2>/dev/null; then
  echo "未检测到 numpy,正在安装..."; python3 -m pip install --quiet numpy || { echo "请手动: pip install numpy"; exit 1; }
fi
echo "============================================================"
echo " 实验: ViT / 分块嵌入 / CLS token / 位置编码"
echo "============================================================"
python3 <<'PY'
import numpy as np
np.set_printoptions(precision=3, suppress=True)
rng=np.random.default_rng(0)
def softmax(x):
    x=x-x.max(axis=-1,keepdims=True); e=np.exp(x); return e/e.sum(axis=-1,keepdims=True)

# 1 分块:8×8 图 → 4 个 4×4 patch
img=rng.standard_normal((8,8))
H=W=8; P=4; n=(H//P)*(W//P)
patches=[]
for i in range(H//P):
    for j in range(W//P):
        patches.append(img[i*P:(i+1)*P,j*P:(j+1)*P].flatten())
patches=np.array(patches)
print(f"【1】分块:{H}×{W} 图 → {n} 个 {P}×{P} patch")
print(f"  每个 patch 展平后维度 = {patches.shape[1]} (= {P}×{P})")
print(f"  patch 序列数 = {n}  (类比文本的 token 数)")

# 2 patch 嵌入:线性投影到 d 维
d=8
E=rng.standard_normal((patches.shape[1],d))
tok=patches@E
print(f"\n【2】patch 嵌入:线性投影 {patches.shape[1]}→{d},得到 token 矩阵 shape={tok.shape}")
print("  解读:每个 patch 变成 d 维向量,就像词嵌入把 token 变成向量。")

# 3 加 CLS token + 位置编码
cls=rng.standard_normal((1,d))
seq=np.vstack([cls,tok])
pos=rng.standard_normal(seq.shape)
seq=seq+pos
print(f"\n【3】加 [CLS] token(1个)+ 位置编码 → 序列 shape={seq.shape}")
print("  解读:[CLS] 是一个可学习向量,放最前面,经注意力聚合所有 patch 信息后用于分类;位置编码告诉模型每个 patch 来自图上哪个位置。")

# 4 注意力让 CLS 聚合
Q=seq@rng.standard_normal((d,d)); K=seq@rng.standard_normal((d,d)); V=seq@rng.standard_normal((d,d))
A=softmax(Q@K.T/np.sqrt(d)); out=A@V
print(f"\n【4】自注意力后 CLS(第0行)对各方块的关注度 = {np.round(A[0],3).tolist()}")
print("  解读:CLS 通过注意力加权所有 patch,关注度高的 patch 贡献更多→它学到了'看哪里'对分类有用。")
PY
echo "============================================================"
cat <<'EOF'
【结论与进阶】
- 小白:ViT 把图切成 patch 当 token,投影成向量,加 CLS 和位置编码后送进 Transformer;分类读 CLS 输出。
- 熟手:patch 大小权衡序列长度与分辨率(16×16 常用);CLS 不是必须,也可全局平均池化;
  ViT 缺少卷积归纳偏置,需大数据预训练才能超越 CNN;Swin 用窗口注意力降复杂度。
- 延伸:把 patch 从4×4改到2×2看序列长度翻倍;思考 ViT 为何小数据上不如 CNN。
EOF
echo "============================================================"
