#!/bin/bash
# ============================================================
# 实验: a.clip
# 说明: 图文对齐、对比学习、双塔结构
# 模块: p.practise/b.model-architectures  模型架构
# ============================================================
# 【第一性原理】
# CLIP 用对比学习把图像和文本对齐到同一向量空间:双塔分别编码图和文,
# 让"匹配的图文对"相似度高、"不匹配的"相似度低。训练目标是让对角线(配对)的相似度最大。
# 学好后,任意给一张图和一句文,算它们向量余弦相似度就知道"匹不匹配"——零样本分类的基础。
# 本实验模拟图文嵌入,算相似度矩阵,并用 InfoNCE 对比损失展示对齐过程。
# ============================================================
set -euo pipefail
if ! python3 -c "import numpy" 2>/dev/null; then
  echo "未检测到 numpy,正在安装..."; python3 -m pip install --quiet numpy || { echo "请手动: pip install numpy"; exit 1; }
fi
echo "============================================================"
echo " 实验: CLIP / 图文对齐 / 对比学习 / 双塔"
echo "============================================================"
python3 <<'PY'
import numpy as np
np.set_printoptions(precision=3, suppress=True)
rng=np.random.default_rng(0)
def softmax(x):
    x=x-x.max(axis=-1,keepdims=True); e=np.exp(x); return e/e.sum(axis=-1,keepdims=True)
def norm(a): return a/np.linalg.norm(a,axis=-1,keepdims=True)
# 4 对图文:同索引=配对。让配对的嵌入相近(加共享信号),不配对的远
n,d=4,8
base=rng.standard_normal((n,d))
img=norm(base+0.5*rng.standard_normal((n,d)))     # 图塔输出
txt=norm(base+0.5*rng.standard_normal((n,d)))     # 文塔输出(与对应图共享base→相近)
sim=img@txt.T
print("【1】双塔:图塔、文塔分别编码,算相似度矩阵(对角线=配对)")
print(f"  相似度矩阵(行=图,列=文)=\n{sim}")
diag=np.diag(sim).mean(); off=sim[~np.eye(n,dtype=bool)].mean()
print(f"  对角线均值(配对)={diag:.3f}  非对角均值(不配对)={off:.3f}")
print("  解读:配对的相似度更高→图文已大致对齐;CLIP 训练就是把这个差距拉得更大。")

# 2 InfoNCE 对比损失:最大化对角线
tau=0.1
print("\n【2】InfoNCE 对比损失(温度τ=%.2f):对每个图,让配对文本的相似度在所有文本中占比最大"%tau)
loss_i=0; loss_t=0
for i in range(n):
    p=softmax(sim[i]/tau); loss_i-=np.log(p[i])
    q=softmax(sim[:,i]/tau); loss_t-=np.log(q[i])
print(f"  图→文 损失={loss_i/n:.3f}  文→图 损失={loss_t/n:.3f}  (越小越对齐)")
print("  解读:损失 = 配对相似度在所有候选里的负对数概率。训练把它压低=把配对推向高相似、不配对拉远。")

# 3 零样本分类
print("\n【3】零样本分类:给一张新图,在多个文本类别中找最像的")
new_img=norm(base[2]+0.3*rng.standard_normal(d))   # 新图接近第2类的图
classes=["一只猫","一条狗","一辆车","一座房"]; txt_c=norm(base+0.5*rng.standard_normal((n,d)))
s=new_img@txt_c.T
print(f"  各类别相似度 = {[f'{c}:{v:.2f}' for c,v in zip(classes,s)]}")
print(f"  预测类别 = {classes[int(s.argmax())]}")
print("  解读:不需要专门训练分类器——把类别写成文本,比谁跟图最像即可分类,这就是 CLIP 零样本能力。")
PY
echo "============================================================"
cat <<'EOF'
【结论与进阶】
- 小白:CLIP 用双塔分别编码图文,对比学习让配对相似度高、不配对低;学好后可比图文相似度做零样本分类。
- 熟手:InfoNCE 本质是把对比变成 N 路分类(配对为正、其余为负);batch 内负例,大 batch 更好;
  温度 τ 控制分布锐度;双塔推理快(可离线编码),但精细理解不如融合式。
- 延伸:把 τ 从0.1调到1.0看损失变化;减少配对共享信号看对角线是否还突出。
EOF
echo "============================================================"
