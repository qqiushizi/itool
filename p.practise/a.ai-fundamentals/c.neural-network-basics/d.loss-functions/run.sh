#!/bin/bash
# ============================================================
# 实验: d.loss-functions
# 说明: MSE/CE/Focal/对比损失 推导与曲线对比
# 模块: p.practise/a.ai-fundamentals  AI 基础
# ============================================================
# 【第一性原理】
# 损失函数衡量"预测离目标有多远",它的梯度告诉模型往哪改。不同任务用不同损失:
#  回归用 MSE(残差平方),梯度=2(预测-目标),对大误差敏感;
#  分类用交叉熵 CE=-log(p_y),配 softmax 后梯度=p-onehot,形式极简;
#  类别不平衡用 Focal=-(1-p)^γ log(p),给难样本加权、易样本降权;
#  嵌入学习用对比损失,拉近相似对、推开不相似对。本实验把四者的曲线与梯度都画出来。
# ============================================================
set -euo pipefail
if ! python3 -c "import numpy" 2>/dev/null; then
  echo "未检测到 numpy,正在安装..."; python3 -m pip install --quiet numpy || { echo "请手动: pip install numpy"; exit 1; }
fi
echo "============================================================"
echo " 实验: 损失函数 / MSE / 交叉熵 / Focal / 对比损失"
echo "============================================================"
python3 <<'PY'
import numpy as np
np.set_printoptions(precision=4, suppress=True)

# 1 MSE(回归)
print("【1】MSE 均方误差:预测-目标的平方均值")
pred=np.array([1.2, 2.1, 0.3]); target=np.array([1.0, 2.0, 0.5])
mse=np.mean((pred-target)**2); grad=2*(pred-target)/len(pred)
print(f"  pred={pred.tolist()}, target={target.tolist()}")
print(f"  MSE={mse:.4f}, 梯度(每个样本)={grad.round(4).tolist()}")
print("  解读:残差越大损失增长越快(平方),梯度=2(预测-目标)把模型往目标推。→ 对离群点很敏感。")

# 2 交叉熵 + softmax
print("\n【2】交叉熵(分类):softmax 后取 -log(p_正确类),梯度=p-onehot")
logits=np.array([2.0, 1.0, 0.1]); logits-=logits.max()
p=np.exp(logits)/np.exp(logits).sum(); y_onehot=np.array([1,0,0])
ce=-np.sum(y_onehot*np.log(p+1e-12)); grad=p-y_onehot
print(f"  logits={logits.tolist()} → softmax p={p.round(4).tolist()}")
print(f"  正确类=0, CE={ce:.4f}, 梯度=p-onehot={grad.round(4).tolist()}")
print("  解读:正确类概率越接近1,CE→0;越接近0,CE→∞(强烈惩罚错判)。配 softmax 后梯度形式极简=p-y。")

# 3 Focal loss(类别不平衡)
print("\n【3】Focal Loss:给难样本加权,压低易样本的损失贡献")
p_pos=np.linspace(0.05,0.95,10); gamma=2.0
ce_loss=-np.log(p_pos)
focal_loss=-(1-p_pos)**gamma*np.log(p_pos)
print(f"  p(预测正确概率)={p_pos.round(2).tolist()}")
print(f"  CE   ={ce_loss.round(3).tolist()}")
print(f"  Focal={focal_loss.round(3).tolist()}")
print("  解读:p=0.9(易样本)时 Focal≈CE×(0.1)^2,被大幅压低;p=0.1(难样本)时 Focal≈CE×0.81,基本保留。\n  → 类别极不平衡时,大量易样本不会淹没少数难样本的学习信号。")

# 4 对比损失(嵌入学习)
print("\n【4】对比损失:相似对拉近、不相似对推开(到 margin 外)")
def dist(a,b): return np.linalg.norm(a-b)
eA=np.array([1.0,0.0]); eP=np.array([1.1,0.1]); eN=np.array([0.0,1.0]); margin=1.0
dpos=dist(eA,eP); dneg=dist(eA,eN)
L_pos=0.5*dpos**2                              # 相似对:y=0 → 拉近
L_neg=0.5*max(0,margin-dneg)**2               # 不相似对:y=1 → 推开到 margin 外
print(f"  锚点 A={eA.tolist()}, 正例 P={eP.tolist()} 距离={dpos:.3f} → 损失={L_pos:.4f}(要小)")
print(f"  锚点 A={eA.tolist()}, 负例 N={eN.tolist()} 距离={dneg:.3f}, margin={margin} → 损失={L_neg:.4f}(要小)")
print("  解读:正例距离越小损失越小(拉近);负例距离≥margin 时损失=0(已足够远)。\n  → 对比学习(如 CLIP/SimCLR)就是靠这种损失把语义相似的表示聚到一起。")
PY
echo "============================================================"
cat <<'EOF'
【结论与进阶】
- 小白:回归用 MSE(平方误差,对大误差敏感);分类用交叉熵(配 softmax,梯度=p-y);
  类别不平衡用 Focal(给难样本加权);嵌入用对比损失(拉近相似、推开不相似)。
- 熟手:CE 配 softmax 的梯度恰好 p-y 是数值上最优雅的组合;Focal 的 γ 调"聚焦"强度(常用2);
  对比损失的关键是 margin 和负样本采样,InfoNCE 用大量负例把对比变成分类。
- 延伸:把 Focal 的 γ 从 2 改到 0(退化为 CE)看损失曲线变化;给对比损失换不同 margin 看推开力度。
EOF
echo "============================================================"
