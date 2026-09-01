#!/bin/bash
# ============================================================
# 实验: a.perceptron-mlp
# 说明: 感知机→MLP 前向/反向手写实现
# 模块: p.practise/a.ai-fundamentals  AI 基础
# ============================================================
# 【第一性原理】
# 感知机:z=w·x+b 再过阶跃,只能画一条直线→只能解线性可分问题。XOR 切不开。
# 解法:加隐藏层(MLP)+非线性激活,把输入"揉"到线性可分的表示,再线性分类。
# 学习靠反向传播:从输出往回,用链式法则算每层权重该改多少,再梯度下降更新。
# 本实验:感知机在 XOR 上失败 → 手写 MLP 前向 → 手算反向(并数值验证)→ 训练学通 XOR。
# ============================================================
set -euo pipefail
if ! python3 -c "import numpy" 2>/dev/null; then
  echo "未检测到 numpy,正在安装..."; python3 -m pip install --quiet numpy || { echo "请手动: pip install numpy"; exit 1; }
fi
echo "============================================================"
echo " 实验: 感知机 → MLP / 前向反向手写 / XOR 之困"
echo " 每步:【做什么&为什么】→【结果】→【结果解读】"
echo "============================================================"
python3 <<'PY'
import numpy as np
np.set_printoptions(precision=4, suppress=True)
def sig(z): return 1/(1+np.exp(-z))
def block(i,t,title,why,res,mean):
    print(f"\n{'='*58}\n【步骤 {i}/{t}】{title}\n{'='*58}")
    print(f"--- 做什么 & 为什么 ---\n  {why}\n--- 结果 ---\n  {res}\n--- 结果解读 ---\n  {mean}")
T=4
X=np.array([[0,0],[0,1],[1,0],[1,1]],float); y=np.array([0,1,1,0])

# 1 单层感知机(实现 AND 门)在 XOR 上失败
w=np.array([0.5,0.5]); b=-0.7
pred=(X@w+b>0).astype(int); acc=(pred==y).mean()
block(1,T,"单层感知机:一条直线分类",
 "用 z=w·x+b>0 当 AND 门,看它在 XOR(相同=0,不同=1)上准不准。",
 f"权重 w={w.tolist()}, b={b}\n  预测={pred.tolist()}  真值={y.tolist()}\n  准确率={acc*100:.0f}%",
 "XOR 四个点用任何一条直线都切不开→线性模型注定失败(准确率卡在 25%/50%)。\n  → 这就是隐藏层存在的理由:先把输入变成线性可分的表示。")

# 2 MLP(2-2-1)前向,未训练权重
rng=np.random.default_rng(1)
W1=rng.standard_normal((2,2)); b1=np.zeros(2); W2=rng.standard_normal((2,1)); b2=np.zeros(1)
h=sig(X@W1+b1); o=sig(h@W2+b2)
block(2,T,"MLP 前向:一层隐藏层把数据变换",
 "结构 2→2→1。隐藏层用 sigmoid 造非线性特征,输出层线性合成。权重还没训练。",
 f"隐藏层 h(4×2)=\n{h}\n  输出={o.ravel().round(3).tolist()}\n  (都≈0.5,说明还没学会,但'非线性变换'的结构已就位)",
 "输出全在 0.5 附近=没学到东西,但关键已经具备:多了一层非线性,表示能力比感知机强。\n  → 接下来只要用梯度下降把权重调对即可(感知机连'调对的可能'都没有)。")

# 3 反向传播手算 + 数值梯度验证(对单个样本 x=[1,1], t=1)
x0=X[3]; t0=1.0
z1=x0@W1+b1; a1=sig(z1); z2=a1@W2+b2; a2=sig(z2); dL=a2-t0
g2=dL*a2*(1-a2)            # dL/dz2
dW2=np.outer(a1,g2)        # dL/dW2 = a1 ⊗ g2
g1=g2@W2.T*a1*(1-a1)       # 误差传回隐藏层并乘 σ'
dW1=np.outer(x0,g1)        # dL/dW1 = x0 ⊗ g1
# 数值梯度:对每个参数做有限差分,与手算解析梯度对拍
e=1e-6
def lossf(W1,b1,W2,b2):
    a=sig(sig(x0@W1+b1)@W2+b2); return (0.5*(a-t0)**2).item()   # 返回标量
def num_grad(arr,which):
    base=[W1.copy(),b1.copy(),W2.copy(),b2.copy()]; shp=base[which].shape
    flat=base[which].ravel().copy(); ng=np.zeros_like(flat)
    for k in range(flat.size):
        p=flat.copy(); p[k]+=e; base[which]=p.reshape(shp); lp=lossf(*base)
        p[k]-=2*e; base[which]=p.reshape(shp); lm=lossf(*base); ng[k]=(lp-lm)/(2*e)
    return ng.reshape(arr.shape)
ng1=num_grad(dW1,0); ng2=num_grad(dW2,2)
err=max(np.abs(dW1-ng1).max(), np.abs(dW2-ng2).max())
block(3,T,"反向传播:链式法则手算 + 数值验证",
 "取样本 x=[1,1],t=1。前向算预测,再用链式逐层求梯度,最后与有限差分数值梯度对比。",
 f"前向 a2={a2.item():.4f}\n  解析 dW1={dW1.ravel().round(4).tolist()}\n  数值 dW1={ng1.ravel().round(4).tolist()}\n  解析 dW2={dW2.ravel().round(4).tolist()}\n  数值 dW2={ng2.ravel().round(4).tolist()}\n  最大误差={err:.2e}",
 "解析梯度与数值梯度几乎一致(误差≈1e-9)→手算的链式法则是对的。\n  → backprop 本质:把复合函数导数从输出往输入逐层拆开;PyTorch autograd 干的就是这件事。")

# 4 训练 MLP 学通 XOR
lr=1.5
for ep in range(3000):
    h=sig(X@W1+b1); o=sig(h@W2+b2); dL=o-y.reshape(-1,1)
    g2=dL*o*(1-o); dW2=h.T@g2; db2=g2.sum(0)
    g1=g2@W2.T*h*(1-h); dW1=X.T@g1; db1=g1.sum(0)
    W2-=lr*dW2; b2-=lr*db2; W1-=lr*dW1; b1-=lr*db1
acc=((o.ravel()>0.5).astype(int)==y).mean()
block(4,T,"训练 MLP:梯度下降学 XOR",
 "用第3步手写的梯度,跑 3000 步梯度下降,看 MLP 能否学通。",
 f"最终输出={o.ravel().round(3).tolist()}\n  预测={(o.ravel()>0.5).astype(int).tolist()}  真值={y.tolist()}\n  准确率={acc*100:.0f}%",
 "MLP 学通了 XOR→单层做不到的事,加一层隐藏层+非线性就做到了。\n  → 深度的意义:每多一层把数据再揉一次,直到线性可分;激活必须非线性,否则多层=单层。")
PY
echo "============================================================"
cat <<'EOF'
【结论与进阶】
- 小白:感知机只能画一条直线,XOR 切不开;加隐藏层(MLP)+非线性激活就能学会;
  反向传播=从输出往回用链式法则算每层权重该改多少,再梯度下降更新。
- 熟手:激活必须非线性否则多层退化成单层;MLP 万能逼近定理保证足够宽能拟合任意连续函数;
  实际用 autograd 自动求导,但会手算 backprop 是调试梯度问题(梯度爆炸/消失)的前提。
- 延伸:把隐藏层宽度改成 1 看是否还学得会;把激活换成线性看 MLP 是否退化成感知机。
EOF
echo "============================================================"
