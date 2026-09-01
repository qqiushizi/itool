#!/bin/bash
# ============================================================
# 实验: c.generalization
# 说明: 过/欠拟合、偏差-方差、学习曲线
# 模块: p.practise/a.ai-fundamentals  AI 基础
# ============================================================
# 【第一性原理】
# 泛化=对新数据的预测能力。误差可分解为 偏差²(模型够不够强)+方差(对不同训练集稳不稳)+噪声。
# 欠拟合:模型太弱,偏差大(训练测试都差);过拟合:模型太强,方差大(训练好测试差)。
# 学习曲线:数据越多,训练误差↑(更难全记)、测试误差↓(更接近真实),两条线收敛说明够用了。
# 本实验用不同阶多项式在重采样数据上量化偏差与方差,并画学习曲线。
# ============================================================
set -euo pipefail
if ! python3 -c "import numpy" 2>/dev/null; then
  echo "未检测到 numpy,正在安装..."; python3 -m pip install --quiet numpy || { echo "请手动: pip install numpy"; exit 1; }
fi
echo "============================================================"
echo " 实验: 偏差-方差 / 过欠拟合 / 学习曲线"
echo "============================================================"
python3 <<'PY'
import numpy as np
rng=np.random.default_rng(7)
def ftrue(x): return np.sin(2*x)
def design(x,d): return np.vstack([x**k for k in range(d+1)]).T
def fit_pred(d,xtr,ytr,xe):
    V=design(xtr,d); w=np.linalg.lstsq(V,ytr,rcond=None)[0]
    return design(xe,d)@w
# 偏差-方差分解:在 x0 点,多次重采样训练集,看预测的均值(偏差)与方差
x0=np.array([0.6]); R=200; n=20
print("【1】偏差-方差分解(在 x0=0.6, 真值≈%.3f):"%ftrue(x0)[0])
for d in [1,3,9]:
    preds=[]
    for _ in range(R):
        xtr=rng.uniform(0,np.pi,n); ytr=ftrue(xtr)+rng.standard_normal(n)*0.3
        preds.append(fit_pred(d,xtr,ytr,x0)[0])
    preds=np.array(preds); bias=(preds.mean()-ftrue(x0)[0])**2; var=preds.var()
    print(f"  阶数 d={d}: 偏差²={bias:.4f}, 方差={var:.4f}, 偏差²+方差={bias+var:.4f}")
print("  解读:d=1 太简单→偏差大、方差小(欠拟合);d=9 太复杂→偏差小、方差大(过拟合);d=3 较平衡。")

# 学习曲线:数据量增加,训练/测试误差变化(x 归一化到 [0,1] 保证数值稳定,多次平均降噪)
print("\n【2】学习曲线(阶数 d=3, 每个数据量重复5次取平均):")
def flr(u): return np.sin(2*np.pi*u)
sizes=[8,15,25,40,60,90]
for n in sizes:
    trs=[]; tes=[]
    for _ in range(5):
        xtr=rng.uniform(0,1,n); ytr=flr(xtr)+rng.standard_normal(n)*0.25
        xe=rng.uniform(0,1,300); ye=flr(xe)+rng.standard_normal(300)*0.25
        Vtr=design(xtr,3); w=np.linalg.lstsq(Vtr,ytr,rcond=None)[0]
        trs.append(np.mean((Vtr@w-ytr)**2)); tes.append(np.mean((design(xe,3)@w-ye)**2))
    print(f"  n={n:<3}: 训练MSE={np.mean(trs):.4f}  测试MSE={np.mean(tes):.4f}")
print("  解读:数据越多,训练MSE↑(更难全记住)、测试MSE↓(更接近真实规律);两条线收敛→数据已够用。")
PY
echo "============================================================"
cat <<'EOF'
【结论与进阶】
- 小白:误差=偏差²+方差+噪声;欠拟合偏差大(模型太弱),过拟合方差大(模型太强);
  数据越多测试误差越低,训练/测试曲线收敛说明数据量够了。
- 熟手:偏差-方差权衡是选模型复杂度的核心;集成(如 bagging)专门降方差,boosting 降偏差;
  学习曲线可判断"该加数据还是该减复杂度":两线都高且近→欠拟合(加复杂度),两线差距大→过拟合(加数据/正则)。
- 延伸:把噪声从0.3调到1.0看方差上限;对比 d=3 与 d=9 的偏差+方差总和。
EOF
echo "============================================================"
