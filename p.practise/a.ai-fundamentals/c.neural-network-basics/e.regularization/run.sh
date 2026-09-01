#!/bin/bash
# ============================================================
# 实验: e.regularization
# 说明: L1/L2/Dropout/BatchNorm 过拟合抑制效果
# 模块: p.practise/a.ai-fundamentals  AI 基础
# ============================================================
# 【第一性原理】
# 模型太强+数据太少 → 死记训练点(过拟合),对新数据很差。正则化限制模型复杂度:
#  L1=|w| 之和→稀疏(很多权重归零,自动选特征);L2=w² 之和→权重变小(平滑,岭回归);
#  Dropout 训练时随机丢弃神经元→逼网络不依赖单一路径≈隐式集成;
#  BatchNorm 把每层激活归一化到均值0方差1再缩放→稳定训练、允许更大学习率。
# 本实验用多项式拟合看过拟合,再用 L2/Dropout/BN 的思想逐一抑制(数据归一化保证数值稳定)。
# ============================================================
set -euo pipefail
if ! python3 -c "import numpy" 2>/dev/null; then
  echo "未检测到 numpy,正在安装..."; python3 -m pip install --quiet numpy || { echo "请手动: pip install numpy"; exit 1; }
fi
echo "============================================================"
echo " 实验: 过拟合 / L1·L2 / Dropout / BatchNorm"
echo "============================================================"
python3 <<'PY'
import numpy as np
np.set_printoptions(precision=4, suppress=True)
rng=np.random.default_rng(3)
def r2(y,yp): return 1-np.sum((y-yp)**2)/np.sum((y-y.mean())**2)
def design(x,d): return np.vstack([x**k for k in range(d+1)]).T
# 数据:x 归一化到 [-1,1] 保证高次幂有界、数值稳定;真实是低次 + 噪声
n=14; deg=8
x=rng.uniform(-1,1,n); y=0.5+0.8*x-0.6*x**2+rng.standard_normal(n)*0.15
xt=rng.uniform(-1,1,60); yt=0.5+0.8*xt-0.6*xt**2
Vtr,Vte=design(x,deg),design(xt,deg)
def ridge(lam): return np.linalg.solve(Vtr.T@Vtr+lam*np.eye(deg+1), Vtr.T@y)

# 1 过拟合:高次最小二乘
w=np.linalg.lstsq(Vtr,y,rcond=None)[0]
print(f"【1】过拟合:{deg} 次多项式拟合 {n} 个点(最小二乘)")
print(f"  训练 R²={r2(y,Vtr@w):.4f}  测试 R²={r2(yt,Vte@w):.4f}")
print(f"  权重 ‖w‖₂={np.linalg.norm(w):.2f}(高阶系数为硬凑噪声而变大)")
print("  解读:训练拟合好但测试明显更差→模型记住了噪声而非规律,这就是过拟合。")

# 2 L2 岭回归:加 λI
print("\n【2】L2 岭回归:λ 增大→权重变小→曲线变平滑")
for lam in [0.0,0.01,0.1,1.0,10.0]:
    w=ridge(lam)
    print(f"  λ={lam:<5}: 训练R²={r2(y,Vtr@w):.4f} 测试R²={r2(yt,Vte@w):.4f} ‖w‖₂={np.linalg.norm(w):.2f}")
print("  解读:λ 从0增大,测试 R² 先升后降——存在一个最优点,这就是偏差-方差权衡。")

# 3 L1/稀疏思想:正则后高阶系数被压小
w=ridge(1.0)
big=[k for k in range(deg+1) if abs(w[k])>0.02]
print(f"\n【3】稀疏(选特征):λ=1 时 |w|>0.02 的阶数 = {big}(共 {deg+1} 阶)")
print(f"  权重 w={w.round(3).tolist()}")
print("  解读:真正有用的是低阶(常数、x、x²),高阶系数被压成≈0。L1 比 L2 更能产生精确的 0,故常用于特征选择。")

# 4 Dropout(集成思想)+ BatchNorm
print("\n【4】Dropout / BatchNorm 的作用(模拟)")
# Dropout≈集成:多次 bootstrap 子集训练后平均,降低预测方差
preds=[]; 
for _ in range(40):
    idx=rng.integers(0,n,n)                 # 有放回重采样≈训练不同子网络
    Vb=Vtr[idx]; wb=np.linalg.solve(Vb.T@Vb+0.5*np.eye(deg+1), Vb.T@y[idx])
    preds.append(Vte@wb)
preds=np.array(preds); ens=preds.mean(0)
print(f"  集成(Dropout类比):单模型平均测试R²={np.mean([r2(yt,p) for p in preds]):.4f} → 集成R²={r2(yt,ens):.4f}")
print(f"  预测逐点标准差均值={preds.std(0).mean():.4f}(集成平滑了单模型的波动)")
print("  解读:Dropout 训练时随机丢神经元=每次训一个子网络;推理取平均≈集成,方差小、泛化更稳。")
# BatchNorm:激活归一化到均值0方差1
act=rng.standard_normal(1000)*5+7; norm=(act-act.mean())/act.std()
print(f"  BatchNorm:激活 μ={act.mean():.2f},σ={act.std():.2f} → 归一化后 μ={norm.mean():.2f},σ={norm.std():.2f}")
print("  解读:BN 把每层激活拉回标准分布,梯度更健康、可用更大学习率,批统计还带来轻微正则化。")
PY
echo "============================================================"
cat <<'EOF'
【结论与进阶】
- 小白:模型太强会死记训练点(过拟合);L2 让权重变小、曲线变平滑;L1 让无用权重归零(选特征);
  Dropout 训练时随机丢神经元≈集成,降低方差;BatchNorm 把激活归一化,训练更稳、学习率可更大。
- 熟手:L2 岭回归有解析解 (XᵀX+λI)w=Xᵀy;λ 靠验证集调(偏差-方差权衡);
  Dropout 推理时不丢但权重缩放(或训练时放大存活),BN 推理用滑动平均统计量;高次拟合务必先归一化输入。
- 延伸:把 deg 降到 2 看是否还过拟合;对比 L1 与 L2 的稀疏程度;调 Dropout/重采样比例看方差变化。
EOF
echo "============================================================"
