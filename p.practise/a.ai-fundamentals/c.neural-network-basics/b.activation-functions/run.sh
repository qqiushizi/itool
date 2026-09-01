#!/bin/bash
# ============================================================
# 实验: b.activation-functions
# 说明: 各激活函数形状、梯度、饱和区对比
# 模块: p.practise/a.ai-fundamentals  AI 基础
# ============================================================
# 【第一性原理】
# 激活函数引入非线性,否则多层网络等价于单层线性变换。但它也带来两个隐患:
#  ① 饱和区梯度→0,深层回传梯度消失(sigmoid/tanh 在两端压平);
#  ② ReLU 负半轴恒 0,神经元可能"死掉"再不更新。
# 选激活 = 在"非线性强度、梯度健康、计算成本"间权衡。本实验用数值看曲线、梯度与衰减。
# ============================================================
set -euo pipefail
if ! python3 -c "import numpy" 2>/dev/null; then
  echo "未检测到 numpy,正在安装..."; python3 -m pip install --quiet numpy || { echo "请手动: pip install numpy"; exit 1; }
fi
echo "============================================================"
echo " 实验: 激活函数 / 形状 / 梯度 / 饱和区 / 梯度消失"
echo "============================================================"
python3 <<'PY'
import numpy as np
np.set_printoptions(precision=4, suppress=True)
def block(i,t,title,why,res,mean):
    print(f"\n{'='*58}\n【步骤 {i}/{t}】{title}\n{'='*58}")
    print(f"--- 做什么 & 为什么 ---\n  {why}\n--- 结果 ---\n  {res}\n--- 结果解读 ---\n  {mean}")
T=4
acts={
 "sigmoid": (lambda z:1/(1+np.exp(-z)), lambda a:a*(1-a)),
 "tanh":    (np.tanh, lambda a:1-a*a),
 "relu":    (lambda z:np.maximum(0,z), lambda a:(a>0).astype(float)),
 "leaky":   (lambda z:np.where(z>0,z,0.1*z), lambda a:np.where(a>0,1,0.1)),
 "gelu":    (lambda z:0.5*z*(1+np.tanh(0.7978*(z+0.0447*z**3))),
             lambda z:0.5*(1+np.tanh(0.7978*(z+0.0447*z**3)))+0.5*z*(1-np.tanh(0.7978*(z+0.0447*z**3))**2)*0.7978*(1+0.1341*z**2)),
}
xs=np.array([-5,-2,-1,0,1,2,5.0])

# 1 形状对比
print("\n【1】五种激活函数在 x=[-5,-2,-1,0,1,2,5] 的输出:")
for n,(f,_) in acts.items():
    print(f"  {n:8s}: {f(xs).round(3).tolist()}")
print("  解读:sigmoid∈(0,1)、tanh∈(-1,1)、ReLU 负半轴=0、Leaky 负半轴有斜率、GELU 处处平滑。")

# 2 导数(梯度)对比
print("\n【2】各激活的导数(回传的梯度大小):")
for n,(f,df) in acts.items():
    g=df(f(xs))
    print(f"  {n:8s}: {g.round(3).tolist()}")
print("  解读:sigmoid 在 |x|>3 导数≈0(饱和→梯度消失);ReLU 正区恒为1(梯度不衰减)、负区=0。")

# 3 饱和区 & 死 ReLU
print("\n【3】饱和区与死 ReLU:")
sig=lambda z:1/(1+np.exp(-z))
print(f"  sigmoid(±10) 的导数 = {sig(-10)*(1-sig(-10)):.2e}, {sig(10)*(1-sig(10)):.2e}  (≈0 → 梯度消失)")
big=np.array([-100,-50,-2,3,8.0])
relu_out=np.maximum(0,big); dead=(relu_out==0).sum()
print(f"  ReLU 对输入 {big.tolist()} 输出 {relu_out.tolist()} → {dead} 个神经元输出0(可能'死')")
print("  解析:大输入让 sigmoid/tanh 梯度归零,深层网络靠不上梯度;ReLU 一旦持续负输入就永久不更新=死神经元。")

# 4 深层梯度回传衰减实测
print("\n【4】堆 10 层,看信号(梯度)回传后还剩多少:")
x0=np.array([2.0])
for n,(f,df) in acts.items():
    a=f(x0); g=df(a)
    for _ in range(9):
        a=f(a); g=g*df(a)   # 逐层乘导数
    print(f"  {n:8s}: 10 层后梯度 = {g[0]:.2e}")
print("  解读:sigmoid/tanh 梯度随层数指数衰减(消失);ReLU/Leaky 在正区基本不衰减→深层网络选它们的原因。")
PY
echo "============================================================"
cat <<'EOF'
【结论与进阶】
- 小白:激活函数给网络加非线性;sigmoid/tanh 两端"压平"会让梯度消失,深层学不动;
  ReLU 简单快、正区梯度不衰减,但负输入可能让神经元"死掉";Leaky/GELU 是改良版。
- 熟手:隐藏层默认 ReLU/GELU/SiLU,输出层按任务选(二分类 sigmoid、多分类 softmax、回归线性);
  梯度消失靠残差连接+合理激活缓解;死 ReLU 靠小学习率、Leaky/ELU 或初始化避免。
- 延伸:把层数从 10 改到 50 看 sigmoid 梯度衰减成多少;给 ReLU 喂全负输入看它是否复活。
EOF
echo "============================================================"
