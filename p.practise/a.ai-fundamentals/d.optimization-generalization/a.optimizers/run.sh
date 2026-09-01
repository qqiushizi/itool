#!/bin/bash
# ============================================================
# 实验: a.optimizers
# 说明: SGD/Momentum/AdaGrad/RMSProp/Adam 优化轨迹对比
# 模块: p.practise/a.ai-fundamentals  AI 基础
# ============================================================
# 【第一性原理】
# 梯度下降靠 -∇f 走,但不同方向步长该多大?病态函数(某方向陡、某方向平)让 SGD 来回震荡。
# 优化器本质是"自适应调步长":动量用历史方向平滑;AdaGrad/RMSProp 按各维历史梯度大小缩放;
# Adam=动量+自适应,集大成。本实验在病态二次型上比谁收敛快、震荡小。
# ============================================================
set -euo pipefail
if ! python3 -c "import numpy" 2>/dev/null; then
  echo "未检测到 numpy,正在安装..."; python3 -m pip install --quiet numpy || { echo "请手动: pip install numpy"; exit 1; }
fi
echo "============================================================"
echo " 实验: 优化器 / SGD / Momentum / AdaGrad / RMSProp / Adam"
echo "============================================================"
python3 <<'PY'
import numpy as np
np.set_printoptions(precision=4, suppress=True)
# 病态二次型 f=10x²+y²:x 方向陡(曲率10)、y 方向平(曲率1)→SGD 在 x 上来回震荡
def grad(p): return np.array([20*p[0], 2*p[1]])
def run(name,update,steps=120,lr=0.1,x0=np.array([-4.0,-4.0])):
    p=x0.copy(); hist=[]; reached=None
    for t in range(1,steps+1):
        g=grad(p); p=update(p,g,t); d=np.linalg.norm(p); hist.append(d)
        if reached is None and d<0.05: reached=t
    return name, hist[-1], reached, max(hist[1:])-min(hist[1:]) if len(hist)>1 else 0
eps=1e-8
up={
 "SGD":      lambda p,g,t: p-0.1*g,
 "Momentum": (lambda v=[0,0]: lambda p,g,t: (v.__setitem__(slice(None),0.9*np.array(v)+g) or (p-0.1*np.array(v))))(),
 "AdaGrad":  (lambda c=[0,0]: lambda p,g,t: (c.__setitem__(slice(None),np.array(c)+g*g) or (p-0.5*g/(np.sqrt(np.array(c))+eps))))(),
 "RMSProp":  (lambda c=[0,0]: lambda p,g,t: (c.__setitem__(slice(None),0.9*np.array(c)+0.1*g*g) or (p-0.3*g/(np.sqrt(np.array(c))+eps))))(),
 "Adam":     (lambda m=[0,0],v=[0,0]: lambda p,g,t: (m.__setitem__(slice(None),0.9*np.array(m)+0.1*g), v.__setitem__(slice(None),0.999*np.array(v)+0.001*g*g), (p-0.1*(np.array(m)/(1-0.9**t))/(np.sqrt(np.array(v)/(1-0.999**t))+eps)))[-1])(),
}
print("在 f=10x²+y² 上从 (-4,-4) 出发,各优化器 120 步后:")
print(f"  {'优化器':<10}{'到原点距离':>12}{'首次<0.05步数':>16}{'末期震荡幅度':>14}")
for n,u in up.items():
    nm,final,reached,osc=run(n,u)
    rs=str(reached) if reached else "未达"
    print(f"  {n:<10}{final:>12.4f}{rs:>16}{osc:>14.4f}")
print("  解读:SGD 因 x 方向太陡而震荡、收敛慢;动量平滑方向;AdaGrad 累积平方梯度自动缩步;")
print("        RMSProp 用衰减平均避免步长过早变小;Adam=动量+自适应,最稳最快收敛。")
print("\n  关键直觉:优化器不是'换个公式',而是解决'各方向该走多大步'——自适应步长让病态曲面也好走。")
PY
echo "============================================================"
cat <<'EOF'
【结论与进阶】
- 小白:SGD 是最朴素的下山;动量像重物滚动更稳;AdaGrad/RMSProp 按梯度大小自动调步长;
  Adam=动量+自适应,是默认首选。
- 熟手:病态函数(条件数大)是优化慢的根因,动量/自适应本质都在改善条件数;
  Adam 的偏差修正在初期很重要;学习率仍是最关键超参,优化器不能弥补错误的 lr。
- 延伸:把曲率比从 10:1 改成 100:1 看 SGD 震荡加剧;对比 Adam 不同 lr 的收敛速度。
EOF
echo "============================================================"
