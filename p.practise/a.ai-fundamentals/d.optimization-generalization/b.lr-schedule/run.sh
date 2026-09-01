#!/bin/bash
# ============================================================
# 实验: b.lr-schedule
# 说明: warmup/cosine/余弦重启 学习率曲线
# 模块: p.practise/a.ai-fundamentals  AI 基础
# ============================================================
# 【第一性原理】
# 学习率太大→发散/震荡,太小→收敛慢。固定 lr 很难两头兼顾。调度策略让 lr 随训练变化:
#  warmup:初期小步避免不稳,线性升到峰值(大模型/大 batch 必备);
#  cosine:升到峰值后按余弦平滑降到极小值,后期精调;
#  cosine restart:周期性回升,跳出局部最优。
# 本实验把几种曲线的 lr 随 epoch 打印出来对比形状。
# ============================================================
set -euo pipefail
if ! python3 -c "import numpy" 2>/dev/null; then
  echo "未检测到 numpy,正在安装..."; python3 -m pip install --quiet numpy || { echo "请手动: pip install numpy"; exit 1; }
fi
echo "============================================================"
echo " 实验: 学习率调度 / warmup / cosine / 余弦重启"
echo "============================================================"
python3 <<'PY'
import numpy as np
E=100; base,mini=1.0,0.01; warm=10
def const(e): return base
def step(e): return base*(0.5**(e//30))
def warmup_cos(e):
    if e<warm: return base*e/warm
    return mini+0.5*(base-mini)*(1+np.cos(np.pi*(e-warm)/(E-warm)))
def cos_restart(e,period=40):
    return mini+0.5*(base-mini)*(1+np.cos(np.pi*(e%period)/period))
schedules={"constant":const,"step_decay":step,"warmup+cosine":warmup_cos,"cosine_restart":cos_restart}
eps=[0,3,9,10,30,50,70,90,99]
print(f"{'epoch':<8}"+"".join(f"{n:>16}" for n in schedules))
for e in eps:
    print(f"{e:<8}"+"".join(f"{schedules[n](e):>16.4f}" for n in schedules))
print("\n  解读:")
print("  - constant:全程不变,简单但后期可能太大、无法精调。")
print("  - step_decay:每 30 步减半,阶梯式下降,经典但需手调节点。")
print("  - warmup+cosine:前10步线性升到峰值,再余弦平滑降到极小——大模型训练标配。")
print("  - cosine_restart:周期性回升,像'喘口气'再冲,有助跳出局部最优。")
PY
echo "============================================================"
cat <<'EOF'
【结论与进阶】
- 小白:lr 不是固定值;warmup 防初期不稳,cosine 后期平滑精调,重启周期性回升跳出局部最优。
- 熟手:warmup 对大 batch/大模型几乎是必需(初期梯度噪声大);cosine 到极小值而非 0,保留微调能力;
  restart 的周期长度对应"愿意花多久重找方向";lr 与 batch size 大致线性缩放(sqrt 否则)。
- 延伸:把 warmup 长度从10改到30看稳定性;对比 cosine 与 step_decay 的最终精度差异。
EOF
echo "============================================================"
