#!/bin/bash
# 实验: b.logistic-regression | 决策边界、损失曲线、概率校准
# 模块: p.practise/a.ai-fundamentals | 配套: demo.py (--lr 0.5 --noise 0.3)
# 【第一性原理】线性回归输出任意实数,分类要概率(0~1)。套 sigmoid σ(z)=1/(1+e^-z) 压缩。
#   决策边界 σ(z)=0.5 即 z=0;损失用交叉熵(对数损失)而非 MSE。
set -euo pipefail
if ! python3 -c "import numpy" 2>/dev/null; then echo "安装 numpy..."; python3 -m pip install --quiet numpy || { echo "请手动: pip install numpy"; exit 1; }; fi
echo "############################################################"
echo "# 实验: 逻辑回归 / sigmoid / 决策边界 / 对数损失 / 概率校准"
echo "# 每步: 【做什么&为什么】→【结果】→【结果解读】"
echo "############################################################"
python3 demo.py --teach
echo "############################################################"
echo "【动手体验】python3 demo.py --lr 0.5 --noise 0.3  看学习率/噪声如何影响收敛"
echo "            python3 demo.py --boundary            看决策边界随训练移动"
echo "############################################################"
