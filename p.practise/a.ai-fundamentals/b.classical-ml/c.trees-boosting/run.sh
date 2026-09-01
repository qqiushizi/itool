#!/bin/bash
# 实验: c.trees-boosting | 决策树分裂/信息增益、GBDT 直觉
# 模块: p.practise/a.ai-fundamentals | 配套: demo.py (--trees 20 --lr 0.3)
# 【第一性原理】决策树用"是/否"问题把数据分纯,每步选信息增益最大的切分。
#   GBDT:后一棵树拟合前一棵的残差,多棵弱树累加=强模型。
set -euo pipefail
if ! python3 -c "import numpy" 2>/dev/null; then echo "安装 numpy..."; python3 -m pip install --quiet numpy || { echo "请手动: pip install numpy"; exit 1; }; fi
echo "############################################################"
echo "# 实验: 决策树分裂 / 信息增益 / GBDT"
echo "# 每步: 【做什么&为什么】→【结果】→【结果解读】"
echo "############################################################"
python3 demo.py --teach
echo "############################################################"
echo "【动手体验】python3 demo.py --trees 20 --lr 0.3   看树数/学习率如何影响"
echo "            python3 demo.py --noise 0.5            看噪声如何让树过拟合"
echo "############################################################"
