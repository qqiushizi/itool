#!/bin/bash
# 实验: d.clustering | KMeans 迭代、距离度量、初始化敏感性、肘部法则
# 模块: p.practise/a.ai-fundamentals | 配套: demo.py (--k 3 --bad)
# 【第一性原理】聚类(无监督)按相似度分组。KMeans:选K中心→就近归类→更新中心→重复。
#   目标最小化组内方差和(inertia)。初始化和距离度量影响结果。
set -euo pipefail
if ! python3 -c "import numpy" 2>/dev/null; then echo "安装 numpy..."; python3 -m pip install --quiet numpy || { echo "请手动: pip install numpy"; exit 1; }; fi
echo "############################################################"
echo "# 实验: KMeans 迭代 / 距离度量 / 初始化敏感性 / 肘部法则"
echo "# 每步: 【做什么&为什么】→【结果】→【结果解读】"
echo "############################################################"
python3 demo.py --teach
echo "############################################################"
echo "【动手体验】python3 demo.py --k 3 --bad       看坏初始化陷局部最优"
echo "            python3 demo.py --elbow            用肘部法则选簇数"
echo "############################################################"
