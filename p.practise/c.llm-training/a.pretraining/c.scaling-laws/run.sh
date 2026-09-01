#!/bin/bash
# ============================================================
# 实验: c.scaling-laws
# 说明: 模型/数据/计算 scaling law 模拟与损失预测
# 模块: p.practise/c.llm-training  LLM 训练
# ============================================================
# 【第一性原理】
# Chinchilla scaling law:损失随参数量 N、数据量 D、计算量 C 幂律下降,
# 形如 L(N,D)=E + A/N^α + B/D^β。关键结论:给定固定算力,模型大小和数据量应按 ~1:20(token/参数)同步增长才最优;
# 很多早期大模型"参数过剩、数据不足"(如 GPT-3),用更少参数+更多数据反而更好(Chinchilla)。
# 本实验用幂律公式预测不同 N、D 的损失,并找给定算力下的最优 N-D 配比。
# ============================================================
set -euo pipefail
if ! python3 -c "import numpy" 2>/dev/null; then
  echo "未检测到 numpy,正在安装..."; python3 -m pip install --quiet numpy || { echo "请手动: pip install numpy"; exit 1; }
fi
echo "============================================================"
echo " 实验: Scaling Law / 损失幂律 / 最优 N-D 配比"
echo "============================================================"
python3 <<'PY'
import numpy as np
np.set_printoptions(precision=3, suppress=True)
# 损失幂律(参数近似 Chinchilla 系数)
E,A,B,alpha,beta=1.7,40,5,0.34,0.28
def loss(N,D): return E+A/N**alpha+B/D**beta
# 1 损失随 N、D 下降
print("【1】损失随参数量 N、数据量 D 下降(幂律):")
for N in [1e8,1e9,1e10,1e11]:
    print(f"  N={N:.0e}, D={20*N:.0e}: L={loss(N,20*N):.3f}")
print("  解读:N、D 增大,损失按幂律下降,但收益递减(每翻倍提升越来越小)。")

# 2 固定算力下的最优配比
print("\n【2】固定计算量 C≈6·N·D 下,找最优 N-D 配比:")
C=6e20   # 固定算力(FLOPs)
best=None
for ratio in [5,10,20,50,100]:   # D/N = token/参数
    N=(C/(6*ratio))**(1/1)       # C=6ND, D=ratio·N
    D=ratio*N; L=loss(N,D)
    print(f"  D/N={ratio:<4}: N={N:.2e}, D={D:.2e}, L={L:.3f}")
    if best is None or L<best[1]: best=(ratio,L)
print(f"  最优配比 D/N≈{best[0]} (损失最低={best[1]:.3f})")
print("  解读:Chinchilla 发现最优 token/参数≈20;偏离它(参数过多数据不足,或反之)都不如均衡。")
PY
echo "============================================================"
cat <<'EOF'
【结论与进阶】
- 小白:损失随模型大小、数据量按幂律下降;固定算力下,模型参数和数据量要均衡增长(约1:20 token/参数)才最优。
- 熟手:Chinchilla 纠正了"只堆参数"的做法,强调数据量;scaling law 可外推预测大模型损失、指导资源分配;
  推理成本使"训练够用、推理高效"的小模型(蒸馏/稀疏)变得重要;数据质量会平移整条曲线。
- 延伸:把 α/β 系数微调看最优配比变化;给定 N 求 L 最优的 D。
EOF
echo "============================================================"
