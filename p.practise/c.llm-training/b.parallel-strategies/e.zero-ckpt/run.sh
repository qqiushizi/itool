#!/bin/bash
# ============================================================
# 实验: e.zero-ckpt
# 说明: ZeRO-1/2/3 显存占用建模
# 模块: p.practise/c.llm-training  LLM 训练
# ============================================================
# 【第一性原理】
# 训练显存=模型参数(2φ)+梯度(2φ)+优化器状态(Adam:φ 的 m,v 共 ~8φ,FP32)+激活。
# ZeRO 逐步切分这些状态到多卡,用通信换显存:
#  ZeRO-1:切优化器状态(每卡只存一部分 m,v)→省最多显存项;
#  ZeRO-2:再切梯度;ZeRO-3:连参数也切(用时 AllGather 取回)。
# 显存从单卡 ~16φ 降到 ZeRO-3 的 ~16φ/N。本实验建模各阶段单卡显存。
# ============================================================
set -euo pipefail
if ! python3 -c "import numpy" 2>/dev/null; then
  echo "未检测到 numpy,正在安装..."; python3 -m pip install --quiet numpy || { echo "请手动: pip install numpy"; exit 1; }
fi
echo "============================================================"
echo " 实验: ZeRO / 优化器分片 / 显存建模"
echo "============================================================"
python3 <<'PY'
import numpy as np
# 以参数量 φ 为单位(FP16参数2φ + FP16梯度2φ + FP32优化器8φ ≈ 12~16 φ·字节)
def mem(phi,N,stage,act=2):
    param=2; grad=2; opt=8   # 单位:φ×字节
    if stage==0: return param+grad+opt+act          # 普通数据并行:全留单卡
    if stage==1: return param+grad+opt/N+act         # ZeRO-1:切优化器
    if stage==2: return param+grad/N+opt/N+act       # ZeRO-2:再切梯度
    if stage==3: return param/N+grad/N+opt/N+act     # ZeRO-3:连参数切(激活另算)
phi=7e9; N=8
print("【1】ZeRO 分片:逐步把优化器/梯度/参数切到多卡(7B, 8卡, FP16+FP32 Adam)")
for st,name in [(0,"普通DP"),(1,"ZeRO-1"),(2,"ZeRO-2"),(3,"ZeRO-3")]:
    m=mem(phi,N,st)*phi/1e9
    print(f"  {name}: 单卡显存≈{m:.1f} GB")
print("  解读:ZeRO-3 把参数/梯度/优化器全切,单卡显存随 N 线性下降,让大模型在少量卡上可训。")

# 2 通信代价
print("\n【2】通信代价(用通信换显存):")
print("  ZeRO-1/2:额外通信≈一次 AllReduce(和 DP 相近)")
print("  ZeRO-3:前向/反向每层都要 AllGather 取回参数→通信显著增加")
print("  解读:ZeRO-3 显存最优但通信最重;实际在显存够用时选 ZeRO-1/2 平衡效率。")

# 3 不同 N
print("\n【3】ZeRO-3 单卡显存随卡数 N 下降:")
for N in [4,8,16,64]:
    m=mem(phi,N,3)*phi/1e9
    print(f"  N={N:<3}: ZeRO-3 单卡≈{m:.1f} GB")
print("  解读:卡越多,每卡分摊的状态越少;这让「参数放不下单卡」的问题被绕过。")
PY
echo "============================================================"
cat <<'EOF'
【结论与进阶】
- 小白:训练显存=参数+梯度+优化器(Adam 最大);ZeRO 把它们逐步切到多卡:Z1切优化器、Z2再切梯度、Z3连参数切,省显存换通信。
- 熟手:Adam 优化器状态占大头(8φ);ZeRO-3 让单卡显存≈16φ/N,但每层 AllGather 增通信;
  激活显存靠梯度检查点单独优化;ZeRO 与 TP/PP 可组合,选型看显存/通信权衡。
- 延伸:把 N 从8改到64看 ZeRO-3 显存下降;对比 ZeRO-2 与 ZeRO-3 的通信量。
EOF
echo "============================================================"
