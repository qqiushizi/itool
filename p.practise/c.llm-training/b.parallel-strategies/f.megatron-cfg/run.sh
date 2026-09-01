#!/bin/bash
# ============================================================
# 实验: f.megatron-cfg
# 说明: 组合并行配置、通信代价估算
# 模块: p.practise/c.llm-training  LLM 训练
# ============================================================
# 【第一性原理】
# 单一并行都不够:DP 受单卡显存、TP 受节点内互联、PP 有气泡。实际大模型用 3D 并行组合:
#  TP(节点内,NVLink 高速)× PP(跨节点,通信少)× DP(剩余卡扩吞吐)。
# 配置思路:先定 TP(≤单节点卡数),再定 PP(按层数分阶段),剩余卡做 DP。
# 关键:让 TP 在节点内(高带宽)、PP 跨节点(低频),避免 TP 跨节点拖慢。本实验做配置组合与通信估算。
# ============================================================
set -euo pipefail
if ! python3 -c "import numpy" 2>/dev/null; then
  echo "未检测到 numpy,正在安装..."; python3 -m pip install --quiet numpy || { echo "请手动: pip install numpy"; exit 1; }
fi
echo "============================================================"
echo " 实验: 3D 并行 / Megatron 配置 / 通信估算"
echo "============================================================"
python3 <<'PY'
import numpy as np
# 1 3D 并行组合
total=64; node=8   # 64卡,每节点8卡
print("【1】3D 并行组合(TP×PP×DP=总卡数):")
for tp,pp in [(8,1),(4,2),(2,4),(1,8)]:
    if total%(tp*pp)==0:
        dp=total//(tp*pp)
        intra = "TP在节点内✓" if tp<=node else "TP跨节点✗(慢)"
        print(f"  TP={tp} PP={pp} DP={dp} (×={tp*pp*dp}) → {intra}")
print("  解读:总卡数=TP×PP×DP;TP 应≤节点卡数(用 NVLink),跨节点 TP 会严重拖慢。")

# 2 各并行的通信位置
print("\n【2】各并行的通信频率与位置:")
print("  TP:层内高频 → 必须节点内(NVLink)")
print("  PP:层边界低频 → 可跨节点(以太网/IB)")
print("  DP:每步一次 AllReduce → 可跨节点")
print("  解读:把高频通信(TP)放节点内、低频(PP/DP)放跨节点,是 3D 并行高效的核心。")

# 3 通信量估算
print("\n【3】通信量示意(单步,隐藏维 h=12288, 字节=2):")
h=12288; b=2
tp_comm=h*b*2          # TP AllReduce 输出
pp_comm=h*b*2          # PP 层边界传激活
for tp in [1,4,8]:
    layers_pp=4
    print(f"  TP={tp}: 每层TP通信≈{tp_comm*tp/1e3:.0f}KB; PP每阶段传≈{pp_comm/1e3:.0f}KB")
print("  解读:TP 通信高频且随切分增,务必高速;PP 通信低频,适合跨节点。组合后总通信=三者叠加。")
PY
echo "============================================================"
cat <<'EOF'
【结论与进阶】
- 小白:3D 并行=TP×PP×DP=总卡数;TP 放节点内(高频高速)、PP/DP 放跨节点(低频);合理组合才高效。
- 熟手:TP≤单节点卡数是硬约束;PP 阶段数按层数和显存定;DP 扩吞吐;
  配置要先定 TP 再 PP 剩 DP;重计算/ZeRO 与 3D 并行叠加是大模型训练标配。
- 延伸:把节点卡数从8改到16看 TP 上限变化;对比不同 TP/PP 组合的总通信量。
EOF
echo "============================================================"
