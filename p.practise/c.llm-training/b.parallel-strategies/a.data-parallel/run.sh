#!/bin/bash
# ============================================================
# 实验: a.data-parallel
# 说明: DP/梯度聚合概念与通信量
# 模块: p.practise/c.llm-training  LLM 训练
# ============================================================
# 【第一性原理】
# 数据并行(DP):每张卡持有完整模型副本,各卡喂不同 batch,各自前向反向算梯度,
# 再 AllReduce 把各卡梯度求平均(保证模型同步),最后各卡用相同梯度更新。
# 通信量:每步需 AllReduce 整个梯度≈2×参数量×字节数(AllReduce 通信量随卡数近不变,靠带宽)。
# 瓶颈:模型必须放进单卡显存;梯度聚合是通信大头。本实验模拟 DP 的梯度聚合与通信量。
# ============================================================
set -euo pipefail
if ! python3 -c "import numpy" 2>/dev/null; then
  echo "未检测到 numpy,正在安装..."; python3 -m pip install --quiet numpy || { echo "请手动: pip install numpy"; exit 1; }
fi
echo "============================================================"
echo " 实验: 数据并行 / 梯度聚合 / AllReduce / 通信量"
echo "============================================================"
python3 <<'PY'
import numpy as np
np.set_printoptions(precision=3, suppress=True)
# 模拟:4卡各算一份梯度,AllReduce 求平均
rng=np.random.default_rng(0)
N=8; grads=[rng.standard_normal(N) for _ in range(4)]
avg=np.mean(grads,axis=0)
print("【1】数据并行:各卡算梯度 → AllReduce 求平均 → 同步更新")
for i,g in enumerate(grads): print(f"  卡{i}梯度={g.round(3).tolist()}")
print(f"  AllReduce平均={avg.round(3).tolist()}  (各卡拿到相同梯度,更新后模型一致)")
print("  解读:DP 保证每卡模型同步;通信=把所有梯度聚合,量≈参数大小(与卡数几乎无关,靠带宽)。")

# 2 通信量估算
print("\n【2】通信量估算(AllReduce 传输≈2×参数×字节,与卡数近无关):")
for params in [1e9,7e9,70e9]:
    bytes_=4  # FP32
    comm=2*params*bytes_   # Ring AllReduce ≈ 2N 数据量
    print(f"  {params:.0e} 参数: 每步通信≈{comm/1e9:.1f} GB")
print("  解读:模型越大,梯度聚合通信越大;这是 DP 的主要瓶颈,靠高速互联(带宽)和梯度压缩缓解。")

# 3 batch 拆分
print("\n【3】batch 拆分:全局 batch 拆到各卡,等效增大总 batch")
gbatch=1024
for n in [1,4,8,16]:
    print(f"  全局batch={gbatch}, {n}卡 → 每卡 {gbatch//n} 样本  (总吞吐≈×{n})")
print("  解读:DP 天然线性扩吞吐;但全局 batch 过大会影响收敛,需学习率/warmup 调整。")
PY
echo "============================================================"
cat <<'EOF'
【结论与进阶】
- 小白:数据并行=每卡完整模型+不同数据,反向后 AllReduce 梯度求平均保证同步;通信量≈参数大小。
- 熟手:Ring AllReduce 通信量≈2×参数,与卡数近无关(靠带宽);DP 受限于单卡装得下模型;
  大全局 batch 需配合 LR scaling 和 warmup;梯度压缩/累积可降通信。
- 延伸:把参数改 70B 看通信量;思考为何 DP 不能解决"模型放不下单卡"。
EOF
echo "============================================================"
