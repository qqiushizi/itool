#!/bin/bash
# ============================================================
# 实验: d.compute-metrics
# 说明: FLOPs 估算、tokens/s、MFU 概念与计算
# 模块: p.practise/c.llm-training  LLM 训练
# ============================================================
# 【第一性原理】
# 训练大模型要会算账。核心指标:
#  FLOPs≈6·N·D(N=参数量,D=训练 token 数),即训一遍的总计算量;
#  tokens/s=吞吐,看硬件实际产出;MFU(模型算力利用率)=实际FLOPs/峰值FLOPs,衡量训练效率。
# MFU<1 因为:通信开销、内存带宽、kernel 效率、流水气泡等。本实验估算训练成本、吞吐与 MFU。
# ============================================================
set -euo pipefail
if ! python3 -c "import numpy" 2>/dev/null; then
  echo "未检测到 numpy,正在安装..."; python3 -m pip install --quiet numpy || { echo "请手动: pip install numpy"; exit 1; }
fi
echo "============================================================"
echo " 实验: 计算量 / FLOPs / tokens-per-s / MFU"
echo "============================================================"
python3 <<'PY'
import numpy as np
# 1 训练 FLOPs 估算
N=7e9; D=2e12   # 7B 参数,2T tokens
flops=6*N*D
print("【1】训练总 FLOPs≈6·N·D")
print(f"  N={N:.0e} 参数, D={D:.0e} tokens → 总 FLOPs={flops:.2e}")
print("  解读:训练成本正比于参数×token;翻倍参数或数据,算力翻倍。这是估算 GPU·天 的基础。")

# 2 吞吐与 GPU·天
gpu_flops=312e12   # 单卡峰值(A100 FP16 ~312 TFLOPS)
n_gpu=1024; mfu=0.45
achieved=gpu_flops*n_gpu*mfu
seconds=flops/achieved
print(f"\n【2】吞吐与训练时长:")
print(f"  {n_gpu} 卡 × {gpu_flops/1e12:.0f} TFLOPS × MFU={mfu} = 实际 {achieved/1e12:.0f} TFLOPS")
print(f"  训练时长 = {flops:.2e} / {achieved:.2e} = {seconds/86400:.1f} GPU·集群·天")
print(f"  tokens/s = {D/seconds:.2e}")
print("  解读:MFU 把峰值打个折(通信/带宽/气泡),实际产出远低于标称;提升 MFU=省真金白银。")

# 3 MFU 的影响
print("\n【3】MFU 对训练成本的影响(同样算力,以 MFU=0.2 为基准):")
t_base=flops/(gpu_flops*n_gpu*0.2)/86400
for mfu in [0.2,0.35,0.45,0.6]:
    t=flops/(gpu_flops*n_gpu*mfu)/86400
    print(f"  MFU={mfu}: 训练 {t:.1f} 天  (相比0.2 省 {(1-t/t_base)*100:.0f}%)")
print("  解读:MFU 每提升,训练时间等比缩短;大规模训练里优化 MFU(通信重叠、算子融合、显存)是核心工程目标。")
PY
echo "============================================================"
cat <<'EOF'
【结论与进阶】
- 小白:训练 FLOPs≈6·N·D;吞吐看 tokens/s;MFU=实际/峰值,衡量训练效率,MFU 越高越省钱。
- 熟手:6·N·D 中 6=2(前向)+4(反向)的近似;MFU 上限受限于通信/带宽/算子效率;
  3D 并行、ZeRO、FlashAttention 都为提 MFU;FP8/稀疏可进一步提升有效算力。
- 延伸:把 N 改成 70B 看算力爆炸;估算不同 MFU 下的 GPU·天成本。
EOF
echo "============================================================"
