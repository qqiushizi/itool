#!/bin/bash
# ============================================================
# 实验: b.lora-qlora
# 说明: LoRA/QLoRA 低秩、量化基座、效果
# 模块: p.practise/c.llm-training  LLM 训练
# ============================================================
# 【第一性原理】
# LoRA:假设权重更新 ΔW 是低秩的,用 ΔW=A·B(A∈R^{d×r},B∈R^{r×d},r<<d)近似,只训 A、B,
# 原权重 W 冻结。可训参数从 d² 降到 2dr,大幅省显存/算力,效果接近全参。
# QLoRA:在 LoRA 基础上把冻结的基座 W 量化到 4bit(NF4),再在量化的基座上训 LoRA→单卡可训超大模型。
# 本实验演示低秩近似的参数节省,并模拟量化基座的显存节省。
# ============================================================
set -euo pipefail
if ! python3 -c "import numpy" 2>/dev/null; then
  echo "未检测到 numpy,正在安装..."; python3 -m pip install --quiet numpy || { echo "请手动: pip install numpy"; exit 1; }
fi
echo "============================================================"
echo " 实验: LoRA / 低秩更新 / QLoRA 量化基座"
echo "============================================================"
python3 <<'PY'
import numpy as np
np.set_printoptions(precision=3, suppress=True)
rng=np.random.default_rng(0)
d=4096
# 1 LoRA 低秩近似参数节省
print("【1】LoRA:ΔW≈A·B,只训 A,B,参数从 d² 降到 2dr")
for r in [8,16,64]:
    full=d*d; lora=2*d*r
    print(f"  d={d}, r={r}: 全参={full:,} → LoRA={lora:,} (节省 {100-lora/full*100:.1f}%, 可训占比 {lora/full*100:.2f}%)")
print("  解读:r 远小于 d 时,可训参数骤降到 <1%,显存/算力大幅省,效果仍接近全参。")

# 2 低秩近似能还原能力(数值演示)
W=rng.standard_normal((50,50))
# 构造一个「真实低秩」的权重更新 ΔW=A·B(秩=4),看小 r 能否还原
A0=rng.standard_normal((50,4)); B0=rng.standard_normal((4,50)); dW=A0@B0; dW=dW/dW.std()*0.1
U,S,Vt=np.linalg.svd(dW,full_matrices=False)
for r in [2,4,8,20]:
    dW_low=U[:,:r]@np.diag(S[:r])@Vt[:r,:]
    err=np.linalg.norm(dW-dW_low)/np.linalg.norm(dW)
    print(f"  用秩 r={r} 近似 ΔW(真实秩4): 相对误差={err:.3f}")
print("  解读:真实更新秩=4 时,r≥4 误差≈0 完美还原,r<4 误差大→LoRA 假设「更新低秩」成立时,小 r 就够。")

# 3 QLoRA:量化基座省显存
print("\n【3】QLoRA:把冻结基座量化到 4bit,再训 LoRA")
phi=7e9
fp16=2*phi/1e9; nf4=0.5*phi/1e9; lora_param=2*4096*16*32/1e9  # 32层 r=16
print(f"  7B 基座: FP16={fp16:.0f}GB → 4bit NF4≈{nf4:.1f}GB + LoRA参数≈{lora_param*1e3:.1f}MB")
print("  解读:量化把基座显存压到 1/4,加上极小的 LoRA 参数→单张消费级显卡也能微调 7B/70B。")
PY
echo "============================================================"
cat <<'EOF'
【结论与进阶】
- 小白:LoRA 用低秩 A·B 近似权重更新,只训极小部分参数,省显存效果好;QLoRA 再把基座量化到4bit,单卡可训超大模型。
- 熟手:r 常取 8~64,越大越接近全参但越贵;LoRA 可合并回 W 加速推理;QLoRA 用 NF4+双量化保精度;
  LoRA 加在 attention 的 Q/V 是经典配置,也可加 FFN。
- 延伸:把 r 从8调到64看参数占比;对比 LoRA 与全参在相同数据上的效果。
EOF
echo "============================================================"
