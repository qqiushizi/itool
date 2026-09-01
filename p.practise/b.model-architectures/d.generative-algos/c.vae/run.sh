#!/bin/bash
# ============================================================
# 实验: c.vae
# 说明: 编码/解码、重参数化、ELBO
# 模块: p.practise/b.model-architectures  模型架构
# ============================================================
# 【第一性原理】
# VAE 学一个"编码器+解码器":编码器把数据 x 映成分布 q(z|x)=N(μ,σ²),从中采样 z,解码器还原 x̂。
# 难点:采样不可导。重参数化技巧:z=μ+σ·ε(ε~N(0,1)),把随机性移到输入,梯度能流过 μ、σ。
# 目标 ELBO=重建项(让 x̂≈x)+ KL 项(让 q(z|x) 靠近先验 N(0,1)),兼顾重建与潜空间规整。
# 本实验演示重参数化与 ELBO 的两项计算。
# ============================================================
set -euo pipefail
if ! python3 -c "import numpy" 2>/dev/null; then
  echo "未检测到 numpy,正在安装..."; python3 -m pip install --quiet numpy || { echo "请手动: pip install numpy"; exit 1; }
fi
echo "============================================================"
echo " 实验: VAE / 编码解码 / 重参数化 / ELBO"
echo "============================================================"
python3 <<'PY'
import numpy as np
np.set_printoptions(precision=4, suppress=True)
rng=np.random.default_rng(0)
# 1 编码器输出 μ,σ;重参数化采样
x=np.array([2.0,0.5,-1.0])
mu=np.array([1.8,0.4,-0.9]); logvar=np.array([0.1,0.2,0.05]); sigma=np.exp(0.5*logvar)
eps=rng.standard_normal(x.shape); z=mu+sigma*eps
print("【1】重参数化:z=μ+σ·ε,把随机性移到 ε,梯度能流过 μ、σ")
print(f"  输入 x={x.tolist()}")
print(f"  编码器输出 μ={mu.tolist()}, σ={sigma.round(3).tolist()}")
print(f"  采样 ε={eps.round(3).tolist()} → z={z.round(3).tolist()  }  (可导!)")
print("  解读:直接采样 z~N(μ,σ²) 不可导;写成 μ+σ·ε 后,梯度沿 μ、σ 反传,这是 VAE 训练的关键。")

# 2 ELBO = 重建 + KL
xhat=z                     # 解码器(简化为恒等)
recon=np.mean((x-xhat)**2) # 重建项(这里用 MSE 示意,实际用 log似然)
kl=0.5*np.sum(np.exp(logvar)+mu**2-1-logvar)   # KL(q(z|x)||N(0,1)) 的闭式解
elbo=-(recon+kl)
print(f"\n【2】ELBO = -(重建 + KL):")
print(f"  重建项(这里MSE示意)={recon:.4f}")
print(f"  KL 项(让 q 靠近标准正态)={kl:.4f}")
print(f"  ELBO={elbo:.4f}  (最大化 ELBO = 同时降重建、规整潜空间)")
print("  解读:重建项保证还原数据,KL 项把潜空间拉向 N(0,1)→让潜空间连续可插值、可采样生成。")

# 3 生成
print("\n【3】生成:从 N(0,1) 采 z,过解码器得到新样本")
z_new=rng.standard_normal(3); print(f"  采样 z={z_new.round(3).tolist()} → 解码器 → 新样本")
print("  解读:因 KL 把潜空间规整成标准正态,从 N(0,1) 采样就能生成合理新数据——这是 VAE 的生成方式。")
PY
echo "============================================================"
cat <<'EOF'
【结论与进阶】
- 小白:VAE=编码器(输出μ,σ)+解码器;重参数化 z=μ+σ·ε 让采样可导;ELBO=重建+KL,兼顾还原与规整潜空间。
- 熟手:KL 的闭式解 0.5·Σ(exp(logvar)+μ²-1-logvar) 仅对高斯成立;β-VAE 调 KL 权衡解耦与重建;
  VAE 生成偏模糊(MSE/高斯假设),GAN 锐利但难训;扩散模型在多数生成任务上超越两者。
- 延伸:把 KL 权重调大看潜空间是否更接近 N(0,1);对比 VAE 与 AE(无 KL)的潜空间分布。
EOF
echo "============================================================"
