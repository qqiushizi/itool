#!/bin/bash
# ============================================================
# 实验: c.weight-only
# 说明: W8A16/W4A16、AWQ/GPTQ 直觉
# 模块: p.practise 第一性原理学习实验
# ============================================================
# 【第一性原理】
# Weight-only quantization: 只量化权重(weight),激活保持 FP16:
#   - W8A16: 权重 INT8, 激活 FP16
#   - W4A16: 权重 INT4, 激活 FP16
#   - 显存减半 (W8A16) / 4× (W4A16), 适合 LLM 推理
# 经典算法:
#   - GPTQ: 二阶导数 (Hessian) 优化量化误差,逐层贪心
#   - AWQ:  激活感知,保护\"重要权重\"(对应高幅值激活的)
#   - AutoGPTQ: GPTQ 的工程实现,bitsandbytes
# 优势 vs 全量化:
#   - 算力仍走 FP16 kernel, 加速有限但不掉点
#   - 权重省 4-8×, KV cache 不动, batch 能大
# ============================================================
set -euo pipefail
if ! python3 -c "import numpy" 2>/dev/null; then python3 -m pip install --quiet numpy; fi
echo "############################################################"
echo "# 实验: c.weight-only | W8A16/W4A16,AWQ/GPTQ 直觉"
echo "############################################################"

python3 <<'PYEOF'
import numpy as np
def hdr(n,t,total): print(f"\n{'='*60}\n【步骤 {n}/{total}】{t}\n{'='*60}")
def why(s): print("\n--- 做什么 & 为什么 ---\n  "+s.replace("\n","\n  "))
def res(s): print("\n--- 结果 ---\n  "+s.replace("\n","\n  "))
def mea(s): print("\n--- 解读 ---\n  "+s.replace("\n","\n  "))
TOTAL = 4

# --- 1. 显存节省 ---
hdr(1,TOTAL,"显存节省:weight-only")
why("""7B 模型: 权重 14 GB (FP16)
  W8A16:  7 GB (权重 INT8)  → 省 7 GB
  W4A16:  3.5 GB (权重 INT4) → 省 10.5 GB
  + KV cache 不动, 仍能跑大 batch
  + 算力 kernel 仍 FP16/INT8, 加速 1.2-1.5×""")
P = 7e9
out = [f"  模式      权重大小    KV (8K×32 batch)   总计"]
out.append(f"  FP16      {P*2/1e9:.0f} GB        ~{32*8192*2*32*32*128*2/1e9:.1f} GB              ~28 GB")
out.append(f"  W8A16     {P*1/1e9:.0f} GB        ~{32*8192*2*32*32*128*2/1e9:.1f} GB              ~21 GB")
out.append(f"  W4A16     {P*0.5/1e9:.1f} GB        ~{32*8192*2*32*32*128*2/1e9:.1f} GB              ~17 GB")
out.append(f"  W4A16 + INT8 KV:   3.5 GB      ~{32*8192*2*32*32*128*1/1e9:.1f} GB (INT8)     ~13 GB")
res("\n".join(out))
mea("Weight-only + KV 量化是 LLM 推理主流,7B 模型单卡 24G 就能跑(单 batch)。")

# --- 2. GPTQ 模拟:Hessian 引导 ---
hdr(2,TOTAL,"GPTQ 二阶信息:Hessian 引导")
why("""GPTQ 思路:用 Hessian 矩阵 H = 2*X^T X 衡量每个权重对输出的影响。
  H 大的权重 → 量化误差放大更多 → 用更精细量化
  H 小的权重 → 误差不放大 → 量化粗一点也无所谓
  逐列量化,每列量化后修正剩余列(贪心)。""")
np.random.seed(0)
W = np.random.randn(8, 8)
X = np.random.randn(100, 8)
H = 2 * X.T @ X / 100
# 简单模拟:per-column 量化,按 H 对角线重要性分配 bits
H_diag = np.diag(H)
out = ["  权重 idx   H_diag (重要性)  bits 分配"]
bits_budget = 4
total = H_diag.sum()
for i in range(8):
    b = max(3, int(8 * H_diag[i] / total * 8 / bits_budget * bits_budget))
    b = min(b, 8)
    out.append(f"  {i:8d}     {H_diag[i]:.3f}              {b} bits")
res("\n".join(out))
mea("GPTQ 关键:不是所有权重同等重要,Hessian 大的(对应重要输入特征)要更多 bit。\n  实际 GPTQ 还做\"逐列修正\",用已量化列的误差补偿后续列,精度进一步提升。")

# --- 3. AWQ:激活感知 ---
hdr(3,TOTAL,"AWQ:激活感知 - 保护\"重要权重\"")
why("""AWQ (Activation-aware Weight Quantization) 观察:
  1% 权重(对应激活幅值 top-1%)贡献了 50%+ 输出幅值
  → 这些权重不能粗暴量化
  AWQ 思路: 缩放对应通道(per-channel scale),让重要权重分布更集中
  简单公式:  W' = W * s,  X' = X / s  (数学等价输出)
  但 s 选\"让 W 重要部分更紧凑\"""")
np.random.seed(0)
W = np.random.randn(8)
X_act = np.abs(np.random.randn(8)) * np.array([0.01, 0.1, 1.0, 5.0, 0.5, 0.3, 10.0, 0.2])
importance = np.abs(W * X_act)
# AWQ 缩放
s = np.sqrt(X_act)   # 简化版本
W_scaled = W * s
X_scaled = X_act / s
res(f"""原始 W  * X_act (输出贡献) = {(W*X_act).round(2)}
  重要性 top-3 权重 idx:  {np.argsort(importance)[-3:][::-1].tolist()}
  AWQ 缩放 s = sqrt(act):  {s.round(2).tolist()}
  缩放后 W' = W * s:        {W_scaled.round(2).tolist()}
  缩放后 X' = X / s:        {X_scaled.round(2).tolist()}
  输出 X' * W' 仍 = 原输出 = {(W_scaled*X_scaled).round(2).tolist()}""")
mea("数学上 W*s * X/s = W*X 完全等价;但 W 缩放后分布更紧凑,INT4 量化误差更小。\n  AWQ 实测 LLaMA-7B INT4 量化, perplexity 比 GPTQ 低 0.1-0.2。")

# --- 4. 实战工具 ---
hdr(4,TOTAL,"主流工具对比")
why("""Weight-only 量化工具(2024+):""")
res("""工具            格式       易用  质量  速度
  bitsandbytes    NF4/INT8   ★★★★  良好  快
  AutoGPTQ        GPTQ       ★★★   优秀  中
  AWQ             AWQ        ★★★   优秀  中
  llama.cpp       Q4_0/Q4_K  ★★★★★  优秀  极快(CPU/GPU)
  vLLM            GPTQ/AWQ   ★★★★  优秀  快
  TensorRT-LLM    INT8/FP8   ★★    极佳  最快""")
mea("""选型:
  - 研究/实验: bitsandbytes (1 行加载)
  - 生产 GPU:  vLLM + GPTQ/AWQ
  - CPU 推理:  llama.cpp (Q4_K_M 是社区默认)
  - 极致性能:  TensorRT-LLM""")
PYEOF

echo ""
echo "############################################################"
cat <<'EOF'
【整体总结】
- 小白:Weight-only = 只量化权重(INT8/INT4),激活保持原精度,显存省 2-4×;
  GPTQ/AWQ 是两大算法,AWQ 略好;bitsandbytes 1 行加载。
- 熟手:per-channel INT4 + group_size=128 是 LLM 推理默认;AWQ 配 g=128 是
  社区 sweet spot;vLLM 加载 GPTQ/AWQ 模型无需转换,直接用。
【进阶】用 AutoGPTQ 量化 LLaMA-7B 到 INT4,对比 FP16 的 perplexity 和显存。
EOF
echo "############################################################"
