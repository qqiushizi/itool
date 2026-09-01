#!/bin/bash
# ============================================================
# 实验: c.memory-bound
# 说明: 访存密集 vs 计算密集、roofline
# 模块: p.practise 第一性原理学习实验
# ============================================================
# 【第一性原理】
# Roofline 模型: 算子性能受限于 2 个上限:
#   - 计算上限: 算力 FLOPS
#   - 访存上限: 带宽 BW
# 算子强度 (I) = FLOPs / 字节数
# 性能 P = min(FLOPS_peak, I * BW)
# - I < FLOPS/BW: 访存密集 (memory-bound)
# - I > FLOPS/BW: 算力密集 (compute-bound)
# LLM 推理算子:
#   - Linear (大):  compute-bound
#   - Attention (decode): memory-bound
#   - LayerNorm: memory-bound
#   - Softmax: memory-bound
#   - Activation: memory-bound
# ============================================================
set -euo pipefail
if ! python3 -c "import numpy" 2>/dev/null; then python3 -m pip install --quiet numpy; fi
echo "############################################################"
echo "# 实验: c.memory-bound | 访存 vs 算力,roofline 模型"
echo "############################################################"

python3 <<'PYEOF'
def hdr(n,t,total): print(f"\n{'='*60}\n【步骤 {n}/{total}】{t}\n{'='*60}")
def why(s): print("\n--- 做什么 & 为什么 ---\n  "+s.replace("\n","\n  "))
def res(s): print("\n--- 结果 ---\n  "+s.replace("\n","\n  "))
def mea(s): print("\n--- 解读 ---\n  "+s.replace("\n","\n  "))
TOTAL = 4

# --- 1. Roofline 公式 ---
hdr(1,TOTAL,"Roofline 模型:算力 vs 访存")
why("""Roofline = 性能与\"算子强度 (I)\"的关系:
  P = min(FLOPS_peak, I * BW)
  - I 小: P = I * BW (访存上限,斜线)
  - I 大: P = FLOPS_peak (算力上限,水平线)
  拐点在 I = FLOPS / BW""")
FLOPS_A100 = 312e12   # FP16
BW_A100 = 2e12        # 2 TB/s
I_turn = FLOPS_A100 / BW_A100
res(f"""A100 关键参数:
  FP16 算力 (FLOPS_peak): {FLOPS_A100/1e12:.0f} TFLOPS
  HBM 带宽 (BW):         {BW_A100/1e12:.1f} TB/s
  拐点算子强度:           I = {I_turn:.0f} FLOPs/Byte
  
  性能:
    I < {I_turn:.0f} (访存密集):  P = I * {BW_A100/1e12:.1f} TFLOPS
    I > {I_turn:.0f} (算力密集):  P = {FLOPS_A100/1e12:.0f} TFLOPS""")
mea("A100 上 156 FLOPs/Byte 是分水岭。\n  decode 阶段 attention I ≈ 50, 远小于 156 → 访存密集 → 受带宽限制。\n  prefill 阶段 attention I ≈ 500, 远超 156 → 算力密集 → 算力能打满。")

# --- 2. 各算子算子强度 ---
hdr(2,TOTAL,"各算子的 I (FLOPs/Byte)")
why("""计算典型 LLM 算子的 I:""")
out = ["  算子                I (FLOPs/Byte)   类型"]
out.append("  Linear (4096x4096)  ~500             算力密集")
out.append("  Linear (12288x4096) ~1500            算力密集 (MLP up)")
out.append("  Attention decode    ~50              访存密集")
out.append("  Attention prefill   ~500             算力密集")
out.append("  RMSNorm             ~10              访存密集")
out.append("  Softmax             ~10              访存密集")
out.append("  RoPE                ~5               访存密集")
out.append("  SwiGLU/GeLU         ~5-20            访存密集")
out.append("  Embedding lookup    ~1               访存密集")
res("\n".join(out))
mea("""LLM 推理算子强度谱:
  算力密集: Linear (尤其大 mlp)、Attention prefill
  访存密集: Attention decode、所有 normalization、激活、embedding
  → 优化\"访存密集\"算子是 LLM 推理性能的关键 (fuse, FP8, INT8)""")

# --- 3. LLM decode 阶段算子耗时占比 ---
hdr(3,TOTAL,"LLM decode 各算子耗时占比(经验)")
why("""7B 模型, decode 一步各算子时间占比:""")
out = ["  算子            占比    优化空间"]
out.append("  Attention       35%     FlashAttn / INT8 KV")
out.append("  Linear (Q,K,V,O) 30%    INT8 / 异步")
out.append("  MLP (gate,up,down) 25%   INT8 / fuse")
out.append("  LayerNorm       5%      fuse")
out.append("  其他 (RoPE等)    5%      fuse")
res("\n".join(out))
mea("""Attention 占最大头,因为每步都要读 KV cache。
  优化 Attention 收益最大:
  - FlashAttn: 把 N×N 矩阵融合, 减少 IO
  - INT8 KV:  KV 减半, 读带宽减半
  - 异步 attention: 跟其他算子 overlap""")

# --- 4. 实测验证:算力 vs 访存 ---
hdr(4,TOTAL,"优化启示:对症下药")
why("""不同瓶颈不同药方:""")
res("""瓶颈类型      症状                  药方
  算力密集      GPU 算力跑满          升精度 / 升 batch / 切 TP
  访存密集      GPU 利用率 < 50%      融合 / 量化 / FlashAttn
  通信密集      NCCL 占比 > 20%      overlap / 切张量并行
  调度密集      GPU 等 launch        CUDA Graph / 持久 kernel""")
mea("""判断方法:
  1. nvidia-smi 看 GPU 利用率
     - 算力密集: 80%+
     - 访存密集: 30-50%
  2. nsys / Nsight 看 kernel 占比
  3. 算子强度对照 roofline
  4. 用 vLLM SLO dashboard 看 TTFT/TPOT 瓶颈

  LLM decode 阶段典型访存密集,所以推理优化都聚焦在\"省 IO\"。""")
PYEOF

echo ""
echo "############################################################"
cat <<'EOF'
【整体总结】
- 小白:Roofline 模型 = 算子性能受算力或带宽上限,LLM decode 大多访存密集;
  Attention 占 decode 35%,KV cache 读取是瓶颈;优化\"省 IO\"是核心。
- 熟手:算子强度 I = FLOPs/Byte,A100 拐点 156;Linear I=500+ 算力密集,
  Attention decode I≈50 访存密集;优化 attention + 融合 norm/激活;
  nsys profile 看哪个 kernel 占比大,针对性 fusion 或量化。
【进阶】用 nsys profile 自己的模型,识别 top-5 kernel,逐个融合或量化。
EOF
echo "############################################################"
