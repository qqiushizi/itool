#!/bin/bash
# ============================================================
# 实验: e.rope-fusion
# 说明: RoPE 融合(旋转位置嵌入与 matmul 融合)
# 模块: p.practise 第一性原理学习实验
# ============================================================
# 【第一性原理】
# RoPE (Rotary Position Embedding) = LLM 的位置编码
#   对 Q, K 应用旋转:
#     q_2i   = q_2i * cos(θ) - q_2i+1 * sin(θ)
#     q_2i+1 = q_2i * sin(θ) + q_2i+1 * cos(θ)
#   标准: RoPE 单独 1 个 kernel, 中间结果回 HBM
#   融合: 与 Q/K 矩阵乘 epilogue 合成 1 个
#     节省: 1 次 HBM write + 1 次 HBM read
#   收益: 1.2-1.5× 加速
# ============================================================
set -euo pipefail
if ! python3 -c "import numpy" 2>/dev/null; then python3 -m pip install --quiet numpy; fi
echo "############################################################"
echo "# 实验: e.rope-fusion | RoPE 融合:旋转位置与 Q/K 算子合一"
echo "############################################################"

python3 <<'PYEOF'
import numpy as np
def hdr(n,total,t): print(f"\n{'='*60}\n【步骤 {n}/{total}】{t}\n{'='*60}")
def why(s): print("\n--- 做什么 & 为什么 ---\n  "+s.replace("\n","\n  "))
def res(s): print("\n--- 结果 ---\n  "+s.replace("\n","\n  "))
def mea(s): print("\n--- 解读 ---\n  "+s.replace("\n","\n  "))
TOTAL = 4

# --- 1. RoPE 数学 ---
hdr(1,TOTAL,"RoPE 公式:旋转位置嵌入")
why("""RoPE 把位置信息编码成\"旋转\", 对 Q, K 的每对相邻元素应用 2D 旋转:
  θ_i = base^(-2i/d),  base=10000
  对每个位置 m, 旋转 m*θ_i 角度
  q_2i'   = q_2i * cos(m*θ_i) - q_2i+1 * sin(m*θ_i)
  q_2i+1' = q_2i * sin(m*θ_i) + q_2i+1 * cos(m*θ_i)
  
  优势: 相对位置通过 dot(q_2i, k_2i+1) 自动表达
  训练短, 推理可扩展 (YaRN 外推)""")
def rope(q, base=10000):
    # q: (seq, head_dim)
    seq, d = q.shape
    half = d // 2
    inv_freq = 1.0 / (base ** (np.arange(0, half) * 2 / d))
    pos = np.arange(seq)
    freqs = np.outer(pos, inv_freq)  # (seq, half)
    cos = np.cos(freqs)
    sin = np.sin(freqs)
    # rotate
    q1 = q[:, :half]
    q2 = q[:, half:]
    q_new = np.concatenate([q1 * cos - q2 * sin, q1 * sin + q2 * cos], axis=-1)
    return q_new
np.random.seed(0)
q = np.random.randn(8, 64)
q_rope = rope(q)
res(f"""8 token, head_dim=64, base=10000:
  q 形状: {q.shape}
  q_rope 形状: {q_rope.shape}
  q_rope 前 4 元素: {q_rope[0, :4].round(3).tolist()}
  q 原前 4 元素:   {q[0, :4].round(3).tolist()}""")
mea("RoPE 是\"绝对位置编码的相对化\":\n  - 绝对: 每个位置 m 都有自己的 sin/cos 矩阵\n  - 相对: dot(q_at_m, k_at_n) 只与 (m-n) 有关\n  这是 LLaMA 等模型能处理任意长度位置的原因")

# --- 2. 融合 vs 不融合 ---
hdr(2,TOTAL,"RoPE 融合:epilogue 1 个 kernel")
why("""不融合:
  Q' = Q @ W_Q              (kernel 1, 写 Q' 到 HBM)
  Q'' = RoPE(Q')            (kernel 2, 读 Q', 写 Q'')
  总: 2 次 HBM IO (Q' 来回)
融合:
  Q' = Q @ W_Q + RoPE(...)  (kernel 1, epilogue 算 RoPE)
  总: 1 次 HBM IO (直接写 Q'')
  节省: 50% IO""")
seq, dim = 4096, 128
io_unfused = 2 * (seq * dim * 2)
io_fused = seq * dim * 2
res(f"""Q @ W_Q + RoPE (seq={seq}, dim={dim}):
  不融合:  写 Q' ({seq*dim*2/1e6:.1f} MB) + 读 Q' ({seq*dim*2/1e6:.1f} MB) = {io_unfused/1e6:.1f} MB
  融合:    写 Q'' ({seq*dim*2/1e6:.1f} MB) = {io_fused/1e6:.1f} MB
  节省:    50% IO
  时间:    {io_unfused/2.7e3:.1f} us -> {io_fused/2.7e3:.1f} us""")
mea("LLaMA 32 层 × 2 (Q 和 K) = 64 次 RoPE, 每步省 50% IO.\n  看似小, 但 decode 阶段 (访存密集) 收益明显。")

# --- 3. AscendC 写 fused QKV+RoPE ---
hdr(3,TOTAL,"Fused QKV+RoPE:LLaMA 1 层典型 epilogue")
why("""LLaMA 1 层的 QKV 计算:
  X @ W_Q -> Q', RoPE(Q') -> Q
  X @ W_K -> K', RoPE(K') -> K
  X @ W_V -> V
3 次矩阵乘 + 2 次 RoPE
融合后: 3 个 kernel, 每个 epilogue 算 RoPE (Q, K)
收益: 1.2-1.5× 加速""")
res("""典型 QKV+RoPE 算子 (AscendC 伪代码):
  __global__ __aicore__ void fused_qkv_rope(
    __gm__ float* X,        // 输入 [seq, dim]
    __gm__ float* Wq, __gm__ float* Wk, __gm__ float* Wv,  // 权重
    __gm__ float* cos, __gm__ float* sin,  // RoPE 表
    __gm__ float* Q, __gm__ float* K, __gm__ float* V,
    uint32_t seq, uint32_t dim
  ) {
    __local__ float x_local[TILE];
    __local__ float q_local[TILE], k_local[TILE], v_local[TILE];
    
    // 1. 搬 X
    DataCopy(x_local, X, TILE);
    
    // 2. 算 Q = X @ Wq, epilogue 算 RoPE
    Matmul(q_local, x_local, Wq);
    ApplyRope(q_local, cos, sin, ...);  // 融合 RoPE
    DataCopy(Q, q_local, TILE);
    
    // 3. 类似 K, V
    ...
  }""")
mea("1 个 kernel 算 Q, K, V + RoPE, 极小 IO。\n  vLLM 内部就是这种 fused QKV+RoPE 算子。")

# --- 4. 实战:框架默认 RoPE 融合 ---
hdr(4,TOTAL,"实战:用框架默认 RoPE 融合")
why("""主流框架的 RoPE 融合:""")
out = ["  框架              RoPE 融合            性能"]
out.append("  vLLM            内置 fused RoPE       1.2-1.5x")
out.append("  HuggingFace      apply_rotary (默认)  1.0x (未融)")
out.append("  HF + FlashAttn   一起融合             1.5x")
out.append("  SGLang          内置 fused            1.3x")
out.append("  MindSpeed-LLM   --use-fused-rotary   1.3x")
out.append("  自研 AscendC     QKV+RoPE 1 kernel    1.5x (极致)")
res("\n".join(out))
mea("""实战:
  - 推理: vLLM/SGLang 默认开, 不用手开
  - 训练: HuggingFace 默认未融, 需用 fused impl
  - 长序列: RoPE 扩展 (YaRN) + 融合 RoPE 一起
  - msprof 看 RoPE 占比, < 3% 算健康 (访存密集已优化)""")
PYEOF

echo ""
echo "############################################################"
cat <<'EOF'
【整体总结】
- 小白:RoPE = LLM 位置编码,把位置变成旋转,内积自动表达相对位置;
  融合 RoPE + Q/K 算子,加速 1.2-1.5×,省 50% IO;框架默认开。
- 熟手:Fused RoPE 是 LLM 推理标配,访存密集阶段收益更大;
  LLaMA 32 层 × 2 (Q/K) = 64 次 RoPE,累积可省 ms 级时间;
  训练用 HF 默认未融,需手动切 fused;长序列配合 YaRN 扩位置编码。
【进阶】vLLM 跑 LLaMA 模型,msprof 看 RoPE 算子占比;对比 apply_rotary
  单独 vs fused 的 decode 时延。
EOF
echo "############################################################"
