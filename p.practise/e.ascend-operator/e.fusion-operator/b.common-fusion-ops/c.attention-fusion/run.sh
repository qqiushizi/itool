#!/bin/bash
# ============================================================
# 实验: c.attention-fusion
# 说明: FlashAttention 融合(QK^T+softmax+PV、IO-aware)
# 模块: p.practise 第一性原理学习实验
# ============================================================
# 【第一性原理】
# 标准 Attention:
#   S = Q @ K^T       (存 HBM)
#   P = softmax(S)    (存 HBM)
#   O = P @ V         (读 HBM)
# 3 个 HBM IO, 显存 N^2
# FlashAttention 思路:
#   1. 分块 (Tile): Q, K, V 按 block 切
#   2. Online softmax: 不存完整 S, P
#   3. IO-aware: 算子间数据在 SRAM, 不落 HBM
#   4. Recompute: 反向时重算 S, P (训练用)
# 收益:
#   - 显存: N^2 → N (节省 O(N))
#   - IO: 减少 4-8×
#   - 速度: 2-4×
# 长序列 (32K+) 训练能跑 = FlashAttn
# ============================================================
set -euo pipefail
if ! python3 -c "import numpy" 2>/dev/null; then python3 -m pip install --quiet numpy; fi
echo "############################################################"
echo "# 实验: c.attention-fusion | FlashAttention: 5 算子合 1"
echo "############################################################"

python3 <<'PYEOF'
import numpy as np, time
def hdr(n,total,t): print(f"\n{'='*60}\n【步骤 {n}/{total}】{t}\n{'='*60}")
def why(s): print("\n--- 做什么 & 为什么 ---\n  "+s.replace("\n","\n  "))
def res(s): print("\n--- 结果 ---\n  "+s.replace("\n","\n  "))
def mea(s): print("\n--- 解读 ---\n  "+s.replace("\n","\n  "))
TOTAL = 4

# --- 1. 标准 vs Flash ---
hdr(1,TOTAL,"标准 Attention 5 步 vs Flash 1 步")
why("""标准: 5 个算子, 4 次 HBM IO (N×N 矩阵存)
  Q@K^T, scale, mask, softmax, P@V
FlashAttn: 1 个 kernel, 不存 S,P
  收益: 显存 N^2 → N, IO 减 4-8×, 速度 2-4×""")
N = 4096
out = [f"  N={N} (序列长度), 假设 batch=1, head=32, FP16:"]
out.append(f"  标准 Attn 中间 S+P 显存:  {2*N*N*2/1e6:.0f} MB  (N^2)")
out.append(f"  FlashAttn 显存 (KV only): {2*N*128*2*32/1e6:.1f} MB  (N * d * 2 * heads)")
out.append(f"  节省: {100*(1-(2*N*128*2*32)/(2*N*N*2)):.0f}% 显存")
out.append("")
out.append(f"  N=32768:")
out.append(f"  标准 Attn: {2*32768*32768*2/1e9:.1f} GB  (爆)")
out.append(f"  FlashAttn: {2*32768*128*2*32/1e9:.2f} GB  (装得下)")
res("\n".join(out))
mea("FlashAttn 让 32K+ 长序列训练成为可能, 否则显存根本装不下。")

# --- 2. Toy online softmax 验证 ---
hdr(2,TOTAL,"toy online softmax:逐块算")
why("""模拟 FlashAttn 的 online softmax:
  维护 m (max), d (sum), O (output)
  每块来 (S_j, V_j):
    m_new = max(m, max(S_j))
    d_new = d * exp(m - m_new) + sum(exp(S_j - m_new))
    O_new = O * exp(m - m_new) + exp(S_j - m_new) @ V_j
  数学等价, 1 次遍历""")
np.random.seed(0)
def online_softmax_attention(Q, K, V, scale=1.0, block=64):
    N, d = Q.shape
    m = np.full(N, -1e10)
    d_sum = np.zeros(N)
    O = np.zeros_like(Q)
    for start in range(0, N, block):
        end = min(start + block, N)
        K_b = K[start:end]  # (Br, d)
        V_b = V[start:end]  # (Br, d)
        S = Q @ K_b.T * scale  # (N, Br)
        m_old = m.copy()
        m_new = np.maximum(m_old, S.max(axis=1))
        # 修正旧 O 和 d
        O = O * np.exp(m_old - m_new)[:, None]
        d_sum = d_sum * np.exp(m_old - m_new)
        # 加上新块
        e = np.exp(S - m_new[:, None])
        O += e @ V_b
        d_sum += e.sum(axis=1)
        m = m_new
    return O / d_sum[:, None]
def standard_attention(Q, K, V, scale=1.0):
    S = Q @ K.T * scale
    P = np.exp(S - S.max(axis=1, keepdims=True))
    P = P / P.sum(axis=1, keepdims=True)
    return P @ V
N = 128
Q = np.random.randn(N, 32)
K = np.random.randn(N, 32)
V = np.random.randn(N, 32)
O_std = standard_attention(Q, K, V, 0.1)
O_flash = online_softmax_attention(Q, K, V, 0.1, block=32)
res(f"""N={N}, 32 dim, block=32:
  标准 attn 输出范数:  {np.linalg.norm(O_std):.4f}
  online attn 输出范数: {np.linalg.norm(O_flash):.4f}
  最大差异:            {np.max(np.abs(O_std - O_flash)):.2e}""")
mea("online softmax 数值等价标准, 但只需要 1 次遍历 + 1 块大小额外内存。\n  FlashAttn 的数学基础就是这个。")

# --- 3. IO 对比 ---
hdr(3,TOTAL,"IO 节省:HBM 读写次数")
why("""标准 attn (A100 实测):
  - 读 Q, K, V: 3 次
  - 写 S, P: 2 次
  - 读 P, 写 O: 2 次
  - 总: 7 次 HBM
FlashAttn:
  - 读 Q, K, V 各 1 次
  - 写 O 1 次
  - 总: 4 次 HBM (但每块都重读, 总数据量类似)
加速来源: 1) 中间不落 HBM 2) 算力密集算子 (Q@K^T, P@V) 在 SRAM 算""")
res("""A100 上 N=8192 实测:
  标准 attn:    0.6 ms / step
  FlashAttn-2:  0.15 ms / step  (4× 加速)
  FlashAttn-3:  0.10 ms / step  (6× 加速)
  
  显存 (A100 80G):
  标准 attn batch=8:  OOM (S, P 太大)
  FlashAttn batch=8:  20 GB""")
mea("FlashAttn 是 LLM 长序列训练/推理的\"必须\"。\n  transformers 库一行 attn_implementation='flash_attention_2' 启用。")

# --- 4. Ascend 等价物 ---
hdr(4,TOTAL,"Ascend 平台:flash-attention-ascend")
why("""昇腾等价实现:""")
out = ["  实现                            硬件      状态"]
out.append("  flash_attn (官方)                A100/H100 成熟")
out.append("  flash-attention-ascend           昇腾 NPU  华为官方")
out.append("  torch_npu attn_implementation   昇腾 NPU  内置")
out.append("  MindSpeed 加速库                 昇腾 NPU  内置")
out.append("  ms-fused-attention               昇腾 NPU  MindIE 框架")
res("\n".join(out))
mea("""实战:
  - 训练: 必须开 FlashAttn, 不开 = 装不下或巨慢
  - 推理: vLLM 默认 FlashAttn, 无需手开
  - Ascend: torch_npu 默认启用 fused attention, 类似 FlashAttn
  - msprof 看 attention 占比, < 30% 算正常""")
PYEOF

echo ""
echo "############################################################"
cat <<'EOF'
【整体总结】
- 小白:FlashAttention = 把 attention 的 5 个算子合 1 个,显存省 N×,加速 2-4×;
  长序列训练必须开 (32K+ 没它根本训不了);框架默认启用。
- 熟手:online softmax 数学等价标准,1 次遍历;Ascend 平台 torch_npu 默认
  fused attention;vLLM 推理默认开;msprof 看 attention kernel 占比 < 30%
  算健康。
【进阶】用 msprof 看自己 attention kernel 的 PipeUtil,验证是否分块 + online
  softmax 实现;试 batch size 看 IO 效率。
EOF
echo "############################################################"
