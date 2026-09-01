#!/bin/bash
# ============================================================
# 实验: f.flash-attention
# 说明: Attention 复杂度、IO-aware、显存节省
# 模块: p.practise 第一性原理学习实验
# ============================================================
# 【第一性原理】
# 标准 attention 算 O = softmax(QK^T)V:
#   1. S = Q @ K^T:  显存 N×N(序列长度的平方!)N=8K 时=64M 项
#   2. P = softmax(S) (要存)
#   3. O = P @ V
# 标准实现要把 S、P 显式存到 HBM,长序列爆显存。
# FlashAttention:
#   - 分块(Tile):把 Q/K/V 按 block 切,从 HBM 搬到 SRAM
#   - 在线 softmax:不存完整 S,逐 block 算 softmax + 累加 O
#   - IO-aware:减少 HBM 读写次数
# 复杂度:计算量 O(N^2 d) 不变,显存 O(N) (从 O(N^2) 降下来!)
#                IO    O(N^2 d^2 / M) → O(N^2 d^2 / M) (略少)
# ============================================================
set -euo pipefail
if ! python3 -c "import numpy" 2>/dev/null; then python3 -m pip install --quiet numpy; fi
echo "############################################################"
echo "# 实验: f.flash-attention | IO-aware 分块 attention"
echo "############################################################"

python3 <<'PY'
import numpy as np
def hdr(n,t,total): print(f"\n{'='*60}\n【步骤 {n}/{total}】{t}\n{'='*60}")
def why(s): print("\n--- 做什么 & 为什么 ---\n  "+s.replace("\n","\n  "))
def res(s): print("\n--- 结果 ---\n  "+s.replace("\n","\n  "))
def mea(s): print("\n--- 解读 ---\n  "+s.replace("\n","\n  "))
TOTAL = 4

# --- 1. 显存:标准 attention O(N^2) vs Flash O(N) ---
hdr(1,TOTAL,"显存:N×N attention matrix 是元凶")
why("""attention 中间 S=softmax(QK^T) 是 N×N 矩阵。
N=序列长,heads=32, FP16:
  N=2048:  32 × 2048^2 × 2B = 256 MB
  N=8192:  32 × 8192^2 × 2B = 4 GB
  N=32768: 32 × 32768^2 × 2B = 64 GB  ← 爆!
FlashAttention 不存 S,P → 显存只跟 N(线性)走。""")
heads = 32; dt = 2
for N in [2048, 4096, 8192, 16384, 32768]:
    std = heads * N*N * dt
    fla = heads * N * 128 * dt * 3  # 约 3 个 N×d_head 块
    print()
res(f"""heads={heads}, FP16:
  N      标准 attention 中间矩阵  FlashAttention 块显存
  2048   {heads*2048**2*dt/1e9:.3f} GB                     ~{heads*2048*128*dt*3/1e9:.3f} GB
  4096   {heads*4096**2*dt/1e9:.3f} GB                     ~{heads*4096*128*dt*3/1e9:.3f} GB
  8192   {heads*8192**2*dt/1e9:.3f} GB                     ~{heads*8192*128*dt*3/1e9:.3f} GB
  16384  {heads*16384**2*dt/1e9:.3f} GB                    ~{heads*16384*128*dt*3/1e9:.3f} GB
  32768  {heads*32768**2*dt/1e9:.3f} GB                    ~{heads*32768*128*dt*3/1e9:.3f} GB""")
mea("""FlashAttention 把 attention 的 N^2 显存变成 N。这是非 transformer 长
上下文能训出来的关键(32K context 以前根本训不了)。""")

# --- 2. toy online softmax ---
hdr(2,TOTAL,"在线 softmax:逐块算 + 数值稳定")
why("""标准 softmax 要先看完所有 x:exp(x_i - max)/Σexp(x_j - max)。
在线版本:流式处理,维护 running max m 和 running sum d:
  m_new = max(m_old, x_new)
  d_new = d_old * exp(m_old - m_new) + sum(exp(x_new - m_new))
  公式等价,但能逐 block 算,无需全 N 数据。""")
np.random.seed(0)
x = np.random.randn(10)*3
# 标准
m = x.max()
p = np.exp(x-m); p = p/p.sum()
# 在线
mb, db = -1e30, 0.0
for xi in x:
    m_new = max(float(mb), xi)
    db = db*np.exp(mb - m_new) + np.exp(xi - m_new)
    mb = m_new
p_online = np.exp(x - mb)/db
res(f"""标准 softmax:  前 5 项 = {p[:5].round(4)}
在线 softmax:    前 5 项 = {p_online[:5].round(4)}
最大差异:        {np.max(np.abs(p-p_online)):.2e}""")
mea("""数学上完全等价,数值上一样稳。FlashAttn 用这个 trick 逐 block
处理 K/V,O = ΣP_ij * V_j 同步累加,每块只需 N×Br×Bc 大小 SRAM。""")

# --- 3. IO 复杂度对比 ---
hdr(3,TOTAL,"IO 复杂度:HBM 读写次数")
why("""A100 HBM 带宽 ~2 TB/s,SRAM ~19 TB/s。差距 ~10×。
标准 attn:HBM 读写 N^2 次(矩阵+中间结果)。
FlashAttn:每块读写 N^2/M 次(M=块大小),总 IO 减少 √M。
为什么:块大小 M 越大,IO 越少,但 SRAM 装不下。M=64~128 是甜点。""")
N = 8192; d = 128
# 标准:每次矩阵乘都过 HBM
std_io = 2 * (N*N*d + N*N + N*N*d)   # 读 QK 写 S 读 S 写 P 读 P 写 O
# Flash:按 M 块
for M in [64, 128, 256]:
    fla_io = 2 * (N*d + N*N*d/M + N*d)
    speedup = std_io / fla_io
    print()
out = []
for M in [64, 128, 256]:
    fi = 2*(N*d + N*N*d/M + N*d)
    out.append(f"  Flash M={M}:            ~{fi/1e6:.1f} M 项,加速 {std_io/fi:.1f}×")
res(f"""N={N}, d={d}, 简化 IO 估算(单位 项数):
  标准 attn 总 HBM 读写: ~{std_io/1e6:.1f} M 项
{chr(10).join(out)}""")
mea("""实际加速 ~2-4×(因为读写比例不均,且 d 也影响)。FlashAttn-2/3 进一步
融合了 softmax / mask / dropout,加速可达 5-10×,显存省 10-20×。""")

# --- 4. FlashAttn 实践:接口和限制 ---
hdr(4,TOTAL,"实战接口 + 限制")
why("""HuggingFace transformers 一行启用:
  model = AutoModelForCausalLM.from_pretrained(..., attn_implementation='flash_attention_2')
限制:
  - 必须有 fp16/bf16(不能 fp32)
  - 需要 Ampere+ GPU(>= sm_80)
  - 不支持某些 attention 变种(如 ALiBi 需手写)
  - dropout 必须在 attn 内部(不能 attn 之后)
  - causal mask 自动处理""")
res("""常见 5 个 hooks:
  flash_attn_func(q, k, v, dropout_p, causal)            # 纯函数
  flash_attn_varlen_func(q, k, v, cu_seqlens)            # 变长序列
  flash_attn_2_cuda(q, k, v, ...)                        # FA2 底层
  transformers' attn_implementation='flash_attention_2'  # HF 一行启用
  attn_implementation='sdpa'                            # PyTorch 自带(自动选 FA/数学)""")
mea("""工程:开 flash_attention_2 + bf16 + grad checkpointing,长序列训
练的标准三件套。Ascend 上对应实现是 FLASHATTN-Ascend / npu_fusion_attention。
真机:在 transformers 库下打开,日志里能看到 'Using flash_attention_2'。""")
PY

echo ""
echo "############################################################"
cat <<'EOF'
【整体总结】
- 小白:标准 attention 要存 N×N 矩阵,长序列爆显存;FlashAttention 把
  它分块处理、用在线 softmax 不用存中间,显存 N^2 → N,加速 2-4×。
- 熟手:FlashAttn + bf16 + grad ckpt 是长序列训练三件套;实现用了
  online softmax + 数值稳定的累加;varlen 模式支持变长 batch。
【进阶】读 FlashAttention 论文(v2/v3);真机打开 attn_implementation='sdpa'
  看 PyTorch 怎么自动选择最优 attn 实现。
EOF
echo "############################################################"
