#!/bin/bash
# ============================================================
# 实验: e.long-context
# 说明: 长上下文显存/延迟、YaRN/外推
# 模块: p.practise 第一性原理学习实验
# ============================================================
# 【第一性原理】
# 长上下文 (32K-1M) 三个核心挑战:
#   1. 显存: KV cache O(seq²) - 实际上 KV cache 是 O(seq × layers)
#   2. 注意力: 标准 attention O(seq²) - FlashAttn O(seq) 显存
#   3. 位置编码: 训练时只见过 4K, 长 context 怎么办?
# 外推 (extrapolation) 方法:
#   - 线性缩放 RoPE 频率 (位置 × 0.25)
#   - YaRN: 分频段不同缩放 (低频拉伸, 高频保留)
#   - ALiBi: 训练时不带位置, 加 bias
#   - xPos: 指数衰减位置
#   - LongRoPE / Self-Extend: 重新搜索频率
# ============================================================
set -euo pipefail
if ! python3 -c "import numpy" 2>/dev/null; then python3 -m pip install --quiet numpy; fi
echo "############################################################"
echo "# 实验: e.long-context | 长上下文:显存/延迟/外推"
echo "############################################################"

python3 <<'PYEOF'
import numpy as np
def hdr(n,t,total): print(f"\n{'='*60}\n【步骤 {n}/{total}】{t}\n{'='*60}")
def why(s): print("\n--- 做什么 & 为什么 ---\n  "+s.replace("\n","\n  "))
def res(s): print("\n--- 结果 ---\n  "+s.replace("\n","\n  "))
def mea(s): print("\n--- 解读 ---\n  "+s.replace("\n","\n  "))
TOTAL = 4

# --- 1. KV cache 显存 ---
hdr(1,TOTAL,"长上下文:KV cache 显存")
why("""LLaMA-7B (32 层, 32 heads, head_dim 128) 单请求 KV cache:
  KV = 2 × 32 × seq × 32 × 128 × 2 bytes
  = 524288 bytes/seq token
  32K context:  524288 × 32768 = 16 GB 单请求!
  128K context: 64 GB 单请求 → 单卡根本装不下""")
L, h, d_h, n_layers = 32, 32, 128, 32
def kv_gb(seq, batch=1):
    return 2 * n_layers * h * d_h * seq * batch * 2 / 1e9
out = [f"  seq_len     单请求 KV (FP16)"]
for seq in [4096, 8192, 16384, 32768, 65536, 131072]:
    out.append(f"  {seq:8d}     {kv_gb(seq):.1f} GB")
res("\n".join(out))
mea("""32K context 7B 模型单请求 16 GB → 8 卡 A100 80G 也只能跑 ~4 并发。
这就是为什么长上下文推理需要:
  1. KV 量化 (INT8 → 8GB, INT4 → 4GB)
  2. KV cache 卸载到 CPU/NVMe
  3. PagedAttention + block 共享
  4. Sliding window / 稀疏 attention""")

# --- 2. 标准 attention vs 稀疏/线性 ---
hdr(2,TOTAL,"长 attention 复杂度对比")
why("""标准 self-attention: O(N²) 显存 + FLOPs
  N=32K 时 N²=1G → 64 GB FP16 显存
  N=128K 时 16G → 1 TB 显存
变体:
  - FlashAttn: 显存 O(N),FLOPs 仍 O(N²)
  - Sliding window: 只看左右各 W, 显存 O(NW)
  - 稀疏 (Longformer): O(N√N)
  - 线性 (Mamba/RetNet): O(N) 显存+FLOPs""")
out = [f"  方法              显存复杂度   FLOPs     32K ctx (GB)"]
out.append("  Standard Attn     O(N²)       O(N²)     16")
out.append("  FlashAttn         O(N)        O(N²)     0.05 (KV only)")
out.append("  Sliding W=1024    O(NW)       O(NW)     0.13")
out.append("  Longformer        O(N√N)      O(N√N)    ~2")
out.append("  Linear (Mamba)    O(N)        O(N)      0.05")
res("\n".join(out))
mea("""长上下文主流方案:
  - 已有模型:  FlashAttn + KV 量化 + 滑动窗口
  - 重新设计:  Mamba/RetNet/Megabyte 等线性 attention
  - 折中:      Mistral sliding window + 全局 token""")

# --- 3. 位置外推 ---
hdr(3,TOTAL,"位置外推:训练 4K 怎么用 32K")
why("""RoPE 训练时 base=10000, 频率 θ_i = base^(-2i/d), 训练时见过 0~4096。
  外推到 32K 时, 位置编码的\"相邻距离\"被破坏。
  YaRN 思路: 不同频率分量用不同缩放
  - 低频 (大 i, 长程位置): 拉伸 → 看到更大位置
  - 高频 (小 i, 短程位置): 保留 → 保持局部分辨力""")
# RoPE base
def rope_freq(base, dim, seq):
    inv_freq = 1.0 / (base ** (np.arange(0, dim, 2) / dim))
    pos = np.arange(seq)
    freqs = np.outer(pos, inv_freq)
    return freqs
base = 10000
dim = 128
seq = 32768
f_normal = rope_freq(base, dim, seq)
# YaRN 拉伸: 低频 × scale
scale = 8  # 32K / 4K
f_yarn = f_normal.copy()
# 低频 (i > dim/4) 才拉伸
threshold = dim // 4
f_yarn[:, threshold:] *= scale
res(f"""RoPE 32K, base=10000:
  最高频率分量:        {f_normal[0,0]:.4f}    (局部, 短程信息)
  最低频率分量:        {f_normal[-1,-1]:.6f}  (全局, 长程信息)
  YaRN scale=8 后最低:  {f_yarn[-1,-1]:.6f}
  效果: 高频保留(局部好), 低频拉伸(看到更大位置)""")
mea("""YaRN 公式: θ'_i = θ_i / s, 低频 (i > threshold) 用 s > 1
  实际 LLaMA-2 7B 用 YaRN 扩到 32K/64K, ppl 涨 < 0.1
  训练时改的 0 步,推理时直接换位置编码""")

# --- 4. 长上下文性能实践 ---
hdr(4,TOTAL,"长上下文性能实践(7B 模型 A100 80G)")
why("""不同 context 长度下,能跑多少并发 + 延迟:""")
out = ["  context   单请求 KV   最大并发    TTFT     TPOT"]
for ctx, kv in [(4096, kv_gb(4096)), (8192, kv_gb(8192)), (16384, kv_gb(16384)), (32768, kv_gb(32768)), (65536, kv_gb(65536))]:
    # 80G 显存, 模型 ~14G, 余 ~60G 给 KV + 激活
    free = 60
    conc = max(1, int(free / (kv + 0.5)))
    ttft = ctx * 0.05  # 估
    tpot = 30 + ctx / 1000 * 5
    out.append(f"  {ctx:6d}    {kv:.1f} GB    ~{conc:3d}        {ttft:6.0f}ms  {tpot:5.1f}ms")
res("\n".join(out))
mea("""优化长上下文:
  1. KV cache 量化 INT8 → 并发翻倍
  2. chunked-prefill → TTFT 稳
  3. YaRN 扩位置编码 → 模型无需重训
  4. Sliding window → 长但仅看局部
  5. KV cache 卸载 CPU/NVMe → 超长 context
  6. 线性 attention (Mamba 类) → 几乎无限长""")
PYEOF

echo ""
echo "############################################################"
cat <<'EOF'
【整体总结】
- 小白:长上下文 = KV cache 涨 + attention 复杂度 O(N²);FlashAttn + KV 量化
  + YaRN 扩位置编码,7B 模型 A100 能跑 32K;滑动窗口和线性 attention 是新方向。
- 熟手:32K context 单请求 16GB KV,8 卡 A100 仅能跑 4 并发;YaRN 训练 0 步
  直接扩;PagedAttention 让长 context 服务可行;生产常用 KV 量化 + 卸载。
【进阶】用 vLLM 实测 4K/8K/32K/128K 下的 TTFT/TPOT 趋势;打开 YaRN 跑 needle
  in haystack 测试召回率。
EOF
echo "############################################################"
