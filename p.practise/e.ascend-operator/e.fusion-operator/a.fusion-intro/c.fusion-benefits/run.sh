#!/bin/bash
# ============================================================
# 实验: c.fusion-benefits
# 说明: 融合收益分析:访存带宽、IO、kernel launch 量化对比
# 模块: p.practise 第一性原理学习实验
# ============================================================
# 【第一性原理】
# 融合收益 = 节省的 IO + 节省的 launch - 额外开销
# 量化方法:
#   1. IO 节省: 中间结果大小 / HBM 带宽 = 节省时间
#   2. Launch 节省: N_kernel * launch_overhead = 节省时间
#   3. 数据局部性: 算子内执行 vs 跨算子执行 的 cache 命中率
# 收益 = IO 节省 + Launch 节省
# 实测融合后算子耗时:
#   t_fused = max(T_io, T_compute) + T_fused_internal
# 决策: 收益 > 复杂度成本时融合
# ============================================================
set -euo pipefail
echo "############################################################"
echo "# 实验: c.fusion-benefits | 收益分析:访存/IO/launch 量化"
echo "############################################################"

python3 <<'PYEOF'
def hdr(n,total,t): print(f"\n{'='*60}\n【步骤 {n}/{total}】{t}\n{'='*60}")
def why(s): print("\n--- 做什么 & 为什么 ---\n  "+s.replace("\n","\n  "))
def res(s): print("\n--- 结果 ---\n  "+s.replace("\n","\n  "))
def mea(s): print("\n--- 解读 ---\n  "+s.replace("\n","\n  "))
TOTAL = 4

# --- 1. IO 节省量化 ---
hdr(1,TOTAL,"IO 节省 = 中间结果字节 / HBM 带宽")
why("""例: Linear -> Bias -> ReLU 三个算子, N=1024*1024:
  中间结果 out1, out2: 各 2 MB (FP16)
  融合前: 读 out1 (2 MB) + 写 out2 (2 MB) = 4 MB 额外 IO
  融合后: 0 额外 IO (在 UB 完成)
  时间节省: 4 MB / 2.7 TB/s = 1.5 us""")
N = 1024*1024
intermediate = 2 * 2  # 2 MB read + 2 MB write
io_saved_us = intermediate / 2.7
res(f"""Linear+Bias+ReLU (N={N}):
  中间结果:    {intermediate} MB
  节省 IO:     {intermediate} MB
  时间节省:    {io_saved_us:.1f} us (HBM 2.7 TB/s)
  
  如果用 100 个 layer 推理: 100 * 1.5 = 150 us 节省""")
mea("节省看似小, 但 LLM 几十层累加起来, 节省 ms 级别。\n  decode 阶段更明显 (访存密集)。")

# --- 2. Launch 节省 ---
hdr(2,TOTAL,"Launch 节省 = 5-10 us / kernel")
why("""N 个独立 kernel vs 1 个融合 kernel:
  不融合: N * (T_compute + T_launch)
  融合:   1 * (T_compute_total + T_launch + T_internal)
  节省:   (N-1) * T_launch  (简化, 实际还要减 IO)
  5 个 5 us 算子:
    不融合: 5 * 5 = 25 us (compute) + 5 * 7 = 35 us (launch) = 60 us
    融合:   25 us (compute) + 7 us (launch) = 32 us
    节省:   28 us (47%)""")
out = ["  N 算子    不融合耗时   融合耗时   节省"]
for n in [3, 5, 10, 20]:
    compute = n * 5
    unfused = compute + n * 7
    fused = compute + 7
    saved = unfused - fused
    out.append(f"  {n:5d}    {unfused:7d} us     {fused:5d} us    {saved:5d} us ({saved/unfused*100:.0f}%)")
res("\n".join(out))
mea("小算子多时, launch 节省占比巨大 (>40%)。\n  LLM 推理中 RMSNorm/RoPE/Add 这些小算子多, 融合收益大。")

# --- 3. FlashAttn 收益分解 ---
hdr(3,TOTAL,"FlashAttn 收益分解")
why("""FlashAttn 把 5 个算子融成 1 个:
  1. Q@K^T  (matmul)
  2. Scale   (mul)
  3. Mask    (add)
  4. Softmax (exp + sum + div)
  5. P@V     (matmul)
  
  标准实现: 5 个 kernel, 5 次 HBM IO (N×N 矩阵存)
  FlashAttn: 1 个 kernel, 0 次 HBM IO (online softmax)""")
out = ["  指标             标准    FlashAttn   提升"]
out.append("  kernel 数         5       1           5×")
out.append("  HBM IO (B=1)     5*N2   O(N*d)     极大")
out.append("  显存 (8K seq)     64MB    几 MB       10×+")
out.append("  速度 (A100)        1×     2-4×        2-4×")
out.append("  长序列可训性      32K+    128K+       4×+")
res("\n".join(out))
mea("FlashAttn 是融合的\"极致案例\" — 把 IO 复杂度和显存复杂度都降下来。\n  正是因为它, 32K+ 长序列训练成为可能。")

# --- 4. 收益-成本决策 ---
hdr(4,TOTAL,"融合决策:收益 vs 成本")
why("""融合前必问的 5 个问题:""")
out = ["  问题                            答案           决策"]
out.append("  1. 节省 IO 多少?                量化           量化 > 5% 才值得")
out.append("  2. 节省 launch 多少?            N_kernel * 7us N>3 才值得")
out.append("  3. 通用性?                       改 shape?      shape 变 → 难融")
out.append("  4. 调试难度?                     出错定位       难融 → 调试难")
out.append("  5. 团队能力?                     手写 AscendC    无能力 → 不融")
res("\n".join(out))
mea("""决策矩阵:
  收益 > 30% AND 团队有手写能力  → 必融
  收益 > 30% AND 团队无能力       → 用框架默认
  收益 < 30%                      → 不融, 用现状
  
  80% 场景: 用框架默认 (vLLM/SGLang/torch.compile)
  15% 场景: 写 fused kernel (Linear+ReLU, RMSNorm+RoPE)
  5% 场景:  手写极致融合 (FlashAttn)""")
PYEOF

echo ""
echo "############################################################"
cat <<'EOF'
【整体总结】
- 小白:融合节省 2 件事 = IO (中间结果不落 HBM) + launch (少 kernel);
  Linear+ReLU 简单融合省 1.5 us × N 层;FlashAttn 极致融合省 90% IO;
  80% 场景用框架默认融合就够, 手写只用于极致场景。
- 熟手:决策矩阵 5 问 (IO 多少、launch 多少、通用性、调试、团队);
  收益 > 30% 才考虑手写;vLLM/SGLang 已 fused 90% 常见算子;
  msprof 对比融合前后 PipeUtil 验证收益。
【进阶】msprof 跑 fused vs unfused 的 Linear+Bias+ReLU, 量化 IO 节省和
  launch 节省分别多少;vs roofline 下界算实际加速比。
EOF
echo "############################################################"
