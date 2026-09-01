#!/bin/bash
# ============================================================
# 实验: a.why-fusion
# 说明: 为什么要融合:减少访存、kernel launch 开销、中间结果落盘
# 模块: p.practise 第一性原理学习实验
# ============================================================
# 【第一性原理】
# 算子融合 = 把多个独立算子合成 1 个 kernel。
# 解决 3 个核心问题:
#   1. 减少访存: 中间结果不落 HBM, 省 IO 带宽
#   2. 减少 launch: 多个 kernel 变 1 个, 省调度开销
#   3. 数据局部性: 计算单元 + 数据都在片上, 快
# 收益:
#   - 简单融合 (Linear+ReLU): 1.3-1.5×
#   - 复杂融合 (FlashAttn): 2-5×
#   - LLM 推理 fused 后整体: 1.5-2× 加速 + 30-50% 显存省
# ============================================================
set -euo pipefail
echo "############################################################"
echo "# 实验: a.why-fusion | 为什么要融合:访存 + launch + 局部性"
echo "############################################################"

python3 <<'PYEOF'
def hdr(n,total,t): print(f"\n{'='*60}\n【步骤 {n}/{total}】{t}\n{'='*60}")
def why(s): print("\n--- 做什么 & 为什么 ---\n  "+s.replace("\n","\n  "))
def res(s): print("\n--- 结果 ---\n  "+s.replace("\n","\n  "))
def mea(s): print("\n--- 解读 ---\n  "+s.replace("\n","\n  "))
TOTAL = 4

# --- 1. 融合 3 大动机 ---
hdr(1,TOTAL,"融合的 3 大动机")
why("""为什么要融合算子?""")
out = ["  动机          节省                  适用"]
out.append("  减少 IO       1-3 次 HBM 读写/算子  访存密集算子 (elementwise, norm)")
out.append("  减少 launch   5-10 us / kernel     小算子 (1-10 us) 调度主导")
out.append("  数据局部性    算子数据留在片上      tile 可装下整个数据流")
res("\n".join(out))
mea("3 个动机常叠加。LLM 推理 90% 算子能从融合中受益。\n  极端例子: FlashAttn 把 5-6 个小算子合成 1 个, 加速 2-5×, 显存省 N×。")

# --- 2. IO 节省量化 ---
hdr(2,TOTAL,"IO 节省量化")
why("""Linear + Bias + ReLU, 1024x1024 中间结果:
  不融合:
    Linear: 读 W (4 MB), 读 X (2 MB), 写 out1 (2 MB)  = 8 MB
    Bias:   读 out1 (2 MB), 写 out2 (2 MB)              = 4 MB
    ReLU:   读 out2 (2 MB), 写 out3 (2 MB)              = 4 MB
    总 IO: 16 MB
  融合:
    读 W (4 MB), 读 X (2 MB), 算, 写 out (2 MB) = 8 MB
    节省: 50%""")
N = 1024*1024
io_unfused = (4 + 2 + 2) + (2 + 2) + (2 + 2)
io_fused = 4 + 2 + 2
res(f"""Linear+Bias+ReLU (1024^2 中间):
  不融合 IO:  {io_unfused} MB  (6 次 HBM 访问)
  融合 IO:    {io_fused} MB  (3 次 HBM 访问)
  节省:       {(1-io_fused/io_unfused)*100:.0f}%
  
  访存时间 (2.7 TB/s):
    不融合:  {io_unfused/2.7e3:.2f} us
    融合:    {io_fused/2.7e3:.2f} us
    节省:    {(1-io_fused/io_unfused)*100:.0f}%""")
mea("IO 节省直接转化为时间节省。在 LLM decode 阶段(访存密集)尤其显著。")

# --- 3. Launch 开销 ---
hdr(3,TOTAL,"Launch 开销:5-10 us / kernel")
why("""每次 kernel launch 都有 CPU → device 的调度:
  - CUDA stream submit: ~5 us
  - 算子参数准备: ~1 us
  - 实际启动: ~1 us
  累计: ~7 us
  10 个小算子 = 70 us, 可能比计算本身还慢""")
out = ["  算子              实际算    launch    总计"]
out.append("  Linear (大)        100 us    7 us      107 us")
out.append("  Bias (小)         2 us      7 us      9 us    ← 78% 是 launch")
out.append("  ReLU (小)         2 us      7 us      9 us    ← 78% 是 launch")
out.append("  Add (小)          2 us      7 us      9 us    ← 78% 是 launch")
out.append("  LayerNorm (小)    5 us      7 us      12 us   ← 58% 是 launch")
res("\n".join(out))
mea("小算子 launch 开销占比巨大。\n  解决: 1) 融合 (减 kernel 数) 2) CUDA Graph (批量 launch)")

# --- 4. 收益与风险 ---
hdr(4,TOTAL,"融合的收益与风险")
why("""融合不是万灵药, 也有代价:""")
out = ["  收益                          风险"]
out.append("  - 1.3-2× 加速 (简单融合)      - 开发成本高, 手写 kernel")
out.append("  - 1.5-3× 显存省 (中间不存)    - 通用性差, 改形状要重写")
out.append("  - 减少 launch, 小算子收益大    - 调试难, 错就错一片")
out.append("  - 编译器易优化                 - 大融合可能 register 溢出")
out.append("  - 适合 LLM 推理(算子固定)      - 训练 shape 变化大, 难融合")
res("\n".join(out))
mea("""最佳实践:
  1. 简单 + 收益大: 必融 (Linear+ReLU, Add+Bias)
  2. 复杂 + 收益大: 团队有能力就融 (FlashAttn)
  3. 复杂 + 收益小: 不融, 调框架默认实现
  4. 融合前先 profile, 别\"以为优化\"""")
PYEOF

echo ""
echo "############################################################"
cat <<'EOF'
【整体总结】
- 小白:融合 = 把多个 kernel 合成 1 个, 减少 IO + launch + 数据局部性;
  简单融合 (Linear+ReLU) 加速 1.3-1.5×, 复杂 (FlashAttn) 2-5×;LLM 推理几乎都融。
- 熟手:小算子 launch 开销占比 78%, 必须融合或用 CUDA Graph;IO 节省 = 数据
  不落 HBM, decode 阶段 (访存密集) 收益最大;手写 kernel 通用性差, 训练 shape
  变化时需重写;profile 后再融, 别\"以为优化\"。
【进阶】用 torch.compile 包装自己的小模型, 对比 fused vs unfused 的 kernel
  数和耗时。
EOF
echo "############################################################"
