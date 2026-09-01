#!/bin/bash
# ============================================================
# 实验: b.elementwise-op
# 说明: Elementwise/激活、Vector
# 模块: p.practise 第一性原理学习实验
# ============================================================
# 【第一性原理】
# Elementwise 算子 = 逐元素操作 (与 GEMM 区分):
#   - Add / Sub / Mul / Div
#   - ReLU / GELU / SiLU / SwiGLU
#   - Cast (类型转换)
#   - 比较/逻辑 (Eq, Lt, Where)
#   - 三角函数 (Sin, Cos)
#   走 Vector 单元 (4096-bit 宽, 每 cycle 4096/16 = 256 个 FP16)
#   特点: 访存密集 (I ≈ 1-5 FLOPS/Byte)
#   优化: 减少 IO, 融合 (fuse) 多个 elementwise
# ============================================================
set -euo pipefail
if ! python3 -c "import numpy" 2>/dev/null; then python3 -m pip install --quiet numpy; fi
echo "############################################################"
echo "# 实验: b.elementwise-op | Elementwise: Vector 单元 + 融合"
echo "############################################################"

python3 <<'PYEOF'
import numpy as np, time
def hdr(n,total,t): print(f"\n{'='*60}\n【步骤 {n}/{total}】{t}\n{'='*60}")
def why(s): print("\n--- 做什么 & 为什么 ---\n  "+s.replace("\n","\n  "))
def res(s): print("\n--- 结果 ---\n  "+s.replace("\n","\n  "))
def mea(s): print("\n--- 解读 ---\n  "+s.replace("\n","\n  "))
TOTAL = 4

# --- 1. 算子分类 ---
hdr(1,TOTAL,"Elementwise 算子分类")
why("""LLM 中常见的 elementwise 算子:""")
out = ["  算子      公式                        出现位置"]
out.append("  Add       z = x + y                   residual")
out.append("  ReLU      y = max(0, x)               activation")
out.append("  GELU      y = x * Φ(x)                transformer")
out.append("  SiLU      y = x * sigmoid(x)          LLaMA FFN")
out.append("  SwiGLU    y = x * sigmoid(g) * g      LLaMA gate")
out.append("  Mul       z = x * y                   attention scoring")
out.append("  Cast      类型转换                    mixed precision")
out.append("  Where     z = x if c else y            mask")
out.append("  Exp/Log   数学函数                    softmax / norm")
res("\n".join(out))
mea("Elementwise 算力小但多, 占 LLM 推理时间 20-30%。\n  优化: 把它们 fuse 到 GEMM 后面 (Linear+Bias+ReLU 一个 kernel)")

# --- 2. Vector 单元 ---
hdr(2,TOTAL,"Vector 单元:4096-bit 宽")
why("""Vector 单元 = SIMD 处理器, 每 cycle 处理 4096 bit 数据。
  FP16 (16 bit) → 256 个元素/cycle
  FP32 (32 bit) → 128 个元素/cycle
  INT8 (8 bit)  → 512 个元素/cycle
  算力 = frequency * 256 (FP16)
  A2 频率 1.6 GHz → 410 GFLOPS Vector 算力 (FP16)""")
res(f"""Vector 单元 (A2):
  数据宽度:    4096 bit
  FP16 吞吐:   256 ops/cycle
  频率:        1.6 GHz
  Vector 算力: {256*1.6e9/1e9:.0f} GFLOPS FP16
  
  对比 Cube:
    Cube FP16:  280 TFLOPS
    Vector FP16: 0.4 TFLOPS
    差 700×""")
mea("Cube 算 GEMM, Vector 算 elementwise, 各自擅长。\n  双发射让 Cube 和 Vector 同时跑, 是昇腾核心优势。")

# --- 3. 实测:elementwise 性能 ---
hdr(3,TOTAL,"CPU 模拟: elementwise 性能")
why("""CPU 上 ReLU 和 GELU 的速度对比:""")
def relu(x): return np.maximum(x, 0)
def gelu(x): return x * 0.5 * (1 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715*x**3)))
x = np.random.randn(1024*1024).astype(np.float32)
t = time.perf_counter()
for _ in range(100): r = relu(x)
t_relu = (time.perf_counter()-t)/100
t = time.perf_counter()
for _ in range(100): r = gelu(x)
t_gelu = (time.perf_counter()-t)/100
res(f"""CPU 1M 元素, 100 次:
  ReLU:  {t_relu*1000:.3f} ms / 次
  GELU:  {t_gelu*1000:.3f} ms / 次
  GELU 慢 {t_gelu/t_relu:.1f}× (有 exp/tanh)""")
mea("ReLU/GELU 差 5-10×, 说明 elementwise 算子性能差异在公式复杂度。\n  实战: 简单算子 (ReLU) 几乎免费, 复杂算子 (GELU) 可能成为瓶颈。")

# --- 4. 算子融合 ---
hdr(4,TOTAL,"算子融合:Fused Linear+Activation")
why("""5 个 elementwise 各 1 个 kernel: 5 launch + 5 HBM write/read
  融合成 1 个 kernel: 1 launch + 1 HBM
  加速: 1.5-2×""")
out = ["  形式                  耗时 (us,估)  备注"]
out.append("  5 个独立 kernel       50            5 launch + 5 IO")
out.append("  融合 1 个 kernel      25            1 launch + 1 IO")
out.append("  GEMM + 5 elem         100           GEMM 不变, 5 elem 25 us")
out.append("  GEMM + fused elem     80            节省 20 us 激活时间")
res("\n".join(out))
mea("""融合原则:
  1. 简单 elementwise (Add, ReLU) 必融
  2. 复杂 (GELU, SiLU) 尽量融
  3. 需要 reduction 的 (softmax) 难融
  4. 大型框架 (torch_npu) 自动融合
  5. 手写 AscendC 时手动实现融合算子""")
PYEOF

echo ""
echo "############################################################"
cat <<'EOF'
【整体总结】
# - 小白:Elementwise 算子 = 逐元素操作 (Add, ReLU, GELU...), 走 Vector 单元;
#   访存密集,优化靠融合;简单算子 (ReLU) 几乎免费,复杂 (GELU) 需注意。
# - 熟手:Vector 算力 0.4 TFLOPS 比 Cube 低 700×,但双发射让两者并行;
#   5 个 elementwise 融合 1 个 kernel 加速 1.5-2×;复杂激活 (SwiGLU) 也可融;
#   msprof 看 vector 单元利用率找优化点。
# 【进阶】用 AscendC 写一个 fused Linear+Bias+ReLU 算子, 对比 3 个独立 kernel
#   的耗时。
EOF
echo "############################################################"
