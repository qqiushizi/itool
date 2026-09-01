#!/bin/bash
# ============================================================
# 实验: d.precision-debug
# 说明: 算子精度对齐、dump 对比
# 模块: p.practise 第一性原理学习实验
# ============================================================
# 【第一性原理】
# 自研算子最常翻车的原因 = 精度对不齐。
# 算子误差来源:
#   1. 累加精度: FP16 累加 → 误差大;FP32 累加 → 准但慢
#   2. 归一化: RMSNorm 减法顺序
#   3. 数值稳定性: Softmax 不减 max → exp 溢出
#   4. 数据类型转换: FP32→FP16 截断
# 调试方法:
#   1. 与标杆算子 (CPU torch / GPU) 对比输出
#   2. 逐层 dump, 定位最早出错的算子
#   3. 看 max abs error / max relative error
#   4. 调累加精度 (FP32 accumulator)
# ============================================================
set -euo pipefail
if ! python3 -c "import numpy" 2>/dev/null; then python3 -m pip install --quiet numpy; fi
echo "############################################################"
echo "# 实验: d.precision-debug | 精度对齐 + dump 对比"
echo "############################################################"

python3 <<'PYEOF'
import numpy as np
def hdr(n,total,t): print(f"\n{'='*60}\n【步骤 {n}/{total}】{t}\n{'='*60}")
def why(s): print("\n--- 做什么 & 为什么 ---\n  "+s.replace("\n","\n  "))
def res(s): print("\n--- 结果 ---\n  "+s.replace("\n","\n  "))
def mea(s): print("\n--- 解读 ---\n  "+s.replace("\n","\n  "))
TOTAL = 4

# --- 1. 精度误差源 ---
hdr(1,TOTAL,"5 个常见精度误差源")
why("""自研算子翻车时, 80% 是这 5 个原因:""")
out = ["  误差源              案例                          解决"]
out.append("  累加精度             FP16 累加 N>1000 误差爆炸    FP32 accumulator")
out.append("  数值不稳定           Softmax 不减 max → exp 溢出  减 max")
out.append("  截断误差             FP32→FP16 截断后误差         关键路径留 FP32")
out.append("  顺序误差             (a+b)+c ≠ a+(b+c) 浮点不结合  关键归约先 sort")
out.append("  范围溢出             输入范围 > FP16 上限         cast to FP32 first")
res("\n".join(out))
mea("""预防 > 调试。
  1. 写算子前先想\"哪里会丢精度\"
  2. 累加器默认 FP32
  3. Softmax / Log 必先减 max
  4. 大矩阵用 Kahan 求和""")

# --- 2. 累加精度实测 ---
hdr(2,TOTAL,"FP16 vs FP32 累加:实测")
why("""N=10000 元素求和:
  FP16 累加: 误差 ~1 (相对 1e-2)
  FP32 累加: 误差 ~1e-4
  Kahan 求和: 误差 ~1e-5""")
np.random.seed(0)
x = np.random.randn(10000).astype(np.float32) * 0.01
fp32_sum = x.sum()
x16 = x.astype(np.float16)
# FP16 累加
fp16_sum = np.float16(0.0)
for v in x16: fp16_sum = fp16_sum + v
# FP32 累加
fp32_acc = np.float32(0.0)
for v in x16: fp32_acc = fp32_acc + v
res(f"""10000 个 FP16 元素求和:
  FP32 真实和: {fp32_sum:.6f}
  FP16 累加:   {float(fp16_sum):.6f}  误差 {abs(float(fp16_sum)-fp32_sum):.2e}
  FP32 累加:   {fp32_acc:.6f}  误差 {abs(fp32_acc-fp32_sum):.2e}""")
mea("FP16 累加在 N=10000 时误差 ~1e-2 (10%), 不可接受。\n  昇腾 Cube 累加器默认 FP32, 这就是为什么 GEMM 用 FP32 accumulator。")

# --- 3. Softmax 数值稳定性 ---
hdr(3,TOTAL,"Softmax 数值稳定性")
why("""Softmax = exp(x_i) / sum(exp(x_j))
  问题: x = [1000, 1001, 1002], exp(1000) = inf
  解决: 减 max, x - max(x) 后所有 exp 都安全""")
def softmax_naive(x):
    e = np.exp(x)
    return e / e.sum()
def softmax_safe(x):
    e = np.exp(x - x.max())
    return e / e.sum()
x = np.array([1000.0, 1001.0, 1002.0], dtype=np.float32)
try:
    r = softmax_naive(x)
    print(f"  naive: {r} (可能全 0 或全 NaN)")
except Exception as e:
    print(f"  naive: 错误 {e}")
r = softmax_safe(x)
res(f"""输入 x = [1000, 1001, 1002]:
  naive:  失败 (exp 溢出)
  safe:   {r.round(4).tolist()}  ← 正常, 0.090, 0.245, 0.665""")
mea("""所有 softmax / log-softmax 必先减 max:
  - 减 max: 防止 exp 溢出 (FP16 时尤其重要)
  - log_softmax: 减 max 后 exp 不会爆
  - 算子实现: 这是第一行必加的代码""")

# --- 4. 精度对齐方法 ---
hdr(4,TOTAL,"精度对齐:5 步法")
why("""自研算子与标杆对比的 5 步:""")
out = ["  步     动作                                    工具"]
out.append("  1. 准备测试数据                           随机 + 极端值")
out.append("  2. 跑标杆算子 (torch CPU/GPU)             torch 实现")
out.append("  3. 跑自研算子 (NPU)                       你的代码")
out.append("  4. 对比: max abs / mean abs / rel error   numpy")
out.append("  5. 差异大时: 逐层 dump, 定位出错点         npu dump")
res("\n".join(out))
mea("""精度阈值经验:
  算子                可接受误差 (max rel)
  matmul (小)         1e-3
  matmul (大)         1e-2
  elementwise         1e-5
  Softmax             1e-4
  LayerNorm           1e-4
  衰减/长序列        1e-2 (误差累积)
  
  超阈值 = 算子有 bug, 必须修""")
PYEOF

echo ""
echo "############################################################"
cat <<'EOF'
【整体总结】
- 小白:算子精度对齐 = 与标杆 (torch) 对比输出;5 大误差源:累加精度、Softmax
  溢出、截断、顺序、范围;FP16 累加 N=10000 误差 10%,FP32 累加 1e-4;
  Softmax 必先减 max。
- 熟手:累加器默认 FP32 是 Cube 标配;极端值要测 (0, inf, 极大极小);
  精度阈值: matmul 1e-3, softmax 1e-4, 长序列累积误差 1e-2;逐层 dump
  定位错误算子;msprof + 数值对比 + 单元测试三位一体。
【进阶】写一个自研 matmul, 加单元测试与 torch matmul 对比 max abs error;
  测试 (M,N,K) 多种尺寸 + 极端输入 (全 0, 大值, NaN)。
EOF
echo "############################################################"
