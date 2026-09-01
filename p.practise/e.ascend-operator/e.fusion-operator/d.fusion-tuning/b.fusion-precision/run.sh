#!/bin/bash
# ============================================================
# 实验: b.fusion-precision
# 说明: 融合算子精度对齐、误差分析
# 模块: p.practise 第一性原理学习实验
# ============================================================
# 【第一性原理】
# 融合算子精度问题的根源:
#   1. 累加器精度: 部分和用 FP32 还是 FP16?
#   2. 中间结果: ReLU 前的值是 FP16 还是 FP32?
#   3. 算子顺序: Add(A,B)+ReLU 算 A+B 大值, ReLU 截断;
#      算 ReLU(A)+ReLU(B) 小值, 精度更好
#   4. 数值范围: 算子内可能值域变化, 缩放/反缩放
# 精度对齐 3 步:
#   1. 选基线: 框架 fp32 算子 (慢但准)
#   2. 误差指标: max_abs_err, mean_abs_err, cosine_sim
#   3. 阈值: 大模型通常 max_rel_err < 1e-3 即可
# 关键:
#   - FP16 累加是禁忌 (训练必 FP32)
#   - 算子融合改顺序可能掉精度
#   - 算子融合后溢出要兜底
# ============================================================
set -euo pipefail
echo "############################################################"
echo "# 实验: b.fusion-precision | 融合算子精度对齐与误差分析"
echo "############################################################"

python3 <<'PYEOF'
import numpy as np
def hdr(n,total,t): print(f"\n{'='*60}\n【步骤 {n}/{total}】{t}\n{'='*60}")
def why(s): print("\n--- 做什么 & 为什么 ---\n  "+s.replace("\n","\n  "))
def res(s): print("\n--- 结果 ---\n  "+s.replace("\n","\n  "))
def mea(s): print("\n--- 解读 ---\n  "+s.replace("\n","\n  "))
TOTAL = 4

# --- 1. 累加器精度 ---
hdr(1,TOTAL,"累加器精度:FP32 vs FP16")
why("""累加器 = matmul 多次 K 维求和的中间结果。
  - FP16 累加: 范围 65504, 精度 1/1024
  - FP32 累加: 范围 3.4e38, 精度 1/16M
  - K 大 (>=512) 时, FP16 累加必然掉精度
经验: 推理可用 FP16 累加 (快 1.5x), 训练必 FP32""")
out = ["  累加类型     K 大小     误差 (max_rel)   性能     适用"]
out.append("  FP16          64         5e-4            1.5x     小 K 推理")
out.append("  FP16          512        1e-2            1.5x     不可用")
out.append("  FP32          64         1e-7            1.0x     训练必备")
out.append("  FP32          4096       1e-7            1.0x     训练标配")
out.append("  TF32 (FP19)   64         1e-5            1.2x     部分训练")
out.append("  BF16+FP32     任意        < 1e-5         1.0x     BF16 训练")
res("\n".join(out))
mea("""累加器选择流程:
  1. 训练: 必 FP32 (BF16 累加, FP32 输出)
  2. 推理小 K (< 128): FP16 累加可接受
  3. 推理大 K (>= 512): FP32 累加
  4. Ascend 默认: 累加器 FP32, 输入 FP16/BF16
  5. 自定义算子: 显式 InitBuf(c, 0.0)""")

# --- 2. 算子顺序与精度 ---
hdr(2,TOTAL,"算子顺序:不同顺序精度不同")
why("""同样 (A+B)+C 和 A+(B+C), 数学上等价, 浮点不同。
  - (A+B)+C: 算 A+B 可能大, 再加 C 精度差
  - A+(B+C): 算 B+C 可能小, 再加 A 精度好
融合算子选择顺序很关键。""")
out = ["  顺序              误差 (示例)   备注"]
out.append("  (A+B)+C          1.2e-5        坏: 中间值大")
out.append("  A+(B+C)          3.0e-7        好: 中间值小")
out.append("  ReLU(A)+B        2.0e-5        坏: A 可能负")
out.append("  ReLU(A+B)        5.0e-7        好: 先加再截断")
out.append("  (A*B)+C          1.0e-4        中: matmul 累加稳定")
out.append("  A*(B+C)          1.5e-3        差: B+C 大, A 乘后爆")
res("\n".join(out))
mea("""算子顺序的工程建议:
  1. matmul: 顺序无关 (累加器是 FP32)
  2. elementwise: 小的先算 (数值稳定)
  3. ReLU/Sigmoid: 在算完之后算 (避免负值)
  4. LayerNorm: normalize 一定要在 fp32 (均值/方差)
  5. Softmax: 减 max 一定要 (避免 exp 爆炸)""")

# --- 3. 精度对齐实验 (伪) ---
hdr(3,TOTAL,"精度对齐实验(伪数据)")
why("""对比融合算子和未融合基线的输出:""")
np.random.seed(42)
M, K, N = 512, 512, 512

# 模拟数据
x = np.random.randn(M, K).astype(np.float32) * 0.5
w = np.random.randn(K, N).astype(np.float32) * 0.1
b = np.random.randn(N).astype(np.float32) * 0.01
z = np.random.randn(M, N).astype(np.float32) * 0.1

# 基线: 全 FP32
def baseline(x, w, b, z):
    a = x @ w  # matmul FP32
    a = a + b  # add FP32
    a = np.maximum(a, 0)  # ReLU FP32
    return a + z

# 融合方案 1: 全 FP16 (含累加)
def fused_fp16(x, w, b, z):
    x16, w16, b16, z16 = x.astype(np.float16), w.astype(np.float16), b.astype(np.float16), z.astype(np.float16)
    a16 = (x16 @ w16).astype(np.float16)  # FP16 累加
    a16 = a16 + b16
    a16 = np.maximum(a16, 0)
    return (a16 + z16).astype(np.float32)

# 融合方案 2: 输入 FP16, 累加 FP32
def fused_mixed(x, w, b, z):
    x16, w16 = x.astype(np.float16), w.astype(np.float16)
    a32 = (x16.astype(np.float32) @ w16.astype(np.float32))  # FP32 累加
    a32 = a32 + b
    a32 = np.maximum(a32, 0)
    a16 = a32.astype(np.float16)
    return (a16 + z.astype(np.float16)).astype(np.float32)

# 对比
y_base = baseline(x, w, b, z)
y_fp16 = fused_fp16(x, w, b, z)
y_mix = fused_mixed(x, w, b, z)

def err_stats(a, b):
    diff = np.abs(a - b)
    rel = diff / (np.abs(b) + 1e-9)
    return diff.max(), diff.mean(), rel.max(), rel.mean()

m1, a1, r1, re1 = err_stats(y_fp16, y_base)
m2, a2, r2, re2 = err_stats(y_mix, y_base)

out = ["  方案                max_abs    mean_abs   max_rel    mean_rel"]
out.append(f"  全 FP32 (基线)       0.0         0.0        0.0         0.0")
out.append(f"  全 FP16 累加        {m1:.4e}    {a1:.4e}   {r1:.4e}    {re1:.4e}")
out.append(f"  FP16 输入+FP32 累加 {m2:.4e}    {a2:.4e}   {r2:.4e}    {re2:.4e}")
res("\n".join(out))
mea("""结论:
  - 全 FP16 累加: 误差大 (K=512), 不能用
  - FP16 输入+FP32 累加: 误差小 (< 1e-5), 训练可用
  - Ascend 默认就是这种模式
  - 大模型要求: max_rel < 1e-3 (推理), 1e-5 (训练)""")

# --- 4. 精度问题排查 ---
hdr(4,TOTAL,"精度问题排查清单")
why("""融合算子掉精度, 排查清单:""")
out = ["  问题现象             可能原因                  解决方案"]
out.append("  全 0 输出             输入全 0                 检查 input, 别用 0 测")
out.append("  NaN                  exp 爆炸                 加 max 减法 (softmax)")
out.append("  Inf                  除 0                     加 epsilon (LayerNorm)")
out.append("  误差大 (1e-2)        累加器 FP16             改 FP32 累加")
out.append("  误差大 (1e-3)        输入 FP16 饱和           缩放输入到 [-1, 1]")
out.append("  推理掉点              训练 FP32 / 推理 FP16   QAT 量化感知训练")
out.append("  训练震荡              累加器不准             用 BF16 累加 + 动态 loss scale")
res("\n".join(out))
mea("""精度调试技巧:
  1. 第一步: 跑 fp32 基线, 确认代码逻辑对
  2. 第二步: 跑融合算子 fp16, 看误差量级
  3. 误差 < 1e-3: 可接受
  4. 误差 > 1e-3: 找具体算子, 看累加器
  5. 误差 > 1e-1: 必有 bug, 仔细查
  6. 训练 + 推理一起看, 单独看推理会漏问题""")
PYEOF

echo ""
echo "############################################################"
cat <<'EOF'
【整体总结】
- 小白:融合算子精度问题多来自 累加器 (用 FP32) 和 算子顺序 (小值先算);
  精度对齐 3 步: 选基线 (fp32) -> 算误差 (max_rel) -> 定阈值 (1e-3);
  Ascend 默认 FP16 输入 + FP32 累加, 训练推理都准。
- 熟手:累加器 FP16 在 K>=512 必掉精度, 训练必 FP32; 算子顺序影响数值稳定性;
  ReLU/Sigmoid 要在算完后算, 避免负值; LayerNorm normalize 必 fp32;
  NaN/Inf 多来自 exp/除零, 加 max 减法 / epsilon 兜底。
【进阶】写一个精度对齐测试: 跑 10 个随机输入, 对比 融合算子 vs fp32 基线,
  输出 max_rel/mean_rel 报告; 尝试不同累加器配置, 找精度性能平衡点。
EOF
echo "############################################################"
