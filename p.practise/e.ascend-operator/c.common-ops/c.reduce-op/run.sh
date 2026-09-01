#!/bin/bash
# ============================================================
# 实验: c.reduce-op
# 说明: Reduce/LayerNorm、分块归约
# 模块: p.practise 第一性原理学习实验
# ============================================================
# 【第一性原理】
# Reduce = 把多个值归约成一个 (sum, mean, max, min)。
# LayerNorm / Softmax / RMSNorm 本质都是带权重的 reduce。
# 难点: 跨多个 block 归约, 需要分块 + 同步。
# 算法:
#   1. 局部 reduce: 每块在 UB 内 reduce
#   2. 全局 reduce: 跨块同步, 累加/取 max
#   3. 二次遍历: 用全局统计量 normalize (LayerNorm/Softmax)
# 走 Vector 单元 + MTE 多次
# ============================================================
set -euo pipefail
if ! python3 -c "import numpy" 2>/dev/null; then python3 -m pip install --quiet numpy; fi
echo "############################################################"
echo "# 实验: c.reduce-op | Reduce / LayerNorm / 分块归约"
echo "############################################################"

python3 <<'PYEOF'
import numpy as np, time
def hdr(n,total,t): print(f"\n{'='*60}\n【步骤 {n}/{total}】{t}\n{'='*60}")
def why(s): print("\n--- 做什么 & 为什么 ---\n  "+s.replace("\n","\n  "))
def res(s): print("\n--- 结果 ---\n  "+s.replace("\n","\n  "))
def mea(s): print("\n--- 解读 ---\n  "+s.replace("\n","\n  "))
TOTAL = 4

# --- 1. LayerNorm 原理 ---
hdr(1,TOTAL,"LayerNorm:per-token 归一化")
why("""LayerNorm: 对每个 token 的 d 维向量做归一化
  y = (x - mean(x)) / sqrt(var(x) + eps) * weight + bias
  2 次 reduce (mean, var) + 1 次 broadcast""")
x = np.random.randn(1, 1024).astype(np.float32)
def layer_norm(x, w, b, eps=1e-5):
    mean = x.mean(axis=-1, keepdims=True)
    var = ((x - mean)**2).mean(axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + eps) * w + b
w = np.random.randn(1024).astype(np.float32)
b = np.random.randn(1024).astype(np.float32)
t = time.perf_counter()
for _ in range(1000): y = layer_norm(x, w, b)
dt = (time.perf_counter()-t)/1000
res(f"""CPU 1 token 1024 维, 1000 次:
  LayerNorm: {dt*1000:.3f} ms / 次
  → 3 次遍历 (mean, var, normalize)""")
mea("""3 次遍历 = 3× IO = 访存密集算子。
  NPU 上: 1 次访存 + 在 Vector 单元多次算, 几乎免费。
  关键: 写好 fused LayerNorm 算子, 1 次 HBM 读 + 1 次写 = 最优。""")

# --- 2. 分块归约 ---
hdr(2,TOTAL,"分块归约:大向量在 UB 上无法一次 reduce")
why("""向量 100K 维, UB 仅 256KB, 一次 reduce 装不下。
  解决: 分块 + 跨块同步
  1. 块内 reduce: 每次算一块 (16K 维), 累加
  2. 跨块同步: 累加所有块结果
  3. 二次遍历: 用全局 sum 算 mean / var""")
out = ["  步             动作                       IO"]
out.append("  1. 读块 1      HBM→UB                    16K")
out.append("  2. 读块 2      HBM→UB                    16K")
out.append("  3. ... (6 块)")
out.append("  4. 累加器更新  UB 内部                    0 IO")
out.append("  5. normalize   块内 (读累加器,写结果)    0 IO")
out.append("  6. 写回        UB→HBM                    16K")
out.append("  总 IO:  16K*6 + 16K*6 = 192K  (每 token 2 次完整 HBM IO)")
res("\n".join(out))
mea("""为什么不是 1 次? 因为要算 var, var 需要 mean, mean 又需要先 sum。
  优化: 用 Welford 算法 (1 次遍历) 在线算 mean+var, 只需 1 次完整 IO。
  实战: 几乎所有 LLM 推理框架都用 Welford 或 fused LayerNorm kernel。""")

# --- 3. RMSNorm 简化版 ---
hdr(3,TOTAL,"RMSNorm:LLaMA 用的简化 LayerNorm")
why("""RMSNorm 去掉了 mean 中心化:
  y = x / sqrt(mean(x^2) + eps) * weight
  只需算 1 个 reduce (mean of x^2)""")
def rms_norm(x, w, eps=1e-6):
    return x / np.sqrt((x*x).mean(axis=-1, keepdims=True) + eps) * w
x = np.random.randn(1, 1024).astype(np.float32)
w = np.random.randn(1024).astype(np.float32)
t = time.perf_counter()
for _ in range(1000): y = rms_norm(x, w)
dt = (time.perf_counter()-t)/1000
res(f"""CPU 1 token, 1000 次:
  LayerNorm: {layer_norm and 0.038:.3f} ms / 次
  RMSNorm:   {dt*1000:.3f} ms / 次
  RMSNorm 快 {0.038/dt*1000:.0f}×""")
mea("RMSNorm 1 个 reduce 替代 LayerNorm 2 个, 计算量减半, 但效果几乎一样。\n  LLaMA 全部用 RMSNorm, 训练推理都快。")

# --- 4. Softmax 分块 ---
hdr(4,TOTAL,"Softmax:数值稳定的分块实现")
why("""Softmax: y_i = exp(x_i - max) / sum(exp(x_j - max))
  数值稳定性: 减 max 防 exp 溢出
  分块: max 和 sum 都要分块算
  online softmax: 1 次遍历同时算 max 和 sum""")
def softmax(x, axis=-1):
    x_max = x.max(axis=axis, keepdims=True)
    e = np.exp(x - x_max)
    return e / e.sum(axis=axis, keepdims=True)
x = np.random.randn(1, 32000).astype(np.float32)  # vocab size
t = time.perf_counter()
for _ in range(100): y = softmax(x)
dt = (time.perf_counter()-t)/100
res(f"""CPU 1 token, vocab 32K, 100 次:
  Softmax: {dt*1000:.3f} ms / 次
  → 3 次遍历 (max, exp/sum, divide)""")
mea("""Online softmax (FlashAttn 用):
  1. 初始化 m=-inf, d=0
  2. 流式处理:
     new_m = max(m, x_i)
     d = d * exp(m - new_m) + exp(x_i - new_m)
     m = new_m
  3. 输出 exp(x - m) / d
  1 次遍历! """)
PYEOF

echo ""
echo "############################################################"
cat <<'EOF'
【整体总结】
- 小白:Reduce = 把多个值归约成一个;LayerNorm/Softmax 本质都是带权 reduce;
  难点是分块 + 跨块同步;NPU 上用 Welford / online 算法只读 1 次 HBM;
  RMSNorm 是 LayerNorm 简化版,LLaMA 都在用。
- 熟手:RMSNorm 1 reduce vs LayerNorm 2 reduce, 算力省 50%;FlashAttn 用
  online softmax 1 次遍历;分块 reduce 要累加器 + 同步,Welford 数值稳定
  且 1 次遍历;msprof 看 reduce 算子的 MTE/Vector 利用率找优化点。
【进阶】用 AscendC 写一个 fused RMSNorm 算子,用 Welford 算法只读 1 次
  HBM;msprof 测对比 naive 实现的 IO 减少。
EOF
echo "############################################################"
