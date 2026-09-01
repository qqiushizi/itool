#!/bin/bash
# ============================================================
# 实验: b.activation-fusion
# 说明: 激活融合(Linear+GELU/SiLU/ReLU 等)
# 模块: p.practise 第一性原理学习实验
# ============================================================
# 【第一性原理】
# Linear + 激活函数 是 LLM 最常见的组合:
#   Linear:  Y = X @ W + b
#   ReLU:    Y = max(0, Y)
#   GELU:    Y = 0.5*Y*(1 + tanh(...))  (transformer)
#   SiLU:    Y = Y * sigmoid(Y)         (LLaMA)
#   SwiGLU:  Y = SiLU(gate) * up       (LLaMA FFN)
# 不融合:  Linear kernel 写 HBM + Activation kernel 读 HBM
# 融合:    Linear + Activation 1 个 kernel, 中间结果留 UB
# 收益: 1.3-1.5×, 主要省 IO
# ============================================================
set -euo pipefail
if ! python3 -c "import numpy" 2>/dev/null; then python3 -m pip install --quiet numpy; fi
echo "############################################################"
echo "# 实验: b.activation-fusion | Linear+激活融合:ReLU/GELU/SiLU"
echo "############################################################"

python3 <<'PYEOF'
import numpy as np
def hdr(n,total,t): print(f"\n{'='*60}\n【步骤 {n}/{total}】{t}\n{'='*60}")
def why(s): print("\n--- 做什么 & 为什么 ---\n  "+s.replace("\n","\n  "))
def res(s): print("\n--- 结果 ---\n  "+s.replace("\n","\n  "))
def mea(s): print("\n--- 解读 ---\n  "+s.replace("\n","\n  "))
TOTAL = 4

# --- 1. 各激活函数 ---
hdr(1,TOTAL,"5 种常见激活函数")
why("""不同激活函数, 算力不同, 融合收益略有不同:""")
def relu(x): return np.maximum(x, 0)
def gelu(x): return 0.5 * x * (1 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715*x**3)))
def silu(x): return x / (1 + np.exp(-x))
def swiglu(g, u): return silu(g) * u
out = ["  激活     公式                       算力     用途"]
out.append("  ReLU     max(0, x)                  1 op     CNN, 简单")
out.append("  GELU     0.5x(1+tanh(...))         ~10 op   Transformer")
out.append("  SiLU     x * sigmoid(x)             5 op    LLaMA")
out.append("  SwiGLU   SiLU(g) * u                7 op    LLaMA FFN")
out.append("  GeLU     0.5x(1+erf(x/sqrt(2)))    ~10 op   严格定义")
res("\n".join(out))
mea("ReLU 最简单 (1 op), GELU 最复杂 (10 op 含 tanh)。\n  复杂激活函数 (SwiGLU) 单独跑很慢, 必融。")

# --- 2. Linear + Bias + ReLU 融合示意 ---
hdr(2,TOTAL,"Linear + Bias + ReLU 融合示意")
why("""标准实现 = 2 个 kernel:
  kernel 1: Linear + Bias → 写 out (中间结果)
  kernel 2: ReLU → 读 out, 写 out_final
融合: 1 个 kernel, 中间结果在 UB
  Linear 计算时直接 + bias + ReLU, 写最终结果
  节省: 1 次 HBM write + 1 次 HBM read""")
N = 4096
io_unfused = (N*N*2) * 3  # 2 reads + 1 write (per token)
io_fused = (N*N*2) * 2  # 2 reads + 1 write final
res(f"""Linear (4096x4096) + Bias + ReLU:
  不融合: 读 X (32 MB) + 读 W (32 MB) + 写 out (32 MB) = 96 MB
            读 out (32 MB) + 写 final (32 MB) = 64 MB
            总 160 MB
  融合:  读 X (32 MB) + 读 W (32 MB) + 写 final (32 MB) = 96 MB
  节省: 64 MB (40%)
  
  HBM 2.7 TB/s, 时间节省: 64 / 2700 = 24 us""")
mea("LLaMA 32 层 × 64 维 = 1920 个 Linear。\n  每个省 24 us × 32 = 768 us。LLM 推理每个 step 省 1 ms 级别。")

# --- 3. SwiGLU 融合 ---
hdr(3,TOTAL,"SwiGLU 融合:LLaMA FFN 的灵魂")
why("""LLaMA FFN:
  gate = x @ W_gate
  up   = x @ W_up
  out  = SwiGLU(gate, up) @ W_down
       = (SiLU(gate) * up) @ W_down

SwiGLU = SiLU(gate) * up, 3 个算子 (SiLU, mul, ...)
融合成 1 个 kernel:SiLU(gate) * up 一次完成
收益: 1.3-1.5×, 显存省 50%""")
res("""LLaMA FFN 算子:
  1. gate = Linear(x)       ← Cube
  2. up   = Linear(x)       ← Cube
  3. SwiGLU = SiLU(gate) * up   ← 1 个 fused kernel (epilogue)
  4. out  = Linear(SwiGLU)  ← Cube (融合 3 的输出)
  
  Fused 算子: Linear(GEMM) + SiLU + Mul (SwiGLU epilogue)
  加速: 1.3-1.5×, 显存省 33%""")
mea("SwiGLU 融合是 LLaMA 性能关键:\n  - 1 个 Linear 写 HBM → Linear+SiLU+Mul 写 HBM, 省 1 次\n  - 显存省 33% (gate 不存)\n  vLLM/MindSpeed 都有 SwiGLU fused 实现。")

# --- 4. 实战:框架默认 fused 激活 ---
hdr(4,TOTAL,"实战:用框架默认 fused 激活")
why("""主流框架的激活融合支持:""")
out = ["  框架              激活 fusion          性能"]
out.append("  PyTorch          torch.nn.functional  自动 (epilogue)")
out.append("  vLLM            --fused-activation  1.3-1.5x")
out.append("  FasterTransformer 内置                  1.5x")
out.append("  SGLang          整层 fused           1.4x")
out.append("  MindSpeed-LLM   --use-fused-swiglu   1.3-1.5x")
out.append("  CUDA + cutlass   Linear + epilogue    1.5x (手写)")
res("\n".join(out))
mea("""实战:
  - 90% 用户: 框架默认 (vLLM/SGLang/MindSpeed)
  - 极致性能: CUTLASS/AscendC 写 epilogue fused kernel
  - 常见组合: Linear + Bias + GELU / SwiGLU / ReLU
  - 检查 profile: 激活 kernel 占比 < 5% 算正常, 大则要开 fused""")
PYEOF

echo ""
echo "############################################################"
cat <<'EOF'
【整体总结】
- 小白:激活融合 = Linear + ReLU/GELU/SiLU 合成 1 个 kernel,加速 1.3-1.5×,
  显存省 30-40%;SwiGLU 是 LLaMA 性能关键,必须融合;框架默认开。
- 熟手:SwiGLU 融合省 33% 显存;Linear+SiLU+Mul epilogue 是常见 pattern;
  CUTLASS / AscendC epilogue 写极致;msprof 看激活 kernel 占比,
  >5% 考虑 fused。
【进阶】用 AscendC 写一个 Linear+Bias+ReLU 算子, 对比 naive 3 kernel 的
  耗时和 IO。
EOF
echo "############################################################"
